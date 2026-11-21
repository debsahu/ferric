//! Continuous through-Proline `13C`-direct-detected IDP backbone resonance assignment solver,
//! temperature- & pH-aware `ncIDP` neighbor-corrected random-coil chemical-shift library,
//! unlabelled permutation-invariant reference offset estimator, multi-island evolutionary +
//! detailed-balance simulated annealing, and calibrated `ProteinReliabilityCurve` QC (`Phase 11`, `T11.2`).
//!
//! Provenance: Clean-room implementation derived strictly from peer-reviewed biomolecular NMR literature:
//! - Bermel, W., Bertini, I., Felli, I. C., Piccioli, M. & Pierattelli, R. (*Prog. Nucl. Magn.
//!   Reson. Spectrosc.* 2006, 48, 25–45; *J. Am. Chem. Soc.* 2006, 128, 3918–3919; *J. Am. Chem.
//!   Soc.* 2009, 131, 15339–15345 — protonless `2D CON`, `2D CACO`, `3D CANCO`, `CBCACON`,
//!   `CCC(CO)N`, and `NCACO` continuous through-Proline IDP backbone assignment).
//! - Tamiola, K., Acar, B. & Mulder, F. A. A. (*J. Am. Chem. Soc.* 2010, 132, 18000–18003 — `ncIDP`
//!   sequence-neighbor random-coil chemical shift library for intrinsically disordered proteins).
//! - Kjaergaard, M., Brander, S. & Poulsen, F. M. (*J. Biomol. NMR* 2011, 49, 139–149) and
//!   Kjaergaard, M. & Poulsen, F. M. (*Prog. Nucl. Magn. Reson. Spectrosc.* 2011, 59, 43–57 —
//!   temperature and sequence effects on random-coil chemical shifts in disordered proteins).
//! - Platzer, G., Okon, M. & McIntosh, L. P. (*J. Biomol. NMR* 2014, 60, 109–129 — pH-dependent
//!   random-coil `1H`, `13C`, and `15N` chemical shifts of ionizable residues).
//! - Schwarzinger, S. et al. (*J. Am. Chem. Soc.* 2001, 123, 2970–2978) and Wishart, D. S. et al.
//!   (*J. Biomol. NMR* 1995, 5, 67–81 — pre-Proline and sidechain aliphatic carbon shifts).

use std::collections::BTreeMap;
use std::f64::consts::TAU;
use std::thread;

use serde::{Deserialize, Serialize};

use crate::c13_detected::{
    cas_llr_adjustment, match_cas_peaks_to_roots, C13DetectedExp, C13PickedDataset, CacoPeakKind,
    CasMode, CasSpectrum, Con2DPeak, CAS_LOG_LIKELIHOOD_NATS, CAS_MATCH_TOL_C_PRIME_PPM,
    CAS_MATCH_TOL_N15_PPM,
};
use crate::nmrstar::Reference;
use crate::peaks::Peak;
use crate::picker::estimate_noise_mad;
use crate::priors::one_to_three;
use crate::qc::{DatasetQcVerdict, ProteinReliabilityCurve};
use crate::rng::Rng;
use crate::solver::{
    AssignResult, Matrix2D, AUTO_REF_C13_SPREAD_WARN_PPM, FORBID, MAX_AUTO_REF_C13_PPM,
    MAX_AUTO_REF_N15_PPM,
};
use crate::spectrum::Spectrum3D;
use crate::spinsystems::{normalize_hncacb_and_cbcaconh_polarity, HncacbPolarity, PolarityReport};

/// Half-forbidden sentinel (`-5e8`) matching WebGPU `idp_scoring.wgsl`.
pub const IDP_FORBID_HALF: f32 = -5.0e8;

/// Maximum per-channel candidate shifts packed per `ConDipeptideSystem` for CPU/WebGPU scoring.
pub const MAX_IDP_CANDS: usize = 6;

/// Maximum deduplicated `13CB(i-1)` candidates retained per spin system and in global `dCB` fitting.
pub const MAX_IDP_CB_CANDS: usize = 4;

/// Maximum sidechain `(i-1)` aliphatic `13C` TOCSY peaks (`CCC(CO)N`) packed per system.
pub const MAX_IDP_SC_CANDS: usize = 6;

/// Hard upper bound on `S * K` (and `S * S`) IDP scoring matrix cells (`25,000,000` cells = `200 MB f64`)
/// preventing unbounded host/GPU allocation on pathological peak lists or sequences.
pub const MAX_IDP_MATRIX_CELLS: usize = 25_000_000;

/// Maximum inter-experiment carrier-drift search half-width (`±0.70 ppm`) in `register_and_build_con_dipeptide_systems`.
pub const SOLVER_MAX_INTER_EXP_DRIFT_PPM: f64 = 0.70;

/// Boundary warning threshold (`|drift| >= 0.65 ppm`) indicating solver inter-experiment drift hit its `±0.70 ppm` clamp.
pub const SOLVER_WARN_INTER_EXP_DRIFT_PPM: f64 = 0.65;

/// Validate that `S * K` and `S * S` matrix dimensions stay within `MAX_IDP_MATRIX_CELLS`.
pub fn check_idp_matrix_bounds(s_len: usize, k_len: usize) -> Result<usize, String> {
    let sk = s_len
        .checked_mul(k_len)
        .ok_or_else(|| format!("IDP matrix S*K overflowed usize (S={s_len}, K={k_len})"))?;
    if sk > MAX_IDP_MATRIX_CELLS {
        return Err(format!(
            "IDP unary matrix S*K = {sk} (S={s_len}, K={k_len}) exceeds cap {MAX_IDP_MATRIX_CELLS}"
        ));
    }
    let ss = s_len
        .checked_mul(s_len)
        .ok_or_else(|| format!("IDP pair matrix S*S overflowed usize (S={s_len})"))?;
    if ss > MAX_IDP_MATRIX_CELLS {
        return Err(format!(
            "IDP pair matrix S*S = {ss} (S={s_len}) exceeds cap {MAX_IDP_MATRIX_CELLS}"
        ));
    }
    Ok(sk)
}

/// Peptide-bond dipeptide spin system anchored on `2D CON(i)` (`13C'(i-1)` – `15N(i)`),
/// active across ALL residue boundaries `k = 1..L-1` including `-Xaa-Pro-` and `-Pro-Pro-`!
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConDipeptideSystem {
    /// 0-based spin-system identifier (`0..S`).
    pub id: usize,
    /// Observed `15N(i)` chemical shift in ppm (`~105–131 ppm` amide, `~132–142 ppm` `Pro` imino).
    pub n_curr: f64,
    /// Observed `13C'(i-1)` chemical shift in ppm (`~168–181 ppm`, direct IPAP).
    pub c_prev: f64,
    /// Relative `2D CON` root intensity.
    pub root_height: f64,
    /// Candidate intraresidue `13CA(i)` shifts in ppm (`CANCO` / `NCACO` / `HNCO->HNCACB`).
    pub ca_curr: Vec<f64>,
    /// Candidate intraresidue `13CB(i)` shifts in ppm (`HNCO->HNCACB` when `1H` amide datasets supplement `CON`; `RD-10`).
    #[serde(default)]
    pub cb_curr: Vec<f64>,
    /// Candidate preceding `13CA(i-1)` shifts in ppm (`CBCACON` / `CANCO` / `CCC(CO)N` / `CACO`).
    pub ca_prev: Vec<f64>,
    /// Candidate preceding `13CB(i-1)` shifts in ppm (`CBCACON` / `CCC(CO)N`, up to `MAX_IDP_CB_CANDS = 4`).
    pub cb_prev: Vec<f64>,
    /// Candidate intraresidue `13C'(i)` shifts in ppm (`NCACO` / `CACO`), linking into `CON(i+1).c_prev`.
    pub c_curr: Vec<f64>,
    /// Additional preceding `(i-1)` aliphatic sidechain carbons (`CG*, CD*, CE*` from `CCC(CO)N`).
    pub sidechain_c_prev: Vec<f64>,
    /// Partner preceding bond `15N(i-1)` candidates `(n_ppm, snr)` from `(H)N(CA)CON` (`RD-13`).
    #[serde(default)]
    pub nn_prev_cands: Vec<(f64, f64)>,
    /// Unlabelled constant-time `2D CACO` opposite-phase `Gly` indicator at `c_prev`.
    pub prev_gly_opposite_phase: bool,
    /// True when `CBCACON` strip evidence was observed/acquired for this dataset.
    pub has_cbcacon_evidence: bool,
    /// Inter-experiment carrier-drift boundary warnings detected during unlabelled assembly.
    #[serde(default)]
    pub registration_warnings: Vec<String>,
}

impl ConDipeptideSystem {
    pub fn new(id: usize, c_prev: f64, n_curr: f64, root_height: f64) -> Self {
        Self {
            id,
            n_curr,
            c_prev,
            root_height,
            ca_curr: Vec::new(),
            cb_curr: Vec::new(),
            ca_prev: Vec::new(),
            cb_prev: Vec::new(),
            c_curr: Vec::new(),
            sidechain_c_prev: Vec::new(),
            nn_prev_cands: Vec::new(),
            prev_gly_opposite_phase: false,
            has_cbcacon_evidence: false,
            registration_warnings: Vec::new(),
        }
    }

    #[inline]
    pub fn has_3d_support(&self) -> bool {
        !self.ca_curr.is_empty()
            || !self.cb_curr.is_empty()
            || !self.ca_prev.is_empty()
            || !self.cb_prev.is_empty()
            || !self.c_curr.is_empty()
            || !self.sidechain_c_prev.is_empty()
            || !self.nn_prev_cands.is_empty()
    }
}

/// Global unlabelled chemical-shift referencing offsets `(dN, dC', dCA, dCB)` in ppm
/// (`observed ≈ predicted_ncidp + offset`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct IdpShiftOffsets {
    pub d_n: f64,
    pub d_c: f64,
    pub d_ca: f64,
    pub d_cb: f64,
}

/// Predicted sequence-specific random-coil shifts and coil dispersion widths for one residue `k`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NcIdpResiduePrediction {
    pub residue: String,
    pub n_ppm: f64,
    pub sigma_n: f64,
    pub c_ppm: f64,
    pub sigma_c: f64,
    pub ca_ppm: f64,
    pub sigma_ca: f64,
    pub cb_ppm: Option<f64>,
    pub sigma_cb: f64,
    /// Expected non-`CA`/`CB` aliphatic sidechain carbons (`CG*, CD*, CE*`) in `[10.0, 56.0] ppm`.
    pub sidechain_c: Vec<f64>,
}

/// Solver & scoring configuration for `13C`-detected IDP backbone assignment (`CPU` & `WebGPU`).
#[derive(Clone, Debug)]
pub struct IdpSolverConfig {
    /// Sample temperature in Kelvin (`default = 298.15`).
    pub temperature_k: f64,
    /// Sample pH (`default = 6.5`).
    pub ph: f64,
    /// Automatically estimate and apply unlabelled `(dN, dC', dCA, dCB)` reference offsets (`default = true`).
    pub auto_reference_offset: bool,
    /// Optional explicit offsets applied when `auto_reference_offset == false`.
    pub manual_offsets: IdpShiftOffsets,
    /// Core `ncIDP` random-coil standard deviations `(ppm)`.
    pub sigma_ca_rc: f64,
    pub sigma_cb_rc: f64,
    pub sigma_c_rc: f64,
    pub sigma_n_rc: f64,
    pub sigma_pro_n_rc: f64,
    /// Sequential inter-system matching tolerances `(ppm)` in `P(s, t)`.
    pub sigma_seq_ca: f64,
    pub sigma_seq_c: f64,
    /// Candidate-multiplicity ambiguity penalty weight `w_mult` (`-w_mult * ln(n_compatible)` in
    /// sequential `P(s, t)` on both CPU and WebGPU, `default = 0.35`).
    pub w_mult: f64,
    /// Optional upstream 3D picker carrier-drift clamp warnings propagated into `IdpAssignResult` QC.
    pub picker_warnings: Vec<String>,
    /// Per-atom unary LLR floor (`default = -5.5`).
    pub llr_floor: f64,
    /// Sequential pair score floor (`default = -3.8`).
    pub pair_floor: f64,
    /// Number of trajectories (`replicas`, `default = 48`).
    pub replicas: usize,
    /// Number of evolutionary islands on CPU (`default = 4`).
    pub islands: usize,
    /// Number of evolutionary crossover generations on CPU (`default = 3`).
    pub generations: usize,
    /// MCMC steps per spin system (`default = 2_400`).
    pub steps_per_system: usize,
    pub t_start: f64,
    pub t_end: f64,
    /// High-confidence threshold for QC / `ProteinReliabilityCurve` (`default = 0.80`).
    pub hc_threshold: f64,
    pub seed: u64,
    /// Optional amino-acid-selective `(C'(i-1), N(i))` 2D subspectra (`RD-4` / `RD-10`).
    pub cas_spectra: Vec<CasSpectrum>,
    /// Elliptical `13C'` match tolerance (ppm) for `CAS` peaks (`default = 0.06`).
    pub cas_c_tol_ppm: f64,
    /// Elliptical `15N` match tolerance (ppm) for `CAS` peaks (`default = 0.20`).
    pub cas_n_tol_ppm: f64,
    /// Log-likelihood magnitude (nats) for a `CAS` residue-type hit (`default = 3.0`).
    pub cas_match_llr: f64,
    /// Optional reciprocal `HN(CA)CONH` `N-N` adjacency pairs `(N_a, N_b)` in ppm (`RD-13`).
    pub reciprocal_nn_pairs: Vec<(f64, f64)>,
}

impl Default for IdpSolverConfig {
    fn default() -> Self {
        Self {
            temperature_k: 298.15,
            ph: 6.5,
            auto_reference_offset: true,
            manual_offsets: IdpShiftOffsets::default(),
            sigma_ca_rc: 0.45,
            sigma_cb_rc: 0.55,
            sigma_c_rc: 0.40,
            sigma_n_rc: 0.85,
            sigma_pro_n_rc: 1.15,
            sigma_seq_ca: 0.085,
            sigma_seq_c: 0.055,
            w_mult: 0.35,
            picker_warnings: Vec::new(),
            llr_floor: -5.5,
            pair_floor: -3.8,
            replicas: 48,
            islands: 4,
            generations: 3,
            steps_per_system: 2_400,
            t_start: 16.0,
            t_end: 0.08,
            hc_threshold: 0.80,
            seed: 42,
            cas_spectra: Vec::new(),
            cas_c_tol_ppm: CAS_MATCH_TOL_C_PRIME_PPM,
            cas_n_tol_ppm: CAS_MATCH_TOL_N15_PPM,
            cas_match_llr: CAS_LOG_LIKELIHOOD_NATS,
            reciprocal_nn_pairs: Vec::new(),
        }
    }
}

/// Complete IDP `13C`-detected assignment result, unlabelled QC verdict, and fitted offsets.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IdpAssignResult {
    /// Assigned sequence index `k in 1..L` for each `ConDipeptideSystem` (`-1` if unassigned).
    pub position: Vec<i32>,
    /// Monotone calibrated posterior confidence in `[0.0, 1.0]` for each `ConDipeptideSystem`.
    pub confidence: Vec<f64>,
    /// Pre-calibration combined Boltzmann specificity + replica consensus score in `[0.0, 1.0]`
    /// before monotone `IDP_CALIBRATION_KNOTS` mapping and post-calibration QC `Warn`/`Abstain` damping.
    #[serde(default)]
    pub raw_posterior: Vec<f64>,
    pub score: f64,
    pub run_scores: Vec<f64>,
    pub fitted_offsets: IdpShiftOffsets,
    pub qc_verdict: DatasetQcVerdict,
    pub support_3d_frac: f64,
    pub connectivity_frac: f64,
    pub qc_reasons: Vec<String>,
}

impl IdpAssignResult {
    /// Convert to standard `AssignResult` for interoperability with `evaluate::score` and `ProteinReliabilityCurve`.
    pub fn as_assign_result(&self) -> AssignResult {
        AssignResult {
            position: self.position.clone(),
            confidence: self.confidence.clone(),
            score: self.score,
            run_scores: self.run_scores.clone(),
        }
    }

    /// Evaluate calibrated `ProteinReliabilityCurve` against spin-system ground-truth indices `truth`.
    pub fn reliability_curve(&self, truth: &[i32], hc_threshold: f64) -> ProteinReliabilityCurve {
        ProteinReliabilityCurve::from_assign_result(&self.as_assign_result(), truth, hc_threshold)
    }
}

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

// ============================================================================
// Temperature- & pH-Aware `ncIDP` Neighbor-Corrected Random-Coil Model
// ============================================================================

/// Baseline `ncIDP` random-coil `(N, C', CA, CB)` shifts in ppm (DSS scale, `298.15 K, pH 6.5`)
/// per Tamiola, Acar & Mulder (*J. Am. Chem. Soc.* 2010, 132, 18000–18003, Table 1) and
/// Kjaergaard & Poulsen (*Prog. Nucl. Magn. Reson. Spectrosc.* 2011, 59, 43–57).
fn ncidp_base_shifts(aa: &str) -> (f64, f64, f64, Option<f64>) {
    match aa {
        "ALA" => (124.5, 177.8, 52.5, Some(19.1)),
        "ARG" => (121.3, 176.3, 56.0, Some(30.8)),
        "ASN" => (119.0, 175.2, 53.1, Some(38.9)),
        "ASP" => (120.7, 176.4, 54.2, Some(41.1)),
        "CYS" => (119.8, 174.6, 58.2, Some(28.0)),
        "GLN" => (120.3, 176.0, 55.8, Some(29.5)),
        "GLU" => (121.2, 176.6, 56.6, Some(30.2)),
        "GLY" => (109.7, 173.9, 45.2, None),
        "HIS" => (119.6, 175.2, 55.8, Some(29.8)),
        "ILE" => (121.2, 176.4, 61.1, Some(38.6)),
        "LEU" => (122.6, 177.4, 55.1, Some(42.3)),
        "LYS" => (121.6, 176.6, 56.3, Some(33.0)),
        "MET" => (120.6, 176.2, 55.4, Some(32.8)),
        "PHE" => (120.9, 175.8, 57.8, Some(39.6)),
        "PRO" => (136.8, 176.9, 63.2, Some(32.1)),
        "SER" => (116.5, 174.6, 58.3, Some(63.8)),
        "THR" => (115.0, 174.6, 61.8, Some(69.8)),
        "TRP" => (121.7, 176.3, 57.4, Some(29.6)),
        "TYR" => (120.8, 175.7, 57.9, Some(38.8)),
        "VAL" => (120.6, 176.3, 62.2, Some(32.8)),
        _ => (120.5, 176.0, 56.0, Some(33.0)),
    }
}

/// Preceding-neighbor `(i-1)` random-coil corrections `(d_n, d_c, d_ca, d_cb)` in ppm exerted by
/// `prev_aa` ($X_{i-1}$) onto residue $X_i$ (Tamiola, Acar & Mulder *JACS* 2010, Table 2;
/// Braun, Wider & Wüthrich *JACS* 1994), zero-centered across standard amino acids relative to
/// `ncidp_base_shifts` so global `(dN, dC', dCA, dCB)` referencing remains strictly unbiased.
fn ncidp_prev_neighbor_delta(prev_aa: &str, curr_aa: &str) -> (f64, f64, f64, f64) {
    let (raw_dn, raw_dc, raw_dca, dcb) = match prev_aa {
        "GLY" => (-0.95, 0.08, 0.12, 0.00),
        "ALA" => (-0.55, 0.05, 0.04, 0.00),
        "SER" => (0.20, -0.08, -0.05, 0.02),
        "THR" => (1.55, -0.18, -0.14, 0.04),
        "VAL" => (2.55, -0.25, -0.22, 0.05),
        "ILE" => (2.85, -0.26, -0.24, 0.05),
        "LEU" => (0.68, -0.06, -0.06, 0.00),
        "PRO" => (1.10, -0.22, -0.18, -0.05),
        "PHE" => (1.42, -0.18, -0.16, -0.06),
        "TYR" => (1.50, -0.18, -0.16, -0.06),
        "TRP" => (1.30, -0.16, -0.18, -0.08),
        "HIS" => (1.05, -0.14, -0.12, -0.03),
        "ASP" => (0.32, -0.05, -0.03, 0.02),
        "GLU" => (0.52, -0.04, -0.04, 0.02),
        "ASN" => (0.42, -0.08, -0.06, 0.00),
        "GLN" => (0.62, -0.06, -0.05, 0.00),
        "LYS" => (0.60, -0.06, -0.05, 0.00),
        "ARG" => (0.62, -0.08, -0.06, 0.00),
        "MET" => (0.60, -0.06, -0.05, 0.00),
        "CYS" => (0.80, -0.12, -0.08, 0.00),
        _ => (0.855, -0.104, -0.083, 0.0),
    };
    let mut dn = raw_dn - 0.855;
    let dc = raw_dc + 0.104;
    let dca = raw_dca + 0.083;
    // Specific `Xaa-Pro` tertiary-imide `15N_Pro` sensitivity (Bermel et al. *JACS* 2009; Tamiola et al. 2010)
    if curr_aa == "PRO" {
        dn = match prev_aa {
            "GLY" => -2.35,
            "ALA" => -0.90,
            "PRO" => -1.25,
            "SER" => -0.55,
            "THR" => 1.05,
            "VAL" => 1.85,
            "ILE" => 2.05,
            "PHE" | "TYR" | "TRP" => 0.80,
            _ => 0.75 * dn,
        };
    }
    (dn, dc, dca, dcb)
}

/// Following-neighbor `(i+1)` random-coil corrections `(d_n, d_c, d_ca, d_cb)` in ppm exerted by
/// `next_aa` ($X_{i+1}$) onto residue $X_i$, including strong steric/electronic pre-Proline shifts
/// when `next_aa == "PRO"` (Tamiola et al. *JACS* 2010; Schwarzinger et al. *JACS* 2001).
fn ncidp_next_neighbor_delta(curr_aa: &str, next_aa: &str) -> (f64, f64, f64, f64) {
    if next_aa == "PRO" {
        return match curr_aa {
            "GLY" => (1.45, -1.50, -1.05, 0.00),
            "PRO" => (0.90, -1.80, -1.80, 0.12),
            _ => (1.65, -1.95, -1.95, -0.35),
        };
    }
    match next_aa {
        "GLY" => (0.42, -0.18, 0.14, 0.00),
        "PHE" | "TYR" | "TRP" => (-0.25, -0.15, -0.14, -0.06),
        "ILE" | "VAL" | "THR" => (-0.11, -0.07, -0.04, 0.04),
        "ALA" => (-0.03, 0.13, 0.08, 0.00),
        _ => (0.05, 0.06, 0.03, 0.00),
    }
}

/// Henderson–Hasselbalch pH titration shifts `(d_n, d_c, d_ca, d_cb)` relative to `pH 6.5`
/// (Platzer, Okon & McIntosh *J. Biomol. NMR* 2014, 60, 109–129).
fn ncidp_ph_delta(aa: &str, ph: f64, is_n_term: bool, is_c_term: bool) -> (f64, f64, f64, f64) {
    let ph_c = ph.clamp(1.0, 11.0);
    let frac_prot = |pka: f64, p: f64| -> f64 { 1.0 / (1.0 + 10.0_f64.powf(p - pka)) };

    let (mut dn, mut dc, mut dca, mut dcb) = match aa {
        "ASP" => {
            let df = frac_prot(3.86, ph_c) - frac_prot(3.86, 6.5);
            (-0.70 * df, -1.80 * df, -1.90 * df, -2.90 * df)
        }
        "GLU" => {
            let df = frac_prot(4.34, ph_c) - frac_prot(4.34, 6.5);
            (-0.60 * df, -1.30 * df, -1.80 * df, -1.80 * df)
        }
        "HIS" => {
            let df = frac_prot(6.45, ph_c) - frac_prot(6.45, 6.5);
            (-0.80 * df, -1.20 * df, -df, -0.90 * df)
        }
        _ => (0.0, 0.0, 0.0, 0.0),
    };
    if is_c_term {
        // C-terminal carboxylate (+2.1 ppm C' at neutral pH vs internal; protonates below pH 3.6)
        let f_deprot = 1.0 - frac_prot(3.55, ph_c);
        dc += 2.10 * f_deprot - 0.80 * (1.0 - f_deprot);
        dca += 0.65 * f_deprot - 0.55 * (1.0 - f_deprot);
        dn += 1.20 * f_deprot;
    }
    if is_n_term {
        let f_prot = frac_prot(7.75, ph_c);
        dca -= 1.35 * f_prot;
        dc -= 1.65 * f_prot;
        dcb += 0.35 * f_prot;
    }
    (dn, dc, dca, dcb)
}

/// Linear temperature coefficients `(d_n, d_c, d_ca, d_cb)` in `ppm` for `(temperature_k - 298.15)`
/// (Kjaergaard, Brander & Poulsen *J. Biomol. NMR* 2011, 49, 139–149).
fn ncidp_temp_delta(aa: &str, temperature_k: f64) -> (f64, f64, f64, f64) {
    let dt = (temperature_k.clamp(270.0, 345.0)) - 298.15;
    if dt.abs() < 1e-6 {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let slope_n = if aa == "PRO" { -0.008 } else { -0.020 };
    let slope_c = -0.0055;
    let slope_ca = match aa {
        "VAL" | "ILE" | "PRO" => 0.0020,
        "ALA" => -0.0030,
        _ => -0.0015,
    };
    let slope_cb = 0.0020;
    (dt * slope_n, dt * slope_c, dt * slope_ca, dt * slope_cb)
}

/// Expected non-`CA`/`CB` aliphatic sidechain `13C` random-coil shifts (`CG*, CD*, CE*` in ppm)
/// observable in `3D CCC(CO)N` (Wishart et al. *J. Biomol. NMR* 1995, 5, 67–81).
pub fn expected_aliphatic_sidechain_c(aa: &str) -> &'static [f64] {
    match aa {
        "PRO" => &[27.3, 50.4],
        "ILE" => &[13.0, 17.6, 27.3],
        "VAL" => &[20.9, 21.5],
        "LEU" => &[23.8, 24.9, 26.9],
        "THR" => &[21.5],
        "LYS" => &[24.8, 29.1, 42.0],
        "ARG" => &[27.2, 43.3],
        "MET" => &[17.0, 32.1],
        "GLU" => &[36.1],
        "GLN" => &[33.8],
        _ => &[],
    }
}

/// Predict temperature-, pH-, and sequence-neighbor-corrected `ncIDP` random-coil shifts across
/// `sequence[0..L]` (including `Pro` `15N` ~ `136.8 ppm` and pre-Proline corrections on `k - 1`).
pub fn predict_ncidp_shifts(
    sequence: &[String],
    cfg: &IdpSolverConfig,
) -> Vec<NcIdpResiduePrediction> {
    let seq = normalize_seq_3(sequence);
    let len = seq.len();
    let mut out = Vec::with_capacity(len);

    for k in 0..len {
        let aa = seq[k].as_str();
        let (mut n_ppm, mut c_ppm, mut ca_ppm, mut cb_opt) = ncidp_base_shifts(aa);

        if k > 0 {
            let (dn, dc, dca, dcb) = ncidp_prev_neighbor_delta(&seq[k - 1], aa);
            n_ppm += dn;
            c_ppm += dc;
            ca_ppm += dca;
            if let Some(cb) = &mut cb_opt {
                *cb += dcb;
            }
            if k == 1 {
                // Inductive +NH3 effect of protonated N-terminal residue `0` on `15N(1)`
                let f_nt = 1.0 / (1.0 + 10.0_f64.powf(cfg.ph.clamp(1.0, 11.0) - 7.75));
                n_ppm += 2.20 * f_nt;
            }
        }
        if k + 1 < len {
            let (dn, dc, dca, dcb) = ncidp_next_neighbor_delta(aa, &seq[k + 1]);
            n_ppm += dn;
            c_ppm += dc;
            ca_ppm += dca;
            if let Some(cb) = &mut cb_opt {
                *cb += dcb;
            }
        }

        let (dn_ph, dc_ph, dca_ph, dcb_ph) = ncidp_ph_delta(aa, cfg.ph, k == 0, k + 1 == len);
        n_ppm += dn_ph;
        c_ppm += dc_ph;
        ca_ppm += dca_ph;
        if let Some(cb) = &mut cb_opt {
            *cb += dcb_ph;
        }

        let (dn_t, dc_t, dca_t, dcb_t) = ncidp_temp_delta(aa, cfg.temperature_k);
        n_ppm += dn_t;
        c_ppm += dc_t;
        ca_ppm += dca_t;
        if let Some(cb) = &mut cb_opt {
            *cb += dcb_t;
        }

        let sigma_n = if aa == "PRO" {
            cfg.sigma_pro_n_rc.max(0.40)
        } else {
            cfg.sigma_n_rc.max(0.35)
        };
        let sigma_cb = if aa == "CYS" {
            cfg.sigma_cb_rc.max(1.80)
        } else {
            cfg.sigma_cb_rc.max(0.25)
        };

        out.push(NcIdpResiduePrediction {
            residue: seq[k].clone(),
            n_ppm,
            sigma_n,
            c_ppm,
            sigma_c: cfg.sigma_c_rc.max(0.20),
            ca_ppm,
            sigma_ca: cfg.sigma_ca_rc.max(0.20),
            cb_ppm: cb_opt,
            sigma_cb,
            sidechain_c: expected_aliphatic_sidechain_c(aa).to_vec(),
        });
    }

    out
}

// ============================================================================
// Unlabelled Permutation-Invariant Global `(dN, dC', dCA, dCB)` Offset Estimator
// ============================================================================

/// Half the 95th-percentile quantile of $\chi^2_1$ ($\frac{1}{2}\chi^2_{1, 0.95} = \frac{1}{2}\times 3.8414588$),
/// combined with $\ln(N_{\text{grid,eff}})$ for Wilks / Gross–Vitells look-elsewhere hypothesis testing
/// (`Wilks, Ann. Math. Stat. 1938, 9, 60–62`; `Gross & Vitells, Eur. Phys. J. C 2010, 70, 525–530`).
const WILKS_HALF_CHI2_1_95: f64 = 1.920_729_410_347_062;

/// Fit a 1D shift offset `d` maximizing the unlabelled sequence-marginal outlier-robust mixture
/// log-likelihood + Huber IRLS polish over canonically sorted observations `obs_sorted`, gated by a
/// Godambe-adjusted composite-likelihood Wilks + look-elsewhere threshold
/// (`Rotnitzky & Jewell, Biometrika 1990, 77, 485–497`; `Varin, Reid & Firth, Stat. Sinica 2011, 21, 5–42`;
/// `Gross & Vitells, Eur. Phys. J. C 2010, 70, 525–530`):
/// $$\Delta\ell_{\text{crit}} = \lambda_{\text{Godambe}}\left[\ln(N_{\text{grid,eff}}) + \frac{1}{2}\chi^2_{1,0.95}\right], \qquad \lambda_{\text{Godambe}} = s_{\text{harm}}^2\,\frac{1}{n}\sum_{i=1}^n h_i.$$
pub(crate) fn fit_1d_unlabelled_marginal_offset(
    obs_sorted: &[f64],
    model_sorted: &[(f64, f64)],
    range_ppm: f64,
    max_abs_offset: f64,
    grid_step: f64,
) -> f64 {
    if obs_sorted.len() < 5 || model_sorted.is_empty() {
        return 0.0;
    }
    let inv_m = 1.0 / (model_sorted.len() as f64);
    let bg = 0.06 / range_ppm.max(4.0);
    let fg = 0.94;

    let inv_s_sum: f64 = model_sorted
        .iter()
        .map(|&(_, sd)| 1.0 / (sd * 1.15).max(0.30))
        .sum();
    let s_harm = (model_sorted.len() as f64) / inv_s_sum.max(1e-12);
    let corr_step = (std::f64::consts::SQRT_2 * s_harm).max(0.25);
    let n_grid_eff = ((2.0 * max_abs_offset) / corr_step).ceil().max(6.0);

    // Mean composite log-density curvature `bar_h = (1/n) sum_i [-d^2/dd^2 ln p(v_i | d)]` at `d = 0`,
    // giving the dimensionless composite-to-labelled curvature ratio `lambda = s_harm^2 * bar_h`.
    let godambe_lambda = |d_ref: f64| -> f64 {
        let h_eps = 0.025f64;
        let inv_h2 = 1.0 / (h_eps * h_eps);
        let eval_one = |v: f64, d: f64| -> f64 {
            let mut density = 0.0f64;
            for &(mu, sd) in model_sorted {
                let s_eff = (sd * 1.15).max(0.30);
                let z = (v - (mu + d)) / s_eff;
                if z.abs() < 8.0 {
                    density += inv_m * (-0.5 * z * z).exp() / (s_eff * TAU.sqrt());
                }
            }
            (fg * density + bg).max(1e-30).ln()
        };
        let mut h_sum = 0.0f64;
        for &v in obs_sorted {
            let lp = eval_one(v, d_ref + h_eps);
            let l0 = eval_one(v, d_ref);
            let lm = eval_one(v, d_ref - h_eps);
            let hi = (-(lp - 2.0 * l0 + lm) * inv_h2).max(0.0);
            h_sum += hi;
        }
        let bar_h = h_sum / (obs_sorted.len() as f64);
        (s_harm * s_harm * bar_h).clamp(0.12, 1.0)
    };
    let ll_crit = godambe_lambda(0.0) * (n_grid_eff.ln() + WILKS_HALF_CHI2_1_95);

    let eval_ll = |d: f64| -> f64 {
        let mut ll = -0.04 * d * d;
        for &v in obs_sorted {
            let mut density = 0.0f64;
            for &(mu, sd) in model_sorted {
                let s_eff = (sd * 1.15).max(0.30);
                let z = (v - (mu + d)) / s_eff;
                if z.abs() < 8.0 {
                    density += inv_m * (-0.5 * z * z).exp() / (s_eff * TAU.sqrt());
                }
            }
            ll += (fg * density + bg).max(1e-30).ln();
        }
        ll
    };

    let n_steps = ((2.0 * max_abs_offset) / grid_step).round() as usize;
    let mut scores = Vec::with_capacity(n_steps + 1);
    let mut best_idx = 0usize;
    let mut best_ll = f64::NEG_INFINITY;
    for i in 0..=n_steps {
        let d = -max_abs_offset + (i as f64) * grid_step;
        let sc = eval_ll(d);
        if sc > best_ll {
            best_ll = sc;
            best_idx = i;
        }
        scores.push(sc);
    }

    let base_ll = eval_ll(0.0);
    let mut d_est = -max_abs_offset + (best_idx as f64) * grid_step;
    if best_idx > 0 && best_idx < n_steps {
        let ym1 = scores[best_idx - 1];
        let y0 = scores[best_idx];
        let yp1 = scores[best_idx + 1];
        let den = ym1 - 2.0 * y0 + yp1;
        if den < -1e-12 {
            let frac = (0.5 * (ym1 - yp1) / den).clamp(-0.5, 0.5);
            d_est += frac * grid_step;
        }
    }

    // 4-step posterior-responsibility Huber IRLS polish (strictly unlabelled & order-independent)
    let huber_k = 1.345f64;
    for _ in 0..4 {
        let mut num = 0.0f64;
        let mut den = 1e-9f64;
        for &v in obs_sorted {
            let mut post_res_num = 0.0f64;
            let mut post_den = bg;
            for &(mu, sd) in model_sorted {
                let s_eff = (sd * 1.10).max(0.30);
                let r = v - (mu + d_est);
                let z = r / s_eff;
                if z.abs() < 7.0 {
                    let comp = fg * inv_m * (-0.5 * z * z).exp() / (s_eff * TAU.sqrt());
                    let huber_w = if z.abs() <= huber_k {
                        1.0
                    } else {
                        huber_k / z.abs()
                    };
                    let w = comp * huber_w / (s_eff * s_eff);
                    post_res_num += w * r;
                    post_den += comp;
                }
            }
            if post_den > 1e-24 {
                num += post_res_num / post_den;
                // Corresponding Fisher weight denominator
                let mut f_w = 0.0f64;
                for &(mu, sd) in model_sorted {
                    let s_eff = (sd * 1.10).max(0.30);
                    let r = v - (mu + d_est);
                    let z = r / s_eff;
                    if z.abs() < 7.0 {
                        let comp = fg * inv_m * (-0.5 * z * z).exp() / (s_eff * TAU.sqrt());
                        let huber_w = if z.abs() <= huber_k {
                            1.0
                        } else {
                            huber_k / z.abs()
                        };
                        f_w += comp * huber_w / (s_eff * s_eff);
                    }
                }
                den += f_w / post_den;
            }
        }
        let step = (num / den).clamp(-0.15, 0.15);
        let cand = (d_est + step).clamp(-max_abs_offset, max_abs_offset);
        if eval_ll(cand) >= eval_ll(d_est) - 1e-6 {
            d_est = cand;
        }
    }

    if eval_ll(d_est) > base_ll + ll_crit && d_est.abs() >= 0.015 {
        d_est
    } else {
        0.0
    }
}

/// Strictly unlabelled, bit-for-bit permutation-invariant global `(dN, dC', dCA, dCB)` offset
/// estimator matching unlabelled scalar peak distributions against the unlabelled `ncIDP`
/// sequence-marginal mixture.
///
/// Permuting the order of `systems` (or candidate vectors within any system) produces `0.0` change.
pub fn estimate_idp_reference_offsets(
    systems: &[ConDipeptideSystem],
    sequence: &[String],
    cfg: &IdpSolverConfig,
) -> IdpShiftOffsets {
    if systems.len() < 5 || sequence.len() < 2 {
        return IdpShiftOffsets::default();
    }
    let preds = predict_ncidp_shifts(sequence, cfg);

    // 1. Canonically sorted unlabelled observations (guarantees 0.0 permutation sensitivity)
    let mut obs_n: Vec<f64> = systems
        .iter()
        .map(|s| s.n_curr)
        .filter(|v| v.is_finite() && (102.0..=145.0).contains(v))
        .collect();
    obs_n.sort_by(|a, b| a.total_cmp(b));

    let mut obs_c: Vec<f64> = systems
        .iter()
        .map(|s| s.c_prev)
        .filter(|v| v.is_finite() && (166.0..=183.5).contains(v))
        .collect();
    obs_c.sort_by(|a, b| a.total_cmp(b));

    let mut obs_ca: Vec<f64> = Vec::with_capacity(systems.len() * 2);
    let mut obs_cb: Vec<f64> = Vec::with_capacity(systems.len());
    for s in systems {
        let mut ca_s: Vec<f64> = s
            .ca_curr
            .iter()
            .chain(s.ca_prev.iter())
            .copied()
            .filter(|v| v.is_finite() && (40.0..=69.5).contains(v))
            .collect();
        ca_s.sort_by(|a, b| a.total_cmp(b));
        ca_s.dedup_by(|a, b| (*a - *b).abs() < 0.04);
        obs_ca.extend(ca_s);

        let mut cb_s: Vec<f64> = s
            .cb_prev
            .iter()
            .chain(s.cb_curr.iter())
            .copied()
            .filter(|v| v.is_finite() && (13.0..=73.5).contains(v))
            .collect();
        cb_s.sort_by(|a, b| a.total_cmp(b));
        cb_s.dedup_by(|a, b| (*a - *b).abs() < 0.06);
        obs_cb.extend(cb_s.into_iter().take(MAX_IDP_CB_CANDS));
    }
    obs_ca.sort_by(|a, b| a.total_cmp(b));
    obs_cb.sort_by(|a, b| a.total_cmp(b));

    // 2. Unlabelled sequence-marginal prior components sorted canonically
    let mut mod_n: Vec<(f64, f64)> = preds[1..].iter().map(|p| (p.n_ppm, p.sigma_n)).collect();
    mod_n.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut mod_c: Vec<(f64, f64)> = preds[..preds.len() - 1]
        .iter()
        .map(|p| (p.c_ppm, p.sigma_c))
        .collect();
    mod_c.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut mod_ca: Vec<(f64, f64)> = preds.iter().map(|p| (p.ca_ppm, p.sigma_ca)).collect();
    mod_ca.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut mod_cb: Vec<(f64, f64)> = preds[..preds.len() - 1]
        .iter()
        .filter_map(|p| p.cb_ppm.map(|cb| (cb, p.sigma_cb)))
        .collect();
    mod_cb.sort_by(|a, b| a.0.total_cmp(&b.0));

    let d_n = fit_1d_unlabelled_marginal_offset(&obs_n, &mod_n, 38.0, 4.0, 0.02);
    let d_c = fit_1d_unlabelled_marginal_offset(&obs_c, &mod_c, 15.0, 4.0, 0.01);
    let d_ca = fit_1d_unlabelled_marginal_offset(&obs_ca, &mod_ca, 29.0, 4.0, 0.01);
    let d_cb = fit_1d_unlabelled_marginal_offset(&obs_cb, &mod_cb, 58.0, 4.0, 0.015);

    IdpShiftOffsets {
        d_n,
        d_c,
        d_ca,
        d_cb,
    }
}

/// Plausibility and referencing-convention audit for `idp-13c` auto-reference offsets (`RD-2`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct IdpAutoRefSafetyReport {
    pub raw_offsets: IdpShiftOffsets,
    pub safe_offsets: IdpShiftOffsets,
    pub refused_nuclei: Vec<String>,
    pub c13_spread_ppm: f64,
    pub polarity_swap_warning: Option<String>,
    pub tms_reference_hint: Option<String>,
    pub warnings: Vec<String>,
}

/// Enforce `±3.5 ppm` (`13C`) and `±1.5 ppm` (`15N`) plausibility caps, warn on non-uniform `> 1.0 ppm`
/// `(13CA, 13CB, 13C')` offset spread (polarity/CA-CB swap signature), and report a `"likely
/// TMS-referenced 13C"` hint when observed carbons uniformly shift `-2..-3 ppm` (`RD-2`).
pub fn evaluate_idp_auto_reference_safety(
    systems: &[ConDipeptideSystem],
    raw: &IdpShiftOffsets,
) -> IdpAutoRefSafetyReport {
    let n_c = systems.len();
    let n_ca = systems
        .iter()
        .filter(|s| !s.ca_curr.is_empty() || !s.ca_prev.is_empty())
        .count();
    let n_cb = systems
        .iter()
        .filter(|s| !s.cb_prev.is_empty() || !s.cb_curr.is_empty())
        .count();

    let mut safe = *raw;
    let mut refused_nuclei = Vec::new();
    let mut warnings = Vec::new();

    if raw.d_n.abs() > MAX_AUTO_REF_N15_PPM {
        warnings.push(format!(
            "refused implausible 15N IDP auto-reference offset ({:+.2} ppm > ±{MAX_AUTO_REF_N15_PPM:.2} ppm cap)",
            raw.d_n
        ));
        refused_nuclei.push("15N".to_string());
        safe.d_n = 0.0;
    }
    for (lbl, val, slot) in [
        ("13C'", raw.d_c, &mut safe.d_c),
        ("13CA", raw.d_ca, &mut safe.d_ca),
        ("13CB", raw.d_cb, &mut safe.d_cb),
    ] {
        if val.abs() > MAX_AUTO_REF_C13_PPM {
            warnings.push(format!(
                "refused implausible {lbl} IDP auto-reference offset ({val:+.2} ppm > ±{MAX_AUTO_REF_C13_PPM:.2} ppm cap)"
            ));
            refused_nuclei.push(lbl.to_string());
            *slot = 0.0;
        }
    }

    // Atomic 13C refusal: if ANY carbon channel is refused, zero all 13C offsets (dC', dCA, dCB)
    // regardless of spread so carbon channels are never left inconsistently referenced.
    if refused_nuclei
        .iter()
        .any(|n| matches!(n.as_str(), "13CA" | "13CB" | "13C'"))
    {
        safe.d_c = 0.0;
        safe.d_ca = 0.0;
        safe.d_cb = 0.0;
    }

    // Only genuinely significant/fitted carbon channels (`|d| >= 0.015 ppm`) participate in
    // carbon spread; `NotSignificant` (`0.0`) channels are excluded so they cannot inflate spread
    // against a uniform TMS shift or raise false polarity warnings.
    let mut obs_c_offsets: Vec<(&str, f64)> = Vec::new();
    if n_c >= 5 && raw.d_c.abs() >= 0.015 {
        obs_c_offsets.push(("C'", raw.d_c));
    }
    if n_ca >= 5 && raw.d_ca.abs() >= 0.015 {
        obs_c_offsets.push(("CA", raw.d_ca));
    }
    if n_cb >= 5 && raw.d_cb.abs() >= 0.015 {
        obs_c_offsets.push(("CB", raw.d_cb));
    }
    let any_c_shifted = obs_c_offsets.iter().any(|x| x.1.abs() > 0.1);
    let mut c13_spread_ppm = 0.0;
    let mut polarity_swap_warning = None;
    if any_c_shifted && obs_c_offsets.len() >= 2 {
        let min_c = obs_c_offsets
            .iter()
            .map(|x| x.1)
            .fold(f64::INFINITY, f64::min);
        let max_c = obs_c_offsets
            .iter()
            .map(|x| x.1)
            .fold(f64::NEG_INFINITY, f64::max);
        c13_spread_ppm = max_c - min_c;
        if c13_spread_ppm > AUTO_REF_C13_SPREAD_WARN_PPM {
            let desc = obs_c_offsets
                .iter()
                .map(|(l, v)| format!("d{l}={v:+.2}"))
                .collect::<Vec<_>>()
                .join(", ");
            let msg = format!(
                "non-uniform 13C IDP offsets ({desc}; spread {c13_spread_ppm:.2} > {AUTO_REF_C13_SPREAD_WARN_PPM:.2} ppm) — check for HNCACB/HN(CO)CACB polarity or CA/CB swap (RD-1 / --hncacb-polarity)"
            );
            warnings.push(msg.clone());
            polarity_swap_warning = Some(msg);
        }
    }

    let mut safe_c_fitted = Vec::new();
    if n_c >= 5 && safe.d_c.abs() > 0.1 {
        safe_c_fitted.push(safe.d_c);
    }
    if n_ca >= 5 && safe.d_ca.abs() > 0.1 {
        safe_c_fitted.push(safe.d_ca);
    }
    if n_cb >= 5 && safe.d_cb.abs() > 0.1 {
        safe_c_fitted.push(safe.d_cb);
    }
    let tms_reference_hint =
        if safe_c_fitted.len() >= 2 && c13_spread_ppm <= AUTO_REF_C13_SPREAD_WARN_PPM {
            let min_s = safe_c_fitted.iter().copied().fold(f64::INFINITY, f64::min);
            let max_s = safe_c_fitted
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let mean_s = safe_c_fitted.iter().sum::<f64>() / (safe_c_fitted.len() as f64);
            if safe_c_fitted
                .iter()
                .all(|v| (-MAX_AUTO_REF_C13_PPM..=-1.65).contains(v))
                && (-3.35..=-1.75).contains(&mean_s)
                && (max_s - min_s) <= AUTO_REF_C13_SPREAD_WARN_PPM
            {
                Some(format!(
                "likely TMS-referenced 13C (uniform {:+.2} ppm carbon offset vs DSS/ncIDP priors)",
                mean_s
            ))
            } else {
                None
            }
        } else {
            None
        };

    IdpAutoRefSafetyReport {
        raw_offsets: *raw,
        safe_offsets: safe,
        refused_nuclei,
        c13_spread_ppm,
        polarity_swap_warning,
        tms_reference_hint,
        warnings,
    }
}

// ============================================================================
// Unsupervised Inter-Experiment Peak Registration & Spin-System Assembly
// ============================================================================

/// Estimate 1D relative drift `delta` (`obs ≈ ref + delta`) across per-query-peak / per-root
/// difference groups via a two-component Gaussian-coincidence vs uniform-background profile
/// mixture log-likelihood, probing out to `±probe_max_in` and applying the Wilks / Gross–Vitells
/// look-elsewhere penalty $\Delta\ell_{\text{crit}} = \ln(N_{\text{grid,eff}}) + \frac{1}{2}\chi^2_{1, 0.95}$
/// together with a minimum distinct-query-unit support guard (`>= 5` distinct query roots/peaks
/// within `±2.5 * bw` of `best_d`) so sparse unanchored noise spikes never receive $O(M)$
/// correlated Cartesian-product votes. Returns `(clamped_d, raw_d)`.
fn fit_unlabelled_pairwise_drift_grouped(
    groups: &[Vec<f64>],
    max_shift: f64,
    probe_max_in: f64,
    bw: f64,
) -> (f64, f64) {
    let active: Vec<&[f64]> = groups
        .iter()
        .filter(|g| !g.is_empty())
        .map(|g| g.as_slice())
        .collect();
    if active.len() < 5 || bw <= 0.0 {
        return (0.0, 0.0);
    }
    let mut diffs_sorted: Vec<f64> = Vec::new();
    for g in &active {
        diffs_sorted.extend_from_slice(g);
    }
    diffs_sorted.sort_by(|a, b| a.total_cmp(b));
    if diffs_sorted.len() < 5 {
        return (0.0, 0.0);
    }

    let probe_max = probe_max_in.max(max_shift);
    let w_half = diffs_sorted
        .iter()
        .fold(probe_max + 3.0 * bw, |acc, &x| acc.max(x.abs()))
        .max(0.25);
    let u_bg = 1.0 / (2.0 * w_half);
    let g_norm = 1.0 / (bw * TAU.sqrt());
    let mut g_buf = vec![0.0f64; diffs_sorted.len()];

    let eval_kernel = |d: f64| -> f64 {
        let mut s = 0.0f64;
        for &df in &diffs_sorted {
            let z = (df - d) / bw;
            if z.abs() < 6.0 {
                s += (-0.5 * z * z).exp();
            }
        }
        s
    };

    let mut eval_profile_ll = |d: f64| -> f64 {
        for (slot, &df) in g_buf.iter_mut().zip(diffs_sorted.iter()) {
            let z = (df - d) / bw;
            *slot = if z.abs() < 8.0 {
                g_norm * (-0.5 * z * z).exp()
            } else {
                0.0
            };
        }
        // Strictly concave 1D EM maximization over coincidence fraction `pi in [0.0, 0.98]`
        let inv_n = 1.0 / (diffs_sorted.len() as f64);
        let mut pi = 0.18f64;
        for _ in 0..14 {
            let mut post_sum = 0.0f64;
            for &g in &g_buf {
                let num = pi * g;
                let den = num + (1.0 - pi) * u_bg;
                if den > 1e-30 {
                    post_sum += num / den;
                }
            }
            let next_pi = (post_sum * inv_n).clamp(0.0, 0.98);
            if (next_pi - pi).abs() < 1e-6 {
                pi = next_pi;
                break;
            }
            pi = next_pi;
        }
        let mut ll = 0.0f64;
        for &g in &g_buf {
            ll += (pi * g + (1.0 - pi) * u_bg).max(1e-30).ln();
        }
        ll
    };

    let mut best_d = 0.0f64;
    let mut best_s = eval_kernel(0.0);
    for step in -140i32..=140 {
        let d = (step as f64) * (probe_max / 140.0);
        let sc = eval_kernel(d);
        if sc > best_s {
            best_s = sc;
            best_d = d;
        }
    }
    let d0 = best_d;
    let mut best_ll = eval_profile_ll(d0);
    best_d = d0;
    for step in -10i32..=10 {
        let d = (d0 + (step as f64) * 0.002).clamp(-probe_max, probe_max);
        let ll = eval_profile_ll(d);
        if ll > best_ll {
            best_ll = ll;
            best_d = d;
        }
    }

    let distinct_hits = active
        .iter()
        .filter(|g| g.iter().any(|&df| (df - best_d).abs() <= 2.5 * bw))
        .count();
    let base_ll = eval_profile_ll(0.0);
    let n_grid_eff = ((2.0 * probe_max) / (std::f64::consts::SQRT_2 * bw))
        .ceil()
        .max(12.0);
    let ll_crit = n_grid_eff.ln() + WILKS_HALF_CHI2_1_95;

    if distinct_hits >= 5 && best_ll > base_ll + ll_crit && best_d.abs() >= 0.012 {
        (best_d.clamp(-max_shift, max_shift), best_d)
    } else {
        (0.0, 0.0)
    }
}

fn dedup_push(vec: &mut Vec<f64>, val: f64, tol: f64, max_len: usize) {
    if !val.is_finite() {
        return;
    }
    if let Some(existing) = vec.iter_mut().find(|&&mut x| (x - val).abs() <= tol) {
        *existing = 0.5 * (*existing + val);
    } else if vec.len() < max_len {
        vec.push(val);
    }
}

/// Register inter-experiment `(ΔC', ΔCA/CB)` carrier drift across `CON`, `CACO`, `CANCO`,
/// `CBCACON`, `CCC(CO)N`, and `NCACO` strictly without sequence/priors, and assemble
/// `ConDipeptideSystem` spin systems anchored on `picked.con_roots`.
pub fn register_and_build_con_dipeptide_systems(
    picked: &C13PickedDataset,
) -> Vec<ConDipeptideSystem> {
    let n_roots = picked.con_roots.len();
    if n_roots == 0 {
        return Vec::new();
    }

    let mut reg_warnings: Vec<String> = Vec::new();
    let mut record_drift_warn = |label: &str, clamped_d: f64, raw_d: f64| {
        if raw_d.abs() >= SOLVER_WARN_INTER_EXP_DRIFT_PPM {
            reg_warnings.push(format!(
                "Solver inter-experiment drift hit/approached ±{:.2} ppm clamp ({label}: fitted {raw_d:+.3} ppm, applied {clamped_d:+.3} ppm); verify spectrum carrier calibration",
                SOLVER_MAX_INTER_EXP_DRIFT_PPM
            ));
        }
    };

    // Group 3D strip peaks by `(exp, anchor_root_idx)` and collect unanchored `NCACO` peaks
    let mut canco_by_root: Vec<Vec<(f64, f64)>> = vec![Vec::new(); n_roots];
    let mut cbca_by_root: Vec<Vec<(f64, i32, f64)>> = vec![Vec::new(); n_roots];
    let mut cccon_by_root: Vec<Vec<(f64, f64)>> = vec![Vec::new(); n_roots];
    let mut anchored_ncaco: Vec<(usize, f64, f64, f64, f64)> = Vec::new();
    let mut unanchored_ncaco: Vec<(f64, f64, f64, f64)> = Vec::new();

    for p in &picked.strip_peaks {
        match (p.exp, p.anchor_root_idx) {
            (C13DetectedExp::Canco3D, Some(r_idx)) if r_idx < n_roots => {
                canco_by_root[r_idx].push((p.c_indirect_ppm, p.height.abs()));
            }
            (C13DetectedExp::Cbcaco3D, Some(r_idx)) if r_idx < n_roots => {
                cbca_by_root[r_idx].push((p.c_indirect_ppm, p.sign, p.height.abs()));
            }
            (C13DetectedExp::Cccon3D, Some(r_idx)) if r_idx < n_roots => {
                cccon_by_root[r_idx].push((p.c_indirect_ppm, p.height.abs()));
            }
            (C13DetectedExp::Ncaco3D, Some(r_idx)) if r_idx < n_roots => {
                anchored_ncaco.push((
                    r_idx,
                    p.c_prime_ppm,
                    p.n_ppm,
                    p.c_indirect_ppm,
                    p.height.abs(),
                ));
            }
            (C13DetectedExp::Ncaco3D, None) => {
                unanchored_ncaco.push((p.c_prime_ppm, p.n_ppm, p.c_indirect_ppm, p.height.abs()));
            }
            _ => {}
        }
    }

    // 1. Align `CBCACON` and `CCC(CO)N` indirect aliphatic axes onto `CANCO`'s `13CA` frame first
    let has_canco = canco_by_root.iter().any(|v| !v.is_empty());
    let mut groups_cbca_vs_canco: Vec<Vec<f64>> = Vec::with_capacity(n_roots);
    if has_canco {
        for r in 0..n_roots {
            let mut grp = Vec::new();
            for &(ca_cb, sgn, _) in &cbca_by_root[r] {
                if sgn >= 0 && (39.0..=71.0).contains(&ca_cb) {
                    for &(ca_cn, _) in &canco_by_root[r] {
                        let df = ca_cb - ca_cn;
                        if df.abs() <= 1.80 {
                            grp.push(df);
                        }
                    }
                }
            }
            if !grp.is_empty() {
                groups_cbca_vs_canco.push(grp);
            }
        }
    }
    let (d_cbca_aliph, raw_cbca_aliph) = fit_unlabelled_pairwise_drift_grouped(
        &groups_cbca_vs_canco,
        SOLVER_MAX_INTER_EXP_DRIFT_PPM,
        1.60,
        0.045,
    );
    record_drift_warn("CBCACON_aliph", d_cbca_aliph, raw_cbca_aliph);

    let mut groups_cccon_vs_ref: Vec<Vec<f64>> = Vec::with_capacity(n_roots);
    for r in 0..n_roots {
        let mut grp = Vec::new();
        for &(c_cc, _) in &cccon_by_root[r] {
            for &(ca_cn, _) in &canco_by_root[r] {
                let df = c_cc - ca_cn;
                if df.abs() <= 1.80 {
                    grp.push(df);
                }
            }
            for &(c_cb, _, _) in &cbca_by_root[r] {
                let df = c_cc - (c_cb - d_cbca_aliph);
                if df.abs() <= 1.80 {
                    grp.push(df);
                }
            }
        }
        if !grp.is_empty() {
            groups_cccon_vs_ref.push(grp);
        }
    }
    let (d_cccon_aliph, raw_cccon_aliph) = fit_unlabelled_pairwise_drift_grouped(
        &groups_cccon_vs_ref,
        SOLVER_MAX_INTER_EXP_DRIFT_PPM,
        1.60,
        0.045,
    );
    record_drift_warn("CCCON_aliph", d_cccon_aliph, raw_cccon_aliph);

    // 2. Align `2D CACO` `(C', CA)` onto `(CON C', CANCO/CBCACON CA)` frame BEFORE `NCACO` gating,
    // gating `C'` candidate differences by shared residue `(k-1)` `13CA` compatibility so narrow
    // random-coil `13C'` dispersion never drowns diagonal matches in $O(N^2)$ cross-residue pairs.
    let (d_caco_c, d_caco_ca) = if picked.caco_peaks.len() >= 5 {
        let root_ca_refs: Vec<Vec<f64>> = (0..n_roots)
            .map(|r_idx| {
                let mut v: Vec<f64> = canco_by_root[r_idx].iter().map(|&(ca, _)| ca).collect();
                for &(ca_cb, sgn, _) in &cbca_by_root[r_idx] {
                    if sgn >= 0 && (39.0..=71.0).contains(&ca_cb) {
                        v.push(ca_cb - d_cbca_aliph);
                    }
                }
                v
            })
            .collect();
        let has_any_root_ca = root_ca_refs.iter().any(|v| !v.is_empty());

        let build_caco_c_groups = |ca_offset: f64, ca_gate_tol: f64| -> Vec<Vec<f64>> {
            let mut grps: Vec<Vec<f64>> = Vec::with_capacity(picked.caco_peaks.len());
            for cp in &picked.caco_peaks {
                let ca_corr = cp.ca_curr_ppm - ca_offset;
                let mut grp = Vec::new();
                for (r_idx, r) in picked.con_roots.iter().enumerate() {
                    let ca_ok = !has_any_root_ca
                        || root_ca_refs[r_idx]
                            .iter()
                            .any(|&ca_r| (ca_corr - ca_r).abs() <= ca_gate_tol);
                    if ca_ok {
                        let df = cp.c_curr_ppm - r.c_prev_ppm;
                        if df.abs() <= 0.90 {
                            grp.push(df);
                        }
                    }
                }
                if !grp.is_empty() {
                    grps.push(grp);
                }
            }
            grps
        };

        let grps_c_1 = build_caco_c_groups(0.0, 1.65);
        let (dc_1, raw_dc_1) = fit_unlabelled_pairwise_drift_grouped(
            &grps_c_1,
            SOLVER_MAX_INTER_EXP_DRIFT_PPM,
            0.80,
            0.035,
        );

        let mut grps_ca: Vec<Vec<f64>> = Vec::with_capacity(picked.caco_peaks.len());
        for cp in &picked.caco_peaks {
            let c_corr = cp.c_curr_ppm - dc_1;
            let mut grp = Vec::new();
            for (r_idx, r) in picked.con_roots.iter().enumerate() {
                if (c_corr - r.c_prev_ppm).abs() <= 0.065 {
                    for &ca_r in &root_ca_refs[r_idx] {
                        let df = cp.ca_curr_ppm - ca_r;
                        if df.abs() <= 1.80 {
                            grp.push(df);
                        }
                    }
                }
            }
            if !grp.is_empty() {
                grps_ca.push(grp);
            }
        }
        let (dca, raw_dca) = fit_unlabelled_pairwise_drift_grouped(
            &grps_ca,
            SOLVER_MAX_INTER_EXP_DRIFT_PPM,
            1.60,
            0.045,
        );

        let (dc, raw_dc) = if has_any_root_ca {
            let grps_c_2 = build_caco_c_groups(dca, 0.22);
            if grps_c_2.len() >= 5 {
                fit_unlabelled_pairwise_drift_grouped(
                    &grps_c_2,
                    SOLVER_MAX_INTER_EXP_DRIFT_PPM,
                    0.80,
                    0.035,
                )
            } else {
                (dc_1, raw_dc_1)
            }
        } else {
            (dc_1, raw_dc_1)
        };
        record_drift_warn("CACO_C'", dc, raw_dc);
        record_drift_warn("CACO_CA", dca, raw_dca);
        (dc, dca)
    } else {
        (0.0, 0.0)
    };

    let caco_corr: Vec<(f64, f64, CacoPeakKind, f64)> = picked
        .caco_peaks
        .iter()
        .map(|p| {
            (
                p.c_curr_ppm - d_caco_c,
                p.ca_curr_ppm - d_caco_ca,
                p.kind,
                p.height.abs(),
            )
        })
        .collect();

    // 3. Unsupervised `(15N, 13CA, 13C')` registration for `3D NCACO` followed by joint `(N, CA_i)`
    // attachment and fanout-multiplicity downweighting so degenerate `15N` planes never broadcast
    // spurious or neighboring `NCACO` peaks unpenalized across unrelated `CON` roots!
    let d_ncaco_n_anch = if !anchored_ncaco.is_empty() {
        let mut grps_anch_n: Vec<Vec<f64>> = Vec::with_capacity(anchored_ncaco.len());
        for &(r_idx, _, n_p, _, _) in &anchored_ncaco {
            let df = n_p - picked.con_roots[r_idx].n_curr_ppm;
            if df.abs() <= 1.80 {
                grps_anch_n.push(vec![df]);
            }
        }
        let (d, raw_d) = fit_unlabelled_pairwise_drift_grouped(
            &grps_anch_n,
            SOLVER_MAX_INTER_EXP_DRIFT_PPM,
            1.60,
            0.055,
        );
        record_drift_warn("NCACO_anch_N", d, raw_d);
        d
    } else {
        0.0
    };

    let d_ncaco_n_unanch = if unanchored_ncaco.is_empty() {
        0.0
    } else if unanchored_ncaco.len() < 10 && anchored_ncaco.len() >= unanchored_ncaco.len() {
        // Residual unanchored peaks from the same 3D NCACO cube inherit its anchored 15N carrier drift
        d_ncaco_n_anch
    } else {
        let mut grps_nc_n: Vec<Vec<f64>> = Vec::with_capacity(unanchored_ncaco.len());
        for &(_, n_p, _, _) in &unanchored_ncaco {
            let mut grp = Vec::new();
            for r in &picked.con_roots {
                let df = n_p - r.n_curr_ppm;
                if df.abs() <= 0.92 {
                    grp.push(df);
                }
            }
            if !grp.is_empty() {
                grps_nc_n.push(grp);
            }
        }
        let (d, raw_d) = fit_unlabelled_pairwise_drift_grouped(
            &grps_nc_n,
            SOLVER_MAX_INTER_EXP_DRIFT_PPM,
            0.80,
            0.055,
        );
        record_drift_warn("NCACO_unanch_N", d, raw_d);
        d
    };

    let mut grps_ncaco_vs_canco: Vec<Vec<f64>> = Vec::new();
    if has_canco {
        for &(r_idx, _, n_p, ca_p, _) in &anchored_ncaco {
            if ((n_p - d_ncaco_n_anch) - picked.con_roots[r_idx].n_curr_ppm).abs() <= 0.16 {
                let mut grp = Vec::new();
                for &(ca_cn, _) in &canco_by_root[r_idx] {
                    let df = ca_p - ca_cn;
                    if df.abs() <= 1.80 {
                        grp.push(df);
                    }
                }
                if !grp.is_empty() {
                    grps_ncaco_vs_canco.push(grp);
                }
            }
        }
        for &(_, n_p, ca_p, _) in &unanchored_ncaco {
            let n_corr = n_p - d_ncaco_n_unanch;
            let mut grp = Vec::new();
            for (r_idx, r) in picked.con_roots.iter().enumerate() {
                if (n_corr - r.n_curr_ppm).abs() <= 0.14 {
                    for &(ca_cn, _) in &canco_by_root[r_idx] {
                        let df = ca_p - ca_cn;
                        if df.abs() <= 1.80 {
                            grp.push(df);
                        }
                    }
                }
            }
            if !grp.is_empty() {
                grps_ncaco_vs_canco.push(grp);
            }
        }
    } else if !caco_corr.is_empty() {
        for &(_, _, ca_p, _) in &unanchored_ncaco {
            let mut grp = Vec::new();
            for &(_, ca_co, _, _) in &caco_corr {
                let df = ca_p - ca_co;
                if df.abs() <= 0.92 {
                    grp.push(df);
                }
            }
            if !grp.is_empty() {
                grps_ncaco_vs_canco.push(grp);
            }
        }
    }
    let (d_ncaco_ca, raw_ncaco_ca) = fit_unlabelled_pairwise_drift_grouped(
        &grps_ncaco_vs_canco,
        SOLVER_MAX_INTER_EXP_DRIFT_PPM,
        0.90,
        0.045,
    );
    record_drift_warn("NCACO_CA", d_ncaco_ca, raw_ncaco_ca);

    let fit_ncaco_c_drift = |pks: &[(f64, f64)]| -> (f64, f64) {
        if pks.len() < 5 {
            return (0.0, 0.0);
        }
        let mut grps: Vec<Vec<f64>> = Vec::with_capacity(pks.len());
        if caco_corr.len() >= 5 {
            for &(c_p, ca_p) in pks {
                let ca_corr = ca_p - d_ncaco_ca;
                let mut grp = Vec::new();
                for &(c_co, ca_co, _, _) in &caco_corr {
                    if (ca_corr - ca_co).abs() <= 0.16 {
                        let df = c_p - c_co;
                        if df.abs() <= 0.90 {
                            grp.push(df);
                        }
                    }
                }
                if !grp.is_empty() {
                    grps.push(grp);
                }
            }
        }
        if grps.len() < 5 {
            grps.clear();
            for &(c_p, _) in pks {
                let mut grp = Vec::new();
                for r in &picked.con_roots {
                    let df = c_p - r.c_prev_ppm;
                    if df.abs() <= 0.90 {
                        grp.push(df);
                    }
                }
                if !grp.is_empty() {
                    grps.push(grp);
                }
            }
        }
        fit_unlabelled_pairwise_drift_grouped(&grps, SOLVER_MAX_INTER_EXP_DRIFT_PPM, 0.80, 0.035)
    };

    let mut uniq_anch_pairs: Vec<(f64, f64)> = anchored_ncaco
        .iter()
        .map(|&(_, c_p, _, ca_p, _)| (c_p, ca_p))
        .collect();
    uniq_anch_pairs.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.total_cmp(&b.1)));
    uniq_anch_pairs.dedup_by(|a, b| (a.0 - b.0).abs() <= 0.005 && (a.1 - b.1).abs() <= 0.01);
    let (d_ncaco_c_anch, raw_ncaco_c_anch) = fit_ncaco_c_drift(&uniq_anch_pairs);
    record_drift_warn("NCACO_anch_C'", d_ncaco_c_anch, raw_ncaco_c_anch);

    let unanch_pairs: Vec<(f64, f64)> = unanchored_ncaco
        .iter()
        .map(|&(c_p, _, ca_p, _)| (c_p, ca_p))
        .collect();
    let d_ncaco_c_unanch = if unanch_pairs.is_empty() {
        0.0
    } else if unanch_pairs.len() < 10 && uniq_anch_pairs.len() >= unanch_pairs.len() {
        d_ncaco_c_anch
    } else {
        let (d, raw_d) = fit_ncaco_c_drift(&unanch_pairs);
        record_drift_warn("NCACO_unanch_C'", d, raw_d);
        d
    };

    #[derive(Clone, Copy)]
    struct ValidatedNcacoCand {
        c_corr: f64,
        ca_corr: f64,
        height_eff: f64,
        dn_abs: f64,
    }

    let mut ncaco_by_root: Vec<Vec<ValidatedNcacoCand>> = vec![Vec::new(); n_roots];

    // Helper: count distinct CANCO shifts at root `r_idx` (after 0.08 ppm dedup) so we know
    // whether CANCO already observed both `CA(i)` and `CA(i-1)` (`>= 2`) vs had a dropout (`< 2`).
    let canco_distinct_counts: Vec<usize> = (0..n_roots)
        .map(|r| {
            let mut v: Vec<f64> = canco_by_root[r].iter().map(|x| x.0).collect();
            v.sort_by(|a, b| a.total_cmp(b));
            v.dedup_by(|a, b| (*a - *b).abs() <= 0.08);
            v.len()
        })
        .collect();

    // (a) Attach pre-anchored `NCACO` peaks with joint `(N, CA_i)` compatibility + fanout penalty
    if !anchored_ncaco.is_empty() {
        // Group identical underlying 3D NCACO peaks emitted across multiple roots sharing a `15N` slice
        let mut grouped: Vec<(f64, f64, f64, f64, Vec<usize>)> = Vec::new();
        for &(r_idx, c_p, n_p, ca_p, height) in &anchored_ncaco {
            if let Some(g) = grouped.iter_mut().find(|g| {
                (g.0 - c_p).abs() <= 0.004
                    && (g.1 - n_p).abs() <= 0.008
                    && (g.2 - ca_p).abs() <= 0.008
            }) {
                if !g.4.contains(&r_idx) {
                    g.4.push(r_idx);
                }
            } else {
                grouped.push((c_p, n_p, ca_p, height, vec![r_idx]));
            }
        }

        for (c_p, n_p, ca_p, height, cand_roots) in grouped {
            let c_corr = c_p - d_ncaco_c_anch;
            let n_corr = n_p - d_ncaco_n_anch;
            let ca_corr = ca_p - d_ncaco_ca;
            let has_caco_corrob = caco_corr.is_empty()
                || caco_corr.iter().any(|&(cc, cac, _, _)| {
                    (cc - c_corr).abs() <= 0.065 && (cac - ca_corr).abs() <= 0.15
                });

            let mut root_hits: Vec<(usize, f64, bool)> = Vec::new();
            for &r_idx in &cand_roots {
                let dn_abs = (n_corr - picked.con_roots[r_idx].n_curr_ppm).abs();
                let ca_match = canco_by_root[r_idx]
                    .iter()
                    .any(|&(ca_cn, _)| (ca_cn - ca_corr).abs() <= 0.22);
                root_hits.push((r_idx, dn_abs, ca_match));
            }
            let any_ca_match = root_hits.iter().any(|h| h.2);
            let kept: Vec<(usize, f64)> = root_hits
                .into_iter()
                .filter_map(|(r_idx, dn_abs, ca_match)| {
                    if !has_canco {
                        has_caco_corrob.then_some((r_idx, dn_abs))
                    } else if any_ca_match {
                        (ca_match || (canco_distinct_counts[r_idx] < 2 && dn_abs <= 0.045))
                            .then_some((r_idx, dn_abs))
                    } else if has_caco_corrob
                        && (canco_distinct_counts[r_idx] < 2 || cand_roots.len() == 1)
                    {
                        Some((r_idx, dn_abs))
                    } else {
                        None
                    }
                })
                .collect();
            let fanout = kept.len().max(1) as f64;
            let h_eff = height / fanout.powf(0.65);
            for (r_idx, dn_abs) in kept {
                ncaco_by_root[r_idx].push(ValidatedNcacoCand {
                    c_corr,
                    ca_corr,
                    height_eff: h_eff,
                    dn_abs,
                });
            }
        }
    }

    // (b) Attach unanchored `3D NCACO` peaks strictly via joint `(N, CA_i)` gating + fanout penalty
    for &(c_p, n_p, ca_p, height) in &unanchored_ncaco {
        let c_corr = c_p - d_ncaco_c_unanch;
        let n_corr = n_p - d_ncaco_n_unanch;
        let ca_corr = ca_p - d_ncaco_ca;
        let has_caco_corrob = caco_corr.is_empty()
            || caco_corr.iter().any(|&(cc, cac, _, _)| {
                (cc - c_corr).abs() <= 0.080 && (cac - ca_corr).abs() <= 0.18
            });

        let mut root_hits: Vec<(usize, f64, bool)> = Vec::new();
        for (r_idx, r) in picked.con_roots.iter().enumerate() {
            let dn_tol = if r.is_proline_imino_region {
                0.16
            } else {
                0.145
            };
            let dn_abs = (n_corr - r.n_curr_ppm).abs();
            if dn_abs <= dn_tol {
                let ca_match = canco_by_root[r_idx]
                    .iter()
                    .any(|&(ca_cn, _)| (ca_cn - ca_corr).abs() <= 0.24);
                root_hits.push((r_idx, dn_abs, ca_match));
            }
        }
        if root_hits.is_empty() {
            continue;
        }

        let any_ca_match = root_hits.iter().any(|h| h.2);
        let mut kept: Vec<(usize, f64)> = if !has_canco {
            if has_caco_corrob {
                root_hits
                    .into_iter()
                    .filter(|&(_, dn_abs, _)| dn_abs <= 0.12)
                    .map(|(r, dn, _)| (r, dn))
                    .collect()
            } else {
                Vec::new()
            }
        } else if any_ca_match {
            root_hits
                .into_iter()
                .filter(|&(r_idx, dn_abs, ca_match)| {
                    ca_match || (canco_distinct_counts[r_idx] < 2 && dn_abs <= 0.065)
                })
                .map(|(r, dn, _)| (r, dn))
                .collect()
        } else {
            // Rescue path when `CA(i)` dropped out in `CANCO` (or BMRB/experimental CACO also missed
            // the residue): only attach to roots that lack a full 2-peak `CANCO` pair (`< 2`)!
            let max_dn_rescue = if has_caco_corrob { 0.125 } else { 0.075 };
            let mut rescue: Vec<(usize, f64)> = root_hits
                .into_iter()
                .filter(|&(r_idx, dn_abs, _)| {
                    canco_distinct_counts[r_idx] < 2 && dn_abs <= max_dn_rescue
                })
                .map(|(r, dn, _)| (r, dn))
                .collect();
            rescue.sort_by(|a, b| a.1.total_cmp(&b.1));
            rescue.truncate(if has_caco_corrob { 2 } else { 1 });
            rescue
        };

        kept.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        let fanout = kept.len().max(1) as f64;
        let h_eff = height / fanout.powf(0.65);
        for (r_idx, dn_abs) in kept {
            ncaco_by_root[r_idx].push(ValidatedNcacoCand {
                c_corr,
                ca_corr,
                height_eff: h_eff,
                dn_abs,
            });
        }
    }

    let any_cbcacon = cbca_by_root.iter().any(|v| !v.is_empty());
    let any_cccon = cccon_by_root.iter().any(|v| !v.is_empty());
    let any_con_3d = has_canco || any_cbcacon || any_cccon;
    let has_both_signs_cbca = picked
        .strip_peaks
        .iter()
        .any(|p| p.exp == C13DetectedExp::Cbcaco3D && p.sign < 0);

    // Compute unlabelled median `2D CON` root height and median primary `CON`-anchored 3D peak
    // height to suppress spurious 2D thermal noise roots even when a single weak 3D noise spike lands nearby.
    let mut con_heights: Vec<f64> = picked.con_roots.iter().map(|r| r.height.abs()).collect();
    con_heights.sort_by(|a, b| a.total_cmp(b));
    let median_con_h = con_heights
        .get(con_heights.len() / 2)
        .copied()
        .unwrap_or(0.0);

    let mut prim_3d_heights: Vec<f64> = (0..n_roots)
        .filter_map(|r| {
            let h1 = canco_by_root[r].iter().map(|x| x.1).fold(0.0f64, f64::max);
            let h2 = cbca_by_root[r].iter().map(|x| x.2).fold(0.0f64, f64::max);
            let h3 = cccon_by_root[r].iter().map(|x| x.1).fold(0.0f64, f64::max);
            let hm = h1.max(h2).max(h3);
            (hm > 0.0).then_some(hm)
        })
        .collect();
    prim_3d_heights.sort_by(|a, b| a.total_cmp(b));
    let median_3d_h = prim_3d_heights
        .get(prim_3d_heights.len() / 2)
        .copied()
        .unwrap_or(0.0);

    // 4. Assemble per-root `ConDipeptideSystem`
    let mut systems = Vec::with_capacity(n_roots);
    for (r_idx, root) in picked.con_roots.iter().enumerate() {
        let mut sys = ConDipeptideSystem::new(r_idx, root.c_prev_ppm, root.n_curr_ppm, root.height);
        sys.has_cbcacon_evidence = any_cbcacon;

        let n_con_3d_pks =
            canco_by_root[r_idx].len() + cbca_by_root[r_idx].len() + cccon_by_root[r_idx].len();
        let max_root_3d_h = canco_by_root[r_idx]
            .iter()
            .map(|x| x.1)
            .chain(cbca_by_root[r_idx].iter().map(|x| x.2))
            .chain(cccon_by_root[r_idx].iter().map(|x| x.1))
            .fold(0.0f64, f64::max);
        let weak_single_spike = n_con_3d_pks == 1
            && median_con_h > 0.0
            && median_3d_h > 0.0
            && root.height.abs() < 0.45 * median_con_h
            && max_root_3d_h < 0.38 * median_3d_h;
        if any_con_3d && (n_con_3d_pks == 0 || weak_single_spike) {
            systems.push(sys);
            continue;
        }

        // Sort strip peaks deterministically by descending intensity (tie-break by shift)
        let mut canco = canco_by_root[r_idx].clone();
        canco.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.total_cmp(&b.0)));

        let mut cbca: Vec<(f64, i32, f64)> = cbca_by_root[r_idx]
            .iter()
            .map(|&(v, s, h)| (v - d_cbca_aliph, s, h))
            .collect();
        cbca.sort_by(|a, b| b.2.total_cmp(&a.2).then_with(|| a.0.total_cmp(&b.0)));

        let mut cccon: Vec<(f64, f64)> = cccon_by_root[r_idx]
            .iter()
            .map(|&(v, h)| (v - d_cccon_aliph, h))
            .collect();
        cccon.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.total_cmp(&b.0)));

        let mut ncaco: Vec<(f64, f64, f64, f64)> = ncaco_by_root[r_idx]
            .iter()
            .map(|nc_c| (nc_c.c_corr, nc_c.ca_corr, nc_c.height_eff, nc_c.dn_abs))
            .collect();
        ncaco.sort_by(|a, b| {
            b.2.total_cmp(&a.2)
                .then_with(|| a.3.total_cmp(&b.3))
                .then_with(|| a.0.total_cmp(&b.0))
        });

        // (a) Extract `CA(i-1)` and up to `MAX_IDP_CB_CANDS` (`4`) `CB(i-1)` candidates from `CBCACON`
        let max_pos_cbca = cbca
            .iter()
            .filter(|&&(_, s, _)| s >= 0)
            .map(|&(_, _, h)| h)
            .fold(0.0f64, f64::max);
        let max_neg_cbca = cbca
            .iter()
            .filter(|&&(_, s, _)| s < 0)
            .map(|&(_, _, h)| h)
            .fold(0.0f64, f64::max);

        for &(val, sgn, h) in &cbca {
            if has_both_signs_cbca {
                if sgn >= 0 && (40.0..=69.5).contains(&val) {
                    if sys.ca_prev.is_empty() || h >= 0.35 * max_pos_cbca {
                        dedup_push(&mut sys.ca_prev, val, 0.09, MAX_IDP_CANDS);
                    }
                } else if sgn < 0
                    && (13.0..=73.5).contains(&val)
                    && (sys.cb_prev.is_empty() || h >= 0.35 * max_neg_cbca)
                {
                    dedup_push(&mut sys.cb_prev, val, 0.11, MAX_IDP_CB_CANDS);
                }
            } else if (41.5..=65.5).contains(&val) && sys.ca_prev.is_empty() {
                dedup_push(&mut sys.ca_prev, val, 0.09, MAX_IDP_CANDS);
            } else if (13.0..=73.5).contains(&val) {
                dedup_push(&mut sys.cb_prev, val, 0.11, MAX_IDP_CB_CANDS);
            }
        }

        // (b) Check `2D CACO` at `c_prev` (`|c_caco - c_prev| <= 0.055 ppm`) for `Gly` opposite-phase
        // and corroboration / missing-relay rescue of `CA(i-1)`
        let mut caco_at_prev: Vec<(f64, CacoPeakKind, f64)> = caco_corr
            .iter()
            .filter(|&&(cc, _, _, _)| (cc - root.c_prev_ppm).abs() <= 0.055)
            .map(|&(_, ca_v, kind, h)| (ca_v, kind, h))
            .collect();
        caco_at_prev.sort_by(|a, b| b.2.total_cmp(&a.2).then_with(|| a.0.total_cmp(&b.0)));

        let mut corroborated_prev = Vec::new();
        for &p in &sys.ca_prev {
            if caco_at_prev
                .iter()
                .any(|&(ca_co, _, _)| (ca_co - p).abs() <= 0.14)
            {
                dedup_push(&mut corroborated_prev, p, 0.09, MAX_IDP_CANDS);
            }
        }
        for &(ca_cn, _) in &canco {
            if caco_at_prev
                .iter()
                .any(|&(ca_co, _, _)| (ca_co - ca_cn).abs() <= 0.14)
            {
                dedup_push(&mut corroborated_prev, ca_cn, 0.09, MAX_IDP_CANDS);
            }
        }
        if !corroborated_prev.is_empty() {
            for &old_p in &sys.ca_prev {
                if max_pos_cbca <= 0.0
                    || cbca.iter().any(|&(v, s, h)| {
                        s >= 0 && (v - old_p).abs() <= 0.09 && h >= 0.35 * max_pos_cbca
                    })
                {
                    dedup_push(&mut corroborated_prev, old_p, 0.09, MAX_IDP_CANDS);
                }
            }
            sys.ca_prev = corroborated_prev;
        } else if sys.ca_prev.is_empty() && caco_at_prev.len() <= 3 {
            for &(ca_co, _, _) in &caco_at_prev {
                dedup_push(&mut sys.ca_prev, ca_co, 0.09, MAX_IDP_CANDS);
            }
        }

        for &(ca_v, kind, _) in &caco_at_prev {
            if kind == CacoPeakKind::GlyOppositePhase
                && (sys.ca_prev.is_empty() || sys.ca_prev.iter().any(|&p| (p - ca_v).abs() <= 0.18))
            {
                sys.prev_gly_opposite_phase = true;
            }
        }
        if sys.prev_gly_opposite_phase
            && sys.ca_prev.first().is_some_and(|&ca| ca < 47.0)
            && max_pos_cbca > 0.0
            && max_neg_cbca < 0.42 * max_pos_cbca
        {
            sys.cb_prev.clear();
        }

        // Suppress `< 0.32 * max_canco` 3rd+ noise spikes in `CANCO`
        let max_canco_h = canco.first().map_or(0.0, |x| x.1);
        let canco_clean: Vec<(f64, f64)> = canco
            .iter()
            .enumerate()
            .filter(|&(idx, &(_, h))| idx < 2 || h >= 0.32 * max_canco_h)
            .map(|(_, &x)| x)
            .collect();

        // Corroborate `NCACO` `(C'(i), CA(i))` against `2D CACO` when `2D CACO` is present
        let max_ncaco_h = ncaco.first().map_or(0.0, |x| x.2);
        let mut corrob_ncaco: Vec<(f64, f64, f64, f64)> = if !caco_corr.is_empty() {
            ncaco
                .iter()
                .copied()
                .filter(|&(c_i, ca_i, _, _)| {
                    caco_corr.iter().any(|&(cc, cac, _, _)| {
                        (cc - c_i).abs() <= 0.060 && (cac - ca_i).abs() <= 0.14
                    })
                })
                .collect()
        } else {
            ncaco
                .iter()
                .copied()
                .filter(|&(_, _, h, _)| h >= 0.35 * max_ncaco_h)
                .collect()
        };
        if corrob_ncaco.is_empty() {
            for &(c_i, ca_i, h, dn_abs) in &ncaco {
                if h >= 0.40 * max_ncaco_h {
                    corrob_ncaco.push((c_i, ca_i, h, dn_abs));
                }
            }
        }
        if corrob_ncaco.len() > 1 {
            corrob_ncaco.sort_by(|a, b| {
                let ma = canco_clean.iter().any(|&(vc, _)| (vc - a.1).abs() <= 0.16);
                let mb = canco_clean.iter().any(|&(vc, _)| (vc - b.1).abs() <= 0.16);
                if ma != mb {
                    return mb.cmp(&ma);
                }
                let n_chi_a = (a.3 / 0.040).powi(2);
                let n_chi_b = (b.3 / 0.040).powi(2);
                if !caco_corr.is_empty() {
                    let da = n_chi_a
                        + caco_corr
                            .iter()
                            .map(|&(cc, cac, _, _)| {
                                ((cc - a.0) / 0.04).powi(2) + ((cac - a.1) / 0.09).powi(2)
                            })
                            .fold(f64::INFINITY, f64::min);
                    let db = n_chi_b
                        + caco_corr
                            .iter()
                            .map(|&(cc, cac, _, _)| {
                                ((cc - b.0) / 0.04).powi(2) + ((cac - b.1) / 0.09).powi(2)
                            })
                            .fold(f64::INFINITY, f64::min);
                    da.total_cmp(&db).then_with(|| b.2.total_cmp(&a.2))
                } else {
                    n_chi_a
                        .total_cmp(&n_chi_b)
                        .then_with(|| b.2.total_cmp(&a.2))
                }
            });
        }

        // (c) + (d) Jointly separate `CANCO` into `ca_curr` / `ca_prev` and attach `NCACO` `(c_curr, ca_curr)`
        let prev_was_empty = sys.ca_prev.is_empty();
        let non_prev_canco: Vec<f64> = canco_clean
            .iter()
            .map(|&(v, _)| v)
            .filter(|&v| !sys.ca_prev.iter().take(1).any(|&p| (p - v).abs() <= 0.13))
            .collect();

        if !non_prev_canco.is_empty() {
            for &v in &non_prev_canco {
                dedup_push(&mut sys.ca_curr, v, 0.045, MAX_IDP_CANDS);
            }
            if prev_was_empty && canco_clean.len() >= 2 {
                for &(v, _) in canco_clean.iter().rev() {
                    dedup_push(&mut sys.ca_prev, v, 0.045, MAX_IDP_CANDS);
                }
            } else {
                for &(v, _) in &canco_clean {
                    if !non_prev_canco.iter().any(|&c| (c - v).abs() <= 0.09) {
                        dedup_push(&mut sys.ca_prev, v, 0.045, MAX_IDP_CANDS);
                    }
                }
            }
            for &(v, _) in &canco_clean {
                if corrob_ncaco
                    .iter()
                    .any(|&(_, ca_nc, _, _)| (ca_nc - v).abs() <= 0.14)
                {
                    dedup_push(&mut sys.ca_curr, v, 0.045, MAX_IDP_CANDS);
                }
            }
            for &(c_i, ca_i, _, _) in &corrob_ncaco {
                if sys.ca_curr.iter().any(|&ca_c| (ca_c - ca_i).abs() <= 0.16) {
                    dedup_push(&mut sys.c_curr, c_i, 0.014, MAX_IDP_CANDS);
                    dedup_push(&mut sys.ca_curr, ca_i, 0.045, MAX_IDP_CANDS);
                }
            }
        } else if !corrob_ncaco.is_empty() {
            for &(c_i, ca_i, _, _) in corrob_ncaco.iter().take(5) {
                dedup_push(&mut sys.c_curr, c_i, 0.014, MAX_IDP_CANDS);
                dedup_push(&mut sys.ca_curr, ca_i, 0.045, MAX_IDP_CANDS);
            }
            for &(v, _) in &canco_clean {
                dedup_push(&mut sys.ca_curr, v, 0.045, MAX_IDP_CANDS);
                dedup_push(&mut sys.ca_prev, v, 0.045, MAX_IDP_CANDS);
            }
        } else {
            for &(v, _) in &canco_clean {
                dedup_push(&mut sys.ca_curr, v, 0.045, MAX_IDP_CANDS);
                dedup_push(&mut sys.ca_prev, v, 0.045, MAX_IDP_CANDS);
            }
        }

        if sys.c_curr.is_empty() && !sys.ca_curr.is_empty() && !caco_corr.is_empty() {
            let ca0 = sys.ca_curr[0];
            let cands_c: Vec<f64> = caco_corr
                .iter()
                .filter(|&&(_, cac, _, _)| (cac - ca0).abs() <= 0.15)
                .map(|&(cc, _, _, _)| cc)
                .collect();
            if cands_c.len() == 1
                && (caco_corr
                    .iter()
                    .map(|&(_, cac, _, _)| (cac - ca0).abs())
                    .fold(1.0, f64::min)
                    <= 0.08)
            {
                dedup_push(&mut sys.c_curr, cands_c[0], 0.014, MAX_IDP_CANDS);
            }
        }

        // (e) Incorporate `3D CCC(CO)N` aliphatic TOCSY relays for residue `(i-1)`
        for &(c_val, _) in &cccon {
            let is_ca = sys.ca_prev.iter().any(|&p| (p - c_val).abs() <= 0.16);
            let is_cb = sys.cb_prev.iter().any(|&p| (p - c_val).abs() <= 0.18);
            if is_ca || is_cb {
                continue;
            }
            if sys.ca_prev.is_empty() && (42.0..=66.0).contains(&c_val) {
                dedup_push(&mut sys.ca_prev, c_val, 0.10, MAX_IDP_CANDS);
            } else if sys.cb_prev.is_empty() && (26.0..=71.5).contains(&c_val) {
                dedup_push(&mut sys.cb_prev, c_val, 0.12, MAX_IDP_CB_CANDS);
            } else if (9.5..=56.0).contains(&c_val) {
                dedup_push(&mut sys.sidechain_c_prev, c_val, 0.15, MAX_IDP_SC_CANDS);
            }
        }
        // Ala H(CC)CON fallback (`CA ~ 52.5 ppm`, `CB ~ 19.1 ppm`) when no `>= 26 ppm` non-CA peak exists
        if sys.cb_prev.is_empty() && cccon.len() <= 3 {
            let has_mid_cb = cccon
                .iter()
                .any(|&(c, _)| (25.0..=41.5).contains(&c) || (62.0..=73.5).contains(&c));
            if !has_mid_cb {
                let mut promoted: Vec<f64> = Vec::new();
                for &(c_val, _) in &cccon {
                    if (15.5..=23.5).contains(&c_val)
                        && !sys.ca_prev.iter().any(|&p| (p - c_val).abs() <= 0.16)
                    {
                        dedup_push(&mut sys.cb_prev, c_val, 0.12, MAX_IDP_CB_CANDS);
                        promoted.push(c_val);
                    }
                }
                if !promoted.is_empty() {
                    sys.sidechain_c_prev
                        .retain(|v| !promoted.iter().any(|&p| (p - *v).abs() <= 1e-6));
                }
            }
        }

        systems.push(sys);
    }

    if !systems.is_empty() && !reg_warnings.is_empty() {
        systems[0].registration_warnings = reg_warnings;
    }

    systems
}

// ============================================================================
// Exact CPU / WebGPU Parity Scoring (`U` & `P` Matrices in Smooth `f32` Math)
// ============================================================================

#[inline]
fn logpdf_2comp_f32(v: f32, mu: f32, sd: f32) -> f32 {
    let s1 = sd.max(0.15);
    let s2 = 3.8 * s1;
    let z1 = (v - mu) / s1;
    let z2 = (v - mu) / s2;
    let inv_sqrt_2pi: f32 = 0.398_942_3;
    let p1 = 0.78 * inv_sqrt_2pi * (-0.5 * z1 * z1).exp() / s1;
    let p2 = 0.22 * inv_sqrt_2pi * (-0.5 * z2 * z2).exp() / s2;
    (p1 + p2).max(1.0e-30).ln()
}

/// Precomputed `20`-AA background shifts for `f32`-exact CPU/WebGPU background normalization.
pub const IDP_BG_N_MUS: [f32; 20] = [
    124.5, 121.3, 119.0, 120.7, 119.8, 120.3, 121.2, 109.7, 119.6, 121.2, 122.6, 121.6, 120.6,
    120.9, 136.8, 116.5, 115.0, 121.7, 120.8, 120.6,
];
pub const IDP_BG_C_MUS: [f32; 20] = [
    177.8, 176.3, 175.2, 176.4, 174.6, 176.0, 176.6, 173.9, 175.2, 176.4, 177.4, 176.6, 176.2,
    175.8, 176.9, 174.6, 174.6, 176.3, 175.7, 176.3,
];
pub const IDP_BG_CA_MUS: [f32; 20] = [
    52.5, 56.0, 53.1, 54.2, 58.2, 55.8, 56.6, 45.2, 55.8, 61.1, 55.1, 56.3, 55.4, 57.8, 63.2, 58.3,
    61.8, 57.4, 57.9, 62.2,
];
pub const IDP_BG_CB_MUS: [f32; 19] = [
    19.1, 30.8, 38.9, 41.1, 28.0, 29.5, 30.2, 29.8, 38.6, 42.3, 33.0, 32.8, 39.6, 32.1, 63.8, 69.8,
    29.6, 38.8, 32.8,
];
pub const IDP_BG_SC_MUS: [f32; 12] = [
    13.0, 17.3, 21.2, 24.4, 27.1, 29.1, 32.1, 33.8, 36.1, 42.0, 43.3, 50.4,
];

#[inline]
fn bg_logpdf_f32(v: f32, mus: &[f32], sd: f32) -> f32 {
    let mut sum = 0.0f32;
    let inv_n = 1.0 / (mus.len() as f32);
    let inv_sqrt_2pi: f32 = 0.398_942_3;
    let s1 = (sd * 1.25).max(0.25);
    for &m in mus {
        let z = (v - m) / s1;
        sum += inv_n * inv_sqrt_2pi * (-0.5 * z * z).exp() / s1;
    }
    (0.95 * sum + 0.05 / 45.0).max(1.0e-30).ln()
}

#[inline]
fn best_cand_llr_f32(
    cands: &[f32],
    mu: f32,
    sd: f32,
    bg_mus: &[f32],
    bg_sd: f32,
    llr_floor: f32,
) -> f32 {
    let n_cands = cands.len().min(MAX_IDP_CANDS);
    if n_cands == 0 {
        return 0.0;
    }
    let mut best = -1.0e30f32;
    for (idx, &v) in cands[..n_cands].iter().enumerate() {
        let rank_pen = 0.25 * (idx as f32);
        let llr = (logpdf_2comp_f32(v, mu, sd) - bg_logpdf_f32(v, bg_mus, bg_sd)).max(llr_floor)
            - rank_pen;
        if llr > best {
            best = llr;
        }
    }
    let mult_gate = (best / 1.5).clamp(0.0, 1.0);
    best - 0.12 * (n_cands as f32).ln() * mult_gate
}

/// Fallible version of [`idp_unary_matrix`] enforcing [`MAX_IDP_MATRIX_CELLS`] (`25,000,000`).
pub fn try_idp_unary_matrix(
    systems: &[ConDipeptideSystem],
    sequence: &[String],
    cfg: &IdpSolverConfig,
    offsets: &IdpShiftOffsets,
) -> Result<Matrix2D, String> {
    check_idp_matrix_bounds(systems.len(), sequence.len())?;
    Ok(idp_unary_matrix(systems, sequence, cfg, offsets))
}

/// Compute the continuous through-Proline IDP dipeptide unary scoring matrix `U` `(S x K)` on CPU.
///
/// - `k == 0` (N-terminus, no preceding peptide bond) is `FORBID` (`-1e9`).
/// - **Every `k in 1..K` — INCLUDING `PRO` at `k` and `PRO` at `k - 1` — is an active spin-system slot!**
/// - Uses smooth `f32`-compatible log-likelihood ratios matching `idp_scoring.wgsl` to `< 1e-5`.
///
/// Provenance of `sys_assign_bonus` (`+2.65` with 3D strip support, `-6.5` for 3D-unsupported
/// 2D roots in multi-3D datasets, `+0.5` in 2D-only mode):
/// - `+2.65` equals the prior log-odds $\ln((1 - p_{\text{spur,3D}}) / p_{\text{spur,3D}})$ at the
///   empirical `~6.6%` false-root rate after 3D strip corroboration (`ln(0.934 / 0.066) = 2.65`).
/// - `-6.5` equals the log-Bayes-factor penalty when all 3 independent `CON`-anchored 3D cubes
///   miss simultaneously (`p_drop^3 ~ 0.0015` vs spurious 2D thermal root prior), preventing
///   3D-unsupported 2D noise roots from filling empty sequence gaps while remaining finite (`> FORBID`).
pub fn idp_unary_matrix(
    systems: &[ConDipeptideSystem],
    sequence: &[String],
    cfg: &IdpSolverConfig,
    offsets: &IdpShiftOffsets,
) -> Matrix2D {
    let s_len = systems.len();
    let k_len = sequence.len();
    if check_idp_matrix_bounds(s_len, k_len).is_err() {
        return Matrix2D::filled(0, 0, FORBID);
    }
    let mut u = Matrix2D::filled(s_len, k_len, FORBID);
    if s_len == 0 || k_len < 2 {
        return u;
    }

    let preds = predict_ncidp_shifts(sequence, cfg);
    let dn = offsets.d_n as f32;
    let dc = offsets.d_c as f32;
    let dca = offsets.d_ca as f32;
    let dcb = offsets.d_cb as f32;
    let llr_floor = cfg.llr_floor as f32;
    let n_with_3d = systems.iter().filter(|s| s.has_3d_support()).count();
    let multi_3d_dataset = n_with_3d * 2 >= s_len;

    // Precompute CAS root hit masks once (`O(S * P_cas)` -> `O(1)` inside `(s_idx, k)` loop)
    let cas_root_coords: Vec<(f64, f64)> = if cfg.cas_spectra.is_empty() {
        Vec::new()
    } else {
        systems.iter().map(|s| (s.c_prev, s.n_curr)).collect()
    };
    let cas_precomp: Vec<(&CasSpectrum, Vec<bool>, usize, usize)> = cfg
        .cas_spectra
        .iter()
        .map(|cas| {
            let hits = match_cas_peaks_to_roots(
                cas,
                &cas_root_coords,
                cfg.cas_c_tol_ppm,
                cfg.cas_n_tol_ppm,
            );
            let n_hits = hits.iter().filter(|&&h| h).count();
            let seq_type_count = match cas.mode {
                CasMode::Cacon => (1..k_len)
                    .filter(|&k| cas.matches_residue(&sequence[k - 1]))
                    .count(),
                CasMode::Canco => (1..k_len)
                    .filter(|&k| cas.matches_residue(&sequence[k]))
                    .count(),
            };
            (cas, hits, n_hits, seq_type_count)
        })
        .collect();

    // Pre-shift background arrays into the experimental frame
    let bg_n: Vec<f32> = IDP_BG_N_MUS.iter().map(|&m| m + dn).collect();
    let bg_c: Vec<f32> = IDP_BG_C_MUS.iter().map(|&m| m + dc).collect();
    let bg_ca: Vec<f32> = IDP_BG_CA_MUS.iter().map(|&m| m + dca).collect();
    let bg_cb: Vec<f32> = IDP_BG_CB_MUS.iter().map(|&m| m + dcb).collect();
    let bg_sc: Vec<f32> = IDP_BG_SC_MUS.iter().map(|&m| m + dcb).collect();

    for (s_idx, sys) in systems.iter().enumerate() {
        let n_curr = sys.n_curr as f32;
        let c_prev = sys.c_prev as f32;
        let ca_curr: Vec<f32> = sys
            .ca_curr
            .iter()
            .take(MAX_IDP_CANDS)
            .map(|&x| x as f32)
            .collect();
        let cb_curr: Vec<f32> = sys
            .cb_curr
            .iter()
            .take(MAX_IDP_CANDS)
            .map(|&x| x as f32)
            .collect();
        let ca_prev: Vec<f32> = sys
            .ca_prev
            .iter()
            .take(MAX_IDP_CANDS)
            .map(|&x| x as f32)
            .collect();
        let cb_prev: Vec<f32> = sys
            .cb_prev
            .iter()
            .take(MAX_IDP_CANDS)
            .map(|&x| x as f32)
            .collect();
        let c_curr: Vec<f32> = sys
            .c_curr
            .iter()
            .take(MAX_IDP_CANDS)
            .map(|&x| x as f32)
            .collect();
        let sc_prev: Vec<f32> = sys
            .sidechain_c_prev
            .iter()
            .take(MAX_IDP_SC_CANDS)
            .map(|&x| x as f32)
            .collect();
        let n_3d_ch = u32::from(!ca_curr.is_empty())
            + u32::from(!cb_curr.is_empty())
            + u32::from(!ca_prev.is_empty())
            + u32::from(!cb_prev.is_empty())
            + u32::from(!c_curr.is_empty())
            + u32::from(!sc_prev.is_empty())
            + u32::from(!sys.nn_prev_cands.is_empty());
        let sys_assign_bonus: f32 = if n_3d_ch >= 1 {
            2.65
        } else if multi_3d_dataset {
            -6.5
        } else {
            0.5
        };

        for k in 1..k_len {
            let cur_p = &preds[k];
            let prv_p = &preds[k - 1];

            // 1. Root `15N(k)` & `13C'(k-1)`
            let mut total: f32 = sys_assign_bonus;
            total += (logpdf_2comp_f32(n_curr, (cur_p.n_ppm as f32) + dn, cur_p.sigma_n as f32)
                - bg_logpdf_f32(n_curr, &bg_n, 1.4))
            .max(llr_floor);
            total += (logpdf_2comp_f32(c_prev, (prv_p.c_ppm as f32) + dc, prv_p.sigma_c as f32)
                - bg_logpdf_f32(c_prev, &bg_c, 0.7))
            .max(llr_floor);

            // 2. `CA(k)` & `CA(k-1)`
            total += best_cand_llr_f32(
                &ca_curr,
                (cur_p.ca_ppm as f32) + dca,
                cur_p.sigma_ca as f32,
                &bg_ca,
                1.2,
                llr_floor,
            );
            total += best_cand_llr_f32(
                &ca_prev,
                (prv_p.ca_ppm as f32) + dca,
                prv_p.sigma_ca as f32,
                &bg_ca,
                1.2,
                llr_floor,
            );

            // 3. `CB(k-1)` and `Gly(k-1)` opposite-phase / missing-CB signatures
            if let Some(mu_cb) = prv_p.cb_ppm {
                if !cb_prev.is_empty() {
                    total += best_cand_llr_f32(
                        &cb_prev,
                        (mu_cb as f32) + dcb,
                        prv_p.sigma_cb as f32,
                        &bg_cb,
                        2.5,
                        llr_floor,
                    );
                } else if sys.has_cbcacon_evidence {
                    total -= 1.35;
                }
                if sys.prev_gly_opposite_phase {
                    total -= 4.50;
                }
            } else {
                if !cb_prev.is_empty() {
                    total -= 4.20;
                } else if sys.has_cbcacon_evidence && !ca_prev.is_empty() {
                    total += 2.10;
                }
                if sys.prev_gly_opposite_phase {
                    total += 3.60;
                }
            }

            // 3b. Intraresidue `CB(k)` when supplemented by `HNCO->HNCACB` (`RD-10`)
            if !cb_curr.is_empty() {
                if let Some(mu_cb_cur) = cur_p.cb_ppm {
                    total += best_cand_llr_f32(
                        &cb_curr,
                        (mu_cb_cur as f32) + dcb,
                        cur_p.sigma_cb as f32,
                        &bg_cb,
                        2.5,
                        llr_floor,
                    );
                } else {
                    total -= 4.20;
                }
            }

            // 4. Intraresidue `C'(k)` (`NCACO` / `CACO`)
            if !c_curr.is_empty() {
                total += 0.85
                    * best_cand_llr_f32(
                        &c_curr,
                        (cur_p.c_ppm as f32) + dc,
                        cur_p.sigma_c as f32,
                        &bg_c,
                        0.7,
                        llr_floor,
                    );
            }

            // 5. Aliphatic sidechain `CCC(CO)N` peaks of `k - 1`
            if !sc_prev.is_empty() {
                if prv_p.sidechain_c.is_empty() {
                    total -= 1.25 * (sc_prev.len().min(3) as f32);
                } else {
                    let exp_sc: Vec<f32> = prv_p
                        .sidechain_c
                        .iter()
                        .take(4)
                        .map(|&m| (m as f32) + dcb)
                        .collect();
                    let mut sc_sum = 0.0f32;
                    for &v in &sc_prev {
                        let mut best_comp = -1.0e30f32;
                        for &mu_s in &exp_sc {
                            let lp = logpdf_2comp_f32(v, mu_s, 0.75);
                            if lp > best_comp {
                                best_comp = lp;
                            }
                        }
                        let llr = (best_comp - bg_logpdf_f32(v, &bg_sc, 1.8)).max(-4.0);
                        sc_sum += 0.55 * llr;
                    }
                    total += sc_sum;
                }
            }

            // 6. Amino-acid-selective `(C'(i-1), N(i))` (`CAS`) residue-type evidence (`RD-4` / `RD-10`)
            if !cas_precomp.is_empty() {
                for (cas, hits, n_hits, seq_type_count) in &cas_precomp {
                    let target_res = match cas.mode {
                        CasMode::Cacon => &sequence[k - 1],
                        CasMode::Canco => &sequence[k],
                    };
                    total += cas_llr_adjustment(
                        cas,
                        hits[s_idx],
                        target_res,
                        cfg.cas_match_llr,
                        *seq_type_count,
                        *n_hits,
                    ) as f32;
                }
            }

            u.set(s_idx, k, total as f64);
        }
    }

    u
}

#[inline]
fn channel_pair_llr_f32(
    a_cands: &[f32],
    b_cands: &[f32],
    sigma: f32,
    range_ppm: f32,
    pair_floor: f32,
) -> f32 {
    let na = a_cands.len().min(MAX_IDP_CANDS);
    let nb = b_cands.len().min(MAX_IDP_CANDS);
    if na == 0 || nb == 0 {
        return 0.0;
    }
    let inv_sqrt_2pi: f32 = 0.398_942_3;
    let s = sigma.max(0.02);
    let bg_log = (1.0 / range_ppm.max(2.0)).ln();
    let norm_log = (inv_sqrt_2pi / s).ln();
    let mut best = -1.0e30f32;
    for (ia, &va) in a_cands[..na].iter().enumerate() {
        for (ib, &vb) in b_cands[..nb].iter().enumerate() {
            let z = (va - vb) / s;
            let lp = norm_log - 0.5 * z * z - 0.14 * ((ia + ib) as f32);
            let llr = (lp - bg_log).max(pair_floor);
            if llr > best {
                best = llr;
            }
        }
    }
    // Candidate-pair multiplicity penalty (`ln(na * nb) == 0` when `na == 1 && nb == 1`)
    let mult_gate = (best / 1.5).clamp(0.0, 1.0);
    best - 0.18 * ((na * nb) as f32).ln() * mult_gate
}

#[inline]
fn pair_compat_weight_f32(raw: f32) -> f32 {
    let u = ((raw - 2.10) / 2.40).clamp(0.0, 1.0);
    u * u * (3.0 - 2.0 * u)
}

/// Fallible version of [`idp_pair_matrix`] enforcing [`MAX_IDP_MATRIX_CELLS`] (`25,000,000`).
pub fn try_idp_pair_matrix(
    systems: &[ConDipeptideSystem],
    cfg: &IdpSolverConfig,
) -> Result<Matrix2D, String> {
    check_idp_matrix_bounds(systems.len(), 2)?;
    Ok(idp_pair_matrix(systems, cfg))
}

/// Compute sequential dipeptide transition matrix `P` `(S x S)` where `P(s, t)` scores `s` at `k`
/// followed immediately by `t` at `k + 1` across TWO sequential pathways:
/// 1. `s.ca_curr` (`CA(k)`) matching `t.ca_prev` (`CA((k+1)-1)`), and
/// 2. `s.c_curr` (`C'(k)`) matching `t.c_prev` (`C'((k+1)-1)` root!).
///
/// Penalized by smooth bidirectional candidate-multiplicity ambiguity
/// `-gate(raw) * w_mult * ln(sqrt(n_out(s) * n_in(t)))`, matching `idp_scoring.wgsl` to `< 1e-5`.
pub fn idp_pair_matrix(systems: &[ConDipeptideSystem], cfg: &IdpSolverConfig) -> Matrix2D {
    let s_len = systems.len();
    if check_idp_matrix_bounds(s_len, 2).is_err() {
        return Matrix2D::filled(0, 0, 0.0);
    }
    let mut p = Matrix2D::filled(s_len, s_len, 0.0);
    if s_len == 0 {
        return p;
    }

    let sig_ca = cfg.sigma_seq_ca as f32;
    let sig_c = cfg.sigma_seq_c as f32;
    let p_floor = cfg.pair_floor as f32;
    let w_mult = (cfg.w_mult as f32).max(0.0);

    let ca_curr: Vec<Vec<f32>> = systems
        .iter()
        .map(|s| {
            s.ca_curr
                .iter()
                .take(MAX_IDP_CANDS)
                .map(|&x| x as f32)
                .collect()
        })
        .collect();
    let ca_prev: Vec<Vec<f32>> = systems
        .iter()
        .map(|s| {
            s.ca_prev
                .iter()
                .take(MAX_IDP_CANDS)
                .map(|&x| x as f32)
                .collect()
        })
        .collect();
    let c_curr: Vec<Vec<f32>> = systems
        .iter()
        .map(|s| {
            s.c_curr
                .iter()
                .take(MAX_IDP_CANDS)
                .map(|&x| x as f32)
                .collect()
        })
        .collect();
    let c_prev_root: Vec<[f32; 1]> = systems.iter().map(|s| [s.c_prev as f32]).collect();

    let mut raw_mat = vec![0.0f32; s_len * s_len];
    for s in 0..s_len {
        for t in 0..s_len {
            if s == t {
                continue;
            }
            let sc_ca = channel_pair_llr_f32(&ca_curr[s], &ca_prev[t], sig_ca, 28.0, p_floor);
            let sc_c = channel_pair_llr_f32(&c_curr[s], &c_prev_root[t], sig_c, 13.0, p_floor);
            // (H)N(CA)CON sequential N-N linking (`RD-13`): `t`'s strip shows `s.n_curr` (`N(i)`)
            let sc_nn = if !systems[t].nn_prev_cands.is_empty() {
                let sn = systems[s].n_curr;
                let tn = systems[t].n_curr;
                let mut best_nn: Option<f32> = None;
                for &(cand_n, snr) in &systems[t].nn_prev_cands {
                    let dn = (cand_n - sn).abs();
                    if dn < 0.25 {
                        // Smoothly down-weight near-diagonal (0.0..=0.35 ppm) candidates that may
                        // overlap the autopeak while retaining genuine N(i-1)≈N(i) repeat links.
                        let d_diag = (cand_n - tn).abs();
                        let diag_weight = if d_diag <= 0.35 {
                            0.35 + 0.65 * (d_diag / 0.35)
                        } else {
                            1.0
                        };
                        let bonus =
                            diag_weight * (2.0 + (snr.clamp(3.0, 12.0) / 4.0) * (1.0 - dn / 0.35));
                        if best_nn.is_none_or(|b| (bonus as f32) > b) {
                            best_nn = Some(bonus as f32);
                        }
                    }
                }
                let has_resolved_off_diag = systems[t]
                    .nn_prev_cands
                    .iter()
                    .any(|&(cn, _)| (cn - tn).abs() > 0.35);
                best_nn.unwrap_or(if has_resolved_off_diag { -0.6 } else { 0.0 })
            } else {
                0.0
            };
            // Optional reciprocal HN(CA)CONH N-N adjacency bonus (`RD-13`)
            let sc_recip = if !cfg.reciprocal_nn_pairs.is_empty() {
                let ns = systems[s].n_curr;
                let nt = systems[t].n_curr;
                let matched = cfg.reciprocal_nn_pairs.iter().any(|&(a, b)| {
                    ((a - ns).abs() < 0.25 && (b - nt).abs() < 0.35)
                        || ((a - nt).abs() < 0.25 && (b - ns).abs() < 0.35)
                });
                if matched {
                    1.5
                } else {
                    0.0
                }
            } else {
                0.0
            };
            raw_mat[s * s_len + t] = sc_ca + sc_c + sc_nn + sc_recip;
        }
    }

    let mut n_out = vec![1.0f32; s_len];
    let mut n_in = vec![1.0f32; s_len];
    if w_mult > 0.0 {
        for s in 0..s_len {
            let mut sum_out = 0.0f32;
            let mut sum_in = 0.0f32;
            for u in 0..s_len {
                if u != s {
                    sum_out += pair_compat_weight_f32(raw_mat[s * s_len + u]);
                    sum_in += pair_compat_weight_f32(raw_mat[u * s_len + s]);
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
            p.set(s, t, final_st as f64);
        }
    }

    p
}

// ============================================================================
// Multi-Island Evolutionary + Detailed-Balance Annealing & QC Calibration
// ============================================================================

/// Initialized IDP replica state `(at[0..K], pos[0..S], exact_score)`.
pub type IdpReplicaState = (Vec<i32>, Vec<i32>, f64);

/// Build valid sequence slot lists `slots[s]` (`k in 1..K` with `U(s, k) > IDP_FORBID_HALF`).
pub fn build_idp_valid_slots(u_mat: &Matrix2D) -> Vec<Vec<u32>> {
    let s_len = u_mat.rows;
    let k_len = u_mat.cols;
    let mut slots = Vec::with_capacity(s_len);
    for s in 0..s_len {
        let mut v = Vec::with_capacity(k_len);
        for k in 1..k_len {
            if u_mat.get(s, k) > (IDP_FORBID_HALF as f64) {
                v.push(k as u32);
            }
        }
        slots.push(v);
    }
    slots
}

/// Evaluate exact total objective $\sum_k U(\text{at}[k], k) + \sum_k P(\text{at}[k], \text{at}[k+1])$.
pub fn evaluate_idp_total_score(at: &[i32], u_mat: &Matrix2D, p_mat: &Matrix2D) -> f64 {
    let k_len = at.len();
    let mut sc = 0.0f64;
    for (k, &s) in at.iter().enumerate() {
        if s >= 0 {
            sc += u_mat.get(s as usize, k);
        }
    }
    for k in 0..k_len.saturating_sub(1) {
        let a = at[k];
        let b = at[k + 1];
        if a >= 0 && b >= 0 {
            sc += p_mat.get(a as usize, b as usize);
        }
    }
    sc
}

#[inline]
fn eval_1_slot(at: &[i32], u_mat: &Matrix2D, p_mat: &Matrix2D, k: usize) -> f64 {
    let s = at[k];
    if s < 0 {
        return 0.0;
    }
    let su = s as usize;
    let mut sc = u_mat.get(su, k);
    if k > 0 && at[k - 1] >= 0 {
        sc += p_mat.get(at[k - 1] as usize, su);
    }
    if k + 1 < at.len() && at[k + 1] >= 0 {
        sc += p_mat.get(su, at[k + 1] as usize);
    }
    sc
}

#[inline]
fn eval_2_slots(at: &[i32], u_mat: &Matrix2D, p_mat: &Matrix2D, ka: usize, kb: usize) -> f64 {
    let lo = ka.min(kb);
    let hi = ka.max(kb);
    let mut sc = 0.0f64;
    let s_lo = at[lo];
    let s_hi = at[hi];
    if s_lo >= 0 {
        let u_lo = s_lo as usize;
        sc += u_mat.get(u_lo, lo);
        if lo > 0 && at[lo - 1] >= 0 {
            sc += p_mat.get(at[lo - 1] as usize, u_lo);
        }
        if at[lo + 1] >= 0 {
            sc += p_mat.get(u_lo, at[lo + 1] as usize);
        }
    }
    if s_hi >= 0 {
        let u_hi = s_hi as usize;
        sc += u_mat.get(u_hi, hi);
        if hi > lo + 1 && at[hi - 1] >= 0 {
            sc += p_mat.get(at[hi - 1] as usize, u_hi);
        }
        if hi + 1 < at.len() && at[hi + 1] >= 0 {
            sc += p_mat.get(u_hi, at[hi + 1] as usize);
        }
    }
    sc
}

fn eval_block_slots(
    at: &[i32],
    u_mat: &Matrix2D,
    p_mat: &Matrix2D,
    k1: usize,
    k2: usize,
    len: usize,
) -> f64 {
    let b0 = k1.min(k2);
    let b1 = k1.max(k2);
    let k_len = at.len();
    let mut sc = 0.0f64;
    for i in 0..len {
        if at[b0 + i] >= 0 {
            sc += u_mat.get(at[b0 + i] as usize, b0 + i);
        }
        if at[b1 + i] >= 0 {
            sc += u_mat.get(at[b1 + i] as usize, b1 + i);
        }
    }
    if k_len > 1 {
        let max_e = k_len - 1;
        let e0_start = b0.saturating_sub(1);
        let e0_end = (b0 + len).min(max_e);
        for e in e0_start..e0_end {
            if at[e] >= 0 && at[e + 1] >= 0 {
                sc += p_mat.get(at[e] as usize, at[e + 1] as usize);
            }
        }
        let e1_start = b1.saturating_sub(1).max(e0_end);
        let e1_end = (b1 + len).min(max_e);
        for e in e1_start..e1_end {
            if at[e] >= 0 && at[e + 1] >= 0 {
                sc += p_mat.get(at[e] as usize, at[e + 1] as usize);
            }
        }
    }
    sc
}

/// Deterministic zero-temperature 2-opt swap + single-relocation + contiguous block-swap polish.
pub fn polish_idp_state(
    at: &mut [i32],
    pos: &mut [i32],
    u_mat: &Matrix2D,
    p_mat: &Matrix2D,
    slots: &[Vec<u32>],
) -> f64 {
    let s_len = pos.len();
    let k_len = at.len();
    for _pass in 0..3 {
        let mut improved = false;
        // 1. Single-root relocation / 2-swap
        for s in 0..s_len {
            for &ku in &slots[s] {
                let k = ku as usize;
                let ks = pos[s];
                if ks == k as i32 {
                    continue;
                }
                let t = at[k];
                if ks < 0 {
                    let before = eval_1_slot(at, u_mat, p_mat, k);
                    if t >= 0 {
                        pos[t as usize] = -1;
                    }
                    at[k] = s as i32;
                    pos[s] = k as i32;
                    let delta = eval_1_slot(at, u_mat, p_mat, k) - before;
                    if delta > 1e-6 {
                        improved = true;
                    } else {
                        pos[s] = -1;
                        at[k] = t;
                        if t >= 0 {
                            pos[t as usize] = k as i32;
                        }
                    }
                } else {
                    let ksu = ks as usize;
                    if t >= 0 && u_mat.get(t as usize, ksu) <= (IDP_FORBID_HALF as f64) {
                        continue;
                    }
                    let before = eval_2_slots(at, u_mat, p_mat, k, ksu);
                    at[k] = s as i32;
                    at[ksu] = t;
                    pos[s] = k as i32;
                    if t >= 0 {
                        pos[t as usize] = ksu as i32;
                    }
                    let delta = eval_2_slots(at, u_mat, p_mat, k, ksu) - before;
                    if delta > 1e-6 {
                        improved = true;
                    } else {
                        at[k] = t;
                        at[ksu] = s as i32;
                        pos[s] = ksu as i32;
                        if t >= 0 {
                            pos[t as usize] = k as i32;
                        }
                    }
                }
            }
            // Check unassigning spurious root if score improves
            let ks = pos[s];
            if ks >= 0 {
                let ksu = ks as usize;
                let before = eval_1_slot(at, u_mat, p_mat, ksu);
                if before < -1e-6 {
                    at[ksu] = -1;
                    pos[s] = -1;
                    improved = true;
                }
            }
        }

        // 2. Short contiguous motif swaps (`L = 2..=6`) to resolve repeated dipeptide/tripeptide motifs
        for blen in 2..=6usize.min(k_len / 2) {
            if k_len <= blen + 1 {
                continue;
            }
            for k1 in 1..=(k_len - blen) {
                for k2 in (k1 + blen)..=(k_len - blen) {
                    let mut valid = true;
                    for i in 0..blen {
                        let a = at[k1 + i];
                        let b = at[k2 + i];
                        if (b >= 0 && u_mat.get(b as usize, k1 + i) <= (IDP_FORBID_HALF as f64))
                            || (a >= 0 && u_mat.get(a as usize, k2 + i) <= (IDP_FORBID_HALF as f64))
                        {
                            valid = false;
                            break;
                        }
                    }
                    if !valid {
                        continue;
                    }
                    let before = eval_block_slots(at, u_mat, p_mat, k1, k2, blen);
                    for i in 0..blen {
                        let a = at[k1 + i];
                        let b = at[k2 + i];
                        at[k1 + i] = b;
                        at[k2 + i] = a;
                        if a >= 0 {
                            pos[a as usize] = (k2 + i) as i32;
                        }
                        if b >= 0 {
                            pos[b as usize] = (k1 + i) as i32;
                        }
                    }
                    let delta = eval_block_slots(at, u_mat, p_mat, k1, k2, blen) - before;
                    if delta > 1e-6 {
                        improved = true;
                    } else {
                        for i in 0..blen {
                            let b = at[k1 + i];
                            let a = at[k2 + i];
                            at[k1 + i] = a;
                            at[k2 + i] = b;
                            if a >= 0 {
                                pos[a as usize] = (k1 + i) as i32;
                            }
                            if b >= 0 {
                                pos[b as usize] = (k2 + i) as i32;
                            }
                        }
                    }
                }
            }
        }

        if !improved {
            break;
        }
    }
    evaluate_idp_total_score(at, u_mat, p_mat)
}

/// Build diverse initial replica states `(at, pos, score)` using randomized disjoint high-weight
/// sequential path covers on `P(s, t)` placed onto optimal sequence windows of `U(s, k)`.
/// Shared identically by `assign_idp_13c_cpu` and `assign_idp_13c_gpu`.
pub fn build_idp_initial_replicas(
    u_mat: &Matrix2D,
    p_mat: &Matrix2D,
    slots: &[Vec<u32>],
    replicas: usize,
    seed: u64,
) -> Vec<IdpReplicaState> {
    let s_len = u_mat.rows;
    let k_len = u_mat.cols;
    let mut master_rng = Rng::new(seed ^ 0x4944_505F_494E_4954);
    let mut out = Vec::with_capacity(replicas);

    // Collect strong sequential candidate edges `(weight, s, t)`
    let mut strong_edges: Vec<(f64, usize, usize)> = Vec::new();
    for s in 0..s_len {
        for t in 0..s_len {
            if s != t {
                let w = p_mat.get(s, t);
                if w >= 2.4 {
                    strong_edges.push((w, s, t));
                }
            }
        }
    }

    for rep in 0..replicas {
        let mut r_rng = Rng::new(master_rng.next_u64());
        let mut at = vec![-1i32; k_len];
        let mut pos = vec![-1i32; s_len];

        // Use greedy chain seeding on 75% of replicas and pure unary-guided seeding on 25% for diversity
        if rep % 4 != 3 && !strong_edges.is_empty() && k_len >= 3 {
            let mut noisy_edges: Vec<(f64, usize, usize)> = strong_edges
                .iter()
                .map(|&(w, s, t)| (w + r_rng.normal(0.0, 0.85), s, t))
                .collect();
            noisy_edges.sort_by(|a, b| b.0.total_cmp(&a.0));

            let mut next_of: Vec<Option<usize>> = vec![None; s_len];
            let mut prev_of: Vec<Option<usize>> = vec![None; s_len];
            let mut comp_id: Vec<usize> = (0..s_len).collect();

            for &(w, s, t) in &noisy_edges {
                if w < 2.2
                    || next_of[s].is_some()
                    || prev_of[t].is_some()
                    || comp_id[s] == comp_id[t]
                {
                    continue;
                }
                // Cap initial seed chain length at `14` so one wrong link never shifts a 60-aa block
                let mut len_s = 1usize;
                let mut cur = s;
                while let Some(p) = prev_of[cur] {
                    len_s += 1;
                    cur = p;
                }
                let mut len_t = 1usize;
                cur = t;
                while let Some(nx) = next_of[cur] {
                    len_t += 1;
                    cur = nx;
                }
                if len_s + len_t > 14 {
                    continue;
                }
                next_of[s] = Some(t);
                prev_of[t] = Some(s);
                let old_c = comp_id[t];
                let new_c = comp_id[s];
                for cid in &mut comp_id {
                    if *cid == old_c {
                        *cid = new_c;
                    }
                }
            }

            // Extract chains of length >= 2, longest first
            let mut chains: Vec<Vec<usize>> = Vec::new();
            for s in 0..s_len {
                if prev_of[s].is_none() && next_of[s].is_some() {
                    let mut ch = vec![s];
                    let mut cur = s;
                    while let Some(nx) = next_of[cur] {
                        ch.push(nx);
                        cur = nx;
                    }
                    chains.push(ch);
                }
            }
            chains.sort_by_key(|b| std::cmp::Reverse(b.len()));

            for ch in &chains {
                let clen = ch.len();
                if clen == 0 || k_len <= clen {
                    continue;
                }
                let mut best_k0: Option<usize> = None;
                let mut best_sc = f64::NEG_INFINITY;
                for k0 in 1..=(k_len - clen) {
                    let mut ok = true;
                    let mut u_sum = 0.0f64;
                    for (m, &sm) in ch.iter().enumerate() {
                        if at[k0 + m] >= 0 {
                            ok = false;
                            break;
                        }
                        let u = u_mat.get(sm, k0 + m);
                        if u <= (IDP_FORBID_HALF as f64) {
                            ok = false;
                            break;
                        }
                        u_sum += u;
                    }
                    if ok {
                        let noisy = u_sum + r_rng.normal(0.0, 1.4 * (clen as f64).sqrt());
                        if noisy > best_sc {
                            best_sc = noisy;
                            best_k0 = Some(k0);
                        }
                    }
                }
                if let Some(k0) = best_k0 {
                    for (m, &sm) in ch.iter().enumerate() {
                        at[k0 + m] = sm as i32;
                        pos[sm] = (k0 + m) as i32;
                    }
                }
            }
        }

        // Fill remaining unassigned roots in randomized order
        let mut rem: Vec<(f64, usize)> = (0..s_len)
            .filter(|&s| pos[s] < 0)
            .map(|s| ((slots[s].len() as f64) + r_rng.f64(), s))
            .collect();
        rem.sort_by(|a, b| a.0.total_cmp(&b.0));

        for &(_, s) in &rem {
            let mut best_k: Option<usize> = None;
            let mut best_val = -0.5f64;
            for &ku in &slots[s] {
                let k = ku as usize;
                if at[k] >= 0 {
                    continue;
                }
                let mut sc = u_mat.get(s, k);
                if k > 1 && at[k - 1] >= 0 {
                    sc += p_mat.get(at[k - 1] as usize, s);
                }
                if k + 1 < k_len && at[k + 1] >= 0 {
                    sc += p_mat.get(s, at[k + 1] as usize);
                }
                let noisy = sc + r_rng.normal(0.0, 1.5);
                if noisy > best_val {
                    best_val = noisy;
                    best_k = Some(k);
                }
            }
            if let Some(k) = best_k {
                at[k] = s as i32;
                pos[s] = k as i32;
            }
        }

        let sc = evaluate_idp_total_score(&at, u_mat, p_mat);
        out.push((at, pos, sc));
    }

    out
}

/// Run exact detailed-balance Metropolis–Hastings simulated annealing on a single CPU replica.
#[allow(clippy::too_many_arguments)]
fn run_idp_cpu_anneal_segment(
    state: &mut IdpReplicaState,
    u_mat: &Matrix2D,
    p_mat: &Matrix2D,
    slots: &[Vec<u32>],
    steps: usize,
    t_start: f64,
    t_end: f64,
    rng: &mut Rng,
) -> (Vec<i32>, Vec<i32>, f64) {
    let s_len = state.1.len();
    let k_len = state.0.len();
    let at = &mut state.0;
    let pos = &mut state.1;
    let mut cur = evaluate_idp_total_score(at, u_mat, p_mat);
    let mut best_sc = cur;
    let mut best_at = at.clone();
    let mut best_pos = pos.clone();

    if s_len == 0 || k_len < 2 || steps == 0 {
        state.2 = best_sc;
        return (best_at, best_pos, best_sc);
    }

    let denom = (steps.saturating_sub(1)).max(1) as f64;
    let log_ratio = (t_end.max(1e-4) / t_start.max(1e-3)).ln();
    let max_blk = 16usize.min((k_len - 1) / 2);

    for step in 0..steps {
        let temp = t_start * (log_ratio * (step as f64 / denom)).exp();
        let draw = rng.f64();

        if draw < 0.70 || max_blk == 0 {
            // Reversible single-root move over `slots[s] U {-1}` (exact detailed balance!)
            let s = rng.usize(s_len);
            let sl_s = slots[s].len();
            if sl_s == 0 {
                continue;
            }
            let opt = rng.usize(sl_s + 1);
            let ks = pos[s];

            if opt == sl_s {
                // Propose unassign `s -> -1`
                if ks < 0 {
                    continue;
                }
                let ksu = ks as usize;
                let before = eval_1_slot(at, u_mat, p_mat, ksu);
                at[ksu] = -1;
                pos[s] = -1;
                let delta = -before;
                if delta >= 0.0 || rng.f64() < (delta / temp).exp() {
                    cur += delta;
                    if cur > best_sc {
                        best_sc = cur;
                        best_at.copy_from_slice(at);
                        best_pos.copy_from_slice(pos);
                    }
                } else {
                    at[ksu] = s as i32;
                    pos[s] = ks;
                }
            } else {
                let k = slots[s][opt] as usize;
                if ks == k as i32 {
                    continue;
                }
                let t = at[k];
                if ks < 0 {
                    let before = eval_1_slot(at, u_mat, p_mat, k);
                    let hastings_log = if t >= 0 {
                        ((sl_s + 1) as f64 / (slots[t as usize].len() + 1) as f64).ln()
                    } else {
                        0.0
                    };
                    if t >= 0 {
                        pos[t as usize] = -1;
                    }
                    at[k] = s as i32;
                    pos[s] = k as i32;
                    let delta = eval_1_slot(at, u_mat, p_mat, k) - before;
                    let log_acc = (delta / temp) + hastings_log;
                    if log_acc >= 0.0 || rng.f64() < log_acc.exp() {
                        cur += delta;
                        if cur > best_sc {
                            best_sc = cur;
                            best_at.copy_from_slice(at);
                            best_pos.copy_from_slice(pos);
                        }
                    } else {
                        pos[s] = -1;
                        at[k] = t;
                        if t >= 0 {
                            pos[t as usize] = k as i32;
                        }
                    }
                } else {
                    let ksu = ks as usize;
                    // Pure reversible 2-swap: reject if `t` is forbidden at `ksu`
                    if t >= 0 && u_mat.get(t as usize, ksu) <= (IDP_FORBID_HALF as f64) {
                        continue;
                    }
                    let before = eval_2_slots(at, u_mat, p_mat, k, ksu);
                    at[k] = s as i32;
                    at[ksu] = t;
                    pos[s] = k as i32;
                    if t >= 0 {
                        pos[t as usize] = ksu as i32;
                    }
                    let delta = eval_2_slots(at, u_mat, p_mat, k, ksu) - before;
                    if delta >= 0.0 || rng.f64() < (delta / temp).exp() {
                        cur += delta;
                        if cur > best_sc {
                            best_sc = cur;
                            best_at.copy_from_slice(at);
                            best_pos.copy_from_slice(pos);
                        }
                    } else {
                        at[k] = t;
                        at[ksu] = s as i32;
                        pos[s] = ksu as i32;
                        if t >= 0 {
                            pos[t as usize] = k as i32;
                        }
                    }
                }
            }
        } else {
            // Equal-length non-overlapping block swap on `1..k_len` (symmetric proposal!)
            let l = 1 + rng.usize(max_blk);
            let span = (k_len - 1) - l + 1;
            if span < 2 {
                continue;
            }
            let k1 = 1 + rng.usize(span);
            let k2 = 1 + rng.usize(span);
            if k1.abs_diff(k2) < l {
                continue;
            }
            let mut valid = true;
            for i in 0..l {
                let a = at[k1 + i];
                let b = at[k2 + i];
                if (b >= 0 && u_mat.get(b as usize, k1 + i) <= (IDP_FORBID_HALF as f64))
                    || (a >= 0 && u_mat.get(a as usize, k2 + i) <= (IDP_FORBID_HALF as f64))
                {
                    valid = false;
                    break;
                }
            }
            if !valid {
                continue;
            }
            let before = eval_block_slots(at, u_mat, p_mat, k1, k2, l);
            for i in 0..l {
                let a = at[k1 + i];
                let b = at[k2 + i];
                at[k1 + i] = b;
                at[k2 + i] = a;
                if a >= 0 {
                    pos[a as usize] = (k2 + i) as i32;
                }
                if b >= 0 {
                    pos[b as usize] = (k1 + i) as i32;
                }
            }
            let delta = eval_block_slots(at, u_mat, p_mat, k1, k2, l) - before;
            if delta >= 0.0 || rng.f64() < (delta / temp).exp() {
                cur += delta;
                if cur > best_sc {
                    best_sc = cur;
                    best_at.copy_from_slice(at);
                    best_pos.copy_from_slice(pos);
                }
            } else {
                for i in 0..l {
                    let b = at[k1 + i];
                    let a = at[k2 + i];
                    at[k1 + i] = a;
                    at[k2 + i] = b;
                    if a >= 0 {
                        pos[a as usize] = (k1 + i) as i32;
                    }
                    if b >= 0 {
                        pos[b as usize] = (k2 + i) as i32;
                    }
                }
            }
        }
    }

    state.2 = cur;
    (best_at, best_pos, best_sc)
}

/// Evolutionary conflict-free segment crossover injecting `donor.0[k_lo..=k_hi]` into `recipient`
/// and repairing any displaced roots greedily.
pub fn crossover_idp_states(
    recipient: &IdpReplicaState,
    donor: &IdpReplicaState,
    u_mat: &Matrix2D,
    p_mat: &Matrix2D,
    slots: &[Vec<u32>],
    rng: &mut Rng,
) -> IdpReplicaState {
    let k_len = recipient.0.len();
    let s_len = recipient.1.len();
    if k_len < 5 || s_len == 0 {
        return recipient.clone();
    }
    let mut at = recipient.0.clone();
    let mut pos = recipient.1.clone();

    let seg_len = (k_len / 5).clamp(3, 28).min(k_len - 1);
    let k_lo = 1 + rng.usize((k_len - seg_len).max(1));
    let k_hi = (k_lo + seg_len - 1).min(k_len - 1);

    // 1. Splice donor segment `[k_lo..=k_hi]`, clearing any conflicting placements cleanly
    for slot_val in &mut at[k_lo..=k_hi] {
        let old_s = *slot_val;
        if old_s >= 0 {
            pos[old_s as usize] = -1;
            *slot_val = -1;
        }
    }
    for k in k_lo..=k_hi {
        let don_s = donor.0[k];
        if don_s >= 0 && u_mat.get(don_s as usize, k) > (IDP_FORBID_HALF as f64) {
            let prev_k = pos[don_s as usize];
            if prev_k >= 0 {
                at[prev_k as usize] = -1;
            }
            at[k] = don_s;
            pos[don_s as usize] = k as i32;
        }
    }

    // 2. Re-insert displaced unassigned roots into best available slots
    for s in 0..s_len {
        if pos[s] >= 0 {
            continue;
        }
        let mut best_k: Option<usize> = None;
        let mut best_v = -0.25f64;
        for &ku in &slots[s] {
            let k = ku as usize;
            if at[k] >= 0 {
                continue;
            }
            let mut v = u_mat.get(s, k);
            if k > 1 && at[k - 1] >= 0 {
                v += p_mat.get(at[k - 1] as usize, s);
            }
            if k + 1 < k_len && at[k + 1] >= 0 {
                v += p_mat.get(s, at[k + 1] as usize);
            }
            if v > best_v {
                best_v = v;
                best_k = Some(k);
            }
        }
        if let Some(k) = best_k {
            at[k] = s as i32;
            pos[s] = k as i32;
        }
    }

    let sc = evaluate_idp_total_score(&at, u_mat, p_mat);
    (at, pos, sc)
}

/// Monotone piecewise-linear posterior calibration map for `13C`-detected IDP assignments.
///
/// Provenance (`Phase 12`, `P1-7` / `Q3` non-circular Leave-One-Protein-Out calibration):
/// - Fitted reproducibly via `evaluate_idp_lopo_calibration` / `fit_idp_calibration_knots_pav`
///   across 5 disjoint real BMRB IDP depositions (`bmr6968` alpha-synuclein 140 aa,
///   `bmr15176` DARPP-32 119 aa, `bmr4375` 8M-urea Ubiquitin 76 aa, `bmr26639` CBP-ID4 207 aa,
///   and `bmr51984` p53-TAD-PRR 102 aa) spanning **both** baseline (`1x`, `sigma_seq ~ 0.028 ppm CA`,
///   8% spurious) and `3x` realistic-noise (`sigma_seq ~ 0.065–0.085 ppm CA`, 16% equal-height
///   spurious 3D peaks) regimes (`N = 2,402` out-of-fold assigned spin systems; locked by
///   `test_idp_5fold_leave_one_protein_out_calibration_across_low_and_3x_realistic_noise`).
/// - Exact 5-fold Leave-One-Protein-Out out-of-fold calibration metrics (`raw_posterior`):
///   * Pooled 5-fold OOF `ECE = 0.0104` (`<= 0.055` gate),
///   * Per-protein OOF `ECE = [0.0171, 0.0300, 0.0495, 0.0129, 0.0149]`.
/// - Hyperparameter provenance:
///   * `TAU_ENS = 6.5`: multi-island Boltzmann ensemble consensus temperature (~2 multi-channel
///     LLR standard deviations across polished island elites).
///   * `TAU_LOCAL = 1.95` (scaled adaptively by empirical sequential `CA` RMS, per-channel
///     candidate multiplicity, and missing-`CCC(CO)N` sidechain degeneracy under realistic noise):
///     single-swap, `>= 3`-cycle relocation, and motif-swap Boltzmann temperature minimizing
///     5-fold LOPO out-of-fold log-loss & ECE.
///   * `MAX_ALT_COMPETITORS = 8`: top-8 single-slot / 2-swap competitors retained in the local
///     partition sum so `p_local` is invariant to sequence length `K` (`76..207+ aa`) rather than
///     diluted by `O(K)` incompatible random-coil tail slots (`exp(-12) * 200`).
///   * `sys_assign_bonus = +2.65` (with `>= 1` 3D channel) vs `-6.5` (3D-unsupported 2D thermal
///     noise root when `>= 50%` of roots have 3D support): log-prior odds `ln(P_true / P_spur)`
///     from 3D relay survival (`>= 99%` when `1 - 0.10^3 = 0.999` of true peptide bonds carry
///     at least one `CANCO`/`CBCACON`/`NCACO` peak vs `< 0.2%` of 3D-empty thermal spikes).
pub const IDP_CALIBRATION_KNOTS: &[(f64, f64)] = &[
    (0.00, 0.0000),
    (0.10, 0.2637),
    (0.25, 0.4478),
    (0.45, 0.6262),
    (0.65, 0.8340),
    (0.80, 0.9584),
    (0.90, 0.9856),
    (0.96, 0.9916),
    (1.00, 0.9940),
];

#[inline]
fn eval_idp_calibration(p_raw: f64) -> f64 {
    crate::qc::eval_monotone_calibration_map(p_raw, IDP_CALIBRATION_KNOTS)
}

/// Invert [`IDP_CALIBRATION_KNOTS`] on `[0, 1]` to recover the pre-calibration raw posterior `p_raw`
/// for strictly non-circular Leave-One-Protein-Out (LOPO) cross-validation.
pub fn invert_idp_calibration(p_cal: f64) -> f64 {
    let y = p_cal.clamp(0.0, 1.0);
    if y <= IDP_CALIBRATION_KNOTS[0].1 {
        return IDP_CALIBRATION_KNOTS[0].0;
    }
    for w in IDP_CALIBRATION_KNOTS.windows(2) {
        let (x0, y0) = w[0];
        let (x1, y1) = w[1];
        if y <= y1 {
            let dy = y1 - y0;
            if dy <= 1e-12 {
                return x0;
            }
            return (x0 + (y - y0) / dy * (x1 - x0)).clamp(0.0, 1.0);
        }
    }
    1.0
}

/// Deterministic Pool-Adjacent-Violators (PAV) isotonic calibration knot fitter mapping raw
/// posteriors `p_raw in [0, 1]` onto `knot_xs` (`0.0 ..= 1.0`).
pub fn fit_idp_calibration_knots_pav(samples: &[(f64, bool)], knot_xs: &[f64]) -> Vec<(f64, f64)> {
    if knot_xs.is_empty() {
        return vec![(0.0, 0.0), (1.0, 1.0)];
    }
    if samples.is_empty() {
        return knot_xs.iter().map(|&x| (x, x.clamp(0.0, 1.0))).collect();
    }

    let mut sorted: Vec<(f64, f64)> = samples
        .iter()
        .map(|&(p, ok)| (p.clamp(0.0, 1.0), if ok { 1.0 } else { 0.0 }))
        .collect();
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.total_cmp(&b.1)));

    // Pool-Adjacent-Violators (Ayer et al. 1955) with weak identity prior regularization
    // `(x_lo, x_hi, sum_y, weight)`
    let mut blocks: Vec<(f64, f64, f64, f64)> = Vec::with_capacity(sorted.len() + 2);
    blocks.push((0.0, 0.0, 0.0, 2.0));
    for (x, y) in sorted {
        blocks.push((x, x, y + 0.08 * x, 1.08));
        while blocks.len() >= 2 {
            let n = blocks.len();
            let mean_prev = blocks[n - 2].2 / blocks[n - 2].3;
            let mean_curr = blocks[n - 1].2 / blocks[n - 1].3;
            if mean_prev > mean_curr {
                let last = blocks.pop().unwrap();
                let prev = blocks.last_mut().unwrap();
                prev.1 = last.1;
                prev.2 += last.2;
                prev.3 += last.3;
            } else {
                break;
            }
        }
    }

    // Evaluate smoothed isotonic step curve around each target knot with local Gaussian kernel
    let mut knots = Vec::with_capacity(knot_xs.len());
    let mut prev_y = 0.0f64;
    for (idx, &xk_in) in knot_xs.iter().enumerate() {
        let xk = xk_in.clamp(0.0, 1.0);
        if idx == 0 && xk == 0.0 {
            knots.push((0.0, 0.0));
            continue;
        }
        let bw = 0.085f64;
        let mut num = 1.5 * xk;
        let mut den = 1.5f64;
        for &(x0, x1, sum_y, wt) in &blocks {
            let xm = 0.5 * (x0 + x1);
            let mean_y = sum_y / wt;
            let z = (xm - xk) / bw;
            let kw = wt * (-0.5 * z * z).exp();
            num += kw * mean_y;
            den += kw;
        }
        let y_est = (num / den).clamp(prev_y, 0.996);
        prev_y = y_est;
        knots.push((xk, y_est));
    }
    knots
}

/// Run exact K-fold Leave-One-Protein-Out (LOPO) cross-validation over per-protein `(p_raw, correct)`
/// sample vectors, returning `(full_fitted_knots, pooled_oof_ece, per_fold_oof_ece)`.
pub fn evaluate_idp_lopo_calibration(
    per_protein_raw_samples: &[Vec<(f64, bool)>],
    knot_xs: &[f64],
    hc_threshold: f64,
) -> (Vec<(f64, f64)>, f64, Vec<f64>) {
    let n_folds = per_protein_raw_samples.len();
    let mut oof_pooled: Vec<(f64, bool)> = Vec::new();
    let mut per_fold_ece = Vec::with_capacity(n_folds);

    for holdout in 0..n_folds {
        let mut train: Vec<(f64, bool)> = Vec::new();
        for (i, fold) in per_protein_raw_samples.iter().enumerate() {
            if i != holdout {
                train.extend_from_slice(fold);
            }
        }
        let fold_knots = fit_idp_calibration_knots_pav(&train, knot_xs);
        let mapped: Vec<(f64, bool)> = per_protein_raw_samples[holdout]
            .iter()
            .map(|&(p_raw, ok)| {
                (
                    crate::qc::eval_monotone_calibration_map(p_raw, &fold_knots),
                    ok,
                )
            })
            .collect();
        let curve = crate::qc::ProteinReliabilityCurve::from_predictions(&mapped, hc_threshold);
        per_fold_ece.push(curve.ece);
        oof_pooled.extend(mapped);
    }

    let mut all_samples: Vec<(f64, bool)> = Vec::new();
    for fold in per_protein_raw_samples {
        all_samples.extend_from_slice(fold);
    }
    let full_knots = fit_idp_calibration_knots_pav(&all_samples, knot_xs);
    let pooled_curve =
        crate::qc::ProteinReliabilityCurve::from_predictions(&oof_pooled, hc_threshold);
    (full_knots, pooled_curve.ece, per_fold_ece)
}

fn idp_aa_index(aa: &str) -> usize {
    match aa.as_str_or_upper() {
        "ALA" => 0,
        "ARG" => 1,
        "ASN" => 2,
        "ASP" => 3,
        "CYS" => 4,
        "GLN" => 5,
        "GLU" => 6,
        "GLY" => 7,
        "HIS" => 8,
        "ILE" => 9,
        "LEU" => 10,
        "LYS" => 11,
        "MET" => 12,
        "PHE" => 13,
        "PRO" => 14,
        "SER" => 15,
        "THR" => 16,
        "TRP" => 17,
        "TYR" => 18,
        "VAL" => 19,
        _ => 20,
    }
}

trait StrUpperHelper {
    fn as_str_or_upper(&self) -> &str;
}
impl StrUpperHelper for str {
    fn as_str_or_upper(&self) -> &str {
        self
    }
}

/// Compute primary-sequence Shannon AA entropy (bits), max single-AA fraction, max single-dipeptide
/// fraction, and top-3 cumulative dipeptide repeat fraction for IDP low-complexity QC gating.
pub fn sequence_complexity_metrics(sequence: &[String]) -> (f64, f64, f64, f64) {
    let norm = normalize_seq_3(sequence);
    let k_len = norm.len();
    if k_len == 0 {
        return (0.0, 1.0, 1.0, 1.0);
    }
    let mut aa_counts = [0usize; 21];
    for r in &norm {
        aa_counts[idp_aa_index(r)] += 1;
    }
    let inv_n = 1.0 / (k_len as f64);
    let mut entropy_bits = 0.0f64;
    let mut max_aa = 0.0f64;
    for &c in &aa_counts {
        if c > 0 {
            let p = (c as f64) * inv_n;
            entropy_bits -= p * p.log2();
            if p > max_aa {
                max_aa = p;
            }
        }
    }

    if k_len < 2 {
        return (entropy_bits, max_aa, 0.0, 0.0);
    }
    let n_dip = k_len - 1;
    let mut dip_counts: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    for k in 1..k_len {
        let a = idp_aa_index(&norm[k - 1]);
        let b = idp_aa_index(&norm[k]);
        *dip_counts.entry((a, b)).or_insert(0) += 1;
    }
    let mut freqs: Vec<usize> = dip_counts.into_values().collect();
    freqs.sort_unstable_by(|a, b| b.cmp(a));
    let max_dip = (freqs.first().copied().unwrap_or(0) as f64) / (n_dip as f64);
    let top3_dip = (freqs.iter().take(3).sum::<usize>() as f64) / (n_dip as f64);
    (entropy_bits, max_aa, max_dip, top3_dip)
}

/// Combine completed CPU or WebGPU replica assignments via Boltzmann consensus, exact local-swap /
/// motif-swap specificity calibration, and unlabelled `DatasetQcVerdict` (`Pass`, `Warn`, `Abstain`).
#[allow(clippy::too_many_arguments)]
pub fn consensus_and_calibrate_idp(
    systems: &[ConDipeptideSystem],
    sequence: &[String],
    u_mat: &Matrix2D,
    p_mat: &Matrix2D,
    slots: &[Vec<u32>],
    replica_positions: &[Vec<i32>],
    replica_scores: &[f64],
    fitted_offsets: IdpShiftOffsets,
    cfg: &IdpSolverConfig,
) -> IdpAssignResult {
    let s_len = systems.len();
    let k_len = sequence.len();
    if s_len == 0 || k_len < 2 || replica_positions.is_empty() {
        return IdpAssignResult {
            position: vec![-1; s_len],
            confidence: vec![0.0; s_len],
            raw_posterior: vec![0.0; s_len],
            score: 0.0,
            run_scores: replica_scores.to_vec(),
            fitted_offsets,
            qc_verdict: DatasetQcVerdict::Abstain,
            support_3d_frac: 0.0,
            connectivity_frac: 0.0,
            qc_reasons: vec!["Insufficient spin systems or sequence length (< 2)".to_string()],
        };
    }

    // Polish top replicas cleanly and rank by exact global score
    let mut polished: Vec<(Vec<i32>, Vec<i32>, f64)> = Vec::with_capacity(replica_positions.len());
    for pos_in in replica_positions {
        let mut at = vec![-1i32; k_len];
        let mut pos = vec![-1i32; s_len];
        for (s, &k) in pos_in.iter().enumerate().take(s_len) {
            if k >= 1 && (k as usize) < k_len && at[k as usize] < 0 {
                at[k as usize] = s as i32;
                pos[s] = k;
            }
        }
        let sc = polish_idp_state(&mut at, &mut pos, u_mat, p_mat, slots);
        polished.push((at, pos, sc));
    }

    let run_scores: Vec<f64> = polished.iter().map(|r| r.2).collect();
    let mut ranked: Vec<usize> = (0..polished.len()).collect();
    ranked.sort_by(|&a, &b| polished[b].2.total_cmp(&polished[a].2));

    let top_n = (polished.len() / 2).max(4).min(polished.len());
    let max_sc = polished[ranked[0]].2;
    const TAU_ENS: f64 = 6.5;
    let mut ens_weights = Vec::with_capacity(top_n);
    let mut w_sum = 0.0f64;
    for &idx in &ranked[..top_n] {
        let w = ((polished[idx].2 - max_sc) / TAU_ENS).max(-30.0).exp();
        ens_weights.push((idx, w));
        w_sum += w;
    }
    let w_sum = w_sum.max(1e-12);

    // 1. Conflict-free consensus injection from Boltzmann-weighted replica marginals
    let mut cand_pairs: Vec<(f64, usize, i32)> = Vec::new();
    let mut unassign_p = vec![0.0f64; s_len];
    for (s, unassign_slot) in unassign_p.iter_mut().enumerate() {
        let mut counts: BTreeMap<i32, f64> = BTreeMap::new();
        for &(r_idx, w) in &ens_weights {
            *counts.entry(polished[r_idx].1[s]).or_insert(0.0) += w;
        }
        for (k, wt) in counts {
            let p = wt / w_sum;
            if k < 0 {
                *unassign_slot = p;
            } else {
                cand_pairs.push((p, s, k));
            }
        }
    }
    cand_pairs.sort_by(|a, b| b.0.total_cmp(&a.0));

    let mut cons_pos = vec![-1i32; s_len];
    let mut cons_at = vec![-1i32; k_len];
    for &(p, s, k) in &cand_pairs {
        if cons_pos[s] < 0 && cons_at[k as usize] < 0 && p >= unassign_p[s] {
            cons_pos[s] = k;
            cons_at[k as usize] = s as i32;
        }
    }
    let mut final_sc = polish_idp_state(&mut cons_at, &mut cons_pos, u_mat, p_mat, slots);
    if max_sc > final_sc + 0.25 {
        cons_at.copy_from_slice(&polished[ranked[0]].0);
        cons_pos.copy_from_slice(&polished[ranked[0]].1);
        final_sc = max_sc;
    }

    // Recompute replica consensus support `p_ens[s]` on final `cons_pos[s]`
    let mut p_ens = vec![0.0f64; s_len];
    for s in 0..s_len {
        let k = cons_pos[s];
        if k >= 0 {
            let mut wt = 0.0f64;
            for &(r_idx, w) in &ens_weights {
                if polished[r_idx].1[s] == k {
                    wt += w;
                }
            }
            p_ens[s] = wt / w_sum;
        }
    }

    // Estimate empirical unlabelled noise & candidate-multiplicity diagnostics BEFORE local
    // Boltzmann specificity so `TAU_LOCAL` scales honestly when matching noise or spurious peak
    // density is elevated (preventing overconfident `ECE` inflation under `3x` realistic noise!).
    let mut ch_cand_sum = 0usize;
    let mut ch_nonempty = 0usize;
    let mut prev_cand_sum = 0usize;
    let mut prev_nonempty = 0usize;
    let mut margin_eval_2d_roots = 0usize;
    let mut compat_2d_out_sum = 0.0f64;

    for (s, sys) in systems.iter().enumerate() {
        for ch in [&sys.ca_curr, &sys.ca_prev, &sys.cb_prev, &sys.c_curr] {
            if !ch.is_empty() {
                ch_nonempty += 1;
                ch_cand_sum += ch.len().min(MAX_IDP_CANDS);
            }
        }
        for ch in [&sys.ca_prev, &sys.cb_prev] {
            if !ch.is_empty() {
                prev_nonempty += 1;
                prev_cand_sum += ch.len().min(MAX_IDP_CANDS);
            }
        }
        if !sys.ca_curr.is_empty() && !sys.c_curr.is_empty() && s_len >= 2 {
            let mut compat_s = 0.0f64;
            for t in 0..s_len {
                if t != s {
                    let pv = p_mat.get(s, t);
                    if pv > 2.60 {
                        let u = ((pv - 2.60) / 2.40).clamp(0.0, 1.0);
                        compat_s += u * u * (3.0 - 2.0 * u);
                    }
                }
            }
            compat_2d_out_sum += compat_s;
            margin_eval_2d_roots += 1;
        }
    }
    let mean_cands_per_ch = if ch_nonempty > 0 {
        (ch_cand_sum as f64) / (ch_nonempty as f64)
    } else {
        1.0
    };
    let mean_prev_cands = if prev_nonempty > 0 {
        (prev_cand_sum as f64) / (prev_nonempty as f64)
    } else {
        1.0
    };

    let mut sq_ca_diffs: Vec<f64> = Vec::new();
    for k in 1..k_len.saturating_sub(1) {
        let sa = cons_at[k];
        let sb = cons_at[k + 1];
        if sa >= 0 && sb >= 0 {
            let sys_a = &systems[sa as usize];
            let sys_b = &systems[sb as usize];
            if !sys_a.ca_curr.is_empty() && !sys_b.ca_prev.is_empty() {
                let mut best_df = f64::INFINITY;
                for &va in sys_a.ca_curr.iter().take(MAX_IDP_CANDS) {
                    for &vb in sys_b.ca_prev.iter().take(MAX_IDP_CANDS) {
                        let df = (va - vb).abs();
                        if df < best_df {
                            best_df = df;
                        }
                    }
                }
                if best_df <= 0.35 {
                    sq_ca_diffs.push(best_df * best_df);
                }
            }
        }
    }
    let emp_rms_seq_ca = if sq_ca_diffs.len() >= 6 {
        (sq_ca_diffs.iter().sum::<f64>() / (sq_ca_diffs.len() as f64)).sqrt()
    } else {
        0.0
    };

    let (h_aa_bits, max_aa_frac, max_dipep_frac, top3_dipep_frac) =
        sequence_complexity_metrics(sequence);
    let seq_low_comp_warn =
        k_len >= 16 && (h_aa_bits < 2.60 || max_aa_frac >= 0.35 || top3_dipep_frac >= 0.35);

    let mut n_assigned = 0usize;
    let mut n_connected = 0usize;
    for (s, &k_i) in cons_pos.iter().enumerate() {
        if k_i >= 1 {
            n_assigned += 1;
            let k = k_i as usize;
            let prev_ok =
                k > 1 && cons_at[k - 1] >= 0 && p_mat.get(cons_at[k - 1] as usize, s) >= 1.0;
            let next_ok = k + 1 < k_len
                && cons_at[k + 1] >= 0
                && p_mat.get(s, cons_at[k + 1] as usize) >= 1.0;
            if prev_ok || next_ok {
                n_connected += 1;
            }
        }
    }
    let connectivity_frac = if n_assigned > 0 {
        (n_connected as f64) / (n_assigned as f64)
    } else {
        0.0
    };

    // 2. Length-invariant local-swap + multi-way cycle + motif-swap Boltzmann specificity `p_local[s]`
    const TAU_LOCAL: f64 = 1.95;
    const MAX_ALT_COMPETITORS: usize = 8;
    let any_cccon = systems.iter().any(|s| !s.sidechain_c_prev.is_empty());
    let walk_pair_weight = if any_cccon { 0.76 } else { 0.66 };
    let walk_base_margin = if any_cccon { 1.95 } else { 1.55 };

    let mut work_at = cons_at.clone();
    let mut single_deltas = Vec::with_capacity(k_len);
    let mut cycle_deltas = Vec::with_capacity(k_len);
    let mut sys_base_1 = vec![0.0f64; s_len];
    let mut sys_top_deltas: Vec<Vec<f64>> = vec![Vec::new(); s_len];
    let mut sys_top_cycle_deltas: Vec<Vec<f64>> = vec![Vec::new(); s_len];
    let mut sys_best_blk = vec![f64::NEG_INFINITY; s_len];
    let mut n_ambig_placements = 0usize;

    for s in 0..s_len {
        let k_i = cons_pos[s];
        if k_i < 0 {
            continue;
        }
        let k_star = k_i as usize;
        sys_base_1[s] = eval_1_slot(&work_at, u_mat, p_mat, k_star);

        let pair_prev = if k_star > 1 && cons_at[k_star - 1] >= 0 {
            p_mat.get(cons_at[k_star - 1] as usize, s).max(0.0)
        } else {
            0.0
        };
        let pair_next = if k_star + 1 < k_len && cons_at[k_star + 1] >= 0 {
            p_mat.get(s, cons_at[k_star + 1] as usize).max(0.0)
        } else {
            0.0
        };
        let u_star = u_mat.get(s, k_star);
        // Multi-way (`>= 3`-cycle) relocation bound: a 2-swap `k_star <-> k_alt` freezes the
        // occupant `t = work_at[k_alt]` into `k_star` even when the true error is a `>= 3`-cycle
        // `s -> k_alt -> k_2 -> k_star`; `cycle_deltas` measures unary preference for `k_alt`
        // minus the sequential walk support anchoring `s` at `k_star`.
        let walk_anchor = walk_pair_weight * (pair_prev + pair_next) + walk_base_margin;

        single_deltas.clear();
        cycle_deltas.clear();
        for &k_alt_u in &slots[s] {
            let k_alt = k_alt_u as usize;
            if k_alt == k_star {
                continue;
            }
            cycle_deltas.push((u_mat.get(s, k_alt) - u_star) - walk_anchor);
            let t = work_at[k_alt];
            if t >= 0 && u_mat.get(t as usize, k_star) <= (IDP_FORBID_HALF as f64) {
                continue;
            }
            let before = eval_2_slots(&work_at, u_mat, p_mat, k_star, k_alt);
            work_at[k_star] = t;
            work_at[k_alt] = s as i32;
            let delta = eval_2_slots(&work_at, u_mat, p_mat, k_star, k_alt) - before;
            work_at[k_star] = s as i32;
            work_at[k_alt] = t;
            single_deltas.push(delta);
        }
        single_deltas.sort_by(|a, b| b.total_cmp(a));
        sys_top_deltas[s].extend(single_deltas.iter().copied().take(MAX_ALT_COMPETITORS));
        cycle_deltas.sort_by(|a, b| b.total_cmp(a));
        sys_top_cycle_deltas[s].extend(cycle_deltas.iter().copied().take(3));

        let mut best_blk_delta = f64::NEG_INFINITY;
        for blen in 2..=5usize.min(k_len.saturating_sub(1)) {
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
                    let mut valid = true;
                    for i in 0..blen {
                        let a = work_at[k_win + i];
                        let b = work_at[k_alt + i];
                        if (b >= 0 && u_mat.get(b as usize, k_win + i) <= (IDP_FORBID_HALF as f64))
                            || (a >= 0
                                && u_mat.get(a as usize, k_alt + i) <= (IDP_FORBID_HALF as f64))
                        {
                            valid = false;
                            break;
                        }
                    }
                    if !valid {
                        continue;
                    }
                    let before = eval_block_slots(&work_at, u_mat, p_mat, k_win, k_alt, blen);
                    for i in 0..blen {
                        work_at.swap(k_win + i, k_alt + i);
                    }
                    let delta =
                        eval_block_slots(&work_at, u_mat, p_mat, k_win, k_alt, blen) - before;
                    for i in 0..blen {
                        work_at.swap(k_win + i, k_alt + i);
                    }
                    if delta > best_blk_delta {
                        best_blk_delta = delta;
                    }
                }
            }
        }
        sys_best_blk[s] = best_blk_delta;
        let best_single_delta = sys_top_deltas[s]
            .first()
            .copied()
            .unwrap_or(f64::NEG_INFINITY);
        if best_single_delta.max(best_blk_delta) >= -1.75 {
            n_ambig_placements += 1;
        }
    }

    let ambig_placement_frac = if n_assigned > 0 {
        (n_ambig_placements as f64) / (n_assigned as f64)
    } else {
        1.0
    };

    let rms_scale = if emp_rms_seq_ca > 0.042 {
        (emp_rms_seq_ca / 0.038).clamp(1.0, 2.50)
    } else {
        1.0
    };
    let cand_scale = if mean_cands_per_ch > 1.54 {
        (mean_cands_per_ch / 1.48).clamp(1.0, 1.60)
    } else {
        1.0
    };
    let no_cccon_scale = if !any_cccon
        && emp_rms_seq_ca >= 0.034
        && (ambig_placement_frac >= 0.12 || connectivity_frac < 0.92)
        && (mean_cands_per_ch > 1.52 || ambig_placement_frac >= 0.22 || connectivity_frac < 0.90)
    {
        // Without 3D CCC(CO)N TOCSY sidechain carbons, amino acids sharing overlapping `(CA, CB)`
        // random-coil basins (`Glu/Gln/Lys/Arg/Met`, `Leu/Phe/Tyr/Asp/Asn`) lack CG/CD/CE
        // disambiguation; under elevated matching noise + low-margin swap ambiguity + spurious
        // candidate multiplicity, widen `tau_local_eff` proportionally.
        let ambig_excess = (ambig_placement_frac - 0.20).max(0.0);
        let mult_excess = (mean_cands_per_ch - 1.54).max(0.0);
        let conn_deficit = (0.90 - connectivity_frac).max(0.0);
        let rms_excess = (emp_rms_seq_ca - 0.055).max(0.0);
        (1.0 + 1.55 * ambig_excess + 0.80 * mult_excess + 3.80 * conn_deficit + 9.0 * rms_excess)
            .clamp(1.0, 1.78)
    } else {
        1.0
    };
    let repeat_scale = if seq_low_comp_warn { 1.85 } else { 1.0 };
    let tau_local_base = TAU_LOCAL * rms_scale * cand_scale * no_cccon_scale * repeat_scale;
    let w_ens = if !any_cccon && emp_rms_seq_ca >= 0.038 {
        0.28
    } else {
        0.34
    };

    let mut raw_posterior = vec![0.0f64; s_len];
    let mut confidence = vec![0.0f64; s_len];

    for s in 0..s_len {
        if cons_pos[s] < 0 {
            raw_posterior[s] = 0.0;
            confidence[s] = 0.0;
            continue;
        }
        let tau_s = tau_local_base;
        let mut alt_sum = ((-sys_base_1[s] - 2.0) / tau_s).clamp(-35.0, 20.0).exp();
        for &delta in &sys_top_deltas[s] {
            alt_sum += (delta / tau_s).clamp(-35.0, 20.0).exp();
        }
        for &cdelta in &sys_top_cycle_deltas[s] {
            alt_sum += (cdelta / (1.15 * tau_s)).clamp(-35.0, 20.0).exp();
        }
        let best_blk_delta = sys_best_blk[s];
        if best_blk_delta > -4.0 {
            alt_sum += (best_blk_delta / (1.15 * tau_s)).clamp(-35.0, 20.0).exp();
        }

        let p_local = 1.0 / (1.0 + alt_sum);
        let p_combined = (1.0 - w_ens) * p_local + w_ens * (p_local * p_ens[s]).sqrt();
        raw_posterior[s] = p_combined;
        confidence[s] = eval_idp_calibration(p_combined);
    }

    // 3. Unlabelled Dataset Quality-Control Verdict (`Pass`, `Warn`, `Abstain`)
    let n_3d_sup = systems.iter().filter(|s| s.has_3d_support()).count();
    let support_3d_frac = (n_3d_sup as f64) / (s_len as f64);

    let n_expected_con = k_len.saturating_sub(1).max(1);
    let root_ratio = (s_len as f64) / (n_expected_con as f64);
    let mean_compat_2d_out = if margin_eval_2d_roots > 0 {
        compat_2d_out_sum / (margin_eval_2d_roots as f64)
    } else {
        0.0
    };

    let mut qc_reasons: Vec<String> = Vec::new();
    let mut severe_abstain = false;
    let mut severe_degeneracy_warn = false;
    let mut drift_or_sparse_warn = false;
    let mut any_warn = false;

    // (a) 3D strip support & sequential connectivity gates
    if support_3d_frac < 0.25 {
        qc_reasons.push(format!(
            "Catastrophic 3D strip support loss ({:.1}% < 25.0% of CON roots supported)",
            support_3d_frac * 100.0
        ));
        severe_abstain = true;
    } else if s_len >= 6 && connectivity_frac < 0.15 {
        qc_reasons.push(format!(
            "Sequential dipeptide connectivity collapsed ({:.1}% < 15.0% linked)",
            connectivity_frac * 100.0
        ));
        severe_abstain = true;
    } else if support_3d_frac < 0.55 || (s_len >= 6 && connectivity_frac < 0.40) {
        qc_reasons.push(format!(
            "Partial 3D support ({:.1}%) or sparse sequential connectivity ({:.1}%)",
            support_3d_frac * 100.0,
            connectivity_frac * 100.0
        ));
        drift_or_sparse_warn = true;
        any_warn = true;
    }

    // (b) 2D CON root-deficit gate (`n_roots < 0.70 * n_expected_con`, catching Q80 root collapse!)
    if k_len >= 20 && root_ratio < 0.35 {
        qc_reasons.push(format!(
            "Severe 2D CON root collapse ({s_len}/{n_expected_con} = {:.1}% < 35.0% expected peptide bonds; severe peak overlap or sensitivity loss)",
            root_ratio * 100.0
        ));
        severe_abstain = true;
    } else if k_len >= 20 && root_ratio < 0.70 {
        qc_reasons.push(format!(
            "2D CON root deficit ({s_len}/{n_expected_con} = {:.1}% < 70.0% expected peptide bonds)",
            root_ratio * 100.0
        ));
        severe_degeneracy_warn = true;
        any_warn = true;
    }

    // (c) Primary-sequence low-complexity / repeat gate (catching poly-Q & FG-repeat!)
    if k_len >= 16 && (h_aa_bits < 1.65 || max_aa_frac >= 0.55 || max_dipep_frac >= 0.35) {
        qc_reasons.push(format!(
            "Extreme low-complexity / homopolymeric repeat sequence (Shannon H={h_aa_bits:.2} bits, dominant AA={:.1}%, dominant dipeptide={:.1}%)",
            max_aa_frac * 100.0,
            max_dipep_frac * 100.0
        ));
        severe_abstain = true;
    } else if seq_low_comp_warn {
        qc_reasons.push(format!(
            "Low-complexity / repetitive IDP sequence (Shannon H={h_aa_bits:.2} bits < 2.60 or repeat fraction max_AA={:.1}%, top3_dipeptide={:.1}% >= 35.0%)",
            max_aa_frac * 100.0,
            top3_dipep_frac * 100.0
        ));
        severe_degeneracy_warn = true;
        any_warn = true;
    }

    // (d) Candidate sequential multiplicity / elevated realistic-noise gate (all metrics length-invariant!):
    // Because `ca_curr` merges `{CANCO, CACO, NCACO}` and `ca_prev` merges `{CANCO, CBCACON}`,
    // clean single-peak 13CA jitter `sigma_peak ~ 0.03-0.045 ppm` (plus small cross-channel digital
    // resolution variance on finite-size IDPs) produces sequential channel-difference RMS up to
    // `~0.064 ppm`, so `emp_rms_seq_ca >= 0.068 ppm` cleanly separates clean 1x experimental
    // precision (`Pass`) from degraded 3x / mis-picked broadening (`Warn`), while
    // `mean_cands_per_ch >= 1.52`, anchored strip multiplicity `mean_prev_cands >= 1.25`,
    // `ambig_placement_frac >= 0.22`, and `compat_2d_density >= 0.14` catch equal-height spurious floods.
    let compat_2d_density = if s_len > 0 {
        mean_compat_2d_out / (s_len as f64)
    } else {
        0.0
    };
    if n_3d_sup >= 8
        && (emp_rms_seq_ca >= 0.068
            || mean_cands_per_ch >= 1.52
            || mean_prev_cands >= 1.25
            || ambig_placement_frac >= 0.22
            || compat_2d_density >= 0.14)
    {
        qc_reasons.push(format!(
            "Elevated sequential ambiguity or matching noise (empirical CA RMS={emp_rms_seq_ca:.3} ppm, mean cands/ch={mean_cands_per_ch:.2}, 2D compat density={:.1}%, low-margin placements={:.1}%)",
            compat_2d_density * 100.0,
            ambig_placement_frac * 100.0
        ));
        any_warn = true;
    }

    // (e) Picker (`|drift| >= 0.38 ppm`) & solver (`|drift| >= 0.65 ppm` / global `>= 3.80 ppm`) carrier-drift clamp warnings
    for w in &cfg.picker_warnings {
        qc_reasons.push(w.clone());
        drift_or_sparse_warn = true;
        any_warn = true;
    }
    for sys in systems {
        for w in &sys.registration_warnings {
            qc_reasons.push(w.clone());
            drift_or_sparse_warn = true;
            any_warn = true;
        }
    }
    for (ax_lbl, dv) in [
        ("15N", fitted_offsets.d_n),
        ("13C'", fitted_offsets.d_c),
        ("13CA", fitted_offsets.d_ca),
        ("13CB", fitted_offsets.d_cb),
    ] {
        if dv.abs() >= 3.80 {
            qc_reasons.push(format!(
                "Global {ax_lbl} reference offset ({dv:+.3} ppm) hit ±4.00 ppm estimator boundary; verify referencing"
            ));
            drift_or_sparse_warn = true;
            any_warn = true;
        }
    }

    let qc_verdict = if severe_abstain {
        DatasetQcVerdict::Abstain
    } else if any_warn {
        DatasetQcVerdict::Warn
    } else {
        DatasetQcVerdict::Pass
    };

    match qc_verdict {
        DatasetQcVerdict::Abstain => {
            let cap = (cfg.hc_threshold - 0.35).clamp(0.10, 0.45);
            for c in &mut confidence {
                *c = (*c * 0.45).min(cap);
            }
        }
        DatasetQcVerdict::Warn => {
            if severe_degeneracy_warn {
                for c in &mut confidence {
                    *c = (*c * 0.76).clamp(0.0, 0.84);
                }
            } else if drift_or_sparse_warn {
                for c in &mut confidence {
                    *c = (*c * 0.94).clamp(0.0, 0.97);
                }
            }
            // Note: for elevated noise / candidate-multiplicity `Warn`, `tau_local_eff` has
            // already widened the Boltzmann specificity denominator before monotone LOPO knot
            // calibration so ambiguous placements drop below `0.80` while uniquely resolved
            // links stay calibrated without post-hoc underconfidence bias.
        }
        DatasetQcVerdict::Pass => {}
    }

    IdpAssignResult {
        position: cons_pos,
        confidence,
        raw_posterior,
        score: final_sc,
        run_scores,
        fitted_offsets,
        qc_verdict,
        support_3d_frac,
        connectivity_frac,
        qc_reasons,
    }
}

/// Fallible multi-island evolutionary + detailed-balance simulated annealing CPU solver enforcing
/// [`MAX_IDP_MATRIX_CELLS`] (`S * K <= 25,000,000` and `S * S <= 25,000,000`, returning `Err` on overflow).
pub fn try_assign_idp_13c_cpu(
    systems: &[ConDipeptideSystem],
    sequence: &[String],
    cfg: &IdpSolverConfig,
) -> Result<IdpAssignResult, String> {
    let s_len = systems.len();
    let k_len = sequence.len();
    check_idp_matrix_bounds(s_len, k_len)?;

    let (fitted_offsets, ref_safety) = if cfg.auto_reference_offset {
        let raw = estimate_idp_reference_offsets(systems, sequence, cfg);
        let safety = evaluate_idp_auto_reference_safety(systems, &raw);
        (safety.safe_offsets, Some(safety))
    } else {
        (cfg.manual_offsets, None)
    };

    if s_len == 0 || k_len < 2 {
        let mut res = consensus_and_calibrate_idp(
            systems,
            sequence,
            &Matrix2D::filled(0, 0, 0.0),
            &Matrix2D::filled(0, 0, 0.0),
            &[],
            &[],
            &[],
            fitted_offsets,
            cfg,
        );
        if let Some(ref sf) = ref_safety {
            for w in &sf.warnings {
                if !res.qc_reasons.contains(w) {
                    res.qc_reasons.push(w.clone());
                }
            }
            if let Some(ref hint) = sf.tms_reference_hint {
                if !res.qc_reasons.contains(hint) {
                    res.qc_reasons.push(hint.clone());
                }
            }
        }
        return Ok(res);
    }

    let u_mat = try_idp_unary_matrix(systems, sequence, cfg, &fitted_offsets)?;
    let p_mat = try_idp_pair_matrix(systems, cfg)?;
    let slots = build_idp_valid_slots(&u_mat);

    let n_islands = cfg.islands.max(1);
    let total_replicas = cfg.replicas.max(n_islands).max(4);
    let gens = cfg.generations.max(1);
    let total_steps = cfg
        .steps_per_system
        .saturating_mul(s_len.max(24))
        .max(20_000);
    let steps_per_gen = (total_steps / gens).max(4_000);

    let mut population =
        build_idp_initial_replicas(&u_mat, &p_mat, &slots, total_replicas, cfg.seed);
    let mut master_rng = Rng::new(cfg.seed ^ 0x4350_555F_4944_5031);

    #[cfg(target_arch = "wasm32")]
    let n_threads = 1usize;
    #[cfg(not(target_arch = "wasm32"))]
    let n_threads = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(total_replicas)
        .max(1);

    for gen_idx in 0..gens {
        let gen_t_start = if gen_idx == 0 {
            cfg.t_start
        } else {
            (cfg.t_start * 0.38_f64.powi(gen_idx as i32)).max(cfg.t_end * 4.0)
        };
        let seeds: Vec<u64> = (0..total_replicas).map(|_| master_rng.next_u64()).collect();

        if n_threads <= 1 {
            for (r, sd) in seeds.into_iter().enumerate() {
                let mut r_rng = Rng::new(sd);
                let (bat, bpos, bsc) = run_idp_cpu_anneal_segment(
                    &mut population[r],
                    &u_mat,
                    &p_mat,
                    &slots,
                    steps_per_gen,
                    gen_t_start,
                    cfg.t_end,
                    &mut r_rng,
                );
                population[r] = (bat, bpos, bsc);
            }
        } else {
            let chunk_sz = total_replicas.div_ceil(n_threads);
            let u_ref = &u_mat;
            let p_ref = &p_mat;
            let sl_ref = &slots;
            let t_end = cfg.t_end;
            let indexed: Vec<(usize, IdpReplicaState, u64)> = population
                .into_iter()
                .zip(seeds)
                .enumerate()
                .map(|(i, (st, sd))| (i, st, sd))
                .collect();

            let batches: Vec<Vec<(usize, IdpReplicaState)>> = thread::scope(|scope| {
                let mut handles = Vec::new();
                for chunk in indexed.chunks(chunk_sz) {
                    handles.push(scope.spawn(move || {
                        let mut out = Vec::with_capacity(chunk.len());
                        for &(idx, ref st_in, sd) in chunk {
                            let mut st = st_in.clone();
                            let mut r_rng = Rng::new(sd);
                            let (bat, bpos, bsc) = run_idp_cpu_anneal_segment(
                                &mut st,
                                u_ref,
                                p_ref,
                                sl_ref,
                                steps_per_gen,
                                gen_t_start,
                                t_end,
                                &mut r_rng,
                            );
                            out.push((idx, (bat, bpos, bsc)));
                        }
                        out
                    }));
                }
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });

            population = vec![(Vec::new(), Vec::new(), 0.0); total_replicas];
            for batch in batches {
                for (idx, st) in batch {
                    population[idx] = st;
                }
            }
        }

        // Polish elites and run island evolutionary segment crossover between generations
        if gen_idx + 1 < gens && total_replicas >= 4 {
            let per_island = (total_replicas / n_islands).max(2);
            for isl in 0..n_islands {
                let lo = isl * per_island;
                let hi = if isl + 1 == n_islands {
                    total_replicas
                } else {
                    ((isl + 1) * per_island).min(total_replicas)
                };
                if hi <= lo + 1 {
                    continue;
                }
                population[lo..hi].sort_by(|a, b| b.2.total_cmp(&a.2));
                for elite in &mut population[lo..(lo + 2).min(hi)] {
                    elite.2 = polish_idp_state(&mut elite.0, &mut elite.1, &u_mat, &p_mat, &slots);
                }
                population[lo..hi].sort_by(|a, b| b.2.total_cmp(&a.2));
                let elite_clone = population[lo].clone();
                let second_clone = population[lo + 1].clone();
                let mid = lo + (hi - lo) / 2;
                for idx in mid..hi {
                    let donor = if idx % 2 == 0 {
                        &elite_clone
                    } else {
                        &second_clone
                    };
                    population[idx] = crossover_idp_states(
                        &population[lo + (idx - mid) % (mid - lo)],
                        donor,
                        &u_mat,
                        &p_mat,
                        &slots,
                        &mut master_rng,
                    );
                }
            }
            if n_islands >= 2 && per_island >= 3 {
                let elites: Vec<IdpReplicaState> = (0..n_islands)
                    .map(|isl| population[isl * per_island].clone())
                    .collect();
                for (isl, elite_st) in elites.into_iter().enumerate() {
                    let target_isl = (isl + 1) % n_islands;
                    let replace_idx =
                        (target_isl * per_island + per_island - 1).min(total_replicas - 1);
                    population[replace_idx] = elite_st;
                }
            }
        }
    }

    let rep_positions: Vec<Vec<i32>> = population.iter().map(|st| st.1.clone()).collect();
    let rep_scores: Vec<f64> = population.iter().map(|st| st.2).collect();

    let mut res = consensus_and_calibrate_idp(
        systems,
        sequence,
        &u_mat,
        &p_mat,
        &slots,
        &rep_positions,
        &rep_scores,
        fitted_offsets,
        cfg,
    );
    if let Some(ref sf) = ref_safety {
        // Always surface auto-reference safety warnings (including 1.0..1.8 ppm spread) in QC notes.
        for w in &sf.warnings {
            if !res.qc_reasons.contains(w) {
                res.qc_reasons.push(w.clone());
            }
        }
        let severe_ref_fail = !sf.refused_nuclei.is_empty() || sf.c13_spread_ppm > 1.80;
        if severe_ref_fail && !sf.warnings.is_empty() && res.qc_verdict == DatasetQcVerdict::Pass {
            res.qc_verdict = DatasetQcVerdict::Warn;
        }
        if let Some(ref hint) = sf.tms_reference_hint {
            if !res.qc_reasons.contains(hint) {
                res.qc_reasons.push(hint.clone());
            }
        }
    }
    Ok(res)
}

/// Multi-island evolutionary + detailed-balance simulated annealing CPU solver for continuous
/// through-Proline `13C`-detected IDP assignment (`T11.2`).
///
/// Delegates to [`try_assign_idp_13c_cpu`], returning an explicit `Abstain` result without
/// allocating `U`/`P` if `S * K > 25,000,000`.
pub fn assign_idp_13c_cpu(
    systems: &[ConDipeptideSystem],
    sequence: &[String],
    cfg: &IdpSolverConfig,
) -> IdpAssignResult {
    match try_assign_idp_13c_cpu(systems, sequence, cfg) {
        Ok(res) => res,
        Err(msg) => IdpAssignResult {
            position: vec![-1; systems.len()],
            confidence: vec![0.0; systems.len()],
            raw_posterior: vec![0.0; systems.len()],
            score: 0.0,
            run_scores: Vec::new(),
            fitted_offsets: cfg.manual_offsets,
            qc_verdict: DatasetQcVerdict::Abstain,
            support_3d_frac: 0.0,
            connectivity_frac: 0.0,
            qc_reasons: vec![msg],
        },
    }
}

/// Fallible end-to-end convenience pipeline enforcing [`MAX_IDP_MATRIX_CELLS`] (`Err` on overflow).
pub fn try_assign_idp_13c_from_picked(
    picked: &C13PickedDataset,
    sequence: &[String],
    cfg: &IdpSolverConfig,
) -> Result<(Vec<ConDipeptideSystem>, IdpAssignResult), String> {
    check_idp_matrix_bounds(picked.con_roots.len(), sequence.len())?;
    let systems = register_and_build_con_dipeptide_systems(picked);
    let result = try_assign_idp_13c_cpu(&systems, sequence, cfg)?;
    Ok((systems, result))
}

/// End-to-end convenience pipeline: register inter-experiment drift across `picked`, build
/// `ConDipeptideSystem`s, and run `assign_idp_13c_cpu`.
pub fn assign_idp_13c_from_picked(
    picked: &C13PickedDataset,
    sequence: &[String],
    cfg: &IdpSolverConfig,
) -> (Vec<ConDipeptideSystem>, IdpAssignResult) {
    let systems = register_and_build_con_dipeptide_systems(picked);
    let result = assign_idp_13c_cpu(&systems, sequence, cfg);
    (systems, result)
}

/// Convert a `13C`-detected IDP assignment (`ConDipeptideSystem`s placed on `k = 1..L-1`,
/// continuously through Prolines!) into a residue-indexed `Reference` chemical shift table.
pub fn idp_to_reference(
    result: &IdpAssignResult,
    systems: &[ConDipeptideSystem],
    sequence: &[String],
    seq_ids: &[i32],
    min_conf: f64,
    de_offset: bool,
) -> (Reference, BTreeMap<usize, f64>) {
    let mut out = Reference::new(sequence.to_vec(), seq_ids.to_vec());
    let mut conf_map: BTreeMap<usize, f64> = BTreeMap::new();
    let off = if de_offset {
        result.fitted_offsets
    } else {
        IdpShiftOffsets::default()
    };
    let preds = predict_ncidp_shifts(sequence, &IdpSolverConfig::default());

    let mut at: BTreeMap<usize, usize> = BTreeMap::new();
    for (s, &k_i) in result.position.iter().enumerate() {
        if k_i >= 1 && result.confidence[s] >= min_conf {
            at.insert(k_i as usize, s);
        }
    }

    for (&k, &s) in &at {
        let sys = &systems[s];
        let c = result.confidence[s];
        conf_map.insert(k, c);
        out.shifts
            .insert((k, "N".to_string()), sys.n_curr - off.d_n);
        out.shifts
            .insert((k - 1, "C".to_string()), sys.c_prev - off.d_c);
        conf_map.entry(k - 1).or_insert(c);

        // `CA(k)`: corroborate against `at[k+1].ca_prev` when `k + 1` is assigned
        if !sys.ca_curr.is_empty() {
            let ca_val = if let Some(&s_nxt) = at.get(&(k + 1)) {
                let nxt_prev = &systems[s_nxt].ca_prev;
                if !nxt_prev.is_empty() {
                    sys.ca_curr
                        .iter()
                        .copied()
                        .min_by(|&a, &b| {
                            let da = nxt_prev
                                .iter()
                                .map(|&p| (a - p).abs())
                                .fold(f64::INFINITY, f64::min);
                            let db = nxt_prev
                                .iter()
                                .map(|&p| (b - p).abs())
                                .fold(f64::INFINITY, f64::min);
                            da.total_cmp(&db)
                        })
                        .unwrap_or(sys.ca_curr[0])
                } else {
                    sys.ca_curr[0]
                }
            } else {
                sys.ca_curr[0]
            };
            out.shifts.insert((k, "CA".to_string()), ca_val - off.d_ca);
        }

        if !sys.ca_prev.is_empty() && !out.shifts.contains_key(&(k - 1, "CA".to_string())) {
            out.shifts
                .insert((k - 1, "CA".to_string()), sys.ca_prev[0] - off.d_ca);
        }
        if !sys.cb_prev.is_empty() && !sequence[k - 1].eq_ignore_ascii_case("GLY") {
            let cb_val = if sys.cb_prev.len() > 1 {
                if let Some(mu_cb) = preds.get(k - 1).and_then(|p| p.cb_ppm) {
                    let target = mu_cb + result.fitted_offsets.d_cb;
                    sys.cb_prev
                        .iter()
                        .copied()
                        .min_by(|&a, &b| (a - target).abs().total_cmp(&(b - target).abs()))
                        .unwrap_or(sys.cb_prev[0])
                } else {
                    sys.cb_prev[0]
                }
            } else {
                sys.cb_prev[0]
            };
            out.shifts
                .insert((k - 1, "CB".to_string()), cb_val - off.d_cb);
        }
        if !sys.cb_curr.is_empty()
            && !sequence[k].eq_ignore_ascii_case("GLY")
            && !out.shifts.contains_key(&(k, "CB".to_string()))
        {
            out.shifts
                .insert((k, "CB".to_string()), sys.cb_curr[0] - off.d_cb);
        }
        if !sys.c_curr.is_empty() && !out.shifts.contains_key(&(k, "C".to_string())) {
            out.shifts
                .insert((k, "C".to_string()), sys.c_curr[0] - off.d_c);
        }
    }

    (out, conf_map)
}

// ============================================================================
// RD-10 & RD-13: CAS + H(CC)CON + HNCO->HNCACB + (H)N(CA)CON / HN(CA)CONH Links
// ============================================================================

/// Shared `HNCACB` `i` vs `i-1` disambiguation helper (`P2-3`):
/// 1) Filters out peaks corroborated as `(i-1)` by `prev_list` or `hncocacb_c` (backfilling
///    `prev_list` when empty);
/// 2) When no `(i-1)` reference matched, uses relative `1J` vs `2J` intensity (`>= 1.35x`) to
///    separate intraresidue `(i)` from preceding `(i-1)` while keeping ambiguous equal-height
///    doublets ordered by intensity.
pub(crate) fn select_hncacb_intra_and_backfill_prev(
    hits: &[(f64, f64)],
    prev_list: &mut Vec<f64>,
    hncocacb_c: &[f64],
    tol_p: f64,
) -> Vec<f64> {
    if hits.is_empty() {
        return Vec::new();
    }
    let is_known_prev = |v: f64, prev: &[f64]| -> bool {
        prev.iter().any(|&p| (p - v).abs() <= tol_p)
            || hncocacb_c.iter().any(|&p| (p - v).abs() <= tol_p)
    };
    let mut filt: Vec<(f64, f64)> = Vec::new();
    let mut matched_prev_any = false;
    for &(v, sc) in hits {
        if is_known_prev(v, prev_list) {
            matched_prev_any = true;
            if prev_list.is_empty() {
                prev_list.push(v);
            }
        } else {
            filt.push((v, sc));
        }
    }
    if matched_prev_any {
        if !filt.is_empty() {
            if filt.len() >= 2 && filt[0].1 >= 1.35 * filt[1].1 {
                return vec![filt[0].0];
            }
            return filt.into_iter().take(2).map(|(v, _)| v).collect();
        }
        return vec![hits[0].0];
    }
    if hits.len() >= 2 && (hits[0].0 - hits[1].0).abs() > 0.25 {
        if hits[0].1 >= 1.35 * hits[1].1 {
            if prev_list.is_empty() {
                prev_list.push(hits[1].0);
            }
            return vec![hits[0].0];
        }
        return vec![hits[0].0, hits[1].0];
    }
    hits.iter().take(1).map(|&(v, _)| v).collect()
}

/// Summary statistics from matching `HNCO -> HNCACB` onto `CON` roots (`RD-10`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct IdpHncoHncacbLinkStats {
    pub n_con_roots_matched_hnco: usize,
    pub n_roots_with_hncacb_curr: usize,
    pub hncacb_polarity: Option<PolarityReport>,
}

/// Match `HNCO` (`H_N(i), N(i), C'(i-1)`) onto `CON` roots by `(C'(i-1), N(i))`, then populate
/// intraresidue `CA(i)` and `CB(i)` from `HNCACB` strips at `(H_N(i), N(i))` (filtering out peaks
/// matching already-assigned `(i-1)` shifts from `H(CC)CON`/`CBCACON`) with automatic `RD-1`
/// polarity normalization (`RD-10`).
pub fn attach_hnco_hncacb_to_con_systems(
    systems: &mut [ConDipeptideSystem],
    hnco_peaks: &[Peak],
    hncacb_peaks: &[Peak],
    polarity_mode: HncacbPolarity,
) -> IdpHncoHncacbLinkStats {
    if systems.is_empty() || hnco_peaks.is_empty() || hncacb_peaks.is_empty() {
        return IdpHncoHncacbLinkStats::default();
    }
    let hnco: Vec<&Peak> = hnco_peaks
        .iter()
        .filter(|p| p.exp == "HNCO" && p.c.is_some())
        .collect();
    if hnco.is_empty() {
        return IdpHncoHncacbLinkStats::default();
    }

    let (norm_peaks, pol_rep) = normalize_hncacb_and_cbcaconh_polarity(hncacb_peaks, polarity_mode);
    let hncacb: Vec<&Peak> = norm_peaks
        .iter()
        .filter(|p| p.exp == "HNCACB" && p.c.is_some())
        .collect();
    // Optional i-1-only CBCACONH / HN(CO)CACB peaks carried in the same peak list for cross-check
    let cbcaconh: Vec<&Peak> = norm_peaks
        .iter()
        .filter(|p| (p.exp == "CBCACONH" || p.exp == "HNCOCACB") && p.c.is_some())
        .collect();
    if hncacb.is_empty() {
        return IdpHncoHncacbLinkStats {
            n_con_roots_matched_hnco: 0,
            n_roots_with_hncacb_curr: 0,
            hncacb_polarity: Some(pol_rep),
        };
    }

    // Check optional small bounded uniform carrier drift between CON and HNCO
    let mut dcs = Vec::new();
    let mut dns = Vec::new();
    for s in systems.iter() {
        let mut best: Option<(f64, f64, f64)> = None;
        for hp in &hnco {
            let dc = hp.c.unwrap_or(0.0) - s.c_prev;
            let dn = hp.n - s.n_curr;
            if dc.abs() <= 0.15 && dn.abs() <= 0.45 {
                let d2 = (dc / 0.06).powi(2) + (dn / 0.25).powi(2);
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
    let (sh_c, sh_n) = if dcs.len() >= 5 {
        dcs.sort_by(|a, b| a.total_cmp(b));
        dns.sort_by(|a, b| a.total_cmp(b));
        let mc = dcs[dcs.len() / 2];
        let mn = dns[dns.len() / 2];
        if mc.abs() <= 0.12 && mn.abs() <= 0.40 {
            (mc, mn)
        } else {
            (0.0, 0.0)
        }
    } else {
        (0.0, 0.0)
    };

    // Global distance-sorted 1-to-1 CON <-> HNCO matching so crowded CON roots do not share an HNCO peak
    let mut pair_cands: Vec<(f64, usize, usize)> = Vec::new();
    for (s_idx, sys) in systems.iter().enumerate() {
        for (h_idx, hp) in hnco.iter().enumerate() {
            let dc = ((hp.c.unwrap_or(0.0) - sh_c) - sys.c_prev).abs();
            let dn = ((hp.n - sh_n) - sys.n_curr).abs();
            if dc <= 0.09 && dn <= 0.35 {
                let d2 = (dc / 0.06).powi(2) + (dn / 0.22).powi(2);
                pair_cands.push((d2, s_idx, h_idx));
            }
        }
    }
    pair_cands.sort_by(|a, b| {
        a.0.total_cmp(&b.0)
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| a.2.cmp(&b.2))
    });
    let mut used_sys = vec![false; systems.len()];
    let mut used_hnco = vec![false; hnco.len()];
    let mut matched_hnco_for_sys: Vec<Option<usize>> = vec![None; systems.len()];
    let mut matched_hnco = 0usize;
    for (_, s_idx, h_idx) in pair_cands {
        if used_sys[s_idx] || used_hnco[h_idx] {
            continue;
        }
        used_sys[s_idx] = true;
        used_hnco[h_idx] = true;
        matched_hnco_for_sys[s_idx] = Some(h_idx);
        matched_hnco += 1;
    }

    let mut attached_curr = 0usize;
    for (s_idx, sys) in systems.iter_mut().enumerate() {
        let Some(h_idx) = matched_hnco_for_sys[s_idx] else {
            continue;
        };
        let hp = hnco[h_idx];

        let mut ca_hits: Vec<(f64, f64)> = Vec::new();
        let mut cb_hits: Vec<(f64, f64)> = Vec::new();
        for cp in &hncacb {
            let dh = (cp.h - hp.h).abs();
            let dn = (cp.n - hp.n).abs();
            let d_hn = (dh / 0.045).hypot(dn / 0.45);
            if d_hn > 1.0 {
                continue;
            }
            let Some(c_val) = cp.c else {
                continue;
            };
            let score = cp.height.map(f64::abs).unwrap_or(1.0) * (-0.5 * d_hn * d_hn).exp();
            if cp.sign >= 0 {
                ca_hits.push((c_val, score));
            } else {
                cb_hits.push((c_val, score));
            }
        }
        ca_hits.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.total_cmp(&b.0)));
        cb_hits.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.total_cmp(&b.0)));

        // Collect optional i-1 HN(CO)CACB / CBCACONH shifts at the same (H_N(i), N(i)) strip
        let mut hncocacb_c: Vec<f64> = Vec::new();
        for bp in &cbcaconh {
            let dh = (bp.h - hp.h).abs();
            let dn = (bp.n - hp.n).abs();
            let d_hn = (dh / 0.045).hypot(dn / 0.45);
            if d_hn <= 1.0 {
                if let Some(c_val) = bp.c {
                    hncocacb_c.push(c_val);
                }
            }
        }

        let ca_src =
            select_hncacb_intra_and_backfill_prev(&ca_hits, &mut sys.ca_prev, &hncocacb_c, 0.18);
        let cb_src =
            select_hncacb_intra_and_backfill_prev(&cb_hits, &mut sys.cb_prev, &hncocacb_c, 0.22);

        let mut any_added = false;
        for v in ca_src {
            if sys.ca_curr.len() < MAX_IDP_CANDS
                && !sys.ca_curr.iter().any(|&x| (x - v).abs() <= 0.08)
            {
                sys.ca_curr.push(v);
                any_added = true;
            }
        }
        for v in cb_src {
            if sys.cb_curr.len() < MAX_IDP_CANDS
                && !sys.cb_curr.iter().any(|&x| (x - v).abs() <= 0.12)
            {
                sys.cb_curr.push(v);
                any_added = true;
            }
        }
        if any_added {
            attached_curr += 1;
        }
    }

    IdpHncoHncacbLinkStats {
        n_con_roots_matched_hnco: matched_hnco,
        n_roots_with_hncacb_curr: attached_curr,
        hncacb_polarity: Some(pol_rep),
    }
}

/// Extract and co-add `(H)N(CA)CON` sequential `N(i-1)` partner peaks at each `CON` root
/// (`(C'(i-1), N(i))`) across one or more replicate 3D cubes (`RD-13`), scaling combined SNR by
/// `1 / sqrt(n_replicates)` and retaining partners matching real `CON` `15N` coordinates.
pub fn pick_hncacon_nn_coadded(
    cubes: &[&Spectrum3D],
    con_roots: &[Con2DPeak],
    snr_threshold: f64,
) -> Vec<Vec<(f64, f64)>> {
    let n_roots = con_roots.len();
    let mut out: Vec<Vec<(f64, f64)>> = vec![Vec::new(); n_roots];
    if cubes.is_empty() || con_roots.is_empty() {
        return out;
    }

    const GRID_LO: f64 = 103.0;
    const GRID_STEP: f64 = 0.05;
    const GRID_LEN: usize = 820; // 103.0..=144.0 ppm

    let nearest_idx = |ax: &crate::spectrum::AxisPpm, ppm: f64| -> (usize, f64) {
        let idx_f = ax.ppm_to_index(ppm);
        let max_i = ax.size.saturating_sub(1);
        let i = (idx_f.round() as isize).clamp(0, max_i as isize) as usize;
        let d = (ax.index_to_ppm(i as f64) - ppm).abs();
        (i, d)
    };

    let sample_linear = |ax: &crate::spectrum::AxisPpm, profile: &[f64], target_ppm: f64| -> f64 {
        let p = ax.ppm_to_index(target_ppm);
        if p < 0.0 || p > ((ax.size.saturating_sub(1)) as f64) {
            return 0.0;
        }
        let lo = (p.floor() as usize).min(ax.size.saturating_sub(1));
        let hi = (lo + 1).min(ax.size.saturating_sub(1));
        let frac = p - (lo as f64);
        profile[lo] * (1.0 - frac) + profile[hi] * frac
    };

    let mut coadd_strips: Vec<Vec<f64>> = vec![vec![0.0; GRID_LEN]; n_roots];
    let mut strip_counts: Vec<usize> = vec![0; n_roots];

    for spec in cubes {
        let noise = (estimate_noise_mad(&spec.data) as f64).max(1e-9);
        // Determine axis assignment (`direct C'` vs `two 15N indirect dimensions`) by chemical shift range
        let is_c_axis = |ax: &crate::spectrum::AxisPpm| {
            let nuc = ax.label.to_ascii_uppercase();
            if nuc.contains('C') {
                return true;
            }
            if nuc.contains('N') {
                return false;
            }
            let c_mid = ax.index_to_ppm((ax.size as f64) * 0.5);
            c_mid > 150.0
        };

        // Detect which axis is C' (X, Y, or Z), and which two are N
        let c_dim = if is_c_axis(&spec.axis_x) {
            0
        } else if is_c_axis(&spec.axis_y) {
            1
        } else {
            2
        };
        let (n_dim_a, n_dim_b) = match c_dim {
            0 => (1, 2),
            1 => (0, 2),
            _ => (0, 1),
        };
        let ax_c = match c_dim {
            0 => &spec.axis_x,
            1 => &spec.axis_y,
            _ => &spec.axis_z,
        };
        let ax_a = match n_dim_a {
            0 => &spec.axis_x,
            1 => &spec.axis_y,
            _ => &spec.axis_z,
        };
        let ax_b = match n_dim_b {
            0 => &spec.axis_x,
            1 => &spec.axis_y,
            _ => &spec.axis_z,
        };

        let get_voxel = |ic: usize, ia: usize, ib: usize| -> f64 {
            let (ix, iy, iz) = match c_dim {
                0 => (ic, ia, ib),
                1 => (ia, ic, ib),
                _ => (ia, ib, ic),
            };
            spec.get(ix, iy, iz) as f64
        };

        // Determine which 15N dimension (`A` or `B`) is the detected CON `N(i)` via diagonal + cross-peak energy
        let c_tol = (ax_c.step_ppm.abs() * 0.75).max(0.06);
        let na_tol = (ax_a.step_ppm.abs() * 0.75).max(0.45);
        let nb_tol = (ax_b.step_ppm.abs() * 0.75).max(0.45);
        let mut diag_a_fixed = 0.0f64; // A fixed at N(i), vary B
        let mut diag_b_fixed = 0.0f64; // B fixed at N(i), vary A
        for root in con_roots.iter().take(80) {
            let (ic, dc) = nearest_idx(ax_c, root.c_prev_ppm);
            if dc > c_tol {
                continue;
            }
            let (ia, _) = nearest_idx(ax_a, root.n_curr_ppm);
            let (ib, _) = nearest_idx(ax_b, root.n_curr_ppm);
            for jb in 0..ax_b.size {
                let s = get_voxel(ic, ia, jb) / noise - 2.0;
                if s > 0.0 {
                    diag_a_fixed += s;
                }
            }
            for ja in 0..ax_a.size {
                let s = get_voxel(ic, ja, ib) / noise - 2.0;
                if s > 0.0 {
                    diag_b_fixed += s;
                }
            }
        }
        let detected_is_a = diag_a_fixed >= diag_b_fixed;
        let ax_partner = if detected_is_a { ax_b } else { ax_a };
        let mut prof = vec![0.0f64; ax_partner.size];

        // Variance-normalized 2D orthogonal stencil: Var = 1.0 under white Gaussian noise
        let sample_2d_snr = |ic: usize, in_idx: usize, jp: usize, is_a: bool| -> f64 {
            let get = |c_i: usize, n_i: usize| -> f64 {
                if is_a {
                    get_voxel(c_i, n_i, jp)
                } else {
                    get_voxel(c_i, jp, n_i)
                }
            };
            let n_max = if is_a { ax_a.size } else { ax_b.size };
            let mut sum = get(ic, in_idx);
            let mut w2 = 1.0f64;
            let w_nbr = 0.15f64;
            for (dc_i, dn_i) in [(-1isize, 0isize), (1, 0), (0, -1), (0, 1)] {
                let ci = (ic as isize) + dc_i;
                let ni = (in_idx as isize) + dn_i;
                if ci >= 0 && (ci as usize) < ax_c.size && ni >= 0 && (ni as usize) < n_max {
                    sum += w_nbr * get(ci as usize, ni as usize);
                    w2 += w_nbr * w_nbr;
                }
            }
            sum / (noise * w2.sqrt())
        };

        for (r_idx, root) in con_roots.iter().enumerate() {
            let (ic, dc) = nearest_idx(ax_c, root.c_prev_ppm);
            if dc > c_tol {
                continue;
            }
            if detected_is_a {
                let (ia, dn) = nearest_idx(ax_a, root.n_curr_ppm);
                if dn > na_tol {
                    continue;
                }
                for (jb, slot) in prof.iter_mut().enumerate() {
                    *slot = sample_2d_snr(ic, ia, jb, true);
                }
            } else {
                let (ib, dn) = nearest_idx(ax_b, root.n_curr_ppm);
                if dn > nb_tol {
                    continue;
                }
                for (ja, slot) in prof.iter_mut().enumerate() {
                    *slot = sample_2d_snr(ic, ib, ja, false);
                }
            }
            strip_counts[r_idx] += 1;
            for (g_i, slot) in coadd_strips[r_idx].iter_mut().enumerate() {
                let n_ppm = GRID_LO + (g_i as f64) * GRID_STEP;
                *slot += sample_linear(ax_partner, &prof, n_ppm);
            }
        }
    }

    let partner_n_tol = cubes
        .iter()
        .map(|s| {
            let dx = s.axis_x.step_ppm.abs();
            let dy = s.axis_y.step_ppm.abs();
            let dz = s.axis_z.step_ppm.abs();
            (dx.max(dy).max(dz) * 0.75).max(0.35)
        })
        .fold(0.35f64, f64::max)
        .min(0.65);

    let thr = snr_threshold.max(2.0);
    for (r_idx, root) in con_roots.iter().enumerate() {
        let cnt = strip_counts[r_idx];
        if cnt == 0 {
            continue;
        }
        let inv_sqrt = 1.0 / (cnt as f64).sqrt();
        let sg = &mut coadd_strips[r_idx];
        for v in sg.iter_mut() {
            *v *= inv_sqrt;
        }
        let mut off_diag: Vec<(f64, f64)> = Vec::new();
        let mut near_diag: Vec<(f64, f64)> = Vec::new();
        let near_diag_partner_tol = partner_n_tol.min(0.25);
        for gi in 1..(GRID_LEN - 1) {
            let s = sg[gi];
            if s > thr && s >= sg[gi - 1] && s > sg[gi + 1] {
                let cand_n = GRID_LO + (gi as f64) * GRID_STEP;
                let d_diag = (cand_n - root.n_curr_ppm).abs();
                if d_diag <= 0.35 {
                    // Allow 0.0..=0.35 ppm near-diagonal links ONLY when another distinct CON root
                    // coincides with `cand_n` (e.g., Ser-repeat N(i-1) ≈ N(i)); weighted downstream
                    // in `idp_pair_matrix` so diagonals never crowd out resolved cross-peaks.
                    if con_roots.iter().enumerate().any(|(j, cr)| {
                        j != r_idx && (cand_n - cr.n_curr_ppm).abs() <= near_diag_partner_tol
                    }) {
                        near_diag.push((cand_n, s));
                    }
                    continue;
                }
                // Retain partner N candidates coinciding with at least one real CON N coordinate
                if con_roots
                    .iter()
                    .enumerate()
                    .any(|(j, cr)| j != r_idx && (cand_n - cr.n_curr_ppm).abs() <= partner_n_tol)
                {
                    off_diag.push((cand_n, s));
                }
            }
        }
        off_diag.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.total_cmp(&b.0)));
        off_diag.truncate(3);
        near_diag.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.total_cmp(&b.0)));
        near_diag.truncate(1);
        off_diag.extend(near_diag);
        out[r_idx] = off_diag;
    }

    out
}

/// Attach picked `(H)N(CA)CON` sequential `N(i-1)` partner candidates onto `ConDipeptideSystem`s (`RD-13`).
pub fn attach_hncacon_nn_links(
    systems: &mut [ConDipeptideSystem],
    nn_cands_by_root: &[Vec<(f64, f64)>],
) {
    for (sys, cands) in systems.iter_mut().zip(nn_cands_by_root.iter()) {
        sys.nn_prev_cands = cands.clone();
    }
}

/// Retain only bidirectional/reciprocal `(N_a, N_b)` amide pairs within `tol_ppm` (`RD-13`, `HN(CA)CONH`).
pub fn filter_reciprocal_nn_pairs(raw: &[(f64, f64)], tol_ppm: f64) -> Vec<(f64, f64)> {
    raw.iter()
        .copied()
        .filter(|&(a, b)| {
            raw.iter()
                .any(|&(a2, b2)| (a2 - b).abs() <= tol_ppm && (b2 - a).abs() <= tol_ppm)
        })
        .collect()
}

/// Extract reciprocal `HN(CA)CONH` `N-N` adjacency pairs `(N_a, N_b)` from a `3D` `(1H, 15N, 15N)`
/// cube given known amide `(H, N)` roots (`RD-13`).
pub fn extract_reciprocal_hncaconh_pairs(
    spec: &Spectrum3D,
    amide_roots: &[(f64, f64)],
    snr_threshold: f64,
) -> Vec<(f64, f64)> {
    if amide_roots.is_empty() {
        return Vec::new();
    }
    let noise = (estimate_noise_mad(&spec.data) as f64).max(1e-9);
    let is_h_axis = |ax: &crate::spectrum::AxisPpm| {
        let nuc = ax.label.to_ascii_uppercase();
        if nuc.contains('H') {
            return true;
        }
        if nuc.contains('N') {
            return false;
        }
        ax.index_to_ppm((ax.size as f64) * 0.5) < 30.0
    };
    let h_dim = if is_h_axis(&spec.axis_x) {
        0
    } else if is_h_axis(&spec.axis_y) {
        1
    } else {
        2
    };
    let (n_dim_a, n_dim_b) = match h_dim {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    };
    let ax_h = match h_dim {
        0 => &spec.axis_x,
        1 => &spec.axis_y,
        _ => &spec.axis_z,
    };
    let ax_a = match n_dim_a {
        0 => &spec.axis_x,
        1 => &spec.axis_y,
        _ => &spec.axis_z,
    };
    let ax_b = match n_dim_b {
        0 => &spec.axis_x,
        1 => &spec.axis_y,
        _ => &spec.axis_z,
    };
    let get_voxel = |ih: usize, ia: usize, ib: usize| -> f64 {
        let (ix, iy, iz) = match h_dim {
            0 => (ih, ia, ib),
            1 => (ia, ih, ib),
            _ => (ia, ib, ih),
        };
        spec.get(ix, iy, iz) as f64
    };
    let nearest_idx = |ax: &crate::spectrum::AxisPpm, ppm: f64| -> (usize, f64) {
        let idx_f = ax.ppm_to_index(ppm);
        let max_i = ax.size.saturating_sub(1);
        let i = (idx_f.round() as isize).clamp(0, max_i as isize) as usize;
        let d = (ax.index_to_ppm(i as f64) - ppm).abs();
        (i, d)
    };

    // First determine which 15N dimension (A or B) is the detected amide N(i) coupled to H(i)
    // by comparing strip energy along the two orientations across amide_roots. Scanning the
    // non-detected axis at root's own H(i) would fix the partner axis at N(i) and yield spurious peaks.
    let mut energy_a_fixed = 0.0f64;
    let mut energy_b_fixed = 0.0f64;
    for &(rh, rn) in amide_roots.iter().take(80) {
        let (ih, dh) = nearest_idx(ax_h, rh);
        if dh > 0.03 {
            continue;
        }
        let (ia, dna) = nearest_idx(ax_a, rn);
        if dna <= 0.45 {
            for jb in 0..ax_b.size {
                let s = get_voxel(ih, ia, jb) / noise - 2.5;
                if s > 0.0 {
                    energy_a_fixed += s;
                }
            }
        }
        let (ib, dnb) = nearest_idx(ax_b, rn);
        if dnb <= 0.45 {
            for ja in 0..ax_a.size {
                let s = get_voxel(ih, ja, ib) / noise - 2.5;
                if s > 0.0 {
                    energy_b_fixed += s;
                }
            }
        }
    }
    let detected_is_a = energy_a_fixed >= energy_b_fixed;

    let mut raw: Vec<(f64, f64)> = Vec::new();
    let thr = snr_threshold.max(3.5) * noise;
    for &(rh, rn) in amide_roots {
        let (ih, dh) = nearest_idx(ax_h, rh);
        if dh > 0.03 {
            continue;
        }
        let (ih_lo, ih_hi) = (
            ih.saturating_sub(1),
            (ih + 1).min(ax_h.size.saturating_sub(1)),
        );

        if detected_is_a {
            let (ia, dna) = nearest_idx(ax_a, rn);
            if dna <= 0.45 && ax_b.size >= 3 {
                let (ia_lo, ia_hi) = (
                    ia.saturating_sub(1),
                    (ia + 1).min(ax_a.size.saturating_sub(1)),
                );
                let line: Vec<f64> = (0..ax_b.size)
                    .map(|jb| {
                        let mut mx = f64::NEG_INFINITY;
                        for ih_i in ih_lo..=ih_hi {
                            for ia_i in ia_lo..=ia_hi {
                                mx = mx.max(get_voxel(ih_i, ia_i, jb));
                            }
                        }
                        mx
                    })
                    .collect();
                let mut cands: Vec<(f64, f64)> = (1..(line.len() - 1))
                    .filter_map(|j| {
                        let n_other = ax_b.index_to_ppm(j as f64);
                        (line[j] > thr
                            && line[j] >= line[j - 1]
                            && line[j] >= line[j + 1]
                            && (n_other - rn).abs() > 0.5)
                            .then_some((line[j], n_other))
                    })
                    .collect();
                cands.sort_by(|x, y| y.0.total_cmp(&x.0));
                for (_, v) in cands.into_iter().take(2) {
                    raw.push((rn, v));
                }
            }
        } else {
            let (ib, dnb) = nearest_idx(ax_b, rn);
            if dnb <= 0.45 && ax_a.size >= 3 {
                let (ib_lo, ib_hi) = (
                    ib.saturating_sub(1),
                    (ib + 1).min(ax_b.size.saturating_sub(1)),
                );
                let line: Vec<f64> = (0..ax_a.size)
                    .map(|ja| {
                        let mut mx = f64::NEG_INFINITY;
                        for ih_i in ih_lo..=ih_hi {
                            for ib_i in ib_lo..=ib_hi {
                                mx = mx.max(get_voxel(ih_i, ja, ib_i));
                            }
                        }
                        mx
                    })
                    .collect();
                let mut cands: Vec<(f64, f64)> = (1..(line.len() - 1))
                    .filter_map(|j| {
                        let n_other = ax_a.index_to_ppm(j as f64);
                        (line[j] > thr
                            && line[j] >= line[j - 1]
                            && line[j] >= line[j + 1]
                            && (n_other - rn).abs() > 0.5)
                            .then_some((line[j], n_other))
                    })
                    .collect();
                cands.sort_by(|x, y| y.0.total_cmp(&x.0));
                for (_, v) in cands.into_iter().take(2) {
                    raw.push((rn, v));
                }
            }
        }
    }

    filter_reciprocal_nn_pairs(&raw, 0.35)
}

/// Unlabelled per-root residue typing summary from `CON` roots + `CAS` + `H(CC)CON` `(i-1)` + `HNCO->HNCACB` `(i)` (`RD-10`).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct IdpCasResidueTypingReport {
    pub n_con_roots: usize,
    pub cacon_hit_counts: BTreeMap<String, usize>,
    pub canco_hit_counts: BTreeMap<String, usize>,
    pub roots_typed_prev: BTreeMap<String, usize>,
    pub roots_typed_curr: BTreeMap<String, usize>,
}

/// Type individual `CON` dipeptide roots `(C'(i-1), N(i))` unlabelled against `CAS` subspectra,
/// `H(CC)CON` `(i-1)` `CA/CB`, and `HNCO->HNCACB` `(i)` `CA/CB` (`RD-10`).
pub fn summarize_idp_cas_residue_typing(
    systems: &[ConDipeptideSystem],
    cas_spectra: &[CasSpectrum],
    cfg: &IdpSolverConfig,
    offsets: &IdpShiftOffsets,
) -> IdpCasResidueTypingReport {
    let n_con = systems.len();
    let mut rep = IdpCasResidueTypingReport {
        n_con_roots: n_con,
        ..IdpCasResidueTypingReport::default()
    };
    if systems.is_empty() {
        return rep;
    }

    let coords: Vec<(f64, f64)> = systems.iter().map(|s| (s.c_prev, s.n_curr)).collect();
    let mut hits_per_cas: Vec<Vec<bool>> = Vec::with_capacity(cas_spectra.len());
    for cas in cas_spectra {
        let hits = match_cas_peaks_to_roots(cas, &coords, cfg.cas_c_tol_ppm, cfg.cas_n_tol_ppm);
        let cnt = hits.iter().filter(|&&h| h).count();
        let target_map = match cas.mode {
            CasMode::Cacon => &mut rep.cacon_hit_counts,
            CasMode::Canco => &mut rep.canco_hit_counts,
        };
        *target_map.entry(cas.name.clone()).or_insert(0) += cnt;
        hits_per_cas.push(hits);
    }

    let aa_all = [
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET",
        "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    ];
    let rc_preds = predict_ncidp_shifts(
        &aa_all.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
        cfg,
    );

    for (s_idx, sys) in systems.iter().enumerate() {
        let has_prev_info = !sys.ca_prev.is_empty()
            || !sys.cb_prev.is_empty()
            || cas_spectra
                .iter()
                .zip(hits_per_cas.iter())
                .any(|(c, h)| c.mode == CasMode::Cacon && h[s_idx]);
        if has_prev_info {
            let mut best_aa = None;
            let mut best_sc = f64::NEG_INFINITY;
            for (aa_i, aa) in aa_all.iter().enumerate() {
                let p = &rc_preds[aa_i];
                let mut sc = -0.5
                    * (((sys.c_prev - offsets.d_c) - p.c_ppm) / 1.2)
                        .powi(2)
                        .min(6.0);
                if !sys.ca_prev.is_empty() {
                    let min_dca = sys
                        .ca_prev
                        .iter()
                        .map(|&v| ((v - offsets.d_ca) - p.ca_ppm).abs())
                        .fold(f64::INFINITY, f64::min);
                    sc -= 0.5 * (min_dca / 0.85).powi(2).min(8.0);
                }
                if let Some(mu_cb) = p.cb_ppm {
                    if !sys.cb_prev.is_empty() {
                        let min_dcb = sys
                            .cb_prev
                            .iter()
                            .map(|&v| ((v - offsets.d_cb) - mu_cb).abs())
                            .fold(f64::INFINITY, f64::min);
                        sc -= 0.5 * (min_dcb / 0.95).powi(2).min(8.0);
                    }
                } else if !sys.cb_prev.is_empty() {
                    sc -= 4.5;
                }
                for (cas, hits) in cas_spectra.iter().zip(hits_per_cas.iter()) {
                    if cas.mode == CasMode::Cacon && hits[s_idx] {
                        let n_hits = hits.iter().filter(|&&h| h).count();
                        sc += cas_llr_adjustment(cas, true, aa, cfg.cas_match_llr, 0, n_hits);
                    }
                }
                if sc > best_sc {
                    best_sc = sc;
                    best_aa = Some(*aa);
                }
            }
            if let Some(aa) = best_aa {
                *rep.roots_typed_prev.entry(aa.to_string()).or_insert(0) += 1;
            }
        }

        let has_curr_info = !sys.ca_curr.is_empty()
            || !sys.cb_curr.is_empty()
            || sys.n_curr >= 131.5
            || cas_spectra
                .iter()
                .zip(hits_per_cas.iter())
                .any(|(c, h)| c.mode == CasMode::Canco && h[s_idx]);
        if has_curr_info {
            let mut best_aa = None;
            let mut best_sc = f64::NEG_INFINITY;
            for (aa_i, aa) in aa_all.iter().enumerate() {
                let p = &rc_preds[aa_i];
                let sig_n = if *aa == "PRO" { 1.4 } else { 2.2 };
                let mut sc = -0.5
                    * (((sys.n_curr - offsets.d_n) - p.n_ppm) / sig_n)
                        .powi(2)
                        .min(8.0);
                if !sys.ca_curr.is_empty() {
                    let min_dca = sys
                        .ca_curr
                        .iter()
                        .map(|&v| ((v - offsets.d_ca) - p.ca_ppm).abs())
                        .fold(f64::INFINITY, f64::min);
                    sc -= 0.5 * (min_dca / 0.85).powi(2).min(8.0);
                }
                if let Some(mu_cb) = p.cb_ppm {
                    if !sys.cb_curr.is_empty() {
                        let min_dcb = sys
                            .cb_curr
                            .iter()
                            .map(|&v| ((v - offsets.d_cb) - mu_cb).abs())
                            .fold(f64::INFINITY, f64::min);
                        sc -= 0.5 * (min_dcb / 0.95).powi(2).min(8.0);
                    }
                } else if !sys.cb_curr.is_empty() {
                    sc -= 4.5;
                }
                for (cas, hits) in cas_spectra.iter().zip(hits_per_cas.iter()) {
                    if cas.mode == CasMode::Canco && hits[s_idx] {
                        let n_hits = hits.iter().filter(|&&h| h).count();
                        sc += cas_llr_adjustment(cas, true, aa, cfg.cas_match_llr, 0, n_hits);
                    }
                }
                if sc > best_sc {
                    best_sc = sc;
                    best_aa = Some(*aa);
                }
            }
            if let Some(aa) = best_aa {
                *rep.roots_typed_curr.entry(aa.to_string()).or_insert(0) += 1;
            }
        }
    }

    rep
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::c13_detected::{CasPeak, Con2DPeak};
    use crate::spectrum::{AxisPpm, Spectrum3D};

    #[test]
    fn test_rd4_rd10_idp_cas_and_hnco_hncacb_improves_hc_and_accuracy() {
        // Diverse 24-residue IDP with Prolines and ambiguous pairs (e.g. GLU/GLN/LYS, SER/THR, ALA/VAL)
        // where local random-coil shift overlaps confuse CON + H(CC)CON without CAS / HNCO->HNCACB.
        let seq: Vec<String> = [
            "MET", "SER", "ALA", "GLU", "LEU", "PRO", "THR", "VAL", "ASP", "GLY", "SER", "GLN",
            "PHE", "ALA", "LYS", "ARG", "THR", "ILE", "PRO", "ASN", "HIS", "GLY", "GLU", "TYR",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let base_cfg = IdpSolverConfig::default();
        let mut preds = predict_ncidp_shifts(&seq, &base_cfg);

        // Introduce realistic IDP shift overlaps on pairs of Ser/Ala/Gly/Glu so baseline without
        // CAS / HNCO->HNCACB cannot rely solely on isolated C'/N random-coil uniqueness:
        // Move bond (SER2->ALA3) close to (SER11->GLN12), (THR7->VAL8) close to (THR17->ILE18),
        // and swap GLU4 vs GLU23 C'/CA slightly within random coil variance.
        let ave_c_ser = 0.5 * (preds[1].c_ppm + preds[10].c_ppm);
        preds[1].c_ppm = ave_c_ser + 0.015;
        preds[10].c_ppm = ave_c_ser - 0.015;
        preds[2].n_ppm = 123.40;
        preds[11].n_ppm = 123.46;
        preds[2].ca_ppm = preds[11].ca_ppm - 0.15;
        preds[11].ca_ppm = preds[2].ca_ppm + 0.15;

        let ave_c_glu = 0.5 * (preds[3].c_ppm + preds[22].c_ppm);
        preds[3].c_ppm = ave_c_glu - 0.02;
        preds[22].c_ppm = ave_c_glu + 0.02;
        preds[4].n_ppm = 121.80;
        preds[23].n_ppm = 121.84;

        let mut systems = Vec::new();
        let mut cacon_ser_pks = Vec::new();
        let mut cacon_ala_pks = Vec::new();
        let mut cacon_gly_pks = Vec::new();
        let mut canco_ser_pks = Vec::new();
        let mut canco_ala_pks = Vec::new();
        let mut canco_gln_pks = Vec::new();
        let mut hnco_pks = Vec::new();
        let mut hncacb_pks = Vec::new();

        for k in 1..seq.len() {
            let prv = &preds[k - 1];
            let cur = &preds[k];
            let c_prev = prv.c_ppm;
            let n_curr = cur.n_ppm;
            let mut s = ConDipeptideSystem::new(k - 1, c_prev, n_curr, 100.0);
            // H(CC)CON i-1 provides CA(i-1) and CB(i-1) (including pre-Pro!)
            s.ca_prev.push(prv.ca_ppm);
            if let Some(cb) = prv.cb_ppm {
                s.cb_prev.push(cb);
            }
            s.has_cbcacon_evidence = true;
            // Partial sequential CANCO CA(i) present (~70% complete), missing around overlapping pairs
            if !matches!(k, 2 | 4 | 11 | 12 | 23) && seq[k] != "PRO" {
                s.ca_curr.push(cur.ca_ppm);
            }
            systems.push(s);

            if seq[k] != "PRO" {
                let h_i = 8.10 + 0.04 * ((k % 9) as f64);
                hnco_pks
                    .push(Peak::triple("HNCO", h_i, n_curr, c_prev, 1).with_intensity(90.0, None));
                // Inverted HNCACB signs (CB positive, CA negative) to exercise RD-1 inside RD-10
                hncacb_pks.push(
                    Peak::triple("HNCACB", h_i, n_curr, cur.ca_ppm, -1).with_intensity(95.0, None),
                );
                if let Some(cb) = cur.cb_ppm {
                    hncacb_pks.push(
                        Peak::triple("HNCACB", h_i, n_curr, cb, 1).with_intensity(85.0, None),
                    );
                }
            }

            let cpk = CasPeak {
                c_prime_ppm: c_prev,
                n_ppm: n_curr,
                height: 100.0,
                volume: 100.0,
            };
            match seq[k - 1].as_str() {
                "SER" => cacon_ser_pks.push(cpk.clone()),
                "ALA" => cacon_ala_pks.push(cpk.clone()),
                "GLY" => cacon_gly_pks.push(cpk.clone()),
                _ => {}
            }
            match seq[k].as_str() {
                "SER" => canco_ser_pks.push(cpk),
                "ALA" => canco_ala_pks.push(cpk),
                "GLN" => canco_gln_pks.push(cpk),
                _ => {}
            }
        }

        // Baseline: no CAS and no HNCO->HNCACB
        let res_no_cas = assign_idp_13c_cpu(&systems, &seq, &base_cfg);
        let acc_no_cas = (0..systems.len())
            .filter(|&i| res_no_cas.position[i] == ((i + 1) as i32))
            .count();
        let hc_no_cas = res_no_cas.confidence.iter().filter(|&&c| c >= 0.80).count();

        let cas = vec![
            CasSpectrum {
                name: "cacon_ser".to_string(),
                mode: CasMode::Cacon,
                residues: vec!["SER".to_string()],
                peaks: cacon_ser_pks,
            },
            CasSpectrum {
                name: "cacon_ala".to_string(),
                mode: CasMode::Cacon,
                residues: vec!["ALA".to_string()],
                peaks: cacon_ala_pks,
            },
            CasSpectrum {
                name: "cacon_gly".to_string(),
                mode: CasMode::Cacon,
                residues: vec!["GLY".to_string()],
                peaks: cacon_gly_pks,
            },
            CasSpectrum {
                name: "canco_ser".to_string(),
                mode: CasMode::Canco,
                residues: vec!["SER".to_string()],
                peaks: canco_ser_pks,
            },
            CasSpectrum {
                name: "canco_ala".to_string(),
                mode: CasMode::Canco,
                residues: vec!["ALA".to_string()],
                peaks: canco_ala_pks,
            },
            CasSpectrum {
                name: "canco_gln".to_string(),
                mode: CasMode::Canco,
                residues: vec!["GLN".to_string()],
                peaks: canco_gln_pks,
            },
        ];
        let mut cfg_cas = base_cfg.clone();
        cfg_cas.cas_spectra = cas.clone();

        // Check RD-4 CAS-only typing & accuracy/HC improvement first
        let typing =
            summarize_idp_cas_residue_typing(&systems, &cas, &cfg_cas, &IdpShiftOffsets::default());
        assert_eq!(typing.cacon_hit_counts.get("cacon_ser"), Some(&2));
        assert_eq!(typing.cacon_hit_counts.get("cacon_ala"), Some(&2));
        assert_eq!(typing.cacon_hit_counts.get("cacon_gly"), Some(&2));
        let res_cas_only = assign_idp_13c_cpu(&systems, &seq, &cfg_cas);
        let acc_cas_only = (0..systems.len())
            .filter(|&i| res_cas_only.position[i] == ((i + 1) as i32))
            .count();
        let hc_cas_only = res_cas_only
            .confidence
            .iter()
            .filter(|&&c| c >= 0.80)
            .count();

        // RD-10 + RD-4: also attach HNCO->HNCACB i correlations
        let mut sys_rd10 = systems.clone();
        let link_stats = attach_hnco_hncacb_to_con_systems(
            &mut sys_rd10,
            &hnco_pks,
            &hncacb_pks,
            HncacbPolarity::Auto,
        );
        assert!(link_stats.n_con_roots_matched_hnco >= 18);
        assert!(link_stats.n_roots_with_hncacb_curr >= 18);

        let res_cas = assign_idp_13c_cpu(&sys_rd10, &seq, &cfg_cas);
        let acc_cas = (0..sys_rd10.len())
            .filter(|&i| res_cas.position[i] == ((i + 1) as i32))
            .count();
        let hc_cas = res_cas.confidence.iter().filter(|&&c| c >= 0.80).count();
        assert!(
            acc_cas_only >= acc_no_cas
                && hc_cas_only >= hc_no_cas
                && acc_cas >= acc_no_cas
                && (acc_cas > acc_no_cas || hc_cas > hc_no_cas)
                && hc_cas > hc_no_cas,
            "CAS + HNCO->HNCACB must improve accuracy & HC count: base=({acc_no_cas}, {hc_no_cas}) cas_only=({acc_cas_only}, {hc_cas_only}) full=({acc_cas}, {hc_cas})"
        );
    }

    #[test]
    fn test_rd13_coadded_hncacon_and_reciprocal_hncaconh_links() {
        // Synthetic 6-residue IDP: MET-ALA-SER-THR-PRO-GLU
        let seq: Vec<String> = ["MET", "ALA", "SER", "THR", "PRO", "GLU"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let base_cfg = IdpSolverConfig::default();
        let preds = predict_ncidp_shifts(&seq, &base_cfg);

        // Give distinct 15N values so N-N linking is clean
        let n_vals = [118.0, 124.2, 115.8, 113.5, 137.0, 121.0];
        let c_vals = [174.5, 177.2, 174.8, 173.1, 176.2, 175.0];

        let mut con_roots = Vec::new();
        let mut systems = Vec::new();
        for k in 1..seq.len() {
            con_roots.push(Con2DPeak {
                c_prev_ppm: c_vals[k - 1],
                n_curr_ppm: n_vals[k],
                height: 100.0,
                volume: 100.0,
                is_proline_imino_region: seq[k] == "PRO",
                deconvolved_from_cluster: false,
            });
            let mut s = ConDipeptideSystem::new(k - 1, c_vals[k - 1], n_vals[k], 100.0);
            s.ca_prev.push(preds[k - 1].ca_ppm);
            if let Some(cb) = preds[k - 1].cb_ppm {
                s.cb_prev.push(cb);
            }
            s.has_cbcacon_evidence = true;
            systems.push(s);
        }

        // Build two replicate 3D (H)N(CA)CON cubes (`F1=15N_partner`, `F2=15N_con`, `F3=13C'`)
        // where single-cube SNR is ~3.0 (below 3.5 threshold), but co-added SNR is ~4.2 > 3.5!
        let nx = 32usize; // C' 170..180
        let ny = 60usize; // N_con 110..140
        let nz = 60usize; // N_partner 110..140
        let ax_c = AxisPpm::from_bounds("13C", nx, 180.0, 170.0, 150.0);
        let ax_n2 = AxisPpm::from_bounds("15N", ny, 140.0, 110.0, 60.8);
        let ax_n1 = AxisPpm::from_bounds("15N", nz, 140.0, 110.0, 60.8);

        let make_cube = |seed: u64| -> Spectrum3D {
            let mut sp = Spectrum3D {
                name: "HNCACON".to_string(),
                exp: "HNCACON".to_string(),
                axis_x: ax_c.clone(),
                axis_y: ax_n2.clone(),
                axis_z: ax_n1.clone(),
                data: vec![0.0f32; nx * ny * nz],
            };
            let mut rng = Rng::new(seed);
            for v in &mut sp.data {
                *v = rng.normal(0.0, 1.0) as f32;
            }
            for k in 1..con_roots.len() {
                let ic = (ax_c.ppm_to_index(c_vals[k]).round() as usize).min(nx - 1);
                let in_con = (ax_n2.ppm_to_index(n_vals[k + 1]).round() as usize).min(ny - 1);
                // Diagonal intensity on N_con == n_vals[k+1]
                let diag_z = (ax_n1.ppm_to_index(n_vals[k + 1]).round() as usize).min(nz - 1);
                sp.data[(diag_z * ny + in_con) * nx + ic] = 8.0;
                // Sequential cross-peak at partner = n_vals[k] (SNR ~ 3.1 per cube -> ~4.4 coadded)
                let part_z = (ax_n1.ppm_to_index(n_vals[k]).round() as usize).min(nz - 1);
                sp.data[(part_z * ny + in_con) * nx + ic] = 3.1;
            }
            sp
        };

        let cube1 = make_cube(11);
        let cube2 = make_cube(22);
        let links_1 = pick_hncacon_nn_coadded(&[&cube1], &con_roots, 3.5);
        assert_eq!(links_1.iter().filter(|v| !v.is_empty()).count(), 0);
        let links_2 = pick_hncacon_nn_coadded(&[&cube1, &cube2], &con_roots, 3.5);
        assert!(
            links_2.iter().filter(|v| !v.is_empty()).count() >= 3,
            "co-added sqrt(2) SNR boost must recover sequential N-N partners: {links_2:?}"
        );

        attach_hncacon_nn_links(&mut systems, &links_2);
        let raw_recip = vec![
            (n_vals[1], n_vals[2]),
            (n_vals[2], n_vals[1]),
            (n_vals[2], 129.9), // unreciprocated outlier must be filtered out
        ];
        let recip = filter_reciprocal_nn_pairs(&raw_recip, 0.35);
        assert_eq!(recip.len(), 2);

        let mut cfg_nn = base_cfg;
        cfg_nn.reciprocal_nn_pairs = recip;
        cfg_nn.replicas = 24;
        cfg_nn.steps_per_system = 1200;
        let res = assign_idp_13c_cpu(&systems, &seq, &cfg_nn);
        assert_eq!(
            res.position,
            vec![1, 2, 3, 4, 5],
            "expected continuous chain through X-Pro via HNCACON N-N links"
        );
    }

    #[test]
    fn test_p2_idp_auto_ref_atomic_refusal_ns_spread_and_qc_notes() {
        let mut systems = Vec::new();
        for i in 0..10 {
            let mut s =
                ConDipeptideSystem::new(i, 174.0 + 0.3 * (i as f64), 120.0 + (i as f64), 100.0);
            s.ca_prev.push(55.0);
            s.cb_prev.push(30.0);
            systems.push(s);
        }

        // 1. Atomic carbon refusal even when spread <= 1.0 ppm (CA refused at -3.8 -> C'/CB also zeroed)
        let partial_refuse = IdpShiftOffsets {
            d_n: -0.3,
            d_c: -3.2,
            d_ca: -3.8,
            d_cb: -3.1,
        };
        let sf_atomic = evaluate_idp_auto_reference_safety(&systems, &partial_refuse);
        assert!(sf_atomic.refused_nuclei.contains(&"13CA".to_string()));
        assert_eq!(sf_atomic.safe_offsets.d_c, 0.0);
        assert_eq!(sf_atomic.safe_offsets.d_ca, 0.0);
        assert_eq!(sf_atomic.safe_offsets.d_cb, 0.0);
        assert_eq!(sf_atomic.safe_offsets.d_n, -0.3);

        // 2. NotSignificant (0.0) carbon channel excluded from spread -> TMS hint preserved, no false polarity warning
        let tms_ns_cb = IdpShiftOffsets {
            d_n: 0.0,
            d_c: -2.62,
            d_ca: -2.58,
            d_cb: 0.0,
        };
        let sf_ns = evaluate_idp_auto_reference_safety(&systems, &tms_ns_cb);
        assert!(
            sf_ns.c13_spread_ppm < 0.1,
            "NotSignificant CB=0.0 must not inflate carbon spread: {sf_ns:?}"
        );
        assert!(sf_ns.polarity_swap_warning.is_none());
        assert!(
            sf_ns
                .tms_reference_hint
                .as_deref()
                .is_some_and(|h| h.contains("TMS")),
            "expected TMS hint when CB=0.0 is excluded: {sf_ns:?}"
        );

        // 3. Moderate spread in 1.0..1.8 ppm emits warning
        let mid_spread = IdpShiftOffsets {
            d_n: 0.0,
            d_c: -1.60,
            d_ca: -0.20,
            d_cb: -0.25,
        };
        let sf_mid = evaluate_idp_auto_reference_safety(&systems, &mid_spread);
        assert!(
            (1.0..=1.8).contains(&sf_mid.c13_spread_ppm),
            "expected 1.0..1.8 spread, got {}",
            sf_mid.c13_spread_ppm
        );
        assert!(sf_mid.polarity_swap_warning.is_some());
        assert!(!sf_mid.warnings.is_empty());
    }

    #[test]
    fn test_p2_hnco_hncacb_intensity_and_hncocacb_prevents_im1_leak() {
        // Two CON roots whose ca_prev/cb_prev are EMPTY:
        // - Root 0: HNCACB has strong 1J CA(i)/CB(i) and weaker 2J CA(i-1)/CB(i-1) -> intensity separates them!
        // - Root 1: HNCACB has equal-height CA/CB doublets + HN(CO)CACB corroborates i-1 -> HN(CO)CACB separates them!
        // - Root 2: nearly coincides with Root 0's HNCO (1-to-1 matching prevents stealing Root 0's HNCO!)
        let mut systems = vec![
            ConDipeptideSystem::new(0, 175.00, 121.00, 100.0),
            ConDipeptideSystem::new(1, 176.20, 118.40, 100.0),
            ConDipeptideSystem::new(2, 175.05, 121.12, 90.0),
        ];

        let hnco_pks = vec![
            Peak::triple("HNCO", 8.25, 121.00, 175.00, 1).with_intensity(100.0, None),
            Peak::triple("HNCO", 8.42, 118.40, 176.20, 1).with_intensity(100.0, None),
        ];
        let hncacb_pks = vec![
            // Root 0: strong intra CA=58.2 / CB=64.0 (height 100/95), weaker i-1 CA=52.1 / CB=19.0 (height 45/40)
            Peak::triple("HNCACB", 8.25, 121.00, 58.2, 1).with_intensity(100.0, None),
            Peak::triple("HNCACB", 8.25, 121.00, 52.1, 1).with_intensity(45.0, None),
            Peak::triple("HNCACB", 8.25, 121.00, 64.0, -1).with_intensity(95.0, None),
            Peak::triple("HNCACB", 8.25, 121.00, 19.0, -1).with_intensity(40.0, None),
            // Root 1: equal-height doublet, disambiguated by HN(CO)CACB i-1 peaks at 56.4 / 30.2
            Peak::triple("HNCACB", 8.42, 118.40, 61.5, 1).with_intensity(80.0, None),
            Peak::triple("HNCACB", 8.42, 118.40, 56.4, 1).with_intensity(82.0, None),
            Peak::triple("HNCACB", 8.42, 118.40, 69.2, -1).with_intensity(80.0, None),
            Peak::triple("HNCACB", 8.42, 118.40, 30.2, -1).with_intensity(81.0, None),
            Peak::triple("HNCOCACB", 8.42, 118.40, 56.42, 1).with_intensity(90.0, None),
            Peak::triple("HNCOCACB", 8.42, 118.40, 30.18, 1).with_intensity(88.0, None),
        ];

        let stats = attach_hnco_hncacb_to_con_systems(
            &mut systems,
            &hnco_pks,
            &hncacb_pks,
            HncacbPolarity::CbNegative,
        );
        // 1-to-1 matching: only 2 HNCO peaks exist so at most 2 CON roots match
        assert_eq!(stats.n_con_roots_matched_hnco, 2);
        assert!(systems[2].ca_curr.is_empty() && systems[2].cb_curr.is_empty());

        // Root 0: only strong intra CA=58.2 / CB=64.0 enter ca_curr/cb_curr; weaker i-1 goes to prev
        assert_eq!(systems[0].ca_curr, vec![58.2]);
        assert_eq!(systems[0].cb_curr, vec![64.0]);
        assert_eq!(systems[0].ca_prev, vec![52.1]);
        assert_eq!(systems[0].cb_prev, vec![19.0]);

        // Root 1: HN(CO)CACB blocks 56.4 / 30.2 from leaking into ca_curr/cb_curr
        assert_eq!(systems[1].ca_curr, vec![61.5]);
        assert_eq!(systems[1].cb_curr, vec![69.2]);
        assert_eq!(systems[1].ca_prev, vec![56.4]);
        assert_eq!(systems[1].cb_prev, vec![30.2]);
    }

    #[test]
    fn test_p2_hncacon_near_diagonal_and_hncaconh_detected_axis() {
        // 1. Near-diagonal (H)N(CA)CON Ser-repeat link (`|N(i-1) - N(i)| = 0.20 ppm <= 0.35 ppm`)
        let con_roots = vec![
            Con2DPeak {
                c_prev_ppm: 174.2,
                n_curr_ppm: 116.20,
                height: 100.0,
                volume: 100.0,
                is_proline_imino_region: false,
                deconvolved_from_cluster: false,
            },
            Con2DPeak {
                c_prev_ppm: 175.4,
                n_curr_ppm: 116.40, // 0.20 ppm from root 0 -> Ser-Ser near-diagonal!
                height: 100.0,
                volume: 100.0,
                is_proline_imino_region: false,
                deconvolved_from_cluster: false,
            },
            Con2DPeak {
                c_prev_ppm: 176.5,
                n_curr_ppm: 124.80, // isolated N -> autopeak must NOT become a fake link
                height: 100.0,
                volume: 100.0,
                is_proline_imino_region: false,
                deconvolved_from_cluster: false,
            },
        ];

        let nx = 24usize;
        let ny = 80usize;
        let nz = 80usize;
        let ax_c = AxisPpm::from_bounds("13C", nx, 179.0, 171.0, 150.0);
        let ax_n2 = AxisPpm::from_bounds("15N", ny, 128.0, 112.0, 60.8);
        let ax_n1 = AxisPpm::from_bounds("15N", nz, 128.0, 112.0, 60.8);
        let mut cube = Spectrum3D {
            name: "HNCACON".to_string(),
            exp: "HNCACON".to_string(),
            axis_x: ax_c.clone(),
            axis_y: ax_n2.clone(),
            axis_z: ax_n1.clone(),
            data: vec![0.0f32; nx * ny * nz],
        };
        let mut rng = Rng::new(42);
        for v in &mut cube.data {
            *v = rng.normal(0.0, 1.0) as f32;
        }
        // Populate autopeaks for all 3 roots plus near-diagonal cross-peak at root 1 -> 116.20
        for r in &con_roots {
            let ic = (ax_c.ppm_to_index(r.c_prev_ppm).round() as usize).min(nx - 1);
            let in_c = (ax_n2.ppm_to_index(r.n_curr_ppm).round() as usize).min(ny - 1);
            let diag_z = (ax_n1.ppm_to_index(r.n_curr_ppm).round() as usize).min(nz - 1);
            cube.data[(diag_z * ny + in_c) * nx + ic] = 9.0;
        }
        let ic1 = (ax_c.ppm_to_index(con_roots[1].c_prev_ppm).round() as usize).min(nx - 1);
        let in1 = (ax_n2.ppm_to_index(con_roots[1].n_curr_ppm).round() as usize).min(ny - 1);
        let part0 = (ax_n1.ppm_to_index(con_roots[0].n_curr_ppm).round() as usize).min(nz - 1);
        cube.data[(part0 * ny + in1) * nx + ic1] = 8.5;

        let links = pick_hncacon_nn_coadded(&[&cube], &con_roots, 3.5);
        assert!(
            links[1].iter().any(|&(n, _)| (n - 116.20).abs() <= 0.25),
            "near-diagonal 0.20 ppm Ser-repeat link must be retained: {:?}",
            links[1]
        );
        assert!(
            links[2].is_empty(),
            "isolated root autopeak with no nearby CON partner must stay empty: {:?}",
            links[2]
        );

        // 2. HN(CA)CONH detected-axis orientation: axis_y is detected N(i), axis_z is partner N(i±1).
        // Inject true reciprocal pair (118.0 <-> 122.0) on (H_0, N_y=118, N_z=122) & (H_1, N_y=122, N_z=118),
        // PLUS a spurious peak at (H_0, N_y=126, N_z=118) that only broken Direction 2 would pick up at H_0!
        let ax_h = AxisPpm::from_bounds("1H", 40, 9.2, 7.2, 600.0);
        let ax_na = AxisPpm::from_bounds("15N", 60, 130.0, 112.0, 60.8);
        let ax_nb = AxisPpm::from_bounds("15N", 60, 130.0, 112.0, 60.8);
        let mut hncaconh = Spectrum3D {
            name: "HNCACONH".to_string(),
            exp: "HNCACONH".to_string(),
            axis_x: ax_h.clone(),
            axis_y: ax_na.clone(),
            axis_z: ax_nb.clone(),
            data: vec![0.0f32; 40 * 60 * 60],
        };
        for v in &mut hncaconh.data {
            *v = rng.normal(0.0, 0.5) as f32;
        }
        let put = |sp: &mut Spectrum3D, h: f64, na: f64, nb: f64, val: f32| {
            let ix = (sp.axis_x.ppm_to_index(h).round() as usize).min(sp.axis_x.size - 1);
            let iy = (sp.axis_y.ppm_to_index(na).round() as usize).min(sp.axis_y.size - 1);
            let iz = (sp.axis_z.ppm_to_index(nb).round() as usize).min(sp.axis_z.size - 1);
            sp.data[(iz * sp.axis_y.size + iy) * sp.axis_x.size + ix] = val;
        };
        let roots = vec![(8.20, 118.0), (8.55, 122.0), (7.80, 126.0)];
        // Diagonal + true reciprocal cross-peaks on detected axis_y -> partner axis_z
        put(&mut hncaconh, 8.20, 118.0, 118.0, 12.0);
        put(&mut hncaconh, 8.55, 122.0, 122.0, 12.0);
        put(&mut hncaconh, 7.80, 126.0, 126.0, 12.0);
        put(&mut hncaconh, 8.20, 118.0, 122.0, 9.0);
        put(&mut hncaconh, 8.55, 122.0, 118.0, 9.0);
        // Spurious noise/bleed at (H_0, N_y=126.0, N_z=118.0) and (H_2, N_y=118.0, N_z=126.0):
        // wrong H for detected axis_y, so a fixed Direction-2 scanner MUST NOT emit (118.0, 126.0).
        put(&mut hncaconh, 8.20, 126.0, 118.0, 9.0);
        put(&mut hncaconh, 7.80, 118.0, 126.0, 9.0);

        let pairs = extract_reciprocal_hncaconh_pairs(&hncaconh, &roots, 4.0);
        assert!(
            pairs
                .iter()
                .any(|&(a, b)| (a - 118.0).abs() < 0.4 && (b - 122.0).abs() < 0.4),
            "true reciprocal (118, 122) pair must be recovered: {pairs:?}"
        );
        assert!(
            !pairs.iter().any(
                |&(a, b)| ((a - 118.0).abs() < 0.4 && (b - 126.0).abs() < 0.4)
                    || ((a - 126.0).abs() < 0.4 && (b - 118.0).abs() < 0.4)
            ),
            "spurious wrong-H Direction-2 pair (118, 126) must NOT be extracted: {pairs:?}"
        );
    }

    #[test]
    fn test_p2_noisy_overlapped_cas_and_im1_hncacb_idp_benchmark() {
        // Realistic stress test with:
        // - coordinate jitter on CAS peaks + 20% breakthrough/false-positive CAS peaks on non-target roots
        // - genuine i-1 HNCACB peaks alongside stronger intra i peaks (and inverted polarity!)
        // - overlapping IDP shifts
        let seq: Vec<String> = [
            "MET", "SER", "ALA", "GLU", "LEU", "PRO", "THR", "VAL", "ASP", "GLY", "SER", "GLN",
            "PHE", "ALA", "LYS", "ARG", "THR", "ILE", "PRO", "ASN", "HIS", "GLY", "GLU", "TYR",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let base_cfg = IdpSolverConfig::default();
        let mut preds = predict_ncidp_shifts(&seq, &base_cfg);
        let mut rng = Rng::new(2026);

        // Create genuine IDP shift overlap so baseline without CAS/HNCO->HNCACB is ambiguous
        let ave_c_ser = 0.5 * (preds[1].c_ppm + preds[10].c_ppm);
        preds[1].c_ppm = ave_c_ser + 0.012;
        preds[10].c_ppm = ave_c_ser - 0.012;
        preds[2].n_ppm = 123.40;
        preds[11].n_ppm = 123.44;
        let ave_c_glu = 0.5 * (preds[3].c_ppm + preds[22].c_ppm);
        preds[3].c_ppm = ave_c_glu - 0.015;
        preds[22].c_ppm = ave_c_glu + 0.015;
        preds[4].n_ppm = 121.80;
        preds[23].n_ppm = 121.84;

        let mut systems = Vec::new();
        let mut cacon_ser_pks = Vec::new();
        let mut cacon_ala_pks = Vec::new();
        let mut canco_ser_pks = Vec::new();
        let mut hnco_pks = Vec::new();
        let mut hncacb_pks = Vec::new();

        for k in 1..seq.len() {
            let prv = &preds[k - 1];
            let cur = &preds[k];
            let c_prev = prv.c_ppm + rng.normal(0.0, 0.015);
            let n_curr = cur.n_ppm + rng.normal(0.0, 0.05);
            let mut s = ConDipeptideSystem::new(k - 1, c_prev, n_curr, 100.0);
            // Leave ca_prev/cb_prev sparse (~50%) so HNCACB i-1 filtering must rely on intensity too
            if k % 2 == 0 {
                s.ca_prev.push(prv.ca_ppm);
                if let Some(cb) = prv.cb_ppm {
                    s.cb_prev.push(cb);
                }
            }
            s.has_cbcacon_evidence = true;
            systems.push(s);

            if seq[k] != "PRO" {
                let h_i = 8.05 + 0.05 * ((k % 11) as f64);
                hnco_pks
                    .push(Peak::triple("HNCO", h_i, n_curr, c_prev, 1).with_intensity(90.0, None));
                // Inverted HNCACB with BOTH strong intra (i) AND weaker sequential (i-1) peaks
                hncacb_pks.push(
                    Peak::triple("HNCACB", h_i, n_curr, cur.ca_ppm, -1).with_intensity(100.0, None),
                );
                hncacb_pks.push(
                    Peak::triple("HNCACB", h_i, n_curr, prv.ca_ppm, -1).with_intensity(45.0, None),
                );
                if let Some(cb) = cur.cb_ppm {
                    hncacb_pks.push(
                        Peak::triple("HNCACB", h_i, n_curr, cb, 1).with_intensity(90.0, None),
                    );
                }
                if let Some(cb_p) = prv.cb_ppm {
                    hncacb_pks.push(
                        Peak::triple("HNCACB", h_i, n_curr, cb_p, 1).with_intensity(40.0, None),
                    );
                }
            }

            let jittered = CasPeak {
                c_prime_ppm: c_prev + rng.normal(0.0, 0.015),
                n_ppm: n_curr + rng.normal(0.0, 0.05),
                height: 85.0,
                volume: 85.0,
            };
            if seq[k - 1] == "SER" || k == 4 || k == 14 {
                // True Ser(i-1) PLUS 2 breakthrough/accidental-overlap false positives at k=4,14!
                cacon_ser_pks.push(jittered.clone());
            }
            if seq[k - 1] == "ALA" || k == 9 {
                cacon_ala_pks.push(jittered.clone());
            }
            if seq[k] == "SER" || k == 19 {
                canco_ser_pks.push(jittered);
            }
        }

        let cas = vec![
            CasSpectrum {
                name: "cacon_ser".to_string(),
                mode: CasMode::Cacon,
                residues: vec!["SER".to_string()],
                peaks: cacon_ser_pks,
            },
            CasSpectrum {
                name: "cacon_ala".to_string(),
                mode: CasMode::Cacon,
                residues: vec!["ALA".to_string()],
                peaks: cacon_ala_pks,
            },
            CasSpectrum {
                name: "canco_ser".to_string(),
                mode: CasMode::Canco,
                residues: vec!["SER".to_string()],
                peaks: canco_ser_pks,
            },
        ];

        let mut sys_guided = systems.clone();
        let stats = attach_hnco_hncacb_to_con_systems(
            &mut sys_guided,
            &hnco_pks,
            &hncacb_pks,
            HncacbPolarity::Auto,
        );
        assert!(stats.n_roots_with_hncacb_curr >= 18);
        // Verify intensity separation prevented weaker i-1 CA from leaking into ca_curr on sparse roots
        for (idx, s) in sys_guided.iter().enumerate() {
            if seq[idx + 1] != "PRO" && !s.ca_curr.is_empty() {
                assert!(
                    s.ca_curr.len() <= 2,
                    "root {idx} leaked extra CA peaks into ca_curr: {:?}",
                    s.ca_curr
                );
            }
        }

        let mut cfg_guided = base_cfg.clone();
        cfg_guided.cas_spectra = cas;
        let res_base = assign_idp_13c_cpu(&systems, &seq, &base_cfg);
        let res_guided = assign_idp_13c_cpu(&sys_guided, &seq, &cfg_guided);
        let acc_base = (0..systems.len())
            .filter(|&i| res_base.position[i] == ((i + 1) as i32))
            .count();
        let acc_guided = (0..sys_guided.len())
            .filter(|&i| res_guided.position[i] == ((i + 1) as i32))
            .count();
        let hc_base = res_base.confidence.iter().filter(|&&c| c >= 0.80).count();
        let hc_guided = res_guided.confidence.iter().filter(|&&c| c >= 0.80).count();
        assert!(
            acc_guided >= acc_base && acc_guided >= 20 && hc_guided >= hc_base,
            "noisy/overlapped CAS + i/i-1 HNCACB must remain robust without brittle mis-assignments: base=({acc_base}, hc={hc_base}) guided=({acc_guided}, hc={hc_guided})"
        );
    }

    #[test]
    fn test_p2_clec16a_real_con_cas_regression() {
        let Ok(data_dir) = std::env::var("NMR_REAL_DATA_DIR") else {
            return;
        };
        let clec_root = std::path::PathBuf::from(data_dir).join("private/clec16a");
        let exp_dir = clec_root.join("13C15N_Clec16aIDPR2_MG_091119");
        let con_dir = exp_dir.join("10007/pdata/1");
        let fasta_path = clec_root.join("clec16a_mouse_892-1050.fasta");
        if !con_dir.exists() || !fasta_path.exists() {
            return;
        }

        let fasta_txt = std::fs::read_to_string(&fasta_path).expect("read Clec16a fasta");
        let seq: Vec<String> = fasta_txt
            .lines()
            .filter(|l| !l.trim_start().starts_with('>'))
            .flat_map(|l| l.trim().chars())
            .filter(|c| c.is_ascii_alphabetic())
            .map(|c| crate::priors::one_to_three(c).to_string())
            .filter(|r| r != "UNK")
            .collect();
        assert!(seq.len() >= 150, "expected ~159aa Clec16a IDPR2 sequence");

        let con_spec = crate::spectrum::read_bruker_pdata_2d(&con_dir).expect("read CON 10007");
        let pcfg = crate::c13_detected::C13PickerConfig::default();
        let con_roots = crate::c13_detected::pick_con_2d(&con_spec, &pcfg);
        assert!(
            con_roots.len() >= 120,
            "expected >= 120 picked CON roots on Clec16a 10007, got {}",
            con_roots.len()
        );

        let mut cas_spectra = Vec::new();
        for (sub, name, mode, res) in [
            ("10013/pdata/1", "cacon_ala", CasMode::Cacon, "ALA"),
            ("10014/pdata/1", "canco_ala", CasMode::Canco, "ALA"),
            ("10015/pdata/1", "cacon_gly", CasMode::Cacon, "GLY"),
        ] {
            let p = exp_dir.join(sub);
            if p.exists() {
                let sp = crate::spectrum::read_bruker_pdata_2d(&p).expect("read CAS 2D");
                let pks = crate::c13_detected::pick_con_2d(&sp, &pcfg)
                    .into_iter()
                    .map(|pk| CasPeak {
                        c_prime_ppm: pk.c_prev_ppm,
                        n_ppm: pk.n_curr_ppm,
                        height: pk.height,
                        volume: pk.volume,
                    })
                    .collect();
                cas_spectra.push(CasSpectrum {
                    name: name.to_string(),
                    mode,
                    residues: vec![res.to_string()],
                    peaks: pks,
                });
            }
        }
        assert_eq!(cas_spectra.len(), 3);

        let ds = C13PickedDataset {
            con_roots,
            caco_peaks: Vec::new(),
            strip_peaks: Vec::new(),
        };
        let systems = register_and_build_con_dipeptide_systems(&ds);
        let cfg = IdpSolverConfig {
            cas_spectra: cas_spectra.clone(),
            replicas: 8,
            steps_per_system: 400,
            seed: 0,
            ..IdpSolverConfig::default()
        };
        let raw_off = estimate_idp_reference_offsets(&systems, &seq, &cfg);
        let safety = evaluate_idp_auto_reference_safety(&systems, &raw_off);
        assert!(
            safety.safe_offsets.d_c < -1.5 && safety.safe_offsets.d_c > -3.5,
            "expected Clec16a TMS-referenced 13C' offset in -3.5..-1.5 ppm, got {:?}",
            safety.safe_offsets
        );
        let typing =
            summarize_idp_cas_residue_typing(&systems, &cas_spectra, &cfg, &safety.safe_offsets);
        assert!(
            typing
                .cacon_hit_counts
                .get("cacon_ala")
                .copied()
                .unwrap_or(0)
                >= 5,
            "expected >= 5 cacon_ala hits on real Clec16a CON+CAS: {typing:?}"
        );
        let res = assign_idp_13c_cpu(&systems, &seq, &cfg);
        let n_placed = res.position.iter().filter(|&&p| p >= 0).count();
        assert!(
            n_placed >= 100,
            "expected >= 100 placed residues on real Clec16a CON+CAS, got {n_placed}"
        );
    }
}
