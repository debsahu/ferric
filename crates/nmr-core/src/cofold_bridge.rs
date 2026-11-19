//! Clean-room `CSP`/`PRE`-guided co-folding restraint bridge (`AlphaFold 3` JSON v2 & `Boltz-2`
//! YAML v1) and multi-pose biomolecular complex NMR re-scorer (`T21.6a`,
//! `docs/specs/21-alphafold-integrative-nmr.md` section 6).
//!
//! # Provenance & Clean-Room Statement
//! Original clean-room implementation derived strictly from peer-reviewed literature and public
//! open file-format specifications (`0` copyleft, academic-licence, or proprietary docking/NMR
//! software consulted; `0` external product names in Rust identifiers):
//! - Abramson, J., Adler, J., Dunger, J. et al. (*Nature* **2024**, 630, 493-500) -- AlphaFold 3
//!   biomolecular complex prediction and public input JSON specification (`dialect: "alphafold3"`,
//!   `version: 2`).
//! - Wohlwend, J., Corso, G., Passaro, S. et al. (*bioRxiv* **2024**, DOI: `10.1101/2024.11.19.624167`)
//!   & Passaro, S., Corso, G., Wohlwend, J. et al. (*bioRxiv* **2025**, DOI: `10.1101/2025.06.14.659707`)
//!   -- MIT-licensed `Boltz-1` / `Boltz-2` declarative YAML `version: 1` `pocket`, `contact`, and
//!   `affinity` input specification.
//! - Shrake, A. & Rupley, J. A. (*J. Mol. Biol.* **1973**, 79, 351-371) -- 92-point Fibonacci
//!   sphere solvent-accessible surface area (`SASA`) quadrature with water probe radius `1.40 A`;
//!   Tien, M. Z., Meyer, A. G., Sydykova, D. K., Spielman, S. J. & Wilke, C. O. (*PLoS ONE*
//!   **2013**, 8, e80635) -- theoretical Gly-X-Gly maximum residue `SASA` reference scale;
//!   Bondi, A. (*J. Phys. Chem.* **1964**, 68, 441-451) -- atomic van der Waals radii.
//! - Williamson, M. P. (*Prog. Nucl. Magn. Reson. Spectrosc.* **2013**, 73, 1-16) & McCoy, M. A.
//!   & Wyss, D. F. (*J. Am. Chem. Soc.* **2002**, 124, 11758-11763) -- iterative `2-sigma`
//!   chemical-shift perturbation (`CSP`) binding-site detection, surface-vs-allosteric core
//!   filtering, and ligand ring-current / hydrogen-bond / desolvation pose scoring.
//! - Haigh, C. W. & Mallion, R. B. (*Mol. Phys.* **1972**, 24, 955-970) & Case, D. A. (*J. Biomol.
//!   NMR* **1995**, 6, 341-346) -- conjugated aromatic ring-current shielding tensors.
//! - Wagner, G., Pardi, A. & Bax, A. (*J. Am. Chem. Soc.* **1983**, 105, 5948-5949) -- inverse-cube
//!   `r_HO^-3` hydrogen-bond chemical shift deshielding.
//! - Nilges, M. (*J. Mol. Biol.* **1995**, 245, 645-660) & Solomon, I. (*Phys. Rev.* **1955**, 99,
//!   559-565) -- `r^-6` sum-averaged intermolecular `NOE` and paramagnetic `PRE` distance bounds.
//! - Losonczi, J. A. et al. (*J. Magn. Reson.* **1999**, 138, 334-342); Cornilescu, G. et al.
//!   (*J. Am. Chem. Soc.* **1998**, 120, 6836-6837); Schmitz, C. et al. (*J. Biomol. NMR* **2008**,
//!   41, 179-189) -- multi-subunit `Saupe` `RDC` `ByPeptidePlane` `Q_free` and 5-parameter
//!   `Delta_chi` `PCS` quality factors.
//! - Mendez, R., Leplae, R., De Maria, L. & Wodak, S. J. (*Proteins* **2003**, 52, 51-67) --
//!   `CAPRI` biomolecular complex interface RMSD (`iRMSD`) definition.

use std::collections::{BTreeMap, BTreeSet};
use std::f64::consts::PI;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::csp::{
    amide_alpha_n, combined_amide_csp_ppm, compute_williamson_2sigma_threshold, CspGlobalFitResult,
    TrimmedCspThresholds,
};
use crate::noesy::{
    v_add, v_cross, v_dist, v_dist_sq, v_dot, v_norm, v_scale, v_sub, v_unit, PdbChain, PdbResidue,
    Vec3,
};
use crate::octet_mol::{
    atomic_number_from_symbol, covalent_radius, expand_abbreviations, parse_smiles,
    perceive_aromaticity, perceive_sssr, read_mol, read_oct, read_sdf, write_canonical_smiles,
    Atom, Bond, BondKind, ChemError, Molecule,
};
use crate::paramagnetic::{
    fit_pcs_tensor_svd, predict_pcs_from_tensor, DeltaChiTensor, PcsObservation,
};
use crate::rdc::{
    cornilescu_q, evaluate_rdc_validation, normalize_rdc_atom_name,
    partition_rdc_free_set_with_mode, RdcBondType, RdcFreeStratification, RdcMediumSet,
    RdcRestraint, RdcSignConvention,
};
use crate::struct_shifts::{
    evaluate_haigh_mallion_ring_current, predict_structure_shifts_with_firewall,
    ConjugatedAromaticRing, StructShiftConfig, StructureModel, MAX_STRUCTURE_FILE_BYTES,
};
use crate::structure::{det_cbrt, det_ln, det_sin_cos, kabsch_rmsd, kabsch_transform};
use crate::torsion_db::{
    HomologFirewall, HomologFirewallQuery, TorsionDb, DEFAULT_FIREWALL_MAX_SEQ_IDENTITY,
};

/// Maximum text/restraint file byte size accepted by `cofold_bridge` (`64 MiB` DoS guard).
pub const MAX_COFOLD_INPUT_BYTES: usize = 64 * 1024 * 1024;

/// Number of Fibonacci quadrature points per atom sphere in Shrake-Rupley (1973) SASA (`92`).
pub const SHRAKE_RUPLEY_SPHERE_POINTS: usize = 92;

/// Water solvent probe radius (`1.40 A`) for Shrake-Rupley (1973) SASA evaluation.
pub const SHRAKE_RUPLEY_PROBE_RADIUS_A: f64 = 1.40;

/// Minimum apo relative SASA (`rSASA >= 0.20`) separating surface-exposed `CSP` pocket contacts
/// from buried `AllostericCorePerturbation` shifts (`docs/specs/21-alphafold-integrative-nmr.md` section 6.1).
pub const DEFAULT_MIN_POCKET_RSASA: f64 = 0.20;

/// Lower floor (`4.5 A`) applied to an exported co-folding `contact` upper bound. Raising a tighter
/// bound up to this floor only *loosens* it (conservative: residue-level Boltz-2 tokens cannot
/// express sub-4.5 A atom-atom NOE bounds).
pub const MIN_COFOLD_CONTACT_UPPER_A: f64 = 4.5;

/// Default ceiling (`14.0 A`) for an exported co-folding `contact` upper bound. A restraint whose
/// computed upper bound exceeds the configured ceiling is **dropped** (never clamped down to the
/// ceiling: clamping a weak `16.5 A` bound to `14.0 A` would fabricate a strong restraint,
/// `21.W3:P1-7`). The configured ceiling must lie inside the Boltz-2 range `[4.0, 20.0] A`.
pub const MAX_COFOLD_CONTACT_UPPER_A: f64 = 14.0;

/// Boltz-2 `contact` / `pocket` `max_distance` hard lower limit (`4.0 A`).
pub const BOLTZ2_MIN_DISTANCE_A: f64 = 4.0;

/// Boltz-2 `contact` / `pocket` `max_distance` hard upper limit (`20.0 A`).
pub const BOLTZ2_MAX_DISTANCE_A: f64 = 20.0;

/// Default Boltz-2 `pocket` `max_distance` (`6.0 A`).
pub const DEFAULT_BOLTZ2_POCKET_MAX_DISTANCE_A: f64 = 6.0;

/// Heavy-atom receptor-to-partner distance cutoff (`5.0 A`) for `CSP` contact ROC-AUC evaluation.
pub const DEFAULT_CSP_CONTACT_CUTOFF_A: f64 = 5.0;

/// Maximum permitted intermolecular `NOE`/`PRE` distance violation (`1.50 A`) before decoy abstention.
pub const DEFAULT_MAX_ALLOWED_VIOLATION_A: f64 = 1.50;

/// Minimum `CSP` `5.0 A` contact ROC-AUC (`0.65`) required when `>= 5` surface `CSP` hits exist.
///
/// Conventional moderate-enrichment operating point (`0.65`; committed alongside the synthetic
/// self-consistency benchmark, not audited as an a-priori holdout, `21F:B17`): for the minimum
/// guard population (`n_pos = 5` surface hits against `n_neg >= 20` non-hits) it sits `~1.0`
/// Mann-Whitney standard deviations above chance
/// (`sigma_AUC = sqrt((n_pos + n_neg + 1) / (12 n_pos n_neg)) ~ 0.147`).
/// Every pose additionally reports the one-sided Mann-Whitney `p` value of its contact AUC.
pub const DEFAULT_MIN_CSP_ROC_AUC: f64 = 0.65;

/// Default MTSL (nitroxide spin-label) tether allowance (`6.0 A`) added to the **upper** (and
/// subtracted from the lower) electron-to-anchor-atom distance bound of a `PRE` `Gamma_2`
/// restraint: the unpaired electron sits up to `~6-8 A` from the labelled `C-beta`/`S-gamma`.
/// Use `0.0 A` for a rigid paramagnetic metal site.
pub const DEFAULT_MTSL_TETHER_PAD_A: f64 = 6.0;

/// Default spectrometer `1H` Larmor frequency (`600.0 MHz`) in the Solomon-Bloembergen
/// spectral-density term.
pub const DEFAULT_PRE_PROTON_MHZ: f64 = 600.0;

/// Default paramagnetic electron spin quantum number `S = 1/2` (nitroxide / MTSL).
pub const DEFAULT_PRE_ELECTRON_SPIN: f64 = 0.5;

/// Wuethrich-class `NOE` upper bounds (`A`): strong / medium / weak (lower bound `1.8 A`).
pub const WUTHRICH_NOE_STRONG_UPPER_A: f64 = 2.7;
pub const WUTHRICH_NOE_MEDIUM_UPPER_A: f64 = 3.3;
pub const WUTHRICH_NOE_WEAK_UPPER_A: f64 = 5.0;

// CODATA 2018 constants for the first-principles Solomon-Bloembergen `PRE` prefactor.
const MU0_OVER_4PI_SI: f64 = 1.0e-7; // T m A^-1 (mu_0 / 4 pi, SI)
const GAMMA_H1_RAD_S_T: f64 = 2.675_221_874_4e8; // rad s^-1 T^-1
const G_ELECTRON: f64 = 2.002_319_304_362_56;
const BOHR_MAGNETON_J_T: f64 = 9.274_010_078_3e-24; // J T^-1
const ANGSTROM6_PER_M6: f64 = 1.0e60;

/// First-principles Solomon-Bloembergen dipolar `PRE` prefactor in `A^6 s^-2`:
///
/// `K = (1/15) (mu_0 / 4 pi)^2 gamma_H^2 g_e^2 mu_B^2 S (S + 1)`  (`= 1.234e16 A^6 s^-2` at `S = 1/2`).
///
/// This replaces the former hard-coded `1.23e7 A^6 s^-1`, which was `~40x` too small at
/// `tau_c = 10 ns` (it corresponds to `tau_c ~ 0.25 ns`) and biased exported `PRE` bounds by
/// `~6 A` (`21.W3:P0-4`). Identical to the `domain_dock` (`S3`) expression.
pub fn solomon_pre_k_a6_s2(spin_s: f64) -> f64 {
    let g_mu_b = G_ELECTRON * BOHR_MAGNETON_J_T;
    (1.0 / 15.0)
        * MU0_OVER_4PI_SI
        * MU0_OVER_4PI_SI
        * GAMMA_H1_RAD_S_T
        * GAMMA_H1_RAD_S_T
        * g_mu_b
        * g_mu_b
        * spin_s
        * (spin_s + 1.0)
        * ANGSTROM6_PER_M6
}

/// Solomon-Bloembergen spectral-density factor `4 tau_c + 3 tau_c / (1 + omega_H^2 tau_c^2)` in
/// seconds (`tau_c` in ns, `1H` Larmor frequency in MHz).
pub fn solomon_spectral_factor_s(tau_c_ns: f64, proton_mhz: f64) -> f64 {
    let tau = tau_c_ns * 1.0e-9;
    let omega = 2.0 * PI * proton_mhz * 1.0e6;
    4.0 * tau + 3.0 * tau / (1.0 + omega * omega * tau * tau)
}

/// Effective `Gamma_2 = K_eff / r^6` numerator (`A^6 s^-1`):
/// `K_eff = K_Solomon * (4 tau_c + 3 tau_c / (1 + omega_H^2 tau_c^2))`.
pub fn solomon_pre_k_eff_a6_s1(
    tau_c_ns: f64,
    proton_mhz: f64,
    spin_s: f64,
) -> Result<f64, NmrError> {
    if !tau_c_ns.is_finite() || !(0.05..=1.0e4).contains(&tau_c_ns) {
        return Err(NmrError::invalid_input(format!(
            "PRE tau_c_ns ({tau_c_ns}) must be finite and inside [0.05, 10000] ns"
        )));
    }
    if !proton_mhz.is_finite() || !(1.0..=5.0e3).contains(&proton_mhz) {
        return Err(NmrError::invalid_input(format!(
            "PRE proton_mhz ({proton_mhz}) must be finite and inside [1, 5000] MHz"
        )));
    }
    if !spin_s.is_finite() || !(0.5..=8.0).contains(&spin_s) {
        return Err(NmrError::invalid_input(format!(
            "PRE electron spin S ({spin_s}) must be finite and inside [0.5, 8.0]"
        )));
    }
    Ok(solomon_pre_k_a6_s2(spin_s) * solomon_spectral_factor_s(tau_c_ns, proton_mhz))
}

/// Paramagnetic-centre-to-proton distance (`A`) from a measured `Gamma_2` (`s^-1`) and `K_eff`.
pub fn solomon_distance_from_gamma2_a(gamma2_s1: f64, k_eff_a6_s1: f64) -> f64 {
    sixth_root(k_eff_a6_s1 / gamma2_s1.max(1e-12))
}

/// Solomon-Bloembergen parameters required to convert measured `Gamma_2` rates into distance
/// bounds. **There is deliberately no `Default`**: `tau_c` is mandatory (`21.W3:P0-4`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PreSolomonParams {
    /// Rotational correlation time of the paramagnetic-centre-to-proton vector (`ns`).
    pub tau_c_ns: f64,
    /// Spectrometer `1H` Larmor frequency (`MHz`).
    pub proton_mhz: f64,
    /// Electron spin `S` (`0.5` for a nitroxide).
    pub spin_s: f64,
    /// MTSL tether allowance (`A`): added to the upper and subtracted from the lower bound
    /// (`6.0 A` for MTSL, `0.0 A` for a rigid paramagnetic metal).
    pub tether_pad_a: f64,
    /// Additional upper-bound margin (`A`, default `1.5`).
    pub upper_margin_a: f64,
}

impl PreSolomonParams {
    /// MTSL defaults (`600 MHz`, `S = 1/2`, `6.0 A` tether, `+1.5 A` margin) for a given `tau_c`.
    pub fn new(tau_c_ns: f64) -> Self {
        Self {
            tau_c_ns,
            proton_mhz: DEFAULT_PRE_PROTON_MHZ,
            spin_s: DEFAULT_PRE_ELECTRON_SPIN,
            tether_pad_a: DEFAULT_MTSL_TETHER_PAD_A,
            upper_margin_a: 1.5,
        }
    }

    /// Rigid paramagnetic metal (`tether_pad_a = 0.0`).
    pub fn rigid_metal(tau_c_ns: f64) -> Self {
        Self {
            tether_pad_a: 0.0,
            ..Self::new(tau_c_ns)
        }
    }

    /// Validated `K_eff` (`A^6 s^-1`).
    pub fn k_eff_a6_s1(&self) -> Result<f64, NmrError> {
        if !self.tether_pad_a.is_finite() || !(0.0..=15.0).contains(&self.tether_pad_a) {
            return Err(NmrError::invalid_input(format!(
                "PRE tether_pad_a ({}) must be finite in [0, 15] A",
                self.tether_pad_a
            )));
        }
        if !self.upper_margin_a.is_finite() || self.upper_margin_a < 0.0 {
            return Err(NmrError::invalid_input(
                "PRE upper_margin_a must be finite and >= 0",
            ));
        }
        solomon_pre_k_eff_a6_s1(self.tau_c_ns, self.proton_mhz, self.spin_s)
    }
}

/// Discrete Wuethrich `NOE` intensity class (`strong 1.8-2.7`, `medium 1.8-3.3`, `weak 1.8-5.0 A`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NoeIntensityClass {
    Strong,
    Medium,
    Weak,
}

impl NoeIntensityClass {
    /// Class upper distance bound (`A`).
    pub const fn upper_a(self) -> f64 {
        match self {
            Self::Strong => WUTHRICH_NOE_STRONG_UPPER_A,
            Self::Medium => WUTHRICH_NOE_MEDIUM_UPPER_A,
            Self::Weak => WUTHRICH_NOE_WEAK_UPPER_A,
        }
    }

    /// Class assigned to a true `r` (`<= 2.7` strong, `<= 3.3` medium, `<= 5.0` weak, else not observable).
    pub fn from_distance_a(r_a: f64) -> Option<Self> {
        if !r_a.is_finite() {
            None
        } else if r_a <= WUTHRICH_NOE_STRONG_UPPER_A {
            Some(Self::Strong)
        } else if r_a <= WUTHRICH_NOE_MEDIUM_UPPER_A {
            Some(Self::Medium)
        } else if r_a <= WUTHRICH_NOE_WEAK_UPPER_A {
            Some(Self::Weak)
        } else {
            None
        }
    }

    /// Next weaker class (`Weak` stays `Weak`): models integration/overlap mis-classification.
    pub const fn weaker(self) -> Self {
        match self {
            Self::Strong => Self::Medium,
            Self::Medium | Self::Weak => Self::Weak,
        }
    }

    /// Parse `S`/`M`/`W` (or `strong`/`medium`/`weak`).
    pub fn parse_token(tok: &str) -> Option<Self> {
        match tok.trim().to_ascii_lowercase().as_str() {
            "s" | "strong" => Some(Self::Strong),
            "m" | "medium" | "med" => Some(Self::Medium),
            "w" | "weak" => Some(Self::Weak),
            _ => None,
        }
    }
}

/// Minimum number of surface-exposed `2-sigma` `CSP` hits (`5`) activating the `AUC_CSP < 0.58` guard.
pub const DEFAULT_MIN_SURFACE_HITS_FOR_AUC_GUARD: usize = 5;

/// Maximum permitted `ByPeptidePlane` `RDC` `Q_free` (`0.65`) before decoy abstention.
pub const DEFAULT_MAX_ALLOWED_RDC_Q_FREE: f64 = 0.65;

/// Standard `CAPRI` heavy-atom cross-subunit distance cutoff (`10.0 A`) defining reference interface residues.
pub const CAPRI_INTERFACE_CUTOFF_A: f64 = 10.0;

// ============================================================================
// 1. Structured Error Type (`NmrError`)
// ============================================================================

/// Error category returned by `nmr-core::cofold_bridge` operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NmrErrorKind {
    EmptyInput,
    InvalidInput,
    ParseError,
    IoError,
    RestraintMismatch,
}

/// Structured error returned by `score_cofold_poses` and co-folding bridge APIs.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NmrError {
    pub kind: NmrErrorKind,
    pub message: String,
}

impl NmrError {
    #[inline]
    pub fn new(kind: NmrErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    #[inline]
    pub fn empty_input(message: impl Into<String>) -> Self {
        Self::new(NmrErrorKind::EmptyInput, message)
    }

    #[inline]
    pub fn invalid_input(message: impl Into<String>) -> Self {
        Self::new(NmrErrorKind::InvalidInput, message)
    }

    #[inline]
    pub fn parse(message: impl Into<String>) -> Self {
        Self::new(NmrErrorKind::ParseError, message)
    }

    #[inline]
    pub fn io(message: impl Into<String>) -> Self {
        Self::new(NmrErrorKind::IoError, message)
    }

    #[inline]
    pub fn restraint_mismatch(message: impl Into<String>) -> Self {
        Self::new(NmrErrorKind::RestraintMismatch, message)
    }

    #[inline]
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for NmrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for NmrError {}

impl From<String> for NmrError {
    fn from(msg: String) -> Self {
        Self::new(NmrErrorKind::InvalidInput, msg)
    }
}

impl From<&str> for NmrError {
    fn from(msg: &str) -> Self {
        Self::new(NmrErrorKind::InvalidInput, msg.to_string())
    }
}

impl From<ChemError> for NmrError {
    fn from(err: ChemError) -> Self {
        Self::new(NmrErrorKind::ParseError, err.message)
    }
}

impl From<std::io::Error> for NmrError {
    fn from(err: std::io::Error) -> Self {
        Self::new(NmrErrorKind::IoError, err.to_string())
    }
}

impl From<NmrError> for String {
    fn from(err: NmrError) -> Self {
        err.message
    }
}

fn read_bounded_utf8_file(path: &Path, label: &str) -> Result<String, NmrError> {
    let meta = fs::metadata(path)
        .map_err(|e| NmrError::io(format!("failed to stat {label} '{}': {e}", path.display())))?;
    if meta.len() > MAX_COFOLD_INPUT_BYTES as u64 {
        return Err(NmrError::invalid_input(format!(
            "{label} '{}' ({} bytes) exceeds 64 MiB limit ({MAX_COFOLD_INPUT_BYTES})",
            path.display(),
            meta.len()
        )));
    }
    let bytes = fs::read(path)
        .map_err(|e| NmrError::io(format!("failed to read {label} '{}': {e}", path.display())))?;
    if bytes.len() > MAX_COFOLD_INPUT_BYTES || bytes.contains(&0u8) {
        return Err(NmrError::parse(format!(
            "{label} '{}' exceeds size cap or contains NUL bytes",
            path.display()
        )));
    }
    String::from_utf8(bytes).map_err(|e| {
        NmrError::parse(format!(
            "invalid UTF-8 in {label} '{}': {e}",
            path.display()
        ))
    })
}

fn check_bounded_text(text: &str, label: &str) -> Result<(), NmrError> {
    if text.len() > MAX_COFOLD_INPUT_BYTES || text.as_bytes().contains(&0u8) {
        return Err(NmrError::parse(format!(
            "{label} exceeds 64 MiB limit or contains NUL bytes"
        )));
    }
    Ok(())
}

// ============================================================================
// 2. Shrake-Rupley (1973) 92-Point Fibonacci-Sphere SASA & Relative SASA
// ============================================================================

/// Per-residue Shrake-Rupley (1973) absolute and relative solvent-accessible surface area.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ShrakeRupleyResidueSasa {
    pub chain_id: String,
    pub seq_id: i32,
    pub res_name: String,
    /// Absolute solvent-accessible surface area (`A^2`) summed over residue heavy atoms.
    pub sasa_angstrom2: f64,
    /// Reference Gly-X-Gly maximum `SASA` (`A^2`, Tien et al. *PLoS ONE* 2013).
    pub max_sasa_angstrom2: f64,
    /// Relative solvent-accessible surface area `rSASA_i = clamp(SASA_i / SASA_max, 0.0, 1.0)`.
    pub rsasa: f64,
}

/// Theoretical Gly-X-Gly maximum residue solvent-accessible surface area (`A^2`) from
/// Tien, Meyer, Sydykova, Spielman & Wilke (*PLoS ONE* **2013**, 8, e80635, Table 1).
#[inline]
pub fn max_asa_tien_2013(res_name: &str) -> f64 {
    match res_name.trim().to_ascii_uppercase().as_str() {
        "ALA" => 129.0,
        "ARG" => 274.0,
        "ASN" => 195.0,
        "ASP" | "ASH" => 193.0,
        "CYS" | "CYX" | "CYM" => 167.0,
        "GLN" => 225.0,
        "GLU" | "GLH" => 223.0,
        "GLY" => 104.0,
        "HIS" | "HID" | "HIE" | "HIP" | "HSD" | "HSE" | "HSP" => 224.0,
        "ILE" => 197.0,
        "LEU" => 201.0,
        "LYS" | "LYN" => 236.0,
        "MET" | "MSE" => 224.0,
        "PHE" => 240.0,
        "PRO" => 159.0,
        "SER" => 155.0,
        "THR" => 172.0,
        "TRP" => 285.0,
        "TYR" => 263.0,
        "VAL" => 174.0,
        _ => 195.0,
    }
}

/// Bondi (1964) / Tsai et al. (1999) heavy-atom van der Waals radius (`A`) by atomic number `Z`.
#[inline]
pub fn bondi_heavy_vdw_radius(z: u8) -> f64 {
    match z {
        6 => 1.70,  // C
        7 => 1.55,  // N
        8 => 1.52,  // O
        9 => 1.47,  // F
        15 => 1.80, // P
        16 => 1.80, // S
        17 => 1.75, // Cl
        34 => 1.90, // Se
        35 => 1.85, // Br
        53 => 1.98, // I
        _ => 1.70,
    }
}

#[inline]
fn infer_heavy_z_from_atom_name(atom_name: &str) -> Option<u8> {
    let trimmed = atom_name.trim();
    if trimmed.is_empty() {
        return None;
    }
    let up = trimmed.to_ascii_uppercase();
    // Skip explicit hydrogens and pseudoatoms (`H*`, `D*`, `1H*`, `2H*`, `3H*`, `Q*`)
    let stripped = up.trim_start_matches(|c: char| c.is_ascii_digit());
    if stripped.is_empty()
        || stripped.starts_with('H')
        || stripped.starts_with('D')
        || stripped.starts_with('Q')
    {
        return None;
    }
    if stripped.starts_with("SE") {
        return Some(34);
    }
    if stripped.starts_with("CL") {
        return Some(17);
    }
    if stripped.starts_with("BR") {
        return Some(35);
    }
    match stripped.as_bytes()[0] {
        b'C' => Some(6),
        b'N' => Some(7),
        b'O' => Some(8),
        b'S' => Some(16),
        b'P' => Some(15),
        b'F' => Some(9),
        b'I' => Some(53),
        _ => atomic_number_from_symbol(&stripped[..1]).filter(|&z| z > 1),
    }
}

fn fibonacci_sphere_points(n_points: usize) -> Vec<Vec3> {
    let n = n_points.max(1);
    // Golden angle phi = pi * (3 - sqrt(5))
    let golden_angle = PI * (3.0 - 5.0_f64.sqrt());
    let inv_n = 1.0 / (n as f64);
    let mut pts = Vec::with_capacity(n);
    for k in 0..n {
        let y = 1.0 - 2.0 * ((k as f64) + 0.5) * inv_n;
        let r_perp = (1.0 - y * y).max(0.0).sqrt();
        let theta = (k as f64) * golden_angle;
        let (sin_t, cos_t) = det_sin_cos(theta);
        pts.push([r_perp * cos_t, y, r_perp * sin_t]);
    }
    pts
}

/// Evaluate Shrake-Rupley (1973) per-atom SASA (`A^2`) across heavy-atom spheres `(center, vdw_radius_a)`
/// using `n_sphere_points` (`92` canonical) and solvent probe radius `probe_radius_a` (`1.40 A`).
pub fn compute_shrake_rupley_atom_sasa(
    spheres: &[(Vec3, f64)],
    probe_radius_a: f64,
    n_sphere_points: usize,
) -> Vec<f64> {
    let n_atoms = spheres.len();
    if n_atoms == 0 {
        return Vec::new();
    }
    let n_pts = n_sphere_points.max(1);
    let probe = probe_radius_a.max(0.0);
    let unit_pts = fibonacci_sphere_points(n_pts);
    let ext_r: Vec<f64> = spheres.iter().map(|&(_, r)| r.max(0.1) + probe).collect();
    let ext_r_sq: Vec<f64> = ext_r.iter().map(|&r| r * r).collect();

    // Pre-build overlapping neighbor lists per atom (`d(a, b) < R_a + R_b`)
    let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); n_atoms];
    for a in 0..n_atoms {
        let ca = spheres[a].0;
        let ra = ext_r[a];
        for b in (a + 1)..n_atoms {
            let r_sum = ra + ext_r[b];
            let d2 = v_dist_sq(ca, spheres[b].0);
            if d2 < r_sum * r_sum {
                neighbors[a].push(b);
                neighbors[b].push(a);
            }
        }
    }

    let four_pi_over_n = (4.0 * PI) / (n_pts as f64);
    let mut areas = vec![0.0_f64; n_atoms];
    for a in 0..n_atoms {
        let nbs = &neighbors[a];
        let ra = ext_r[a];
        if nbs.is_empty() {
            areas[a] = 4.0 * PI * ra * ra;
            continue;
        }
        let ca = spheres[a].0;
        let mut exposed = 0usize;
        let mut last_occluder: usize = 0;

        'pt_loop: for &u in &unit_pts {
            let p = [ca[0] + ra * u[0], ca[1] + ra * u[1], ca[2] + ra * u[2]];
            // Test cached last occluder first (adjacent Fibonacci points frequently hit the same neighbor)
            let b0 = nbs[last_occluder];
            if v_dist_sq(p, spheres[b0].0) < ext_r_sq[b0] {
                continue 'pt_loop;
            }
            for (nb_idx, &b) in nbs.iter().enumerate() {
                if nb_idx == last_occluder {
                    continue;
                }
                if v_dist_sq(p, spheres[b].0) < ext_r_sq[b] {
                    last_occluder = nb_idx;
                    continue 'pt_loop;
                }
            }
            exposed += 1;
        }
        areas[a] = four_pi_over_n * ra * ra * (exposed as f64);
    }
    areas
}

/// Compute 92-point Shrake-Rupley (`1973`, probe radius `1.40 A`) per-residue `SASA` and `rSASA`
/// across a [`StructureModel`], optionally including additional non-protein heavy-atom occluders
/// `(coord, atomic_number_z)` (e.g. small-molecule ligands or nucleic acids in a bound complex).
pub fn compute_shrake_rupley_rsasa(
    model: &StructureModel,
    extra_heavy_occluders: &[(Vec3, u8)],
) -> Vec<ShrakeRupleyResidueSasa> {
    let mut spheres: Vec<(Vec3, f64)> = Vec::new();
    let mut atom_res_owner: Vec<usize> = Vec::new();
    let mut has_beyond_cb = vec![false; model.residues.len()];

    for (r_idx, res) in model.residues.iter().enumerate() {
        for (aname, &pos) in &res.atoms {
            let Some(z) = infer_heavy_z_from_atom_name(aname) else {
                continue;
            };
            if !matches!(aname.as_str(), "N" | "CA" | "C" | "O" | "OXT" | "CB") {
                has_beyond_cb[r_idx] = true;
            }
            spheres.push((pos, bondi_heavy_vdw_radius(z)));
            atom_res_owner.push(r_idx);
        }
    }
    for &(pos, z) in extra_heavy_occluders {
        if z > 1 {
            spheres.push((pos, bondi_heavy_vdw_radius(z)));
        }
    }

    let atom_areas = compute_shrake_rupley_atom_sasa(
        &spheres,
        SHRAKE_RUPLEY_PROBE_RADIUS_A,
        SHRAKE_RUPLEY_SPHERE_POINTS,
    );
    let mut res_sasa = vec![0.0_f64; model.residues.len()];
    for (a_idx, &r_idx) in atom_res_owner.iter().enumerate() {
        res_sasa[r_idx] += atom_areas[a_idx];
    }

    model
        .residues
        .iter()
        .enumerate()
        .map(|(r_idx, res)| {
            let up = res.res_name.trim().to_ascii_uppercase();
            // If a non-Gly/Ala model supplied only backbone+CB stubs without sidechain heavy atoms,
            // normalize against the Ala CB-stub Gly-X-Gly ceiling (129 A^2) rather than full sidechain area.
            let max_sasa = if !has_beyond_cb[r_idx] && up != "GLY" && up != "ALA" {
                129.0
            } else {
                max_asa_tien_2013(&up)
            };
            let sasa = res_sasa[r_idx];
            let rsasa = (sasa / max_sasa.max(1.0)).clamp(0.0, 1.0);
            ShrakeRupleyResidueSasa {
                chain_id: res.chain_id.clone(),
                seq_id: res.seq_id,
                res_name: res.res_name.clone(),
                sasa_angstrom2: sasa,
                max_sasa_angstrom2: max_sasa,
                rsasa,
            }
        })
        .collect()
}

// ============================================================================
// 3. Polymer & Small-Molecule Entity Specs (`protein`/`rna`/`dna`/`ligand`)
// ============================================================================

/// Biopolymer sequence entity class supported by `AlphaFold 3` JSON and `Boltz-2` YAML.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CofoldPolymerKind {
    Protein,
    Rna,
    Dna,
}

impl CofoldPolymerKind {
    pub const fn as_key(self) -> &'static str {
        match self {
            Self::Protein => "protein",
            Self::Rna => "rna",
            Self::Dna => "dna",
        }
    }
}

/// Small-molecule / cofactor ligand specification (`SMILES` or PDB `CCD` codes).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CofoldLigandSpec {
    /// Canonical or input `SMILES` string validated via `nmr-core::octet_mol`.
    Smiles(String),
    /// One or more PDB Chemical Component Dictionary (`CCD`) codes (e.g. `["ATP"]`, `["ZN"]`).
    CcdCodes(Vec<String>),
}

/// Biomolecular chain/ligand entity in a co-folding job.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CofoldEntitySpec {
    Polymer {
        kind: CofoldPolymerKind,
        chain_id: String,
        sequence: String,
    },
    Ligand {
        chain_id: String,
        spec: CofoldLigandSpec,
    },
}

impl CofoldEntitySpec {
    pub fn chain_id(&self) -> &str {
        match self {
            Self::Polymer { chain_id, .. } | Self::Ligand { chain_id, .. } => chain_id,
        }
    }

    pub fn is_polymer(&self) -> bool {
        matches!(self, Self::Polymer { .. })
    }

    pub fn polymer_len(&self) -> Option<usize> {
        match self {
            Self::Polymer { sequence, .. } => Some(sequence.chars().count()),
            Self::Ligand { .. } => None,
        }
    }

    pub fn validate(&self) -> Result<(), NmrError> {
        validate_chain_id(self.chain_id())?;
        match self {
            Self::Polymer { kind, sequence, .. } => {
                let seq = sequence.trim();
                if seq.is_empty() {
                    return Err(NmrError::invalid_input(format!(
                        "empty {} sequence on chain '{}'",
                        kind.as_key(),
                        self.chain_id()
                    )));
                }
                for ch in seq.chars() {
                    let up = ch.to_ascii_uppercase();
                    let ok = match kind {
                        // Standard 20-aa alphabet plus `X` (unknown). `B`/`J`/`O`/`U`/`Z`
                        // (ambiguity codes, pyrrolysine, selenocysteine) are rejected: neither
                        // AlphaFold 3 nor Boltz-2 folds them as standard residues (`21.W3` P2).
                        CofoldPolymerKind::Protein => is_standard_protein_letter(up),
                        CofoldPolymerKind::Rna => matches!(up, 'A' | 'U' | 'G' | 'C' | 'N'),
                        CofoldPolymerKind::Dna => matches!(up, 'A' | 'T' | 'G' | 'C' | 'N'),
                    };
                    if !ok {
                        return Err(NmrError::invalid_input(format!(
                            "invalid residue letter '{ch}' for {} chain '{}'",
                            kind.as_key(),
                            self.chain_id()
                        )));
                    }
                }
                Ok(())
            }
            Self::Ligand { spec, chain_id } => match spec {
                CofoldLigandSpec::Smiles(smi) => {
                    if smi.trim().is_empty() {
                        return Err(NmrError::invalid_input(format!(
                            "empty ligand SMILES on chain '{chain_id}'"
                        )));
                    }
                    let _ = parse_smiles(smi.trim())?;
                    Ok(())
                }
                CofoldLigandSpec::CcdCodes(codes) => {
                    if codes.is_empty() {
                        return Err(NmrError::invalid_input(format!(
                            "empty CCD code list on ligand chain '{chain_id}'"
                        )));
                    }
                    for code in codes {
                        let c = code.trim();
                        if c.is_empty()
                            || c.len() > 5
                            || !c.chars().all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
                        {
                            return Err(NmrError::invalid_input(format!(
                                "invalid CCD code '{code}' on ligand chain '{chain_id}'"
                            )));
                        }
                    }
                    Ok(())
                }
            },
        }
    }
}

/// `true` for the 20 standard amino-acid one-letter codes and `X` (unknown residue).
fn is_standard_protein_letter(up: char) -> bool {
    matches!(
        up,
        'A' | 'C'
            | 'D'
            | 'E'
            | 'F'
            | 'G'
            | 'H'
            | 'I'
            | 'K'
            | 'L'
            | 'M'
            | 'N'
            | 'P'
            | 'Q'
            | 'R'
            | 'S'
            | 'T'
            | 'V'
            | 'W'
            | 'Y'
            | 'X'
    )
}

fn validate_chain_id(chain_id: &str) -> Result<(), NmrError> {
    let trimmed = chain_id.trim();
    if trimmed.is_empty()
        || trimmed.len() > 16
        || !trimmed
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_')
    {
        return Err(NmrError::invalid_input(format!(
            "invalid chain_id {chain_id:?}; expected 1..=16 ASCII alphanumeric/underscore characters"
        )));
    }
    Ok(())
}

/// AlphaFold 3 JSON chain/entity `id`s are uppercase-letter strings (`A`, `B`, ... `AA`).
fn validate_af3_chain_id(chain_id: &str) -> Result<(), NmrError> {
    let t = chain_id.trim();
    if t.is_empty() || t.len() > 4 || !t.chars().all(|c| c.is_ascii_uppercase()) {
        return Err(NmrError::invalid_input(format!(
            "AlphaFold 3 requires uppercase-letter chain ids (1-4 characters, e.g. 'A'); got {chain_id:?}"
        )));
    }
    Ok(())
}

/// Covalent atom address `(chain_id, 1_based_seq_idx, atom_name)` for AlphaFold 3 `bondedAtomPairs`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BondedAtomRef {
    pub chain_id: String,
    pub one_based_res_idx: usize,
    pub atom_name: String,
}

/// Covalent crosslink / ligand bond pair for AlphaFold 3 `bondedAtomPairs`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BondedAtomPairSpec {
    pub atom1: BondedAtomRef,
    pub atom2: BondedAtomRef,
}

/// Parse a multi-chain FASTA string into [`CofoldEntitySpec`] polymer entries (`protein`, `rna`, `dna`).
///
/// Supported FASTA header conventions:
/// - `>A|protein`, `>B|rna`, `>C|dna`
/// - `>chain_A mol:rna`, `>D type=dna`
/// - Plain `>A` or `>receptor` (auto-assigns `A`, `B`, ... if header is non-chain-like, and infers
///   `Rna` when composed of `{A,U,G,C}` containing `U`, otherwise `Protein`).
pub fn parse_cofold_fasta_str(fasta_text: &str) -> Result<Vec<CofoldEntitySpec>, NmrError> {
    check_bounded_text(fasta_text, "FASTA")?;
    let mut raw_records: Vec<(String, String)> = Vec::new();
    let mut cur_header: Option<String> = None;
    let mut cur_seq = String::new();

    for raw_line in fasta_text.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with(';') || line.starts_with('#') {
            continue;
        }
        if let Some(hdr) = line.strip_prefix('>') {
            if let Some(prev_h) = cur_header.take() {
                raw_records.push((prev_h, std::mem::take(&mut cur_seq)));
            }
            cur_header = Some(hdr.trim().to_string());
        } else {
            for ch in line.chars() {
                if !ch.is_whitespace() && ch != '*' && ch != '-' {
                    cur_seq.push(ch.to_ascii_uppercase());
                }
            }
        }
    }
    if let Some(prev_h) = cur_header.take() {
        raw_records.push((prev_h, cur_seq));
    }
    if raw_records.is_empty() {
        // Also accept raw single-line sequence without `>` header
        let bare: String = fasta_text
            .chars()
            .filter(|c| !c.is_whitespace())
            .map(|c| c.to_ascii_uppercase())
            .collect();
        if !bare.is_empty() && bare.chars().all(|c| c.is_ascii_alphabetic()) {
            raw_records.push(("A".to_string(), bare));
        } else {
            return Err(NmrError::empty_input("FASTA input contains zero sequences"));
        }
    }

    let mut used_ids: BTreeSet<String> = BTreeSet::new();
    let mut entities = Vec::with_capacity(raw_records.len());

    for (idx, (hdr, seq)) in raw_records.into_iter().enumerate() {
        if seq.is_empty() {
            return Err(NmrError::parse(format!(
                "FASTA entry '{hdr}' has an empty sequence"
            )));
        }
        let hdr_low = hdr.to_ascii_lowercase();
        let explicit_kind = if hdr_low.contains("|dna")
            || hdr_low.contains("mol:dna")
            || hdr_low.contains("type=dna")
            || hdr_low.starts_with("dna:")
            || hdr_low == "dna"
        {
            Some(CofoldPolymerKind::Dna)
        } else if hdr_low.contains("|rna")
            || hdr_low.contains("mol:rna")
            || hdr_low.contains("type=rna")
            || hdr_low.starts_with("rna:")
            || hdr_low == "rna"
        {
            Some(CofoldPolymerKind::Rna)
        } else if hdr_low.contains("|protein")
            || hdr_low.contains("mol:protein")
            || hdr_low.contains("type=protein")
            || hdr_low.starts_with("protein:")
        {
            Some(CofoldPolymerKind::Protein)
        } else {
            None
        };

        let kind = match explicit_kind {
            Some(k) => k,
            None => {
                if seq.contains('U')
                    && seq
                        .chars()
                        .all(|c| matches!(c, 'A' | 'U' | 'G' | 'C' | 'N'))
                {
                    CofoldPolymerKind::Rna
                } else {
                    CofoldPolymerKind::Protein
                }
            }
        };

        let first_tok = hdr.split(['|', ' ', '\t', ':']).next().unwrap_or("").trim();
        let candidate_id = if let Some(stripped) = first_tok
            .strip_prefix("chain_")
            .or_else(|| first_tok.strip_prefix("CHAIN_"))
        {
            stripped.to_string()
        } else if (1..=4).contains(&first_tok.len())
            && first_tok.chars().all(|c| c.is_ascii_alphanumeric())
            && !matches!(
                first_tok.to_ascii_lowercase().as_str(),
                "dna" | "rna" | "prot"
            )
        {
            first_tok.to_ascii_uppercase()
        } else {
            default_chain_id_for_index(idx)
        };

        let chain_id = if !used_ids.contains(&candidate_id) {
            candidate_id
        } else {
            let mut k = idx;
            loop {
                let alt = default_chain_id_for_index(k);
                if !used_ids.contains(&alt) {
                    break alt;
                }
                k += 1;
            }
        };
        used_ids.insert(chain_id.clone());

        let entity = CofoldEntitySpec::Polymer {
            kind,
            chain_id,
            sequence: seq,
        };
        entity.validate()?;
        entities.push(entity);
    }

    Ok(entities)
}

fn default_chain_id_for_index(idx: usize) -> String {
    if idx < 26 {
        char::from(b'A' + (idx as u8)).to_string()
    } else {
        format!("C{}", idx + 1)
    }
}

/// Convert an `Octet-Draw` small-molecule text (`SMILES`, `MOL` V2000/V3000, `SDF`, or `.oct`)
/// into a validated `(CofoldLigandSpec::Smiles(canonical_smiles), Molecule)` pair.
pub fn ligand_spec_from_octet_str(
    text: &str,
    format_hint: Option<&str>,
) -> Result<(CofoldLigandSpec, Molecule), NmrError> {
    check_bounded_text(text, "ligand molecule")?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(NmrError::empty_input("empty ligand input"));
    }
    let hint = format_hint.unwrap_or("").to_ascii_lowercase();
    let mut mol = if hint == "oct"
        || trimmed.starts_with("OCTET")
        || trimmed.contains("\"format\":\"octet")
        || trimmed.starts_with('{')
    {
        let doc = read_oct(trimmed)?;
        doc.merged_molecule()
    } else if hint == "sdf" || text.contains("$$$$") {
        // Pass untrimmed `text` so blank CTfile header lines 1..3 are preserved intact!
        let records = read_sdf(text)?;
        let first = records
            .into_iter()
            .next()
            .ok_or_else(|| NmrError::empty_input("SDF contains zero molecule records"))?;
        first.molecule
    } else if hint == "mol"
        || text.contains("M  END")
        || text.contains("V2000")
        || text.contains("V3000")
    {
        // Pass untrimmed `text` so blank CTfile V2000/V3000 header lines 1..3 are preserved intact!
        read_mol(text)?
    } else {
        let first_tok = trimmed.split_whitespace().next().unwrap_or("");
        parse_smiles(first_tok)?
    };

    if mol.atoms.iter().any(|a| a.superatom_label.is_some()) {
        let _ = expand_abbreviations(&mut mol)?;
    }
    if mol.atoms.is_empty() {
        return Err(NmrError::empty_input(
            "parsed ligand molecule has zero atoms",
        ));
    }
    let can_smiles = write_canonical_smiles(&mol);
    if can_smiles.trim().is_empty() {
        return Err(NmrError::parse(
            "failed to generate canonical SMILES for ligand molecule",
        ));
    }
    Ok((CofoldLigandSpec::Smiles(can_smiles), mol))
}

/// Load a small-molecule ligand file (`.smi`, `.smiles`, `.mol`, `.sdf`, or `.oct`) via `nmr-core::octet_mol`.
pub fn ligand_spec_from_octet_file(
    path: impl AsRef<Path>,
) -> Result<(CofoldLigandSpec, Molecule), NmrError> {
    let p = path.as_ref();
    let text = read_bounded_utf8_file(p, "ligand file")?;
    let ext = p
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase());
    ligand_spec_from_octet_str(&text, ext.as_deref())
}

// ============================================================================
// 4. Experimental Interface Restraint Extractor (`extract_nmr_interface_restraints`)
// ============================================================================

/// Surface-vs-core classification of a residue's chemical shift perturbation (`CSP`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CspPerturbationClass {
    /// `2-sigma` `CSP` hit with apo `rSASA >= min_pocket_rsasa` (`0.20`): exported as active `pocket` contact.
    SurfacePocketContact,
    /// `2-sigma` `CSP` hit buried in the apo core (`rSASA < min_pocket_rsasa`): excluded from surface `pocket`!
    AllostericCorePerturbation,
    /// `2-sigma` `CSP` hit when no apo 3D model was provided (`rSASA` unavailable): retained in `pocket`
    /// **only** when the caller explicitly opted in (`allow_unfiltered_no_apo`).
    UnfilteredNoApoStructure,
    /// Surface-exposed (`rSASA >= min_pocket_rsasa`) `2-sigma` hit that is spatially isolated from the
    /// dominant surface `CSP` patch (a surface-exposed allosteric / long-range perturbation, `21.W3:P1-8`):
    /// excluded from the Boltz-2 `pocket` (which asserts direct contact) and reported separately.
    DistalSurfaceOutlier,
    /// `2-sigma` hit on a residue with no resolved apo coordinates (missing density, tag, or
    /// unaligned residue): its exposure is unknown so it is not exported to `pocket`.
    UnresolvedInApo,
    /// Residue below the Williamson iterative `2-sigma` threshold.
    BelowThreshold,
}

/// Per-residue experimental `CSP` observation and Shrake-Rupley `rSASA` classification.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReceptorCspObservation {
    pub chain_id: String,
    /// `1`-based sequence position (`1..=L_receptor`) used in `Boltz-2` `pocket` / `contact` tokens.
    pub one_based_seq_idx: usize,
    /// Author / NMR-STAR `seq_id`.
    pub seq_id: i32,
    pub res_name: String,
    pub csp_obs_ppm: f64,
    pub dh_obs_ppm: Option<f64>,
    pub dn_obs_ppm: Option<f64>,
    pub apo_rsasa: Option<f64>,
    pub is_2sigma_hit: bool,
    pub perturbation_class: CspPerturbationClass,
}

/// Active pocket or allosteric core `CSP` residue record.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PocketContactRestraint {
    pub chain_id: String,
    /// `1`-based polymer sequence index (`1..=L`).
    pub one_based_seq_idx: usize,
    pub seq_id: i32,
    pub res_name: String,
    pub csp_ppm: f64,
    pub apo_rsasa: Option<f64>,
    pub classification: CspPerturbationClass,
}

/// Token identifier in a `Boltz-2` `contact` constraint: `1`-based residue index for polymers
/// (`[A, 42]`) or atom name for small-molecule ligands (`[L, C1]`).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum BoltzContactToken {
    PolymerResidue {
        chain_id: String,
        one_based_seq_idx: usize,
    },
    LigandAtom {
        chain_id: String,
        atom_name: String,
    },
}

impl BoltzContactToken {
    pub fn chain_id(&self) -> &str {
        match self {
            Self::PolymerResidue { chain_id, .. } | Self::LigandAtom { chain_id, .. } => chain_id,
        }
    }
}

/// Experimental source of an intermolecular distance restraint.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DistanceRestraintSource {
    IntermolecularNoe,
    PreBleachingOrBounds,
    PreGamma2Rate,
}

/// Intermolecular `NOE` or `T11.4` `PRE` restraint between two subunits/molecules.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IntermolecularDistanceRestraint {
    pub source: DistanceRestraintSource,
    pub chain_1: String,
    pub seq_id_1: i32,
    pub one_based_seq_idx_1: Option<usize>,
    pub atom_1: String,
    pub chain_2: String,
    pub seq_id_2: i32,
    pub one_based_seq_idx_2: Option<usize>,
    pub atom_2: String,
    /// Lower distance bound (`A`, `0.0` when upper-bound only).
    pub lower_a: f64,
    /// Upper distance bound (`A`, `f64::INFINITY` for pure lower-bound diamagnetic-like PRE).
    pub upper_a: f64,
    /// Measurement distance tolerance `sigma_a` (`A`, `> 0`) for quadratic violation scoring.
    pub sigma_a: f64,
    /// Optional quantitative Solomon `PRE` `(gamma2_obs_s1, sigma_s1, k_eff_a6_s1)` metadata, where
    /// `k_eff_a6_s1 = K_Solomon * (4 tau_c + 3 tau_c / (1 + omega_H^2 tau_c^2))`.
    pub pre_gamma2: Option<(f64, f64, f64)>,
    /// Explicit spectrometer `1H` Larmor frequency (`MHz`) used to convert `Gamma_2` PRE rates
    /// (`Some(mhz)` for `PreGamma2Rate`, `None` otherwise; echoed in JSON & CLI diagnostics, `21F:B18`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pre_proton_mhz: Option<f64>,
    /// Attractive upper bound exported to `Boltz-2` `contact` (`None` if non-attractive, distal, or
    /// **dropped** because its upper bound exceeds the configured ceiling; never clamped down).
    pub exported_contact_max_distance_a: Option<f64>,
    pub boltz_token_1: Option<BoltzContactToken>,
    pub boltz_token_2: Option<BoltzContactToken>,
}

/// Chain-aware `RDC` restraint for multi-subunit co-folded complex scoring.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CofoldRdcRestraint {
    pub medium_id: String,
    pub chain_1: String,
    pub seq_id_1: i32,
    pub res_name_1: String,
    pub atom_name_1: String,
    pub chain_2: String,
    pub seq_id_2: i32,
    pub res_name_2: String,
    pub atom_name_2: String,
    pub bond_type: RdcBondType,
    pub d_obs_hz: f64,
    pub sigma_hz: f64,
    pub d_max_hz: f64,
    pub is_free: bool,
}

/// Collection of chain-aware `RDC` restraints in one alignment medium.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CofoldRdcMediumSet {
    pub medium_id: String,
    pub restraints: Vec<CofoldRdcRestraint>,
}

/// Chain-aware `PCS` observation for multi-subunit co-folded complex scoring.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CofoldPcsObservation {
    pub chain_id: String,
    pub seq_id: i32,
    pub atom_name: String,
    pub pcs_obs_ppm: f64,
    pub sigma_ppm: f64,
    pub is_free: bool,
}

/// Paramagnetic metal / lanthanide tag `PCS` dataset for complex pose evaluation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CofoldPcsTagSet {
    pub tag_id: String,
    /// Optional explicit 3D metal position `(x, y, z)` in `A`.
    pub metal_pos: Option<Vec3>,
    /// Optional `(chain_id, seq_id, atom_name)` anchor resolved dynamically in each pose's frame.
    pub metal_anchor: Option<(String, i32, String)>,
    /// Optional host subunit `chain_id` whose intra-chain PCSs determine the 5-param `Delta_chi` tensor.
    pub host_chain_id: Option<String>,
    pub pre_fitted_tensor: Option<DeltaChiTensor>,
    pub observations: Vec<CofoldPcsObservation>,
}

/// How experimental restraint `seq_id`s are related to the target (FASTA) sequence.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RestraintNumbering {
    /// Pick the numbering whose residue-name cross-check against the FASTA passes (apo author
    /// numbering preferred on ties). Fails closed when neither passes.
    #[default]
    Auto,
    /// Restraint `seq_id`s use the **author numbering of the apo structure** (aligned to the FASTA
    /// by sequence, so offsets, tags and missing-density gaps are handled).
    ApoAuthor,
    /// Restraint `seq_id`s already equal the 1-based FASTA / target-sequence index.
    FastaIndex,
}

impl RestraintNumbering {
    pub fn parse_token(tok: &str) -> Option<Self> {
        match tok.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "apo" | "apo-author" | "author" => Some(Self::ApoAuthor),
            "fasta" | "fasta-index" | "index" => Some(Self::FastaIndex),
            _ => None,
        }
    }
}

/// Configuration for `extract_nmr_interface_restraints`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CofoldRestraintExtractConfig {
    /// Minimum apo Shrake-Rupley `rSASA` (`default = 0.20`) required for surface `pocket` membership.
    pub min_pocket_rsasa: f64,
    /// Default Boltz-2 `pocket` `max_distance` (`default = 6.0 A`, must lie in `[4.0, 20.0] A`).
    pub pocket_max_distance_a: f64,
    /// Minimum `contact` `max_distance` floor (`default = 4.5 A`; tighter bounds are loosened to it).
    pub min_contact_distance_a: f64,
    /// Ceiling for an exported `contact` `max_distance` (`default = 14.0 A`, inside `[4.0, 20.0] A`).
    /// Restraints whose upper bound exceeds it are **dropped, never clamped**.
    pub max_contact_distance_a: f64,
    /// Solomon `PRE` distance margin (`+1.5 A`) added to `(K_eff / Gamma_2)^(1/6)` for contact upper bounds.
    pub pre_gamma2_upper_margin_a: f64,
    /// Whether `force: true` is emitted in Boltz-2 `pocket`/`contact` constraints (`default = false`).
    pub force_constraints: bool,
    /// Whether `properties: [{affinity: {binder}}]` is requested in Boltz-2 YAML (`default = false`).
    pub request_boltz2_affinity: bool,
    /// Permit exporting an **unfiltered** `pocket` when no apo structure is supplied
    /// (`default = false`: fail closed, `21.W3:P1-8`). Without exposure information, buried
    /// allosteric hits cannot be separated from surface contacts.
    pub allow_unfiltered_no_apo: bool,
    /// Build the bundle for pose re-scoring only (never exported to Boltz-2/AF3): intermolecular
    /// restraints on `SMILES`-ligand atoms are then kept for scoring instead of refused.
    pub scoring_only: bool,
    /// Relationship between restraint `seq_id`s and the target sequence (`default = Auto`).
    pub numbering: RestraintNumbering,
    /// Enable spatial clustering of surface `CSP` hits (isolated distal hits are demoted).
    pub enable_pocket_cluster_filter: bool,
    /// Single-linkage cutoff (`A`) between `C-beta` positions of surface `CSP` hits (`default 12.0`).
    pub pocket_cluster_cutoff_a: f64,
    /// Minimum size of a non-dominant surface cluster to stay in `pocket` (`default 3`).
    pub min_secondary_cluster_size: usize,
    /// Minimum amino-acid identity (`default 0.90`) of the structure-to-FASTA alignment and of the
    /// residue-name cross-check of restraint rows.
    pub min_alignment_identity: f64,
}

impl Default for CofoldRestraintExtractConfig {
    fn default() -> Self {
        Self {
            min_pocket_rsasa: DEFAULT_MIN_POCKET_RSASA,
            pocket_max_distance_a: DEFAULT_BOLTZ2_POCKET_MAX_DISTANCE_A,
            min_contact_distance_a: MIN_COFOLD_CONTACT_UPPER_A,
            max_contact_distance_a: MAX_COFOLD_CONTACT_UPPER_A,
            pre_gamma2_upper_margin_a: 1.50,
            force_constraints: false,
            request_boltz2_affinity: false,
            allow_unfiltered_no_apo: false,
            scoring_only: false,
            numbering: RestraintNumbering::Auto,
            enable_pocket_cluster_filter: true,
            pocket_cluster_cutoff_a: 12.0,
            min_secondary_cluster_size: 3,
            min_alignment_identity: 0.90,
        }
    }
}

/// Raw per-residue `CSP` input tuple `(seq_id, res_name, csp_ppm, dh_ppm_opt, dn_ppm_opt)`.
pub type RawCspObservationRow = (i32, String, f64, Option<f64>, Option<f64>);

/// Raw experimental inputs passed to [`extract_nmr_interface_restraints`].
#[derive(Clone, Debug, Default)]
pub struct CofoldRestraintInput {
    pub job_name: String,
    pub entities: Vec<CofoldEntitySpec>,
    /// Optional parsed `Octet` `Molecule` graphs keyed by ligand `chain_id`.
    pub ligand_molecules: BTreeMap<String, Molecule>,
    pub receptor_chain_id: Option<String>,
    pub binder_chain_id: Option<String>,
    /// Optional apo receptor 3D coordinate model (`.cif`/`.pdb`) for Shrake-Rupley `rSASA >= 0.20` filtering.
    pub apo_receptor_model: Option<StructureModel>,
    /// Per-residue `(seq_id, res_name, csp_ppm, dh_ppm_opt, dn_ppm_opt)` observations on the receptor.
    pub raw_csp_rows: Vec<RawCspObservationRow>,
    /// Raw intermolecular `NOE` and `PRE` distance/rate restraints.
    pub raw_distance_restraints: Vec<IntermolecularDistanceRestraint>,
    /// Optional multi-subunit `RDC` datasets.
    pub rdc_media: Vec<CofoldRdcMediumSet>,
    /// Optional multi-subunit `PCS` datasets.
    pub pcs_tags: Vec<CofoldPcsTagSet>,
    /// Optional covalent `bondedAtomPairs` for AF3 JSON v2.
    pub bonded_atom_pairs: Vec<BondedAtomPairSpec>,
    /// Optional experimental `K_d` (`uM`) from `T13.6` `CSP` isotherm fitting.
    pub experimental_kd_um: Option<f64>,
}

/// Complete, serializable NMR interface restraint bundle (`<prefix>.nmr_restraints.json` sidecar)
/// driving `export_af3_json`, `export_boltz2_yaml`, and `score_cofold_poses`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CofoldRestraintBundle {
    pub schema_version: u32,
    pub job_name: String,
    pub receptor_chain_id: String,
    pub binder_chain_id: String,
    pub entities: Vec<CofoldEntitySpec>,
    #[serde(default)]
    pub ligand_molecules: BTreeMap<String, Molecule>,
    pub min_pocket_rsasa: f64,
    pub pocket_max_distance_a: f64,
    pub force_constraints: bool,
    pub request_boltz2_affinity: bool,
    pub csp_thresholds: Option<TrimmedCspThresholds>,
    pub csp_observations: Vec<ReceptorCspObservation>,
    pub pocket_contacts: Vec<PocketContactRestraint>,
    pub allosteric_core_perturbations: Vec<PocketContactRestraint>,
    pub distance_restraints: Vec<IntermolecularDistanceRestraint>,
    #[serde(default)]
    pub rdc_media: Vec<CofoldRdcMediumSet>,
    #[serde(default)]
    pub pcs_tags: Vec<CofoldPcsTagSet>,
    #[serde(default)]
    pub bonded_atom_pairs: Vec<BondedAtomPairSpec>,
    pub experimental_kd_um: Option<f64>,
    pub af3_conditioning_note: String,
    pub diagnostics: Vec<String>,
    /// Number of restraints whose computed upper bound exceeded the configured
    /// `max_contact_distance_a` ceiling and were **dropped** from the Boltz-2 export instead of being
    /// clamped to the ceiling (`21.W3:P1-7`). They are still used, un-clamped, for pose scoring.
    #[serde(default)]
    pub dropped_weak_restraints_count: usize,
    /// Surface-exposed `2-sigma` `CSP` hits isolated from the dominant surface patch (not in `pocket`).
    #[serde(default)]
    pub distal_surface_csp_hits: Vec<PocketContactRestraint>,
    /// `SMILES`-ligand atom contacts kept for scoring only (never exportable; see `scoring_only`).
    #[serde(default)]
    pub unexportable_ligand_atom_contacts_count: usize,
    /// Per polymer chain, `1`-based target index `k` -> experimental restraint `seq_id` at position
    /// `k - 1`. Empty for a chain whose restraint numbering equals the FASTA index. Used by
    /// [`score_cofold_poses`] to re-index `1..N` AF3/Boltz pose residues onto experimental numbering.
    #[serde(default)]
    pub chain_exp_numbering: BTreeMap<String, Vec<i32>>,
    /// Which restraint numbering was resolved for the receptor (`auto` decision audit).
    #[serde(default)]
    pub numbering_resolution: String,
}

impl CofoldRestraintBundle {
    /// Convert standard single-chain [`RdcMediumSet`] slices into chain-tagged [`CofoldRdcMediumSet`]
    /// entries on `chain_id` and attach them to this bundle.
    pub fn with_chain_rdc_media(mut self, chain_id: &str, media: &[RdcMediumSet]) -> Self {
        for m in media {
            let restraints = m
                .restraints
                .iter()
                .map(|r| CofoldRdcRestraint {
                    medium_id: r.medium_id.clone(),
                    chain_1: chain_id.to_string(),
                    seq_id_1: r.seq_id_1,
                    res_name_1: r.res_name_1.clone(),
                    atom_name_1: r.atom_name_1.clone(),
                    chain_2: chain_id.to_string(),
                    seq_id_2: r.seq_id_2,
                    res_name_2: r.res_name_2.clone(),
                    atom_name_2: r.atom_name_2.clone(),
                    bond_type: r.bond_type,
                    d_obs_hz: r.d_obs_hz,
                    sigma_hz: r.sigma_hz,
                    d_max_hz: r.d_max_hz,
                    is_free: r.is_free,
                })
                .collect();
            self.rdc_media.push(CofoldRdcMediumSet {
                medium_id: m.medium_id.clone(),
                restraints,
            });
        }
        self
    }
}

// ============================================================================
// 4a. Sequence-Aligned Residue Indexing (`21.W3:P1-9`)
// ============================================================================

const ALIGN_MATCH: i32 = 3;
const ALIGN_MISMATCH: i32 = -2;
const ALIGN_GAP_OPEN: i32 = -6;
const ALIGN_GAP_EXTEND: i32 = -1;
const ALIGN_NEG_INF: i32 = i32::MIN / 4;

/// Maximum dynamic-programming cells (`observed x target`) of the structure-to-FASTA alignment.
pub const MAX_ALIGNMENT_CELLS: usize = 40_000_000;

/// One-letter code (`b'X'` = unknown) of a three-letter residue name (standard + common variants).
pub fn aa_one_letter_from_name(name: &str) -> u8 {
    match name.trim().to_ascii_uppercase().as_str() {
        "ALA" => b'A',
        "ARG" => b'R',
        "ASN" => b'N',
        "ASP" | "ASH" => b'D',
        "CYS" | "CYX" | "CYM" => b'C',
        "GLN" => b'Q',
        "GLU" | "GLH" => b'E',
        "GLY" => b'G',
        "HIS" | "HID" | "HIE" | "HIP" | "HSD" | "HSE" | "HSP" => b'H',
        "ILE" => b'I',
        "LEU" => b'L',
        "LYS" | "LYN" => b'K',
        "MET" | "MSE" => b'M',
        "PHE" => b'F',
        "PRO" => b'P',
        "SER" => b'S',
        "THR" => b'T',
        "TRP" => b'W',
        "TYR" => b'Y',
        "VAL" => b'V',
        _ => b'X',
    }
}

/// Result of aligning an observed residue string (e.g. the residues present in a coordinate file)
/// against the target (FASTA) sequence.
#[derive(Clone, Debug, PartialEq)]
pub struct SequenceAlignment {
    /// Observed residue `k` -> `0`-based target index (`None` for tag / insertion residues).
    pub observed_to_target: Vec<Option<usize>>,
    pub n_observed: usize,
    pub n_target: usize,
    pub n_aligned: usize,
    /// Aligned pairs where neither side is the unknown residue `X`.
    pub n_comparable: usize,
    pub n_identical: usize,
}

impl SequenceAlignment {
    /// Amino-acid identity over comparable aligned pairs.
    pub fn identity(&self) -> f64 {
        if self.n_comparable == 0 {
            0.0
        } else {
            self.n_identical as f64 / self.n_comparable as f64
        }
    }

    /// Fraction of observed residues that aligned to the target.
    pub fn coverage(&self) -> f64 {
        if self.n_observed == 0 {
            0.0
        } else {
            self.n_aligned as f64 / self.n_observed as f64
        }
    }
}

#[inline]
fn align_pair_score(a: u8, b: u8) -> i32 {
    if a == b'X' || b == b'X' {
        0
    } else if a == b {
        ALIGN_MATCH
    } else {
        ALIGN_MISMATCH
    }
}

/// Affine-gap (Gotoh) overlap alignment of `observed` residues onto `target` with free end gaps on
/// both sides, so N-terminal expression tags, missing termini and **internal missing-density gaps**
/// are all represented explicitly instead of compacting observed residues to `1..N_obs`.
pub fn align_observed_to_target(
    observed: &[u8],
    target: &[u8],
) -> Result<SequenceAlignment, NmrError> {
    let n = observed.len();
    let m = target.len();
    if n == 0 || m == 0 {
        return Err(NmrError::empty_input(
            "cannot align an empty residue string to the target sequence",
        ));
    }
    let w = m + 1;
    let cells = (n + 1)
        .checked_mul(w)
        .filter(|&c| c <= MAX_ALIGNMENT_CELLS)
        .ok_or_else(|| {
            NmrError::invalid_input(format!(
                "sequence alignment of {n} observed x {m} target residues exceeds the {MAX_ALIGNMENT_CELLS}-cell cap"
            ))
        })?;
    let mut tb = vec![0u8; cells];
    let mut m_prev = vec![0i32; w];
    let mut x_prev = vec![ALIGN_NEG_INF; w];
    let mut m_cur = vec![0i32; w];
    let mut x_cur = vec![ALIGN_NEG_INF; w];
    let mut y_cur = vec![ALIGN_NEG_INF; w];
    let mut y_prev = vec![ALIGN_NEG_INF; w];
    let mut best_score = ALIGN_NEG_INF;
    let mut best_ij = (0usize, 0usize);

    for i in 1..=n {
        m_cur[0] = 0;
        x_cur[0] = ALIGN_NEG_INF;
        y_cur[0] = ALIGN_NEG_INF;
        for j in 1..=m {
            let s = align_pair_score(observed[i - 1], target[j - 1]);
            let (pm, px, py) = (m_prev[j - 1], x_prev[j - 1], y_prev[j - 1]);
            let (mut bm, mut from) = (pm, 0u8);
            if px > bm {
                bm = px;
                from = 1;
            }
            if py > bm {
                bm = py;
                from = 2;
            }
            m_cur[j] = bm + s;
            let mut bits = from;
            let xo = m_prev[j] + ALIGN_GAP_OPEN;
            let xe = x_prev[j] + ALIGN_GAP_EXTEND;
            if xe > xo {
                x_cur[j] = xe;
                bits |= 4;
            } else {
                x_cur[j] = xo;
            }
            let yo = m_cur[j - 1] + ALIGN_GAP_OPEN;
            let ye = y_cur[j - 1] + ALIGN_GAP_EXTEND;
            if ye > yo {
                y_cur[j] = ye;
                bits |= 8;
            } else {
                y_cur[j] = yo;
            }
            tb[i * w + j] = bits;
            if (i == n || j == m) && m_cur[j] > best_score {
                best_score = m_cur[j];
                best_ij = (i, j);
            }
        }
        std::mem::swap(&mut m_prev, &mut m_cur);
        std::mem::swap(&mut x_prev, &mut x_cur);
        std::mem::swap(&mut y_prev, &mut y_cur);
    }
    if best_score <= ALIGN_NEG_INF / 2 {
        return Err(NmrError::invalid_input(
            "structure residues could not be aligned to the target sequence",
        ));
    }

    let (mut i, mut j) = best_ij;
    let mut state = 0u8;
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    while i > 0 && j > 0 {
        let bits = tb[i * w + j];
        match state {
            0 => {
                pairs.push((i - 1, j - 1));
                state = bits & 3;
                i -= 1;
                j -= 1;
            }
            1 => {
                state = if bits & 4 != 0 { 1 } else { 0 };
                i -= 1;
            }
            _ => {
                state = if bits & 8 != 0 { 2 } else { 0 };
                j -= 1;
            }
        }
    }
    pairs.reverse();

    let mut observed_to_target = vec![None; n];
    let mut n_comparable = 0usize;
    let mut n_identical = 0usize;
    for &(oi, tj) in &pairs {
        observed_to_target[oi] = Some(tj);
        if observed[oi] != b'X' && target[tj] != b'X' {
            n_comparable += 1;
            if observed[oi] == target[tj] {
                n_identical += 1;
            }
        }
    }
    Ok(SequenceAlignment {
        observed_to_target,
        n_observed: n,
        n_target: m,
        n_aligned: pairs.len(),
        n_comparable,
        n_identical,
    })
}

/// Explicit amino-acid identity cross-check of a structure-to-FASTA alignment.
fn require_alignment_quality(
    label: &str,
    a: &SequenceAlignment,
    min_identity: f64,
) -> Result<(), NmrError> {
    if a.n_aligned < 3.min(a.n_observed) || a.coverage() < 0.5 {
        return Err(NmrError::restraint_mismatch(format!(
            "{label}: only {}/{} observed residues align to the {}-residue target sequence (coverage {:.2} < 0.50)",
            a.n_aligned,
            a.n_observed,
            a.n_target,
            a.coverage()
        )));
    }
    if a.identity() < min_identity {
        return Err(NmrError::restraint_mismatch(format!(
            "{label}: amino-acid identity {:.3} over {} aligned residues is below the {:.2} cross-check (structure does not match the FASTA sequence)",
            a.identity(),
            a.n_comparable,
            min_identity
        )));
    }
    Ok(())
}

/// Observed `(seq_id, aa1, model_pos)` residues of one chain of a [`StructureModel`] in file order.
fn model_chain_observed(model: &StructureModel, chain_id: &str) -> Vec<(i32, u8, usize)> {
    model
        .residues
        .iter()
        .enumerate()
        .filter(|(_, r)| r.chain_id == chain_id)
        .map(|(pos, r)| (r.seq_id, r.aa1.to_ascii_uppercase(), pos))
        .collect()
}

/// Choose the apo-model chain matching `receptor_chain` (exact id, sole chain, or legacy `A`).
fn select_model_chain(model: &StructureModel, receptor_chain: &str) -> Option<String> {
    let mut chains: BTreeSet<&str> = BTreeSet::new();
    for r in &model.residues {
        chains.insert(r.chain_id.as_str());
    }
    if chains.contains(receptor_chain) {
        return Some(receptor_chain.to_string());
    }
    if chains.len() == 1 {
        return chains.iter().next().map(|s| (*s).to_string());
    }
    chains.contains("A").then(|| "A".to_string())
}

/// Gap-preserving sequence inferred from observed `(seq_id, aa1)` when no FASTA is supplied:
/// missing residues inside the observed span become `X`, so indices follow `seq_id - min_seq_id + 1`.
fn infer_target_from_observed(obs: &[(i32, u8, usize)]) -> (Vec<u8>, BTreeMap<i32, usize>) {
    let mut map = BTreeMap::new();
    let (Some(min_id), Some(max_id)) =
        (obs.iter().map(|o| o.0).min(), obs.iter().map(|o| o.0).max())
    else {
        return (Vec::new(), map);
    };
    let span = (i64::from(max_id) - i64::from(min_id) + 1) as usize;
    if span <= 3 * obs.len() + 50 {
        let mut seq = vec![b'X'; span];
        for &(sid, aa, _) in obs {
            let idx0 = (sid - min_id) as usize;
            seq[idx0] = aa;
            map.insert(sid, idx0 + 1);
        }
        (seq, map)
    } else {
        let mut seq = Vec::with_capacity(obs.len());
        for (k, &(sid, aa, _)) in obs.iter().enumerate() {
            seq.push(aa);
            map.insert(sid, k + 1);
        }
        (seq, map)
    }
}

/// Residue-indexing state of one polymer chain: target sequence, structure-to-target map and the
/// resolved relationship between experimental restraint `seq_id`s and the target index.
#[derive(Clone, Debug)]
struct ChainIndexing {
    target: Vec<u8>,
    /// Structure author `seq_id` -> `1`-based target index (aligned residues only).
    author_to_idx: BTreeMap<i32, usize>,
    /// `1`-based target index -> position in the structure model's `residues` vector.
    idx_to_pos: BTreeMap<usize, usize>,
    numbering: RestraintNumbering,
}

impl ChainIndexing {
    fn exp_to_idx(&self, seq_id: i32) -> Option<usize> {
        let l = self.target.len();
        let in_range = |v: i64| -> Option<usize> {
            (v >= 1 && (l == 0 || v as usize <= l)).then_some(v as usize)
        };
        match self.numbering {
            RestraintNumbering::FastaIndex | RestraintNumbering::Auto => {
                in_range(i64::from(seq_id))
            }
            RestraintNumbering::ApoAuthor => {
                if let Some(&idx) = self.author_to_idx.get(&seq_id) {
                    return Some(idx);
                }
                // Missing-density / unaligned residue: extend the nearest aligned residue's offset.
                let below = self.author_to_idx.range(..seq_id).next_back();
                let above = self.author_to_idx.range(seq_id..).next();
                let anchor = match (below, above) {
                    (Some(b), Some(a)) => {
                        if (seq_id - *b.0) <= (*a.0 - seq_id) {
                            Some(b)
                        } else {
                            Some(a)
                        }
                    }
                    (Some(b), None) => Some(b),
                    (None, Some(a)) => Some(a),
                    (None, None) => None,
                }?;
                in_range(*anchor.1 as i64 + i64::from(seq_id) - i64::from(*anchor.0))
            }
        }
    }

    /// Experimental `seq_id` at every `1`-based target index (length = target length).
    fn idx_to_exp(&self) -> Vec<i32> {
        let l = self.target.len();
        match self.numbering {
            RestraintNumbering::FastaIndex | RestraintNumbering::Auto => {
                (1..=l).map(|k| k as i32).collect()
            }
            RestraintNumbering::ApoAuthor => {
                let by_idx: BTreeMap<usize, i32> = self
                    .author_to_idx
                    .iter()
                    .map(|(&sid, &idx)| (idx, sid))
                    .collect();
                (1..=l)
                    .map(|k| {
                        if let Some(&sid) = by_idx.get(&k) {
                            return sid;
                        }
                        let below = by_idx.range(..k).next_back();
                        let above = by_idx.range(k..).next();
                        let anchor = match (below, above) {
                            (Some(b), Some(a)) => {
                                if k - *b.0 <= *a.0 - k {
                                    Some(b)
                                } else {
                                    Some(a)
                                }
                            }
                            (Some(b), None) => Some(b),
                            (None, Some(a)) => Some(a),
                            (None, None) => None,
                        };
                        match anchor {
                            Some((&ak, &asid)) => asid + (k as i32 - ak as i32),
                            None => k as i32,
                        }
                    })
                    .collect()
            }
        }
    }
}

/// Fraction of `(seq_id, aa1)` rows whose residue letter agrees with the target at the index the
/// given numbering assigns (rows that map outside the target count as mismatches).
fn row_name_identity(ix: &ChainIndexing, rows: &[(i32, u8)]) -> f64 {
    if rows.is_empty() {
        return 1.0;
    }
    let ok = rows
        .iter()
        .filter(|&&(sid, aa)| {
            ix.exp_to_idx(sid)
                .map(|idx| {
                    let t = ix.target.get(idx - 1).copied().unwrap_or(b'X');
                    t == b'X' || t == aa
                })
                .unwrap_or(false)
        })
        .count();
    ok as f64 / rows.len() as f64
}

#[inline]
fn is_smiles_ligand_chain(entities: &[CofoldEntitySpec], chain: &str) -> bool {
    entities.iter().any(|e| {
        matches!(
            e,
            CofoldEntitySpec::Ligand {
                chain_id,
                spec: CofoldLigandSpec::Smiles(_)
            } if chain_id == chain
        )
    })
}

fn validate_extract_config(cfg: &CofoldRestraintExtractConfig) -> Result<(), NmrError> {
    if !cfg.min_pocket_rsasa.is_finite() || !(0.0..=1.0).contains(&cfg.min_pocket_rsasa) {
        return Err(NmrError::invalid_input(format!(
            "min_pocket_rsasa ({}) must be finite in [0.0, 1.0]",
            cfg.min_pocket_rsasa
        )));
    }
    let in_boltz =
        |v: f64| v.is_finite() && (BOLTZ2_MIN_DISTANCE_A..=BOLTZ2_MAX_DISTANCE_A).contains(&v);
    if !in_boltz(cfg.pocket_max_distance_a) {
        return Err(NmrError::invalid_input(format!(
            "pocket_max_distance_a ({}) must lie in the Boltz-2 range [{BOLTZ2_MIN_DISTANCE_A}, {BOLTZ2_MAX_DISTANCE_A}] A",
            cfg.pocket_max_distance_a
        )));
    }
    if !in_boltz(cfg.min_contact_distance_a)
        || !in_boltz(cfg.max_contact_distance_a)
        || cfg.max_contact_distance_a < cfg.min_contact_distance_a
    {
        return Err(NmrError::invalid_input(format!(
            "invalid [min_contact_distance_a, max_contact_distance_a] = [{}, {}]; both must lie in [{BOLTZ2_MIN_DISTANCE_A}, {BOLTZ2_MAX_DISTANCE_A}] A with min <= max",
            cfg.min_contact_distance_a, cfg.max_contact_distance_a
        )));
    }
    if !cfg.pre_gamma2_upper_margin_a.is_finite() || cfg.pre_gamma2_upper_margin_a < 0.0 {
        return Err(NmrError::invalid_input(
            "pre_gamma2_upper_margin_a must be finite and >= 0",
        ));
    }
    if !cfg.pocket_cluster_cutoff_a.is_finite() || cfg.pocket_cluster_cutoff_a <= 0.0 {
        return Err(NmrError::invalid_input(
            "pocket_cluster_cutoff_a must be finite and > 0",
        ));
    }
    if !cfg.min_alignment_identity.is_finite()
        || cfg.min_alignment_identity <= 0.0
        || cfg.min_alignment_identity > 1.0
    {
        return Err(NmrError::invalid_input(
            "min_alignment_identity must be in (0.0, 1.0]",
        ));
    }
    Ok(())
}

/// Single-linkage clusters (union-find) of points closer than `cutoff`.
fn single_linkage_clusters(points: &[Vec3], cutoff: f64) -> Vec<Vec<usize>> {
    let n = points.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    let cut_sq = cutoff * cutoff;
    for a in 0..n {
        for b in (a + 1)..n {
            if v_dist_sq(points[a], points[b]) <= cut_sq {
                let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
                if ra != rb {
                    parent[ra.max(rb)] = ra.min(rb);
                }
            }
        }
    }
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for k in 0..n {
        let r = find(&mut parent, k);
        groups.entry(r).or_default().push(k);
    }
    groups.into_values().collect()
}

/// Validate an AlphaFold 3 `bondedAtomPairs` endpoint against the entity list: the chain must be a
/// polymer or a **CCD** ligand (a `SMILES` ligand has no addressable atom names) and the 1-based
/// residue index must lie inside the entity.
fn validate_bonded_atom_pair(
    bp: &BondedAtomPairSpec,
    entities: &[CofoldEntitySpec],
) -> Result<(), NmrError> {
    for ep in [&bp.atom1, &bp.atom2] {
        if ep.one_based_res_idx == 0 {
            return Err(NmrError::invalid_input(
                "AlphaFold 3 bondedAtomPairs require 1-based residue indices (>= 1)",
            ));
        }
        if ep.atom_name.trim().is_empty() {
            return Err(NmrError::invalid_input(format!(
                "bondedAtomPairs endpoint on chain '{}' has an empty atom name",
                ep.chain_id
            )));
        }
        let ent = entities
            .iter()
            .find(|e| e.chain_id() == ep.chain_id)
            .ok_or_else(|| {
                NmrError::invalid_input(format!(
                    "bondedAtomPairs references chain '{}' that is not a declared entity",
                    ep.chain_id
                ))
            })?;
        match ent {
            CofoldEntitySpec::Polymer { .. } => {
                let len = ent.polymer_len().unwrap_or(0);
                if ep.one_based_res_idx > len {
                    return Err(NmrError::invalid_input(format!(
                        "bondedAtomPairs residue index {} exceeds polymer chain '{}' length ({len})",
                        ep.one_based_res_idx, ep.chain_id
                    )));
                }
            }
            CofoldEntitySpec::Ligand {
                spec: CofoldLigandSpec::Smiles(_),
                ..
            } => {
                return Err(NmrError::invalid_input(format!(
                    "bondedAtomPairs endpoint on SMILES ligand chain '{}' is invalid: only polymer chains and CCD ligands expose addressable atom names",
                    ep.chain_id
                )));
            }
            CofoldEntitySpec::Ligand {
                spec: CofoldLigandSpec::CcdCodes(codes),
                ..
            } => {
                if ep.one_based_res_idx > codes.len() {
                    return Err(NmrError::invalid_input(format!(
                        "bondedAtomPairs residue index {} exceeds the {} CCD component(s) of ligand chain '{}'",
                        ep.one_based_res_idx,
                        codes.len(),
                        ep.chain_id
                    )));
                }
            }
        }
    }
    Ok(())
}

/// Extract surface-filtered `2-sigma` `CSP` pocket contacts (`rSASA >= 0.20`), classify buried
/// `AllostericCorePerturbation` shifts (`rSASA < 0.20`), demote isolated distal surface hits, and
/// canonicalize `NOE`/`PRE` intermolecular contacts into a [`CofoldRestraintBundle`].
///
/// Hardening (`21.W3` P1-7..P1-11):
/// - fails closed (`Err`) when `CSP` rows exist but no apo structure was supplied, unless
///   `allow_unfiltered_no_apo` is set;
/// - maps every restraint `seq_id` to the FASTA index by **sequence alignment** with an explicit
///   amino-acid identity cross-check (offsets, tags and missing-density gaps are preserved);
/// - **drops** (never clamps) contact bounds above `max_contact_distance_a`;
/// - refuses exportable atom-level contacts on `SMILES` ligands (their atom names cannot be derived).
pub fn extract_nmr_interface_restraints(
    input: &CofoldRestraintInput,
    cfg: &CofoldRestraintExtractConfig,
) -> Result<CofoldRestraintBundle, NmrError> {
    validate_extract_config(cfg)?;

    if !input.raw_csp_rows.is_empty()
        && input.apo_receptor_model.is_none()
        && !cfg.allow_unfiltered_no_apo
    {
        return Err(NmrError::invalid_input(
            "CSP rows were supplied without an apo receptor 3D structure: without solvent exposure, buried allosteric hits cannot be separated from surface contacts, so the unfiltered pocket export is refused (`UnfilteredNoApoStructure`). Supply --apo-structure, or opt in explicitly with --allow-unfiltered-no-apo (CofoldRestraintExtractConfig::allow_unfiltered_no_apo)",
        ));
    }

    let mut diagnostics = Vec::new();
    if !input.raw_csp_rows.is_empty() && input.apo_receptor_model.is_none() {
        diagnostics.push(
            "WARNING: UnfilteredNoApoStructure opt-in active (no apo 3D structure supplied); buried allosteric core hits cannot be separated from surface pocket contacts."
                .to_string(),
        );
    }

    // Entity list; when no polymer entities were supplied, infer gap- and offset-preserving receptor polymers from the apo model.
    let mut entities = input.entities.clone();
    let mut inferred_author_maps: BTreeMap<String, BTreeMap<i32, usize>> = BTreeMap::new();
    if !entities.iter().any(|e| e.is_polymer()) {
        if let Some(ref apo) = input.apo_receptor_model {
            let chain_ids: BTreeSet<String> =
                apo.residues.iter().map(|r| r.chain_id.clone()).collect();
            let mut inferred_polymers = Vec::new();
            for cid in chain_ids {
                let obs = model_chain_observed(apo, &cid);
                let (seq, map) = infer_target_from_observed(&obs);
                if seq.is_empty() {
                    continue;
                }
                inferred_author_maps.insert(cid.clone(), map);
                inferred_polymers.push(CofoldEntitySpec::Polymer {
                    kind: CofoldPolymerKind::Protein,
                    chain_id: cid,
                    sequence: String::from_utf8_lossy(&seq).into_owned(),
                });
            }
            if !inferred_polymers.is_empty() {
                inferred_polymers.extend(entities);
                entities = inferred_polymers;
                diagnostics.push(
                    "No FASTA supplied: polymer sequences were inferred from the apo model with 'X' placeholders for missing-density gaps; indices are relative to the first observed residue."
                        .to_string(),
                );
            }
        }
    }
    for ent in &entities {
        ent.validate()?;
    }

    let mut seen_chains = BTreeSet::new();
    for ent in &entities {
        if !seen_chains.insert(ent.chain_id().to_string()) {
            return Err(NmrError::invalid_input(format!(
                "duplicate entity chain_id '{}'",
                ent.chain_id()
            )));
        }
    }

    let receptor_chain_id = match &input.receptor_chain_id {
        Some(c) => c.trim().to_string(),
        None => entities
            .iter()
            .find(|e| e.is_polymer())
            .map(|e| e.chain_id().to_string())
            .unwrap_or_else(|| "A".to_string()),
    };
    validate_chain_id(&receptor_chain_id)?;

    let binder_chain_id = match &input.binder_chain_id {
        Some(c) => c.trim().to_string(),
        None => entities
            .iter()
            .find(|e| !e.is_polymer())
            .or_else(|| entities.iter().find(|e| e.chain_id() != receptor_chain_id))
            .map(|e| e.chain_id().to_string())
            .unwrap_or_else(|| "B".to_string()),
    };
    validate_chain_id(&binder_chain_id)?;

    for bp in &input.bonded_atom_pairs {
        validate_bonded_atom_pair(bp, &entities)?;
    }

    let polymer_lengths: BTreeMap<String, usize> = entities
        .iter()
        .filter_map(|e| e.polymer_len().map(|len| (e.chain_id().to_string(), len)))
        .collect();
    let ligand_chains: BTreeSet<String> = entities
        .iter()
        .filter(|e| !e.is_polymer())
        .map(|e| e.chain_id().to_string())
        .collect();

    // ---- Receptor residue indexing: apo structure aligned to the target sequence ----------------
    let receptor_target: Option<Vec<u8>> = entities.iter().find_map(|e| match e {
        CofoldEntitySpec::Polymer {
            kind: CofoldPolymerKind::Protein,
            chain_id,
            sequence,
        } if *chain_id == receptor_chain_id => {
            Some(sequence.trim().to_ascii_uppercase().into_bytes())
        }
        _ => None,
    });

    let mut apo_rsasa_by_idx: BTreeMap<usize, f64> = BTreeMap::new();
    let mut apo_cb_by_idx: BTreeMap<usize, Vec3> = BTreeMap::new();
    let mut receptor_ix = ChainIndexing {
        target: receptor_target.unwrap_or_default(),
        author_to_idx: BTreeMap::new(),
        idx_to_pos: BTreeMap::new(),
        numbering: RestraintNumbering::FastaIndex,
    };
    let mut have_apo_map = false;

    if let Some(ref apo) = input.apo_receptor_model {
        let apo_chain = select_model_chain(apo, &receptor_chain_id).ok_or_else(|| {
            NmrError::restraint_mismatch(format!(
                "apo structure has no chain matching receptor chain '{receptor_chain_id}'"
            ))
        })?;
        let obs = model_chain_observed(apo, &apo_chain);
        if let Some(inferred) = inferred_author_maps.get(&apo_chain) {
            for &(sid, _, pos) in &obs {
                if let Some(&idx) = inferred.get(&sid) {
                    receptor_ix.author_to_idx.insert(sid, idx);
                    receptor_ix.idx_to_pos.insert(idx, pos);
                }
            }
            have_apo_map = true;
        } else if !receptor_ix.target.is_empty() {
            let obs_letters: Vec<u8> = obs.iter().map(|o| o.1).collect();
            let aln = align_observed_to_target(&obs_letters, &receptor_ix.target)?;
            require_alignment_quality(
                &format!("apo structure chain '{apo_chain}' vs FASTA chain '{receptor_chain_id}'"),
                &aln,
                cfg.min_alignment_identity,
            )?;
            for (k, tgt) in aln.observed_to_target.iter().enumerate() {
                if let Some(t0) = tgt {
                    receptor_ix.author_to_idx.insert(obs[k].0, t0 + 1);
                    receptor_ix.idx_to_pos.insert(t0 + 1, obs[k].2);
                }
            }
            diagnostics.push(format!(
                "apo chain '{apo_chain}' aligned to FASTA chain '{receptor_chain_id}': {}/{} residues aligned, identity {:.3} (tags/gaps preserved)",
                aln.n_aligned,
                aln.n_observed,
                aln.identity()
            ));
            have_apo_map = true;
        } else {
            return Err(NmrError::restraint_mismatch(format!(
                "receptor chain '{receptor_chain_id}' has no protein sequence to align the apo structure against"
            )));
        }

        let sasa_vec = compute_shrake_rupley_rsasa(apo, &[]);
        for (&idx, &pos) in &receptor_ix.idx_to_pos {
            if let Some(s) = sasa_vec.get(pos) {
                apo_rsasa_by_idx.insert(idx, s.rsasa);
            }
            if let Some(r) = apo.residues.get(pos) {
                apo_cb_by_idx.insert(idx, r.cb.unwrap_or(r.ca));
            }
        }
    } else if !input.raw_csp_rows.is_empty() {
        diagnostics.push(
            "No apo 3D structure supplied; 2-sigma CSP hits retained without Shrake-Rupley rSASA core filtering (explicit allow_unfiltered_no_apo opt-in)."
                .to_string(),
        );
    }

    // ---- Resolve how experimental seq_ids relate to the target sequence ------------------------
    let mut name_rows: Vec<(i32, u8)> = input
        .raw_csp_rows
        .iter()
        .filter_map(|(sid, name, _, _, _)| {
            let aa = aa_one_letter_from_name(name);
            (aa != b'X').then_some((*sid, aa))
        })
        .collect();
    if name_rows.is_empty() {
        for m in &input.rdc_media {
            for r in &m.restraints {
                if r.chain_1 == receptor_chain_id {
                    let aa = aa_one_letter_from_name(&r.res_name_1);
                    if aa != b'X' {
                        name_rows.push((r.seq_id_1, aa));
                    }
                }
                if r.chain_2 == receptor_chain_id && r.seq_id_2 != r.seq_id_1 {
                    let aa = aa_one_letter_from_name(&r.res_name_2);
                    if aa != b'X' {
                        name_rows.push((r.seq_id_2, aa));
                    }
                }
            }
        }
    }
    let mut cand_apo = receptor_ix.clone();
    cand_apo.numbering = RestraintNumbering::ApoAuthor;
    let mut cand_fasta = receptor_ix.clone();
    cand_fasta.numbering = RestraintNumbering::FastaIndex;
    let have_target = !receptor_ix.target.is_empty();
    let id_apo = if have_apo_map {
        row_name_identity(&cand_apo, &name_rows)
    } else {
        0.0
    };
    let id_fasta = row_name_identity(&cand_fasta, &name_rows);
    let min_id = cfg.min_alignment_identity;
    let (resolved, resolution_note) = match cfg.numbering {
        RestraintNumbering::ApoAuthor => {
            if !have_apo_map {
                return Err(NmrError::invalid_input(
                    "numbering=apo requires an apo receptor structure",
                ));
            }
            if have_target && !name_rows.is_empty() && id_apo < min_id {
                return Err(NmrError::restraint_mismatch(format!(
                    "residue-name cross-check failed under apo author numbering (identity {id_apo:.3} < {min_id:.2})"
                )));
            }
            (
                RestraintNumbering::ApoAuthor,
                "apo author numbering (explicit)".to_string(),
            )
        }
        RestraintNumbering::FastaIndex => {
            if have_target && !name_rows.is_empty() && id_fasta < min_id {
                return Err(NmrError::restraint_mismatch(format!(
                    "residue-name cross-check failed under FASTA-index numbering (identity {id_fasta:.3} < {min_id:.2})"
                )));
            }
            (
                RestraintNumbering::FastaIndex,
                "FASTA-index numbering (explicit)".to_string(),
            )
        }
        RestraintNumbering::Auto => {
            if name_rows.is_empty() || !have_target {
                if have_apo_map {
                    let dist_sids: Vec<i32> = input
                        .raw_distance_restraints
                        .iter()
                        .flat_map(|r| {
                            let mut v = Vec::new();
                            if r.chain_1 == receptor_chain_id && r.one_based_seq_idx_1.is_none() {
                                v.push(r.seq_id_1);
                            }
                            if r.chain_2 == receptor_chain_id && r.one_based_seq_idx_2.is_none() {
                                v.push(r.seq_id_2);
                            }
                            v
                        })
                        .collect();
                    let all_apo = dist_sids.iter().all(|&s| cand_apo.exp_to_idx(s).is_some());
                    let all_fasta = dist_sids
                        .iter()
                        .all(|&s| cand_fasta.exp_to_idx(s).is_some());
                    if !dist_sids.is_empty() && !all_apo && all_fasta {
                        (
                            RestraintNumbering::FastaIndex,
                            "FASTA-index numbering (unverified by residue names; distance restraint seq_ids fit FASTA indices)"
                                .to_string(),
                        )
                    } else {
                        (
                            RestraintNumbering::ApoAuthor,
                            "apo author numbering (unverified: no residue names to cross-check)"
                                .to_string(),
                        )
                    }
                } else {
                    (
                        RestraintNumbering::FastaIndex,
                        "FASTA-index numbering (unverified: no apo structure / residue names)"
                            .to_string(),
                    )
                }
            } else if have_apo_map && id_apo >= min_id {
                (
                    RestraintNumbering::ApoAuthor,
                    format!("apo author numbering (residue-name identity {id_apo:.3}; FASTA-index identity {id_fasta:.3})"),
                )
            } else if id_fasta >= min_id {
                (
                    RestraintNumbering::FastaIndex,
                    format!("FASTA-index numbering (residue-name identity {id_fasta:.3}; apo-author identity {id_apo:.3})"),
                )
            } else {
                return Err(NmrError::restraint_mismatch(format!(
                    "restraint residue names match neither apo author numbering (identity {id_apo:.3}) nor FASTA indices (identity {id_fasta:.3}); refusing to guess the restraint numbering (cross-check threshold {min_id:.2})"
                )));
            }
        }
    };
    receptor_ix.numbering = resolved;
    diagnostics.push(format!("Restraint numbering resolved: {resolution_note}."));

    let mut chain_exp_numbering: BTreeMap<String, Vec<i32>> = BTreeMap::new();
    if have_target {
        let v = receptor_ix.idx_to_exp();
        if v.iter().enumerate().any(|(k, &e)| e != (k as i32 + 1)) {
            chain_exp_numbering.insert(receptor_chain_id.clone(), v);
        }
    }

    // Experimental `seq_id` -> 1-based target index (receptor chain: aligned numbering; other
    // polymer chains carry no structure so `seq_id` is taken as the FASTA index).
    let resolve_one_based = |chain_id: &str, seq_id: i32| -> Result<usize, NmrError> {
        if chain_id == receptor_chain_id {
            return receptor_ix.exp_to_idx(seq_id).ok_or_else(|| {
                NmrError::restraint_mismatch(format!(
                    "residue seq_id {seq_id} on receptor chain '{chain_id}' cannot be mapped into the {}-residue target sequence under the resolved numbering",
                    receptor_ix.target.len()
                ))
            });
        }
        if seq_id < 1 {
            return Err(NmrError::invalid_input(format!(
                "residue seq_id {seq_id} on chain '{chain_id}' cannot map to a 1-based sequence index"
            )));
        }
        let idx1 = seq_id as usize;
        if let Some(&len) = polymer_lengths.get(chain_id) {
            if idx1 > len {
                return Err(NmrError::restraint_mismatch(format!(
                    "1-based residue index {idx1} exceeds polymer chain '{chain_id}' length ({len})"
                )));
            }
        }
        Ok(idx1)
    };

    // 1. Williamson (2013) iterative 2-sigma CSP threshold + Shrake-Rupley rSASA pocket filtering
    let mut csp_observations: Vec<ReceptorCspObservation> =
        Vec::with_capacity(input.raw_csp_rows.len());
    let mut pocket_contacts = Vec::new();
    let mut allosteric_core_perturbations = Vec::new();
    let mut distal_surface_csp_hits = Vec::new();

    let csp_thresholds = if !input.raw_csp_rows.is_empty() {
        let mut pairs = Vec::with_capacity(input.raw_csp_rows.len());
        for &(seq_id, _, csp_val, _, _) in &input.raw_csp_rows {
            if !csp_val.is_finite() || csp_val < 0.0 {
                return Err(NmrError::invalid_input(format!(
                    "invalid negative or non-finite CSP value ({csp_val}) at seq_id {seq_id}"
                )));
            }
            pairs.push((seq_id, csp_val));
        }
        let thresh = compute_williamson_2sigma_threshold(&pairs);
        let sig_set: BTreeSet<i32> = thresh.significant_seq_ids.iter().copied().collect();

        for (seq_id, res_name, csp_val, dh_opt, dn_opt) in &input.raw_csp_rows {
            let one_based = resolve_one_based(&receptor_chain_id, *seq_id)?;
            let apo_rsasa = apo_rsasa_by_idx.get(&one_based).copied();
            let is_hit = sig_set.contains(seq_id);
            let p_class = if !is_hit {
                CspPerturbationClass::BelowThreshold
            } else if input.apo_receptor_model.is_none() {
                CspPerturbationClass::UnfilteredNoApoStructure
            } else if let Some(rs) = apo_rsasa {
                if rs >= cfg.min_pocket_rsasa {
                    CspPerturbationClass::SurfacePocketContact
                } else {
                    CspPerturbationClass::AllostericCorePerturbation
                }
            } else {
                CspPerturbationClass::UnresolvedInApo
            };
            csp_observations.push(ReceptorCspObservation {
                chain_id: receptor_chain_id.clone(),
                one_based_seq_idx: one_based,
                seq_id: *seq_id,
                res_name: res_name.clone(),
                csp_obs_ppm: *csp_val,
                dh_obs_ppm: *dh_opt,
                dn_obs_ppm: *dn_opt,
                apo_rsasa,
                is_2sigma_hit: is_hit,
                perturbation_class: p_class,
            });
        }

        // Spatial cluster filter: a surface-exposed hit isolated from the dominant surface patch is a
        // distal (allosteric / long-range) perturbation, not a direct contact (`21.W3:P1-8`).
        if cfg.enable_pocket_cluster_filter {
            let surf: Vec<usize> = csp_observations
                .iter()
                .enumerate()
                .filter(|(_, o)| {
                    o.perturbation_class == CspPerturbationClass::SurfacePocketContact
                        && apo_cb_by_idx.contains_key(&o.one_based_seq_idx)
                })
                .map(|(k, _)| k)
                .collect();
            if surf.len() >= 2 {
                let pts: Vec<Vec3> = surf
                    .iter()
                    .map(|&k| apo_cb_by_idx[&csp_observations[k].one_based_seq_idx])
                    .collect();
                let clusters = single_linkage_clusters(&pts, cfg.pocket_cluster_cutoff_a);
                if clusters.len() > 1 {
                    let cluster_weight = |c: &Vec<usize>| -> f64 {
                        c.iter()
                            .map(|&p| csp_observations[surf[p]].csp_obs_ppm)
                            .sum()
                    };
                    let primary = clusters
                        .iter()
                        .enumerate()
                        .max_by(|(ia, a), (ib, b)| {
                            cluster_weight(a)
                                .total_cmp(&cluster_weight(b))
                                .then_with(|| ib.cmp(ia))
                        })
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    let mut n_demoted = 0usize;
                    for (ci, cl) in clusters.iter().enumerate() {
                        if ci != primary && cl.len() < cfg.min_secondary_cluster_size {
                            for &p in cl {
                                csp_observations[surf[p]].perturbation_class =
                                    CspPerturbationClass::DistalSurfaceOutlier;
                                n_demoted += 1;
                            }
                        }
                    }
                    if n_demoted > 0 {
                        diagnostics.push(format!(
                            "WARNING: {n_demoted} surface-exposed 2-sigma CSP hit(s) lie > {:.1} A (C-beta single linkage) from the dominant surface patch and were demoted to DistalSurfaceOutlier (not exported as Boltz-2 pocket contacts); they may be surface-exposed allosteric perturbations.",
                            cfg.pocket_cluster_cutoff_a
                        ));
                    }
                }
            }
        }

        for o in &csp_observations {
            let rec = || PocketContactRestraint {
                chain_id: o.chain_id.clone(),
                one_based_seq_idx: o.one_based_seq_idx,
                seq_id: o.seq_id,
                res_name: o.res_name.clone(),
                csp_ppm: o.csp_obs_ppm,
                apo_rsasa: o.apo_rsasa,
                classification: o.perturbation_class,
            };
            match o.perturbation_class {
                CspPerturbationClass::SurfacePocketContact
                | CspPerturbationClass::UnfilteredNoApoStructure => pocket_contacts.push(rec()),
                CspPerturbationClass::AllostericCorePerturbation => {
                    allosteric_core_perturbations.push(rec())
                }
                CspPerturbationClass::DistalSurfaceOutlier => distal_surface_csp_hits.push(rec()),
                CspPerturbationClass::UnresolvedInApo => {
                    diagnostics.push(format!(
                        "2-sigma CSP hit at seq_id {} (target index {}) has no resolved apo coordinates (missing density / tag); excluded from the pocket export.",
                        o.seq_id, o.one_based_seq_idx
                    ));
                }
                CspPerturbationClass::BelowThreshold => {}
            }
        }

        // Deterministic sort by `(chain_id, one_based_seq_idx)`
        let key = |p: &PocketContactRestraint| (p.chain_id.clone(), p.one_based_seq_idx);
        pocket_contacts.sort_by_key(key);
        pocket_contacts.dedup_by(|a, b| {
            a.chain_id == b.chain_id && a.one_based_seq_idx == b.one_based_seq_idx
        });
        allosteric_core_perturbations.sort_by_key(key);
        distal_surface_csp_hits.sort_by_key(key);

        if !allosteric_core_perturbations.is_empty() {
            diagnostics.push(format!(
                "Classified {} buried 2-sigma CSP hit(s) (rSASA < {:.2}) as AllostericCorePerturbation and excluded them from surface pocket contacts.",
                allosteric_core_perturbations.len(),
                cfg.min_pocket_rsasa
            ));
        }

        Some(thresh)
    } else {
        None
    };

    // 2. Intermolecular NOE & PRE distance restraints -> Boltz-2 contact tokens.
    //    Upper bounds above `max_contact_distance_a` are DROPPED (never clamped to the ceiling).
    let mut distance_restraints = Vec::with_capacity(input.raw_distance_restraints.len());
    let mut dropped_weak = 0usize;
    let mut dropped_pre_gamma2 = 0usize;
    let mut total_pre_gamma2 = 0usize;
    let mut pre_proton_mhz_seen: Vec<f64> = Vec::new();
    let mut unexportable_lig_contacts = 0usize;
    for raw in &input.raw_distance_restraints {
        if !raw.lower_a.is_finite()
            || raw.lower_a < 0.0
            || raw.upper_a.is_nan()
            || raw.upper_a < raw.lower_a
            || !raw.sigma_a.is_finite()
            || raw.sigma_a <= 0.0
        {
            return Err(NmrError::invalid_input(format!(
                "invalid distance bounds [{}, {}] (sigma={}) between {}:{} and {}:{}",
                raw.lower_a,
                raw.upper_a,
                raw.sigma_a,
                raw.chain_1,
                raw.seq_id_1,
                raw.chain_2,
                raw.seq_id_2
            )));
        }

        let is_lig_1 = ligand_chains.contains(&raw.chain_1);
        let is_lig_2 = ligand_chains.contains(&raw.chain_2);

        let idx1_opt = if is_lig_1 {
            None
        } else {
            Some(match raw.one_based_seq_idx_1 {
                Some(v) if v >= 1 => v,
                _ => resolve_one_based(&raw.chain_1, raw.seq_id_1)?,
            })
        };
        let idx2_opt = if is_lig_2 {
            None
        } else {
            Some(match raw.one_based_seq_idx_2 {
                Some(v) if v >= 1 => v,
                _ => resolve_one_based(&raw.chain_2, raw.seq_id_2)?,
            })
        };

        let is_pre_gamma2 = raw.source == DistanceRestraintSource::PreGamma2Rate;
        if is_pre_gamma2 {
            total_pre_gamma2 += 1;
            if let Some(mhz) = raw.pre_proton_mhz {
                if !pre_proton_mhz_seen
                    .iter()
                    .any(|&x| (x - mhz).abs() < 1e-6)
                {
                    pre_proton_mhz_seen.push(mhz);
                }
            }
        }
        let attractive = raw.chain_1 != raw.chain_2 && raw.upper_a.is_finite();
        let mut exported_max = None;
        if attractive {
            if raw.upper_a > cfg.max_contact_distance_a {
                dropped_weak += 1;
                if is_pre_gamma2 {
                    dropped_pre_gamma2 += 1;
                }
            } else {
                exported_max = Some(raw.upper_a.max(cfg.min_contact_distance_a));
            }
        }

        // `SMILES` ligand atoms cannot be addressed in Boltz-2 `contact` tokens (the exported
        // SMILES is re-canonicalised, so user atom names do not match how Boltz names atoms).
        let touches_smiles_atom = (is_lig_1 && is_smiles_ligand_chain(&entities, &raw.chain_1))
            || (is_lig_2 && is_smiles_ligand_chain(&entities, &raw.chain_2));
        if exported_max.is_some() && touches_smiles_atom {
            if cfg.scoring_only {
                exported_max = None;
                unexportable_lig_contacts += 1;
            } else {
                return Err(NmrError::invalid_input(format!(
                    "atom-level contact {}:{}:{} <-> {}:{}:{} involves a SMILES ligand: Boltz-2/AF3 re-derive atom names from the canonicalised SMILES, so user atom names cannot be exported. Use a CCD ligand, a residue-level restraint, or build the bundle for scoring only (CofoldRestraintExtractConfig::scoring_only / `ferric cofold score`)",
                    raw.chain_1, raw.seq_id_1, raw.atom_1, raw.chain_2, raw.seq_id_2, raw.atom_2
                )));
            }
        }

        let (tok1, tok2) = match (exported_max, idx1_opt, idx2_opt) {
            (Some(_), i1, i2) => {
                let t1 = if is_lig_1 {
                    BoltzContactToken::LigandAtom {
                        chain_id: raw.chain_1.clone(),
                        atom_name: raw.atom_1.trim().to_ascii_uppercase(),
                    }
                } else {
                    BoltzContactToken::PolymerResidue {
                        chain_id: raw.chain_1.clone(),
                        one_based_seq_idx: i1.ok_or_else(|| {
                            NmrError::invalid_input("missing residue index for polymer endpoint 1")
                        })?,
                    }
                };
                let t2 = if is_lig_2 {
                    BoltzContactToken::LigandAtom {
                        chain_id: raw.chain_2.clone(),
                        atom_name: raw.atom_2.trim().to_ascii_uppercase(),
                    }
                } else {
                    BoltzContactToken::PolymerResidue {
                        chain_id: raw.chain_2.clone(),
                        one_based_seq_idx: i2.ok_or_else(|| {
                            NmrError::invalid_input("missing residue index for polymer endpoint 2")
                        })?,
                    }
                };
                (Some(t1), Some(t2))
            }
            _ => (None, None),
        };

        distance_restraints.push(IntermolecularDistanceRestraint {
            source: raw.source,
            chain_1: raw.chain_1.clone(),
            seq_id_1: raw.seq_id_1,
            one_based_seq_idx_1: idx1_opt,
            atom_1: raw.atom_1.trim().to_ascii_uppercase(),
            chain_2: raw.chain_2.clone(),
            seq_id_2: raw.seq_id_2,
            one_based_seq_idx_2: idx2_opt,
            atom_2: raw.atom_2.trim().to_ascii_uppercase(),
            lower_a: raw.lower_a,
            upper_a: raw.upper_a,
            sigma_a: raw.sigma_a,
            pre_gamma2: raw.pre_gamma2,
            pre_proton_mhz: raw.pre_proton_mhz,
            exported_contact_max_distance_a: exported_max,
            boltz_token_1: tok1,
            boltz_token_2: tok2,
        });
    }
    if total_pre_gamma2 > 0 && !pre_proton_mhz_seen.is_empty() {
        let mhz_list: Vec<String> = pre_proton_mhz_seen
            .iter()
            .map(|m| format!("{m:.1}"))
            .collect();
        diagnostics.push(format!(
            "Solomon Gamma2 PRE restraints ({total_pre_gamma2}) calibrated at explicit 1H Larmor frequency proton_mhz={} MHz.",
            mhz_list.join(", ")
        ));
    }
    if dropped_pre_gamma2 > 0 {
        diagnostics.push(format!(
            "WARNING: Boltz-2 contact export dropped {dropped_pre_gamma2}/{total_pre_gamma2} Solomon Gamma2 PRE restraint(s) because their tether-padded upper bound exceeds the {:.1} A contact export ceiling (with the default 6.0 A MTSL tether allowance + 1.5 A margin, most MTSL Gamma2 PRE restraints exceed 14.0 A and are inert in Boltz-2 export unless `--max-contact-dist 20.0` or `--pre-rigid-metal` is set; all PRE restraints remain active in `ferric cofold score`).",
            cfg.max_contact_distance_a
        ));
    }
    if dropped_weak > 0 {
        diagnostics.push(format!(
            "Dropped {dropped_weak} weak intermolecular restraint(s) whose upper bound exceeds the {:.1} A export ceiling (dropped, never clamped); they remain in the bundle with their original bounds for pose scoring.",
            cfg.max_contact_distance_a
        ));
    }
    if unexportable_lig_contacts > 0 {
        diagnostics.push(format!(
            "{unexportable_lig_contacts} SMILES-ligand atom contact(s) kept for pose scoring only (not exportable to Boltz-2/AF3)."
        ));
    }

    let job_name = if input.job_name.trim().is_empty() {
        "ferric_cofold_complex".to_string()
    } else {
        input.job_name.trim().to_string()
    };

    Ok(CofoldRestraintBundle {
        schema_version: 1,
        job_name,
        receptor_chain_id,
        binder_chain_id,
        entities,
        ligand_molecules: input.ligand_molecules.clone(),
        min_pocket_rsasa: cfg.min_pocket_rsasa,
        pocket_max_distance_a: cfg.pocket_max_distance_a,
        force_constraints: cfg.force_constraints,
        request_boltz2_affinity: cfg.request_boltz2_affinity,
        csp_thresholds,
        csp_observations,
        pocket_contacts,
        allosteric_core_perturbations,
        distance_restraints,
        rdc_media: input.rdc_media.clone(),
        pcs_tags: input.pcs_tags.clone(),
        bonded_atom_pairs: input.bonded_atom_pairs.clone(),
        experimental_kd_um: input.experimental_kd_um,
        af3_conditioning_note: "Public AlphaFold 3 JSON v2 (dialect='alphafold3', version=2) specifies polymer sequences, small-molecule SMILES/CCD ligands, modelSeeds, and covalent bondedAtomPairs only; non-covalent NMR pocket and PRE/NOE contact conditioning are enforced natively in Boltz-2 YAML v1 and downstream via `ferric cofold score` using this `.nmr_restraints.json` sidecar.".to_string(),
        diagnostics,
        dropped_weak_restraints_count: dropped_weak,
        distal_surface_csp_hits,
        unexportable_ligand_atom_contacts_count: unexportable_lig_contacts,
        chain_exp_numbering,
        numbering_resolution: resolution_note,
    })
}

// ============================================================================
// 4b. Restraint Table Parsers (`CSP`, Intermolecular `NOE`, `PRE`, `RDC`, `PCS`)
// ============================================================================

/// Extract `(seq_id, res_name, csp_ppm, dh_opt, dn_opt)` rows + optional `K_d` from a [`CspGlobalFitResult`].
pub fn csp_rows_from_global_fit(
    fit: &CspGlobalFitResult,
) -> (Vec<RawCspObservationRow>, Option<f64>) {
    let mut rows = Vec::with_capacity(fit.trajectories.len());
    if !fit.trajectories.is_empty() {
        for t in &fit.trajectories {
            let last_matched = t.points.iter().rev().find_map(|p| p.matched_peak);
            let (dh, dn) = match last_matched {
                Some((h, n)) => (Some(h - t.apo_h_ppm), Some(n - t.apo_n_ppm)),
                None => (None, None),
            };
            rows.push((t.seq_id, t.res_name.clone(), t.max_observed_csp_ppm, dh, dn));
        }
    } else {
        for r in &fit.residue_fits {
            rows.push((
                r.seq_id,
                r.res_name.clone(),
                r.max_observed_csp_ppm,
                None,
                None,
            ));
        }
    }
    let kd = fit.global_kd_um.is_finite().then_some(fit.global_kd_um);
    (rows, kd)
}

/// Parse experimental `CSP` observations from either `ferric csp` JSON (`CspGlobalFitResult`) or TSV
/// (`seq_index  seq_id  res_name  max_csp_ppm ...`, `seq_id  res_name  csp_ppm`, or `seq_id  res_name  dh_ppm  dn_ppm`).
pub fn parse_csp_restraint_str(
    text: &str,
) -> Result<(Vec<RawCspObservationRow>, Option<f64>), NmrError> {
    check_bounded_text(text, "CSP input")?;
    let trimmed_all = text.trim();
    if trimmed_all.is_empty() {
        return Err(NmrError::empty_input("CSP input is empty"));
    }
    if trimmed_all.starts_with('{') {
        let fit: CspGlobalFitResult = serde_json::from_str(trimmed_all)
            .map_err(|e| NmrError::parse(format!("invalid CSP JSON: {e}")))?;
        return Ok(csp_rows_from_global_fit(&fit));
    }

    let mut kd_opt: Option<f64> = None;
    let mut header_cols: Option<Vec<String>> = None;
    let mut rows = Vec::new();

    for (line_idx, raw_line) in text.lines().enumerate() {
        let line_no = line_idx + 1;
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(comment) = line.strip_prefix('#') {
            for tok in comment.split(['\t', ' ']) {
                if let Some(val_str) = tok.strip_prefix("global_kd_um=") {
                    if let Ok(kv) = val_str.parse::<f64>() {
                        if kv.is_finite() && kv > 0.0 {
                            kd_opt = Some(kv);
                        }
                    }
                }
            }
            continue;
        }

        let cols: Vec<&str> = if line.contains('\t') {
            line.split('\t').map(str::trim).collect()
        } else {
            line.split_whitespace().collect()
        };
        if cols.is_empty() {
            continue;
        }

        let low0 = cols[0].to_ascii_lowercase();
        if header_cols.is_none()
            && matches!(
                low0.as_str(),
                "seq_index" | "seq_id" | "resid" | "residue" | "chain_id" | "chain"
            )
        {
            header_cols = Some(cols.iter().map(|s| s.to_ascii_lowercase()).collect());
            continue;
        }

        if let Some(ref hdr) = header_cols {
            let find_col = |names: &[&str]| -> Option<usize> {
                hdr.iter().position(|h| names.iter().any(|n| h == n))
            };
            let sid_c = find_col(&["seq_id", "resid", "residue"]).ok_or_else(|| {
                NmrError::parse(format!(
                    "CSP TSV line {line_no}: header missing seq_id column"
                ))
            })?;
            if cols.len() <= sid_c {
                return Err(NmrError::parse(format!(
                    "CSP TSV line {line_no}: truncated row"
                )));
            }
            let seq_id: i32 = cols[sid_c].parse().map_err(|e| {
                NmrError::parse(format!(
                    "CSP TSV line {line_no}: invalid seq_id '{}': {e}",
                    cols[sid_c]
                ))
            })?;
            let res_name = find_col(&["res_name", "resname", "aa", "res"])
                .and_then(|c| cols.get(c))
                .map(|s| (*s).to_string())
                .unwrap_or_else(|| "UNK".to_string());
            let dh = find_col(&["dh_ppm", "delta_h_ppm", "dh"])
                .and_then(|c| cols.get(c))
                .and_then(|s| s.parse::<f64>().ok())
                .filter(|v| v.is_finite());
            let dn = find_col(&["dn_ppm", "delta_n_ppm", "dn"])
                .and_then(|c| cols.get(c))
                .and_then(|s| s.parse::<f64>().ok())
                .filter(|v| v.is_finite());
            let csp_val = if let Some(cc) =
                find_col(&["max_csp_ppm", "csp_ppm", "csp", "ddelta_ppm"])
            {
                let raw_c = cols.get(cc).ok_or_else(|| {
                    NmrError::parse(format!("CSP TSV line {line_no}: missing CSP column value"))
                })?;
                let v: f64 = raw_c.parse().map_err(|e| {
                    NmrError::parse(format!(
                        "CSP TSV line {line_no}: invalid CSP '{raw_c}': {e}"
                    ))
                })?;
                if !v.is_finite() || v < 0.0 {
                    return Err(NmrError::parse(format!(
                        "CSP TSV line {line_no}: non-finite or negative CSP ({v})"
                    )));
                }
                v
            } else if let (Some(dh_v), Some(dn_v)) = (dh, dn) {
                combined_amide_csp_ppm(dh_v, dn_v, &res_name)
            } else {
                return Err(NmrError::parse(format!(
                    "CSP TSV line {line_no}: header must contain csp_ppm/max_csp_ppm or (dh_ppm, dn_ppm)"
                )));
            };
            rows.push((seq_id, res_name, csp_val, dh, dn));
        } else if cols.len() >= 3 && cols[1].parse::<f64>().is_err() {
            // Headerless `seq_id  res_name  csp_ppm` or `seq_id  res_name  dh_ppm  dn_ppm`
            let seq_id: i32 = cols[0].parse().map_err(|e| {
                NmrError::parse(format!(
                    "CSP TSV line {line_no}: invalid seq_id '{}': {e}",
                    cols[0]
                ))
            })?;
            let res_name = cols[1].to_string();
            let v1: f64 = cols[2].parse().map_err(|e| {
                NmrError::parse(format!(
                    "CSP TSV line {line_no}: invalid float '{}': {e}",
                    cols[2]
                ))
            })?;
            if cols.len() >= 4 {
                if let Ok(v2) = cols[3].parse::<f64>() {
                    if !v1.is_finite() || !v2.is_finite() {
                        return Err(NmrError::parse(format!(
                            "CSP TSV line {line_no}: non-finite dH/dN"
                        )));
                    }
                    let c = combined_amide_csp_ppm(v1, v2, &res_name);
                    rows.push((seq_id, res_name, c, Some(v1), Some(v2)));
                    continue;
                }
            }
            if !v1.is_finite() || v1 < 0.0 {
                return Err(NmrError::parse(format!(
                    "CSP TSV line {line_no}: invalid CSP ({v1})"
                )));
            }
            rows.push((seq_id, res_name, v1, None, None));
        } else if cols.len() >= 2 {
            // Headerless `seq_id  csp_ppm`
            let seq_id: i32 = cols[0].parse().map_err(|e| {
                NmrError::parse(format!(
                    "CSP TSV line {line_no}: invalid seq_id '{}': {e}",
                    cols[0]
                ))
            })?;
            let csp_val: f64 = cols[1].parse().map_err(|e| {
                NmrError::parse(format!(
                    "CSP TSV line {line_no}: invalid csp_ppm '{}': {e}",
                    cols[1]
                ))
            })?;
            if !csp_val.is_finite() || csp_val < 0.0 {
                return Err(NmrError::parse(format!(
                    "CSP TSV line {line_no}: invalid CSP ({csp_val})"
                )));
            }
            rows.push((seq_id, "UNK".to_string(), csp_val, None, None));
        } else {
            return Err(NmrError::parse(format!(
                "CSP TSV line {line_no}: expected at least 2 columns"
            )));
        }
    }

    if rows.is_empty() {
        return Err(NmrError::empty_input(
            "CSP table contains zero residue rows",
        ));
    }
    Ok((rows, kd_opt))
}

/// Parse intermolecular `NOE` distance restraints from TSV text.
///
/// Supported row schemas:
/// - `chain_1  seq_id_1  atom_1  chain_2  seq_id_2  atom_2  upper_a  [lower_a]  [sigma_a]`
/// - For small-molecule ligand `chain_2`, `chain_1  seq_id_1  atom_1  chain_2  atom_2  upper_a  [lower_a]  [sigma_a]`
///   (`seq_id_2` defaults to `1`).
pub fn parse_intermolecular_noe_tsv_str(
    text: &str,
) -> Result<Vec<IntermolecularDistanceRestraint>, NmrError> {
    check_bounded_text(text, "intermolecular NOE TSV")?;
    let mut out = Vec::new();
    for (line_idx, raw_line) in text.lines().enumerate() {
        let line_no = line_idx + 1;
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = if line.contains('\t') {
            line.split('\t').map(str::trim).collect()
        } else {
            line.split_whitespace().collect()
        };
        if cols.is_empty() {
            continue;
        }
        let low0 = cols[0].to_ascii_lowercase();
        if matches!(low0.as_str(), "chain_1" | "chain1" | "ch1") {
            continue;
        }
        if cols.len() < 6 {
            return Err(NmrError::parse(format!(
                "NOE TSV line {line_no}: expected >= 6 columns, got {}",
                cols.len()
            )));
        }
        let chain_1 = cols[0].to_string();
        let seq_id_1: i32 = cols[1].parse().map_err(|e| {
            NmrError::parse(format!(
                "NOE TSV line {line_no}: invalid seq_id_1 '{}': {e}",
                cols[1]
            ))
        })?;
        let atom_1 = cols[2].to_ascii_uppercase();
        let chain_2 = cols[3].to_string();

        // Distinguish `chain_2  seq_id_2  atom_2  upper_a` vs ligand `chain_2  atom_2  upper_a`
        let (seq_id_2, atom_2, rem_start) = if let Ok(s2) = cols[4].parse::<i32>() {
            if cols.len() < 7 {
                return Err(NmrError::parse(format!(
                    "NOE TSV line {line_no}: missing upper_a column"
                )));
            }
            (s2, cols[5].to_ascii_uppercase(), 6usize)
        } else {
            (1, cols[4].to_ascii_uppercase(), 5usize)
        };

        let upper_a: f64 = if let Some(cls) = NoeIntensityClass::parse_token(cols[rem_start]) {
            cls.upper_a()
        } else {
            cols[rem_start].parse().map_err(|e| {
                NmrError::parse(format!(
                    "NOE TSV line {line_no}: invalid upper_a '{}': {e}",
                    cols[rem_start]
                ))
            })?
        };
        let lower_a: f64 = match cols
            .get(rem_start + 1)
            .copied()
            .filter(|s| !s.is_empty() && *s != ".")
        {
            Some(s) => s.parse().map_err(|e| {
                NmrError::parse(format!(
                    "NOE TSV line {line_no}: invalid lower_a '{s}': {e}"
                ))
            })?,
            None => 1.80,
        };
        let sigma_a: f64 = match cols
            .get(rem_start + 2)
            .copied()
            .filter(|s| !s.is_empty() && *s != ".")
        {
            Some(s) => s.parse().map_err(|e| {
                NmrError::parse(format!(
                    "NOE TSV line {line_no}: invalid sigma_a '{s}': {e}"
                ))
            })?,
            None => 0.50,
        };
        if !upper_a.is_finite()
            || !lower_a.is_finite()
            || !sigma_a.is_finite()
            || lower_a < 0.0
            || upper_a < lower_a
            || sigma_a <= 0.0
        {
            return Err(NmrError::parse(format!(
                "NOE TSV line {line_no}: invalid bounds [{lower_a}, {upper_a}] or sigma {sigma_a}"
            )));
        }

        out.push(IntermolecularDistanceRestraint {
            source: DistanceRestraintSource::IntermolecularNoe,
            chain_1,
            seq_id_1,
            one_based_seq_idx_1: None,
            atom_1,
            chain_2,
            seq_id_2,
            one_based_seq_idx_2: None,
            atom_2,
            lower_a,
            upper_a,
            sigma_a,
            pre_gamma2: None,
            pre_proton_mhz: None,
            exported_contact_max_distance_a: None,
            boltz_token_1: None,
            boltz_token_2: None,
        });
    }
    if out.is_empty() {
        return Err(NmrError::empty_input(
            "intermolecular NOE TSV contains zero restraints",
        ));
    }
    Ok(out)
}

/// Parse intermolecular `T11.4` `PRE` bleaching / distance-bound / Solomon `Gamma_2` restraints
/// using [`parse_intermolecular_pre_tsv_str_with_params`].
///
/// For `Gamma_2` rate tables, `tau_c_ns` MUST be provided in the TSV header (`# tau_c_ns: <ns>`);
/// otherwise this function fails closed (`Err`) instead of assuming a bogus constant (`21.W3:P0-4`).
pub fn parse_intermolecular_pre_tsv_str(
    text: &str,
    default_anchor: Option<(&str, i32, &str)>,
    default_target_chain: &str,
    gamma2_upper_margin_a: f64,
) -> Result<Vec<IntermolecularDistanceRestraint>, NmrError> {
    parse_intermolecular_pre_tsv_str_with_overrides(
        text,
        default_anchor,
        default_target_chain,
        None,
        None,
        None,
        None,
        gamma2_upper_margin_a,
    )
}

/// Parse intermolecular `T11.4` `PRE` bleaching / distance-bound / Solomon `Gamma_2` restraints from
/// TSV text with explicit [`PreSolomonParams`] (mandatory `tau_c_ns` and `proton_mhz`, plus `spin_s`
/// and `tether_pad_a`).
///
/// Header directives:
/// - `# anchor: <chain_id> <seq_id> <atom_name>`
/// - `# tau_c_ns: <f64>` (required for `gamma2` unless `solomon_params` is `Some`)
/// - `# proton_mhz: <f64>` (required for `gamma2` unless `solomon_params` is `Some`; no silent 600 MHz default, `21F:B18`)
/// - `# spin_s: <f64>` (default `0.5`)
/// - `# tether_pad_a: <f64>` (default `6.0 A` for MTSL; `0.0 A` for rigid paramagnetic metal)
/// - `# mode: bounds | gamma2 | bleach`
pub fn parse_intermolecular_pre_tsv_str_with_params(
    text: &str,
    default_anchor: Option<(&str, i32, &str)>,
    default_target_chain: &str,
    solomon_params: Option<&PreSolomonParams>,
) -> Result<Vec<IntermolecularDistanceRestraint>, NmrError> {
    parse_intermolecular_pre_tsv_str_with_overrides(
        text,
        default_anchor,
        default_target_chain,
        solomon_params.map(|p| p.tau_c_ns),
        solomon_params.map(|p| p.proton_mhz),
        solomon_params.map(|p| p.spin_s),
        solomon_params.map(|p| p.tether_pad_a),
        solomon_params.map(|p| p.upper_margin_a).unwrap_or(1.5),
    )
}

/// Parse intermolecular `PRE` TSV with independent CLI/caller overrides for `tau_c_ns`, `proton_mhz`,
/// `spin_s`, `tether_pad_a`, and `gamma2_upper_margin_a` (`21F:B18`).
///
/// For `gamma2` rows, BOTH `tau_c_ns` and `proton_mhz` MUST be explicitly provided via either the
/// override arguments or `# tau_c_ns:` / `# proton_mhz:` TSV headers (fails closed otherwise).
#[allow(clippy::too_many_arguments)]
pub fn parse_intermolecular_pre_tsv_str_with_overrides(
    text: &str,
    default_anchor: Option<(&str, i32, &str)>,
    default_target_chain: &str,
    cli_tau_c_ns: Option<f64>,
    cli_proton_mhz: Option<f64>,
    cli_spin_s: Option<f64>,
    cli_tether_pad_a: Option<f64>,
    gamma2_upper_margin_a: f64,
) -> Result<Vec<IntermolecularDistanceRestraint>, NmrError> {
    check_bounded_text(text, "intermolecular PRE TSV")?;
    let mut header_anchor: Option<(String, i32, String)> =
        default_anchor.map(|(c, s, a)| (c.to_string(), s, a.to_ascii_uppercase()));
    let mut header_tau_c_ns: Option<f64> = cli_tau_c_ns;
    let mut header_proton_mhz: Option<f64> = cli_proton_mhz;
    let mut header_spin_s: f64 = cli_spin_s.unwrap_or(DEFAULT_PRE_ELECTRON_SPIN);
    let mut header_tether_pad_a: f64 = cli_tether_pad_a.unwrap_or(DEFAULT_MTSL_TETHER_PAD_A);
    let header_upper_margin_a: f64 = gamma2_upper_margin_a;
    let mut mode_hint: Option<String> = None;
    let mut out = Vec::new();

    for (line_idx, raw_line) in text.lines().enumerate() {
        let line_no = line_idx + 1;
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(comment) = line.strip_prefix('#') {
            let c = comment.trim();
            if let Some(rest) = c
                .strip_prefix("anchor:")
                .or_else(|| c.strip_prefix("site_anchor:"))
            {
                let parts: Vec<&str> = rest.split_whitespace().collect();
                if parts.len() >= 3 {
                    let sid: i32 = parts[1].parse().map_err(|e| {
                        NmrError::parse(format!(
                            "PRE TSV line {line_no}: invalid # anchor seq_id '{}': {e}",
                            parts[1]
                        ))
                    })?;
                    header_anchor =
                        Some((parts[0].to_string(), sid, parts[2].to_ascii_uppercase()));
                } else if parts.len() == 2 {
                    let sid: i32 = parts[0].parse().map_err(|e| {
                        NmrError::parse(format!(
                            "PRE TSV line {line_no}: invalid # site_anchor seq_id '{}': {e}",
                            parts[0]
                        ))
                    })?;
                    header_anchor = Some(("B".to_string(), sid, parts[1].to_ascii_uppercase()));
                }
            } else if let Some(rest) = c
                .strip_prefix("tau_c_ns:")
                .or_else(|| c.strip_prefix("tau_c:"))
            {
                let tv: f64 = rest.trim().parse().map_err(|e| {
                    NmrError::parse(format!("PRE TSV line {line_no}: invalid # tau_c_ns: {e}"))
                })?;
                if !tv.is_finite() || !(0.05..=1.0e4).contains(&tv) {
                    return Err(NmrError::parse(format!(
                        "PRE TSV line {line_no}: # tau_c_ns ({tv}) must be finite in [0.05, 10000] ns"
                    )));
                }
                if cli_tau_c_ns.is_none() {
                    header_tau_c_ns = Some(tv);
                }
            } else if let Some(rest) = c
                .strip_prefix("proton_mhz:")
                .or_else(|| c.strip_prefix("field_mhz:"))
                .or_else(|| c.strip_prefix("omega_h_mhz:"))
            {
                let fv: f64 = rest.trim().parse().map_err(|e| {
                    NmrError::parse(format!("PRE TSV line {line_no}: invalid # proton_mhz: {e}"))
                })?;
                if !fv.is_finite() || !(1.0..=5.0e3).contains(&fv) {
                    return Err(NmrError::parse(format!(
                        "PRE TSV line {line_no}: # proton_mhz ({fv}) must be finite in [1, 5000] MHz"
                    )));
                }
                if cli_proton_mhz.is_none() {
                    header_proton_mhz = Some(fv);
                }
            } else if let Some(rest) = c.strip_prefix("spin_s:") {
                let sv: f64 = rest.trim().parse().map_err(|e| {
                    NmrError::parse(format!("PRE TSV line {line_no}: invalid # spin_s: {e}"))
                })?;
                if !sv.is_finite() || !(0.5..=8.0).contains(&sv) {
                    return Err(NmrError::parse(format!(
                        "PRE TSV line {line_no}: # spin_s ({sv}) must be finite in [0.5, 8.0]"
                    )));
                }
                if cli_spin_s.is_none() {
                    header_spin_s = sv;
                }
            } else if let Some(rest) = c
                .strip_prefix("tether_pad_a:")
                .or_else(|| c.strip_prefix("mtsl_pad_a:"))
            {
                let pv: f64 = rest.trim().parse().map_err(|e| {
                    NmrError::parse(format!(
                        "PRE TSV line {line_no}: invalid # tether_pad_a: {e}"
                    ))
                })?;
                if !pv.is_finite() || !(0.0..=15.0).contains(&pv) {
                    return Err(NmrError::parse(format!(
                        "PRE TSV line {line_no}: # tether_pad_a ({pv}) must be finite in [0, 15] A"
                    )));
                }
                if cli_tether_pad_a.is_none() {
                    header_tether_pad_a = pv;
                }
            } else if let Some(rest) = c.strip_prefix("mode:") {
                mode_hint = Some(rest.trim().to_ascii_lowercase());
            }
            continue;
        }

        let cols: Vec<&str> = if line.contains('\t') {
            line.split('\t').map(str::trim).collect()
        } else {
            line.split_whitespace().collect()
        };
        if cols.is_empty() {
            continue;
        }
        let low0 = cols[0].to_ascii_lowercase();
        if matches!(
            low0.as_str(),
            "chain_1" | "site_chain" | "site_id" | "seq_id" | "target_chain" | "resid"
        ) {
            if cols.iter().any(|c| c.eq_ignore_ascii_case("lower_a")) {
                mode_hint = Some("bounds".to_string());
            } else if cols.iter().any(|c| c.eq_ignore_ascii_case("gamma2_s1")) {
                mode_hint = Some("gamma2".to_string());
            }
            continue;
        }

        // Determine if row carries two explicit endpoints (`>= 8` cols or `cols[3]` is chain + `cols[4]` seq_id/atom)
        let (c1, s1, a1, c2, s2, a2, rem): (String, i32, String, String, i32, String, &[&str]) =
            if cols.len() >= 8 && cols[1].parse::<i32>().is_ok() && cols[4].parse::<i32>().is_ok() {
                let s1_v: i32 = cols[1].parse().map_err(|e| {
                    NmrError::parse(format!("PRE line {line_no}: invalid seq_id_1: {e}"))
                })?;
                let s2_v: i32 = cols[4].parse().map_err(|e| {
                    NmrError::parse(format!("PRE line {line_no}: invalid seq_id_2: {e}"))
                })?;
                (
                    cols[0].to_string(),
                    s1_v,
                    cols[2].to_ascii_uppercase(),
                    cols[3].to_string(),
                    s2_v,
                    cols[5].to_ascii_uppercase(),
                    &cols[6..],
                )
            } else if cols.len() >= 7
                && cols[1].parse::<i32>().is_ok()
                && cols[4].parse::<f64>().is_err()
                && cols[5].parse::<f64>().is_ok()
            {
                // Ligand site `chain_1( polymer ) seq_id_1 atom_1 chain_2( ligand ) atom_2 val1 val2`
                let s1_v: i32 = cols[1].parse().map_err(|e| {
                    NmrError::parse(format!("PRE line {line_no}: invalid seq_id_1: {e}"))
                })?;
                (
                    cols[0].to_string(),
                    s1_v,
                    cols[2].to_ascii_uppercase(),
                    cols[3].to_string(),
                    1,
                    cols[4].to_ascii_uppercase(),
                    &cols[5..],
                )
            } else {
                let (anc_c, anc_s, anc_a) = header_anchor.clone().ok_or_else(|| {
                    NmrError::parse(format!(
                        "PRE TSV line {line_no}: single-endpoint row requires `# anchor: <chain> <seq_id> <atom>`"
                    ))
                })?;
                if cols[0].parse::<i32>().is_ok() {
                    if cols.len() < 3 {
                        return Err(NmrError::parse(format!(
                            "PRE TSV line {line_no}: truncated row"
                        )));
                    }
                    let ts: i32 = cols[0].parse().map_err(|e| {
                        NmrError::parse(format!("PRE line {line_no}: invalid seq_id: {e}"))
                    })?;
                    (
                        default_target_chain.to_string(),
                        ts,
                        cols[1].to_ascii_uppercase(),
                        anc_c,
                        anc_s,
                        anc_a,
                        &cols[2..],
                    )
                } else {
                    if cols.len() < 4 {
                        return Err(NmrError::parse(format!(
                            "PRE TSV line {line_no}: truncated row"
                        )));
                    }
                    let ts: i32 = cols[1].parse().map_err(|e| {
                        NmrError::parse(format!("PRE line {line_no}: invalid seq_id: {e}"))
                    })?;
                    (
                        cols[0].to_string(),
                        ts,
                        cols[2].to_ascii_uppercase(),
                        anc_c,
                        anc_s,
                        anc_a,
                        &cols[3..],
                    )
                }
            };

        if rem.is_empty() {
            return Err(NmrError::parse(format!(
                "PRE TSV line {line_no}: missing numeric PRE observation"
            )));
        }

        if rem[0].eq_ignore_ascii_case("bleach") || rem[0].eq_ignore_ascii_case("bleached") {
            let r_bleach: f64 = rem.get(1).copied().unwrap_or("11.0").parse().map_err(|e| {
                NmrError::parse(format!("PRE line {line_no}: invalid bleach radius: {e}"))
            })?;
            if !r_bleach.is_finite() || r_bleach <= 0.0 {
                return Err(NmrError::parse(format!(
                    "PRE line {line_no}: bleach radius must be > 0"
                )));
            }
            out.push(IntermolecularDistanceRestraint {
                source: DistanceRestraintSource::PreBleachingOrBounds,
                chain_1: c1,
                seq_id_1: s1,
                one_based_seq_idx_1: None,
                atom_1: a1,
                chain_2: c2,
                seq_id_2: s2,
                one_based_seq_idx_2: None,
                atom_2: a2,
                lower_a: 1.80,
                upper_a: r_bleach,
                sigma_a: 1.00,
                pre_gamma2: None,
                pre_proton_mhz: None,
                exported_contact_max_distance_a: None,
                boltz_token_1: None,
                boltz_token_2: None,
            });
            continue;
        }

        let is_bounds = mode_hint.as_deref() == Some("bounds");
        if is_bounds {
            if rem.len() < 2 {
                return Err(NmrError::parse(format!(
                    "PRE TSV line {line_no}: bounds mode requires lower_a and upper_a"
                )));
            }
            let lower_a: f64 = rem[0].parse().map_err(|e| {
                NmrError::parse(format!("PRE line {line_no}: invalid lower_a: {e}"))
            })?;
            let upper_a: f64 = if rem[1].eq_ignore_ascii_case("inf") {
                f64::INFINITY
            } else {
                rem[1].parse().map_err(|e| {
                    NmrError::parse(format!("PRE line {line_no}: invalid upper_a: {e}"))
                })?
            };
            let sigma_a: f64 = match rem.get(2).copied().filter(|s| !s.is_empty() && *s != ".") {
                Some(s) => s.parse().map_err(|e| {
                    NmrError::parse(format!("PRE line {line_no}: invalid sigma_a: {e}"))
                })?,
                None => 1.00,
            };
            out.push(IntermolecularDistanceRestraint {
                source: DistanceRestraintSource::PreBleachingOrBounds,
                chain_1: c1,
                seq_id_1: s1,
                one_based_seq_idx_1: None,
                atom_1: a1,
                chain_2: c2,
                seq_id_2: s2,
                one_based_seq_idx_2: None,
                atom_2: a2,
                lower_a,
                upper_a,
                sigma_a,
                pre_gamma2: None,
                pre_proton_mhz: None,
                exported_contact_max_distance_a: None,
                boltz_token_1: None,
                boltz_token_2: None,
            });
        } else {
            let tau_c_ns = header_tau_c_ns.ok_or_else(|| {
                NmrError::invalid_input(format!(
                    "PRE line {line_no}: Gamma_2 PRE conversion requires rotational correlation time tau_c_ns (--pre-tau-c or `# tau_c_ns: <ns>` header); no default tau_c is assumed (21.W3:P0-4)"
                ))
            })?;
            let proton_mhz = header_proton_mhz.ok_or_else(|| {
                NmrError::invalid_input(format!(
                    "PRE line {line_no}: Gamma_2 PRE conversion requires explicit 1H spectrometer Larmor frequency proton_mhz (--pre-proton-mhz / --proton-mhz or `# proton_mhz: <MHz>` header); no default 600 MHz is assumed (21F:B18)"
                ))
            })?;
            let params = PreSolomonParams {
                tau_c_ns,
                proton_mhz,
                spin_s: header_spin_s,
                tether_pad_a: header_tether_pad_a,
                upper_margin_a: header_upper_margin_a,
            };
            let k_eff = params.k_eff_a6_s1()?;

            let gamma2: f64 = rem[0].parse().map_err(|e| {
                NmrError::parse(format!(
                    "PRE line {line_no}: invalid gamma2_s1 '{}': {e}",
                    rem[0]
                ))
            })?;
            let sigma_g2: f64 = match rem.get(1).copied().filter(|s| !s.is_empty() && *s != ".") {
                Some(s) => s.parse().map_err(|e| {
                    NmrError::parse(format!("PRE line {line_no}: invalid sigma_s1 '{s}': {e}"))
                })?,
                None => 2.0,
            };
            if !gamma2.is_finite() || gamma2 < 0.0 || !sigma_g2.is_finite() || sigma_g2 <= 0.0 {
                return Err(NmrError::parse(format!(
                    "PRE line {line_no}: non-physical gamma2/sigma values"
                )));
            }
            let (lower_a, upper_a) = solomon_gamma2_to_distance_bounds(
                gamma2,
                sigma_g2,
                k_eff,
                params.tether_pad_a,
                params.upper_margin_a,
            );
            out.push(IntermolecularDistanceRestraint {
                source: DistanceRestraintSource::PreGamma2Rate,
                chain_1: c1,
                seq_id_1: s1,
                one_based_seq_idx_1: None,
                atom_1: a1,
                chain_2: c2,
                seq_id_2: s2,
                one_based_seq_idx_2: None,
                atom_2: a2,
                lower_a,
                upper_a,
                sigma_a: 1.00,
                pre_gamma2: Some((gamma2, sigma_g2, k_eff)),
                pre_proton_mhz: Some(proton_mhz),
                exported_contact_max_distance_a: None,
                boltz_token_1: None,
                boltz_token_2: None,
            });
        }
    }

    if out.is_empty() {
        return Err(NmrError::empty_input("PRE TSV contains zero restraints"));
    }
    Ok(out)
}

#[inline]
fn sixth_root(x: f64) -> f64 {
    det_cbrt(x.max(0.0).sqrt())
}

/// Convert a Solomon `Gamma_2` (`s^-1`) and `sigma` (`s^-1`) into `(lower_a, upper_a)` bounds
/// including configurable MTSL `tether_pad_a` (`6.0 A` for MTSL, `0.0 A` for rigid metal).
pub fn solomon_gamma2_to_distance_bounds(
    gamma2_s1: f64,
    sigma_s1: f64,
    k_eff_a6_s1: f64,
    tether_pad_a: f64,
    upper_margin_a: f64,
) -> (f64, f64) {
    let pad = tether_pad_a.max(0.0);
    let g_hi = (gamma2_s1 + 2.0 * sigma_s1).max(0.5);
    let r_min = (sixth_root(k_eff_a6_s1 / g_hi) - pad - 1.0).max(1.80);
    // Near-zero PRE (`gamma2 <= 2 * sigma` and `< 3.0 s^-1`) gives a lower-bound exclusion zone only!
    if gamma2_s1 <= 2.0 * sigma_s1 && gamma2_s1 < 3.0 {
        (r_min, f64::INFINITY)
    } else {
        let g_lo = (gamma2_s1 - 2.0 * sigma_s1).max(0.5 * gamma2_s1).max(0.5);
        let r_max = sixth_root(k_eff_a6_s1 / g_lo) + pad + upper_margin_a.max(0.0);
        (r_min.min(r_max - 0.5), r_max)
    }
}

/// Parse chain-aware `RDC` TSV text (accepts both standard `nmr-core::rdc` TSVs defaulting to
/// `default_chain` and multi-chain TSVs prefixed by `chain_1` / `chain_id`).
pub fn parse_cofold_rdc_tsv_str(
    text: &str,
    default_chain: &str,
) -> Result<Vec<CofoldRdcMediumSet>, NmrError> {
    check_bounded_text(text, "RDC TSV")?;
    // First check if standard single-chain parser succeeds directly unless explicit `chain_1`/`chain` header is present
    let has_chain_header = text.lines().any(|l| {
        let t = l.trim();
        !t.is_empty()
            && !t.starts_with('#')
            && t.split_whitespace().any(|c| {
                matches!(
                    c.to_ascii_lowercase().as_str(),
                    "chain" | "chain_id" | "chain_1"
                )
            })
    });
    if !has_chain_header {
        let single = crate::rdc::parse_rdc_tsv_str(text).map_err(NmrError::parse)?;
        let bundle = CofoldRestraintBundle {
            schema_version: 1,
            job_name: String::new(),
            receptor_chain_id: default_chain.to_string(),
            binder_chain_id: "B".to_string(),
            entities: Vec::new(),
            ligand_molecules: BTreeMap::new(),
            min_pocket_rsasa: DEFAULT_MIN_POCKET_RSASA,
            pocket_max_distance_a: DEFAULT_BOLTZ2_POCKET_MAX_DISTANCE_A,
            force_constraints: false,
            request_boltz2_affinity: false,
            csp_thresholds: None,
            csp_observations: Vec::new(),
            pocket_contacts: Vec::new(),
            allosteric_core_perturbations: Vec::new(),
            distance_restraints: Vec::new(),
            rdc_media: Vec::new(),
            pcs_tags: Vec::new(),
            bonded_atom_pairs: Vec::new(),
            experimental_kd_um: None,
            af3_conditioning_note: String::new(),
            diagnostics: Vec::new(),
            dropped_weak_restraints_count: 0,
            distal_surface_csp_hits: Vec::new(),
            unexportable_ligand_atom_contacts_count: 0,
            chain_exp_numbering: BTreeMap::new(),
            numbering_resolution: String::new(),
        }
        .with_chain_rdc_media(default_chain, &single);
        return Ok(bundle.rdc_media);
    }

    // Multi-chain headered RDC TSV: `medium_id  chain_id  seq_id_1  res_name_1  atom_1  seq_id_2  res_name_2  atom_2  d_obs_hz  [sigma_hz]  [is_free]`
    let mut med_order: Vec<String> = Vec::new();
    let mut med_map: BTreeMap<String, Vec<CofoldRdcRestraint>> = BTreeMap::new();
    let mut hdr_opt: Option<Vec<String>> = None;

    for (line_idx, raw_line) in text.lines().enumerate() {
        let line_no = line_idx + 1;
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = if line.contains('\t') {
            line.split('\t').map(str::trim).collect()
        } else {
            line.split_whitespace().collect()
        };
        if hdr_opt.is_none() {
            hdr_opt = Some(cols.iter().map(|s| s.to_ascii_lowercase()).collect());
            continue;
        }
        let hdr = hdr_opt
            .as_ref()
            .ok_or_else(|| NmrError::parse("missing RDC header"))?;
        let col_idx = |names: &[&str]| hdr.iter().position(|h| names.iter().any(|n| h == n));

        let med_id = col_idx(&["medium_id", "medium"])
            .and_then(|i| cols.get(i))
            .copied()
            .unwrap_or("medium_1")
            .to_string();
        let ch1 = col_idx(&["chain_1", "chain_id", "chain"])
            .and_then(|i| cols.get(i))
            .copied()
            .unwrap_or(default_chain)
            .to_string();
        let ch2 = col_idx(&["chain_2"])
            .and_then(|i| cols.get(i))
            .copied()
            .unwrap_or(ch1.as_str())
            .to_string();
        let s1_i = col_idx(&["seq_id_1", "seq_id", "resid"]).ok_or_else(|| {
            NmrError::parse(format!("RDC TSV line {line_no}: missing seq_id_1 column"))
        })?;
        let seq_id_1: i32 = cols
            .get(s1_i)
            .ok_or_else(|| NmrError::parse(format!("RDC TSV line {line_no}: truncated row")))?
            .parse()
            .map_err(|e| NmrError::parse(format!("RDC line {line_no}: invalid seq_id_1: {e}")))?;
        let seq_id_2: i32 = match col_idx(&["seq_id_2"]).and_then(|i| cols.get(i)) {
            Some(s) => s.parse().map_err(|e| {
                NmrError::parse(format!("RDC line {line_no}: invalid seq_id_2: {e}"))
            })?,
            None => seq_id_1,
        };
        let res_name_1 = col_idx(&["res_name_1", "res_name", "resname"])
            .and_then(|i| cols.get(i))
            .copied()
            .unwrap_or("ALA")
            .to_string();
        let res_name_2 = col_idx(&["res_name_2"])
            .and_then(|i| cols.get(i))
            .copied()
            .unwrap_or(res_name_1.as_str())
            .to_string();
        let atom_name_1 = col_idx(&["atom_name_1", "atom_1", "atom1"])
            .and_then(|i| cols.get(i))
            .copied()
            .unwrap_or("N")
            .to_string();
        let atom_name_2 = col_idx(&["atom_name_2", "atom_2", "atom2"])
            .and_then(|i| cols.get(i))
            .copied()
            .unwrap_or("H")
            .to_string();
        let bond_type = match col_idx(&["bond_type", "type"]).and_then(|i| cols.get(i)) {
            Some(bt) => RdcBondType::parse_label(bt).map_err(NmrError::parse)?,
            None => RdcBondType::infer_from_atoms(seq_id_1, &atom_name_1, seq_id_2, &atom_name_2)
                .map_err(NmrError::parse)?,
        };
        let d_i = col_idx(&["d_obs_hz", "d_hz", "rdc_hz", "d_obs"]).ok_or_else(|| {
            NmrError::parse(format!("RDC TSV line {line_no}: missing d_obs_hz column"))
        })?;
        let d_obs_hz: f64 = cols
            .get(d_i)
            .ok_or_else(|| NmrError::parse(format!("RDC line {line_no}: truncated row")))?
            .parse()
            .map_err(|e| NmrError::parse(format!("RDC line {line_no}: invalid d_obs_hz: {e}")))?;
        let sigma_hz: f64 = match col_idx(&["sigma_hz", "sigma", "err"]).and_then(|i| cols.get(i)) {
            Some(s) if !s.is_empty() && *s != "." => s.parse().map_err(|e| {
                NmrError::parse(format!("RDC line {line_no}: invalid sigma_hz: {e}"))
            })?,
            _ => bond_type.default_sigma_hz(),
        };
        let is_free = col_idx(&["is_free", "free"])
            .and_then(|i| cols.get(i))
            .is_some_and(|s| {
                matches!(
                    s.to_ascii_lowercase().as_str(),
                    "1" | "true" | "free" | "yes"
                )
            });

        if !d_obs_hz.is_finite() || !sigma_hz.is_finite() || sigma_hz <= 0.0 {
            return Err(NmrError::parse(format!(
                "RDC line {line_no}: non-finite d_obs_hz or invalid sigma_hz"
            )));
        }

        if !med_map.contains_key(&med_id) {
            med_order.push(med_id.clone());
        }
        med_map
            .entry(med_id.clone())
            .or_default()
            .push(CofoldRdcRestraint {
                medium_id: med_id,
                chain_1: ch1,
                seq_id_1,
                res_name_1,
                atom_name_1,
                chain_2: ch2,
                seq_id_2,
                res_name_2,
                atom_name_2,
                bond_type,
                d_obs_hz,
                sigma_hz,
                d_max_hz: bond_type.signed_d_max_hz(RdcSignConvention::SignNormalized),
                is_free,
            });
    }

    if med_order.is_empty() {
        return Err(NmrError::empty_input("RDC TSV contains zero restraints"));
    }
    Ok(med_order
        .into_iter()
        .map(|mid| CofoldRdcMediumSet {
            restraints: med_map.remove(&mid).unwrap_or_default(),
            medium_id: mid,
        })
        .collect())
}

/// Parse chain-aware `PCS` TSV text for complex pose evaluation.
///
/// Header directives:
/// - `# metal_pos: <x> <y> <z>`
/// - `# metal_anchor: <chain_id> <seq_id> <atom_name>` (or `<seq_id> <atom_name>`)
/// - `# host_chain: <chain_id>`
pub fn parse_cofold_pcs_tsv_str(
    text: &str,
    default_chain: &str,
) -> Result<Vec<CofoldPcsTagSet>, NmrError> {
    check_bounded_text(text, "PCS TSV")?;
    let mut metal_pos: Option<Vec3> = None;
    let mut metal_anchor: Option<(String, i32, String)> = None;
    let mut host_chain_id: Option<String> = None;
    let mut obs = Vec::new();

    for (line_idx, raw_line) in text.lines().enumerate() {
        let line_no = line_idx + 1;
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(comment) = line.strip_prefix('#') {
            let c = comment.trim();
            if let Some(rest) = c.strip_prefix("metal_pos:") {
                let nums: Vec<&str> = rest
                    .split([',', ' ', '\t'])
                    .filter(|s| !s.is_empty())
                    .collect();
                if nums.len() != 3 {
                    return Err(NmrError::parse(format!(
                        "PCS line {line_no}: # metal_pos expects 3 floats"
                    )));
                }
                let x: f64 = nums[0]
                    .parse()
                    .map_err(|e| NmrError::parse(format!("{e}")))?;
                let y: f64 = nums[1]
                    .parse()
                    .map_err(|e| NmrError::parse(format!("{e}")))?;
                let z: f64 = nums[2]
                    .parse()
                    .map_err(|e| NmrError::parse(format!("{e}")))?;
                if !x.is_finite() || !y.is_finite() || !z.is_finite() {
                    return Err(NmrError::parse("non-finite # metal_pos"));
                }
                metal_pos = Some([x, y, z]);
            } else if let Some(rest) = c.strip_prefix("metal_anchor:") {
                let parts: Vec<&str> = rest.split_whitespace().collect();
                if parts.len() >= 3 {
                    let sid: i32 = parts[1]
                        .parse()
                        .map_err(|e| NmrError::parse(format!("PCS line {line_no}: {e}")))?;
                    metal_anchor = Some((parts[0].to_string(), sid, parts[2].to_ascii_uppercase()));
                } else if parts.len() == 2 {
                    let sid: i32 = parts[0]
                        .parse()
                        .map_err(|e| NmrError::parse(format!("PCS line {line_no}: {e}")))?;
                    metal_anchor = Some((
                        default_chain.to_string(),
                        sid,
                        parts[1].to_ascii_uppercase(),
                    ));
                }
            } else if let Some(rest) = c.strip_prefix("host_chain:") {
                let hc = rest.trim();
                if !hc.is_empty() {
                    host_chain_id = Some(hc.to_string());
                }
            }
            continue;
        }

        let cols: Vec<&str> = if line.contains('\t') {
            line.split('\t').map(str::trim).collect()
        } else {
            line.split_whitespace().collect()
        };
        if cols.is_empty() {
            continue;
        }
        let low0 = cols[0].to_ascii_lowercase();
        if matches!(
            low0.as_str(),
            "chain" | "chain_id" | "seq_id" | "resid" | "tag_id"
        ) {
            continue;
        }

        let (chain_id, seq_id, atom_name, pcs_ppm, sigma_ppm, is_free) =
            if let Ok(sid) = cols[0].parse::<i32>() {
                if cols.len() < 2 {
                    return Err(NmrError::parse(format!(
                        "PCS line {line_no}: truncated row"
                    )));
                }
                if let Ok(pcs) = cols[1].parse::<f64>() {
                    let sig = cols
                        .get(2)
                        .and_then(|s| s.parse::<f64>().ok())
                        .unwrap_or(0.05);
                    let fr = cols.get(3).is_some_and(|s| {
                        matches!(s.to_ascii_lowercase().as_str(), "1" | "true" | "free")
                    });
                    (
                        default_chain.to_string(),
                        sid,
                        "H".to_string(),
                        pcs,
                        sig,
                        fr,
                    )
                } else {
                    if cols.len() < 3 {
                        return Err(NmrError::parse(format!(
                            "PCS line {line_no}: truncated row"
                        )));
                    }
                    let pcs: f64 = cols[2]
                        .parse()
                        .map_err(|e| NmrError::parse(format!("PCS line {line_no}: {e}")))?;
                    let sig = cols
                        .get(3)
                        .and_then(|s| s.parse::<f64>().ok())
                        .unwrap_or(0.05);
                    let fr = cols.get(4).is_some_and(|s| {
                        matches!(s.to_ascii_lowercase().as_str(), "1" | "true" | "free")
                    });
                    (
                        default_chain.to_string(),
                        sid,
                        normalize_rdc_atom_name(cols[1]).to_string(),
                        pcs,
                        sig,
                        fr,
                    )
                }
            } else {
                if cols.len() < 4 {
                    return Err(NmrError::parse(format!(
                        "PCS line {line_no}: expected chain_id seq_id atom_name pcs_ppm"
                    )));
                }
                let sid: i32 = cols[1]
                    .parse()
                    .map_err(|e| NmrError::parse(format!("PCS line {line_no}: {e}")))?;
                let pcs: f64 = cols[3]
                    .parse()
                    .map_err(|e| NmrError::parse(format!("PCS line {line_no}: {e}")))?;
                let sig = cols
                    .get(4)
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.05);
                let fr = cols.get(5).is_some_and(|s| {
                    matches!(s.to_ascii_lowercase().as_str(), "1" | "true" | "free")
                });
                (
                    cols[0].to_string(),
                    sid,
                    normalize_rdc_atom_name(cols[2]).to_string(),
                    pcs,
                    sig,
                    fr,
                )
            };

        if !pcs_ppm.is_finite() || !sigma_ppm.is_finite() || sigma_ppm <= 0.0 {
            return Err(NmrError::parse(format!(
                "PCS line {line_no}: non-finite pcs_ppm or invalid sigma_ppm"
            )));
        }
        obs.push(CofoldPcsObservation {
            chain_id,
            seq_id,
            atom_name,
            pcs_obs_ppm: pcs_ppm,
            sigma_ppm,
            is_free,
        });
    }

    if obs.is_empty() {
        return Err(NmrError::empty_input("PCS TSV contains zero observations"));
    }
    Ok(vec![CofoldPcsTagSet {
        tag_id: "pcs_tag_1".to_string(),
        metal_pos,
        metal_anchor,
        host_chain_id,
        pre_fitted_tensor: None,
        observations: obs,
    }])
}

// ============================================================================
// 5. Schema-Honest Exporters (`export_af3_json` & `export_boltz2_yaml`)
// ============================================================================

/// Options for deterministic public `AlphaFold 3` JSON (`dialect: "alphafold3"`, `version: 2`) export.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Af3ExportConfig {
    /// Non-empty list of integer random seeds (`modelSeeds`, default `vec![1]`).
    pub model_seeds: Vec<u32>,
}

impl Default for Af3ExportConfig {
    fn default() -> Self {
        Self {
            model_seeds: vec![1],
        }
    }
}

/// Export deterministic, schema-conformant public `AlphaFold 3` JSON (`dialect == "alphafold3"`,
/// `version == 2`, Abramson et al. *Nature* 2024).
///
/// Strictly adheres to the public AF3 v2 input specification (`name`, `modelSeeds`, `sequences`,
/// `bondedAtomPairs`, `dialect`, `version`) with zero invented/non-existent pocket keys; pair this
/// output with [`export_nmr_restraints_sidecar_json`] (`<prefix>.nmr_restraints.json`) for
/// downstream `ferric cofold score` pose selection.
pub fn export_af3_json(
    restraints: &CofoldRestraintBundle,
    cfg: &Af3ExportConfig,
) -> Result<String, NmrError> {
    if restraints.entities.is_empty() {
        return Err(NmrError::empty_input(
            "cannot export AlphaFold 3 JSON with zero sequences/entities",
        ));
    }
    if cfg.model_seeds.is_empty() {
        return Err(NmrError::invalid_input(
            "AlphaFold 3 JSON requires at least one integer seed in modelSeeds",
        ));
    }
    for ent in &restraints.entities {
        ent.validate()?;
        validate_af3_chain_id(ent.chain_id())?;
    }

    let mut seq_entries: Vec<Value> = Vec::with_capacity(restraints.entities.len());
    for ent in &restraints.entities {
        match ent {
            CofoldEntitySpec::Polymer {
                kind,
                chain_id,
                sequence,
            } => {
                let mut inner = Map::new();
                inner.insert("id".to_string(), Value::String(chain_id.clone()));
                inner.insert(
                    "sequence".to_string(),
                    Value::String(sequence.trim().to_ascii_uppercase()),
                );
                let mut outer = Map::new();
                outer.insert(kind.as_key().to_string(), Value::Object(inner));
                seq_entries.push(Value::Object(outer));
            }
            CofoldEntitySpec::Ligand { chain_id, spec } => {
                let mut inner = Map::new();
                inner.insert("id".to_string(), Value::String(chain_id.clone()));
                match spec {
                    CofoldLigandSpec::Smiles(smi) => {
                        let mol = parse_smiles(smi.trim())?;
                        let can = write_canonical_smiles(&mol);
                        inner.insert("smiles".to_string(), Value::String(can));
                    }
                    CofoldLigandSpec::CcdCodes(codes) => {
                        let arr: Vec<Value> = codes
                            .iter()
                            .map(|c| Value::String(c.trim().to_ascii_uppercase()))
                            .collect();
                        inner.insert("ccdCodes".to_string(), Value::Array(arr));
                    }
                }
                let mut outer = Map::new();
                outer.insert("ligand".to_string(), Value::Object(inner));
                seq_entries.push(Value::Object(outer));
            }
        }
    }

    let mut bonded_pairs: Vec<Value> = Vec::with_capacity(restraints.bonded_atom_pairs.len());
    for bp in &restraints.bonded_atom_pairs {
        validate_bonded_atom_pair(bp, &restraints.entities)?;
        bonded_pairs.push(json!([
            [
                bp.atom1.chain_id,
                bp.atom1.one_based_res_idx,
                bp.atom1.atom_name
            ],
            [
                bp.atom2.chain_id,
                bp.atom2.one_based_res_idx,
                bp.atom2.atom_name
            ]
        ]));
    }

    let name = if restraints.job_name.trim().is_empty() {
        "ferric_cofold_complex"
    } else {
        restraints.job_name.trim()
    };

    let mut root = Map::new();
    root.insert("name".to_string(), Value::String(name.to_string()));
    root.insert(
        "modelSeeds".to_string(),
        Value::Array(cfg.model_seeds.iter().map(|&s| json!(s)).collect()),
    );
    root.insert("sequences".to_string(), Value::Array(seq_entries));
    root.insert("bondedAtomPairs".to_string(), Value::Array(bonded_pairs));
    root.insert(
        "dialect".to_string(),
        Value::String("alphafold3".to_string()),
    );
    root.insert("version".to_string(), json!(2));

    let mut json_str = serde_json::to_string_pretty(&Value::Object(root))
        .map_err(|e| NmrError::invalid_input(format!("failed to serialize AF3 JSON: {e}")))?;
    json_str.push('\n');
    Ok(json_str)
}

/// Serialize a [`CofoldRestraintBundle`] sidecar (`<prefix>.nmr_restraints.json`) for downstream
/// `ferric cofold score` pose selection.
pub fn export_nmr_restraints_sidecar_json(
    restraints: &CofoldRestraintBundle,
) -> Result<String, NmrError> {
    let mut s = serde_json::to_string_pretty(restraints)
        .map_err(|e| NmrError::invalid_input(format!("failed to serialize sidecar JSON: {e}")))?;
    s.push('\n');
    Ok(s)
}

/// Options overriding `Boltz-2` YAML (`version: 1`) export behavior.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Boltz2ExportConfig {
    /// Override `pocket` `max_distance` (`None` = use `restraints.pocket_max_distance_a`, default `6.0 A`).
    pub pocket_max_distance_override: Option<f64>,
    /// Override `force` boolean on `pocket` & `contact` constraints (`None` = use `restraints.force_constraints`).
    pub force_override: Option<bool>,
    /// Override `properties: [{affinity: {binder}}]` emission (`None` = use `restraints.request_boltz2_affinity`).
    pub include_affinity_override: Option<bool>,
}

fn yaml_single_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "''"))
}

fn yaml_safe_scalar(s: &str) -> String {
    let trimmed = s.trim();
    let starts_digit = trimmed.chars().next().is_some_and(|c| c.is_ascii_digit());
    if !trimmed.is_empty()
        && !starts_digit
        && trimmed.parse::<f64>().is_err()
        && trimmed
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !matches!(
            trimmed.to_ascii_lowercase().as_str(),
            "true" | "false" | "null" | "yes" | "no" | "on" | "off" | "y" | "n"
        )
    {
        trimmed.to_string()
    } else {
        yaml_single_quote(trimmed)
    }
}

fn format_boltz_token_yaml(tok: &BoltzContactToken) -> String {
    match tok {
        BoltzContactToken::PolymerResidue {
            chain_id,
            one_based_seq_idx,
        } => format!("[{}, {}]", yaml_safe_scalar(chain_id), one_based_seq_idx),
        BoltzContactToken::LigandAtom {
            chain_id,
            atom_name,
        } => format!(
            "[{}, {}]",
            yaml_safe_scalar(chain_id),
            yaml_safe_scalar(atom_name)
        ),
    }
}

/// Export deterministic zero-dependency `Boltz-2` YAML (`version: 1`, Wohlwend et al. 2024;
/// Passaro et al. 2025) carrying `sequences`, surface-filtered `CSP` `pocket` contacts,
/// intermolecular `NOE`/`PRE` `contact` distance bounds, and optional `affinity` property.
pub fn export_boltz2_yaml(
    restraints: &CofoldRestraintBundle,
    cfg: &Boltz2ExportConfig,
) -> Result<String, NmrError> {
    if restraints.entities.is_empty() {
        return Err(NmrError::empty_input(
            "cannot export Boltz-2 YAML with zero sequences/entities",
        ));
    }
    for ent in &restraints.entities {
        ent.validate()?;
    }

    let pocket_max_d = cfg
        .pocket_max_distance_override
        .unwrap_or(restraints.pocket_max_distance_a);
    if !pocket_max_d.is_finite()
        || !(BOLTZ2_MIN_DISTANCE_A..=BOLTZ2_MAX_DISTANCE_A).contains(&pocket_max_d)
    {
        return Err(NmrError::invalid_input(format!(
            "Boltz-2 pocket max_distance ({pocket_max_d}) must lie in [{BOLTZ2_MIN_DISTANCE_A}, {BOLTZ2_MAX_DISTANCE_A}] A"
        )));
    }
    let force_flag = cfg.force_override.unwrap_or(restraints.force_constraints);
    let include_affinity = cfg
        .include_affinity_override
        .unwrap_or(restraints.request_boltz2_affinity);

    let mut y = String::with_capacity(2048);
    y.push_str("version: 1\n");
    y.push_str("sequences:\n");

    for ent in &restraints.entities {
        match ent {
            CofoldEntitySpec::Polymer {
                kind,
                chain_id,
                sequence,
            } => {
                y.push_str(&format!("  - {}:\n", kind.as_key()));
                y.push_str(&format!("      id: {}\n", yaml_safe_scalar(chain_id)));
                y.push_str(&format!(
                    "      sequence: {}\n",
                    sequence.trim().to_ascii_uppercase()
                ));
            }
            CofoldEntitySpec::Ligand { chain_id, spec } => {
                y.push_str("  - ligand:\n");
                y.push_str(&format!("      id: {}\n", yaml_safe_scalar(chain_id)));
                match spec {
                    CofoldLigandSpec::Smiles(smi) => {
                        let mol = parse_smiles(smi.trim())?;
                        let can = write_canonical_smiles(&mol);
                        y.push_str(&format!("      smiles: {}\n", yaml_single_quote(&can)));
                    }
                    CofoldLigandSpec::CcdCodes(codes) => {
                        if codes.len() != 1 {
                            return Err(NmrError::invalid_input(format!(
                                "Boltz-2 YAML ligand 'ccd' on chain '{chain_id}' requires a single CCD code string, got {} codes (use separate ligand entities or SMILES)",
                                codes.len()
                            )));
                        }
                        y.push_str(&format!(
                            "      ccd: {}\n",
                            yaml_safe_scalar(&codes[0].to_ascii_uppercase())
                        ));
                    }
                }
            }
        }
    }

    // Collect deterministic `pocket` and `contact` constraints
    let mut sorted_pocket = restraints.pocket_contacts.clone();
    sorted_pocket.sort_by(|a, b| {
        (&a.chain_id, a.one_based_seq_idx).cmp(&(&b.chain_id, b.one_based_seq_idx))
    });
    sorted_pocket
        .dedup_by(|a, b| a.chain_id == b.chain_id && a.one_based_seq_idx == b.one_based_seq_idx);

    let mut contact_items: Vec<(BoltzContactToken, BoltzContactToken, f64)> = restraints
        .distance_restraints
        .iter()
        .filter_map(|r| {
            match (
                &r.boltz_token_1,
                &r.boltz_token_2,
                r.exported_contact_max_distance_a,
            ) {
                (Some(t1), Some(t2), Some(d_max)) => Some((t1.clone(), t2.clone(), d_max)),
                _ => None,
            }
        })
        .collect();
    // Deduplicate identical token pairs keeping the tightest upper distance bound
    let mut dedup_contacts: BTreeMap<(BoltzContactToken, BoltzContactToken), f64> = BTreeMap::new();
    for (t1, t2, d_max) in contact_items.drain(..) {
        let pair = if t1 <= t2 { (t1, t2) } else { (t2, t1) };
        dedup_contacts
            .entry(pair)
            .and_modify(|cur| {
                if d_max < *cur {
                    *cur = d_max;
                }
            })
            .or_insert(d_max);
    }

    if !sorted_pocket.is_empty() || !dedup_contacts.is_empty() {
        y.push_str("constraints:\n");
        if !sorted_pocket.is_empty() {
            for p in &sorted_pocket {
                if p.one_based_seq_idx == 0 {
                    return Err(NmrError::invalid_input(
                        "Boltz-2 pocket contacts require 1-based residue indices (>= 1)",
                    ));
                }
            }
            y.push_str("  - pocket:\n");
            y.push_str(&format!(
                "      binder: {}\n",
                yaml_safe_scalar(&restraints.binder_chain_id)
            ));
            y.push_str("      contacts:\n");
            for p in &sorted_pocket {
                y.push_str(&format!(
                    "        - [{}, {}]\n",
                    yaml_safe_scalar(&p.chain_id),
                    p.one_based_seq_idx
                ));
            }
            y.push_str(&format!("      max_distance: {pocket_max_d:.1}\n"));
            y.push_str(&format!(
                "      force: {}\n",
                if force_flag { "true" } else { "false" }
            ));
        }

        for ((t1, t2), d_max) in &dedup_contacts {
            if !d_max.is_finite()
                || !(BOLTZ2_MIN_DISTANCE_A..=BOLTZ2_MAX_DISTANCE_A).contains(d_max)
            {
                return Err(NmrError::invalid_input(format!(
                    "Boltz-2 contact max_distance ({d_max}) must lie in [{BOLTZ2_MIN_DISTANCE_A}, {BOLTZ2_MAX_DISTANCE_A}] A"
                )));
            }
            for tok in [t1, t2] {
                if let BoltzContactToken::LigandAtom { chain_id, .. } = tok {
                    if is_smiles_ligand_chain(&restraints.entities, chain_id) {
                        return Err(NmrError::invalid_input(format!(
                            "Boltz-2 YAML cannot export atom-level contact on SMILES ligand chain '{chain_id}'"
                        )));
                    }
                }
            }
            y.push_str("  - contact:\n");
            y.push_str(&format!("      token1: {}\n", format_boltz_token_yaml(t1)));
            y.push_str(&format!("      token2: {}\n", format_boltz_token_yaml(t2)));
            y.push_str(&format!("      max_distance: {d_max:.2}\n"));
            y.push_str(&format!(
                "      force: {}\n",
                if force_flag { "true" } else { "false" }
            ));
        }
    }

    if include_affinity {
        let binder_is_ligand = restraints
            .entities
            .iter()
            .any(|e| !e.is_polymer() && e.chain_id() == restraints.binder_chain_id);
        if !binder_is_ligand {
            return Err(NmrError::invalid_input(format!(
                "Boltz-2 affinity property requires binder '{}' to be a small-molecule ligand entity",
                restraints.binder_chain_id
            )));
        }
        y.push_str("properties:\n");
        y.push_str("  - affinity:\n");
        y.push_str(&format!(
            "      binder: {}\n",
            yaml_safe_scalar(&restraints.binder_chain_id)
        ));
    }

    Ok(y)
}

// ============================================================================
// 6. Full-Atom Complex Pose Parser (`.pdb`/`.cif`) & `octet_mol` Ligand Rings
// ============================================================================

#[derive(Clone, Debug)]
struct ComplexAtomRecord {
    is_hetatm: bool,
    chain_id: String,
    seq_id: i32,
    ins_code: Option<String>,
    res_name: String,
    atom_name: String,
    z: u8,
    pos: Vec3,
}

#[derive(Clone, Debug)]
struct ParsedComplexPose {
    protein_model: StructureModel,
    all_atoms: Vec<ComplexAtomRecord>,
    non_protein_aromatic_rings: Vec<ConjugatedAromaticRing>,
}

fn is_water_res_name(res_name: &str) -> bool {
    matches!(
        res_name.trim().to_ascii_uppercase().as_str(),
        "HOH" | "WAT" | "DOD" | "SOL" | "TIP3" | "H2O"
    )
}

fn clean_element_symbol_z(raw_el: &str, raw_atom: &str) -> u8 {
    let cleaned: String = raw_el
        .trim()
        .chars()
        .take_while(|c| c.is_ascii_alphabetic())
        .collect();
    if !cleaned.is_empty() {
        if let Some(z) = atomic_number_from_symbol(&cleaned) {
            if z > 0 {
                return z;
            }
        }
    }
    if let Some(z) = infer_heavy_z_from_atom_name(raw_atom) {
        return z;
    }
    let stripped = raw_atom
        .trim()
        .trim_start_matches(|c: char| c.is_ascii_digit())
        .to_ascii_uppercase();
    if stripped.starts_with('H') || stripped.starts_with('D') {
        1
    } else {
        6
    }
}

fn tokenize_cif_minimal(text: &str) -> Result<Vec<(String, bool)>, NmrError> {
    let mut tokens = Vec::new();
    let mut lines = text.lines().peekable();
    while let Some(line) = lines.next() {
        if let Some(semi) = line.strip_prefix(';') {
            let mut block = String::from(semi);
            for next_line in lines.by_ref() {
                if next_line.starts_with(';') {
                    break;
                }
                block.push('\n');
                block.push_str(next_line);
            }
            tokens.push((block, true));
            continue;
        }
        let bytes = line.as_bytes();
        let n = bytes.len();
        let mut i = 0usize;
        while i < n {
            let b = bytes[i];
            if b.is_ascii_whitespace() {
                i += 1;
                continue;
            }
            if b == b'#' {
                break;
            }
            if b == b'\'' || b == b'"' {
                let q = b;
                i += 1;
                let start = i;
                while i < n {
                    if bytes[i] == q && (i + 1 == n || bytes[i + 1].is_ascii_whitespace()) {
                        break;
                    }
                    i += 1;
                }
                if i >= n {
                    return Err(NmrError::parse("unterminated quoted string in mmCIF"));
                }
                let val = std::str::from_utf8(&bytes[start..i])
                    .map_err(|e| NmrError::parse(format!("invalid UTF-8 in mmCIF token: {e}")))?;
                tokens.push((val.to_string(), true));
                i += 1;
            } else {
                let start = i;
                while i < n && !bytes[i].is_ascii_whitespace() {
                    i += 1;
                }
                let val = std::str::from_utf8(&bytes[start..i])
                    .map_err(|e| NmrError::parse(format!("invalid UTF-8 in mmCIF token: {e}")))?;
                tokens.push((val.to_string(), false));
            }
        }
    }
    Ok(tokens)
}

fn parse_all_atoms_from_pose_str(
    text: &str,
    format_hint: Option<&str>,
) -> Result<Vec<ComplexAtomRecord>, NmrError> {
    if text.len() > MAX_STRUCTURE_FILE_BYTES || text.as_bytes().contains(&0u8) {
        return Err(NmrError::parse(
            "pose coordinate input exceeds 64 MiB limit or contains NUL bytes",
        ));
    }
    let is_cif = match format_hint {
        Some("cif" | "mmcif") => true,
        Some("pdb" | "ent") => false,
        _ => {
            let head: String = text
                .lines()
                .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'))
                .take(20)
                .collect::<Vec<_>>()
                .join("\n");
            head.starts_with("data_") || head.contains("_atom_site.") || head.contains("loop_")
        }
    };

    if is_cif {
        let tokens = tokenize_cif_minimal(text)?;
        let mut atoms = Vec::new();
        let n_tok = tokens.len();
        let mut pos = 0usize;
        while pos < n_tok {
            if !tokens[pos].1 && tokens[pos].0.eq_ignore_ascii_case("loop_") {
                pos += 1;
                let mut cols = Vec::new();
                while pos < n_tok && !tokens[pos].1 && tokens[pos].0.starts_with('_') {
                    cols.push(tokens[pos].0.to_ascii_lowercase());
                    pos += 1;
                }
                if cols.is_empty() {
                    return Err(NmrError::parse("malformed mmCIF loop_ with zero headers"));
                }
                let mut vals: Vec<&str> = Vec::new();
                while pos < n_tok {
                    let (ref v, vq) = tokens[pos];
                    if !vq
                        && (v.eq_ignore_ascii_case("loop_")
                            || v.eq_ignore_ascii_case("stop_")
                            || v.starts_with("data_")
                            || v.starts_with("save_")
                            || v.starts_with('_'))
                    {
                        if v.eq_ignore_ascii_case("stop_") {
                            pos += 1;
                        }
                        break;
                    }
                    vals.push(v.as_str());
                    pos += 1;
                }
                let ncols = cols.len();
                if !vals.len().is_multiple_of(ncols) {
                    return Err(NmrError::parse("truncated mmCIF loop_ row"));
                }
                if cols[0].starts_with("_atom_site.") {
                    let find_c = |name: &str| cols.iter().position(|c| c == name);
                    let grp_c = find_c("_atom_site.group_pdb");
                    let x_c = find_c("_atom_site.cartn_x")
                        .ok_or_else(|| NmrError::parse("mmCIF _atom_site missing Cartn_x"))?;
                    let y_c = find_c("_atom_site.cartn_y")
                        .ok_or_else(|| NmrError::parse("mmCIF _atom_site missing Cartn_y"))?;
                    let z_c = find_c("_atom_site.cartn_z")
                        .ok_or_else(|| NmrError::parse("mmCIF _atom_site missing Cartn_z"))?;
                    let comp_c = find_c("_atom_site.auth_comp_id")
                        .or_else(|| find_c("_atom_site.label_comp_id"))
                        .ok_or_else(|| NmrError::parse("mmCIF _atom_site missing comp_id"))?;
                    let atom_c = find_c("_atom_site.label_atom_id")
                        .or_else(|| find_c("_atom_site.auth_atom_id"))
                        .ok_or_else(|| NmrError::parse("mmCIF _atom_site missing atom_id"))?;
                    let asym_c = find_c("_atom_site.auth_asym_id")
                        .or_else(|| find_c("_atom_site.label_asym_id"));
                    let auth_seq_c = find_c("_atom_site.auth_seq_id");
                    let label_seq_c = find_c("_atom_site.label_seq_id");
                    let el_c = find_c("_atom_site.type_symbol");
                    let alt_c = find_c("_atom_site.label_alt_id");
                    let ins_c = find_c("_atom_site.pdbx_pdb_ins_code");
                    let model_c = find_c("_atom_site.pdbx_pdb_model_num");

                    let mut first_model: Option<&str> = None;
                    for row in vals.chunks_exact(ncols) {
                        if let Some(mc) = model_c {
                            let m = row[mc];
                            match first_model {
                                None => first_model = Some(m),
                                Some(fm) if fm != m => continue,
                                _ => {}
                            }
                        }
                        if let Some(ac) = alt_c {
                            let alt = row[ac];
                            if !matches!(alt, "." | "?" | "" | "A" | "1") {
                                continue;
                            }
                        }
                        let ins_code = ins_c.and_then(|ic| {
                            let ins = row[ic].trim();
                            (!matches!(ins, "." | "?" | "")).then(|| ins.to_string())
                        });
                        let is_hetatm = grp_c
                            .map(|gc| row[gc].eq_ignore_ascii_case("HETATM"))
                            .unwrap_or(false);
                        let res_name = row[comp_c].trim().to_ascii_uppercase();
                        if is_water_res_name(&res_name) {
                            continue;
                        }
                        let chain_id = asym_c
                            .map(|c| row[c].trim())
                            .filter(|s| !s.is_empty() && *s != "." && *s != "?")
                            .unwrap_or("A")
                            .to_string();
                        // AF3 and Boltz-2 `.cif` outputs set `label_seq_id = .` on non-polymer HETATM ligands!
                        let raw_seq = auth_seq_c
                            .map(|c| row[c])
                            .filter(|s| *s != "." && *s != "?")
                            .or_else(|| {
                                label_seq_c
                                    .map(|c| row[c])
                                    .filter(|s| *s != "." && *s != "?")
                            });
                        let seq_id: i32 = match raw_seq {
                            Some(s) => s.parse().map_err(|e| {
                                NmrError::parse(format!("invalid mmCIF seq_id '{s}': {e}"))
                            })?,
                            None if is_hetatm => 1,
                            None => {
                                return Err(NmrError::parse(
                                    "mmCIF ATOM record missing auth_seq_id and label_seq_id",
                                ));
                            }
                        };
                        let x: f64 = row[x_c]
                            .parse()
                            .map_err(|e| NmrError::parse(format!("invalid Cartn_x: {e}")))?;
                        let y: f64 = row[y_c]
                            .parse()
                            .map_err(|e| NmrError::parse(format!("invalid Cartn_y: {e}")))?;
                        let z: f64 = row[z_c]
                            .parse()
                            .map_err(|e| NmrError::parse(format!("invalid Cartn_z: {e}")))?;
                        if !x.is_finite() || !y.is_finite() || !z.is_finite() {
                            return Err(NmrError::parse("non-finite coordinate in mmCIF"));
                        }
                        let atom_name = row[atom_c].trim().to_ascii_uppercase();
                        let raw_el = el_c.map(|c| row[c]).unwrap_or("");
                        let z_num = clean_element_symbol_z(raw_el, &atom_name);
                        atoms.push(ComplexAtomRecord {
                            is_hetatm,
                            chain_id,
                            seq_id,
                            ins_code,
                            res_name,
                            atom_name,
                            z: z_num,
                            pos: [x, y, z],
                        });
                    }
                }
            } else {
                pos += 1;
            }
        }
        if atoms.is_empty() {
            return Err(NmrError::empty_input(
                "mmCIF pose contains zero non-water atoms",
            ));
        }
        Ok(atoms)
    } else {
        let mut atoms = Vec::new();
        for (line_idx, line) in text.lines().enumerate() {
            let line_no = line_idx + 1;
            if line.starts_with("ENDMDL") {
                break;
            }
            let is_atom = line.starts_with("ATOM  ");
            let is_het = line.starts_with("HETATM");
            if !is_atom && !is_het {
                continue;
            }
            if !line.is_ascii() || line.len() < 54 {
                return Err(NmrError::parse(format!(
                    "PDB line {line_no}: truncated or non-ASCII ATOM/HETATM record"
                )));
            }
            let alt = line.as_bytes()[16];
            if alt != b' ' && alt != b'A' && alt != b'1' {
                continue;
            }
            let i_code = line.as_bytes()[26];
            let ins_code = (i_code != b' ').then(|| (i_code as char).to_string());
            let res_name = line[17..20].trim().to_ascii_uppercase();
            if is_water_res_name(&res_name) {
                continue;
            }
            let atom_name = line[12..16].trim().to_ascii_uppercase();
            let chain_raw = line[21..22].trim();
            let chain_id = if chain_raw.is_empty() {
                "A".to_string()
            } else {
                chain_raw.to_string()
            };
            let seq_raw = line[22..26].trim();
            let seq_id: i32 = seq_raw.parse().map_err(|e| {
                NmrError::parse(format!(
                    "PDB line {line_no}: invalid residue seq_id '{seq_raw}': {e}"
                ))
            })?;
            let x: f64 = line[30..38]
                .trim()
                .parse()
                .map_err(|e| NmrError::parse(format!("PDB line {line_no}: invalid X: {e}")))?;
            let y: f64 = line[38..46]
                .trim()
                .parse()
                .map_err(|e| NmrError::parse(format!("PDB line {line_no}: invalid Y: {e}")))?;
            let z: f64 = line[46..54]
                .trim()
                .parse()
                .map_err(|e| NmrError::parse(format!("PDB line {line_no}: invalid Z: {e}")))?;
            if !x.is_finite() || !y.is_finite() || !z.is_finite() {
                return Err(NmrError::parse(format!(
                    "PDB line {line_no}: non-finite atomic coordinate"
                )));
            }
            let raw_el = if line.len() >= 78 { &line[76..78] } else { "" };
            let z_num = clean_element_symbol_z(raw_el, &atom_name);
            atoms.push(ComplexAtomRecord {
                is_hetatm: is_het,
                chain_id,
                seq_id,
                ins_code,
                res_name,
                atom_name,
                z: z_num,
                pos: [x, y, z],
            });
        }
        if atoms.is_empty() {
            return Err(NmrError::empty_input(
                "PDB pose contains zero non-water atoms",
            ));
        }
        Ok(atoms)
    }
}

/// Match heavy atoms of a topological `octet_mol::Molecule` (`topo_mol`, e.g. from SMILES/MOL/.oct)
/// to 3D `HETATM` heavy atoms `(z, pos)` via degree + element pruned backtracking isomorphism.
fn match_topological_ligand_to_3d(
    topo_mol: &Molecule,
    atoms_3d: &[&ComplexAtomRecord],
) -> Option<Vec<usize>> {
    let topo_heavy: Vec<usize> = (0..topo_mol.atoms.len())
        .filter(|&i| topo_mol.atoms[i].z > 1)
        .collect();
    let n = topo_heavy.len();
    if n == 0 || n != atoms_3d.len() || n > 128 {
        return None;
    }

    let mut topo_pos_in_heavy = vec![usize::MAX; topo_mol.atoms.len()];
    for (hi, &orig) in topo_heavy.iter().enumerate() {
        topo_pos_in_heavy[orig] = hi;
    }
    let mut adj_topo = vec![vec![false; n]; n];
    let mut deg_topo = vec![0usize; n];
    for b in &topo_mol.bonds {
        if matches!(
            b.kind,
            BondKind::Dative | BondKind::Hydrogen | BondKind::Ionic
        ) {
            continue;
        }
        let u = topo_pos_in_heavy[b.a];
        let v = topo_pos_in_heavy[b.b];
        if u < n && v < n && !adj_topo[u][v] {
            adj_topo[u][v] = true;
            adj_topo[v][u] = true;
            deg_topo[u] += 1;
            deg_topo[v] += 1;
        }
    }

    let mut adj_3d = vec![vec![false; n]; n];
    let mut deg_3d = vec![0usize; n];
    for i in 0..n {
        let ri = covalent_radius(atoms_3d[i].z);
        for j in (i + 1)..n {
            let r_max = ri + covalent_radius(atoms_3d[j].z) + 0.45;
            let d = v_dist(atoms_3d[i].pos, atoms_3d[j].pos);
            if d >= 0.40 && d <= r_max {
                adj_3d[i][j] = true;
                adj_3d[j][i] = true;
                deg_3d[i] += 1;
                deg_3d[j] += 1;
            }
        }
    }

    // Order topological vertices by rarest `(z, deg)` first to prune search immediately
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&u| {
        let z = topo_mol.atoms[topo_heavy[u]].z;
        let deg = deg_topo[u];
        let cand_cnt = (0..n)
            .filter(|&w| atoms_3d[w].z == z && deg_3d[w] == deg)
            .count();
        (cand_cnt, std::cmp::Reverse(deg))
    });

    let mut mapping = vec![usize::MAX; n]; // topo_heavy_idx -> 3d_idx
    let mut used_3d = vec![false; n];
    let mut steps = 0usize;

    struct IsoSearchCtx<'a> {
        order: &'a [usize],
        topo_mol: &'a Molecule,
        topo_heavy: &'a [usize],
        atoms_3d: &'a [&'a ComplexAtomRecord],
        adj_topo: &'a [Vec<bool>],
        deg_topo: &'a [usize],
        adj_3d: &'a [Vec<bool>],
        deg_3d: &'a [usize],
    }

    fn dfs(
        step_idx: usize,
        ctx: &IsoSearchCtx<'_>,
        mapping: &mut [usize],
        used_3d: &mut [bool],
        steps: &mut usize,
    ) -> bool {
        if step_idx == ctx.order.len() {
            return true;
        }
        *steps += 1;
        if *steps > 50_000 {
            return false;
        }
        let u = ctx.order[step_idx];
        let z_u = ctx.topo_mol.atoms[ctx.topo_heavy[u]].z;
        let d_u = ctx.deg_topo[u];

        for w in 0..ctx.atoms_3d.len() {
            if used_3d[w] || ctx.atoms_3d[w].z != z_u || ctx.deg_3d[w] != d_u {
                continue;
            }
            let mut consistent = true;
            for &prev_u in &ctx.order[..step_idx] {
                let prev_w = mapping[prev_u];
                if ctx.adj_topo[u][prev_u] != ctx.adj_3d[w][prev_w] {
                    consistent = false;
                    break;
                }
            }
            if !consistent {
                continue;
            }
            mapping[u] = w;
            used_3d[w] = true;
            if dfs(step_idx + 1, ctx, mapping, used_3d, steps) {
                return true;
            }
            mapping[u] = usize::MAX;
            used_3d[w] = false;
        }
        false
    }

    let ctx = IsoSearchCtx {
        order: &order,
        topo_mol,
        topo_heavy: &topo_heavy,
        atoms_3d,
        adj_topo: &adj_topo,
        deg_topo: &deg_topo,
        adj_3d: &adj_3d,
        deg_3d: &deg_3d,
    };

    if dfs(0, &ctx, &mut mapping, &mut used_3d, &mut steps) {
        let mut full_map = vec![usize::MAX; topo_mol.atoms.len()];
        for (hi, &orig) in topo_heavy.iter().enumerate() {
            full_map[orig] = mapping[hi];
        }
        Some(full_map)
    } else {
        None
    }
}

fn ring_geometry_and_planarity(vertices: &[Vec3]) -> Option<(Vec3, Vec3, f64)> {
    let m = vertices.len();
    if m < 5 {
        return None;
    }
    let inv_m = 1.0 / (m as f64);
    let mut center = [0.0; 3];
    for &v in vertices {
        center = v_add(center, v);
    }
    center = v_scale(center, inv_m);

    let mut normal_acc = [0.0; 3];
    for k in 0..m {
        let a = v_sub(vertices[k], center);
        let b = v_sub(vertices[(k + 1) % m], center);
        normal_acc = v_add(normal_acc, v_cross(a, b));
    }
    let n_len = v_norm(normal_acc);
    if n_len <= 1e-6 {
        return None;
    }
    let normal = v_scale(normal_acc, 1.0 / n_len);
    let rms_oop = (vertices
        .iter()
        .map(|&v| v_dot(v_sub(v, center), normal).powi(2))
        .sum::<f64>()
        * inv_m)
        .sqrt();
    Some((center, normal, rms_oop))
}

fn ring_intensity_factor_for_elements(z_cycle: &[u8]) -> f64 {
    let n_count = z_cycle.iter().filter(|&&z| z == 7).count();
    if z_cycle.len() == 6 {
        if z_cycle.iter().all(|&z| z == 6) {
            1.00
        } else {
            0.85
        }
    } else if z_cycle.len() == 5 {
        // Case (1995) / Haigh-Mallion 5-ring intensity factors: pyrrole/indole 5-ring (1 N) ~ 0.90,
        // imidazole / His-like 5-ring (>= 2 N) ~ 0.43, other 5-rings ~ 0.56.
        if n_count == 1 {
            0.90
        } else if n_count >= 2 {
            0.43
        } else {
            0.56
        }
    } else {
        0.75
    }
}

/// Perceive planar Huckel-aromatic rings on non-protein residues (small-molecule ligands & RNA/DNA
/// nucleobases) using `nmr-core::octet_mol::{perceive_sssr, perceive_aromaticity}`.
fn perceive_non_protein_aromatic_rings(
    all_atoms: &[ComplexAtomRecord],
    protein_residues: &BTreeSet<(String, i32)>,
    ligand_molecules: &BTreeMap<String, Molecule>,
) -> Vec<ConjugatedAromaticRing> {
    let mut groups: BTreeMap<(String, i32, String), Vec<&ComplexAtomRecord>> = BTreeMap::new();
    for a in all_atoms {
        if protein_residues.contains(&(a.chain_id.clone(), a.seq_id)) && !a.is_hetatm {
            continue;
        }
        groups
            .entry((a.chain_id.clone(), a.seq_id, a.res_name.clone()))
            .or_default()
            .push(a);
    }

    let mut out_rings = Vec::new();
    for ((chain_id, seq_id, res_name), grp_atoms) in groups {
        let heavy_atoms: Vec<&ComplexAtomRecord> =
            grp_atoms.iter().copied().filter(|a| a.z > 1).collect();
        if heavy_atoms.len() < 5 || heavy_atoms.len() > 256 {
            continue;
        }

        // Strategy A: Exact topological SMILES/MOL/.oct Huckel perception mapped onto 3D HETATM via isomorphism
        if let Some(topo_mol) = ligand_molecules.get(&chain_id) {
            if let Some(topo_to_3d) = match_topological_ligand_to_3d(topo_mol, &heavy_atoms) {
                let arom = perceive_aromaticity(topo_mol);
                let sssr = perceive_sssr(topo_mol);
                for (r_idx, cyc) in sssr.iter().enumerate() {
                    if !(5..=6).contains(&cyc.len()) || !cyc.iter().all(|&u| arom.aromatic_atoms[u])
                    {
                        continue;
                    }
                    let mut verts = Vec::with_capacity(cyc.len());
                    let mut z_cyc = Vec::with_capacity(cyc.len());
                    let mut all_mapped = true;
                    for &u in cyc {
                        let w = topo_to_3d[u];
                        if w >= heavy_atoms.len() {
                            all_mapped = false;
                            break;
                        }
                        verts.push(heavy_atoms[w].pos);
                        z_cyc.push(heavy_atoms[w].z);
                    }
                    if !all_mapped {
                        continue;
                    }
                    if let Some((center, normal, rms_oop)) = ring_geometry_and_planarity(&verts) {
                        if rms_oop <= 0.20 {
                            out_rings.push(ConjugatedAromaticRing {
                                chain_id: chain_id.clone(),
                                seq_id,
                                residue_index: usize::MAX,
                                res_name: res_name.clone(),
                                ring_label: format!("LIG_RING_{}", r_idx + 1),
                                vertices: verts,
                                center,
                                normal,
                                intensity_factor: ring_intensity_factor_for_elements(&z_cyc),
                            });
                        }
                    }
                }
                continue;
            }
        }

        // Strategy B: 3D covalent graph + planarity + `octet_mol::perceive_aromaticity` Huckel 4n+2 verification
        // (covers CCD ligands, standalone HETATM poses, and RNA/DNA purine/pyrimidine nucleobases).
        let n_h = heavy_atoms.len();
        let mut mol3d = Molecule::default();
        for ha in &heavy_atoms {
            let sym = crate::octet_mol::symbol_from_atomic_number(ha.z).to_string();
            let mut atom = Atom::new(&sym);
            atom.z = ha.z;
            atom.x = ha.pos[0];
            atom.y = ha.pos[1];
            atom.z_coord = Some(ha.pos[2]);
            mol3d.atoms.push(atom);
        }
        for i in 0..n_h {
            let ri = covalent_radius(heavy_atoms[i].z);
            for j in (i + 1)..n_h {
                let d = v_dist(heavy_atoms[i].pos, heavy_atoms[j].pos);
                if d >= 0.40 && d <= ri + covalent_radius(heavy_atoms[j].z) + 0.42 {
                    mol3d.bonds.push(Bond::new(i, j, BondKind::Single));
                }
            }
        }

        let sssr = perceive_sssr(&mol3d);
        if sssr.is_empty() {
            continue;
        }

        // Identify planar conjugated 5/6-membered candidate cycles
        let mut planar_cycles: Vec<(Vec<usize>, Vec<Vec3>, Vec3, Vec3)> = Vec::new();
        let mut in_planar_ring = vec![false; n_h];
        for cyc in &sssr {
            if !(5..=6).contains(&cyc.len()) {
                continue;
            }
            if !cyc.iter().all(|&u| {
                matches!(heavy_atoms[u].z, 5 | 6 | 7 | 8 | 15 | 16 | 34)
                    && mol3d.covalent_neighbors(u).len() <= 3
            }) {
                continue;
            }
            let verts: Vec<Vec3> = cyc.iter().map(|&u| heavy_atoms[u].pos).collect();
            let Some((center, normal, rms_oop)) = ring_geometry_and_planarity(&verts) else {
                continue;
            };
            if rms_oop > 0.12 {
                continue;
            }
            // Reject saturated sp3 rings (`C-C > 1.48 A`)
            let mut max_cc_bond = 0.0_f64;
            for k in 0..cyc.len() {
                let u = cyc[k];
                let v = cyc[(k + 1) % cyc.len()];
                if heavy_atoms[u].z == 6 && heavy_atoms[v].z == 6 {
                    max_cc_bond = max_cc_bond.max(v_dist(heavy_atoms[u].pos, heavy_atoms[v].pos));
                }
            }
            if max_cc_bond > 1.48 {
                continue;
            }
            for &u in cyc {
                in_planar_ring[u] = true;
            }
            planar_cycles.push((cyc.clone(), verts, center, normal));
        }

        if planar_cycles.is_empty() {
            continue;
        }

        // Promote bonds inside planar conjugated cycles to `BondKind::Aromatic`, exocyclic short `C=O`/`C=S`
        // bonds (`<= 1.28 A`) to `BondKind::Double`, and assign valence-consistent `explicit_hydrogens`
        // when 3D H coordinates were omitted so `octet_mol::perceive_aromaticity` checks Huckel 4n+2!
        for b in &mut mol3d.bonds {
            let u = b.a;
            let v = b.b;
            let d = v_dist(heavy_atoms[u].pos, heavy_atoms[v].pos);
            if in_planar_ring[u] && in_planar_ring[v] {
                b.kind = BondKind::Aromatic;
            } else if (in_planar_ring[u] != in_planar_ring[v])
                && d <= 1.28
                && (matches!(heavy_atoms[u].z, 8 | 16) || matches!(heavy_atoms[v].z, 8 | 16))
            {
                b.kind = BondKind::Double;
            }
        }
        for u in 0..n_h {
            if !in_planar_ring[u] {
                continue;
            }
            mol3d.atoms[u].aromatic = true;
            let heavy_deg = mol3d.covalent_neighbors(u).len();
            let has_exo_double = mol3d
                .covalent_neighbors(u)
                .iter()
                .any(|&(_, bi)| mol3d.bonds[bi].kind == BondKind::Double);
            mol3d.atoms[u].explicit_hydrogens = match heavy_atoms[u].z {
                6 => Some(if heavy_deg <= 2 && !has_exo_double {
                    1
                } else {
                    0
                }),
                7 => {
                    // Count explicit 3D protons within 1.18 A if present; otherwise 2-coord planar N is pyridine-like (0 H)
                    // unless part of a 5-ring pyrrole/imidazole NH needed for 6 pi electrons.
                    let exp_h = grp_atoms
                        .iter()
                        .filter(|a| a.z == 1 && v_dist(a.pos, heavy_atoms[u].pos) <= 1.18)
                        .count();
                    Some(exp_h.min(1) as u8)
                }
                _ => Some(0),
            };
        }

        let mut arom = perceive_aromaticity(&mol3d);
        // For heavy-atom-only 5-membered N-heterocycles (e.g. pyrrole/imidazole/indole/purine without explicit H in PDB),
        // test protonating one 2-coordinate 5-ring nitrogen (`[nH]`) if unprotonated Huckel gave 5 pi electrons.
        for (cyc, _, _, _) in &planar_cycles {
            if cyc.len() == 5 && !cyc.iter().all(|&u| arom.aromatic_atoms[u]) {
                for &u in cyc {
                    if heavy_atoms[u].z == 7
                        && mol3d.covalent_neighbors(u).len() == 2
                        && mol3d.atoms[u].explicit_hydrogens == Some(0)
                    {
                        mol3d.atoms[u].explicit_hydrogens = Some(1);
                        let recheck = perceive_aromaticity(&mol3d);
                        if cyc.iter().all(|&w| recheck.aromatic_atoms[w]) {
                            arom = recheck;
                            break;
                        }
                        mol3d.atoms[u].explicit_hydrogens = Some(0);
                    }
                }
            }
        }

        for (r_idx, (cyc, verts, center, normal)) in planar_cycles.into_iter().enumerate() {
            if cyc.iter().all(|&u| arom.aromatic_atoms[u]) {
                let z_cyc: Vec<u8> = cyc.iter().map(|&u| heavy_atoms[u].z).collect();
                out_rings.push(ConjugatedAromaticRing {
                    chain_id: chain_id.clone(),
                    seq_id,
                    residue_index: usize::MAX,
                    res_name: res_name.clone(),
                    ring_label: format!("HET_RING_{}", r_idx + 1),
                    vertices: verts,
                    center,
                    normal,
                    intensity_factor: ring_intensity_factor_for_elements(&z_cyc),
                });
            }
        }
    }

    out_rings
}

fn parse_complex_pose(
    coord_text: &str,
    format_hint: Option<&str>,
    force_bfactor_plddt: bool,
    ligand_molecules: &BTreeMap<String, Molecule>,
) -> Result<ParsedComplexPose, NmrError> {
    let all_atoms = parse_all_atoms_from_pose_str(coord_text, format_hint)?;
    let protein_model = StructureModel::parse_str(coord_text, format_hint, force_bfactor_plddt)
        .map_err(NmrError::parse)?;
    let protein_res_keys: BTreeSet<(String, i32)> = protein_model
        .residues
        .iter()
        .map(|r| (r.chain_id.clone(), r.seq_id))
        .collect();
    let non_protein_aromatic_rings =
        perceive_non_protein_aromatic_rings(&all_atoms, &protein_res_keys, ligand_molecules);
    Ok(ParsedComplexPose {
        protein_model,
        all_atoms,
        non_protein_aromatic_rings,
    })
}

/// Compute `CAPRI` heavy-atom interface RMSD (`iRMSD` in `A`; Mendez et al. *Proteins* **2003**)
/// between a candidate complex pose and a reference native complex pose.
///
/// Interface residues are defined on `reference_coord_text` as all receptor residues within `10.0 A`
/// heavy-atom distance of the binding partner(s), together with all partner interface residues (within
/// `10.0 A` of the receptor) and all non-water small-molecule ligand heavy atoms.
pub fn compute_interface_rmsd_angstrom(
    candidate_coord_text: &str,
    candidate_format_hint: Option<&str>,
    reference_coord_text: &str,
    reference_format_hint: Option<&str>,
    receptor_chain_id: &str,
) -> Result<f64, NmrError> {
    let ref_atoms = parse_all_atoms_from_pose_str(reference_coord_text, reference_format_hint)?;
    let cand_atoms = parse_all_atoms_from_pose_str(candidate_coord_text, candidate_format_hint)?;
    compute_interface_rmsd_from_atoms(&cand_atoms, &ref_atoms, receptor_chain_id)
}

fn compute_interface_rmsd_from_atoms(
    cand_atoms: &[ComplexAtomRecord],
    ref_atoms: &[ComplexAtomRecord],
    receptor_chain_id: &str,
) -> Result<f64, NmrError> {
    let ref_rec: Vec<&ComplexAtomRecord> = ref_atoms
        .iter()
        .filter(|a| a.z > 1 && a.chain_id == receptor_chain_id)
        .collect();
    let ref_partner: Vec<&ComplexAtomRecord> = ref_atoms
        .iter()
        .filter(|a| a.z > 1 && a.chain_id != receptor_chain_id)
        .collect();
    if ref_rec.is_empty() || ref_partner.is_empty() {
        return Err(NmrError::invalid_input(
            "compute_interface_rmsd_angstrom requires both receptor and partner heavy atoms in reference pose",
        ));
    }

    let cut_sq = CAPRI_INTERFACE_CUTOFF_A * CAPRI_INTERFACE_CUTOFF_A;
    let mut iface_res: BTreeSet<(String, i32, Option<String>)> = BTreeSet::new();
    for ra in &ref_rec {
        for pa in &ref_partner {
            if v_dist_sq(ra.pos, pa.pos) <= cut_sq {
                iface_res.insert((ra.chain_id.clone(), ra.seq_id, ra.ins_code.clone()));
                iface_res.insert((pa.chain_id.clone(), pa.seq_id, pa.ins_code.clone()));
            }
        }
    }

    // Build `(chain_id, seq_id, ins_code, atom_name) -> pos` map on candidate
    let mut cand_map: BTreeMap<(String, i32, Option<String>, String), Vec3> = BTreeMap::new();
    for a in cand_atoms {
        if a.z > 1 {
            cand_map
                .entry((
                    a.chain_id.clone(),
                    a.seq_id,
                    a.ins_code.clone(),
                    a.atom_name.clone(),
                ))
                .or_insert(a.pos);
        }
    }

    // Check if the binding partner is a non-polymer `HETATM` small-molecule ligand vs polymer chain:
    // - For receptor + small-molecule ligand complexes (`PoseBusters` / `AF3` / `Boltz-2` pocket-aligned
    //   ligand iRMSD): superpose the receptor interface pocket backbone (`<= 10.0 A` of reference ligand)
    //   via proper Kabsch `(R, c_mov, c_ref)` and compute heavy-atom RMSD on the ligand in that pocket frame!
    // - For polymer-polymer complexes (`CAPRI` iRMSD): superpose matched backbone `(N, CA, C, O)` /
    //   nucleic `(P, C4', C1')` atoms across both receptor and partner interface residues.
    let has_polymer_partner_iface = ref_partner.iter().any(|pa| {
        !pa.is_hetatm
            && iface_res.contains(&(pa.chain_id.clone(), pa.seq_id, pa.ins_code.clone()))
            && matches!(
                pa.atom_name.as_str(),
                "N" | "CA" | "C" | "O" | "P" | "C4'" | "C1'"
            )
    });

    if !has_polymer_partner_iface {
        let mut pocket_ref = Vec::new();
        let mut pocket_cand = Vec::new();
        for ra in &ref_rec {
            if !iface_res.contains(&(ra.chain_id.clone(), ra.seq_id, ra.ins_code.clone())) {
                continue;
            }
            if !matches!(
                ra.atom_name.as_str(),
                "N" | "CA" | "C" | "O" | "P" | "C4'" | "C1'"
            ) {
                continue;
            }
            if let Some(&cp) = cand_map.get(&(
                ra.chain_id.clone(),
                ra.seq_id,
                ra.ins_code.clone(),
                ra.atom_name.clone(),
            )) {
                pocket_ref.push(ra.pos);
                pocket_cand.push(cp);
            }
        }
        // Fallback to all receptor backbone atoms if reference ligand had < 4 pocket backbone atoms within 10 A
        if pocket_ref.len() < 4 {
            pocket_ref.clear();
            pocket_cand.clear();
            for ra in &ref_rec {
                if matches!(ra.atom_name.as_str(), "N" | "CA" | "C" | "O") {
                    if let Some(&cp) = cand_map.get(&(
                        ra.chain_id.clone(),
                        ra.seq_id,
                        ra.ins_code.clone(),
                        ra.atom_name.clone(),
                    )) {
                        pocket_ref.push(ra.pos);
                        pocket_cand.push(cp);
                    }
                }
            }
        }

        let mut lig_ref = Vec::new();
        let mut lig_cand = Vec::new();
        for pa in &ref_partner {
            if let Some(&cp) = cand_map.get(&(
                pa.chain_id.clone(),
                pa.seq_id,
                pa.ins_code.clone(),
                pa.atom_name.clone(),
            )) {
                lig_ref.push(pa.pos);
                lig_cand.push(cp);
            }
        }

        if pocket_ref.len() >= 3 && !lig_ref.is_empty() {
            let (rot, c_mov, c_ref, _) = kabsch_transform(&pocket_cand, &pocket_ref);
            let mut sq_sum = 0.0_f64;
            for (lp_c, lp_r) in lig_cand.iter().zip(lig_ref.iter()) {
                let d = v_sub(*lp_c, c_mov);
                let aligned = [
                    rot[0][0] * d[0] + rot[0][1] * d[1] + rot[0][2] * d[2] + c_ref[0],
                    rot[1][0] * d[0] + rot[1][1] * d[1] + rot[1][2] * d[2] + c_ref[1],
                    rot[2][0] * d[0] + rot[2][1] * d[1] + rot[2][2] * d[2] + c_ref[2],
                ];
                sq_sum += v_dist_sq(aligned, *lp_r);
            }
            return Ok((sq_sum / (lig_ref.len() as f64)).sqrt());
        }
    }

    let mut pts_ref = Vec::new();
    let mut pts_cand = Vec::new();
    for a in ref_atoms {
        if a.z <= 1 {
            continue;
        }
        let in_iface =
            iface_res.contains(&(a.chain_id.clone(), a.seq_id, a.ins_code.clone())) || a.is_hetatm;
        if !in_iface {
            continue;
        }
        let use_atom = a.is_hetatm
            || matches!(
                a.atom_name.as_str(),
                "N" | "CA" | "C" | "O" | "P" | "C4'" | "C1'"
            );
        if !use_atom {
            continue;
        }
        if let Some(&c_pos) = cand_map.get(&(
            a.chain_id.clone(),
            a.seq_id,
            a.ins_code.clone(),
            a.atom_name.clone(),
        )) {
            pts_ref.push(a.pos);
            pts_cand.push(c_pos);
        }
    }

    if pts_ref.len() < 4 {
        return Err(NmrError::restraint_mismatch(
            "fewer than 4 matched interface heavy atoms between candidate and reference poses",
        ));
    }
    Ok(kabsch_rmsd(&pts_cand, &pts_ref))
}

// ============================================================================
// 7. Multi-Pose Complex NMR Re-Scorer (`score_cofold_poses`)
// ============================================================================

/// Optional co-folding model confidence metrics (`iptm`, `ptm`, `ranking_score`, interface `PAE`).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CofoldConfidenceMetadata {
    pub iptm: Option<f64>,
    pub ptm: Option<f64>,
    pub ranking_score: Option<f64>,
    pub mean_interface_pae_a: Option<f64>,
}

impl CofoldConfidenceMetadata {
    /// Parse `AlphaFold 3` `summary_confidences.json` or `Boltz-2` `confidence_*.json` text.
    pub fn parse_json_str(json_text: &str) -> Result<Self, NmrError> {
        check_bounded_text(json_text, "cofold confidence JSON")?;
        let v: Value = serde_json::from_str(json_text)
            .map_err(|e| NmrError::parse(format!("invalid confidence JSON: {e}")))?;
        let f_opt = |keys: &[&str]| -> Option<f64> {
            keys.iter()
                .find_map(|k| v.get(*k).and_then(Value::as_f64))
                .filter(|x| x.is_finite())
        };
        Ok(Self {
            iptm: f_opt(&["iptm", "ligand_iptm", "protein_iptm"]),
            ptm: f_opt(&["ptm"]),
            ranking_score: f_opt(&["ranking_score", "confidence_score"]),
            mean_interface_pae_a: f_opt(&["mean_interface_pae", "interface_pae"]),
        })
    }
}

/// Single candidate `.cif`/`.pdb` complex pose passed to [`score_cofold_poses`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StructurePoseInput {
    pub pose_id: String,
    pub source_path: Option<PathBuf>,
    pub coord_text: String,
    pub format_hint: Option<String>,
    pub cofold_confidence: Option<CofoldConfidenceMetadata>,
}

impl StructurePoseInput {
    pub fn from_coord_str(
        pose_id: impl Into<String>,
        coord_text: impl Into<String>,
        format_hint: Option<&str>,
    ) -> Self {
        Self {
            pose_id: pose_id.into(),
            source_path: None,
            coord_text: coord_text.into(),
            format_hint: format_hint.map(str::to_string),
            cofold_confidence: None,
        }
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, NmrError> {
        let p = path.as_ref();
        let coord_text = read_bounded_utf8_file(p, "pose structure")?;
        let pose_id = p
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("pose")
            .to_string();
        let format_hint = p
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_ascii_lowercase());
        Ok(Self {
            pose_id,
            source_path: Some(p.to_path_buf()),
            coord_text,
            format_hint,
            cofold_confidence: None,
        })
    }

    pub fn with_confidence(mut self, conf: CofoldConfidenceMetadata) -> Self {
        self.cofold_confidence = Some(conf);
        self
    }
}

/// How [`score_cofold_poses`] computes the per-residue predicted `CSP` term `S_CSP` (`A7`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CspScoringMode {
    /// Contact / burial enrichment (`AUC_contact` + `delta_rSASA` burial rank correlation + pocket
    /// distance penalty) without adding `T21.2` `Delta_delta_calc` torsion shifts, which sit below
    /// `T21.2`'s `~0.55 ppm` `1H` RMSE noise floor whenever apo and complex backbone conformations
    /// differ (`A7`).
    #[default]
    BurialAndContactOnly,
    /// Include `T21.2` `Delta_delta_calc` + ligand Haigh-Mallion + H-bond terms alongside burial.
    FullForwardShiftAndBurial,
}

fn default_min_alignment_identity() -> f64 {
    0.90
}

/// Configuration for [`score_cofold_poses`] (`docs/specs/21-alphafold-integrative-nmr.md` section 6.2).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PoseScoreConfig {
    /// Weight `w_CSP` multiplying `S_CSP(m)` in `S_total(m)` (`default = 4.0`).
    pub w_csp: f64,
    /// Weight `w_NOE` multiplying `E_viol^{NOE/PRE}(m)` in `S_total(m)` (`default = 1.5`).
    pub w_noe_pre: f64,
    /// Weight `w_RDC` multiplying `ByPeptidePlane Q_free^{RDC}(m)` in `S_total(m)` (`default = 2.5`).
    pub w_rdc: f64,
    /// Weight `w_PCS` multiplying `Q^{PCS}(m)` in `S_total(m)` (`default = 2.5`).
    pub w_pcs: f64,
    /// Heavy-atom contact cutoff (`default = 5.0 A`) for `CSP` pocket contact ROC-AUC.
    pub contact_cutoff_angstrom: f64,
    /// Maximum allowed intermolecular distance violation (`default = 1.50 A`) before decoy flag.
    pub max_allowed_distance_violation_angstrom: f64,
    /// Minimum `CSP` contact ROC-AUC (`default = 0.65`) when `>= min_surface_hits_for_auc_guard` exist.
    pub min_csp_roc_auc: f64,
    /// Minimum surface `2-sigma` `CSP` hit count (`default = 5`) activating `min_csp_roc_auc`.
    pub min_surface_hits_for_auc_guard: usize,
    /// Maximum allowed `ByPeptidePlane` `RDC` `Q_free` (`default = 0.65`) before decoy flag.
    pub max_allowed_rdc_q_free: f64,
    /// Held-out fraction (`default = 0.20`) for `ByPeptidePlane` `RDC` `Q_free` partitioning.
    pub rdc_free_fraction: f64,
    /// Deterministic PRNG seed for `ByPeptidePlane` `RDC` `Q_free` holdout when `is_free` is unset.
    pub rdc_free_seed: u64,
    /// Optional apo receptor `.cif`/`.pdb` text used as the uncomplexed baseline for `T21.2` differential shifts.
    pub apo_receptor_coord_text: Option<String>,
    pub apo_receptor_format_hint: Option<String>,
    /// Optional native/reference complex `(coord_text, format_hint)` for `CAPRI` `iRMSD` reporting.
    pub reference_complex_pose: Option<(String, Option<String>)>,
    /// Forward shift predictor configuration (`T21.2` `struct_shifts`).
    pub struct_shift_config: StructShiftConfig,
    /// `HomologFirewall` exclusion query (`max_seq_identity <= 0.25`, `TM-score < 0.50`).
    pub firewall_query: HomologFirewallQuery,
    /// Whether `S_CSP` uses burial/contact-only (`A7` default) or includes `T21.2` `Delta_delta_calc`.
    #[serde(default)]
    pub csp_scoring_mode: CspScoringMode,
    /// Minimum amino-acid identity (`default = 0.90`) for pose-to-target alignment re-indexing.
    #[serde(default = "default_min_alignment_identity")]
    pub min_alignment_identity: f64,
}

impl Default for PoseScoreConfig {
    fn default() -> Self {
        Self {
            w_csp: 4.0,
            w_noe_pre: 1.5,
            w_rdc: 2.5,
            w_pcs: 2.5,
            contact_cutoff_angstrom: DEFAULT_CSP_CONTACT_CUTOFF_A,
            max_allowed_distance_violation_angstrom: DEFAULT_MAX_ALLOWED_VIOLATION_A,
            min_csp_roc_auc: DEFAULT_MIN_CSP_ROC_AUC,
            min_surface_hits_for_auc_guard: DEFAULT_MIN_SURFACE_HITS_FOR_AUC_GUARD,
            max_allowed_rdc_q_free: DEFAULT_MAX_ALLOWED_RDC_Q_FREE,
            rdc_free_fraction: 0.20,
            rdc_free_seed: 0x434F_464F_4C44_3231,
            apo_receptor_coord_text: None,
            apo_receptor_format_hint: None,
            reference_complex_pose: None,
            struct_shift_config: StructShiftConfig {
                max_seq_identity: DEFAULT_FIREWALL_MAX_SEQ_IDENTITY,
                ..StructShiftConfig::default()
            },
            firewall_query: HomologFirewallQuery::default(),
            csp_scoring_mode: CspScoringMode::BurialAndContactOnly,
            min_alignment_identity: default_min_alignment_identity(),
        }
    }
}

/// Specific NMR guard condition violated by a candidate pose.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PoseViolationFlag {
    /// `max_distance_violation_angstrom > max_allowed_distance_violation_angstrom` (`1.50 A`).
    MaxDistanceViolationExceeded,
    /// `csp_contact_roc_auc < min_csp_roc_auc` (`0.65`) with `>= 5` surface `CSP` pocket hits.
    LowCspContactEnrichment,
    /// `ByPeptidePlane` `RDC` `Q_free > max_allowed_rdc_q_free` (`0.65`).
    HighRdcQFree,
}

/// Global ensemble verdict emitted by [`score_cofold_poses`] (`section 6.2(4)`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PoseVerdict {
    /// At least one candidate pose passed all active NMR concordance guards.
    RankedConclusive,
    /// Every candidate pose violated distance, `CSP` contact enrichment, or `RDC` `Q_free` guards.
    InconclusiveOrAllViolated,
}

/// Per-residue observed vs forward-predicted complex-minus-apo differential `CSP` diagnostic.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResidueCspCalcPoint {
    pub seq_id: i32,
    pub res_name: String,
    pub csp_obs_ppm: f64,
    pub csp_calc_ppm: f64,
    pub delta_h_calc_ppm: f64,
    pub delta_n_calc_ppm: f64,
    pub delta_rsasa_burial: f64,
    pub min_partner_heavy_dist_a: f64,
    pub is_surface_2sigma_hit: bool,
}

/// Full NMR score breakdown for one candidate complex pose.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PoseScoreEntry {
    /// `1`-based rank across all evaluated poses (`1` = highest composite NMR support).
    pub rank: usize,
    pub pose_id: String,
    pub source_path: Option<PathBuf>,
    /// Calibrated composite score `S_total(m) = w_CSP S_CSP - w_NOE E_viol - w_RDC Q_free - w_PCS Q_PCS`.
    pub total_score: f64,
    pub passes_nmr_guards: bool,
    pub violation_flags: Vec<PoseViolationFlag>,
    pub s_csp: f64,
    pub csp_spearman_rho: Option<f64>,
    pub csp_contact_roc_auc: Option<f64>,
    /// One-sided Mann-Whitney U `p`-value testing `AUC_contact > 0.50` (`None` if no ROC samples).
    #[serde(default)]
    pub csp_contact_p_value: Option<f64>,
    pub mean_pocket_excess_dist_a: f64,
    pub noe_pre_violation_energy: f64,
    pub max_distance_violation_angstrom: f64,
    pub n_violated_distance_restraints: usize,
    pub rdc_q_work: Option<f64>,
    pub rdc_q_free: Option<f64>,
    pub pcs_q_factor: Option<f64>,
    pub irmsd_to_reference_angstrom: Option<f64>,
    pub cofold_confidence: Option<CofoldConfidenceMetadata>,
    pub exclusion_list_sha256: String,
    pub residue_csp_points: Vec<ResidueCspCalcPoint>,
}

/// Multi-pose co-folded complex NMR re-scoring report returned by [`score_cofold_poses`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PoseScoreReport {
    pub job_name: String,
    pub verdict: PoseVerdict,
    /// `Some(pose_id)` of Rank `#1` when `verdict == PoseVerdict::RankedConclusive`, else `None`.
    pub top_pose_id: Option<String>,
    pub n_poses_evaluated: usize,
    pub n_admissible_poses: usize,
    pub n_surface_csp_hits: usize,
    pub n_distance_restraints: usize,
    pub n_rdc_media: usize,
    pub n_pcs_tags: usize,
    /// `CspScoringMode` used to evaluate `S_CSP` (`A7`).
    #[serde(default)]
    pub csp_scoring_mode: CspScoringMode,
    /// Spearman rank correlation `rho_s(-iRMSD, total_score)` when `cfg.reference_complex_pose` is supplied.
    pub funnel_spearman_rho_vs_neg_irmsd: Option<f64>,
    pub poses: Vec<PoseScoreEntry>,
}

/// Tie-adjusted Spearman rank correlation `rho_s in [-1, +1]` (`0.0` when `< 3` points or zero variance).
pub fn spearman_rank_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n < 3 || y.len() != n {
        return 0.0;
    }
    let rx = average_midranks(x);
    let ry = average_midranks(y);
    let nf = n as f64;
    let mx = rx.iter().sum::<f64>() / nf;
    let my = ry.iter().sum::<f64>() / nf;
    let mut cov = 0.0;
    let mut vx = 0.0;
    let mut vy = 0.0;
    for i in 0..n {
        let dx = rx[i] - mx;
        let dy = ry[i] - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    let denom = (vx * vy).sqrt();
    if denom <= 1e-15 {
        0.0
    } else {
        (cov / denom).clamp(-1.0, 1.0)
    }
}

fn average_midranks(vals: &[f64]) -> Vec<f64> {
    let n = vals.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| vals[a].total_cmp(&vals[b]));
    let mut ranks = vec![0.0_f64; n];
    let mut i = 0usize;
    while i < n {
        let mut j = i + 1;
        while j < n && (vals[order[j]] - vals[order[i]]).abs() <= 1e-12 {
            j += 1;
        }
        // 1-based midrank of span `i..j` is `(i + 1 + j) / 2`
        let mid = 0.5 * ((i + 1 + j) as f64);
        for &orig in &order[i..j] {
            ranks[orig] = mid;
        }
        i = j;
    }
    ranks
}

/// Exact Mann-Whitney U / trapezoidal ROC-AUC (`in [0, 1]`, `0.5` on tied/constant scores).
pub fn binary_roc_auc(labels_and_scores: &[(bool, f64)]) -> Option<f64> {
    let n_pos = labels_and_scores.iter().filter(|&&(y, _)| y).count();
    let n_neg = labels_and_scores.len().saturating_sub(n_pos);
    if n_pos == 0 || n_neg == 0 {
        return None;
    }
    let mut sorted: Vec<(bool, f64)> = labels_and_scores.to_vec();
    sorted.sort_by(|a, b| a.1.total_cmp(&b.1));

    let n = sorted.len();
    let mut rank_sum_pos = 0.0_f64;
    let mut i = 0usize;
    while i < n {
        let mut j = i + 1;
        while j < n && (sorted[j].1 - sorted[i].1).abs() <= 1e-12 {
            j += 1;
        }
        let midrank = 0.5 * ((i + 1 + j) as f64);
        for &(is_pos, _) in &sorted[i..j] {
            if is_pos {
                rank_sum_pos += midrank;
            }
        }
        i = j;
    }

    let nf_pos = n_pos as f64;
    let nf_neg = n_neg as f64;
    let u_stat = rank_sum_pos - 0.5 * nf_pos * (nf_pos + 1.0);
    Some((u_stat / (nf_pos * nf_neg)).clamp(0.0, 1.0))
}

/// Complementary error function `erfc(x)` for `x >= 0` (Abramowitz & Stegun 7.1.26, max err `1.5e-7`),
/// used for `wasm32`-clean deterministic Mann-Whitney `p`-values.
fn erfc_approx(x: f64) -> f64 {
    if x < 0.0 {
        return 2.0 - erfc_approx(-x);
    }
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    (poly * (-x * x).exp()).clamp(0.0, 2.0)
}

/// One-sided tie-corrected Mann-Whitney U `p`-value testing `AUC_contact > 0.50`.
pub fn mann_whitney_p_value(labels_and_scores: &[(bool, f64)]) -> Option<f64> {
    let n_pos = labels_and_scores.iter().filter(|&&(y, _)| y).count();
    let n_neg = labels_and_scores.len().saturating_sub(n_pos);
    if n_pos == 0 || n_neg == 0 {
        return None;
    }
    let mut sorted: Vec<(bool, f64)> = labels_and_scores.to_vec();
    sorted.sort_by(|a, b| a.1.total_cmp(&b.1));

    let n = sorted.len();
    let mut rank_sum_pos = 0.0_f64;
    let mut tie_corr = 0.0_f64;
    let mut i = 0usize;
    while i < n {
        let mut j = i + 1;
        while j < n && (sorted[j].1 - sorted[i].1).abs() <= 1e-12 {
            j += 1;
        }
        let t = (j - i) as f64;
        if t > 1.0 {
            tie_corr += t * t * t - t;
        }
        let midrank = 0.5 * ((i + 1 + j) as f64);
        for &(is_pos, _) in &sorted[i..j] {
            if is_pos {
                rank_sum_pos += midrank;
            }
        }
        i = j;
    }

    let nf_pos = n_pos as f64;
    let nf_neg = n_neg as f64;
    let nf = n as f64;
    let u_stat = rank_sum_pos - 0.5 * nf_pos * (nf_pos + 1.0);
    let mu = 0.5 * nf_pos * nf_neg;
    let var_term = (nf + 1.0)
        - if nf > 1.0 {
            tie_corr / (nf * (nf - 1.0))
        } else {
            0.0
        };
    let var = (nf_pos * nf_neg * var_term / 12.0).max(0.0);
    if var <= 1e-15 {
        return Some(1.0);
    }
    let z = (u_stat - mu) / var.sqrt();
    Some((0.5 * erfc_approx(z / std::f64::consts::SQRT_2)).clamp(0.0, 1.0))
}

fn filter_model_to_chain(model: &StructureModel, receptor_chain_id: &str) -> StructureModel {
    let mut old_to_new: BTreeMap<usize, usize> = BTreeMap::new();
    let mut residues = Vec::new();
    for (old_i, r) in model.residues.iter().enumerate() {
        if r.chain_id == receptor_chain_id {
            old_to_new.insert(old_i, residues.len());
            residues.push(r.clone());
        }
    }
    let aromatic_rings = model
        .aromatic_rings
        .iter()
        .filter_map(|ring| {
            old_to_new.get(&ring.residue_index).map(|&new_i| {
                let mut cloned = ring.clone();
                cloned.residue_index = new_i;
                cloned
            })
        })
        .collect();
    StructureModel {
        provenance: model.provenance,
        residues,
        aromatic_rings,
        diagnostics: model.diagnostics.clone(),
    }
}

/// Build a unified [`HomologFirewallQuery`] that excludes any template triggered by either the
/// individual chains or the full complex so `isolated_receptor` and `complex_pose` share the
/// exact same non-homologous Fourier torsion surfaces!
fn build_unified_firewall_query(
    base_query: &HomologFirewallQuery,
    max_seq_id: f64,
    isolated_receptor: &StructureModel,
    complex_model: &StructureModel,
) -> HomologFirewallQuery {
    let db = TorsionDb::embedded();
    let mut excluded_pdbs: BTreeSet<String> = base_query.query_pdb_ids.iter().cloned().collect();

    for m in [isolated_receptor, complex_model] {
        if m.residues.is_empty() {
            continue;
        }
        let mut q = base_query.clone();
        if q.query_ca_coords.is_none() && m.residues.len() >= 8 {
            q.query_ca_coords = Some(m.residues.iter().map(|r| r.ca).collect());
        }
        let seq1: Vec<u8> = m.residues.iter().map(|r| r.aa1).collect();
        let fw = HomologFirewall::new(q).with_max_seq_identity(max_seq_id);
        let (_, excl, _, _) = fw.apply(db, &seq1);
        for rec in excl {
            excluded_pdbs.insert(rec.pdb_id);
        }
    }

    let mut unified = base_query.clone();
    unified.query_pdb_ids = excluded_pdbs.into_iter().collect();
    if unified.query_ca_coords.is_none() && isolated_receptor.residues.len() >= 8 {
        unified.query_ca_coords = Some(isolated_receptor.residues.iter().map(|r| r.ca).collect());
    }
    unified
}

/// Resolve all matching 3D atom coordinates on `pose` for `(chain_id, seq_id, atom_pattern)`
/// (supporting amide `H`/`HN`, degenerate methylene `HB*`/`HG*`, methyl `HD1*`/`HG1*`, and `*` / `ANY` wildcards).
fn resolve_pose_atom_coords(
    pose: &ParsedComplexPose,
    chain_id: &str,
    seq_id: i32,
    atom_pattern: &str,
) -> Vec<Vec3> {
    let pat = atom_pattern.trim().to_ascii_uppercase();
    let mut coords = Vec::new();

    // First check proton-completed `protein_model` residues (which carry reconstructed ideal `H` and `HA`!)
    for r in &pose.protein_model.residues {
        if r.chain_id != chain_id || r.seq_id != seq_id {
            continue;
        }
        match pat.as_str() {
            "H" | "HN" => {
                if let Some(h) = r.h {
                    coords.push(h);
                }
            }
            "N" => coords.push(r.n),
            "CA" => coords.push(r.ca),
            "HA" => {
                coords.push(r.ha);
            }
            "HA2" => {
                if let Some(p) = r.ha2 {
                    coords.push(p);
                }
            }
            "HA3" => {
                if let Some(p) = r.ha3 {
                    coords.push(p);
                }
            }
            "C" => coords.push(r.c),
            "O" => coords.push(r.o),
            "CB" | "QB" => {
                if let Some(cb) = r.cb {
                    coords.push(cb);
                }
            }
            "*" | "ANY" | "ALL" => {
                for &p in r.atoms.values() {
                    coords.push(p);
                }
            }
            _ => {
                if let Some(prefix) = pat.strip_suffix('*') {
                    for (an, &p) in &r.atoms {
                        if an.starts_with(prefix) {
                            coords.push(p);
                        }
                    }
                    if coords.is_empty() && prefix.starts_with("HB") {
                        if let Some(cb) = r.cb {
                            coords.push(cb);
                        }
                    }
                } else if let Some(&p) = r.atoms.get(&pat) {
                    coords.push(p);
                }
            }
        }
        if !coords.is_empty() {
            return coords;
        }
    }

    // Check `all_atoms` (`HETATM` small-molecule ligands, RNA/DNA nucleotides, cofactors, metals)
    let chain_atoms: Vec<&ComplexAtomRecord> = pose
        .all_atoms
        .iter()
        .filter(|a| a.chain_id == chain_id && (a.seq_id == seq_id || a.is_hetatm))
        .collect();
    for a in &chain_atoms {
        if pat == "*" || pat == "ANY" || pat == "ALL" {
            if a.z > 1 {
                coords.push(a.pos);
            }
        } else if let Some(prefix) = pat.strip_suffix('*') {
            if a.atom_name.starts_with(prefix) {
                coords.push(a.pos);
            }
        } else if a.atom_name == pat {
            coords.push(a.pos);
        }
    }
    coords
}

fn evaluate_pose_rdc_q(
    pose: &ParsedComplexPose,
    media_in: &[CofoldRdcMediumSet],
    free_fraction: f64,
    free_seed: u64,
) -> Result<(f64, f64), NmrError> {
    if media_in.is_empty() {
        return Ok((f64::NAN, f64::NAN));
    }

    // Map each distinct protein `chain_id` to a disjoint `100_000 * chain_slot` synthetic seq_id block
    // so multi-subunit complexes (`A: 1..76`, `B: 1..56`) have zero `seq_id` collisions and zero
    // cross-chain `ByPeptidePlane` `(i-1, i)` coupling!
    let mut chain_slots: BTreeMap<String, i32> = BTreeMap::new();
    for r in &pose.protein_model.residues {
        let next_slot = (chain_slots.len() as i32) * 100_000;
        chain_slots.entry(r.chain_id.clone()).or_insert(next_slot);
    }

    let mut pdb_residues = Vec::with_capacity(pose.protein_model.residues.len());
    for r in &pose.protein_model.residues {
        let base = chain_slots.get(&r.chain_id).copied().unwrap_or(0);
        let ch_char = r.chain_id.chars().next().unwrap_or('A');
        pdb_residues.push(PdbResidue {
            seq_id: base + r.seq_id,
            res_name: r.res_name.clone(),
            chain_id: ch_char,
            n: r.n,
            h: r.h,
            ca: r.ca,
            ha: r.ha,
            ha2: r.ha2,
            ha3: r.ha3,
            c: r.c,
            o: r.o,
            cb: r.cb,
            qb: r.cb,
        });
    }
    let combined_chain = PdbChain {
        residues: pdb_residues,
    };

    let mut remapped_media: Vec<RdcMediumSet> = Vec::with_capacity(media_in.len());
    let mut any_preset_free = false;
    for m in media_in {
        let mut rs = Vec::with_capacity(m.restraints.len());
        for r in &m.restraints {
            let Some(&b1) = chain_slots.get(&r.chain_1) else {
                return Err(NmrError::restraint_mismatch(format!(
                    "RDC chain '{}' not found in pose protein chains",
                    r.chain_1
                )));
            };
            let Some(&b2) = chain_slots.get(&r.chain_2) else {
                return Err(NmrError::restraint_mismatch(format!(
                    "RDC chain '{}' not found in pose protein chains",
                    r.chain_2
                )));
            };
            if r.is_free {
                any_preset_free = true;
            }
            rs.push(RdcRestraint {
                medium_id: r.medium_id.clone(),
                seq_id_1: b1 + r.seq_id_1,
                res_name_1: r.res_name_1.clone(),
                atom_name_1: r.atom_name_1.clone(),
                seq_id_2: b2 + r.seq_id_2,
                res_name_2: r.res_name_2.clone(),
                atom_name_2: r.atom_name_2.clone(),
                bond_type: r.bond_type,
                d_obs_hz: r.d_obs_hz,
                sigma_hz: r.sigma_hz,
                d_max_hz: r.d_max_hz,
                is_free: r.is_free,
            });
        }
        remapped_media.push(RdcMediumSet {
            medium_id: m.medium_id.clone(),
            restraints: rs,
        });
    }

    if !any_preset_free && free_fraction > 0.0 {
        partition_rdc_free_set_with_mode(
            &mut remapped_media,
            free_fraction,
            free_seed,
            RdcFreeStratification::ByPeptidePlane,
        );
    }

    let summary = evaluate_rdc_validation(&combined_chain, &remapped_media)
        .map_err(NmrError::restraint_mismatch)?;
    Ok((summary.overall_q_work, summary.overall_q_free))
}

fn evaluate_pose_pcs_q(
    pose: &ParsedComplexPose,
    tags: &[CofoldPcsTagSet],
) -> Result<Option<f64>, NmrError> {
    if tags.is_empty() {
        return Ok(None);
    }
    let mut all_pairs: Vec<(f64, f64)> = Vec::new();

    for tag in tags {
        let metal_pos = if let Some((ref m_chain, m_seq, ref m_atom)) = tag.metal_anchor {
            resolve_pose_atom_coords(pose, m_chain, m_seq, m_atom)
                .into_iter()
                .next()
                .ok_or_else(|| {
                    NmrError::restraint_mismatch(format!(
                        "PCS metal anchor {}:{}:{} not found in pose",
                        m_chain, m_seq, m_atom
                    ))
                })?
        } else if let Some(mp) = tag.metal_pos {
            mp
        } else {
            // Auto-detect paramagnetic/transition/lanthanide HETATM metal ion in pose
            pose.all_atoms
                .iter()
                .find(|a| a.is_hetatm && crate::octet_mol::is_metal(a.z))
                .map(|a| a.pos)
                .ok_or_else(|| {
                    NmrError::invalid_input(format!(
                        "PCS tag '{}' specifies neither metal_pos, metal_anchor, nor a HETATM metal ion",
                        tag.tag_id
                    ))
                })?
        };

        let mut fit_obs = Vec::new();
        let mut eval_obs = Vec::new();
        for o in &tag.observations {
            let Some(coord) = resolve_pose_atom_coords(pose, &o.chain_id, o.seq_id, &o.atom_name)
                .into_iter()
                .next()
            else {
                continue;
            };
            let pcs_pt =
                PcsObservation::new(o.seq_id, coord, o.pcs_obs_ppm).with_uncertainty(o.sigma_ppm);
            let in_host = tag
                .host_chain_id
                .as_ref()
                .map(|hc| hc == &o.chain_id)
                .unwrap_or(true);
            if !o.is_free && in_host {
                fit_obs.push(pcs_pt);
            }
            eval_obs.push((pcs_pt, o.is_free, in_host));
        }

        let tensor = if let Some(ref pre) = tag.pre_fitted_tensor {
            pre.clone()
        } else {
            // If intra-host working observations < 6, fit across all working observations
            if fit_obs.len() < 6 {
                fit_obs = eval_obs
                    .iter()
                    .filter(|&&(_, is_free, _)| !is_free)
                    .map(|&(pt, _, _)| pt)
                    .collect();
            }
            if fit_obs.len() < 5 {
                return Err(NmrError::invalid_input(format!(
                    "PCS tag '{}' has only {} resolved working observations (need >= 5 for 5-param Delta_chi SVD)",
                    tag.tag_id,
                    fit_obs.len()
                )));
            }
            fit_pcs_tensor_svd(&fit_obs, metal_pos).map_err(NmrError::restraint_mismatch)?
        };

        // Evaluate Cornilescu Q preferentially on cross-subunit / held-out observations when >= 3 exist,
        // otherwise across all observations in `eval_obs`.
        let cross_or_free: Vec<(f64, f64)> = eval_obs
            .iter()
            .filter(|&&(_, is_free, in_host)| is_free || !in_host)
            .map(|&(pt, _, _)| {
                (
                    pt.pcs_ppm,
                    predict_pcs_from_tensor(pt.coord_angstrom, &tensor),
                )
            })
            .collect();
        if cross_or_free.len() >= 3 {
            all_pairs.extend(cross_or_free);
        } else {
            for &(pt, _, _) in &eval_obs {
                all_pairs.push((
                    pt.pcs_ppm,
                    predict_pcs_from_tensor(pt.coord_angstrom, &tensor),
                ));
            }
        }
    }

    if all_pairs.is_empty() {
        Ok(None)
    } else {
        let q = cornilescu_q(&all_pairs);
        Ok(q.is_finite().then_some(q))
    }
}

/// Re-index protein chain `chain_id` inside a [`StructureModel`] (and optionally matching
/// non-HETATM atoms in `all_atoms`) by sequence alignment against `target`, mapping `0`-based
/// target index `t0` to `exp_numbering[t0]` (or `(t0 + 1) as i32`). Unaligned tag/insertion
/// residues receive disjoint negative IDs `-100_000 - k` so they never collide with experimental
/// numbering while remaining physically present for SASA/sterics.
fn reindex_model_chain_to_experimental(
    model: &mut StructureModel,
    mut all_atoms: Option<&mut Vec<ComplexAtomRecord>>,
    chain_id: &str,
    target: &[u8],
    exp_numbering: Option<&[i32]>,
    min_identity: f64,
    label: &str,
) -> Result<(), NmrError> {
    if target.is_empty() || target.iter().all(|&b| b == b'X') {
        return Ok(());
    }
    let obs = model_chain_observed(model, chain_id);
    if obs.is_empty() {
        return Ok(());
    }
    let obs_letters: Vec<u8> = obs.iter().map(|o| o.1).collect();
    let aln = align_observed_to_target(&obs_letters, target)?;
    require_alignment_quality(label, &aln, min_identity)?;

    let mut pos_to_new: BTreeMap<usize, i32> = BTreeMap::new();
    let mut old_to_new: BTreeMap<i32, i32> = BTreeMap::new();
    for (k, tgt) in aln.observed_to_target.iter().enumerate() {
        let (old_sid, _, pos) = obs[k];
        let new_sid = match tgt {
            Some(t0) => exp_numbering
                .and_then(|v| v.get(*t0).copied())
                .unwrap_or((*t0 + 1) as i32),
            None => -100_000 - (k as i32),
        };
        pos_to_new.insert(pos, new_sid);
        old_to_new.insert(old_sid, new_sid);
    }

    for (&pos, &new_sid) in &pos_to_new {
        model.residues[pos].seq_id = new_sid;
    }
    for ring in &mut model.aromatic_rings {
        if let Some(&new_sid) = pos_to_new.get(&ring.residue_index) {
            ring.seq_id = new_sid;
        }
    }
    if let Some(ref mut atoms) = all_atoms {
        for a in atoms.iter_mut() {
            if !a.is_hetatm && a.chain_id == chain_id && a.ins_code.is_none() {
                if let Some(&new_sid) = old_to_new.get(&a.seq_id) {
                    a.seq_id = new_sid;
                }
            }
        }
    }
    Ok(())
}

fn reindex_pose_to_experimental(
    pose: &mut ParsedComplexPose,
    restraints: &CofoldRestraintBundle,
    min_identity: f64,
    label: &str,
) -> Result<(), NmrError> {
    for ent in &restraints.entities {
        if let CofoldEntitySpec::Polymer {
            kind: CofoldPolymerKind::Protein,
            chain_id,
            sequence,
        } = ent
        {
            let target = sequence.trim().to_ascii_uppercase().into_bytes();
            let exp_nums = restraints
                .chain_exp_numbering
                .get(chain_id)
                .map(Vec::as_slice);
            reindex_model_chain_to_experimental(
                &mut pose.protein_model,
                Some(&mut pose.all_atoms),
                chain_id,
                &target,
                exp_nums,
                min_identity,
                &format!("{label} chain '{chain_id}'"),
            )?;
        }
    }
    Ok(())
}

/// Rank and cross-validate $M \ge 1$ candidate co-folded complex poses (`.cif`/`.pdb`) against
/// experimental `CSP`, intermolecular `NOE`/`PRE`, `ByPeptidePlane` `RDC` `Q_free`, and `PCS`
/// restraints (`docs/specs/21-alphafold-integrative-nmr.md` section 6.2).
pub fn score_cofold_poses(
    poses: &[StructurePoseInput],
    restraints: &CofoldRestraintBundle,
    cfg: &PoseScoreConfig,
) -> Result<PoseScoreReport, NmrError> {
    if poses.is_empty() {
        return Err(NmrError::empty_input(
            "score_cofold_poses requires at least one candidate pose",
        ));
    }
    if !cfg.w_csp.is_finite()
        || !cfg.w_noe_pre.is_finite()
        || !cfg.w_rdc.is_finite()
        || !cfg.w_pcs.is_finite()
    {
        return Err(NmrError::invalid_input(
            "PoseScoreConfig weights must be finite",
        ));
    }

    let receptor_chain = if restraints.receptor_chain_id.trim().is_empty() {
        "A"
    } else {
        restraints.receptor_chain_id.trim()
    };
    let receptor_target: Option<Vec<u8>> = restraints.entities.iter().find_map(|e| match e {
        CofoldEntitySpec::Polymer {
            kind: CofoldPolymerKind::Protein,
            chain_id,
            sequence,
        } if chain_id == receptor_chain => Some(sequence.trim().to_ascii_uppercase().into_bytes()),
        _ => None,
    });

    let mut opt_apo_model = match &cfg.apo_receptor_coord_text {
        Some(txt) => Some(
            StructureModel::parse_str(
                txt,
                cfg.apo_receptor_format_hint.as_deref(),
                cfg.struct_shift_config.force_bfactor_plddt,
            )
            .map_err(NmrError::parse)?,
        ),
        None => None,
    };
    if let (Some(ref mut apo), Some(ref target)) = (&mut opt_apo_model, &receptor_target) {
        if let Some(apo_ch) = select_model_chain(apo, receptor_chain) {
            let exp_nums = restraints
                .chain_exp_numbering
                .get(receptor_chain)
                .map(Vec::as_slice);
            reindex_model_chain_to_experimental(
                apo,
                None,
                &apo_ch,
                target,
                exp_nums,
                cfg.min_alignment_identity,
                "apo baseline structure",
            )?;
        }
    }

    let opt_ref_pose = match &cfg.reference_complex_pose {
        Some((ref_txt, ref_hint)) => {
            let mut ref_parsed = parse_complex_pose(
                ref_txt,
                ref_hint.as_deref(),
                cfg.struct_shift_config.force_bfactor_plddt,
                &restraints.ligand_molecules,
            )?;
            reindex_pose_to_experimental(
                &mut ref_parsed,
                restraints,
                cfg.min_alignment_identity,
                "reference pose",
            )?;
            Some(ref_parsed.all_atoms)
        }
        None => None,
    };

    let mut obs_csp_by_seq: BTreeMap<i32, &ReceptorCspObservation> = BTreeMap::new();
    let mut n_surface_csp_hits = 0usize;
    for obs in &restraints.csp_observations {
        obs_csp_by_seq.insert(obs.seq_id, obs);
        if matches!(
            obs.perturbation_class,
            CspPerturbationClass::SurfacePocketContact
                | CspPerturbationClass::UnfilteredNoApoStructure
        ) {
            n_surface_csp_hits += 1;
        }
    }
    if obs_csp_by_seq.is_empty() && !restraints.pocket_contacts.is_empty() {
        n_surface_csp_hits = restraints.pocket_contacts.len();
    }

    let mut entries: Vec<PoseScoreEntry> = Vec::with_capacity(poses.len());

    for pose_in in poses {
        let mut parsed = parse_complex_pose(
            &pose_in.coord_text,
            pose_in.format_hint.as_deref(),
            cfg.struct_shift_config.force_bfactor_plddt,
            &restraints.ligand_molecules,
        )?;
        reindex_pose_to_experimental(
            &mut parsed,
            restraints,
            cfg.min_alignment_identity,
            &format!("pose '{}'", pose_in.pose_id),
        )?;

        let isolated_from_pose = filter_model_to_chain(&parsed.protein_model, receptor_chain);
        if isolated_from_pose.residues.is_empty() {
            return Err(NmrError::restraint_mismatch(format!(
                "pose '{}' contains zero protein residues on receptor chain '{}'",
                pose_in.pose_id, receptor_chain
            )));
        }

        let baseline_receptor = match &opt_apo_model {
            Some(apo) => {
                let sub = filter_model_to_chain(apo, receptor_chain);
                if sub.residues.is_empty() {
                    apo.clone()
                } else {
                    sub
                }
            }
            None => isolated_from_pose.clone(),
        };

        // 1. HomologFirewall SHA-256 audit + optional `T21.2` complex-vs-apo differential shifts
        let unified_fw = build_unified_firewall_query(
            &cfg.firewall_query,
            cfg.struct_shift_config.max_seq_identity,
            &baseline_receptor,
            &parsed.protein_model,
        );
        let mut apo_shift_map: BTreeMap<i32, (f64, f64)> = BTreeMap::new();
        let mut comp_shift_map: BTreeMap<i32, (f64, f64)> = BTreeMap::new();
        let exclusion_list_sha256 = match cfg.csp_scoring_mode {
            CspScoringMode::FullForwardShiftAndBurial => {
                let diff_shift_cfg = StructShiftConfig {
                    override_uniform_plddt: Some(100.0),
                    ..cfg.struct_shift_config.clone()
                };
                let apo_shifts = predict_structure_shifts_with_firewall(
                    &baseline_receptor,
                    &unified_fw,
                    &diff_shift_cfg,
                )
                .map_err(NmrError::parse)?;
                let complex_shifts = predict_structure_shifts_with_firewall(
                    &parsed.protein_model,
                    &unified_fw,
                    &diff_shift_cfg,
                )
                .map_err(NmrError::parse)?;
                for r in &apo_shifts.residues {
                    let h = r.backbone.get("H").map(|b| b.struct_ppm).unwrap_or(0.0);
                    let n = r.backbone.get("N").map(|b| b.struct_ppm).unwrap_or(0.0);
                    apo_shift_map.insert(r.seq_id, (h, n));
                }
                for r in &complex_shifts.residues {
                    if r.chain_id == receptor_chain {
                        let h = r.backbone.get("H").map(|b| b.struct_ppm).unwrap_or(0.0);
                        let n = r.backbone.get("N").map(|b| b.struct_ppm).unwrap_or(0.0);
                        comp_shift_map.insert(r.seq_id, (h, n));
                    }
                }
                complex_shifts.exclusion_list_sha256
            }
            CspScoringMode::BurialAndContactOnly => {
                let seq1: Vec<u8> = parsed
                    .protein_model
                    .residues
                    .iter()
                    .map(|r| r.aa1)
                    .collect();
                let fw = HomologFirewall::new(unified_fw)
                    .with_max_seq_identity(cfg.struct_shift_config.max_seq_identity);
                let (_, _, sha, _) = fw.apply(TorsionDb::embedded(), &seq1);
                sha
            }
        };

        // Non-receptor partner heavy atoms & non-protein H-bond acceptors/donors
        let partner_heavy: Vec<(Vec3, u8)> = parsed
            .all_atoms
            .iter()
            .filter(|a| a.z > 1 && a.chain_id != receptor_chain)
            .map(|a| (a.pos, a.z))
            .collect();
        let protein_res_keys: BTreeSet<(String, i32)> = parsed
            .protein_model
            .residues
            .iter()
            .map(|r| (r.chain_id.clone(), r.seq_id))
            .collect();
        let non_protein_polar_atoms: Vec<&ComplexAtomRecord> = parsed
            .all_atoms
            .iter()
            .filter(|a| {
                a.chain_id != receptor_chain
                    && (!protein_res_keys.contains(&(a.chain_id.clone(), a.seq_id)) || a.is_hetatm)
                    && matches!(a.z, 7 | 8 | 9 | 16)
            })
            .collect();

        // Shrake-Rupley rSASA on isolated receptor vs full complex pose `m`
        let sasa_iso = compute_shrake_rupley_rsasa(&isolated_from_pose, &[]);
        let sasa_comp = compute_shrake_rupley_rsasa(&isolated_from_pose, &partner_heavy);

        let mut residue_csp_points = Vec::new();
        let mut obs_vec = Vec::new();
        let mut calc_vec = Vec::new();
        let mut roc_samples: Vec<(bool, f64)> = Vec::new();
        let mut min_dist_by_seq_id: BTreeMap<i32, f64> = BTreeMap::new();

        const INV_R0_HB_CUBE: f64 = 1.0 / (2.60 * 2.60 * 2.60);

        for (r_idx, res) in isolated_from_pose.residues.iter().enumerate() {
            let (mut dh_calc, mut dn_calc) = if cfg.csp_scoring_mode
                == CspScoringMode::FullForwardShiftAndBurial
            {
                let (apo_h, apo_n) = apo_shift_map
                    .get(&res.seq_id)
                    .copied()
                    .unwrap_or((0.0, 0.0));
                let (cmp_h, cmp_n) = comp_shift_map
                    .get(&res.seq_id)
                    .copied()
                    .unwrap_or((apo_h, apo_n));
                let mut dh = cmp_h - apo_h;
                let mut dn = cmp_n - apo_n;

                // Add non-protein partner (`HETATM` small-molecule ligand / RNA / DNA) Haigh-Mallion ring currents
                // and Wagner-Pardi-Bax intermolecular H-bonds.
                if cfg.struct_shift_config.enable_ring_current
                    && !parsed.non_protein_aromatic_rings.is_empty()
                {
                    if let Some(h_pos) = res.h {
                        let (rc_h, _) = evaluate_haigh_mallion_ring_current(
                            &parsed.non_protein_aromatic_rings,
                            h_pos,
                            usize::MAX,
                            false,
                            1.0,
                        );
                        dh += rc_h.clamp(-1.80, 1.40);
                    }
                    if res.res_name != "PRO" {
                        let (rc_n, _) = evaluate_haigh_mallion_ring_current(
                            &parsed.non_protein_aromatic_rings,
                            res.n,
                            usize::MAX,
                            false,
                            0.65,
                        );
                        dn += rc_n.clamp(-1.50, 1.50);
                    }
                }

                if cfg.struct_shift_config.enable_hbond && !non_protein_polar_atoms.is_empty() {
                    if let Some(h_pos) = res.h {
                        let u_hn = v_unit(v_sub(res.n, h_pos));
                        let mut best_hb = 0.0_f64;
                        for pa in &non_protein_polar_atoms {
                            let d_ha = v_dist(h_pos, pa.pos);
                            if (1.55..=3.20).contains(&d_ha) {
                                let u_ha = v_unit(v_sub(pa.pos, h_pos));
                                let cos_ang = v_dot(u_hn, u_ha);
                                if cos_ang <= -0.34202014 {
                                    let excess =
                                        (1.0 / d_ha.max(1.65).powi(3) - INV_R0_HB_CUBE).max(0.0);
                                    best_hb = best_hb.max(excess * cos_ang * cos_ang);
                                }
                            }
                        }
                        dh += (4.15 * best_hb).min(1.35);
                        dn += (8.80 * best_hb).min(3.00);
                    }
                    // Preceding carbonyl `O_{i-1}` accepting H-bond from ligand polar donor polarizes peptide `(i-1, i)`
                    if r_idx > 0 && isolated_from_pose.residues[r_idx - 1].seq_id + 1 == res.seq_id
                    {
                        let prev_o = isolated_from_pose.residues[r_idx - 1].o;
                        let mut best_co_w = 0.0_f64;
                        for pa in &non_protein_polar_atoms {
                            if matches!(pa.z, 7 | 8) {
                                let d_oa = v_dist(prev_o, pa.pos);
                                if (2.40..=3.35).contains(&d_oa) {
                                    let w = ((3.35 - d_oa) / (3.35 - 2.65)).clamp(0.0, 1.0);
                                    best_co_w = best_co_w.max(w);
                                }
                            }
                        }
                        dh += 0.22 * best_co_w;
                        dn += 1.10 * best_co_w;
                    }
                }
                (dh, dn)
            } else {
                (0.0, 0.0)
            };
            // Keep `mut` warning-free when only read downstream
            let _ = (&mut dh_calc, &mut dn_calc);

            // Minimum heavy-atom distance from receptor residue `res` to any partner heavy atom
            let mut d_min = f64::INFINITY;
            for (aname, &apos) in &res.atoms {
                if infer_heavy_z_from_atom_name(aname).is_none() {
                    continue;
                }
                for &(ppos, _) in &partner_heavy {
                    let d = v_dist(apos, ppos);
                    if d < d_min {
                        d_min = d;
                    }
                }
            }
            min_dist_by_seq_id.insert(res.seq_id, d_min);

            let delta_rsasa = (sasa_iso[r_idx].rsasa - sasa_comp[r_idx].rsasa).max(0.0);
            // Compactly supported short-range desolvation/polarization envelope (`0.0` for `d_min >= 6.5 A`)
            let prox = if d_min < 6.5 {
                let u = ((6.5 - d_min.max(3.2)) / (6.5 - 3.2)).clamp(0.0, 1.0);
                u * u * (3.0 - 2.0 * u)
            } else {
                0.0
            };
            let burial_csp = 0.12 * delta_rsasa + 0.08 * prox;
            let alpha_n = amide_alpha_n(&res.res_name);
            let dir_csp = dh_calc.hypot(alpha_n * dn_calc);
            let csp_calc = match cfg.csp_scoring_mode {
                CspScoringMode::BurialAndContactOnly => burial_csp,
                CspScoringMode::FullForwardShiftAndBurial => dir_csp.hypot(burial_csp),
            };

            if let Some(obs) = obs_csp_by_seq.get(&res.seq_id) {
                // Exclude buried `AllostericCorePerturbation`, distal surface outliers, and
                // unresolved-in-apo shifts from direct surface interface correlation/ROC.
                let is_non_interface_csp = matches!(
                    obs.perturbation_class,
                    CspPerturbationClass::AllostericCorePerturbation
                        | CspPerturbationClass::DistalSurfaceOutlier
                        | CspPerturbationClass::UnresolvedInApo
                );
                let is_surf_hit = matches!(
                    obs.perturbation_class,
                    CspPerturbationClass::SurfacePocketContact
                        | CspPerturbationClass::UnfilteredNoApoStructure
                );
                let is_surf_residue = sasa_iso[r_idx].rsasa >= restraints.min_pocket_rsasa
                    || obs
                        .apo_rsasa
                        .map(|v| v >= restraints.min_pocket_rsasa)
                        .unwrap_or(true);

                if !is_non_interface_csp {
                    obs_vec.push(obs.csp_obs_ppm);
                    calc_vec.push(csp_calc);
                    if is_surf_hit || is_surf_residue {
                        // First + second-shell physical CSP proximity envelope (`8.0 A`, covering both
                        // direct `<= 5 A` VdW contact and `5-8 A` ring-current / electric-field shifts
                        // while vanishing `== 0` for detached / distal decoys beyond `8 A`).
                        let csp_shell_cut = (cfg.contact_cutoff_angstrom * 1.6).max(8.0);
                        let c_score = if d_min <= csp_shell_cut {
                            (csp_shell_cut - d_min) + 1e-6
                        } else {
                            0.0
                        };
                        roc_samples.push((is_surf_hit, c_score));
                    }
                }

                residue_csp_points.push(ResidueCspCalcPoint {
                    seq_id: res.seq_id,
                    res_name: res.res_name.clone(),
                    csp_obs_ppm: obs.csp_obs_ppm,
                    csp_calc_ppm: csp_calc,
                    delta_h_calc_ppm: dh_calc,
                    delta_n_calc_ppm: dn_calc,
                    delta_rsasa_burial: delta_rsasa,
                    min_partner_heavy_dist_a: d_min,
                    is_surface_2sigma_hit: is_surf_hit,
                });
            }
        }

        // Also support scoring when only `pocket_contacts` were populated without full `csp_observations`
        if roc_samples.is_empty() && !restraints.pocket_contacts.is_empty() {
            let hit_sids: BTreeSet<i32> = restraints
                .pocket_contacts
                .iter()
                .map(|p| p.seq_id)
                .collect();
            for (r_idx, res) in isolated_from_pose.residues.iter().enumerate() {
                if sasa_iso[r_idx].rsasa < restraints.min_pocket_rsasa
                    && !hit_sids.contains(&res.seq_id)
                {
                    continue;
                }
                let d_min = min_dist_by_seq_id
                    .get(&res.seq_id)
                    .copied()
                    .unwrap_or(f64::INFINITY);
                let csp_shell_cut = (cfg.contact_cutoff_angstrom * 1.6).max(8.0);
                let c_score = if d_min <= csp_shell_cut {
                    (csp_shell_cut - d_min) + 1e-6
                } else {
                    0.0
                };
                roc_samples.push((hit_sids.contains(&res.seq_id), c_score));
            }
        }

        // Evaluate Spearman rank correlation on the informative active union
        // `{i : obs_i >= threshold_1sigma || calc_i >= 0.015}` (falling back to all surface
        // observations when `< 5` active points exist) so random thermal spectrometer jitter
        // across unperturbed distal baseline residues (`obs < 1-sigma && calc == 0`) does not
        // inject artificial rank-tie dilution while still penalizing both missed pocket shifts
        // and false-positive decoy contacts!
        let noise_floor_obs = restraints
            .csp_thresholds
            .as_ref()
            .map(|t| t.threshold_1sigma_ppm)
            .unwrap_or(0.015);
        let mut active_obs = Vec::new();
        let mut active_calc = Vec::new();
        for (&o_v, &c_v) in obs_vec.iter().zip(calc_vec.iter()) {
            if o_v >= noise_floor_obs || c_v >= 0.015 {
                active_obs.push(o_v);
                active_calc.push(c_v);
            }
        }
        let csp_spearman_rho = if active_obs.len() >= 5 {
            Some(spearman_rank_correlation(&active_obs, &active_calc))
        } else if obs_vec.len() >= 3 {
            Some(spearman_rank_correlation(&obs_vec, &calc_vec))
        } else {
            None
        };
        let csp_contact_roc_auc = binary_roc_auc(&roc_samples);
        let csp_contact_p_value = mann_whitney_p_value(&roc_samples);

        // Mean excess distance beyond `pocket_max_distance_a` (`6.0 A`) across surface pocket hits
        let mean_pocket_excess_dist_a = if !restraints.pocket_contacts.is_empty() {
            let pocket_lim = restraints.pocket_max_distance_a.max(4.5);
            let sum_ex: f64 = restraints
                .pocket_contacts
                .iter()
                .map(|p| {
                    let dm = min_dist_by_seq_id.get(&p.seq_id).copied().unwrap_or(25.0);
                    (dm - pocket_lim).max(0.0)
                })
                .sum();
            sum_ex / (restraints.pocket_contacts.len() as f64)
        } else {
            0.0
        };
        let pocket_funnel_penalty = 0.12 * mean_pocket_excess_dist_a
            + 0.25 * det_ln(1.0 + (mean_pocket_excess_dist_a / 5.0).powi(2));

        // Use `rho.max(0.0)` and `(2*auc - 1).max(0.0)` for `S_CSP` so a detached ligand in bulk
        // solvent (`calc == 0` -> `rho = 0, auc = 0.5`) never outscores a closer surface-adjacent pose!
        let s_csp = match (csp_spearman_rho, csp_contact_roc_auc) {
            (Some(rho), Some(auc)) => {
                0.50 * rho.max(0.0) + 0.50 * (2.0 * auc - 1.0).max(0.0) - pocket_funnel_penalty
            }
            (Some(rho), None) => rho.max(0.0) - pocket_funnel_penalty,
            (None, Some(auc)) => (2.0 * auc - 1.0).max(0.0) - pocket_funnel_penalty,
            (None, None) => 0.0,
        };

        // 2. Intermolecular `NOE` & `PRE` `r^-6` sum-averaged violation energy
        let mut viol_energy_sum = 0.0_f64;
        let mut max_viol_a = 0.0_f64;
        let mut n_violated_dist = 0usize;
        let mut n_eval_dist = 0usize;

        for dr in &restraints.distance_restraints {
            let pts_1 = resolve_pose_atom_coords(&parsed, &dr.chain_1, dr.seq_id_1, &dr.atom_1);
            let pts_2 = resolve_pose_atom_coords(&parsed, &dr.chain_2, dr.seq_id_2, &dr.atom_2);
            if pts_1.is_empty() || pts_2.is_empty() {
                return Err(NmrError::restraint_mismatch(format!(
                    "pose '{}' missing atoms for distance restraint {}:{}:{} <-> {}:{}:{}",
                    pose_in.pose_id,
                    dr.chain_1,
                    dr.seq_id_1,
                    dr.atom_1,
                    dr.chain_2,
                    dr.seq_id_2,
                    dr.atom_2
                )));
            }
            let mut inv_r6_sum = 0.0_f64;
            for &a in &pts_1 {
                for &b in &pts_2 {
                    let r2 = v_dist_sq(a, b).max(0.25);
                    inv_r6_sum += 1.0 / (r2 * r2 * r2);
                }
            }
            let r_eff = 1.0 / sixth_root(inv_r6_sum.max(1e-30));
            let viol = if r_eff < dr.lower_a {
                dr.lower_a - r_eff
            } else if r_eff > dr.upper_a {
                r_eff - dr.upper_a
            } else {
                0.0
            };
            if viol > 0.0 {
                n_violated_dist += 1;
                if viol > max_viol_a {
                    max_viol_a = viol;
                }
            }
            let norm_v = viol / dr.sigma_a.max(0.10);
            viol_energy_sum += norm_v * norm_v;
            n_eval_dist += 1;
        }
        let noe_pre_violation_energy = if n_eval_dist > 0 {
            viol_energy_sum / (n_eval_dist as f64)
        } else {
            0.0
        };

        // 3. Complex multi-subunit Saupe SVD (`ByPeptidePlane` `20%` `RDC` `Q_free`)
        let (rdc_q_work_raw, rdc_q_free_raw) = evaluate_pose_rdc_q(
            &parsed,
            &restraints.rdc_media,
            cfg.rdc_free_fraction,
            cfg.rdc_free_seed,
        )?;
        let rdc_q_work = rdc_q_work_raw.is_finite().then_some(rdc_q_work_raw);
        let rdc_q_free = rdc_q_free_raw.is_finite().then_some(rdc_q_free_raw);
        let rdc_q_for_score = rdc_q_free.or(rdc_q_work).unwrap_or(0.0);

        // 4. 5-parameter `Delta_chi` `PCS` `Q`-factor
        let pcs_q_factor = evaluate_pose_pcs_q(&parsed, &restraints.pcs_tags)?;

        // Composite score `S_total(m)` (`section 6.2`)
        let total_score = cfg.w_csp * s_csp
            - cfg.w_noe_pre * noe_pre_violation_energy
            - cfg.w_rdc * rdc_q_for_score
            - cfg.w_pcs * pcs_q_factor.unwrap_or(0.0);

        // Decoy-cluster abstention guard flags (`section 6.2(4)`)
        let mut violation_flags = Vec::new();
        if n_eval_dist > 0 && max_viol_a > cfg.max_allowed_distance_violation_angstrom {
            violation_flags.push(PoseViolationFlag::MaxDistanceViolationExceeded);
        }
        if n_surface_csp_hits >= cfg.min_surface_hits_for_auc_guard {
            if let Some(auc) = csp_contact_roc_auc {
                if auc < cfg.min_csp_roc_auc {
                    violation_flags.push(PoseViolationFlag::LowCspContactEnrichment);
                }
            }
        }
        if let Some(qf) = rdc_q_free {
            if qf > cfg.max_allowed_rdc_q_free {
                violation_flags.push(PoseViolationFlag::HighRdcQFree);
            }
        }
        let passes_nmr_guards = violation_flags.is_empty();

        let irmsd_to_reference_angstrom = opt_ref_pose
            .as_ref()
            .map(|ref_atoms| {
                compute_interface_rmsd_from_atoms(&parsed.all_atoms, ref_atoms, receptor_chain)
            })
            .transpose()?;

        entries.push(PoseScoreEntry {
            rank: 0,
            pose_id: pose_in.pose_id.clone(),
            source_path: pose_in.source_path.clone(),
            total_score,
            passes_nmr_guards,
            violation_flags,
            s_csp,
            csp_spearman_rho,
            csp_contact_roc_auc,
            csp_contact_p_value,
            mean_pocket_excess_dist_a,
            noe_pre_violation_energy,
            max_distance_violation_angstrom: max_viol_a,
            n_violated_distance_restraints: n_violated_dist,
            rdc_q_work,
            rdc_q_free,
            pcs_q_factor,
            irmsd_to_reference_angstrom,
            cofold_confidence: pose_in.cofold_confidence.clone(),
            exclusion_list_sha256,
            residue_csp_points,
        });
    }

    // Rank poses deterministically: admissible poses ahead of guard-violating poses, then descending `total_score`, then `pose_id`
    entries.sort_by(|a, b| {
        b.passes_nmr_guards
            .cmp(&a.passes_nmr_guards)
            .then_with(|| b.total_score.total_cmp(&a.total_score))
            .then_with(|| a.pose_id.cmp(&b.pose_id))
    });
    for (idx, e) in entries.iter_mut().enumerate() {
        e.rank = idx + 1;
    }

    let n_admissible = entries.iter().filter(|e| e.passes_nmr_guards).count();
    let verdict = if n_admissible > 0 {
        PoseVerdict::RankedConclusive
    } else {
        PoseVerdict::InconclusiveOrAllViolated
    };
    let top_pose_id = if verdict == PoseVerdict::RankedConclusive {
        entries.first().map(|e| e.pose_id.clone())
    } else {
        None
    };

    let funnel_spearman_rho_vs_neg_irmsd = if entries.len() >= 3
        && entries
            .iter()
            .all(|e| e.irmsd_to_reference_angstrom.is_some())
    {
        let neg_irmsd: Vec<f64> = entries
            .iter()
            .map(|e| -e.irmsd_to_reference_angstrom.unwrap_or(0.0))
            .collect();
        let scores: Vec<f64> = entries.iter().map(|e| e.total_score).collect();
        Some(spearman_rank_correlation(&neg_irmsd, &scores))
    } else {
        None
    };

    Ok(PoseScoreReport {
        job_name: restraints.job_name.clone(),
        verdict,
        top_pose_id,
        n_poses_evaluated: entries.len(),
        n_admissible_poses: n_admissible,
        n_surface_csp_hits,
        n_distance_restraints: restraints.distance_restraints.len(),
        n_rdc_media: restraints.rdc_media.len(),
        n_pcs_tags: restraints.pcs_tags.len(),
        csp_scoring_mode: cfg.csp_scoring_mode,
        funnel_spearman_rho_vs_neg_irmsd,
        poses: entries,
    })
}

/// Format a [`PoseScoreReport`] as deterministic TSV.
pub fn format_pose_score_tsv(report: &PoseScoreReport) -> String {
    let verdict_str = match report.verdict {
        PoseVerdict::RankedConclusive => "RankedConclusive",
        PoseVerdict::InconclusiveOrAllViolated => "InconclusiveOrAllViolated",
    };
    let top_str = report.top_pose_id.as_deref().unwrap_or("ABSTAINED");
    let mut out = format!(
        "# job_name={}\tverdict={}\ttop_pose={}\tn_poses={}\tn_admissible={}\n",
        report.job_name, verdict_str, top_str, report.n_poses_evaluated, report.n_admissible_poses
    );
    out.push_str(
        "rank\tpose_id\ttotal_score\tpasses_guards\ts_csp\tcsp_spearman_rho\tcsp_roc_auc\tnoe_pre_viol_energy\tmax_viol_a\trdc_q_free\tpcs_q\tirmsd_a\tviolation_flags\n",
    );
    let fmt_opt = |v: Option<f64>, prec: usize| match v {
        Some(x) if x.is_finite() => format!("{x:.prec$}"),
        _ => "NA".to_string(),
    };
    for p in &report.poses {
        let flags = if p.violation_flags.is_empty() {
            "none".to_string()
        } else {
            p.violation_flags
                .iter()
                .map(|f| match f {
                    PoseViolationFlag::MaxDistanceViolationExceeded => "max_dist_viol",
                    PoseViolationFlag::LowCspContactEnrichment => "low_csp_auc",
                    PoseViolationFlag::HighRdcQFree => "high_rdc_qfree",
                })
                .collect::<Vec<_>>()
                .join(",")
        };
        out.push_str(&format!(
            "{}\t{}\t{:.4}\t{}\t{:.4}\t{}\t{}\t{:.4}\t{:.3}\t{}\t{}\t{}\t{}\n",
            p.rank,
            p.pose_id,
            p.total_score,
            p.passes_nmr_guards,
            p.s_csp,
            fmt_opt(p.csp_spearman_rho, 4),
            fmt_opt(p.csp_contact_roc_auc, 4),
            p.noe_pre_violation_energy,
            p.max_distance_violation_angstrom,
            fmt_opt(p.rdc_q_free, 4),
            fmt_opt(p.pcs_q_factor, 4),
            fmt_opt(p.irmsd_to_reference_angstrom, 3),
            flags,
        ));
    }
    out
}

/// Format a [`PoseScoreReport`] as pretty JSON.
pub fn format_pose_score_json(report: &PoseScoreReport) -> Result<String, NmrError> {
    let mut s = serde_json::to_string_pretty(report).map_err(|e| {
        NmrError::invalid_input(format!("failed to serialize PoseScoreReport: {e}"))
    })?;
    s.push('\n');
    Ok(s)
}
