//! Clean-room 3D protein coordinate (`.pdb` / `.cif`) to backbone (`H^N`, `15N`, `13CA`, `13CB`,
//! `13C'`, `1HA`) and `ILVMAT` sidechain methyl (`1H^m`, `13C^m`) chemical-shift forward predictor
//! (`T21.2`, `docs/specs/21-alphafold-integrative-nmr.md` section 2).
//!
//! # Provenance & Clean-Room Statement
//! Original clean-room implementation derived strictly from peer-reviewed biomolecular NMR
//! physics and fitted exclusively on-the-fly (`< 15 ms`) to FERRIC's public-domain (`CC0 1.0`)
//! `data/torsion_db/torsion_corpus.tsv` (`129` chains, `13,647` residues) behind the mandatory
//! 4-layer `HomologFirewall` (`max_seq_identity <= 0.25`, `TM-score < 0.50`), with zero external
//! parameter tables, neural weights, or restricted software artifacts:
//! - Shen, Y. & Bax, A. (*J. Biomol. NMR* **2010**, 48, 13-22) -- additive decomposition of
//!   protein backbone and `13CB` chemical shifts into local `(phi, psi, chi1)` conformation,
//!   hydrogen-bond geometry, conjugated aromatic ring-current fields, and peptide-group
//!   anisotropy.
//! - Haigh, C. W. & Mallion, R. B. (*Mol. Phys.* **1972**, 24, 955-970; *Prog. Nucl. Magn. Reson.
//!   Spectrosc.* **1979**, 13, 303-344) & Case, D. A. (*J. Biomol. NMR* **1995**, 6, 341-346) --
//!   signed triangular-projection conjugated aromatic ring-current shielding tensors across
//!   `Phe`, `Tyr`, `His`, and `Trp` 5/6-membered pi-systems.
//! - Empirical H-bond shift term: inverse-cube excess distance $\Delta\delta^{\text{HB}} \propto \left(r_{\text{HO}}^{-3} - r_0^{-3}\right)_+ \cos^2\theta_{\text{NHO}}$
//!   (an original empirical functional form inspired by the inverse-cube distance scaling in
//!   Wagner, G., Pardi, A. & Bax, A., *J. Am. Chem. Soc.* **1983**, 105, 5948–5949, and Pardi, A.,
//!   Wagner, G. & Wuthrich, K., *Eur. J. Biochem.* **1983**, 137, 445–454; coefficients are
//!   empirical FERRIC parameters rather than the original Wagner–Pardi–Bax regression).
//! - McConnell, H. M. (*J. Chem. Phys.* **1957**, 27, 226-229); Buckingham, A. D. (*Can. J.
//!   Chem.* **1960**, 38, 300-307); Sitkoff, D. & Case, D. A. (*Prog. Nucl. Magn. Reson.
//!   Spectrosc.* **1998**, 32, 165-190) (peptide carbonyl magnetic susceptibility anisotropy and
//!   bond-parallel electric-field shielding `Delta_delta_EF = -A_parallel * E_parallel`); and
//!   MacKerell, A. D. et al. (*J. Phys. Chem. B* **1998**, 102, 3586–3616) (`CHARMM22` peptide
//!   backbone partial charges `C': +0.51, O: -0.51, N: -0.47, H: +0.31 e`).
//! - Sahakyan, A. B., Vranken, W. F., Cavalli, A. & Vendruscolo, M. (*Angew. Chem. Int. Ed.*
//!   **2011**, 50, 9620-9623) -- sidechain methyl (`Ile CD1/HD1*`, `Leu CD1/HD1* & CD2/HD2*`,
//!   `Val CG1/HG1* & CG2/HG2*`, `Met CE/HE*`, `Ala CB/HB*`, `Thr CG2/HG2*`) `gamma`-gauche
//!   rotamer shielding and 3-proton rotamer-averaged aromatic ring-current shifts.
//! - Tamiola, K., Acar, B. & Mulder, F. A. A. (*J. Am. Chem. Soc.* **2010**, 132, 18000-18003)
//!   (`ncIDP` random-coil baseline); Kjaergaard, M., Brander, S. & Poulsen, F. M. (*J. Biomol.
//!   NMR* **2011**, 49, 139-149) (linear temperature coefficients); Platzer, G., Okon, M. &
//!   McIntosh, L. P. (*J. Biomol. NMR* **2014**, 60, 109-129) (Henderson-Hasselbalch pH shifts).
//! - Kabsch, W. & Sander, C. (*Biopolymers* **1983**, 22, 2577-2637) -- backbone partial-charge
//!   electrostatic hydrogen-bond energy (`E_HB < -0.5 kcal/mol`).
//! - Jumper, J. et al. (*Nature* **2021**, 596, 583-589) & Varadi, M. et al. (*Nucleic Acids
//!   Res.* **2022**, 50, D439-D444; **2024**, 52, D368-D375) -- `pLDDT` local confidence and
//!   ModelArchive `_ma_qa_metric_local` mmCIF records.
//! - Parsons, J. et al. (*J. Comput. Chem.* **2005**, 26, 1063-1068) (`NeRF`) & Engh, R. A. &
//!   Huber, R. (*Acta Crystallogr. A* **1991**, 47, 392-400) (ideal covalent geometry).

use std::collections::BTreeMap;
use std::f64::consts::PI;
use std::fs;
use std::path::Path;
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use crate::dihedrals::{periodic_angle_abs_diff_deg, random_coil_shifts};
use crate::methyl_trosy::MethylGroupKind;
use crate::nef::{
    infer_element_and_isotope, normalize_nef_atom_name, write_nef_str, NefAssignedShift,
    NefDocument, NefMolecularSystem, NefShiftList,
};
use crate::nmrstar::Reference;
use crate::noesy::{
    calc_dihedral_deg, reconstruct_ca_substituents, v_add, v_cross, v_dist, v_dist_sq, v_dot,
    v_norm, v_scale, v_sub, v_unit, PdbChain, Vec3,
};
use crate::priors::three_to_one;
use crate::structure::{
    det_acos, det_exp, det_sin_cos, nerf_place_atom, ANGLE_CA_C_O_DEG, ANGLE_C_N_H_DEG,
    ANGLE_N_CA_C_DEG, ANGLE_N_CA_C_GLY_DEG, BOND_CA_HA, BOND_C_O, BOND_N_H,
};
use crate::torsion_db::{
    ExcludedTemplateRecord, HomologFirewall, HomologFirewallQuery, TorsionCorpusChain, TorsionDb,
    DEFAULT_FIREWALL_MAX_SEQ_IDENTITY,
};

/// Maximum coordinate file byte size accepted by `.pdb`/`.cif` parsers (`64 MiB` DoS guard).
pub const MAX_STRUCTURE_FILE_BYTES: usize = 64 * 1024 * 1024;

/// Maximum number of polymer residues accepted per coordinate model (`32,768` DoS guard).
pub const MAX_STRUCTURE_RESIDUES: usize = 32_768;

/// Haigh-Mallion proton ring-current constant `B_HM,H` (`5.455 ppm * A`; Case *J. Biomol. NMR* 1995).
pub const HAIGH_MALLION_PROTON_CONSTANT: f64 = 5.455;

/// Outer distance cutoff (`12.0 A`) for conjugated aromatic ring-current evaluation.
pub const RING_CURRENT_CUTOFF_ANGSTROM: f64 = 12.0;

/// Short-range clash damping floor (`1.80 A`) on vertex-to-spin distances `r_{up}` in Haigh-Mallion sums.
pub const RING_CURRENT_CLASH_FLOOR_ANGSTROM: f64 = 1.80;

/// Six canonical backbone shift nuclei predicted by `nmr-core::struct_shifts`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum BackboneShiftAtom {
    H = 0,
    N = 1,
    CA = 2,
    CB = 3,
    C = 4,
    HA = 5,
}

impl BackboneShiftAtom {
    pub const ALL: [Self; 6] = [Self::H, Self::N, Self::CA, Self::CB, Self::C, Self::HA];

    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::H => "H",
            Self::N => "N",
            Self::CA => "CA",
            Self::CB => "CB",
            Self::C => "C",
            Self::HA => "HA",
        }
    }

    /// Baseline folded-state predictive standard deviation `sigma_{struct,a}` (ppm).
    #[inline]
    pub const fn struct_sigma_ppm(self) -> f64 {
        match self {
            Self::H => 0.45,
            Self::N => 2.45,
            Self::CA => 1.05,
            Self::CB => 1.15,
            Self::C => 1.08,
            Self::HA => 0.28,
        }
    }

    /// Baseline disordered / low-`pLDDT` total predictive standard deviation `sigma_{low_plddt,a}` (ppm).
    /// Must be `>= struct_sigma_ppm()` (`21.W1:P2-1`) so predictive uncertainty NEVER shrinks as
    /// `pLDDT` falls, reflecting combined conformational disorder and coil-model uncertainty.
    #[inline]
    pub const fn coil_sigma_ppm(self) -> f64 {
        match self {
            Self::H => 0.62,
            Self::N => 3.10,
            Self::CA => 1.55,
            Self::CB => 1.65,
            Self::C => 1.50,
            Self::HA => 0.36,
        }
    }
}

/// Experimental vs computational provenance of an input 3D coordinate model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum StructCoordinateProvenance {
    /// Crystallographic, solution NMR, cryo-EM, or diffraction model (`B_iso` is thermal B-factor, NOT `pLDDT`).
    Experimental,
    /// Predicted model carrying `pLDDT` in `_ma_qa_metric_local` or computational `B_iso`.
    Computational,
    /// Unmarked coordinate source without explicit `pLDDT` metadata.
    Unknown,
}

impl StructCoordinateProvenance {
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Experimental => "Experimental",
            Self::Computational => "Computational",
            Self::Unknown => "Unknown",
        }
    }
}

/// Five structural residue classes used for hierarchical trigonometric Fourier torsion surfaces.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StructuralResidueClass {
    General = 0,
    Gly = 1,
    Pro = 2,
    PrePro = 3,
    BetaBranched = 4,
}

impl StructuralResidueClass {
    pub const ALL: [Self; 5] = [
        Self::General,
        Self::Gly,
        Self::Pro,
        Self::PrePro,
        Self::BetaBranched,
    ];

    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    #[inline]
    pub fn classify(res_name: &str, next_is_pro: bool) -> Self {
        let up = res_name.trim();
        if up.eq_ignore_ascii_case("GLY") {
            Self::Gly
        } else if up.eq_ignore_ascii_case("PRO") {
            Self::Pro
        } else if next_is_pro {
            Self::PrePro
        } else if up.eq_ignore_ascii_case("ILE")
            || up.eq_ignore_ascii_case("VAL")
            || up.eq_ignore_ascii_case("THR")
        {
            Self::BetaBranched
        } else {
            Self::General
        }
    }
}

/// Planar conjugated aromatic ring (`Phe`, `Tyr`, `His`, `Trp` 5-ring, `Trp` 6-ring) for Haigh-Mallion shielding.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConjugatedAromaticRing {
    pub chain_id: String,
    pub seq_id: i32,
    pub residue_index: usize,
    pub res_name: String,
    pub ring_label: String,
    pub vertices: Vec<Vec3>,
    pub center: Vec3,
    pub normal: Vec3,
    /// Ring-current intensity factor `i_R` relative to benzene (`Case 1995`).
    pub intensity_factor: f64,
}

/// Parsed and deterministically proton/sidechain-completed residue coordinates.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StructResidueCoords {
    pub chain_id: String,
    pub seq_id: i32,
    pub res_name: String,
    pub aa1: u8,
    pub n: Vec3,
    pub h: Option<Vec3>,
    pub ca: Vec3,
    pub ha: Vec3,
    pub ha2: Option<Vec3>,
    pub ha3: Option<Vec3>,
    pub c: Vec3,
    pub o: Vec3,
    pub cb: Option<Vec3>,
    pub atoms: BTreeMap<String, Vec3>,
    pub phi_deg: Option<f64>,
    pub psi_deg: Option<f64>,
    pub omega_deg: Option<f64>,
    pub chi1_deg: Option<f64>,
    pub chi2_deg: Option<f64>,
    pub cys_oxidized: bool,
    pub plddt: Option<f64>,
}

/// 3D protein structure model ready for forward chemical-shift prediction.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StructureModel {
    pub provenance: StructCoordinateProvenance,
    pub residues: Vec<StructResidueCoords>,
    pub aromatic_rings: Vec<ConjugatedAromaticRing>,
    pub diagnostics: Vec<String>,
}

impl StructureModel {
    /// Fail-closed check verifying that `self` has positive evidence of being a predicted structure
    /// model carrying valid per-residue `pLDDT` values, or that `treat_as_model` was explicitly
    /// enabled (`21.W1:P0-2`, `21.W2:P1-3`).
    pub fn require_predicted_model(&self, treat_as_model: bool) -> Result<(), String> {
        require_predicted_structure_model(self, treat_as_model)
    }
}

/// Fail-closed provenance gate for [`StructureModel`] (`21.W1:P0-2`, `21.W2:P1-3`).
pub fn require_predicted_structure_model(
    model: &StructureModel,
    treat_as_model: bool,
) -> Result<(), String> {
    if model.residues.is_empty() {
        return Err("predicted StructureModel check failed: zero residues".to_string());
    }
    let has_all_plddt = model.residues.iter().all(|r| r.plddt.is_some());
    if !has_all_plddt {
        return Err(format!(
            "predicted StructureModel check failed (provenance={:?}, diagnostics={:?}): per-residue pLDDT is unavailable (experimental/headerless/constant-B structure); pass --treat-as-model / --force-bfactor-plddt only if B-factors genuinely encode pLDDT",
            model.provenance, model.diagnostics
        ));
    }
    if model.provenance == StructCoordinateProvenance::Computational || treat_as_model {
        return Ok(());
    }
    Err(format!(
        "predicted StructureModel provenance check failed (provenance={:?}, diagnostics={:?}): positive ModelArchive/AFDB evidence is required unless --treat-as-model / --force-bfactor-plddt is explicitly supplied",
        model.provenance, model.diagnostics
    ))
}

/// Configuration for 3D structure-to-chemical-shift forward prediction.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StructShiftConfig {
    /// Sample pH (`default = 6.5`, used in Platzer Henderson-Hasselbalch coil titration).
    pub sample_ph: f64,
    /// Sample temperature in Kelvin (`default = 298.15`, used in Kjaergaard linear coil coefficients).
    pub temperature_kelvin: f64,
    /// Predict `ILVMAT` sidechain methyl (`1H^m`, `13C^m`) shifts alongside the 6 backbone nuclei.
    pub include_methyl_ilvmat: bool,
    /// L2 ridge regularization penalty `lambda_ridge` (`default = 2.5`) on Fourier torsion surfaces.
    pub ridge_lambda: f64,
    /// Maximum sequence identity (`default = 0.25`) enforced by `Layer 3` of `HomologFirewall`.
    pub max_seq_identity: f64,
    /// Enable Haigh-Mallion aromatic ring-current shielding (`true` by default; `false` for ablation tests).
    pub enable_ring_current: bool,
    /// Enable Wagner-Pardi-Bax inverse-cube hydrogen-bond shifts (`true` by default).
    pub enable_hbond: bool,
    /// Enable McConnell peptide carbonyl anisotropy & Buckingham electric-field shifts (`true` by default).
    pub enable_peptide_ef: bool,
    /// Force interpretation of PDB `tempFactor` / mmCIF `B_iso_or_equiv` as `pLDDT` even on experimental headers.
    pub force_bfactor_plddt: bool,
    /// Optional uniform `pLDDT` override in `[0.0, 100.0]` applied across all residues.
    pub override_uniform_plddt: Option<f64>,
}

impl Default for StructShiftConfig {
    fn default() -> Self {
        Self {
            sample_ph: 6.5,
            temperature_kelvin: 298.15,
            include_methyl_ilvmat: false,
            ridge_lambda: 2.5,
            max_seq_identity: DEFAULT_FIREWALL_MAX_SEQ_IDENTITY,
            enable_ring_current: true,
            enable_hbond: true,
            enable_peptide_ef: true,
            force_bfactor_plddt: false,
            override_uniform_plddt: None,
        }
    }
}

impl StructShiftConfig {
    pub fn with_ph(mut self, ph: f64) -> Self {
        self.sample_ph = ph;
        self
    }

    pub fn with_temperature_kelvin(mut self, temp_k: f64) -> Self {
        self.temperature_kelvin = temp_k;
        self
    }

    pub fn with_methyl_ilvmat(mut self, enabled: bool) -> Self {
        self.include_methyl_ilvmat = enabled;
        self
    }

    pub fn with_ring_current(mut self, enabled: bool) -> Self {
        self.enable_ring_current = enabled;
        self
    }

    pub fn with_uniform_plddt(mut self, plddt: f64) -> Self {
        self.override_uniform_plddt = Some(plddt.clamp(0.0, 100.0));
        self
    }
}

/// Additive component breakdown for a single predicted backbone shift.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResidueAtomShiftBreakdown {
    pub atom: String,
    /// Final `pLDDT`-blended predicted shift `hat_delta_{i,a}` (ppm).
    pub predicted_ppm: f64,
    /// Final `pLDDT`-blended predictive uncertainty `hat_sigma_{i,a}` (ppm).
    pub predicted_sigma_ppm: f64,
    /// Unblended 3D structural shift `delta^{struct}_{i,a}` (ppm).
    pub struct_ppm: f64,
    /// Sequence-, pH-, and temperature-corrected `ncIDP` random-coil shift `delta^{RC}_{i,a}` (ppm).
    pub rc_ppm: f64,
    /// CC0-firewalled Fourier torsion & local secondary-structure term `Delta_delta^{tor}_{i,a}` (ppm).
    pub delta_torsion_ppm: f64,
    /// Wagner-Pardi-Bax direct hydrogen-bond shift `Delta_delta^{HB}_{i,a}` (ppm).
    pub delta_hbond_ppm: f64,
    /// Haigh-Mallion conjugated aromatic ring-current shift `Delta_delta^{RC-arom}_{i,a}` (ppm).
    pub delta_ring_current_ppm: f64,
    /// McConnell peptide carbonyl anisotropy + Buckingham bond-parallel electric-field shift (ppm).
    pub delta_peptide_ef_ppm: f64,
}

/// Predicted `(13C^m, 1H^m)` chemical shifts and physical breakdown for one `ILVMAT` methyl site.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MethylShiftPrediction {
    pub kind: MethylGroupKind,
    pub carbon_atom: String,
    pub proton_atom: String,
    pub c_ppm: f64,
    pub c_sigma_ppm: f64,
    pub c_struct_ppm: f64,
    pub c_rc_ppm: f64,
    pub c_gamma_gauche_ppm: f64,
    pub c_ring_current_ppm: f64,
    pub h_ppm: f64,
    pub h_sigma_ppm: f64,
    pub h_struct_ppm: f64,
    pub h_rc_ppm: f64,
    pub h_ring_current_ppm: f64,
    pub h_peptide_ef_ppm: f64,
    pub nearest_aromatic_dist_angstrom: Option<f64>,
}

/// Per-residue predicted backbone and optional `ILVMAT` methyl shifts.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResidueShiftPrediction {
    pub chain_id: String,
    pub seq_id: i32,
    pub res_name: String,
    pub plddt: Option<f64>,
    pub w_af: f64,
    pub phi_deg: Option<f64>,
    pub psi_deg: Option<f64>,
    pub chi1_deg: Option<f64>,
    pub cys_oxidized: bool,
    pub nearest_arom_h_dist_angstrom: Option<f64>,
    pub nearest_arom_ha_dist_angstrom: Option<f64>,
    pub backbone: BTreeMap<String, ResidueAtomShiftBreakdown>,
    pub methyls: Vec<MethylShiftPrediction>,
}

/// Complete 3D structure-to-chemical-shift prediction report with `HomologFirewall` audit digest.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StructShiftPredictionReport {
    pub residues: Vec<ResidueShiftPrediction>,
    pub sample_ph: f64,
    pub temperature_kelvin: f64,
    pub include_methyl_ilvmat: bool,
    pub provenance: StructCoordinateProvenance,
    pub retained_template_count: usize,
    pub excluded_template_ids: Vec<String>,
    pub excluded_details: Vec<ExcludedTemplateRecord>,
    pub exclusion_list_sha256: String,
    pub max_retained_seq_identity: f64,
    pub corpus_sha256: String,
    /// Number of database template chains shorter than 8 CA atoms skipped by Layer 4 (`TM-score`)
    /// without disabling Layer 4 for the rest of the corpus (`21.W1:P1-4`).
    #[serde(default)]
    pub l4_short_templates_skipped: usize,
}

// ============================================================================
// 1. Smooth `pLDDT`-Sigmoid Confidence Weighting (`w_AF`)
// ============================================================================

/// Smooth `pLDDT`-sigmoid structural confidence weight `w_AF(pLDDT) in [0.0, 1.0)`
/// (`docs/specs/21-alphafold-integrative-nmr.md` section 2.1(6)):
/// - Equals `1 / (1 + exp(-(pLDDT - 70.0) / 8.0))` across `pLDDT >= 15.0` (`0.5` at `70.0`,
///   `0.07586 < 0.08` at `50.0`),
/// - Smoothly C^2-tapers to `0.0` on `pLDDT in [0.0, 15.0]` so `|hat_delta - delta_RC| < 1e-6`
///   as `pLDDT -> 0`.
#[inline]
pub fn af_plddt_confidence_weight(plddt: f64) -> f64 {
    if !plddt.is_finite() || plddt <= 0.0 {
        return 0.0;
    }
    let p = plddt.min(100.0);
    let sig = 1.0 / (1.0 + det_exp(-(p - 70.0) / 8.0));
    if p >= 15.0 {
        sig
    } else {
        let t = p / 15.0;
        let smooth = t * t * t * (10.0 + t * (-15.0 + 6.0 * t));
        smooth * sig
    }
}

// ============================================================================
// 2. Hostile-Input-Safe `.pdb` & `.cif` (mmCIF) Parsers + Proton/Sidechain Completion
// ============================================================================

#[inline]
fn normalize_std_res_name(raw: &str) -> Option<&'static str> {
    match raw.trim().to_ascii_uppercase().as_str() {
        "ALA" => Some("ALA"),
        "ARG" => Some("ARG"),
        "ASN" => Some("ASN"),
        "ASP" | "ASH" => Some("ASP"),
        "CYS" | "CYX" | "CYM" => Some("CYS"),
        "GLN" => Some("GLN"),
        "GLU" | "GLH" => Some("GLU"),
        "GLY" => Some("GLY"),
        "HIS" | "HID" | "HIE" | "HIP" | "HSD" | "HSE" | "HSP" => Some("HIS"),
        "ILE" => Some("ILE"),
        "LEU" => Some("LEU"),
        "LYS" | "LYN" => Some("LYS"),
        "MET" | "MSE" => Some("MET"),
        "PHE" => Some("PHE"),
        "PRO" => Some("PRO"),
        "SER" => Some("SER"),
        "THR" => Some("THR"),
        "TRP" => Some("TRP"),
        "TYR" => Some("TYR"),
        "VAL" => Some("VAL"),
        _ => None,
    }
}

#[inline]
fn normalize_atom_name(res_std: &str, raw_atom: &str) -> String {
    let up = raw_atom.trim().to_ascii_uppercase();
    match up.as_str() {
        "HN" | "H1" | "HT1" => "H".to_string(),
        "OT1" | "O1" => "O".to_string(),
        "SE" if res_std == "MET" => "SD".to_string(),
        "CD" if res_std == "ILE" => "CD1".to_string(),
        "1HA" | "HA1" if res_std == "GLY" => "HA2".to_string(),
        "2HA" if res_std == "GLY" => "HA3".to_string(),
        "1HB" | "HB1" if res_std == "ALA" => "HB1".to_string(),
        "2HB" if res_std == "ALA" => "HB2".to_string(),
        "3HB" if res_std == "ALA" => "HB3".to_string(),
        _ => up,
    }
}

#[derive(Default)]
struct RawParsedResidue {
    chain_id: String,
    seq_id: i32,
    res_name: String,
    atoms: BTreeMap<String, Vec3>,
    ca_bfactor: Option<f64>,
    bfactor_sum: f64,
    bfactor_count: usize,
    explicit_plddt: Option<f64>,
}

impl StructureModel {
    /// Parse a `.pdb` or `.cif`/`.mmcif` coordinate file into a proton-completed [`StructureModel`].
    pub fn parse_file(path: impl AsRef<Path>, force_bfactor_plddt: bool) -> Result<Self, String> {
        let p = path.as_ref();
        let meta = fs::metadata(p)
            .map_err(|e| format!("failed to stat coordinate file '{}': {e}", p.display()))?;
        if meta.len() > (MAX_STRUCTURE_FILE_BYTES as u64) {
            return Err(format!(
                "coordinate file '{}' ({} bytes) exceeds 64 MiB limit ({MAX_STRUCTURE_FILE_BYTES})",
                p.display(),
                meta.len()
            ));
        }
        let bytes = fs::read(p)
            .map_err(|e| format!("failed to read coordinate file '{}': {e}", p.display()))?;
        if bytes.len() > MAX_STRUCTURE_FILE_BYTES {
            return Err(format!(
                "coordinate file '{}' exceeds 64 MiB limit",
                p.display()
            ));
        }
        let text = std::str::from_utf8(&bytes)
            .map_err(|e| format!("coordinate file '{}' is not valid UTF-8: {e}", p.display()))?;
        let ext = p
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_ascii_lowercase());
        Self::parse_str(text, ext.as_deref(), force_bfactor_plddt)
    }

    /// Parse `.pdb` or `.cif`/`.mmcif` text into a [`StructureModel`].
    pub fn parse_str(
        text: &str,
        format_hint: Option<&str>,
        force_bfactor_plddt: bool,
    ) -> Result<Self, String> {
        if text.len() > MAX_STRUCTURE_FILE_BYTES {
            return Err(format!(
                "coordinate input ({} bytes) exceeds 64 MiB limit ({MAX_STRUCTURE_FILE_BYTES})",
                text.len()
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
            Self::parse_cif_str(text, force_bfactor_plddt)
        } else {
            Self::parse_pdb_str(text, force_bfactor_plddt)
        }
    }

    /// Parse first-model PDB text into a [`StructureModel`], rejecting `NaN`/`Inf` coordinates.
    pub fn parse_pdb_str(pdb_text: &str, force_bfactor_plddt: bool) -> Result<Self, String> {
        if pdb_text.len() > MAX_STRUCTURE_FILE_BYTES {
            return Err("PDB input exceeds 64 MiB limit".to_string());
        }
        let mut provenance = StructCoordinateProvenance::Unknown;
        let mut raw_residues: Vec<RawParsedResidue> = Vec::new();
        let mut idx_map: BTreeMap<(String, i32), usize> = BTreeMap::new();
        let mut explicit_disulfides: Vec<((String, i32), (String, i32))> = Vec::new();
        let mut diagnostics = Vec::new();

        for (line_idx, line) in pdb_text.lines().enumerate() {
            let line_no = line_idx + 1;
            if line.starts_with("ENDMDL") {
                break;
            }
            if line.starts_with("EXPDTA") {
                let up = line.to_ascii_uppercase();
                if up.contains("X-RAY")
                    || up.contains("NMR")
                    || up.contains("ELECTRON")
                    || up.contains("NEUTRON")
                    || up.contains("DIFFRACTION")
                    || up.contains("MICROSCOPY")
                    || up.contains("TOMOGRAPHY")
                {
                    provenance = StructCoordinateProvenance::Experimental;
                }
                continue;
            }
            if line.starts_with("HEADER")
                || line.starts_with("TITLE")
                || line.starts_with("COMPND")
                || line.starts_with("REMARK")
            {
                // Never infer computational provenance from free-text PDB REMARK/TITLE/COMPND/HEADER
                // lines (`21.W1:P0-2`); only check explicit experimental markers.
                let up = line.to_ascii_uppercase();
                if (up.contains("X-RAY")
                    || up.contains("CRYSTAL STRUCTURE")
                    || up.contains("CRYSTALLOGRAPHIC")
                    || up.contains("SOLUTION NMR")
                    || up.contains("DIPOLAR COUPLINGS"))
                    && provenance == StructCoordinateProvenance::Unknown
                {
                    provenance = StructCoordinateProvenance::Experimental;
                }
                continue;
            }
            if line.starts_with("CRYST1") && line.len() >= 33 {
                if let (Ok(a), Ok(b), Ok(c)) = (
                    line[6..15].trim().parse::<f64>(),
                    line[15..24].trim().parse::<f64>(),
                    line[24..33].trim().parse::<f64>(),
                ) {
                    // Non-trivial crystallographic unit cell (excluding dummy 1x1x1 P1 boxes)
                    if a > 2.5
                        && b > 2.5
                        && c > 2.5
                        && provenance == StructCoordinateProvenance::Unknown
                    {
                        provenance = StructCoordinateProvenance::Experimental;
                    }
                }
                continue;
            }
            if line.starts_with("SSBOND") && line.len() >= 35 {
                let c1 = line.get(15..16).unwrap_or("A").trim();
                let s1 = line.get(17..21).and_then(|s| s.trim().parse::<i32>().ok());
                let c2 = line.get(29..30).unwrap_or("A").trim();
                let s2 = line.get(31..35).and_then(|s| s.trim().parse::<i32>().ok());
                if let (Some(seq1), Some(seq2)) = (s1, s2) {
                    let ch1 = if c1.is_empty() { "A" } else { c1 };
                    let ch2 = if c2.is_empty() { "A" } else { c2 };
                    explicit_disulfides.push(((ch1.to_string(), seq1), (ch2.to_string(), seq2)));
                }
                continue;
            }
            if !line.starts_with("ATOM  ") && !(line.starts_with("HETATM") && line.contains("MSE"))
            {
                continue;
            }
            if !line.is_ascii() {
                return Err(format!(
                    "PDB line {line_no}: non-ASCII characters in ATOM record"
                ));
            }
            if line.len() < 54 {
                return Err(format!(
                    "PDB line {line_no}: truncated ATOM record (< 54 columns)"
                ));
            }
            let alt = line.as_bytes()[16];
            if alt != b' ' && alt != b'A' && alt != b'1' {
                continue;
            }
            let i_code = line.as_bytes()[26];
            if i_code != b' ' {
                continue;
            }
            let Some(res_std) = normalize_std_res_name(&line[17..20]) else {
                continue;
            };
            let atom_name = normalize_atom_name(res_std, &line[12..16]);
            let chain_raw = line[21..22].trim();
            let chain_id = if chain_raw.is_empty() {
                "A".to_string()
            } else {
                chain_raw.to_string()
            };
            let seq_id: i32 = line[22..26]
                .trim()
                .parse()
                .map_err(|e| format!("PDB line {line_no}: invalid residue number: {e}"))?;
            let x: f64 = line[30..38]
                .trim()
                .parse()
                .map_err(|e| format!("PDB line {line_no}: invalid X coordinate: {e}"))?;
            let y: f64 = line[38..46]
                .trim()
                .parse()
                .map_err(|e| format!("PDB line {line_no}: invalid Y coordinate: {e}"))?;
            let z: f64 = line[46..54]
                .trim()
                .parse()
                .map_err(|e| format!("PDB line {line_no}: invalid Z coordinate: {e}"))?;
            if !x.is_finite() || !y.is_finite() || !z.is_finite() {
                return Err(format!("PDB line {line_no}: non-finite atomic coordinate"));
            }
            let b_iso = if line.len() >= 66 {
                let b_str = line[60..66].trim();
                if b_str.is_empty() {
                    None
                } else {
                    let b: f64 = b_str.parse().map_err(|e| {
                        format!("PDB line {line_no}: invalid tempFactor '{b_str}': {e}")
                    })?;
                    if !b.is_finite() {
                        return Err(format!("PDB line {line_no}: non-finite tempFactor"));
                    }
                    Some(b)
                }
            } else {
                None
            };

            let key = (chain_id.clone(), seq_id);
            let r_idx = match idx_map.get(&key) {
                Some(&idx) => idx,
                None => {
                    if raw_residues.len() >= MAX_STRUCTURE_RESIDUES {
                        return Err(format!(
                            "PDB exceeds maximum residue limit ({MAX_STRUCTURE_RESIDUES})"
                        ));
                    }
                    let idx = raw_residues.len();
                    raw_residues.push(RawParsedResidue {
                        chain_id,
                        seq_id,
                        res_name: res_std.to_string(),
                        ..Default::default()
                    });
                    idx_map.insert(key, idx);
                    idx
                }
            };
            let rr = &mut raw_residues[r_idx];
            rr.atoms.entry(atom_name.clone()).or_insert([x, y, z]);
            if let Some(b) = b_iso {
                if atom_name == "CA" && rr.ca_bfactor.is_none() {
                    rr.ca_bfactor = Some(b);
                }
                if matches!(atom_name.as_str(), "N" | "CA" | "C" | "O") {
                    rr.bfactor_sum += b;
                    rr.bfactor_count += 1;
                }
            }
        }

        finalize_parsed_structure(
            raw_residues,
            provenance,
            force_bfactor_plddt,
            &explicit_disulfides,
            &mut diagnostics,
        )
    }

    /// Parse ModelArchive / PDBx mmCIF text (`.cif`), extracting `_ma_qa_metric_local` `pLDDT`
    /// with guarded `_atom_site.B_iso_or_equiv` fallback.
    pub fn parse_cif_str(cif_text: &str, force_bfactor_plddt: bool) -> Result<Self, String> {
        if cif_text.len() > MAX_STRUCTURE_FILE_BYTES {
            return Err("mmCIF input exceeds 64 MiB limit".to_string());
        }
        let tokens = tokenize_cif(cif_text)?;
        if tokens.is_empty() {
            return Err("empty mmCIF input".to_string());
        }

        let mut provenance = StructCoordinateProvenance::Unknown;
        // Shared `pLDDT` provenance rule (`21F:B1`): `af_concordance::PlddtProvenanceEvidence`.
        let mut prov_ev = crate::af_concordance::PlddtProvenanceEvidence::default();
        // Scalar `_ma_qa_metric` (id, name, type, mode).
        let mut scalar_ma: [Option<String>; 4] = [None, None, None, None];
        // Raw `_ma_qa_metric_local` rows: (label_asym_id, label_seq_id, metric_id, value).
        let mut local_rows: Vec<(String, i32, Option<String>, f64)> = Vec::new();
        let mut explicit_disulfides: Vec<((String, i32), (String, i32))> = Vec::new();
        let mut raw_residues: Vec<RawParsedResidue> = Vec::new();
        let mut idx_map: BTreeMap<(String, i32), usize> = BTreeMap::new();
        let mut label_to_res_idx: BTreeMap<(String, i32), usize> = BTreeMap::new();
        let mut diagnostics = Vec::new();

        // First scan key-value pairs and loops
        let mut pos = 0usize;
        let n_tok = tokens.len();
        while pos < n_tok {
            let (ref tok, quoted) = tokens[pos];
            if !quoted && tok.eq_ignore_ascii_case("loop_") {
                pos += 1;
                let mut cols = Vec::new();
                while pos < n_tok && !tokens[pos].1 && tokens[pos].0.starts_with('_') {
                    cols.push(tokens[pos].0.clone());
                    pos += 1;
                }
                if cols.is_empty() {
                    return Err("malformed mmCIF loop_ with zero column headers".to_string());
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
                    return Err(format!(
                        "truncated mmCIF loop ({}) with {} values not divisible by {} columns",
                        cols[0],
                        vals.len(),
                        ncols
                    ));
                }

                let prefix = cols[0].split('.').next().unwrap_or("").to_ascii_lowercase();
                if prefix == "_exptl" {
                    if let Some(m_col) = find_cif_col(&cols, "_exptl.method") {
                        for row in vals.chunks_exact(ncols) {
                            let up = row[m_col].to_ascii_uppercase();
                            if up.contains("X-RAY")
                                || up.contains("NMR")
                                || up.contains("ELECTRON")
                                || up.contains("DIFFRACTION")
                                || up.contains("MICROSCOPY")
                            {
                                provenance = StructCoordinateProvenance::Experimental;
                            }
                        }
                    }
                } else if prefix == "_software" {
                    if let Some(n_col) = find_cif_col(&cols, "_software.name") {
                        if vals.chunks_exact(ncols).any(|row| {
                            crate::af_concordance::is_recognized_predictor_software(row[n_col])
                        }) {
                            prov_ev.recognized_predictor_header = true;
                        }
                    }
                } else if prefix == "_ma_qa_metric" {
                    let id_col = find_cif_col(&cols, "_ma_qa_metric.id");
                    let name_col = find_cif_col(&cols, "_ma_qa_metric.name");
                    let type_col = find_cif_col(&cols, "_ma_qa_metric.type");
                    let mode_col = find_cif_col(&cols, "_ma_qa_metric.mode");
                    for row in vals.chunks_exact(ncols) {
                        let get = |c: Option<usize>| c.map(|c| row[c]).unwrap_or("");
                        prov_ev.push_metric(
                            get(id_col),
                            get(name_col),
                            get(type_col),
                            get(mode_col),
                        );
                    }
                } else if prefix == "_ma_qa_metric_local" {
                    let asym_col = find_cif_col(&cols, "_ma_qa_metric_local.label_asym_id")
                        .ok_or("mmCIF _ma_qa_metric_local missing label_asym_id")?;
                    let seq_col = find_cif_col(&cols, "_ma_qa_metric_local.label_seq_id")
                        .ok_or("mmCIF _ma_qa_metric_local missing label_seq_id")?;
                    let val_col = find_cif_col(&cols, "_ma_qa_metric_local.metric_value")
                        .ok_or("mmCIF _ma_qa_metric_local missing metric_value")?;
                    let mid_col = find_cif_col(&cols, "_ma_qa_metric_local.metric_id");
                    {
                        for row in vals.chunks_exact(ncols) {
                            let mid = mid_col.map(|mc| row[mc].to_string());
                            prov_ev.local_row_metric_ids.push(mid.clone());
                            let seq_id: i32 = row[seq_col].parse().map_err(|e| {
                                format!(
                                    "invalid _ma_qa_metric_local.label_seq_id '{}': {e}",
                                    row[seq_col]
                                )
                            })?;
                            let mv: f64 = row[val_col].parse().map_err(|e| {
                                format!(
                                    "invalid _ma_qa_metric_local.metric_value '{}': {e}",
                                    row[val_col]
                                )
                            })?;
                            if !mv.is_finite() {
                                return Err("non-finite pLDDT in _ma_qa_metric_local".to_string());
                            }
                            local_rows.push((row[asym_col].to_string(), seq_id, mid, mv));
                        }
                    }
                } else if prefix == "_struct_conn" {
                    let type_col = find_cif_col(&cols, "_struct_conn.conn_type_id");
                    let c1_col = find_cif_col(&cols, "_struct_conn.ptnr1_auth_asym_id")
                        .or_else(|| find_cif_col(&cols, "_struct_conn.ptnr1_label_asym_id"));
                    let s1_col = find_cif_col(&cols, "_struct_conn.ptnr1_auth_seq_id")
                        .or_else(|| find_cif_col(&cols, "_struct_conn.ptnr1_label_seq_id"));
                    let c2_col = find_cif_col(&cols, "_struct_conn.ptnr2_auth_asym_id")
                        .or_else(|| find_cif_col(&cols, "_struct_conn.ptnr2_label_asym_id"));
                    let s2_col = find_cif_col(&cols, "_struct_conn.ptnr2_auth_seq_id")
                        .or_else(|| find_cif_col(&cols, "_struct_conn.ptnr2_label_seq_id"));
                    if let (Some(tc), Some(c1), Some(s1), Some(c2), Some(s2)) =
                        (type_col, c1_col, s1_col, c2_col, s2_col)
                    {
                        for row in vals.chunks_exact(ncols) {
                            if row[tc].to_ascii_lowercase().starts_with("disulf") {
                                if let (Ok(q1), Ok(q2)) = (row[s1].parse(), row[s2].parse()) {
                                    explicit_disulfides.push((
                                        (row[c1].to_string(), q1),
                                        (row[c2].to_string(), q2),
                                    ));
                                }
                            }
                        }
                    }
                } else if prefix == "_atom_site" {
                    let grp_col = find_cif_col(&cols, "_atom_site.group_PDB");
                    let x_col = find_cif_col(&cols, "_atom_site.Cartn_x")
                        .ok_or("mmCIF _atom_site missing Cartn_x")?;
                    let y_col = find_cif_col(&cols, "_atom_site.Cartn_y")
                        .ok_or("mmCIF _atom_site missing Cartn_y")?;
                    let z_col = find_cif_col(&cols, "_atom_site.Cartn_z")
                        .ok_or("mmCIF _atom_site missing Cartn_z")?;
                    let comp_col = find_cif_col(&cols, "_atom_site.auth_comp_id")
                        .or_else(|| find_cif_col(&cols, "_atom_site.label_comp_id"))
                        .ok_or("mmCIF _atom_site missing comp_id")?;
                    let atom_col = find_cif_col(&cols, "_atom_site.label_atom_id")
                        .or_else(|| find_cif_col(&cols, "_atom_site.auth_atom_id"))
                        .ok_or("mmCIF _atom_site missing atom_id")?;
                    let auth_asym_col = find_cif_col(&cols, "_atom_site.auth_asym_id");
                    let label_asym_col = find_cif_col(&cols, "_atom_site.label_asym_id");
                    let auth_seq_col = find_cif_col(&cols, "_atom_site.auth_seq_id");
                    let label_seq_col = find_cif_col(&cols, "_atom_site.label_seq_id");
                    let alt_col = find_cif_col(&cols, "_atom_site.label_alt_id");
                    let ins_col = find_cif_col(&cols, "_atom_site.pdbx_PDB_ins_code");
                    let model_col = find_cif_col(&cols, "_atom_site.pdbx_PDB_model_num");
                    let biso_col = find_cif_col(&cols, "_atom_site.B_iso_or_equiv");

                    let mut first_model: Option<&str> = None;
                    for row in vals.chunks_exact(ncols) {
                        if let Some(mc) = model_col {
                            let m_str = row[mc];
                            match first_model {
                                None => first_model = Some(m_str),
                                Some(fm) if fm != m_str => continue,
                                _ => {}
                            }
                        }
                        if let Some(gc) = grp_col {
                            let g = row[gc];
                            if !g.eq_ignore_ascii_case("ATOM")
                                && !(g.eq_ignore_ascii_case("HETATM")
                                    && row[comp_col].eq_ignore_ascii_case("MSE"))
                            {
                                continue;
                            }
                        }
                        if let Some(ac) = alt_col {
                            let a = row[ac];
                            if a != "." && a != "?" && a != "A" && a != "1" {
                                continue;
                            }
                        }
                        if let Some(ic) = ins_col {
                            let ins = row[ic];
                            if !ins.is_empty() && ins != "." && ins != "?" {
                                continue;
                            }
                        }
                        let Some(res_std) = normalize_std_res_name(row[comp_col]) else {
                            continue;
                        };
                        let atom_name = normalize_atom_name(res_std, row[atom_col]);
                        let chain_id = auth_asym_col
                            .or(label_asym_col)
                            .map(|c| row[c])
                            .filter(|s| !s.is_empty() && *s != "." && *s != "?")
                            .unwrap_or("A")
                            .to_string();
                        let label_asym = label_asym_col
                            .map(|c| row[c])
                            .unwrap_or(chain_id.as_str())
                            .to_string();
                        let seq_str = auth_seq_col
                            .and_then(|c| {
                                let v = row[c];
                                (v != "." && v != "?").then_some(v)
                            })
                            .or_else(|| label_seq_col.map(|c| row[c]))
                            .ok_or("mmCIF _atom_site missing auth_seq_id/label_seq_id")?;
                        let seq_id: i32 = seq_str.parse().map_err(|e| {
                            format!("mmCIF _atom_site invalid seq_id '{seq_str}': {e}")
                        })?;
                        let label_seq: Option<i32> =
                            label_seq_col.and_then(|c| row[c].parse::<i32>().ok());

                        let x: f64 = row[x_col].parse().map_err(|e| {
                            format!("mmCIF _atom_site invalid Cartn_x '{}': {e}", row[x_col])
                        })?;
                        let y: f64 = row[y_col].parse().map_err(|e| {
                            format!("mmCIF _atom_site invalid Cartn_y '{}': {e}", row[y_col])
                        })?;
                        let z: f64 = row[z_col].parse().map_err(|e| {
                            format!("mmCIF _atom_site invalid Cartn_z '{}': {e}", row[z_col])
                        })?;
                        if !x.is_finite() || !y.is_finite() || !z.is_finite() {
                            return Err(
                                "mmCIF _atom_site contains non-finite coordinate".to_string()
                            );
                        }
                        let b_iso = if let Some(bc) = biso_col {
                            let bs = row[bc];
                            if bs == "." || bs == "?" || bs.is_empty() {
                                None
                            } else {
                                let b: f64 = bs.parse().map_err(|e| {
                                    format!("mmCIF _atom_site invalid B_iso '{bs}': {e}")
                                })?;
                                if !b.is_finite() {
                                    return Err(
                                        "mmCIF _atom_site contains non-finite B_iso_or_equiv"
                                            .to_string(),
                                    );
                                }
                                Some(b)
                            }
                        } else {
                            None
                        };

                        let key = (chain_id.clone(), seq_id);
                        let r_idx = match idx_map.get(&key) {
                            Some(&idx) => idx,
                            None => {
                                if raw_residues.len() >= MAX_STRUCTURE_RESIDUES {
                                    return Err(format!(
                                        "mmCIF exceeds maximum residue limit ({MAX_STRUCTURE_RESIDUES})"
                                    ));
                                }
                                let idx = raw_residues.len();
                                raw_residues.push(RawParsedResidue {
                                    chain_id,
                                    seq_id,
                                    res_name: res_std.to_string(),
                                    ..Default::default()
                                });
                                idx_map.insert(key, idx);
                                idx
                            }
                        };
                        if let Some(lseq) = label_seq {
                            label_to_res_idx.insert((label_asym, lseq), r_idx);
                        }
                        let rr = &mut raw_residues[r_idx];
                        rr.atoms.entry(atom_name.clone()).or_insert([x, y, z]);
                        if let Some(b) = b_iso {
                            if atom_name == "CA" && rr.ca_bfactor.is_none() {
                                rr.ca_bfactor = Some(b);
                            }
                            if matches!(atom_name.as_str(), "N" | "CA" | "C" | "O") {
                                rr.bfactor_sum += b;
                                rr.bfactor_count += 1;
                            }
                        }
                    }
                }
            } else if !quoted && tok.starts_with("data_") {
                if crate::af_concordance::is_afdb_entry_id(&tok[5..]) {
                    prov_ev.recognized_predictor_header = true;
                }
                pos += 1;
            } else if !quoted && tok.starts_with('_') {
                if pos + 1 < n_tok {
                    let key_low = tok.to_ascii_lowercase();
                    let val_raw = tokens[pos + 1].0.trim();
                    let val_up = val_raw.to_ascii_uppercase();
                    if key_low == "_exptl.method" {
                        if val_up.contains("X-RAY")
                            || val_up.contains("NMR")
                            || val_up.contains("ELECTRON")
                            || val_up.contains("DIFFRACTION")
                            || val_up.contains("MICROSCOPY")
                        {
                            provenance = StructCoordinateProvenance::Experimental;
                        }
                    } else if key_low == "_entry.id" {
                        if crate::af_concordance::is_afdb_entry_id(val_raw) {
                            prov_ev.recognized_predictor_header = true;
                        }
                    } else if key_low == "_ma_qa_metric.id" {
                        scalar_ma[0] = Some(val_raw.to_string());
                    } else if key_low == "_ma_qa_metric.name" {
                        scalar_ma[1] = Some(val_raw.to_string());
                    } else if key_low == "_ma_qa_metric.type" {
                        scalar_ma[2] = Some(val_raw.to_string());
                    } else if key_low == "_ma_qa_metric.mode" {
                        scalar_ma[3] = Some(val_raw.to_string());
                    } else if key_low == "_software.name"
                        && crate::af_concordance::is_recognized_predictor_software(val_raw)
                    {
                        prov_ev.recognized_predictor_header = true;
                    }
                    pos += 2;
                } else {
                    pos += 1;
                }
            } else {
                pos += 1;
            }
        }

        if scalar_ma.iter().any(Option::is_some) {
            let [id, nm, tp, md] = scalar_ma.map(Option::unwrap_or_default);
            prov_ev.push_metric(&id, &nm, &tp, &md);
        }
        // Single shared pLDDT provenance rule (`21F:B1`): verified local pLDDT rows, or a
        // recognized predictor header with no declared QA schema. A declared pLDDT metric without
        // matching `_ma_qa_metric_local` rows fails closed (no B-factor fallback / rescaling).
        let verdict = prov_ev.verdict();
        if provenance != StructCoordinateProvenance::Experimental {
            provenance = if verdict == crate::af_concordance::PlddtProvenanceVerdict::NotPlddt {
                StructCoordinateProvenance::Unknown
            } else {
                StructCoordinateProvenance::Computational
            };
        }
        for label in prov_ev.ignored_metric_labels() {
            diagnostics.push(format!("ignored non-pLDDT / unverified QA metric: {label}"));
        }

        // Attach explicit `_ma_qa_metric_local` pLDDTs (verified local pLDDT metric ids only)
        if verdict == crate::af_concordance::PlddtProvenanceVerdict::LocalPlddtRows {
            let mut local_plddt_by_label: BTreeMap<(String, i32), f64> = BTreeMap::new();
            for (lasym, lseq, mid, mv) in local_rows {
                if prov_ev.local_row_is_plddt(mid.as_deref()) {
                    local_plddt_by_label.insert((lasym, lseq), mv);
                }
            }
            for ((lasym, lseq), mv) in &local_plddt_by_label {
                if let Some(&r_idx) = label_to_res_idx
                    .get(&(lasym.clone(), *lseq))
                    .or_else(|| idx_map.get(&(lasym.clone(), *lseq)))
                {
                    raw_residues[r_idx].explicit_plddt = Some(*mv);
                }
            }
        }

        finalize_parsed_structure(
            raw_residues,
            provenance,
            force_bfactor_plddt,
            &explicit_disulfides,
            &mut diagnostics,
        )
    }

    /// Construct a [`StructureModel`] from an already-parsed [`PdbChain`].
    pub fn from_pdb_chain(chain: &PdbChain) -> Result<Self, String> {
        if chain.residues.is_empty() {
            return Err("empty PdbChain".to_string());
        }
        let mut raw_residues = Vec::with_capacity(chain.residues.len());
        for r in &chain.residues {
            let mut atoms = BTreeMap::new();
            atoms.insert("N".to_string(), r.n);
            if let Some(h) = r.h {
                atoms.insert("H".to_string(), h);
            }
            atoms.insert("CA".to_string(), r.ca);
            atoms.insert("HA".to_string(), r.ha);
            if let Some(ha2) = r.ha2 {
                atoms.insert("HA2".to_string(), ha2);
            }
            if let Some(ha3) = r.ha3 {
                atoms.insert("HA3".to_string(), ha3);
            }
            atoms.insert("C".to_string(), r.c);
            atoms.insert("O".to_string(), r.o);
            if let Some(cb) = r.cb {
                atoms.insert("CB".to_string(), cb);
            }
            raw_residues.push(RawParsedResidue {
                chain_id: r.chain_id.to_string(),
                seq_id: r.seq_id,
                res_name: r.res_name.to_ascii_uppercase(),
                atoms,
                ..Default::default()
            });
        }
        let mut diagnostics = Vec::new();
        finalize_parsed_structure(
            raw_residues,
            StructCoordinateProvenance::Experimental,
            false,
            &[],
            &mut diagnostics,
        )
    }

    /// Deterministically reconstruct a complete 3D [`StructureModel`] (peptide planes `N, H, C, O`,
    /// chiral `CB, HA`, and `chi1`-oriented sidechain heavy atoms & aromatic rings) pinned to an
    /// exact `C_alpha` trace `[CA_0, ..., CA_{L-1}]` and `(phi_i, psi_i, chi1_i)` torsion angles.
    pub fn from_ca_and_torsions(
        chain_id: &str,
        seq_ids: &[i32],
        res_names: &[String],
        ca_coords: &[[f64; 3]],
        phi_deg: &[Option<f64>],
        psi_deg: &[Option<f64>],
        chi1_deg: &[Option<f64>],
    ) -> Result<Self, String> {
        let l = ca_coords.len();
        if l == 0 {
            return Err("cannot reconstruct StructureModel from empty CA trace".to_string());
        }
        if seq_ids.len() != l
            || res_names.len() != l
            || phi_deg.len() != l
            || psi_deg.len() != l
            || chi1_deg.len() != l
        {
            return Err(
                "mismatched slice lengths in StructureModel::from_ca_and_torsions".to_string(),
            );
        }
        for c in ca_coords {
            if !c[0].is_finite() || !c[1].is_finite() || !c[2].is_finite() {
                return Err("non-finite CA coordinate in from_ca_and_torsions".to_string());
            }
        }

        // Reconstruct peptide planes `(C_i, O_i, N_{i+1}, H_{i+1})` between consecutive CA_i and CA_{i+1}
        let mut n_pos: Vec<Vec3> = vec![[0.0; 3]; l];
        let mut h_pos: Vec<Option<Vec3>> = vec![None; l];
        let mut c_pos: Vec<Vec3> = vec![[0.0; 3]; l];
        let mut o_pos: Vec<Vec3> = vec![[0.0; 3]; l];

        // Place virtual N_0 so that when peptide plane 0 is optimized, phi_0 / angle(N_0-CA_0-C_0) is ideal
        if l == 1 {
            n_pos[0] = v_add(ca_coords[0], [-1.458, 0.0, 0.0]);
            c_pos[0] = v_add(ca_coords[0], [0.500, 1.440, 0.0]);
            o_pos[0] = v_add(c_pos[0], [0.0, 1.231, 0.0]);
        } else {
            let ex0 = v_unit(v_sub(ca_coords[1], ca_coords[0]));
            let ref_up = if ex0[2].abs() < 0.85 {
                [0.0, 0.0, 1.0]
            } else {
                [0.0, 1.0, 0.0]
            };
            let ey0 = v_unit(v_cross(ref_up, ex0));
            // N_0 backward from CA_0 at 111.2 deg relative to first peptide direction
            n_pos[0] = v_add(
                ca_coords[0],
                v_add(v_scale(ex0, -0.527), v_scale(ey0, 1.360)),
            );

            for i in 0..(l - 1) {
                let ca_a = ca_coords[i];
                let ca_b = ca_coords[i + 1];
                let d_ca = v_dist(ca_a, ca_b);
                let intact = (2.55..=4.45).contains(&d_ca) && (seq_ids[i + 1] == seq_ids[i] + 1);
                if !intact {
                    // Chain gap: terminate C_i/O_i locally and re-seed N_{i+1}
                    let u_back = v_unit(v_sub(n_pos[i], ca_a));
                    let perp = ortho_unit(u_back);
                    c_pos[i] = v_add(
                        ca_a,
                        v_add(
                            v_scale(u_back, -0.362 * 1.525),
                            v_scale(perp, 0.932 * 1.525),
                        ),
                    );
                    o_pos[i] = nerf_place_atom(
                        n_pos[i],
                        ca_a,
                        c_pos[i],
                        BOND_C_O,
                        ANGLE_CA_C_O_DEG.to_radians(),
                        psi_deg[i].unwrap_or(140.0).to_radians() + PI,
                    );
                    let ex_next = if i + 2 < l {
                        v_unit(v_sub(ca_coords[i + 2], ca_b))
                    } else {
                        [1.0, 0.0, 0.0]
                    };
                    let ey_next = ortho_unit(ex_next);
                    n_pos[i + 1] = v_add(
                        ca_b,
                        v_add(v_scale(ex_next, -0.527), v_scale(ey_next, 1.360)),
                    );
                    continue;
                }

                let ex = v_unit(v_sub(ca_b, ca_a));
                let u0 = ortho_unit(ex);
                let v0 = v_cross(ex, u0);
                let is_cis = d_ca < 3.25;
                let next_is_pro = res_names[i + 1].eq_ignore_ascii_case("PRO");
                let target_ang = if res_names[i].eq_ignore_ascii_case("GLY") {
                    ANGLE_N_CA_C_GLY_DEG
                } else {
                    ANGLE_N_CA_C_DEG
                };

                let eval_plane = |gamma: f64| -> (Vec3, Vec3, Vec3, Option<Vec3>) {
                    let (sg, cg) = det_sin_cos(gamma);
                    let ey = v_add(v_scale(u0, cg), v_scale(v0, sg));
                    if is_cis {
                        // Cis peptide plane (CA_i and CA_{i+1} syn across C-N, d_CA ~ 2.95 A)
                        let c_p = v_add(ca_a, v_add(v_scale(ex, 0.270 * d_ca), v_scale(ey, 1.290)));
                        let n_p = v_add(ca_a, v_add(v_scale(ex, 0.720 * d_ca), v_scale(ey, 1.230)));
                        let o_p = v_add(c_p, v_add(v_scale(ex, -0.600), v_scale(ey, 1.075)));
                        let h_p = if next_is_pro {
                            None
                        } else {
                            Some(v_add(n_p, v_add(v_scale(ex, 0.520), v_scale(ey, 0.865))))
                        };
                        (c_p, o_p, n_p, h_p)
                    } else {
                        // Exact Engh-Huber trans peptide plane (r_CAC=1.525, r_CN=1.329, r_NCA=1.458)
                        let c_p = v_add(
                            ca_a,
                            v_add(v_scale(ex, 0.3755 * d_ca), v_scale(ey, -0.5340)),
                        );
                        let o_p = v_add(c_p, v_add(v_scale(ex, 0.2201), v_scale(ey, -1.2112)));
                        let n_p =
                            v_add(ca_a, v_add(v_scale(ex, 0.6298 * d_ca), v_scale(ey, 0.3774)));
                        let h_p = if next_is_pro {
                            None
                        } else {
                            Some(v_add(n_p, v_add(v_scale(ex, -0.2460), v_scale(ey, 0.9795))))
                        };
                        (c_p, o_p, n_p, h_p)
                    }
                };

                let score_gamma = |gamma: f64| -> f64 {
                    let (c_p, _, n_next, _) = eval_plane(gamma);
                    let mut loss = 0.0;
                    if i > 0 {
                        let u_n = v_unit(v_sub(n_pos[i], ca_a));
                        let u_c = v_unit(v_sub(c_p, ca_a));
                        let ang_deg = det_acos(v_dot(u_n, u_c)).to_degrees();
                        let da = ang_deg - target_ang;
                        loss += 0.25 * da * da;
                        if let Some(psi_target) = psi_deg[i] {
                            let psi_cur = calc_dihedral_deg(n_pos[i], ca_a, c_p, n_next);
                            let dpsi = periodic_angle_abs_diff_deg(psi_cur, psi_target);
                            loss += dpsi * dpsi;
                        }
                    } else if i + 2 < l {
                        // At i == 0, orient plane 0 using virtual CA_0-CA_1-CA_2 and phi_1 if available
                        let ex1 = v_unit(v_sub(ca_coords[2], ca_b));
                        let u_n1 = v_unit(v_sub(c_p, ca_b));
                        let ang1 = det_acos(v_dot(u_n1, ex1)).to_degrees();
                        loss += 0.05 * (ang1 - 105.0) * (ang1 - 105.0);
                    }
                    loss
                };

                let mut best_g = 0.0;
                let mut best_loss = f64::INFINITY;
                for step in 0..72 {
                    let g = -PI + (step as f64) * (2.0 * PI / 72.0);
                    let sc = score_gamma(g);
                    if sc < best_loss {
                        best_loss = sc;
                        best_g = g;
                    }
                }
                // Local refinement around best_g
                let mut step_rad = (2.5_f64).to_radians();
                for _ in 0..8 {
                    for dir in [-1.0, 1.0] {
                        let cand = best_g + dir * step_rad;
                        let sc = score_gamma(cand);
                        if sc < best_loss {
                            best_loss = sc;
                            best_g = cand;
                        }
                    }
                    step_rad *= 0.5;
                }

                let (c_p, o_p, n_next, h_next) = eval_plane(best_g);
                c_pos[i] = c_p;
                o_pos[i] = o_p;
                n_pos[i + 1] = n_next;
                h_pos[i + 1] = h_next;

                // Refine N_0 once C_0 and N_1 are known if psi_0 is provided
                if i == 0 {
                    if let Some(psi0) = psi_deg[0] {
                        n_pos[0] = nerf_place_atom(
                            n_next,
                            c_p,
                            ca_a,
                            1.458,
                            target_ang.to_radians(),
                            psi0.to_radians(),
                        );
                    }
                }
            }

            // Place C_{L-1} and O_{L-1} at C-terminus using phi_{L-1} (if known)
            let last = l - 1;
            let ang_last = if res_names[last].eq_ignore_ascii_case("GLY") {
                ANGLE_N_CA_C_GLY_DEG.to_radians()
            } else {
                ANGLE_N_CA_C_DEG.to_radians()
            };
            let phi_last = phi_deg[last].unwrap_or(-65.0).to_radians();
            c_pos[last] = nerf_place_atom(
                c_pos[last - 1],
                n_pos[last],
                ca_coords[last],
                1.525,
                ang_last,
                phi_last,
            );
            o_pos[last] = nerf_place_atom(
                n_pos[last],
                ca_coords[last],
                c_pos[last],
                BOND_C_O,
                ANGLE_CA_C_O_DEG.to_radians(),
                PI,
            );
        }

        let mut raw_residues = Vec::with_capacity(l);
        for i in 0..l {
            let r_std = normalize_std_res_name(&res_names[i]).unwrap_or("ALA");
            let mut atoms = BTreeMap::new();
            atoms.insert("N".to_string(), n_pos[i]);
            if let Some(h) = h_pos[i] {
                atoms.insert("H".to_string(), h);
            }
            atoms.insert("CA".to_string(), ca_coords[i]);
            atoms.insert("C".to_string(), c_pos[i]);
            atoms.insert("O".to_string(), o_pos[i]);

            let (cb_opt, ha, ha2, ha3, _) =
                reconstruct_ca_substituents(r_std, n_pos[i], ca_coords[i], c_pos[i]);
            atoms.insert("HA".to_string(), ha);
            if let Some(h2) = ha2 {
                atoms.insert("HA2".to_string(), h2);
            }
            if let Some(h3) = ha3 {
                atoms.insert("HA3".to_string(), h3);
            }
            if let Some(cb) = cb_opt {
                atoms.insert("CB".to_string(), cb);
                reconstruct_sidechain_from_chi1(
                    &mut atoms,
                    r_std,
                    n_pos[i],
                    ca_coords[i],
                    cb,
                    chi1_deg[i],
                );
            }

            raw_residues.push(RawParsedResidue {
                chain_id: chain_id.to_string(),
                seq_id: seq_ids[i],
                res_name: r_std.to_string(),
                atoms,
                ..Default::default()
            });
        }

        let mut diagnostics = Vec::new();
        let mut model = finalize_parsed_structure(
            raw_residues,
            StructCoordinateProvenance::Experimental,
            false,
            &[],
            &mut diagnostics,
        )?;
        // Preserve exact supplied `(phi, psi, chi1)` when provided so numerical round-trip is exact
        for i in 0..model.residues.len().min(l) {
            if phi_deg[i].is_some() {
                model.residues[i].phi_deg = phi_deg[i];
            }
            if psi_deg[i].is_some() {
                model.residues[i].psi_deg = psi_deg[i];
            }
            if chi1_deg[i].is_some() {
                model.residues[i].chi1_deg = chi1_deg[i];
            }
        }
        Ok(model)
    }
}

#[inline]
fn ortho_unit(u: Vec3) -> Vec3 {
    let ref_v = if u[2].abs() < 0.85 {
        [0.0, 0.0, 1.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    v_unit(v_cross(ref_v, u))
}

fn find_cif_col(cols: &[String], target: &str) -> Option<usize> {
    cols.iter().position(|c| c.eq_ignore_ascii_case(target))
}

fn tokenize_cif(text: &str) -> Result<Vec<(String, bool)>, String> {
    let mut out = Vec::new();
    let mut lines = text.lines().enumerate().peekable();
    while let Some((line_idx, line)) = lines.next() {
        let line_no = line_idx + 1;
        if let Some(rest) = line.strip_prefix(';') {
            let mut buf = String::from(rest);
            let mut closed = false;
            for (_, next_line) in lines.by_ref() {
                if next_line.starts_with(';') {
                    closed = true;
                    break;
                }
                buf.push('\n');
                buf.push_str(next_line);
            }
            if !closed {
                return Err(format!("mmCIF line {line_no}: unterminated ';' text block"));
            }
            out.push((buf.trim().to_string(), true));
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
                let mut closed = false;
                while i < n {
                    if bytes[i] == q && (i + 1 == n || bytes[i + 1].is_ascii_whitespace()) {
                        closed = true;
                        break;
                    }
                    i += 1;
                }
                if !closed {
                    return Err(format!("mmCIF line {line_no}: unterminated quoted token"));
                }
                out.push((line[start..i].to_string(), true));
                i += 1;
            } else {
                let start = i;
                while i < n && !bytes[i].is_ascii_whitespace() {
                    i += 1;
                }
                out.push((line[start..i].to_string(), false));
            }
        }
    }
    Ok(out)
}

/// Complete sidechain heavy atoms from `(N, CA, CB)` and `chi1_deg` when sidechain coordinates are absent.
fn reconstruct_sidechain_from_chi1(
    atoms: &mut BTreeMap<String, Vec3>,
    res_name: &str,
    n: Vec3,
    ca: Vec3,
    cb: Vec3,
    chi1_deg: Option<f64>,
) {
    let tet = 110.5_f64.to_radians();
    let chi1 = chi1_deg.unwrap_or(-65.0).to_radians();
    match res_name {
        "CYS" => {
            atoms
                .entry("SG".to_string())
                .or_insert_with(|| nerf_place_atom(n, ca, cb, 1.808, 114.0_f64.to_radians(), chi1));
        }
        "SER" => {
            atoms
                .entry("OG".to_string())
                .or_insert_with(|| nerf_place_atom(n, ca, cb, 1.417, tet, chi1));
        }
        "THR" => {
            atoms
                .entry("OG1".to_string())
                .or_insert_with(|| nerf_place_atom(n, ca, cb, 1.430, tet, chi1));
            atoms.entry("CG2".to_string()).or_insert_with(|| {
                nerf_place_atom(n, ca, cb, 1.521, tet, chi1 - 120.0_f64.to_radians())
            });
        }
        "VAL" => {
            atoms
                .entry("CG1".to_string())
                .or_insert_with(|| nerf_place_atom(n, ca, cb, 1.521, tet, chi1));
            atoms.entry("CG2".to_string()).or_insert_with(|| {
                nerf_place_atom(n, ca, cb, 1.521, tet, chi1 + 120.0_f64.to_radians())
            });
        }
        "ILE" => {
            let cg1 = *atoms
                .entry("CG1".to_string())
                .or_insert_with(|| nerf_place_atom(n, ca, cb, 1.530, tet, chi1));
            atoms.entry("CG2".to_string()).or_insert_with(|| {
                nerf_place_atom(n, ca, cb, 1.521, tet, chi1 + 120.0_f64.to_radians())
            });
            atoms
                .entry("CD1".to_string())
                .or_insert_with(|| nerf_place_atom(ca, cb, cg1, 1.516, 113.8_f64.to_radians(), PI));
        }
        "LEU" => {
            let cg = *atoms
                .entry("CG".to_string())
                .or_insert_with(|| nerf_place_atom(n, ca, cb, 1.530, 115.5_f64.to_radians(), chi1));
            atoms
                .entry("CD1".to_string())
                .or_insert_with(|| nerf_place_atom(ca, cb, cg, 1.521, tet, PI));
            atoms
                .entry("CD2".to_string())
                .or_insert_with(|| nerf_place_atom(ca, cb, cg, 1.521, tet, 65.0_f64.to_radians()));
        }
        "MET" => {
            let cg = *atoms
                .entry("CG".to_string())
                .or_insert_with(|| nerf_place_atom(n, ca, cb, 1.520, 113.5_f64.to_radians(), chi1));
            let sd = *atoms
                .entry("SD".to_string())
                .or_insert_with(|| nerf_place_atom(ca, cb, cg, 1.807, 112.0_f64.to_radians(), PI));
            atoms.entry("CE".to_string()).or_insert_with(|| {
                nerf_place_atom(
                    cb,
                    cg,
                    sd,
                    1.790,
                    100.9_f64.to_radians(),
                    70.0_f64.to_radians(),
                )
            });
        }
        "PHE" | "TYR" => {
            let cg = *atoms
                .entry("CG".to_string())
                .or_insert_with(|| nerf_place_atom(n, ca, cb, 1.502, 113.8_f64.to_radians(), chi1));
            let chi2 = 90.0_f64.to_radians();
            let cd1 = *atoms.entry("CD1".to_string()).or_insert_with(|| {
                nerf_place_atom(ca, cb, cg, 1.390, 120.0_f64.to_radians(), chi2)
            });
            let cd2 = *atoms.entry("CD2".to_string()).or_insert_with(|| {
                nerf_place_atom(ca, cb, cg, 1.390, 120.0_f64.to_radians(), chi2 + PI)
            });
            let ce1 = *atoms
                .entry("CE1".to_string())
                .or_insert_with(|| nerf_place_atom(cb, cg, cd1, 1.390, 120.0_f64.to_radians(), PI));
            let ce2 = *atoms
                .entry("CE2".to_string())
                .or_insert_with(|| nerf_place_atom(cb, cg, cd2, 1.390, 120.0_f64.to_radians(), PI));
            let cz = *atoms.entry("CZ".to_string()).or_insert_with(|| {
                nerf_place_atom(cg, cd1, ce1, 1.390, 120.0_f64.to_radians(), 0.0)
            });
            let _ = ce2;
            if res_name == "TYR" {
                atoms.entry("OH".to_string()).or_insert_with(|| {
                    nerf_place_atom(cd1, ce1, cz, 1.376, 120.0_f64.to_radians(), PI)
                });
            }
        }
        "HIS" => {
            let cg = *atoms
                .entry("CG".to_string())
                .or_insert_with(|| nerf_place_atom(n, ca, cb, 1.495, 113.8_f64.to_radians(), chi1));
            let chi2 = -75.0_f64.to_radians();
            let nd1 = *atoms.entry("ND1".to_string()).or_insert_with(|| {
                nerf_place_atom(ca, cb, cg, 1.378, 122.7_f64.to_radians(), chi2)
            });
            let cd2 = *atoms.entry("CD2".to_string()).or_insert_with(|| {
                nerf_place_atom(ca, cb, cg, 1.354, 130.8_f64.to_radians(), chi2 + PI)
            });
            let ce1 = *atoms
                .entry("CE1".to_string())
                .or_insert_with(|| nerf_place_atom(cb, cg, nd1, 1.320, 108.5_f64.to_radians(), PI));
            atoms
                .entry("NE2".to_string())
                .or_insert_with(|| nerf_place_atom(cb, cg, cd2, 1.374, 107.2_f64.to_radians(), PI));
            let _ = ce1;
        }
        "TRP" => {
            let cg = *atoms
                .entry("CG".to_string())
                .or_insert_with(|| nerf_place_atom(n, ca, cb, 1.498, 113.8_f64.to_radians(), chi1));
            let chi2 = -90.0_f64.to_radians();
            let cd1 = *atoms.entry("CD1".to_string()).or_insert_with(|| {
                nerf_place_atom(ca, cb, cg, 1.365, 126.9_f64.to_radians(), chi2)
            });
            let cd2 = *atoms.entry("CD2".to_string()).or_insert_with(|| {
                nerf_place_atom(ca, cb, cg, 1.433, 126.4_f64.to_radians(), chi2 + PI)
            });
            let ne1 = *atoms
                .entry("NE1".to_string())
                .or_insert_with(|| nerf_place_atom(cb, cg, cd1, 1.375, 110.2_f64.to_radians(), PI));
            let ce2 = *atoms
                .entry("CE2".to_string())
                .or_insert_with(|| nerf_place_atom(cb, cg, cd2, 1.412, 107.2_f64.to_radians(), PI));
            let ce3 = *atoms.entry("CE3".to_string()).or_insert_with(|| {
                nerf_place_atom(cb, cg, cd2, 1.400, 133.9_f64.to_radians(), 0.0)
            });
            let cz2 = *atoms.entry("CZ2".to_string()).or_insert_with(|| {
                nerf_place_atom(cg, cd2, ce2, 1.396, 122.4_f64.to_radians(), PI)
            });
            let cz3 = *atoms.entry("CZ3".to_string()).or_insert_with(|| {
                nerf_place_atom(cg, cd2, ce3, 1.392, 118.6_f64.to_radians(), PI)
            });
            atoms.entry("CH2".to_string()).or_insert_with(|| {
                nerf_place_atom(cd2, ce2, cz2, 1.380, 117.5_f64.to_radians(), 0.0)
            });
            let _ = (ne1, cz3);
        }
        _ => {
            if chi1_deg.is_some() {
                atoms
                    .entry("CG".to_string())
                    .or_insert_with(|| nerf_place_atom(n, ca, cb, 1.525, tet, chi1));
            }
        }
    }
}

type DisulfideResiduePair = ((String, i32), (String, i32));

/// Complete deterministic protons (`H^N`, `HA`/`HA2`/`HA3`, `ILVMAT` 3-proton methyl rotamers),
/// compute `(phi, psi, omega, chi1, chi2)`, detect disulfides, extract conjugated aromatic rings,
/// and resolve guarded per-residue `pLDDT`.
fn finalize_parsed_structure(
    raw_residues: Vec<RawParsedResidue>,
    provenance: StructCoordinateProvenance,
    force_bfactor_plddt: bool,
    explicit_disulfides: &[DisulfideResiduePair],
    diagnostics: &mut Vec<String>,
) -> Result<StructureModel, String> {
    // Retain residues that have complete backbone `(N, CA, C)`
    let kept: Vec<RawParsedResidue> = raw_residues
        .into_iter()
        .filter(|r| {
            r.atoms.contains_key("N") && r.atoms.contains_key("CA") && r.atoms.contains_key("C")
        })
        .collect();
    if kept.is_empty() {
        return Err("coordinate model contains zero standard protein residues with complete (N, CA, C) backbone".to_string());
    }

    let l = kept.len();
    let has_explicit_ma_plddt = kept.iter().any(|r| r.explicit_plddt.is_some());

    // Determine whether `B_iso` should be interpreted as `pLDDT` when `_ma_qa_metric_local` is absent
    let mut use_bfactor_plddt = false;
    let mut rescale_01_to_100 = false;
    if !has_explicit_ma_plddt {
        if provenance == StructCoordinateProvenance::Experimental && !force_bfactor_plddt {
            use_bfactor_plddt = false;
        } else {
            let b_vals: Vec<f64> = kept
                .iter()
                .filter_map(|r| {
                    r.ca_bfactor.or_else(|| {
                        (r.bfactor_count > 0).then(|| r.bfactor_sum / (r.bfactor_count as f64))
                    })
                })
                .collect();
            if b_vals.len() == l {
                let min_b = b_vals.iter().copied().fold(f64::INFINITY, f64::min);
                let max_b = b_vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let is_constant_or_zero = max_b <= 1.0e-6 || (l >= 2 && (max_b - min_b) <= 1.0e-6);
                if (force_bfactor_plddt || provenance == StructCoordinateProvenance::Computational)
                    && !is_constant_or_zero
                    && min_b >= 0.0
                    && max_b <= 100.001
                {
                    use_bfactor_plddt = true;
                    let mut sorted = b_vals.clone();
                    sorted.sort_by(f64::total_cmp);
                    let med = sorted[sorted.len() / 2];
                    if max_b <= 1.0 && med > 0.15 {
                        rescale_01_to_100 = true;
                        diagnostics.push(
                            "rescaled computational [0, 1] B_iso confidence column to [0, 100] pLDDT"
                                .to_string(),
                        );
                    }
                }
                // Never treat headerless PDB/mmCIF B <= 100 as computational pLDDT (`21.W1:P0-2`)!
            }
        }
    }

    // First pass: extract (N, CA, C), reconstruct O if missing, reconstruct H^N, CB, HA, methyls
    let mut residues: Vec<StructResidueCoords> = Vec::with_capacity(l);
    for i in 0..l {
        let rr = &kept[i];
        let mut atoms = rr.atoms.clone();
        let n = atoms["N"];
        let ca = atoms["CA"];
        let c = atoms["C"];

        let peptide_prev = i > 0
            && kept[i - 1].chain_id == rr.chain_id
            && v_dist(kept[i - 1].atoms["C"], n) <= 2.50;
        let peptide_next = i + 1 < l
            && kept[i + 1].chain_id == rr.chain_id
            && v_dist(c, kept[i + 1].atoms["N"]) <= 2.50;

        // Reconstruct carbonyl O if absent
        let o = match atoms.get("O").copied() {
            Some(ov) => ov,
            None => {
                let ov = if peptide_next {
                    let next_n = kept[i + 1].atoms["N"];
                    let next_ca = kept[i + 1].atoms["CA"];
                    nerf_place_atom(next_ca, next_n, c, BOND_C_O, 123.0_f64.to_radians(), 0.0)
                } else {
                    nerf_place_atom(n, ca, c, BOND_C_O, ANGLE_CA_C_O_DEG.to_radians(), PI)
                };
                atoms.insert("O".to_string(), ov);
                ov
            }
        };

        // Reconstruct CB and chiral HA (or Gly HA2/HA3) when missing
        let (rec_cb, rec_ha, rec_ha2, rec_ha3, _) =
            reconstruct_ca_substituents(&rr.res_name, n, ca, c);
        let cb = if rr.res_name == "GLY" {
            None
        } else {
            let cb_v = atoms.get("CB").copied().or(rec_cb);
            if let Some(v) = cb_v {
                atoms.entry("CB".to_string()).or_insert(v);
            }
            cb_v
        };

        let (ha, ha2, ha3) = if rr.res_name == "GLY" {
            let h2 = atoms.get("HA2").copied().or(rec_ha2);
            let h3 = atoms.get("HA3").copied().or(rec_ha3);
            if let Some(v) = h2 {
                atoms.entry("HA2".to_string()).or_insert(v);
            }
            if let Some(v) = h3 {
                atoms.entry("HA3".to_string()).or_insert(v);
            }
            let h_mid = match (atoms.get("HA").copied(), h2, h3) {
                (Some(hm), _, _) => hm,
                (None, Some(a), Some(b)) => v_scale(v_add(a, b), 0.5),
                _ => rec_ha,
            };
            atoms.entry("HA".to_string()).or_insert(h_mid);
            (h_mid, h2, h3)
        } else {
            let ha_v = match atoms.get("HA").copied() {
                Some(hv) => hv,
                None => {
                    if let Some(cb_v) = cb {
                        let sum_u = v_add(
                            v_add(v_unit(v_sub(n, ca)), v_unit(v_sub(c, ca))),
                            v_unit(v_sub(cb_v, ca)),
                        );
                        if v_norm(sum_u) > 1e-6 {
                            v_add(ca, v_scale(v_unit(sum_u), -BOND_CA_HA))
                        } else {
                            rec_ha
                        }
                    } else {
                        rec_ha
                    }
                }
            };
            atoms.entry("HA".to_string()).or_insert(ha_v);
            (ha_v, None, None)
        };

        // Reconstruct missing amide H^N via ideal trans-peptide geometry for non-Proline peptide-bonded residues
        let h = if rr.res_name == "PRO" {
            None
        } else if let Some(&hv) = atoms.get("H") {
            Some(hv)
        } else if peptide_prev {
            let prev_c = kept[i - 1].atoms["C"];
            let prev_ca = kept[i - 1].atoms["CA"];
            let omg_prev = calc_dihedral_deg(prev_ca, prev_c, n, ca).to_radians();
            let hv = nerf_place_atom(
                prev_ca,
                prev_c,
                n,
                BOND_N_H,
                ANGLE_C_N_H_DEG.to_radians(),
                omg_prev + PI,
            );
            atoms.insert("H".to_string(), hv);
            Some(hv)
        } else {
            None
        };

        // Backbone dihedrals
        let phi_deg = if peptide_prev {
            Some(calc_dihedral_deg(kept[i - 1].atoms["C"], n, ca, c))
        } else {
            None
        };
        let psi_deg = if peptide_next {
            Some(calc_dihedral_deg(n, ca, c, kept[i + 1].atoms["N"]))
        } else {
            None
        };
        let omega_deg = if peptide_prev {
            Some(calc_dihedral_deg(
                kept[i - 1].atoms["CA"],
                kept[i - 1].atoms["C"],
                n,
                ca,
            ))
        } else {
            None
        };

        // Sidechain chi1 and chi2
        let chi1_atom = match rr.res_name.as_str() {
            "CYS" => Some("SG"),
            "ILE" | "VAL" => Some("CG1"),
            "SER" => Some("OG"),
            "THR" => Some("OG1"),
            "GLY" | "ALA" => None,
            _ => Some("CG"),
        };
        let chi1_deg = match (cb, chi1_atom.and_then(|an| atoms.get(an).copied())) {
            (Some(cb_v), Some(xg)) => Some(calc_dihedral_deg(n, ca, cb_v, xg)),
            _ => None,
        };
        let chi2_deg = match rr.res_name.as_str() {
            "ILE" => match (cb, atoms.get("CG1"), atoms.get("CD1")) {
                (Some(cb_v), Some(&cg1), Some(&cd1)) => Some(calc_dihedral_deg(ca, cb_v, cg1, cd1)),
                _ => None,
            },
            "LEU" => match (cb, atoms.get("CG"), atoms.get("CD1")) {
                (Some(cb_v), Some(&cg), Some(&cd1)) => Some(calc_dihedral_deg(ca, cb_v, cg, cd1)),
                _ => None,
            },
            "MET" => match (cb, atoms.get("CG"), atoms.get("SD")) {
                (Some(cb_v), Some(&cg), Some(&sd)) => Some(calc_dihedral_deg(ca, cb_v, cg, sd)),
                _ => None,
            },
            "PHE" | "TYR" | "TRP" => match (cb, atoms.get("CG"), atoms.get("CD1")) {
                (Some(cb_v), Some(&cg), Some(&cd1)) => Some(calc_dihedral_deg(ca, cb_v, cg, cd1)),
                _ => None,
            },
            "HIS" => match (cb, atoms.get("CG"), atoms.get("ND1")) {
                (Some(cb_v), Some(&cg), Some(&nd1)) => Some(calc_dihedral_deg(ca, cb_v, cg, nd1)),
                _ => None,
            },
            _ => None,
        };

        // Ensure sidechain methyl heavy atoms and 3-proton tetrahedral rotamers exist for ILVMAT
        if let Some(cb_v) = cb {
            complete_ilvmat_methyl_rotamers(&mut atoms, &rr.res_name, n, ca, cb_v);
        }

        // Resolve residue pLDDT
        let plddt = if let Some(p) = rr.explicit_plddt {
            Some(p.clamp(0.0, 100.0))
        } else if use_bfactor_plddt {
            let b = rr.ca_bfactor.or_else(|| {
                (rr.bfactor_count > 0).then(|| rr.bfactor_sum / (rr.bfactor_count as f64))
            });
            b.map(|v| {
                let scaled = if rescale_01_to_100 { v * 100.0 } else { v };
                scaled.clamp(0.0, 100.0)
            })
        } else {
            None
        };

        residues.push(StructResidueCoords {
            chain_id: rr.chain_id.clone(),
            seq_id: rr.seq_id,
            res_name: rr.res_name.clone(),
            aa1: three_to_one(&rr.res_name) as u8,
            n,
            h,
            ca,
            ha,
            ha2,
            ha3,
            c,
            o,
            cb,
            atoms,
            phi_deg,
            psi_deg,
            omega_deg,
            chi1_deg,
            chi2_deg,
            cys_oxidized: false,
            plddt,
        });
    }

    // Disulfide detection: explicit SSBOND/_struct_conn OR 3D SG-SG geometry (`d_SG <= 2.85 A`)
    for ((c1, s1), (c2, s2)) in explicit_disulfides {
        for r in &mut residues {
            if r.res_name == "CYS"
                && ((&r.chain_id == c1 && r.seq_id == *s1)
                    || (&r.chain_id == c2 && r.seq_id == *s2))
            {
                r.cys_oxidized = true;
            }
        }
    }
    let cys_indices: Vec<usize> = (0..residues.len())
        .filter(|&idx| residues[idx].res_name == "CYS")
        .collect();
    for a_pos in 0..cys_indices.len() {
        let i = cys_indices[a_pos];
        for &j in &cys_indices[(a_pos + 1)..] {
            let adj_seq = residues[i].chain_id == residues[j].chain_id
                && (residues[i].seq_id - residues[j].seq_id).abs() == 1;
            if let (Some(&sg_i), Some(&sg_j)) =
                (residues[i].atoms.get("SG"), residues[j].atoms.get("SG"))
            {
                let d_sg = v_dist(sg_i, sg_j);
                let d_cb_ok = match (residues[i].cb, residues[j].cb) {
                    (Some(cbi), Some(cbj)) => v_dist(cbi, cbj) <= 4.95,
                    _ => true,
                };
                let cut = if adj_seq { 2.75 } else { 3.45 };
                if d_sg <= cut && d_cb_ok {
                    residues[i].cys_oxidized = true;
                    residues[j].cys_oxidized = true;
                }
            } else if !adj_seq {
                if let (Some(cb_i), Some(cb_j)) = (residues[i].cb, residues[j].cb) {
                    if v_dist(cb_i, cb_j) <= 4.55 {
                        residues[i].cys_oxidized = true;
                        residues[j].cys_oxidized = true;
                    }
                }
            }
        }
    }

    // Extract conjugated aromatic rings (`Phe`, `Tyr`, `His`, `Trp` 5-ring & 6-ring; Case 1995)
    let aromatic_rings = extract_conjugated_aromatic_rings(&residues);

    Ok(StructureModel {
        provenance,
        residues,
        aromatic_rings,
        diagnostics: diagnostics.clone(),
    })
}

fn complete_ilvmat_methyl_rotamers(
    atoms: &mut BTreeMap<String, Vec3>,
    res_name: &str,
    n: Vec3,
    ca: Vec3,
    cb: Vec3,
) {
    let tet = 109.47_f64.to_radians();
    // First ensure methyl heavy carbons exist if missing in backbone-only coordinates
    reconstruct_sidechain_from_chi1(atoms, res_name, n, ca, cb, None);

    let place_three_protons =
        |atoms: &mut BTreeMap<String, Vec3>, gp: Vec3, p: Vec3, mc: Vec3, h_names: [&str; 3]| {
            if h_names.iter().all(|&hn| atoms.contains_key(hn)) {
                return;
            }
            for (k, &hn) in h_names.iter().enumerate() {
                let tor = PI + (k as f64) * (2.0 * PI / 3.0);
                let hp = nerf_place_atom(gp, p, mc, 1.090, tet, tor);
                atoms.entry(hn.to_string()).or_insert(hp);
            }
        };

    match res_name {
        "ALA" => place_three_protons(atoms, n, ca, cb, ["HB1", "HB2", "HB3"]),
        "THR" => {
            if let Some(&cg2) = atoms.get("CG2") {
                place_three_protons(atoms, ca, cb, cg2, ["HG21", "HG22", "HG23"]);
            }
        }
        "VAL" => {
            if let Some(&cg1) = atoms.get("CG1") {
                place_three_protons(atoms, ca, cb, cg1, ["HG11", "HG12", "HG13"]);
            }
            if let Some(&cg2) = atoms.get("CG2") {
                place_three_protons(atoms, ca, cb, cg2, ["HG21", "HG22", "HG23"]);
            }
        }
        "LEU" => {
            if let Some(&cg) = atoms.get("CG") {
                if let Some(&cd1) = atoms.get("CD1") {
                    place_three_protons(atoms, cb, cg, cd1, ["HD11", "HD12", "HD13"]);
                }
                if let Some(&cd2) = atoms.get("CD2") {
                    place_three_protons(atoms, cb, cg, cd2, ["HD21", "HD22", "HD23"]);
                }
            }
        }
        "ILE" => {
            if let (Some(&cg1), Some(&cd1)) = (atoms.get("CG1"), atoms.get("CD1")) {
                place_three_protons(atoms, cb, cg1, cd1, ["HD11", "HD12", "HD13"]);
            }
        }
        "MET" => {
            if let (Some(&cg), Some(&sd), Some(&ce)) =
                (atoms.get("CG"), atoms.get("SD"), atoms.get("CE"))
            {
                let _ = cg;
                place_three_protons(atoms, cg, sd, ce, ["HE1", "HE2", "HE3"]);
            }
        }
        _ => {}
    }
}

fn extract_conjugated_aromatic_rings(
    residues: &[StructResidueCoords],
) -> Vec<ConjugatedAromaticRing> {
    const SPECS: &[(&str, &str, &[&str], f64)] = &[
        (
            "PHE",
            "6-ring",
            &["CG", "CD1", "CE1", "CZ", "CE2", "CD2"],
            1.00,
        ),
        (
            "TYR",
            "6-ring",
            &["CG", "CD1", "CE1", "CZ", "CE2", "CD2"],
            0.84,
        ),
        ("HIS", "5-ring", &["CG", "ND1", "CE1", "NE2", "CD2"], 0.53),
        ("TRP", "5-ring", &["CG", "CD1", "NE1", "CE2", "CD2"], 0.56),
        (
            "TRP",
            "6-ring",
            &["CD2", "CE2", "CZ2", "CH2", "CZ3", "CE3"],
            1.04,
        ),
    ];

    let mut rings = Vec::new();
    for (r_idx, r) in residues.iter().enumerate() {
        for &(rn, label, vnames, i_r) in SPECS {
            if r.res_name == rn {
                let mut verts = Vec::with_capacity(vnames.len());
                for &vn in vnames {
                    if let Some(&xyz) = r.atoms.get(vn) {
                        verts.push(xyz);
                    }
                }
                if verts.len() == vnames.len() {
                    let inv = 1.0 / (verts.len() as f64);
                    let mut cen = [0.0; 3];
                    for &p in &verts {
                        cen = v_add(cen, p);
                    }
                    cen = v_scale(cen, inv);
                    // Compute consistent right-handed ring normal aligned with vertex winding
                    let mut n_acc = [0.0; 3];
                    let m = verts.len();
                    for k in 0..m {
                        let u = v_sub(verts[k], cen);
                        let v = v_sub(verts[(k + 1) % m], cen);
                        n_acc = v_add(n_acc, v_cross(u, v));
                    }
                    let normal = v_unit(n_acc);
                    rings.push(ConjugatedAromaticRing {
                        chain_id: r.chain_id.clone(),
                        seq_id: r.seq_id,
                        residue_index: r_idx,
                        res_name: r.res_name.clone(),
                        ring_label: label.to_string(),
                        vertices: verts,
                        center: cen,
                        normal,
                        intensity_factor: i_r,
                    });
                }
            }
        }
    }
    rings
}

// ============================================================================
// 3. Sequence/pH/T-Corrected `ncIDP` Random-Coil Baseline (`Tamiola 2010`, `Platzer 2014`, `Kjaergaard 2011`)
// ============================================================================

/// Compute `ncIDP` random-coil chemical shifts `[H, N, CA, CB, C', HA]` (ppm) for residue `k`
/// including `(i +- 1)` neighbor effects (pre-Proline `Delta_CA = -1.95`, `Delta_C' = -1.95 ppm`),
/// Platzer Henderson-Hasselbalch pH titration, and Kjaergaard linear temperature coefficients.
pub fn ncidp_random_coil_backbone(
    sequence: &[String],
    chain_ids: &[String],
    seq_ids: &[i32],
    k: usize,
    cys_oxidized: bool,
    ph: f64,
    temp_kelvin: f64,
) -> [Option<f64>; 6] {
    let res = sequence[k].trim().to_ascii_uppercase();
    let has_prev = k > 0 && chain_ids[k - 1] == chain_ids[k] && seq_ids[k - 1] + 1 == seq_ids[k];
    let has_next = k + 1 < sequence.len()
        && chain_ids[k + 1] == chain_ids[k]
        && seq_ids[k] + 1 == seq_ids[k + 1];

    let prev = has_prev.then(|| sequence[k - 1].trim().to_ascii_uppercase());
    let next = has_next.then(|| sequence[k + 1].trim().to_ascii_uppercase());

    let (mut ca, mut cb, mut c, mut ha, mut h, mut n) = match res.as_str() {
        "ALA" => (52.48, Some(19.07), 177.82, 4.32, Some(8.26), Some(123.80)),
        "ARG" => (56.02, Some(30.76), 176.35, 4.34, Some(8.26), Some(121.30)),
        "ASN" => (53.12, Some(38.74), 175.24, 4.74, Some(8.40), Some(118.90)),
        "ASP" => (54.22, Some(41.05), 176.32, 4.62, Some(8.32), Some(120.40)),
        "CYS" => {
            if cys_oxidized {
                (55.55, Some(41.10), 174.60, 4.71, Some(8.35), Some(118.80))
            } else {
                (58.25, Some(28.02), 174.65, 4.55, Some(8.32), Some(118.80))
            }
        }
        "GLN" => (55.75, Some(29.35), 176.04, 4.34, Some(8.34), Some(119.90)),
        "GLU" => (56.62, Some(30.02), 176.62, 4.33, Some(8.38), Some(120.50)),
        "GLY" => (45.14, None, 173.95, 3.97, Some(8.33), Some(109.10)),
        "HIS" => (55.52, Some(29.25), 174.85, 4.68, Some(8.38), Some(118.60)),
        "ILE" => (61.12, Some(38.68), 176.42, 4.17, Some(8.04), Some(120.20)),
        "LEU" => (55.12, Some(42.32), 177.55, 4.35, Some(8.18), Some(122.00)),
        "LYS" => (56.24, Some(32.95), 176.62, 4.32, Some(8.29), Some(121.10)),
        "MET" => (55.42, Some(32.78), 176.28, 4.48, Some(8.28), Some(119.80)),
        "PHE" => (57.74, Some(39.55), 175.82, 4.62, Some(8.28), Some(120.50)),
        "PRO" => (63.28, Some(32.02), 177.26, 4.42, None, None),
        "SER" => (58.30, Some(63.76), 174.65, 4.47, Some(8.31), Some(115.90)),
        "THR" => (61.78, Some(69.75), 174.72, 4.35, Some(8.16), Some(113.90)),
        "TRP" => (57.42, Some(29.58), 176.12, 4.66, Some(8.22), Some(121.40)),
        "TYR" => (57.88, Some(38.75), 175.88, 4.55, Some(8.14), Some(120.40)),
        "VAL" => (62.22, Some(32.82), 176.32, 4.12, Some(8.06), Some(119.60)),
        _ => (56.00, Some(33.00), 176.00, 4.35, Some(8.25), Some(120.00)),
    };

    // 1. Succeeding residue (i + 1) covalent neighbor effects (including pre-Proline -1.95 ppm on CA & C')
    if let Some(ref nx) = next {
        if nx == "PRO" {
            if res == "GLY" {
                ca -= 1.00;
                c -= 1.50;
            } else {
                ca -= 1.95;
                c -= 1.95;
                if let Some(ref mut cb_v) = cb {
                    *cb_v += 0.22;
                }
            }
            ha += 0.24;
            if let Some(ref mut n_v) = n {
                *n_v += 1.05;
            }
        } else if matches!(nx.as_str(), "VAL" | "ILE" | "THR") {
            c -= 0.22;
            if let Some(ref mut n_v) = n {
                *n_v -= 0.35;
            }
        } else if matches!(nx.as_str(), "PHE" | "TYR" | "TRP") {
            ca -= 0.12;
            c -= 0.18;
            if let Some(ref mut h_v) = h {
                *h_v -= 0.05;
            }
        } else if nx == "GLY" {
            c += 0.15;
        }
    }

    // 2. Preceding residue (i - 1) peptide-bond effects
    if let Some(ref pv) = prev {
        match pv.as_str() {
            "GLY" => {
                if let Some(ref mut n_v) = n {
                    *n_v -= 0.95;
                }
                if let Some(ref mut h_v) = h {
                    *h_v += 0.04;
                }
            }
            "VAL" | "ILE" | "THR" => {
                if let Some(ref mut n_v) = n {
                    *n_v += 2.15;
                }
                if let Some(ref mut h_v) = h {
                    *h_v += 0.07;
                }
                ca -= 0.12;
            }
            "PHE" | "TYR" | "TRP" => {
                if let Some(ref mut n_v) = n {
                    *n_v += 0.85;
                }
                if let Some(ref mut h_v) = h {
                    *h_v -= 0.08;
                }
                ca -= 0.15;
            }
            "PRO" => {
                if let Some(ref mut n_v) = n {
                    *n_v += 0.70;
                }
                ca -= 0.10;
            }
            "ALA" => {
                if let Some(ref mut n_v) = n {
                    *n_v -= 0.35;
                }
            }
            _ => {}
        }
    }

    // 3. Henderson-Hasselbalch pH titration relative to pH 6.5 (Platzer et al. 2014)
    let eff_ph = ph.clamp(0.5, 13.5);
    let frac_prot = |pka: f64, cur_ph: f64| -> f64 {
        1.0 / (1.0 + det_exp(std::f64::consts::LN_10 * (cur_ph - pka)))
    };
    let delta_f = |pka: f64| -> f64 { frac_prot(pka, eff_ph) - frac_prot(pka, 6.5) };

    match res.as_str() {
        "ASP" => {
            let df = delta_f(3.90);
            ca -= 1.48 * df;
            if let Some(ref mut cb_v) = cb {
                *cb_v -= 3.25 * df;
            }
            c -= 1.05 * df;
            ha += 0.14 * df;
            if let Some(ref mut h_v) = h {
                *h_v += 0.16 * df;
            }
            if let Some(ref mut n_v) = n {
                *n_v -= 0.35 * df;
            }
        }
        "GLU" => {
            let df = delta_f(4.30);
            ca -= 1.38 * df;
            if let Some(ref mut cb_v) = cb {
                *cb_v -= 1.65 * df;
            }
            c -= 0.75 * df;
            ha += 0.09 * df;
            if let Some(ref mut h_v) = h {
                *h_v += 0.10 * df;
            }
            if let Some(ref mut n_v) = n {
                *n_v += 0.25 * df;
            }
        }
        "HIS" => {
            let df = delta_f(6.35);
            ca -= 1.25 * df;
            if let Some(ref mut cb_v) = cb {
                *cb_v -= 0.85 * df;
            }
            c -= 0.75 * df;
            ha += 0.12 * df;
            if let Some(ref mut h_v) = h {
                *h_v += 0.22 * df;
            }
            if let Some(ref mut n_v) = n {
                *n_v += 0.90 * df;
            }
        }
        _ => {}
    }

    // Chain-end adjustments ONLY when true chain N- or C-terminus
    let is_chain_start = k == 0 || chain_ids[k - 1] != chain_ids[k];
    let is_chain_end = k + 1 == sequence.len() || chain_ids[k + 1] != chain_ids[k];
    if is_chain_start {
        ca -= 0.80;
        c -= 3.50;
        h = None;
        n = None;
    } else if is_chain_end {
        let f_cterm = frac_prot(3.60, eff_ph);
        ca += 0.65 * (1.0 - f_cterm) - 0.85 * f_cterm;
        c += 2.80 * (1.0 - f_cterm) - 0.40 * f_cterm;
        if let Some(ref mut n_v) = n {
            *n_v += 4.20 * (1.0 - f_cterm) - 0.80 * f_cterm;
        }
        if let Some(ref mut h_v) = h {
            *h_v -= 0.25 * (1.0 - f_cterm);
        }
    }

    // 4. Kjaergaard et al. (2011) linear temperature coefficients relative to 298.15 K
    let dt = (temp_kelvin - 298.15).clamp(-40.0, 60.0);
    if dt.abs() > 1e-6 {
        if let Some(ref mut h_v) = h {
            *h_v += -0.0046 * dt;
        }
        if let Some(ref mut n_v) = n {
            *n_v += -0.0170 * dt;
        }
        ca += -0.0025 * dt;
        if let Some(ref mut cb_v) = cb {
            *cb_v += 0.0030 * dt;
        }
        c += -0.0055 * dt;
    }

    [h, n, Some(ca), cb, Some(c), Some(ha)]
}

// ============================================================================
// 4. Haigh-Mallion Ring Currents, Wagner-Pardi-Bax H-Bonds & Peptide Anisotropy/EF
// ============================================================================

/// Evaluate Haigh-Mallion (1972) / Case (1995) aromatic ring-current chemical shift perturbation
/// `Delta_delta^{RC-arom}` (ppm, negative = upfield above/below ring face, positive = downfield
/// in-plane outside ring) at target spin coordinate `r_p`, together with the minimum distance
/// from `r_p` to any non-self aromatic ring atom/center.
pub fn evaluate_haigh_mallion_ring_current(
    rings: &[ConjugatedAromaticRing],
    r_p: Vec3,
    target_residue_idx: usize,
    is_same_aromatic_sidechain: bool,
    susceptibility_scale: f64,
) -> (f64, Option<f64>) {
    let cutoff_sq = RING_CURRENT_CUTOFF_ANGSTROM * RING_CURRENT_CUTOFF_ANGSTROM;
    let mut delta_ppm = 0.0;
    let mut min_dist: Option<f64> = None;

    for ring in rings {
        if is_same_aromatic_sidechain && ring.residue_index == target_residue_idx {
            continue;
        }
        let d_cen_sq = v_dist_sq(r_p, ring.center);
        if d_cen_sq > cutoff_sq {
            continue;
        }
        let mut ring_min_d = d_cen_sq.sqrt();
        for &v in &ring.vertices {
            let dv = v_dist(r_p, v);
            if dv < ring_min_d {
                ring_min_d = dv;
            }
        }
        min_dist = Some(match min_dist {
            Some(cur) => cur.min(ring_min_d),
            None => ring_min_d,
        });

        let m = ring.vertices.len();
        let mut geom_sum = 0.0;
        for k in 0..m {
            let u = v_sub(ring.vertices[k], r_p);
            let v = v_sub(ring.vertices[(k + 1) % m], r_p);
            let r_u = v_norm(u).max(RING_CURRENT_CLASH_FLOOR_ANGSTROM);
            let r_v = v_norm(v).max(RING_CURRENT_CLASH_FLOOR_ANGSTROM);
            let signed_area2 = v_dot(ring.normal, v_cross(u, v));
            geom_sum += 0.5 * signed_area2 * (1.0 / (r_u * r_u * r_u) + 1.0 / (r_v * r_v * r_v));
        }
        // Positive Haigh-Mallion shielding sigma_HM > 0 above the pi cloud corresponds to an
        // upfield chemical shift perturbation Delta_delta = -sigma_HM < 0 (Case 1995).
        let shielding_ppm =
            ring.intensity_factor * HAIGH_MALLION_PROTON_CONSTANT * susceptibility_scale * geom_sum;
        delta_ppm -= shielding_ppm;
    }

    (delta_ppm, min_dist)
}

#[derive(Clone, Copy, Debug, Default)]
struct ResiduePhysicalShifts {
    /// Direct Wagner-Pardi-Bax H-bond shifts `[H, N, CA, CB, C', HA]` (ppm).
    hbond: [f64; 6],
    /// McConnell peptide carbonyl anisotropy + Buckingham electric-field shifts `[H, N, CA, CB, C', HA]` (ppm).
    peptide_ef: [f64; 6],
    /// Haigh-Mallion conjugated aromatic ring-current shifts `[H, N, CA, CB, C', HA]` (ppm).
    ring_current: [f64; 6],
    /// Kabsch-Sander electrostatic H-bond energies & secondary-structure indicators for Fourier model.
    ks_nh_energy: f64,
    ks_co_energy: f64,
    is_helical_hbond: f64,
    is_sheet_hbond: f64,
    nearest_arom_h: Option<f64>,
    nearest_arom_ha: Option<f64>,
}

/// Canonicalize a [`StructureModel`] onto ideal Engh–Huber `(CA, phi, psi, chi1)` peptide-plane
/// and sidechain `OG`/`OG1`/`OH` geometry per chain segment (`21.W1:P2-3`) so that H-bond shift
/// terms (`hbond`, `ks_nh_energy`, `ks_co_energy`, `is_helical_hbond`, `is_sheet_hbond`) are
/// evaluated on identical backbone/acceptor geometry during both training (`torsion_corpus.tsv`)
/// and query inference.
fn canonicalize_model_for_hbond(model: &StructureModel) -> Vec<StructResidueCoords> {
    let l = model.residues.len();
    if l == 0 {
        return Vec::new();
    }
    let mut out_res = Vec::with_capacity(l);
    let mut seg_start = 0usize;
    while seg_start < l {
        let ch_id = &model.residues[seg_start].chain_id;
        let mut seg_end = seg_start + 1;
        while seg_end < l && model.residues[seg_end].chain_id == *ch_id {
            seg_end += 1;
        }
        let slice = &model.residues[seg_start..seg_end];
        if slice.len() >= 2 {
            let seq_ids: Vec<i32> = slice.iter().map(|r| r.seq_id).collect();
            let res_names: Vec<String> = slice.iter().map(|r| r.res_name.clone()).collect();
            let ca_coords: Vec<[f64; 3]> = slice.iter().map(|r| r.ca).collect();
            let phi_v: Vec<Option<f64>> = slice.iter().map(|r| r.phi_deg).collect();
            let psi_v: Vec<Option<f64>> = slice.iter().map(|r| r.psi_deg).collect();
            let chi1_v: Vec<Option<f64>> = slice.iter().map(|r| r.chi1_deg).collect();
            if let Ok(rebuilt) = StructureModel::from_ca_and_torsions(
                ch_id, &seq_ids, &res_names, &ca_coords, &phi_v, &psi_v, &chi1_v,
            ) {
                if rebuilt.residues.len() == slice.len() {
                    out_res.extend(rebuilt.residues);
                    seg_start = seg_end;
                    continue;
                }
            }
        }
        out_res.extend_from_slice(slice);
        seg_start = seg_end;
    }
    out_res
}

/// Evaluate per-residue empirical H-bond chemical-shift contributions `[H, N, CA, CB, C', HA]` (ppm)
/// on the harmonized canonical `(CA, phi, psi, chi1)` peptide-plane geometry (`21.W1:P2-3`).
pub fn evaluate_hbond_shifts_ppm(model: &StructureModel) -> Vec<[f64; 6]> {
    evaluate_all_physical_terms(model)
        .into_iter()
        .map(|p| p.hbond)
        .collect()
}

/// Evaluate Kabsch-Sander electrostatic H-bond indicators, empirical `r_HO^-3 cos^2(theta)` H-bond
/// shifts (inspired by Wagner, Pardi & Bax 1983), McConnell peptide carbonyl anisotropy,
/// CHARMM22 (MacKerell et al. 1998) / Buckingham bond-parallel electric field, and
/// Haigh-Mallion aromatic ring-current shifts across all residues of `model`.
fn evaluate_all_physical_terms(model: &StructureModel) -> Vec<ResiduePhysicalShifts> {
    let l = model.residues.len();
    let mut out = vec![ResiduePhysicalShifts::default(); l];
    let hb_residues = canonicalize_model_for_hbond(model);

    // Collect candidate sidechain hydroxyl acceptors (`OG`, `OG1`, `OH`) present in both canonical
    // training and query geometries (`21.W1:P2-3`)
    let mut sc_oxygens: Vec<(usize, Vec3)> = Vec::new();
    for (idx, r) in hb_residues.iter().enumerate() {
        for oname in ["OG", "OG1", "OH"] {
            if let Some(&ox) = r.atoms.get(oname) {
                sc_oxygens.push((idx, ox));
            }
        }
    }

    // 1. Kabsch-Sander electrostatic H-bonds + empirical `r_HO^-3 cos^2(theta)` H-bond shifts
    //    E_HB(donor_co k -> acceptor_nh i) = 0.42 * 0.20 * 332 * (1/r_ON + 1/r_CH - 1/r_OH - 1/r_CN)
    const KS_FACTOR: f64 = 0.42 * 0.20 * 332.0; // 27.888 kcal*A/mol
    const R0_HB: f64 = 2.60;
    const INV_R0_CUBE: f64 = 1.0 / (R0_HB * R0_HB * R0_HB);

    let mut turn4 = vec![false; l];
    let mut turn3 = vec![false; l];

    for i in 0..l {
        let ri = &hb_residues[i];
        // Amide N_i - H_i H-bond donor to backbone O_k and sidechain O_k
        if let Some(h_i) = ri.h {
            let mut best_nh_hb = 0.0_f64;
            let mut best_ks_e = 0.0_f64;
            for (k, rk) in hb_residues.iter().enumerate() {
                let same_chain = rk.chain_id == ri.chain_id;
                let seq_sep = (ri.seq_id - rk.seq_id).abs();
                if same_chain && seq_sep < 2 {
                    continue;
                }
                let d_oh = v_dist(rk.o, h_i);
                if d_oh <= 5.2 {
                    let r_on = v_dist(rk.o, ri.n).max(0.5);
                    let r_ch = v_dist(rk.c, h_i).max(0.5);
                    let r_oh = d_oh.max(0.5);
                    let r_cn = v_dist(rk.c, ri.n).max(0.5);
                    let e_ks = KS_FACTOR * (1.0 / r_on + 1.0 / r_ch - 1.0 / r_oh - 1.0 / r_cn);
                    if e_ks < best_ks_e {
                        best_ks_e = e_ks;
                    }
                    if e_ks < out[k].ks_co_energy {
                        out[k].ks_co_energy = e_ks;
                    }
                    if e_ks < -0.5 && same_chain {
                        if ri.seq_id - rk.seq_id == 4 && i >= 4 {
                            turn4[k] = true;
                        } else if ri.seq_id - rk.seq_id == 3 && i >= 3 {
                            turn3[k] = true;
                        } else if seq_sep >= 3 {
                            out[i].is_sheet_hbond = 1.0;
                            out[k].is_sheet_hbond = 1.0;
                        }
                    }
                }
                if (1.55..=3.20).contains(&d_oh) {
                    // Angle N_i - H_i ... O_k
                    let u_hn = v_unit(v_sub(ri.n, h_i));
                    let u_ho = v_unit(v_sub(rk.o, h_i));
                    let cos_ang = v_dot(u_hn, u_ho); // cos(theta_NHO); >= cos(110 deg) = -0.342 => obtuse angle means cos_ang <= -0.342!
                    if cos_ang <= -0.34202014 {
                        let cos2 = cos_ang * cos_ang;
                        let inv_r3 = 1.0 / (d_oh.max(1.65).powi(3));
                        let excess = (inv_r3 - INV_R0_CUBE).max(0.0);
                        let term = excess * cos2;
                        if term > best_nh_hb {
                            best_nh_hb = term;
                        }
                        // Reciprocal carbonyl 13C'_k deshielding when O_k accepts H-bond
                        let c_term = 5.40 * excess * cos2;
                        if c_term > out[k].hbond[BackboneShiftAtom::C.index()] {
                            out[k].hbond[BackboneShiftAtom::C.index()] = c_term;
                        }
                    }
                }
            }
            for &(sc_idx, ox) in &sc_oxygens {
                if sc_idx == i {
                    continue;
                }
                let d_oh = v_dist(ox, h_i);
                if (1.55..=2.60).contains(&d_oh) {
                    let u_hn = v_unit(v_sub(ri.n, h_i));
                    let u_ho = v_unit(v_sub(ox, h_i));
                    let cos_ang = v_dot(u_hn, u_ho);
                    if cos_ang <= -0.34202014 {
                        let excess = (1.0 / (d_oh.max(1.65).powi(3)) - INV_R0_CUBE).max(0.0);
                        let term = 0.75 * excess * cos_ang * cos_ang;
                        if term > best_nh_hb {
                            best_nh_hb = term;
                        }
                    }
                }
            }
            out[i].ks_nh_energy = best_ks_e;
            // Empirical inverse-cube amide 1H^N & 15N H-bond deshielding
            out[i].hbond[BackboneShiftAtom::H.index()] = (4.15 * best_nh_hb).min(1.35);
            out[i].hbond[BackboneShiftAtom::N.index()] = (8.80 * best_nh_hb).min(3.00);
        }

        // Weak C_alpha - H_alpha ... O=C H-bond deshielding in beta-sheets
        let mut best_ha_hb = 0.0_f64;
        for ha_pt in [Some(ri.ha), ri.ha2, ri.ha3].into_iter().flatten() {
            for rk in &hb_residues {
                if rk.chain_id == ri.chain_id && (ri.seq_id - rk.seq_id).abs() < 2 {
                    continue;
                }
                let d_ho = v_dist(ha_pt, rk.o);
                if (1.80..=2.75).contains(&d_ho) {
                    let u_ch = v_unit(v_sub(ri.ca, ha_pt));
                    let u_ho = v_unit(v_sub(rk.o, ha_pt));
                    let cos_ang = v_dot(u_ch, u_ho);
                    if cos_ang <= -0.34202014 {
                        let excess =
                            (1.0 / (d_ho.max(1.85).powi(3)) - (1.0 / 2.75_f64.powi(3))).max(0.0);
                        best_ha_hb = best_ha_hb.max(2.60 * excess * cos_ang * cos_ang);
                    }
                }
            }
        }
        out[i].hbond[BackboneShiftAtom::HA.index()] = best_ha_hb.min(0.55);
    }

    // Mark helical spans from overlapping 4-turns / 3-turns (Kabsch & Sander 1983)
    for k in 0..l {
        if turn4[k] && k + 1 < l && turn4[k + 1] {
            for slot in out.iter_mut().take((k + 4).min(l - 1) + 1).skip(k + 1) {
                slot.is_helical_hbond = 1.0;
                slot.is_sheet_hbond = 0.0;
            }
        } else if turn3[k] && k + 1 < l && turn3[k + 1] {
            for slot in out.iter_mut().take((k + 3).min(l - 1) + 1).skip(k + 1) {
                if slot.is_helical_hbond == 0.0 {
                    slot.is_helical_hbond = 0.75;
                }
            }
        }
    }

    // 2. McConnell (1957) peptide carbonyl magnetic susceptibility anisotropy &
    //    Buckingham (1960) / Sitkoff & Case (1998) bond-parallel electric-field projection
    let pep_groups = build_peptide_dipoles(model);

    for (i, out_i) in out.iter_mut().enumerate() {
        let ri = &model.residues[i];
        // Evaluate on H^N and HA
        if let Some(h_i) = ri.h {
            let u_nh = v_unit(v_sub(h_i, ri.n));
            let (aniso, ef) =
                eval_peptide_aniso_and_ef(&pep_groups, h_i, u_nh, &ri.chain_id, ri.seq_id, true);
            out_i.peptide_ef[BackboneShiftAtom::H.index()] = (aniso + ef).clamp(-0.65, 0.65);
        }
        let u_caha = v_unit(v_sub(ri.ha, ri.ca));
        let (aniso_ha, ef_ha) =
            eval_peptide_aniso_and_ef(&pep_groups, ri.ha, u_caha, &ri.chain_id, ri.seq_id, false);
        out_i.peptide_ef[BackboneShiftAtom::HA.index()] = (aniso_ha + ef_ha).clamp(-0.45, 0.45);

        // 3. Haigh-Mallion conjugated aromatic ring-current shielding (excluding intra-residue self-terms,
        //    whose average contribution is already absorbed in the ncIDP random-coil baseline)
        if let Some(h_i) = ri.h {
            let (rc_h, min_d_h) =
                evaluate_haigh_mallion_ring_current(&model.aromatic_rings, h_i, i, true, 1.0);
            out_i.ring_current[BackboneShiftAtom::H.index()] = rc_h.clamp(-1.80, 1.40);
            out_i.nearest_arom_h = min_d_h;
        }
        let (rc_ha, min_d_ha) =
            evaluate_haigh_mallion_ring_current(&model.aromatic_rings, ri.ha, i, true, 1.0);
        out_i.ring_current[BackboneShiftAtom::HA.index()] = rc_ha.clamp(-1.50, 1.20);
        out_i.nearest_arom_ha = min_d_ha;

        // Heavy-atom isotropic ring-current shielding
        if ri.res_name != "PRO" {
            let (rc_n, _) =
                evaluate_haigh_mallion_ring_current(&model.aromatic_rings, ri.n, i, true, 0.65);
            out_i.ring_current[BackboneShiftAtom::N.index()] = rc_n.clamp(-1.50, 1.50);
        }
        let (rc_ca, _) =
            evaluate_haigh_mallion_ring_current(&model.aromatic_rings, ri.ca, i, true, 0.55);
        out_i.ring_current[BackboneShiftAtom::CA.index()] = rc_ca.clamp(-1.00, 1.00);
        if let Some(cb_v) = ri.cb {
            let (rc_cb, _) =
                evaluate_haigh_mallion_ring_current(&model.aromatic_rings, cb_v, i, true, 0.55);
            out_i.ring_current[BackboneShiftAtom::CB.index()] = rc_cb.clamp(-1.00, 1.00);
        }
        let (rc_c, _) =
            evaluate_haigh_mallion_ring_current(&model.aromatic_rings, ri.c, i, true, 0.55);
        out_i.ring_current[BackboneShiftAtom::C.index()] = rc_c.clamp(-1.00, 1.00);
    }

    out
}

fn eval_peptide_aniso_and_ef<T>(
    _dummy: &[T],
    _r_p: Vec3,
    _u_bond: Vec3,
    _chain_id: &str,
    _seq_id: i32,
    _is_amide: bool,
) -> (f64, f64)
where
    T: PeptideDipoleAccess,
{
    let mut aniso = 0.0;
    let mut e_field = [0.0; 3];
    let cut_sq = 64.0; // 8.0 A

    for pg in _dummy {
        let same_chain = pg.chain_id() == _chain_id;
        let sep = (_seq_id - pg.seq_id()).abs();
        // Exclude covalently adjacent peptide bonds whose local effects belong in the Fourier torsion surface
        if same_chain && sep <= 1 {
            continue;
        }
        let dv = v_sub(_r_p, pg.center());
        let r2 = v_norm_sq_local(dv);
        if r2 > cut_sq {
            continue;
        }
        let r = r2.sqrt().max(2.10);
        let inv_r = 1.0 / r;
        let inv_r3 = inv_r * inv_r * inv_r;
        let c1 = v_dot(dv, pg.e1()) * inv_r;
        let c2 = v_dot(dv, pg.e2()) * inv_r;
        let c3 = v_dot(dv, pg.e3()) * inv_r;
        // Out-of-plane carbonyl pi susceptibility anisotropy (McConnell 1957; Sitkoff & Case 1998):
        // c3^2 -> 1 above/below the peptide plane gives upfield shift (< 0)
        aniso += (0.28 * (1.0 - 3.0 * c3 * c3) + 0.08 * (c1 * c1 - c2 * c2)) * inv_r3;

        // Empirical H-bond shift term (FERRIC coefficients with inverse-cube r_HO^-3 scaling
        // inspired by Wagner, Pardi & Bax 1983) & CHARMM22 partial charges (MacKerell 1998)
        // ...
        // Note: training and query structures share identical `evaluate_all_physical_terms`.
        // Screened CHARMM22 (MacKerell et al., J. Phys. Chem. B 1998, 102, 3586-3616)
        // peptide backbone partial charges: C': +0.51, O: -0.51, N: -0.47, H: +0.31 e
        let add_q = |ef: &mut Vec3, pos: Vec3, q: f64| {
            let dr = v_sub(_r_p, pos);
            let d = v_norm(dr).max(2.00);
            let s = q / (d * d * d);
            *ef = v_add(*ef, v_scale(dr, s));
        };
        add_q(&mut e_field, pg.c_pos(), 0.51);
        add_q(&mut e_field, pg.o_pos(), -0.51);
        if let Some(np) = pg.n_next() {
            add_q(&mut e_field, np, -0.47);
        }
        if let Some(hp) = pg.h_next() {
            add_q(&mut e_field, hp, 0.31);
        }
    }

    // Buckingham (1960) / Sitkoff & Case (1998) bond-parallel electric field projection
    let a_par = if _is_amide { 0.85 } else { 0.55 };
    let ef_ppm = a_par * v_dot(_u_bond, e_field);
    (aniso, ef_ppm)
}

#[inline]
fn v_norm_sq_local(a: Vec3) -> f64 {
    a[0] * a[0] + a[1] * a[1] + a[2] * a[2]
}

trait PeptideDipoleAccess {
    fn chain_id(&self) -> &str;
    fn seq_id(&self) -> i32;
    fn center(&self) -> Vec3;
    fn e1(&self) -> Vec3;
    fn e2(&self) -> Vec3;
    fn e3(&self) -> Vec3;
    fn c_pos(&self) -> Vec3;
    fn o_pos(&self) -> Vec3;
    fn n_next(&self) -> Option<Vec3>;
    fn h_next(&self) -> Option<Vec3>;
}

// Implement PeptideDipoleAccess cleanly via a concrete module-level struct
#[derive(Clone, Debug)]
struct PeptideDipole {
    chain_id: String,
    seq_id: i32,
    center: Vec3,
    e1: Vec3,
    e2: Vec3,
    e3: Vec3,
    c_pos: Vec3,
    o_pos: Vec3,
    n_next: Option<Vec3>,
    h_next: Option<Vec3>,
}

impl PeptideDipoleAccess for PeptideDipole {
    #[inline]
    fn chain_id(&self) -> &str {
        &self.chain_id
    }
    #[inline]
    fn seq_id(&self) -> i32 {
        self.seq_id
    }
    #[inline]
    fn center(&self) -> Vec3 {
        self.center
    }
    #[inline]
    fn e1(&self) -> Vec3 {
        self.e1
    }
    #[inline]
    fn e2(&self) -> Vec3 {
        self.e2
    }
    #[inline]
    fn e3(&self) -> Vec3 {
        self.e3
    }
    #[inline]
    fn c_pos(&self) -> Vec3 {
        self.c_pos
    }
    #[inline]
    fn o_pos(&self) -> Vec3 {
        self.o_pos
    }
    #[inline]
    fn n_next(&self) -> Option<Vec3> {
        self.n_next
    }
    #[inline]
    fn h_next(&self) -> Option<Vec3> {
        self.h_next
    }
}

// ============================================================================
// 5. CC0-Firewalled Trigonometric Fourier Hypersurfaces (`K = 3`, `49` + Neighbor/Chi1/SS Terms)
// ============================================================================

/// Feature dimension `D_FEAT` for Fourier torsion hypersurfaces:
/// - `0..49`: `(phi_i, psi_i)` tensor-product Fourier series `K = 3` (`7 x 7 = 49` basis functions),
/// - `49..59`: preceding-residue `(phi_{i-1}, psi_{i-1})` `K' = 2` harmonics + cross terms (`10` basis functions),
/// - `59..63`: succeeding-residue `(phi_{i+1}, psi_{i+1})` `K'' = 1` harmonics (`4` basis functions),
/// - `63..69`: sidechain `chi1` harmonics `{I_chi1, cos(chi1), sin(chi1), cos(3*chi1), cos(psi)*cos(chi1), sin(psi)*sin(chi1)}` (`6` basis functions),
/// - `69..73`: Kabsch-Sander H-bond & secondary-structure indicators (`4` basis functions).
const FOURIER_BASE_DIM: usize = 73;
const AA_ALPHABET: &[u8; 20] = b"ARNDCQEGHILKMFPSTWYV";
const TOTAL_BASIS_DIM: usize = FOURIER_BASE_DIM + 20;

#[inline]
fn aa20_index(aa1: u8) -> Option<usize> {
    AA_ALPHABET
        .iter()
        .position(|&c| c == aa1.to_ascii_uppercase())
}

#[inline]
fn trig_harmonics_3(deg_opt: Option<f64>) -> [f64; 7] {
    let Some(deg) = deg_opt else {
        return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    };
    let rad = deg.to_radians();
    let (s1, c1) = det_sin_cos(rad);
    let (s2, c2) = det_sin_cos(2.0 * rad);
    let (s3, c3) = det_sin_cos(3.0 * rad);
    [1.0, c1, s1, c2, s2, c3, s3]
}

#[allow(clippy::too_many_arguments)]
fn build_torsion_basis_vector(
    aa1: u8,
    res_name: &str,
    phi_i: Option<f64>,
    psi_i: Option<f64>,
    phi_prev: Option<f64>,
    psi_prev: Option<f64>,
    phi_next: Option<f64>,
    psi_next: Option<f64>,
    chi1_i: Option<f64>,
    phys: &ResiduePhysicalShifts,
) -> [f64; TOTAL_BASIS_DIM] {
    let mut x = [0.0_f64; TOTAL_BASIS_DIM];
    let hp = trig_harmonics_3(phi_i);
    let hs = trig_harmonics_3(psi_i);

    // 0..49: 7 x 7 tensor-product Fourier basis on (phi_i, psi_i)
    let mut idx = 0usize;
    for &fp in &hp {
        for &fs in &hs {
            x[idx] = fp * fs;
            idx += 1;
        }
    }

    // 49..59: K' = 2 preceding (phi_{i-1}, psi_{i-1}) harmonics
    let hp_prev = trig_harmonics_3(phi_prev);
    let hs_prev = trig_harmonics_3(psi_prev);
    x[49] = hp_prev[1];
    x[50] = hp_prev[2];
    x[51] = hp_prev[3];
    x[52] = hp_prev[4];
    x[53] = hs_prev[1];
    x[54] = hs_prev[2];
    x[55] = hs_prev[3];
    x[56] = hs_prev[4];
    x[57] = hp_prev[1] * hs_prev[1];
    x[58] = hp_prev[2] * hs_prev[2];

    // 59..63: K'' = 1 succeeding (phi_{i+1}, psi_{i+1}) harmonics
    let hp_next = trig_harmonics_3(phi_next);
    let hs_next = trig_harmonics_3(psi_next);
    x[59] = hp_next[1];
    x[60] = hp_next[2];
    x[61] = hs_next[1];
    x[62] = hs_next[2];

    // 63..69: sidechain chi1 harmonics on non-Gly/Ala/Pro residues
    if !matches!(res_name, "GLY" | "ALA" | "PRO") {
        if let Some(chi1) = chi1_i {
            let r1 = chi1.to_radians();
            let (sc1, cc1) = det_sin_cos(r1);
            let (_, cc3) = det_sin_cos(3.0 * r1);
            x[63] = 1.0;
            x[64] = cc1;
            x[65] = sc1;
            x[66] = cc3;
            x[67] = hs[1] * cc1;
            x[68] = hs[2] * sc1;
        }
    }

    // 69..73: Kabsch-Sander electrostatic H-bond & secondary-structure cooperativity terms
    x[69] = (-phys.ks_nh_energy).clamp(0.0, 3.5);
    x[70] = (-phys.ks_co_energy).clamp(0.0, 3.5);
    x[71] = phys.is_helical_hbond;
    x[72] = phys.is_sheet_hbond;

    // 73..93: 20-amino-acid baseline residual indicator
    if let Some(a_idx) = aa20_index(aa1) {
        x[FOURIER_BASE_DIM + a_idx] = 1.0;
    }

    x
}

/// Fitted ridge-regularized trigonometric Fourier hypersurface weights across all 6 backbone
/// nuclei and 5 structural residue classes (`General`, `Gly`, `Pro`, `PrePro`, `BetaBranched`).
#[derive(Clone, Debug)]
pub struct FittedTorsionSurfaces {
    /// `weights[atom_idx][class_idx]` in `R^{TOTAL_BASIS_DIM}`.
    pub weights: [[[f64; TOTAL_BASIS_DIM]; 5]; 6],
    /// Out-of-sample / residual RMSD estimates per `(atom_idx, class_idx)` in ppm.
    pub residual_sigma: [[f64; 5]; 6],
}

impl FittedTorsionSurfaces {
    #[inline]
    pub fn eval(
        &self,
        atom: BackboneShiftAtom,
        cls: StructuralResidueClass,
        basis: &[f64; TOTAL_BASIS_DIM],
    ) -> f64 {
        let w = &self.weights[atom.index()][cls.index()];
        let mut s = 0.0;
        for d in 0..TOTAL_BASIS_DIM {
            s += w[d] * basis[d];
        }
        s
    }
}

const PACKED_TRI_LEN: usize = TOTAL_BASIS_DIM * (TOTAL_BASIS_DIM + 1) / 2;

#[derive(Clone)]
struct ChainAtomClassSuffStats {
    atom_idx: u8,
    cls_idx: u8,
    count: usize,
    sum_y2: f64,
    rhs: Box<[f64; TOTAL_BASIS_DIM]>,
    /// Lower-triangular Gram entries `G[p][q]` packed at `p * (p + 1) / 2 + q` (`0 <= q <= p`).
    g_tri: Vec<f64>,
}

struct PrecomputedCorpusChain {
    chain_key: String,
    buckets: Vec<ChainAtomClassSuffStats>,
}

static PRECOMPUTED_CORPUS: OnceLock<Vec<PrecomputedCorpusChain>> = OnceLock::new();

fn precompute_single_corpus_chain(ch: &TorsionCorpusChain) -> PrecomputedCorpusChain {
    let l = ch.residues.len();
    let seq_ids: Vec<i32> = ch.residues.iter().map(|r| r.seq_id).collect();
    let res_names: Vec<String> = ch.residues.iter().map(|r| r.res_name.clone()).collect();
    let chain_ids: Vec<String> = vec![ch.chain_id.clone(); l];
    let ca_coords: Vec<[f64; 3]> = ch.residues.iter().map(|r| r.ca_coord).collect();
    let phi_v: Vec<Option<f64>> = ch.residues.iter().map(|r| r.phi_deg).collect();
    let psi_v: Vec<Option<f64>> = ch.residues.iter().map(|r| r.psi_deg).collect();
    let chi1_v: Vec<Option<f64>> = ch.residues.iter().map(|r| r.chi1_deg).collect();

    let model_opt = StructureModel::from_ca_and_torsions(
        &ch.chain_id,
        &seq_ids,
        &res_names,
        &ca_coords,
        &phi_v,
        &psi_v,
        &chi1_v,
    )
    .ok();
    let phys_vec = model_opt
        .as_ref()
        .map(evaluate_all_physical_terms)
        .unwrap_or_else(|| vec![ResiduePhysicalShifts::default(); l]);

    let mut acc: BTreeMap<(u8, u8), ChainAtomClassSuffStats> = BTreeMap::new();

    for (i, phys) in phys_vec.iter().enumerate() {
        let cr = &ch.residues[i];
        // Skip chain termini lacking internal backbone torsions when fitting interior surfaces
        if cr.phi_deg.is_none() && cr.psi_deg.is_none() {
            continue;
        }
        let next_intact = i + 1 < l && ch.residues[i + 1].seq_id == cr.seq_id + 1;
        let prev_intact = i > 0 && ch.residues[i - 1].seq_id + 1 == cr.seq_id;
        let next_is_pro = next_intact && ch.residues[i + 1].res_name == "PRO";

        let cys_ox = model_opt
            .as_ref()
            .and_then(|m| m.residues.get(i))
            .is_some_and(|mr| mr.cys_oxidized);

        // Reconstruct absolute DSS chemical shift from Wishart-1995 secondary shift in `torsion_corpus.tsv`,
        // then subtract `ncIDP` baseline AND non-local physical shifts (`HB + RC-arom + pep/EF`)!
        let w_rc = random_coil_shifts(&cr.res_name, cys_ox, next_is_pro);
        let wishart_arr: [Option<f64>; 6] = [
            (cr.res_name != "PRO").then_some(w_rc.h),
            (cr.res_name != "PRO").then_some(w_rc.n),
            Some(w_rc.ca),
            (cr.res_name != "GLY").then_some(w_rc.cb),
            Some(w_rc.c),
            Some(w_rc.ha),
        ];
        let ncidp_arr =
            ncidp_random_coil_backbone(&res_names, &chain_ids, &seq_ids, i, cys_ox, 6.5, 298.15);

        let mut targets = [None; 6];
        for a in 0..6 {
            if let (Some(sec_w), Some(w_base), Some(nc_base)) =
                (cr.sec_shifts[a], wishart_arr[a], ncidp_arr[a])
            {
                let obs_dss = w_base + sec_w;
                let non_local = phys.hbond[a] + phys.ring_current[a] + phys.peptide_ef[a];
                let resid = (obs_dss - nc_base) - non_local;
                let max_ok = match a {
                    0 | 5 => 3.2,
                    1 => 18.0,
                    _ => 9.5,
                };
                if resid.abs() <= max_ok {
                    targets[a] = Some(resid);
                }
            }
        }
        if targets.iter().all(Option::is_none) {
            continue;
        }

        let phi_prev = prev_intact.then(|| ch.residues[i - 1].phi_deg).flatten();
        let psi_prev = prev_intact.then(|| ch.residues[i - 1].psi_deg).flatten();
        let phi_next = next_intact.then(|| ch.residues[i + 1].phi_deg).flatten();
        let psi_next = next_intact.then(|| ch.residues[i + 1].psi_deg).flatten();

        let cls = StructuralResidueClass::classify(&cr.res_name, next_is_pro);
        let c_u8 = cls.index() as u8;
        let x = build_torsion_basis_vector(
            cr.aa1,
            &cr.res_name,
            cr.phi_deg,
            cr.psi_deg,
            phi_prev,
            psi_prev,
            phi_next,
            psi_next,
            cr.chi1_deg,
            phys,
        );

        for (a, y_opt) in targets.into_iter().enumerate() {
            let Some(y) = y_opt else { continue };
            let a_u8 = a as u8;
            let st = acc
                .entry((a_u8, c_u8))
                .or_insert_with(|| ChainAtomClassSuffStats {
                    atom_idx: a_u8,
                    cls_idx: c_u8,
                    count: 0,
                    sum_y2: 0.0,
                    rhs: Box::new([0.0_f64; TOTAL_BASIS_DIM]),
                    g_tri: vec![0.0_f64; PACKED_TRI_LEN],
                });
            st.count += 1;
            st.sum_y2 += y * y;
            for p in 0..TOTAL_BASIS_DIM {
                let xp = x[p];
                if xp == 0.0 {
                    continue;
                }
                st.rhs[p] += xp * y;
                let row_off = p * (p + 1) / 2;
                for (q, &xq) in x.iter().enumerate().take(p + 1) {
                    st.g_tri[row_off + q] += xp * xq;
                }
            }
        }
    }

    PrecomputedCorpusChain {
        chain_key: ch.chain_key.clone(),
        buckets: acc.into_values().collect(),
    }
}

fn embedded_precomputed_corpus() -> &'static [PrecomputedCorpusChain] {
    PRECOMPUTED_CORPUS.get_or_init(|| {
        let db = TorsionDb::embedded();
        db.chains
            .iter()
            .map(precompute_single_corpus_chain)
            .collect()
    })
}

/// Solve symmetric positive-definite linear system `A * w = b` (`N = TOTAL_BASIS_DIM`) via Cholesky
/// `A = L L^T`, simultaneously returning the exact inverse diagonal `diag(A^{-1})_d = ||L^{-1} e_d||_2^2`
/// for Golub–Heath–Wahba (1979) Generalized Cross-Validation (`GCV`) out-of-sample degrees of freedom (`21.W1:P2-1`).
#[allow(clippy::needless_range_loop)]
fn solve_spd_cholesky_with_inv_diag(
    a: &[[f64; TOTAL_BASIS_DIM]; TOTAL_BASIS_DIM],
    b: &[f64; TOTAL_BASIS_DIM],
) -> ([f64; TOTAL_BASIS_DIM], [f64; TOTAL_BASIS_DIM]) {
    let mut l = [[0.0_f64; TOTAL_BASIS_DIM]; TOTAL_BASIS_DIM];
    for i in 0..TOTAL_BASIS_DIM {
        for j in 0..=i {
            let mut sum = a[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                l[i][i] = sum.max(1e-9).sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }

    let mut y = [0.0_f64; TOTAL_BASIS_DIM];
    for i in 0..TOTAL_BASIS_DIM {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[i][k] * y[k];
        }
        y[i] = sum / l[i][i];
    }

    let mut w = [0.0_f64; TOTAL_BASIS_DIM];
    for i_rev in 0..TOTAL_BASIS_DIM {
        let i = TOTAL_BASIS_DIM - 1 - i_rev;
        let mut sum = y[i];
        for k in (i + 1)..TOTAL_BASIS_DIM {
            sum -= l[k][i] * w[k];
        }
        w[i] = sum / l[i][i];
    }

    // Lower-triangular inverse `Z = L^{-1}` -> `diag(A^{-1})_d = sum_{i >= d} Z_{i, d}^2`
    let mut z = [[0.0_f64; TOTAL_BASIS_DIM]; TOTAL_BASIS_DIM];
    for d in 0..TOTAL_BASIS_DIM {
        z[d][d] = 1.0 / l[d][d];
        for i in (d + 1)..TOTAL_BASIS_DIM {
            let mut sum = 0.0;
            for k in d..i {
                sum += l[i][k] * z[k][d];
            }
            z[i][d] = -sum / l[i][i];
        }
    }
    let mut inv_diag = [0.0_f64; TOTAL_BASIS_DIM];
    for d in 0..TOTAL_BASIS_DIM {
        let mut sq = 0.0;
        for i in d..TOTAL_BASIS_DIM {
            sq += z[i][d] * z[i][d];
        }
        inv_diag[d] = sq;
    }

    (w, inv_diag)
}

/// Fit hierarchical ridge-regularized trigonometric Fourier hypersurfaces (`K = 3` `(phi_i, psi_i)` +
/// `K' = 2` `(phi_{i-1}, psi_{i-1})` + `chi1` + H-bond/secondary-structure terms) strictly on the
/// `HomologFirewall`-retained template chains (`< 2 ms` via exact additive chain sufficient statistics).
pub fn fit_torsion_fourier_surfaces(
    retained_chains: &[&TorsionCorpusChain],
    lambda_ridge: f64,
) -> FittedTorsionSurfaces {
    let base_lambda = lambda_ridge.max(0.25);
    let embedded_cache = embedded_precomputed_corpus();
    let cache_map: BTreeMap<&str, &PrecomputedCorpusChain> = embedded_cache
        .iter()
        .map(|c| (c.chain_key.as_str(), c))
        .collect();

    let mut owned_fallback: Vec<PrecomputedCorpusChain> = Vec::new();
    for &ch in retained_chains {
        if !cache_map.contains_key(ch.chain_key.as_str()) {
            owned_fallback.push(precompute_single_corpus_chain(ch));
        }
    }
    let fallback_map: BTreeMap<&str, &PrecomputedCorpusChain> = owned_fallback
        .iter()
        .map(|c| (c.chain_key.as_str(), c))
        .collect();

    let selected_chains: Vec<&PrecomputedCorpusChain> = retained_chains
        .iter()
        .filter_map(|ch| {
            cache_map
                .get(ch.chain_key.as_str())
                .or_else(|| fallback_map.get(ch.chain_key.as_str()))
                .copied()
        })
        .collect();

    let mut diag_pen = [base_lambda; TOTAL_BASIS_DIM];
    let deg_map = [0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
    for ip in 0..7 {
        for is_ in 0..7 {
            let harm_deg_sq = deg_map[ip] * deg_map[ip] + deg_map[is_] * deg_map[is_];
            diag_pen[ip * 7 + is_] = base_lambda * (1.0 + 0.45 * harm_deg_sq);
        }
    }
    diag_pen[0] = 0.5 * base_lambda;
    diag_pen[49..63].fill(2.0 * base_lambda);
    diag_pen[63..69].fill(1.8 * base_lambda);
    diag_pen[69..73].fill(1.5 * base_lambda);
    diag_pen[FOURIER_BASE_DIM..TOTAL_BASIS_DIM].fill(3.5 * base_lambda);

    // Aggregate sufficient statistics strictly across `retained_chains` (heap-allocated for wasm32 stack safety)
    let mut g_unreg = vec![[[0.0_f64; TOTAL_BASIS_DIM]; TOTAL_BASIS_DIM]; 6 * 5];
    let mut b_unreg = [[[0.0_f64; TOTAL_BASIS_DIM]; 5]; 6];
    let mut sum_y2 = [[0.0_f64; 5]; 6];
    let mut counts = [[0usize; 5]; 6];

    for pch in &selected_chains {
        for bkt in &pch.buckets {
            let a = bkt.atom_idx as usize;
            let c = bkt.cls_idx as usize;
            counts[a][c] += bkt.count;
            sum_y2[a][c] += bkt.sum_y2;
            let rhs_dst = &mut b_unreg[a][c];
            let g_dst = &mut g_unreg[a * 5 + c];
            for (p, row_dst) in g_dst.iter_mut().enumerate() {
                rhs_dst[p] += bkt.rhs[p];
                let row_off = p * (p + 1) / 2;
                for (q, cell) in row_dst.iter_mut().enumerate().take(p + 1) {
                    *cell += bkt.g_tri[row_off + q];
                }
            }
        }
    }

    let mut weights = [[[0.0_f64; TOTAL_BASIS_DIM]; 5]; 6];
    let mut residual_sigma = [[1.0_f64; 5]; 6];
    let class_shrink = [1.8, 3.2, 4.5, 4.5, 2.4];

    for atom in BackboneShiftAtom::ALL {
        let a_idx = atom.index();
        // Symmetrize unregularized class Gram matrices and sum into pooled global system
        let mut g_glob = [[0.0_f64; TOTAL_BASIS_DIM]; TOTAL_BASIS_DIM];
        let mut b_glob = [0.0_f64; TOTAL_BASIS_DIM];
        let mut n_tot_atom = 0usize;
        for d in 0..TOTAL_BASIS_DIM {
            g_glob[d][d] = diag_pen[d];
        }
        for c_idx in 0..5 {
            n_tot_atom += counts[a_idx][c_idx];
            let gc_mut = &mut g_unreg[a_idx * 5 + c_idx];
            for p in 0..TOTAL_BASIS_DIM {
                b_glob[p] += b_unreg[a_idx][c_idx][p];
                for q in 0..=p {
                    let v = gc_mut[p][q];
                    gc_mut[q][p] = v;
                    g_glob[p][q] += v;
                    if p != q {
                        g_glob[q][p] += v;
                    }
                }
            }
        }
        let (w_glob, inv_diag_glob) = solve_spd_cholesky_with_inv_diag(&g_glob, &b_glob);
        // Effective global degrees of freedom `tr((G_glob + Lambda)^{-1} G_glob) = D - sum_d inv_diag_glob[d] * diag_pen[d]`
        let mut nu_glob = TOTAL_BASIS_DIM as f64;
        for d in 0..TOTAL_BASIS_DIM {
            nu_glob -= inv_diag_glob[d] * diag_pen[d];
        }
        nu_glob = nu_glob.max(0.0);

        // Class-specific hierarchical ridge solve: `(G_c + lambda_c I) dw_c = b_c - G_c w_glob`
        let base_sd = atom.struct_sigma_ppm();
        for cls in StructuralResidueClass::ALL {
            let c_idx = cls.index();
            let gc_raw = &g_unreg[a_idx * 5 + c_idx];
            let bc_raw = &b_unreg[a_idx][c_idx];
            let mut gc_reg = *gc_raw;
            let mut rhs_res = [0.0_f64; TOTAL_BASIS_DIM];
            for p in 0..TOTAL_BASIS_DIM {
                gc_reg[p][p] += diag_pen[p] * class_shrink[c_idx];
                let mut gw_p = 0.0;
                for q in 0..TOTAL_BASIS_DIM {
                    gw_p += gc_raw[p][q] * w_glob[q];
                }
                rhs_res[p] = bc_raw[p] - gw_p;
            }
            let (dw, inv_diag_cls) = solve_spd_cholesky_with_inv_diag(&gc_reg, &rhs_res);
            let mut w_cls = [0.0_f64; TOTAL_BASIS_DIM];
            for d in 0..TOTAL_BASIS_DIM {
                w_cls[d] = w_glob[d] + dw[d];
            }
            weights[a_idx][c_idx] = w_cls;

            // Golub–Heath–Wahba (1979) Generalized Cross-Validation (`GCV` / out-of-sample LOO PRESS)
            // predictive residual RMSD (`21.W1:P2-1`):
            //   nu_eff = tr((G_c + Lambda_c)^{-1} G_c) + (N_c / N_tot) * nu_glob_shrunk
            //   sigma_GCV = sqrt(RSS_c / N_c) / (1 - nu_eff / N_c)
            let n_c = counts[a_idx][c_idx];
            if n_c >= 25 {
                let mut wb = 0.0;
                let mut wgw = 0.0;
                let mut nu_cls = TOTAL_BASIS_DIM as f64;
                for p in 0..TOTAL_BASIS_DIM {
                    wb += w_cls[p] * bc_raw[p];
                    let mut gp = 0.0;
                    for q in 0..TOTAL_BASIS_DIM {
                        gp += gc_raw[p][q] * w_cls[q];
                    }
                    wgw += w_cls[p] * gp;
                    nu_cls -= inv_diag_cls[p] * (diag_pen[p] * class_shrink[c_idx]);
                }
                let share_glob = if n_tot_atom > 0 {
                    ((n_c as f64) / (n_tot_atom as f64)) * nu_glob
                } else {
                    0.0
                };
                let nu_eff = (nu_cls.max(0.0) + 0.5 * share_glob).min(0.75 * (n_c as f64));
                let rss = (sum_y2[a_idx][c_idx] - 2.0 * wb + wgw).max(0.0);
                let gcv_denom = (1.0 - nu_eff / (n_c as f64)).max(0.25);
                let gcv_oos_rms = (rss / (n_c as f64)).sqrt() / gcv_denom;
                residual_sigma[a_idx][c_idx] = (0.60 * gcv_oos_rms + 0.40 * base_sd).max(base_sd);
            } else {
                residual_sigma[a_idx][c_idx] = base_sd;
            }
        }
    }

    FittedTorsionSurfaces {
        weights,
        residual_sigma,
    }
}

// ============================================================================
// 6. Sahakyan et al. (2011) `ILVMAT` Sidechain Methyl (`13C^m`, `1H^m`) Forward Model
// ============================================================================

/// Smooth continuous gamma-gauche substituent indicator:
/// - `0.0` at trans (`theta = +-180 deg`),
/// - `1.0` at staggered gauche (`theta = +-60 deg`).
#[inline]
fn gamma_gauche_indicator(theta_deg: f64) -> f64 {
    let (_, c) = det_sin_cos(theta_deg.to_radians());
    ((1.0 + c) / 1.5).clamp(0.0, 1.25)
}

#[allow(clippy::too_many_arguments)]
fn predict_residue_ilvmat_methyls(
    model: &StructureModel,
    pep_dipoles: &[PeptideDipole],
    r_idx: usize,
    ala_cb_pred: Option<&ResidueAtomShiftBreakdown>,
    w_af: f64,
    cfg: &StructShiftConfig,
) -> Vec<MethylShiftPrediction> {
    let r = &model.residues[r_idx];
    let mut out = Vec::new();
    let dt = (cfg.temperature_kelvin - 298.15).clamp(-40.0, 60.0);

    for kind in MethylGroupKind::ALL {
        if kind.res_name() != r.res_name {
            continue;
        }
        let c_name = kind.carbon_atom();
        let h_names = kind.proton_atoms();
        let h_label = match kind {
            MethylGroupKind::IleCd1 | MethylGroupKind::LeuCd1 => "HD1*",
            MethylGroupKind::LeuCd2 => "HD2*",
            MethylGroupKind::ValCg1 => "HG1*",
            MethylGroupKind::ValCg2 | MethylGroupKind::ThrCg2 => "HG2*",
            MethylGroupKind::MetCe => "HE*",
            MethylGroupKind::AlaCb => "HB*",
        };

        let Some(&c_xyz) = r.atoms.get(c_name) else {
            continue;
        };
        let h_pts: Vec<Vec3> = h_names
            .iter()
            .filter_map(|&hn| r.atoms.get(hn).copied())
            .collect();
        if h_pts.is_empty() {
            continue;
        }

        // 3-proton rotamer-averaged Haigh-Mallion aromatic ring-current & peptide anisotropy
        let mut h_rc_arom = 0.0;
        let mut h_pep_ef = 0.0;
        let mut nearest_arom: Option<f64> = None;
        for &hp in &h_pts {
            let (rc_p, min_d) =
                evaluate_haigh_mallion_ring_current(&model.aromatic_rings, hp, r_idx, false, 1.0);
            h_rc_arom += rc_p;
            if let Some(d) = min_d {
                nearest_arom = Some(match nearest_arom {
                    Some(cur) => cur.min(d),
                    None => d,
                });
            }
            let u_ch = v_unit(v_sub(hp, c_xyz));
            let (aniso, ef) =
                eval_peptide_aniso_and_ef(pep_dipoles, hp, u_ch, &r.chain_id, r.seq_id, false);
            h_pep_ef += aniso + ef;
        }
        let inv_nh = 1.0 / (h_pts.len() as f64);
        h_rc_arom = (h_rc_arom * inv_nh).clamp(-1.85, 1.25);
        h_pep_ef = (h_pep_ef * inv_nh).clamp(-0.35, 0.35);

        let (mut c_rc_arom, _) =
            evaluate_haigh_mallion_ring_current(&model.aromatic_rings, c_xyz, r_idx, false, 0.55);
        c_rc_arom = c_rc_arom.clamp(-0.90, 0.90);
        if !cfg.enable_ring_current {
            h_rc_arom = 0.0;
            c_rc_arom = 0.0;
        }
        if !cfg.enable_peptide_ef {
            h_pep_ef = 0.0;
        }

        // Sahakyan et al. (2011) 13C^m gamma-gauche rotamer shielding & coil baselines
        let (c_rc_base, c_gg, h_rc_base, h_rot_adj, sd_c_struct, sd_c_coil, sd_h_struct, sd_h_coil): (
            f64,
            f64,
            f64,
            f64,
            f64,
            f64,
            f64,
            f64,
        ) = match kind {
                MethylGroupKind::AlaCb => {
                    let (c_rc, c_tor) = match ala_cb_pred {
                        Some(cb_b) => (cb_b.rc_ppm, cb_b.delta_torsion_ppm),
                        None => (19.07 + 0.0030 * dt, 0.0),
                    };
                    let h_tor = r
                        .psi_deg
                        .map(|psi| -0.05 * det_sin_cos(psi.to_radians()).1)
                        .unwrap_or(0.0);
                    (
                        c_rc,
                        c_tor,
                        1.38 - 0.0010 * dt,
                        h_tor,
                        1.15,
                        1.65,
                        0.19,
                        0.26,
                    )
                }
                MethylGroupKind::IleCd1 => {
                    // Two gamma-heavy atoms across CB-CG1: CA and CG2 (coil ~1.25 gauche -> 13.45 ppm)
                    let mut n_gauche = 1.25;
                    if let (Some(cb_v), Some(&cg1)) = (r.cb, r.atoms.get("CG1")) {
                        let th_ca = calc_dihedral_deg(r.ca, cb_v, cg1, c_xyz);
                        let g_ca = gamma_gauche_indicator(th_ca);
                        let g_cg2 = if let Some(&cg2) = r.atoms.get("CG2") {
                            gamma_gauche_indicator(calc_dihedral_deg(cg2, cb_v, cg1, c_xyz))
                        } else {
                            1.0
                        };
                        n_gauche = g_ca + g_cg2;
                    }
                    let c_rc = 13.45 + 0.0020 * dt;
                    let c_gg = -4.20 * (n_gauche - 1.25);
                    let h_rot = -0.06 * (n_gauche - 1.25);
                    (
                        c_rc,
                        c_gg,
                        0.85 - 0.0010 * dt,
                        h_rot,
                        1.18,
                        1.55,
                        0.18,
                        0.24,
                    )
                }
                MethylGroupKind::LeuCd1 | MethylGroupKind::LeuCd2 => {
                    // Single gamma-heavy atom across CB-CG: CA (~0.45 coil gauche fraction)
                    let is_cd1 = kind == MethylGroupKind::LeuCd1;
                    let c_rc = (if is_cd1 { 24.45 } else { 23.95 }) + 0.0020 * dt;
                    let g_ca = match (r.cb, r.atoms.get("CG")) {
                        (Some(cb_v), Some(&cg)) => {
                            gamma_gauche_indicator(calc_dihedral_deg(r.ca, cb_v, cg, c_xyz))
                        }
                        _ => 0.50,
                    };
                    let ref_g = if is_cd1 { 0.38 } else { 0.48 };
                    let c_gg = -4.75 * (g_ca - ref_g);
                    let h_rc = (if is_cd1 { 0.88 } else { 0.85 }) - 0.0010 * dt;
                    let h_rot = -0.05 * (g_ca - ref_g);
                    (c_rc, c_gg, h_rc, h_rot, 1.22, 1.65, 0.18, 0.25)
                }
                MethylGroupKind::ValCg1 | MethylGroupKind::ValCg2 => {
                    // Two backbone gamma-heavy atoms across CA-CB: N and C'
                    let is_cg1 = kind == MethylGroupKind::ValCg1;
                    let c_rc = (if is_cg1 { 21.45 } else { 20.95 }) + 0.0020 * dt;
                    let (g_n, g_co) = match r.cb {
                        Some(cb_v) => (
                            gamma_gauche_indicator(calc_dihedral_deg(r.n, r.ca, cb_v, c_xyz)),
                            gamma_gauche_indicator(calc_dihedral_deg(r.c, r.ca, cb_v, c_xyz)),
                        ),
                        None => (0.5, 0.5),
                    };
                    // Gamma-N shields ~ -3.1 ppm, gamma-C' shields ~ -2.1 ppm relative to trans
                    let eff_g = 3.10 * g_n + 2.10 * g_co;
                    let ref_eff = if is_cg1 { 2.45 } else { 2.85 };
                    let c_gg = -(eff_g - ref_eff);
                    let h_rc = (if is_cg1 { 0.91 } else { 0.88 }) - 0.0010 * dt;
                    let h_rot = -0.03 * ((g_n + g_co) - 1.05);
                    (c_rc, c_gg, h_rc, h_rot, 1.15, 1.48, 0.18, 0.24)
                }
                MethylGroupKind::ThrCg2 => {
                    let c_rc = 21.55 + 0.0020 * dt;
                    let (g_n, g_co) = match r.cb {
                        Some(cb_v) => (
                            gamma_gauche_indicator(calc_dihedral_deg(r.n, r.ca, cb_v, c_xyz)),
                            gamma_gauche_indicator(calc_dihedral_deg(r.c, r.ca, cb_v, c_xyz)),
                        ),
                        None => (0.5, 0.5),
                    };
                    let c_gg = -(1.65 * g_n + 1.25 * g_co - 1.55);
                    let h_rc = 1.19 - 0.0010 * dt;
                    (c_rc, c_gg, h_rc, 0.0, 1.02, 1.22, 0.17, 0.21)
                }
                MethylGroupKind::MetCe => {
                    // Chi3 = dihedral(CB, CG, SD, CE): gauche (~+-70 deg) shields 13CE by ~-2.6 ppm vs trans
                    let g_cb = match (r.cb, r.atoms.get("CG"), r.atoms.get("SD")) {
                        (Some(cb_v), Some(&cg), Some(&sd)) => {
                            gamma_gauche_indicator(calc_dihedral_deg(cb_v, cg, sd, c_xyz))
                        }
                        _ => 0.78,
                    };
                    let c_rc = 17.05 + 0.0020 * dt;
                    let c_gg = -2.65 * (g_cb - 0.78);
                    let h_rc = 2.06 - 0.0010 * dt;
                    (c_rc, c_gg, h_rc, 0.0, 1.15, 1.48, 0.19, 0.26)
                }
            };

        let c_rc = c_rc_base;
        let c_struct = c_rc_base + c_gg + c_rc_arom;
        let h_struct = h_rc_base + h_rot_adj + h_rc_arom + h_pep_ef;

        let c_pred = w_af * c_struct + (1.0 - w_af) * c_rc;
        let h_pred = w_af * h_struct + (1.0 - w_af) * h_rc_base;
        let w2 = w_af * w_af;
        let eff_sd_c_coil = sd_c_coil.max(sd_c_struct);
        let eff_sd_h_coil = sd_h_coil.max(sd_h_struct);
        let c_sigma =
            (w2 * sd_c_struct * sd_c_struct + (1.0 - w2) * eff_sd_c_coil * eff_sd_c_coil).sqrt();
        let h_sigma =
            (w2 * sd_h_struct * sd_h_struct + (1.0 - w2) * eff_sd_h_coil * eff_sd_h_coil).sqrt();

        out.push(MethylShiftPrediction {
            kind,
            carbon_atom: c_name.to_string(),
            proton_atom: h_label.to_string(),
            c_ppm: c_pred,
            c_sigma_ppm: c_sigma,
            c_struct_ppm: c_struct,
            c_rc_ppm: c_rc_base,
            c_gamma_gauche_ppm: c_gg,
            c_ring_current_ppm: c_rc_arom,
            h_ppm: h_pred,
            h_sigma_ppm: h_sigma,
            h_struct_ppm: h_struct,
            h_rc_ppm: h_rc_base,
            h_ring_current_ppm: h_rc_arom,
            h_peptide_ef_ppm: h_pep_ef,
            nearest_aromatic_dist_angstrom: nearest_arom,
        });
    }

    out
}

// ============================================================================
// 7. Public Prediction Pipeline & Export Formats (`.str`, `.nef`, `.tsv`, `.json`)
// ============================================================================

/// Predict 3D structure chemical shifts enforcing the default 4-layer `HomologFirewall`
/// (`max_seq_identity <= 0.25`, `TM-score < 0.50` against `structure`'s `CA` trace).
pub fn predict_structure_shifts(
    structure: &StructureModel,
    cfg: &StructShiftConfig,
) -> Result<StructShiftPredictionReport, String> {
    predict_structure_shifts_with_firewall(structure, &HomologFirewallQuery::default(), cfg)
}

/// Predict 3D structure chemical shifts with an explicit [`HomologFirewallQuery`], automatically
/// populating `query_ca_coords` from `structure`'s `CA` trace so `Layer 4` (`TM-score < 0.50`)
/// is always enforced.
pub fn predict_structure_shifts_with_firewall(
    structure: &StructureModel,
    firewall_query: &HomologFirewallQuery,
    cfg: &StructShiftConfig,
) -> Result<StructShiftPredictionReport, String> {
    predict_structure_shifts_with_db(TorsionDb::embedded(), structure, firewall_query, cfg)
}

/// Predict 3D structure chemical shifts against a specific [`TorsionDb`] corpus under the
/// 4-layer [`HomologFirewall`].
pub fn predict_structure_shifts_with_db(
    db: &TorsionDb,
    structure: &StructureModel,
    firewall_query: &HomologFirewallQuery,
    cfg: &StructShiftConfig,
) -> Result<StructShiftPredictionReport, String> {
    if structure.residues.is_empty() {
        return Err("cannot predict shifts for an empty StructureModel".to_string());
    }
    if !cfg.sample_ph.is_finite() || !cfg.temperature_kelvin.is_finite() {
        return Err("sample_ph and temperature_kelvin must be finite".to_string());
    }

    // Build effective 4-layer HomologFirewall query with structure's C_alpha trace pinned in Layer 4
    // (requiring >= 8 CA atoms so Zhang-Skolnick sliding-window superposition is well-defined; `21.W1:P1-4`).
    let mut eff_query = firewall_query.clone();
    if eff_query.query_ca_coords.is_none() {
        eff_query.query_ca_coords = Some(structure.residues.iter().map(|r| r.ca).collect());
    }
    if eff_query
        .query_ca_coords
        .as_ref()
        .is_some_and(|ca| ca.len() < 8)
    {
        return Err(format!(
            "HomologFirewall Layer 4 (TM-score < 0.50) requires query structure with >= 8 CA atoms, got {} (fail-closed; 21.W1:P1-4)",
            eff_query.query_ca_coords.as_ref().map_or(0, Vec::len)
        ));
    }
    let query_seq1: Vec<u8> = structure.residues.iter().map(|r| r.aa1).collect();

    // Filter out database template chains shorter than 8 CA atoms ONLY (never disable Layer 4
    // across the rest of the database! `21.W1:P1-4`).
    let mut l4_short_templates_skipped = 0usize;
    let eligible_chains: Vec<TorsionCorpusChain> = db
        .chains
        .iter()
        .filter(|c| {
            if c.residues.len() < 8 {
                l4_short_templates_skipped += 1;
                false
            } else {
                true
            }
        })
        .cloned()
        .collect();
    let eligible_db = TorsionDb {
        chains: eligible_chains,
    };

    let firewall = HomologFirewall::new(eff_query).with_max_seq_identity(cfg.max_seq_identity);
    let (retained_chains, excluded_details, exclusion_list_sha256, max_retained_seq_identity) =
        firewall.apply(&eligible_db, &query_seq1);
    if retained_chains.is_empty() {
        return Err(
            "HomologFirewall excluded all template chains; cannot fit empirical torsion surfaces"
                .to_string(),
        );
    }

    let surfaces = fit_torsion_fourier_surfaces(&retained_chains, cfg.ridge_lambda);
    let phys_all = evaluate_all_physical_terms(structure);

    let l = structure.residues.len();
    let seq_names: Vec<String> = structure
        .residues
        .iter()
        .map(|r| r.res_name.clone())
        .collect();
    let chain_ids: Vec<String> = structure
        .residues
        .iter()
        .map(|r| r.chain_id.clone())
        .collect();
    let seq_ids: Vec<i32> = structure.residues.iter().map(|r| r.seq_id).collect();

    let pep_dipoles: Vec<PeptideDipole> = if cfg.include_methyl_ilvmat {
        build_peptide_dipoles(structure)
    } else {
        Vec::new()
    };

    let mut out_residues = Vec::with_capacity(l);
    for (i, phys) in phys_all.iter().enumerate() {
        let r = &structure.residues[i];
        let next_intact = i + 1 < l
            && structure.residues[i + 1].chain_id == r.chain_id
            && structure.residues[i + 1].seq_id == r.seq_id + 1;
        let prev_intact = i > 0
            && structure.residues[i - 1].chain_id == r.chain_id
            && structure.residues[i - 1].seq_id + 1 == r.seq_id;
        let next_is_pro = next_intact && structure.residues[i + 1].res_name == "PRO";

        let rc_arr = ncidp_random_coil_backbone(
            &seq_names,
            &chain_ids,
            &seq_ids,
            i,
            r.cys_oxidized,
            cfg.sample_ph,
            cfg.temperature_kelvin,
        );

        let phi_prev = prev_intact
            .then(|| structure.residues[i - 1].phi_deg)
            .flatten();
        let psi_prev = prev_intact
            .then(|| structure.residues[i - 1].psi_deg)
            .flatten();
        let phi_next = next_intact
            .then(|| structure.residues[i + 1].phi_deg)
            .flatten();
        let psi_next = next_intact
            .then(|| structure.residues[i + 1].psi_deg)
            .flatten();

        let cls = StructuralResidueClass::classify(&r.res_name, next_is_pro);
        let basis = build_torsion_basis_vector(
            r.aa1,
            &r.res_name,
            r.phi_deg,
            r.psi_deg,
            phi_prev,
            psi_prev,
            phi_next,
            psi_next,
            r.chi1_deg,
            phys,
        );

        let eff_plddt = cfg.override_uniform_plddt.or(r.plddt);
        let w_af = match eff_plddt {
            Some(p) => af_plddt_confidence_weight(p),
            None => 1.0,
        };
        let w_af_sq = w_af * w_af;

        let mut bb_map = BTreeMap::new();
        for atom in BackboneShiftAtom::ALL {
            let a_idx = atom.index();
            let Some(rc_ppm) = rc_arr[a_idx] else {
                continue;
            };
            // On chain termini lacking both phi and psi, structural torsion delta is damped
            let has_local_torsion = r.phi_deg.is_some() || r.psi_deg.is_some();
            let d_tor = if has_local_torsion {
                surfaces.eval(atom, cls, &basis)
            } else {
                0.0
            };
            let d_hb = if cfg.enable_hbond {
                phys.hbond[a_idx]
            } else {
                0.0
            };
            let d_rc_arom = if cfg.enable_ring_current {
                phys.ring_current[a_idx]
            } else {
                0.0
            };
            let d_pep = if cfg.enable_peptide_ef {
                phys.peptide_ef[a_idx]
            } else {
                0.0
            };

            let struct_ppm = rc_ppm + d_tor + d_hb + d_rc_arom + d_pep;
            let pred_ppm = w_af * struct_ppm + (1.0 - w_af) * rc_ppm;

            let sd_struct = surfaces.residual_sigma[a_idx][cls.index()];
            let sd_coil = atom.coil_sigma_ppm().max(sd_struct * 1.15);
            let pred_sigma =
                (w_af_sq * sd_struct * sd_struct + (1.0 - w_af_sq) * sd_coil * sd_coil).sqrt();

            bb_map.insert(
                atom.as_str().to_string(),
                ResidueAtomShiftBreakdown {
                    atom: atom.as_str().to_string(),
                    predicted_ppm: pred_ppm,
                    predicted_sigma_ppm: pred_sigma,
                    struct_ppm,
                    rc_ppm,
                    delta_torsion_ppm: d_tor,
                    delta_hbond_ppm: d_hb,
                    delta_ring_current_ppm: d_rc_arom,
                    delta_peptide_ef_ppm: d_pep,
                },
            );
        }

        let methyls = if cfg.include_methyl_ilvmat {
            predict_residue_ilvmat_methyls(structure, &pep_dipoles, i, bb_map.get("CB"), w_af, cfg)
        } else {
            Vec::new()
        };

        out_residues.push(ResidueShiftPrediction {
            chain_id: r.chain_id.clone(),
            seq_id: r.seq_id,
            res_name: r.res_name.clone(),
            plddt: eff_plddt,
            w_af,
            phi_deg: r.phi_deg,
            psi_deg: r.psi_deg,
            chi1_deg: r.chi1_deg,
            cys_oxidized: r.cys_oxidized,
            nearest_arom_h_dist_angstrom: phys.nearest_arom_h,
            nearest_arom_ha_dist_angstrom: phys.nearest_arom_ha,
            backbone: bb_map,
            methyls,
        });
    }

    let excluded_template_ids: Vec<String> = excluded_details
        .iter()
        .map(|e| e.chain_key.clone())
        .collect();

    Ok(StructShiftPredictionReport {
        residues: out_residues,
        sample_ph: cfg.sample_ph,
        temperature_kelvin: cfg.temperature_kelvin,
        include_methyl_ilvmat: cfg.include_methyl_ilvmat,
        provenance: structure.provenance,
        retained_template_count: retained_chains.len(),
        excluded_template_ids,
        excluded_details,
        exclusion_list_sha256,
        max_retained_seq_identity,
        corpus_sha256: TorsionDb::embedded_sha256(),
        l4_short_templates_skipped,
    })
}

fn build_peptide_dipoles(model: &StructureModel) -> Vec<PeptideDipole> {
    let l = model.residues.len();
    let mut out = Vec::with_capacity(l);
    for k in 0..l {
        let rk = &model.residues[k];
        let next_intact = k + 1 < l
            && model.residues[k + 1].chain_id == rk.chain_id
            && v_dist(rk.c, model.residues[k + 1].n) <= 2.50;
        let e1 = v_unit(v_sub(rk.o, rk.c));
        let in_plane = if next_intact {
            v_sub(model.residues[k + 1].n, rk.c)
        } else {
            v_sub(rk.ca, rk.c)
        };
        let mut e3 = v_cross(e1, in_plane);
        if v_norm(e3) < 1e-9 {
            e3 = ortho_unit(e1);
        } else {
            e3 = v_unit(e3);
        }
        let e2 = v_cross(e3, e1);
        out.push(PeptideDipole {
            chain_id: rk.chain_id.clone(),
            seq_id: rk.seq_id,
            center: v_add(v_scale(rk.c, 0.45), v_scale(rk.o, 0.55)),
            e1,
            e2,
            e3,
            c_pos: rk.c,
            o_pos: rk.o,
            n_next: next_intact.then(|| model.residues[k + 1].n),
            h_next: if next_intact {
                model.residues[k + 1].h
            } else {
                None
            },
        });
    }
    out
}

impl StructShiftPredictionReport {
    /// Convert predicted backbone (and optional `ILVMAT` methyl) chemical shifts into an
    /// [`nmr_core::nmrstar::Reference`].
    pub fn to_reference(&self) -> Reference {
        let sequence: Vec<String> = self.residues.iter().map(|r| r.res_name.clone()).collect();
        let seq_ids: Vec<i32> = self.residues.iter().map(|r| r.seq_id).collect();
        let mut ref_data = Reference::new(sequence, seq_ids);
        for (idx, r) in self.residues.iter().enumerate() {
            for atom in BackboneShiftAtom::ALL {
                let key_name = atom.as_str();
                if let Some(b) = r.backbone.get(key_name) {
                    ref_data
                        .shifts
                        .insert((idx, key_name.to_string()), b.predicted_ppm);
                    ref_data
                        .ambiguity_codes
                        .insert((idx, key_name.to_string()), 1);
                }
            }
            for m in &r.methyls {
                if m.kind != MethylGroupKind::AlaCb {
                    ref_data
                        .shifts
                        .insert((idx, m.carbon_atom.clone()), m.c_ppm);
                    let amb_c = if matches!(
                        m.kind,
                        MethylGroupKind::LeuCd1
                            | MethylGroupKind::LeuCd2
                            | MethylGroupKind::ValCg1
                            | MethylGroupKind::ValCg2
                    ) {
                        2
                    } else {
                        1
                    };
                    ref_data
                        .ambiguity_codes
                        .insert((idx, m.carbon_atom.clone()), amb_c);
                }
                for hn in m.kind.proton_atoms() {
                    ref_data.shifts.insert((idx, hn.to_string()), m.h_ppm);
                    ref_data.ambiguity_codes.insert((idx, hn.to_string()), 1);
                }
            }
        }
        ref_data
    }

    /// Render predicted shifts as an NMR-STAR v3 string recording `exclusion_list_sha256` and
    /// `max_retained_seq_identity` in the header comment block.
    pub fn to_nmrstar_str(&self) -> String {
        let mut lines = vec![
            format!(
                "# FERRIC predict-shifts (nmr-core::struct_shifts v{})",
                crate::VERSION
            ),
            format!("# exclusion_list_sha256: {}", self.exclusion_list_sha256),
            format!(
                "# max_retained_seq_identity: {:.4}",
                self.max_retained_seq_identity
            ),
            format!("# corpus_sha256: {}", self.corpus_sha256),
            format!(
                "# sample_ph: {:.2}  temperature_kelvin: {:.2}",
                self.sample_ph, self.temperature_kelvin
            ),
            String::new(),
            "data_ferric_predicted_shifts".to_string(),
            String::new(),
            "save_entity_1".to_string(),
            "   _Entity.Sf_category  entity".to_string(),
            "   _Entity.ID           1".to_string(),
            String::new(),
            "   loop_".to_string(),
            "      _Entity_poly_seq.Entity_ID".to_string(),
            "      _Entity_poly_seq.Num".to_string(),
            "      _Entity_poly_seq.Mon_ID".to_string(),
            String::new(),
        ];
        for r in &self.residues {
            lines.push(format!("      1 {:5} {:3}", r.seq_id, r.res_name));
        }
        lines.extend([
            "   stop_".to_string(),
            "save_".to_string(),
            String::new(),
            "save_assigned_chem_shift_list_1".to_string(),
            "   _Assigned_chem_shift_list.Sf_category  assigned_chemical_shifts".to_string(),
            String::new(),
            "   loop_".to_string(),
            "      _Atom_chem_shift.ID".to_string(),
            "      _Atom_chem_shift.Entity_ID".to_string(),
            "      _Atom_chem_shift.Seq_ID".to_string(),
            "      _Atom_chem_shift.Comp_ID".to_string(),
            "      _Atom_chem_shift.Atom_ID".to_string(),
            "      _Atom_chem_shift.Atom_type".to_string(),
            "      _Atom_chem_shift.Val".to_string(),
            "      _Atom_chem_shift.Val_err".to_string(),
            "      _Atom_chem_shift.Ambiguity_code".to_string(),
            "      _Atom_chem_shift.Details".to_string(),
            String::new(),
        ]);

        let mut row_id = 1usize;
        for r in &self.residues {
            for atom in BackboneShiftAtom::ALL {
                let Some(b) = r.backbone.get(atom.as_str()) else {
                    continue;
                };
                let atype = b
                    .atom
                    .chars()
                    .find(|c| c.is_ascii_alphabetic())
                    .unwrap_or('X');
                lines.push(format!(
                    "      {:5} 1 {:5} {:3} {:4} {} {:9.3} {:7.3} 1 w_af={:.3}",
                    row_id,
                    r.seq_id,
                    r.res_name,
                    b.atom,
                    atype,
                    b.predicted_ppm,
                    b.predicted_sigma_ppm,
                    r.w_af
                ));
                row_id += 1;
            }
            for m in &r.methyls {
                if m.kind != MethylGroupKind::AlaCb {
                    lines.push(format!(
                        "      {:5} 1 {:5} {:3} {:4} C {:9.3} {:7.3} 1 w_af={:.3}",
                        row_id, r.seq_id, r.res_name, m.carbon_atom, m.c_ppm, m.c_sigma_ppm, r.w_af
                    ));
                    row_id += 1;
                }
                for hn in m.kind.proton_atoms() {
                    lines.push(format!(
                        "      {:5} 1 {:5} {:3} {:4} H {:9.3} {:7.3} 1 w_af={:.3}",
                        row_id, r.seq_id, r.res_name, hn, m.h_ppm, m.h_sigma_ppm, r.w_af
                    ));
                    row_id += 1;
                }
            }
        }

        lines.push("   stop_".to_string());
        lines.push("save_".to_string());
        lines.push(String::new());
        lines.join("\n")
    }

    /// Build a CCPN NEF v1.1 [`NefDocument`] carrying predicted shifts and predictive uncertainties.
    pub fn to_nef_document(&self, entry_name: &str) -> NefDocument {
        let ref_data = self.to_reference();
        let chain_code = self
            .residues
            .first()
            .map(|r| r.chain_id.as_str())
            .unwrap_or("A");
        let mut doc = NefDocument::new(entry_name);
        doc.molecular_system = NefMolecularSystem::from_reference(&ref_data, chain_code);

        let mut shifts = Vec::new();
        for r in &self.residues {
            for atom in BackboneShiftAtom::ALL {
                let Some(b) = r.backbone.get(atom.as_str()) else {
                    continue;
                };
                let (el, iso) = infer_element_and_isotope(&b.atom);
                shifts.push(NefAssignedShift {
                    chain_code: r.chain_id.clone(),
                    sequence_code: Some(r.seq_id),
                    residue_name: r.res_name.clone(),
                    atom_name: b.atom.clone(),
                    element: el,
                    isotope_number: iso,
                    value: b.predicted_ppm,
                    value_uncertainty: Some(b.predicted_sigma_ppm),
                });
            }
            for m in &r.methyls {
                if m.kind != MethylGroupKind::AlaCb {
                    shifts.push(NefAssignedShift {
                        chain_code: r.chain_id.clone(),
                        sequence_code: Some(r.seq_id),
                        residue_name: r.res_name.clone(),
                        atom_name: m.carbon_atom.clone(),
                        element: Some("C".to_string()),
                        isotope_number: Some(13),
                        value: m.c_ppm,
                        value_uncertainty: Some(m.c_sigma_ppm),
                    });
                }
                let nef_h = normalize_nef_atom_name(&r.res_name, &m.proton_atom);
                shifts.push(NefAssignedShift {
                    chain_code: r.chain_id.clone(),
                    sequence_code: Some(r.seq_id),
                    residue_name: r.res_name.clone(),
                    atom_name: nef_h,
                    element: Some("H".to_string()),
                    isotope_number: Some(1),
                    value: m.h_ppm,
                    value_uncertainty: Some(m.h_sigma_ppm),
                });
            }
        }

        doc.shift_lists.push(NefShiftList {
            name: "predicted".to_string(),
            atom_chem_shift_units: "ppm".to_string(),
            shifts,
            shift_insertion_codes: BTreeMap::new(),
        });
        doc
    }

    /// Render predicted shifts as a CCPN NEF v1.1 string with `HomologFirewall` provenance header.
    pub fn to_nef_str(&self, entry_name: &str) -> String {
        let doc = self.to_nef_document(entry_name);
        format!(
            "# exclusion_list_sha256: {}\n# max_retained_seq_identity: {:.4}\n{}",
            self.exclusion_list_sha256,
            self.max_retained_seq_identity,
            write_nef_str(&doc)
        )
    }

    /// Render predicted shifts and per-atom physical breakdown as a deterministic TSV table.
    pub fn to_tsv_str(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "# exclusion_list_sha256: {}\n",
            self.exclusion_list_sha256
        ));
        out.push_str(&format!(
            "# max_retained_seq_identity: {:.4}\n",
            self.max_retained_seq_identity
        ));
        out.push_str(&format!(
            "# retained_templates: {}\tl4_short_templates_skipped: {}\tcorpus_sha256: {}\n",
            self.retained_template_count, self.l4_short_templates_skipped, self.corpus_sha256
        ));
        out.push_str(&format!(
            "# sample_ph: {:.2}\ttemperature_kelvin: {:.2}\tprovenance: {}\n",
            self.sample_ph,
            self.temperature_kelvin,
            self.provenance.as_str()
        ));
        out.push_str(
            "chain_id\tseq_id\tres_name\tatom\tpredicted_ppm\tsigma_ppm\tplddt\tw_af\tstruct_ppm\trc_ppm\tdelta_torsion_ppm\tdelta_hbond_ppm\tdelta_ring_current_ppm\tdelta_peptide_ef_ppm\n",
        );

        for r in &self.residues {
            let plddt_str = r
                .plddt
                .map(|p| format!("{p:.2}"))
                .unwrap_or_else(|| ".".to_string());
            for atom in BackboneShiftAtom::ALL {
                let Some(b) = r.backbone.get(atom.as_str()) else {
                    continue;
                };
                out.push_str(&format!(
                    "{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{}\t{:.6}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\n",
                    r.chain_id,
                    r.seq_id,
                    r.res_name,
                    b.atom,
                    b.predicted_ppm,
                    b.predicted_sigma_ppm,
                    plddt_str,
                    r.w_af,
                    b.struct_ppm,
                    b.rc_ppm,
                    b.delta_torsion_ppm,
                    b.delta_hbond_ppm,
                    b.delta_ring_current_ppm,
                    b.delta_peptide_ef_ppm
                ));
            }
            for m in &r.methyls {
                if m.kind != MethylGroupKind::AlaCb {
                    out.push_str(&format!(
                        "{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{}\t{:.6}\t{:.4}\t{:.4}\t{:.4}\t0.0000\t{:.4}\t0.0000\n",
                        r.chain_id,
                        r.seq_id,
                        r.res_name,
                        m.carbon_atom,
                        m.c_ppm,
                        m.c_sigma_ppm,
                        plddt_str,
                        r.w_af,
                        m.c_struct_ppm,
                        m.c_rc_ppm,
                        m.c_gamma_gauche_ppm,
                        m.c_ring_current_ppm
                    ));
                }
                out.push_str(&format!(
                    "{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{}\t{:.6}\t{:.4}\t{:.4}\t0.0000\t0.0000\t{:.4}\t{:.4}\n",
                    r.chain_id,
                    r.seq_id,
                    r.res_name,
                    m.proton_atom,
                    m.h_ppm,
                    m.h_sigma_ppm,
                    plddt_str,
                    r.w_af,
                    m.h_struct_ppm,
                    m.h_rc_ppm,
                    m.h_ring_current_ppm,
                    m.h_peptide_ef_ppm
                ));
            }
        }
        out
    }

    /// Render the complete prediction report (including component breakdown & `HomologFirewall`
    /// audit digest) as pretty-printed JSON.
    pub fn to_json_str(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self).map_err(|e| format!("JSON serialization error: {e}"))
    }
}

/// Estimate global chemical-shift reference offsets `(obs - pred)` per backbone nucleus (`[H, N, CA, CB, C', HA]`)
/// strictly from the unordered multiset of observed shifts vs unordered multiset of predicted shifts
/// (`0` residue-assignment labels used; bit-for-bit permutation invariant under `k -> pi(k)`).
pub fn estimate_unlabelled_shift_offsets(
    obs_ref: &Reference,
    pred_report: &StructShiftPredictionReport,
) -> [f64; 6] {
    let quantile = |sorted: &[f64], q: f64| -> f64 {
        let n = sorted.len();
        if n == 0 {
            return 0.0;
        }
        if n == 1 {
            return sorted[0];
        }
        let pos = q.clamp(0.0, 1.0) * ((n - 1) as f64);
        let lo = pos.floor() as usize;
        let hi = pos.ceil() as usize;
        let frac = pos - (lo as f64);
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    };
    let qmatch = |obs: &mut [f64], pred: &mut [f64]| -> f64 {
        if obs.len() < 4 || pred.len() < 4 {
            return 0.0;
        }
        obs.sort_by(f64::total_cmp);
        pred.sort_by(f64::total_cmp);
        let mut diffs = [0.0_f64; 15];
        for (m, slot) in diffs.iter_mut().enumerate() {
            let q = 0.15 + 0.70 * ((m as f64) / 14.0);
            *slot = quantile(obs, q) - quantile(pred, q);
        }
        diffs.sort_by(f64::total_cmp);
        diffs[7]
    };

    type HaShiftTriple = (Option<f64>, Option<f64>, Option<f64>);
    let mut out = [0.0_f64; 6];
    for atom in BackboneShiftAtom::ALL {
        let a_idx = atom.index();
        let aname = atom.as_str();
        let mut obs_vals: Vec<f64> = if aname == "HA" {
            let mut by_res: BTreeMap<usize, HaShiftTriple> = BTreeMap::new();
            for ((ri, at), &v) in &obs_ref.shifts {
                if !v.is_finite() {
                    continue;
                }
                let e = by_res.entry(*ri).or_default();
                match at.as_str() {
                    "HA" => e.0 = Some(v),
                    "HA2" => e.1 = Some(v),
                    "HA3" => e.2 = Some(v),
                    _ => {}
                }
            }
            by_res
                .into_values()
                .filter_map(|(ha, h2, h3)| match (ha, h2, h3) {
                    (Some(v), _, _) => Some(v),
                    (None, Some(a), Some(b)) => Some(0.5 * (a + b)),
                    (None, Some(a), None) => Some(a),
                    (None, None, Some(b)) => Some(b),
                    (None, None, None) => None,
                })
                .collect()
        } else {
            obs_ref
                .shifts
                .iter()
                .filter_map(|((_, at), &v)| (at == aname && v.is_finite()).then_some(v))
                .collect()
        };
        let mut pred_vals: Vec<f64> = pred_report
            .residues
            .iter()
            .filter_map(|r| r.backbone.get(aname).map(|b| b.predicted_ppm))
            .collect();
        out[a_idx] = qmatch(&mut obs_vals, &mut pred_vals);
    }
    out
}

/// Estimate global chemical-shift reference offsets `(obs - pred)` per backbone nucleus
/// using Huber M-estimation over sequence-aligned `(obs, pred)` pairs.
pub fn estimate_shift_reference_offsets(
    obs_ref: &Reference,
    pred_report: &StructShiftPredictionReport,
) -> [f64; 6] {
    let p_keys: Vec<(i32, &str)> = pred_report
        .residues
        .iter()
        .map(|r| (r.seq_id, r.res_name.as_str()))
        .collect();
    let t_keys: Vec<(i32, &str)> = obs_ref
        .seq_ids
        .iter()
        .zip(obs_ref.sequence.iter())
        .map(|(&s, rn)| (s, rn.as_str()))
        .collect();
    let pairs = crate::dihedrals::align_by_res_names(&p_keys, &t_keys);

    let mut offsets = [0.0_f64; 6];
    let huber_k = [0.55, 2.80, 1.40, 1.50, 1.35, 0.40];

    for atom in BackboneShiftAtom::ALL {
        let a_idx = atom.index();
        let aname = atom.as_str();
        let mut diffs = Vec::new();
        for &(pi, ti) in &pairs {
            let obs_opt = obs_ref
                .shifts
                .get(&(ti, aname.to_string()))
                .copied()
                .or_else(|| {
                    (aname == "HA").then(|| {
                        match (
                            obs_ref.shifts.get(&(ti, "HA2".to_string())).copied(),
                            obs_ref.shifts.get(&(ti, "HA3".to_string())).copied(),
                        ) {
                            (Some(a), Some(b)) => Some(0.5 * (a + b)),
                            (Some(a), None) => Some(a),
                            (None, Some(b)) => Some(b),
                            (None, None) => None,
                        }
                    })?
                });
            if let (Some(obs_v), Some(pred_b)) =
                (obs_opt, pred_report.residues[pi].backbone.get(aname))
            {
                if obs_v.is_finite() {
                    diffs.push(obs_v - pred_b.predicted_ppm);
                }
            }
        }
        if diffs.len() >= 4 {
            diffs.sort_by(f64::total_cmp);
            let mut mu = diffs[diffs.len() / 2];
            let k = huber_k[a_idx];
            for _ in 0..8 {
                let mut w_sum = 0.0;
                let mut wx_sum = 0.0;
                for &d in &diffs {
                    let r = (d - mu).abs();
                    let w = if r <= k { 1.0 } else { k / r };
                    w_sum += w;
                    wx_sum += w * d;
                }
                if w_sum > 1e-12 {
                    mu = wx_sum / w_sum;
                }
            }
            offsets[a_idx] = mu;
        }
    }
    offsets
}
