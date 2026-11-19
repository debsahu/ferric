//! Pure-Rust clean-room `LEGOLAS` ANI-AEV neural 3D-coordinate-to-backbone chemical-shift
//! predictor (`H^N`, `15N`, `13CA`, `13CB`, `13C'`, `1HA`), ensemble epistemic uncertainty
//! estimator, versioned `LEGOLAS-V1` SHA-256 weight bundle codec, and analytical-backpropagation
//! Adam trainer (`T22.12`).
//!
//! # Provenance & License Attribution
//! Clean-room original Rust implementation derived strictly from peer-reviewed literature and
//! the MIT-licensed `LEGOLAS` architecture:
//! - Darrows, M. Y., Kodituwakku, D., Xue, J., Pickering, I., Terrel, N. S. & Roitberg, A. E.,
//!   "LEGOLAS: A Machine Learning Method for Rapid and Accurate Predictions of Protein NMR
//!   Chemical Shifts" (*J. Chem. Theory Comput.* **2025**, 21(8), 4266–4275;
//!   DOI: `10.1021/acs.jctc.5c00026`; reference repository:
//!   <https://github.com/roitberg-group/legolas>, **MIT License**) — species- and
//!   nucleus-resolved ANI Atomic Environment Vector (`AEV`) neural ensemble architecture for
//!   biomolecular NMR chemical-shift prediction and ensemble epistemic uncertainty estimation.
//! - Smith, J. S., Isayev, O. & Roitberg, A. E. (*Chem. Sci.* **2017**, 8, 3192–3203;
//!   DOI: `10.1039/C6SC05720A`) — ANI-1 modified Behler–Parrinello radial and angular Atomic
//!   Environment Vector (`AEV`) symmetry functions with smooth cosine cutoffs and shifted
//!   angular basis terms.
//! - Behler, J. & Parrinello, M. (*Phys. Rev. Lett.* **2007**, 98, 146401) — translation-,
//!   rotation-, and permutation-invariant local atomic symmetry-function descriptors.
//! - Kingma, D. P. & Ba, J. (*ICLR* **2015**, arXiv:1412.6980) — Adam first-order stochastic
//!   gradient optimizer.

use std::collections::{BTreeMap, HashMap};
use std::f64::consts::PI;

use serde::{Deserialize, Serialize};

use crate::rng::Rng;
use crate::struct_shifts::{BackboneShiftAtom, StructureModel};
use crate::torsion_db::sha256_hex;

/// Canonical version tag for `LEGOLAS-V1` JSON weight bundles.
pub const LEGOLAS_BUNDLE_FORMAT_V1: &str = "LEGOLAS-V1";

/// 8-byte binary header magic for `LEGOLAS-V1` compact binary weight bundles.
pub const LEGOLAS_BINARY_MAGIC: &[u8; 8] = b"LEGLSV01";

/// Number of supported biomolecular chemical species (`H`, `C`, `N`, `O`, `S`).
pub const NUM_SPECIES: usize = 5;

/// Number of unordered neighbor species pairs `C(5 + 1, 2) = 15`.
pub const NUM_SPECIES_PAIRS: usize = 15;

/// Canonical provenance string carried by untrained / randomly initialized architectures
/// without bundled upstream weights (`22.12:P1-1`).
pub const UNTRAINED_WEIGHTS_PROVENANCE: &str = "untrained-architecture-no-bundled-weights";

/// Hard DoS ceiling on atom count accepted by `LEGOLAS` AEV and shift evaluation.
pub const MAX_LEGOLAS_ATOMS: usize = 65_536;

/// Hard DoS ceiling on radial or angular neighbors per query atom (`22.12:P2-2`), preventing
/// `O(N^2)` three-body angular blowup on dense hostile coordinate inputs.
pub const MAX_LEGOLAS_NEIGHBORS_PER_ATOM: usize = 256;

/// Hard DoS ceiling on atoms occupying a single spatial hash cell (`22.12:P2-2`), preventing
/// coincident-atom bucket-scan quadratic blowup on hostile coordinate inputs.
pub const MAX_LEGOLAS_ATOMS_PER_CELL: usize = 1024;

/// Hard coordinate magnitude cap (Angstroms) preventing integer cell-index overflow on hostile input.
pub const MAX_LEGOLAS_COORD_ANGSTROM: f64 = 1_000_000.0;

/// Hard serialized bundle byte ceiling (`32 MiB`) for JSON and binary `LEGOLAS-V1` loaders.
pub const MAX_LEGOLAS_BUNDLE_BYTES: usize = 32 * 1024 * 1024;

/// Maximum ensemble members per nucleus accepted by bundle validation.
pub const MAX_LEGOLAS_ENSEMBLE_MEMBERS: usize = 64;

/// Maximum dense layers per MLP accepted by bundle validation.
pub const MAX_LEGOLAS_LAYERS: usize = 8;

/// Maximum hidden layer width accepted by bundle validation.
pub const MAX_LEGOLAS_HIDDEN_DIM: usize = 1024;

/// Maximum total AEV descriptor dimension accepted by configuration validation.
pub const MAX_LEGOLAS_AEV_DIM: usize = 4096;

/// Minimum nonzero interatomic distance (Angstroms) guarded against coincident-atom `0/0`.
pub const MIN_PAIR_DIST_ANGSTROM: f64 = 1e-12;

/// Structured errors returned by `LEGOLAS` chemical-shift prediction and validation (`22.12:P1-1`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LegolasError {
    /// Refused to predict chemical shifts from an untrained / randomly initialized model without
    /// verified trained weights.
    UntrainedWeights { provenance: String },
    /// Configuration, coordinate, neighbor-cap, or model validation failed.
    Validation(String),
    /// Weight bundle serialization, deserialization, or SHA-256 verification failed.
    Bundle(String),
    /// WebGPU device-limit or compute dispatch error.
    Gpu(String),
}

impl std::fmt::Display for LegolasError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UntrainedWeights { provenance } => write!(
                f,
                "refusing LEGOLAS chemical-shift prediction with untrained weights (weights_provenance={provenance:?}); load verified trained weights or opt into allow_untrained_for_testing for numerical parity tests"
            ),
            Self::Validation(msg) => write!(f, "LEGOLAS validation error: {msg}"),
            Self::Bundle(msg) => write!(f, "LEGOLAS bundle error: {msg}"),
            Self::Gpu(msg) => write!(f, "LEGOLAS GPU error: {msg}"),
        }
    }
}

impl std::error::Error for LegolasError {}

/// Explicit inference options for `LegolasModel` / `LegolasGpuEngine`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct LegolasPredictOptions {
    /// Explicit test-only opt-in to run forward inference on an untrained / random-initialized
    /// architecture (e.g. for SE(3) or CPU/GPU numerical parity checks).
    ///
    /// Defaults to `false`: standard inference MUST fail closed with
    /// `Err(LegolasError::UntrainedWeights { .. })` when `!model.metadata.weights_trained`.
    pub allow_untrained_for_testing: bool,
}

/// Biomolecular chemical element species resolved by ANI-AEV radial and angular channels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Species {
    H = 0,
    C = 1,
    N = 2,
    O = 3,
    S = 4,
}

impl Species {
    pub const ALL: [Self; NUM_SPECIES] = [Self::H, Self::C, Self::N, Self::O, Self::S];

    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    #[inline]
    pub const fn as_u32(self) -> u32 {
        self as u32
    }

    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::H => "H",
            Self::C => "C",
            Self::N => "N",
            Self::O => "O",
            Self::S => "S",
        }
    }

    #[inline]
    pub const fn from_index(idx: usize) -> Option<Self> {
        match idx {
            0 => Some(Self::H),
            1 => Some(Self::C),
            2 => Some(Self::N),
            3 => Some(Self::O),
            4 => Some(Self::S),
            _ => None,
        }
    }

    /// Parse an explicit PDB/mmCIF element symbol (`cols 77-78` or `_atom_site.type_symbol`)
    /// into one of the five biomolecular ANI species (`H`, `C`, `N`, `O`, `S`).
    ///
    /// Rejects all 2-character metal/halide/metalloid symbols (`NA`, `CA`, `MG`, `ZN`, `FE`,
    /// `CL`, `BR`, `CU`, `SE`, `CO`, `CD`, `NI`, `HG`, `OS`, etc.) as `None`.
    pub fn from_element_symbol(element: &str) -> Option<Self> {
        let trimmed = element.trim();
        let sym = trimmed.trim_end_matches(|c: char| c.is_ascii_digit() || c == '+' || c == '-');
        if sym.is_empty() {
            return None;
        }
        let up = sym.to_ascii_uppercase();
        match up.as_str() {
            "H" | "D" => Some(Self::H),
            "C" => Some(Self::C),
            "N" => Some(Self::N),
            "O" => Some(Self::O),
            "S" => Some(Self::S),
            _ => None,
        }
    }

    /// Infer biomolecular species (`H`, `C`, `N`, `O`, `S`) from full PDB/mmCIF atom context.
    ///
    /// Precedence:
    /// 1. Explicit `element` column (`cols 77-78` / `_atom_site.type_symbol`) when present and non-empty.
    /// 2. For `HETATM` records or non-standard-amino-acid residues (e.g. `CA`, `NA`, `MG`, `ZN`,
    ///    `FE`, `CL`, `BR`, `CU`), 2-character element symbols are checked before 1-character
    ///    fallback so ions are never misparsed as `C` or `N`.
    /// 3. For standard polymer amino-acid `ATOM` records, unambiguous 2-character non-biomolecular
    ///    elements (`NA`, `CL`, `CU`, `CO`, `SE`, `MG`, `ZN`, `FE`, `BR`, etc.) and left-aligned
    ///    4-character PDB 2-letter element fields (`"CA  "`) are still rejected before falling back
    ///    to the leading element letter (`CA`/`CD`/`CE` -> `C`, `NE`/`ND1` -> `N`, `HG`/`HE` -> `H`).
    pub fn from_pdb_atom(
        element: Option<&str>,
        atom_name: &str,
        residue_name: Option<&str>,
        is_hetatm: bool,
    ) -> Option<Self> {
        if let Some(el) = element.map(str::trim).filter(|s| !s.is_empty()) {
            return Self::from_element_symbol(el);
        }
        let is_std_aa = residue_name.is_some_and(is_standard_amino_acid_residue);
        if is_hetatm || (residue_name.is_some() && !is_std_aa) {
            return Self::from_hetatm_or_ion_name(atom_name);
        }
        Self::from_atom_or_element_name(atom_name)
    }

    /// Infer biomolecular species from a `HETATM` / ion atom name, checking all 2-character
    /// periodic-table symbols (including `CA`, `NA`, `CL`, `CU`, `CO`, `CD`, `CE`, `NE`, `ND`,
    /// `HE`, `HG`, `SE`, `OS`, `MG`, `ZN`, `FE`, `BR`, etc.) before 1-character fallback.
    fn from_hetatm_or_ion_name(name: &str) -> Option<Self> {
        let trimmed = name.trim();
        if trimmed.is_empty() {
            return None;
        }
        let up = trimmed.to_ascii_uppercase();
        let bytes = up.as_bytes();
        // Leading digit + H/D (e.g. "1H", "2H2") is a hydrogen naming convention.
        if bytes.len() >= 2 && bytes[0].is_ascii_digit() {
            return match bytes[1] {
                b'H' | b'D' => Some(Self::H),
                _ => None,
            };
        }
        let alpha_prefix: String = up.chars().take_while(|c| c.is_ascii_alphabetic()).collect();
        if alpha_prefix.is_empty() {
            return None;
        }
        if alpha_prefix.len() >= 2 && is_two_char_non_hcnos_element(&alpha_prefix[..2]) {
            return None;
        }
        match alpha_prefix.as_bytes()[0] {
            b'H' | b'D' => Some(Self::H),
            b'C' => Some(Self::C),
            b'N' => Some(Self::N),
            b'O' => Some(Self::O),
            b'S' => Some(Self::S),
            _ => None,
        }
    }

    /// Infer biomolecular species (`H`, `C`, `N`, `O`, `S`) from a PDB/mmCIF atom or element name.
    ///
    /// Checks PDB fixed-column left-aligned 2-char ions (e.g. `"CA  "` vs `" CA "`) and
    /// unambiguous 2-character metal/halide/metalloid symbols (`NA`, `CL`, `CU`, `CO`, `CR`,
    /// `CS`, `MG`, `ZN`, `FE`, `BR`, `NI`, `SE`, `SI`, `SR`, `SN`, `SB`, `SC`, `SM`, `OS`, `HF`,
    /// etc.) before falling back to the 1-character biomolecular symbol. When an explicit
    /// PDB/mmCIF `element` column or `HETATM` flag is available, prefer `Self::from_pdb_atom` or
    /// `Self::from_element_symbol`.
    pub fn from_atom_or_element_name(name: &str) -> Option<Self> {
        // PDB 4-char fixed-width convention: cols 13-14 left-aligned 2-letter element with
        // trailing spaces ("CA  " = Calcium ion; whereas " CA " = alpha-carbon).
        if name.len() == 4 {
            let b = name.as_bytes();
            if b[0].is_ascii_alphabetic()
                && b[1].is_ascii_alphabetic()
                && b[2] == b' '
                && b[3] == b' '
            {
                let sym = format!(
                    "{}{}",
                    (b[0] as char).to_ascii_uppercase(),
                    (b[1] as char).to_ascii_uppercase()
                );
                if is_two_char_non_hcnos_element(&sym) {
                    return None;
                }
            }
        }

        let trimmed = name.trim();
        if trimmed.is_empty() {
            return None;
        }
        let up = trimmed.to_ascii_uppercase();
        let bytes = up.as_bytes();

        // Leading digit (e.g. "1HB", "2HD1", "3HG2") is standard PDB proton nomenclature.
        if bytes[0].is_ascii_digit() {
            let first_alpha = bytes.iter().copied().find(|b| b.is_ascii_alphabetic())?;
            return match first_alpha {
                b'H' | b'D' => Some(Self::H),
                _ => None,
            };
        }

        let alpha_prefix: String = up.chars().take_while(|c| c.is_ascii_alphabetic()).collect();
        if alpha_prefix.is_empty() {
            return None;
        }

        // Reject unambiguous 2-character metal/halide/metalloid symbols that never occur as
        // standard protein backbone/sidechain atom names before 1-character fallback.
        if alpha_prefix.len() >= 2
            && is_unambiguous_two_char_non_protein_element(&alpha_prefix[..2])
        {
            return None;
        }

        match alpha_prefix.as_bytes()[0] {
            b'H' | b'D' => Some(Self::H),
            b'C' => Some(Self::C),
            b'N' => Some(Self::N),
            b'O' => Some(Self::O),
            b'S' => Some(Self::S),
            _ => None,
        }
    }

    /// Symmetric unordered neighbor species pair index in `0..NUM_SPECIES_PAIRS` (`0..15`).
    ///
    /// Maps `(a, b)` and `(b, a)` identically to upper-triangular index:
    /// `(lo * (2 * NUM_SPECIES + 1 - lo)) / 2 + (hi - lo)`.
    #[inline]
    pub const fn pair_index(a: Self, b: Self) -> usize {
        let ia = a.index();
        let ib = b.index();
        let (lo, hi) = if ia <= ib { (ia, ib) } else { (ib, ia) };
        (lo * (2 * NUM_SPECIES + 1 - lo)) / 2 + (hi - lo)
    }
}

/// True for standard proteinogenic amino-acid 3-letter residue codes (plus common protonation
/// and NMR/crystallographic aliases like `HID`, `HIE`, `HIP`, `CYX`, `MSE`).
fn is_standard_amino_acid_residue(res_name: &str) -> bool {
    let up = res_name.trim().to_ascii_uppercase();
    matches!(
        up.as_str(),
        "ALA"
            | "ARG"
            | "ASN"
            | "ASP"
            | "CYS"
            | "GLN"
            | "GLU"
            | "GLY"
            | "HIS"
            | "ILE"
            | "LEU"
            | "LYS"
            | "MET"
            | "PHE"
            | "PRO"
            | "SER"
            | "THR"
            | "TRP"
            | "TYR"
            | "VAL"
            | "HID"
            | "HIE"
            | "HIP"
            | "HSD"
            | "HSE"
            | "HSP"
            | "CYX"
            | "CYM"
            | "ASH"
            | "GLH"
            | "LYN"
            | "MSE"
            | "SEC"
            | "PYL"
    )
}

/// Two-character periodic-table element symbols that are NOT in `{H, C, N, O, S}` and can never
/// appear as valid standard protein atom names (`NA`, `CL`, `CU`, `CO`, `SE`, `MG`, `ZN`, `FE`,
/// `BR`, etc.).
fn is_unambiguous_two_char_non_protein_element(sym2: &str) -> bool {
    matches!(
        sym2,
        "NA" | "CL"
            | "CU"
            | "CO"
            | "CR"
            | "CS"
            | "CF"
            | "CM"
            | "CN"
            | "MG"
            | "ZN"
            | "FE"
            | "BR"
            | "NI"
            | "NB"
            | "NP"
            | "NO"
            | "SE"
            | "SI"
            | "SR"
            | "SN"
            | "SB"
            | "SC"
            | "SM"
            | "OS"
            | "HF"
            | "MN"
            | "MO"
            | "BA"
            | "BE"
            | "BI"
            | "AG"
            | "AL"
            | "AR"
            | "AS"
            | "AU"
            | "LI"
            | "PB"
            | "PD"
            | "PT"
            | "RB"
            | "RU"
            | "TI"
            | "TL"
            | "XE"
            | "ZR"
    )
}

/// Complete set of two-character periodic-table element symbols relevant to `HETATM` / ion /
/// explicit-element parsing (includes `CA`, `CD`, `CE`, `NE`, `ND`, `HE`, `HG`, `HO`, which
/// collide with standard protein atom names like alpha/delta/epsilon carbon or amide/sidechain
/// protons/nitrogens only inside polymer `ATOM` records).
fn is_two_char_non_hcnos_element(sym2: &str) -> bool {
    is_unambiguous_two_char_non_protein_element(sym2)
        || matches!(sym2, "CA" | "CD" | "CE" | "NE" | "ND" | "HE" | "HG" | "HO")
}

/// Six canonical protein backbone NMR target nuclei predicted by `LEGOLAS`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum TargetNucleus {
    /// Amide proton (`H^N`).
    Hn = 0,
    /// Backbone amide nitrogen (`15N`).
    N15 = 1,
    /// Alpha carbon (`13CA`).
    Ca13 = 2,
    /// Beta carbon (`13CB`).
    Cb13 = 3,
    /// Backbone carbonyl carbon (`13C'`).
    Co13 = 4,
    /// Alpha proton (`1HA`).
    Ha1 = 5,
}

impl TargetNucleus {
    pub const ALL: [Self; 6] = [
        Self::Hn,
        Self::N15,
        Self::Ca13,
        Self::Cb13,
        Self::Co13,
        Self::Ha1,
    ];

    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    #[inline]
    pub const fn from_index(idx: usize) -> Option<Self> {
        match idx {
            0 => Some(Self::Hn),
            1 => Some(Self::N15),
            2 => Some(Self::Ca13),
            3 => Some(Self::Cb13),
            4 => Some(Self::Co13),
            5 => Some(Self::Ha1),
            _ => None,
        }
    }

    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Hn => "H",
            Self::N15 => "N",
            Self::Ca13 => "CA",
            Self::Cb13 => "CB",
            Self::Co13 => "C",
            Self::Ha1 => "HA",
        }
    }

    #[inline]
    pub const fn species(self) -> Species {
        match self {
            Self::Hn | Self::Ha1 => Species::H,
            Self::N15 => Species::N,
            Self::Ca13 | Self::Cb13 | Self::Co13 => Species::C,
        }
    }

    #[inline]
    pub const fn to_backbone_shift_atom(self) -> BackboneShiftAtom {
        match self {
            Self::Hn => BackboneShiftAtom::H,
            Self::N15 => BackboneShiftAtom::N,
            Self::Ca13 => BackboneShiftAtom::CA,
            Self::Cb13 => BackboneShiftAtom::CB,
            Self::Co13 => BackboneShiftAtom::C,
            Self::Ha1 => BackboneShiftAtom::HA,
        }
    }

    #[inline]
    pub const fn from_backbone_shift_atom(atom: BackboneShiftAtom) -> Self {
        match atom {
            BackboneShiftAtom::H => Self::Hn,
            BackboneShiftAtom::N => Self::N15,
            BackboneShiftAtom::CA => Self::Ca13,
            BackboneShiftAtom::CB => Self::Cb13,
            BackboneShiftAtom::C => Self::Co13,
            BackboneShiftAtom::HA => Self::Ha1,
        }
    }

    /// Typical biomolecular backbone chemical-shift center (ppm) for output standardization.
    #[inline]
    pub const fn default_mean_ppm(self) -> f64 {
        match self {
            Self::Hn => 8.25,
            Self::N15 => 119.5,
            Self::Ca13 => 55.5,
            Self::Cb13 => 36.0,
            Self::Co13 => 175.5,
            Self::Ha1 => 4.35,
        }
    }

    /// Typical biomolecular backbone chemical-shift scale (ppm) for output standardization.
    #[inline]
    pub const fn default_scale_ppm(self) -> f64 {
        match self {
            Self::Hn => 0.85,
            Self::N15 => 6.0,
            Self::Ca13 => 5.5,
            Self::Cb13 => 12.0,
            Self::Co13 => 3.0,
            Self::Ha1 => 0.65,
        }
    }
}

/// 3D atom coordinate with species and optional backbone NMR target metadata.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LegolasAtom {
    pub species: Species,
    pub position: [f64; 3],
    pub residue_seq_id: Option<i32>,
    pub atom_name: Option<String>,
    pub target_nucleus: Option<TargetNucleus>,
}

impl LegolasAtom {
    pub fn new(species: Species, position: [f64; 3]) -> Self {
        Self {
            species,
            position,
            residue_seq_id: None,
            atom_name: None,
            target_nucleus: None,
        }
    }

    pub fn with_target(
        species: Species,
        position: [f64; 3],
        residue_seq_id: i32,
        atom_name: impl Into<String>,
        target_nucleus: TargetNucleus,
    ) -> Self {
        Self {
            species,
            position,
            residue_seq_id: Some(residue_seq_id),
            atom_name: Some(atom_name.into()),
            target_nucleus: Some(target_nucleus),
        }
    }
}

/// Angular evaluation convention for ANI-AEV three-body angles `theta_ijk`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum AngularFormula {
    /// Exact geometric angle `atan2(||d_ij x d_ik||, d_ij . d_ik)` (Kahan's formula; matches
    /// exact Smith et al. 2017 derivation to machine epsilon).
    #[default]
    ExactAtan2,
    /// TorchANI / upstream LEGOLAS numerical convention `acos(0.95 * cos(theta_ijk))`
    /// (Gao et al. *J. Chem. Inf. Model.* 2020), keeping `d(acos)/d(cos)` bounded away from `+-1`.
    TorchAniAcos095,
}

impl AngularFormula {
    #[inline]
    pub const fn gpu_code(self) -> u32 {
        match self {
            Self::ExactAtan2 => 0,
            Self::TorchAniAcos095 => 1,
        }
    }

    #[inline]
    pub const fn bin_code(self) -> u8 {
        match self {
            Self::ExactAtan2 => 0,
            Self::TorchAniAcos095 => 1,
        }
    }

    #[inline]
    pub const fn from_bin_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::ExactAtan2),
            1 => Some(Self::TorchAniAcos095),
            _ => None,
        }
    }
}

/// Parameterization of the ANI-1 / LEGOLAS radial and modified angular Atomic Environment Vector.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AevConfig {
    /// Radial cutoff radius `R_c^R` in Angstroms (ANI-1 / LEGOLAS default: `5.2`).
    pub r_cut_radial: f64,
    /// Angular cutoff radius `R_c^A` in Angstroms (ANI-1 / LEGOLAS default: `3.5`).
    pub r_cut_angular: f64,
    /// Radial Gaussian width parameters `eta_R` (`A^-2`).
    pub eta_radial: Vec<f64>,
    /// Radial Gaussian center shifts `R_s` (`A`).
    pub rs_radial: Vec<f64>,
    /// Angular radial-midpoint Gaussian width parameters `eta_A` (`A^-2`).
    pub eta_angular: Vec<f64>,
    /// Angular radial-midpoint center shifts `R_s^A` (`A`).
    pub rs_angular: Vec<f64>,
    /// Angular sharpness exponents `zeta` (`> 0`).
    pub zeta_angular: Vec<f64>,
    /// Angular phase shifts `theta_s` in radians.
    pub theta_s_angular: Vec<f64>,
    /// Three-body angle convention (`ExactAtan2` vs TorchANI `acos(0.95 * cos(theta))`).
    #[serde(default)]
    pub angular_formula: AngularFormula,
}

impl AevConfig {
    /// Standard ANI-1 / LEGOLAS 5-species (`H, C, N, O, S`) AEV configuration:
    /// `5 * 16 = 80` radial + `15 * 4 * 8 = 480` angular = `560` total features
    /// (exact geometric `atan2` angle convention).
    pub fn ani1_standard() -> Self {
        let r_cut_radial = 5.2;
        let r_cut_angular = 3.5;
        let n_rs_r = 16usize;
        let rs_radial = (0..n_rs_r)
            .map(|i| 0.9 + (r_cut_radial - 0.9) * (i as f64) / (n_rs_r as f64))
            .collect();
        let n_rs_a = 4usize;
        let rs_angular = (0..n_rs_a)
            .map(|i| 0.9 + (r_cut_angular - 0.9) * (i as f64) / (n_rs_a as f64))
            .collect();
        let n_theta = 8usize;
        let theta_s_angular = (0..n_theta)
            .map(|i| ((i as f64) + 0.5) * PI / (n_theta as f64))
            .collect();
        Self {
            r_cut_radial,
            r_cut_angular,
            eta_radial: vec![16.0],
            rs_radial,
            eta_angular: vec![8.0],
            rs_angular,
            zeta_angular: vec![32.0],
            theta_s_angular,
            angular_formula: AngularFormula::ExactAtan2,
        }
    }

    /// Standard ANI-1x / TorchANI / upstream LEGOLAS 5-species (`H, C, N, O, S`) AEV configuration
    /// using TorchANI's `acos(0.95 * cos(theta))` angular convention (ready for upstream weight
    /// loading).
    pub fn torchani_standard() -> Self {
        Self::ani1_standard().with_angular_formula(AngularFormula::TorchAniAcos095)
    }

    /// Compact 5-species (`H, C, N, O, S`) AEV configuration for fast CPU/GPU evaluation and
    /// synthetic self-consistency testing (`5 * 8 = 40` radial + `15 * 3 * 4 = 180` angular = `220`
    /// total features).
    pub fn compact() -> Self {
        let r_cut_radial = 5.2;
        let r_cut_angular = 3.5;
        let n_rs_r = 8usize;
        let rs_radial = (0..n_rs_r)
            .map(|i| 0.9 + (r_cut_radial - 0.9) * (i as f64) / (n_rs_r as f64))
            .collect();
        let n_rs_a = 3usize;
        let rs_angular = (0..n_rs_a)
            .map(|i| 0.9 + (r_cut_angular - 0.9) * (i as f64) / (n_rs_a as f64))
            .collect();
        let n_theta = 4usize;
        let theta_s_angular = (0..n_theta)
            .map(|i| ((i as f64) + 0.5) * PI / (n_theta as f64))
            .collect();
        Self {
            r_cut_radial,
            r_cut_angular,
            eta_radial: vec![16.0],
            rs_radial,
            eta_angular: vec![8.0],
            rs_angular,
            zeta_angular: vec![16.0],
            theta_s_angular,
            angular_formula: AngularFormula::ExactAtan2,
        }
    }

    /// Return a copy of this `AevConfig` using the specified `AngularFormula`.
    pub fn with_angular_formula(mut self, angular_formula: AngularFormula) -> Self {
        self.angular_formula = angular_formula;
        self
    }

    #[inline]
    pub fn radial_dim_per_species(&self) -> usize {
        self.eta_radial.len().saturating_mul(self.rs_radial.len())
    }

    #[inline]
    pub fn radial_dim(&self) -> usize {
        NUM_SPECIES.saturating_mul(self.radial_dim_per_species())
    }

    #[inline]
    pub fn angular_dim_per_pair(&self) -> usize {
        self.eta_angular
            .len()
            .saturating_mul(self.rs_angular.len())
            .saturating_mul(self.zeta_angular.len())
            .saturating_mul(self.theta_s_angular.len())
    }

    #[inline]
    pub fn angular_dim(&self) -> usize {
        NUM_SPECIES_PAIRS.saturating_mul(self.angular_dim_per_pair())
    }

    #[inline]
    pub fn total_dim(&self) -> usize {
        self.radial_dim().saturating_add(self.angular_dim())
    }

    /// Validate all cutoff radii, symmetry-function grids, finiteness, and dimension bounds.
    pub fn validate(&self) -> Result<(), String> {
        if !self.r_cut_radial.is_finite() || self.r_cut_radial <= 0.0 || self.r_cut_radial > 50.0 {
            return Err(format!(
                "invalid AEV r_cut_radial {} (must be finite in (0, 50])",
                self.r_cut_radial
            ));
        }
        if !self.r_cut_angular.is_finite() || self.r_cut_angular <= 0.0 || self.r_cut_angular > 50.0
        {
            return Err(format!(
                "invalid AEV r_cut_angular {} (must be finite in (0, 50])",
                self.r_cut_angular
            ));
        }
        if self.eta_radial.is_empty()
            || self.rs_radial.is_empty()
            || self.eta_angular.is_empty()
            || self.rs_angular.is_empty()
            || self.zeta_angular.is_empty()
            || self.theta_s_angular.is_empty()
        {
            return Err("AEV parameter grids must all be non-empty".to_string());
        }
        for &v in &self.eta_radial {
            if !v.is_finite() || v <= 0.0 {
                return Err(format!("invalid eta_radial {v} (must be finite > 0)"));
            }
        }
        for &v in &self.rs_radial {
            if !v.is_finite() || v < 0.0 {
                return Err(format!("invalid rs_radial {v} (must be finite >= 0)"));
            }
        }
        for &v in &self.eta_angular {
            if !v.is_finite() || v <= 0.0 {
                return Err(format!("invalid eta_angular {v} (must be finite > 0)"));
            }
        }
        for &v in &self.rs_angular {
            if !v.is_finite() || v < 0.0 {
                return Err(format!("invalid rs_angular {v} (must be finite >= 0)"));
            }
        }
        for &v in &self.zeta_angular {
            if !v.is_finite() || v <= 0.0 || v > 256.0 {
                return Err(format!(
                    "invalid zeta_angular {v} (must be finite in (0, 256])"
                ));
            }
        }
        for &v in &self.theta_s_angular {
            if !v.is_finite() {
                return Err(format!("invalid theta_s_angular {v} (must be finite)"));
            }
        }

        let rad_per_sp = self
            .eta_radial
            .len()
            .checked_mul(self.rs_radial.len())
            .ok_or_else(|| "AEV radial dimension overflow".to_string())?;
        let rad_dim = NUM_SPECIES
            .checked_mul(rad_per_sp)
            .ok_or_else(|| "AEV radial dimension overflow".to_string())?;
        let ang_per_pair = self
            .eta_angular
            .len()
            .checked_mul(self.rs_angular.len())
            .and_then(|v| v.checked_mul(self.zeta_angular.len()))
            .and_then(|v| v.checked_mul(self.theta_s_angular.len()))
            .ok_or_else(|| "AEV angular dimension overflow".to_string())?;
        let ang_dim = NUM_SPECIES_PAIRS
            .checked_mul(ang_per_pair)
            .ok_or_else(|| "AEV angular dimension overflow".to_string())?;
        let total = rad_dim
            .checked_add(ang_dim)
            .ok_or_else(|| "AEV total dimension overflow".to_string())?;
        if total == 0 || total > MAX_LEGOLAS_AEV_DIM {
            return Err(format!(
                "AEV total dimension {total} out of allowed bounds 1..={MAX_LEGOLAS_AEV_DIM}"
            ));
        }
        Ok(())
    }
}

/// Smooth Behler–Parrinello / ANI cosine cutoff function:
/// `f_c(R) = 0.5 * cos(pi * R / R_c) + 0.5` for `0 <= R <= R_c`, `0` for `R > R_c`.
#[inline]
pub fn cosine_cutoff(r: f64, r_cut: f64) -> f64 {
    if r <= 0.0 {
        1.0
    } else if r >= r_cut {
        0.0
    } else {
        0.5 * (PI * r / r_cut).cos() + 0.5
    }
}

/// Validate an atom slice against `MAX_LEGOLAS_ATOMS`, coordinate finiteness, and bounds.
pub fn validate_atoms(atoms: &[LegolasAtom]) -> Result<(), String> {
    if atoms.len() > MAX_LEGOLAS_ATOMS {
        return Err(format!(
            "atom count {} exceeds MAX_LEGOLAS_ATOMS ({MAX_LEGOLAS_ATOMS})",
            atoms.len()
        ));
    }
    for (idx, atom) in atoms.iter().enumerate() {
        for axis in 0..3 {
            let c = atom.position[axis];
            if !c.is_finite() || c.abs() > MAX_LEGOLAS_COORD_ANGSTROM {
                return Err(format!(
                    "atom {idx} has invalid coordinate {c} on axis {axis} (must be finite with |x| <= {MAX_LEGOLAS_COORD_ANGSTROM})"
                ));
            }
        }
    }
    Ok(())
}

#[inline]
fn cell_coords(pos: [f64; 3], inv_cell_size: f64) -> (i64, i64, i64) {
    (
        (pos[0] * inv_cell_size).floor() as i64,
        (pos[1] * inv_cell_size).floor() as i64,
        (pos[2] * inv_cell_size).floor() as i64,
    )
}

#[derive(Clone, Copy)]
struct AngularNeighbor {
    species: Species,
    dx: f64,
    dy: f64,
    dz: f64,
    r: f64,
    fc_ang: f64,
}

fn evaluate_single_aev_from_neighbors(
    cfg: &AevConfig,
    rad_nbrs: &[(Species, f64, f64)],
    ang_nbrs: &[AngularNeighbor],
) -> Vec<f64> {
    let rad_per_sp = cfg.radial_dim_per_species();
    let rad_dim = cfg.radial_dim();
    let ang_per_pair = cfg.angular_dim_per_pair();
    let total_dim = cfg.total_dim();
    let n_rs_r = cfg.rs_radial.len();
    let n_rs_a = cfg.rs_angular.len();
    let n_zeta = cfg.zeta_angular.len();
    let n_theta = cfg.theta_s_angular.len();

    let mut aev = vec![0.0_f64; total_dim];

    // 1. Radial symmetry functions G_m^R partitioned by neighbor species
    for &(sp_j, r_ij, fc_r) in rad_nbrs {
        let sp_base = sp_j.index() * rad_per_sp;
        for (ei, &eta_r) in cfg.eta_radial.iter().enumerate() {
            let row_base = sp_base + ei * n_rs_r;
            for (ri, &rs_r) in cfg.rs_radial.iter().enumerate() {
                let dr = r_ij - rs_r;
                let exponent = -eta_r * dr * dr;
                if exponent > -80.0 {
                    aev[row_base + ri] += exponent.exp() * fc_r;
                }
            }
        }
    }

    // 2. Modified ANI angular symmetry functions G_m^A partitioned by unordered species pairs
    let mut ang_factors = vec![0.0_f64; n_zeta * n_theta];
    for (a, nj) in ang_nbrs.iter().enumerate() {
        for nk in ang_nbrs.iter().skip(a + 1) {
            let dot = nj.dx * nk.dx + nj.dy * nk.dy + nj.dz * nk.dz;
            let theta = match cfg.angular_formula {
                AngularFormula::ExactAtan2 => {
                    let cx = nj.dy * nk.dz - nj.dz * nk.dy;
                    let cy = nj.dz * nk.dx - nj.dx * nk.dz;
                    let cz = nj.dx * nk.dy - nj.dy * nk.dx;
                    let cross_norm = (cx * cx + cy * cy + cz * cz).sqrt();
                    cross_norm.atan2(dot)
                }
                AngularFormula::TorchAniAcos095 => {
                    let cos_t = (dot / (nj.r * nk.r).max(1e-24)).clamp(-1.0, 1.0);
                    (0.95 * cos_t).clamp(-0.95, 0.95).acos()
                }
            };
            let r_mid = 0.5 * (nj.r + nk.r);
            let fc_prod = nj.fc_ang * nk.fc_ang;
            let pair_idx = Species::pair_index(nj.species, nk.species);
            let pair_base = rad_dim + pair_idx * ang_per_pair;

            // Precompute 2^{1-zeta} * (1 + cos(theta - theta_s))^zeta = 2 * ((1 + cos)/2)^zeta
            for (zi, &zeta) in cfg.zeta_angular.iter().enumerate() {
                for (ti, &theta_s) in cfg.theta_s_angular.iter().enumerate() {
                    let half_u = 0.5 * (1.0 + (theta - theta_s).cos());
                    ang_factors[zi * n_theta + ti] = if half_u <= 1e-12 {
                        0.0
                    } else {
                        2.0 * half_u.powf(zeta)
                    };
                }
            }

            let zt_len = n_zeta * n_theta;
            for (ei, &eta_a) in cfg.eta_angular.iter().enumerate() {
                for (ri, &rs_a) in cfg.rs_angular.iter().enumerate() {
                    let dm = r_mid - rs_a;
                    let exponent = -eta_a * dm * dm;
                    if exponent <= -80.0 {
                        continue;
                    }
                    let rad_scale = exponent.exp() * fc_prod;
                    let slot_base = pair_base + (ei * n_rs_a + ri) * zt_len;
                    for (zt_idx, &ang_part) in ang_factors.iter().enumerate() {
                        if ang_part != 0.0 {
                            aev[slot_base + zt_idx] += ang_part * rad_scale;
                        }
                    }
                }
            }
        }
    }

    aev
}

/// Compute exact ANI Atomic Environment Vectors (`AEV`s) for `query_indices` using an `O(N)`
/// 3D spatial cell-list neighborhood search.
pub fn compute_aevs_for_indices(
    atoms: &[LegolasAtom],
    query_indices: &[usize],
    cfg: &AevConfig,
) -> Result<Vec<Vec<f64>>, String> {
    cfg.validate()?;
    validate_atoms(atoms)?;
    for &q in query_indices {
        if q >= atoms.len() {
            return Err(format!(
                "query atom index {q} out of range for {} atoms",
                atoms.len()
            ));
        }
    }
    if query_indices.is_empty() {
        return Ok(Vec::new());
    }

    let max_rcut = cfg.r_cut_radial.max(cfg.r_cut_angular);
    let cell_size = max_rcut + 1e-4;
    let inv_cell_size = 1.0 / cell_size;
    let r_cut_rad_sq = cfg.r_cut_radial * cfg.r_cut_radial;
    let r_cut_ang_sq = cfg.r_cut_angular * cfg.r_cut_angular;
    let min_dist_sq = MIN_PAIR_DIST_ANGSTROM * MIN_PAIR_DIST_ANGSTROM;

    let mut cells: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::with_capacity(atoms.len());
    for (idx, atom) in atoms.iter().enumerate() {
        let key = cell_coords(atom.position, inv_cell_size);
        let bucket = cells.entry(key).or_default();
        bucket.push(idx);
        if bucket.len() > MAX_LEGOLAS_ATOMS_PER_CELL {
            return Err(format!(
                "spatial cell {key:?} occupancy ({}) exceeds MAX_LEGOLAS_ATOMS_PER_CELL ({MAX_LEGOLAS_ATOMS_PER_CELL}; MAX_LEGOLAS_NEIGHBORS_PER_ATOM = {MAX_LEGOLAS_NEIGHBORS_PER_ATOM})",
                bucket.len()
            ));
        }
    }

    let mut out = Vec::with_capacity(query_indices.len());
    let mut rad_nbrs: Vec<(Species, f64, f64)> = Vec::new();
    let mut ang_nbrs: Vec<AngularNeighbor> = Vec::new();

    for &i in query_indices {
        let pos_i = atoms[i].position;
        let (cx, cy, cz) = cell_coords(pos_i, inv_cell_size);
        rad_nbrs.clear();
        ang_nbrs.clear();

        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let Some(bucket) = cells.get(&(cx + dx, cy + dy, cz + dz)) else {
                        continue;
                    };
                    for &j in bucket {
                        if j == i {
                            continue;
                        }
                        let pos_j = atoms[j].position;
                        let vx = pos_j[0] - pos_i[0];
                        let vy = pos_j[1] - pos_i[1];
                        let vz = pos_j[2] - pos_i[2];
                        let r_sq = vx * vx + vy * vy + vz * vz;
                        if r_sq <= min_dist_sq {
                            continue;
                        }
                        if r_sq <= r_cut_rad_sq || r_sq <= r_cut_ang_sq {
                            let r = r_sq.sqrt();
                            if r <= cfg.r_cut_radial {
                                let fc_r = cosine_cutoff(r, cfg.r_cut_radial);
                                if fc_r > 0.0 {
                                    if rad_nbrs.len() >= MAX_LEGOLAS_NEIGHBORS_PER_ATOM {
                                        return Err(format!(
                                            "atom {i} radial neighbor count exceeds MAX_LEGOLAS_NEIGHBORS_PER_ATOM ({MAX_LEGOLAS_NEIGHBORS_PER_ATOM})"
                                        ));
                                    }
                                    rad_nbrs.push((atoms[j].species, r, fc_r));
                                }
                            }
                            if r <= cfg.r_cut_angular {
                                let fc_a = cosine_cutoff(r, cfg.r_cut_angular);
                                if fc_a > 0.0 {
                                    if ang_nbrs.len() >= MAX_LEGOLAS_NEIGHBORS_PER_ATOM {
                                        return Err(format!(
                                            "atom {i} angular neighbor count exceeds MAX_LEGOLAS_NEIGHBORS_PER_ATOM ({MAX_LEGOLAS_NEIGHBORS_PER_ATOM})"
                                        ));
                                    }
                                    ang_nbrs.push(AngularNeighbor {
                                        species: atoms[j].species,
                                        dx: vx,
                                        dy: vy,
                                        dz: vz,
                                        r,
                                        fc_ang: fc_a,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }

        out.push(evaluate_single_aev_from_neighbors(
            cfg, &rad_nbrs, &ang_nbrs,
        ));
    }

    Ok(out)
}

/// Compute exact ANI Atomic Environment Vectors (`AEV`s) for every atom in `atoms`.
pub fn compute_all_aevs(atoms: &[LegolasAtom], cfg: &AevConfig) -> Result<Vec<Vec<f64>>, String> {
    let indices: Vec<usize> = (0..atoms.len()).collect();
    compute_aevs_for_indices(atoms, &indices, cfg)
}

/// Direct `O(N^2)` reference AEV implementation for cross-checking spatial cell-list exactness.
pub fn compute_aevs_bruteforce_reference(
    atoms: &[LegolasAtom],
    query_indices: &[usize],
    cfg: &AevConfig,
) -> Result<Vec<Vec<f64>>, String> {
    cfg.validate()?;
    validate_atoms(atoms)?;
    for &q in query_indices {
        if q >= atoms.len() {
            return Err(format!(
                "query atom index {q} out of range for {} atoms",
                atoms.len()
            ));
        }
    }
    let min_dist_sq = MIN_PAIR_DIST_ANGSTROM * MIN_PAIR_DIST_ANGSTROM;
    let mut out = Vec::with_capacity(query_indices.len());
    let mut rad_nbrs: Vec<(Species, f64, f64)> = Vec::new();
    let mut ang_nbrs: Vec<AngularNeighbor> = Vec::new();

    for &i in query_indices {
        let pos_i = atoms[i].position;
        rad_nbrs.clear();
        ang_nbrs.clear();
        for (j, aj) in atoms.iter().enumerate() {
            if j == i {
                continue;
            }
            let vx = aj.position[0] - pos_i[0];
            let vy = aj.position[1] - pos_i[1];
            let vz = aj.position[2] - pos_i[2];
            let r_sq = vx * vx + vy * vy + vz * vz;
            if r_sq <= min_dist_sq {
                continue;
            }
            let r = r_sq.sqrt();
            if r <= cfg.r_cut_radial {
                let fc_r = cosine_cutoff(r, cfg.r_cut_radial);
                if fc_r > 0.0 {
                    if rad_nbrs.len() >= MAX_LEGOLAS_NEIGHBORS_PER_ATOM {
                        return Err(format!(
                            "atom {i} radial neighbor count exceeds MAX_LEGOLAS_NEIGHBORS_PER_ATOM ({MAX_LEGOLAS_NEIGHBORS_PER_ATOM})"
                        ));
                    }
                    rad_nbrs.push((aj.species, r, fc_r));
                }
            }
            if r <= cfg.r_cut_angular {
                let fc_a = cosine_cutoff(r, cfg.r_cut_angular);
                if fc_a > 0.0 {
                    if ang_nbrs.len() >= MAX_LEGOLAS_NEIGHBORS_PER_ATOM {
                        return Err(format!(
                            "atom {i} angular neighbor count exceeds MAX_LEGOLAS_NEIGHBORS_PER_ATOM ({MAX_LEGOLAS_NEIGHBORS_PER_ATOM})"
                        ));
                    }
                    ang_nbrs.push(AngularNeighbor {
                        species: aj.species,
                        dx: vx,
                        dy: vy,
                        dz: vz,
                        r,
                        fc_ang: fc_a,
                    });
                }
            }
        }
        out.push(evaluate_single_aev_from_neighbors(
            cfg, &rad_nbrs, &ang_nbrs,
        ));
    }

    Ok(out)
}

/// Smooth hidden-layer activation functions supported by `LEGOLAS` MLPs on CPU and WebGPU.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum ActivationKind {
    /// Continuously differentiable ELU (`ANI` / `LEGOLAS` default `alpha = 0.1`).
    Celu { alpha: f64 },
    /// Clamped tanh-GELU activation.
    Gelu,
    /// Sigmoid Linear Unit (`SiLU` / Swish).
    Silu,
}

impl ActivationKind {
    pub const fn ani_default() -> Self {
        Self::Celu { alpha: 0.1 }
    }

    pub fn validate(self) -> Result<(), String> {
        if let Self::Celu { alpha } = self {
            if !alpha.is_finite() || alpha <= 0.0 || alpha > 100.0 {
                return Err(format!(
                    "invalid CELU alpha {alpha} (must be finite in (0, 100])"
                ));
            }
        }
        Ok(())
    }

    #[inline]
    pub fn eval(self, x: f64) -> f64 {
        match self {
            Self::Celu { alpha } => {
                if x >= 0.0 {
                    x
                } else {
                    let z = (x / alpha).clamp(-40.0, 0.0);
                    alpha * (z.exp() - 1.0)
                }
            }
            Self::Gelu => {
                const SQRT_2_OVER_PI: f64 = 0.797_884_560_802_865_4;
                let inner = (SQRT_2_OVER_PI * (x + 0.044_715 * x * x * x)).clamp(-9.5, 9.5);
                let e2 = (2.0 * inner).exp();
                let t = (e2 - 1.0) / (e2 + 1.0);
                0.5 * x * (1.0 + t)
            }
            Self::Silu => {
                let xc = x.clamp(-20.0, 20.0);
                let sig = 1.0 / (1.0 + (-xc).exp());
                x * sig
            }
        }
    }

    #[inline]
    pub fn deriv(self, x: f64) -> f64 {
        match self {
            Self::Celu { alpha } => {
                if x >= 0.0 {
                    1.0
                } else {
                    let z = x / alpha;
                    if z <= -40.0 {
                        0.0
                    } else {
                        z.exp()
                    }
                }
            }
            Self::Gelu => {
                const SQRT_2_OVER_PI: f64 = 0.797_884_560_802_865_4;
                let raw = SQRT_2_OVER_PI * (x + 0.044_715 * x * x * x);
                if raw >= 9.5 {
                    return 1.0;
                }
                if raw <= -9.5 {
                    return 0.0;
                }
                let e2 = (2.0 * raw).exp();
                let t = (e2 - 1.0) / (e2 + 1.0);
                let d_inner = SQRT_2_OVER_PI * (1.0 + 3.0 * 0.044_715 * x * x);
                0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * d_inner
            }
            Self::Silu => {
                if x >= 20.0 {
                    return 1.0;
                }
                if x <= -20.0 {
                    return 0.0;
                }
                let sig = 1.0 / (1.0 + (-x).exp());
                sig * (1.0 + x * (1.0 - sig))
            }
        }
    }

    #[inline]
    pub const fn gpu_code(self) -> u32 {
        match self {
            Self::Celu { .. } => 1,
            Self::Gelu => 2,
            Self::Silu => 3,
        }
    }

    #[inline]
    pub const fn celu_alpha_or_default(self) -> f64 {
        match self {
            Self::Celu { alpha } => alpha,
            Self::Gelu | Self::Silu => 0.1,
        }
    }
}

/// Dense affine transformation layer `y = W * x + b` with row-major `W [out_dim, in_dim]`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DenseLayer {
    pub in_dim: usize,
    pub out_dim: usize,
    pub weights: Vec<f64>,
    pub biases: Vec<f64>,
}

impl DenseLayer {
    /// Deterministic He/Glorot-scaled layer initialization via `nmr_core::rng::Rng`.
    pub fn new_seeded(in_dim: usize, out_dim: usize, rng: &mut Rng) -> Result<Self, String> {
        if in_dim == 0 || out_dim == 0 {
            return Err("DenseLayer dimensions must be non-zero".to_string());
        }
        let n_weights = out_dim
            .checked_mul(in_dim)
            .ok_or_else(|| "DenseLayer weight count overflow".to_string())?;
        let sd = (2.0 / ((in_dim + out_dim) as f64)).sqrt();
        let mut weights = Vec::with_capacity(n_weights);
        for _ in 0..n_weights {
            weights.push(rng.normal(0.0, sd));
        }
        let biases = vec![0.0_f64; out_dim];
        Ok(Self {
            in_dim,
            out_dim,
            weights,
            biases,
        })
    }

    pub fn validate(&self, expected_in_dim: usize, is_output: bool) -> Result<(), String> {
        if self.in_dim != expected_in_dim {
            return Err(format!(
                "DenseLayer in_dim mismatch: got {}, expected {expected_in_dim}",
                self.in_dim
            ));
        }
        if self.out_dim == 0 {
            return Err("DenseLayer out_dim must be non-zero".to_string());
        }
        if is_output {
            if self.out_dim != 1 {
                return Err(format!(
                    "output DenseLayer out_dim must be 1, got {}",
                    self.out_dim
                ));
            }
        } else if self.out_dim > MAX_LEGOLAS_HIDDEN_DIM {
            return Err(format!(
                "hidden DenseLayer out_dim {} exceeds MAX_LEGOLAS_HIDDEN_DIM ({MAX_LEGOLAS_HIDDEN_DIM})",
                self.out_dim
            ));
        }
        let expected_w = self
            .out_dim
            .checked_mul(self.in_dim)
            .ok_or_else(|| "DenseLayer weight length overflow".to_string())?;
        if self.weights.len() != expected_w {
            return Err(format!(
                "DenseLayer weights length {} != out_dim*in_dim ({expected_w})",
                self.weights.len()
            ));
        }
        if self.biases.len() != self.out_dim {
            return Err(format!(
                "DenseLayer biases length {} != out_dim ({})",
                self.biases.len(),
                self.out_dim
            ));
        }
        if self.weights.iter().any(|w| !w.is_finite()) {
            return Err("DenseLayer contains non-finite (NaN/Inf) weight".to_string());
        }
        if self.biases.iter().any(|b| !b.is_finite()) {
            return Err("DenseLayer contains non-finite (NaN/Inf) bias".to_string());
        }
        Ok(())
    }
}

/// Multi-layer perceptron mapping a single atom's AEV vector to a scalar raw chemical shift.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LegolasMlp {
    pub layers: Vec<DenseLayer>,
}

impl LegolasMlp {
    /// Construct a deterministic seeded MLP with architecture `[aev_dim, hidden..., 1]`.
    pub fn new_seeded(aev_dim: usize, hidden_dims: &[usize], seed: u64) -> Result<Self, String> {
        let n_layers = hidden_dims.len() + 1;
        if n_layers > MAX_LEGOLAS_LAYERS {
            return Err(format!(
                "MLP layer count {n_layers} exceeds MAX_LEGOLAS_LAYERS ({MAX_LEGOLAS_LAYERS})"
            ));
        }
        let mut rng = Rng::new(seed);
        let mut layers = Vec::with_capacity(n_layers);
        let mut prev = aev_dim;
        for &h in hidden_dims {
            if h == 0 || h > MAX_LEGOLAS_HIDDEN_DIM {
                return Err(format!(
                    "invalid hidden dim {h} (must be in 1..={MAX_LEGOLAS_HIDDEN_DIM})"
                ));
            }
            layers.push(DenseLayer::new_seeded(prev, h, &mut rng)?);
            prev = h;
        }
        layers.push(DenseLayer::new_seeded(prev, 1, &mut rng)?);
        Ok(Self { layers })
    }

    pub fn input_dim(&self) -> usize {
        self.layers.first().map_or(0, |l| l.in_dim)
    }

    pub fn output_dim(&self) -> usize {
        self.layers.last().map_or(0, |l| l.out_dim)
    }

    pub fn layer_shape(&self) -> Vec<(usize, usize)> {
        self.layers.iter().map(|l| (l.in_dim, l.out_dim)).collect()
    }

    pub fn validate(&self, expected_aev_dim: usize) -> Result<(), String> {
        if self.layers.is_empty() || self.layers.len() > MAX_LEGOLAS_LAYERS {
            return Err(format!(
                "MLP layer count {} out of allowed range 1..={MAX_LEGOLAS_LAYERS}",
                self.layers.len()
            ));
        }
        let mut expected_in = expected_aev_dim;
        let last_idx = self.layers.len() - 1;
        for (idx, layer) in self.layers.iter().enumerate() {
            layer.validate(expected_in, idx == last_idx)?;
            expected_in = layer.out_dim;
        }
        Ok(())
    }

    /// Evaluate the scalar raw output of the MLP on a single AEV feature vector.
    pub fn forward(&self, aev: &[f64], activation: ActivationKind) -> Result<f64, String> {
        if self.layers.is_empty() {
            return Err("MLP has zero layers".to_string());
        }
        let first_in = self.layers[0].in_dim;
        if aev.len() != first_in {
            return Err(format!(
                "AEV length {} does not match MLP input_dim {first_in}",
                aev.len()
            ));
        }
        let last_idx = self.layers.len() - 1;
        let mut cur: Vec<f64> = aev.to_vec();
        let mut next: Vec<f64> = Vec::new();
        for (li, layer) in self.layers.iter().enumerate() {
            next.clear();
            next.reserve(layer.out_dim);
            let is_hidden = li < last_idx;
            for o in 0..layer.out_dim {
                let row = &layer.weights[o * layer.in_dim..(o + 1) * layer.in_dim];
                let mut acc = layer.biases[o];
                for (&w, &x) in row.iter().zip(cur.iter()) {
                    acc += w * x;
                }
                if is_hidden {
                    acc = activation.eval(acc);
                }
                next.push(acc);
            }
            std::mem::swap(&mut cur, &mut next);
        }
        cur.first()
            .copied()
            .ok_or_else(|| "MLP produced empty output".to_string())
    }
}

/// Ensemble shift prediction and epistemic standard deviation for a single nucleus.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NucleusShiftEstimate {
    pub nucleus: TargetNucleus,
    /// Ensemble mean chemical shift in ppm.
    pub mean_ppm: f64,
    /// Epistemic ensemble standard deviation `sigma_ens` in ppm (`0.0` when `E == 1`).
    pub sigma_ens_ppm: f64,
    /// Individual ensemble member predictions in ppm.
    pub member_predictions_ppm: Vec<f64>,
}

/// Per-nucleus ensemble of `E >= 1` MLPs predicting chemical shift mean and epistemic uncertainty.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NucleusEnsemble {
    pub nucleus: TargetNucleus,
    pub activation: ActivationKind,
    pub output_mean: f64,
    pub output_scale: f64,
    pub members: Vec<LegolasMlp>,
    /// Private trained-state gate: `false` on random/untrained construction and set to `true`
    /// strictly via `train_nucleus_ensemble` (with `epochs >= 1`) or verified trained bundle loading.
    #[serde(default)]
    weights_trained: bool,
}

impl NucleusEnsemble {
    /// Construct a custom (potentially heterogeneous) **untrained** `E`-member MLP ensemble.
    ///
    /// Starts with `is_trained() == false` so `predict_aev` fails closed with
    /// `Err(LegolasError::UntrainedWeights { .. })` until trained via `train_nucleus_ensemble`.
    pub fn from_members(
        nucleus: TargetNucleus,
        activation: ActivationKind,
        output_mean: f64,
        output_scale: f64,
        members: Vec<LegolasMlp>,
    ) -> Result<Self, String> {
        let aev_dim = members
            .first()
            .map(LegolasMlp::input_dim)
            .ok_or_else(|| format!("ensemble {nucleus:?} must contain at least one member"))?;
        let ens = Self {
            nucleus,
            activation,
            output_mean,
            output_scale,
            members,
            weights_trained: false,
        };
        ens.validate(aev_dim)?;
        Ok(ens)
    }

    /// Create a deterministically seeded **untrained** `E`-member MLP ensemble for `nucleus`.
    #[allow(clippy::too_many_arguments)]
    pub fn new_seeded(
        nucleus: TargetNucleus,
        aev_dim: usize,
        hidden_dims: &[usize],
        activation: ActivationKind,
        n_members: usize,
        output_mean: f64,
        output_scale: f64,
        base_seed: u64,
    ) -> Result<Self, String> {
        activation.validate()?;
        if n_members == 0 || n_members > MAX_LEGOLAS_ENSEMBLE_MEMBERS {
            return Err(format!(
                "ensemble size {n_members} out of bounds 1..={MAX_LEGOLAS_ENSEMBLE_MEMBERS}"
            ));
        }
        if !output_mean.is_finite() || !output_scale.is_finite() || output_scale <= 0.0 {
            return Err(format!(
                "invalid output standardization mean={output_mean}, scale={output_scale}"
            ));
        }
        let mut members = Vec::with_capacity(n_members);
        for m in 0..n_members {
            let seed = base_seed
                .wrapping_add((nucleus.index() as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
                .wrapping_add((m as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
            members.push(LegolasMlp::new_seeded(aev_dim, hidden_dims, seed)?);
        }
        Ok(Self {
            nucleus,
            activation,
            output_mean,
            output_scale,
            members,
            weights_trained: false,
        })
    }

    /// True if this ensemble's weights were trained via `train_nucleus_ensemble` or loaded from a
    /// verified trained `LEGOLAS-V1` bundle.
    #[inline]
    pub fn is_trained(&self) -> bool {
        self.weights_trained
    }

    pub fn validate(&self, expected_aev_dim: usize) -> Result<(), String> {
        self.activation.validate()?;
        if !self.output_mean.is_finite()
            || !self.output_scale.is_finite()
            || self.output_scale <= 0.0
        {
            return Err(format!(
                "ensemble {:?} has invalid output standardization (mean={}, scale={})",
                self.nucleus, self.output_mean, self.output_scale
            ));
        }
        if self.members.is_empty() || self.members.len() > MAX_LEGOLAS_ENSEMBLE_MEMBERS {
            return Err(format!(
                "ensemble {:?} member count {} out of bounds 1..={MAX_LEGOLAS_ENSEMBLE_MEMBERS}",
                self.nucleus,
                self.members.len()
            ));
        }
        for (idx, m) in self.members.iter().enumerate() {
            m.validate(expected_aev_dim)
                .map_err(|e| format!("ensemble {:?} member {idx} invalid: {e}", self.nucleus))?;
        }
        Ok(())
    }

    /// True if every member shares identical layer shapes (enabling batched single-pass GPU launch).
    pub fn is_homogeneous_architecture(&self) -> bool {
        let Some(first) = self.members.first() else {
            return false;
        };
        let shape = first.layer_shape();
        self.members
            .iter()
            .skip(1)
            .all(|m| m.layer_shape() == shape)
    }

    /// Evaluate the ensemble on a single precomputed AEV vector.
    ///
    /// **Fails closed** with `Err(LegolasError::UntrainedWeights { .. })` when `!self.is_trained()`
    /// so external callers cannot bypass the model-level untrained-weights gate via
    /// `model.ensembles[..].predict_aev()`.
    pub fn predict_aev(&self, aev: &[f64]) -> Result<NucleusShiftEstimate, LegolasError> {
        self.predict_aev_with_options(aev, &LegolasPredictOptions::default())
    }

    /// Evaluate the ensemble on a single precomputed AEV vector with explicit inference options.
    ///
    /// Fails closed with `Err(LegolasError::UntrainedWeights { .. })` unless either
    /// `self.is_trained()` is `true` or `options.allow_untrained_for_testing` is explicitly enabled.
    pub fn predict_aev_with_options(
        &self,
        aev: &[f64],
        options: &LegolasPredictOptions,
    ) -> Result<NucleusShiftEstimate, LegolasError> {
        if !self.weights_trained && !options.allow_untrained_for_testing {
            return Err(LegolasError::UntrainedWeights {
                provenance: UNTRAINED_WEIGHTS_PROVENANCE.to_string(),
            });
        }
        self.predict_aev_unchecked(aev)
            .map_err(LegolasError::Validation)
    }

    /// Crate-internal unchecked ensemble forward pass used inside the training loop (to compute
    /// initial pre-training RMSE) and by gated outer entry points.
    pub(crate) fn predict_aev_unchecked(
        &self,
        aev: &[f64],
    ) -> Result<NucleusShiftEstimate, String> {
        if self.members.is_empty() {
            return Err(format!("ensemble {:?} has zero members", self.nucleus));
        }
        let mut preds = Vec::with_capacity(self.members.len());
        let mut sum = 0.0_f64;
        for m in &self.members {
            let raw = m.forward(aev, self.activation)?;
            let ppm = self.output_mean + self.output_scale * raw;
            if !ppm.is_finite() {
                return Err(format!(
                    "ensemble {:?} produced non-finite prediction {ppm}",
                    self.nucleus
                ));
            }
            preds.push(ppm);
            sum += ppm;
        }
        let n = preds.len() as f64;
        let mean_ppm = sum / n;
        let sigma_ens_ppm = if preds.len() > 1 {
            let var = preds
                .iter()
                .map(|&p| {
                    let d = p - mean_ppm;
                    d * d
                })
                .sum::<f64>()
                / ((preds.len() - 1) as f64);
            var.max(0.0).sqrt()
        } else {
            0.0
        };
        Ok(NucleusShiftEstimate {
            nucleus: self.nucleus,
            mean_ppm,
            sigma_ens_ppm,
            member_predictions_ppm: preds,
        })
    }
}

fn validate_trained_weights_provenance(provenance: &str) -> Result<(), String> {
    let trimmed = provenance.trim();
    if trimmed.is_empty() {
        return Err("trained LEGOLAS weights_provenance must be non-empty".to_string());
    }
    let lower = trimmed.to_ascii_lowercase();
    if lower == UNTRAINED_WEIGHTS_PROVENANCE
        || lower.contains("untrained")
        || lower.contains("random-init")
        || lower == "none"
        || lower == "n/a"
    {
        return Err(format!(
            "weights_trained=true contradicts non-trained weights_provenance {trimmed:?}"
        ));
    }
    Ok(())
}

/// Descriptive metadata embedded in a `LEGOLAS-V1` model bundle.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegolasModelMetadata {
    pub model_name: String,
    pub provenance: String,
    pub license: String,
    pub ferric_version: String,
    /// `true` ONLY when numerical weights were trained via the training flow or loaded from a
    /// verified trained bundle. Kept private so external callers cannot arbitrarily toggle it
    /// (`22.12:P1-1`).
    weights_trained: bool,
    /// Explicit provenance tag for the numerical weights (e.g.
    /// `"untrained-architecture-no-bundled-weights"` for random initialization).
    weights_provenance: String,
}

impl Default for LegolasModelMetadata {
    fn default() -> Self {
        Self::untrained()
    }
}

impl LegolasModelMetadata {
    /// Honest metadata for an untrained / randomly initialized `LEGOLAS` architecture without
    /// bundled upstream weights (`22.12:P1-1`).
    pub fn untrained() -> Self {
        Self {
            model_name: "legolas-ani-aev-backbone-untrained".to_string(),
            provenance:
                "Clean-room LEGOLAS ANI-AEV architecture (untrained; no bundled upstream weights)"
                    .to_string(),
            license: "MIT".to_string(),
            ferric_version: crate::VERSION.to_string(),
            weights_trained: false,
            weights_provenance: UNTRAINED_WEIGHTS_PROVENANCE.to_string(),
        }
    }

    /// Crate-internal constructor for metadata carrying trained weights with explicit provenance,
    /// restricted to the training flow and verified bundle loaders.
    pub(crate) fn trained(weights_provenance: impl Into<String>) -> Result<Self, String> {
        let wp = weights_provenance.into();
        validate_trained_weights_provenance(&wp)?;
        Ok(Self {
            model_name: "legolas-ani-aev-backbone".to_string(),
            provenance: "Clean-room LEGOLAS ANI-AEV architecture (Darrows et al. JCTC 2025; Smith et al. Chem. Sci. 2017)".to_string(),
            license: "MIT".to_string(),
            ferric_version: crate::VERSION.to_string(),
            weights_trained: true,
            weights_provenance: wp,
        })
    }

    /// True if this metadata marks trained weights.
    #[inline]
    pub fn weights_trained(&self) -> bool {
        self.weights_trained
    }

    /// Return the explicit weight provenance string.
    #[inline]
    pub fn weights_provenance(&self) -> &str {
        &self.weights_provenance
    }
}

/// Complete multi-nucleus `LEGOLAS` ANI-AEV shift predictor model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LegolasModel {
    pub aev_config: AevConfig,
    pub ensembles: BTreeMap<TargetNucleus, NucleusEnsemble>,
    pub metadata: LegolasModelMetadata,
}

/// Ergonomic alias for `LegolasModel` as a shift predictor.
pub type LegolasPredictor = LegolasModel;

/// Predicted chemical shift for one tagged target atom.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AtomShiftPrediction {
    pub atom_index: usize,
    pub residue_seq_id: Option<i32>,
    pub atom_name: Option<String>,
    pub estimate: NucleusShiftEstimate,
}

/// Versioned `LEGOLAS-V1` JSON weight bundle envelope with SHA-256 payload integrity checksum.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LegolasWeightBundle {
    pub format: String,
    pub weights_trained: bool,
    pub weights_provenance: String,
    pub payload_sha256: String,
    pub model: LegolasModel,
}

impl LegolasModel {
    /// Construct a deterministically seeded **untrained** model covering all 6 backbone nuclei
    /// (`H^N`, `15N`, `13CA`, `13CB`, `13C'`, `1HA`).
    ///
    /// Carries `weights_trained = false` and `weights_provenance =
    /// "untrained-architecture-no-bundled-weights"`. Default prediction calls (`predict_shifts`,
    /// `predict_atoms`, `predict_structure_model`) fail closed with
    /// `Err(LegolasError::UntrainedWeights)` until verified trained weights are loaded or trained.
    pub fn new_seeded_all_backbone(
        aev_config: AevConfig,
        hidden_dims: &[usize],
        activation: ActivationKind,
        n_members: usize,
        base_seed: u64,
    ) -> Result<Self, String> {
        aev_config.validate()?;
        let aev_dim = aev_config.total_dim();
        let mut ensembles = BTreeMap::new();
        for nuc in TargetNucleus::ALL {
            let ens = NucleusEnsemble::new_seeded(
                nuc,
                aev_dim,
                hidden_dims,
                activation,
                n_members,
                nuc.default_mean_ppm(),
                nuc.default_scale_ppm(),
                base_seed,
            )?;
            ensembles.insert(nuc, ens);
        }
        Ok(Self {
            aev_config,
            ensembles,
            metadata: LegolasModelMetadata::untrained(),
        })
    }

    /// Explicit alias for `new_seeded_all_backbone`, emphasizing random untrained initialization.
    pub fn init_random_all_backbone(
        aev_config: AevConfig,
        hidden_dims: &[usize],
        activation: ActivationKind,
        n_members: usize,
        base_seed: u64,
    ) -> Result<Self, String> {
        Self::new_seeded_all_backbone(aev_config, hidden_dims, activation, n_members, base_seed)
    }

    /// Construct a trained `LegolasModel` strictly from ensembles that have already been trained
    /// via `train_nucleus_ensemble` (or loaded from a verified trained `LEGOLAS-V1` bundle).
    ///
    /// Fails closed if `ensembles` is empty, if any ensemble still carries untrained weights
    /// (`!ens.is_trained()`), or if `weights_provenance` is empty/untrained.
    pub fn from_trained_ensembles(
        aev_config: AevConfig,
        ensembles: BTreeMap<TargetNucleus, NucleusEnsemble>,
        weights_provenance: impl Into<String>,
    ) -> Result<Self, String> {
        aev_config.validate()?;
        if ensembles.is_empty() {
            return Err("LegolasModel must contain at least one nucleus ensemble".to_string());
        }
        for (&key, ens) in &ensembles {
            if !ens.is_trained() {
                return Err(format!(
                    "cannot construct trained LegolasModel: ensemble {key:?} has untrained weights; run train_nucleus_ensemble first"
                ));
            }
        }
        let metadata = LegolasModelMetadata::trained(weights_provenance)?;
        let model = Self {
            aev_config,
            ensembles,
            metadata,
        };
        model.validate()?;
        Ok(model)
    }

    /// Train the ensemble for `nucleus` via analytical-backpropagation Adam (`epochs >= 1`).
    ///
    /// Marks the target `NucleusEnsemble` as trained upon completion. Once **every** ensemble
    /// stored in `self.ensembles` has been trained, promotes `self.metadata` to trained with
    /// `weights_provenance` so default `predict_shifts` is enabled.
    #[allow(clippy::too_many_arguments)]
    pub fn train_nucleus(
        &mut self,
        nucleus: TargetNucleus,
        aevs: &[Vec<f64>],
        targets_ppm: &[f64],
        epochs: usize,
        cfg: &LegolasTrainConfig,
        seed: u64,
        weights_provenance: impl Into<String>,
    ) -> Result<EnsembleTrainReport, String> {
        let wp = weights_provenance.into();
        validate_trained_weights_provenance(&wp)?;
        let ens = self
            .ensembles
            .get_mut(&nucleus)
            .ok_or_else(|| format!("LegolasModel has no ensemble for nucleus {nucleus:?}"))?;
        let report = train_nucleus_ensemble(ens, aevs, targets_ppm, epochs, cfg, seed)?;
        if !self.ensembles.is_empty() && self.ensembles.values().all(NucleusEnsemble::is_trained) {
            self.mark_weights_trained(wp)?;
        }
        Ok(report)
    }

    /// True if this model carries verified trained weights (rather than untrained random init).
    #[inline]
    pub fn is_trained(&self) -> bool {
        self.metadata.weights_trained
    }

    /// Return the explicit weight provenance string recorded in metadata.
    #[inline]
    pub fn weights_provenance(&self) -> &str {
        &self.metadata.weights_provenance
    }

    /// Crate-internal helper restricted to the training flow: marks this model's metadata as
    /// trained ONLY after verifying every ensemble in `self.ensembles` has been trained and
    /// `weights_provenance` is a valid non-untrained provenance tag.
    pub(crate) fn mark_weights_trained(
        &mut self,
        weights_provenance: impl Into<String>,
    ) -> Result<(), String> {
        if self.ensembles.is_empty() {
            return Err("cannot mark empty LegolasModel as trained".to_string());
        }
        for (&key, ens) in &self.ensembles {
            if !ens.is_trained() {
                return Err(format!(
                    "cannot mark LegolasModel as trained while ensemble {key:?} has untrained weights"
                ));
            }
        }
        let meta = LegolasModelMetadata::trained(weights_provenance)?;
        self.metadata = meta;
        Ok(())
    }

    /// Validate AEV config, metadata consistency, every nucleus ensemble, layer dimensions, and finiteness.
    pub fn validate(&self) -> Result<(), String> {
        self.aev_config.validate()?;
        if self.metadata.weights_trained {
            validate_trained_weights_provenance(&self.metadata.weights_provenance)?;
        } else if self.metadata.weights_provenance.trim().is_empty() {
            return Err(
                "untrained LEGOLAS model must carry non-empty weights_provenance".to_string(),
            );
        }
        if self.ensembles.is_empty() {
            return Err("LegolasModel must contain at least one nucleus ensemble".to_string());
        }
        let aev_dim = self.aev_config.total_dim();
        for (&key, ens) in &self.ensembles {
            if key != ens.nucleus {
                return Err(format!(
                    "LegolasModel ensemble key {:?} does not match ensemble.nucleus {:?}",
                    key, ens.nucleus
                ));
            }
            if self.metadata.weights_trained && !ens.is_trained() {
                return Err(format!(
                    "LegolasModel has weights_trained=true but ensemble {key:?} has untrained weights"
                ));
            }
            ens.validate(aev_dim)?;
        }
        Ok(())
    }

    /// Predict backbone chemical shifts and epistemic ensemble uncertainties for all atoms in
    /// `atoms` that carry a `target_nucleus` present in `self.ensembles`.
    ///
    /// **Fails closed** with `Err(LegolasError::UntrainedWeights { .. })` whenever
    /// `!self.metadata.weights_trained` (`22.12:P1-1`).
    pub fn predict_shifts(
        &self,
        atoms: &[LegolasAtom],
    ) -> Result<Vec<AtomShiftPrediction>, LegolasError> {
        self.predict_shifts_with_options(atoms, &LegolasPredictOptions::default())
    }

    /// Predict backbone chemical shifts with explicit inference options.
    ///
    /// Fails closed with `Err(LegolasError::UntrainedWeights { .. })` unless either
    /// `self.metadata.weights_trained` is `true` or `options.allow_untrained_for_testing` is
    /// explicitly enabled for numerical/parity testing.
    pub fn predict_shifts_with_options(
        &self,
        atoms: &[LegolasAtom],
        options: &LegolasPredictOptions,
    ) -> Result<Vec<AtomShiftPrediction>, LegolasError> {
        self.validate().map_err(LegolasError::Validation)?;
        if !self.metadata.weights_trained && !options.allow_untrained_for_testing {
            return Err(LegolasError::UntrainedWeights {
                provenance: self.metadata.weights_provenance.clone(),
            });
        }
        validate_atoms(atoms).map_err(LegolasError::Validation)?;

        let mut query_indices = Vec::new();
        for (idx, atom) in atoms.iter().enumerate() {
            if let Some(nuc) = atom.target_nucleus {
                if self.ensembles.contains_key(&nuc) {
                    query_indices.push(idx);
                }
            }
        }
        if query_indices.is_empty() {
            return Ok(Vec::new());
        }

        let aevs = compute_aevs_for_indices(atoms, &query_indices, &self.aev_config)
            .map_err(LegolasError::Validation)?;
        let mut preds = Vec::with_capacity(query_indices.len());
        for (&atom_idx, aev) in query_indices.iter().zip(aevs.iter()) {
            let atom = &atoms[atom_idx];
            let Some(nuc) = atom.target_nucleus else {
                continue;
            };
            let Some(ens) = self.ensembles.get(&nuc) else {
                continue;
            };
            let estimate = ens.predict_aev_with_options(aev, options)?;
            preds.push(AtomShiftPrediction {
                atom_index: atom_idx,
                residue_seq_id: atom.residue_seq_id,
                atom_name: atom.atom_name.clone(),
                estimate,
            });
        }
        Ok(preds)
    }

    /// Predict backbone chemical shifts for `atoms` (alias for `predict_shifts`; fails closed on
    /// untrained weights).
    pub fn predict_atoms(
        &self,
        atoms: &[LegolasAtom],
    ) -> Result<Vec<AtomShiftPrediction>, LegolasError> {
        self.predict_shifts(atoms)
    }

    /// Predict backbone chemical shifts for `atoms` with explicit inference options.
    pub fn predict_atoms_with_options(
        &self,
        atoms: &[LegolasAtom],
        options: &LegolasPredictOptions,
    ) -> Result<Vec<AtomShiftPrediction>, LegolasError> {
        self.predict_shifts_with_options(atoms, options)
    }

    /// Extract all `H, C, N, O, S` coordinates from a `StructureModel` and predict backbone shifts
    /// (fails closed on untrained weights).
    pub fn predict_structure_model(
        &self,
        model: &StructureModel,
    ) -> Result<Vec<AtomShiftPrediction>, LegolasError> {
        self.predict_structure_model_with_options(model, &LegolasPredictOptions::default())
    }

    /// Extract coordinates from a `StructureModel` and predict backbone shifts with explicit options.
    pub fn predict_structure_model_with_options(
        &self,
        model: &StructureModel,
        options: &LegolasPredictOptions,
    ) -> Result<Vec<AtomShiftPrediction>, LegolasError> {
        let atoms = extract_atoms_from_structure_model(model).map_err(LegolasError::Validation)?;
        self.predict_shifts_with_options(&atoms, options)
    }

    /// Compute the deterministic lowercase hex SHA-256 checksum of this model's canonical payload.
    pub fn payload_sha256(&self) -> Result<String, String> {
        let payload = canonical_payload_bytes(self)?;
        Ok(sha256_hex(&payload))
    }

    /// Serialize this model into a versioned `LEGOLAS-V1` JSON bundle with SHA-256 checksum
    /// computed over the canonical compact JSON representation of `"model"` (whitespace-invariant
    /// and immune to decimal float-parser ULP jitter, while detecting any single-character tamper).
    pub fn to_json_bundle(&self) -> Result<String, String> {
        self.validate()?;
        let compact_model = serde_json::to_string(self)
            .map_err(|e| format!("failed to serialize LEGOLAS-V1 model JSON: {e}"))?;
        let payload_sha256 = sha256_hex(compact_model.as_bytes());
        let bundle = LegolasWeightBundle {
            format: LEGOLAS_BUNDLE_FORMAT_V1.to_string(),
            weights_trained: self.metadata.weights_trained,
            weights_provenance: self.metadata.weights_provenance.clone(),
            payload_sha256,
            model: self.clone(),
        };
        let json = serde_json::to_string_pretty(&bundle)
            .map_err(|e| format!("failed to serialize LEGOLAS-V1 JSON bundle: {e}"))?;
        if json.len() > MAX_LEGOLAS_BUNDLE_BYTES {
            return Err(format!(
                "serialized LEGOLAS-V1 JSON bundle ({} B) exceeds MAX_LEGOLAS_BUNDLE_BYTES ({MAX_LEGOLAS_BUNDLE_BYTES} B)",
                json.len()
            ));
        }
        Ok(json)
    }

    /// Deserialize and verify a `LEGOLAS-V1` JSON bundle (format, size cap, schema, metadata
    /// consistency, finiteness, and SHA-256 payload integrity).
    pub fn from_json_bundle(json: &str) -> Result<Self, String> {
        if json.len() > MAX_LEGOLAS_BUNDLE_BYTES {
            return Err(format!(
                "LEGOLAS-V1 JSON bundle size ({} B) exceeds MAX_LEGOLAS_BUNDLE_BYTES ({MAX_LEGOLAS_BUNDLE_BYTES} B)",
                json.len()
            ));
        }
        let canonical_model_bytes = extract_canonical_json_field(json, "model")?;
        let computed_sha = sha256_hex(&canonical_model_bytes);
        let bundle: LegolasWeightBundle = serde_json::from_str(json)
            .map_err(|e| format!("failed to parse LEGOLAS-V1 JSON bundle: {e}"))?;
        if bundle.format != LEGOLAS_BUNDLE_FORMAT_V1 {
            return Err(format!(
                "unsupported LEGOLAS bundle format {:?} (expected {LEGOLAS_BUNDLE_FORMAT_V1:?})",
                bundle.format
            ));
        }
        if bundle.weights_trained != bundle.model.metadata.weights_trained {
            return Err(format!(
                "LEGOLAS-V1 bundle weights_trained ({}) disagrees with model.metadata.weights_trained ({})",
                bundle.weights_trained, bundle.model.metadata.weights_trained
            ));
        }
        if bundle.weights_provenance != bundle.model.metadata.weights_provenance {
            return Err(format!(
                "LEGOLAS-V1 bundle weights_provenance ({:?}) disagrees with model.metadata.weights_provenance ({:?})",
                bundle.weights_provenance, bundle.model.metadata.weights_provenance
            ));
        }
        if bundle.payload_sha256 != computed_sha {
            return Err(format!(
                "LEGOLAS-V1 SHA-256 mismatch: bundle has {}, computed {computed_sha}",
                bundle.payload_sha256
            ));
        }
        bundle.model.validate()?;
        Ok(bundle.model)
    }

    /// Serialize this model into a compact `LEGOLAS-V1` binary bundle (`LEGLSV01` + SHA-256 + LE payload).
    pub fn to_binary_bundle(&self) -> Result<Vec<u8>, String> {
        self.validate()?;
        let payload = canonical_payload_bytes(self)?;
        let total_len = 8usize
            .checked_add(64)
            .and_then(|v| v.checked_add(4))
            .and_then(|v| v.checked_add(payload.len()))
            .ok_or_else(|| "LEGOLAS-V1 binary bundle length overflow".to_string())?;
        if total_len > MAX_LEGOLAS_BUNDLE_BYTES {
            return Err(format!(
                "LEGOLAS-V1 binary bundle ({total_len} B) exceeds MAX_LEGOLAS_BUNDLE_BYTES ({MAX_LEGOLAS_BUNDLE_BYTES} B)"
            ));
        }
        let payload_len_u32 = u32::try_from(payload.len())
            .map_err(|_| "LEGOLAS-V1 binary payload length exceeds u32".to_string())?;
        let sha_hex = sha256_hex(&payload);
        let mut out = Vec::with_capacity(total_len);
        out.extend_from_slice(LEGOLAS_BINARY_MAGIC);
        out.extend_from_slice(sha_hex.as_bytes());
        out.extend_from_slice(&payload_len_u32.to_le_bytes());
        out.extend_from_slice(&payload);
        Ok(out)
    }

    /// Deserialize and verify a compact `LEGOLAS-V1` binary bundle.
    pub fn from_binary_bundle(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() > MAX_LEGOLAS_BUNDLE_BYTES {
            return Err(format!(
                "LEGOLAS-V1 binary bundle size ({} B) exceeds MAX_LEGOLAS_BUNDLE_BYTES ({MAX_LEGOLAS_BUNDLE_BYTES} B)",
                bytes.len()
            ));
        }
        const HEADER_LEN: usize = 8 + 64 + 4;
        if bytes.len() < HEADER_LEN {
            return Err(format!(
                "LEGOLAS-V1 binary bundle too short ({} B < {HEADER_LEN} B)",
                bytes.len()
            ));
        }
        if &bytes[..8] != LEGOLAS_BINARY_MAGIC {
            return Err("invalid LEGOLAS-V1 binary magic header".to_string());
        }
        let stored_sha = std::str::from_utf8(&bytes[8..72])
            .map_err(|_| "invalid UTF-8 in LEGOLAS-V1 binary SHA-256 field".to_string())?;
        let payload_len = u32::from_le_bytes([bytes[72], bytes[73], bytes[74], bytes[75]]) as usize;
        if bytes.len() != HEADER_LEN + payload_len {
            return Err(format!(
                "LEGOLAS-V1 binary payload length mismatch: header says {payload_len} B, buffer has {} B",
                bytes.len() - HEADER_LEN
            ));
        }
        let payload = &bytes[HEADER_LEN..];
        let computed_sha = sha256_hex(payload);
        if stored_sha != computed_sha {
            return Err(format!(
                "LEGOLAS-V1 binary SHA-256 mismatch: header has {stored_sha}, computed {computed_sha}"
            ));
        }
        let model = decode_canonical_payload_bytes(payload)?;
        model.validate()?;
        Ok(model)
    }
}

/// Extract all `H, C, N, O, S` coordinates from a `StructureModel`, deduplicated per residue
/// and tagged with canonical backbone `TargetNucleus` annotations (`H`, `N`, `CA`, `CB`, `C`, `HA`).
pub fn extract_atoms_from_structure_model(
    model: &StructureModel,
) -> Result<Vec<LegolasAtom>, String> {
    let mut out = Vec::new();
    for res in &model.residues {
        let mut per_res: BTreeMap<String, (Species, [f64; 3], Option<TargetNucleus>)> =
            BTreeMap::new();

        for (raw_name, &pos) in &res.atoms {
            let norm = raw_name.trim().to_ascii_uppercase();
            let Some(sp) = Species::from_pdb_atom(None, raw_name, Some(&res.res_name), false)
            else {
                continue;
            };
            let target = match norm.as_str() {
                "H" | "HN" => Some(TargetNucleus::Hn),
                "N" => Some(TargetNucleus::N15),
                "CA" => Some(TargetNucleus::Ca13),
                "CB" => Some(TargetNucleus::Cb13),
                "C" => Some(TargetNucleus::Co13),
                "HA" | "HA2" => Some(TargetNucleus::Ha1),
                _ => None,
            };
            per_res.insert(norm, (sp, pos, target));
        }

        // Ensure canonical completed backbone atoms from StructResidueCoords are present and tagged
        per_res.insert(
            "N".to_string(),
            (Species::N, res.n, Some(TargetNucleus::N15)),
        );
        if let Some(h_pos) = res.h {
            per_res.insert(
                "H".to_string(),
                (Species::H, h_pos, Some(TargetNucleus::Hn)),
            );
        }
        per_res.insert(
            "CA".to_string(),
            (Species::C, res.ca, Some(TargetNucleus::Ca13)),
        );
        per_res.insert(
            "C".to_string(),
            (Species::C, res.c, Some(TargetNucleus::Co13)),
        );
        per_res.insert("O".to_string(), (Species::O, res.o, None));
        if let Some(cb_pos) = res.cb {
            per_res.insert(
                "CB".to_string(),
                (Species::C, cb_pos, Some(TargetNucleus::Cb13)),
            );
        }
        if res.res_name.eq_ignore_ascii_case("GLY") {
            if let Some(ha2) = res.ha2 {
                per_res.insert(
                    "HA2".to_string(),
                    (Species::H, ha2, Some(TargetNucleus::Ha1)),
                );
            }
            if let Some(ha3) = res.ha3 {
                per_res.insert("HA3".to_string(), (Species::H, ha3, None));
            }
        } else {
            per_res.insert(
                "HA".to_string(),
                (Species::H, res.ha, Some(TargetNucleus::Ha1)),
            );
        }

        for (atom_name, (species, position, target_nucleus)) in per_res {
            out.push(LegolasAtom {
                species,
                position,
                residue_seq_id: Some(res.seq_id),
                atom_name: Some(atom_name),
                target_nucleus,
            });
            if out.len() > MAX_LEGOLAS_ATOMS {
                return Err(format!(
                    "structure model atom count exceeds MAX_LEGOLAS_ATOMS ({MAX_LEGOLAS_ATOMS})"
                ));
            }
        }
    }
    validate_atoms(&out)?;
    Ok(out)
}

// ============================================================================
// Canonical JSON Field Extractor & Binary Payload Codec (SHA-256 & Bundles)
// ============================================================================

/// Extract a top-level JSON object field value and strip all insignificant ASCII whitespace
/// outside string literals, yielding byte-for-byte parity with `serde_json::to_string`.
fn extract_canonical_json_field(json: &str, target_key: &str) -> Result<Vec<u8>, String> {
    let bytes = json.as_bytes();
    let n = bytes.len();
    let skip_ws = |mut i: usize| -> usize {
        while i < n && matches!(bytes[i], b' ' | b'\t' | b'\r' | b'\n') {
            i += 1;
        }
        i
    };
    let scan_string = |start: usize| -> Result<(usize, usize), String> {
        if start >= n || bytes[start] != b'"' {
            return Err("expected JSON string".to_string());
        }
        let mut i = start + 1;
        let mut esc = false;
        while i < n {
            let b = bytes[i];
            if esc {
                esc = false;
            } else if b == b'\\' {
                esc = true;
            } else if b == b'"' {
                return Ok((start, i + 1));
            }
            i += 1;
        }
        Err("unterminated JSON string".to_string())
    };
    let scan_value = |start: usize| -> Result<(usize, usize), String> {
        let s = skip_ws(start);
        if s >= n {
            return Err("unexpected EOF expecting JSON value".to_string());
        }
        match bytes[s] {
            b'"' => scan_string(s),
            b'{' | b'[' => {
                let mut i = s;
                let mut depth = 0usize;
                let mut in_str = false;
                let mut esc = false;
                while i < n {
                    let b = bytes[i];
                    if in_str {
                        if esc {
                            esc = false;
                        } else if b == b'\\' {
                            esc = true;
                        } else if b == b'"' {
                            in_str = false;
                        }
                    } else {
                        match b {
                            b'"' => in_str = true,
                            b'{' | b'[' => depth += 1,
                            b'}' | b']' => {
                                depth = depth
                                    .checked_sub(1)
                                    .ok_or_else(|| "unbalanced JSON brackets".to_string())?;
                                if depth == 0 {
                                    return Ok((s, i + 1));
                                }
                            }
                            _ => {}
                        }
                    }
                    i += 1;
                }
                Err("unterminated JSON container".to_string())
            }
            _ => {
                let mut i = s;
                while i < n
                    && !matches!(bytes[i], b',' | b'}' | b']' | b' ' | b'\t' | b'\r' | b'\n')
                {
                    i += 1;
                }
                if i == s {
                    Err("empty JSON primitive value".to_string())
                } else {
                    Ok((s, i))
                }
            }
        }
    };

    let mut i = skip_ws(0);
    if i >= n || bytes[i] != b'{' {
        return Err("LEGOLAS-V1 JSON bundle must be a top-level JSON object".to_string());
    }
    i += 1;

    let mut found_span: Option<(usize, usize)> = None;
    loop {
        i = skip_ws(i);
        if i >= n {
            return Err("unterminated top-level JSON object".to_string());
        }
        if bytes[i] == b'}' {
            break;
        }
        let (ks, ke) = scan_string(i)?;
        let key_raw = &json[ks + 1..ke - 1];
        i = skip_ws(ke);
        if i >= n || bytes[i] != b':' {
            return Err("expected ':' after top-level JSON key".to_string());
        }
        let (vs, ve) = scan_value(i + 1)?;
        if key_raw == target_key {
            if found_span.is_some() {
                return Err(format!("duplicate top-level JSON field {target_key:?}"));
            }
            found_span = Some((vs, ve));
        }
        i = skip_ws(ve);
        if i < n && bytes[i] == b',' {
            i += 1;
        } else if i < n && bytes[i] == b'}' {
            break;
        } else {
            return Err("expected ',' or '}' in top-level JSON object".to_string());
        }
    }

    let (vs, ve) =
        found_span.ok_or_else(|| format!("missing required JSON field {target_key:?}"))?;
    let mut out = Vec::with_capacity(ve - vs);
    let mut in_str = false;
    let mut esc = false;
    for &b in &bytes[vs..ve] {
        if in_str {
            out.push(b);
            if esc {
                esc = false;
            } else if b == b'\\' {
                esc = true;
            } else if b == b'"' {
                in_str = false;
            }
        } else if b == b'"' {
            in_str = true;
            out.push(b);
        } else if !matches!(b, b' ' | b'\t' | b'\r' | b'\n') {
            out.push(b);
        }
    }
    Ok(out)
}

fn push_u32_le(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn push_f64_le(buf: &mut Vec<u8>, v: f64) {
    // Normalize -0.0 to +0.0 so JSON roundtrips preserve bit-exact SHA-256 checksums.
    let norm = if v == 0.0 { 0.0_f64 } else { v };
    buf.extend_from_slice(&norm.to_le_bytes());
}

fn push_str_le(buf: &mut Vec<u8>, s: &str) -> Result<(), String> {
    let len = u32::try_from(s.len()).map_err(|_| "string length exceeds u32".to_string())?;
    push_u32_le(buf, len);
    buf.extend_from_slice(s.as_bytes());
    Ok(())
}

fn push_f64_slice_le(buf: &mut Vec<u8>, vals: &[f64]) -> Result<(), String> {
    let len = u32::try_from(vals.len()).map_err(|_| "slice length exceeds u32".to_string())?;
    push_u32_le(buf, len);
    for &v in vals {
        push_f64_le(buf, v);
    }
    Ok(())
}

/// Encode a validated `LegolasModel` into canonical deterministic little-endian bytes.
pub fn canonical_payload_bytes(model: &LegolasModel) -> Result<Vec<u8>, String> {
    model.validate()?;
    let mut buf = Vec::new();
    push_str_le(&mut buf, LEGOLAS_BUNDLE_FORMAT_V1)?;
    push_str_le(&mut buf, &model.metadata.model_name)?;
    push_str_le(&mut buf, &model.metadata.provenance)?;
    push_str_le(&mut buf, &model.metadata.license)?;
    push_str_le(&mut buf, &model.metadata.ferric_version)?;
    buf.push(u8::from(model.metadata.weights_trained));
    push_str_le(&mut buf, &model.metadata.weights_provenance)?;

    // AEV config
    push_f64_le(&mut buf, model.aev_config.r_cut_radial);
    push_f64_le(&mut buf, model.aev_config.r_cut_angular);
    push_f64_slice_le(&mut buf, &model.aev_config.eta_radial)?;
    push_f64_slice_le(&mut buf, &model.aev_config.rs_radial)?;
    push_f64_slice_le(&mut buf, &model.aev_config.eta_angular)?;
    push_f64_slice_le(&mut buf, &model.aev_config.rs_angular)?;
    push_f64_slice_le(&mut buf, &model.aev_config.zeta_angular)?;
    push_f64_slice_le(&mut buf, &model.aev_config.theta_s_angular)?;
    buf.push(model.aev_config.angular_formula.bin_code());

    // Ensembles in deterministic BTreeMap key order
    let n_ens = u32::try_from(model.ensembles.len())
        .map_err(|_| "ensemble count exceeds u32".to_string())?;
    push_u32_le(&mut buf, n_ens);
    for (&nuc, ens) in &model.ensembles {
        buf.push(nuc as u8);
        match ens.activation {
            ActivationKind::Celu { alpha } => {
                buf.push(1u8);
                push_f64_le(&mut buf, alpha);
            }
            ActivationKind::Gelu => {
                buf.push(2u8);
                push_f64_le(&mut buf, 0.0);
            }
            ActivationKind::Silu => {
                buf.push(3u8);
                push_f64_le(&mut buf, 0.0);
            }
        }
        push_f64_le(&mut buf, ens.output_mean);
        push_f64_le(&mut buf, ens.output_scale);
        buf.push(u8::from(ens.weights_trained));
        let n_members =
            u32::try_from(ens.members.len()).map_err(|_| "member count exceeds u32".to_string())?;
        push_u32_le(&mut buf, n_members);
        for mlp in &ens.members {
            let n_layers = u32::try_from(mlp.layers.len())
                .map_err(|_| "layer count exceeds u32".to_string())?;
            push_u32_le(&mut buf, n_layers);
            for layer in &mlp.layers {
                let in_d =
                    u32::try_from(layer.in_dim).map_err(|_| "in_dim exceeds u32".to_string())?;
                let out_d =
                    u32::try_from(layer.out_dim).map_err(|_| "out_dim exceeds u32".to_string())?;
                push_u32_le(&mut buf, in_d);
                push_u32_le(&mut buf, out_d);
                push_f64_slice_le(&mut buf, &layer.weights)?;
                push_f64_slice_le(&mut buf, &layer.biases)?;
            }
        }
    }

    Ok(buf)
}

struct ByteCursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> ByteCursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| "binary payload offset overflow".to_string())?;
        if end > self.data.len() {
            return Err("unexpected EOF while decoding LEGOLAS-V1 binary payload".to_string());
        }
        let slice = &self.data[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn read_u8(&mut self) -> Result<u8, String> {
        let b = self.read_bytes(1)?;
        Ok(b[0])
    }

    fn read_u32_le(&mut self) -> Result<u32, String> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_f64_le(&mut self) -> Result<f64, String> {
        let b = self.read_bytes(8)?;
        Ok(f64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn read_str(&mut self, max_len: usize) -> Result<String, String> {
        let len = self.read_u32_le()? as usize;
        if len > max_len {
            return Err(format!("string length {len} exceeds limit {max_len}"));
        }
        let raw = self.read_bytes(len)?;
        let s = std::str::from_utf8(raw)
            .map_err(|_| "invalid UTF-8 string in LEGOLAS-V1 binary payload".to_string())?;
        Ok(s.to_string())
    }

    fn read_f64_vec(&mut self, max_len: usize) -> Result<Vec<f64>, String> {
        let len = self.read_u32_le()? as usize;
        if len > max_len {
            return Err(format!("f64 array length {len} exceeds limit {max_len}"));
        }
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            out.push(self.read_f64_le()?);
        }
        Ok(out)
    }
}

fn decode_canonical_payload_bytes(payload: &[u8]) -> Result<LegolasModel, String> {
    let mut cur = ByteCursor::new(payload);
    let format = cur.read_str(64)?;
    if format != LEGOLAS_BUNDLE_FORMAT_V1 {
        return Err(format!(
            "unsupported binary payload format {format:?} (expected {LEGOLAS_BUNDLE_FORMAT_V1:?})"
        ));
    }
    let model_name = cur.read_str(512)?;
    let provenance = cur.read_str(2048)?;
    let license = cur.read_str(256)?;
    let ferric_version = cur.read_str(64)?;
    let trained_flag = cur.read_u8()?;
    let weights_trained = match trained_flag {
        0 => false,
        1 => true,
        _ => {
            return Err(format!(
                "invalid weights_trained flag {trained_flag} in binary payload (expected 0 or 1)"
            ));
        }
    };
    let weights_provenance = cur.read_str(512)?;
    let metadata = LegolasModelMetadata {
        model_name,
        provenance,
        license,
        ferric_version,
        weights_trained,
        weights_provenance,
    };

    let r_cut_radial = cur.read_f64_le()?;
    let r_cut_angular = cur.read_f64_le()?;
    let eta_radial = cur.read_f64_vec(MAX_LEGOLAS_AEV_DIM)?;
    let rs_radial = cur.read_f64_vec(MAX_LEGOLAS_AEV_DIM)?;
    let eta_angular = cur.read_f64_vec(MAX_LEGOLAS_AEV_DIM)?;
    let rs_angular = cur.read_f64_vec(MAX_LEGOLAS_AEV_DIM)?;
    let zeta_angular = cur.read_f64_vec(MAX_LEGOLAS_AEV_DIM)?;
    let theta_s_angular = cur.read_f64_vec(MAX_LEGOLAS_AEV_DIM)?;
    let ang_code = cur.read_u8()?;
    let angular_formula = AngularFormula::from_bin_code(ang_code)
        .ok_or_else(|| format!("invalid angular_formula code {ang_code} in binary payload"))?;
    let aev_config = AevConfig {
        r_cut_radial,
        r_cut_angular,
        eta_radial,
        rs_radial,
        eta_angular,
        rs_angular,
        zeta_angular,
        theta_s_angular,
        angular_formula,
    };
    aev_config.validate()?;

    let n_ens = cur.read_u32_le()? as usize;
    if n_ens == 0 || n_ens > TargetNucleus::ALL.len() {
        return Err(format!("invalid ensemble count {n_ens} in binary payload"));
    }
    let max_weights = MAX_LEGOLAS_AEV_DIM * MAX_LEGOLAS_HIDDEN_DIM;
    let mut ensembles = BTreeMap::new();
    for _ in 0..n_ens {
        let nuc_u8 = cur.read_u8()?;
        let nuc = TargetNucleus::from_index(nuc_u8 as usize)
            .ok_or_else(|| format!("invalid target nucleus index {nuc_u8}"))?;
        let act_tag = cur.read_u8()?;
        let act_param = cur.read_f64_le()?;
        let activation = match act_tag {
            1 => ActivationKind::Celu { alpha: act_param },
            2 => ActivationKind::Gelu,
            3 => ActivationKind::Silu,
            _ => return Err(format!("invalid activation tag {act_tag}")),
        };
        let output_mean = cur.read_f64_le()?;
        let output_scale = cur.read_f64_le()?;
        let ens_trained_flag = cur.read_u8()?;
        let ens_weights_trained = match ens_trained_flag {
            0 => false,
            1 => true,
            _ => {
                return Err(format!(
                    "invalid ensemble weights_trained flag {ens_trained_flag} in binary payload (expected 0 or 1)"
                ));
            }
        };
        let n_members = cur.read_u32_le()? as usize;
        if n_members == 0 || n_members > MAX_LEGOLAS_ENSEMBLE_MEMBERS {
            return Err(format!(
                "invalid member count {n_members} in binary payload"
            ));
        }
        let mut members = Vec::with_capacity(n_members);
        for _ in 0..n_members {
            let n_layers = cur.read_u32_le()? as usize;
            if n_layers == 0 || n_layers > MAX_LEGOLAS_LAYERS {
                return Err(format!("invalid layer count {n_layers} in binary payload"));
            }
            let mut layers = Vec::with_capacity(n_layers);
            for _ in 0..n_layers {
                let in_dim = cur.read_u32_le()? as usize;
                let out_dim = cur.read_u32_le()? as usize;
                let weights = cur.read_f64_vec(max_weights)?;
                let biases = cur.read_f64_vec(MAX_LEGOLAS_HIDDEN_DIM)?;
                layers.push(DenseLayer {
                    in_dim,
                    out_dim,
                    weights,
                    biases,
                });
            }
            members.push(LegolasMlp { layers });
        }
        ensembles.insert(
            nuc,
            NucleusEnsemble {
                nucleus: nuc,
                activation,
                output_mean,
                output_scale,
                members,
                weights_trained: ens_weights_trained,
            },
        );
    }

    if cur.pos != payload.len() {
        return Err(format!(
            "trailing unconsumed bytes in LEGOLAS-V1 binary payload ({} bytes)",
            payload.len() - cur.pos
        ));
    }

    Ok(LegolasModel {
        aev_config,
        ensembles,
        metadata,
    })
}

// ============================================================================
// Analytical Backpropagation & Adam Optimizer
// ============================================================================

/// Exact analytical parameter gradients for one `LegolasMlp`.
#[derive(Clone, Debug, PartialEq)]
pub struct MlpGradients {
    pub layer_d_weights: Vec<Vec<f64>>,
    pub layer_d_biases: Vec<Vec<f64>>,
}

/// Adam optimizer moment buffers for one `LegolasMlp`.
#[derive(Clone, Debug, PartialEq)]
pub struct AdamState {
    pub step: u64,
    pub m_weights: Vec<Vec<f64>>,
    pub v_weights: Vec<Vec<f64>>,
    pub m_biases: Vec<Vec<f64>>,
    pub v_biases: Vec<Vec<f64>>,
}

impl AdamState {
    pub fn new(mlp: &LegolasMlp) -> Self {
        let m_weights = mlp
            .layers
            .iter()
            .map(|l| vec![0.0_f64; l.weights.len()])
            .collect();
        let v_weights = mlp
            .layers
            .iter()
            .map(|l| vec![0.0_f64; l.weights.len()])
            .collect();
        let m_biases = mlp
            .layers
            .iter()
            .map(|l| vec![0.0_f64; l.biases.len()])
            .collect();
        let v_biases = mlp
            .layers
            .iter()
            .map(|l| vec![0.0_f64; l.biases.len()])
            .collect();
        Self {
            step: 0,
            m_weights,
            v_weights,
            m_biases,
            v_biases,
        }
    }
}

/// Hyperparameters for `train_legolas_mlp_step` (Adam + optional L2 weight decay and norm clipping).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LegolasTrainConfig {
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub weight_decay: f64,
    pub grad_clip_norm: Option<f64>,
}

impl Default for LegolasTrainConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-2,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
            grad_clip_norm: Some(10.0),
        }
    }
}

impl LegolasTrainConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err(format!("invalid learning_rate {}", self.learning_rate));
        }
        if !self.beta1.is_finite() || self.beta1 <= 0.0 || self.beta1 >= 1.0 {
            return Err(format!("invalid beta1 {}", self.beta1));
        }
        if !self.beta2.is_finite() || self.beta2 <= 0.0 || self.beta2 >= 1.0 {
            return Err(format!("invalid beta2 {}", self.beta2));
        }
        if !self.epsilon.is_finite() || self.epsilon <= 0.0 {
            return Err(format!("invalid epsilon {}", self.epsilon));
        }
        if !self.weight_decay.is_finite() || self.weight_decay < 0.0 {
            return Err(format!("invalid weight_decay {}", self.weight_decay));
        }
        if let Some(clip) = self.grad_clip_norm {
            if !clip.is_finite() || clip <= 0.0 {
                return Err(format!("invalid grad_clip_norm {clip}"));
            }
        }
        Ok(())
    }
}

/// Compute exact batch Mean Squared Error (MSE) loss `L = (1/B) sum_b (y_hat_b - y_b)^2` and
/// analytical backpropagation gradients `dL/dW`, `dL/db` across all layers of `mlp`.
pub fn mlp_forward_and_gradients(
    mlp: &LegolasMlp,
    activation: ActivationKind,
    output_mean: f64,
    output_scale: f64,
    inputs: &[&[f64]],
    targets_ppm: &[f64],
) -> Result<(f64, MlpGradients), String> {
    activation.validate()?;
    if mlp.layers.is_empty() {
        return Err("MLP has zero layers".to_string());
    }
    let in_dim = mlp.layers[0].in_dim;
    mlp.validate(in_dim)?;
    if inputs.is_empty() || inputs.len() != targets_ppm.len() {
        return Err(format!(
            "batch inputs ({}) and targets ({}) must be non-empty and equal length",
            inputs.len(),
            targets_ppm.len()
        ));
    }
    if !output_mean.is_finite() || !output_scale.is_finite() || output_scale <= 0.0 {
        return Err("invalid output_mean/output_scale".to_string());
    }

    let n_layers = mlp.layers.len();
    let last_idx = n_layers - 1;
    let mut layer_d_weights: Vec<Vec<f64>> = mlp
        .layers
        .iter()
        .map(|l| vec![0.0_f64; l.weights.len()])
        .collect();
    let mut layer_d_biases: Vec<Vec<f64>> = mlp
        .layers
        .iter()
        .map(|l| vec![0.0_f64; l.biases.len()])
        .collect();

    let batch_size = inputs.len() as f64;
    let mut total_sq_err = 0.0_f64;

    let mut pre_acts: Vec<Vec<f64>> = mlp
        .layers
        .iter()
        .map(|l| vec![0.0_f64; l.out_dim])
        .collect();
    let mut post_acts: Vec<Vec<f64>> = mlp
        .layers
        .iter()
        .map(|l| vec![0.0_f64; l.out_dim])
        .collect();

    for (&x, &target) in inputs.iter().zip(targets_ppm.iter()) {
        if x.len() != in_dim {
            return Err(format!(
                "input AEV length {} does not match MLP input_dim {in_dim}",
                x.len()
            ));
        }
        if !target.is_finite() {
            return Err(format!("non-finite training target {target}"));
        }

        // Forward pass storing pre- and post-activations per layer
        for li in 0..n_layers {
            let layer = &mlp.layers[li];
            let (before, at_and_after) = post_acts.split_at_mut(li);
            let prev: &[f64] = if li == 0 { x } else { &before[li - 1] };
            let cur_post = &mut at_and_after[0];
            let is_hidden = li < last_idx;
            for o in 0..layer.out_dim {
                let row = &layer.weights[o * layer.in_dim..(o + 1) * layer.in_dim];
                let mut z = layer.biases[o];
                for (&w, &p) in row.iter().zip(prev.iter()) {
                    z += w * p;
                }
                pre_acts[li][o] = z;
                cur_post[o] = if is_hidden { activation.eval(z) } else { z };
            }
        }

        let pred_ppm = output_mean + output_scale * post_acts[last_idx][0];
        let err = pred_ppm - target;
        total_sq_err += err * err;

        // Backward pass: dL/dy_raw = (2 / B) * (pred_ppm - target) * output_scale
        let mut delta = vec![(2.0 / batch_size) * err * output_scale];
        for li in (0..n_layers).rev() {
            let layer = &mlp.layers[li];
            let prev: &[f64] = if li == 0 { x } else { &post_acts[li - 1] };
            for o in 0..layer.out_dim {
                let d_o = delta[o];
                layer_d_biases[li][o] += d_o;
                let w_grad_row = &mut layer_d_weights[li][o * layer.in_dim..(o + 1) * layer.in_dim];
                for (g_w, &p) in w_grad_row.iter_mut().zip(prev.iter()) {
                    *g_w += d_o * p;
                }
            }

            if li > 0 {
                let prev_dim = layer.in_dim;
                let mut next_delta = vec![0.0_f64; prev_dim];
                for (o, &d_o) in delta.iter().enumerate().take(layer.out_dim) {
                    let row = &layer.weights[o * prev_dim..(o + 1) * prev_dim];
                    for (nd, &w) in next_delta.iter_mut().zip(row.iter()) {
                        *nd += d_o * w;
                    }
                }
                for (j, nd) in next_delta.iter_mut().enumerate() {
                    *nd *= activation.deriv(pre_acts[li - 1][j]);
                }
                delta = next_delta;
            }
        }
    }

    Ok((
        total_sq_err / batch_size,
        MlpGradients {
            layer_d_weights,
            layer_d_biases,
        },
    ))
}

/// Perform one analytical-backpropagation Adam optimization step on `mlp` and return the batch
/// MSE loss (ppm^2) evaluated prior to the parameter update.
#[allow(clippy::too_many_arguments)]
pub fn train_legolas_mlp_step(
    mlp: &mut LegolasMlp,
    adam: &mut AdamState,
    activation: ActivationKind,
    output_mean: f64,
    output_scale: f64,
    inputs: &[&[f64]],
    targets_ppm: &[f64],
    cfg: &LegolasTrainConfig,
) -> Result<f64, String> {
    cfg.validate()?;
    if adam.m_weights.len() != mlp.layers.len() || adam.m_biases.len() != mlp.layers.len() {
        return Err("AdamState layer count does not match LegolasMlp".to_string());
    }

    let (loss, mut grads) = mlp_forward_and_gradients(
        mlp,
        activation,
        output_mean,
        output_scale,
        inputs,
        targets_ppm,
    )?;

    // Optional L2 regularization on weights (not biases)
    if cfg.weight_decay > 0.0 {
        for (li, layer) in mlp.layers.iter().enumerate() {
            for (g, &w) in grads.layer_d_weights[li]
                .iter_mut()
                .zip(layer.weights.iter())
            {
                *g += cfg.weight_decay * w;
            }
        }
    }

    // Optional global L2 gradient norm clipping
    if let Some(max_norm) = cfg.grad_clip_norm {
        let mut sq_norm = 0.0_f64;
        for gw in &grads.layer_d_weights {
            for &g in gw {
                sq_norm += g * g;
            }
        }
        for gb in &grads.layer_d_biases {
            for &g in gb {
                sq_norm += g * g;
            }
        }
        let norm = sq_norm.sqrt();
        if norm > max_norm && norm > 0.0 {
            let scale = max_norm / norm;
            for gw in &mut grads.layer_d_weights {
                for g in gw.iter_mut() {
                    *g *= scale;
                }
            }
            for gb in &mut grads.layer_d_biases {
                for g in gb.iter_mut() {
                    *g *= scale;
                }
            }
        }
    }

    adam.step = adam.step.saturating_add(1);
    let t = adam.step as f64;
    let bc1 = 1.0 - cfg.beta1.powf(t);
    let bc2 = 1.0 - cfg.beta2.powf(t);

    for (li, layer) in mlp.layers.iter_mut().enumerate() {
        if adam.m_weights[li].len() != layer.weights.len()
            || adam.m_biases[li].len() != layer.biases.len()
        {
            return Err(format!("AdamState shape mismatch at layer {li}"));
        }
        for i in 0..layer.weights.len() {
            let g = grads.layer_d_weights[li][i];
            let m = cfg.beta1 * adam.m_weights[li][i] + (1.0 - cfg.beta1) * g;
            let v = cfg.beta2 * adam.v_weights[li][i] + (1.0 - cfg.beta2) * g * g;
            adam.m_weights[li][i] = m;
            adam.v_weights[li][i] = v;
            let m_hat = m / bc1;
            let v_hat = v / bc2;
            layer.weights[i] -= cfg.learning_rate * m_hat / (v_hat.sqrt() + cfg.epsilon);
        }
        for i in 0..layer.biases.len() {
            let g = grads.layer_d_biases[li][i];
            let m = cfg.beta1 * adam.m_biases[li][i] + (1.0 - cfg.beta1) * g;
            let v = cfg.beta2 * adam.v_biases[li][i] + (1.0 - cfg.beta2) * g * g;
            adam.m_biases[li][i] = m;
            adam.v_biases[li][i] = v;
            let m_hat = m / bc1;
            let v_hat = v / bc2;
            layer.biases[i] -= cfg.learning_rate * m_hat / (v_hat.sqrt() + cfg.epsilon);
        }
    }

    Ok(loss)
}

/// Summary metrics returned by `train_nucleus_ensemble`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EnsembleTrainReport {
    pub initial_rmse_ppm: f64,
    pub final_rmse_ppm: f64,
    pub mean_epoch_mse: Vec<f64>,
}

/// Train every member of a `NucleusEnsemble` with deterministic bootstrap resampling (when
/// `members.len() > 1`) and full-batch Adam steps, tracking ensemble RMSE reduction.
///
/// Requires `epochs >= 1` and marks `ensemble.is_trained() == true` upon completion.
pub fn train_nucleus_ensemble(
    ensemble: &mut NucleusEnsemble,
    aevs: &[Vec<f64>],
    targets_ppm: &[f64],
    epochs: usize,
    cfg: &LegolasTrainConfig,
    seed: u64,
) -> Result<EnsembleTrainReport, String> {
    cfg.validate()?;
    if epochs == 0 {
        return Err("training epochs must be >= 1".to_string());
    }
    if aevs.is_empty() || aevs.len() != targets_ppm.len() {
        return Err("training AEVs and targets must be non-empty and equal length".to_string());
    }
    let aev_dim = aevs[0].len();
    ensemble.validate(aev_dim)?;

    let eval_rmse = |ens: &NucleusEnsemble| -> Result<f64, String> {
        let mut sq = 0.0_f64;
        for (x, &y) in aevs.iter().zip(targets_ppm.iter()) {
            let est = ens.predict_aev_unchecked(x)?;
            let d = est.mean_ppm - y;
            sq += d * d;
        }
        Ok((sq / (aevs.len() as f64)).sqrt())
    };

    let initial_rmse_ppm = eval_rmse(ensemble)?;
    let n_samples = aevs.len();
    let n_members = ensemble.members.len();
    let mut rng = Rng::new(seed);

    // Deterministic bootstrap subsets per member for real epistemic ensemble diversity
    let mut member_indices: Vec<Vec<usize>> = Vec::with_capacity(n_members);
    for m in 0..n_members {
        if n_members == 1 || n_samples <= 2 {
            member_indices.push((0..n_samples).collect());
        } else {
            let mut sub = Vec::with_capacity(n_samples);
            // Guarantee broad coverage while injecting distinct bootstrap multiplicity
            for i in 0..n_samples {
                if (i + m) % n_members != 0 || n_samples < n_members {
                    sub.push(i);
                } else {
                    sub.push(rng.usize(n_samples));
                }
            }
            member_indices.push(sub);
        }
    }

    let mut adams: Vec<AdamState> = ensemble.members.iter().map(AdamState::new).collect();
    let mut mean_epoch_mse = Vec::with_capacity(epochs);

    for _epoch in 0..epochs {
        let mut mse_sum = 0.0_f64;
        for (m, mlp) in ensemble.members.iter_mut().enumerate() {
            let idxs = &member_indices[m];
            let batch_x: Vec<&[f64]> = idxs.iter().map(|&i| aevs[i].as_slice()).collect();
            let batch_y: Vec<f64> = idxs.iter().map(|&i| targets_ppm[i]).collect();
            let loss = train_legolas_mlp_step(
                mlp,
                &mut adams[m],
                ensemble.activation,
                ensemble.output_mean,
                ensemble.output_scale,
                &batch_x,
                &batch_y,
                cfg,
            )?;
            mse_sum += loss;
        }
        mean_epoch_mse.push(mse_sum / (n_members as f64));
    }

    ensemble.weights_trained = true;
    let final_rmse_ppm = eval_rmse(ensemble)?;
    Ok(EnsembleTrainReport {
        initial_rmse_ppm,
        final_rmse_ppm,
        mean_epoch_mse,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_internal_mark_weights_trained_rejects_untrained_ensembles_and_bad_provenance() {
        let cfg = AevConfig::compact();
        let model_res = LegolasModel::init_random_all_backbone(
            cfg,
            &[8],
            ActivationKind::Celu { alpha: 0.1 },
            1,
            7,
        );
        assert!(model_res.is_ok(), "init model failed: {:?}", model_res);
        let Ok(mut model) = model_res else {
            return;
        };

        // Internal mark_weights_trained must refuse while any ensemble is untrained
        let mark_res = model.mark_weights_trained("valid-trained-provenance");
        assert!(
            matches!(&mark_res, Err(msg) if msg.contains("untrained weights")),
            "expected untrained-weights rejection, got: {:?}",
            mark_res
        );
        assert!(!model.is_trained());

        // LegolasModelMetadata::trained must reject empty / non-trained labels
        for bad in [
            "",
            "   ",
            UNTRAINED_WEIGHTS_PROVENANCE,
            "untrained-custom",
            "model-untrained-v1",
            "random-init-seed42",
            "none",
            "N/A",
        ] {
            assert!(
                LegolasModelMetadata::trained(bad).is_err(),
                "expected rejection of bad provenance {bad:?}"
            );
        }
    }
}
