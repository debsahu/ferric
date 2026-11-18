//! AlphaFold / ModelArchive mmCIF, PDB, and `PAE` JSON ingestion (`<= 64 MiB` / `<= 4,096`
//! hostile-input-guarded, fail-closed model provenance, `wasm32`-clean with zero network I/O) and
//! FERRIC's original heuristic solution NMR rigidity concordance (`pLDDT` vs experimental
//! Lipari–Szabo $S^2$ or shift-derived, secondary-structure-coupled Berjanskii–Wishart `RCI-S²`)
//! & 6-class gap-free per-residue structural concordance/discordance classifier.
//!
//! Provenance: Clean-room original implementation derived strictly from peer-reviewed literature
//! and public open-data archive specifications (`0` copyleft, academic-licence, or proprietary
//! software consulted; `0` external product names in identifiers or prose):
//! 1. Jumper, J., Evans, R., Pritzel, A. et al. (2021). "Highly accurate protein structure
//!    prediction with AlphaFold." *Nature* **596**, 583–589 — per-residue predicted Local Distance
//!    Difference Test (`pLDDT` $\in [0, 100]$) and $N \times N$ Predicted Aligned Error (`PAE`, Å).
//! 2. Mariani, V., Biasini, M., Barbato, A. & Schwede, T. (2013). "lDDT: a local superposition-free
//!    score for comparing protein structures and models using distance difference tests."
//!    *Bioinformatics* **29**(21), 2722–2728.
//! 3. Varadi, M., Anyango, S., Deshpande, M. et al. (2022). *Nucleic Acids Res.* **50**, D439–D444;
//!    and Varadi, M., Bertoni, D., Magana, P. et al. (2024). *Nucleic Acids Res.* **52**, D368–D375 —
//!    AlphaFold Protein Structure Database (`AFDB` v4/v6, `CC-BY-4.0`), ModelArchive `_ma_qa_metric_local`
//!    mmCIF quality-assessment schema, and `PAE` JSON dialects.
//! 4. Abramson, J., Adler, J., Dunger, J. et al. (2024). *Nature* **630**, 493–500 — AlphaFold 3
//!    `confidences.json` token-level `PAE` (`token_chain_ids`, `token_res_ids`) and atom `pLDDT`s.
//! 5. Tunyasuvunakool, K. et al. (2021). *Nature* **596**, 590–596; and Alderson, T. R.,
//!    Pritisanac, I., Kolaric, D., Moses, A. M. & Forman-Kay, J. D. (2023). *Proc. Natl. Acad.
//!    Sci. U.S.A.* **120**, e2304302120 — disorder (`pLDDT < 50`) vs high-`pLDDT` conditionally
//!    folding molecular recognition features ($\alpha$-MoRFs) in intrinsically disordered regions.
//! 6. Berjanskii, M. V. & Wishart, D. S. (2005). "A simple method to predict protein flexibility
//!    using secondary chemical shifts." *J. Am. Chem. Soc.* **127**(43), 14970–14971 — Random Coil
//!    Index (`RCI`) and order parameter $S^2_{\text{RCI}} = 1 - 0.4\ln(1 + 17.7\cdot\text{RCI})$.
//! 7. Original FERRIC `pLDDT`-vs-$S^2$ concordance heuristic inspired by Fowler, N. J., Sljoka, A.
//!    & Williamson, M. P. (2020), *Nat. Commun.* **11**, 5982; and Fowler, N. J. & Williamson, M. P.
//!    (2022), *Structure* **30**(7), 925–933. Note: Fowler & Williamson compute coordinate-based
//!    constraint-counting rigidity profiles (`FIRST`) and calibrate them against `RCI`, whereas this
//!    module directly correlates AlphaFold `pLDDT` with experimental Lipari–Szabo $S^2$ (preferred)
//!    or shift-derived, secondary-structure-coupled `RCI-S²`, reporting both whole-chain and
//!    ordered-core (`pLDDT >= 70`) Spearman $\rho_s$ with moving-block bootstrap (`block_len = 5`)
//!    95% confidence intervals (`21.W1:P1-1`, `A5`).

use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::path::Path;

use serde::de::{self, Deserializer, SeqAccess, Visitor};
use serde::{Deserialize, Serialize};

use crate::dihedrals::{periodic_angle_abs_diff_deg, SecondaryStructure};
use crate::idp_propensity::{
    compute_idp_propensity, GlobalShiftOffsets as PropensityOffsets, PropensityConfig,
    PropensityReport,
};
use crate::kabsch_sander_ss::{
    classify_kabsch_sander, BackboneResidueCoords, KabschSanderState8, SecondaryStructure4,
    SecondaryStructureReport,
};
use crate::nmrstar::Reference;
use crate::noesy::{
    reconstruct_ca_substituents, v_add, v_dist, v_scale, v_sub, v_unit, PdbChain, PdbResidue,
};
use crate::structure::{det_exp, det_ln};
use crate::torsion_db::{
    estimate_global_shift_offsets, HomologFirewall, HomologFirewallQuery, TorsionPredictionClass,
    DEFAULT_FIREWALL_MAX_SEQ_IDENTITY,
};

/// Hard pre-allocation cap (`64 MiB`) on coordinate input files (`.cif`, `.mmcif`, `.pdb`).
pub const MAX_AF_STRUCTURE_BYTES: usize = 64 * 1024 * 1024;

/// Hard pre-allocation dimension ceiling (`4,096` residues -> `<= 64 MiB` `Vec<f32>` storage)
/// on $N \times N$ Predicted Aligned Error (`PAE`) matrices.
pub const MAX_PAE_RESIDUES: usize = 4_096;

/// Hard pre-allocation byte cap (`256 MiB`) on raw `PAE` JSON text prior to streaming deserialization.
pub const MAX_PAE_JSON_BYTES: usize = 256 * 1024 * 1024;

/// Maximum physically valid `PAE` entry (`64.0 Å`).
pub const MAX_PAE_VALUE_ANGSTROM: f32 = 64.0;

/// Minimum non-terminal evaluated residue count required for global Spearman $\rho_s$ & rigidity RMSD.
pub const MIN_CONCORDANCE_EVAL_RESIDUES: usize = 10;

// ============================================================================
// 1. AlphaFold / ModelArchive mmCIF & PDB Ingestion
// ============================================================================

/// Coordinate provenance inferred from mmCIF / PDB metadata headers (`21.W1:P0-2`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CoordinateProvenance {
    /// Computational structure prediction verified by positive ModelArchive / AFDB provenance
    /// (`_ma_qa_metric_local` linked to a metric explicitly named `pLDDT`, or a verified
    /// `_ma_protocol_step` / `_ma_model_list` / `AF-*` ModelArchive mmCIF block with valid
    /// non-constant `[0, 100]` `B_iso_or_equiv`).
    ComputationalModel,
    /// Experimental structure (`X-RAY DIFFRACTION`, `SOLUTION NMR`, `ELECTRON MICROSCOPY`, etc.,
    /// or non-trivial crystallographic `CRYST1` unit cell) whose isotropic $B$-factors represent
    /// thermal displacement ($\text{\AA}^2$), not `pLDDT`.
    Experimental,
    /// Unmarked / headerless coordinate file, PDB file with `REMARK` mentioning AlphaFold,
    /// non-`pLDDT` `_ma_qa_metric_local` (`QMEANDisCo`), or constant/zero `B_iso_or_equiv`.
    Unknown,
}

/// Exact table / column from which per-residue `pLDDT` values were obtained.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PlddtSource {
    /// ModelArchive `_ma_qa_metric_local` quality-assessment loop verified as `pLDDT` in `_ma_qa_metric`.
    MaQaMetricLocal,
    /// Computational `_atom_site.B_iso_or_equiv` or explicitly forced PDB `tempFactor` column.
    BFactorColumn,
    /// Forced `B_iso_or_equiv` / `tempFactor` extraction (`--force-bfactor-plddt` / `--treat-as-model`) on an experimental file.
    ForcedExperimentalBFactor,
    /// No verified `pLDDT` available (experimental, headerless, unverified `B_iso`, or non-`pLDDT` QA metric).
    Unavailable,
}

/// Structured audit diagnostic emitted during `.cif` / `.pdb` ingestion.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AfIngestDiagnostic {
    /// Unit-interval `[0.0, 1.0]` `pLDDT` values (`median > 0.15`, `max <= 1.0`, non-constant) were
    /// automatically rescaled to `[0.0, 100.0]`.
    RescaledUnitPlddtTo100,
    /// Experimental coordinate file (`X-RAY DIFFRACTION`, `SOLUTION NMR`, `ELECTRON MICROSCOPY`, ...)
    /// had thermal $B$-factors suppressed (`plddt_per_residue = None`).
    ExperimentalBfactorSuppressed { method: String },
    /// Experimental $B$-factors were explicitly forced into `pLDDT` via `force_bfactor_plddt = true`.
    ForcedExperimentalBfactorAsPlddt { method: String },
    /// Coordinate file contained `B_iso_or_equiv > 100.0` and had $B$-factors suppressed.
    OutOfRangeBfactorSuppressed,
    /// `_ma_qa_metric_local` referenced a non-`pLDDT` quality metric (e.g., `QMEANDisCo`) or lacked
    /// a matching `pLDDT` definition in `_ma_qa_metric` and was rejected (`21.W1:P2-6`).
    NonPlddtQaMetricIgnored { metric_name: String },
    /// Constant or all-zero `B_iso_or_equiv` / `tempFactor` values (e.g. all `0.0` or all `0.5`)
    /// were rejected as dummy $B$-factors rather than `pLDDT` (`21.W1:P0-2`).
    ConstantOrZeroBfactorRejected,
    /// Headerless or unverified PDB/mmCIF `B_iso_or_equiv` / `tempFactor` values (`<= 100`) were
    /// suppressed because positive ModelArchive / AFDB provenance was absent (`21.W1:P0-2`).
    UnverifiedBfactorNotTreatedAsPlddt,
}

/// Options controlling `.cif` / `.pdb` coordinate and `pLDDT` ingestion.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AfIngestOptions {
    /// Optional target polymer chain identifier (defaults to the first polymer chain in Model 1).
    pub target_chain_id: Option<String>,
    /// Override the crystallographic / experimental $B$-factor guard and interpret `B_iso_or_equiv` /
    /// `tempFactor` as `pLDDT` even when `EXPDTA` / `_exptl.method` declares experimental coordinates.
    pub force_bfactor_plddt: bool,
}

/// Parsed polymer residue from an AlphaFold or experimental `.cif` / `.pdb` model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AfResidueRecord {
    pub seq_index: usize,
    pub seq_id: i32,
    pub label_seq_id: Option<i32>,
    pub chain_id: String,
    pub res_name: String,
    pub n: [f64; 3],
    pub ca: [f64; 3],
    pub c: [f64; 3],
    pub o: [f64; 3],
    pub cb: Option<[f64; 3]>,
    pub explicit_h: Option<[f64; 3]>,
    pub atoms: BTreeMap<String, [f64; 3]>,
    pub plddt: Option<f64>,
}

/// Parsed 3D coordinate model with provenance, per-residue `pLDDT` (`[0, 100]`), and audit diagnostics.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AfStructureModel {
    pub entry_id: Option<String>,
    pub detected_pdb_id: Option<String>,
    pub detected_uniprot_id: Option<String>,
    pub exptl_method: Option<String>,
    pub chain_id: String,
    pub provenance: CoordinateProvenance,
    pub plddt_source: PlddtSource,
    pub plddt_per_residue: Option<Vec<f64>>,
    pub residues: Vec<AfResidueRecord>,
    pub diagnostics: Vec<AfIngestDiagnostic>,
}

impl AfStructureModel {
    /// Sequence of 3-letter residue codes (`0..L`).
    pub fn sequence_3letter(&self) -> Vec<String> {
        self.residues.iter().map(|r| r.res_name.clone()).collect()
    }

    /// Sequence numbering (`seq_id` per residue).
    pub fn seq_ids(&self) -> Vec<i32> {
        self.residues.iter().map(|r| r.seq_id).collect()
    }

    /// Extract $\text{C}^\alpha$ coordinates (`Vec<[f64; 3]>`) for `TM-score` `HomologFirewall` Layer 4.
    pub fn ca_coords(&self) -> Vec<[f64; 3]> {
        self.residues.iter().map(|r| r.ca).collect()
    }

    /// Fail-closed check verifying that `self` has positive evidence of being a predicted structure
    /// model with valid per-residue `pLDDT` values, or that `treat_as_model` was explicitly enabled
    /// (`21.W1:P0-2`, `P2-6`).
    pub fn require_predicted_model(&self, treat_as_model: bool) -> Result<(), String> {
        require_predicted_model(self, treat_as_model)
    }

    /// Build `BackboneResidueCoords` for clean-room Kabsch–Sander secondary-structure classification.
    pub fn backbone_coords_for_ss(&self) -> Vec<BackboneResidueCoords> {
        self.residues
            .iter()
            .map(|r| BackboneResidueCoords {
                seq_id: r.seq_id,
                res_name: r.res_name.clone(),
                chain_id: r.chain_id.clone(),
                n: r.n,
                ca: r.ca,
                c: r.c,
                o: r.o,
                explicit_h: r.explicit_h,
            })
            .collect()
    }

    /// Convert into `nmr_core::noesy::PdbChain` (completing `HA`, `CB`, `QB`, and amide `H` when absent).
    pub fn to_pdb_chain(&self) -> PdbChain {
        let m = self.residues.len();
        let mut out = Vec::with_capacity(m);
        for i in 0..m {
            let r = &self.residues[i];
            let (rec_cb, rec_ha, rec_ha2, rec_ha3, rec_qb) =
                reconstruct_ca_substituents(&r.res_name, r.n, r.ca, r.c);
            let cb = if r.res_name == "GLY" {
                None
            } else {
                r.cb.or(rec_cb)
            };
            let qb = if r.res_name == "GLY" {
                None
            } else if let Some(cb_pos) = cb {
                let dir = v_unit(v_sub(cb_pos, r.ca));
                Some(v_add(cb_pos, v_scale(dir, 0.3633)))
            } else {
                rec_qb
            };
            let (ha, ha2, ha3) = if r.res_name == "GLY" {
                let g2 = r.atoms.get("HA2").copied().or(rec_ha2);
                let g3 = r.atoms.get("HA3").copied().or(rec_ha3);
                let mid = match (g2, g3) {
                    (Some(a), Some(b)) => v_scale(v_add(a, b), 0.5),
                    (Some(a), None) => a,
                    _ => r.atoms.get("HA").copied().unwrap_or(rec_ha),
                };
                (mid, g2, g3)
            } else {
                (
                    r.atoms
                        .get("HA")
                        .copied()
                        .or_else(|| r.atoms.get("HA2").copied())
                        .unwrap_or(rec_ha),
                    None,
                    None,
                )
            };
            let h = if r.res_name == "PRO" {
                None
            } else if let Some(h_pos) = r.explicit_h {
                Some(h_pos)
            } else if i > 0 && v_dist(self.residues[i - 1].c, r.n) <= 2.5 {
                let u1 = v_unit(v_sub(r.n, self.residues[i - 1].c));
                let u2 = v_unit(v_sub(r.n, r.ca));
                Some(v_add(r.n, v_scale(v_unit(v_add(u1, u2)), 1.010)))
            } else {
                None
            };
            let ch_char = r.chain_id.chars().next().unwrap_or('A');
            out.push(PdbResidue {
                seq_id: r.seq_id,
                res_name: r.res_name.clone(),
                chain_id: ch_char,
                n: r.n,
                h,
                ca: r.ca,
                ha,
                ha2,
                ha3,
                c: r.c,
                o: r.o,
                cb,
                qb,
            });
        }
        PdbChain { residues: out }
    }
}

/// Fail-closed provenance gate verifying that `model` has positive evidence of being a predicted
/// structure model carrying valid per-residue `pLDDT` values (`_ma_qa_metric_local` linked to an
/// explicit `pLDDT` metric in `_ma_qa_metric`, or a verified ModelArchive/AFDB `_ma_protocol_step` /
/// `AF-*` mmCIF block with non-constant `[0, 100]` `B_iso_or_equiv`), OR that the caller explicitly
/// passed `treat_as_model == true` (`--treat-as-model` / `--force-bfactor-plddt`) with valid
/// non-constant `pLDDT` values (`21.W1:P0-2`, `21.W1:P2-6`).
pub fn require_predicted_model(
    model: &AfStructureModel,
    treat_as_model: bool,
) -> Result<(), String> {
    if model.residues.is_empty() {
        return Err("predicted model check failed: structure contains zero residues".to_string());
    }
    if model
        .diagnostics
        .contains(&AfIngestDiagnostic::ConstantOrZeroBfactorRejected)
    {
        return Err(format!(
            "predicted model check failed (chain '{}'): B_iso_or_equiv / tempFactor column is all-zero or constant (e.g. 0.0 or 0.5) and cannot be interpreted as pLDDT (diagnostics: {:?})",
            model.chain_id, model.diagnostics
        ));
    }
    let has_full_plddt = model
        .plddt_per_residue
        .as_ref()
        .is_some_and(|v| v.len() == model.residues.len());

    if model.provenance == CoordinateProvenance::ComputationalModel
        && has_full_plddt
        && matches!(
            model.plddt_source,
            PlddtSource::MaQaMetricLocal | PlddtSource::BFactorColumn
        )
    {
        return Ok(());
    }

    if treat_as_model
        && has_full_plddt
        && matches!(
            model.plddt_source,
            PlddtSource::MaQaMetricLocal
                | PlddtSource::BFactorColumn
                | PlddtSource::ForcedExperimentalBFactor
        )
    {
        return Ok(());
    }

    Err(format!(
        "predicted model provenance check failed for chain '{}' (entry_id={:?}, provenance={:?}, exptl_method={:?}, plddt_source={:?}, diagnostics={:?}): positive ModelArchive/AFDB pLDDT evidence (_ma_qa_metric_local with _ma_qa_metric name='pLDDT' or verified AFDB/ModelArchive mmCIF block) is required; pass --treat-as-model / --force-bfactor-plddt only if B-factors genuinely encode pLDDT",
        model.chain_id,
        model.entry_id,
        model.provenance,
        model.exptl_method,
        model.plddt_source,
        model.diagnostics
    ))
}

/// Parse an AlphaFold or experimental `.cif` / `.mmcif` / `.pdb` file from disk (`<= 64 MiB` cap).
pub fn parse_af_structure_model(
    path: impl AsRef<Path>,
    opts: &AfIngestOptions,
) -> Result<AfStructureModel, String> {
    let p = path.as_ref();
    let meta = fs::metadata(p)
        .map_err(|e| format!("failed to stat structure file '{}': {e}", p.display()))?;
    if meta.len() > MAX_AF_STRUCTURE_BYTES as u64 {
        return Err(format!(
            "structure file '{}' is {} bytes, exceeding MAX_AF_STRUCTURE_BYTES ({} bytes / 64 MiB)",
            p.display(),
            meta.len(),
            MAX_AF_STRUCTURE_BYTES
        ));
    }
    let bytes =
        fs::read(p).map_err(|e| format!("failed to read structure file '{}': {e}", p.display()))?;
    if bytes.len() > MAX_AF_STRUCTURE_BYTES {
        return Err(format!(
            "structure file '{}' exceeds MAX_AF_STRUCTURE_BYTES ({} bytes)",
            p.display(),
            MAX_AF_STRUCTURE_BYTES
        ));
    }
    if bytes.contains(&0u8) {
        return Err(format!(
            "structure file '{}' contains binary NUL bytes",
            p.display()
        ));
    }
    let text = std::str::from_utf8(&bytes)
        .map_err(|e| format!("structure file '{}' is not valid UTF-8: {e}", p.display()))?;
    parse_af_structure_model_str(text, opts)
}

/// Parse an AlphaFold or experimental `.cif` / `.mmcif` / `.pdb` string (`<= 64 MiB` cap).
pub fn parse_af_structure_model_str(
    text: &str,
    opts: &AfIngestOptions,
) -> Result<AfStructureModel, String> {
    if text.len() > MAX_AF_STRUCTURE_BYTES {
        return Err(format!(
            "structure text is {} bytes, exceeding MAX_AF_STRUCTURE_BYTES ({} bytes / 64 MiB)",
            text.len(),
            MAX_AF_STRUCTURE_BYTES
        ));
    }
    if text.as_bytes().contains(&0u8) {
        return Err("structure text contains binary NUL bytes".to_string());
    }
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err("structure input is empty".to_string());
    }

    if looks_like_mmcif(trimmed) {
        parse_mmcif_af_model(text, opts)
    } else {
        parse_pdb_af_model(text, opts)
    }
}

fn looks_like_mmcif(trimmed: &str) -> bool {
    for line in trimmed.lines().take(80) {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        if t.starts_with("data_")
            || t.eq_ignore_ascii_case("loop_")
            || t.starts_with("_atom_site.")
            || t.starts_with("_entry.")
            || t.starts_with("_ma_")
        {
            return true;
        }
        if t.starts_with("HEADER")
            || t.starts_with("ATOM  ")
            || t.starts_with("HETATM")
            || t.starts_with("REMARK")
            || t.starts_with("EXPDTA")
            || t.starts_with("MODEL ")
        {
            return false;
        }
    }
    trimmed.contains("_atom_site.")
}

fn is_experimental_method(method: &str) -> bool {
    let up = method.to_ascii_uppercase();
    [
        "X-RAY DIFFRACTION",
        "SOLUTION NMR",
        "SOLID-STATE NMR",
        "ELECTRON MICROSCOPY",
        "NEUTRON DIFFRACTION",
        "FIBER DIFFRACTION",
        "ELECTRON CRYSTALLOGRAPHY",
        "POWDER DIFFRACTION",
        "INFRARED SPECTROSCOPY",
    ]
    .iter()
    .any(|kw| up.contains(kw))
}

fn normalize_polymer_res_name(raw: &str) -> String {
    let up = raw.trim().to_ascii_uppercase();
    match up.as_str() {
        "MSE" => "MET".to_string(),
        "HID" | "HIE" | "HIP" | "HSD" | "HSE" | "HSP" => "HIS".to_string(),
        "CYX" | "CYM" => "CYS".to_string(),
        _ => up,
    }
}

fn is_standard_or_common_amino_acid(res_name: &str) -> bool {
    matches!(
        res_name,
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
            | "SEC"
            | "PYL"
            | "UNK"
    )
}

fn extract_afdb_uniprot_from_id(id_raw: &str) -> Option<String> {
    let up = id_raw.trim().to_ascii_uppercase();
    // Matches e.g. `AF-P0CG48-F1` or `AF_AFP0CG48F1`
    if let Some(rest) = up.strip_prefix("AF-") {
        if let Some((acc, _)) = rest.split_once('-') {
            if (6..=10).contains(&acc.len())
                && acc
                    .bytes()
                    .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit())
            {
                return Some(acc.to_string());
            }
        }
    }
    None
}

#[derive(Default)]
struct RawPolymerRes {
    seq_id: i32,
    label_seq_id: Option<i32>,
    chain_id: String,
    res_name: String,
    n: Option<[f64; 3]>,
    ca: Option<[f64; 3]>,
    c: Option<[f64; 3]>,
    o: Option<[f64; 3]>,
    cb: Option<[f64; 3]>,
    explicit_h: Option<[f64; 3]>,
    atoms: BTreeMap<String, [f64; 3]>,
    b_ca: Option<f64>,
    b_backbone_sum: f64,
    b_backbone_cnt: usize,
    b_backbone_all_positive: bool,
}

impl RawPolymerRes {
    fn new(seq_id: i32, label_seq_id: Option<i32>, chain_id: String, res_name: String) -> Self {
        Self {
            seq_id,
            label_seq_id,
            chain_id,
            res_name,
            b_backbone_all_positive: true,
            ..Self::default()
        }
    }

    fn record_atom(&mut self, atom_name: &str, xyz: [f64; 3], b_iso: Option<f64>) {
        self.atoms.entry(atom_name.to_string()).or_insert(xyz);
        match atom_name {
            "N" => {
                if self.n.is_none() {
                    self.n = Some(xyz);
                    if let Some(b) = b_iso {
                        self.b_backbone_sum += b;
                        self.b_backbone_cnt += 1;
                        if b <= 0.0 {
                            self.b_backbone_all_positive = false;
                        }
                    }
                }
            }
            "CA" => {
                if self.ca.is_none() {
                    self.ca = Some(xyz);
                    self.b_ca = b_iso;
                    if let Some(b) = b_iso {
                        self.b_backbone_sum += b;
                        self.b_backbone_cnt += 1;
                        if b <= 0.0 {
                            self.b_backbone_all_positive = false;
                        }
                    }
                }
            }
            "C" => {
                if self.c.is_none() {
                    self.c = Some(xyz);
                    if let Some(b) = b_iso {
                        self.b_backbone_sum += b;
                        self.b_backbone_cnt += 1;
                        if b <= 0.0 {
                            self.b_backbone_all_positive = false;
                        }
                    }
                }
            }
            "O" | "OT1" | "OXT" => {
                if self.o.is_none() {
                    self.o = Some(xyz);
                    if atom_name == "O" {
                        if let Some(b) = b_iso {
                            self.b_backbone_sum += b;
                            self.b_backbone_cnt += 1;
                            if b <= 0.0 {
                                self.b_backbone_all_positive = false;
                            }
                        }
                    }
                }
            }
            "CB" => {
                if self.cb.is_none() {
                    self.cb = Some(xyz);
                }
            }
            "H" | "HN" if self.explicit_h.is_none() => {
                self.explicit_h = Some(xyz);
            }
            _ => {}
        }
    }

    fn effective_bfactor(&self) -> Option<f64> {
        if self.b_backbone_cnt >= 3 && self.b_backbone_all_positive {
            Some(self.b_backbone_sum / (self.b_backbone_cnt as f64))
        } else {
            self.b_ca
        }
    }
}

fn median_of_slice(vals: &[f64]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    let mut sorted = vals.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        0.5 * (sorted[mid - 1] + sorted[mid])
    } else {
        sorted[mid]
    }
}

fn is_dummy_constant_or_zero_plddt(vals: &[f64], is_explicit_ma_local: bool) -> bool {
    if vals.is_empty() {
        return true;
    }
    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    for &v in vals {
        if v.is_finite() {
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }
    }
    if !min_v.is_finite() || !max_v.is_finite() || max_v <= 1.0e-6 {
        return true;
    }
    // Constant unit-interval values (e.g. all 0.50 dummy B-factors) must never be rescaled to 50
    if max_v <= 1.0 && (max_v - min_v) <= 1.0e-6 {
        return true;
    }
    if !is_explicit_ma_local && vals.len() >= 2 && (max_v - min_v) <= 1.0e-6 {
        return true;
    }
    false
}

fn normalize_plddt_vector(
    mut vals: Vec<f64>,
    diagnostics: &mut Vec<AfIngestDiagnostic>,
    ctx: &str,
) -> Result<Vec<f64>, String> {
    if vals.is_empty() {
        return Err(format!("{ctx}: empty pLDDT vector"));
    }
    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    for (idx, &v) in vals.iter().enumerate() {
        if !v.is_finite() {
            return Err(format!("{ctx}: non-finite pLDDT at residue index {idx}"));
        }
        if !(-1.0e-6..=100.000_1).contains(&v) {
            return Err(format!(
                "{ctx}: pLDDT value {v:.4} at residue index {idx} outside [0, 100]"
            ));
        }
        min_v = min_v.min(v);
        max_v = max_v.max(v);
    }
    let med = median_of_slice(&vals);
    if max_v <= 1.0 && med > 0.15 && (max_v - min_v) > 1.0e-6 {
        for v in &mut vals {
            *v = (*v * 100.0).clamp(0.0, 100.0);
        }
        diagnostics.push(AfIngestDiagnostic::RescaledUnitPlddtTo100);
    } else {
        for v in &mut vals {
            *v = v.clamp(0.0, 100.0);
        }
    }
    Ok(vals)
}

#[allow(clippy::too_many_arguments)]
fn finalize_af_structure_model(
    raw_residues: Vec<RawPolymerRes>,
    entry_id: Option<String>,
    detected_pdb_id: Option<String>,
    detected_uniprot_id: Option<String>,
    exptl_method: Option<String>,
    has_computational_marker: bool,
    ma_local_plddt: Option<BTreeMap<(String, i32), f64>>,
    ma_local_by_seq: Option<BTreeMap<i32, f64>>,
    pre_diagnostics: Vec<AfIngestDiagnostic>,
    opts: &AfIngestOptions,
) -> Result<AfStructureModel, String> {
    let complete: Vec<RawPolymerRes> = raw_residues
        .into_iter()
        .filter(|r| r.n.is_some() && r.ca.is_some() && r.c.is_some())
        .collect();
    if complete.is_empty() {
        return Err(
            "no complete protein backbone (N, CA, C) residues found in structure".to_string(),
        );
    }

    let m = complete.len();
    let chain_id = complete[0].chain_id.clone();
    let mut diagnostics = pre_diagnostics;

    // Reconstruct missing carbonyl O if needed (`C=O` ~ 1.231 A)
    let mut o_coords = Vec::with_capacity(m);
    for i in 0..m {
        let r = &complete[i];
        let (Some(n), Some(ca), Some(c)) = (r.n, r.ca, r.c) else {
            return Err("internal backbone invariant violated".to_string());
        };
        let o_pos = if let Some(o_exp) = r.o {
            o_exp
        } else if i + 1 < m {
            if let Some(n_next) = complete[i + 1].n {
                if v_dist(c, n_next) <= 2.5 {
                    let w1 = v_unit(v_sub(ca, c));
                    let w2 = v_unit(v_sub(n_next, c));
                    v_sub(c, v_scale(v_unit(v_add(w1, w2)), 1.231))
                } else {
                    let w1 = v_unit(v_sub(n, ca));
                    let w2 = v_unit(v_sub(ca, c));
                    v_add(c, v_scale(v_unit(v_sub(w2, w1)), 1.231))
                }
            } else {
                [c[0] + 1.231, c[1], c[2]]
            }
        } else {
            let w1 = v_unit(v_sub(n, ca));
            let w2 = v_unit(v_sub(ca, c));
            v_add(c, v_scale(v_unit(v_sub(w2, w1)), 1.231))
        };
        o_coords.push(o_pos);
    }

    // Resolve provenance & per-residue pLDDT under fail-closed positive-evidence rules (`21.W1:P0-2`, `P2-6`)
    let (provenance, plddt_source, plddt_per_residue) = if let Some(ref ma_map) = ma_local_plddt {
        let mut vals = Vec::with_capacity(m);
        for r in &complete {
            let lseq = r.label_seq_id.unwrap_or(r.seq_id);
            let v_opt = ma_map
                .get(&(r.chain_id.clone(), lseq))
                .copied()
                .or_else(|| ma_map.get(&(r.chain_id.clone(), r.seq_id)).copied())
                .or_else(|| {
                    ma_local_by_seq
                        .as_ref()
                        .and_then(|by_s| by_s.get(&lseq).or_else(|| by_s.get(&r.seq_id)).copied())
                });
            let Some(v) = v_opt else {
                return Err(format!(
                    "_ma_qa_metric_local is missing pLDDT entry for chain '{}' residue seq_id {} (label_seq_id {:?})",
                    r.chain_id, r.seq_id, r.label_seq_id
                ));
            };
            vals.push(v);
        }
        let norm = normalize_plddt_vector(vals, &mut diagnostics, "_ma_qa_metric_local")?;
        if is_dummy_constant_or_zero_plddt(&norm, true) {
            diagnostics.push(AfIngestDiagnostic::ConstantOrZeroBfactorRejected);
            (
                CoordinateProvenance::Unknown,
                PlddtSource::Unavailable,
                None,
            )
        } else {
            (
                CoordinateProvenance::ComputationalModel,
                PlddtSource::MaQaMetricLocal,
                Some(norm),
            )
        }
    } else {
        let exp_method_opt = exptl_method
            .as_deref()
            .filter(|meth| is_experimental_method(meth));
        if let Some(meth) = exp_method_opt {
            if !opts.force_bfactor_plddt {
                diagnostics.push(AfIngestDiagnostic::ExperimentalBfactorSuppressed {
                    method: meth.to_string(),
                });
                (
                    CoordinateProvenance::Experimental,
                    PlddtSource::Unavailable,
                    None,
                )
            } else {
                let mut raw_b = Vec::with_capacity(m);
                for r in &complete {
                    let b = r.effective_bfactor().ok_or_else(|| {
                        format!(
                            "--force-bfactor-plddt requested, but residue {} {} lacks B_iso_or_equiv / tempFactor",
                            r.res_name, r.seq_id
                        )
                    })?;
                    raw_b.push(b);
                }
                if is_dummy_constant_or_zero_plddt(&raw_b, false) {
                    diagnostics.push(AfIngestDiagnostic::ConstantOrZeroBfactorRejected);
                    (
                        CoordinateProvenance::Experimental,
                        PlddtSource::Unavailable,
                        None,
                    )
                } else {
                    let norm =
                        normalize_plddt_vector(raw_b, &mut diagnostics, "forced B-factor pLDDT")?;
                    diagnostics.push(AfIngestDiagnostic::ForcedExperimentalBfactorAsPlddt {
                        method: meth.to_string(),
                    });
                    (
                        CoordinateProvenance::Experimental,
                        PlddtSource::ForcedExperimentalBFactor,
                        Some(norm),
                    )
                }
            }
        } else {
            // Check B_iso_or_equiv / tempFactor
            let mut raw_b = Vec::with_capacity(m);
            let mut all_present = true;
            for r in &complete {
                if let Some(b) = r.effective_bfactor() {
                    if !b.is_finite() {
                        return Err(format!(
                            "non-finite B_iso_or_equiv / tempFactor on residue {} {}",
                            r.res_name, r.seq_id
                        ));
                    }
                    if b < -1.0e-6 {
                        return Err(format!(
                            "negative B_iso_or_equiv / tempFactor ({b:.3}) on residue {} {}",
                            r.res_name, r.seq_id
                        ));
                    }
                    raw_b.push(b);
                } else {
                    all_present = false;
                    break;
                }
            }

            if !all_present || raw_b.is_empty() {
                (
                    CoordinateProvenance::Unknown,
                    PlddtSource::Unavailable,
                    None,
                )
            } else if raw_b.iter().any(|&b| b > 100.000_1) {
                if has_computational_marker || opts.force_bfactor_plddt {
                    return Err(
                        "computational model B_iso_or_equiv / tempFactor exceeds 100.0 (invalid pLDDT)"
                            .to_string(),
                    );
                }
                diagnostics.push(AfIngestDiagnostic::OutOfRangeBfactorSuppressed);
                (
                    CoordinateProvenance::Experimental,
                    PlddtSource::Unavailable,
                    None,
                )
            } else if is_dummy_constant_or_zero_plddt(&raw_b, false) {
                diagnostics.push(AfIngestDiagnostic::ConstantOrZeroBfactorRejected);
                (
                    CoordinateProvenance::Unknown,
                    PlddtSource::Unavailable,
                    None,
                )
            } else if has_computational_marker || opts.force_bfactor_plddt {
                let norm = normalize_plddt_vector(raw_b, &mut diagnostics, "B_iso_or_equiv pLDDT")?;
                (
                    CoordinateProvenance::ComputationalModel,
                    PlddtSource::BFactorColumn,
                    Some(norm),
                )
            } else {
                // Headerless or unverified PDB/mmCIF with B <= 100: fail closed (`21.W1:P0-2`)!
                diagnostics.push(AfIngestDiagnostic::UnverifiedBfactorNotTreatedAsPlddt);
                (
                    CoordinateProvenance::Unknown,
                    PlddtSource::Unavailable,
                    None,
                )
            }
        }
    };

    let mut residues = Vec::with_capacity(m);
    for (idx, (r, o_pos)) in complete.into_iter().zip(o_coords).enumerate() {
        let (Some(n), Some(ca), Some(c)) = (r.n, r.ca, r.c) else {
            return Err("internal backbone coordinate invariant violated".to_string());
        };
        let plddt = plddt_per_residue.as_ref().map(|v| v[idx]);
        residues.push(AfResidueRecord {
            seq_index: idx,
            seq_id: r.seq_id,
            label_seq_id: r.label_seq_id,
            chain_id: r.chain_id,
            res_name: r.res_name,
            n,
            ca,
            c,
            o: o_pos,
            cb: r.cb,
            explicit_h: r.explicit_h,
            atoms: r.atoms,
            plddt,
        });
    }

    Ok(AfStructureModel {
        entry_id,
        detected_pdb_id,
        detected_uniprot_id,
        exptl_method,
        chain_id,
        provenance,
        plddt_source,
        plddt_per_residue,
        residues,
        diagnostics,
    })
}

// ---- mmCIF Tokenizer & Parser ----

#[derive(Clone, Debug)]
struct CifTok {
    text: String,
    quoted: bool,
}

fn tokenize_cif_line(line: &str, out: &mut Vec<CifTok>) -> Result<(), String> {
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
                    let seg = &line[start..i];
                    out.push(CifTok {
                        text: seg.to_string(),
                        quoted: true,
                    });
                    i += 1;
                    closed = true;
                    break;
                }
                i += 1;
            }
            if !closed {
                return Err("unterminated quoted string in mmCIF input".to_string());
            }
        } else {
            let start = i;
            while i < n && !bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            out.push(CifTok {
                text: line[start..i].to_string(),
                quoted: false,
            });
        }
    }
    Ok(())
}

fn tokenize_mmcif(text: &str) -> Result<Vec<CifTok>, String> {
    let mut tokens = Vec::new();
    let mut in_semi = false;
    let mut semi_buf = String::new();
    for line in text.lines() {
        if in_semi {
            if let Some(rest) = line.strip_prefix(';') {
                tokens.push(CifTok {
                    text: semi_buf.trim().to_string(),
                    quoted: true,
                });
                semi_buf.clear();
                in_semi = false;
                tokenize_cif_line(rest, &mut tokens)?;
            } else {
                if !semi_buf.is_empty() {
                    semi_buf.push('\n');
                }
                semi_buf.push_str(line);
            }
        } else if let Some(rest) = line.strip_prefix(';') {
            in_semi = true;
            if !rest.trim().is_empty() {
                semi_buf.push_str(rest.trim());
            }
        } else {
            tokenize_cif_line(line, &mut tokens)?;
        }
    }
    if in_semi {
        return Err("unterminated semicolon text block in mmCIF input".to_string());
    }
    Ok(tokens)
}

#[inline]
fn is_cif_control_tok(tok: &CifTok) -> bool {
    !tok.quoted
        && (tok.text.eq_ignore_ascii_case("loop_")
            || tok.text.eq_ignore_ascii_case("stop_")
            || tok.text.starts_with("data_")
            || tok.text.starts_with("save_")
            || tok.text.starts_with('_'))
}

fn parse_req_finite_f64(raw: &str, col_name: &str, ctx: &str) -> Result<f64, String> {
    let t = raw.trim();
    let v: f64 = t
        .parse()
        .map_err(|e| format!("{ctx}: invalid {col_name} '{t}': {e}"))?;
    if !v.is_finite() {
        return Err(format!("{ctx}: non-finite {col_name} '{t}'"));
    }
    Ok(v)
}

fn parse_opt_finite_f64(raw: &str, col_name: &str, ctx: &str) -> Result<Option<f64>, String> {
    let t = raw.trim();
    if t.is_empty() || t == "." || t == "?" {
        return Ok(None);
    }
    let v: f64 = t
        .parse()
        .map_err(|e| format!("{ctx}: invalid {col_name} '{t}': {e}"))?;
    if !v.is_finite() {
        return Err(format!("{ctx}: non-finite {col_name} '{t}'"));
    }
    Ok(Some(v))
}

fn parse_mmcif_af_model(text: &str, opts: &AfIngestOptions) -> Result<AfStructureModel, String> {
    let tokens = tokenize_mmcif(text)?;
    let mut entry_id: Option<String> = None;
    let mut detected_pdb_id: Option<String> = None;
    let mut detected_uniprot_id: Option<String> = None;
    let mut exptl_methods: Vec<String> = Vec::new();
    let mut has_computational_marker = false;

    // `_ma_qa_metric` definitions: metric_id -> (name_lower, mode_lower)
    let mut ma_metric_defs: BTreeMap<String, (String, String)> = BTreeMap::new();
    // Raw `_ma_qa_metric_local` rows: (asym_id, label_seq_id, metric_id, value)
    let mut ma_local_rows: Vec<(String, i32, String, f64)> = Vec::new();

    let mut raw_residues: Vec<RawPolymerRes> = Vec::new();
    let mut res_idx_map: BTreeMap<(String, i32), usize> = BTreeMap::new();
    let mut selected_chain: Option<String> = opts
        .target_chain_id
        .as_ref()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    let mut selected_model_num: Option<String> = None;

    let mut scalar_ma_id: Option<String> = None;
    let mut scalar_ma_name: Option<String> = None;
    let mut scalar_ma_mode: Option<String> = None;
    let mut scalar_ref_db: Option<String> = None;
    let mut scalar_ref_acc: Option<String> = None;

    let mut idx = 0usize;
    while idx < tokens.len() {
        let tok = &tokens[idx];
        if !tok.quoted && tok.text.starts_with("data_") {
            if entry_id.is_none() && tok.text.len() > 5 {
                let d_id = tok.text[5..].to_string();
                if let Some(unp) = extract_afdb_uniprot_from_id(&d_id) {
                    if detected_uniprot_id.is_none() {
                        detected_uniprot_id = Some(unp);
                    }
                    has_computational_marker = true;
                }
                entry_id = Some(d_id);
            }
            idx += 1;
            continue;
        }

        // Scalar `_tag value`
        if !tok.quoted && tok.text.starts_with('_') {
            if idx + 1 >= tokens.len() || is_cif_control_tok(&tokens[idx + 1]) {
                return Err(format!("truncated mmCIF scalar tag '{}'", tok.text));
            }
            let key_low = tok.text.to_ascii_lowercase();
            let val = tokens[idx + 1].text.trim();
            if key_low.starts_with("_ma_protocol_step.")
                || key_low.starts_with("_ma_data.")
                || key_low.starts_with("_ma_model_list.")
            {
                has_computational_marker = true;
            }
            match key_low.as_str() {
                "_entry.id" if val != "." && val != "?" && !val.is_empty() => {
                    entry_id = Some(val.to_string());
                    if val.len() == 4 && val.bytes().all(|b| b.is_ascii_alphanumeric()) {
                        detected_pdb_id = Some(val.to_ascii_uppercase());
                    }
                    if let Some(unp) = extract_afdb_uniprot_from_id(val) {
                        if detected_uniprot_id.is_none() {
                            detected_uniprot_id = Some(unp);
                        }
                        has_computational_marker = true;
                    }
                }
                "_exptl.method" if val != "." && val != "?" && !val.is_empty() => {
                    exptl_methods.push(val.to_string());
                }
                "_ma_qa_metric.id" => scalar_ma_id = Some(val.to_string()),
                "_ma_qa_metric.name" | "_ma_qa_metric.type" => {
                    if scalar_ma_name.is_none() || key_low == "_ma_qa_metric.name" {
                        scalar_ma_name = Some(val.to_ascii_lowercase());
                    }
                }
                "_ma_qa_metric.mode" => scalar_ma_mode = Some(val.to_ascii_lowercase()),
                "_struct_ref.db_name" => scalar_ref_db = Some(val.to_ascii_uppercase()),
                "_struct_ref.pdbx_db_accession" if val != "." && val != "?" && !val.is_empty() => {
                    scalar_ref_acc = Some(val.to_ascii_uppercase());
                }
                _ => {}
            }
            idx += 2;
            continue;
        }

        if !tok.quoted && tok.text.eq_ignore_ascii_case("loop_") {
            idx += 1;
            let mut headers = Vec::new();
            while idx < tokens.len() && !tokens[idx].quoted && tokens[idx].text.starts_with('_') {
                headers.push(tokens[idx].text.to_ascii_lowercase());
                idx += 1;
            }
            let ncols = headers.len();
            if ncols == 0 {
                return Err("mmCIF loop_ contains zero column headers".to_string());
            }
            if headers.iter().any(|h| {
                h.starts_with("_ma_protocol_step.")
                    || h.starts_with("_ma_data.")
                    || h.starts_with("_ma_model_list.")
            }) {
                has_computational_marker = true;
            }

            let find_col = |cands: &[&str]| -> Option<usize> {
                cands
                    .iter()
                    .find_map(|target| headers.iter().position(|h| h == target))
            };

            let is_atom_site = headers.iter().any(|h| h.starts_with("_atom_site."));
            let is_ma_metric = headers.iter().any(|h| h.starts_with("_ma_qa_metric."));
            let is_ma_local = headers
                .iter()
                .any(|h| h.starts_with("_ma_qa_metric_local."));
            let is_exptl = headers.iter().any(|h| h.starts_with("_exptl."));
            let is_struct_ref = headers.iter().any(|h| h.starts_with("_struct_ref."));

            // Collect rows strictly checking exact `ncols` alignment before next control token
            let mut rows: Vec<&[CifTok]> = Vec::new();
            let mut row_no = 0usize;
            while idx < tokens.len() && !is_cif_control_tok(&tokens[idx]) {
                row_no += 1;
                if let Some(off) = tokens[idx..]
                    .iter()
                    .take(ncols)
                    .position(is_cif_control_tok)
                {
                    return Err(format!(
                        "truncated mmCIF loop ({}) row {row_no}: expected {ncols} columns before '{}', got {off}",
                        headers[0],
                        tokens[idx + off].text
                    ));
                }
                if idx + ncols > tokens.len() {
                    return Err(format!(
                        "truncated mmCIF loop ({}) row {row_no}: expected {ncols} columns, got {}",
                        headers[0],
                        tokens.len() - idx
                    ));
                }
                rows.push(&tokens[idx..idx + ncols]);
                idx += ncols;
            }
            if rows.is_empty() && (is_atom_site || is_ma_local) {
                return Err(format!(
                    "empty mmCIF loop ({}) with zero data rows",
                    headers[0]
                ));
            }

            if is_exptl {
                if let Some(c_m) = find_col(&["_exptl.method"]) {
                    for r in &rows {
                        let m_str = r[c_m].text.trim();
                        if !m_str.is_empty() && m_str != "." && m_str != "?" {
                            exptl_methods.push(m_str.to_string());
                        }
                    }
                }
            } else if is_struct_ref {
                if let (Some(c_db), Some(c_acc)) = (
                    find_col(&["_struct_ref.db_name"]),
                    find_col(&["_struct_ref.pdbx_db_accession"]),
                ) {
                    for r in &rows {
                        if r[c_db].text.trim().eq_ignore_ascii_case("UNP") {
                            let acc = r[c_acc].text.trim().to_ascii_uppercase();
                            if !acc.is_empty() && acc != "." && acc != "?" {
                                detected_uniprot_id = Some(acc);
                            }
                        }
                    }
                }
            } else if is_ma_metric {
                let c_id = find_col(&["_ma_qa_metric.id"])
                    .ok_or_else(|| "mmCIF _ma_qa_metric missing _ma_qa_metric.id".to_string())?;
                let c_name = find_col(&["_ma_qa_metric.name"]);
                let c_type = find_col(&["_ma_qa_metric.type"]);
                let c_mode = find_col(&["_ma_qa_metric.mode"]);
                for r in &rows {
                    let m_id = r[c_id].text.trim().to_string();
                    let nm_part = c_name
                        .map(|c| r[c].text.trim().to_ascii_lowercase())
                        .unwrap_or_default();
                    let tp_part = c_type
                        .map(|c| r[c].text.trim().to_ascii_lowercase())
                        .unwrap_or_default();
                    let m_name = if nm_part.is_empty() {
                        tp_part
                    } else if tp_part.is_empty() || nm_part == tp_part {
                        nm_part
                    } else {
                        format!("{nm_part}/{tp_part}")
                    };
                    let m_mode = c_mode
                        .map(|c| r[c].text.trim().to_ascii_lowercase())
                        .unwrap_or_default();
                    ma_metric_defs.insert(m_id, (m_name, m_mode));
                }
            } else if is_ma_local {
                let c_asym = find_col(&[
                    "_ma_qa_metric_local.label_asym_id",
                    "_ma_qa_metric_local.auth_asym_id",
                ])
                .ok_or_else(|| {
                    "mmCIF _ma_qa_metric_local missing label_asym_id column".to_string()
                })?;
                let c_seq = find_col(&[
                    "_ma_qa_metric_local.label_seq_id",
                    "_ma_qa_metric_local.auth_seq_id",
                ])
                .ok_or_else(|| {
                    "mmCIF _ma_qa_metric_local missing label_seq_id column".to_string()
                })?;
                let c_mid = find_col(&["_ma_qa_metric_local.metric_id"]);
                let c_val = find_col(&["_ma_qa_metric_local.metric_value"]).ok_or_else(|| {
                    "mmCIF _ma_qa_metric_local missing metric_value column".to_string()
                })?;
                for (r_idx, r) in rows.iter().enumerate() {
                    let ctx = format!("_ma_qa_metric_local row {}", r_idx + 1);
                    let asym = r[c_asym].text.trim().to_string();
                    let seq_id: i32 = r[c_seq].text.trim().parse().map_err(|e| {
                        format!("{ctx}: invalid label_seq_id '{}': {e}", r[c_seq].text)
                    })?;
                    let mid = c_mid
                        .map(|c| r[c].text.trim().to_string())
                        .unwrap_or_else(|| "1".to_string());
                    let val = parse_req_finite_f64(&r[c_val].text, "metric_value", &ctx)?;
                    ma_local_rows.push((asym, seq_id, mid, val));
                }
            } else if is_atom_site {
                let c_group = find_col(&["_atom_site.group_pdb"]);
                let c_atom = find_col(&["_atom_site.label_atom_id", "_atom_site.auth_atom_id"])
                    .ok_or_else(|| {
                        "mmCIF _atom_site missing label_atom_id / auth_atom_id".to_string()
                    })?;
                let c_alt = find_col(&["_atom_site.label_alt_id"]);
                let c_comp = find_col(&["_atom_site.label_comp_id", "_atom_site.auth_comp_id"])
                    .ok_or_else(|| {
                        "mmCIF _atom_site missing label_comp_id / auth_comp_id".to_string()
                    })?;
                let c_auth_asym = find_col(&["_atom_site.auth_asym_id"]);
                let c_label_asym = find_col(&["_atom_site.label_asym_id"]);
                let c_auth_seq = find_col(&["_atom_site.auth_seq_id"]);
                let c_label_seq = find_col(&["_atom_site.label_seq_id"]);
                if c_auth_seq.is_none() && c_label_seq.is_none() {
                    return Err("mmCIF _atom_site missing auth_seq_id / label_seq_id".to_string());
                }
                let c_ins = find_col(&["_atom_site.pdbx_pdb_ins_code"]);
                let c_x = find_col(&["_atom_site.cartn_x"])
                    .ok_or_else(|| "mmCIF _atom_site missing Cartn_x".to_string())?;
                let c_y = find_col(&["_atom_site.cartn_y"])
                    .ok_or_else(|| "mmCIF _atom_site missing Cartn_y".to_string())?;
                let c_z = find_col(&["_atom_site.cartn_z"])
                    .ok_or_else(|| "mmCIF _atom_site missing Cartn_z".to_string())?;
                let c_bfac = find_col(&["_atom_site.b_iso_or_equiv"]);
                let c_model = find_col(&["_atom_site.pdbx_pdb_model_num"]);

                for (r_idx, r) in rows.iter().enumerate() {
                    let ctx = format!("mmCIF _atom_site row {}", r_idx + 1);
                    let x = parse_req_finite_f64(&r[c_x].text, "Cartn_x", &ctx)?;
                    let y = parse_req_finite_f64(&r[c_y].text, "Cartn_y", &ctx)?;
                    let z = parse_req_finite_f64(&r[c_z].text, "Cartn_z", &ctx)?;
                    let b_iso = match c_bfac {
                        Some(cb) => parse_opt_finite_f64(&r[cb].text, "B_iso_or_equiv", &ctx)?,
                        None => None,
                    };

                    if let Some(cm) = c_model {
                        let m_tok = r[cm].text.trim();
                        if !m_tok.is_empty() && m_tok != "." && m_tok != "?" {
                            if let Some(ref want_m) = selected_model_num {
                                if m_tok != want_m {
                                    continue;
                                }
                            } else {
                                selected_model_num = Some(m_tok.to_string());
                            }
                        }
                    }

                    let group = c_group.map(|c| r[c].text.trim()).unwrap_or("ATOM");
                    let comp_norm = normalize_polymer_res_name(&r[c_comp].text);
                    if group.eq_ignore_ascii_case("HETATM")
                        && !is_standard_or_common_amino_acid(&comp_norm)
                    {
                        continue;
                    }

                    if let Some(ca) = c_alt {
                        let alt = r[ca].text.trim();
                        if !alt.is_empty() && alt != "." && alt != "?" && alt != "A" && alt != "1" {
                            continue;
                        }
                    }
                    if let Some(ci) = c_ins {
                        let ins = r[ci].text.trim();
                        if !ins.is_empty() && ins != "." && ins != "?" {
                            continue;
                        }
                    }

                    let label_asym = c_label_asym
                        .map(|c| r[c].text.trim())
                        .filter(|s| !s.is_empty() && *s != "." && *s != "?");
                    let auth_asym = c_auth_asym
                        .map(|c| r[c].text.trim())
                        .filter(|s| !s.is_empty() && *s != "." && *s != "?");
                    let chain_tok = label_asym.or(auth_asym).unwrap_or("A").to_string();

                    if let Some(ref want_ch) = selected_chain {
                        let matches_label = label_asym.is_some_and(|la| la == want_ch);
                        let matches_auth = auth_asym.is_some_and(|aa| aa == want_ch);
                        if chain_tok != *want_ch && !matches_label && !matches_auth {
                            continue;
                        }
                    } else {
                        selected_chain = Some(chain_tok.clone());
                    }

                    let label_seq_opt: Option<i32> = c_label_seq.and_then(|c| {
                        let t = r[c].text.trim();
                        if t == "." || t == "?" || t.is_empty() {
                            None
                        } else {
                            t.parse::<i32>().ok()
                        }
                    });
                    let auth_seq_opt: Option<i32> = c_auth_seq.and_then(|c| {
                        let t = r[c].text.trim();
                        if t == "." || t == "?" || t.is_empty() {
                            None
                        } else {
                            t.parse::<i32>().ok()
                        }
                    });
                    let Some(seq_id) = auth_seq_opt.or(label_seq_opt) else {
                        return Err(format!("{ctx}: unparseable sequence number"));
                    };

                    let atom_name = r[c_atom].text.trim().to_ascii_uppercase();
                    let key = (chain_tok.clone(), seq_id);
                    let r_slot = *res_idx_map.entry(key).or_insert_with(|| {
                        let pos = raw_residues.len();
                        raw_residues.push(RawPolymerRes::new(
                            seq_id,
                            label_seq_id_or(label_seq_opt, seq_id),
                            chain_tok.clone(),
                            comp_norm.clone(),
                        ));
                        pos
                    });
                    raw_residues[r_slot].record_atom(&atom_name, [x, y, z], b_iso);
                }
            }
            continue;
        }

        idx += 1;
    }

    if let Some(id_s) = scalar_ma_id {
        ma_metric_defs.insert(
            id_s,
            (
                scalar_ma_name.unwrap_or_default(),
                scalar_ma_mode.unwrap_or_default(),
            ),
        );
    }
    if detected_uniprot_id.is_none()
        && scalar_ref_db.as_deref() == Some("UNP")
        && scalar_ref_acc.is_some()
    {
        detected_uniprot_id = scalar_ref_acc;
    }

    // Filter `_ma_qa_metric_local` rows strictly to verified `pLDDT` metric IDs (`21.W1:P2-6`)
    let mut pre_diagnostics: Vec<AfIngestDiagnostic> = Vec::new();
    let plddt_ids: Vec<String> = ma_metric_defs
        .iter()
        .filter(|(_, (nm, mode))| nm.contains("plddt") && !mode.contains("global"))
        .map(|(k, _)| k.clone())
        .collect();
    if !plddt_ids.is_empty() {
        has_computational_marker = true;
    }

    let (ma_local_plddt, ma_local_by_seq) = if !ma_local_rows.is_empty() {
        if plddt_ids.is_empty() {
            if ma_metric_defs.is_empty() {
                pre_diagnostics.push(AfIngestDiagnostic::NonPlddtQaMetricIgnored {
                    metric_name: "unspecified_ma_qa_metric".to_string(),
                });
            } else {
                for (nm, mode) in ma_metric_defs.values() {
                    if !mode.contains("global") {
                        pre_diagnostics.push(AfIngestDiagnostic::NonPlddtQaMetricIgnored {
                            metric_name: if nm.is_empty() {
                                "unnamed_ma_qa_metric".to_string()
                            } else {
                                nm.clone()
                            },
                        });
                    }
                }
            }
            (None, None)
        } else {
            let mut by_chain_seq: BTreeMap<(String, i32), f64> = BTreeMap::new();
            let mut by_seq: BTreeMap<i32, f64> = BTreeMap::new();
            let mut ignored_mids: Vec<String> = Vec::new();
            for (asym, seq_id, mid, val) in ma_local_rows {
                if plddt_ids.iter().any(|want| want == &mid) {
                    by_chain_seq.entry((asym, seq_id)).or_insert(val);
                    by_seq.entry(seq_id).or_insert(val);
                } else if !ignored_mids.contains(&mid) {
                    ignored_mids.push(mid.clone());
                    let m_label = ma_metric_defs
                        .get(&mid)
                        .map(|(nm, _)| {
                            if nm.is_empty() {
                                format!("metric_id={mid}")
                            } else {
                                nm.clone()
                            }
                        })
                        .unwrap_or_else(|| format!("metric_id={mid}"));
                    pre_diagnostics.push(AfIngestDiagnostic::NonPlddtQaMetricIgnored {
                        metric_name: m_label,
                    });
                }
            }
            if by_chain_seq.is_empty() {
                (None, None)
            } else {
                (Some(by_chain_seq), Some(by_seq))
            }
        }
    } else {
        (None, None)
    };

    let exptl_joined = if exptl_methods.is_empty() {
        None
    } else {
        Some(exptl_methods.join("; "))
    };

    finalize_af_structure_model(
        raw_residues,
        entry_id,
        detected_pdb_id,
        detected_uniprot_id,
        exptl_joined,
        has_computational_marker,
        ma_local_plddt,
        ma_local_by_seq,
        pre_diagnostics,
        opts,
    )
}

#[inline]
fn label_seq_id_or(opt: Option<i32>, fallback: i32) -> Option<i32> {
    Some(opt.unwrap_or(fallback))
}

// ---- PDB Parser ----

fn pdb_col_slice(bytes: &[u8], start: usize, end: usize) -> Option<&str> {
    let end_clamped = end.min(bytes.len());
    if start >= end_clamped {
        return None;
    }
    let sub = &bytes[start..end_clamped];
    if !sub.is_ascii() {
        return None;
    }
    std::str::from_utf8(sub).ok()
}

fn parse_pdb_af_model(text: &str, opts: &AfIngestOptions) -> Result<AfStructureModel, String> {
    let mut entry_id: Option<String> = None;
    let mut detected_pdb_id: Option<String> = None;
    let mut detected_uniprot_id: Option<String> = None;
    let mut exptl_parts: Vec<String> = Vec::new();
    let mut saw_cryst1_cell = false;

    let mut raw_residues: Vec<RawPolymerRes> = Vec::new();
    let mut res_idx_map: BTreeMap<(String, i32), usize> = BTreeMap::new();
    let mut selected_chain: Option<String> = opts
        .target_chain_id
        .as_ref()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    for (line_idx, line) in text.lines().enumerate() {
        let line_no = line_idx + 1;
        if line.starts_with("ENDMDL") && !raw_residues.is_empty() {
            break;
        }
        // Explicitly never read HELIX or SHEET records!
        if line.starts_with("HELIX ") || line.starts_with("SHEET ") {
            continue;
        }
        if line.starts_with("HEADER") {
            let bytes = line.as_bytes();
            if let Some(id_col) = pdb_col_slice(bytes, 62, 66) {
                let id_trim = id_col.trim().to_ascii_uppercase();
                if id_trim.len() == 4 && id_trim.bytes().all(|b| b.is_ascii_alphanumeric()) {
                    entry_id = Some(id_trim.clone());
                    detected_pdb_id = Some(id_trim);
                }
            }
            continue;
        }
        if let Some(rest) = line.strip_prefix("EXPDTA") {
            let m_trim = rest.trim();
            if !m_trim.is_empty() {
                exptl_parts.push(m_trim.to_string());
            }
            continue;
        }
        if line.starts_with("CRYST1") && line.len() >= 33 {
            if let (Ok(a), Ok(b), Ok(c)) = (
                line[6..15].trim().parse::<f64>(),
                line[15..24].trim().parse::<f64>(),
                line[24..33].trim().parse::<f64>(),
            ) {
                if a > 2.5 && b > 2.5 && c > 2.5 {
                    saw_cryst1_cell = true;
                }
            }
            continue;
        }
        if line.starts_with("TITLE ") || line.starts_with("REMARK") || line.starts_with("COMPND") {
            // Never set `has_computational_marker` from free-text PDB REMARK/TITLE/COMPND lines
            // (`21.W1:P0-2`: molecular-replacement crystal structures routinely mention AlphaFold!).
            continue;
        }
        if line.starts_with("DBREF ") {
            let toks: Vec<&str> = line.split_whitespace().collect();
            // e.g. `DBREF  1D3Z A    1    76  UNP    P62988   UBIQ_HUMAN`
            if let Some(unp_pos) = toks.iter().position(|t| t.eq_ignore_ascii_case("UNP")) {
                if let Some(&acc) = toks.get(unp_pos + 1) {
                    let up = acc.trim().to_ascii_uppercase();
                    if (6..=10).contains(&up.len()) && up.bytes().all(|b| b.is_ascii_alphanumeric())
                    {
                        detected_uniprot_id = Some(up);
                    }
                }
            }
            continue;
        }

        if !line.starts_with("ATOM  ") && !line.starts_with("HETATM") {
            continue;
        }
        let bytes = line.as_bytes();
        if bytes.len() < 54 {
            return Err(format!(
                "truncated PDB coordinate record on line {line_no} (length {} < 54)",
                bytes.len()
            ));
        }

        let alt = bytes[16];
        if alt != b' ' && alt != b'A' && alt != b'1' {
            continue;
        }
        if bytes[26] != b' ' {
            continue;
        }

        let ctx = format!("PDB line {line_no}");
        let atom_col =
            pdb_col_slice(bytes, 12, 16).ok_or_else(|| format!("{ctx}: non-ASCII atom column"))?;
        let res_col = pdb_col_slice(bytes, 17, 20)
            .ok_or_else(|| format!("{ctx}: non-ASCII residue column"))?;
        let res_norm = normalize_polymer_res_name(res_col);
        if line.starts_with("HETATM") && !is_standard_or_common_amino_acid(&res_norm) {
            continue;
        }

        let ch_byte = bytes[21];
        let chain_tok = if ch_byte == b' ' || !ch_byte.is_ascii_graphic() {
            "A".to_string()
        } else {
            (ch_byte as char).to_string()
        };
        if let Some(ref want_ch) = selected_chain {
            if &chain_tok != want_ch {
                continue;
            }
        } else {
            selected_chain = Some(chain_tok.clone());
        }

        let seq_col = pdb_col_slice(bytes, 22, 26)
            .ok_or_else(|| format!("{ctx}: non-ASCII seq_id column"))?;
        let seq_id: i32 = seq_col
            .trim()
            .parse()
            .map_err(|e| format!("{ctx}: invalid residue seq_id '{}': {e}", seq_col.trim()))?;

        let x_col =
            pdb_col_slice(bytes, 30, 38).ok_or_else(|| format!("{ctx}: invalid X column"))?;
        let y_col =
            pdb_col_slice(bytes, 38, 46).ok_or_else(|| format!("{ctx}: invalid Y column"))?;
        let z_col =
            pdb_col_slice(bytes, 46, 54).ok_or_else(|| format!("{ctx}: invalid Z column"))?;

        let x = parse_req_finite_f64(x_col, "X", &ctx)?;
        let y = parse_req_finite_f64(y_col, "Y", &ctx)?;
        let z = parse_req_finite_f64(z_col, "Z", &ctx)?;

        let b_iso = if let Some(b_col) = pdb_col_slice(bytes, 60, 66) {
            let b_trim = b_col.trim();
            if b_trim.is_empty() {
                None
            } else {
                Some(parse_req_finite_f64(b_trim, "tempFactor", &ctx)?)
            }
        } else {
            None
        };

        let atom_name = atom_col.trim().to_ascii_uppercase();
        let key = (chain_tok.clone(), seq_id);
        let r_slot = *res_idx_map.entry(key).or_insert_with(|| {
            let pos = raw_residues.len();
            raw_residues.push(RawPolymerRes::new(
                seq_id,
                Some(seq_id),
                chain_tok.clone(),
                res_norm.clone(),
            ));
            pos
        });
        raw_residues[r_slot].record_atom(&atom_name, [x, y, z], b_iso);
    }

    let exptl_joined = if !exptl_parts.is_empty() {
        Some(exptl_parts.join(" "))
    } else if saw_cryst1_cell {
        Some("X-RAY DIFFRACTION (CRYST1)".to_string())
    } else {
        None
    };

    finalize_af_structure_model(
        raw_residues,
        entry_id,
        detected_pdb_id,
        detected_uniprot_id,
        exptl_joined,
        false,
        None,
        None,
        Vec::new(),
        opts,
    )
}

// ============================================================================
// 2. Hostile-Input-Guarded PAE JSON N x N Parser (`<= 4,096` Pre-Allocation Cap)
// ============================================================================

/// Public `PAE` JSON schema dialect detected during ingestion.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PaeJsonDialect {
    /// `AFDB` v3/v4 dense `predicted_aligned_error` (`[[f32; N]; N]`).
    AfdbV3V4,
    /// `AFDB` legacy v1/v2 flat `residue1` / `residue2` / `distance` arrays.
    AfdbLegacySparse,
    /// `AlphaFold 2` / `ColabFold` (`{"pae": [[f32; N]; N], "max_pae": ..., "plddt": ...}`).
    ColabFoldAf2,
    /// `AlphaFold 3` `confidences.json` (`token_chain_ids` + `token_res_ids` + `pae`).
    AlphaFold3Confidences,
}

/// Validated square $N \times N$ Predicted Aligned Error (`PAE`, Å) matrix (`1 <= N <= 4,096`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PaeMatrix {
    pub n_residues: usize,
    pub values_angstrom: Vec<f32>,
    pub max_pae: f32,
    pub dialect: PaeJsonDialect,
    pub chain_id: Option<String>,
    pub plddt_from_json: Option<Vec<f64>>,
}

impl PaeMatrix {
    /// Raw directed `PAE(i, j)` in Angstroms (`0 <= i, j < n_residues`).
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> Option<f32> {
        if i < self.n_residues && j < self.n_residues {
            Some(self.values_angstrom[i * self.n_residues + j])
        } else {
            None
        }
    }

    /// Symmetrized $\overline{\text{PAE}}(i, j) = \frac{1}{2}(\text{PAE}(i, j) + \text{PAE}(j, i))$ in Angstroms.
    #[inline]
    pub fn symmetrized(&self, i: usize, j: usize) -> Option<f64> {
        let a = self.get(i, j)? as f64;
        let b = self.get(j, i)? as f64;
        Some(0.5 * (a + b))
    }

    /// Verify exact dimension match $N == L_{\text{chain}}$ against a polymer structure or shift sequence.
    pub fn validate_chain_length(&self, expected_len: usize) -> Result<(), String> {
        if self.n_residues != expected_len {
            return Err(format!(
                "PAE matrix dimension N = {} does not match polymer chain length L = {}",
                self.n_residues, expected_len
            ));
        }
        Ok(())
    }

    /// Cross-check PAE matrix dimension AND any embedded ColabFold/AF JSON `plddt` vector against
    /// `structure` (`21.W1:P2-9`).
    pub fn validate_against_structure(&self, structure: &AfStructureModel) -> Result<(), String> {
        let l = structure.residues.len();
        self.validate_chain_length(l)?;
        if let Some(ref json_plddt) = self.plddt_from_json {
            if json_plddt.len() != l {
                return Err(format!(
                    "PAE JSON plddt length ({}) does not match structure residue count L = {}",
                    json_plddt.len(),
                    l
                ));
            }
            if let Some(ref struct_plddt) = structure.plddt_per_residue {
                if struct_plddt.len() == l {
                    for i in 0..l {
                        let diff = (json_plddt[i] - struct_plddt[i]).abs();
                        if diff > 1.0 {
                            return Err(format!(
                                "ColabFold/PAE JSON plddt mismatch at residue index {} (seq_id {}): JSON plddt {:.2} vs structure pLDDT {:.2} (|diff| = {:.2} > 1.0)",
                                i, structure.residues[i].seq_id, json_plddt[i], struct_plddt[i], diff
                            ));
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

#[derive(Default)]
struct BoundedPae2d {
    n_cols: usize,
    flat: Vec<f32>,
    n_rows: usize,
}

impl<'de> Deserialize<'de> for BoundedPae2d {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct OuterVisitor;
        impl<'de> Visitor<'de> for OuterVisitor {
            type Value = BoundedPae2d;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "a square 2D PAE array with 1 <= N <= {MAX_PAE_RESIDUES}")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                if let Some(hint) = seq.size_hint() {
                    if hint > MAX_PAE_RESIDUES {
                        return Err(de::Error::custom(format!(
                            "PAE row count {hint} exceeds MAX_PAE_RESIDUES ({MAX_PAE_RESIDUES})"
                        )));
                    }
                }
                let mut n_cols: Option<usize> = None;
                let mut flat: Vec<f32> = Vec::new();
                let mut n_rows = 0usize;

                while let Some(BoundedPaeRow(row)) = seq.next_element::<BoundedPaeRow>()? {
                    n_rows += 1;
                    if n_rows > MAX_PAE_RESIDUES {
                        return Err(de::Error::custom(format!(
                            "PAE matrix dimension exceeds MAX_PAE_RESIDUES ({MAX_PAE_RESIDUES})"
                        )));
                    }
                    let rlen = row.len();
                    if rlen == 0 {
                        return Err(de::Error::custom("PAE matrix row cannot be empty"));
                    }
                    if let Some(expected_c) = n_cols {
                        if rlen != expected_c {
                            return Err(de::Error::custom(format!(
                                "ragged PAE row {}: length {} != expected {}",
                                n_rows - 1,
                                rlen,
                                expected_c
                            )));
                        }
                    } else {
                        n_cols = Some(rlen);
                        flat.reserve_exact(rlen.min(MAX_PAE_RESIDUES) * rlen.min(MAX_PAE_RESIDUES));
                    }
                    flat.extend_from_slice(&row);
                }

                let ncols =
                    n_cols.ok_or_else(|| de::Error::custom("PAE matrix is empty (N = 0)"))?;
                if n_rows != ncols {
                    return Err(de::Error::custom(format!(
                        "non-square PAE matrix: {n_rows} rows != {ncols} columns"
                    )));
                }
                Ok(BoundedPae2d {
                    n_cols: ncols,
                    flat,
                    n_rows,
                })
            }
        }
        deserializer.deserialize_seq(OuterVisitor)
    }
}

struct BoundedPaeRow(Vec<f32>);

impl<'de> Deserialize<'de> for BoundedPaeRow {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct RowVisitor;
        impl<'de> Visitor<'de> for RowVisitor {
            type Value = BoundedPaeRow;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(
                    f,
                    "a row of finite PAE floats in [0.0, 64.0] with length <= {MAX_PAE_RESIDUES}"
                )
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                if let Some(hint) = seq.size_hint() {
                    if hint > MAX_PAE_RESIDUES {
                        return Err(de::Error::custom(format!(
                            "PAE row length {hint} exceeds MAX_PAE_RESIDUES ({MAX_PAE_RESIDUES})"
                        )));
                    }
                }
                let mut out = Vec::new();
                while let Some(val) = seq.next_element::<f64>()? {
                    if out.len() >= MAX_PAE_RESIDUES {
                        return Err(de::Error::custom(format!(
                            "PAE row length exceeds MAX_PAE_RESIDUES ({MAX_PAE_RESIDUES})"
                        )));
                    }
                    if !val.is_finite() {
                        return Err(de::Error::custom("non-finite PAE value"));
                    }
                    if !(0.0..=(MAX_PAE_VALUE_ANGSTROM as f64)).contains(&val) {
                        return Err(de::Error::custom(format!(
                            "PAE value {val} outside physically valid [0.0, {MAX_PAE_VALUE_ANGSTROM}] A"
                        )));
                    }
                    out.push(val as f32);
                }
                Ok(BoundedPaeRow(out))
            }
        }
        deserializer.deserialize_seq(RowVisitor)
    }
}

struct BoundedSparseResVec(Vec<usize>);

impl<'de> Deserialize<'de> for BoundedSparseResVec {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = BoundedSparseResVec;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(
                    f,
                    "1-based residue index array bounded by {MAX_PAE_RESIDUES}"
                )
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let max_cells = MAX_PAE_RESIDUES * MAX_PAE_RESIDUES;
                let mut out = Vec::new();
                while let Some(idx_1based) = seq.next_element::<i64>()? {
                    if out.len() >= max_cells {
                        return Err(de::Error::custom(format!(
                            "sparse PAE array exceeds MAX_PAE_RESIDUES^2 ({max_cells})"
                        )));
                    }
                    if idx_1based < 1 || (idx_1based as usize) > MAX_PAE_RESIDUES {
                        return Err(de::Error::custom(format!(
                            "sparse PAE residue index {idx_1based} outside 1..={MAX_PAE_RESIDUES}"
                        )));
                    }
                    out.push(idx_1based as usize);
                }
                Ok(BoundedSparseResVec(out))
            }
        }
        deserializer.deserialize_seq(V)
    }
}

struct BoundedSparseDistVec(Vec<f32>);

impl<'de> Deserialize<'de> for BoundedSparseDistVec {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = BoundedSparseDistVec;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "finite PAE distance array in [0.0, 64.0]")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let max_cells = MAX_PAE_RESIDUES * MAX_PAE_RESIDUES;
                let mut out = Vec::new();
                while let Some(v) = seq.next_element::<f64>()? {
                    if out.len() >= max_cells {
                        return Err(de::Error::custom(format!(
                            "sparse PAE distance array exceeds MAX_PAE_RESIDUES^2 ({max_cells})"
                        )));
                    }
                    if !v.is_finite() || !(0.0..=(MAX_PAE_VALUE_ANGSTROM as f64)).contains(&v) {
                        return Err(de::Error::custom(format!(
                            "sparse PAE distance {v} outside [0.0, {MAX_PAE_VALUE_ANGSTROM}] A"
                        )));
                    }
                    out.push(v as f32);
                }
                Ok(BoundedSparseDistVec(out))
            }
        }
        deserializer.deserialize_seq(V)
    }
}

#[derive(Deserialize)]
struct RawPaeObject {
    #[serde(default)]
    predicted_aligned_error: Option<BoundedPae2d>,
    #[serde(default)]
    max_predicted_aligned_error: Option<f64>,
    #[serde(default)]
    pae: Option<BoundedPae2d>,
    #[serde(default)]
    max_pae: Option<f64>,
    #[serde(default)]
    plddt: Option<Vec<f64>>,
    #[serde(default)]
    token_chain_ids: Option<Vec<String>>,
    #[serde(default)]
    token_res_ids: Option<Vec<i32>>,
    #[serde(default)]
    residue1: Option<BoundedSparseResVec>,
    #[serde(default)]
    residue2: Option<BoundedSparseResVec>,
    #[serde(default)]
    distance: Option<BoundedSparseDistVec>,
}

struct FirstArrayPaeObject(RawPaeObject);

impl<'de> Deserialize<'de> for FirstArrayPaeObject {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = FirstArrayPaeObject;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "non-empty JSON array of PAE objects")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let first: RawPaeObject = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::custom("PAE JSON top-level array is empty"))?;
                while seq.next_element::<de::IgnoredAny>()?.is_some() {}
                Ok(FirstArrayPaeObject(first))
            }
        }
        deserializer.deserialize_seq(V)
    }
}

/// Read and validate a `PAE` JSON file from disk (`<= 256 MiB` raw cap, `1 <= N <= 4,096`).
pub fn parse_pae_json_file(
    path: impl AsRef<Path>,
    target_chain_id: Option<&str>,
    expected_chain_len: Option<usize>,
) -> Result<PaeMatrix, String> {
    let p = path.as_ref();
    let meta =
        fs::metadata(p).map_err(|e| format!("failed to stat PAE JSON '{}': {e}", p.display()))?;
    if meta.len() > MAX_PAE_JSON_BYTES as u64 {
        return Err(format!(
            "PAE JSON '{}' is {} bytes, exceeding safety limit ({} bytes)",
            p.display(),
            meta.len(),
            MAX_PAE_JSON_BYTES
        ));
    }
    let text = fs::read_to_string(p)
        .map_err(|e| format!("failed to read PAE JSON '{}': {e}", p.display()))?;
    parse_pae_json_str_for_chain(&text, target_chain_id, expected_chain_len)
}

/// Parse any supported public `PAE` JSON dialect (`AFDB` v3/v4, legacy `AFDB` v1/v2 sparse,
/// `ColabFold`/`AF2`, or `AlphaFold 3` `confidences.json`) into a validated [`PaeMatrix`].
pub fn parse_pae_json_str(json_str: &str) -> Result<PaeMatrix, String> {
    parse_pae_json_str_for_chain(json_str, None, None)
}

/// Parse `PAE` JSON with optional target polymer chain selection (`AF3` `token_chain_ids`) and
/// optional exact chain length check (`N == L_chain`).
pub fn parse_pae_json_str_for_chain(
    json_str: &str,
    target_chain_id: Option<&str>,
    expected_chain_len: Option<usize>,
) -> Result<PaeMatrix, String> {
    if json_str.len() > MAX_PAE_JSON_BYTES {
        return Err(format!(
            "PAE JSON input ({} bytes) exceeds MAX_PAE_JSON_BYTES ({MAX_PAE_JSON_BYTES})",
            json_str.len()
        ));
    }
    let trimmed = json_str.trim();
    if trimmed.is_empty() {
        return Err("PAE JSON input is empty".to_string());
    }

    let obj: RawPaeObject = if trimmed.starts_with('[') {
        serde_json::from_str::<FirstArrayPaeObject>(trimmed)
            .map_err(|e| format!("invalid PAE JSON: {e}"))?
            .0
    } else if trimmed.starts_with('{') {
        serde_json::from_str::<RawPaeObject>(trimmed)
            .map_err(|e| format!("invalid PAE JSON: {e}"))?
    } else {
        return Err("invalid PAE JSON: expected top-level '[' or '{'".to_string());
    };

    let mut matrix = if let Some(dense) = obj.predicted_aligned_error {
        debug_assert_eq!(dense.n_rows, dense.n_cols);
        let max_pae = resolve_max_pae(obj.max_predicted_aligned_error, &dense.flat)?;
        PaeMatrix {
            n_residues: dense.n_rows,
            values_angstrom: dense.flat,
            max_pae,
            dialect: PaeJsonDialect::AfdbV3V4,
            chain_id: None,
            plddt_from_json: validate_opt_plddt_json(obj.plddt, dense.n_rows)?,
        }
    } else if let Some(dense) = obj.pae {
        if obj.token_chain_ids.is_some() || obj.token_res_ids.is_some() {
            aggregate_af3_token_pae(
                dense,
                obj.token_chain_ids.as_deref(),
                obj.token_res_ids.as_deref(),
                obj.max_pae,
                target_chain_id,
            )?
        } else {
            let max_pae = resolve_max_pae(obj.max_pae, &dense.flat)?;
            PaeMatrix {
                n_residues: dense.n_rows,
                values_angstrom: dense.flat,
                max_pae,
                dialect: PaeJsonDialect::ColabFoldAf2,
                chain_id: None,
                plddt_from_json: validate_opt_plddt_json(obj.plddt, dense.n_rows)?,
            }
        }
    } else if let (
        Some(BoundedSparseResVec(r1)),
        Some(BoundedSparseResVec(r2)),
        Some(BoundedSparseDistVec(d)),
    ) = (obj.residue1, obj.residue2, obj.distance)
    {
        if r1.is_empty() || r1.len() != r2.len() || r1.len() != d.len() {
            return Err(format!(
                "malformed legacy AFDB sparse PAE arrays: residue1={}, residue2={}, distance={}",
                r1.len(),
                r2.len(),
                d.len()
            ));
        }
        let n = r1.iter().chain(r2.iter()).copied().max().unwrap_or(0);
        if n == 0 || n > MAX_PAE_RESIDUES {
            return Err(format!(
                "sparse PAE dimension N = {n} outside 1..={MAX_PAE_RESIDUES}"
            ));
        }
        if r1.len() != n * n {
            return Err(format!(
                "incomplete legacy sparse PAE matrix: expected {} entries for N = {n}, got {}",
                n * n,
                r1.len()
            ));
        }
        let mut flat = vec![f32::NAN; n * n];
        for ((&i1, &j1), &val) in r1.iter().zip(r2.iter()).zip(d.iter()) {
            flat[(i1 - 1) * n + (j1 - 1)] = val;
        }
        if flat.iter().any(|v| !v.is_finite()) {
            return Err("legacy sparse PAE matrix is missing one or more (i, j) cells".to_string());
        }
        let max_pae = resolve_max_pae(obj.max_predicted_aligned_error.or(obj.max_pae), &flat)?;
        PaeMatrix {
            n_residues: n,
            values_angstrom: flat,
            max_pae,
            dialect: PaeJsonDialect::AfdbLegacySparse,
            chain_id: None,
            plddt_from_json: None,
        }
    } else {
        return Err(
            "PAE JSON missing 'predicted_aligned_error', 'pae', or ('residue1','residue2','distance')"
                .to_string(),
        );
    };

    if let Some(exp_len) = expected_chain_len {
        matrix.validate_chain_length(exp_len)?;
    }
    if matrix.chain_id.is_none() {
        matrix.chain_id = target_chain_id.map(ToString::to_string);
    }
    Ok(matrix)
}

fn resolve_max_pae(explicit: Option<f64>, flat: &[f32]) -> Result<f32, String> {
    let obs_max = flat.iter().copied().fold(0.0_f32, f32::max);
    if let Some(m) = explicit {
        if !m.is_finite() || m <= 0.0 || m > (MAX_PAE_VALUE_ANGSTROM as f64) {
            return Err(format!("invalid max_pae value {m}"));
        }
        Ok((m as f32).max(obs_max))
    } else {
        Ok(obs_max.max(31.75))
    }
}

fn validate_opt_plddt_json(
    raw: Option<Vec<f64>>,
    expected_n: usize,
) -> Result<Option<Vec<f64>>, String> {
    let Some(mut vals) = raw else {
        return Ok(None);
    };
    if vals.len() != expected_n {
        return Err(format!(
            "PAE JSON plddt length ({}) != PAE dimension ({expected_n})",
            vals.len()
        ));
    }
    let mut dummy_diag = Vec::new();
    vals = normalize_plddt_vector(vals, &mut dummy_diag, "PAE JSON plddt")?;
    Ok(Some(vals))
}

fn aggregate_af3_token_pae(
    dense: BoundedPae2d,
    token_chains: Option<&[String]>,
    token_res_ids: Option<&[i32]>,
    explicit_max_pae: Option<f64>,
    target_chain_id: Option<&str>,
) -> Result<PaeMatrix, String> {
    let n_tok = dense.n_rows;
    let chains =
        token_chains.ok_or_else(|| "AF3 confidences.json missing token_chain_ids".to_string())?;
    let res_ids =
        token_res_ids.ok_or_else(|| "AF3 confidences.json missing token_res_ids".to_string())?;
    if chains.len() != n_tok || res_ids.len() != n_tok {
        return Err(format!(
            "AF3 token metadata mismatch: pae={n_tok}, token_chain_ids={}, token_res_ids={}",
            chains.len(),
            res_ids.len()
        ));
    }

    let chosen_chain = match target_chain_id {
        Some(tc) if !tc.trim().is_empty() => tc.trim().to_string(),
        _ => {
            let mut distinct_chains: Vec<String> = Vec::new();
            for ch in chains {
                let c = ch.trim();
                if !c.is_empty() && !distinct_chains.iter().any(|x| x == c) {
                    distinct_chains.push(c.to_string());
                }
            }
            if distinct_chains.len() > 1 {
                return Err(format!(
                    "AF3 confidences.json contains multiple chains ({}) — explicit --chain <id> selection is required (21.W1:P2-9)",
                    distinct_chains.join(", ")
                ));
            }
            distinct_chains
                .into_iter()
                .next()
                .ok_or_else(|| "AF3 token_chain_ids is empty".to_string())?
        }
    };

    // Group tokens belonging to `chosen_chain` by unique `token_res_id` in chain order
    let mut res_groups: Vec<(i32, Vec<usize>)> = Vec::new();
    let mut res_pos_map: BTreeMap<i32, usize> = BTreeMap::new();
    for t in 0..n_tok {
        if chains[t].trim() != chosen_chain {
            continue;
        }
        let rid = res_ids[t];
        let g_idx = *res_pos_map.entry(rid).or_insert_with(|| {
            let pos = res_groups.len();
            res_groups.push((rid, Vec::new()));
            pos
        });
        res_groups[g_idx].1.push(t);
    }

    if res_groups.is_empty() {
        return Err(format!(
            "AF3 confidences.json contains zero tokens for target chain '{chosen_chain}'"
        ));
    }
    let n_res = res_groups.len();
    if n_res > MAX_PAE_RESIDUES {
        return Err(format!(
            "AF3 aggregated chain length {n_res} exceeds MAX_PAE_RESIDUES ({MAX_PAE_RESIDUES})"
        ));
    }

    let mut out = vec![0.0_f32; n_res * n_res];
    for a in 0..n_res {
        let toks_a = &res_groups[a].1;
        for b in 0..n_res {
            let toks_b = &res_groups[b].1;
            let mut sum = 0.0_f64;
            for &ta in toks_a {
                for &tb in toks_b {
                    sum += dense.flat[ta * n_tok + tb] as f64;
                }
            }
            let mean = sum / ((toks_a.len() * toks_b.len()) as f64);
            out[a * n_res + b] = mean as f32;
        }
    }
    let max_pae = resolve_max_pae(explicit_max_pae, &out)?;
    Ok(PaeMatrix {
        n_residues: n_res,
        values_angstrom: out,
        max_pae,
        dialect: PaeJsonDialect::AlphaFold3Confidences,
        chain_id: Some(chosen_chain),
        plddt_from_json: None,
    })
}

// ============================================================================
// 3. Fowler-Sljoka-Williamson Rigidity Concordance & 4-Class Discordance Classifier
// ============================================================================

/// Experimental per-residue Lipari–Szabo generalized order parameter $S^2_i \in [0, 1]$ row
/// parsed from `--s2 <modelfree.tsv>`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExperimentalS2Entry {
    pub seq_id: i32,
    pub res_name: Option<String>,
    pub s2: f64,
    pub sigma_s2: Option<f64>,
}

/// Source of the experimental rigidity profile $S^2_{\text{exp},i}$.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExperimentalRigiditySource {
    /// Lipari–Szabo $S^2$ order parameters supplied via `--s2 <modelfree.tsv>` (preferred; independent of secondary shifts).
    LipariSzaboModelFree,
    /// Shift-derived, secondary-structure-coupled 6-nucleus Berjanskii–Wishart `RCI-S²` from backbone shifts.
    BerjanskiiWishartRciS2,
}

impl ExperimentalRigiditySource {
    #[inline]
    pub const fn provenance_label(self) -> &'static str {
        match self {
            Self::LipariSzaboModelFree => "experimental Lipari-Szabo S^2",
            Self::BerjanskiiWishartRciS2 => "shift-derived secondary-structure-coupled RCI-S^2",
        }
    }
}

/// Original FERRIC `pLDDT`-vs-$S^2$ rigidity concordance summary (`21.W1:P1-1`, `A5`).
///
/// Note: `c_corr_score` and `c_rmsd_score` are retained on `Evaluated` strictly as uncalibrated
/// display transformations (`50*(1+rho_s)` and `100*exp(-(rmsd/0.22)^2)`) for backwards struct
/// compatibility; downstream quantitative analyses should inspect [`AfConcordanceReport::spearman_rho`],
/// [`AfConcordanceReport::spearman_rho_ci95`], [`AfConcordanceReport::ordered_core_spearman_rho`],
/// and [`AfConcordanceReport::ordered_core_spearman_rho_ci95`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ConcordanceSummary {
    /// Evaluated across $N_{\text{eval}} \ge 10$ assigned non-terminal residues with non-zero variance in both `pLDDT` and $S^2_{\text{exp}}$.
    Evaluated {
        n_eval: usize,
        spearman_rho_s: f64,
        /// Uncalibrated display transformation `50 * (1 + spearman_rho_s)` (`[0, 100]`).
        c_corr_score: f64,
        rmsd_rigidity: f64,
        /// Uncalibrated display transformation `100 * exp(-(rmsd_rigidity / 0.22)^2)` (`[0, 100]`).
        c_rmsd_score: f64,
    },
    /// Fewer than `10` assigned non-terminal residues have both `pLDDT` and $S^2_{\text{exp}}$,
    /// OR `pLDDT` / $S^2_{\text{exp}}$ has zero variance so Spearman $\rho_s$ is undefined.
    InsufficientCoverage { n_eval: usize, reason: String },
}

impl ConcordanceSummary {
    /// Return whole-chain Spearman $\rho_s$, or `None` when coverage is insufficient or variance is zero.
    #[inline]
    pub fn spearman_rho(&self) -> Option<f64> {
        match self {
            Self::Evaluated { spearman_rho_s, .. } if spearman_rho_s.is_finite() => {
                Some(*spearman_rho_s)
            }
            _ => None,
        }
    }
}

/// Per-residue AlphaFold vs Solution NMR structural concordance / discordance class (`T21.1` §1.3.3, `21.W1:P1-1`, `A5`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AfNmrConcordanceClass {
    /// `pLDDT >= 70`, $S^2_{\text{exp}} \ge 0.65$, no `Helix <-> Sheet` inversion, and $\Delta\theta < 60^\circ$ on `Good` torsions.
    ConcordantOrdered,
    /// `pLDDT < 70` and $S^2_{\text{exp}} < 0.65$ (flexible loop, tail, linker, or IDP in both AF and solution).
    ConcordantDisordered,
    /// `pLDDT >= 70` with 3D `Helix` or `Sheet`, while solution NMR reports disorder ($S^2_{\text{exp}} < 0.60$ and $p_{\text{coil}} + p_{\text{PPII}} \ge 0.65$).
    /// (Replaces legacy `ConditionalFoldingOrCrystalArtifact` — AF models have no crystal; `21.W1:P1-1`, `A5`.)
    #[serde(alias = "ConditionalFoldingOrCrystalArtifact")]
    ModelOrderedSolutionDisordered,
    /// Ordered conformational switch ($\Delta\theta \ge 60^\circ$ or `Helix <-> Sheet` flip) or rigid-in-NMR / low-`pLDDT` hinge ($S^2_{\text{exp}} \ge 0.72$, `pLDDT < 70`).
    DynamicHingeOrAltState,
    /// Borderline / ambiguous regime where thresholds do not cleanly separate ordered vs disordered
    /// (e.g. `pLDDT >= 70` with `0.60 <= S^2 < 0.65` and structured NMR, `pLDDT < 70` with `0.65 <= S^2 < 0.72`,
    /// or `pLDDT >= 70` with `S^2 < 0.60` and `p_coil + p_PPII < 0.65`).
    Indeterminate,
    /// Fewer than 2 assigned backbone chemical shifts and no experimental $S^2_i$ (or missing `pLDDT`).
    UnassignedNmr,
}

impl AfNmrConcordanceClass {
    /// Backwards-compatible associated constant alias for [`Self::ModelOrderedSolutionDisordered`].
    #[allow(non_upper_case_globals)]
    pub const ConditionalFoldingOrCrystalArtifact: Self = Self::ModelOrderedSolutionDisordered;

    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ConcordantOrdered => "ConcordantOrdered",
            Self::ConcordantDisordered => "ConcordantDisordered",
            Self::ModelOrderedSolutionDisordered => "ModelOrderedSolutionDisordered",
            Self::DynamicHingeOrAltState => "DynamicHingeOrAltState",
            Self::Indeterminate => "Indeterminate",
            Self::UnassignedNmr => "UnassignedNmr",
        }
    }

    /// Color-vision-safe hex color specified in `docs/specs/21-alphafold-integrative-nmr.md` §3.1.
    #[inline]
    pub const fn hex_color(self) -> &'static str {
        match self {
            Self::ConcordantOrdered => "#16A34A",
            Self::ConcordantDisordered => "#64748B",
            Self::ModelOrderedSolutionDisordered => "#D97706",
            Self::DynamicHingeOrAltState => "#DC2626",
            Self::Indeterminate => "#94A3B8",
            Self::UnassignedNmr => "#CBD5E1",
        }
    }
}

/// Configuration for `evaluate_af_nmr_concordance`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AfConcordanceConfig {
    pub temperature_kelvin: f64,
    pub sample_ph: Option<f64>,
    pub exclude_pdb_ids: Vec<String>,
    pub exclude_bmrb_ids: Vec<String>,
    pub exclude_uniprot_id: Option<String>,
    pub exclude_topology_id: Option<String>,
    /// Sequence-identity cutoff for `T12.3` `HomologFirewall` (defaults to `0.25`, never `> 0.25`).
    pub firewall_max_seq_identity: f64,
}

impl Default for AfConcordanceConfig {
    fn default() -> Self {
        Self {
            temperature_kelvin: 298.15,
            sample_ph: None,
            exclude_pdb_ids: Vec::new(),
            exclude_bmrb_ids: Vec::new(),
            exclude_uniprot_id: None,
            exclude_topology_id: None,
            firewall_max_seq_identity: DEFAULT_FIREWALL_MAX_SEQ_IDENTITY,
        }
    }
}

/// Per-residue AlphaFold vs Solution NMR concordance record.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResidueConcordanceRecord {
    pub struct_index: usize,
    pub seq_id: i32,
    pub res_name: String,
    pub plddt: Option<f64>,
    pub s2_af_hat: Option<f64>,
    pub s2_exp: Option<f64>,
    pub n_backbone_shifts: usize,
    pub state8_af: KabschSanderState8,
    pub state4_af: SecondaryStructure4,
    pub ssp_nmr: Option<f64>,
    pub p_alpha_nmr: f64,
    pub p_beta_nmr: f64,
    pub p_ppii_nmr: f64,
    pub p_coil_nmr: f64,
    pub phi_af_deg: Option<f64>,
    pub psi_af_deg: Option<f64>,
    pub phi_nmr_deg: Option<f64>,
    pub psi_nmr_deg: Option<f64>,
    pub torsion_class_nmr: TorsionPredictionClass,
    pub delta_theta_deg: Option<f64>,
    pub concordance_class: AfNmrConcordanceClass,
}

/// Complete protein-wide AlphaFold vs Solution NMR validation and concordance report.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AfConcordanceReport {
    pub entry_id: Option<String>,
    pub chain_id: String,
    pub provenance: CoordinateProvenance,
    pub plddt_source: PlddtSource,
    pub rigidity_source: ExperimentalRigiditySource,
    pub ingest_diagnostics: Vec<AfIngestDiagnostic>,
    pub summary: ConcordanceSummary,
    /// Whole-chain Spearman $\rho_s$ across non-terminal assigned residues (`None` if `< 10` residues or zero variance; `21.W1:P1-1`).
    pub spearman_rho: Option<f64>,
    /// Moving-block bootstrap (`block_len = 5`, `n_boot = 500`) 95% CI `[lo, hi]` for whole-chain Spearman $\rho_s$ (`21.W1:P1-1`, `A5`).
    pub spearman_rho_ci95: Option<[f64; 2]>,
    /// Count of non-terminal assigned residues in the ordered core (`pLDDT >= 70`).
    pub n_ordered_core: usize,
    /// Ordered-core (`pLDDT >= 70`) Spearman $\rho_s$ (`None` if `< 10` core residues or zero variance; `21.W1:P1-1`, `A5`).
    pub ordered_core_spearman_rho: Option<f64>,
    /// Moving-block bootstrap (`block_len = 5`, `n_boot = 500`) 95% CI `[lo, hi]` for ordered-core Spearman $\rho_s$.
    pub ordered_core_spearman_rho_ci95: Option<[f64; 2]>,
    /// Effective rigidity RMSD between $\hat{S}^2_{\text{AF}}$ and $S^2_{\text{exp}}$ (`None` if `< 10` residues).
    pub rmsd_rigidity: Option<f64>,
    pub secondary_structure: SecondaryStructureReport,
    pub propensity: PropensityReport,
    pub residues: Vec<ResidueConcordanceRecord>,
    pub n_concordant_ordered: usize,
    pub n_concordant_disordered: usize,
    pub n_conditional_folding: usize,
    pub n_dynamic_hinge_or_alt: usize,
    pub n_indeterminate: usize,
    pub n_unassigned_nmr: usize,
    pub pae_n_residues: Option<usize>,
    pub pae_median_intra_ordered: Option<f64>,
    pub exclusion_list_sha256: String,
    pub max_retained_seq_identity: f64,
    pub excluded_template_ids: Vec<String>,
}

/// Map per-residue $\text{pLDDT}_i \in [0, 100]$ onto effective solution rigidity
/// $\hat{S}^2_{\text{AF},i} = \operatorname{clamp}((\text{pLDDT}_i - 35.0) / 55.0,\; 0.0,\; 1.0)$.
#[inline]
pub fn plddt_to_effective_s2(plddt: f64) -> f64 {
    ((plddt - 35.0) / 55.0).clamp(0.0, 1.0)
}

/// Parse experimental Lipari–Szabo $S^2$ values from a `modelfree.tsv` file (`ferric modelfree`
/// output or `seq_id\ts2` / `seq_id\tres_name\ts2` TSV).
pub fn parse_modelfree_s2_tsv_file(
    path: impl AsRef<Path>,
) -> Result<Vec<ExperimentalS2Entry>, String> {
    let p = path.as_ref();
    let txt = fs::read_to_string(p)
        .map_err(|e| format!("failed to read modelfree S2 TSV '{}': {e}", p.display()))?;
    parse_modelfree_s2_tsv_str(&txt)
}

/// Parse experimental Lipari–Szabo $S^2$ values from TSV text, rejecting `NaN`/`Inf` and out-of-range entries.
pub fn parse_modelfree_s2_tsv_str(tsv_text: &str) -> Result<Vec<ExperimentalS2Entry>, String> {
    let mut c_seq: Option<usize> = None;
    let mut c_res: Option<usize> = None;
    let mut c_s2: Option<usize> = None;
    let mut c_sig: Option<usize> = None;
    let mut header_seen = false;
    let mut out = Vec::new();

    for (idx, raw_line) in tsv_text.lines().enumerate() {
        let line_no = idx + 1;
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').map(str::trim).collect();
        if !header_seen && cols.first().is_some_and(|c| c.parse::<i32>().is_err()) {
            header_seen = true;
            for (ci, &col) in cols.iter().enumerate() {
                match col.to_ascii_lowercase().as_str() {
                    "seq_id" | "resid" | "residue_id" | "seq" => c_seq = Some(ci),
                    "res_name" | "residue" | "aa" => c_res = Some(ci),
                    "s2" | "s2_exp" | "order_param" => c_s2 = Some(ci),
                    "sigma_s2" | "s2_err" | "err_s2" => c_sig = Some(ci),
                    _ => {}
                }
            }
            if c_seq.is_none() || c_s2.is_none() {
                return Err(format!(
                    "modelfree TSV header on line {line_no} missing required 'seq_id' and 's2' columns"
                ));
            }
            continue;
        }

        let (seq_idx, res_idx, s2_idx, sig_idx) = if header_seen {
            (c_seq.unwrap_or(0), c_res, c_s2.unwrap_or(1), c_sig)
        } else if cols.len() >= 4
            && cols[3].parse::<f64>().is_ok()
            && cols[1].parse::<f64>().is_err()
        {
            // `seq_id res_name model s2 ...`
            (0, Some(1), 3, (cols.len() > 4).then_some(4))
        } else if cols.len() >= 3 && cols[1].parse::<f64>().is_err() {
            (0, Some(1), 2, (cols.len() > 3).then_some(3))
        } else if cols.len() >= 2 {
            (0, None, 1, (cols.len() > 2).then_some(2))
        } else {
            return Err(format!("malformed modelfree S2 row on line {line_no}"));
        };

        let s_tok = cols
            .get(seq_idx)
            .ok_or_else(|| format!("line {line_no}: missing seq_id column"))?;
        let seq_id: i32 = s_tok
            .parse()
            .map_err(|e| format!("line {line_no}: invalid seq_id '{s_tok}': {e}"))?;
        let s2_tok = cols
            .get(s2_idx)
            .ok_or_else(|| format!("line {line_no}: missing s2 column"))?;
        if *s2_tok == "." || *s2_tok == "?" || s2_tok.eq_ignore_ascii_case("NA") {
            continue;
        }
        let s2_raw: f64 = s2_tok
            .parse()
            .map_err(|e| format!("line {line_no}: invalid s2 '{s2_tok}': {e}"))?;
        if !s2_raw.is_finite() || !(-0.02..=1.05).contains(&s2_raw) {
            return Err(format!(
                "line {line_no}: s2 value {s2_raw} outside valid [0.0, 1.0] bounds"
            ));
        }
        let sigma_s2 = match sig_idx.and_then(|si| cols.get(si)) {
            Some(&st) if !st.is_empty() && st != "." && st != "?" => {
                let sv: f64 = st
                    .parse()
                    .map_err(|e| format!("line {line_no}: invalid sigma_s2 '{st}': {e}"))?;
                if !sv.is_finite() || sv < 0.0 {
                    return Err(format!(
                        "line {line_no}: invalid non-finite/negative sigma_s2"
                    ));
                }
                Some(sv)
            }
            _ => None,
        };
        let res_name = res_idx
            .and_then(|ri| cols.get(ri))
            .map(|s| s.to_ascii_uppercase())
            .filter(|s| !s.is_empty() && s != ".");

        out.push(ExperimentalS2Entry {
            seq_id,
            res_name,
            s2: s2_raw.clamp(0.0, 1.0),
            sigma_s2,
        });
    }

    if out.is_empty() {
        return Err("modelfree S2 TSV contains zero valid residue rows".to_string());
    }
    Ok(out)
}

/// Compute 6-nucleus unlabelled-offset-corrected Berjanskii & Wishart (*J. Am. Chem. Soc.* **2005**,
/// 127, 14970–14971) Random Coil Index (`RCI`) and order parameter $S^2_{\text{RCI}} \in [0, 1]$
/// from `ncIDP` secondary shifts `[dH, dN, dCA, dCB, dC', dHA]`.
///
/// Uses exact Berjanskii–Wishart (2005) end-cap `0.2` (`RCI = 0.2 / (1 + 3.25 * avg_z)` when
/// scaling to `[0, 0.2]` — note `S^2_RCI = 1 - 0.4 * ln(1 + 17.7 * RCI)`, eq. 3 of JACS 2005,
/// 127, 14970–14971) and **gap-aware** smoothing: unassigned residues / Prolines lacking any
/// assigned shift are marked `None` prior to smoothing so a default flexible `RCI` is NEVER
/// smeared into assigned neighbors (`21.W1:P2-7`, `P2-10`).
pub fn compute_berjanskii_wishart_rci_s2(sec_shifts: &[[Option<f64>; 6]]) -> (Vec<f64>, Vec<f64>) {
    // Balanced 6-nucleus `[H, N, CA, CB, C', HA]` reference scales and Berjanskii-Wishart (2005, Table 1)
    // weights giving equal sensitivity to alpha-helical (CA, C', HA) and beta-strand (HA, CB, H, C') shifts.
    const SCALES: [f64; 6] = [0.42, 2.65, 1.65, 1.65, 1.55, 0.34];
    const WEIGHTS: [f64; 6] = [1.5, 0.8, 2.4, 2.0, 2.1, 3.0];
    const RCI_UNASSIGNED_FALLBACK: f64 = 0.22;

    let n = sec_shifts.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    let mut raw_rci: Vec<Option<f64>> = vec![None; n];
    for (i, slot) in raw_rci.iter_mut().enumerate() {
        let has_any_shift_at_i = sec_shifts[i].iter().any(Option::is_some);
        if !has_any_shift_at_i {
            continue;
        }
        let mut num = 0.0_f64;
        let mut den = 0.0_f64;
        for k in 0..6 {
            let mut sm_sum = 0.0_f64;
            let mut sm_w = 0.0_f64;
            for (d, wt) in [(-1isize, 0.25_f64), (0, 0.50), (1, 0.25)] {
                let j = i as isize + d;
                if j >= 0 && (j as usize) < n {
                    if let Some(v) = sec_shifts[j as usize][k] {
                        sm_sum += wt * v.abs();
                        sm_w += wt;
                    }
                }
            }
            if sm_w > 0.0 {
                let sm = sm_sum / sm_w;
                num += WEIGHTS[k] * (sm / SCALES[k]);
                den += WEIGHTS[k];
            }
        }
        if den > 0.0 {
            let avg_z = num / den;
            *slot = Some(RCI_UNASSIGNED_FALLBACK / (1.0 + 3.25 * avg_z));
        }
    }

    // 5-residue triangular smoothing `[1, 2, 3, 2, 1]` on interior residues (Berjanskii & Wishart 2005),
    // skipping unassigned/gap residues (`raw_rci[j] == None`) so default flexible RCI never leaks
    // across gaps or Prolines (`21.W1:P2-7`), and using the exact Berjanskii–Wishart (2005)
    // `S^2_RCI = 1 - 0.4 * ln(1 + 17.7 * RCI)` constant (`21.W1:P2-10`).
    let mut rci_out = Vec::with_capacity(n);
    let mut s2_out = Vec::with_capacity(n);
    for i in 0..n {
        if raw_rci[i].is_none() {
            rci_out.push(RCI_UNASSIGNED_FALLBACK);
            s2_out.push((1.0 - 0.4 * det_ln(1.0 + 17.7 * RCI_UNASSIGNED_FALLBACK)).clamp(0.0, 1.0));
            continue;
        }
        let near_term = i < 3 || i + 3 >= n;
        let (s, w) = if near_term {
            let mut acc = 0.0_f64;
            let mut wt_sum = 0.0_f64;
            for (d, wt) in [(-1isize, 0.25_f64), (0, 0.50), (1, 0.25)] {
                let j = i as isize + d;
                if j >= 0 && (j as usize) < n {
                    if let Some(rv) = raw_rci[j as usize] {
                        acc += wt * rv;
                        wt_sum += wt;
                    }
                }
            }
            (acc, wt_sum)
        } else {
            let mut acc = 0.0_f64;
            let mut wt_sum = 0.0_f64;
            for (d, wt) in [(-2isize, 1.0_f64), (-1, 2.0), (0, 3.0), (1, 2.0), (2, 1.0)] {
                let j = (i as isize + d) as usize;
                if let Some(rv) = raw_rci[j] {
                    acc += wt * rv;
                    wt_sum += wt;
                }
            }
            (acc, wt_sum)
        };
        let rci_v = s / w.max(1.0e-9);
        let s2_v = (1.0 - 0.4 * det_ln(1.0 + 17.7 * rci_v)).clamp(0.0, 1.0);
        rci_out.push(rci_v);
        s2_out.push(s2_v);
    }
    (rci_out, s2_out)
}

/// Tie-adjusted fractional-rank Spearman correlation $\rho_s \in [-1, +1]$.
/// Returns `None` when `n < 2` or either variable has zero variance (constant `pLDDT` or `S^2`, `21.W1:P1-1`).
pub fn tie_adjusted_spearman_rho(xs: &[f64], ys: &[f64]) -> Option<f64> {
    let n = xs.len().min(ys.len());
    if n < 2 {
        return None;
    }
    let rx = average_fractional_ranks(&xs[..n]);
    let ry = average_fractional_ranks(&ys[..n]);
    let mean_x = rx.iter().sum::<f64>() / (n as f64);
    let mean_y = ry.iter().sum::<f64>() / (n as f64);
    let mut cov = 0.0_f64;
    let mut vx = 0.0_f64;
    let mut vy = 0.0_f64;
    for i in 0..n {
        let dx = rx[i] - mean_x;
        let dy = ry[i] - mean_y;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    let denom = (vx * vy).sqrt();
    if denom < 1.0e-12 {
        None
    } else {
        Some((cov / denom).clamp(-1.0, 1.0))
    }
}

/// Deterministic moving-block bootstrap (`block_len = 5`, `n_boot = 500`) 95% confidence interval
/// `[q025, q975]` for Spearman $\rho_s$ (`21.W1:P1-1`, `A5`).
pub fn block_bootstrap_spearman_rho_ci95(
    xs: &[f64],
    ys: &[f64],
    block_len: usize,
    n_boot: usize,
) -> Option<[f64; 2]> {
    let n = xs.len().min(ys.len());
    if n < 4 || n_boot < 20 {
        return None;
    }
    // Return None if full sample itself has zero variance
    tie_adjusted_spearman_rho(xs, ys)?;

    let b_len = block_len.max(1).min(n);
    let n_starts = n - b_len + 1;
    let n_blocks = n.div_ceil(b_len);

    let mut state: u64 = 0x9E37_79B9_7F4A_7C15_u64 ^ ((n as u64) << 17);
    let mut next_u64 = || -> u64 {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        state.wrapping_mul(0x2545_F491_4F6C_DD1D_u64)
    };

    let mut boot_rhos = Vec::with_capacity(n_boot);
    let mut bx = Vec::with_capacity(n);
    let mut by = Vec::with_capacity(n);
    for _ in 0..n_boot {
        bx.clear();
        by.clear();
        for _ in 0..n_blocks {
            let start = (next_u64() as usize) % n_starts;
            let take = (n - bx.len()).min(b_len);
            bx.extend_from_slice(&xs[start..(start + take)]);
            by.extend_from_slice(&ys[start..(start + take)]);
        }
        if let Some(r) = tie_adjusted_spearman_rho(&bx, &by) {
            boot_rhos.push(r);
        }
    }
    if boot_rhos.len() < 10 {
        return None;
    }
    boot_rhos.sort_by(f64::total_cmp);
    let m = boot_rhos.len();
    let lo_idx = ((0.025 * ((m - 1) as f64)).floor() as usize).min(m - 1);
    let hi_idx = ((0.975 * ((m - 1) as f64)).ceil() as usize).min(m - 1);
    Some([boot_rhos[lo_idx], boot_rhos[hi_idx]])
}

fn average_fractional_ranks(vals: &[f64]) -> Vec<f64> {
    let n = vals.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| vals[a].total_cmp(&vals[b]).then_with(|| a.cmp(&b)));
    let mut ranks = vec![0.0_f64; n];
    let mut i = 0usize;
    while i < n {
        let mut j = i + 1;
        while j < n && (vals[order[j]] - vals[order[i]]).abs() <= 1.0e-12 {
            j += 1;
        }
        // Average of 1-based ranks `(i + 1) ..= j` is `(i + 1 + j) / 2`
        let avg_rank = ((i + 1 + j) as f64) * 0.5;
        for &orig_idx in &order[i..j] {
            ranks[orig_idx] = avg_rank;
        }
        i = j;
    }
    ranks
}

fn align_structure_to_nmr(
    structure: &AfStructureModel,
    shifts: &Reference,
) -> Result<Vec<Option<usize>>, String> {
    let l_struct = structure.residues.len();
    let l_nmr = shifts.sequence.len();
    if l_nmr == 0 {
        return Err("NMR shift table has empty sequence".to_string());
    }

    // 1. Direct (seq_id, res_name) match check
    let mut nmr_by_seq: BTreeMap<i32, usize> = BTreeMap::new();
    for (k, &sid) in shifts.seq_ids.iter().enumerate() {
        nmr_by_seq.insert(sid, k);
    }
    let mut direct_map = vec![None; l_struct];
    let mut direct_exact = 0usize;
    let mut direct_overlap = 0usize;
    for (i, sr) in structure.residues.iter().enumerate() {
        if let Some(&nk) = nmr_by_seq.get(&sr.seq_id) {
            direct_overlap += 1;
            if shifts.sequence[nk].eq_ignore_ascii_case(&sr.res_name) {
                direct_exact += 1;
                direct_map[i] = Some(nk);
            }
        }
    }
    let min_l = l_struct.min(l_nmr).max(1);
    if direct_overlap > 0 && direct_exact * 5 >= min_l * 4 {
        return Ok(direct_map);
    }

    // 2. Fallback to ungapped sequence-offset alignment (handling cloning tags or N-Met cleavage)
    let shift_min = -((l_nmr as isize) - 1);
    let shift_max = (l_struct as isize) - 1;
    let mut best_matches = 0usize;
    let mut best_shift = 0isize;
    for sh in shift_min..=shift_max {
        let mut m_cnt = 0usize;
        for (ni, n_res) in shifts.sequence.iter().enumerate() {
            let si_isize = (ni as isize) + sh;
            if si_isize >= 0 && (si_isize as usize) < l_struct {
                let si = si_isize as usize;
                if structure.residues[si].res_name.eq_ignore_ascii_case(n_res) {
                    m_cnt += 1;
                }
            }
        }
        if m_cnt > best_matches || (m_cnt == best_matches && m_cnt > 0 && sh == 0) {
            best_matches = m_cnt;
            best_shift = sh;
        }
    }

    if best_matches * 2 < min_l {
        return Err(format!(
            "structure sequence (L = {l_struct}) does not match NMR shift sequence (L = {l_nmr}): only {best_matches} identical aligned residues"
        ));
    }

    let mut map = vec![None; l_struct];
    for ni in 0..l_nmr {
        let si_isize = (ni as isize) + best_shift;
        if si_isize >= 0 && (si_isize as usize) < l_struct {
            map[si_isize as usize] = Some(ni);
        }
    }
    Ok(map)
}

/// Evaluate FERRIC's original `pLDDT`-vs-$S^2$ rigidity concordance heuristic (whole-chain and
/// ordered-core `pLDDT >= 70` Spearman $\rho_s$ with moving-block bootstrap `block_len = 5` 95% CIs,
/// plus `RMSD_rig`) and 6-class gap-free per-residue structural concordance/discordance
/// (`AfNmrConcordanceClass`), enforcing `<= 25%` sequence-identity & `TM < 0.50` `HomologFirewall`
/// on `T12.3` $(\hat{\phi}, \hat{\psi})$ predictions (`21.W1:P1-1`, `P2-9`, `A5`).
pub fn evaluate_af_nmr_concordance(
    structure: &AfStructureModel,
    shifts: &Reference,
    pae: Option<&PaeMatrix>,
    s2_override: Option<&[ExperimentalS2Entry]>,
    cfg: &AfConcordanceConfig,
) -> Result<AfConcordanceReport, String> {
    let l_struct = structure.residues.len();
    if l_struct == 0 {
        return Err("structure contains zero residues".to_string());
    }
    if let Some(pm) = pae {
        pm.validate_against_structure(structure)?;
    }

    let struct_to_nmr = align_structure_to_nmr(structure, shifts)?;

    // 1. Clean-room Kabsch-Sander 8 -> 4 secondary-structure classification on 3D coordinates
    let ss_report = classify_kabsch_sander(&structure.backbone_coords_for_ss())?;

    // 2. Unlabelled 3-state `(Coil, Helix, Sheet)` EM global referencing offset check
    //    (Wang & Wishart 2005) paired with T11.3 `ncIDP` neighbor/pH/chain-end baseline so folded
    //    beta-rich proteins never have beta-strand secondary shifts subtracted as coil offsets!
    let em_offs = estimate_global_shift_offsets(shifts);
    let prop_cfg = PropensityConfig {
        temperature_kelvin: cfg.temperature_kelvin,
        sample_ph: cfg.sample_ph,
        auto_acidic_ph_detection: true,
        auto_offset_correction: false,
        manual_offsets: PropensityOffsets {
            d_ca: em_offs.dca,
            d_cb: em_offs.dcb,
            d_c: em_offs.dc,
            d_ha: em_offs.dha,
            d_h: em_offs.dh,
            d_n: em_offs.dn,
        },
        ..PropensityConfig::default()
    };
    let propensity = compute_idp_propensity(shifts, &prop_cfg);

    // 3. 6-nucleus unlabelled-offset-corrected Berjanskii-Wishart RCI-S^2 from ncIDP secondary shifts
    let mut ncidp_sec_shifts = vec![[None; 6]; shifts.sequence.len()];
    for (k, rp) in propensity.residues.iter().enumerate() {
        ncidp_sec_shifts[k] = [
            rp.secondary_shifts.get("H").copied(),
            rp.secondary_shifts.get("N").copied(),
            rp.secondary_shifts.get("CA").copied(),
            rp.secondary_shifts.get("CB").copied(),
            rp.secondary_shifts.get("C").copied(),
            rp.secondary_shifts.get("HA").copied(),
        ];
    }
    let (_, rci_s2_nmr) = compute_berjanskii_wishart_rci_s2(&ncidp_sec_shifts);

    // 4. Homolog-firewalled (`<= 25%` seq-id + `TM-score < 0.50`) T12.3 torsion prediction
    let mut fw_pdbs = cfg.exclude_pdb_ids.clone();
    if let Some(ref pid) = structure.detected_pdb_id {
        if !fw_pdbs.iter().any(|p| p.eq_ignore_ascii_case(pid)) {
            fw_pdbs.push(pid.clone());
        }
    }
    let fw_unp = cfg
        .exclude_uniprot_id
        .clone()
        .or_else(|| structure.detected_uniprot_id.clone());
    let fw_query = HomologFirewallQuery {
        query_pdb_ids: fw_pdbs,
        query_bmrb_ids: cfg.exclude_bmrb_ids.clone(),
        query_uniprot_id: fw_unp,
        query_family_id: None,
        query_topology_id: cfg.exclude_topology_id.clone(),
        query_ca_coords: Some(structure.ca_coords()),
    };
    let max_sid = cfg
        .firewall_max_seq_identity
        .clamp(0.0, DEFAULT_FIREWALL_MAX_SEQ_IDENTITY);
    let torsion_rep = HomologFirewall::new(fw_query)
        .with_max_seq_identity(max_sid)
        .predict(shifts);

    // Optional experimental Lipari-Szabo S^2 map by seq_id
    let s2_exp_map: Option<BTreeMap<i32, f64>> = s2_override.map(|rows| {
        rows.iter()
            .map(|e| (e.seq_id, e.s2.clamp(0.0, 1.0)))
            .collect()
    });
    let rigidity_source = if s2_exp_map.is_some() {
        ExperimentalRigiditySource::LipariSzaboModelFree
    } else {
        ExperimentalRigiditySource::BerjanskiiWishartRciS2
    };

    let mut eval_plddt = Vec::new();
    let mut eval_s2_af = Vec::new();
    let mut eval_s2_exp = Vec::new();
    let mut core_plddt = Vec::new();
    let mut core_s2_exp = Vec::new();

    let mut records = Vec::with_capacity(l_struct);
    let mut n_concordant_ordered = 0usize;
    let mut n_concordant_disordered = 0usize;
    let mut n_conditional_folding = 0usize;
    let mut n_dynamic_hinge_or_alt = 0usize;
    let mut n_indeterminate = 0usize;
    let mut n_unassigned_nmr = 0usize;

    for (i, &nmr_idx_opt) in struct_to_nmr.iter().enumerate().take(l_struct) {
        let sr = &structure.residues[i];
        let ss_res = &ss_report.residues[i];
        let plddt = sr.plddt;
        let s2_af_hat = plddt.map(plddt_to_effective_s2);
        let (
            n_bb_shifts,
            ssp_nmr,
            p_alpha,
            p_beta,
            p_ppii,
            p_coil,
            phi_nmr,
            psi_nmr,
            tor_class,
            tor_ss,
            rci_s2_val,
            nmr_seq_id,
        ) = if let Some(nk) = nmr_idx_opt {
            let rp = &propensity.residues[nk];
            let tp = &torsion_rep.residues[nk];
            (
                rp.n_observed_nuclei,
                rp.ssp_score,
                rp.p_alpha,
                rp.p_beta,
                rp.p_ppii,
                rp.p_coil,
                tp.phi_deg,
                tp.psi_deg,
                tp.class,
                tp.ss_class,
                Some(rci_s2_nmr[nk]),
                shifts.seq_ids.get(nk).copied().unwrap_or(sr.seq_id),
            )
        } else {
            (
                0usize,
                None,
                0.05,
                0.05,
                0.25,
                0.65,
                None,
                None,
                TorsionPredictionClass::Unassigned,
                SecondaryStructure::Coil,
                None,
                sr.seq_id,
            )
        };

        let explicit_s2 = s2_exp_map
            .as_ref()
            .and_then(|m| m.get(&sr.seq_id).or_else(|| m.get(&nmr_seq_id)).copied());

        let has_nmr_assignment = explicit_s2.is_some() || n_bb_shifts >= 2;
        let s2_exp = if !has_nmr_assignment {
            None
        } else if let Some(s2_ls) = explicit_s2 {
            Some(s2_ls)
        } else {
            rci_s2_val
        };

        let delta_theta_deg = match (ss_res.phi_deg, ss_res.psi_deg, phi_nmr, psi_nmr) {
            (Some(pa), Some(sa), Some(pn), Some(sn))
                if tor_class == TorsionPredictionClass::Good =>
            {
                let dp = periodic_angle_abs_diff_deg(pa, pn);
                let ds = periodic_angle_abs_diff_deg(sa, sn);
                Some((dp * dp + ds * ds).sqrt())
            }
            _ => None,
        };

        // Accumulate non-terminal assigned residues (`0 < i < L - 1`) for global and ordered-core concordance
        if i > 0 && i + 1 < l_struct {
            if let (Some(pl), Some(sa), Some(se)) = (plddt, s2_af_hat, s2_exp) {
                eval_plddt.push(pl);
                eval_s2_af.push(sa);
                eval_s2_exp.push(se);
                if pl >= 70.0 {
                    core_plddt.push(pl);
                    core_s2_exp.push(se);
                }
            }
        }

        // Classify per-residue concordance / discordance with zero threshold gaps (`21.W1:P1-1`, `A5`):
        // - `pLDDT >= 70`:
        //   * `S^2 < 0.60` & AF Helix/Sheet & `p_coil + p_ppii >= 0.65` => `ModelOrderedSolutionDisordered`
        //   * `S^2 < 0.60` otherwise => `Indeterminate` (no clear solution disorder vs dynamic split)
        //   * `0.60 <= S^2 < 0.65` => `Indeterminate` (borderline rigidity gap)
        //   * `S^2 >= 0.65` & (`ss_inversion` || `delta_theta >= 60`) => `DynamicHingeOrAltState`
        //   * `S^2 >= 0.65` otherwise => `ConcordantOrdered`
        // - `pLDDT < 70`:
        //   * `S^2 >= 0.72` => `DynamicHingeOrAltState`
        //   * `0.65 <= S^2 < 0.72` => `Indeterminate` (borderline rigidity gap)
        //   * `S^2 < 0.65` => `ConcordantDisordered`
        let concordance_class = match (plddt, s2_exp) {
            (None, _) | (_, None) => AfNmrConcordanceClass::UnassignedNmr,
            (Some(pl), Some(se)) => {
                let af_is_helix_or_sheet = matches!(
                    ss_res.state4,
                    SecondaryStructure4::Helix | SecondaryStructure4::Sheet
                );
                // Detect clear Helix <-> Sheet secondary-structure inversion between 3D AF and solution NMR
                let nmr_clearly_sheet = ssp_nmr.is_some_and(|s| s <= -0.25)
                    || (p_beta >= 0.42 && p_beta > p_alpha + 0.15)
                    || (tor_class == TorsionPredictionClass::Good
                        && tor_ss == SecondaryStructure::Sheet);
                let nmr_clearly_helix = ssp_nmr.is_some_and(|s| s >= 0.25)
                    || (p_alpha >= 0.42 && p_alpha > p_beta + 0.15)
                    || (tor_class == TorsionPredictionClass::Good
                        && tor_ss == SecondaryStructure::Helix);
                let ss_inversion = (ss_res.state4 == SecondaryStructure4::Helix
                    && nmr_clearly_sheet)
                    || (ss_res.state8 == KabschSanderState8::E && nmr_clearly_helix);

                if pl >= 70.0 {
                    if se < 0.60 {
                        if af_is_helix_or_sheet && (p_coil + p_ppii) >= 0.65 {
                            AfNmrConcordanceClass::ModelOrderedSolutionDisordered
                        } else {
                            AfNmrConcordanceClass::Indeterminate
                        }
                    } else if se < 0.65 {
                        AfNmrConcordanceClass::Indeterminate
                    } else if ss_inversion || delta_theta_deg.is_some_and(|dt| dt >= 60.0) {
                        AfNmrConcordanceClass::DynamicHingeOrAltState
                    } else {
                        AfNmrConcordanceClass::ConcordantOrdered
                    }
                } else if se >= 0.72 {
                    AfNmrConcordanceClass::DynamicHingeOrAltState
                } else if se >= 0.65 {
                    AfNmrConcordanceClass::Indeterminate
                } else {
                    AfNmrConcordanceClass::ConcordantDisordered
                }
            }
        };

        match concordance_class {
            AfNmrConcordanceClass::ConcordantOrdered => n_concordant_ordered += 1,
            AfNmrConcordanceClass::ConcordantDisordered => n_concordant_disordered += 1,
            AfNmrConcordanceClass::ModelOrderedSolutionDisordered => {
                n_conditional_folding += 1;
            }
            AfNmrConcordanceClass::DynamicHingeOrAltState => n_dynamic_hinge_or_alt += 1,
            AfNmrConcordanceClass::Indeterminate => n_indeterminate += 1,
            AfNmrConcordanceClass::UnassignedNmr => n_unassigned_nmr += 1,
        }

        records.push(ResidueConcordanceRecord {
            struct_index: i,
            seq_id: sr.seq_id,
            res_name: sr.res_name.clone(),
            plddt,
            s2_af_hat,
            s2_exp,
            n_backbone_shifts: n_bb_shifts,
            state8_af: ss_res.state8,
            state4_af: ss_res.state4,
            ssp_nmr,
            p_alpha_nmr: p_alpha,
            p_beta_nmr: p_beta,
            p_ppii_nmr: p_ppii,
            p_coil_nmr: p_coil,
            phi_af_deg: ss_res.phi_deg,
            psi_af_deg: ss_res.psi_deg,
            phi_nmr_deg: phi_nmr,
            psi_nmr_deg: psi_nmr,
            torsion_class_nmr: tor_class,
            delta_theta_deg,
            concordance_class,
        });
    }

    let n_eval = eval_plddt.len();
    let n_ordered_core = core_plddt.len();
    let (
        summary,
        spearman_rho,
        spearman_rho_ci95,
        ordered_core_spearman_rho,
        ordered_core_spearman_rho_ci95,
        rmsd_rigidity_opt,
    ) = if n_eval < MIN_CONCORDANCE_EVAL_RESIDUES {
        let reason = if structure.plddt_per_residue.is_none() {
            "structure lacks per-residue pLDDT (experimental coordinate B-factor guard active)"
                .to_string()
        } else {
            format!(
                "only {n_eval} assigned non-terminal residues (< {MIN_CONCORDANCE_EVAL_RESIDUES} required)"
            )
        };
        (
            ConcordanceSummary::InsufficientCoverage { n_eval, reason },
            None,
            None,
            None,
            None,
            None,
        )
    } else {
        let rho_s = tie_adjusted_spearman_rho(&eval_plddt, &eval_s2_exp);
        let rho_ci = block_bootstrap_spearman_rho_ci95(&eval_plddt, &eval_s2_exp, 5, 500);
        let (core_rho_s, core_rho_ci) = if n_ordered_core >= MIN_CONCORDANCE_EVAL_RESIDUES {
            (
                tie_adjusted_spearman_rho(&core_plddt, &core_s2_exp),
                block_bootstrap_spearman_rho_ci95(&core_plddt, &core_s2_exp, 5, 500),
            )
        } else {
            (None, None)
        };
        let mse: f64 = eval_s2_af
            .iter()
            .zip(eval_s2_exp.iter())
            .map(|(&sa, &se)| {
                let d = sa - se;
                d * d
            })
            .sum::<f64>()
            / (n_eval as f64);
        let rmsd_rig = mse.sqrt();
        let ratio = rmsd_rig / 0.22;
        let c_rmsd = (100.0 * det_exp(-(ratio * ratio))).clamp(0.0, 100.0);
        let sum_val = match rho_s {
            Some(r) => ConcordanceSummary::Evaluated {
                n_eval,
                spearman_rho_s: r,
                c_corr_score: (50.0 * (1.0 + r)).clamp(0.0, 100.0),
                rmsd_rigidity: rmsd_rig,
                c_rmsd_score: c_rmsd,
            },
            None => ConcordanceSummary::InsufficientCoverage {
                n_eval,
                reason:
                    "zero variance in pLDDT or experimental rigidity; Spearman rho_s is undefined"
                        .to_string(),
            },
        };
        (
            sum_val,
            rho_s,
            rho_ci,
            core_rho_s,
            core_rho_ci,
            Some(rmsd_rig),
        )
    };

    // Optional PAE summary across ordered residues (`pLDDT >= 70`)
    let pae_median_intra_ordered = pae.and_then(|pm| {
        let ordered: Vec<usize> = structure
            .residues
            .iter()
            .enumerate()
            .filter_map(|(idx, r)| r.plddt.filter(|&p| p >= 70.0).map(|_| idx))
            .collect();
        if ordered.len() < 2 {
            return None;
        }
        let mut vals = Vec::with_capacity(ordered.len() * (ordered.len() - 1) / 2);
        for a in 0..ordered.len() {
            for b in (a + 1)..ordered.len() {
                if let Some(sp) = pm.symmetrized(ordered[a], ordered[b]) {
                    vals.push(sp);
                }
            }
        }
        (!vals.is_empty()).then(|| median_of_slice(&vals))
    });

    Ok(AfConcordanceReport {
        entry_id: structure.entry_id.clone(),
        chain_id: structure.chain_id.clone(),
        provenance: structure.provenance,
        plddt_source: structure.plddt_source,
        rigidity_source,
        ingest_diagnostics: structure.diagnostics.clone(),
        summary,
        spearman_rho,
        spearman_rho_ci95,
        n_ordered_core,
        ordered_core_spearman_rho,
        ordered_core_spearman_rho_ci95,
        rmsd_rigidity: rmsd_rigidity_opt,
        secondary_structure: ss_report,
        propensity,
        residues: records,
        n_concordant_ordered,
        n_concordant_disordered,
        n_conditional_folding,
        n_dynamic_hinge_or_alt,
        n_indeterminate,
        n_unassigned_nmr,
        pae_n_residues: pae.map(|pm| pm.n_residues),
        pae_median_intra_ordered,
        exclusion_list_sha256: torsion_rep.exclusion_list_sha256,
        max_retained_seq_identity: torsion_rep.max_retained_seq_identity,
        excluded_template_ids: torsion_rep.excluded_template_ids,
    })
}

/// Format an [`AfConcordanceReport`] as clean deterministic TSV.
pub fn format_af_concordance_tsv(rep: &AfConcordanceReport) -> String {
    let mut out = String::new();
    let fmt_opt_f64 = |v: Option<f64>, prec: usize| -> String {
        match v {
            Some(x) => format!("{x:.prec$}"),
            None => "undefined".to_string(),
        }
    };
    let fmt_ci = |ci: Option<[f64; 2]>| -> String {
        match ci {
            Some([lo, hi]) => format!("[{lo:.4},{hi:.4}]"),
            None => "undefined".to_string(),
        }
    };
    match &rep.summary {
        ConcordanceSummary::Evaluated {
            n_eval,
            spearman_rho_s,
            c_corr_score,
            rmsd_rigidity,
            c_rmsd_score,
        } => {
            out.push_str(&format!(
                "# ferric_af_concordance chain={} provenance={:?} plddt_source={:?} rigidity_source=\"{}\" n_eval={} spearman_rho_s={:.4} spearman_rho_ci95={} n_ordered_core={} ordered_core_rho_s={} ordered_core_ci95={} rmsd_rig={:.4} uncal_c_corr={:.2} uncal_c_rmsd={:.2} indeterminate={} firewall_sha256={} max_retained_seq_id={:.4}\n",
                rep.chain_id,
                rep.provenance,
                rep.plddt_source,
                rep.rigidity_source.provenance_label(),
                n_eval,
                spearman_rho_s,
                fmt_ci(rep.spearman_rho_ci95),
                rep.n_ordered_core,
                fmt_opt_f64(rep.ordered_core_spearman_rho, 4),
                fmt_ci(rep.ordered_core_spearman_rho_ci95),
                rmsd_rigidity,
                c_corr_score,
                c_rmsd_score,
                rep.n_indeterminate,
                rep.exclusion_list_sha256,
                rep.max_retained_seq_identity
            ));
        }
        ConcordanceSummary::InsufficientCoverage { n_eval, reason } => {
            out.push_str(&format!(
                "# ferric_af_concordance chain={} provenance={:?} status=InsufficientCoverage n_eval={} spearman_rho_s={} rmsd_rig={} reason=\"{}\" firewall_sha256={}\n",
                rep.chain_id,
                rep.provenance,
                n_eval,
                fmt_opt_f64(rep.spearman_rho, 4),
                fmt_opt_f64(rep.rmsd_rigidity, 4),
                reason,
                rep.exclusion_list_sha256
            ));
        }
    }
    out.push_str("seq_id\tres_name\tplddt\ts2_af_hat\ts2_exp\tstate8_af\tstate4_af\tssp_nmr\tp_alpha\tp_beta\tp_ppii\tp_coil\tdelta_theta_deg\tconcordance_class\n");
    let fmt_opt = |v: Option<f64>, prec: usize| -> String {
        match v {
            Some(x) => format!("{x:.prec$}"),
            None => ".".to_string(),
        }
    };
    for r in &rep.residues {
        out.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{}\t{}\n",
            r.seq_id,
            r.res_name,
            fmt_opt(r.plddt, 2),
            fmt_opt(r.s2_af_hat, 4),
            fmt_opt(r.s2_exp, 4),
            r.state8_af.as_str(),
            r.state4_af.as_str(),
            fmt_opt(r.ssp_nmr, 3),
            r.p_alpha_nmr,
            r.p_beta_nmr,
            r.p_ppii_nmr,
            r.p_coil_nmr,
            fmt_opt(r.delta_theta_deg, 1),
            r.concordance_class.as_str()
        ));
    }
    out
}

/// Format an [`AfConcordanceReport`] as pretty-printed JSON.
pub fn format_af_concordance_json(rep: &AfConcordanceReport) -> Result<String, String> {
    serde_json::to_string_pretty(rep)
        .map_err(|e| format!("failed to serialize concordance JSON: {e}"))
}
