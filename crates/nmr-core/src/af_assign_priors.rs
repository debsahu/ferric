//! Opt-in 3D-structure-prior-assisted protein backbone chemical-shift scoring (`U_{s,k}`) and
//! `PAE`-gated `3D 15N-edited NOESY-HSQC` amide-amide contact connectivity (`T21.4`,
//! `docs/specs/21-alphafold-integrative-nmr.md` section 4).
//!
//! # Provenance & Clean-Room Statement
//! Original clean-room implementation derived strictly from peer-reviewed statistical and
//! biomolecular NMR literature, consuming 3D shift predictions from `nmr-core::struct_shifts`
//! (`T21.2`, fitted on-the-fly exclusively to FERRIC's `CC0` `torsion_corpus.tsv` behind the
//! 4-layer `max_seq_identity <= 0.25`, `TM-score < 0.50` `HomologFirewall`), with zero external
//! parameter tables, neural weights, or restricted software artifacts:
//! - Lange, K. L., Little, R. J. A. & Taylor, J. M. G. (*J. Am. Stat. Assoc.* **1989**, 84,
//!   881-896) -- robust statistical modeling with heavy-tailed Student-`t(nu = 4)` likelihoods
//!   `ell_{t4}(z) = -(5/2) ln(1 + z^2 / 4)`.
//! - Wuthrich, K. (*NMR of Proteins and Nucleic Acids*, Wiley, **1986**) & Grzesiek, S.,
//!   Wingfield, P., Stahl, S., Kaufman, J. D. & Bax, A. (*J. Biomol. NMR* **1995**, 5, 225-233) --
//!   `3D 15N-edited NOESY-HSQC` `d_{NN}(i, j) <= 5.5 A` amide-proton sequential and cross-strand
//!   connectivity in perdeuterated and sparse-carbon backbone assignment.
//! - Jumper, J. et al. (*Nature* **2021**, 596, 583-589) & Varadi, M. et al. (*Nucleic Acids Res.*
//!   **2022**, 50, D439-D444; **2024**, 52, D368-D375) -- per-residue `pLDDT` confidence weighting
//!   and symmetrized Predicted Aligned Error (`PAE <= 6.0 A`) inter-domain contact gating.
//! - Shen, Y. & Bax, A. (*J. Biomol. NMR* **2010**, 48, 13-22) -- 3D backbone chemical-shift
//!   physics implemented in `nmr-core::struct_shifts`.
//! - Huber, P. J. (*Ann. Math. Stat.* **1964**, 35, 73-101) -- Huber M-estimation iteratively
//!   reweighted least squares (`IRLS`) for outlier-resistant location registration.

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::af_concordance::{parse_pae_json_file, parse_pae_json_str_for_chain, PaeMatrix};
use crate::dihedrals::align_by_res_names;
use crate::nef::read_nef_str;
use crate::noesy::v_dist;
use crate::peaks::{read_peaks_str, Peak};
use crate::priors::{
    delta_covalent, expected_covalent_atom_offset, one_to_three, Priors, ShiftOffsets, ATOMS,
};
use crate::solver::{Matrix2D, FORBID};
use crate::spinsystems::SpinSystem;
use crate::struct_shifts::{
    af_plddt_confidence_weight, predict_structure_shifts_with_firewall, ResidueShiftPrediction,
    StructCoordinateProvenance, StructShiftConfig, StructShiftPredictionReport, StructureModel,
    MAX_STRUCTURE_FILE_BYTES,
};
use crate::torsion_db::{HomologFirewallQuery, DEFAULT_FIREWALL_MAX_SEQ_IDENTITY};

/// Default scaling factor `lambda_AF = 0.65` applied to clamped Student-`t(nu = 4)` 3D shift LLRs.
pub const DEFAULT_AF_WEIGHT: f64 = 0.65;

/// Default symmetric per-residue Student-`t(nu = 4)` LLR clamp ceiling `U_max^{clamp} = 4.0` nats.
pub const DEFAULT_AF_CLAMP_NATS: f64 = 4.0;

/// Degrees of freedom `nu = 4` for the heavy-tailed Student-`t` 3D shift likelihood.
pub const STUDENT_T4_DOF: f64 = 4.0;

/// Normalization constant `3 / 8 = 0.375` of the standard Student-`t(nu = 4)` density
/// `f_{t4}(z) = (3 / 8) * (1 + z^2 / 4)^(-5/2)`.
pub const STUDENT_T4_NORM_CONST: f64 = 0.375;

/// Maximum amide-proton distance `d(H^N_j, H^N_k) <= 5.5 A` for admitting a pair into `C_AF`.
pub const AF_NOESY_MAX_DIST_ANGSTROM: f64 = 5.5;

/// Maximum sequential `(k - 1, k)` `d_{NN}` cutoff (`4.5 A`) for sequential NOE corroboration.
pub const AF_NOESY_SEQ_MAX_DIST_ANGSTROM: f64 = 4.5;

/// Maximum medium/long-range `|j - k| >= 2` `d_{NN}` cutoff (`5.0 A`) for cross-strand NOE factors.
pub const AF_NOESY_LR_MAX_DIST_ANGSTROM: f64 = 5.0;

/// Minimum endpoint `pLDDT >= 70.0` required on both residues `(j, k)` to admit a `15N-NOESY` contact.
pub const AF_NOESY_MIN_PLDDT: f64 = 70.0;

/// Maximum symmetrized `0.5 * (PAE(j, k) + PAE(k, j)) <= 6.0 A` required when `PAE` is supplied.
pub const AF_NOESY_MAX_SYMMETRIZED_PAE_ANGSTROM: f64 = 6.0;

/// Maximum local sequence separation `|j - k| <= 5` admitted when no `PAE` matrix is supplied.
pub const AF_NOESY_LOCAL_SEQ_SEP_WITHOUT_PAE: usize = 5;

/// Hard upper bound (`+2.50` nats) on sequential `(k - 1, k)` `15N-NOESY` log-odds bonus.
pub const AF_NOESY_MAX_SEQ_BONUS_NATS: f64 = 2.50;

/// Hard upper bound (`+2.20` nats) on medium/long-range `|j - k| >= 2` `15N-NOESY` pairwise bonus.
pub const AF_NOESY_MAX_LR_BONUS_NATS: f64 = 2.20;

/// Exact integer carrier-center ppm constants `[H, N, CA, CB, C]` subtracted in `f64` on the host
/// before `f32` WebGPU buffer upload so `scoring.wgsl` achieves `max |U_cpu - U_gpu| < 1.0e-5`
/// even near `176 ppm` carbonyl `13C'`.
pub const AF_GPU_CARRIER_CENTERS_PPM: [f64; 5] = [8.0, 120.0, 56.0, 44.0, 176.0];

/// Unnormalized heavy-tailed Student-`t(nu = 4)` log-kernel `ell_{t4}(z) = -(5/2) ln(1 + z^2 / 4)`
/// (`docs/specs/21-alphafold-integrative-nmr.md` section 4.2; Lange, Little & Taylor 1989).
#[inline]
pub fn ell_student_t4(z: f64) -> f64 {
    -2.5 * (1.0 + 0.25 * z * z).ln()
}

/// Configuration for opt-in 3D-prior-assisted backbone and `PAE`-gated `15N-NOESY` assignment.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AfAssignConfig {
    /// Global weight `lambda_AF` scaling the clamped per-residue 3D shift LLR (`default = 0.65`).
    pub af_weight: f64,
    /// Symmetric per-residue Student-`t4` LLR clamp ceiling `U_max^{clamp}` in nats (`default = 4.0`).
    pub af_clamp_nats: f64,
    /// Symmetrized `PAE` cutoff in Angstroms for admitting `3D 15N-NOESY` contacts (`default = 6.0`).
    pub pae_cutoff_angstrom: f64,
    /// Minimum endpoint `pLDDT` required on both `(j, k)` residues for `15N-NOESY` contacts (`default = 70.0`).
    pub min_contact_plddt: f64,
    /// `(tol_h_ppm, tol_n_ppm)` strip attachment tolerance for `3D 15N-NOESY` crosspeaks (`default = (0.04, 0.40)`).
    pub noesy_strip_tol_hn: (f64, f64),
    /// Gaussian matching width `sigma_h_noe` (ppm) on the indirect `1H` NOE dimension (`default = 0.025`).
    pub noesy_indirect_sigma_h_ppm: f64,
    /// Hard indirect `1H` matching cutoff (ppm) on `3D 15N-NOESY` crosspeaks (`default = 0.065`).
    pub noesy_indirect_cutoff_h_ppm: f64,
    /// Underlying `T21.2` forward 3D shift predictor configuration (`max_seq_identity <= 0.25`).
    pub struct_shift_cfg: StructShiftConfig,
}

impl Default for AfAssignConfig {
    fn default() -> Self {
        Self {
            af_weight: DEFAULT_AF_WEIGHT,
            af_clamp_nats: DEFAULT_AF_CLAMP_NATS,
            pae_cutoff_angstrom: AF_NOESY_MAX_SYMMETRIZED_PAE_ANGSTROM,
            min_contact_plddt: AF_NOESY_MIN_PLDDT,
            noesy_strip_tol_hn: (0.04, 0.40),
            noesy_indirect_sigma_h_ppm: 0.025,
            noesy_indirect_cutoff_h_ppm: 0.065,
            struct_shift_cfg: StructShiftConfig::default(),
        }
    }
}

impl AfAssignConfig {
    /// Validate that all configuration scalars are finite and within physically admissible bounds.
    pub fn validate(&self) -> Result<(), String> {
        if !self.af_weight.is_finite() || !(0.0..=5.0).contains(&self.af_weight) {
            return Err(format!(
                "invalid af_weight {}: expected finite value in [0.0, 5.0]",
                self.af_weight
            ));
        }
        if !self.af_clamp_nats.is_finite() || !(0.25..=20.0).contains(&self.af_clamp_nats) {
            return Err(format!(
                "invalid af_clamp_nats {}: expected finite value in [0.25, 20.0] nats",
                self.af_clamp_nats
            ));
        }
        if !self.pae_cutoff_angstrom.is_finite()
            || !(0.5..=32.0).contains(&self.pae_cutoff_angstrom)
        {
            return Err(format!(
                "invalid pae_cutoff_angstrom {}: expected finite value in [0.5, 32.0] A",
                self.pae_cutoff_angstrom
            ));
        }
        if !self.min_contact_plddt.is_finite() || !(0.0..=100.0).contains(&self.min_contact_plddt) {
            return Err(format!(
                "invalid min_contact_plddt {}: expected finite value in [0.0, 100.0]",
                self.min_contact_plddt
            ));
        }
        if !self.noesy_strip_tol_hn.0.is_finite()
            || self.noesy_strip_tol_hn.0 <= 0.0
            || !self.noesy_strip_tol_hn.1.is_finite()
            || self.noesy_strip_tol_hn.1 <= 0.0
        {
            return Err(
                "invalid noesy_strip_tol_hn: tolerances must be finite and > 0".to_string(),
            );
        }
        if !self.noesy_indirect_sigma_h_ppm.is_finite()
            || self.noesy_indirect_sigma_h_ppm <= 0.0
            || !self.noesy_indirect_cutoff_h_ppm.is_finite()
            || self.noesy_indirect_cutoff_h_ppm <= 0.0
        {
            return Err(
                "invalid noesy_indirect_sigma_h_ppm / cutoff: must be finite and > 0".to_string(),
            );
        }
        if !self.struct_shift_cfg.max_seq_identity.is_finite()
            || !(0.0..=DEFAULT_FIREWALL_MAX_SEQ_IDENTITY + 1e-12)
                .contains(&self.struct_shift_cfg.max_seq_identity)
        {
            return Err(format!(
                "HomologFirewall max_seq_identity {} exceeds mandatory <= {:.2} ceiling",
                self.struct_shift_cfg.max_seq_identity, DEFAULT_FIREWALL_MAX_SEQ_IDENTITY
            ));
        }
        Ok(())
    }
}

/// Global chemical-shift referencing offsets `(obs_ppm - af_pred_ppm)` across `[H, N, CA, CB, C']`,
/// estimated strictly without ground-truth labels via grid profile-likelihood + Huber IRLS.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AfShiftOffsets {
    pub h: f64,
    pub n: f64,
    pub ca: f64,
    pub cb: f64,
    pub c: f64,
}

impl AfShiftOffsets {
    #[inline]
    pub const fn as_array(self) -> [f64; 5] {
        [self.h, self.n, self.ca, self.cb, self.c]
    }

    #[inline]
    pub fn get(&self, atom: &str) -> f64 {
        match atom {
            "H" => self.h,
            "N" => self.n,
            "CA" => self.ca,
            "CB" => self.cb,
            "C" => self.c,
            _ => 0.0,
        }
    }

    #[inline]
    pub const fn to_shift_offsets(self) -> ShiftOffsets {
        ShiftOffsets {
            h: self.h,
            n: self.n,
            ca: self.ca,
            cb: self.cb,
            c: self.c,
        }
    }
}

/// Per-residue, per-atom (`H=0, N=1, CA=2, CB=3, C=4`) heavy-tailed Student-`t(nu = 4)` parameters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AfResidueAtomPrior {
    pub present: bool,
    pub mu_af_ppm: f64,
    pub sigma_af_ppm: f64,
    pub mu_bg_ppm: f64,
    pub sigma_bg_ppm: f64,
    pub log_sigma_ratio: f64,
}

impl AfResidueAtomPrior {
    /// Evaluate `ell_{t4}(z^{AF}) - ell_{bg}(obs_ppm)` in nats after subtracting `off_ppm`.
    #[inline]
    pub fn atom_llr(&self, obs_ppm: f64, off_ppm: f64) -> f64 {
        if !self.present {
            return 0.0;
        }
        let v_corr = obs_ppm - off_ppm;
        let z_af = (v_corr - self.mu_af_ppm) / self.sigma_af_ppm;
        let z_bg = (v_corr - self.mu_bg_ppm) / self.sigma_bg_ppm;
        ell_student_t4(z_af) - (ell_student_t4(z_bg) + self.log_sigma_ratio)
    }
}

/// Unassigned `3D 15N-edited NOESY-HSQC` peak `(H_direct, N15, H_indirect, volume)`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Noesy15nPeak {
    pub h_direct_ppm: f64,
    pub n15_ppm: f64,
    pub h_indirect_ppm: f64,
    pub volume: f64,
}

/// Admitted confident amide-amide contact `(j, k)` in `C_AF` (`j < k`, `d_{NN} <= 5.5 A`,
/// `pLDDT_j >= 70`, `pLDDT_k >= 70`, and `sym_PAE(j, k) <= 6.0 A` or `|j - k| <= 5` if PAE absent).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AfAdmittedAmideContact {
    pub seq_j: usize,
    pub seq_k: usize,
    pub dist_hn_angstrom: f64,
    pub symmetrized_pae_angstrom: Option<f64>,
    pub plddt_j: f64,
    pub plddt_k: f64,
    /// Precomputed distance-scaled ceiling before multiplying by `evid_NOE(s_1, s_2) in [0, 1]`.
    pub max_distance_scale_nats: f64,
    pub cap_nats: f64,
    pub is_sequential: bool,
}

/// Audit summary of `PAE` and `pLDDT` contact gating on `3D 15N-NOESY` amide-amide contacts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AfNoesyContactSummary {
    pub total_spatial_amide_pairs_le_5p5: usize,
    pub admitted_contacts_total: usize,
    pub admitted_sequential_contacts: usize,
    pub admitted_medium_long_contacts: usize,
    pub rejected_high_pae_contacts: usize,
    pub rejected_low_plddt_contacts: usize,
    pub noesy_peak_count: usize,
}

/// Precomputed opt-in 3D AlphaFold shift priors (`T21.2` + `HomologFirewall`) and `PAE`-gated
/// `3D 15N-NOESY` amide-amide contact graph aligned to a target assignment `sequence`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AfAssignPriors {
    pub cfg: AfAssignConfig,
    pub sequence: Vec<String>,
    pub seq_ids: Vec<i32>,
    /// Mapping from sequence index `k in 0..K` to residue index in `shift_report.residues`.
    pub seq_to_struct: Vec<Option<usize>>,
    /// Per-residue `pLDDT` aligned to `sequence` (`0.0` when unaligned).
    pub plddt_per_residue: Vec<f64>,
    /// Per-residue smooth `pLDDT`-sigmoid weight `w_AF(pLDDT_k) in [0.0, 1.0)` aligned to `sequence`.
    pub w_af_per_residue: Vec<f64>,
    /// Flat `[k * 5 + atom_idx]` Student-`t4` parameters (`H=0, N=1, CA=2, CB=3, C=4`).
    pub atom_priors: Vec<AfResidueAtomPrior>,
    /// Admitted confident amide-amide contacts `(j < k)` in `C_AF`.
    pub admitted_contacts: Vec<AfAdmittedAmideContact>,
    /// Optional unassigned `3D 15N-edited NOESY` crosspeaks.
    pub noesy_15n_peaks: Vec<Noesy15nPeak>,
    /// Contact gating telemetry (`admitted`, `rejected_high_pae_contacts`, `rejected_low_plddt_contacts`).
    pub contact_summary: AfNoesyContactSummary,
    /// Full `T21.2` shift prediction report carrying `exclusion_list_sha256` & `max_retained_seq_identity`.
    pub shift_report: StructShiftPredictionReport,
}

#[inline]
fn normalize_seq_tokens(sequence: &[String]) -> Vec<String> {
    sequence
        .iter()
        .map(|r| {
            let t = r.trim();
            if t.len() == 1 {
                one_to_three(t.chars().next().unwrap_or('X')).to_string()
            } else {
                t.to_ascii_uppercase()
            }
        })
        .collect()
}

#[inline]
fn effective_contact_plddt(
    provenance: StructCoordinateProvenance,
    res_plddt: Option<f64>,
    override_plddt: Option<f64>,
    pae_plddt: Option<f64>,
) -> f64 {
    if let Some(ov) = override_plddt {
        return ov.clamp(0.0, 100.0);
    }
    if let Some(p) = res_plddt.or(pae_plddt) {
        return p.clamp(0.0, 100.0);
    }
    match provenance {
        // Computational model missing explicit pLDDT fails closed for NOE gating
        StructCoordinateProvenance::Computational => 0.0,
        // Experimental / unmarked coordinates without B-factor pLDDT treat backbone coordinates as ordered
        StructCoordinateProvenance::Experimental | StructCoordinateProvenance::Unknown => 100.0,
    }
}

impl AfAssignPriors {
    /// Build `AfAssignPriors` from a parsed [`StructureModel`], target `sequence`, optional
    /// [`PaeMatrix`], and optional `3D 15N-NOESY` crosspeaks, enforcing the 4-layer
    /// `HomologFirewall` (`max_seq_identity <= 0.25`, `TM-score < 0.50`).
    pub fn from_structure_model(
        structure: &StructureModel,
        sequence: &[String],
        seq_ids_opt: Option<&[i32]>,
        pae_opt: Option<&PaeMatrix>,
        noesy_15n_peaks: Vec<Noesy15nPeak>,
        firewall_query: &HomologFirewallQuery,
        cfg: &AfAssignConfig,
    ) -> Result<Self, String> {
        cfg.validate()?;
        let norm_seq = normalize_seq_tokens(sequence);
        let k_len = norm_seq.len();
        if k_len == 0 {
            return Err("cannot build AfAssignPriors for an empty target sequence".to_string());
        }
        let seq_ids: Vec<i32> = match seq_ids_opt {
            Some(ids) if ids.len() == k_len => ids.to_vec(),
            Some(ids) => {
                return Err(format!(
                    "seq_ids length ({}) does not match sequence length ({k_len})",
                    ids.len()
                ))
            }
            None => (1..=(k_len as i32)).collect(),
        };

        // Validate PAE matrix dimensions if supplied
        let mut structure_work = structure.clone();
        if let Some(pae) = pae_opt {
            if pae.n_residues != structure_work.residues.len() && pae.n_residues != k_len {
                return Err(format!(
                    "PAE matrix dimension N = {} matches neither structure residue count ({}) nor target sequence length ({k_len})",
                    pae.n_residues,
                    structure_work.residues.len()
                ));
            }
            // Populate per-residue pLDDT from PAE JSON (e.g. ColabFold / AF3 JSON) when absent on coordinates
            if let Some(ref plddt_json) = pae.plddt_from_json {
                if plddt_json.len() == structure_work.residues.len() {
                    for (r, &p_val) in structure_work.residues.iter_mut().zip(plddt_json.iter()) {
                        if r.plddt.is_none() && p_val.is_finite() {
                            r.plddt = Some(p_val.clamp(0.0, 100.0));
                        }
                    }
                }
            }
        }

        let shift_report = predict_structure_shifts_with_firewall(
            &structure_work,
            firewall_query,
            &cfg.struct_shift_cfg,
        )?;

        let seq_keys: Vec<(i32, &str)> = seq_ids
            .iter()
            .zip(norm_seq.iter())
            .map(|(&sid, rn)| (sid, rn.as_str()))
            .collect();
        let struct_keys: Vec<(i32, &str)> = structure_work
            .residues
            .iter()
            .map(|r| (r.seq_id, r.res_name.as_str()))
            .collect();
        let aligned_pairs = align_by_res_names(&seq_keys, &struct_keys);
        if aligned_pairs.is_empty() {
            return Err(
                "3D structure sequence could not be aligned to target assignment sequence"
                    .to_string(),
            );
        }

        let mut seq_to_struct = vec![None; k_len];
        for (ki, si) in aligned_pairs {
            if ki < k_len && si < shift_report.residues.len() {
                seq_to_struct[ki] = Some(si);
            }
        }

        let bmrb = Priors::bmrb();
        let mut plddt_per_residue = vec![0.0_f64; k_len];
        let mut w_af_per_residue = vec![0.0_f64; k_len];
        let mut atom_priors = vec![AfResidueAtomPrior::default(); k_len * 5];

        for k in 0..k_len {
            let Some(si) = seq_to_struct[k] else {
                continue;
            };
            let pred_r = &shift_report.residues[si];
            let pae_p = pae_opt
                .and_then(|pae| pae.plddt_from_json.as_ref())
                .and_then(|v| {
                    if v.len() == structure_work.residues.len() {
                        v.get(si).copied()
                    } else {
                        v.get(k).copied()
                    }
                });
            let eff_p = effective_contact_plddt(
                structure_work.provenance,
                pred_r.plddt,
                cfg.struct_shift_cfg.override_uniform_plddt,
                pae_p,
            );
            let w_af = match cfg
                .struct_shift_cfg
                .override_uniform_plddt
                .or(pred_r.plddt)
                .or(pae_p)
            {
                Some(p) => af_plddt_confidence_weight(p),
                None => pred_r.w_af,
            };
            plddt_per_residue[k] = eff_p;
            w_af_per_residue[k] = w_af;

            let rname = &norm_seq[k];
            let cov_k = delta_covalent(&norm_seq, k);
            for (atom_idx, &atom_name) in ATOMS.iter().enumerate() {
                if (rname == "PRO" && atom_name == "H") || (rname == "GLY" && atom_name == "CB") {
                    continue;
                }
                let Some((mu_bmrb, sd_bmrb)) = bmrb.get(rname, atom_name) else {
                    continue;
                };
                let Some(pred_b) = pred_r.backbone.get(atom_name) else {
                    continue;
                };
                if !pred_b.predicted_ppm.is_finite() || !pred_b.predicted_sigma_ppm.is_finite() {
                    continue;
                }
                // Out-of-sample non-homologous 3D shift uncertainty floor (specs/21 section 8 T21.2)
                let sigma_af_floor = match atom_name {
                    "H" => 0.48,
                    "N" => 2.60,
                    "CA" => 1.20,
                    "CB" => 1.30,
                    _ => 1.15,
                };
                let sigma_bg_floor = if atom_name == "H" { 0.12 } else { 0.35 };
                let sigma_af = pred_b.predicted_sigma_ppm.max(sigma_af_floor);
                let sigma_bg = sd_bmrb.max(sigma_bg_floor);
                let mu_bg = mu_bmrb + expected_covalent_atom_offset(cov_k, atom_name);

                atom_priors[k * 5 + atom_idx] = AfResidueAtomPrior {
                    present: true,
                    mu_af_ppm: pred_b.predicted_ppm,
                    sigma_af_ppm: sigma_af,
                    mu_bg_ppm: mu_bg,
                    sigma_bg_ppm: sigma_bg,
                    log_sigma_ratio: (sigma_af / sigma_bg).ln(),
                };
            }
        }

        // Build PAE- & pLDDT-gated confident amide-amide contact graph `C_AF`
        let mut admitted_contacts = Vec::new();
        let mut summary = AfNoesyContactSummary {
            noesy_peak_count: noesy_15n_peaks.len(),
            ..Default::default()
        };

        for j in 1..k_len {
            if norm_seq[j] == "PRO" {
                continue;
            }
            let Some(sj) = seq_to_struct[j] else {
                continue;
            };
            let rj = &structure_work.residues[sj];
            let Some(hj) = rj.h else {
                continue;
            };
            for k in (j + 1)..k_len {
                if norm_seq[k] == "PRO" {
                    continue;
                }
                let Some(sk) = seq_to_struct[k] else {
                    continue;
                };
                let rk = &structure_work.residues[sk];
                let Some(hk) = rk.h else {
                    continue;
                };
                let d_nn = v_dist(hj, hk);
                if !d_nn.is_finite() || !(1.20..=AF_NOESY_MAX_DIST_ANGSTROM).contains(&d_nn) {
                    continue;
                }
                summary.total_spatial_amide_pairs_le_5p5 += 1;

                let pj = plddt_per_residue[j];
                let pk = plddt_per_residue[k];
                if pj < cfg.min_contact_plddt || pk < cfg.min_contact_plddt {
                    summary.rejected_low_plddt_contacts += 1;
                    continue;
                }

                let sym_pae = if let Some(pae) = pae_opt {
                    let (pae_i, pae_k) = if pae.n_residues == structure_work.residues.len() {
                        (sj, sk)
                    } else {
                        (j, k)
                    };
                    let Some(sp) = pae.symmetrized(pae_i, pae_k) else {
                        summary.rejected_high_pae_contacts += 1;
                        continue;
                    };
                    if sp > cfg.pae_cutoff_angstrom {
                        summary.rejected_high_pae_contacts += 1;
                        continue;
                    }
                    Some(sp)
                } else {
                    // When PAE JSON is omitted, admit ONLY local sequence separation |j - k| <= 5
                    // on the same intact polypeptide chain so unverified inter-domain contacts never fire.
                    let seq_sep = k - j;
                    let same_contiguous_chain =
                        rj.chain_id == rk.chain_id && (rk.seq_id - rj.seq_id) == (seq_sep as i32);
                    if seq_sep > AF_NOESY_LOCAL_SEQ_SEP_WITHOUT_PAE || !same_contiguous_chain {
                        summary.rejected_high_pae_contacts += 1;
                        continue;
                    }
                    None
                };

                let is_seq = k == j + 1;
                if is_seq {
                    if d_nn <= AF_NOESY_SEQ_MAX_DIST_ANGSTROM {
                        let d_clamped = d_nn.max(2.20);
                        let max_scale = (1.85 * (3.0 / d_clamped).powi(6))
                            .clamp(0.0, AF_NOESY_MAX_SEQ_BONUS_NATS);
                        admitted_contacts.push(AfAdmittedAmideContact {
                            seq_j: j,
                            seq_k: k,
                            dist_hn_angstrom: d_nn,
                            symmetrized_pae_angstrom: sym_pae,
                            plddt_j: pj,
                            plddt_k: pk,
                            max_distance_scale_nats: max_scale,
                            cap_nats: AF_NOESY_MAX_SEQ_BONUS_NATS,
                            is_sequential: true,
                        });
                        summary.admitted_contacts_total += 1;
                        summary.admitted_sequential_contacts += 1;
                    }
                } else if d_nn <= AF_NOESY_LR_MAX_DIST_ANGSTROM {
                    let d_clamped = d_nn.max(2.20);
                    let max_scale =
                        (1.75 * (3.2 / d_clamped).powi(6)).clamp(0.0, AF_NOESY_MAX_LR_BONUS_NATS);
                    admitted_contacts.push(AfAdmittedAmideContact {
                        seq_j: j,
                        seq_k: k,
                        dist_hn_angstrom: d_nn,
                        symmetrized_pae_angstrom: sym_pae,
                        plddt_j: pj,
                        plddt_k: pk,
                        max_distance_scale_nats: max_scale,
                        cap_nats: AF_NOESY_MAX_LR_BONUS_NATS,
                        is_sequential: false,
                    });
                    summary.admitted_contacts_total += 1;
                    summary.admitted_medium_long_contacts += 1;
                }
            }
        }

        Ok(Self {
            cfg: cfg.clone(),
            sequence: norm_seq,
            seq_ids,
            seq_to_struct,
            plddt_per_residue,
            w_af_per_residue,
            atom_priors,
            admitted_contacts,
            noesy_15n_peaks,
            contact_summary: summary,
            shift_report,
        })
    }

    /// Parse coordinate, optional `PAE` JSON, and optional `3D 15N-NOESY` files from disk and build
    /// [`AfAssignPriors`].
    pub fn from_files(
        structure_path: impl AsRef<Path>,
        sequence: &[String],
        seq_ids_opt: Option<&[i32]>,
        pae_path_opt: Option<&Path>,
        noesy_15n_path_opt: Option<&Path>,
        firewall_query: &HomologFirewallQuery,
        cfg: &AfAssignConfig,
    ) -> Result<Self, String> {
        let model = StructureModel::parse_file(
            structure_path.as_ref(),
            cfg.struct_shift_cfg.force_bfactor_plddt,
        )?;
        let chain_hint = model.residues.first().map(|r| r.chain_id.as_str());
        let pae = match pae_path_opt {
            Some(p) => Some(parse_pae_json_file(p, chain_hint, None)?),
            None => None,
        };
        let noesy_peaks = match noesy_15n_path_opt {
            Some(p) => parse_noesy_15n_peaks_file(p)?,
            None => Vec::new(),
        };
        Self::from_structure_model(
            &model,
            sequence,
            seq_ids_opt,
            pae.as_ref(),
            noesy_peaks,
            firewall_query,
            cfg,
        )
    }

    /// Build [`AfAssignPriors`] directly from in-memory coordinate, optional `PAE` JSON, and
    /// optional `3D 15N-NOESY` strings (`wasm32`-clean).
    #[allow(clippy::too_many_arguments)]
    pub fn from_strs(
        structure_text: &str,
        format_hint: Option<&str>,
        sequence: &[String],
        seq_ids_opt: Option<&[i32]>,
        pae_json_opt: Option<&str>,
        noesy_15n_text_opt: Option<&str>,
        firewall_query: &HomologFirewallQuery,
        cfg: &AfAssignConfig,
    ) -> Result<Self, String> {
        let model = StructureModel::parse_str(
            structure_text,
            format_hint,
            cfg.struct_shift_cfg.force_bfactor_plddt,
        )?;
        let chain_hint = model.residues.first().map(|r| r.chain_id.as_str());
        let pae = match pae_json_opt {
            Some(txt) => Some(parse_pae_json_str_for_chain(txt, chain_hint, None)?),
            None => None,
        };
        let noesy_peaks = match noesy_15n_text_opt {
            Some(txt) => parse_noesy_15n_peaks_str(txt)?,
            None => Vec::new(),
        };
        Self::from_structure_model(
            &model,
            sequence,
            seq_ids_opt,
            pae.as_ref(),
            noesy_peaks,
            firewall_query,
            cfg,
        )
    }

    #[inline]
    pub fn sequence_len(&self) -> usize {
        self.sequence.len()
    }

    #[inline]
    pub fn atom_prior(&self, k: usize, atom_idx: usize) -> AfResidueAtomPrior {
        self.atom_priors
            .get(k * 5 + atom_idx)
            .copied()
            .unwrap_or_default()
    }

    /// Evaluate the clamped heavy-tailed Student-`t(nu = 4)` 3D shift prior `Delta U^{AF}_{s,k}`
    /// (in nats) for placing spin system `ss` onto sequence index `k`, given unlabelled `offsets`.
    pub fn compute_af_delta_u_cell(
        &self,
        ss: &SpinSystem,
        k: usize,
        offsets: &AfShiftOffsets,
        use_hn: bool,
        max_cands_cap: Option<usize>,
    ) -> f64 {
        let k_len = self.sequence.len();
        if k == 0 || k >= k_len || self.sequence[k] == "PRO" || self.cfg.af_weight <= 0.0 {
            return 0.0;
        }
        let w_k = self.w_af_per_residue[k];
        if w_k <= 0.0 {
            return 0.0;
        }

        // Sum strictly over observed intra-residue `a in O_s` (`off == 0`: CA_i, CB_i, C'_i, N_i, H_i)
        // at residue `k`, matching specs/21 section 4.2 eq. without double-counting `k - 1`.
        let mut raw_sum = 0.0_f64;
        for (atom_c, atom_name) in ["CA", "CB", "C"].iter().enumerate() {
            let atom_idx = atom_c + 2;
            let ae = self.atom_prior(k, atom_idx);
            if !ae.present {
                continue;
            }
            let Some(raw_list) = ss.cands.get(&((*atom_name).to_string(), 0)) else {
                continue;
            };
            let off_ppm = offsets.get(atom_name);
            let slice = match max_cands_cap {
                Some(cap) => &raw_list[..raw_list.len().min(cap)],
                None => raw_list.as_slice(),
            };
            let mut n_pos = 0usize;
            let mut best_af = f64::NEG_INFINITY;
            for &(val, sup) in slice {
                if sup > 0 {
                    n_pos += 1;
                    let term = ae.atom_llr(val, off_ppm);
                    if term > best_af {
                        best_af = term;
                    }
                }
            }
            if n_pos > 0 {
                raw_sum += best_af - (n_pos as f64).ln();
            }
        }

        if use_hn {
            let ae_n = self.atom_prior(k, 1);
            let ae_h = self.atom_prior(k, 0);
            raw_sum += ae_n.atom_llr(ss.n, offsets.n) + ae_h.atom_llr(ss.h, offsets.h);
        }

        let clamped = raw_sum.clamp(-self.cfg.af_clamp_nats, self.cfg.af_clamp_nats);
        self.cfg.af_weight * w_k * clamped
    }

    /// Add `Delta U^{AF}_{s,k}` in-place onto all admissible (`> FORBID * 0.5`) cells of an
    /// unlocked `U (S x K)` matrix, estimating global unlabelled offsets automatically.
    pub fn apply_to_unary_matrix(
        &self,
        u: &mut Matrix2D,
        systems: &[SpinSystem],
        use_hn: bool,
    ) -> AfShiftOffsets {
        let offsets = estimate_af_shift_offsets(systems, self);
        self.apply_to_unary_matrix_with_offsets(u, systems, &offsets, use_hn);
        offsets
    }

    /// Add `Delta U^{AF}_{s,k}` in-place onto all admissible (`> FORBID * 0.5`) cells of `u`
    /// using pre-estimated `offsets`.
    pub fn apply_to_unary_matrix_with_offsets(
        &self,
        u: &mut Matrix2D,
        systems: &[SpinSystem],
        offsets: &AfShiftOffsets,
        use_hn: bool,
    ) {
        let s_len = u.rows.min(systems.len());
        let k_len = u.cols.min(self.sequence.len());
        if s_len == 0 || k_len <= 1 || self.cfg.af_weight <= 0.0 {
            return;
        }
        let max_cands = systems
            .iter()
            .flat_map(|ss| ss.cands.values().map(|c| c.len()))
            .max()
            .unwrap_or(1)
            .max(1);

        for k in 1..k_len {
            if self.sequence[k] == "PRO" || self.w_af_per_residue[k] <= 0.0 {
                continue;
            }
            for (s, ss) in systems.iter().enumerate().take(s_len) {
                let base = u.get(s, k);
                if base <= FORBID * 0.5 {
                    continue;
                }
                let du = self.compute_af_delta_u_cell(ss, k, offsets, use_hn, Some(max_cands));
                u.set(s, k, base + du);
            }
        }
    }

    /// Build a fast `O(1)`-per-move pairwise `15N-NOESY` lookup table for `systems` (`None` when
    /// either `noesy_15n_peaks` or `admitted_contacts` is empty).
    pub fn build_noesy_pairwise_table(
        &self,
        systems: &[SpinSystem],
    ) -> Option<AfNoesyPairwiseTable> {
        AfNoesyPairwiseTable::build(systems, self)
    }
}

// ============================================================================
// 1. Unlabelled Grid Profile-Likelihood + Huber IRLS Offset Registration
// ============================================================================

#[inline]
fn spin_system_canonical_rank_key(ss: &SpinSystem) -> (usize, usize, u64, u64, u64) {
    let mut slots = 0usize;
    let mut sup_sum = 0usize;
    let mut first_ca = 0.0_f64;
    for ((atom, off), cands) in &ss.cands {
        let mut best_pos: Option<(usize, f64)> = None;
        for &(v, sup) in cands {
            if sup > 0 && v.is_finite() {
                best_pos = match best_pos {
                    None => Some((sup, v)),
                    Some((bs, bv)) => {
                        if sup > bs || (sup == bs && v.total_cmp(&bv).is_lt()) {
                            Some((sup, v))
                        } else {
                            Some((bs, bv))
                        }
                    }
                };
            }
        }
        if let Some((bs, bv)) = best_pos {
            slots += 1;
            sup_sum += bs;
            if atom == "CA" && *off == 0 {
                first_ca = bv;
            }
        }
    }
    (
        sup_sum,
        slots,
        ss.h.to_bits(),
        ss.n.to_bits(),
        first_ca.to_bits(),
    )
}

#[inline]
fn best_positive_slot_value(ss: &SpinSystem, atom: &str, off: i32) -> Option<f64> {
    let list = ss.cands.get(&(atom.to_string(), off))?;
    let mut best: Option<(usize, f64)> = None;
    for &(v, sup) in list {
        if sup > 0 && v.is_finite() {
            best = match best {
                None => Some((sup, v)),
                Some((bs, bv)) => {
                    if sup > bs || (sup == bs && v.total_cmp(&bv).is_lt()) {
                        Some((sup, v))
                    } else {
                        Some((bs, bv))
                    }
                }
            };
        }
    }
    best.map(|(_, v)| v)
}

#[inline]
fn fit_single_channel_profile_huber_irls(
    obs_sorted: &[f64],
    preds_sorted: &[(f64, f64)],
    max_search_ppm: f64,
    grid_step_ppm: f64,
    spectral_range_ppm: f64,
    huber_k_ppm: f64,
) -> f64 {
    if obs_sorted.len() < 4 || preds_sorted.len() < 4 {
        return 0.0;
    }
    const OUTLIER_EPS: f64 = 0.05;
    let fg_scale = (1.0 - OUTLIER_EPS) / (preds_sorted.len() as f64);
    let bg_const = OUTLIER_EPS / spectral_range_ppm;

    let n_steps = ((2.0 * max_search_ppm) / grid_step_ppm).round() as usize + 1;
    let mut scores = vec![0.0_f64; n_steps];
    let mut best_idx = n_steps / 2;
    let mut best_ll = f64::NEG_INFINITY;

    // Stage 1: Unlabelled Student-t4 marginal mixture grid profile-likelihood
    for (g_idx, slot) in scores.iter_mut().enumerate() {
        let d = -max_search_ppm + (g_idx as f64) * grid_step_ppm;
        let mut ll = 0.0_f64;
        for &x in obs_sorted {
            let mut mix = 0.0_f64;
            for &(mu, sig) in preds_sorted {
                let z = (x - (mu + d)) / sig;
                let dens = STUDENT_T4_NORM_CONST * (1.0 + 0.25 * z * z).powf(-2.5) / sig;
                mix += dens;
            }
            ll += (fg_scale * mix + bg_const).max(1e-30).ln();
        }
        *slot = ll;
        if ll > best_ll {
            best_ll = ll;
            best_idx = g_idx;
        }
    }

    let mut d_est = -max_search_ppm + (best_idx as f64) * grid_step_ppm;
    if best_idx > 0 && best_idx + 1 < n_steps {
        let ym1 = scores[best_idx - 1];
        let y0 = scores[best_idx];
        let yp1 = scores[best_idx + 1];
        let denom = ym1 - 2.0 * y0 + yp1;
        if denom < -1e-12 {
            let sub = (0.5 * (ym1 - yp1) / denom).clamp(-0.5, 0.5);
            d_est += sub * grid_step_ppm;
        }
    }

    // Stage 2: Responsibility-weighted Huber IRLS refinement (`12` iterations)
    for _ in 0..12 {
        let mut num = 0.0_f64;
        let mut den = 0.0_f64;
        for &x in obs_sorted {
            let mut tot_p = bg_const;
            for &(mu, sig) in preds_sorted {
                let z = (x - (mu + d_est)) / sig;
                let u = fg_scale * STUDENT_T4_NORM_CONST * (1.0 + 0.25 * z * z).powf(-2.5) / sig;
                tot_p += u;
            }
            let inv_tot = 1.0 / tot_p.max(1e-30);
            for &(mu, sig) in preds_sorted {
                let raw_diff = x - mu;
                let resid = (raw_diff - d_est).abs();
                let z = resid / sig;
                let u = fg_scale * STUDENT_T4_NORM_CONST * (1.0 + 0.25 * z * z).powf(-2.5) / sig;
                let resp = u * inv_tot;
                let huber_w = if resid <= huber_k_ppm {
                    1.0
                } else {
                    huber_k_ppm / resid
                };
                let w = resp * huber_w / (sig * sig);
                num += w * raw_diff;
                den += w;
            }
        }
        if den <= 1e-15 {
            break;
        }
        let next_d = (num / den).clamp(-max_search_ppm, max_search_ppm);
        let step_diff = (next_d - d_est).abs();
        d_est = next_d;
        if step_diff < 1e-13 {
            break;
        }
    }

    d_est
}

/// Estimate global referencing offsets `(dH, dN, dCA, dCB, dC')` (`obs - pred`) strictly from the
/// unordered multiset of `SpinSystem` shifts and unordered `T21.2` [`ResidueShiftPrediction`]s
/// via grid profile-likelihood + Huber IRLS (`0` ground-truth labels; exact `< 1e-12` permutation
/// invariance under arbitrary permutations of both `systems` and `predictions`).
pub fn estimate_af_shift_offsets_from_predictions(
    systems: &[SpinSystem],
    predictions: &[ResidueShiftPrediction],
) -> AfShiftOffsets {
    if systems.is_empty() || predictions.is_empty() {
        return AfShiftOffsets::default();
    }

    // Canonicalize spin-system selection deterministically so arbitrary permutations of `systems`
    // select the exact same subset in identical canonical order.
    let max_selected = ((predictions.len() * 4) / 3).max(24);
    let mut ranked: Vec<&SpinSystem> = systems
        .iter()
        .filter(|ss| spin_system_canonical_rank_key(ss).1 >= 2)
        .collect();
    if ranked.len() < 8 {
        ranked = systems
            .iter()
            .filter(|ss| ss.has_observed_carbons())
            .collect();
    }
    if ranked.is_empty() {
        ranked = systems.iter().collect();
    }
    ranked.sort_by(|a, b| {
        let ka = spin_system_canonical_rank_key(a);
        let kb = spin_system_canonical_rank_key(b);
        kb.0.cmp(&ka.0)
            .then_with(|| kb.1.cmp(&ka.1))
            .then_with(|| a.h.total_cmp(&b.h))
            .then_with(|| a.n.total_cmp(&b.n))
            .then_with(|| ka.4.cmp(&kb.4))
    });
    if ranked.len() > max_selected {
        ranked.truncate(max_selected);
    }

    let collect_preds =
        |atom: &str, is_pro_excluded: bool, is_gly_excluded: bool| -> Vec<(f64, f64)> {
            let floor = if atom == "H" { 0.15 } else { 0.40 };
            let mut v: Vec<(f64, f64)> = predictions
                .iter()
                .filter(|r| {
                    !(is_pro_excluded && r.res_name == "PRO")
                        && !(is_gly_excluded && r.res_name == "GLY")
                })
                .filter_map(|r| {
                    let b = r.backbone.get(atom)?;
                    if b.predicted_ppm.is_finite() && b.predicted_sigma_ppm.is_finite() {
                        Some((b.predicted_ppm, b.predicted_sigma_ppm.max(floor)))
                    } else {
                        None
                    }
                })
                .collect();
            v.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.total_cmp(&b.1)));
            v
        };

    let collect_carbon_obs = |atom: &str, lo: f64, hi: f64| -> Vec<f64> {
        let mut obs = Vec::new();
        for ss in &ranked {
            for off in [0, -1] {
                if let Some(v) = best_positive_slot_value(ss, atom, off) {
                    if (lo..=hi).contains(&v) {
                        obs.push(v);
                    }
                }
            }
        }
        obs.sort_by(f64::total_cmp);
        obs
    };

    let mut obs_h: Vec<f64> = ranked
        .iter()
        .map(|ss| ss.h)
        .filter(|v| v.is_finite() && (4.5..=13.0).contains(v))
        .collect();
    obs_h.sort_by(f64::total_cmp);

    let mut obs_n: Vec<f64> = ranked
        .iter()
        .map(|ss| ss.n)
        .filter(|v| v.is_finite() && (95.0..=145.0).contains(v))
        .collect();
    obs_n.sort_by(f64::total_cmp);

    let obs_ca = collect_carbon_obs("CA", 34.0, 76.0);
    let obs_cb = collect_carbon_obs("CB", 8.0, 78.0);
    let obs_c = collect_carbon_obs("C", 160.0, 190.0);

    let preds_h = collect_preds("H", true, false);
    let preds_n = collect_preds("N", true, false);
    let preds_ca = collect_preds("CA", false, false);
    let preds_cb = collect_preds("CB", false, true);
    let preds_c = collect_preds("C", false, false);

    let h = fit_single_channel_profile_huber_irls(&obs_h, &preds_h, 2.5, 0.01, 5.0, 0.35);
    let n = fit_single_channel_profile_huber_irls(&obs_n, &preds_n, 12.0, 0.05, 40.0, 1.80);
    let ca = fit_single_channel_profile_huber_irls(&obs_ca, &preds_ca, 12.0, 0.05, 36.0, 0.90);
    let cb = if obs_cb.len() >= 4 {
        fit_single_channel_profile_huber_irls(&obs_cb, &preds_cb, 12.0, 0.05, 64.0, 1.00)
    } else {
        ca
    };
    let c = fit_single_channel_profile_huber_irls(&obs_c, &preds_c, 12.0, 0.05, 24.0, 0.90);

    AfShiftOffsets { h, n, ca, cb, c }
}

/// Estimate global referencing offsets `(dH, dN, dCA, dCB, dC')` between unordered `systems` and
/// unordered 3D shift predictions inside `af_priors` (`0` ground-truth labels; `< 1e-12`
/// permutation-invariant).
pub fn estimate_af_shift_offsets(
    systems: &[SpinSystem],
    af_priors: &AfAssignPriors,
) -> AfShiftOffsets {
    // Restrict predicted multiset to residues aligned to the target sequence
    let aligned_preds: Vec<ResidueShiftPrediction> = af_priors
        .seq_to_struct
        .iter()
        .filter_map(|&si_opt| si_opt.and_then(|si| af_priors.shift_report.residues.get(si)))
        .cloned()
        .collect();
    estimate_af_shift_offsets_from_predictions(systems, &aligned_preds)
}

// ============================================================================
// 2. PAE-Gated 3D 15N-NOESY Amide-Amide Contact Pairwise Table & Parsers
// ============================================================================

#[derive(Clone, Copy, Debug)]
struct AdjNoeContact {
    contact_id: usize,
    partner_k: usize,
    max_distance_scale_nats: f64,
    cap_nats: f64,
    is_sequential: bool,
}

/// Fast `O(1)`-per-move pairwise scorer for `PAE`-gated `3D 15N-NOESY` sequential `(k - 1, k)` and
/// medium/long-range `|j - k| >= 2` amide-amide contacts.
#[derive(Clone, Debug)]
pub struct AfNoesyPairwiseTable {
    s_len: usize,
    k_len: usize,
    /// Symmetric `S x S` reciprocity-weighted experimental NOE crosspeak evidence in `[0.0, 1.0]`.
    evid_matrix: Vec<f64>,
    contacts: Vec<AfAdmittedAmideContact>,
    adj_by_k: Vec<Vec<AdjNoeContact>>,
    seq_edge_scale: Vec<Option<f64>>,
}

impl AfNoesyPairwiseTable {
    /// Precompute `(s_1, s_2)` bidirectional `3D 15N-NOESY` crosspeak reciprocity and map onto
    /// `af_priors.admitted_contacts` (`C_AF`).
    pub fn build(systems: &[SpinSystem], af_priors: &AfAssignPriors) -> Option<Self> {
        let s_len = systems.len();
        let k_len = af_priors.sequence.len();
        if s_len < 2
            || k_len < 2
            || af_priors.noesy_15n_peaks.is_empty()
            || af_priors.admitted_contacts.is_empty()
        {
            return None;
        }

        let (tol_h, tol_n) = af_priors.cfg.noesy_strip_tol_hn;
        let sig_h = af_priors.cfg.noesy_indirect_sigma_h_ppm.max(1e-4);
        let cut_h = af_priors.cfg.noesy_indirect_cutoff_h_ppm.max(sig_h);

        // Attach indirect 1H crosspeaks to each HSQC root strip `(ss.h, ss.n)`
        let mut strip_h2: Vec<Vec<f64>> = vec![Vec::new(); s_len];
        for pk in &af_priors.noesy_15n_peaks {
            if !pk.h_direct_ppm.is_finite()
                || !pk.n15_ppm.is_finite()
                || !pk.h_indirect_ppm.is_finite()
            {
                continue;
            }
            for (s, ss) in systems.iter().enumerate() {
                let d_hn = ((pk.h_direct_ppm - ss.h) / tol_h).hypot((pk.n15_ppm - ss.n) / tol_n);
                if d_hn <= 1.0 && (pk.h_indirect_ppm - ss.h).abs() >= 0.022 {
                    strip_h2[s].push(pk.h_indirect_ppm);
                }
            }
        }

        if strip_h2.iter().all(Vec::is_empty) {
            return None;
        }

        // Local 1H^N spectral crowding factor per spin system
        let crowd_window = 2.0 * sig_h;
        let mut h_crowding = vec![1usize; s_len];
        for s in 0..s_len {
            let mut cnt = 1usize;
            for t in 0..s_len {
                if t != s && (systems[t].h - systems[s].h).abs() <= crowd_window {
                    cnt += 1;
                }
            }
            h_crowding[s] = cnt;
        }

        let best_match = |h2_list: &[f64], target_h: f64| -> f64 {
            let mut best = 0.0_f64;
            for &h2 in h2_list {
                let dh = (h2 - target_h).abs();
                if dh <= cut_h {
                    let m = (-0.5 * (dh / sig_h).powi(2)).exp();
                    if m > best {
                        best = m;
                    }
                }
            }
            best
        };

        let mut evid_matrix = vec![0.0_f64; s_len * s_len];
        let mut any_evid = false;
        for s1 in 0..s_len {
            for s2 in (s1 + 1)..s_len {
                let m_fwd = best_match(&strip_h2[s1], systems[s2].h);
                let m_rev = best_match(&strip_h2[s2], systems[s1].h);
                if m_fwd > 0.0 || m_rev > 0.0 {
                    let crowd = (0.5 * ((h_crowding[s1] + h_crowding[s2]) as f64))
                        .max(1.0)
                        .sqrt();
                    let evid = ((0.35 * (m_fwd + m_rev) + 0.30 * (m_fwd * m_rev).sqrt()) / crowd)
                        .clamp(0.0, 1.0);
                    if evid > 1e-9 {
                        evid_matrix[s1 * s_len + s2] = evid;
                        evid_matrix[s2 * s_len + s1] = evid;
                        any_evid = true;
                    }
                }
            }
        }
        if !any_evid {
            return None;
        }

        let mut adj_by_k = vec![Vec::new(); k_len];
        let mut seq_edge_scale = vec![None; k_len];
        for (cid, c) in af_priors.admitted_contacts.iter().enumerate() {
            if c.seq_j >= k_len || c.seq_k >= k_len {
                continue;
            }
            adj_by_k[c.seq_j].push(AdjNoeContact {
                contact_id: cid,
                partner_k: c.seq_k,
                max_distance_scale_nats: c.max_distance_scale_nats,
                cap_nats: c.cap_nats,
                is_sequential: c.is_sequential,
            });
            adj_by_k[c.seq_k].push(AdjNoeContact {
                contact_id: cid,
                partner_k: c.seq_j,
                max_distance_scale_nats: c.max_distance_scale_nats,
                cap_nats: c.cap_nats,
                is_sequential: c.is_sequential,
            });
            if c.is_sequential && c.seq_j + 1 == c.seq_k {
                seq_edge_scale[c.seq_k] = Some(c.max_distance_scale_nats);
            }
        }

        Some(Self {
            s_len,
            k_len,
            evid_matrix,
            contacts: af_priors.admitted_contacts.clone(),
            adj_by_k,
            seq_edge_scale,
        })
    }

    #[inline]
    pub fn num_contacts(&self) -> usize {
        self.contacts.len()
    }

    #[inline]
    pub fn evidence(&self, s1: usize, s2: usize) -> f64 {
        if s1 < self.s_len && s2 < self.s_len {
            self.evid_matrix[s1 * self.s_len + s2]
        } else {
            0.0
        }
    }

    /// Evaluate the exact `PAE`-gated pairwise bonus for placing `(s1, s2)` at `(j, k)` (`0.0`
    /// whenever `(min(j,k), max(j,k))` is not in `C_AF`, including any `PAE > 6.0 A` pair).
    pub fn pair_bonus(&self, s1: usize, s2: usize, j: usize, k: usize) -> f64 {
        if s1 == s2
            || s1 >= self.s_len
            || s2 >= self.s_len
            || j == k
            || j >= self.k_len
            || k >= self.k_len
        {
            return 0.0;
        }
        let ev = self.evid_matrix[s1 * self.s_len + s2];
        if ev <= 0.0 {
            return 0.0;
        }
        for e in &self.adj_by_k[j] {
            if e.partner_k == k {
                return (e.max_distance_scale_nats * ev).clamp(0.0, e.cap_nats);
            }
        }
        0.0
    }

    /// Sequential `(k - 1, k)` NOE bonus `Delta P^{NOE}_{s_1, s_2}(k - 1, k) in [0, +2.50]` nats.
    #[inline]
    pub fn sequential_bonus(&self, s_prev: usize, s_curr: usize, k_curr: usize) -> f64 {
        if s_prev == s_curr
            || s_prev >= self.s_len
            || s_curr >= self.s_len
            || k_curr == 0
            || k_curr >= self.k_len
        {
            return 0.0;
        }
        let Some(scale) = self.seq_edge_scale[k_curr] else {
            return 0.0;
        };
        let ev = self.evid_matrix[s_prev * self.s_len + s_curr];
        (scale * ev).clamp(0.0, AF_NOESY_MAX_SEQ_BONUS_NATS)
    }

    /// Sum of all incident sequential + medium/long-range NOE bonuses on `spin` placed at `slot`
    /// given current sequence assignment `at`.
    #[inline]
    pub fn site_bonus(&self, spin: usize, slot: usize, at: &[i32]) -> f64 {
        if spin >= self.s_len || slot >= self.k_len {
            return 0.0;
        }
        let mut sc = 0.0_f64;
        for e in &self.adj_by_k[slot] {
            if let Some(&sp_raw) = at.get(e.partner_k) {
                if sp_raw >= 0 {
                    let sp = sp_raw as usize;
                    if sp < self.s_len && sp != spin {
                        let ev = self.evid_matrix[spin * self.s_len + sp];
                        sc += (e.max_distance_scale_nats * ev).clamp(0.0, e.cap_nats);
                    }
                }
            }
        }
        sc
    }

    /// Sum of only medium/long-range (`|j - k| >= 2`) NOE bonuses incident on `(spin, slot)`.
    #[inline]
    pub fn medium_long_site_bonus(&self, spin: usize, slot: usize, at: &[i32]) -> f64 {
        if spin >= self.s_len || slot >= self.k_len {
            return 0.0;
        }
        let mut sc = 0.0_f64;
        for e in &self.adj_by_k[slot] {
            if e.is_sequential {
                continue;
            }
            if let Some(&sp_raw) = at.get(e.partner_k) {
                if sp_raw >= 0 {
                    let sp = sp_raw as usize;
                    if sp < self.s_len && sp != spin {
                        let ev = self.evid_matrix[spin * self.s_len + sp];
                        sc += (e.max_distance_scale_nats * ev).clamp(0.0, e.cap_nats);
                    }
                }
            }
        }
        sc
    }

    /// Deduplicated local NOE score across all contacts incident on `changed_residues`, using
    /// `contact_marks` + `mark_gen` so a contact `(j, k)` with both `j, k in changed_residues`
    /// is evaluated once and only once (preserving exact MCMC delta telescoping).
    #[inline]
    pub fn local_subset_bonus(
        &self,
        at: &[i32],
        changed_residues: &[usize],
        contact_marks: &mut [u32],
        mark_gen: u32,
    ) -> f64 {
        let mut sc = 0.0_f64;
        for &k in changed_residues {
            if k >= self.k_len {
                continue;
            }
            for e in &self.adj_by_k[k] {
                if contact_marks[e.contact_id] == mark_gen {
                    continue;
                }
                contact_marks[e.contact_id] = mark_gen;
                let a = at.get(k).copied().unwrap_or(-1);
                let b = at.get(e.partner_k).copied().unwrap_or(-1);
                if a >= 0 && b >= 0 && a != b {
                    let sa = a as usize;
                    let sb = b as usize;
                    if sa < self.s_len && sb < self.s_len {
                        let ev = self.evid_matrix[sa * self.s_len + sb];
                        sc += (e.max_distance_scale_nats * ev).clamp(0.0, e.cap_nats);
                    }
                }
            }
        }
        sc
    }

    /// Exact total `15N-NOESY` pairwise bonus summed over all admitted contacts `(j < k)` in `C_AF`.
    pub fn total_bonus(&self, at: &[i32]) -> f64 {
        let mut sc = 0.0_f64;
        for c in &self.contacts {
            let a = at.get(c.seq_j).copied().unwrap_or(-1);
            let b = at.get(c.seq_k).copied().unwrap_or(-1);
            if a >= 0 && b >= 0 && a != b {
                let sa = a as usize;
                let sb = b as usize;
                if sa < self.s_len && sb < self.s_len {
                    let ev = self.evid_matrix[sa * self.s_len + sb];
                    sc += (c.max_distance_scale_nats * ev).clamp(0.0, c.cap_nats);
                }
            }
        }
        sc
    }
}

/// Convert generic `nmr-core` [`Peak`]s (`NOESY15N`) into [`Noesy15nPeak`] records.
pub fn noesy_15n_peaks_from_peaks(peaks: &[Peak]) -> Result<Vec<Noesy15nPeak>, String> {
    let mut out = Vec::new();
    for p in peaks {
        let exp_up = p
            .exp
            .trim()
            .to_ascii_uppercase()
            .replace(['(', ')', '-', '_'], "");
        let is_n15_noesy = matches!(
            exp_up.as_str(),
            "NOESY15N" | "15NNOESY" | "N15NOESY" | "15NNOESYHSQC" | "NOESYHSQC" | "3D15NNOESY"
        );
        if !is_n15_noesy {
            continue;
        }
        let Some(h2) = p.h2.or(p.c) else {
            continue;
        };
        let vol = p.volume.or(p.height).unwrap_or(1.0);
        if !p.h.is_finite() || !p.n.is_finite() || !h2.is_finite() || !vol.is_finite() {
            return Err(
                "non-finite chemical shift or volume in 3D 15N-NOESY peak list".to_string(),
            );
        }
        out.push(Noesy15nPeak {
            h_direct_ppm: p.h,
            n15_ppm: p.n,
            h_indirect_ppm: h2,
            volume: vol.abs().max(1.0),
        });
    }
    Ok(out)
}

/// Parse a `3D 15N-edited NOESY-HSQC` peak list from `.tsv` or CCPN `.nef` text.
pub fn parse_noesy_15n_peaks_str(text: &str) -> Result<Vec<Noesy15nPeak>, String> {
    if text.len() > MAX_STRUCTURE_FILE_BYTES {
        return Err(format!(
            "15N-NOESY peak input ({} bytes) exceeds 64 MiB limit",
            text.len()
        ));
    }
    let head = text
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty() && !l.starts_with('#'))
        .unwrap_or("");
    if head.starts_with("data_nef") || text.contains("save_nef_nmr_spectrum") {
        let doc = read_nef_str(text)?;
        let pks = doc.to_peaks();
        return noesy_15n_peaks_from_peaks(&pks);
    }

    // First try standard FERRIC Peak TSV format (`experiment\tH\tN\tC\tsign\theight\tvolume\tH2`)
    if let Ok(std_peaks) = read_peaks_str(text) {
        if let Ok(conv) = noesy_15n_peaks_from_peaks(&std_peaks) {
            if !conv.is_empty() {
                return Ok(conv);
            }
        }
    }

    // Fallback: compact 3/4-column `(H_direct, N15, H_indirect[, volume])` or `(exp, H_direct, N15, H_indirect[, volume])` TSV
    let mut out = Vec::new();
    for (line_idx, raw_line) in text.lines().enumerate() {
        let line_no = line_idx + 1;
        let trimmed = raw_line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = if trimmed.contains('\t') {
            trimmed.split('\t').map(str::trim).collect()
        } else {
            trimmed.split_whitespace().collect()
        };
        if cols.is_empty() {
            continue;
        }
        let low0 = cols[0].to_ascii_lowercase();
        if matches!(
            low0.as_str(),
            "experiment" | "exp" | "h" | "h_direct" | "hn" | "f3" | "w1" | "peak_id"
        ) {
            continue;
        }
        let start = if cols[0].parse::<f64>().is_ok() {
            0usize
        } else {
            1usize
        };
        if cols.len() < start + 3 {
            return Err(format!(
                "15N-NOESY line {line_no}: expected at least 3 numeric columns (H_direct, N15, H_indirect)"
            ));
        }
        let parse_f = |s: &str, name: &str| -> Result<f64, String> {
            let v: f64 = s
                .parse()
                .map_err(|e| format!("15N-NOESY line {line_no}: invalid {name} '{s}': {e}"))?;
            if !v.is_finite() {
                return Err(format!("15N-NOESY line {line_no}: non-finite {name} '{s}'"));
            }
            Ok(v)
        };
        let h_dir = parse_f(cols[start], "H_direct")?;
        let n15 = parse_f(cols[start + 1], "N15")?;
        let h_ind = parse_f(cols[start + 2], "H_indirect")?;
        let vol = if let Some(&v_tok) = cols.get(start + 3) {
            if v_tok.is_empty() || v_tok == "." {
                1.0
            } else {
                parse_f(v_tok, "volume")?.abs().max(1.0)
            }
        } else {
            1.0
        };
        out.push(Noesy15nPeak {
            h_direct_ppm: h_dir,
            n15_ppm: n15,
            h_indirect_ppm: h_ind,
            volume: vol,
        });
    }
    Ok(out)
}

/// Read and parse a `3D 15N-edited NOESY-HSQC` peak list (`.tsv` or `.nef`) from disk (`<= 64 MiB`).
pub fn parse_noesy_15n_peaks_file(path: impl AsRef<Path>) -> Result<Vec<Noesy15nPeak>, String> {
    let p = path.as_ref();
    let meta = fs::metadata(p)
        .map_err(|e| format!("failed to stat 15N-NOESY file '{}': {e}", p.display()))?;
    if meta.len() > MAX_STRUCTURE_FILE_BYTES as u64 {
        return Err(format!(
            "15N-NOESY file '{}' ({} bytes) exceeds 64 MiB limit",
            p.display(),
            meta.len()
        ));
    }
    let bytes =
        fs::read(p).map_err(|e| format!("failed to read 15N-NOESY file '{}': {e}", p.display()))?;
    if bytes.len() > MAX_STRUCTURE_FILE_BYTES {
        return Err(format!(
            "15N-NOESY file '{}' exceeds 64 MiB limit",
            p.display()
        ));
    }
    let text = std::str::from_utf8(&bytes)
        .map_err(|e| format!("15N-NOESY file '{}' is not valid UTF-8: {e}", p.display()))?;
    parse_noesy_15n_peaks_str(text)
}
