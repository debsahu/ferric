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
//! - Genest, C. & Zidek, J. V. (*Statist. Sci.* **1986**, 1, 114-148) -- logarithmic
//!   (log-linear) opinion pooling, used here as the per-nucleus replacement rule
//!   `ell_a = (1 - omega) ell_seq,a + omega ell_AF,a` (section "Replacement algebra" below).
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
//!
//! # Replacement algebra (`21.W2:P1-1`, design critique `A2`)
//! The sequence-only unary score is a sum of per-observation log-likelihood ratios against one
//! pooled residue-averaged background `bg_a(v)` (`Priors::background`):
//! `ell_seq,a(v) = max(ln N(v | mu_res,a + d_cov + off_a, sd_res,a) - bg_a(v), -12)`.
//! For every observed nucleus `a in {H, N, CA, CB, C'}` of the target residue `r` (`r = k` for
//! intra-residue observations, `r = k - 1` for the `(i - 1)` carbons) the AF prior **replaces**
//! that term by the log-linear pool
//! `ell_a(v) = (1 - omega_r) ell_seq,a(v) + omega_r ell_AF,a(v)`,
//! `ell_AF,a(v) = max(ln t4(v | mu_AF,r,a + off_a, sigma_AF,r,a) - bg_a(v), -12)`,
//! with `omega_r = clamp(lambda_AF * w_AF(pLDDT_r), 0, 1)`. Both terms share the same
//! background `bg_a`, the same `-12` floor and the same reference frame (`off_a` is the
//! sequence-only offset actually applied by `prepare_scoring_priors` / evo-graph, so
//! `--no-auto-ref` sets it to zero for both). Candidate choice is joint: each slot keeps
//! `max_c [ell_a(v_c) + support_bonus (sup_c - 1)] - ln(n_pos)`, so `ln(n_pos)` appears exactly
//! once (inside the unchanged sequence-only term) and cancels in
//! `Delta U^{AF}_{s,k} = U^{pool}_{s,k} - U^{seq}_{s,k}`, which is finally clamped to
//! `[-lambda_AF U_max, +lambda_AF U_max]`.
//!
//! The pooled per-nucleus likelihood is **not** separately calibrated: confidences produced
//! with AF priors carry `uncalibrated_af_prior = true` (`confidence_calibrated = false`,
//! `21.W2:P1-2`).

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::af_concordance::{parse_pae_json_str_for_chain, PaeMatrix};
use crate::dihedrals::align_by_res_names;
use crate::nef::read_nef_str;
use crate::noesy::v_dist;
use crate::peaks::{read_peaks_str, Peak};
use crate::priors::{
    covalent_atom_offset, delta_covalent, one_to_three, Priors, ShiftOffsets, ATOMS,
};
use crate::solver::{Matrix2D, ScoreConfig, FORBID};
use crate::spinsystems::SpinSystem;
use crate::struct_shifts::{
    af_plddt_confidence_weight, predict_structure_shifts_with_firewall, ResidueShiftPrediction,
    StructCoordinateProvenance, StructShiftConfig, StructShiftPredictionReport, StructureModel,
    MAX_STRUCTURE_FILE_BYTES,
};
use crate::torsion_db::{HomologFirewallQuery, DEFAULT_FIREWALL_MAX_SEQ_IDENTITY};

/// Default AF replacement weight `lambda_AF = 0.65` (`omega_r = lambda_AF * w_AF(pLDDT_r)`).
pub const DEFAULT_AF_WEIGHT: f64 = 0.65;

/// Default per-residue clamp `U_max^{clamp} = 4.0` nats (final `|Delta U| <= lambda_AF U_max`).
pub const DEFAULT_AF_CLAMP_NATS: f64 = 4.0;

/// Default `15N-NOESY` contact-bonus weight (`P2-3`, matching `DEFAULT_AF_WEIGHT`).
pub const DEFAULT_NOE_WEIGHT: f64 = 0.65;

/// Degrees of freedom `nu = 4` for the heavy-tailed Student-`t` 3D shift likelihood.
pub const STUDENT_T4_DOF: f64 = 4.0;

/// Normalization constant `3 / 8 = 0.375` of the standard Student-`t(nu = 4)` density
/// `f_{t4}(z) = (3 / 8) * (1 + z^2 / 4)^(-5/2)`.
pub const STUDENT_T4_NORM_CONST: f64 = 0.375;

/// Shared per-observation log-ratio floor (identical to the sequence-only `llr` floor).
pub const AF_LLR_FLOOR_NATS: f64 = -12.0;

/// `[dev-set]` `T21.2` leave-one-protein-out RMSE scale floor `[H, N, CA, CB, C']` in ppm
/// (`21.W2:P1-2`, `P2-2`, `21F:B3`).
///
/// Grounded in `T21.2` **dev-set** leave-one-protein-out RMSEs (`H^N 0.505`, `N 2.804`,
/// `CA 0.985`, `CB 1.140`, `C' 1.009` ppm; dev-set evidence per `21.W1:P1-3`), rounded up.
/// Floors `T21.2`'s per-residue predictive uncertainty `sigma_hat = pred_b.predicted_sigma_ppm`
/// so `sigma_AF` never shrinks below either `sigma_hat` or this `[dev-set]` floor and widens
/// monotonically toward `max(sigma_base, sd_BMRB)` as `pLDDT` drops:
/// `sigma_base = max(sigma_hat, AF_SIGMA_DEV_SET_PPM[a])`,
/// `sigma_AF^2 = w^2 sigma_base^2 + (1 - w^2) max(sigma_base, sd_BMRB)^2`.
pub const AF_SIGMA_DEV_SET_PPM: [f64; 5] = [0.52, 2.80, 1.05, 1.20, 1.05];

/// Backwards-compatible alias for [`AF_SIGMA_DEV_SET_PPM`] (`[dev-set]` `T21.2` LOPO RMSE floor;
/// NOT an independent held-out out-of-sample benchmark).
pub const AF_SIGMA_OOS_PPM: [f64; 5] = AF_SIGMA_DEV_SET_PPM;

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

/// Hard upper bound (`+2.50` nats, scaled by `noe_weight`) on the summed `15N-NOESY` bonus
/// incident on any single residue (`P2-3`). Enforced symmetrically on each edge `(j, k)` via
/// `min(1, site_cap / max(M_j, M_k))` so MCMC `local_subset_bonus` telescopes identically.
pub const AF_NOESY_MAX_SITE_BONUS_NATS: f64 = 2.50;

/// Generic sequential `d_NN` (`3.5 A`) assumed by the structure-free sequential `15N-NOESY`
/// baseline ([`AfAssignPriors::sequence_only_noesy`]; `21.W2:P0-2` ablation control).
pub const STRUCTURE_FREE_SEQ_DNN_ANGSTROM: f64 = 3.5;

/// Exact integer carrier-center ppm constants `[H, N, CA, CB, C]` subtracted in `f64` on the host
/// before `f32` WebGPU AF buffer upload so `compute_unary_af` achieves `max |U_cpu - U_gpu| < 1e-5`.
pub const AF_GPU_CARRIER_CENTERS_PPM: [f64; 5] = [8.0, 120.0, 56.0, 44.0, 176.0];

/// Unnormalized heavy-tailed Student-`t(nu = 4)` log-kernel `ell_{t4}(z) = -(5/2) ln(1 + z^2 / 4)`
/// (`docs/specs/21-alphafold-integrative-nmr.md` section 4.2; Lange, Little & Taylor 1989).
#[inline]
pub fn ell_student_t4(z: f64) -> f64 {
    -2.5 * (1.0 + 0.25 * z * z).ln()
}

/// Normalized Student-`t(nu = 4)` log-density of `v` with location `mu` and scale `sigma`.
#[inline]
pub fn ln_student_t4_density(v: f64, mu: f64, sigma: f64) -> f64 {
    let z = (v - mu) / sigma;
    STUDENT_T4_NORM_CONST.ln() - sigma.ln() + ell_student_t4(z)
}

#[inline]
fn atom_index(atom: &str) -> Option<usize> {
    ATOMS.iter().position(|&a| a == atom)
}

#[inline]
fn is_experimental_method_token(s: &str) -> bool {
    let up = s.to_ascii_uppercase();
    up.contains("X-RAY")
        || up.contains("NMR")
        || up.contains("ELECTRON")
        || up.contains("DIFFRACTION")
        || up.contains("MICROSCOPY")
        || up.contains("CRYSTAL")
        || up.contains("NEUTRON")
}

/// Positive-evidence check that a coordinate text is a predicted model (`21.W2:P1-3`, `21F:B1`).
///
/// Local fail-closed stand-in matching `S2`'s `require_predicted_model()` contract: returns `true`
/// ONLY for non-experimental mmCIF text carrying either:
/// 1. a `_ma_qa_metric` entry whose `name` is explicitly `pLDDT` (case-insensitive, non-global;
///    rejecting `QMEANDisCo` and other non-`pLDDT` metrics) paired with matching
///    `_ma_qa_metric_local` rows (or non-constant finite `_atom_site.B_iso_or_equiv` in `[0, 100]`), or
/// 2. a recognized AFDB/ColabFold header (`data_AF-*`, `_entry.id AF-*`, or `_software.name`
///    naming `AlphaFold`/`ColabFold`/`Boltz`) with non-constant finite `_atom_site.B_iso_or_equiv`
///    values in `[0, 100]` and no conflicting non-`pLDDT` `_ma_qa_metric` declaration.
///
/// Generic ModelArchive blocks (`_ma_protocol_step`, `_ma_data`, `_ma_model_list`), headerless
/// PDBs, `EXPDTA`/`_exptl` experimental files (`1D3Z`, `1UBQ`), `CRYST1` unit cells,
/// `REMARK`/title mentions of AlphaFold, and raw `B <= 100` PDBs all return `false`.
pub fn text_has_predicted_model_evidence(text: &str) -> bool {
    classify_predicted_model_evidence(text).is_some()
}

fn classify_predicted_model_evidence(text: &str) -> Option<&'static str> {
    let lines: Vec<&str> = text.lines().map(str::trim).collect();
    if lines.is_empty() {
        return None;
    }

    // Reject any PDB or mmCIF declaring experimental provenance or crystallographic unit cells.
    for &line in &lines {
        if line.starts_with("EXPDTA") && is_experimental_method_token(line) {
            return None;
        }
        if line.starts_with("CRYST1") && line.len() >= 33 {
            if let (Ok(a), Ok(b), Ok(c)) = (
                line[6..15].trim().parse::<f64>(),
                line[15..24].trim().parse::<f64>(),
                line[24..33].trim().parse::<f64>(),
            ) {
                if a > 2.5 && b > 2.5 && c > 2.5 {
                    return None;
                }
            }
        }
        if let Some(rest) = line.strip_prefix("_exptl.method") {
            if is_experimental_method_token(rest) {
                return None;
            }
        }
    }

    let mut has_mmcif_data = false;
    let mut has_recognized_af_header = false;
    let mut saw_ma_qa_metric = false;
    let mut plddt_metric_ids: Vec<String> = Vec::new();
    let mut has_kv_plddt_metric = false;
    let mut kv_ma_mode_global = false;
    let mut has_matching_local_plddt = false;
    let mut b_min = f64::INFINITY;
    let mut b_max = f64::NEG_INFINITY;
    let mut b_count = 0usize;
    let mut b_out_of_range = false;

    let mut i = 0usize;
    while i < lines.len() {
        let line = lines[i];
        if let Some(rest) = line.strip_prefix("data_") {
            has_mmcif_data = true;
            if rest.trim().to_ascii_uppercase().starts_with("AF-") {
                has_recognized_af_header = true;
            }
        }
        if let Some(rest) = line.strip_prefix("_entry.id") {
            let val = rest.trim().trim_matches(|c| c == '\'' || c == '"');
            if val.to_ascii_uppercase().starts_with("AF-") {
                has_recognized_af_header = true;
            }
        }
        if let Some(rest) = line.strip_prefix("_software.name") {
            let up = rest
                .trim()
                .trim_matches(|c| c == '\'' || c == '"')
                .to_ascii_uppercase();
            if up.contains("ALPHAFOLD") || up.contains("COLABFOLD") || up.contains("BOLTZ") {
                has_recognized_af_header = true;
            }
        }
        if let Some(rest) = line.strip_prefix("_ma_qa_metric.name") {
            saw_ma_qa_metric = true;
            let val = rest.trim().trim_matches(|c| c == '\'' || c == '"');
            if val.to_ascii_lowercase().contains("plddt") {
                has_kv_plddt_metric = true;
            }
        }
        if let Some(rest) = line.strip_prefix("_ma_qa_metric.mode") {
            saw_ma_qa_metric = true;
            let val = rest.trim().trim_matches(|c| c == '\'' || c == '"');
            if val.to_ascii_lowercase().contains("global") {
                kv_ma_mode_global = true;
            }
        }

        if line.eq_ignore_ascii_case("loop_") {
            let mut cols: Vec<&str> = Vec::new();
            let mut j = i + 1;
            while j < lines.len() && lines[j].starts_with('_') {
                cols.push(lines[j].split_whitespace().next().unwrap_or(""));
                j += 1;
            }
            let first_low = cols
                .first()
                .map(|s| s.to_ascii_lowercase())
                .unwrap_or_default();
            if first_low.starts_with("_exptl.") {
                let m_col = cols
                    .iter()
                    .position(|c| c.eq_ignore_ascii_case("_exptl.method"));
                while j < lines.len() {
                    let row = lines[j];
                    if row.is_empty()
                        || row.starts_with('#')
                        || row.starts_with('_')
                        || row.eq_ignore_ascii_case("loop_")
                        || row.eq_ignore_ascii_case("stop_")
                    {
                        break;
                    }
                    if m_col.is_some() && is_experimental_method_token(row) {
                        return None;
                    }
                    j += 1;
                }
                i = j;
                continue;
            } else if first_low.starts_with("_software.") {
                let n_col = cols
                    .iter()
                    .position(|c| c.eq_ignore_ascii_case("_software.name"));
                while j < lines.len() {
                    let row = lines[j];
                    if row.is_empty()
                        || row.starts_with('#')
                        || row.starts_with('_')
                        || row.eq_ignore_ascii_case("loop_")
                        || row.eq_ignore_ascii_case("stop_")
                    {
                        break;
                    }
                    let toks = split_cif_row_tokens(row);
                    if toks.len() == cols.len() {
                        if let Some(nm) = n_col.and_then(|c| toks.get(c)) {
                            let up = nm.to_ascii_uppercase();
                            if up.contains("ALPHAFOLD")
                                || up.contains("COLABFOLD")
                                || up.contains("BOLTZ")
                            {
                                has_recognized_af_header = true;
                            }
                        }
                    }
                    j += 1;
                }
                i = j;
                continue;
            } else if first_low.starts_with("_ma_qa_metric.") {
                saw_ma_qa_metric = true;
                let id_col = cols
                    .iter()
                    .position(|c| c.eq_ignore_ascii_case("_ma_qa_metric.id"));
                let name_col = cols
                    .iter()
                    .position(|c| c.eq_ignore_ascii_case("_ma_qa_metric.name"));
                let mode_col = cols
                    .iter()
                    .position(|c| c.eq_ignore_ascii_case("_ma_qa_metric.mode"));
                while j < lines.len() {
                    let row = lines[j];
                    if row.is_empty()
                        || row.starts_with('#')
                        || row.starts_with('_')
                        || row.eq_ignore_ascii_case("loop_")
                        || row.eq_ignore_ascii_case("stop_")
                    {
                        break;
                    }
                    let toks = split_cif_row_tokens(row);
                    if toks.len() == cols.len() {
                        let name_ok = name_col
                            .and_then(|c| toks.get(c))
                            .is_some_and(|t| t.to_ascii_lowercase().contains("plddt"));
                        let mode_ok = mode_col
                            .and_then(|c| toks.get(c))
                            .is_none_or(|t| !t.to_ascii_lowercase().contains("global"));
                        if name_ok && mode_ok {
                            if let Some(ic) = id_col.and_then(|c| toks.get(c)) {
                                plddt_metric_ids.push((*ic).to_string());
                            } else {
                                has_kv_plddt_metric = true;
                            }
                        }
                    }
                    j += 1;
                }
                i = j;
                continue;
            } else if first_low.starts_with("_ma_qa_metric_local.") {
                let mid_col = cols
                    .iter()
                    .position(|c| c.eq_ignore_ascii_case("_ma_qa_metric_local.metric_id"));
                let val_col = cols
                    .iter()
                    .position(|c| c.eq_ignore_ascii_case("_ma_qa_metric_local.metric_value"));
                if val_col.is_some() {
                    while j < lines.len() {
                        let row = lines[j];
                        if row.is_empty()
                            || row.starts_with('#')
                            || row.starts_with('_')
                            || row.eq_ignore_ascii_case("loop_")
                            || row.eq_ignore_ascii_case("stop_")
                        {
                            break;
                        }
                        let toks = split_cif_row_tokens(row);
                        if toks.len() == cols.len() {
                            let kv_local_plddt = has_kv_plddt_metric && !kv_ma_mode_global;
                            let id_matches = match mid_col {
                                Some(mc) => {
                                    (!plddt_metric_ids.is_empty()
                                        && toks.get(mc).is_some_and(|id| {
                                            plddt_metric_ids.iter().any(|p| p == id)
                                        }))
                                        || (plddt_metric_ids.is_empty() && kv_local_plddt)
                                }
                                None => kv_local_plddt || !plddt_metric_ids.is_empty(),
                            };
                            let val_ok = val_col
                                .and_then(|vc| toks.get(vc))
                                .and_then(|s| s.parse::<f64>().ok())
                                .is_some_and(|v| v.is_finite() && (0.0..=100.001).contains(&v));
                            if id_matches && val_ok {
                                has_matching_local_plddt = true;
                            }
                        }
                        j += 1;
                    }
                }
                i = j;
                continue;
            } else if first_low.starts_with("_atom_site.") {
                let b_col = cols
                    .iter()
                    .position(|c| c.eq_ignore_ascii_case("_atom_site.B_iso_or_equiv"));
                if let Some(bc) = b_col {
                    while j < lines.len() {
                        let row = lines[j];
                        if row.is_empty()
                            || row.starts_with('#')
                            || row.starts_with('_')
                            || row.eq_ignore_ascii_case("loop_")
                            || row.eq_ignore_ascii_case("stop_")
                        {
                            break;
                        }
                        let toks = split_cif_row_tokens(row);
                        if toks.len() == cols.len() {
                            if let Some(bs) = toks.get(bc) {
                                if let Ok(b) = bs.parse::<f64>() {
                                    if !b.is_finite() || !(0.0..=100.001).contains(&b) {
                                        b_out_of_range = true;
                                    } else {
                                        b_min = b_min.min(b);
                                        b_max = b_max.max(b);
                                        b_count += 1;
                                    }
                                }
                            }
                        }
                        j += 1;
                    }
                }
                i = j;
                continue;
            }
        }
        i += 1;
    }

    if !has_mmcif_data {
        return None;
    }
    if has_matching_local_plddt {
        return Some("mmCIF _ma_qa_metric (name=pLDDT) + _ma_qa_metric_local values");
    }
    let has_valid_bfactor = !b_out_of_range && b_count >= 2 && (b_max - b_min) > 1.0e-3;
    let has_local_plddt_def =
        !plddt_metric_ids.is_empty() || (has_kv_plddt_metric && !kv_ma_mode_global);
    if has_valid_bfactor && has_local_plddt_def {
        return Some("mmCIF _ma_qa_metric (name=pLDDT) + non-constant B_iso_or_equiv pLDDT");
    }
    if has_valid_bfactor && has_recognized_af_header && !saw_ma_qa_metric {
        return Some("mmCIF recognized AFDB/ColabFold header + non-constant B_iso_or_equiv pLDDT");
    }
    None
}

fn split_cif_row_tokens(row: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let bytes = row.as_bytes();
    let n = bytes.len();
    let mut i = 0usize;
    while i < n {
        while i < n && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= n {
            break;
        }
        if bytes[i] == b'\'' || bytes[i] == b'"' {
            let q = bytes[i];
            let start = i + 1;
            i += 1;
            while i < n && !(bytes[i] == q && (i + 1 == n || bytes[i + 1].is_ascii_whitespace())) {
                i += 1;
            }
            out.push(&row[start..i]);
            if i < n {
                i += 1;
            }
        } else {
            let start = i;
            while i < n && !bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            out.push(&row[start..i]);
        }
    }
    out
}

/// Fail-closed model-provenance gate on a parsed [`StructureModel`] (`21.W2:P1-3`), delegating
/// directly to [`crate::struct_shifts::require_predicted_structure_model`] when `!treat_as_model`
/// and enforcing non-constant/non-zero `pLDDT` on predicted models.
pub fn af_require_predicted_model(
    model: &StructureModel,
    treat_as_model: bool,
) -> Result<(), String> {
    if treat_as_model {
        return Ok(());
    }
    crate::struct_shifts::require_predicted_structure_model(model, false)?;
    if model.provenance == StructCoordinateProvenance::Computational
        && !model.residues.is_empty()
        && model.residues.iter().all(|r| r.plddt.is_some())
    {
        let mut p_min = f64::INFINITY;
        let mut p_max = f64::NEG_INFINITY;
        for r in &model.residues {
            let p = r.plddt.unwrap_or(0.0);
            if !p.is_finite() || !(0.0..=100.001).contains(&p) {
                return Err("model contains non-finite or out-of-range pLDDT".to_string());
            }
            p_min = p_min.min(p);
            p_max = p_max.max(p);
        }
        // Reject constant all-0 or all-50 (rescaled 0.5) dummy B-factors
        if model.residues.len() >= 2 && p_max <= 1e-6 {
            return Err(
                "--af-structure refused: all-zero pLDDT / B-factor column is not a valid predicted model; pass --treat-as-model to opt in explicitly"
                    .to_string(),
            );
        }
        return Ok(());
    }
    Err(format!(
        "--af-structure requires positive evidence of a predicted model (ModelArchive/AFDB mmCIF \
         `_ma_qa_metric_local` values for a metric named pLDDT); got provenance={} with {}/{} \
         residues carrying pLDDT. Experimental (X-ray / solution NMR / EM) or headerless \
         coordinates are refused; pass --treat-as-model to use them explicitly as a \
         native-structure oracle",
        model.provenance.as_str(),
        model.residues.iter().filter(|r| r.plddt.is_some()).count(),
        model.residues.len()
    ))
}

const fn default_noe_weight() -> f64 {
    DEFAULT_NOE_WEIGHT
}

/// Configuration for opt-in 3D-prior-assisted backbone and `PAE`-gated `15N-NOESY` assignment.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AfAssignConfig {
    /// AF replacement weight `lambda_AF in [0, 1]` (`omega_r = lambda_AF w_AF(pLDDT_r)`; default 0.65).
    pub af_weight: f64,
    /// Per-residue clamp `U_max^{clamp}` in nats (`|Delta U| <= lambda_AF U_max`; default 4.0).
    pub af_clamp_nats: f64,
    /// Weight on `PAE`-gated `15N-NOESY` contact bonuses (`[0, 1]`, default 0.65, `P2-3`).
    #[serde(default = "default_noe_weight")]
    pub noe_weight: f64,
    /// Explicit opt-in to treat coordinates without predicted-model evidence as a model
    /// (`--treat-as-model`; e.g. a native-structure oracle). Default `false` (fail closed).
    #[serde(default)]
    pub treat_as_model: bool,
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
            noe_weight: DEFAULT_NOE_WEIGHT,
            treat_as_model: false,
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
        if !self.af_weight.is_finite() || !(0.0..=1.0).contains(&self.af_weight) {
            return Err(format!(
                "invalid af_weight {}: expected finite replacement weight in [0.0, 1.0]",
                self.af_weight
            ));
        }
        if !self.noe_weight.is_finite() || !(0.0..=1.0).contains(&self.noe_weight) {
            return Err(format!(
                "invalid noe_weight {}: expected finite value in [0.0, 1.0]",
                self.noe_weight
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

    /// Final per-cell clamp `lambda_AF * U_max` in nats.
    #[inline]
    pub fn delta_u_ceiling_nats(&self) -> f64 {
        self.af_weight * self.af_clamp_nats
    }
}

/// Global chemical-shift referencing offsets `(obs_ppm - pred_ppm)` across `[H, N, CA, CB, C']`.
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

    #[inline]
    pub const fn from_shift_offsets(o: &ShiftOffsets) -> Self {
        Self {
            h: o.h,
            n: o.n,
            ca: o.ca,
            cb: o.cb,
            c: o.c,
        }
    }
}

/// Per-residue, per-atom (`H=0, N=1, CA=2, CB=3, C=4`) AF-conditioned shift prior.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AfResidueAtomPrior {
    pub present: bool,
    /// `T21.2` predicted shift (DSS frame; `pLDDT`-blended toward random coil by `T21.2`).
    pub mu_af_ppm: f64,
    /// Student-`t4` scale `sigma_AF >= AF_SIGMA_OOS_PPM[a]`, never shrinking as `pLDDT` drops.
    pub sigma_af_ppm: f64,
    /// Replacement weight `omega_r = clamp(lambda_AF w_AF(pLDDT_r), 0, 1)` of this residue.
    pub omega: f64,
}

impl AfResidueAtomPrior {
    /// `ell_AF,a(v) = max(ln t4(v | mu_AF + off, sigma_AF) - bg_a(v), -12)` against the shared
    /// pooled background `bg` (`Priors::background(atom, v)` of the sequence-only priors).
    #[inline]
    pub fn af_llr(&self, v: f64, off_ppm: f64, bg: f64) -> f64 {
        (ln_student_t4_density(v, self.mu_af_ppm + off_ppm, self.sigma_af_ppm) - bg)
            .max(AF_LLR_FLOOR_NATS)
    }
}

/// Telemetry of one `Delta U^{AF}` application (`21.W2:P1-1` audit).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AfReplacementStats {
    pub cells_scored: usize,
    pub cells_nonzero: usize,
    pub cells_at_clamp: usize,
    pub max_abs_delta_u: f64,
    pub ceiling_nats: f64,
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
    /// Precomputed distance-scaled ceiling before multiplying by `noe_weight` and `evid_NOE`.
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
    /// Per-residue smooth `pLDDT`-sigmoid weight `w_AF(pLDDT_k) in [0.0, 1.0]` aligned to `sequence`.
    pub w_af_per_residue: Vec<f64>,
    /// Flat `[k * 5 + atom_idx]` AF shift priors (`H=0, N=1, CA=2, CB=3, C=4`).
    pub atom_priors: Vec<AfResidueAtomPrior>,
    /// Admitted confident amide-amide contacts `(j < k)` in `C_AF`.
    pub admitted_contacts: Vec<AfAdmittedAmideContact>,
    /// Optional unassigned `3D 15N-edited NOESY` crosspeaks.
    pub noesy_15n_peaks: Vec<Noesy15nPeak>,
    /// Contact gating telemetry (`admitted`, `rejected_high_pae_contacts`, `rejected_low_plddt_contacts`).
    pub contact_summary: AfNoesyContactSummary,
    /// Full `T21.2` shift prediction report carrying `exclusion_list_sha256` & `max_retained_seq_identity`.
    pub shift_report: StructShiftPredictionReport,
    /// Human-readable record of the predicted-model evidence that admitted this structure.
    #[serde(default)]
    pub model_evidence: String,
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

impl AfAssignPriors {
    /// Build `AfAssignPriors` from a parsed [`StructureModel`], target `sequence`, optional
    /// [`PaeMatrix`], and optional `3D 15N-NOESY` crosspeaks, enforcing the 4-layer
    /// `HomologFirewall` (`max_seq_identity <= 0.25`, `TM-score < 0.50`) and the fail-closed
    /// predicted-model gate ([`af_require_predicted_model`], `21.W2:P1-3`).
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

        let mut structure_work = structure.clone();
        if let Some(pae) = pae_opt {
            if pae.n_residues != structure_work.residues.len() && pae.n_residues != k_len {
                return Err(format!(
                    "PAE matrix dimension N = {} matches neither structure residue count ({}) nor target sequence length ({k_len})",
                    pae.n_residues,
                    structure_work.residues.len()
                ));
            }
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
        if let Some(ov) = cfg.struct_shift_cfg.override_uniform_plddt {
            for r in &mut structure_work.residues {
                r.plddt = Some(ov.clamp(0.0, 100.0));
            }
        }
        af_require_predicted_model(&structure_work, cfg.treat_as_model)?;

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
            // Fail closed unless explicit `treat_as_model` opted in to unannotated coordinates.
            let (p, w_af) = match structure_work.residues[si].plddt {
                Some(raw_p) => {
                    let pc = raw_p.clamp(0.0, 100.0);
                    (pc, af_plddt_confidence_weight(pc))
                }
                None if cfg.treat_as_model => (100.0, 1.0),
                None => {
                    return Err(format!(
                        "aligned residue {} {} lacks pLDDT; pass --treat-as-model to opt in explicitly",
                        structure_work.residues[si].res_name, structure_work.residues[si].seq_id
                    ));
                }
            };
            plddt_per_residue[k] = p;
            w_af_per_residue[k] = w_af;
            let omega = (cfg.af_weight * w_af).clamp(0.0, 1.0);

            let pred_r = &shift_report.residues[si];
            let rname = &norm_seq[k];
            for (atom_idx, &atom_name) in ATOMS.iter().enumerate() {
                if (rname == "PRO" && atom_name == "H") || (rname == "GLY" && atom_name == "CB") {
                    continue;
                }
                let Some((_, sd_bmrb)) = bmrb.get(rname, atom_name) else {
                    continue;
                };
                let Some(pred_b) = pred_r.backbone.get(atom_name) else {
                    continue;
                };
                if !pred_b.predicted_ppm.is_finite() {
                    continue;
                }
                let s_floor = AF_SIGMA_DEV_SET_PPM[atom_idx];
                let s_base = if pred_b.predicted_sigma_ppm.is_finite() {
                    pred_b.predicted_sigma_ppm.max(s_floor)
                } else {
                    s_floor
                };
                let s_wide = s_base.max(sd_bmrb);
                let w2 = w_af * w_af;
                let sigma_af = (w2 * s_base * s_base + (1.0 - w2) * s_wide * s_wide).sqrt();
                atom_priors[k * 5 + atom_idx] = AfResidueAtomPrior {
                    present: true,
                    mu_af_ppm: pred_b.predicted_ppm,
                    sigma_af_ppm: sigma_af,
                    omega,
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

        let model_evidence = if cfg.treat_as_model {
            "explicit --treat-as-model opt-in (coordinates NOT verified as a predicted model; native-structure oracle if experimental)".to_string()
        } else {
            format!(
                "provenance={} with per-residue pLDDT on all {} residues",
                structure_work.provenance.as_str(),
                structure_work.residues.len()
            )
        };

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
            model_evidence,
        })
    }

    /// Structure-free `15N-NOESY` baseline: zero AF shift priors (`omega = 0` everywhere) and
    /// one generic sequential `(k - 1, k)` amide-amide contact per non-Proline pair at
    /// [`STRUCTURE_FREE_SEQ_DNN_ANGSTROM`]. Gives the sequence-only baseline the exact same
    /// NOESY peak list without any 3D model (`21.W2:P0-2` ablation control; original method).
    pub fn sequence_only_noesy(
        sequence: &[String],
        noesy_15n_peaks: Vec<Noesy15nPeak>,
        cfg: &AfAssignConfig,
    ) -> Result<Self, String> {
        cfg.validate()?;
        let norm_seq = normalize_seq_tokens(sequence);
        let k_len = norm_seq.len();
        if k_len == 0 {
            return Err("empty target sequence".to_string());
        }
        let d = STRUCTURE_FREE_SEQ_DNN_ANGSTROM;
        let scale = (1.85 * (3.0 / d).powi(6)).clamp(0.0, AF_NOESY_MAX_SEQ_BONUS_NATS);
        let mut admitted_contacts = Vec::new();
        for k in 2..k_len {
            if norm_seq[k] == "PRO" || norm_seq[k - 1] == "PRO" {
                continue;
            }
            admitted_contacts.push(AfAdmittedAmideContact {
                seq_j: k - 1,
                seq_k: k,
                dist_hn_angstrom: d,
                symmetrized_pae_angstrom: None,
                plddt_j: 0.0,
                plddt_k: 0.0,
                max_distance_scale_nats: scale,
                cap_nats: AF_NOESY_MAX_SEQ_BONUS_NATS,
                is_sequential: true,
            });
        }
        let summary = AfNoesyContactSummary {
            noesy_peak_count: noesy_15n_peaks.len(),
            admitted_contacts_total: admitted_contacts.len(),
            admitted_sequential_contacts: admitted_contacts.len(),
            ..Default::default()
        };
        let mut cfg_eff = cfg.clone();
        cfg_eff.af_weight = 0.0;
        Ok(Self {
            cfg: cfg_eff,
            seq_ids: (1..=(k_len as i32)).collect(),
            seq_to_struct: vec![None; k_len],
            plddt_per_residue: vec![0.0; k_len],
            w_af_per_residue: vec![0.0; k_len],
            atom_priors: vec![AfResidueAtomPrior::default(); k_len * 5],
            admitted_contacts,
            noesy_15n_peaks,
            contact_summary: summary,
            shift_report: StructShiftPredictionReport {
                residues: Vec::new(),
                sample_ph: cfg.struct_shift_cfg.sample_ph,
                temperature_kelvin: cfg.struct_shift_cfg.temperature_kelvin,
                include_methyl_ilvmat: false,
                provenance: StructCoordinateProvenance::Unknown,
                retained_template_count: 0,
                excluded_template_ids: Vec::new(),
                excluded_details: Vec::new(),
                exclusion_list_sha256: String::new(),
                max_retained_seq_identity: 0.0,
                corpus_sha256: String::new(),
                l4_short_templates_skipped: 0,
            },
            sequence: norm_seq,
            model_evidence: "structure-free sequential NOESY baseline (no 3D model)".to_string(),
        })
    }

    /// Parse coordinate, optional `PAE` JSON, and optional `3D 15N-NOESY` files from disk and build
    /// [`AfAssignPriors`]. Fails closed unless the coordinate text carries positive predicted-model
    /// evidence ([`text_has_predicted_model_evidence`]) or `cfg.treat_as_model` is explicitly set
    /// (`21.W2:P1-3`).
    pub fn from_files(
        structure_path: impl AsRef<Path>,
        sequence: &[String],
        seq_ids_opt: Option<&[i32]>,
        pae_path_opt: Option<&Path>,
        noesy_15n_path_opt: Option<&Path>,
        firewall_query: &HomologFirewallQuery,
        cfg: &AfAssignConfig,
    ) -> Result<Self, String> {
        let p = structure_path.as_ref();
        let meta = fs::metadata(p)
            .map_err(|e| format!("failed to stat structure file '{}': {e}", p.display()))?;
        if meta.len() > MAX_STRUCTURE_FILE_BYTES as u64 {
            return Err(format!(
                "structure file '{}' exceeds 64 MiB limit",
                p.display()
            ));
        }
        let text = fs::read_to_string(p)
            .map_err(|e| format!("failed to read structure file '{}': {e}", p.display()))?;
        let fmt_hint = p.extension().and_then(|e| e.to_str()).map(str::to_string);
        let noesy_peaks = match noesy_15n_path_opt {
            Some(np) => parse_noesy_15n_peaks_file(np)?,
            None => Vec::new(),
        };
        let pae_text = match pae_path_opt {
            Some(pp) => {
                let pm = fs::metadata(pp)
                    .map_err(|e| format!("failed to stat PAE file '{}': {e}", pp.display()))?;
                if pm.len() > 256 * 1024 * 1024 {
                    return Err(format!("PAE file '{}' exceeds 256 MiB limit", pp.display()));
                }
                Some(
                    fs::read_to_string(pp)
                        .map_err(|e| format!("failed to read PAE file '{}': {e}", pp.display()))?,
                )
            }
            None => None,
        };
        Self::from_strs_with_peaks(
            &text,
            fmt_hint.as_deref(),
            sequence,
            seq_ids_opt,
            pae_text.as_deref(),
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
        let noesy_peaks = match noesy_15n_text_opt {
            Some(txt) => parse_noesy_15n_peaks_str(txt)?,
            None => Vec::new(),
        };
        Self::from_strs_with_peaks(
            structure_text,
            format_hint,
            sequence,
            seq_ids_opt,
            pae_json_opt,
            noesy_peaks,
            firewall_query,
            cfg,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn from_strs_with_peaks(
        structure_text: &str,
        format_hint: Option<&str>,
        sequence: &[String],
        seq_ids_opt: Option<&[i32]>,
        pae_json_opt: Option<&str>,
        noesy_peaks: Vec<Noesy15nPeak>,
        firewall_query: &HomologFirewallQuery,
        cfg: &AfAssignConfig,
    ) -> Result<Self, String> {
        let evidence_label = classify_predicted_model_evidence(structure_text);
        if !cfg.treat_as_model {
            let af_check = crate::af_concordance::parse_af_structure_model_str(
                structure_text,
                &crate::af_concordance::AfIngestOptions {
                    force_bfactor_plddt: cfg.struct_shift_cfg.force_bfactor_plddt,
                    target_chain_id: None,
                },
            )?;
            crate::af_concordance::require_predicted_model(&af_check, false)?;
            if evidence_label.is_none() {
                return Err(
                    "--af-structure refused: no positive predicted-model evidence (an mmCIF \
                     `_ma_qa_metric` named pLDDT with `_ma_qa_metric_local` values, or recognized \
                     AFDB/ColabFold header with non-constant pLDDT). Generic ModelArchive blocks, \
                     headerless, experimental (X-ray / solution NMR / EM), and REMARK-only files \
                     are not treated as models; pass --treat-as-model to opt in explicitly"
                        .to_string(),
                );
            }
        }
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
        let mut out = Self::from_structure_model(
            &model,
            sequence,
            seq_ids_opt,
            pae.as_ref(),
            noesy_peaks,
            firewall_query,
            cfg,
        )?;
        if let Some(ev) = evidence_label {
            out.model_evidence = ev.to_string();
        }
        Ok(out)
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

    /// `Delta U^{AF}_{s,k}` (nats) for placing `ss` at `k` under the per-nucleus replacement
    /// algebra (module docs), BEFORE the final `lambda_AF U_max` clamp.
    ///
    /// `seq_priors` must be the sequence-only priors in the experimental frame (offsets already
    /// applied, as used for `U^{seq}`), and `offsets` the very same offsets, applied to the AF
    /// means so both terms live in one unified reference frame.
    pub fn raw_delta_u_cell(
        &self,
        ss: &SpinSystem,
        sequence: &[String],
        k: usize,
        seq_priors: &Priors,
        offsets: &ShiftOffsets,
        score_cfg: &ScoreConfig,
    ) -> f64 {
        self.raw_delta_u_cell_capped(ss, sequence, k, seq_priors, offsets, score_cfg, None)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn raw_delta_u_cell_capped(
        &self,
        ss: &SpinSystem,
        sequence: &[String],
        k: usize,
        seq_priors: &Priors,
        offsets: &ShiftOffsets,
        score_cfg: &ScoreConfig,
        max_cands_cap: Option<usize>,
    ) -> f64 {
        let k_len = self.sequence.len().min(sequence.len());
        if k == 0 || k >= k_len || sequence[k] == "PRO" {
            return 0.0;
        }
        let res = &sequence[k];
        let prev = &sequence[k - 1];
        let cov_k = delta_covalent(sequence, k);
        let cov_km1 = delta_covalent(sequence, k - 1);
        let mut delta = 0.0_f64;
        for ((atom, off), raw_cands) in &ss.cands {
            let Some(a_idx) = atom_index(atom) else {
                continue;
            };
            let (target_k, target_res, cov) = if *off == 0 {
                (k, res, cov_k)
            } else {
                (k - 1, prev, cov_km1)
            };
            let ae = self.atom_prior(target_k, a_idx);
            if !ae.present || ae.omega <= 0.0 {
                continue;
            }
            let slice = match max_cands_cap {
                Some(cap) => &raw_cands[..raw_cands.len().min(cap)],
                None => raw_cands.as_slice(),
            };
            let d_cov = covalent_atom_offset(cov, atom);
            let off_a = offsets.get(atom);
            let mut best_seq = f64::NEG_INFINITY;
            let mut best_pool = f64::NEG_INFINITY;
            for &(v, sup) in slice {
                if sup == 0 {
                    continue;
                }
                let bonus = score_cfg.support_bonus * ((sup as f64) - 1.0);
                let l_seq = seq_priors.llr_shifted(target_res, atom, v, d_cov);
                let l_af = ae.af_llr(v, off_a, seq_priors.background(atom, v));
                let seq_c = l_seq + bonus;
                let pool_c = (1.0 - ae.omega) * l_seq + ae.omega * l_af + bonus;
                if seq_c > best_seq {
                    best_seq = seq_c;
                }
                if pool_c > best_pool {
                    best_pool = pool_c;
                }
            }
            if best_seq.is_finite() && best_pool.is_finite() {
                // `- ln(n_pos)` is identical in both terms and cancels identically (`21.W2:P1-1`).
                delta += best_pool - best_seq;
            }
        }
        if score_cfg.use_hn {
            let ae_n = self.atom_prior(k, 1);
            if ae_n.present && ae_n.omega > 0.0 {
                let l_seq = seq_priors.llr_shifted(res, "N", ss.n, cov_k.0);
                let l_af = ae_n.af_llr(ss.n, offsets.n, seq_priors.background("N", ss.n));
                delta += ae_n.omega * (l_af - l_seq);
            }
            let ae_h = self.atom_prior(k, 0);
            if ae_h.present && ae_h.omega > 0.0 {
                let l_seq = seq_priors.llr(res, "H", ss.h);
                let l_af = ae_h.af_llr(ss.h, offsets.h, seq_priors.background("H", ss.h));
                delta += ae_h.omega * (l_af - l_seq);
            }
        }
        delta
    }

    /// Add the clamped per-nucleus replacement `Delta U^{AF}` onto every admissible
    /// (`> FORBID / 2`) cell of a sequence-only `U (S x K)` matrix (before user locks).
    pub fn apply_replacement_to_unary(
        &self,
        u: &mut Matrix2D,
        systems: &[SpinSystem],
        sequence: &[String],
        seq_priors: &Priors,
        offsets: &ShiftOffsets,
        score_cfg: &ScoreConfig,
    ) -> AfReplacementStats {
        let ceiling = self.cfg.delta_u_ceiling_nats();
        let mut stats = AfReplacementStats {
            ceiling_nats: ceiling,
            ..Default::default()
        };
        let s_len = u.rows.min(systems.len());
        let k_len = u.cols.min(self.sequence.len()).min(sequence.len());
        if s_len == 0 || k_len <= 1 || ceiling <= 0.0 {
            return stats;
        }
        let max_cands = systems
            .iter()
            .flat_map(|ss| ss.cands.values().map(|c| c.len()))
            .max()
            .unwrap_or(1)
            .max(1);
        for k in 1..k_len {
            if sequence[k] == "PRO" {
                continue;
            }
            for (s, ss) in systems.iter().enumerate().take(s_len) {
                let base = u.get(s, k);
                if base <= FORBID * 0.5 {
                    continue;
                }
                stats.cells_scored += 1;
                let raw = self.raw_delta_u_cell_capped(
                    ss,
                    sequence,
                    k,
                    seq_priors,
                    offsets,
                    score_cfg,
                    Some(max_cands),
                );
                let du = raw.clamp(-ceiling, ceiling);
                if du != 0.0 {
                    stats.cells_nonzero += 1;
                    if raw.abs() >= ceiling {
                        stats.cells_at_clamp += 1;
                    }
                    stats.max_abs_delta_u = stats.max_abs_delta_u.max(du.abs());
                    u.set(s, k, base + du);
                }
            }
        }
        stats
    }

    /// Build a fast `O(1)`-per-move pairwise `15N-NOESY` lookup table for `systems` (`None` when
    /// either `noesy_15n_peaks` or `admitted_contacts` is empty, or `noe_weight == 0`).
    pub fn build_noesy_pairwise_table(
        &self,
        systems: &[SpinSystem],
    ) -> Option<AfNoesyPairwiseTable> {
        AfNoesyPairwiseTable::build(systems, self)
    }
}

// ============================================================================
// 1. Unlabelled Grid Profile-Likelihood + Outlier-Resistant Huber IRLS Offsets
// ============================================================================

#[inline]
fn spin_system_canonical_rank_key(ss: &SpinSystem) -> (bool, usize, usize, u64, u64, u64) {
    let mut slots = 0usize;
    let mut sup_sum = 0usize;
    let mut first_ca = 0.0_f64;
    let mut has_ca = false;
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
            if atom == "CA" {
                has_ca = true;
                if *off == 0 {
                    first_ca = bv;
                }
            }
        }
    }
    (
        has_ca,
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
fn is_nh2_sidechain_pair_member(idx: usize, systems: &[SpinSystem]) -> bool {
    let ss = &systems[idx];
    let has_ca0 = best_positive_slot_value(ss, "CA", 0).is_some();
    if has_ca0 {
        return false;
    }
    for (j, other) in systems.iter().enumerate() {
        if j == idx {
            continue;
        }
        let other_ca0 = best_positive_slot_value(other, "CA", 0).is_some();
        if other_ca0 {
            continue;
        }
        let dn = (ss.n - other.n).abs();
        let dh = (ss.h - other.h).abs();
        if dn <= 0.25 && (0.15..=1.60).contains(&dh) {
            return true;
        }
    }
    false
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
    const OUTLIER_EPS: f64 = 0.20;
    let fg_scale = (1.0 - OUTLIER_EPS) / (preds_sorted.len() as f64);
    let bg_const = OUTLIER_EPS / spectral_range_ppm;

    let n_steps = ((2.0 * max_search_ppm) / grid_step_ppm).round() as usize + 1;
    let mut scores = vec![0.0_f64; n_steps];
    let mut best_idx = n_steps / 2;
    let mut best_ll = f64::NEG_INFINITY;

    // Stage 1: Unlabelled Student-t4 + uniform outlier marginal mixture profile-likelihood
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

    // Stage 2: Inlier-gated Huber IRLS refinement (`12` iterations)
    for _ in 0..12 {
        let mut num = 0.0_f64;
        let mut den = 0.0_f64;
        for &x in obs_sorted {
            let mut fg_sum = 0.0_f64;
            for &(mu, sig) in preds_sorted {
                let z = (x - (mu + d_est)) / sig;
                fg_sum += fg_scale * STUDENT_T4_NORM_CONST * (1.0 + 0.25 * z * z).powf(-2.5) / sig;
            }
            let tot_p = (fg_sum + bg_const).max(1e-30);
            let p_inlier = fg_sum / tot_p;
            let inv_tot = p_inlier / tot_p;
            for &(mu, sig) in preds_sorted {
                let raw_diff = x - mu;
                let resid = (raw_diff - d_est).abs();
                if resid > 3.0 * sig {
                    continue;
                }
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
/// via outlier-gated profile-likelihood + Huber IRLS (`0` ground-truth labels; exact `< 1e-12`
/// permutation invariance under arbitrary permutations of both `systems` and `predictions`, and
/// hardened against missing roots + sidechain `NH2`-like extras per `21.W2:P2-1`).
pub fn estimate_af_shift_offsets_from_predictions(
    systems: &[SpinSystem],
    predictions: &[ResidueShiftPrediction],
) -> AfShiftOffsets {
    if systems.is_empty() || predictions.is_empty() {
        return AfShiftOffsets::default();
    }

    let non_nh2: Vec<&SpinSystem> = systems
        .iter()
        .enumerate()
        .filter_map(|(i, ss)| (!is_nh2_sidechain_pair_member(i, systems)).then_some(ss))
        .collect();
    let pool: &[&SpinSystem] = if non_nh2.len() >= 6 {
        &non_nh2
    } else {
        // Fallback only on tiny synthetic inputs
        &systems.iter().collect::<Vec<_>>()
    };

    let ca_backed: Vec<&SpinSystem> = pool
        .iter()
        .copied()
        .filter(|ss| {
            let k = spin_system_canonical_rank_key(ss);
            k.0 && k.2 >= 2
        })
        .collect();
    let mut ranked: Vec<&SpinSystem> = if ca_backed.len() >= 8 {
        ca_backed
    } else {
        let multi: Vec<&SpinSystem> = pool
            .iter()
            .copied()
            .filter(|ss| spin_system_canonical_rank_key(ss).2 >= 2)
            .collect();
        if multi.len() >= 6 {
            multi
        } else {
            let carb: Vec<&SpinSystem> = pool
                .iter()
                .copied()
                .filter(|ss| ss.has_observed_carbons())
                .collect();
            if carb.is_empty() {
                pool.to_vec()
            } else {
                carb
            }
        }
    };

    ranked.sort_by(|a, b| {
        let ka = spin_system_canonical_rank_key(a);
        let kb = spin_system_canonical_rank_key(b);
        kb.0.cmp(&ka.0)
            .then_with(|| kb.2.cmp(&ka.2))
            .then_with(|| kb.1.cmp(&ka.1))
            .then_with(|| a.h.total_cmp(&b.h))
            .then_with(|| a.n.total_cmp(&b.n))
            .then_with(|| ka.5.cmp(&kb.5))
    });
    let max_selected = predictions.len().max(16);
    if ranked.len() > max_selected {
        ranked.truncate(max_selected);
    }

    let collect_preds = |atom: &str,
                         floor_ppm: f64,
                         is_pro_excluded: bool,
                         is_gly_excluded: bool|
     -> Vec<(f64, f64)> {
        let mut v: Vec<(f64, f64)> = predictions
            .iter()
            .filter(|r| {
                !(is_pro_excluded && r.res_name == "PRO")
                    && !(is_gly_excluded && r.res_name == "GLY")
            })
            .filter_map(|r| {
                let b = r.backbone.get(atom)?;
                if b.predicted_ppm.is_finite() && b.predicted_sigma_ppm.is_finite() {
                    Some((b.predicted_ppm, b.predicted_sigma_ppm.max(floor_ppm)))
                } else {
                    None
                }
            })
            .collect();
        v.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.total_cmp(&b.1)));
        v
    };

    // Prefer intra-residue (`off = 0`, 1 observation per spin system) when well-populated (`>= 8`),
    // falling back to sequential (`off = -1`) on `CBCACONH`/`HNCO`-only sets so residues are not
    // asymmetrically double-counted across missing sequential neighbors.
    let collect_carbon_obs = |atom: &str, lo: f64, hi: f64| -> Vec<f64> {
        let mut obs0 = Vec::new();
        let mut obsm1 = Vec::new();
        for ss in &ranked {
            if let Some(v) = best_positive_slot_value(ss, atom, 0) {
                if (lo..=hi).contains(&v) {
                    obs0.push(v);
                }
            }
            if let Some(v) = best_positive_slot_value(ss, atom, -1) {
                if (lo..=hi).contains(&v) {
                    obsm1.push(v);
                }
            }
        }
        let mut obs = if obs0.len() >= 8 && obs0.len() * 2 >= obsm1.len() {
            obs0
        } else if obsm1.len() >= 8 && obsm1.len() > obs0.len() * 2 {
            obsm1
        } else {
            obs0.extend(obsm1);
            obs0
        };
        obs.sort_by(f64::total_cmp);
        obs
    };

    let mut obs_h: Vec<f64> = ranked
        .iter()
        .map(|ss| ss.h)
        .filter(|v| v.is_finite() && (5.0..=12.5).contains(v))
        .collect();
    obs_h.sort_by(f64::total_cmp);

    let mut obs_n: Vec<f64> = ranked
        .iter()
        .map(|ss| ss.n)
        .filter(|v| v.is_finite() && (98.0..=142.0).contains(v))
        .collect();
    obs_n.sort_by(f64::total_cmp);

    let obs_ca = collect_carbon_obs("CA", 34.0, 76.0);
    let obs_cb = collect_carbon_obs("CB", 8.0, 78.0);
    let obs_c = collect_carbon_obs("C", 160.0, 190.0);

    let preds_h = collect_preds("H", AF_SIGMA_DEV_SET_PPM[0], true, false);
    let preds_n = collect_preds("N", AF_SIGMA_DEV_SET_PPM[1], true, false);
    let preds_ca = collect_preds("CA", AF_SIGMA_DEV_SET_PPM[2], false, false);
    let preds_cb = collect_preds("CB", AF_SIGMA_DEV_SET_PPM[3], false, true);
    let preds_c = collect_preds("C", AF_SIGMA_DEV_SET_PPM[4], false, false);

    let h = fit_single_channel_profile_huber_irls(&obs_h, &preds_h, 2.5, 0.01, 5.0, 0.30);
    let n = fit_single_channel_profile_huber_irls(&obs_n, &preds_n, 12.0, 0.05, 40.0, 1.50);
    let ca = fit_single_channel_profile_huber_irls(&obs_ca, &preds_ca, 12.0, 0.05, 36.0, 0.75);
    let cb = if obs_cb.len() >= 4 {
        fit_single_channel_profile_huber_irls(&obs_cb, &preds_cb, 12.0, 0.05, 64.0, 0.85)
    } else {
        ca
    };
    let c = fit_single_channel_profile_huber_irls(&obs_c, &preds_c, 12.0, 0.05, 24.0, 0.75);

    AfShiftOffsets { h, n, ca, cb, c }
}

/// Estimate global referencing offsets `(dH, dN, dCA, dCB, dC')` between unordered `systems` and
/// unordered 3D shift predictions inside `af_priors` (`0` ground-truth labels; `< 1e-12`
/// permutation-invariant).
pub fn estimate_af_shift_offsets(
    systems: &[SpinSystem],
    af_priors: &AfAssignPriors,
) -> AfShiftOffsets {
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
    eff_scale_nats: f64,
    eff_cap_nats: f64,
    is_sequential: bool,
}

#[derive(Clone, Copy, Debug)]
struct ScaledContactEdge {
    seq_j: usize,
    seq_k: usize,
    eff_scale_nats: f64,
    eff_cap_nats: f64,
}

/// Fast `O(1)`-per-move pairwise scorer for `PAE`-gated `3D 15N-NOESY` sequential `(k - 1, k)` and
/// medium/long-range `|j - k| >= 2` amide-amide contacts, scaled by `noe_weight` and symmetrically
/// capped per residue at `noe_weight * AF_NOESY_MAX_SITE_BONUS_NATS` (`P2-3`).
#[derive(Clone, Debug)]
pub struct AfNoesyPairwiseTable {
    s_len: usize,
    k_len: usize,
    site_cap_nats: f64,
    /// Symmetric `S x S` reciprocity-weighted experimental NOE crosspeak evidence in `[0.0, 1.0]`.
    evid_matrix: Vec<f64>,
    edges: Vec<ScaledContactEdge>,
    adj_by_k: Vec<Vec<AdjNoeContact>>,
    seq_edge_scale: Vec<Option<(f64, f64)>>,
}

impl AfNoesyPairwiseTable {
    /// Precompute `(s_1, s_2)` bidirectional `3D 15N-NOESY` crosspeak reciprocity and map onto
    /// `af_priors.admitted_contacts` (`C_AF`) with symmetric per-site cap and `noe_weight` scaling.
    pub fn build(systems: &[SpinSystem], af_priors: &AfAssignPriors) -> Option<Self> {
        let s_len = systems.len();
        let k_len = af_priors.sequence.len();
        let noe_w = af_priors.cfg.noe_weight;
        if s_len < 2
            || k_len < 2
            || noe_w <= 0.0
            || af_priors.noesy_15n_peaks.is_empty()
            || af_priors.admitted_contacts.is_empty()
        {
            return None;
        }

        let (tol_h, tol_n) = af_priors.cfg.noesy_strip_tol_hn;
        let sig_h = af_priors.cfg.noesy_indirect_sigma_h_ppm.max(1e-4);
        let cut_h = af_priors.cfg.noesy_indirect_cutoff_h_ppm.max(sig_h);

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

        // Local 1H^N degeneracy multiplicity within `cut_h` (`P2-3`): linear `1 / crowd`
        // specificity damping + reciprocal requirement under amide-H overlap.
        let mut h_crowding = vec![1usize; s_len];
        for s in 0..s_len {
            let mut cnt = 1usize;
            for t in 0..s_len {
                if t != s && (systems[t].h - systems[s].h).abs() <= cut_h {
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
                    let crowd = (0.5 * ((h_crowding[s1] + h_crowding[s2]) as f64)).max(1.0);
                    let bidir = (m_fwd * m_rev).sqrt();
                    let unidir = 0.5 * (m_fwd + m_rev);
                    // Suppress one-sided matches when either amide proton is degenerate (`crowd > 1.5`)
                    let raw = if crowd > 1.5 && bidir <= 0.0 {
                        0.0
                    } else {
                        0.85 * bidir + 0.15 * unidir
                    };
                    let evid = (raw / crowd).clamp(0.0, 1.0);
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

        // Pre-compute per-residue total weighted ceiling `M_r` so every undirected edge `(j, k)`
        // is symmetrically damped by `min(1, site_cap / max(M_j, M_k))` (`P2-3`).
        let site_cap = noe_w * AF_NOESY_MAX_SITE_BONUS_NATS;
        let mut m_res = vec![0.0_f64; k_len];
        for c in &af_priors.admitted_contacts {
            if c.seq_j < k_len && c.seq_k < k_len {
                let w0 = (noe_w * c.max_distance_scale_nats).min(noe_w * c.cap_nats);
                m_res[c.seq_j] += w0;
                m_res[c.seq_k] += w0;
            }
        }

        let mut edges = Vec::with_capacity(af_priors.admitted_contacts.len());
        let mut adj_by_k = vec![Vec::new(); k_len];
        let mut seq_edge_scale = vec![None; k_len];
        for c in &af_priors.admitted_contacts {
            if c.seq_j >= k_len || c.seq_k >= k_len {
                continue;
            }
            let w0 = noe_w * c.max_distance_scale_nats;
            let cap0 = noe_w * c.cap_nats;
            let max_m = m_res[c.seq_j].max(m_res[c.seq_k]).max(1e-12);
            let sym_factor = (site_cap / max_m).min(1.0);
            let eff_scale = w0 * sym_factor;
            let eff_cap = (cap0 * sym_factor).min(site_cap);
            let cid = edges.len();
            edges.push(ScaledContactEdge {
                seq_j: c.seq_j,
                seq_k: c.seq_k,
                eff_scale_nats: eff_scale,
                eff_cap_nats: eff_cap,
            });
            adj_by_k[c.seq_j].push(AdjNoeContact {
                contact_id: cid,
                partner_k: c.seq_k,
                eff_scale_nats: eff_scale,
                eff_cap_nats: eff_cap,
                is_sequential: c.is_sequential,
            });
            adj_by_k[c.seq_k].push(AdjNoeContact {
                contact_id: cid,
                partner_k: c.seq_j,
                eff_scale_nats: eff_scale,
                eff_cap_nats: eff_cap,
                is_sequential: c.is_sequential,
            });
            if c.is_sequential && c.seq_j + 1 == c.seq_k {
                seq_edge_scale[c.seq_k] = Some((eff_scale, eff_cap));
            }
        }

        Some(Self {
            s_len,
            k_len,
            site_cap_nats: site_cap,
            evid_matrix,
            edges,
            adj_by_k,
            seq_edge_scale,
        })
    }

    #[inline]
    pub fn num_contacts(&self) -> usize {
        self.edges.len()
    }

    #[inline]
    pub fn site_cap_nats(&self) -> f64 {
        self.site_cap_nats
    }

    #[inline]
    pub fn evidence(&self, s1: usize, s2: usize) -> f64 {
        if s1 < self.s_len && s2 < self.s_len {
            self.evid_matrix[s1 * self.s_len + s2]
        } else {
            0.0
        }
    }

    /// Evaluate the exact `PAE`-gated, `noe_weight`-scaled, symmetrically site-capped pairwise
    /// bonus for placing `(s1, s2)` at `(j, k)` (`0.0` whenever `(min(j,k), max(j,k))` is not in
    /// `C_AF`, including any `PAE > 6.0 A` pair).
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
                return (e.eff_scale_nats * ev).clamp(0.0, e.eff_cap_nats);
            }
        }
        0.0
    }

    /// Sequential `(k - 1, k)` NOE bonus `Delta P^{NOE}_{s_1, s_2}(k - 1, k)`.
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
        let Some((scale, cap)) = self.seq_edge_scale[k_curr] else {
            return 0.0;
        };
        let ev = self.evid_matrix[s_prev * self.s_len + s_curr];
        (scale * ev).clamp(0.0, cap)
    }

    /// Sum of all incident sequential + medium/long-range NOE bonuses on `spin` placed at `slot`
    /// given current sequence assignment `at` (strictly `<= noe_weight * AF_NOESY_MAX_SITE_BONUS_NATS`).
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
                        sc += (e.eff_scale_nats * ev).clamp(0.0, e.eff_cap_nats);
                    }
                }
            }
        }
        sc.min(self.site_cap_nats)
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
                        sc += (e.eff_scale_nats * ev).clamp(0.0, e.eff_cap_nats);
                    }
                }
            }
        }
        sc.min(self.site_cap_nats)
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
                        sc += (e.eff_scale_nats * ev).clamp(0.0, e.eff_cap_nats);
                    }
                }
            }
        }
        sc
    }

    /// Exact total `15N-NOESY` pairwise bonus summed over all admitted contacts `(j < k)` in `C_AF`.
    pub fn total_bonus(&self, at: &[i32]) -> f64 {
        let mut sc = 0.0_f64;
        for e in &self.edges {
            let a = at.get(e.seq_j).copied().unwrap_or(-1);
            let b = at.get(e.seq_k).copied().unwrap_or(-1);
            if a >= 0 && b >= 0 && a != b {
                let sa = a as usize;
                let sb = b as usize;
                if sa < self.s_len && sb < self.s_len {
                    let ev = self.evid_matrix[sa * self.s_len + sb];
                    sc += (e.eff_scale_nats * ev).clamp(0.0, e.eff_cap_nats);
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

    if let Ok(std_peaks) = read_peaks_str(text) {
        if let Ok(conv) = noesy_15n_peaks_from_peaks(&std_peaks) {
            if !conv.is_empty() {
                return Ok(conv);
            }
        }
    }

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
