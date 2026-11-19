//! Bayesian excited-state (`CEST`, `CPMG`, off-resonance `R1rho`) structural hypothesis engine
//! (`T21.6b`, `docs/specs/21-alphafold-integrative-nmr.md` section 7.1).
//!
//! # Provenance & Clean-Room Statement
//! Original clean-room implementation derived strictly from peer-reviewed biomolecular NMR
//! chemical-exchange spectroscopy, chemical-shift forward physics, and Bayesian model selection
//! literature, with zero external code, parameter files, or restricted software artifacts:
//! - Vallurupalli, P., Hansen, D. F. & Kay, L. E. (*Proc. Natl. Acad. Sci. U.S.A.* **2008**, 105,
//!   11766-11771); Bouvignies, G., Vallurupalli, P., Hansen, D. F. et al. (*Nature* **2011**, 477,
//!   111-114); Vallurupalli, P., Sekhar, A., Yuwen, T. & Kay, L. E. (*J. Biomol. NMR* **2017**,
//!   67, 243-271) -- excited-state chemical shifts `varpi_B = varpi_A + Delta_varpi_AB` from
//!   `CPMG` and `CEST` experiments and structural characterization of invisible conformers.
//! - Skrynnikov, N. R., Dahlquist, F. W. & Kay, L. E. (*J. Am. Chem. Soc.* **2002**, 124,
//!   12352-12360) -- signed vs unsigned `|Delta_varpi_AB|` relaxation dispersion shift differences
//!   and sign marginalization.
//! - Tamiola, K., Acar, B. & Mulder, F. A. A. (*J. Am. Chem. Soc.* **2010**, 132, 18000-18003);
//!   Kjaergaard, M., Brander, S. & Poulsen, F. M. (*J. Biomol. NMR* **2011**, 49, 139-149);
//!   Platzer, G., Okon, M. & McIntosh, L. P. (*J. Biomol. NMR* **2014**, 60, 109-129) --
//!   sequence-, temperature-, and pH-corrected `ncIDP` random-coil chemical shifts for cooperative
//!   local unfolding (`H1`).
//! - Schubert, M., Labudde, D., Oschkinat, H. & Schmieder, P. (*J. Biomol. NMR* **2002**, 24,
//!   149-154) & Shen, Y. & Bax, A. (*J. Biomol. NMR* **2010**, 46, 199-204) -- `13CB - 13CG`
//!   (`Delta_beta_gamma: 4.5 <-> 9.6 ppm`) and neighboring peptide `13CA`/`13C'`/`15N`/`1H^N`
//!   chemical shift invariants for `Xxx-Pro` *trans* <-> *cis* isomerization (`H2`).
//! - Haigh, C. W. & Mallion, R. B. (*Mol. Phys.* **1972**, 24, 955-970) & Case, D. A. (*J. Biomol.
//!   NMR* **1995**, 6, 341-346) -- conjugated aromatic ring-current shielding tensors across
//!   `Phe`/`Tyr`/`Trp`/`His` `(chi1, chi2)` sidechain rotamer flips (`H3a`).
//! - Pelton, J. G., Torchia, D. A., Meadow, N. D. & Roseman, S. (*Protein Sci.* **1993**, 2,
//!   543-558) -- histidine imidazole `15ND1`/`15NE2`/`13CD2`/`13CG` tautomer (`Nd1-H <-> Ne2-H`)
//!   and protonation (`His^0 <-> HisH^+`) chemical shift transitions (`H3b`).
//! - Shen, Y. & Bax, A. (*J. Biomol. NMR* **2010**, 48, 13-22) & Wayment-Steele, H. K., Ojoawo, A.,
//!   Otten, R. et al. (*Nature* **2024**, 625, 832-839) -- candidate alternative 3D conformers /
//!   subsampled-MSA AlphaFold ensembles scored by differential 3D shift prediction (`H4`).
//! - Schwarz, G. (*Ann. Stat.* **1978**, 6, 461-464) & Kass, R. E. & Raftery, A. E. (*J. Am. Stat.
//!   Assoc.* **1995**, 90, 773-795) -- Bayesian Information Criterion (`BIC`) and `2 ln BF`
//!   evidence thresholds.
//!
//! # Scope Limits & Modeling Assumptions
//! - **No partial-unfolding or population scaling parameter**: `H1`..`H4` compare experimental
//!   excited-state shifts `Delta_varpi_AB = varpi_B - varpi_A` against 100%-switched endpoint
//!   structural hypotheses (full `ncIDP` random coil in `H1`, full `trans <-> cis` in `H2`, full
//!   rotamer/tautomer switch in `H3`, or supplied discrete endpoint conformers in `H4`). There is
//!   no fractional folding/population parameter `p_partial in (0, 1)` inside `mu_Delta`; partial-unfolding
//!   intermediates (e.g., Fyn SH3-like folding intermediates unless supplied as explicit `H4`
//!   endpoint conformers) may abstain as `UnmodeledSignificantExchange` or remain unmodeled.
//! - **Discrete-scan multiplicity & distinct-residue BIC**: Each hypothesis marginalizes over its
//!   evaluated discrete candidate set (`ln mean L_m` via log-sum-exp, penalizing scanning `M_H`
//!   candidates by `-ln M_H`) and counts `n = n_distinct_residues` in `BIC` so correlated nuclei on
//!   the same residue are not double-counted as independent degrees of freedom.

use std::collections::{BTreeMap, BTreeSet};
use std::f64::consts::PI;

use serde::{Deserialize, Serialize};

use crate::dihedrals::{align_by_res_names, periodic_angle_abs_diff_deg};
use crate::methyl_trosy::MethylGroupKind;
use crate::nef::read_nef_str;
use crate::nmrstar::{read_star_str_all_atoms, Reference};
use crate::noesy::{v_add, v_cross, v_dist, v_dot, v_scale, v_sub, v_unit, Vec3};
use crate::struct_shifts::{
    evaluate_haigh_mallion_ring_current, predict_structure_shifts_with_firewall,
    ConjugatedAromaticRing, StructShiftConfig, StructShiftPredictionReport, StructureModel,
};
use crate::structure::{det_exp, nerf_place_atom, BOND_C_O};
use crate::topology::{HIS_DOUBLY_PROTONATED_PRIORS, HIS_ND1H_PRIORS, HIS_NE2H_PRIORS};
use crate::torsion_db::{HomologFirewallQuery, DEFAULT_FIREWALL_MAX_SEQ_IDENTITY};

/// Maximum input TSV byte size accepted by [`parse_exchange_observations_tsv`] (`16 MiB` DoS guard).
pub const MAX_EXCHANGE_TSV_BYTES: usize = 16 * 1024 * 1024;

/// Maximum number of exchange shift observations accepted in one dataset (`32,768` DoS guard).
pub const MAX_EXCHANGE_OBSERVATIONS: usize = 32_768;

/// Maximum number of candidate alternative 3D conformers evaluated in one call (`64` DoS guard).
pub const MAX_ALT_STRUCTURES: usize = 64;

/// Degrees of freedom `nu = 5` for heavy-tailed Student-`t_5` differential shift likelihoods.
pub const STUDENT_T5_DOF: f64 = 5.0;

/// Minimum matched exchange observations required before committing a winning hypothesis (`N_obs >= 4`).
pub const MIN_COMMITTED_OBSERVATIONS: usize = 4;

/// Minimum Kass-Raftery `2 ln BF_{1,2} = BIC(H_{(2)}) - BIC(H_{(1)})` required to commit (`>= 6.0`).
pub const MIN_TWO_LN_BF_TOP1_VS_TOP2: f64 = 6.0;

/// Minimum Kass-Raftery `2 ln BF_{1,H0} = BIC(H_0) - BIC(H_{(1)})` required over Null (`>= 10.0`).
pub const MIN_TWO_LN_BF_VS_NULL: f64 = 10.0;

/// Maximum reduced `chi^2_nu` permitted on `H_{(1)}` to commit (`<= 2.5`).
pub const MAX_COMMITTED_REDUCED_CHI2: f64 = 2.5;

/// One-sided confidence level of the `chi^2_{p, nu}` upper bound used by the error-scale
/// consistency diagnostic (`p = 0.99`, i.e. a 1% false-alarm rate under correctly stated errors).
pub const ERROR_SCALE_CHI2_CONFIDENCE: f64 = 0.99;

/// Outlier clip (in robust-scale units) for the internal error-scale check: a best-model residual
/// is treated as a localized unmodeled-exchange outlier when `|z| > 3.5 * max(1, s_MAD)`, where
/// `s_MAD = median(|z|) / 0.674490` is the robust (median absolute deviation) scale of the
/// standardized residuals. `3.5` matches the unmodeled-exchange Bayes-factor threshold. Clipping
/// relative to the robust scale (sigma clipping) keeps a GLOBAL error under-statement (all `|z|`
/// inflated, `s_MAD ~ 2-3`) inside the inlier set while removing a few LOCALIZED exchanging sites
/// (`s_MAD ~ 1`). Under correctly stated Gaussian errors `P(|z| > 3.5) = 4.7e-4` and `E[z^2]`
/// drops by only `0.6%`, so the `chi^2` bound stays (slightly) conservative.
pub const ERROR_SCALE_OUTLIER_ABS_Z: f64 = 3.5;

/// A best-model site is a "spectator" (internal error-scale control) when its likelihood sigma is
/// within `5%` of the stated experimental error, i.e. the forward-model term contributes `< 32%`
/// of it in quadrature.
const SPECTATOR_SIGMA_RATIO: f64 = 1.05;

/// Minimum number of inlier spectator sites for the internal error-scale check; with fewer, the
/// all-site best-model `chi^2` (`nu = n_obs - k`) is used instead.
pub const MIN_ERROR_SCALE_SPECTATOR_SITES: usize = 5;

/// Median of the half-normal distribution `|N(0, 1)|` (`Phi^{-1}(0.75)`), the MAD consistency
/// constant.
const HALF_NORMAL_MEDIAN: f64 = 0.674_489_750_196_081_7;

/// Spatial `CA-CA` cutoff (`8.0 A`) for `H2: XxxProCisTrans` candidate Proline scanning.
pub const PROLINE_SPATIAL_CUTOFF_ANGSTROM: f64 = 8.0;

/// Sequence-separation window (`|i - p| <= 3`) for `H2: XxxProCisTrans` candidate Proline scanning.
pub const PROLINE_SEQ_WINDOW: i32 = 3;

/// Spatial cutoff (`10.0 A`) from exchanging spins to candidate aromatic sidechains in `H3`.
pub const AROMATIC_ROTAMER_CUTOFF_ANGSTROM: f64 = 10.0;

/// Sign-information mode of an experimental chemical-exchange shift difference.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ExchangeSignMode {
    /// Signed shift difference `Delta_varpi_AB = varpi_B - varpi_A in R` (`CEST`, `DEST`,
    /// Trott-Palmer off-resonance `R1rho`, `HSQC`/`HMQC` slow-exchange minor peaks).
    Signed,
    /// Unsigned shift magnitude `y = |Delta_varpi_AB| >= 0` (`CPMG` relaxation dispersion without
    /// HSQC/HMQC sign-determination experiments; Skrynnikov et al. *JACS* 2002).
    UnsignedMagnitude,
}

impl ExchangeSignMode {
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Signed => "signed",
            Self::UnsignedMagnitude => "unsigned",
        }
    }

    pub fn parse_label(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "signed" | "cest" | "dest" | "r1rho" | "true" | "1" | "+-" => Ok(Self::Signed),
            "unsigned" | "unsigned_magnitude" | "abs" | "magnitude" | "cpmg" | "false" | "0" => {
                Ok(Self::UnsignedMagnitude)
            }
            other => Err(format!(
                "unknown exchange sign_mode '{other}' (expected 'signed' or 'unsigned')"
            )),
        }
    }
}

/// Single per-nucleus chemical-exchange shift difference observation `(i, a)`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExchangeShiftObservation {
    pub chain_id: String,
    pub seq_id: i32,
    pub res_name: Option<String>,
    pub atom: String,
    /// Signed `Delta_varpi_AB = varpi_B - varpi_A` (`Signed`) or `|Delta_varpi_AB| >= 0` (`UnsignedMagnitude`) in ppm.
    pub dw_ppm: f64,
    /// Experimental `1-sigma` uncertainty `sigma_exp > 0` in ppm.
    pub dw_err_ppm: f64,
    pub sign_mode: ExchangeSignMode,
    /// Optional major/ground-state shift `varpi_A` (ppm) reported directly by `CEST` or `R1rho` fitting.
    pub major_ppm: Option<f64>,
}

/// Candidate biophysical excited-state hypothesis families (`H0` through `H4`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ExcitedStateHypothesisKind {
    /// `H0`: Null / unstructured baseline perturbation (`mu_Delta = 0`, `k = 0`).
    NullOrUnstructuredPerturbation,
    /// `H1`: Contiguous `[u, v]` cooperative unfolding to `ncIDP` random coil (`k = 2`).
    LocalUnfolding,
    /// `H2`: `Xxx-Pro` peptide-bond *trans* <-> *cis* isomerization (`k = 1`).
    XxxProCisTrans,
    /// `H3`: Steric-clash-checked aromatic `(chi1, chi2)` ring-current rotamer flip or Pelton `His`
    /// imidazole tautomer / protonation switch (`k = 2`).
    RotamerOrHisTautomerFlip,
    /// `H4`: Discrete alternative 3D conformer (`--alt-structures`) scored via `T21.2` (`k = 1`).
    Alternate3DConformer,
}

impl ExcitedStateHypothesisKind {
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NullOrUnstructuredPerturbation => "NullOrUnstructuredPerturbation",
            Self::LocalUnfolding => "LocalUnfolding",
            Self::XxxProCisTrans => "XxxProCisTrans",
            Self::RotamerOrHisTautomerFlip => "RotamerOrHisTautomerFlip",
            Self::Alternate3DConformer => "Alternate3DConformer",
        }
    }

    #[inline]
    pub const fn short_code(self) -> &'static str {
        match self {
            Self::NullOrUnstructuredPerturbation => "H0",
            Self::LocalUnfolding => "H1",
            Self::XxxProCisTrans => "H2",
            Self::RotamerOrHisTautomerFlip => "H3",
            Self::Alternate3DConformer => "H4",
        }
    }

    #[inline]
    pub const fn default_k(self) -> usize {
        match self {
            Self::NullOrUnstructuredPerturbation => 0,
            Self::LocalUnfolding => 2,
            Self::XxxProCisTrans => 1,
            Self::RotamerOrHisTautomerFlip => 2,
            Self::Alternate3DConformer => 1,
        }
    }
}

/// Direction of an `Xxx-Pro` `omega_{p-1}` peptide-bond isomerization (`H2`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProCisTransDirection {
    TransToCis,
    CisToTrans,
}

impl ProCisTransDirection {
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TransToCis => "trans_to_cis",
            Self::CisToTrans => "cis_to_trans",
        }
    }

    #[inline]
    pub const fn sign_factor(self) -> f64 {
        match self {
            Self::TransToCis => 1.0,
            Self::CisToTrans => -1.0,
        }
    }
}

/// Histidine imidazole tautomer or protonation exchange pathway (`H3b`, Pelton et al. 1993).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum HisExchangeTransition {
    /// Neutral `pi -> tau` tautomer flip (`N_delta1-H -> N_epsilon2-H`: `ND1 +82.0`, `NE2 -82.0`,
    /// `CD2 -7.2`, `CG +8.3 ppm`).
    Nd1HToNe2H,
    /// Neutral `tau -> pi` tautomer flip (`N_epsilon2-H -> N_delta1-H`: `ND1 -82.0`, `NE2 +82.0`,
    /// `CD2 +7.2`, `CG -8.3 ppm`).
    Ne2HToNd1H,
    /// Neutral `tau` (`Ne2H`) protonation to cationic imidazolium `HisH^+`.
    Ne2HToImidazolium,
    /// Cationic imidazolium `HisH^+` deprotonation to neutral `tau` (`Ne2H`).
    ImidazoliumToNe2H,
    /// Neutral `pi` (`Nd1H`) protonation to cationic imidazolium `HisH^+`.
    Nd1HToImidazolium,
    /// Cationic imidazolium `HisH^+` deprotonation to neutral `pi` (`Nd1H`).
    ImidazoliumToNd1H,
    /// Unsigned CPMG `|Delta_varpi|` (without ground `ND1`/`NE2` shifts) cannot distinguish forward vs
    /// reverse tautomer/protonation direction.
    AmbiguousDirectionOnUnsignedCpmg,
}

impl HisExchangeTransition {
    pub const ALL: [Self; 6] = [
        Self::Nd1HToNe2H,
        Self::Ne2HToNd1H,
        Self::Ne2HToImidazolium,
        Self::ImidazoliumToNe2H,
        Self::Nd1HToImidazolium,
        Self::ImidazoliumToNd1H,
    ];

    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Nd1HToNe2H => "Nd1H_to_Ne2H",
            Self::Ne2HToNd1H => "Ne2H_to_Nd1H",
            Self::Ne2HToImidazolium => "Ne2H_to_HisH+",
            Self::ImidazoliumToNe2H => "HisH+_to_Ne2H",
            Self::Nd1HToImidazolium => "Nd1H_to_HisH+",
            Self::ImidazoliumToNd1H => "HisH+_to_Nd1H",
            Self::AmbiguousDirectionOnUnsignedCpmg => "AmbiguousDirectionOnUnsignedCpmg",
        }
    }

    #[inline]
    pub const fn reverse(self) -> Self {
        match self {
            Self::Nd1HToNe2H => Self::Ne2HToNd1H,
            Self::Ne2HToNd1H => Self::Nd1HToNe2H,
            Self::Ne2HToImidazolium => Self::ImidazoliumToNe2H,
            Self::ImidazoliumToNe2H => Self::Ne2HToImidazolium,
            Self::Nd1HToImidazolium => Self::ImidazoliumToNd1H,
            Self::ImidazoliumToNd1H => Self::Nd1HToImidazolium,
            Self::AmbiguousDirectionOnUnsignedCpmg => Self::AmbiguousDirectionOnUnsignedCpmg,
        }
    }
}

/// Maximum-likelihood biophysical parameters `hat_theta_H` for a hypothesis class.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mechanism")]
pub enum HypothesisBestParameters {
    Null,
    LocalUnfolding {
        chain_id: String,
        start_seq_id: i32,
        end_seq_id: i32,
        start_res_name: String,
        end_res_name: String,
        core_residue_count: usize,
    },
    XxxProCisTrans {
        chain_id: String,
        pro_seq_id: i32,
        prev_seq_id: Option<i32>,
        prev_res_name: Option<String>,
        direction: ProCisTransDirection,
        ground_omega_deg: Option<f64>,
        predicted_delta_beta_gamma_ppm: f64,
    },
    AromaticRotamerFlip {
        chain_id: String,
        seq_id: i32,
        res_name: String,
        ground_chi1_deg: Option<f64>,
        ground_chi2_deg: Option<f64>,
        excited_chi1_deg: f64,
        excited_chi2_deg: f64,
        min_steric_dist_angstrom: f64,
    },
    HisTautomerOrProtonation {
        chain_id: String,
        seq_id: i32,
        transition: HisExchangeTransition,
        delta_nd1_ppm: f64,
        delta_ne2_ppm: f64,
        delta_cd2_ppm: f64,
        delta_cg_ppm: f64,
    },
    Alternate3DConformer {
        model_index: usize,
        model_label: String,
        exclusion_list_sha256: String,
        max_retained_seq_identity: f64,
    },
}

/// Per-observation forward prediction and Student-`t_5` residual under a hypothesis.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObservationPredictionRow {
    pub chain_id: String,
    pub seq_id: i32,
    pub res_name: String,
    pub atom: String,
    pub sign_mode: ExchangeSignMode,
    pub ground_ppm: f64,
    pub observed_dw_ppm: f64,
    pub observed_err_ppm: f64,
    /// Signed forward-model differential shift `mu_Delta = hat_delta_B - hat_delta_A` (ppm).
    pub predicted_dw_ppm: f64,
    /// Effective shift in observation space (`predicted_dw_ppm` if `Signed`, `|predicted_dw_ppm|` if `UnsignedMagnitude`).
    pub predicted_effective_dw_ppm: f64,
    pub sigma_tot_ppm: f64,
    pub log_likelihood: f64,
    pub z_residual: f64,
}

/// Summary of an individual parameter realization within a hypothesis family (`[u, v]`, `Pro p`,
/// `(arom r, chi1, chi2)`, or `alt_model m`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SubCandidateSummary {
    pub label: String,
    pub log_likelihood: f64,
    pub bic: f64,
    pub reduced_chi2: f64,
    pub affected_seq_ids: Vec<i32>,
}

/// Complete `BIC` and Bayes-factor evaluation for one candidate hypothesis class `H`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HypothesisScore {
    pub kind: ExcitedStateHypothesisKind,
    pub num_params_k: usize,
    pub n_obs: usize,
    #[serde(default)]
    pub n_distinct_residues: usize,
    #[serde(default)]
    pub n_candidates_evaluated: usize,
    pub max_log_likelihood: f64,
    #[serde(default)]
    pub marginal_log_likelihood: f64,
    pub bic: f64,
    pub posterior_weight: f64,
    pub two_ln_bf_vs_runner_up: Option<f64>,
    #[serde(default)]
    pub two_ln_bf_vs_best_other_non_null: Option<f64>,
    pub delta_bic_from_best: f64,
    pub two_ln_bf_vs_null: f64,
    pub reduced_chi2: f64,
    pub rmse_ppm: f64,
    pub pearson_r: Option<f64>,
    pub spearman_rho: Option<f64>,
    pub best_locus_summary: String,
    pub affected_seq_ids: Vec<i32>,
    pub parameters: HypothesisBestParameters,
    pub predictions: Vec<ObservationPredictionRow>,
    pub top_sub_candidates: Vec<SubCandidateSummary>,
}

/// Specific reason classification when [`ExcitedStateVerdict::AmbiguousOrUnmodeledState`] is returned.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExcitedStateAbstentionKind {
    InsufficientDistinctResidues,
    ConsistentWithNoise,
    UnmodeledSignificantExchange,
    AmbiguousBetweenMechanisms,
    /// The stated `dw_err_ppm` are statistically inconsistent with the data (best-model inlier or
    /// control-site `chi^2_nu` above the `chi^2_{0.99, nu}` bound), so neither a committed
    /// mechanism nor `UnmodeledSignificantExchange` can be reported as a confident call.
    ErrorScaleInconsistent,
}

impl ExcitedStateAbstentionKind {
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InsufficientDistinctResidues => "InsufficientDistinctResidues",
            Self::ConsistentWithNoise => "ConsistentWithNoise",
            Self::UnmodeledSignificantExchange => "UnmodeledSignificantExchange",
            Self::AmbiguousBetweenMechanisms => "AmbiguousBetweenMechanisms",
            Self::ErrorScaleInconsistent => "ErrorScaleInconsistent",
        }
    }
}

/// Final calibrated decision of [`classify_excited_state`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", content = "hypothesis")]
pub enum ExcitedStateVerdict {
    /// Unambiguous excited-state classification (`N_obs >= 4`, `2 ln BF_{1,2} >= 6.0`,
    /// `2 ln BF_{1,H0} >= 10.0`, and `chi^2_nu <= 2.5`).
    Committed(ExcitedStateHypothesisKind),
    /// Honest abstention when observations are sparse, pure noise, ambiguous between competing
    /// mechanisms, or poorly fit by all modeled conformer classes (`chi^2_nu > 2.5`).
    AmbiguousOrUnmodeledState,
}

impl ExcitedStateVerdict {
    #[inline]
    pub const fn is_committed(self) -> bool {
        matches!(self, Self::Committed(_))
    }

    #[inline]
    pub const fn committed_hypothesis(self) -> Option<ExcitedStateHypothesisKind> {
        match self {
            Self::Committed(h) => Some(h),
            Self::AmbiguousOrUnmodeledState => None,
        }
    }

    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Committed(ExcitedStateHypothesisKind::NullOrUnstructuredPerturbation) => {
                "Committed(NullOrUnstructuredPerturbation)"
            }
            Self::Committed(ExcitedStateHypothesisKind::LocalUnfolding) => {
                "Committed(LocalUnfolding)"
            }
            Self::Committed(ExcitedStateHypothesisKind::XxxProCisTrans) => {
                "Committed(XxxProCisTrans)"
            }
            Self::Committed(ExcitedStateHypothesisKind::RotamerOrHisTautomerFlip) => {
                "Committed(RotamerOrHisTautomerFlip)"
            }
            Self::Committed(ExcitedStateHypothesisKind::Alternate3DConformer) => {
                "Committed(Alternate3DConformer)"
            }
            Self::AmbiguousOrUnmodeledState => "AmbiguousOrUnmodeledState",
        }
    }
}

/// Candidate alternative 3D coordinate model supplied for `H4: Alternate3DConformer`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AltConformerInput {
    pub label: String,
    pub model: StructureModel,
}

/// Configuration for [`classify_excited_state`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExcitedStateConfig {
    pub sample_ph: f64,
    pub temperature_kelvin: f64,
    pub max_seq_identity: f64,
    pub min_obs_to_commit: usize,
    pub min_two_ln_bf_12: f64,
    pub min_two_ln_bf_null: f64,
    pub max_reduced_chi2: f64,
    pub sigma_fwd_scale: f64,
    /// Explicitly records whether CPMG input rows were pre-filtered by `significant_exchange=true`.
    #[serde(default)]
    pub cpmg_prefiltered_significant_only: bool,
}

impl Default for ExcitedStateConfig {
    fn default() -> Self {
        Self {
            sample_ph: 6.5,
            temperature_kelvin: 298.15,
            max_seq_identity: DEFAULT_FIREWALL_MAX_SEQ_IDENTITY,
            min_obs_to_commit: MIN_COMMITTED_OBSERVATIONS,
            min_two_ln_bf_12: MIN_TWO_LN_BF_TOP1_VS_TOP2,
            min_two_ln_bf_null: MIN_TWO_LN_BF_VS_NULL,
            max_reduced_chi2: MAX_COMMITTED_REDUCED_CHI2,
            sigma_fwd_scale: 1.0,
            cpmg_prefiltered_significant_only: false,
        }
    }
}

/// Diagnostic summarizing reduced-$\chi^2$ consistency under $H_0$ and the top-ranked hypothesis,
/// explicitly flagging the risk that underestimated experimental error bars (`2x`–`3x`) can cause
/// false $H_0$ rejection and spurious `UnmodeledSignificantExchange` or non-null ranking.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Chi2ConsistencyDiagnostic {
    pub h0_reduced_chi2: f64,
    pub top1_reduced_chi2: f64,
    /// `chi^2_{0.99, nu}/nu` bound for the best-fitting model over all sites (`nu = n_obs - k`).
    pub consistency_upper_bound: f64,
    /// `H0` reduced `chi^2_nu` above its own `chi^2_{0.99, n_obs}/n_obs` bound. This is evidence of
    /// exchange OR of under-stated errors and is informational only (it never fires the warning).
    pub h0_exceeds_consistency_bound: bool,
    /// Best-fitting model reduced `chi^2_nu` above [`Self::consistency_upper_bound`] (all sites).
    pub top1_exceeds_consistency_bound: bool,
    pub implied_error_scale_factor_h0: f64,
    pub implied_error_scale_factor_top1: f64,
    #[serde(default)]
    pub error_scale_warning: Option<String>,
    /// One-sided confidence level of every `chi^2` bound ([`ERROR_SCALE_CHI2_CONFIDENCE`]).
    #[serde(default)]
    pub confidence_level: f64,
    /// Best-fitting hypothesis and its degrees of freedom `nu = n_obs - k_params`.
    #[serde(default)]
    pub top1_hypothesis: Option<ExcitedStateHypothesisKind>,
    #[serde(default)]
    pub top1_dof: usize,
    /// `chi^2_{0.99, n_obs}/n_obs` bound for `H0` (`k = 0`).
    #[serde(default)]
    pub h0_consistency_upper_bound: f64,
    /// Best-model sites predicting `dw = 0` (`sigma_tot <= 1.05 dw_err`): internal controls.
    #[serde(default)]
    pub n_spectator_sites: usize,
    /// Robust scale `s_MAD = median(|z|) / 0.674490` of the best model's spectator residuals.
    #[serde(default)]
    pub robust_scale_mad: f64,
    /// Spectator-site residuals with `|z| > 3.5 * max(1, s_MAD)` excluded as localized outliers.
    #[serde(default)]
    pub n_outlier_sites_excluded: usize,
    /// Reduced `chi^2_nu` of the best model over its inlier spectator sites (predicted `dw = 0`,
    /// `sigma = dw_err`; `nu = n_inliers`) when `>= 5` inliers form a majority of spectators
    /// (internal controls); `None` otherwise.
    #[serde(default)]
    pub inlier_reduced_chi2: Option<f64>,
    #[serde(default)]
    pub inlier_dof: Option<usize>,
    #[serde(default)]
    pub inlier_consistency_upper_bound: Option<f64>,
    /// Supplied non-exchanging control sites (`site_role = control`, expected `dw = 0`).
    #[serde(default)]
    pub n_control_sites: usize,
    #[serde(default)]
    pub control_reduced_chi2: Option<f64>,
    #[serde(default)]
    pub control_consistency_upper_bound: Option<f64>,
    #[serde(default)]
    pub control_exceeds_consistency_bound: Option<bool>,
    /// Which statistic decided [`Self::error_scale_inconsistent`]: `control_sites`,
    /// `best_model_spectator_inliers`, or `best_model_all_sites`.
    #[serde(default)]
    pub error_scale_evidence: String,
    /// `true` when the stated errors are statistically inconsistent with the data; the verdict is
    /// then downgraded to [`ExcitedStateAbstentionKind::ErrorScaleInconsistent`] for committed,
    /// `UnmodeledSignificantExchange` and `ConsistentWithNoise` outcomes.
    #[serde(default)]
    pub error_scale_inconsistent: bool,
}

/// Complete Bayesian excited-state structural classification report.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExcitedStateReport {
    pub verdict: ExcitedStateVerdict,
    pub verdict_reason: String,
    #[serde(default)]
    pub abstention_kind: Option<ExcitedStateAbstentionKind>,
    pub n_observations: usize,
    #[serde(default)]
    pub n_distinct_residues: usize,
    #[serde(default)]
    pub n_distinct_exchanging_residues: usize,
    pub n_signed_observations: usize,
    pub n_unsigned_observations: usize,
    pub n_unmatched_observations: usize,
    pub two_ln_bf_top1_vs_top2: f64,
    #[serde(default)]
    pub two_ln_bf_top1_vs_runnerup_non_null: Option<f64>,
    #[serde(default)]
    pub runnerup_non_null_hypothesis: Option<ExcitedStateHypothesisKind>,
    pub two_ln_bf_top1_vs_null: f64,
    #[serde(default)]
    pub unmodeled_exchange_two_ln_bf_vs_null: f64,
    pub top1_reduced_chi2: f64,
    #[serde(default)]
    pub cpmg_prefiltered_significant_only: bool,
    pub sample_ph: f64,
    pub temperature_kelvin: f64,
    pub ground_exclusion_list_sha256: String,
    pub ground_max_retained_seq_identity: f64,
    pub ranked_hypotheses: Vec<HypothesisScore>,
    pub warnings: Vec<String>,
    #[serde(default)]
    pub error_scale_warning: Option<String>,
    #[serde(default)]
    pub chi2_consistency_diagnostic: Option<Chi2ConsistencyDiagnostic>,
    #[serde(default)]
    pub scope_limits_note: String,
}

// ============================================================================
// 1. Exact Normalized Student-t_5 Signed & Folded-Unsigned Log-Likelihoods
// ============================================================================

/// Exact normalization log-constant `ln(Gamma(3) / (sqrt(5 * pi) * Gamma(5/2))) = ln(8 / (3 * pi * sqrt(5)))`
/// for the unit-scale Student-`t(nu = 5)` probability density on `(-inf, +inf)`.
#[inline]
fn ln_student_t5_norm_const() -> f64 {
    (8.0 / (3.0 * PI * 5.0_f64.sqrt())).ln()
}

/// Numerically stable `ln(exp(a) + exp(b))`.
#[inline]
fn log_sum_exp_pair(a: f64, b: f64) -> f64 {
    let hi = a.max(b);
    let lo = a.min(b);
    if !hi.is_finite() {
        return hi;
    }
    hi + (lo - hi).exp().ln_1p()
}

/// Exact unit-scale Student-`t(nu = 5)` log-density `ln f_{t_5}(z)` on `(-inf, +inf)`,
/// satisfying `integral_{-inf}^{+inf} exp(student_t5_log_pdf_std(z)) dz == 1`.
#[inline]
pub fn student_t5_log_pdf_std(z: f64) -> f64 {
    ln_student_t5_norm_const() - 3.0 * (0.2 * z * z).ln_1p()
}

/// Exact normalized signed Student-`t_5` log-likelihood (`ExchangeSignMode::Signed`, `CEST` / `R1rho`):
/// `ell_signed(dw_obs | mu_pred, sigma_tot) = ln f_{t_5}((dw_obs - mu_pred) / sigma_tot) - ln(sigma_tot)`.
pub fn signed_student_t5_log_likelihood(
    dw_obs: f64,
    mu_pred: f64,
    sigma_tot: f64,
) -> Result<f64, String> {
    if !dw_obs.is_finite() || !mu_pred.is_finite() || !sigma_tot.is_finite() || sigma_tot <= 0.0 {
        return Err(format!(
            "invalid signed Student-t5 parameters (dw_obs={dw_obs}, mu_pred={mu_pred}, sigma_tot={sigma_tot})"
        ));
    }
    let z = (dw_obs - mu_pred) / sigma_tot;
    Ok(student_t5_log_pdf_std(z) - sigma_tot.ln())
}

/// Exact normalized folded Student-`t_5` unsigned log-likelihood (`ExchangeSignMode::UnsignedMagnitude`,
/// `CPMG` `y = |Delta_varpi_AB| >= 0`, Skrynnikov et al. 2002):
/// `ell_unsigned(y | mu, sigma) = ln[f_{t_5}((y - mu)/sigma) + f_{t_5}((-y - mu)/sigma)] - ln(sigma)`.
///
/// Properly normalized on `[0, +inf)`: `integral_0^{+inf} exp(ell_unsigned(y | mu, sigma)) dy == 1`
/// for every `mu in R` and `sigma > 0` (including `mu == 0`).
pub fn unsigned_folded_student_t5_log_likelihood(
    abs_dw_obs: f64,
    mu_pred: f64,
    sigma_tot: f64,
) -> Result<f64, String> {
    if !abs_dw_obs.is_finite() || !mu_pred.is_finite() || !sigma_tot.is_finite() || sigma_tot <= 0.0
    {
        return Err(format!(
            "invalid unsigned Student-t5 parameters (abs_dw_obs={abs_dw_obs}, mu_pred={mu_pred}, sigma_tot={sigma_tot})"
        ));
    }
    if abs_dw_obs < 0.0 {
        return Err(format!(
            "unsigned magnitude exchange observation must be >= 0.0 ppm, got {abs_dw_obs}"
        ));
    }
    let z_pos = (abs_dw_obs - mu_pred) / sigma_tot;
    let z_neg = (-abs_dw_obs - mu_pred) / sigma_tot;
    let ll = log_sum_exp_pair(student_t5_log_pdf_std(z_pos), student_t5_log_pdf_std(z_neg))
        - sigma_tot.ln();
    Ok(ll)
}

// ----------------------------------------------------------------------------
// chi^2 distribution (error-scale consistency diagnostic, `21F:B19`)
// ----------------------------------------------------------------------------

/// Lanczos (`g = 7`, `n = 9`) `ln Gamma(z)` for `z > 0` (Press et al., *Numerical Recipes* 3rd
/// ed., section 6.1), accurate to `~1e-15` relative.
fn ln_gamma_lanczos(z: f64) -> f64 {
    const COEFFS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if z < 0.5 {
        // Reflection: Gamma(z) Gamma(1 - z) = pi / sin(pi z)
        return (PI / (PI * z).sin()).ln() - ln_gamma_lanczos(1.0 - z);
    }
    let zm = z - 1.0;
    let mut x = COEFFS[0];
    for (i, &c) in COEFFS.iter().enumerate().skip(1) {
        x += c / (zm + i as f64);
    }
    let t = zm + 7.5;
    0.5 * (2.0 * PI).ln() + (zm + 0.5) * t.ln() - t + x.ln()
}

/// Regularized lower incomplete gamma `P(a, x)` (series for `x < a + 1`, Lentz continued
/// fraction for the complement otherwise; *Numerical Recipes* section 6.2).
fn regularized_lower_gamma(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let ln_pref = -x + a * x.ln() - ln_gamma_lanczos(a);
    if x < a + 1.0 {
        let mut term = 1.0 / a;
        let mut sum = term;
        let mut ap = a;
        for _ in 0..10_000 {
            ap += 1.0;
            term *= x / ap;
            sum += term;
            if term.abs() < sum.abs() * 1e-16 {
                break;
            }
        }
        (sum * ln_pref.exp()).clamp(0.0, 1.0)
    } else {
        const FPMIN: f64 = 1e-300;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / FPMIN;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..10_000 {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < FPMIN {
                d = FPMIN;
            }
            c = b + an / c;
            if c.abs() < FPMIN {
                c = FPMIN;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 1e-16 {
                break;
            }
        }
        (1.0 - ln_pref.exp() * h).clamp(0.0, 1.0)
    }
}

/// Cumulative distribution function of the `chi^2_nu` distribution, `P(nu/2, x/2)`.
pub fn chi2_cdf(x: f64, dof: f64) -> f64 {
    if !(x.is_finite() && dof.is_finite()) || dof <= 0.0 || x <= 0.0 {
        return 0.0;
    }
    regularized_lower_gamma(0.5 * dof, 0.5 * x)
}

/// Exact `p`-quantile `chi^2_{p, nu}` (`P(X <= q) = p`) by bisection on [`chi2_cdf`] to `1e-12`
/// relative precision. (Wilson & Hilferty 1931 is used only to seed the upper bracket.)
pub fn chi2_upper_quantile(p: f64, dof: f64) -> f64 {
    if !(p.is_finite() && dof.is_finite()) || dof <= 0.0 || p <= 0.0 {
        return 0.0;
    }
    let p = p.min(1.0 - 1e-15);
    // Wilson-Hilferty seed with a generous z (covers p <= 1 - 1e-15)
    let k = dof;
    let wh = k * (1.0 - 2.0 / (9.0 * k) + 8.3 * (2.0 / (9.0 * k)).sqrt()).powi(3);
    let mut hi = wh.max(k + 10.0 * (2.0 * k).sqrt() + 10.0);
    while chi2_cdf(hi, dof) < p {
        hi *= 2.0;
    }
    let mut lo = 0.0_f64;
    for _ in 0..400 {
        let mid = 0.5 * (lo + hi);
        if chi2_cdf(mid, dof) < p {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo <= 1e-12 * hi.max(1e-300) {
            break;
        }
    }
    0.5 * (lo + hi)
}

/// Reduced-`chi^2` upper bound `chi^2_{p, nu} / nu` at [`ERROR_SCALE_CHI2_CONFIDENCE`].
fn reduced_chi2_upper_bound(dof: usize) -> f64 {
    let nu = dof.max(1) as f64;
    chi2_upper_quantile(ERROR_SCALE_CHI2_CONFIDENCE, nu) / nu
}

/// Baseline differential forward-model uncertainty `sigma_{fwd,a}` (ppm) for `Delta_varpi_AB`
/// (shared across all hypotheses `H0..H4` so spectator residues with `mu_Delta == 0` carry zero
/// `ln(sigma)` bias between competing models).
pub fn differential_forward_sigma_ppm(res_name: &str, atom: &str) -> f64 {
    let rn = res_name.trim().to_ascii_uppercase();
    let at = atom.trim().to_ascii_uppercase();
    match at.as_str() {
        "ND1" | "NE2" if rn == "HIS" => 3.40,
        "NE1" if rn == "TRP" => 0.90,
        "N" => 0.65,
        "H" | "HN" => 0.14,
        "HA" | "HA2" | "HA3" => 0.09,
        "HD1*" | "HD2*" | "HG1*" | "HG2*" | "HE*" | "HB*" => 0.08,
        "HD1" | "HD2" | "HE1" | "HE2" | "HZ" | "HZ2" | "HZ3" | "HH2" => 0.12,
        "CG" | "CD2" | "CE1" if rn == "HIS" => 0.85,
        "CG" | "CD1" | "CD2" | "CE1" | "CE2" | "CE3" | "CZ" | "CZ2" | "CZ3" | "CH2"
            if matches!(rn.as_str(), "PHE" | "TYR" | "TRP") =>
        {
            0.70
        }
        "CA" | "CB" | "C" | "CG" | "CG1" | "CG2" | "CD" | "CD1" | "CE" => 0.42,
        other if other.starts_with('H') => 0.12,
        other if other.starts_with('N') => 0.85,
        _ => 0.45,
    }
}

// ============================================================================
// 2. Atom Normalization, Shift Parsing (`NMR-STAR` / `NEF`) & Exchange TSV I/O
// ============================================================================

/// Canonical nucleus keys for which the differential forward model has explicit physics
/// (`differential_forward_sigma_ppm`, ring-current, H-bond, `ncIDP` random-coil, `Xxx-Pro`,
/// aromatic-rotamer and His-tautomer generators): backbone `N`/`H`/`CA`/`CB`/`C`/`HA(2/3)`,
/// side-chain carbons, His/Trp ring nitrogens, aromatic ring protons and `ILVMAT` methyl
/// pseudo-atoms. Any other atom string is rejected (fail closed) instead of being scored with a
/// generic fallback sigma (`21F:B9`).
pub const SUPPORTED_EXCHANGE_ATOMS: &[&str] = &[
    "N", "H", "CA", "CB", "C", "HA", "HA2", "HA3", "CG", "CG1", "CG2", "CD", "CD1", "CD2", "CE",
    "CE1", "CE2", "CE3", "CZ", "CZ2", "CZ3", "CH2", "ND1", "NE2", "NE1", "HD1", "HD2", "HE1",
    "HE2", "HZ", "HZ2", "HZ3", "HH2", "HB*", "HG1*", "HG2*", "HD1*", "HD2*", "HE*",
];

/// Static (non-echoing) error text for an exchange observation whose atom is outside
/// [`SUPPORTED_EXCHANGE_ATOMS`].
pub const UNSUPPORTED_EXCHANGE_ATOM_MESSAGE: &str = "unsupported exchange nucleus: atom is not in the supported set (N, H, CA, CB, C, HA/HA2/HA3, side-chain C, His ND1/NE2, Trp NE1, aromatic ring H, methyl HB*/HG1*/HG2*/HD1*/HD2*/HE*)";

/// Whether an already-canonicalized atom key is modeled by the excited-state forward physics.
pub fn is_supported_exchange_atom(canonical_atom: &str) -> bool {
    SUPPORTED_EXCHANGE_ATOMS.contains(&canonical_atom)
}

/// Parser-side check when the residue type may be unknown: accept atoms that are supported
/// directly or that canonicalize onto a supported methyl / glycine pseudo-atom for some residue.
fn is_supported_exchange_atom_any_residue(res_name: Option<&str>, raw_atom: &str) -> bool {
    if is_supported_exchange_atom(&canonicalize_exchange_atom_name(res_name, raw_atom)) {
        return true;
    }
    res_name.is_none()
        && ["ILE", "LEU", "VAL", "THR", "MET", "ALA", "GLY"]
            .iter()
            .any(|rn| {
                is_supported_exchange_atom(&canonicalize_exchange_atom_name(Some(rn), raw_atom))
            })
}

/// Canonicalize IUPAC / spectrometer atom names onto FERRIC's shift keys.
pub fn canonicalize_exchange_atom_name(res_name: Option<&str>, raw_atom: &str) -> String {
    let up = raw_atom.trim().to_ascii_uppercase();
    let rn = res_name.unwrap_or("").trim().to_ascii_uppercase();
    match up.as_str() {
        "HN" | "1H" | "H_N" => "H".to_string(),
        "15N" | "AMIDE_N" => "N".to_string(),
        "CO" | "C'" | "13CO" | "13C'" => "C".to_string(),
        "13CA" => "CA".to_string(),
        "13CB" => "CB".to_string(),
        "13CG" => "CG".to_string(),
        "HA1" if rn != "GLY" => "HA".to_string(),
        "1HA" if rn == "GLY" => "HA2".to_string(),
        "2HA" if rn == "GLY" => "HA3".to_string(),
        "HD1%" | "HD1#" | "HD1*" => "HD1*".to_string(),
        "HD11" | "HD12" | "HD13" | "HD1" if matches!(rn.as_str(), "ILE" | "LEU") => {
            "HD1*".to_string()
        }
        "HD2%" | "HD2#" | "HD2*" => "HD2*".to_string(),
        "HD21" | "HD22" | "HD23" | "HD2" if rn == "LEU" => "HD2*".to_string(),
        "HG1%" | "HG1#" | "HG1*" => "HG1*".to_string(),
        "HG11" | "HG12" | "HG13" | "HG1" if rn == "VAL" => "HG1*".to_string(),
        "HG2%" | "HG2#" | "HG2*" => "HG2*".to_string(),
        "HG21" | "HG22" | "HG23" | "HG2" if matches!(rn.as_str(), "VAL" | "THR") => {
            "HG2*".to_string()
        }
        "HE%" | "HE#" | "HE*" => "HE*".to_string(),
        "HE1" | "HE2" | "HE3" if rn == "MET" => "HE*".to_string(),
        "HB%" | "HB#" | "HB*" => "HB*".to_string(),
        "HB1" | "HB2" | "HB3" | "HB" if rn == "ALA" => "HB*".to_string(),
        "CD" if rn == "ILE" => "CD1".to_string(),
        _ => up,
    }
}

/// Parse ground-state assigned chemical shifts from `NMR-STAR` (`.str`) or `NEF` (`.nef`) text,
/// retaining both backbone and sidechain atoms (`CG`, `ND1`, `NE2`, `CD2`, `ILVMAT` methyls).
pub fn parse_ground_shifts_str(text: &str, format_hint: Option<&str>) -> Result<Reference, String> {
    let is_nef = match format_hint {
        Some(h) if h.eq_ignore_ascii_case("nef") => true,
        Some(h) if h.eq_ignore_ascii_case("str") || h.eq_ignore_ascii_case("star") => false,
        _ => text.contains("save_nef_molecular_system") || text.contains("_nef_chemical_shift."),
    };
    if is_nef {
        let doc = read_nef_str(text)?;
        doc.to_reference()
    } else {
        read_star_str_all_atoms(text, None)
    }
}

/// Scope-limit summary emitted alongside [`ExcitedStateReport`].
pub const EXCITED_STATE_SCOPE_LIMITS_NOTE: &str = "Endpoint-only structural Delta_varpi models (no partial-unfolding or fractional population parameter inside mu_Delta); CPMG unsigned data cannot resolve tautomer direction without ground sidechain 15N shifts.";

/// Parsed exchange shift observations plus provenance flags from TSV parsing.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParsedExchangeTable {
    pub observations: Vec<ExchangeShiftObservation>,
    pub cpmg_prefiltered_significant_only: bool,
    pub n_filtered_nonsignificant_cpmg_rows: usize,
    /// Rows tagged `site_role = control` (or `control = true`): non-exchanging reference sites
    /// whose fitted `dw` (expected `0`) and stated `dw_err_ppm` calibrate the error scale.
    #[serde(default)]
    pub control_observations: Vec<ExchangeShiftObservation>,
}

/// Parse [`ExchangeShiftObservation`] rows and metadata from TSV text, rejecting any duplicate
/// `(chain_id, seq_id, normalized_atom)` observations with `Err`.
pub fn parse_exchange_observations_tsv_with_meta(
    tsv_text: &str,
) -> Result<ParsedExchangeTable, String> {
    if tsv_text.len() > MAX_EXCHANGE_TSV_BYTES {
        return Err(format!(
            "exchange TSV ({} bytes) exceeds 16 MiB limit ({MAX_EXCHANGE_TSV_BYTES})",
            tsv_text.len()
        ));
    }

    let mut header_map: Option<BTreeMap<String, usize>> = None;
    let mut out = Vec::new();
    let mut controls = Vec::new();
    let mut seen_keys: BTreeSet<(String, i32, String)> = BTreeSet::new();
    let mut cpmg_prefiltered_significant_only = false;
    let mut n_filtered_nonsignificant_cpmg_rows = 0usize;

    for (line_idx, raw_line) in tsv_text.lines().enumerate() {
        let line_no = line_idx + 1;
        let trimmed = raw_line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = raw_line.split('\t').map(str::trim).collect();
        if header_map.is_none() {
            let first_low = cols[0].to_ascii_lowercase();
            if matches!(
                first_low.as_str(),
                "chain_id" | "chain" | "seq_id" | "resi" | "residue"
            ) || cols.iter().any(|c| {
                matches!(
                    c.to_ascii_lowercase().as_str(),
                    "dw_ppm" | "dw_ab_ppm" | "abs_dw_ppm" | "delta_ppm"
                )
            }) {
                let mut hm = BTreeMap::new();
                for (ci, &cname) in cols.iter().enumerate() {
                    hm.insert(cname.to_ascii_lowercase(), ci);
                }
                header_map = Some(hm);
                continue;
            }
        }

        let mut is_control = false;
        let obs = if let Some(ref hm) = header_map {
            let get_col = |names: &[&str]| -> Option<&str> {
                for &nm in names {
                    if let Some(&idx) = hm.get(nm) {
                        if let Some(&val) = cols.get(idx) {
                            if !val.is_empty() && val != "." && !val.eq_ignore_ascii_case("na") {
                                return Some(val);
                            }
                        }
                    }
                }
                None
            };

            if hm.contains_key("significant_exchange") {
                cpmg_prefiltered_significant_only = true;
            }
            if let Some(sig_flag) = get_col(&["significant_exchange"]) {
                if sig_flag.eq_ignore_ascii_case("false") || sig_flag == "0" {
                    n_filtered_nonsignificant_cpmg_rows += 1;
                    continue;
                }
            }

            if let Some(role) = get_col(&["site_role", "role"]) {
                is_control = match role.to_ascii_lowercase().as_str() {
                    "control" | "reference" | "non_exchanging" => true,
                    "exchange" | "exchanging" => false,
                    _ => {
                        return Err(format!(
                            "line {line_no}: site_role must be 'exchange' or 'control'"
                        ));
                    }
                };
            } else if let Some(flag) = get_col(&["control", "is_control"]) {
                is_control = flag.eq_ignore_ascii_case("true") || flag == "1";
            }
            let chain_id = get_col(&["chain_id", "chain"]).unwrap_or("A").to_string();
            let seq_str = get_col(&["seq_id", "resi", "res_seq"])
                .ok_or_else(|| format!("line {line_no}: missing seq_id column"))?;
            let seq_id: i32 = seq_str
                .parse()
                .map_err(|e| format!("line {line_no}: invalid seq_id '{seq_str}': {e}"))?;
            let res_name = get_col(&["res_name", "resn", "comp_id"]).map(str::to_ascii_uppercase);
            let raw_atom = get_col(&["atom", "atom_id", "nucleus"]).unwrap_or("N");
            if !is_supported_exchange_atom_any_residue(res_name.as_deref(), raw_atom) {
                return Err(format!(
                    "line {line_no}: {UNSUPPORTED_EXCHANGE_ATOM_MESSAGE}"
                ));
            }
            let atom = canonicalize_exchange_atom_name(res_name.as_deref(), raw_atom);

            let has_abs_col = get_col(&["abs_dw_ppm"]).is_some();
            let has_cpmg_gate =
                hm.contains_key("significant_exchange") || hm.contains_key("alpha_rex");

            let dw_str =
                get_col(&["dw_ppm", "dw_ab_ppm", "abs_dw_ppm", "delta_ppm"]).ok_or_else(|| {
                    format!("line {line_no}: missing dw_ppm / dw_ab_ppm / abs_dw_ppm value")
                })?;
            let dw_ppm: f64 = dw_str
                .parse()
                .map_err(|e| format!("line {line_no}: invalid dw_ppm '{dw_str}': {e}"))?;
            if !dw_ppm.is_finite() {
                return Err(format!("line {line_no}: non-finite dw_ppm"));
            }

            let err_str = get_col(&[
                "dw_err_ppm",
                "dw_ppm_err",
                "dw_ab_ppm_err",
                "sigma_ppm",
                "err_ppm",
            ])
            .unwrap_or("0.05");
            let dw_err_ppm: f64 = err_str
                .parse()
                .map_err(|e| format!("line {line_no}: invalid dw_err_ppm '{err_str}': {e}"))?;
            if !dw_err_ppm.is_finite() || dw_err_ppm <= 0.0 {
                return Err(format!(
                    "line {line_no}: dw_err_ppm must be finite and > 0.0, got {dw_err_ppm}"
                ));
            }

            let sign_mode = if let Some(sm_raw) = get_col(&["sign_mode", "mode", "signed"]) {
                ExchangeSignMode::parse_label(sm_raw).map_err(|e| format!("line {line_no}: {e}"))?
            } else if has_abs_col || has_cpmg_gate {
                ExchangeSignMode::UnsignedMagnitude
            } else {
                ExchangeSignMode::Signed
            };

            if sign_mode == ExchangeSignMode::UnsignedMagnitude && dw_ppm < 0.0 {
                return Err(format!(
                    "line {line_no}: unsigned magnitude observation has negative dw_ppm ({dw_ppm})"
                ));
            }

            let major_ppm = match get_col(&["major_ppm", "ground_ppm", "varpi_a"]) {
                Some(ms) => {
                    let mv: f64 = ms
                        .parse()
                        .map_err(|e| format!("line {line_no}: invalid major_ppm '{ms}': {e}"))?;
                    if !mv.is_finite() {
                        return Err(format!("line {line_no}: non-finite major_ppm"));
                    }
                    Some(mv)
                }
                None => None,
            };

            ExchangeShiftObservation {
                chain_id,
                seq_id,
                res_name,
                atom,
                dw_ppm,
                dw_err_ppm,
                sign_mode,
                major_ppm,
            }
        } else {
            // Headerless 5- or 6-column fallback: seq_id, res_name, atom, dw_ppm, dw_err_ppm[, sign_mode]
            if cols.len() < 5 {
                return Err(format!(
                    "line {line_no}: expected header or >= 5 TSV columns (seq_id, res_name, atom, dw_ppm, dw_err_ppm), got {}",
                    cols.len()
                ));
            }
            let seq_id: i32 = cols[0]
                .parse()
                .map_err(|e| format!("line {line_no}: invalid seq_id '{}': {e}", cols[0]))?;
            let res_name =
                (!cols[1].is_empty() && cols[1] != ".").then(|| cols[1].to_ascii_uppercase());
            if !is_supported_exchange_atom_any_residue(res_name.as_deref(), cols[2]) {
                return Err(format!(
                    "line {line_no}: {UNSUPPORTED_EXCHANGE_ATOM_MESSAGE}"
                ));
            }
            let atom = canonicalize_exchange_atom_name(res_name.as_deref(), cols[2]);
            let dw_ppm: f64 = cols[3]
                .parse()
                .map_err(|e| format!("line {line_no}: invalid dw_ppm '{}': {e}", cols[3]))?;
            let dw_err_ppm: f64 = cols[4]
                .parse()
                .map_err(|e| format!("line {line_no}: invalid dw_err_ppm '{}': {e}", cols[4]))?;
            if !dw_ppm.is_finite() || !dw_err_ppm.is_finite() || dw_err_ppm <= 0.0 {
                return Err(format!(
                    "line {line_no}: non-finite or non-positive values in exchange row"
                ));
            }
            let sign_mode = match cols.get(5).copied() {
                Some(s) if !s.is_empty() => {
                    ExchangeSignMode::parse_label(s).map_err(|e| format!("line {line_no}: {e}"))?
                }
                _ => ExchangeSignMode::Signed,
            };
            if sign_mode == ExchangeSignMode::UnsignedMagnitude && dw_ppm < 0.0 {
                return Err(format!(
                    "line {line_no}: unsigned observation has negative dw_ppm ({dw_ppm})"
                ));
            }
            ExchangeShiftObservation {
                chain_id: "A".to_string(),
                seq_id,
                res_name,
                atom,
                dw_ppm,
                dw_err_ppm,
                sign_mode,
                major_ppm: None,
            }
        };

        if !seen_keys.insert((obs.chain_id.clone(), obs.seq_id, obs.atom.clone())) {
            return Err(format!(
                "line {line_no}: duplicate exchange observation for (chain_id='{}', seq_id={}, atom='{}')",
                obs.chain_id, obs.seq_id, obs.atom
            ));
        }

        if out.len() + controls.len() >= MAX_EXCHANGE_OBSERVATIONS {
            return Err(format!(
                "exchange dataset exceeds maximum observation limit ({MAX_EXCHANGE_OBSERVATIONS})"
            ));
        }
        if is_control {
            controls.push(obs);
        } else {
            out.push(obs);
        }
    }

    if out.is_empty() {
        return Err("no valid chemical-exchange observations found in TSV".to_string());
    }
    Ok(ParsedExchangeTable {
        observations: out,
        cpmg_prefiltered_significant_only,
        n_filtered_nonsignificant_cpmg_rows,
        control_observations: controls,
    })
}

/// Parse [`ExchangeShiftObservation`] rows from TSV text.
///
/// Accepts:
/// 1. Explicit multi-nucleus exchange tables (`[chain_id]`, `seq_id`, `[res_name]`, `atom`,
///    `dw_ppm` / `dw_ab_ppm` / `abs_dw_ppm`, `dw_err_ppm` / `dw_ppm_err` / `dw_ab_ppm_err`,
///    `[sign_mode]`, `[major_ppm]`),
/// 2. Direct `ferric cest` fit output TSVs (`dw_ab_ppm`, `dw_ab_ppm_err` -> `Signed`, `atom = "N"`
///    unless `atom` column is supplied), and
/// 3. Direct `ferric cpmg` fit output TSVs (`dw_ppm`, `dw_ppm_err`, `significant_exchange` ->
///    `UnsignedMagnitude`, filtering `significant_exchange == false` rows when that column is present).
pub fn parse_exchange_observations_tsv(
    tsv_text: &str,
) -> Result<Vec<ExchangeShiftObservation>, String> {
    parse_exchange_observations_tsv_with_meta(tsv_text).map(|p| p.observations)
}

/// Serialize [`ExchangeShiftObservation`] rows to canonical deterministic TSV format.
pub fn write_exchange_observations_tsv(obs: &[ExchangeShiftObservation]) -> String {
    let mut out = String::from(
        "chain_id\tseq_id\tres_name\tatom\tdw_ppm\tdw_err_ppm\tsign_mode\tmajor_ppm\n",
    );
    for o in obs {
        let rn = o.res_name.as_deref().unwrap_or(".");
        let maj = o
            .major_ppm
            .map(|v| format!("{v:.4}"))
            .unwrap_or_else(|| ".".to_string());
        out.push_str(&format!(
            "{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{}\t{}\n",
            o.chain_id,
            o.seq_id,
            rn,
            o.atom,
            o.dw_ppm,
            o.dw_err_ppm,
            o.sign_mode.as_str(),
            maj
        ));
    }
    out
}

/// Format an [`ExcitedStateReport`] as a deterministic human/machine-readable TSV summary.
pub fn format_excited_state_tsv(report: &ExcitedStateReport) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "# verdict: {}\treason: {}\n",
        report.verdict.as_str(),
        report.verdict_reason
    ));
    let non_null_bf = report
        .two_ln_bf_top1_vs_runnerup_non_null
        .map(|v| format!("{v:.4}"))
        .unwrap_or_else(|| "NA".to_string());
    out.push_str(&format!(
        "# n_obs: {}\tn_distinct_residues: {}\tn_distinct_exchanging_residues: {}\tsigned: {}\tunsigned: {}\t2lnBF_12: {:.4}\t2lnBF_non_null: {}\t2lnBF_null: {:.4}\ttop1_reduced_chi2: {:.4}\tcpmg_prefiltered_significant_only: {}\n",
        report.n_observations,
        report.n_distinct_residues,
        report.n_distinct_exchanging_residues,
        report.n_signed_observations,
        report.n_unsigned_observations,
        report.two_ln_bf_top1_vs_top2,
        non_null_bf,
        report.two_ln_bf_top1_vs_null,
        report.top1_reduced_chi2,
        report.cpmg_prefiltered_significant_only,
    ));
    out.push_str(&format!(
        "# ground_exclusion_list_sha256: {}\tmax_retained_seq_identity: {:.4}\n",
        report.ground_exclusion_list_sha256, report.ground_max_retained_seq_identity
    ));
    if let Some(ref diag) = report.chi2_consistency_diagnostic {
        out.push_str(&format!(
            "# chi2_consistency_diagnostic: h0_rchi2={:.4}\ttop1_rchi2={:.4}\ttop1_dof={}\tbound={:.4}\timplied_scale_h0={:.2}x\timplied_scale_top1={:.2}x\tinlier_rchi2={}\tcontrol_rchi2={}\tevidence={}\terror_scale_inconsistent={}\n",
            diag.h0_reduced_chi2,
            diag.top1_reduced_chi2,
            diag.top1_dof,
            diag.consistency_upper_bound,
            diag.implied_error_scale_factor_h0,
            diag.implied_error_scale_factor_top1,
            diag.inlier_reduced_chi2
                .map(|v| format!("{v:.4}"))
                .unwrap_or_else(|| "NA".to_string()),
            diag.control_reduced_chi2
                .map(|v| format!("{v:.4}"))
                .unwrap_or_else(|| "NA".to_string()),
            diag.error_scale_evidence,
            diag.error_scale_inconsistent
        ));
    }
    if let Some(ref warn) = report.error_scale_warning {
        out.push_str(&format!("# error_scale_warning: {warn}\n"));
    }
    out.push_str(
        "rank\thypothesis_code\thypothesis\tk_params\tm_cands\tmax_log_lik\tmarginal_log_lik\tbic\tdelta_bic\t2ln_bf_vs_null\tposterior_weight\treduced_chi2\trmse_ppm\tpearson_r\tspearman_rho\tlocus_summary\n",
    );
    for (idx, h) in report.ranked_hypotheses.iter().enumerate() {
        let pr = h
            .pearson_r
            .map(|v| format!("{v:.4}"))
            .unwrap_or_else(|| "NA".to_string());
        let sr = h
            .spearman_rho
            .map(|v| format!("{v:.4}"))
            .unwrap_or_else(|| "NA".to_string());
        out.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.6}\t{:.4}\t{:.4}\t{}\t{}\t{}\n",
            idx + 1,
            h.kind.short_code(),
            h.kind.as_str(),
            h.num_params_k,
            h.n_candidates_evaluated,
            h.max_log_likelihood,
            h.marginal_log_likelihood,
            h.bic,
            h.delta_bic_from_best,
            h.two_ln_bf_vs_null,
            h.posterior_weight,
            h.reduced_chi2,
            h.rmse_ppm,
            pr,
            sr,
            h.best_locus_summary
        ));
    }
    out
}

/// Format an [`ExcitedStateReport`] as pretty-printed JSON.
pub fn format_excited_state_json(report: &ExcitedStateReport) -> Result<String, String> {
    serde_json::to_string_pretty(report)
        .map_err(|e| format!("failed to serialize ExcitedStateReport JSON: {e}"))
}

// ============================================================================
// 3. Internal Observation Matching, Coil/Aromatic Physics & Correlation Helpers
// ============================================================================

#[derive(Clone, Debug)]
struct ResolvedObs {
    res_idx: usize,
    chain_id: String,
    seq_id: i32,
    res_name: String,
    atom: String,
    sign_mode: ExchangeSignMode,
    dw_ppm: f64,
    dw_err_ppm: f64,
    sigma_fwd_ppm: f64,
    sigma_tot_ppm: f64,
    ground_ppm: f64,
    /// Experimental ground-state shift (`obs.major_ppm` or `ground_shifts`), `None` if fallback-only.
    explicit_ground_ppm: Option<f64>,
    /// Predicted unblended ground-state 3D shift `delta^{struct}_A` (when modeled by `T21.2`).
    pred_a_struct_ppm: Option<f64>,
    /// Sequence/pH/T-corrected random-coil target `delta^{RC}` for `H1: LocalUnfolding`.
    coil_target_ppm: f64,
    /// Representative 3D coordinate(s) for distance cutoffs and Haigh-Mallion ring-current shifts.
    spin_coords: Vec<Vec3>,
}

fn lookup_reference_shift(
    ref_data: &Reference,
    ref_idx: usize,
    res_name: &str,
    canon_atom: &str,
) -> Option<f64> {
    if let Some(&v) = ref_data.shifts.get(&(ref_idx, canon_atom.to_string())) {
        if v.is_finite() {
            return Some(v);
        }
    }
    match canon_atom {
        "H" => ref_data.shifts.get(&(ref_idx, "HN".to_string())).copied(),
        "C" => ref_data.shifts.get(&(ref_idx, "CO".to_string())).copied(),
        "HA" => match (
            ref_data.shifts.get(&(ref_idx, "HA2".to_string())).copied(),
            ref_data.shifts.get(&(ref_idx, "HA3".to_string())).copied(),
        ) {
            (Some(a), Some(b)) => Some(0.5 * (a + b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => ref_data.shifts.get(&(ref_idx, "HA1".to_string())).copied(),
        },
        "HD1*" if matches!(res_name, "ILE" | "LEU") => ["HD11", "HD12", "HD13", "HD1"]
            .iter()
            .find_map(|k| ref_data.shifts.get(&(ref_idx, (*k).to_string())).copied()),
        "HD2*" if res_name == "LEU" => ["HD21", "HD22", "HD23", "HD2"]
            .iter()
            .find_map(|k| ref_data.shifts.get(&(ref_idx, (*k).to_string())).copied()),
        "HG1*" if res_name == "VAL" => ["HG11", "HG12", "HG13", "HG1"]
            .iter()
            .find_map(|k| ref_data.shifts.get(&(ref_idx, (*k).to_string())).copied()),
        "HG2*" if matches!(res_name, "VAL" | "THR") => ["HG21", "HG22", "HG23", "HG2"]
            .iter()
            .find_map(|k| ref_data.shifts.get(&(ref_idx, (*k).to_string())).copied()),
        "HE*" if res_name == "MET" => ["HE1", "HE2", "HE3", "HE"]
            .iter()
            .find_map(|k| ref_data.shifts.get(&(ref_idx, (*k).to_string())).copied()),
        "HB*" if res_name == "ALA" => ["HB1", "HB2", "HB3", "HB"]
            .iter()
            .find_map(|k| ref_data.shifts.get(&(ref_idx, (*k).to_string())).copied()),
        _ => None,
    }
    .filter(|v| v.is_finite())
}

fn lookup_pred_struct_and_rc(
    pred_report: &StructShiftPredictionReport,
    res_idx: usize,
    canon_atom: &str,
) -> (Option<f64>, Option<f64>) {
    let rp = &pred_report.residues[res_idx];
    if let Some(bb) = rp.backbone.get(canon_atom) {
        return (Some(bb.struct_ppm), Some(bb.rc_ppm));
    }
    if canon_atom == "HA2" || canon_atom == "HA3" {
        if let Some(bb) = rp.backbone.get("HA") {
            return (Some(bb.struct_ppm), Some(bb.rc_ppm));
        }
    }
    for m in &rp.methyls {
        if m.carbon_atom == canon_atom {
            return (Some(m.c_struct_ppm), Some(m.c_rc_ppm));
        }
        if m.proton_atom == canon_atom {
            return (Some(m.h_struct_ppm), Some(m.h_rc_ppm));
        }
    }
    (None, None)
}

/// Unfolded coil shift target for non-backbone / non-ILVMAT sidechain nuclei (`Pro CG/CD`, `His`
/// pH-dependent tautomer equilibrium, aromatic ring `CH`).
fn fallback_sidechain_coil_ppm(
    res_name: &str,
    canon_atom: &str,
    ground_ppm: Option<f64>,
    ground_omega_deg: Option<f64>,
    sample_ph: f64,
) -> f64 {
    if res_name == "PRO" {
        let ground_is_cis = ground_omega_deg.is_some_and(|w| w.abs() < 90.0);
        match canon_atom {
            "CG" => {
                // Unfolded Xxx-Pro equilibrates to ~90% trans (CG ~ 27.40) / 10% cis (CG ~ 24.80) -> ~27.14 ppm
                if ground_is_cis {
                    return ground_ppm.map(|g| g + 2.34).unwrap_or(27.14);
                }
                return 27.40;
            }
            "CD" => return 50.35,
            _ => {}
        }
    }
    if res_name == "HIS" {
        let eff_ph = sample_ph.clamp(1.0, 13.0);
        let f_prot = 1.0 / (1.0 + det_exp(std::f64::consts::LN_10 * (eff_ph - 6.35)));
        let mix_his = |tau: f64, pi_s: f64, cat: f64| -> f64 {
            f_prot * cat + (1.0 - f_prot) * (0.80 * tau + 0.20 * pi_s)
        };
        match canon_atom {
            "ND1" => {
                return mix_his(
                    HIS_NE2H_PRIORS.nd1.0,
                    HIS_ND1H_PRIORS.nd1.0,
                    HIS_DOUBLY_PROTONATED_PRIORS.nd1.0,
                )
            }
            "NE2" => {
                return mix_his(
                    HIS_NE2H_PRIORS.ne2.0,
                    HIS_ND1H_PRIORS.ne2.0,
                    HIS_DOUBLY_PROTONATED_PRIORS.ne2.0,
                )
            }
            "CG" => {
                return mix_his(
                    HIS_NE2H_PRIORS.cg.0,
                    HIS_ND1H_PRIORS.cg.0,
                    HIS_DOUBLY_PROTONATED_PRIORS.cg.0,
                )
            }
            "CD2" => {
                return mix_his(
                    HIS_NE2H_PRIORS.cd2.0,
                    HIS_ND1H_PRIORS.cd2.0,
                    HIS_DOUBLY_PROTONATED_PRIORS.cd2.0,
                )
            }
            "CE1" => {
                return mix_his(
                    HIS_NE2H_PRIORS.ce1.0,
                    HIS_ND1H_PRIORS.ce1.0,
                    HIS_DOUBLY_PROTONATED_PRIORS.ce1.0,
                )
            }
            "HD2" => {
                return mix_his(
                    HIS_NE2H_PRIORS.hd2.0,
                    HIS_ND1H_PRIORS.hd2.0,
                    HIS_DOUBLY_PROTONATED_PRIORS.hd2.0,
                )
            }
            "HE1" => {
                return mix_his(
                    HIS_NE2H_PRIORS.he1.0,
                    HIS_ND1H_PRIORS.he1.0,
                    HIS_DOUBLY_PROTONATED_PRIORS.he1.0,
                )
            }
            _ => {}
        }
    }
    ground_ppm.unwrap_or(0.0)
}

fn resolve_spin_coords(model: &StructureModel, res_idx: usize, canon_atom: &str) -> Vec<Vec3> {
    let r = &model.residues[res_idx];
    match canon_atom {
        "H" => r.h.into_iter().collect(),
        "N" => vec![r.n],
        "CA" => vec![r.ca],
        "CB" => r.cb.into_iter().collect(),
        "C" => vec![r.c],
        "HA" => vec![r.ha],
        "HD1*" if matches!(r.res_name.as_str(), "ILE" | "LEU") => ["HD11", "HD12", "HD13"]
            .iter()
            .filter_map(|an| r.atoms.get(*an).copied())
            .collect(),
        "HD2*" if r.res_name == "LEU" => ["HD21", "HD22", "HD23"]
            .iter()
            .filter_map(|an| r.atoms.get(*an).copied())
            .collect(),
        "HG1*" if r.res_name == "VAL" => ["HG11", "HG12", "HG13"]
            .iter()
            .filter_map(|an| r.atoms.get(*an).copied())
            .collect(),
        "HG2*" if matches!(r.res_name.as_str(), "VAL" | "THR") => ["HG21", "HG22", "HG23"]
            .iter()
            .filter_map(|an| r.atoms.get(*an).copied())
            .collect(),
        "HE*" if r.res_name == "MET" => ["HE1", "HE2", "HE3"]
            .iter()
            .filter_map(|an| r.atoms.get(*an).copied())
            .collect(),
        "HB*" if r.res_name == "ALA" => ["HB1", "HB2", "HB3"]
            .iter()
            .filter_map(|an| r.atoms.get(*an).copied())
            .collect(),
        other => r
            .atoms
            .get(other)
            .copied()
            .or_else(|| r.cb.or(Some(r.ca)))
            .into_iter()
            .collect(),
    }
}

fn pearson_correlation(xs: &[f64], ys: &[f64]) -> Option<f64> {
    let n = xs.len();
    if n < 3 || ys.len() != n {
        return None;
    }
    let inv_n = 1.0 / (n as f64);
    let mx = xs.iter().sum::<f64>() * inv_n;
    let my = ys.iter().sum::<f64>() * inv_n;
    let mut cov = 0.0;
    let mut vx = 0.0;
    let mut vy = 0.0;
    for i in 0..n {
        let dx = xs[i] - mx;
        let dy = ys[i] - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    if vx <= 1e-12 || vy <= 1e-12 {
        return None;
    }
    Some((cov / (vx * vy).sqrt()).clamp(-1.0, 1.0))
}

fn average_ranks(vals: &[f64]) -> Vec<f64> {
    let n = vals.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| vals[a].total_cmp(&vals[b]));
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && (vals[order[j]] - vals[order[i]]).abs() <= 1e-12 {
            j += 1;
        }
        let avg = 0.5 * ((i + j - 1) as f64) + 1.0;
        for &idx in &order[i..j] {
            ranks[idx] = avg;
        }
        i = j;
    }
    ranks
}

fn spearman_correlation(xs: &[f64], ys: &[f64]) -> Option<f64> {
    if xs.len() < 3 || xs.len() != ys.len() {
        return None;
    }
    let rx = average_ranks(xs);
    let ry = average_ranks(ys);
    pearson_correlation(&rx, &ry)
}

/// Numerically stable marginalized log-likelihood `ln( (1/M) * sum_{m=1}^M exp(ll_m) )`
/// via log-sum-exp, naturally penalizing scanning `M` discrete candidates by `-ln(M)`.
#[inline]
fn log_mean_exp(lls: &[f64]) -> f64 {
    if lls.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max_ll = lls.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max_ll.is_finite() {
        return max_ll;
    }
    let sum_exp: f64 = lls.iter().map(|&ll| (ll - max_ll).exp()).sum();
    max_ll + sum_exp.ln() - (lls.len() as f64).ln()
}

/// Effective prediction-space scale `sigma(mu)`: equals experimental `dw_err_ppm` at `mu = 0`
/// (spectators and `H0`) and smoothly ramps to `sqrt(dw_err^2 + sigma_fwd^2)` when `|mu| > 0`,
/// eliminating artificial `ln(sigma_fwd / sigma_exp)` bias on unperturbed spectator residues.
#[inline]
fn effective_sigma_for_prediction(obs: &ResolvedObs, mu_pred: f64, is_null: bool) -> f64 {
    if is_null || obs.sigma_fwd_ppm <= 0.0 {
        return obs.dw_err_ppm;
    }
    let abs_mu = mu_pred.abs();
    if abs_mu <= 1e-12 {
        return obs.dw_err_ppm;
    }
    let ramp = abs_mu / (abs_mu * abs_mu + (0.25 * obs.sigma_fwd_ppm).powi(2)).sqrt();
    let sigma_fwd_eff = obs.sigma_fwd_ppm * ramp;
    let _ = obs.sigma_tot_ppm;
    (obs.dw_err_ppm * obs.dw_err_ppm + sigma_fwd_eff * sigma_fwd_eff).sqrt()
}

#[inline]
fn eval_single_obs_ll_and_z_with_sigma(
    obs: &ResolvedObs,
    mu_pred: f64,
    sigma: f64,
) -> Result<(f64, f64, f64), String> {
    match obs.sign_mode {
        ExchangeSignMode::Signed => {
            let ll = signed_student_t5_log_likelihood(obs.dw_ppm, mu_pred, sigma)?;
            let z = (obs.dw_ppm - mu_pred) / sigma;
            Ok((ll, mu_pred, z))
        }
        ExchangeSignMode::UnsignedMagnitude => {
            let ll = unsigned_folded_student_t5_log_likelihood(obs.dw_ppm, mu_pred, sigma)?;
            let eff = mu_pred.abs();
            let z = (obs.dw_ppm - eff) / sigma;
            Ok((ll, eff, z))
        }
    }
}

#[inline]
fn eval_single_obs_ll_and_z(obs: &ResolvedObs, mu_pred: f64) -> Result<(f64, f64, f64), String> {
    let sigma = effective_sigma_for_prediction(obs, mu_pred, false);
    eval_single_obs_ll_and_z_with_sigma(obs, mu_pred, sigma)
}

struct BestCandidateState {
    log_likelihood: f64,
    mu_preds: Vec<f64>,
    summary: String,
    affected_seq_ids: Vec<i32>,
    parameters: HypothesisBestParameters,
}

type RotatedAromaticGeometry = (Vec<ConjugatedAromaticRing>, Vec<(&'static str, Vec3)>);

#[allow(clippy::too_many_arguments)]
fn build_hypothesis_score(
    kind: ExcitedStateHypothesisKind,
    num_params_k: usize,
    n_distinct_residues: usize,
    marginal_log_likelihood: f64,
    n_candidates_evaluated: usize,
    resolved: &[ResolvedObs],
    winner: BestCandidateState,
    top_sub_candidates: Vec<SubCandidateSummary>,
) -> Result<HypothesisScore, String> {
    let is_null = kind == ExcitedStateHypothesisKind::NullOrUnstructuredPerturbation;
    let n_obs = resolved.len();
    let mut total_ll = 0.0;
    let mut sum_z2 = 0.0;
    let mut sum_sq_err = 0.0;
    let mut obs_eff = Vec::with_capacity(n_obs);
    let mut pred_eff = Vec::with_capacity(n_obs);
    let mut predictions = Vec::with_capacity(n_obs);

    for (obs, &mu) in resolved.iter().zip(winner.mu_preds.iter()) {
        let sigma = effective_sigma_for_prediction(obs, mu, is_null);
        let (ll, eff_mu, z) = eval_single_obs_ll_and_z_with_sigma(obs, mu, sigma)?;
        total_ll += ll;
        sum_z2 += z * z;
        let raw_resid = obs.dw_ppm - eff_mu;
        sum_sq_err += raw_resid * raw_resid;
        obs_eff.push(obs.dw_ppm);
        pred_eff.push(eff_mu);
        predictions.push(ObservationPredictionRow {
            chain_id: obs.chain_id.clone(),
            seq_id: obs.seq_id,
            res_name: obs.res_name.clone(),
            atom: obs.atom.clone(),
            sign_mode: obs.sign_mode,
            ground_ppm: obs.ground_ppm,
            observed_dw_ppm: obs.dw_ppm,
            observed_err_ppm: obs.dw_err_ppm,
            predicted_dw_ppm: mu,
            predicted_effective_dw_ppm: eff_mu,
            sigma_tot_ppm: sigma,
            log_likelihood: ll,
            z_residual: z,
        });
    }

    let eff_marginal = if is_null || n_candidates_evaluated <= 1 {
        total_ll
    } else {
        marginal_log_likelihood
    };
    let dof = n_obs.saturating_sub(num_params_k).max(1) as f64;
    let reduced_chi2 = sum_z2 / dof;
    let rmse_ppm = (sum_sq_err / (n_obs.max(1) as f64)).sqrt();
    let bic =
        -2.0 * eff_marginal + (num_params_k as f64) * (n_distinct_residues.max(1) as f64).ln();

    Ok(HypothesisScore {
        kind,
        num_params_k,
        n_obs,
        n_distinct_residues,
        n_candidates_evaluated: n_candidates_evaluated.max(1),
        max_log_likelihood: total_ll,
        marginal_log_likelihood: eff_marginal,
        bic,
        posterior_weight: 0.0,
        two_ln_bf_vs_runner_up: None,
        two_ln_bf_vs_best_other_non_null: None,
        delta_bic_from_best: 0.0,
        two_ln_bf_vs_null: 0.0,
        reduced_chi2,
        rmse_ppm,
        pearson_r: pearson_correlation(&obs_eff, &pred_eff),
        spearman_rho: spearman_correlation(&obs_eff, &pred_eff),
        best_locus_summary: winner.summary,
        affected_seq_ids: winner.affected_seq_ids,
        parameters: winner.parameters,
        predictions,
        top_sub_candidates,
    })
}

// ============================================================================
// 4. Biophysical Forward Hypothesis Generators (`H1`, `H2`, `H3`, `H4`)
// ============================================================================

/// Evaluate `H1: LocalUnfolding` (`k = 2`) across all contiguous intra-chain windows `[u, v]`
/// covering at least one observed residue, with `1`-residue `0.5` boundary damping at `u - 1`
/// and `v + 1`.
fn evaluate_h1_local_unfolding(
    model: &StructureModel,
    resolved: &[ResolvedObs],
    n_distinct_residues: usize,
) -> Result<Option<HypothesisScore>, String> {
    let l = model.residues.len();
    if l == 0 || resolved.is_empty() {
        return Ok(None);
    }

    // Group observations by residue index `0..L`
    let mut obs_by_res: Vec<Vec<usize>> = vec![Vec::new(); l];
    let mut has_obs = vec![false; l];
    for (oi, o) in resolved.iter().enumerate() {
        obs_by_res[o.res_idx].push(oi);
        has_obs[o.res_idx] = true;
    }

    // Precompute per-residue log-likelihood and chi^2 contributions at weights `0.0`, `0.5`, `1.0`
    let mut ll_w0 = vec![0.0_f64; l];
    let mut ll_w05 = vec![0.0_f64; l];
    let mut ll_w1 = vec![0.0_f64; l];
    let mut z2_w0 = vec![0.0_f64; l];
    let mut z2_w05 = vec![0.0_f64; l];
    let mut z2_w1 = vec![0.0_f64; l];

    for r_idx in 0..l {
        for &oi in &obs_by_res[r_idx] {
            let o = &resolved[oi];
            let full_delta = o.coil_target_ppm - o.ground_ppm;
            let (l0, _, z0) = eval_single_obs_ll_and_z(o, 0.0)?;
            let (lh, _, zh) = eval_single_obs_ll_and_z(o, 0.5 * full_delta)?;
            let (lf, _, zf) = eval_single_obs_ll_and_z(o, full_delta)?;
            ll_w0[r_idx] += l0;
            ll_w05[r_idx] += lh;
            ll_w1[r_idx] += lf;
            z2_w0[r_idx] += z0 * z0;
            z2_w05[r_idx] += zh * zh;
            z2_w1[r_idx] += zf * zf;
        }
    }

    let base_ll: f64 = ll_w0.iter().sum();
    let base_z2: f64 = z2_w0.iter().sum();
    let mut pref_dll_full = vec![0.0_f64; l + 1];
    let mut pref_dz2_full = vec![0.0_f64; l + 1];
    let mut pref_obs_cnt = vec![0usize; l + 1];
    for i in 0..l {
        pref_dll_full[i + 1] = pref_dll_full[i] + (ll_w1[i] - ll_w0[i]);
        pref_dz2_full[i + 1] = pref_dz2_full[i] + (z2_w1[i] - z2_w0[i]);
        pref_obs_cnt[i + 1] = pref_obs_cnt[i] + usize::from(has_obs[i]);
    }

    let k_h = ExcitedStateHypothesisKind::LocalUnfolding.default_k();
    let ln_n = (n_distinct_residues.max(1) as f64).ln();
    let dof = resolved.len().saturating_sub(k_h).max(1) as f64;

    // Partition model into contiguous same-chain segments (`seq_id` consecutive)
    let mut seg_start = 0usize;
    let mut best_uv: Option<(usize, usize, f64)> = None;
    let mut cand_lls: Vec<f64> = Vec::new();
    let mut sub_cands: Vec<(f64, SubCandidateSummary)> = Vec::new();

    while seg_start < l {
        let mut seg_end = seg_start;
        while seg_end + 1 < l
            && model.residues[seg_end + 1].chain_id == model.residues[seg_end].chain_id
            && model.residues[seg_end + 1].seq_id == model.residues[seg_end].seq_id + 1
        {
            seg_end += 1;
        }

        for u in seg_start..=seg_end {
            for v in u..=seg_end {
                if pref_obs_cnt[v + 1] == pref_obs_cnt[u] {
                    continue;
                }
                let mut ll = base_ll + (pref_dll_full[v + 1] - pref_dll_full[u]);
                let mut z2 = base_z2 + (pref_dz2_full[v + 1] - pref_dz2_full[u]);
                if u > seg_start {
                    ll += ll_w05[u - 1] - ll_w0[u - 1];
                    z2 += z2_w05[u - 1] - z2_w0[u - 1];
                }
                if v < seg_end {
                    ll += ll_w05[v + 1] - ll_w0[v + 1];
                    z2 += z2_w05[v + 1] - z2_w0[v + 1];
                }

                cand_lls.push(ll);
                if best_uv.is_none_or(|(_, _, b_ll)| ll > b_ll) {
                    best_uv = Some((u, v, ll));
                }
                let bic = -2.0 * ll + (k_h as f64) * ln_n;
                sub_cands.push((
                    ll,
                    SubCandidateSummary {
                        label: format!(
                            "chain {} {}{}..{}{}",
                            model.residues[u].chain_id,
                            model.residues[u].res_name,
                            model.residues[u].seq_id,
                            model.residues[v].res_name,
                            model.residues[v].seq_id
                        ),
                        log_likelihood: ll,
                        bic,
                        reduced_chi2: z2 / dof,
                        affected_seq_ids: (u..=v).map(|idx| model.residues[idx].seq_id).collect(),
                    },
                ));
            }
        }

        seg_start = seg_end + 1;
    }

    let Some((u_best, v_best, _)) = best_uv else {
        return Ok(None);
    };

    let marginal_ll = log_mean_exp(&cand_lls);
    let n_cands = cand_lls.len();
    sub_cands.sort_by(|a, b| b.0.total_cmp(&a.0));
    let top_sub: Vec<SubCandidateSummary> = sub_cands.into_iter().take(5).map(|x| x.1).collect();

    let u_chain = &model.residues[u_best].chain_id;
    let left_bound = (u_best > 0
        && model.residues[u_best - 1].chain_id == *u_chain
        && model.residues[u_best - 1].seq_id + 1 == model.residues[u_best].seq_id)
        .then(|| u_best - 1);
    let right_bound = (v_best + 1 < l
        && model.residues[v_best + 1].chain_id == *u_chain
        && model.residues[v_best].seq_id + 1 == model.residues[v_best + 1].seq_id)
        .then(|| v_best + 1);

    let mu_preds: Vec<f64> = resolved
        .iter()
        .map(|o| {
            let full = o.coil_target_ppm - o.ground_ppm;
            if (u_best..=v_best).contains(&o.res_idx) {
                full
            } else if Some(o.res_idx) == left_bound || Some(o.res_idx) == right_bound {
                0.5 * full
            } else {
                0.0
            }
        })
        .collect();

    let mut affected: Vec<i32> = (u_best..=v_best)
        .map(|idx| model.residues[idx].seq_id)
        .collect();
    if let Some(lb) = left_bound {
        affected.insert(0, model.residues[lb].seq_id);
    }
    if let Some(rb) = right_bound {
        affected.push(model.residues[rb].seq_id);
    }

    let summary = format!(
        "LocalUnfolding chain {} {}{}..{}{} ({} core residues, +/-1 boundary 0.5 damping)",
        model.residues[u_best].chain_id,
        model.residues[u_best].res_name,
        model.residues[u_best].seq_id,
        model.residues[v_best].res_name,
        model.residues[v_best].seq_id,
        v_best - u_best + 1
    );
    let params = HypothesisBestParameters::LocalUnfolding {
        chain_id: model.residues[u_best].chain_id.clone(),
        start_seq_id: model.residues[u_best].seq_id,
        end_seq_id: model.residues[v_best].seq_id,
        start_res_name: model.residues[u_best].res_name.clone(),
        end_res_name: model.residues[v_best].res_name.clone(),
        core_residue_count: v_best - u_best + 1,
    };

    let score = build_hypothesis_score(
        ExcitedStateHypothesisKind::LocalUnfolding,
        k_h,
        n_distinct_residues,
        marginal_ll,
        n_cands,
        resolved,
        BestCandidateState {
            log_likelihood: 0.0,
            mu_preds,
            summary,
            affected_seq_ids: affected,
            parameters: params,
        },
        top_sub,
    )?;
    Ok(Some(score))
}

/// Compute local Wagner-Pardi-Bax H-bond + McConnell carbonyl anisotropy perturbation at residue
/// `target_idx` arising specifically from carbonyl group `c_pos = o_pos` of residue `pep_idx`.
fn eval_single_carbonyl_hb_and_aniso(
    model: &StructureModel,
    pep_idx: usize,
    o_pos: Vec3,
    target_idx: usize,
    atom: &str,
) -> f64 {
    if pep_idx == target_idx {
        return 0.0;
    }
    let rp = &model.residues[pep_idx];
    let rt = &model.residues[target_idx];
    let c_pos = rp.c;
    const INV_R0_CUBE: f64 = 1.0 / (2.60 * 2.60 * 2.60);

    match atom {
        "H" | "N" => {
            let Some(h_t) = rt.h else {
                return 0.0;
            };
            let mut hb = 0.0;
            let same_chain = rp.chain_id == rt.chain_id;
            let seq_sep = (rt.seq_id - rp.seq_id).abs();
            if !(same_chain && seq_sep < 2) {
                let d_oh = v_dist(o_pos, h_t);
                if (1.55..=3.20).contains(&d_oh) {
                    let u_hn = v_unit(v_sub(rt.n, h_t));
                    let u_ho = v_unit(v_sub(o_pos, h_t));
                    let cos_ang = v_dot(u_hn, u_ho);
                    if cos_ang <= -0.34202014 {
                        let excess = (1.0 / d_oh.max(1.65).powi(3) - INV_R0_CUBE).max(0.0);
                        let term = excess * cos_ang * cos_ang;
                        hb = if atom == "H" {
                            (4.15 * term).min(1.35)
                        } else {
                            (8.80 * term).min(3.00)
                        };
                    }
                }
            }
            if atom == "N" {
                return hb;
            }
            // McConnell C=O dipole anisotropy on H^N
            let cen = v_add(v_scale(c_pos, 0.45), v_scale(o_pos, 0.55));
            let dv = v_sub(h_t, cen);
            let r = (dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2])
                .sqrt()
                .max(2.10);
            if r > 8.0 {
                return hb;
            }
            let e1 = v_unit(v_sub(o_pos, c_pos));
            let in_pl = v_sub(rp.ca, c_pos);
            let e3_raw = v_cross(e1, in_pl);
            let e3 = v_unit(e3_raw);
            let c3 = v_dot(dv, e3) / r;
            let aniso = 0.28 * (1.0 - 3.0 * c3 * c3) / (r * r * r);
            hb + aniso
        }
        "HA" => {
            let cen = v_add(v_scale(c_pos, 0.45), v_scale(o_pos, 0.55));
            let dv = v_sub(rt.ha, cen);
            let r = (dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2])
                .sqrt()
                .max(2.10);
            if r > 8.0 {
                return 0.0;
            }
            let e1 = v_unit(v_sub(o_pos, c_pos));
            let e3 = v_unit(v_cross(e1, v_sub(rp.ca, c_pos)));
            let c3 = v_dot(dv, e3) / r;
            0.28 * (1.0 - 3.0 * c3 * c3) / (r * r * r)
        }
        _ => 0.0,
    }
}

/// Evaluate `H2: XxxProCisTrans` (`k = 1`, Schubert et al. 2002; Shen & Bax 2010) across all
/// Proline residues within `<= 8.0 A` (`CA-CA`) or `|i - p| <= 3` of exchanging observations.
fn evaluate_h2_proline_cis_trans(
    model: &StructureModel,
    ground_shifts: Option<&Reference>,
    ref_map_by_res_idx: &[Option<usize>],
    resolved: &[ResolvedObs],
    n_distinct_residues: usize,
) -> Result<Option<HypothesisScore>, String> {
    let l = model.residues.len();
    if l == 0 || resolved.is_empty() {
        return Ok(None);
    }

    let k_h = ExcitedStateHypothesisKind::XxxProCisTrans.default_k();
    let ln_n = (n_distinct_residues.max(1) as f64).ln();
    let dof = resolved.len().saturating_sub(k_h).max(1) as f64;

    let mut best_cand: Option<BestCandidateState> = None;
    let mut cand_lls: Vec<f64> = Vec::new();
    let mut sub_cands: Vec<(f64, SubCandidateSummary)> = Vec::new();

    for (p_idx, &ref_idx_opt) in ref_map_by_res_idx.iter().enumerate().take(l) {
        let rp = &model.residues[p_idx];
        if rp.res_name != "PRO" {
            continue;
        }
        // Evaluate ALL Prolines in the structure (`M_H2 = N_Pro`) without data-peeking pre-filtering
        // so discrete-scan multiplicity `ln(1/M_H2 * sum_m L_m)` honestly penalizes scanning `N_Pro` loci.

        let prev_intact = p_idx > 0
            && model.residues[p_idx - 1].chain_id == rp.chain_id
            && model.residues[p_idx - 1].seq_id + 1 == rp.seq_id
            && v_dist(model.residues[p_idx - 1].c, rp.n) <= 2.50;
        let next_intact = p_idx + 1 < l
            && model.residues[p_idx + 1].chain_id == rp.chain_id
            && rp.seq_id + 1 == model.residues[p_idx + 1].seq_id
            && v_dist(rp.c, model.residues[p_idx + 1].n) <= 2.50;

        // Infer ground-state trans vs cis from `omega_p` or ground `CB - CG` shift invariant (`> 7.15 ppm` -> cis)
        let ground_bg = ground_shifts.and_then(|gs| {
            let ri = ref_idx_opt?;
            let cb = lookup_reference_shift(gs, ri, "PRO", "CB")?;
            let cg = lookup_reference_shift(gs, ri, "PRO", "CG")?;
            Some(cb - cg)
        });
        let ground_is_cis = match rp.omega_deg {
            Some(w) => w.abs() < 90.0,
            None => ground_bg.is_some_and(|bg| bg > 7.15),
        };
        let direction = if ground_is_cis {
            ProCisTransDirection::CisToTrans
        } else {
            ProCisTransDirection::TransToCis
        };
        let s_iso = direction.sign_factor();

        // Compute rotated `O_{p-1}` coordinate (180 deg across the Xxx_{p-1}-Pro_p peptide bond)
        let o_prev_pair = prev_intact.then(|| {
            let c_prev = model.residues[p_idx - 1].c;
            let ca_prev = model.residues[p_idx - 1].ca;
            let o_orig = model.residues[p_idx - 1].o;
            let o_flip = nerf_place_atom(
                ca_prev,
                rp.n,
                c_prev,
                BOND_C_O,
                123.0_f64.to_radians(),
                if ground_is_cis { 0.0 } else { PI },
            );
            (p_idx - 1, o_orig, o_flip)
        });

        let mut mu_preds = Vec::with_capacity(resolved.len());
        let mut affected_set = BTreeSet::new();
        affected_set.insert(rp.seq_id);
        if prev_intact {
            affected_set.insert(model.residues[p_idx - 1].seq_id);
        }
        if next_intact {
            affected_set.insert(model.residues[p_idx + 1].seq_id);
        }

        for o in resolved {
            let ro = &model.residues[o.res_idx];
            let same_chain = ro.chain_id == rp.chain_id;
            let seq_diff = ro.seq_id - rp.seq_id;
            let ca_dist = v_dist(ro.ca, rp.ca);
            if !(same_chain && seq_diff.abs() <= PROLINE_SEQ_WINDOW)
                && ca_dist > PROLINE_SPATIAL_CUTOFF_ANGSTROM
            {
                mu_preds.push(0.0);
                continue;
            }

            let d_hb_pep = match o_prev_pair {
                Some((pep_i, o_orig, o_flip)) => {
                    eval_single_carbonyl_hb_and_aniso(model, pep_i, o_flip, o.res_idx, &o.atom)
                        - eval_single_carbonyl_hb_and_aniso(
                            model, pep_i, o_orig, o.res_idx, &o.atom,
                        )
                }
                None => 0.0,
            };

            let mu = if o.res_idx == p_idx {
                // Canonical Schubert et al. (2002) & Shen & Bax (2010) Proline p isomerization shifts
                match o.atom.as_str() {
                    "CB" => s_iso * 2.40,
                    "CG" => s_iso * (-2.60),
                    "CA" => s_iso * (-0.55),
                    "C" => s_iso * (-0.85),
                    "CD" => s_iso * 0.45,
                    "HA" => s_iso * 0.28,
                    _ => d_hb_pep,
                }
            } else if prev_intact && o.res_idx + 1 == p_idx {
                // Preceding residue Xxx_{p-1}
                match o.atom.as_str() {
                    "CA" => s_iso * (-1.15),
                    "C" => s_iso * (-1.35),
                    "CB" => s_iso * 0.35,
                    "HA" => s_iso * 0.32,
                    "N" => s_iso * (-1.10) + d_hb_pep,
                    "H" => s_iso * 0.18 + d_hb_pep,
                    _ => d_hb_pep,
                }
            } else if next_intact && o.res_idx == p_idx + 1 {
                // Following residue p + 1 amide & backbone peptide-anisotropy / H-bond switch
                match o.atom.as_str() {
                    "N" => s_iso * (-1.85) + d_hb_pep,
                    "H" => s_iso * (-0.32) + d_hb_pep,
                    "CA" => s_iso * (-0.35),
                    "C" => s_iso * (-0.25) + d_hb_pep,
                    "HA" => s_iso * (-0.12) + d_hb_pep,
                    _ => d_hb_pep,
                }
            } else if same_chain && seq_diff.abs() == 2 {
                let base = match o.atom.as_str() {
                    "N" => s_iso * (-0.45),
                    "H" => s_iso * (-0.08),
                    "CA" | "C" => s_iso * (-0.20),
                    _ => 0.0,
                };
                base + d_hb_pep
            } else {
                d_hb_pep
            };

            if mu.abs() >= 0.05 {
                affected_set.insert(ro.seq_id);
            }
            mu_preds.push(mu);
        }

        let mut ll = 0.0;
        let mut z2 = 0.0;
        for (o, &mu) in resolved.iter().zip(mu_preds.iter()) {
            let (l_i, _, z_i) = eval_single_obs_ll_and_z(o, mu)?;
            ll += l_i;
            z2 += z_i * z_i;
        }
        cand_lls.push(ll);
        let bic = -2.0 * ll + (k_h as f64) * ln_n;
        let affected: Vec<i32> = affected_set.into_iter().collect();

        let prev_label = if prev_intact {
            format!(
                "{}{}-",
                model.residues[p_idx - 1].res_name,
                model.residues[p_idx - 1].seq_id
            )
        } else {
            String::new()
        };
        let label = format!(
            "chain {} {}PRO{} ({})",
            rp.chain_id,
            prev_label,
            rp.seq_id,
            direction.as_str()
        );
        sub_cands.push((
            ll,
            SubCandidateSummary {
                label: label.clone(),
                log_likelihood: ll,
                bic,
                reduced_chi2: z2 / dof,
                affected_seq_ids: affected.clone(),
            },
        ));

        if best_cand.as_ref().is_none_or(|b| ll > b.log_likelihood) {
            let summary = format!(
                "XxxProCisTrans {} (Delta_beta_gamma = {:+.2} ppm)",
                label,
                s_iso * 5.00
            );
            let params = HypothesisBestParameters::XxxProCisTrans {
                chain_id: rp.chain_id.clone(),
                pro_seq_id: rp.seq_id,
                prev_seq_id: prev_intact.then(|| model.residues[p_idx - 1].seq_id),
                prev_res_name: prev_intact.then(|| model.residues[p_idx - 1].res_name.clone()),
                direction,
                ground_omega_deg: rp.omega_deg,
                predicted_delta_beta_gamma_ppm: s_iso * 5.00,
            };
            best_cand = Some(BestCandidateState {
                log_likelihood: ll,
                mu_preds,
                summary,
                affected_seq_ids: affected,
                parameters: params,
            });
        }
    }

    let Some(winner) = best_cand else {
        return Ok(None);
    };
    let marginal_ll = log_mean_exp(&cand_lls);
    let n_cands = cand_lls.len();
    sub_cands.sort_by(|a, b| b.0.total_cmp(&a.0));
    let top_sub = sub_cands.into_iter().take(5).map(|x| x.1).collect();

    let score = build_hypothesis_score(
        ExcitedStateHypothesisKind::XxxProCisTrans,
        k_h,
        n_distinct_residues,
        marginal_ll,
        n_cands,
        resolved,
        winner,
        top_sub,
    )?;
    Ok(Some(score))
}

/// Reconstruct conjugated aromatic ring(s) and distal heavy atoms for residue `r_idx` (`PHE`,
/// `TYR`, `HIS`, `TRP`) at candidate `(chi1_deg, chi2_deg)` using ideal Engh-Huber `NeRF` geometry.
fn build_rotated_aromatic_rings(
    model: &StructureModel,
    r_idx: usize,
    chi1_deg: f64,
    chi2_deg: f64,
) -> Option<RotatedAromaticGeometry> {
    let r = &model.residues[r_idx];
    let cb = r.cb?;
    let chi1 = chi1_deg.to_radians();
    let chi2 = chi2_deg.to_radians();

    let make_ring = |label: &str, verts: Vec<Vec3>, i_r: f64| -> ConjugatedAromaticRing {
        let inv = 1.0 / (verts.len() as f64);
        let mut cen = [0.0; 3];
        for &p in &verts {
            cen = v_add(cen, p);
        }
        cen = v_scale(cen, inv);
        let mut n_acc = [0.0; 3];
        let m = verts.len();
        for k in 0..m {
            let u = v_sub(verts[k], cen);
            let v = v_sub(verts[(k + 1) % m], cen);
            n_acc = v_add(n_acc, v_cross(u, v));
        }
        ConjugatedAromaticRing {
            chain_id: r.chain_id.clone(),
            seq_id: r.seq_id,
            residue_index: r_idx,
            res_name: r.res_name.clone(),
            ring_label: label.to_string(),
            vertices: verts,
            center: cen,
            normal: v_unit(n_acc),
            intensity_factor: i_r,
        }
    };

    match r.res_name.as_str() {
        "PHE" | "TYR" => {
            let cg = nerf_place_atom(r.n, r.ca, cb, 1.502, 113.8_f64.to_radians(), chi1);
            let cd1 = nerf_place_atom(r.ca, cb, cg, 1.390, 120.0_f64.to_radians(), chi2);
            let cd2 = nerf_place_atom(r.ca, cb, cg, 1.390, 120.0_f64.to_radians(), chi2 + PI);
            let ce1 = nerf_place_atom(cb, cg, cd1, 1.390, 120.0_f64.to_radians(), PI);
            let ce2 = nerf_place_atom(cb, cg, cd2, 1.390, 120.0_f64.to_radians(), PI);
            let cz = nerf_place_atom(cg, cd1, ce1, 1.390, 120.0_f64.to_radians(), 0.0);
            let mut heavy = vec![
                ("CG", cg),
                ("CD1", cd1),
                ("CD2", cd2),
                ("CE1", ce1),
                ("CE2", ce2),
                ("CZ", cz),
            ];
            let i_r = if r.res_name == "PHE" { 1.00 } else { 0.84 };
            if r.res_name == "TYR" {
                let oh = nerf_place_atom(cd1, ce1, cz, 1.376, 120.0_f64.to_radians(), PI);
                heavy.push(("OH", oh));
            }
            let ring = make_ring("6-ring", vec![cg, cd1, ce1, cz, ce2, cd2], i_r);
            Some((vec![ring], heavy))
        }
        "HIS" => {
            let cg = nerf_place_atom(r.n, r.ca, cb, 1.495, 113.8_f64.to_radians(), chi1);
            let nd1 = nerf_place_atom(r.ca, cb, cg, 1.378, 122.7_f64.to_radians(), chi2);
            let cd2 = nerf_place_atom(r.ca, cb, cg, 1.354, 130.8_f64.to_radians(), chi2 + PI);
            let ce1 = nerf_place_atom(cb, cg, nd1, 1.320, 108.5_f64.to_radians(), PI);
            let ne2 = nerf_place_atom(cb, cg, cd2, 1.374, 107.2_f64.to_radians(), PI);
            let heavy = vec![
                ("CG", cg),
                ("ND1", nd1),
                ("CD2", cd2),
                ("CE1", ce1),
                ("NE2", ne2),
            ];
            let ring = make_ring("5-ring", vec![cg, nd1, ce1, ne2, cd2], 0.53);
            Some((vec![ring], heavy))
        }
        "TRP" => {
            let cg = nerf_place_atom(r.n, r.ca, cb, 1.498, 113.8_f64.to_radians(), chi1);
            let cd1 = nerf_place_atom(r.ca, cb, cg, 1.365, 126.9_f64.to_radians(), chi2);
            let cd2 = nerf_place_atom(r.ca, cb, cg, 1.433, 126.4_f64.to_radians(), chi2 + PI);
            let ne1 = nerf_place_atom(cb, cg, cd1, 1.375, 110.2_f64.to_radians(), PI);
            let ce2 = nerf_place_atom(cb, cg, cd2, 1.412, 107.2_f64.to_radians(), PI);
            let ce3 = nerf_place_atom(cb, cg, cd2, 1.400, 133.9_f64.to_radians(), 0.0);
            let cz2 = nerf_place_atom(cg, cd2, ce2, 1.396, 122.4_f64.to_radians(), PI);
            let cz3 = nerf_place_atom(cg, cd2, ce3, 1.392, 118.6_f64.to_radians(), PI);
            let ch2 = nerf_place_atom(cd2, ce2, cz2, 1.380, 117.5_f64.to_radians(), 0.0);
            let heavy = vec![
                ("CG", cg),
                ("CD1", cd1),
                ("CD2", cd2),
                ("NE1", ne1),
                ("CE2", ce2),
                ("CE3", ce3),
                ("CZ2", cz2),
                ("CZ3", cz3),
                ("CH2", ch2),
            ];
            let r5 = make_ring("5-ring", vec![cg, cd1, ne1, ce2, cd2], 0.56);
            let r6 = make_ring("6-ring", vec![cd2, ce2, cz2, ch2, cz3, ce3], 1.04);
            Some((vec![r5, r6], heavy))
        }
        _ => None,
    }
}

/// Check steric compatibility of a candidate rotated aromatic ring against the rigid polypeptide
/// backbone `(N, CA, C, O)` (`>= 2.20 A`) and `CB` scaffold (`>= 2.00 A`), returning
/// `Some(min_dist_A)` if compatible (surrounding distal flexible sidechains can co-repack in the
/// excited state, whereas backbone/CB interpenetration is sterically forbidden).
fn check_rotamer_steric_clearance(
    model: &StructureModel,
    r_idx: usize,
    placed_heavy: &[(&str, Vec3)],
) -> Option<f64> {
    let mut min_d = f64::INFINITY;
    let r_self = &model.residues[r_idx];

    for &(aname, apos) in placed_heavy {
        // Check distal ring atoms (`CE*`, `CZ*`, `NE2`, `CH2`, `OH`) against own residue backbone `(N, C, O)`
        if !matches!(aname, "CG" | "CD1" | "CD2" | "ND1") {
            for own_bb in [r_self.n, r_self.c, r_self.o] {
                let d = v_dist(apos, own_bb);
                if d < 2.15 {
                    return None;
                }
                min_d = min_d.min(d);
            }
        }
        for (j, rj) in model.residues.iter().enumerate() {
            if j == r_idx {
                continue;
            }
            if v_dist(apos, rj.ca) > 9.5 {
                continue;
            }
            for bb_pt in [rj.n, rj.ca, rj.c, rj.o] {
                let d = v_dist(apos, bb_pt);
                if d < 2.20 {
                    return None;
                }
                min_d = min_d.min(d);
            }
            if let Some(cb_j) = rj.cb {
                let d = v_dist(apos, cb_j);
                if d < 2.00 {
                    return None;
                }
                min_d = min_d.min(d);
            }
            // Reject severe covalent overlap against rigid disulfide SG atoms
            if rj.cys_oxidized {
                if let Some(&sg_j) = rj.atoms.get("SG") {
                    let d = v_dist(apos, sg_j);
                    if d < 2.00 {
                        return None;
                    }
                    min_d = min_d.min(d);
                }
            }
        }
    }
    Some(min_d)
}

fn eval_obs_ring_current_contribution(rings: &[ConjugatedAromaticRing], obs: &ResolvedObs) -> f64 {
    if obs.spin_coords.is_empty() {
        return 0.0;
    }
    let (scale, lo, hi) = match obs.atom.as_str() {
        "H" => (1.0, -1.80, 1.40),
        "HA" | "HA2" | "HA3" => (1.0, -1.50, 1.20),
        "HD1*" | "HD2*" | "HG1*" | "HG2*" | "HE*" | "HB*" => (1.0, -1.85, 1.25),
        "N" => (0.65, -1.50, 1.50),
        "CA" | "CB" | "C" => (0.55, -1.00, 1.00),
        "CD1" | "CD2" | "CG1" | "CG2" | "CE" => (0.55, -0.90, 0.90),
        a if a.starts_with('H') => (1.0, -1.60, 1.30),
        _ => (0.55, -1.00, 1.00),
    };
    let is_backbone_or_self_ring =
        matches!(obs.atom.as_str(), "H" | "N" | "CA" | "CB" | "C" | "HA")
            || matches!(obs.res_name.as_str(), "PHE" | "TYR" | "HIS" | "TRP");
    let mut acc = 0.0;
    for &pt in &obs.spin_coords {
        let (d_ppm, _) = evaluate_haigh_mallion_ring_current(
            rings,
            pt,
            obs.res_idx,
            is_backbone_or_self_ring,
            scale,
        );
        acc += d_ppm;
    }
    let mean = acc / (obs.spin_coords.len() as f64);
    mean.clamp(lo, hi)
}

fn eval_his_transition_delta_ppm(transition: HisExchangeTransition, atom: &str) -> f64 {
    // Pelton et al. (1993) Table 1, Platzer et al. (2014) & topology::HIS_*_PRIORS canonical shifts:
    // Nd1H -> Ne2H: ND1 +82.0, NE2 -82.0, CD2 -7.2, CG +8.3 (135.5 - 127.2), CB +2.2, CA +0.55, N -0.85, H -0.12
    match transition {
        HisExchangeTransition::Nd1HToNe2H => match atom {
            "ND1" => 82.0,
            "NE2" => -82.0,
            "CD2" => -7.2,
            "CG" => HIS_NE2H_PRIORS.cg.0 - HIS_ND1H_PRIORS.cg.0, // +8.3 ppm
            "CE1" => 0.2,
            "HD2" => -0.12,
            "HE1" => -0.04,
            "CB" => 2.20,
            "CA" => 0.55,
            "C" => 0.35,
            "N" => -0.85,
            "H" => -0.12,
            _ => 0.0,
        },
        HisExchangeTransition::Ne2HToNd1H => {
            -eval_his_transition_delta_ppm(HisExchangeTransition::Nd1HToNe2H, atom)
        }
        HisExchangeTransition::Ne2HToImidazolium => match atom {
            "ND1" => HIS_DOUBLY_PROTONATED_PRIORS.nd1.0 - HIS_NE2H_PRIORS.nd1.0, // -72.5 ppm
            "NE2" => HIS_DOUBLY_PROTONATED_PRIORS.ne2.0 - HIS_NE2H_PRIORS.ne2.0, // +5.0 ppm
            "CG" => HIS_DOUBLY_PROTONATED_PRIORS.cg.0 - HIS_NE2H_PRIORS.cg.0,    // -5.7 ppm
            "CD2" => HIS_DOUBLY_PROTONATED_PRIORS.cd2.0 - HIS_NE2H_PRIORS.cd2.0, // +0.8 ppm
            "CE1" => HIS_DOUBLY_PROTONATED_PRIORS.ce1.0 - HIS_NE2H_PRIORS.ce1.0, // -2.2 ppm
            "HD2" => HIS_DOUBLY_PROTONATED_PRIORS.hd2.0 - HIS_NE2H_PRIORS.hd2.0, // +0.32 ppm
            "HE1" => HIS_DOUBLY_PROTONATED_PRIORS.he1.0 - HIS_NE2H_PRIORS.he1.0, // +0.78 ppm
            "CB" => -1.80,
            "CA" => -1.25,
            "C" => -0.75,
            "N" => 0.90,
            "H" => 0.22,
            "HA" => 0.12,
            _ => 0.0,
        },
        HisExchangeTransition::ImidazoliumToNe2H => {
            -eval_his_transition_delta_ppm(HisExchangeTransition::Ne2HToImidazolium, atom)
        }
        HisExchangeTransition::Nd1HToImidazolium => match atom {
            "ND1" => HIS_DOUBLY_PROTONATED_PRIORS.nd1.0 - HIS_ND1H_PRIORS.nd1.0, // +7.5 ppm
            "NE2" => HIS_DOUBLY_PROTONATED_PRIORS.ne2.0 - HIS_ND1H_PRIORS.ne2.0, // -76.0 ppm
            "CG" => HIS_DOUBLY_PROTONATED_PRIORS.cg.0 - HIS_ND1H_PRIORS.cg.0,    // +2.6 ppm
            "CD2" => HIS_DOUBLY_PROTONATED_PRIORS.cd2.0 - HIS_ND1H_PRIORS.cd2.0, // -6.2 ppm
            "CE1" => HIS_DOUBLY_PROTONATED_PRIORS.ce1.0 - HIS_ND1H_PRIORS.ce1.0, // -2.0 ppm
            "HD2" => HIS_DOUBLY_PROTONATED_PRIORS.hd2.0 - HIS_ND1H_PRIORS.hd2.0, // +0.20 ppm
            "HE1" => HIS_DOUBLY_PROTONATED_PRIORS.he1.0 - HIS_ND1H_PRIORS.he1.0, // +0.74 ppm
            "CB" => 0.40,
            "CA" => -1.25,
            "C" => -0.75,
            "N" => 0.90,
            "H" => 0.22,
            "HA" => 0.12,
            _ => 0.0,
        },
        HisExchangeTransition::ImidazoliumToNd1H => {
            -eval_his_transition_delta_ppm(HisExchangeTransition::Nd1HToImidazolium, atom)
        }
        HisExchangeTransition::AmbiguousDirectionOnUnsignedCpmg => 0.0,
    }
}

/// Evaluate `H3: RotamerOrHisTautomerFlip` (`k = 2`) across:
/// - (a) Steric-clash-compatible `Phe`/`Tyr`/`Trp`/`His` `(chi1 in {+60, 180, -60}, chi2 in {+90, -90, 0})`
///   ring-current rotamer flips within `10.0 A` of exchanging spins, and
/// - (b) Pelton (1993) `His` imidazole `Nd1-H <-> Ne2-H` tautomer & `His^0 <-> HisH^+` protonation switches
///   (eligible strictly when sidechain `ND1`/`NE2`/`CD2`/`CG`/`CE1` observations are present on that `HIS`).
fn evaluate_h3_rotamer_or_his_flip(
    model: &StructureModel,
    ground_shifts: Option<&Reference>,
    ref_map_by_res_idx: &[Option<usize>],
    resolved: &[ResolvedObs],
    n_distinct_residues: usize,
) -> Result<Option<HypothesisScore>, String> {
    let l = model.residues.len();
    if l == 0 || resolved.is_empty() {
        return Ok(None);
    }

    let k_h = ExcitedStateHypothesisKind::RotamerOrHisTautomerFlip.default_k();
    let ln_n = (n_distinct_residues.max(1) as f64).ln();
    let dof = resolved.len().saturating_sub(k_h).max(1) as f64;

    // Precompute ground-state total ring-current contribution at every observation
    let ground_rc_at_obs: Vec<f64> = resolved
        .iter()
        .map(|o| eval_obs_ring_current_contribution(&model.aromatic_rings, o))
        .collect();

    let mut best_cand: Option<BestCandidateState> = None;
    let mut cand_lls: Vec<f64> = Vec::new();
    let mut sub_cands: Vec<(f64, SubCandidateSummary)> = Vec::new();

    const CHI1_STATES: [f64; 3] = [60.0, 180.0, -60.0];
    const CHI2_STATES: [f64; 3] = [90.0, -90.0, 0.0];

    for (r_idx, &ref_idx_opt) in ref_map_by_res_idx.iter().enumerate().take(l) {
        let rr = &model.residues[r_idx];
        if !matches!(rr.res_name.as_str(), "PHE" | "TYR" | "HIS" | "TRP") {
            continue;
        }
        // Within `10.0 A` of at least one observed spin
        let anchor = rr.cb.unwrap_or(rr.ca);
        let ground_rings_for_r: Vec<&ConjugatedAromaticRing> = model
            .aromatic_rings
            .iter()
            .filter(|rg| rg.residue_index == r_idx)
            .collect();
        let within_cutoff = resolved.iter().any(|o| {
            if o.res_idx == r_idx {
                return true;
            }
            o.spin_coords.iter().any(|&sp| {
                v_dist(sp, anchor) <= AROMATIC_ROTAMER_CUTOFF_ANGSTROM
                    || ground_rings_for_r
                        .iter()
                        .any(|rg| v_dist(sp, rg.center) <= AROMATIC_ROTAMER_CUTOFF_ANGSTROM)
            })
        });
        if !within_cutoff {
            continue;
        }

        // (a) Scan steric-clash-checked `(chi1, chi2)` aromatic rotamers
        let non_self_rings: Vec<ConjugatedAromaticRing> = model
            .aromatic_rings
            .iter()
            .filter(|rg| rg.residue_index != r_idx)
            .cloned()
            .collect();

        for &chi1_cand in &CHI1_STATES {
            for &chi2_cand in &CHI2_STATES {
                // For C2-symmetric PHE/TYR 6-rings, chi2 = -90 deg and +90 deg are identical planes
                if matches!(rr.res_name.as_str(), "PHE" | "TYR") && chi2_cand == -90.0 {
                    continue;
                }
                let Some((cand_rings, heavy_atoms)) =
                    build_rotated_aromatic_rings(model, r_idx, chi1_cand, chi2_cand)
                else {
                    continue;
                };

                // Skip trivial identity rotamers whose ring center and plane barely move from ground state
                if let (Some(g0), Some(c0)) = (ground_rings_for_r.first(), cand_rings.first()) {
                    let center_shift = v_dist(g0.center, c0.center);
                    let normal_dot = v_dot(g0.normal, c0.normal).abs();
                    if center_shift < 0.45 && normal_dot > 0.94 {
                        continue;
                    }
                }

                let Some(min_steric_d) = check_rotamer_steric_clearance(model, r_idx, &heavy_atoms)
                else {
                    continue;
                };

                let mut combined_rings = non_self_rings.clone();
                combined_rings.extend(cand_rings);

                // Local gamma-gauche shift on residue r itself if chi1 changed
                let d_chi1_self_cb = match rr.chi1_deg {
                    Some(g_chi1) if periodic_angle_abs_diff_deg(g_chi1, chi1_cand) >= 35.0 => {
                        let g_gauche_old =
                            usize::from(periodic_angle_abs_diff_deg(g_chi1, 60.0) < 55.0) + 1;
                        let g_gauche_new =
                            usize::from(periodic_angle_abs_diff_deg(chi1_cand, 60.0) < 55.0) + 1;
                        -1.65 * ((g_gauche_new as f64) - (g_gauche_old as f64))
                    }
                    _ => 0.0,
                };

                let mut mu_preds = Vec::with_capacity(resolved.len());
                let mut affected_set = BTreeSet::new();
                affected_set.insert(rr.seq_id);

                for (oi, o) in resolved.iter().enumerate() {
                    let new_rc = eval_obs_ring_current_contribution(&combined_rings, o);
                    let mut d_mu = new_rc - ground_rc_at_obs[oi];
                    if o.res_idx == r_idx && o.atom == "CB" {
                        d_mu += d_chi1_self_cb;
                    } else if o.res_idx == r_idx && o.atom == "CA" {
                        d_mu -= 0.35 * d_chi1_self_cb;
                    }
                    if d_mu.abs() >= 0.05 {
                        affected_set.insert(o.seq_id);
                    }
                    mu_preds.push(d_mu);
                }

                let mut ll = 0.0;
                let mut z2 = 0.0;
                for (o, &mu) in resolved.iter().zip(mu_preds.iter()) {
                    let (l_i, _, z_i) = eval_single_obs_ll_and_z(o, mu)?;
                    ll += l_i;
                    z2 += z_i * z_i;
                }
                cand_lls.push(ll);
                let bic = -2.0 * ll + (k_h as f64) * ln_n;
                let affected: Vec<i32> = affected_set.into_iter().collect();
                let label = format!(
                    "chain {} {}{} rotamer chi1={:+.0} deg, chi2={:+.0} deg (min_d={:.2} A)",
                    rr.chain_id, rr.res_name, rr.seq_id, chi1_cand, chi2_cand, min_steric_d
                );
                sub_cands.push((
                    ll,
                    SubCandidateSummary {
                        label: label.clone(),
                        log_likelihood: ll,
                        bic,
                        reduced_chi2: z2 / dof,
                        affected_seq_ids: affected.clone(),
                    },
                ));

                if best_cand.as_ref().is_none_or(|b| ll > b.log_likelihood) {
                    let summary = format!("AromaticRotamerFlip {label}");
                    let params = HypothesisBestParameters::AromaticRotamerFlip {
                        chain_id: rr.chain_id.clone(),
                        seq_id: rr.seq_id,
                        res_name: rr.res_name.clone(),
                        ground_chi1_deg: rr.chi1_deg,
                        ground_chi2_deg: rr.chi2_deg,
                        excited_chi1_deg: chi1_cand,
                        excited_chi2_deg: chi2_cand,
                        min_steric_dist_angstrom: min_steric_d,
                    };
                    best_cand = Some(BestCandidateState {
                        log_likelihood: ll,
                        mu_preds,
                        summary,
                        affected_seq_ids: affected,
                        parameters: params,
                    });
                }
            }
        }

        // (b) Pelton et al. (1993) Histidine imidazole tautomer (`Nd1H <-> Ne2H`) & protonation transitions:
        //     eligible ONLY when sidechain `ND1`/`NE2`/`CD2`/`CG`/`CE1` data are observed on this HIS.
        let has_his_sidechain_data = rr.res_name == "HIS"
            && resolved.iter().any(|o| {
                o.res_idx == r_idx
                    && matches!(o.atom.as_str(), "ND1" | "NE2" | "CD2" | "CG" | "CE1")
            });
        if has_his_sidechain_data {
            // Check if EXPERIMENTAL ground-state ND1 / NE2 shifts pin the starting histidine state
            // (never use `fallback_sidechain_coil_ppm`, which would bias unsigned CPMG direction!).
            let mut obs_ground_nd1: Option<f64> = resolved
                .iter()
                .find(|o| o.res_idx == r_idx && o.atom == "ND1")
                .and_then(|o| o.explicit_ground_ppm.filter(|&v| v > 100.0));
            let mut obs_ground_ne2: Option<f64> = resolved
                .iter()
                .find(|o| o.res_idx == r_idx && o.atom == "NE2")
                .and_then(|o| o.explicit_ground_ppm.filter(|&v| v > 100.0));
            if let (Some(gs), Some(ri)) = (ground_shifts, ref_idx_opt) {
                obs_ground_nd1 =
                    obs_ground_nd1.or_else(|| lookup_reference_shift(gs, ri, "HIS", "ND1"));
                obs_ground_ne2 =
                    obs_ground_ne2.or_else(|| lookup_reference_shift(gs, ri, "HIS", "NE2"));
            }

            let mut his_evals: Vec<(HisExchangeTransition, f64, BestCandidateState)> = Vec::new();
            for tr in HisExchangeTransition::ALL {
                // Filter transitions incompatible with known ground-state imidazole 15N chemical shifts
                let compatible = match tr {
                    HisExchangeTransition::Ne2HToNd1H
                    | HisExchangeTransition::Ne2HToImidazolium => {
                        obs_ground_nd1.is_none_or(|v| v >= 210.0)
                            && obs_ground_ne2.is_none_or(|v| v <= 205.0)
                    }
                    HisExchangeTransition::Nd1HToNe2H
                    | HisExchangeTransition::Nd1HToImidazolium => {
                        obs_ground_nd1.is_none_or(|v| v <= 205.0)
                            && obs_ground_ne2.is_none_or(|v| v >= 210.0)
                    }
                    HisExchangeTransition::ImidazoliumToNe2H
                    | HisExchangeTransition::ImidazoliumToNd1H => {
                        obs_ground_nd1.is_none_or(|v| v <= 205.0)
                            && obs_ground_ne2.is_none_or(|v| v <= 205.0)
                    }
                    HisExchangeTransition::AmbiguousDirectionOnUnsignedCpmg => false,
                };
                if !compatible {
                    continue;
                }

                let is_charge_change = !matches!(
                    tr,
                    HisExchangeTransition::Nd1HToNe2H | HisExchangeTransition::Ne2HToNd1H
                );
                let charge_sign = match tr {
                    HisExchangeTransition::Ne2HToImidazolium
                    | HisExchangeTransition::Nd1HToImidazolium => 1.0,
                    HisExchangeTransition::ImidazoliumToNe2H
                    | HisExchangeTransition::ImidazoliumToNd1H => -1.0,
                    _ => 0.0,
                };
                let his_ring_cen = ground_rings_for_r
                    .first()
                    .map(|rg| rg.center)
                    .unwrap_or(anchor);

                let mut mu_preds = Vec::with_capacity(resolved.len());
                let mut affected_set = BTreeSet::new();
                affected_set.insert(rr.seq_id);

                for o in resolved {
                    let mu = if o.res_idx == r_idx {
                        eval_his_transition_delta_ppm(tr, &o.atom)
                    } else {
                        // Local electrostatic / H-bond perturbation on proximal amides (`<= 6.5 A`)
                        let ro = &model.residues[o.res_idx];
                        let d_ring = o
                            .spin_coords
                            .first()
                            .map(|&sp| v_dist(sp, his_ring_cen))
                            .unwrap_or_else(|| v_dist(ro.ca, his_ring_cen));
                        if d_ring <= 6.5 {
                            let inv_r2 = 1.0 / d_ring.max(2.4).powi(2);
                            if is_charge_change {
                                match o.atom.as_str() {
                                    "H" => charge_sign * 1.60 * inv_r2,
                                    "N" => charge_sign * 4.50 * inv_r2,
                                    _ => 0.0,
                                }
                            } else {
                                match o.atom.as_str() {
                                    "H" => 0.55 * inv_r2,
                                    "N" => 1.40 * inv_r2,
                                    _ => 0.0,
                                }
                            }
                        } else {
                            0.0
                        }
                    };
                    if mu.abs() >= 0.05 {
                        affected_set.insert(o.seq_id);
                    }
                    mu_preds.push(mu);
                }

                let mut ll = 0.0;
                let mut z2 = 0.0;
                for (o, &mu) in resolved.iter().zip(mu_preds.iter()) {
                    let (l_i, _, z_i) = eval_single_obs_ll_and_z(o, mu)?;
                    ll += l_i;
                    z2 += z_i * z_i;
                }
                cand_lls.push(ll);
                let bic = -2.0 * ll + (k_h as f64) * ln_n;
                let affected: Vec<i32> = affected_set.into_iter().collect();
                let label = format!("chain {} HIS{} {}", rr.chain_id, rr.seq_id, tr.as_str());
                sub_cands.push((
                    ll,
                    SubCandidateSummary {
                        label: label.clone(),
                        log_likelihood: ll,
                        bic,
                        reduced_chi2: z2 / dof,
                        affected_seq_ids: affected.clone(),
                    },
                ));

                let d_nd1 = eval_his_transition_delta_ppm(tr, "ND1");
                let d_ne2 = eval_his_transition_delta_ppm(tr, "NE2");
                let d_cd2 = eval_his_transition_delta_ppm(tr, "CD2");
                let d_cg = eval_his_transition_delta_ppm(tr, "CG");
                let summary = format!(
                    "HisTautomerOrProtonation {label} (dND1={d_nd1:+.1}, dNE2={d_ne2:+.1}, dCD2={d_cd2:+.1}, dCG={d_cg:+.1} ppm)"
                );
                let params = HypothesisBestParameters::HisTautomerOrProtonation {
                    chain_id: rr.chain_id.clone(),
                    seq_id: rr.seq_id,
                    transition: tr,
                    delta_nd1_ppm: d_nd1,
                    delta_ne2_ppm: d_ne2,
                    delta_cd2_ppm: d_cd2,
                    delta_cg_ppm: d_cg,
                };
                his_evals.push((
                    tr,
                    ll,
                    BestCandidateState {
                        log_likelihood: ll,
                        mu_preds,
                        summary,
                        affected_seq_ids: affected,
                        parameters: params,
                    },
                ));
            }

            // Detect unsigned CPMG direction tie between forward and reverse His transitions
            let tie_map: Vec<(HisExchangeTransition, f64)> =
                his_evals.iter().map(|(t, l, _)| (*t, *l)).collect();
            for (tr, ll, mut state) in his_evals {
                let rev = tr.reverse();
                let has_rev_tie = rev != tr
                    && tie_map.iter().any(|&(other_t, other_ll)| {
                        other_t == rev && (other_ll - ll).abs() <= 1e-6
                    });
                if has_rev_tie {
                    let d_nd1 = eval_his_transition_delta_ppm(tr, "ND1").abs();
                    let d_ne2 = eval_his_transition_delta_ppm(tr, "NE2").abs();
                    let d_cd2 = eval_his_transition_delta_ppm(tr, "CD2").abs();
                    let d_cg = eval_his_transition_delta_ppm(tr, "CG").abs();
                    state.summary = format!(
                        "HisTautomerOrProtonation chain {} HIS{} AmbiguousDirectionOnUnsignedCpmg (|dND1|={d_nd1:.1}, |dNE2|={d_ne2:.1}, |dCD2|={d_cd2:.1}, |dCG|={d_cg:.1} ppm)",
                        rr.chain_id, rr.seq_id
                    );
                    state.parameters = HypothesisBestParameters::HisTautomerOrProtonation {
                        chain_id: rr.chain_id.clone(),
                        seq_id: rr.seq_id,
                        transition: HisExchangeTransition::AmbiguousDirectionOnUnsignedCpmg,
                        delta_nd1_ppm: d_nd1,
                        delta_ne2_ppm: d_ne2,
                        delta_cd2_ppm: d_cd2,
                        delta_cg_ppm: d_cg,
                    };
                }
                if best_cand.as_ref().is_none_or(|b| ll > b.log_likelihood) {
                    best_cand = Some(state);
                }
            }
        }
    }

    let Some(winner) = best_cand else {
        return Ok(None);
    };
    let marginal_ll = log_mean_exp(&cand_lls);
    let n_cands = cand_lls.len();
    sub_cands.sort_by(|a, b| b.0.total_cmp(&a.0));
    let top_sub = sub_cands.into_iter().take(5).map(|x| x.1).collect();

    let score = build_hypothesis_score(
        ExcitedStateHypothesisKind::RotamerOrHisTautomerFlip,
        k_h,
        n_distinct_residues,
        marginal_ll,
        n_cands,
        resolved,
        winner,
        top_sub,
    )?;
    Ok(Some(score))
}

/// Evaluate `H4: Alternate3DConformer` (`k = 1`, Bouvignies et al. 2011; Wayment-Steele et al.
/// 2024) across supplied candidate 3D conformers under the 4-layer `HomologFirewall`.
fn evaluate_h4_alt_conformers(
    ground_model: &StructureModel,
    pred_a: &StructShiftPredictionReport,
    alt_structures: &[AltConformerInput],
    firewall_query: &HomologFirewallQuery,
    struct_cfg: &StructShiftConfig,
    resolved: &[ResolvedObs],
    n_distinct_residues: usize,
) -> Result<Option<HypothesisScore>, String> {
    if alt_structures.is_empty() || resolved.is_empty() {
        return Ok(None);
    }
    if alt_structures.len() > MAX_ALT_STRUCTURES {
        return Err(format!(
            "alt_structures count ({}) exceeds limit ({MAX_ALT_STRUCTURES})",
            alt_structures.len()
        ));
    }

    let k_h = ExcitedStateHypothesisKind::Alternate3DConformer.default_k();
    let ln_n = (n_distinct_residues.max(1) as f64).ln();
    let dof = resolved.len().saturating_sub(k_h).max(1) as f64;

    let g_keys: Vec<(i32, &str)> = ground_model
        .residues
        .iter()
        .map(|r| (r.seq_id, r.res_name.as_str()))
        .collect();

    let mut best_cand: Option<BestCandidateState> = None;
    let mut cand_lls: Vec<f64> = Vec::new();
    let mut sub_cands: Vec<(f64, SubCandidateSummary)> = Vec::new();

    for (m_idx, alt) in alt_structures.iter().enumerate() {
        let pred_b =
            predict_structure_shifts_with_firewall(&alt.model, firewall_query, struct_cfg)?;

        // Align `(chain_id, seq_id, res_name)` directly first, with sequence-alignment fallback
        let mut exact_map: BTreeMap<(&str, i32), usize> = BTreeMap::new();
        for (bi, rb) in alt.model.residues.iter().enumerate() {
            exact_map.insert((rb.chain_id.as_str(), rb.seq_id), bi);
        }
        let b_keys: Vec<(i32, &str)> = alt
            .model
            .residues
            .iter()
            .map(|r| (r.seq_id, r.res_name.as_str()))
            .collect();
        let aligned_pairs = align_by_res_names(&g_keys, &b_keys);
        let mut fallback_map = vec![None; ground_model.residues.len()];
        for (gi, bi) in aligned_pairs {
            fallback_map[gi] = Some(bi);
        }

        let mut mu_preds = Vec::with_capacity(resolved.len());
        let mut affected_set = BTreeSet::new();

        for o in resolved {
            let rg = &ground_model.residues[o.res_idx];
            let b_res_opt = exact_map
                .get(&(rg.chain_id.as_str(), rg.seq_id))
                .copied()
                .filter(|&bi| alt.model.residues[bi].res_name == rg.res_name)
                .or(fallback_map[o.res_idx]);

            let mu = match (o.pred_a_struct_ppm, b_res_opt) {
                (Some(sa), Some(bi)) => {
                    let (sb_opt, _) = lookup_pred_struct_and_rc(&pred_b, bi, &o.atom);
                    sb_opt.map(|sb| sb - sa).unwrap_or(0.0)
                }
                _ => {
                    let _ = pred_a;
                    0.0
                }
            };
            if mu.abs() >= 0.25 {
                affected_set.insert(o.seq_id);
            }
            mu_preds.push(mu);
        }

        let mut ll = 0.0;
        let mut z2 = 0.0;
        for (o, &mu) in resolved.iter().zip(mu_preds.iter()) {
            let (l_i, _, z_i) = eval_single_obs_ll_and_z(o, mu)?;
            ll += l_i;
            z2 += z_i * z_i;
        }
        cand_lls.push(ll);
        let bic = -2.0 * ll + (k_h as f64) * ln_n;
        let affected: Vec<i32> = affected_set.into_iter().collect();
        let label = format!("alt[{}] '{}'", m_idx, alt.label);
        sub_cands.push((
            ll,
            SubCandidateSummary {
                label: label.clone(),
                log_likelihood: ll,
                bic,
                reduced_chi2: z2 / dof,
                affected_seq_ids: affected.clone(),
            },
        ));

        if best_cand.as_ref().is_none_or(|b| ll > b.log_likelihood) {
            let summary = format!(
                "Alternate3DConformer {} (firewall max_seq_id={:.3}, sha256={}..)",
                label,
                pred_b.max_retained_seq_identity,
                &pred_b.exclusion_list_sha256[..12.min(pred_b.exclusion_list_sha256.len())]
            );
            let params = HypothesisBestParameters::Alternate3DConformer {
                model_index: m_idx,
                model_label: alt.label.clone(),
                exclusion_list_sha256: pred_b.exclusion_list_sha256.clone(),
                max_retained_seq_identity: pred_b.max_retained_seq_identity,
            };
            best_cand = Some(BestCandidateState {
                log_likelihood: ll,
                mu_preds,
                summary,
                affected_seq_ids: affected,
                parameters: params,
            });
        }
    }

    let Some(winner) = best_cand else {
        return Ok(None);
    };
    let marginal_ll = log_mean_exp(&cand_lls);
    let n_cands = cand_lls.len();
    sub_cands.sort_by(|a, b| b.0.total_cmp(&a.0));
    let top_sub = sub_cands.into_iter().take(5).map(|x| x.1).collect();

    let score = build_hypothesis_score(
        ExcitedStateHypothesisKind::Alternate3DConformer,
        k_h,
        n_distinct_residues,
        marginal_ll,
        n_cands,
        resolved,
        winner,
        top_sub,
    )?;
    Ok(Some(score))
}

// ============================================================================
// 5. Public Bayesian Hypothesis Ranking & Calibration Abstention Engine
// ============================================================================

/// Classify an invisible excited protein state (`varpi_B = varpi_A + Delta_varpi_AB`) across
/// signed (`CEST`/`R1rho`) and/or unsigned (`CPMG` `|Delta_varpi|`) chemical-exchange observations
/// enforcing the default 4-layer `HomologFirewall` (`max_seq_identity <= 0.25`, `TM-score < 0.50`).
pub fn classify_excited_state(
    ground_shifts: Option<&Reference>,
    ground_structure: &StructureModel,
    observations: &[ExchangeShiftObservation],
    alt_structures: &[AltConformerInput],
    cfg: &ExcitedStateConfig,
) -> Result<ExcitedStateReport, String> {
    classify_excited_state_with_firewall(
        ground_shifts,
        ground_structure,
        observations,
        alt_structures,
        &HomologFirewallQuery::default(),
        cfg,
    )
}

/// Classify an invisible excited protein state with an explicit [`HomologFirewallQuery`].
pub fn classify_excited_state_with_firewall(
    ground_shifts: Option<&Reference>,
    ground_structure: &StructureModel,
    observations: &[ExchangeShiftObservation],
    alt_structures: &[AltConformerInput],
    firewall_query: &HomologFirewallQuery,
    cfg: &ExcitedStateConfig,
) -> Result<ExcitedStateReport, String> {
    classify_excited_state_with_controls(
        ground_shifts,
        ground_structure,
        observations,
        &[],
        alt_structures,
        firewall_query,
        cfg,
    )
}

/// Classify an invisible excited protein state, additionally calibrating the stated experimental
/// error scale on non-exchanging `control_sites` (expected `Delta_varpi = 0`; e.g. rows tagged
/// `site_role = control` in the exchange TSV). With controls supplied, their
/// `chi^2 = sum (dw / dw_err)^2` against the `chi^2_{0.99, n_ctrl}` bound decides whether the
/// stated errors are consistent; otherwise the best model's inlier residuals are used.
pub fn classify_excited_state_with_controls(
    ground_shifts: Option<&Reference>,
    ground_structure: &StructureModel,
    observations: &[ExchangeShiftObservation],
    control_sites: &[ExchangeShiftObservation],
    alt_structures: &[AltConformerInput],
    firewall_query: &HomologFirewallQuery,
    cfg: &ExcitedStateConfig,
) -> Result<ExcitedStateReport, String> {
    if control_sites.len() > MAX_EXCHANGE_OBSERVATIONS {
        return Err(format!(
            "control_sites length ({}) exceeds limit ({MAX_EXCHANGE_OBSERVATIONS})",
            control_sites.len()
        ));
    }
    let mut control_sum_z2 = 0.0_f64;
    for (ci, c) in control_sites.iter().enumerate() {
        if !is_supported_exchange_atom_any_residue(c.res_name.as_deref(), &c.atom) {
            return Err(format!(
                "control site #{ci}: {UNSUPPORTED_EXCHANGE_ATOM_MESSAGE}"
            ));
        }
        if !c.dw_ppm.is_finite() || !c.dw_err_ppm.is_finite() || c.dw_err_ppm <= 0.0 {
            return Err(format!(
                "control site #{ci} has non-finite dw_ppm or non-positive dw_err_ppm"
            ));
        }
        // Signed: z = dw / err. Unsigned |dw| under no exchange has the same second moment.
        let z = c.dw_ppm / c.dw_err_ppm;
        control_sum_z2 += z * z;
    }
    if ground_structure.residues.is_empty() {
        return Err("ground_structure contains zero residues".to_string());
    }
    if observations.is_empty() {
        return Err("observations table cannot be empty".to_string());
    }
    if observations.len() > MAX_EXCHANGE_OBSERVATIONS {
        return Err(format!(
            "observations length ({}) exceeds limit ({MAX_EXCHANGE_OBSERVATIONS})",
            observations.len()
        ));
    }
    if !cfg.sample_ph.is_finite()
        || !cfg.temperature_kelvin.is_finite()
        || !cfg.sigma_fwd_scale.is_finite()
        || cfg.sigma_fwd_scale <= 0.0
    {
        return Err(
            "ExcitedStateConfig contains non-finite or non-positive parameters".to_string(),
        );
    }

    let struct_cfg = StructShiftConfig {
        sample_ph: cfg.sample_ph,
        temperature_kelvin: cfg.temperature_kelvin,
        include_methyl_ilvmat: true,
        max_seq_identity: cfg.max_seq_identity.min(DEFAULT_FIREWALL_MAX_SEQ_IDENTITY),
        ..StructShiftConfig::default()
    };

    let pred_a =
        predict_structure_shifts_with_firewall(ground_structure, firewall_query, &struct_cfg)?;

    // Map `ground_structure` residues -> `ground_shifts` indices
    let l = ground_structure.residues.len();
    let mut ref_map_by_res_idx: Vec<Option<usize>> = vec![None; l];
    if let Some(gs) = ground_shifts {
        let s_keys: Vec<(i32, &str)> = ground_structure
            .residues
            .iter()
            .map(|r| (r.seq_id, r.res_name.as_str()))
            .collect();
        let r_keys: Vec<(i32, &str)> = gs
            .seq_ids
            .iter()
            .zip(gs.sequence.iter())
            .map(|(&sid, rn)| (sid, rn.as_str()))
            .collect();
        for (si, ri) in align_by_res_names(&s_keys, &r_keys) {
            ref_map_by_res_idx[si] = Some(ri);
        }
    }

    // Build `(chain_id, seq_id)` and `seq_id` lookup maps onto `ground_structure`
    let mut by_chain_seq: BTreeMap<(&str, i32), usize> = BTreeMap::new();
    let mut by_seq_only: BTreeMap<i32, usize> = BTreeMap::new();
    for (idx, r) in ground_structure.residues.iter().enumerate() {
        by_chain_seq.insert((r.chain_id.as_str(), r.seq_id), idx);
        by_seq_only.entry(r.seq_id).or_insert(idx);
    }

    let mut resolved = Vec::with_capacity(observations.len());
    let mut warnings = Vec::new();
    let mut n_unmatched = 0usize;
    let mut n_signed = 0usize;
    let mut n_unsigned = 0usize;
    let mut seen_raw_keys: BTreeSet<(String, i32, String)> = BTreeSet::new();
    let mut seen_resolved_keys: BTreeSet<(String, i32, String)> = BTreeSet::new();

    for (oi, obs) in observations.iter().enumerate() {
        if !is_supported_exchange_atom_any_residue(obs.res_name.as_deref(), &obs.atom) {
            return Err(format!(
                "observation #{oi}: {UNSUPPORTED_EXCHANGE_ATOM_MESSAGE}"
            ));
        }
        if !obs.dw_ppm.is_finite() || !obs.dw_err_ppm.is_finite() || obs.dw_err_ppm <= 0.0 {
            return Err(format!(
                "observation #{oi} (seq_id={}, atom='{}') has invalid dw_ppm={} or dw_err_ppm={}",
                obs.seq_id, obs.atom, obs.dw_ppm, obs.dw_err_ppm
            ));
        }
        if obs.sign_mode == ExchangeSignMode::UnsignedMagnitude && obs.dw_ppm < 0.0 {
            return Err(format!(
                "observation #{oi} (seq_id={}, atom='{}') is UnsignedMagnitude but has negative dw_ppm={}",
                obs.seq_id, obs.atom, obs.dw_ppm
            ));
        }

        let pre_canon_atom = canonicalize_exchange_atom_name(obs.res_name.as_deref(), &obs.atom);
        let raw_chain = obs.chain_id.trim();
        let raw_chain_norm = if raw_chain.is_empty() { "A" } else { raw_chain };
        if !seen_raw_keys.insert((
            raw_chain_norm.to_string(),
            obs.seq_id,
            pre_canon_atom.clone(),
        )) {
            return Err(format!(
                "duplicate exchange observation for (chain_id='{raw_chain_norm}', seq_id={}, atom='{pre_canon_atom}')",
                obs.seq_id
            ));
        }

        let Some(res_idx) = by_chain_seq
            .get(&(obs.chain_id.as_str(), obs.seq_id))
            .copied()
            .or_else(|| by_seq_only.get(&obs.seq_id).copied())
        else {
            n_unmatched += 1;
            warnings.push(format!(
                "skipped unmatched exchange observation at chain '{}' seq_id {} atom '{}'",
                obs.chain_id, obs.seq_id, obs.atom
            ));
            continue;
        };

        let r_struct = &ground_structure.residues[res_idx];
        if let Some(ref exp_rn) = obs.res_name {
            let norm_exp = exp_rn.trim().to_ascii_uppercase();
            if !norm_exp.is_empty() && norm_exp != r_struct.res_name {
                let compat_his = r_struct.res_name == "HIS"
                    && matches!(
                        norm_exp.as_str(),
                        "HID" | "HIE" | "HIP" | "HSD" | "HSE" | "HSP"
                    );
                let compat_cys =
                    r_struct.res_name == "CYS" && matches!(norm_exp.as_str(), "CYX" | "CYM");
                if !compat_his && !compat_cys {
                    return Err(format!(
                        "residue mismatch at seq_id {}: exchange table has '{}', structure has '{}'",
                        obs.seq_id, norm_exp, r_struct.res_name
                    ));
                }
            }
        }

        let canon_atom = canonicalize_exchange_atom_name(Some(&r_struct.res_name), &obs.atom);
        if !is_supported_exchange_atom(&canon_atom) {
            return Err(format!(
                "observation #{oi} (seq_id={}): {UNSUPPORTED_EXCHANGE_ATOM_MESSAGE}",
                obs.seq_id
            ));
        }
        if !seen_resolved_keys.insert((
            r_struct.chain_id.clone(),
            r_struct.seq_id,
            canon_atom.clone(),
        )) {
            return Err(format!(
                "duplicate exchange observation for (chain_id='{}', seq_id={}, atom='{canon_atom}')",
                r_struct.chain_id, r_struct.seq_id
            ));
        }

        let (pred_struct_opt, pred_rc_opt) =
            lookup_pred_struct_and_rc(&pred_a, res_idx, &canon_atom);
        let gs_ppm_opt = match (ground_shifts, ref_map_by_res_idx[res_idx]) {
            (Some(gs), Some(ri)) => lookup_reference_shift(gs, ri, &r_struct.res_name, &canon_atom),
            _ => None,
        };
        let explicit_ground_ppm = obs.major_ppm.or(gs_ppm_opt);
        let ground_ppm_opt = explicit_ground_ppm.or(pred_struct_opt);
        let coil_ppm = pred_rc_opt.unwrap_or_else(|| {
            fallback_sidechain_coil_ppm(
                &r_struct.res_name,
                &canon_atom,
                ground_ppm_opt,
                r_struct.omega_deg,
                cfg.sample_ph,
            )
        });
        let ground_ppm = ground_ppm_opt.unwrap_or(coil_ppm);

        let sigma_fwd =
            cfg.sigma_fwd_scale * differential_forward_sigma_ppm(&r_struct.res_name, &canon_atom);
        let sigma_tot = (obs.dw_err_ppm * obs.dw_err_ppm + sigma_fwd * sigma_fwd).sqrt();
        let spin_coords = resolve_spin_coords(ground_structure, res_idx, &canon_atom);

        match obs.sign_mode {
            ExchangeSignMode::Signed => n_signed += 1,
            ExchangeSignMode::UnsignedMagnitude => n_unsigned += 1,
        }
        let _ = MethylGroupKind::AlaCb;

        resolved.push(ResolvedObs {
            res_idx,
            chain_id: r_struct.chain_id.clone(),
            seq_id: r_struct.seq_id,
            res_name: r_struct.res_name.clone(),
            atom: canon_atom,
            sign_mode: obs.sign_mode,
            dw_ppm: obs.dw_ppm,
            dw_err_ppm: obs.dw_err_ppm,
            sigma_fwd_ppm: sigma_fwd,
            sigma_tot_ppm: sigma_tot,
            ground_ppm,
            explicit_ground_ppm,
            pred_a_struct_ppm: pred_struct_opt,
            coil_target_ppm: coil_ppm,
            spin_coords,
        });
    }

    if resolved.is_empty() {
        return Err("zero exchange observations matched residues in ground_structure".to_string());
    }

    if cfg.cpmg_prefiltered_significant_only {
        warnings.push(
            "CPMG input pre-filtered by significant_exchange=true (non-exchanging Delta_varpi=0 rows excluded by pre-filter)"
                .to_string(),
        );
    }

    let distinct_res_set: BTreeSet<(String, i32)> = resolved
        .iter()
        .map(|o| (o.chain_id.clone(), o.seq_id))
        .collect();
    let n_distinct_residues = distinct_res_set.len();

    let distinct_exch_res_set: BTreeSet<(String, i32)> = resolved
        .iter()
        .filter(|o| o.dw_ppm.abs() > 2.0 * o.dw_err_ppm)
        .map(|o| (o.chain_id.clone(), o.seq_id))
        .collect();
    let n_distinct_exchanging_residues = distinct_exch_res_set.len();

    // Evaluate H0 (Null, parameterized solely with experimental sigma_exp) and biophysical hypotheses H1..H4
    let mut hyps = Vec::with_capacity(5);
    let zeros = vec![0.0_f64; resolved.len()];
    let h0 = build_hypothesis_score(
        ExcitedStateHypothesisKind::NullOrUnstructuredPerturbation,
        0,
        n_distinct_residues,
        0.0,
        1,
        &resolved,
        BestCandidateState {
            log_likelihood: 0.0,
            mu_preds: zeros,
            summary:
                "NullOrUnstructuredPerturbation (mu_Delta = 0, sigma = sigma_exp across all spins)"
                    .to_string(),
            affected_seq_ids: Vec::new(),
            parameters: HypothesisBestParameters::Null,
        },
        Vec::new(),
    )?;
    let bic_null = h0.bic;
    let h0_reduced_chi2 = h0.reduced_chi2;

    // Empirical unmodeled-exchange Bayes factor against H0 (for sites exceeding |z_0| > 3.5)
    let mut unmod_delta_ll = 0.0;
    let mut unmod_active_res: BTreeSet<(String, i32)> = BTreeSet::new();
    for (o, p0) in resolved.iter().zip(h0.predictions.iter()) {
        if p0.z_residual.abs() > 3.5 {
            let (ll_fit, _, _) = eval_single_obs_ll_and_z_with_sigma(o, o.dw_ppm, o.dw_err_ppm)?;
            if ll_fit > p0.log_likelihood {
                unmod_delta_ll += ll_fit - p0.log_likelihood;
                unmod_active_res.insert((o.chain_id.clone(), o.seq_id));
            }
        }
    }
    let ln_n_res = (n_distinct_residues.max(1) as f64).ln();
    let unmodeled_exchange_two_ln_bf_vs_null =
        (2.0 * unmod_delta_ll - (unmod_active_res.len() as f64) * ln_n_res).max(0.0);

    hyps.push(h0);

    if let Some(h1) = evaluate_h1_local_unfolding(ground_structure, &resolved, n_distinct_residues)?
    {
        hyps.push(h1);
    }
    if let Some(h2) = evaluate_h2_proline_cis_trans(
        ground_structure,
        ground_shifts,
        &ref_map_by_res_idx,
        &resolved,
        n_distinct_residues,
    )? {
        hyps.push(h2);
    }
    if let Some(h3) = evaluate_h3_rotamer_or_his_flip(
        ground_structure,
        ground_shifts,
        &ref_map_by_res_idx,
        &resolved,
        n_distinct_residues,
    )? {
        hyps.push(h3);
    }
    if let Some(h4) = evaluate_h4_alt_conformers(
        ground_structure,
        &pred_a,
        alt_structures,
        firewall_query,
        &struct_cfg,
        &resolved,
        n_distinct_residues,
    )? {
        hyps.push(h4);
    }

    // Sort hypotheses ascending by BIC (lowest BIC = rank 1)
    hyps.sort_by(|a, b| a.bic.total_cmp(&b.bic));

    let min_bic = hyps[0].bic;
    let two_ln_bf_12 = if hyps.len() >= 2 {
        hyps[1].bic - hyps[0].bic
    } else {
        0.0
    };
    let mut weight_sum = 0.0;
    for h in &mut hyps {
        h.delta_bic_from_best = h.bic - min_bic;
        h.two_ln_bf_vs_null = bic_null - h.bic;
        let w = (-0.5 * h.delta_bic_from_best).exp();
        h.posterior_weight = w;
        weight_sum += w;
    }
    if weight_sum > 0.0 {
        for h in &mut hyps {
            h.posterior_weight /= weight_sum;
        }
    }
    hyps[0].two_ln_bf_vs_runner_up = Some(two_ln_bf_12);

    // Pairwise head-to-head non-null comparison for every non-null hypothesis
    let non_null_bics: Vec<(ExcitedStateHypothesisKind, f64)> = hyps
        .iter()
        .filter(|h| h.kind != ExcitedStateHypothesisKind::NullOrUnstructuredPerturbation)
        .map(|h| (h.kind, h.bic))
        .collect();
    for h in &mut hyps {
        if h.kind != ExcitedStateHypothesisKind::NullOrUnstructuredPerturbation {
            if let Some(&(_, best_other_bic)) = non_null_bics.iter().find(|&&(k, _)| k != h.kind) {
                h.two_ln_bf_vs_best_other_non_null = Some(best_other_bic - h.bic);
            }
        }
    }
    let (two_ln_bf_top1_vs_runnerup_non_null, runnerup_non_null_hypothesis) =
        if hyps[0].kind != ExcitedStateHypothesisKind::NullOrUnstructuredPerturbation {
            match non_null_bics
                .iter()
                .find(|&&(k, _)| k != hyps[0].kind)
                .copied()
            {
                Some((rk, rbic)) => (Some(rbic - hyps[0].bic), Some(rk)),
                None => (None, None),
            }
        } else {
            (None, None)
        };

    let top = &hyps[0];
    let n_obs = resolved.len();
    let top_bf_null = top.two_ln_bf_vs_null;
    let top_rchi2 = top.reduced_chi2;
    let h0_rejected_by_data = h0_reduced_chi2 > cfg.max_reduced_chi2
        || top_bf_null >= cfg.min_two_ln_bf_null
        || unmodeled_exchange_two_ln_bf_vs_null >= cfg.min_two_ln_bf_null;

    let (verdict, abstention_kind, verdict_reason) = if n_distinct_residues < cfg.min_obs_to_commit
    {
        (
            ExcitedStateVerdict::AmbiguousOrUnmodeledState,
            Some(ExcitedStateAbstentionKind::InsufficientDistinctResidues),
            format!(
                "abstained: distinct residue count N_res={} (N_obs={}) < minimum {}",
                n_distinct_residues, n_obs, cfg.min_obs_to_commit
            ),
        )
    } else if top.kind == ExcitedStateHypothesisKind::NullOrUnstructuredPerturbation {
        if h0_rejected_by_data {
            (
                ExcitedStateVerdict::AmbiguousOrUnmodeledState,
                Some(ExcitedStateAbstentionKind::UnmodeledSignificantExchange),
                format!(
                    "UnmodeledSignificantExchange: significant exchange rejects H0 (H0 reduced chi^2_nu={:.3} > {:.2}, unmodeled 2 ln BF={:.2}), but no modeled mechanism H1..H4 fits the shift pattern",
                    h0_reduced_chi2, cfg.max_reduced_chi2, unmodeled_exchange_two_ln_bf_vs_null
                ),
            )
        } else {
            (
                ExcitedStateVerdict::AmbiguousOrUnmodeledState,
                Some(ExcitedStateAbstentionKind::ConsistentWithNoise),
                format!(
                    "abstained: H0 (NullOrUnstructuredPerturbation) has the lowest BIC and fits within experimental noise (H0 reduced chi^2_nu={:.3} <= {:.2})",
                    h0_reduced_chi2, cfg.max_reduced_chi2
                ),
            )
        }
    } else if top_rchi2 > cfg.max_reduced_chi2 {
        (
            ExcitedStateVerdict::AmbiguousOrUnmodeledState,
            Some(ExcitedStateAbstentionKind::UnmodeledSignificantExchange),
            format!(
                "UnmodeledSignificantExchange: H0 rejected (2 ln BF_(1,H0)={:.2}, H0 reduced chi^2_nu={:.3}), yet top hypothesis {} has reduced chi^2_nu={:.3} > {:.2} goodness-of-fit ceiling",
                top_bf_null,
                h0_reduced_chi2,
                top.kind.as_str(),
                top_rchi2,
                cfg.max_reduced_chi2
            ),
        )
    } else if top_bf_null < cfg.min_two_ln_bf_null {
        if h0_rejected_by_data {
            (
                ExcitedStateVerdict::AmbiguousOrUnmodeledState,
                Some(ExcitedStateAbstentionKind::UnmodeledSignificantExchange),
                format!(
                    "UnmodeledSignificantExchange: significant exchange rejects H0 (H0 reduced chi^2_nu={:.3}, unmodeled 2 ln BF={:.2}), while top hypothesis {} has 2 ln BF_(1,H0) = {:.2} < {:.2}",
                    h0_reduced_chi2,
                    unmodeled_exchange_two_ln_bf_vs_null,
                    top.kind.as_str(),
                    top_bf_null,
                    cfg.min_two_ln_bf_null
                ),
            )
        } else {
            (
                ExcitedStateVerdict::AmbiguousOrUnmodeledState,
                Some(ExcitedStateAbstentionKind::ConsistentWithNoise),
                format!(
                    "abstained: 2 ln BF_(1,H0) = {:.2} < {:.2} strong-evidence threshold over Null",
                    top_bf_null, cfg.min_two_ln_bf_null
                ),
            )
        }
    } else if two_ln_bf_12 < cfg.min_two_ln_bf_12 {
        (
            ExcitedStateVerdict::AmbiguousOrUnmodeledState,
            Some(ExcitedStateAbstentionKind::AmbiguousBetweenMechanisms),
            format!(
                "abstained: 2 ln BF_(1,2) = {:.2} < {:.2} Kass-Raftery positive/strong separation over runner-up {}",
                two_ln_bf_12,
                cfg.min_two_ln_bf_12,
                hyps.get(1).map(|h| h.kind.as_str()).unwrap_or("none")
            ),
        )
    } else if n_distinct_exchanging_residues < cfg.min_obs_to_commit {
        (
            ExcitedStateVerdict::AmbiguousOrUnmodeledState,
            Some(ExcitedStateAbstentionKind::InsufficientDistinctResidues),
            format!(
                "abstained: distinct exchanging residue count N_exch_res={} (N_res={}, N_obs={}) < minimum {} required to commit",
                n_distinct_exchanging_residues,
                n_distinct_residues,
                n_obs,
                cfg.min_obs_to_commit
            ),
        )
    } else {
        (
            ExcitedStateVerdict::Committed(top.kind),
            None,
            format!(
                "committed {}: N_exch_res={} (N_res={}, N_obs={}), 2 ln BF_(1,2)={:.2} >= {:.1}, 2 ln BF_(1,H0)={:.2} >= {:.1}, reduced chi^2_nu={:.3} <= {:.2}",
                top.kind.as_str(),
                n_distinct_exchanging_residues,
                n_distinct_residues,
                n_obs,
                two_ln_bf_12,
                cfg.min_two_ln_bf_12,
                top_bf_null,
                cfg.min_two_ln_bf_null,
                top_rchi2,
                cfg.max_reduced_chi2
            ),
        )
    };

    // ------------------------------------------------------------------------
    // Error-scale consistency (`21F:B19`). Evidence of MIS-SPECIFIED errors only:
    //   (i)  supplied control sites: chi^2_ctrl > chi^2_{0.99, n_ctrl};
    //   (ii) otherwise the best-fitting model's residuals at its spectator sites (predicted
    //        Delta_varpi = 0, so sigma = dw_err only) after robust sigma clipping
    //        (|z| <= 3.5 x max(1, s_MAD)), when >= 5 inliers form a majority of spectators:
    //        chi^2 > chi^2_{0.99, n_in};
    //   (iii) otherwise all sites: chi^2 > chi^2_{0.99, n_obs - k}.
    // H0 rejection is NOT evidence of error mis-scaling (it is what every real detection does).
    // ------------------------------------------------------------------------
    let top_k = top.num_params_k;
    let top_dof = n_obs.saturating_sub(top_k).max(1);
    let consistency_upper_bound = reduced_chi2_upper_bound(top_dof);
    let top1_exceeds_consistency_bound = top_rchi2 > consistency_upper_bound;
    let h0_consistency_upper_bound = reduced_chi2_upper_bound(n_obs.max(1));
    let h0_exceeds_consistency_bound = h0_reduced_chi2 > h0_consistency_upper_bound;
    let implied_error_scale_factor_h0 = h0_reduced_chi2.max(0.0).sqrt();
    let implied_error_scale_factor_top1 = top_rchi2.max(0.0).sqrt();

    // Internal controls: "spectator" sites where the best model predicts no shift, so their
    // likelihood sigma is purely the stated experimental error (forward-model sigma ramped to
    // ~0; `sigma_tot <= 1.05 dw_err`). Residuals at sites with a non-zero prediction also carry
    // the heuristic forward-model error and are therefore NOT evidence about dw_err_ppm.
    let mut abs_z: Vec<f64> = top
        .predictions
        .iter()
        .filter(|p| p.sigma_tot_ppm <= SPECTATOR_SIGMA_RATIO * p.observed_err_ppm)
        .map(|p| p.z_residual.abs())
        .filter(|z| z.is_finite())
        .collect();
    let n_spectators = abs_z.len();
    abs_z.sort_by(|a, b| a.total_cmp(b));
    let median_abs_z = if abs_z.is_empty() {
        0.0
    } else if abs_z.len() % 2 == 1 {
        abs_z[abs_z.len() / 2]
    } else {
        0.5 * (abs_z[abs_z.len() / 2 - 1] + abs_z[abs_z.len() / 2])
    };
    let robust_scale_mad = median_abs_z / HALF_NORMAL_MEDIAN;
    let outlier_clip_abs_z = ERROR_SCALE_OUTLIER_ABS_Z * robust_scale_mad.max(1.0);
    let inlier_z2: Vec<f64> = abs_z
        .iter()
        .filter(|&&z| z <= outlier_clip_abs_z)
        .map(|z| z * z)
        .collect();
    let n_inliers = inlier_z2.len();
    let n_outlier_sites_excluded = n_spectators.saturating_sub(n_inliers);
    let inliers_usable =
        n_inliers >= MIN_ERROR_SCALE_SPECTATOR_SITES && 2 * n_inliers > n_spectators;
    let (inlier_reduced_chi2, inlier_dof, inlier_consistency_upper_bound) = if inliers_usable {
        // Spectator residuals are not fitted (mu = 0), so nu = n_inliers.
        let dof = n_inliers;
        (
            Some(inlier_z2.iter().sum::<f64>() / dof as f64),
            Some(dof),
            Some(reduced_chi2_upper_bound(dof)),
        )
    } else {
        (None, None, None)
    };

    let n_control_sites = control_sites.len();
    let (control_reduced_chi2, control_consistency_upper_bound, control_exceeds_consistency_bound) =
        if n_control_sites > 0 {
            let rc = control_sum_z2 / n_control_sites as f64;
            let b = reduced_chi2_upper_bound(n_control_sites);
            (Some(rc), Some(b), Some(rc > b))
        } else {
            (None, None, None)
        };

    let (
        error_scale_inconsistent,
        error_scale_evidence,
        evidence_rchi2,
        evidence_bound,
        evidence_dof,
    ) = if let (Some(rc), Some(b), Some(ex)) = (
        control_reduced_chi2,
        control_consistency_upper_bound,
        control_exceeds_consistency_bound,
    ) {
        (ex, "control_sites", rc, b, n_control_sites)
    } else if let (Some(rc), Some(b), Some(dof)) = (
        inlier_reduced_chi2,
        inlier_consistency_upper_bound,
        inlier_dof,
    ) {
        (rc > b, "best_model_spectator_inliers", rc, b, dof)
    } else {
        (
            top1_exceeds_consistency_bound,
            "best_model_all_sites",
            top_rchi2,
            consistency_upper_bound,
            top_dof,
        )
    };

    let error_scale_warning = error_scale_inconsistent.then(|| {
        let evidence_txt = match error_scale_evidence {
            "control_sites" => format!(
                "{n_control_sites} non-exchanging control sites have reduced chi^2_nu={evidence_rchi2:.3} > chi^2_{{0.99,{evidence_dof}}}/{evidence_dof}={evidence_bound:.3}"
            ),
            "best_model_spectator_inliers" => format!(
                "best-fitting model {} has reduced chi^2_nu={evidence_rchi2:.3} over its {n_inliers} inlier spectator sites (predicted dw = 0 (|z|<={outlier_clip_abs_z:.2}, i.e. 3.5 x max(1, s_MAD={robust_scale_mad:.2}); {n_outlier_sites_excluded} localized outliers excluded)) > chi^2_{{0.99,{evidence_dof}}}/{evidence_dof}={evidence_bound:.3}",
                top.kind.as_str()
            ),
            _ => format!(
                "best-fitting model {} has reduced chi^2_nu={evidence_rchi2:.3} over all {n_obs} sites > chi^2_{{0.99,{evidence_dof}}}/{evidence_dof}={evidence_bound:.3}",
                top.kind.as_str()
            ),
        };
        format!(
            "Error-scale / chi^2 consistency diagnostic: {evidence_txt} (implied error scale ~{:.2}x). The stated dw_err_ppm are statistically inconsistent with the data. Under-stated experimental error bars (e.g. 2x-3x) produce exactly this signature and cause false H0 rejection and spurious UnmodeledSignificantExchange or non-null hypothesis ranking, so no confident call is reported (ErrorScaleInconsistent); calibrate dw_err_ppm on non-exchanging control sites (exchange TSV site_role=control).",
            evidence_rchi2.max(0.0).sqrt()
        )
    });

    // Downgrade any confident outcome when the stated errors are inconsistent with the data.
    let (verdict, abstention_kind, verdict_reason) = if error_scale_inconsistent
        && (verdict.is_committed()
            || matches!(
                abstention_kind,
                Some(ExcitedStateAbstentionKind::UnmodeledSignificantExchange)
                    | Some(ExcitedStateAbstentionKind::ConsistentWithNoise)
            )) {
        let pre = match abstention_kind {
            Some(k) => k.as_str().to_string(),
            None => verdict.as_str().to_string(),
        };
        (
            ExcitedStateVerdict::AmbiguousOrUnmodeledState,
            Some(ExcitedStateAbstentionKind::ErrorScaleInconsistent),
            format!(
                "ErrorScaleInconsistent: abstained because the stated dw_err_ppm fail the chi^2 consistency check ({error_scale_evidence}: reduced chi^2_nu={evidence_rchi2:.3} > {evidence_bound:.3} at nu={evidence_dof}, p=0.99); the {pre} outcome that would otherwise be reported is equally explained by under-stated error bars and is not a confident call. Pre-check reason: {verdict_reason}"
            ),
        )
    } else {
        (verdict, abstention_kind, verdict_reason)
    };

    if let Some(ref w) = error_scale_warning {
        warnings.push(w.clone());
    }
    let chi2_consistency_diagnostic = Some(Chi2ConsistencyDiagnostic {
        h0_reduced_chi2,
        top1_reduced_chi2: top_rchi2,
        consistency_upper_bound,
        h0_exceeds_consistency_bound,
        top1_exceeds_consistency_bound,
        implied_error_scale_factor_h0,
        implied_error_scale_factor_top1,
        error_scale_warning: error_scale_warning.clone(),
        confidence_level: ERROR_SCALE_CHI2_CONFIDENCE,
        top1_hypothesis: Some(top.kind),
        top1_dof: top_dof,
        h0_consistency_upper_bound,
        n_spectator_sites: n_spectators,
        robust_scale_mad,
        n_outlier_sites_excluded,
        inlier_reduced_chi2,
        inlier_dof,
        inlier_consistency_upper_bound,
        n_control_sites,
        control_reduced_chi2,
        control_consistency_upper_bound,
        control_exceeds_consistency_bound,
        error_scale_evidence: error_scale_evidence.to_string(),
        error_scale_inconsistent,
    });

    Ok(ExcitedStateReport {
        verdict,
        verdict_reason,
        abstention_kind,
        n_observations: n_obs,
        n_distinct_residues,
        n_distinct_exchanging_residues,
        n_signed_observations: n_signed,
        n_unsigned_observations: n_unsigned,
        n_unmatched_observations: n_unmatched,
        two_ln_bf_top1_vs_top2: two_ln_bf_12,
        two_ln_bf_top1_vs_runnerup_non_null,
        runnerup_non_null_hypothesis,
        two_ln_bf_top1_vs_null: top_bf_null,
        unmodeled_exchange_two_ln_bf_vs_null,
        top1_reduced_chi2: top_rchi2,
        cpmg_prefiltered_significant_only: cfg.cpmg_prefiltered_significant_only,
        sample_ph: cfg.sample_ph,
        temperature_kelvin: cfg.temperature_kelvin,
        ground_exclusion_list_sha256: pred_a.exclusion_list_sha256,
        ground_max_retained_seq_identity: pred_a.max_retained_seq_identity,
        ranked_hypotheses: hyps,
        warnings,
        error_scale_warning,
        chi2_consistency_diagnostic,
        scope_limits_note: EXCITED_STATE_SCOPE_LIMITS_NOTE.to_string(),
    })
}
