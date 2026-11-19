//! Symmetrized `PAE` normalized-Laplacian spectral domain clustering, Tolman $\vartheta_d / \vartheta_1$
//! Generalized Degree of Order (`GDO`) inter-domain motion testing, covalent rigid-domain + `NeRF`
//! flexible-linker kinematics (`0.000 Å` intra-domain distortion), and hybrid multi-medium
//! `RDC` / `PCS` / `PRE` / inter-domain `NOE` multi-domain docking (`T21.5`,
//! `docs/specs/21-alphafold-integrative-nmr.md` §5).
//!
//! Provenance: Clean-room original implementation derived strictly from peer-reviewed literature
//! (`0` copyleft, academic-licence, or proprietary software consulted; `0` external product names):
//! 1. Ng, A. Y., Jordan, M. I. & Weiss, Y. (2001). "On spectral clustering: analysis and an
//!    algorithm." *Adv. Neural Inf. Process. Syst.* **14**, 849-856; and Von Luxburg, U. (2007).
//!    "A tutorial on spectral clustering." *Stat. Comput.* **17**(4), 395-416 — normalized
//!    symmetric graph Laplacian $\mathbf{L}_{\text{sym}} = \mathbf{I} - \mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$
//!    and eigengap model selection.
//! 2. Hubert, L. & Arabie, P. (1985). "Comparing partitions." *J. Classif.* **2**(1), 193-218 —
//!    chance-corrected Adjusted Rand Index (`ARI`).
//! 3. Tolman, J. R., Al-Hashimi, H. M., Kay, L. E. & Prestegard, J. H. (2001). "Structural and
//!    dynamic analysis of residual dipolar coupling data for proteins." *J. Am. Chem. Soc.*
//!    **123**(7), 1416-1424 — Generalized Degree of Order ($\vartheta = \sqrt{\frac{2}{3}\sum_{\alpha,\beta} S_{\alpha\beta}^2}$)
//!    and inter-domain orientational motion detection.
//! 4. Braddock, D. T., Cai, M., Baber, J. L., Huang, Y. & Clore, G. M. (2001). *J. Am. Chem. Soc.*
//!    **123**(35), 8634-8635; and Clore, G. M. & Schwieters, C. D. (2004). *J. Am. Chem. Soc.*
//!    **126**(9), 2923-2938 — dipolar coupling and Solomon `PRE` ($r_{\text{M}}^{-6}$) rigid-body
//!    multi-domain and linker-torsion refinement.
//! 5. Bertini, I., Gupta, Y. K., Luchinat, C., Parigi, G., Peana, M., Sgheri, L. & Yuan, J. (2007).
//!    *J. Am. Chem. Soc.* **129**(42), 12786-12794; and Schmitz, C. et al. (2008). *J. Biomol. NMR*
//!    **41**(3), 179-189 — simultaneous translational and orientational multi-domain determination
//!    from paramagnetic $\Delta\boldsymbol{\chi}$ pseudocontact shifts (`PCS`).
//! 6. Parsons, J., Holmes, J. B., Rojas, J. M., Tsai, J. & Strauss, C. E. M. (2005). *J. Comput. Chem.*
//!    **26**(10), 1063-1068 (`NeRF`); Abe, H., Braun, W., Noguti, T. & Go, N. (1984). *Comput. Chem.*
//!    **8**(4), 239-247 (`O(L)` suffix-sum torque gradient); Nocedal, J. (1980). *Math. Comp.*
//!    **35**(151), 773-782 (two-loop `L-BFGS`); Lovell, S. C. et al. (2003). *Proteins* **50**(3),
//!    437-450 (Ramachandran torsion prior).

use std::collections::{BTreeMap, BTreeSet};
use std::f64::consts::PI;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::af_concordance::{AfResidueRecord, AfStructureModel, PaeMatrix};
use crate::noesy::{
    calc_dihedral_deg, reconstruct_ca_substituents, v_add, v_cross, v_dist, v_dist_sq, v_dot,
    v_norm, v_scale, v_sub, v_unit, AmbiguousDistanceRestraint, NoeProton, PdbChain, PdbResidue,
    Vec3,
};
use crate::paramagnetic::{
    fit_pcs_tensor_svd, refine_metal_position_and_tensor, DeltaChiTensor, PcsObservation,
    PCS_CARTESIAN_PREFACTOR,
};
use crate::rdc::{
    calc_rdc_from_unit_vec, cornilescu_q, decompose_saupe_matrix, normalize_rdc_atom_name,
    rdc_peptide_plane_id, RdcBackCalcPoint, RdcBondType, RdcFreeStratification, RdcMediumSet,
    RdcRestraint, RdcValidationSummary, SaupeTensorFit, DMAX_NH_HZ, MAX_SAUPE_SVD_COND,
};
use crate::structure::{
    det_acos, det_cbrt, det_exp, det_ln, det_sin_cos, kabsch_rmsd, nerf_place_atom, DetRng,
    ANGLE_CA_C_N_DEG, ANGLE_CA_C_O_DEG, ANGLE_C_N_CA_DEG, ANGLE_C_N_H_DEG, ANGLE_N_CA_C_DEG,
    ANGLE_N_CA_C_GLY_DEG, BOND_C_N, BOND_N_H,
};

/// Maximum restraint text size (`64 MiB`) accepted by `PCS` / `PRE` / `RDC` / `NOE` parsers.
pub const MAX_DOMAIN_RESTRAINT_BYTES: usize = 64 * 1024 * 1024;

/// Default minimum `pLDDT` threshold defining the ordered residue subgraph $\mathcal{O}$.
pub const DEFAULT_MIN_ORDERED_PLDDT: f64 = 70.0;

/// Default Gaussian kernel width $\sigma_{\text{PAE}} = 5.0\text{ \AA}$ on symmetrized `PAE`.
pub const DEFAULT_PAE_SIGMA_ANGSTROM: f64 = 5.0;

/// Default hard cutoff (`12.0 Å`) above which symmetrized `PAE` affinity edges vanish.
pub const DEFAULT_PAE_AFFINITY_CUTOFF_ANGSTROM: f64 = 12.0;

/// Default maximum intra-domain median $\overline{\text{PAE}}_{\text{intra}} \le 5.0\text{ \AA}$.
pub const DEFAULT_MAX_INTRA_PAE_MEDIAN_ANGSTROM: f64 = 5.0;

/// Default minimum residue count $|\mathcal{D}_d| \ge 20$ for an autonomous globular domain.
pub const DEFAULT_MIN_DOMAIN_RESIDUES: usize = 20;

/// Default maximum candidate domain count $D_{\max} = 8$ in Laplacian eigengap search.
pub const DEFAULT_MAX_PAE_DOMAINS: usize = 8;

/// Minimum working (`!is_free`) RDC count per `(domain, medium)` required for Tolman GDO SVD.
pub const MIN_TOLMAN_DOMAIN_WORK_RDCS: usize = 8;

/// Default Tolman GDO inter-domain coupling threshold (`\rho_{\text{GDO}} < 0.80` at `> 2\sigma_{\text{MC}}`).
pub const TOLMAN_GDO_RATIO_THRESHOLD: f64 = 0.80;

/// Default backbone/proton coordinate noise floor (`0.08 Å`, $\approx 4.5^\circ$ on N–H) propagated
/// into Tolman $\vartheta_{d,m}$ Monte Carlo uncertainty alongside measurement $\sigma_k$ and
/// empirical intra-domain SVD excess $\chi^2_\nu$ (`21.W2:P1-6`).
pub const DEFAULT_TOLMAN_COORD_NOISE_ANGSTROM: f64 = 0.08;

/// Default pre-dock intra-domain `ByPeptidePlane Q_free` ceiling (`0.50`) above which a rigid
/// domain fails its own internal coordinate fit (`21.W2:P1-6` / `A8`).
pub const DEFAULT_MAX_INTRA_DOMAIN_RDC_Q_FREE: f64 = 0.50;

/// Maximum Saupe Frobenius cosine similarity (`|cos(S_a, S_b)| < 0.85`, i.e. $\ge 31.8^\circ$)
/// required to count two RDC alignment media as genuinely non-coaxial (`21.W2:P2-8`).
pub const MAX_NON_COAXIAL_SAUPE_COS: f64 = 0.85;

/// Default outward offset (`7.0 Å`, typical `6–9 Å` DOTA-M8 / CLaNP / EDTA-Cys chelate arm) used
/// as the initial metal guess when `# metal_anchor:` is supplied without `# metal_pos:` (`21.W2:P2-10`).
pub const DEFAULT_LANTHANIDE_ANCHOR_OFFSET_ANGSTROM: f64 = 7.0;

/// Default MTSL nitroxide 5-bond Cys–S–S–CH₂–pyrroline flexible-tether distance allowance (`6.0 Å`)
/// applied when spin-label PRE bounds are anchored to a residue atom rather than an exact fixed
/// 3D paramagnetic center (`Battiste & Wagner 2000`; `Iwahara, Schwieters & Clore 2004`).
pub const DEFAULT_MTSL_TETHER_ALLOWANCE_ANGSTROM: f64 = 6.0;

/// CODATA 2018/2022 vacuum magnetic permeability factor $\mu_0 / (4\pi) = 10^{-7}\text{ T}\cdot\text{m}\cdot\text{A}^{-1}$.
pub const CODATA_MU0_OVER_4PI_SI: f64 = 1.0e-7;

/// CODATA 2018/2022 $^1\text{H}$ gyromagnetic ratio $\gamma_{\text{H}} = 2.675\,221\,874\,4 \times 10^8\text{ rad}\cdot\text{s}^{-1}\cdot\text{T}^{-1}$.
pub const CODATA_GAMMA_PROTON_RAD_S_T: f64 = 2.675_221_874_4e8;

/// CODATA 2018/2022 free-electron $g$-factor $g_e = 2.002\,319\,304\,362\,56$.
pub const CODATA_ELECTRON_G_FACTOR: f64 = 2.002_319_304_362_56;

/// CODATA 2018/2022 Bohr magneton $\mu_B = 9.274\,010\,078\,3 \times 10^{-24}\text{ J}\cdot\text{T}^{-1}$.
pub const CODATA_BOHR_MAGNETON_J_T: f64 = 9.274_010_078_3e-24;

/// First-principles SI Solomon–Bloembergen dipolar relaxation prefactor for $^1\text{H}$ + $S = 1/2$
/// nitroxide radical in $\text{\AA}^6\,\text{s}^{-2}$ (`Solomon, Phys. Rev. 1955, 99, 559-565`):
/// $$K_{\text{Solomon}} = \frac{1}{15}\left(\frac{\mu_0}{4\pi}\right)^2 \gamma_{\text{H}}^2 g_e^2 \mu_B^2 S(S+1) \times 10^{60} \approx 1.2339 \times 10^{16}\text{ \AA}^6\,\text{s}^{-2}.$$
pub const K_SOLOMON_PROTON_NITROXIDE_A6_S2: f64 = (1.0 / 15.0)
    * (CODATA_MU0_OVER_4PI_SI * CODATA_MU0_OVER_4PI_SI)
    * (CODATA_GAMMA_PROTON_RAD_S_T * CODATA_GAMMA_PROTON_RAD_S_T)
    * (CODATA_ELECTRON_G_FACTOR * CODATA_ELECTRON_G_FACTOR)
    * (CODATA_BOHR_MAGNETON_J_T * CODATA_BOHR_MAGNETON_J_T)
    * (0.5 * 1.5)
    * 1.0e60;

/// Compute the general SI Solomon–Bloembergen dipolar prefactor ($\text{\AA}^6\,\text{s}^{-2}$) for
/// nuclear gyromagnetic ratio `gamma_i_rad_s_t`, electronic $g$-factor `g_factor`, and spin quantum
/// number `spin_s` ($S \ge 1/2$, e.g. $S = 1/2$ for nitroxide/$\text{Cu}^{2+}$, $5/2$ for $\text{Mn}^{2+}$, $7/2$ for $\text{Gd}^{3+}$).
#[inline]
pub fn compute_solomon_dipolar_constant_a6_s2(
    spin_s: f64,
    g_factor: f64,
    gamma_i_rad_s_t: f64,
) -> Result<f64, String> {
    if !spin_s.is_finite() || spin_s <= 0.0 {
        return Err(format!("invalid electronic spin S ({spin_s}); must be > 0"));
    }
    if !g_factor.is_finite() || g_factor <= 0.0 {
        return Err(format!(
            "invalid electronic g_factor ({g_factor}); must be > 0"
        ));
    }
    if !gamma_i_rad_s_t.is_finite() || gamma_i_rad_s_t.abs() <= 0.0 {
        return Err(format!(
            "invalid nuclear gyromagnetic ratio ({gamma_i_rad_s_t})"
        ));
    }
    Ok((1.0 / 15.0)
        * (CODATA_MU0_OVER_4PI_SI * CODATA_MU0_OVER_4PI_SI)
        * (gamma_i_rad_s_t * gamma_i_rad_s_t)
        * (g_factor * g_factor)
        * (CODATA_BOHR_MAGNETON_J_T * CODATA_BOHR_MAGNETON_J_T)
        * (spin_s * (spin_s + 1.0))
        * 1.0e60)
}

/// Evaluate the effective Solomon–Bloembergen transverse dipolar factor $K_{\text{PRE}}$ in
/// $\text{\AA}^6\,\text{s}^{-1}$ such that $\Gamma_2(r) = K_{\text{PRE}}\,r^{-6}$:
/// $$K_{\text{PRE}}(\tau_c, \omega_{\text{H}}) = K_{\text{Solomon}}(S, g, \gamma_{\text{H}})\left(4\tau_c + \frac{3\tau_c}{1 + \omega_{\text{H}}^2 \tau_c^2}\right), \quad \omega_{\text{H}} = 2\pi \nu_{\text{H}}.$$
/// Fails closed (`Err`) if `tau_c_ns` or `proton_mhz` is non-positive or non-finite.
pub fn compute_solomon_k_pre_a6_s1(
    tau_c_ns: f64,
    proton_mhz: f64,
    spin_s: f64,
    g_factor: f64,
) -> Result<f64, String> {
    if !tau_c_ns.is_finite() || tau_c_ns <= 0.0 {
        return Err(format!(
            "PRE rotational correlation time tau_c_ns ({tau_c_ns}) must be finite and > 0 ns"
        ));
    }
    if !proton_mhz.is_finite() || proton_mhz <= 0.0 {
        return Err(format!(
            "PRE spectrometer proton Larmor frequency proton_mhz ({proton_mhz}) must be finite and > 0 MHz"
        ));
    }
    let k_sol =
        compute_solomon_dipolar_constant_a6_s2(spin_s, g_factor, CODATA_GAMMA_PROTON_RAD_S_T)?;
    let tau_c_s = tau_c_ns * 1.0e-9;
    let omega_h_rad_s = 2.0 * PI * (proton_mhz * 1.0e6);
    let wt = omega_h_rad_s * tau_c_s;
    let j_factor_s = 4.0 * tau_c_s + (3.0 * tau_c_s) / (1.0 + wt * wt);
    Ok(k_sol * j_factor_s)
}

/// Exact Solomon–Bloembergen $\Gamma_2(r)$ forward rate ($\text{s}^{-1}$) at distance `r_angstrom` (Å).
pub fn evaluate_solomon_gamma2_s1(
    r_angstrom: f64,
    tau_c_ns: f64,
    proton_mhz: f64,
    spin_s: f64,
    g_factor: f64,
) -> Result<f64, String> {
    if !r_angstrom.is_finite() || r_angstrom <= 0.0 {
        return Err(format!(
            "PRE internuclear distance r_angstrom ({r_angstrom}) must be finite and > 0 Å"
        ));
    }
    let k_pre = compute_solomon_k_pre_a6_s1(tau_c_ns, proton_mhz, spin_s, g_factor)?;
    let r2 = r_angstrom * r_angstrom;
    Ok(k_pre / (r2 * r2 * r2))
}

/// Invert a measured Solomon $\Gamma_2 \pm \sigma$ ($\text{s}^{-1}$) rate into `(r_center_a, r_lower_a, r_upper_a)`
/// in Å using exact $r = (K_{\text{PRE}} / \Gamma_2)^{1/6}$ plus configurable flexible-tether allowance
/// `mtsl_tether_allowance_a` ($\ge 0\text{ \AA}$, e.g. `6.0 Å` for MTSL).
pub fn solomon_gamma2_to_distance_bounds(
    gamma2_obs_s1: f64,
    sigma_s1: f64,
    k_pre_a6_s1: f64,
    mtsl_tether_allowance_a: f64,
) -> Result<(f64, f64, f64), String> {
    if !gamma2_obs_s1.is_finite() || gamma2_obs_s1 < 0.0 {
        return Err(format!("invalid gamma2_obs_s1 ({gamma2_obs_s1})"));
    }
    if !sigma_s1.is_finite() || sigma_s1 <= 0.0 {
        return Err(format!("invalid sigma_s1 ({sigma_s1})"));
    }
    if !k_pre_a6_s1.is_finite() || k_pre_a6_s1 <= 0.0 {
        return Err(format!("invalid k_pre_a6_s1 ({k_pre_a6_s1})"));
    }
    if !mtsl_tether_allowance_a.is_finite() || mtsl_tether_allowance_a < 0.0 {
        return Err(format!(
            "invalid mtsl_tether_allowance_a ({mtsl_tether_allowance_a}); must be >= 0"
        ));
    }
    let sixth_rt = |x: f64| -> f64 { det_cbrt(x.max(1e-30)).sqrt() };
    if gamma2_obs_s1 <= 2.0 * sigma_s1 {
        // Weak / baseline PRE: provides a lower distance bound only (`r >= r_min - tether`)
        let g_up = (gamma2_obs_s1 + 2.0 * sigma_s1).max(0.5);
        let r_min = (sixth_rt(k_pre_a6_s1 / g_up) - mtsl_tether_allowance_a).max(2.0);
        let r_center = if gamma2_obs_s1 > 1e-6 {
            sixth_rt(k_pre_a6_s1 / gamma2_obs_s1)
        } else {
            r_min
        };
        Ok((r_center, r_min, f64::INFINITY))
    } else {
        let r_center = sixth_rt(k_pre_a6_s1 / gamma2_obs_s1);
        let g_hi = gamma2_obs_s1 + 2.0 * sigma_s1;
        let g_lo = (gamma2_obs_s1 - 2.0 * sigma_s1)
            .max(0.25 * gamma2_obs_s1)
            .max(0.1);
        let r_lo = (sixth_rt(k_pre_a6_s1 / g_hi) - mtsl_tether_allowance_a).max(2.0);
        let r_hi = sixth_rt(k_pre_a6_s1 / g_lo) + mtsl_tether_allowance_a;
        Ok((r_center, r_lo.min(r_hi), r_hi))
    }
}

// ============================================================================
// 1. Symmetrized PAE Spectral Domain Clustering (Ng, Jordan & Weiss 2001)
// ============================================================================

/// Configuration for symmetrized `PAE` normalized-Laplacian spectral domain clustering.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PaeClusterConfig {
    /// Minimum per-residue `pLDDT` (`[0, 100]`) required for initial ordered-core membership.
    pub min_ordered_plddt: f64,
    /// Gaussian kernel width $\sigma_{\text{PAE}}$ in Å (`default = 5.0`).
    pub sigma_pae_angstrom: f64,
    /// Hard upper cutoff on $\overline{\text{PAE}}_{ij}$ in Å (`default = 12.0`).
    pub pae_cutoff_angstrom: f64,
    /// Maximum allowed intra-domain median $\overline{\text{PAE}}_{\text{intra}}(d)$ in Å (`default = 5.0`).
    pub max_intra_pae_median_angstrom: f64,
    /// Minimum globular domain size $|\mathcal{D}_d|$ in residues (`default = 20`).
    pub min_domain_size: usize,
    /// Maximum number of rigid domains $D_{\max}$ considered (`1 <= D <= max_domains`, `default = 8`).
    pub max_domains: usize,
    /// Per-residue median $\overline{\text{PAE}}$ ceiling (`default = 7.0 Å`) to its assigned domain
    /// core above which boundary/helical-linker residues are ejected into `FlexibleLinker`.
    pub boundary_outlier_pae_angstrom: f64,
    /// Whether distinct polymer `chain_id`s are prevented from merging into one covalent block.
    pub separate_distinct_chains: bool,
    /// Number of deterministic $k$-means++ restarts per candidate $D$ (`default = 16`).
    pub kmeans_restarts: usize,
    /// Deterministic PRNG seed for $k$-means++.
    pub seed: u64,
}

impl Default for PaeClusterConfig {
    fn default() -> Self {
        Self {
            min_ordered_plddt: DEFAULT_MIN_ORDERED_PLDDT,
            sigma_pae_angstrom: DEFAULT_PAE_SIGMA_ANGSTROM,
            pae_cutoff_angstrom: DEFAULT_PAE_AFFINITY_CUTOFF_ANGSTROM,
            max_intra_pae_median_angstrom: DEFAULT_MAX_INTRA_PAE_MEDIAN_ANGSTROM,
            min_domain_size: DEFAULT_MIN_DOMAIN_RESIDUES,
            max_domains: DEFAULT_MAX_PAE_DOMAINS,
            boundary_outlier_pae_angstrom: 7.0,
            separate_distinct_chains: true,
            kmeans_restarts: 16,
            seed: 0x5041_455F_444F_4D31,
        }
    }
}

/// Structural role assigned to each residue `0..L` by `PAE` spectral domain partitioning.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResidueDomainRole {
    /// Member of 1-based rigid domain `domain_id` (`1..=D`) with internal torsions frozen.
    RigidDomain { domain_id: usize },
    /// Member of 1-based flexible inter-domain torsion linker `linker_id` (`1..=L_link`).
    FlexibleLinker { linker_id: usize },
    /// Unrestrained disordered N-terminal tail preceding the first rigid domain on its chain.
    NTermTail,
    /// Unrestrained disordered C-terminal tail following the last rigid domain on its chain.
    CTermTail,
}

/// Summary descriptor for one rigid domain $\mathcal{D}_d$ (`domain_id` is 1-based `1..=D`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RigidDomainSpec {
    pub domain_id: usize,
    pub chain_id: String,
    /// Sorted 0-based residue indices belonging to this rigid domain.
    pub residue_indices: Vec<usize>,
    /// Inclusive `(start_seq_id, end_seq_id)` contiguous spans comprising this domain.
    pub seq_id_spans: Vec<(i32, i32)>,
    /// Median symmetrized $\overline{\text{PAE}}_{ij}$ (Å) across all residue pairs in $\mathcal{D}_d$.
    pub median_intra_pae_angstrom: f64,
    /// Mean per-residue `pLDDT` across $\mathcal{D}_d$ (`100.0` when `pLDDT` is unavailable).
    pub mean_plddt: f64,
}

/// Summary descriptor for one flexible covalent torsion-angle linker $\mathcal{L}_l$ (`linker_id` is 1-based).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FlexibleLinkerSpec {
    pub linker_id: usize,
    pub chain_id: String,
    pub upstream_domain_id: usize,
    pub downstream_domain_id: usize,
    /// Contiguous 0-based sequence indices in this flexible linker.
    pub residue_indices: Vec<usize>,
    pub start_seq_id: i32,
    pub end_seq_id: i32,
    pub mean_plddt: f64,
}

/// Complete `PAE` spectral domain partition across `0..L`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PaeDomainPartition {
    pub n_residues: usize,
    pub n_ordered_core: usize,
    pub domains: Vec<RigidDomainSpec>,
    pub linkers: Vec<FlexibleLinkerSpec>,
    pub residue_roles: Vec<ResidueDomainRole>,
    /// Compact integer label per residue (`0` = linker/tail, `1..=D` = 1-based `domain_id`).
    pub domain_labels: Vec<usize>,
    /// Smallest normalized-Laplacian eigenvalues $\lambda_0 \le \lambda_1 \le \dots$ (`<= 12` stored).
    pub laplacian_eigenvalues: Vec<f64>,
    /// Selected eigengap $\lambda_D - \lambda_{D-1}$ (`0-based` index $D$, i.e. $\lambda_{D+1} - \lambda_D$ in 1-based notation).
    pub selected_eigengap: f64,
}

/// Compute the Hubert & Arabie (1985) chance-corrected Adjusted Rand Index (`ARI` $\in [-1, 1]$,
/// `1.0` for identical partitions up to label permutation) between two equal-length label slices.
pub fn adjusted_rand_index(labels_a: &[usize], labels_b: &[usize]) -> Result<f64, String> {
    let n = labels_a.len();
    if labels_b.len() != n {
        return Err(format!(
            "adjusted_rand_index length mismatch: {} vs {}",
            labels_a.len(),
            labels_b.len()
        ));
    }
    if n <= 1 {
        return Ok(1.0);
    }

    let choose2 = |k: usize| -> f64 {
        if k < 2 {
            0.0
        } else {
            ((k as u128) * ((k - 1) as u128) / 2) as f64
        }
    };

    let mut counts_a: BTreeMap<usize, usize> = BTreeMap::new();
    let mut counts_b: BTreeMap<usize, usize> = BTreeMap::new();
    let mut contingency: BTreeMap<(usize, usize), usize> = BTreeMap::new();

    for (&a, &b) in labels_a.iter().zip(labels_b.iter()) {
        *counts_a.entry(a).or_insert(0) += 1;
        *counts_b.entry(b).or_insert(0) += 1;
        *contingency.entry((a, b)).or_insert(0) += 1;
    }

    let sum_nij: f64 = contingency.values().map(|&c| choose2(c)).sum();
    let sum_ai: f64 = counts_a.values().map(|&c| choose2(c)).sum();
    let sum_bj: f64 = counts_b.values().map(|&c| choose2(c)).sum();
    let total_pairs = choose2(n);
    if total_pairs <= 0.0 {
        return Ok(1.0);
    }

    let expected = (sum_ai * sum_bj) / total_pairs;
    let max_index = 0.5 * (sum_ai + sum_bj);
    let denom = max_index - expected;
    if denom.abs() <= 1e-15 {
        return Ok(if (sum_nij - expected).abs() <= 1e-12 {
            1.0
        } else {
            0.0
        });
    }
    Ok((sum_nij - expected) / denom)
}

/// Deterministic row-major threshold cyclic Jacobi symmetric eigensolver (`A = V * diag(evals) * V^T`),
/// implemented strictly with IEEE-754 `+`, `-`, `*`, `/`, and `.sqrt()` (`0` OS `libm` calls) so
/// spectral embeddings are bit-identical across macOS `aarch64`, Linux `x86_64`, and `wasm32`.
fn jacobi_symmetric_eigen_deterministic(a_in: &[f64], m: usize) -> (Vec<f64>, Vec<f64>) {
    if m == 0 {
        return (Vec::new(), Vec::new());
    }
    let mut a = a_in.to_vec();
    // Store eigenvectors row-wise in `v_t` (`v_t[k * m + i]` = i-th component of k-th eigenvector)
    // so paired Jacobi updates stream contiguously across memory.
    let mut v_t = vec![0.0_f64; m * m];
    for i in 0..m {
        v_t[i * m + i] = 1.0;
    }
    if m == 1 {
        return (vec![a[0]], v_t);
    }

    let mut diag = vec![0.0_f64; m];
    for i in 0..m {
        diag[i] = a[i * m + i];
    }

    for sweep in 0..32 {
        let mut off_sq = 0.0_f64;
        for p in 0..m {
            let row_p = &a[p * m..p * m + m];
            for &val in &row_p[(p + 1)..m] {
                off_sq += val * val;
            }
        }
        let off_norm = off_sq.sqrt();
        if off_norm <= 1e-13 * (m as f64).sqrt() {
            break;
        }
        let thresh = if sweep < 3 {
            0.20 * off_norm / ((m * m) as f64)
        } else {
            1e-15
        };

        for p in 0..(m - 1) {
            for q in (p + 1)..m {
                let apq = a[p * m + q];
                let abs_apq = apq.abs();
                if abs_apq <= thresh {
                    continue;
                }
                let app = diag[p];
                let aqq = diag[q];
                if abs_apq <= 1e-15 * (app.abs() + aqq.abs() + 1e-30) {
                    a[p * m + q] = 0.0;
                    a[q * m + p] = 0.0;
                    continue;
                }

                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                let h = t * apq;

                diag[p] = app - h;
                diag[q] = aqq + h;
                a[p * m + p] = diag[p];
                a[q * m + q] = diag[q];
                a[p * m + q] = 0.0;
                a[q * m + p] = 0.0;

                // Update upper-triangular off-diagonal elements involving `p` and `q`
                for r in 0..p {
                    let arp = a[r * m + p];
                    let arq = a[r * m + q];
                    a[r * m + p] = c * arp - s * arq;
                    a[r * m + q] = s * arp + c * arq;
                }
                for r in (p + 1)..q {
                    let apr = a[p * m + r];
                    let arq = a[r * m + q];
                    a[p * m + r] = c * apr - s * arq;
                    a[r * m + q] = s * apr + c * arq;
                }
                {
                    let (left, right) = a.split_at_mut(q * m);
                    let row_p = &mut left[p * m + q + 1..p * m + m];
                    let row_q = &mut right[q + 1..m];
                    for (apr, aqr) in row_p.iter_mut().zip(row_q.iter_mut()) {
                        let up = *apr;
                        let uq = *aqr;
                        *apr = c * up - s * uq;
                        *aqr = s * up + c * uq;
                    }
                }

                // Update contiguous eigenvector rows `p` and `q`
                let (v_left, v_right) = v_t.split_at_mut(q * m);
                let vp = &mut v_left[p * m..p * m + m];
                let vq = &mut v_right[0..m];
                for (x, y) in vp.iter_mut().zip(vq.iter_mut()) {
                    let xp = *x;
                    let yq = *y;
                    *x = c * xp - s * yq;
                    *y = s * xp + c * yq;
                }
            }
        }
    }

    // Sort eigenvalues ascending with deterministic index tie-breaker and canonicalize column signs
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&i, &j| diag[i].total_cmp(&diag[j]).then_with(|| i.cmp(&j)));

    let mut evals_sorted = Vec::with_capacity(m);
    let mut v_rows_sorted = vec![0.0_f64; m * m];
    for (new_k, &old_k) in order.iter().enumerate() {
        evals_sorted.push(diag[old_k].max(0.0));
        let src = &v_t[old_k * m..old_k * m + m];
        let mut best_idx = 0usize;
        let mut best_abs = -1.0_f64;
        for (idx, &val) in src.iter().enumerate() {
            let av = val.abs();
            if av > best_abs + 1e-15 {
                best_abs = av;
                best_idx = idx;
            }
        }
        let sign = if src[best_idx] < 0.0 { -1.0 } else { 1.0 };
        let dst = &mut v_rows_sorted[new_k * m..new_k * m + m];
        for (d, &s) in dst.iter_mut().zip(src.iter()) {
            *d = sign * s;
        }
    }

    (evals_sorted, v_rows_sorted)
}

fn exact_median_f64(vals: &mut [f64]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    vals.sort_by(|a, b| a.total_cmp(b));
    let mid = vals.len() / 2;
    if vals.len() % 2 == 1 {
        vals[mid]
    } else {
        0.5 * (vals[mid - 1] + vals[mid])
    }
}

/// Deterministic $k$-means++ clustering on row-normalized $M \times D$ spectral coordinates `y_flat`.
fn deterministic_kmeans_pp(
    y_flat: &[f64],
    m: usize,
    d: usize,
    restarts: usize,
    seed: u64,
) -> Vec<usize> {
    if d <= 1 || m == 0 {
        return vec![0usize; m];
    }
    let mut rng = DetRng::new(seed ^ ((d as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)));
    let mut best_labels = vec![0usize; m];
    let mut best_inertia = f64::INFINITY;
    let n_runs = restarts.max(1);

    let mut centroids = vec![0.0_f64; d * d];
    let mut min_d2 = vec![f64::INFINITY; m];
    let mut labels = vec![0usize; m];
    let mut counts = vec![0usize; d];
    let mut sums = vec![0.0_f64; d * d];

    for _run in 0..n_runs {
        // 1. k-means++ seeding
        let first = rng.range_usize(0, m);
        centroids[0..d].copy_from_slice(&y_flat[first * d..(first + 1) * d]);
        min_d2.fill(f64::INFINITY);

        for c_idx in 1..d {
            let prev_c = &centroids[(c_idx - 1) * d..c_idx * d];
            let mut total_w = 0.0_f64;
            for u in 0..m {
                let yu = &y_flat[u * d..(u + 1) * d];
                let mut dist2 = 0.0_f64;
                for k in 0..d {
                    let df = yu[k] - prev_c[k];
                    dist2 += df * df;
                }
                if dist2 < min_d2[u] {
                    min_d2[u] = dist2;
                }
                total_w += min_d2[u];
            }
            let chosen = if total_w <= 1e-30 {
                rng.range_usize(0, m)
            } else {
                let thresh = rng.f64() * total_w;
                let mut acc = 0.0_f64;
                let mut pick = m - 1;
                for (u, &w) in min_d2.iter().enumerate() {
                    acc += w;
                    if acc >= thresh {
                        pick = u;
                        break;
                    }
                }
                pick
            };
            centroids[c_idx * d..(c_idx + 1) * d]
                .copy_from_slice(&y_flat[chosen * d..(chosen + 1) * d]);
        }

        // 2. Lloyd iterations
        for _iter in 0..60 {
            let mut changed = false;
            counts.fill(0);
            sums.fill(0.0);

            for u in 0..m {
                let yu = &y_flat[u * d..(u + 1) * d];
                let mut best_c = 0usize;
                let mut best_c_d2 = f64::INFINITY;
                for c_idx in 0..d {
                    let cu = &centroids[c_idx * d..(c_idx + 1) * d];
                    let mut dist2 = 0.0_f64;
                    for k in 0..d {
                        let df = yu[k] - cu[k];
                        dist2 += df * df;
                    }
                    if dist2 + 1e-16 < best_c_d2 {
                        best_c_d2 = dist2;
                        best_c = c_idx;
                    }
                }
                if labels[u] != best_c {
                    labels[u] = best_c;
                    changed = true;
                }
                counts[best_c] += 1;
                let dst = &mut sums[best_c * d..(best_c + 1) * d];
                for k in 0..d {
                    dst[k] += yu[k];
                }
            }

            // Recompute centroids (re-seeding empty cluster to farthest point if any)
            for c_idx in 0..d {
                if counts[c_idx] > 0 {
                    let inv = 1.0 / (counts[c_idx] as f64);
                    let src = &sums[c_idx * d..(c_idx + 1) * d];
                    let dst = &mut centroids[c_idx * d..(c_idx + 1) * d];
                    for k in 0..d {
                        dst[k] = src[k] * inv;
                    }
                } else {
                    let mut far_u = 0usize;
                    let mut far_d2 = -1.0_f64;
                    for u in 0..m {
                        let yu = &y_flat[u * d..(u + 1) * d];
                        let cl = labels[u];
                        let cu = &centroids[cl * d..(cl + 1) * d];
                        let mut dist2 = 0.0_f64;
                        for k in 0..d {
                            let df = yu[k] - cu[k];
                            dist2 += df * df;
                        }
                        if dist2 > far_d2 {
                            far_d2 = dist2;
                            far_u = u;
                        }
                    }
                    centroids[c_idx * d..(c_idx + 1) * d]
                        .copy_from_slice(&y_flat[far_u * d..(far_u + 1) * d]);
                    changed = true;
                }
            }

            if !changed {
                break;
            }
        }

        let mut inertia = 0.0_f64;
        for u in 0..m {
            let yu = &y_flat[u * d..(u + 1) * d];
            let cu = &centroids[labels[u] * d..(labels[u] + 1) * d];
            for k in 0..d {
                let df = yu[k] - cu[k];
                inertia += df * df;
            }
        }
        if inertia + 1e-14 < best_inertia {
            best_inertia = inertia;
            best_labels.copy_from_slice(&labels);
        }
    }

    // Canonicalize cluster indices `0..d` by ascending minimum ordered index `u`
    let mut min_pos = vec![usize::MAX; d];
    for (u, &cl) in best_labels.iter().enumerate() {
        if u < min_pos[cl] {
            min_pos[cl] = u;
        }
    }
    let mut order: Vec<usize> = (0..d).collect();
    order.sort_by_key(|&cl| (min_pos[cl], cl));
    let mut remap = vec![0usize; d];
    for (new_c, &old_c) in order.iter().enumerate() {
        remap[old_c] = new_c;
    }
    for cl in &mut best_labels {
        *cl = remap[*cl];
    }
    best_labels
}

fn build_seq_id_spans(indices: &[usize], seq_ids: &[i32]) -> Vec<(i32, i32)> {
    if indices.is_empty() {
        return Vec::new();
    }
    let mut spans = Vec::new();
    let mut cur_start = seq_ids[indices[0]];
    let mut prev_idx = indices[0];
    for &idx in &indices[1..] {
        if idx == prev_idx + 1 {
            prev_idx = idx;
        } else {
            spans.push((cur_start, seq_ids[prev_idx]));
            cur_start = seq_ids[idx];
            prev_idx = idx;
        }
    }
    spans.push((cur_start, seq_ids[prev_idx]));
    spans
}

fn domain_intra_pae_median(pae: &PaeMatrix, indices: &[usize]) -> f64 {
    if indices.len() <= 1 {
        return 0.0;
    }
    let mut vals = Vec::with_capacity(indices.len() * (indices.len() - 1) / 2);
    for (a_pos, &i) in indices.iter().enumerate() {
        for &j in &indices[(a_pos + 1)..] {
            if let Some(p) = pae.symmetrized(i, j) {
                vals.push(p);
            }
        }
    }
    exact_median_f64(&mut vals)
}

fn domain_inter_pae_median(pae: &PaeMatrix, a_indices: &[usize], b_indices: &[usize]) -> f64 {
    if a_indices.is_empty() || b_indices.is_empty() {
        return 0.0;
    }
    let mut vals = Vec::with_capacity(a_indices.len() * b_indices.len());
    for &i in a_indices {
        for &j in b_indices {
            if i != j {
                if let Some(p) = pae.symmetrized(i, j) {
                    vals.push(p);
                }
            }
        }
    }
    exact_median_f64(&mut vals)
}

/// Partition an `AfStructureModel` + `PaeMatrix` into confident rigid domains and flexible linkers.
pub fn cluster_pae_domains_for_model(
    model: &AfStructureModel,
    pae: &PaeMatrix,
    cfg: &PaeClusterConfig,
) -> Result<PaeDomainPartition, String> {
    let l = model.residues.len();
    pae.validate_chain_length(l)?;
    let seq_ids: Vec<i32> = model.residues.iter().map(|r| r.seq_id).collect();
    let chain_ids: Vec<String> = model.residues.iter().map(|r| r.chain_id.clone()).collect();
    let plddts: Option<Vec<f64>> = model
        .plddt_per_residue
        .clone()
        .or_else(|| pae.plddt_from_json.clone());
    cluster_pae_domains(
        pae,
        plddts.as_deref(),
        Some(&seq_ids),
        Some(&chain_ids),
        cfg,
    )
}

/// Partition an $N \times N$ `PaeMatrix` (with optional `pLDDT`, `seq_ids`, and `chain_ids`) into
/// globular rigid domains (`1..=D`) and flexible torsion linkers via Ng–Jordan–Weiss (2001)
/// normalized symmetric Laplacian spectral clustering.
pub fn cluster_pae_domains(
    pae: &PaeMatrix,
    plddt_per_residue: Option<&[f64]>,
    seq_ids_opt: Option<&[i32]>,
    chain_ids_opt: Option<&[String]>,
    cfg: &PaeClusterConfig,
) -> Result<PaeDomainPartition, String> {
    let n = pae.n_residues;
    if n == 0 {
        return Err("cannot cluster an empty PAE matrix (n_residues = 0)".to_string());
    }
    if pae.values_angstrom.len() != n * n {
        return Err(format!(
            "PAE flat buffer length {} != n_residues^2 ({})",
            pae.values_angstrom.len(),
            n * n
        ));
    }
    if pae
        .values_angstrom
        .iter()
        .any(|v| !v.is_finite() || *v < 0.0)
    {
        return Err("PAE matrix contains negative or non-finite entries".to_string());
    }
    if !cfg.sigma_pae_angstrom.is_finite() || cfg.sigma_pae_angstrom <= 0.0 {
        return Err(format!(
            "invalid sigma_pae_angstrom ({}); must be finite and > 0",
            cfg.sigma_pae_angstrom
        ));
    }
    if !cfg.pae_cutoff_angstrom.is_finite() || cfg.pae_cutoff_angstrom <= 0.0 {
        return Err(format!(
            "invalid pae_cutoff_angstrom ({}); must be finite and > 0",
            cfg.pae_cutoff_angstrom
        ));
    }
    if cfg.max_domains == 0 {
        return Err("PaeClusterConfig::max_domains must be >= 1".to_string());
    }

    let eff_plddt: Vec<f64> = if let Some(pl) = plddt_per_residue {
        if pl.len() != n {
            return Err(format!(
                "plddt_per_residue length {} != PAE n_residues {n}",
                pl.len()
            ));
        }
        for (i, &v) in pl.iter().enumerate() {
            if !v.is_finite() || !(0.0..=100.0).contains(&v) {
                return Err(format!(
                    "invalid pLDDT ({v}) at residue index {i}; expected finite [0, 100]"
                ));
            }
        }
        pl.to_vec()
    } else if let Some(ref pl_json) = pae.plddt_from_json {
        if pl_json.len() == n {
            pl_json.clone()
        } else {
            vec![100.0; n]
        }
    } else {
        vec![100.0; n]
    };

    let seq_ids: Vec<i32> = match seq_ids_opt {
        Some(s) => {
            if s.len() != n {
                return Err(format!("seq_ids length {} != PAE n_residues {n}", s.len()));
            }
            s.to_vec()
        }
        None => (1..=(n as i32)).collect(),
    };
    let chain_ids: Vec<String> = match chain_ids_opt {
        Some(c) => {
            if c.len() != n {
                return Err(format!(
                    "chain_ids length {} != PAE n_residues {n}",
                    c.len()
                ));
            }
            c.to_vec()
        }
        None => vec![pae.chain_id.clone().unwrap_or_else(|| "A".to_string()); n],
    };

    let raw_ordered: Vec<usize> = (0..n)
        .filter(|&i| eff_plddt[i] >= cfg.min_ordered_plddt)
        .collect();
    let raw_m = raw_ordered.len();

    // Degenerate disordered chain (< min_domain_size ordered residues): return 0 rigid domains cleanly
    let eff_min_dom = cfg.min_domain_size.max(1);
    if raw_m < eff_min_dom {
        return Ok(PaeDomainPartition {
            n_residues: n,
            n_ordered_core: raw_m,
            domains: Vec::new(),
            linkers: Vec::new(),
            residue_roles: vec![ResidueDomainRole::NTermTail; n],
            domain_labels: vec![0; n],
            laplacian_eigenvalues: Vec::new(),
            selected_eigengap: 0.0,
        });
    }

    // Filter out isolated ordered micro-components (`< eff_min_dom`, e.g. short ordered helical turns
    // inside flexible linkers or terminal tails that have `PAE > pae_cutoff_angstrom` to every domain)
    // so they cannot inject spurious zero eigenvalues into `L_sym`.
    let cutoff = cfg.pae_cutoff_angstrom;
    let mut comp_id = vec![usize::MAX; raw_m];
    let mut comp_members: Vec<Vec<usize>> = Vec::new();
    let mut bfs_q: Vec<usize> = Vec::new();
    for start_u in 0..raw_m {
        if comp_id[start_u] != usize::MAX {
            continue;
        }
        let cid = comp_members.len();
        comp_id[start_u] = cid;
        bfs_q.clear();
        bfs_q.push(start_u);
        let mut head = 0usize;
        while head < bfs_q.len() {
            let u = bfs_q[head];
            head += 1;
            let i = raw_ordered[u];
            for v in 0..raw_m {
                if comp_id[v] != usize::MAX {
                    continue;
                }
                let j = raw_ordered[v];
                if cfg.separate_distinct_chains && chain_ids[i] != chain_ids[j] {
                    continue;
                }
                if pae.symmetrized(i, j).unwrap_or(f64::INFINITY) <= cutoff {
                    comp_id[v] = cid;
                    bfs_q.push(v);
                }
            }
        }
        comp_members.push(bfs_q.clone());
    }
    let mut keep_raw = vec![false; raw_m];
    for members in &comp_members {
        if members.len() >= eff_min_dom {
            for &u in members {
                keep_raw[u] = true;
            }
        }
    }
    let ordered: Vec<usize> = raw_ordered
        .iter()
        .enumerate()
        .filter_map(|(u, &idx)| keep_raw[u].then_some(idx))
        .collect();
    let m = ordered.len();
    if m < eff_min_dom {
        return Ok(PaeDomainPartition {
            n_residues: n,
            n_ordered_core: m,
            domains: Vec::new(),
            linkers: Vec::new(),
            residue_roles: vec![ResidueDomainRole::NTermTail; n],
            domain_labels: vec![0; n],
            laplacian_eigenvalues: Vec::new(),
            selected_eigengap: 0.0,
        });
    }

    // 1. Build symmetrized Gaussian PAE affinity matrix A on ordered core O (`m x m`)
    let inv_two_sig2 = 1.0 / (2.0 * cfg.sigma_pae_angstrom * cfg.sigma_pae_angstrom);
    let mut affinity = vec![0.0_f64; m * m];
    let mut deg = vec![0.0_f64; m];

    for u in 0..m {
        let i = ordered[u];
        affinity[u * m + u] = 1.0;
        deg[u] += 1.0;
        for v in (u + 1)..m {
            let j = ordered[v];
            if cfg.separate_distinct_chains && chain_ids[i] != chain_ids[j] {
                continue;
            }
            let p_sym = pae.symmetrized(i, j).unwrap_or(f64::INFINITY);
            if p_sym <= cutoff {
                let w = det_exp(-(p_sym * p_sym) * inv_two_sig2);
                affinity[u * m + v] = w;
                affinity[v * m + u] = w;
                deg[u] += w;
                deg[v] += w;
            }
        }
    }

    // 2. Normalized symmetric Laplacian L_sym = I - D^(-1/2) A D^(-1/2)
    let inv_sqrt_deg: Vec<f64> = deg.iter().map(|&d| 1.0 / d.max(1e-30).sqrt()).collect();
    let mut l_sym = vec![0.0_f64; m * m];
    for u in 0..m {
        let du = inv_sqrt_deg[u];
        l_sym[u * m + u] = (1.0 - affinity[u * m + u] * du * du).max(0.0);
        for v in (u + 1)..m {
            let val = -affinity[u * m + v] * du * inv_sqrt_deg[v];
            l_sym[u * m + v] = val;
            l_sym[v * m + u] = val;
        }
    }

    let (evals, v_rows) = jacobi_symmetric_eigen_deterministic(&l_sym, m);
    let max_d_try = cfg.max_domains.min(m / eff_min_dom).max(1).min(m);

    // 3. Evaluate candidate D in 1..=max_d_try using Ng-Jordan-Weiss row-normalized embeddings
    struct CandPartition {
        d: usize,
        eigengap: f64,
        all_valid: bool,
        all_sized: bool,
        ordered_labels: Vec<usize>,
    }

    let mut candidates: Vec<CandPartition> = Vec::with_capacity(max_d_try);
    for d in 1..=max_d_try {
        let eigengap = if d < m {
            (evals[d] - evals[d - 1]).max(0.0)
        } else {
            0.0
        };
        let ordered_labels = if d == 1 {
            vec![0usize; m]
        } else {
            let mut y_flat = vec![0.0_f64; m * d];
            for u in 0..m {
                let mut norm2 = 0.0_f64;
                for k in 0..d {
                    let v_uk = v_rows[k * m + u];
                    y_flat[u * d + k] = v_uk;
                    norm2 += v_uk * v_uk;
                }
                if norm2 > 1e-30 {
                    let inv_n = 1.0 / norm2.sqrt();
                    for k in 0..d {
                        y_flat[u * d + k] *= inv_n;
                    }
                } else {
                    y_flat[u * d] = 1.0;
                }
            }
            deterministic_kmeans_pp(&y_flat, m, d, cfg.kmeans_restarts, cfg.seed)
        };

        let mut cluster_res: Vec<Vec<usize>> = vec![Vec::new(); d];
        for (u, &cl) in ordered_labels.iter().enumerate() {
            cluster_res[cl].push(ordered[u]);
        }
        // If separate_distinct_chains is active, reject a partition that merges distinct chains into one cluster
        let chain_clean = !cfg.separate_distinct_chains
            || cluster_res.iter().all(|idxs| {
                idxs.first()
                    .is_none_or(|&i0| idxs.iter().all(|&ix| chain_ids[ix] == chain_ids[i0]))
            });
        let all_sized = chain_clean && cluster_res.iter().all(|idxs| idxs.len() >= eff_min_dom);
        let mut max_med = 0.0_f64;
        for idxs in &cluster_res {
            let med = domain_intra_pae_median(pae, idxs);
            if med > max_med {
                max_med = med;
            }
        }
        // Reject artificial cuts of a single continuous globular domain (`P2-12`): any two clusters
        // on the same polymer chain must exhibit genuinely decoupled cross-domain PAE
        // (`domain_inter_pae_median > cfg.pae_cutoff_angstrom`).
        let mut inter_decoupled = true;
        if all_sized && d >= 2 {
            for ca in 0..d {
                for cb in (ca + 1)..d {
                    let same_chain_pair = match (cluster_res[ca].first(), cluster_res[cb].first()) {
                        (Some(&ia), Some(&ib)) => chain_ids[ia] == chain_ids[ib],
                        _ => false,
                    };
                    if same_chain_pair {
                        let inter_med =
                            domain_inter_pae_median(pae, &cluster_res[ca], &cluster_res[cb]);
                        if inter_med <= cfg.pae_cutoff_angstrom {
                            inter_decoupled = false;
                            break;
                        }
                    }
                }
                if !inter_decoupled {
                    break;
                }
            }
        }
        let all_valid =
            all_sized && inter_decoupled && max_med <= cfg.max_intra_pae_median_angstrom;
        candidates.push(CandPartition {
            d,
            eigengap,
            all_valid,
            all_sized: all_sized && inter_decoupled,
            ordered_labels,
        });
    }

    // Never fall back to `min_by(max_median_pae)` (which artificially bisects an elongated single
    // domain `75/75` or returns clusters violating `max_intra_pae_median_angstrom`). If no `all_valid`
    // partition exists directly, start from the minimal decoupled per-chain domain count (`d = 1` on
    // a single chain) and strictly enforce `median_intra_pae <= max_intra_pae_median_angstrom` below.
    let chosen_idx = if let Some((best_i, _)) = candidates
        .iter()
        .enumerate()
        .filter(|(_, c)| c.all_valid)
        .max_by(|(_, a), (_, b)| a.eigengap.total_cmp(&b.eigengap).then(b.d.cmp(&a.d)))
    {
        best_i
    } else if let Some((min_decoupled_i, _)) =
        candidates.iter().enumerate().find(|(_, c)| c.all_sized)
    {
        min_decoupled_i
    } else {
        0
    };

    let chosen = &candidates[chosen_idx];
    let d_final = chosen.d;
    let selected_eigengap = chosen.eigengap;

    // 4. Map ordered labels onto full sequence `0..n` (1-based domain_id, 0 = unassigned/linker/tail)
    let mut raw_dom = vec![0usize; n];
    for (u, &cl) in chosen.ordered_labels.iter().enumerate() {
        raw_dom[ordered[u]] = cl + 1;
    }

    // Eject high-PAE boundary/linker residues (e.g. ordered helical linkers with PAE > boundary_outlier_pae_angstrom
    // to their assigned domain core) while preserving `>= eff_min_dom` core size.
    for dom_id in 1..=d_final {
        let members: Vec<usize> = (0..n).filter(|&i| raw_dom[i] == dom_id).collect();
        if members.len() <= eff_min_dom {
            continue;
        }
        let mut scored: Vec<(f64, usize)> = members
            .iter()
            .map(|&i| {
                let mut row_vals: Vec<f64> = members
                    .iter()
                    .filter(|&&j| j != i)
                    .filter_map(|&j| pae.symmetrized(i, j))
                    .collect();
                (exact_median_f64(&mut row_vals), i)
            })
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut cur_len = members.len();
        for (med_to_dom, idx) in scored {
            if cur_len <= eff_min_dom || med_to_dom <= cfg.boundary_outlier_pae_angstrom {
                break;
            }
            raw_dom[idx] = 0;
            cur_len -= 1;
        }
    }

    // Absorb short (`<= 4` residue) intra-domain surface turns (`raw_dom == 0`) flanked on BOTH
    // sides by the SAME domain on the SAME chain when their median PAE to that domain is low,
    // keeping globular domains contiguous without splitting rigid bodies at surface loops.
    let mut scan = 0usize;
    while scan < n {
        if raw_dom[scan] != 0 {
            scan += 1;
            continue;
        }
        let mut end = scan;
        while end < n && raw_dom[end] == 0 && chain_ids[end] == chain_ids[scan] {
            end += 1;
        }
        let gap_len = end - scan;
        if gap_len <= 4 && scan > 0 && end < n {
            let left_d = raw_dom[scan - 1];
            let right_d = raw_dom[end];
            if left_d != 0
                && left_d == right_d
                && chain_ids[scan - 1] == chain_ids[scan]
                && chain_ids[end] == chain_ids[scan]
            {
                let dom_members: Vec<usize> = (0..n).filter(|&k| raw_dom[k] == left_d).collect();
                let mut ok_turn = true;
                for g in scan..end {
                    let mut pvals: Vec<f64> = dom_members
                        .iter()
                        .filter_map(|&k| pae.symmetrized(g, k))
                        .collect();
                    if exact_median_f64(&mut pvals) > cfg.boundary_outlier_pae_angstrom {
                        ok_turn = false;
                        break;
                    }
                }
                if ok_turn {
                    for slot in &mut raw_dom[scan..end] {
                        *slot = left_d;
                    }
                }
            }
        }
        scan = end;
    }

    // Strict post-invariant (`P2-12`): every emitted rigid domain MUST satisfy BOTH
    // `|D_d| >= eff_min_dom` AND `domain_intra_pae_median <= cfg.max_intra_pae_median_angstrom`.
    // Gently trim peripheral high-PAE edge residues if needed (e.g. flexible ends of a sloped single
    // domain), or dissolve any molten cluster that has no confident `<= max_intra_pae_median_angstrom` core.
    for dom_id in 1..=d_final {
        loop {
            let members: Vec<usize> = (0..n).filter(|&i| raw_dom[i] == dom_id).collect();
            if members.is_empty() {
                break;
            }
            if members.len() < eff_min_dom {
                for i in members {
                    raw_dom[i] = 0;
                }
                break;
            }
            let cur_med = domain_intra_pae_median(pae, &members);
            if cur_med <= cfg.max_intra_pae_median_angstrom {
                break;
            }
            if members.len() == eff_min_dom {
                for i in members {
                    raw_dom[i] = 0;
                }
                break;
            }
            // Trim the single member with highest median symmetrized PAE to the rest of the cluster
            let mut worst_idx = members[0];
            let mut worst_med = -1.0_f64;
            for &i in &members {
                let mut rvals: Vec<f64> = members
                    .iter()
                    .filter(|&&j| j != i)
                    .filter_map(|&j| pae.symmetrized(i, j))
                    .collect();
                let m_i = exact_median_f64(&mut rvals);
                if m_i > worst_med {
                    worst_med = m_i;
                    worst_idx = i;
                }
            }
            raw_dom[worst_idx] = 0;
        }
    }

    // Re-compact domain IDs 1..=D_kept in N-to-C order
    let mut active_old_ids: Vec<(usize, usize)> = Vec::new();
    for old_d in 1..=d_final {
        if let Some(first_i) = (0..n).find(|&i| raw_dom[i] == old_d) {
            active_old_ids.push((first_i, old_d));
        }
    }
    active_old_ids.sort_unstable();
    let mut old_to_new = vec![0usize; d_final + 1];
    for (new_idx, &(_, old_d)) in active_old_ids.iter().enumerate() {
        old_to_new[old_d] = new_idx + 1;
    }
    for d_slot in &mut raw_dom {
        if *d_slot > 0 {
            *d_slot = old_to_new[*d_slot];
        }
    }
    let d_kept = active_old_ids.len();

    // 5. Classify non-domain spans per chain into NTermTail, CTermTail, and FlexibleLinker
    let mut roles = vec![ResidueDomainRole::NTermTail; n];
    let mut linkers: Vec<FlexibleLinkerSpec> = Vec::new();

    let mut ch_start = 0usize;
    while ch_start < n {
        let mut ch_end = ch_start + 1;
        while ch_end < n && chain_ids[ch_end] == chain_ids[ch_start] {
            ch_end += 1;
        }
        let first_dom_pos = (ch_start..ch_end).find(|&i| raw_dom[i] > 0);
        let last_dom_pos = (ch_start..ch_end).rev().find(|&i| raw_dom[i] > 0);

        match (first_dom_pos, last_dom_pos) {
            (Some(f_pos), Some(l_pos)) => {
                for r_slot in &mut roles[ch_start..f_pos] {
                    *r_slot = ResidueDomainRole::NTermTail;
                }
                for r_slot in &mut roles[(l_pos + 1)..ch_end] {
                    *r_slot = ResidueDomainRole::CTermTail;
                }
                let mut p = f_pos;
                while p <= l_pos {
                    if raw_dom[p] > 0 {
                        roles[p] = ResidueDomainRole::RigidDomain {
                            domain_id: raw_dom[p],
                        };
                        p += 1;
                    } else {
                        let lk_start = p;
                        while p <= l_pos && raw_dom[p] == 0 {
                            p += 1;
                        }
                        let lk_end = p;
                        let up_dom = raw_dom[lk_start - 1];
                        let dn_dom = raw_dom[lk_end];
                        let linker_id = linkers.len() + 1;
                        for r_slot in &mut roles[lk_start..lk_end] {
                            *r_slot = ResidueDomainRole::FlexibleLinker { linker_id };
                        }
                        let res_idxs: Vec<usize> = (lk_start..lk_end).collect();
                        let mean_pl = res_idxs.iter().map(|&i| eff_plddt[i]).sum::<f64>()
                            / (res_idxs.len() as f64);
                        linkers.push(FlexibleLinkerSpec {
                            linker_id,
                            chain_id: chain_ids[ch_start].clone(),
                            upstream_domain_id: up_dom,
                            downstream_domain_id: dn_dom,
                            residue_indices: res_idxs,
                            start_seq_id: seq_ids[lk_start],
                            end_seq_id: seq_ids[lk_end - 1],
                            mean_plddt: mean_pl,
                        });
                    }
                }
            }
            _ => {
                for r_slot in &mut roles[ch_start..ch_end] {
                    *r_slot = ResidueDomainRole::NTermTail;
                }
            }
        }
        ch_start = ch_end;
    }

    let mut domains = Vec::with_capacity(d_kept);
    for dom_id in 1..=d_kept {
        let idxs: Vec<usize> = (0..n).filter(|&i| raw_dom[i] == dom_id).collect();
        let med_pae = domain_intra_pae_median(pae, &idxs);
        let mean_pl = idxs.iter().map(|&i| eff_plddt[i]).sum::<f64>() / (idxs.len().max(1) as f64);
        let ch = idxs
            .first()
            .map(|&i| chain_ids[i].clone())
            .unwrap_or_else(|| "A".to_string());
        let spans = build_seq_id_spans(&idxs, &seq_ids);
        domains.push(RigidDomainSpec {
            domain_id: dom_id,
            chain_id: ch,
            residue_indices: idxs,
            seq_id_spans: spans,
            median_intra_pae_angstrom: med_pae,
            mean_plddt: mean_pl,
        });
    }

    Ok(PaeDomainPartition {
        n_residues: n,
        n_ordered_core: m,
        domains,
        linkers,
        residue_roles: roles,
        domain_labels: raw_dom,
        laplacian_eigenvalues: evals.into_iter().take(12).collect(),
        selected_eigengap,
    })
}

// ============================================================================
// 2. Tolman et al. (2001) Per-Domain Generalized Degree of Order Motion Test
//    & Domain-Stratified `ByPeptidePlane` `RdcFree` Partitioning
// ============================================================================

/// Inter-domain orientational coupling verdict from Tolman $\vartheta_d / \vartheta_1$ GDO analysis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DomainMotionRegime {
    /// Domains tumble with compatible Generalized Degrees of Order ($\min_d \rho_{\text{GDO}} \ge 0.80$
    /// or within $2\sigma_{\text{MC}}$ of $1.0$).
    RigidlyCoupledDomains,
    /// Significant inter-domain orientational averaging detected ($\min_d \rho_{\text{GDO}} < 0.80$
    /// at $> 2\sigma_{\text{MC}}$); single-conformer shared-tensor interpretation requires caution.
    DecoupledInterDomainMotion,
    /// Fewer than 2 domains met the $n_{\text{work}}(d, m) \ge 8$, $\kappa \le 10^8$ SVD criterion.
    InsufficientDomainRdcs,
}

/// Fitted Saupe tensor and Tolman Generalized Degree of Order ($\vartheta_{d,m}$) for one `(domain, medium)`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DomainMediumGdoFit {
    pub domain_id: usize,
    pub medium_id: String,
    pub n_work: usize,
    pub n_free: usize,
    /// Generalized Degree of Order $\vartheta_{d,m} = \sqrt{\frac{2}{3}\sum_{\alpha,\beta} (S^{(d,m)}_{\alpha\beta})^2}$.
    pub gdo_theta: f64,
    /// Parametric Monte Carlo standard error $\sigma_{\text{MC}}(\vartheta_{d,m})$.
    pub gdo_sigma_mc: f64,
    /// Ratio $\vartheta_{d,m} / \vartheta_{1,m}$ (`1.0` for Domain 1 when fitted, else `NaN`).
    pub ratio_to_domain_1: f64,
    /// Normalized order ratio $\rho_{\text{GDO}}(d, m) = \vartheta_{d,m} / \max_{d'} \vartheta_{d',m} \in (0, 1]$.
    pub rho_gdo: f64,
    /// Parametric Monte Carlo uncertainty $\sigma_{\text{MC}}(\rho_{\text{GDO}}(d, m))$.
    pub rho_sigma_mc: f64,
    pub da_nh_hz: f64,
    pub rhombicity: f64,
    pub condition_number: f64,
    pub q_work: f64,
    pub q_free: f64,
    pub saupe_fit: SaupeTensorFit,
}

/// Complete Tolman et al. (2001) inter-domain motion diagnostic across all rigid domains and media.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TolmanGdoMotionReport {
    pub regime: DomainMotionRegime,
    pub domain_fits: Vec<DomainMediumGdoFit>,
    pub skipped_domains: Vec<(usize, String, String)>,
    pub min_rho_gdo: f64,
    pub max_abs_delta_over_theta1: f64,
    pub warning_message: Option<String>,
}

/// Compute Tolman's Generalized Degree of Order $\vartheta = \sqrt{\frac{2}{3}\sum_{\alpha,\beta} S_{\alpha\beta}^2}$.
#[inline]
pub fn compute_saupe_gdo(s_matrix: &[[f64; 3]; 3]) -> f64 {
    let mut sum_sq = 0.0_f64;
    for row in s_matrix {
        for &val in row {
            sum_sq += val * val;
        }
    }
    ((2.0 / 3.0) * sum_sq).sqrt()
}

/// Chain-explicit residual dipolar coupling restraint (`(chain_id, seq_id)` keyed for multi-chain
/// homodimers/heteromers with overlapping residue numbers, `21.W2:P1-8`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DomainRdcRestraint {
    pub medium_id: String,
    #[serde(default)]
    pub chain_id_1: Option<String>,
    pub seq_id_1: i32,
    pub res_name_1: String,
    pub atom_name_1: String,
    #[serde(default)]
    pub chain_id_2: Option<String>,
    pub seq_id_2: i32,
    pub res_name_2: String,
    pub atom_name_2: String,
    pub bond_type: RdcBondType,
    pub d_obs_hz: f64,
    pub sigma_hz: f64,
    pub d_max_hz: f64,
    pub is_free: bool,
}

impl DomainRdcRestraint {
    pub fn from_rdc_restraint(r: &RdcRestraint) -> Self {
        Self {
            medium_id: r.medium_id.clone(),
            chain_id_1: None,
            seq_id_1: r.seq_id_1,
            res_name_1: r.res_name_1.clone(),
            atom_name_1: r.atom_name_1.clone(),
            chain_id_2: None,
            seq_id_2: r.seq_id_2,
            res_name_2: r.res_name_2.clone(),
            atom_name_2: r.atom_name_2.clone(),
            bond_type: r.bond_type,
            d_obs_hz: r.d_obs_hz,
            sigma_hz: r.sigma_hz,
            d_max_hz: r.d_max_hz,
            is_free: r.is_free,
        }
    }

    pub fn to_rdc_restraint(&self) -> RdcRestraint {
        RdcRestraint {
            medium_id: self.medium_id.clone(),
            seq_id_1: self.seq_id_1,
            res_name_1: self.res_name_1.clone(),
            atom_name_1: self.atom_name_1.clone(),
            seq_id_2: self.seq_id_2,
            res_name_2: self.res_name_2.clone(),
            atom_name_2: self.atom_name_2.clone(),
            bond_type: self.bond_type,
            d_obs_hz: self.d_obs_hz,
            sigma_hz: self.sigma_hz,
            d_max_hz: self.d_max_hz,
            is_free: self.is_free,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        self.to_rdc_restraint().validate()
    }

    #[inline]
    pub fn peptide_plane_seq_id(&self) -> i32 {
        rdc_peptide_plane_id(&self.to_rdc_restraint())
    }
}

/// Alignment medium of `(chain_id, seq_id)`-keyed RDC restraints.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DomainRdcMediumSet {
    pub medium_id: String,
    pub restraints: Vec<DomainRdcRestraint>,
}

impl DomainRdcMediumSet {
    pub fn from_rdc_media(media: &[RdcMediumSet]) -> Vec<Self> {
        media
            .iter()
            .map(|m| Self {
                medium_id: m.medium_id.clone(),
                restraints: m
                    .restraints
                    .iter()
                    .map(DomainRdcRestraint::from_rdc_restraint)
                    .collect(),
            })
            .collect()
    }
}

/// Multi-chain residue lookup index keyed strictly on `(chain_id, seq_id)`, with fail-closed
/// ambiguity detection whenever a restraint omits `chain_id` on an assembly where `seq_id` appears
/// on multiple chains (`21.W2:P1-8`).
#[derive(Clone, Debug, Default)]
pub struct ChainSeqIndex {
    pub exact: BTreeMap<(String, i32), usize>,
    pub by_seq_only: BTreeMap<i32, Vec<(String, usize)>>,
}

impl ChainSeqIndex {
    pub fn build(chain_ids: &[String], seq_ids: &[i32]) -> Result<Self, String> {
        if chain_ids.len() != seq_ids.len() {
            return Err(format!(
                "ChainSeqIndex length mismatch: {} chain_ids vs {} seq_ids",
                chain_ids.len(),
                seq_ids.len()
            ));
        }
        let mut exact = BTreeMap::new();
        let mut by_seq_only: BTreeMap<i32, Vec<(String, usize)>> = BTreeMap::new();
        for (idx, (cid_raw, &sid)) in chain_ids.iter().zip(seq_ids.iter()).enumerate() {
            let cid = if cid_raw.trim().is_empty() {
                "A".to_string()
            } else {
                cid_raw.trim().to_string()
            };
            if exact.insert((cid.clone(), sid), idx).is_some() {
                return Err(format!(
                    "duplicate residue ({cid}:{sid}) in structure model"
                ));
            }
            by_seq_only.entry(sid).or_default().push((cid, idx));
        }
        Ok(Self { exact, by_seq_only })
    }

    pub fn from_kin(kin: &DomainLinkerKinematics) -> Result<Self, String> {
        let cids: Vec<String> = kin.residues.iter().map(|r| r.chain_id.clone()).collect();
        let sids: Vec<i32> = kin.residues.iter().map(|r| r.seq_id).collect();
        Self::build(&cids, &sids)
    }

    pub fn from_pdb_chain(chain: &PdbChain) -> Result<Self, String> {
        let cids: Vec<String> = chain
            .residues
            .iter()
            .map(|r| r.chain_id.to_string())
            .collect();
        let sids: Vec<i32> = chain.residues.iter().map(|r| r.seq_id).collect();
        Self::build(&cids, &sids)
    }

    /// Resolve `(chain_opt, seq_id)` -> `Result<Option<usize>, String>`.
    /// Returns `Err` if `chain_opt` is `None`/empty AND `seq_id` exists on `>= 2` chains!
    pub fn resolve(&self, chain_opt: Option<&str>, seq_id: i32) -> Result<Option<usize>, String> {
        if let Some(c) = chain_opt
            .map(str::trim)
            .filter(|s| !s.is_empty() && *s != ".")
        {
            return Ok(self.exact.get(&(c.to_string(), seq_id)).copied());
        }
        match self.by_seq_only.get(&seq_id) {
            None => Ok(None),
            Some(list) if list.len() == 1 => Ok(Some(list[0].1)),
            Some(list) => {
                let chains: Vec<&str> = list.iter().map(|(c, _)| c.as_str()).collect();
                Err(format!(
                    "ambiguous un-chained restraint on seq_id {seq_id} across multiple chains ({}); specify explicit chain_id",
                    chains.join(", ")
                ))
            }
        }
    }
}

/// Check whether residue `idx` belongs to a confident rigid domain (`domain_id >= 1` AND
/// `pLDDT >= min_ordered_plddt` when `pLDDT` is present, `21.W2:P1-4`).
#[inline]
fn is_confident_rigid_residue(
    idx: usize,
    partition: &PaeDomainPartition,
    plddts_opt: Option<&[Option<f64>]>,
    min_ordered_plddt: f64,
) -> Option<usize> {
    let dom = partition.domain_labels.get(idx).copied().unwrap_or(0);
    if dom == 0 {
        return None;
    }
    if !matches!(
        partition.residue_roles.get(idx),
        Some(ResidueDomainRole::RigidDomain { .. })
    ) {
        return None;
    }
    if let Some(pls) = plddts_opt {
        if let Some(Some(pl)) = pls.get(idx) {
            if *pl < min_ordered_plddt {
                return None;
            }
        }
    }
    Some(dom)
}

/// Filter `DomainRdcMediumSet` to strictly intra-domain confident rigid-domain couplings
/// (`dom_1 == dom_2 >= 1` and `pLDDT >= min_ordered_plddt`, `21.W2:P1-4`), returning
/// `(filtered_media, n_excluded_flexible_or_low_plddt)`.
pub(crate) fn filter_confident_rigid_domain_rdcs(
    media: &[DomainRdcMediumSet],
    cseq: &ChainSeqIndex,
    partition: &PaeDomainPartition,
    plddts_opt: Option<&[Option<f64>]>,
    min_ordered_plddt: f64,
) -> Result<(Vec<DomainRdcMediumSet>, usize), String> {
    let mut out = Vec::with_capacity(media.len());
    let mut excluded = 0usize;
    for m in media {
        let mut kept = Vec::with_capacity(m.restraints.len());
        for r in &m.restraints {
            r.validate()?;
            let idx1 = cseq.resolve(r.chain_id_1.as_deref(), r.seq_id_1)?;
            let idx2 = cseq.resolve(r.chain_id_2.as_deref(), r.seq_id_2)?;
            let (Some(i1), Some(i2)) = (idx1, idx2) else {
                continue;
            };
            let d1 = is_confident_rigid_residue(i1, partition, plddts_opt, min_ordered_plddt);
            let d2 = is_confident_rigid_residue(i2, partition, plddts_opt, min_ordered_plddt);
            match (d1, d2) {
                (Some(da), Some(db)) if da == db => kept.push(r.clone()),
                _ => {
                    excluded += 1;
                }
            }
        }
        out.push(DomainRdcMediumSet {
            medium_id: m.medium_id.clone(),
            restraints: kept,
        });
    }
    Ok((out, excluded))
}

/// Deterministically hold out `free_fraction` (e.g. `0.20`) of peptide planes **stratified strictly
/// within confident rigid domains `1..=D`** (`RdcFreeStratification::ByPeptidePlane`, `0` linker/tail
/// couplings in `RdcFree`), keyed by `(chain_id, seq_id)` while preserving $n_{\text{work}}(d, m) \ge 8$.
#[allow(clippy::too_many_arguments)]
pub fn partition_domain_stratified_domain_rdc_free_set(
    media: &mut [DomainRdcMediumSet],
    cseq: &ChainSeqIndex,
    chain_ids: &[String],
    seq_ids: &[i32],
    partition: &PaeDomainPartition,
    plddts_opt: Option<&[Option<f64>]>,
    min_ordered_plddt: f64,
    free_fraction: f64,
    seed: u64,
) -> Result<(), String> {
    let frac = free_fraction.clamp(0.0, 0.50);
    for m in media.iter_mut() {
        for r in &mut m.restraints {
            r.is_free = false;
        }
    }
    if frac <= 0.0 || media.is_empty() || partition.domains.is_empty() {
        return Ok(());
    }

    type RigidPlaneKey = ((String, i32), usize);
    let resolve_plane_key_and_dom =
        |r: &DomainRdcRestraint| -> Result<Option<RigidPlaneKey>, String> {
            let Some(i1) = cseq.resolve(r.chain_id_1.as_deref(), r.seq_id_1)? else {
                return Ok(None);
            };
            let Some(i2) = cseq.resolve(r.chain_id_2.as_deref(), r.seq_id_2)? else {
                return Ok(None);
            };
            let Some(d1) = is_confident_rigid_residue(i1, partition, plddts_opt, min_ordered_plddt)
            else {
                return Ok(None);
            };
            let Some(d2) = is_confident_rigid_residue(i2, partition, plddts_opt, min_ordered_plddt)
            else {
                return Ok(None);
            };
            if d1 != d2 {
                return Ok(None);
            }
            let pid = r.peptide_plane_seq_id();
            let p_idx = if pid == seq_ids[i1] {
                i1
            } else if pid == seq_ids[i2] {
                i2
            } else {
                cseq.resolve(Some(&chain_ids[i1]), pid)?.unwrap_or(i1)
            };
            // Also verify the plane anchor residue itself (e.g. `i + 1` for CACO(i)) is in the same confident rigid domain
            if is_confident_rigid_residue(p_idx, partition, plddts_opt, min_ordered_plddt)
                != Some(d1)
            {
                return Ok(None);
            }
            Ok(Some(((chain_ids[p_idx].clone(), pid), d1)))
        };

    let n_media = media.len();
    let max_dom = partition.domains.len();
    let mut selected_planes: BTreeSet<(String, i32)> = BTreeSet::new();
    let mut rng = DetRng::new(seed ^ 0x5244_435F_444F_4D46);

    // Stratify strictly across confident rigid domains `1..=max_dom` (NEVER `bucket == 0` linker/tail!)
    for bucket in 1..=max_dom {
        let mut plane_counts: BTreeMap<(String, i32), Vec<usize>> = BTreeMap::new();
        let mut med_totals = vec![0usize; n_media];
        let mut bucket_total = 0usize;

        for (m_idx, m) in media.iter().enumerate() {
            for r in &m.restraints {
                let Some((pkey, dom_p)) = resolve_plane_key_and_dom(r)? else {
                    continue;
                };
                if dom_p != bucket {
                    continue;
                }
                med_totals[m_idx] += 1;
                bucket_total += 1;
                let entry = plane_counts
                    .entry(pkey)
                    .or_insert_with(|| vec![0usize; n_media]);
                entry[m_idx] += 1;
            }
        }

        let min_keep = MIN_TOLMAN_DOMAIN_WORK_RDCS;
        if bucket_total == 0 || plane_counts.is_empty() || med_totals.iter().all(|&t| t <= min_keep)
        {
            continue;
        }

        let target_free = ((bucket_total as f64) * frac).round().max(1.0) as usize;
        let mut planes: Vec<(String, i32)> = plane_counts.keys().cloned().collect();
        for i in (1..planes.len()).rev() {
            let j = rng.range_usize(0, i + 1);
            planes.swap(i, j);
        }

        let mut med_free = vec![0usize; n_media];
        let mut cur_free = 0usize;

        for pkey in planes {
            let counts = &plane_counts[&pkey];
            let add_tot: usize = counts.iter().sum();
            if add_tot == 0 {
                continue;
            }
            let keeps_work = counts.iter().enumerate().all(|(m_idx, &c)| {
                if c == 0 {
                    true
                } else if med_totals[m_idx] <= min_keep {
                    false
                } else {
                    med_totals[m_idx] - (med_free[m_idx] + c) >= min_keep
                }
            });
            if !keeps_work {
                continue;
            }
            let cur_err = cur_free.abs_diff(target_free);
            let nxt_err = (cur_free + add_tot).abs_diff(target_free);
            if cur_free == 0 || nxt_err <= cur_err {
                selected_planes.insert(pkey);
                for (m_idx, &c) in counts.iter().enumerate() {
                    med_free[m_idx] += c;
                }
                cur_free += add_tot;
                if cur_free >= target_free {
                    break;
                }
            }
        }
    }

    for m in media.iter_mut() {
        for r in &mut m.restraints {
            if let Ok(Some((pkey, _))) = resolve_plane_key_and_dom(r) {
                if selected_planes.contains(&pkey) {
                    r.is_free = true;
                }
            }
        }
    }
    Ok(())
}

/// Deterministically hold out `free_fraction` (e.g. `0.20`) of peptide planes **stratified strictly
/// within confident rigid domains `1..=D`** (`RdcFreeStratification::ByPeptidePlane`, `0` linker/tail
/// couplings in `RdcFree`).
pub fn partition_domain_stratified_rdc_free_set(
    media: &mut [RdcMediumSet],
    partition: &PaeDomainPartition,
    seq_ids: &[i32],
    free_fraction: f64,
    seed: u64,
) {
    let chain_ids = vec!["A".to_string(); seq_ids.len()];
    let Ok(cseq) = ChainSeqIndex::build(&chain_ids, seq_ids) else {
        return;
    };
    let mut dom_media = DomainRdcMediumSet::from_rdc_media(media);
    if partition_domain_stratified_domain_rdc_free_set(
        &mut dom_media,
        &cseq,
        &chain_ids,
        seq_ids,
        partition,
        None,
        DEFAULT_MIN_ORDERED_PLDDT,
        free_fraction,
        seed,
    )
    .is_ok()
    {
        for (dst_m, src_m) in media.iter_mut().zip(dom_media.iter()) {
            for (dst_r, src_r) in dst_m.restraints.iter_mut().zip(src_m.restraints.iter()) {
                dst_r.is_free = src_r.is_free;
            }
        }
    }
}

/// Resolve an atom coordinate on `PdbResidue` matching `nmr_core::rdc::resolve_chain_rdc_atom`.
fn resolve_pdb_residue_atom_pos(res: &PdbResidue, atom_name: &str) -> Option<Vec3> {
    let norm = normalize_rdc_atom_name(atom_name);
    let is_gly = res.res_name.eq_ignore_ascii_case("GLY");
    match norm {
        "N" => Some(res.n),
        "H" => res.h,
        "CA" => Some(res.ca),
        "HA" => (!is_gly).then_some(res.ha),
        "HA2" => is_gly.then_some(res.ha2).flatten(),
        "HA3" => is_gly.then_some(res.ha3).flatten(),
        "C" => Some(res.c),
        "O" => Some(res.o),
        "CB" => res.cb,
        _ => None,
    }
}

/// Chain-explicit `(chain_id, seq_id)` 5-parameter Losonczi Saupe SVD fit (`n_work >= 6`) returning
/// `(SaupeTensorFit, Vec<(chain_id, RdcBackCalcPoint)>)`.
pub(crate) fn fit_saupe_tensor_domain_rdc(
    chain: &PdbChain,
    cseq: &ChainSeqIndex,
    medium: &DomainRdcMediumSet,
) -> Result<(SaupeTensorFit, Vec<(String, RdcBackCalcPoint)>), String> {
    if chain.residues.is_empty() {
        return Err("cannot fit Saupe tensor on an empty PdbChain".to_string());
    }
    let mut qr = DockGivensQr5::new();
    let mut resolved: Vec<(&DomainRdcRestraint, String, Vec3)> = Vec::new();
    let mut n_work = 0usize;

    for r in &medium.restraints {
        r.validate()?;
        let (Some(i1), Some(i2)) = (
            cseq.resolve(r.chain_id_1.as_deref(), r.seq_id_1)?,
            cseq.resolve(r.chain_id_2.as_deref(), r.seq_id_2)?,
        ) else {
            continue;
        };
        let (Some(p1), Some(p2)) = (
            resolve_pdb_residue_atom_pos(&chain.residues[i1], &r.atom_name_1),
            resolve_pdb_residue_atom_pos(&chain.residues[i2], &r.atom_name_2),
        ) else {
            continue;
        };
        let dv = v_sub(p2, p1);
        let dist = v_norm(dv);
        if !dist.is_finite() || dist < 1e-6 {
            return Err(format!(
                "zero/degenerate internuclear distance ({dist}) on {}:{} - {}:{}",
                r.seq_id_1, r.atom_name_1, r.seq_id_2, r.atom_name_2
            ));
        }
        let u = v_scale(dv, 1.0 / dist);
        let ch_str = chain.residues[i1].chain_id.to_string();
        resolved.push((r, ch_str, u));

        if !r.is_free {
            let ws = r.d_max_hz / r.sigma_hz;
            qr.push_row(
                [
                    ws * (u[1] * u[1] - u[0] * u[0]),
                    ws * (u[2] * u[2] - u[0] * u[0]),
                    ws * (2.0 * u[0] * u[1]),
                    ws * (2.0 * u[0] * u[2]),
                    ws * (2.0 * u[1] * u[2]),
                ],
                r.d_obs_hz / r.sigma_hz,
            );
            n_work += 1;
        }
    }

    if n_work < 6 {
        return Err(format!(
            "alignment medium '{}' resolved only {n_work} finite working RDCs; require n >= 6",
            medium.medium_id
        ));
    }
    let (s_matrix, cond) = qr.solve_saupe_with_cond().ok_or_else(|| {
        format!(
            "alignment medium '{}' Saupe SVD ill-conditioned (cond > {:.1e})",
            medium.medium_id, MAX_SAUPE_SVD_COND
        )
    })?;
    let s_5 = [
        s_matrix[1][1],
        s_matrix[2][2],
        s_matrix[0][1],
        s_matrix[0][2],
        s_matrix[1][2],
    ];
    let (principal_s, _rot, euler_zyz_deg) = decompose_saupe_matrix(s_matrix);
    let da_nh_hz = 0.5 * DMAX_NH_HZ * principal_s[2];
    let rhombicity = if principal_s[2].abs() > 1e-18 {
        ((2.0 / 3.0) * (principal_s[0] - principal_s[1]) / principal_s[2]).clamp(0.0, 2.0 / 3.0)
    } else {
        0.0
    };

    let mut work_norm_pairs = Vec::with_capacity(n_work);
    let mut free_norm_pairs = Vec::new();
    let mut work_sq_hz = 0.0_f64;
    let mut free_sq_hz = 0.0_f64;
    let mut points = Vec::with_capacity(resolved.len());

    for (r, ch_str, u) in resolved {
        let d_calc = calc_rdc_from_unit_vec(u, r.d_max_hz, &s_matrix);
        let scale_nh = DMAX_NH_HZ / r.d_max_hz;
        let obs_norm = r.d_obs_hz * scale_nh;
        let calc_norm = d_calc * scale_nh;
        let diff_hz = r.d_obs_hz - d_calc;
        if r.is_free {
            free_norm_pairs.push((obs_norm, calc_norm));
            free_sq_hz += diff_hz * diff_hz;
        } else {
            work_norm_pairs.push((obs_norm, calc_norm));
            work_sq_hz += diff_hz * diff_hz;
        }
        points.push((
            ch_str,
            RdcBackCalcPoint {
                medium_id: medium.medium_id.clone(),
                seq_id: r.seq_id_1,
                res_name: r.res_name_1.clone(),
                bond_label: r.bond_type.canonical_label().to_string(),
                d_obs_hz: r.d_obs_hz,
                d_calc_hz: d_calc,
                d_obs_norm_nh_hz: obs_norm,
                d_calc_norm_nh_hz: calc_norm,
                sigma_hz: r.sigma_hz,
                is_free: r.is_free,
            },
        ));
    }

    let n_free = free_norm_pairs.len();
    let q_work = cornilescu_q(&work_norm_pairs);
    let q_free = cornilescu_q(&free_norm_pairs);
    let rmsd_work_hz = (work_sq_hz / (n_work as f64)).sqrt();
    let rmsd_free_hz = if n_free > 0 {
        (free_sq_hz / (n_free as f64)).sqrt()
    } else {
        f64::NAN
    };

    Ok((
        SaupeTensorFit {
            medium_id: medium.medium_id.clone(),
            s_5,
            s_matrix,
            principal_s,
            da_nh_hz,
            rhombicity,
            euler_zyz_deg,
            condition_number: cond,
            n_work,
            n_free,
            q_work,
            q_free,
            rmsd_work_hz,
            rmsd_free_hz,
        },
        points,
    ))
}

/// Perturb atom coordinates of `chain` by isotropic Gaussian structural noise `sigma_coord_a` (Å)
/// for Tolman $\vartheta$ Monte Carlo propagation of intra-domain coordinate error (`21.W2:P1-6`).
fn jitter_pdb_chain_coords(chain: &PdbChain, sigma_coord_a: f64, rng: &mut DetRng) -> PdbChain {
    if sigma_coord_a <= 0.0 {
        return chain.clone();
    }
    let mut jv = |p: Vec3| -> Vec3 {
        [
            p[0] + rng.normal(0.0, sigma_coord_a),
            p[1] + rng.normal(0.0, sigma_coord_a),
            p[2] + rng.normal(0.0, sigma_coord_a),
        ]
    };
    let mut out = Vec::with_capacity(chain.residues.len());
    for r in &chain.residues {
        out.push(PdbResidue {
            seq_id: r.seq_id,
            res_name: r.res_name.clone(),
            chain_id: r.chain_id,
            n: jv(r.n),
            h: r.h.map(&mut jv),
            ca: jv(r.ca),
            ha: jv(r.ha),
            ha2: r.ha2.map(&mut jv),
            ha3: r.ha3.map(&mut jv),
            c: jv(r.c),
            o: jv(r.o),
            cb: r.cb.map(&mut jv),
            qb: r.qb.map(&mut jv),
        });
    }
    PdbChain { residues: out }
}

/// Evaluate Tolman et al. (2001) per-domain Saupe SVD & Generalized Degree of Order ($\vartheta_{d,m}$)
/// across all rigid domains (`n_work >= 8`, `cond <= 1e8`, strictly `!is_free` working couplings)
/// with parametric Monte Carlo uncertainty propagation including BOTH measurement $\sigma_k$ AND
/// coordinate-noise floor `coord_noise_angstrom` + empirical intra-domain SVD excess $\chi^2_\nu$ (`21.W2:P1-6`).
pub fn evaluate_tolman_domain_gdo(
    chain: &PdbChain,
    partition: &PaeDomainPartition,
    media: &[RdcMediumSet],
    n_mc_trials: usize,
    seed: u64,
) -> TolmanGdoMotionReport {
    let Ok(cseq) = ChainSeqIndex::from_pdb_chain(chain) else {
        return TolmanGdoMotionReport {
            regime: DomainMotionRegime::InsufficientDomainRdcs,
            domain_fits: Vec::new(),
            skipped_domains: Vec::new(),
            min_rho_gdo: f64::NAN,
            max_abs_delta_over_theta1: f64::NAN,
            warning_message: None,
        };
    };
    let dom_media = DomainRdcMediumSet::from_rdc_media(media);
    evaluate_tolman_domain_gdo_multi_chain(
        chain,
        &cseq,
        partition,
        None,
        DEFAULT_MIN_ORDERED_PLDDT,
        &dom_media,
        n_mc_trials,
        DEFAULT_TOLMAN_COORD_NOISE_ANGSTROM,
        seed,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_tolman_domain_gdo_multi_chain(
    chain: &PdbChain,
    cseq: &ChainSeqIndex,
    partition: &PaeDomainPartition,
    plddts_opt: Option<&[Option<f64>]>,
    min_ordered_plddt: f64,
    media: &[DomainRdcMediumSet],
    n_mc_trials: usize,
    coord_noise_angstrom: f64,
    seed: u64,
) -> TolmanGdoMotionReport {
    if partition.domains.is_empty() || media.is_empty() || chain.residues.is_empty() {
        return TolmanGdoMotionReport {
            regime: DomainMotionRegime::InsufficientDomainRdcs,
            domain_fits: Vec::new(),
            skipped_domains: Vec::new(),
            min_rho_gdo: f64::NAN,
            max_abs_delta_over_theta1: f64::NAN,
            warning_message: None,
        };
    }

    let trials = n_mc_trials.clamp(32, 2_000);
    let eff_coord_sig = coord_noise_angstrom.max(0.0);
    let mut rng = DetRng::new(seed ^ 0x544F_4C4D_414E_3031);
    let mut domain_fits: Vec<DomainMediumGdoFit> = Vec::new();
    let mut skipped_domains: Vec<(usize, String, String)> = Vec::new();
    let mut decoupled_triggered = false;
    let mut min_rho_gdo = f64::INFINITY;
    let mut max_abs_delta_over_theta1 = 0.0_f64;
    let mut has_multi_domain_medium = false;

    for medium in media {
        struct RawDomFit {
            domain_id: usize,
            fit: SaupeTensorFit,
            theta: f64,
            mc_thetas: Vec<f64>,
        }
        let mut med_fits: Vec<RawDomFit> = Vec::new();

        for dom in &partition.domains {
            let dom_idx_set: BTreeSet<usize> = dom
                .residue_indices
                .iter()
                .copied()
                .filter(|&idx| {
                    is_confident_rigid_residue(idx, partition, plddts_opt, min_ordered_plddt)
                        == Some(dom.domain_id)
                })
                .collect();

            let mut sub_restraints: Vec<DomainRdcRestraint> = Vec::new();
            for r in &medium.restraints {
                let (Ok(Some(i1)), Ok(Some(i2))) = (
                    cseq.resolve(r.chain_id_1.as_deref(), r.seq_id_1),
                    cseq.resolve(r.chain_id_2.as_deref(), r.seq_id_2),
                ) else {
                    continue;
                };
                if dom_idx_set.contains(&i1) && dom_idx_set.contains(&i2) {
                    sub_restraints.push(r.clone());
                }
            }

            let sub_work_cnt = sub_restraints.iter().filter(|r| !r.is_free).count();
            if sub_work_cnt < MIN_TOLMAN_DOMAIN_WORK_RDCS {
                skipped_domains.push((
                    dom.domain_id,
                    medium.medium_id.clone(),
                    format!(
                        "only {sub_work_cnt} working RDCs in domain {} (require >= {MIN_TOLMAN_DOMAIN_WORK_RDCS})",
                        dom.domain_id
                    ),
                ));
                continue;
            }
            let sub_medium = DomainRdcMediumSet {
                medium_id: medium.medium_id.clone(),
                restraints: sub_restraints,
            };
            match fit_saupe_tensor_domain_rdc(chain, cseq, &sub_medium) {
                Ok((fit, pts)) => {
                    if fit.n_work < MIN_TOLMAN_DOMAIN_WORK_RDCS
                        || fit.condition_number > MAX_SAUPE_SVD_COND
                    {
                        skipped_domains.push((
                            dom.domain_id,
                            medium.medium_id.clone(),
                            format!(
                                "resolved n_work={} or cond={:.2e} outside bounds",
                                fit.n_work, fit.condition_number
                            ),
                        ));
                        continue;
                    }
                    let theta = compute_saupe_gdo(&fit.s_matrix);
                    // Compute empirical reduced chi^2 on working couplings so forward-model / AF
                    // coordinate error beyond nominal measurement sigma inflates MC uncertainty
                    // instead of spuriously over-calling `DecoupledInterDomainMotion` (`21.W2:P1-6`).
                    let mut work_chi2 = 0.0_f64;
                    let mut work_cnt = 0usize;
                    for (_, pt) in &pts {
                        if !pt.is_free {
                            let z = (pt.d_obs_hz - pt.d_calc_hz) / pt.sigma_hz.max(1e-6);
                            work_chi2 += z * z;
                            work_cnt += 1;
                        }
                    }
                    let dof = work_cnt.saturating_sub(5).max(1) as f64;
                    let chi2_red_scale = (work_chi2 / dof).max(1.0).sqrt();

                    let mut mc_thetas = Vec::with_capacity(trials);
                    let work_only: Vec<DomainRdcRestraint> = sub_medium
                        .restraints
                        .iter()
                        .filter(|r| !r.is_free)
                        .cloned()
                        .collect();
                    for _ in 0..trials {
                        let trial_chain = jitter_pdb_chain_coords(chain, eff_coord_sig, &mut rng);
                        let mut trial_med = DomainRdcMediumSet {
                            medium_id: medium.medium_id.clone(),
                            restraints: work_only.clone(),
                        };
                        for r in &mut trial_med.restraints {
                            r.d_obs_hz = rng.normal(r.d_obs_hz, r.sigma_hz * chi2_red_scale);
                        }
                        if let Ok((t_fit, _)) =
                            fit_saupe_tensor_domain_rdc(&trial_chain, cseq, &trial_med)
                        {
                            mc_thetas.push(compute_saupe_gdo(&t_fit.s_matrix));
                        }
                    }
                    med_fits.push(RawDomFit {
                        domain_id: dom.domain_id,
                        fit,
                        theta,
                        mc_thetas,
                    });
                }
                Err(err) => {
                    skipped_domains.push((dom.domain_id, medium.medium_id.clone(), err));
                }
            }
        }

        if med_fits.is_empty() {
            continue;
        }
        if med_fits.len() >= 2 {
            has_multi_domain_medium = true;
        }

        let max_idx = med_fits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.theta.total_cmp(&b.theta))
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        let theta_max = med_fits[max_idx].theta.max(1e-30);
        let theta_dom1 = med_fits
            .iter()
            .find(|f| f.domain_id == 1)
            .map(|f| f.theta)
            .unwrap_or(med_fits[0].theta)
            .max(1e-30);

        for (f_idx, rf) in med_fits.iter().enumerate() {
            let sig_theta = sample_std_dev(&rf.mc_thetas);
            let rho = (rf.theta / theta_max).clamp(0.0, 1.0);
            let ratio_d1 = rf.theta / theta_dom1;

            let sig_rho = if f_idx == max_idx {
                0.0
            } else {
                let n_pair = rf.mc_thetas.len().min(med_fits[max_idx].mc_thetas.len());
                if n_pair >= 8 {
                    let ratios: Vec<f64> = (0..n_pair)
                        .map(|t| rf.mc_thetas[t] / med_fits[max_idx].mc_thetas[t].max(1e-30))
                        .collect();
                    sample_std_dev(&ratios)
                } else {
                    let sm = sample_std_dev(&med_fits[max_idx].mc_thetas);
                    rho * ((sig_theta / rf.theta.max(1e-30)).powi(2) + (sm / theta_max).powi(2))
                        .sqrt()
                }
            };

            if med_fits.len() >= 2 {
                if rho < min_rho_gdo {
                    min_rho_gdo = rho;
                }
                let rel_d1 = (rf.theta - theta_dom1).abs() / theta_dom1;
                if rel_d1 > max_abs_delta_over_theta1 {
                    max_abs_delta_over_theta1 = rel_d1;
                }
                if rho < TOLMAN_GDO_RATIO_THRESHOLD
                    && (TOLMAN_GDO_RATIO_THRESHOLD - rho) > 2.0 * sig_rho
                {
                    decoupled_triggered = true;
                }
            }

            domain_fits.push(DomainMediumGdoFit {
                domain_id: rf.domain_id,
                medium_id: medium.medium_id.clone(),
                n_work: rf.fit.n_work,
                n_free: rf.fit.n_free,
                gdo_theta: rf.theta,
                gdo_sigma_mc: sig_theta,
                ratio_to_domain_1: ratio_d1,
                rho_gdo: rho,
                rho_sigma_mc: sig_rho,
                da_nh_hz: rf.fit.da_nh_hz,
                rhombicity: rf.fit.rhombicity,
                condition_number: rf.fit.condition_number,
                q_work: rf.fit.q_work,
                q_free: rf.fit.q_free,
                saupe_fit: rf.fit.clone(),
            });
        }
    }

    if !has_multi_domain_medium {
        return TolmanGdoMotionReport {
            regime: DomainMotionRegime::InsufficientDomainRdcs,
            domain_fits,
            skipped_domains,
            min_rho_gdo: f64::NAN,
            max_abs_delta_over_theta1: f64::NAN,
            warning_message: if partition.domains.len() >= 2 {
                Some(
                    "Fewer than 2 rigid domains met the n_work >= 8 Saupe SVD threshold for Tolman GDO inter-domain motion testing."
                        .to_string(),
                )
            } else {
                None
            },
        };
    }

    let (regime, warning_message) = if decoupled_triggered {
        (
            DomainMotionRegime::DecoupledInterDomainMotion,
            Some(format!(
                "Tolman GDO motion test detected DecoupledInterDomainMotion (min rho_GDO = {:.3} < {:.2} at > 2*sigma_MC): large-amplitude inter-domain orientational averaging is present; single-conformer shared-tensor static docking would over-interpret solution dynamics, so per-domain dipolar amplitudes are scaled by rho_GDO (use --ensemble-size > 1 for multi-conformer representation).",
                min_rho_gdo, TOLMAN_GDO_RATIO_THRESHOLD
            )),
        )
    } else {
        (DomainMotionRegime::RigidlyCoupledDomains, None)
    };

    TolmanGdoMotionReport {
        regime,
        domain_fits,
        skipped_domains,
        min_rho_gdo,
        max_abs_delta_over_theta1,
        warning_message,
    }
}

fn sample_std_dev(vals: &[f64]) -> f64 {
    if vals.len() <= 1 {
        return 0.0;
    }
    let mean = vals.iter().sum::<f64>() / (vals.len() as f64);
    let var = vals
        .iter()
        .map(|&v| {
            let d = v - mean;
            d * d
        })
        .sum::<f64>()
        / ((vals.len() - 1) as f64);
    var.sqrt()
}

// ============================================================================
// 3. Multi-Domain RDC, PCS, PRE & NOE Restraint Types and TSV Parsers
// ============================================================================

/// Single pseudocontact shift (`PCS`) restraint resolved to `(chain_id, seq_id, atom_name)`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DomainPcsRestraint {
    pub tag_id: String,
    #[serde(default)]
    pub chain_id: Option<String>,
    pub seq_id: i32,
    pub atom_name: String,
    pub pcs_obs_ppm: f64,
    pub sigma_ppm: f64,
    pub is_free: bool,
}

/// Paramagnetic lanthanide / metalloprotein `PCS` dataset anchored in a rigid domain $d_0$.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DomainPcsTagSet {
    pub tag_id: String,
    /// Optional explicit 3D metal position `(x, y, z)` in Å in the input AF model frame.
    pub metal_pos_initial: Option<Vec3>,
    /// Optional `chain_id` for `anchor_residue_atom` (`None` = unambiguous single-chain lookup).
    #[serde(default)]
    pub anchor_chain_id: Option<String>,
    /// Optional `(seq_id, atom_name)` anchor inside the host rigid domain used when `metal_pos_initial` is `None`.
    pub anchor_residue_atom: Option<(i32, String)>,
    /// Optional explicit 1-based host `domain_id` (`None` = auto-detect from `metal_pos_initial` or intra-domain PCSs).
    pub host_domain_id: Option<usize>,
    /// Optional pre-fitted `DeltaChiTensor` in the input AF model frame (`None` = fit from working intra-$d_0$ PCSs).
    pub pre_fitted_tensor: Option<DeltaChiTensor>,
    /// When `true` (or when `metal_pos_initial` is `None` and `>= 8` working intra-domain PCSs exist),
    /// jointly refines the 3D metal position + 5-parameter $\Delta\boldsymbol{\chi}$ via 8-parameter VarPro (`21.W2:P2-10`).
    #[serde(default)]
    pub refine_metal_pos: bool,
    /// Search radius (Å) around the initial metal guess during 8-parameter VarPro (`None` = config default `8.0 Å`).
    #[serde(default)]
    pub metal_search_radius_a: Option<f64>,
    pub restraints: Vec<DomainPcsRestraint>,
}

/// Functional form of a paramagnetic relaxation enhancement (`PRE`) restraint.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum PreRestraintKind {
    /// Quantitative Solomon $\Gamma_2 = K_{\text{PRE}} / r_{\text{M}}^6$ transverse relaxation rate ($\text{s}^{-1}$).
    Gamma2Rate {
        gamma2_obs_s1: f64,
        sigma_s1: f64,
        k_pre_a6_s1: f64,
    },
    /// Distance bounds $[r_{\text{lower}}, r_{\text{upper}}]$ in Å derived from `PRE` intensity ratios / bleaching.
    DistanceBounds {
        lower_a: f64,
        upper_a: f64,
        sigma_a: f64,
    },
}

/// Single `PRE` restraint between paramagnetic site `site_id` and nuclear `(chain_id, seq_id, atom_name)`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DomainPreRestraint {
    pub site_id: String,
    #[serde(default)]
    pub chain_id: Option<String>,
    pub seq_id: i32,
    pub atom_name: String,
    pub kind: PreRestraintKind,
}

/// Paramagnetic spin-label / metal `PRE` dataset anchored in a rigid domain $d_0$.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DomainPreSiteSet {
    pub site_id: String,
    /// Explicit 3D paramagnetic center `(x, y, z)` in Å in the input AF model frame.
    pub site_pos_initial: Option<Vec3>,
    /// Optional `chain_id` for `anchor_residue_atom`.
    #[serde(default)]
    pub anchor_chain_id: Option<String>,
    /// Optional `(seq_id, atom_name)` covalent spin-label anchor inside host domain $d_0$.
    pub anchor_residue_atom: Option<(i32, String)>,
    pub host_domain_id: Option<usize>,
    /// Mandatory effective electron-nuclear correlation time $\tau_c$ in ns whenever `Gamma2Rate`
    /// restraints are present (`21.W3:P0-4`; fail closed `Err` if omitted!).
    #[serde(default)]
    pub tau_c_ns: Option<f64>,
    /// Mandatory $^1\text{H}$ spectrometer Larmor frequency in MHz for the dispersive Solomon term
    /// $3\tau_c / (1 + \omega_{\text{H}}^2 \tau_c^2)$ (`21.W3:P0-4`; fail closed `Err` if omitted!).
    #[serde(default)]
    pub proton_mhz: Option<f64>,
    /// Optional flexible spin-label tether allowance in Å (`None` = `0.0 Å` for `# site_pos` or
    /// `cfg.mtsl_tether_allowance_a` for `# site_anchor`).
    #[serde(default)]
    pub mtsl_tether_allowance_a: Option<f64>,
    pub restraints: Vec<DomainPreRestraint>,
}

/// Deterministically hold out `free_fraction` (`0.20`) of `PCS` observations stratified by
/// `(tag_id, rigid_domain_id)` on `(chain_id, seq_id)` residue units (`21.W2:P1-5`), preserving
/// sufficient working intra-domain PCSs (`>= 8` when 8-param metal refinement is active, else `>= 6`)
/// in the host domain so $\Delta\boldsymbol{\chi}$ fitting never starves.
pub fn partition_domain_stratified_pcs_free_set(
    pcs_tags: &mut [DomainPcsTagSet],
    cseq: &ChainSeqIndex,
    partition: &PaeDomainPartition,
    plddts_opt: Option<&[Option<f64>]>,
    min_ordered_plddt: f64,
    free_fraction: f64,
    seed: u64,
) -> Result<(), String> {
    let frac = free_fraction.clamp(0.0, 0.50);
    for t in pcs_tags.iter_mut() {
        for r in &mut t.restraints {
            r.is_free = false;
        }
    }
    if frac <= 0.0 || pcs_tags.is_empty() || partition.domains.is_empty() {
        return Ok(());
    }

    let mut rng = DetRng::new(seed ^ 0x5043_535F_4652_4545);
    let max_dom = partition.domains.len();

    for tag in pcs_tags.iter_mut() {
        let host_dom_hint = tag.host_domain_id.or_else(|| {
            tag.anchor_residue_atom.as_ref().and_then(|(sid, _)| {
                cseq.resolve(tag.anchor_chain_id.as_deref(), *sid)
                    .ok()
                    .flatten()
                    .and_then(|idx| partition.domain_labels.get(idx).copied())
                    .filter(|&d| d >= 1)
            })
        });
        let min_intra_keep = if tag.refine_metal_pos || tag.metal_pos_initial.is_none() {
            9usize
        } else {
            6usize
        };

        // Stratify strictly across confident rigid domains `1..=max_dom`
        for dom_id in 1..=max_dom {
            let mut res_obs_indices: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
            let mut bucket_total = 0usize;
            for (r_idx, pr) in tag.restraints.iter().enumerate() {
                let Some(res_i) = cseq.resolve(pr.chain_id.as_deref(), pr.seq_id)? else {
                    continue;
                };
                if is_confident_rigid_residue(res_i, partition, plddts_opt, min_ordered_plddt)
                    == Some(dom_id)
                {
                    res_obs_indices.entry(res_i).or_default().push(r_idx);
                    bucket_total += 1;
                }
            }
            let min_keep = if host_dom_hint == Some(dom_id) || host_dom_hint.is_none() {
                // When host_domain_id is auto-detected from coordinates later, conservatively keep >= min_intra_keep
                if host_dom_hint == Some(dom_id) {
                    min_intra_keep
                } else {
                    min_intra_keep.min(bucket_total.saturating_sub(1))
                }
            } else {
                1usize
            };
            if bucket_total <= min_keep || res_obs_indices.len() <= 1 {
                continue;
            }
            let target_free = ((bucket_total as f64) * frac).round().max(1.0) as usize;
            let mut res_keys: Vec<usize> = res_obs_indices.keys().copied().collect();
            for i in (1..res_keys.len()).rev() {
                let j = rng.range_usize(0, i + 1);
                res_keys.swap(i, j);
            }
            let mut cur_free = 0usize;
            for rk in res_keys {
                let idxs = &res_obs_indices[&rk];
                let add = idxs.len();
                if bucket_total - (cur_free + add) < min_keep {
                    continue;
                }
                let cur_err = cur_free.abs_diff(target_free);
                let nxt_err = (cur_free + add).abs_diff(target_free);
                if cur_free == 0 || nxt_err <= cur_err {
                    for &ri in idxs {
                        tag.restraints[ri].is_free = true;
                    }
                    cur_free += add;
                    if cur_free >= target_free {
                        break;
                    }
                }
            }
        }
    }
    Ok(())
}

fn parse_chain_seq_token(
    raw: &str,
    kind_label: &str,
    line_no: usize,
) -> Result<(Option<String>, i32), String> {
    let t = raw.trim();
    if let Some((ch_part, seq_part)) = t.split_once(':') {
        let ch = ch_part.trim();
        if ch.is_empty() {
            return Err(format!(
                "{kind_label} line {line_no}: empty chain prefix in '{raw}'"
            ));
        }
        let sid: i32 = seq_part
            .trim()
            .parse()
            .map_err(|e| format!("{kind_label} line {line_no}: invalid seq_id in '{raw}': {e}"))?;
        Ok((Some(ch.to_string()), sid))
    } else {
        let sid: i32 = t
            .parse()
            .map_err(|e| format!("{kind_label} line {line_no}: invalid seq_id '{raw}': {e}"))?;
        Ok((None, sid))
    }
}

fn parse_anchor_spec(
    rest: &str,
    kind_label: &str,
    line_no: usize,
) -> Result<(Option<String>, i32, String), String> {
    let parts: Vec<&str> = rest.split_whitespace().collect();
    if parts.is_empty() {
        return Err(format!(
            "{kind_label} line {line_no}: empty anchor directive"
        ));
    }
    // Supports `# metal_anchor: 66 CB`, `# metal_anchor: 66:CB`, `# metal_anchor: A:66:CB`, `# metal_anchor: A:66 CB`, or `# metal_anchor: A 66 CB`
    if parts.len() == 1 && parts[0].contains(':') {
        let colon_parts: Vec<&str> = parts[0].split(':').collect();
        if colon_parts.len() == 3 {
            let sid: i32 = colon_parts[1].trim().parse().map_err(|e| {
                format!(
                    "{kind_label} line {line_no}: invalid anchor seq_id in '{}': {e}",
                    parts[0]
                )
            })?;
            return Ok((
                Some(colon_parts[0].trim().to_string()),
                sid,
                normalize_rdc_atom_name(colon_parts[2]).to_string(),
            ));
        } else if colon_parts.len() == 2 && colon_parts[0].trim().parse::<i32>().is_ok() {
            let sid: i32 = colon_parts[0].trim().parse().map_err(|e| {
                format!(
                    "{kind_label} line {line_no}: invalid anchor seq_id in '{}': {e}",
                    parts[0]
                )
            })?;
            return Ok((
                None,
                sid,
                normalize_rdc_atom_name(colon_parts[1]).to_string(),
            ));
        }
    }
    if parts.len() >= 2 && !parts[0].contains(':') && parts[0].parse::<i32>().is_err() {
        let ch = Some(parts[0].trim().to_string());
        let sid: i32 = parts[1].parse().map_err(|e| {
            format!(
                "{kind_label} line {line_no}: invalid anchor seq_id '{}': {e}",
                parts[1]
            )
        })?;
        let aname = parts
            .get(2)
            .map(|s| normalize_rdc_atom_name(s).to_string())
            .unwrap_or_else(|| "CB".to_string());
        Ok((ch, sid, aname))
    } else {
        let (ch, sid) = parse_chain_seq_token(parts[0], kind_label, line_no)?;
        let aname = parts
            .get(1)
            .map(|s| normalize_rdc_atom_name(s).to_string())
            .unwrap_or_else(|| "CB".to_string());
        Ok((ch, sid, aname))
    }
}

/// Read chain-aware multi-domain `RDC` restraints from an ASCII/UTF-8 TSV file (`<= 64 MiB`).
pub fn read_domain_rdc_tsv(path: impl AsRef<Path>) -> Result<Vec<DomainRdcMediumSet>, String> {
    let p = path.as_ref();
    let meta =
        fs::metadata(p).map_err(|e| format!("failed to stat RDC TSV '{}': {e}", p.display()))?;
    if meta.len() > MAX_DOMAIN_RESTRAINT_BYTES as u64 {
        return Err(format!(
            "RDC TSV '{}' exceeds MAX_DOMAIN_RESTRAINT_BYTES ({} B)",
            p.display(),
            MAX_DOMAIN_RESTRAINT_BYTES
        ));
    }
    let bytes =
        fs::read(p).map_err(|e| format!("failed to read RDC TSV '{}': {e}", p.display()))?;
    if bytes.len() > MAX_DOMAIN_RESTRAINT_BYTES || bytes.contains(&0u8) {
        return Err(format!(
            "RDC TSV '{}' exceeds size cap or contains NUL bytes",
            p.display()
        ));
    }
    let text = std::str::from_utf8(&bytes)
        .map_err(|e| format!("invalid UTF-8 in RDC TSV '{}': {e}", p.display()))?;
    parse_domain_rdc_tsv_str(text)
}

/// Parse single-chain or multi-chain `RDC` TSV text into `Vec<DomainRdcMediumSet>`.
///
/// Accepts standard 11/12-column `nmr_core::rdc` TSVs (including `A:12` `chain:seq_id` tokens)
/// as well as explicit `chain_id` / `(chain_id_1, chain_id_2)` columns.
pub fn parse_domain_rdc_tsv_str(text: &str) -> Result<Vec<DomainRdcMediumSet>, String> {
    if text.len() > MAX_DOMAIN_RESTRAINT_BYTES || text.as_bytes().contains(&0u8) {
        return Err("RDC TSV text exceeds size cap or contains NUL bytes".to_string());
    }
    let mut sign_conv = crate::rdc::RdcSignConvention::SignNormalized;
    let mut header_cols: Option<Vec<String>> = None;
    let mut media_order: Vec<String> = Vec::new();
    let mut media_rows: BTreeMap<String, Vec<DomainRdcRestraint>> = BTreeMap::new();

    for (line_idx, raw_line) in text.lines().enumerate() {
        let line_no = line_idx + 1;
        let trimmed = raw_line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(comment) = trimmed.strip_prefix('#') {
            if let Some((_, rest)) = parse_directive_key_and_tag(comment.trim(), "sign_convention")
            {
                sign_conv = crate::rdc::RdcSignConvention::parse_label(rest)?;
            }
            continue;
        }
        let cols: Vec<&str> = if trimmed.contains('\t') {
            trimmed.split('\t').map(str::trim).collect()
        } else {
            trimmed.split_whitespace().collect()
        };
        let low0 = cols[0].to_ascii_lowercase();
        if matches!(
            low0.as_str(),
            "medium_id" | "medium" | "chain_id" | "chain_1" | "seq_id_1"
        ) && cols
            .iter()
            .any(|c| c.eq_ignore_ascii_case("d_obs_hz") || c.eq_ignore_ascii_case("rdc_hz"))
        {
            header_cols = Some(cols.iter().map(|c| c.to_ascii_lowercase()).collect());
            continue;
        }

        let find_col = |names: &[&str]| -> Option<usize> {
            let hc = header_cols.as_ref()?;
            hc.iter().position(|c| names.iter().any(|n| c == *n))
        };

        let (
            med_id,
            ch1_col,
            s1_raw,
            rn1,
            at1,
            ch2_col,
            s2_raw,
            rn2,
            at2,
            bt_opt,
            d_obs_raw,
            sig_opt,
            dmax_opt,
            free_opt,
        ) = if header_cols.is_some() {
            let m_i = find_col(&["medium_id", "medium"]);
            let c1_i = find_col(&["chain_id_1", "chain_1", "chain_id", "chain"]);
            let c2_i = find_col(&["chain_id_2", "chain_2"]).or(c1_i);
            let s1_i = find_col(&["seq_id_1", "resid_1", "seq_1"])
                .ok_or_else(|| format!("RDC line {line_no}: missing seq_id_1 column"))?;
            let r1_i = find_col(&["res_name_1", "res_1"]);
            let a1_i = find_col(&["atom_name_1", "atom_1"])
                .ok_or_else(|| format!("RDC line {line_no}: missing atom_name_1 column"))?;
            let s2_i = find_col(&["seq_id_2", "resid_2", "seq_2"]).unwrap_or(s1_i);
            let r2_i = find_col(&["res_name_2", "res_2"]).or(r1_i);
            let a2_i = find_col(&["atom_name_2", "atom_2"])
                .ok_or_else(|| format!("RDC line {line_no}: missing atom_name_2 column"))?;
            let bt_i = find_col(&["bond_type", "type"]);
            let do_i = find_col(&["d_obs_hz", "rdc_hz", "d_hz"])
                .ok_or_else(|| format!("RDC line {line_no}: missing d_obs_hz column"))?;
            let sg_i = find_col(&["sigma_hz", "err_hz", "sigma"]);
            let dm_i = find_col(&["d_max_hz", "dmax_hz", "dmax"]);
            let fr_i = find_col(&["is_free", "free"]);

            let get_c = |idx: usize| -> Result<&str, String> {
                cols.get(idx).copied().ok_or_else(|| {
                    format!(
                        "RDC line {line_no}: row has fewer columns ({}) than header",
                        cols.len()
                    )
                })
            };
            (
                m_i.and_then(|i| cols.get(i).copied()).unwrap_or("medium_1"),
                c1_i.and_then(|i| cols.get(i).copied()),
                get_c(s1_i)?,
                r1_i.and_then(|i| cols.get(i).copied()).unwrap_or("."),
                get_c(a1_i)?,
                c2_i.and_then(|i| cols.get(i).copied()),
                get_c(s2_i)?,
                r2_i.and_then(|i| cols.get(i).copied()).unwrap_or("."),
                get_c(a2_i)?,
                bt_i.and_then(|i| cols.get(i).copied()),
                get_c(do_i)?,
                sg_i.and_then(|i| cols.get(i).copied()),
                dm_i.and_then(|i| cols.get(i).copied()),
                fr_i.and_then(|i| cols.get(i).copied()),
            )
        } else if cols.len() >= 11 && parse_chain_seq_token(cols[1], "RDC", line_no).is_ok() {
            // Canonical 11/12-column schema: medium_id seq_1 res_1 atom_1 seq_2 res_2 atom_2 bond_type d_obs sigma d_max [is_free]
            (
                cols[0],
                None,
                cols[1],
                cols[2],
                cols[3],
                None,
                cols[4],
                cols[5],
                cols[6],
                Some(cols[7]),
                cols[8],
                Some(cols[9]),
                Some(cols[10]),
                cols.get(11).copied(),
            )
        } else {
            // Delegate non-standard headerless single-chain schemas to `nmr_core::rdc::parse_rdc_tsv_str`
            let single = crate::rdc::parse_rdc_tsv_str_with_convention(text, sign_conv)?;
            return Ok(DomainRdcMediumSet::from_rdc_media(&single));
        };

        let (tok_c1, seq_id_1) = parse_chain_seq_token(s1_raw, "RDC", line_no)?;
        let (tok_c2, seq_id_2) = parse_chain_seq_token(s2_raw, "RDC", line_no)?;
        let chain_id_1 = tok_c1.or_else(|| {
            ch1_col
                .map(str::trim)
                .filter(|s| !s.is_empty() && *s != ".")
                .map(str::to_string)
        });
        let chain_id_2 = tok_c2
            .or_else(|| {
                ch2_col
                    .map(str::trim)
                    .filter(|s| !s.is_empty() && *s != ".")
                    .map(str::to_string)
            })
            .or_else(|| chain_id_1.clone());

        let atom_name_1 = normalize_rdc_atom_name(at1).to_string();
        let atom_name_2 = normalize_rdc_atom_name(at2).to_string();
        let bond_type = match bt_opt.filter(|s| !s.is_empty() && *s != ".") {
            Some(bt_s) => RdcBondType::parse_label(bt_s)?,
            None => RdcBondType::infer_from_atoms(seq_id_1, &atom_name_1, seq_id_2, &atom_name_2)?,
        };
        let d_obs_hz = parse_finite_f64(d_obs_raw, "d_obs_hz", line_no)?;
        let sigma_hz = parse_pos_sigma_opt(sig_opt, bond_type.default_sigma_hz(), line_no)?;
        let d_max_hz = match dmax_opt.filter(|s| !s.is_empty() && *s != ".") {
            Some(dm_s) => parse_finite_f64(dm_s, "d_max_hz", line_no)?,
            None => bond_type.signed_d_max_hz(sign_conv),
        };
        let is_free = parse_free_flag_opt(free_opt);

        let r_item = DomainRdcRestraint {
            medium_id: med_id.to_string(),
            chain_id_1,
            seq_id_1,
            res_name_1: rn1.to_string(),
            atom_name_1,
            chain_id_2,
            seq_id_2,
            res_name_2: rn2.to_string(),
            atom_name_2,
            bond_type,
            d_obs_hz,
            sigma_hz,
            d_max_hz,
            is_free,
        };
        r_item.validate()?;
        if !media_rows.contains_key(med_id) {
            media_order.push(med_id.to_string());
        }
        media_rows
            .entry(med_id.to_string())
            .or_default()
            .push(r_item);
    }

    if media_order.is_empty() {
        return Err("RDC TSV contains zero restraints".to_string());
    }
    Ok(media_order
        .into_iter()
        .map(|mid| DomainRdcMediumSet {
            restraints: media_rows.remove(&mid).unwrap_or_default(),
            medium_id: mid,
        })
        .collect())
}

/// Read multi-domain `PCS` restraints from an ASCII/UTF-8 TSV file (`<= 64 MiB`).
pub fn read_domain_pcs_tsv(path: impl AsRef<Path>) -> Result<Vec<DomainPcsTagSet>, String> {
    let p = path.as_ref();
    let meta =
        fs::metadata(p).map_err(|e| format!("failed to stat PCS TSV '{}': {e}", p.display()))?;
    if meta.len() > MAX_DOMAIN_RESTRAINT_BYTES as u64 {
        return Err(format!(
            "PCS TSV '{}' exceeds MAX_DOMAIN_RESTRAINT_BYTES ({} B)",
            p.display(),
            MAX_DOMAIN_RESTRAINT_BYTES
        ));
    }
    let bytes =
        fs::read(p).map_err(|e| format!("failed to read PCS TSV '{}': {e}", p.display()))?;
    if bytes.len() > MAX_DOMAIN_RESTRAINT_BYTES || bytes.contains(&0u8) {
        return Err(format!(
            "PCS TSV '{}' exceeds size cap or contains NUL bytes",
            p.display()
        ));
    }
    let text = std::str::from_utf8(&bytes)
        .map_err(|e| format!("invalid UTF-8 in PCS TSV '{}': {e}", p.display()))?;
    parse_domain_pcs_tsv_str(text)
}

/// Parse multi-domain `PCS` restraints from TSV text.
///
/// Header directives:
/// - `# metal_pos: <x> <y> <z>` or `# metal_pos[<tag_id>]: <x> <y> <z>`
/// - `# metal_domain: <domain_id>` or `# metal_domain[<tag_id>]: <domain_id>`
/// - `# metal_anchor: [chain_id] <seq_id> [atom_name]` (or `A:66 CB`)
/// - `# refine_metal_pos: true | false`
/// - `# metal_search_radius_a: <f64>`
///
/// Supported row schemas:
/// - `seq_id  pcs_ppm  [sigma_ppm]  [is_free]` (with `seq_id` or `chain:seq_id`)
/// - `seq_id  atom_name  pcs_ppm  [sigma_ppm]  [is_free]`
/// - `chain_id  seq_id  atom_name  pcs_ppm  [sigma_ppm]  [is_free]`
/// - `tag_id  [chain_id]  seq_id  atom_name  pcs_ppm  [sigma_ppm]  [is_free]`
pub fn parse_domain_pcs_tsv_str(text: &str) -> Result<Vec<DomainPcsTagSet>, String> {
    if text.len() > MAX_DOMAIN_RESTRAINT_BYTES || text.as_bytes().contains(&0u8) {
        return Err("PCS TSV text exceeds size cap or contains NUL bytes".to_string());
    }

    let mut metal_pos_map: BTreeMap<String, Vec3> = BTreeMap::new();
    let mut host_dom_map: BTreeMap<String, usize> = BTreeMap::new();
    let mut anchor_map: BTreeMap<String, (Option<String>, i32, String)> = BTreeMap::new();
    let mut refine_map: BTreeMap<String, bool> = BTreeMap::new();
    let mut radius_map: BTreeMap<String, f64> = BTreeMap::new();
    let mut has_tag_col = false;
    let mut has_chain_col = false;
    let mut tag_order: Vec<String> = Vec::new();
    let mut tag_rows: BTreeMap<String, Vec<DomainPcsRestraint>> = BTreeMap::new();

    for (line_idx, raw_line) in text.lines().enumerate() {
        let line_no = line_idx + 1;
        let trimmed = raw_line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(comment) = trimmed.strip_prefix('#') {
            let c = comment.trim();
            if let Some((k_tag, rest)) = parse_directive_key_and_tag(c, "metal_pos") {
                let nums: Vec<&str> = rest
                    .split([',', ' ', '\t'])
                    .filter(|s| !s.is_empty())
                    .collect();
                if nums.len() != 3 {
                    return Err(format!(
                        "PCS line {line_no}: # metal_pos expects 3 coordinates, got '{}'",
                        rest
                    ));
                }
                let x = parse_finite_f64(nums[0], "metal_x", line_no)?;
                let y = parse_finite_f64(nums[1], "metal_y", line_no)?;
                let z = parse_finite_f64(nums[2], "metal_z", line_no)?;
                metal_pos_map.insert(k_tag, [x, y, z]);
            } else if let Some((k_tag, rest)) = parse_directive_key_and_tag(c, "metal_domain") {
                let d_id: usize = rest.trim().parse().map_err(|e| {
                    format!("PCS line {line_no}: invalid # metal_domain '{rest}': {e}")
                })?;
                if d_id == 0 {
                    return Err(format!(
                        "PCS line {line_no}: # metal_domain must be >= 1 (1-based)"
                    ));
                }
                host_dom_map.insert(k_tag, d_id);
            } else if let Some((k_tag, rest)) = parse_directive_key_and_tag(c, "metal_anchor") {
                let (ach, sid, aname) = parse_anchor_spec(rest, "PCS", line_no)?;
                anchor_map.insert(k_tag, (ach, sid, aname));
            } else if let Some((k_tag, rest)) = parse_directive_key_and_tag(c, "refine_metal_pos") {
                refine_map.insert(k_tag, parse_free_flag_opt(Some(rest)));
            } else if let Some((k_tag, rest)) =
                parse_directive_key_and_tag(c, "metal_search_radius_a")
            {
                let rv = parse_finite_f64(rest, "metal_search_radius_a", line_no)?;
                if rv < 0.0 {
                    return Err(format!(
                        "PCS line {line_no}: # metal_search_radius_a ({rv}) must be >= 0"
                    ));
                }
                radius_map.insert(k_tag, rv);
            }
            continue;
        }

        let cols: Vec<&str> = if trimmed.contains('\t') {
            trimmed.split('\t').map(str::trim).collect()
        } else {
            trimmed.split_whitespace().collect()
        };
        let low0 = cols[0].to_ascii_lowercase();
        if matches!(low0.as_str(), "tag_id" | "site_id") {
            has_tag_col = true;
            has_chain_col = cols
                .get(1)
                .is_some_and(|c| matches!(c.to_ascii_lowercase().as_str(), "chain_id" | "chain"));
            continue;
        }
        if matches!(low0.as_str(), "chain_id" | "chain") {
            has_chain_col = true;
            continue;
        }
        if matches!(low0.as_str(), "seq_id" | "resid" | "residue" | "seq") {
            continue;
        }

        let (tag_id, rem0): (String, &[&str]) = if has_tag_col {
            if cols.len() < 4 {
                return Err(format!(
                    "PCS line {line_no}: expected at least 4 columns with tag_id, got {}",
                    cols.len()
                ));
            }
            (cols[0].to_string(), &cols[1..])
        } else {
            ("tag1".to_string(), &cols[..])
        };

        let (col_chain_opt, rem): (Option<String>, &[&str]) = if has_chain_col {
            if rem0.len() < 3 {
                return Err(format!(
                    "PCS line {line_no}: expected chain_id, seq_id, and pcs_ppm"
                ));
            }
            let ch = Some(rem0[0].to_string()).filter(|s| !s.is_empty() && s != ".");
            (ch, &rem0[1..])
        } else {
            (None, rem0)
        };

        if rem.len() < 2 {
            return Err(format!(
                "PCS line {line_no}: expected at least seq_id and pcs_ppm"
            ));
        }
        let (tok_chain, seq_id) = parse_chain_seq_token(rem[0], "PCS", line_no)?;
        let chain_id = tok_chain.or(col_chain_opt);

        let (atom_name, pcs_obs_ppm, sigma_ppm, is_free) = if rem[1].parse::<f64>().is_ok() {
            let pcs = parse_finite_f64(rem[1], "pcs_ppm", line_no)?;
            let sig = parse_pos_sigma_opt(rem.get(2).copied(), 0.05, line_no)?;
            let fr = parse_free_flag_opt(rem.get(3).copied());
            ("H".to_string(), pcs, sig, fr)
        } else if rem.len() >= 3 && rem[2].parse::<f64>().is_ok() {
            let at = normalize_rdc_atom_name(rem[1]).to_string();
            let pcs = parse_finite_f64(rem[2], "pcs_ppm", line_no)?;
            let sig = parse_pos_sigma_opt(rem.get(3).copied(), 0.05, line_no)?;
            let fr = parse_free_flag_opt(rem.get(4).copied());
            (at, pcs, sig, fr)
        } else if rem.len() >= 4 {
            let at = normalize_rdc_atom_name(rem[2]).to_string();
            let pcs = parse_finite_f64(rem[3], "pcs_ppm", line_no)?;
            let sig = parse_pos_sigma_opt(rem.get(4).copied(), 0.05, line_no)?;
            let fr = parse_free_flag_opt(rem.get(5).copied());
            (at, pcs, sig, fr)
        } else {
            return Err(format!(
                "PCS line {line_no}: unable to parse PCS row '{trimmed}'"
            ));
        };

        if !tag_rows.contains_key(&tag_id) {
            tag_order.push(tag_id.clone());
        }
        tag_rows
            .entry(tag_id.clone())
            .or_default()
            .push(DomainPcsRestraint {
                tag_id,
                chain_id,
                seq_id,
                atom_name,
                pcs_obs_ppm,
                sigma_ppm,
                is_free,
            });
    }

    if tag_order.is_empty() {
        return Err("PCS TSV contains zero observations".to_string());
    }

    let mut out = Vec::with_capacity(tag_order.len());
    for tid in tag_order {
        let restraints = tag_rows.remove(&tid).unwrap_or_default();
        let metal_pos_initial = metal_pos_map
            .get(&tid)
            .or_else(|| metal_pos_map.get(""))
            .copied();
        let host_domain_id = host_dom_map
            .get(&tid)
            .or_else(|| host_dom_map.get(""))
            .copied();
        let anch_full = anchor_map.get(&tid).or_else(|| anchor_map.get("")).cloned();
        let (anchor_chain_id, anchor_residue_atom) = match anch_full {
            Some((ach, sid, an)) => (ach, Some((sid, an))),
            None => (None, None),
        };
        let refine_metal_pos = refine_map
            .get(&tid)
            .or_else(|| refine_map.get(""))
            .copied()
            .unwrap_or(false);
        let metal_search_radius_a = radius_map.get(&tid).or_else(|| radius_map.get("")).copied();
        out.push(DomainPcsTagSet {
            tag_id: tid,
            metal_pos_initial,
            anchor_chain_id,
            anchor_residue_atom,
            host_domain_id,
            pre_fitted_tensor: None,
            refine_metal_pos,
            metal_search_radius_a,
            restraints,
        });
    }
    Ok(out)
}

/// Read multi-domain `PRE` restraints from an ASCII/UTF-8 TSV file (`<= 64 MiB`).
pub fn read_domain_pre_tsv(path: impl AsRef<Path>) -> Result<Vec<DomainPreSiteSet>, String> {
    read_domain_pre_tsv_with_defaults(path, None, None, None)
}

/// Read multi-domain `PRE` restraints with optional CLI-supplied `tau_c_ns`, `proton_mhz`, and
/// `mtsl_tether_allowance_a` defaults.
pub fn read_domain_pre_tsv_with_defaults(
    path: impl AsRef<Path>,
    cli_tau_c_ns: Option<f64>,
    cli_proton_mhz: Option<f64>,
    cli_mtsl_tether_a: Option<f64>,
) -> Result<Vec<DomainPreSiteSet>, String> {
    let p = path.as_ref();
    let meta =
        fs::metadata(p).map_err(|e| format!("failed to stat PRE TSV '{}': {e}", p.display()))?;
    if meta.len() > MAX_DOMAIN_RESTRAINT_BYTES as u64 {
        return Err(format!(
            "PRE TSV '{}' exceeds MAX_DOMAIN_RESTRAINT_BYTES ({} B)",
            p.display(),
            MAX_DOMAIN_RESTRAINT_BYTES
        ));
    }
    let bytes =
        fs::read(p).map_err(|e| format!("failed to read PRE TSV '{}': {e}", p.display()))?;
    if bytes.len() > MAX_DOMAIN_RESTRAINT_BYTES || bytes.contains(&0u8) {
        return Err(format!(
            "PRE TSV '{}' exceeds size cap or contains NUL bytes",
            p.display()
        ));
    }
    let text = std::str::from_utf8(&bytes)
        .map_err(|e| format!("invalid UTF-8 in PRE TSV '{}': {e}", p.display()))?;
    parse_domain_pre_tsv_str_with_defaults(text, cli_tau_c_ns, cli_proton_mhz, cli_mtsl_tether_a)
}

/// Parse multi-domain Solomon `PRE` restraints from TSV text (`fail closed` if `tau_c_ns` or
/// `proton_mhz` is omitted for `gamma2` rates, `21.W3:P0-4`).
///
/// Header directives:
/// - `# site_pos: <x> <y> <z>` or `# site_pos[<site_id>]: <x> <y> <z>`
/// - `# site_anchor: [chain_id] <seq_id> [atom_name]` (or `A:68 CB`)
/// - `# site_domain: <domain_id>`
/// - `# tau_c_ns: <f64>` (**REQUIRED** for `gamma2` mode unless passed via CLI `--pre-tau-c-ns`)
/// - `# proton_mhz: <f64>` (**REQUIRED** for `gamma2` mode unless passed via CLI `--pre-proton-mhz`)
/// - `# spin_s: <f64>` (optional electronic spin $S$, default `0.5` for nitroxide MTSL)
/// - `# g_factor: <f64>` (optional electronic $g$, default CODATA `g_e = 2.0023193`)
/// - `# mtsl_tether_allowance_a: <f64>` (optional flexible-linker allowance in Å, e.g. `6.0`)
/// - `# mode: gamma2 | bounds`
pub fn parse_domain_pre_tsv_str(text: &str) -> Result<Vec<DomainPreSiteSet>, String> {
    parse_domain_pre_tsv_str_with_defaults(text, None, None, None)
}

/// Parse multi-domain Solomon `PRE` restraints with optional CLI-level `tau_c_ns`, `proton_mhz`,
/// and `mtsl_tether_allowance_a` fallbacks (`0` silent `1.23e7` magic default!).
pub fn parse_domain_pre_tsv_str_with_defaults(
    text: &str,
    cli_tau_c_ns: Option<f64>,
    cli_proton_mhz: Option<f64>,
    cli_mtsl_tether_a: Option<f64>,
) -> Result<Vec<DomainPreSiteSet>, String> {
    if text.len() > MAX_DOMAIN_RESTRAINT_BYTES || text.as_bytes().contains(&0u8) {
        return Err("PRE TSV text exceeds size cap or contains NUL bytes".to_string());
    }

    let mut site_pos_map: BTreeMap<String, Vec3> = BTreeMap::new();
    let mut site_anchor_map: BTreeMap<String, (Option<String>, i32, String)> = BTreeMap::new();
    let mut site_dom_map: BTreeMap<String, usize> = BTreeMap::new();
    let mut tau_c_map: BTreeMap<String, f64> = BTreeMap::new();
    let mut mhz_map: BTreeMap<String, f64> = BTreeMap::new();
    let mut tether_map: BTreeMap<String, f64> = BTreeMap::new();
    let mut spin_s_map: BTreeMap<String, f64> = BTreeMap::new();
    let mut g_factor_map: BTreeMap<String, f64> = BTreeMap::new();
    let mut saw_legacy_k_pre = false;
    let mut bounds_mode = false;
    let mut has_site_col = false;
    let mut has_chain_col = false;
    let mut site_order: Vec<String> = Vec::new();
    let mut site_rows: BTreeMap<String, Vec<DomainPreRestraint>> = BTreeMap::new();

    for (line_idx, raw_line) in text.lines().enumerate() {
        let line_no = line_idx + 1;
        let trimmed = raw_line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(comment) = trimmed.strip_prefix('#') {
            let c = comment.trim();
            if let Some((k_site, rest)) = parse_directive_key_and_tag(c, "site_pos") {
                let nums: Vec<&str> = rest
                    .split([',', ' ', '\t'])
                    .filter(|s| !s.is_empty())
                    .collect();
                if nums.len() != 3 {
                    return Err(format!(
                        "PRE line {line_no}: # site_pos expects 3 coordinates, got '{rest}'"
                    ));
                }
                let x = parse_finite_f64(nums[0], "site_x", line_no)?;
                let y = parse_finite_f64(nums[1], "site_y", line_no)?;
                let z = parse_finite_f64(nums[2], "site_z", line_no)?;
                site_pos_map.insert(k_site, [x, y, z]);
            } else if let Some((k_site, rest)) = parse_directive_key_and_tag(c, "site_anchor") {
                let (ach, sid, aname) = parse_anchor_spec(rest, "PRE", line_no)?;
                site_anchor_map.insert(k_site, (ach, sid, aname));
            } else if let Some((k_site, rest)) = parse_directive_key_and_tag(c, "site_domain") {
                let d_id: usize = rest.trim().parse().map_err(|e| {
                    format!("PRE line {line_no}: invalid # site_domain '{rest}': {e}")
                })?;
                if d_id == 0 {
                    return Err(format!(
                        "PRE line {line_no}: # site_domain must be >= 1 (1-based)"
                    ));
                }
                site_dom_map.insert(k_site, d_id);
            } else if let Some((k_site, rest)) = parse_directive_key_and_tag(c, "tau_c_ns")
                .or_else(|| parse_directive_key_and_tag(c, "tau_c"))
            {
                let tc = parse_finite_f64(rest.trim(), "tau_c_ns", line_no)?;
                if tc <= 0.0 {
                    return Err(format!("PRE line {line_no}: # tau_c_ns ({tc}) must be > 0"));
                }
                tau_c_map.insert(k_site, tc);
            } else if let Some((k_site, rest)) = parse_directive_key_and_tag(c, "proton_mhz")
                .or_else(|| parse_directive_key_and_tag(c, "field_mhz"))
                .or_else(|| parse_directive_key_and_tag(c, "omega_h_mhz"))
            {
                let mhz = parse_finite_f64(rest.trim(), "proton_mhz", line_no)?;
                if mhz <= 0.0 {
                    return Err(format!(
                        "PRE line {line_no}: # proton_mhz ({mhz}) must be > 0"
                    ));
                }
                mhz_map.insert(k_site, mhz);
            } else if let Some((k_site, rest)) = parse_directive_key_and_tag(c, "spin_s") {
                let s_val = parse_finite_f64(rest.trim(), "spin_s", line_no)?;
                if s_val <= 0.0 {
                    return Err(format!("PRE line {line_no}: # spin_s must be > 0"));
                }
                spin_s_map.insert(k_site, s_val);
            } else if let Some((k_site, rest)) = parse_directive_key_and_tag(c, "g_factor") {
                let g_val = parse_finite_f64(rest.trim(), "g_factor", line_no)?;
                if g_val <= 0.0 {
                    return Err(format!("PRE line {line_no}: # g_factor must be > 0"));
                }
                g_factor_map.insert(k_site, g_val);
            } else if let Some((k_site, rest)) =
                parse_directive_key_and_tag(c, "mtsl_tether_allowance_a")
                    .or_else(|| parse_directive_key_and_tag(c, "tether_allowance_a"))
            {
                let tv = parse_finite_f64(rest.trim(), "mtsl_tether_allowance_a", line_no)?;
                if tv < 0.0 {
                    return Err(format!(
                        "PRE line {line_no}: # mtsl_tether_allowance_a ({tv}) must be >= 0"
                    ));
                }
                tether_map.insert(k_site, tv);
            } else if parse_directive_key_and_tag(c, "k_pre").is_some() {
                saw_legacy_k_pre = true;
            } else if let Some((_, rest)) = parse_directive_key_and_tag(c, "mode") {
                bounds_mode = rest.trim().eq_ignore_ascii_case("bounds");
            }
            continue;
        }

        let cols: Vec<&str> = if trimmed.contains('\t') {
            trimmed.split('\t').map(str::trim).collect()
        } else {
            trimmed.split_whitespace().collect()
        };
        let low0 = cols[0].to_ascii_lowercase();
        if matches!(low0.as_str(), "site_id" | "tag_id") {
            has_site_col = true;
            has_chain_col = cols
                .get(1)
                .is_some_and(|c| matches!(c.to_ascii_lowercase().as_str(), "chain_id" | "chain"));
            if cols.iter().any(|c| c.eq_ignore_ascii_case("lower_a")) {
                bounds_mode = true;
            }
            continue;
        }
        if matches!(low0.as_str(), "chain_id" | "chain") {
            has_chain_col = true;
            if cols.iter().any(|c| c.eq_ignore_ascii_case("lower_a")) {
                bounds_mode = true;
            }
            continue;
        }
        if matches!(low0.as_str(), "seq_id" | "resid" | "residue") {
            if cols.iter().any(|c| c.eq_ignore_ascii_case("lower_a")) {
                bounds_mode = true;
            }
            continue;
        }

        let (site_id, rem0): (String, &[&str]) = if has_site_col {
            if cols.len() < 5 {
                return Err(format!(
                    "PRE line {line_no}: expected at least 5 columns with site_id, got {}",
                    cols.len()
                ));
            }
            (cols[0].to_string(), &cols[1..])
        } else {
            ("pre1".to_string(), &cols[..])
        };

        let (col_chain_opt, rem): (Option<String>, &[&str]) = if has_chain_col {
            if rem0.len() < 5 {
                return Err(format!(
                    "PRE line {line_no}: expected chain_id, seq_id, atom_name, and 2 numeric columns"
                ));
            }
            let ch = Some(rem0[0].to_string()).filter(|s| !s.is_empty() && s != ".");
            (ch, &rem0[1..])
        } else {
            (None, rem0)
        };

        if rem.len() < 4 {
            return Err(format!(
                "PRE line {line_no}: expected seq_id, atom_name, and 2 numeric columns"
            ));
        }
        let (tok_chain, seq_id) = parse_chain_seq_token(rem[0], "PRE", line_no)?;
        let chain_id = tok_chain.or(col_chain_opt);
        let atom_name = normalize_rdc_atom_name(rem[1]).to_string();

        let eff_tau_c = tau_c_map
            .get(&site_id)
            .or_else(|| tau_c_map.get(""))
            .copied()
            .or(cli_tau_c_ns);
        let eff_mhz = mhz_map
            .get(&site_id)
            .or_else(|| mhz_map.get(""))
            .copied()
            .or(cli_proton_mhz);

        let kind = if bounds_mode {
            let lower_a = parse_finite_f64(rem[2], "lower_a", line_no)?;
            let upper_a = parse_finite_f64(rem[3], "upper_a", line_no)?;
            if lower_a <= 0.0 || upper_a < lower_a {
                return Err(format!(
                    "PRE line {line_no}: invalid distance bounds [{lower_a}, {upper_a}]"
                ));
            }
            let sigma_a = parse_pos_sigma_opt(rem.get(4).copied(), 1.0, line_no)?;
            PreRestraintKind::DistanceBounds {
                lower_a,
                upper_a,
                sigma_a,
            }
        } else {
            let gamma2_obs_s1 = parse_finite_f64(rem[2], "gamma2_s1", line_no)?;
            if gamma2_obs_s1 < 0.0 {
                return Err(format!(
                    "PRE line {line_no}: gamma2_s1 ({gamma2_obs_s1}) must be >= 0"
                ));
            }
            let sigma_s1 = parse_pos_sigma_opt(rem.get(3).copied(), 2.0, line_no)?;
            let tau_c_ns = eff_tau_c.ok_or_else(|| {
                if saw_legacy_k_pre {
                    format!(
                        "PRE line {line_no}: `# k_pre` without explicit rotational correlation time is retired; specify mandatory `# tau_c_ns: <ns>` and `# proton_mhz: <MHz>` (or `--pre-tau-c-ns` / `--pre-proton-mhz`)"
                    )
                } else {
                    format!(
                        "PRE line {line_no}: Solomon Gamma2 PRE restraints require mandatory `# tau_c_ns: <ns>` (or `--pre-tau-c-ns <ns>`) and `# proton_mhz: <MHz>` (or `--pre-proton-mhz <MHz>`)"
                    )
                }
            })?;
            let proton_mhz = eff_mhz.ok_or_else(|| {
                format!(
                    "PRE line {line_no}: Solomon Gamma2 PRE restraints require mandatory `# proton_mhz: <MHz>` (or `--pre-proton-mhz <MHz>`) alongside tau_c_ns={tau_c_ns}"
                )
            })?;
            let spin_s_eff = spin_s_map
                .get(&site_id)
                .or_else(|| spin_s_map.get(""))
                .copied()
                .unwrap_or(0.5);
            let g_factor_eff = g_factor_map
                .get(&site_id)
                .or_else(|| g_factor_map.get(""))
                .copied()
                .unwrap_or(CODATA_ELECTRON_G_FACTOR);
            let k_pre_a6_s1 =
                compute_solomon_k_pre_a6_s1(tau_c_ns, proton_mhz, spin_s_eff, g_factor_eff)?;
            PreRestraintKind::Gamma2Rate {
                gamma2_obs_s1,
                sigma_s1,
                k_pre_a6_s1,
            }
        };

        if !site_rows.contains_key(&site_id) {
            site_order.push(site_id.clone());
        }
        site_rows
            .entry(site_id.clone())
            .or_default()
            .push(DomainPreRestraint {
                site_id,
                chain_id,
                seq_id,
                atom_name,
                kind,
            });
    }

    if site_order.is_empty() {
        return Err("PRE TSV contains zero restraints".to_string());
    }

    let mut out = Vec::with_capacity(site_order.len());
    for sid in site_order {
        let restraints = site_rows.remove(&sid).unwrap_or_default();
        let site_pos_initial = site_pos_map
            .get(&sid)
            .or_else(|| site_pos_map.get(""))
            .copied();
        let anch_full = site_anchor_map
            .get(&sid)
            .or_else(|| site_anchor_map.get(""))
            .cloned();
        let (anchor_chain_id, anchor_residue_atom) = match anch_full {
            Some((ach, rsid, an)) => (ach, Some((rsid, an))),
            None => (None, None),
        };
        let host_domain_id = site_dom_map
            .get(&sid)
            .or_else(|| site_dom_map.get(""))
            .copied();
        let tau_c_ns = tau_c_map
            .get(&sid)
            .or_else(|| tau_c_map.get(""))
            .copied()
            .or(cli_tau_c_ns);
        let proton_mhz = mhz_map
            .get(&sid)
            .or_else(|| mhz_map.get(""))
            .copied()
            .or(cli_proton_mhz);
        let mtsl_tether_allowance_a = tether_map
            .get(&sid)
            .or_else(|| tether_map.get(""))
            .copied()
            .or(cli_mtsl_tether_a);
        out.push(DomainPreSiteSet {
            site_id: sid,
            site_pos_initial,
            anchor_chain_id,
            anchor_residue_atom,
            host_domain_id,
            tau_c_ns,
            proton_mhz,
            mtsl_tether_allowance_a,
            restraints,
        });
    }
    Ok(out)
}

/// Read chain-aware inter-domain / ambiguous `NOE` distance restraints (`(chain_id, seq_id)` keyed,
/// `21.W2:P1-8` & `P2-11`) from an ASCII/UTF-8 TSV file (`<= 64 MiB`).
pub fn read_domain_noe_tsv(
    path: impl AsRef<Path>,
    af_residues: &[AfResidueRecord],
) -> Result<Vec<AmbiguousDistanceRestraint>, String> {
    let p = path.as_ref();
    let meta =
        fs::metadata(p).map_err(|e| format!("failed to stat NOE TSV '{}': {e}", p.display()))?;
    if meta.len() > MAX_DOMAIN_RESTRAINT_BYTES as u64 {
        return Err(format!(
            "NOE TSV '{}' exceeds MAX_DOMAIN_RESTRAINT_BYTES ({} B)",
            p.display(),
            MAX_DOMAIN_RESTRAINT_BYTES
        ));
    }
    let bytes =
        fs::read(p).map_err(|e| format!("failed to read NOE TSV '{}': {e}", p.display()))?;
    if bytes.len() > MAX_DOMAIN_RESTRAINT_BYTES || bytes.contains(&0u8) {
        return Err(format!(
            "NOE TSV '{}' exceeds size cap or contains NUL bytes",
            p.display()
        ));
    }
    let text = std::str::from_utf8(&bytes)
        .map_err(|e| format!("invalid UTF-8 in NOE TSV '{}': {e}", p.display()))?;
    parse_domain_noe_tsv_str(text, af_residues)
}

/// Parse single-chain or multi-chain ambiguous/unambiguous `NOE` TSV text into
/// `Vec<AmbiguousDistanceRestraint>`, grouping rows sharing a `peak_id` into Nilges $r^{-6}$-summed
/// candidate sets (`21.W2:P1-8`, `P2-11`).
pub fn parse_domain_noe_tsv_str(
    text: &str,
    af_residues: &[AfResidueRecord],
) -> Result<Vec<AmbiguousDistanceRestraint>, String> {
    if text.len() > MAX_DOMAIN_RESTRAINT_BYTES || text.as_bytes().contains(&0u8) {
        return Err("NOE TSV text exceeds size cap or contains NUL bytes".to_string());
    }
    let chain_ids: Vec<String> = af_residues.iter().map(|r| r.chain_id.clone()).collect();
    let seq_ids: Vec<i32> = af_residues.iter().map(|r| r.seq_id).collect();
    let cseq = ChainSeqIndex::build(&chain_ids, &seq_ids)?;

    let parse_noe_site = |raw_at: &str| -> Option<NoeProton> {
        let up = raw_at.trim().to_ascii_uppercase();
        match up.as_str() {
            "H" | "HN" => Some(NoeProton::AmideH),
            "HA" | "HA#" | "QA" | "HA2" | "HA3" => Some(NoeProton::AlphaHA),
            "HB" | "HB#" | "QB" | "HB1" | "HB2" | "HB3" => Some(NoeProton::BetaQB),
            _ => None,
        }
    };

    let mut groups: BTreeMap<usize, (f64, f64, Vec<crate::noesy::NoeCandidatePair>)> =
        BTreeMap::new();
    let mut auto_peak_id = 1_000_000usize;
    let mut has_peak_col = false;
    let mut has_chain_cols = false;

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
        let low0 = cols[0].to_ascii_lowercase();
        if matches!(
            low0.as_str(),
            "peak_id"
                | "restraint_id"
                | "id"
                | "chain_1"
                | "chain_id_1"
                | "chain_a"
                | "seq_id_1"
                | "resid_1"
                | "res_1"
                | "res_a"
                | "seq_a"
        ) {
            has_peak_col = matches!(low0.as_str(), "peak_id" | "restraint_id" | "id");
            has_chain_cols = cols.iter().any(|c| {
                matches!(
                    c.to_ascii_lowercase().as_str(),
                    "chain_1" | "chain_id_1" | "chain_a"
                )
            });
            continue;
        }

        let (peak_id, rem) = if has_peak_col {
            let pid: usize = cols[0]
                .parse()
                .map_err(|e| format!("NOE line {line_no}: invalid peak_id '{}': {e}", cols[0]))?;
            (pid, &cols[1..])
        } else {
            let pid = auto_peak_id;
            auto_peak_id += 1;
            (pid, &cols[..])
        };

        let (c1_opt, s1, at1, c2_opt, s2, at2, lo_a, hi_a) = if has_chain_cols {
            if rem.len() < 8 {
                return Err(format!(
                    "NOE line {line_no}: expected chain_1 seq_1 atom_1 chain_2 seq_2 atom_2 lower_a upper_a"
                ));
            }
            let (_, s1) = parse_chain_seq_token(rem[1], "NOE", line_no)?;
            let (_, s2) = parse_chain_seq_token(rem[4], "NOE", line_no)?;
            let lo = parse_finite_f64(rem[6], "lower_a", line_no)?;
            let hi = parse_finite_f64(rem[7], "upper_a", line_no)?;
            (
                Some(rem[0].to_string()),
                s1,
                rem[2],
                Some(rem[3].to_string()),
                s2,
                rem[5],
                lo,
                hi,
            )
        } else if rem.len() >= 6
            && parse_noe_site(rem[1]).is_some()
            && parse_noe_site(rem[3]).is_some()
        {
            // `seq_1 atom_1 seq_2 atom_2 lower_a upper_a` (with optional `A:10` tokens)
            let (tc1, s1) = parse_chain_seq_token(rem[0], "NOE", line_no)?;
            let (tc2, s2) = parse_chain_seq_token(rem[2], "NOE", line_no)?;
            let lo = parse_finite_f64(rem[4], "lower_a", line_no)?;
            let hi = parse_finite_f64(rem[5], "upper_a", line_no)?;
            (tc1, s1, rem[1], tc2, s2, rem[3], lo, hi)
        } else if rem.len() >= 5
            && parse_noe_site(rem[1]).is_some()
            && parse_noe_site(rem[3]).is_some()
        {
            // `seq_1 atom_1 seq_2 atom_2 upper_a`
            let (tc1, s1) = parse_chain_seq_token(rem[0], "NOE", line_no)?;
            let (tc2, s2) = parse_chain_seq_token(rem[2], "NOE", line_no)?;
            let hi = parse_finite_f64(rem[4], "upper_a", line_no)?;
            (tc1, s1, rem[1], tc2, s2, rem[3], 1.8, hi)
        } else if rem.len() == 1
            && !has_peak_col
            && af_residues
                .iter()
                .map(|r| &r.chain_id)
                .collect::<BTreeSet<_>>()
                .len()
                == 1
        {
            return Err(format!(
                "NOE line {line_no}: unrecognized single-column NOE row '{trimmed}'"
            ));
        } else {
            // Fallback to `nmr_core::structure::parse_exp_noe_tsv_str` only when single-chain
            let uniq_chains: BTreeSet<&str> =
                af_residues.iter().map(|r| r.chain_id.as_str()).collect();
            if uniq_chains.len() > 1 {
                return Err(format!(
                    "NOE line {line_no}: multi-chain NOE TSV requires explicit chain:seq_id or chain_1/chain_2 columns"
                ));
            }
            let seq3: Vec<String> = af_residues.iter().map(|r| r.res_name.clone()).collect();
            return crate::structure::parse_exp_noe_tsv_str(text, &seq3, &seq_ids);
        };

        if lo_a <= 0.0 || hi_a < lo_a {
            return Err(format!(
                "NOE line {line_no}: invalid bounds [{lo_a}, {hi_a}]"
            ));
        }
        let (Some(site1), Some(site2)) = (parse_noe_site(at1), parse_noe_site(at2)) else {
            continue;
        };
        let (Some(res_a), Some(res_b)) = (
            cseq.resolve(c1_opt.as_deref(), s1)?,
            cseq.resolve(c2_opt.as_deref(), s2)?,
        ) else {
            continue;
        };
        let entry = groups
            .entry(peak_id)
            .or_insert_with(|| (lo_a, hi_a, Vec::new()));
        entry.2.push(crate::noesy::NoeCandidatePair {
            res_a,
            atom_a: site1,
            res_b,
            atom_b: site2,
            shift_score: 1.0,
            network_score: 1.0,
            weight: 1.0,
        });
    }

    let mut out = Vec::with_capacity(groups.len());
    for (pid, (lo, hi, cands)) in groups {
        if cands.is_empty() {
            continue;
        }
        let is_unambiguous = cands.len() == 1;
        out.push(AmbiguousDistanceRestraint {
            peak_index: pid,
            exp: "DOMAIN_NOE".to_string(),
            lower_bound: lo,
            upper_bound: hi,
            all_candidates: cands.clone(),
            candidates: cands,
            is_unambiguous,
        });
    }
    if out.is_empty() {
        return Err("NOE TSV contains zero resolved restraints".to_string());
    }
    Ok(out)
}

fn parse_directive_key_and_tag<'a>(comment: &'a str, key: &str) -> Option<(String, &'a str)> {
    let rest = comment.strip_prefix(key)?;
    if let Some(after_colon) = rest.strip_prefix(':') {
        return Some((String::new(), after_colon.trim()));
    }
    let after_lbrack = rest.strip_prefix('[')?;
    let (tag, rem) = after_lbrack.split_once(']')?;
    let after_colon = rem.trim_start().strip_prefix(':')?;
    Some((tag.trim().to_string(), after_colon.trim()))
}

fn parse_finite_f64(raw: &str, name: &str, line_no: usize) -> Result<f64, String> {
    let v: f64 = raw
        .trim()
        .parse()
        .map_err(|e| format!("line {line_no}: invalid {name} '{raw}': {e}"))?;
    if !v.is_finite() {
        return Err(format!("line {line_no}: non-finite {name} '{raw}'"));
    }
    Ok(v)
}

fn parse_pos_sigma_opt(
    raw_opt: Option<&str>,
    default_val: f64,
    line_no: usize,
) -> Result<f64, String> {
    let Some(raw) = raw_opt else {
        return Ok(default_val);
    };
    let t = raw.trim();
    if t.is_empty() || t == "." {
        return Ok(default_val);
    }
    let v = parse_finite_f64(t, "sigma", line_no)?;
    if v <= 0.0 {
        return Err(format!("line {line_no}: sigma ({v}) must be > 0"));
    }
    Ok(v)
}

fn parse_free_flag_opt(raw_opt: Option<&str>) -> bool {
    let Some(raw) = raw_opt else {
        return false;
    };
    matches!(
        raw.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "free" | "yes"
    )
}

// ============================================================================
// 4. Covalent Rigid-Domain + NeRF Linker Kinematics (`0.000 Å` Distortion)
//    & Analytical Multi-Domain Energy / Torque-Gradient Engine
// ============================================================================

/// Residue Ramachandran basin class (`Lovell et al. 2003`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DockResClass {
    General = 0,
    Gly = 1,
    Pro = 2,
}

/// Contiguous rigid-domain block $\mathcal{B}_b = [\text{start\_res}, \text{end\_res})$ (`0.000 Å` internal drift).
/// When a domain $d$ has multiple discontinuous sequence spans (e.g. an insertion-domain topology
/// $D_1 = [s_0..e_0) \cup [s_1..e_1)$), every span of $d$ shares the **exact same rigid-body frame**
/// `master_block_id` (`master_block_id == block_id` on the primary span, `master_block_id < block_id`
/// on re-entry spans; `21.W2:P0-4`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RigidBlockGeom {
    pub block_id: usize,
    pub master_block_id: usize,
    pub domain_id: usize,
    pub chain_idx: usize,
    pub start_res: usize,
    pub end_res: usize,
}

/// Per-residue covalent internal-coordinate and local rigid-frame geometry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResidueKinGeom {
    pub seq_id: i32,
    pub res_name: String,
    pub chain_id: String,
    pub chain_idx: usize,
    pub res_class: DockResClass,
    pub rigid_block_id: Option<usize>,
    pub master_block_id: Option<usize>,
    pub domain_id: usize,
    pub phi_active: bool,
    pub psi_active: bool,
    pub rama_active: bool,
    pub is_chain_start: bool,
    pub is_chain_end: bool,
    pub is_block_start: bool,
    pub is_reentry_start: bool,
    /// Bond length $r(C'_{i-1}, N_i)$ and interior angle $\angle(CA_{i-1} - C'_{i-1} - N_i)$ (rad).
    pub r_c_prev_n: f64,
    pub ang_ca_prev_c_prev_n: f64,
    /// Bond length $r(N_i, CA_i)$ and interior angle $\angle(C'_{i-1} - N_i - CA_i)$ (rad).
    pub r_n_ca: f64,
    pub ang_c_prev_n_ca: f64,
    /// Bond length $r(CA_i, C'_i)$ and interior angle $\angle(N_i - CA_i - C'_i)$ (rad).
    pub r_ca_c: f64,
    pub ang_n_ca_c: f64,
    /// Bond length $r(C'_i, O_i)$ and interior angle $\angle(CA_i - C'_i - O_i)$ (rad).
    pub r_c_o: f64,
    pub ang_ca_c_o: f64,
    /// Virtual $C'_{-1}$ reference coordinate in chain-root frame (`is_chain_start == true`).
    pub c_prev_virtual: Vec3,
    /// Input reference coordinates `(N, H, CA, HA/HA2, C, O, CB/HA3)` and presence flags (`true` if atom exists).
    pub ref_slots: [(Vec3, bool); 7],
    /// Local coordinates in rigid domain master block's orthonormal triad `(N_{s_0}, e_x, e_y, e_z)`.
    pub local_slots: [(Vec3, bool); 7],
    /// Additional heavy sidechain atoms `(atom_name, local_or_ref_xyz)` preserved rigidly for `.pdb`/`.cif` export.
    pub extra_heavy_local: Vec<(String, Vec3)>,
    pub plddt: Option<f64>,
}

/// Covalent multi-domain + flexible-linker (and multi-chain $\text{SE}(3)$) forward/backward kinematic model.
#[derive(Clone, Debug)]
pub struct DomainLinkerKinematics {
    pub residues: Vec<ResidueKinGeom>,
    pub rigid_blocks: Vec<RigidBlockGeom>,
    pub chain_ranges: Vec<(usize, usize)>,
    pub chain_ids: Vec<String>,
    pub chain_ref_centroids: Vec<Vec3>,
    /// Input AlphaFold reference torsions `(phi_rad, psi_rad, omega_rad)` across `0..L`.
    pub ref_phi_rad: Vec<f64>,
    pub ref_psi_rad: Vec<f64>,
    pub ref_omega_rad: Vec<f64>,
    /// Indices of active linker torsions (`0..L` where `phi_active || psi_active`).
    pub active_linker_residues: Vec<usize>,
    /// Reference input `PdbChain` per rigid domain for exact `0.000 Å` intra-domain RMSD verification.
    pub ref_chain: PdbChain,
}

/// Dynamical state vector $(\boldsymbol{\theta}_{\text{link}}, \{\mathbf{q}_c, \mathbf{t}_c\})$ of a multi-domain replica.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DomainDockState {
    pub phi_rad: Vec<f64>,
    pub psi_rad: Vec<f64>,
    pub omega_rad: Vec<f64>,
    /// Unit quaternion `[w, x, y, z]` per polymer chain (`chain 0` stays fixed at `[1, 0, 0, 0]`).
    pub chain_quats: Vec<[f64; 4]>,
    /// Translation `[tx, ty, tz]` (Å) per polymer chain (`chain 0` stays fixed at `[0, 0, 0]`).
    pub chain_trans: Vec<Vec3>,
}

/// Objective term weights for multi-domain docking.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DomainDockEnergyWeights {
    pub w_rdc: f64,
    pub w_pcs: f64,
    pub w_pre: f64,
    pub w_noe: f64,
    pub noe_soft_window: f64,
    pub w_vdw: f64,
    pub vdw_scale: f64,
    pub w_rama: f64,
}

impl Default for DomainDockEnergyWeights {
    fn default() -> Self {
        Self {
            w_rdc: 1.0,
            w_pcs: 1.0,
            w_pre: 1.0,
            w_noe: 2.5,
            noe_soft_window: 3.0,
            w_vdw: 1.5,
            vdw_scale: 0.92,
            w_rama: 0.45,
        }
    }
}

/// Full 7-component energy breakdown of a multi-domain docking state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DomainDockEnergyBreakdown {
    pub total: f64,
    pub rdc: f64,
    pub pcs: f64,
    pub pre: f64,
    pub noe: f64,
    pub vdw: f64,
    pub rama: f64,
}

/// Analytical torsion and multi-chain $\mathfrak{se}(3)$ gradients at a `DomainDockState`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DomainDockGradient {
    /// `dE / d(phi_i)` (`0.0` on frozen domain residues).
    pub grad_phi: Vec<f64>,
    /// `dE / d(psi_i)` (`0.0` on frozen domain residues).
    pub grad_psi: Vec<f64>,
    /// World-frame right-trivialized rotational gradient `dE / d(omega_c)` around `c_af + trans_c` (`[0;3]` on chain 0).
    pub grad_chain_rot: Vec<Vec3>,
    /// Translational gradient `dE / d(trans_c)` (`[0;3]` on chain 0).
    pub grad_chain_trans: Vec<Vec3>,
}

#[inline]
pub(crate) fn wrap_angle_rad(rad: f64) -> f64 {
    let mut r = (rad + PI) % (2.0 * PI);
    if r <= 0.0 {
        r += 2.0 * PI;
    }
    r - PI
}

#[inline]
pub fn quat_normalize(q: [f64; 4]) -> [f64; 4] {
    let n2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    if n2 <= 1e-30 || !n2.is_finite() {
        [1.0, 0.0, 0.0, 0.0]
    } else {
        let inv = 1.0 / n2.sqrt();
        let s = if q[0] < 0.0 { -inv } else { inv };
        [q[0] * s, q[1] * s, q[2] * s, q[3] * s]
    }
}

/// Proper rotation matrix $\mathbf{R}(\mathbf{q}) \in \text{SO}(3)$ for unit quaternion `[w, x, y, z]`.
#[inline]
pub fn quat_to_rot_matrix(q_raw: [f64; 4]) -> [[f64; 3]; 3] {
    let [w, x, y, z] = quat_normalize(q_raw);
    let xx = x * x;
    let yy = y * y;
    let zz = z * z;
    let xy = x * y;
    let xz = x * z;
    let yz = y * z;
    let wx = w * x;
    let wy = w * y;
    let wz = w * z;
    [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ]
}

/// Convert a proper `3x3` rotation matrix $\mathbf{R} \in \text{SO}(3)$ to canonical unit quaternion `[w, x, y, z]`.
pub fn rot_matrix_to_quat(r: &[[f64; 3]; 3]) -> [f64; 4] {
    let tr = r[0][0] + r[1][1] + r[2][2];
    let q = if tr > 0.0 {
        let s = (tr + 1.0).max(0.0).sqrt() * 2.0;
        [
            0.25 * s,
            (r[2][1] - r[1][2]) / s,
            (r[0][2] - r[2][0]) / s,
            (r[1][0] - r[0][1]) / s,
        ]
    } else if r[0][0] > r[1][1] && r[0][0] > r[2][2] {
        let s = (1.0 + r[0][0] - r[1][1] - r[2][2]).max(0.0).sqrt() * 2.0;
        [
            (r[2][1] - r[1][2]) / s,
            0.25 * s,
            (r[0][1] + r[1][0]) / s,
            (r[0][2] + r[2][0]) / s,
        ]
    } else if r[1][1] > r[2][2] {
        let s = (1.0 + r[1][1] - r[0][0] - r[2][2]).max(0.0).sqrt() * 2.0;
        [
            (r[0][2] - r[2][0]) / s,
            (r[0][1] + r[1][0]) / s,
            0.25 * s,
            (r[1][2] + r[2][1]) / s,
        ]
    } else {
        let s = (1.0 + r[2][2] - r[0][0] - r[1][1]).max(0.0).sqrt() * 2.0;
        [
            (r[1][0] - r[0][1]) / s,
            (r[0][2] + r[2][0]) / s,
            (r[1][2] + r[2][1]) / s,
            0.25 * s,
        ]
    };
    quat_normalize(q)
}

/// Apply world-frame exponential update $\mathbf{q}_{\text{new}} = \exp(\frac{1}{2}\delta\boldsymbol{\omega}) \otimes \mathbf{q}$.
#[inline]
pub fn quat_apply_world_rot_step(q: [f64; 4], d_omega: Vec3) -> [f64; 4] {
    let angle = v_norm(d_omega);
    if angle <= 1e-16 || !angle.is_finite() {
        return quat_normalize(q);
    }
    let half = 0.5 * angle;
    let (sh, ch) = det_sin_cos(half);
    let k = sh / angle;
    let dq = [ch, d_omega[0] * k, d_omega[1] * k, d_omega[2] * k];
    quat_normalize([
        dq[0] * q[0] - dq[1] * q[1] - dq[2] * q[2] - dq[3] * q[3],
        dq[0] * q[1] + dq[1] * q[0] + dq[2] * q[3] - dq[3] * q[2],
        dq[0] * q[2] - dq[1] * q[3] + dq[2] * q[0] + dq[3] * q[1],
        dq[0] * q[3] + dq[1] * q[2] - dq[2] * q[1] + dq[3] * q[0],
    ])
}

#[inline]
fn mat3_mul_vec(m: &[[f64; 3]; 3], v: Vec3) -> Vec3 {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

#[inline]
fn mat3_transpose_mul_vec(m: &[[f64; 3]; 3], v: Vec3) -> Vec3 {
    [
        m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
        m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
        m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
    ]
}

#[inline]
fn mat3_mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    out
}

#[inline]
fn mat3_conjugate_sym(r: &[[f64; 3]; 3], s: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    // Compute R * S * R^T
    let rs = mat3_mul(r, s);
    let mut out = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = rs[i][0] * r[j][0] + rs[i][1] * r[j][1] + rs[i][2] * r[j][2];
        }
    }
    out
}

/// Build orthonormal right-handed local frame $\mathbf{R} = [\mathbf{e}_x, \mathbf{e}_y, \mathbf{e}_z]$
/// from `(n, ca, c)` with columns `e_x = unit(ca - n)`, `e_y = perp(c - ca)`, `e_z = e_x x e_y`.
#[inline]
pub(crate) fn triad_frame_from_n_ca_c(n: Vec3, ca: Vec3, c: Vec3) -> [[f64; 3]; 3] {
    let ex = v_unit(v_sub(ca, n));
    let v_cc = v_sub(c, ca);
    let perp = v_sub(v_cc, v_scale(ex, v_dot(v_cc, ex)));
    let ey = v_unit(perp);
    let ez = v_cross(ex, ey);
    [
        [ex[0], ey[0], ez[0]],
        [ex[1], ey[1], ez[1]],
        [ex[2], ey[2], ez[2]],
    ]
}

#[inline]
fn interior_angle_rad(a: Vec3, b: Vec3, c: Vec3, fallback_deg: f64) -> f64 {
    let u1 = v_sub(a, b);
    let u2 = v_sub(c, b);
    let n1 = v_norm(u1);
    let n2 = v_norm(u2);
    if n1 > 1e-6 && n2 > 1e-6 {
        let cos_th = (v_dot(u1, u2) / (n1 * n2)).clamp(-1.0, 1.0);
        det_acos(cos_th)
    } else {
        fallback_deg.to_radians()
    }
}

/// World-frame origin and proper `3x3` rotation matrix of a rigid domain block.
pub type BlockWorldFrame = (Vec3, [[f64; 3]; 3]);

impl DomainLinkerKinematics {
    /// Construct a single-chain or multi-chain `DomainLinkerKinematics` model from an `AfStructureModel`
    /// and its `PaeDomainPartition`, freezing all internal domain coordinates (`0.000 Å` drift) while
    /// activating $(\phi_l, \psi_l)$ across all `FlexibleLinker` residues.
    pub fn from_af_model(
        model: &AfStructureModel,
        partition: &PaeDomainPartition,
    ) -> Result<Self, String> {
        Self::from_residue_records(&model.residues, partition)
    }

    /// Construct a multi-chain `DomainLinkerKinematics` model from multiple chain `AfStructureModel`s
    /// concatenated in order, paired with a combined `PaeDomainPartition`.
    pub fn from_multi_chain_models(
        models: &[AfStructureModel],
        partition: &PaeDomainPartition,
    ) -> Result<Self, String> {
        if models.is_empty() {
            return Err("from_multi_chain_models requires at least 1 AfStructureModel".to_string());
        }
        let mut all_res: Vec<AfResidueRecord> = Vec::new();
        for (m_idx, m) in models.iter().enumerate() {
            for r in &m.residues {
                let mut rc = r.clone();
                rc.seq_index = all_res.len();
                if rc.chain_id.trim().is_empty() {
                    rc.chain_id = format!("{}", (b'A' + (m_idx as u8).min(25)) as char);
                }
                all_res.push(rc);
            }
        }
        Self::from_residue_records(&all_res, partition)
    }

    pub fn from_residue_records(
        af_residues: &[AfResidueRecord],
        partition: &PaeDomainPartition,
    ) -> Result<Self, String> {
        let l = af_residues.len();
        if l == 0 {
            return Err("cannot build DomainLinkerKinematics from 0 residues".to_string());
        }
        if partition.n_residues != l || partition.residue_roles.len() != l {
            return Err(format!(
                "partition residue count {} does not match structure residue count {l}",
                partition.n_residues
            ));
        }

        // Convert to proton-completed `PdbChain` matching `AfStructureModel::to_pdb_chain`
        let tmp_model = AfStructureModel {
            entry_id: None,
            detected_pdb_id: None,
            detected_uniprot_id: None,
            exptl_method: None,
            chain_id: af_residues[0].chain_id.clone(),
            provenance: crate::af_concordance::CoordinateProvenance::ComputationalModel,
            plddt_source: crate::af_concordance::PlddtSource::Unavailable,
            plddt_per_residue: None,
            residues: af_residues.to_vec(),
            diagnostics: Vec::new(),
        };
        let ref_chain = tmp_model.to_pdb_chain();

        // Identify contiguous polymer chains (split by `chain_id` OR physical chain break `d(C_{i-1}, N_i) > 2.5 Å`)
        let mut chain_ranges: Vec<(usize, usize)> = Vec::new();
        let mut chain_ids: Vec<String> = Vec::new();
        let mut res_chain_idx = vec![0usize; l];
        let mut c_start = 0usize;
        while c_start < l {
            let cid = af_residues[c_start].chain_id.clone();
            let mut c_end = c_start + 1;
            while c_end < l
                && af_residues[c_end].chain_id == cid
                && (v_dist(ref_chain.residues[c_end - 1].c, ref_chain.residues[c_end].n) <= 2.5
                    || af_residues[c_end].seq_id == af_residues[c_end - 1].seq_id + 1)
            {
                c_end += 1;
            }
            let c_idx = chain_ranges.len();
            for slot in &mut res_chain_idx[c_start..c_end] {
                *slot = c_idx;
            }
            chain_ranges.push((c_start, c_end));
            chain_ids.push(cid);
            c_start = c_end;
        }

        let mut chain_ref_centroids = Vec::with_capacity(chain_ranges.len());
        for &(cs, ce) in &chain_ranges {
            let mut sum = [0.0_f64; 3];
            let cnt = (ce - cs).max(1) as f64;
            for r in &ref_chain.residues[cs..ce] {
                sum = v_add(sum, r.ca);
            }
            chain_ref_centroids.push(v_scale(sum, 1.0 / cnt));
        }

        // Identify contiguous rigid blocks `[s_b, e_b)` inside each chain, tying all sequence spans
        // of the same `domain_id` to a single `master_block_id` (`21.W2:P0-4`: `0.000 Å` internal drift
        // across discontinuous insertion-domain topologies such as `D1 = [0..22] ∪ [50..72]`).
        let mut rigid_blocks: Vec<RigidBlockGeom> = Vec::new();
        let mut res_block_id: Vec<Option<usize>> = vec![None; l];
        let mut first_block_for_domain: BTreeMap<usize, usize> = BTreeMap::new();
        for (c_idx, &(cs, ce)) in chain_ranges.iter().enumerate() {
            let mut p = cs;
            while p < ce {
                let dom = partition.domain_labels[p];
                if dom == 0 {
                    p += 1;
                    continue;
                }
                let b_start = p;
                while p < ce && partition.domain_labels[p] == dom {
                    p += 1;
                }
                let b_end = p;
                let bid = rigid_blocks.len();
                let master_bid = *first_block_for_domain.entry(dom).or_insert(bid);
                for slot in &mut res_block_id[b_start..b_end] {
                    *slot = Some(bid);
                }
                rigid_blocks.push(RigidBlockGeom {
                    block_id: bid,
                    master_block_id: master_bid,
                    domain_id: dom,
                    chain_idx: c_idx,
                    start_res: b_start,
                    end_res: b_end,
                });
            }
        }

        // Precompute each rigid block's initial AF triad frame `(N_{s_master}, R_master^AF)`
        // (re-entry spans inherit their domain's master frame so all spans share one rigid body!).
        let mut block_frames: Vec<(Vec3, [[f64; 3]; 3])> = Vec::with_capacity(rigid_blocks.len());
        for blk in &rigid_blocks {
            if blk.master_block_id == blk.block_id {
                let rs = &ref_chain.residues[blk.start_res];
                block_frames.push((rs.n, triad_frame_from_n_ca_c(rs.n, rs.ca, rs.c)));
            } else {
                let master_frame = block_frames[blk.master_block_id];
                block_frames.push(master_frame);
            }
        }

        let mut residues = Vec::with_capacity(l);
        let mut ref_phi_rad = vec![-60.0_f64.to_radians(); l];
        let mut ref_psi_rad = vec![-45.0_f64.to_radians(); l];
        let mut ref_omega_rad = vec![PI; l];
        let mut active_linker_residues = Vec::new();

        for i in 0..l {
            let r_pdb = &ref_chain.residues[i];
            let r_af = &af_residues[i];
            let c_idx = res_chain_idx[i];
            let (cs, ce) = chain_ranges[c_idx];
            let is_chain_start = i == cs;
            let is_chain_end = i + 1 == ce;
            let blk_opt = res_block_id[i];
            let master_blk_opt = blk_opt.map(|bid| rigid_blocks[bid].master_block_id);
            let is_block_start = blk_opt.is_some_and(|bid| rigid_blocks[bid].start_res == i);
            let is_reentry_start = blk_opt.is_some_and(|bid| {
                rigid_blocks[bid].start_res == i && rigid_blocks[bid].master_block_id != bid
            });
            let is_gly = r_pdb.res_name.eq_ignore_ascii_case("GLY");
            let is_pro = r_pdb.res_name.eq_ignore_ascii_case("PRO");
            let res_class = if is_gly {
                DockResClass::Gly
            } else if is_pro {
                DockResClass::Pro
            } else {
                DockResClass::General
            };

            let is_linker = matches!(
                partition.residue_roles[i],
                ResidueDomainRole::FlexibleLinker { .. }
            );
            // Inside rigid blocks, internal torsions are strictly frozen; on flexible linkers,
            // `(phi_l, psi_l)` vary freely (`phi` fixed on `PRO` or at `chain_start`).
            let phi_active = is_linker && blk_opt.is_none() && !is_chain_start && !is_pro;
            let psi_active = is_linker && blk_opt.is_none() && (i + 1 < ce);
            let rama_active = phi_active || psi_active;
            if phi_active || psi_active {
                active_linker_residues.push(i);
            }

            // Construct virtual C_{-1} for chain root `cs` so `phi_{cs}` reproduces `C_{cs}` exactly
            let c_prev_virtual = if is_chain_start {
                let ex = v_unit(v_sub(r_pdb.ca, r_pdb.n));
                let v_cc = v_sub(r_pdb.c, r_pdb.ca);
                let ey = v_unit(v_sub(v_cc, v_scale(ex, v_dot(v_cc, ex))));
                let ang = ANGLE_C_N_CA_DEG.to_radians();
                let (sa, ca_ang) = det_sin_cos(ang);
                v_add(
                    r_pdb.n,
                    v_add(v_scale(ex, BOND_C_N * ca_ang), v_scale(ey, BOND_C_N * sa)),
                )
            } else {
                [0.0; 3]
            };

            let (r_c_prev_n, ang_ca_prev_c_prev_n, r_n_ca, ang_c_prev_n_ca, phi_r) =
                if !is_chain_start {
                    let prev = &ref_chain.residues[i - 1];
                    let raw_cn = v_dist(prev.c, r_pdb.n);
                    let rn_ca = v_dist(r_pdb.n, r_pdb.ca).clamp(1.20, 1.80);
                    if is_reentry_start && (raw_cn - BOND_C_N).abs() > 0.15 {
                        // Input model has a broken/stretched re-entry peptide bond: enforce
                        // canonical peptide-bond closure geometry (`1.329 Å`, `21F:B5`).
                        (
                            BOND_C_N,
                            ANGLE_CA_C_N_DEG.to_radians(),
                            rn_ca,
                            ANGLE_C_N_CA_DEG.to_radians(),
                            -120.0_f64.to_radians(),
                        )
                    } else {
                        let rc_n = raw_cn.clamp(1.10, 2.50);
                        let a_cacn =
                            interior_angle_rad(prev.ca, prev.c, r_pdb.n, ANGLE_CA_C_N_DEG);
                        let a_cnca =
                            interior_angle_rad(prev.c, r_pdb.n, r_pdb.ca, ANGLE_C_N_CA_DEG);
                        let ph =
                            calc_dihedral_deg(prev.c, r_pdb.n, r_pdb.ca, r_pdb.c).to_radians();
                        (rc_n, a_cacn, rn_ca, a_cnca, ph)
                    }
                } else {
                    let rn_ca = v_dist(r_pdb.n, r_pdb.ca).clamp(1.20, 1.80);
                    let a_cnca =
                        interior_angle_rad(c_prev_virtual, r_pdb.n, r_pdb.ca, ANGLE_C_N_CA_DEG);
                    let ph =
                        calc_dihedral_deg(c_prev_virtual, r_pdb.n, r_pdb.ca, r_pdb.c).to_radians();
                    (BOND_C_N, ANGLE_CA_C_N_DEG.to_radians(), rn_ca, a_cnca, ph)
                };

            let def_ncac = if is_gly {
                ANGLE_N_CA_C_GLY_DEG
            } else {
                ANGLE_N_CA_C_DEG
            };
            let r_ca_c = v_dist(r_pdb.ca, r_pdb.c).clamp(1.25, 1.85);
            let ang_n_ca_c = interior_angle_rad(r_pdb.n, r_pdb.ca, r_pdb.c, def_ncac);
            let r_c_o = v_dist(r_pdb.c, r_pdb.o).clamp(1.05, 1.50);
            let ang_ca_c_o = interior_angle_rad(r_pdb.ca, r_pdb.c, r_pdb.o, ANGLE_CA_C_O_DEG);

            let (psi_r, omg_r) = if i + 1 < ce {
                let nxt = &ref_chain.residues[i + 1];
                let ps = calc_dihedral_deg(r_pdb.n, r_pdb.ca, r_pdb.c, nxt.n).to_radians();
                let next_is_reentry = res_block_id[i + 1].is_some_and(|bid| {
                    rigid_blocks[bid].start_res == i + 1 && rigid_blocks[bid].master_block_id != bid
                });
                let om = if next_is_reentry && (v_dist(r_pdb.c, nxt.n) - BOND_C_N).abs() > 0.15 {
                    PI
                } else {
                    calc_dihedral_deg(r_pdb.ca, r_pdb.c, nxt.n, nxt.ca).to_radians()
                };
                (ps, om)
            } else {
                let ps = wrap_angle_rad(
                    calc_dihedral_deg(r_pdb.n, r_pdb.ca, r_pdb.c, r_pdb.o).to_radians() - PI,
                );
                (ps, PI)
            };

            ref_phi_rad[i] = wrap_angle_rad(phi_r);
            ref_psi_rad[i] = wrap_angle_rad(psi_r);
            ref_omega_rad[i] = wrap_angle_rad(omg_r);

            // 7 canonical atom slots matching `rdc_refine.wgsl`
            let slot_ha = if is_gly {
                r_pdb.ha2.map_or((r_pdb.ha, true), |p| (p, true))
            } else {
                (r_pdb.ha, true)
            };
            let slot_cb = if is_gly {
                r_pdb.ha3.map_or(([0.0; 3], false), |p| (p, true))
            } else {
                r_pdb.cb.map_or(([0.0; 3], false), |p| (p, true))
            };
            let ref_slots: [(Vec3, bool); 7] = [
                (r_pdb.n, true),
                r_pdb.h.map_or(([0.0; 3], false), |p| (p, true)),
                (r_pdb.ca, true),
                slot_ha,
                (r_pdb.c, true),
                (r_pdb.o, true),
                slot_cb,
            ];

            let mut local_slots = ref_slots;
            let mut extra_heavy_local = Vec::new();
            if let Some(bid) = blk_opt {
                let (orig_n, ref frame_rot) = block_frames[bid];
                for s_idx in 0..7 {
                    if ref_slots[s_idx].1 {
                        local_slots[s_idx].0 =
                            mat3_transpose_mul_vec(frame_rot, v_sub(ref_slots[s_idx].0, orig_n));
                    }
                }
                for (aname, &axyz) in &r_af.atoms {
                    if !matches!(
                        aname.as_str(),
                        "N" | "H" | "HN" | "CA" | "HA" | "HA2" | "HA3" | "C" | "O" | "CB"
                    ) {
                        extra_heavy_local.push((
                            aname.clone(),
                            mat3_transpose_mul_vec(frame_rot, v_sub(axyz, orig_n)),
                        ));
                    }
                }
            }

            residues.push(ResidueKinGeom {
                seq_id: r_pdb.seq_id,
                res_name: r_pdb.res_name.clone(),
                chain_id: r_af.chain_id.clone(),
                chain_idx: c_idx,
                res_class,
                rigid_block_id: blk_opt,
                master_block_id: master_blk_opt,
                domain_id: partition.domain_labels[i],
                phi_active,
                psi_active,
                rama_active,
                is_chain_start,
                is_chain_end,
                is_block_start,
                is_reentry_start,
                r_c_prev_n,
                ang_ca_prev_c_prev_n,
                r_n_ca,
                ang_c_prev_n_ca,
                r_ca_c,
                ang_n_ca_c,
                r_c_o,
                ang_ca_c_o,
                c_prev_virtual,
                ref_slots,
                local_slots,
                extra_heavy_local,
                plddt: r_af.plddt,
            });
        }

        Ok(Self {
            residues,
            rigid_blocks,
            chain_ranges,
            chain_ids,
            chain_ref_centroids,
            ref_phi_rad,
            ref_psi_rad,
            ref_omega_rad,
            active_linker_residues,
            ref_chain,
        })
    }

    /// Initial state corresponding to the exact input AlphaFold coordinates (`0.000 Å` difference).
    pub fn initial_state(&self) -> DomainDockState {
        let nc = self.chain_ranges.len();
        DomainDockState {
            phi_rad: self.ref_phi_rad.clone(),
            psi_rad: self.ref_psi_rad.clone(),
            omega_rad: self.ref_omega_rad.clone(),
            chain_quats: vec![[1.0, 0.0, 0.0, 0.0]; nc],
            chain_trans: vec![[0.0, 0.0, 0.0]; nc],
        }
    }

    /// Reconstruct world-frame 7-slot atom array (`7 * L`) and per-rigid-block world frames `(origin_n, R_b)`.
    /// All spans belonging to a discontinuous multi-span domain share `blk_world[master_block_id]` (`0.000 Å` internal drift).
    pub fn forward_slots_and_block_frames(
        &self,
        state: &DomainDockState,
    ) -> (Vec<(Vec3, bool)>, Vec<BlockWorldFrame>) {
        let l = self.residues.len();
        let mut slots = vec![([0.0_f64; 3], false); 7 * l];
        let mut blk_world = vec![([0.0_f64; 3], [[0.0_f64; 3]; 3]); self.rigid_blocks.len()];
        let ang_c_n_h = ANGLE_C_N_H_DEG.to_radians();

        for (c_idx, &(cs, ce)) in self.chain_ranges.iter().enumerate() {
            let rot_c = quat_to_rot_matrix(state.chain_quats[c_idx]);
            let c_af = self.chain_ref_centroids[c_idx];
            let tr_c = if c_idx == 0 {
                [0.0; 3]
            } else {
                state.chain_trans[c_idx]
            };
            let xform_root = |p: Vec3| -> Vec3 {
                v_add(v_add(c_af, tr_c), mat3_mul_vec(&rot_c, v_sub(p, c_af)))
            };

            let mut i = cs;
            while i < ce {
                let rg = &self.residues[i];
                if let Some(bid) = rg.rigid_block_id {
                    let blk = &self.rigid_blocks[bid];
                    let (n_s, r_b) = if blk.master_block_id != bid {
                        blk_world[blk.master_block_id]
                    } else {
                        let (n_s, ca_s, c_s) = if i == cs {
                            (
                                xform_root(rg.ref_slots[0].0),
                                xform_root(rg.ref_slots[2].0),
                                xform_root(rg.ref_slots[4].0),
                            )
                        } else {
                            let prev_base = 7 * (i - 1);
                            let n_prev = slots[prev_base].0;
                            let ca_prev = slots[prev_base + 2].0;
                            let c_prev = slots[prev_base + 4].0;
                            let psi_prev = state.psi_rad[i - 1];
                            let omg_prev = state.omega_rad[i - 1];
                            let n_cur = nerf_place_atom(
                                n_prev,
                                ca_prev,
                                c_prev,
                                rg.r_c_prev_n,
                                rg.ang_ca_prev_c_prev_n,
                                psi_prev,
                            );
                            let ca_cur = nerf_place_atom(
                                ca_prev,
                                c_prev,
                                n_cur,
                                rg.r_n_ca,
                                rg.ang_c_prev_n_ca,
                                omg_prev,
                            );
                            let c_cur = nerf_place_atom(
                                c_prev,
                                n_cur,
                                ca_cur,
                                rg.r_ca_c,
                                rg.ang_n_ca_c,
                                self.ref_phi_rad[i],
                            );
                            (n_cur, ca_cur, c_cur)
                        };
                        (n_s, triad_frame_from_n_ca_c(n_s, ca_s, c_s))
                    };

                    blk_world[bid] = (n_s, r_b);

                    for j in blk.start_res..blk.end_res {
                        let rj = &self.residues[j];
                        let j_base = 7 * j;
                        for s_idx in 0..7 {
                            let (loc_p, present) = rj.local_slots[s_idx];
                            if present {
                                slots[j_base + s_idx] =
                                    (v_add(n_s, mat3_mul_vec(&r_b, loc_p)), true);
                            }
                        }
                    }
                    i = blk.end_res;
                    continue;
                }

                // Flexible linker / tail residue `i`
                let r_base = 7 * i;
                let (n_cur, ca_cur, c_cur) = if i == cs {
                    let c_prev_v = xform_root(rg.c_prev_virtual);
                    let n0 = xform_root(rg.ref_slots[0].0);
                    let ca0 = xform_root(rg.ref_slots[2].0);
                    if rg.ref_slots[1].1 {
                        slots[r_base + 1] = (xform_root(rg.ref_slots[1].0), true);
                    }
                    let phi0 = if rg.phi_active {
                        state.phi_rad[i]
                    } else {
                        self.ref_phi_rad[i]
                    };
                    let c0 = nerf_place_atom(c_prev_v, n0, ca0, rg.r_ca_c, rg.ang_n_ca_c, phi0);
                    (n0, ca0, c0)
                } else {
                    let prev_base = 7 * (i - 1);
                    let n_prev = slots[prev_base].0;
                    let ca_prev = slots[prev_base + 2].0;
                    let c_prev = slots[prev_base + 4].0;
                    let psi_prev = if self.residues[i - 1].psi_active {
                        state.psi_rad[i - 1]
                    } else {
                        self.ref_psi_rad[i - 1]
                    };
                    let omg_prev = state.omega_rad[i - 1];
                    let n_i = nerf_place_atom(
                        n_prev,
                        ca_prev,
                        c_prev,
                        rg.r_c_prev_n,
                        rg.ang_ca_prev_c_prev_n,
                        psi_prev,
                    );
                    let ca_i = nerf_place_atom(
                        ca_prev,
                        c_prev,
                        n_i,
                        rg.r_n_ca,
                        rg.ang_c_prev_n_ca,
                        omg_prev,
                    );
                    if rg.res_class != DockResClass::Pro {
                        let h_i = nerf_place_atom(
                            ca_prev,
                            c_prev,
                            n_i,
                            BOND_N_H,
                            ang_c_n_h,
                            omg_prev + PI,
                        );
                        slots[r_base + 1] = (h_i, true);
                    }
                    let phi_i = if rg.phi_active {
                        state.phi_rad[i]
                    } else {
                        self.ref_phi_rad[i]
                    };
                    let c_i = nerf_place_atom(c_prev, n_i, ca_i, rg.r_ca_c, rg.ang_n_ca_c, phi_i);
                    (n_i, ca_i, c_i)
                };

                slots[r_base] = (n_cur, true);
                slots[r_base + 2] = (ca_cur, true);
                slots[r_base + 4] = (c_cur, true);

                let psi_cur = if rg.psi_active {
                    state.psi_rad[i]
                } else {
                    self.ref_psi_rad[i]
                };
                let o_cur =
                    nerf_place_atom(n_cur, ca_cur, c_cur, rg.r_c_o, rg.ang_ca_c_o, psi_cur + PI);
                slots[r_base + 5] = (o_cur, true);

                // Tetrahedral HA / CB reconstruction matching `rdc_refine.wgsl` & `domain_dock.wgsl`
                let b1 = v_unit(v_sub(n_cur, ca_cur));
                let b2 = v_unit(v_sub(c_cur, ca_cur));
                let u_bis = v_unit(v_add(b1, b2));
                let nrm = v_unit(v_cross(b1, b2));
                let dir_cb = v_unit(v_add(v_scale(u_bis, -0.5810), v_scale(nrm, 0.8139)));
                let dir_ha = v_unit(v_add(v_scale(u_bis, -0.5810), v_scale(nrm, -0.8139)));
                if rg.res_class == DockResClass::Gly {
                    slots[r_base + 3] = (v_add(ca_cur, v_scale(dir_ha, 1.090)), true);
                    slots[r_base + 6] = (v_add(ca_cur, v_scale(dir_cb, 1.090)), true);
                } else {
                    slots[r_base + 3] = (v_add(ca_cur, v_scale(dir_ha, 1.090)), true);
                    slots[r_base + 6] = (v_add(ca_cur, v_scale(dir_cb, 1.526)), true);
                }

                i += 1;
            }
        }

        (slots, blk_world)
    }

    /// Materialize a `PdbChain` at `state`.
    pub fn build_pdb_chain(&self, state: &DomainDockState) -> PdbChain {
        let (slots, _) = self.forward_slots_and_block_frames(state);
        self.slots_to_pdb_chain(&slots)
    }

    pub(crate) fn slots_to_pdb_chain(&self, slots: &[(Vec3, bool)]) -> PdbChain {
        let l = self.residues.len();
        let mut out = Vec::with_capacity(l);
        for i in 0..l {
            let rg = &self.residues[i];
            let base = 7 * i;
            let n = slots[base].0;
            let h = slots[base + 1].1.then_some(slots[base + 1].0);
            let ca = slots[base + 2].0;
            let c = slots[base + 4].0;
            let o = slots[base + 5].0;
            let is_gly = rg.res_class == DockResClass::Gly;
            let (ha, ha2, ha3, cb, qb) = if is_gly {
                let g2 = slots[base + 3].1.then_some(slots[base + 3].0);
                let g3 = slots[base + 6].1.then_some(slots[base + 6].0);
                let mid = match (g2, g3) {
                    (Some(a), Some(b)) => v_scale(v_add(a, b), 0.5),
                    (Some(a), None) => a,
                    _ => reconstruct_ca_substituents("GLY", n, ca, c).1,
                };
                (mid, g2, g3, None, None)
            } else {
                let ha_p = slots[base + 3].0;
                let cb_opt = slots[base + 6].1.then_some(slots[base + 6].0);
                let qb_opt = cb_opt.map(|cb_p| {
                    let dir = v_unit(v_sub(cb_p, ca));
                    v_add(cb_p, v_scale(dir, 0.3633))
                });
                (ha_p, None, None, cb_opt, qb_opt)
            };
            let ch_char = rg.chain_id.chars().next().unwrap_or('A');
            out.push(PdbResidue {
                seq_id: rg.seq_id,
                res_name: rg.res_name.clone(),
                chain_id: ch_char,
                n,
                h,
                ca,
                ha,
                ha2,
                ha3,
                c,
                o,
                cb,
                qb,
            });
        }
        PdbChain { residues: out }
    }

    /// Verify exact intra-domain rigid preservation: returns the maximum Kabsch backbone (`N, CA, C`)
    /// RMSD (Å) across all rigid domains `1..=D` between `chain` and the input AlphaFold coordinates.
    pub fn max_intra_domain_rmsd(&self, chain: &PdbChain, partition: &PaeDomainPartition) -> f64 {
        let mut max_rmsd = 0.0_f64;
        for dom in &partition.domains {
            let mut pts_cur = Vec::with_capacity(dom.residue_indices.len() * 3);
            let mut pts_ref = Vec::with_capacity(dom.residue_indices.len() * 3);
            for &idx in &dom.residue_indices {
                if let (Some(rc), Some(rr)) =
                    (chain.residues.get(idx), self.ref_chain.residues.get(idx))
                {
                    pts_cur.extend([rc.n, rc.ca, rc.c]);
                    pts_ref.extend([rr.n, rr.ca, rr.c]);
                }
            }
            if pts_cur.len() >= 3 {
                let rmsd = kabsch_rmsd(&pts_cur, &pts_ref);
                if rmsd > max_rmsd {
                    max_rmsd = rmsd;
                }
            }
        }
        if max_rmsd < 1e-12 {
            0.0
        } else {
            max_rmsd
        }
    }

    /// Return `true` if any rigid domain has discontinuous re-entry spans (`master_block_id != block_id`).
    pub fn has_reentry_blocks(&self) -> bool {
        self.rigid_blocks
            .iter()
            .any(|b| b.master_block_id != b.block_id)
    }

    /// Maximum peptide-bond `|d(C_{s_1 - 1}, N_{s_1}) - r_c_prev_n|` deviation (Å) across all
    /// discontinuous insertion-domain re-entry boundaries (`21F:B5`). Returns `0.0` when there are
    /// no discontinuous re-entry boundaries.
    pub fn max_reentry_cn_err_angstrom(&self, state: &DomainDockState) -> f64 {
        let (slots, _) = self.forward_slots_and_block_frames(state);
        self.max_reentry_cn_err_from_slots(&slots)
    }

    /// Compute maximum re-entry peptide-bond C–N deviation (Å) directly from evaluated atom slots.
    pub(crate) fn max_reentry_cn_err_from_slots(&self, slots: &[(Vec3, bool)]) -> f64 {
        let mut max_err = 0.0_f64;
        for blk in &self.rigid_blocks {
            if blk.master_block_id == blk.block_id {
                continue;
            }
            let s_1 = blk.start_res;
            let (cs, _) = self.chain_ranges[blk.chain_idx];
            if s_1 <= cs {
                continue;
            }
            let c_prev = slots[7 * (s_1 - 1) + 4].0;
            let n_reentry = slots[7 * s_1].0;
            let ideal_cn = self.residues[s_1].r_c_prev_n;
            let err = (v_dist(c_prev, n_reentry) - ideal_cn).abs();
            if err > max_err {
                max_err = err;
            }
        }
        max_err
    }
}

// ---- Prepared Restraint Tables (Zero `is_free` Leakage; Shared with `nmr-gpu`) ----

/// Pre-resolved working (`!is_free`) RDC restraint for CPU and WebGPU inner loops.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PreparedDockRdcTerm {
    pub medium_idx: usize,
    pub atom_a: usize,
    pub atom_b: usize,
    pub seg_a: usize,
    pub seg_b: usize,
    pub d_obs_hz: f64,
    pub d_max_eff_hz: f64,
    pub inv_sigma: f64,
    pub inv_sigma2: f64,
}

/// Fitted intra-domain $\Delta\boldsymbol{\chi}$ tag in local rigid-block coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedDockPcsTag {
    pub tag_id: String,
    pub host_domain_id: usize,
    pub host_block_id: usize,
    pub host_seg_id: usize,
    pub metal_local_pos: Vec3,
    pub dchi_local: [[f64; 3]; 3],
    pub pcs_offset_ppm: f64,
    pub fitted_tensor_initial: DeltaChiTensor,
}

/// Pre-resolved working (`!is_free`) cross-domain `PCS` restraint.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PreparedDockPcsTerm {
    pub tag_idx: usize,
    pub atom_slot: usize,
    pub seg_idx: usize,
    pub pcs_obs_eff_ppm: f64,
    pub inv_sigma2: f64,
}

/// Pre-resolved paramagnetic `PRE` center anchored in a rigid domain block.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedDockPreSite {
    pub site_id: String,
    pub host_domain_id: usize,
    pub host_block_id: usize,
    pub host_seg_id: usize,
    pub site_local_pos: Vec3,
}

/// Pre-resolved cross-domain `PRE` restraint (`mode == 0` -> Solomon $\Gamma_2$; `mode == 1` -> distance bounds).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PreparedDockPreTerm {
    pub site_idx: usize,
    pub atom_slot: usize,
    pub seg_idx: usize,
    pub is_bounds_mode: bool,
    pub gamma2_obs_s1: f64,
    pub inv_sigma2: f64,
    pub k_pre_a6_s1: f64,
    pub lower_a: f64,
    pub upper_a: f64,
}

/// Single candidate proton pair inside a (potentially ambiguous) inter-domain `NOE` restraint (`21.W2:P2-11`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PreparedDockNoeCandidate {
    pub atom_a: usize,
    pub atom_b: usize,
    pub seg_a: usize,
    pub seg_b: usize,
}

/// Pre-resolved inter-domain `NOE` distance restraint evaluated via Nilges (1995)
/// $r^{-6}$-summed effective distance $d_{\text{eff}} = (\sum_c r_c^{-6})^{-1/6}$ (`21.W2:P2-11`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PreparedDockNoeTerm {
    pub cand_start: usize,
    pub cand_len: usize,
    pub lower_a: f64,
    pub upper_a: f64,
}

/// Compiled strictly-working (`!is_free`) restraint bundle shared identically by CPU and WebGPU kernels.
#[derive(Clone, Debug, Default)]
pub struct PreparedDomainDockRestraints {
    pub n_media: usize,
    pub rdc_terms: Vec<PreparedDockRdcTerm>,
    pub pcs_tags: Vec<PreparedDockPcsTag>,
    pub pcs_terms: Vec<PreparedDockPcsTerm>,
    pub pre_sites: Vec<PreparedDockPreSite>,
    pub pre_terms: Vec<PreparedDockPreTerm>,
    pub noe_terms: Vec<PreparedDockNoeTerm>,
    pub noe_candidates: Vec<PreparedDockNoeCandidate>,
}

#[inline]
pub(crate) fn kin_res_seg_nhca(res_idx: usize, rg: &ResidueKinGeom) -> usize {
    if rg.is_reentry_start {
        2 * res_idx + 1
    } else {
        2 * res_idx
    }
}

#[inline]
pub(crate) fn kin_res_seg_o(res_idx: usize, rg: &ResidueKinGeom) -> usize {
    if rg.is_chain_end {
        2 * res_idx + 1
    } else {
        2 * res_idx + 2
    }
}

pub(crate) fn resolve_kin_atom_slot_and_seg_multi(
    kin: &DomainLinkerKinematics,
    cseq: &ChainSeqIndex,
    chain_opt: Option<&str>,
    seq_id: i32,
    atom_name: &str,
) -> Result<Option<(usize, usize)>, String> {
    let Some(res_idx) = cseq.resolve(chain_opt, seq_id)? else {
        return Ok(None);
    };
    let Some(rg) = kin.residues.get(res_idx) else {
        return Ok(None);
    };
    let norm = normalize_rdc_atom_name(atom_name);
    let is_gly = rg.res_class == DockResClass::Gly;
    let base = 7 * res_idx;
    let seg_nhca = kin_res_seg_nhca(res_idx, rg);
    let seg_o = kin_res_seg_o(res_idx, rg);
    let (slot, seg) = match norm {
        "N" => (base, seg_nhca),
        "H" => {
            if !rg.ref_slots[1].1 && rg.res_class == DockResClass::Pro {
                return Ok(None);
            }
            (base + 1, seg_nhca)
        }
        "CA" => (base + 2, seg_nhca),
        "HA" => {
            if is_gly {
                return Ok(None);
            }
            (base + 3, 2 * res_idx + 1)
        }
        "HA2" => {
            if !is_gly {
                return Ok(None);
            }
            (base + 3, 2 * res_idx + 1)
        }
        "HA3" => {
            if !is_gly {
                return Ok(None);
            }
            (base + 6, 2 * res_idx + 1)
        }
        "C" => (base + 4, 2 * res_idx + 1),
        "O" => (base + 5, seg_o),
        "CB" => {
            if is_gly {
                return Ok(None);
            }
            (base + 6, 2 * res_idx + 1)
        }
        _ => return Ok(None),
    };
    Ok(Some((slot, seg)))
}

impl PreparedDomainDockRestraints {
    /// Compile working (`!is_free`) `RDC`, `PCS` (fitting $\Delta\boldsymbol{\chi}_{\text{local}}$ in
    /// host domain $d_0$), `PRE`, and inter-domain `NOE` restraints against `kin`.
    pub fn compile(
        kin: &DomainLinkerKinematics,
        partition: &PaeDomainPartition,
        rdc_media: &[RdcMediumSet],
        pcs_tags: &[DomainPcsTagSet],
        pre_sites: &[DomainPreSiteSet],
        noe_adrs: &[AmbiguousDistanceRestraint],
        gdo_scale_map: Option<&BTreeMap<(usize, usize), f64>>,
    ) -> Result<Self, String> {
        let dom_media = DomainRdcMediumSet::from_rdc_media(rdc_media);
        Self::compile_multi_chain(
            kin,
            partition,
            &dom_media,
            pcs_tags,
            pre_sites,
            noe_adrs,
            gdo_scale_map,
            DEFAULT_MIN_ORDERED_PLDDT,
            false,
            8.0,
            DEFAULT_MTSL_TETHER_ALLOWANCE_ANGSTROM,
        )
    }

    /// Multi-chain `(chain_id, seq_id)` restraint compiler enforcing static-tensor domain hygiene
    /// (`21.W2:P1-4`), optional 8-parameter lanthanide metal + $\Delta\boldsymbol{\chi}$ VarPro
    /// refinement (`21.W2:P2-10`), MTSL tether allowance (`21.W3:P0-4`), unsoftened PRE bounds (`21.W2:P2-9`),
    /// and Nilges $r^{-6}$-summed ambiguous NOEs (`21.W2:P2-11`).
    #[allow(clippy::too_many_arguments)]
    pub fn compile_multi_chain(
        kin: &DomainLinkerKinematics,
        partition: &PaeDomainPartition,
        rdc_media: &[DomainRdcMediumSet],
        pcs_tags: &[DomainPcsTagSet],
        pre_sites: &[DomainPreSiteSet],
        noe_adrs: &[AmbiguousDistanceRestraint],
        gdo_scale_map: Option<&BTreeMap<(usize, usize), f64>>,
        min_ordered_plddt: f64,
        default_refine_metal_pos: bool,
        default_metal_search_radius_a: f64,
        default_mtsl_tether_allowance_a: f64,
    ) -> Result<Self, String> {
        let cseq = ChainSeqIndex::from_kin(kin)?;
        let kin_plddts: Vec<Option<f64>> = kin.residues.iter().map(|r| r.plddt).collect();

        // 1. Working (`!is_free`) RDCs strictly inside confident rigid domains (`21.W2:P1-4`)
        let mut rdc_terms = Vec::new();
        for (m_idx, med) in rdc_media.iter().enumerate() {
            for r in &med.restraints {
                r.validate()?;
                if r.is_free {
                    continue;
                }
                let (Some((atom_a, seg_a)), Some((atom_b, seg_b))) = (
                    resolve_kin_atom_slot_and_seg_multi(
                        kin,
                        &cseq,
                        r.chain_id_1.as_deref(),
                        r.seq_id_1,
                        &r.atom_name_1,
                    )?,
                    resolve_kin_atom_slot_and_seg_multi(
                        kin,
                        &cseq,
                        r.chain_id_2.as_deref(),
                        r.seq_id_2,
                        &r.atom_name_2,
                    )?,
                ) else {
                    continue;
                };
                let res_a = atom_a / 7;
                let res_b = atom_b / 7;
                let (Some(dom_a), Some(dom_b)) = (
                    is_confident_rigid_residue(
                        res_a,
                        partition,
                        Some(&kin_plddts),
                        min_ordered_plddt,
                    ),
                    is_confident_rigid_residue(
                        res_b,
                        partition,
                        Some(&kin_plddts),
                        min_ordered_plddt,
                    ),
                ) else {
                    continue;
                };
                if dom_a != dom_b {
                    continue;
                }
                let rho_scale = gdo_scale_map
                    .and_then(|mp| mp.get(&(dom_a, m_idx)).copied())
                    .unwrap_or(1.0);
                let inv_sig = 1.0 / r.sigma_hz;
                rdc_terms.push(PreparedDockRdcTerm {
                    medium_idx: m_idx,
                    atom_a,
                    atom_b,
                    seg_a,
                    seg_b,
                    d_obs_hz: r.d_obs_hz,
                    d_max_eff_hz: r.d_max_hz * rho_scale,
                    inv_sigma: inv_sig,
                    inv_sigma2: inv_sig * inv_sig,
                });
            }
        }

        let init_state = kin.initial_state();
        let (init_slots, init_blk_frames) = kin.forward_slots_and_block_frames(&init_state);

        // 2. PCS tags: resolve host domain master block & metal position (`7.0 Å` default lanthanide
        //    chelate offset when `# metal_anchor` is used, plus optional 8-parameter VarPro metal+DeltaChi
        //    refinement via `refine_metal_position_and_tensor`, `21.W2:P2-10`), and compile working
        //    (`!is_free`) cross-domain PCS terms!
        let mut prep_pcs_tags = Vec::new();
        let mut prep_pcs_terms = Vec::new();

        for tag in pcs_tags {
            let (mut metal_world, host_dom_id, master_host_blk_id) =
                resolve_paramagnetic_host_and_pos_multi(
                    kin,
                    partition,
                    &cseq,
                    &init_slots,
                    tag.metal_pos_initial,
                    tag.anchor_chain_id.as_deref(),
                    tag.anchor_residue_atom.as_ref(),
                    tag.host_domain_id,
                    &tag.tag_id,
                    7.0,
                )?;

            let use_varpro_metal = tag.refine_metal_pos || default_refine_metal_pos;
            let search_rad = tag
                .metal_search_radius_a
                .filter(|&v| v > 0.0)
                .unwrap_or_else(|| default_metal_search_radius_a.max(2.0));

            let fitted_tensor = if let Some(ref pre_t) = tag.pre_fitted_tensor {
                metal_world = pre_t.metal_pos_angstrom;
                pre_t.clone()
            } else {
                let mut intra_obs: Vec<PcsObservation> = Vec::new();
                for pr in &tag.restraints {
                    if pr.is_free {
                        continue;
                    }
                    let Some((slot, _)) = resolve_kin_atom_slot_and_seg_multi(
                        kin,
                        &cseq,
                        pr.chain_id.as_deref(),
                        pr.seq_id,
                        &pr.atom_name,
                    )?
                    else {
                        continue;
                    };
                    let res_i = slot / 7;
                    if kin.residues[res_i].domain_id == host_dom_id
                        && kin.residues[res_i].master_block_id == Some(master_host_blk_id)
                        && init_slots[slot].1
                    {
                        intra_obs.push(
                            PcsObservation::new(pr.seq_id, init_slots[slot].0, pr.pcs_obs_ppm)
                                .with_uncertainty(pr.sigma_ppm),
                        );
                    }
                }
                if use_varpro_metal {
                    if intra_obs.len() < 9 {
                        return Err(format!(
                            "PCS tag '{}' 8-parameter metal position + DeltaChi refinement requires >= 9 working intra-domain observations in host domain {}, got {}",
                            tag.tag_id,
                            host_dom_id,
                            intra_obs.len()
                        ));
                    }
                    let step_a = (search_rad / 4.0).clamp(0.75, 2.0);
                    let refined = refine_metal_position_and_tensor(
                        &intra_obs,
                        metal_world,
                        search_rad,
                        step_a,
                    )?;
                    metal_world = refined.metal_pos_angstrom;
                    refined
                } else {
                    if intra_obs.len() < 6 {
                        return Err(format!(
                            "PCS tag '{}' has only {} working intra-domain observations in host domain {} (require >= 6 for 5-parameter DeltaChi SVD)",
                            tag.tag_id,
                            intra_obs.len(),
                            host_dom_id
                        ));
                    }
                    fit_pcs_tensor_svd(&intra_obs, metal_world)?
                }
            };

            let (orig_n, ref r_b) = init_blk_frames[master_host_blk_id];
            let metal_local_pos = mat3_transpose_mul_vec(r_b, v_sub(metal_world, orig_n));
            let mut r_bt = [[0.0_f64; 3]; 3];
            for r in 0..3 {
                for c in 0..3 {
                    r_bt[r][c] = r_b[c][r];
                }
            }
            let dchi_local = mat3_conjugate_sym(&r_bt, &fitted_tensor.matrix_mol);
            let host_seg_id = 2 * kin.rigid_blocks[master_host_blk_id].start_res + 1;
            let tag_idx = prep_pcs_tags.len();
            let pcs_off = fitted_tensor.pcs_offset_ppm;

            prep_pcs_tags.push(PreparedDockPcsTag {
                tag_id: tag.tag_id.clone(),
                host_domain_id: host_dom_id,
                host_block_id: master_host_blk_id,
                host_seg_id,
                metal_local_pos,
                dchi_local,
                pcs_offset_ppm: pcs_off,
                fitted_tensor_initial: fitted_tensor,
            });

            for pr in &tag.restraints {
                if pr.is_free {
                    continue;
                }
                let Some((slot, seg)) = resolve_kin_atom_slot_and_seg_multi(
                    kin,
                    &cseq,
                    pr.chain_id.as_deref(),
                    pr.seq_id,
                    &pr.atom_name,
                )?
                else {
                    continue;
                };
                let res_i = slot / 7;
                if kin.residues[res_i].master_block_id == Some(master_host_blk_id) {
                    continue;
                }
                let inv_s = 1.0 / pr.sigma_ppm.max(1e-12);
                prep_pcs_terms.push(PreparedDockPcsTerm {
                    tag_idx,
                    atom_slot: slot,
                    seg_idx: seg,
                    pcs_obs_eff_ppm: pr.pcs_obs_ppm - pcs_off,
                    inv_sigma2: inv_s * inv_s,
                });
            }
        }

        // 3. PRE sites & cross-domain Solomon r^-6 / distance-bound terms (`21.W3:P0-4` & `21.W2:P2-9`)
        let mut prep_pre_sites = Vec::new();
        let mut prep_pre_terms = Vec::new();
        for ps in pre_sites {
            let (site_world, host_dom_id, master_host_blk_id) =
                resolve_paramagnetic_host_and_pos_multi(
                    kin,
                    partition,
                    &cseq,
                    &init_slots,
                    ps.site_pos_initial,
                    ps.anchor_chain_id.as_deref(),
                    ps.anchor_residue_atom.as_ref(),
                    ps.host_domain_id,
                    &ps.site_id,
                    2.5,
                )?;
            let (orig_n, ref r_b) = init_blk_frames[master_host_blk_id];
            let site_local_pos = mat3_transpose_mul_vec(r_b, v_sub(site_world, orig_n));
            let host_seg_id = 2 * kin.rigid_blocks[master_host_blk_id].start_res + 1;
            let site_idx = prep_pre_sites.len();
            prep_pre_sites.push(PreparedDockPreSite {
                site_id: ps.site_id.clone(),
                host_domain_id: host_dom_id,
                host_block_id: master_host_blk_id,
                host_seg_id,
                site_local_pos,
            });

            // Explicit `# site_pos` (rigid metal / explicit electron coordinates) defaults to `0.0 Å`
            // unless overridden per-site; `# site_anchor` (e.g. `CYS:CB` nitroxide spin label)
            // applies `default_mtsl_tether_allowance_a` (`DEFAULT_MTSL_TETHER_ALLOWANCE_ANGSTROM`, `21F:B18`).
            let tether_a = ps
                .mtsl_tether_allowance_a
                .unwrap_or_else(|| {
                    if ps.site_pos_initial.is_some() && ps.anchor_residue_atom.is_none() {
                        0.0
                    } else {
                        default_mtsl_tether_allowance_a
                    }
                })
                .max(0.0);

            for r in &ps.restraints {
                let Some((slot, seg)) = resolve_kin_atom_slot_and_seg_multi(
                    kin,
                    &cseq,
                    r.chain_id.as_deref(),
                    r.seq_id,
                    &r.atom_name,
                )?
                else {
                    continue;
                };
                let res_i = slot / 7;
                if kin.residues[res_i].master_block_id == Some(master_host_blk_id) {
                    continue;
                }
                match r.kind {
                    PreRestraintKind::Gamma2Rate {
                        gamma2_obs_s1,
                        sigma_s1,
                        k_pre_a6_s1,
                    } => {
                        if tether_a > 0.0 {
                            // Flexible MTSL nitroxide tether allowance (`Iwahara et al. 2004`):
                            // convert `Gamma2 +/- sigma` via exact `r^-6` into `[r_lo - tether_a, r_hi + tether_a]`.
                            let g_hi = (gamma2_obs_s1 + sigma_s1.max(1e-6)).max(1e-6);
                            let g_lo = (gamma2_obs_s1 - sigma_s1.max(1e-6)).max(0.0);
                            let r_lo = (det_cbrt(k_pre_a6_s1 / g_hi).sqrt() - tether_a).max(2.0);
                            let r_hi = if g_lo > 1e-6 {
                                det_cbrt(k_pre_a6_s1 / g_lo).sqrt() + tether_a
                            } else {
                                80.0 + tether_a
                            };
                            let r_nom = det_cbrt(k_pre_a6_s1 / gamma2_obs_s1.max(1e-6)).sqrt();
                            let sig_r = (r_nom * sigma_s1 / (6.0 * gamma2_obs_s1.max(1e-3)))
                                .clamp(0.5, 4.0);
                            let inv_s = 1.0 / sig_r;
                            prep_pre_terms.push(PreparedDockPreTerm {
                                site_idx,
                                atom_slot: slot,
                                seg_idx: seg,
                                is_bounds_mode: true,
                                gamma2_obs_s1: 0.0,
                                inv_sigma2: inv_s * inv_s,
                                k_pre_a6_s1: 0.0,
                                lower_a: r_lo,
                                upper_a: r_hi.max(r_lo),
                            });
                        } else {
                            let inv_s = 1.0 / sigma_s1.max(1e-12);
                            // Exact unsoftened `[r_lo, r_hi]` outer guide (`+/- 3.0 * sigma +/- 1.0 Å`,
                            // `0.0` energy & gradient at the exact unsoftened `r = (K_PRE / Gamma2)^{1/6}`!).
                            let g_hi = (gamma2_obs_s1 + 3.0 * sigma_s1).max(1e-6);
                            let g_lo = (gamma2_obs_s1 - 3.0 * sigma_s1).max(0.0);
                            let r_lo = (det_cbrt(k_pre_a6_s1 / g_hi).sqrt() - 1.0).max(3.5);
                            let r_hi = if g_lo > 1e-6 {
                                det_cbrt(k_pre_a6_s1 / g_lo).sqrt() + 1.0
                            } else {
                                80.0
                            };
                            prep_pre_terms.push(PreparedDockPreTerm {
                                site_idx,
                                atom_slot: slot,
                                seg_idx: seg,
                                is_bounds_mode: false,
                                gamma2_obs_s1,
                                inv_sigma2: inv_s * inv_s,
                                k_pre_a6_s1,
                                lower_a: r_lo,
                                upper_a: r_hi.max(r_lo),
                            });
                        }
                    }
                    PreRestraintKind::DistanceBounds {
                        lower_a,
                        upper_a,
                        sigma_a,
                    } => {
                        let inv_s = 1.0 / sigma_a.max(1e-12);
                        let lo_eff = (lower_a - tether_a).max(1.8);
                        let hi_eff = (upper_a + tether_a).max(lo_eff);
                        prep_pre_terms.push(PreparedDockPreTerm {
                            site_idx,
                            atom_slot: slot,
                            seg_idx: seg,
                            is_bounds_mode: true,
                            gamma2_obs_s1: 0.0,
                            inv_sigma2: inv_s * inv_s,
                            k_pre_a6_s1: 0.0,
                            lower_a: lo_eff,
                            upper_a: hi_eff,
                        });
                    }
                }
            }
        }

        // 4. Inter-domain / linker NOE distance restraints with Nilges (1995) r^-6 sum-averaging (`21.W2:P2-11`)
        let mut noe_terms = Vec::new();
        let mut noe_candidates = Vec::new();
        let map_noe_slot = |res_idx: usize, site: NoeProton| -> Option<(usize, usize)> {
            let rg = kin.residues.get(res_idx)?;
            let base = 7 * res_idx;
            let seg_nhca = kin_res_seg_nhca(res_idx, rg);
            match site {
                NoeProton::AmideH => {
                    (rg.res_class != DockResClass::Pro).then_some((base + 1, seg_nhca))
                }
                NoeProton::AlphaHA => Some((base + 3, 2 * res_idx + 1)),
                NoeProton::BetaQB => {
                    (rg.res_class != DockResClass::Gly).then_some((base + 6, 2 * res_idx + 1))
                }
            }
        };

        for adr in noe_adrs {
            if adr.candidates.is_empty() {
                continue;
            }
            let mut resolved_cands: Vec<PreparedDockNoeCandidate> = Vec::new();
            let mut has_varying_pair = false;
            for cand in &adr.candidates {
                if cand.res_a >= kin.residues.len() || cand.res_b >= kin.residues.len() {
                    continue;
                }
                let (Some((atom_a, seg_a)), Some((atom_b, seg_b))) = (
                    map_noe_slot(cand.res_a, cand.atom_a),
                    map_noe_slot(cand.res_b, cand.atom_b),
                ) else {
                    continue;
                };
                let ma = kin.residues[cand.res_a].master_block_id;
                let mb = kin.residues[cand.res_b].master_block_id;
                if !(ma.is_some() && ma == mb) {
                    has_varying_pair = true;
                }
                resolved_cands.push(PreparedDockNoeCandidate {
                    atom_a,
                    atom_b,
                    seg_a,
                    seg_b,
                });
            }
            if !has_varying_pair || resolved_cands.is_empty() {
                continue;
            }
            let cand_start = noe_candidates.len();
            let cand_len = resolved_cands.len();
            noe_candidates.extend(resolved_cands);
            noe_terms.push(PreparedDockNoeTerm {
                cand_start,
                cand_len,
                lower_a: adr.lower_bound,
                upper_a: adr.upper_bound,
            });
        }

        Ok(Self {
            n_media: rdc_media.len(),
            rdc_terms,
            pcs_tags: prep_pcs_tags,
            pcs_terms: prep_pcs_terms,
            pre_sites: prep_pre_sites,
            pre_terms: prep_pre_terms,
            noe_terms,
            noe_candidates,
        })
    }

    /// Fit exact `f64` global 5-parameter Saupe matrices $\mathbf{S}^{(m)}$ on working (`!is_free`)
    /// `rdc_terms` at precomputed world `slots`.
    pub fn fit_global_saupe_matrices_on_slots(&self, slots: &[(Vec3, bool)]) -> Vec<[[f64; 3]; 3]> {
        let mut out = vec![[[0.0_f64; 3]; 3]; self.n_media];
        if self.n_media == 0 || self.rdc_terms.is_empty() {
            return out;
        }
        let mut qrs = vec![DockGivensQr5::new(); self.n_media];
        let mut bad = vec![false; self.n_media];
        for t in &self.rdc_terms {
            let (pa, oka) = slots[t.atom_a];
            let (pb, okb) = slots[t.atom_b];
            if !oka || !okb {
                continue;
            }
            let dv = v_sub(pb, pa);
            let dist = v_norm(dv);
            if !dist.is_finite() || dist < 1e-6 {
                bad[t.medium_idx] = true;
                continue;
            }
            let inv_d = 1.0 / dist;
            let ux = dv[0] * inv_d;
            let uy = dv[1] * inv_d;
            let uz = dv[2] * inv_d;
            let w_scale = t.d_max_eff_hz * t.inv_sigma;
            qrs[t.medium_idx].push_row(
                [
                    w_scale * (uy * uy - ux * ux),
                    w_scale * (uz * uz - ux * ux),
                    w_scale * (2.0 * ux * uy),
                    w_scale * (2.0 * ux * uz),
                    w_scale * (2.0 * uy * uz),
                ],
                t.d_obs_hz * t.inv_sigma,
            );
        }
        for m in 0..self.n_media {
            if !bad[m] {
                if let Some(s) = qrs[m].solve_saupe() {
                    out[m] = s;
                }
            }
        }
        out
    }
}

#[allow(clippy::too_many_arguments)]
fn resolve_paramagnetic_host_and_pos_multi(
    kin: &DomainLinkerKinematics,
    partition: &PaeDomainPartition,
    cseq: &ChainSeqIndex,
    init_slots: &[(Vec3, bool)],
    explicit_pos: Option<Vec3>,
    anchor_chain_opt: Option<&str>,
    anchor_opt: Option<&(i32, String)>,
    explicit_dom: Option<usize>,
    label: &str,
    default_anchor_offset_a: f64,
) -> Result<(Vec3, usize, usize), String> {
    if kin.rigid_blocks.is_empty() {
        return Err(format!(
            "cannot anchor paramagnetic site '{label}' without at least 1 rigid domain"
        ));
    }

    let pos = if let Some(p) = explicit_pos {
        p
    } else if let Some((sid, aname)) = anchor_opt {
        let (slot, _) =
            resolve_kin_atom_slot_and_seg_multi(kin, cseq, anchor_chain_opt, *sid, aname)?
                .ok_or_else(|| {
                    format!("paramagnetic anchor {sid}:{aname} for '{label}' not found on chain")
                })?;
        let base_p = init_slots[slot].0;
        let res_i = slot / 7;
        let ca_p = init_slots[7 * res_i + 2].0;
        let dir = if v_dist(base_p, ca_p) > 0.2 {
            v_unit(v_sub(base_p, ca_p))
        } else {
            [1.0, 0.0, 0.0]
        };
        v_add(base_p, v_scale(dir, default_anchor_offset_a))
    } else {
        return Err(format!(
            "paramagnetic site '{label}' requires explicit `# metal_pos` / `# site_pos` or `# metal_anchor` / `# site_anchor`"
        ));
    };

    let host_dom_id = if let Some(d_id) = explicit_dom {
        if d_id == 0 || d_id > partition.domains.len() {
            return Err(format!(
                "paramagnetic site '{label}' specifies domain_id={d_id}, outside 1..={}",
                partition.domains.len()
            ));
        }
        d_id
    } else if let Some((sid, _)) = anchor_opt {
        let res_i = cseq
            .resolve(anchor_chain_opt, *sid)?
            .ok_or_else(|| format!("anchor seq_id {sid} for '{label}' not found"))?;
        let d = kin.residues[res_i].domain_id;
        if d == 0 {
            return Err(format!(
                "anchor seq_id {sid} for '{label}' lies in a flexible linker/tail rather than a rigid domain"
            ));
        }
        d
    } else {
        let mut best_dom = kin.rigid_blocks[0].domain_id;
        let mut best_d2 = f64::INFINITY;
        for blk in &kin.rigid_blocks {
            for j in blk.start_res..blk.end_res {
                let d2 = v_dist_sq(pos, init_slots[7 * j + 2].0);
                if d2 < best_d2 {
                    best_d2 = d2;
                    best_dom = blk.domain_id;
                }
            }
        }
        best_dom
    };

    // Resolve the domain's primary `master_block_id` so multi-span domains share one rigid frame!
    let master_blk_id = kin
        .rigid_blocks
        .iter()
        .find(|b| b.domain_id == host_dom_id && b.master_block_id == b.block_id)
        .or_else(|| kin.rigid_blocks.iter().find(|b| b.domain_id == host_dom_id))
        .map(|b| b.master_block_id)
        .ok_or_else(|| format!("no rigid block found for domain {host_dom_id}"))?;

    Ok((pos, host_dom_id, master_blk_id))
}

#[derive(Clone, Copy)]
struct DockGivensQr5 {
    r: [[f64; 5]; 5],
    z: [f64; 5],
    n_rows: usize,
}

impl DockGivensQr5 {
    #[inline]
    fn new() -> Self {
        Self {
            r: [[0.0; 5]; 5],
            z: [0.0; 5],
            n_rows: 0,
        }
    }

    #[inline]
    fn push_row(&mut self, mut x: [f64; 5], mut beta: f64) {
        for j in 0..5 {
            let b_val = x[j];
            if b_val == 0.0 {
                continue;
            }
            let a_val = self.r[j][j];
            let scale = a_val.abs().max(b_val.abs());
            if scale <= 1e-300 {
                continue;
            }
            let na = a_val / scale;
            let nb = b_val / scale;
            let r_norm = scale * (na * na + nb * nb).sqrt();
            let c = a_val / r_norm;
            let s = b_val / r_norm;
            self.r[j][j] = r_norm;
            x[j] = 0.0;
            for (r_jc, x_c) in self.r[j][(j + 1)..5]
                .iter_mut()
                .zip(x[(j + 1)..5].iter_mut())
            {
                let r_old = *r_jc;
                let x_old = *x_c;
                *r_jc = c * r_old + s * x_old;
                *x_c = -s * r_old + c * x_old;
            }
            let z_j = self.z[j];
            self.z[j] = c * z_j + s * beta;
            beta = -s * z_j + c * beta;
        }
        self.n_rows += 1;
    }

    fn solve_saupe(&self) -> Option<[[f64; 3]; 3]> {
        self.solve_saupe_with_cond().map(|(s, _)| s)
    }

    fn solve_saupe_with_cond(&self) -> Option<([[f64; 3]; 3], f64)> {
        if self.n_rows < 6 {
            return None;
        }
        let mut w = self.r;
        let mut v = [[0.0_f64; 5]; 5];
        for (i, row) in v.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        for _ in 0..40 {
            let mut max_off = 0.0_f64;
            for p in 0..4 {
                for q in (p + 1)..5 {
                    let mut app = 0.0_f64;
                    let mut aqq = 0.0_f64;
                    let mut apq = 0.0_f64;
                    for row in &w {
                        app += row[p] * row[p];
                        aqq += row[q] * row[q];
                        apq += row[p] * row[q];
                    }
                    let sc = (app * aqq).sqrt();
                    if sc <= 1e-300 {
                        continue;
                    }
                    let rel = apq.abs() / sc;
                    if rel > max_off {
                        max_off = rel;
                    }
                    if rel <= 1e-15 {
                        continue;
                    }
                    let zeta = (aqq - app) / (2.0 * apq);
                    let t = if zeta >= 0.0 {
                        1.0 / (zeta + (1.0 + zeta * zeta).sqrt())
                    } else {
                        -1.0 / (-zeta + (1.0 + zeta * zeta).sqrt())
                    };
                    let c = 1.0 / (1.0 + t * t).sqrt();
                    let s = c * t;
                    for row in &mut w {
                        let wp = row[p];
                        let wq = row[q];
                        row[p] = c * wp - s * wq;
                        row[q] = s * wp + c * wq;
                    }
                    for row in &mut v {
                        let vp = row[p];
                        let vq = row[q];
                        row[p] = c * vp - s * vq;
                        row[q] = s * vp + c * vq;
                    }
                }
            }
            if max_off <= 1e-15 {
                break;
            }
        }
        let mut sigma = [0.0_f64; 5];
        let mut s_max = 0.0_f64;
        let mut s_min = f64::INFINITY;
        for (j, sig_j) in sigma.iter_mut().enumerate() {
            let sj = w.iter().map(|r| r[j] * r[j]).sum::<f64>().sqrt();
            *sig_j = sj;
            s_max = s_max.max(sj);
            s_min = s_min.min(sj);
        }
        if !s_max.is_finite() || !s_min.is_finite() || s_max <= 1e-30 || s_min <= 0.0 {
            return None;
        }
        let cond = s_max / s_min;
        if cond > MAX_SAUPE_SVD_COND {
            return None;
        }
        let mut scaled_ut_z = [0.0_f64; 5];
        for j in 0..5 {
            let wt_z: f64 = w
                .iter()
                .zip(self.z.iter())
                .map(|(wr, &zr)| wr[j] * zr)
                .sum();
            scaled_ut_z[j] = wt_z / (sigma[j] * sigma[j]);
        }
        let mut s_5 = [0.0_f64; 5];
        for i in 0..5 {
            s_5[i] = (0..5).map(|j| v[i][j] * scaled_ut_z[j]).sum();
        }
        let (syy, szz, sxy, sxz, syz) = (s_5[0], s_5[1], s_5[2], s_5[3], s_5[4]);
        let sxx = -(syy + szz);
        Some(([[sxx, sxy, sxz], [sxy, syy, syz], [sxz, syz, szz]], cond))
    }
}

// ---- Deterministic 3D Spatial Grid Hash for Heavy-Atom Steric Excluded Volume ----

#[derive(Clone, Copy)]
struct HeavyAtomEntry {
    res_idx: usize,
    chain_idx: usize,
    master_block_id: Option<usize>,
    seg_idx: usize,
    kind: u8,
    pos: Vec3,
}

#[inline]
fn dock_vdw_min_r2_base(ka: u8, kb: u8) -> f64 {
    match (ka.min(kb), ka.max(kb)) {
        (0, 2) => 2.65 * 2.65,
        (2, 2) => 2.80 * 2.80,
        (3, 3) => 3.25 * 3.25,
        (_, 3) => 3.00 * 3.00,
        _ => 2.85 * 2.85,
    }
}

/// Exact unsoftened Solomon $r^{-6}$ rate and derivative $\frac{d\Gamma_2}{d(r^2)}$ for all
/// physical distances $r \ge 3.5\text{ \AA}$ ($s = r^2 \ge 12.25\text{ \AA}^2$), with a $C^1$-continuous
/// linear-in-$r^2$ steric-core cap only below $r_{\text{core}} = 3.5\text{ \AA}$ (`21.W2:P2-9`).
#[inline]
pub(crate) fn solomon_r6_rate_and_ds(r2_raw: f64, k_pre_a6_s1: f64) -> (f64, f64) {
    const S_CORE: f64 = 12.25;
    if r2_raw >= S_CORE {
        let inv_s = 1.0 / r2_raw;
        let inv_s3 = inv_s * inv_s * inv_s;
        (k_pre_a6_s1 * inv_s3, -3.0 * k_pre_a6_s1 * inv_s3 * inv_s)
    } else {
        let inv_sc = 1.0 / S_CORE;
        let inv_sc3 = inv_sc * inv_sc * inv_sc;
        let g2_core = k_pre_a6_s1 * inv_sc3;
        (
            g2_core * (4.0 - 3.0 * r2_raw * inv_sc),
            -3.0 * g2_core * inv_sc,
        )
    }
}

/// Deterministic zero-`HashMap` 3D spatial voxel grid (`cell_size >= max(r_min)`) evaluating exact
/// inter-domain and linker quartic soft-core clashes in linear expected time.
fn accumulate_grid_hashed_vdw_cpu<F>(
    kin: &DomainLinkerKinematics,
    slots: &[(Vec3, bool)],
    w_vdw: f64,
    vdw_scale: f64,
    mut add_seg_grad: F,
) -> f64
where
    F: FnMut(usize, Vec3, Vec3),
{
    if w_vdw <= 0.0 || vdw_scale <= 0.0 || kin.residues.is_empty() {
        return 0.0;
    }
    let l = kin.residues.len();
    let s2 = vdw_scale * vdw_scale;
    let mut total_vdw = 0.0_f64;

    // 1. Local `(i, i+1)` Ramachandran steric guard on flexible residues
    for i in 0..l.saturating_sub(1) {
        let ri = &kin.residues[i];
        let rj = &kin.residues[i + 1];
        if ri.chain_idx != rj.chain_idx {
            continue;
        }
        if ri.master_block_id.is_some() && ri.master_block_id == rj.master_block_id {
            continue;
        }
        let oi = slots[7 * i + 5].0;
        let cj = slots[7 * (i + 1) + 4].0;
        let oj = slots[7 * (i + 1) + 5].0;
        let seg_oi = kin_res_seg_o(i, ri);
        let seg_cj = 2 * (i + 1) + 1;
        let seg_oj = kin_res_seg_o(i + 1, rj);
        let pairs = [
            (seg_oi, oi, 2u8, seg_cj, cj, 1u8, dock_vdw_min_r2_base(2, 1)),
            (seg_oi, oi, 2u8, seg_oj, oj, 2u8, dock_vdw_min_r2_base(2, 2)),
        ];
        for (sa, pa, _, sb, pb, _, base_r2) in pairs {
            let r2_min = s2 * base_r2;
            let diff = v_sub(pa, pb);
            let d2 = v_dot(diff, diff);
            if d2 < r2_min {
                let delta = r2_min - d2;
                total_vdw += w_vdw * delta * delta;
                let g = v_scale(diff, -4.0 * w_vdw * delta);
                add_seg_grad(sa, pa, g);
                add_seg_grad(sb, pb, [-g[0], -g[1], -g[2]]);
            }
        }
        if rj.res_class != DockResClass::Gly && slots[7 * (i + 1) + 6].1 {
            let cb_j = slots[7 * (i + 1) + 6].0;
            let r2_min = s2 * (2.90 * 2.90);
            let diff = v_sub(oi, cb_j);
            let d2 = v_dot(diff, diff);
            if d2 < r2_min {
                let delta = r2_min - d2;
                total_vdw += w_vdw * delta * delta;
                let g = v_scale(diff, -4.0 * w_vdw * delta);
                add_seg_grad(seg_oi, oi, g);
                add_seg_grad(seg_cj, cb_j, [-g[0], -g[1], -g[2]]);
            }
        }
    }

    // 2. Non-local pairs (`|i - j| >= 2` or distinct chains, skipping intra-domain pairs)
    //    indexed via a deterministic fixed-prime open-addressed / linked-list 3D spatial grid (`cell = 3.5 Å`).
    let cell_size = (3.35 * vdw_scale.max(0.75)).max(2.5);
    let inv_cell = 1.0 / cell_size;

    let mut atoms = Vec::with_capacity(l * 5);
    for (i, rg) in kin.residues.iter().enumerate() {
        let base = 7 * i;
        let seg_nhca = kin_res_seg_nhca(i, rg);
        let seg_o = kin_res_seg_o(i, rg);
        atoms.push(HeavyAtomEntry {
            res_idx: i,
            chain_idx: rg.chain_idx,
            master_block_id: rg.master_block_id,
            seg_idx: seg_nhca,
            kind: 0,
            pos: slots[base].0,
        });
        atoms.push(HeavyAtomEntry {
            res_idx: i,
            chain_idx: rg.chain_idx,
            master_block_id: rg.master_block_id,
            seg_idx: seg_nhca,
            kind: 1,
            pos: slots[base + 2].0,
        });
        atoms.push(HeavyAtomEntry {
            res_idx: i,
            chain_idx: rg.chain_idx,
            master_block_id: rg.master_block_id,
            seg_idx: 2 * i + 1,
            kind: 1,
            pos: slots[base + 4].0,
        });
        atoms.push(HeavyAtomEntry {
            res_idx: i,
            chain_idx: rg.chain_idx,
            master_block_id: rg.master_block_id,
            seg_idx: seg_o,
            kind: 2,
            pos: slots[base + 5].0,
        });
        if rg.res_class != DockResClass::Gly && slots[base + 6].1 {
            atoms.push(HeavyAtomEntry {
                res_idx: i,
                chain_idx: rg.chain_idx,
                master_block_id: rg.master_block_id,
                seg_idx: 2 * i + 1,
                kind: 3,
                pos: slots[base + 6].0,
            });
        }
    }

    let n_atoms = atoms.len();
    let table_cap = (n_atoms * 4).next_power_of_two().max(64);
    let mask = table_cap - 1;
    let mut cell_keys: Vec<(i32, i32, i32)> = vec![(i32::MIN, i32::MIN, i32::MIN); table_cap];
    let mut cell_head: Vec<i32> = vec![-1; table_cap];
    let mut next_atom: Vec<i32> = vec![-1; n_atoms];

    #[inline]
    fn hash_voxel(ix: i32, iy: i32, iz: i32, mask: usize) -> usize {
        let h = (ix as u64)
            .wrapping_mul(73_856_093)
            .wrapping_add((iy as u64).wrapping_mul(19_349_663))
            .wrapping_add((iz as u64).wrapping_mul(83_492_791));
        (h as usize) & mask
    }

    for (a_idx, at) in atoms.iter().enumerate() {
        let ix = (at.pos[0] * inv_cell).floor() as i32;
        let iy = (at.pos[1] * inv_cell).floor() as i32;
        let iz = (at.pos[2] * inv_cell).floor() as i32;

        // Query the 27 neighboring cells for already-inserted atoms (`b_idx < a_idx`)
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let qkey = (ix + dx, iy + dy, iz + dz);
                    let mut slot = hash_voxel(qkey.0, qkey.1, qkey.2, mask);
                    loop {
                        let head = cell_head[slot];
                        if head < 0 {
                            break;
                        }
                        if cell_keys[slot] == qkey {
                            let mut cur = head;
                            while cur >= 0 {
                                let b = &atoms[cur as usize];
                                cur = next_atom[cur as usize];

                                if b.master_block_id.is_some()
                                    && b.master_block_id == at.master_block_id
                                {
                                    continue;
                                }
                                let same_chain = b.chain_idx == at.chain_idx;
                                if same_chain && at.res_idx < b.res_idx + 2 {
                                    continue;
                                }
                                // Skip 1-4 peptide neighbor `C_{i} .. N_{i+2}` matching `structure.rs`
                                if same_chain
                                    && at.res_idx == b.res_idx + 2
                                    && b.kind == 1
                                    && at.kind == 0
                                    && b.seg_idx == 2 * b.res_idx + 1
                                {
                                    continue;
                                }

                                let r2_min = s2 * dock_vdw_min_r2_base(b.kind, at.kind);
                                let diff = v_sub(b.pos, at.pos);
                                let d2 = v_dot(diff, diff);
                                if d2 < r2_min {
                                    let delta = r2_min - d2;
                                    total_vdw += w_vdw * delta * delta;
                                    let g = v_scale(diff, -4.0 * w_vdw * delta);
                                    add_seg_grad(b.seg_idx, b.pos, g);
                                    add_seg_grad(at.seg_idx, at.pos, [-g[0], -g[1], -g[2]]);
                                }
                            }
                            break;
                        }
                        slot = (slot + 1) & mask;
                    }
                }
            }
        }

        // Insert `a_idx` into its own voxel bucket
        let key = (ix, iy, iz);
        let mut slot = hash_voxel(ix, iy, iz, mask);
        loop {
            if cell_head[slot] < 0 {
                cell_keys[slot] = key;
                cell_head[slot] = a_idx as i32;
                break;
            }
            if cell_keys[slot] == key {
                next_atom[a_idx] = cell_head[slot];
                cell_head[slot] = a_idx as i32;
                break;
            }
            slot = (slot + 1) & mask;
        }
    }

    total_vdw
}

/// Differentiable Lovell et al. (2003) Ramachandran basin potential & gradient `(V, dV/dphi, dV/dpsi)`
/// on flexible linker torsions (exact match to `domain_dock.wgsl`).
pub(crate) fn dock_ramachandran_energy_and_grad(
    rc: DockResClass,
    phi: f64,
    psi: f64,
) -> (f64, f64, f64) {
    const GEN_BASINS: &[(f64, f64, f64, f64, f64)] = &[
        (-63.0, -42.0, 22.0, 22.0, 1.00),
        (-122.0, 135.0, 28.0, 28.0, 1.00),
        (-138.0, 168.0, 26.0, 24.0, 0.92),
        (-75.0, 145.0, 22.0, 24.0, 0.85),
        (-85.0, -12.0, 20.0, 22.0, 0.40),
        (56.0, 38.0, 16.0, 18.0, 0.25),
    ];
    const PRO_BASINS: &[(f64, f64, f64, f64, f64)] = &[
        (-63.0, 145.0, 13.0, 24.0, 1.00),
        (-61.0, -38.0, 13.0, 22.0, 0.95),
    ];
    const GLY_BASINS: &[(f64, f64, f64, f64, f64)] = &[
        (-65.0, -41.0, 30.0, 30.0, 1.00),
        (65.0, 41.0, 30.0, 30.0, 1.00),
        (-80.0, 155.0, 34.0, 34.0, 0.95),
        (80.0, -155.0, 34.0, 34.0, 0.95),
        (82.0, 12.0, 30.0, 30.0, 0.90),
        (-82.0, -12.0, 30.0, 30.0, 0.90),
        (180.0, 180.0, 36.0, 36.0, 0.85),
    ];

    let basins = match rc {
        DockResClass::General => GEN_BASINS,
        DockResClass::Pro => PRO_BASINS,
        DockResClass::Gly => GLY_BASINS,
    };

    let floor_log = -9.0_f64;
    let mut max_log = floor_log;
    let mut logs = [0.0_f64; 8];
    let mut dlog_dphi = [0.0_f64; 8];
    let mut dlog_dpsi = [0.0_f64; 8];

    for (bi, &(p0_d, s0_d, sp_d, ss_d, amp)) in basins.iter().enumerate() {
        let dphi = wrap_angle_rad(phi - p0_d.to_radians());
        let dpsi = wrap_angle_rad(psi - s0_d.to_radians());
        let inv_vp = 1.0 / sp_d.to_radians().powi(2);
        let inv_vs = 1.0 / ss_d.to_radians().powi(2);
        let lg = det_ln(amp) - 0.5 * (dphi * dphi * inv_vp + dpsi * dpsi * inv_vs);
        logs[bi] = lg;
        dlog_dphi[bi] = -dphi * inv_vp;
        dlog_dpsi[bi] = -dpsi * inv_vs;
        if lg > max_log {
            max_log = lg;
        }
    }

    let mut sum_exp = det_exp(floor_log - max_log);
    let mut num_phi = 0.0_f64;
    let mut num_psi = 0.0_f64;
    for bi in 0..basins.len() {
        let w = det_exp(logs[bi] - max_log);
        sum_exp += w;
        num_phi += w * dlog_dphi[bi];
        num_psi += w * dlog_dpsi[bi];
    }
    let log_p = max_log + det_ln(sum_exp);
    let raw_v = -log_p - 0.35;
    if raw_v <= 0.0 {
        (0.0, 0.0, 0.0)
    } else {
        (raw_v, -num_phi / sum_exp, -num_psi / sum_exp)
    }
}

#[inline]
fn seg_local_pivot(slots: &[(Vec3, bool)], seg: usize) -> Vec3 {
    if seg == 0 {
        slots[0].0
    } else if (seg & 1) == 1 {
        let i = (seg - 1) / 2;
        slots[7 * i + 2].0
    } else {
        let i = (seg - 2) / 2;
        slots[7 * i + 4].0
    }
}

/// Evaluate the full composite docking energy (`RDC` + `PCS` + `PRE` + `NOE` + `vdW` + `Rama`)
/// and exact analytical linker-torsion + multi-chain $\mathfrak{se}(3)$ gradients on CPU (`f64`)
/// given fixed per-medium Saupe matrices `saupe_matrices` (exact single-step reference for WebGPU parity).
pub fn evaluate_domain_dock_fixed_saupe_cpu(
    kin: &DomainLinkerKinematics,
    prep: &PreparedDomainDockRestraints,
    state: &DomainDockState,
    saupe_matrices: &[[[f64; 3]; 3]],
    weights: &DomainDockEnergyWeights,
) -> (DomainDockEnergyBreakdown, DomainDockGradient) {
    let (slots, blk_frames) = kin.forward_slots_and_block_frames(state);
    evaluate_domain_dock_on_precomputed_slots(
        kin,
        prep,
        state,
        &slots,
        &blk_frames,
        saupe_matrices,
        weights,
        None,
    )
}

/// Evaluate the full variable-projection (`VarPro`) composite docking energy and exact Envelope-Theorem
/// analytical gradients on CPU (`f64`), fitting each medium's global Saupe matrix on the fly from
/// strictly working (`!is_free`) couplings.
///
/// Always enforces stiff covalent re-entry peptide-bond closure on discontinuous insertion domains
/// regardless of `weights.w_rama` (`21F:B5`).
pub fn evaluate_domain_dock_state_cpu(
    kin: &DomainLinkerKinematics,
    prep: &PreparedDomainDockRestraints,
    state: &DomainDockState,
    weights: &DomainDockEnergyWeights,
) -> (DomainDockEnergyBreakdown, DomainDockGradient) {
    evaluate_domain_dock_state_cpu_with_loop_k(
        kin,
        prep,
        state,
        weights,
        Some(500.0_f64.max(25.0 * weights.w_rama)),
    )
}

fn evaluate_domain_dock_state_cpu_with_loop_k(
    kin: &DomainLinkerKinematics,
    prep: &PreparedDomainDockRestraints,
    state: &DomainDockState,
    weights: &DomainDockEnergyWeights,
    k_loop_override: Option<f64>,
) -> (DomainDockEnergyBreakdown, DomainDockGradient) {
    let (slots, blk_frames) = kin.forward_slots_and_block_frames(state);
    let saupe = if weights.w_rdc > 0.0 && prep.n_media > 0 {
        prep.fit_global_saupe_matrices_on_slots(&slots)
    } else {
        vec![[[0.0; 3]; 3]; prep.n_media]
    };
    evaluate_domain_dock_on_precomputed_slots(
        kin,
        prep,
        state,
        &slots,
        &blk_frames,
        &saupe,
        weights,
        k_loop_override,
    )
}

#[allow(clippy::too_many_arguments)]
fn evaluate_domain_dock_on_precomputed_slots(
    kin: &DomainLinkerKinematics,
    prep: &PreparedDomainDockRestraints,
    state: &DomainDockState,
    slots: &[(Vec3, bool)],
    blk_frames: &[(Vec3, [[f64; 3]; 3])],
    saupe_matrices: &[[[f64; 3]; 3]],
    weights: &DomainDockEnergyWeights,
    k_loop_override: Option<f64>,
) -> (DomainDockEnergyBreakdown, DomainDockGradient) {
    let l = kin.residues.len();
    let n_chains = kin.chain_ranges.len();
    let n_segs = 2 * l + 1;
    let mut seg_force = vec![[0.0_f64; 3]; n_segs];
    let mut seg_torque = vec![[0.0_f64; 3]; n_segs];

    #[inline]
    fn add_seg_ft(
        seg_force: &mut [Vec3],
        seg_torque: &mut [Vec3],
        slots: &[(Vec3, bool)],
        seg: usize,
        pos: Vec3,
        g: Vec3,
    ) {
        let piv = seg_local_pivot(slots, seg);
        seg_force[seg] = v_add(seg_force[seg], g);
        seg_torque[seg] = v_add(seg_torque[seg], v_cross(v_sub(pos, piv), g));
    }

    let mut eb = DomainDockEnergyBreakdown::default();

    // 1. Multi-medium working RDC energy & forces
    if weights.w_rdc > 0.0 && saupe_matrices.len() == prep.n_media {
        let w_rdc = weights.w_rdc;
        for term in &prep.rdc_terms {
            let (pa, oka) = slots[term.atom_a];
            let (pb, okb) = slots[term.atom_b];
            if !oka || !okb {
                continue;
            }
            let dv = v_sub(pb, pa);
            let d2 = v_dot(dv, dv);
            if d2 < 1e-12 {
                continue;
            }
            let inv_r = 1.0 / d2.sqrt();
            let u = v_scale(dv, inv_r);
            let s = &saupe_matrices[term.medium_idx];
            let su = mat3_mul_vec(s, u);
            let ut_s_u = v_dot(u, su);
            let d_calc = term.d_max_eff_hz * ut_s_u;
            let resid = d_calc - term.d_obs_hz;
            eb.rdc += 0.5 * w_rdc * resid * resid * term.inv_sigma2;

            let pref = w_rdc * resid * term.inv_sigma2 * (2.0 * term.d_max_eff_hz * inv_r);
            let grad_pb = v_scale(v_sub(su, v_scale(u, ut_s_u)), pref);
            add_seg_ft(
                &mut seg_force,
                &mut seg_torque,
                slots,
                term.seg_b,
                pb,
                grad_pb,
            );
            add_seg_ft(
                &mut seg_force,
                &mut seg_torque,
                slots,
                term.seg_a,
                pa,
                [-grad_pb[0], -grad_pb[1], -grad_pb[2]],
            );
        }
    }

    // 2. Translational + orientational cross-domain PCS energy & forces (including moving-tag torque!)
    if weights.w_pcs > 0.0 && !prep.pcs_terms.is_empty() {
        let w_pcs = weights.w_pcs;
        let mut tag_world: Vec<(Vec3, [[f64; 3]; 3])> = Vec::with_capacity(prep.pcs_tags.len());
        for tag in &prep.pcs_tags {
            let (orig_n, ref r_b) = blk_frames[tag.host_block_id];
            let m_pos = v_add(orig_n, mat3_mul_vec(r_b, tag.metal_local_pos));
            let dchi_w = mat3_conjugate_sym(r_b, &tag.dchi_local);
            tag_world.push((m_pos, dchi_w));
        }

        for term in &prep.pcs_terms {
            let (pa, oka) = slots[term.atom_slot];
            if !oka {
                continue;
            }
            let tag = &prep.pcs_tags[term.tag_idx];
            let (m_pos, ref dchi_w) = tag_world[term.tag_idx];
            let v = v_sub(pa, m_pos);
            let r2_raw = v_dot(v, v);
            let r2 = r2_raw + 0.25; // smooth 0.5 Å short-range regularization
            let inv_r = 1.0 / r2.sqrt();
            let inv_r2 = inv_r * inv_r;
            let inv_r5 = inv_r2 * inv_r2 * inv_r;
            let inv_r7 = inv_r5 * inv_r2;

            let mv = mat3_mul_vec(dchi_w, v);
            let v_mv = v_dot(v, mv);
            let pcs_calc = PCS_CARTESIAN_PREFACTOR * v_mv * inv_r5;
            let resid = pcs_calc - term.pcs_obs_eff_ppm;
            eb.pcs += 0.5 * w_pcs * resid * resid * term.inv_sigma2;

            let d_e_d_pcs = w_pcs * resid * term.inv_sigma2;
            let g_v = v_scale(
                v_sub(v_scale(mv, 2.0 * inv_r5), v_scale(v, 5.0 * v_mv * inv_r7)),
                d_e_d_pcs * PCS_CARTESIAN_PREFACTOR,
            );
            add_seg_ft(
                &mut seg_force,
                &mut seg_torque,
                slots,
                term.seg_idx,
                pa,
                g_v,
            );
            add_seg_ft(
                &mut seg_force,
                &mut seg_torque,
                slots,
                tag.host_seg_id,
                m_pos,
                [-g_v[0], -g_v[1], -g_v[2]],
            );

            // Pure rotational torque on host domain when host domain rotates in world frame (`2 * (M v) x v`)
            let rot_tq = v_scale(
                v_cross(mv, v),
                d_e_d_pcs * (2.0 * PCS_CARTESIAN_PREFACTOR * inv_r5),
            );
            seg_torque[tag.host_seg_id] = v_add(seg_torque[tag.host_seg_id], rot_tq);
        }
    }

    // 3. Exact unsoftened Solomon r^-6 PRE energy & forces (`21.W3:P0-4` & `21.W2:P2-9`)
    if weights.w_pre > 0.0 && !prep.pre_terms.is_empty() {
        let w_pre = weights.w_pre;
        let site_world: Vec<Vec3> = prep
            .pre_sites
            .iter()
            .map(|ps| {
                let (orig_n, ref r_b) = blk_frames[ps.host_block_id];
                v_add(orig_n, mat3_mul_vec(r_b, ps.site_local_pos))
            })
            .collect();

        for term in &prep.pre_terms {
            let (pa, oka) = slots[term.atom_slot];
            if !oka {
                continue;
            }
            let ps = &prep.pre_sites[term.site_idx];
            let s_pos = site_world[term.site_idx];
            let dv = v_sub(pa, s_pos);
            let r2_raw = v_dot(dv, dv);

            if term.is_bounds_mode {
                let d2 = r2_raw + 1e-4;
                let d = d2.sqrt();
                let (u_err, sign) = if d < term.lower_a {
                    (term.lower_a - d, -1.0)
                } else if d > term.upper_a {
                    (d - term.upper_a, 1.0)
                } else {
                    (0.0, 0.0)
                };
                if u_err > 0.0 {
                    eb.pre += 0.5 * w_pre * u_err * u_err * term.inv_sigma2;
                    let g = v_scale(dv, sign * w_pre * u_err * term.inv_sigma2 / d);
                    add_seg_ft(&mut seg_force, &mut seg_torque, slots, term.seg_idx, pa, g);
                    add_seg_ft(
                        &mut seg_force,
                        &mut seg_torque,
                        slots,
                        ps.host_seg_id,
                        s_pos,
                        [-g[0], -g[1], -g[2]],
                    );
                }
            } else {
                // Exact unsoftened `K_PRE * r^-6` for all `r >= 3.5 Å` (`C^1` core cap below `3.5 Å`)
                // with Huber cap `kappa = 8.0` + zero-bias outer basin guide `[lower_a, upper_a]`.
                let (g2_calc, dg2_ds) = solomon_r6_rate_and_ds(r2_raw, term.k_pre_a6_s1);
                let resid = g2_calc - term.gamma2_obs_s1;
                let inv_sig = term.inv_sigma2.sqrt();
                let z = resid * inv_sig;
                let kappa = 8.0_f64;
                let (huber_e, d_huber_dz) = if z.abs() <= kappa {
                    (0.5 * z * z, z)
                } else {
                    (kappa * (z.abs() - 0.5 * kappa), kappa * z.signum())
                };
                eb.pre += w_pre * huber_e;
                let d_e_dg2 = w_pre * d_huber_dz * inv_sig;
                let mut g_pa = v_scale(dv, d_e_dg2 * 2.0 * dg2_ds);

                let d = (r2_raw + 1e-4).sqrt();
                if d > term.upper_a {
                    let u = d - term.upper_a;
                    eb.pre += 0.5 * w_pre * u * u;
                    g_pa = v_add(g_pa, v_scale(dv, w_pre * u / d));
                } else if d < term.lower_a {
                    let u = term.lower_a - d;
                    eb.pre += 0.5 * w_pre * u * u;
                    g_pa = v_add(g_pa, v_scale(dv, -w_pre * u / d));
                }
                add_seg_ft(
                    &mut seg_force,
                    &mut seg_torque,
                    slots,
                    term.seg_idx,
                    pa,
                    g_pa,
                );
                add_seg_ft(
                    &mut seg_force,
                    &mut seg_torque,
                    slots,
                    ps.host_seg_id,
                    s_pos,
                    [-g_pa[0], -g_pa[1], -g_pa[2]],
                );
            }
        }
    }

    // 4. Inter-domain / linker NOE distance restraints with Nilges (1995) r^-6 sum-averaging (`21.W2:P2-11`)
    if weights.w_noe > 0.0 && !prep.noe_terms.is_empty() {
        let w_noe = weights.w_noe;
        let soft = weights.noe_soft_window.max(0.5);
        for term in &prep.noe_terms {
            let c_end = (term.cand_start + term.cand_len).min(prep.noe_candidates.len());
            if term.cand_start >= c_end {
                continue;
            }
            let cands = &prep.noe_candidates[term.cand_start..c_end];
            let mut s_min = f64::INFINITY;
            for cand in cands {
                let (pa, oka) = slots[cand.atom_a];
                let (pb, okb) = slots[cand.atom_b];
                if oka && okb {
                    let diff = v_sub(pa, pb);
                    let sc = v_dot(diff, diff) + 1e-6;
                    if sc < s_min {
                        s_min = sc;
                    }
                }
            }
            if !s_min.is_finite() {
                continue;
            }
            let mut tilde_s = 0.0_f64;
            for cand in cands {
                let (pa, oka) = slots[cand.atom_a];
                let (pb, okb) = slots[cand.atom_b];
                if oka && okb {
                    let diff = v_sub(pa, pb);
                    let sc = v_dot(diff, diff) + 1e-6;
                    let q = s_min / sc;
                    tilde_s += q * q * q;
                }
            }
            if tilde_s <= 0.0 {
                continue;
            }
            let d_eff = s_min.sqrt() / det_cbrt(tilde_s).sqrt();
            let (v_e, dv_dd) = if d_eff < term.lower_a {
                let u = term.lower_a - d_eff;
                (w_noe * u * u, -2.0 * w_noe * u)
            } else if d_eff <= term.upper_a {
                (0.0, 0.0)
            } else {
                let u = d_eff - term.upper_a;
                if u <= soft {
                    (w_noe * u * u, 2.0 * w_noe * u)
                } else {
                    (
                        w_noe * (soft * soft + 2.0 * soft * (u - soft)),
                        2.0 * w_noe * soft,
                    )
                }
            };
            eb.noe += v_e;
            if dv_dd != 0.0 {
                let denom = s_min * tilde_s;
                for cand in cands {
                    let (pa, oka) = slots[cand.atom_a];
                    let (pb, okb) = slots[cand.atom_b];
                    if !oka || !okb {
                        continue;
                    }
                    let diff = v_sub(pa, pb);
                    let sc = v_dot(diff, diff) + 1e-6;
                    let q = s_min / sc;
                    let w_c = q * q * q;
                    let g = v_scale(diff, dv_dd * (d_eff * w_c * q) / denom);
                    add_seg_ft(&mut seg_force, &mut seg_torque, slots, cand.seg_a, pa, g);
                    add_seg_ft(
                        &mut seg_force,
                        &mut seg_torque,
                        slots,
                        cand.seg_b,
                        pb,
                        [-g[0], -g[1], -g[2]],
                    );
                }
            }
        }
    }

    // 5. Grid-hashed heavy-atom quartic soft-core excluded volume
    if weights.w_vdw > 0.0 {
        eb.vdw = accumulate_grid_hashed_vdw_cpu(
            kin,
            slots,
            weights.w_vdw,
            weights.vdw_scale,
            |seg, pos, g| add_seg_ft(&mut seg_force, &mut seg_torque, slots, seg, pos, g),
        );
    }

    // 5b. Cyclic covalent peptide loop-closure restraint across discontinuous re-entry domain
    //     boundaries (`21.W2:P0-4`, `21F:B5`: always-on during docking/L-BFGS via `k_loop_override`
    //     regardless of `weights.w_rama`, while `None` matches `domain_dock.wgsl` for fixed-Saupe
    //     GPU parity!).
    let k_loop = match k_loop_override {
        Some(k) => k,
        None => {
            if weights.w_rama > 0.0 {
                25.0 * weights.w_rama
            } else {
                0.0
            }
        }
    };
    if k_loop > 0.0 {
        for blk in &kin.rigid_blocks {
            if blk.master_block_id == blk.block_id {
                continue;
            }
            let s_1 = blk.start_res;
            let (cs, _) = kin.chain_ranges[blk.chain_idx];
            if s_1 <= cs {
                continue;
            }
            let rg = &kin.residues[s_1];
            let prev_rg = &kin.residues[s_1 - 1];
            let n_prev = slots[7 * (s_1 - 1)].0;
            let ca_prev = slots[7 * (s_1 - 1) + 2].0;
            let c_prev = slots[7 * (s_1 - 1) + 4].0;
            let psi_prev = if prev_rg.psi_active {
                state.psi_rad[s_1 - 1]
            } else {
                kin.ref_psi_rad[s_1 - 1]
            };
            let omg_prev = state.omega_rad[s_1 - 1];
            let n_loop = nerf_place_atom(
                n_prev,
                ca_prev,
                c_prev,
                rg.r_c_prev_n,
                rg.ang_ca_prev_c_prev_n,
                psi_prev,
            );
            let ca_loop = nerf_place_atom(
                ca_prev,
                c_prev,
                n_loop,
                rg.r_n_ca,
                rg.ang_c_prev_n_ca,
                omg_prev,
            );
            let c_loop = nerf_place_atom(
                c_prev,
                n_loop,
                ca_loop,
                rg.r_ca_c,
                rg.ang_n_ca_c,
                kin.ref_phi_rad[s_1],
            );
            let n_rig = slots[7 * s_1].0;
            let ca_rig = slots[7 * s_1 + 2].0;
            let c_rig = slots[7 * s_1 + 4].0;
            let seg_loop = 2 * s_1;
            let seg_master = 2 * kin.rigid_blocks[blk.master_block_id].start_res + 1;
            for (p_lp, p_rg) in [(n_loop, n_rig), (ca_loop, ca_rig), (c_loop, c_rig)] {
                let df = v_sub(p_lp, p_rg);
                eb.rama += 0.5 * k_loop * v_dot(df, df);
                let g_lp = v_scale(df, k_loop);
                add_seg_ft(&mut seg_force, &mut seg_torque, slots, seg_loop, p_lp, g_lp);
                add_seg_ft(
                    &mut seg_force,
                    &mut seg_torque,
                    slots,
                    seg_master,
                    p_rg,
                    [-g_lp[0], -g_lp[1], -g_lp[2]],
                );
            }
        }
    }

    // 6. Local-pivot backward tree suffix-sum torque propagation -> linker `(grad_phi, grad_psi)`
    //    and multi-chain `(grad_chain_rot, grad_chain_trans)`
    let mut grad_phi = vec![0.0_f64; l];
    let mut grad_psi = vec![0.0_f64; l];
    let mut grad_chain_rot = vec![[0.0_f64; 3]; n_chains];
    let mut grad_chain_trans = vec![[0.0_f64; 3]; n_chains];

    for (c_idx, &(cs, ce)) in kin.chain_ranges.iter().enumerate().rev() {
        if cs >= ce {
            continue;
        }
        let mut suf_f = [0.0_f64; 3];
        let mut suf_t = [0.0_f64; 3];
        let mut cur_q = slots[7 * (ce - 1) + 4].0;

        for i_rev in (cs..ce).rev() {
            let rg = &kin.residues[i_rev];
            let ni = slots[7 * i_rev].0;
            let cai = slots[7 * i_rev + 2].0;
            let ci = slots[7 * i_rev + 4].0;

            if i_rev + 1 < ce {
                suf_t = v_add(suf_t, v_cross(v_sub(cur_q, ci), suf_f));
                cur_q = ci;
                let seg_psi = 2 * i_rev + 2;
                suf_f = v_add(suf_f, seg_force[seg_psi]);
                suf_t = v_add(suf_t, seg_torque[seg_psi]);
                if rg.psi_active {
                    let axis_psi = v_unit(v_sub(ci, cai));
                    grad_psi[i_rev] = v_dot(axis_psi, suf_t);
                }
            }

            suf_t = v_add(suf_t, v_cross(v_sub(cur_q, cai), suf_f));
            cur_q = cai;
            let seg_phi = 2 * i_rev + 1;
            suf_f = v_add(suf_f, seg_force[seg_phi]);
            suf_t = v_add(suf_t, seg_torque[seg_phi]);
            if rg.phi_active {
                let axis_phi = v_unit(v_sub(cai, ni));
                grad_phi[i_rev] = v_dot(axis_phi, suf_t);
            }

            // Re-entry rigid-block start (`master_block_id != rigid_block_id`): transfer the entire
            // downstream wrench directly onto `master_block_id`'s rigid segment (`2 * s_0 + 1`) so
            // intermediate linker torsions `[b_1.end .. s_1)` do not inherit rigid-domain forces!
            if rg.is_reentry_start {
                if let Some(m_bid) = rg.master_block_id {
                    let s_0 = kin.rigid_blocks[m_bid].start_res;
                    let seg_master = 2 * s_0 + 1;
                    let piv_master = slots[7 * s_0 + 2].0;
                    seg_force[seg_master] = v_add(seg_force[seg_master], suf_f);
                    seg_torque[seg_master] = v_add(
                        seg_torque[seg_master],
                        v_add(suf_t, v_cross(v_sub(cur_q, piv_master), suf_f)),
                    );
                    suf_f = [0.0; 3];
                    suf_t = [0.0; 3];
                }
            }

            // If `i_rev == cs` (chain root), include segment `2 * cs` (`N_{cs}, H_{cs}, CA_{cs}`)
            // around its exact local pivot `seg_local_pivot(slots, 2 * cs)` so partner chains
            // `c_idx >= 1` capture 100% of root-segment forces and torques!
            if i_rev == cs {
                let seg_root = 2 * cs;
                let piv_root = seg_local_pivot(slots, seg_root);
                suf_t = v_add(suf_t, v_cross(v_sub(cur_q, piv_root), suf_f));
                cur_q = piv_root;
                suf_f = v_add(suf_f, seg_force[seg_root]);
                suf_t = v_add(suf_t, seg_torque[seg_root]);
            }
        }

        if c_idx >= 1 {
            let c_world = v_add(kin.chain_ref_centroids[c_idx], state.chain_trans[c_idx]);
            let tq_center = v_add(suf_t, v_cross(v_sub(cur_q, c_world), suf_f));
            grad_chain_trans[c_idx] = suf_f;
            grad_chain_rot[c_idx] = tq_center;
        }
    }

    // 7. Lovell Ramachandran potential on active linker residues
    if weights.w_rama > 0.0 {
        let w_rama = weights.w_rama;
        for &i in &kin.active_linker_residues {
            let rg = &kin.residues[i];
            if !rg.rama_active {
                continue;
            }
            let phi_i = if rg.phi_active {
                state.phi_rad[i]
            } else {
                kin.ref_phi_rad[i]
            };
            let psi_i = if rg.psi_active {
                state.psi_rad[i]
            } else {
                kin.ref_psi_rad[i]
            };
            let (v_r, gp_r, gs_r) = dock_ramachandran_energy_and_grad(rg.res_class, phi_i, psi_i);
            eb.rama += w_rama * v_r;
            if rg.phi_active {
                grad_phi[i] += w_rama * gp_r;
            }
            if rg.psi_active {
                grad_psi[i] += w_rama * gs_r;
            }
        }
    }

    eb.total = eb.rdc + eb.pcs + eb.pre + eb.noe + eb.vdw + eb.rama;
    (
        eb,
        DomainDockGradient {
            grad_phi,
            grad_psi,
            grad_chain_rot,
            grad_chain_trans,
        },
    )
}

// ============================================================================
// 5. Observability Candor (`F8`), Multi-Start PAS/PCS/PRE Linker Seeding,
//    Host `f64` Two-Loop L-BFGS Polish, and `.pdb` / `.cif` / JSON / TSV I/O
// ============================================================================

/// Explicit observability classification carrying forward `F8` single-medium dipolar candor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DipolarObservabilityRegime {
    /// Only `PAE` spectral domain clustering performed (`0` experimental restraints supplied).
    UnrestrainedClusteringOnly,
    /// Single alignment medium (or coaxial media) without `PCS`, `PRE`, or inter-domain `NOE` restraints:
    /// intrinsic 4-fold PAS $\pi$-rotation cone degeneracy and zero direct translational sensitivity.
    UnderdeterminedSingleMediumRdcOnly,
    /// At least 2 non-coaxial alignment media paired with covalent linker & steric constraints,
    /// uniquely resolving relative 3D domain orientations while constraining translation indirectly via linker reach.
    MultiMediumRdcLinkerConstrained,
    /// Translational-plus-orientational paramagnetic `PCS` ($\Delta\boldsymbol{\chi}$, $r_{\text{M}}^{-3}$),
    /// `PRE` ($\Gamma_2$, $r_{\text{M}}^{-6}$), and/or inter-domain `NOE` restraints present,
    /// directly determining both 3D domain translation and orientation.
    FullyRestrainedTranslationalAndOrientational,
}

/// Pre-dock and observability quality gate verdict (`21.W2:P1-6` & `21.W2:P2-8`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DomainDockQualityVerdict {
    /// All pre-dock intra-domain coordinate fits and observability gates passed cleanly.
    PassedAllChecks,
    /// One or more domains exceeded `max_intra_domain_rdc_q_free` (`> 0.50`) or had `< 8` working
    /// intra-domain RDCs, permitted only because `allow_poor_intra_domain_fit == true`.
    IntraDomainMismatchWarning,
    /// Multi-domain docking proceeded on a single alignment medium (or coaxial media) without
    /// sufficient cross-domain translational restraints, permitted only because `allow_underdetermined_single_medium == true`.
    UnderdeterminedSingleMediumWarning,
    /// Both `IntraDomainMismatchWarning` and `UnderdeterminedSingleMediumWarning` were overridden.
    IntraDomainAndSingleMediumWarning,
}

/// Independent pre-dock intra-domain Saupe `Q_work` and held-out `ByPeptidePlane Q_free` check (`21.W2:P1-6`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PreDockDomainRdcCheck {
    pub domain_id: usize,
    pub medium_id: String,
    pub n_work: usize,
    pub n_free: usize,
    pub q_work: f64,
    pub q_free: f64,
    pub passed: bool,
}

/// Prepared input bundle produced by [`prepare_domain_dock_inputs_full`] and shared between CPU and WebGPU engines.
#[derive(Clone, Debug)]
pub struct PreparedDomainDockPipeline {
    pub domain_rdc_media: Vec<DomainRdcMediumSet>,
    pub pcs_tags: Vec<DomainPcsTagSet>,
    pub tolman_gdo: TolmanGdoMotionReport,
    pub prep: PreparedDomainDockRestraints,
    pub pre_dock_domain_checks: Vec<PreDockDomainRdcCheck>,
    pub quality_verdict: DomainDockQualityVerdict,
    pub extra_candor_notes: Vec<String>,
    pub n_excluded_flexible_rdcs: usize,
}

/// Configuration for multi-domain docking (`dock_domains_cpu` and `nmr-gpu::domain_dock_gpu`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DomainDockConfig {
    pub pae_cfg: PaeClusterConfig,
    pub weights: DomainDockEnergyWeights,
    /// Cross-validation `ByPeptidePlane` RDC holdout fraction (`default = 0.20`).
    pub rdc_free_fraction: f64,
    /// Whether to auto-partition `is_free` when all input RDCs have `is_free == false`.
    pub auto_partition_rdc_free: bool,
    /// Cross-validation domain-stratified `PCS` holdout fraction (`default = 0.20`, `21.W2:P1-5`).
    #[serde(default = "default_pcs_free_fraction")]
    pub pcs_free_fraction: f64,
    /// Whether to auto-partition `is_free` on `PCS` tags when all input PCS restraints have `is_free == false`.
    #[serde(default = "default_true_flag")]
    pub auto_partition_pcs_free: bool,
    /// Maximum permissible pre-dock intra-domain held-out `ByPeptidePlane Q_free` (`default = 0.50`, `21.W2:P1-6`).
    #[serde(default = "default_max_intra_domain_q_free")]
    pub max_intra_domain_rdc_q_free: f64,
    /// Override default refusal when a rigid domain fails its own pre-dock internal RDC check (`21.W2:P1-6`).
    #[serde(default)]
    pub allow_poor_intra_domain_fit: bool,
    /// Override default refusal when multi-domain docking is requested with `< 2` non-coaxial RDC media
    /// and insufficient cross-domain `PCS`/`PRE`/`NOE` translational restraints (`21.W2:P2-8`).
    #[serde(default)]
    pub allow_underdetermined_single_medium: bool,
    /// Backbone Cartesian coordinate noise floor (Å) included in Tolman $\vartheta_d$ MC error propagation (`default = 0.08`, `21.W2:A8`).
    #[serde(default = "default_tolman_coord_noise_a")]
    pub tolman_coord_noise_angstrom: f64,
    /// Optional global enable for 8-parameter lanthanide metal position + $\Delta\boldsymbol{\chi}$ VarPro refinement (`21.W2:P2-10`).
    #[serde(default)]
    pub refine_pcs_metal_pos: bool,
    /// Search radius (Å) around initial metal anchor for 8-parameter VarPro PCS refinement (`default = 8.0`).
    #[serde(default = "default_pcs_search_radius_a")]
    pub pcs_metal_search_radius_a: f64,
    /// Default rotational correlation time $\tau_c$ (ns) for Solomon PRE $\Gamma_2$ restraints (`21.W3:P0-4`).
    #[serde(default)]
    pub pre_tau_c_ns: Option<f64>,
    /// Default $^1\text{H}$ Larmor frequency (MHz) for Solomon–Bloembergen PRE dispersion (`21.W3:P0-4`).
    #[serde(default)]
    pub pre_proton_mhz: Option<f64>,
    /// Configurable flexible MTSL nitroxide tether allowance (Å, `default = 6.0` =
    /// `DEFAULT_MTSL_TETHER_ALLOWANCE_ANGSTROM` for `# site_anchor` spin labels; `# site_pos`
    /// explicit coordinates default to `0.0 Å` unless overridden per-site, `21F:B18`).
    #[serde(default = "default_mtsl_tether_allowance_a")]
    pub mtsl_tether_allowance_a: f64,
    /// Optional chain-explicit `DomainRdcMediumSet` list overriding `rdc_media_in` for multi-chain complexes (`21.W2:P1-8`).
    #[serde(default)]
    pub chain_rdc_media: Vec<DomainRdcMediumSet>,
    /// Number of docked conformers retained in the output ensemble (`default = 10`).
    pub ensemble_size: usize,
    /// Number of multi-start replicas (`<= 256` per WebGPU dispatch; `default = 64`).
    pub n_replicas: usize,
    /// Number of coarse multi-replica annealing / heavy-ball steps (`default = 56`).
    pub anneal_steps: usize,
    /// Maximum host `f64` two-loop `L-BFGS` polish iterations per selected conformer (`default = 140`).
    pub lbfgs_iters: usize,
    /// Number of parametric MC trials for Tolman $\vartheta_d / \vartheta_1$ GDO error bars (`default = 200`).
    pub tolman_mc_trials: usize,
    /// Deterministic PRNG seed.
    pub seed: u64,
}

fn default_pcs_free_fraction() -> f64 {
    0.20
}

fn default_true_flag() -> bool {
    true
}

fn default_max_intra_domain_q_free() -> f64 {
    0.50
}

fn default_tolman_coord_noise_a() -> f64 {
    DEFAULT_TOLMAN_COORD_NOISE_ANGSTROM
}

fn default_pcs_search_radius_a() -> f64 {
    8.0
}

fn default_mtsl_tether_allowance_a() -> f64 {
    DEFAULT_MTSL_TETHER_ALLOWANCE_ANGSTROM
}

impl Default for DomainDockConfig {
    fn default() -> Self {
        Self {
            pae_cfg: PaeClusterConfig::default(),
            weights: DomainDockEnergyWeights::default(),
            rdc_free_fraction: 0.20,
            auto_partition_rdc_free: true,
            pcs_free_fraction: 0.20,
            auto_partition_pcs_free: true,
            max_intra_domain_rdc_q_free: 0.50,
            allow_poor_intra_domain_fit: false,
            allow_underdetermined_single_medium: false,
            tolman_coord_noise_angstrom: DEFAULT_TOLMAN_COORD_NOISE_ANGSTROM,
            refine_pcs_metal_pos: false,
            pcs_metal_search_radius_a: 8.0,
            pre_tau_c_ns: None,
            pre_proton_mhz: None,
            mtsl_tether_allowance_a: DEFAULT_MTSL_TETHER_ALLOWANCE_ANGSTROM,
            chain_rdc_media: Vec::new(),
            ensemble_size: 10,
            n_replicas: 64,
            anneal_steps: 56,
            lbfgs_iters: 140,
            tolman_mc_trials: 200,
            seed: 42,
        }
    }
}

/// Single refined multi-domain conformer in the output ensemble.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DockedConformer {
    pub model_index: usize,
    pub state: DomainDockState,
    pub energy: DomainDockEnergyBreakdown,
    pub max_intra_domain_rmsd_angstrom: f64,
    /// Maximum peptide-bond `|d(C, N) - r_c_prev_n|` deviation (Å) at discontinuous insertion-domain
    /// re-entry boundaries (`0.0` when all domains are contiguous, `21F:B5`).
    #[serde(default)]
    pub max_reentry_cn_err_a: f64,
}

/// Per-domain cross-validated quality summary on Model 1.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PerDomainValidationSummary {
    pub domain_id: usize,
    pub n_residues: usize,
    pub intra_domain_distortion_rmsd_angstrom: f64,
    pub pre_dock_rdc_q_work: f64,
    pub pre_dock_rdc_q_free: f64,
    pub rdc_q_work: f64,
    pub rdc_q_free: f64,
}

/// Structured multi-domain docking diagnostic and validation report.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DomainDockReport {
    pub engine_used: String,
    pub quality_verdict: DomainDockQualityVerdict,
    pub partition: PaeDomainPartition,
    pub tolman_gdo: TolmanGdoMotionReport,
    pub observability_regime: DipolarObservabilityRegime,
    pub candor_notes: Vec<String>,
    pub pre_dock_domain_checks: Vec<PreDockDomainRdcCheck>,
    pub n_excluded_flexible_rdcs: usize,
    pub rdc_stratification: RdcFreeStratification,
    pub global_rdc_q_work: f64,
    pub global_rdc_q_free: f64,
    pub pcs_q_intra: f64,
    /// Working (`!is_free`) cross-domain PCS Cornilescu Q-factor (`None` when `n_pcs_work == 0`, `21.W2:P1-5`).
    pub pcs_q_work: Option<f64>,
    /// Strictly held-out (`is_free == true`) cross-domain PCS Cornilescu Q-factor — NEVER falls back
    /// to `pcs_q_work` (`None` when `n_pcs_free == 0`, `21.W2:P1-5`).
    pub pcs_q_free: Option<f64>,
    pub n_pcs_work: usize,
    pub n_pcs_free: usize,
    pub max_intra_domain_rmsd_angstrom: f64,
    /// Maximum peptide-bond `|d(C, N) - r_c_prev_n|` deviation (Å) across discontinuous
    /// insertion-domain re-entry boundaries on Model 1 (`0.0` when all domains are contiguous, `21F:B5`).
    #[serde(default)]
    pub max_reentry_cn_err_a: f64,
    pub ensemble_pairwise_bb_rmsd_angstrom: f64,
    pub per_domain_validation: Vec<PerDomainValidationSummary>,
    pub rdc_validation: Option<RdcValidationSummary>,
    pub best_energy: DomainDockEnergyBreakdown,
    pub elapsed_ms: f64,
}

/// Full output bundle from `dock_domains_cpu` / `dock_domains_gpu`.
#[derive(Clone, Debug)]
pub struct DomainDockResult {
    pub kinematics: DomainLinkerKinematics,
    pub conformers: Vec<DockedConformer>,
    pub chains: Vec<PdbChain>,
    pub report: DomainDockReport,
}

/// Non-coaxial Saupe tensor cosine threshold (`|cos(S_a, S_b)| < 0.85` <=> `>= ~32°` principal-axis separation, `21.W2:P2-8`).
pub const MAX_COAXIAL_SAUPE_COS_SIM: f64 = 0.85;

/// Classify dipolar/paramagnetic observability using genuine compiled cross-domain working restraint
/// counts and a strict non-coaxial Saupe threshold (`|cos| < 0.85`, `21.W2:P2-8`).
pub fn classify_dipolar_observability(
    partition: &PaeDomainPartition,
    rdc_media: &[RdcMediumSet],
    pcs_tags: &[DomainPcsTagSet],
    pre_sites: &[DomainPreSiteSet],
    noe_terms_count: usize,
    chain_for_saupe: &PdbChain,
) -> (DipolarObservabilityRegime, Vec<String>) {
    let dom_media = DomainRdcMediumSet::from_rdc_media(rdc_media);
    let cseq = ChainSeqIndex::from_pdb_chain(chain_for_saupe).unwrap_or_else(|_| ChainSeqIndex {
        exact: BTreeMap::new(),
        by_seq_only: BTreeMap::new(),
    });
    // Estimate cross-domain working PCS/PRE counts when called standalone without `prep`
    let mut approx_cross_pcs = 0usize;
    for t in pcs_tags {
        let hd = t.host_domain_id.unwrap_or(1);
        for r in &t.restraints {
            if !r.is_free {
                if let Ok(Some(idx)) = cseq.resolve(r.chain_id.as_deref(), r.seq_id) {
                    if partition.domain_labels.get(idx).copied().unwrap_or(0) != hd {
                        approx_cross_pcs += 1;
                    }
                }
            }
        }
    }
    let mut approx_cross_pre = 0usize;
    for s in pre_sites {
        let hd = s.host_domain_id.unwrap_or(1);
        for r in &s.restraints {
            if let Ok(Some(idx)) = cseq.resolve(r.chain_id.as_deref(), r.seq_id) {
                if partition.domain_labels.get(idx).copied().unwrap_or(0) != hd {
                    approx_cross_pre += 1;
                }
            }
        }
    }
    classify_dipolar_observability_multi(
        partition,
        &dom_media,
        approx_cross_pcs,
        approx_cross_pre,
        noe_terms_count,
        chain_for_saupe,
        &cseq,
    )
}

pub(crate) fn classify_dipolar_observability_multi(
    partition: &PaeDomainPartition,
    rdc_media: &[DomainRdcMediumSet],
    n_cross_pcs_work: usize,
    n_cross_pre_work: usize,
    n_cross_noe_work: usize,
    chain_for_saupe: &PdbChain,
    cseq: &ChainSeqIndex,
) -> (DipolarObservabilityRegime, Vec<String>) {
    let mut notes = Vec::new();
    let any_rdc = rdc_media.iter().any(|m| !m.restraints.is_empty());
    let any_dist = n_cross_pcs_work > 0 || n_cross_pre_work > 0 || n_cross_noe_work > 0;

    if !any_rdc && !any_dist {
        if partition.domains.len() >= 2 {
            notes.push(
                "PAE spectral domain clustering identified multiple domains, but no experimental cross-domain RDC, PCS, PRE, or NOE restraints were supplied; coordinates retain the input AlphaFold inter-domain pose."
                    .to_string(),
            );
        }
        return (
            DipolarObservabilityRegime::UnrestrainedClusteringOnly,
            notes,
        );
    }

    // Require sufficient non-degenerate 3D cross-domain distance+orientation restraints (`>= 5` cross-domain
    // working PCSs, `>= 3` cross-domain PREs, or `>= 3` inter-domain NOEs; `21.W2:P2-8`) before declaring
    // `FullyRestrainedTranslationalAndOrientational` (an intra-domain-only PCS tag or a single PRE distance
    // cannot lift 6-DOF rigid-body / 4-fold Saupe cone degeneracy!).
    let sufficient_trans_orient = n_cross_pcs_work >= 5
        || n_cross_pre_work >= 3
        || n_cross_noe_work >= 3
        || (n_cross_pcs_work + n_cross_pre_work + n_cross_noe_work >= 4);
    if sufficient_trans_orient {
        return (
            DipolarObservabilityRegime::FullyRestrainedTranslationalAndOrientational,
            notes,
        );
    }

    // Check effective independent non-coaxial alignment media (`|cos(S_a, S_b)| < 0.85`, `21.W2:P2-8`)
    let mut fitted_s5: Vec<[f64; 5]> = Vec::new();
    for m in rdc_media {
        if let Ok((fit, _)) = fit_saupe_tensor_domain_rdc(chain_for_saupe, cseq, m) {
            fitted_s5.push(fit.s_5);
        }
    }
    let has_two_non_coaxial = if fitted_s5.len() >= 2 {
        let mut found = false;
        for a in 0..fitted_s5.len() {
            for b in (a + 1)..fitted_s5.len() {
                let sa = &fitted_s5[a];
                let sb = &fitted_s5[b];
                let mat_a = [
                    [-(sa[0] + sa[1]), sa[2], sa[3]],
                    [sa[2], sa[0], sa[4]],
                    [sa[3], sa[4], sa[1]],
                ];
                let mat_b = [
                    [-(sb[0] + sb[1]), sb[2], sb[3]],
                    [sb[2], sb[0], sb[4]],
                    [sb[3], sb[4], sb[1]],
                ];
                let mut dot = 0.0_f64;
                let mut na = 0.0_f64;
                let mut nb = 0.0_f64;
                for r in 0..3 {
                    for c in 0..3 {
                        dot += mat_a[r][c] * mat_b[r][c];
                        na += mat_a[r][c] * mat_a[r][c];
                        nb += mat_b[r][c] * mat_b[r][c];
                    }
                }
                let cos_sim = dot.abs() / (na * nb).max(1e-30).sqrt();
                if cos_sim < MAX_COAXIAL_SAUPE_COS_SIM {
                    found = true;
                    break;
                }
            }
            if found {
                break;
            }
        }
        found
    } else {
        false
    };

    if !has_two_non_coaxial {
        notes.push(format!(
            "Candor note [F8 single-medium cone degeneracy]: fewer than 2 non-coaxial RDC alignment media (|cos| < {MAX_COAXIAL_SAUPE_COS_SIM:.2}) were supplied without sufficient cross-domain PCS ({n_cross_pcs_work}, need >= 5), PRE ({n_cross_pre_work}, need >= 3), or inter-domain NOE ({n_cross_noe_work}, need >= 3) restraints. A single Saupe alignment tensor carries intrinsic 4-fold principal-axis pi-rotation cone degeneracy and zero direct translational sensitivity."
        ));
        (
            DipolarObservabilityRegime::UnderdeterminedSingleMediumRdcOnly,
            notes,
        )
    } else {
        notes.push(
            "Candor note [multi-medium RDCs without long-range distance restraints]: >= 2 non-coaxial alignment media resolve 4-fold rotational PAS degeneracy across rigid domains, while inter-domain 3D translation is bounded indirectly by covalent linker geometry and steric excluded volume rather than direct PCS/PRE/NOE distance restraints."
                .to_string(),
        );
        (
            DipolarObservabilityRegime::MultiMediumRdcLinkerConstrained,
            notes,
        )
    }
}

/// Generate `n_replicas` physically diverse covalent starting states (`0` `is_free` leakage!)
/// combining:
/// - Replica 0: exact input AlphaFold pose,
/// - Analytical 8-branch $\mathcal{D}_2$ Saupe PAS relative rotations + `PCS`/`PRE`/`NOE` 3D placement
///   closed onto the exact covalent `NeRF` linker manifold via target-triad `L-BFGS` (including
///   re-entry linker closure for discontinuous multi-span domains!), and
/// - Quasi-uniform $\text{SO}(3)$ / Ramachandran basin torsion exploration.
pub fn generate_domain_dock_seed_states(
    kin: &DomainLinkerKinematics,
    partition: &PaeDomainPartition,
    prep: &PreparedDomainDockRestraints,
    weights: &DomainDockEnergyWeights,
    n_replicas: usize,
    seed: u64,
) -> Vec<DomainDockState> {
    let target_r = n_replicas.max(1);
    let base_state = kin.initial_state();
    let mut seeds = Vec::with_capacity(target_r);
    seeds.push(base_state.clone());
    if target_r == 1
        || (kin.active_linker_residues.is_empty() && kin.chain_ranges.len() <= 1)
        || partition.domains.len() <= 1
    {
        while seeds.len() < target_r {
            seeds.push(base_state.clone());
        }
        return seeds;
    }

    let mut rng = DetRng::new(seed ^ 0x444F_434B_5345_4544);
    let (init_slots, init_blk_frames) = kin.forward_slots_and_block_frames(&base_state);

    let n_blocks = kin.rigid_blocks.len();
    if n_blocks >= 2 {
        let b0 = &kin.rigid_blocks[0];
        let mut cand_quats_b1: Vec<[[f64; 3]; 3]> =
            vec![[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]];

        const PROPER_SYM_8: [[[f64; 3]; 3]; 8] = [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
            [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
            [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
            [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
        ];

        for m_idx in 0..prep.n_media {
            let s_b0 = fit_block_work_saupe(kin, prep, &init_slots, 0, m_idx);
            let s_b1 = fit_block_work_saupe(kin, prep, &init_slots, 1, m_idx);
            if let (Some(mat0), Some(mat1)) = (s_b0, s_b1) {
                let (_, rot0, _) = decompose_saupe_matrix(mat0);
                let (_, rot1, _) = decompose_saupe_matrix(mat1);
                let mut rot1_t = [[0.0_f64; 3]; 3];
                for r in 0..3 {
                    for c in 0..3 {
                        rot1_t[r][c] = rot1[c][r];
                    }
                }
                for sym in &PROPER_SYM_8 {
                    let q_mat = mat3_mul(&mat3_mul(&rot0, sym), &rot1_t);
                    cand_quats_b1.push(q_mat);
                }
            }
        }

        for _ in 0..24 {
            let u1 = rng.f64();
            let u2 = rng.f64() * 2.0 * PI;
            let u3 = rng.f64() * 2.0 * PI;
            let sq1 = (1.0 - u1).max(0.0).sqrt();
            let sq2 = u1.max(0.0).sqrt();
            let (s2, c2) = det_sin_cos(u2);
            let (s3, c3) = det_sin_cos(u3);
            let q = [sq1 * s2, sq1 * c2, sq2 * s3, sq2 * c3];
            cand_quats_b1.push(quat_to_rot_matrix(q));
        }

        let mut scored_rots: Vec<(f64, [[f64; 3]; 3])> = Vec::with_capacity(cand_quats_b1.len());
        for q_mat in cand_quats_b1 {
            let rdc_score = score_block1_rotation_work_rdc(kin, prep, &init_slots, &q_mat);
            scored_rots.push((rdc_score, q_mat));
        }
        scored_rots.sort_by(|a, b| a.0.total_cmp(&b.0));

        let mut top_rots: Vec<[[f64; 3]; 3]> = Vec::new();
        for (_, q_mat) in scored_rots {
            let dup = top_rots.iter().any(|prev| {
                let mut tr = 0.0_f64;
                for r in 0..3 {
                    for c in 0..3 {
                        tr += prev[r][c] * q_mat[r][c];
                    }
                }
                tr > 2.985
            });
            if !dup {
                top_rots.push(q_mat);
                if top_rots.len() >= 14 {
                    break;
                }
            }
        }

        let b1 = &kin.rigid_blocks[1];
        let same_chain_01 = b0.chain_idx == b1.chain_idx;
        let anchor_c_prev = init_slots[7 * (b0.end_res - 1) + 4].0;
        let init_n_b1 = init_slots[7 * b1.start_res].0;
        let n_link_res = if same_chain_01 {
            b1.start_res.saturating_sub(b0.end_res)
        } else {
            0
        };
        let max_reach = if same_chain_01 {
            3.50 * (n_link_res as f64) + 1.45
        } else {
            80.0
        };

        let n_pose_budget = (target_r * 3 / 4).max(4).min(target_r - 1);
        let seeds_per_rot = (n_pose_budget / top_rots.len().max(1)).clamp(1, 4);

        for q_mat in &top_rots {
            if seeds.len() > n_pose_budget {
                break;
            }
            let target_n_b1 = optimize_block1_translation_target(
                kin,
                prep,
                &init_slots,
                &init_blk_frames,
                q_mat,
                anchor_c_prev,
                init_n_b1,
                max_reach,
                weights,
            );

            if !same_chain_01 {
                let c_idx = b1.chain_idx;
                let c_af = kin.chain_ref_centroids[c_idx];
                let q_unit = rot_matrix_to_quat(q_mat);
                let base_t = v_sub(
                    v_sub(target_n_b1, c_af),
                    mat3_mul_vec(q_mat, v_sub(init_n_b1, c_af)),
                );
                for sub in 0..seeds_per_rot {
                    if seeds.len() > n_pose_budget {
                        break;
                    }
                    let mut st = base_state.clone();
                    if sub == 0 {
                        st.chain_quats[c_idx] = q_unit;
                        st.chain_trans[c_idx] = base_t;
                    } else {
                        let d_omg = [
                            rng.normal(0.0, 0.08),
                            rng.normal(0.0, 0.08),
                            rng.normal(0.0, 0.08),
                        ];
                        st.chain_quats[c_idx] = quat_apply_world_rot_step(q_unit, d_omg);
                        st.chain_trans[c_idx] = [
                            base_t[0] + rng.normal(0.0, 1.0),
                            base_t[1] + rng.normal(0.0, 1.0),
                            base_t[2] + rng.normal(0.0, 1.0),
                        ];
                    }
                    seeds.push(st);
                }
            } else if n_link_res > 0 {
                let (_, ref r_b1_init) = init_blk_frames[1];
                let r_b1_target = mat3_mul(q_mat, r_b1_init);
                for sub in 0..seeds_per_rot {
                    if seeds.len() > n_pose_budget {
                        break;
                    }
                    let closed = close_covalent_linker_segment(
                        kin,
                        &base_state,
                        b0.end_res,
                        b1.start_res,
                        target_n_b1,
                        &r_b1_target,
                        sub,
                        &mut rng,
                    );
                    seeds.push(closed);
                }
            }
        }

        // Downstream blocks `b_idx >= 2`: close re-entry linkers onto `master_block_id`'s exact
        // rigid peptide bond (`21.W2:P0-4`), or diversify independent downstream blocks via PAS!
        if n_blocks >= 3 && seeds.len() > 1 {
            let cur_len = seeds.len();
            for s_idx in 1..cur_len {
                for b_idx in 2..n_blocks {
                    let b_prev = &kin.rigid_blocks[b_idx - 1];
                    let b_cur = &kin.rigid_blocks[b_idx];
                    if b_prev.chain_idx != b_cur.chain_idx || b_cur.start_res <= b_prev.end_res {
                        continue;
                    }
                    let (cur_slots, cur_frames) = kin.forward_slots_and_block_frames(&seeds[s_idx]);
                    if b_cur.master_block_id != b_cur.block_id {
                        let s_re = b_cur.start_res;
                        let n_re = cur_slots[7 * s_re].0;
                        let (_, ref r_master) = cur_frames[b_cur.master_block_id];
                        let closed = close_covalent_linker_segment(
                            kin,
                            &seeds[s_idx],
                            b_prev.end_res,
                            s_re,
                            n_re,
                            r_master,
                            s_idx % 3,
                            &mut rng,
                        );
                        seeds[s_idx] = closed;
                    } else if let (Some(m0), Some(m_cur)) = (
                        fit_block_work_saupe(kin, prep, &cur_slots, 0, 0),
                        fit_block_work_saupe(kin, prep, &cur_slots, b_idx, 0),
                    ) {
                        let (_, r0, _) = decompose_saupe_matrix(m0);
                        let (_, rc, _) = decompose_saupe_matrix(m_cur);
                        let mut rc_t = [[0.0; 3]; 3];
                        for r in 0..3 {
                            for c in 0..3 {
                                rc_t[r][c] = rc[c][r];
                            }
                        }
                        let sym = &PROPER_SYM_8[s_idx % 4];
                        let q_rel = mat3_mul(&mat3_mul(&r0, sym), &rc_t);
                        let r_target = mat3_mul(&q_rel, &cur_frames[b_idx].1);
                        let anchor_c = cur_slots[7 * (b_prev.end_res - 1) + 4].0;
                        let cur_n = cur_slots[7 * b_cur.start_res].0;
                        let closed = close_covalent_linker_segment(
                            kin,
                            &seeds[s_idx],
                            b_prev.end_res,
                            b_cur.start_res,
                            v_add(anchor_c, v_scale(v_sub(cur_n, anchor_c), 0.85)),
                            &r_target,
                            0,
                            &mut rng,
                        );
                        seeds[s_idx] = closed;
                    }
                }
            }
        }
    }

    const BASIN_CENTERS_DEG: &[(f64, f64)] = &[
        (-63.0, -42.0),
        (-122.0, 135.0),
        (-75.0, 145.0),
        (-85.0, -12.0),
        (56.0, 38.0),
    ];
    while seeds.len() < target_r {
        let mut st = base_state.clone();
        let mode = seeds.len() % 3;
        for &i in &kin.active_linker_residues {
            let rg = &kin.residues[i];
            if mode == 0 {
                if rg.phi_active {
                    st.phi_rad[i] = wrap_angle_rad(st.phi_rad[i] + rng.normal(0.0, 0.45));
                }
                if rg.psi_active {
                    st.psi_rad[i] = wrap_angle_rad(st.psi_rad[i] + rng.normal(0.0, 0.45));
                }
            } else {
                let (bp, bs) = BASIN_CENTERS_DEG[rng.range_usize(0, BASIN_CENTERS_DEG.len())];
                if rg.phi_active {
                    st.phi_rad[i] = wrap_angle_rad(bp.to_radians() + rng.normal(0.0, 0.25));
                }
                if rg.psi_active {
                    st.psi_rad[i] = wrap_angle_rad(bs.to_radians() + rng.normal(0.0, 0.25));
                }
            }
        }
        for c_idx in 1..kin.chain_ranges.len() {
            let d_omg = [
                rng.normal(0.0, 0.6),
                rng.normal(0.0, 0.6),
                rng.normal(0.0, 0.6),
            ];
            st.chain_quats[c_idx] = quat_apply_world_rot_step(st.chain_quats[c_idx], d_omg);
            st.chain_trans[c_idx] = [
                rng.normal(0.0, 5.0),
                rng.normal(0.0, 5.0),
                rng.normal(0.0, 5.0),
            ];
        }
        seeds.push(st);
    }

    seeds
}

fn fit_block_work_saupe(
    kin: &DomainLinkerKinematics,
    prep: &PreparedDomainDockRestraints,
    slots: &[(Vec3, bool)],
    block_id: usize,
    medium_idx: usize,
) -> Option<[[f64; 3]; 3]> {
    let master_id = kin.rigid_blocks.get(block_id)?.master_block_id;
    let mut qr = DockGivensQr5::new();
    for t in &prep.rdc_terms {
        if t.medium_idx != medium_idx {
            continue;
        }
        let ra = t.atom_a / 7;
        let rb = t.atom_b / 7;
        if kin.residues[ra].master_block_id != Some(master_id)
            || kin.residues[rb].master_block_id != Some(master_id)
        {
            continue;
        }
        let (pa, oka) = slots[t.atom_a];
        let (pb, okb) = slots[t.atom_b];
        if !oka || !okb {
            continue;
        }
        let dv = v_sub(pb, pa);
        let dist = v_norm(dv);
        if dist < 1e-6 {
            continue;
        }
        let inv_d = 1.0 / dist;
        let (ux, uy, uz) = (dv[0] * inv_d, dv[1] * inv_d, dv[2] * inv_d);
        let ws = t.d_max_eff_hz * t.inv_sigma;
        qr.push_row(
            [
                ws * (uy * uy - ux * ux),
                ws * (uz * uz - ux * ux),
                ws * (2.0 * ux * uy),
                ws * (2.0 * ux * uz),
                ws * (2.0 * uy * uz),
            ],
            t.d_obs_hz * t.inv_sigma,
        );
    }
    qr.solve_saupe()
}

fn score_block1_rotation_work_rdc(
    kin: &DomainLinkerKinematics,
    prep: &PreparedDomainDockRestraints,
    init_slots: &[(Vec3, bool)],
    q_b1: &[[f64; 3]; 3],
) -> f64 {
    if prep.n_media == 0 || prep.rdc_terms.is_empty() {
        return 0.0;
    }
    let m0 = kin.rigid_blocks[0].master_block_id;
    let m1 = kin.rigid_blocks[1].master_block_id;
    let b1 = &kin.rigid_blocks[1];
    let n_b1 = init_slots[7 * b1.start_res].0;

    let mut chi2 = 0.0_f64;
    for m_idx in 0..prep.n_media {
        let mut qr = DockGivensQr5::new();
        let mut unit_vecs: Vec<(Vec3, f64, f64, f64)> = Vec::new();
        for t in &prep.rdc_terms {
            if t.medium_idx != m_idx {
                continue;
            }
            let ra = t.atom_a / 7;
            let rb = t.atom_b / 7;
            let in_b0 = kin.residues[ra].master_block_id == Some(m0)
                && kin.residues[rb].master_block_id == Some(m0);
            let in_b1 = m1 != m0
                && kin.residues[ra].master_block_id == Some(m1)
                && kin.residues[rb].master_block_id == Some(m1);
            if !in_b0 && !in_b1 {
                continue;
            }
            let (mut pa, oka) = init_slots[t.atom_a];
            let (mut pb, okb) = init_slots[t.atom_b];
            if !oka || !okb {
                continue;
            }
            if in_b1 {
                pa = v_add(n_b1, mat3_mul_vec(q_b1, v_sub(pa, n_b1)));
                pb = v_add(n_b1, mat3_mul_vec(q_b1, v_sub(pb, n_b1)));
            }
            let dv = v_sub(pb, pa);
            let dist = v_norm(dv);
            if dist < 1e-6 {
                continue;
            }
            let u = v_scale(dv, 1.0 / dist);
            let ws = t.d_max_eff_hz * t.inv_sigma;
            qr.push_row(
                [
                    ws * (u[1] * u[1] - u[0] * u[0]),
                    ws * (u[2] * u[2] - u[0] * u[0]),
                    ws * (2.0 * u[0] * u[1]),
                    ws * (2.0 * u[0] * u[2]),
                    ws * (2.0 * u[1] * u[2]),
                ],
                t.d_obs_hz * t.inv_sigma,
            );
            unit_vecs.push((u, t.d_obs_hz, t.d_max_eff_hz, t.inv_sigma2));
        }
        if let Some(s_mat) = qr.solve_saupe() {
            for (u, d_obs, d_max, inv_s2) in unit_vecs {
                let dc = calc_rdc_from_unit_vec(u, d_max, &s_mat);
                let df = dc - d_obs;
                chi2 += 0.5 * df * df * inv_s2;
            }
        }
    }
    chi2
}

#[allow(clippy::too_many_arguments)]
fn optimize_block1_translation_target(
    kin: &DomainLinkerKinematics,
    prep: &PreparedDomainDockRestraints,
    init_slots: &[(Vec3, bool)],
    init_blk_frames: &[(Vec3, [[f64; 3]; 3])],
    q_b1: &[[f64; 3]; 3],
    anchor_c_prev: Vec3,
    init_n_b1: Vec3,
    max_reach: f64,
    weights: &DomainDockEnergyWeights,
) -> Vec3 {
    if prep.pcs_terms.is_empty() && prep.pre_terms.is_empty() && prep.noe_terms.is_empty() {
        let cur_v = v_sub(init_n_b1, anchor_c_prev);
        let cur_d = v_norm(cur_v);
        if cur_d <= max_reach * 0.85 && cur_d >= 3.5 {
            return init_n_b1;
        }
        let dir = if cur_d > 1e-6 {
            v_scale(cur_v, 1.0 / cur_d)
        } else {
            [1.0, 0.0, 0.0]
        };
        return v_add(anchor_c_prev, v_scale(dir, (max_reach * 0.65).max(4.5)));
    }

    let m0 = kin.rigid_blocks[0].master_block_id;
    let m1 = kin.rigid_blocks[1].master_block_id;
    let ca_b0: Vec<Vec3> = kin
        .residues
        .iter()
        .enumerate()
        .filter(|(_, r)| r.master_block_id == Some(m0))
        .map(|(j, _)| init_slots[7 * j + 2].0)
        .collect();
    let ca_b1_rot: Vec<Vec3> = kin
        .residues
        .iter()
        .enumerate()
        .filter(|(_, r)| m1 != m0 && r.master_block_id == Some(m1))
        .map(|(j, _)| mat3_mul_vec(q_b1, v_sub(init_slots[7 * j + 2].0, init_n_b1)))
        .collect();

    let eval_trans = |n_pos: Vec3| -> (f64, Vec3) {
        let mut e = 0.0_f64;
        let mut g = [0.0_f64; 3];

        let dv_reach = v_sub(n_pos, anchor_c_prev);
        let d_reach = (v_dot(dv_reach, dv_reach) + 1e-6).sqrt();
        let soft_reach = max_reach * 0.92;
        if d_reach > soft_reach {
            let ex = d_reach - soft_reach;
            e += 12.0 * ex * ex;
            g = v_add(g, v_scale(dv_reach, 24.0 * ex / d_reach));
        }

        for &c0 in &ca_b0 {
            for &rel1 in &ca_b1_rot {
                let c1 = v_add(n_pos, rel1);
                let df = v_sub(c1, c0);
                let d2 = v_dot(df, df);
                if d2 < 21.16 {
                    let delta = 21.16 - d2;
                    e += 0.5 * delta * delta;
                    g = v_add(g, v_scale(df, -2.0 * delta));
                }
            }
        }

        if weights.w_pcs > 0.0 {
            for term in &prep.pcs_terms {
                let res_a = term.atom_slot / 7;
                let tag = &prep.pcs_tags[term.tag_idx];
                let a_in_b1 = m1 != m0 && kin.residues[res_a].master_block_id == Some(m1);
                let a_in_b0 = kin.residues[res_a].master_block_id == Some(m0);
                if !(a_in_b1 && tag.host_block_id == m0) && !(a_in_b0 && tag.host_block_id == m1) {
                    continue;
                }
                let (pa, m_pos, dchi_w, sign_n) = if a_in_b1 {
                    let rel_a = mat3_mul_vec(q_b1, v_sub(init_slots[term.atom_slot].0, init_n_b1));
                    let m0_p = v_add(
                        init_blk_frames[m0].0,
                        mat3_mul_vec(&init_blk_frames[m0].1, tag.metal_local_pos),
                    );
                    let dw0 = mat3_conjugate_sym(&init_blk_frames[m0].1, &tag.dchi_local);
                    (v_add(n_pos, rel_a), m0_p, dw0, 1.0)
                } else {
                    let pa0 = init_slots[term.atom_slot].0;
                    let r_b1_w = mat3_mul(q_b1, &init_blk_frames[m1].1);
                    let m1_p = v_add(n_pos, mat3_mul_vec(&r_b1_w, tag.metal_local_pos));
                    let dw1 = mat3_conjugate_sym(&r_b1_w, &tag.dchi_local);
                    (pa0, m1_p, dw1, -1.0)
                };
                let v = v_sub(pa, m_pos);
                let r2 = v_dot(v, v) + 0.25;
                let inv_r = 1.0 / r2.sqrt();
                let inv_r2 = inv_r * inv_r;
                let inv_r5 = inv_r2 * inv_r2 * inv_r;
                let inv_r7 = inv_r5 * inv_r2;
                let mv = mat3_mul_vec(&dchi_w, v);
                let v_mv = v_dot(v, mv);
                let pcs_calc = PCS_CARTESIAN_PREFACTOR * v_mv * inv_r5;
                let resid = pcs_calc - term.pcs_obs_eff_ppm;
                e += 0.5 * weights.w_pcs * resid * resid * term.inv_sigma2;
                let de = weights.w_pcs * resid * term.inv_sigma2 * sign_n * PCS_CARTESIAN_PREFACTOR;
                let gv = v_scale(
                    v_sub(v_scale(mv, 2.0 * inv_r5), v_scale(v, 5.0 * v_mv * inv_r7)),
                    de,
                );
                g = v_add(g, gv);
            }
        }

        if weights.w_pre > 0.0 {
            for term in &prep.pre_terms {
                let res_a = term.atom_slot / 7;
                let ps = &prep.pre_sites[term.site_idx];
                let a_in_b1 = m1 != m0 && kin.residues[res_a].master_block_id == Some(m1);
                let a_in_b0 = kin.residues[res_a].master_block_id == Some(m0);
                if !(a_in_b1 && ps.host_block_id == m0) && !(a_in_b0 && ps.host_block_id == m1) {
                    continue;
                }
                let (pa, s_pos, sign_n) = if a_in_b1 {
                    let rel_a = mat3_mul_vec(q_b1, v_sub(init_slots[term.atom_slot].0, init_n_b1));
                    let s0 = v_add(
                        init_blk_frames[m0].0,
                        mat3_mul_vec(&init_blk_frames[m0].1, ps.site_local_pos),
                    );
                    (v_add(n_pos, rel_a), s0, 1.0)
                } else {
                    let pa0 = init_slots[term.atom_slot].0;
                    let r_b1_w = mat3_mul(q_b1, &init_blk_frames[m1].1);
                    let s1 = v_add(n_pos, mat3_mul_vec(&r_b1_w, ps.site_local_pos));
                    (pa0, s1, -1.0)
                };
                let dv = v_sub(pa, s_pos);
                let r2_raw = v_dot(dv, dv);
                let d = (r2_raw + 1e-4).sqrt();
                if term.is_bounds_mode {
                    if d < term.lower_a {
                        let u = term.lower_a - d;
                        e += 0.5 * weights.w_pre * u * u * term.inv_sigma2;
                        g = v_add(
                            g,
                            v_scale(dv, -sign_n * weights.w_pre * u * term.inv_sigma2 / d),
                        );
                    } else if d > term.upper_a {
                        let u = d - term.upper_a;
                        e += 0.5 * weights.w_pre * u * u * term.inv_sigma2;
                        g = v_add(
                            g,
                            v_scale(dv, sign_n * weights.w_pre * u * term.inv_sigma2 / d),
                        );
                    }
                } else {
                    let (g2_c, dg2_ds) = solomon_r6_rate_and_ds(r2_raw, term.k_pre_a6_s1);
                    let inv_s = term.inv_sigma2.sqrt();
                    let z = (g2_c - term.gamma2_obs_s1) * inv_s;
                    let kappa = 8.0_f64;
                    let (he, dh) = if z.abs() <= kappa {
                        (0.5 * z * z, z)
                    } else {
                        (kappa * (z.abs() - 0.5 * kappa), kappa * z.signum())
                    };
                    e += weights.w_pre * he;
                    let de = weights.w_pre * dh * inv_s * sign_n;
                    g = v_add(g, v_scale(dv, de * 2.0 * dg2_ds));
                    if d > term.upper_a {
                        let u = d - term.upper_a;
                        e += 0.5 * weights.w_pre * u * u;
                        g = v_add(g, v_scale(dv, sign_n * weights.w_pre * u / d));
                    } else if d < term.lower_a {
                        let u = term.lower_a - d;
                        e += 0.5 * weights.w_pre * u * u;
                        g = v_add(g, v_scale(dv, -sign_n * weights.w_pre * u / d));
                    }
                }
            }
        }

        if weights.w_noe > 0.0 {
            for term in &prep.noe_terms {
                let c_end = (term.cand_start + term.cand_len).min(prep.noe_candidates.len());
                if term.cand_start >= c_end {
                    continue;
                }
                let cands = &prep.noe_candidates[term.cand_start..c_end];
                let mut cand_evals: Vec<(Vec3, f64, f64)> = Vec::new();
                let mut s_min = f64::INFINITY;
                for cand in cands {
                    let ra = cand.atom_a / 7;
                    let rb = cand.atom_b / 7;
                    let ma = kin.residues[ra].master_block_id;
                    let mb = kin.residues[rb].master_block_id;
                    let a0_b1 = ma == Some(m0) && m1 != m0 && mb == Some(m1);
                    let a1_b0 = m1 != m0 && ma == Some(m1) && mb == Some(m0);
                    if !a0_b1 && !a1_b0 {
                        continue;
                    }
                    let (pa, pb, sign_n) = if a1_b0 {
                        let rel_a = mat3_mul_vec(q_b1, v_sub(init_slots[cand.atom_a].0, init_n_b1));
                        (v_add(n_pos, rel_a), init_slots[cand.atom_b].0, 1.0)
                    } else {
                        let rel_b = mat3_mul_vec(q_b1, v_sub(init_slots[cand.atom_b].0, init_n_b1));
                        (init_slots[cand.atom_a].0, v_add(n_pos, rel_b), -1.0)
                    };
                    let df = v_sub(pa, pb);
                    let sc = v_dot(df, df) + 1e-6;
                    if sc < s_min {
                        s_min = sc;
                    }
                    cand_evals.push((df, sc, sign_n));
                }
                if cand_evals.is_empty() || !s_min.is_finite() {
                    continue;
                }
                let mut tilde_s = 0.0_f64;
                for &(_, sc, _) in &cand_evals {
                    let q = s_min / sc;
                    tilde_s += q * q * q;
                }
                if tilde_s <= 0.0 {
                    continue;
                }
                let d_eff = s_min.sqrt() / det_cbrt(tilde_s).sqrt();
                let (v_e, dv_dd) = if d_eff > term.upper_a {
                    let u = d_eff - term.upper_a;
                    (weights.w_noe * u * u, 2.0 * weights.w_noe * u)
                } else if d_eff < term.lower_a {
                    let u = term.lower_a - d_eff;
                    (weights.w_noe * u * u, -2.0 * weights.w_noe * u)
                } else {
                    (0.0, 0.0)
                };
                e += v_e;
                if dv_dd != 0.0 {
                    let denom = s_min * tilde_s;
                    for &(df, sc, sign_n) in &cand_evals {
                        let q = s_min / sc;
                        let w_c = q * q * q;
                        g = v_add(g, v_scale(df, sign_n * dv_dd * (d_eff * w_c * q) / denom));
                    }
                }
            }
        }

        (e, g)
    };

    let mut grid_cands: Vec<(f64, Vec3)> = Vec::with_capacity(390);
    let (e_init, _) = eval_trans(init_n_b1);
    grid_cands.push((e_init, init_n_b1));

    let same_chain_search = kin
        .rigid_blocks
        .get(1)
        .is_none_or(|b1| b1.chain_idx == kin.rigid_blocks[0].chain_idx);
    let shell_origin = if same_chain_search {
        anchor_c_prev
    } else {
        kin.chain_ref_centroids
            .first()
            .copied()
            .unwrap_or(anchor_c_prev)
    };
    let r_cap = if same_chain_search { 35.0 } else { 60.0 };
    let r_shells = [
        (max_reach * 0.24).clamp(3.6, r_cap * 0.35),
        (max_reach * 0.38).clamp(5.0, r_cap * 0.48),
        (max_reach * 0.52).clamp(6.5, r_cap * 0.62),
        (max_reach * 0.66).clamp(8.0, r_cap * 0.76),
        (max_reach * 0.80).clamp(9.5, r_cap * 0.90),
        (max_reach * 0.91).clamp(11.0, r_cap),
    ];
    let golden = PI * (3.0 - 5.0_f64.sqrt());
    for &rad in &r_shells {
        for k in 0..64 {
            let y = 1.0 - 2.0 * ((k as f64) + 0.5) / 64.0;
            let r_xy = (1.0 - y * y).max(0.0).sqrt();
            let th = golden * (k as f64);
            let (st, ct) = det_sin_cos(th);
            let cand = [
                shell_origin[0] + rad * r_xy * ct,
                shell_origin[1] + rad * y,
                shell_origin[2] + rad * r_xy * st,
            ];
            let (ce, _) = eval_trans(cand);
            grid_cands.push((ce, cand));
        }
    }
    grid_cands.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut start_basins: Vec<Vec3> = Vec::with_capacity(4);
    for &(_, p) in &grid_cands {
        if start_basins.iter().all(|&prev| v_dist_sq(prev, p) > 20.25) {
            start_basins.push(p);
            if start_basins.len() >= 4 {
                break;
            }
        }
    }
    if start_basins.is_empty() {
        start_basins.push(init_n_b1);
    }

    let mut global_best_p = start_basins[0];
    let mut global_best_e = f64::INFINITY;
    for start_p in start_basins {
        let mut cur_p = start_p;
        let (mut cur_e, mut cur_g) = eval_trans(cur_p);
        let mut s_h: Vec<Vec3> = Vec::with_capacity(5);
        let mut y_h: Vec<Vec3> = Vec::with_capacity(5);
        let mut r_h: Vec<f64> = Vec::with_capacity(5);

        for _ in 0..60 {
            let g_norm = v_norm(cur_g);
            if g_norm < 1e-5 {
                break;
            }
            let mut q = cur_g;
            let kh = s_h.len();
            let mut al = [0.0_f64; 5];
            for idx in (0..kh).rev() {
                al[idx] = r_h[idx] * v_dot(s_h[idx], q);
                q = v_sub(q, v_scale(y_h[idx], al[idx]));
            }
            let gam = if kh > 0 {
                let sy = v_dot(s_h[kh - 1], y_h[kh - 1]);
                let yy = v_dot(y_h[kh - 1], y_h[kh - 1]);
                if yy > 1e-18 {
                    (sy / yy).clamp(1e-4, 10.0)
                } else {
                    0.2 / g_norm.max(1.0)
                }
            } else {
                (1.0 / g_norm.max(1.0)).clamp(1e-3, 1.0)
            };
            q = v_scale(q, gam);
            for idx in 0..kh {
                let bt = r_h[idx] * v_dot(y_h[idx], q);
                q = v_add(q, v_scale(s_h[idx], al[idx] - bt));
            }
            let mut dir = [-q[0], -q[1], -q[2]];
            let mut dg = v_dot(dir, cur_g);
            if dg >= 0.0 {
                s_h.clear();
                y_h.clear();
                r_h.clear();
                let sc = (1.0 / g_norm.max(1.0)).min(1.0);
                dir = v_scale(cur_g, -sc);
                dg = v_dot(dir, cur_g);
            }
            let d_len = v_norm(dir);
            if d_len > 3.0 {
                let sc = 3.0 / d_len;
                dir = v_scale(dir, sc);
                dg *= sc;
            }
            let mut alpha = 1.0_f64;
            let mut accepted = false;
            let mut next_p = cur_p;
            let mut next_e = cur_e;
            let mut next_g = cur_g;
            for _ in 0..10 {
                let cand = v_add(cur_p, v_scale(dir, alpha));
                let (ce, cg) = eval_trans(cand);
                if ce <= cur_e + 1e-4 * alpha * dg {
                    next_p = cand;
                    next_e = ce;
                    next_g = cg;
                    accepted = true;
                    break;
                }
                alpha *= 0.5;
            }
            if !accepted {
                break;
            }
            let sv = v_sub(next_p, cur_p);
            let yv = v_sub(next_g, cur_g);
            let sy = v_dot(sv, yv);
            if sy > 1e-14 {
                if s_h.len() == 5 {
                    s_h.remove(0);
                    y_h.remove(0);
                    r_h.remove(0);
                }
                s_h.push(sv);
                y_h.push(yv);
                r_h.push(1.0 / sy);
            }
            cur_p = next_p;
            cur_e = next_e;
            cur_g = next_g;
        }
        if cur_e < global_best_e {
            global_best_e = cur_e;
            global_best_p = cur_p;
        }
    }

    global_best_p
}

/// Close flexible linker `[lk_start .. lk_end)` onto target triad `(target_n, r_target)` of residue
/// `s = lk_end` strictly on the covalent `NeRF` manifold via fast two-loop `L-BFGS`.
#[allow(clippy::too_many_arguments)]
fn close_covalent_linker_segment(
    kin: &DomainLinkerKinematics,
    start_state: &DomainDockState,
    lk_start: usize,
    lk_end: usize,
    target_n: Vec3,
    r_target: &[[f64; 3]; 3],
    basin_variant: usize,
    rng: &mut DetRng,
) -> DomainDockState {
    let mut best_state = start_state.clone();
    if lk_start >= lk_end || lk_end >= kin.residues.len() || lk_start == 0 {
        return best_state;
    }

    match basin_variant {
        0 => {}
        1 => {
            for i in lk_start..lk_end {
                if kin.residues[i].phi_active {
                    best_state.phi_rad[i] =
                        wrap_angle_rad((-122.0_f64).to_radians() + rng.normal(0.0, 0.18));
                }
                if kin.residues[i].psi_active {
                    best_state.psi_rad[i] =
                        wrap_angle_rad((138.0_f64).to_radians() + rng.normal(0.0, 0.18));
                }
            }
        }
        2 => {
            for i in lk_start..lk_end {
                if kin.residues[i].phi_active {
                    best_state.phi_rad[i] =
                        wrap_angle_rad((-65.0_f64).to_radians() + rng.normal(0.0, 0.25));
                }
                if kin.residues[i].psi_active {
                    best_state.psi_rad[i] =
                        wrap_angle_rad((-40.0_f64).to_radians() + rng.normal(0.0, 0.25));
                }
            }
        }
        _ => {
            for i in lk_start..lk_end {
                if kin.residues[i].phi_active {
                    best_state.phi_rad[i] =
                        wrap_angle_rad(best_state.phi_rad[i] + rng.normal(0.0, 0.65));
                }
                if kin.residues[i].psi_active {
                    best_state.psi_rad[i] =
                        wrap_angle_rad(best_state.psi_rad[i] + rng.normal(0.0, 0.65));
                }
            }
        }
    }

    let s = lk_end;
    let rs_geom = &kin.residues[s];
    let target_ca_rel = mat3_mul_vec(r_target, rs_geom.local_slots[2].0);
    let target_c_rel = mat3_mul_vec(r_target, rs_geom.local_slots[4].0);

    let (base_slots, _) = kin.forward_slots_and_block_frames(&best_state);
    let anchor_n = base_slots[7 * (lk_start - 1)].0;
    let anchor_ca = base_slots[7 * (lk_start - 1) + 2].0;
    let anchor_c = base_slots[7 * (lk_start - 1) + 4].0;
    let anchor_psi = if kin.residues[lk_start - 1].psi_active {
        best_state.psi_rad[lk_start - 1]
    } else {
        kin.ref_psi_rad[lk_start - 1]
    };
    let anchor_omg = best_state.omega_rad[lk_start - 1];

    let m_len = lk_end - lk_start;
    let n_dim = 2 * m_len;
    let mut x = vec![0.0_f64; n_dim];
    for k in 0..m_len {
        let i = lk_start + k;
        x[2 * k] = best_state.phi_rad[i];
        x[2 * k + 1] = best_state.psi_rad[i];
    }

    let eval_closure = |xv: &[f64], g_out: &mut [f64]| -> f64 {
        g_out.fill(0.0);
        let mut n_vec = vec![[0.0_f64; 3]; m_len + 1];
        let mut ca_vec = vec![[0.0_f64; 3]; m_len + 1];
        let mut c_vec = vec![[0.0_f64; 3]; m_len + 1];

        let mut prev_n = anchor_n;
        let mut prev_ca = anchor_ca;
        let mut prev_c = anchor_c;
        let mut prev_psi = anchor_psi;
        let mut prev_omg = anchor_omg;

        for k in 0..=m_len {
            let i = lk_start + k;
            let rg = &kin.residues[i];
            let n_cur = nerf_place_atom(
                prev_n,
                prev_ca,
                prev_c,
                rg.r_c_prev_n,
                rg.ang_ca_prev_c_prev_n,
                prev_psi,
            );
            let ca_cur = nerf_place_atom(
                prev_ca,
                prev_c,
                n_cur,
                rg.r_n_ca,
                rg.ang_c_prev_n_ca,
                prev_omg,
            );
            let phi_cur = if k < m_len && rg.phi_active {
                xv[2 * k]
            } else {
                kin.ref_phi_rad[i]
            };
            let c_cur = nerf_place_atom(prev_c, n_cur, ca_cur, rg.r_ca_c, rg.ang_n_ca_c, phi_cur);
            n_vec[k] = n_cur;
            ca_vec[k] = ca_cur;
            c_vec[k] = c_cur;

            if k < m_len {
                prev_n = n_cur;
                prev_ca = ca_cur;
                prev_c = c_cur;
                prev_psi = if rg.psi_active {
                    xv[2 * k + 1]
                } else {
                    kin.ref_psi_rad[i]
                };
                prev_omg = best_state.omega_rad[i];
            }
        }

        let n_s = n_vec[m_len];
        let ca_s = ca_vec[m_len];
        let c_s = c_vec[m_len];
        let dn = v_sub(n_s, target_n);
        let d_ca_rel = v_sub(v_sub(ca_s, n_s), target_ca_rel);
        let d_c_rel = v_sub(v_sub(c_s, n_s), target_c_rel);

        let w_pos = 40.0_f64;
        let w_ori = 200.0_f64;
        let mut obj = 0.5 * w_pos * v_dot(dn, dn)
            + 0.5 * w_ori * (v_dot(d_ca_rel, d_ca_rel) + v_dot(d_c_rel, d_c_rel));

        let f_ca = v_scale(d_ca_rel, w_ori);
        let f_c = v_scale(d_c_rel, w_ori);
        let f_n = v_sub(v_scale(dn, w_pos), v_add(f_ca, f_c));

        let suf_f = v_add(f_n, v_add(f_ca, f_c));
        let mut suf_t = v_add(
            v_cross(v_sub(ca_s, n_s), f_ca),
            v_cross(v_sub(c_s, n_s), f_c),
        );
        let mut cur_q = n_s;

        for k in (0..m_len).rev() {
            let i = lk_start + k;
            let rg = &kin.residues[i];
            let ni = n_vec[k];
            let cai = ca_vec[k];
            let ci = c_vec[k];

            suf_t = v_add(suf_t, v_cross(v_sub(cur_q, ci), suf_f));
            cur_q = ci;
            if rg.psi_active {
                let axis_psi = v_unit(v_sub(ci, cai));
                g_out[2 * k + 1] = v_dot(axis_psi, suf_t);
            }

            suf_t = v_add(suf_t, v_cross(v_sub(cur_q, cai), suf_f));
            cur_q = cai;
            if rg.phi_active {
                let axis_phi = v_unit(v_sub(cai, ni));
                g_out[2 * k] = v_dot(axis_phi, suf_t);
            }

            if rg.rama_active {
                let ph = if rg.phi_active {
                    xv[2 * k]
                } else {
                    kin.ref_phi_rad[i]
                };
                let ps = if rg.psi_active {
                    xv[2 * k + 1]
                } else {
                    kin.ref_psi_rad[i]
                };
                let (vr, gp, gs) = dock_ramachandran_energy_and_grad(rg.res_class, ph, ps);
                obj += 0.02 * vr;
                if rg.phi_active {
                    g_out[2 * k] += 0.02 * gp;
                }
                if rg.psi_active {
                    g_out[2 * k + 1] += 0.02 * gs;
                }
            }
        }
        obj
    };

    let mut g = vec![0.0_f64; n_dim];
    let mut cur_obj = eval_closure(&x, &mut g);
    let m_hist = 6usize;
    let mut s_hist: Vec<Vec<f64>> = Vec::with_capacity(m_hist);
    let mut y_hist: Vec<Vec<f64>> = Vec::with_capacity(m_hist);
    let mut rho_hist: Vec<f64> = Vec::with_capacity(m_hist);

    for _ in 0..140 {
        let g_norm = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        if g_norm < 1e-4 {
            break;
        }
        let mut q = g.clone();
        let k_h = s_hist.len();
        let mut alpha = vec![0.0_f64; k_h];
        for idx in (0..k_h).rev() {
            let dot_sq: f64 = s_hist[idx].iter().zip(q.iter()).map(|(a, b)| a * b).sum();
            alpha[idx] = rho_hist[idx] * dot_sq;
            for j in 0..n_dim {
                q[j] -= alpha[idx] * y_hist[idx][j];
            }
        }
        let gamma = if k_h > 0 {
            let sy: f64 = s_hist[k_h - 1]
                .iter()
                .zip(y_hist[k_h - 1].iter())
                .map(|(a, b)| a * b)
                .sum();
            let yy: f64 = y_hist[k_h - 1].iter().map(|b| b * b).sum();
            if yy > 1e-18 {
                (sy / yy).clamp(1e-4, 5.0)
            } else {
                0.05
            }
        } else {
            (0.25 / g_norm.max(1.0)).clamp(1e-3, 0.25)
        };
        for v in &mut q {
            *v *= gamma;
        }
        for idx in 0..k_h {
            let dot_yq: f64 = y_hist[idx].iter().zip(q.iter()).map(|(a, b)| a * b).sum();
            let beta = rho_hist[idx] * dot_yq;
            for j in 0..n_dim {
                q[j] += s_hist[idx][j] * (alpha[idx] - beta);
            }
        }

        let mut dir: Vec<f64> = q.into_iter().map(|v| -v).collect();
        let mut dg: f64 = dir.iter().zip(g.iter()).map(|(d, gv)| d * gv).sum();
        if dg >= 0.0 {
            s_hist.clear();
            y_hist.clear();
            rho_hist.clear();
            let sc = (0.15 / g_norm.max(1.0)).min(0.15);
            dir = g.iter().map(|gv| -sc * gv).collect();
            dg = dir.iter().zip(g.iter()).map(|(d, gv)| d * gv).sum();
        }
        let max_d = dir.iter().fold(0.0_f64, |a, v| a.max(v.abs())).max(1e-12);
        if max_d > 0.35 {
            let sc = 0.35 / max_d;
            for d in &mut dir {
                *d *= sc;
            }
            dg *= sc;
        }

        let mut step = 1.0_f64;
        let mut accepted = false;
        let mut next_x = x.clone();
        let mut next_g = vec![0.0_f64; n_dim];
        let mut next_obj = cur_obj;
        for _ in 0..8 {
            for j in 0..n_dim {
                next_x[j] = wrap_angle_rad(x[j] + step * dir[j]);
            }
            let cand_obj = eval_closure(&next_x, &mut next_g);
            if cand_obj <= cur_obj + 1e-4 * step * dg {
                next_obj = cand_obj;
                accepted = true;
                break;
            }
            step *= 0.5;
        }
        if !accepted {
            break;
        }
        let s_vec: Vec<f64> = (0..n_dim)
            .map(|j| wrap_angle_rad(next_x[j] - x[j]))
            .collect();
        let y_vec: Vec<f64> = (0..n_dim).map(|j| next_g[j] - g[j]).collect();
        let sy: f64 = s_vec.iter().zip(y_vec.iter()).map(|(a, b)| a * b).sum();
        if sy > 1e-14 {
            if s_hist.len() == m_hist {
                s_hist.remove(0);
                y_hist.remove(0);
                rho_hist.remove(0);
            }
            s_hist.push(s_vec);
            y_hist.push(y_vec);
            rho_hist.push(1.0 / sy);
        }
        x = next_x;
        g = next_g;
        cur_obj = next_obj;
    }

    for k in 0..m_len {
        let i = lk_start + k;
        if kin.residues[i].phi_active {
            best_state.phi_rad[i] = x[2 * k];
        }
        if kin.residues[i].psi_active {
            best_state.psi_rad[i] = x[2 * k + 1];
        }
    }
    best_state
}

/// Polish a `DomainDockState` to tight gradient convergence via host `f64` two-loop `L-BFGS`
/// (`Nocedal 1980`) over active linker torsions and partner-chain $\mathfrak{se}(3)$ tangent coordinates.
pub fn polish_domain_dock_state_lbfgs(
    kin: &DomainLinkerKinematics,
    prep: &PreparedDomainDockRestraints,
    initial: &DomainDockState,
    weights: &DomainDockEnergyWeights,
    max_iters: usize,
) -> (DomainDockState, DomainDockEnergyBreakdown) {
    let base_k_loop = 500.0_f64.max(25.0 * weights.w_rama);
    let (mut state, mut eb) = polish_domain_dock_state_lbfgs_with_loop_k(
        kin,
        prep,
        initial,
        weights,
        max_iters,
        base_k_loop,
    );
    if kin.has_reentry_blocks() && kin.max_reentry_cn_err_angstrom(&state) > 0.05 {
        let (closed_state, _) = polish_domain_dock_state_lbfgs_with_loop_k(
            kin,
            prep,
            &state,
            weights,
            max_iters.max(80),
            4000.0_f64.max(base_k_loop),
        );
        if kin.max_reentry_cn_err_angstrom(&closed_state)
            < kin.max_reentry_cn_err_angstrom(&state)
        {
            state = closed_state;
            eb = evaluate_domain_dock_state_cpu(kin, prep, &state, weights).0;
        }
    }
    (state, eb)
}

fn polish_domain_dock_state_lbfgs_with_loop_k(
    kin: &DomainLinkerKinematics,
    prep: &PreparedDomainDockRestraints,
    initial: &DomainDockState,
    weights: &DomainDockEnergyWeights,
    max_iters: usize,
    k_loop: f64,
) -> (DomainDockState, DomainDockEnergyBreakdown) {
    let mut active_tors: Vec<(usize, bool)> = Vec::new();
    for &i in &kin.active_linker_residues {
        if kin.residues[i].phi_active {
            active_tors.push((i, false));
        }
        if kin.residues[i].psi_active {
            active_tors.push((i, true));
        }
    }
    let n_tors = active_tors.len();
    let n_partner_chains = kin.chain_ranges.len().saturating_sub(1);
    let trans_scale = 4.0_f64;
    let n_dim = n_tors + 6 * n_partner_chains;

    let mut cur_state = initial.clone();
    let (mut cur_eb, cur_grad) =
        evaluate_domain_dock_state_cpu_with_loop_k(kin, prep, &cur_state, weights, Some(k_loop));
    if n_dim == 0 || max_iters == 0 {
        return (cur_state, cur_eb);
    }

    let pack_grad = |gr: &DomainDockGradient| -> Vec<f64> {
        let mut gv = Vec::with_capacity(n_dim);
        for &(res_i, is_psi) in &active_tors {
            gv.push(if is_psi {
                gr.grad_psi[res_i]
            } else {
                gr.grad_phi[res_i]
            });
        }
        for c_idx in 1..kin.chain_ranges.len() {
            gv.extend_from_slice(&gr.grad_chain_rot[c_idx]);
            gv.push(gr.grad_chain_trans[c_idx][0] * trans_scale);
            gv.push(gr.grad_chain_trans[c_idx][1] * trans_scale);
            gv.push(gr.grad_chain_trans[c_idx][2] * trans_scale);
        }
        gv
    };

    let apply_step = |base: &DomainDockState, step: f64, dir: &[f64]| -> DomainDockState {
        let mut nxt = base.clone();
        for (k, &(res_i, is_psi)) in active_tors.iter().enumerate() {
            let delta = step * dir[k];
            if is_psi {
                nxt.psi_rad[res_i] = wrap_angle_rad(nxt.psi_rad[res_i] + delta);
            } else {
                nxt.phi_rad[res_i] = wrap_angle_rad(nxt.phi_rad[res_i] + delta);
            }
        }
        for p_idx in 0..n_partner_chains {
            let c_idx = p_idx + 1;
            let off = n_tors + 6 * p_idx;
            let d_omg = [step * dir[off], step * dir[off + 1], step * dir[off + 2]];
            nxt.chain_quats[c_idx] = quat_apply_world_rot_step(nxt.chain_quats[c_idx], d_omg);
            nxt.chain_trans[c_idx][0] += step * dir[off + 3] * trans_scale;
            nxt.chain_trans[c_idx][1] += step * dir[off + 4] * trans_scale;
            nxt.chain_trans[c_idx][2] += step * dir[off + 5] * trans_scale;
        }
        nxt
    };

    let mut g = pack_grad(&cur_grad);
    let m_hist = 8usize;
    let mut s_hist: Vec<Vec<f64>> = Vec::with_capacity(m_hist);
    let mut y_hist: Vec<Vec<f64>> = Vec::with_capacity(m_hist);
    let mut rho_hist: Vec<f64> = Vec::with_capacity(m_hist);

    for _ in 0..max_iters {
        let g_norm = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        if g_norm < 1e-4 {
            break;
        }
        let mut q = g.clone();
        let k_h = s_hist.len();
        let mut alpha = vec![0.0_f64; k_h];
        for idx in (0..k_h).rev() {
            let dot_sq: f64 = s_hist[idx].iter().zip(q.iter()).map(|(a, b)| a * b).sum();
            alpha[idx] = rho_hist[idx] * dot_sq;
            for j in 0..n_dim {
                q[j] -= alpha[idx] * y_hist[idx][j];
            }
        }
        let gamma = if k_h > 0 {
            let sy: f64 = s_hist[k_h - 1]
                .iter()
                .zip(y_hist[k_h - 1].iter())
                .map(|(a, b)| a * b)
                .sum();
            let yy: f64 = y_hist[k_h - 1].iter().map(|b| b * b).sum();
            if yy > 1e-18 {
                (sy / yy).clamp(1e-8, 10.0)
            } else {
                (0.05 / g_norm.max(1.0)).clamp(1e-8, 0.05)
            }
        } else {
            (0.08 / g_norm.max(1.0)).clamp(1e-8, 0.08)
        };
        for v in &mut q {
            *v *= gamma;
        }
        for idx in 0..k_h {
            let dot_yq: f64 = y_hist[idx].iter().zip(q.iter()).map(|(a, b)| a * b).sum();
            let beta = rho_hist[idx] * dot_yq;
            for j in 0..n_dim {
                q[j] += s_hist[idx][j] * (alpha[idx] - beta);
            }
        }

        let mut dir: Vec<f64> = q.into_iter().map(|v| -v).collect();
        let mut dg: f64 = dir.iter().zip(g.iter()).map(|(d, gv)| d * gv).sum();
        if dg >= 0.0 {
            s_hist.clear();
            y_hist.clear();
            rho_hist.clear();
            let sc = (0.06 / g_norm.max(1.0)).min(0.06);
            dir = g.iter().map(|gv| -sc * gv).collect();
            dg = dir.iter().zip(g.iter()).map(|(d, gv)| d * gv).sum();
        }

        let max_d = dir.iter().fold(0.0_f64, |a, v| a.max(v.abs())).max(1e-12);
        if max_d > 0.20 {
            let sc = 0.20 / max_d;
            for d in &mut dir {
                *d *= sc;
            }
            dg *= sc;
        }

        let mut step = 1.0_f64;
        let mut accepted = false;
        let mut next_state = cur_state.clone();
        let mut next_eb = cur_eb;
        let mut next_g = g.clone();
        for _ in 0..20 {
            let cand_st = apply_step(&cur_state, step, &dir);
            let (cand_eb, cand_gr) = evaluate_domain_dock_state_cpu_with_loop_k(
                kin,
                prep,
                &cand_st,
                weights,
                Some(k_loop),
            );
            if cand_eb.total <= cur_eb.total + 1e-4 * step * dg {
                next_state = cand_st;
                next_eb = cand_eb;
                next_g = pack_grad(&cand_gr);
                accepted = true;
                break;
            }
            step *= 0.5;
        }
        if !accepted && !s_hist.is_empty() {
            s_hist.clear();
            y_hist.clear();
            rho_hist.clear();
            continue;
        }
        if !accepted {
            break;
        }

        let s_vec: Vec<f64> = dir.iter().map(|&d| step * d).collect();
        let y_vec: Vec<f64> = next_g
            .iter()
            .zip(g.iter())
            .map(|(ng, og)| ng - og)
            .collect();
        let sy: f64 = s_vec.iter().zip(y_vec.iter()).map(|(a, b)| a * b).sum();
        if sy > 1e-14 {
            if s_hist.len() == m_hist {
                s_hist.remove(0);
                y_hist.remove(0);
                rho_hist.remove(0);
            }
            s_hist.push(s_vec);
            y_hist.push(y_vec);
            rho_hist.push(1.0 / sy);
        }

        cur_state = next_state;
        cur_eb = next_eb;
        g = next_g;
    }

    (cur_state, cur_eb)
}

/// Evaluate independent pre-dock intra-domain `RDC Q_work` & held-out `ByPeptidePlane Q_free`
/// across all confident rigid domains (`21.W2:P1-6`).
fn evaluate_pre_dock_intra_domain_rdcs(
    kin: &DomainLinkerKinematics,
    cseq: &ChainSeqIndex,
    partition: &PaeDomainPartition,
    filtered_media: &[DomainRdcMediumSet],
    max_intra_q_free: f64,
    require_min_8_work: bool,
) -> Result<Vec<PreDockDomainRdcCheck>, String> {
    let mut checks = Vec::new();
    let kin_plddts: Vec<Option<f64>> = kin.residues.iter().map(|r| r.plddt).collect();
    for dom in &partition.domains {
        let d_id = dom.domain_id;
        for med in filtered_media {
            let mut sub_restraints = Vec::new();
            let mut n_work = 0usize;
            let mut n_free = 0usize;
            for r in &med.restraints {
                let (Some(i1), Some(i2)) = (
                    cseq.resolve(r.chain_id_1.as_deref(), r.seq_id_1)?,
                    cseq.resolve(r.chain_id_2.as_deref(), r.seq_id_2)?,
                ) else {
                    continue;
                };
                if is_confident_rigid_residue(
                    i1,
                    partition,
                    Some(&kin_plddts),
                    DEFAULT_MIN_ORDERED_PLDDT,
                ) == Some(d_id)
                    && is_confident_rigid_residue(
                        i2,
                        partition,
                        Some(&kin_plddts),
                        DEFAULT_MIN_ORDERED_PLDDT,
                    ) == Some(d_id)
                {
                    if r.is_free {
                        n_free += 1;
                    } else {
                        n_work += 1;
                    }
                    sub_restraints.push(r.clone());
                }
            }
            if n_work == 0 && n_free == 0 {
                continue;
            }
            if n_work < MIN_TOLMAN_DOMAIN_WORK_RDCS {
                let passed = !require_min_8_work;
                checks.push(PreDockDomainRdcCheck {
                    domain_id: d_id,
                    medium_id: med.medium_id.clone(),
                    n_work,
                    n_free,
                    q_work: f64::NAN,
                    q_free: f64::NAN,
                    passed,
                });
                continue;
            }
            let sub_med = DomainRdcMediumSet {
                medium_id: med.medium_id.clone(),
                restraints: sub_restraints,
            };
            match fit_saupe_tensor_domain_rdc(&kin.ref_chain, cseq, &sub_med) {
                Ok((fit, _)) => {
                    let q_gate = if n_free > 0 && fit.q_free.is_finite() {
                        fit.q_free
                    } else {
                        fit.q_work
                    };
                    let passed = q_gate.is_finite()
                        && q_gate <= max_intra_q_free
                        && fit.q_work.is_finite()
                        && fit.q_work <= max_intra_q_free;
                    checks.push(PreDockDomainRdcCheck {
                        domain_id: d_id,
                        medium_id: med.medium_id.clone(),
                        n_work,
                        n_free,
                        q_work: fit.q_work,
                        q_free: fit.q_free,
                        passed,
                    });
                }
                Err(_) => {
                    checks.push(PreDockDomainRdcCheck {
                        domain_id: d_id,
                        medium_id: med.medium_id.clone(),
                        n_work,
                        n_free,
                        q_work: f64::NAN,
                        q_free: f64::NAN,
                        passed: false,
                    });
                }
            }
        }
    }
    Ok(checks)
}

/// Full scientific hygiene & input preparation pipeline shared identically by CPU (`nmr-core`)
/// and WebGPU (`nmr-gpu`):
/// 1. Enforces mandatory `tau_c_ns` + `proton_mhz` on Solomon `Gamma2` PRE restraints (`21.W3:P0-4`),
/// 2. Excludes flexible linker, tail, inter-domain, and `pLDDT < min_ordered_plddt` RDCs (`21.W2:P1-4`),
/// 3. Stratifies `ByPeptidePlane` `RdcFree` and `20%` `PCS` `is_free` holdouts strictly inside confident rigid domains (`21.W2:P1-4`, `P1-5`),
/// 4. Runs pre-dock intra-domain coordinate consistency check (`intra_domain_q_free <= 0.50`, `21.W2:P1-6`),
/// 5. Runs coordinate-noise-aware Tolman $\vartheta_d / \vartheta_1$ GDO analysis (`21.W2:A8`),
/// 6. Compiles `PreparedDomainDockRestraints` and enforces the single-medium / coaxial observability gate (`21.W2:P2-8`).
pub fn prepare_domain_dock_inputs_full(
    kin: &DomainLinkerKinematics,
    partition: &PaeDomainPartition,
    rdc_media: &mut [RdcMediumSet],
    pcs_tags: &[DomainPcsTagSet],
    pre_sites: &[DomainPreSiteSet],
    noe_adrs: &[AmbiguousDistanceRestraint],
    cfg: &DomainDockConfig,
) -> Result<PreparedDomainDockPipeline, String> {
    let cseq = ChainSeqIndex::from_kin(kin)?;
    let chain_ids: Vec<String> = kin.residues.iter().map(|r| r.chain_id.clone()).collect();
    let seq_ids: Vec<i32> = kin.residues.iter().map(|r| r.seq_id).collect();
    let kin_plddts: Vec<Option<f64>> = kin.residues.iter().map(|r| r.plddt).collect();
    let min_plddt = cfg.pae_cfg.min_ordered_plddt;

    // 1. Mandatory `tau_c_ns` + `proton_mhz` check on all Solomon `Gamma2Rate` PRE restraints (`21.W3:P0-4`)
    let mut normalized_pre_sites = pre_sites.to_vec();
    for ps in &mut normalized_pre_sites {
        let has_gamma2 = ps
            .restraints
            .iter()
            .any(|r| matches!(r.kind, PreRestraintKind::Gamma2Rate { .. }));
        if has_gamma2 {
            let tc = ps.tau_c_ns.or(cfg.pre_tau_c_ns).ok_or_else(|| {
                format!(
                    "PRE site '{}': Solomon Gamma2 restraints require explicit rotational correlation time tau_c_ns (`# tau_c_ns: <ns>` or `--pre-tau-c-ns <ns>`)",
                    ps.site_id
                )
            })?;
            if !tc.is_finite() || tc <= 0.0 {
                return Err(format!(
                    "PRE site '{}': tau_c_ns ({tc}) must be finite and > 0",
                    ps.site_id
                ));
            }
            let mhz = ps.proton_mhz.or(cfg.pre_proton_mhz).ok_or_else(|| {
                format!(
                    "PRE site '{}': Solomon–Bloembergen Gamma2 restraints require explicit 1H Larmor frequency proton_mhz (`# proton_mhz: <MHz>` or `--pre-proton-mhz <MHz>`)",
                    ps.site_id
                )
            })?;
            if !mhz.is_finite() || mhz <= 0.0 {
                return Err(format!(
                    "PRE site '{}': proton_mhz ({mhz}) must be finite and > 0",
                    ps.site_id
                ));
            }
            ps.tau_c_ns = Some(tc);
            ps.proton_mhz = Some(mhz);
            let default_exact_k =
                compute_solomon_k_pre_a6_s1(tc, mhz, 0.5, CODATA_ELECTRON_G_FACTOR)?;
            for r in &mut ps.restraints {
                if let PreRestraintKind::Gamma2Rate { k_pre_a6_s1, .. } = &mut r.kind {
                    // Only populate unset/invalid K (<= 0 or non-finite); preserve explicit per-site
                    // K (e.g. Gd3+ S=7/2 or Mn2+ S=5/2 parsed from `# spin_s` / `# g_factor`, `21F:B4`).
                    if !k_pre_a6_s1.is_finite() || *k_pre_a6_s1 <= 0.0 {
                        *k_pre_a6_s1 = default_exact_k;
                    }
                }
            }
        }
    }

    // 2. Build chain-explicit RDC media and filter out flexible linker / tail / low-pLDDT RDCs (`21.W2:P1-4`)
    let use_chain_cfg_rdcs = !cfg.chain_rdc_media.is_empty();
    let raw_dom_media = if use_chain_cfg_rdcs {
        cfg.chain_rdc_media.clone()
    } else {
        DomainRdcMediumSet::from_rdc_media(rdc_media)
    };
    let (mut dom_rdc_media, n_excluded_flexible_rdcs) = filter_confident_rigid_domain_rdcs(
        &raw_dom_media,
        &cseq,
        partition,
        Some(&kin_plddts),
        min_plddt,
    )?;

    if cfg.auto_partition_rdc_free
        && cfg.rdc_free_fraction > 0.0
        && !dom_rdc_media.is_empty()
        && !dom_rdc_media
            .iter()
            .any(|m| m.restraints.iter().any(|r| r.is_free))
    {
        partition_domain_stratified_domain_rdc_free_set(
            &mut dom_rdc_media,
            &cseq,
            &chain_ids,
            &seq_ids,
            partition,
            Some(&kin_plddts),
            min_plddt,
            cfg.rdc_free_fraction,
            cfg.seed,
        )?;
    }
    if !use_chain_cfg_rdcs {
        // Propagate filtered + stratified RDC sets back to caller slice when single-chain `rdc_media` was supplied
        for (dst_m, src_m) in rdc_media.iter_mut().zip(dom_rdc_media.iter()) {
            dst_m.restraints = src_m
                .restraints
                .iter()
                .map(DomainRdcRestraint::to_rdc_restraint)
                .collect();
        }
    }

    // 3. Domain-stratified `20%` `PCS` holdout (`21.W2:P1-5`)
    let mut pcs_tags_prepared = pcs_tags.to_vec();
    if cfg.auto_partition_pcs_free
        && cfg.pcs_free_fraction > 0.0
        && !pcs_tags_prepared.is_empty()
        && !pcs_tags_prepared
            .iter()
            .any(|t| t.restraints.iter().any(|r| r.is_free))
    {
        partition_domain_stratified_pcs_free_set(
            &mut pcs_tags_prepared,
            &cseq,
            partition,
            Some(&kin_plddts),
            min_plddt,
            cfg.pcs_free_fraction,
            cfg.seed,
        )?;
    }

    // 4. Tolman GDO inter-domain motion test including coordinate-noise floor (`21.W2:A8`)
    let tolman_gdo = evaluate_tolman_domain_gdo_multi_chain(
        &kin.ref_chain,
        &cseq,
        partition,
        Some(&kin_plddts),
        min_plddt,
        &dom_rdc_media,
        cfg.tolman_mc_trials,
        cfg.tolman_coord_noise_angstrom,
        cfg.seed,
    );

    let gdo_scale_map: Option<BTreeMap<(usize, usize), f64>> =
        if tolman_gdo.regime == DomainMotionRegime::DecoupledInterDomainMotion {
            let mut mp = BTreeMap::new();
            for (m_idx, med) in dom_rdc_media.iter().enumerate() {
                for fit in &tolman_gdo.domain_fits {
                    if fit.medium_id == med.medium_id {
                        mp.insert((fit.domain_id, m_idx), fit.rho_gdo.clamp(0.05, 1.0));
                    }
                }
            }
            Some(mp)
        } else {
            None
        };

    // 5. Compile working restraints
    let prep = PreparedDomainDockRestraints::compile_multi_chain(
        kin,
        partition,
        &dom_rdc_media,
        &pcs_tags_prepared,
        &normalized_pre_sites,
        noe_adrs,
        gdo_scale_map.as_ref(),
        min_plddt,
        cfg.refine_pcs_metal_pos,
        cfg.pcs_metal_search_radius_a,
        cfg.mtsl_tether_allowance_a,
    )?;

    let has_strong_dist = prep.pcs_terms.len() >= 5
        || prep.pre_terms.len() >= 3
        || prep.noe_terms.len() >= 3
        || (prep.pcs_terms.len() + prep.pre_terms.len() + prep.noe_terms.len() >= 4);

    // 6. Pre-dock intra-domain coordinate consistency gate (`21.W2:P1-6`)
    let mut extra_candor_notes = Vec::new();
    if n_excluded_flexible_rdcs > 0 {
        extra_candor_notes.push(format!(
            "Static-tensor hygiene [21.W2:P1-4]: excluded {n_excluded_flexible_rdcs} flexible-linker, terminal-tail, inter-domain, or low-pLDDT (< {min_plddt:.0}) RDC(s) from both static Saupe SVD and held-out RdcFree cross-validation."
        ));
    }
    let pre_dock_domain_checks = evaluate_pre_dock_intra_domain_rdcs(
        kin,
        &cseq,
        partition,
        &dom_rdc_media,
        cfg.max_intra_domain_rdc_q_free,
        !has_strong_dist && partition.domains.len() >= 2,
    )?;
    let mut poor_intra = false;
    for chk in &pre_dock_domain_checks {
        if !chk.passed {
            poor_intra = true;
            let reason = if chk.n_work < MIN_TOLMAN_DOMAIN_WORK_RDCS {
                format!(
                    "domain {} in medium '{}' has only {} working intra-domain RDCs (< {} required without cross-domain PCS/PRE/NOE restraints)",
                    chk.domain_id, chk.medium_id, chk.n_work, MIN_TOLMAN_DOMAIN_WORK_RDCS
                )
            } else {
                format!(
                    "domain {} in medium '{}' failed pre-dock intra-domain coordinate consistency (intra Q_work={:.4}, held-out Q_free={:.4} > {:.2})",
                    chk.domain_id,
                    chk.medium_id,
                    chk.q_work,
                    chk.q_free,
                    cfg.max_intra_domain_rdc_q_free
                )
            };
            if !cfg.allow_poor_intra_domain_fit {
                return Err(format!(
                    "Refusing rigid-body multi-domain docking [21.W2:P1-6]: {reason}. Fix intra-domain coordinates first or pass --allow-poor-intra-domain-fit (`allow_poor_intra_domain_fit = true`) to override with IntraDomainMismatchWarning."
                ));
            }
            extra_candor_notes.push(format!(
                "Candor warning [21.W2:P1-6 IntraDomainMismatchWarning]: {reason}; rigid-body inter-domain pose may absorb internal coordinate errors."
            ));
        }
    }

    // 7. Single-medium / coaxial RDC-only refusal gate (`21.W2:P2-8`)
    let mut single_med_warn = false;
    if partition.domains.len() >= 2 {
        let (obs_reg, _) = classify_dipolar_observability_multi(
            partition,
            &dom_rdc_media,
            prep.pcs_terms.len(),
            prep.pre_terms.len(),
            prep.noe_terms.len(),
            &kin.ref_chain,
            &cseq,
        );
        if obs_reg == DipolarObservabilityRegime::UnderdeterminedSingleMediumRdcOnly {
            if !cfg.allow_underdetermined_single_medium {
                return Err(format!(
                    "Refusing underdetermined multi-domain docking [F8 / 21.W2:P2-8]: fewer than 2 non-coaxial RDC media (|cos| < {MAX_COAXIAL_SAUPE_COS_SIM:.2}) and insufficient cross-domain PCS ({}) / PRE ({}) / NOE ({}) restraints leave a 4-fold PAS cone degeneracy and zero translational observability. Supply >= 2 non-coaxial media or PCS/PRE/NOE restraints, or pass --allow-underdetermined-single-medium (`allow_underdetermined_single_medium = true`) to explore degenerate solutions.",
                    prep.pcs_terms.len(),
                    prep.pre_terms.len(),
                    prep.noe_terms.len()
                ));
            }
            single_med_warn = true;
        }
    }

    let quality_verdict = match (poor_intra, single_med_warn) {
        (false, false) => DomainDockQualityVerdict::PassedAllChecks,
        (true, false) => DomainDockQualityVerdict::IntraDomainMismatchWarning,
        (false, true) => DomainDockQualityVerdict::UnderdeterminedSingleMediumWarning,
        (true, true) => DomainDockQualityVerdict::IntraDomainAndSingleMediumWarning,
    };

    Ok(PreparedDomainDockPipeline {
        domain_rdc_media: dom_rdc_media,
        pcs_tags: pcs_tags_prepared,
        tolman_gdo,
        prep,
        pre_dock_domain_checks,
        quality_verdict,
        extra_candor_notes,
        n_excluded_flexible_rdcs,
    })
}

/// Prepare `rdc_media` (`ByPeptidePlane` domain-stratified `RdcFree` holdout when needed), run
/// Tolman $\vartheta_d / \vartheta_1$ GDO analysis, and compile `PreparedDomainDockRestraints`.
pub fn prepare_domain_dock_inputs(
    kin: &DomainLinkerKinematics,
    partition: &PaeDomainPartition,
    rdc_media: &mut [RdcMediumSet],
    pcs_tags: &[DomainPcsTagSet],
    pre_sites: &[DomainPreSiteSet],
    noe_adrs: &[AmbiguousDistanceRestraint],
    cfg: &DomainDockConfig,
) -> Result<(TolmanGdoMotionReport, PreparedDomainDockRestraints), String> {
    let pipe = prepare_domain_dock_inputs_full(
        kin, partition, rdc_media, pcs_tags, pre_sites, noe_adrs, cfg,
    )?;
    Ok((pipe.tolman_gdo, pipe.prep))
}

/// Run the complete CPU (`f64`) multi-domain docking pipeline (`PAE` spectral clustering +
/// pre-dock intra-domain check + Tolman GDO motion test + multi-start `PAS`/`PCS`/`PRE` covalent
/// linker seeding + heavy-ball annealing + two-loop `L-BFGS` polish).
pub fn dock_domains_cpu(
    model: &AfStructureModel,
    pae: &PaeMatrix,
    rdc_media_in: &[RdcMediumSet],
    pcs_tags: &[DomainPcsTagSet],
    pre_sites: &[DomainPreSiteSet],
    noe_adrs: &[AmbiguousDistanceRestraint],
    cfg: &DomainDockConfig,
) -> Result<DomainDockResult, String> {
    let partition = cluster_pae_domains_for_model(model, pae, &cfg.pae_cfg)?;
    let kin = DomainLinkerKinematics::from_af_model(model, &partition)?;
    dock_domains_with_kinematics_cpu(
        kin,
        partition,
        rdc_media_in,
        pcs_tags,
        pre_sites,
        noe_adrs,
        cfg,
    )
}

/// Run multi-chain `(chain_id, seq_id)` CPU (`f64`) docking across multiple polymer chains (`21.W2:P1-8`).
pub fn dock_domains_multi_chain_cpu(
    models: &[AfStructureModel],
    pae: &PaeMatrix,
    domain_rdc_media: &[DomainRdcMediumSet],
    pcs_tags: &[DomainPcsTagSet],
    pre_sites: &[DomainPreSiteSet],
    noe_adrs: &[AmbiguousDistanceRestraint],
    cfg: &DomainDockConfig,
) -> Result<DomainDockResult, String> {
    if models.is_empty() {
        return Err("dock_domains_multi_chain_cpu requires >= 1 AfStructureModel".to_string());
    }
    let mut seq_ids = Vec::new();
    let mut chain_ids = Vec::new();
    let mut plddts = Vec::new();
    let mut any_plddt = false;
    for (m_idx, m) in models.iter().enumerate() {
        for r in &m.residues {
            seq_ids.push(r.seq_id);
            let cid = if r.chain_id.trim().is_empty() {
                format!("{}", (b'A' + (m_idx as u8).min(25)) as char)
            } else {
                r.chain_id.clone()
            };
            chain_ids.push(cid);
            if r.plddt.is_some() {
                any_plddt = true;
            }
            plddts.push(r.plddt.unwrap_or(100.0));
        }
    }
    let partition = cluster_pae_domains(
        pae,
        any_plddt.then_some(plddts.as_slice()),
        Some(&seq_ids),
        Some(&chain_ids),
        &cfg.pae_cfg,
    )?;
    let kin = DomainLinkerKinematics::from_multi_chain_models(models, &partition)?;
    let mut cfg_mc = cfg.clone();
    cfg_mc.chain_rdc_media = domain_rdc_media.to_vec();
    dock_domains_with_kinematics_cpu(kin, partition, &[], pcs_tags, pre_sites, noe_adrs, &cfg_mc)
}

pub fn dock_domains_with_kinematics_cpu(
    kin: DomainLinkerKinematics,
    partition: PaeDomainPartition,
    rdc_media_in: &[RdcMediumSet],
    pcs_tags: &[DomainPcsTagSet],
    pre_sites: &[DomainPreSiteSet],
    noe_adrs: &[AmbiguousDistanceRestraint],
    cfg: &DomainDockConfig,
) -> Result<DomainDockResult, String> {
    let mut rdc_media = rdc_media_in.to_vec();
    let pipe = prepare_domain_dock_inputs_full(
        &kin,
        &partition,
        &mut rdc_media,
        pcs_tags,
        pre_sites,
        noe_adrs,
        cfg,
    )?;
    let prep = &pipe.prep;

    let has_any_restraints = !prep.rdc_terms.is_empty()
        || !prep.pcs_terms.is_empty()
        || !prep.pre_terms.is_empty()
        || !prep.noe_terms.is_empty();

    let n_ens = cfg.ensemble_size.max(1);
    let polished = if !has_any_restraints || partition.domains.len() <= 1 {
        let st0 = kin.initial_state();
        let (eb0, _) = evaluate_domain_dock_state_cpu(&kin, prep, &st0, &cfg.weights);
        vec![(st0, eb0)]
    } else {
        let mut states = generate_domain_dock_seed_states(
            &kin,
            &partition,
            prep,
            &cfg.weights,
            cfg.n_replicas.max(n_ens),
            cfg.seed,
        );
        let l = kin.residues.len();
        let nc = kin.chain_ranges.len();
        let mut vel_phi = vec![vec![0.0_f64; l]; states.len()];
        let mut vel_psi = vec![vec![0.0_f64; l]; states.len()];
        let mut vel_rot = vec![vec![[0.0_f64; 3]; nc]; states.len()];
        let mut vel_trans = vec![vec![[0.0_f64; 3]; nc]; states.len()];
        let mut best_states = states.clone();
        let mut best_energies: Vec<f64> = states
            .iter()
            .map(|s| {
                evaluate_domain_dock_state_cpu(&kin, prep, s, &cfg.weights)
                    .0
                    .total
            })
            .collect();

        let total_steps = cfg.anneal_steps.max(1);
        for step in 0..total_steps {
            let prog = if total_steps > 1 {
                (step as f64) / ((total_steps - 1) as f64)
            } else {
                1.0
            };
            let ramp = (prog / 0.65).min(1.0);
            let mut cur_w = cfg.weights;
            cur_w.w_vdw = cfg.weights.w_vdw * (0.20 + 0.80 * ramp * ramp);
            cur_w.vdw_scale = cfg.weights.vdw_scale * (0.78 + 0.22 * ramp);

            for (ri, st) in states.iter_mut().enumerate() {
                let (eb, gr) = evaluate_domain_dock_state_cpu(&kin, prep, st, &cur_w);
                if ramp >= 1.0 && eb.total < best_energies[ri] {
                    best_energies[ri] = eb.total;
                    best_states[ri] = st.clone();
                }

                let mut g2 = 0.0_f64;
                for &i in &kin.active_linker_residues {
                    g2 += gr.grad_phi[i] * gr.grad_phi[i] + gr.grad_psi[i] * gr.grad_psi[i];
                }
                for c_idx in 1..nc {
                    g2 += v_dot(gr.grad_chain_rot[c_idx], gr.grad_chain_rot[c_idx])
                        + v_dot(gr.grad_chain_trans[c_idx], gr.grad_chain_trans[c_idx]);
                }
                let g_norm = g2.sqrt().max(1.0);
                let lr = (0.025 * (1.0 - 0.65 * prog)) / (1.0 + 0.25 * g_norm);
                let mom = 0.45 * (1.0 - 0.40 * prog);

                for &i in &kin.active_linker_residues {
                    if kin.residues[i].phi_active {
                        vel_phi[ri][i] = mom * vel_phi[ri][i] - lr * gr.grad_phi[i];
                        st.phi_rad[i] =
                            wrap_angle_rad(st.phi_rad[i] + vel_phi[ri][i].clamp(-0.04, 0.04));
                    }
                    if kin.residues[i].psi_active {
                        vel_psi[ri][i] = mom * vel_psi[ri][i] - lr * gr.grad_psi[i];
                        st.psi_rad[i] =
                            wrap_angle_rad(st.psi_rad[i] + vel_psi[ri][i].clamp(-0.04, 0.04));
                    }
                }
                for c_idx in 1..nc {
                    for ax in 0..3 {
                        vel_rot[ri][c_idx][ax] =
                            mom * vel_rot[ri][c_idx][ax] - lr * gr.grad_chain_rot[c_idx][ax];
                        vel_trans[ri][c_idx][ax] = mom * vel_trans[ri][c_idx][ax]
                            - (4.0 * lr) * gr.grad_chain_trans[c_idx][ax];
                        st.chain_trans[c_idx][ax] += vel_trans[ri][c_idx][ax].clamp(-0.6, 0.6);
                    }
                    let domg = [
                        vel_rot[ri][c_idx][0].clamp(-0.08, 0.08),
                        vel_rot[ri][c_idx][1].clamp(-0.08, 0.08),
                        vel_rot[ri][c_idx][2].clamp(-0.08, 0.08),
                    ];
                    st.chain_quats[c_idx] = quat_apply_world_rot_step(st.chain_quats[c_idx], domg);
                }
            }
        }

        select_and_polish_top_states(
            &kin,
            prep,
            best_states,
            &cfg.weights,
            n_ens,
            cfg.lbfgs_iters,
        )
    };

    finalize_domain_dock_pipeline(
        kin,
        partition,
        pipe,
        pre_sites,
        polished,
        "CPU (f64)".to_string(),
        0.0,
    )
}

/// Multi-chain-safe Kabsch backbone (`N`, `CA`, `C`) RMSD (Å) keyed strictly by `(chain_id, seq_id)`
/// so homodimers / multi-chain assemblies with overlapping residue numbering never collide (`21.W2:P1-8`).
pub fn multi_chain_backbone_rmsd(
    chain_a: &PdbChain,
    chain_b: &PdbChain,
    ordered_mask_a: Option<&[bool]>,
) -> f64 {
    let mut b_map: BTreeMap<(char, i32), usize> = BTreeMap::new();
    for (bi, rb) in chain_b.residues.iter().enumerate() {
        b_map.entry((rb.chain_id, rb.seq_id)).or_insert(bi);
    }
    let mut pts_a = Vec::with_capacity(chain_a.residues.len() * 3);
    let mut pts_b = Vec::with_capacity(chain_a.residues.len() * 3);
    for (ai, ra) in chain_a.residues.iter().enumerate() {
        if let Some(mask) = ordered_mask_a {
            if ai < mask.len() && !mask[ai] {
                continue;
            }
        }
        if let Some(&bi) = b_map.get(&(ra.chain_id, ra.seq_id)) {
            let rb = &chain_b.residues[bi];
            pts_a.extend([ra.n, ra.ca, ra.c]);
            pts_b.extend([rb.n, rb.ca, rb.c]);
        }
    }
    if pts_a.len() < 3 {
        return f64::NAN;
    }
    kabsch_rmsd(&pts_a, &pts_b)
}

/// Select diverse low-energy candidate states and run host `f64` two-loop `L-BFGS` polish.
pub fn select_and_polish_top_states(
    kin: &DomainLinkerKinematics,
    prep: &PreparedDomainDockRestraints,
    candidates: Vec<DomainDockState>,
    weights: &DomainDockEnergyWeights,
    ensemble_size: usize,
    lbfgs_iters: usize,
) -> Vec<(DomainDockState, DomainDockEnergyBreakdown)> {
    let mut scored: Vec<(f64, DomainDockState)> = candidates
        .into_iter()
        .map(|st| {
            let (eb, _) = evaluate_domain_dock_state_cpu(kin, prep, &st, weights);
            (eb.total, st)
        })
        .collect();
    scored.sort_by(|a, b| a.0.total_cmp(&b.0));

    let polish_budget = ensemble_size.max(6).min(scored.len().max(1)).min(16);
    let mut picked: Vec<DomainDockState> = Vec::with_capacity(polish_budget);
    let mut deferred: Vec<DomainDockState> = Vec::new();
    for (_, st) in scored {
        let ch = kin.build_pdb_chain(&st);
        let too_close = picked.iter().any(|prev| {
            let pch = kin.build_pdb_chain(prev);
            multi_chain_backbone_rmsd(&ch, &pch, None) < 0.25
        });
        if !too_close && picked.len() < polish_budget {
            picked.push(st);
        } else {
            deferred.push(st);
        }
    }
    for st in deferred {
        if picked.len() >= polish_budget {
            break;
        }
        picked.push(st);
    }

    let mut polished: Vec<(DomainDockState, DomainDockEnergyBreakdown)> = picked
        .into_iter()
        .map(|st| polish_domain_dock_state_lbfgs(kin, prep, &st, weights, lbfgs_iters))
        .collect();
    polished.sort_by(|a, b| a.1.total.total_cmp(&b.1.total));
    polished.truncate(ensemble_size.max(1));
    polished
}

/// Assemble the validated `DomainDockResult` from a `PreparedDomainDockPipeline` (preserving
/// `(chain_id, seq_id)` RDC/PCS validation, honest `pcs_q_free: Option<f64>`, `0.000 Å` intra-domain
/// RMSD check, and pre-dock / `F8` candor notes).
pub fn finalize_domain_dock_pipeline(
    kin: DomainLinkerKinematics,
    partition: PaeDomainPartition,
    pipe: PreparedDomainDockPipeline,
    _pre_sites: &[DomainPreSiteSet],
    polished: Vec<(DomainDockState, DomainDockEnergyBreakdown)>,
    engine_used: String,
    elapsed_ms: f64,
) -> Result<DomainDockResult, String> {
    if polished.is_empty() {
        return Err("finalize_domain_dock_pipeline received 0 conformers".to_string());
    }
    let cseq = ChainSeqIndex::from_kin(&kin)?;
    let prep = &pipe.prep;

    let mut conformers = Vec::with_capacity(polished.len());
    let mut chains = Vec::with_capacity(polished.len());
    let mut max_intra_all = 0.0_f64;

    for (idx, (state, energy)) in polished.into_iter().enumerate() {
        let (slots, _) = kin.forward_slots_and_block_frames(&state);
        let ch = kin.slots_to_pdb_chain(&slots);
        let intra_rmsd = kin.max_intra_domain_rmsd(&ch, &partition);
        let reentry_cn_err = kin.max_reentry_cn_err_from_slots(&slots);
        if intra_rmsd > max_intra_all {
            max_intra_all = intra_rmsd;
        }
        conformers.push(DockedConformer {
            model_index: idx + 1,
            state,
            energy,
            max_intra_domain_rmsd_angstrom: intra_rmsd,
            max_reentry_cn_err_a: reentry_cn_err,
        });
        chains.push(ch);
    }

    let best_chain = &chains[0];
    let (observability_regime, mut candor_notes) = classify_dipolar_observability_multi(
        &partition,
        &pipe.domain_rdc_media,
        prep.pcs_terms.len(),
        prep.pre_terms.len(),
        prep.noe_terms.len(),
        best_chain,
        &cseq,
    );
    candor_notes.extend(pipe.extra_candor_notes);
    if let Some(ref w_msg) = pipe.tolman_gdo.warning_message {
        candor_notes.push(w_msg.clone());
    }

    // Evaluate chain-explicit global and per-domain RDC Q_work & ByPeptidePlane Q_free on Model 1
    let mut media_fits = Vec::new();
    let mut all_points = Vec::new();
    let mut chained_points: Vec<(String, RdcBackCalcPoint)> = Vec::new();
    let mut work_pairs_all = Vec::new();
    let mut free_pairs_all = Vec::new();

    for m in &pipe.domain_rdc_media {
        if let Ok((fit, pts)) = fit_saupe_tensor_domain_rdc(best_chain, &cseq, m) {
            media_fits.push(fit);
            for (ch_id, pt) in pts {
                if pt.is_free {
                    free_pairs_all.push((pt.d_obs_norm_nh_hz, pt.d_calc_norm_nh_hz));
                } else {
                    work_pairs_all.push((pt.d_obs_norm_nh_hz, pt.d_calc_norm_nh_hz));
                }
                all_points.push(pt.clone());
                chained_points.push((ch_id, pt));
            }
        }
    }

    let rdc_validation = if !media_fits.is_empty() {
        let overall_q_work = cornilescu_q(&work_pairs_all);
        let overall_q_free = cornilescu_q(&free_pairs_all);
        Some(RdcValidationSummary {
            media: media_fits,
            points: all_points,
            overall_q_work,
            overall_q_free,
        })
    } else {
        None
    };
    let (global_rdc_q_work, global_rdc_q_free) = match &rdc_validation {
        Some(rv) => (rv.overall_q_work, rv.overall_q_free),
        None => (f64::NAN, f64::NAN),
    };

    // Per-domain validation summary keyed by `(chain_id, seq_id)` (`21.W2:P1-8`)
    let mut per_domain_validation = Vec::with_capacity(partition.domains.len());
    for dom in &partition.domains {
        let mut pts_cur = Vec::with_capacity(dom.residue_indices.len() * 3);
        let mut pts_ref = Vec::with_capacity(dom.residue_indices.len() * 3);
        let mut dom_cseq_set: BTreeSet<(String, i32)> = BTreeSet::new();
        for &idx in &dom.residue_indices {
            pts_cur.extend([
                best_chain.residues[idx].n,
                best_chain.residues[idx].ca,
                best_chain.residues[idx].c,
            ]);
            pts_ref.extend([
                kin.ref_chain.residues[idx].n,
                kin.ref_chain.residues[idx].ca,
                kin.ref_chain.residues[idx].c,
            ]);
            dom_cseq_set.insert((
                best_chain.residues[idx].chain_id.to_string(),
                best_chain.residues[idx].seq_id,
            ));
        }
        let mut d_rmsd = if pts_cur.len() >= 3 {
            kabsch_rmsd(&pts_cur, &pts_ref)
        } else {
            0.0
        };
        if d_rmsd < 1e-12 {
            d_rmsd = 0.0;
        }
        let (dq_work, dq_free) = if !chained_points.is_empty() {
            let mut wp = Vec::new();
            let mut fp = Vec::new();
            for (ch_id, pt) in &chained_points {
                if dom_cseq_set.contains(&(ch_id.clone(), pt.seq_id)) {
                    if pt.is_free {
                        fp.push((pt.d_obs_norm_nh_hz, pt.d_calc_norm_nh_hz));
                    } else {
                        wp.push((pt.d_obs_norm_nh_hz, pt.d_calc_norm_nh_hz));
                    }
                }
            }
            (cornilescu_q(&wp), cornilescu_q(&fp))
        } else {
            (f64::NAN, f64::NAN)
        };
        let pre_chk = pipe
            .pre_dock_domain_checks
            .iter()
            .filter(|c| c.domain_id == dom.domain_id)
            .fold((f64::NAN, f64::NAN), |(bw, bf), c| {
                let nw = if bw.is_nan() {
                    c.q_work
                } else if c.q_work.is_finite() {
                    bw.max(c.q_work)
                } else {
                    bw
                };
                let nf = if bf.is_nan() {
                    c.q_free
                } else if c.q_free.is_finite() {
                    bf.max(c.q_free)
                } else {
                    bf
                };
                (nw, nf)
            });
        per_domain_validation.push(PerDomainValidationSummary {
            domain_id: dom.domain_id,
            n_residues: dom.residue_indices.len(),
            intra_domain_distortion_rmsd_angstrom: d_rmsd,
            pre_dock_rdc_q_work: pre_chk.0,
            pre_dock_rdc_q_free: pre_chk.1,
            rdc_q_work: dq_work,
            rdc_q_free: dq_free,
        });
    }

    // Evaluate honest PCS quality factors (`pcs_q_work` vs strictly held-out `pcs_q_free`, `21.W2:P1-5`)
    let (best_slots, best_blk_frames) = kin.forward_slots_and_block_frames(&conformers[0].state);
    let mut pcs_intra_pairs = Vec::new();
    let mut pcs_inter_work_pairs = Vec::new();
    let mut pcs_free_pairs = Vec::new();

    for (t_idx, tag_in) in pipe.pcs_tags.iter().enumerate() {
        let Some(prep_tag) = prep.pcs_tags.get(t_idx) else {
            continue;
        };
        let (orig_n, ref r_b) = best_blk_frames[prep_tag.host_block_id];
        let m_pos = v_add(orig_n, mat3_mul_vec(r_b, prep_tag.metal_local_pos));
        let dchi_w = mat3_conjugate_sym(r_b, &prep_tag.dchi_local);
        for pr in &tag_in.restraints {
            let Some((slot, _)) = resolve_kin_atom_slot_and_seg_multi(
                &kin,
                &cseq,
                pr.chain_id.as_deref(),
                pr.seq_id,
                &pr.atom_name,
            )?
            else {
                continue;
            };
            if !best_slots[slot].1 {
                continue;
            }
            let v = v_sub(best_slots[slot].0, m_pos);
            let r2 = v_dot(v, v);
            if r2 < 1e-12 {
                continue;
            }
            let r5 = r2 * r2 * r2.sqrt();
            let mv = mat3_mul_vec(&dchi_w, v);
            let calc = (PCS_CARTESIAN_PREFACTOR / r5) * v_dot(v, mv) + prep_tag.pcs_offset_ppm;
            let res_i = slot / 7;
            let is_host = kin.residues[res_i].master_block_id == Some(prep_tag.host_block_id);
            if pr.is_free && !is_host {
                pcs_free_pairs.push((pr.pcs_obs_ppm, calc));
            } else if is_host {
                if pr.is_free {
                    pcs_free_pairs.push((pr.pcs_obs_ppm, calc));
                } else {
                    pcs_intra_pairs.push((pr.pcs_obs_ppm, calc));
                }
            } else {
                pcs_inter_work_pairs.push((pr.pcs_obs_ppm, calc));
            }
        }
    }

    let pcs_q_intra = cornilescu_q(&pcs_intra_pairs);
    let n_pcs_work = pcs_inter_work_pairs.len();
    let n_pcs_free = pcs_free_pairs.len();
    let pcs_q_work = (n_pcs_work > 0).then(|| cornilescu_q(&pcs_inter_work_pairs));
    // NEVER fall back from `pcs_q_free` to `pcs_q_work`: `None` when `n_pcs_free == 0`!
    let pcs_q_free = (n_pcs_free > 0).then(|| cornilescu_q(&pcs_free_pairs));

    // Pairwise ensemble backbone RMSD over rigid domains
    let mut dom_mask = vec![false; kin.residues.len()];
    for dom in &partition.domains {
        for &idx in &dom.residue_indices {
            if idx < dom_mask.len() {
                dom_mask[idx] = true;
            }
        }
    }
    let use_dom_mask = dom_mask.iter().any(|&b| b);
    let ensemble_pairwise_bb_rmsd_angstrom = if chains.len() >= 2 {
        let mut sum = 0.0_f64;
        let mut cnt = 0usize;
        for i in 0..chains.len() {
            for j in (i + 1)..chains.len() {
                sum += multi_chain_backbone_rmsd(
                    &chains[i],
                    &chains[j],
                    use_dom_mask.then_some(dom_mask.as_slice()),
                );
                cnt += 1;
            }
        }
        sum / (cnt.max(1) as f64)
    } else {
        0.0
    };

    let best_energy = conformers[0].energy;
    let report = DomainDockReport {
        engine_used,
        quality_verdict: pipe.quality_verdict,
        partition,
        tolman_gdo: pipe.tolman_gdo,
        observability_regime,
        candor_notes,
        pre_dock_domain_checks: pipe.pre_dock_domain_checks,
        n_excluded_flexible_rdcs: pipe.n_excluded_flexible_rdcs,
        rdc_stratification: RdcFreeStratification::ByPeptidePlane,
        global_rdc_q_work,
        global_rdc_q_free,
        pcs_q_intra,
        pcs_q_work,
        pcs_q_free,
        n_pcs_work,
        n_pcs_free,
        max_intra_domain_rmsd_angstrom: max_intra_all,
        max_reentry_cn_err_a: conformers[0].max_reentry_cn_err_a,
        ensemble_pairwise_bb_rmsd_angstrom,
        per_domain_validation,
        rdc_validation,
        best_energy,
        elapsed_ms,
    };

    Ok(DomainDockResult {
        kinematics: kin,
        conformers,
        chains,
        report,
    })
}

/// Compatibility wrapper around [`finalize_domain_dock_pipeline`].
#[allow(clippy::too_many_arguments)]
pub fn finalize_domain_dock_result(
    kin: DomainLinkerKinematics,
    partition: PaeDomainPartition,
    tolman_gdo: TolmanGdoMotionReport,
    prep: &PreparedDomainDockRestraints,
    rdc_media: &[RdcMediumSet],
    pcs_tags: &[DomainPcsTagSet],
    pre_sites: &[DomainPreSiteSet],
    polished: Vec<(DomainDockState, DomainDockEnergyBreakdown)>,
    engine_used: String,
    elapsed_ms: f64,
) -> Result<DomainDockResult, String> {
    let pipe = PreparedDomainDockPipeline {
        domain_rdc_media: DomainRdcMediumSet::from_rdc_media(rdc_media),
        pcs_tags: pcs_tags.to_vec(),
        tolman_gdo,
        prep: prep.clone(),
        pre_dock_domain_checks: Vec::new(),
        quality_verdict: DomainDockQualityVerdict::PassedAllChecks,
        extra_candor_notes: Vec::new(),
        n_excluded_flexible_rdcs: 0,
    };
    finalize_domain_dock_pipeline(
        kin,
        partition,
        pipe,
        pre_sites,
        polished,
        engine_used,
        elapsed_ms,
    )
}

// ---- PDB, mmCIF, JSON, and TSV Export Helpers ----

fn fmt_opt_q(val: Option<f64>) -> String {
    match val {
        Some(v) if v.is_finite() => format!("{v:.4}"),
        _ => "NA".to_string(),
    }
}

/// Format a docked multi-domain ensemble in standard wwPDB `MODEL ... ENDMDL` format,
/// preserving all rigid-domain sidechain heavy atoms, per-residue `pLDDT` `tempFactor`s,
/// and structured `REMARK` provenance/candor headers.
pub fn format_docked_pdb_ensemble(result: &DomainDockResult, title: &str) -> String {
    if result.conformers.is_empty() {
        return "END\n".to_string();
    }
    let rep = &result.report;
    let l = result.kinematics.residues.len();
    let mut out = String::with_capacity(result.conformers.len() * l * 560);

    let clean_title: String = title
        .chars()
        .filter(|c| c.is_ascii_graphic() || *c == ' ')
        .take(66)
        .collect();
    out.push_str(&format!("TITLE     {clean_title}\n"));
    out.push_str(&format!(
        "REMARK   1 FERRIC T21.5 PAE MULTI-DOMAIN DOCKING ENSEMBLE ({} MODELS, ENGINE={})\n",
        result.conformers.len(),
        rep.engine_used
    ));
    out.push_str(&format!(
        "REMARK   2 DOMAINS={} LINKERS={} REGIME={:?} TOLMAN={:?} VERDICT={:?}\n",
        rep.partition.domains.len(),
        rep.partition.linkers.len(),
        rep.observability_regime,
        rep.tolman_gdo.regime,
        rep.quality_verdict
    ));
    out.push_str(&format!(
        "REMARK   2 RDC_Q_WORK={:.4} BY_PEPTIDE_PLANE_Q_FREE={:.4} PCS_Q_WORK={} PCS_Q_FREE={} INTRA_RMSD={:.3}\n",
        rep.global_rdc_q_work,
        rep.global_rdc_q_free,
        fmt_opt_q(rep.pcs_q_work),
        fmt_opt_q(rep.pcs_q_free),
        rep.max_intra_domain_rmsd_angstrom
    ));
    for note in &rep.candor_notes {
        let ascii_note: String = note
            .chars()
            .filter(|c| c.is_ascii_graphic() || *c == ' ')
            .take(68)
            .collect();
        out.push_str(&format!("REMARK   3 {ascii_note}\n"));
    }

    for conf in &result.conformers {
        let (slots, blk_frames) = result
            .kinematics
            .forward_slots_and_block_frames(&conf.state);
        out.push_str(&format!("MODEL     {:>4}\n", conf.model_index));
        out.push_str(&format!(
            "REMARK   4 ENERGY TOTAL={:.3} RDC={:.3} PCS={:.3} PRE={:.3} NOE={:.3} VDW={:.3} RAMA={:.3}\n",
            conf.energy.total,
            conf.energy.rdc,
            conf.energy.pcs,
            conf.energy.pre,
            conf.energy.noe,
            conf.energy.vdw,
            conf.energy.rama
        ));

        let mut serial = 1usize;
        let emit_atom = |buf: &mut String,
                         ser: &mut usize,
                         at_name: &str,
                         res_name: &str,
                         ch_char: char,
                         seq_id: i32,
                         xyz: Vec3,
                         bfac: f64,
                         elem: &str| {
            let name_col = if at_name.len() < 4 {
                format!(" {at_name:<3}")
            } else {
                format!("{at_name:<4}")
            };
            buf.push_str(&format!(
                "ATOM  {:>5} {} {:>3} {}{:>4}    {:>8.3}{:>8.3}{:>8.3}  1.00{:>6.2}          {:>2}\n",
                (*ser).min(99_999),
                name_col,
                res_name,
                ch_char,
                seq_id,
                xyz[0],
                xyz[1],
                xyz[2],
                bfac.clamp(0.0, 100.0),
                elem
            ));
            *ser += 1;
        };

        for (i, rg) in result.kinematics.residues.iter().enumerate() {
            let base = 7 * i;
            let ch_char = rg.chain_id.chars().next().unwrap_or('A');
            let bfac = rg.plddt.unwrap_or(0.0);
            let is_gly = rg.res_class == DockResClass::Gly;

            emit_atom(
                &mut out,
                &mut serial,
                "N",
                &rg.res_name,
                ch_char,
                rg.seq_id,
                slots[base].0,
                bfac,
                "N",
            );
            if slots[base + 1].1 {
                emit_atom(
                    &mut out,
                    &mut serial,
                    "H",
                    &rg.res_name,
                    ch_char,
                    rg.seq_id,
                    slots[base + 1].0,
                    bfac,
                    "H",
                );
            }
            emit_atom(
                &mut out,
                &mut serial,
                "CA",
                &rg.res_name,
                ch_char,
                rg.seq_id,
                slots[base + 2].0,
                bfac,
                "C",
            );
            if is_gly {
                if slots[base + 3].1 {
                    emit_atom(
                        &mut out,
                        &mut serial,
                        "HA2",
                        &rg.res_name,
                        ch_char,
                        rg.seq_id,
                        slots[base + 3].0,
                        bfac,
                        "H",
                    );
                }
                if slots[base + 6].1 {
                    emit_atom(
                        &mut out,
                        &mut serial,
                        "HA3",
                        &rg.res_name,
                        ch_char,
                        rg.seq_id,
                        slots[base + 6].0,
                        bfac,
                        "H",
                    );
                }
            } else {
                if slots[base + 3].1 {
                    emit_atom(
                        &mut out,
                        &mut serial,
                        "HA",
                        &rg.res_name,
                        ch_char,
                        rg.seq_id,
                        slots[base + 3].0,
                        bfac,
                        "H",
                    );
                }
                if slots[base + 6].1 {
                    emit_atom(
                        &mut out,
                        &mut serial,
                        "CB",
                        &rg.res_name,
                        ch_char,
                        rg.seq_id,
                        slots[base + 6].0,
                        bfac,
                        "C",
                    );
                }
            }
            emit_atom(
                &mut out,
                &mut serial,
                "C",
                &rg.res_name,
                ch_char,
                rg.seq_id,
                slots[base + 4].0,
                bfac,
                "C",
            );
            emit_atom(
                &mut out,
                &mut serial,
                "O",
                &rg.res_name,
                ch_char,
                rg.seq_id,
                slots[base + 5].0,
                bfac,
                "O",
            );

            if let Some(bid) = rg.rigid_block_id {
                let (orig_n, ref r_b) = blk_frames[bid];
                for (aname, loc_xyz) in &rg.extra_heavy_local {
                    let w_xyz = v_add(orig_n, mat3_mul_vec(r_b, *loc_xyz));
                    let elem = aname
                        .chars()
                        .find(|c| c.is_ascii_alphabetic())
                        .map(|c| c.to_ascii_uppercase().to_string())
                        .unwrap_or_else(|| "C".to_string());
                    emit_atom(
                        &mut out,
                        &mut serial,
                        aname,
                        &rg.res_name,
                        ch_char,
                        rg.seq_id,
                        w_xyz,
                        bfac,
                        &elem,
                    );
                }
            }

            let is_chain_end =
                i + 1 == l || result.kinematics.residues[i + 1].chain_idx != rg.chain_idx;
            if is_chain_end {
                out.push_str(&format!(
                    "TER   {:>5}      {:>3} {}{:>4}\n",
                    serial.min(99_999),
                    rg.res_name,
                    ch_char,
                    rg.seq_id
                ));
                serial += 1;
            }
        }
        out.push_str("ENDMDL\n");
    }

    out.push_str("END\n");
    out
}

/// Write a docked multi-domain PDB ensemble to disk.
pub fn write_docked_pdb_ensemble(
    path: impl AsRef<Path>,
    result: &DomainDockResult,
    title: &str,
) -> Result<(), String> {
    let text = format_docked_pdb_ensemble(result, title);
    fs::write(path.as_ref(), text).map_err(|e| {
        format!(
            "failed to write docked PDB ensemble '{}': {e}",
            path.as_ref().display()
        )
    })
}

/// Format a docked multi-domain ensemble in ModelArchive-compatible mmCIF format (`_atom_site` +
/// `_ma_qa_metric_local` preserving per-residue `pLDDT` and structured candor comments).
pub fn format_docked_mmcif_ensemble(result: &DomainDockResult, entry_id: &str) -> String {
    let clean_id: String = entry_id
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-')
        .collect();
    let id_tok = if clean_id.is_empty() {
        "ferric_docked"
    } else {
        clean_id.as_str()
    };

    let rep = &result.report;
    let l = result.kinematics.residues.len();
    let mut cif = String::with_capacity(result.conformers.len() * l * 420);
    cif.push_str(&format!("data_{id_tok}\n"));
    cif.push_str(&format!("_entry.id {id_tok}\n"));
    cif.push_str("_ma_protocol_step.step_id 1\n");
    cif.push_str(&format!(
        "# FERRIC T21.5 domain_dock: domains={} linkers={} regime={:?} tolman={:?} verdict={:?} q_work={:.4} q_free={:.4} pcs_q_work={} pcs_q_free={} intra_rmsd={:.3}\n",
        rep.partition.domains.len(),
        rep.partition.linkers.len(),
        rep.observability_regime,
        rep.tolman_gdo.regime,
        rep.quality_verdict,
        rep.global_rdc_q_work,
        rep.global_rdc_q_free,
        fmt_opt_q(rep.pcs_q_work),
        fmt_opt_q(rep.pcs_q_free),
        rep.max_intra_domain_rmsd_angstrom
    ));
    for note in &rep.candor_notes {
        cif.push_str(&format!("# {note}\n"));
    }

    let has_plddt = result.kinematics.residues.iter().any(|r| r.plddt.is_some());
    if has_plddt {
        cif.push_str(
            "loop_\n_ma_qa_metric.id\n_ma_qa_metric.name\n_ma_qa_metric.mode\n1 pLDDT local\n#\n",
        );
    }

    cif.push_str("loop_\n_atom_site.group_PDB\n_atom_site.id\n_atom_site.type_symbol\n_atom_site.label_atom_id\n_atom_site.label_alt_id\n_atom_site.label_comp_id\n_atom_site.label_asym_id\n_atom_site.auth_asym_id\n_atom_site.label_seq_id\n_atom_site.auth_seq_id\n_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n_atom_site.occupancy\n_atom_site.B_iso_or_equiv\n_atom_site.pdbx_pdb_model_num\n");

    let mut atom_id = 1usize;
    for conf in &result.conformers {
        let (slots, blk_frames) = result
            .kinematics
            .forward_slots_and_block_frames(&conf.state);
        let mnum = conf.model_index;
        for (i, rg) in result.kinematics.residues.iter().enumerate() {
            let base = 7 * i;
            let lseq = i + 1;
            let aseq = rg.seq_id;
            let asym = if rg.chain_id.is_empty() {
                "A"
            } else {
                rg.chain_id.as_str()
            };
            let bfac = rg.plddt.unwrap_or(0.0);
            let is_gly = rg.res_class == DockResClass::Gly;

            let mut push_row = |sym: &str, aname: &str, xyz: Vec3| {
                cif.push_str(&format!(
                    "ATOM {atom_id} {sym} {aname} . {} {asym} {asym} {lseq} {aseq} {:.3} {:.3} {:.3} 1.00 {:.2} {mnum}\n",
                    rg.res_name, xyz[0], xyz[1], xyz[2], bfac
                ));
                atom_id += 1;
            };

            push_row("N", "N", slots[base].0);
            if slots[base + 1].1 {
                push_row("H", "H", slots[base + 1].0);
            }
            push_row("C", "CA", slots[base + 2].0);
            if is_gly {
                if slots[base + 3].1 {
                    push_row("H", "HA2", slots[base + 3].0);
                }
                if slots[base + 6].1 {
                    push_row("H", "HA3", slots[base + 6].0);
                }
            } else {
                if slots[base + 3].1 {
                    push_row("H", "HA", slots[base + 3].0);
                }
                if slots[base + 6].1 {
                    push_row("C", "CB", slots[base + 6].0);
                }
            }
            push_row("C", "C", slots[base + 4].0);
            push_row("O", "O", slots[base + 5].0);

            if let Some(bid) = rg.rigid_block_id {
                let (orig_n, ref r_b) = blk_frames[bid];
                for (aname, loc_xyz) in &rg.extra_heavy_local {
                    let w_xyz = v_add(orig_n, mat3_mul_vec(r_b, *loc_xyz));
                    let sym = aname
                        .chars()
                        .find(|c| c.is_ascii_alphabetic())
                        .map(|c| c.to_ascii_uppercase().to_string())
                        .unwrap_or_else(|| "C".to_string());
                    push_row(&sym, aname, w_xyz);
                }
            }
        }
    }
    cif.push_str("#\n");

    if has_plddt {
        cif.push_str("loop_\n_ma_qa_metric_local.label_asym_id\n_ma_qa_metric_local.label_seq_id\n_ma_qa_metric_local.label_comp_id\n_ma_qa_metric_local.metric_id\n_ma_qa_metric_local.metric_value\n");
        for (i, rg) in result.kinematics.residues.iter().enumerate() {
            if let Some(pl) = rg.plddt {
                let asym = if rg.chain_id.is_empty() {
                    "A"
                } else {
                    rg.chain_id.as_str()
                };
                cif.push_str(&format!("{asym} {} {} 1 {:.2}\n", i + 1, rg.res_name, pl));
            }
        }
        cif.push_str("#\n");
    }

    cif
}

/// Write a docked multi-domain mmCIF ensemble to disk.
pub fn write_docked_mmcif_ensemble(
    path: impl AsRef<Path>,
    result: &DomainDockResult,
    entry_id: &str,
) -> Result<(), String> {
    let text = format_docked_mmcif_ensemble(result, entry_id);
    fs::write(path.as_ref(), text).map_err(|e| {
        format!(
            "failed to write docked mmCIF ensemble '{}': {e}",
            path.as_ref().display()
        )
    })
}

/// Format `DomainDockReport` as pretty-printed JSON.
pub fn format_domain_dock_json(report: &DomainDockReport) -> Result<String, String> {
    serde_json::to_string_pretty(report)
        .map_err(|e| format!("failed to serialize DomainDockReport to JSON: {e}"))
}

/// Format `DomainDockReport` as deterministic ASCII TSV.
pub fn format_domain_dock_tsv(report: &DomainDockReport) -> String {
    let mut out = String::new();
    out.push_str(&format!("# engine_used\t{}\n", report.engine_used));
    out.push_str(&format!(
        "# quality_verdict\t{:?}\n",
        report.quality_verdict
    ));
    out.push_str(&format!(
        "# observability_regime\t{:?}\n",
        report.observability_regime
    ));
    out.push_str(&format!(
        "# tolman_gdo_regime\t{:?}\n",
        report.tolman_gdo.regime
    ));
    out.push_str(&format!(
        "# n_excluded_flexible_rdcs\t{}\n",
        report.n_excluded_flexible_rdcs
    ));
    out.push_str(&format!(
        "# global_rdc_q_work\t{:.4}\n",
        report.global_rdc_q_work
    ));
    out.push_str(&format!(
        "# by_peptide_plane_q_free\t{:.4}\n",
        report.global_rdc_q_free
    ));
    out.push_str(&format!("# pcs_q_work\t{}\n", fmt_opt_q(report.pcs_q_work)));
    out.push_str(&format!("# pcs_q_free\t{}\n", fmt_opt_q(report.pcs_q_free)));
    out.push_str(&format!(
        "# max_intra_domain_rmsd_angstrom\t{:.4}\n",
        report.max_intra_domain_rmsd_angstrom
    ));
    out.push_str(&format!(
        "# max_reentry_cn_err_a\t{:.4}\n",
        report.max_reentry_cn_err_a
    ));
    for note in &report.candor_notes {
        out.push_str(&format!("# candor_note\t{note}\n"));
    }
    out.push_str(
        "domain_id\tchain_id\tn_residues\tseq_spans\tmedian_intra_pae_a\tmean_plddt\tintra_rmsd_a\tpre_dock_rdc_q_work\tpre_dock_rdc_q_free\trdc_q_work\trdc_q_free\n",
    );
    for (dom, val) in report
        .partition
        .domains
        .iter()
        .zip(report.per_domain_validation.iter())
    {
        let spans_str: Vec<String> = dom
            .seq_id_spans
            .iter()
            .map(|(a, b)| format!("{a}-{b}"))
            .collect();
        out.push_str(&format!(
            "{}\t{}\t{}\t{}\t{:.3}\t{:.2}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\n",
            dom.domain_id,
            dom.chain_id,
            dom.residue_indices.len(),
            spans_str.join(","),
            dom.median_intra_pae_angstrom,
            dom.mean_plddt,
            val.intra_domain_distortion_rmsd_angstrom,
            val.pre_dock_rdc_q_work,
            val.pre_dock_rdc_q_free,
            val.rdc_q_work,
            val.rdc_q_free
        ));
    }
    if !report.tolman_gdo.domain_fits.is_empty() {
        out.push_str(
            "# tolman_gdo_table\ndomain_id\tmedium_id\tn_work\tgdo_theta\tgdo_sigma_mc\tratio_to_dom1\trho_gdo\trho_sigma_mc\tda_nh_hz\trhombicity\n",
        );
        for fit in &report.tolman_gdo.domain_fits {
            out.push_str(&format!(
                "{}\t{}\t{}\t{:.6e}\t{:.6e}\t{:.4}\t{:.4}\t{:.4}\t{:.3}\t{:.4}\n",
                fit.domain_id,
                fit.medium_id,
                fit.n_work,
                fit.gdo_theta,
                fit.gdo_sigma_mc,
                fit.ratio_to_domain_1,
                fit.rho_gdo,
                fit.rho_sigma_mc,
                fit.da_nh_hz,
                fit.rhombicity
            ));
        }
    }
    out
}
