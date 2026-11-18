//! Lipari–Szabo / Clore / Mandel–Akke–Palmer Model-Free (`M1`–`M5`) internal dynamics and
//! Isotropic, Axially Symmetric (prolate & oblate), and Fully Anisotropic rotational diffusion
//! tensor optimization engine (`T13.3`, `docs/specs/16-dynamics-and-csp.md` §2).
//!
//! Provenance: Clean-room implementation strictly from peer-reviewed NMR spin-relaxation,
//! rotational diffusion, and statistical model-selection literature (zero external GPL, AGPL, or
//! academic-licence software consulted; zero external product names in identifiers):
//! - Lipari, G. & Szabo, A. (*J. Am. Chem. Soc.* **1982a**, 104, 4546–4559; **1982b**, 104,
//!   4559–4570) — model-free generalized order parameter $S^2$ and effective internal correlation
//!   time $\tau_e$ (`M1`, `M2`).
//! - Clore, G. M., Szabo, A., Bax, A., Kay, L. E., Driscoll, P. C. & Gronenborn, A. M. (*J. Am.
//!   Chem. Soc.* **1990**, 112, 4989–4991) — extended two-timescale model-free formalism (`M5`:
//!   $S^2 = S_f^2 S_s^2$, slow internal correlation time $\tau_s$).
//! - Kay, L. E., Torchia, D. A. & Bax, A. (*Biochemistry* **1989**, 28, 8972–8979) — $^{15}\text{N}$
//!   $T_1, T_2, \text{NOE}$ equations and $R_2 / R_1$ correlation-time estimation.
//! - Farrow, N. A. et al. (*Biochemistry* **1994**, 33, 5984–6003 & *J. Biomol. NMR* **1995**, 6,
//!   153–162) — multi-observable $^{15}\text{N}$ backbone relaxation analysis.
//! - Mandel, A. M., Akke, M. & Palmer, A. G. III (*J. Mol. Biol.* **1995**, 246, 144–163) —
//!   chemical exchange $R_{\text{ex}}$ models (`M3`, `M4`), parameter identifiability, Monte Carlo
//!   error analysis, and nested $F$-test model selection.
//! - Woessner, D. E. (*J. Chem. Phys.* **1962**, 37, 647–654) & Lee, L. K., Rance, M., Chazin,
//!   W. J. & Palmer, A. G. III (*J. Biomol. NMR* **1997**, 9, 287–298) — exact 3-component axially
//!   symmetric (prolate & oblate) and 5-component fully anisotropic rotational diffusion spectral
//!   densities.
//! - d'Auvergne, E. J. & Gooley, P. R. (*J. Biomol. NMR* **2003**, 25, 25–39 & **2006**, 35,
//!   117–135) — AIC / BIC model selection, model elimination of boundary/unidentifiable dynamic
//!   fits, and global diffusion + local internal motion optimization.

use std::collections::BTreeMap;
use std::f64::consts::PI;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::noesy::{parse_pdb_first_model, v_norm, v_scale, v_sub, PdbChain};
use crate::relax_fit::{
    estimate_initial_axial_diffusion_from_r2_r1, estimate_trimmed_tau_m, woessner_axial_components,
    RelaxPhysicsConfig, ResidueRelaxRates, TrimmedTauMConfig,
};
use crate::rng::Rng;

// ============================================================================
// Canonical Model-Free Models (M1..M5) & Selection Criteria
// ============================================================================

/// Canonical Lipari–Szabo / Clore / Palmer per-residue internal dynamics models (`M1`–`M5`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ModelFreeId {
    /// `M1` ($k = 1$): $(S^2)$ — very fast internal motion ($\tau_e \to 0$), no exchange ($R_{\text{ex}} = 0$).
    M1,
    /// `M2` ($k = 2$): $(S^2, \tau_e)$ — single detectable internal correlation time, $R_{\text{ex}} = 0$.
    M2,
    /// `M3` ($k = 2$): $(S^2, R_{\text{ex}})$ — fast internal motion plus $\mu\text{s–ms}$ chemical exchange $R_{\text{ex}} \propto B_0^2$.
    M3,
    /// `M4` ($k = 3$): $(S^2, \tau_e, R_{\text{ex}})$ — detectable internal $\tau_e$ plus chemical exchange $R_{\text{ex}} \propto B_0^2$.
    M4,
    /// `M5` ($k = 3$): $(S_f^2, S_s^2, \tau_s)$ where $S^2 = S_f^2 S_s^2$ and $\tau_f \to 0$ (Clore et al. 1990).
    M5,
}

impl ModelFreeId {
    pub const ALL: [Self; 5] = [Self::M1, Self::M2, Self::M3, Self::M4, Self::M5];

    /// Number of fitted local per-residue parameters $k$.
    #[inline]
    pub fn num_params(self) -> usize {
        match self {
            Self::M1 => 1,
            Self::M2 | Self::M3 => 2,
            Self::M4 | Self::M5 => 3,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::M1 => "M1",
            Self::M2 => "M2",
            Self::M3 => "M3",
            Self::M4 => "M4",
            Self::M5 => "M5",
        }
    }
}

/// Statistical criterion for per-residue `M1`–`M5` and global diffusion tensor model selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelSelectionCriterion {
    /// Akaike Information Criterion: $\mathrm{AIC} = \chi^2 + 2 k$ (d'Auvergne & Gooley 2003).
    Aic,
    /// Small-sample corrected Akaike Information Criterion $\mathrm{AIC}_c$.
    Aicc,
    /// Bayesian Information Criterion: $\mathrm{BIC} = \chi^2 + k \ln n$.
    Bic,
    /// Mandel, Akke & Palmer (1995) hierarchical $\chi^2$ / $F$-test ($\alpha \approx 0.05$).
    PalmerFTest,
}

// ============================================================================
// Rotational Diffusion Tensor (Isotropic, Axial Prolate/Oblate, Fully Anisotropic)
// ============================================================================

/// Requested global rotational diffusion tensor symmetry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiffusionTensorKind {
    /// Spherical rotor: single correlation time $\tau_m = (6 D_{\text{iso}})^{-1}$ (1 global parameter).
    Isotropic,
    /// Axially symmetric rotor (prolate $D_\parallel > D_\perp$ or oblate $D_\parallel < D_\perp$):
    /// $(D_\perp, D_\parallel, \theta, \phi)$ (4 global parameters, requires PDB $\text{N–H}$ vectors).
    AxialSymmetric,
    /// Fully asymmetric rotor: $(D_x, D_y, D_z, \alpha, \beta, \gamma)$ with ZYZ Euler angles
    /// (6 global parameters, requires PDB $\text{N–H}$ vectors).
    FullyAnisotropic,
    /// Fit all applicable diffusion symmetries and select the global minimum under AIC/BIC.
    Auto,
}

/// Converged global rotational diffusion tensor.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DiffusionTensor {
    Isotropic {
        tau_m_sec: f64,
    },
    AxialSymmetric {
        d_perp_s_inv: f64,
        d_par_s_inv: f64,
        theta_rad: f64,
        phi_rad: f64,
    },
    FullyAnisotropic {
        dx_s_inv: f64,
        dy_s_inv: f64,
        dz_s_inv: f64,
        alpha_rad: f64,
        beta_rad: f64,
        gamma_rad: f64,
    },
}

impl DiffusionTensor {
    /// Number of global diffusion tensor degrees of freedom.
    pub fn num_global_params(&self) -> usize {
        match self {
            Self::Isotropic { .. } => 1,
            Self::AxialSymmetric { .. } => 4,
            Self::FullyAnisotropic { .. } => 6,
        }
    }

    /// Effective isotropic correlation time $\tau_m = (6 D_{\text{iso}})^{-1}$ in seconds.
    pub fn effective_tau_m_sec(&self) -> f64 {
        match *self {
            Self::Isotropic { tau_m_sec } => tau_m_sec,
            Self::AxialSymmetric {
                d_perp_s_inv,
                d_par_s_inv,
                ..
            } => {
                let d_iso = (2.0 * d_perp_s_inv + d_par_s_inv) / 3.0;
                1.0 / (6.0 * d_iso).max(1.0e-3)
            }
            Self::FullyAnisotropic {
                dx_s_inv,
                dy_s_inv,
                dz_s_inv,
                ..
            } => {
                let d_iso = (dx_s_inv + dy_s_inv + dz_s_inv) / 3.0;
                1.0 / (6.0 * d_iso).max(1.0e-3)
            }
        }
    }

    /// Evaluate exact Woessner (1962) / Lee et al. (1997) rotational correlation weights and time
    /// constants `[(c_a, tau_a_sec)]` ($\sum_a c_a = 1$) for unit bond vector `u_opt` in the PDB frame.
    pub fn rotational_components(&self, u_opt: Option<[f64; 3]>) -> Vec<(f64, f64)> {
        match *self {
            Self::Isotropic { tau_m_sec } => vec![(1.0, tau_m_sec)],
            Self::AxialSymmetric {
                d_perp_s_inv,
                d_par_s_inv,
                theta_rad,
                phi_rad,
            } => {
                let Some(u) = u_opt else {
                    return vec![(1.0, self.effective_tau_m_sec())];
                };
                let (st, ct) = theta_rad.sin_cos();
                let (sp, cp) = phi_rad.sin_cos();
                let axis = [st * cp, st * sp, ct];
                woessner_axial_components(u, axis, d_perp_s_inv, d_par_s_inv).to_vec()
            }
            Self::FullyAnisotropic {
                dx_s_inv,
                dy_s_inv,
                dz_s_inv,
                alpha_rad,
                beta_rad,
                gamma_rad,
            } => {
                let Some(u) = u_opt else {
                    return vec![(1.0, self.effective_tau_m_sec())];
                };
                woessner_anisotropic_components(
                    u,
                    [dx_s_inv, dy_s_inv, dz_s_inv],
                    [alpha_rad, beta_rad, gamma_rad],
                )
                .to_vec()
            }
        }
    }
}

/// Active ZYZ Euler rotation matrix $R(\alpha, \beta, \gamma) = R_z(\gamma) R_y(\beta) R_z(\alpha)$
/// mapping molecular-frame coordinates $\mathbf{u}$ into the principal diffusion tensor frame $\mathbf{e} = R \mathbf{u}$.
#[inline]
pub fn euler_zyz_matrix(alpha: f64, beta: f64, gamma: f64) -> [[f64; 3]; 3] {
    let (sa, ca) = alpha.sin_cos();
    let (sb, cb) = beta.sin_cos();
    let (sg, cg) = gamma.sin_cos();
    [
        [cg * cb * ca - sg * sa, cg * cb * sa + sg * ca, -cg * sb],
        [-sg * cb * ca - cg * sa, -sg * cb * sa + cg * ca, sg * sb],
        [sb * ca, sb * sa, cb],
    ]
}

/// Exact 5-component Woessner (1962) / Lee et al. (1997) asymmetric rotor weights and correlation
/// times `[(c_1, tau_1), ..., (c_5, tau_5)]` ($\sum_{a=1}^5 c_a = 1$).
pub fn woessner_anisotropic_components(
    u_mol: [f64; 3],
    d_xyz: [f64; 3],
    euler_zyz: [f64; 3],
) -> [(f64, f64); 5] {
    let nrm = (u_mol[0] * u_mol[0] + u_mol[1] * u_mol[1] + u_mol[2] * u_mol[2])
        .sqrt()
        .max(1.0e-12);
    let u = [u_mol[0] / nrm, u_mol[1] / nrm, u_mol[2] / nrm];
    let rot = euler_zyz_matrix(euler_zyz[0], euler_zyz[1], euler_zyz[2]);
    let ex = rot[0][0] * u[0] + rot[0][1] * u[1] + rot[0][2] * u[2];
    let ey = rot[1][0] * u[0] + rot[1][1] * u[1] + rot[1][2] * u[2];
    let ez = rot[2][0] * u[0] + rot[2][1] * u[1] + rot[2][2] * u[2];

    let (dx, dy, dz) = (d_xyz[0], d_xyz[1], d_xyz[2]);
    let ds = (dx + dy + dz) / 3.0;
    let delta_sq = (dx * dx + dy * dy + dz * dz - dx * dy - dx * dz - dy * dz).max(0.0);
    let delta = delta_sq.sqrt();

    let ex2 = ex * ex;
    let ey2 = ey * ey;
    let ez2 = ez * ez;

    let c1 = 3.0 * ey2 * ez2;
    let c2 = 3.0 * ex2 * ez2;
    let c3 = 3.0 * ex2 * ey2;
    let tau1 = 1.0 / (4.0 * dx + dy + dz).max(1.0e-3);
    let tau2 = 1.0 / (dx + 4.0 * dy + dz).max(1.0e-3);
    let tau3 = 1.0 / (dx + dy + 4.0 * dz).max(1.0e-3);

    let p_sym = 0.5 * (3.0 * (ex2 * ex2 + ey2 * ey2 + ez2 * ez2) - 1.0);
    if delta <= 1.0e-11 * ds.max(1.0) {
        let tau_iso = 1.0 / (6.0 * ds).max(1.0e-3);
        return [
            (c1, tau1),
            (c2, tau2),
            (c3, tau3),
            (0.5 * p_sym, tau_iso),
            (0.5 * p_sym, tau_iso),
        ];
    }

    let del_x = (dx - ds) / delta;
    let del_y = (dy - ds) / delta;
    let del_z = (dz - ds) / delta;

    let q_asym = 0.5
        * (del_x * (3.0 * ex2 * ex2 + 6.0 * ey2 * ez2 - 1.0)
            + del_y * (3.0 * ey2 * ey2 + 6.0 * ex2 * ez2 - 1.0)
            + del_z * (3.0 * ez2 * ez2 + 6.0 * ex2 * ey2 - 1.0));

    let tau4 = 1.0 / (6.0 * ds + 2.0 * delta).max(1.0e-3);
    let tau5 = 1.0 / (6.0 * ds - 2.0 * delta).max(1.0e-3);
    let c4 = 0.5 * (p_sym - q_asym);
    let c5 = 0.5 * (p_sym + q_asym);

    [(c1, tau1), (c2, tau2), (c3, tau3), (c4, tau4), (c5, tau5)]
}

/// Extract normalized $\text{N–H}$ unit bond vectors `(seq_id, [ux, uy, uz])` from a parsed `PdbChain`.
pub fn extract_nh_unit_vectors_from_chain(chain: &PdbChain) -> Vec<(i32, [f64; 3])> {
    let mut out = Vec::with_capacity(chain.residues.len());
    for res in &chain.residues {
        if res.res_name.eq_ignore_ascii_case("PRO") {
            continue;
        }
        if let Some(h_pos) = res.h {
            let d = v_sub(h_pos, res.n);
            let nrm = v_norm(d);
            if nrm > 0.4 {
                out.push((res.seq_id, v_scale(d, 1.0 / nrm)));
            }
        }
    }
    out
}

/// Parse the first model of a PDB file and extract backbone amide $\text{N–H}$ unit vectors.
pub fn extract_nh_unit_vectors_from_pdb(path: &Path) -> Result<Vec<(i32, [f64; 3])>, String> {
    let chain = parse_pdb_first_model(path)?;
    Ok(extract_nh_unit_vectors_from_chain(&chain))
}

// ============================================================================
// Forward Multi-Field Lipari–Szabo / Clore / Palmer Relaxation Equations
// ============================================================================

/// Local per-residue Lipari–Szabo / Clore / Palmer parameter set.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LocalMfParams {
    /// Generalized order parameter $S^2 = S_f^2 S_s^2 \in [0, 1]$.
    pub s2: f64,
    /// Fast order parameter $S_f^2 \in [0, 1]$ (`1.0` for `M1`–`M4`, fitted in `M5`).
    pub sf2: f64,
    /// Slow order parameter $S_s^2 \in [0, 1]$ (`s2` for `M1`–`M4`, fitted in `M5`).
    pub ss2: f64,
    /// Internal correlation time $\tau_e$ (or $\tau_s$ for `M5`) in seconds (`0.0` for `M1` & `M3`).
    pub tau_e_sec: f64,
    /// Chemical exchange contribution $R_{\text{ex,ref}}$ ($\text{s}^{-1}$) referenced at `ref_field_mhz`
    /// (`0.0` for `M1`, `M2`, `M5`), scaling quadratically as $R_{\text{ex}}(B_0) = R_{\text{ex,ref}} (B_0 / B_{0,\text{ref}})^2$.
    pub rex_ref_s_inv: f64,
}

impl LocalMfParams {
    pub fn m1(s2: f64) -> Self {
        Self {
            s2,
            sf2: 1.0,
            ss2: s2,
            tau_e_sec: 0.0,
            rex_ref_s_inv: 0.0,
        }
    }

    pub fn m2(s2: f64, tau_e_sec: f64) -> Self {
        Self {
            s2,
            sf2: 1.0,
            ss2: s2,
            tau_e_sec,
            rex_ref_s_inv: 0.0,
        }
    }

    pub fn m3(s2: f64, rex_ref_s_inv: f64) -> Self {
        Self {
            s2,
            sf2: 1.0,
            ss2: s2,
            tau_e_sec: 0.0,
            rex_ref_s_inv,
        }
    }

    pub fn m4(s2: f64, tau_e_sec: f64, rex_ref_s_inv: f64) -> Self {
        Self {
            s2,
            sf2: 1.0,
            ss2: s2,
            tau_e_sec,
            rex_ref_s_inv,
        }
    }

    pub fn m5(sf2: f64, ss2: f64, tau_s_sec: f64) -> Self {
        Self {
            s2: sf2 * ss2,
            sf2,
            ss2,
            tau_e_sec: tau_s_sec,
            rex_ref_s_inv: 0.0,
        }
    }
}

/// Evaluate the Lipari–Szabo / Clore spectral density $J(\omega)$ ($\text{s rad}^{-1}$) at angular
/// frequency `omega_rad_s` given rotational diffusion modes `rot_comps = [(c_a, tau_a)]` and `local`.
#[inline]
pub fn spectral_density_modelfree(
    omega_rad_s: f64,
    rot_comps: &[(f64, f64)],
    local: &LocalMfParams,
) -> f64 {
    let s2 = local.s2;
    let amp_slow = (local.sf2 - s2).max(0.0); // equals `1 - S^2` for M1..M4 (`sf2 == 1`) and `S_f^2 (1 - S_s^2)` for M5!
    let te = local.tau_e_sec;

    let mut acc = 0.0_f64;
    for &(ca, tau_a) in rot_comps {
        let w_ta = omega_rad_s * tau_a;
        let mut term = s2 * tau_a / (1.0 + w_ta * w_ta);
        if te > 1.0e-16 && amp_slow > 0.0 {
            let tau_eff = (tau_a * te) / (tau_a + te);
            let w_te = omega_rad_s * tau_eff;
            term += amp_slow * tau_eff / (1.0 + w_te * w_te);
        }
        acc += ca * term;
    }
    0.4 * acc
}

/// Predict $^{15}\text{N}$ $(R_1, R_2, \mathrm{hetNOE})$ at static proton field `field_mhz` from
/// `rot_comps`, `local` parameters, `ref_field_mhz`, and `phys`.
pub fn predict_r1_r2_noe(
    rot_comps: &[(f64, f64)],
    local: &LocalMfParams,
    field_mhz: f64,
    ref_field_mhz: f64,
    phys: &RelaxPhysicsConfig,
) -> (f64, f64, f64) {
    let d = phys.dipolar_d_rad_s();
    let d2 = d * d;
    let c = phys.csa_c_rad_s(field_mhz);
    let c2 = c * c;
    let (wh, wn) = phys.signed_omegas_rad_s(field_mhz);

    let j0 = spectral_density_modelfree(0.0, rot_comps, local);
    let jn = spectral_density_modelfree(wn, rot_comps, local);
    let jh = spectral_density_modelfree(wh, rot_comps, local);
    let j_diff = spectral_density_modelfree(wh - wn, rot_comps, local);
    let j_sum = spectral_density_modelfree(wh + wn, rot_comps, local);

    let r1 = 0.25 * d2 * (j_diff + 3.0 * jn + 6.0 * j_sum) + c2 * jn;
    let field_ratio = field_mhz / ref_field_mhz.max(1.0);
    let rex_field = local.rex_ref_s_inv * field_ratio * field_ratio;
    let r2 = 0.125 * d2 * (4.0 * j0 + j_diff + 3.0 * jn + 6.0 * jh + 6.0 * j_sum)
        + (c2 / 6.0) * (4.0 * j0 + 3.0 * jn)
        + rex_field;
    let sigma_nh = 0.25 * d2 * (6.0 * j_sum - j_diff);
    let noe = 1.0 + (phys.gamma_h / phys.gamma_n) * (sigma_nh / r1.max(1.0e-30));

    (r1, r2, noe)
}

// ============================================================================
// Multi-Field Dataset Input, Per-Residue M1..M5 Fitting & Model Selection
// ============================================================================

/// Multi-field experimental relaxation observables for a single residue.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MultiFieldResidueInput {
    pub seq_id: i32,
    pub res_name: String,
    /// Optional unit $\text{N–H}$ bond vector in the PDB frame (required for axial/anisotropic diffusion).
    pub nh_unit_vec: Option<[f64; 3]>,
    /// Measurements at one or more static magnetic fields (e.g. `600 MHz` + `800 MHz`).
    pub fields: Vec<ResidueRelaxRates>,
}

impl MultiFieldResidueInput {
    pub fn total_observables(&self) -> usize {
        self.fields
            .iter()
            .map(|f| 2 + usize::from(f.hetnoe.is_some()))
            .sum()
    }
}

/// Configuration for Lipari–Szabo / Palmer model-free fitting and global diffusion optimization.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelFreeConfig {
    pub diffusion_kind: DiffusionTensorKind,
    pub selection_criterion: ModelSelectionCriterion,
    /// Reference proton Larmor frequency (MHz) for reporting $R_{\text{ex,ref}}$ (`None` = minimum field in dataset).
    pub ref_field_mhz: Option<f64>,
    pub physics: RelaxPhysicsConfig,
    /// Apply d'Auvergne & Gooley (2003) + Mandel, Akke & Palmer (1995) boundary & Fisher-identifiability model elimination.
    pub enable_model_elimination: bool,
    /// Number of outer global-diffusion <-> local-model alternating passes (default `5`).
    pub max_outer_iters: usize,
    /// Number of parametric Monte Carlo replicates for local parameter uncertainties (`0` = analytical Fisher SE, default `200`).
    pub mc_replicates: usize,
    pub seed: u64,
}

impl Default for ModelFreeConfig {
    fn default() -> Self {
        Self {
            diffusion_kind: DiffusionTensorKind::Isotropic,
            selection_criterion: ModelSelectionCriterion::Aic,
            ref_field_mhz: None,
            physics: RelaxPhysicsConfig::default(),
            enable_model_elimination: true,
            max_outer_iters: 5,
            mc_replicates: 0,
            seed: 2026_1303,
        }
    }
}

/// Fitted result for one candidate model (`M1`–`M5`) on a single residue.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CandidateModelFit {
    pub model: ModelFreeId,
    pub params: LocalMfParams,
    pub sigma_s2: f64,
    pub sigma_sf2: f64,
    pub sigma_ss2: f64,
    pub sigma_tau_e_sec: f64,
    pub sigma_rex_ref: f64,
    pub chi2: f64,
    pub aic: f64,
    pub aicc: f64,
    pub bic: f64,
    /// `true` if eliminated by d'Auvergne & Gooley (2003) / Mandel–Akke–Palmer (1995) physical & identifiability rules.
    pub eliminated: bool,
}

/// Selected model-free dynamic summary for one residue.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResidueModelFreeResult {
    pub seq_id: i32,
    pub res_name: String,
    pub selected_model: ModelFreeId,
    pub params: LocalMfParams,
    pub sigma_s2: f64,
    pub sigma_sf2: f64,
    pub sigma_ss2: f64,
    pub sigma_tau_e_sec: f64,
    pub sigma_rex_ref: f64,
    pub chi2: f64,
    pub dof: usize,
    pub aic: f64,
    pub bic: f64,
    pub candidate_fits: Vec<CandidateModelFit>,
}

/// Complete protein-wide model-free & rotational diffusion tensor result.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelFreeDatasetResult {
    pub diffusion_tensor: DiffusionTensor,
    pub ref_field_mhz: f64,
    pub total_chi2: f64,
    pub global_aic: f64,
    pub global_bic: f64,
    pub outer_iterations: usize,
    pub residues: Vec<ResidueModelFreeResult>,
}

fn solve_small_spd(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![if a[0].abs() > 1.0e-24 {
            b[0] / a[0]
        } else {
            0.0
        }];
    }
    let mut m = vec![0.0_f64; n * (n + 1)];
    for r in 0..n {
        for c in 0..n {
            m[r * (n + 1) + c] = a[r * n + c];
        }
        m[r * (n + 1) + n] = b[r];
    }
    for col in 0..n {
        let mut piv = col;
        let mut mv = m[col * (n + 1) + col].abs();
        for r in (col + 1)..n {
            let v = m[r * (n + 1) + col].abs();
            if v > mv {
                mv = v;
                piv = r;
            }
        }
        if mv <= 1.0e-26 {
            return vec![0.0; n];
        }
        if piv != col {
            for c in 0..=n {
                m.swap(col * (n + 1) + c, piv * (n + 1) + c);
            }
        }
        let inv = 1.0 / m[col * (n + 1) + col];
        for r in (col + 1)..n {
            let f = m[r * (n + 1) + col] * inv;
            for c in col..=n {
                m[r * (n + 1) + c] -= f * m[col * (n + 1) + c];
            }
        }
    }
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut s = m[i * (n + 1) + n];
        for j in (i + 1)..n {
            s -= m[i * (n + 1) + j] * x[j];
        }
        x[i] = s / m[i * (n + 1) + i];
    }
    x
}

fn decode_internal_vec(model: ModelFreeId, q: &[f64], tau_m_eff: f64) -> LocalMfParams {
    match model {
        ModelFreeId::M1 => LocalMfParams::m1(q[0].clamp(0.0, 1.0)),
        ModelFreeId::M2 => {
            let s2 = q[0].clamp(0.0, 1.0);
            let te = q[1].exp().clamp(0.5e-12, 0.50 * tau_m_eff);
            LocalMfParams::m2(s2, te)
        }
        ModelFreeId::M3 => LocalMfParams::m3(q[0].clamp(0.0, 1.0), q[1].clamp(0.0, 80.0)),
        ModelFreeId::M4 => {
            let s2 = q[0].clamp(0.0, 1.0);
            let te = q[1].exp().clamp(0.5e-12, 0.50 * tau_m_eff);
            let rex = q[2].clamp(0.0, 80.0);
            LocalMfParams::m4(s2, te, rex)
        }
        ModelFreeId::M5 => {
            let sf2 = q[0].clamp(0.05, 1.0);
            let ss2 = q[1].clamp(0.05, 1.0);
            let ts = q[2].exp().clamp(100.0e-12, 0.60 * tau_m_eff);
            LocalMfParams::m5(sf2, ss2, ts)
        }
    }
}

fn clamp_internal_vec(model: ModelFreeId, q: &mut [f64], tau_m_eff: f64) {
    match model {
        ModelFreeId::M1 => {
            q[0] = q[0].clamp(0.0, 1.0);
        }
        ModelFreeId::M2 => {
            q[0] = q[0].clamp(0.0, 1.0);
            q[1] = q[1].clamp((0.5e-12_f64).ln(), (0.50 * tau_m_eff).max(2.0e-12).ln());
        }
        ModelFreeId::M3 => {
            q[0] = q[0].clamp(0.0, 1.0);
            q[1] = q[1].clamp(0.0, 80.0);
        }
        ModelFreeId::M4 => {
            q[0] = q[0].clamp(0.0, 1.0);
            q[1] = q[1].clamp((0.5e-12_f64).ln(), (0.50 * tau_m_eff).max(2.0e-12).ln());
            q[2] = q[2].clamp(0.0, 80.0);
        }
        ModelFreeId::M5 => {
            q[0] = q[0].clamp(0.05, 1.0);
            q[1] = q[1].clamp(0.05, 1.0);
            q[2] = q[2].clamp((100.0e-12_f64).ln(), (0.60 * tau_m_eff).max(200.0e-12).ln());
        }
    }
}

fn residuals_for_residue(
    fields: &[ResidueRelaxRates],
    rot_comps: &[(f64, f64)],
    local: &LocalMfParams,
    ref_field_mhz: f64,
    phys: &RelaxPhysicsConfig,
    out: &mut [f64],
) -> f64 {
    let mut idx = 0usize;
    let mut chi2 = 0.0_f64;
    for f in fields {
        let (r1_p, r2_p, noe_p) =
            predict_r1_r2_noe(rot_comps, local, f.field_mhz, ref_field_mhz, phys);
        let z1 = (r1_p - f.r1) / f.sigma_r1.max(1.0e-6);
        let z2 = (r2_p - f.r2) / f.sigma_r2.max(1.0e-6);
        out[idx] = z1;
        out[idx + 1] = z2;
        chi2 += z1 * z1 + z2 * z2;
        idx += 2;
        if let Some(noe_obs) = f.hetnoe {
            let sig_n = f.sigma_hetnoe.unwrap_or(0.02).max(1.0e-5);
            let zn = (noe_p - noe_obs) / sig_n;
            out[idx] = zn;
            chi2 += zn * zn;
            idx += 1;
        }
    }
    chi2
}

#[allow(clippy::too_many_arguments)]
fn polish_local_lm(
    model: ModelFreeId,
    q_init: &[f64],
    fields: &[ResidueRelaxRates],
    rot_comps: &[(f64, f64)],
    tau_m_eff: f64,
    ref_field_mhz: f64,
    phys: &RelaxPhysicsConfig,
    max_iters: usize,
) -> (Vec<f64>, f64) {
    let k = model.num_params();
    let m: usize = fields
        .iter()
        .map(|f| 2 + usize::from(f.hetnoe.is_some()))
        .sum();
    let mut q = q_init.to_vec();
    clamp_internal_vec(model, &mut q, tau_m_eff);

    let mut r_cur = vec![0.0_f64; m];
    let mut r_tmp = vec![0.0_f64; m];
    let mut chi2_cur = residuals_for_residue(
        fields,
        rot_comps,
        &decode_internal_vec(model, &q, tau_m_eff),
        ref_field_mhz,
        phys,
        &mut r_cur,
    );
    let mut lambda = 1.0e-3_f64;

    for _iter in 0..max_iters {
        let mut j_cols = vec![vec![0.0_f64; m]; k];
        for p_idx in 0..k {
            let h = 1.0e-6_f64;
            let mut qp = q.clone();
            qp[p_idx] += h;
            clamp_internal_vec(model, &mut qp, tau_m_eff);
            let dh = qp[p_idx] - q[p_idx];
            if dh.abs() > 1.0e-12 {
                let _ = residuals_for_residue(
                    fields,
                    rot_comps,
                    &decode_internal_vec(model, &qp, tau_m_eff),
                    ref_field_mhz,
                    phys,
                    &mut r_tmp,
                );
                let inv = 1.0 / dh;
                for i in 0..m {
                    j_cols[p_idx][i] = (r_tmp[i] - r_cur[i]) * inv;
                }
            } else {
                let mut qm = q.clone();
                qm[p_idx] -= h;
                clamp_internal_vec(model, &mut qm, tau_m_eff);
                let dm = q[p_idx] - qm[p_idx];
                if dm.abs() > 1.0e-12 {
                    let _ = residuals_for_residue(
                        fields,
                        rot_comps,
                        &decode_internal_vec(model, &qm, tau_m_eff),
                        ref_field_mhz,
                        phys,
                        &mut r_tmp,
                    );
                    let inv = 1.0 / dm;
                    for i in 0..m {
                        j_cols[p_idx][i] = (r_cur[i] - r_tmp[i]) * inv;
                    }
                }
            }
        }

        let mut jtj = vec![0.0_f64; k * k];
        let mut jtr = vec![0.0_f64; k];
        for i in 0..m {
            for a in 0..k {
                jtr[a] -= j_cols[a][i] * r_cur[i];
                for b in 0..k {
                    jtj[a * k + b] += j_cols[a][i] * j_cols[b][i];
                }
            }
        }

        let mut accepted = false;
        for _sub in 0..6 {
            let mut lhs = jtj.clone();
            for d in 0..k {
                lhs[d * k + d] += lambda * jtj[d * k + d].max(1.0e-8);
            }
            let step = solve_small_spd(&lhs, &jtr, k);
            let mut cand = q.clone();
            for d in 0..k {
                cand[d] += step[d];
            }
            clamp_internal_vec(model, &mut cand, tau_m_eff);
            let c2_cand = residuals_for_residue(
                fields,
                rot_comps,
                &decode_internal_vec(model, &cand, tau_m_eff),
                ref_field_mhz,
                phys,
                &mut r_tmp,
            );
            if c2_cand < chi2_cur - 1.0e-14 {
                let impr = chi2_cur - c2_cand;
                q = cand;
                chi2_cur = c2_cand;
                r_cur.copy_from_slice(&r_tmp);
                lambda = (lambda * 0.3).max(1.0e-8);
                accepted = true;
                if impr < 1.0e-11 {
                    return (q, chi2_cur);
                }
                break;
            }
            lambda = (lambda * 6.0).min(1.0e6);
        }
        if !accepted {
            break;
        }
    }

    (q, chi2_cur)
}

/// Compute analytical Fisher $1\sigma$ standard errors on physical local parameters
/// `(sigma_s2, sigma_sf2, sigma_ss2, sigma_tau_e, sigma_rex)` at a converged `LocalMfParams`.
fn local_fisher_uncertainties(
    model: ModelFreeId,
    local: &LocalMfParams,
    fields: &[ResidueRelaxRates],
    rot_comps: &[(f64, f64)],
    ref_field_mhz: f64,
    phys: &RelaxPhysicsConfig,
) -> (f64, f64, f64, f64, f64) {
    let k = model.num_params();
    let m: usize = fields
        .iter()
        .map(|f| 2 + usize::from(f.hetnoe.is_some()))
        .sum();
    let mut r0 = vec![0.0_f64; m];
    let mut rp = vec![0.0_f64; m];
    let _ = residuals_for_residue(fields, rot_comps, local, ref_field_mhz, phys, &mut r0);

    // Differentiate directly in physical coordinates `(S^2, tau_e, R_ex)` or `(S_f^2, S_s^2, tau_s)`
    let phys_vec: Vec<f64> = match model {
        ModelFreeId::M1 => vec![local.s2],
        ModelFreeId::M2 => vec![local.s2, local.tau_e_sec],
        ModelFreeId::M3 => vec![local.s2, local.rex_ref_s_inv],
        ModelFreeId::M4 => vec![local.s2, local.tau_e_sec, local.rex_ref_s_inv],
        ModelFreeId::M5 => vec![local.sf2, local.ss2, local.tau_e_sec],
    };
    let steps: Vec<f64> = match model {
        ModelFreeId::M1 => vec![1.0e-5],
        ModelFreeId::M2 => vec![1.0e-5, (1.0e-4 * local.tau_e_sec).max(1.0e-14)],
        ModelFreeId::M3 => vec![1.0e-5, 1.0e-4],
        ModelFreeId::M4 => vec![1.0e-5, (1.0e-4 * local.tau_e_sec).max(1.0e-14), 1.0e-4],
        ModelFreeId::M5 => vec![1.0e-5, 1.0e-5, (1.0e-4 * local.tau_e_sec).max(1.0e-13)],
    };

    let mut j_cols = vec![vec![0.0_f64; m]; k];
    for c in 0..k {
        let mut pv = phys_vec.clone();
        let h = if pv[c] + steps[c] <= 1.0
            || c > 0 && !(model == ModelFreeId::M5 && c == 1 && pv[c] + steps[c] > 1.0)
        {
            steps[c]
        } else {
            -steps[c]
        };
        pv[c] += h;
        let pert = match model {
            ModelFreeId::M1 => LocalMfParams::m1(pv[0]),
            ModelFreeId::M2 => LocalMfParams::m2(pv[0], pv[1].max(1.0e-15)),
            ModelFreeId::M3 => LocalMfParams::m3(pv[0], pv[1]),
            ModelFreeId::M4 => LocalMfParams::m4(pv[0], pv[1].max(1.0e-15), pv[2]),
            ModelFreeId::M5 => LocalMfParams::m5(pv[0], pv[1], pv[2].max(1.0e-14)),
        };
        let _ = residuals_for_residue(fields, rot_comps, &pert, ref_field_mhz, phys, &mut rp);
        let inv_h = 1.0 / h;
        for i in 0..m {
            j_cols[c][i] = (rp[i] - r0[i]) * inv_h;
        }
    }

    let mut fim = vec![0.0_f64; k * k];
    for (a, col_a) in j_cols.iter().enumerate() {
        for (b, col_b) in j_cols.iter().enumerate() {
            fim[a * k + b] = col_a.iter().zip(col_b.iter()).map(|(x, y)| x * y).sum();
        }
    }
    // Tiny ridge safeguard for singular boundary derivatives
    for d in 0..k {
        fim[d * k + d] += 1.0e-18;
    }
    let mut cov = vec![0.0_f64; k * k];
    for col in 0..k {
        let mut e = vec![0.0_f64; k];
        e[col] = 1.0;
        let sol = solve_small_spd(&fim, &e, k);
        for row in 0..k {
            cov[row * k + col] = sol[row];
        }
    }

    match model {
        ModelFreeId::M1 => {
            let s = cov[0].max(0.0).sqrt();
            (s, 0.0, s, 0.0, 0.0)
        }
        ModelFreeId::M2 => {
            let s = cov[0].max(0.0).sqrt();
            let ste = cov[3].max(0.0).sqrt();
            (s, 0.0, s, ste, 0.0)
        }
        ModelFreeId::M3 => {
            let s = cov[0].max(0.0).sqrt();
            let srex = cov[3].max(0.0).sqrt();
            (s, 0.0, s, 0.0, srex)
        }
        ModelFreeId::M4 => {
            let s = cov[0].max(0.0).sqrt();
            let ste = cov[4].max(0.0).sqrt();
            let srex = cov[8].max(0.0).sqrt();
            (s, 0.0, s, ste, srex)
        }
        ModelFreeId::M5 => {
            let s_sf = cov[0].max(0.0).sqrt();
            let s_ss = cov[4].max(0.0).sqrt();
            let s_ts = cov[8].max(0.0).sqrt();
            // Propagate variance of S^2 = S_f^2 * S_s^2 including off-diagonal covariance cov[1]!
            let var_s2 = local.ss2 * local.ss2 * cov[0]
                + local.sf2 * local.sf2 * cov[4]
                + 2.0 * local.sf2 * local.ss2 * cov[1];
            (var_s2.max(0.0).sqrt(), s_sf, s_ss, s_ts, 0.0)
        }
    }
}

/// Fit one specific model (`M1`–`M5`) on a single residue using deterministic multi-start + LM.
pub fn fit_single_model_on_residue(
    model: ModelFreeId,
    res: &MultiFieldResidueInput,
    rot_comps: &[(f64, f64)],
    tau_m_eff: f64,
    ref_field_mhz: f64,
    cfg: &ModelFreeConfig,
) -> CandidateModelFit {
    let m_obs = res.total_observables();

    let eval_start = |q: &[f64]| -> f64 {
        let local = decode_internal_vec(model, q, tau_m_eff);
        let mut chi2 = 0.0_f64;
        for f in &res.fields {
            let (r1_p, r2_p, noe_p) =
                predict_r1_r2_noe(rot_comps, &local, f.field_mhz, ref_field_mhz, &cfg.physics);
            let z1 = (r1_p - f.r1) / f.sigma_r1.max(1.0e-6);
            let z2 = (r2_p - f.r2) / f.sigma_r2.max(1.0e-6);
            chi2 += z1 * z1 + z2 * z2;
            if let Some(noe_obs) = f.hetnoe {
                let sig_n = f.sigma_hetnoe.unwrap_or(0.02).max(1.0e-5);
                let zn = (noe_p - noe_obs) / sig_n;
                chi2 += zn * zn;
            }
        }
        chi2
    };

    let best_q_and_chi2 = match model {
        ModelFreeId::M1 => {
            let mut best = (vec![0.85], f64::INFINITY);
            for g in 1..=19 {
                let s2 = (g as f64) * 0.05;
                let c2 = eval_start(&[s2]);
                if c2 < best.1 {
                    best = (vec![s2], c2);
                }
            }
            polish_local_lm(
                model,
                &best.0,
                &res.fields,
                rot_comps,
                tau_m_eff,
                ref_field_mhz,
                &cfg.physics,
                28,
            )
        }
        ModelFreeId::M2 => {
            let mut starts: Vec<[f64; 2]> = Vec::new();
            for &s2 in &[0.35_f64, 0.65, 0.82, 0.92] {
                for &te in &[
                    4.0e-12_f64,
                    20.0e-12,
                    65.0e-12,
                    220.0e-12,
                    800.0e-12,
                    2.0e-9,
                ] {
                    if te < 0.48 * tau_m_eff {
                        starts.push([s2, te.ln()]);
                    }
                }
            }
            starts.sort_by(|a, b| eval_start(a).total_cmp(&eval_start(b)));
            let mut best = (starts[0].to_vec(), f64::INFINITY);
            for st in starts.iter().take(3) {
                let cand = polish_local_lm(
                    model,
                    st,
                    &res.fields,
                    rot_comps,
                    tau_m_eff,
                    ref_field_mhz,
                    &cfg.physics,
                    36,
                );
                if cand.1 < best.1 {
                    best = cand;
                }
            }
            best
        }
        ModelFreeId::M3 => {
            let mut starts: Vec<[f64; 2]> = Vec::new();
            for &s2 in &[0.55_f64, 0.78, 0.88, 0.95] {
                for &rex in &[0.2_f64, 1.0, 3.0, 7.0, 15.0] {
                    starts.push([s2, rex]);
                }
            }
            starts.sort_by(|a, b| eval_start(a).total_cmp(&eval_start(b)));
            let mut best = (starts[0].to_vec(), f64::INFINITY);
            for st in starts.iter().take(2) {
                let cand = polish_local_lm(
                    model,
                    st,
                    &res.fields,
                    rot_comps,
                    tau_m_eff,
                    ref_field_mhz,
                    &cfg.physics,
                    32,
                );
                if cand.1 < best.1 {
                    best = cand;
                }
            }
            best
        }
        ModelFreeId::M4 => {
            let mut starts: Vec<[f64; 3]> = Vec::new();
            for &s2 in &[0.55_f64, 0.78, 0.88] {
                for &te in &[15.0e-12_f64, 65.0e-12, 300.0e-12, 1.2e-9] {
                    if te < 0.48 * tau_m_eff {
                        for &rex in &[0.8_f64, 3.0, 8.0] {
                            starts.push([s2, te.ln(), rex]);
                        }
                    }
                }
            }
            starts.sort_by(|a, b| eval_start(a).total_cmp(&eval_start(b)));
            let mut best = (starts[0].to_vec(), f64::INFINITY);
            for st in starts.iter().take(3) {
                let cand = polish_local_lm(
                    model,
                    st,
                    &res.fields,
                    rot_comps,
                    tau_m_eff,
                    ref_field_mhz,
                    &cfg.physics,
                    40,
                );
                if cand.1 < best.1 {
                    best = cand;
                }
            }
            best
        }
        ModelFreeId::M5 => {
            let mut starts: Vec<[f64; 3]> = Vec::new();
            for &sf2 in &[0.74_f64, 0.85, 0.93] {
                for &ss2 in &[0.42_f64, 0.65, 0.82] {
                    for &ts in &[0.25e-9_f64, 0.60e-9, 1.15e-9, 2.10e-9] {
                        if ts < 0.55 * tau_m_eff {
                            starts.push([sf2, ss2, ts.ln()]);
                        }
                    }
                }
            }
            if starts.is_empty() {
                starts.push([0.85, 0.75, (0.25 * tau_m_eff).max(120.0e-12).ln()]);
            }
            starts.sort_by(|a, b| eval_start(a).total_cmp(&eval_start(b)));
            let mut best = (starts[0].to_vec(), f64::INFINITY);
            for st in starts.iter().take(4) {
                let cand = polish_local_lm(
                    model,
                    st,
                    &res.fields,
                    rot_comps,
                    tau_m_eff,
                    ref_field_mhz,
                    &cfg.physics,
                    45,
                );
                if cand.1 < best.1 {
                    best = cand;
                }
            }
            best
        }
    };

    let (q_opt, chi2) = best_q_and_chi2;
    let params = decode_internal_vec(model, &q_opt, tau_m_eff);
    let (sigma_s2, sigma_sf2, sigma_ss2, sigma_tau_e_sec, sigma_rex_ref) =
        local_fisher_uncertainties(
            model,
            &params,
            &res.fields,
            rot_comps,
            ref_field_mhz,
            &cfg.physics,
        );

    let k = model.num_params();
    let kf = k as f64;
    let nf = (m_obs.max(1)) as f64;
    let aic = chi2 + 2.0 * kf;
    let aicc = if m_obs <= k + 1 {
        f64::INFINITY
    } else {
        aic + (2.0 * kf * (kf + 1.0)) / ((m_obs - k - 1) as f64)
    };
    let bic = chi2 + kf * nf.ln();

    // d'Auvergne & Gooley (2003, 2006) + Mandel, Akke & Palmer (1995) model elimination:
    // Eliminate over-parameterized fits that hit physical boundaries or have unidentifiable
    // non-M1 parameters (`param / sigma_Fisher < z_min` or underdetermined `k > m_obs`).
    let eliminated = if k > m_obs {
        true
    } else if !cfg.enable_model_elimination || model == ModelFreeId::M1 {
        false
    } else {
        match model {
            ModelFreeId::M1 => false,
            ModelFreeId::M2 => {
                params.s2 > 0.985
                    || params.tau_e_sec < 2.0e-12
                    || params.tau_e_sec > 0.48 * tau_m_eff
                    || sigma_tau_e_sec > 0.55 * params.tau_e_sec
            }
            ModelFreeId::M3 => {
                params.rex_ref_s_inv < 0.20 || sigma_rex_ref > 0.50 * params.rex_ref_s_inv
            }
            ModelFreeId::M4 => {
                params.s2 > 0.985
                    || params.tau_e_sec < 2.0e-12
                    || params.tau_e_sec > 0.48 * tau_m_eff
                    || sigma_tau_e_sec > 0.55 * params.tau_e_sec
                    || params.rex_ref_s_inv < 0.20
                    || sigma_rex_ref > 0.50 * params.rex_ref_s_inv
            }
            ModelFreeId::M5 => {
                // Clore et al. (J. Am. Chem. Soc. 1990, 112, 4989-4991);
                // Mandel, Akke & Palmer (J. Mol. Biol. 1995, 246, 144-163);
                // d'Auvergne & Gooley (J. Biomol. NMR 2003, 25, 25-39 & 2006, 35, 117-135):
                // Extended two-timescale model M5 requires non-degenerate fast and slow amplitudes
                // and an identifiable slow internal correlation time tau_s.
                let slow_weight = params.sf2 - params.s2;
                params.sf2 > 0.955
                    || params.ss2 > 0.86
                    || slow_weight < 0.12
                    || params.tau_e_sec < 180.0e-12
                    || params.tau_e_sec > 0.50 * tau_m_eff
                    || sigma_tau_e_sec > 0.42 * params.tau_e_sec
                    || sigma_sf2 > 0.48 * (1.0 - params.sf2)
                    || sigma_ss2 > 0.48 * (1.0 - params.ss2)
            }
        }
    };

    CandidateModelFit {
        model,
        params,
        sigma_s2,
        sigma_sf2,
        sigma_ss2,
        sigma_tau_e_sec,
        sigma_rex_ref,
        chi2,
        aic,
        aicc,
        bic,
        eliminated,
    }
}

/// Fit all 5 canonical models `M1`–`M5` on `res` and select the optimal model under `cfg.selection_criterion`.
pub fn fit_and_select_residue_modelfree(
    res: &MultiFieldResidueInput,
    tensor: &DiffusionTensor,
    ref_field_mhz: f64,
    cfg: &ModelFreeConfig,
) -> ResidueModelFreeResult {
    let rot_comps = tensor.rotational_components(res.nh_unit_vec);
    let tau_m_eff = tensor.effective_tau_m_sec();
    let m_obs = res.total_observables();

    let candidates: Vec<CandidateModelFit> = ModelFreeId::ALL
        .iter()
        .map(|&m| fit_single_model_on_residue(m, res, &rot_comps, tau_m_eff, ref_field_mhz, cfg))
        .collect();

    let chosen_idx = match cfg.selection_criterion {
        ModelSelectionCriterion::Aic => {
            let raw_best = candidates
                .iter()
                .enumerate()
                .filter(|(_, c)| !c.eliminated)
                .min_by(|a, b| a.1.aic.total_cmp(&b.1.aic))
                .map(|(i, _)| i)
                .unwrap_or(0);
            // Finite-sample Hurvich-Tsai AICc parsimony guard on non-nested 3-parameter M5
            // (Hurvich & Tsai, Biometrika 1989, 76, 297-307; Burnham & Anderson 2002):
            // at 1-3 static fields (`m_obs <= 9`), asymptotic AIC (`Delta chi^2 > 2`) under-penalizes
            // splitting `S^2` into `S_f^2 * S_s^2`; require M5 to also improve small-sample AICc
            // (or BIC when `m_obs <= 4`) over the best uneliminated M1-M4 model.
            if raw_best == 4 && cfg.enable_model_elimination && m_obs <= 9 {
                let best_non_m5 = candidates[..4]
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| !c.eliminated)
                    .min_by(|a, b| a.1.aic.total_cmp(&b.1.aic))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                let c_m5 = &candidates[4];
                let c_alt = &candidates[best_non_m5];
                let m5_wins_parsimony = if c_m5.aicc.is_finite() && c_alt.aicc.is_finite() {
                    c_m5.aicc < c_alt.aicc
                } else {
                    c_m5.bic < c_alt.bic
                };
                if m5_wins_parsimony {
                    4
                } else {
                    best_non_m5
                }
            } else {
                raw_best
            }
        }
        ModelSelectionCriterion::Aicc => candidates
            .iter()
            .enumerate()
            .filter(|(_, c)| !c.eliminated && c.aicc.is_finite())
            .min_by(|a, b| a.1.aicc.total_cmp(&b.1.aicc))
            .map(|(i, _)| i)
            .unwrap_or(0),
        ModelSelectionCriterion::Bic => candidates
            .iter()
            .enumerate()
            .filter(|(_, c)| !c.eliminated)
            .min_by(|a, b| a.1.bic.total_cmp(&b.1.bic))
            .map(|(i, _)| i)
            .unwrap_or(0),
        ModelSelectionCriterion::PalmerFTest => {
            // Mandel, Akke & Palmer (1995) alpha = 0.05 nested chi^2 / F-test hierarchy (`Delta chi^2 > 3.841` for df=1, `> 5.991` for df=2)
            let c1 = &candidates[0];
            let mut best_i = 0usize;
            // Check 2-parameter models M2 (idx 1) and M3 (idx 2) against M1
            let mut best_2p: Option<usize> = None;
            for idx in [1usize, 2usize] {
                if !candidates[idx].eliminated
                    && (c1.chi2 - candidates[idx].chi2) > 3.841
                    && best_2p.is_none_or(|b| candidates[idx].chi2 < candidates[b].chi2)
                {
                    best_2p = Some(idx);
                }
            }
            if let Some(i2) = best_2p {
                best_i = i2;
                // Check 3-parameter M4 (idx 3) and M5 (idx 4) against best 2-parameter model
                for idx in [3usize, 4usize] {
                    if !candidates[idx].eliminated
                        && (candidates[i2].chi2 - candidates[idx].chi2) > 3.841
                        && candidates[idx].chi2 < candidates[best_i].chi2
                    {
                        best_i = idx;
                    }
                }
            } else {
                // Direct 2-dof jump from M1 to M4 or M5 (`Delta chi^2 > 5.991`)
                for idx in [3usize, 4usize] {
                    if !candidates[idx].eliminated
                        && (c1.chi2 - candidates[idx].chi2) > 5.991
                        && (best_i == 0 || candidates[idx].chi2 < candidates[best_i].chi2)
                    {
                        best_i = idx;
                    }
                }
            }
            best_i
        }
    };

    let mut sel = candidates[chosen_idx].clone();

    // Optional parametric Monte Carlo uncertainty refinement on the selected model
    if cfg.mc_replicates > 0 {
        let mut rng =
            Rng::new(cfg.seed ^ ((res.seq_id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)));
        let q_seed: Vec<f64> = match sel.model {
            ModelFreeId::M1 => vec![sel.params.s2],
            ModelFreeId::M2 => vec![sel.params.s2, sel.params.tau_e_sec.max(1.0e-12).ln()],
            ModelFreeId::M3 => vec![sel.params.s2, sel.params.rex_ref_s_inv],
            ModelFreeId::M4 => vec![
                sel.params.s2,
                sel.params.tau_e_sec.max(1.0e-12).ln(),
                sel.params.rex_ref_s_inv,
            ],
            ModelFreeId::M5 => vec![
                sel.params.sf2,
                sel.params.ss2,
                sel.params.tau_e_sec.max(100.0e-12).ln(),
            ],
        };

        let pred_fields: Vec<(f64, f64, f64)> = res
            .fields
            .iter()
            .map(|f| {
                predict_r1_r2_noe(
                    &rot_comps,
                    &sel.params,
                    f.field_mhz,
                    ref_field_mhz,
                    &cfg.physics,
                )
            })
            .collect();

        let mut syn_fields = res.fields.clone();
        let mut s2_mc = Vec::with_capacity(cfg.mc_replicates);
        let mut sf2_mc = Vec::with_capacity(cfg.mc_replicates);
        let mut ss2_mc = Vec::with_capacity(cfg.mc_replicates);
        let mut te_mc = Vec::with_capacity(cfg.mc_replicates);
        let mut rex_mc = Vec::with_capacity(cfg.mc_replicates);

        for _ in 0..cfg.mc_replicates {
            for (sf, &(r1_0, r2_0, noe_0)) in syn_fields.iter_mut().zip(pred_fields.iter()) {
                sf.r1 = (r1_0 + rng.normal(0.0, sf.sigma_r1)).max(1.0e-4);
                sf.r2 = (r2_0 + rng.normal(0.0, sf.sigma_r2)).max(1.0e-4);
                if sf.hetnoe.is_some() {
                    sf.hetnoe = Some(noe_0 + rng.normal(0.0, sf.sigma_hetnoe.unwrap_or(0.02)));
                }
            }
            let (qm, _) = polish_local_lm(
                sel.model,
                &q_seed,
                &syn_fields,
                &rot_comps,
                tau_m_eff,
                ref_field_mhz,
                &cfg.physics,
                25,
            );
            let pm = decode_internal_vec(sel.model, &qm, tau_m_eff);
            s2_mc.push(pm.s2);
            sf2_mc.push(pm.sf2);
            ss2_mc.push(pm.ss2);
            te_mc.push(pm.tau_e_sec);
            rex_mc.push(pm.rex_ref_s_inv);
        }
        let sd = |xs: &[f64]| -> f64 {
            if xs.len() <= 1 {
                return 0.0;
            }
            let mu = xs.iter().sum::<f64>() / (xs.len() as f64);
            (xs.iter().map(|&x| (x - mu) * (x - mu)).sum::<f64>() / ((xs.len() - 1) as f64)).sqrt()
        };
        sel.sigma_s2 = sd(&s2_mc);
        sel.sigma_sf2 = sd(&sf2_mc);
        sel.sigma_ss2 = sd(&ss2_mc);
        sel.sigma_tau_e_sec = sd(&te_mc);
        sel.sigma_rex_ref = sd(&rex_mc);
    }

    let dof = m_obs.saturating_sub(sel.model.num_params());
    ResidueModelFreeResult {
        seq_id: res.seq_id,
        res_name: res.res_name.clone(),
        selected_model: sel.model,
        params: sel.params,
        sigma_s2: sel.sigma_s2,
        sigma_sf2: sel.sigma_sf2,
        sigma_ss2: sel.sigma_ss2,
        sigma_tau_e_sec: sel.sigma_tau_e_sec,
        sigma_rex_ref: sel.sigma_rex_ref,
        chi2: sel.chi2,
        dof,
        aic: sel.aic,
        bic: sel.bic,
        candidate_fits: candidates,
    }
}

fn fit_all_residues_parallel(
    residues: &[MultiFieldResidueInput],
    tensor: &DiffusionTensor,
    ref_field_mhz: f64,
    cfg: &ModelFreeConfig,
) -> Vec<ResidueModelFreeResult> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .max(1);
        let chunk_sz = residues.len().div_ceil(n_threads).max(1);
        let mut out: Vec<Option<ResidueModelFreeResult>> = vec![None; residues.len()];
        std::thread::scope(|s| {
            for (cin, cout) in residues.chunks(chunk_sz).zip(out.chunks_mut(chunk_sz)) {
                s.spawn(move || {
                    for (r, dst) in cin.iter().zip(cout.iter_mut()) {
                        *dst = Some(fit_and_select_residue_modelfree(
                            r,
                            tensor,
                            ref_field_mhz,
                            cfg,
                        ));
                    }
                });
            }
        });
        out.into_iter().flatten().collect()
    }
    #[cfg(target_arch = "wasm32")]
    {
        residues
            .iter()
            .map(|r| fit_and_select_residue_modelfree(r, tensor, ref_field_mhz, cfg))
            .collect()
    }
}

// ============================================================================
// Global Rotational Diffusion Tensor Optimization (Alternating Profiled LM)
// ============================================================================

fn encode_local_seed(model: ModelFreeId, p: &LocalMfParams) -> Vec<f64> {
    match model {
        ModelFreeId::M1 => vec![p.s2],
        ModelFreeId::M2 => vec![p.s2, p.tau_e_sec.max(1.0e-12).ln()],
        ModelFreeId::M3 => vec![p.s2, p.rex_ref_s_inv],
        ModelFreeId::M4 => vec![p.s2, p.tau_e_sec.max(1.0e-12).ln(), p.rex_ref_s_inv],
        ModelFreeId::M5 => vec![p.sf2, p.ss2, p.tau_e_sec.max(100.0e-12).ln()],
    }
}

/// Evaluate global profiled residual vector across well-behaved residues (`chi2 < 25`), profiling out
/// each residue's fast local `S^2, tau_e, R_ex` parameters via 6 warm-started LM iterations so global
/// diffusion parameters `(tau_m / D_par, D_perp / D_x, D_y, D_z)` decouple cleanly from `S^2`!
fn global_profiled_chi2(
    tensor: &DiffusionTensor,
    residues: &[MultiFieldResidueInput],
    cur_local: &[ResidueModelFreeResult],
    active_mask: &[bool],
    ref_field_mhz: f64,
    phys: &RelaxPhysicsConfig,
) -> f64 {
    let tau_m_eff = tensor.effective_tau_m_sec();
    let mut total_c2 = 0.0_f64;
    for ((res, loc), &on) in residues
        .iter()
        .zip(cur_local.iter())
        .zip(active_mask.iter())
    {
        if !on {
            continue;
        }
        let comps = tensor.rotational_components(res.nh_unit_vec);
        let q0 = encode_local_seed(loc.selected_model, &loc.params);
        let (_, c2) = polish_local_lm(
            loc.selected_model,
            &q0,
            &res.fields,
            &comps,
            tau_m_eff,
            ref_field_mhz,
            phys,
            6,
        );
        total_c2 += c2;
    }
    total_c2
}

fn optimize_global_tensor_step(
    current: &DiffusionTensor,
    residues: &[MultiFieldResidueInput],
    cur_local: &[ResidueModelFreeResult],
    ref_field_mhz: f64,
    phys: &RelaxPhysicsConfig,
) -> DiffusionTensor {
    // Include all converged residues (excluding extreme unmodeled outliers with chi2 > 35)
    let mut active_mask: Vec<bool> = cur_local.iter().map(|r| r.chi2 < 35.0).collect();
    if active_mask.iter().filter(|&&b| b).count() < 5 {
        active_mask.fill(true);
    }

    match *current {
        DiffusionTensor::Isotropic { tau_m_sec } => {
            // Safeguarded golden-section / parabolic minimization on `ln(tau_m)` in `[0.5*tm, 2.0*tm]`
            let mut lo = (0.65 * tau_m_sec).max(0.2e-9).ln();
            let mut hi = (1.50 * tau_m_sec).min(200.0e-9).ln();
            let inv_phi = 0.618_033_988_749_894_9_f64;
            let mut x1 = hi - inv_phi * (hi - lo);
            let mut x2 = lo + inv_phi * (hi - lo);
            let eval_tm = |lt: f64| -> f64 {
                global_profiled_chi2(
                    &DiffusionTensor::Isotropic {
                        tau_m_sec: lt.exp(),
                    },
                    residues,
                    cur_local,
                    &active_mask,
                    ref_field_mhz,
                    phys,
                )
            };
            let mut f1 = eval_tm(x1);
            let mut f2 = eval_tm(x2);
            for _ in 0..24 {
                if (hi - lo) < 1.0e-6 {
                    break;
                }
                if f1 < f2 {
                    hi = x2;
                    x2 = x1;
                    f2 = f1;
                    x1 = hi - inv_phi * (hi - lo);
                    f1 = eval_tm(x1);
                } else {
                    lo = x1;
                    x1 = x2;
                    f1 = f2;
                    x2 = lo + inv_phi * (hi - lo);
                    f2 = eval_tm(x2);
                }
            }
            let best_lt = 0.5 * (lo + hi);
            DiffusionTensor::Isotropic {
                tau_m_sec: best_lt.exp(),
            }
        }
        DiffusionTensor::AxialSymmetric {
            d_perp_s_inv,
            d_par_s_inv,
            theta_rad,
            phi_rad,
        } => {
            let mut p = [d_perp_s_inv.ln(), d_par_s_inv.ln(), theta_rad, phi_rad];
            let decode_ax = |v: &[f64; 4]| -> DiffusionTensor {
                let (st, ct) = v[2].sin_cos();
                let (sp, cp) = v[3].sin_cos();
                let mut ax = [st * cp, st * sp, ct];
                if ax[2] < 0.0 {
                    ax = [-ax[0], -ax[1], -ax[2]];
                }
                DiffusionTensor::AxialSymmetric {
                    d_perp_s_inv: v[0].exp(),
                    d_par_s_inv: v[1].exp(),
                    theta_rad: ax[2].clamp(-1.0, 1.0).acos(),
                    phi_rad: ax[1].atan2(ax[0]),
                }
            };
            let mut f_cur = global_profiled_chi2(
                &decode_ax(&p),
                residues,
                cur_local,
                &active_mask,
                ref_field_mhz,
                phys,
            );
            let mut step_scale = [0.04_f64, 0.04, 0.05, 0.05];
            // Coordinate-pattern + Nelder/Newton directional polish in 4D `(ln D_perp, ln D_par, theta, phi)`
            for _pass in 0..14 {
                let mut improved = false;
                // Compute central finite-difference gradient & diagonal Hessian in 4D for quasi-Newton step
                let mut grad = [0.0_f64; 4];
                let mut h_diag = [0.0_f64; 4];
                for d in 0..4 {
                    let h = step_scale[d].max(2.0e-4) * 0.25;
                    let mut pp = p;
                    let mut pm = p;
                    pp[d] += h;
                    pm[d] -= h;
                    let fp = global_profiled_chi2(
                        &decode_ax(&pp),
                        residues,
                        cur_local,
                        &active_mask,
                        ref_field_mhz,
                        phys,
                    );
                    let fm = global_profiled_chi2(
                        &decode_ax(&pm),
                        residues,
                        cur_local,
                        &active_mask,
                        ref_field_mhz,
                        phys,
                    );
                    grad[d] = (fp - fm) / (2.0 * h);
                    h_diag[d] = ((fp - 2.0 * f_cur + fm) / (h * h)).abs().max(1.0);
                }
                for &alpha in &[1.0_f64, 0.5, 0.2, 0.05] {
                    let mut cand = p;
                    for d in 0..4 {
                        cand[d] -= alpha * (grad[d] / h_diag[d]).clamp(-0.12, 0.12);
                    }
                    let fc = global_profiled_chi2(
                        &decode_ax(&cand),
                        residues,
                        cur_local,
                        &active_mask,
                        ref_field_mhz,
                        phys,
                    );
                    if fc < f_cur - 1.0e-9 {
                        p = cand;
                        f_cur = fc;
                        improved = true;
                        break;
                    }
                }
                for sc in &mut step_scale {
                    *sc *= if improved { 0.85 } else { 0.45 };
                }
                if step_scale[0] < 1.0e-4 && !improved {
                    break;
                }
            }
            decode_ax(&p)
        }
        DiffusionTensor::FullyAnisotropic {
            dx_s_inv,
            dy_s_inv,
            dz_s_inv,
            alpha_rad,
            beta_rad,
            gamma_rad,
        } => {
            let mut p = [
                dx_s_inv.ln(),
                dy_s_inv.ln(),
                dz_s_inv.ln(),
                alpha_rad,
                beta_rad,
                gamma_rad,
            ];
            let decode_aniso = |v: &[f64; 6]| -> DiffusionTensor {
                DiffusionTensor::FullyAnisotropic {
                    dx_s_inv: v[0].exp(),
                    dy_s_inv: v[1].exp(),
                    dz_s_inv: v[2].exp(),
                    alpha_rad: v[3],
                    beta_rad: v[4].clamp(0.0, PI),
                    gamma_rad: v[5],
                }
            };
            let mut f_cur = global_profiled_chi2(
                &decode_aniso(&p),
                residues,
                cur_local,
                &active_mask,
                ref_field_mhz,
                phys,
            );
            let mut step_scale = [0.03_f64, 0.03, 0.03, 0.04, 0.04, 0.04];
            for _pass in 0..14 {
                let mut improved = false;
                let mut grad = [0.0_f64; 6];
                let mut h_diag = [0.0_f64; 6];
                for d in 0..6 {
                    let h = step_scale[d].max(2.0e-4) * 0.25;
                    let mut pp = p;
                    let mut pm = p;
                    pp[d] += h;
                    pm[d] -= h;
                    let fp = global_profiled_chi2(
                        &decode_aniso(&pp),
                        residues,
                        cur_local,
                        &active_mask,
                        ref_field_mhz,
                        phys,
                    );
                    let fm = global_profiled_chi2(
                        &decode_aniso(&pm),
                        residues,
                        cur_local,
                        &active_mask,
                        ref_field_mhz,
                        phys,
                    );
                    grad[d] = (fp - fm) / (2.0 * h);
                    h_diag[d] = ((fp - 2.0 * f_cur + fm) / (h * h)).abs().max(1.0);
                }
                for &alpha in &[1.0_f64, 0.5, 0.2, 0.05] {
                    let mut cand = p;
                    for d in 0..6 {
                        cand[d] -= alpha * (grad[d] / h_diag[d]).clamp(-0.10, 0.10);
                    }
                    let fc = global_profiled_chi2(
                        &decode_aniso(&cand),
                        residues,
                        cur_local,
                        &active_mask,
                        ref_field_mhz,
                        phys,
                    );
                    if fc < f_cur - 1.0e-9 {
                        p = cand;
                        f_cur = fc;
                        improved = true;
                        break;
                    }
                }
                for sc in &mut step_scale {
                    *sc *= if improved { 0.85 } else { 0.45 };
                }
                if step_scale[0] < 1.0e-4 && !improved {
                    break;
                }
            }
            decode_aniso(&p)
        }
    }
}

fn fit_with_fixed_diffusion_symmetry(
    kind: DiffusionTensorKind,
    residues: &[MultiFieldResidueInput],
    ref_field_mhz: f64,
    cfg: &ModelFreeConfig,
) -> Result<ModelFreeDatasetResult, String> {
    if residues.is_empty() {
        return Err("fit_modelfree_dataset requires >= 1 residue".to_string());
    }
    let primary_rates: Vec<ResidueRelaxRates> = residues
        .iter()
        .filter_map(|r| r.fields.first().cloned())
        .collect();
    let nh_vecs: Vec<(i32, [f64; 3])> = residues
        .iter()
        .filter_map(|r| r.nh_unit_vec.map(|u| (r.seq_id, u)))
        .collect();

    let trim_cfg = TrimmedTauMConfig {
        physics: cfg.physics,
        trim_sd: if kind == DiffusionTensorKind::Isotropic {
            1.5
        } else {
            3.0
        },
        ..TrimmedTauMConfig::default()
    };
    let iso_init = estimate_trimmed_tau_m(&primary_rates, &trim_cfg)?;

    let mut tensor = match kind {
        DiffusionTensorKind::Isotropic | DiffusionTensorKind::Auto => DiffusionTensor::Isotropic {
            tau_m_sec: iso_init.tau_m_sec,
        },
        DiffusionTensorKind::AxialSymmetric => {
            if nh_vecs.len() < 5 {
                return Err(format!(
                    "AxialSymmetric diffusion requires >= 5 residues with PDB N-H unit vectors, got {}",
                    nh_vecs.len()
                ));
            }
            let ax =
                estimate_initial_axial_diffusion_from_r2_r1(&primary_rates, &nh_vecs, &trim_cfg)?;
            DiffusionTensor::AxialSymmetric {
                d_perp_s_inv: ax.d_perp_s_inv,
                d_par_s_inv: ax.d_par_s_inv,
                theta_rad: ax.theta_rad,
                phi_rad: ax.phi_rad,
            }
        }
        DiffusionTensorKind::FullyAnisotropic => {
            if nh_vecs.len() < 7 {
                return Err(format!(
                    "FullyAnisotropic diffusion requires >= 7 residues with PDB N-H unit vectors, got {}",
                    nh_vecs.len()
                ));
            }
            let ax =
                estimate_initial_axial_diffusion_from_r2_r1(&primary_rates, &nh_vecs, &trim_cfg)?;
            // Seed asymmetric rotor from converged axial symmetry axis `(alpha = phi, beta = theta)` with slight rhombic split.
            // `euler_zyz_matrix(alpha, beta, 0)` has third row `[sin(beta)cos(alpha), sin(beta)sin(alpha), cos(beta)]`,
            // aligning principal Z (`dz`) identically with `axis = [sin(theta)cos(phi), sin(theta)sin(phi), cos(theta)]`
            // for both prolate (`d_par >= d_perp`) and oblate (`d_par < d_perp`) seeds!
            DiffusionTensor::FullyAnisotropic {
                dx_s_inv: 0.96 * ax.d_perp_s_inv,
                dy_s_inv: 1.04 * ax.d_perp_s_inv,
                dz_s_inv: ax.d_par_s_inv,
                alpha_rad: ax.phi_rad,
                beta_rad: ax.theta_rad,
                gamma_rad: 0.25,
            }
        }
    };

    // Alternating optimization without MC during inner iterations (`mc_replicates = 0`), then run MC once at convergence
    let inner_cfg = ModelFreeConfig {
        mc_replicates: 0,
        ..cfg.clone()
    };

    let mut local_res = fit_all_residues_parallel(residues, &tensor, ref_field_mhz, &inner_cfg);
    let mut used_iters = 1usize;

    for outer in 0..cfg.max_outer_iters {
        let prev_tm = tensor.effective_tau_m_sec();
        let prev_models: Vec<ModelFreeId> = local_res.iter().map(|r| r.selected_model).collect();

        tensor =
            optimize_global_tensor_step(&tensor, residues, &local_res, ref_field_mhz, &cfg.physics);
        local_res = fit_all_residues_parallel(residues, &tensor, ref_field_mhz, &inner_cfg);
        used_iters = outer + 1;

        let new_tm = tensor.effective_tau_m_sec();
        let same_models = local_res
            .iter()
            .zip(prev_models.iter())
            .all(|(a, &b)| a.selected_model == b);
        if ((new_tm - prev_tm) / prev_tm).abs() < 1.0e-5 && same_models && outer >= 1 {
            break;
        }
    }

    if cfg.mc_replicates > 0 {
        local_res = fit_all_residues_parallel(residues, &tensor, ref_field_mhz, cfg);
    }

    let total_chi2: f64 = local_res.iter().map(|r| r.chi2).sum();
    let k_local: usize = local_res
        .iter()
        .map(|r| r.selected_model.num_params())
        .sum();
    let k_total = (tensor.num_global_params() + k_local) as f64;
    let n_total = (residues
        .iter()
        .map(MultiFieldResidueInput::total_observables)
        .sum::<usize>())
    .max(1) as f64;
    let global_aic = total_chi2 + 2.0 * k_total;
    let global_bic = total_chi2 + k_total * n_total.ln();

    Ok(ModelFreeDatasetResult {
        diffusion_tensor: tensor,
        ref_field_mhz,
        total_chi2,
        global_aic,
        global_bic,
        outer_iterations: used_iters,
        residues: local_res,
    })
}

/// Run complete single- or multi-field Lipari–Szabo / Clore / Mandel–Akke–Palmer Model-Free (`M1`–`M5`)
/// and rotational diffusion (`Isotropic`, `AxialSymmetric`, `FullyAnisotropic`, or `Auto`) analysis.
pub fn fit_modelfree_dataset(
    residues: &[MultiFieldResidueInput],
    cfg: &ModelFreeConfig,
) -> Result<ModelFreeDatasetResult, String> {
    if residues.is_empty() {
        return Err("fit_modelfree_dataset requires >= 1 residue".to_string());
    }
    let ref_field_mhz = cfg.ref_field_mhz.unwrap_or_else(|| {
        residues
            .iter()
            .flat_map(|r| r.fields.iter().map(|f| f.field_mhz))
            .fold(f64::INFINITY, f64::min)
    });
    if !ref_field_mhz.is_finite() || ref_field_mhz <= 0.0 {
        return Err("invalid or empty magnetic field list in MultiFieldResidueInput".to_string());
    }

    if cfg.diffusion_kind != DiffusionTensorKind::Auto {
        return fit_with_fixed_diffusion_symmetry(cfg.diffusion_kind, residues, ref_field_mhz, cfg);
    }

    let n_with_vec = residues.iter().filter(|r| r.nh_unit_vec.is_some()).count();
    let mut candidates = vec![fit_with_fixed_diffusion_symmetry(
        DiffusionTensorKind::Isotropic,
        residues,
        ref_field_mhz,
        cfg,
    )?];
    if n_with_vec >= 5 {
        if let Ok(ax) = fit_with_fixed_diffusion_symmetry(
            DiffusionTensorKind::AxialSymmetric,
            residues,
            ref_field_mhz,
            cfg,
        ) {
            candidates.push(ax);
        }
    }
    if n_with_vec >= 7 {
        if let Ok(aniso) = fit_with_fixed_diffusion_symmetry(
            DiffusionTensorKind::FullyAnisotropic,
            residues,
            ref_field_mhz,
            cfg,
        ) {
            candidates.push(aniso);
        }
    }

    let mut cand_iter = candidates.into_iter();
    let first = cand_iter
        .next()
        .ok_or_else(|| "no rotational diffusion tensor candidates evaluated".to_string())?;
    let best = match cfg.selection_criterion {
        ModelSelectionCriterion::Bic => cand_iter.fold(first, |acc, c| {
            if c.global_bic < acc.global_bic {
                c
            } else {
                acc
            }
        }),
        _ => cand_iter.fold(first, |acc, c| {
            if c.global_aic < acc.global_aic {
                c
            } else {
                acc
            }
        }),
    };
    Ok(best)
}

/// Assemble `Vec<MultiFieldResidueInput>` from flat single/multi-field `ResidueRelaxRates` tables
/// and optional `(seq_id, unit_nh_vec)` structural coordinates.
pub fn group_multifield_rates(
    rate_tables: &[Vec<ResidueRelaxRates>],
    nh_vectors: Option<&[(i32, [f64; 3])]>,
) -> Vec<MultiFieldResidueInput> {
    let vec_map: BTreeMap<i32, [f64; 3]> = nh_vectors
        .map(|vs| vs.iter().copied().collect())
        .unwrap_or_default();
    let mut grouped: BTreeMap<i32, (String, Vec<ResidueRelaxRates>)> = BTreeMap::new();
    for tbl in rate_tables {
        for r in tbl {
            let entry = grouped
                .entry(r.seq_id)
                .or_insert_with(|| (r.res_name.clone(), Vec::new()));
            entry.1.push(r.clone());
        }
    }
    grouped
        .into_iter()
        .map(|(seq_id, (res_name, mut fields))| {
            fields.sort_by(|a, b| a.field_mhz.total_cmp(&b.field_mhz));
            MultiFieldResidueInput {
                seq_id,
                res_name,
                nh_unit_vec: vec_map.get(&seq_id).copied(),
                fields,
            }
        })
        .collect()
}

/// Export a `ModelFreeDatasetResult` to clean tabular TSV format (`docs/specs/16-dynamics-and-csp.md` §2.6).
pub fn write_modelfree_tsv_str(res: &ModelFreeDatasetResult) -> String {
    let d_ratio = match res.diffusion_tensor {
        DiffusionTensor::Isotropic { .. } => 1.0,
        DiffusionTensor::AxialSymmetric {
            d_perp_s_inv,
            d_par_s_inv,
            ..
        } => d_par_s_inv / d_perp_s_inv.max(1.0),
        DiffusionTensor::FullyAnisotropic {
            dx_s_inv,
            dy_s_inv,
            dz_s_inv,
            ..
        } => 2.0 * dz_s_inv / (dx_s_inv + dy_s_inv).max(1.0),
    };
    let mut out = String::new();
    out.push_str(&format!(
        "# nmrassign_modelfree ref_field_mhz={:.3} effective_tau_m_ns={:.4} d_par_over_d_perp={:.4} total_chi2={:.4} global_aic={:.4}\n",
        res.ref_field_mhz,
        res.diffusion_tensor.effective_tau_m_sec() * 1.0e9,
        d_ratio,
        res.total_chi2,
        res.global_aic
    ));
    out.push_str("seq_id\tres_name\tmodel\ts2\tsigma_s2\tsf2\tss2\ttau_e_ps\tsigma_tau_e_ps\trex_ref\tsigma_rex_ref\tchi2\taic\n");
    for r in &res.residues {
        out.push_str(&format!(
            "{}\t{}\t{}\t{:.5}\t{:.5}\t{:.5}\t{:.5}\t{:.3}\t{:.3}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\n",
            r.seq_id,
            r.res_name,
            r.selected_model.as_str(),
            r.params.s2,
            r.sigma_s2,
            r.params.sf2,
            r.params.ss2,
            r.params.tau_e_sec * 1.0e12,
            r.sigma_tau_e_sec * 1.0e12,
            r.params.rex_ref_s_inv,
            r.sigma_rex_ref,
            r.chi2,
            r.aic
        ));
    }
    out
}
