//! Clean-room Kabsch–Sander (1983) electrostatic hydrogen-bond 8-state secondary-structure
//! classifier and deterministic `8 -> 4`-state NMR conformational reduction (`Helix`, `Sheet`,
//! `Ppii`, `Coil`).
//!
//! Provenance: Clean-room original implementation derived strictly from peer-reviewed structural
//! biology literature (`0` external copyleft, academic-licence, or proprietary software consulted;
//! `0` external product names in identifiers or prose):
//! 1. Kabsch, W. & Sander, C. (1983). "Dictionary of protein secondary structure: Pattern
//!    recognition of hydrogen-bonded and geometrical features." *Biopolymers* **22**(12),
//!    2577–2637 — ideal trans-peptide amide proton reconstruction (`1.00 Å` along `C'_{j-1}-O_{j-1}`),
//!    partial-charge electrostatic carbonyl-amide interaction energy ($q_1 = 0.42$, $q_2 = 0.20$,
//!    $332\text{ kcal mol}^{-1}$, $E_{\text{HB}} < -0.5\text{ kcal mol}^{-1}$, top-2 H-bonds per
//!    donor and acceptor), $n$-turn helices ($n \in \{3, 4, 5\}$), parallel and antiparallel
//!    $\beta$-bridges / $\beta$-bulge ladders, virtual $\text{C}^\alpha$ bend angle $\kappa_i > 70^\circ$,
//!    and 8-state hierarchy (`H > E > B > G > I > T > S > C`, where multi-bridge $\beta$-sheet
//!    ladder `E` absorbs an auxiliary single bridge `B` on the same residue, and any orphaned
//!    $3_{10}$-helix `G` fragment of `< 3` residues or $\pi$-helix `I` fragment of `< 5` residues
//!    left after `H/E/B` boundary override is demoted back to `T > S > C`).
//! 2. Adzhubei, A. A., Sternberg, M. J. E. & Makarov, A. A. (2013). "Polyproline-II helix in
//!    proteins: Structure and function." *J. Mol. Biol.* **425**(12), 2100–2132 — left-handed
//!    extended polyproline-II (`PPII`) $(\phi, \psi)$ conformational basin ($\phi \in [-105^\circ, -45^\circ]$,
//!    $\psi \in [+115^\circ, +175^\circ]$) in non-hydrogen-bonded polypeptide segments.

use serde::{Deserialize, Serialize};

use crate::idp_propensity::ConformationalState;
use crate::noesy::{calc_dihedral_deg, v_add, v_dist, v_dot, v_scale, v_sub, v_unit, PdbChain};
use crate::structure::det_acos;

/// Carbonyl partial charge magnitude $q_1 = 0.42\,e$ (Kabsch & Sander *Biopolymers* 1983, p. 2580).
pub const KABSCH_SANDER_Q1: f64 = 0.42;

/// Amide partial charge magnitude $q_2 = 0.20\,e$ (Kabsch & Sander *Biopolymers* 1983, p. 2580).
pub const KABSCH_SANDER_Q2: f64 = 0.20;

/// Electrostatic unit conversion factor ($332\text{ kcal mol}^{-1}\text{ \AA }e^{-2}$).
pub const KABSCH_SANDER_COULOMB_KCAL_A: f64 = 332.0;

/// Kabsch–Sander electrostatic hydrogen-bond threshold ($E_{\text{HB}} < -0.5\text{ kcal mol}^{-1}$).
pub const KABSCH_SANDER_HBOND_CUTOFF_KCAL: f64 = -0.5;

/// Short-range interatomic clash floor (`0.5 Å`) preventing singular electrostatic energies.
pub const KABSCH_SANDER_MIN_DIST_A: f64 = 0.5;

/// Maximum intact covalent peptide-bond distance $d(\text{C}'_{j-1}, \text{N}_j) \le 2.5\text{ \AA}$.
pub const MAX_PEPTIDE_BOND_C_N_A: f64 = 2.5;

/// Ideal trans-peptide amide $\text{N–H}$ bond length (`1.00 Å`, Kabsch & Sander 1983, p. 2580).
pub const IDEAL_AMIDE_N_H_BOND_A: f64 = 1.00;

/// Kabsch & Sander (1983) 8-state secondary-structure taxonomy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KabschSanderState8 {
    /// `H`: 4-helix ($\alpha$-helix, consecutive overlapping $i \to i+4$ turns).
    H,
    /// `B`: Residue in an isolated $\beta$-bridge (single-pair parallel or antiparallel bridge).
    B,
    /// `E`: Extended $\beta$-strand participating in a $\beta$-ladder of $\ge 2$ bridges (including bulges).
    E,
    /// `G`: 3-helix ($3_{10}$-helix, consecutive overlapping $i \to i+3$ turns).
    G,
    /// `I`: 5-helix ($\pi$-helix, consecutive overlapping $i \to i+5$ turns).
    I,
    /// `T`: Hydrogen-bonded $n$-turn ($n \in \{3, 4, 5\}$) not part of a helix or sheet.
    T,
    /// `S`: High-curvature backbone bend (virtual $\text{C}^\alpha$ angle $\kappa_i > 70^\circ$).
    S,
    /// `C`: Irregular loop / random coil.
    C,
}

impl KabschSanderState8 {
    /// Single-character Kabsch–Sander state symbol (`H, B, E, G, I, T, S, C`).
    #[inline]
    pub const fn code(self) -> char {
        match self {
            Self::H => 'H',
            Self::B => 'B',
            Self::E => 'E',
            Self::G => 'G',
            Self::I => 'I',
            Self::T => 'T',
            Self::S => 'S',
            Self::C => 'C',
        }
    }

    /// Static 1-character string slice (`"H"`, `"B"`, `"E"`, `"G"`, `"I"`, `"T"`, `"S"`, `"C"`).
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::H => "H",
            Self::B => "B",
            Self::E => "E",
            Self::G => "G",
            Self::I => "I",
            Self::T => "T",
            Self::S => "S",
            Self::C => "C",
        }
    }

    /// Human-readable secondary-structure label.
    #[inline]
    pub const fn label(self) -> &'static str {
        match self {
            Self::H => "Alpha-Helix (4-helix)",
            Self::B => "Isolated Beta-Bridge",
            Self::E => "Extended Beta-Strand",
            Self::G => "3_10-Helix (3-helix)",
            Self::I => "Pi-Helix (5-helix)",
            Self::T => "Hydrogen-Bonded Turn",
            Self::S => "Backbone Bend",
            Self::C => "Coil",
        }
    }
}

/// Deterministic 4-state NMR secondary-structure alphabet (`Helix`, `Sheet`, `Ppii`, `Coil`),
/// matching `nmr_core::idp_propensity::ConformationalState`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SecondaryStructure4 {
    /// Helical state (`{H, G, I}`).
    Helix,
    /// Extended $\beta$-sheet / $\beta$-bridge state (`{E, B}`).
    Sheet,
    /// Non-hydrogen-bonded left-handed polyproline-II extended helix (`P`, Adzhubei et al. 2013).
    Ppii,
    /// Remaining irregular / turn / bend coil states (`C`).
    Coil,
}

impl SecondaryStructure4 {
    /// Single-character 4-state code (`H`, `E`, `P`, `C`).
    #[inline]
    pub const fn code(self) -> char {
        match self {
            Self::Helix => 'H',
            Self::Sheet => 'E',
            Self::Ppii => 'P',
            Self::Coil => 'C',
        }
    }

    /// Static 1-character string slice (`"H"`, `"E"`, `"P"`, `"C"`).
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Helix => "H",
            Self::Sheet => "E",
            Self::Ppii => "P",
            Self::Coil => "C",
        }
    }

    /// Human-readable 4-state label.
    #[inline]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Helix => "Helix",
            Self::Sheet => "Sheet",
            Self::Ppii => "Polyproline-II",
            Self::Coil => "Coil",
        }
    }

    /// Convert to the corresponding `idp_propensity::ConformationalState`.
    #[inline]
    pub const fn to_conformational_state(self) -> ConformationalState {
        match self {
            Self::Helix => ConformationalState::Alpha,
            Self::Sheet => ConformationalState::Beta,
            Self::Ppii => ConformationalState::Ppii,
            Self::Coil => ConformationalState::Coil,
        }
    }
}

impl From<ConformationalState> for SecondaryStructure4 {
    #[inline]
    fn from(state: ConformationalState) -> Self {
        match state {
            ConformationalState::Alpha => Self::Helix,
            ConformationalState::Beta => Self::Sheet,
            ConformationalState::Ppii => Self::Ppii,
            ConformationalState::Coil => Self::Coil,
        }
    }
}

/// Backbone coordinates for a single polymer residue supplied to the Kabsch–Sander classifier.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BackboneResidueCoords {
    pub seq_id: i32,
    pub res_name: String,
    pub chain_id: String,
    pub n: [f64; 3],
    pub ca: [f64; 3],
    pub c: [f64; 3],
    pub o: [f64; 3],
    /// Explicit amide `H`/`HN` atom coordinate when present in the input model (`None` if absent,
    /// in which case Kabsch–Sander `1.00 Å` trans-peptide reconstruction is applied).
    pub explicit_h: Option<[f64; 3]>,
}

impl BackboneResidueCoords {
    /// Convert a `PdbChain` into `Vec<BackboneResidueCoords>`, optionally ignoring pre-reconstructed
    /// `PdbResidue::h` when `use_ideal_ks_amide_h` is `true`.
    pub fn from_pdb_chain(chain: &PdbChain, use_ideal_ks_amide_h: bool) -> Vec<Self> {
        chain
            .residues
            .iter()
            .map(|r| Self {
                seq_id: r.seq_id,
                res_name: r.res_name.clone(),
                chain_id: r.chain_id.to_string(),
                n: r.n,
                ca: r.ca,
                c: r.c,
                o: r.o,
                explicit_h: if use_ideal_ks_amide_h { None } else { r.h },
            })
            .collect()
    }
}

/// Top electrostatic hydrogen-bond partner record (`E_HB < -0.5 kcal/mol`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct KabschSanderHBondPartner {
    /// Zero-based residue index of the partner residue.
    pub partner_index: usize,
    /// Partner `seq_id`.
    pub partner_seq_id: i32,
    /// Electrostatic energy $E_{\text{HB}}$ in $\text{kcal mol}^{-1}$ (negative).
    pub energy_kcal_mol: f64,
}

/// Per-residue Kabsch–Sander 8-state & 4-state secondary-structure assignment and geometry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResidueSecondaryStructure {
    pub seq_index: usize,
    pub seq_id: i32,
    pub res_name: String,
    pub chain_id: String,
    pub state8: KabschSanderState8,
    pub state4: SecondaryStructure4,
    pub phi_deg: Option<f64>,
    pub psi_deg: Option<f64>,
    pub kappa_deg: Option<f64>,
    /// Reconstructed or explicit amide `H` coordinate (`None` for Proline, N-terminus, or chain break).
    pub amide_h: Option<[f64; 3]>,
    /// Up to two strongest carbonyl $\text{C}'_i=\text{O}_i \to \text{H–N}_j$ H-bonds donated by carbonyl $i$.
    pub co_acceptors: Vec<KabschSanderHBondPartner>,
    /// Up to two strongest amide $\text{N}_i\text{–H}_i \leftarrow \text{O=C}'_k$ H-bonds accepted by amide $i$.
    pub nh_donors: Vec<KabschSanderHBondPartner>,
}

/// Complete chain secondary-structure classification report.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SecondaryStructureReport {
    pub residues: Vec<ResidueSecondaryStructure>,
    pub n_helix: usize,
    pub n_sheet: usize,
    pub n_ppii: usize,
    pub n_coil: usize,
}

impl SecondaryStructureReport {
    /// Compact 8-state string across all residues (e.g. `"CEEEEEETTEEEEE..."`).
    pub fn state8_string(&self) -> String {
        self.residues.iter().map(|r| r.state8.code()).collect()
    }

    /// Compact 4-state string across all residues (`H, E, P, C`).
    pub fn state4_string(&self) -> String {
        self.residues.iter().map(|r| r.state4.code()).collect()
    }
}

#[inline]
fn is_proline(res_name: &str) -> bool {
    res_name.trim().eq_ignore_ascii_case("PRO")
}

#[inline]
fn coords_are_finite(p: [f64; 3]) -> bool {
    p[0].is_finite() && p[1].is_finite() && p[2].is_finite()
}

/// Reconstruct the ideal trans-peptide amide proton coordinate
/// $\mathbf{r}_{\text{H}_j} = \mathbf{r}_{\text{N}_j} + 1.00\text{ \AA}\cdot\widehat{(\mathbf{r}_{\text{C}'_{j-1}} - \mathbf{r}_{\text{O}_{j-1}})}$
/// per Kabsch & Sander (*Biopolymers* **1983**, 22, p. 2580).
#[inline]
pub fn reconstruct_ideal_trans_amide_h(
    c_prev: [f64; 3],
    o_prev: [f64; 3],
    n_cur: [f64; 3],
) -> Option<[f64; 3]> {
    if v_dist(c_prev, n_cur) > MAX_PEPTIDE_BOND_C_N_A {
        return None;
    }
    let co_vec = v_sub(c_prev, o_prev);
    let co_len_sq = v_dot(co_vec, co_vec);
    if !co_len_sq.is_finite() || co_len_sq < 1.0e-12 {
        return None;
    }
    let dir = v_unit(co_vec);
    Some(v_add(n_cur, v_scale(dir, IDEAL_AMIDE_N_H_BOND_A)))
}

/// Evaluate the Kabsch–Sander electrostatic interaction energy ($\text{kcal mol}^{-1}$)
/// between carbonyl group $\text{C}'_i=\text{O}_i$ and amide group $\text{N}_j\text{–H}_j$:
/// $$E_{\text{HB}}(i \to j) = q_1 q_2 \left(\frac{1}{r_{\text{ON}}} + \frac{1}{r_{\text{CH}}} - \frac{1}{r_{\text{OH}}} - \frac{1}{r_{\text{CN}}}\right)\cdot 332$$
#[inline]
pub fn kabsch_sander_hbond_energy_kcal(
    c_i: [f64; 3],
    o_i: [f64; 3],
    n_j: [f64; 3],
    h_j: [f64; 3],
) -> f64 {
    let r_on = v_dist(o_i, n_j).max(KABSCH_SANDER_MIN_DIST_A);
    let r_ch = v_dist(c_i, h_j).max(KABSCH_SANDER_MIN_DIST_A);
    let r_oh = v_dist(o_i, h_j).max(KABSCH_SANDER_MIN_DIST_A);
    let r_cn = v_dist(c_i, n_j).max(KABSCH_SANDER_MIN_DIST_A);
    let scale = KABSCH_SANDER_Q1 * KABSCH_SANDER_Q2 * KABSCH_SANDER_COULOMB_KCAL_A;
    scale * (1.0 / r_on + 1.0 / r_ch - 1.0 / r_oh - 1.0 / r_cn)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BridgeKind {
    Parallel,
    Antiparallel,
}

#[derive(Clone, Copy, Debug)]
struct BetaBridge {
    kind: BridgeKind,
    i: usize,
    j: usize,
}

#[inline]
fn segment_is_intact(intact_prev: &[bool], start: usize, end: usize) -> bool {
    if start >= end {
        return true;
    }
    intact_prev[(start + 1)..=end].iter().all(|&ok| ok)
}

#[inline]
fn is_ppii_dihedral(phi_deg: f64, psi_deg: f64) -> bool {
    (-105.0..=-45.0).contains(&phi_deg) && (115.0..=175.0).contains(&psi_deg)
}

/// Classify 8-state Kabsch–Sander secondary structure and deterministic `8 -> 4`-state NMR
/// secondary structure directly from 3D backbone coordinates (never reading `HELIX` or `SHEET` records).
pub fn classify_kabsch_sander(
    coords: &[BackboneResidueCoords],
) -> Result<SecondaryStructureReport, String> {
    let l = coords.len();
    if l == 0 {
        return Err("cannot classify secondary structure on an empty residue list".to_string());
    }

    for (idx, r) in coords.iter().enumerate() {
        if !coords_are_finite(r.n)
            || !coords_are_finite(r.ca)
            || !coords_are_finite(r.c)
            || !coords_are_finite(r.o)
        {
            return Err(format!(
                "non-finite backbone (N, CA, C, O) coordinates at residue index {idx} (seq_id {})",
                r.seq_id
            ));
        }
        if let Some(h) = r.explicit_h {
            if !coords_are_finite(h) {
                return Err(format!(
                    "non-finite explicit amide H coordinate at residue index {idx} (seq_id {})",
                    r.seq_id
                ));
            }
        }
    }

    // 1. Intact peptide bonds (`d(C'_{j-1}, N_j) <= 2.5 A` on same chain) & amide proton coordinates
    let mut intact_prev = vec![false; l];
    let mut amide_h: Vec<Option<[f64; 3]>> = vec![None; l];
    for j in 0..l {
        if j > 0
            && coords[j - 1].chain_id == coords[j].chain_id
            && v_dist(coords[j - 1].c, coords[j].n) <= MAX_PEPTIDE_BOND_C_N_A
        {
            intact_prev[j] = true;
        }
        if is_proline(&coords[j].res_name) {
            amide_h[j] = None;
        } else if let Some(h_exp) = coords[j].explicit_h {
            amide_h[j] = Some(h_exp);
        } else if intact_prev[j] {
            amide_h[j] =
                reconstruct_ideal_trans_amide_h(coords[j - 1].c, coords[j - 1].o, coords[j].n);
        }
    }

    // 2. Backbone (phi, psi) torsions and C-alpha virtual bend angle kappa_i
    let mut phi_deg = vec![None; l];
    let mut psi_deg = vec![None; l];
    let mut kappa_deg = vec![None; l];
    for i in 0..l {
        if i > 0 && intact_prev[i] {
            phi_deg[i] = Some(calc_dihedral_deg(
                coords[i - 1].c,
                coords[i].n,
                coords[i].ca,
                coords[i].c,
            ));
        }
        if i + 1 < l && intact_prev[i + 1] {
            psi_deg[i] = Some(calc_dihedral_deg(
                coords[i].n,
                coords[i].ca,
                coords[i].c,
                coords[i + 1].n,
            ));
        }
        if i >= 2 && i + 2 < l && segment_is_intact(&intact_prev, i - 2, i + 2) {
            let v1 = v_sub(coords[i].ca, coords[i - 2].ca);
            let v2 = v_sub(coords[i + 2].ca, coords[i].ca);
            let n1 = v_dot(v1, v1).sqrt();
            let n2 = v_dot(v2, v2).sqrt();
            if n1 > 1.0e-9 && n2 > 1.0e-9 {
                let cos_k = (v_dot(v1, v2) / (n1 * n2)).clamp(-1.0, 1.0);
                kappa_deg[i] = Some(det_acos(cos_k).to_degrees());
            }
        }
    }

    // 3. Pairwise Kabsch-Sander H-bond energies (`|i - j| >= 2` on same chain, `E_HB < -0.5 kcal/mol`),
    //    retaining up to the 2 strongest H-bonds per C'=O donor `i` and per N-H acceptor `j`.
    let mut top_co: Vec<Vec<(usize, f64)>> = vec![Vec::new(); l];
    let mut top_nh: Vec<Vec<(usize, f64)>> = vec![Vec::new(); l];

    for i in 0..l {
        for j in 0..l {
            if coords[i].chain_id == coords[j].chain_id && i.abs_diff(j) < 2 {
                continue;
            }
            let Some(h_j) = amide_h[j] else {
                continue;
            };
            // Fast conservative CA-CA screen: beyond 9.5 A, Kabsch-Sander E_HB cannot reach -0.5 kcal/mol
            if v_dist(coords[i].ca, coords[j].ca) > 9.5 {
                continue;
            }
            let e = kabsch_sander_hbond_energy_kcal(coords[i].c, coords[i].o, coords[j].n, h_j);
            if e < KABSCH_SANDER_HBOND_CUTOFF_KCAL {
                top_co[i].push((j, e));
                top_nh[j].push((i, e));
            }
        }
    }

    for list in &mut top_co {
        list.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        list.truncate(2);
    }
    for list in &mut top_nh {
        list.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        list.truncate(2);
    }

    let has_hbond = |i: usize, j: usize| -> bool {
        top_co[i].iter().any(|&(jj, _)| jj == j) && top_nh[j].iter().any(|&(ii, _)| ii == i)
    };

    // Retained active H-bonds per residue (mutually in top-2 of both partners)
    let mut active_co: Vec<Vec<KabschSanderHBondPartner>> = vec![Vec::new(); l];
    let mut active_nh: Vec<Vec<KabschSanderHBondPartner>> = vec![Vec::new(); l];
    for i in 0..l {
        for &(j, e) in &top_co[i] {
            if top_nh[j].iter().any(|&(ii, _)| ii == i) {
                active_co[i].push(KabschSanderHBondPartner {
                    partner_index: j,
                    partner_seq_id: coords[j].seq_id,
                    energy_kcal_mol: e,
                });
            }
        }
        for &(k, e) in &top_nh[i] {
            if top_co[k].iter().any(|&(jj, _)| jj == i) {
                active_nh[i].push(KabschSanderHBondPartner {
                    partner_index: k,
                    partner_seq_id: coords[k].seq_id,
                    energy_kcal_mol: e,
                });
            }
        }
    }

    // 4. Elementary n-turns (n in {3, 4, 5}) and consecutive overlapping n-helices (G, H, I)
    let mut turn3 = vec![false; l];
    let mut turn4 = vec![false; l];
    let mut turn5 = vec![false; l];
    let mut is_turn = vec![false; l];

    for i in 0..l {
        if i + 3 < l && segment_is_intact(&intact_prev, i, i + 3) && has_hbond(i, i + 3) {
            turn3[i] = true;
            for slot in &mut is_turn[(i + 1)..(i + 3)] {
                *slot = true;
            }
        }
        if i + 4 < l && segment_is_intact(&intact_prev, i, i + 4) && has_hbond(i, i + 4) {
            turn4[i] = true;
            for slot in &mut is_turn[(i + 1)..(i + 4)] {
                *slot = true;
            }
        }
        if i + 5 < l && segment_is_intact(&intact_prev, i, i + 5) && has_hbond(i, i + 5) {
            turn5[i] = true;
            for slot in &mut is_turn[(i + 1)..(i + 5)] {
                *slot = true;
            }
        }
    }

    let mut is_h4 = vec![false; l];
    let mut is_h3 = vec![false; l];
    let mut is_h5 = vec![false; l];

    for i in 1..l {
        if i + 3 < l && turn4[i - 1] && turn4[i] && intact_prev[i] {
            for slot in &mut is_h4[i..=(i + 3)] {
                *slot = true;
            }
        }
        if i + 2 < l && turn3[i - 1] && turn3[i] && intact_prev[i] {
            for slot in &mut is_h3[i..=(i + 2)] {
                *slot = true;
            }
        }
        if i + 4 < l && turn5[i - 1] && turn5[i] && intact_prev[i] {
            for slot in &mut is_h5[i..=(i + 4)] {
                *slot = true;
            }
        }
    }

    // 5. Parallel & antiparallel beta-bridges and beta-bulge ladders (B and E)
    let mut bridges: Vec<BetaBridge> = Vec::new();
    if l >= 5 {
        for i in 1..(l - 1) {
            if !segment_is_intact(&intact_prev, i - 1, i + 1) {
                continue;
            }
            for j in (i + 1)..(l - 1) {
                if coords[i].chain_id == coords[j].chain_id && j - i < 3 {
                    continue;
                }
                if !segment_is_intact(&intact_prev, j - 1, j + 1) {
                    continue;
                }
                let parallel = (has_hbond(i - 1, j) && has_hbond(j, i + 1))
                    || (has_hbond(j - 1, i) && has_hbond(i, j + 1));
                let antiparallel = (has_hbond(i, j) && has_hbond(j, i))
                    || (has_hbond(i - 1, j + 1) && has_hbond(j - 1, i + 1));
                if parallel {
                    bridges.push(BetaBridge {
                        kind: BridgeKind::Parallel,
                        i,
                        j,
                    });
                }
                if antiparallel {
                    bridges.push(BetaBridge {
                        kind: BridgeKind::Antiparallel,
                        i,
                        j,
                    });
                }
            }
        }
    }

    // Connected components (ladders) with Kabsch-Sander 1-vs-4 residue beta-bulge rule
    let nb = bridges.len();
    let mut visited = vec![false; nb];
    let mut is_sheet_e = vec![false; l];
    let mut is_bridge_b = vec![false; l];

    let bridges_linked = |a: BetaBridge, b: BetaBridge| -> bool {
        if a.kind != b.kind {
            return false;
        }
        let (first, second) = if a.i <= b.i { (a, b) } else { (b, a) };
        if first.i == second.i {
            return false;
        }
        if coords[first.i].chain_id != coords[second.i].chain_id
            || coords[first.j].chain_id != coords[second.j].chain_id
        {
            return false;
        }
        let di = second.i - first.i;
        match first.kind {
            BridgeKind::Parallel => {
                if second.j <= first.j {
                    return false;
                }
                let dj = second.j - first.j;
                let bulge_ok = (di <= 2 && dj <= 5) || (di <= 5 && dj <= 2);
                bulge_ok
                    && segment_is_intact(&intact_prev, first.i, second.i)
                    && segment_is_intact(&intact_prev, first.j, second.j)
            }
            BridgeKind::Antiparallel => {
                if first.j <= second.j {
                    return false;
                }
                let dj = first.j - second.j;
                let bulge_ok = (di <= 2 && dj <= 5) || (di <= 5 && dj <= 2);
                bulge_ok
                    && segment_is_intact(&intact_prev, first.i, second.i)
                    && segment_is_intact(&intact_prev, second.j, first.j)
            }
        }
    };

    for start_b in 0..nb {
        if visited[start_b] {
            continue;
        }
        visited[start_b] = true;
        let mut comp = vec![start_b];
        let mut head = 0usize;
        while head < comp.len() {
            let u = comp[head];
            head += 1;
            for v in 0..nb {
                if !visited[v] && bridges_linked(bridges[u], bridges[v]) {
                    visited[v] = true;
                    comp.push(v);
                }
            }
        }

        if comp.len() == 1 {
            let br = bridges[comp[0]];
            is_bridge_b[br.i] = true;
            is_bridge_b[br.j] = true;
        } else {
            let mut i_min = usize::MAX;
            let mut i_max = 0usize;
            let mut j_min = usize::MAX;
            let mut j_max = 0usize;
            for &idx_b in &comp {
                let br = bridges[idx_b];
                i_min = i_min.min(br.i);
                i_max = i_max.max(br.i);
                j_min = j_min.min(br.j);
                j_max = j_max.max(br.j);
            }
            for slot in &mut is_sheet_e[i_min..=i_max] {
                *slot = true;
            }
            for slot in &mut is_sheet_e[j_min..=j_max] {
                *slot = true;
            }
        }
    }

    // 6. Resolve 8-state hierarchy (`H > B/E > G > I > T > S > C`, where multi-bridge sheet E
    //    absorbs an auxiliary isolated bridge B on the same strand residue, while isolated B
    //    outranks G > I > T > S > C).
    let mut state8 = vec![KabschSanderState8::C; l];
    for i in 0..l {
        state8[i] = if is_h4[i] {
            KabschSanderState8::H
        } else if is_sheet_e[i] {
            KabschSanderState8::E
        } else if is_bridge_b[i] {
            KabschSanderState8::B
        } else if is_h3[i] {
            KabschSanderState8::G
        } else if is_h5[i] {
            KabschSanderState8::I
        } else if is_turn[i] {
            KabschSanderState8::T
        } else if kappa_deg[i].is_some_and(|k| k > 70.0) {
            KabschSanderState8::S
        } else {
            KabschSanderState8::C
        };
    }

    // Demote any orphaned 1-residue G or I fragment left after H/E/B boundary override back to T
    for target in [KabschSanderState8::G, KabschSanderState8::I] {
        let min_len = if target == KabschSanderState8::G {
            3
        } else {
            5
        };
        let mut idx = 0usize;
        while idx < l {
            if state8[idx] == target {
                let mut end = idx + 1;
                while end < l && state8[end] == target && intact_prev[end] {
                    end += 1;
                }
                if end - idx < min_len {
                    for (off, slot) in state8[idx..end].iter_mut().enumerate() {
                        let k = idx + off;
                        *slot = if is_turn[k] {
                            KabschSanderState8::T
                        } else if kappa_deg[k].is_some_and(|ang| ang > 70.0) {
                            KabschSanderState8::S
                        } else {
                            KabschSanderState8::C
                        };
                    }
                }
                idx = end;
            } else {
                idx += 1;
            }
        }
    }

    // 7. Collapse KabschSanderState8 + (phi_i, psi_i) into 4-state NMR alphabet SecondaryStructure4:
    //    - Helix: {H, G, I}
    //    - Sheet: {E, B}
    //    - Ppii:  {T, S, C} lacking backbone H-bonds with phi in [-105, -45] and psi in [+115, +175]
    //             across >= 2 consecutive residues or on/adjacent to Proline (Adzhubei et al. 2013)
    //    - Coil:  remaining {T, S, C}
    let mut ppii_cand = vec![false; l];
    for i in 0..l {
        let non_regular = matches!(
            state8[i],
            KabschSanderState8::T | KabschSanderState8::S | KabschSanderState8::C
        );
        let has_any_backbone_hb = !active_co[i].is_empty() || !active_nh[i].is_empty();
        if non_regular && !has_any_backbone_hb {
            if let (Some(phi), Some(psi)) = (phi_deg[i], psi_deg[i]) {
                if is_ppii_dihedral(phi, psi) {
                    ppii_cand[i] = true;
                }
            }
        }
    }

    let mut state4 = vec![SecondaryStructure4::Coil; l];
    let mut n_helix = 0usize;
    let mut n_sheet = 0usize;
    let mut n_ppii = 0usize;
    let mut n_coil = 0usize;

    for i in 0..l {
        let s4 = match state8[i] {
            KabschSanderState8::H | KabschSanderState8::G | KabschSanderState8::I => {
                SecondaryStructure4::Helix
            }
            KabschSanderState8::E | KabschSanderState8::B => SecondaryStructure4::Sheet,
            KabschSanderState8::T | KabschSanderState8::S | KabschSanderState8::C => {
                if ppii_cand[i] {
                    let consec_prev = i > 0 && intact_prev[i] && ppii_cand[i - 1];
                    let consec_next = i + 1 < l && intact_prev[i + 1] && ppii_cand[i + 1];
                    let near_pro = is_proline(&coords[i].res_name)
                        || (i > 0 && intact_prev[i] && is_proline(&coords[i - 1].res_name))
                        || (i + 1 < l && intact_prev[i + 1] && is_proline(&coords[i + 1].res_name));
                    if consec_prev || consec_next || near_pro {
                        SecondaryStructure4::Ppii
                    } else {
                        SecondaryStructure4::Coil
                    }
                } else {
                    SecondaryStructure4::Coil
                }
            }
        };
        state4[i] = s4;
        match s4 {
            SecondaryStructure4::Helix => n_helix += 1,
            SecondaryStructure4::Sheet => n_sheet += 1,
            SecondaryStructure4::Ppii => n_ppii += 1,
            SecondaryStructure4::Coil => n_coil += 1,
        }
    }

    let mut residues = Vec::with_capacity(l);
    for i in 0..l {
        residues.push(ResidueSecondaryStructure {
            seq_index: i,
            seq_id: coords[i].seq_id,
            res_name: coords[i].res_name.clone(),
            chain_id: coords[i].chain_id.clone(),
            state8: state8[i],
            state4: state4[i],
            phi_deg: phi_deg[i],
            psi_deg: psi_deg[i],
            kappa_deg: kappa_deg[i],
            amide_h: amide_h[i],
            co_acceptors: std::mem::take(&mut active_co[i]),
            nh_donors: std::mem::take(&mut active_nh[i]),
        });
    }

    Ok(SecondaryStructureReport {
        residues,
        n_helix,
        n_sheet,
        n_ppii,
        n_coil,
    })
}

/// Convenience helper classifying a `PdbChain` using ideal trans-peptide amide H reconstruction
/// (`use_ideal_ks_amide_h = true`) or the `PdbChain`'s existing `h` coordinates.
pub fn classify_pdb_chain_secondary_structure(
    chain: &PdbChain,
    use_ideal_ks_amide_h: bool,
) -> Result<SecondaryStructureReport, String> {
    let coords = BackboneResidueCoords::from_pdb_chain(chain, use_ideal_ks_amide_h);
    classify_kabsch_sander(&coords)
}
