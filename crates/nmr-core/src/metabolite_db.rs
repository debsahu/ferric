//! Offline CC0 `BMRB-BMSE` Compact Aqueous Metabolite Reference Index, Fast `< 5 ms`
//! Peak-List / `STOCSY`-Cluster Matcher, 1D Reference Spectrum Synthesizer, and Opt-In
//! Desktop `BMRB v2` / `NP-MRD` Online Query Engine.
//!
//! # Clean-Room Scientific & Open-Data Provenance
//! Original clean-room implementation built strictly from public-domain / permissive open
//! scientific data and peer-reviewed NMR metabolomics literature:
//! - **BMRB (Biological Magnetic Resonance Data Bank, `BMRB-BMSE`, `CC0` / Public Domain)**:
//!   Ulrich, E. L., Akutsu, H., Doreleijers, J. F., Harano, Y., Ioannidis, Y. E., Lin, J.,
//!   Livny, M., Mading, S., Maziuk, D., Miller, Z., Nakatani, E., Schulte, C. F., Tolmie, D. E.,
//!   Kent Wenger, R., Yao, H. & Markley, J. L., "BioMagResBank," *Nucleic Acids Res.* **2008**,
//!   36(Database issue), D402–D408 (`10.1093/nar/gkm957`); <https://bmrb.io/metabolomics/>.
//! - **NP-MRD (Natural Products Magnetic Resonance Database, `CC-BY-4.0`)**:
//!   Wishart, D. S., Sayeeda, Z., Budinski, Z., Guo, A., Lee, B. L., Berjanskii, M., Rout, M.,
//!   Peters, H., Dizon, R., Mah, R., Torres-Calzada, C., Hiebert-Giesbrecht, M., Varshavi, D.,
//!   Varshavi, D., Oler, E., Allen, D., Cao, X., Gautam, V., Maras, A., Poynton, E. F.,
//!   Tavangar, P., Yang, V., van Santen, J. A., Linington, R. G. & Hu, X., "NP-MRD: the Natural
//!   Products Magnetic Resonance Database," *Nucleic Acids Res.* **2022**, 50(D1), D622–D631
//!   (`10.1093/nar/gkab1052`); <https://np-mrd.org/>.
//! - **STOCSY (Statistical Total Correlation Spectroscopy)**:
//!   Cloarec, O., Dumas, M.-E., Craig, A., Barton, R. H., Trygg, J., Hudson, J., Blancher, C.,
//!   Gauguier, D., Lindon, J. C., Holmes, E. & Nicholson, J. K., *Anal. Chem.* **2005**, 77(5),
//!   1282–1289 (`10.1021/ac048630x`).
//!
//! **Clean-room notice**: Strictly zero `HMDB` (non-commercial restriction) or `Chenomx`
//! (proprietary) data, libraries, or shift tables were consulted or bundled. All online queries
//! are strictly opt-in, desktop-only (`#[cfg(not(target_arch = "wasm32"))]`), enforce an exact
//! HTTPS host allowlist (`api.bmrb.io`, `np-mrd.org` only), use a `5 s` connect/read timeout
//! and `4 MiB` response cap, disable cross-host redirects, and fail closed when offline.

use serde::{Deserialize, Serialize};

use crate::metabolomics::{SpectralBucket, StocsyResult};

/// Exact strictly-allowlisted HTTPS hosts permitted for opt-in online metabolite queries.
pub const ALLOWED_ONLINE_HOSTS: &[&str] = &["api.bmrb.io", "np-mrd.org"];

/// Canonical BMRB v2 REST API base URL.
pub const BMRB_V2_API_BASE: &str = "https://api.bmrb.io/v2";

/// Canonical NP-MRD REST API base URL.
pub const NPMRD_API_BASE: &str = "https://np-mrd.org";

/// Hard upper bound on any single online HTTP response payload (`4 MiB`).
pub const MAX_ONLINE_RESPONSE_BYTES: usize = 4 * 1024 * 1024;

/// Connect and total transfer timeout in seconds for opt-in online metabolite queries (`5 s`).
pub const ONLINE_QUERY_TIMEOUT_SECS: u64 = 5;

/// Default STOCSY Pearson correlation floor (`r >= 0.85`) for coherent spin-system clustering.
pub const DEFAULT_STOCSY_CLUSTER_MIN_R: f64 = 0.85;

/// Supported NMR nuclei for offline/online metabolite identification and reference synthesis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MetaboliteNucleus {
    /// Proton (`1H`, aqueous D2O reference DSS/TSP = 0.00 ppm).
    H1,
    /// Carbon-13 (`13C`, aqueous D2O reference DSS = 0.00 ppm).
    C13,
}

impl MetaboliteNucleus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::H1 => "1H",
            Self::C13 => "13C",
        }
    }

    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "1h" | "h1" | "h" | "proton" => Ok(Self::H1),
            "13c" | "c13" | "c" | "carbon" => Ok(Self::C13),
            other => Err(format!(
                "unsupported nucleus {other:?} (expected 1h or 13c)"
            )),
        }
    }

    /// Recommended default matching tolerance in ppm for this nucleus.
    pub fn default_tol_ppm(&self) -> f64 {
        match self {
            Self::H1 => 0.03,
            Self::C13 => 0.50,
        }
    }
}

/// First-order 1D `1H` NMR multiplet splitting pattern.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PeakMultiplicity {
    Singlet,
    Doublet,
    Triplet,
    Quartet,
    DoubletOfDoublets,
    DoubletOfTriplets,
    Quintet,
    Multiplet,
}

impl PeakMultiplicity {
    pub fn code(&self) -> &'static str {
        match self {
            Self::Singlet => "s",
            Self::Doublet => "d",
            Self::Triplet => "t",
            Self::Quartet => "q",
            Self::DoubletOfDoublets => "dd",
            Self::DoubletOfTriplets => "dt",
            Self::Quintet => "quint",
            Self::Multiplet => "m",
        }
    }

    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "s" | "singlet" => Some(Self::Singlet),
            "d" | "doublet" => Some(Self::Doublet),
            "t" | "triplet" => Some(Self::Triplet),
            "q" | "quartet" => Some(Self::Quartet),
            "dd" | "doublet_of_doublets" | "doublet-of-doublets" => Some(Self::DoubletOfDoublets),
            "dt" | "td" | "doublet_of_triplets" => Some(Self::DoubletOfTriplets),
            "quint" | "quintet" | "p" | "pentet" => Some(Self::Quintet),
            "m" | "multiplet" | "br" | "bs" => Some(Self::Multiplet),
            _ => None,
        }
    }

    /// Pairwise multiplicity consistency score in `[0.0, 1.0]`.
    pub fn compatibility(self, other: Self) -> f64 {
        if self == other {
            return 1.0;
        }
        if matches!(self, Self::Multiplet) || matches!(other, Self::Multiplet) {
            return 0.70;
        }
        match (self, other) {
            (Self::DoubletOfDoublets, Self::Triplet)
            | (Self::Triplet, Self::DoubletOfDoublets)
            | (Self::DoubletOfDoublets, Self::Quartet)
            | (Self::Quartet, Self::DoubletOfDoublets)
            | (Self::DoubletOfTriplets, Self::Triplet)
            | (Self::Triplet, Self::DoubletOfTriplets) => 0.65,
            _ => 0.15,
        }
    }
}

/// Reference `1H` peak entry for an aqueous (`D2O`, pH ~7.0–7.4) metabolite.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetabolitePeak1H {
    pub assignment: String,
    pub ppm: f64,
    pub multiplicity: PeakMultiplicity,
    pub j_hz: Vec<f64>,
    pub n_protons: u8,
}

/// Reference `13C` chemical shift entry for an aqueous (`D2O`, pH ~7.0–7.4) metabolite.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetaboliteShift13C {
    pub assignment: String,
    pub ppm: f64,
    pub n_carbons: u8,
}

/// Provenance database source for a metabolite entry or search candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MetaboliteSource {
    /// Built-in offline CC0 `BMRB-BMSE` compact index.
    BmrbBmseOffline,
    /// Opt-in live desktop query against `BMRB v2` REST API (`https://api.bmrb.io/v2/...`).
    BmrbOnline,
    /// Opt-in live desktop query against `NP-MRD` (`https://np-mrd.org/...`, `CC-BY-4.0`).
    NpMrdOnline,
}

impl MetaboliteSource {
    pub fn label(&self) -> &'static str {
        match self {
            Self::BmrbBmseOffline => "BMRB-BMSE (CC0 Offline)",
            Self::BmrbOnline => "BMRB v2 (Online)",
            Self::NpMrdOnline => "NP-MRD (CC-BY-4.0 Online)",
        }
    }
}

/// Canonical metabolite record with `BMRB-BMSE` ID, formula, `SMILES`, `1H` peaks, and `13C` shifts.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetaboliteEntry {
    pub bmse_id: String,
    pub name: String,
    pub formula: String,
    pub smiles: String,
    pub solvent: String,
    pub ph: f64,
    pub peaks_1h: Vec<MetabolitePeak1H>,
    pub shifts_13c: Vec<MetaboliteShift13C>,
    pub source: MetaboliteSource,
}

/// User or STOCSY query peak supplied to [`query_offline_metabolites`] or online lookup.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QueryPeak {
    pub ppm: f64,
    /// Optional observed multiplicity (`s`, `d`, `t`, `q`, `dd`, `m`).
    pub multiplicity: Option<PeakMultiplicity>,
    /// Optional observed scalar J-couplings in Hz.
    pub j_hz: Vec<f64>,
    /// Optional relative peak height / integral.
    pub intensity: Option<f64>,
    /// Optional `STOCSY` Pearson correlation `r` to an active driver resonance (`[-1, 1]`).
    /// When present on query peaks, peaks with `r >= 0.85` define a coherent spin-system cluster.
    pub stocsy_r: Option<f64>,
}

impl QueryPeak {
    pub fn from_ppm(ppm: f64) -> Self {
        Self {
            ppm,
            multiplicity: None,
            j_hz: Vec::new(),
            intensity: None,
            stocsy_r: None,
        }
    }

    pub fn with_multiplet(ppm: f64, multiplicity: PeakMultiplicity, j_hz: Vec<f64>) -> Self {
        Self {
            ppm,
            multiplicity: Some(multiplicity),
            j_hz,
            intensity: None,
            stocsy_r: None,
        }
    }

    pub fn from_stocsy_cluster(ppm: f64, stocsy_r: f64) -> Self {
        Self {
            ppm,
            multiplicity: None,
            j_hz: Vec::new(),
            intensity: None,
            stocsy_r: Some(stocsy_r),
        }
    }
}

/// Individual matched reference-to-query peak correspondence.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MatchedPeakPair {
    pub assignment: String,
    pub ref_ppm: f64,
    pub query_ppm: f64,
    pub delta_ppm: f64,
    pub mult_j_consistency: f64,
}

/// Ranked candidate metabolite hit from offline or online matching.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetaboliteMatch {
    pub entry: MetaboliteEntry,
    pub nucleus: MetaboliteNucleus,
    /// Composite confidence score strictly in `[0.0, 1.0]`.
    pub score: f64,
    /// Fraction of reference peaks matched (`matched_peaks / total_ref_peaks`).
    pub recall: f64,
    /// Fraction of active query peaks explained (`matched_query_peaks / total_query_peaks`).
    pub precision: f64,
    /// Mean shift-proximity score in `[0.0, 1.0]`.
    pub shift_score: f64,
    /// Multiplicity and scalar J-coupling consistency score in `[0.0, 1.0]`.
    pub mult_j_score: f64,
    /// Number of matched reference peaks.
    pub matched_ref_peaks: usize,
    /// Total reference peaks for this metabolite in the queried nucleus.
    pub total_ref_peaks: usize,
    /// Mean absolute chemical-shift error (`|ref_ppm - query_ppm|`) in ppm.
    pub mean_abs_error_ppm: f64,
    /// Detailed peak-by-peak assignment breakdown.
    pub matched_pairs: Vec<MatchedPeakPair>,
}

/// Synthesized 1D reference spectrum (Lorentzian curve + discrete stick list) for 1-click overlay.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReferenceSpectrum1D {
    pub bmse_id: String,
    pub name: String,
    pub nucleus: MetaboliteNucleus,
    pub obs_mhz: f64,
    /// Descending ppm axis (`max_ppm` -> `min_ppm`).
    pub ppm_grid: Vec<f64>,
    /// Normalized Lorentzian intensity profile in `[0.0, 1.0]`.
    pub intensities: Vec<f64>,
    /// Discrete stick lines `(ppm, relative_height)` with max height `1.0`.
    pub sticks: Vec<(f64, f64)>,
}

/// Online source toggle flags for opt-in desktop queries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OnlineQuerySources {
    pub bmrb: bool,
    pub npmrd: bool,
}

impl OnlineQuerySources {
    pub const fn all() -> Self {
        Self {
            bmrb: true,
            npmrd: true,
        }
    }

    pub const fn bmrb_only() -> Self {
        Self {
            bmrb: true,
            npmrd: false,
        }
    }

    pub const fn npmrd_only() -> Self {
        Self {
            bmrb: false,
            npmrd: true,
        }
    }
}

/// Validated allowlisted online host identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OnlineHost {
    BmrbApiV2,
    NpMrd,
}

/// Enforce the strict HTTPS host allowlist (`https://api.bmrb.io/...` and `https://np-mrd.org/...`
/// ONLY), rejecting non-HTTPS schemes, userinfo (`@`), custom ports, IP literals, lookalike
/// subdomains/suffixes, control characters, empty paths, and `..` traversal.
pub fn validate_metabolite_online_url(url: &str) -> Result<OnlineHost, String> {
    let u = url.trim();
    if u.is_empty() || u.bytes().any(|b| b <= 0x20 || b == 0x7f || b == b'\\') {
        return Err(format!(
            "blocked unsafe URL {url:?}: contains control characters, whitespace, or backslashes"
        ));
    }
    let Some(after_scheme) = u.strip_prefix("https://") else {
        return Err(format!(
            "blocked URL {url:?}: only strict https:// requests to api.bmrb.io or np-mrd.org are permitted"
        ));
    };
    let split_idx = after_scheme
        .find(['/', '?', '#'])
        .unwrap_or(after_scheme.len());
    let authority = &after_scheme[..split_idx];
    let rest = &after_scheme[split_idx..];

    if authority.contains('@') || authority.contains(':') {
        return Err(format!(
            "blocked URL {url:?}: userinfo and custom ports are forbidden"
        ));
    }

    let host = match authority {
        "api.bmrb.io" => OnlineHost::BmrbApiV2,
        "np-mrd.org" => OnlineHost::NpMrd,
        other => {
            return Err(format!(
                "blocked host {other:?} in URL {url:?}: strictly allowlisted to api.bmrb.io and np-mrd.org"
            ));
        }
    };

    if !rest.starts_with('/') || rest.len() <= 1 || rest.contains("..") {
        return Err(format!(
            "blocked unsafe or empty path in metabolite URL {url:?}"
        ));
    }
    Ok(host)
}

// ============================================================================
// Static Offline CC0 BMRB-BMSE Compact Metabolite Index (46 Canonical Metabolites)
// ============================================================================

struct StaticPeak1H {
    assignment: &'static str,
    ppm: f64,
    multiplicity: PeakMultiplicity,
    j_hz: &'static [f64],
    n_protons: u8,
}

struct StaticShift13C {
    assignment: &'static str,
    ppm: f64,
    n_carbons: u8,
}

struct StaticMetaboliteEntry {
    bmse_id: &'static str,
    name: &'static str,
    formula: &'static str,
    smiles: &'static str,
    ph: f64,
    peaks_1h: &'static [StaticPeak1H],
    shifts_13c: &'static [StaticShift13C],
}

impl StaticMetaboliteEntry {
    fn to_entry(&self) -> MetaboliteEntry {
        MetaboliteEntry {
            bmse_id: self.bmse_id.to_string(),
            name: self.name.to_string(),
            formula: self.formula.to_string(),
            smiles: self.smiles.to_string(),
            solvent: "D2O".to_string(),
            ph: self.ph,
            peaks_1h: self
                .peaks_1h
                .iter()
                .map(|p| MetabolitePeak1H {
                    assignment: p.assignment.to_string(),
                    ppm: p.ppm,
                    multiplicity: p.multiplicity,
                    j_hz: p.j_hz.to_vec(),
                    n_protons: p.n_protons,
                })
                .collect(),
            shifts_13c: self
                .shifts_13c
                .iter()
                .map(|c| MetaboliteShift13C {
                    assignment: c.assignment.to_string(),
                    ppm: c.ppm,
                    n_carbons: c.n_carbons,
                })
                .collect(),
            source: MetaboliteSource::BmrbBmseOffline,
        }
    }
}

macro_rules! p1h {
    ($asgn:expr, $ppm:expr, $mult:ident, [$($j:expr),*], $nh:expr) => {
        StaticPeak1H {
            assignment: $asgn,
            ppm: $ppm,
            multiplicity: PeakMultiplicity::$mult,
            j_hz: &[$($j),*],
            n_protons: $nh,
        }
    };
}

macro_rules! c13 {
    ($asgn:expr, $ppm:expr, $nc:expr) => {
        StaticShift13C {
            assignment: $asgn,
            ppm: $ppm,
            n_carbons: $nc,
        }
    };
}

const OFFLINE_BMRB_METABOLITES: &[StaticMetaboliteEntry] = &[
    // 1. L-Lactate
    StaticMetaboliteEntry {
        bmse_id: "bmse000036",
        name: "L-Lactate",
        formula: "C3H6O3",
        smiles: "C[C@H](O)C(=O)O",
        ph: 7.4,
        peaks_1h: &[
            p1h!("CH3", 1.330, Doublet, [6.9], 3),
            p1h!("CH", 4.110, Quartet, [6.9], 1),
        ],
        shifts_13c: &[
            c13!("C3(CH3)", 22.8, 1),
            c13!("C2(CH)", 71.2, 1),
            c13!("C1(COO-)", 185.1, 1),
        ],
    },
    // 2. L-Alanine
    StaticMetaboliteEntry {
        bmse_id: "bmse000028",
        name: "L-Alanine",
        formula: "C3H7NO2",
        smiles: "C[C@H](N)C(=O)O",
        ph: 7.4,
        peaks_1h: &[
            p1h!("beta-CH3", 1.478, Doublet, [7.2], 3),
            p1h!("alpha-CH", 3.778, Quartet, [7.2], 1),
        ],
        shifts_13c: &[
            c13!("Cbeta", 18.9, 1),
            c13!("Calpha", 53.3, 1),
            c13!("C(COO-)", 178.5, 1),
        ],
    },
    // 3. alpha-D-Glucose
    StaticMetaboliteEntry {
        bmse_id: "bmse000016a",
        name: "alpha-D-Glucose",
        formula: "C6H12O6",
        smiles: "OC[C@H]1O[C@H](O)[C@H](O)[C@@H](O)[C@@H]1O",
        ph: 7.4,
        peaks_1h: &[
            p1h!("H4", 3.408, Triplet, [9.4], 1),
            p1h!("H2", 3.533, DoubletOfDoublets, [9.8, 3.7], 1),
            p1h!("H3", 3.710, Triplet, [9.5], 1),
            p1h!("H5/H6", 3.833, Multiplet, [9.5, 4.5], 3),
            p1h!("H1-alpha", 5.233, Doublet, [3.7], 1),
        ],
        shifts_13c: &[
            c13!("C6", 63.4, 1),
            c13!("C4", 72.4, 1),
            c13!("C2", 74.2, 1),
            c13!("C5", 74.3, 1),
            c13!("C3", 75.5, 1),
            c13!("C1", 94.8, 1),
        ],
    },
    // 4. beta-D-Glucose
    StaticMetaboliteEntry {
        bmse_id: "bmse000016b",
        name: "beta-D-Glucose",
        formula: "C6H12O6",
        smiles: "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O",
        ph: 7.4,
        peaks_1h: &[
            p1h!("H2", 3.242, DoubletOfDoublets, [9.2, 7.9], 1),
            p1h!("H4", 3.400, Triplet, [9.3], 1),
            p1h!("H3/H5", 3.475, Triplet, [9.2], 2),
            p1h!("H6b", 3.720, DoubletOfDoublets, [12.2, 5.8], 1),
            p1h!("H6a", 3.895, DoubletOfDoublets, [12.2, 2.1], 1),
            p1h!("H1-beta", 4.642, Doublet, [7.9], 1),
        ],
        shifts_13c: &[
            c13!("C6", 63.5, 1),
            c13!("C4", 72.4, 1),
            c13!("C2", 76.9, 1),
            c13!("C3", 78.5, 1),
            c13!("C5", 78.7, 1),
            c13!("C1", 98.7, 1),
        ],
    },
    // 5. Acetate
    StaticMetaboliteEntry {
        bmse_id: "bmse000191",
        name: "Acetate",
        formula: "C2H4O2",
        smiles: "CC(=O)O",
        ph: 7.4,
        peaks_1h: &[p1h!("CH3", 1.918, Singlet, [], 3)],
        shifts_13c: &[c13!("C2(CH3)", 26.0, 1), c13!("C1(COO-)", 184.0, 1)],
    },
    // 6. Citrate
    StaticMetaboliteEntry {
        bmse_id: "bmse000074",
        name: "Citrate",
        formula: "C6H8O7",
        smiles: "OC(CC(=O)O)(CC(=O)O)C(=O)O",
        ph: 7.4,
        peaks_1h: &[
            p1h!("CH2a", 2.535, Doublet, [15.3], 2),
            p1h!("CH2b", 2.675, Doublet, [15.3], 2),
        ],
        shifts_13c: &[
            c13!("C2/C4(CH2)", 48.2, 2),
            c13!("C3(C-OH)", 77.7, 1),
            c13!("C1/C5(COO-)", 181.5, 2),
            c13!("C6(COO-)", 184.2, 1),
        ],
    },
    // 7. Succinate
    StaticMetaboliteEntry {
        bmse_id: "bmse000181",
        name: "Succinate",
        formula: "C4H6O4",
        smiles: "O=C(O)CCC(=O)O",
        ph: 7.4,
        peaks_1h: &[p1h!("2xCH2", 2.405, Singlet, [], 4)],
        shifts_13c: &[c13!("C2/C3(CH2)", 36.8, 2), c13!("C1/C4(COO-)", 184.9, 2)],
    },
    // 8. Creatine
    StaticMetaboliteEntry {
        bmse_id: "bmse000059",
        name: "Creatine",
        formula: "C4H9N3O2",
        smiles: "CN(CC(=O)O)C(=N)N",
        ph: 7.4,
        peaks_1h: &[
            p1h!("N-CH3", 3.032, Singlet, [], 3),
            p1h!("N-CH2", 3.928, Singlet, [], 2),
        ],
        shifts_13c: &[
            c13!("N-CH3", 39.6, 1),
            c13!("CH2", 56.5, 1),
            c13!("C=N", 159.8, 1),
            c13!("COO-", 177.2, 1),
        ],
    },
    // 9. Creatinine
    StaticMetaboliteEntry {
        bmse_id: "bmse000060",
        name: "Creatinine",
        formula: "C4H7N3O",
        smiles: "CN1CC(=O)N=C1N",
        ph: 7.4,
        peaks_1h: &[
            p1h!("N-CH3", 3.043, Singlet, [], 3),
            p1h!("CH2", 4.052, Singlet, [], 2),
        ],
        shifts_13c: &[
            c13!("N-CH3", 32.9, 1),
            c13!("CH2", 59.0, 1),
            c13!("C=N", 171.8, 1),
            c13!("C=O", 191.5, 1),
        ],
    },
    // 10. Glycine
    StaticMetaboliteEntry {
        bmse_id: "bmse000087",
        name: "Glycine",
        formula: "C2H5NO2",
        smiles: "NCC(=O)O",
        ph: 7.4,
        peaks_1h: &[p1h!("CH2", 3.555, Singlet, [], 2)],
        shifts_13c: &[c13!("Calpha", 44.2, 1), c13!("COO-", 175.2, 1)],
    },
    // 11. Taurine
    StaticMetaboliteEntry {
        bmse_id: "bmse000140",
        name: "Taurine",
        formula: "C2H7NO3S",
        smiles: "NCCS(=O)(=O)O",
        ph: 7.4,
        peaks_1h: &[
            p1h!("CH2-NH3", 3.260, Triplet, [6.6], 2),
            p1h!("CH2-SO3", 3.425, Triplet, [6.6], 2),
        ],
        shifts_13c: &[c13!("CH2-NH3", 38.2, 1), c13!("CH2-SO3", 50.4, 1)],
    },
    // 12. Choline
    StaticMetaboliteEntry {
        bmse_id: "bmse000057",
        name: "Choline",
        formula: "C5H14NO+",
        smiles: "C[N+](C)(C)CCO",
        ph: 7.4,
        peaks_1h: &[
            p1h!("N(CH3)3", 3.200, Singlet, [], 9),
            p1h!("CH2-N", 3.515, Triplet, [5.0], 2),
            p1h!("CH2-O", 4.060, Multiplet, [5.0], 2),
        ],
        shifts_13c: &[
            c13!("N(CH3)3", 56.6, 3),
            c13!("CH2-O", 58.3, 1),
            c13!("CH2-N", 70.1, 1),
        ],
    },
    // 13. TMAO
    StaticMetaboliteEntry {
        bmse_id: "bmse000279",
        name: "TMAO",
        formula: "C3H9NO",
        smiles: "C[N+](C)(C)[O-]",
        ph: 7.4,
        peaks_1h: &[p1h!("3xCH3", 3.266, Singlet, [], 9)],
        shifts_13c: &[c13!("3xCH3", 62.2, 3)],
    },
    // 14. L-Glutamine
    StaticMetaboliteEntry {
        bmse_id: "bmse000033",
        name: "L-Glutamine",
        formula: "C5H10N2O3",
        smiles: "NC(=O)CC[C@H](N)C(=O)O",
        ph: 7.4,
        peaks_1h: &[
            p1h!("beta-CH2", 2.135, Multiplet, [7.2], 2),
            p1h!("gamma-CH2", 2.452, Multiplet, [7.5], 2),
            p1h!("alpha-CH", 3.770, Triplet, [6.2], 1),
        ],
        shifts_13c: &[
            c13!("Cbeta", 29.0, 1),
            c13!("Cgamma", 33.6, 1),
            c13!("Calpha", 57.0, 1),
            c13!("C(COO-)", 176.6, 1),
            c13!("Cdelta(CONH2)", 180.4, 1),
        ],
    },
    // 15. L-Glutamate
    StaticMetaboliteEntry {
        bmse_id: "bmse000032",
        name: "L-Glutamate",
        formula: "C5H9NO4",
        smiles: "N[C@@H](CCC(=O)O)C(=O)O",
        ph: 7.4,
        peaks_1h: &[
            p1h!("beta-CH2", 2.062, Multiplet, [7.2], 2),
            p1h!("gamma-CH2", 2.348, Multiplet, [7.4], 2),
            p1h!("alpha-CH", 3.755, DoubletOfDoublets, [7.2, 4.7], 1),
        ],
        shifts_13c: &[
            c13!("Cbeta", 29.7, 1),
            c13!("Cgamma", 36.2, 1),
            c13!("Calpha", 57.4, 1),
            c13!("C(COO-)", 177.3, 1),
            c13!("Cdelta(COO-)", 183.9, 1),
        ],
    },
    // 16. L-Valine
    StaticMetaboliteEntry {
        bmse_id: "bmse000049",
        name: "L-Valine",
        formula: "C5H11NO2",
        smiles: "CC(C)[C@H](N)C(=O)O",
        ph: 7.4,
        peaks_1h: &[
            p1h!("gamma1-CH3", 0.985, Doublet, [7.0], 3),
            p1h!("gamma2-CH3", 1.037, Doublet, [7.0], 3),
            p1h!("beta-CH", 2.268, Multiplet, [7.0, 4.4], 1),
            p1h!("alpha-CH", 3.605, Doublet, [4.4], 1),
        ],
        shifts_13c: &[
            c13!("Cgamma1", 19.4, 1),
            c13!("Cgamma2", 20.8, 1),
            c13!("Cbeta", 31.9, 1),
            c13!("Calpha", 63.2, 1),
            c13!("C(COO-)", 177.0, 1),
        ],
    },
    // 17. L-Leucine
    StaticMetaboliteEntry {
        bmse_id: "bmse000038",
        name: "L-Leucine",
        formula: "C6H13NO2",
        smiles: "CC(C)C[C@H](N)C(=O)O",
        ph: 7.4,
        peaks_1h: &[
            p1h!("delta1-CH3", 0.952, Doublet, [6.4], 3),
            p1h!("delta2-CH3", 0.964, Doublet, [6.4], 3),
            p1h!("beta-CH2/gamma-CH", 1.706, Multiplet, [6.4], 3),
            p1h!("alpha-CH", 3.728, Triplet, [6.2], 1),
        ],
        shifts_13c: &[
            c13!("Cdelta1", 23.6, 1),
            c13!("Cdelta2", 24.8, 1),
            c13!("Cgamma", 26.8, 1),
            c13!("Cbeta", 42.6, 1),
            c13!("Calpha", 56.2, 1),
            c13!("C(COO-)", 178.3, 1),
        ],
    },
    // 18. L-Isoleucine
    StaticMetaboliteEntry {
        bmse_id: "bmse000035",
        name: "L-Isoleucine",
        formula: "C6H13NO2",
        smiles: "CC[C@H](C)[C@H](N)C(=O)O",
        ph: 7.4,
        peaks_1h: &[
            p1h!("delta-CH3", 0.933, Triplet, [7.4], 3),
            p1h!("gamma2-CH3", 1.005, Doublet, [7.0], 3),
            p1h!("gamma1a-CH2", 1.255, Multiplet, [7.4, 7.0], 1),
            p1h!("gamma1b-CH2", 1.465, Multiplet, [7.4, 7.0], 1),
            p1h!("beta-CH", 1.975, Multiplet, [7.0, 4.0], 1),
            p1h!("alpha-CH", 3.667, Doublet, [4.0], 1),
        ],
        shifts_13c: &[
            c13!("Cdelta1", 13.8, 1),
            c13!("Cgamma2", 17.5, 1),
            c13!("Cgamma1", 27.2, 1),
            c13!("Cbeta", 38.6, 1),
            c13!("Calpha", 62.4, 1),
            c13!("C(COO-)", 176.9, 1),
        ],
    },
    // 19. L-Tyrosine
    StaticMetaboliteEntry {
        bmse_id: "bmse000048",
        name: "L-Tyrosine",
        formula: "C9H11NO3",
        smiles: "N[C@@H](Cc1ccc(O)cc1)C(=O)O",
        ph: 7.4,
        peaks_1h: &[
            p1h!("beta-CH2a", 3.055, DoubletOfDoublets, [14.5, 8.0], 1),
            p1h!("beta-CH2b", 3.195, DoubletOfDoublets, [14.5, 5.2], 1),
            p1h!("alpha-CH", 3.935, DoubletOfDoublets, [8.0, 5.2], 1),
            p1h!("ring-H3/H5", 6.895, Doublet, [8.4], 2),
            p1h!("ring-H2/H6", 7.190, Doublet, [8.4], 2),
        ],
        shifts_13c: &[
            c13!("Cbeta", 38.3, 1),
            c13!("Calpha", 58.9, 1),
            c13!("Cepsilon", 118.6, 2),
            c13!("Cgamma", 129.6, 1),
            c13!("Cdelta", 133.4, 2),
            c13!("Czeta", 157.5, 1),
            c13!("C(COO-)", 176.8, 1),
        ],
    },
    // 20. L-Phenylalanine
    StaticMetaboliteEntry {
        bmse_id: "bmse000043",
        name: "L-Phenylalanine",
        formula: "C9H11NO2",
        smiles: "N[C@@H](Cc1ccccc1)C(=O)O",
        ph: 7.4,
        peaks_1h: &[
            p1h!("beta-CH2a", 3.125, DoubletOfDoublets, [14.5, 8.0], 1),
            p1h!("beta-CH2b", 3.280, DoubletOfDoublets, [14.5, 5.2], 1),
            p1h!("alpha-CH", 3.990, DoubletOfDoublets, [8.0, 5.2], 1),
            p1h!("ortho-CH", 7.328, Doublet, [7.3], 2),
            p1h!("para-CH", 7.375, Triplet, [7.3], 1),
            p1h!("meta-CH", 7.425, Triplet, [7.4], 2),
        ],
        shifts_13c: &[
            c13!("Cbeta", 39.2, 1),
            c13!("Calpha", 58.8, 1),
            c13!("Czeta", 130.3, 1),
            c13!("Cepsilon", 131.7, 2),
            c13!("Cdelta", 132.1, 2),
            c13!("Cgamma", 137.9, 1),
            c13!("C(COO-)", 176.7, 1),
        ],
    },
    // 21. L-Histidine
    StaticMetaboliteEntry {
        bmse_id: "bmse000034",
        name: "L-Histidine",
        formula: "C6H9N3O2",
        smiles: "N[C@@H](Cc1c[nH]cn1)C(=O)O",
        ph: 7.4,
        peaks_1h: &[
            p1h!("beta-CH2a", 3.155, DoubletOfDoublets, [15.5, 7.8], 1),
            p1h!("beta-CH2b", 3.245, DoubletOfDoublets, [15.5, 4.8], 1),
            p1h!("alpha-CH", 3.985, DoubletOfDoublets, [7.8, 4.8], 1),
            p1h!("ring-C4H", 7.085, Singlet, [], 1),
            p1h!("ring-C2H", 7.865, Singlet, [], 1),
        ],
        shifts_13c: &[
            c13!("Cbeta", 30.3, 1),
            c13!("Calpha", 57.5, 1),
            c13!("Cdelta2", 120.1, 1),
            c13!("Cgamma", 134.5, 1),
            c13!("Cepsilon1", 138.8, 1),
            c13!("C(COO-)", 176.4, 1),
        ],
    },
    // 22. Formate
    StaticMetaboliteEntry {
        bmse_id: "bmse000206",
        name: "Formate",
        formula: "CH2O2",
        smiles: "O=CO",
        ph: 7.4,
        peaks_1h: &[p1h!("CH", 8.448, Singlet, [], 1)],
        shifts_13c: &[c13!("C1", 173.8, 1)],
    },
    // 23. Pyruvate
    StaticMetaboliteEntry {
        bmse_id: "bmse000174",
        name: "Pyruvate",
        formula: "C3H4O3",
        smiles: "CC(=O)C(=O)O",
        ph: 7.4,
        peaks_1h: &[p1h!("CH3", 2.368, Singlet, [], 3)],
        shifts_13c: &[
            c13!("C3(CH3)", 29.2, 1),
            c13!("C1(COO-)", 173.0, 1),
            c13!("C2(C=O)", 207.8, 1),
        ],
    },
    // 24. Fumarate
    StaticMetaboliteEntry {
        bmse_id: "bmse000157",
        name: "Fumarate",
        formula: "C4H4O4",
        smiles: "O=C(O)/C=C/C(=O)O",
        ph: 7.4,
        peaks_1h: &[p1h!("2xCH", 6.515, Singlet, [], 2)],
        shifts_13c: &[c13!("C2/C3(CH=)", 138.1, 2), c13!("C1/C4(COO-)", 177.5, 2)],
    },
    // 25. Ethanol
    StaticMetaboliteEntry {
        bmse_id: "bmse000297",
        name: "Ethanol",
        formula: "C2H6O",
        smiles: "CCO",
        ph: 7.0,
        peaks_1h: &[
            p1h!("CH3", 1.178, Triplet, [7.1], 3),
            p1h!("CH2", 3.650, Quartet, [7.1], 2),
        ],
        shifts_13c: &[c13!("C2(CH3)", 19.6, 1), c13!("C1(CH2)", 60.2, 1)],
    },
    // 26. Methanol
    StaticMetaboliteEntry {
        bmse_id: "bmse000298",
        name: "Methanol",
        formula: "CH4O",
        smiles: "CO",
        ph: 7.0,
        peaks_1h: &[p1h!("CH3", 3.350, Singlet, [], 3)],
        shifts_13c: &[c13!("C1(CH3)", 51.6, 1)],
    },
    // 27. Acetone
    StaticMetaboliteEntry {
        bmse_id: "bmse000201",
        name: "Acetone",
        formula: "C3H6O",
        smiles: "CC(=O)C",
        ph: 7.0,
        peaks_1h: &[p1h!("2xCH3", 2.225, Singlet, [], 6)],
        shifts_13c: &[c13!("C1/C3(CH3)", 32.9, 2), c13!("C2(C=O)", 218.0, 1)],
    },
    // 28. Betaine
    StaticMetaboliteEntry {
        bmse_id: "bmse000053",
        name: "Betaine",
        formula: "C5H11NO2",
        smiles: "C[N+](C)(C)CC(=O)[O-]",
        ph: 7.4,
        peaks_1h: &[
            p1h!("N(CH3)3", 3.255, Singlet, [], 9),
            p1h!("CH2", 3.895, Singlet, [], 2),
        ],
        shifts_13c: &[
            c13!("N(CH3)3", 56.1, 3),
            c13!("CH2", 68.9, 1),
            c13!("COO-", 172.0, 1),
        ],
    },
    // 29. Myo-inositol
    StaticMetaboliteEntry {
        bmse_id: "bmse000112",
        name: "Myo-inositol",
        formula: "C6H12O6",
        smiles: "O[C@H]1[C@H](O)[C@@H](O)[C@H](O)[C@@H](O)[C@H]1O",
        ph: 7.4,
        peaks_1h: &[
            p1h!("H5", 3.278, Triplet, [9.3], 1),
            p1h!("H1/H3", 3.532, DoubletOfDoublets, [9.9, 2.9], 2),
            p1h!("H4/H6", 3.620, Triplet, [9.5], 2),
            p1h!("H2", 4.058, Triplet, [2.9], 1),
        ],
        shifts_13c: &[
            c13!("C1/C3", 73.9, 2),
            c13!("C2", 75.0, 1),
            c13!("C4/C6", 75.2, 2),
            c13!("C5", 77.1, 1),
        ],
    },
    // 30. L-Serine
    StaticMetaboliteEntry {
        bmse_id: "bmse000046",
        name: "L-Serine",
        formula: "C3H7NO3",
        smiles: "N[C@@H](CO)C(=O)O",
        ph: 7.4,
        peaks_1h: &[
            p1h!("alpha-CH", 3.838, DoubletOfDoublets, [5.8, 3.8], 1),
            p1h!("beta-CH2a", 3.942, DoubletOfDoublets, [12.2, 5.8], 1),
            p1h!("beta-CH2b", 3.980, DoubletOfDoublets, [12.2, 3.8], 1),
        ],
        shifts_13c: &[
            c13!("Calpha", 59.2, 1),
            c13!("Cbeta", 63.0, 1),
            c13!("C(COO-)", 175.1, 1),
        ],
    },
    // 31. L-Threonine
    StaticMetaboliteEntry {
        bmse_id: "bmse000047",
        name: "L-Threonine",
        formula: "C4H9NO3",
        smiles: "C[C@@H](O)[C@H](N)C(=O)O",
        ph: 7.4,
        peaks_1h: &[
            p1h!("gamma-CH3", 1.324, Doublet, [6.6], 3),
            p1h!("alpha-CH", 3.580, Doublet, [4.9], 1),
            p1h!("beta-CH", 4.248, Multiplet, [6.6, 4.9], 1),
        ],
        shifts_13c: &[
            c13!("Cgamma", 22.3, 1),
            c13!("Calpha", 63.2, 1),
            c13!("Cbeta", 68.8, 1),
            c13!("C(COO-)", 175.8, 1),
        ],
    },
    // 32. L-Proline
    StaticMetaboliteEntry {
        bmse_id: "bmse000044",
        name: "L-Proline",
        formula: "C5H9NO2",
        smiles: "O=C(O)[C@@H]1CCCN1",
        ph: 7.4,
        peaks_1h: &[
            p1h!("gamma-CH2", 2.002, Multiplet, [7.2], 2),
            p1h!("beta-CH2a", 2.065, Multiplet, [7.2], 1),
            p1h!("beta-CH2b", 2.342, Multiplet, [7.2], 1),
            p1h!("delta-CH2a", 3.332, Multiplet, [7.2], 1),
            p1h!("delta-CH2b", 3.412, Multiplet, [7.2], 1),
            p1h!("alpha-CH", 4.125, DoubletOfDoublets, [8.6, 6.4], 1),
        ],
        shifts_13c: &[
            c13!("Cgamma", 26.5, 1),
            c13!("Cbeta", 31.7, 1),
            c13!("Cdelta", 48.9, 1),
            c13!("Calpha", 64.0, 1),
            c13!("C(COO-)", 177.2, 1),
        ],
    },
    // 33. L-Lysine
    StaticMetaboliteEntry {
        bmse_id: "bmse000039",
        name: "L-Lysine",
        formula: "C6H14N2O2",
        smiles: "NCCCC[C@H](N)C(=O)O",
        ph: 7.4,
        peaks_1h: &[
            p1h!("gamma-CH2", 1.468, Multiplet, [7.5], 2),
            p1h!("delta-CH2", 1.720, Multiplet, [7.5], 2),
            p1h!("beta-CH2", 1.900, Multiplet, [7.2], 2),
            p1h!("epsilon-CH2", 3.022, Triplet, [7.5], 2),
            p1h!("alpha-CH", 3.752, Triplet, [6.1], 1),
        ],
        shifts_13c: &[
            c13!("Cgamma", 24.2, 1),
            c13!("Cdelta", 29.2, 1),
            c13!("Cbeta", 32.7, 1),
            c13!("Cepsilon", 41.9, 1),
            c13!("Calpha", 57.3, 1),
            c13!("C(COO-)", 177.3, 1),
        ],
    },
    // 34. L-Arginine
    StaticMetaboliteEntry {
        bmse_id: "bmse000029",
        name: "L-Arginine",
        formula: "C6H14N4O2",
        smiles: "N=C(N)NCCC[C@H](N)C(=O)O",
        ph: 7.4,
        peaks_1h: &[
            p1h!("gamma-CH2", 1.685, Multiplet, [7.2], 2),
            p1h!("beta-CH2", 1.908, Multiplet, [7.0], 2),
            p1h!("delta-CH2", 3.240, Triplet, [6.9], 2),
            p1h!("alpha-CH", 3.762, Triplet, [6.1], 1),
        ],
        shifts_13c: &[
            c13!("Cgamma", 26.7, 1),
            c13!("Cbeta", 30.4, 1),
            c13!("Cdelta", 43.3, 1),
            c13!("Calpha", 57.1, 1),
            c13!("Czeta", 159.5, 1),
            c13!("C(COO-)", 177.1, 1),
        ],
    },
    // 35. L-Aspartate
    StaticMetaboliteEntry {
        bmse_id: "bmse000031",
        name: "L-Aspartate",
        formula: "C4H7NO4",
        smiles: "N[C@@H](CC(=O)O)C(=O)O",
        ph: 7.4,
        peaks_1h: &[
            p1h!("beta-CH2a", 2.672, DoubletOfDoublets, [17.5, 8.8], 1),
            p1h!("beta-CH2b", 2.805, DoubletOfDoublets, [17.5, 3.7], 1),
            p1h!("alpha-CH", 3.895, DoubletOfDoublets, [8.8, 3.7], 1),
        ],
        shifts_13c: &[
            c13!("Cbeta", 39.4, 1),
            c13!("Calpha", 55.0, 1),
            c13!("C(COO-)", 177.2, 1),
            c13!("Cgamma(COO-)", 180.5, 1),
        ],
    },
    // 36. L-Asparagine
    StaticMetaboliteEntry {
        bmse_id: "bmse000030",
        name: "L-Asparagine",
        formula: "C4H8N2O3",
        smiles: "NC(=O)C[C@H](N)C(=O)O",
        ph: 7.4,
        peaks_1h: &[
            p1h!("beta-CH2a", 2.845, DoubletOfDoublets, [16.9, 7.6], 1),
            p1h!("beta-CH2b", 2.948, DoubletOfDoublets, [16.9, 4.3], 1),
            p1h!("alpha-CH", 3.998, DoubletOfDoublets, [7.6, 4.3], 1),
        ],
        shifts_13c: &[
            c13!("Cbeta", 37.4, 1),
            c13!("Calpha", 54.1, 1),
            c13!("C(COO-)", 176.2, 1),
            c13!("Cgamma(CONH2)", 177.3, 1),
        ],
    },
    // 37. L-Methionine
    StaticMetaboliteEntry {
        bmse_id: "bmse000042",
        name: "L-Methionine",
        formula: "C5H11NO2S",
        smiles: "CSCC[C@H](N)C(=O)O",
        ph: 7.4,
        peaks_1h: &[
            p1h!("S-CH3", 2.130, Singlet, [], 3),
            p1h!("beta-CH2", 2.150, Multiplet, [7.2], 2),
            p1h!("gamma-CH2", 2.635, Triplet, [7.5], 2),
            p1h!("alpha-CH", 3.855, DoubletOfDoublets, [7.0, 5.5], 1),
        ],
        shifts_13c: &[
            c13!("Cepsilon(CH3)", 16.8, 1),
            c13!("Cgamma", 31.7, 1),
            c13!("Cbeta", 32.6, 1),
            c13!("Calpha", 56.8, 1),
            c13!("C(COO-)", 177.0, 1),
        ],
    },
    // 38. L-Tryptophan
    StaticMetaboliteEntry {
        bmse_id: "bmse000050",
        name: "L-Tryptophan",
        formula: "C11H12N2O2",
        smiles: "N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O",
        ph: 7.4,
        peaks_1h: &[
            p1h!("beta-CH2a", 3.300, DoubletOfDoublets, [15.4, 8.1], 1),
            p1h!("beta-CH2b", 3.478, DoubletOfDoublets, [15.4, 4.8], 1),
            p1h!("alpha-CH", 4.052, DoubletOfDoublets, [8.1, 4.8], 1),
            p1h!("H5", 7.195, Triplet, [7.5], 1),
            p1h!("H6", 7.278, Triplet, [7.5], 1),
            p1h!("H2", 7.318, Singlet, [], 1),
            p1h!("H7", 7.535, Doublet, [8.1], 1),
            p1h!("H4", 7.728, Doublet, [7.9], 1),
        ],
        shifts_13c: &[
            c13!("Cbeta", 29.1, 1),
            c13!("Calpha", 57.8, 1),
            c13!("C3", 110.2, 1),
            c13!("C7", 114.6, 1),
            c13!("C4", 121.1, 1),
            c13!("C5", 122.1, 1),
            c13!("C6", 124.8, 1),
            c13!("C2", 127.7, 1),
            c13!("C3a", 129.4, 1),
            c13!("C7a", 139.0, 1),
            c13!("C(COO-)", 177.1, 1),
        ],
    },
    // 39. Hippurate
    StaticMetaboliteEntry {
        bmse_id: "bmse000222",
        name: "Hippurate",
        formula: "C9H9NO3",
        smiles: "O=C(O)CNC(=O)c1ccccc1",
        ph: 7.4,
        peaks_1h: &[
            p1h!("CH2", 3.968, Doublet, [5.8], 2),
            p1h!("meta-CH", 7.548, Triplet, [7.6], 2),
            p1h!("para-CH", 7.635, Triplet, [7.4], 1),
            p1h!("ortho-CH", 7.832, Doublet, [7.6], 2),
        ],
        shifts_13c: &[
            c13!("CH2", 46.7, 1),
            c13!("Cortho", 129.6, 2),
            c13!("Cmeta", 131.4, 2),
            c13!("Cpara", 134.5, 1),
            c13!("Cipso", 136.3, 1),
            c13!("CONH", 173.4, 1),
            c13!("COO-", 179.3, 1),
        ],
    },
    // 40. 3-Hydroxybutyrate
    StaticMetaboliteEntry {
        bmse_id: "bmse000143",
        name: "3-Hydroxybutyrate",
        formula: "C4H8O3",
        smiles: "C[C@@H](O)CC(=O)O",
        ph: 7.4,
        peaks_1h: &[
            p1h!("gamma-CH3", 1.196, Doublet, [6.3], 3),
            p1h!("alpha-CH2a", 2.305, DoubletOfDoublets, [14.5, 6.2], 1),
            p1h!("alpha-CH2b", 2.405, DoubletOfDoublets, [14.5, 7.3], 1),
            p1h!("beta-CH", 4.150, Multiplet, [6.5], 1),
        ],
        shifts_13c: &[
            c13!("C4(CH3)", 24.5, 1),
            c13!("C2(CH2)", 49.2, 1),
            c13!("C3(CH)", 68.5, 1),
            c13!("C1(COO-)", 183.1, 1),
        ],
    },
    // 41. Acetoacetate
    StaticMetaboliteEntry {
        bmse_id: "bmse000370",
        name: "Acetoacetate",
        formula: "C4H6O3",
        smiles: "CC(=O)CC(=O)O",
        ph: 7.4,
        peaks_1h: &[
            p1h!("CH3", 2.278, Singlet, [], 3),
            p1h!("CH2", 3.442, Singlet, [], 2),
        ],
        shifts_13c: &[
            c13!("C4(CH3)", 32.3, 1),
            c13!("C2(CH2)", 56.2, 1),
            c13!("C1(COO-)", 177.6, 1),
            c13!("C3(C=O)", 212.9, 1),
        ],
    },
    // 42. Dimethylamine
    StaticMetaboliteEntry {
        bmse_id: "bmse000277",
        name: "Dimethylamine",
        formula: "C2H7N",
        smiles: "CNC",
        ph: 7.4,
        peaks_1h: &[p1h!("2xCH3", 2.720, Singlet, [], 6)],
        shifts_13c: &[c13!("2xCH3", 37.3, 2)],
    },
    // 43. Glycerol
    StaticMetaboliteEntry {
        bmse_id: "bmse000090",
        name: "Glycerol",
        formula: "C3H8O3",
        smiles: "OCC(O)CO",
        ph: 7.0,
        peaks_1h: &[
            p1h!("CH2a", 3.558, DoubletOfDoublets, [11.7, 6.5], 2),
            p1h!("CH2b", 3.652, DoubletOfDoublets, [11.7, 4.4], 2),
            p1h!("CH", 3.782, Multiplet, [6.5, 4.4], 1),
        ],
        shifts_13c: &[c13!("C1/C3(CH2)", 65.3, 2), c13!("C2(CH)", 74.8, 1)],
    },
    // 44. Phosphocholine
    StaticMetaboliteEntry {
        bmse_id: "bmse000098",
        name: "Phosphocholine",
        formula: "C5H15NO4P+",
        smiles: "C[N+](C)(C)CCOP(=O)(O)O",
        ph: 7.4,
        peaks_1h: &[
            p1h!("N(CH3)3", 3.218, Singlet, [], 9),
            p1h!("CH2-N", 3.595, Multiplet, [5.0], 2),
            p1h!("CH2-O", 4.165, Multiplet, [5.0], 2),
        ],
        shifts_13c: &[
            c13!("N(CH3)3", 56.7, 3),
            c13!("CH2-O", 61.1, 1),
            c13!("CH2-N", 69.1, 1),
        ],
    },
    // 45. L-Cysteine
    StaticMetaboliteEntry {
        bmse_id: "bmse000037",
        name: "L-Cysteine",
        formula: "C3H7NO2S",
        smiles: "N[C@@H](CS)C(=O)O",
        ph: 7.4,
        peaks_1h: &[
            p1h!("beta-CH2a", 3.015, DoubletOfDoublets, [14.8, 5.5], 1),
            p1h!("beta-CH2b", 3.105, DoubletOfDoublets, [14.8, 4.2], 1),
            p1h!("alpha-CH", 3.972, DoubletOfDoublets, [5.5, 4.2], 1),
        ],
        shifts_13c: &[
            c13!("Cbeta", 27.6, 1),
            c13!("Calpha", 58.6, 1),
            c13!("C(COO-)", 175.1, 1),
        ],
    },
    // 46. Uracil
    StaticMetaboliteEntry {
        bmse_id: "bmse000161",
        name: "Uracil",
        formula: "C4H4N2O2",
        smiles: "O=c1cc[nH]c(=O)[nH]1",
        ph: 7.4,
        peaks_1h: &[
            p1h!("H5", 5.802, Doublet, [7.7], 1),
            p1h!("H6", 7.536, Doublet, [7.7], 1),
        ],
        shifts_13c: &[
            c13!("C5", 103.5, 1),
            c13!("C6", 145.7, 1),
            c13!("C2", 155.1, 1),
            c13!("C4", 169.3, 1),
        ],
    },
];

/// Return the number of canonical aqueous metabolites in the built-in CC0 `BMRB-BMSE` index.
pub fn offline_metabolite_count() -> usize {
    OFFLINE_BMRB_METABOLITES.len()
}

/// Return all canonical aqueous metabolites in the built-in CC0 `BMRB-BMSE` compact index.
pub fn offline_metabolite_database() -> Vec<MetaboliteEntry> {
    OFFLINE_BMRB_METABOLITES
        .iter()
        .map(StaticMetaboliteEntry::to_entry)
        .collect()
}

/// Look up an offline metabolite entry by case-insensitive `bmse_id` or common name.
pub fn lookup_offline_metabolite(query: &str) -> Option<MetaboliteEntry> {
    let q = query.trim().to_ascii_lowercase();
    OFFLINE_BMRB_METABOLITES
        .iter()
        .find(|e| e.bmse_id.eq_ignore_ascii_case(&q) || e.name.to_ascii_lowercase() == q)
        .map(StaticMetaboliteEntry::to_entry)
}

/// Extract a coherent `STOCSY` peak cluster (`r >= min_r`, default `0.85`) as [`QueryPeak`]s.
pub fn extract_stocsy_cluster_query_peaks(
    buckets: &[SpectralBucket],
    stocsy: &StocsyResult,
    min_r: f64,
) -> Vec<QueryPeak> {
    let threshold = if min_r.is_finite() {
        min_r.clamp(0.0, 1.0)
    } else {
        DEFAULT_STOCSY_CLUSTER_MIN_R
    };
    let mut peaks = Vec::new();
    for (idx, bucket) in buckets.iter().enumerate() {
        let r = stocsy.bucket_correlation.get(idx).copied().unwrap_or(0.0);
        if r.is_finite() && (r >= threshold || idx == stocsy.driver_bucket_index) {
            peaks.push(QueryPeak::from_stocsy_cluster(
                bucket.center_ppm,
                r.clamp(0.0, 1.0),
            ));
        }
    }
    peaks
}

// ============================================================================
// Fast (< 5 ms) Offline Matcher (Peak Lists + STOCSY Clusters)
// ============================================================================

fn score_mult_and_j(
    q_mult: Option<PeakMultiplicity>,
    q_j: &[f64],
    ref_mult: PeakMultiplicity,
    ref_j: &[f64],
) -> (f64, bool) {
    let has_m = q_mult.is_some();
    let has_j = !q_j.is_empty();
    if !has_m && !has_j {
        return (1.0, false);
    }
    let m_score = q_mult.map(|qm| qm.compatibility(ref_mult)).unwrap_or(1.0);
    let j_score = if has_j {
        if ref_j.is_empty() {
            // Query reports scalar splitting while reference is a singlet.
            0.25
        } else {
            let mut sum = 0.0_f64;
            for &qj in q_j {
                let min_diff = ref_j
                    .iter()
                    .map(|&rj| (qj - rj).abs())
                    .fold(f64::INFINITY, f64::min);
                sum += (-0.5 * (min_diff / 1.5).powi(2)).exp();
            }
            (sum / (q_j.len() as f64)).clamp(0.0, 1.0)
        }
    } else {
        1.0
    };
    let combined = match (has_m, has_j) {
        (true, true) => 0.55 * m_score + 0.45 * j_score,
        (true, false) => m_score,
        (false, true) => j_score,
        (false, false) => 1.0,
    };
    (combined.clamp(0.0, 1.0), true)
}

fn filter_active_query_peaks(query_peaks: &[QueryPeak]) -> Vec<&QueryPeak> {
    let valid: Vec<&QueryPeak> = query_peaks.iter().filter(|q| q.ppm.is_finite()).collect();
    if valid.is_empty() {
        return valid;
    }
    let is_stocsy_mode = valid.iter().any(|q| q.stocsy_r.is_some());
    if !is_stocsy_mode {
        return valid;
    }
    let high_r: Vec<&QueryPeak> = valid
        .iter()
        .copied()
        .filter(|q| {
            q.stocsy_r
                .map(|r| r.is_finite() && r >= DEFAULT_STOCSY_CLUSTER_MIN_R)
                .unwrap_or(true)
        })
        .collect();
    if !high_r.is_empty() {
        high_r
    } else {
        valid
            .into_iter()
            .filter(|q| q.stocsy_r.map(|r| r.is_finite() && r > 0.0).unwrap_or(true))
            .collect()
    }
}

/// Fast `< 5 ms` offline matcher against the built-in CC0 `BMRB-BMSE` metabolite index.
///
/// Supports both full picked 1D peak lists (`1H` or `13C`) and `STOCSY` correlated peak clusters
/// (`r >= 0.85`). Scores chemical-shift recall, query precision, Gaussian shift proximity, and
/// multiplicity/`J`-coupling consistency into a composite score in `[0.0, 1.0]`.
pub fn query_offline_metabolites(
    query_peaks: &[QueryPeak],
    nucleus: MetaboliteNucleus,
    tol_ppm: f64,
    top_k: usize,
) -> Vec<MetaboliteMatch> {
    if top_k == 0 || !tol_ppm.is_finite() || tol_ppm <= 0.0 {
        return Vec::new();
    }
    let active_q = filter_active_query_peaks(query_peaks);
    if active_q.is_empty() {
        return Vec::new();
    }

    let mut matches = Vec::with_capacity(OFFLINE_BMRB_METABOLITES.len());

    for entry in OFFLINE_BMRB_METABOLITES {
        let m = match nucleus {
            MetaboliteNucleus::H1 => score_entry_h1(entry, &active_q, tol_ppm),
            MetaboliteNucleus::C13 => score_entry_c13(entry, &active_q, tol_ppm),
        };
        if let Some(hit) = m {
            matches.push(hit);
        }
    }

    matches.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| b.matched_ref_peaks.cmp(&a.matched_ref_peaks))
            .then_with(|| a.mean_abs_error_ppm.total_cmp(&b.mean_abs_error_ppm))
            .then_with(|| a.entry.bmse_id.cmp(&b.entry.bmse_id))
    });
    matches.truncate(top_k);
    matches
}

fn score_entry_h1(
    entry: &StaticMetaboliteEntry,
    active_q: &[&QueryPeak],
    tol_ppm: f64,
) -> Option<MetaboliteMatch> {
    let n_ref = entry.peaks_1h.len();
    if n_ref == 0 {
        return None;
    }

    // Collect all candidate (ref_idx, q_idx, abs_diff, mult_j, used_mj) within tol_ppm.
    let mut pairs: Vec<(usize, usize, f64, f64, bool)> = Vec::new();
    for (ri, rp) in entry.peaks_1h.iter().enumerate() {
        for (qi, qp) in active_q.iter().enumerate() {
            let diff = (rp.ppm - qp.ppm).abs();
            if diff <= tol_ppm {
                let (mj, used_mj) =
                    score_mult_and_j(qp.multiplicity, &qp.j_hz, rp.multiplicity, rp.j_hz);
                pairs.push((ri, qi, diff, mj, used_mj));
            }
        }
    }
    if pairs.is_empty() {
        return None;
    }

    // Greedy 1-to-1 assignment prioritizing smallest shift error (modulated gently by mult/J).
    pairs.sort_by(|a, b| {
        let cost_a = a.2 / tol_ppm + 0.35 * (1.0 - a.3);
        let cost_b = b.2 / tol_ppm + 0.35 * (1.0 - b.3);
        cost_a
            .total_cmp(&cost_b)
            .then_with(|| a.2.total_cmp(&b.2))
            .then_with(|| a.0.cmp(&b.0))
    });

    let mut ref_used = vec![false; n_ref];
    let mut q_used = vec![false; active_q.len()];
    let mut matched_pairs = Vec::new();
    let mut sum_shift = 0.0_f64;
    let mut sum_mj = 0.0_f64;
    let mut sum_err = 0.0_f64;
    let mut sum_stocsy_r = 0.0_f64;
    let mut any_mj_checked = false;

    let is_stocsy_mode = active_q.iter().any(|q| q.stocsy_r.is_some());
    let max_stocsy_r = active_q
        .iter()
        .filter_map(|q| q.stocsy_r.filter(|r| r.is_finite()))
        .fold(0.0_f64, f64::max);
    let mut matched_driver = false;

    for &(ri, qi, diff, mj, used_mj) in &pairs {
        if ref_used[ri] || q_used[qi] {
            continue;
        }
        ref_used[ri] = true;
        q_used[qi] = true;
        let rp = &entry.peaks_1h[ri];
        let qp = active_q[qi];
        let s_prox = (-1.5 * (diff / tol_ppm).powi(2)).exp();
        sum_shift += s_prox;
        sum_mj += mj;
        sum_err += diff;
        if let Some(r) = qp.stocsy_r.filter(|v| v.is_finite()) {
            let rc = r.clamp(0.0, 1.0);
            sum_stocsy_r += rc;
            if rc >= max_stocsy_r - 0.02 {
                matched_driver = true;
            }
        }
        if used_mj {
            any_mj_checked = true;
        }
        matched_pairs.push(MatchedPeakPair {
            assignment: rp.assignment.to_string(),
            ref_ppm: rp.ppm,
            query_ppm: qp.ppm,
            delta_ppm: qp.ppm - rp.ppm,
            mult_j_consistency: mj,
        });
    }

    // Secondary pass: allow closely overlapping doublets/multiplets of the SAME metabolite
    // (e.g. Leucine 0.952/0.964 coalesced into a single AIB bucket) to also claim an already-used
    // query peak if within 0.75 * tol_ppm, without inflating matched_query_count.
    for &(ri, qi, diff, mj, used_mj) in &pairs {
        if ref_used[ri] || diff > 0.75 * tol_ppm {
            continue;
        }
        ref_used[ri] = true;
        let rp = &entry.peaks_1h[ri];
        let qp = active_q[qi];
        let s_prox = (-1.5 * (diff / tol_ppm).powi(2)).exp();
        sum_shift += s_prox;
        sum_mj += mj;
        sum_err += diff;
        if let Some(r) = qp.stocsy_r.filter(|v| v.is_finite()) {
            let rc = r.clamp(0.0, 1.0);
            sum_stocsy_r += rc;
            if rc >= max_stocsy_r - 0.02 {
                matched_driver = true;
            }
        }
        if used_mj {
            any_mj_checked = true;
        }
        matched_pairs.push(MatchedPeakPair {
            assignment: rp.assignment.to_string(),
            ref_ppm: rp.ppm,
            query_ppm: qp.ppm,
            delta_ppm: qp.ppm - rp.ppm,
            mult_j_consistency: mj,
        });
    }

    let matched_ref = matched_pairs.len();
    if matched_ref == 0 {
        return None;
    }
    let matched_q = q_used.iter().filter(|&&u| u).count();

    let recall = (matched_ref as f64) / (n_ref as f64);
    let precision = (matched_q as f64) / (active_q.len() as f64);
    let shift_score = (sum_shift / (matched_ref as f64)).clamp(0.0, 1.0);
    let mult_j_score = (sum_mj / (matched_ref as f64)).clamp(0.0, 1.0);
    let mean_abs_error_ppm = sum_err / (matched_ref as f64);

    let mut composite = if any_mj_checked {
        0.45 * recall + 0.20 * precision + 0.20 * shift_score + 0.15 * mult_j_score
    } else if is_stocsy_mode {
        let mean_r = (sum_stocsy_r / (matched_ref as f64)).clamp(0.0, 1.0);
        0.42 * recall + 0.16 * precision + 0.24 * shift_score + 0.18 * mean_r
    } else {
        0.50 * recall + 0.22 * precision + 0.28 * shift_score
    };
    if is_stocsy_mode && max_stocsy_r >= DEFAULT_STOCSY_CLUSTER_MIN_R && !matched_driver {
        // Intermolecular pathway correlation (does not explain the STOCSY driver resonance).
        composite *= 0.65;
    }

    Some(MetaboliteMatch {
        entry: entry.to_entry(),
        nucleus: MetaboliteNucleus::H1,
        score: composite.clamp(0.0, 1.0),
        recall,
        precision,
        shift_score,
        mult_j_score,
        matched_ref_peaks: matched_ref,
        total_ref_peaks: n_ref,
        mean_abs_error_ppm,
        matched_pairs,
    })
}

fn score_entry_c13(
    entry: &StaticMetaboliteEntry,
    active_q: &[&QueryPeak],
    tol_ppm: f64,
) -> Option<MetaboliteMatch> {
    let n_ref = entry.shifts_13c.len();
    if n_ref == 0 {
        return None;
    }

    let mut pairs: Vec<(usize, usize, f64)> = Vec::new();
    for (ri, rc) in entry.shifts_13c.iter().enumerate() {
        for (qi, qp) in active_q.iter().enumerate() {
            let diff = (rc.ppm - qp.ppm).abs();
            if diff <= tol_ppm {
                pairs.push((ri, qi, diff));
            }
        }
    }
    if pairs.is_empty() {
        return None;
    }

    pairs.sort_by(|a, b| a.2.total_cmp(&b.2).then_with(|| a.0.cmp(&b.0)));

    let mut ref_used = vec![false; n_ref];
    let mut q_used = vec![false; active_q.len()];
    let mut matched_pairs = Vec::new();
    let mut sum_shift = 0.0_f64;
    let mut sum_err = 0.0_f64;
    let mut sum_stocsy_r = 0.0_f64;
    let is_stocsy_mode = active_q.iter().any(|q| q.stocsy_r.is_some());
    let max_stocsy_r = active_q
        .iter()
        .filter_map(|q| q.stocsy_r.filter(|r| r.is_finite()))
        .fold(0.0_f64, f64::max);
    let mut matched_driver = false;

    for &(ri, qi, diff) in &pairs {
        if ref_used[ri] || q_used[qi] {
            continue;
        }
        ref_used[ri] = true;
        q_used[qi] = true;
        let rc = &entry.shifts_13c[ri];
        let qp = active_q[qi];
        sum_shift += (-1.5 * (diff / tol_ppm).powi(2)).exp();
        sum_err += diff;
        if let Some(r) = qp.stocsy_r.filter(|v| v.is_finite()) {
            let rc_clamped = r.clamp(0.0, 1.0);
            sum_stocsy_r += rc_clamped;
            if rc_clamped >= max_stocsy_r - 0.02 {
                matched_driver = true;
            }
        }
        matched_pairs.push(MatchedPeakPair {
            assignment: rc.assignment.to_string(),
            ref_ppm: rc.ppm,
            query_ppm: qp.ppm,
            delta_ppm: qp.ppm - rc.ppm,
            mult_j_consistency: 1.0,
        });
    }

    let matched_ref = matched_pairs.len();
    if matched_ref == 0 {
        return None;
    }
    let matched_q = q_used.iter().filter(|&&u| u).count();

    let recall = (matched_ref as f64) / (n_ref as f64);
    let precision = (matched_q as f64) / (active_q.len() as f64);
    let shift_score = (sum_shift / (matched_ref as f64)).clamp(0.0, 1.0);
    let mean_abs_error_ppm = sum_err / (matched_ref as f64);
    let mut composite = if is_stocsy_mode {
        let mean_r = (sum_stocsy_r / (matched_ref as f64)).clamp(0.0, 1.0);
        0.42 * recall + 0.16 * precision + 0.24 * shift_score + 0.18 * mean_r
    } else {
        0.50 * recall + 0.22 * precision + 0.28 * shift_score
    };
    if is_stocsy_mode && max_stocsy_r >= DEFAULT_STOCSY_CLUSTER_MIN_R && !matched_driver {
        composite *= 0.65;
    }
    let composite = composite.clamp(0.0, 1.0);

    Some(MetaboliteMatch {
        entry: entry.to_entry(),
        nucleus: MetaboliteNucleus::C13,
        score: composite,
        recall,
        precision,
        shift_score,
        mult_j_score: 1.0,
        matched_ref_peaks: matched_ref,
        total_ref_peaks: n_ref,
        mean_abs_error_ppm,
        matched_pairs,
    })
}

// ============================================================================
// 1-Click 1D Reference Spectrum Synthesizer (Sticks + Lorentzian Lineshape)
// ============================================================================

fn expand_peak_1h_sticks(peak: &MetabolitePeak1H, obs_mhz: f64) -> Vec<(f64, f64)> {
    let c0 = peak.ppm;
    let total_h = (peak.n_protons.max(1)) as f64;
    let mhz = obs_mhz.max(60.0);
    let j0 = peak.j_hz.first().copied().unwrap_or(0.0) / mhz;
    let j1 = peak.j_hz.get(1).copied().unwrap_or(j0 * mhz) / mhz;

    match peak.multiplicity {
        PeakMultiplicity::Singlet => vec![(c0, total_h)],
        PeakMultiplicity::Doublet => vec![
            (c0 - 0.5 * j0, 0.5 * total_h),
            (c0 + 0.5 * j0, 0.5 * total_h),
        ],
        PeakMultiplicity::Triplet => vec![
            (c0 - j0, 0.25 * total_h),
            (c0, 0.50 * total_h),
            (c0 + j0, 0.25 * total_h),
        ],
        PeakMultiplicity::Quartet => vec![
            (c0 - 1.5 * j0, 0.125 * total_h),
            (c0 - 0.5 * j0, 0.375 * total_h),
            (c0 + 0.5 * j0, 0.375 * total_h),
            (c0 + 1.5 * j0, 0.125 * total_h),
        ],
        PeakMultiplicity::DoubletOfDoublets => vec![
            (c0 - 0.5 * (j0 + j1), 0.25 * total_h),
            (c0 - 0.5 * (j0 - j1), 0.25 * total_h),
            (c0 + 0.5 * (j0 - j1), 0.25 * total_h),
            (c0 + 0.5 * (j0 + j1), 0.25 * total_h),
        ],
        PeakMultiplicity::DoubletOfTriplets => {
            let mut out = Vec::with_capacity(6);
            for s0 in [-0.5, 0.5] {
                for (s1, w1) in [(-1.0, 0.25), (0.0, 0.50), (1.0, 0.25)] {
                    out.push((c0 + s0 * j0 + s1 * j1, 0.5 * w1 * total_h));
                }
            }
            out
        }
        PeakMultiplicity::Quintet => vec![
            (c0 - 2.0 * j0, 0.0625 * total_h),
            (c0 - j0, 0.25 * total_h),
            (c0, 0.375 * total_h),
            (c0 + j0, 0.25 * total_h),
            (c0 + 2.0 * j0, 0.0625 * total_h),
        ],
        PeakMultiplicity::Multiplet => {
            let step = if j0 > 1e-6 { 0.5 * j0 } else { 4.5 / mhz };
            vec![
                (c0 - 2.0 * step, 0.12 * total_h),
                (c0 - step, 0.22 * total_h),
                (c0, 0.32 * total_h),
                (c0 + step, 0.22 * total_h),
                (c0 + 2.0 * step, 0.12 * total_h),
            ]
        }
    }
}

/// Synthesize a normalized 1D reference spectrum (`ppm_grid`, Lorentzian `intensities`, and
/// multiplet `sticks`) for 1-click reference overlay onto 1D cohort plots.
pub fn synthesize_reference_spectrum_1d(
    entry: &MetaboliteEntry,
    nucleus: MetaboliteNucleus,
    mhz: f64,
    n_pts: usize,
) -> ReferenceSpectrum1D {
    let obs_mhz = if mhz.is_finite() && mhz > 10.0 {
        mhz
    } else {
        600.0
    };
    let n = n_pts.max(64);

    let (raw_sticks, default_lo, default_hi, gamma_ppm) = match nucleus {
        MetaboliteNucleus::H1 => {
            let mut sticks = Vec::new();
            for p in &entry.peaks_1h {
                if p.ppm.is_finite() {
                    sticks.extend(expand_peak_1h_sticks(p, obs_mhz));
                }
            }
            let hwhm_ppm = (0.85 / obs_mhz).clamp(0.0005, 0.015);
            (sticks, 0.0_f64, 10.0_f64, hwhm_ppm)
        }
        MetaboliteNucleus::C13 => {
            let sticks: Vec<(f64, f64)> = entry
                .shifts_13c
                .iter()
                .filter(|c| c.ppm.is_finite())
                .map(|c| (c.ppm, c.n_carbons.max(1) as f64))
                .collect();
            (sticks, 0.0_f64, 220.0_f64, 0.06_f64)
        }
    };

    let mut min_ppm = default_lo;
    let mut max_ppm = default_hi;
    for &(ppm, _) in &raw_sticks {
        min_ppm = min_ppm.min(ppm - 0.5);
        max_ppm = max_ppm.max(ppm + 0.5);
    }
    let step = (max_ppm - min_ppm) / ((n - 1) as f64);
    let mut ppm_grid = Vec::with_capacity(n);
    for i in 0..n {
        ppm_grid.push(max_ppm - (i as f64) * step);
    }

    let mut intensities = vec![0.0_f64; n];
    let g2 = gamma_ppm * gamma_ppm;
    let window = (gamma_ppm * 35.0).max(step * 4.0);

    for &(center, amp) in &raw_sticks {
        let hi_i = (((max_ppm - (center + window)) / step).floor() as isize).max(0) as usize;
        let lo_i = (((max_ppm - (center - window)) / step).ceil() as isize)
            .min((n - 1) as isize)
            .max(0) as usize;
        for i in hi_i..=lo_i {
            let d = ppm_grid[i] - center;
            intensities[i] += amp * g2 / (d * d + g2);
        }
    }

    let max_y = intensities.iter().copied().fold(0.0_f64, f64::max);
    if max_y > 1e-18 {
        for y in &mut intensities {
            *y /= max_y;
        }
    }

    let max_stick = raw_sticks.iter().map(|&(_, h)| h).fold(0.0_f64, f64::max);
    let sticks = if max_stick > 1e-18 {
        raw_sticks
            .into_iter()
            .map(|(p, h)| (p, (h / max_stick).clamp(0.0, 1.0)))
            .collect()
    } else {
        raw_sticks
    };

    ReferenceSpectrum1D {
        bmse_id: entry.bmse_id.clone(),
        name: entry.name.clone(),
        nucleus,
        obs_mhz,
        ppm_grid,
        intensities,
        sticks,
    }
}

// ============================================================================
// Opt-In Desktop Online Query: BMRB v2 REST API + NP-MRD (Strict Allowlist)
// ============================================================================

/// Pluggable HTTPS GET transport for desktop online queries (production uses `curl`, unit tests
/// inject deterministic mocks without touching the network).
#[cfg(not(target_arch = "wasm32"))]
pub trait OnlineMetaboliteTransport {
    fn get_https_json(&mut self, url: &str) -> Result<String, String>;
}

/// Production desktop HTTPS client invoking `curl` with strict host allowlist, `5 s` connect/read
/// timeout, no cross-host redirects, and a hard `4 MiB` response byte cap.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Default)]
pub struct CurlMetaboliteTransport;

#[cfg(not(target_arch = "wasm32"))]
impl OnlineMetaboliteTransport for CurlMetaboliteTransport {
    fn get_https_json(&mut self, url: &str) -> Result<String, String> {
        use std::io::Read;
        use std::process::{Command, Stdio};

        let _host = validate_metabolite_online_url(url)?;
        let timeout_str = ONLINE_QUERY_TIMEOUT_SECS.to_string();
        let max_bytes_str = MAX_ONLINE_RESPONSE_BYTES.to_string();

        let mut child = Command::new("curl")
            .args([
                "-fsS",
                "--proto",
                "=https",
                "--connect-timeout",
                &timeout_str,
                "--max-time",
                &timeout_str,
                "--max-filesize",
                &max_bytes_str,
                "-H",
                "Accept: application/json",
                "-A",
                "ferric-metabolite-db",
                url,
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("failed to invoke curl for {url}: {e}"))?;

        let mut body = Vec::new();
        if let Some(stdout) = child.stdout.take() {
            let mut limited = stdout.take((MAX_ONLINE_RESPONSE_BYTES + 1) as u64);
            limited
                .read_to_end(&mut body)
                .map_err(|e| format!("failed reading response from {url}: {e}"))?;
        }
        if body.len() > MAX_ONLINE_RESPONSE_BYTES {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!(
                "online response from {url} exceeded hard {MAX_ONLINE_RESPONSE_BYTES}-byte (4 MiB) cap"
            ));
        }

        let mut stderr_buf = Vec::new();
        if let Some(stderr) = child.stderr.take() {
            let mut limited_err = stderr.take(16 * 1024);
            let _ = limited_err.read_to_end(&mut stderr_buf);
        }

        let status = child
            .wait()
            .map_err(|e| format!("failed waiting for curl process on {url}: {e}"))?;
        if !status.success() {
            let err_txt = String::from_utf8_lossy(&stderr_buf);
            return Err(format!(
                "online metabolite query failed for {url} (exit {status}): {}",
                err_txt.trim()
            ));
        }

        String::from_utf8(body)
            .map_err(|e| format!("online response from {url} was not valid UTF-8: {e}"))
    }
}

/// Build the strictly allowlisted `BMRB v2` chemical-shift search URL for a query peak.
pub fn build_bmrb_v2_shift_search_url(
    ppm: f64,
    nucleus: MetaboliteNucleus,
    tol_ppm: f64,
) -> Result<String, String> {
    if !ppm.is_finite() || !tol_ppm.is_finite() || tol_ppm <= 0.0 {
        return Err("invalid non-finite ppm or tolerance for BMRB v2 query".to_string());
    }
    let atom_type = match nucleus {
        MetaboliteNucleus::H1 => "H",
        MetaboliteNucleus::C13 => "C",
    };
    let url = format!(
        "{BMRB_V2_API_BASE}/search/chemical_shifts?shift_val={ppm:.4}&threshold={tol_ppm:.4}&atom_type={atom_type}&database=metabolomics"
    );
    validate_metabolite_online_url(&url)?;
    Ok(url)
}

/// Build the strictly allowlisted `NP-MRD` shift search URL for a set of query peaks.
pub fn build_npmrd_shift_search_url(
    query_peaks: &[QueryPeak],
    nucleus: MetaboliteNucleus,
    tol_ppm: f64,
) -> Result<String, String> {
    if !tol_ppm.is_finite() || tol_ppm <= 0.0 {
        return Err("invalid non-finite tolerance for NP-MRD query".to_string());
    }
    let shifts: Vec<String> = query_peaks
        .iter()
        .filter(|q| q.ppm.is_finite())
        .take(24)
        .map(|q| format!("{:.4}", q.ppm))
        .collect();
    if shifts.is_empty() {
        return Err("at least one finite query peak is required for NP-MRD query".to_string());
    }
    let nuc = nucleus.as_str();
    let url = format!(
        "{NPMRD_API_BASE}/api/v1/spectra/search?nucleus={nuc}&shifts={}&tolerance={tol_ppm:.4}",
        shifts.join(",")
    );
    validate_metabolite_online_url(&url)?;
    Ok(url)
}

/// Parse `BMRB v2` REST API JSON (`/v2/search/chemical_shifts`) into [`MetaboliteMatch`] hits.
pub fn parse_bmrb_v2_response(
    json_text: &str,
    query_peaks: &[QueryPeak],
    nucleus: MetaboliteNucleus,
    tol_ppm: f64,
) -> Result<Vec<MetaboliteMatch>, String> {
    use std::collections::BTreeMap;

    let root: serde_json::Value = serde_json::from_str(json_text)
        .map_err(|e| format!("failed to parse BMRB v2 JSON response: {e}"))?;

    // Collect (entry_id -> (name, matched shifts))
    let mut grouped: BTreeMap<String, (String, Vec<(String, f64)>)> = BTreeMap::new();

    if let (Some(cols), Some(rows)) = (
        root.get("columns").and_then(|v| v.as_array()),
        root.get("data").and_then(|v| v.as_array()),
    ) {
        let col_names: Vec<String> = cols
            .iter()
            .filter_map(|c| c.as_str().map(|s| s.to_ascii_lowercase()))
            .collect();
        let find_col = |needles: &[&str]| -> Option<usize> {
            col_names
                .iter()
                .position(|c| needles.iter().any(|n| c == n || c.ends_with(n)))
        };
        let id_col = find_col(&["entry_id", "bmse_id"]);
        let name_col = find_col(&["comp_id", "name", "compound_name"]);
        let val_col = find_col(&["val", "shift_val", "ppm"]);
        let atom_col = find_col(&["atom_id", "assignment"]);

        if let (Some(ic), Some(vc)) = (id_col, val_col) {
            for row in rows {
                let Some(arr) = row.as_array() else {
                    continue;
                };
                let Some(entry_id) = arr.get(ic).and_then(|v| v.as_str()) else {
                    continue;
                };
                let shift_val = arr.get(vc).and_then(|v| {
                    v.as_f64()
                        .or_else(|| v.as_str().and_then(|s| s.parse::<f64>().ok()))
                });
                let Some(ppm) = shift_val.filter(|v| v.is_finite()) else {
                    continue;
                };
                let cname = name_col
                    .and_then(|nc| arr.get(nc))
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.trim().is_empty())
                    .unwrap_or(entry_id)
                    .to_string();
                let atom = atom_col
                    .and_then(|ac| arr.get(ac))
                    .and_then(|v| v.as_str())
                    .unwrap_or("H")
                    .to_string();
                let slot = grouped
                    .entry(entry_id.to_string())
                    .or_insert_with(|| (cname, Vec::new()));
                slot.1.push((atom, ppm));
            }
        }
    } else if let Some(arr) = root.as_array().or_else(|| {
        root.get("results")
            .or_else(|| root.get("data"))
            .and_then(|v| v.as_array())
    }) {
        for item in arr {
            let Some(obj) = item.as_object() else {
                continue;
            };
            let entry_id = obj
                .get("Entry_ID")
                .or_else(|| obj.get("entry_id"))
                .or_else(|| obj.get("bmse_id"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim();
            if entry_id.is_empty() {
                continue;
            }
            let cname = obj
                .get("Comp_ID")
                .or_else(|| obj.get("name"))
                .and_then(|v| v.as_str())
                .unwrap_or(entry_id)
                .to_string();
            let shift_val = obj.get("Val").or_else(|| obj.get("ppm")).and_then(|v| {
                v.as_f64()
                    .or_else(|| v.as_str().and_then(|s| s.parse::<f64>().ok()))
            });
            if let Some(ppm) = shift_val.filter(|v| v.is_finite()) {
                let atom = obj
                    .get("Atom_ID")
                    .or_else(|| obj.get("atom_id"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("peak")
                    .to_string();
                let slot = grouped
                    .entry(entry_id.to_string())
                    .or_insert_with(|| (cname, Vec::new()));
                slot.1.push((atom, ppm));
            }
        }
    }

    let active_q = filter_active_query_peaks(query_peaks);
    let q_denom = active_q.len().max(1) as f64;
    let mut hits = Vec::new();
    for (bmse_id, (name, shifts)) in grouped {
        let mut matched_pairs = Vec::new();
        let mut sum_err = 0.0_f64;
        let mut sum_shift = 0.0_f64;
        let mut peaks_1h = Vec::new();
        let mut shifts_13c = Vec::new();

        for (atom, rppm) in &shifts {
            match nucleus {
                MetaboliteNucleus::H1 => peaks_1h.push(MetabolitePeak1H {
                    assignment: atom.clone(),
                    ppm: *rppm,
                    multiplicity: PeakMultiplicity::Multiplet,
                    j_hz: Vec::new(),
                    n_protons: 1,
                }),
                MetaboliteNucleus::C13 => shifts_13c.push(MetaboliteShift13C {
                    assignment: atom.clone(),
                    ppm: *rppm,
                    n_carbons: 1,
                }),
            }
            if let Some(best_q) = active_q
                .iter()
                .min_by(|a, b| (a.ppm - rppm).abs().total_cmp(&(b.ppm - rppm).abs()))
            {
                let diff = (best_q.ppm - rppm).abs();
                if diff <= tol_ppm {
                    sum_err += diff;
                    sum_shift += (-1.5 * (diff / tol_ppm).powi(2)).exp();
                    matched_pairs.push(MatchedPeakPair {
                        assignment: atom.clone(),
                        ref_ppm: *rppm,
                        query_ppm: best_q.ppm,
                        delta_ppm: best_q.ppm - rppm,
                        mult_j_consistency: 1.0,
                    });
                }
            }
        }
        if matched_pairs.is_empty() {
            continue;
        }
        let n_matched = matched_pairs.len();
        let total_ref = shifts.len().max(n_matched);
        let recall = (n_matched as f64) / (total_ref as f64);
        let precision = ((n_matched as f64) / q_denom).clamp(0.0, 1.0);
        let shift_score = (sum_shift / (n_matched as f64)).clamp(0.0, 1.0);
        let mean_abs_error_ppm = sum_err / (n_matched as f64);
        let score = (0.45 * recall + 0.25 * precision + 0.30 * shift_score).clamp(0.0, 1.0);

        let offline_ref = lookup_offline_metabolite(&bmse_id);
        hits.push(MetaboliteMatch {
            entry: MetaboliteEntry {
                bmse_id,
                name: offline_ref.as_ref().map(|e| e.name.clone()).unwrap_or(name),
                formula: offline_ref
                    .as_ref()
                    .map(|e| e.formula.clone())
                    .unwrap_or_default(),
                smiles: offline_ref
                    .as_ref()
                    .map(|e| e.smiles.clone())
                    .unwrap_or_default(),
                solvent: "D2O".to_string(),
                ph: 7.4,
                peaks_1h,
                shifts_13c,
                source: MetaboliteSource::BmrbOnline,
            },
            nucleus,
            score,
            recall,
            precision,
            shift_score,
            mult_j_score: 1.0,
            matched_ref_peaks: n_matched,
            total_ref_peaks: total_ref,
            mean_abs_error_ppm,
            matched_pairs,
        });
    }
    Ok(hits)
}

/// Parse `NP-MRD` JSON search response into [`MetaboliteMatch`] hits.
pub fn parse_npmrd_response(
    json_text: &str,
    query_peaks: &[QueryPeak],
    nucleus: MetaboliteNucleus,
    tol_ppm: f64,
) -> Result<Vec<MetaboliteMatch>, String> {
    let root: serde_json::Value = serde_json::from_str(json_text)
        .map_err(|e| format!("failed to parse NP-MRD JSON response: {e}"))?;
    let items = root
        .as_array()
        .or_else(|| root.get("results").and_then(|v| v.as_array()))
        .or_else(|| root.get("data").and_then(|v| v.as_array()))
        .or_else(|| root.get("hits").and_then(|v| v.as_array()));

    let Some(arr) = items else {
        return Ok(Vec::new());
    };

    let active_q = filter_active_query_peaks(query_peaks);
    let q_denom = active_q.len().max(1) as f64;
    let mut hits = Vec::new();

    for item in arr {
        let Some(obj) = item.as_object() else {
            continue;
        };
        let acc = obj
            .get("np_mrd_id")
            .or_else(|| obj.get("accession"))
            .or_else(|| obj.get("id"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        if acc.is_empty() {
            continue;
        }
        let name = obj
            .get("name")
            .or_else(|| obj.get("compound_name"))
            .and_then(|v| v.as_str())
            .unwrap_or(&acc)
            .to_string();
        let formula = obj
            .get("formula")
            .or_else(|| obj.get("molecular_formula"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let smiles = obj
            .get("smiles")
            .or_else(|| obj.get("canonical_smiles"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let mut ref_ppms = Vec::new();
        if let Some(p_arr) = obj
            .get("peaks")
            .or_else(|| obj.get("shifts"))
            .and_then(|v| v.as_array())
        {
            for pv in p_arr {
                if let Some(p) = pv.as_f64().filter(|v| v.is_finite()) {
                    ref_ppms.push(p);
                } else if let Some(p) = pv
                    .as_object()
                    .and_then(|po| po.get("ppm").or_else(|| po.get("shift")))
                    .and_then(|v| v.as_f64())
                    .filter(|v| v.is_finite())
                {
                    ref_ppms.push(p);
                }
            }
        }

        let mut peaks_1h = Vec::new();
        let mut shifts_13c = Vec::new();
        let mut matched_pairs = Vec::new();
        let mut sum_err = 0.0_f64;
        let mut sum_shift = 0.0_f64;

        for (idx, &rppm) in ref_ppms.iter().enumerate() {
            let lbl = format!("P{}", idx + 1);
            match nucleus {
                MetaboliteNucleus::H1 => peaks_1h.push(MetabolitePeak1H {
                    assignment: lbl.clone(),
                    ppm: rppm,
                    multiplicity: PeakMultiplicity::Multiplet,
                    j_hz: Vec::new(),
                    n_protons: 1,
                }),
                MetaboliteNucleus::C13 => shifts_13c.push(MetaboliteShift13C {
                    assignment: lbl.clone(),
                    ppm: rppm,
                    n_carbons: 1,
                }),
            }
            if let Some(best_q) = active_q
                .iter()
                .min_by(|a, b| (a.ppm - rppm).abs().total_cmp(&(b.ppm - rppm).abs()))
            {
                let diff = (best_q.ppm - rppm).abs();
                if diff <= tol_ppm {
                    sum_err += diff;
                    sum_shift += (-1.5 * (diff / tol_ppm).powi(2)).exp();
                    matched_pairs.push(MatchedPeakPair {
                        assignment: lbl,
                        ref_ppm: rppm,
                        query_ppm: best_q.ppm,
                        delta_ppm: best_q.ppm - rppm,
                        mult_j_consistency: 1.0,
                    });
                }
            }
        }

        let reported_score = obj
            .get("score")
            .and_then(|v| v.as_f64())
            .filter(|v| v.is_finite())
            .map(|v| v.clamp(0.0, 1.0));

        let (score, recall, precision, shift_score, mean_abs_error_ppm, matched_ref, total_ref) =
            if !matched_pairs.is_empty() {
                let nm = matched_pairs.len();
                let nr = ref_ppms.len().max(nm);
                let rec = (nm as f64) / (nr as f64);
                let prec = ((nm as f64) / q_denom).clamp(0.0, 1.0);
                let sh = (sum_shift / (nm as f64)).clamp(0.0, 1.0);
                let sc = reported_score
                    .unwrap_or_else(|| (0.45 * rec + 0.25 * prec + 0.30 * sh).clamp(0.0, 1.0));
                (sc, rec, prec, sh, sum_err / (nm as f64), nm, nr)
            } else if let Some(sc) = reported_score {
                (sc, sc, sc, sc, 0.0, 0, ref_ppms.len())
            } else {
                continue;
            };

        hits.push(MetaboliteMatch {
            entry: MetaboliteEntry {
                bmse_id: acc,
                name,
                formula,
                smiles,
                solvent: "D2O".to_string(),
                ph: 7.0,
                peaks_1h,
                shifts_13c,
                source: MetaboliteSource::NpMrdOnline,
            },
            nucleus,
            score,
            recall,
            precision,
            shift_score,
            mult_j_score: 1.0,
            matched_ref_peaks: matched_ref,
            total_ref_peaks: total_ref,
            mean_abs_error_ppm,
            matched_pairs,
        });
    }

    Ok(hits)
}

/// Opt-in desktop online query using a pluggable transport (useful for deterministic unit tests).
#[cfg(not(target_arch = "wasm32"))]
pub fn query_online_bmrb_and_npmrd_with_transport<T: OnlineMetaboliteTransport>(
    query_peaks: &[QueryPeak],
    nucleus: MetaboliteNucleus,
    tol_ppm: f64,
    sources: OnlineQuerySources,
    transport: &mut T,
) -> Result<Vec<MetaboliteMatch>, String> {
    if !sources.bmrb && !sources.npmrd {
        return Err("no online metabolite sources selected".to_string());
    }
    if !tol_ppm.is_finite() || tol_ppm <= 0.0 {
        return Err(format!("invalid shift tolerance {tol_ppm} ppm"));
    }
    let active_q = filter_active_query_peaks(query_peaks);
    if active_q.is_empty() {
        return Err("at least one finite query peak is required for online search".to_string());
    }

    let mut combined: Vec<MetaboliteMatch> = Vec::new();
    let mut errors: Vec<String> = Vec::new();
    let mut succeeded_sources = 0usize;

    if sources.bmrb {
        let mut bmrb_ok = false;
        // Query up to 3 representative peaks against BMRB v2 chemical_shifts endpoint.
        let mut bmrb_raw_hits = Vec::new();
        for qp in active_q.iter().take(3) {
            let url = build_bmrb_v2_shift_search_url(qp.ppm, nucleus, tol_ppm)?;
            match transport.get_https_json(&url) {
                Ok(body) => match parse_bmrb_v2_response(&body, query_peaks, nucleus, tol_ppm) {
                    Ok(mut parsed) => {
                        bmrb_ok = true;
                        bmrb_raw_hits.append(&mut parsed);
                    }
                    Err(e) => errors.push(format!("BMRB v2 parse error: {e}")),
                },
                Err(e) => errors.push(format!("BMRB v2 network error: {e}")),
            }
        }
        if bmrb_ok {
            succeeded_sources += 1;
            combined.extend(bmrb_raw_hits);
        }
    }

    if sources.npmrd {
        let url = build_npmrd_shift_search_url(query_peaks, nucleus, tol_ppm)?;
        match transport.get_https_json(&url) {
            Ok(body) => match parse_npmrd_response(&body, query_peaks, nucleus, tol_ppm) {
                Ok(parsed) => {
                    succeeded_sources += 1;
                    combined.extend(parsed);
                }
                Err(e) => errors.push(format!("NP-MRD parse error: {e}")),
            },
            Err(e) => errors.push(format!("NP-MRD network error: {e}")),
        }
    }

    if succeeded_sources == 0 {
        return Err(format!(
            "all online metabolite queries failed (offline or unreachable): {}",
            errors.join("; ")
        ));
    }

    // Deduplicate by (source, bmse_id), keeping highest-scoring entry.
    combined.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.mean_abs_error_ppm.total_cmp(&b.mean_abs_error_ppm))
            .then_with(|| a.entry.bmse_id.cmp(&b.entry.bmse_id))
    });
    let mut deduped: Vec<MetaboliteMatch> = Vec::with_capacity(combined.len());
    for hit in combined {
        let exists = deduped
            .iter()
            .any(|d| d.entry.source == hit.entry.source && d.entry.bmse_id == hit.entry.bmse_id);
        if !exists {
            deduped.push(hit);
        }
    }
    Ok(deduped)
}

/// Opt-in desktop online metabolite search querying `BMRB v2` (`https://api.bmrb.io/v2/...`)
/// and/or `NP-MRD` (`https://np-mrd.org/...`).
///
/// Strictly enforces HTTPS host allowlist (`api.bmrb.io`, `np-mrd.org` only), `5 s` connect/read
/// timeout, `4 MiB` response cap, no cross-host redirects, and fails closed with `Err` when
/// offline. Never invoked automatically without explicit user opt-in.
#[cfg(not(target_arch = "wasm32"))]
pub fn query_online_bmrb_and_npmrd(
    query_peaks: &[QueryPeak],
    nucleus: MetaboliteNucleus,
    tol_ppm: f64,
    sources: OnlineQuerySources,
) -> Result<Vec<MetaboliteMatch>, String> {
    let mut transport = CurlMetaboliteTransport;
    query_online_bmrb_and_npmrd_with_transport(
        query_peaks,
        nucleus,
        tol_ppm,
        sources,
        &mut transport,
    )
}
