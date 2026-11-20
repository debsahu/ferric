//! Offline `BMRB-BMSE` Compact Aqueous Metabolite Reference Index, Fast `< 5 ms`
//! Peak-List / `STOCSY`-Cluster Matcher, 1D Reference Spectrum Synthesizer, and Opt-In
//! Desktop `BMRB v2` Online Query Engine (with honest `NP-MRD` web search URL).
//!
//! # Clean-Room Scientific & Open-Data Provenance
//! Original clean-room implementation built strictly from open scientific archives and
//! peer-reviewed NMR metabolomics literature:
//! - **BMRB (Biological Magnetic Resonance Data Bank, `BMRB-BMSE`; free access, © UConn Health,
//!   acknowledgement requested)**:
//!   Ulrich, E. L., Akutsu, H., Doreleijers, J. F., Harano, Y., Ioannidis, Y. E., Lin, J.,
//!   Livny, M., Mading, S., Maziuk, D., Miller, Z., Nakatani, E., Schulte, C. F., Tolmie, D. E.,
//!   Kent Wenger, R., Yao, H. & Markley, J. L., "BioMagResBank," *Nucleic Acids Res.* **2008**,
//!   36(Database issue), D402–D408 (`10.1093/nar/gkm957`); <https://bmrb.io/metabolomics/>.
//!   Every entry in `OFFLINE_BMRB_METABOLITES` was verified directly against the live
//!   `https://api.bmrb.io/v2/entry/<bmse_id>` REST API (accession ID, `Chem_comp.Name`,
//!   `Formula`, `Chem_comp_SMILES`, `Sample_conditions` pH, and `Assigned_chem_shift` /
//!   `spectral_peak_1H` chemical shifts).
//! - **NP-MRD (Natural Products Magnetic Resonance Database, `CC-BY-4.0`)**:
//!   Wishart, D. S., Sayeeda, Z., Budinski, Z., Guo, A., Lee, B. L., Berjanskii, M., Rout, M.,
//!   Peters, H., Dizon, R., Mah, R., Torres-Calzada, C., Hiebert-Giesbrecht, M., Varshavi, D.,
//!   Varshavi, D., Oler, E., Allen, D., Cao, X., Gautam, V., Maras, A., Poynton, E. F.,
//!   Tavangar, P., Yang, V., van Santen, J. A., Linington, R. G. & Hu, X., "NP-MRD: the Natural
//!   Products Magnetic Resonance Database," *Nucleic Acids Res.* **2022**, 50(D1), D622–D631
//!   (`10.1093/nar/gkab1052`); <https://np-mrd.org/>. NP-MRD does not expose an unauthenticated
//!   JSON shift-search API, so FERRIC provides the direct web search URL (`https://np-mrd.org/`)
//!   instead of calling a nonexistent endpoint.
//! - **STOCSY (Statistical Total Correlation Spectroscopy)**:
//!   Cloarec, O., Dumas, M.-E., Craig, A., Barton, R. H., Trygg, J., Hudson, J., Blancher, C.,
//!   Gauguier, D., Lindon, J. C., Holmes, E. & Nicholson, J. K., *Anal. Chem.* **2005**, 77(5),
//!   1282–1289 (`10.1021/ac048630x`).
//!
//! **Clean-room notice**: Strictly zero `HMDB` (non-commercial restriction) or `Chenomx`
//! (proprietary) data, libraries, or shift tables were consulted or bundled. All online queries
//! are strictly opt-in, desktop-only (`#[cfg(not(target_arch = "wasm32"))]`), enforce an exact
//! HTTPS host allowlist (`api.bmrb.io`, `np-mrd.org` only), invoke `curl -q` (ignoring
//! `~/.curlrc`), use a `5 s` connect/read timeout and `4 MiB` response cap, disable cross-host
//! redirects, and fail closed when offline.

use serde::{Deserialize, Serialize};

use crate::metabolomics::{SpectralBucket, StocsyResult};

/// Exact strictly-allowlisted HTTPS hosts permitted for opt-in online metabolite queries.
pub const ALLOWED_ONLINE_HOSTS: &[&str] = &["api.bmrb.io", "np-mrd.org"];

/// Canonical BMRB v2 REST API base URL.
pub const BMRB_V2_API_BASE: &str = "https://api.bmrb.io/v2";

/// Canonical NP-MRD web portal base URL (manual browser search; no public JSON shift-search API).
pub const NPMRD_WEB_URL: &str = "https://np-mrd.org/";

/// Backwards-compatible alias for [`NPMRD_WEB_URL`] host base.
pub const NPMRD_API_BASE: &str = "https://np-mrd.org";

/// Hard upper bound on chemical-shift matching tolerance (`2.0 ppm`) for CLI/online queries.
pub const MAX_METABOLITE_TOL_PPM: f64 = 2.0;

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
        if matches!(self, Self::Singlet) || matches!(other, Self::Singlet) {
            return if matches!(self, Self::Multiplet) || matches!(other, Self::Multiplet) {
                0.20
            } else {
                0.15
            };
        }
        if matches!(self, Self::Multiplet) || matches!(other, Self::Multiplet) {
            return 0.68;
        }
        match (self, other) {
            (Self::DoubletOfDoublets, Self::Triplet)
            | (Self::Triplet, Self::DoubletOfDoublets)
            | (Self::DoubletOfDoublets, Self::Quartet)
            | (Self::Quartet, Self::DoubletOfDoublets)
            | (Self::DoubletOfTriplets, Self::Triplet)
            | (Self::Triplet, Self::DoubletOfTriplets)
            | (Self::Doublet, Self::Quartet)
            | (Self::Quartet, Self::Doublet)
            | (Self::Triplet, Self::Quintet)
            | (Self::Quintet, Self::Triplet) => 0.62,
            _ => 0.20,
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
    /// Built-in offline `BMRB-BMSE` compact index (free access, © UConn Health, acknowledgement requested).
    BmrbBmseOffline,
    /// Opt-in live desktop query against `BMRB v2` REST API (`https://api.bmrb.io/v2/...`).
    BmrbOnline,
    /// Reference marker for `NP-MRD` (`https://np-mrd.org/`, `CC-BY-4.0` web portal).
    NpMrdOnline,
}

impl MetaboliteSource {
    pub fn label(&self) -> &'static str {
        match self {
            Self::BmrbBmseOffline => "BMRB-BMSE (Offline)",
            Self::BmrbOnline => "BMRB v2 (Online)",
            Self::NpMrdOnline => "NP-MRD (CC-BY-4.0 Web)",
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
// Static Offline BMRB-BMSE Compact Metabolite Index (57 Live-Verified Entries)
// Every entry verified against https://api.bmrb.io/v2/entry/<bmse_id>
// (free access, © UConn Health, acknowledgement requested).
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
    aliases: &'static [&'static str],
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
    // 1. L-(+) Lactic acid (bmse000208, D2O, pH 7.40)
    StaticMetaboliteEntry {
        bmse_id: "bmse000208",
        name: "L-(+) Lactic acid",
        aliases: &["L-Lactate", "L-Lactic acid"],
        formula: "C3H6O3",
        smiles: "C[C@@H](C(=O)O)O",
        ph: 7.40,
        peaks_1h: &[
            p1h!("H7/H8/H9", 1.318, Doublet, [7.5], 3),
            p1h!("H10", 4.101, Quartet, [7.0], 1),
        ],
        shifts_13c: &[
            c13!("C1", 22.812, 1),
            c13!("C2", 71.227, 1),
            c13!("C3", 185.321, 1),
        ],
    },
    // 2. L-Alanine (bmse000028, D2O, pH 7.40)
    StaticMetaboliteEntry {
        bmse_id: "bmse000028",
        name: "L-Alanine",
        aliases: &["L-Alanine"],
        formula: "C3H7NO2",
        smiles: "C[C@@H](C(=O)O)N",
        ph: 7.40,
        peaks_1h: &[
            p1h!("H7/H8/H9", 1.471, Doublet, [7.2], 3),
            p1h!("H10", 3.771, Quartet, [7.2], 1),
        ],
        shifts_13c: &[
            c13!("C1", 18.830, 1),
            c13!("C2", 53.200, 1),
            c13!("C3", 178.560, 1),
        ],
    },
    // 3. D-(+)-Glucose (bmse000015, D2O, pH 7.40)
    StaticMetaboliteEntry {
        bmse_id: "bmse000015",
        name: "D-(+)-Glucose",
        aliases: &["D-Glucose", "Glucose"],
        formula: "C6H12O6",
        smiles: "C([C@@H]1[C@H]([C@@H]([C@H]([C@H](O1)O)O)O)O)O",
        ph: 7.40,
        peaks_1h: &[
            p1h!("H13..H18(6H)", 3.232, Triplet, [8.8], 1),
            p1h!("H13..H18(6H)", 3.401, Multiplet, [], 1),
            p1h!("H13..H18(6H)", 3.459, Multiplet, [], 1),
            p1h!("H13..H18(6H)", 3.517, Multiplet, [], 1),
            p1h!("H13..H18(6H)", 3.726, Multiplet, [], 1),
            p1h!("H13..H18(6H)", 3.828, Multiplet, [], 1),
            p1h!("H13..H18(6H)", 3.887, DoubletOfDoublets, [], 1),
            p1h!("H19", 4.636, Doublet, [8.0], 1),
            p1h!("H19", 5.222, Doublet, [3.5], 1),
        ],
        shifts_13c: &[
            c13!("C1", 63.284, 1),
            c13!("C1", 63.454, 1),
            c13!("C2..C5", 72.309, 1),
            c13!("C2..C5", 72.352, 1),
            c13!("C2..C5", 74.166, 1),
            c13!("C2..C5", 74.202, 1),
            c13!("C2..C5", 75.480, 1),
            c13!("C2..C5", 76.854, 1),
            c13!("C2..C5", 78.474, 1),
            c13!("C2..C5", 78.690, 1),
            c13!("C6", 94.826, 1),
            c13!("C6", 98.643, 1),
        ],
    },
    // 4. Acetic acid (bmse000191, D2O, pH 7.40)
    StaticMetaboliteEntry {
        bmse_id: "bmse000191",
        name: "Acetic acid",
        aliases: &["Acetate", "Acetic acid"],
        formula: "C2H4O2",
        smiles: "CC(=O)O",
        ph: 7.40,
        peaks_1h: &[p1h!("H5/H6/H7", 1.905, Singlet, [], 3)],
        shifts_13c: &[c13!("C1", 25.990, 1), c13!("C2", 184.250, 1)],
    },
    // 5. Citrate (bmse000076, D2O, pH 7.40)
    // Live-verified against https://api.bmrb.io/v2/entry/bmse000076: `assigned_chemical_shifts`
    // (`_Atom_chem_shift`) stores the degenerate spin-system midpoint 2.579 ppm (an empty valley
    // between the two strongly coupled prochiral CH2 AB doublets), whereas saveframe
    // `spectral_peak_HSQC` loop `_Peak_char` (`Spectral_dim_ID = 1`) reports the observed 1H AB
    // peak positions at 2.531 ppm (`Peak_ID = 2`) and 2.661 ppm (`Peak_ID = 1`), with
    // |2J_AB| = 15.2 Hz from saveframes `spectral_peak_1Hp5` / `spectral_peak_1H_2` loop
    // `_Spectral_transition_char` (transitions 2.5086, 2.5389, 2.6286, 2.6589 ppm at 500 MHz).
    StaticMetaboliteEntry {
        bmse_id: "bmse000076",
        name: "Citrate",
        aliases: &["Citrate", "Citric acid"],
        formula: "C6H8O7",
        smiles: "C(C(=O)O)C(CC(=O)O)(C(=O)O)O",
        ph: 7.40,
        peaks_1h: &[
            p1h!("H14..H17", 2.531, Doublet, [15.2], 2),
            p1h!("H14..H17", 2.661, Doublet, [15.2], 2),
        ],
        shifts_13c: &[
            c13!("C1/C2", 48.446, 2),
            c13!("C6", 77.963, 1),
            c13!("C3/C4", 182.021, 2),
            c13!("C5", 184.740, 1),
        ],
    },
    // 6. Succinic acid (bmse000183, D2O, pH 7.40)
    StaticMetaboliteEntry {
        bmse_id: "bmse000183",
        name: "Succinic acid",
        aliases: &["Succinate", "Succinic acid"],
        formula: "C4H6O4",
        smiles: "C(CC(=O)O)C(=O)O",
        ph: 7.40,
        peaks_1h: &[p1h!("H10..H9(4H)", 2.391, Singlet, [], 4)],
        shifts_13c: &[c13!("C1/C2", 36.915, 2), c13!("C3/C4", 185.203, 2)],
    },
    // 7. Creatine (bmse000078, D2O, pH 7.40)
    StaticMetaboliteEntry {
        bmse_id: "bmse000078",
        name: "Creatine",
        aliases: &["Creatine"],
        formula: "C4H9N3O2",
        smiles: "CN(CC(=O)O)C(=N)N",
        ph: 7.40,
        peaks_1h: &[
            p1h!("H10/H11/H12", 3.024, Singlet, [], 3),
            p1h!("H13/H14", 3.920, Singlet, [], 2),
        ],
        shifts_13c: &[
            c13!("C1", 39.653, 1),
            c13!("C2", 56.555, 1),
            c13!("C4", 159.820, 1),
            c13!("C3", 177.426, 1),
        ],
    },
    // 8. Creatinine (bmse000155, D2O, pH 7.40)
    StaticMetaboliteEntry {
        bmse_id: "bmse000155",
        name: "Creatinine",
        aliases: &["Creatinine"],
        formula: "C4H7N3O",
        smiles: "CN1CC(=O)N=C1N",
        ph: 7.40,
        peaks_1h: &[
            p1h!("H10/H11/H9", 3.034, Singlet, [], 3),
            p1h!("H12/H13", 4.042, Singlet, [], 2),
        ],
        shifts_13c: &[
            c13!("C1", 32.930, 1),
            c13!("C2", 59.156, 1),
            c13!("C4", 172.002, 1),
            c13!("C3", 191.757, 1),
        ],
    },
    // 9. Glycine (bmse000089, D2O, pH 7.40)
    StaticMetaboliteEntry {
        bmse_id: "bmse000089",
        name: "Glycine",
        aliases: &["Glycine"],
        formula: "C2H5NO2",
        smiles: "C(C(=O)O)N",
        ph: 7.40,
        peaks_1h: &[p1h!("H6/H7", 3.545, Singlet, [], 2)],
        shifts_13c: &[c13!("C1", 44.133, 1), c13!("C2", 175.225, 1)],
    },
    // 10. Taurine (bmse000120, D2O, pH 7.40)
    StaticMetaboliteEntry {
        bmse_id: "bmse000120",
        name: "Taurine",
        aliases: &["Taurine"],
        formula: "C2H7NO3S",
        smiles: "C(CS(=O)(=O)O)N",
        ph: 7.40,
        peaks_1h: &[
            p1h!("H10/H11", 3.246, Triplet, [6.8], 2),
            p1h!("H8/H9", 3.410, Triplet, [6.8], 2),
        ],
        shifts_13c: &[c13!("C1", 38.068, 1), c13!("C2", 50.215, 1)],
    },
    // 11. Choline (bmse000285, D2O, pH 7.40)
    StaticMetaboliteEntry {
        bmse_id: "bmse000285",
        name: "Choline",
        aliases: &["Choline"],
        formula: "C5H14NO+",
        smiles: "C[N+](C)(C)CCO",
        ph: 7.40,
        peaks_1h: &[
            p1h!("H10..H9(9H)", 3.195, Singlet, [], 9),
            p1h!("H17/H18", 3.514, Triplet, [5.0], 2),
            p1h!("H19/H20", 4.053, Multiplet, [], 2),
        ],
        shifts_13c: &[
            c13!("C1/C2/C3", 56.584, 3),
            c13!("C5", 58.325, 1),
            c13!("C4", 70.117, 1),
        ],
    },
    // 12. Trimethylamine N-oxide (bmse000426, D2O, pH 7.40)
    StaticMetaboliteEntry {
        bmse_id: "bmse000426",
        name: "Trimethylamine N-oxide",
        aliases: &["TMAO", "Trimethylamine N-oxide"],
        formula: "C3H9NO",
        smiles: "C[N+](C)(C)[O-]",
        ph: 7.40,
        peaks_1h: &[p1h!("H10..H9(9H)", 3.249, Singlet, [], 9)],
        shifts_13c: &[c13!("C1/C2/C3", 62.262, 3)],
    },
    // 13. L-Glutamine (bmse000038, D2O, pH 7.40)
    StaticMetaboliteEntry {
        bmse_id: "bmse000038",
        name: "L-Glutamine",
        aliases: &["L-Glutamine"],
        formula: "C5H10N2O3",
        smiles: "C(CC(=O)N)[C@@H](C(=O)O)N",
        ph: 7.40,
        peaks_1h: &[
            p1h!("H11/H12", 2.130, Quartet, [6.9], 2),
            p1h!("H13/H14", 2.447, Multiplet, [], 2),
            p1h!("H15", 3.764, Triplet, [6.2], 1),
        ],
        shifts_13c: &[
            c13!("C1", 28.950, 1),
            c13!("C2", 33.520, 1),
            c13!("C3", 56.830, 1),
            c13!("C5", 176.830, 1),
            c13!("C4", 180.370, 1),
        ],
    },
    // 14. L-Glutamic acid (bmse000037, D2O, pH 7.40)
    StaticMetaboliteEntry {
        bmse_id: "bmse000037",
        name: "L-Glutamic acid",
        aliases: &["L-Glutamate", "L-Glutamic acid"],
        formula: "C5H9NO4",
        smiles: "C(CC(=O)O)C(C(=O)O)N",
        ph: 7.40,
        peaks_1h: &[
            p1h!("H11/H12", 2.078, Multiplet, [], 2),
            p1h!("H13/H14", 2.339, Multiplet, [], 2),
            p1h!("H15", 3.747, DoubletOfDoublets, [], 1),
        ],
        shifts_13c: &[
            c13!("C1", 29.728, 1),
            c13!("C2", 36.200, 1),
            c13!("C3", 57.357, 1),
            c13!("C5", 177.360, 1),
            c13!("C4", 184.088, 1),
        ],
    },
    // 15. L-Valine (bmse000052, D2O, pH 7.40)
    StaticMetaboliteEntry {
        bmse_id: "bmse000052",
        name: "L-Valine",
        aliases: &["L-Valine"],
        formula: "C5H11NO2",
        smiles: "CC(C)[C@@H](C(=O)O)N",
        ph: 7.40,
        peaks_1h: &[
            p1h!("H10..H9(6H)", 0.981, Doublet, [7.0], 1),
            p1h!("H10..H9(6H)", 1.034, Doublet, [7.0], 1),
            p1h!("H15", 2.266, Multiplet, [], 1),
            p1h!("H16", 3.599, DoubletOfDoublets, [], 1),
        ],
        shifts_13c: &[
            c13!("C1/C2", 19.368, 1),
            c13!("C1/C2", 20.696, 1),
            c13!("C3", 31.834, 1),
            c13!("C4", 63.084, 1),
            c13!("C5", 177.086, 1),
        ],
    },
    // 16. L-Leucine (bmse000042, D2O, pH 7.40)
    StaticMetaboliteEntry {
        bmse_id: "bmse000042",
        name: "L-Leucine",
        aliases: &["L-Leucine"],
        formula: "C6H13NO2",
        smiles: "CC(C)C[C@@H](C(=O)O)N",
        ph: 7.40,
        peaks_1h: &[
            p1h!("H10..H15(6H)", 0.949, Triplet, [5.5], 6),
            p1h!("H16/H17/H18", 1.701, Multiplet, [], 3),
            p1h!("H19", 3.719, Multiplet, [], 1),
        ],
        shifts_13c: &[
            c13!("C1/C2", 23.589, 1),
            c13!("C1/C2", 24.751, 1),
            c13!("C4", 26.871, 1),
            c13!("C3", 42.526, 1),
            c13!("C5", 56.112, 1),
            c13!("C6", 178.382, 1),
        ],
    },
    // 17. L-Isoleucine (bmse000041, D2O, pH 7.40)
    StaticMetaboliteEntry {
        bmse_id: "bmse000041",
        name: "L-Isoleucine",
        aliases: &["L-Isoleucine"],
        formula: "C6H13NO2",
        smiles: "CC[C@H](C)[C@@H](C(=O)O)N",
        ph: 7.40,
        peaks_1h: &[
            p1h!("H10/H11/H12", 0.927, Triplet, [7.5], 1),
            p1h!("H13/H14/H15", 0.998, Doublet, [7.0], 1),
            p1h!("H16/H17", 1.249, Multiplet, [], 1),
            p1h!("H16/H17", 1.458, Multiplet, [], 1),
            p1h!("H18", 1.969, Multiplet, [], 1),
            p1h!("H19", 3.657, Doublet, [4.0], 1),
        ],
        shifts_13c: &[
            c13!("C1", 13.834, 1),
            c13!("C2", 17.411, 1),
            c13!("C3", 27.174, 1),
            c13!("C4", 38.614, 1),
            c13!("C5", 62.249, 1),
            c13!("C6", 176.972, 1),
        ],
    },
    // 18. L-Tyrosine (bmse000051, D2O, pH 11.15)
    StaticMetaboliteEntry {
        bmse_id: "bmse000051",
        name: "L-Tyrosine",
        aliases: &["L-Tyrosine"],
        formula: "C9H11NO3",
        smiles: "C1=CC(=CC=C1C[C@@H](C(=O)O)N)O",
        ph: 11.15,
        peaks_1h: &[
            p1h!("H18/H19", 3.055, Multiplet, [], 1),
            p1h!("H18/H19", 3.200, Multiplet, [], 1),
            p1h!("H20", 3.936, Multiplet, [], 1),
            p1h!("H16/H17", 6.898, Multiplet, [], 1),
            p1h!("H14/H15", 7.192, Multiplet, [], 1),
        ],
        shifts_13c: &[
            c13!("C5", 38.277, 1),
            c13!("C8", 58.838, 1),
            c13!("C3/C4", 118.610, 2),
            c13!("C6", 129.523, 1),
            c13!("C1/C2", 133.531, 2),
            c13!("C7", 157.683, 1),
            c13!("C9", 176.964, 1),
        ],
    },
    // 19. L-Phenylalanine (bmse000045, D2O, pH 7.40)
    StaticMetaboliteEntry {
        bmse_id: "bmse000045",
        name: "L-Phenylalanine",
        aliases: &["L-Phenylalanine"],
        formula: "C9H11NO2",
        smiles: "C1=CC=C(C=C1)C[C@@H](C(=O)O)N",
        ph: 7.40,
        peaks_1h: &[
            p1h!("H18/H19", 3.110, DoubletOfDoublets, [], 1),
            p1h!("H18/H19", 3.271, DoubletOfDoublets, [], 1),
            p1h!("H20", 3.975, Quartet, [4.5], 1),
            p1h!("H13..H17(5H)", 7.369, Multiplet, [], 1),
        ],
        shifts_13c: &[
            c13!("C6", 39.095, 1),
            c13!("C8", 58.744, 1),
            c13!("C1", 130.418, 1),
            c13!("C2/C3", 131.834, 2),
            c13!("C4/C5", 132.093, 2),
            c13!("C7", 137.798, 1),
            c13!("C9", 176.774, 1),
        ],
    },
    // 20. L-Histidine (bmse000039, D2O, pH 7.40)
    StaticMetaboliteEntry {
        bmse_id: "bmse000039",
        name: "L-Histidine",
        aliases: &["L-Histidine"],
        formula: "C6H9N3O2",
        smiles: "C1=C(NC=N1)C[C@@H](C(=O)O)N",
        ph: 7.40,
        peaks_1h: &[
            p1h!("H12/H13", 3.131, DoubletOfDoublets, [], 1),
            p1h!("H12/H13", 3.234, DoubletOfDoublets, [], 1),
            p1h!("H16", 3.980, Quartet, [4.0], 1),
            p1h!("H14", 7.066, Singlet, [], 1),
            p1h!("H15", 7.826, Singlet, [], 1),
        ],
        shifts_13c: &[
            c13!("C1", 30.696, 1),
            c13!("C5", 57.435, 1),
            c13!("C2", 119.574, 1),
            c13!("C4", 134.445, 1),
            c13!("C3", 138.873, 1),
            c13!("C6", 176.642, 1),
        ],
    },
    // 21. Formate (bmse000203, D2O, pH 7.40)
    StaticMetaboliteEntry {
        bmse_id: "bmse000203",
        name: "Formate",
        aliases: &["Formate", "Formic acid"],
        formula: "CH2O2",
        smiles: "C(=O)O",
        ph: 7.40,
        peaks_1h: &[p1h!("H4", 8.447, Singlet, [], 1)],
        shifts_13c: &[c13!("C1", 173.892, 1)],
    },
    // 22. Pyruvic acid (bmse000112, D2O, pH 7.40)
    StaticMetaboliteEntry {
        bmse_id: "bmse000112",
        name: "Pyruvic acid",
        aliases: &["Pyruvate", "Pyruvic acid"],
        formula: "C3H4O3",
        smiles: "CC(=O)C(=O)O",
        ph: 7.40,
        peaks_1h: &[p1h!("H7/H8/H9", 2.364, Singlet, [], 3)],
        shifts_13c: &[
            c13!("C1", 29.204, 1),
            c13!("C3", 172.967, 1),
            c13!("C2", 207.760, 1),
        ],
    },
    // 23. Fumaric acid (bmse000083, D2O, pH 7.40)
    StaticMetaboliteEntry {
        bmse_id: "bmse000083",
        name: "Fumaric acid",
        aliases: &["Fumarate", "Fumaric acid"],
        formula: "C4H4O4",
        smiles: "C(=C\\C(=O)O)/C(=O)O",
        ph: 7.40,
        peaks_1h: &[p1h!("H10/H9", 6.508, Singlet, [], 2)],
        shifts_13c: &[c13!("C1/C2", 138.036, 2), c13!("C3/C4", 177.427, 2)],
    },
    // 24. Ethanol (bmse000297, D2O, pH 7.40)
    StaticMetaboliteEntry {
        bmse_id: "bmse000297",
        name: "Ethanol",
        aliases: &["Ethanol"],
        formula: "C2H6O",
        smiles: "CCO",
        ph: 7.40,
        peaks_1h: &[
            p1h!("H4/H5/H6", 1.175, Triplet, [7.2], 3),
            p1h!("H7/H8", 3.645, Quartet, [7.1], 2),
        ],
        shifts_13c: &[c13!("C1", 19.480, 1), c13!("C2", 60.130, 1)],
    },
    // 25. Methanol (bmse000294, D2O, pH 7.40)
    StaticMetaboliteEntry {
        bmse_id: "bmse000294",
        name: "Methanol",
        aliases: &["Methanol"],
        formula: "CH4O",
        smiles: "CO",
        ph: 7.40,
        peaks_1h: &[p1h!("H3/H4/H5", 3.346, Singlet, [], 3)],
        shifts_13c: &[c13!("C1", 51.570, 1)],
    },
    // 26. Betaine (bmse000069, D2O, pH 7.40)
    StaticMetaboliteEntry {
        bmse_id: "bmse000069",
        name: "Betaine",
        aliases: &["Betaine"],
        formula: "C5H11NO2",
        smiles: "C[N+](C)(C)CC(=O)[O-]",
        ph: 7.40,
        peaks_1h: &[
            p1h!("H10..H9(9H)", 3.255, Singlet, [], 9),
            p1h!("H18/H19", 3.895, Singlet, [], 2),
        ],
        shifts_13c: &[
            c13!("C1/C2/C3", 56.053, 3),
            c13!("C4", 68.838, 1),
            c13!("C5", 172.009, 1),
        ],
    },
    // 27. myo-Inositol (bmse000103, D2O, pH 7.40)
    StaticMetaboliteEntry {
        bmse_id: "bmse000103",
        name: "myo-Inositol",
        aliases: &["Myo-inositol", "myo-Inositol"],
        formula: "C6H12O6",
        smiles: "C1(C(C(C(C(C1O)O)O)O)O)O",
        ph: 7.40,
        peaks_1h: &[
            p1h!("H13..H18(6H)", 3.264, Triplet, [9.2], 1),
            p1h!("H13..H18(6H)", 3.520, DoubletOfDoublets, [], 1),
            p1h!("H13..H18(6H)", 3.609, Triplet, [9.8], 1),
            p1h!("H13..H18(6H)", 4.047, Triplet, [2.5], 1),
        ],
        shifts_13c: &[
            c13!("C1..C6", 73.849, 1),
            c13!("C1..C6", 74.906, 1),
            c13!("C1..C6", 75.115, 1),
            c13!("C1..C6", 77.073, 1),
        ],
    },
    // 28. L-Serine (bmse000048, D2O, pH 7.40)
    StaticMetaboliteEntry {
        bmse_id: "bmse000048",
        name: "L-Serine",
        aliases: &["L-Serine"],
        formula: "C3H7NO3",
        smiles: "C([C@@H](C(=O)O)N)O",
        ph: 7.40,
        peaks_1h: &[
            p1h!("H10", 3.828, Multiplet, [], 1),
            p1h!("H8/H9", 3.952, Multiplet, [], 2),
        ],
        shifts_13c: &[
            c13!("C2", 59.096, 1),
            c13!("C1", 62.906, 1),
            c13!("C3", 175.227, 1),
        ],
    },
    // 29. L-Threonine (bmse000049, D2O, pH 7.40)
    StaticMetaboliteEntry {
        bmse_id: "bmse000049",
        name: "L-Threonine",
        aliases: &["L-Threonine"],
        formula: "C4H9NO3",
        smiles: "C[C@H]([C@@H](C(=O)O)N)O",
        ph: 7.40,
        peaks_1h: &[
            p1h!("H10/H11/H9", 1.318, Doublet, [6.5], 3),
            p1h!("H13", 3.573, Doublet, [5.0], 1),
            p1h!("H12", 4.241, Quintet, [], 1),
        ],
        shifts_13c: &[
            c13!("C1", 22.179, 1),
            c13!("C3", 63.172, 1),
            c13!("C2", 68.679, 1),
            c13!("C4", 175.689, 1),
        ],
    },
    // 30. L-Proline (bmse000047, D2O, pH 7.40)
    StaticMetaboliteEntry {
        bmse_id: "bmse000047",
        name: "L-Proline",
        aliases: &["L-Proline"],
        formula: "C5H9NO2",
        smiles: "C1C[C@@H](NC1)C(=O)O",
        ph: 7.40,
        peaks_1h: &[
            p1h!("H10..H9(4H)", 2.022, Multiplet, [], 1),
            p1h!("H11/H12", 2.337, Multiplet, [], 1),
            p1h!("H13/H14", 3.366, Multiplet, [], 1),
            p1h!("H15", 4.119, Multiplet, [], 1),
        ],
        shifts_13c: &[
            c13!("C1", 26.487, 1),
            c13!("C2", 31.728, 1),
            c13!("C3", 48.775, 1),
            c13!("C4", 63.922, 1),
            c13!("C5", 177.483, 1),
        ],
    },
    // 31. L-Lysine (bmse000043, D2O, pH 7.40)
    StaticMetaboliteEntry {
        bmse_id: "bmse000043",
        name: "L-Lysine",
        aliases: &["L-Lysine"],
        formula: "C6H14N2O2",
        smiles: "C(CCN)C[C@@H](C(=O)O)N",
        ph: 7.40,
        peaks_1h: &[
            p1h!("H11/H12", 1.465, Multiplet, [], 2),
            p1h!("H13/H14", 1.716, Quintet, [], 2),
            p1h!("H15/H16", 1.895, Multiplet, [], 2),
            p1h!("H17/H18", 3.012, Triplet, [7.5], 2),
            p1h!("H19", 3.745, Triplet, [6.0], 1),
        ],
        shifts_13c: &[
            c13!("C1", 24.145, 1),
            c13!("C2", 29.137, 1),
            c13!("C3", 32.628, 1),
            c13!("C4", 41.754, 1),
            c13!("C5", 57.190, 1),
            c13!("C6", 177.484, 1),
        ],
    },
    // 32. L-Arginine (bmse000029, D2O, pH 7.40)
    StaticMetaboliteEntry {
        bmse_id: "bmse000029",
        name: "L-Arginine",
        aliases: &["L-Arginine"],
        formula: "C6H14N4O2",
        smiles: "C(C[C@@H](C(=O)O)N)CN=C(N)N",
        ph: 7.40,
        peaks_1h: &[
            p1h!("H13/H14", 1.679, Multiplet, [], 2),
            p1h!("H15/H16", 1.909, Multiplet, [], 2),
            p1h!("H17/H18", 3.236, Triplet, [6.8], 2),
            p1h!("H19", 3.764, Triplet, [6.0], 1),
        ],
        shifts_13c: &[
            c13!("C1", 26.577, 1),
            c13!("C2", 30.281, 1),
            c13!("C3", 43.201, 1),
            c13!("C4", 57.002, 1),
            c13!("C6", 159.504, 1),
            c13!("C5", 177.238, 1),
        ],
    },
    // 33. L-Aspartic acid (bmse000031, D2O, pH 7.40)
    StaticMetaboliteEntry {
        bmse_id: "bmse000031",
        name: "L-Aspartic acid",
        aliases: &["L-Aspartate", "L-Aspartic acid"],
        formula: "C4H7NO4",
        smiles: "C([C@@H](C(=O)O)N)C(=O)O",
        ph: 7.40,
        peaks_1h: &[
            p1h!("H11", 2.703, Doublet, [8.8], 1),
            p1h!("H10", 2.786, Multiplet, [], 1),
            p1h!("H12", 3.889, Multiplet, [], 1),
        ],
        shifts_13c: &[
            c13!("C1", 39.250, 1),
            c13!("C2", 54.910, 1),
            c13!("C4", 177.000, 1),
            c13!("C3", 180.300, 1),
        ],
    },
    // 34. L-Asparagine (bmse000030, D2O, pH 7.40)
    StaticMetaboliteEntry {
        bmse_id: "bmse000030",
        name: "L-Asparagine",
        aliases: &["L-Asparagine"],
        formula: "C4H8N2O3",
        smiles: "C([C@@H](C(=O)O)N)C(=O)N",
        ph: 7.40,
        peaks_1h: &[
            p1h!("H10/H11", 2.894, Quartet, [], 2),
            p1h!("H12", 3.991, Multiplet, [], 1),
        ],
        shifts_13c: &[
            c13!("C1", 37.278, 1),
            c13!("C2", 53.990, 1),
            c13!("C3/C4", 176.206, 1),
            c13!("C3/C4", 177.173, 1),
        ],
    },
    // 35. L-Methionine (bmse000044, D2O, pH 7.40)
    StaticMetaboliteEntry {
        bmse_id: "bmse000044",
        name: "L-Methionine",
        aliases: &["L-Methionine"],
        formula: "C5H11NO2S",
        smiles: "CSCC[C@@H](C(=O)O)N",
        ph: 7.40,
        peaks_1h: &[
            p1h!("H10..H14(5H)", 2.122, Multiplet, [], 1),
            p1h!("H13/H14", 2.183, Multiplet, [], 1),
            p1h!("H15/H16", 2.629, Triplet, [7.5], 1),
            p1h!("H17", 3.850, Triplet, [], 1),
        ],
        shifts_13c: &[
            c13!("C1", 16.636, 1),
            c13!("C3", 31.513, 1),
            c13!("C2", 32.395, 1),
            c13!("C4", 56.584, 1),
            c13!("C5", 177.093, 1),
        ],
    },
    // 36. L-Tryptophan (bmse000050, D2O, pH 7.40)
    StaticMetaboliteEntry {
        bmse_id: "bmse000050",
        name: "L-Tryptophan",
        aliases: &["L-Tryptophan"],
        formula: "C11H12N2O2",
        smiles: "C1=CC=C2C(=C1)C(=CN2)C[C@@H](C(=O)O)N",
        ph: 7.40,
        peaks_1h: &[
            p1h!("H20/H21", 3.292, DoubletOfDoublets, [], 1),
            p1h!("H20/H21", 3.471, DoubletOfDoublets, [], 1),
            p1h!("H23", 4.036, Multiplet, [], 1),
            p1h!("H17", 7.190, Triplet, [7.2], 1),
            p1h!("H16/H22", 7.281, Multiplet, [], 1),
            p1h!("H19", 7.529, Doublet, [7.5], 1),
            p1h!("H18", 7.721, Doublet, [8.0], 1),
        ],
        shifts_13c: &[
            c13!("C5", 29.152, 1),
            c13!("C9", 57.764, 1),
            c13!("C7", 110.188, 1),
            c13!("C4", 114.700, 1),
            c13!("C3", 121.196, 1),
            c13!("C2", 122.200, 1),
            c13!("C1", 124.871, 1),
            c13!("C6", 127.797, 1),
            c13!("C8", 129.377, 1),
            c13!("C10", 139.075, 1),
            c13!("C11", 177.332, 1),
        ],
    },
    // 37. Hippuric acid (bmse000408, D2O, pH 7.40)
    StaticMetaboliteEntry {
        bmse_id: "bmse000408",
        name: "Hippuric acid",
        aliases: &["Hippurate", "Hippuric acid"],
        formula: "C9H9NO3",
        smiles: "C1=CC=C(C=C1)C(=O)NCC(=O)O",
        ph: 7.40,
        peaks_1h: &[
            p1h!("H19/H20", 3.944, Singlet, [], 2),
            p1h!("H15/H16", 7.530, Triplet, [7.7], 2),
            p1h!("H14", 7.618, Triplet, [7.5], 1),
            p1h!("H17/H18", 7.808, Doublet, [7.0], 2),
        ],
        shifts_13c: &[
            c13!("C6", 46.502, 1),
            c13!("C4/C5", 129.868, 2),
            c13!("C2/C3", 131.499, 2),
            c13!("C1", 134.902, 1),
            c13!("C7", 136.105, 1),
            c13!("C9", 173.257, 1),
            c13!("C8", 179.579, 1),
        ],
    },
    // 38. 3-Hydroxybutyrate (bmse000161, D2O, pH 7.40)
    StaticMetaboliteEntry {
        bmse_id: "bmse000161",
        name: "3-Hydroxybutyrate",
        aliases: &["3-Hydroxybutyrate"],
        formula: "C4H8O3",
        smiles: "CC(CC(=O)O)O",
        ph: 7.40,
        peaks_1h: &[
            p1h!("H10/H8/H9", 1.187, Doublet, [7.0], 3),
            p1h!("H11/H12", 2.343, Multiplet, [], 2),
            p1h!("H13", 4.139, Multiplet, [], 1),
        ],
        shifts_13c: &[
            c13!("C1", 24.442, 1),
            c13!("C2", 49.253, 1),
            c13!("C3", 68.433, 1),
            c13!("C4", 183.228, 1),
        ],
    },
    // 39. Glycerol (bmse000184, D2O, pH 7.40)
    StaticMetaboliteEntry {
        bmse_id: "bmse000184",
        name: "Glycerol",
        aliases: &["Glycerol"],
        formula: "C3H8O3",
        smiles: "C(C(CO)O)O",
        ph: 7.40,
        peaks_1h: &[
            p1h!("H10..H9(4H)", 3.591, Multiplet, [], 4),
            p1h!("H11", 3.768, Multiplet, [], 1),
        ],
        shifts_13c: &[c13!("C1/C2", 65.187, 2), c13!("C3", 74.769, 1)],
    },
    // 40. L-Cysteine (bmse000034, D2O, pH 7.40)
    StaticMetaboliteEntry {
        bmse_id: "bmse000034",
        name: "L-Cysteine",
        aliases: &["L-Cysteine"],
        formula: "C3H7NO2S",
        smiles: "C([C@@H](C(=O)O)N)S",
        ph: 7.40,
        peaks_1h: &[
            p1h!("H8/H9", 3.044, Multiplet, [], 2),
            p1h!("H10", 3.952, Triplet, [4.7], 1),
        ],
        shifts_13c: &[
            c13!("C1", 27.647, 1),
            c13!("C2", 58.680, 1),
            c13!("C3", 175.486, 1),
        ],
    },
    // 41. Uracil (bmse000187, D2O, pH 7.40)
    StaticMetaboliteEntry {
        bmse_id: "bmse000187",
        name: "Uracil",
        aliases: &["Uracil"],
        formula: "C4H4N2O2",
        smiles: "C1=CNC(=O)NC1=O",
        ph: 7.40,
        peaks_1h: &[
            p1h!("H9", 5.792, Doublet, [8.0], 1),
            p1h!("H10", 7.526, Doublet, [7.5], 1),
        ],
        shifts_13c: &[
            c13!("C1", 103.795, 1),
            c13!("C2", 146.265, 1),
            c13!("C4", 155.933, 1),
            c13!("C3", 170.295, 1),
        ],
    },
    // 42. L-Malic acid (bmse000238, D2O, pH 7.40)
    StaticMetaboliteEntry {
        bmse_id: "bmse000238",
        name: "L-Malic acid",
        aliases: &["L-Malate", "L-Malic acid"],
        formula: "C4H6O5",
        smiles: "C([C@@H](C(=O)O)O)C(=O)O",
        ph: 7.40,
        peaks_1h: &[
            p1h!("H10", 2.355, Multiplet, [], 1),
            p1h!("H12", 2.663, Multiplet, [], 1),
            p1h!("H11", 4.286, Multiplet, [], 1),
        ],
        shifts_13c: &[
            c13!("C1", 45.310, 1),
            c13!("C2", 73.050, 1),
            c13!("C4", 182.570, 1),
            c13!("C3", 183.770, 1),
        ],
    },
    // 43. alpha-Ketoglutaric acid (bmse000064, D2O, pH 7.40)
    StaticMetaboliteEntry {
        bmse_id: "bmse000064",
        name: "alpha-Ketoglutaric acid",
        aliases: &["2-Oxoglutarate", "alpha-Ketoglutarate"],
        formula: "C5H6O5",
        smiles: "C(CC(=O)O)C(=O)C(=O)O",
        ph: 7.40,
        peaks_1h: &[
            p1h!("H11..H14(4H)", 2.428, Triplet, [6.8], 1),
            p1h!("H11..H14(4H)", 2.996, Triplet, [6.8], 1),
        ],
        shifts_13c: &[c13!("C1/C2", 33.325, 1), c13!("C1/C2", 38.573, 1)],
    },
    // 44. gamma-Aminobutyric acid (bmse000340, D2O, pH 7.40)
    StaticMetaboliteEntry {
        bmse_id: "bmse000340",
        name: "gamma-Aminobutyric acid",
        aliases: &["GABA", "4-Aminobutyrate"],
        formula: "C4H9NO2",
        smiles: "C(CC(=O)O)CN",
        ph: 7.40,
        peaks_1h: &[
            p1h!("H8/H9", 1.893, Multiplet, [], 2),
            p1h!("H10/H11", 2.286, Triplet, [7.5], 2),
            p1h!("H12/H13", 2.998, Triplet, [8.0], 2),
        ],
        shifts_13c: &[
            c13!("C1", 26.393, 1),
            c13!("C2", 37.130, 1),
            c13!("C3", 42.005, 1),
            c13!("C4", 184.370, 1),
        ],
    },
    // 45. L-Ornithine (bmse000162, D2O, pH 7.40)
    StaticMetaboliteEntry {
        bmse_id: "bmse000162",
        name: "L-Ornithine",
        aliases: &["L-Ornithine"],
        formula: "C5H12N2O2",
        smiles: "C(C[C@@H](C(=O)O)N)CN",
        ph: 7.40,
        peaks_1h: &[
            p1h!("H10/H11", 1.796, Multiplet, [], 2),
            p1h!("H12/H13", 1.932, Multiplet, [], 2),
            p1h!("H14/H15", 3.043, Triplet, [], 2),
            p1h!("H16", 3.771, Triplet, [], 1),
        ],
        shifts_13c: &[
            c13!("C1", 25.467, 1),
            c13!("C2", 30.183, 1),
            c13!("C3", 41.558, 1),
            c13!("C4", 56.803, 1),
            c13!("C5", 177.052, 1),
        ],
    },
    // 46. L-Citrulline (bmse000032, D2O, pH 7.40)
    StaticMetaboliteEntry {
        bmse_id: "bmse000032",
        name: "L-Citrulline",
        aliases: &["L-Citrulline"],
        formula: "C6H13N3O3",
        smiles: "C(C[C@@H](C(=O)O)N)CNC(=O)N",
        ph: 7.40,
        peaks_1h: &[
            p1h!("H13/H14", 1.559, Multiplet, [], 2),
            p1h!("H15/H16", 1.865, Multiplet, [], 2),
            p1h!("H17/H18", 3.134, Triplet, [], 2),
            p1h!("H19", 3.743, Triplet, [6.0], 1),
        ],
        shifts_13c: &[
            c13!("C1", 27.680, 1),
            c13!("C2", 30.459, 1),
            c13!("C3", 41.913, 1),
            c13!("C4", 57.203, 1),
            c13!("C6", 164.305, 1),
            c13!("C5", 177.428, 1),
        ],
    },
    // 47. L(-)-Carnitine (bmse000211, D2O, pH 7.40)
    StaticMetaboliteEntry {
        bmse_id: "bmse000211",
        name: "L(-)-Carnitine",
        aliases: &["L-Carnitine"],
        formula: "C7H15NO3",
        smiles: "C[N+](C)(C)C[C@@H](CC(=O)[O-])O",
        ph: 7.40,
        peaks_1h: &[
            p1h!("H21/H22", 2.425, Triplet, [], 2),
            p1h!("H12..H20(9H)", 3.215, Singlet, [], 9),
            p1h!("H23/H24", 3.419, Doublet, [], 2),
            p1h!("H25", 4.555, Multiplet, [], 1),
        ],
        shifts_13c: &[
            c13!("C4", 45.684, 1),
            c13!("C1/C2/C3", 56.798, 3),
            c13!("C6", 66.787, 1),
            c13!("C5", 73.890, 1),
            c13!("C7", 180.793, 1),
        ],
    },
    // 48. Acetyl-L-carnitine (bmse000464, D2O, pH 7.40)
    StaticMetaboliteEntry {
        bmse_id: "bmse000464",
        name: "Acetyl-L-carnitine",
        aliases: &["Acetyl-L-carnitine"],
        formula: "C9H17NO4",
        smiles: "CC(=O)O[C@H](CC(=O)[O-])C[N+](C)(C)C",
        ph: 7.40,
        peaks_1h: &[
            p1h!("H15/H16/H17", 2.129, Singlet, [], 3),
            p1h!("H28", 2.492, DoubletOfDoublets, [], 1),
            p1h!("H27", 2.632, DoubletOfDoublets, [], 1),
            p1h!("H18..H26(9H)", 3.183, Singlet, [], 9),
            p1h!("H30", 3.597, DoubletOfDoublets, [], 1),
            p1h!("H29", 3.845, DoubletOfDoublets, [], 1),
            p1h!("H31", 5.594, Quartet, [7.0], 1),
        ],
        shifts_13c: &[
            c13!("C1", 23.308, 1),
            c13!("C5", 42.910, 1),
            c13!("C2/C3/C4", 56.514, 3),
            c13!("C8", 69.582, 1),
            c13!("C6", 70.848, 1),
            c13!("C7", 175.470, 1),
            c13!("C9", 179.356, 1),
        ],
    },
    // 49. Inosine (bmse000098, D2O, pH 7.40)
    StaticMetaboliteEntry {
        bmse_id: "bmse000098",
        name: "Inosine",
        aliases: &["Inosine"],
        formula: "C10H12N4O5",
        smiles: "C1=NC(=O)C2=C(N1)N(C=N2)C3C(C(C(O3)CO)O)O",
        ph: 7.40,
        peaks_1h: &[
            p1h!("H20/H21", 3.882, Multiplet, [], 1),
            p1h!("H24", 4.278, Quartet, [2.7], 1),
            p1h!("H25", 4.439, Triplet, [3.6], 1),
            p1h!("H26", 4.752, Triplet, [4.2], 1),
            p1h!("H27", 6.066, Doublet, [4.8], 1),
            p1h!("H22/H23", 8.189, Singlet, [], 1),
            p1h!("H22/H23", 8.310, Singlet, [], 1),
        ],
        shifts_13c: &[
            c13!("C1", 64.020, 1),
            c13!("C6", 73.081, 1),
            c13!("C7", 76.776, 1),
            c13!("C4", 88.270, 1),
            c13!("C10", 91.113, 1),
            c13!("C5", 126.857, 1),
            c13!("C2/C3", 142.882, 1),
            c13!("C2/C3", 148.918, 1),
            c13!("C8", 151.077, 1),
            c13!("C9", 161.346, 1),
        ],
    },
    // 50. Sarcosine (bmse000160, D2O, pH 7.40)
    StaticMetaboliteEntry {
        bmse_id: "bmse000160",
        name: "Sarcosine",
        aliases: &["Sarcosine"],
        formula: "C3H7NO2",
        smiles: "CNCC(=O)O",
        ph: 7.40,
        peaks_1h: &[
            p1h!("H7/H8/H9", 2.725, Singlet, [], 3),
            p1h!("H10/H11", 3.601, Singlet, [], 2),
        ],
        shifts_13c: &[
            c13!("C1", 35.350, 1),
            c13!("C2", 53.515, 1),
            c13!("C3", 174.180, 1),
        ],
    },
    // 51. N,N-Dimethylglycine (bmse000241, D2O, pH 7.40)
    StaticMetaboliteEntry {
        bmse_id: "bmse000241",
        name: "N,N-Dimethylglycine",
        aliases: &["N,N-Dimethylglycine", "Dimethylglycine"],
        formula: "C4H9NO2",
        smiles: "CN(C)CC(=O)O",
        ph: 7.40,
        peaks_1h: &[
            p1h!("H10..H9(6H)", 2.917, Singlet, [], 6),
            p1h!("H14/H15", 3.716, Singlet, [], 2),
        ],
        shifts_13c: &[
            c13!("C1/C2", 46.229, 2),
            c13!("C3", 62.573, 1),
            c13!("C4", 173.105, 1),
        ],
    },
    // 52. Trimethylamine (bmse000224, D2O, pH 7.40)
    StaticMetaboliteEntry {
        bmse_id: "bmse000224",
        name: "Trimethylamine",
        aliases: &["Trimethylamine"],
        formula: "C3H9N",
        smiles: "CN(C)C",
        ph: 7.40,
        peaks_1h: &[p1h!("H10..H9(9H)", 2.879, Singlet, [], 9)],
        shifts_13c: &[c13!("C1/C2/C3", 47.359, 3)],
    },
    // 53. Allantoin (bmse000437, D2O, pH 7.40)
    StaticMetaboliteEntry {
        bmse_id: "bmse000437",
        name: "Allantoin",
        aliases: &["Allantoin"],
        formula: "C4H6N4O3",
        smiles: "C1(C(=O)NC(=O)N1)NC(=O)N",
        ph: 7.40,
        peaks_1h: &[p1h!("H12", 5.375, Singlet, [], 1)],
        shifts_13c: &[
            c13!("C1", 65.942, 1),
            c13!("C3", 162.132, 1),
            c13!("C4", 162.548, 1),
            c13!("C2", 178.950, 1),
        ],
    },
    // 54. Benzoate (bmse000300, D2O, pH 7.40)
    StaticMetaboliteEntry {
        bmse_id: "bmse000300",
        name: "Benzoate",
        aliases: &["Benzoate", "Benzoic acid"],
        formula: "C7H6O2",
        smiles: "C1=CC=C(C=C1)C(=O)O",
        ph: 7.40,
        peaks_1h: &[
            p1h!("H11/H12", 7.469, Triplet, [7.4], 2),
            p1h!("H10", 7.542, Triplet, [7.2], 1),
            p1h!("H13/H14", 7.863, Doublet, [7.2], 2),
        ],
        shifts_13c: &[
            c13!("C2/C3", 131.018, 2),
            c13!("C4/C5", 131.497, 2),
            c13!("C1", 133.971, 1),
            c13!("C6", 138.912, 1),
            c13!("C7", 178.537, 1),
        ],
    },
    // 55. Butyric acid (bmse000402, D2O, pH 7.40)
    StaticMetaboliteEntry {
        bmse_id: "bmse000402",
        name: "Butyric acid",
        aliases: &["Butyrate", "Butyric acid"],
        formula: "C4H8O2",
        smiles: "CCCC(=O)O",
        ph: 7.40,
        peaks_1h: &[
            p1h!("H7/H8/H9", 0.881, Triplet, [7.3], 3),
            p1h!("H10/H11", 1.545, Multiplet, [], 2),
            p1h!("H12/H13", 2.143, Triplet, [7.2], 2),
        ],
        shifts_13c: &[
            c13!("C1", 15.949, 1),
            c13!("C2", 22.083, 1),
            c13!("C3", 42.329, 1),
            c13!("C4", 186.894, 1),
        ],
    },
    // 56. Isobutyric acid (bmse000439, D2O, pH 7.40)
    StaticMetaboliteEntry {
        bmse_id: "bmse000439",
        name: "Isobutyric acid",
        aliases: &["Isobutyrate", "Isobutyric acid"],
        formula: "C4H8O2",
        smiles: "CC(C)C(=O)O",
        ph: 7.40,
        peaks_1h: &[
            p1h!("H10..H9(6H)", 1.049, Doublet, [7.0], 6),
            p1h!("H13", 2.374, Multiplet, [], 1),
        ],
        shifts_13c: &[
            c13!("C1/C2", 22.105, 2),
            c13!("C3", 39.672, 1),
            c13!("C4", 190.599, 1),
        ],
    },
    // 57. Nicotinamide (bmse000281, D2O, pH 7.40)
    StaticMetaboliteEntry {
        bmse_id: "bmse000281",
        name: "Nicotinamide",
        aliases: &["Nicotinamide"],
        formula: "C6H6N2O",
        smiles: "C1=CC(=CN=C1)C(=O)N",
        ph: 7.40,
        peaks_1h: &[
            p1h!("H10", 7.567, DoubletOfDoublets, [], 1),
            p1h!("H11", 8.211, Doublet, [8.4], 1),
            p1h!("H12", 8.686, Doublet, [], 1),
            p1h!("H13", 8.894, Singlet, [], 1),
        ],
        shifts_13c: &[
            c13!("C1", 126.922, 1),
            c13!("C5", 131.938, 1),
            c13!("C2", 139.152, 1),
            c13!("C4", 150.342, 1),
            c13!("C3", 154.525, 1),
            c13!("C6", 173.377, 1),
        ],
    },
];

/// Return the number of canonical aqueous metabolites in the built-in `BMRB-BMSE` index.
pub fn offline_metabolite_count() -> usize {
    OFFLINE_BMRB_METABOLITES.len()
}

/// Return all canonical aqueous metabolites in the built-in `BMRB-BMSE` compact index.
pub fn offline_metabolite_database() -> Vec<MetaboliteEntry> {
    OFFLINE_BMRB_METABOLITES
        .iter()
        .map(StaticMetaboliteEntry::to_entry)
        .collect()
}

/// Look up an offline metabolite entry by case-insensitive `bmse_id`, exact BMRB compound name,
/// or verified common biochemical alias (e.g. `"L-Lactate"` -> `bmse000208`).
pub fn lookup_offline_metabolite(query: &str) -> Option<MetaboliteEntry> {
    let q = query.trim().to_ascii_lowercase();
    if q.is_empty() {
        return None;
    }
    OFFLINE_BMRB_METABOLITES
        .iter()
        .find(|e| {
            e.bmse_id.eq_ignore_ascii_case(&q)
                || e.name.to_ascii_lowercase() == q
                || e.aliases.iter().any(|a| a.to_ascii_lowercase() == q)
        })
        .map(StaticMetaboliteEntry::to_entry)
}

/// Extract a coherent `STOCSY` peak cluster (`r >= min_r`, default `0.85`) as [`QueryPeak`]s.
///
/// Suppresses monotonically decaying Lorentzian wing buckets by retaining the driver bucket plus
/// local maxima of `stocsy.bucket_covariance` among buckets with `r >= min_r` (falling back to
/// all `r >= min_r` buckets if `bucket_covariance` is empty or flat).
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
    let has_cov = stocsy.bucket_covariance.len() == buckets.len()
        && stocsy
            .bucket_covariance
            .iter()
            .any(|&c| c.is_finite() && c > 0.0);

    let mut peaks = Vec::new();
    for (idx, bucket) in buckets.iter().enumerate() {
        let r = stocsy.bucket_correlation.get(idx).copied().unwrap_or(0.0);
        let is_driver = idx == stocsy.driver_bucket_index;
        if !r.is_finite() || (!is_driver && r < threshold) {
            continue;
        }
        if !is_driver && has_cov {
            let cov = stocsy.bucket_covariance[idx];
            if !cov.is_finite() || cov <= 0.0 {
                continue;
            }
            let left_wing = idx > 0
                && (buckets[idx - 1].center_ppm - bucket.center_ppm).abs() <= 0.08
                && stocsy.bucket_covariance[idx - 1] > cov;
            let right_wing = idx + 1 < buckets.len()
                && (buckets[idx + 1].center_ppm - bucket.center_ppm).abs() <= 0.08
                && stocsy.bucket_covariance[idx + 1] > cov;
            if left_wing || right_wing {
                continue;
            }
        }
        peaks.push(QueryPeak::from_stocsy_cluster(
            bucket.center_ppm,
            r.clamp(0.0, 1.0),
        ));
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
            if matches!(ref_mult, PeakMultiplicity::Singlet) {
                0.20
            } else {
                0.68
            }
        } else {
            let mut sum = 0.0_f64;
            let mut min_s = 1.0_f64;
            for &qj in q_j {
                let min_diff = ref_j
                    .iter()
                    .map(|&rj| (qj - rj).abs())
                    .fold(f64::INFINITY, f64::min);
                let s = (-0.5 * (min_diff / 1.5).powi(2)).exp();
                sum += s;
                min_s = min_s.min(s);
            }
            let mean_s = sum / (q_j.len() as f64);
            (0.5 * mean_s + 0.5 * min_s).clamp(0.0, 1.0)
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

/// Local chance-match probability / information-content specificity factor in `(0.0, 1.0]`.
///
/// Replaces a fixed peak-count prior with a local Poisson coincidence model:
/// 1. Around each matched reference shift `r_i`, a Gaussian kernel density estimate (`kde_other`)
///    measures local spectral crowding from *other* query peaks, yielding the local chance
///    coincidence probability `p_chance = 1 - exp(-2 * w_eff * rate_eff)` and surprisal `-ln(p_chance)`.
///    Isolated resonances (e.g. formate `8.45 ppm`, acetate `1.91 ppm`) earn high information content,
///    whereas crowded sugar-band coincidences (`3.2–4.0 ppm`) earn proportionally smaller credit.
/// 2. Multi-peak matches (`K >= 2`) must exhibit internally consistent signed shift offsets
///    (`query_ppm - ref_ppm`), tolerating global referencing offsets (e.g. `TSP` vs `DSS`) while
///    penalizing opposite-sign scattered coincidences.
/// 3. Every unmatched expected reference peak (`M - K` missing peaks, plus `unknown_unmatched` for
///    partial online search hits) incurs a false-negative log-evidence penalty (moderated only if
///    the missing shift lies in a heavily crowded region where peak coalescence is plausible), so
///    partial-match false positives (`2/4`, `3/5`, `3/4`) never bury genuine `1/1` singlets while
///    high-recall multi-peak spin systems (`L-Isoleucine 5/6`, `D-Glucose 9/9`) accumulate strong
///    additive log-evidence and outrank single-peak coincidences.
#[allow(clippy::too_many_arguments)]
fn local_chance_specificity_factor(
    ref_ppms: &[f64],
    ref_used: &[bool],
    matched: &[(usize, usize, f64, f64)],
    active_q: &[&QueryPeak],
    nucleus: MetaboliteNucleus,
    tol_ppm: f64,
    any_mj_checked: bool,
    unknown_unmatched: usize,
) -> f64 {
    let k = matched.len();
    if k == 0 || active_q.is_empty() {
        return 0.0;
    }
    let (kde_bw, span, crowd_scale) = match nucleus {
        MetaboliteNucleus::H1 => (0.12, 9.5, 10.0),
        MetaboliteNucleus::C13 => (2.40, 200.0, 0.50),
    };
    let norm_gauss = 1.0 / ((2.0 * std::f64::consts::PI).sqrt() * kde_bw);
    let n_unexplained = active_q.len().saturating_sub(k);

    let mut log_info = 0.0_f64;
    let mut sum_delta = 0.0_f64;

    for &(ri, qi, diff, mj) in matched {
        let rp = ref_ppms[ri];
        let qp = active_q[qi].ppm;
        sum_delta += qp - rp;

        let mut kde_other = 0.0_f64;
        for (idx, q_other) in active_q.iter().enumerate() {
            if idx != qi {
                let z = (q_other.ppm - rp) / kde_bw;
                if z.abs() <= 6.0 {
                    kde_other += norm_gauss * (-0.5 * z * z).exp();
                }
            }
        }
        let rate_prior = if n_unexplained == 0 {
            0.20 / span
        } else {
            0.35 * ((n_unexplained as f64) / span) + 0.35 / span
        };
        let rate_eff = kde_other + rate_prior;
        let w_eff = if any_mj_checked {
            0.22 * tol_ppm + 0.55 * diff
        } else {
            0.08 * tol_ppm + 0.85 * diff
        };
        let p_pos = 1.0 - (-2.0 * w_eff * rate_eff).exp();
        let p_chance = p_pos.clamp(1e-4, 0.96);
        let mj_weight = if any_mj_checked {
            mj.clamp(0.05, 1.0).powf(1.5)
        } else {
            1.0
        };
        let mj_nats = if any_mj_checked {
            2.2 * (mj - 0.55)
        } else {
            0.0
        };
        log_info += (-p_chance.ln()) * mj_weight + mj_nats;
    }

    if k >= 2 {
        let mean_delta = sum_delta / (k as f64);
        let mut sum_sq = 0.0_f64;
        for &(ri, qi, _, _) in matched {
            let dev = (active_q[qi].ppm - ref_ppms[ri]) - mean_delta;
            sum_sq += dev * dev;
        }
        let var_delta = sum_sq / (k as f64);
        let (spread_tol, free_band) = if any_mj_checked {
            (0.25 * tol_ppm, 0.10 * tol_ppm)
        } else {
            (0.20 * tol_ppm, 0.06 * tol_ppm)
        };
        let excess_var = (var_delta - free_band * free_band).max(0.0);
        if spread_tol > 1e-12 {
            log_info -= 0.5 * (excess_var / (spread_tol * spread_tol)) * ((k - 1) as f64);
        }
    }

    let base_miss = if any_mj_checked { 4.5 } else { 6.2 };
    for (ri, &used) in ref_used.iter().enumerate() {
        if !used {
            let rp = ref_ppms[ri];
            let mut kde_here = 0.0_f64;
            for q in active_q {
                let z = (q.ppm - rp) / kde_bw;
                if z.abs() <= 6.0 {
                    kde_here += norm_gauss * (-0.5 * z * z).exp();
                }
            }
            let crowd_relief = (kde_here / crowd_scale).clamp(0.0, 1.0);
            log_info -= base_miss - 1.8 * crowd_relief;
        }
    }
    if unknown_unmatched > 0 {
        log_info -= 3.5 * (unknown_unmatched as f64);
    }

    let (slope, midpoint) = if any_mj_checked {
        (0.30, 3.1)
    } else {
        (0.28, 3.6)
    };
    let odds_evidence = 1.0 / (1.0 + (-slope * (log_info - midpoint)).exp());
    let precision = ((k as f64) / (active_q.len() as f64)).clamp(0.0, 1.0);
    if unknown_unmatched > 0 {
        odds_evidence.clamp(0.0, 1.0)
    } else {
        (precision + (1.0 - precision) * odds_evidence).clamp(0.0, 1.0)
    }
}

/// Fast `< 5 ms` offline matcher against the built-in `BMRB-BMSE` metabolite index.
///
/// Supports both full picked 1D peak lists (`1H` or `13C`) and `STOCSY` correlated peak clusters
/// (`r >= 0.85`). Scores chemical-shift recall, query precision, Gaussian shift proximity,
/// multiplicity/`J`-coupling consistency, and local chance-match information content into a
/// composite score in `[0.0, 1.0]`.
pub fn query_offline_metabolites(
    query_peaks: &[QueryPeak],
    nucleus: MetaboliteNucleus,
    tol_ppm: f64,
    top_k: usize,
) -> Vec<MetaboliteMatch> {
    if top_k == 0 || !tol_ppm.is_finite() || tol_ppm <= 0.0 || tol_ppm > MAX_METABOLITE_TOL_PPM {
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

    // Strict 1-to-1 greedy assignment prioritizing smallest shift error and multiplicity consistency.
    pairs.sort_by(|a, b| {
        let cost_a = a.2 / tol_ppm + 0.70 * (1.0 - a.3);
        let cost_b = b.2 / tol_ppm + 0.70 * (1.0 - b.3);
        cost_a
            .total_cmp(&cost_b)
            .then_with(|| a.2.total_cmp(&b.2))
            .then_with(|| a.0.cmp(&b.0))
    });

    let mut ref_used = vec![false; n_ref];
    let mut q_used = vec![false; active_q.len()];
    let mut matched_pairs = Vec::new();
    let mut matched_tuples: Vec<(usize, usize, f64, f64)> = Vec::new();
    let mut sum_shift = 0.0_f64;
    let mut sum_mj = 0.0_f64;
    let mut prod_mj = 1.0_f64;
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
        prod_mj *= mj.max(0.05);
        sum_err += diff;
        if let Some(r) = qp.stocsy_r.filter(|v| v.is_finite()) {
            let rc = r.clamp(0.0, 1.0);
            // Weight STOCSY coherence sharply (r^4) so intramolecular spin-system peaks
            // (r >= 0.98 -> r^4 >= 0.92) decisively outrank intermolecular pathway correlations
            // (r ~ 0.88 -> r^4 ~ 0.60).
            sum_stocsy_r += rc.powi(4);
            if rc >= max_stocsy_r - 0.02 {
                matched_driver = true;
            }
        }
        if used_mj {
            any_mj_checked = true;
        }
        matched_tuples.push((ri, qi, diff, mj));
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
    let mean_mj = (sum_mj / (matched_ref as f64)).clamp(0.0, 1.0);
    let geom_mj = prod_mj.powf(1.0 / (matched_ref as f64)).clamp(0.0, 1.0);
    let mult_j_score = if any_mj_checked {
        (0.5 * mean_mj + 0.5 * geom_mj).clamp(0.0, 1.0)
    } else {
        1.0
    };
    let mean_abs_error_ppm = sum_err / (matched_ref as f64);

    let ref_ppms: Vec<f64> = entry.peaks_1h.iter().map(|p| p.ppm).collect();
    let spec_factor = local_chance_specificity_factor(
        &ref_ppms,
        &ref_used,
        &matched_tuples,
        active_q,
        MetaboliteNucleus::H1,
        tol_ppm,
        any_mj_checked,
        0,
    );

    let mut composite = recall.powf(0.80)
        * if any_mj_checked {
            0.35 + 0.30 * shift_score + 0.35 * mult_j_score
        } else if is_stocsy_mode {
            let mean_r = (sum_stocsy_r / (matched_ref as f64)).clamp(0.0, 1.0);
            0.40 + 0.36 * shift_score + 0.24 * mean_r
        } else {
            0.45 + 0.55 * shift_score
        };
    composite *= spec_factor;
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
    let mut matched_tuples: Vec<(usize, usize, f64, f64)> = Vec::new();
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
            sum_stocsy_r += rc_clamped.powi(4);
            if rc_clamped >= max_stocsy_r - 0.02 {
                matched_driver = true;
            }
        }
        matched_tuples.push((ri, qi, diff, 1.0));
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
    let ref_ppms: Vec<f64> = entry.shifts_13c.iter().map(|c| c.ppm).collect();
    let spec_factor = local_chance_specificity_factor(
        &ref_ppms,
        &ref_used,
        &matched_tuples,
        active_q,
        MetaboliteNucleus::C13,
        tol_ppm,
        false,
        0,
    );
    let mut composite = recall.powf(0.80)
        * if is_stocsy_mode {
            let mean_r = (sum_stocsy_r / (matched_ref as f64)).clamp(0.0, 1.0);
            0.40 + 0.36 * shift_score + 0.24 * mean_r
        } else {
            0.45 + 0.55 * shift_score
        };
    composite *= spec_factor;
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
// Opt-In Desktop Online Query: BMRB v2 REST API + Honest NP-MRD Web Search URL
// ============================================================================

/// Pluggable HTTPS GET transport for desktop online queries (production uses `curl -q`, unit tests
/// inject deterministic mocks without touching the network).
#[cfg(not(target_arch = "wasm32"))]
pub trait OnlineMetaboliteTransport {
    fn get_https_json(&mut self, url: &str) -> Result<String, String>;
}

/// Production desktop HTTPS client invoking `curl -q` (ignoring `~/.curlrc`) with strict host
/// allowlist, `5 s` connect/read timeout, no cross-host redirects, and a hard `4 MiB` byte cap.
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
                "-q",
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
///
/// Uses the live `BMRB v2` `shift=` query parameter (`https://api.bmrb.io/v2/search/chemical_shifts`).
pub fn build_bmrb_v2_shift_search_url(
    ppm: f64,
    nucleus: MetaboliteNucleus,
    tol_ppm: f64,
) -> Result<String, String> {
    if !ppm.is_finite()
        || !tol_ppm.is_finite()
        || tol_ppm <= 0.0
        || tol_ppm > MAX_METABOLITE_TOL_PPM
    {
        return Err(format!(
            "invalid ppm or tolerance for BMRB v2 query (tol must be in (0.0, {MAX_METABOLITE_TOL_PPM}])"
        ));
    }
    let atom_type = match nucleus {
        MetaboliteNucleus::H1 => "H",
        MetaboliteNucleus::C13 => "C",
    };
    let url = format!(
        "{BMRB_V2_API_BASE}/search/chemical_shifts?shift={ppm:.4}&threshold={tol_ppm:.4}&atom_type={atom_type}&database=metabolomics"
    );
    validate_metabolite_online_url(&url)?;
    Ok(url)
}

/// Build the strictly allowlisted `BMRB v2` compact entry metadata + full shift-table URL for
/// a single accession (e.g. `bmse000208`).
pub fn build_bmrb_v2_entry_metadata_url(bmse_id: &str) -> Result<String, String> {
    let id = bmse_id.trim();
    if id.is_empty() || id.len() > 32 || !id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
    {
        return Err(format!("invalid BMRB accession ID {bmse_id:?}"));
    }
    let url = format!(
        "{BMRB_V2_API_BASE}/entry/{id}?tag=Chem_comp.Name&tag=Chem_comp.Formula&tag=Chem_comp_SMILES.String&tag=Atom_chem_shift.Atom_ID&tag=Atom_chem_shift.Atom_type&tag=Atom_chem_shift.Val"
    );
    validate_metabolite_online_url(&url)?;
    Ok(url)
}

/// Honest `NP-MRD` (`CC-BY-4.0`) web portal URL for manual browser search.
///
/// NP-MRD does not publish an unauthenticated JSON shift-search REST endpoint, so FERRIC
/// links directly to the web portal rather than calling a nonexistent API path.
pub fn npmrd_web_search_url() -> &'static str {
    NPMRD_WEB_URL
}

fn is_placeholder_comp_id(raw: &str) -> bool {
    let t = raw.trim();
    t.is_empty() || t == "." || t == "?" || t.bytes().all(|b| b.is_ascii_digit())
}

/// Deduplicate equivalent chemical shifts (e.g. methyl protons `H7`/`H8`/`H9` at identical ppm
/// or `13C-14N` triplet lines) into distinct resonance clusters.
fn group_equivalent_shifts(
    raw_shifts: &[(String, f64)],
    merge_eps_ppm: f64,
) -> Vec<(String, f64, u8)> {
    let mut sorted: Vec<(&str, f64)> = raw_shifts
        .iter()
        .filter(|(_, p)| p.is_finite())
        .map(|(a, p)| (a.as_str(), *p))
        .collect();
    sorted.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(b.0)));

    let mut groups: Vec<(Vec<String>, f64, usize)> = Vec::new();
    for (atom, ppm) in sorted {
        if let Some(last) = groups.last_mut() {
            let mean_ppm = last.1 / (last.2 as f64);
            if (ppm - mean_ppm).abs() <= merge_eps_ppm {
                if !last.0.iter().any(|existing| existing == atom) {
                    last.0.push(atom.to_string());
                }
                last.1 += ppm;
                last.2 += 1;
                continue;
            }
        }
        groups.push((vec![atom.to_string()], ppm, 1));
    }

    groups
        .into_iter()
        .map(|(atoms, sum_ppm, count)| {
            let label = if atoms.len() <= 3 {
                atoms.join("/")
            } else {
                format!("{}..{}", atoms[0], atoms[atoms.len() - 1])
            };
            let mean_ppm = sum_ppm / (count as f64);
            (label, mean_ppm, (atoms.len().min(12) as u8).max(1))
        })
        .collect()
}

/// Score a deduplicated online shift list against `active_q` with strict 1-to-1 matching.
///
/// When `total_ref_known` is `false` (partial rows from `/v2/search/chemical_shifts`), a
/// conservative denominator `(n_matched + 3)` is used so single-atom partial hits never report
/// a spurious `recall = 1.0`.
#[allow(clippy::too_many_arguments)]
fn build_online_bmrb_match(
    bmse_id: String,
    name: String,
    formula: String,
    smiles: String,
    peaks_1h: Vec<MetabolitePeak1H>,
    shifts_13c: Vec<MetaboliteShift13C>,
    active_q: &[&QueryPeak],
    nucleus: MetaboliteNucleus,
    tol_ppm: f64,
    total_ref_known: bool,
) -> Option<MetaboliteMatch> {
    let ref_list: Vec<(&str, f64)> = match nucleus {
        MetaboliteNucleus::H1 => peaks_1h
            .iter()
            .map(|p| (p.assignment.as_str(), p.ppm))
            .collect(),
        MetaboliteNucleus::C13 => shifts_13c
            .iter()
            .map(|c| (c.assignment.as_str(), c.ppm))
            .collect(),
    };
    if ref_list.is_empty() || active_q.is_empty() {
        return None;
    }

    let mut pairs: Vec<(usize, usize, f64)> = Vec::new();
    for (ri, &(_, rppm)) in ref_list.iter().enumerate() {
        for (qi, qp) in active_q.iter().enumerate() {
            let diff = (rppm - qp.ppm).abs();
            if diff <= tol_ppm {
                pairs.push((ri, qi, diff));
            }
        }
    }
    if pairs.is_empty() {
        return None;
    }
    pairs.sort_by(|a, b| a.2.total_cmp(&b.2).then_with(|| a.0.cmp(&b.0)));

    let mut ref_used = vec![false; ref_list.len()];
    let mut q_used = vec![false; active_q.len()];
    let mut matched_pairs = Vec::new();
    let mut matched_tuples: Vec<(usize, usize, f64, f64)> = Vec::new();
    let mut sum_err = 0.0_f64;
    let mut sum_shift = 0.0_f64;

    for &(ri, qi, diff) in &pairs {
        if ref_used[ri] || q_used[qi] {
            continue;
        }
        ref_used[ri] = true;
        q_used[qi] = true;
        let (asgn, rppm) = ref_list[ri];
        let qp = active_q[qi];
        sum_err += diff;
        sum_shift += (-1.5 * (diff / tol_ppm).powi(2)).exp();
        matched_tuples.push((ri, qi, diff, 1.0));
        matched_pairs.push(MatchedPeakPair {
            assignment: asgn.to_string(),
            ref_ppm: rppm,
            query_ppm: qp.ppm,
            delta_ppm: qp.ppm - rppm,
            mult_j_consistency: 1.0,
        });
    }

    let n_matched = matched_pairs.len();
    if n_matched == 0 {
        return None;
    }
    let ref_len = ref_list.len();
    let matched_q = q_used.iter().filter(|&&u| u).count();
    let (total_ref, unknown_unmatched) = if total_ref_known {
        (ref_len.max(n_matched), 0usize)
    } else {
        // Partial search response does not report unmatched peaks of the compound;
        // do not inflate recall to 1.0.
        (n_matched + 3, 3usize)
    };
    let recall = (n_matched as f64) / (total_ref as f64);
    let precision = (matched_q as f64) / (active_q.len() as f64);
    let shift_score = (sum_shift / (n_matched as f64)).clamp(0.0, 1.0);
    let mean_abs_error_ppm = sum_err / (n_matched as f64);
    let ref_ppms: Vec<f64> = ref_list.iter().map(|&(_, p)| p).collect();
    let spec_factor = local_chance_specificity_factor(
        &ref_ppms,
        &ref_used,
        &matched_tuples,
        active_q,
        nucleus,
        tol_ppm,
        false,
        unknown_unmatched,
    );
    let raw_score = recall.powf(0.80) * (0.45 + 0.55 * shift_score);
    let score = (raw_score * spec_factor).clamp(0.0, 1.0);
    let reported_total_ref = if total_ref_known {
        ref_len.max(n_matched)
    } else {
        0
    };

    Some(MetaboliteMatch {
        entry: MetaboliteEntry {
            bmse_id,
            name,
            formula,
            smiles,
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
        total_ref_peaks: reported_total_ref,
        mean_abs_error_ppm,
        matched_pairs,
    })
}

type BmrbSearchRowGroup = (Option<String>, Vec<(String, f64)>);

/// Extract grouped `(bmse_id -> (api_name_opt, Vec<(atom, ppm)>))` from a raw `BMRB v2`
/// `/v2/search/chemical_shifts` JSON payload.
fn extract_bmrb_v2_search_rows(
    json_text: &str,
) -> Result<std::collections::BTreeMap<String, BmrbSearchRowGroup>, String> {
    use std::collections::BTreeMap;

    let root: serde_json::Value = serde_json::from_str(json_text)
        .map_err(|e| format!("failed to parse BMRB v2 JSON response: {e}"))?;

    let mut grouped: BTreeMap<String, BmrbSearchRowGroup> = BTreeMap::new();

    if let (Some(cols), Some(rows)) = (
        root.get("columns").and_then(|v| v.as_array()),
        root.get("data").and_then(|v| v.as_array()),
    ) {
        let col_names: Vec<String> = cols
            .iter()
            .filter_map(|c| c.as_str().map(|s| s.to_ascii_lowercase()))
            .collect();
        let find_col = |needles: &[&str]| -> Option<usize> {
            col_names.iter().position(|c| {
                let tail = c.rsplit('.').next().unwrap_or(c.as_str());
                needles.iter().any(|&n| c == n || tail == n)
            })
        };
        let id_col = find_col(&["entry_id", "bmse_id"]);
        // Prefer explicit compound name columns over Comp_ID (which BMRB sets to "1" in metabolomics).
        let explicit_name_col = find_col(&["compound_name", "chem_comp_name", "title", "name"]);
        let comp_id_col = find_col(&["comp_id"]);
        let val_col = find_col(&["val", "shift_val", "ppm"]);
        let atom_col = find_col(&["atom_id", "assignment"]);

        if let (Some(ic), Some(vc)) = (id_col, val_col) {
            for row in rows {
                let Some(arr) = row.as_array() else {
                    continue;
                };
                let Some(entry_id) = arr
                    .get(ic)
                    .and_then(|v| v.as_str())
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                else {
                    continue;
                };
                let shift_val = arr.get(vc).and_then(|v| {
                    v.as_f64()
                        .or_else(|| v.as_str().and_then(|s| s.parse::<f64>().ok()))
                });
                let Some(ppm) = shift_val.filter(|v| v.is_finite()) else {
                    continue;
                };
                let cname = explicit_name_col
                    .and_then(|nc| arr.get(nc))
                    .and_then(|v| v.as_str())
                    .map(str::trim)
                    .filter(|s| !is_placeholder_comp_id(s))
                    .or_else(|| {
                        comp_id_col
                            .and_then(|nc| arr.get(nc))
                            .and_then(|v| v.as_str())
                            .map(str::trim)
                            .filter(|s| !is_placeholder_comp_id(s))
                    })
                    .map(str::to_string);
                let atom = atom_col
                    .and_then(|ac| arr.get(ac))
                    .and_then(|v| v.as_str())
                    .unwrap_or("H")
                    .to_string();
                let slot = grouped
                    .entry(entry_id.to_string())
                    .or_insert_with(|| (cname.clone(), Vec::new()));
                if slot.0.is_none() && cname.is_some() {
                    slot.0 = cname;
                }
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
                .get("Title")
                .or_else(|| obj.get("compound_name"))
                .or_else(|| obj.get("name"))
                .or_else(|| obj.get("Comp_ID"))
                .and_then(|v| v.as_str())
                .map(str::trim)
                .filter(|s| !is_placeholder_comp_id(s))
                .map(str::to_string);
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
                    .or_insert_with(|| (cname.clone(), Vec::new()));
                if slot.0.is_none() && cname.is_some() {
                    slot.0 = cname;
                }
                slot.1.push((atom, ppm));
            }
        }
    }

    Ok(grouped)
}

/// Parse a `BMRB v2` REST API JSON payload (`/v2/search/chemical_shifts`) into [`MetaboliteMatch`]
/// hits using ONLY metadata returned by the API itself (never overwriting names or SMILES from
/// the offline index).
///
/// Because `/v2/search/chemical_shifts` returns only atoms matching the query window (without
/// the compound's remaining peaks), hits parsed from this response alone use a conservative
/// partial-hit recall denominator so single-atom hits never report a spurious `recall = 1.0`.
pub fn parse_bmrb_v2_response(
    json_text: &str,
    query_peaks: &[QueryPeak],
    nucleus: MetaboliteNucleus,
    tol_ppm: f64,
) -> Result<Vec<MetaboliteMatch>, String> {
    if !tol_ppm.is_finite() || tol_ppm <= 0.0 || tol_ppm > MAX_METABOLITE_TOL_PPM {
        return Err(format!(
            "invalid shift tolerance {tol_ppm} ppm (must be in (0.0, {MAX_METABOLITE_TOL_PPM}])"
        ));
    }
    let grouped = extract_bmrb_v2_search_rows(json_text)?;
    let active_q = filter_active_query_peaks(query_peaks);
    if active_q.is_empty() {
        return Ok(Vec::new());
    }
    let merge_eps = match nucleus {
        MetaboliteNucleus::H1 => 0.005,
        MetaboliteNucleus::C13 => 0.08,
    };

    let mut hits = Vec::new();
    for (bmse_id, (api_name, shifts)) in grouped {
        let deduped = group_equivalent_shifts(&shifts, merge_eps);
        let mut peaks_1h = Vec::new();
        let mut shifts_13c = Vec::new();
        for (atom, rppm, count) in deduped {
            match nucleus {
                MetaboliteNucleus::H1 => peaks_1h.push(MetabolitePeak1H {
                    assignment: atom,
                    ppm: rppm,
                    multiplicity: PeakMultiplicity::Multiplet,
                    j_hz: Vec::new(),
                    n_protons: count,
                }),
                MetaboliteNucleus::C13 => shifts_13c.push(MetaboliteShift13C {
                    assignment: atom,
                    ppm: rppm,
                    n_carbons: count,
                }),
            }
        }
        let display_name = api_name.unwrap_or_else(|| bmse_id.clone());
        if let Some(hit) = build_online_bmrb_match(
            bmse_id,
            display_name,
            String::new(),
            String::new(),
            peaks_1h,
            shifts_13c,
            &active_q,
            nucleus,
            tol_ppm,
            false,
        ) {
            hits.push(hit);
        }
    }
    Ok(hits)
}

/// Parse a `BMRB v2` entry metadata JSON payload (`/v2/entry/<bmse_id>?tag=...`) and score the
/// complete entry shift list against `query_peaks` using genuine `recall` and API-sourced
/// `Chem_comp.Name`, `Chem_comp.Formula`, and `Chem_comp_SMILES.String`.
pub fn parse_bmrb_v2_entry_metadata_and_score(
    bmse_id: &str,
    json_text: &str,
    query_peaks: &[QueryPeak],
    nucleus: MetaboliteNucleus,
    tol_ppm: f64,
) -> Result<Option<MetaboliteMatch>, String> {
    if !tol_ppm.is_finite() || tol_ppm <= 0.0 || tol_ppm > MAX_METABOLITE_TOL_PPM {
        return Err(format!(
            "invalid shift tolerance {tol_ppm} ppm (must be in (0.0, {MAX_METABOLITE_TOL_PPM}])"
        ));
    }
    let root: serde_json::Value = serde_json::from_str(json_text)
        .map_err(|e| format!("failed to parse BMRB v2 entry metadata JSON for {bmse_id}: {e}"))?;
    let entry_obj = root
        .get(bmse_id)
        .or_else(|| root.as_object().and_then(|m| m.values().next()))
        .and_then(|v| v.as_object())
        .ok_or_else(|| format!("BMRB v2 entry metadata JSON for {bmse_id} had no entry object"))?;

    let first_non_empty_str = |key: &str| -> Option<String> {
        entry_obj
            .get(key)
            .and_then(|v| v.as_array())
            .and_then(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_str())
                    .map(str::trim)
                    .find(|s| !s.is_empty() && *s != "." && *s != "?")
            })
            .map(str::to_string)
    };

    let name = first_non_empty_str("Chem_comp.Name")
        .or_else(|| first_non_empty_str("Entry.Title"))
        .unwrap_or_else(|| bmse_id.to_string());
    let formula = first_non_empty_str("Chem_comp.Formula")
        .map(|f| f.replace(' ', ""))
        .unwrap_or_default();
    let smiles = entry_obj
        .get("Chem_comp_SMILES.String")
        .and_then(|v| v.as_array())
        .map(|arr| {
            let candidates: Vec<&str> = arr
                .iter()
                .filter_map(|x| x.as_str())
                .map(str::trim)
                .filter(|s| !s.is_empty() && *s != "." && *s != "?")
                .collect();
            candidates
                .iter()
                .copied()
                .find(|s| crate::octet_mol::parse_smiles(s).is_ok())
                .or_else(|| candidates.first().copied())
                .unwrap_or("")
                .to_string()
        })
        .unwrap_or_default();

    let atom_ids = entry_obj
        .get("Atom_chem_shift.Atom_ID")
        .and_then(|v| v.as_array());
    let atom_types = entry_obj
        .get("Atom_chem_shift.Atom_type")
        .and_then(|v| v.as_array());
    let vals = entry_obj
        .get("Atom_chem_shift.Val")
        .and_then(|v| v.as_array());

    let mut raw_h: Vec<(String, f64)> = Vec::new();
    let mut raw_c: Vec<(String, f64)> = Vec::new();
    if let (Some(a_types), Some(a_vals)) = (atom_types, vals) {
        let n = a_types.len().min(a_vals.len());
        for i in 0..n {
            let atype = a_types[i].as_str().unwrap_or("").trim();
            let ppm = a_vals[i]
                .as_f64()
                .or_else(|| a_vals[i].as_str().and_then(|s| s.parse::<f64>().ok()));
            let Some(ppm) = ppm.filter(|v| v.is_finite()) else {
                continue;
            };
            let aid = atom_ids
                .and_then(|ids| ids.get(i))
                .and_then(|v| v.as_str())
                .unwrap_or(atype)
                .to_string();
            match atype {
                "H" => raw_h.push((aid, ppm)),
                "C" => raw_c.push((aid, ppm)),
                _ => {}
            }
        }
    }

    let peaks_1h: Vec<MetabolitePeak1H> = group_equivalent_shifts(&raw_h, 0.005)
        .into_iter()
        .map(|(assignment, ppm, n_protons)| MetabolitePeak1H {
            assignment,
            ppm,
            multiplicity: PeakMultiplicity::Multiplet,
            j_hz: Vec::new(),
            n_protons,
        })
        .collect();
    let shifts_13c: Vec<MetaboliteShift13C> = group_equivalent_shifts(&raw_c, 0.08)
        .into_iter()
        .map(|(assignment, ppm, n_carbons)| MetaboliteShift13C {
            assignment,
            ppm,
            n_carbons,
        })
        .collect();

    let active_q = filter_active_query_peaks(query_peaks);
    Ok(build_online_bmrb_match(
        bmse_id.to_string(),
        name,
        formula,
        smiles,
        peaks_1h,
        shifts_13c,
        &active_q,
        nucleus,
        tol_ppm,
        true,
    ))
}

/// Opt-in desktop online query using a pluggable transport (useful for deterministic unit tests).
///
/// Queries `https://api.bmrb.io/v2/search/chemical_shifts?shift=...` for up to 3 representative
/// query peaks, aggregates top candidate accessions, and fetches compact entry metadata
/// (`/v2/entry/<bmse_id>?tag=...`) so real compound names, SMILES, and full-spectrum recall
/// are reported directly from BMRB.
#[cfg(not(target_arch = "wasm32"))]
pub fn query_online_bmrb_and_npmrd_with_transport<T: OnlineMetaboliteTransport>(
    query_peaks: &[QueryPeak],
    nucleus: MetaboliteNucleus,
    tol_ppm: f64,
    sources: OnlineQuerySources,
    transport: &mut T,
) -> Result<Vec<MetaboliteMatch>, String> {
    use std::collections::{BTreeMap, BTreeSet};

    if !sources.bmrb && !sources.npmrd {
        return Err("no online metabolite sources selected".to_string());
    }
    if !sources.bmrb && sources.npmrd {
        return Err(format!(
            "NP-MRD does not expose a public JSON shift-search API; use web search at {NPMRD_WEB_URL}"
        ));
    }
    if !tol_ppm.is_finite() || tol_ppm <= 0.0 || tol_ppm > MAX_METABOLITE_TOL_PPM {
        return Err(format!(
            "invalid shift tolerance {tol_ppm} ppm (must be in (0.0, {MAX_METABOLITE_TOL_PPM}])"
        ));
    }
    let active_q = filter_active_query_peaks(query_peaks);
    if active_q.is_empty() {
        return Err("at least one finite query peak is required for online search".to_string());
    }

    let mut errors: Vec<String> = Vec::new();
    let mut bmrb_ok = false;

    // Aggregate raw candidate accessions across up to 3 queried peaks:
    // bmse_id -> (matched_query_peak_indices, min_err_ppm, fallback_api_name, raw_shifts)
    #[allow(clippy::type_complexity)]
    let mut candidates: BTreeMap<
        String,
        (BTreeSet<usize>, f64, Option<String>, Vec<(String, f64)>),
    > = BTreeMap::new();

    for (q_idx, qp) in active_q.iter().take(3).enumerate() {
        let url = build_bmrb_v2_shift_search_url(qp.ppm, nucleus, tol_ppm)?;
        match transport.get_https_json(&url) {
            Ok(body) => match extract_bmrb_v2_search_rows(&body) {
                Ok(rows_by_id) => {
                    bmrb_ok = true;
                    for (bmse_id, (api_name, shifts)) in rows_by_id {
                        let min_d = shifts
                            .iter()
                            .map(|(_, p)| (*p - qp.ppm).abs())
                            .fold(f64::INFINITY, f64::min);
                        if min_d <= tol_ppm {
                            let slot = candidates.entry(bmse_id).or_insert_with(|| {
                                (BTreeSet::new(), f64::INFINITY, None, Vec::new())
                            });
                            slot.0.insert(q_idx);
                            slot.1 = slot.1.min(min_d);
                            if slot.2.is_none() && api_name.is_some() {
                                slot.2 = api_name;
                            }
                            slot.3.extend(shifts);
                        }
                    }
                }
                Err(e) => errors.push(format!("BMRB v2 parse error: {e}")),
            },
            Err(e) => errors.push(format!("BMRB v2 network error: {e}")),
        }
    }

    if !bmrb_ok {
        return Err(format!(
            "BMRB v2 online metabolite query failed (offline or unreachable): {}",
            errors.join("; ")
        ));
    }

    // Rank candidates by distinct matched query peaks DESC, then smallest shift error ASC.
    let mut ranked_ids: Vec<(String, usize, f64)> = candidates
        .iter()
        .map(|(id, (qset, min_d, _, _))| (id.clone(), qset.len(), *min_d))
        .collect();
    ranked_ids.sort_by(|a, b| {
        b.1.cmp(&a.1)
            .then_with(|| a.2.total_cmp(&b.2))
            .then_with(|| a.0.cmp(&b.0))
    });
    ranked_ids.truncate(8);

    let merge_eps = match nucleus {
        MetaboliteNucleus::H1 => 0.005,
        MetaboliteNucleus::C13 => 0.08,
    };
    let mut combined: Vec<MetaboliteMatch> = Vec::new();

    for (bmse_id, _, _) in ranked_ids {
        let mut resolved = false;
        if let Ok(meta_url) = build_bmrb_v2_entry_metadata_url(&bmse_id) {
            if let Ok(meta_body) = transport.get_https_json(&meta_url) {
                if let Ok(Some(hit)) = parse_bmrb_v2_entry_metadata_and_score(
                    &bmse_id,
                    &meta_body,
                    query_peaks,
                    nucleus,
                    tol_ppm,
                ) {
                    combined.push(hit);
                    resolved = true;
                }
            }
        }
        if !resolved {
            if let Some((_, _, api_name, shifts)) = candidates.get(&bmse_id) {
                let deduped = group_equivalent_shifts(shifts, merge_eps);
                let mut peaks_1h = Vec::new();
                let mut shifts_13c = Vec::new();
                for (atom, rppm, count) in deduped {
                    match nucleus {
                        MetaboliteNucleus::H1 => peaks_1h.push(MetabolitePeak1H {
                            assignment: atom,
                            ppm: rppm,
                            multiplicity: PeakMultiplicity::Multiplet,
                            j_hz: Vec::new(),
                            n_protons: count,
                        }),
                        MetaboliteNucleus::C13 => shifts_13c.push(MetaboliteShift13C {
                            assignment: atom,
                            ppm: rppm,
                            n_carbons: count,
                        }),
                    }
                }
                let display_name = api_name.clone().unwrap_or_else(|| bmse_id.clone());
                if let Some(hit) = build_online_bmrb_match(
                    bmse_id,
                    display_name,
                    String::new(),
                    String::new(),
                    peaks_1h,
                    shifts_13c,
                    &active_q,
                    nucleus,
                    tol_ppm,
                    false,
                ) {
                    combined.push(hit);
                }
            }
        }
    }

    combined.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| b.matched_ref_peaks.cmp(&a.matched_ref_peaks))
            .then_with(|| a.mean_abs_error_ppm.total_cmp(&b.mean_abs_error_ppm))
            .then_with(|| a.entry.bmse_id.cmp(&b.entry.bmse_id))
    });
    Ok(combined)
}

/// Opt-in desktop online metabolite search querying `BMRB v2` (`https://api.bmrb.io/v2/...`).
///
/// Strictly enforces HTTPS host allowlist (`api.bmrb.io`, `np-mrd.org` only), invokes `curl -q`
/// (ignoring `~/.curlrc`), enforces `5 s` connect/read timeout and `4 MiB` response cap, disables
/// cross-host redirects, and fails closed with `Err` when offline. Never invoked automatically
/// without explicit user opt-in. For NP-MRD, use [`npmrd_web_search_url`].
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
