//! Pseudo-3D relaxation & chemical-exchange plane series integrator (`T13.1`, `docs/specs/16-dynamics-and-csp.md` §1).
//!
//! Decodes Bruker (`vdlist`, `vclist`, `fqlist` + `ser`/`acqus`/`acqu2s`/`acqu3s`) and Varian/Agilent
//! (`procpar` arrayed `d2`/`relaxT`/`ncyc`/`satfrq`/`tof` + `fid`) pseudo-3D relaxation and exchange
//! series in both plane-queued (`[plane][t1][quad][t2]`) and $t_1$-interleaved
//! (`[t1][plane][quad][t2]` and `[t1][quad][plane][t2]`) acquisition loop orders, as well as
//! pre-processed `Vec<Spectrum2D>` stacks.
//!
//! Processes all 2D planes with a locked `ProcessingRecipe`, fits shared 2D Gaussian / pseudo-Voigt
//! footprints $(\delta_H, \delta_X, \sigma_H, \sigma_X, \eta_H, \eta_X)$ on the high-SNR reference
//! stack, extracts unbiased linear least-squares amplitudes across all planes (eliminating positive
//! low-SNR local-maximum noise-picking bias), estimates uncertainties via 2D MAD with spatial
//! noise-autocorrelation and duplicate-delay replicate variance pooling, and exports/imports clean
//! ASCII TSV and NEF v1.1 relaxation saveframes.
//!
//! Provenance: Clean-room implementation strictly from peer-reviewed NMR literature and public
//! format specifications (zero external GPL/academic tool code consulted; zero external product
//! names in identifiers):
//! - Farrow, N. A., Muhandiram, R., Singer, A. U., Pascal, S. M., Kay, C. M., Gish, G.,
//!   Shoelson, S. E., Pawson, T., Forman-Kay, J. D. & Kay, L. E. (*Biochemistry* **1994**, 33,
//!   5984–6003) — $^{15}\text{N}$ $T_1$, $T_2$, and $\{^1\text{H}\}\text{–}^{15}\text{N}$ NOE pulse
//!   schemes, crosspeak integration, and duplicate-plane uncertainty propagation.
//! - Farrow, N. A., Zhang, O., Szabo, A., Torchia, D. A. & Kay, L. E. (*J. Biomol. NMR* **1995**,
//!   6, 153–162) — reduced spectral density mapping & relaxation data reduction.
//! - Kay, L. E., Torchia, D. A. & Bax, A. (*Biochemistry* **1989**, 28, 8972–8979) — backbone
//!   $^{15}\text{N}$ relaxation measurements & interleaved acquisition principles.
//! - Lipari, G. & Szabo, A. (*J. Am. Chem. Soc.* **1982**, 104, 4546–4559 & 4559–4570).
//! - Clore, G. M., Szabo, A., Bax, A., Kay, L. E., Driscoll, P. C. & Gronenborn, A. M.
//!   (*J. Am. Chem. Soc.* **1990**, 112, 4989–4991).
//! - Mandel, A. M., Akke, M. & Palmer, A. G. III (*J. Mol. Biol.* **1995**, 246, 144–163).
//! - Woessner, D. E. (*J. Chem. Phys.* **1962**, 37, 647–654).
//! - d'Auvergne, E. J. & Gooley, P. R. (*J. Biomol. NMR* **2003**, 25, 25–39).
//! - Korzhnev, D. M., Billeter, M., Arseniev, A. S. & Orekhov, V. Y. (*Prog. Nucl. Magn. Reson.
//!   Spectrosc.* **2001**, 38, 197–266) — quantitative lineshape/footprint integration vs peak-height bias.
//! - Public format specifications: Bruker JCAMP-DX v5.0 (`acqus`, `vdlist`, `vclist`, `fqlist`, `ser`),
//!   Varian/Agilent VnmrJ `procpar` & binary `fid`, and CCPN NEF v1.1 (Gutmanas et al.,
//!   *Nat. Struct. Mol. Biol.* **2015**, 22, 433–434).

use std::collections::{BTreeMap, HashMap};
use std::f64::consts::PI;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::fid::{
    bruker_dsp_group_delay, infer_bruker_raw_indirect_reverse, Complex64, Fid2D, FidAxisMeta,
    FidVendor, QuadMode, StoredDimProc, StoredWindowHint,
};
use crate::process::{process_fid_2d, process_fid_2d_mut, PhaseSource, ProcessingRecipe};
use crate::spectrum::{AxisPpm, Spectrum2D};

/// Gaussian MAD normalization factor $1 / \Phi^{-1}(3/4) = 1.4826022185$.
pub const MAD_TO_SIGMA: f64 = 1.482_602_218_505_602;

/// Relaxation or chemical-exchange experiment class for a pseudo-3D series.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelaxExperimentKind {
    R1,
    R2,
    R1rho,
    HetNoe,
    Cpmg,
    Cest,
    Custom,
}

impl RelaxExperimentKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::R1 => "R1",
            Self::R2 => "R2",
            Self::R1rho => "R1rho",
            Self::HetNoe => "hetNOE",
            Self::Cpmg => "CPMG",
            Self::Cest => "CEST",
            Self::Custom => "custom",
        }
    }

    pub fn parse_label(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "r1" | "t1" => Ok(Self::R1),
            "r2" | "t2" => Ok(Self::R2),
            "r1rho" | "t1rho" => Ok(Self::R1rho),
            "hetnoe" | "noe" | "hnoe" => Ok(Self::HetNoe),
            "cpmg" => Ok(Self::Cpmg),
            "cest" | "dest" => Ok(Self::Cest),
            "custom" => Ok(Self::Custom),
            other => Err(format!("unrecognized relaxation experiment kind '{other}'")),
        }
    }
}

/// Frequency offset entry parsed from Bruker `fqlist` or Varian `satfrq` / `tof`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FqOffset {
    /// Offset from the indirect channel carrier (`SFO` / `dof2`) in Hz.
    pub offset_hz: f64,
    /// Absolute chemical shift position of the saturation/spin-lock field in ppm.
    pub offset_ppm: f64,
}

/// Physical independent-variable schedule across planes of a pseudo-3D experiment.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RelaxSchedule {
    /// Variable relaxation delays in seconds (`vdlist`, Varian `relaxT` / `d2`).
    VariableDelaysSec { delays_sec: Vec<f64> },
    /// Integer CPMG / spin-lock loop counts multiplied by a unit cycle time (`vclist`, Varian `ncyc`).
    LoopCounts {
        counts: Vec<u32>,
        unit_cycle_sec: f64,
        values_sec: Vec<f64>,
    },
    /// RF saturation / spin-lock frequency offsets (`fqlist`, Varian `satfrq` / `tof`).
    FrequencyOffsets { offsets: Vec<FqOffset> },
}

impl RelaxSchedule {
    /// Number of 2D planes in the schedule.
    pub fn len(&self) -> usize {
        match self {
            Self::VariableDelaysSec { delays_sec } => delays_sec.len(),
            Self::LoopCounts { counts, .. } => counts.len(),
            Self::FrequencyOffsets { offsets } => offsets.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Primary scalar independent variable per plane (`seconds` for delays/loops, `offset_hz` for `fqlist`).
    pub fn primary_values(&self) -> Vec<f64> {
        match self {
            Self::VariableDelaysSec { delays_sec } => delays_sec.clone(),
            Self::LoopCounts { values_sec, .. } => values_sec.clone(),
            Self::FrequencyOffsets { offsets } => offsets.iter().map(|o| o.offset_hz).collect(),
        }
    }

    /// Physical unit label for `primary_values()`.
    pub fn unit_label(&self) -> &'static str {
        match self {
            Self::VariableDelaysSec { .. } | Self::LoopCounts { .. } => "s",
            Self::FrequencyOffsets { .. } => "Hz",
        }
    }

    /// Short schedule kind tag for TSV/NEF serialization.
    pub fn schedule_tag(&self) -> &'static str {
        match self {
            Self::VariableDelaysSec { .. } => "vdlist",
            Self::LoopCounts { .. } => "vclist",
            Self::FrequencyOffsets { .. } => "fqlist",
        }
    }
}

/// Acquisition loop nesting order for a raw pseudo-3D time-domain dataset (`ser` or `fid`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Pseudo3DInterleaving {
    /// Plane-queued `[plane][t1][quad][t2]`: each 2D plane is acquired completely before stepping to the next delay/offset.
    PlaneQueued,
    /// $t_1$-interleaved `[t1][plane][quad][t2]`: for each $t_1$ increment, steps across planes with quadrature inner-most (e.g. Varian `array = "relaxT,phase"`).
    T1PlaneQuad,
    /// $t_1$-interleaved `[t1][quad][plane][t2]`: for each $t_1$ increment and quadrature component, steps across all planes inner-most (e.g. Bruker inner `vd` loop or Varian `array = "phase,relaxT"`).
    T1QuadPlane,
}

/// 2D lineshape model used for locked-footprint crosspeak integration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FootprintShapeModel {
    /// Separable 2D Gaussian $(\eta_H = \eta_X = 0)$.
    Gaussian,
    /// Separable 2D pseudo-Voigt $(1 - \eta) G(\Delta\delta; \sigma) + \eta L(\Delta\delta; \sqrt{2\ln 2}\,\sigma)$.
    PseudoVoigt,
}

/// Target crosspeak seed (`1H`, `15N`/`13C`) for series integration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TargetPeakSeed {
    pub seq_id: i32,
    pub res_name: String,
    pub atom_h: String,
    pub atom_x: String,
    pub delta_h_ppm: f64,
    pub delta_x_ppm: f64,
}

/// Fitted shared 2D crosspeak footprint locked across all planes in a pseudo-3D series.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LockedPeakFootprint {
    pub seq_id: i32,
    pub res_name: String,
    pub atom_h: String,
    pub atom_x: String,
    /// Refined centroid along the direct axis (`1H`, ppm).
    pub delta_h_ppm: f64,
    /// Refined centroid along the indirect axis (`15N`/`13C`, ppm).
    pub delta_x_ppm: f64,
    /// Gaussian-equivalent width $\sigma_H$ in ppm ($\mathrm{FWHM}_H = 2\sqrt{2\ln 2}\,\sigma_H$).
    pub sigma_h_ppm: f64,
    /// Gaussian-equivalent width $\sigma_X$ in ppm ($\mathrm{FWHM}_X = 2\sqrt{2\ln 2}\,\sigma_X$).
    pub sigma_x_ppm: f64,
    /// Lorentzian mixing fraction $\eta_H \in [0, 1]$ along direct axis (`0` for pure Gaussian).
    pub eta_h: f64,
    /// Lorentzian mixing fraction $\eta_X \in [0, 1]$ along indirect axis (`0` for pure Gaussian).
    pub eta_x: f64,
    /// Overlap cluster index (`peaks` sharing overlapping pixel patches are solved jointly).
    pub overlap_cluster_id: usize,
}

impl LockedPeakFootprint {
    /// Analytical 2D integral factor (`ppm^2`) converting unit-height amplitude $A$ into peak volume.
    pub fn unit_volume_factor_ppm2(&self) -> f64 {
        voigt_1d_area(self.sigma_h_ppm, self.eta_h) * voigt_1d_area(self.sigma_x_ppm, self.eta_x)
    }
}

/// Configuration for locked-footprint pseudo-3D plane integration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RelaxIntegrationConfig {
    /// Lineshape family (`Gaussian` or `PseudoVoigt`).
    pub shape_model: FootprintShapeModel,
    /// Half-width of local integration patch in grid points along `(direct, indirect)` (default `(4, 4)`).
    pub patch_half_width_pts: [usize; 2],
    /// Maximum centroid refinement shift `(direct_ppm, indirect_ppm)` on the reference stack (default `(0.035, 0.30)`).
    pub max_centroid_shift_ppm: [f64; 2],
    /// Optional explicit reference plane indices for footprint fitting (`None` = auto high-SNR plane(s)).
    pub ref_plane_indices: Option<Vec<usize>>,
    /// Whether to include a per-cluster constant baseline offset column in the linear least-squares solve.
    pub fit_local_baseline: bool,
    /// Tolerance on primary schedule variable `|x_a - x_b|` for grouping duplicate planes (default `1e-7`).
    pub duplicate_tol: f64,
    /// Estimate spatial pixel noise autocovariance from background regions (accounts for FT zero-filling & apodization).
    pub account_spatial_noise_corr: bool,
}

impl Default for RelaxIntegrationConfig {
    fn default() -> Self {
        Self {
            shape_model: FootprintShapeModel::PseudoVoigt,
            patch_half_width_pts: [4, 4],
            max_centroid_shift_ppm: [0.035, 0.30],
            ref_plane_indices: None,
            fit_local_baseline: false,
            duplicate_tol: 1.0e-7,
            account_spatial_noise_corr: true,
        }
    }
}

/// Extracted intensity and uncertainty for one residue across all planes of a relaxation/exchange series.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResidueRelaxTrace {
    pub footprint: LockedPeakFootprint,
    /// Linear least-squares peak-height amplitudes $\hat{A}_{r,p}$ across planes `0..n_planes`.
    pub intensities: Vec<f64>,
    /// Propagated $1\sigma$ amplitude uncertainties across planes `0..n_planes`.
    pub sigmas: Vec<f64>,
    /// Analytical pseudo-Voigt/Gaussian integrated volumes across planes `0..n_planes`.
    pub volumes: Vec<f64>,
}

/// Complete integrated pseudo-3D relaxation or exchange dataset ready for curve fitting (`T13.2`–`T13.5`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RelaxSeriesDataset {
    pub name: String,
    pub experiment_kind: RelaxExperimentKind,
    /// Proton spectrometer observation frequency in MHz (e.g. `600.0`, `800.0`).
    pub field_mhz: f64,
    pub schedule: RelaxSchedule,
    /// Per-plane robust 2D MAD pixel noise $\sigma_{\text{mad}, p}$.
    pub plane_noise_mad: Vec<f64>,
    /// Boolean flag marking planes belonging to a duplicate-delay / duplicate-offset group ($m_g \ge 2$).
    pub plane_is_duplicate: Vec<bool>,
    /// Empirical variance-ratio scale $\hat{\kappa}$ estimated from duplicate planes (`1.0` if no duplicates).
    pub duplicate_variance_scale: f64,
    /// Total duplicate degrees of freedom $\nu_{\text{dup}} = \sum_{g,r} (m_g - 1)$.
    pub duplicate_dof: usize,
    pub traces: Vec<ResidueRelaxTrace>,
}

/// Decoded raw time-domain pseudo-3D dataset before 2D Fourier processing.
#[derive(Clone, Debug)]
pub struct RawPseudo3D {
    pub name: String,
    pub vendor: FidVendor,
    pub experiment_kind: RelaxExperimentKind,
    pub interleaving: Pseudo3DInterleaving,
    pub schedule: RelaxSchedule,
    pub planes: Vec<Fid2D>,
}

/// Parsed Varian `procpar` relaxation schedule and inferred loop order.
#[derive(Clone, Debug, PartialEq)]
pub struct VarianRelaxSchedule {
    pub param_name: String,
    pub experiment_kind: RelaxExperimentKind,
    pub schedule: RelaxSchedule,
    pub interleaving: Pseudo3DInterleaving,
}

// ============================================================================
// Schedule Parsers (`vdlist`, `vclist`, `fqlist`, Varian `procpar`)
// ============================================================================

/// Strip trailing `#` or `;` comments from a schedule line.
fn strip_schedule_comment(line: &str) -> &str {
    let s = line.split('#').next().unwrap_or("");
    s.split(';').next().unwrap_or("").trim()
}

/// Parse a single Bruker delay token with optional unit suffix (`s`, `ms`/`m`, `us`/`u`, `ns`/`n`).
pub fn parse_bruker_delay_token_sec(tok: &str) -> Result<f64, String> {
    let t = tok.trim();
    if t.is_empty() {
        return Err("empty delay token".to_string());
    }
    let lower = t.to_ascii_lowercase();
    let (num_str, mult) = if let Some(prefix) = lower.strip_suffix("ms") {
        (prefix, 1.0e-3)
    } else if let Some(prefix) = lower.strip_suffix("us") {
        (prefix, 1.0e-6)
    } else if let Some(prefix) = lower.strip_suffix("ns") {
        (prefix, 1.0e-9)
    } else if let Some(prefix) = lower.strip_suffix('s') {
        (prefix, 1.0)
    } else if let Some(prefix) = lower.strip_suffix('m') {
        (prefix, 1.0e-3)
    } else if let Some(prefix) = lower.strip_suffix('u') {
        (prefix, 1.0e-6)
    } else if let Some(prefix) = lower.strip_suffix('n') {
        (prefix, 1.0e-9)
    } else {
        (lower.as_str(), 1.0)
    };

    let val: f64 = num_str
        .trim()
        .parse()
        .map_err(|e| format!("invalid Bruker delay token '{t}': {e}"))?;
    if !val.is_finite() || val < 0.0 {
        return Err(format!(
            "Bruker delay token '{t}' must be finite and non-negative"
        ));
    }
    Ok(val * mult)
}

/// Parse a Bruker `vdlist` variable-delay text schedule into seconds.
pub fn parse_bruker_vdlist(text: &str) -> Result<Vec<f64>, String> {
    let mut out = Vec::new();
    for (line_no, raw) in text.lines().enumerate() {
        let clean = strip_schedule_comment(raw);
        if clean.is_empty() {
            continue;
        }
        for tok in clean.split_whitespace() {
            let sec = parse_bruker_delay_token_sec(tok)
                .map_err(|e| format!("vdlist line {}: {e}", line_no + 1))?;
            out.push(sec);
        }
    }
    if out.is_empty() {
        return Err("vdlist schedule contains zero delay entries".to_string());
    }
    Ok(out)
}

/// Parse a Bruker `vclist` variable loop-counter text schedule, multiplying each integer count by
/// `unit_cycle_sec` (`>= 0`).
pub fn parse_bruker_vclist(
    text: &str,
    unit_cycle_sec: f64,
) -> Result<(Vec<u32>, Vec<f64>), String> {
    if !unit_cycle_sec.is_finite() || unit_cycle_sec < 0.0 {
        return Err(format!(
            "invalid vclist unit_cycle_sec={unit_cycle_sec}; must be finite and >= 0"
        ));
    }
    let mut counts = Vec::new();
    let mut secs = Vec::new();
    for (line_no, raw) in text.lines().enumerate() {
        let clean = strip_schedule_comment(raw);
        if clean.is_empty() {
            continue;
        }
        for tok in clean.split_whitespace() {
            let c: u32 = tok.parse().map_err(|e| {
                format!(
                    "invalid integer in vclist line {} ('{tok}'): {e}",
                    line_no + 1
                )
            })?;
            counts.push(c);
            secs.push((c as f64) * unit_cycle_sec);
        }
    }
    if counts.is_empty() {
        return Err("vclist schedule contains zero loop-counter entries".to_string());
    }
    Ok((counts, secs))
}

/// Convert integer CPMG loop counts (`vclist` / Varian `ncyc`) into effective CPMG pulsing
/// frequencies $\nu_{\text{CPMG}} = \frac{n_{\text{cyc}} \cdot n_{\text{echo}}}{2 T_{\text{relax}}}$ (Hz).
///
/// For a standard symmetric constant-time CPMG element $(\tau_{\text{cp}} - 180^\circ - 2\tau_{\text{cp}} - 180^\circ - \tau_{\text{cp}})_{n_{\text{cyc}}}$
/// containing `echoes_per_loop = 2` $180^\circ$ pulses per loop across total time $T_{\text{relax}}$,
/// $\nu_{\text{CPMG}} = n_{\text{cyc}} / T_{\text{relax}}$ (`0.0` when `n_cyc == 0` for the reference $I_0$ plane).
pub fn cpmg_frequencies_from_vclist(
    counts: &[u32],
    t_relax_sec: f64,
    echoes_per_loop: u32,
) -> Result<Vec<f64>, String> {
    if !t_relax_sec.is_finite() || t_relax_sec <= 0.0 {
        return Err(format!(
            "invalid CPMG t_relax_sec={t_relax_sec}; must be finite and > 0"
        ));
    }
    if echoes_per_loop == 0 {
        return Err("CPMG echoes_per_loop must be >= 1".to_string());
    }
    Ok(counts
        .iter()
        .map(|&c| ((c as f64) * (echoes_per_loop as f64)) / (2.0 * t_relax_sec))
        .collect())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FqRefMode {
    SfoHz,
    SfoPpm,
    BfHz,
    BfPpm,
}

/// Parse a Bruker `fqlist` (`fq1list`/`fq2list`/`fq3list`) frequency-offset schedule for CEST /
/// off-resonance $R_{1\rho}$, supporting TopSpin header directives (`sfo hz`, `sfo ppm`, `bf hz`,
/// `bf ppm`, `p`/`ppm`, `h`/`hz`) as well as per-token suffixes (`Hz`, `kHz`, `ppm`, `p`).
pub fn parse_bruker_fqlist(
    text: &str,
    obs_mhz: f64,
    carrier_ppm: f64,
) -> Result<Vec<FqOffset>, String> {
    if !obs_mhz.is_finite() || obs_mhz <= 0.0 {
        return Err(format!(
            "invalid obs_mhz={obs_mhz} for fqlist; must be finite and > 0"
        ));
    }
    if !carrier_ppm.is_finite() {
        return Err(format!("invalid carrier_ppm={carrier_ppm} for fqlist"));
    }

    let mut mode = FqRefMode::SfoHz;
    let mut out = Vec::new();

    for (line_no, raw) in text.lines().enumerate() {
        let clean = strip_schedule_comment(raw);
        if clean.is_empty() {
            continue;
        }
        let low = clean.to_ascii_lowercase();
        match low.as_str() {
            "sfo hz" | "sfo h" | "h" | "hz" => {
                mode = FqRefMode::SfoHz;
                continue;
            }
            "sfo ppm" | "sfo p" => {
                mode = FqRefMode::SfoPpm;
                continue;
            }
            "bf hz" | "bf h" => {
                mode = FqRefMode::BfHz;
                continue;
            }
            "bf ppm" | "bf p" | "p" | "ppm" => {
                mode = FqRefMode::BfPpm;
                continue;
            }
            _ => {}
        }

        for tok in clean.split_whitespace() {
            let t_low = tok.to_ascii_lowercase();
            let (num_str, tok_mode, scale) = if let Some(p) = t_low.strip_suffix("khz") {
                (p, FqRefMode::SfoHz, 1.0e3)
            } else if let Some(p) = t_low.strip_suffix("hz") {
                (p, mode_if_hz(mode), 1.0)
            } else if let Some(p) = t_low.strip_suffix("ppm") {
                (p, mode_if_ppm(mode), 1.0)
            } else if let Some(p) = t_low.strip_suffix('p') {
                (p, mode_if_ppm(mode), 1.0)
            } else {
                (t_low.as_str(), mode, 1.0)
            };

            let val: f64 = num_str.trim().parse().map_err(|e| {
                format!(
                    "invalid frequency offset in fqlist line {} ('{tok}'): {e}",
                    line_no + 1
                )
            })?;
            if !val.is_finite() {
                return Err(format!(
                    "non-finite frequency offset in fqlist line {}",
                    line_no + 1
                ));
            }
            let v = val * scale;
            let (offset_hz, offset_ppm) = match tok_mode {
                FqRefMode::SfoHz => (v, carrier_ppm + v / obs_mhz),
                FqRefMode::SfoPpm => (v * obs_mhz, carrier_ppm + v),
                FqRefMode::BfHz => {
                    let ppm = v / obs_mhz;
                    ((ppm - carrier_ppm) * obs_mhz, ppm)
                }
                FqRefMode::BfPpm => ((v - carrier_ppm) * obs_mhz, v),
            };
            out.push(FqOffset {
                offset_hz,
                offset_ppm,
            });
        }
    }

    if out.is_empty() {
        return Err("fqlist schedule contains zero frequency offsets".to_string());
    }
    Ok(out)
}

#[inline]
fn mode_if_hz(cur: FqRefMode) -> FqRefMode {
    match cur {
        FqRefMode::BfHz => FqRefMode::BfHz,
        _ => FqRefMode::SfoHz,
    }
}

#[inline]
fn mode_if_ppm(cur: FqRefMode) -> FqRefMode {
    match cur {
        FqRefMode::SfoPpm => FqRefMode::SfoPpm,
        _ => FqRefMode::BfPpm,
    }
}

// ============================================================================
// Internal Bruker JCAMP-DX & Varian `procpar` Minimal Helpers
// ============================================================================

fn parse_jcamp_params_str(text: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    let lines: Vec<&str> = text.lines().collect();
    let mut i = 0usize;
    while i < lines.len() {
        let line = lines[i].trim();
        i += 1;
        let Some(rest) = line.strip_prefix("##$") else {
            continue;
        };
        let Some((key, val_raw)) = rest.split_once('=') else {
            continue;
        };
        let key = key.trim().to_string();
        let val_trim = val_raw.trim();
        if val_trim.starts_with('(') {
            // Array value on subsequent line(s), e.g. `(0..7)`
            let mut arr_vals = Vec::new();
            if let Some((_, after_paren)) = val_trim.split_once(')') {
                let tail = after_paren.trim();
                if !tail.is_empty() {
                    arr_vals.push(tail.to_string());
                }
            }
            while i < lines.len() {
                let next = lines[i].trim();
                if next.starts_with("##") || next.starts_with("$$") {
                    break;
                }
                if !next.is_empty() {
                    arr_vals.push(next.to_string());
                }
                i += 1;
            }
            map.insert(key, arr_vals.join(" "));
        } else {
            let cleaned = val_trim
                .trim_start_matches('<')
                .trim_end_matches('>')
                .trim()
                .to_string();
            map.insert(key, cleaned);
        }
    }
    map
}

fn read_jcamp_file_opt(dir: &Path, name: &str) -> Option<HashMap<String, String>> {
    let p = find_file_ci_local(dir, name)?;
    let txt = fs::read_to_string(p).ok()?;
    Some(parse_jcamp_params_str(&txt))
}

fn read_merged_jcamp_local(
    dir: &Path,
    primary: &str,
    fallback: &str,
) -> Result<HashMap<String, String>, String> {
    let mut out = HashMap::new();
    if let Some(m) = read_jcamp_file_opt(dir, fallback) {
        out.extend(m);
    }
    if let Some(m) = read_jcamp_file_opt(dir, primary) {
        out.extend(m);
    }
    if out.is_empty() {
        return Err(format!(
            "missing Bruker parameter file {primary} (or {fallback}) in {}",
            dir.display()
        ));
    }
    Ok(out)
}

fn find_file_ci_local(dir: &Path, target: &str) -> Option<std::path::PathBuf> {
    let direct = dir.join(target);
    if direct.is_file() {
        return Some(direct);
    }
    let entries = fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_file()
            && p.file_name()
                .and_then(|s| s.to_str())
                .is_some_and(|n| n.eq_ignore_ascii_case(target))
        {
            return Some(p);
        }
    }
    None
}

fn normalize_nuc(raw: &str, fallback: &str) -> String {
    let s = raw
        .trim()
        .trim_matches('<')
        .trim_matches('>')
        .trim()
        .to_ascii_uppercase();
    if s.is_empty() || s == "OFF" || s == "NONE" {
        return fallback.to_string();
    }
    if s == "H1" || s == "1H" || s == "H" {
        "1H".to_string()
    } else if s == "N15" || s == "15N" || s == "N" {
        "15N".to_string()
    } else if s == "C13" || s == "13C" || s == "C" {
        "13C".to_string()
    } else {
        s
    }
}

fn extract_stored_proc_jcamp(proc_map: Option<&HashMap<String, String>>) -> StoredDimProc {
    let Some(p) = proc_map else {
        return StoredDimProc::default();
    };
    let p0_deg = p.get("PHC0").and_then(|s| s.parse::<f64>().ok());
    let p1_deg = p.get("PHC1").and_then(|s| s.parse::<f64>().ok());
    let si = p
        .get("SI")
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v >= 2);
    let td_eff = p
        .get("TDeff")
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 0);
    let fcor = p
        .get("FCOR")
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0);
    let reverse = p.get("REVERSE").is_some_and(|v| {
        v.eq_ignore_ascii_case("yes") || v == "1" || v.eq_ignore_ascii_case("true")
    });
    let bc_mod = p.get("BC_mod").and_then(|s| s.parse::<i32>().ok());
    let bcfw_ppm = p.get("BCFW").and_then(|s| s.parse::<f64>().ok());

    let window = p.get("WDW").and_then(|s| s.parse::<i32>().ok()).map(|wdw| {
        let ssb = p
            .get("SSB")
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(2.0);
        let offset_pi_frac = if ssb >= 1.0 { 1.0 / ssb } else { 0.0 };
        match wdw {
            0 => StoredWindowHint::None,
            1 => StoredWindowHint::Exponential {
                lb_hz: p
                    .get("LB")
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0)
                    .max(0.0),
            },
            2 => StoredWindowHint::SineBell { offset_pi_frac },
            3 => StoredWindowHint::SquaredSineBell { offset_pi_frac },
            _ => StoredWindowHint::SquaredSineBell {
                offset_pi_frac: 0.5,
            },
        }
    });

    StoredDimProc {
        p0_deg,
        p1_deg,
        window,
        si,
        td_eff,
        fcor,
        reverse,
        bc_mod,
        bcfw_ppm,
    }
}

fn bruker_first_ppm(
    proc_map: Option<&HashMap<String, String>>,
    sfo1: f64,
    obs_mhz: f64,
    sw_hz: f64,
) -> f64 {
    if let Some(off) = proc_map
        .and_then(|p| p.get("OFFSET"))
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|v| v.is_finite())
    {
        return off;
    }
    ((sfo1 - obs_mhz) * 1.0e6 + 0.5 * sw_hz) / obs_mhz.max(1.0e-9)
}

#[derive(Clone, Debug)]
enum LocalProcparVal {
    Numbers(Vec<f64>),
    Strings(Vec<String>),
}

impl LocalProcparVal {
    fn first_f64(&self) -> Option<f64> {
        match self {
            Self::Numbers(v) => v.first().copied(),
            Self::Strings(v) => v.first().and_then(|s| s.parse().ok()),
        }
    }

    fn first_str(&self) -> Option<&str> {
        match self {
            Self::Strings(v) => v.first().map(String::as_str),
            Self::Numbers(_) => None,
        }
    }

    fn numbers(&self) -> Option<&[f64]> {
        match self {
            Self::Numbers(v) => Some(v.as_slice()),
            Self::Strings(_) => None,
        }
    }
}

fn unquote_pp(s: &str) -> String {
    let t = s.trim();
    if t.len() >= 2
        && ((t.starts_with('"') && t.ends_with('"')) || (t.starts_with('\'') && t.ends_with('\'')))
    {
        t[1..t.len() - 1].to_string()
    } else {
        t.to_string()
    }
}

fn parse_procpar_local(text: &str) -> Result<HashMap<String, LocalProcparVal>, String> {
    let mut map = HashMap::new();
    let lines: Vec<&str> = text.lines().collect();
    let mut idx = 0usize;
    while idx < lines.len() {
        let hdr = lines[idx].trim();
        idx += 1;
        if hdr.is_empty() {
            continue;
        }
        let parts: Vec<&str> = hdr.split_whitespace().collect();
        if parts.len() < 11 {
            continue;
        }
        let name = parts[0].to_string();
        let Ok(basictype) = parts[2].parse::<i32>() else {
            continue;
        };
        if idx >= lines.len() {
            break;
        }
        if basictype == 2 {
            let first = lines[idx].trim();
            idx += 1;
            let (cnt_s, rem) = first.split_once(char::is_whitespace).unwrap_or((first, ""));
            let count: usize = cnt_s.parse().unwrap_or(1);
            let mut strs = Vec::with_capacity(count);
            if !rem.trim().is_empty() {
                strs.push(unquote_pp(rem.trim()));
            }
            while strs.len() < count && idx < lines.len() {
                strs.push(unquote_pp(lines[idx].trim()));
                idx += 1;
            }
            map.insert(name, LocalProcparVal::Strings(strs));
        } else {
            let mut nums = Vec::new();
            let toks: Vec<&str> = lines[idx].split_whitespace().collect();
            idx += 1;
            if let Some(cnt_tok) = toks.first() {
                let count: usize = cnt_tok.parse().unwrap_or(0);
                for &t in &toks[1..] {
                    if let Ok(v) = t.parse::<f64>() {
                        nums.push(v);
                    }
                }
                while nums.len() < count && idx < lines.len() {
                    let more: Vec<&str> = lines[idx].split_whitespace().collect();
                    idx += 1;
                    if more.is_empty() {
                        break;
                    }
                    for &t in &more {
                        if let Ok(v) = t.parse::<f64>() {
                            nums.push(v);
                        }
                    }
                }
            }
            map.insert(name, LocalProcparVal::Numbers(nums));
        }
        if idx < lines.len() {
            idx += 1;
        }
    }
    Ok(map)
}

/// Parse Varian `procpar` arrayed relaxation/exchange schedule (`relaxT`, `d2`, `ncyc`, `satfrq`, `tof`)
/// and deduce the loop nesting order (`PlaneQueued`, `T1PlaneQuad`, or `T1QuadPlane`).
pub fn parse_varian_relax_schedule(
    procpar_text: &str,
    unit_cycle_sec_override: Option<f64>,
) -> Result<VarianRelaxSchedule, String> {
    let pp = parse_procpar_local(procpar_text)?;
    parse_varian_relax_schedule_from_map(&pp, unit_cycle_sec_override)
}

fn parse_varian_relax_schedule_from_map(
    pp: &HashMap<String, LocalProcparVal>,
    unit_cycle_sec_override: Option<f64>,
) -> Result<VarianRelaxSchedule, String> {
    let array_raw = pp
        .get("array")
        .and_then(LocalProcparVal::first_str)
        .unwrap_or("");
    let array_elems: Vec<String> = array_raw
        .split(',')
        .map(|s| s.trim().trim_matches('(').trim_matches(')').to_string())
        .filter(|s| !s.is_empty())
        .collect();

    let candidates = [
        "relaxT", "ncyc", "satfrq", "d2", "time_T2", "tau", "tof", "dof2",
    ];
    let mut chosen_param: Option<String> = None;
    for elem in &array_elems {
        if !elem.eq_ignore_ascii_case("phase")
            && !elem.eq_ignore_ascii_case("phase2")
            && pp
                .get(elem)
                .and_then(LocalProcparVal::numbers)
                .is_some_and(|v| v.len() >= 2)
        {
            chosen_param = Some(elem.clone());
            break;
        }
    }
    if chosen_param.is_none() {
        for cand in candidates {
            if pp
                .get(cand)
                .and_then(LocalProcparVal::numbers)
                .is_some_and(|v| v.len() >= 2)
            {
                chosen_param = Some(cand.to_string());
                break;
            }
        }
    }

    let param_name = chosen_param.ok_or_else(|| {
        "no arrayed relaxation/exchange parameter (e.g. relaxT, d2, ncyc, satfrq) with >= 2 values found in Varian procpar".to_string()
    })?;
    let vals = pp
        .get(&param_name)
        .and_then(LocalProcparVal::numbers)
        .ok_or_else(|| {
            format!("missing numeric values for Varian arrayed parameter '{param_name}'")
        })?;

    // Determine loop interleaving from VnmrJ `array` ordering:
    // In VnmrJ, `ni` (`t1`) is outer to `array` unless explicitly plane-queued (`seq_mode = "plane_queued"`):
    // - `array = "relaxT,phase"` -> `relaxT` outer, `phase` inner inside each `t1` -> `T1PlaneQuad`
    // - `array = "phase,relaxT"` -> `phase` outer, `relaxT` inner inside each `t1` -> `T1QuadPlane`
    let seq_mode = pp
        .get("seq_mode")
        .and_then(LocalProcparVal::first_str)
        .unwrap_or("");
    let interleaving = if seq_mode.eq_ignore_ascii_case("plane_queued") {
        Pseudo3DInterleaving::PlaneQueued
    } else {
        let pos_param = array_elems.iter().position(|e| e == &param_name);
        let pos_phase = array_elems
            .iter()
            .position(|e| e.eq_ignore_ascii_case("phase"));
        match (pos_param, pos_phase) {
            (Some(pp_i), Some(ph_i)) if ph_i < pp_i => Pseudo3DInterleaving::T1QuadPlane,
            (Some(_), Some(_)) => Pseudo3DInterleaving::T1PlaneQuad,
            _ => Pseudo3DInterleaving::PlaneQueued,
        }
    };

    let p_low = param_name.to_ascii_lowercase();
    let pslabel = pp
        .get("pslabel")
        .and_then(LocalProcparVal::first_str)
        .unwrap_or("")
        .to_ascii_lowercase();

    if p_low == "ncyc" {
        let tau_unit = unit_cycle_sec_override
            .or_else(|| {
                pp.get("tau_cpmg")
                    .and_then(LocalProcparVal::first_f64)
                    .map(|t| 4.0 * t)
            })
            .or_else(|| pp.get("unit_cycle").and_then(LocalProcparVal::first_f64))
            .unwrap_or(0.001);
        let mut counts = Vec::with_capacity(vals.len());
        let mut secs = Vec::with_capacity(vals.len());
        for &v in vals {
            if !v.is_finite() || v < 0.0 {
                return Err(format!("invalid Varian ncyc value {v}"));
            }
            let c = v.round() as u32;
            counts.push(c);
            secs.push((c as f64) * tau_unit);
        }
        Ok(VarianRelaxSchedule {
            param_name,
            experiment_kind: RelaxExperimentKind::Cpmg,
            schedule: RelaxSchedule::LoopCounts {
                counts,
                unit_cycle_sec: tau_unit,
                values_sec: secs,
            },
            interleaving,
        })
    } else if p_low == "satfrq" || p_low == "tof" || p_low == "dof2" {
        let sfrq = pp
            .get("sfrq")
            .and_then(LocalProcparVal::first_f64)
            .unwrap_or(600.0);
        let dfrq2 = pp
            .get("dfrq2")
            .and_then(LocalProcparVal::first_f64)
            .unwrap_or(sfrq * 0.101_329_118);
        let dof2_ref = pp
            .get("dof2")
            .and_then(LocalProcparVal::first_f64)
            .unwrap_or(0.0);
        let bf_n = (sfrq / (1.0 + 4.766e-6)) * 0.101_329_118;
        let carrier_ppm = ((dfrq2 - bf_n) * 1.0e6 + dof2_ref) / bf_n.max(1.0);
        let offsets: Vec<FqOffset> = vals
            .iter()
            .map(|&v| {
                // `satfrq` in Hz relative to carrier (`dof2_ref`) if small (`|v| < 5e5`), else absolute Hz
                let off_hz = if p_low == "dof2" { v - dof2_ref } else { v };
                FqOffset {
                    offset_hz: off_hz,
                    offset_ppm: carrier_ppm + off_hz / bf_n.max(1.0),
                }
            })
            .collect();
        Ok(VarianRelaxSchedule {
            param_name,
            experiment_kind: RelaxExperimentKind::Cest,
            schedule: RelaxSchedule::FrequencyOffsets { offsets },
            interleaving,
        })
    } else {
        for &v in vals {
            if !v.is_finite() || v < 0.0 {
                return Err(format!(
                    "invalid negative or non-finite Varian delay {v} in '{param_name}'"
                ));
            }
        }
        let exp_kind = if pslabel.contains("t2") || pslabel.contains("r2") {
            RelaxExperimentKind::R2
        } else if pslabel.contains("r1rho") || pslabel.contains("t1rho") {
            RelaxExperimentKind::R1rho
        } else if pslabel.contains("noe") {
            RelaxExperimentKind::HetNoe
        } else {
            RelaxExperimentKind::R1
        };
        Ok(VarianRelaxSchedule {
            param_name,
            experiment_kind: exp_kind,
            schedule: RelaxSchedule::VariableDelaysSec {
                delays_sec: vals.to_vec(),
            },
            interleaving,
        })
    }
}

// ============================================================================
// Raw Pseudo-3D Bruker (`ser`) & Varian (`fid`) Series Decoders
// ============================================================================

#[inline]
fn decode_bruker_trace_local(
    slice: &[u8],
    n_cplx: usize,
    big_endian: bool,
    dtypa: i32,
) -> Vec<Complex64> {
    let mut out = Vec::with_capacity(n_cplx);
    match dtypa {
        2 => {
            for k in 0..n_cplx {
                let b = k * 16;
                let r: [u8; 8] = slice[b..b + 8].try_into().unwrap();
                let m: [u8; 8] = slice[b + 8..b + 16].try_into().unwrap();
                let re = if big_endian {
                    f64::from_be_bytes(r)
                } else {
                    f64::from_le_bytes(r)
                };
                let im = if big_endian {
                    f64::from_be_bytes(m)
                } else {
                    f64::from_le_bytes(m)
                };
                out.push([re, im]);
            }
        }
        1 => {
            for k in 0..n_cplx {
                let b = k * 8;
                let r: [u8; 4] = slice[b..b + 4].try_into().unwrap();
                let m: [u8; 4] = slice[b + 4..b + 8].try_into().unwrap();
                let re = if big_endian {
                    f32::from_be_bytes(r) as f64
                } else {
                    f32::from_le_bytes(r) as f64
                };
                let im = if big_endian {
                    f32::from_be_bytes(m) as f64
                } else {
                    f32::from_le_bytes(m) as f64
                };
                out.push([re, im]);
            }
        }
        _ => {
            for k in 0..n_cplx {
                let b = k * 8;
                let r: [u8; 4] = slice[b..b + 4].try_into().unwrap();
                let m: [u8; 4] = slice[b + 4..b + 8].try_into().unwrap();
                let re = if big_endian {
                    i32::from_be_bytes(r) as f64
                } else {
                    i32::from_le_bytes(r) as f64
                };
                let im = if big_endian {
                    i32::from_be_bytes(m) as f64
                } else {
                    i32::from_le_bytes(m) as f64
                };
                out.push([re, im]);
            }
        }
    }
    out
}

#[inline]
fn recombine_quad_pair_local(
    quad_mode: QuadMode,
    k1: usize,
    z0: &[Complex64],
    z1: &[Complex64],
    out_cos: &mut [Complex64],
    out_sin: &mut [Complex64],
) {
    let n = z0.len();
    match quad_mode {
        QuadMode::EchoAntiecho => {
            for i in 0..n {
                let (r0, m0) = (z0[i][0], z0[i][1]);
                let (r1, m1) = (z1[i][0], z1[i][1]);
                out_cos[i] = [0.5 * (r0 + r1), 0.5 * (m0 + m1)];
                out_sin[i] = [0.5 * (m0 - m1), 0.5 * (r1 - r0)];
            }
        }
        QuadMode::StatesTppi => {
            let sign = if k1 % 2 == 1 { -1.0 } else { 1.0 };
            for i in 0..n {
                out_cos[i] = [sign * z0[i][0], sign * z0[i][1]];
                out_sin[i] = [sign * z1[i][0], sign * z1[i][1]];
            }
        }
        _ => {
            out_cos.copy_from_slice(z0);
            out_sin.copy_from_slice(z1);
        }
    }
}

#[inline]
fn trace_index_for_interleaving(
    interleaving: Pseudo3DInterleaving,
    plane_idx: usize,
    n_planes: usize,
    k1: usize,
    n1: usize,
    q: usize,
    q_mult: usize,
) -> usize {
    match interleaving {
        Pseudo3DInterleaving::PlaneQueued => (plane_idx * n1 + k1) * q_mult + q,
        Pseudo3DInterleaving::T1PlaneQuad => (k1 * n_planes + plane_idx) * q_mult + q,
        Pseudo3DInterleaving::T1QuadPlane => (k1 * q_mult + q) * n_planes + plane_idx,
    }
}

fn infer_bruker_pseudo3d_interleaving(
    pp_text: Option<&str>,
    acqus: &HashMap<String, String>,
    chem_is_acqu2s: bool,
) -> Pseudo3DInterleaving {
    if let Some(txt) = pp_text {
        for raw_line in txt.lines() {
            if let Some((_, cmt)) = raw_line.split_once(';') {
                let low = cmt.trim().to_ascii_lowercase();
                if low.contains("interleaving: plane_queued")
                    || low.contains("pseudo3d: plane_queued")
                {
                    return Pseudo3DInterleaving::PlaneQueued;
                }
                if low.contains("interleaving: t1_plane_quad")
                    || low.contains("pseudo3d: t1_plane_quad")
                {
                    return Pseudo3DInterleaving::T1PlaneQuad;
                }
                if low.contains("interleaving: t1_quad_plane")
                    || low.contains("pseudo3d: t1_quad_plane")
                {
                    return Pseudo3DInterleaving::T1QuadPlane;
                }
            }
        }
        let mut pos_id0 = None;
        let mut pos_ivd = None;
        let mut pos_quad = None;
        for (idx, line) in txt.lines().enumerate() {
            let code = line.split(';').next().unwrap_or("").trim();
            if code.is_empty() || code.starts_with('#') {
                continue;
            }
            if pos_id0.is_none() && (code.contains("id0") || code.contains("id10")) {
                pos_id0 = Some(idx);
            }
            if pos_ivd.is_none()
                && (code.contains("ivd") || code.contains("ivc") || code.contains("ifq"))
            {
                pos_ivd = Some(idx);
            }
            if pos_quad.is_none()
                && (code.contains("F1PH")
                    || code.contains("F2PH")
                    || code.contains("igrad")
                    || code.contains("ip1")
                    || code.contains("ip3"))
            {
                pos_quad = Some(idx);
            }
        }
        // In Bruker pulse programs, inner loops appear BEFORE outer `lo to ...` statements at the bottom:
        if let (Some(i_t1), Some(i_vd)) = (pos_id0, pos_ivd) {
            if i_t1 < i_vd {
                return Pseudo3DInterleaving::PlaneQueued;
            }
            if let Some(i_q) = pos_quad {
                if i_vd < i_q {
                    return Pseudo3DInterleaving::T1QuadPlane;
                }
            }
            return Pseudo3DInterleaving::T1PlaneQuad;
        }
    }

    let aqseq: i32 = acqus.get("AQSEQ").and_then(|s| s.parse().ok()).unwrap_or(0);
    // When `acqu2s` is chemical shift (`F2` inner) and `acqu3s` is `QF` (`F1` outer) under default `AQSEQ=0` (`321`),
    // planes are outer -> `PlaneQueued`. When `acqu2s` is `QF` (`F2` inner) and `acqu3s` is chemical shift (`F1` outer),
    // planes step inside each `(t1, quad)` increment -> `T1QuadPlane`.
    if chem_is_acqu2s {
        if aqseq == 1 {
            Pseudo3DInterleaving::T1QuadPlane
        } else {
            Pseudo3DInterleaving::PlaneQueued
        }
    } else if aqseq == 1 {
        Pseudo3DInterleaving::PlaneQueued
    } else {
        Pseudo3DInterleaving::T1QuadPlane
    }
}

/// Read a raw Bruker pseudo-3D relaxation or exchange `ser` directory (`acqus`, `acqu2s`, optional `acqu3s`,
/// and `vdlist` / `vclist` / `fqlist`), decoding both plane-queued and $t_1$-interleaved `ser` binaries
/// into per-plane `Fid2D` slices.
pub fn read_bruker_pseudo3d(
    dir: impl AsRef<Path>,
    interleaving_override: Option<Pseudo3DInterleaving>,
    unit_cycle_sec_for_vclist: Option<f64>,
) -> Result<RawPseudo3D, String> {
    let dir = dir.as_ref();
    let acqus = read_merged_jcamp_local(dir, "acqus", "acqu")?;
    let acqu2s = read_merged_jcamp_local(dir, "acqu2s", "acqu2")?;
    let acqu3s = read_merged_jcamp_local(dir, "acqu3s", "acqu3").ok();

    let pdata1 = dir.join("pdata").join("1");
    let procs = read_jcamp_file_opt(&pdata1, "procs");
    let proc2s = read_jcamp_file_opt(&pdata1, "proc2s");
    let proc3s = read_jcamp_file_opt(&pdata1, "proc3s");

    // Parse direct axis (`F3`/`F2`, `1H`)
    let td_x: usize = acqus
        .get("TD")
        .ok_or("missing ##$TD in Bruker acqus")?
        .parse()
        .map_err(|e| format!("invalid direct TD: {e}"))?;
    if td_x < 2 || !td_x.is_multiple_of(2) {
        return Err(format!("invalid Bruker direct TD={td_x}"));
    }
    let bytorda: i32 = acqus
        .get("BYTORDA")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let dtypa: i32 = acqus.get("DTYPA").and_then(|s| s.parse().ok()).unwrap_or(0);
    let big_endian = bytorda != 0;
    let word_bytes = if dtypa == 2 { 8 } else { 4 };

    let sfo1_x: f64 = acqus
        .get("SFO1")
        .or_else(|| acqus.get("BF1"))
        .ok_or("missing ##$SFO1 in Bruker acqus")?
        .parse()
        .map_err(|e| format!("invalid SFO1: {e}"))?;
    let bf1_x: f64 = acqus
        .get("BF1")
        .and_then(|s| s.parse().ok())
        .unwrap_or(sfo1_x);
    let sw_hz_x: f64 = acqus
        .get("SW_h")
        .and_then(|s| s.parse().ok())
        .filter(|&v| v > 0.0)
        .or_else(|| {
            acqus
                .get("SW")
                .and_then(|s| s.parse::<f64>().ok())
                .map(|sw| sw * sfo1_x)
        })
        .ok_or("missing ##$SW_h / ##$SW in Bruker acqus")?;
    let obs_x = procs
        .as_ref()
        .and_then(|p| p.get("SF"))
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|&v| v > 0.0)
        .unwrap_or(bf1_x);
    let first_ppm_x = bruker_first_ppm(procs.as_ref(), sfo1_x, obs_x, sw_hz_x);

    let grpdly_raw = acqus
        .get("GRPDLY")
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(-1.0);
    let dspfvs = acqus
        .get("DSPFVS")
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or(0);
    let decim = acqus
        .get("DECIM")
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or(1);
    let group_delay = if grpdly_raw >= 0.0 {
        grpdly_raw
    } else {
        bruker_dsp_group_delay(dspfvs, decim)?
    };
    let mut stored_x = extract_stored_proc_jcamp(procs.as_ref());
    if let Some(ref mut tde) = stored_x.td_eff {
        *tde = (*tde / 2).min(td_x / 2).max(1);
    }

    let axis_x = FidAxisMeta {
        label: normalize_nuc(acqus.get("NUC1").map(String::as_str).unwrap_or("1H"), "1H"),
        td_raw: td_x,
        complex_points: td_x / 2,
        sw_hz: sw_hz_x,
        obs_mhz: obs_x,
        carrier_mhz: sfo1_x,
        first_ppm: first_ppm_x,
        quad_mode: QuadMode::Dqd,
        states_recombined: true,
        group_delay,
        conjugate_for_fft: true,
        stored_proc: stored_x,
    };

    // Identify chemical-shift indirect axis (`acqu2s` vs `acqu3s`)
    let fn2: i32 = acqu2s
        .get("FnMODE")
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let (chem_acqu, chem_proc, chem_is_acqu2s) = if let Some(ref a3) = acqu3s {
        let fn3: i32 = a3.get("FnMODE").and_then(|s| s.parse().ok()).unwrap_or(1);
        if fn2 == 1 && fn3 != 1 {
            (a3, proc3s.as_ref(), false)
        } else {
            (&acqu2s, proc2s.as_ref(), true)
        }
    } else {
        (&acqu2s, proc2s.as_ref(), true)
    };

    let sfo1_y: f64 = chem_acqu
        .get("SFO1")
        .or_else(|| chem_acqu.get("BF1"))
        .ok_or("missing ##$SFO1 in Bruker indirect acqus")?
        .parse()
        .map_err(|e| format!("invalid indirect SFO1: {e}"))?;
    let bf1_y: f64 = chem_acqu
        .get("BF1")
        .and_then(|s| s.parse().ok())
        .unwrap_or(sfo1_y);
    let sw_hz_y: f64 = chem_acqu
        .get("SW")
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|&v| v > 0.0)
        .map(|sw| sw * sfo1_y)
        .or_else(|| {
            chem_acqu
                .get("SW_h")
                .and_then(|s| s.parse::<f64>().ok())
                .filter(|&v| v > 0.0)
        })
        .ok_or("missing ##$SW / ##$SW_h in Bruker indirect acqus")?;
    let obs_y = chem_proc
        .and_then(|p| p.get("SF"))
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|&v| v > 0.0)
        .unwrap_or(bf1_y);
    let carrier_ppm_y = (sfo1_y - obs_y) * 1.0e6 / obs_y.max(1.0e-9);

    // Locate schedule file (`vdlist`, `vclist`, or `fqlist` / `fq1list` / `fq2list` / `fq3list`)
    let pulprog = acqus.get("PULPROG").cloned().unwrap_or_default();
    let p_low = pulprog.to_ascii_lowercase();
    let (schedule, exp_kind) = if let Some(vd_path) = find_file_ci_local(dir, "vdlist") {
        let txt = fs::read_to_string(&vd_path)
            .map_err(|e| format!("failed to read {}: {e}", vd_path.display()))?;
        let delays_sec = parse_bruker_vdlist(&txt)?;
        let kind = if p_low.contains("t2") || p_low.contains("r2") {
            RelaxExperimentKind::R2
        } else if p_low.contains("t1rho") || p_low.contains("r1rho") {
            RelaxExperimentKind::R1rho
        } else {
            RelaxExperimentKind::R1
        };
        (RelaxSchedule::VariableDelaysSec { delays_sec }, kind)
    } else if let Some(vc_path) = find_file_ci_local(dir, "vclist") {
        let txt = fs::read_to_string(&vc_path)
            .map_err(|e| format!("failed to read {}: {e}", vc_path.display()))?;
        let unit_sec = unit_cycle_sec_for_vclist.unwrap_or_else(|| {
            // Bruker `D[20]` or `2 * D[20]` spin-echo / CPMG delay fallback (`16 ms` typical CPMG/T2 block)
            acqus
                .get("D")
                .and_then(|d_arr| d_arr.split_whitespace().nth(20))
                .and_then(|s| s.parse::<f64>().ok())
                .filter(|&v| v > 0.0)
                .unwrap_or(0.016)
        });
        let (counts, values_sec) = parse_bruker_vclist(&txt, unit_sec)?;
        let kind = if p_low.contains("cpmg") || p_low.contains("rex") {
            RelaxExperimentKind::Cpmg
        } else {
            RelaxExperimentKind::R2
        };
        (
            RelaxSchedule::LoopCounts {
                counts,
                unit_cycle_sec: unit_sec,
                values_sec,
            },
            kind,
        )
    } else if let Some(fq_path) = find_file_ci_local(dir, "fqlist")
        .or_else(|| find_file_ci_local(dir, "fq3list"))
        .or_else(|| find_file_ci_local(dir, "fq2list"))
        .or_else(|| find_file_ci_local(dir, "fq1list"))
    {
        let txt = fs::read_to_string(&fq_path)
            .map_err(|e| format!("failed to read {}: {e}", fq_path.display()))?;
        let offsets = parse_bruker_fqlist(&txt, obs_y, carrier_ppm_y)?;
        (
            RelaxSchedule::FrequencyOffsets { offsets },
            RelaxExperimentKind::Cest,
        )
    } else {
        return Err(format!(
            "missing Bruker pseudo-3D schedule file (vdlist, vclist, or fqlist) in {}",
            dir.display()
        ));
    };

    let n_planes = schedule.len();
    let fn_chem: i32 = chem_acqu
        .get("FnMODE")
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let quad_y = match fn_chem {
        1 => QuadMode::Qf,
        4 => QuadMode::States,
        5 => QuadMode::StatesTppi,
        6 => QuadMode::EchoAntiecho,
        other => {
            return Err(format!(
                "unsupported Bruker indirect FnMODE={other} in {}",
                dir.display()
            ));
        }
    };
    let q_mult = if quad_y == QuadMode::Qf { 1 } else { 2 };

    let td_chem_raw: usize = chem_acqu
        .get("TD")
        .ok_or("missing ##$TD in Bruker indirect acqus")?
        .parse()
        .map_err(|e| format!("invalid indirect TD: {e}"))?;
    let td_per_plane_y = if acqu3s.is_none() && td_chem_raw.is_multiple_of(n_planes) {
        td_chem_raw / n_planes
    } else {
        td_chem_raw
    };
    if td_per_plane_y < q_mult || !td_per_plane_y.is_multiple_of(q_mult) {
        return Err(format!(
            "invalid per-plane indirect TD={td_per_plane_y} for q_mult={q_mult} in {}",
            dir.display()
        ));
    }
    let n1 = td_per_plane_y / q_mult;

    let pp_txt = find_file_ci_local(dir, "pulseprogram").and_then(|p| fs::read_to_string(p).ok());
    let mut stored_y = extract_stored_proc_jcamp(chem_proc);
    if let Some(ref mut tde) = stored_y.td_eff {
        *tde = (*tde / q_mult).min(n1).max(1);
    }
    if chem_proc.and_then(|p| p.get("REVERSE")).is_none() {
        stored_y.reverse = infer_bruker_raw_indirect_reverse(
            pp_txt.as_deref(),
            &pulprog,
            if chem_is_acqu2s { 2 } else { 1 },
            quad_y,
        );
    }

    let axis_y = FidAxisMeta {
        label: normalize_nuc(
            chem_acqu.get("NUC1").map(String::as_str).unwrap_or("15N"),
            "15N",
        ),
        td_raw: td_per_plane_y,
        complex_points: n1,
        sw_hz: sw_hz_y,
        obs_mhz: obs_y,
        carrier_mhz: sfo1_y,
        first_ppm: bruker_first_ppm(chem_proc, sfo1_y, obs_y, sw_hz_y),
        quad_mode: quad_y,
        states_recombined: quad_y != QuadMode::Qf,
        group_delay: 0.0,
        conjugate_for_fft: true,
        stored_proc: stored_y,
    };

    let ser_path = find_file_ci_local(dir, "ser")
        .ok_or_else(|| format!("missing Bruker ser binary in {}", dir.display()))?;
    let bytes =
        fs::read(&ser_path).map_err(|e| format!("failed to read {}: {e}", ser_path.display()))?;

    let total_traces = n_planes * n1 * q_mult;
    let exact_stride = td_x * word_bytes;
    let block_stride = exact_stride.div_ceil(1024) * 1024;
    let stride = if bytes.len() == total_traces * exact_stride {
        exact_stride
    } else if bytes.len() >= total_traces * block_stride {
        block_stride
    } else if bytes.len() >= total_traces * exact_stride {
        exact_stride
    } else {
        return Err(format!(
            "Bruker pseudo-3D ser truncated in {}: expected >= {} B ({} traces x {} B), got {} B",
            dir.display(),
            total_traces * exact_stride,
            total_traces,
            exact_stride,
            bytes.len()
        ));
    };

    let interleaving = interleaving_override.unwrap_or_else(|| {
        infer_bruker_pseudo3d_interleaving(pp_txt.as_deref(), &acqus, chem_is_acqu2s)
    });

    let n0 = axis_x.complex_points;
    let base_name = dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("bruker_pseudo3d")
        .to_string();

    let mut planes = Vec::with_capacity(n_planes);
    let mut tmp_cos = vec![[0.0, 0.0]; n0];
    let mut tmp_sin = vec![[0.0, 0.0]; n0];

    for p_idx in 0..n_planes {
        let mut p_data = vec![[0.0, 0.0]; n1 * 2 * n0];
        for k1 in 0..n1 {
            if quad_y == QuadMode::Qf {
                let tidx =
                    trace_index_for_interleaving(interleaving, p_idx, n_planes, k1, n1, 0, 1);
                let off = tidx * stride;
                let tr = decode_bruker_trace_local(
                    &bytes[off..off + exact_stride],
                    n0,
                    big_endian,
                    dtypa,
                );
                p_data[(k1 * 2) * n0..(k1 * 2 + 1) * n0].copy_from_slice(&tr);
            } else {
                let t0 = trace_index_for_interleaving(interleaving, p_idx, n_planes, k1, n1, 0, 2);
                let t1 = trace_index_for_interleaving(interleaving, p_idx, n_planes, k1, n1, 1, 2);
                let z0 = decode_bruker_trace_local(
                    &bytes[t0 * stride..t0 * stride + exact_stride],
                    n0,
                    big_endian,
                    dtypa,
                );
                let z1 = decode_bruker_trace_local(
                    &bytes[t1 * stride..t1 * stride + exact_stride],
                    n0,
                    big_endian,
                    dtypa,
                );
                recombine_quad_pair_local(quad_y, k1, &z0, &z1, &mut tmp_cos, &mut tmp_sin);
                p_data[(k1 * 2) * n0..(k1 * 2 + 1) * n0].copy_from_slice(&tmp_cos);
                p_data[(k1 * 2 + 1) * n0..(k1 * 2 + 2) * n0].copy_from_slice(&tmp_sin);
            }
        }
        planes.push(Fid2D {
            name: format!("{base_name}_plane_{p_idx:03}"),
            vendor: FidVendor::Bruker,
            pulse_program: pulprog.clone(),
            axis_x: axis_x.clone(),
            axis_y: axis_y.clone(),
            data: p_data,
            nus_mask: None,
        });
    }

    Ok(RawPseudo3D {
        name: base_name,
        vendor: FidVendor::Bruker,
        experiment_kind: exp_kind,
        interleaving,
        schedule,
        planes,
    })
}

/// Read a raw Varian/Agilent arrayed pseudo-3D relaxation or exchange directory (`procpar` + `fid`),
/// supporting both plane-queued and $t_1$-interleaved (`array = "relaxT,phase"` vs `"phase,relaxT"`)
/// datasets into per-plane `Fid2D` slices.
pub fn read_varian_pseudo3d(
    dir: impl AsRef<Path>,
    interleaving_override: Option<Pseudo3DInterleaving>,
    unit_cycle_sec_override: Option<f64>,
) -> Result<RawPseudo3D, String> {
    let dir = dir.as_ref();
    let procpar_path = find_file_ci_local(dir, "procpar")
        .ok_or_else(|| format!("missing Varian procpar in {}", dir.display()))?;
    let fid_path = find_file_ci_local(dir, "fid")
        .ok_or_else(|| format!("missing Varian fid binary in {}", dir.display()))?;

    let pp_txt = fs::read_to_string(&procpar_path)
        .map_err(|e| format!("failed to read {}: {e}", procpar_path.display()))?;
    let pp = parse_procpar_local(&pp_txt)?;
    let sched_info = parse_varian_relax_schedule_from_map(&pp, unit_cycle_sec_override)?;

    let bytes =
        fs::read(&fid_path).map_err(|e| format!("failed to read {}: {e}", fid_path.display()))?;
    if bytes.len() < 32 {
        return Err(format!(
            "Varian fid {} shorter than 32-byte header",
            fid_path.display()
        ));
    }
    let nblocks = i32::from_be_bytes(bytes[0..4].try_into().unwrap()).max(0) as usize;
    let ntraces = i32::from_be_bytes(bytes[4..8].try_into().unwrap()).max(0) as usize;
    let np = i32::from_be_bytes(bytes[8..12].try_into().unwrap()).max(0) as usize;
    let ebytes = i32::from_be_bytes(bytes[12..16].try_into().unwrap()).max(0) as usize;
    let bbytes = i32::from_be_bytes(bytes[20..24].try_into().unwrap()).max(0) as usize;
    let status = u16::from_be_bytes(bytes[26..28].try_into().unwrap());
    let nbheaders = i32::from_be_bytes(bytes[28..32].try_into().unwrap()).max(1) as usize;

    if np < 2 || !np.is_multiple_of(2) {
        return Err(format!("invalid Varian np={np} in {}", fid_path.display()));
    }
    let needed = 32 + nblocks * bbytes;
    if bytes.len() < needed {
        return Err(format!(
            "Varian pseudo-3D fid {} truncated: expected {needed} B, got {} B",
            fid_path.display(),
            bytes.len()
        ));
    }

    let n0 = np / 2;
    let hdr_per_block = nbheaders * 28;
    let is_float = (status & 0x0008) != 0;
    let is_32 = (status & 0x0004) != 0 || ebytes == 4;

    let mut raw_traces: Vec<Vec<Complex64>> = Vec::with_capacity(nblocks * ntraces);
    for b in 0..nblocks {
        let b_off = 32 + b * bbytes;
        let data_start = b_off + hdr_per_block;
        for tr in 0..ntraces {
            let t_off = data_start + tr * np * ebytes;
            let mut cvec = Vec::with_capacity(n0);
            if is_float && ebytes == 4 {
                for k in 0..n0 {
                    let p = t_off + k * 8;
                    let re = f32::from_be_bytes(bytes[p..p + 4].try_into().unwrap()) as f64;
                    let im = f32::from_be_bytes(bytes[p + 4..p + 8].try_into().unwrap()) as f64;
                    cvec.push([re, im]);
                }
            } else if is_32 && ebytes == 4 {
                for k in 0..n0 {
                    let p = t_off + k * 8;
                    let re = i32::from_be_bytes(bytes[p..p + 4].try_into().unwrap()) as f64;
                    let im = i32::from_be_bytes(bytes[p + 4..p + 8].try_into().unwrap()) as f64;
                    cvec.push([re, im]);
                }
            } else if ebytes == 2 {
                for k in 0..n0 {
                    let p = t_off + k * 4;
                    let re = i16::from_be_bytes(bytes[p..p + 2].try_into().unwrap()) as f64;
                    let im = i16::from_be_bytes(bytes[p + 2..p + 4].try_into().unwrap()) as f64;
                    cvec.push([re, im]);
                }
            } else {
                return Err(format!(
                    "unsupported Varian sample format (status=0x{status:04x}, ebytes={ebytes})"
                ));
            }
            raw_traces.push(cvec);
        }
    }

    let n_planes = sched_info.schedule.len();
    let ni = pp
        .get("ni")
        .and_then(LocalProcparVal::first_f64)
        .ok_or("missing `ni` in Varian procpar")?
        .round() as usize;
    if ni < 1 {
        return Err("invalid Varian ni < 1".to_string());
    }

    let phase_arr = pp
        .get("phase")
        .and_then(LocalProcparVal::numbers)
        .unwrap_or(&[1.0, 2.0]);
    let q_mult = if phase_arr.len() == 1 { 1 } else { 2 };
    let expected_traces = n_planes * ni * q_mult;
    if raw_traces.len() < expected_traces {
        return Err(format!(
            "Varian pseudo-3D trace count mismatch: got {}, expected {expected_traces} (n_planes={n_planes}, ni={ni}, q_mult={q_mult})",
            raw_traces.len()
        ));
    }

    let fcoef_raw = pp
        .get("f1coef")
        .or_else(|| pp.get("fcoef"))
        .and_then(LocalProcparVal::first_str);
    let (quad_y, coef) = if q_mult == 1 {
        (QuadMode::Qf, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    } else if let Some(s) = fcoef_raw.filter(|s| !s.trim().is_empty()) {
        let parsed: Result<Vec<f64>, _> = s.split_whitespace().map(str::parse::<f64>).collect();
        let vals = parsed.map_err(|e| format!("invalid Varian f1coef '{s}': {e}"))?;
        if vals.len() != 8 {
            return Err(format!("Varian f1coef '{s}' must have 8 floats"));
        }
        let mut c = [0.0; 8];
        c.copy_from_slice(&vals);
        let is_ea = c[1].abs() > 0.0 || c[3].abs() > 0.0 || c[5].abs() > 0.0 || c[7].abs() > 0.0;
        (
            if is_ea {
                QuadMode::EchoAntiecho
            } else {
                QuadMode::States
            },
            c,
        )
    } else {
        (QuadMode::States, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    };

    let sfrq = pp
        .get("sfrq")
        .and_then(LocalProcparVal::first_f64)
        .ok_or("missing `sfrq` in Varian procpar")?;
    let sw_x = pp
        .get("sw")
        .and_then(LocalProcparVal::first_f64)
        .ok_or("missing `sw` in Varian procpar")?;
    let sw_y = pp
        .get("sw1")
        .and_then(LocalProcparVal::first_f64)
        .ok_or("missing `sw1` in Varian procpar")?;
    let temp_c = pp
        .get("temp")
        .and_then(LocalProcparVal::first_f64)
        .filter(|&t| (-20.0..=100.0).contains(&t))
        .unwrap_or(25.0);
    let h2o_ppm = 4.766 - 0.0119 * (temp_c - 25.0);
    let bf_1h = sfrq / (1.0 + h2o_ppm * 1.0e-6);
    let first_ppm_x = ((sfrq - bf_1h) * 1.0e6 + 0.5 * sw_x) / bf_1h;

    let dn2 = normalize_nuc(
        pp.get("dn2")
            .and_then(LocalProcparVal::first_str)
            .unwrap_or("15N"),
        "15N",
    );
    let dfrq2 = pp
        .get("dfrq2")
        .and_then(LocalProcparVal::first_f64)
        .or_else(|| pp.get("dfrq").and_then(LocalProcparVal::first_f64))
        .unwrap_or(bf_1h * 0.101_329_118);
    let xi_y = if dn2 == "13C" {
        0.251_449_530
    } else {
        0.101_329_118
    };
    let obs_y = bf_1h * xi_y;
    let first_ppm_y = ((dfrq2 - obs_y) * 1.0e6 + 0.5 * sw_y) / obs_y;

    let stored_x = StoredDimProc {
        p0_deg: pp.get("rp").and_then(LocalProcparVal::first_f64),
        p1_deg: pp.get("lp").and_then(LocalProcparVal::first_f64),
        si: pp
            .get("fn")
            .and_then(LocalProcparVal::first_f64)
            .map(|v| (v as usize) / 2)
            .filter(|&v| v >= 2),
        ..Default::default()
    };
    let stored_y = StoredDimProc {
        p0_deg: pp.get("rp1").and_then(LocalProcparVal::first_f64),
        p1_deg: pp.get("lp1").and_then(LocalProcparVal::first_f64),
        si: pp
            .get("fn1")
            .and_then(LocalProcparVal::first_f64)
            .map(|v| (v as usize) / 2)
            .filter(|&v| v >= 2),
        ..Default::default()
    };

    let axis_x = FidAxisMeta {
        label: "1H".to_string(),
        td_raw: np,
        complex_points: n0,
        sw_hz: sw_x,
        obs_mhz: bf_1h,
        carrier_mhz: sfrq,
        first_ppm: first_ppm_x,
        quad_mode: QuadMode::Dqd,
        states_recombined: true,
        group_delay: 0.0,
        conjugate_for_fft: false,
        stored_proc: stored_x,
    };
    let axis_y = FidAxisMeta {
        label: dn2,
        td_raw: ni * q_mult,
        complex_points: ni,
        sw_hz: sw_y,
        obs_mhz: obs_y,
        carrier_mhz: dfrq2,
        first_ppm: first_ppm_y,
        quad_mode: quad_y,
        states_recombined: quad_y != QuadMode::Qf,
        group_delay: 0.0,
        conjugate_for_fft: false,
        stored_proc: stored_y,
    };

    let interleaving = interleaving_override.unwrap_or(sched_info.interleaving);
    let pslabel = pp
        .get("pslabel")
        .and_then(LocalProcparVal::first_str)
        .unwrap_or("gNhsqc")
        .to_string();
    let base_name = dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("varian_pseudo3d")
        .to_string();

    let mut planes = Vec::with_capacity(n_planes);
    let mut out_cos = vec![[0.0, 0.0]; n0];
    let mut out_sin = vec![[0.0, 0.0]; n0];

    for p_idx in 0..n_planes {
        let mut p_data = vec![[0.0, 0.0]; ni * 2 * n0];
        for k1 in 0..ni {
            if q_mult == 1 {
                let tidx =
                    trace_index_for_interleaving(interleaving, p_idx, n_planes, k1, ni, 0, 1);
                p_data[(k1 * 2) * n0..(k1 * 2 + 1) * n0].copy_from_slice(&raw_traces[tidx]);
            } else {
                let t0 = trace_index_for_interleaving(interleaving, p_idx, n_planes, k1, ni, 0, 2);
                let t1 = trace_index_for_interleaving(interleaving, p_idx, n_planes, k1, ni, 1, 2);
                let z1 = &raw_traces[t0];
                let z2 = &raw_traces[t1];
                let [c0, c1, c2, c3, c4, c5, c6, c7] = coef;
                for i in 0..n0 {
                    let (r1, m1) = (z1[i][0], z1[i][1]);
                    let (r2, m2) = (z2[i][0], z2[i][1]);
                    out_cos[i] = [
                        (c0 * r1 + c1 * m1) + (c2 * r2 + c3 * m2),
                        (c0 * m1 - c1 * r1) + (c2 * m2 - c3 * r2),
                    ];
                    out_sin[i] = [
                        (c4 * r1 + c5 * m1) + (c6 * r2 + c7 * m2),
                        (c4 * m1 - c5 * r1) + (c6 * m2 - c7 * r2),
                    ];
                }
                p_data[(k1 * 2) * n0..(k1 * 2 + 1) * n0].copy_from_slice(&out_cos);
                p_data[(k1 * 2 + 1) * n0..(k1 * 2 + 2) * n0].copy_from_slice(&out_sin);
            }
        }
        planes.push(Fid2D {
            name: format!("{base_name}_plane_{p_idx:03}"),
            vendor: FidVendor::Varian,
            pulse_program: pslabel.clone(),
            axis_x: axis_x.clone(),
            axis_y: axis_y.clone(),
            data: p_data,
            nus_mask: None,
        });
    }

    Ok(RawPseudo3D {
        name: base_name,
        vendor: FidVendor::Varian,
        experiment_kind: sched_info.experiment_kind,
        interleaving,
        schedule: sched_info.schedule,
        planes,
    })
}

/// Process a stack of pseudo-3D `Fid2D` planes using a **single locked `ProcessingRecipe`**:
/// solves any automatic phase parameters on the highest-energy reference plane once, locks those
/// explicit `(p0_deg, p1_deg)` values into `ProcessingRecipe`, and applies identical apodization,
/// zero-filling, phase rotation, and ROI bounds to every plane.
pub fn process_pseudo3d_fids(
    fids: &[Fid2D],
    recipe_override: Option<&ProcessingRecipe>,
) -> Result<(Vec<Spectrum2D>, ProcessingRecipe), String> {
    if fids.is_empty() {
        return Err("cannot process empty pseudo-3D FID stack".to_string());
    }
    let ref_idx = fids
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            let ea: f64 = a.data.iter().map(|z| z[0] * z[0] + z[1] * z[1]).sum();
            let eb: f64 = b.data.iter().map(|z| z[0] * z[0] + z[1] * z[1]).sum();
            ea.total_cmp(&eb)
        })
        .map(|(i, _)| i)
        .unwrap_or(0);

    let mut locked = match recipe_override {
        Some(r) => r.clone(),
        None => ProcessingRecipe::from_fid_2d(&fids[ref_idx]),
    };

    let ref_spec = process_fid_2d_mut(&fids[ref_idx], &mut locked)?;
    // Lock resolved phases explicitly so low-SNR or inverted planes never re-run auto-phasing
    for d in &mut locked.dims {
        let p0 = d.p0_deg.unwrap_or(0.0);
        let p1 = d.p1_deg.unwrap_or(0.0);
        d.p0_deg = Some(p0);
        d.p1_deg = Some(p1);
        d.phase_source = PhaseSource::Explicit;
    }

    let mut out = Vec::with_capacity(fids.len());
    for (idx, fid) in fids.iter().enumerate() {
        if idx == ref_idx {
            out.push(ref_spec.clone());
        } else {
            out.push(process_fid_2d(fid, &locked)?);
        }
    }
    Ok((out, locked))
}

// ============================================================================
// 2D MAD Noise, Pseudo-Voigt Footprints & Locked Linear Least-Squares Integrator
// ============================================================================

fn median_f64(vals: &mut [f64]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    let n = vals.len();
    let mid = n / 2;
    vals.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    let upper = vals[mid];
    if n % 2 == 1 {
        upper
    } else {
        let lower = vals[..mid]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        0.5 * (lower + upper)
    }
}

/// Robust 2D Median Absolute Deviation noise estimator: $\sigma_{\text{mad}} = 1.4826 \cdot \mathrm{median}(|S - \mathrm{median}(S)|)$.
pub fn estimate_plane_noise_mad_2d(spec: &Spectrum2D) -> f64 {
    let mut buf: Vec<f64> = spec
        .data
        .iter()
        .map(|&v| v as f64)
        .filter(|v| v.is_finite())
        .collect();
    if buf.is_empty() {
        return 1.0e-12;
    }
    let med = median_f64(&mut buf);
    for v in &mut buf {
        *v = (*v - med).abs();
    }
    (MAD_TO_SIGMA * median_f64(&mut buf)).max(1.0e-12)
}

/// 1D unit-height pseudo-Voigt profile ($\psi(0) = 1$) with Gaussian width $\sigma$ and Lorentzian fraction $\eta \in [0, 1]$.
#[inline]
pub fn voigt_1d_unit(delta: f64, sigma: f64, eta: f64) -> f64 {
    let inv_s = 1.0 / sigma.max(1.0e-9);
    let u = delta * inv_s;
    let g = (-0.5 * u * u).exp();
    if eta <= 0.0 {
        g
    } else {
        // Equal-FWHM Lorentzian: HWHM gamma = sqrt(2 ln 2) * sigma => delta^2 / gamma^2 = u^2 / (2 ln 2)
        let lor = 1.0 / (1.0 + u * u / (2.0 * std::f64::consts::LN_2));
        (1.0 - eta) * g + eta * lor
    }
}

/// Analytical 1D integral of `voigt_1d_unit(delta, sigma, eta)` over $(-\infty, +\infty)$.
#[inline]
pub fn voigt_1d_area(sigma: f64, eta: f64) -> f64 {
    let s = sigma.max(1.0e-9);
    let area_g = (2.0 * PI).sqrt() * s;
    let area_l = PI * (2.0 * std::f64::consts::LN_2).sqrt() * s;
    (1.0 - eta) * area_g + eta * area_l
}

/// Separable 2D unit-height pseudo-Voigt footprint $\Psi(x, y) \in (0, 1]$.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn footprint_eval_2d(
    ppm_x: f64,
    ppm_y: f64,
    delta_h: f64,
    delta_x: f64,
    sigma_h: f64,
    sigma_x: f64,
    eta_h: f64,
    eta_x: f64,
) -> f64 {
    voigt_1d_unit(ppm_x - delta_h, sigma_h, eta_h) * voigt_1d_unit(ppm_y - delta_x, sigma_x, eta_x)
}

fn solve_spd_cholesky_or_ridge(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        let d = a[0];
        return vec![if d.abs() > 1.0e-20 { b[0] / d } else { 0.0 }];
    }
    let mut diag_max = 0.0_f64;
    for i in 0..n {
        diag_max = diag_max.max(a[i * n + i].abs());
    }
    let mut reg = 1.0e-12 * diag_max.max(1.0);
    for _ in 0..6 {
        let mut l = vec![0.0_f64; n * n];
        let mut ok = true;
        for i in 0..n {
            for j in 0..=i {
                let mut sum = a[i * n + j];
                if i == j {
                    sum += reg;
                }
                for k in 0..j {
                    sum -= l[i * n + k] * l[j * n + k];
                }
                if i == j {
                    if sum <= 1.0e-24 {
                        ok = false;
                        break;
                    }
                    l[i * n + i] = sum.sqrt();
                } else {
                    l[i * n + j] = sum / l[j * n + j];
                }
            }
            if !ok {
                break;
            }
        }
        if ok {
            let mut y = vec![0.0_f64; n];
            for i in 0..n {
                let mut sum = b[i];
                for k in 0..i {
                    sum -= l[i * n + k] * y[k];
                }
                y[i] = sum / l[i * n + i];
            }
            let mut x = vec![0.0_f64; n];
            for i in (0..n).rev() {
                let mut sum = y[i];
                for k in (i + 1)..n {
                    sum -= l[k * n + i] * x[k];
                }
                x[i] = sum / l[i * n + i];
            }
            return x;
        }
        reg *= 100.0;
    }
    vec![0.0; n]
}

fn invert_spd_matrix(a: &[f64], n: usize) -> Vec<f64> {
    let mut inv = vec![0.0_f64; n * n];
    let mut e = vec![0.0_f64; n];
    for col in 0..n {
        e.fill(0.0);
        e[col] = 1.0;
        let sol = solve_spd_cholesky_or_ridge(a, &e, n);
        for row in 0..n {
            inv[row * n + col] = sol[row];
        }
    }
    inv
}

fn patch_bounds_for_seed(
    ax_x: &AxisPpm,
    ax_y: &AxisPpm,
    delta_h: f64,
    delta_x: f64,
    hw_x: usize,
    hw_y: usize,
) -> (usize, usize, usize, usize) {
    let cx = ax_x
        .ppm_to_index(delta_h)
        .round()
        .clamp(0.0, (ax_x.size.saturating_sub(1)) as f64) as usize;
    let cy = ax_y
        .ppm_to_index(delta_x)
        .round()
        .clamp(0.0, (ax_y.size.saturating_sub(1)) as f64) as usize;
    let x0 = cx.saturating_sub(hw_x);
    let x1 = (cx + hw_x).min(ax_x.size.saturating_sub(1));
    let y0 = cy.saturating_sub(hw_y);
    let y1 = (cy + hw_y).min(ax_y.size.saturating_sub(1));
    (x0, x1, y0, y1)
}

/// Fit a single 2D Gaussian / pseudo-Voigt footprint on the reference spectrum `ref_spec` around `seed`
/// using Variable-Projected Levenberg–Marquardt (amplitude profiled out in closed form at every step).
fn fit_footprint_on_reference(
    ref_spec: &Spectrum2D,
    seed: &TargetPeakSeed,
    cfg: &RelaxIntegrationConfig,
) -> LockedPeakFootprint {
    let ax_x = &ref_spec.axis_x;
    let ax_y = &ref_spec.axis_y;
    let dx_ppm = ax_x.step_ppm.abs().max(1.0e-5);
    let dy_ppm = ax_y.step_ppm.abs().max(1.0e-4);

    let (x0, x1, y0, y1) = patch_bounds_for_seed(
        ax_x,
        ax_y,
        seed.delta_h_ppm,
        seed.delta_x_ppm,
        cfg.patch_half_width_pts[0],
        cfg.patch_half_width_pts[1],
    );

    let mut coords = Vec::with_capacity((x1 - x0 + 1) * (y1 - y0 + 1));
    let mut y_obs = Vec::with_capacity(coords.len());
    let mut best_local = (seed.delta_h_ppm, seed.delta_x_ppm, 0.0_f64);
    for iy in y0..=y1 {
        let py = ax_y.index_to_ppm(iy as f64);
        for ix in x0..=x1 {
            let px = ax_x.index_to_ppm(ix as f64);
            let val = ref_spec.get(ix, iy) as f64;
            coords.push((px, py));
            y_obs.push(val);
            if (px - seed.delta_h_ppm).abs() <= cfg.max_centroid_shift_ppm[0]
                && (py - seed.delta_x_ppm).abs() <= cfg.max_centroid_shift_ppm[1]
                && val.abs() > best_local.2.abs()
            {
                best_local = (px, py, val);
            }
        }
    }

    let fit_voigt = cfg.shape_model == FootprintShapeModel::PseudoVoigt;
    let n_params = if fit_voigt { 6 } else { 4 };
    let mut p = [
        best_local.0,
        best_local.1,
        (1.35 * dx_ppm),
        (1.35 * dy_ppm),
        if fit_voigt { 0.20 } else { 0.0 },
        if fit_voigt { 0.20 } else { 0.0 },
    ];

    let clamp_params = |q: &mut [f64; 6]| {
        q[0] = q[0].clamp(
            seed.delta_h_ppm - cfg.max_centroid_shift_ppm[0],
            seed.delta_h_ppm + cfg.max_centroid_shift_ppm[0],
        );
        q[1] = q[1].clamp(
            seed.delta_x_ppm - cfg.max_centroid_shift_ppm[1],
            seed.delta_x_ppm + cfg.max_centroid_shift_ppm[1],
        );
        q[2] = q[2].clamp(0.45 * dx_ppm, 6.0 * dx_ppm);
        q[3] = q[3].clamp(0.45 * dy_ppm, 6.0 * dy_ppm);
        if fit_voigt {
            q[4] = q[4].clamp(0.0, 1.0);
            q[5] = q[5].clamp(0.0, 1.0);
        } else {
            q[4] = 0.0;
            q[5] = 0.0;
        }
    };
    clamp_params(&mut p);

    let eval_varpro = |q: &[f64; 6], psi_out: &mut [f64]| -> (f64, f64) {
        let mut s_yy = 0.0_f64;
        let mut s_pp = 0.0_f64;
        for (i, &(px, py)) in coords.iter().enumerate() {
            let psi = footprint_eval_2d(px, py, q[0], q[1], q[2], q[3], q[4], q[5]);
            psi_out[i] = psi;
            s_yy += psi * y_obs[i];
            s_pp += psi * psi;
        }
        let amp = if s_pp > 1.0e-20 { s_yy / s_pp } else { 0.0 };
        let mut rss = 0.0_f64;
        for (i, &yo) in y_obs.iter().enumerate() {
            let r = yo - amp * psi_out[i];
            rss += r * r;
        }
        (rss, amp)
    };

    let m = coords.len();
    let mut psi_cur = vec![0.0_f64; m];
    let mut psi_tmp = vec![0.0_f64; m];
    let (mut rss_cur, mut amp_cur) = eval_varpro(&p, &mut psi_cur);
    let mut lambda = 1.0e-3_f64;
    let eps_step = [
        1.0e-5 * dx_ppm,
        1.0e-5 * dy_ppm,
        1.0e-5 * dx_ppm,
        1.0e-5 * dy_ppm,
        1.0e-4,
        1.0e-4,
    ];

    for _iter in 0..25 {
        if amp_cur.abs() <= 1.0e-20 {
            break;
        }
        // Finite-difference Jacobian of residual r_i(q) = amp(q)*psi_i(q) - y_i
        let mut j_cols = vec![vec![0.0_f64; m]; n_params];
        for j in 0..n_params {
            let mut q_plus = p;
            q_plus[j] += eps_step[j];
            let (_, amp_p) = eval_varpro(&q_plus, &mut psi_tmp);
            let inv_h = 1.0 / eps_step[j];
            for i in 0..m {
                j_cols[j][i] = (amp_p * psi_tmp[i] - amp_cur * psi_cur[i]) * inv_h;
            }
        }

        let mut jtj = vec![0.0_f64; n_params * n_params];
        let mut jtr = vec![0.0_f64; n_params];
        for i in 0..m {
            let res_i = y_obs[i] - amp_cur * psi_cur[i];
            for a in 0..n_params {
                jtr[a] += j_cols[a][i] * res_i;
                for b in 0..=a {
                    jtj[a * n_params + b] += j_cols[a][i] * j_cols[b][i];
                }
            }
        }
        for a in 0..n_params {
            for b in 0..a {
                jtj[b * n_params + a] = jtj[a * n_params + b];
            }
        }

        let mut accepted = false;
        for _sub in 0..6 {
            let mut lhs = jtj.clone();
            for d in 0..n_params {
                let diag = lhs[d * n_params + d].max(1.0e-12);
                lhs[d * n_params + d] += lambda * diag;
            }
            let step = solve_spd_cholesky_or_ridge(&lhs, &jtr, n_params);
            let mut cand = p;
            for d in 0..n_params {
                cand[d] += step[d];
            }
            clamp_params(&mut cand);
            let (rss_cand, amp_cand) = eval_varpro(&cand, &mut psi_tmp);
            if rss_cand < rss_cur {
                let rel_impr = (rss_cur - rss_cand) / rss_cur.max(1.0e-30);
                p = cand;
                rss_cur = rss_cand;
                amp_cur = amp_cand;
                psi_cur.copy_from_slice(&psi_tmp);
                lambda = (lambda * 0.3).max(1.0e-7);
                accepted = true;
                if rel_impr < 1.0e-9 {
                    return LockedPeakFootprint {
                        seq_id: seed.seq_id,
                        res_name: seed.res_name.clone(),
                        atom_h: seed.atom_h.clone(),
                        atom_x: seed.atom_x.clone(),
                        delta_h_ppm: p[0],
                        delta_x_ppm: p[1],
                        sigma_h_ppm: p[2],
                        sigma_x_ppm: p[3],
                        eta_h: p[4],
                        eta_x: p[5],
                        overlap_cluster_id: 0,
                    };
                }
                break;
            }
            lambda = (lambda * 5.0).min(1.0e6);
        }
        if !accepted {
            break;
        }
    }

    LockedPeakFootprint {
        seq_id: seed.seq_id,
        res_name: seed.res_name.clone(),
        atom_h: seed.atom_h.clone(),
        atom_x: seed.atom_x.clone(),
        delta_h_ppm: p[0],
        delta_x_ppm: p[1],
        sigma_h_ppm: p[2],
        sigma_x_ppm: p[3],
        eta_h: p[4],
        eta_x: p[5],
        overlap_cluster_id: 0,
    }
}

/// Jointly fit footprints for an overlapping multi-peak cluster (`members.len() >= 2`) on `ref_spec`
/// using multi-peak Variable-Projected Levenberg–Marquardt (projecting out all cluster member
/// amplitudes simultaneously at every iteration so neighboring peaks never stretch single-peak widths).
fn refine_cluster_footprints_on_reference(
    ref_spec: &Spectrum2D,
    seeds: &[TargetPeakSeed],
    members: &[usize],
    init_sigma_eta: [f64; 4],
    patches: &[(usize, usize, usize, usize)],
    footprints: &mut [LockedPeakFootprint],
    cfg: &RelaxIntegrationConfig,
) {
    let k_members = members.len();
    if k_members < 2 {
        return;
    }
    let ax_x = &ref_spec.axis_x;
    let ax_y = &ref_spec.axis_y;
    let dx_ppm = ax_x.step_ppm.abs().max(1.0e-5);
    let dy_ppm = ax_y.step_ppm.abs().max(1.0e-4);

    let mut pix_coords: Vec<(usize, usize)> = Vec::new();
    for &r_idx in members {
        let (x0, x1, y0, y1) = patches[r_idx];
        for iy in y0..=y1 {
            for ix in x0..=x1 {
                pix_coords.push((ix, iy));
            }
        }
    }
    pix_coords.sort_unstable();
    pix_coords.dedup();
    let m_pix = pix_coords.len();
    if m_pix <= k_members + 2 {
        return;
    }

    let coords_ppm: Vec<(f64, f64)> = pix_coords
        .iter()
        .map(|&(ix, iy)| (ax_x.index_to_ppm(ix as f64), ax_y.index_to_ppm(iy as f64)))
        .collect();
    let y_obs: Vec<f64> = pix_coords
        .iter()
        .map(|&(ix, iy)| ref_spec.get(ix, iy) as f64)
        .collect();

    // Parameter vector `q`: `[delta_h_0, delta_x_0, ..., delta_h_{K-1}, delta_x_{K-1}, sigma_h, sigma_x (, eta_h, eta_x)]`
    // Shared line-shape widths across the local overlapping multiplet prevent ill-conditioned width trade-offs
    // while allowing exact sub-pixel centroid & simultaneous multi-peak amplitude deconvolution.
    let fit_voigt = cfg.shape_model == FootprintShapeModel::PseudoVoigt;
    let shape_dim = if fit_voigt { 4 } else { 2 };
    let n_params = 2 * k_members + shape_dim;
    let mut p = vec![0.0_f64; n_params];
    for (c, &r_idx) in members.iter().enumerate() {
        p[2 * c] = seeds[r_idx].delta_h_ppm;
        p[2 * c + 1] = seeds[r_idx].delta_x_ppm;
    }
    p[2 * k_members] = init_sigma_eta[0];
    p[2 * k_members + 1] = init_sigma_eta[1];
    if fit_voigt {
        p[2 * k_members + 2] = init_sigma_eta[2];
        p[2 * k_members + 3] = init_sigma_eta[3];
    }

    let clamp_params = |q: &mut [f64]| {
        for (c, &r_idx) in members.iter().enumerate() {
            let s = &seeds[r_idx];
            q[2 * c] = q[2 * c].clamp(
                s.delta_h_ppm - cfg.max_centroid_shift_ppm[0],
                s.delta_h_ppm + cfg.max_centroid_shift_ppm[0],
            );
            q[2 * c + 1] = q[2 * c + 1].clamp(
                s.delta_x_ppm - cfg.max_centroid_shift_ppm[1],
                s.delta_x_ppm + cfg.max_centroid_shift_ppm[1],
            );
        }
        q[2 * k_members] = q[2 * k_members].clamp(0.45 * dx_ppm, 6.0 * dx_ppm);
        q[2 * k_members + 1] = q[2 * k_members + 1].clamp(0.45 * dy_ppm, 6.0 * dy_ppm);
        if fit_voigt {
            q[2 * k_members + 2] = q[2 * k_members + 2].clamp(0.0, 1.0);
            q[2 * k_members + 3] = q[2 * k_members + 3].clamp(0.0, 1.0);
        }
    };
    clamp_params(&mut p);

    let k_lin = k_members + usize::from(cfg.fit_local_baseline);
    let eval_joint_varpro = |q: &[f64], y_hat_out: &mut [f64]| -> f64 {
        let sh = q[2 * k_members];
        let sx = q[2 * k_members + 1];
        let eh = if fit_voigt { q[2 * k_members + 2] } else { 0.0 };
        let ex = if fit_voigt { q[2 * k_members + 3] } else { 0.0 };

        let mut design = vec![0.0_f64; m_pix * k_lin];
        for (pi, &(px, py)) in coords_ppm.iter().enumerate() {
            for c in 0..k_members {
                design[pi * k_lin + c] =
                    footprint_eval_2d(px, py, q[2 * c], q[2 * c + 1], sh, sx, eh, ex);
            }
            if cfg.fit_local_baseline {
                design[pi * k_lin + k_members] = 1.0;
            }
        }
        let mut g = vec![0.0_f64; k_lin * k_lin];
        let mut rhs = vec![0.0_f64; k_lin];
        for pi in 0..m_pix {
            let row = &design[pi * k_lin..(pi + 1) * k_lin];
            let yo = y_obs[pi];
            for a in 0..k_lin {
                rhs[a] += row[a] * yo;
                for b in 0..=a {
                    g[a * k_lin + b] += row[a] * row[b];
                }
            }
        }
        for a in 0..k_lin {
            for b in 0..a {
                g[b * k_lin + a] = g[a * k_lin + b];
            }
        }
        let amps = solve_spd_cholesky_or_ridge(&g, &rhs, k_lin);
        let mut rss = 0.0_f64;
        for pi in 0..m_pix {
            let row = &design[pi * k_lin..(pi + 1) * k_lin];
            let mut yh = 0.0_f64;
            for a in 0..k_lin {
                yh += row[a] * amps[a];
            }
            y_hat_out[pi] = yh;
            let r = y_obs[pi] - yh;
            rss += r * r;
        }
        rss
    };

    let mut yhat_cur = vec![0.0_f64; m_pix];
    let mut yhat_tmp = vec![0.0_f64; m_pix];
    let mut rss_cur = eval_joint_varpro(&p, &mut yhat_cur);
    let mut lambda = 1.0e-3_f64;

    let mut eps_step = vec![0.0_f64; n_params];
    for c in 0..k_members {
        eps_step[2 * c] = 1.0e-5 * dx_ppm;
        eps_step[2 * c + 1] = 1.0e-5 * dy_ppm;
    }
    eps_step[2 * k_members] = 1.0e-5 * dx_ppm;
    eps_step[2 * k_members + 1] = 1.0e-5 * dy_ppm;
    if fit_voigt {
        eps_step[2 * k_members + 2] = 1.0e-4;
        eps_step[2 * k_members + 3] = 1.0e-4;
    }

    for _iter in 0..30 {
        let mut j_cols = vec![vec![0.0_f64; m_pix]; n_params];
        for j in 0..n_params {
            let mut q_plus = p.clone();
            q_plus[j] += eps_step[j];
            let _ = eval_joint_varpro(&q_plus, &mut yhat_tmp);
            let inv_h = 1.0 / eps_step[j];
            for i in 0..m_pix {
                j_cols[j][i] = (yhat_tmp[i] - yhat_cur[i]) * inv_h;
            }
        }
        let mut jtj = vec![0.0_f64; n_params * n_params];
        let mut jtr = vec![0.0_f64; n_params];
        for i in 0..m_pix {
            let res_i = y_obs[i] - yhat_cur[i];
            for a in 0..n_params {
                jtr[a] += j_cols[a][i] * res_i;
                for b in 0..=a {
                    jtj[a * n_params + b] += j_cols[a][i] * j_cols[b][i];
                }
            }
        }
        for a in 0..n_params {
            for b in 0..a {
                jtj[b * n_params + a] = jtj[a * n_params + b];
            }
        }

        let mut accepted = false;
        for _sub in 0..6 {
            let mut lhs = jtj.clone();
            for d in 0..n_params {
                let diag = lhs[d * n_params + d].max(1.0e-12);
                lhs[d * n_params + d] += lambda * diag;
            }
            let step = solve_spd_cholesky_or_ridge(&lhs, &jtr, n_params);
            let mut cand = p.clone();
            for d in 0..n_params {
                cand[d] += step[d];
            }
            clamp_params(&mut cand);
            let rss_cand = eval_joint_varpro(&cand, &mut yhat_tmp);
            if rss_cand < rss_cur {
                let rel_impr = (rss_cur - rss_cand) / rss_cur.max(1.0e-30);
                p = cand;
                rss_cur = rss_cand;
                yhat_cur.copy_from_slice(&yhat_tmp);
                lambda = (lambda * 0.3).max(1.0e-7);
                accepted = true;
                if rel_impr < 1.0e-9 {
                    break;
                }
                break;
            }
            lambda = (lambda * 5.0).min(1.0e6);
        }
        if !accepted {
            break;
        }
    }

    let sh = p[2 * k_members];
    let sx = p[2 * k_members + 1];
    let eh = if fit_voigt { p[2 * k_members + 2] } else { 0.0 };
    let ex = if fit_voigt { p[2 * k_members + 3] } else { 0.0 };
    for (c, &r_idx) in members.iter().enumerate() {
        footprints[r_idx].delta_h_ppm = p[2 * c];
        footprints[r_idx].delta_x_ppm = p[2 * c + 1];
        footprints[r_idx].sigma_h_ppm = sh;
        footprints[r_idx].sigma_x_ppm = sx;
        footprints[r_idx].eta_h = eh;
        footprints[r_idx].eta_x = ex;
    }
}

/// Estimate stationary 2D background noise spatial autocorrelation $\hat{\rho}(\Delta x, \Delta y)$
/// over lags `|\Delta x| <= max_dx, |\Delta y| <= max_dy` from MAD-masked background pixels.
fn estimate_noise_spatial_autocorr_2d(
    spec: &Spectrum2D,
    mad_sigma: f64,
    max_dx: usize,
    max_dy: usize,
) -> Vec<f64> {
    let nx = spec.nx();
    let ny = spec.ny();
    let w_dx = 2 * max_dx + 1;
    let w_dy = 2 * max_dy + 1;
    let mut rho = vec![0.0_f64; w_dy * w_dx];
    rho[max_dy * w_dx + max_dx] = 1.0;

    if nx < 8 || ny < 8 {
        return rho;
    }

    let mut vals: Vec<f64> = spec
        .data
        .iter()
        .map(|&v| v as f64)
        .filter(|v| v.is_finite())
        .collect();
    let med = median_f64(&mut vals);
    let thr = 2.75 * mad_sigma;

    let mut mask = vec![false; ny * nx];
    let mut centered = vec![0.0_f64; ny * nx];
    let mut var_sum = 0.0_f64;
    let mut var_cnt = 0usize;
    for i in 0..(ny * nx) {
        let d = (spec.data[i] as f64) - med;
        if d.is_finite() && d.abs() <= thr {
            mask[i] = true;
            centered[i] = d;
            var_sum += d * d;
            var_cnt += 1;
        }
    }
    if var_cnt < 64 || var_sum <= 1.0e-30 {
        return rho;
    }
    let bg_var = var_sum / (var_cnt as f64);

    for dy in -(max_dy as isize)..=(max_dy as isize) {
        for dx in -(max_dx as isize)..=(max_dx as isize) {
            if dx == 0 && dy == 0 {
                continue;
            }
            let y_start = (-dy).max(0) as usize;
            let y_end = ((ny as isize) - dy.max(0)).max(0) as usize;
            let x_start = (-dx).max(0) as usize;
            let x_end = ((nx as isize) - dx.max(0)).max(0) as usize;
            let mut c_sum = 0.0_f64;
            let mut c_cnt = 0usize;
            for iy in y_start..y_end {
                let iy2 = ((iy as isize) + dy) as usize;
                for ix in x_start..x_end {
                    let ix2 = ((ix as isize) + dx) as usize;
                    let idx1 = iy * nx + ix;
                    let idx2 = iy2 * nx + ix2;
                    if mask[idx1] && mask[idx2] {
                        c_sum += centered[idx1] * centered[idx2];
                        c_cnt += 1;
                    }
                }
            }
            if c_cnt >= 32 {
                let r = (c_sum / (c_cnt as f64)) / bg_var;
                rho[((dy + (max_dy as isize)) as usize) * w_dx
                    + ((dx + (max_dx as isize)) as usize)] = r.clamp(-0.95, 0.99);
            }
        }
    }
    rho
}

/// Build high-SNR composite reference 2D spectrum for shared footprint fitting.
fn build_reference_composite(
    planes: &[Spectrum2D],
    plane_mad: &[f64],
    seeds: &[TargetPeakSeed],
    explicit_refs: Option<&[usize]>,
) -> Spectrum2D {
    let n_vox = planes[0].data.len();
    if let Some(indices) = explicit_refs.filter(|v| !v.is_empty()) {
        let mut comp = vec![0.0_f64; n_vox];
        let mut w_sum = 0.0_f64;
        for &idx in indices {
            if idx < planes.len() {
                let w = 1.0 / (plane_mad[idx] * plane_mad[idx]).max(1.0e-24);
                for (dst, &src) in comp.iter_mut().zip(planes[idx].data.iter()) {
                    *dst += w * (src as f64);
                }
                w_sum += w;
            }
        }
        if w_sum > 0.0 {
            let inv = 1.0 / w_sum;
            let data: Vec<f32> = comp.into_iter().map(|v| (v * inv) as f32).collect();
            return Spectrum2D {
                name: "relax_ref_composite".to_string(),
                axis_x: planes[0].axis_x.clone(),
                axis_y: planes[0].axis_y.clone(),
                data,
            };
        }
    }

    // Rank planes by median seed SNR; select the single highest-SNR plane (plus any near-identical
    // high-SNR early planes >= 85% of max SNR) so low-SNR decay planes remain 100% independent!
    let mut plane_snrs = Vec::with_capacity(planes.len());
    for (p, sp) in planes.iter().enumerate() {
        let mut sum_abs = 0.0_f64;
        for s in seeds {
            let ix = sp
                .axis_x
                .ppm_to_index(s.delta_h_ppm)
                .round()
                .clamp(0.0, (sp.nx().saturating_sub(1)) as f64) as usize;
            let iy = sp
                .axis_y
                .ppm_to_index(s.delta_x_ppm)
                .round()
                .clamp(0.0, (sp.ny().saturating_sub(1)) as f64) as usize;
            sum_abs += (sp.get(ix, iy) as f64).abs();
        }
        plane_snrs.push(sum_abs / plane_mad[p].max(1.0e-12));
    }
    let best_p = plane_snrs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap_or(0);
    planes[best_p].clone()
}

/// Integrate a stack of pre-processed `Spectrum2D` planes using **shared/locked 2D Gaussian/Voigt
/// footprints** and connected-component linear least squares, with 2D MAD + duplicate-delay
/// replicate variance pooling.
pub fn integrate_relax_planes(
    name: impl Into<String>,
    experiment_kind: RelaxExperimentKind,
    schedule: RelaxSchedule,
    planes: &[Spectrum2D],
    seeds: &[TargetPeakSeed],
    cfg: &RelaxIntegrationConfig,
) -> Result<RelaxSeriesDataset, String> {
    let n_planes = planes.len();
    if n_planes == 0 {
        return Err("integrate_relax_planes requires at least 1 Spectrum2D plane".to_string());
    }
    if schedule.len() != n_planes {
        return Err(format!(
            "schedule length ({}) does not match number of 2D planes ({n_planes})",
            schedule.len()
        ));
    }
    if seeds.is_empty() {
        return Err("integrate_relax_planes requires at least 1 TargetPeakSeed".to_string());
    }

    let nx = planes[0].nx();
    let ny = planes[0].ny();
    for (idx, sp) in planes.iter().enumerate() {
        if sp.nx() != nx || sp.ny() != ny || sp.data.len() != nx * ny {
            return Err(format!(
                "plane {idx} grid dimensions ({}x{}) do not match plane 0 ({nx}x{ny})",
                sp.nx(),
                sp.ny()
            ));
        }
    }

    let plane_noise_mad: Vec<f64> = planes.iter().map(estimate_plane_noise_mad_2d).collect();
    let ref_spec = build_reference_composite(
        planes,
        &plane_noise_mad,
        seeds,
        cfg.ref_plane_indices.as_deref(),
    );

    let m_peaks = seeds.len();
    // Partition peaks into connected overlap components based on seed patch bounding-box intersection first
    let hw_x = cfg.patch_half_width_pts[0];
    let hw_y = cfg.patch_half_width_pts[1];
    let seed_patches: Vec<(usize, usize, usize, usize)> = seeds
        .iter()
        .map(|s| {
            patch_bounds_for_seed(
                &planes[0].axis_x,
                &planes[0].axis_y,
                s.delta_h_ppm,
                s.delta_x_ppm,
                hw_x,
                hw_y,
            )
        })
        .collect();

    let mut parent: Vec<usize> = (0..m_peaks).collect();
    fn uf_find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    for i in 0..m_peaks {
        for j in (i + 1)..m_peaks {
            let (ax0, ax1, ay0, ay1) = seed_patches[i];
            let (bx0, bx1, by0, by1) = seed_patches[j];
            if ax0 <= bx1 && bx0 <= ax1 && ay0 <= by1 && by0 <= ay1 {
                let ri = uf_find(&mut parent, i);
                let rj = uf_find(&mut parent, j);
                if ri != rj {
                    parent[rj] = ri;
                }
            }
        }
    }
    let mut cluster_map: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for i in 0..m_peaks {
        let r = uf_find(&mut parent, i);
        cluster_map.entry(r).or_default().push(i);
    }
    let clusters: Vec<Vec<usize>> = cluster_map.into_values().collect();

    let mut footprints: Vec<LockedPeakFootprint> = seeds
        .iter()
        .map(|s| fit_footprint_on_reference(&ref_spec, s, cfg))
        .collect();

    // Estimate robust baseline line-shape from isolated singleton peaks if present
    let dx_ppm = planes[0].axis_x.step_ppm.abs().max(1.0e-5);
    let dy_ppm = planes[0].axis_y.step_ppm.abs().max(1.0e-4);
    let fit_voigt = cfg.shape_model == FootprintShapeModel::PseudoVoigt;
    let mut single_sh: Vec<f64> = Vec::new();
    let mut single_sx: Vec<f64> = Vec::new();
    let mut single_eh: Vec<f64> = Vec::new();
    let mut single_ex: Vec<f64> = Vec::new();
    for members in &clusters {
        if members.len() == 1 {
            let fp = &footprints[members[0]];
            single_sh.push(fp.sigma_h_ppm);
            single_sx.push(fp.sigma_x_ppm);
            single_eh.push(fp.eta_h);
            single_ex.push(fp.eta_x);
        }
    }
    let init_sigma_eta = if !single_sh.is_empty() {
        [
            median_f64(&mut single_sh),
            median_f64(&mut single_sx),
            median_f64(&mut single_eh),
            median_f64(&mut single_ex),
        ]
    } else {
        [
            1.35 * dx_ppm,
            1.35 * dy_ppm,
            if fit_voigt { 0.20 } else { 0.0 },
            if fit_voigt { 0.20 } else { 0.0 },
        ]
    };

    for (cid, members) in clusters.iter().enumerate() {
        if members.len() >= 2 {
            refine_cluster_footprints_on_reference(
                &ref_spec,
                seeds,
                members,
                init_sigma_eta,
                &seed_patches,
                &mut footprints,
                cfg,
            );
        }
        for &r_idx in members {
            footprints[r_idx].overlap_cluster_id = cid;
        }
    }

    let patches: Vec<(usize, usize, usize, usize)> = footprints
        .iter()
        .map(|fp| {
            patch_bounds_for_seed(
                &planes[0].axis_x,
                &planes[0].axis_y,
                fp.delta_h_ppm,
                fp.delta_x_ppm,
                hw_x,
                hw_y,
            )
        })
        .collect();

    // Estimate spatial noise autocorrelation from plane 0 if enabled
    let max_lag_x = 2 * hw_x;
    let max_lag_y = 2 * hw_y;
    let w_lag_x = 2 * max_lag_x + 1;
    let rho_2d = if cfg.account_spatial_noise_corr {
        Some(estimate_noise_spatial_autocorr_2d(
            &planes[0],
            plane_noise_mad[0],
            max_lag_x,
            max_lag_y,
        ))
    } else {
        None
    };

    let mut raw_amps = vec![vec![0.0_f64; n_planes]; m_peaks];
    let mut base_sigmas = vec![vec![0.0_f64; n_planes]; m_peaks];

    let ax_x = &planes[0].axis_x;
    let ax_y = &planes[0].axis_y;

    for members in &clusters {
        // Unique pixels in union of cluster patches
        let mut pix_coords: Vec<(usize, usize)> = Vec::new();
        for &r_idx in members {
            let (x0, x1, y0, y1) = patches[r_idx];
            for iy in y0..=y1 {
                for ix in x0..=x1 {
                    pix_coords.push((ix, iy));
                }
            }
        }
        pix_coords.sort_unstable();
        pix_coords.dedup();

        let n_pix = pix_coords.len();
        let k_cols = members.len() + usize::from(cfg.fit_local_baseline);
        let mut design = vec![0.0_f64; n_pix * k_cols];
        for (pi, &(ix, iy)) in pix_coords.iter().enumerate() {
            let px = ax_x.index_to_ppm(ix as f64);
            let py = ax_y.index_to_ppm(iy as f64);
            for (col, &r_idx) in members.iter().enumerate() {
                let fp = &footprints[r_idx];
                design[pi * k_cols + col] = footprint_eval_2d(
                    px,
                    py,
                    fp.delta_h_ppm,
                    fp.delta_x_ppm,
                    fp.sigma_h_ppm,
                    fp.sigma_x_ppm,
                    fp.eta_h,
                    fp.eta_x,
                );
            }
            if cfg.fit_local_baseline {
                design[pi * k_cols + members.len()] = 1.0;
            }
        }

        let mut g_mat = vec![0.0_f64; k_cols * k_cols];
        for pi in 0..n_pix {
            let row = &design[pi * k_cols..(pi + 1) * k_cols];
            for a in 0..k_cols {
                for b in 0..k_cols {
                    g_mat[a * k_cols + b] += row[a] * row[b];
                }
            }
        }
        let g_inv = invert_spd_matrix(&g_mat, k_cols);

        // Precompute linear estimator weights W = G^{-1} Psi^T (`[members.len() x n_pix]`) and noise variance factor
        let mut weights = vec![0.0_f64; members.len() * n_pix];
        let mut unit_sd_factors = vec![0.0_f64; members.len()];
        for col in 0..members.len() {
            let mut white_var = 0.0_f64;
            for pi in 0..n_pix {
                let mut w_cp = 0.0_f64;
                for b in 0..k_cols {
                    w_cp += g_inv[col * k_cols + b] * design[pi * k_cols + b];
                }
                weights[col * n_pix + pi] = w_cp;
                white_var += w_cp * w_cp;
            }
            let eff_var = if let Some(ref rho) = rho_2d {
                let mut corr_var = 0.0_f64;
                let w_row = &weights[col * n_pix..(col + 1) * n_pix];
                for p1 in 0..n_pix {
                    let (ix1, iy1) = pix_coords[p1];
                    let w1 = w_row[p1];
                    for p2 in 0..n_pix {
                        let (ix2, iy2) = pix_coords[p2];
                        let dx = (ix1 as isize) - (ix2 as isize);
                        let dy = (iy1 as isize) - (iy2 as isize);
                        if dx.unsigned_abs() <= max_lag_x && dy.unsigned_abs() <= max_lag_y {
                            let r = rho[((dy + (max_lag_y as isize)) as usize) * w_lag_x
                                + ((dx + (max_lag_x as isize)) as usize)];
                            corr_var += w1 * w_row[p2] * r;
                        }
                    }
                }
                corr_var.max(white_var)
            } else {
                white_var
            };
            unit_sd_factors[col] = eff_var.max(1.0e-24).sqrt();
        }

        // Apply linear estimator across every plane `0..n_planes`
        for p_idx in 0..n_planes {
            let sp = &planes[p_idx];
            let mad = plane_noise_mad[p_idx];
            for (col, &r_idx) in members.iter().enumerate() {
                let w_row = &weights[col * n_pix..(col + 1) * n_pix];
                let mut amp = 0.0_f64;
                for (pi, &(ix, iy)) in pix_coords.iter().enumerate() {
                    amp += w_row[pi] * (sp.get(ix, iy) as f64);
                }
                raw_amps[r_idx][p_idx] = amp;
                base_sigmas[r_idx][p_idx] = mad * unit_sd_factors[col];
            }
        }
    }

    // Duplicate-delay / duplicate-offset detection & pooled empirical-Bayes variance refinement
    let x_vals = schedule.primary_values();
    let mut visited = vec![false; n_planes];
    let mut dup_groups: Vec<Vec<usize>> = Vec::new();
    let mut plane_is_duplicate = vec![false; n_planes];

    for i in 0..n_planes {
        if visited[i] {
            continue;
        }
        let mut grp = vec![i];
        visited[i] = true;
        for j in (i + 1)..n_planes {
            if !visited[j] && (x_vals[i] - x_vals[j]).abs() <= cfg.duplicate_tol {
                visited[j] = true;
                grp.push(j);
            }
        }
        if grp.len() >= 2 {
            for &g_idx in &grp {
                plane_is_duplicate[g_idx] = true;
            }
            dup_groups.push(grp);
        }
    }

    let mut duplicate_dof = 0usize;
    let mut chi2_dup = 0.0_f64;
    for grp in &dup_groups {
        let m_g = grp.len();
        for r_idx in 0..m_peaks {
            let mut w_sum = 0.0_f64;
            let mut wy_sum = 0.0_f64;
            for &p_idx in grp {
                let s = base_sigmas[r_idx][p_idx].max(1.0e-18);
                let w = 1.0 / (s * s);
                w_sum += w;
                wy_sum += w * raw_amps[r_idx][p_idx];
            }
            let mean_val = wy_sum / w_sum.max(1.0e-30);
            for &p_idx in grp {
                let s = base_sigmas[r_idx][p_idx].max(1.0e-18);
                let z = (raw_amps[r_idx][p_idx] - mean_val) / s;
                chi2_dup += z * z;
            }
            duplicate_dof += m_g - 1;
        }
    }

    // Empirical-Bayes shrinkage with light prior nu_0 = 4 toward 1.0 so small duplicate counts stay stable
    let duplicate_variance_scale = if duplicate_dof > 0 {
        let nu0 = 4.0_f64;
        ((nu0 + chi2_dup) / (nu0 + (duplicate_dof as f64)))
            .max(0.25)
            .sqrt()
    } else {
        1.0
    };

    let mut traces = Vec::with_capacity(m_peaks);
    for (r_idx, fp) in footprints.into_iter().enumerate() {
        let vol_factor = fp.unit_volume_factor_ppm2();
        let intensities = raw_amps[r_idx].clone();
        let sigmas: Vec<f64> = base_sigmas[r_idx]
            .iter()
            .map(|&s| s * duplicate_variance_scale)
            .collect();
        let volumes: Vec<f64> = intensities.iter().map(|&a| a * vol_factor).collect();
        traces.push(ResidueRelaxTrace {
            footprint: fp,
            intensities,
            sigmas,
            volumes,
        });
    }

    Ok(RelaxSeriesDataset {
        name: name.into(),
        experiment_kind,
        field_mhz: planes[0].axis_x.obs_mhz,
        schedule,
        plane_noise_mad,
        plane_is_duplicate,
        duplicate_variance_scale,
        duplicate_dof,
        traces,
    })
}

/// Naive local-maximum intensity extractor inside a `+-search_radius_pts` window around each seed.
///
/// Provided as a diagnostic comparison reference: at low SNR (e.g. `SNR = 4`), selecting
/// $\max_{(x,y) \in W} S_p(x, y)$ introduces substantial positive noise-selection bias (`> 8%`),
/// whereas `integrate_relax_planes` locked-footprint linear least squares is strictly unbiased.
pub fn extract_naive_local_max_series(
    planes: &[Spectrum2D],
    seeds: &[TargetPeakSeed],
    search_radius_pts: [usize; 2],
) -> Vec<Vec<f64>> {
    let mut out = vec![vec![0.0_f64; planes.len()]; seeds.len()];
    for (p_idx, sp) in planes.iter().enumerate() {
        for (r_idx, s) in seeds.iter().enumerate() {
            let (x0, x1, y0, y1) = patch_bounds_for_seed(
                &sp.axis_x,
                &sp.axis_y,
                s.delta_h_ppm,
                s.delta_x_ppm,
                search_radius_pts[0],
                search_radius_pts[1],
            );
            let mut max_val = f64::NEG_INFINITY;
            for iy in y0..=y1 {
                for ix in x0..=x1 {
                    max_val = max_val.max(sp.get(ix, iy) as f64);
                }
            }
            out[r_idx][p_idx] = max_val;
        }
    }
    out
}

// ============================================================================
// TSV & NEF v1.1 Relaxation-Series Import / Export
// ============================================================================

/// Serialize a `RelaxSeriesDataset` into deterministic ASCII TSV text.
pub fn write_relax_series_tsv_str(ds: &RelaxSeriesDataset) -> String {
    let mut s = String::new();
    s.push_str(&format!("# name: {}\n", ds.name));
    s.push_str(&format!(
        "# experiment_kind: {}\n",
        ds.experiment_kind.as_str()
    ));
    s.push_str(&format!("# field_mhz: {:.6}\n", ds.field_mhz));
    s.push_str(&format!(
        "# schedule_kind: {}\n",
        ds.schedule.schedule_tag()
    ));
    if let RelaxSchedule::LoopCounts { unit_cycle_sec, .. } = &ds.schedule {
        s.push_str(&format!("# unit_cycle_sec: {:.9}\n", unit_cycle_sec));
    }
    s.push_str(&format!(
        "# duplicate_variance_scale: {:.6}\n",
        ds.duplicate_variance_scale
    ));
    s.push_str(&format!("# duplicate_dof: {}\n", ds.duplicate_dof));
    s.push_str("seq_id\tres_name\tatom_h\tatom_x\tdelta_h_ppm\tdelta_x_ppm\tsigma_h_ppm\tsigma_x_ppm\tplane_idx\tx_value\tx_extra\tintensity\tsigma_intensity\tvolume\tplane_mad\tis_duplicate\n");

    let x_vals = ds.schedule.primary_values();
    for tr in &ds.traces {
        let fp = &tr.footprint;
        for (p_idx, &x_val) in x_vals.iter().enumerate() {
            let x_extra = match &ds.schedule {
                RelaxSchedule::LoopCounts { counts, .. } => counts[p_idx].to_string(),
                RelaxSchedule::FrequencyOffsets { offsets } => {
                    format!("{:.6}", offsets[p_idx].offset_ppm)
                }
                RelaxSchedule::VariableDelaysSec { .. } => ".".to_string(),
            };
            s.push_str(&format!(
                "{}\t{}\t{}\t{}\t{:.5}\t{:.5}\t{:.5}\t{:.5}\t{}\t{:.9}\t{}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{}\n",
                fp.seq_id,
                fp.res_name,
                fp.atom_h,
                fp.atom_x,
                fp.delta_h_ppm,
                fp.delta_x_ppm,
                fp.sigma_h_ppm,
                fp.sigma_x_ppm,
                p_idx,
                x_val,
                x_extra,
                tr.intensities[p_idx],
                tr.sigmas[p_idx],
                tr.volumes[p_idx],
                ds.plane_noise_mad[p_idx],
                u8::from(ds.plane_is_duplicate[p_idx]),
            ));
        }
    }
    s
}

/// Write a `RelaxSeriesDataset` to disk as an ASCII TSV file.
pub fn write_relax_series_tsv(
    path: impl AsRef<Path>,
    ds: &RelaxSeriesDataset,
) -> Result<(), String> {
    let txt = write_relax_series_tsv_str(ds);
    fs::write(path.as_ref(), txt).map_err(|e| {
        format!(
            "failed to write relaxation TSV {}: {e}",
            path.as_ref().display()
        )
    })
}

/// Parse a `RelaxSeriesDataset` from ASCII TSV text produced by `write_relax_series_tsv_str`.
pub fn parse_relax_series_tsv_str(text: &str) -> Result<RelaxSeriesDataset, String> {
    if !text.is_ascii() {
        return Err("relaxation series TSV contains non-ASCII characters".to_string());
    }
    let mut name = "relax_series".to_string();
    let mut exp_kind = RelaxExperimentKind::R1;
    let mut field_mhz = 600.0_f64;
    let mut sched_tag = "vdlist".to_string();
    let mut unit_cycle_sec = 0.001_f64;
    let mut dup_scale = 1.0_f64;
    let mut dup_dof = 0usize;

    #[allow(clippy::type_complexity)]
    let mut plane_meta: BTreeMap<usize, (f64, String, f64, bool)> = BTreeMap::new();
    let mut res_order: Vec<(i32, String, String)> = Vec::new();
    #[allow(clippy::type_complexity)]
    let mut res_rows: BTreeMap<
        (i32, String, String),
        (LockedPeakFootprint, BTreeMap<usize, (f64, f64, f64)>),
    > = BTreeMap::new();

    for (line_no, raw) in text.lines().enumerate() {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(cmt) = trimmed.strip_prefix('#') {
            if let Some((k, v)) = cmt.split_once(':') {
                let k = k.trim().to_ascii_lowercase();
                let v = v.trim();
                match k.as_str() {
                    "name" => name = v.to_string(),
                    "experiment_kind" => exp_kind = RelaxExperimentKind::parse_label(v)?,
                    "field_mhz" => {
                        field_mhz = v
                            .parse()
                            .map_err(|e| format!("invalid field_mhz '{v}': {e}"))?;
                    }
                    "schedule_kind" => sched_tag = v.to_ascii_lowercase(),
                    "unit_cycle_sec" => {
                        unit_cycle_sec = v
                            .parse()
                            .map_err(|e| format!("invalid unit_cycle_sec '{v}': {e}"))?;
                    }
                    "duplicate_variance_scale" => {
                        dup_scale = v.parse().unwrap_or(1.0);
                    }
                    "duplicate_dof" => {
                        dup_dof = v.parse().unwrap_or(0);
                    }
                    _ => {}
                }
            }
            continue;
        }
        if trimmed.starts_with("seq_id") {
            continue;
        }

        let cols: Vec<&str> = trimmed.split('\t').collect();
        if cols.len() < 16 {
            return Err(format!(
                "relaxation TSV line {} has {} columns, expected 16",
                line_no + 1,
                cols.len()
            ));
        }
        let seq_id: i32 = cols[0]
            .parse()
            .map_err(|e| format!("invalid seq_id on line {}: {e}", line_no + 1))?;
        let res_name = cols[1].to_string();
        let atom_h = cols[2].to_string();
        let atom_x = cols[3].to_string();
        let dh: f64 = cols[4]
            .parse()
            .map_err(|e| format!("invalid delta_h_ppm on line {}: {e}", line_no + 1))?;
        let dx: f64 = cols[5]
            .parse()
            .map_err(|e| format!("invalid delta_x_ppm on line {}: {e}", line_no + 1))?;
        let sh: f64 = cols[6]
            .parse()
            .map_err(|e| format!("invalid sigma_h_ppm on line {}: {e}", line_no + 1))?;
        let sx: f64 = cols[7]
            .parse()
            .map_err(|e| format!("invalid sigma_x_ppm on line {}: {e}", line_no + 1))?;
        let p_idx: usize = cols[8]
            .parse()
            .map_err(|e| format!("invalid plane_idx on line {}: {e}", line_no + 1))?;
        let x_val: f64 = cols[9]
            .parse()
            .map_err(|e| format!("invalid x_value on line {}: {e}", line_no + 1))?;
        let x_extra = cols[10].to_string();
        let inten: f64 = cols[11]
            .parse()
            .map_err(|e| format!("invalid intensity on line {}: {e}", line_no + 1))?;
        let sig: f64 = cols[12]
            .parse()
            .map_err(|e| format!("invalid sigma_intensity on line {}: {e}", line_no + 1))?;
        let vol: f64 = cols[13]
            .parse()
            .map_err(|e| format!("invalid volume on line {}: {e}", line_no + 1))?;
        let mad: f64 = cols[14]
            .parse()
            .map_err(|e| format!("invalid plane_mad on line {}: {e}", line_no + 1))?;
        let is_dup = cols[15] == "1" || cols[15].eq_ignore_ascii_case("true");

        plane_meta.insert(p_idx, (x_val, x_extra, mad, is_dup));
        let key = (seq_id, atom_h.clone(), atom_x.clone());
        let entry = res_rows.entry(key.clone()).or_insert_with(|| {
            res_order.push(key);
            (
                LockedPeakFootprint {
                    seq_id,
                    res_name,
                    atom_h,
                    atom_x,
                    delta_h_ppm: dh,
                    delta_x_ppm: dx,
                    sigma_h_ppm: sh,
                    sigma_x_ppm: sx,
                    eta_h: 0.0,
                    eta_x: 0.0,
                    overlap_cluster_id: 0,
                },
                BTreeMap::new(),
            )
        });
        entry.1.insert(p_idx, (inten, sig, vol));
    }

    if plane_meta.is_empty() || res_order.is_empty() {
        return Err("relaxation TSV contains zero data rows".to_string());
    }
    let n_planes = plane_meta.len();
    let mut x_vals = Vec::with_capacity(n_planes);
    let mut x_extras = Vec::with_capacity(n_planes);
    let mut plane_noise_mad = Vec::with_capacity(n_planes);
    let mut plane_is_duplicate = Vec::with_capacity(n_planes);

    for p_idx in 0..n_planes {
        let (xv, xe, mad, idup) = plane_meta
            .remove(&p_idx)
            .ok_or_else(|| format!("missing plane_idx {p_idx} in relaxation TSV"))?;
        x_vals.push(xv);
        x_extras.push(xe);
        plane_noise_mad.push(mad);
        plane_is_duplicate.push(idup);
    }

    let schedule = match sched_tag.as_str() {
        "vclist" => {
            let mut counts = Vec::with_capacity(n_planes);
            for s in &x_extras {
                counts.push(s.parse::<u32>().unwrap_or(0));
            }
            RelaxSchedule::LoopCounts {
                counts,
                unit_cycle_sec,
                values_sec: x_vals,
            }
        }
        "fqlist" => {
            let mut offsets = Vec::with_capacity(n_planes);
            for (xv, xe) in x_vals.into_iter().zip(x_extras.iter()) {
                let ppm = xe.parse::<f64>().unwrap_or(0.0);
                offsets.push(FqOffset {
                    offset_hz: xv,
                    offset_ppm: ppm,
                });
            }
            RelaxSchedule::FrequencyOffsets { offsets }
        }
        _ => RelaxSchedule::VariableDelaysSec { delays_sec: x_vals },
    };

    let mut traces = Vec::with_capacity(res_order.len());
    for key in res_order {
        let (fp, mut pmap) = res_rows.remove(&key).unwrap();
        let mut intensities = Vec::with_capacity(n_planes);
        let mut sigmas = Vec::with_capacity(n_planes);
        let mut volumes = Vec::with_capacity(n_planes);
        for p_idx in 0..n_planes {
            let (it, sg, vl) = pmap.remove(&p_idx).ok_or_else(|| {
                format!(
                    "missing plane {p_idx} for residue {} in relaxation TSV",
                    fp.seq_id
                )
            })?;
            intensities.push(it);
            sigmas.push(sg);
            volumes.push(vl);
        }
        traces.push(ResidueRelaxTrace {
            footprint: fp,
            intensities,
            sigmas,
            volumes,
        });
    }

    Ok(RelaxSeriesDataset {
        name,
        experiment_kind: exp_kind,
        field_mhz,
        schedule,
        plane_noise_mad,
        plane_is_duplicate,
        duplicate_variance_scale: dup_scale,
        duplicate_dof: dup_dof,
        traces,
    })
}

/// Read a `RelaxSeriesDataset` from an ASCII TSV file on disk.
pub fn read_relax_series_tsv(path: impl AsRef<Path>) -> Result<RelaxSeriesDataset, String> {
    let txt = fs::read_to_string(path.as_ref()).map_err(|e| {
        format!(
            "failed to read relaxation TSV {}: {e}",
            path.as_ref().display()
        )
    })?;
    parse_relax_series_tsv_str(&txt)
}

/// Serialize a `RelaxSeriesDataset` into an open NEF v1.1 vendor-extensible relaxation saveframe
/// (`save_nmrassign_relaxation_series_<name>`).
pub fn write_relax_series_nef_str(ds: &RelaxSeriesDataset) -> String {
    let safe_name: String = ds
        .name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    let mut s = String::new();
    s.push_str(&format!("data_nmrassign_relax_{safe_name}\n\n"));
    s.push_str("save_nef_nmr_meta_data\n");
    s.push_str("   _nef_nmr_meta_data.sf_category      nef_nmr_meta_data\n");
    s.push_str("   _nef_nmr_meta_data.sf_framecode     nef_nmr_meta_data\n");
    s.push_str("   _nef_nmr_meta_data.format_name      nmr_exchange_format\n");
    s.push_str("   _nef_nmr_meta_data.format_version   1.1\n");
    s.push_str("   _nef_nmr_meta_data.program_name     nmr_assign\n");
    s.push_str(&format!(
        "   _nef_nmr_meta_data.program_version  {}\n",
        crate::VERSION
    ));
    s.push_str("save_\n\n");

    s.push_str(&format!("save_nmrassign_relaxation_series_{safe_name}\n"));
    s.push_str("   _nmrassign_relax_list.sf_category               nmrassign_relaxation_series\n");
    s.push_str(&format!(
        "   _nmrassign_relax_list.sf_framecode              nmrassign_relaxation_series_{safe_name}\n"
    ));
    s.push_str(&format!(
        "   _nmrassign_relax_list.series_name               {}\n",
        safe_name
    ));
    s.push_str(&format!(
        "   _nmrassign_relax_list.experiment_kind           {}\n",
        ds.experiment_kind.as_str()
    ));
    s.push_str(&format!(
        "   _nmrassign_relax_list.spectrometer_frequency_1h {:.6}\n",
        ds.field_mhz
    ));
    s.push_str(&format!(
        "   _nmrassign_relax_list.schedule_kind             {}\n",
        ds.schedule.schedule_tag()
    ));
    if let RelaxSchedule::LoopCounts { unit_cycle_sec, .. } = &ds.schedule {
        s.push_str(&format!(
            "   _nmrassign_relax_list.unit_cycle_sec            {:.9}\n",
            unit_cycle_sec
        ));
    }
    s.push_str(&format!(
        "   _nmrassign_relax_list.duplicate_variance_scale  {:.6}\n",
        ds.duplicate_variance_scale
    ));
    s.push_str(&format!(
        "   _nmrassign_relax_list.duplicate_dof             {}\n\n",
        ds.duplicate_dof
    ));

    s.push_str("   loop_\n");
    s.push_str("      _nmrassign_relax_plane.plane_index\n");
    s.push_str("      _nmrassign_relax_plane.x_value\n");
    s.push_str("      _nmrassign_relax_plane.x_unit\n");
    s.push_str("      _nmrassign_relax_plane.x_extra\n");
    s.push_str("      _nmrassign_relax_plane.noise_mad\n");
    s.push_str("      _nmrassign_relax_plane.is_duplicate\n\n");

    let x_vals = ds.schedule.primary_values();
    let unit = ds.schedule.unit_label();
    for (p_idx, &xv) in x_vals.iter().enumerate() {
        let extra = match &ds.schedule {
            RelaxSchedule::LoopCounts { counts, .. } => counts[p_idx].to_string(),
            RelaxSchedule::FrequencyOffsets { offsets } => {
                format!("{:.6}", offsets[p_idx].offset_ppm)
            }
            RelaxSchedule::VariableDelaysSec { .. } => ".".to_string(),
        };
        s.push_str(&format!(
            "      {} {:.9} {} {} {:.6e} {}\n",
            p_idx,
            xv,
            unit,
            extra,
            ds.plane_noise_mad[p_idx],
            if ds.plane_is_duplicate[p_idx] {
                "true"
            } else {
                "false"
            }
        ));
    }
    s.push_str("   stop_\n\n");

    s.push_str("   loop_\n");
    s.push_str("      _nmrassign_relax_data.index\n");
    s.push_str("      _nmrassign_relax_data.sequence_code\n");
    s.push_str("      _nmrassign_relax_data.residue_name\n");
    s.push_str("      _nmrassign_relax_data.atom_h\n");
    s.push_str("      _nmrassign_relax_data.atom_x\n");
    s.push_str("      _nmrassign_relax_data.delta_h_ppm\n");
    s.push_str("      _nmrassign_relax_data.delta_x_ppm\n");
    s.push_str("      _nmrassign_relax_data.sigma_h_ppm\n");
    s.push_str("      _nmrassign_relax_data.sigma_x_ppm\n");
    s.push_str("      _nmrassign_relax_data.plane_index\n");
    s.push_str("      _nmrassign_relax_data.value\n");
    s.push_str("      _nmrassign_relax_data.value_uncertainty\n");
    s.push_str("      _nmrassign_relax_data.volume\n\n");

    let mut row_id = 1usize;
    for tr in &ds.traces {
        let fp = &tr.footprint;
        for p_idx in 0..x_vals.len() {
            s.push_str(&format!(
                "      {} {} {} {} {} {:.5} {:.5} {:.5} {:.5} {} {:.6e} {:.6e} {:.6e}\n",
                row_id,
                fp.seq_id,
                fp.res_name,
                fp.atom_h,
                fp.atom_x,
                fp.delta_h_ppm,
                fp.delta_x_ppm,
                fp.sigma_h_ppm,
                fp.sigma_x_ppm,
                p_idx,
                tr.intensities[p_idx],
                tr.sigmas[p_idx],
                tr.volumes[p_idx],
            ));
            row_id += 1;
        }
    }
    s.push_str("   stop_\n");
    s.push_str("save_\n");
    s
}

/// Parse a `RelaxSeriesDataset` from a NEF v1.1 `nmrassign_relaxation_series` saveframe.
pub fn parse_relax_series_nef_str(nef_text: &str) -> Result<RelaxSeriesDataset, String> {
    let mut name = "relax_series".to_string();
    let mut exp_kind = RelaxExperimentKind::R1;
    let mut field_mhz = 600.0_f64;
    let mut sched_tag = "vdlist".to_string();
    let mut unit_cycle_sec = 0.001_f64;
    let mut dup_scale = 1.0_f64;
    let mut dup_dof = 0usize;

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum LoopSection {
        None,
        Plane,
        Data,
    }
    let mut sec = LoopSection::None;

    #[allow(clippy::type_complexity)]
    let mut plane_meta: BTreeMap<usize, (f64, String, f64, bool)> = BTreeMap::new();
    let mut res_order: Vec<(i32, String, String)> = Vec::new();
    #[allow(clippy::type_complexity)]
    let mut res_rows: BTreeMap<
        (i32, String, String),
        (LockedPeakFootprint, BTreeMap<usize, (f64, f64, f64)>),
    > = BTreeMap::new();

    for (line_no, raw) in nef_text.lines().enumerate() {
        let t = raw.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        if t == "loop_" {
            sec = LoopSection::None;
            continue;
        }
        if t == "stop_" || t == "save_" {
            sec = LoopSection::None;
            continue;
        }
        if t.starts_with("_nmrassign_relax_plane.") {
            sec = LoopSection::Plane;
            continue;
        }
        if t.starts_with("_nmrassign_relax_data.") {
            sec = LoopSection::Data;
            continue;
        }
        if let Some(rest) = t.strip_prefix("_nmrassign_relax_list.") {
            let mut it = rest.split_whitespace();
            if let (Some(k), Some(v)) = (it.next(), it.next()) {
                match k {
                    "series_name" => name = v.to_string(),
                    "experiment_kind" => exp_kind = RelaxExperimentKind::parse_label(v)?,
                    "spectrometer_frequency_1h" => {
                        field_mhz = v.parse().map_err(|e| format!("invalid NEF MHz: {e}"))?;
                    }
                    "schedule_kind" => sched_tag = v.to_ascii_lowercase(),
                    "unit_cycle_sec" => {
                        unit_cycle_sec = v.parse().unwrap_or(0.001);
                    }
                    "duplicate_variance_scale" => {
                        dup_scale = v.parse().unwrap_or(1.0);
                    }
                    "duplicate_dof" => {
                        dup_dof = v.parse().unwrap_or(0);
                    }
                    _ => {}
                }
            }
            continue;
        }

        match sec {
            LoopSection::Plane => {
                let cols: Vec<&str> = t.split_whitespace().collect();
                if cols.len() < 6 {
                    return Err(format!(
                        "NEF relax_plane line {} has {} tokens, expected 6",
                        line_no + 1,
                        cols.len()
                    ));
                }
                let p_idx: usize = cols[0]
                    .parse()
                    .map_err(|e| format!("invalid NEF plane_index: {e}"))?;
                let xv: f64 = cols[1]
                    .parse()
                    .map_err(|e| format!("invalid NEF x_value: {e}"))?;
                let extra = cols[3].to_string();
                let mad: f64 = cols[4]
                    .parse()
                    .map_err(|e| format!("invalid NEF noise_mad: {e}"))?;
                let is_dup = cols[5].eq_ignore_ascii_case("true") || cols[5] == "1";
                plane_meta.insert(p_idx, (xv, extra, mad, is_dup));
            }
            LoopSection::Data => {
                let cols: Vec<&str> = t.split_whitespace().collect();
                if cols.len() < 13 {
                    return Err(format!(
                        "NEF relax_data line {} has {} tokens, expected 13",
                        line_no + 1,
                        cols.len()
                    ));
                }
                let seq_id: i32 = cols[1]
                    .parse()
                    .map_err(|e| format!("invalid NEF sequence_code: {e}"))?;
                let res_name = cols[2].to_string();
                let atom_h = cols[3].to_string();
                let atom_x = cols[4].to_string();
                let dh: f64 = cols[5]
                    .parse()
                    .map_err(|e| format!("invalid NEF delta_h_ppm: {e}"))?;
                let dx: f64 = cols[6]
                    .parse()
                    .map_err(|e| format!("invalid NEF delta_x_ppm: {e}"))?;
                let sh: f64 = cols[7]
                    .parse()
                    .map_err(|e| format!("invalid NEF sigma_h_ppm: {e}"))?;
                let sx: f64 = cols[8]
                    .parse()
                    .map_err(|e| format!("invalid NEF sigma_x_ppm: {e}"))?;
                let p_idx: usize = cols[9]
                    .parse()
                    .map_err(|e| format!("invalid NEF data plane_index: {e}"))?;
                let val: f64 = cols[10]
                    .parse()
                    .map_err(|e| format!("invalid NEF data value: {e}"))?;
                let sig: f64 = cols[11]
                    .parse()
                    .map_err(|e| format!("invalid NEF data uncertainty: {e}"))?;
                let vol: f64 = cols[12]
                    .parse()
                    .map_err(|e| format!("invalid NEF data volume: {e}"))?;

                let key = (seq_id, atom_h.clone(), atom_x.clone());
                let entry = res_rows.entry(key.clone()).or_insert_with(|| {
                    res_order.push(key);
                    (
                        LockedPeakFootprint {
                            seq_id,
                            res_name,
                            atom_h,
                            atom_x,
                            delta_h_ppm: dh,
                            delta_x_ppm: dx,
                            sigma_h_ppm: sh,
                            sigma_x_ppm: sx,
                            eta_h: 0.0,
                            eta_x: 0.0,
                            overlap_cluster_id: 0,
                        },
                        BTreeMap::new(),
                    )
                });
                entry.1.insert(p_idx, (val, sig, vol));
            }
            LoopSection::None => {}
        }
    }

    if plane_meta.is_empty() || res_order.is_empty() {
        return Err("NEF text contains no nmrassign_relaxation_series saveframe data".to_string());
    }
    let n_planes = plane_meta.len();
    let mut x_vals = Vec::with_capacity(n_planes);
    let mut x_extras = Vec::with_capacity(n_planes);
    let mut plane_noise_mad = Vec::with_capacity(n_planes);
    let mut plane_is_duplicate = Vec::with_capacity(n_planes);

    for p_idx in 0..n_planes {
        let (xv, xe, mad, idup) = plane_meta
            .remove(&p_idx)
            .ok_or_else(|| format!("missing plane_index {p_idx} in NEF relaxation saveframe"))?;
        x_vals.push(xv);
        x_extras.push(xe);
        plane_noise_mad.push(mad);
        plane_is_duplicate.push(idup);
    }

    let schedule = match sched_tag.as_str() {
        "vclist" => {
            let counts = x_extras
                .iter()
                .map(|s| s.parse::<u32>().unwrap_or(0))
                .collect();
            RelaxSchedule::LoopCounts {
                counts,
                unit_cycle_sec,
                values_sec: x_vals,
            }
        }
        "fqlist" => {
            let offsets = x_vals
                .into_iter()
                .zip(x_extras.iter())
                .map(|(xv, xe)| FqOffset {
                    offset_hz: xv,
                    offset_ppm: xe.parse::<f64>().unwrap_or(0.0),
                })
                .collect();
            RelaxSchedule::FrequencyOffsets { offsets }
        }
        _ => RelaxSchedule::VariableDelaysSec { delays_sec: x_vals },
    };

    let mut traces = Vec::with_capacity(res_order.len());
    for key in res_order {
        let (fp, mut pmap) = res_rows.remove(&key).unwrap();
        let mut intensities = Vec::with_capacity(n_planes);
        let mut sigmas = Vec::with_capacity(n_planes);
        let mut volumes = Vec::with_capacity(n_planes);
        for p_idx in 0..n_planes {
            let (it, sg, vl) = pmap
                .remove(&p_idx)
                .ok_or_else(|| format!("missing plane {p_idx} for seq {} in NEF", fp.seq_id))?;
            intensities.push(it);
            sigmas.push(sg);
            volumes.push(vl);
        }
        traces.push(ResidueRelaxTrace {
            footprint: fp,
            intensities,
            sigmas,
            volumes,
        });
    }

    Ok(RelaxSeriesDataset {
        name,
        experiment_kind: exp_kind,
        field_mhz,
        schedule,
        plane_noise_mad,
        plane_is_duplicate,
        duplicate_variance_scale: dup_scale,
        duplicate_dof: dup_dof,
        traces,
    })
}
