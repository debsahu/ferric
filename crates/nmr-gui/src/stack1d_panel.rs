//! Interactive Multi-Spectrum 1D Comparison, Overlay, Tilted Stack, Difference & Kinetics Array
//! Studio (`T14.3`, native desktop & `wasm32-unknown-unknown`).
//!
//! Provenance: Clean-room original implementation derived strictly from peer-reviewed literature
//! and open numerical analysis / spectroscopic visualization principles (zero external source
//! code, macros, or proprietary product conventions inspected):
//! - Savorani, F., Tomasi, G. & Engelsen, S. B. (*J. Magn. Reson.* **2010**, 202, 190–202) —
//!   interval correlation shifting via fast Fourier transform (`FFT`) cross-correlation
//!   for regional peak-drift alignment across 1D NMR spectral arrays.
//! - Cooley, J. W. & Tukey, J. W. (*Math. Comput.* **1965**, 19, 297–301) — radix-2 fast Fourier
//!   transform (`FFT` / `IFFT`) for zero-padded linear cross-correlation.
//! - Keys, R. G. (*IEEE Trans. Acoust. Speech Signal Process.* **1981**, 29, 1153–1160) — cubic
//!   convolution (`Catmull–Rom`) sub-sample spectral interpolation.
//! - Craig, A., Cloarec, O., Holmes, E., Nicholson, J. K. & Lindon, J. C. (*Anal. Chem.* **2006**,
//!   78, 2262–2267) — scaling and normalization methods (total spectral area and internal
//!   reference peak height / integral scaling) for quantitative 1D NMR array analysis.
//! - Mayer, M. & Meyer, B. (*Angew. Chem. Int. Ed.* **1999**, 38, 1784–1788) — Saturation Transfer
//!   Difference (`STD`) NMR difference spectroscopy ($S_{\text{off}}(\delta) - \alpha S_{\text{on}}(\delta)$).
//! - Dalvit, C., Fogliatto, G., Stewart, A., Veronesi, M. & Stockman, B. (*J. Biomol. NMR*
//!   **2001**, 21, 349–359) — Water-Ligand Observed via Gradient Spectroscopy (`WaterLOGSY`) 1D
//!   ligand-binding arrays.

use std::f64::consts::PI;
use std::fmt::Write as _;

use eframe::egui;
use nmr_core::metabolomics::CohortSample1D;
use nmr_core::multiplet::{
    analyze_multiplet_peaks, format_journal_nmr_report, JournalNmrReport, MultipletAnalysis,
    MultipletConfig, MultipletSubPeak, PublicationStyle, ReportNucleus, ReportedSignal,
    TextEncoding,
};
use nmr_core::nmr1d::{
    apply_phase_1d, auto_calibrate_solvent_1d, auto_phase_and_apply_1d, correct_baseline_1d,
    estimate_noise_sigma_1d, hilbert_reconstruct_imaginary_1d, lookup_deuterated_solvent,
    match_trace_impurities, pick_peaks_1d, solvent_residual_shifts, AutoPhase1DConfig,
    BaselineConfig1D, BaselineMethod1D, Experiment1DKind, Nucleus1D, PeakPickConfig1D, Pivot1D,
    Spectrum1D,
};
use nmr_core::ops::fft_arbitrary_in_place;
use nmr_core::spectrum::AxisPpm;

/// Primary mouse tool mode on the interactive 1D NMR canvas.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ToolMode1d {
    /// Two-click (`Click 1 -> Click 2`) or click-and-drag horizontal chemical-shift zoom.
    #[default]
    Zoom,
    /// Left-drag horizontal chemical-shift pan.
    Pan,
    /// Left-drag region-of-interest (`ROI`) integral bracket creation (`Shift + Drag` works in any mode).
    IntegrateRoi,
}

impl ToolMode1d {
    pub const ALL: [Self; 3] = [Self::Zoom, Self::Pan, Self::IntegrateRoi];

    pub const fn label(self) -> &'static str {
        match self {
            Self::Zoom => "🔍 Zoom",
            Self::Pan => "✋ Pan",
            Self::IntegrateRoi => "∫ Integrate ROI",
        }
    }

    pub const fn tooltip(self) -> &'static str {
        match self {
            Self::Zoom => {
                "Click 1st point -> Click 2nd point (or Click-Drag) to zoom horizontally; Scroll wheel scales vertical peak intensity"
            }
            Self::Pan => {
                "Drag horizontally to pan across chemical shift (Right-drag, Middle-drag, or Space+Drag also pan in Zoom mode)"
            }
            Self::IntegrateRoi => {
                "Drag across a multiplet or region to add an integrated ROI bracket (Shift+Drag also integrates in Zoom mode)"
            }
        }
    }
}

/// Active analytical tab in the 4-tab 1D workstation bottom dock.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum InspectorTab1d {
    #[default]
    Peaks,
    Multiplets,
    PhaseBaseline,
    MultiTraceStack,
}

/// Selectable automatic baseline correction model for the 1D Phase & Baseline tab.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum BaselineModelChoice1d {
    #[default]
    Whittaker,
    AsymmetricWhittaker,
    Bernstein,
}

impl BaselineModelChoice1d {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Whittaker => "Whittaker (Wavelet)",
            Self::AsymmetricWhittaker => "Asymmetric Whittaker",
            Self::Bernstein => "Bernstein Polynomial",
        }
    }
}

/// First-order phase (`PHC1`) pivot selection rule.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum PivotMode1d {
    #[default]
    MaxPeak,
    AtPpm,
}

/// Picked 1D resonance peak entry with frequency, SNR, linewidth, and solvent/impurity annotation.
#[derive(Clone, Debug, PartialEq)]
pub struct Peak1dEntry {
    pub ppm: f64,
    pub hz: f64,
    pub height: f64,
    pub snr: f64,
    pub fwhm_hz: f64,
    pub note: String,
    pub is_solvent_or_impurity: bool,
}

/// Format a 0-based multiplet index into spreadsheet-style uppercase letters (`0 -> "A"`, `26 -> "AA"`).
pub fn multiplet_letter_id(mut idx: usize) -> String {
    let mut chars = Vec::new();
    loop {
        chars.push((b'A' + (idx % 26) as u8) as char);
        if idx < 26 {
            break;
        }
        idx = idx / 26 - 1;
    }
    chars.into_iter().rev().collect()
}

/// Curated dark-workbench color palette (`[r, g, b, a]`) for overlaid & stacked 1D NMR traces.
pub const STACK1D_TRACE_PALETTE: [[u8; 4]; 8] = [
    [56, 189, 248, 255],  // Sky cyan
    [52, 211, 153, 255],  // Emerald mint
    [251, 191, 36, 255],  // Solar amber
    [192, 132, 252, 255], // Orchid violet
    [251, 113, 133, 255], // Coral rose
    [45, 212, 191, 255],  // Aqua teal
    [250, 204, 21, 255],  // Bright gold
    [129, 140, 248, 255], // Periwinkle indigo
];

/// Return a default `[r, g, b, a]` trace color cycling cleanly by trace index.
#[inline]
pub const fn default_trace_color(idx: usize) -> [u8; 4] {
    STACK1D_TRACE_PALETTE[idx % STACK1D_TRACE_PALETTE.len()]
}

/// Primary display mode for the multi-spectrum 1D comparison & kinetics studio (`T14.3`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Stack1dViewMode {
    /// Color-coded multi-trace overlay sharing a common baseline with per-trace scale,
    /// vertical offset, chemical-shift offset, visibility, and lock.
    #[default]
    SuperimposedOverlay,
    /// Vertical + horizontal/oblique angle offset stack for titration, time-course, and
    /// multiplicity-edited (`13C` / `DEPT-90` / `DEPT-135`) clarity.
    StackedOffset,
    /// Residual / difference trace $D_i(\delta) = S_i(\delta) - \alpha_i S_{\text{ref}}(\delta)$
    /// with interactive manual $\alpha$ or ordinary least-squares (`OLS`) auto-matched $\alpha_i$
    /// over a selectable reference region (`STD-NMR`, batch impurity QC, reaction monitoring).
    DifferenceSpectrum,
    /// Pseudo-2D raster intensity heatmap across array index / time / ligand equivalents / pH /
    /// temperature (`x` = inverted NMR ppm high-to-low, `y` = array trace / covariate).
    Pseudo2dHeatmap,
}

impl Stack1dViewMode {
    /// Ordered catalog of all 4 [`Stack1dViewMode`] options.
    pub const ALL: [Self; 4] = [
        Self::SuperimposedOverlay,
        Self::StackedOffset,
        Self::DifferenceSpectrum,
        Self::Pseudo2dHeatmap,
    ];

    /// Short human-readable UI tab label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::SuperimposedOverlay => "Overlay",
            Self::StackedOffset => "Tilted Stack",
            Self::DifferenceSpectrum => "Difference (S_i - α·S_ref)",
            Self::Pseudo2dHeatmap => "Pseudo-2D Heatmap",
        }
    }
}

/// Physical meaning of the numeric per-trace array covariate (`time`, `[L]/[P]`, `pH`, `T`, etc.).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum CovariateKind {
    #[default]
    ArrayIndex,
    TimeSeconds,
    TimeMinutes,
    LigandEquivalents,
    ConcentrationMicromolar,
    Ph,
    TemperatureKelvin,
}

impl CovariateKind {
    pub const ALL: [Self; 7] = [
        Self::ArrayIndex,
        Self::TimeSeconds,
        Self::TimeMinutes,
        Self::LigandEquivalents,
        Self::ConcentrationMicromolar,
        Self::Ph,
        Self::TemperatureKelvin,
    ];

    pub const fn label(self) -> &'static str {
        match self {
            Self::ArrayIndex => "Array Index (#)",
            Self::TimeSeconds => "Time (s)",
            Self::TimeMinutes => "Time (min)",
            Self::LigandEquivalents => "Ligand ([L]/[P] eq)",
            Self::ConcentrationMicromolar => "Concentration (μM)",
            Self::Ph => "pH",
            Self::TemperatureKelvin => "Temperature (K)",
        }
    }

    pub const fn unit_suffix(self) -> &'static str {
        match self {
            Self::ArrayIndex => "",
            Self::TimeSeconds => " s",
            Self::TimeMinutes => " min",
            Self::LigandEquivalents => " eq",
            Self::ConcentrationMicromolar => " μM",
            Self::Ph => "",
            Self::TemperatureKelvin => " K",
        }
    }
}

/// Cross-spectrum intensity normalization rule (Craig et al., *Anal. Chem.* 2006, 78, 2262–2267).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum NormalizationMode {
    /// Preserve raw or manually scaled trace intensities (`scale = 1.0`).
    #[default]
    None,
    /// Normalize each unlocked trace so its signed trapezoidal total spectral integral equals `1.0`
    /// (automatically falling back to absolute area if the net signed integral vanishes).
    TotalArea,
    /// Normalize each unlocked trace so $\int |S_i(\delta)|\,d\delta = 1.0$ (suitable for
    /// multiplicity-edited `DEPT-135` traces with inverted `CH2` resonances).
    TotalAbsArea,
    /// Normalize each unlocked trace so $\max_{\delta \in [\text{ppm\_min}, \text{ppm\_max}]} |S_i(\delta)| = 1.0$.
    ReferencePeakHeight { ppm_min: f64, ppm_max: f64 },
    /// Normalize each unlocked trace so its trapezoidal integral over `[ppm_min, ppm_max]` equals
    /// `1.0` (invariant to $T_2^*$ line-broadening across temperature/pH/titration series).
    ReferenceRegionIntegral { ppm_min: f64, ppm_max: f64 },
}

/// Estimation rule for the scaling factor $\alpha_i$ in $D_i(\delta) = S_i(\delta) - \alpha_i S_{\text{ref}}(\delta)$.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DifferenceAlphaMode {
    /// Apply `manual_alpha` (or per-trace `Trace1d::alpha_override` if set).
    Manual,
    /// Analytically minimize $\sum_{\delta \in [\text{ppm\_min}, \text{ppm\_max}]} (S_i(\delta) - \alpha_i S_{\text{ref}}(\delta) - \beta_i)^2$
    /// over `[ppm_min, ppm_max]` (`fit_intercept = false` solves pure `S_i - alpha * S_ref`).
    LeastSquaresRegion {
        ppm_min: f64,
        ppm_max: f64,
        fit_intercept: bool,
    },
}

impl Default for DifferenceAlphaMode {
    fn default() -> Self {
        Self::LeastSquaresRegion {
            ppm_min: 0.80,
            ppm_max: 1.60,
            fit_intercept: false,
        }
    }
}

/// Scope of interval-correlation FFT cross-correlation peak-shift alignment (Savorani et al., 2010).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AlignmentScope {
    /// Estimate sub-sample chemical-shift drift inside the first configured reference interval
    /// and apply a uniform whole-trace `shift_offset_ppm` correction.
    GlobalShiftFromRegion,
    /// Independently shift each configured `[ppm_min, ppm_max]` interval in-place using cubic
    /// convolution resampling and endpoint boundary fill (Savorani et al., 2010).
    #[default]
    LocalIntervals,
}

impl AlignmentScope {
    pub const fn label(self) -> &'static str {
        match self {
            Self::GlobalShiftFromRegion => "Global Shift (from Region)",
            Self::LocalIntervals => "Local Intervals (FFT)",
        }
    }
}

/// User-defined chemical-shift interval `[ppm_min, ppm_max]` for regional FFT alignment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AlignmentInterval {
    pub ppm_min: f64,
    pub ppm_max: f64,
    /// Maximum allowed lag search half-window in ppm (`> 0`).
    pub max_shift_ppm: f64,
}

impl AlignmentInterval {
    pub fn new(ppm_a: f64, ppm_b: f64, max_shift_ppm: f64) -> Self {
        Self {
            ppm_min: ppm_a.min(ppm_b),
            ppm_max: ppm_a.max(ppm_b),
            max_shift_ppm: max_shift_ppm.abs().max(1e-4),
        }
    }
}

/// Diagnostic record produced for each aligned `(trace, interval)` pair during FFT alignment.
#[derive(Clone, Debug, PartialEq)]
pub struct IntervalShiftRecord {
    pub trace_id: String,
    pub ppm_min: f64,
    pub ppm_max: f64,
    /// Estimated sub-sample lag $\hat{\tau}$ in grid points.
    pub shift_points: f64,
    /// Applied chemical shift correction $\Delta\delta_{\text{corr}}$ in ppm.
    pub shift_ppm: f64,
    /// Normalized zero-mean cross-correlation peak height in `[-1.0, 1.0]`.
    pub correlation: f64,
}

/// Selected chemical-shift region of interest (`ROI`) `[ppm_min, ppm_max]` integrated across all
/// traces to build kinetics, titration, or reaction-progress matrices.
#[derive(Clone, Debug, PartialEq)]
pub struct RoiRegion1d {
    pub label: String,
    pub ppm_min: f64,
    pub ppm_max: f64,
    /// Subtract straight-line chord connecting `S(ppm_min)` and `S(ppm_max)` when `true`.
    pub edge_baseline_correction: bool,
}

impl RoiRegion1d {
    pub fn new(label: impl Into<String>, ppm_a: f64, ppm_b: f64) -> Self {
        Self {
            label: label.into(),
            ppm_min: ppm_a.min(ppm_b),
            ppm_max: ppm_a.max(ppm_b),
            edge_baseline_correction: false,
        }
    }

    #[inline]
    pub fn ordered_bounds(&self) -> (f64, f64) {
        (
            self.ppm_min.min(self.ppm_max),
            self.ppm_min.max(self.ppm_max),
        )
    }
}

/// Formula defining `normalized_ratio` in the multi-trace ROI kinetics / titration table.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RoiRatioFormula {
    /// Smart kinetics/titration default:
    /// - When `k >= 2` ROIs exist: `area_region_1 / sum_{j=1..k} area_region_j` (reaction progress / mole fraction).
    /// - When `k == 1` ROI exists: `area_region_1(trace_i) / area_region_1(ref_trace)` (normalized decay/growth).
    #[default]
    AutoReactionOrDecay,
    /// Mole-fraction / conversion ratio `area_region_1 / sum_{j=1..k} area_region_j`.
    FirstOverSum,
    /// Internal-standard ratio `area[target_idx] / area[ref_idx]`.
    TargetOverRefRegion { target_idx: usize, ref_idx: usize },
    /// Relative to reference trace `area_i[region_idx] / area_ref[region_idx]`.
    RelativeToRefTrace { region_idx: usize },
}

impl RoiRatioFormula {
    pub const fn label(self) -> &'static str {
        match self {
            Self::AutoReactionOrDecay => "Auto (A1 / ΣA or I/I_ref)",
            Self::FirstOverSum => "Mole Fraction (A1 / ΣA)",
            Self::TargetOverRefRegion { .. } => "Internal Std (A1 / A_ref)",
            Self::RelativeToRefTrace { .. } => "Decay Ratio (A1_i / A1_ref)",
        }
    }
}

/// One trace row of the multi-trace ROI integration table.
#[derive(Clone, Debug, PartialEq)]
pub struct RoiMatrixRow {
    pub trace_id: String,
    pub covariate: f64,
    pub areas: Vec<f64>,
    pub normalized_ratio: f64,
}

/// Complete multi-trace ROI integration matrix across `N` traces and `K` chemical-shift regions.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RoiMatrix {
    pub regions: Vec<RoiRegion1d>,
    pub rows: Vec<RoiMatrixRow>,
}

impl RoiMatrix {
    /// Format the ROI matrix as a clean tab-separated values (`TSV`) document with canonical
    /// columns: `trace_id`, `covariate`, `area_region_1..k`, `normalized_ratio`.
    pub fn to_tsv_string(&self) -> String {
        let k = self.regions.len();
        let mut out = String::with_capacity(128 + self.rows.len() * (32 + 18 * k));
        out.push_str("trace_id\tcovariate");
        for j in 0..k {
            let _ = write!(out, "\tarea_region_{}", j + 1);
        }
        out.push_str("\tnormalized_ratio\n");

        for row in &self.rows {
            let clean_id: String = row
                .trace_id
                .chars()
                .map(|c| {
                    if c == '\t' || c == '\n' || c == '\r' {
                        '_'
                    } else {
                        c
                    }
                })
                .collect();
            let _ = write!(out, "{clean_id}\t{:.6}", row.covariate);
            for &area in &row.areas {
                let _ = write!(out, "\t{:.9e}", area);
            }
            let _ = writeln!(out, "\t{:.9}", row.normalized_ratio);
        }
        out
    }
}

/// Evaluated difference trace $D_i(\delta) = S_i(\delta) - (\alpha_i S_{\text{ref}}(\delta) + \beta_i)$.
#[derive(Clone, Debug, PartialEq)]
pub struct DifferenceTraceResult {
    pub ppm: Vec<f64>,
    pub diff_intensities: Vec<f64>,
    pub alpha: f64,
    pub beta: f64,
    pub residual_rms: f64,
}

// ============================================================================
// Level-of-Detail (`LOD`) Min/Max Pyramid for 60+ fps 32 × 64k Rendering
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq)]
struct LodBucket {
    min_val: f64,
    max_val: f64,
    min_idx: u32,
    max_idx: u32,
}

/// Multi-resolution power-of-two min/max envelope pyramid guaranteeing zero peak-apex loss and
/// sub-millisecond viewport decimation across `32+` stacked `64k`-point 1D NMR traces.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LodPyramid1d {
    /// Level `l` (`0`-indexed) summarizes blocks of size `2^(l + 1)` samples from `intensities`.
    levels: Vec<Vec<LodBucket>>,
}

impl LodPyramid1d {
    /// Build a complete dyadic min/max pyramid from `intensities` in $O(N)$ time.
    pub fn build(intensities: &[f64]) -> Self {
        let n = intensities.len();
        if n < 4 {
            return Self { levels: Vec::new() };
        }
        let mut levels: Vec<Vec<LodBucket>> = Vec::new();

        // Level 0: stride-2 pairs
        let n0 = n.div_ceil(2);
        let mut lvl0 = Vec::with_capacity(n0);
        for b in 0..n0 {
            let i0 = 2 * b;
            let i1 = (i0 + 1).min(n - 1);
            let v0 = intensities[i0];
            let v1 = intensities[i1];
            let (min_val, min_idx) = if v0 <= v1 {
                (v0, i0 as u32)
            } else {
                (v1, i1 as u32)
            };
            let (max_val, max_idx) = if v0 >= v1 {
                (v0, i0 as u32)
            } else {
                (v1, i1 as u32)
            };
            lvl0.push(LodBucket {
                min_val,
                max_val,
                min_idx,
                max_idx,
            });
        }
        levels.push(lvl0);

        while let Some(prev) = levels.last() {
            if prev.len() <= 64 {
                break;
            }
            let next_len = prev.len().div_ceil(2);
            let mut next = Vec::with_capacity(next_len);
            for b in 0..next_len {
                let a = prev[2 * b];
                let c = prev[(2 * b + 1).min(prev.len() - 1)];
                let (min_val, min_idx) = if a.min_val <= c.min_val {
                    (a.min_val, a.min_idx)
                } else {
                    (c.min_val, c.min_idx)
                };
                let (max_val, max_idx) = if a.max_val >= c.max_val {
                    (a.max_val, a.max_idx)
                } else {
                    (c.max_val, c.max_idx)
                };
                next.push(LodBucket {
                    min_val,
                    max_val,
                    min_idx,
                    max_idx,
                });
            }
            levels.push(next);
        }

        Self { levels }
    }
}

/// Equality-preserving wrapper attaching a full [`Spectrum1D`] (real + optional imaginary + axis/solvent metadata)
/// to a [`Trace1d`] for live interactive phasing, baseline correction, solvent calibration, and multiplet analysis.
#[derive(Clone, Debug, PartialEq)]
pub struct AttachedSpectrum1d(pub Spectrum1D);

/// Single 1D NMR spectrum trace inside [`Stack1dState`].
#[derive(Clone, Debug, PartialEq)]
pub struct Trace1d {
    pub id: String,
    pub label: String,
    pub covariate: f64,
    ppm: Vec<f64>,
    is_descending: bool,
    raw_intensities: Vec<f64>,
    intensities: Vec<f64>,
    pub color_rgba: [u8; 4],
    pub scale: f64,
    pub vertical_offset: f64,
    /// Uniform chemical-shift alignment delta added to `ppm[i]` (`effective_ppm = ppm[i] + shift_offset_ppm`).
    pub shift_offset_ppm: f64,
    /// Optional per-trace manual $\alpha$ override in `DifferenceSpectrum` mode.
    pub alpha_override: Option<f64>,
    pub visible: bool,
    pub locked: bool,
    /// True when populated by a built-in synthetic demo so real user drops replace demo traces cleanly.
    pub is_demo: bool,
    /// Optional underlying [`Spectrum1D`] enabling live interactive `PHC0`/`PHC1` phasing & multiplet analysis.
    pub raw_spec: Option<AttachedSpectrum1d>,
    generation: u64,
    lod: LodPyramid1d,
}

impl Trace1d {
    /// Construct a validated monotonic 1D NMR trace from `ppm` and `intensities`.
    ///
    /// Supports both standard Bruker/NMRPipe descending `ppm` axes (`high -> low`) and ascending
    /// `ppm` axes (`low -> high`). Non-finite samples are sanitized to `0.0`, and unsorted axes
    /// are sorted into canonical descending NMR order automatically.
    pub fn new(
        id: impl Into<String>,
        label: impl Into<String>,
        covariate: f64,
        ppm: Vec<f64>,
        intensities: Vec<f64>,
    ) -> Self {
        let len = ppm.len().min(intensities.len());
        let mut pairs: Vec<(f64, f64)> = ppm
            .into_iter()
            .zip(intensities)
            .take(len)
            .filter(|(p, _)| p.is_finite())
            .map(|(p, y)| (p, if y.is_finite() { y } else { 0.0 }))
            .collect();

        let is_strictly_desc = pairs.windows(2).all(|w| w[0].0 > w[1].0);
        let is_strictly_asc = pairs.windows(2).all(|w| w[0].0 < w[1].0);
        let is_descending = if is_strictly_desc {
            true
        } else if is_strictly_asc {
            false
        } else {
            pairs.sort_by(|a, b| b.0.total_cmp(&a.0));
            pairs.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-13);
            true
        };

        let clean_ppm: Vec<f64> = pairs.iter().map(|&(p, _)| p).collect();
        let clean_y: Vec<f64> = pairs.iter().map(|&(_, y)| y).collect();
        let lod = LodPyramid1d::build(&clean_y);

        Self {
            id: id.into(),
            label: label.into(),
            covariate: if covariate.is_finite() {
                covariate
            } else {
                0.0
            },
            ppm: clean_ppm,
            is_descending,
            raw_intensities: clean_y.clone(),
            intensities: clean_y,
            color_rgba: STACK1D_TRACE_PALETTE[0],
            scale: 1.0,
            vertical_offset: 0.0,
            shift_offset_ppm: 0.0,
            alpha_override: None,
            visible: true,
            locked: false,
            is_demo: false,
            raw_spec: None,
            generation: 1,
            lod,
        }
    }

    /// Construct a [`Trace1d`] from a full [`Spectrum1D`], retaining `raw_spec` for live interactive
    /// phasing (`PHC0`/`PHC1`), Whittaker baseline correction, solvent calibration, and multiplet reporting.
    pub fn from_spectrum1d(id: impl Into<String>, covariate: f64, spec: Spectrum1D) -> Self {
        let ppm: Vec<f64> = (0..spec.len()).map(|i| spec.ppm_at(i)).collect();
        let y = spec.real.clone();
        let lbl = spec.name.clone();
        let mut tr = Self::new(id, lbl, covariate, ppm, y);
        tr.raw_spec = Some(AttachedSpectrum1d(spec));
        tr
    }

    /// Mark whether this trace belongs to a synthetic demo preset.
    #[inline]
    pub fn with_demo(mut self, is_demo: bool) -> Self {
        self.is_demo = is_demo;
        self
    }

    /// Synchronize `ppm` and working `intensities` after mutating an attached [`Spectrum1D`].
    pub fn sync_from_spectrum1d(&mut self, spec: &Spectrum1D) {
        if self.locked {
            return;
        }
        let ppm_vec: Vec<f64> = (0..spec.len()).map(|i| spec.ppm_at(i)).collect();
        if ppm_vec.len() == spec.real.len() && ppm_vec.len() >= 2 {
            self.ppm = ppm_vec;
            self.is_descending = self.ppm.windows(2).all(|w| w[0] > w[1]);
            self.intensities = spec.real.clone();
            self.lod = LodPyramid1d::build(&self.intensities);
            self.generation = self.generation.wrapping_add(1);
        }
    }

    /// Construct a [`Trace1d`] directly from an [`AxisPpm`] descriptor and `f32` spectral slice.
    pub fn from_axis_and_f32(
        id: impl Into<String>,
        label: impl Into<String>,
        covariate: f64,
        axis: &AxisPpm,
        data: &[f32],
    ) -> Self {
        let n = axis.size.min(data.len());
        let mut ppm = Vec::with_capacity(n);
        let mut intensities = Vec::with_capacity(n);
        for (i, &val) in data.iter().take(n).enumerate() {
            ppm.push(axis.index_to_ppm(i as f64));
            intensities.push(val as f64);
        }
        Self::new(id, label, covariate, ppm, intensities)
    }

    /// Builder helper assigning the trace display color (`[r, g, b, a]`).
    #[inline]
    pub fn with_color(mut self, color_rgba: [u8; 4]) -> Self {
        self.color_rgba = color_rgba;
        self
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.ppm.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.ppm.is_empty()
    }

    #[inline]
    pub fn raw_ppm(&self) -> &[f64] {
        &self.ppm
    }

    #[inline]
    pub fn raw_intensities(&self) -> &[f64] {
        &self.raw_intensities
    }

    #[inline]
    pub fn intensities(&self) -> &[f64] {
        &self.intensities
    }

    #[inline]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Effective chemical shift at sample `idx` incorporating `shift_offset_ppm`.
    #[inline]
    pub fn effective_ppm(&self, idx: usize) -> f64 {
        self.ppm[idx] + self.shift_offset_ppm
    }

    /// Effective scaled intensity at sample `idx` incorporating `scale`.
    #[inline]
    pub fn effective_intensity(&self, idx: usize) -> f64 {
        self.intensities[idx] * self.scale
    }

    /// Replace working `intensities` (rebuilding the LOD pyramid) unless `self.locked` is set.
    pub fn set_working_intensities(&mut self, next: Vec<f64>) {
        if self.locked || next.len() != self.ppm.len() {
            return;
        }
        self.intensities = next;
        self.lod = LodPyramid1d::build(&self.intensities);
        self.generation = self.generation.wrapping_add(1);
    }

    /// Restore raw unaligned, unnormalized intensities and reset `scale = 1.0`, `shift_offset_ppm = 0.0`
    /// unless `self.locked` is set.
    pub fn reset_to_raw(&mut self) {
        if self.locked {
            return;
        }
        self.intensities.clone_from(&self.raw_intensities);
        self.scale = 1.0;
        self.shift_offset_ppm = 0.0;
        self.lod = LodPyramid1d::build(&self.intensities);
        self.generation = self.generation.wrapping_add(1);
    }

    /// Ordered `(min_ppm, max_ppm)` chemical shift bounds incorporating `shift_offset_ppm`.
    pub fn ppm_bounds(&self) -> Option<(f64, f64)> {
        let first = *self.ppm.first()? + self.shift_offset_ppm;
        let last = *self.ppm.last()? + self.shift_offset_ppm;
        Some((first.min(last), first.max(last)))
    }

    /// Mean unsigned grid step $|\Delta\delta|$ in ppm.
    pub fn mean_step_ppm(&self) -> f64 {
        if self.ppm.len() < 2 {
            return 1e-3;
        }
        let (lo, hi) = self.ppm_bounds().unwrap_or((0.0, 1.0));
        ((hi - lo) / ((self.ppm.len() - 1) as f64)).max(1e-9)
    }

    /// Maximum absolute effective intensity $\max_k |S_i(\delta_k)|$ across the trace.
    pub fn max_abs_effective(&self) -> f64 {
        let s = self.scale.abs();
        if let Some(top) = self.lod.levels.last() {
            top.iter().fold(0.0_f64, |acc, b| {
                acc.max(b.min_val.abs()).max(b.max_val.abs())
            }) * s
        } else {
            self.intensities
                .iter()
                .fold(0.0_f64, |acc, &v| acc.max(v.abs()))
                * s
        }
    }

    /// Inclusive native index span `[i_start, i_end]` covering `[ppm_min, ppm_max]` (with 1-sample
    /// margin on each side for clean boundary clipping).
    pub fn index_range_for_ppm(&self, ppm_a: f64, ppm_b: f64) -> Option<(usize, usize)> {
        let n = self.ppm.len();
        if n == 0 {
            return None;
        }
        let lo = ppm_a.min(ppm_b) - self.shift_offset_ppm;
        let hi = ppm_a.max(ppm_b) - self.shift_offset_ppm;

        if self.is_descending {
            // ppm[0] >= ppm[1] >= ... >= ppm[n-1]
            if hi < self.ppm[n - 1] || lo > self.ppm[0] {
                return None;
            }
            let i0 = self.ppm.partition_point(|&p| p > hi).saturating_sub(1);
            let i1 = self.ppm.partition_point(|&p| p >= lo).min(n - 1);
            Some((i0.min(i1), i0.max(i1)))
        } else {
            // ppm[0] <= ppm[1] <= ... <= ppm[n-1]
            if hi < self.ppm[0] || lo > self.ppm[n - 1] {
                return None;
            }
            let i0 = self.ppm.partition_point(|&p| p < lo).saturating_sub(1);
            let i1 = self.ppm.partition_point(|&p| p <= hi).min(n - 1);
            Some((i0.min(i1), i0.max(i1)))
        }
    }

    /// Evaluate exact piecewise-linear interpolated scaled intensity $S_i(\delta)$ at `target_ppm`
    /// in $O(\log N)$ time (returning `0.0` outside the trace's chemical-shift domain).
    pub fn sample_at_ppm(&self, target_ppm: f64) -> f64 {
        let n = self.ppm.len();
        if n == 0 || !target_ppm.is_finite() {
            return 0.0;
        }
        let raw_q = target_ppm - self.shift_offset_ppm;
        if n == 1 {
            return if (raw_q - self.ppm[0]).abs() <= 1e-6 {
                self.effective_intensity(0)
            } else {
                0.0
            };
        }

        if self.is_descending {
            if raw_q > self.ppm[0] || raw_q < self.ppm[n - 1] {
                return 0.0;
            }
            let idx = self.ppm.partition_point(|&p| p > raw_q);
            if idx == 0 {
                return self.effective_intensity(0);
            }
            if idx >= n {
                return self.effective_intensity(n - 1);
            }
            let p0 = self.ppm[idx - 1];
            let p1 = self.ppm[idx];
            let denom = p1 - p0;
            if denom.abs() <= 1e-18 {
                return self.effective_intensity(idx);
            }
            let t = ((raw_q - p0) / denom).clamp(0.0, 1.0);
            let y0 = self.effective_intensity(idx - 1);
            let y1 = self.effective_intensity(idx);
            y0 + t * (y1 - y0)
        } else {
            if raw_q < self.ppm[0] || raw_q > self.ppm[n - 1] {
                return 0.0;
            }
            let idx = self.ppm.partition_point(|&p| p < raw_q);
            if idx == 0 {
                return self.effective_intensity(0);
            }
            if idx >= n {
                return self.effective_intensity(n - 1);
            }
            let p0 = self.ppm[idx - 1];
            let p1 = self.ppm[idx];
            let denom = p1 - p0;
            if denom.abs() <= 1e-18 {
                return self.effective_intensity(idx);
            }
            let t = ((raw_q - p0) / denom).clamp(0.0, 1.0);
            let y0 = self.effective_intensity(idx - 1);
            let y1 = self.effective_intensity(idx);
            y0 + t * (y1 - y0)
        }
    }

    /// Extract a peak-preserving `(effective_ppm, effective_intensity)` polyline within
    /// `[view_ppm_min, view_ppm_max]` bounded to at most `~2 * target_columns` vertices using the
    /// precomputed dyadic [`LodPyramid1d`].
    pub fn decimate_for_viewport(
        &self,
        view_ppm_min: f64,
        view_ppm_max: f64,
        target_columns: usize,
    ) -> Vec<(f64, f64)> {
        let Some((i0, i1)) = self.index_range_for_ppm(view_ppm_min, view_ppm_max) else {
            return Vec::new();
        };
        let visible_len = i1 - i0 + 1;
        let max_cols = target_columns.max(32);

        // Exact full-resolution path when zoomed in
        if visible_len <= 2 * max_cols || self.lod.levels.is_empty() {
            let mut out = Vec::with_capacity(visible_len);
            for idx in i0..=i1 {
                out.push((self.effective_ppm(idx), self.effective_intensity(idx)));
            }
            return out;
        }

        // Choose highest dyadic pyramid level whose bucket count over [i0..=i1] stays >= max_cols
        let mut chosen_lvl = 0usize;
        for l in 0..self.lod.levels.len() {
            let stride = 1usize << (l + 1);
            if visible_len / stride >= max_cols {
                chosen_lvl = l;
            } else {
                break;
            }
        }

        let stride = 1usize << (chosen_lvl + 1);
        let buckets = &self.lod.levels[chosen_lvl];
        let b0 = i0 / stride;
        let b1 = (i1 / stride).min(buckets.len().saturating_sub(1));
        let mut out = Vec::with_capacity((b1.saturating_sub(b0) + 1) * 2 + 2);
        let scale = self.scale;

        for &bk in &buckets[b0..=b1] {
            let idx_min = bk.min_idx as usize;
            let idx_max = bk.max_idx as usize;
            let (first_i, first_v, second_i, second_v) = if scale >= 0.0 {
                if idx_min <= idx_max {
                    (idx_min, bk.min_val * scale, idx_max, bk.max_val * scale)
                } else {
                    (idx_max, bk.max_val * scale, idx_min, bk.min_val * scale)
                }
            } else if idx_min <= idx_max {
                (idx_min, bk.min_val * scale, idx_max, bk.max_val * scale)
            } else {
                (idx_max, bk.max_val * scale, idx_min, bk.min_val * scale)
            };

            if first_i >= i0 && first_i <= i1 {
                out.push((self.effective_ppm(first_i), first_v));
            }
            if second_i != first_i && second_i >= i0 && second_i <= i1 {
                out.push((self.effective_ppm(second_i), second_v));
            }
        }
        out
    }
}

// ============================================================================
// Pure `f64` Quantitative Algorithms: Integration, Normalization, OLS Difference & FFT Interval Alignment
// ============================================================================

/// Integrate a `Trace1d` over `[ppm_min, ppm_max]` using exact continuous piecewise-linear
/// trapezoidal quadrature (including exact linear interpolation at both `[ppm_min, ppm_max]`
/// boundaries so grid-misaligned or shifted traces retain `< 1e-12` analytical accuracy).
pub fn integrate_trace_region_f64(
    trace: &Trace1d,
    ppm_a: f64,
    ppm_b: f64,
    subtract_edge_baseline: bool,
) -> f64 {
    let lo_req = ppm_a.min(ppm_b);
    let hi_req = ppm_a.max(ppm_b);
    if !lo_req.is_finite() || !hi_req.is_finite() || hi_req <= lo_req || trace.len() < 2 {
        return 0.0;
    }
    let Some((trace_min, trace_max)) = trace.ppm_bounds() else {
        return 0.0;
    };
    let lo = lo_req.max(trace_min);
    let hi = hi_req.min(trace_max);
    if hi <= lo {
        return 0.0;
    }

    let y_lo = trace.sample_at_ppm(lo);
    let y_hi = trace.sample_at_ppm(hi);

    let Some((i0, i1)) = trace.index_range_for_ppm(lo, hi) else {
        return 0.0;
    };

    let mut area = 0.0_f64;
    let mut prev_p = lo;
    let mut prev_y = y_lo;

    if trace.is_descending {
        // Iterate from i1 down to i0 so effective_ppm increases monotonically from lo -> hi
        for idx in (i0..=i1).rev() {
            let p = trace.effective_ppm(idx);
            if p > lo && p < hi {
                let y = trace.effective_intensity(idx);
                area += 0.5 * (prev_y + y) * (p - prev_p);
                prev_p = p;
                prev_y = y;
            }
        }
    } else {
        for idx in i0..=i1 {
            let p = trace.effective_ppm(idx);
            if p > lo && p < hi {
                let y = trace.effective_intensity(idx);
                area += 0.5 * (prev_y + y) * (p - prev_p);
                prev_p = p;
                prev_y = y;
            }
        }
    }

    if hi > prev_p {
        area += 0.5 * (prev_y + y_hi) * (hi - prev_p);
    }

    if subtract_edge_baseline {
        area -= 0.5 * (y_lo + y_hi) * (hi - lo);
    }
    area
}

/// Integrate $\int_{\text{ppm\_min}}^{\text{ppm\_max}} |S_i(\delta)|\,d\delta$ across `trace`.
pub fn integrate_trace_abs_region_f64(trace: &Trace1d, ppm_a: f64, ppm_b: f64) -> f64 {
    let lo_req = ppm_a.min(ppm_b);
    let hi_req = ppm_a.max(ppm_b);
    if !lo_req.is_finite() || !hi_req.is_finite() || hi_req <= lo_req || trace.len() < 2 {
        return 0.0;
    }
    let Some((trace_min, trace_max)) = trace.ppm_bounds() else {
        return 0.0;
    };
    let lo = lo_req.max(trace_min);
    let hi = hi_req.min(trace_max);
    if hi <= lo {
        return 0.0;
    }

    let Some((i0, i1)) = trace.index_range_for_ppm(lo, hi) else {
        return 0.0;
    };

    let mut knots: Vec<(f64, f64)> = Vec::with_capacity(i1 - i0 + 3);
    knots.push((lo, trace.sample_at_ppm(lo)));
    if trace.is_descending {
        for idx in (i0..=i1).rev() {
            let p = trace.effective_ppm(idx);
            if p > lo && p < hi {
                knots.push((p, trace.effective_intensity(idx)));
            }
        }
    } else {
        for idx in i0..=i1 {
            let p = trace.effective_ppm(idx);
            if p > lo && p < hi {
                knots.push((p, trace.effective_intensity(idx)));
            }
        }
    }
    knots.push((hi, trace.sample_at_ppm(hi)));

    let mut area = 0.0_f64;
    for w in knots.windows(2) {
        let (p0, y0) = w[0];
        let (p1, y1) = w[1];
        let dp = p1 - p0;
        if dp <= 0.0 {
            continue;
        }
        if (y0 >= 0.0 && y1 >= 0.0) || (y0 <= 0.0 && y1 <= 0.0) {
            area += 0.5 * (y0.abs() + y1.abs()) * dp;
        } else {
            // Exact zero-crossing split for piecewise-linear |S(delta)|
            let frac = y0.abs() / (y0.abs() + y1.abs());
            area += 0.5 * y0.abs() * (dp * frac) + 0.5 * y1.abs() * (dp * (1.0 - frac));
        }
    }
    area
}

/// Normalize unlocked traces in-place according to `mode` (Craig et al., *Anal. Chem.* 2006, 78,
/// 2262–2267). Locked traces (`trace.locked == true`) are strictly preserved untouched.
pub fn normalize_traces(traces: &mut [Trace1d], mode: NormalizationMode) {
    for trace in traces.iter_mut() {
        if trace.locked || trace.is_empty() {
            continue;
        }
        match mode {
            NormalizationMode::None => {
                trace.scale = 1.0;
                trace.generation = trace.generation.wrapping_add(1);
            }
            NormalizationMode::TotalArea => {
                // Evaluate unscaled integral by temporarily reading scale = 1.0
                let saved_scale = trace.scale;
                trace.scale = 1.0;
                let (lo, hi) = trace.ppm_bounds().unwrap_or((0.0, 1.0));
                let signed = integrate_trace_region_f64(trace, lo, hi, false);
                let norm = if signed.abs() > 1e-15 {
                    signed.abs()
                } else {
                    integrate_trace_abs_region_f64(trace, lo, hi)
                };
                trace.scale = if norm > 1e-30 {
                    if signed < -1e-15 {
                        -1.0 / norm
                    } else {
                        1.0 / norm
                    }
                } else {
                    saved_scale
                };
                trace.generation = trace.generation.wrapping_add(1);
            }
            NormalizationMode::TotalAbsArea => {
                let saved_scale = trace.scale;
                trace.scale = 1.0;
                let (lo, hi) = trace.ppm_bounds().unwrap_or((0.0, 1.0));
                let norm = integrate_trace_abs_region_f64(trace, lo, hi);
                trace.scale = if norm > 1e-30 {
                    1.0 / norm
                } else {
                    saved_scale
                };
                trace.generation = trace.generation.wrapping_add(1);
            }
            NormalizationMode::ReferencePeakHeight { ppm_min, ppm_max } => {
                let saved_scale = trace.scale;
                trace.scale = 1.0;
                let lo = ppm_min.min(ppm_max);
                let hi = ppm_min.max(ppm_max);
                let mut peak_abs = 0.0_f64;
                if let Some((i0, i1)) = trace.index_range_for_ppm(lo, hi) {
                    for idx in i0..=i1 {
                        let p = trace.effective_ppm(idx);
                        if p >= lo && p <= hi {
                            peak_abs = peak_abs.max(trace.intensities[idx].abs());
                        }
                    }
                }
                peak_abs = peak_abs
                    .max(trace.sample_at_ppm(lo).abs())
                    .max(trace.sample_at_ppm(hi).abs());
                if peak_abs > 1e-30 {
                    trace.scale = 1.0 / peak_abs;
                    trace.generation = trace.generation.wrapping_add(1);
                } else {
                    trace.scale = saved_scale;
                }
            }
            NormalizationMode::ReferenceRegionIntegral { ppm_min, ppm_max } => {
                let saved_scale = trace.scale;
                trace.scale = 1.0;
                let signed = integrate_trace_region_f64(trace, ppm_min, ppm_max, false);
                let norm = if signed.abs() > 1e-15 {
                    signed
                } else {
                    integrate_trace_abs_region_f64(trace, ppm_min, ppm_max)
                };
                trace.scale = if norm.abs() > 1e-30 {
                    1.0 / norm
                } else {
                    saved_scale
                };
                trace.generation = trace.generation.wrapping_add(1);
            }
        }
    }
}

/// Estimate ordinary least-squares (`OLS`) scaling factor $\alpha$ (and optional baseline offset $\beta$)
/// minimizing $\sum_{\delta_k \in [\text{ppm\_min}, \text{ppm\_max}]} (S_{\text{target}}(\delta_k) - (\alpha S_{\text{ref}}(\delta_k) + \beta))^2$.
///
/// Returns `(alpha, beta, residual_rms)`.
pub fn estimate_difference_alpha_ols(
    target: &Trace1d,
    reference: &Trace1d,
    ppm_a: f64,
    ppm_b: f64,
    fit_intercept: bool,
) -> (f64, f64, f64) {
    let lo = ppm_a.min(ppm_b);
    let hi = ppm_a.max(ppm_b);
    let Some((i0, i1)) = target.index_range_for_ppm(lo, hi) else {
        return (1.0, 0.0, 0.0);
    };

    let mut xs = Vec::with_capacity(i1 - i0 + 1);
    let mut ys = Vec::with_capacity(i1 - i0 + 1);
    for idx in i0..=i1 {
        let p = target.effective_ppm(idx);
        if p >= lo && p <= hi {
            let y_tgt = target.effective_intensity(idx);
            let x_ref = if target.ppm.len() == reference.ppm.len()
                && (target.effective_ppm(idx) - reference.effective_ppm(idx)).abs() <= 1e-13
            {
                reference.effective_intensity(idx)
            } else {
                reference.sample_at_ppm(p)
            };
            xs.push(x_ref);
            ys.push(y_tgt);
        }
    }

    let n = xs.len();
    if n == 0 {
        return (1.0, 0.0, 0.0);
    }

    let (alpha, beta) = if fit_intercept && n >= 2 {
        let inv_n = 1.0 / (n as f64);
        let mean_x = xs.iter().sum::<f64>() * inv_n;
        let mean_y = ys.iter().sum::<f64>() * inv_n;
        let mut sxx = 0.0_f64;
        let mut sxy = 0.0_f64;
        for (&x, &y) in xs.iter().zip(&ys) {
            let dx = x - mean_x;
            sxx += dx * dx;
            sxy += dx * (y - mean_y);
        }
        if sxx > 1e-30 {
            let a = sxy / sxx;
            let b = mean_y - a * mean_x;
            (a, b)
        } else {
            (1.0, mean_y - mean_x)
        }
    } else {
        let mut sxx = 0.0_f64;
        let mut sxy = 0.0_f64;
        for (&x, &y) in xs.iter().zip(&ys) {
            sxx += x * x;
            sxy += x * y;
        }
        let a = if sxx > 1e-30 { sxy / sxx } else { 1.0 };
        (a, 0.0)
    };

    let mut rss = 0.0_f64;
    for (&x, &y) in xs.iter().zip(&ys) {
        let r = y - (alpha * x + beta);
        rss += r * r;
    }
    let rms = (rss / (n as f64)).sqrt();
    (alpha, beta, rms)
}

/// Compute the full difference spectrum $D_i(\delta) = S_i(\delta) - (\alpha_i S_{\text{ref}}(\delta) + \beta_i)$
/// on `target`'s chemical-shift grid.
pub fn compute_difference_trace(
    target: &Trace1d,
    reference: &Trace1d,
    alpha_mode: DifferenceAlphaMode,
    manual_alpha: f64,
) -> DifferenceTraceResult {
    let (alpha, beta, residual_rms) = if let Some(override_a) = target.alpha_override {
        (override_a, 0.0, 0.0)
    } else {
        match alpha_mode {
            DifferenceAlphaMode::Manual => (manual_alpha, 0.0, 0.0),
            DifferenceAlphaMode::LeastSquaresRegion {
                ppm_min,
                ppm_max,
                fit_intercept,
            } => estimate_difference_alpha_ols(target, reference, ppm_min, ppm_max, fit_intercept),
        }
    };

    let n = target.len();
    let mut ppm = Vec::with_capacity(n);
    let mut diff_intensities = Vec::with_capacity(n);
    let same_grid = n == reference.len()
        && n > 0
        && (target.effective_ppm(0) - reference.effective_ppm(0)).abs() <= 1e-13
        && (target.effective_ppm(n - 1) - reference.effective_ppm(n - 1)).abs() <= 1e-13;

    for idx in 0..n {
        let p = target.effective_ppm(idx);
        let s_i = target.effective_intensity(idx);
        let s_ref = if same_grid {
            reference.effective_intensity(idx)
        } else {
            reference.sample_at_ppm(p)
        };
        ppm.push(p);
        diff_intensities.push(s_i - (alpha * s_ref + beta));
    }

    DifferenceTraceResult {
        ppm,
        diff_intensities,
        alpha,
        beta,
        residual_rms,
    }
}

/// Iterative in-place radix-2 Cooley–Tukey FFT / IFFT (`buf.len()` must be a power of two).
///
/// Provenance: Cooley & Tukey (*Math. Comput.* **1965**, 19, 297–301).
pub fn fft_radix2_in_place(buf: &mut [(f64, f64)], inverse: bool) {
    let n = buf.len();
    if n <= 1 {
        return;
    }
    debug_assert!(
        n.is_power_of_two(),
        "FFT buffer length must be a power of two"
    );

    // Bit-reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            buf.swap(i, j);
        }
    }

    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2usize;
    while len <= n {
        let half = len >> 1;
        let theta = sign * 2.0 * PI / (len as f64);
        let w_step = (theta.cos(), theta.sin());
        let mut start = 0usize;
        while start < n {
            let mut w = (1.0_f64, 0.0_f64);
            for k in 0..half {
                let u = buf[start + k];
                let v = buf[start + k + half];
                let tv = (v.0 * w.0 - v.1 * w.1, v.0 * w.1 + v.1 * w.0);
                buf[start + k] = (u.0 + tv.0, u.1 + tv.1);
                buf[start + k + half] = (u.0 - tv.0, u.1 - tv.1);
                w = (
                    w.0 * w_step.0 - w.1 * w_step.1,
                    w.0 * w_step.1 + w.1 * w_step.0,
                );
            }
            start += len;
        }
        len <<= 1;
    }

    if inverse {
        let inv_n = 1.0 / (n as f64);
        for c in buf.iter_mut() {
            c.0 *= inv_n;
            c.1 *= inv_n;
        }
    }
}

/// Estimate sub-sample lag $\hat{\tau}$ (in samples, bounded by `[-max_shift_pts, +max_shift_pts]`)
/// maximizing the zero-padded FFT cross-correlation $C[m] = \sum_p \tilde{r}[p]\,\tilde{s}[p + m]$
/// such that resampling $s_{\text{aligned}}[p] = s(p + \hat{\tau})$ aligns `target_seg` onto `ref_seg`.
///
/// Provenance: Savorani, Tomasi & Engelsen (*J. Magn. Reson.* **2010**, 202, 190–202, §2.1).
/// Returns `(lag_points, normalized_correlation)`.
pub fn estimate_fft_shift_points(
    ref_seg: &[f64],
    target_seg: &[f64],
    max_shift_pts: usize,
) -> (f64, f64) {
    let l = ref_seg.len().min(target_seg.len());
    if l < 3 {
        return (0.0, 0.0);
    }
    let capped_shift = max_shift_pts.min(l.saturating_sub(2));
    if capped_shift == 0 {
        return (0.0, 1.0);
    }

    let inv_l = 1.0 / (l as f64);
    let mean_r = ref_seg[..l].iter().sum::<f64>() * inv_l;
    let mean_s = target_seg[..l].iter().sum::<f64>() * inv_l;

    let mut norm_r2 = 0.0_f64;
    let mut norm_s2 = 0.0_f64;
    for k in 0..l {
        let dr = ref_seg[k] - mean_r;
        let ds = target_seg[k] - mean_s;
        norm_r2 += dr * dr;
        norm_s2 += ds * ds;
    }
    let denom = (norm_r2 * norm_s2).sqrt();
    if denom <= 1e-30 {
        return (0.0, 0.0);
    }

    // Zero-pad to >= 2 * L power-of-two to eliminate circular wrap-around aliasing
    let m_fft = (2 * l).next_power_of_two().max(8);
    let mut f_r = vec![(0.0_f64, 0.0_f64); m_fft];
    let mut f_s = vec![(0.0_f64, 0.0_f64); m_fft];
    for k in 0..l {
        f_r[k].0 = ref_seg[k] - mean_r;
        f_s[k].0 = target_seg[k] - mean_s;
    }

    fft_radix2_in_place(&mut f_r, false);
    fft_radix2_in_place(&mut f_s, false);

    // Cross-spectrum: conj(F_r[k]) * F_s[k] -> IFFT[m] = sum_p r[p] * s[p + m]
    for k in 0..m_fft {
        let r = f_r[k];
        let s = f_s[k];
        f_r[k] = (r.0 * s.0 + r.1 * s.1, r.0 * s.1 - r.1 * s.0);
    }
    fft_radix2_in_place(&mut f_r, true);

    let corr_at = |lag: isize| -> f64 {
        let idx = if lag >= 0 {
            (lag as usize) % m_fft
        } else {
            (m_fft as isize + lag).rem_euclid(m_fft as isize) as usize
        };
        f_r[idx].0
    };

    let bound = capped_shift as isize;
    let mut best_lag = 0isize;
    let mut best_val = corr_at(0);
    for lag in -bound..=bound {
        let v = corr_at(lag);
        if v > best_val {
            best_val = v;
            best_lag = lag;
        }
    }

    // Sub-sample 3-point parabolic peak refinement (when strictly interior to search window)
    let sub_delta = if best_lag > -bound && best_lag < bound {
        let c_m1 = corr_at(best_lag - 1);
        let c_0 = best_val;
        let c_p1 = corr_at(best_lag + 1);
        let curv = c_m1 - 2.0 * c_0 + c_p1;
        if curv < -1e-18 {
            ((c_m1 - c_p1) / (2.0 * curv)).clamp(-0.5, 0.5)
        } else {
            0.0
        }
    } else {
        0.0
    };

    let refined_lag = (best_lag as f64) + sub_delta;

    // Evaluate true overlap-window Pearson correlation at `refined_lag` (free of zero-pad edge bias)
    let mut ov_pairs: Vec<(f64, f64)> = Vec::with_capacity(l);
    let max_idx = (l - 1) as f64;
    for (p, &r_val) in ref_seg.iter().take(l).enumerate() {
        let q = (p as f64) + refined_lag;
        if q >= 0.0 && q <= max_idx {
            ov_pairs.push((r_val, cubic_catmull_rom_sample(&target_seg[..l], q)));
        }
    }
    let norm_corr = if ov_pairs.len() >= 3 {
        let inv_m = 1.0 / (ov_pairs.len() as f64);
        let mr = ov_pairs.iter().map(|&(r, _)| r).sum::<f64>() * inv_m;
        let ms = ov_pairs.iter().map(|&(_, s)| s).sum::<f64>() * inv_m;
        let (mut sr2, mut ss2, mut srs) = (0.0_f64, 0.0_f64, 0.0_f64);
        for &(r, s) in &ov_pairs {
            let dr = r - mr;
            let ds = s - ms;
            sr2 += dr * dr;
            ss2 += ds * ds;
            srs += dr * ds;
        }
        let d = (sr2 * ss2).sqrt();
        if d > 1e-30 {
            (srs / d).clamp(-1.0, 1.0)
        } else {
            (best_val / denom).clamp(-1.0, 1.0)
        }
    } else {
        (best_val / denom).clamp(-1.0, 1.0)
    };
    (refined_lag, norm_corr)
}

/// 4-point Catmull–Rom cubic convolution interpolator (`Keys` 1981) with endpoint clamping
/// (interval boundary value fill, Savorani et al. 2010 §2.2).
#[inline]
pub fn cubic_catmull_rom_sample(seg: &[f64], pos: f64) -> f64 {
    let n = seg.len();
    if n == 0 {
        return 0.0;
    }
    if pos <= 0.0 {
        return seg[0];
    }
    let last = (n - 1) as f64;
    if pos >= last {
        return seg[n - 1];
    }

    let i1 = pos.floor() as isize;
    let t = pos - (i1 as f64);
    let get = |idx: isize| -> f64 { seg[idx.clamp(0, (n - 1) as isize) as usize] };

    let y0 = get(i1 - 1);
    let y1 = get(i1);
    let y2 = get(i1 + 1);
    let y3 = get(i1 + 2);

    let a0 = -0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3;
    let a1 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3;
    let a2 = -0.5 * y0 + 0.5 * y2;
    let a3 = y1;
    ((a0 * t + a1) * t + a2) * t + a3
}

/// Resample `data[start..=end_inclusive]` in-place by fractional lag `lag_points`
/// (`aligned[k] = raw(k + lag_points)`).
pub fn shift_interval_in_place(
    data: &mut [f64],
    start: usize,
    end_inclusive: usize,
    lag_points: f64,
) {
    if start >= end_inclusive || end_inclusive >= data.len() || lag_points.abs() <= 1e-12 {
        return;
    }
    let orig: Vec<f64> = data[start..=end_inclusive].to_vec();
    for (k, slot) in data[start..=end_inclusive].iter_mut().enumerate() {
        *slot = cubic_catmull_rom_sample(&orig, (k as f64) + lag_points);
    }
}

/// Align all unlocked traces in `traces` (`i != ref_idx`) against `traces[ref_idx]` across
/// `intervals` via interval correlation-shifting FFT cross-correlation (Savorani et al.,
/// *J. Magn. Reson.* **2010**, 202, 190–202).
pub fn align_traces_interval_fft(
    traces: &mut [Trace1d],
    ref_idx: usize,
    intervals: &[AlignmentInterval],
    scope: AlignmentScope,
) -> Vec<IntervalShiftRecord> {
    if ref_idx >= traces.len() || intervals.is_empty() {
        return Vec::new();
    }
    let ref_snapshot = traces[ref_idx].clone();
    let ref_dt_ppm = ref_snapshot.mean_step_ppm();
    let mut records = Vec::new();

    for (idx, trace) in traces.iter_mut().enumerate() {
        if idx == ref_idx || trace.locked || trace.len() < 4 {
            continue;
        }
        match scope {
            AlignmentScope::GlobalShiftFromRegion => {
                let iv = intervals[0];
                let lo = iv.ppm_min.min(iv.ppm_max);
                let hi = iv.ppm_min.max(iv.ppm_max);
                if hi <= lo {
                    continue;
                }
                let dt = ref_dt_ppm.min(trace.mean_step_ppm()).max(1e-6);
                let n_pts = (((hi - lo) / dt).round() as usize + 1).clamp(8, 131_072);
                let step = (hi - lo) / ((n_pts - 1) as f64);

                let mut ref_seg = Vec::with_capacity(n_pts);
                let mut tgt_seg = Vec::with_capacity(n_pts);
                for k in 0..n_pts {
                    let p = lo + (k as f64) * step;
                    ref_seg.push(ref_snapshot.sample_at_ppm(p));
                    tgt_seg.push(trace.sample_at_ppm(p));
                }
                let max_pts = ((iv.max_shift_ppm / step).round() as usize).max(1);
                let (lag_pts, corr) = estimate_fft_shift_points(&ref_seg, &tgt_seg, max_pts);
                let delta_ppm = -lag_pts * step;
                trace.shift_offset_ppm += delta_ppm;
                trace.generation = trace.generation.wrapping_add(1);
                records.push(IntervalShiftRecord {
                    trace_id: trace.id.clone(),
                    ppm_min: lo,
                    ppm_max: hi,
                    shift_points: lag_pts,
                    shift_ppm: delta_ppm,
                    correlation: corr,
                });
            }
            AlignmentScope::LocalIntervals => {
                let mut modified = false;
                let mut updated = trace.intensities.clone();
                let signed_step = if trace.len() >= 2 {
                    trace.ppm[1] - trace.ppm[0]
                } else {
                    -ref_dt_ppm
                };

                for iv in intervals {
                    let lo = iv.ppm_min.min(iv.ppm_max);
                    let hi = iv.ppm_min.max(iv.ppm_max);
                    let Some((i0, i1)) = trace.index_range_for_ppm(lo, hi) else {
                        continue;
                    };
                    let seg_len = i1 - i0 + 1;
                    if seg_len < 6 {
                        continue;
                    }
                    let mut ref_seg = Vec::with_capacity(seg_len);
                    let mut tgt_seg = Vec::with_capacity(seg_len);
                    for (k, &upd_val) in updated.iter().enumerate().take(i1 + 1).skip(i0) {
                        ref_seg.push(ref_snapshot.sample_at_ppm(trace.effective_ppm(k)));
                        tgt_seg.push(upd_val * trace.scale);
                    }
                    let max_pts =
                        ((iv.max_shift_ppm / signed_step.abs().max(1e-9)).round() as usize).max(1);
                    let (lag_pts, corr) = estimate_fft_shift_points(&ref_seg, &tgt_seg, max_pts);
                    shift_interval_in_place(&mut updated, i0, i1, lag_pts);
                    modified = true;
                    records.push(IntervalShiftRecord {
                        trace_id: trace.id.clone(),
                        ppm_min: lo,
                        ppm_max: hi,
                        shift_points: lag_pts,
                        shift_ppm: -lag_pts * signed_step,
                        correlation: corr,
                    });
                }

                if modified {
                    trace.set_working_intensities(updated);
                }
            }
        }
    }

    records
}

/// Evaluate all `regions` across `traces` to produce a quantitative kinetics / titration /
/// reaction-progress [`RoiMatrix`].
pub fn compute_roi_matrix(
    traces: &[Trace1d],
    regions: &[RoiRegion1d],
    ref_trace_idx: usize,
    formula: RoiRatioFormula,
) -> RoiMatrix {
    let k = regions.len();
    let mut raw_areas: Vec<Vec<f64>> = Vec::with_capacity(traces.len());
    for trace in traces {
        let mut row_areas = Vec::with_capacity(k);
        for reg in regions {
            let (lo, hi) = reg.ordered_bounds();
            row_areas.push(integrate_trace_region_f64(
                trace,
                lo,
                hi,
                reg.edge_baseline_correction,
            ));
        }
        raw_areas.push(row_areas);
    }

    let safe_ref = ref_trace_idx.min(traces.len().saturating_sub(1));
    let mut rows = Vec::with_capacity(traces.len());

    for (i, trace) in traces.iter().enumerate() {
        let areas = &raw_areas[i];
        let normalized_ratio = if areas.is_empty() {
            0.0
        } else {
            match formula {
                RoiRatioFormula::AutoReactionOrDecay => {
                    if k >= 2 {
                        let sum: f64 = areas.iter().sum();
                        if sum.abs() > 1e-30 {
                            areas[0] / sum
                        } else {
                            0.0
                        }
                    } else {
                        let denom = raw_areas
                            .get(safe_ref)
                            .and_then(|r| r.first())
                            .copied()
                            .unwrap_or(0.0);
                        if denom.abs() > 1e-30 {
                            areas[0] / denom
                        } else {
                            0.0
                        }
                    }
                }
                RoiRatioFormula::FirstOverSum => {
                    let sum: f64 = areas.iter().sum();
                    if sum.abs() > 1e-30 {
                        areas[0] / sum
                    } else {
                        0.0
                    }
                }
                RoiRatioFormula::TargetOverRefRegion {
                    target_idx,
                    ref_idx,
                } => {
                    let num = areas.get(target_idx).copied().unwrap_or(0.0);
                    let den = areas.get(ref_idx).copied().unwrap_or(0.0);
                    if den.abs() > 1e-30 {
                        num / den
                    } else {
                        0.0
                    }
                }
                RoiRatioFormula::RelativeToRefTrace { region_idx } => {
                    let num = areas.get(region_idx).copied().unwrap_or(0.0);
                    let den = raw_areas
                        .get(safe_ref)
                        .and_then(|r| r.get(region_idx))
                        .copied()
                        .unwrap_or(0.0);
                    if den.abs() > 1e-30 {
                        num / den
                    } else {
                        0.0
                    }
                }
            }
        };

        rows.push(RoiMatrixRow {
            trace_id: trace.id.clone(),
            covariate: trace.covariate,
            areas: areas.clone(),
            normalized_ratio,
        });
    }

    RoiMatrix {
        regions: regions.to_vec(),
        rows,
    }
}

/// Diverging dark-workbench colormap mapping signed normalized intensity `v in [-1.0, 1.0]` to RGBA.
#[inline]
fn diverging_heatmap_rgba(v: f64) -> [u8; 4] {
    let clamped = v.clamp(-1.0, 1.0);
    // Base dark canvas: (14, 18, 26)
    if clamped >= 0.0 {
        let t = clamped.sqrt();
        if t < 0.55 {
            let u = t / 0.55;
            [
                (14.0 + u * (30.0 - 14.0)).round() as u8,
                (18.0 + u * (175.0 - 18.0)).round() as u8,
                (26.0 + u * (238.0 - 26.0)).round() as u8,
                255,
            ]
        } else {
            let u = (t - 0.55) / 0.45;
            [
                (30.0 + u * (252.0 - 30.0)).round() as u8,
                (175.0 + u * (246.0 - 175.0)).round() as u8,
                (238.0 + u * (198.0 - 238.0)).round() as u8,
                255,
            ]
        }
    } else {
        let t = (-clamped).sqrt();
        [
            (14.0 + t * (244.0 - 14.0)).round() as u8,
            (18.0 + t * (68.0 - 18.0)).round() as u8,
            (26.0 + t * (102.0 - 26.0)).round() as u8,
            255,
        ]
    }
}

/// Synthesize a `width × height` unmultiplied RGBA buffer (`4 * width * height` bytes) for
/// [`Stack1dViewMode::Pseudo2dHeatmap`] across `[view_ppm_max -> view_ppm_min]` (left-to-right
/// inverted NMR axis) and visible traces along `y`.
#[allow(clippy::too_many_arguments)]
pub fn synthesize_pseudo2d_rgba(
    traces: &[Trace1d],
    view_ppm_min: f64,
    view_ppm_max: f64,
    width: usize,
    height: usize,
    contrast_floor_frac: f64,
    contrast_ceil_frac: f64,
    sort_by_covariate: bool,
) -> Vec<u8> {
    let w = width.max(1);
    let h = height.max(1);
    let mut rgba = vec![0u8; w * h * 4];

    let mut ordered_indices: Vec<usize> = traces
        .iter()
        .enumerate()
        .filter_map(|(i, t)| t.visible.then_some(i))
        .collect();

    if ordered_indices.is_empty() {
        for px in rgba.chunks_exact_mut(4) {
            px.copy_from_slice(&[14, 18, 26, 255]);
        }
        return rgba;
    }

    if sort_by_covariate {
        ordered_indices.sort_by(|&a, &b| traces[a].covariate.total_cmp(&traces[b].covariate));
    }

    let global_max = ordered_indices
        .iter()
        .fold(0.0_f64, |acc, &i| acc.max(traces[i].max_abs_effective()))
        .max(1e-18);

    let floor = contrast_floor_frac.clamp(0.0, 0.90) * global_max;
    let ceil = (contrast_ceil_frac.clamp(0.02, 1.0) * global_max).max(floor + 1e-18);
    let inv_span = 1.0 / (ceil - floor);

    let lo = view_ppm_min.min(view_ppm_max);
    let hi = view_ppm_min.max(view_ppm_max);
    let ppm_span = (hi - lo).max(1e-9);
    let n_vis = ordered_indices.len();

    // Precompute column chemical-shift bounds (inverted NMR axis: col 0 = hi ppm, col w-1 = lo ppm)
    let col_ppm_edges: Vec<(f64, f64)> = (0..w)
        .map(|cx| {
            let p_left = hi - (cx as f64 / w as f64) * ppm_span;
            let p_right = hi - ((cx + 1) as f64 / w as f64) * ppm_span;
            (p_right.min(p_left), p_right.max(p_left))
        })
        .collect();

    // Precompute one `[w]` intensity row per visible trace using peak-preserving max-abs bin pooling
    let mut trace_rows: Vec<Vec<f64>> = Vec::with_capacity(n_vis);
    for &t_idx in &ordered_indices {
        let tr = &traces[t_idx];
        let mut row = vec![0.0_f64; w];
        for (cx, &(bin_lo, bin_hi)) in col_ppm_edges.iter().enumerate() {
            let mid = 0.5 * (bin_lo + bin_hi);
            let mut best = tr.sample_at_ppm(mid);
            if let Some((i0, i1)) = tr.index_range_for_ppm(bin_lo, bin_hi) {
                if i1 - i0 <= 64 {
                    for idx in i0..=i1 {
                        let p = tr.effective_ppm(idx);
                        if p >= bin_lo && p <= bin_hi {
                            let val = tr.effective_intensity(idx);
                            if val.abs() > best.abs() {
                                best = val;
                            }
                        }
                    }
                } else {
                    // Use decimated extrema when zooming out on huge traces
                    let pts = tr.decimate_for_viewport(bin_lo, bin_hi, 2);
                    for &(_, val) in &pts {
                        if val.abs() > best.abs() {
                            best = val;
                        }
                    }
                }
            }
            row[cx] = best;
        }
        trace_rows.push(row);
    }

    for ry in 0..h {
        let pos_f = if h <= 1 || n_vis <= 1 {
            0.0
        } else {
            (ry as f64 / (h - 1) as f64) * ((n_vis - 1) as f64)
        };
        let r0 = (pos_f.floor() as usize).min(n_vis - 1);
        let r1 = (r0 + 1).min(n_vis - 1);
        let frac = pos_f - (r0 as f64);

        let row_offset = ry * w * 4;
        let row0 = &trace_rows[r0];
        let row1 = &trace_rows[r1];
        for (cx, (&v0, &v1)) in row0.iter().zip(row1).enumerate() {
            let val = v0 * (1.0 - frac) + v1 * frac;
            let mag = val.abs();
            let norm_signed = if mag <= floor {
                0.0
            } else {
                val.signum() * ((mag - floor) * inv_span).min(1.0)
            };
            let color = diverging_heatmap_rgba(norm_signed);
            let off = row_offset + cx * 4;
            rgba[off..off + 4].copy_from_slice(&color);
        }
    }

    rgba
}

// ============================================================================
// Synthetic Analytical 1D Demo Builders (Kinetics & STD-NMR Titration)
// ============================================================================

fn add_lorentzian(y: &mut [f64], ppm: &[f64], center: f64, fwhm_ppm: f64, area: f64) {
    let gamma = (0.5 * fwhm_ppm).max(1e-5);
    let g2 = gamma * gamma;
    let inv_pi = 1.0 / PI;
    for (slot, &p) in y.iter_mut().zip(ppm) {
        let dp = p - center;
        *slot += area * inv_pi * gamma / (dp * dp + g2);
    }
}

/// Build a deterministic 8-timepoint first-order reaction kinetics 1D `1H` array (`A -> B`,
/// `k = 0.045 min^-1`) with an invariant internal standard (`1.20 ppm`) and slight realistic
/// temperature/shim pH peak drift so users can test Overlay, Tilted Stack, regional FFT alignment, and ROI
/// Kinetics TSV export in one click.
pub fn build_kinetics_demo_traces() -> (Vec<Trace1d>, Vec<RoiRegion1d>) {
    let n_pts = 4096usize;
    let hi_ppm = 8.60_f64;
    let lo_ppm = 0.40_f64;
    let step = (lo_ppm - hi_ppm) / ((n_pts - 1) as f64);
    let ppm: Vec<f64> = (0..n_pts).map(|i| hi_ppm + (i as f64) * step).collect();

    let times_min = [0.0, 5.0, 12.0, 20.0, 32.0, 48.0, 70.0, 100.0];
    let k_rate = 0.042_f64;
    let mut traces = Vec::with_capacity(times_min.len());

    for (idx, &t_min) in times_min.iter().enumerate() {
        let frac_a = (-k_rate * t_min).exp();
        let frac_b = 1.0 - frac_a;
        let slight_drift = 0.0035 * ((idx as f64) - 3.0) * 0.35;

        let mut y = vec![0.0_f64; n_pts];
        // Reactant A: aromatic doublet near 7.65 ppm + quartet CH2 near 4.18 ppm
        add_lorentzian(&mut y, &ppm, 7.665 + slight_drift, 0.008, 1.0 * frac_a);
        add_lorentzian(&mut y, &ppm, 7.635 + slight_drift, 0.008, 1.0 * frac_a);
        for (m, w) in [(-1.5, 1.0), (-0.5, 3.0), (0.5, 3.0), (1.5, 1.0)] {
            add_lorentzian(
                &mut y,
                &ppm,
                4.18 + m * 0.012 + slight_drift * 0.5,
                0.007,
                (w / 4.0) * frac_a,
            );
        }

        // Product B: shifted aromatic doublet near 7.22 ppm + singlet near 3.68 ppm
        add_lorentzian(&mut y, &ppm, 7.235 + slight_drift, 0.008, 1.0 * frac_b);
        add_lorentzian(&mut y, &ppm, 7.205 + slight_drift, 0.008, 1.0 * frac_b);
        add_lorentzian(&mut y, &ppm, 3.68 + slight_drift * 0.4, 0.007, 2.0 * frac_b);

        // Internal reference standard at 1.20 ppm (constant 3H singlet)
        add_lorentzian(&mut y, &ppm, 1.20, 0.007, 1.5);

        let tr = Trace1d::new(
            format!("t_{idx:02}_{t_min:.0}m"),
            format!("t = {t_min:.0} min"),
            t_min,
            ppm.clone(),
            y,
        )
        .with_color(default_trace_color(idx))
        .with_demo(true);
        traces.push(tr);
    }

    let rois = vec![
        RoiRegion1d::new("Product ArH", 7.12, 7.32),
        RoiRegion1d::new("Reactant ArH", 7.55, 7.76),
    ];
    (traces, rois)
}

/// Build a deterministic `STD-NMR` / batch-impurity difference demo (`S_off` reference + `S_on`
/// saturation series where a non-binding co-solute at `1.20 ppm` cancels under OLS $\alpha$ while
/// ligand epitopes at `7.42 ppm` and `3.85 ppm` emerge cleanly in `DifferenceSpectrum` mode).
pub fn build_std_difference_demo_traces() -> (Vec<Trace1d>, Vec<RoiRegion1d>) {
    let n_pts = 4096usize;
    let hi_ppm = 8.40_f64;
    let lo_ppm = 0.50_f64;
    let step = (lo_ppm - hi_ppm) / ((n_pts - 1) as f64);
    let ppm: Vec<f64> = (0..n_pts).map(|i| hi_ppm + (i as f64) * step).collect();

    let sat_times_s: [f64; 5] = [0.0, 0.5, 1.0, 2.0, 4.0];
    let mut traces = Vec::with_capacity(sat_times_s.len());

    for (idx, &t_sat) in sat_times_s.iter().enumerate() {
        // Receiver gain / scan-count scaling perturbation recovered by OLS alpha on [1.05, 1.35] ppm
        let scan_scale = 1.0 - 0.04 * (idx as f64);
        let std_ar = 0.28 * (1.0 - (-0.85 * t_sat).exp());
        let std_aliph = 0.14 * (1.0 - (-0.70 * t_sat).exp());

        let mut y = vec![0.0_f64; n_pts];
        // Binding ligand aromatic epitope (attenuated in S_on, positive in S_on - alpha*S_off or S_off - S_on)
        add_lorentzian(&mut y, &ppm, 7.43, 0.009, scan_scale * (1.0 - std_ar));
        add_lorentzian(&mut y, &ppm, 7.40, 0.009, scan_scale * (1.0 - std_ar));
        // Binding ligand aliphatic epitope
        add_lorentzian(&mut y, &ppm, 3.85, 0.008, scan_scale * (1.2 - std_aliph));
        // Non-binding internal matrix peak at 1.20 ppm (0% STD transfer)
        add_lorentzian(&mut y, &ppm, 1.20, 0.008, scan_scale * 2.0);

        let lbl = if idx == 0 {
            "Off-Resonance (S_ref)".to_string()
        } else {
            format!("On-Resonance ({t_sat:.1} s)")
        };
        traces.push(
            Trace1d::new(format!("std_{idx}"), lbl, t_sat, ppm.clone(), y)
                .with_color(default_trace_color(idx))
                .with_demo(true),
        );
    }

    let rois = vec![
        RoiRegion1d::new("Aromatic Epitope", 7.30, 7.54),
        RoiRegion1d::new("Aliphatic Epitope", 3.74, 3.96),
    ];
    (traces, rois)
}

// ============================================================================
// XML-Escaped Publication Vector SVG Exporter
// ============================================================================

#[inline]
fn xml_escape(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len() + 8);
    for ch in raw.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            c if c.is_control() && c != '\t' && c != '\n' => {}
            c => out.push(c),
        }
    }
    out
}

fn nice_nmr_tick_step(span_ppm: f64, target_ticks: usize) -> f64 {
    let raw = (span_ppm.abs() / (target_ticks.max(2) as f64)).max(1e-6);
    let exp = raw.log10().floor();
    let base = 10.0_f64.powf(exp);
    let frac = raw / base;
    let mult = if frac <= 1.5 {
        1.0
    } else if frac <= 3.5 {
        2.0
    } else if frac <= 7.5 {
        5.0
    } else {
        10.0
    };
    mult * base
}

/// Export the active [`Stack1dState`] viewport as a deterministic, XML-escaped publication SVG
/// document (`SuperimposedOverlay`, `StackedOffset`, or `DifferenceSpectrum`) with an inverted
/// NMR chemical-shift ruler.
pub fn export_stack1d_svg(state: &Stack1dState, width: u32, height: u32) -> String {
    let w = width.max(320) as f64;
    let h = height.max(220) as f64;
    let pad_l = 48.0;
    let pad_r = 28.0;
    let pad_t = 26.0;
    let pad_b = 44.0;
    let plot_w = (w - pad_l - pad_r).max(60.0);
    let plot_h = (h - pad_t - pad_b).max(60.0);

    let lo = state.view_ppm_min.min(state.view_ppm_max);
    let hi = state.view_ppm_min.max(state.view_ppm_max);
    let span = (hi - lo).max(1e-6);
    let ppm_to_x = |ppm: f64| -> f64 { pad_l + ((hi - ppm) / span) * plot_w };

    let mut svg = String::with_capacity(16_384);
    let _ = writeln!(
        svg,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\">",
        width.max(320),
        height.max(220),
        width.max(320),
        height.max(220)
    );
    let _ = writeln!(
        svg,
        "  <rect x=\"0\" y=\"0\" width=\"{w:.1}\" height=\"{h:.1}\" fill=\"#0e121a\"/>"
    );
    let _ = writeln!(
        svg,
        "  <defs><clipPath id=\"plotClip\"><rect x=\"{pad_l:.1}\" y=\"{pad_t:.1}\" width=\"{plot_w:.1}\" height=\"{plot_h:.1}\"/></clipPath></defs>"
    );

    // ROI highlight bands
    for roi in &state.roi_regions {
        let (r_lo, r_hi) = roi.ordered_bounds();
        let x0 = ppm_to_x(r_hi).clamp(pad_l, pad_l + plot_w);
        let x1 = ppm_to_x(r_lo).clamp(pad_l, pad_l + plot_w);
        if x1 > x0 {
            let _ = writeln!(
                svg,
                "  <rect x=\"{x0:.2}\" y=\"{pad_t:.1}\" width=\"{:.2}\" height=\"{plot_h:.1}\" fill=\"#38bdf8\" fill-opacity=\"0.09\"/>",
                x1 - x0
            );
            let _ = writeln!(
                svg,
                "  <text x=\"{:.2}\" y=\"{:.1}\" fill=\"#7dd3fc\" font-family=\"monospace\" font-size=\"10\" text-anchor=\"middle\">{}</text>",
                0.5 * (x0 + x1),
                pad_t + 13.0,
                xml_escape(&roi.label)
            );
        }
    }

    let y_base = pad_t + plot_h * 0.84;
    let max_int = state.reference_intensity_ceiling().max(1e-12);
    let y_px_per_unit = (plot_h * 0.70 * state.y_gain) / max_int;

    let _ = writeln!(svg, "  <g clip-path=\"url(#plotClip)\">");
    let vis_indices = state.ordered_visible_trace_indices();
    let ref_idx = state.safe_ref_trace_idx();

    for (stack_ord, &t_idx) in vis_indices.iter().enumerate() {
        let tr = &state.traces[t_idx];
        if state.view_mode == Stack1dViewMode::DifferenceSpectrum
            && state.hide_ref_in_difference
            && t_idx == ref_idx
        {
            continue;
        }

        let (dx_px, dy_px) = if state.view_mode == Stack1dViewMode::StackedOffset {
            let step_v = state.stack_vertical_step_px * (stack_ord as f64);
            let tilt_rad = state.stack_tilt_deg.to_radians();
            (step_v * tilt_rad.tan(), -step_v)
        } else {
            (0.0, 0.0)
        };
        let v_off_px = -tr.vertical_offset * plot_h * 0.5;

        let pts: Vec<(f64, f64)> = if state.view_mode == Stack1dViewMode::DifferenceSpectrum
            && t_idx < state.traces.len()
            && !state.traces.is_empty()
        {
            let diff = compute_difference_trace(
                tr,
                &state.traces[ref_idx],
                state.diff_alpha_mode,
                state.manual_alpha,
            );
            let tmp = Trace1d::new(
                tr.id.clone(),
                tr.label.clone(),
                tr.covariate,
                diff.ppm,
                diff.diff_intensities,
            );
            tmp.decimate_for_viewport(lo, hi, plot_w.ceil() as usize)
        } else {
            tr.decimate_for_viewport(lo, hi, plot_w.ceil() as usize)
        };

        if pts.len() >= 2 {
            let [r, g, b, _] = tr.color_rgba;
            let mut path = String::with_capacity(pts.len() * 16);
            for (k, &(p, val)) in pts.iter().enumerate() {
                let sx = ppm_to_x(p) + dx_px;
                let sy = y_base + dy_px + v_off_px - val * y_px_per_unit;
                if k == 0 {
                    let _ = write!(path, "M{sx:.2},{sy:.2}");
                } else {
                    let _ = write!(path, " L{sx:.2},{sy:.2}");
                }
            }
            let _ = writeln!(
                svg,
                "    <path d=\"{path}\" fill=\"none\" stroke=\"#{r:02x}{g:02x}{b:02x}\" stroke-width=\"1.25\"/>"
            );
        }
    }
    let _ = writeln!(svg, "  </g>");

    // Plot frame + inverted chemical-shift ruler
    let _ = writeln!(
        svg,
        "  <rect x=\"{pad_l:.1}\" y=\"{pad_t:.1}\" width=\"{plot_w:.1}\" height=\"{plot_h:.1}\" fill=\"none\" stroke=\"#334155\" stroke-width=\"1\"/>"
    );
    let tick_step = nice_nmr_tick_step(span, 9);
    let first_tick = (lo / tick_step).ceil() as i64;
    let last_tick = (hi / tick_step).floor() as i64;
    let axis_y = pad_t + plot_h;
    for k in first_tick..=last_tick {
        let p = (k as f64) * tick_step;
        let tx = ppm_to_x(p);
        let _ = writeln!(
            svg,
            "  <line x1=\"{tx:.2}\" y1=\"{axis_y:.1}\" x2=\"{tx:.2}\" y2=\"{:.1}\" stroke=\"#94a3b8\" stroke-width=\"1\"/>",
            axis_y + 5.0
        );
        let _ = writeln!(
            svg,
            "  <text x=\"{tx:.2}\" y=\"{:.1}\" fill=\"#cbd5e1\" font-family=\"monospace\" font-size=\"10\" text-anchor=\"middle\">{p:.2}</text>",
            axis_y + 17.0
        );
    }
    let _ = writeln!(
        svg,
        "  <text x=\"{:.1}\" y=\"{:.1}\" fill=\"#94a3b8\" font-family=\"sans-serif\" font-size=\"11\" text-anchor=\"middle\">Chemical Shift δ (ppm)</text>",
        pad_l + 0.5 * plot_w,
        h - 8.0
    );
    let _ = writeln!(svg, "</svg>");
    svg
}

// ============================================================================
// Interactive `Stack1dState` & `render_stack1d_panel` (`egui` Desktop + `wasm32`)
// ============================================================================

#[derive(Clone, Debug, PartialEq)]
struct HeatmapCacheKey {
    width: usize,
    height: usize,
    view_min_bits: u64,
    view_max_bits: u64,
    floor_bits: u64,
    ceil_bits: u64,
    sort_by_cov: bool,
    trace_sig: u64,
}

/// Self-contained state for the Interactive Multi-Spectrum 1D Comparison, Overlay, Stack,
/// Difference & Kinetics Array Studio (`T14.3`).
#[derive(Clone)]
pub struct Stack1dState {
    pub traces: Vec<Trace1d>,
    pub view_mode: Stack1dViewMode,
    pub covariate_kind: CovariateKind,
    pub ref_trace_idx: usize,

    // Viewport & zoom/gain controls (inverted NMR axis: high ppm left -> low ppm right)
    pub view_ppm_min: f64,
    pub view_ppm_max: f64,
    pub y_gain: f64,
    pub show_integral_curves: bool,
    pub draw_roi_drag_mode: bool,

    // Tilted / vertical stack controls
    pub stack_vertical_step_px: f64,
    pub stack_tilt_deg: f64,
    pub stack_reverse_order: bool,

    // Difference mode (`S_i - alpha * S_ref`) controls
    pub diff_alpha_mode: DifferenceAlphaMode,
    pub manual_alpha: f64,
    pub diff_ref_ppm_min: f64,
    pub diff_ref_ppm_max: f64,
    pub diff_fit_intercept: bool,
    pub show_scaled_ref_overlay: bool,
    pub hide_ref_in_difference: bool,

    // Pseudo-2D heatmap controls & cached GPU texture
    pub heatmap_floor_frac: f64,
    pub heatmap_ceil_frac: f64,
    pub heatmap_sort_by_covariate: bool,
    heatmap_texture: Option<egui::TextureHandle>,
    heatmap_cache_key: Option<HeatmapCacheKey>,

    // Normalization & regional FFT alignment controls
    pub norm_region_ppm_min: f64,
    pub norm_region_ppm_max: f64,
    pub alignment_scope: AlignmentScope,
    pub alignment_intervals: Vec<AlignmentInterval>,
    pub draft_align_ppm_min: f64,
    pub draft_align_ppm_max: f64,
    pub draft_align_max_shift_ppm: f64,
    pub last_alignment_records: Vec<IntervalShiftRecord>,

    // Multi-trace ROI integration & TSV/SVG export state
    pub roi_regions: Vec<RoiRegion1d>,
    pub roi_ratio_formula: RoiRatioFormula,
    pub draft_roi_label: String,
    pub draft_roi_ppm_min: f64,
    pub draft_roi_ppm_max: f64,
    pub active_roi_drag_start_ppm: Option<f64>,
    pub cursor_ppm: Option<f64>,
    pub last_exported_tsv: Option<String>,
    pub last_exported_svg: Option<String>,
    pub show_tsv_preview: bool,
    pub status_message: String,

    // Interactive 1D Phase, Baseline, Multiplet & Journal Report controls
    pub show_1d_analysis_drawer: bool,
    pub show_native_open_bar: bool,
    pub native_open_path_input: String,
    pub pending_open_native_path: Option<String>,
    pub pending_send_to_metabolomics: bool,
    pub phc0_deg: f64,
    pub phc1_deg: f64,
    pub solvent_choice: String,
    pub target_total_protons: f64,
    pub journal_style: PublicationStyle,
    pub detected_multiplets: Vec<MultipletAnalysis>,
    pub detected_multiplet_integrals: Vec<f64>,
    pub formatted_journal_report: String,

    // Spectroscopist 1D Workstation Navigation, Peak Table & Referencing controls
    pub tool_mode: ToolMode1d,
    pub zoom_anchor_ppm: Option<f64>,
    pub active_zoom_drag_start_ppm: Option<f64>,
    active_zoom_drag_start_pos: Option<egui::Pos2>,
    pub zoom_history: Vec<(f64, f64)>,
    pub zoom_redo_stack: Vec<(f64, f64)>,
    pub baseline_y_frac: f32,
    pub selected_inspector_tab: InspectorTab1d,
    pub picked_peaks: Vec<Peak1dEntry>,
    pub selected_peak_idx: Option<usize>,
    pub selected_multiplet_idx: Option<usize>,
    pub picker_min_snr: f64,
    pub estimated_noise_sigma: f64,
    pub show_peak_labels: bool,
    pub peak_labels_in_hz: bool,
    pub show_j_trees: bool,
    pub show_multiplet_boxes: bool,
    pub show_threshold_bar: bool,
    dragging_threshold_bar: bool,
    pub custom_ref_target_ppm: f64,
    pub phase_pivot_mode: PivotMode1d,
    pub phase_pivot_ppm: f64,
    pub phase_pivot_pick_armed: bool,
    pub baseline_model_choice: BaselineModelChoice1d,
    pub baseline_lambda_log10: f64,
    pub baseline_bernstein_deg: usize,
    pub reprocess_lb_hz: f64,
    pub reprocess_zf_factor: usize,
}

impl std::fmt::Debug for Stack1dState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Stack1dState")
            .field("n_traces", &self.traces.len())
            .field("view_mode", &self.view_mode)
            .field("view_ppm_min", &self.view_ppm_min)
            .field("view_ppm_max", &self.view_ppm_max)
            .field("n_rois", &self.roi_regions.len())
            .field("n_peaks", &self.picked_peaks.len())
            .finish()
    }
}

impl Default for Stack1dState {
    fn default() -> Self {
        Self {
            traces: Vec::new(),
            view_mode: Stack1dViewMode::SuperimposedOverlay,
            covariate_kind: CovariateKind::TimeMinutes,
            ref_trace_idx: 0,
            view_ppm_min: 0.0,
            view_ppm_max: 10.0,
            y_gain: 1.0,
            show_integral_curves: true,
            draw_roi_drag_mode: false,
            stack_vertical_step_px: 26.0,
            stack_tilt_deg: 18.0,
            stack_reverse_order: false,
            diff_alpha_mode: DifferenceAlphaMode::LeastSquaresRegion {
                ppm_min: 1.05,
                ppm_max: 1.35,
                fit_intercept: false,
            },
            manual_alpha: 1.0,
            diff_ref_ppm_min: 1.05,
            diff_ref_ppm_max: 1.35,
            diff_fit_intercept: false,
            show_scaled_ref_overlay: false,
            hide_ref_in_difference: true,
            heatmap_floor_frac: 0.015,
            heatmap_ceil_frac: 0.65,
            heatmap_sort_by_covariate: false,
            heatmap_texture: None,
            heatmap_cache_key: None,
            norm_region_ppm_min: 1.05,
            norm_region_ppm_max: 1.35,
            alignment_scope: AlignmentScope::LocalIntervals,
            alignment_intervals: vec![
                AlignmentInterval::new(7.10, 7.80, 0.06),
                AlignmentInterval::new(3.55, 4.35, 0.06),
            ],
            draft_align_ppm_min: 7.10,
            draft_align_ppm_max: 7.80,
            draft_align_max_shift_ppm: 0.05,
            last_alignment_records: Vec::new(),
            roi_regions: Vec::new(),
            roi_ratio_formula: RoiRatioFormula::AutoReactionOrDecay,
            draft_roi_label: "Region".to_string(),
            draft_roi_ppm_min: 7.12,
            draft_roi_ppm_max: 7.32,
            active_roi_drag_start_ppm: None,
            cursor_ppm: None,
            last_exported_tsv: None,
            last_exported_svg: None,
            show_tsv_preview: false,
            status_message: String::new(),
            show_1d_analysis_drawer: true,
            show_native_open_bar: false,
            native_open_path_input: String::new(),
            pending_open_native_path: None,
            pending_send_to_metabolomics: false,
            phc0_deg: 0.0,
            phc1_deg: 0.0,
            solvent_choice: "CDCl3".to_string(),
            target_total_protons: 10.0,
            journal_style: PublicationStyle::Acs,
            detected_multiplets: Vec::new(),
            detected_multiplet_integrals: Vec::new(),
            formatted_journal_report: String::new(),
            tool_mode: ToolMode1d::Zoom,
            zoom_anchor_ppm: None,
            active_zoom_drag_start_ppm: None,
            active_zoom_drag_start_pos: None,
            zoom_history: Vec::new(),
            zoom_redo_stack: Vec::new(),
            baseline_y_frac: 0.82,
            selected_inspector_tab: InspectorTab1d::Peaks,
            picked_peaks: Vec::new(),
            selected_peak_idx: None,
            selected_multiplet_idx: None,
            picker_min_snr: 6.0,
            estimated_noise_sigma: 1e-6,
            show_peak_labels: true,
            peak_labels_in_hz: false,
            show_j_trees: true,
            show_multiplet_boxes: true,
            show_threshold_bar: false,
            dragging_threshold_bar: false,
            custom_ref_target_ppm: 0.0,
            phase_pivot_mode: PivotMode1d::MaxPeak,
            phase_pivot_ppm: 7.26,
            phase_pivot_pick_armed: false,
            baseline_model_choice: BaselineModelChoice1d::Whittaker,
            baseline_lambda_log10: 6.0,
            baseline_bernstein_deg: 6,
            reprocess_lb_hz: 0.3,
            reprocess_zf_factor: 2,
        }
    }
}

impl Stack1dState {
    /// Active reference spectrometer observe frequency in MHz (fallback `500.0 MHz`).
    pub fn active_obs_mhz(&self) -> f64 {
        if self.traces.is_empty() {
            return 500.0;
        }
        let idx = self.safe_ref_trace_idx();
        self.traces
            .get(idx)
            .and_then(|t| t.raw_spec.as_ref())
            .map(|att| att.0.axis.obs_mhz.max(60.0))
            .unwrap_or(500.0)
    }

    /// Active reference nucleus (`1H` fallback when unannotated).
    pub fn active_nucleus(&self) -> Nucleus1D {
        if self.traces.is_empty() {
            return Nucleus1D::H1;
        }
        let idx = self.safe_ref_trace_idx();
        self.traces
            .get(idx)
            .and_then(|t| t.raw_spec.as_ref())
            .map(|att| att.0.experiment.nucleus())
            .unwrap_or(Nucleus1D::H1)
    }

    /// Push the current `[view_ppm_min, view_ppm_max]` onto `zoom_history` and zoom horizontally into `[ppm_a, ppm_b]`.
    pub fn push_zoom_window(&mut self, ppm_a: f64, ppm_b: f64) {
        let new_lo = ppm_a.min(ppm_b);
        let new_hi = ppm_a.max(ppm_b);
        if !new_lo.is_finite() || !new_hi.is_finite() || (new_hi - new_lo) < 1e-4 {
            return;
        }
        let cur_lo = self.view_ppm_min.min(self.view_ppm_max);
        let cur_hi = self.view_ppm_min.max(self.view_ppm_max);
        if (cur_lo - new_lo).abs() > 1e-6 || (cur_hi - new_hi).abs() > 1e-6 {
            if self.zoom_history.len() >= 64 {
                self.zoom_history.remove(0);
            }
            self.zoom_history.push((cur_lo, cur_hi));
            self.zoom_redo_stack.clear();
        }
        self.view_ppm_min = new_lo;
        self.view_ppm_max = new_hi;
        self.zoom_anchor_ppm = None;
        self.active_zoom_drag_start_ppm = None;
        self.active_zoom_drag_start_pos = None;
    }

    /// Handle two-click spectroscopist zoom (`Click 1` sets `zoom_anchor_ppm`; `Click 2` zooms into `[p1, p2]`).
    /// Returns `true` when the second click completes a zoom transition.
    pub fn handle_canvas_zoom_click(&mut self, click_ppm: f64) -> bool {
        if !click_ppm.is_finite() {
            return false;
        }
        match self.zoom_anchor_ppm {
            None => {
                self.zoom_anchor_ppm = Some(click_ppm);
                false
            }
            Some(p1) => {
                if (click_ppm - p1).abs() >= 1e-4 {
                    self.push_zoom_window(p1, click_ppm);
                    true
                } else {
                    self.zoom_anchor_ppm = None;
                    false
                }
            }
        }
    }

    /// Cancel an active 1st-click zoom anchor (`Esc` or Right-Click).
    pub fn cancel_zoom_anchor(&mut self) {
        self.zoom_anchor_ppm = None;
        self.active_zoom_drag_start_ppm = None;
        self.active_zoom_drag_start_pos = None;
    }

    /// Pop the previous horizontal chemical-shift window from `zoom_history`.
    pub fn undo_zoom(&mut self) -> bool {
        if let Some((prev_lo, prev_hi)) = self.zoom_history.pop() {
            let cur_lo = self.view_ppm_min.min(self.view_ppm_max);
            let cur_hi = self.view_ppm_min.max(self.view_ppm_max);
            self.zoom_redo_stack.push((cur_lo, cur_hi));
            self.view_ppm_min = prev_lo;
            self.view_ppm_max = prev_hi;
            self.zoom_anchor_ppm = None;
            true
        } else {
            false
        }
    }

    /// Re-apply an undone horizontal chemical-shift window from `zoom_redo_stack`.
    pub fn redo_zoom(&mut self) -> bool {
        if let Some((next_lo, next_hi)) = self.zoom_redo_stack.pop() {
            let cur_lo = self.view_ppm_min.min(self.view_ppm_max);
            let cur_hi = self.view_ppm_min.max(self.view_ppm_max);
            self.zoom_history.push((cur_lo, cur_hi));
            self.view_ppm_min = next_lo;
            self.view_ppm_max = next_hi;
            self.zoom_anchor_ppm = None;
            true
        } else {
            false
        }
    }

    /// Reset horizontal `[view_ppm_min, view_ppm_max]` to full trace bounds while preserving `y_gain`.
    pub fn fit_full_x_only(&mut self) {
        if let Some((lo, hi)) = self.global_ppm_bounds() {
            let pad = 0.015 * (hi - lo).max(0.1);
            self.push_zoom_window(lo - pad, hi + pad);
        }
    }

    /// Auto-scale `y_gain` (`0.001..=100_000.0`) so the tallest visible peak inside `[ppm_a, ppm_b]`
    /// sits cleanly at ~75% of the plot headroom above `baseline_y_frac`.
    pub fn fit_y_to_region(&mut self, ppm_a: f64, ppm_b: f64) {
        if self.traces.is_empty() {
            return;
        }
        let lo = ppm_a.min(ppm_b);
        let hi = ppm_a.max(ppm_b);
        let global_ceil = self.reference_intensity_ceiling();
        let mut local_max = 0.0_f64;
        for tr in self.traces.iter().filter(|t| t.visible) {
            if let Some((i0, i1)) = tr.index_range_for_ppm(lo, hi) {
                if i1 - i0 <= 4096 {
                    for idx in i0..=i1 {
                        let p = tr.effective_ppm(idx);
                        if p >= lo && p <= hi {
                            local_max = local_max.max(tr.effective_intensity(idx).abs());
                        }
                    }
                } else {
                    for &(_, v) in &tr.decimate_for_viewport(lo, hi, 512) {
                        local_max = local_max.max(v.abs());
                    }
                }
            }
        }
        if local_max > 1e-18 {
            let target_frac = (0.75 * (self.baseline_y_frac as f64)).clamp(0.18, 0.72);
            let gain = (target_frac / 0.70) * (global_ceil / local_max);
            self.y_gain = gain.clamp(0.001, 100_000.0);
        }
    }

    /// Auto-scale `y_gain` to fit the tallest peak inside the current `[view_ppm_min, view_ppm_max]`.
    pub fn fit_y_to_visible_window(&mut self) {
        self.fit_y_to_region(self.view_ppm_min, self.view_ppm_max);
    }

    /// Spectroscopist vertical scroll rule: plain vertical wheel scales `y_gain` (`0.001..=100_000×`)
    /// without altering `view_ppm_min..view_ppm_max`; `Shift + Scroll` shifts `baseline_y_frac`.
    pub fn apply_vertical_scroll_gain(&mut self, scroll_y: f32, shift_down: bool) {
        if !scroll_y.is_finite() || scroll_y.abs() <= 1e-5 {
            return;
        }
        if shift_down {
            self.baseline_y_frac = (self.baseline_y_frac - scroll_y * 0.0015).clamp(0.12, 0.94);
        } else {
            let g_factor = (scroll_y as f64 * 0.0045).exp().clamp(0.25, 4.0);
            self.y_gain = (self.y_gain * g_factor).clamp(0.001, 100_000.0);
        }
    }

    /// Zoom horizontally to picked peak `peak_idx` and auto-scale `y_gain` so its apex sits at ~75% height.
    pub fn zoom_to_peak(&mut self, peak_idx: usize) {
        let Some(pk) = self.picked_peaks.get(peak_idx).cloned() else {
            return;
        };
        self.selected_peak_idx = Some(peak_idx);
        self.phase_pivot_ppm = pk.ppm;
        let pad = match self.active_nucleus() {
            Nucleus1D::H1 => 0.12,
            _ => 2.0,
        };
        self.push_zoom_window(pk.ppm - pad, pk.ppm + pad);
        self.fit_y_to_region(pk.ppm - 0.5 * pad, pk.ppm + 0.5 * pad);
    }

    /// Zoom horizontally to detected multiplet `mult_idx` and auto-scale `y_gain` to fit its lines.
    pub fn zoom_to_multiplet(&mut self, mult_idx: usize) {
        let Some(m) = self.detected_multiplets.get(mult_idx).cloned() else {
            return;
        };
        self.selected_multiplet_idx = Some(mult_idx);
        let m_lo = m.range_ppm.0.min(m.range_ppm.1);
        let m_hi = m.range_ppm.0.max(m.range_ppm.1);
        let margin = (m_hi - m_lo).max(0.06) * 0.8;
        self.push_zoom_window(m_lo - margin, m_hi + margin);
        self.fit_y_to_region(m_lo - 0.02, m_hi + 0.02);
    }

    /// Normalize multiplet `mult_idx` to `target_protons` (e.g. `1.0`, `2.0`, `3.0` H), rescaling
    /// all multiplet integrals and regenerating the journal characterization report.
    pub fn normalize_multiplet_integral_to(&mut self, mult_idx: usize, target_protons: f64) {
        if target_protons <= 0.0 || mult_idx >= self.detected_multiplet_integrals.len() {
            return;
        }
        let cur = self.detected_multiplet_integrals[mult_idx];
        if cur <= 1e-12 {
            return;
        }
        let factor = target_protons / cur;
        let mut sum_h = 0.0_f64;
        for (an, integ) in self
            .detected_multiplets
            .iter_mut()
            .zip(self.detected_multiplet_integrals.iter_mut())
        {
            *integ *= factor;
            sum_h += *integ;
            an.proton_count = Some((*integ).round().max(1.0) as usize);
        }
        self.target_total_protons = sum_h;
        self.reformat_journal_report();
        self.status_message = format!(
            "Normalized multiplet {} ({:.2} ppm) to {:.2} H (total = {:.1} H).",
            multiplet_letter_id(mult_idx),
            self.detected_multiplets[mult_idx].center_ppm,
            target_protons,
            sum_h
        );
    }

    /// Reference picked peak `peak_idx` to `target_ppm`, shifting the active trace (and unlocked sibling
    /// traces), viewport bounds, ROI brackets, picked peaks, and multiplets consistently.
    pub fn reference_selected_peak_to_ppm(&mut self, peak_idx: usize, target_ppm: f64) {
        let Some(obs_ppm) = self.picked_peaks.get(peak_idx).map(|p| p.ppm) else {
            return;
        };
        if !target_ppm.is_finite() {
            return;
        }
        let Some(ref_i) = self.ensure_active_raw_spec() else {
            return;
        };
        let delta = target_ppm - obs_ppm;
        if let Some(ref mut att) = self.traces[ref_i].raw_spec {
            let _ = att.0.calibrate_reference_peak(obs_ppm, target_ppm);
            let updated = att.0.clone();
            self.traces[ref_i].sync_from_spectrum1d(&updated);
        }
        for (i, tr) in self.traces.iter_mut().enumerate() {
            if i != ref_i && !tr.locked {
                if let Some(ref mut att) = tr.raw_spec {
                    att.0.axis.first_ppm += delta;
                    let updated = att.0.clone();
                    tr.sync_from_spectrum1d(&updated);
                } else {
                    tr.shift_offset_ppm += delta;
                    tr.generation = tr.generation.wrapping_add(1);
                }
            }
        }
        self.view_ppm_min += delta;
        self.view_ppm_max += delta;
        self.phase_pivot_ppm += delta;
        if let Some(ref mut anc) = self.zoom_anchor_ppm {
            *anc += delta;
        }
        for (lo, hi) in &mut self.zoom_history {
            *lo += delta;
            *hi += delta;
        }
        for roi in &mut self.roi_regions {
            roi.ppm_min += delta;
            roi.ppm_max += delta;
        }
        self.refresh_active_trace_peaks_and_multiplets();
        self.selected_peak_idx = self
            .picked_peaks
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                (a.ppm - target_ppm)
                    .abs()
                    .total_cmp(&(b.ppm - target_ppm).abs())
            })
            .map(|(k, _)| k);
        self.status_message = format!(
            "Referenced peak {:.4} -> {:.4} ppm (Δδ = {:+.4} ppm).",
            obs_ppm, target_ppm, delta
        );
    }

    /// Format `self.picked_peaks` as a clean TSV table for clipboard export.
    pub fn export_peaks_tsv(&self) -> String {
        let mut out = String::from("index\tppm\thz\tintensity\tsnr\tfwhm_hz\tnote\n");
        for (i, pk) in self.picked_peaks.iter().enumerate() {
            let _ = writeln!(
                out,
                "{}\t{:.4}\t{:.2}\t{:.6e}\t{:.1}\t{:.2}\t{}",
                i + 1,
                pk.ppm,
                pk.hz,
                pk.height,
                pk.snr,
                pk.fwhm_hz,
                pk.note
            );
        }
        out
    }

    /// Load real user 1D [`Spectrum1D`] traces into `Stack1dState` (automatically replacing any
    /// synthetic demo traces so user drops start on a clean slate, opening the 1D analysis drawer,
    /// and running automatic peak picking, multiplet & journal characterization).
    pub fn load_user_spectra(&mut self, spectra: Vec<Spectrum1D>) {
        if spectra.is_empty() {
            return;
        }
        if self.traces.iter().all(|t| t.is_demo) {
            self.traces.clear();
            self.roi_regions.clear();
        }
        let base_idx = self.traces.len();
        for (k, mut spec) in spectra.into_iter().enumerate() {
            if spec.imag.is_none() {
                hilbert_reconstruct_imaginary_1d(&mut spec);
            }
            if k == 0 {
                if let Some(ref s) = spec.solvent {
                    self.solvent_choice = s.clone();
                }
            }
            let idx = base_idx + k;
            let tr = Trace1d::from_spectrum1d(format!("user_1d_{idx}"), idx as f64, spec)
                .with_color(default_trace_color(idx));
            self.traces.push(tr);
        }
        self.ref_trace_idx = base_idx.min(self.traces.len().saturating_sub(1));
        self.phc0_deg = 0.0;
        self.phc1_deg = 0.0;
        self.show_1d_analysis_drawer = true;
        self.zoom_history.clear();
        self.zoom_redo_stack.clear();
        self.zoom_anchor_ppm = None;
        if self.traces.len() >= 2 && self.view_mode == Stack1dViewMode::SuperimposedOverlay {
            self.view_mode = Stack1dViewMode::StackedOffset;
            self.selected_inspector_tab = InspectorTab1d::MultiTraceStack;
        } else {
            self.selected_inspector_tab = InspectorTab1d::Peaks;
        }
        self.fit_full_ppm_bounds();
        self.y_gain = 1.0;
        self.refresh_active_trace_peaks_and_multiplets();
    }

    fn ensure_active_raw_spec(&mut self) -> Option<usize> {
        if self.traces.is_empty() {
            return None;
        }
        let idx = self.safe_ref_trace_idx();
        self.ref_trace_idx = idx;
        if self.traces[idx].raw_spec.is_none() {
            let tr = &self.traces[idx];
            if tr.ppm.len() >= 2 {
                let first = tr.ppm[0] + tr.shift_offset_ppm;
                let last = *tr.ppm.last().unwrap_or(&tr.ppm[0]) + tr.shift_offset_ppm;
                let axis = AxisPpm::from_bounds("1H", tr.ppm.len(), first, last, 500.0);
                let mut sp = Spectrum1D::new(
                    tr.label.clone(),
                    Experiment1DKind::Proton1D,
                    axis,
                    tr.intensities.clone(),
                    None,
                );
                hilbert_reconstruct_imaginary_1d(&mut sp);
                self.traces[idx].raw_spec = Some(AttachedSpectrum1d(sp));
            }
        } else if let Some(ref mut att) = self.traces[idx].raw_spec {
            if att.0.imag.is_none() {
                hilbert_reconstruct_imaginary_1d(&mut att.0);
            }
        }
        Some(idx)
    }

    fn current_pivot(&self) -> Pivot1D {
        match self.phase_pivot_mode {
            PivotMode1d::MaxPeak => Pivot1D::MaxPeak,
            PivotMode1d::AtPpm => Pivot1D::Ppm(self.phase_pivot_ppm),
        }
    }

    /// Apply `(self.phc0_deg, self.phc1_deg)` relative to the stored `raw_spec` on the active reference trace.
    pub fn apply_live_phase_to_active_trace(&mut self) {
        let Some(idx) = self.ensure_active_raw_spec() else {
            return;
        };
        let ph0 = self.phc0_deg;
        let ph1 = self.phc1_deg;
        let piv = self.current_pivot();
        if let Some(ref att) = self.traces[idx].raw_spec {
            let mut phased = att.0.clone();
            apply_phase_1d(&mut phased, ph0, ph1, piv);
            self.traces[idx].sync_from_spectrum1d(&phased);
        }
    }

    /// Commit live `(phc0_deg, phc1_deg)` permanently into `raw_spec` and refresh peaks/multiplets.
    pub fn commit_live_phase_active_trace(&mut self) {
        let Some(idx) = self.ensure_active_raw_spec() else {
            return;
        };
        let ph0 = self.phc0_deg;
        let ph1 = self.phc1_deg;
        let piv = self.current_pivot();
        if let Some(ref mut att) = self.traces[idx].raw_spec {
            apply_phase_1d(&mut att.0, ph0, ph1, piv);
            let updated = att.0.clone();
            self.traces[idx].sync_from_spectrum1d(&updated);
            self.phc0_deg = 0.0;
            self.phc1_deg = 0.0;
        }
        self.refresh_active_trace_peaks_and_multiplets();
    }

    /// Run minimum-entropy automatic phase correction on the active reference trace and commit it.
    pub fn auto_phase_active_trace(&mut self) {
        let Some(idx) = self.ensure_active_raw_spec() else {
            return;
        };
        let mut msg = String::new();
        if let Some(ref mut att) = self.traces[idx].raw_spec {
            let res = auto_phase_and_apply_1d(&mut att.0, &AutoPhase1DConfig::default());
            let updated = att.0.clone();
            self.traces[idx].sync_from_spectrum1d(&updated);
            self.phc0_deg = 0.0;
            self.phc1_deg = 0.0;
            msg = format!(
                "Auto-phased '{}' (ΔPHC0 = {:.1}°, ΔPHC1 = {:.1}°).",
                self.traces[idx].label, res.phc0_deg, res.phc1_deg
            );
        }
        self.refresh_active_trace_peaks_and_multiplets();
        if !msg.is_empty() {
            self.status_message = msg;
        }
    }

    /// Run automatic baseline flattening (`Whittaker`, `AsymmetricWhittaker`, or `Bernstein`) on the active trace.
    pub fn baseline_correct_active_trace(&mut self) {
        let Some(idx) = self.ensure_active_raw_spec() else {
            return;
        };
        let ph0 = self.phc0_deg;
        let ph1 = self.phc1_deg;
        let piv = self.current_pivot();
        let lambda = 10.0_f64.powf(self.baseline_lambda_log10.clamp(2.0, 9.0));
        let method = match self.baseline_model_choice {
            BaselineModelChoice1d::Whittaker => BaselineMethod1D::Whittaker { lambda },
            BaselineModelChoice1d::AsymmetricWhittaker => BaselineMethod1D::AsymmetricWhittaker {
                lambda,
                asymmetry_p: 0.002,
            },
            BaselineModelChoice1d::Bernstein => BaselineMethod1D::Bernstein {
                degree: self.baseline_bernstein_deg.clamp(2, 12),
            },
        };
        let model_lbl = self.baseline_model_choice.label();
        let mut lbl_trace = String::new();
        if let Some(ref mut att) = self.traces[idx].raw_spec {
            if ph0.abs() > 1e-6 || ph1.abs() > 1e-6 {
                apply_phase_1d(&mut att.0, ph0, ph1, piv);
                self.phc0_deg = 0.0;
                self.phc1_deg = 0.0;
            }
            let cfg = BaselineConfig1D {
                method,
                ..BaselineConfig1D::default()
            };
            let _ = correct_baseline_1d(&mut att.0, &cfg);
            if att.0.imag.is_some() {
                hilbert_reconstruct_imaginary_1d(&mut att.0);
            }
            let updated = att.0.clone();
            self.traces[idx].sync_from_spectrum1d(&updated);
            lbl_trace = self.traces[idx].label.clone();
        }
        self.refresh_active_trace_peaks_and_multiplets();
        if !lbl_trace.is_empty() {
            self.status_message =
                format!("Applied {model_lbl} baseline correction to '{lbl_trace}'.");
        }
    }

    /// Reprocess active trace via causal pseudo-FID reconstruction, exponential line broadening (`reprocess_lb_hz`),
    /// and optional zero-filling (`reprocess_zf_factor`).
    pub fn reprocess_active_trace_lb_zf(&mut self) {
        let Some(idx) = self.ensure_active_raw_spec() else {
            return;
        };
        let lb = self.reprocess_lb_hz.clamp(0.0, 30.0);
        let zf = self.reprocess_zf_factor.clamp(1, 4);
        let mut msg = String::new();
        if let Some(ref mut att) = self.traces[idx].raw_spec {
            let n = att.0.len();
            if n >= 8 {
                let half = n / 2;
                let mut work = vec![[0.0, 0.0]; n];
                for (m, dst) in work.iter_mut().enumerate() {
                    let j = (half + n - (m % n)) % n;
                    *dst = [att.0.real[j], 0.0];
                }
                fft_arbitrary_in_place(&mut work, true);
                if n.is_multiple_of(2) {
                    for c in work.iter_mut().take(half).skip(1) {
                        c[0] *= 2.0;
                        c[1] *= 2.0;
                    }
                    for c in work.iter_mut().skip(half + 1) {
                        *c = [0.0, 0.0];
                    }
                } else {
                    for c in work.iter_mut().take(half + 1).skip(1) {
                        c[0] *= 2.0;
                        c[1] *= 2.0;
                    }
                    for c in work.iter_mut().skip(half + 1) {
                        *c = [0.0, 0.0];
                    }
                }
                let sw_hz = ((n as f64) * att.0.bin_width_hz()).max(1.0);
                let dt = 1.0 / sw_hz;
                if lb > 0.0 {
                    for (k, c) in work.iter_mut().enumerate().take(half + 1) {
                        let w = (-PI * lb * (k as f64) * dt).exp();
                        c[0] *= w;
                        c[1] *= w;
                    }
                }
                let n_zf = (n * zf).min(131_072).max(n);
                let mut work_zf = vec![[0.0, 0.0]; n_zf];
                let copy_len = (half + 1).min(n_zf);
                work_zf[..copy_len].copy_from_slice(&work[..copy_len]);
                fft_arbitrary_in_place(&mut work_zf, false);

                let half_zf = n_zf / 2;
                let mut real = Vec::with_capacity(n_zf);
                let mut imag = Vec::with_capacity(n_zf);
                for j in 0..n_zf {
                    let m = (half_zf + n_zf - (j % n_zf)) % n_zf;
                    real.push(work_zf[m][0]);
                    imag.push(work_zf[m][1]);
                }
                let first_ppm = att.0.axis.first_ppm;
                let last_ppm = att.0.ppm_at(n - 1);
                let new_axis = AxisPpm::from_bounds(
                    &att.0.axis.label,
                    n_zf,
                    first_ppm,
                    last_ppm,
                    att.0.axis.obs_mhz,
                );
                att.0.axis = new_axis;
                att.0.real = real;
                att.0.imag = Some(imag);
                let updated = att.0.clone();
                self.traces[idx].sync_from_spectrum1d(&updated);
                msg = format!(
                    "Applied LB = {:.2} Hz, ZF = {}× ({} pts) to '{}'.",
                    lb, zf, n_zf, self.traces[idx].label
                );
            }
        }
        self.refresh_active_trace_peaks_and_multiplets();
        if !msg.is_empty() {
            self.status_message = msg;
        }
    }

    /// Calibrate the chemical-shift axis of the active trace against residual `self.solvent_choice`.
    pub fn calibrate_solvent_active_trace(&mut self) {
        let Some(idx) = self.ensure_active_raw_spec() else {
            return;
        };
        let solv = self.solvent_choice.trim().to_string();
        let mut ok_delta = None;
        let mut err_msg = None;
        if let Some(ref mut att) = self.traces[idx].raw_spec {
            match auto_calibrate_solvent_1d(&mut att.0, &solv, 0.35) {
                Ok(delta_ppm) => {
                    att.0.solvent = Some(solv.clone());
                    let updated = att.0.clone();
                    self.traces[idx].sync_from_spectrum1d(&updated);
                    ok_delta = Some(delta_ppm);
                }
                Err(e) => {
                    err_msg = Some(e);
                }
            }
        }
        if let Some(delta_ppm) = ok_delta {
            self.fit_full_ppm_bounds();
            self.refresh_active_trace_peaks_and_multiplets();
            self.status_message = format!(
                "Calibrated '{}' to {} residual peak (Δδ = {:+.4} ppm).",
                self.traces[idx].label, solv, delta_ppm
            );
        } else if let Some(e) = err_msg {
            self.status_message = format!("Solvent calibration note: {e}");
        }
    }

    /// Pick 1D peaks on the active reference trace, annotate residual solvents & trace laboratory
    /// impurities, cluster `1H` peaks into multiplets (`Hoye`), and format the journal report.
    pub fn refresh_active_trace_peaks_and_multiplets(&mut self) {
        let Some(idx) = self.ensure_active_raw_spec() else {
            self.picked_peaks.clear();
            self.detected_multiplets.clear();
            self.detected_multiplet_integrals.clear();
            self.formatted_journal_report.clear();
            return;
        };
        let ph0 = self.phc0_deg;
        let ph1 = self.phc1_deg;
        let piv = self.current_pivot();
        let Some(ref att) = self.traces[idx].raw_spec else {
            return;
        };
        let mut working = att.0.clone();
        if ph0.abs() > 1e-6 || ph1.abs() > 1e-6 {
            apply_phase_1d(&mut working, ph0, ph1, piv);
        }

        let mhz = working.axis.obs_mhz.max(60.0);
        let nuc = working.experiment.nucleus();
        let solv_name = working
            .solvent
            .clone()
            .unwrap_or_else(|| self.solvent_choice.clone());
        let max_abs = working
            .real
            .iter()
            .fold(0.0_f64, |m, &v| m.max(v.abs()))
            .max(1e-12);
        let est_sigma = estimate_noise_sigma_1d(&working)
            .max(max_abs * 1e-4)
            .max(1e-12);
        self.estimated_noise_sigma = est_sigma;

        let pcfg = PeakPickConfig1D {
            min_snr: self.picker_min_snr.clamp(1.5, 5000.0),
            min_rel_height: 0.0005,
            ..PeakPickConfig1D::default()
        };
        let mut raw_peaks = pick_peaks_1d(&working, &pcfg);
        if raw_peaks.len() > 256 {
            raw_peaks.sort_by(|a, b| b.height.abs().total_cmp(&a.height.abs()));
            raw_peaks.truncate(256);
        }
        raw_peaks.sort_by(|a, b| b.ppm.total_cmp(&a.ppm));

        let solv_tol = match nuc {
            Nucleus1D::H1 => 0.018,
            _ => 0.40,
        };
        let solv_entry = lookup_deuterated_solvent(&solv_name);
        let solv_refs = solvent_residual_shifts(&solv_name, &nuc).unwrap_or(&[]);

        let mut entries = Vec::with_capacity(raw_peaks.len());
        for pk in &raw_peaks {
            let mut note = String::new();
            let mut is_solv = false;
            if pk.ppm.abs() <= solv_tol {
                note = "TMS / DSS".to_string();
                is_solv = true;
            } else if let Some(r) = solv_refs
                .iter()
                .find(|r| (pk.ppm - r.ppm).abs() <= solv_tol)
            {
                note = format!("{} ({})", solv_name, r.site_label);
                is_solv = true;
            } else if nuc == Nucleus1D::H1
                && solv_entry.is_some_and(|se| (pk.ppm - se.h2o_h1_ppm).abs() <= 0.025)
            {
                note = "H2O / HDO".to_string();
                is_solv = true;
            } else if let Some(imp) =
                match_trace_impurities(&solv_name, &nuc, pk.ppm, 0.012).first()
            {
                // Annotate possible trace laboratory impurities in the Peak Table without excluding
                // genuine analyte aliphatic/aromatic multiplets from multiplet clustering.
                note = format!("Possible {} ({})", imp.entry.compound, imp.entry.site);
            }
            let snr_eff = if pk.snr.is_finite() && pk.snr > 0.0 {
                pk.snr.min(99_999.0)
            } else {
                (pk.height.abs() / est_sigma).min(99_999.0)
            };
            entries.push(Peak1dEntry {
                ppm: pk.ppm,
                hz: pk.ppm * mhz,
                height: pk.height,
                snr: snr_eff,
                fwhm_hz: pk.est_fwhm_hz.max(0.5),
                note,
                is_solvent_or_impurity: is_solv,
            });
        }
        self.picked_peaks = entries;
        if self
            .selected_peak_idx
            .is_some_and(|s| s >= self.picked_peaks.len())
        {
            self.selected_peak_idx = None;
        }

        if raw_peaks.is_empty() {
            self.detected_multiplets.clear();
            self.detected_multiplet_integrals.clear();
            self.formatted_journal_report.clear();
            return;
        }

        // Cluster positive non-TMS/non-solvent peaks (falling back to all positive peaks if all matched)
        let non_solv_pos: Vec<_> = raw_peaks
            .iter()
            .zip(self.picked_peaks.iter())
            .filter(|(pk, ent)| pk.height > 0.0 && !ent.is_solvent_or_impurity)
            .map(|(pk, _)| *pk)
            .collect();
        let cluster_source: Vec<_> = if non_solv_pos.is_empty() {
            raw_peaks.into_iter().filter(|pk| pk.height > 0.0).collect()
        } else {
            non_solv_pos
        };

        let gap_ppm = match nuc {
            Nucleus1D::H1 => (28.0 / mhz).max(0.035),
            _ => (6.0 / mhz).max(0.02),
        };
        let mut clusters: Vec<Vec<MultipletSubPeak>> = Vec::new();
        for pk in cluster_source {
            let sub =
                MultipletSubPeak::new(pk.ppm, pk.height).with_fwhm_hz(pk.est_fwhm_hz.max(0.8));
            if let Some(last_cluster) = clusters.last_mut() {
                let prev_ppm = last_cluster.last().map(|s| s.ppm).unwrap_or(pk.ppm);
                if (prev_ppm - pk.ppm).abs() <= gap_ppm && last_cluster.len() < 24 {
                    last_cluster.push(sub);
                    continue;
                }
            }
            clusters.push(vec![sub]);
        }

        let mcfg = MultipletConfig::default();
        let mut analyses: Vec<MultipletAnalysis> = Vec::new();
        let mut areas: Vec<f64> = Vec::new();
        for cl in clusters.into_iter().take(48) {
            if let Ok(an) = analyze_multiplet_peaks(&cl, mhz, &mcfg) {
                let lo = an.range_ppm.0.min(an.range_ppm.1) - 0.015;
                let hi = an.range_ppm.0.max(an.range_ppm.1) + 0.015;
                let area = integrate_trace_region_f64(&self.traces[idx], lo, hi, true).max(0.0);
                areas.push(area);
                analyses.push(an);
            }
        }

        let total_area: f64 = areas.iter().sum();
        let mut norm_integrals = Vec::with_capacity(areas.len());
        if self.target_total_protons > 0.0 && total_area > 1e-12 {
            for (an, &area) in analyses.iter_mut().zip(&areas) {
                let h_est = (area / total_area) * self.target_total_protons;
                an.proton_count = Some(h_est.round().max(1.0) as usize);
                norm_integrals.push(h_est);
            }
        } else {
            norm_integrals.extend_from_slice(&areas);
        }

        self.detected_multiplets = analyses;
        self.detected_multiplet_integrals = norm_integrals;
        if self
            .selected_multiplet_idx
            .is_some_and(|s| s >= self.detected_multiplets.len())
        {
            self.selected_multiplet_idx = None;
        }
        self.reformat_journal_report();
    }

    /// Pick 1D peaks, cluster into multiplets, deconvolve coupling trees (`Hoye`), and format the
    /// publication characterization line (`ACS` / `JOC` / `RSC` / `Nature`).
    pub fn analyze_active_trace_multiplets(&mut self) {
        self.refresh_active_trace_peaks_and_multiplets();
        if self.picked_peaks.is_empty() {
            self.status_message =
                "No 1D peaks above SNR threshold found on active trace.".to_string();
        } else if let Some(tr) = self.traces.get(self.safe_ref_trace_idx()) {
            self.status_message = format!(
                "Picked {} peak(s) and analyzed {} multiplet(s) on '{}'.",
                self.picked_peaks.len(),
                self.detected_multiplets.len(),
                tr.label
            );
        }
    }

    /// Regenerate `self.formatted_journal_report` from `self.detected_multiplets` under `self.journal_style`.
    pub fn reformat_journal_report(&mut self) {
        if self.detected_multiplets.is_empty() {
            self.formatted_journal_report.clear();
            return;
        }
        let idx = self.safe_ref_trace_idx();
        let (nuc, decoupled, mhz, solv) = self
            .traces
            .get(idx)
            .and_then(|t| t.raw_spec.as_ref())
            .map(|att| {
                let rn = match att.0.experiment {
                    Experiment1DKind::Proton1D => ReportNucleus::H1,
                    Experiment1DKind::Carbon13Decoupled
                    | Experiment1DKind::Dept90
                    | Experiment1DKind::Dept135 => ReportNucleus::C13,
                    Experiment1DKind::Fluorine19 => ReportNucleus::F19,
                    Experiment1DKind::Phosphorus31 => ReportNucleus::P31,
                    Experiment1DKind::Custom(_) => ReportNucleus::H1,
                };
                let dec = matches!(att.0.experiment, Experiment1DKind::Carbon13Decoupled);
                let s = att
                    .0
                    .solvent
                    .clone()
                    .unwrap_or_else(|| self.solvent_choice.clone());
                (rn, dec, att.0.axis.obs_mhz.max(60.0), s)
            })
            .unwrap_or((ReportNucleus::H1, false, 500.0, self.solvent_choice.clone()));

        let signals: Vec<ReportedSignal> = self
            .detected_multiplets
            .iter()
            .map(|an| {
                if decoupled {
                    ReportedSignal::new_decoupled(an.center_ppm)
                } else {
                    ReportedSignal::from_analysis(an, an.proton_count, None)
                }
            })
            .collect();

        let rep = JournalNmrReport {
            nucleus: nuc,
            proton_decoupled: decoupled,
            mhz,
            solvent: solv,
            signals,
        };
        self.formatted_journal_report =
            format_journal_nmr_report(&rep, self.journal_style, TextEncoding::Unicode);
    }

    /// Convert visible traces (`>= 3`) into [`CohortSample1D`] records for one-click Metabolomics handoff.
    pub fn build_metabolomics_cohort_samples(&self) -> Vec<CohortSample1D> {
        let vis: Vec<&Trace1d> = self.traces.iter().filter(|t| t.visible).collect();
        let half = vis.len() / 2;
        vis.into_iter()
            .enumerate()
            .map(|(i, tr)| {
                let spec = if let Some(ref att) = tr.raw_spec {
                    att.0.clone()
                } else {
                    let first = tr.ppm.first().copied().unwrap_or(10.0);
                    let last = tr.ppm.last().copied().unwrap_or(0.0);
                    let axis = AxisPpm::from_bounds("1H", tr.ppm.len(), first, last, 600.0);
                    Spectrum1D::new(
                        tr.label.clone(),
                        Experiment1DKind::Proton1D,
                        axis,
                        tr.intensities.clone(),
                        None,
                    )
                };
                let group = if i < half.max(1) {
                    "Group A"
                } else {
                    "Group B"
                };
                CohortSample1D {
                    id: tr.label.clone(),
                    group: group.to_string(),
                    spectrum: spec,
                    covariates: std::collections::BTreeMap::new(),
                }
            })
            .collect()
    }

    /// Create a fresh [`Stack1dState`] pre-populated with `traces`, fitted to their chemical
    /// shift envelope, and auto-picked for immediate peak & multiplet inspection.
    pub fn with_traces(traces: Vec<Trace1d>) -> Self {
        let mut s = Self {
            traces,
            ..Self::default()
        };
        s.fit_full_ppm_bounds();
        s.refresh_active_trace_peaks_and_multiplets();
        s
    }

    /// Append a [`Trace1d`] and expand/fit the viewport if this is the first trace loaded.
    pub fn add_trace(&mut self, mut trace: Trace1d) {
        if trace.color_rgba == STACK1D_TRACE_PALETTE[0] && !self.traces.is_empty() {
            trace.color_rgba = default_trace_color(self.traces.len());
        }
        let was_empty = self.traces.is_empty();
        self.traces.push(trace);
        if was_empty {
            self.fit_full_ppm_bounds();
            self.refresh_active_trace_peaks_and_multiplets();
        }
    }

    /// Global `(min_ppm, max_ppm)` chemical-shift envelope across all loaded traces.
    pub fn global_ppm_bounds(&self) -> Option<(f64, f64)> {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for tr in &self.traces {
            if let Some((a, b)) = tr.ppm_bounds() {
                lo = lo.min(a);
                hi = hi.max(b);
            }
        }
        if lo.is_finite() && hi.is_finite() && hi > lo {
            Some((lo, hi))
        } else {
            None
        }
    }

    /// Reset `(view_ppm_min, view_ppm_max)`, `y_gain`, and `baseline_y_frac` to fit all loaded traces.
    pub fn fit_full_ppm_bounds(&mut self) {
        if let Some((lo, hi)) = self.global_ppm_bounds() {
            let pad = 0.015 * (hi - lo).max(0.1);
            self.view_ppm_min = lo - pad;
            self.view_ppm_max = hi + pad;
        }
        self.y_gain = 1.0;
        self.baseline_y_frac = 0.82;
        self.zoom_anchor_ppm = None;
    }

    #[inline]
    pub fn safe_ref_trace_idx(&self) -> usize {
        self.ref_trace_idx.min(self.traces.len().saturating_sub(1))
    }

    /// Compute the current ROI integration matrix across all traces.
    pub fn evaluate_roi_matrix(&self) -> RoiMatrix {
        compute_roi_matrix(
            &self.traces,
            &self.roi_regions,
            self.safe_ref_trace_idx(),
            self.roi_ratio_formula,
        )
    }

    /// Export the multi-trace ROI kinetics / titration table as a TSV string.
    pub fn export_roi_tsv(&mut self) -> String {
        let tsv = self.evaluate_roi_matrix().to_tsv_string();
        self.last_exported_tsv = Some(tsv.clone());
        tsv
    }

    fn sync_diff_alpha_mode_bounds(&mut self) {
        if matches!(
            self.diff_alpha_mode,
            DifferenceAlphaMode::LeastSquaresRegion { .. }
        ) {
            self.diff_alpha_mode = DifferenceAlphaMode::LeastSquaresRegion {
                ppm_min: self.diff_ref_ppm_min.min(self.diff_ref_ppm_max),
                ppm_max: self.diff_ref_ppm_min.max(self.diff_ref_ppm_max),
                fit_intercept: self.diff_fit_intercept,
            };
        }
    }

    fn ordered_visible_trace_indices(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = self
            .traces
            .iter()
            .enumerate()
            .filter_map(|(i, t)| t.visible.then_some(i))
            .collect();
        if self.view_mode == Stack1dViewMode::StackedOffset && self.stack_reverse_order {
            indices.reverse();
        }
        indices
    }

    fn reference_intensity_ceiling(&self) -> f64 {
        let max_v = self
            .traces
            .iter()
            .filter(|t| t.visible)
            .fold(0.0_f64, |acc, t| acc.max(t.max_abs_effective()));
        max_v.max(1e-9)
    }

    fn trace_signature_hash(&self) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325_u64;
        for (i, tr) in self.traces.iter().enumerate() {
            h ^= (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            for &b in tr.id.as_bytes() {
                h = h.wrapping_mul(0x0000_0100_0000_01b3) ^ u64::from(b);
            }
            h = h.wrapping_mul(0x0000_0100_0000_01b3) ^ (tr.len() as u64);
            h = h.wrapping_mul(0x0000_0100_0000_01b3) ^ tr.generation;
            h = h.wrapping_mul(0x0000_0100_0000_01b3) ^ tr.scale.to_bits();
            h = h.wrapping_mul(0x0000_0100_0000_01b3) ^ tr.shift_offset_ppm.to_bits();
            h = h.wrapping_mul(0x0000_0100_0000_01b3) ^ u64::from(tr.visible);
            h = h.wrapping_mul(0x0000_0100_0000_01b3) ^ tr.covariate.to_bits();
            if let Some((p_lo, p_hi)) = tr.ppm_bounds() {
                h = h.wrapping_mul(0x0000_0100_0000_01b3) ^ p_lo.to_bits();
                h = h.wrapping_mul(0x0000_0100_0000_01b3) ^ p_hi.to_bits();
            }
            if let Some(top) = tr.lod.levels.last() {
                for bk in top {
                    h = h.wrapping_mul(0x0000_0100_0000_01b3) ^ bk.min_val.to_bits();
                    h = h.wrapping_mul(0x0000_0100_0000_01b3) ^ bk.max_val.to_bits();
                }
            } else {
                for &v in &tr.intensities {
                    h = h.wrapping_mul(0x0000_0100_0000_01b3) ^ v.to_bits();
                }
            }
        }
        h
    }
}

/// Render the self-contained Interactive Multi-Spectrum 1D Comparison, Overlay, Stack, Difference
/// & Kinetics Array Studio inside `ui`.
pub fn render_stack1d_panel(ui: &mut egui::Ui, state: &mut Stack1dState) {
    // Keep legacy `draw_roi_drag_mode` synced if set programmatically
    if state.draw_roi_drag_mode && state.tool_mode == ToolMode1d::Zoom {
        state.tool_mode = ToolMode1d::IntegrateRoi;
    }

    // 1. Primary File, View-Mode & Demo Toolbar
    ui.horizontal_wrapped(|ui| {
        ui.strong("1D NMR Studio");
        ui.separator();
        if ui
            .button("📂 Open Bruker Folder")
            .on_hover_text("Open a Bruker study / expno / pdata folder (1r+procs or fid+acqus)")
            .clicked()
        {
            #[cfg(target_arch = "wasm32")]
            crate::ingest_panel::trigger_browser_folder_picker();
            #[cfg(not(target_arch = "wasm32"))]
            {
                state.show_native_open_bar = !state.show_native_open_bar;
            }
        }
        if ui
            .button("📄 Open 1D Files")
            .on_hover_text(
                "Open .jdx, .ucsf/.usf, .ft1, .zip/.tar.gz Bruker archive, .tsv, or 1r+procs / fid+acqus",
            )
            .clicked()
        {
            #[cfg(target_arch = "wasm32")]
            crate::ingest_panel::trigger_browser_files_picker();
            #[cfg(not(target_arch = "wasm32"))]
            {
                state.show_native_open_bar = !state.show_native_open_bar;
            }
        }
        if !state.traces.is_empty()
            && ui
                .button("🗑 Clear")
                .on_hover_text("Clear all 1D traces")
                .clicked()
        {
            state.traces.clear();
            state.roi_regions.clear();
            state.picked_peaks.clear();
            state.detected_multiplets.clear();
            state.detected_multiplet_integrals.clear();
            state.formatted_journal_report.clear();
            state.zoom_history.clear();
            state.zoom_redo_stack.clear();
            state.zoom_anchor_ppm = None;
            state.status_message = "Cleared 1D traces.".to_string();
        }
        ui.separator();
        for mode in Stack1dViewMode::ALL {
            if ui
                .selectable_value(&mut state.view_mode, mode, mode.label())
                .changed()
                && mode != Stack1dViewMode::SuperimposedOverlay
            {
                state.selected_inspector_tab = InspectorTab1d::MultiTraceStack;
                state.show_1d_analysis_drawer = true;
            }
        }

        if !state.traces.is_empty() && state.traces.len() > 1 {
            ui.separator();
            let safe_ref = state.safe_ref_trace_idx();
            let prev_ref = state.ref_trace_idx;
            state.ref_trace_idx = safe_ref;
            egui::ComboBox::from_id_salt("stack1d_ref_trace_combo")
                .selected_text(format!("Ref: {}", state.traces[safe_ref].label))
                .width(145.0)
                .show_ui(ui, |ui| {
                    for (idx, tr) in state.traces.iter().enumerate() {
                        ui.selectable_value(
                            &mut state.ref_trace_idx,
                            idx,
                            format!("#{idx} {}", tr.label),
                        );
                    }
                });
            if state.ref_trace_idx != prev_ref {
                state.refresh_active_trace_peaks_and_multiplets();
            }
        }

        ui.separator();
        if ui
            .button("Kinetics Demo")
            .on_hover_text("Load synthetic 8-timepoint first-order A -> B reaction kinetics 1D array")
            .clicked()
        {
            let (traces, rois) = build_kinetics_demo_traces();
            state.traces = traces;
            state.roi_regions = rois;
            state.covariate_kind = CovariateKind::TimeMinutes;
            state.ref_trace_idx = 0;
            state.zoom_history.clear();
            state.zoom_redo_stack.clear();
            state.fit_full_ppm_bounds();
            state.refresh_active_trace_peaks_and_multiplets();
            state.status_message = "Loaded 8-trace reaction kinetics 1D array.".to_string();
        }
        if ui
            .button("STD Demo")
            .on_hover_text("Load synthetic STD-NMR off/on-resonance saturation difference series")
            .clicked()
        {
            let (traces, rois) = build_std_difference_demo_traces();
            state.traces = traces;
            state.roi_regions = rois;
            state.covariate_kind = CovariateKind::TimeSeconds;
            state.ref_trace_idx = 0;
            state.view_mode = Stack1dViewMode::DifferenceSpectrum;
            state.selected_inspector_tab = InspectorTab1d::MultiTraceStack;
            state.diff_ref_ppm_min = 1.05;
            state.diff_ref_ppm_max = 1.35;
            state.diff_alpha_mode = DifferenceAlphaMode::LeastSquaresRegion {
                ppm_min: 1.05,
                ppm_max: 1.35,
                fit_intercept: false,
            };
            state.zoom_history.clear();
            state.zoom_redo_stack.clear();
            state.fit_full_ppm_bounds();
            state.y_gain = 3.2;
            state.refresh_active_trace_peaks_and_multiplets();
            state.status_message =
                "Loaded 5-trace STD-NMR difference array (OLS α matched on [1.05, 1.35] ppm)."
                    .to_string();
        }

        if !state.traces.is_empty() {
            if ui
                .button("Copy SVG")
                .on_hover_text("Export vector SVG of the current 1D viewport to clipboard")
                .clicked()
            {
                let svg = export_stack1d_svg(state, 1200, 640);
                ui.ctx().copy_text(svg.clone());
                state.last_exported_svg = Some(svg);
                state.status_message = "Copied 1200×640 vector SVG to clipboard.".to_string();
            }
            if state.traces.len() >= 3
                && ui
                    .button("🧪 Send to Metabolomics")
                    .on_hover_text(
                        "Transfer loaded 1D spectra directly into Metabolomics PCA / OPLS-DA / STOCSY Studio",
                    )
                    .clicked()
            {
                state.pending_send_to_metabolomics = true;
            }
        }
    });

    // 2. Spectroscopist Navigation, Zoom History & Vertical Intensity (`y_gain`) Bar
    if !state.traces.is_empty() {
        ui.horizontal_wrapped(|ui| {
            for tm in ToolMode1d::ALL {
                if ui
                    .selectable_value(&mut state.tool_mode, tm, tm.label())
                    .on_hover_text(tm.tooltip())
                    .changed()
                {
                    state.draw_roi_drag_mode = tm == ToolMode1d::IntegrateRoi;
                    if tm != ToolMode1d::Zoom {
                        state.cancel_zoom_anchor();
                    }
                }
            }
            ui.separator();

            if ui
                .add_enabled(!state.zoom_history.is_empty(), egui::Button::new("⬅ Undo"))
                .on_hover_text("Undo last horizontal zoom (or press Backspace / Z over plot)")
                .clicked()
            {
                state.undo_zoom();
            }
            if ui
                .add_enabled(
                    !state.zoom_redo_stack.is_empty(),
                    egui::Button::new("➡ Redo"),
                )
                .on_hover_text("Redo horizontal zoom")
                .clicked()
            {
                state.redo_zoom();
            }
            if ui
                .button("Full X")
                .on_hover_text("Reset chemical-shift axis to full bounds while keeping vertical gain")
                .clicked()
            {
                state.fit_full_x_only();
            }
            if ui
                .button("Fit Y")
                .on_hover_text("Auto-scale vertical intensity so the tallest peak in view sits at ~75% height")
                .clicked()
            {
                state.fit_y_to_visible_window();
            }
            if ui
                .button("Full View")
                .on_hover_text("Reset full ppm bounds, vertical gain = 1×, and baseline (Double-Click or press F over plot)")
                .clicked()
            {
                state.fit_full_ppm_bounds();
            }

            ui.separator();
            ui.label("Intensity:");
            ui.add(
                egui::DragValue::new(&mut state.y_gain)
                    .range(0.001..=100_000.0)
                    .speed(0.08)
                    .max_decimals(3)
                    .suffix("×"),
            )
            .on_hover_text("Vertical peak gain (Scroll wheel over spectrum scales up/down; Shift+Scroll moves baseline)");
            if ui
                .button("×2")
                .on_hover_text("Double vertical peak height (×2)")
                .clicked()
            {
                state.y_gain = (state.y_gain * 2.0).clamp(0.001, 100_000.0);
            }
            if ui
                .button("÷2")
                .on_hover_text("Halve vertical peak height (÷2)")
                .clicked()
            {
                state.y_gain = (state.y_gain * 0.5).clamp(0.001, 100_000.0);
            }

            ui.separator();
            ui.toggle_value(&mut state.show_peak_labels, "🏷 Peaks")
                .on_hover_text("Show picked peak chemical-shift labels on the spectrum");
            ui.toggle_value(&mut state.show_j_trees, "🌳 J-Trees")
                .on_hover_text("Show multiplet splitting J-trees & clickable multiplet header pills");
            ui.toggle_value(&mut state.show_integral_curves, "∫ Integrals")
                .on_hover_text("Show cumulative integral S-curves & normalized proton counts");
            ui.toggle_value(&mut state.show_threshold_bar, "📏 Threshold")
                .on_hover_text("Show draggable horizontal peak-picking SNR threshold bar on the plot");
        });
    }

    #[cfg(not(target_arch = "wasm32"))]
    if state.show_native_open_bar {
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("1D Folder / File Path:").strong());
            let resp = ui.add(
                egui::TextEdit::singleline(&mut state.native_open_path_input)
                    .hint_text("Enter Bruker folder, pdata/1, 1r, fid, .jdx, .ucsf/.usf, or .zip path (or drag & drop onto window)…")
                    .desired_width(440.0),
            );
            let enter = resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
            if (ui.button("Load 1D Path").clicked() || enter)
                && !state.native_open_path_input.trim().is_empty()
            {
                state.pending_open_native_path =
                    Some(state.native_open_path_input.trim().to_string());
            }
            if ui.small_button("✕").clicked() {
                state.show_native_open_bar = false;
            }
        });
    }

    if !state.status_message.is_empty() {
        ui.label(
            egui::RichText::new(&state.status_message)
                .size(11.5)
                .color(egui::Color32::from_rgb(125, 211, 252)),
        );
    }

    if state.traces.is_empty() {
        ui.add_space(10.0);
        ui.label(
            "No 1D traces loaded yet. Drop a Bruker folder / .zip / .jdx / .usf onto the window, or click 'Kinetics Demo' / 'STD Demo' above.",
        );
        return;
    }

    // Ensure picked peaks exist on first render if traces were populated directly into `state.traces`
    if state.picked_peaks.is_empty() && state.detected_multiplets.is_empty() {
        state.refresh_active_trace_peaks_and_multiplets();
    }

    // Keep `last_exported_tsv` synced when `show_tsv_preview` is enabled
    if state.show_tsv_preview && !state.roi_regions.is_empty() {
        state.last_exported_tsv = Some(state.evaluate_roi_matrix().to_tsv_string());
    }

    // 3. Interactive 1D NMR Canvas + Full-Spectrum Overview Minimap
    paint_interactive_1d_canvas(ui, state);

    // 4. Clean 4-Tab Analytical Bottom Workstation Dock
    ui.add_space(4.0);
    ui.horizontal_wrapped(|ui| {
        let pk_lbl = format!("📌 Peaks ({})", state.picked_peaks.len());
        let mp_lbl = format!(
            "🌳 Multiplets & Report ({})",
            state.detected_multiplets.len()
        );
        let st_lbl = format!("📚 Multi-Trace Stack ({})", state.traces.len());
        for (tab, label) in [
            (InspectorTab1d::Peaks, pk_lbl.as_str()),
            (InspectorTab1d::Multiplets, mp_lbl.as_str()),
            (InspectorTab1d::PhaseBaseline, "🎛 Phase & Baseline"),
            (InspectorTab1d::MultiTraceStack, st_lbl.as_str()),
        ] {
            if ui
                .selectable_label(
                    state.show_1d_analysis_drawer && state.selected_inspector_tab == tab,
                    label,
                )
                .clicked()
            {
                if state.show_1d_analysis_drawer && state.selected_inspector_tab == tab {
                    state.show_1d_analysis_drawer = false;
                } else {
                    state.selected_inspector_tab = tab;
                    state.show_1d_analysis_drawer = true;
                }
            }
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let toggle_txt = if state.show_1d_analysis_drawer {
                "▾ Collapse Dock"
            } else {
                "▸ Expand Dock"
            };
            if ui.small_button(toggle_txt).clicked() {
                state.show_1d_analysis_drawer = !state.show_1d_analysis_drawer;
            }
        });
    });

    if !state.show_1d_analysis_drawer {
        return;
    }

    egui::Frame::group(ui.style())
        .fill(egui::Color32::from_rgb(15, 21, 32))
        .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(45, 58, 78)))
        .inner_margin(egui::Margin::symmetric(8, 6))
        .show(ui, |ui| match state.selected_inspector_tab {
            InspectorTab1d::Peaks => render_peaks_inspector_tab(ui, state),
            InspectorTab1d::Multiplets => render_multiplets_inspector_tab(ui, state),
            InspectorTab1d::PhaseBaseline => render_phase_baseline_inspector_tab(ui, state),
            InspectorTab1d::MultiTraceStack => render_multitrace_stack_inspector_tab(ui, state),
        });
}

fn render_peaks_inspector_tab(ui: &mut egui::Ui, state: &mut Stack1dState) {
    ui.horizontal_wrapped(|ui| {
        ui.strong("Peak Picker:");
        ui.label("Min SNR:");
        if ui
            .add(
                egui::DragValue::new(&mut state.picker_min_snr)
                    .range(1.5..=5000.0)
                    .speed(0.2)
                    .fixed_decimals(1),
            )
            .changed()
        {
            state.refresh_active_trace_peaks_and_multiplets();
        }
        if ui.button("🔄 Re-Pick Peaks").clicked() {
            state.refresh_active_trace_peaks_and_multiplets();
        }
        ui.checkbox(&mut state.show_threshold_bar, "Drag Threshold Line");
        ui.checkbox(&mut state.show_peak_labels, "Show Labels");
        ui.checkbox(&mut state.peak_labels_in_hz, "Labels in Hz");
        if !state.picked_peaks.is_empty() && ui.button("📋 Copy Peaks TSV").clicked() {
            let tsv = state.export_peaks_tsv();
            ui.ctx().copy_text(tsv);
            state.status_message = format!(
                "Copied {} picked peaks (TSV) to clipboard.",
                state.picked_peaks.len()
            );
        }
        ui.label(
            egui::RichText::new("💡 Click any peak row to zoom & auto-scale vertical gain")
                .weak()
                .size(11.0),
        );
    });

    // 1-Click Chemical Shift Referencing bar on selected peak
    if let Some(sel_idx) = state.selected_peak_idx {
        if let Some(sel_pk) = state.picked_peaks.get(sel_idx).cloned() {
            ui.add_space(2.0);
            ui.horizontal_wrapped(|ui| {
                ui.label(
                    egui::RichText::new(format!(
                        "🎯 Reference Peak #{} ({:.4} ppm) ->",
                        sel_idx + 1,
                        sel_pk.ppm
                    ))
                    .color(egui::Color32::from_rgb(125, 211, 252))
                    .strong(),
                );
                let mut ref_target: Option<f64> = None;
                let nuc = state.active_nucleus();
                let presets: &[(&str, f64)] = match nuc {
                    Nucleus1D::C13 => &[
                        ("TMS (0.00)", 0.00),
                        ("CDCl3 (77.16)", 77.16),
                        ("DMSO-d6 (39.52)", 39.52),
                        ("CD3OD (49.00)", 49.00),
                        ("Acetone (29.84)", 29.84),
                    ],
                    _ => &[
                        ("TMS (0.00)", 0.00),
                        ("CDCl3 (7.26)", 7.26),
                        ("DMSO-d6 (2.50)", 2.50),
                        ("D2O (4.79)", 4.79),
                        ("CD3OD (3.31)", 3.31),
                    ],
                };
                for &(lbl, tgt) in presets {
                    if ui.small_button(lbl).clicked() {
                        ref_target = Some(tgt);
                    }
                }
                ui.separator();
                ui.label("Custom δ:");
                ui.add(
                    egui::DragValue::new(&mut state.custom_ref_target_ppm)
                        .speed(0.01)
                        .fixed_decimals(4),
                );
                if ui.small_button("Set δ").clicked() {
                    ref_target = Some(state.custom_ref_target_ppm);
                }
                if let Some(tgt) = ref_target {
                    state.reference_selected_peak_to_ppm(sel_idx, tgt);
                }
            });
        }
    }

    if state.picked_peaks.is_empty() {
        ui.label(
            egui::RichText::new("No peaks above current SNR threshold. Lower 'Min SNR' or drag the Threshold line down.")
                .weak(),
        );
        return;
    }

    let mut zoom_req: Option<usize> = None;
    let mut delete_req: Option<usize> = None;
    egui::ScrollArea::vertical()
        .id_salt("stack1d_peaks_table_scroll")
        .max_height(155.0)
        .show(ui, |ui| {
            egui::Grid::new("stack1d_peaks_grid")
                .striped(true)
                .spacing([14.0, 3.0])
                .show(ui, |ui| {
                    ui.strong("#");
                    ui.strong("δ (ppm)");
                    ui.strong("ν (Hz)");
                    ui.strong("Intensity");
                    ui.strong("SNR");
                    ui.strong("FWHM (Hz)");
                    ui.strong("Annotation / Solvent");
                    ui.strong("");
                    ui.end_row();

                    for (i, pk) in state.picked_peaks.iter().enumerate() {
                        let is_sel = state.selected_peak_idx == Some(i);
                        let row_col = if is_sel {
                            egui::Color32::from_rgb(56, 189, 248)
                        } else if pk.is_solvent_or_impurity {
                            egui::Color32::from_rgb(251, 191, 36)
                        } else {
                            egui::Color32::from_rgb(226, 232, 240)
                        };
                        if ui
                            .selectable_label(is_sel, format!("{}", i + 1))
                            .on_hover_text("Click to zoom to peak & auto-fit Y")
                            .clicked()
                        {
                            zoom_req = Some(i);
                        }
                        if ui
                            .selectable_label(
                                is_sel,
                                egui::RichText::new(format!("{:.4}", pk.ppm))
                                    .monospace()
                                    .color(row_col),
                            )
                            .on_hover_text("Click to zoom to peak & auto-fit Y")
                            .clicked()
                        {
                            zoom_req = Some(i);
                        }
                        if ui
                            .selectable_label(
                                is_sel,
                                egui::RichText::new(format!("{:.1}", pk.hz)).monospace(),
                            )
                            .clicked()
                        {
                            zoom_req = Some(i);
                        }
                        if ui
                            .selectable_label(
                                is_sel,
                                egui::RichText::new(format!("{:.3e}", pk.height)).monospace(),
                            )
                            .clicked()
                        {
                            zoom_req = Some(i);
                        }
                        ui.monospace(format!("{:.1}", pk.snr));
                        ui.monospace(format!("{:.2}", pk.fwhm_hz));
                        ui.label(
                            egui::RichText::new(if pk.note.is_empty() {
                                "—"
                            } else {
                                pk.note.as_str()
                            })
                            .color(if pk.is_solvent_or_impurity {
                                egui::Color32::from_rgb(251, 191, 36)
                            } else {
                                egui::Color32::from_rgb(148, 163, 184)
                            }),
                        );
                        if ui
                            .small_button("✕")
                            .on_hover_text("Remove peak from list")
                            .clicked()
                        {
                            delete_req = Some(i);
                        }
                        ui.end_row();
                    }
                });
        });

    if let Some(rm) = delete_req {
        if rm < state.picked_peaks.len() {
            state.picked_peaks.remove(rm);
            if state.selected_peak_idx == Some(rm) {
                state.selected_peak_idx = None;
            }
        }
    } else if let Some(zi) = zoom_req {
        state.custom_ref_target_ppm = state.picked_peaks.get(zi).map(|p| p.ppm).unwrap_or(0.0);
        state.zoom_to_peak(zi);
    }
}

fn render_multiplets_inspector_tab(ui: &mut egui::Ui, state: &mut Stack1dState) {
    ui.horizontal_wrapped(|ui| {
        if ui.button("🔍 Analyze Multiplets").clicked() {
            state.analyze_active_trace_multiplets();
        }
        ui.label("Total H:");
        if ui
            .add(
                egui::DragValue::new(&mut state.target_total_protons)
                    .range(0.0..=500.0)
                    .speed(0.5)
                    .fixed_decimals(1),
            )
            .changed()
        {
            state.refresh_active_trace_peaks_and_multiplets();
        }
        ui.checkbox(&mut state.show_multiplet_boxes, "Header Boxes");
        ui.checkbox(&mut state.show_j_trees, "Splitting J-Trees");
        ui.checkbox(&mut state.show_integral_curves, "Integrals");
        ui.separator();
        let mut style_changed = false;
        for (st, lbl) in [
            (PublicationStyle::Acs, "ACS"),
            (PublicationStyle::Joc, "JOC"),
            (PublicationStyle::Rsc, "RSC"),
            (PublicationStyle::Nature, "Nature"),
        ] {
            if ui
                .selectable_value(&mut state.journal_style, st, lbl)
                .changed()
            {
                style_changed = true;
            }
        }
        if style_changed {
            state.reformat_journal_report();
        }
        if !state.formatted_journal_report.is_empty()
            && ui.button("📋 Copy Journal Report").clicked()
        {
            ui.ctx().copy_text(state.formatted_journal_report.clone());
            state.status_message =
                "Copied journal NMR characterization report to clipboard.".to_string();
        }
    });

    if !state.formatted_journal_report.is_empty() {
        ui.add_space(2.0);
        ui.monospace(
            egui::RichText::new(&state.formatted_journal_report)
                .size(11.5)
                .color(egui::Color32::from_rgb(185, 245, 215)),
        );
    }

    if state.detected_multiplets.is_empty() {
        ui.label(
            egui::RichText::new("No multiplets detected yet. Click 'Analyze Multiplets' above.")
                .weak(),
        );
        return;
    }

    let mut zoom_mult: Option<usize> = None;
    let mut norm_req: Option<(usize, f64)> = None;
    let mut del_mult: Option<usize> = None;
    ui.add_space(3.0);
    egui::ScrollArea::vertical()
        .id_salt("stack1d_multiplet_table_scroll")
        .max_height(145.0)
        .show(ui, |ui| {
            egui::Grid::new("stack1d_multiplet_grid")
                .striped(true)
                .spacing([12.0, 3.0])
                .show(ui, |ui| {
                    ui.strong("ID");
                    ui.strong("Range (ppm)");
                    ui.strong("δ (ppm)");
                    ui.strong("Class");
                    ui.strong("J (Hz)");
                    ui.strong("Integral (H)");
                    ui.strong("Normalize");
                    ui.strong("");
                    ui.end_row();
                    for (mi, m) in state.detected_multiplets.iter().enumerate() {
                        let is_sel = state.selected_multiplet_idx == Some(mi);
                        let id_str = multiplet_letter_id(mi);
                        let m_lo = m.range_ppm.0.min(m.range_ppm.1);
                        let m_hi = m.range_ppm.0.max(m.range_ppm.1);
                        if ui
                            .selectable_label(is_sel, egui::RichText::new(&id_str).strong())
                            .on_hover_text("Click to zoom into multiplet & auto-fit Y")
                            .clicked()
                        {
                            zoom_mult = Some(mi);
                        }
                        if ui
                            .selectable_label(
                                is_sel,
                                egui::RichText::new(format!("{m_hi:.3}..{m_lo:.3}")).monospace(),
                            )
                            .clicked()
                        {
                            zoom_mult = Some(mi);
                        }
                        if ui
                            .selectable_label(
                                is_sel,
                                egui::RichText::new(format!("{:.3}", m.center_ppm)).monospace(),
                            )
                            .clicked()
                        {
                            zoom_mult = Some(mi);
                        }
                        if ui
                            .selectable_label(
                                is_sel,
                                egui::RichText::new(m.class.as_str()).monospace(),
                            )
                            .clicked()
                        {
                            zoom_mult = Some(mi);
                        }
                        let js = m
                            .j_values_hz()
                            .into_iter()
                            .map(|j| format!("{j:.1}"))
                            .collect::<Vec<_>>()
                            .join(", ");
                        ui.monospace(if js.is_empty() { "—".to_string() } else { js });
                        let integ = state
                            .detected_multiplet_integrals
                            .get(mi)
                            .copied()
                            .unwrap_or(0.0);
                        let h_txt = match m.proton_count {
                            Some(pc) => format!("{integ:.2} ({pc}H)"),
                            None => format!("{integ:.2e}"),
                        };
                        ui.monospace(h_txt);
                        ui.horizontal(|ui| {
                            for (lbl, nh) in [("1H", 1.0), ("2H", 2.0), ("3H", 3.0)] {
                                if ui
                                    .small_button(lbl)
                                    .on_hover_text(format!(
                                        "Normalize all multiplet integrals so {id_str} = {lbl}"
                                    ))
                                    .clicked()
                                {
                                    norm_req = Some((mi, nh));
                                }
                            }
                        });
                        if ui.small_button("✕").clicked() {
                            del_mult = Some(mi);
                        }
                        ui.end_row();
                    }
                });
        });

    if let Some((mi, nh)) = norm_req {
        state.normalize_multiplet_integral_to(mi, nh);
    } else if let Some(rm) = del_mult {
        if rm < state.detected_multiplets.len() {
            state.detected_multiplets.remove(rm);
            if rm < state.detected_multiplet_integrals.len() {
                state.detected_multiplet_integrals.remove(rm);
            }
            if state.selected_multiplet_idx == Some(rm) {
                state.selected_multiplet_idx = None;
            }
            state.reformat_journal_report();
        }
    } else if let Some(mi) = zoom_mult {
        state.zoom_to_multiplet(mi);
    }
}

fn render_phase_baseline_inspector_tab(ui: &mut egui::Ui, state: &mut Stack1dState) {
    ui.horizontal_wrapped(|ui| {
        ui.strong("Interactive Phase:");
        let mut p_changed = false;
        p_changed |= ui
            .add(
                egui::Slider::new(&mut state.phc0_deg, -180.0..=180.0)
                    .text("PHC0 (°)")
                    .fixed_decimals(1),
            )
            .changed();
        for (lbl, d0) in [("-1°", -1.0), ("-0.1°", -0.1), ("+0.1°", 0.1), ("+1°", 1.0)] {
            if ui.small_button(lbl).clicked() {
                state.phc0_deg = (state.phc0_deg + d0).clamp(-180.0, 180.0);
                p_changed = true;
            }
        }
        ui.separator();
        p_changed |= ui
            .add(
                egui::Slider::new(&mut state.phc1_deg, -180.0..=180.0)
                    .text("PHC1 (°)")
                    .fixed_decimals(1),
            )
            .changed();
        for (lbl, d1) in [("-1°", -1.0), ("+1°", 1.0)] {
            if ui.small_button(lbl).clicked() {
                state.phc1_deg = (state.phc1_deg + d1).clamp(-180.0, 180.0);
                p_changed = true;
            }
        }
        if p_changed {
            state.apply_live_phase_to_active_trace();
        }
        if ui
            .button("Commit Phase")
            .on_hover_text("Bake current PHC0/PHC1 into spectrum and update peaks/multiplets")
            .clicked()
        {
            state.commit_live_phase_active_trace();
        }
        if ui.button("Auto-Phase").clicked() {
            state.auto_phase_active_trace();
        }
    });

    ui.add_space(2.0);
    ui.horizontal_wrapped(|ui| {
        ui.label("PHC1 Pivot:");
        let p_mode_changed_a = ui
            .selectable_value(
                &mut state.phase_pivot_mode,
                PivotMode1d::MaxPeak,
                "Max Peak",
            )
            .changed();
        let p_mode_changed_b = ui
            .selectable_value(
                &mut state.phase_pivot_mode,
                PivotMode1d::AtPpm,
                "At δ (ppm)",
            )
            .changed();
        let piv_drag = ui
            .add_enabled(
                state.phase_pivot_mode == PivotMode1d::AtPpm,
                egui::DragValue::new(&mut state.phase_pivot_ppm)
                    .speed(0.01)
                    .fixed_decimals(3)
                    .suffix(" ppm"),
            )
            .changed();
        if ui
            .toggle_value(&mut state.phase_pivot_pick_armed, "🎯 Click Pivot on Plot")
            .on_hover_text("Click any peak on the 1D plot to place the first-order phase pivot")
            .clicked()
            && state.phase_pivot_pick_armed
        {
            state.phase_pivot_mode = PivotMode1d::AtPpm;
        }
        if p_mode_changed_a || p_mode_changed_b || piv_drag {
            state.apply_live_phase_to_active_trace();
        }

        ui.separator();
        ui.strong("Baseline:");
        egui::ComboBox::from_id_salt("stack1d_baseline_model_combo")
            .selected_text(state.baseline_model_choice.label())
            .width(155.0)
            .show_ui(ui, |ui| {
                for m in [
                    BaselineModelChoice1d::Whittaker,
                    BaselineModelChoice1d::AsymmetricWhittaker,
                    BaselineModelChoice1d::Bernstein,
                ] {
                    ui.selectable_value(&mut state.baseline_model_choice, m, m.label());
                }
            });
        if state.baseline_model_choice == BaselineModelChoice1d::Bernstein {
            ui.add(
                egui::DragValue::new(&mut state.baseline_bernstein_deg)
                    .range(2..=12)
                    .prefix("deg "),
            );
        } else {
            ui.add(
                egui::DragValue::new(&mut state.baseline_lambda_log10)
                    .range(2.0..=9.0)
                    .speed(0.1)
                    .fixed_decimals(1)
                    .prefix("log₁₀λ "),
            );
        }
        if ui.button("Apply Baseline").clicked() {
            state.baseline_correct_active_trace();
        }
    });

    ui.add_space(2.0);
    ui.horizontal_wrapped(|ui| {
        ui.strong("Solvent Ref:");
        egui::ComboBox::from_id_salt("stack1d_solvent_combo")
            .selected_text(&state.solvent_choice)
            .width(105.0)
            .show_ui(ui, |ui| {
                for s in ["CDCl3", "DMSO-d6", "D2O", "CD3OD", "Acetone-d6", "C6D6"] {
                    ui.selectable_value(&mut state.solvent_choice, s.to_string(), s);
                }
            });
        if ui.button("Auto-Calibrate δ").clicked() {
            state.calibrate_solvent_active_trace();
        }
        ui.separator();
        ui.strong("Apodization & Zero-Fill:");
        ui.add(
            egui::DragValue::new(&mut state.reprocess_lb_hz)
                .range(0.0..=30.0)
                .speed(0.05)
                .fixed_decimals(2)
                .prefix("LB ")
                .suffix(" Hz"),
        );
        for zf in [1usize, 2, 4] {
            ui.selectable_value(&mut state.reprocess_zf_factor, zf, format!("{zf}× ZF"));
        }
        if ui
            .button("Apply LB / ZF")
            .on_hover_text(
                "Apply exponential line broadening & zero-filling via causal pseudo-FID FFT",
            )
            .clicked()
        {
            state.reprocess_active_trace_lb_zf();
        }
    });
}

fn render_multitrace_stack_inspector_tab(ui: &mut egui::Ui, state: &mut Stack1dState) {
    match state.view_mode {
        Stack1dViewMode::SuperimposedOverlay => {}
        Stack1dViewMode::StackedOffset => {
            ui.horizontal_wrapped(|ui| {
                ui.label(egui::RichText::new("Stack Layout:").weak());
                ui.add(
                    egui::Slider::new(&mut state.stack_vertical_step_px, 0.0..=90.0)
                        .text("Vertical step (px)"),
                );
                ui.add(
                    egui::Slider::new(&mut state.stack_tilt_deg, -45.0..=60.0)
                        .text("Oblique angle (°)"),
                );
                ui.checkbox(&mut state.stack_reverse_order, "Top-to-bottom reverse");
            });
        }
        Stack1dViewMode::DifferenceSpectrum => {
            ui.horizontal_wrapped(|ui| {
                ui.label(egui::RichText::new("Difference S_i - α·S_ref:").weak());
                let mut is_auto = matches!(
                    state.diff_alpha_mode,
                    DifferenceAlphaMode::LeastSquaresRegion { .. }
                );
                if ui.radio_value(&mut is_auto, false, "Manual α").changed() && !is_auto {
                    state.diff_alpha_mode = DifferenceAlphaMode::Manual;
                }
                if ui
                    .radio_value(&mut is_auto, true, "Least-Squares Auto α")
                    .changed()
                    && is_auto
                {
                    state.sync_diff_alpha_mode_bounds();
                }

                if is_auto {
                    ui.label("Ref region [ppm]:");
                    let c1 = ui
                        .add(
                            egui::DragValue::new(&mut state.diff_ref_ppm_min)
                                .speed(0.01)
                                .fixed_decimals(3),
                        )
                        .changed();
                    ui.label("..");
                    let c2 = ui
                        .add(
                            egui::DragValue::new(&mut state.diff_ref_ppm_max)
                                .speed(0.01)
                                .fixed_decimals(3),
                        )
                        .changed();
                    let c3 = ui
                        .checkbox(&mut state.diff_fit_intercept, "+ β baseline")
                        .changed();
                    if c1 || c2 || c3 {
                        state.sync_diff_alpha_mode_bounds();
                    }
                } else {
                    ui.add(
                        egui::Slider::new(&mut state.manual_alpha, -2.0..=3.0)
                            .text("α")
                            .fixed_decimals(4),
                    );
                }
                ui.checkbox(&mut state.show_scaled_ref_overlay, "Show α·S_ref");
                ui.checkbox(&mut state.hide_ref_in_difference, "Hide S_ref (0)");
            });
        }
        Stack1dViewMode::Pseudo2dHeatmap => {
            ui.horizontal_wrapped(|ui| {
                ui.label(egui::RichText::new("Pseudo-2D Raster:").weak());
                ui.add(
                    egui::Slider::new(&mut state.heatmap_floor_frac, 0.0..=0.40)
                        .text("Noise floor"),
                );
                ui.add(egui::Slider::new(&mut state.heatmap_ceil_frac, 0.05..=1.0).text("Ceiling"));
                ui.checkbox(&mut state.heatmap_sort_by_covariate, "Sort Y by covariate");
            });
        }
    }

    ui.horizontal_wrapped(|ui| {
        ui.strong("Normalize:");
        if ui.button("Total Area").clicked() {
            normalize_traces(&mut state.traces, NormalizationMode::TotalArea);
            state.status_message =
                "Applied total-area normalization across unlocked traces.".to_string();
        }
        if ui
            .button("Total |Area|")
            .on_hover_text("Normalize by integral of |S(δ)| (ideal for DEPT-135)")
            .clicked()
        {
            normalize_traces(&mut state.traces, NormalizationMode::TotalAbsArea);
            state.status_message = "Applied absolute total-area normalization.".to_string();
        }
        ui.label("Ref [ppm]:");
        ui.add(
            egui::DragValue::new(&mut state.norm_region_ppm_min)
                .speed(0.01)
                .fixed_decimals(3),
        );
        ui.label("..");
        ui.add(
            egui::DragValue::new(&mut state.norm_region_ppm_max)
                .speed(0.01)
                .fixed_decimals(3),
        );
        if ui.button("Ref Peak").clicked() {
            normalize_traces(
                &mut state.traces,
                NormalizationMode::ReferencePeakHeight {
                    ppm_min: state.norm_region_ppm_min,
                    ppm_max: state.norm_region_ppm_max,
                },
            );
        }
        if ui.button("Ref Integral").clicked() {
            normalize_traces(
                &mut state.traces,
                NormalizationMode::ReferenceRegionIntegral {
                    ppm_min: state.norm_region_ppm_min,
                    ppm_max: state.norm_region_ppm_max,
                },
            );
        }
        if ui.button("Reset Raw").clicked() {
            for tr in &mut state.traces {
                tr.reset_to_raw();
            }
            state.last_alignment_records.clear();
        }
        ui.separator();
        if ui
            .button("Run FFT Alignment")
            .on_hover_text("Align unlocked traces to reference trace via FFT cross-correlation")
            .clicked()
        {
            if state.alignment_intervals.is_empty() {
                state.alignment_intervals.push(AlignmentInterval::new(
                    state.draft_align_ppm_min,
                    state.draft_align_ppm_max,
                    state.draft_align_max_shift_ppm,
                ));
            }
            let ref_i = state.safe_ref_trace_idx();
            let intervals = state.alignment_intervals.clone();
            let scope = state.alignment_scope;
            let recs = align_traces_interval_fft(&mut state.traces, ref_i, &intervals, scope);
            state.last_alignment_records = recs;
        }
        if ui.button("Copy ROI TSV").clicked() {
            let tsv = state.export_roi_tsv();
            ui.ctx().copy_text(tsv);
            state.show_tsv_preview = true;
        }
        ui.checkbox(&mut state.show_tsv_preview, "TSV Preview");
    });

    egui::CollapsingHeader::new(format!("Traces & Covariates ({})", state.traces.len()))
        .id_salt("stack1d_trace_table_header")
        .default_open(state.traces.len() > 1)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("Covariate Axis:");
                egui::ComboBox::from_id_salt("stack1d_cov_kind_combo")
                    .selected_text(state.covariate_kind.label())
                    .show_ui(ui, |ui| {
                        for ck in CovariateKind::ALL {
                            ui.selectable_value(&mut state.covariate_kind, ck, ck.label());
                        }
                    });
            });

            let ref_idx = state.safe_ref_trace_idx();
            egui::ScrollArea::vertical()
                .id_salt("stack1d_trace_table_scroll")
                .max_height(125.0)
                .show(ui, |ui| {
                    egui::Grid::new("stack1d_trace_grid")
                        .striped(true)
                        .spacing([10.0, 3.0])
                        .show(ui, |ui| {
                            ui.strong("Vis");
                            ui.strong("Lock");
                            ui.strong("Color");
                            ui.strong("Trace");
                            ui.strong(state.covariate_kind.label());
                            ui.strong("Scale");
                            ui.strong("V-Offset");
                            ui.strong("Δδ (ppm)");
                            ui.end_row();

                            for (idx, tr) in state.traces.iter_mut().enumerate() {
                                ui.checkbox(&mut tr.visible, "");
                                ui.checkbox(&mut tr.locked, "");
                                let (swatch_rect, swatch_resp) = ui.allocate_exact_size(
                                    egui::vec2(16.0, 14.0),
                                    egui::Sense::click(),
                                );
                                let [r, g, b, a] = tr.color_rgba;
                                ui.painter().rect_filled(
                                    swatch_rect,
                                    3.0,
                                    egui::Color32::from_rgba_unmultiplied(r, g, b, a),
                                );
                                if swatch_resp.clicked() {
                                    let next_p = STACK1D_TRACE_PALETTE
                                        .iter()
                                        .position(|&c| c == tr.color_rgba)
                                        .map(|p| (p + 1) % STACK1D_TRACE_PALETTE.len())
                                        .unwrap_or(0);
                                    tr.color_rgba = STACK1D_TRACE_PALETTE[next_p];
                                }
                                let badge = if idx == ref_idx {
                                    format!("★ {}", tr.label)
                                } else {
                                    tr.label.clone()
                                };
                                ui.label(badge);
                                ui.add_enabled(
                                    !tr.locked,
                                    egui::DragValue::new(&mut tr.covariate)
                                        .speed(0.1)
                                        .fixed_decimals(2),
                                );
                                if ui
                                    .add_enabled(
                                        !tr.locked,
                                        egui::DragValue::new(&mut tr.scale)
                                            .speed(0.02)
                                            .fixed_decimals(4),
                                    )
                                    .changed()
                                {
                                    tr.generation = tr.generation.wrapping_add(1);
                                }
                                ui.add_enabled(
                                    !tr.locked,
                                    egui::DragValue::new(&mut tr.vertical_offset)
                                        .speed(0.01)
                                        .fixed_decimals(3),
                                );
                                if ui
                                    .add_enabled(
                                        !tr.locked,
                                        egui::DragValue::new(&mut tr.shift_offset_ppm)
                                            .speed(0.001)
                                            .fixed_decimals(4),
                                    )
                                    .changed()
                                {
                                    tr.generation = tr.generation.wrapping_add(1);
                                }
                                ui.end_row();
                            }
                        });
                });
        });

    egui::CollapsingHeader::new(format!(
        "Multi-Trace ROI Kinetics Matrix ({} ROIs)",
        state.roi_regions.len()
    ))
    .id_salt("stack1d_roi_matrix_header")
    .default_open(!state.roi_regions.is_empty() && state.traces.len() > 1)
    .show(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.label("New ROI:");
            ui.add(egui::TextEdit::singleline(&mut state.draft_roi_label).desired_width(90.0));
            ui.add(
                egui::DragValue::new(&mut state.draft_roi_ppm_min)
                    .speed(0.01)
                    .fixed_decimals(3)
                    .prefix("min "),
            );
            ui.add(
                egui::DragValue::new(&mut state.draft_roi_ppm_max)
                    .speed(0.01)
                    .fixed_decimals(3)
                    .prefix("max "),
            );
            if ui.button("+ Add ROI").clicked() {
                let lbl = if state.draft_roi_label.trim().is_empty() {
                    format!("region_{}", state.roi_regions.len() + 1)
                } else {
                    state.draft_roi_label.trim().to_string()
                };
                state.roi_regions.push(RoiRegion1d::new(
                    lbl,
                    state.draft_roi_ppm_min,
                    state.draft_roi_ppm_max,
                ));
            }
            if !state.roi_regions.is_empty() && ui.button("Clear ROIs").clicked() {
                state.roi_regions.clear();
            }
        });

        if !state.roi_regions.is_empty() {
            let matrix = state.evaluate_roi_matrix();
            if state.show_tsv_preview {
                let mut tsv = matrix.to_tsv_string();
                state.last_exported_tsv = Some(tsv.clone());
                ui.add(
                    egui::TextEdit::multiline(&mut tsv)
                        .font(egui::TextStyle::Monospace)
                        .desired_rows(4)
                        .desired_width(f32::INFINITY),
                );
            }
        }
    });
}

fn paint_interactive_1d_canvas(ui: &mut egui::Ui, state: &mut Stack1dState) {
    let avail_w = ui.available_width().max(320.0);
    let target_frac = if state.show_1d_analysis_drawer {
        0.60
    } else {
        0.86
    };
    let canvas_h = (ui.available_height() * target_frac).clamp(340.0, 680.0);
    let (canvas_rect, resp) =
        ui.allocate_exact_size(egui::vec2(avail_w, canvas_h), egui::Sense::click_and_drag());

    let pad_l = 14.0_f32;
    let pad_r = 14.0_f32;
    let pad_t = 10.0_f32;
    let ruler_h = 22.0_f32;
    let minimap_h = 36.0_f32;
    let bottom_reserve = ruler_h + minimap_h + 10.0;

    let plot_rect = egui::Rect::from_min_max(
        egui::pos2(canvas_rect.left() + pad_l, canvas_rect.top() + pad_t),
        egui::pos2(
            (canvas_rect.right() - pad_r).max(canvas_rect.left() + pad_l + 40.0),
            (canvas_rect.bottom() - bottom_reserve).max(canvas_rect.top() + pad_t + 80.0),
        ),
    );
    let minimap_rect = egui::Rect::from_min_max(
        egui::pos2(plot_rect.left(), canvas_rect.bottom() - minimap_h - 4.0),
        egui::pos2(plot_rect.right(), canvas_rect.bottom() - 4.0),
    );

    let (g_lo_raw, g_hi_raw) = state.global_ppm_bounds().unwrap_or((0.0, 10.0));
    let g_pad = 0.015 * (g_hi_raw - g_lo_raw).max(0.1);
    let g_lo = g_lo_raw - g_pad;
    let g_hi = g_hi_raw + g_pad;
    let g_span = (g_hi - g_lo).max(1e-6);

    let mut lo = state.view_ppm_min.min(state.view_ppm_max);
    let mut hi = state.view_ppm_min.max(state.view_ppm_max);
    if (hi - lo) < 1e-5 {
        hi = lo + 1e-5;
    }

    let screen_x_to_ppm = |sx: f32, cur_lo: f64, cur_hi: f64| -> f64 {
        let frac = ((sx - plot_rect.left()) / plot_rect.width().max(1.0)) as f64;
        cur_hi - frac * (cur_hi - cur_lo)
    };
    let minimap_x_to_ppm = |sx: f32| -> f64 {
        let frac =
            ((sx - minimap_rect.left()) / minimap_rect.width().max(1.0)).clamp(0.0, 1.0) as f64;
        g_hi - frac * g_span
    };

    // Keyboard shortcuts (Esc cancels zoom anchor; Backspace/Z undoes zoom; F/R resets full view)
    let wants_kb = ui.ctx().egui_wants_keyboard_input();
    if !wants_kb {
        let (esc, undo_key, full_key) = ui.input(|i| {
            (
                i.key_pressed(egui::Key::Escape),
                i.key_pressed(egui::Key::Backspace) || i.key_pressed(egui::Key::Z),
                i.key_pressed(egui::Key::F) || i.key_pressed(egui::Key::R),
            )
        });
        if esc {
            state.cancel_zoom_anchor();
            state.phase_pivot_pick_armed = false;
        }
        if resp.hovered() {
            if undo_key && state.undo_zoom() {
                lo = state.view_ppm_min.min(state.view_ppm_max);
                hi = state.view_ppm_min.max(state.view_ppm_max);
            } else if full_key {
                state.fit_full_ppm_bounds();
                lo = state.view_ppm_min.min(state.view_ppm_max);
                hi = state.view_ppm_min.max(state.view_ppm_max);
            }
        }
    }

    // Right-click cancels active 1st-click zoom anchor
    if resp.secondary_clicked() {
        state.cancel_zoom_anchor();
        state.phase_pivot_pick_armed = false;
    }

    // Double-click inside plot resets full NMR view
    if resp.double_clicked()
        && resp
            .interact_pointer_pos()
            .is_none_or(|p| plot_rect.contains(p))
    {
        state.fit_full_ppm_bounds();
        lo = state.view_ppm_min.min(state.view_ppm_max);
        hi = state.view_ppm_min.max(state.view_ppm_max);
    }

    // Hover & Wheel: plain vertical scroll scales `y_gain` (`0.001..=100_000×`), Shift+Scroll shifts baseline,
    // Ctrl/Cmd+Scroll or pinch zooms horizontally, and vertical scroll is ALWAYS consumed so outer ScrollAreas never hijack it.
    if let Some(hover_pos) = resp.hover_pos() {
        if plot_rect.contains(hover_pos) {
            let anchor_ppm = screen_x_to_ppm(hover_pos.x, lo, hi);
            state.cursor_ppm = Some(anchor_ppm);
        } else {
            state.cursor_ppm = None;
        }

        if canvas_rect.contains(hover_pos) {
            let (wheel_sy, smooth_sy, scroll_x, zoom_delta, shift_down, cmd_or_ctrl) =
                ui.input(|i| {
                    let wy: f32 = i
                        .events
                        .iter()
                        .filter_map(|ev| match ev {
                            egui::Event::MouseWheel { delta, .. } => Some(delta.y),
                            _ => None,
                        })
                        .sum();
                    (
                        wy,
                        i.smooth_scroll_delta.y,
                        i.smooth_scroll_delta.x,
                        i.zoom_delta(),
                        i.modifiers.shift,
                        i.modifiers.command || i.modifiers.ctrl,
                    )
                });
            let sy = if smooth_sy.abs() >= wheel_sy.abs() {
                smooth_sy
            } else {
                wheel_sy * 16.0
            };
            if smooth_sy.abs() > 0.0 || wheel_sy.abs() > 0.0 {
                ui.input_mut(|i| {
                    i.smooth_scroll_delta.y = 0.0;
                });
            }

            let anchor_ppm = screen_x_to_ppm(
                hover_pos.x.clamp(plot_rect.left(), plot_rect.right()),
                lo,
                hi,
            );
            if (zoom_delta - 1.0).abs() > 1e-4 {
                let factor = (1.0 / (zoom_delta as f64)).clamp(0.25, 4.0);
                let new_lo = anchor_ppm - (anchor_ppm - lo) * factor;
                let new_hi = anchor_ppm + (hi - anchor_ppm) * factor;
                if (new_hi - new_lo) >= 1e-4 && (new_hi - new_lo) <= 1000.0 {
                    lo = new_lo;
                    hi = new_hi;
                    state.view_ppm_min = lo;
                    state.view_ppm_max = hi;
                }
            } else if cmd_or_ctrl && sy.abs() > 0.05 {
                let z_factor = (-sy as f64 * 0.0035).exp().clamp(0.4, 2.5);
                let new_lo = anchor_ppm - (anchor_ppm - lo) * z_factor;
                let new_hi = anchor_ppm + (hi - anchor_ppm) * z_factor;
                if (new_hi - new_lo) >= 1e-4 && (new_hi - new_lo) <= 1000.0 {
                    lo = new_lo;
                    hi = new_hi;
                    state.view_ppm_min = lo;
                    state.view_ppm_max = hi;
                }
            } else if sy.abs() > 0.05 {
                state.apply_vertical_scroll_gain(sy, shift_down);
            }

            if scroll_x.abs() > 0.1 {
                let d_ppm = (scroll_x / plot_rect.width().max(1.0)) as f64 * (hi - lo);
                lo += d_ppm;
                hi += d_ppm;
                state.view_ppm_min = lo;
                state.view_ppm_max = hi;
            }
        }
    } else {
        state.cursor_ppm = None;
    }

    let max_int = state.reference_intensity_ceiling();
    let y_base = plot_rect.top() + plot_rect.height() * state.baseline_y_frac;
    let y_px_per_unit = ((plot_rect.height() as f64) * 0.70 * state.y_gain) / max_int;
    let thr_int = state.picker_min_snr * state.estimated_noise_sigma;
    let thr_y = (y_base - (thr_int * y_px_per_unit) as f32)
        .clamp(plot_rect.top() + 8.0, plot_rect.bottom() - 8.0);

    // Precompute clickable multiplet header pill rectangles across `plot_rect`
    let cur_span = (hi - lo).max(1e-6);
    let mut multiplet_pill_rects: Vec<(usize, egui::Rect)> = Vec::new();
    if state.show_multiplet_boxes && state.view_mode != Stack1dViewMode::Pseudo2dHeatmap {
        let mut last_pill_right = plot_rect.left() - 10.0;
        for (mi, m) in state.detected_multiplets.iter().enumerate() {
            let m_lo = m.range_ppm.0.min(m.range_ppm.1);
            let m_hi = m.range_ppm.0.max(m.range_ppm.1);
            if m_hi < lo || m_lo > hi {
                continue;
            }
            let cx =
                plot_rect.left() + (((hi - m.center_ppm) / cur_span) as f32) * plot_rect.width();
            if cx < plot_rect.left() + 18.0 || cx > plot_rect.right() - 18.0 {
                continue;
            }
            let pill_w = 88.0_f32;
            let pill_h = 16.0_f32;
            let x0 = (cx - 0.5 * pill_w).max(last_pill_right + 4.0);
            if x0 + pill_w > plot_rect.right() - 4.0 {
                continue;
            }
            let rect = egui::Rect::from_min_size(
                egui::pos2(x0, plot_rect.top() + 24.0),
                egui::vec2(pill_w, pill_h),
            );
            last_pill_right = rect.right();
            multiplet_pill_rects.push((mi, rect));
        }
    }

    let (shift_held, space_held, sec_or_mid_down) = ui.input(|i| {
        (
            i.modifiers.shift,
            i.key_down(egui::Key::Space),
            i.pointer.secondary_down() || i.pointer.middle_down(),
        )
    });
    let roi_drag_wanted = state.tool_mode == ToolMode1d::IntegrateRoi || shift_held;
    let pan_drag_wanted = state.tool_mode == ToolMode1d::Pan || space_held || sec_or_mid_down;

    // Minimap click / drag jump
    let ptr_pos = resp.interact_pointer_pos();
    if (resp.clicked() || resp.dragged())
        && ptr_pos.is_some_and(|p| minimap_rect.contains(p))
        && !state.dragging_threshold_bar
        && state.active_roi_drag_start_ppm.is_none()
        && state.active_zoom_drag_start_ppm.is_none()
    {
        if let Some(p) = ptr_pos {
            let center = minimap_x_to_ppm(p.x);
            let cur_w = (hi - lo).min(g_span);
            let mut new_lo = center - 0.5 * cur_w;
            let mut new_hi = center + 0.5 * cur_w;
            if new_lo < g_lo {
                new_hi += g_lo - new_lo;
                new_lo = g_lo;
            }
            if new_hi > g_hi {
                new_lo -= new_hi - g_hi;
                new_hi = g_hi;
            }
            lo = new_lo;
            hi = new_hi;
            state.view_ppm_min = lo;
            state.view_ppm_max = hi;
        }
    } else {
        // Primary single click inside `plot_rect`: multiplet pill click, phase pivot pick, or 2-click zoom
        if resp.clicked_by(egui::PointerButton::Primary) && !resp.double_clicked() {
            if let Some(pos) = ptr_pos {
                if let Some(&(mi, _)) = multiplet_pill_rects.iter().find(|(_, r)| r.contains(pos)) {
                    state.zoom_to_multiplet(mi);
                    lo = state.view_ppm_min.min(state.view_ppm_max);
                    hi = state.view_ppm_min.max(state.view_ppm_max);
                } else if plot_rect.contains(pos) {
                    let click_ppm = screen_x_to_ppm(pos.x, lo, hi);
                    if state.phase_pivot_pick_armed {
                        state.phase_pivot_ppm = click_ppm;
                        state.phase_pivot_mode = PivotMode1d::AtPpm;
                        state.phase_pivot_pick_armed = false;
                        state.apply_live_phase_to_active_trace();
                        state.status_message =
                            format!("Set PHC1 phase pivot at δ = {click_ppm:.4} ppm.");
                    } else if state.tool_mode == ToolMode1d::Zoom
                        && !shift_held
                        && !space_held
                        && state.handle_canvas_zoom_click(click_ppm)
                    {
                        lo = state.view_ppm_min.min(state.view_ppm_max);
                        hi = state.view_ppm_min.max(state.view_ppm_max);
                    }
                }
            }
        }

        // Drag start inside `plot_rect`
        if resp.drag_started() {
            if let Some(pos) = ptr_pos {
                if plot_rect.contains(pos) {
                    if state.show_threshold_bar && (pos.y - thr_y).abs() <= 8.5 {
                        state.dragging_threshold_bar = true;
                        state.active_roi_drag_start_ppm = None;
                        state.active_zoom_drag_start_ppm = None;
                        state.active_zoom_drag_start_pos = None;
                    } else if roi_drag_wanted {
                        state.active_roi_drag_start_ppm = Some(screen_x_to_ppm(pos.x, lo, hi));
                        state.active_zoom_drag_start_ppm = None;
                        state.active_zoom_drag_start_pos = None;
                    } else if !pan_drag_wanted && state.tool_mode == ToolMode1d::Zoom {
                        state.active_zoom_drag_start_ppm = Some(screen_x_to_ppm(pos.x, lo, hi));
                        state.active_zoom_drag_start_pos = Some(pos);
                        state.active_roi_drag_start_ppm = None;
                    }
                }
            }
        }

        // Active drag motion
        if resp.dragged() {
            if state.dragging_threshold_bar {
                if let Some(pos) = ptr_pos {
                    let new_int = ((y_base - pos.y) as f64 / y_px_per_unit.max(1e-18))
                        .max(state.estimated_noise_sigma * 1.5);
                    state.picker_min_snr =
                        (new_int / state.estimated_noise_sigma.max(1e-18)).clamp(1.5, 5000.0);
                    state.refresh_active_trace_peaks_and_multiplets();
                }
            } else if state.active_roi_drag_start_ppm.is_none()
                && state.active_zoom_drag_start_ppm.is_none()
                && pan_drag_wanted
            {
                let dx = resp.drag_delta().x;
                if dx.abs() > 0.0 {
                    let d_ppm = (dx / plot_rect.width().max(1.0)) as f64 * (hi - lo);
                    lo += d_ppm;
                    hi += d_ppm;
                    state.view_ppm_min = lo;
                    state.view_ppm_max = hi;
                }
            } else if let (Some(s_pos), Some(cur_pos)) = (state.active_zoom_drag_start_pos, ptr_pos)
            {
                if (cur_pos.x - s_pos.x).abs() >= 6.0 {
                    state.zoom_anchor_ppm = None;
                }
            }
        }

        // Drag completion
        if resp.drag_stopped() {
            state.dragging_threshold_bar = false;
            if let (Some(start_ppm), Some(end_pos)) =
                (state.active_roi_drag_start_ppm.take(), ptr_pos)
            {
                let end_ppm = screen_x_to_ppm(end_pos.x, lo, hi);
                if (end_ppm - start_ppm).abs() >= 1e-3 {
                    let idx = state.roi_regions.len() + 1;
                    state.roi_regions.push(RoiRegion1d::new(
                        format!("region_{idx}"),
                        start_ppm,
                        end_ppm,
                    ));
                    state.status_message = format!(
                        "Added ROI region_{idx} [{:.3}, {:.3}] ppm.",
                        start_ppm.min(end_ppm),
                        start_ppm.max(end_ppm)
                    );
                }
            }
            if let (Some(start_ppm), Some(s_pos), Some(end_pos)) = (
                state.active_zoom_drag_start_ppm.take(),
                state.active_zoom_drag_start_pos.take(),
                ptr_pos,
            ) {
                if (end_pos.x - s_pos.x).abs() >= 6.0 {
                    let end_ppm = screen_x_to_ppm(end_pos.x, lo, hi);
                    if (end_ppm - start_ppm).abs() >= 1e-4 {
                        state.push_zoom_window(start_ppm, end_ppm);
                        lo = state.view_ppm_min.min(state.view_ppm_max);
                        hi = state.view_ppm_min.max(state.view_ppm_max);
                    }
                }
            } else {
                state.active_zoom_drag_start_ppm = None;
                state.active_zoom_drag_start_pos = None;
            }
        }
    }

    let span = (hi - lo).max(1e-6);
    let ppm_to_screen_x =
        |ppm: f64| -> f32 { plot_rect.left() + (((hi - ppm) / span) as f32) * plot_rect.width() };

    let painter = ui.painter_at(canvas_rect);
    painter.rect(
        canvas_rect,
        6.0,
        egui::Color32::from_rgb(12, 16, 24),
        egui::Stroke::new(1.0, egui::Color32::from_rgb(45, 55, 72)),
        egui::StrokeKind::Inside,
    );

    let plot_painter = painter.with_clip_rect(plot_rect);

    // Subtle vertical chemical-shift gridlines
    let tick_step = nice_nmr_tick_step(span, ((plot_rect.width() / 90.0) as usize).max(4));
    let first_tick = (lo / tick_step).ceil() as i64;
    let last_tick = (hi / tick_step).floor() as i64;
    for k in first_tick..=last_tick {
        let p = (k as f64) * tick_step;
        let sx = ppm_to_screen_x(p);
        plot_painter.line_segment(
            [
                egui::pos2(sx, plot_rect.top()),
                egui::pos2(sx, plot_rect.bottom()),
            ],
            egui::Stroke::new(1.0, egui::Color32::from_rgba_unmultiplied(51, 65, 85, 75)),
        );
    }

    // Difference OLS reference window shading when active
    if state.view_mode == Stack1dViewMode::DifferenceSpectrum {
        if let DifferenceAlphaMode::LeastSquaresRegion {
            ppm_min, ppm_max, ..
        } = state.diff_alpha_mode
        {
            let x0 =
                ppm_to_screen_x(ppm_min.max(ppm_max)).clamp(plot_rect.left(), plot_rect.right());
            let x1 =
                ppm_to_screen_x(ppm_min.min(ppm_max)).clamp(plot_rect.left(), plot_rect.right());
            if x1 > x0 {
                plot_painter.rect_filled(
                    egui::Rect::from_min_max(
                        egui::pos2(x0, plot_rect.top()),
                        egui::pos2(x1, plot_rect.bottom()),
                    ),
                    0.0,
                    egui::Color32::from_rgba_unmultiplied(251, 191, 36, 22),
                );
                plot_painter.text(
                    egui::pos2(0.5 * (x0 + x1), plot_rect.top() + 8.0),
                    egui::Align2::CENTER_TOP,
                    "OLS α Ref",
                    egui::FontId::monospace(10.0),
                    egui::Color32::from_rgb(251, 191, 36),
                );
            }
        }
    }

    let ref_idx = state.safe_ref_trace_idx();
    let y_base = plot_rect.top() + plot_rect.height() * state.baseline_y_frac;
    let y_px_per_unit = ((plot_rect.height() as f64) * 0.70 * state.y_gain) / max_int;

    // Render Mode Body
    if state.view_mode == Stack1dViewMode::Pseudo2dHeatmap {
        let tex_w = (plot_rect.width().round() as usize).clamp(64, 2048);
        let tex_h = (plot_rect.height().round() as usize).clamp(32, 1024);
        let key = HeatmapCacheKey {
            width: tex_w,
            height: tex_h,
            view_min_bits: lo.to_bits(),
            view_max_bits: hi.to_bits(),
            floor_bits: state.heatmap_floor_frac.to_bits(),
            ceil_bits: state.heatmap_ceil_frac.to_bits(),
            sort_by_cov: state.heatmap_sort_by_covariate,
            trace_sig: state.trace_signature_hash(),
        };

        if state.heatmap_cache_key.as_ref() != Some(&key) || state.heatmap_texture.is_none() {
            let rgba = synthesize_pseudo2d_rgba(
                &state.traces,
                lo,
                hi,
                tex_w,
                tex_h,
                state.heatmap_floor_frac,
                state.heatmap_ceil_frac,
                state.heatmap_sort_by_covariate,
            );
            let img = egui::ColorImage::from_rgba_unmultiplied([tex_w, tex_h], &rgba);
            if let Some(ref mut handle) = state.heatmap_texture {
                handle.set(img, egui::TextureOptions::LINEAR);
            } else {
                state.heatmap_texture = Some(ui.ctx().load_texture(
                    "stack1d_pseudo2d_heatmap",
                    img,
                    egui::TextureOptions::LINEAR,
                ));
            }
            state.heatmap_cache_key = Some(key);
        }

        if let Some(ref tex) = state.heatmap_texture {
            plot_painter.image(
                tex.id(),
                plot_rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );
        }
    } else {
        // Zero baseline reference line
        plot_painter.line_segment(
            [
                egui::pos2(plot_rect.left(), y_base),
                egui::pos2(plot_rect.right(), y_base),
            ],
            egui::Stroke::new(
                1.0,
                egui::Color32::from_rgba_unmultiplied(100, 116, 139, 95),
            ),
        );

        let vis_indices = state.ordered_visible_trace_indices();
        let max_cols = (plot_rect.width().ceil() as usize).max(64);

        for (stack_ord, &t_idx) in vis_indices.iter().enumerate() {
            let tr = &state.traces[t_idx];
            if state.view_mode == Stack1dViewMode::DifferenceSpectrum
                && state.hide_ref_in_difference
                && t_idx == ref_idx
            {
                continue;
            }

            let (dx_px, dy_px) = if state.view_mode == Stack1dViewMode::StackedOffset {
                let step_v = (state.stack_vertical_step_px as f32) * (stack_ord as f32);
                let tilt_rad = (state.stack_tilt_deg as f32).to_radians();
                (step_v * tilt_rad.tan(), -step_v)
            } else {
                (0.0, 0.0)
            };
            let v_off_px = -(tr.vertical_offset as f32) * plot_rect.height() * 0.5;

            let [r, g, b, a] = tr.color_rgba;
            let color = egui::Color32::from_rgba_unmultiplied(r, g, b, a);

            if state.view_mode == Stack1dViewMode::DifferenceSpectrum {
                let diff = compute_difference_trace(
                    tr,
                    &state.traces[ref_idx],
                    state.diff_alpha_mode,
                    state.manual_alpha,
                );
                if state.show_scaled_ref_overlay && t_idx != ref_idx {
                    let mut scaled_ref = state.traces[ref_idx].clone();
                    scaled_ref.scale *= diff.alpha;
                    let ref_pts = scaled_ref.decimate_for_viewport(lo, hi, max_cols);
                    if ref_pts.len() >= 2 {
                        let screen_ref: Vec<egui::Pos2> = ref_pts
                            .into_iter()
                            .map(|(p, val)| {
                                egui::pos2(
                                    ppm_to_screen_x(p),
                                    y_base + v_off_px - ((val + diff.beta) * y_px_per_unit) as f32,
                                )
                            })
                            .collect();
                        plot_painter.add(egui::Shape::line(
                            screen_ref,
                            egui::Stroke::new(
                                1.0,
                                egui::Color32::from_rgba_unmultiplied(148, 163, 184, 120),
                            ),
                        ));
                    }
                }
                let diff_tr = Trace1d::new(
                    tr.id.clone(),
                    tr.label.clone(),
                    tr.covariate,
                    diff.ppm,
                    diff.diff_intensities,
                );
                let pts = diff_tr.decimate_for_viewport(lo, hi, max_cols);
                if pts.len() >= 2 {
                    let screen_pts: Vec<egui::Pos2> = pts
                        .into_iter()
                        .map(|(p, val)| {
                            egui::pos2(
                                ppm_to_screen_x(p),
                                y_base + v_off_px - (val * y_px_per_unit) as f32,
                            )
                        })
                        .collect();
                    plot_painter.add(egui::Shape::line(
                        screen_pts,
                        egui::Stroke::new(1.35, color),
                    ));
                }
            } else {
                let pts = tr.decimate_for_viewport(lo, hi, max_cols);
                if pts.len() >= 2 {
                    let screen_pts: Vec<egui::Pos2> = pts
                        .into_iter()
                        .map(|(p, val)| {
                            egui::pos2(
                                ppm_to_screen_x(p) + dx_px,
                                y_base + dy_px + v_off_px - (val * y_px_per_unit) as f32,
                            )
                        })
                        .collect();
                    plot_painter.add(egui::Shape::line(
                        screen_pts,
                        egui::Stroke::new(1.35, color),
                    ));
                }
                if state.view_mode == Stack1dViewMode::StackedOffset {
                    plot_painter.text(
                        egui::pos2(
                            (plot_rect.right() - 8.0 + dx_px.min(0.0)).min(plot_rect.right() - 4.0),
                            (y_base + dy_px + v_off_px - 2.0)
                                .clamp(plot_rect.top() + 8.0, plot_rect.bottom() - 4.0),
                        ),
                        egui::Align2::RIGHT_BOTTOM,
                        &tr.label,
                        egui::FontId::monospace(10.0),
                        color,
                    );
                }
            }
        }
    }

    // Draggable horizontal peak-picking SNR threshold line
    if state.show_threshold_bar && state.view_mode != Stack1dViewMode::Pseudo2dHeatmap {
        let thr_y_live = (y_base
            - (state.picker_min_snr * state.estimated_noise_sigma * y_px_per_unit) as f32)
            .clamp(plot_rect.top() + 8.0, plot_rect.bottom() - 8.0);
        let dash_w = 6.0_f32;
        let gap_w = 4.0_f32;
        let mut x_cur = plot_rect.left();
        let thr_col = egui::Color32::from_rgb(251, 146, 60);
        while x_cur < plot_rect.right() {
            let x_end = (x_cur + dash_w).min(plot_rect.right());
            plot_painter.line_segment(
                [egui::pos2(x_cur, thr_y_live), egui::pos2(x_end, thr_y_live)],
                egui::Stroke::new(1.3, thr_col),
            );
            x_cur += dash_w + gap_w;
        }
        plot_painter.text(
            egui::pos2(
                plot_rect.right() - 8.0,
                (thr_y_live - 3.0).max(plot_rect.top() + 12.0),
            ),
            egui::Align2::RIGHT_BOTTOM,
            format!("Threshold SNR ≥ {:.1} (drag ↕)", state.picker_min_snr),
            egui::FontId::monospace(10.0),
            thr_col,
        );
    }

    // Phase pivot vertical marker when `PivotMode1d::AtPpm` is active on Phase & Baseline tab
    if state.selected_inspector_tab == InspectorTab1d::PhaseBaseline
        && state.phase_pivot_mode == PivotMode1d::AtPpm
    {
        let px = ppm_to_screen_x(state.phase_pivot_ppm);
        if px >= plot_rect.left() && px <= plot_rect.right() {
            plot_painter.line_segment(
                [
                    egui::pos2(px, plot_rect.top()),
                    egui::pos2(px, plot_rect.bottom()),
                ],
                egui::Stroke::new(1.1, egui::Color32::from_rgb(192, 132, 252)),
            );
            plot_painter.text(
                egui::pos2(px + 4.0, plot_rect.bottom() - 16.0),
                egui::Align2::LEFT_BOTTOM,
                format!("Pivot {:.3}", state.phase_pivot_ppm),
                egui::FontId::monospace(9.5),
                egui::Color32::from_rgb(216, 180, 254),
            );
        }
    }

    // On-Canvas Picked Peak Leader Ticks & Collision-Avoided Chemical Shift Flags
    if state.show_peak_labels && state.view_mode != Stack1dViewMode::Pseudo2dHeatmap {
        let mut vis_peaks: Vec<(usize, &Peak1dEntry)> = state
            .picked_peaks
            .iter()
            .enumerate()
            .filter(|(_, p)| p.ppm >= lo && p.ppm <= hi)
            .collect();
        // Tallest / selected peaks claim horizontal label slots first when zoomed out
        vis_peaks.sort_by(|&(ia, a), &(ib, b)| {
            let sel_a = state.selected_peak_idx == Some(ia);
            let sel_b = state.selected_peak_idx == Some(ib);
            sel_b
                .cmp(&sel_a)
                .then_with(|| b.height.abs().total_cmp(&a.height.abs()))
        });

        let mut occupied_tiers: [Vec<(f32, f32)>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        let ref_scale = state.traces[ref_idx].scale;
        for (p_idx, pk) in vis_peaks {
            let sx = ppm_to_screen_x(pk.ppm);
            if sx < plot_rect.left() + 14.0 || sx > plot_rect.right() - 14.0 {
                continue;
            }
            let txt = if state.peak_labels_in_hz {
                format!("{:.1}", pk.hz)
            } else if span <= 1.5 {
                format!("{:.4}", pk.ppm)
            } else {
                format!("{:.3}", pk.ppm)
            };
            let half_w = (txt.len() as f32) * 3.4 + 6.0;
            let mut chosen_tier: Option<usize> = None;
            for (t_idx, intervals) in occupied_tiers.iter_mut().enumerate() {
                let overlaps = intervals
                    .iter()
                    .any(|&(a, b)| (sx - half_w) < b && (sx + half_w) > a);
                if !overlaps {
                    intervals.push((sx - half_w, sx + half_w));
                    chosen_tier = Some(t_idx);
                    break;
                }
            }
            let Some(tier) = chosen_tier else {
                continue;
            };

            let apex_y = (y_base - (pk.height * ref_scale * y_px_per_unit) as f32)
                .clamp(plot_rect.top() + 44.0, y_base - 4.0);
            let flag_y = (apex_y - 12.0 - (tier as f32) * 12.0).max(plot_rect.top() + 42.0);
            let is_sel = state.selected_peak_idx == Some(p_idx);
            let (tick_col, txt_col) = if is_sel {
                (
                    egui::Color32::from_rgb(56, 189, 248),
                    egui::Color32::from_rgb(125, 211, 252),
                )
            } else if pk.is_solvent_or_impurity {
                (
                    egui::Color32::from_rgba_unmultiplied(251, 191, 36, 185),
                    egui::Color32::from_rgb(252, 211, 77),
                )
            } else {
                (
                    egui::Color32::from_rgba_unmultiplied(148, 163, 184, 155),
                    egui::Color32::from_rgb(226, 232, 240),
                )
            };

            plot_painter.line_segment(
                [egui::pos2(sx, apex_y - 2.0), egui::pos2(sx, flag_y + 2.0)],
                egui::Stroke::new(if is_sel { 1.5 } else { 1.0 }, tick_col),
            );
            let full_lbl = if pk.is_solvent_or_impurity && !pk.note.is_empty() && span <= 4.0 {
                let short_solv = pk.note.split_whitespace().next().unwrap_or(&pk.note);
                format!("{txt} ({short_solv})")
            } else {
                txt
            };
            plot_painter.text(
                egui::pos2(sx, flag_y),
                egui::Align2::CENTER_BOTTOM,
                full_lbl,
                egui::FontId::monospace(if is_sel { 10.5 } else { 9.5 }),
                txt_col,
            );
        }
    }

    // On-Canvas Multiplet Header Pills, Recursive J-Splitting Trees & Pedestal S-Curve Integrals
    if state.view_mode != Stack1dViewMode::Pseudo2dHeatmap {
        for &(mi, pill_rect) in &multiplet_pill_rects {
            let Some(m) = state.detected_multiplets.get(mi) else {
                continue;
            };
            let is_sel = state.selected_multiplet_idx == Some(mi);
            let id_str = multiplet_letter_id(mi);
            let fill = if is_sel {
                egui::Color32::from_rgba_unmultiplied(14, 116, 144, 220)
            } else {
                egui::Color32::from_rgba_unmultiplied(30, 41, 59, 210)
            };
            let stroke_col = if is_sel {
                egui::Color32::from_rgb(56, 189, 248)
            } else {
                egui::Color32::from_rgb(100, 116, 139)
            };
            plot_painter.rect(
                pill_rect,
                3.0,
                fill,
                egui::Stroke::new(1.0, stroke_col),
                egui::StrokeKind::Inside,
            );
            plot_painter.text(
                pill_rect.center(),
                egui::Align2::CENTER_CENTER,
                format!("{id_str} ({}) {:.2}", m.class.as_str(), m.center_ppm),
                egui::FontId::monospace(9.5),
                egui::Color32::from_rgb(241, 245, 249),
            );
        }

        for (mi, m) in state.detected_multiplets.iter().enumerate() {
            let m_lo = m.range_ppm.0.min(m.range_ppm.1) - 0.012;
            let m_hi = m.range_ppm.0.max(m.range_ppm.1) + 0.012;
            if m_hi < lo || m_lo > hi {
                continue;
            }
            let x0 = ppm_to_screen_x(m_hi).clamp(plot_rect.left(), plot_rect.right());
            let x1 = ppm_to_screen_x(m_lo).clamp(plot_rect.left(), plot_rect.right());

            // Recursive J-coupling splitting tree rungs when zoomed in (`span <= 2.2 ppm` or selected)
            if state.show_j_trees
                && !m.couplings.is_empty()
                && m.tree_levels.len() >= 2
                && (span <= 2.2 || state.selected_multiplet_idx == Some(mi))
            {
                let mhz = m.spectrometer_mhz.max(60.0);
                let top_y = plot_rect.top() + 44.0;
                let step_y = 13.0_f32;
                let tree_col = egui::Color32::from_rgb(192, 132, 252);
                for d in 0..(m.tree_levels.len() - 1) {
                    let parent_lvl = &m.tree_levels[d];
                    let child_lvl = &m.tree_levels[d + 1];
                    let Some(j_hz) = child_lvl.applied_j_hz else {
                        continue;
                    };
                    let n_eq = child_lvl.equivalent_partners.max(1);
                    let y_p = top_y + (d as f32) * step_y;
                    let y_c = top_y + ((d + 1) as f32) * step_y;
                    for &(nu_p, _) in &parent_lvl.nodes_hz {
                        let sx_p = ppm_to_screen_x(nu_p / mhz);
                        for k in 0..=n_eq {
                            let offset_hz = (0.5 * (n_eq as f64) - (k as f64)) * j_hz;
                            let sx_c = ppm_to_screen_x((nu_p + offset_hz) / mhz);
                            plot_painter.line_segment(
                                [egui::pos2(sx_p, y_p), egui::pos2(sx_c, y_c)],
                                egui::Stroke::new(1.15, tree_col),
                            );
                        }
                    }
                    if let Some(&(nu_first, _)) = parent_lvl.nodes_hz.first() {
                        let lbl_x = ppm_to_screen_x(nu_first / mhz);
                        plot_painter.text(
                            egui::pos2(lbl_x, y_p - 1.0),
                            egui::Align2::CENTER_BOTTOM,
                            format!("J={j_hz:.1}Hz"),
                            egui::FontId::monospace(9.0),
                            egui::Color32::from_rgb(233, 213, 255),
                        );
                    }
                }
            }

            // Multiplet S-curve cumulative integral + pedestal numeric proton label (`1.00 H`)
            if state.show_integral_curves && (x1 - x0) >= 8.0 {
                let steps = ((x1 - x0).ceil() as usize).clamp(8, 72);
                let mut cum_pts = Vec::with_capacity(steps + 1);
                let mut running = 0.0_f64;
                let dp = (m_hi - m_lo) / (steps as f64);
                let ref_tr = &state.traces[ref_idx];
                let mut prev_val = ref_tr.sample_at_ppm(m_hi);
                cum_pts.push((m_hi, 0.0_f64));
                for s in 1..=steps {
                    let cur_p = m_hi - (s as f64) * dp;
                    let cur_val = ref_tr.sample_at_ppm(cur_p);
                    running += 0.5 * (prev_val + cur_val) * dp;
                    prev_val = cur_val;
                    cum_pts.push((cur_p, running));
                }
                let max_cum = cum_pts
                    .iter()
                    .fold(0.0_f64, |acc, &(_, c)| acc.max(c.abs()))
                    .max(1e-18);
                let base_int_y = (y_base - plot_rect.height() * 0.14).max(plot_rect.top() + 55.0);
                let int_h = (plot_rect.height() * 0.18).min(64.0);
                let curve_pos: Vec<egui::Pos2> = cum_pts
                    .into_iter()
                    .map(|(p, c)| {
                        egui::pos2(
                            ppm_to_screen_x(p),
                            base_int_y - ((c / max_cum) as f32) * int_h,
                        )
                    })
                    .collect();
                plot_painter.add(egui::Shape::line(
                    curve_pos,
                    egui::Stroke::new(
                        1.15,
                        egui::Color32::from_rgba_unmultiplied(52, 211, 153, 185),
                    ),
                ));

                if let Some(&h_val) = state.detected_multiplet_integrals.get(mi) {
                    let ped_y = (y_base + 5.0).min(plot_rect.bottom() - 13.0);
                    let ped_col = egui::Color32::from_rgb(110, 231, 183);
                    plot_painter.line_segment(
                        [egui::pos2(x0, ped_y - 3.0), egui::pos2(x0, ped_y)],
                        egui::Stroke::new(1.1, ped_col),
                    );
                    plot_painter.line_segment(
                        [egui::pos2(x0, ped_y), egui::pos2(x1, ped_y)],
                        egui::Stroke::new(1.1, ped_col),
                    );
                    plot_painter.line_segment(
                        [egui::pos2(x1, ped_y - 3.0), egui::pos2(x1, ped_y)],
                        egui::Stroke::new(1.1, ped_col),
                    );
                    if (x1 - x0) >= 24.0 {
                        plot_painter.text(
                            egui::pos2(0.5 * (x0 + x1), ped_y + 1.5),
                            egui::Align2::CENTER_TOP,
                            format!("{h_val:.2} H"),
                            egui::FontId::monospace(9.5),
                            ped_col,
                        );
                    }
                }
            }
        }
    }

    // ROI Integral Bracket Overlays + Cumulative Integral Curves on Reference Trace
    for (j, roi) in state.roi_regions.iter().enumerate() {
        let (r_lo, r_hi) = roi.ordered_bounds();
        if r_hi < lo || r_lo > hi {
            continue;
        }
        let x0 = ppm_to_screen_x(r_hi).clamp(plot_rect.left(), plot_rect.right());
        let x1 = ppm_to_screen_x(r_lo).clamp(plot_rect.left(), plot_rect.right());
        if x1 > x0 {
            plot_painter.rect_filled(
                egui::Rect::from_min_max(
                    egui::pos2(x0, plot_rect.top()),
                    egui::pos2(x1, plot_rect.bottom()),
                ),
                0.0,
                egui::Color32::from_rgba_unmultiplied(56, 189, 248, 20),
            );
            let bracket_col = egui::Color32::from_rgb(125, 211, 252);
            let by = plot_rect.bottom() - 4.0;
            plot_painter.line_segment(
                [egui::pos2(x0, by - 5.0), egui::pos2(x0, by)],
                egui::Stroke::new(1.3, bracket_col),
            );
            plot_painter.line_segment(
                [egui::pos2(x0, by), egui::pos2(x1, by)],
                egui::Stroke::new(1.3, bracket_col),
            );
            plot_painter.line_segment(
                [egui::pos2(x1, by - 5.0), egui::pos2(x1, by)],
                egui::Stroke::new(1.3, bracket_col),
            );

            let ref_area = integrate_trace_region_f64(
                &state.traces[ref_idx],
                r_lo,
                r_hi,
                roi.edge_baseline_correction,
            );
            plot_painter.text(
                egui::pos2(0.5 * (x0 + x1), plot_rect.top() + 4.0),
                egui::Align2::CENTER_TOP,
                format!("R{} {}: {:.3}", j + 1, roi.label, ref_area),
                egui::FontId::monospace(10.0),
                bracket_col,
            );
        }
    }

    // Active drag-out ROI rubber-band preview
    if let (Some(start_ppm), Some(cur_pos)) =
        (state.active_roi_drag_start_ppm, resp.interact_pointer_pos())
    {
        let cur_ppm = screen_x_to_ppm(cur_pos.x, lo, hi);
        let x0 = ppm_to_screen_x(start_ppm.max(cur_ppm)).clamp(plot_rect.left(), plot_rect.right());
        let x1 = ppm_to_screen_x(start_ppm.min(cur_ppm)).clamp(plot_rect.left(), plot_rect.right());
        plot_painter.rect(
            egui::Rect::from_min_max(
                egui::pos2(x0, plot_rect.top()),
                egui::pos2(x1, plot_rect.bottom()),
            ),
            2.0,
            egui::Color32::from_rgba_unmultiplied(52, 211, 153, 40),
            egui::Stroke::new(1.5, egui::Color32::from_rgb(52, 211, 153)),
            egui::StrokeKind::Inside,
        );
    }

    // Active 2-Click Zoom Anchor OR Click-Drag Horizontal Zoom Band + Live Δδ / Δν Caliper Pill
    let zoom_band: Option<(f64, Option<f64>, bool)> =
        if let (Some(s_ppm), Some(s_pos), Some(c_pos)) = (
            state.active_zoom_drag_start_ppm,
            state.active_zoom_drag_start_pos,
            resp.interact_pointer_pos(),
        ) {
            if (c_pos.x - s_pos.x).abs() >= 6.0 {
                Some((s_ppm, Some(screen_x_to_ppm(c_pos.x, lo, hi)), true))
            } else {
                state
                    .zoom_anchor_ppm
                    .map(|p1| (p1, state.cursor_ppm, false))
            }
        } else {
            state
                .zoom_anchor_ppm
                .map(|p1| (p1, state.cursor_ppm, false))
        };

    if let Some((p1, maybe_p2, is_drag)) = zoom_band {
        let sx1 = ppm_to_screen_x(p1).clamp(plot_rect.left(), plot_rect.right());
        let z_col = egui::Color32::from_rgb(250, 204, 21);
        plot_painter.line_segment(
            [
                egui::pos2(sx1, plot_rect.top()),
                egui::pos2(sx1, plot_rect.bottom()),
            ],
            egui::Stroke::new(1.8, z_col),
        );
        if let Some(p2) = maybe_p2 {
            let sx2 = ppm_to_screen_x(p2).clamp(plot_rect.left(), plot_rect.right());
            let bx0 = sx1.min(sx2);
            let bx1 = sx1.max(sx2);
            if bx1 > bx0 {
                plot_painter.rect(
                    egui::Rect::from_min_max(
                        egui::pos2(bx0, plot_rect.top()),
                        egui::pos2(bx1, plot_rect.bottom()),
                    ),
                    2.0,
                    egui::Color32::from_rgba_unmultiplied(250, 204, 21, 36),
                    egui::Stroke::new(1.4, z_col),
                    egui::StrokeKind::Inside,
                );
            }
            let dp = (p2 - p1).abs();
            let dhz = dp * state.active_obs_mhz();
            let hint = if is_drag {
                "Release to zoom"
            } else {
                "Click 2nd point to zoom · Esc/Right-click cancel"
            };
            let z_txt = format!(
                "🔍 {p1:.3} -> {p2:.3} ppm  (Δδ = {dp:.4} ppm | Δν = {dhz:.2} Hz)  · {hint}"
            );
            let galley = painter.layout_no_wrap(
                z_txt,
                egui::FontId::monospace(10.5),
                egui::Color32::from_rgb(254, 240, 138),
            );
            let z_rect = egui::Rect::from_min_size(
                egui::pos2(plot_rect.left() + 8.0, plot_rect.top() + 26.0),
                galley.size() + egui::vec2(12.0, 6.0),
            );
            plot_painter.rect(
                z_rect,
                4.0,
                egui::Color32::from_rgba_unmultiplied(23, 23, 34, 238),
                egui::Stroke::new(1.0, z_col),
                egui::StrokeKind::Inside,
            );
            plot_painter.galley(
                z_rect.min + egui::vec2(6.0, 3.0),
                galley,
                egui::Color32::WHITE,
            );
        }
    }

    // Crosshair Cursor & Live δ (ppm) / ν (Hz) / Intensity Readout Overlay
    if let Some(c_ppm) = state.cursor_ppm {
        let cx = ppm_to_screen_x(c_ppm);
        if cx >= plot_rect.left() && cx <= plot_rect.right() {
            plot_painter.line_segment(
                [
                    egui::pos2(cx, plot_rect.top()),
                    egui::pos2(cx, plot_rect.bottom()),
                ],
                egui::Stroke::new(
                    1.0,
                    egui::Color32::from_rgba_unmultiplied(226, 232, 240, 165),
                ),
            );

            let obs_mhz = state.active_obs_mhz();
            let mut hud = format!("δ = {c_ppm:.4} ppm ({:.1} Hz)", c_ppm * obs_mhz);
            let mut shown = 0usize;
            for (idx, tr) in state.traces.iter().enumerate() {
                if !tr.visible {
                    continue;
                }
                let val = if state.view_mode == Stack1dViewMode::DifferenceSpectrum {
                    if state.hide_ref_in_difference && idx == ref_idx {
                        continue;
                    }
                    let (a, b, _) = if let Some(ov) = tr.alpha_override {
                        (ov, 0.0, 0.0)
                    } else {
                        match state.diff_alpha_mode {
                            DifferenceAlphaMode::Manual => (state.manual_alpha, 0.0, 0.0),
                            DifferenceAlphaMode::LeastSquaresRegion {
                                ppm_min,
                                ppm_max,
                                fit_intercept,
                            } => estimate_difference_alpha_ols(
                                tr,
                                &state.traces[ref_idx],
                                ppm_min,
                                ppm_max,
                                fit_intercept,
                            ),
                        }
                    };
                    tr.sample_at_ppm(c_ppm) - (a * state.traces[ref_idx].sample_at_ppm(c_ppm) + b)
                } else {
                    tr.sample_at_ppm(c_ppm)
                };
                if shown < 4 {
                    let _ = write!(hud, " | {}: {:.3}", tr.label, val);
                }
                shown += 1;
            }
            if shown > 4 {
                let _ = write!(hud, " (+{} more)", shown - 4);
            }

            let galley = painter.layout_no_wrap(
                hud,
                egui::FontId::monospace(10.0),
                egui::Color32::from_rgb(241, 245, 249),
            );
            let hud_rect = egui::Rect::from_min_size(
                egui::pos2(plot_rect.left() + 8.0, plot_rect.top() + 4.0),
                galley.size() + egui::vec2(10.0, 4.0),
            );
            plot_painter.rect_filled(
                hud_rect,
                4.0,
                egui::Color32::from_rgba_unmultiplied(15, 23, 42, 225),
            );
            plot_painter.galley(
                hud_rect.min + egui::vec2(5.0, 2.0),
                galley,
                egui::Color32::WHITE,
            );
        }
    }

    // Bottom NMR Chemical Shift Ruler (`high ppm` left -> `low ppm` right)
    painter.rect_stroke(
        plot_rect,
        0.0,
        egui::Stroke::new(1.0, egui::Color32::from_rgb(71, 85, 105)),
        egui::StrokeKind::Inside,
    );
    let axis_y = plot_rect.bottom();
    for k in first_tick..=last_tick {
        let p = (k as f64) * tick_step;
        let sx = ppm_to_screen_x(p);
        if sx >= plot_rect.left() && sx <= plot_rect.right() {
            painter.line_segment(
                [egui::pos2(sx, axis_y), egui::pos2(sx, axis_y + 4.0)],
                egui::Stroke::new(1.0, egui::Color32::from_rgb(148, 163, 184)),
            );
            painter.text(
                egui::pos2(sx, axis_y + 5.0),
                egui::Align2::CENTER_TOP,
                if span <= 0.8 {
                    format!("{p:.3}")
                } else {
                    format!("{p:.2}")
                },
                egui::FontId::monospace(10.0),
                egui::Color32::from_rgb(203, 213, 225),
            );
        }
    }
    painter.text(
        egui::pos2(plot_rect.right() - 2.0, axis_y + 5.0),
        egui::Align2::RIGHT_TOP,
        "δ (ppm)",
        egui::FontId::proportional(10.0),
        egui::Color32::from_rgb(148, 163, 184),
    );

    // 36 px Full-Spectrum Overview Minimap Strip (click or drag to jump anywhere in 1 click)
    let mm_painter = painter.with_clip_rect(minimap_rect);
    mm_painter.rect(
        minimap_rect,
        3.0,
        egui::Color32::from_rgb(17, 24, 39),
        egui::Stroke::new(1.0, egui::Color32::from_rgb(51, 65, 85)),
        egui::StrokeKind::Inside,
    );
    let ref_tr = &state.traces[ref_idx];
    let mm_cols = (minimap_rect.width().ceil() as usize).clamp(48, 512);
    let mm_pts = ref_tr.decimate_for_viewport(g_lo, g_hi, mm_cols);
    let mm_ceil = ref_tr.max_abs_effective().max(1e-12);
    if mm_pts.len() >= 2 {
        let mm_base = minimap_rect.top() + minimap_rect.height() * 0.84;
        let mm_scale = (minimap_rect.height() as f64 * 0.72) / mm_ceil;
        let line_pts: Vec<egui::Pos2> = mm_pts
            .into_iter()
            .map(|(p, v)| {
                let x = minimap_rect.left() + (((g_hi - p) / g_span) as f32) * minimap_rect.width();
                let y = (mm_base - (v * mm_scale) as f32)
                    .clamp(minimap_rect.top() + 2.0, minimap_rect.bottom() - 2.0);
                egui::pos2(x, y)
            })
            .collect();
        mm_painter.add(egui::Shape::line(
            line_pts,
            egui::Stroke::new(1.0, egui::Color32::from_rgb(100, 180, 230)),
        ));
    }

    let vx0 = (minimap_rect.left() + (((g_hi - hi) / g_span) as f32) * minimap_rect.width())
        .clamp(minimap_rect.left(), minimap_rect.right());
    let vx1 = (minimap_rect.left() + (((g_hi - lo) / g_span) as f32) * minimap_rect.width())
        .clamp(minimap_rect.left(), minimap_rect.right());
    let win_x0 = vx0.min(vx1);
    let win_x1 = vx0.max(vx1).max(win_x0 + 4.0);
    if win_x0 > minimap_rect.left() {
        mm_painter.rect_filled(
            egui::Rect::from_min_max(minimap_rect.min, egui::pos2(win_x0, minimap_rect.bottom())),
            0.0,
            egui::Color32::from_rgba_unmultiplied(8, 12, 20, 145),
        );
    }
    if win_x1 < minimap_rect.right() {
        mm_painter.rect_filled(
            egui::Rect::from_min_max(egui::pos2(win_x1, minimap_rect.top()), minimap_rect.max),
            0.0,
            egui::Color32::from_rgba_unmultiplied(8, 12, 20, 145),
        );
    }
    mm_painter.rect(
        egui::Rect::from_min_max(
            egui::pos2(win_x0, minimap_rect.top() + 1.0),
            egui::pos2(win_x1, minimap_rect.bottom() - 1.0),
        ),
        2.0,
        egui::Color32::from_rgba_unmultiplied(56, 189, 248, 32),
        egui::Stroke::new(1.4, egui::Color32::from_rgb(56, 189, 248)),
        egui::StrokeKind::Inside,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_total_area_and_reference_peak_normalization() {
        let n = 2001usize;
        let ppm: Vec<f64> = (0..n).map(|i| 10.0 - (i as f64) * 0.005).collect();

        let mut y1 = vec![0.0_f64; n];
        add_lorentzian(&mut y1, &ppm, 7.20, 0.02, 2.0);
        add_lorentzian(&mut y1, &ppm, 1.50, 0.02, 1.0);

        let y2: Vec<f64> = y1.iter().map(|&v| v * 3.75).collect();
        let y3_locked: Vec<f64> = y1.iter().map(|&v| v * 5.0).collect();

        let t1 = Trace1d::new("s1", "S1", 0.0, ppm.clone(), y1);
        let t2 = Trace1d::new("s2", "S2", 1.0, ppm.clone(), y2);
        let mut t3 = Trace1d::new("s3", "S3_locked", 2.0, ppm.clone(), y3_locked);
        t3.locked = true;

        let mut traces = vec![t1, t2, t3];
        normalize_traces(&mut traces, NormalizationMode::TotalArea);

        let area_1 = integrate_trace_region_f64(&traces[0], 0.0, 10.0, false);
        let area_2 = integrate_trace_region_f64(&traces[1], 0.0, 10.0, false);
        assert!((area_1 - 1.0).abs() < 1e-12, "area_1 = {area_1}");
        assert!((area_2 - 1.0).abs() < 1e-12, "area_2 = {area_2}");
        for idx in 0..n {
            assert!(
                (traces[0].effective_intensity(idx) - traces[1].effective_intensity(idx)).abs()
                    < 1e-12
            );
        }
        // Locked trace must remain strictly untouched (scale == 1.0)
        assert_eq!(traces[2].scale, 1.0);

        // Reference peak-height normalization on [1.30, 1.70] ppm (including negative prior `scale = -2.0`)
        traces[0].scale = -2.0;
        normalize_traces(
            &mut traces,
            NormalizationMode::ReferencePeakHeight {
                ppm_min: 1.30,
                ppm_max: 1.70,
            },
        );
        let apex_1 = traces[0].sample_at_ppm(1.50);
        let apex_2 = traces[1].sample_at_ppm(1.50);
        assert!((apex_1 - 1.0).abs() < 1e-12, "apex_1 = {apex_1}");
        assert!((apex_2 - 1.0).abs() < 1e-12, "apex_2 = {apex_2}");

        // Out-of-bounds empty region preserves prior negative scale untouched
        traces[0].scale = -2.0;
        normalize_traces(
            &mut traces,
            NormalizationMode::ReferencePeakHeight {
                ppm_min: 25.0,
                ppm_max: 30.0,
            },
        );
        assert_eq!(traces[0].scale, -2.0);
        normalize_traces(
            &mut traces,
            NormalizationMode::ReferencePeakHeight {
                ppm_min: 1.30,
                ppm_max: 1.70,
            },
        );
        assert!((traces[0].sample_at_ppm(1.50) - 1.0).abs() < 1e-12);

        // Verify `trace_signature_hash` invalidates when fresh traces of equal length (`gen=1, scale=1`) are swapped in
        let s_a = Stack1dState::with_traces(vec![Trace1d::new(
            "fresh_a",
            "A",
            0.0,
            ppm.clone(),
            vec![1.0; n],
        )]);
        let s_b = Stack1dState::with_traces(vec![Trace1d::new(
            "fresh_b",
            "B",
            0.0,
            ppm.clone(),
            vec![1.0; n],
        )]);
        assert_ne!(s_a.trace_signature_hash(), s_b.trace_signature_hash());

        // Reference region-integral normalization on [1.30, 1.70] ppm
        normalize_traces(
            &mut traces,
            NormalizationMode::ReferenceRegionIntegral {
                ppm_min: 1.30,
                ppm_max: 1.70,
            },
        );
        let ref_int_1 = integrate_trace_region_f64(&traces[0], 1.30, 1.70, false);
        let ref_int_2 = integrate_trace_region_f64(&traces[1], 1.30, 1.70, false);
        assert!((ref_int_1 - 1.0).abs() < 1e-12);
        assert!((ref_int_2 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_difference_spectrum_least_squares_alpha_estimation() {
        let n = 4001usize;
        let ppm: Vec<f64> = (0..n).map(|i| 9.0 - (i as f64) * 0.002).collect();

        // Reference matrix spectrum: large co-solute peak at 1.25 ppm + scaffold peak at 3.50 ppm
        let mut y_ref = vec![0.0_f64; n];
        add_lorentzian(&mut y_ref, &ppm, 1.25, 0.014, 4.0);
        add_lorentzian(&mut y_ref, &ppm, 3.50, 0.016, 2.5);

        // 1. Machine-precision (< 1e-14) OLS recovery when analyte peak has Gaussian support (zero wing in [1.05, 1.45])
        let true_alpha = 0.635_f64;
        let y_gauss_analyte: Vec<f64> = ppm
            .iter()
            .map(|&p| 120.0 * (-((p - 7.45) / 0.012).powi(2)).exp())
            .collect();
        let y_gauss_sample: Vec<f64> = y_ref
            .iter()
            .zip(&y_gauss_analyte)
            .map(|(&r, &a)| true_alpha * r + a)
            .collect();

        let tr_ref = Trace1d::new("ref", "S_ref", 0.0, ppm.clone(), y_ref.clone());
        let tr_gauss_smp = Trace1d::new("gsmp", "S_gauss_sample", 1.0, ppm.clone(), y_gauss_sample);
        let gdiff = compute_difference_trace(
            &tr_gauss_smp,
            &tr_ref,
            DifferenceAlphaMode::LeastSquaresRegion {
                ppm_min: 1.05,
                ppm_max: 1.45,
                fit_intercept: false,
            },
            1.0,
        );
        assert!(
            (gdiff.alpha - true_alpha).abs() < 1e-14,
            "exact OLS alpha = {}, expected {true_alpha}",
            gdiff.alpha
        );
        let gdiff_tr = Trace1d::new("gdiff", "GDiff", 1.0, gdiff.ppm, gdiff.diff_intensities);
        assert!(integrate_trace_abs_region_f64(&gdiff_tr, 1.05, 1.45) < 1e-13);

        // 2. Realistic Lorentzian sample (where algebraic 1/dx^2 Cauchy wing at dx = 6.2 ppm is ~2.2e-5 of 4.0 = 99.9994% cancellation)
        let mut y_sample: Vec<f64> = y_ref.iter().map(|&v| true_alpha * v).collect();
        let mut y_analyte_only = vec![0.0_f64; n];
        add_lorentzian(&mut y_analyte_only, &ppm, 7.45, 0.012, 1.2);
        for (s, &a) in y_sample.iter_mut().zip(&y_analyte_only) {
            *s += a;
        }
        let tr_smp = Trace1d::new("smp", "S_sample", 1.0, ppm.clone(), y_sample);

        let diff = compute_difference_trace(
            &tr_smp,
            &tr_ref,
            DifferenceAlphaMode::LeastSquaresRegion {
                ppm_min: 1.05,
                ppm_max: 1.45,
                fit_intercept: false,
            },
            1.0,
        );
        assert!(
            (diff.alpha - true_alpha).abs() < 1e-6,
            "recovered alpha = {}, expected {true_alpha}",
            diff.alpha
        );

        let diff_tr = Trace1d::new("diff", "Diff", 1.0, diff.ppm, diff.diff_intensities);
        let residual_matrix_area = integrate_trace_abs_region_f64(&diff_tr, 1.05, 1.45);
        assert!(
            residual_matrix_area < 3e-5,
            "matrix residual area = {residual_matrix_area}"
        );

        let analyte_ref = Trace1d::new("ana", "Analyte", 1.0, ppm, y_analyte_only);
        let expected_area = integrate_trace_region_f64(&analyte_ref, 7.20, 7.70, false);
        let diff_analyte_area = integrate_trace_region_f64(&diff_tr, 7.20, 7.70, false);
        assert!(
            (diff_analyte_area - expected_area).abs() < 1e-5,
            "diff_analyte_area = {diff_analyte_area} vs {expected_area}"
        );
        println!(
            "W1_METRIC_JSON:{{\"task\":\"T14.3\",\"kind\":\"t14_3_difference_ols\",\"gauss_alpha_err\":{:.3e},\"lorentz_alpha_err\":{:.3e},\"residual_matrix_area\":{:.3e},\"analyte_area_err\":{:.3e}}}",
            (gdiff.alpha - true_alpha).abs(),
            (diff.alpha - true_alpha).abs(),
            residual_matrix_area,
            (diff_analyte_area - expected_area).abs()
        );
    }

    #[test]
    fn test_fft_interval_regional_and_multi_interval_alignment() {
        // 1. Verify FFT / IFFT roundtrip precision (< 1e-12)
        let mut buf: Vec<(f64, f64)> = (0..64)
            .map(|k| ((0.3 * k as f64).sin(), (0.17 * k as f64).cos()))
            .collect();
        let orig = buf.clone();
        fft_radix2_in_place(&mut buf, false);
        fft_radix2_in_place(&mut buf, true);
        for (a, b) in buf.iter().zip(&orig) {
            assert!((a.0 - b.0).abs() < 1e-12 && (a.1 - b.1).abs() < 1e-12);
        }

        // 2. Multi-interval correlation-shifting alignment with opposite regional pH shifts (+0.032 ppm and -0.024 ppm)
        let n = 4096usize;
        let ppm: Vec<f64> = (0..n)
            .map(|i| 9.0 - (i as f64) * (8.0 / (n - 1) as f64))
            .collect();

        let mut y_ref = vec![0.0_f64; n];
        add_lorentzian(&mut y_ref, &ppm, 7.500, 0.016, 1.5);
        add_lorentzian(&mut y_ref, &ppm, 3.200, 0.016, 1.2);

        let drift_ar = 0.032_f64; // shifted downfield to 7.532 ppm
        let drift_al = -0.024_f64; // shifted upfield to 3.176 ppm
        let mut y_drifted = vec![0.0_f64; n];
        add_lorentzian(&mut y_drifted, &ppm, 7.500 + drift_ar, 0.016, 1.5);
        add_lorentzian(&mut y_drifted, &ppm, 3.200 + drift_al, 0.016, 1.2);

        let mut traces = vec![
            Trace1d::new("ref", "Ref", 0.0, ppm.clone(), y_ref),
            Trace1d::new("tgt", "Drifted", 1.0, ppm.clone(), y_drifted.clone()),
        ];

        let intervals = vec![
            AlignmentInterval::new(7.20, 7.80, 0.08),
            AlignmentInterval::new(2.90, 3.50, 0.08),
        ];

        let recs =
            align_traces_interval_fft(&mut traces, 0, &intervals, AlignmentScope::LocalIntervals);
        assert_eq!(recs.len(), 2);
        let ar_err = (recs[0].shift_ppm - (-drift_ar)).abs();
        let al_err = (recs[1].shift_ppm - (-drift_al)).abs();
        assert!(
            ar_err < 1e-3,
            "Ar shift_ppm = {}, expected {}",
            recs[0].shift_ppm,
            -drift_ar
        );
        assert!(
            al_err < 1e-3,
            "Aliph shift_ppm = {}, expected {}",
            recs[1].shift_ppm,
            -drift_al
        );
        assert!(recs[0].correlation > 0.999);
        assert!(recs[1].correlation > 0.999);

        // Verify aligned peak maxima now sit directly at 7.500 ppm and 3.200 ppm
        assert!(traces[1].sample_at_ppm(7.500) > 0.98 * traces[0].sample_at_ppm(7.500));
        assert!(traces[1].sample_at_ppm(3.200) > 0.98 * traces[0].sample_at_ppm(3.200));

        // 3. Global shift mode from single reference region
        let mut y_global_drift = vec![0.0_f64; n];
        add_lorentzian(&mut y_global_drift, &ppm, 7.500 - 0.0185, 0.016, 1.5);
        add_lorentzian(&mut y_global_drift, &ppm, 3.200 - 0.0185, 0.016, 1.2);
        let mut global_traces = vec![
            traces[0].clone(),
            Trace1d::new("glob", "GlobalDrift", 2.0, ppm, y_global_drift),
        ];
        let grecs = align_traces_interval_fft(
            &mut global_traces,
            0,
            &intervals[..1],
            AlignmentScope::GlobalShiftFromRegion,
        );
        assert_eq!(grecs.len(), 1);
        let glob_err = (global_traces[1].shift_offset_ppm - 0.0185).abs();
        assert!(
            glob_err < 1e-3,
            "shift_offset_ppm = {}",
            global_traces[1].shift_offset_ppm
        );
        println!(
            "W1_METRIC_JSON:{{\"task\":\"T14.3\",\"kind\":\"t14_3_interval_fft_align\",\"ar_shift_err_ppm\":{:.3e},\"aliph_shift_err_ppm\":{:.3e},\"global_shift_err_ppm\":{:.3e},\"min_corr\":{:.6}}}",
            ar_err,
            al_err,
            glob_err,
            recs[0].correlation.min(recs[1].correlation)
        );
    }

    #[test]
    fn test_roi_integration_accuracy_and_tsv_matrix_export() {
        // Exact analytical integration check against f(x) = 2*x + 1 on [1.25, 3.75]:
        // Integral = [x^2 + x]_{1.25}^{3.75} = (14.0625 + 3.75) - (1.5625 + 1.25) = 15.0
        let ppm_asc: Vec<f64> = (0..=500).map(|i| (i as f64) * 0.01).collect();
        let y_lin: Vec<f64> = ppm_asc.iter().map(|&x| 2.0 * x + 1.0).collect();
        let tr_lin = Trace1d::new("lin", "Linear", 0.0, ppm_asc, y_lin);
        let exact_int = integrate_trace_region_f64(&tr_lin, 1.253, 3.747, false);
        let analytic = (3.747_f64.powi(2) + 3.747) - (1.253_f64.powi(2) + 1.253);
        assert!(
            (exact_int - analytic).abs() < 1e-12,
            "exact_int = {exact_int}, analytic = {analytic}"
        );

        // Kinetics ROI matrix & TSV formatting check
        let (traces, rois) = build_kinetics_demo_traces();
        let mut state = Stack1dState::with_traces(traces);
        state.roi_regions = rois;
        let tsv = state.export_roi_tsv();

        let lines: Vec<&str> = tsv.lines().collect();
        assert_eq!(
            lines[0],
            "trace_id\tcovariate\tarea_region_1\tarea_region_2\tnormalized_ratio"
        );
        assert_eq!(lines.len(), 9); // header + 8 timepoints

        let mat = state.evaluate_roi_matrix();
        // At t = 0 min, product conversion ratio ~ 0.0; at t = 100 min, 1 - exp(-4.2) ~ 0.985
        assert!(mat.rows[0].normalized_ratio < 0.01);
        assert!(mat.rows[7].normalized_ratio > 0.97 && mat.rows[7].normalized_ratio <= 1.0);
        for w in mat.rows.windows(2) {
            assert!(
                w[1].normalized_ratio > w[0].normalized_ratio,
                "Product conversion must increase monotonically over time"
            );
        }
        println!(
            "W1_METRIC_JSON:{{\"task\":\"T14.3\",\"kind\":\"t14_3_kinetics_roi_matrix\",\"quadrature_err\":{:.3e},\"n_traces\":{},\"conv_t0\":{:.6},\"conv_t100\":{:.6}}}",
            (exact_int - analytic).abs(),
            mat.rows.len(),
            mat.rows[0].normalized_ratio,
            mat.rows[7].normalized_ratio
        );
    }

    #[test]
    fn test_lod_pyramid_preserves_sharp_apices_across_32x64k_traces() {
        let n_pts = 65_536usize;
        let ppm: Vec<f64> = (0..n_pts)
            .map(|i| 12.0 - (i as f64) * (13.0 / (n_pts - 1) as f64))
            .collect();

        let mut base_y = vec![0.0_f64; n_pts];
        base_y[12_345] = 42.5; // 1-point razor-sharp positive singlet
        base_y[44_321] = -19.25; // 1-point inverted DEPT-135 CH2 singlet

        let traces: Vec<Trace1d> = (0..32)
            .map(|k| {
                Trace1d::new(
                    format!("t{k}"),
                    format!("Trace {k}"),
                    k as f64,
                    ppm.clone(),
                    base_y.clone(),
                )
            })
            .collect();

        for tr in &traces {
            let decimated = tr.decimate_for_viewport(-1.0, 12.0, 512);
            assert!(decimated.len() <= 2050);
            let max_v = decimated
                .iter()
                .fold(f64::NEG_INFINITY, |m, &(_, v)| m.max(v));
            let min_v = decimated.iter().fold(f64::INFINITY, |m, &(_, v)| m.min(v));
            assert!((max_v - 42.5).abs() < 1e-12);
            assert!((min_v - (-19.25)).abs() < 1e-12);
        }

        // Also check Pseudo-2D RGBA synthesis & SVG export hygiene
        let state = Stack1dState::with_traces(traces[..4].to_vec());
        let rgba = synthesize_pseudo2d_rgba(&state.traces, -1.0, 12.0, 256, 64, 0.01, 0.8, false);
        assert_eq!(rgba.len(), 256 * 64 * 4);

        let svg = export_stack1d_svg(&state, 800, 400);
        assert!(svg.starts_with("<svg "));
        assert!(svg.contains("Chemical Shift δ (ppm)"));
    }

    #[test]
    fn test_headless_egui_render_all_4_view_modes() {
        let egui_ctx = egui::Context::default();
        let mut state = Stack1dState::default();

        // 1. Empty state frame
        let _ = egui_ctx.run_ui(egui::RawInput::default(), |ui| {
            render_stack1d_panel(ui, &mut state);
        });

        // 2. Populate kinetics demo and render across all 4 display modes + TSV preview + crosshair
        let (traces, rois) = build_kinetics_demo_traces();
        state.traces = traces;
        state.roi_regions = rois;
        state.fit_full_ppm_bounds();
        state.show_tsv_preview = true;
        state.show_scaled_ref_overlay = true;

        for mode in Stack1dViewMode::ALL {
            state.view_mode = mode;
            state.cursor_ppm = Some(7.22);
            let _ = egui_ctx.run_ui(egui::RawInput::default(), |ui| {
                render_stack1d_panel(ui, &mut state);
            });
        }
        assert!(state.last_exported_tsv.is_some());
        println!(
            "W1_METRIC_JSON:{{\"task\":\"T14.3\",\"kind\":\"t14_3_headless_gui\",\"view_modes_rendered\":{},\"lod_32x64k_ok\":true}}",
            Stack1dViewMode::ALL.len()
        );
    }

    #[test]
    fn test_user_spectrum1d_live_phase_and_multiplet_journal_report() {
        let mut state = Stack1dState::default();
        let (demo_traces, _) = build_kinetics_demo_traces();
        state.traces = demo_traces;
        assert!(state.traces.iter().all(|t| t.is_demo));

        // Build a synthetic 500 MHz 1H Spectrum1D with an aromatic doublet (7.40 ppm, J = 8.0 Hz) + methyl singlet (1.25 ppm)
        let n = 4096usize;
        let axis = AxisPpm::from_bounds("1H", n, 9.0, 0.0, 500.0);
        let ppm: Vec<f64> = (0..n).map(|i| axis.index_to_ppm(i as f64)).collect();
        let mut y = vec![0.0_f64; n];
        let dj = 8.0 / 500.0;
        add_lorentzian(&mut y, &ppm, 7.40 + 0.5 * dj, 0.004, 1.0);
        add_lorentzian(&mut y, &ppm, 7.40 - 0.5 * dj, 0.004, 1.0);
        add_lorentzian(&mut y, &ppm, 1.25, 0.004, 3.0);
        let mut spec = Spectrum1D::new("ibuprofen_test", Experiment1DKind::Proton1D, axis, y, None);
        spec.solvent = Some("CDCl3".to_string());

        state.target_total_protons = 5.0;
        state.load_user_spectra(vec![spec]);

        // Demo traces were replaced cleanly by the real user spectrum
        assert_eq!(state.traces.len(), 1);
        assert!(!state.traces[0].is_demo);
        assert!(state.show_1d_analysis_drawer);
        assert!(state.detected_multiplets.len() >= 2);
        assert!(state.formatted_journal_report.contains("NMR (500 MHz"));

        // Live phase adjustment + auto-phase recovery
        state.phc0_deg = 35.0;
        state.apply_live_phase_to_active_trace();
        state.auto_phase_active_trace();
        state.journal_style = PublicationStyle::Nature;
        state.reformat_journal_report();
        assert!(state.formatted_journal_report.contains("500 MHz"));
    }

    #[test]
    fn test_peak_and_multiplet_click_zoom_and_auto_y_gain() {
        let n = 4096usize;
        let axis = AxisPpm::from_bounds("1H", n, 9.0, 0.0, 500.0);
        let ppm: Vec<f64> = (0..n).map(|i| axis.index_to_ppm(i as f64)).collect();
        let mut y = vec![0.0_f64; n];
        // Giant methyl singlet (area 30) at 1.20 ppm + weak aromatic doublet (area 1 each) at 7.80 ppm
        let dj = 8.0 / 500.0;
        add_lorentzian(&mut y, &ppm, 7.80 + 0.5 * dj, 0.004, 1.0);
        add_lorentzian(&mut y, &ppm, 7.80 - 0.5 * dj, 0.004, 1.0);
        add_lorentzian(&mut y, &ppm, 1.20, 0.004, 30.0);
        let spec = Spectrum1D::new(
            "dynamic_range_1d",
            Experiment1DKind::Proton1D,
            axis,
            y,
            None,
        );

        let mut state = Stack1dState::default();
        state.load_user_spectra(vec![spec]);
        assert!(state.picked_peaks.len() >= 3);
        assert!(state.detected_multiplets.len() >= 2);
        assert!((state.y_gain - 1.0).abs() < 1e-9);

        // 1. Click weak aromatic peak (~7.808 ppm) -> zooms to [pk - 0.12, pk + 0.12] and boosts y_gain >> 10x!
        let ar_pk_idx = state
            .picked_peaks
            .iter()
            .position(|p| (p.ppm - 7.80).abs() < 0.05)
            .expect("aromatic peak picked");
        let ar_ppm = state.picked_peaks[ar_pk_idx].ppm;
        state.zoom_to_peak(ar_pk_idx);
        assert_eq!(state.selected_peak_idx, Some(ar_pk_idx));
        assert!((state.view_ppm_min - (ar_ppm - 0.12)).abs() < 1e-6);
        assert!((state.view_ppm_max - (ar_ppm + 0.12)).abs() < 1e-6);
        assert!(
            state.y_gain > 15.0,
            "expected y_gain boosted for weak aromatic peak, got {}",
            state.y_gain
        );
        assert_eq!(state.zoom_history.len(), 1);

        // 2. Click aromatic multiplet row -> zooms around multiplet range + auto-fits y_gain
        let ar_mult_idx = state
            .detected_multiplets
            .iter()
            .position(|m| (m.center_ppm - 7.80).abs() < 0.05)
            .expect("aromatic multiplet detected");
        state.zoom_to_multiplet(ar_mult_idx);
        assert_eq!(state.selected_multiplet_idx, Some(ar_mult_idx));
        assert!(state.view_ppm_min > 7.60 && state.view_ppm_max < 8.00);
        assert!(state.y_gain > 15.0);

        // Quick 2H integral normalization on aromatic doublet makes aliphatic singlet ~30H
        state.normalize_multiplet_integral_to(ar_mult_idx, 2.0);
        assert!((state.detected_multiplet_integrals[ar_mult_idx] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_two_click_zoom_history_and_undo_redo() {
        let (traces, _) = build_kinetics_demo_traces();
        let mut state = Stack1dState::with_traces(traces);
        let full_lo = state.view_ppm_min;
        let full_hi = state.view_ppm_max;

        // 1st click sets zoom_anchor_ppm without changing view bounds
        assert!(!state.handle_canvas_zoom_click(7.80));
        assert_eq!(state.zoom_anchor_ppm, Some(7.80));
        assert!((state.view_ppm_min - full_lo).abs() < 1e-9);
        assert!((state.view_ppm_max - full_hi).abs() < 1e-9);

        // Esc / cancel clears anchor cleanly
        state.cancel_zoom_anchor();
        assert!(state.zoom_anchor_ppm.is_none());

        // 1st click -> 2nd click zooms to [7.10, 7.80] and records full bounds in zoom_history
        assert!(!state.handle_canvas_zoom_click(7.80));
        assert!(state.handle_canvas_zoom_click(7.10));
        assert!(state.zoom_anchor_ppm.is_none());
        assert!((state.view_ppm_min - 7.10).abs() < 1e-9);
        assert!((state.view_ppm_max - 7.80).abs() < 1e-9);
        assert_eq!(state.zoom_history.len(), 1);

        // Undo restores full bounds; Redo re-applies [7.10, 7.80]
        assert!(state.undo_zoom());
        assert!((state.view_ppm_min - full_lo).abs() < 1e-9);
        assert!((state.view_ppm_max - full_hi).abs() < 1e-9);
        assert!(state.redo_zoom());
        assert!((state.view_ppm_min - 7.10).abs() < 1e-9);
        assert!((state.view_ppm_max - 7.80).abs() < 1e-9);
    }

    #[test]
    fn test_plain_vertical_scroll_scales_y_gain_without_changing_ppm_bounds() {
        let (traces, _) = build_kinetics_demo_traces();
        let mut state = Stack1dState::with_traces(traces);
        state.push_zoom_window(7.10, 7.80);
        let lo_before = state.view_ppm_min;
        let hi_before = state.view_ppm_max;
        let base_before = state.baseline_y_frac;

        // Plain scroll up increases y_gain well past 250x up toward 100,000x while keeping ppm bounds identical
        for _ in 0..40 {
            state.apply_vertical_scroll_gain(120.0, false);
        }
        assert!(
            state.y_gain > 1_000.0 && state.y_gain <= 100_000.0,
            "y_gain = {}",
            state.y_gain
        );
        assert_eq!(state.view_ppm_min, lo_before);
        assert_eq!(state.view_ppm_max, hi_before);
        assert_eq!(state.baseline_y_frac, base_before);

        // Shift + Scroll adjusts baseline_y_frac without changing ppm bounds or y_gain
        let gain_locked = state.y_gain;
        state.apply_vertical_scroll_gain(60.0, true);
        assert!(state.baseline_y_frac < base_before);
        assert_eq!(state.y_gain, gain_locked);
        assert_eq!(state.view_ppm_min, lo_before);
        assert_eq!(state.view_ppm_max, hi_before);
    }

    #[test]
    fn test_single_peak_chemical_shift_referencing_shifts_trace_peaks_and_multiplets() {
        let n = 4096usize;
        let axis = AxisPpm::from_bounds("1H", n, 9.0, -0.5, 500.0);
        let ppm: Vec<f64> = (0..n).map(|i| axis.index_to_ppm(i as f64)).collect();
        let mut y = vec![0.0_f64; n];
        // Uncalibrated TMS sitting at +0.140 ppm and CDCl3 residual at 7.400 ppm (+0.14 ppm offset)
        let dj = 7.5 / 500.0;
        add_lorentzian(&mut y, &ppm, 7.400, 0.004, 1.0);
        add_lorentzian(&mut y, &ppm, 4.260 + 0.5 * dj, 0.004, 2.0);
        add_lorentzian(&mut y, &ppm, 4.260 - 0.5 * dj, 0.004, 2.0);
        add_lorentzian(&mut y, &ppm, 0.140, 0.004, 3.0);
        let mut spec = Spectrum1D::new("uncal_1h", Experiment1DKind::Proton1D, axis, y, None);
        spec.solvent = Some("CDCl3".to_string());

        let mut state = Stack1dState::default();
        state.load_user_spectra(vec![spec]);

        let tms_idx = state
            .picked_peaks
            .iter()
            .position(|p| (p.ppm - 0.140).abs() < 0.01)
            .expect("uncalibrated TMS peak");
        state.reference_selected_peak_to_ppm(tms_idx, 0.000);

        // Verify TMS is now at 0.000 ppm, CDCl3 residual is at 7.260 ppm (annotated as CDCl3!), and doublet is at 4.120 ppm
        let new_tms = state
            .picked_peaks
            .iter()
            .find(|p| p.ppm.abs() < 0.005)
            .expect("referenced TMS at 0.00 ppm");
        assert!(new_tms.is_solvent_or_impurity);

        let cdcl3_pk = state
            .picked_peaks
            .iter()
            .find(|p| (p.ppm - 7.260).abs() < 0.01)
            .expect("CDCl3 peak at 7.26 ppm");
        assert!(cdcl3_pk.is_solvent_or_impurity);
        assert!(cdcl3_pk.note.contains("CDCl3"));

        let dbl = state
            .detected_multiplets
            .iter()
            .find(|m| (m.center_ppm - 4.120).abs() < 0.01)
            .expect("doublet shifted to 4.120 ppm");
        assert_eq!(dbl.class.as_str(), "d");

        // Exercise headless egui rendering across all 4 bottom-dock tabs + threshold bar + J-trees
        let egui_ctx = egui::Context::default();
        state.show_threshold_bar = true;
        state.zoom_anchor_ppm = Some(4.20);
        state.cursor_ppm = Some(4.05);
        for tab in [
            InspectorTab1d::Peaks,
            InspectorTab1d::Multiplets,
            InspectorTab1d::PhaseBaseline,
            InspectorTab1d::MultiTraceStack,
        ] {
            state.selected_inspector_tab = tab;
            let _ = egui_ctx.run_ui(egui::RawInput::default(), |ui| {
                render_stack1d_panel(ui, &mut state);
            });
        }
    }
}
