//! Interactive GUI Dynamics & Ligand Titration Studio (`T13.6`, `T22.6`, `T22.8`) for native
//! desktop and `wasm32-unknown-unknown` WebGPU targets.
//!
//! Renders linked per-residue sequence bar plots (`R1`, `R2`, `hetNOE`, `S^2`, `R_ex`, and
//! `Delta_delta_comb`), exponential relaxation decay curves (`I(t) = I0 exp(-R t)`), multi-`B1`
//! `CW-CEST`, `DANTE-CEST`, and `cos-CEST` saturation dip profiles with Cramér–Rao covariance
//! diagnostics, multi-field `CPMG` dispersion curves, fast-exchange tight-binding quadratic `CSP`
//! `K_d` isotherms, and cross-series 2D titration trajectory tracking & coupled 2D Bloch–McConnell
//! lineshape fitting, with 1-click scalar putty projection onto [`QuantitativeTrackData`]
//! (`mol3d_overlay`).
//!
//! Provenance: Clean-room original interactive visualization widget grounded in:
//! - Lipari, G. & Szabo, A. (*J. Am. Chem. Soc.* 1982, 104, 4546–4559 — model-free `S^2`, `tau_e`, `R_ex`).
//! - McConnell, H. M. (*J. Chem. Phys.* 1958, 28, 430–431), Vallurupalli, P., Bouvignies, G. &
//!   Kay, L. E. (*J. Am. Chem. Soc.* 2012, 134, 8148–8161), Fawzi, N. L. et al. (*Nature* 2011,
//!   480, 268–272), and Higham, N. J. (*SIAM J. Matrix Anal. Appl.* 2005, 26, 1179–1193) — `CEST`/`DEST`.
//! - Yuwen, T., Kay, L. E. & Bouvignies, G. (*ChemPhysChem* 2018, 19, 1707–1710,
//!   DOI: 10.1002/cphc.201800249 — `DANTE-CEST` pulse-train Bloch–McConnell propagator).
//! - Yuwen, T., Bouvignies, G. & Kay, L. E. (*J. Magn. Reson.* 2018, 292, 1–7,
//!   DOI: 10.1016/j.jmr.2018.04.013 — cosine-modulated `cos-CEST` dual-sideband saturation propagator).
//! - Luz, Z. & Meiboom, S. (*J. Chem. Phys.* 1963, 39, 366–370), Carver, J. P. & Richards, R. E.
//!   (*J. Magn. Reson.* 1972, 6, 89–105), Baldwin, A. J. (*J. Magn. Reson.* 2014, 244, 114–124),
//!   and Trott, O. & Palmer, A. G. III (*J. Magn. Reson.* 2002, 154, 157–160) — `CPMG` & `R1rho`.
//! - Williamson, M. P. (*Prog. Nucl. Magn. Reson. Spectrosc.* 2013, 73, 1–16 — amide `CSP`,
//!   iterative `2 sigma` threshold, exchange regimes, and quadratic tight-binding `K_d` isotherms).
//! - Waudby, C. A., Ramos, A., Cabrita, L. D. & Christodoulou, J. (*Sci. Rep.* 2016, 6, 24826,
//!   DOI: 10.1038/srep24826 — coupled 2D Bloch–McConnell HSQC titration lineshape analysis) and Kuhn, H. W. (*Naval Res.
//!   Logist. Quart.* 1955, 2, 83–97) / Munkres, J. (*J. Soc. Indust. Appl. Math.* 1957, 5, 32–38).

use std::collections::BTreeMap;
use std::f64::consts::TAU;

use eframe::egui;
use nmr_core::cest::{
    fit_cest_global_cpu, read_cest_profiles_tsv, simulate_cest_2state_profile, Cest2StateParams,
    CestFitConfig, CestGlobalFitResult, CestNormalization, CestProfileDataset,
};
use nmr_core::cpmg::{
    evaluate_cpmg_model, fit_cpmg_global_cpu, r2eff_baldwin2014, read_cpmg_series_tsv,
    CpmgFitConfig, CpmgGlobalFitResult, CpmgSeriesDataset,
};
use nmr_core::csp::{
    fit_csp_global_kd, fit_naive_michaelis_menten_kd, naive_michaelis_menten_csp,
    quadratic_tight_binding_csp, CspAnalysisConfig, CspGlobalFitResult, ResidueCspTrajectory,
    ResidueTitrationPoint, TrimmedCspThresholds,
};
use nmr_core::multi_cest::{
    deepest_dip_heuristic_seed, fit_multi_cest_grid_lm, fit_multi_cest_lm,
    simulate_multi_cest_profile, MultiCestFitParams, MultiCestFitResult, MultiCestLmConfig,
    MultiCestProfile, MultiCestScheme, MultiCestSimParams,
};
use nmr_core::rng::Rng;
use nmr_core::titration_2d::{
    binding_equilibrium_rates, classify_exchange_regime, fit_titration_2d_lineshape,
    simulate_2d_lineshape_analytical, track_titration_trajectories, Bloch2dKineticState,
    Bloch2dResidueParams, ExchangeRegime, ExchangeRegimeInput, Peak2d, ResidueExchangeRegimeReport,
    ResidueTitration2dSeries, Roi2dGrid, Roi2dPatch, SeriesPeakList, Titration2dFitConfig,
    Titration2dFitResult, TitrationConcentrationPoint, TrackStepStatus, TrajectoryTrackConfig,
    TrajectoryTrackingResult, FAST_EXCHANGE_ALPHA_MAX, SLOW_EXCHANGE_ALPHA_MIN,
};

use crate::mol3d_overlay::{ColormapKind, QuantitativeTrack, QuantitativeTrackData};
use crate::vnmrj_rail::draw_folder_tab;

/// Active analysis view tab inside [`DynamicsStudioPanel`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum DynamicsStudioTab {
    #[default]
    RelaxationModelFree,
    CestDest,
    CpmgDispersion,
    LigandCsp,
}

impl DynamicsStudioTab {
    pub const ALL: [Self; 4] = [
        Self::RelaxationModelFree,
        Self::CestDest,
        Self::CpmgDispersion,
        Self::LigandCsp,
    ];

    pub const fn label(self) -> &'static str {
        match self {
            Self::RelaxationModelFree => "R1 / R2 / hetNOE / S²",
            Self::CestDest => "CEST / DEST Dips",
            Self::CpmgDispersion => "CPMG Dispersion",
            Self::LigandCsp => "Ligand CSP & Kd",
        }
    }
}

/// Irradiation scheme selector inside `Dynamics -> Exchange -> CEST` (`T22.6`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum CestSchemeTab {
    #[default]
    CwCest,
    DanteCest,
    CosCest,
}

impl CestSchemeTab {
    pub const ALL: [Self; 3] = [Self::CwCest, Self::DanteCest, Self::CosCest];

    pub const fn label(self) -> &'static str {
        match self {
            Self::CwCest => "CW-CEST",
            Self::DanteCest => "DANTE-CEST",
            Self::CosCest => "cos-CEST",
        }
    }
}

/// Analysis mode toggle inside `Dynamics -> Exchange -> CSP` (`T22.8`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum CspSubTab {
    #[default]
    FastExchangeCsp,
    Titration2dLineshape,
}

impl CspSubTab {
    pub const ALL: [Self; 2] = [Self::FastExchangeCsp, Self::Titration2dLineshape];

    pub const fn label(self) -> &'static str {
        match self {
            Self::FastExchangeCsp => "Fast-Exchange CSP",
            Self::Titration2dLineshape => "2D Titration Lineshape (T22.8)",
        }
    }
}

/// Labeled 2D curve polyline `(label, [(x, y), ...])`.
pub type NamedCurve2d = (String, Vec<(f64, f64)>);

type GroupedMultiCestTsv = BTreeMap<i64, (f64, Vec<f64>, Vec<f64>, Vec<f64>)>;

/// Persistent non-destructively editable UI state for `T22.6` (`DANTE-CEST` & `cos-CEST`).
#[derive(Clone, Debug)]
pub struct MultiCestUiState {
    pub larmor_mhz: f64,
    pub major_ppm: f64,
    pub b1_hz: f64,
    pub b1_second_hz: f64,
    pub use_dual_b1: bool,
    pub t_ex_s: f64,
    pub n_pulses: u32,
    pub tau_pulse_us: f64,
    pub mod_freq_hz: f64,
    pub n_steps_per_cycle: u32,
    pub b1_inhomogeneity_frac: f64,
    pub n_b1_quad: usize,
    pub normalization: CestNormalization,

    pub demo_p_b: f64,
    pub demo_k_ex_s_inv: f64,
    pub demo_dw_ab_ppm: f64,
    pub demo_r1_s_inv: f64,
    pub demo_r2a_s_inv: f64,
    pub demo_r2b_s_inv: f64,
    pub demo_noise_sigma: f64,
    pub demo_seed: u64,

    pub tsv_input: String,
    pub show_params_editor: bool,

    pub dante_profiles: Vec<MultiCestProfile>,
    pub dante_sim_curves: Vec<NamedCurve2d>,
    pub dante_fit: Option<MultiCestFitResult>,
    pub dante_is_synthetic_demo: bool,

    pub cos_profiles: Vec<MultiCestProfile>,
    pub cos_sim_curves: Vec<NamedCurve2d>,
    pub cos_fit: Option<MultiCestFitResult>,
    pub cos_is_synthetic_demo: bool,

    pub last_status: String,
    pub last_error: Option<String>,
}

impl Default for MultiCestUiState {
    fn default() -> Self {
        Self {
            larmor_mhz: 60.8106,
            major_ppm: 118.0,
            b1_hz: 14.0,
            b1_second_hz: 28.0,
            use_dual_b1: true,
            t_ex_s: 0.40,
            n_pulses: 266,
            tau_pulse_us: 15.0,
            mod_freq_hz: 320.0,
            n_steps_per_cycle: 24,
            b1_inhomogeneity_frac: 0.0,
            n_b1_quad: 1,
            normalization: CestNormalization::ZeroTimePhaseCycled,
            demo_p_b: 0.025,
            demo_k_ex_s_inv: 180.0,
            demo_dw_ab_ppm: 4.2,
            demo_r1_s_inv: 1.3,
            demo_r2a_s_inv: 14.0,
            demo_r2b_s_inv: 22.0,
            demo_noise_sigma: 0.004,
            demo_seed: 220_601,
            tsv_input: String::new(),
            show_params_editor: false,
            dante_profiles: Vec::new(),
            dante_sim_curves: Vec::new(),
            dante_fit: None,
            dante_is_synthetic_demo: false,
            cos_profiles: Vec::new(),
            cos_sim_curves: Vec::new(),
            cos_fit: None,
            cos_is_synthetic_demo: false,
            last_status: String::new(),
            last_error: None,
        }
    }
}

/// Persistent non-destructively editable UI state for `T22.8` (2D titration tracking & lineshape fit).
#[derive(Clone, Debug)]
pub struct Titration2dUiState {
    pub h_larmor_mhz: f64,
    pub protein_conc_um: f64,
    pub max_step_ppm: f64,
    pub n_scale_factor: f64,
    pub demo_kd_um: f64,
    pub demo_k_off_s_inv: f64,
    pub demo_snr: f64,
    pub demo_seed: u64,
    pub initial_kd_hint_um: f64,
    pub initial_k_off_hint_s_inv: f64,

    pub tsv_input: String,
    pub show_params_editor: bool,
    pub selected_residue_idx: usize,
    pub selected_step_idx: usize,

    pub series_steps: Vec<SeriesPeakList>,
    pub tracking_result: Option<TrajectoryTrackingResult>,
    pub regime_reports: Vec<ResidueExchangeRegimeReport>,
    pub roi_series: Vec<ResidueTitration2dSeries>,
    pub lineshape_fit: Option<Titration2dFitResult>,
    pub is_synthetic_demo: bool,

    pub last_status: String,
    pub last_error: Option<String>,
}

impl Default for Titration2dUiState {
    fn default() -> Self {
        Self {
            h_larmor_mhz: 600.0,
            protein_conc_um: 50.0,
            max_step_ppm: 0.45,
            n_scale_factor: 0.14,
            demo_kd_um: 25.0,
            demo_k_off_s_inv: 120.0,
            demo_snr: 50.0,
            demo_seed: 2026_2208,
            initial_kd_hint_um: 25.0,
            initial_k_off_hint_s_inv: 120.0,
            tsv_input: String::new(),
            show_params_editor: false,
            selected_residue_idx: 0,
            selected_step_idx: 2,
            series_steps: Vec::new(),
            tracking_result: None,
            regime_reports: Vec::new(),
            roi_series: Vec::new(),
            lineshape_fit: None,
            is_synthetic_demo: false,
            last_status: String::new(),
            last_error: None,
        }
    }
}

/// Sequence bar metric selectable in the Relaxation & Model-Free view.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum SequenceMetricKind {
    R1,
    R2,
    HetNoe,
    #[default]
    OrderParameterS2,
    ExchangeRex,
    LigandCsp,
}

impl SequenceMetricKind {
    pub const RELAX_METRICS: [Self; 5] = [
        Self::R1,
        Self::R2,
        Self::HetNoe,
        Self::OrderParameterS2,
        Self::ExchangeRex,
    ];

    pub const fn label(self) -> &'static str {
        match self {
            Self::R1 => "R1 (s⁻¹)",
            Self::R2 => "R2 (s⁻¹)",
            Self::HetNoe => "{¹H}-¹⁵N NOE",
            Self::OrderParameterS2 => "Lipari–Szabo S²",
            Self::ExchangeRex => "R_ex (s⁻¹)",
            Self::LigandCsp => "Δδ_comb (ppm)",
        }
    }
}

/// Single observed `(delay_s, intensity, sigma)` point on an `R1` or `R2` exponential decay curve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecayCurvePoint {
    pub delay_s: f64,
    pub intensity: f64,
    pub uncertainty: f64,
}

/// Per-residue relaxation rates (`R1`, `R2`, `hetNOE`), Lipari–Szabo model-free parameters
/// (`S^2`, `R_ex`), and exponential decay series (`I(t) = I0 * exp(-R * t)`) for display in
/// [`DynamicsStudioPanel`].
#[derive(Clone, Debug, PartialEq)]
pub struct ResidueRelaxationRecord {
    pub seq_id: i32,
    pub res_name: String,
    pub r1_s_inv: Option<(f64, f64)>,
    pub r2_s_inv: Option<(f64, f64)>,
    pub het_noe: Option<(f64, f64)>,
    pub s2: Option<(f64, f64)>,
    pub rex_s_inv: Option<(f64, f64)>,
    pub r1_i0: f64,
    pub r1_decay: Vec<DecayCurvePoint>,
    pub r2_i0: f64,
    pub r2_decay: Vec<DecayCurvePoint>,
}

impl ResidueRelaxationRecord {
    pub fn metric_value_err(&self, metric: SequenceMetricKind) -> Option<(f64, f64)> {
        match metric {
            SequenceMetricKind::R1 => self.r1_s_inv,
            SequenceMetricKind::R2 => self.r2_s_inv,
            SequenceMetricKind::HetNoe => self.het_noe,
            SequenceMetricKind::OrderParameterS2 => self.s2,
            SequenceMetricKind::ExchangeRex => self.rex_s_inv,
            SequenceMetricKind::LigandCsp => None,
        }
    }
}

/// One-click quantitative 3D structure overlay payload emitted by [`DynamicsStudioPanel`] ready to
/// project `S^2`, `R_ex`, or `Delta_delta_comb` (`CSP`) onto [`QuantitativeTrackData`].
#[derive(Clone, Debug, PartialEq)]
pub struct QuantitativeOverlayProjection {
    pub track: QuantitativeTrack,
    pub colormap: ColormapKind,
    pub putty_enabled: bool,
    /// Map from `seq_id` (e.g. `1..=N` NMR-STAR sequence numbering) to quantitative scalar value.
    pub values_by_seq_id: BTreeMap<i32, f64>,
    /// Suggested upper display clamp (`s^-1` for `R_ex`, `ppm` for `CSP`, `1.0` for `S^2`).
    pub display_max: f64,
}

impl QuantitativeOverlayProjection {
    /// Apply this projection to a [`QuantitativeTrackData`] instance using `seq_ids` (`0`-based
    /// polymer residue index -> `seq_id`). If `seq_ids` is empty, assumes canonical `1`-based
    /// numbering (`res_idx = (seq_id - 1) as usize` for `seq_id >= 1`).
    pub fn apply_to(&self, target: &mut QuantitativeTrackData, seq_ids: &[i32]) {
        let mut seq_to_idx = BTreeMap::new();
        for (idx, &sid) in seq_ids.iter().enumerate() {
            seq_to_idx.insert(sid, idx);
        }

        let resolve_idx = |sid: i32| -> Option<usize> {
            if !seq_to_idx.is_empty() {
                seq_to_idx.get(&sid).copied()
            } else if sid >= 1 {
                Some((sid - 1) as usize)
            } else {
                None
            }
        };

        target.active_track = self.track;
        target.colormap = self.colormap;
        target.putty_enabled = self.putty_enabled;

        match self.track {
            QuantitativeTrack::LipariSzaboS2 => {
                target.lipari_szabo_s2.clear();
                for (&sid, &val) in &self.values_by_seq_id {
                    if let Some(idx) = resolve_idx(sid) {
                        target.lipari_szabo_s2.insert(idx, val.clamp(0.0, 1.0));
                    }
                }
            }
            QuantitativeTrack::ChemicalExchangeRex => {
                target.rex_s_inv.clear();
                for (&sid, &val) in &self.values_by_seq_id {
                    if let Some(idx) = resolve_idx(sid) {
                        target.rex_s_inv.insert(idx, val.max(0.0));
                    }
                }
                if self.display_max > 0.0 {
                    target.rex_max_s_inv = self.display_max;
                }
            }
            QuantitativeTrack::LigandCsp => {
                target.csp_ppm.clear();
                for (&sid, &val) in &self.values_by_seq_id {
                    if let Some(idx) = resolve_idx(sid) {
                        target.csp_ppm.insert(idx, val.max(0.0));
                    }
                }
                if self.display_max > 0.0 {
                    target.csp_max_ppm = self.display_max;
                }
            }
            _ => {}
        }
    }
}

/// Frame interaction result returned by [`DynamicsStudioPanel::show`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DynamicsStudioOutput {
    /// Newly clicked residue `seq_id` in any sequence bar plot or residue selector.
    pub clicked_seq_id: Option<i32>,
    /// One-click 3D structure putty projection triggered this frame.
    pub overlay_projection: Option<QuantitativeOverlayProjection>,
}

#[derive(Clone, Debug)]
struct SequenceBarEntry {
    seq_id: i32,
    res_name: String,
    value: f64,
    err: f64,
    highlight: bool,
}

/// Self-contained `egui` Dynamics & Ligand Titration Studio (`desktop` + `wasm32`).
#[derive(Clone, Debug)]
pub struct DynamicsStudioPanel {
    pub active_tab: DynamicsStudioTab,
    pub active_relax_metric: SequenceMetricKind,
    pub selected_seq_id: Option<i32>,
    pub show_r2_decay: bool,
    pub show_naive_mm_comparison: bool,
    pub putty_enabled: bool,
    pub overlay_colormap: ColormapKind,

    pub relaxation_records: Vec<ResidueRelaxationRecord>,
    pub cest_profiles: Vec<CestProfileDataset>,
    pub cest_fit: Option<CestGlobalFitResult>,
    pub cest_scheme_tab: CestSchemeTab,
    pub multi_cest: MultiCestUiState,
    pub cpmg_series: Vec<CpmgSeriesDataset>,
    pub cpmg_fit: Option<CpmgGlobalFitResult>,
    pub csp_trajectories: Vec<ResidueCspTrajectory>,
    pub csp_fit: Option<CspGlobalFitResult>,
    pub csp_threshold: Option<TrimmedCspThresholds>,
    pub csp_sub_tab: CspSubTab,
    pub titration_2d: Titration2dUiState,
    pub last_canvas_rect: Option<egui::Rect>,
    pub last_plot_rect: Option<egui::Rect>,
}

impl Default for DynamicsStudioPanel {
    fn default() -> Self {
        Self {
            active_tab: DynamicsStudioTab::RelaxationModelFree,
            active_relax_metric: SequenceMetricKind::OrderParameterS2,
            selected_seq_id: None,
            show_r2_decay: false,
            show_naive_mm_comparison: true,
            putty_enabled: true,
            overlay_colormap: ColormapKind::Cividis,
            relaxation_records: Vec::new(),
            cest_profiles: Vec::new(),
            cest_fit: None,
            cest_scheme_tab: CestSchemeTab::CwCest,
            multi_cest: MultiCestUiState::default(),
            cpmg_series: Vec::new(),
            cpmg_fit: None,
            csp_trajectories: Vec::new(),
            csp_fit: None,
            csp_threshold: None,
            csp_sub_tab: CspSubTab::FastExchangeCsp,
            titration_2d: Titration2dUiState::default(),
            last_canvas_rect: None,
            last_plot_rect: None,
        }
    }
}

impl DynamicsStudioPanel {
    pub fn new() -> Self {
        Self::default()
    }

    /// Populate per-residue `R1 / R2 / hetNOE / S^2 / R_ex` records and select the first residue if none is active.
    pub fn set_relaxation_records(&mut self, mut records: Vec<ResidueRelaxationRecord>) {
        records.sort_by_key(|r| r.seq_id);
        if self.selected_seq_id.is_none() {
            self.selected_seq_id = records.first().map(|r| r.seq_id);
        }
        self.relaxation_records = records;
    }

    /// Populate multi-`B1` `CEST` / `DEST` saturation profiles and optional global fit result.
    pub fn set_cest_data(
        &mut self,
        profiles: Vec<CestProfileDataset>,
        fit: Option<CestGlobalFitResult>,
    ) {
        if self.selected_seq_id.is_none() {
            self.selected_seq_id = profiles.first().map(|p| p.seq_id);
        }
        self.cest_profiles = profiles;
        self.cest_fit = fit;
    }

    /// Populate multi-field `CPMG` dispersion series and optional global fit result.
    pub fn set_cpmg_data(
        &mut self,
        series: Vec<CpmgSeriesDataset>,
        fit: Option<CpmgGlobalFitResult>,
    ) {
        if self.selected_seq_id.is_none() {
            self.selected_seq_id = series.first().map(|s| s.seq_id);
        }
        self.cpmg_series = series;
        self.cpmg_fit = fit;
    }

    /// Populate ligand titration `CSP` trajectories, Williamson `2 sigma` threshold, and global `K_d` fit.
    pub fn set_csp_data(
        &mut self,
        trajectories: Vec<ResidueCspTrajectory>,
        fit: Option<CspGlobalFitResult>,
        threshold: Option<TrimmedCspThresholds>,
    ) {
        if self.selected_seq_id.is_none() {
            self.selected_seq_id = trajectories.first().map(|t| t.seq_id);
        }
        let resolved_thresh = threshold.or_else(|| fit.as_ref().map(|f| f.thresholds.clone()));
        self.csp_trajectories = trajectories;
        self.csp_fit = fit;
        self.csp_threshold = resolved_thresh;
    }

    /// Return `true` if `text` matches a `CEST`/`DEST` profile TSV, `CPMG` series TSV, `CSP`
    /// tracked trajectory TSV, or `T22.8` 2D titration peak-series TSV supported by
    /// [`Self::import_exchange_or_csp_tsv`].
    pub fn looks_like_exchange_or_csp_tsv(text: &str) -> bool {
        let trimmed = text.trim_start();
        trimmed.starts_with("# nmrassign_cest_profiles")
            || trimmed.starts_with("# nmrassign_cpmg_series")
            || trimmed.starts_with("# nmrassign_csp_trajectories")
            || trimmed.starts_with("# nmrassign_titration_2d")
            || trimmed.contains("seq_id\tres_name\tlarmor_mhz\tt_ex_s\tb1_hz\t")
            || trimmed.contains("seq_id\tres_name\tlarmor_mhz\tt_relax_s\tnu_cpmg_hz\t")
            || trimmed.contains("seq_index\tseq_id\tres_name\tprotein_conc_um\tligand_conc_um\t")
            || trimmed
                .contains("step_index\tstep_label\tligand_conc_um\tprotein_conc_um\tpeak_id\t")
    }

    /// Parse and fit a pasted or dropped `CEST`/`DEST` profile TSV, `CPMG` series TSV, or `CSP`
    /// trajectory TSV (`desktop` + `wasm32` compatible).
    pub fn import_exchange_or_csp_tsv(&mut self, text: &str) -> Result<String, String> {
        let trimmed = text.trim();
        if trimmed.starts_with("# nmrassign_cest_profiles")
            || trimmed.contains("seq_id\tres_name\tlarmor_mhz\tt_ex_s\tb1_hz\t")
        {
            let profiles = read_cest_profiles_tsv(trimmed)?;
            if profiles.is_empty() {
                return Err("no CEST/DEST profiles found in TSV".to_string());
            }
            let cfg = CestFitConfig {
                max_lm_iters: 35,
                compute_chi2_surface: false,
                ..CestFitConfig::default()
            };
            let fit = fit_cest_global_cpu(&profiles, &cfg)?;
            let summary = format!(
                "Loaded {} CEST profiles ({} residues): p_B = {:.2}% ± {:.2}%, k_ex = {:.1} ± {:.1} s⁻¹",
                profiles.len(),
                fit.residues.len(),
                fit.p_b * 100.0,
                fit.p_b_err * 100.0,
                fit.k_ex_s_inv,
                fit.k_ex_err,
            );
            self.active_tab = DynamicsStudioTab::CestDest;
            self.selected_seq_id = profiles.first().map(|p| p.seq_id);
            self.set_cest_data(profiles, Some(fit));
            return Ok(summary);
        }

        if trimmed.starts_with("# nmrassign_cpmg_series")
            || trimmed.contains("seq_id\tres_name\tlarmor_mhz\tt_relax_s\tnu_cpmg_hz\t")
        {
            let series = read_cpmg_series_tsv(trimmed)?;
            if series.is_empty() {
                return Err("no CPMG series found in TSV".to_string());
            }
            let fit = fit_cpmg_global_cpu(&series, &CpmgFitConfig::default())?;
            let summary = format!(
                "Loaded {} CPMG curves ({} exchanging residues): p_B = {:.2}% ± {:.2}%, k_ex = {:.1} ± {:.1} s⁻¹",
                series.len(),
                fit.residues.len(),
                fit.p_b * 100.0,
                fit.p_b_err * 100.0,
                fit.k_ex_s_inv,
                fit.k_ex_err,
            );
            self.active_tab = DynamicsStudioTab::CpmgDispersion;
            self.selected_seq_id = fit
                .residues
                .first()
                .map(|r| r.seq_id)
                .or_else(|| series.first().map(|s| s.seq_id));
            self.set_cpmg_data(series, Some(fit));
            return Ok(summary);
        }

        if trimmed.starts_with("# nmrassign_csp_trajectories")
            || trimmed.contains("seq_index\tseq_id\tres_name\tprotein_conc_um\tligand_conc_um\t")
        {
            let mut by_res: BTreeMap<(usize, i32), ResidueCspTrajectory> = BTreeMap::new();
            for (line_no, raw_line) in trimmed.lines().enumerate() {
                let line = raw_line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                let cols: Vec<&str> = line.split('\t').collect();
                if cols[0] == "seq_index" {
                    continue;
                }
                if cols.len() < 8 {
                    return Err(format!(
                        "line {}: expected 8 columns in CSP trajectory TSV, got {}",
                        line_no + 1,
                        cols.len()
                    ));
                }
                let seq_index: usize = cols[0]
                    .parse()
                    .map_err(|e| format!("line {}: invalid seq_index: {e}", line_no + 1))?;
                let seq_id: i32 = cols[1]
                    .parse()
                    .map_err(|e| format!("line {}: invalid seq_id: {e}", line_no + 1))?;
                let res_name = cols[2].trim().to_string();
                let protein_conc_um: f64 = cols[3]
                    .parse()
                    .map_err(|e| format!("line {}: invalid protein_conc_um: {e}", line_no + 1))?;
                let ligand_conc_um: f64 = cols[4]
                    .parse()
                    .map_err(|e| format!("line {}: invalid ligand_conc_um: {e}", line_no + 1))?;
                let matched_peak =
                    if cols[5].eq_ignore_ascii_case("na") || cols[6].eq_ignore_ascii_case("na") {
                        None
                    } else {
                        let h: f64 = cols[5]
                            .parse()
                            .map_err(|e| format!("line {}: invalid h_ppm: {e}", line_no + 1))?;
                        let n: f64 = cols[6]
                            .parse()
                            .map_err(|e| format!("line {}: invalid n_ppm: {e}", line_no + 1))?;
                        Some((h, n))
                    };
                let csp_ppm = if cols[7].eq_ignore_ascii_case("na") {
                    None
                } else {
                    Some(
                        cols[7]
                            .parse::<f64>()
                            .map_err(|e| format!("line {}: invalid csp_ppm: {e}", line_no + 1))?,
                    )
                };
                let entry =
                    by_res
                        .entry((seq_index, seq_id))
                        .or_insert_with(|| ResidueCspTrajectory {
                            seq_index,
                            seq_id,
                            res_name,
                            apo_h_ppm: matched_peak.map(|p| p.0).unwrap_or(0.0),
                            apo_n_ppm: matched_peak.map(|p| p.1).unwrap_or(0.0),
                            points: Vec::new(),
                            max_observed_csp_ppm: 0.0,
                        });
                if ligand_conc_um <= 1e-12 {
                    if let Some((h, n)) = matched_peak {
                        entry.apo_h_ppm = h;
                        entry.apo_n_ppm = n;
                    }
                }
                if let Some(c) = csp_ppm {
                    entry.max_observed_csp_ppm = entry.max_observed_csp_ppm.max(c);
                }
                entry.points.push(ResidueTitrationPoint {
                    protein_conc_um,
                    ligand_conc_um,
                    matched_peak,
                    csp_ppm,
                });
            }
            let trajectories: Vec<ResidueCspTrajectory> = by_res.into_values().collect();
            let cfg = CspAnalysisConfig {
                n_mc: 120,
                ..CspAnalysisConfig::default()
            };
            let fit = fit_csp_global_kd(&trajectories, &cfg)?;
            let summary = format!(
                "Loaded {} CSP trajectories ({} >=2σ pocket sites): Kd = {:.2} ± {:.2} µM",
                trajectories.len(),
                fit.thresholds.significant_seq_ids.len(),
                fit.global_kd_um,
                fit.global_kd_sd_um,
            );
            self.active_tab = DynamicsStudioTab::LigandCsp;
            self.csp_sub_tab = CspSubTab::FastExchangeCsp;
            self.selected_seq_id = fit
                .thresholds
                .significant_seq_ids
                .first()
                .copied()
                .or_else(|| trajectories.first().map(|t| t.seq_id));
            self.set_csp_data(trajectories, Some(fit), None);
            return Ok(summary);
        }

        if trimmed.starts_with("# nmrassign_titration_2d")
            || trimmed
                .contains("step_index\tstep_label\tligand_conc_um\tprotein_conc_um\tpeak_id\t")
        {
            self.active_tab = DynamicsStudioTab::LigandCsp;
            self.csp_sub_tab = CspSubTab::Titration2dLineshape;
            return self.import_titration_2d_tsv(trimmed);
        }

        Err("unrecognized exchange/CSP TSV header".to_string())
    }

    /// Populate synthetic dual-`B1` `CEST`, dual-field `Baldwin2014` `CPMG`, and tight-binding
    /// `CSP` `K_d` datasets aligned with `sequence`/`seq_ids` when empty (`desktop` + `wasm32`).
    pub fn ensure_demo_exchange_and_csp(
        &mut self,
        sequence: &[String],
        seq_ids: &[i32],
    ) -> Result<String, String> {
        let default_seq = [
            "MET", "GLN", "ILE", "PHE", "VAL", "LYS", "THR", "LEU", "THR", "GLY", "LYS", "THR",
            "ILE", "THR", "LEU", "GLU", "VAL", "GLU", "PRO", "SER", "ASP", "THR", "ILE", "GLU",
        ];
        let n_res = if sequence.is_empty() {
            default_seq.len()
        } else {
            sequence.len().min(36)
        };
        let res_at = |k: usize| -> (i32, String) {
            let rn = sequence
                .get(k)
                .cloned()
                .unwrap_or_else(|| default_seq[k % default_seq.len()].to_string());
            let sid = seq_ids.get(k).copied().unwrap_or((k + 1) as i32);
            (sid, rn)
        };

        let ex_indices = [
            2usize.min(n_res.saturating_sub(1)),
            6usize.min(n_res.saturating_sub(1)),
            12usize.min(n_res.saturating_sub(1)),
            (n_res * 3 / 4).min(n_res.saturating_sub(1)),
        ];

        if self.cest_profiles.is_empty() {
            let offsets: Vec<f64> = (0..=48).map(|j| 108.0 + 0.45 * (j as f64)).collect();
            let mut profs = Vec::new();
            for (ei, &k) in ex_indices.iter().enumerate() {
                let (sid, rn) = res_at(k);
                let major = 114.5 + 2.6 * (ei as f64);
                let dw = if ei % 2 == 0 {
                    4.8 + 0.5 * (ei as f64)
                } else {
                    -4.4
                };
                for &b1_hz in &[15.0, 30.0] {
                    let p = Cest2StateParams {
                        p_b: 0.015,
                        k_ex_s_inv: 150.0,
                        major_ppm: major,
                        dw_ab_ppm: dw,
                        r1a_s_inv: 1.40,
                        r2a_s_inv: 14.0,
                        r1b_s_inv: 1.40,
                        r2b_s_inv: 19.0,
                        larmor_mhz: 60.8106,
                        t_ex_s: 0.40,
                        b1_hz,
                        b1_inhomogeneity_frac: 0.0,
                        n_b1_quad: 1,
                        normalization: CestNormalization::ZeroTimePhaseCycled,
                    };
                    let y = simulate_cest_2state_profile(&p, &offsets)?;
                    profs.push(CestProfileDataset {
                        seq_id: sid,
                        res_name: rn.clone(),
                        larmor_mhz: 60.8106,
                        t_ex_s: 0.40,
                        b1_hz,
                        b1_inhomogeneity_frac: 0.0,
                        normalization: CestNormalization::ZeroTimePhaseCycled,
                        major_ppm_hint: Some(major),
                        offsets_ppm: offsets.clone(),
                        intensities: y,
                        uncertainties: vec![0.003; offsets.len()],
                    });
                }
            }
            let cfg = CestFitConfig {
                n_b1_quad: 1,
                max_lm_iters: 25,
                compute_chi2_surface: false,
                ..CestFitConfig::default()
            };
            let fit = fit_cest_global_cpu(&profs, &cfg)?;
            self.set_cest_data(profs, Some(fit));
        }

        if self.cpmg_series.is_empty() {
            let nus = [
                50.0, 100.0, 150.0, 200.0, 300.0, 400.0, 600.0, 800.0, 1000.0,
            ];
            let mut series = Vec::new();
            for (ei, &k) in ex_indices.iter().enumerate() {
                let (sid, rn) = res_at(k);
                let dw_ppm = 2.8 + 0.6 * (ei as f64);
                for &(larmor_mhz, r20) in &[(60.8106, 11.8), (81.0808, 14.2)] {
                    let dw_rad = TAU * larmor_mhz * dw_ppm;
                    let r2_vals: Vec<f64> = nus
                        .iter()
                        .map(|&nu| r2eff_baldwin2014(r20, r20, 0.035, 900.0, dw_rad, 0.040, nu))
                        .collect();
                    series.push(CpmgSeriesDataset {
                        seq_id: sid,
                        res_name: rn.clone(),
                        larmor_mhz,
                        t_relax_s: 0.040,
                        nu_cpmg_hz: nus.to_vec(),
                        r2_eff_s_inv: r2_vals,
                        r2_eff_err: vec![0.12; nus.len()],
                    });
                }
            }
            let fit = fit_cpmg_global_cpu(&series, &CpmgFitConfig::default())?;
            self.set_cpmg_data(series, Some(fit));
        }

        if self.csp_trajectories.is_empty() {
            let ligands = [0.0, 20.0, 45.0, 75.0, 110.0, 160.0, 250.0, 450.0];
            let mut trajs = Vec::with_capacity(n_res);
            for k in 0..n_res {
                let (sid, rn) = res_at(k);
                let is_pocket = ex_indices.contains(&k);
                let dmax = if is_pocket {
                    0.26 + 0.05 * ((k % 3) as f64)
                } else {
                    0.012 + 0.004 * ((k % 4) as f64)
                };
                let apo_h = 7.60 + 0.04 * (k as f64);
                let apo_n = 115.0 + 0.45 * (k as f64);
                let mut max_c = 0.0f64;
                let pts: Vec<ResidueTitrationPoint> = ligands
                    .iter()
                    .map(|&l| {
                        let c = quadratic_tight_binding_csp(dmax, 100.0, l, 8.0);
                        max_c = max_c.max(c);
                        ResidueTitrationPoint {
                            protein_conc_um: 100.0,
                            ligand_conc_um: l,
                            matched_peak: Some((apo_h + c, apo_n)),
                            csp_ppm: Some(c),
                        }
                    })
                    .collect();
                trajs.push(ResidueCspTrajectory {
                    seq_index: k,
                    seq_id: sid,
                    res_name: rn,
                    apo_h_ppm: apo_h,
                    apo_n_ppm: apo_n,
                    points: pts,
                    max_observed_csp_ppm: max_c,
                });
            }
            let cfg = CspAnalysisConfig {
                n_mc: 60,
                ..CspAnalysisConfig::default()
            };
            let fit = fit_csp_global_kd(&trajs, &cfg)?;
            self.set_csp_data(trajs, Some(fit), None);
        }

        let cest_pb = self.cest_fit.as_ref().map(|f| f.p_b * 100.0).unwrap_or(0.0);
        let cpmg_kex = self.cpmg_fit.as_ref().map(|f| f.k_ex_s_inv).unwrap_or(0.0);
        let csp_kd = self.csp_fit.as_ref().map(|f| f.global_kd_um).unwrap_or(0.0);
        Ok(format!(
            "Dynamics Studio ready: CEST p_B = {cest_pb:.2}%, CPMG k_ex = {cpmg_kex:.0} s⁻¹, CSP tight-binding Kd = {csp_kd:.2} µM"
        ))
    }

    // =========================================================================
    // T22.6: DANTE-CEST & cos-CEST Multi-Frequency Simulation & 2-State Fitting
    // =========================================================================

    /// Build a validated [`MultiCestScheme`] from the current non-destructively editable
    /// `self.multi_cest` parameters for `scheme_tab` and effective RF field `b1_eff_hz`.
    pub fn build_multi_cest_scheme(
        &self,
        scheme_tab: CestSchemeTab,
        b1_eff_hz: f64,
    ) -> Result<MultiCestScheme, String> {
        let mc = &self.multi_cest;
        let scheme = match scheme_tab {
            CestSchemeTab::CwCest => MultiCestScheme::Cw {
                t_ex_s: mc.t_ex_s,
                b1_hz: b1_eff_hz,
            },
            CestSchemeTab::DanteCest => {
                if mc.n_pulses == 0 {
                    return Err("DANTE-CEST requires n_pulses >= 1".to_string());
                }
                if !mc.t_ex_s.is_finite() || mc.t_ex_s <= 0.0 {
                    return Err("DANTE-CEST requires finite t_ex_s > 0".to_string());
                }
                let cycle_s = mc.t_ex_s / f64::from(mc.n_pulses);
                let tau_pulse_s = (mc.tau_pulse_us * 1e-6).clamp(1e-6, 0.25 * cycle_s);
                let tau_delay_s = (cycle_s - tau_pulse_s).max(0.0);
                let duty = tau_pulse_s / (tau_pulse_s + tau_delay_s).max(1e-12);
                MultiCestScheme::Dante {
                    tau_pulse_s,
                    tau_delay_s,
                    n_pulses: mc.n_pulses,
                    b1_pulse_hz: b1_eff_hz / duty.max(1e-9),
                }
            }
            CestSchemeTab::CosCest => MultiCestScheme::CosModulated {
                t_ex_s: mc.t_ex_s,
                b1_eff_hz,
                mod_freq_hz: mc.mod_freq_hz,
                n_steps_per_cycle: mc.n_steps_per_cycle.clamp(4, 256),
            },
        };
        scheme.validate()?;
        Ok(scheme)
    }

    fn multi_cest_offset_grid(&self, scheme_tab: CestSchemeTab) -> Vec<f64> {
        let major = self.multi_cest.major_ppm;
        match scheme_tab {
            CestSchemeTab::DanteCest => (-28..=28).map(|k| major + 0.19 * (k as f64)).collect(),
            CestSchemeTab::CosCest => (-24..=36).map(|k| major + 0.32 * (k as f64)).collect(),
            CestSchemeTab::CwCest => (-28..=28).map(|k| major + 0.30 * (k as f64)).collect(),
        }
    }

    fn compute_dense_multi_cest_curves(
        &self,
        profiles: &[MultiCestProfile],
        fit_params: MultiCestFitParams,
    ) -> Result<Vec<NamedCurve2d>, String> {
        let mut out = Vec::with_capacity(profiles.len());
        for prof in profiles {
            let lo = prof
                .offsets_ppm
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min);
            let hi = prof
                .offsets_ppm
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            if !lo.is_finite() || !hi.is_finite() || hi <= lo {
                continue;
            }
            let n_dense = 110usize;
            let dense_offs: Vec<f64> = (0..=n_dense)
                .map(|k| lo + (hi - lo) * (k as f64) / (n_dense as f64))
                .collect();
            let sim_p = MultiCestSimParams {
                p_b: fit_params.p_b,
                k_ex_s_inv: fit_params.k_ex_s_inv,
                major_ppm: prof.major_ppm,
                dw_ab_ppm: fit_params.dw_ab_ppm,
                r1a_s_inv: fit_params.r1_s_inv,
                r2a_s_inv: fit_params.r2a_s_inv,
                r1b_s_inv: fit_params.r1_s_inv,
                r2b_s_inv: fit_params.r2b_s_inv,
                larmor_mhz: prof.larmor_mhz,
                b1_inhomogeneity_frac: prof.b1_inhomogeneity_frac,
                n_b1_quad: prof.n_b1_quad,
                normalization: prof.normalization,
                scheme: prof.scheme.clone(),
            };
            let sim_y = simulate_multi_cest_profile(&sim_p, &dense_offs)?;
            let label = format!("B1 = {:.1} Hz", prof.scheme.effective_b1_hz());
            out.push((label, dense_offs.into_iter().zip(sim_y).collect()));
        }
        Ok(out)
    }

    fn run_multi_cest_2state_fit_on_profiles(
        profiles: &[MultiCestProfile],
    ) -> Result<MultiCestFitResult, String> {
        let lm_cfg = MultiCestLmConfig {
            max_iters: 35,
            grid_dw_range_ppm: (-6.0, 6.0),
            grid_dw_step_ppm: 0.8,
            ..MultiCestLmConfig::default()
        };
        let seed = deepest_dip_heuristic_seed(profiles)?;
        let mut best = fit_multi_cest_lm(profiles, seed, &lm_cfg)?;
        if !best.converged || best.reduced_chi2 > 2.2 {
            if let Ok(grid_fit) = fit_multi_cest_grid_lm(profiles, &lm_cfg) {
                if grid_fit.chi2.is_finite() && grid_fit.chi2 < best.chi2 {
                    best = grid_fit;
                }
            }
        }
        Ok(best)
    }

    /// Simulate a forward preview profile for `DANTE-CEST` or `cos-CEST` using the current
    /// editable parameters in `self.multi_cest` (`nmr_core::multi_cest::simulate_multi_cest_profile`).
    pub fn simulate_multi_cest_preview(
        &mut self,
        scheme_tab: CestSchemeTab,
    ) -> Result<usize, String> {
        let target = if scheme_tab == CestSchemeTab::CwCest {
            CestSchemeTab::DanteCest
        } else {
            scheme_tab
        };
        let b1_fields: Vec<f64> = if self.multi_cest.use_dual_b1 {
            vec![self.multi_cest.b1_hz, self.multi_cest.b1_second_hz]
        } else {
            vec![self.multi_cest.b1_hz]
        };
        let offsets = self.multi_cest_offset_grid(target);
        let mut curves = Vec::with_capacity(b1_fields.len());
        for b1 in b1_fields {
            let scheme = match self.build_multi_cest_scheme(target, b1) {
                Ok(s) => s,
                Err(e) => {
                    self.multi_cest.last_error = Some(e.clone());
                    return Err(e);
                }
            };
            let sim_p = MultiCestSimParams {
                p_b: self.multi_cest.demo_p_b,
                k_ex_s_inv: self.multi_cest.demo_k_ex_s_inv,
                major_ppm: self.multi_cest.major_ppm,
                dw_ab_ppm: self.multi_cest.demo_dw_ab_ppm,
                r1a_s_inv: self.multi_cest.demo_r1_s_inv,
                r2a_s_inv: self.multi_cest.demo_r2a_s_inv,
                r1b_s_inv: self.multi_cest.demo_r1_s_inv,
                r2b_s_inv: self.multi_cest.demo_r2b_s_inv,
                larmor_mhz: self.multi_cest.larmor_mhz,
                b1_inhomogeneity_frac: self.multi_cest.b1_inhomogeneity_frac,
                n_b1_quad: self.multi_cest.n_b1_quad,
                normalization: self.multi_cest.normalization,
                scheme,
            };
            let y = match simulate_multi_cest_profile(&sim_p, &offsets) {
                Ok(v) => v,
                Err(e) => {
                    self.multi_cest.last_error = Some(e.clone());
                    return Err(e);
                }
            };
            curves.push((
                format!("B1 = {b1:.1} Hz"),
                offsets.iter().copied().zip(y).collect(),
            ));
        }

        let n_pts = offsets.len();
        match target {
            CestSchemeTab::DanteCest => self.multi_cest.dante_sim_curves = curves,
            CestSchemeTab::CosCest => self.multi_cest.cos_sim_curves = curves,
            CestSchemeTab::CwCest => {}
        }
        self.multi_cest.last_error = None;
        self.multi_cest.last_status = format!(
            "Simulated {} preview ({n_pts} offsets, B1 = {:.1} Hz, T_EX = {:.2} s)",
            target.label(),
            self.multi_cest.b1_hz,
            self.multi_cest.t_ex_s
        );
        Ok(n_pts)
    }

    /// Generate and fit a realistic `[Synthetic Demo]` multi-B1 `DANTE-CEST` or `cos-CEST`
    /// dataset using the current editable parameters in `self.multi_cest`.
    pub fn load_multi_cest_synthetic_demo(
        &mut self,
        scheme_tab: CestSchemeTab,
    ) -> Result<String, String> {
        let target = if scheme_tab == CestSchemeTab::CwCest {
            CestSchemeTab::DanteCest
        } else {
            scheme_tab
        };
        let b1_fields: Vec<f64> = if self.multi_cest.use_dual_b1 {
            vec![self.multi_cest.b1_hz, self.multi_cest.b1_second_hz]
        } else {
            vec![self.multi_cest.b1_hz]
        };
        let offsets = self.multi_cest_offset_grid(target);
        let sigma = self.multi_cest.demo_noise_sigma.max(1e-4);
        let seed_offset = if target == CestSchemeTab::CosCest {
            1u64
        } else {
            0u64
        };
        let mut rng = Rng::new(self.multi_cest.demo_seed.wrapping_add(seed_offset));

        let mut profiles = Vec::with_capacity(b1_fields.len());
        let mut tsv_lines = vec![
            "# nmrassign_multi_cest_profile".to_string(),
            "offset_ppm\tintensity\tsigma\tb1_hz".to_string(),
        ];

        for b1 in b1_fields {
            let scheme = match self.build_multi_cest_scheme(target, b1) {
                Ok(s) => s,
                Err(e) => {
                    self.multi_cest.last_error = Some(e.clone());
                    return Err(e);
                }
            };
            let sim_p = MultiCestSimParams {
                p_b: self.multi_cest.demo_p_b,
                k_ex_s_inv: self.multi_cest.demo_k_ex_s_inv,
                major_ppm: self.multi_cest.major_ppm,
                dw_ab_ppm: self.multi_cest.demo_dw_ab_ppm,
                r1a_s_inv: self.multi_cest.demo_r1_s_inv,
                r2a_s_inv: self.multi_cest.demo_r2a_s_inv,
                r1b_s_inv: self.multi_cest.demo_r1_s_inv,
                r2b_s_inv: self.multi_cest.demo_r2b_s_inv,
                larmor_mhz: self.multi_cest.larmor_mhz,
                b1_inhomogeneity_frac: self.multi_cest.b1_inhomogeneity_frac,
                n_b1_quad: self.multi_cest.n_b1_quad,
                normalization: self.multi_cest.normalization,
                scheme: scheme.clone(),
            };
            let clean = match simulate_multi_cest_profile(&sim_p, &offsets) {
                Ok(v) => v,
                Err(e) => {
                    self.multi_cest.last_error = Some(e.clone());
                    return Err(e);
                }
            };
            let noisy: Vec<f64> = clean.iter().map(|&y| y + rng.normal(0.0, sigma)).collect();
            for (&off, &val) in offsets.iter().zip(noisy.iter()) {
                tsv_lines.push(format!("{off:.4}\t{val:.6}\t{sigma:.5}\t{b1:.2}"));
            }
            profiles.push(MultiCestProfile {
                larmor_mhz: self.multi_cest.larmor_mhz,
                major_ppm: self.multi_cest.major_ppm,
                b1_inhomogeneity_frac: self.multi_cest.b1_inhomogeneity_frac,
                n_b1_quad: self.multi_cest.n_b1_quad,
                normalization: self.multi_cest.normalization,
                scheme,
                offsets_ppm: offsets.clone(),
                intensities: noisy,
                uncertainties: vec![sigma; offsets.len()],
            });
        }

        let fit = match Self::run_multi_cest_2state_fit_on_profiles(&profiles) {
            Ok(f) => f,
            Err(e) => {
                self.multi_cest.last_error = Some(e.clone());
                return Err(e);
            }
        };
        let curves = match self.compute_dense_multi_cest_curves(&profiles, fit.params) {
            Ok(c) => c,
            Err(e) => {
                self.multi_cest.last_error = Some(e.clone());
                return Err(e);
            }
        };

        let summary = format_multi_cest_summary(target, &fit, true);
        self.multi_cest.tsv_input = tsv_lines.join("\n");
        match target {
            CestSchemeTab::DanteCest => {
                self.multi_cest.dante_profiles = profiles;
                self.multi_cest.dante_sim_curves = curves;
                self.multi_cest.dante_fit = Some(fit);
                self.multi_cest.dante_is_synthetic_demo = true;
            }
            CestSchemeTab::CosCest => {
                self.multi_cest.cos_profiles = profiles;
                self.multi_cest.cos_sim_curves = curves;
                self.multi_cest.cos_fit = Some(fit);
                self.multi_cest.cos_is_synthetic_demo = true;
            }
            CestSchemeTab::CwCest => {}
        }
        self.multi_cest.last_error = None;
        self.multi_cest.last_status = summary.clone();
        Ok(summary)
    }

    /// Parse pasted `DANTE-CEST` or `cos-CEST` profile TSV (`offset_ppm\tintensity\tsigma[\tb1_hz]`)
    /// and run the exact 2-state Bloch–McConnell LM fitter (`nmr_core::multi_cest`).
    pub fn import_multi_cest_tsv(
        &mut self,
        scheme_tab: CestSchemeTab,
        tsv: &str,
    ) -> Result<String, String> {
        let target = if scheme_tab == CestSchemeTab::CwCest {
            CestSchemeTab::DanteCest
        } else {
            scheme_tab
        };
        self.multi_cest.tsv_input = tsv.to_string();

        let mut grouped: GroupedMultiCestTsv = BTreeMap::new();
        for (line_idx, raw_line) in tsv.lines().enumerate() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let cols: Vec<&str> = line.split('\t').map(str::trim).collect();
            if cols.len() < 3 {
                let err = format!(
                    "line {}: expected at least 3 tab-separated columns (offset_ppm, intensity, sigma[, b1_hz])",
                    line_idx + 1
                );
                self.multi_cest.last_error = Some(err.clone());
                return Err(err);
            }
            if cols[0].eq_ignore_ascii_case("offset_ppm") {
                continue;
            }
            let off: f64 = cols[0].parse().map_err(|e| {
                let msg = format!("line {}: invalid offset_ppm: {e}", line_idx + 1);
                self.multi_cest.last_error = Some(msg.clone());
                msg
            })?;
            let inten: f64 = cols[1].parse().map_err(|e| {
                let msg = format!("line {}: invalid intensity: {e}", line_idx + 1);
                self.multi_cest.last_error = Some(msg.clone());
                msg
            })?;
            let sig: f64 = cols[2].parse().map_err(|e| {
                let msg = format!("line {}: invalid sigma: {e}", line_idx + 1);
                self.multi_cest.last_error = Some(msg.clone());
                msg
            })?;
            let b1: f64 = if let Some(b1_str) = cols.get(3).filter(|s| !s.is_empty()) {
                b1_str.parse().map_err(|e| {
                    let msg = format!("line {}: invalid b1_hz: {e}", line_idx + 1);
                    self.multi_cest.last_error = Some(msg.clone());
                    msg
                })?
            } else {
                self.multi_cest.b1_hz
            };
            let b1_key = (b1 * 1000.0).round() as i64;
            let entry = grouped
                .entry(b1_key)
                .or_insert_with(|| (b1, Vec::new(), Vec::new(), Vec::new()));
            entry.1.push(off);
            entry.2.push(inten);
            entry.3.push(sig);
        }

        if grouped.is_empty() {
            let err = "multi-CEST TSV contained no data rows".to_string();
            self.multi_cest.last_error = Some(err.clone());
            return Err(err);
        }

        let mut profiles = Vec::with_capacity(grouped.len());
        for (_, (b1, offsets_ppm, intensities, uncertainties)) in grouped {
            let scheme = match self.build_multi_cest_scheme(target, b1) {
                Ok(s) => s,
                Err(e) => {
                    self.multi_cest.last_error = Some(e.clone());
                    return Err(e);
                }
            };
            let prof = MultiCestProfile {
                larmor_mhz: self.multi_cest.larmor_mhz,
                major_ppm: self.multi_cest.major_ppm,
                b1_inhomogeneity_frac: self.multi_cest.b1_inhomogeneity_frac,
                n_b1_quad: self.multi_cest.n_b1_quad,
                normalization: self.multi_cest.normalization,
                scheme,
                offsets_ppm,
                intensities,
                uncertainties,
            };
            if let Err(e) = prof.validate() {
                self.multi_cest.last_error = Some(e.clone());
                return Err(e);
            }
            profiles.push(prof);
        }

        let fit = match Self::run_multi_cest_2state_fit_on_profiles(&profiles) {
            Ok(f) => f,
            Err(e) => {
                self.multi_cest.last_error = Some(e.clone());
                return Err(e);
            }
        };
        let curves = match self.compute_dense_multi_cest_curves(&profiles, fit.params) {
            Ok(c) => c,
            Err(e) => {
                self.multi_cest.last_error = Some(e.clone());
                return Err(e);
            }
        };
        let summary = format_multi_cest_summary(target, &fit, false);
        match target {
            CestSchemeTab::DanteCest => {
                self.multi_cest.dante_profiles = profiles;
                self.multi_cest.dante_sim_curves = curves;
                self.multi_cest.dante_fit = Some(fit);
                self.multi_cest.dante_is_synthetic_demo = false;
            }
            CestSchemeTab::CosCest => {
                self.multi_cest.cos_profiles = profiles;
                self.multi_cest.cos_sim_curves = curves;
                self.multi_cest.cos_fit = Some(fit);
                self.multi_cest.cos_is_synthetic_demo = false;
            }
            CestSchemeTab::CwCest => {}
        }
        self.multi_cest.last_error = None;
        self.multi_cest.last_status = summary.clone();
        Ok(summary)
    }

    /// Re-run the 2-state `DANTE-CEST` or `cos-CEST` fit non-destructively using the current
    /// editable parameters (re-fitting imported profiles if present, or generating/fitting the
    /// synthetic demo dataset otherwise).
    pub fn fit_multi_cest_2state(&mut self, scheme_tab: CestSchemeTab) -> Result<String, String> {
        let target = if scheme_tab == CestSchemeTab::CwCest {
            CestSchemeTab::DanteCest
        } else {
            scheme_tab
        };
        let (has_user_profiles, existing_profiles) = match target {
            CestSchemeTab::DanteCest => (
                !self.multi_cest.dante_is_synthetic_demo
                    && !self.multi_cest.dante_profiles.is_empty(),
                self.multi_cest.dante_profiles.clone(),
            ),
            CestSchemeTab::CosCest => (
                !self.multi_cest.cos_is_synthetic_demo && !self.multi_cest.cos_profiles.is_empty(),
                self.multi_cest.cos_profiles.clone(),
            ),
            CestSchemeTab::CwCest => (false, Vec::new()),
        };

        if !has_user_profiles {
            return self.load_multi_cest_synthetic_demo(target);
        }

        // Update schemes & physical metadata from current editable controls while preserving each
        // imported profile's own effective B1 (NEVER overwrite profile 0 with the demo slider).
        let mut updated = Vec::with_capacity(existing_profiles.len());
        for mut prof in existing_profiles {
            let b1_eff = prof.scheme.effective_b1_hz().max(0.0);
            let scheme = match self.build_multi_cest_scheme(target, b1_eff) {
                Ok(s) => s,
                Err(e) => {
                    self.multi_cest.last_error = Some(e.clone());
                    return Err(e);
                }
            };
            prof.larmor_mhz = self.multi_cest.larmor_mhz;
            prof.major_ppm = self.multi_cest.major_ppm;
            prof.b1_inhomogeneity_frac = self.multi_cest.b1_inhomogeneity_frac;
            prof.n_b1_quad = self.multi_cest.n_b1_quad;
            prof.normalization = self.multi_cest.normalization;
            prof.scheme = scheme;
            if let Err(e) = prof.validate() {
                self.multi_cest.last_error = Some(e.clone());
                return Err(e);
            }
            updated.push(prof);
        }

        let fit = match Self::run_multi_cest_2state_fit_on_profiles(&updated) {
            Ok(f) => f,
            Err(e) => {
                self.multi_cest.last_error = Some(e.clone());
                return Err(e);
            }
        };
        let curves = match self.compute_dense_multi_cest_curves(&updated, fit.params) {
            Ok(c) => c,
            Err(e) => {
                self.multi_cest.last_error = Some(e.clone());
                return Err(e);
            }
        };
        let summary = format_multi_cest_summary(target, &fit, false);
        match target {
            CestSchemeTab::DanteCest => {
                self.multi_cest.dante_profiles = updated;
                self.multi_cest.dante_sim_curves = curves;
                self.multi_cest.dante_fit = Some(fit);
            }
            CestSchemeTab::CosCest => {
                self.multi_cest.cos_profiles = updated;
                self.multi_cest.cos_sim_curves = curves;
                self.multi_cest.cos_fit = Some(fit);
            }
            CestSchemeTab::CwCest => {}
        }
        self.multi_cest.last_error = None;
        self.multi_cest.last_status = summary.clone();
        Ok(summary)
    }

    // =========================================================================
    // T22.8: 2D Titration Trajectory Tracking, Exchange Regime & Lineshape Fit
    // =========================================================================

    fn build_trajectory_track_config(&self) -> TrajectoryTrackConfig {
        TrajectoryTrackConfig {
            n_scale_factor: self.titration_2d.n_scale_factor.clamp(0.05, 0.50),
            max_step_ppm: self.titration_2d.max_step_ppm.clamp(0.05, 2.0),
            disappear_cost: self.titration_2d.max_step_ppm.clamp(0.05, 2.0).powi(2),
            ..TrajectoryTrackConfig::default()
        }
    }

    /// Generate and analyze a realistic `[Synthetic Demo]` 2D titration series exercising
    /// velocity-extrapolated Kuhn–Munkres trajectory tracking (crossing paths + midpoint dropout),
    /// per-step exchange-regime diagnostics (`|dw|/k_off`), and global coupled 2D Bloch–McConnell
    /// lineshape fitting (`Kd`, `k_off`, `dwH`, `dwN`).
    pub fn load_titration_2d_synthetic_demo(&mut self) -> Result<String, String> {
        let kd_um = self.titration_2d.demo_kd_um.clamp(0.5, 5000.0);
        let k_off = self.titration_2d.demo_k_off_s_inv.clamp(1.0, 20_000.0);
        let p_tot_um = self.titration_2d.protein_conc_um.clamp(1.0, 2000.0);
        let h_mhz = self.titration_2d.h_larmor_mhz.clamp(200.0, 1200.0);
        let snr = self.titration_2d.demo_snr.clamp(10.0, 500.0);
        let ligand_concs = [0.0_f64, 15.0, 40.0, 80.0, 160.0, 320.0];

        // 1. Build 2D peak series with a crossing pair (A10 / L11), a midpoint exchange-broadened
        //    dropout (I44 at step 2), and a shifting pocket residue (D61) + editable TSV.
        let track_defs: [(&str, &str, f64, f64, f64, f64, bool); 4] = [
            ("A10", "ALA", 7.82, 118.20, 0.055, 0.36, false),
            ("L11", "LEU", 7.82, 120.20, 0.055, -0.36, false),
            ("I44", "ILE", 8.20, 118.40, 0.018, 0.14, true),
            ("D61", "ASP", 8.92, 123.60, -0.012, 0.14, false),
        ];

        let mut steps = Vec::with_capacity(ligand_concs.len());
        let mut tsv_lines = vec![
            "# nmrassign_titration_2d_peaks".to_string(),
            "step_index\tstep_label\tligand_conc_um\tprotein_conc_um\tpeak_id\th_ppm\tn_ppm\tlabel\tres_name".to_string(),
        ];

        for (t, &l_um) in ligand_concs.iter().enumerate() {
            let step_label = format!("{l_um:.0} uM");
            let mut peaks = Vec::new();
            let tf = t as f64;
            for (id, &(lbl, res, h0, n0, dh, dn, drops_mid)) in track_defs.iter().enumerate() {
                if drops_mid && t == 2 {
                    continue;
                }
                let h = h0 + dh * tf;
                let n = n0 + dn * tf;
                peaks.push(Peak2d {
                    peak_id: id,
                    h_ppm: h,
                    n_ppm: n,
                    intensity: Some(1.0),
                    label: if t == 0 { Some(lbl.to_string()) } else { None },
                    res_name: if t == 0 { Some(res.to_string()) } else { None },
                });
                let lbl_col = if t == 0 { lbl } else { "" };
                let res_col = if t == 0 { res } else { "" };
                tsv_lines.push(format!(
                    "{t}\t{step_label}\t{l_um:.2}\t{p_tot_um:.2}\t{id}\t{h:.4}\t{n:.3}\t{lbl_col}\t{res_col}"
                ));
            }
            if t % 2 == 1 {
                peaks.reverse();
            }
            steps.push(SeriesPeakList {
                label: step_label,
                series_coord: l_um,
                protein_conc_um: Some(p_tot_um),
                ligand_conc_um: Some(l_um),
                peaks,
            });
        }

        let track_cfg = self.build_trajectory_track_config();
        let tracking = track_titration_trajectories(&steps, &track_cfg).map_err(|e| {
            let msg = format!("2D trajectory tracking error: {e}");
            self.titration_2d.last_error = Some(msg.clone());
            msg
        })?;

        // 2. Synthesize coupled 2D Bloch–McConnell ROI patches for two pocket residues (I44, D61)
        //    spanning fast/intermediate/slow regimes cleanly determined by (Kd, k_off).
        let true_residues: [(&str, &str, Bloch2dResidueParams); 2] = [
            (
                "I44",
                "ILE",
                Bloch2dResidueParams {
                    delta_a_h_ppm: 8.20,
                    delta_b_h_ppm: 8.29, // dw_H = +0.09 ppm
                    delta_a_n_ppm: 118.40,
                    delta_b_n_ppm: 119.10, // dw_N = +0.70 ppm
                    r2a_h_s_inv: 22.0,
                    r2b_h_s_inv: 29.0,
                    r2a_n_s_inv: 18.0,
                    r2b_n_s_inv: 24.0,
                    amplitude: 4600.0,
                },
            ),
            (
                "D61",
                "ASP",
                Bloch2dResidueParams {
                    delta_a_h_ppm: 8.92,
                    delta_b_h_ppm: 8.86, // dw_H = -0.06 ppm
                    delta_a_n_ppm: 123.60,
                    delta_b_n_ppm: 124.30, // dw_N = +0.70 ppm
                    r2a_h_s_inv: 20.0,
                    r2b_h_s_inv: 27.0,
                    r2a_n_s_inv: 17.0,
                    r2b_n_s_inv: 23.0,
                    amplitude: 3900.0,
                },
            ),
        ];

        let mut rng = Rng::new(self.titration_2d.demo_seed);
        let mut roi_series = Vec::with_capacity(true_residues.len());
        for (label, res_name, true_p) in &true_residues {
            let n_min = true_p.delta_a_n_ppm.min(true_p.delta_b_n_ppm) - 0.55;
            let n_max = true_p.delta_a_n_ppm.max(true_p.delta_b_n_ppm) + 0.55;
            let h_min = true_p.delta_a_h_ppm.min(true_p.delta_b_h_ppm) - 0.08;
            let h_max = true_p.delta_a_h_ppm.max(true_p.delta_b_h_ppm) + 0.08;
            let grid =
                Roi2dGrid::uniform(n_min, n_max, 15, h_min, h_max, 15, h_mhz).map_err(|e| {
                    let msg = format!("ROI grid error: {e}");
                    self.titration_2d.last_error = Some(msg.clone());
                    msg
                })?;
            let apo_peak = true_p.amplitude / (true_p.r2a_n_s_inv * true_p.r2a_h_s_inv);
            let noise_sigma = (apo_peak / snr).max(1e-6);

            let mut patches = Vec::with_capacity(ligand_concs.len());
            for (step_idx, &l_um) in ligand_concs.iter().enumerate() {
                let eq = binding_equilibrium_rates(p_tot_um, l_um, kd_um, k_off).map_err(|e| {
                    let msg = format!("binding equilibrium error: {e}");
                    self.titration_2d.last_error = Some(msg.clone());
                    msg
                })?;
                let st = Bloch2dKineticState::from(eq);
                let mut intensities = simulate_2d_lineshape_analytical(&grid, true_p, &st)
                    .map_err(|e| {
                        let msg = format!("2D lineshape simulation error: {e}");
                        self.titration_2d.last_error = Some(msg.clone());
                        msg
                    })?;
                for v in &mut intensities {
                    *v += rng.normal(0.0, noise_sigma);
                }
                patches.push(Roi2dPatch {
                    step_index: step_idx,
                    protein_conc_um: p_tot_um,
                    ligand_conc_um: l_um,
                    grid: grid.clone(),
                    intensities,
                    noise_sigma,
                    spectrum_scale: 1.0,
                });
            }
            roi_series.push(ResidueTitration2dSeries {
                label: (*label).to_string(),
                res_name: (*res_name).to_string(),
                patches,
                initial_guess: None,
            });
        }

        let fit_cfg = Titration2dFitConfig {
            max_lm_iterations: 45,
            n_coarse_starts: 2,
            ..Titration2dFitConfig::default()
        };
        let fit = fit_titration_2d_lineshape(&roi_series, &fit_cfg).map_err(|e| {
            let msg = format!("2D Bloch-McConnell fit error: {e}");
            self.titration_2d.last_error = Some(msg.clone());
            msg
        })?;

        let regime_reports: Vec<ResidueExchangeRegimeReport> = fit
            .residues
            .iter()
            .map(|r| r.exchange_regime.clone())
            .collect();
        let summary = format!(
            "[Synthetic Demo] Tracked {} trajectories ({} dropouts) | 2D Fit: Kd = {:.2} ± {:.2} µM, k_off = {:.1} ± {:.1} s⁻¹, χ²_red = {:.3}",
            tracking.trajectories.len(),
            tracking
                .trajectories
                .iter()
                .map(|t| t.n_dropouts)
                .sum::<usize>(),
            fit.kd_um,
            fit.kd_sd_um,
            fit.k_off_s_inv,
            fit.k_off_sd_s_inv,
            fit.reduced_chi2,
        );

        self.titration_2d.tsv_input = tsv_lines.join("\n");
        self.titration_2d.series_steps = steps;
        self.titration_2d.tracking_result = Some(tracking);
        self.titration_2d.regime_reports = regime_reports;
        self.titration_2d.roi_series = roi_series;
        self.titration_2d.lineshape_fit = Some(fit);
        self.titration_2d.is_synthetic_demo = true;
        self.titration_2d.selected_residue_idx = 0;
        self.titration_2d.selected_step_idx = 2.min(ligand_concs.len().saturating_sub(1));
        self.titration_2d.last_error = None;
        self.titration_2d.last_status = summary.clone();
        Ok(summary)
    }

    /// Parse pasted 2D titration peak-series TSV
    /// (`step_index\tstep_label\tligand_conc_um\tprotein_conc_um\tpeak_id\th_ppm\tn_ppm[\tlabel\tres_name]`),
    /// run velocity-extrapolated Kuhn–Munkres trajectory tracking, and compute per-step exchange-regime
    /// diagnostics (`|dw|/k_off`).
    pub fn import_titration_2d_tsv(&mut self, tsv: &str) -> Result<String, String> {
        self.titration_2d.tsv_input = tsv.to_string();

        let mut step_map: BTreeMap<usize, SeriesPeakList> = BTreeMap::new();
        for (line_idx, raw_line) in tsv.lines().enumerate() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let cols: Vec<&str> = line.split('\t').map(str::trim).collect();
            if cols
                .first()
                .is_some_and(|c| c.eq_ignore_ascii_case("step_index"))
            {
                continue;
            }
            if cols.len() < 7 {
                let err = format!(
                    "line {}: expected >= 7 columns (step_index, step_label, ligand_conc_um, protein_conc_um, peak_id, h_ppm, n_ppm[, label, res_name])",
                    line_idx + 1
                );
                self.titration_2d.last_error = Some(err.clone());
                return Err(err);
            }
            let step_index: usize = cols[0].parse().map_err(|e| {
                let msg = format!("line {}: invalid step_index: {e}", line_idx + 1);
                self.titration_2d.last_error = Some(msg.clone());
                msg
            })?;
            let step_label = if cols[1].is_empty() {
                format!("step_{step_index}")
            } else {
                cols[1].to_string()
            };
            let ligand_conc_um: f64 = cols[2].parse().map_err(|e| {
                let msg = format!("line {}: invalid ligand_conc_um: {e}", line_idx + 1);
                self.titration_2d.last_error = Some(msg.clone());
                msg
            })?;
            let protein_conc_um: f64 = cols[3].parse().map_err(|e| {
                let msg = format!("line {}: invalid protein_conc_um: {e}", line_idx + 1);
                self.titration_2d.last_error = Some(msg.clone());
                msg
            })?;
            let peak_id: usize = cols[4].parse().map_err(|e| {
                let msg = format!("line {}: invalid peak_id: {e}", line_idx + 1);
                self.titration_2d.last_error = Some(msg.clone());
                msg
            })?;
            let h_ppm: f64 = cols[5].parse().map_err(|e| {
                let msg = format!("line {}: invalid h_ppm: {e}", line_idx + 1);
                self.titration_2d.last_error = Some(msg.clone());
                msg
            })?;
            let n_ppm: f64 = cols[6].parse().map_err(|e| {
                let msg = format!("line {}: invalid n_ppm: {e}", line_idx + 1);
                self.titration_2d.last_error = Some(msg.clone());
                msg
            })?;
            let label = cols
                .get(7)
                .filter(|s| !s.is_empty())
                .map(|s| (*s).to_string());
            let res_name = cols
                .get(8)
                .filter(|s| !s.is_empty())
                .map(|s| (*s).to_string());

            let entry = step_map
                .entry(step_index)
                .or_insert_with(|| SeriesPeakList {
                    label: step_label,
                    series_coord: ligand_conc_um,
                    protein_conc_um: Some(protein_conc_um),
                    ligand_conc_um: Some(ligand_conc_um),
                    peaks: Vec::new(),
                });
            entry.peaks.push(Peak2d {
                peak_id,
                h_ppm,
                n_ppm,
                intensity: Some(1.0),
                label,
                res_name,
            });
        }

        let steps: Vec<SeriesPeakList> = step_map.into_values().collect();
        if steps.len() < 2 {
            let err = "2D titration TSV requires at least 2 monotonic titration steps".to_string();
            self.titration_2d.last_error = Some(err.clone());
            return Err(err);
        }

        let track_cfg = self.build_trajectory_track_config();
        let tracking = track_titration_trajectories(&steps, &track_cfg).map_err(|e| {
            let msg = format!("2D trajectory tracking error: {e}");
            self.titration_2d.last_error = Some(msg.clone());
            msg
        })?;

        // Build concentration points for exchange-regime classification on every tracked residue
        let conc_pts: Vec<TitrationConcentrationPoint> = steps
            .iter()
            .enumerate()
            .map(|(i, s)| TitrationConcentrationPoint {
                step_index: i,
                label: s.label.clone(),
                protein_conc_um: s
                    .protein_conc_um
                    .unwrap_or(self.titration_2d.protein_conc_um)
                    .max(1e-6),
                ligand_conc_um: s.ligand_conc_um.unwrap_or(s.series_coord).max(0.0),
            })
            .collect();

        let kd_hint = self.titration_2d.initial_kd_hint_um.max(1e-3);
        let koff_hint = self.titration_2d.initial_k_off_hint_s_inv.max(0.1);
        let h_mhz = self.titration_2d.h_larmor_mhz.max(100.0);

        let mut regime_reports = Vec::new();
        for traj in &tracking.trajectories {
            let first_pos = traj.observations.iter().find_map(|o| match o.status {
                TrackStepStatus::Matched { h_ppm, n_ppm, .. } => Some((h_ppm, n_ppm)),
                _ => None,
            });
            let last_pos = traj.observations.iter().rev().find_map(|o| match o.status {
                TrackStepStatus::Matched { h_ppm, n_ppm, .. } => Some((h_ppm, n_ppm)),
                _ => None,
            });
            if let (Some((h0, n0)), Some((h1, n1))) = (first_pos, last_pos) {
                let lbl = traj
                    .label
                    .clone()
                    .unwrap_or_else(|| format!("T{}", traj.track_id));
                if let Ok(rep) = classify_exchange_regime(
                    &ExchangeRegimeInput {
                        label: lbl,
                        kd_um: kd_hint,
                        k_off_s_inv: koff_hint,
                        dw_h_ppm: h1 - h0,
                        dw_n_ppm: n1 - n0,
                        h_larmor_mhz: h_mhz,
                        n_larmor_mhz: None,
                    },
                    &conc_pts,
                ) {
                    regime_reports.push(rep);
                }
            }
        }

        let n_tracks = tracking.trajectories.len();
        let n_ambig = tracking.ambiguity_flags.len();
        let summary = format!(
            "Tracked {n_tracks} trajectories across {} steps ({n_ambig} ambiguity flags); exchange-regime audit evaluated for {} tracks (2D lineshape fit requires 2D ROI planes or [Synthetic Demo])",
            steps.len(),
            regime_reports.len(),
        );

        self.titration_2d.series_steps = steps;
        self.titration_2d.tracking_result = Some(tracking);
        self.titration_2d.regime_reports = regime_reports;
        self.titration_2d.roi_series.clear();
        self.titration_2d.lineshape_fit = None;
        self.titration_2d.is_synthetic_demo = false;
        self.titration_2d.selected_residue_idx = 0;
        self.titration_2d.selected_step_idx = 0;
        self.titration_2d.last_error = None;
        self.titration_2d.last_status = summary.clone();
        Ok(summary)
    }

    /// Re-run 2D titration trajectory tracking, exchange-regime diagnostics, and coupled 2D
    /// Bloch–McConnell lineshape fitting non-destructively from current editable parameters/TSV.
    pub fn run_titration_2d_analysis(&mut self) -> Result<String, String> {
        if self.titration_2d.is_synthetic_demo
            || (self.titration_2d.series_steps.is_empty()
                && self.titration_2d.tsv_input.trim().is_empty())
        {
            self.load_titration_2d_synthetic_demo()
        } else {
            let tsv = self.titration_2d.tsv_input.clone();
            self.import_titration_2d_tsv(&tsv)
        }
    }

    /// Build a 1-click [`QuantitativeOverlayProjection`] for Lipari–Szabo `S^2`, chemical exchange
    /// `R_ex` (preferring model-free `R_ex`, falling back to `CPMG` observed/fitted `R_ex`), or
    /// ligand titration `Delta_delta_comb` (`CSP`).
    pub fn build_overlay_projection(
        &self,
        track: QuantitativeTrack,
    ) -> Option<QuantitativeOverlayProjection> {
        let mut values_by_seq_id = BTreeMap::new();
        let mut max_val = 0.0f64;

        match track {
            QuantitativeTrack::LipariSzaboS2 => {
                for r in &self.relaxation_records {
                    if let Some((s2, _)) = r.s2 {
                        let v = s2.clamp(0.0, 1.0);
                        values_by_seq_id.insert(r.seq_id, v);
                        max_val = max_val.max(v);
                    }
                }
                max_val = 1.0;
            }
            QuantitativeTrack::ChemicalExchangeRex => {
                for r in &self.relaxation_records {
                    if let Some((rex, _)) = r.rex_s_inv {
                        let v = rex.max(0.0);
                        values_by_seq_id.insert(r.seq_id, v);
                        max_val = max_val.max(v);
                    }
                }
                if values_by_seq_id.is_empty() {
                    if let Some(fit) = &self.cpmg_fit {
                        for rf in &fit.residues {
                            let v = rf.gate.observed_rex_s_inv.max(0.0);
                            values_by_seq_id.insert(rf.seq_id, v);
                            max_val = max_val.max(v);
                        }
                    } else {
                        for s in &self.cpmg_series {
                            let r_min =
                                s.r2_eff_s_inv.iter().copied().fold(f64::INFINITY, f64::min);
                            let r_max = s
                                .r2_eff_s_inv
                                .iter()
                                .copied()
                                .fold(f64::NEG_INFINITY, f64::max);
                            if r_max > r_min {
                                let v = r_max - r_min;
                                let entry = values_by_seq_id.entry(s.seq_id).or_insert(0.0);
                                *entry = entry.max(v);
                                max_val = max_val.max(*entry);
                            }
                        }
                    }
                }
            }
            QuantitativeTrack::LigandCsp => {
                for t in &self.csp_trajectories {
                    let v = t.max_observed_csp_ppm.max(0.0);
                    values_by_seq_id.insert(t.seq_id, v);
                    max_val = max_val.max(v);
                }
            }
            _ => return None,
        }

        if values_by_seq_id.is_empty() {
            return None;
        }

        Some(QuantitativeOverlayProjection {
            track,
            colormap: self.overlay_colormap,
            putty_enabled: self.putty_enabled,
            values_by_seq_id,
            display_max: max_val.max(1e-3),
        })
    }

    /// Render the interactive studio and invoke `on_project` immediately if the user clicks one of
    /// the 1-click 3D Putty projection buttons.
    pub fn show_with_callback<F>(
        &mut self,
        ui: &mut egui::Ui,
        mut on_project: F,
    ) -> DynamicsStudioOutput
    where
        F: FnMut(&QuantitativeOverlayProjection),
    {
        let out = self.show(ui);
        if let Some(proj) = &out.overlay_projection {
            on_project(proj);
        }
        out
    }

    /// Render the complete interactive Dynamics & CSP Studio inside `ui`.
    pub fn show(&mut self, ui: &mut egui::Ui) -> DynamicsStudioOutput {
        let mut output = DynamicsStudioOutput::default();

        // Primary Toolbar (<= 8 visible primary interactive controls)
        ui.horizontal_wrapped(|ui| {
            for tab in DynamicsStudioTab::ALL {
                if ui
                    .selectable_label(self.active_tab == tab, tab.label())
                    .clicked()
                {
                    self.active_tab = tab;
                }
            }
            ui.separator();

            let default_track = match self.active_tab {
                DynamicsStudioTab::RelaxationModelFree => {
                    if self.active_relax_metric == SequenceMetricKind::ExchangeRex {
                        QuantitativeTrack::ChemicalExchangeRex
                    } else {
                        QuantitativeTrack::LipariSzaboS2
                    }
                }
                DynamicsStudioTab::CestDest | DynamicsStudioTab::CpmgDispersion => {
                    QuantitativeTrack::ChemicalExchangeRex
                }
                DynamicsStudioTab::LigandCsp => QuantitativeTrack::LigandCsp,
            };
            let proj_avail = match default_track {
                QuantitativeTrack::LipariSzaboS2 => {
                    self.relaxation_records.iter().any(|r| r.s2.is_some())
                }
                QuantitativeTrack::ChemicalExchangeRex => {
                    self.relaxation_records
                        .iter()
                        .any(|r| r.rex_s_inv.is_some())
                        || !self.cpmg_series.is_empty()
                }
                QuantitativeTrack::LigandCsp => !self.csp_trajectories.is_empty(),
                _ => false,
            };
            if ui
                .add_enabled(proj_avail, egui::Button::new("Project -> 3D Putty"))
                .on_hover_text("Project active quantitative dynamics/CSP metric onto 3D structure putty")
                .clicked()
            {
                output.overlay_projection = self.build_overlay_projection(default_track);
            }

            if (self.cest_profiles.is_empty()
                || self.cpmg_series.is_empty()
                || self.csp_trajectories.is_empty())
                && ui
                    .button("Load Demo CEST / CPMG / CSP")
                    .on_hover_text(
                        "Populate synthetic dual-B1 CEST, multi-field Baldwin2014 CPMG, and tight-binding CSP Kd curves",
                    )
                    .clicked()
            {
                let _ = self.ensure_demo_exchange_and_csp(&[], &[]);
            }
        });

        // Progressive-disclosure Advanced Fitting & Export section (collapsed by default)
        egui::CollapsingHeader::new(egui::RichText::new("Advanced Fitting & Export").size(12.5))
            .default_open(false)
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.checkbox(&mut self.putty_enabled, "Putty tube");
                    egui::ComboBox::from_id_salt("dyn_studio_cmap")
                        .selected_text(self.overlay_colormap.label())
                        .show_ui(ui, |ui| {
                            for cmap in ColormapKind::ALL {
                                ui.selectable_value(&mut self.overlay_colormap, cmap, cmap.label());
                            }
                        });

                    let s2_avail = self.relaxation_records.iter().any(|r| r.s2.is_some());
                    if ui
                        .add_enabled(s2_avail, egui::Button::new("Project S^2 -> 3D"))
                        .clicked()
                    {
                        output.overlay_projection =
                            self.build_overlay_projection(QuantitativeTrack::LipariSzaboS2);
                    }

                    let rex_avail = self
                        .relaxation_records
                        .iter()
                        .any(|r| r.rex_s_inv.is_some())
                        || !self.cpmg_series.is_empty();
                    if ui
                        .add_enabled(rex_avail, egui::Button::new("Project R_ex -> 3D"))
                        .clicked()
                    {
                        output.overlay_projection =
                            self.build_overlay_projection(QuantitativeTrack::ChemicalExchangeRex);
                    }

                    let csp_avail = !self.csp_trajectories.is_empty();
                    if ui
                        .add_enabled(csp_avail, egui::Button::new("Project CSP -> 3D"))
                        .clicked()
                    {
                        output.overlay_projection =
                            self.build_overlay_projection(QuantitativeTrack::LigandCsp);
                    }

                    if self.active_tab == DynamicsStudioTab::LigandCsp {
                        ui.checkbox(
                            &mut self.show_naive_mm_comparison,
                            "Overlay naive [L]_free ~ [L]_t curve",
                        );
                    }
                });
            });

        ui.separator();

        self.last_canvas_rect = Some(ui.available_rect_before_wrap().intersect(ui.clip_rect()));
        self.last_plot_rect = None;

        match self.active_tab {
            DynamicsStudioTab::RelaxationModelFree => {
                self.render_relaxation_tab(ui, &mut output);
            }
            DynamicsStudioTab::CestDest => {
                self.render_cest_tab(ui, &mut output);
            }
            DynamicsStudioTab::CpmgDispersion => {
                self.render_cpmg_tab(ui, &mut output);
            }
            DynamicsStudioTab::LigandCsp => {
                self.render_csp_tab(ui, &mut output);
            }
        }

        output
    }

    fn record_plot_rect(&mut self, rect: egui::Rect) {
        self.last_plot_rect = Some(match self.last_plot_rect {
            Some(prev) => prev.union(rect),
            None => rect,
        });
    }

    fn render_relaxation_tab(&mut self, ui: &mut egui::Ui, output: &mut DynamicsStudioOutput) {
        draw_guidance_header(
            ui,
            "Input: Per-residue R1/R2 decay & hetNOE series -> Output: Lipari-Szabo S2, R_ex & 3D putty projection",
        );
        ui.horizontal_wrapped(|ui| {
            ui.label(egui::RichText::new("Sequence Track:").size(12.5));
            egui::ComboBox::from_id_salt("dyn_relax_metric_combo")
                .selected_text(self.active_relax_metric.label())
                .show_ui(ui, |ui| {
                    for m in SequenceMetricKind::RELAX_METRICS {
                        ui.selectable_value(&mut self.active_relax_metric, m, m.label());
                    }
                });
            ui.separator();
            let decay_lbl = if self.show_r2_decay {
                "R2 Decay Curve"
            } else {
                "R1 Decay Curve"
            };
            egui::ComboBox::from_id_salt("dyn_relax_decay_combo")
                .selected_text(decay_lbl)
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.show_r2_decay, false, "R1 Decay Curve");
                    ui.selectable_value(&mut self.show_r2_decay, true, "R2 Decay Curve");
                });
        });

        if self.relaxation_records.is_empty() {
            ui.label("No per-residue R1 / R2 / hetNOE / Model-Free records loaded.");
            return;
        }

        let bars: Vec<SequenceBarEntry> = self
            .relaxation_records
            .iter()
            .filter_map(|r| {
                let (val, err) = r.metric_value_err(self.active_relax_metric)?;
                Some(SequenceBarEntry {
                    seq_id: r.seq_id,
                    res_name: r.res_name.clone(),
                    value: val,
                    err,
                    highlight: self.active_relax_metric == SequenceMetricKind::ExchangeRex
                        && val >= 2.0,
                })
            })
            .collect();

        let (clicked, bar_rect) = draw_sequence_bar_plot(
            ui,
            self.active_relax_metric.label(),
            &bars,
            self.selected_seq_id,
            &[],
            egui::Color32::from_rgb(56, 189, 248),
        );
        self.record_plot_rect(bar_rect);
        if let Some(clicked_sid) = clicked {
            self.selected_seq_id = Some(clicked_sid);
            output.clicked_seq_id = Some(clicked_sid);
        }

        ui.add_space(6.0);
        let active_rec = self
            .selected_seq_id
            .and_then(|sid| self.relaxation_records.iter().find(|r| r.seq_id == sid))
            .or_else(|| self.relaxation_records.first());

        if let Some(rec) = active_rec {
            let (pts, rate_opt, i0, title) = if self.show_r2_decay {
                (&rec.r2_decay, rec.r2_s_inv, rec.r2_i0, "R2 Transverse")
            } else {
                (&rec.r1_decay, rec.r1_s_inv, rec.r1_i0, "R1 Longitudinal")
            };
            let header = match rate_opt {
                Some((r, e)) => format!(
                    "{title} Exponential Decay — {}{} (R = {:.3} ± {:.3} s⁻¹)",
                    rec.res_name, rec.seq_id, r, e
                ),
                None => format!("{title} Exponential Decay — {}{}", rec.res_name, rec.seq_id),
            };
            let obs: Vec<(f64, f64, f64)> = pts
                .iter()
                .map(|p| (p.delay_s, p.intensity, p.uncertainty))
                .collect();
            let mut curves = Vec::new();
            if let Some((rate, _)) = rate_opt {
                let t_max = pts
                    .iter()
                    .map(|p| p.delay_s)
                    .fold(0.0f64, f64::max)
                    .max(0.1);
                curves.push((
                    build_exponential_decay_curve(i0.max(1e-6), rate, t_max, 80),
                    egui::Color32::from_rgb(52, 211, 153),
                    false,
                ));
            }
            let curve_rect = draw_xy_curve_canvas(
                ui,
                &header,
                "Delay t (s)",
                "Intensity I(t)",
                &[(&obs, egui::Color32::from_rgb(125, 211, 252))],
                &curves,
                &[],
                false,
            );
            self.record_plot_rect(curve_rect);
        }
    }

    fn render_cest_tab(&mut self, ui: &mut egui::Ui, output: &mut DynamicsStudioOutput) {
        let guidance = match self.cest_scheme_tab {
            CestSchemeTab::CwCest => {
                "Input: Multi-B1 CW-CEST saturation profiles (ppm, I/I0) -> Output: Global 2-state p_B, k_ex, dw_AB & fitted curves"
            }
            CestSchemeTab::DanteCest => {
                "Input: DANTE-CEST params (B1, T_EX, N_pulses) or TSV -> Output: Exact 7x7 pulse-train profile, 2-state LM fit & CRLB"
            }
            CestSchemeTab::CosCest => {
                "Input: cos-CEST params (nu_mod, B1, T_EX) or TSV -> Output: Dual-sideband profile, 2-state LM fit & CRLB"
            }
        };
        draw_guidance_header(ui, guidance);

        // Connected folder-tab scheme selector (`vnmrj_rail::draw_folder_tab`)
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 3.0;
            for scheme in CestSchemeTab::ALL {
                if draw_folder_tab(ui, scheme.label(), self.cest_scheme_tab == scheme).clicked() {
                    self.cest_scheme_tab = scheme;
                }
            }
        });
        ui.add_space(2.0);

        match self.cest_scheme_tab {
            CestSchemeTab::CwCest => self.render_cw_cest_body(ui, output),
            CestSchemeTab::DanteCest => self.render_multi_cest_body(ui, CestSchemeTab::DanteCest),
            CestSchemeTab::CosCest => self.render_multi_cest_body(ui, CestSchemeTab::CosCest),
        }
    }

    fn render_cw_cest_body(&mut self, ui: &mut egui::Ui, output: &mut DynamicsStudioOutput) {
        if self.cest_profiles.is_empty() {
            ui.label("No CW-CEST / DEST saturation profiles loaded. Click Load Demo or switch to DANTE-CEST / cos-CEST.");
            return;
        }

        if let Some(fit) = &self.cest_fit {
            draw_static_label(
                ui,
                egui::RichText::new(format!(
                    "Global CW-CEST Fit: p_B = {:.2}% ± {:.2}%  |  k_ex = {:.1} ± {:.1} s⁻¹  |  χ²_red = {:.3}",
                    fit.p_b * 100.0,
                    fit.p_b_err * 100.0,
                    fit.k_ex_s_inv,
                    fit.k_ex_err,
                    fit.reduced_chi2
                ))
                .size(12.0),
            );
            let bars: Vec<SequenceBarEntry> = fit
                .residues
                .iter()
                .map(|r| SequenceBarEntry {
                    seq_id: r.seq_id,
                    res_name: r.res_name.clone(),
                    value: r.dw_ab_ppm.abs(),
                    err: r.dw_ab_ppm_err,
                    highlight: r.dw_ab_ppm.abs() >= 1.0,
                })
                .collect();
            let (clicked, bar_rect) = draw_sequence_bar_plot(
                ui,
                "|Δϖ_AB| (ppm)",
                &bars,
                self.selected_seq_id,
                &[],
                egui::Color32::from_rgb(167, 139, 250),
            );
            self.record_plot_rect(bar_rect);
            if let Some(clicked_sid) = clicked {
                self.selected_seq_id = Some(clicked_sid);
                output.clicked_seq_id = Some(clicked_sid);
            }
        } else {
            let mut seen = BTreeMap::new();
            for p in &self.cest_profiles {
                let depth = 1.0 - p.intensities.iter().copied().fold(1.0f64, f64::min);
                let entry = seen
                    .entry(p.seq_id)
                    .or_insert_with(|| (p.res_name.clone(), 0.0f64));
                entry.1 = entry.1.max(depth);
            }
            let bars: Vec<SequenceBarEntry> = seen
                .into_iter()
                .map(|(sid, (rn, depth))| SequenceBarEntry {
                    seq_id: sid,
                    res_name: rn,
                    value: depth,
                    err: 0.0,
                    highlight: false,
                })
                .collect();
            let (clicked, bar_rect) = draw_sequence_bar_plot(
                ui,
                "Max Dip (1 - I/I0)",
                &bars,
                self.selected_seq_id,
                &[],
                egui::Color32::from_rgb(167, 139, 250),
            );
            self.record_plot_rect(bar_rect);
            if let Some(clicked_sid) = clicked {
                self.selected_seq_id = Some(clicked_sid);
                output.clicked_seq_id = Some(clicked_sid);
            }
        }

        ui.add_space(6.0);
        let Some(first_prof) = self.cest_profiles.first() else {
            return;
        };
        let target_sid = self.selected_seq_id.unwrap_or(first_prof.seq_id);
        let res_profs: Vec<&CestProfileDataset> = self
            .cest_profiles
            .iter()
            .filter(|p| p.seq_id == target_sid)
            .collect();
        let Some(lead_prof) = res_profs.first() else {
            return;
        };

        let palette = [
            egui::Color32::from_rgb(56, 189, 248),
            egui::Color32::from_rgb(251, 146, 60),
            egui::Color32::from_rgb(52, 211, 153),
        ];
        let mut obs_groups: Vec<Vec<(f64, f64, f64)>> = Vec::new();
        let mut curves = Vec::new();
        let mut vlines = Vec::new();

        let res_fit = self
            .cest_fit
            .as_ref()
            .and_then(|f| f.residues.iter().find(|r| r.seq_id == target_sid));

        if let Some(rf) = res_fit {
            vlines.push((rf.major_ppm, egui::Color32::from_rgb(148, 163, 184)));
            vlines.push((
                rf.major_ppm + rf.dw_ab_ppm,
                egui::Color32::from_rgb(244, 114, 182),
            ));
        }

        for (pi, prof) in res_profs.iter().enumerate() {
            let col = palette[pi % palette.len()];
            let pts: Vec<(f64, f64, f64)> = (0..prof.offsets_ppm.len())
                .map(|k| {
                    (
                        prof.offsets_ppm[k],
                        prof.intensities[k],
                        prof.uncertainties[k],
                    )
                })
                .collect();
            obs_groups.push(pts);

            if let (Some(fit), Some(rf)) = (&self.cest_fit, res_fit) {
                let lo = prof
                    .offsets_ppm
                    .iter()
                    .copied()
                    .fold(f64::INFINITY, f64::min);
                let hi = prof
                    .offsets_ppm
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max);
                let dense_offs: Vec<f64> = (0..=120)
                    .map(|k| lo + (hi - lo) * (k as f64) / 120.0)
                    .collect();
                let p = Cest2StateParams {
                    p_b: fit.p_b,
                    k_ex_s_inv: fit.k_ex_s_inv,
                    major_ppm: rf.major_ppm,
                    dw_ab_ppm: rf.dw_ab_ppm,
                    r1a_s_inv: rf.r1a_s_inv,
                    r2a_s_inv: rf.r2a_s_inv,
                    r1b_s_inv: rf.r1a_s_inv,
                    r2b_s_inv: rf.r2b_s_inv,
                    larmor_mhz: prof.larmor_mhz,
                    t_ex_s: prof.t_ex_s,
                    b1_hz: prof.b1_hz,
                    b1_inhomogeneity_frac: prof.b1_inhomogeneity_frac,
                    n_b1_quad: 5,
                    normalization: prof.normalization,
                };
                if let Ok(sim) = simulate_cest_2state_profile(&p, &dense_offs) {
                    let poly: Vec<(f64, f64)> = dense_offs.into_iter().zip(sim).collect();
                    curves.push((poly, col, false));
                }
            }
        }

        let obs_refs: Vec<ObsPointSlice<'_>> = obs_groups
            .iter()
            .enumerate()
            .map(|(i, g)| (g.as_slice(), palette[i % palette.len()]))
            .collect();
        let title = match res_fit {
            Some(rf) => format!(
                "CW-CEST Profile — {}{} (ϖ_A = {:.2} ppm, Δϖ_AB = {:+.2} ppm)",
                lead_prof.res_name, target_sid, rf.major_ppm, rf.dw_ab_ppm
            ),
            None => format!("CW-CEST Profile — {}{}", lead_prof.res_name, target_sid),
        };
        let curve_rect = draw_xy_curve_canvas(
            ui,
            &title,
            "RF Carrier Offset ϖ_rf (ppm)",
            "I / I0",
            &obs_refs,
            &curves,
            &vlines,
            true,
        );
        self.record_plot_rect(curve_rect);
    }

    fn render_multi_cest_body(&mut self, ui: &mut egui::Ui, scheme_tab: CestSchemeTab) {
        // Ensure a live forward preview curve is rendered immediately on first visit
        let needs_preview = match scheme_tab {
            CestSchemeTab::DanteCest => {
                self.multi_cest.dante_sim_curves.is_empty()
                    && self.multi_cest.dante_profiles.is_empty()
            }
            CestSchemeTab::CosCest => {
                self.multi_cest.cos_sim_curves.is_empty() && self.multi_cest.cos_profiles.is_empty()
            }
            CestSchemeTab::CwCest => false,
        };
        if needs_preview {
            let _ = self.simulate_multi_cest_preview(scheme_tab);
        }

        let is_demo = match scheme_tab {
            CestSchemeTab::DanteCest => self.multi_cest.dante_is_synthetic_demo,
            CestSchemeTab::CosCest => self.multi_cest.cos_is_synthetic_demo,
            CestSchemeTab::CwCest => false,
        };

        // Compact primary control strip (only 2 controls when Params & TSV drawer is collapsed)
        ui.horizontal_wrapped(|ui| {
            let fit_btn_label = if is_demo {
                "Fit 2-State [Synthetic Demo]"
            } else {
                "Fit 2-State (LM)"
            };
            if ui
                .button(fit_btn_label)
                .on_hover_text(
                    "Run exact 7x7 Bloch-McConnell Levenberg-Marquardt fit with Cramer-Rao covariance",
                )
                .clicked()
            {
                let _ = self.fit_multi_cest_2state(scheme_tab);
            }

            let edit_lbl = if self.multi_cest.show_params_editor {
                "Hide Params & TSV"
            } else {
                "Params & TSV"
            };
            if ui
                .selectable_label(self.multi_cest.show_params_editor, edit_lbl)
                .on_hover_text(
                    "Non-destructively edit B1, T_EX, N_pulses, nu_mod, noise, seed, or paste TSV",
                )
                .clicked()
            {
                self.multi_cest.show_params_editor = !self.multi_cest.show_params_editor;
            }

            ui.separator();
            let active_imported_b1s: Vec<f64> = if !is_demo {
                let profs = match scheme_tab {
                    CestSchemeTab::DanteCest => &self.multi_cest.dante_profiles,
                    CestSchemeTab::CosCest => &self.multi_cest.cos_profiles,
                    CestSchemeTab::CwCest => &[][..],
                };
                profs.iter().map(|p| p.scheme.effective_b1_hz()).collect()
            } else {
                Vec::new()
            };
            let b1_desc = if active_imported_b1s.is_empty() {
                format!("{:.1} Hz", self.multi_cest.b1_hz)
            } else if active_imported_b1s.len() == 1 {
                format!("{:.1} Hz", active_imported_b1s[0])
            } else {
                let parts: Vec<String> = active_imported_b1s
                    .iter()
                    .map(|v| format!("{v:.1}"))
                    .collect();
                format!("[{}] Hz", parts.join(", "))
            };
            let param_summary = match scheme_tab {
                CestSchemeTab::DanteCest => {
                    let comb_hz = f64::from(self.multi_cest.n_pulses.max(1))
                        / self.multi_cest.t_ex_s.max(1e-6);
                    format!(
                        "B1 = {b1_desc} | T_EX = {:.2} s | N_pulses = {} | comb = {:.0} Hz",
                        self.multi_cest.t_ex_s, self.multi_cest.n_pulses, comb_hz
                    )
                }
                CestSchemeTab::CosCest => {
                    let sb_ppm =
                        self.multi_cest.mod_freq_hz / self.multi_cest.larmor_mhz.max(1.0);
                    format!(
                        "nu_mod = {:.0} Hz (±{:.2} ppm) | B1 = {b1_desc} | T_EX = {:.2} s",
                        self.multi_cest.mod_freq_hz, sb_ppm, self.multi_cest.t_ex_s
                    )
                }
                CestSchemeTab::CwCest => String::new(),
            };
            draw_static_label(
                ui,
                egui::RichText::new(param_summary)
                    .size(11.5)
                    .color(egui::Color32::from_rgb(148, 163, 184)),
            );
            if is_demo {
                draw_static_label(
                    ui,
                    egui::RichText::new("[Synthetic Demo]")
                        .size(11.5)
                        .strong()
                        .color(egui::Color32::from_rgb(251, 191, 36)),
                );
            }
        });

        // Non-destructively editable parameter & TSV drawer
        if self.multi_cest.show_params_editor {
            egui::Frame::group(ui.style()).show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.label(egui::RichText::new("b1_hz:").size(12.0));
                    ui.add(
                        egui::DragValue::new(&mut self.multi_cest.b1_hz)
                            .speed(0.5)
                            .range(1.0..=200.0)
                            .suffix(" Hz"),
                    )
                    .on_hover_text(
                        "Effective B1 for synthetic demos or 3-column TSV imports; imported profiles with per-profile B1 keep their own B1 on refit",
                    );
                    ui.checkbox(&mut self.multi_cest.use_dual_b1, "Dual B1");
                    if self.multi_cest.use_dual_b1 {
                        ui.add(
                            egui::DragValue::new(&mut self.multi_cest.b1_second_hz)
                                .speed(0.5)
                                .range(1.0..=200.0)
                                .suffix(" Hz"),
                        );
                    }
                    ui.label(egui::RichText::new("t_ex_s:").size(12.0));
                    ui.add(
                        egui::DragValue::new(&mut self.multi_cest.t_ex_s)
                            .speed(0.01)
                            .range(0.05..=2.0)
                            .suffix(" s"),
                    );
                    match scheme_tab {
                        CestSchemeTab::DanteCest => {
                            ui.label(egui::RichText::new("n_pulses:").size(12.0));
                            ui.add(
                                egui::DragValue::new(&mut self.multi_cest.n_pulses)
                                    .speed(2.0)
                                    .range(4..=2000),
                            );
                            ui.label(egui::RichText::new("tau_p:").size(12.0));
                            ui.add(
                                egui::DragValue::new(&mut self.multi_cest.tau_pulse_us)
                                    .speed(0.5)
                                    .range(2.0..=200.0)
                                    .suffix(" us"),
                            );
                        }
                        CestSchemeTab::CosCest => {
                            ui.label(egui::RichText::new("mod_freq_hz:").size(12.0));
                            ui.add(
                                egui::DragValue::new(&mut self.multi_cest.mod_freq_hz)
                                    .speed(5.0)
                                    .range(20.0..=5000.0)
                                    .suffix(" Hz"),
                            );
                            ui.label(egui::RichText::new("steps/cyc:").size(12.0));
                            ui.add(
                                egui::DragValue::new(&mut self.multi_cest.n_steps_per_cycle)
                                    .speed(1.0)
                                    .range(8..=128),
                            );
                        }
                        CestSchemeTab::CwCest => {}
                    }
                });
                ui.horizontal_wrapped(|ui| {
                    ui.label(egui::RichText::new("major_ppm:").size(12.0));
                    ui.add(
                        egui::DragValue::new(&mut self.multi_cest.major_ppm)
                            .speed(0.1)
                            .range(90.0..=145.0),
                    );
                    ui.label(egui::RichText::new("larmor:").size(12.0));
                    ui.add(
                        egui::DragValue::new(&mut self.multi_cest.larmor_mhz)
                            .speed(0.5)
                            .range(20.0..=150.0)
                            .suffix(" MHz"),
                    );
                    ui.label(egui::RichText::new("noise:").size(12.0));
                    ui.add(
                        egui::DragValue::new(&mut self.multi_cest.demo_noise_sigma)
                            .speed(0.0005)
                            .range(0.0005..=0.05),
                    );
                    ui.label(egui::RichText::new("seed:").size(12.0));
                    ui.add(egui::DragValue::new(&mut self.multi_cest.demo_seed).speed(1.0));
                    if ui.button("Simulate Curve").clicked() {
                        let _ = self.simulate_multi_cest_preview(scheme_tab);
                    }
                    if ui.button("Load [Synthetic Demo]").clicked() {
                        let _ = self.load_multi_cest_synthetic_demo(scheme_tab);
                    }
                });
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("TSV (offset_ppm\\tintensity\\tsigma[\\tb1_hz]):")
                            .size(11.5),
                    );
                    if ui.button("Import TSV & Fit").clicked() {
                        let buf = self.multi_cest.tsv_input.clone();
                        let _ = self.import_multi_cest_tsv(scheme_tab, &buf);
                    }
                });
                egui::ScrollArea::vertical()
                    .id_salt("multi_cest_tsv_scroll")
                    .max_height(68.0)
                    .show(ui, |ui| {
                        ui.add(
                            egui::TextEdit::multiline(&mut self.multi_cest.tsv_input)
                                .desired_rows(3)
                                .desired_width(f32::INFINITY)
                                .font(egui::TextStyle::Monospace),
                        );
                    });
            });
        }

        // Honest fit readout + Option CRLB covariance status (`covariance_valid`)
        if let Some(err) = &self.multi_cest.last_error {
            draw_static_label(
                ui,
                egui::RichText::new(format!("Error: {err}"))
                    .size(12.0)
                    .color(egui::Color32::from_rgb(248, 113, 113)),
            );
        }
        let active_fit = match scheme_tab {
            CestSchemeTab::DanteCest => self.multi_cest.dante_fit.as_ref(),
            CestSchemeTab::CosCest => self.multi_cest.cos_fit.as_ref(),
            CestSchemeTab::CwCest => None,
        };
        if let Some(fit) = active_fit {
            let readout = format_multi_cest_summary(scheme_tab, fit, is_demo);
            let col = if fit.covariance_valid {
                egui::Color32::from_rgb(167, 243, 208)
            } else {
                egui::Color32::from_rgb(251, 191, 36)
            };
            draw_static_label(ui, egui::RichText::new(readout).size(12.0).color(col));
        } else {
            draw_static_label(
                ui,
                egui::RichText::new(
                    "Forward preview active — click Fit 2-State to run LM optimization and Cramer-Rao covariance.",
                )
                .size(11.5)
                .color(egui::Color32::from_rgb(148, 163, 184)),
            );
        }

        // Plot observed points + fitted/simulated multi-frequency curve directly on the CEST canvas
        let palette = [
            egui::Color32::from_rgb(56, 189, 248),
            egui::Color32::from_rgb(251, 146, 60),
            egui::Color32::from_rgb(52, 211, 153),
        ];
        let (profiles, sim_curves) = match scheme_tab {
            CestSchemeTab::DanteCest => (
                &self.multi_cest.dante_profiles,
                &self.multi_cest.dante_sim_curves,
            ),
            CestSchemeTab::CosCest => (
                &self.multi_cest.cos_profiles,
                &self.multi_cest.cos_sim_curves,
            ),
            CestSchemeTab::CwCest => return,
        };

        let mut obs_groups: Vec<Vec<(f64, f64, f64)>> = Vec::with_capacity(profiles.len());
        for prof in profiles {
            let pts: Vec<(f64, f64, f64)> = (0..prof.offsets_ppm.len())
                .map(|k| {
                    (
                        prof.offsets_ppm[k],
                        prof.intensities[k],
                        prof.uncertainties[k],
                    )
                })
                .collect();
            obs_groups.push(pts);
        }
        let obs_refs: Vec<ObsPointSlice<'_>> = obs_groups
            .iter()
            .enumerate()
            .map(|(i, g)| (g.as_slice(), palette[i % palette.len()]))
            .collect();

        let curves: Vec<StyledPolyline> = sim_curves
            .iter()
            .enumerate()
            .map(|(i, (_, poly))| (poly.clone(), palette[i % palette.len()], false))
            .collect();

        let mut vlines = vec![(
            self.multi_cest.major_ppm,
            egui::Color32::from_rgb(148, 163, 184),
        )];
        let dw_guide = active_fit
            .map(|f| f.params.dw_ab_ppm)
            .unwrap_or(self.multi_cest.demo_dw_ab_ppm);
        vlines.push((
            self.multi_cest.major_ppm + dw_guide,
            egui::Color32::from_rgb(244, 114, 182),
        ));
        if scheme_tab == CestSchemeTab::CosCest {
            let sb_ppm = self.multi_cest.mod_freq_hz / self.multi_cest.larmor_mhz.max(1.0);
            vlines.push((
                self.multi_cest.major_ppm + sb_ppm,
                egui::Color32::from_rgb(96, 165, 250),
            ));
            vlines.push((
                self.multi_cest.major_ppm - sb_ppm,
                egui::Color32::from_rgb(96, 165, 250),
            ));
        }

        let demo_tag = if is_demo { " [Synthetic Demo]" } else { "" };
        let title = match active_fit {
            Some(fit) => format!(
                "{} Profile{demo_tag} — p_B={:.2}%, k_ex={:.0} s⁻¹, Δϖ={:+.2} ppm",
                scheme_tab.label(),
                fit.params.p_b * 100.0,
                fit.params.k_ex_s_inv,
                fit.params.dw_ab_ppm
            ),
            None => format!(
                "{} Forward Preview — ϖ_A={:.1} ppm, B1={:.1} Hz",
                scheme_tab.label(),
                self.multi_cest.major_ppm,
                self.multi_cest.b1_hz
            ),
        };
        let curve_rect = draw_xy_curve_canvas(
            ui,
            &title,
            "RF Carrier Offset ϖ_rf (ppm)",
            "I / I0",
            &obs_refs,
            &curves,
            &vlines,
            true,
        );
        self.record_plot_rect(curve_rect);
    }

    fn render_cpmg_tab(&mut self, ui: &mut egui::Ui, output: &mut DynamicsStudioOutput) {
        draw_guidance_header(
            ui,
            "Input: Multi-field CPMG R2,eff(nu_CPMG) curves -> Output: Global p_B, k_ex, |dw| & Baldwin2014 dispersion fits",
        );
        if self.cpmg_series.is_empty() {
            ui.label("No CPMG relaxation dispersion series loaded.");
            return;
        }

        if let Some(fit) = &self.cpmg_fit {
            draw_static_label(
                ui,
                egui::RichText::new(format!(
                    "Global CPMG ({}) Fit: p_B = {:.2}% ± {:.2}%  |  k_ex = {:.0} ± {:.0} s⁻¹  |  χ²_red = {:.3}",
                    fit.model.as_str(),
                    fit.p_b * 100.0,
                    fit.p_b_err * 100.0,
                    fit.k_ex_s_inv,
                    fit.k_ex_err,
                    fit.reduced_chi2
                ))
                .size(12.0),
            );
        }

        let mut rex_map: BTreeMap<i32, (String, f64, bool)> = BTreeMap::new();
        for s in &self.cpmg_series {
            let r_min = s.r2_eff_s_inv.iter().copied().fold(f64::INFINITY, f64::min);
            let r_max = s
                .r2_eff_s_inv
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let rex = (r_max - r_min).max(0.0);
            let e = rex_map
                .entry(s.seq_id)
                .or_insert_with(|| (s.res_name.clone(), 0.0, rex >= 2.0));
            e.1 = e.1.max(rex);
            e.2 = e.1 >= 2.0;
        }
        if let Some(fit) = &self.cpmg_fit {
            for rf in &fit.residues {
                rex_map.insert(
                    rf.seq_id,
                    (
                        rf.res_name.clone(),
                        rf.gate.observed_rex_s_inv,
                        rf.gate.significant_exchange,
                    ),
                );
            }
        }

        let bars: Vec<SequenceBarEntry> = rex_map
            .into_iter()
            .map(|(sid, (rn, rex, sig))| SequenceBarEntry {
                seq_id: sid,
                res_name: rn,
                value: rex,
                err: 0.0,
                highlight: sig,
            })
            .collect();

        let (clicked, bar_rect) = draw_sequence_bar_plot(
            ui,
            "ΔR2,eff (s⁻¹)",
            &bars,
            self.selected_seq_id,
            &[(2.0, egui::Color32::from_rgb(251, 191, 36))],
            egui::Color32::from_rgb(52, 211, 153),
        );
        self.record_plot_rect(bar_rect);
        if let Some(clicked_sid) = clicked {
            self.selected_seq_id = Some(clicked_sid);
            output.clicked_seq_id = Some(clicked_sid);
        }

        ui.add_space(6.0);
        let Some(first_series) = self.cpmg_series.first() else {
            return;
        };
        let target_sid = self.selected_seq_id.unwrap_or(first_series.seq_id);
        let res_series: Vec<&CpmgSeriesDataset> = self
            .cpmg_series
            .iter()
            .filter(|s| s.seq_id == target_sid)
            .collect();
        let Some(lead_series) = res_series.first() else {
            return;
        };

        let palette = [
            egui::Color32::from_rgb(56, 189, 248),
            egui::Color32::from_rgb(251, 146, 60),
            egui::Color32::from_rgb(167, 139, 250),
        ];
        let mut obs_groups = Vec::new();
        let mut curves = Vec::new();
        let res_fit = self
            .cpmg_fit
            .as_ref()
            .and_then(|f| f.residues.iter().find(|r| r.seq_id == target_sid));

        for (fi, s) in res_series.iter().enumerate() {
            let col = palette[fi % palette.len()];
            let pts: Vec<(f64, f64, f64)> = (0..s.nu_cpmg_hz.len())
                .map(|k| (s.nu_cpmg_hz[k], s.r2_eff_s_inv[k], s.r2_eff_err[k]))
                .collect();
            obs_groups.push(pts);

            if let (Some(fit), Some(rf)) = (&self.cpmg_fit, res_fit) {
                let r20 = rf
                    .field_r20
                    .iter()
                    .find(|&&(mhz, _, _)| (mhz - s.larmor_mhz).abs() < 1e-3)
                    .map(|&(_, v, _)| v)
                    .unwrap_or(12.0);
                let nu_min = s
                    .nu_cpmg_hz
                    .iter()
                    .copied()
                    .fold(f64::INFINITY, f64::min)
                    .max(10.0);
                let nu_max = s
                    .nu_cpmg_hz
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max)
                    .max(nu_min + 50.0);
                let dw_rad = TAU * s.larmor_mhz * rf.dw_ppm;
                let poly: Vec<(f64, f64)> = (0..=80)
                    .map(|k| {
                        let nu = nu_min + (nu_max - nu_min) * (k as f64) / 80.0;
                        let r2 = evaluate_cpmg_model(
                            fit.model,
                            r20,
                            r20,
                            fit.p_b,
                            fit.k_ex_s_inv,
                            dw_rad,
                            s.t_relax_s,
                            nu,
                        );
                        (nu, r2)
                    })
                    .collect();
                curves.push((poly, col, false));
            }
        }

        let obs_refs: Vec<ObsPointSlice<'_>> = obs_groups
            .iter()
            .enumerate()
            .map(|(i, g)| (g.as_slice(), palette[i % palette.len()]))
            .collect();
        let title = match res_fit {
            Some(rf) => format!(
                "CPMG Relaxation Dispersion — {}{} (|Δω| = {:.2} ± {:.2} ppm)",
                lead_series.res_name, target_sid, rf.dw_ppm, rf.dw_ppm_err
            ),
            None => format!(
                "CPMG Relaxation Dispersion — {}{} (Non-exchanging / Flat)",
                lead_series.res_name, target_sid
            ),
        };
        let curve_rect = draw_xy_curve_canvas(
            ui,
            &title,
            "ν_CPMG (Hz)",
            "R2,eff (s⁻¹)",
            &obs_refs,
            &curves,
            &[],
            false,
        );
        self.record_plot_rect(curve_rect);
    }

    fn render_csp_tab(&mut self, ui: &mut egui::Ui, output: &mut DynamicsStudioOutput) {
        let guidance = match self.csp_sub_tab {
            CspSubTab::FastExchangeCsp => {
                "Input: Fast-exchange HSQC titration trajectories -> Output: Williamson 2-sigma pocket sites & tight-binding Kd"
            }
            CspSubTab::Titration2dLineshape => {
                "Input: Multi-step 2D (1H, 15N) titration TSV or [Synthetic Demo] -> Output: Tracked paths, |dw|/k_off regime & 2D (Kd, k_off, dwH, dwN)"
            }
        };
        draw_guidance_header(ui, guidance);

        // Connected folder-tab mode toggle (`vnmrj_rail::draw_folder_tab`)
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 3.0;
            for sub in CspSubTab::ALL {
                if draw_folder_tab(ui, sub.label(), self.csp_sub_tab == sub).clicked() {
                    self.csp_sub_tab = sub;
                }
            }
        });
        ui.add_space(2.0);

        match self.csp_sub_tab {
            CspSubTab::FastExchangeCsp => self.render_fast_exchange_csp_body(ui, output),
            CspSubTab::Titration2dLineshape => self.render_titration_2d_body(ui),
        }
    }

    fn render_fast_exchange_csp_body(
        &mut self,
        ui: &mut egui::Ui,
        output: &mut DynamicsStudioOutput,
    ) {
        if self.csp_trajectories.is_empty() {
            ui.label("No ligand titration HSQC CSP trajectories loaded. Click Load Demo or switch to 2D Titration Lineshape (T22.8).");
            return;
        }

        if let Some(fit) = &self.csp_fit {
            draw_static_label(
                ui,
                egui::RichText::new(format!(
                    "Global Quadratic Tight-Binding Kd = {:.2} µM (95% CI [{:.2}, {:.2}] µM)  |  Naive MM Kd = {:.2} µM",
                    fit.global_kd_um,
                    fit.global_kd_ci95_low_um,
                    fit.global_kd_ci95_high_um,
                    fit.global_naive_mm_kd_um,
                ))
                .size(12.0),
            );
        }

        let mut hlines = Vec::new();
        let thresh_2s = self
            .csp_threshold
            .as_ref()
            .map(|t| {
                hlines.push((
                    t.threshold_1sigma_ppm,
                    egui::Color32::from_rgb(148, 163, 184),
                ));
                hlines.push((
                    t.threshold_2sigma_ppm,
                    egui::Color32::from_rgb(248, 113, 113),
                ));
                t.threshold_2sigma_ppm
            })
            .unwrap_or(f64::INFINITY);

        let bars: Vec<SequenceBarEntry> = self
            .csp_trajectories
            .iter()
            .map(|t| SequenceBarEntry {
                seq_id: t.seq_id,
                res_name: t.res_name.clone(),
                value: t.max_observed_csp_ppm,
                err: 0.0,
                highlight: t.max_observed_csp_ppm >= thresh_2s,
            })
            .collect();

        let (clicked, bar_rect) = draw_sequence_bar_plot(
            ui,
            "Δδ_comb (ppm)",
            &bars,
            self.selected_seq_id,
            &hlines,
            egui::Color32::from_rgb(56, 189, 248),
        );
        self.record_plot_rect(bar_rect);
        if let Some(clicked_sid) = clicked {
            self.selected_seq_id = Some(clicked_sid);
            output.clicked_seq_id = Some(clicked_sid);
        }

        ui.add_space(6.0);
        let Some(first_traj) = self.csp_trajectories.first() else {
            return;
        };
        let target_sid = self.selected_seq_id.unwrap_or(first_traj.seq_id);
        let Some(traj) = self
            .csp_trajectories
            .iter()
            .find(|t| t.seq_id == target_sid)
        else {
            return;
        };

        let obs_pl_y: Vec<(f64, f64, f64)> = traj
            .points
            .iter()
            .filter_map(|p| p.csp_ppm.map(|y| (p.protein_conc_um, p.ligand_conc_um, y)))
            .collect();
        let obs_plot: Vec<(f64, f64, f64)> =
            obs_pl_y.iter().map(|&(_p, l, y)| (l, y, 0.004)).collect();
        let l_max = obs_plot
            .iter()
            .map(|&(l, _, _)| l)
            .fold(0.0f64, f64::max)
            .max(10.0);
        let p_nom = traj
            .points
            .first()
            .map(|p| p.protein_conc_um)
            .unwrap_or(100.0);

        let mut curves = Vec::new();
        let res_kd = self
            .csp_fit
            .as_ref()
            .and_then(|f| f.residue_fits.iter().find(|r| r.seq_id == target_sid));
        let global_dmax = self
            .csp_fit
            .as_ref()
            .and_then(|f| {
                f.global_ddelta_max_by_seq_id
                    .iter()
                    .find(|&&(sid, _, _)| sid == target_sid)
                    .map(|&(_, dm, _)| dm)
            })
            .or_else(|| res_kd.map(|r| r.ddelta_max_ppm));

        if let (Some(fit), Some(dmax)) = (&self.csp_fit, global_dmax) {
            let quad_poly =
                build_tight_binding_isotherm_curve(p_nom, fit.global_kd_um, dmax, l_max, 100);
            curves.push((quad_poly, egui::Color32::from_rgb(52, 211, 153), false));
        }

        if self.show_naive_mm_comparison && obs_pl_y.len() >= 3 {
            let slice_refs = [obs_pl_y.as_slice()];
            let (mm_kd, mm_dmaxs, _) = fit_naive_michaelis_menten_kd(&slice_refs, 0.004);
            let mm_dmax = mm_dmaxs.first().copied().unwrap_or(0.0);
            let mm_poly: Vec<(f64, f64)> = (0..=100)
                .map(|k| {
                    let l = l_max * (k as f64) / 100.0;
                    (l, naive_michaelis_menten_csp(mm_dmax, l, mm_kd))
                })
                .collect();
            curves.push((mm_poly, egui::Color32::from_rgb(248, 113, 113), true));
        }

        let title = match (&self.csp_fit, global_dmax) {
            (Some(fit), Some(dmax)) => format!(
                "Binding Isotherm — {}{} (Quadratic Kd = {:.2} µM, Δδ_max = {:.3} ppm)",
                traj.res_name, target_sid, fit.global_kd_um, dmax
            ),
            _ => format!("Binding Isotherm — {}{}", traj.res_name, target_sid),
        };
        let curve_rect = draw_xy_curve_canvas(
            ui,
            &title,
            "Total Ligand [L]_t (µM)",
            "Δδ_comb (ppm)",
            &[(&obs_plot, egui::Color32::from_rgb(125, 211, 252))],
            &curves,
            &[],
            false,
        );
        self.record_plot_rect(curve_rect);
    }

    fn render_titration_2d_body(&mut self, ui: &mut egui::Ui) {
        // Compact primary control strip (3 controls when Params & TSV drawer is collapsed)
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("Load [Synthetic Demo]")
                .on_hover_text(
                    "Load 6-step crossing + dropout 2D titration series and fit coupled 2D Bloch-McConnell lineshapes",
                )
                .clicked()
            {
                let _ = self.load_titration_2d_synthetic_demo();
            }
            if ui
                .button("Track & Fit 2D")
                .on_hover_text(
                    "Re-run velocity-extrapolated trajectory tracking, |dw|/k_off regime audit, and 2D lineshape fit",
                )
                .clicked()
            {
                let _ = self.run_titration_2d_analysis();
            }
            let edit_lbl = if self.titration_2d.show_params_editor {
                "Hide Params & TSV"
            } else {
                "Params & TSV"
            };
            if ui
                .selectable_label(self.titration_2d.show_params_editor, edit_lbl)
                .on_hover_text(
                    "Non-destructively edit 2D titration TSV, spectrometer MHz, [P]_t, tracking gates, and demo seeds",
                )
                .clicked()
            {
                self.titration_2d.show_params_editor = !self.titration_2d.show_params_editor;
            }

            if self.titration_2d.is_synthetic_demo {
                ui.separator();
                draw_static_label(
                    ui,
                    egui::RichText::new("[Synthetic Demo]")
                        .size(11.5)
                        .strong()
                        .color(egui::Color32::from_rgb(251, 191, 36)),
                );
            }
        });

        // Non-destructively editable parameter & TSV drawer
        if self.titration_2d.show_params_editor {
            egui::Frame::group(ui.style()).show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.label(egui::RichText::new("1H MHz:").size(12.0));
                    ui.add(
                        egui::DragValue::new(&mut self.titration_2d.h_larmor_mhz)
                            .speed(10.0)
                            .range(200.0..=1200.0),
                    );
                    ui.label(egui::RichText::new("[P]_t:").size(12.0));
                    ui.add(
                        egui::DragValue::new(&mut self.titration_2d.protein_conc_um)
                            .speed(2.0)
                            .range(1.0..=2000.0)
                            .suffix(" uM"),
                    );
                    ui.label(egui::RichText::new("max_step:").size(12.0));
                    ui.add(
                        egui::DragValue::new(&mut self.titration_2d.max_step_ppm)
                            .speed(0.01)
                            .range(0.05..=2.0)
                            .suffix(" ppm"),
                    );
                    ui.label(egui::RichText::new("alpha_N:").size(12.0));
                    ui.add(
                        egui::DragValue::new(&mut self.titration_2d.n_scale_factor)
                            .speed(0.01)
                            .range(0.05..=0.50),
                    );
                });
                ui.horizontal_wrapped(|ui| {
                    ui.label(egui::RichText::new("Demo Kd:").size(12.0));
                    ui.add(
                        egui::DragValue::new(&mut self.titration_2d.demo_kd_um)
                            .speed(1.0)
                            .range(1.0..=1000.0)
                            .suffix(" uM"),
                    );
                    ui.label(egui::RichText::new("k_off:").size(12.0));
                    ui.add(
                        egui::DragValue::new(&mut self.titration_2d.demo_k_off_s_inv)
                            .speed(5.0)
                            .range(5.0..=10_000.0)
                            .suffix(" s⁻¹"),
                    );
                    ui.label(egui::RichText::new("SNR:").size(12.0));
                    ui.add(
                        egui::DragValue::new(&mut self.titration_2d.demo_snr)
                            .speed(2.0)
                            .range(10.0..=500.0),
                    );
                    ui.label(egui::RichText::new("seed:").size(12.0));
                    ui.add(egui::DragValue::new(&mut self.titration_2d.demo_seed).speed(1.0));
                    if let Some(fit) = &self.titration_2d.lineshape_fit {
                        if !fit.residues.is_empty() {
                            ui.separator();
                            let sel_r = self
                                .titration_2d
                                .selected_residue_idx
                                .min(fit.residues.len().saturating_sub(1));
                            egui::ComboBox::from_id_salt("t228_res_sel")
                                .selected_text(&fit.residues[sel_r].label)
                                .show_ui(ui, |ui| {
                                    for (ri, r) in fit.residues.iter().enumerate() {
                                        ui.selectable_value(
                                            &mut self.titration_2d.selected_residue_idx,
                                            ri,
                                            &r.label,
                                        );
                                    }
                                });
                            if let Some(series) = self.titration_2d.roi_series.get(sel_r) {
                                let max_s = series.patches.len().saturating_sub(1);
                                let mut s_idx = self.titration_2d.selected_step_idx.min(max_s);
                                ui.label(egui::RichText::new("Step:").size(12.0));
                                ui.add(
                                    egui::DragValue::new(&mut s_idx)
                                        .speed(0.2)
                                        .range(0..=max_s),
                                );
                                self.titration_2d.selected_step_idx = s_idx;
                            }
                        }
                    }
                });
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new(
                            "TSV (step_idx\\tlabel\\t[L]_uM\\t[P]_uM\\tpeak_id\\th_ppm\\tn_ppm[\\tres_lbl\\tres]):",
                        )
                        .size(11.5),
                    );
                    if ui.button("Import TSV & Track").clicked() {
                        let buf = self.titration_2d.tsv_input.clone();
                        let _ = self.import_titration_2d_tsv(&buf);
                    }
                });
                egui::ScrollArea::vertical()
                    .id_salt("titration_2d_tsv_scroll")
                    .max_height(68.0)
                    .show(ui, |ui| {
                        ui.add(
                            egui::TextEdit::multiline(&mut self.titration_2d.tsv_input)
                                .desired_rows(3)
                                .desired_width(f32::INFINITY)
                                .font(egui::TextStyle::Monospace),
                        );
                    });
            });
        }

        if let Some(err) = &self.titration_2d.last_error {
            draw_static_label(
                ui,
                egui::RichText::new(format!("Error: {err}"))
                    .size(12.0)
                    .color(egui::Color32::from_rgb(248, 113, 113)),
            );
        }

        // Fit & exchange-regime summary header
        if let Some(fit) = &self.titration_2d.lineshape_fit {
            let regime_badge = if fit.fast_exchange_comparison.fast_exchange_assumption_valid {
                "Fast-CSP Valid: Yes"
            } else {
                "Fast-CSP Valid: No (use 2D Bloch-McConnell)"
            };
            let demo_prefix = if self.titration_2d.is_synthetic_demo {
                "[Synthetic Demo] "
            } else {
                ""
            };
            draw_static_label(
                ui,
                egui::RichText::new(format!(
                    "{demo_prefix}2D Fit: Kd = {:.2} ± {:.2} µM | k_off = {:.1} ± {:.1} s⁻¹ | k_on = {:.2} µM⁻¹s⁻¹ | χ²_red = {:.3} | {regime_badge}",
                    fit.kd_um,
                    fit.kd_sd_um,
                    fit.k_off_s_inv,
                    fit.k_off_sd_s_inv,
                    fit.k_on_um_inv_s_inv,
                    fit.reduced_chi2,
                ))
                .size(12.0)
                .color(egui::Color32::from_rgb(167, 243, 208)),
            );
        } else if let Some(tr) = &self.titration_2d.tracking_result {
            let n_drop: usize = tr.trajectories.iter().map(|t| t.n_dropouts).sum();
            draw_static_label(
                ui,
                egui::RichText::new(format!(
                    "Tracked {} trajectories across {} steps ({n_drop} dropouts, {} ambiguity flags) | Regime audit at Kd={:.1} µM, k_off={:.0} s⁻¹",
                    tr.trajectories.len(),
                    self.titration_2d.series_steps.len(),
                    tr.ambiguity_flags.len(),
                    self.titration_2d.initial_kd_hint_um,
                    self.titration_2d.initial_k_off_hint_s_inv,
                ))
                .size(12.0)
                .color(egui::Color32::from_rgb(125, 211, 252)),
            );
        } else {
            draw_static_label(
                ui,
                egui::RichText::new(
                    "Click Load [Synthetic Demo] or open Params & TSV to paste a multi-step 2D (1H, 15N) titration peak series.",
                )
                .size(11.5)
                .color(egui::Color32::from_rgb(148, 163, 184)),
            );
        }

        // Main dual-pane canvas (left: 2D trajectory map; right: 2D Bloch-McConnell lineshape & |dw|/k_off diagnostics)
        let avail = ui.available_size();
        let canvas_h = (avail.y - 4.0).max(260.0);
        let gap = 8.0_f32;
        let half_w = ((avail.x - gap) * 0.5).max(220.0);

        let mut left_rect = egui::Rect::NOTHING;
        let mut right_rect = egui::Rect::NOTHING;
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = gap;
            left_rect = draw_titration_2d_trajectory_canvas(
                ui,
                egui::vec2(half_w, canvas_h),
                self.titration_2d.tracking_result.as_ref(),
                self.titration_2d.is_synthetic_demo,
            );
            right_rect = draw_titration_2d_lineshape_canvas(
                ui,
                egui::vec2(half_w, canvas_h),
                &self.titration_2d,
            );
        });
        if left_rect.is_positive() {
            self.record_plot_rect(left_rect);
        }
        if right_rect.is_positive() {
            self.record_plot_rect(right_rect);
        }
    }
}

fn draw_static_label(ui: &mut egui::Ui, text: impl Into<egui::WidgetText>) {
    ui.add(egui::Label::new(text).selectable(false));
}

fn draw_guidance_header(ui: &mut egui::Ui, text: &str) {
    draw_static_label(
        ui,
        egui::RichText::new(text)
            .size(11.5)
            .color(egui::Color32::from_rgb(148, 163, 184)),
    );
}

fn format_multi_cest_summary(
    scheme_tab: CestSchemeTab,
    fit: &MultiCestFitResult,
    is_demo: bool,
) -> String {
    let prefix = if is_demo { "[Synthetic Demo] " } else { "" };
    if fit.covariance_valid {
        if let Some(se) = fit.std_errors {
            let crlb_tag = if fit.covariance_regularized {
                "CRLB: Valid (regularized)"
            } else {
                "CRLB: Valid"
            };
            return format!(
                "{prefix}{} Fit: p_B = {:.2}% ± {:.2}% | k_ex = {:.1} ± {:.1} s⁻¹ | Δϖ = {:+.2} ± {:.2} ppm | χ²_red = {:.3} | {crlb_tag}",
                scheme_tab.label(),
                fit.params.p_b * 100.0,
                se[0] * 100.0,
                fit.params.k_ex_s_inv,
                se[1],
                fit.params.dw_ab_ppm,
                se[2],
                fit.reduced_chi2,
            );
        }
    }
    format!(
        "{prefix}{} Fit: p_B = {:.2}% (SE N/A) | k_ex = {:.1} s⁻¹ (SE N/A) | Δϖ = {:+.2} ppm (SE N/A) | χ²_red = {:.3} | CRLB: Unavailable (rank-deficient Fisher)",
        scheme_tab.label(),
        fit.params.p_b * 100.0,
        fit.params.k_ex_s_inv,
        fit.params.dw_ab_ppm,
        fit.reduced_chi2,
    )
}

/// Sample a smooth exponential relaxation decay curve `I(t) = I0 * exp(-rate * t)` over `[0, t_max]`.
pub fn build_exponential_decay_curve(
    i0: f64,
    rate_s_inv: f64,
    t_max_s: f64,
    n_samples: usize,
) -> Vec<(f64, f64)> {
    let n = n_samples.max(2);
    (0..=n)
        .map(|k| {
            let t = t_max_s * (k as f64) / (n as f64);
            (t, i0 * (-rate_s_inv * t).exp())
        })
        .collect()
}

/// Sample an exact quadratic tight-binding `1:1` isotherm curve over `[0, l_max_um]`.
pub fn build_tight_binding_isotherm_curve(
    protein_um: f64,
    kd_um: f64,
    delta_max_ppm: f64,
    l_max_um: f64,
    n_samples: usize,
) -> Vec<(f64, f64)> {
    let n = n_samples.max(2);
    (0..=n)
        .map(|k| {
            let l = l_max_um * (k as f64) / (n as f64);
            (
                l,
                quadratic_tight_binding_csp(delta_max_ppm, protein_um, l, kd_um),
            )
        })
        .collect()
}

fn draw_sequence_bar_plot(
    ui: &mut egui::Ui,
    y_label: &str,
    bars: &[SequenceBarEntry],
    selected_seq_id: Option<i32>,
    horiz_thresholds: &[(f64, egui::Color32)],
    default_bar_col: egui::Color32,
) -> (Option<i32>, egui::Rect) {
    let bar_h = (ui.available_height() * 0.42).max(165.0);
    let desired = egui::vec2(ui.available_width().max(260.0), bar_h);
    let (rect, response) = ui.allocate_exact_size(desired, egui::Sense::click_and_drag());
    let painter = ui.painter_at(rect);

    painter.rect_filled(rect, 4.0, egui::Color32::from_rgb(16, 20, 29));
    painter.rect_stroke(
        rect,
        4.0,
        egui::Stroke::new(1.0, egui::Color32::from_rgb(45, 55, 72)),
        egui::StrokeKind::Inside,
    );

    if bars.is_empty() {
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "No sequence values",
            egui::FontId::proportional(12.0),
            egui::Color32::GRAY,
        );
        return (None, rect);
    }

    let pad_l = 46.0f32;
    let pad_r = 12.0f32;
    let pad_t = 20.0f32;
    let pad_b = 22.0f32;
    let plot = egui::Rect::from_min_max(
        egui::pos2(rect.left() + pad_l, rect.top() + pad_t),
        egui::pos2(rect.right() - pad_r, rect.bottom() - pad_b),
    );

    let mut y_max = bars
        .iter()
        .map(|b| b.value + b.err.max(0.0))
        .fold(0.0f64, f64::max);
    for &(th, _) in horiz_thresholds {
        if th.is_finite() {
            y_max = y_max.max(th * 1.1);
        }
    }
    let y_min = bars
        .iter()
        .map(|b| (b.value - b.err.max(0.0)).min(0.0))
        .fold(0.0f64, f64::min);
    let y_span = (y_max * 1.12 - y_min).max(1e-6);

    let y_to_px = |y: f64| -> f32 {
        plot.bottom() - (((y - y_min) / y_span) as f32).clamp(0.0, 1.0) * plot.height()
    };
    let y0_px = y_to_px(0.0);

    painter.line_segment(
        [
            egui::pos2(plot.left(), y0_px),
            egui::pos2(plot.right(), y0_px),
        ],
        egui::Stroke::new(1.0, egui::Color32::from_rgb(71, 85, 105)),
    );

    for &(th, col) in horiz_thresholds {
        if th.is_finite() {
            let py = y_to_px(th);
            draw_dashed_hline(&painter, plot.left(), plot.right(), py, col);
        }
    }

    let n = bars.len();
    let slot_w = (plot.width() / (n as f32)).max(2.0);
    let bar_w = (slot_w * 0.74).clamp(1.5, 22.0);

    let mut clicked_sid = None;
    let pointer_pos = response
        .hover_pos()
        .or_else(|| response.interact_pointer_pos());
    let mut hovered_entry: Option<&SequenceBarEntry> = None;

    for (i, b) in bars.iter().enumerate() {
        let cx = plot.left() + ((i as f32) + 0.5) * slot_w;
        let top_y = y_to_px(b.value.max(0.0));
        let bot_y = y_to_px(b.value.min(0.0));
        let bar_rect = egui::Rect::from_min_max(
            egui::pos2(cx - 0.5 * bar_w, top_y.min(y0_px)),
            egui::pos2(cx + 0.5 * bar_w, bot_y.max(y0_px).max(top_y + 1.5)),
        );
        let is_sel = selected_seq_id == Some(b.seq_id);
        let fill = if is_sel {
            egui::Color32::from_rgb(250, 204, 21)
        } else if b.highlight {
            egui::Color32::from_rgb(248, 113, 113)
        } else {
            default_bar_col
        };
        painter.rect_filled(bar_rect, 1.5, fill);

        if b.err > 0.0 {
            let ey_hi = y_to_px(b.value + b.err);
            let ey_lo = y_to_px(b.value - b.err);
            painter.line_segment(
                [egui::pos2(cx, ey_hi), egui::pos2(cx, ey_lo)],
                egui::Stroke::new(1.0, egui::Color32::from_rgb(226, 232, 240)),
            );
        }

        let slot_rect = egui::Rect::from_min_max(
            egui::pos2(cx - 0.5 * slot_w, plot.top()),
            egui::pos2(cx + 0.5 * slot_w, plot.bottom()),
        );
        if pointer_pos.is_some_and(|pos| slot_rect.contains(pos)) {
            hovered_entry = Some(b);
            if response.clicked() {
                clicked_sid = Some(b.seq_id);
            }
        }

        let label_stride = (n / 10).max(1);
        if i.is_multiple_of(label_stride) {
            painter.text(
                egui::pos2(cx, plot.bottom() + 4.0),
                egui::Align2::CENTER_TOP,
                format!("{}", b.seq_id),
                egui::FontId::monospace(11.0),
                egui::Color32::from_rgb(148, 163, 184),
            );
        }
    }

    painter.text(
        egui::pos2(rect.left() + 8.0, rect.top() + 4.0),
        egui::Align2::LEFT_TOP,
        y_label,
        egui::FontId::proportional(12.5),
        egui::Color32::from_rgb(226, 232, 240),
    );
    painter.text(
        egui::pos2(plot.left() - 4.0, plot.top()),
        egui::Align2::RIGHT_TOP,
        format!("{:.2}", y_min + y_span),
        egui::FontId::monospace(11.0),
        egui::Color32::from_rgb(148, 163, 184),
    );

    if let Some(h) = hovered_entry {
        response.on_hover_text(format!(
            "{}{} : {:.4} +- {:.4}",
            h.res_name, h.seq_id, h.value, h.err
        ));
    }

    (clicked_sid, rect)
}

type ObsPointSlice<'a> = (&'a [(f64, f64, f64)], egui::Color32);
type StyledPolyline = (Vec<(f64, f64)>, egui::Color32, bool);

#[allow(clippy::too_many_arguments)]
fn draw_xy_curve_canvas(
    ui: &mut egui::Ui,
    title: &str,
    x_label: &str,
    y_label: &str,
    obs_series: &[ObsPointSlice<'_>],
    curves: &[StyledPolyline],
    vert_guides: &[(f64, egui::Color32)],
    invert_x: bool,
) -> egui::Rect {
    let curve_h = (ui.available_height() - 4.0).max(205.0);
    let desired = egui::vec2(ui.available_width().max(260.0), curve_h);
    let (rect, _resp) = ui.allocate_exact_size(desired, egui::Sense::hover());
    let painter = ui.painter_at(rect);

    painter.rect_filled(rect, 4.0, egui::Color32::from_rgb(14, 18, 26));
    painter.rect_stroke(
        rect,
        4.0,
        egui::Stroke::new(1.0, egui::Color32::from_rgb(45, 55, 72)),
        egui::StrokeKind::Inside,
    );

    let pad_l = 50.0f32;
    let pad_r = 14.0f32;
    let pad_t = 24.0f32;
    let pad_b = 26.0f32;
    let plot = egui::Rect::from_min_max(
        egui::pos2(rect.left() + pad_l, rect.top() + pad_t),
        egui::pos2(rect.right() - pad_r, rect.bottom() - pad_b),
    );

    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;

    for (pts, _) in obs_series {
        for &(x, y, e) in *pts {
            x_min = x_min.min(x);
            x_max = x_max.max(x);
            y_min = y_min.min(y - e.abs());
            y_max = y_max.max(y + e.abs());
        }
    }
    for (poly, _, _) in curves {
        for &(x, y) in poly {
            x_min = x_min.min(x);
            x_max = x_max.max(x);
            y_min = y_min.min(y);
            y_max = y_max.max(y);
        }
    }

    if !x_min.is_finite() || !y_min.is_finite() {
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "No curve data",
            egui::FontId::proportional(12.0),
            egui::Color32::GRAY,
        );
        return rect;
    }

    let x_pad = ((x_max - x_min).abs() * 0.04).max(1e-6);
    let y_pad = ((y_max - y_min).abs() * 0.08).max(1e-4);
    let xa = x_min - x_pad;
    let xb = x_max + x_pad;
    let ya = y_min - y_pad;
    let yb = y_max + y_pad;

    let map_pt = |x: f64, y: f64| -> egui::Pos2 {
        let tx = ((x - xa) / (xb - xa).max(1e-12)) as f32;
        let ux = if invert_x { 1.0 - tx } else { tx };
        let ty = ((y - ya) / (yb - ya).max(1e-12)) as f32;
        egui::pos2(
            plot.left() + ux.clamp(0.0, 1.0) * plot.width(),
            plot.bottom() - ty.clamp(0.0, 1.0) * plot.height(),
        )
    };

    painter.rect_stroke(
        plot,
        0.0,
        egui::Stroke::new(1.0, egui::Color32::from_rgb(51, 65, 85)),
        egui::StrokeKind::Inside,
    );

    for &(vx, col) in vert_guides {
        if vx >= xa && vx <= xb {
            let p_top = map_pt(vx, yb);
            let p_bot = map_pt(vx, ya);
            draw_dashed_vline(&painter, p_top.x, p_top.y, p_bot.y, col);
        }
    }

    for (poly, col, dashed) in curves {
        if poly.len() >= 2 {
            let pts_px: Vec<egui::Pos2> = poly.iter().map(|&(x, y)| map_pt(x, y)).collect();
            if *dashed {
                for seg in pts_px.windows(2).step_by(2) {
                    painter.line_segment([seg[0], seg[1]], egui::Stroke::new(1.6, *col));
                }
            } else {
                painter.add(egui::Shape::line(pts_px, egui::Stroke::new(1.8, *col)));
            }
        }
    }

    for (pts, col) in obs_series {
        for &(x, y, e) in *pts {
            let c = map_pt(x, y);
            if e > 0.0 {
                let hi = map_pt(x, y + e);
                let lo = map_pt(x, y - e);
                painter.line_segment([hi, lo], egui::Stroke::new(1.0, *col));
            }
            painter.circle_filled(c, 2.8, *col);
        }
    }

    painter.text(
        egui::pos2(rect.left() + 8.0, rect.top() + 4.0),
        egui::Align2::LEFT_TOP,
        title,
        egui::FontId::proportional(12.5),
        egui::Color32::from_rgb(226, 232, 240),
    );
    painter.text(
        egui::pos2(plot.center().x, rect.bottom() - 3.0),
        egui::Align2::CENTER_BOTTOM,
        format!("{x_label}   [{y_label}]"),
        egui::FontId::monospace(11.0),
        egui::Color32::from_rgb(148, 163, 184),
    );
    painter.text(
        egui::pos2(plot.left() - 4.0, plot.top()),
        egui::Align2::RIGHT_TOP,
        format!("{yb:.2}"),
        egui::FontId::monospace(11.0),
        egui::Color32::from_rgb(148, 163, 184),
    );
    painter.text(
        egui::pos2(plot.left() - 4.0, plot.bottom()),
        egui::Align2::RIGHT_BOTTOM,
        format!("{ya:.2}"),
        egui::FontId::monospace(11.0),
        egui::Color32::from_rgb(148, 163, 184),
    );

    rect
}

fn draw_titration_2d_trajectory_canvas(
    ui: &mut egui::Ui,
    size: egui::Vec2,
    tracking: Option<&TrajectoryTrackingResult>,
    is_demo: bool,
) -> egui::Rect {
    let (rect, _resp) = ui.allocate_exact_size(size, egui::Sense::hover());
    let painter = ui.painter_at(rect);

    painter.rect_filled(rect, 4.0, egui::Color32::from_rgb(14, 18, 26));
    painter.rect_stroke(
        rect,
        4.0,
        egui::Stroke::new(1.0, egui::Color32::from_rgb(45, 55, 72)),
        egui::StrokeKind::Inside,
    );

    let title = if is_demo {
        "2D (1H, 15N) Trajectory Map [Synthetic Demo]"
    } else {
        "2D (1H, 15N) Trajectory Map"
    };
    painter.text(
        egui::pos2(rect.left() + 8.0, rect.top() + 5.0),
        egui::Align2::LEFT_TOP,
        title,
        egui::FontId::proportional(12.5),
        egui::Color32::from_rgb(226, 232, 240),
    );

    let Some(tr) = tracking else {
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "No 2D titration trajectories loaded",
            egui::FontId::proportional(12.0),
            egui::Color32::GRAY,
        );
        return rect;
    };

    let pad_l = 52.0_f32;
    let pad_r = 14.0_f32;
    let pad_t = 26.0_f32;
    let pad_b = 28.0_f32;
    let plot = egui::Rect::from_min_max(
        egui::pos2(rect.left() + pad_l, rect.top() + pad_t),
        egui::pos2(rect.right() - pad_r, rect.bottom() - pad_b),
    );

    let mut h_min = f64::INFINITY;
    let mut h_max = f64::NEG_INFINITY;
    let mut n_min = f64::INFINITY;
    let mut n_max = f64::NEG_INFINITY;

    for traj in &tr.trajectories {
        for obs in &traj.observations {
            match obs.status {
                TrackStepStatus::Matched { h_ppm, n_ppm, .. } => {
                    h_min = h_min.min(h_ppm);
                    h_max = h_max.max(h_ppm);
                    n_min = n_min.min(n_ppm);
                    n_max = n_max.max(n_ppm);
                }
                TrackStepStatus::DroppedOut {
                    predicted_h_ppm,
                    predicted_n_ppm,
                    ..
                } => {
                    h_min = h_min.min(predicted_h_ppm);
                    h_max = h_max.max(predicted_h_ppm);
                    n_min = n_min.min(predicted_n_ppm);
                    n_max = n_max.max(predicted_n_ppm);
                }
                _ => {}
            }
        }
    }

    if !h_min.is_finite() || !n_min.is_finite() {
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "No matched 2D peaks",
            egui::FontId::proportional(12.0),
            egui::Color32::GRAY,
        );
        return rect;
    }

    let h_pad = ((h_max - h_min).abs() * 0.14).max(0.05);
    let n_pad = ((n_max - n_min).abs() * 0.14).max(0.4);
    let ha = h_min - h_pad;
    let hb = h_max + h_pad;
    let na = n_min - n_pad;
    let nb = n_max + n_pad;

    // Standard IUPAC HSQC axes: 1H high->low left-to-right, 15N low->high top-to-bottom
    let map_hn = |h: f64, n: f64| -> egui::Pos2 {
        let uh = (1.0 - (h - ha) / (hb - ha).max(1e-12)) as f32;
        let un = ((n - na) / (nb - na).max(1e-12)) as f32;
        egui::pos2(
            plot.left() + uh.clamp(0.0, 1.0) * plot.width(),
            plot.top() + un.clamp(0.0, 1.0) * plot.height(),
        )
    };

    painter.rect_stroke(
        plot,
        0.0,
        egui::Stroke::new(1.0, egui::Color32::from_rgb(51, 65, 85)),
        egui::StrokeKind::Inside,
    );

    let track_palette = [
        egui::Color32::from_rgb(56, 189, 248),
        egui::Color32::from_rgb(251, 146, 60),
        egui::Color32::from_rgb(52, 211, 153),
        egui::Color32::from_rgb(192, 132, 252),
        egui::Color32::from_rgb(244, 114, 182),
    ];

    let mut placed_labels: Vec<egui::Rect> = Vec::new();
    for (ti, traj) in tr.trajectories.iter().enumerate() {
        let base_col = track_palette[ti % track_palette.len()];
        let mut poly: Vec<egui::Pos2> = Vec::new();
        let n_obs = traj.observations.len().max(1);
        let mut label_anchor: Option<egui::Pos2> = None;

        for (oi, obs) in traj.observations.iter().enumerate() {
            let frac = (oi as f32) / (n_obs.saturating_sub(1).max(1) as f32);
            match obs.status {
                TrackStepStatus::Matched { h_ppm, n_ppm, .. } => {
                    let pt = map_hn(h_ppm, n_ppm);
                    if label_anchor.is_none() {
                        label_anchor = Some(pt);
                    }
                    poly.push(pt);
                    let r = 2.6 + 1.6 * frac;
                    let step_col = egui::Color32::from_rgb(
                        ((1.0 - frac) * f32::from(base_col.r()) + frac * 250.0) as u8,
                        ((1.0 - frac) * f32::from(base_col.g()) + frac * 204.0) as u8,
                        ((1.0 - frac) * f32::from(base_col.b()) + frac * 21.0) as u8,
                    );
                    painter.circle_filled(pt, r, step_col);
                }
                TrackStepStatus::DroppedOut {
                    predicted_h_ppm,
                    predicted_n_ppm,
                    ..
                } => {
                    let pt = map_hn(predicted_h_ppm, predicted_n_ppm);
                    poly.push(pt);
                    painter.circle_stroke(
                        pt,
                        4.5,
                        egui::Stroke::new(1.3, egui::Color32::from_rgb(248, 113, 113)),
                    );
                }
                _ => {}
            }
        }
        if poly.len() >= 2 {
            painter.add(egui::Shape::line(poly, egui::Stroke::new(1.5, base_col)));
        }
        if let Some(anchor) = label_anchor {
            let lbl = traj
                .label
                .clone()
                .unwrap_or_else(|| format!("T{}", traj.track_id));
            let font = egui::FontId::monospace(11.0);
            let est_w = (lbl.len() as f32) * 7.2 + 6.0;
            let est_h = 14.0_f32;
            let cand_pos = egui::pos2(
                (anchor.x + 6.0)
                    .min(plot.right() - est_w - 2.0)
                    .max(plot.left() + 2.0),
                (anchor.y - 14.0)
                    .max(plot.top() + 2.0)
                    .min(plot.bottom() - est_h - 2.0),
            );
            let cand_rect = egui::Rect::from_min_size(cand_pos, egui::vec2(est_w, est_h));
            if !placed_labels.iter().any(|r| r.intersects(cand_rect)) {
                placed_labels.push(cand_rect);
                painter.text(cand_pos, egui::Align2::LEFT_TOP, lbl, font, base_col);
            }
        }
    }

    // Highlight ambiguity flags on matching steps
    for flag in &tr.ambiguity_flags {
        if let Some(traj) = tr.trajectories.iter().find(|t| t.track_id == flag.track_id) {
            if let Some(obs) = traj
                .observations
                .iter()
                .find(|o| o.step_index == flag.step_index)
            {
                if let TrackStepStatus::Matched { h_ppm, n_ppm, .. } = obs.status {
                    let pt = map_hn(h_ppm, n_ppm);
                    painter.circle_stroke(
                        pt,
                        7.0,
                        egui::Stroke::new(1.5, egui::Color32::from_rgb(251, 191, 36)),
                    );
                }
            }
        }
    }

    // Clean axis bounds & title (zero overlap)
    painter.text(
        egui::pos2(plot.left() - 4.0, plot.top()),
        egui::Align2::RIGHT_TOP,
        format!("{na:.1}"),
        egui::FontId::monospace(11.0),
        egui::Color32::from_rgb(148, 163, 184),
    );
    painter.text(
        egui::pos2(plot.left() - 4.0, plot.bottom()),
        egui::Align2::RIGHT_BOTTOM,
        format!("{nb:.1}"),
        egui::FontId::monospace(11.0),
        egui::Color32::from_rgb(148, 163, 184),
    );
    painter.text(
        egui::pos2(plot.left() + 2.0, rect.bottom() - 4.0),
        egui::Align2::LEFT_BOTTOM,
        format!("{hb:.2}"),
        egui::FontId::monospace(11.0),
        egui::Color32::from_rgb(148, 163, 184),
    );
    painter.text(
        egui::pos2(plot.right() - 2.0, rect.bottom() - 4.0),
        egui::Align2::RIGHT_BOTTOM,
        format!("{ha:.2}"),
        egui::FontId::monospace(11.0),
        egui::Color32::from_rgb(148, 163, 184),
    );
    painter.text(
        egui::pos2(plot.center().x, rect.bottom() - 4.0),
        egui::Align2::CENTER_BOTTOM,
        "1H (ppm)  /  15N (ppm)",
        egui::FontId::monospace(11.0),
        egui::Color32::from_rgb(148, 163, 184),
    );

    rect
}

fn draw_titration_2d_lineshape_canvas(
    ui: &mut egui::Ui,
    size: egui::Vec2,
    state: &Titration2dUiState,
) -> egui::Rect {
    let (rect, _resp) = ui.allocate_exact_size(size, egui::Sense::hover());
    let painter = ui.painter_at(rect);

    painter.rect_filled(rect, 4.0, egui::Color32::from_rgb(14, 18, 26));
    painter.rect_stroke(
        rect,
        4.0,
        egui::Stroke::new(1.0, egui::Color32::from_rgb(45, 55, 72)),
        egui::StrokeKind::Inside,
    );

    if let Some(fit) = &state.lineshape_fit {
        let sel_r = state
            .selected_residue_idx
            .min(fit.residues.len().saturating_sub(1));
        let Some(fitted_res) = fit.residues.get(sel_r) else {
            return rect;
        };

        let header = format!(
            "2D Bloch-McConnell ROI — {} (ΔωH={:+.3}, ΔωN={:+.2} ppm)",
            fitted_res.label, fitted_res.dw_h_ppm, fitted_res.dw_n_ppm
        );
        painter.text(
            egui::pos2(rect.left() + 8.0, rect.top() + 5.0),
            egui::Align2::LEFT_TOP,
            header,
            egui::FontId::proportional(12.5),
            egui::Color32::from_rgb(226, 232, 240),
        );

        let split_y = rect.top() + (rect.height() * 0.58).max(120.0);
        let plot = egui::Rect::from_min_max(
            egui::pos2(rect.left() + 48.0, rect.top() + 26.0),
            egui::pos2(rect.right() - 14.0, (split_y - 24.0).max(rect.top() + 60.0)),
        );

        // Render observed ROI heatmap + fitted 2D Bloch-McConnell marching-squares contours
        if let Some(series) = state.roi_series.get(sel_r) {
            let s_idx = state
                .selected_step_idx
                .min(series.patches.len().saturating_sub(1));
            if let Some(patch) = series.patches.get(s_idx) {
                let n1 = patch.grid.n_ppm_axis.len();
                let n2 = patch.grid.h_ppm_axis.len();
                let peak_obs = patch
                    .intensities
                    .iter()
                    .copied()
                    .fold(0.0_f64, f64::max)
                    .max(1e-9);

                let cw = plot.width() / (n2.max(1) as f32);
                let ch = plot.height() / (n1.max(1) as f32);

                for i in 0..n1 {
                    for j in 0..n2 {
                        let v = (patch.intensities[i * n2 + j] / peak_obs).clamp(0.0, 1.0) as f32;
                        if v > 0.04 {
                            // Invert 1H (j) so high ppm is on left
                            let col_idx = n2 - 1 - j;
                            let cell = egui::Rect::from_min_size(
                                egui::pos2(
                                    plot.left() + (col_idx as f32) * cw,
                                    plot.top() + (i as f32) * ch,
                                ),
                                egui::vec2(cw + 0.5, ch + 0.5),
                            );
                            let alpha = (v * 185.0) as u8;
                            painter.rect_filled(
                                cell,
                                0.0,
                                egui::Color32::from_rgba_unmultiplied(56, 189, 248, alpha),
                            );
                        }
                    }
                }

                // Fitted analytical 2D Bloch-McConnell contour lines at 25%, 55%, 80% of max
                if let Ok(eq) = binding_equilibrium_rates(
                    patch.protein_conc_um,
                    patch.ligand_conc_um,
                    fit.kd_um,
                    fit.k_off_s_inv,
                ) {
                    if let Ok(sim) = simulate_2d_lineshape_analytical(
                        &patch.grid,
                        &fitted_res.params,
                        &Bloch2dKineticState::from(eq),
                    ) {
                        let peak_sim = sim.iter().copied().fold(0.0_f64, f64::max).max(1e-9);
                        let contour_cols = [
                            (0.25, egui::Color32::from_rgb(52, 211, 153)),
                            (0.55, egui::Color32::from_rgb(250, 204, 21)),
                            (0.80, egui::Color32::from_rgb(248, 113, 113)),
                        ];
                        for (frac, col) in contour_cols {
                            let lvl = frac * peak_sim;
                            draw_grid_iso_contours(
                                &painter,
                                plot,
                                n1,
                                n2,
                                &sim,
                                lvl,
                                egui::Stroke::new(1.3, col),
                            );
                        }
                    }
                }

                painter.rect_stroke(
                    plot,
                    0.0,
                    egui::Stroke::new(1.0, egui::Color32::from_rgb(51, 65, 85)),
                    egui::StrokeKind::Inside,
                );
                if let (Some(&n_lo), Some(&n_hi), Some(&h_lo), Some(&h_hi)) = (
                    patch.grid.n_ppm_axis.first(),
                    patch.grid.n_ppm_axis.last(),
                    patch.grid.h_ppm_axis.first(),
                    patch.grid.h_ppm_axis.last(),
                ) {
                    painter.text(
                        egui::pos2(plot.left() - 4.0, plot.top()),
                        egui::Align2::RIGHT_TOP,
                        format!("{n_lo:.1}"),
                        egui::FontId::monospace(11.0),
                        egui::Color32::from_rgb(148, 163, 184),
                    );
                    painter.text(
                        egui::pos2(plot.left() - 4.0, plot.bottom()),
                        egui::Align2::RIGHT_BOTTOM,
                        format!("{n_hi:.1}"),
                        egui::FontId::monospace(11.0),
                        egui::Color32::from_rgb(148, 163, 184),
                    );
                    painter.text(
                        egui::pos2(plot.center().x, plot.bottom() + 4.0),
                        egui::Align2::CENTER_TOP,
                        format!(
                            "Step {} ([L]={:.0} uM) | 1H [{h_hi:.2}..{h_lo:.2}] | Blue=Obs, Green/Gold=Fit",
                            s_idx, patch.ligand_conc_um
                        ),
                        egui::FontId::monospace(11.0),
                        egui::Color32::from_rgb(148, 163, 184),
                    );
                }
            }
        }

        // Bottom 42%: Exchange-regime |dw|/k_off & fitted parameters table
        let mut row_y = split_y + 4.0;
        let max_y = rect.bottom() - 16.0;
        if row_y <= max_y {
            painter.text(
                egui::pos2(rect.left() + 10.0, row_y),
                egui::Align2::LEFT_TOP,
                "Res   Regime        |dwH|/koff  |dwN|/koff   dwH(ppm)  dwN(ppm)",
                egui::FontId::monospace(11.0),
                egui::Color32::from_rgb(148, 163, 184),
            );
            row_y += 17.0;
        }
        for rf in &fit.residues {
            if row_y > max_y {
                break;
            }
            let (a_off_h, a_off_n) = rf
                .exchange_regime
                .steps
                .first()
                .map(|s| (s.alpha_off_h, s.alpha_off_n))
                .unwrap_or((0.0, 0.0));
            let reg_col = match rf.exchange_regime.overall_worst_regime {
                ExchangeRegime::Fast => egui::Color32::from_rgb(52, 211, 153),
                ExchangeRegime::Intermediate => egui::Color32::from_rgb(251, 191, 36),
                ExchangeRegime::Slow => egui::Color32::from_rgb(248, 113, 113),
            };
            let line = format!(
                "{:<5} {:<13} {:>8.2}    {:>8.2}    {:>+7.3}   {:>+7.2}",
                rf.label,
                rf.exchange_regime.overall_worst_regime.as_str(),
                a_off_h,
                a_off_n,
                rf.dw_h_ppm,
                rf.dw_n_ppm,
            );
            painter.text(
                egui::pos2(rect.left() + 10.0, row_y),
                egui::Align2::LEFT_TOP,
                line,
                egui::FontId::monospace(11.0),
                reg_col,
            );
            row_y += 17.0;
        }
        if row_y <= max_y {
            let cmp = &fit.fast_exchange_comparison;
            let cmp_line = format!(
                "Fast-CSP Kd={:.1} uM (disc={:.1}%) | 2D Kd={:.1} uM, k_off={:.0} s^-1",
                cmp.csp_kd_um,
                cmp.kd_relative_discrepancy * 100.0,
                fit.kd_um,
                fit.k_off_s_inv,
            );
            painter.text(
                egui::pos2(rect.left() + 10.0, row_y),
                egui::Align2::LEFT_TOP,
                cmp_line,
                egui::FontId::monospace(11.0),
                egui::Color32::from_rgb(203, 213, 225),
            );
        }
        return rect;
    }

    // Fallback when only peak-list TSV is tracked (no 2D ROI planes yet):
    // draw |dw|/k_off exchange-regime bar chart with Fast (0.3) and Slow (3.0) thresholds.
    painter.text(
        egui::pos2(rect.left() + 8.0, rect.top() + 5.0),
        egui::Align2::LEFT_TOP,
        "Exchange Regime Diagnostics (|Δω| / k_off)",
        egui::FontId::proportional(12.5),
        egui::Color32::from_rgb(226, 232, 240),
    );
    if state.regime_reports.is_empty() {
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "No exchange-regime diagnostics yet",
            egui::FontId::proportional(12.0),
            egui::Color32::GRAY,
        );
        return rect;
    }

    let plot = egui::Rect::from_min_max(
        egui::pos2(rect.left() + 48.0, rect.top() + 28.0),
        egui::pos2(rect.right() - 14.0, rect.bottom() - 28.0),
    );
    let max_alpha = state
        .regime_reports
        .iter()
        .filter_map(|r| r.steps.first().map(|s| s.alpha_off_h.max(s.alpha_off_n)))
        .fold(SLOW_EXCHANGE_ALPHA_MIN * 1.15, f64::max);

    let y_px = |a: f64| -> f32 {
        plot.bottom() - ((a / max_alpha.max(1e-6)) as f32).clamp(0.0, 1.0) * plot.height()
    };
    painter.rect_stroke(
        plot,
        0.0,
        egui::Stroke::new(1.0, egui::Color32::from_rgb(51, 65, 85)),
        egui::StrokeKind::Inside,
    );
    draw_dashed_hline(
        &painter,
        plot.left(),
        plot.right(),
        y_px(FAST_EXCHANGE_ALPHA_MAX),
        egui::Color32::from_rgb(52, 211, 153),
    );
    draw_dashed_hline(
        &painter,
        plot.left(),
        plot.right(),
        y_px(SLOW_EXCHANGE_ALPHA_MIN),
        egui::Color32::from_rgb(248, 113, 113),
    );

    let n = state.regime_reports.len();
    let slot_w = (plot.width() / (n.max(1) as f32)).max(12.0);
    let bar_w = (slot_w * 0.55).clamp(6.0, 34.0);
    for (i, rep) in state.regime_reports.iter().enumerate() {
        let val = rep
            .steps
            .first()
            .map(|s| s.alpha_off_h.max(s.alpha_off_n))
            .unwrap_or(0.0);
        let cx = plot.left() + ((i as f32) + 0.5) * slot_w;
        let top = y_px(val);
        let col = match rep.overall_worst_regime {
            ExchangeRegime::Fast => egui::Color32::from_rgb(52, 211, 153),
            ExchangeRegime::Intermediate => egui::Color32::from_rgb(251, 191, 36),
            ExchangeRegime::Slow => egui::Color32::from_rgb(248, 113, 113),
        };
        painter.rect_filled(
            egui::Rect::from_min_max(
                egui::pos2(cx - 0.5 * bar_w, top),
                egui::pos2(cx + 0.5 * bar_w, plot.bottom()),
            ),
            2.0,
            col,
        );
        let stride = (n / 8).max(1);
        if i.is_multiple_of(stride) {
            painter.text(
                egui::pos2(cx, plot.bottom() + 4.0),
                egui::Align2::CENTER_TOP,
                &rep.label,
                egui::FontId::monospace(11.0),
                egui::Color32::from_rgb(148, 163, 184),
            );
        }
    }
    painter.text(
        egui::pos2(plot.left() - 4.0, plot.top()),
        egui::Align2::RIGHT_TOP,
        format!("{max_alpha:.1}"),
        egui::FontId::monospace(11.0),
        egui::Color32::from_rgb(148, 163, 184),
    );
    rect
}

/// Simple marching-squares iso-contour segments over a `(n1 x n2)` row-major grid (with inverted `1H` X-axis).
fn draw_grid_iso_contours(
    painter: &egui::Painter,
    plot: egui::Rect,
    n1: usize,
    n2: usize,
    vals: &[f64],
    level: f64,
    stroke: egui::Stroke,
) {
    if n1 < 2 || n2 < 2 || vals.len() < n1 * n2 {
        return;
    }
    let dx = plot.width() / ((n2 - 1) as f32);
    let dy = plot.height() / ((n1 - 1) as f32);
    let pt_px = |i: f32, j: f32| -> egui::Pos2 {
        let j_inv = ((n2 - 1) as f32) - j;
        egui::pos2(plot.left() + j_inv * dx, plot.top() + i * dy)
    };
    let interp = |a: f64, b: f64| -> f32 {
        let d = b - a;
        if d.abs() < 1e-18 {
            0.5
        } else {
            ((level - a) / d).clamp(0.0, 1.0) as f32
        }
    };

    for i in 0..(n1 - 1) {
        for j in 0..(n2 - 1) {
            let v00 = vals[i * n2 + j];
            let v01 = vals[i * n2 + j + 1];
            let v11 = vals[(i + 1) * n2 + j + 1];
            let v10 = vals[(i + 1) * n2 + j];
            let mut crossings = [egui::Pos2::ZERO; 4];
            let mut count = 0usize;

            if (v00 >= level) != (v01 >= level) {
                crossings[count] = pt_px(i as f32, j as f32 + interp(v00, v01));
                count += 1;
            }
            if (v01 >= level) != (v11 >= level) {
                crossings[count] = pt_px(i as f32 + interp(v01, v11), (j + 1) as f32);
                count += 1;
            }
            if (v11 >= level) != (v10 >= level) {
                crossings[count] = pt_px((i + 1) as f32, (j + 1) as f32 - interp(v11, v10));
                count += 1;
            }
            if (v10 >= level) != (v00 >= level) {
                crossings[count] = pt_px((i + 1) as f32 - interp(v10, v00), j as f32);
                count += 1;
            }

            if count >= 2 {
                painter.line_segment([crossings[0], crossings[1]], stroke);
            }
            if count == 4 {
                painter.line_segment([crossings[2], crossings[3]], stroke);
            }
        }
    }
}

fn draw_dashed_hline(painter: &egui::Painter, x0: f32, x1: f32, y: f32, col: egui::Color32) {
    let mut x = x0;
    while x < x1 {
        let xe = (x + 6.0).min(x1);
        painter.line_segment(
            [egui::pos2(x, y), egui::pos2(xe, y)],
            egui::Stroke::new(1.1, col),
        );
        x += 10.0;
    }
}

fn draw_dashed_vline(painter: &egui::Painter, x: f32, y0: f32, y1: f32, col: egui::Color32) {
    let (mut y, yn) = if y0 <= y1 { (y0, y1) } else { (y1, y0) };
    while y < yn {
        let ye = (y + 5.0).min(yn);
        painter.line_segment(
            [egui::pos2(x, y), egui::pos2(x, ye)],
            egui::Stroke::new(1.1, col),
        );
        y += 9.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nmr_core::cest::CestNormalization;
    use nmr_core::csp::{fit_csp_global_kd, CspAnalysisConfig, ResidueTitrationPoint};

    #[test]
    fn test_one_click_overlay_projection_into_quantitative_track_data() {
        let mut panel = DynamicsStudioPanel::new();
        panel.set_relaxation_records(vec![
            ResidueRelaxationRecord {
                seq_id: 10,
                res_name: "ALA".to_string(),
                r1_s_inv: Some((1.45, 0.03)),
                r2_s_inv: Some((12.2, 0.25)),
                het_noe: Some((0.82, 0.02)),
                s2: Some((0.88, 0.02)),
                rex_s_inv: Some((0.0, 0.1)),
                r1_i0: 1.0,
                r1_decay: vec![DecayCurvePoint {
                    delay_s: 0.1,
                    intensity: 0.865,
                    uncertainty: 0.01,
                }],
                r2_i0: 1.0,
                r2_decay: Vec::new(),
            },
            ResidueRelaxationRecord {
                seq_id: 20,
                res_name: "VAL".to_string(),
                r1_s_inv: Some((1.38, 0.03)),
                r2_s_inv: Some((18.6, 0.35)),
                het_noe: Some((0.76, 0.03)),
                s2: Some((0.64, 0.03)),
                rex_s_inv: Some((6.4, 0.4)),
                r1_i0: 1.0,
                r1_decay: Vec::new(),
                r2_i0: 1.0,
                r2_decay: Vec::new(),
            },
        ]);

        panel.set_csp_data(
            vec![
                ResidueCspTrajectory {
                    seq_index: 0,
                    seq_id: 10,
                    res_name: "ALA".to_string(),
                    apo_h_ppm: 8.2,
                    apo_n_ppm: 120.0,
                    points: Vec::new(),
                    max_observed_csp_ppm: 0.03,
                },
                ResidueCspTrajectory {
                    seq_index: 1,
                    seq_id: 20,
                    res_name: "VAL".to_string(),
                    apo_h_ppm: 8.5,
                    apo_n_ppm: 122.0,
                    points: Vec::new(),
                    max_observed_csp_ppm: 0.42,
                },
            ],
            None,
            None,
        );

        let seq_ids = [10, 20];
        let mut track_data = QuantitativeTrackData::default();

        // 1. Project S^2
        let p_s2 = panel
            .build_overlay_projection(QuantitativeTrack::LipariSzaboS2)
            .expect("S^2 projection");
        p_s2.apply_to(&mut track_data, &seq_ids);
        assert_eq!(track_data.active_track, QuantitativeTrack::LipariSzaboS2);
        assert!(track_data.putty_enabled);
        assert!((track_data.lipari_szabo_s2[&0] - 0.88).abs() < 1e-9);
        assert!((track_data.lipari_szabo_s2[&1] - 0.64).abs() < 1e-9);

        // 2. Project R_ex
        let p_rex = panel
            .build_overlay_projection(QuantitativeTrack::ChemicalExchangeRex)
            .expect("R_ex projection");
        p_rex.apply_to(&mut track_data, &seq_ids);
        assert_eq!(
            track_data.active_track,
            QuantitativeTrack::ChemicalExchangeRex
        );
        assert!((track_data.rex_s_inv[&1] - 6.4).abs() < 1e-9);
        assert!((track_data.rex_max_s_inv - 6.4).abs() < 1e-9);

        // 3. Project CSP (binding-pocket 3D highlight projection onto QuantitativeTrackData)
        let p_csp = panel
            .build_overlay_projection(QuantitativeTrack::LigandCsp)
            .expect("CSP projection");
        p_csp.apply_to(&mut track_data, &seq_ids);
        assert_eq!(track_data.active_track, QuantitativeTrack::LigandCsp);
        assert!((track_data.csp_ppm[&0] - 0.03).abs() < 1e-9);
        assert!((track_data.csp_ppm[&1] - 0.42).abs() < 1e-9);
        assert!((track_data.csp_max_ppm - 0.42).abs() < 1e-9);

        let n_pocket_above_0p15 = track_data.csp_ppm.values().filter(|&&v| v >= 0.15).count();
        assert_eq!(n_pocket_above_0p15, 1);
        println!(
            "W4_METRIC_JSON: {{\"task\":\"T13.6\",\"kind\":\"gui_3d_pocket_overlay\",\"n_residues\":{},\"n_pocket_highlighted\":{},\"csp_max_ppm\":{:.4},\"rex_max_s_inv\":{:.4},\"s2_projected\":true,\"rex_projected\":true,\"csp_projected\":true,\"putty_enabled\":{},\"passed\":true}}",
            seq_ids.len(),
            n_pocket_above_0p15,
            track_data.csp_max_ppm,
            track_data.rex_max_s_inv,
            track_data.putty_enabled,
        );
    }

    #[test]
    fn test_headless_egui_rendering_across_all_four_dynamics_studio_tabs() {
        let mut panel = DynamicsStudioPanel::new();
        panel.set_relaxation_records(vec![ResidueRelaxationRecord {
            seq_id: 1,
            res_name: "MET".to_string(),
            r1_s_inv: Some((1.5, 0.04)),
            r2_s_inv: Some((13.0, 0.3)),
            het_noe: Some((0.80, 0.02)),
            s2: Some((0.85, 0.02)),
            rex_s_inv: Some((2.5, 0.3)),
            r1_i0: 100.0,
            r1_decay: vec![
                DecayCurvePoint {
                    delay_s: 0.02,
                    intensity: 97.0,
                    uncertainty: 1.0,
                },
                DecayCurvePoint {
                    delay_s: 0.50,
                    intensity: 47.2,
                    uncertainty: 1.0,
                },
            ],
            r2_i0: 100.0,
            r2_decay: Vec::new(),
        }]);

        panel.set_cest_data(
            vec![CestProfileDataset {
                seq_id: 1,
                res_name: "MET".to_string(),
                larmor_mhz: 60.8106,
                t_ex_s: 0.40,
                b1_hz: 15.0,
                b1_inhomogeneity_frac: 0.0,
                normalization: CestNormalization::ThermalEquilibrium,
                major_ppm_hint: Some(120.0),
                offsets_ppm: vec![116.0, 120.0, 124.0],
                intensities: vec![0.99, 0.05, 0.85],
                uncertainties: vec![0.005, 0.005, 0.005],
            }],
            None,
        );

        panel.set_cpmg_data(
            vec![CpmgSeriesDataset {
                seq_id: 1,
                res_name: "MET".to_string(),
                larmor_mhz: 60.8106,
                t_relax_s: 0.040,
                nu_cpmg_hz: vec![50.0, 200.0, 800.0],
                r2_eff_s_inv: vec![18.2, 14.1, 12.6],
                r2_eff_err: vec![0.2, 0.2, 0.2],
            }],
            None,
        );

        let pts: Vec<ResidueTitrationPoint> = [0.0, 25.0, 75.0, 150.0, 300.0]
            .iter()
            .map(|&l| {
                let csp = quadratic_tight_binding_csp(0.35, 100.0, l, 10.0);
                ResidueTitrationPoint {
                    protein_conc_um: 100.0,
                    ligand_conc_um: l,
                    matched_peak: Some((8.1 + csp, 120.0)),
                    csp_ppm: Some(csp),
                }
            })
            .collect();
        let trajs = vec![ResidueCspTrajectory {
            seq_index: 0,
            seq_id: 1,
            res_name: "MET".to_string(),
            apo_h_ppm: 8.1,
            apo_n_ppm: 120.0,
            points: pts,
            max_observed_csp_ppm: 0.34,
        }];
        let cfg = CspAnalysisConfig {
            n_mc: 24,
            ..CspAnalysisConfig::default()
        };
        let kd_fit = fit_csp_global_kd(&trajs, &cfg).ok();
        panel.set_csp_data(trajs, kd_fit, None);

        let ctx = egui::Context::default();
        for tab in DynamicsStudioTab::ALL {
            panel.active_tab = tab;
            let mut out = ctx.run_ui(egui::RawInput::default(), |ui| {
                let _ = panel.show(ui);
            });
            out.textures_delta.clear();
        }

        let decay = build_exponential_decay_curve(1.0, 2.0, 1.0, 10);
        assert_eq!(decay.len(), 11);
        assert!((decay[0].1 - 1.0).abs() < 1e-12);
        assert!(decay[10].1 < decay[0].1);
    }

    #[test]
    fn test_demo_population_and_tsv_import_across_cest_cpmg_and_csp() {
        use nmr_core::cest::write_cest_profiles_tsv;
        use nmr_core::cpmg::write_cpmg_series_tsv;
        use nmr_core::csp::write_csp_trajectories_tsv;

        let mut panel = DynamicsStudioPanel::new();
        let summary = panel
            .ensure_demo_exchange_and_csp(&[], &[])
            .expect("populate demo exchange & csp");
        assert!(summary.contains("CEST"));
        assert!(panel.cest_fit.is_some());
        assert!(panel.cpmg_fit.is_some());
        assert!(panel.csp_fit.is_some());

        let cest_tsv = write_cest_profiles_tsv(&panel.cest_profiles);
        let cpmg_tsv = write_cpmg_series_tsv(&panel.cpmg_series);
        let csp_tsv = write_csp_trajectories_tsv(&panel.csp_trajectories);

        assert!(DynamicsStudioPanel::looks_like_exchange_or_csp_tsv(
            &cest_tsv
        ));
        assert!(DynamicsStudioPanel::looks_like_exchange_or_csp_tsv(
            &cpmg_tsv
        ));
        assert!(DynamicsStudioPanel::looks_like_exchange_or_csp_tsv(
            &csp_tsv
        ));

        let mut fresh = DynamicsStudioPanel::new();
        fresh
            .import_exchange_or_csp_tsv(&cest_tsv)
            .expect("import cest tsv");
        assert_eq!(fresh.active_tab, DynamicsStudioTab::CestDest);
        fresh
            .import_exchange_or_csp_tsv(&cpmg_tsv)
            .expect("import cpmg tsv");
        assert_eq!(fresh.active_tab, DynamicsStudioTab::CpmgDispersion);
        fresh
            .import_exchange_or_csp_tsv(&csp_tsv)
            .expect("import csp tsv");
        assert_eq!(fresh.active_tab, DynamicsStudioTab::LigandCsp);
    }
}
