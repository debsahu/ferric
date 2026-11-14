//! Interactive GUI Dynamics & Ligand Titration Studio (`T13.6`) for native desktop and
//! `wasm32-unknown-unknown` WebGPU targets.
//!
//! Renders linked per-residue sequence bar plots (`R1`, `R2`, `hetNOE`, `S^2`, `R_ex`, and
//! `Delta_delta_comb`), exponential relaxation decay curves (`I(t) = I0 exp(-R t)`), multi-`B1`
//! `CEST`/`DEST` saturation dip profiles, multi-field `CPMG` dispersion curves, and tight-binding
//! quadratic `CSP` `K_d` isotherms, with 1-click scalar putty projection onto
//! [`QuantitativeTrackData`] (`mol3d_overlay`).
//!
//! Provenance: Clean-room original interactive visualization widget grounded in:
//! - Lipari, G. & Szabo, A. (*J. Am. Chem. Soc.* 1982, 104, 4546–4559 — model-free `S^2`, `tau_e`, `R_ex`).
//! - McConnell, H. M. (*J. Chem. Phys.* 1958, 28, 430–431), Vallurupalli, P., Bouvignies, G. &
//!   Kay, L. E. (*J. Am. Chem. Soc.* 2012, 134, 8148–8161), Fawzi, N. L. et al. (*Nature* 2011,
//!   480, 268–272), and Higham, N. J. (*SIAM J. Matrix Anal. Appl.* 2005, 26, 1179–1193) — `CEST`/`DEST`.
//! - Luz, Z. & Meiboom, S. (*J. Chem. Phys.* 1963, 39, 366–370), Carver, J. P. & Richards, R. E.
//!   (*J. Magn. Reson.* 1972, 6, 89–105), Baldwin, A. J. (*J. Magn. Reson.* 2014, 244, 114–124),
//!   and Trott, O. & Palmer, A. G. III (*J. Magn. Reson.* 2002, 154, 157–160) — `CPMG` & `R1rho`.
//! - Williamson, M. P. (*Prog. Nucl. Magn. Reson. Spectrosc.* 2013, 73, 1–16 — amide `CSP`,
//!   iterative `2 sigma` threshold, and quadratic tight-binding `K_d` isotherms).

use std::collections::BTreeMap;
use std::f64::consts::TAU;

use eframe::egui;
use nmr_core::cest::{
    simulate_cest_2state_profile, Cest2StateParams, CestGlobalFitResult, CestProfileDataset,
};
use nmr_core::cpmg::{evaluate_cpmg_model, CpmgGlobalFitResult, CpmgSeriesDataset};
use nmr_core::csp::{
    fit_naive_michaelis_menten_kd, naive_michaelis_menten_csp, quadratic_tight_binding_csp,
    CspGlobalFitResult, ResidueCspTrajectory, TrimmedCspThresholds,
};

use crate::mol3d_overlay::{ColormapKind, QuantitativeTrack, QuantitativeTrackData};

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
    pub cpmg_series: Vec<CpmgSeriesDataset>,
    pub cpmg_fit: Option<CpmgGlobalFitResult>,
    pub csp_trajectories: Vec<ResidueCspTrajectory>,
    pub csp_fit: Option<CspGlobalFitResult>,
    pub csp_threshold: Option<TrimmedCspThresholds>,
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
            cpmg_series: Vec::new(),
            cpmg_fit: None,
            csp_trajectories: Vec::new(),
            csp_fit: None,
            csp_threshold: None,
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
                .add_enabled(s2_avail, egui::Button::new("Project S² -> 3D"))
                .on_hover_text("Project Lipari–Szabo S² order parameters onto 3D structure putty")
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
                .on_hover_text("Project conformational exchange R_ex onto 3D structure putty")
                .clicked()
            {
                output.overlay_projection =
                    self.build_overlay_projection(QuantitativeTrack::ChemicalExchangeRex);
            }

            let csp_avail = !self.csp_trajectories.is_empty();
            if ui
                .add_enabled(csp_avail, egui::Button::new("Project CSP -> 3D"))
                .on_hover_text("Project ligand titration Δδ_comb onto 3D structure putty")
                .clicked()
            {
                output.overlay_projection =
                    self.build_overlay_projection(QuantitativeTrack::LigandCsp);
            }
        });
        ui.separator();

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

    fn render_relaxation_tab(&mut self, ui: &mut egui::Ui, output: &mut DynamicsStudioOutput) {
        ui.horizontal_wrapped(|ui| {
            ui.label("Sequence Track:");
            for m in SequenceMetricKind::RELAX_METRICS {
                if ui
                    .selectable_label(self.active_relax_metric == m, m.label())
                    .clicked()
                {
                    self.active_relax_metric = m;
                }
            }
            ui.separator();
            ui.selectable_value(&mut self.show_r2_decay, false, "R1 Decay Curve");
            ui.selectable_value(&mut self.show_r2_decay, true, "R2 Decay Curve");
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

        if let Some(clicked) = draw_sequence_bar_plot(
            ui,
            self.active_relax_metric.label(),
            &bars,
            self.selected_seq_id,
            &[],
            egui::Color32::from_rgb(56, 189, 248),
        ) {
            self.selected_seq_id = Some(clicked);
            output.clicked_seq_id = Some(clicked);
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
            draw_xy_curve_canvas(
                ui,
                &header,
                "Delay t (s)",
                "Intensity I(t)",
                &[(&obs, egui::Color32::from_rgb(125, 211, 252))],
                &curves,
                &[],
                false,
            );
        }
    }

    fn render_cest_tab(&mut self, ui: &mut egui::Ui, output: &mut DynamicsStudioOutput) {
        if self.cest_profiles.is_empty() {
            ui.label("No CEST / DEST saturation profiles loaded.");
            return;
        }

        if let Some(fit) = &self.cest_fit {
            ui.label(format!(
                "Global CEST Fit: p_B = {:.2}% ± {:.2}%  |  k_ex = {:.1} ± {:.1} s⁻¹  |  χ²_red = {:.3}",
                fit.p_b * 100.0,
                fit.p_b_err * 100.0,
                fit.k_ex_s_inv,
                fit.k_ex_err,
                fit.reduced_chi2
            ));
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
            if let Some(clicked) = draw_sequence_bar_plot(
                ui,
                "|Δϖ_AB| (ppm)",
                &bars,
                self.selected_seq_id,
                &[],
                egui::Color32::from_rgb(167, 139, 250),
            ) {
                self.selected_seq_id = Some(clicked);
                output.clicked_seq_id = Some(clicked);
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
            if let Some(clicked) = draw_sequence_bar_plot(
                ui,
                "Max Dip (1 - I/I0)",
                &bars,
                self.selected_seq_id,
                &[],
                egui::Color32::from_rgb(167, 139, 250),
            ) {
                self.selected_seq_id = Some(clicked);
                output.clicked_seq_id = Some(clicked);
            }
        }

        ui.add_space(6.0);
        let target_sid = self
            .selected_seq_id
            .unwrap_or_else(|| self.cest_profiles[0].seq_id);
        let res_profs: Vec<&CestProfileDataset> = self
            .cest_profiles
            .iter()
            .filter(|p| p.seq_id == target_sid)
            .collect();
        if res_profs.is_empty() {
            return;
        }

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
                "CEST / DEST Saturation Profile — {}{} (ϖ_A = {:.2} ppm, Δϖ_AB = {:+.2} ppm)",
                res_profs[0].res_name, target_sid, rf.major_ppm, rf.dw_ab_ppm
            ),
            None => format!(
                "CEST / DEST Saturation Profile — {}{}",
                res_profs[0].res_name, target_sid
            ),
        };
        draw_xy_curve_canvas(
            ui,
            &title,
            "RF Carrier Offset ϖ_rf (ppm)",
            "I / I0",
            &obs_refs,
            &curves,
            &vlines,
            true,
        );
    }

    fn render_cpmg_tab(&mut self, ui: &mut egui::Ui, output: &mut DynamicsStudioOutput) {
        if self.cpmg_series.is_empty() {
            ui.label("No CPMG relaxation dispersion series loaded.");
            return;
        }

        if let Some(fit) = &self.cpmg_fit {
            ui.label(format!(
                "Global CPMG ({}) Fit: p_B = {:.2}% ± {:.2}%  |  k_ex = {:.0} ± {:.0} s⁻¹  |  χ²_red = {:.3}",
                fit.model.as_str(),
                fit.p_b * 100.0,
                fit.p_b_err * 100.0,
                fit.k_ex_s_inv,
                fit.k_ex_err,
                fit.reduced_chi2
            ));
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

        if let Some(clicked) = draw_sequence_bar_plot(
            ui,
            "ΔR2,eff (s⁻¹)",
            &bars,
            self.selected_seq_id,
            &[(2.0, egui::Color32::from_rgb(251, 191, 36))],
            egui::Color32::from_rgb(52, 211, 153),
        ) {
            self.selected_seq_id = Some(clicked);
            output.clicked_seq_id = Some(clicked);
        }

        ui.add_space(6.0);
        let target_sid = self
            .selected_seq_id
            .unwrap_or_else(|| self.cpmg_series[0].seq_id);
        let res_series: Vec<&CpmgSeriesDataset> = self
            .cpmg_series
            .iter()
            .filter(|s| s.seq_id == target_sid)
            .collect();
        if res_series.is_empty() {
            return;
        }

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
                res_series[0].res_name, target_sid, rf.dw_ppm, rf.dw_ppm_err
            ),
            None => format!(
                "CPMG Relaxation Dispersion — {}{} (Non-exchanging / Flat)",
                res_series[0].res_name, target_sid
            ),
        };
        draw_xy_curve_canvas(
            ui,
            &title,
            "ν_CPMG (Hz)",
            "R2,eff (s⁻¹)",
            &obs_refs,
            &curves,
            &[],
            false,
        );
    }

    fn render_csp_tab(&mut self, ui: &mut egui::Ui, output: &mut DynamicsStudioOutput) {
        if self.csp_trajectories.is_empty() {
            ui.label("No ligand titration HSQC CSP trajectories loaded.");
            return;
        }

        ui.horizontal_wrapped(|ui| {
            if let Some(fit) = &self.csp_fit {
                ui.label(format!(
                    "Global Quadratic Tight-Binding Kd = {:.2} µM (95% CI [{:.2}, {:.2}] µM)  |  Naive MM Kd = {:.2} µM",
                    fit.global_kd_um,
                    fit.global_kd_ci95_low_um,
                    fit.global_kd_ci95_high_um,
                    fit.global_naive_mm_kd_um,
                ));
            }
            ui.checkbox(
                &mut self.show_naive_mm_comparison,
                "Overlay naive [L]_free ≈ [L]_t curve",
            );
        });

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

        if let Some(clicked) = draw_sequence_bar_plot(
            ui,
            "Δδ_comb (ppm)",
            &bars,
            self.selected_seq_id,
            &hlines,
            egui::Color32::from_rgb(56, 189, 248),
        ) {
            self.selected_seq_id = Some(clicked);
            output.clicked_seq_id = Some(clicked);
        }

        ui.add_space(6.0);
        let target_sid = self
            .selected_seq_id
            .unwrap_or_else(|| self.csp_trajectories[0].seq_id);
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
        draw_xy_curve_canvas(
            ui,
            &title,
            "Total Ligand [L]_t (µM)",
            "Δδ_comb (ppm)",
            &[(&obs_plot, egui::Color32::from_rgb(125, 211, 252))],
            &curves,
            &[],
            false,
        );
    }
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
) -> Option<i32> {
    let desired = egui::vec2(ui.available_width().max(260.0), 145.0);
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
        return None;
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
                egui::FontId::monospace(9.5),
                egui::Color32::from_rgb(148, 163, 184),
            );
        }
    }

    painter.text(
        egui::pos2(rect.left() + 8.0, rect.top() + 4.0),
        egui::Align2::LEFT_TOP,
        y_label,
        egui::FontId::proportional(11.5),
        egui::Color32::from_rgb(226, 232, 240),
    );
    painter.text(
        egui::pos2(plot.left() - 4.0, plot.top()),
        egui::Align2::RIGHT_TOP,
        format!("{:.2}", y_min + y_span),
        egui::FontId::monospace(9.5),
        egui::Color32::from_rgb(148, 163, 184),
    );

    if let Some(h) = hovered_entry {
        response.on_hover_text(format!(
            "{}{} : {:.4} ± {:.4}",
            h.res_name, h.seq_id, h.value, h.err
        ));
    }

    clicked_sid
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
) {
    let desired = egui::vec2(ui.available_width().max(260.0), 190.0);
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
        return;
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
        egui::FontId::proportional(11.5),
        egui::Color32::from_rgb(226, 232, 240),
    );
    painter.text(
        egui::pos2(plot.center().x, rect.bottom() - 3.0),
        egui::Align2::CENTER_BOTTOM,
        format!("{x_label}   [{y_label}]"),
        egui::FontId::monospace(9.5),
        egui::Color32::from_rgb(148, 163, 184),
    );
    painter.text(
        egui::pos2(plot.left() - 4.0, plot.top()),
        egui::Align2::RIGHT_TOP,
        format!("{yb:.2}"),
        egui::FontId::monospace(9.5),
        egui::Color32::from_rgb(148, 163, 184),
    );
    painter.text(
        egui::pos2(plot.left() - 4.0, plot.bottom()),
        egui::Align2::RIGHT_BOTTOM,
        format!("{ya:.2}"),
        egui::FontId::monospace(9.5),
        egui::Color32::from_rgb(148, 163, 184),
    );
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
            let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
                let _ = panel.show(ui);
            });
        }

        let decay = build_exponential_decay_curve(1.0, 2.0, 1.0, 10);
        assert_eq!(decay.len(), 11);
        assert!((decay[0].1 - 1.0).abs() < 1e-12);
        assert!(decay[10].1 < decay[0].1);
    }
}
