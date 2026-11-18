//! Interactive 2D Diffusion-Ordered Spectroscopy (`DOSY`) studio panel (`T14.4`, desktop + `wasm32`).
//!
//! Provenance: Clean-room original human-computer interaction and scientific visualization widget
//! built on `nmr_core::dosy` implementing:
//! - Stejskal, E. O. & Tanner, J. E. (*J. Chem. Phys.* 1965, 42, 288–292 — pulsed-gradient
//!   attenuation $\ln(I / I_0) = -b D$ decay curves and multi-exponential curvature inspection).
//! - Wu, D. H., Chen, A. D. & Johnson, C. S. Jr. (*J. Magn. Reson. A* 1995, 115, 260–264 —
//!   `BPP-LED` diffusion encoding).
//! - Johnson, C. S. Jr. (*Prog. Nucl. Magn. Reson. Spectrosc.* 1999, 34, 203–256 — 2D
//!   $(\delta\text{ ppm}, \log_{10} D)$ Gaussian contour/spot display with aligned 1D top trace).
//! - Provencher, S. W. (*Comput. Phys. Commun.* 1982, 27, 213–227 — constrained Tikhonov
//!   Inverse Laplace Transform `ILT` distribution with explicit ill-posedness caveats).
//! - Einstein, A. (*Ann. Phys.* 1905, 322, 549–560) & Evans et al. (*Angew. Chem. Int. Ed.*
//!   2013, 52, 3199–3202 — Stokes–Einstein hydrodynamic radius $R_h$ and power-law `MW` readout).

use eframe::egui;
use nmr_core::dosy::{
    compute_b_values_s_m2, estimate_hydrodynamics, fit_stejskal_tanner_multi, solve_tikhonov_ilt,
    synthesize_dosy_grid_from_peaks, DosyFitConfig, DosyGrid2D, DosyGridConfig, DosyNucleus,
    DosyPeakEntry, DosySequenceType, DosySolvent, GradientShape, IltConfig, IltLambdaStrategy,
    ModelSelectionCriterion, SoluteCompactnessClass, ILT_ARTIFACT_CAVEAT,
};

/// Actions emitted by [`render_dosy_panel`] to synchronize with the surrounding workbench.
#[derive(Clone, Debug, PartialEq)]
pub enum DosyPanelAction {
    /// User clicked or selected peak index `idx` at chemical shift `ppm`.
    SelectedPeak { index: usize, ppm: f64 },
    /// User toggled Tikhonov `ILT` mode on or off.
    ToggledIlt { enabled: bool },
    /// User loaded the built-in synthetic multi-species mixture dataset.
    LoadedDemoMixture,
}

/// Coordinate mapper shared identically between the top 1D spectrum canvas and the 2D
/// `(ppm, log10(D))` DOSY canvas so every 1D multiplet aligns to the exact horizontal pixel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DosyViewportTransform {
    /// High-ppm (left) edge of the visible chemical shift window.
    pub ppm_left_max: f64,
    /// Low-ppm (right) edge of the visible chemical shift window.
    pub ppm_right_min: f64,
    /// Top edge of the 2D canvas in $\log_{10}(D / \text{m}^2\text{s}^{-1})$ (slower diffusion at top or bottom).
    pub log10_d_top: f64,
    /// Bottom edge of the 2D canvas in $\log_{10}(D / \text{m}^2\text{s}^{-1})$.
    pub log10_d_bottom: f64,
}

impl DosyViewportTransform {
    #[inline]
    fn effective_ppm_span(&self) -> f64 {
        let raw = self.ppm_left_max - self.ppm_right_min;
        if raw.abs() < 1.0e-6 {
            if raw < 0.0 {
                -1.0e-6
            } else {
                1.0e-6
            }
        } else {
            raw
        }
    }

    #[inline]
    fn effective_log10_d_span(&self) -> f64 {
        let raw = self.log10_d_bottom - self.log10_d_top;
        if raw.abs() < 1.0e-6 {
            if raw >= 0.0 {
                1.0e-6
            } else {
                -1.0e-6
            }
        } else {
            raw
        }
    }

    /// Map chemical shift `ppm` to screen `x` inside `rect` (standard NMR descending left-to-right).
    #[inline]
    pub fn ppm_to_x(&self, rect: egui::Rect, ppm: f64) -> f32 {
        let span = self.effective_ppm_span();
        let frac = ((self.ppm_left_max - ppm) / span) as f32;
        rect.left() + frac * rect.width()
    }

    /// Map screen `x` inside `rect` back to chemical shift `ppm`.
    #[inline]
    pub fn x_to_ppm(&self, rect: egui::Rect, x: f32) -> f64 {
        let w = rect.width().max(1.0);
        let frac = ((x - rect.left()) / w) as f64;
        self.ppm_left_max - frac * self.effective_ppm_span()
    }

    /// Map $\log_{10}(D)$ to screen `y` inside `rect_2d`.
    #[inline]
    pub fn log10_d_to_y(&self, rect_2d: egui::Rect, log10_d: f64) -> f32 {
        let denom = self.effective_log10_d_span();
        let frac = ((log10_d - self.log10_d_top) / denom) as f32;
        rect_2d.top() + frac * rect_2d.height()
    }

    /// Map screen `y` inside `rect_2d` back to $\log_{10}(D)$.
    #[inline]
    pub fn y_to_log10_d(&self, rect_2d: egui::Rect, y: f32) -> f64 {
        let h = rect_2d.height().max(1.0);
        let frac = ((y - rect_2d.top()) / h) as f64;
        self.log10_d_top + frac * self.effective_log10_d_span()
    }
}

/// Self-contained UI state for the interactive 2D `DOSY` `(ppm, log10(D))` workbench panel.
#[derive(Clone, Debug)]
pub struct DosyPanelState {
    /// Dataset title displayed in the header bar.
    pub dataset_title: String,
    /// Stejskal–Tanner / Wu–Johnson attenuation values $b_m$ (`s / m^2`).
    pub b_values_s_m2: Vec<f64>,
    /// Direct 1D chemical-shift axis (`ppm`, ascending or descending).
    pub ppm_axis: Vec<f64>,
    /// Lowest-gradient ($b_{\min}$) 1D reference trace aligned above the 2D DOSY canvas.
    pub top_spectrum_1d: Vec<f64>,
    /// Fitted peaks with discrete 1–3 exponential models (`AIC`/`BIC` + bootstrap CIs) and optional `ILT`.
    pub peaks: Vec<DosyPeakEntry>,
    /// Cached 2D `(ppm, log10(D))` synthesized contour/spot matrix.
    pub grid_2d: Option<DosyGrid2D>,
    /// Currently inspected peak index in `peaks`.
    pub selected_peak_idx: usize,
    /// Solvent used for Stokes–Einstein viscosity $\eta(T)$ and MW estimation.
    pub solvent: DosySolvent,
    /// Sample temperature $T$ (`K`, default `298.15 K`).
    pub temperature_k: f64,
    /// Compactness class for empirical power-law `MW` estimation.
    pub compactness_class: SoluteCompactnessClass,
    /// Active information criterion (`BIC` default vs `AIC`).
    pub criterion: ModelSelectionCriterion,
    /// Opt-in toggle for continuous Tikhonov-regularized Inverse Laplace Transform (`ILT`).
    pub enable_tikhonov_ilt: bool,
    /// Regularization selection strategy when `enable_tikhonov_ilt` is active.
    pub ilt_strategy: IltLambdaStrategy,
    /// Contour intensity floor as fraction of maximum 2D spot amplitude (`0.02..=0.50`).
    pub contour_floor_frac: f32,
    /// Number of geometric contour levels drawn per 2D diffusion spot (`3..=10`).
    pub contour_levels: usize,
    /// Viewport bounds synchronizing the top 1D trace and main 2D `(ppm, log10(D))` canvas.
    pub viewport: DosyViewportTransform,
    /// Grid synthesis configuration.
    pub grid_cfg: DosyGridConfig,
    /// Whether the right-side Inspector Table / decay column (`260.0 px`) is expanded.
    pub show_inspector_table: bool,
    /// Last rendered 2D + 1D DOSY canvas bounding rectangle.
    pub last_canvas_rect: Option<egui::Rect>,
    /// Optional status banner message.
    pub status_line: String,
}

impl Default for DosyPanelState {
    fn default() -> Self {
        Self::with_synthetic_demo()
    }
}

impl DosyPanelState {
    /// Create an empty `DosyPanelState` ready to receive ingested `DOSY` data.
    pub fn new_empty() -> Self {
        let grid_cfg = DosyGridConfig::default();
        Self {
            dataset_title: "No DOSY series loaded".to_string(),
            b_values_s_m2: Vec::new(),
            ppm_axis: Vec::new(),
            top_spectrum_1d: Vec::new(),
            peaks: Vec::new(),
            grid_2d: None,
            selected_peak_idx: 0,
            solvent: DosySolvent::D2O,
            temperature_k: 298.15,
            compactness_class: SoluteCompactnessClass::SmallMolecule,
            criterion: ModelSelectionCriterion::Bic,
            enable_tikhonov_ilt: false,
            ilt_strategy: IltLambdaStrategy::Gcv,
            contour_floor_frac: 0.06,
            contour_levels: 6,
            viewport: DosyViewportTransform {
                ppm_left_max: 9.0,
                ppm_right_min: 0.5,
                // Conventional DOSY vertical orientation: fast diffusion (larger log10 D) near top,
                // slower macromolecular diffusion near bottom, or vice versa. Putting -8.6 at top
                // and -10.8 at bottom places smaller solutes higher and macromolecules lower.
                log10_d_top: -8.6,
                log10_d_bottom: -10.8,
            },
            grid_cfg,
            show_inspector_table: false,
            last_canvas_rect: None,
            status_line: "Click 'Load Demo' or import Bruker difflist / Varian procpar."
                .to_string(),
        }
    }

    /// Construct a ready-to-inspect synthetic `BPP-LED` mixture containing:
    /// 1. `7.45 ppm`: pure macromolecular/oligomer species ($D_1 = 2.0 \times 10^{-10}\text{ m}^2/\text{s}$, `BIC -> K=1`),
    /// 2. `3.68 ppm`: overlapped mixed resonance ($D_1 = 2.0 \times 10^{-10}$ + $D_2 = 8.5 \times 10^{-10}\text{ m}^2/\text{s}$, `BIC -> K=2`),
    /// 3. `1.22 ppm`: pure small-molecule ligand ($D_2 = 8.5 \times 10^{-10}\text{ m}^2/\text{s}$, `BIC -> K=1`).
    pub fn with_synthetic_demo() -> Self {
        let mut state = Self::new_empty();
        state.dataset_title = "Synthetic BPP-LED Mixture (D2O, 298.15 K)".to_string();

        let grads: Vec<f64> = (0..20).map(|i| 2.0 + (i as f64) * (53.0 / 19.0)).collect();
        let b_vals = compute_b_values_s_m2(
            &grads,
            DosyNucleus::H1,
            DosySequenceType::BppLed,
            GradientShape::SineBell,
            0.100,
            0.0024,
            0.0002,
        )
        .unwrap_or_default();

        let n_ppm = 512usize;
        let ppm_axis: Vec<f64> = (0..n_ppm)
            .map(|i| 0.5 + (i as f64) * (8.5 / ((n_ppm - 1) as f64)))
            .collect();

        let d_slow = 2.0e-10_f64;
        let d_fast = 8.5e-10_f64;
        let fit_cfg = DosyFitConfig {
            bootstrap_samples: 80,
            rng_seed: 0x1404_D057,
            ..DosyFitConfig::default()
        };

        let specs = [
            (
                "Aromatic Slow Species (7.45 ppm)",
                7.45_f64,
                0.045_f64,
                vec![(1.00, d_slow)],
            ),
            (
                "Overlapped Mixture (3.68 ppm)",
                3.68_f64,
                0.050_f64,
                vec![(0.48, d_slow), (0.52, d_fast)],
            ),
            (
                "Aliphatic Ligand (1.22 ppm)",
                1.22_f64,
                0.040_f64,
                vec![(1.00, d_fast)],
            ),
        ];

        let mut top_1d = vec![0.0_f64; n_ppm];
        let mut peaks = Vec::with_capacity(specs.len());

        for (p_idx, (label, center_ppm, fwhm_ppm, comps)) in specs.into_iter().enumerate() {
            let mut rng = nmr_core::rng::Rng::new(9001 + p_idx as u64);
            let intensities: Vec<f64> = b_vals
                .iter()
                .map(|&b| {
                    let clean: f64 = comps
                        .iter()
                        .map(|&(amp, d)| 100.0 * amp * (-b * d).exp())
                        .sum();
                    clean + rng.normal(0.0, 0.30)
                })
                .collect();

            let hwhm = 0.5 * fwhm_ppm;
            let b0_amp = intensities.first().copied().unwrap_or(100.0);
            for (ci, &ppm) in ppm_axis.iter().enumerate() {
                let dx = (ppm - center_ppm) / hwhm;
                top_1d[ci] += b0_amp / (1.0 + dx * dx);
            }

            if let Ok(multi_exp) = fit_stejskal_tanner_multi(&b_vals, &intensities, &fit_cfg) {
                peaks.push(DosyPeakEntry {
                    label: label.to_string(),
                    center_ppm,
                    fwhm_ppm,
                    intensities,
                    multi_exp,
                    ilt: None,
                });
            }
        }

        state.b_values_s_m2 = b_vals;
        state.ppm_axis = ppm_axis;
        state.top_spectrum_1d = top_1d;
        state.peaks = peaks;
        state.selected_peak_idx = 1.min(state.peaks.len().saturating_sub(1));
        state.rebuild_2d_grid();
        state.status_line = format!(
            "Loaded {} DOSY peaks ({} gradient steps, BPP-LED sine-bell).",
            state.peaks.len(),
            state.b_values_s_m2.len()
        );
        state
    }

    /// Reset viewport zoom to span the full loaded `ppm_axis` and `grid_cfg` $\log_{10} D$ range.
    pub fn reset_viewport(&mut self) {
        if let (Some(&first), Some(&last)) = (self.ppm_axis.first(), self.ppm_axis.last()) {
            self.viewport.ppm_left_max = first.max(last);
            self.viewport.ppm_right_min = first.min(last);
        }
        self.viewport.log10_d_top = self.grid_cfg.log10_d_max;
        self.viewport.log10_d_bottom = self.grid_cfg.log10_d_min;
    }

    /// Resolve the active discrete fit (`1..=3` components) for `pk` according to `self.criterion`.
    pub fn active_fit_for_peak<'a>(
        &self,
        pk: &'a DosyPeakEntry,
    ) -> (usize, &'a nmr_core::dosy::DosyExpModelFit) {
        let score_of = |f: &nmr_core::dosy::DosyExpModelFit| match self.criterion {
            ModelSelectionCriterion::Bic => f.bic,
            ModelSelectionCriterion::Aic => f.aic,
        };
        if let Some(best) = pk
            .multi_exp
            .candidate_fits
            .iter()
            .filter(|f| f.is_well_resolved)
            .min_by(|a, b| score_of(a).total_cmp(&score_of(b)))
        {
            (best.k_components, best)
        } else {
            (pk.multi_exp.selected_k, &pk.multi_exp.best_fit)
        }
    }

    /// Ensure Tikhonov `ILT` distributions are computed for all peaks when `enable_tikhonov_ilt` is active.
    pub fn ensure_ilt_computed(&mut self) {
        if !self.enable_tikhonov_ilt || self.b_values_s_m2.len() < 4 {
            return;
        }
        let lo_ld = self
            .grid_cfg
            .log10_d_min
            .min(self.grid_cfg.log10_d_max - 0.1);
        let hi_ld = self.grid_cfg.log10_d_max.max(lo_ld + 0.1);
        let ilt_cfg = IltConfig {
            n_grid: 64,
            d_min_m2_s: 10.0_f64.powf(lo_ld),
            d_max_m2_s: 10.0_f64.powf(hi_ld),
            lambda_strategy: self.ilt_strategy,
            ..IltConfig::default()
        };
        for pk in &mut self.peaks {
            if pk.ilt.is_none() {
                pk.ilt = solve_tikhonov_ilt(&self.b_values_s_m2, &pk.intensities, &ilt_cfg).ok();
            }
        }
    }

    /// Re-synthesize `self.grid_2d` from `self.peaks` using discrete or `ILT` vertical profiles.
    pub fn rebuild_2d_grid(&mut self) {
        if self.ppm_axis.is_empty()
            || self.top_spectrum_1d.len() != self.ppm_axis.len()
            || self.peaks.is_empty()
        {
            self.grid_2d = None;
            return;
        }
        if self.enable_tikhonov_ilt {
            self.ensure_ilt_computed();
        }
        self.grid_2d = synthesize_dosy_grid_from_peaks(
            &self.ppm_axis,
            &self.top_spectrum_1d,
            &self.peaks,
            &self.grid_cfg,
            self.enable_tikhonov_ilt,
        )
        .ok();
    }

    /// Export fitted DOSY components and Stokes-Einstein hydrodynamics as TSV text.
    pub fn export_tsv(&self) -> String {
        let mut out = String::from(
            "peak_index\tlabel\tcenter_ppm\tselected_k\tcomp_index\tfraction\td_m2_s\tci95_low_m2_s\tci95_high_m2_s\tr_h_angstrom\test_mw_da\n",
        );
        for (pi, pk) in self.peaks.iter().enumerate() {
            let (k_sel, fit) = self.active_fit_for_peak(pk);
            for (ci, comp) in fit.components.iter().enumerate() {
                let (rh, mw) = estimate_hydrodynamics(
                    comp.d_m2_s,
                    self.solvent,
                    self.temperature_k,
                    self.compactness_class,
                )
                .map(|h| (h.r_h_angstrom, h.estimated_mw_da))
                .unwrap_or((0.0, 0.0));
                out.push_str(&format!(
                    "{}\t{}\t{:.4}\t{}\t{}\t{:.4}\t{:.4e}\t{:.4e}\t{:.4e}\t{:.2}\t{:.1}\n",
                    pi + 1,
                    pk.label,
                    pk.center_ppm,
                    k_sel,
                    ci + 1,
                    comp.fraction,
                    comp.d_m2_s,
                    comp.ci95_d_m2_s.0,
                    comp.ci95_d_m2_s.1,
                    rh,
                    mw
                ));
            }
        }
        out
    }
}

/// Render the complete interactive 2D `DOSY` studio panel (`wasm32`-compatible).
pub fn render_dosy_panel(ui: &mut egui::Ui, state: &mut DosyPanelState) -> Vec<DosyPanelAction> {
    let mut actions = Vec::new();

    // ---- Top Control Bar (<= 8 visible primary interactive controls) ----
    ui.horizontal_wrapped(|ui| {
        ui.label(
            egui::RichText::new("2D DOSY Diffusion Studio")
                .size(15.0)
                .strong(),
        );
        ui.separator();
        ui.label(
            egui::RichText::new(&state.dataset_title)
                .size(12.5)
                .strong(),
        );

        if ui.button("Fit Stejskal-Tanner").clicked() {
            state.rebuild_2d_grid();
            state.status_line = format!(
                "Fitted Stejskal-Tanner models across {} DOSY peaks.",
                state.peaks.len()
            );
        }
        if ui.button("Stokes-Einstein MW").clicked() {
            state.show_inspector_table = true;
            if let Some(pk) = state.peaks.get(state.selected_peak_idx) {
                let (_, fit) = state.active_fit_for_peak(pk);
                if let Some(c0) = fit.components.first() {
                    if let Ok(hyd) = estimate_hydrodynamics(
                        c0.d_m2_s,
                        state.solvent,
                        state.temperature_k,
                        state.compactness_class,
                    ) {
                        state.status_line = format!(
                            "Peak {:.2} ppm: R_h = {:.2} A, Est. MW = {:.0} Da ({})",
                            pk.center_ppm,
                            hyd.r_h_angstrom,
                            hyd.estimated_mw_da,
                            state.solvent.label()
                        );
                    }
                }
            }
        }
        if ui.button("Reset Zoom").clicked() {
            state.reset_viewport();
        }

        ui.separator();
        ui.label(egui::RichText::new("Solvent:").size(12.5));
        egui::ComboBox::from_id_salt("dosy_solvent_combo")
            .selected_text(state.solvent.label())
            .show_ui(ui, |ui| {
                for solv in DosySolvent::ALL {
                    ui.selectable_value(&mut state.solvent, solv, solv.label());
                }
            });

        if ui
            .selectable_label(state.show_inspector_table, "Inspector Table")
            .clicked()
        {
            state.show_inspector_table = !state.show_inspector_table;
        }
    });

    // Progressive-disclosure Advanced Parameters section (collapsed by default)
    egui::CollapsingHeader::new(egui::RichText::new("Advanced Parameters").size(12.5))
        .default_open(false)
        .show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                if ui.button("Load Demo").clicked() {
                    *state = DosyPanelState::with_synthetic_demo();
                    actions.push(DosyPanelAction::LoadedDemoMixture);
                }
                if ui.button("Export TSV").clicked() {
                    let tsv = state.export_tsv();
                    ui.ctx().copy_text(tsv);
                    state.status_line =
                        "Copied DOSY Stejskal-Tanner & MW TSV to clipboard.".to_string();
                }
                ui.separator();
                // Keep "T (K):" and its DragValue inside a nested horizontal block so they never split across rows
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("T (K):").size(12.5));
                    let mut t_k = state.temperature_k;
                    if ui
                        .add(
                            egui::DragValue::new(&mut t_k)
                                .range(273.15..=373.15)
                                .speed(0.5)
                                .fixed_decimals(2),
                        )
                        .changed()
                    {
                        state.temperature_k = t_k.clamp(273.15, 373.15);
                    }
                });

                if let Ok(eta) = state.solvent.viscosity_pa_s(state.temperature_k) {
                    ui.label(
                        egui::RichText::new(format!("eta = {:.3} mPa.s", eta * 1.0e3))
                            .size(11.0)
                            .color(egui::Color32::from_rgb(148, 163, 184)),
                    );
                }

                ui.separator();
                ui.label(egui::RichText::new("Class:").size(12.5));
                egui::ComboBox::from_id_salt("dosy_class_combo")
                    .selected_text(state.compactness_class.label())
                    .show_ui(ui, |ui| {
                        for cl in [
                            SoluteCompactnessClass::SmallMolecule,
                            SoluteCompactnessClass::GlobularProtein,
                            SoluteCompactnessClass::ExpandedPolymer,
                        ] {
                            ui.selectable_value(&mut state.compactness_class, cl, cl.label());
                        }
                    });

                ui.separator();
                ui.label(egui::RichText::new("Criterion:").size(12.5));
                let crit_lbl = match state.criterion {
                    ModelSelectionCriterion::Bic => "BIC",
                    ModelSelectionCriterion::Aic => "AIC",
                };
                egui::ComboBox::from_id_salt("dosy_crit_combo")
                    .selected_text(crit_lbl)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut state.criterion,
                            ModelSelectionCriterion::Bic,
                            "BIC (strict)",
                        );
                        ui.selectable_value(
                            &mut state.criterion,
                            ModelSelectionCriterion::Aic,
                            "AIC (Akaike)",
                        );
                    });

                ui.separator();
                let mut ilt_on = state.enable_tikhonov_ilt;
                if ui
                    .checkbox(&mut ilt_on, "Tikhonov ILT (opt-in)")
                    .on_hover_text(ILT_ARTIFACT_CAVEAT)
                    .changed()
                {
                    state.enable_tikhonov_ilt = ilt_on;
                    state.rebuild_2d_grid();
                    actions.push(DosyPanelAction::ToggledIlt { enabled: ilt_on });
                }

                if state.enable_tikhonov_ilt {
                    let cur_lbl = match state.ilt_strategy {
                        IltLambdaStrategy::Gcv => "GCV",
                        IltLambdaStrategy::LCurve => "L-Curve",
                        IltLambdaStrategy::DiscrepancyPrinciple { .. } => "Discrepancy",
                        IltLambdaStrategy::Fixed(_) => "Fixed lambda",
                    };
                    let mut changed_strat = false;
                    egui::ComboBox::from_id_salt("dosy_ilt_strat_combo")
                        .selected_text(cur_lbl)
                        .show_ui(ui, |ui| {
                            if ui
                                .selectable_label(
                                    matches!(state.ilt_strategy, IltLambdaStrategy::Gcv),
                                    "GCV",
                                )
                                .clicked()
                            {
                                state.ilt_strategy = IltLambdaStrategy::Gcv;
                                changed_strat = true;
                            }
                            if ui
                                .selectable_label(
                                    matches!(state.ilt_strategy, IltLambdaStrategy::LCurve),
                                    "L-Curve",
                                )
                                .clicked()
                            {
                                state.ilt_strategy = IltLambdaStrategy::LCurve;
                                changed_strat = true;
                            }
                        });
                    if changed_strat {
                        for pk in &mut state.peaks {
                            pk.ilt = None;
                        }
                        state.rebuild_2d_grid();
                    }
                }

                ui.separator();
                ui.label(egui::RichText::new("Contour floor:").size(11.0));
                ui.add(
                    egui::Slider::new(&mut state.contour_floor_frac, 0.02..=0.50)
                        .fixed_decimals(2)
                        .show_value(true),
                );
                ui.label(
                    egui::RichText::new(format!("Peaks: {}", state.peaks.len()))
                        .size(11.0)
                        .color(egui::Color32::from_rgb(148, 163, 184)),
                );
            });
        });

    // Explicit scientific caveat banner when Tikhonov ILT is enabled
    if state.enable_tikhonov_ilt {
        egui::Frame::NONE
            .fill(egui::Color32::from_rgb(56, 38, 14))
            .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(217, 119, 6)))
            .inner_margin(egui::Margin::symmetric(8, 5))
            .corner_radius(4.0)
            .show(ui, |ui| {
                ui.label(
                    egui::RichText::new(format!("[!] {ILT_ARTIFACT_CAVEAT}"))
                        .size(11.0)
                        .color(egui::Color32::from_rgb(253, 224, 71)),
                );
            });
    }

    ui.add_space(2.0);

    if state.peaks.is_empty() {
        ui.centered_and_justified(|ui| {
            ui.label(egui::RichText::new(&state.status_line).size(12.5));
        });
        return actions;
    }

    // ---- Main Two-Column Split: Left = Aligned 1D + 2D (ppm, log10(D)), Right = Collapsible 260 px Inspector ----
    let avail = ui.available_size();
    let total_h = avail.y.max(380.0);

    if state.show_inspector_table && avail.x >= 680.0 {
        let right_w = 260.0_f32;
        let left_w = (avail.x - right_w - 8.0).max(320.0);
        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui.set_width(left_w);
                draw_dosy_2d_canvas(ui, state, left_w, total_h, &mut actions);
            });
            ui.separator();
            ui.vertical(|ui| {
                ui.set_width(right_w);
                render_right_decay_and_hydrodynamics(ui, state, right_w, total_h, &mut actions);
            });
        });
    } else {
        let left_w = avail.x.max(280.0);
        draw_dosy_2d_canvas(ui, state, left_w, total_h, &mut actions);
    }

    actions
}

pub fn draw_dosy_2d_canvas(
    ui: &mut egui::Ui,
    state: &mut DosyPanelState,
    width: f32,
    height: f32,
    actions: &mut Vec<DosyPanelAction>,
) {
    let left_gutter = 58.0_f32;
    let right_pad = 6.0_f32;
    let top_1d_h = (height * 0.22).clamp(72.0, 130.0);
    let bottom_axis_h = 30.0_f32;
    let map_2d_h = (height - top_1d_h - bottom_axis_h - 4.0).max(160.0);

    // 1. Top aligned 1D trace region
    let (full_1d_rect, _) =
        ui.allocate_exact_size(egui::vec2(width, top_1d_h), egui::Sense::hover());
    let plot_1d_rect = egui::Rect::from_min_max(
        egui::pos2(full_1d_rect.left() + left_gutter, full_1d_rect.top() + 4.0),
        egui::pos2(
            (full_1d_rect.right() - right_pad).max(full_1d_rect.left() + left_gutter + 20.0),
            full_1d_rect.bottom() - 2.0,
        ),
    );

    let painter_1d = ui.painter_at(full_1d_rect);
    painter_1d.rect_filled(plot_1d_rect, 3.0, egui::Color32::from_rgb(15, 20, 30));
    painter_1d.rect_stroke(
        plot_1d_rect,
        3.0,
        egui::Stroke::new(1.0, egui::Color32::from_rgb(51, 65, 85)),
        egui::StrokeKind::Inside,
    );
    painter_1d.text(
        egui::pos2(full_1d_rect.left() + 4.0, plot_1d_rect.center().y),
        egui::Align2::LEFT_CENTER,
        "1D I(b0)",
        egui::FontId::proportional(11.0),
        egui::Color32::from_rgb(148, 163, 184),
    );

    if !state.ppm_axis.is_empty() && state.top_spectrum_1d.len() == state.ppm_axis.len() {
        let max_y = state
            .top_spectrum_1d
            .iter()
            .fold(1.0e-12_f64, |a, &b| a.max(b.abs()));
        let trace_painter = painter_1d.with_clip_rect(plot_1d_rect);
        let mut pts = Vec::with_capacity(state.ppm_axis.len());
        let p_lo = state
            .viewport
            .ppm_right_min
            .min(state.viewport.ppm_left_max);
        let p_hi = state
            .viewport
            .ppm_right_min
            .max(state.viewport.ppm_left_max);
        for (&ppm, &val) in state.ppm_axis.iter().zip(state.top_spectrum_1d.iter()) {
            if ppm < p_lo || ppm > p_hi {
                continue;
            }
            let x = state.viewport.ppm_to_x(plot_1d_rect, ppm);
            let yn = (val / max_y).clamp(-0.08, 1.05) as f32;
            let y = plot_1d_rect.bottom() - 4.0 - yn * (plot_1d_rect.height() - 10.0);
            pts.push(egui::pos2(x, y));
        }
        if pts.len() >= 2 {
            trace_painter.add(egui::Shape::line(
                pts,
                egui::Stroke::new(1.3, egui::Color32::from_rgb(125, 211, 252)),
            ));
        }
    }

    // 2. Main 2D (ppm, log10(D)) contour/spot plot sharing identical horizontal [left..right] bounds!
    let (full_2d_rect, resp_2d) = ui.allocate_exact_size(
        egui::vec2(width, map_2d_h + bottom_axis_h),
        egui::Sense::click_and_drag(),
    );
    let plot_2d_rect = egui::Rect::from_min_max(
        egui::pos2(full_2d_rect.left() + left_gutter, full_2d_rect.top()),
        egui::pos2(plot_1d_rect.right(), full_2d_rect.top() + map_2d_h),
    );
    state.last_canvas_rect = Some(full_1d_rect.union(full_2d_rect));

    let painter_2d = ui.painter_at(full_2d_rect);
    painter_2d.rect_filled(plot_2d_rect, 3.0, egui::Color32::from_rgb(11, 15, 24));
    painter_2d.rect_stroke(
        plot_2d_rect,
        3.0,
        egui::Stroke::new(1.0, egui::Color32::from_rgb(64, 79, 104)),
        egui::StrokeKind::Inside,
    );

    // Explicit y-axis title on the left margin
    painter_2d.text(
        egui::pos2(full_2d_rect.left() + 2.0, plot_2d_rect.top() + 2.0),
        egui::Align2::LEFT_TOP,
        "log10(D / m^2 s^-1)",
        egui::FontId::proportional(11.0),
        egui::Color32::from_rgb(148, 163, 184),
    );

    // Horizontal log10(D) grid lines & labels (padded in 0.08..=0.92 and clamped so top tick never clips)
    let n_y_ticks = 5usize;
    for t in 0..=n_y_ticks {
        let frac = 0.08 + 0.84 * ((t as f64) / (n_y_ticks as f64));
        let ld = state.viewport.log10_d_top
            + frac * (state.viewport.log10_d_bottom - state.viewport.log10_d_top);
        let y_raw = state.viewport.log10_d_to_y(plot_2d_rect, ld);
        let y = y_raw.clamp(plot_2d_rect.top() + 18.0, plot_2d_rect.bottom() - 8.0);
        painter_2d.line_segment(
            [
                egui::pos2(plot_2d_rect.left(), y),
                egui::pos2(plot_2d_rect.right(), y),
            ],
            egui::Stroke::new(0.6, egui::Color32::from_rgb(30, 41, 59)),
        );
        painter_2d.text(
            egui::pos2(plot_2d_rect.left() - 5.0, y),
            egui::Align2::RIGHT_CENTER,
            format!("{ld:.2}"),
            egui::FontId::monospace(11.0),
            egui::Color32::from_rgb(148, 163, 184),
        );
    }

    // Vertical ppm grid lines & bottom axis ticks (formatted as "{:.1}" with single bottom title)
    let n_x_ticks = 6usize;
    for t in 0..=n_x_ticks {
        let frac = (t as f64) / (n_x_ticks as f64);
        let ppm = state.viewport.ppm_left_max
            - frac * (state.viewport.ppm_left_max - state.viewport.ppm_right_min);
        let x = state.viewport.ppm_to_x(plot_2d_rect, ppm);
        painter_2d.line_segment(
            [
                egui::pos2(x, plot_2d_rect.top()),
                egui::pos2(x, plot_2d_rect.bottom()),
            ],
            egui::Stroke::new(0.6, egui::Color32::from_rgb(30, 41, 59)),
        );
        let align_x = if t == 0 {
            egui::Align2::LEFT_CENTER
        } else if t == n_x_ticks {
            egui::Align2::RIGHT_CENTER
        } else {
            egui::Align2::CENTER_CENTER
        };
        painter_2d.text(
            egui::pos2(x, plot_2d_rect.bottom() + 8.0),
            align_x,
            format!("{ppm:.1}"),
            egui::FontId::monospace(11.0),
            egui::Color32::from_rgb(148, 163, 184),
        );
    }
    painter_2d.text(
        egui::pos2(plot_2d_rect.center().x, plot_2d_rect.bottom() + 21.0),
        egui::Align2::CENTER_CENTER,
        "1H Chemical Shift (ppm)",
        egui::FontId::proportional(11.0),
        egui::Color32::from_rgb(186, 202, 224),
    );

    // Render 2D DOSY elliptical contour spots + vertical alignment guides
    let spot_painter = painter_2d.with_clip_rect(plot_2d_rect);
    let guide_1d_painter = painter_1d.with_clip_rect(plot_1d_rect);
    for (p_idx, pk) in state.peaks.iter().enumerate() {
        let is_sel = p_idx == state.selected_peak_idx;
        let x_pk = state.viewport.ppm_to_x(plot_2d_rect, pk.center_ppm);
        if x_pk < plot_2d_rect.left() || x_pk > plot_2d_rect.right() {
            continue;
        }

        let guide_col = if is_sel {
            egui::Color32::from_rgba_unmultiplied(250, 204, 21, 95)
        } else {
            egui::Color32::from_rgba_unmultiplied(100, 116, 139, 45)
        };
        guide_1d_painter.line_segment(
            [
                egui::pos2(x_pk, plot_1d_rect.top()),
                egui::pos2(x_pk, plot_1d_rect.bottom()),
            ],
            egui::Stroke::new(1.0, guide_col),
        );
        spot_painter.line_segment(
            [
                egui::pos2(x_pk, plot_2d_rect.top()),
                egui::pos2(x_pk, plot_2d_rect.bottom()),
            ],
            egui::Stroke::new(1.0, guide_col),
        );

        let rx_base = ((state
            .viewport
            .ppm_to_x(plot_2d_rect, pk.center_ppm - pk.fwhm_ppm)
            - x_pk)
            .abs())
        .clamp(4.0, 28.0);

        if state.enable_tikhonov_ilt {
            if let Some(ilt) = &pk.ilt {
                for mode in &ilt.modes {
                    if mode.area_fraction < state.contour_floor_frac as f64 {
                        continue;
                    }
                    let y_m = state.viewport.log10_d_to_y(plot_2d_rect, mode.log10_d);
                    let ry_base = ((state
                        .viewport
                        .log10_d_to_y(plot_2d_rect, mode.log10_d + mode.sd_log10_d)
                        - y_m)
                        .abs())
                    .clamp(4.0, 36.0);
                    draw_contour_spot(
                        &spot_painter,
                        egui::pos2(x_pk, y_m),
                        rx_base,
                        ry_base,
                        state.contour_levels.clamp(3, 10),
                        egui::Color32::from_rgb(192, 132, 252),
                        is_sel,
                    );
                }
                continue;
            }
        }

        let (_, active_fit) = state.active_fit_for_peak(pk);
        for (c_idx, comp) in active_fit.components.iter().enumerate() {
            if comp.fraction < state.contour_floor_frac as f64 {
                continue;
            }
            let y_c = state.viewport.log10_d_to_y(plot_2d_rect, comp.log10_d);
            let sig_ld = comp.sd_log10_d.clamp(0.03, 0.25);
            let ry_base = ((state
                .viewport
                .log10_d_to_y(plot_2d_rect, comp.log10_d + sig_ld)
                - y_c)
                .abs())
            .clamp(4.5, 34.0);
            let base_col = match c_idx {
                0 => egui::Color32::from_rgb(56, 189, 248),
                1 => egui::Color32::from_rgb(52, 211, 153),
                _ => egui::Color32::from_rgb(251, 146, 60),
            };
            draw_contour_spot(
                &spot_painter,
                egui::pos2(x_pk, y_c),
                rx_base,
                ry_base,
                state.contour_levels.clamp(3, 10),
                base_col,
                is_sel,
            );
        }
    }

    // Drag to pan synchronized 1D + 2D viewport
    if resp_2d.dragged() {
        let delta = resp_2d.drag_delta();
        if delta.length_sq() > 0.0 {
            let d_ppm = (delta.x as f64 / plot_2d_rect.width().max(1.0) as f64)
                * state.viewport.effective_ppm_span();
            let d_ld = (delta.y as f64 / plot_2d_rect.height().max(1.0) as f64)
                * state.viewport.effective_log10_d_span();
            state.viewport.ppm_left_max += d_ppm;
            state.viewport.ppm_right_min += d_ppm;
            state.viewport.log10_d_top -= d_ld;
            state.viewport.log10_d_bottom -= d_ld;
        }
    }

    // Scroll wheel zoom anchored at cursor inside 2D DOSY canvas
    if let Some(hover) = resp_2d.hover_pos() {
        if plot_2d_rect.contains(hover) {
            let scroll_y = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll_y.abs() > 0.5 {
                let factor = (1.0 - (scroll_y as f64) * 0.0025).clamp(0.75, 1.35);
                let anchor_ppm = state.viewport.x_to_ppm(plot_2d_rect, hover.x);
                let anchor_ld = state.viewport.y_to_log10_d(plot_2d_rect, hover.y);
                state.viewport.ppm_left_max =
                    anchor_ppm + (state.viewport.ppm_left_max - anchor_ppm) * factor;
                state.viewport.ppm_right_min =
                    anchor_ppm + (state.viewport.ppm_right_min - anchor_ppm) * factor;
                state.viewport.log10_d_top =
                    anchor_ld + (state.viewport.log10_d_top - anchor_ld) * factor;
                state.viewport.log10_d_bottom =
                    anchor_ld + (state.viewport.log10_d_bottom - anchor_ld) * factor;
            }
        }
    }

    // Click to select nearest peak
    if resp_2d.clicked() {
        if let Some(pos) = resp_2d.interact_pointer_pos() {
            if plot_2d_rect.contains(pos) {
                let clicked_ppm = state.viewport.x_to_ppm(plot_2d_rect, pos.x);
                if let Some((best_i, best_pk)) = state.peaks.iter().enumerate().min_by(|a, b| {
                    (a.1.center_ppm - clicked_ppm)
                        .abs()
                        .total_cmp(&(b.1.center_ppm - clicked_ppm).abs())
                }) {
                    state.selected_peak_idx = best_i;
                    actions.push(DosyPanelAction::SelectedPeak {
                        index: best_i,
                        ppm: best_pk.center_ppm,
                    });
                }
            }
        }
    }

    // Hover crosshair readout (ppm, log10(D), D, R_h)
    if let Some(hover) = resp_2d.hover_pos() {
        if plot_2d_rect.contains(hover) {
            let h_ppm = state.viewport.x_to_ppm(plot_2d_rect, hover.x);
            let h_ld = state.viewport.y_to_log10_d(plot_2d_rect, hover.y);
            let h_d = 10.0_f64.powf(h_ld);
            let rh_str = estimate_hydrodynamics(
                h_d,
                state.solvent,
                state.temperature_k,
                state.compactness_class,
            )
            .map(|h| {
                format!(
                    "R_h={:.1} Å, MW≈{:.0} Da",
                    h.r_h_angstrom, h.estimated_mw_da
                )
            })
            .unwrap_or_default();
            painter_2d.text(
                egui::pos2(plot_2d_rect.left() + 6.0, plot_2d_rect.top() + 6.0),
                egui::Align2::LEFT_TOP,
                format!(
                    "delta={h_ppm:.3} ppm | log10(D)={h_ld:.2} (D={:.2e} m^2/s) | {rh_str}",
                    h_d
                ),
                egui::FontId::monospace(11.0),
                egui::Color32::from_rgb(226, 232, 240),
            );
        }
    }
}

fn draw_contour_spot(
    painter: &egui::Painter,
    center: egui::Pos2,
    rx: f32,
    ry: f32,
    n_levels: usize,
    color: egui::Color32,
    highlighted: bool,
) {
    let segs = 24usize;
    for lvl in 0..n_levels {
        let scale = 1.0 - (lvl as f32) / ((n_levels as f32) + 0.5);
        let mut ring = Vec::with_capacity(segs + 1);
        for s in 0..=segs {
            let ang = (s as f32) * std::f32::consts::TAU / (segs as f32);
            ring.push(egui::pos2(
                center.x + rx * scale * ang.cos(),
                center.y + ry * scale * ang.sin(),
            ));
        }
        let width = if highlighted && lvl == 0 { 1.9 } else { 1.1 };
        painter.add(egui::Shape::line(ring, egui::Stroke::new(width, color)));
    }
    if highlighted {
        painter.circle_filled(center, 2.5, egui::Color32::from_rgb(250, 204, 21));
    }
}

fn render_right_decay_and_hydrodynamics(
    ui: &mut egui::Ui,
    state: &mut DosyPanelState,
    width: f32,
    height: f32,
    actions: &mut Vec<DosyPanelAction>,
) {
    let sel_idx = state
        .selected_peak_idx
        .min(state.peaks.len().saturating_sub(1));
    state.selected_peak_idx = sel_idx;

    // Peak selector + BIC model-order badge
    ui.horizontal_wrapped(|ui| {
        ui.label(egui::RichText::new("Peak:").size(12.5));
        let cur_label = state.peaks[sel_idx].label.clone();
        egui::ComboBox::from_id_salt("dosy_peak_selector")
            .selected_text(cur_label)
            .show_ui(ui, |ui| {
                for (idx, pk) in state.peaks.iter().enumerate() {
                    if ui
                        .selectable_value(&mut state.selected_peak_idx, idx, &pk.label)
                        .clicked()
                    {
                        actions.push(DosyPanelAction::SelectedPeak {
                            index: idx,
                            ppm: pk.center_ppm,
                        });
                    }
                }
            });

        let pk = &state.peaks[state.selected_peak_idx];
        let (k_sel, _) = state.active_fit_for_peak(pk);
        let crit_prefix = match state.criterion {
            ModelSelectionCriterion::Bic => "BIC",
            ModelSelectionCriterion::Aic => "AIC",
        };
        let (badge_bg, badge_fg, badge_txt) = match k_sel {
            1 => (
                egui::Color32::from_rgb(20, 83, 45),
                egui::Color32::from_rgb(134, 239, 172),
                format!("{crit_prefix}: K = 1 (Mono-exp)"),
            ),
            2 => (
                egui::Color32::from_rgb(30, 58, 138),
                egui::Color32::from_rgb(147, 197, 253),
                format!("{crit_prefix}: K = 2 (Bi-exp)"),
            ),
            _ => (
                egui::Color32::from_rgb(124, 45, 18),
                egui::Color32::from_rgb(253, 186, 116),
                format!("{crit_prefix}: K = 3 (Tri-exp)"),
            ),
        };
        egui::Frame::NONE
            .fill(badge_bg)
            .stroke(egui::Stroke::new(1.0, badge_fg))
            .inner_margin(egui::Margin::symmetric(7, 3))
            .corner_radius(4.0)
            .show(ui, |ui| {
                ui.label(
                    egui::RichText::new(badge_txt)
                        .size(11.0)
                        .strong()
                        .color(badge_fg),
                );
            });
    });

    let pk = &state.peaks[state.selected_peak_idx];
    let (active_k, active_fit) = state.active_fit_for_peak(pk);

    // BIC / AIC comparison strip across candidate K
    ui.horizontal_wrapped(|ui| {
        for cand in &pk.multi_exp.candidate_fits {
            let tag = if cand.k_components == active_k {
                "*"
            } else {
                " "
            };
            ui.label(
                egui::RichText::new(format!(
                    "{tag}K={}: BIC={:.1}, AIC={:.1}{}",
                    cand.k_components,
                    cand.bic,
                    cand.aic,
                    if cand.is_well_resolved {
                        ""
                    } else {
                        " (unres.)"
                    }
                ))
                .monospace()
                .size(11.0)
                .color(egui::Color32::from_rgb(148, 163, 184)),
            );
        }
    });

    ui.add_space(4.0);

    // Stokes-Einstein R_h & MW Component Table
    egui::Frame::NONE
        .fill(egui::Color32::from_rgb(17, 24, 39))
        .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(51, 65, 85)))
        .inner_margin(egui::Margin::same(6))
        .corner_radius(4.0)
        .show(ui, |ui| {
            ui.label(
                egui::RichText::new("Stokes-Einstein & 95% CIs")
                    .strong()
                    .size(11.0),
            );
            egui::Grid::new("dosy_comp_hydro_grid")
                .striped(true)
                .spacing([4.0, 3.0])
                .show(ui, |ui| {
                    ui.label(egui::RichText::new("#").size(11.0).strong());
                    ui.label(egui::RichText::new("Frac").size(11.0).strong());
                    ui.label(egui::RichText::new("D").size(11.0).strong());
                    ui.label(egui::RichText::new("95% CI").size(11.0).strong());
                    ui.label(egui::RichText::new("Rh").size(11.0).strong());
                    ui.label(egui::RichText::new("MW").size(11.0).strong());
                    ui.end_row();

                    for (ci, comp) in active_fit.components.iter().enumerate() {
                        ui.label(
                            egui::RichText::new(format!("#{}", ci + 1))
                                .monospace()
                                .size(11.0),
                        );
                        ui.label(
                            egui::RichText::new(format!("{:.1}%", comp.fraction * 100.0))
                                .monospace()
                                .size(11.0),
                        );
                        ui.label(
                            egui::RichText::new(format!("{:.2}", comp.d_m2_s * 1.0e10))
                                .monospace()
                                .size(11.0),
                        );
                        ui.label(
                            egui::RichText::new(format!(
                                "[{:.1},{:.1}]",
                                comp.ci95_d_m2_s.0 * 1.0e10,
                                comp.ci95_d_m2_s.1 * 1.0e10
                            ))
                            .monospace()
                            .size(11.0),
                        );
                        if let Ok(hyd) = estimate_hydrodynamics(
                            comp.d_m2_s,
                            state.solvent,
                            state.temperature_k,
                            state.compactness_class,
                        ) {
                            ui.label(
                                egui::RichText::new(format!("{:.1}", hyd.r_h_angstrom))
                                    .monospace()
                                    .size(11.0)
                                    .color(egui::Color32::from_rgb(125, 211, 252)),
                            );
                            let mw_txt = if hyd.estimated_mw_da >= 2_500.0 {
                                format!("{:.1}k", hyd.estimated_mw_da * 1.0e-3)
                            } else {
                                format!("{:.0}", hyd.estimated_mw_da)
                            };
                            ui.label(
                                egui::RichText::new(mw_txt)
                                    .monospace()
                                    .size(11.0)
                                    .color(egui::Color32::from_rgb(134, 239, 172)),
                            );
                        } else {
                            ui.label(egui::RichText::new("-").size(11.0));
                            ui.label(egui::RichText::new("-").size(11.0));
                        }
                        ui.end_row();
                    }
                });
        });

    ui.add_space(6.0);

    // Stejskal-Tanner ln(I / I0) vs b Curve + Residuals (+ Optional ILT Plot)
    let rem_h = (height - 150.0).max(180.0);
    let has_ilt_plot = state.enable_tikhonov_ilt && pk.ilt.is_some();
    let decay_h = if has_ilt_plot {
        (rem_h * 0.44).max(85.0)
    } else {
        (rem_h * 0.65).max(110.0)
    };
    let resid_h = (rem_h * 0.25).max(52.0);

    render_stejskal_tanner_log_plot(
        ui,
        &state.b_values_s_m2,
        &pk.intensities,
        active_fit,
        width,
        decay_h,
    );
    ui.add_space(4.0);
    render_residuals_plot(ui, &state.b_values_s_m2, active_fit, width, resid_h);

    if let (true, Some(ilt)) = (state.enable_tikhonov_ilt, &pk.ilt) {
        ui.add_space(4.0);
        let ilt_h = (rem_h - decay_h - resid_h - 12.0).max(68.0);
        render_ilt_distribution_plot(ui, ilt, active_fit, width, ilt_h);
    }
}

fn render_stejskal_tanner_log_plot(
    ui: &mut egui::Ui,
    b_vals: &[f64],
    intensities: &[f64],
    fit: &nmr_core::dosy::DosyExpModelFit,
    width: f32,
    height: f32,
) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    let inner = egui::Rect::from_min_max(
        egui::pos2(rect.left() + 42.0, rect.top() + 16.0),
        egui::pos2(
            (rect.right() - 8.0).max(rect.left() + 60.0),
            rect.bottom() - 14.0,
        ),
    );
    painter.rect_filled(inner, 3.0, egui::Color32::from_rgb(13, 18, 28));
    painter.rect_stroke(
        inner,
        3.0,
        egui::Stroke::new(1.0, egui::Color32::from_rgb(51, 65, 85)),
        egui::StrokeKind::Inside,
    );
    painter.text(
        egui::pos2(inner.left(), rect.top() + 2.0),
        egui::Align2::LEFT_TOP,
        "Stejskal-Tanner: ln(I / I0) vs b",
        egui::FontId::proportional(11.0),
        egui::Color32::from_rgb(203, 213, 225),
    );

    if b_vals.is_empty() || intensities.len() != b_vals.len() {
        return;
    }

    let i0_sum: f64 = fit
        .components
        .iter()
        .map(|c| c.amplitude)
        .sum::<f64>()
        .max(intensities[0].abs().max(1.0e-12));
    let b_max = b_vals.iter().fold(1.0e-12_f64, |a, &b| a.max(b));

    let mut ln_min = -0.5_f64;
    for &yv in &fit.fitted_curve {
        if yv > 1.0e-4 * i0_sum {
            ln_min = ln_min.min((yv / i0_sum).ln());
        }
    }
    ln_min = (ln_min - 0.25).clamp(-6.5, -0.5);
    let ln_max = 0.15_f64;

    let map_pt = |b: f64, ln_v: f64| -> egui::Pos2 {
        let xf = (b / b_max).clamp(0.0, 1.0) as f32;
        let yf = ((ln_max - ln_v.clamp(ln_min, ln_max)) / (ln_max - ln_min)) as f32;
        egui::pos2(
            inner.left() + xf * inner.width(),
            inner.top() + yf * inner.height(),
        )
    };

    // Individual component straight lines in ln(I / I0)
    if fit.components.len() > 1 {
        for (ci, comp) in fit.components.iter().enumerate() {
            let p0 = map_pt(0.0, comp.fraction.max(1.0e-6).ln());
            let p1 = map_pt(b_max, comp.fraction.max(1.0e-6).ln() - b_max * comp.d_m2_s);
            let col = if ci == 0 {
                egui::Color32::from_rgba_unmultiplied(56, 189, 248, 130)
            } else {
                egui::Color32::from_rgba_unmultiplied(52, 211, 153, 130)
            };
            painter.line_segment([p0, p1], egui::Stroke::new(1.0, col));
        }
    }

    // Total fitted curve
    let mut fit_pts = Vec::with_capacity(64);
    for s in 0..=60 {
        let b = (s as f64) * b_max / 60.0;
        let yh: f64 = fit
            .components
            .iter()
            .map(|c| c.amplitude * (-b * c.d_m2_s).exp())
            .sum();
        if yh > 1.0e-6 * i0_sum {
            fit_pts.push(map_pt(b, (yh / i0_sum).ln()));
        }
    }
    if fit_pts.len() >= 2 {
        painter.add(egui::Shape::line(
            fit_pts,
            egui::Stroke::new(1.8, egui::Color32::from_rgb(250, 204, 21)),
        ));
    }

    // Experimental data points
    for (&b, &yi) in b_vals.iter().zip(intensities.iter()) {
        if yi > 0.01 * i0_sum {
            let pt = map_pt(b, (yi / i0_sum).ln());
            painter.circle_filled(pt, 2.8, egui::Color32::from_rgb(248, 250, 252));
        }
    }

    painter.text(
        egui::pos2(inner.left() - 4.0, inner.top()),
        egui::Align2::RIGHT_TOP,
        "0.0",
        egui::FontId::monospace(11.0),
        egui::Color32::from_rgb(148, 163, 184),
    );
    painter.text(
        egui::pos2(inner.left() - 4.0, inner.bottom()),
        egui::Align2::RIGHT_BOTTOM,
        format!("{ln_min:.1}"),
        egui::FontId::monospace(11.0),
        egui::Color32::from_rgb(148, 163, 184),
    );
    painter.text(
        egui::pos2(inner.right(), rect.bottom() - 1.0),
        egui::Align2::RIGHT_BOTTOM,
        format!("b_max = {:.2}e9 s/m^2", b_max * 1.0e-9),
        egui::FontId::monospace(11.0),
        egui::Color32::from_rgb(148, 163, 184),
    );
}

fn render_residuals_plot(
    ui: &mut egui::Ui,
    b_vals: &[f64],
    fit: &nmr_core::dosy::DosyExpModelFit,
    width: f32,
    height: f32,
) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    let inner = egui::Rect::from_min_max(
        egui::pos2(rect.left() + 42.0, rect.top() + 14.0),
        egui::pos2(
            (rect.right() - 8.0).max(rect.left() + 60.0),
            rect.bottom() - 3.0,
        ),
    );
    painter.rect_filled(inner, 3.0, egui::Color32::from_rgb(13, 18, 28));
    painter.rect_stroke(
        inner,
        3.0,
        egui::Stroke::new(1.0, egui::Color32::from_rgb(51, 65, 85)),
        egui::StrokeKind::Inside,
    );
    painter.text(
        egui::pos2(inner.left(), rect.top()),
        egui::Align2::LEFT_TOP,
        format!("Residuals I_obs - I_fit (RMSE = {:.3})", fit.rmse),
        egui::FontId::proportional(11.0),
        egui::Color32::from_rgb(148, 163, 184),
    );

    let zero_y = inner.center().y;
    painter.line_segment(
        [
            egui::pos2(inner.left(), zero_y),
            egui::pos2(inner.right(), zero_y),
        ],
        egui::Stroke::new(0.8, egui::Color32::from_rgb(71, 85, 105)),
    );

    let res = &fit.residuals;
    if b_vals.is_empty() || res.len() != b_vals.len() {
        return;
    }
    let b_max = b_vals.iter().fold(1.0e-12_f64, |a, &b| a.max(b));
    let r_max = res.iter().fold(1.0e-9_f64, |a, &r| a.max(r.abs())) * 1.15;

    for (&b, &rv) in b_vals.iter().zip(res.iter()) {
        let xf = (b / b_max).clamp(0.0, 1.0) as f32;
        let x = inner.left() + xf * inner.width();
        let y = zero_y - ((rv / r_max) as f32) * (0.45 * inner.height());
        painter.line_segment(
            [egui::pos2(x, zero_y), egui::pos2(x, y)],
            egui::Stroke::new(1.2, egui::Color32::from_rgb(56, 189, 248)),
        );
        painter.circle_filled(
            egui::pos2(x, y),
            2.2,
            egui::Color32::from_rgb(125, 211, 252),
        );
    }
}

fn render_ilt_distribution_plot(
    ui: &mut egui::Ui,
    ilt: &nmr_core::dosy::IltResult,
    fit: &nmr_core::dosy::DosyExpModelFit,
    width: f32,
    height: f32,
) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    let inner = egui::Rect::from_min_max(
        egui::pos2(rect.left() + 42.0, rect.top() + 14.0),
        egui::pos2(
            (rect.right() - 8.0).max(rect.left() + 60.0),
            rect.bottom() - 4.0,
        ),
    );
    painter.rect_filled(inner, 3.0, egui::Color32::from_rgb(20, 16, 34));
    painter.rect_stroke(
        inner,
        3.0,
        egui::Stroke::new(1.0, egui::Color32::from_rgb(126, 34, 206)),
        egui::StrokeKind::Inside,
    );
    painter.text(
        egui::pos2(inner.left(), rect.top()),
        egui::Align2::LEFT_TOP,
        format!(
            "Tikhonov ILT f(log10 D) [lambda* = {:.2e}, {} modes]",
            ilt.selected_lambda,
            ilt.modes.len()
        ),
        egui::FontId::proportional(11.0),
        egui::Color32::from_rgb(216, 180, 254),
    );

    if ilt.log10_d_grid.len() < 2 || ilt.spectrum.len() != ilt.log10_d_grid.len() {
        return;
    }
    let l_min = ilt.log10_d_grid[0];
    let l_max = *ilt.log10_d_grid.last().unwrap_or(&-8.5);
    let s_max = ilt.spectrum.iter().fold(1.0e-12_f64, |a, &b| a.max(b));

    let mut pts = Vec::with_capacity(ilt.spectrum.len());
    for (&ld, &sv) in ilt.log10_d_grid.iter().zip(ilt.spectrum.iter()) {
        let xf = ((ld - l_min) / (l_max - l_min).max(1.0e-9)) as f32;
        let yf = (sv / s_max).clamp(0.0, 1.0) as f32;
        pts.push(egui::pos2(
            inner.left() + xf * inner.width(),
            inner.bottom() - 2.0 - yf * (inner.height() - 6.0),
        ));
    }
    painter.add(egui::Shape::line(
        pts,
        egui::Stroke::new(1.6, egui::Color32::from_rgb(192, 132, 252)),
    ));

    // Overlay active discrete component markers for direct comparison
    for comp in &fit.components {
        let xf = ((comp.log10_d - l_min) / (l_max - l_min).max(1.0e-9)) as f32;
        if (0.0..=1.0).contains(&xf) {
            let x = inner.left() + xf * inner.width();
            painter.line_segment(
                [egui::pos2(x, inner.top()), egui::pos2(x, inner.bottom())],
                egui::Stroke::new(
                    1.0,
                    egui::Color32::from_rgba_unmultiplied(250, 204, 21, 180),
                ),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dosy_panel_demo_state_and_headless_egui_render() {
        let mut state = DosyPanelState::with_synthetic_demo();
        assert_eq!(state.peaks.len(), 3);
        // Pure slow peak -> K=1, mixed peak -> K=2, pure fast peak -> K=1
        assert_eq!(state.peaks[0].multi_exp.selected_k, 1);
        assert_eq!(state.peaks[1].multi_exp.selected_k, 2);
        assert_eq!(state.peaks[2].multi_exp.selected_k, 1);
        assert!(state.grid_2d.is_some());

        // Verify exact horizontal alignment between top 1D trace and 2D DOSY canvas
        let r1 = egui::Rect::from_min_max(egui::pos2(56.0, 10.0), egui::pos2(520.0, 90.0));
        let r2 = egui::Rect::from_min_max(egui::pos2(56.0, 95.0), egui::pos2(520.0, 420.0));
        for ppm in [7.45, 3.68, 1.22] {
            let x1 = state.viewport.ppm_to_x(r1, ppm);
            let x2 = state.viewport.ppm_to_x(r2, ppm);
            assert!((x1 - x2).abs() < 1.0e-5);
            let back = state.viewport.x_to_ppm(r2, x2);
            assert!((back - ppm).abs() < 1.0e-6);
        }

        // Verify log10(D) vertical coordinate round-trip (standard and inverted orientations)
        for ld in [-8.9, -9.7, -10.5] {
            let y = state.viewport.log10_d_to_y(r2, ld);
            let back_ld = state.viewport.y_to_log10_d(r2, y);
            assert!((back_ld - ld).abs() < 1.0e-6);
        }

        // Toggle ILT on and run headless egui frames across BIC/AIC and solvents
        state.enable_tikhonov_ilt = true;
        state.rebuild_2d_grid();
        assert!(state.peaks[1].ilt.is_some());

        let ctx = egui::Context::default();
        let mut out1 = ctx.run_ui(egui::RawInput::default(), |ui| {
            let _actions = render_dosy_panel(ui, &mut state);
        });
        out1.textures_delta.clear();

        state.criterion = ModelSelectionCriterion::Aic;
        state.solvent = DosySolvent::CDCl3;
        state.temperature_k = 303.15;
        let mut out2 = ctx.run_ui(egui::RawInput::default(), |ui| {
            let _actions = render_dosy_panel(ui, &mut state);
        });
        out2.textures_delta.clear();
    }

    #[test]
    fn dosy_panel_empty_and_degenerate_viewport_safe() {
        let mut empty = DosyPanelState::new_empty();
        assert!(empty.grid_2d.is_none());

        // Degenerate zero-width ppm/log10_d viewport must never produce NaN/Inf
        empty.viewport.ppm_left_max = 4.0;
        empty.viewport.ppm_right_min = 4.0;
        empty.viewport.log10_d_top = -9.5;
        empty.viewport.log10_d_bottom = -9.5;
        let rect = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(200.0, 200.0));
        let x = empty.viewport.ppm_to_x(rect, 4.0);
        let y = empty.viewport.log10_d_to_y(rect, -9.5);
        assert!(x.is_finite() && y.is_finite());
        assert!((empty.viewport.x_to_ppm(rect, x) - 4.0).abs() < 1.0e-8);
        assert!((empty.viewport.y_to_log10_d(rect, y) - (-9.5)).abs() < 1.0e-8);

        let ctx = egui::Context::default();
        let mut out = ctx.run_ui(egui::RawInput::default(), |ui| {
            let actions = render_dosy_panel(ui, &mut empty);
            assert!(actions.is_empty());
        });
        out.textures_delta.clear();
    }
}
