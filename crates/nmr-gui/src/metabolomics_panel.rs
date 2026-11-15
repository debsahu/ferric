//! Interactive 4-Quadrant 1D NMR Metabolomics, `PCA` / `OPLS-DA` Chemometrics & `STOCSY`
//! Biomarker Discovery Studio (`T14.5`, native desktop + `wasm32-unknown-unknown`).
//!
//! Provenance: Clean-room original human-computer interaction and plotting implementation of
//! `docs/specs/19-metabolomics-chemometrics.md` §9.2 built strictly on `eframe::egui` and
//! `nmr_core::metabolomics` (Savorani et al. *J. Magn. Reson.* 2010; De Meyer et al. *Anal. Chem.*
//! 2008; Dieterle et al. *Anal. Chem.* 2006; Trygg & Wold *J. Chemom.* 2002; Bylesjö et al.
//! *J. Chemom.* 2006; Westerhuis et al. *Metabolomics* 2008; Wiklund et al. *Anal. Chem.* 2008;
//! Cloarec et al. *Anal. Chem.* 2005; CC0 `BMRB` metabolomics reference library).

use eframe::egui;
use nmr_core::metabolomics::{
    compute_stocsy, export_biomarker_ranking_csv, export_bucket_table_csv,
    export_study_summary_json, generate_demo_biofluid_cohort, run_metabolomics_pipeline,
    AlignmentMode, AlignmentReference, BiofluidMaskPreset, BucketingMethod, CohortSample1D,
    ConfidenceEllipse2D, MetabolomicsPipelineConfig, MetabolomicsStudyResult, NormalizationMethod,
    ScalingMethod, StocsyResult,
};

/// Active trace visualization mode in Quadrant A (top-left).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpectrumViewMode {
    /// Diluted-corrected cohort 1D overlay colored by sample group.
    CohortOverlay,
    /// Cloarec et al. (2005) `STOCSY` driver covariance trace heat-colored by Pearson $r^2 \in [0, 1]$.
    StocsyCovarianceHeat,
}

/// Active multivariate score-space projection in Quadrant B (top-right).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScorePlotMode {
    /// `OPLS-DA` predictive score $t_p$ vs first orthogonal score $t_{o,1}$.
    OplsDa,
    /// Unsupervised `PCA` `PC1` vs `PC2`.
    Pca,
    /// Supervised NIPALS `PLS-DA` `LV1` vs `LV2`.
    PlsDa,
}

/// Active feature-selection plot in Quadrant C (bottom-left).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeaturePlotMode {
    /// Wiklund et al. (2008) `OPLS-DA` `S-plot`: $\operatorname{Cov}(t_p, \tilde{X}_j)$ vs $\operatorname{Corr}(t_p, \tilde{X}_j)$.
    OplsSPlot,
    /// Univariate Welch $t$-test Volcano plot: $\log_2(\text{FC}_j)$ vs $-\log_{10}(q_{\text{BH},j})$.
    VolcanoPlot,
}

/// Map `STOCSY` squared correlation $r^2 \in [0, 1]$ to a perceptual cool-to-hot RGB color
/// (deep slate-blue `0.0` -> cyan `0.35` -> gold `0.70` -> bright crimson-red `>= 0.90`).
pub fn stocsy_r2_color(r2: f64) -> egui::Color32 {
    let t = r2.clamp(0.0, 1.0) as f32;
    let (r, g, b) = if t < 0.35 {
        let u = t / 0.35;
        (
            38.0 + u * (34.0 - 38.0),
            78.0 + u * (211.0 - 78.0),
            165.0 + u * (238.0 - 165.0),
        )
    } else if t < 0.70 {
        let u = (t - 0.35) / 0.35;
        (
            34.0 + u * (250.0 - 34.0),
            211.0 + u * (204.0 - 211.0),
            238.0 + u * (21.0 - 238.0),
        )
    } else {
        let u = (t - 0.70) / 0.30;
        (
            250.0 + u * (239.0 - 250.0),
            204.0 + u * (52.0 - 204.0),
            21.0 + u * (42.0 - 21.0),
        )
    };
    egui::Color32::from_rgb(r.round() as u8, g.round() as u8, b.round() as u8)
}

fn group_color(group_idx: usize) -> egui::Color32 {
    const PALETTE: &[egui::Color32] = &[
        egui::Color32::from_rgb(56, 189, 248),  // Sky blue (Control)
        egui::Color32::from_rgb(251, 113, 133), // Coral rose (Case)
        egui::Color32::from_rgb(74, 222, 128),  // Emerald
        egui::Color32::from_rgb(250, 204, 21),  // Amber
        egui::Color32::from_rgb(192, 132, 252), // Violet
    ];
    PALETTE[group_idx % PALETTE.len()]
}

/// Interactive state for the 4-Quadrant Metabolomics & Biomarker Discovery Studio.
pub struct MetabolomicsStudioPanel {
    pub samples: Vec<CohortSample1D>,
    pub config: MetabolomicsPipelineConfig,
    pub study: Option<MetabolomicsStudyResult>,
    pub active_stocsy: Option<StocsyResult>,
    pub selected_bucket_idx: Option<usize>,

    pub spectrum_mode: SpectrumViewMode,
    pub score_mode: ScorePlotMode,
    pub feature_mode: FeaturePlotMode,
    pub show_bucket_boundaries: bool,
    pub auto_zoom_on_select: bool,

    /// Visible horizontal chemical-shift window `(low_ppm, high_ppm)` in Quadrant A.
    pub view_ppm_bounds: (f64, f64),
    pub status_banner: String,
    pub demo_seed: u64,
}

impl Default for MetabolomicsStudioPanel {
    fn default() -> Self {
        Self::new()
    }
}

impl MetabolomicsStudioPanel {
    /// Create a fresh studio panel; click "Load 24-Sample Biofluid Demo" or supply cohort spectra.
    pub fn new() -> Self {
        let config = MetabolomicsPipelineConfig::default();
        let view_bounds = (config.min_ppm, config.max_ppm);
        Self {
            samples: Vec::new(),
            config,
            study: None,
            active_stocsy: None,
            selected_bucket_idx: None,
            spectrum_mode: SpectrumViewMode::CohortOverlay,
            score_mode: ScorePlotMode::OplsDa,
            feature_mode: FeaturePlotMode::OplsSPlot,
            show_bucket_boundaries: true,
            auto_zoom_on_select: true,
            view_ppm_bounds: view_bounds,
            status_banner:
                "No cohort loaded — click 'Load 24-Sample Biofluid Demo' to run untargeted 1D metabolomics."
                    .to_string(),
            demo_seed: 20261115,
        }
    }

    /// Load a cohort of 1D samples and execute the full metabolomics + `OPLS-DA` + `STOCSY` pipeline.
    pub fn set_cohort_and_run(&mut self, samples: Vec<CohortSample1D>) {
        self.samples = samples;
        self.recompute_pipeline();
    }

    /// Generate the built-in 24-spectrum synthetic Case vs Control human biofluid cohort and solve.
    pub fn load_demo_cohort(&mut self) {
        let cohort = generate_demo_biofluid_cohort(self.demo_seed);
        self.view_ppm_bounds = (self.config.min_ppm, self.config.max_ppm);
        self.set_cohort_and_run(cohort);
    }

    /// Re-run the full `nmr_core::metabolomics::run_metabolomics_pipeline` on `self.samples`.
    pub fn recompute_pipeline(&mut self) {
        if self.samples.is_empty() {
            self.study = None;
            self.active_stocsy = None;
            self.selected_bucket_idx = None;
            return;
        }
        let res = run_metabolomics_pipeline(&self.samples, &self.config);
        let default_b = res.biomarkers.first().map(|b| b.bucket_index);
        self.active_stocsy = res.top_biomarker_stocsy.clone();
        self.selected_bucket_idx = default_b;
        let n_s = res.sample_ids.len();
        let n_b = res.buckets.len();
        let summary_txt = if let Some(ref op) = res.opls_da {
            format!(
                "Analyzed {n_s} spectra -> {n_b} buckets | OPLS-DA R²Y={:.3}, CV Q²={:.3}, perm p={:.3} ({})",
                op.r2_y,
                op.q2,
                op.permutation.p_value_q2,
                if op.permutation.is_validated {
                    "VALIDATED"
                } else {
                    "OVERFIT WARNING"
                }
            )
        } else {
            format!("Analyzed {n_s} spectra -> {n_b} buckets (PCA only)")
        };
        self.status_banner = summary_txt;
        self.study = Some(res);
    }

    /// Select a bucket index (`0..P`), compute live `STOCSY` + `BMRB` matching, and optionally zoom Quadrant A.
    pub fn select_bucket_and_run_stocsy(&mut self, bucket_idx: usize, zoom_to_bucket: bool) {
        let Some(ref study) = self.study else {
            return;
        };
        if bucket_idx >= study.buckets.len() {
            return;
        }
        self.selected_bucket_idx = Some(bucket_idx);
        let b = &study.buckets[bucket_idx];
        let center = b.center_ppm;
        if zoom_to_bucket {
            let half_span = 0.45_f64.max(b.width_ppm * 5.0);
            let lo = (center - half_span).max(study.config.min_ppm);
            let hi = (center + half_span).min(study.config.max_ppm);
            if hi > lo + 0.02 {
                self.view_ppm_bounds = (lo, hi);
            }
        }
        self.active_stocsy = compute_stocsy(
            bucket_idx,
            &study.buckets,
            &study.normalized_bucket_matrix,
            Some((&study.aligned_spectra, &study.dilution_factors)),
            &study.config.mask_preset.zones(),
        );
        if let Some(ref st) = self.active_stocsy {
            let top_hit = st
                .metabolite_matches
                .first()
                .map(|m| format!("{} ({}, score {:.2})", m.name, m.bmrb_id, m.score))
                .unwrap_or_else(|| "no BMRB hit".to_string());
            self.status_banner = format!(
                "STOCSY driver B{:03} @ {:.3} ppm -> top match: {}",
                bucket_idx, st.driver_ppm, top_hit
            );
        }
    }

    /// Render the complete 4-Quadrant Metabolomics & Biomarker Discovery Studio.
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        self.draw_top_toolbar(ui);
        ui.add_space(4.0);

        if self.study.is_none() {
            self.draw_empty_state(ui);
            return;
        }

        let avail = ui.available_size();
        let half_h = ((avail.y - 8.0) * 0.5).max(210.0);
        let half_w = ((avail.x - 8.0) * 0.5).max(280.0);

        // Top Row: Quadrant A (1D Overlay / STOCSY) + Quadrant B (PCA / OPLS-DA Score Plot)
        ui.horizontal(|ui| {
            ui.allocate_ui(egui::vec2(half_w, half_h), |ui| {
                self.draw_quadrant_a_spectra(ui, egui::vec2(half_w, half_h));
            });
            ui.add_space(6.0);
            ui.allocate_ui(egui::vec2(half_w, half_h), |ui| {
                self.draw_quadrant_b_scores(ui, egui::vec2(half_w, half_h));
            });
        });

        ui.add_space(6.0);

        // Bottom Row: Quadrant C (Clickable S-Plot / Volcano) + Quadrant D (Biomarker & BMRB Table)
        ui.horizontal(|ui| {
            ui.allocate_ui(egui::vec2(half_w, half_h), |ui| {
                self.draw_quadrant_c_splot_volcano(ui, egui::vec2(half_w, half_h));
            });
            ui.add_space(6.0);
            ui.allocate_ui(egui::vec2(half_w, half_h), |ui| {
                self.draw_quadrant_d_biomarker_table(ui, egui::vec2(half_w, half_h));
            });
        });
    }

    fn draw_empty_state(&mut self, ui: &mut egui::Ui) {
        egui::Frame::new()
            .fill(egui::Color32::from_rgb(16, 22, 32))
            .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(45, 58, 82)))
            .corner_radius(6.0)
            .inner_margin(egui::Margin::same(24))
            .show(ui, |ui| {
                ui.vertical_centered(|ui| {
                    ui.add_space(28.0);
                    ui.heading(
                        egui::RichText::new(
                            "Untargeted 1D NMR Metabolomics, Chemometrics & STOCSY Biomarker Studio",
                        )
                        .color(egui::Color32::from_rgb(140, 220, 255)),
                    );
                    ui.add_space(8.0);
                    ui.label(
                        "Interval FFT Peak Alignment • Adaptive Intelligent Binning • PQN Normalization • PCA / OPLS-DA • S-Plot / Volcano • Live STOCSY + 28-Metabolite CC0 BMRB Library",
                    );
                    ui.add_space(16.0);
                    if ui
                        .add(
                            egui::Button::new(
                                egui::RichText::new("▶ Load 24-Sample Biofluid Demo Cohort")
                                    .size(14.5)
                                    .strong()
                                    .color(egui::Color32::WHITE),
                            )
                            .fill(egui::Color32::from_rgb(22, 118, 178))
                            .min_size(egui::vec2(290.0, 36.0)),
                        )
                        .clicked()
                    {
                        self.load_demo_cohort();
                    }
                    ui.add_space(28.0);
                });
            });
    }

    fn draw_top_toolbar(&mut self, ui: &mut egui::Ui) {
        let mut trigger_rerun = false;

        egui::Frame::new()
            .fill(egui::Color32::from_rgb(18, 24, 35))
            .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(44, 56, 78)))
            .corner_radius(5.0)
            .inner_margin(egui::Margin::symmetric(8, 6))
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    if ui
                        .add(
                            egui::Button::new(
                                egui::RichText::new("Load 24-Sample Demo")
                                    .strong()
                                    .color(egui::Color32::from_rgb(185, 245, 255)),
                            )
                            .fill(egui::Color32::from_rgb(24, 82, 128)),
                        )
                        .on_hover_text(
                            "Synthesize 24 realistic human biofluid 1D 1H spectra (12 Control, 12 Case) with sample dilutions, pH jitter, and Lactate/Valine/TMAO biomarkers",
                        )
                        .clicked()
                    {
                        self.demo_seed = self.demo_seed.wrapping_add(1);
                        self.load_demo_cohort();
                    }

                    ui.separator();

                    // Bucketing selector
                    let is_adaptive =
                        matches!(self.config.bucketing, BucketingMethod::AdaptiveIntelligent { .. });
                    ui.label(egui::RichText::new("Bucketing:").small());
                    if ui.selectable_label(is_adaptive, "Adaptive (AIB)").clicked() && !is_adaptive
                    {
                        self.config.bucketing = BucketingMethod::default();
                        trigger_rerun = true;
                    }
                    if ui.selectable_label(!is_adaptive, "Uniform 0.04").clicked() && is_adaptive {
                        self.config.bucketing = BucketingMethod::Uniform {
                            bin_width_ppm: 0.04,
                            min_bucket_snr: 3.0,
                        };
                        trigger_rerun = true;
                    }

                    ui.separator();

                    // Alignment toggle
                    let mut align_on = !matches!(self.config.alignment, AlignmentMode::Disabled);
                    if ui
                        .checkbox(&mut align_on, "Interval FFT Align")
                        .on_hover_text(
                            "Savorani et al. (2010) valley-snapped FFT cross-correlation local peak alignment",
                        )
                        .changed()
                    {
                        self.config.alignment = if align_on {
                            AlignmentMode::AutomaticIntervals {
                                interval_width_ppm: 0.24,
                                max_shift_ppm: 0.05,
                                reference: AlignmentReference::Median,
                            }
                        } else {
                            AlignmentMode::Disabled
                        };
                        trigger_rerun = true;
                    }

                    ui.separator();

                    // Normalization combo
                    let norm_label = match self.config.normalization {
                        NormalizationMethod::ProbabilisticQuotient => "PQN",
                        NormalizationMethod::TotalSpectralArea => "TSA",
                        NormalizationMethod::InternalStandard { .. } => "TSP (0 ppm)",
                        NormalizationMethod::None => "None",
                    };
                    egui::ComboBox::from_label("Norm")
                        .selected_text(norm_label)
                        .width(92.0)
                        .show_ui(ui, |ui| {
                            if ui
                                .selectable_label(
                                    matches!(
                                        self.config.normalization,
                                        NormalizationMethod::ProbabilisticQuotient
                                    ),
                                    "PQN (Dieterle 2006)",
                                )
                                .clicked()
                            {
                                self.config.normalization =
                                    NormalizationMethod::ProbabilisticQuotient;
                                trigger_rerun = true;
                            }
                            if ui
                                .selectable_label(
                                    matches!(
                                        self.config.normalization,
                                        NormalizationMethod::TotalSpectralArea
                                    ),
                                    "Total Area (TSA)",
                                )
                                .clicked()
                            {
                                self.config.normalization = NormalizationMethod::TotalSpectralArea;
                                trigger_rerun = true;
                            }
                            if ui
                                .selectable_label(
                                    matches!(
                                        self.config.normalization,
                                        NormalizationMethod::InternalStandard { .. }
                                    ),
                                    "Internal Std (TSP 0.0)",
                                )
                                .clicked()
                            {
                                self.config.normalization = NormalizationMethod::InternalStandard {
                                    target_ppm: 0.0,
                                    half_window_ppm: 0.08,
                                };
                                trigger_rerun = true;
                            }
                            if ui
                                .selectable_label(
                                    matches!(self.config.normalization, NormalizationMethod::None),
                                    "None",
                                )
                                .clicked()
                            {
                                self.config.normalization = NormalizationMethod::None;
                                trigger_rerun = true;
                            }
                        });

                    // Scaling combo
                    let scale_label = match self.config.scaling {
                        ScalingMethod::Pareto => "Pareto",
                        ScalingMethod::UnitVariance => "UV",
                        ScalingMethod::MeanCenter => "Center",
                        ScalingMethod::None => "None",
                    };
                    egui::ComboBox::from_label("Scale")
                        .selected_text(scale_label)
                        .width(76.0)
                        .show_ui(ui, |ui| {
                            for (m, lbl) in [
                                (ScalingMethod::Pareto, "Pareto"),
                                (ScalingMethod::UnitVariance, "Unit Variance (UV)"),
                                (ScalingMethod::MeanCenter, "Mean Center"),
                                (ScalingMethod::None, "None"),
                            ] {
                                if ui.selectable_label(self.config.scaling == m, lbl).clicked() {
                                    self.config.scaling = m;
                                    trigger_rerun = true;
                                }
                            }
                        });

                    // Mask preset combo
                    let mask_label = match self.config.mask_preset {
                        BiofluidMaskPreset::Urine => "Urine (H2O+Urea)",
                        BiofluidMaskPreset::SerumPlasma => "Serum (H2O)",
                        BiofluidMaskPreset::AqueousMinimal => "H2O Only",
                        BiofluidMaskPreset::None => "No Mask",
                        BiofluidMaskPreset::Custom(_) => "Custom",
                    };
                    egui::ComboBox::from_label("Mask")
                        .selected_text(mask_label)
                        .width(118.0)
                        .show_ui(ui, |ui| {
                            if ui
                                .selectable_label(
                                    self.config.mask_preset == BiofluidMaskPreset::Urine,
                                    "Urine (H2O + Urea + TSP)",
                                )
                                .clicked()
                            {
                                self.config.mask_preset = BiofluidMaskPreset::Urine;
                                trigger_rerun = true;
                            }
                            if ui
                                .selectable_label(
                                    self.config.mask_preset == BiofluidMaskPreset::SerumPlasma,
                                    "Serum / Plasma (H2O + TSP)",
                                )
                                .clicked()
                            {
                                self.config.mask_preset = BiofluidMaskPreset::SerumPlasma;
                                trigger_rerun = true;
                            }
                            if ui
                                .selectable_label(
                                    self.config.mask_preset == BiofluidMaskPreset::AqueousMinimal,
                                    "Minimal H2O (4.70-4.95)",
                                )
                                .clicked()
                            {
                                self.config.mask_preset = BiofluidMaskPreset::AqueousMinimal;
                                trigger_rerun = true;
                            }
                            if ui
                                .selectable_label(
                                    self.config.mask_preset == BiofluidMaskPreset::None,
                                    "None",
                                )
                                .clicked()
                            {
                                self.config.mask_preset = BiofluidMaskPreset::None;
                                trigger_rerun = true;
                            }
                        });

                    if !self.samples.is_empty() && ui.button("↻ Re-Run").clicked() {
                        trigger_rerun = true;
                    }

                    if let Some(ref st) = self.study {
                        ui.separator();
                        if ui
                            .small_button("Copy Biomarkers CSV")
                            .on_hover_text("Copy ranked candidate biomarker table CSV to clipboard")
                            .clicked()
                        {
                            ui.ctx().copy_text(export_biomarker_ranking_csv(st));
                            self.status_banner =
                                "Copied ranked biomarker discovery CSV to clipboard.".to_string();
                        }
                        if ui
                            .small_button("Copy Buckets CSV")
                            .on_hover_text("Copy normalized N x P bucket matrix CSV to clipboard")
                            .clicked()
                        {
                            ui.ctx().copy_text(export_bucket_table_csv(st));
                            self.status_banner =
                                "Copied normalized bucket matrix CSV to clipboard.".to_string();
                        }
                        if ui
                            .small_button("Copy Summary JSON")
                            .on_hover_text("Copy structured JSON study summary to clipboard")
                            .clicked()
                        {
                            if let Ok(js) = export_study_summary_json(st, 20) {
                                ui.ctx().copy_text(js);
                                self.status_banner =
                                    "Copied JSON study summary to clipboard.".to_string();
                            }
                        }
                    }
                });

                // Second row: Model Validation & Overfitting Guardrail Banner
                if let Some(ref study) = self.study {
                    ui.add_space(3.0);
                    ui.horizontal_wrapped(|ui| {
                        let pca_r2 = study
                            .pca
                            .cumulative_variance_ratio
                            .get(1)
                            .or_else(|| study.pca.cumulative_variance_ratio.first())
                            .copied()
                            .unwrap_or(0.0);
                        ui.label(
                            egui::RichText::new(format!(
                                "Cohort: {} spectra × {} buckets | PCA 2-comp R²X={:.1}%",
                                study.sample_ids.len(),
                                study.buckets.len(),
                                pca_r2 * 100.0
                            ))
                            .small()
                            .monospace()
                            .color(egui::Color32::from_rgb(175, 205, 235)),
                        );

                        if let Some(ref op) = study.opls_da {
                            ui.separator();
                            let (badge_txt, fill_c, txt_c) = if op.permutation.is_validated {
                                (
                                    format!(
                                        "✓ OPLS-DA VALIDATED (R²Y={:.3}, CV Q²={:.3}, perm p={:.3})",
                                        op.r2_y, op.q2, op.permutation.p_value_q2
                                    ),
                                    egui::Color32::from_rgb(20, 72, 48),
                                    egui::Color32::from_rgb(167, 243, 208),
                                )
                            } else {
                                (
                                    format!(
                                        "⚠ OVERFIT / UNVALIDATED (R²Y={:.3}, CV Q²={:.3}, perm p={:.3})",
                                        op.r2_y, op.q2, op.permutation.p_value_q2
                                    ),
                                    egui::Color32::from_rgb(98, 34, 34),
                                    egui::Color32::from_rgb(254, 202, 202),
                                )
                            };
                            egui::Frame::new()
                                .fill(fill_c)
                                .corner_radius(3.0)
                                .inner_margin(egui::Margin::symmetric(6, 2))
                                .show(ui, |ui| {
                                    ui.label(
                                        egui::RichText::new(badge_txt)
                                            .small()
                                            .monospace()
                                            .strong()
                                            .color(txt_c),
                                    );
                                });
                        }

                        ui.separator();
                        ui.label(
                            egui::RichText::new(&self.status_banner)
                                .small()
                                .color(egui::Color32::from_rgb(203, 213, 225)),
                        );
                    });
                }
            });

        if trigger_rerun {
            self.recompute_pipeline();
        }
    }

    // ========================================================================
    // Quadrant A: Cohort 1D Spectral Overlay & Live STOCSY r^2 Heat Trace
    // ========================================================================

    fn draw_quadrant_a_spectra(&mut self, ui: &mut egui::Ui, size: egui::Vec2) {
        let mut clicked_bucket: Option<usize> = None;

        egui::Frame::new()
            .fill(egui::Color32::from_rgb(14, 18, 26))
            .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(42, 54, 74)))
            .corner_radius(5.0)
            .inner_margin(egui::Margin::same(6))
            .show(ui, |ui| {
                ui.set_min_size(size - egui::vec2(12.0, 12.0));
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("A. Cohort 1D Overlay & STOCSY Trace")
                            .strong()
                            .color(egui::Color32::from_rgb(135, 215, 255)),
                    );
                    ui.separator();
                    ui.selectable_value(
                        &mut self.spectrum_mode,
                        SpectrumViewMode::CohortOverlay,
                        "1D Overlay",
                    );
                    ui.selectable_value(
                        &mut self.spectrum_mode,
                        SpectrumViewMode::StocsyCovarianceHeat,
                        "STOCSY Cov + r²",
                    );
                    ui.checkbox(&mut self.show_bucket_boundaries, "Buckets");
                    if ui.small_button("Full Sweep").clicked() {
                        if let Some(ref st) = self.study {
                            self.view_ppm_bounds = (st.config.min_ppm, st.config.max_ppm);
                        }
                    }
                });

                let canvas_h = (ui.available_height() - 4.0).max(120.0);
                let canvas_w = ui.available_width().max(180.0);
                let (rect, resp) = ui.allocate_exact_size(
                    egui::vec2(canvas_w, canvas_h),
                    egui::Sense::click_and_drag(),
                );
                let painter = ui.painter_at(rect);
                painter.rect_filled(rect, 3.0, egui::Color32::from_rgb(10, 14, 21));

                let Some(ref study) = self.study else {
                    return;
                };
                let plot_rect = egui::Rect::from_min_max(
                    egui::pos2(rect.left() + 8.0, rect.top() + 8.0),
                    egui::pos2(rect.right() - 8.0, rect.bottom() - 22.0),
                );
                if plot_rect.width() <= 16.0 || plot_rect.height() <= 16.0 {
                    return;
                }

                // Interactive horizontal pan & scroll-wheel zoom on descending NMR ppm axis
                let (mut v_lo, mut v_hi) = self.view_ppm_bounds;
                let span = (v_hi - v_lo).max(0.03);
                if resp.dragged() {
                    let dx = resp.drag_delta().x as f64;
                    let d_ppm = dx * span / (plot_rect.width() as f64);
                    v_lo += d_ppm;
                    v_hi += d_ppm;
                }
                if resp.hovered() {
                    let scroll_y = ui.input(|i| i.smooth_scroll_delta.y);
                    if scroll_y.abs() > 0.1 {
                        let factor = (1.0 - (scroll_y as f64) * 0.0025).clamp(0.65, 1.45);
                        let ptr_frac = ui
                            .input(|i| i.pointer.hover_pos())
                            .map(|p| {
                                ((p.x - plot_rect.left()) / plot_rect.width()).clamp(0.0, 1.0)
                                    as f64
                            })
                            .unwrap_or(0.5);
                        let anchor_ppm = v_hi - ptr_frac * span;
                        let new_span = (span * factor)
                            .clamp(0.04, study.config.max_ppm - study.config.min_ppm);
                        v_hi = anchor_ppm + ptr_frac * new_span;
                        v_lo = v_hi - new_span;
                    }
                }
                let full_lo = study.config.min_ppm;
                let full_hi = study.config.max_ppm;
                let cur_span = (v_hi - v_lo).clamp(0.04, (full_hi - full_lo).max(0.05));
                if v_lo < full_lo {
                    v_lo = full_lo;
                    v_hi = v_lo + cur_span;
                }
                if v_hi > full_hi {
                    v_hi = full_hi;
                    v_lo = (v_hi - cur_span).max(full_lo);
                }
                self.view_ppm_bounds = (v_lo, v_hi);

                // Descending NMR axis: high ppm on left -> low ppm on right
                let ppm_to_x = |ppm: f64| -> f32 {
                    let frac = ((v_hi - ppm) / (v_hi - v_lo).max(1e-9)) as f32;
                    plot_rect.left() + frac * plot_rect.width()
                };
                let x_to_ppm = |x: f32| -> f64 {
                    let frac = ((x - plot_rect.left()) / plot_rect.width()).clamp(0.0, 1.0) as f64;
                    v_hi - frac * (v_hi - v_lo)
                };

                // 1. Paint excluded solvent/water/urea zones
                for z in study.config.mask_preset.zones() {
                    let zl = z.low_ppm.max(v_lo);
                    let zh = z.high_ppm.min(v_hi);
                    if zh > zl {
                        let x0 = ppm_to_x(zh);
                        let x1 = ppm_to_x(zl);
                        painter.rect_filled(
                            egui::Rect::from_min_max(
                                egui::pos2(x0, plot_rect.top()),
                                egui::pos2(x1, plot_rect.bottom()),
                            ),
                            0.0,
                            egui::Color32::from_rgba_unmultiplied(90, 40, 45, 55),
                        );
                    }
                }

                // 2. Paint bucket boundary lines & active STOCSY driver band
                if self.show_bucket_boundaries {
                    for b in &study.buckets {
                        if b.high_ppm < v_lo || b.low_ppm > v_hi {
                            continue;
                        }
                        let xl = ppm_to_x(b.high_ppm);
                        let xr = ppm_to_x(b.low_ppm);
                        let is_sel = self.selected_bucket_idx == Some(b.index);
                        if is_sel {
                            painter.rect_filled(
                                egui::Rect::from_min_max(
                                    egui::pos2(xl, plot_rect.top()),
                                    egui::pos2(xr, plot_rect.bottom()),
                                ),
                                0.0,
                                egui::Color32::from_rgba_unmultiplied(250, 204, 21, 38),
                            );
                        }
                        painter.line_segment(
                            [
                                egui::pos2(xl, plot_rect.top()),
                                egui::pos2(xl, plot_rect.bottom()),
                            ],
                            egui::Stroke::new(
                                if is_sel { 1.4 } else { 0.6 },
                                if is_sel {
                                    egui::Color32::from_rgb(250, 204, 21)
                                } else {
                                    egui::Color32::from_rgba_unmultiplied(75, 95, 128, 85)
                                },
                            ),
                        );
                    }
                }

                // Determine visible grid range [i_start..=i_end]
                let m = study.common_ppm_grid.len();
                let i_start = study.common_ppm_grid.partition_point(|&p| p > v_hi);
                let i_end = study
                    .common_ppm_grid
                    .partition_point(|&p| p >= v_lo)
                    .saturating_sub(1)
                    .min(m.saturating_sub(1));

                if i_start < i_end && m > 1 {
                    let max_cols = (plot_rect.width().ceil() as usize).max(160);
                    let stride = ((i_end - i_start) / max_cols).max(1);

                    match self.spectrum_mode {
                        SpectrumViewMode::CohortOverlay => {
                            let mut unique_groups: Vec<String> = Vec::new();
                            for g in &study.sample_groups {
                                if !unique_groups.contains(g) {
                                    unique_groups.push(g.clone());
                                }
                            }
                            let mut y_max = 1e-12_f64;
                            let mut y_min = 0.0_f64;
                            for (si, row) in study.aligned_spectra.iter().enumerate() {
                                let df = study
                                    .dilution_factors
                                    .get(si)
                                    .copied()
                                    .unwrap_or(1.0)
                                    .max(1e-15);
                                for &v in row[i_start..=i_end].iter().step_by(stride) {
                                    let nv = v / df;
                                    y_max = y_max.max(nv);
                                    y_min = y_min.min(nv);
                                }
                            }
                            let y_span = (y_max - y_min).max(1e-12);
                            let val_to_y = |v: f64| -> f32 {
                                let frac = ((v - y_min) / y_span) as f32;
                                plot_rect.bottom() - frac * plot_rect.height()
                            };

                            for (si, row) in study.aligned_spectra.iter().enumerate() {
                                let g_idx = unique_groups
                                    .iter()
                                    .position(|g| g == &study.sample_groups[si])
                                    .unwrap_or(0);
                                let base_c = group_color(g_idx);
                                let col = egui::Color32::from_rgba_unmultiplied(
                                    base_c.r(),
                                    base_c.g(),
                                    base_c.b(),
                                    135,
                                );
                                let df = study
                                    .dilution_factors
                                    .get(si)
                                    .copied()
                                    .unwrap_or(1.0)
                                    .max(1e-15);
                                let mut pts = Vec::with_capacity((i_end - i_start) / stride + 2);
                                let mut k = i_start;
                                while k <= i_end {
                                    pts.push(egui::pos2(
                                        ppm_to_x(study.common_ppm_grid[k]),
                                        val_to_y(row[k] / df),
                                    ));
                                    k += stride;
                                }
                                painter.add(egui::Shape::line(pts, egui::Stroke::new(1.0, col)));
                            }

                            // STOCSY r^2 heat-ribbon along top of plot when driver is active
                            if let Some(ref st) = self.active_stocsy {
                                for b in &study.buckets {
                                    if b.high_ppm < v_lo || b.low_ppm > v_hi {
                                        continue;
                                    }
                                    let r2 = st.bucket_r2.get(b.index).copied().unwrap_or(0.0);
                                    if r2 >= 0.15 {
                                        let xl = ppm_to_x(b.high_ppm.min(v_hi));
                                        let xr = ppm_to_x(b.low_ppm.max(v_lo));
                                        painter.rect_filled(
                                            egui::Rect::from_min_max(
                                                egui::pos2(xl, plot_rect.top()),
                                                egui::pos2(xr.max(xl + 2.0), plot_rect.top() + 5.0),
                                            ),
                                            1.0,
                                            stocsy_r2_color(r2),
                                        );
                                    }
                                }
                            }
                        }
                        SpectrumViewMode::StocsyCovarianceHeat => {
                            if let Some(ref st) = self.active_stocsy {
                                if st.spectral_covariance.len() == m {
                                    let mut y_max = 1e-18_f64;
                                    let mut y_min = -1e-18_f64;
                                    for &v in st.spectral_covariance[i_start..=i_end]
                                        .iter()
                                        .step_by(stride)
                                    {
                                        y_max = y_max.max(v);
                                        y_min = y_min.min(v);
                                    }
                                    let y_span = (y_max - y_min).max(1e-18);
                                    let val_to_y = |v: f64| -> f32 {
                                        let frac = ((v - y_min) / y_span) as f32;
                                        plot_rect.bottom() - frac * plot_rect.height()
                                    };
                                    let y0 = val_to_y(0.0);
                                    painter.line_segment(
                                        [
                                            egui::pos2(plot_rect.left(), y0),
                                            egui::pos2(plot_rect.right(), y0),
                                        ],
                                        egui::Stroke::new(0.8, egui::Color32::from_rgb(60, 72, 92)),
                                    );

                                    let mut k = i_start;
                                    while k + stride <= i_end {
                                        let k2 = k + stride;
                                        let p0 = egui::pos2(
                                            ppm_to_x(study.common_ppm_grid[k]),
                                            val_to_y(st.spectral_covariance[k]),
                                        );
                                        let p1 = egui::pos2(
                                            ppm_to_x(study.common_ppm_grid[k2]),
                                            val_to_y(st.spectral_covariance[k2]),
                                        );
                                        let r2_seg = st.spectral_r2[k].max(st.spectral_r2[k2]);
                                        painter.line_segment(
                                            [p0, p1],
                                            egui::Stroke::new(1.5, stocsy_r2_color(r2_seg)),
                                        );
                                        k = k2;
                                    }
                                }
                            }
                        }
                    }
                }

                // 3. Draw chemical-shift axis ticks along bottom
                painter.line_segment(
                    [
                        egui::pos2(plot_rect.left(), plot_rect.bottom()),
                        egui::pos2(plot_rect.right(), plot_rect.bottom()),
                    ],
                    egui::Stroke::new(1.0, egui::Color32::from_rgb(82, 98, 126)),
                );
                let n_ticks = 6usize;
                for t in 0..=n_ticks {
                    let frac = (t as f64) / (n_ticks as f64);
                    let ppm = v_hi - frac * (v_hi - v_lo);
                    let x = ppm_to_x(ppm);
                    painter.line_segment(
                        [
                            egui::pos2(x, plot_rect.bottom()),
                            egui::pos2(x, plot_rect.bottom() + 4.0),
                        ],
                        egui::Stroke::new(1.0, egui::Color32::from_rgb(110, 128, 158)),
                    );
                    painter.text(
                        egui::pos2(x, plot_rect.bottom() + 5.0),
                        egui::Align2::CENTER_TOP,
                        format!("{ppm:.2}"),
                        egui::FontId::monospace(10.0),
                        egui::Color32::from_rgb(165, 182, 208),
                    );
                }

                // Click to select nearest bucket & compute live STOCSY
                if resp.clicked() {
                    if let Some(pos) = resp.interact_pointer_pos() {
                        let click_ppm = x_to_ppm(pos.x);
                        clicked_bucket = study
                            .buckets
                            .iter()
                            .min_by(|a, b| {
                                (a.center_ppm - click_ppm)
                                    .abs()
                                    .total_cmp(&(b.center_ppm - click_ppm).abs())
                            })
                            .map(|b| b.index);
                    }
                }
            });

        if let Some(bi) = clicked_bucket {
            self.select_bucket_and_run_stocsy(bi, false);
        }
    }

    // ========================================================================
    // Quadrant B: PCA / OPLS-DA Score Plot with 95% Hotelling T^2 Ellipses
    // ========================================================================

    fn draw_quadrant_b_scores(&mut self, ui: &mut egui::Ui, size: egui::Vec2) {
        egui::Frame::new()
            .fill(egui::Color32::from_rgb(14, 18, 26))
            .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(42, 54, 74)))
            .corner_radius(5.0)
            .inner_margin(egui::Margin::same(6))
            .show(ui, |ui| {
                ui.set_min_size(size - egui::vec2(12.0, 12.0));
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("B. Score Space & 95% Hotelling T²")
                            .strong()
                            .color(egui::Color32::from_rgb(135, 215, 255)),
                    );
                    ui.separator();
                    ui.selectable_value(&mut self.score_mode, ScorePlotMode::OplsDa, "OPLS-DA");
                    ui.selectable_value(&mut self.score_mode, ScorePlotMode::Pca, "PCA");
                    ui.selectable_value(&mut self.score_mode, ScorePlotMode::PlsDa, "PLS-DA");
                });

                let canvas_h = (ui.available_height() - 4.0).max(120.0);
                let canvas_w = ui.available_width().max(180.0);
                let (rect, resp) = ui.allocate_exact_size(
                    egui::vec2(canvas_w, canvas_h),
                    egui::Sense::hover(),
                );
                let painter = ui.painter_at(rect);
                painter.rect_filled(rect, 3.0, egui::Color32::from_rgb(10, 14, 21));

                let Some(ref study) = self.study else {
                    return;
                };
                let plot_rect = egui::Rect::from_min_max(
                    egui::pos2(rect.left() + 34.0, rect.top() + 10.0),
                    egui::pos2(rect.right() - 12.0, rect.bottom() - 22.0),
                );
                if plot_rect.width() <= 20.0 || plot_rect.height() <= 20.0 {
                    return;
                }

                // Gather (x, y, sample_idx, is_outlier) points + ellipses + axis labels
                struct ScorePoint {
                    x: f64,
                    y: f64,
                    sample_idx: usize,
                    outlier: bool,
                }
                let mut pts: Vec<ScorePoint> = Vec::new();
                let mut ellipses: Vec<(ConfidenceEllipse2D, egui::Color32, bool)> = Vec::new();

                let mut unique_groups: Vec<String> = Vec::new();
                for g in &study.sample_groups {
                    if !unique_groups.contains(g) {
                        unique_groups.push(g.clone());
                    }
                }

                let (x_lbl, y_lbl) = match self.score_mode {
                    ScorePlotMode::OplsDa if study.opls_da.is_some() => {
                        let op = study.opls_da.as_ref().unwrap();
                        let to1 = op.orthogonal_scores.first();
                        for (k, &si) in op.active_sample_indices.iter().enumerate() {
                            let sx = op.predictive_scores[k];
                            let sy = to1.map(|v| v[k]).unwrap_or(0.0);
                            let out = study.pca.hotelling_t2[si] > study.pca.hotelling_t2_limit_95
                                || study.pca.q_residuals[si] > study.pca.q_residual_limit_95;
                            pts.push(ScorePoint {
                                x: sx,
                                y: sy,
                                sample_idx: si,
                                outlier: out,
                            });
                        }
                        ellipses.push((
                            op.global_ellipse_95.clone(),
                            egui::Color32::from_rgba_unmultiplied(148, 163, 184, 110),
                            true,
                        ));
                        for ge in &op.group_ellipses_95 {
                            let gi = unique_groups
                                .iter()
                                .position(|g| g == &ge.label)
                                .unwrap_or(0);
                            ellipses.push((ge.clone(), group_color(gi), false));
                        }
                        (
                            format!("tp ({:.1}% R²Xp)", op.r2_x_predictive * 100.0),
                            format!("to1 ({:.1}% R²Xo)", op.r2_x_orthogonal * 100.0),
                        )
                    }
                    ScorePlotMode::PlsDa if study.pls_da.is_some() => {
                        let pl = study.pls_da.as_ref().unwrap();
                        for (k, &si) in pl.active_sample_indices.iter().enumerate() {
                            let sx = pl.scores[k].first().copied().unwrap_or(0.0);
                            let sy = pl.scores[k].get(1).copied().unwrap_or(0.0);
                            let out = study.pca.hotelling_t2[si] > study.pca.hotelling_t2_limit_95;
                            pts.push(ScorePoint {
                                x: sx,
                                y: sy,
                                sample_idx: si,
                                outlier: out,
                            });
                        }
                        (
                            format!("PLS LV1 (R²Y={:.2})", pl.r2_y),
                            format!("PLS LV2 (Q²={:.2})", pl.q2),
                        )
                    }
                    _ => {
                        for (si, sc) in study.pca.scores.iter().enumerate() {
                            let sx = sc.first().copied().unwrap_or(0.0);
                            let sy = sc.get(1).copied().unwrap_or(0.0);
                            let out = study.pca.hotelling_t2[si] > study.pca.hotelling_t2_limit_95
                                || study.pca.q_residuals[si] > study.pca.q_residual_limit_95;
                            pts.push(ScorePoint {
                                x: sx,
                                y: sy,
                                sample_idx: si,
                                outlier: out,
                            });
                        }
                        ellipses.push((
                            study.pca.global_ellipse_95.clone(),
                            egui::Color32::from_rgba_unmultiplied(148, 163, 184, 110),
                            true,
                        ));
                        for ge in &study.pca.group_ellipses_95 {
                            let gi = unique_groups
                                .iter()
                                .position(|g| g == &ge.label)
                                .unwrap_or(0);
                            ellipses.push((ge.clone(), group_color(gi), false));
                        }
                        let r1 = study
                            .pca
                            .explained_variance_ratio
                            .first()
                            .copied()
                            .unwrap_or(0.0)
                            * 100.0;
                        let r2 = study
                            .pca
                            .explained_variance_ratio
                            .get(1)
                            .copied()
                            .unwrap_or(0.0)
                            * 100.0;
                        (format!("PC1 ({r1:.1}%)"), format!("PC2 ({r2:.1}%)"))
                    }
                };

                let mut x_min = -1.0_f64;
                let mut x_max = 1.0_f64;
                let mut y_min = -1.0_f64;
                let mut y_max = 1.0_f64;
                if !pts.is_empty() {
                    x_min = pts.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
                    x_max = pts.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max);
                    y_min = pts.iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
                    y_max = pts.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max);
                }
                for (ell, _, _) in &ellipses {
                    for v in ell.sample_points(36) {
                        x_min = x_min.min(v[0]);
                        x_max = x_max.max(v[0]);
                        y_min = y_min.min(v[1]);
                        y_max = y_max.max(v[1]);
                    }
                }
                let pad_x = ((x_max - x_min).abs() * 0.12).max(0.25);
                let pad_y = ((y_max - y_min).abs() * 0.12).max(0.25);
                x_min -= pad_x;
                x_max += pad_x;
                y_min -= pad_y;
                y_max += pad_y;

                let to_screen = |x: f64, y: f64| -> egui::Pos2 {
                    let fx = ((x - x_min) / (x_max - x_min).max(1e-12)) as f32;
                    let fy = ((y - y_min) / (y_max - y_min).max(1e-12)) as f32;
                    egui::pos2(
                        plot_rect.left() + fx * plot_rect.width(),
                        plot_rect.bottom() - fy * plot_rect.height(),
                    )
                };

                // Zero crosshair
                let zero = to_screen(0.0, 0.0);
                if zero.x >= plot_rect.left() && zero.x <= plot_rect.right() {
                    painter.line_segment(
                        [
                            egui::pos2(zero.x, plot_rect.top()),
                            egui::pos2(zero.x, plot_rect.bottom()),
                        ],
                        egui::Stroke::new(0.8, egui::Color32::from_rgb(48, 60, 82)),
                    );
                }
                if zero.y >= plot_rect.top() && zero.y <= plot_rect.bottom() {
                    painter.line_segment(
                        [
                            egui::pos2(plot_rect.left(), zero.y),
                            egui::pos2(plot_rect.right(), zero.y),
                        ],
                        egui::Stroke::new(0.8, egui::Color32::from_rgb(48, 60, 82)),
                    );
                }

                // Draw 95% Hotelling T^2 ellipses
                for (ell, col, is_global) in &ellipses {
                    let poly: Vec<egui::Pos2> = ell
                        .sample_points(64)
                        .into_iter()
                        .map(|p| to_screen(p[0], p[1]))
                        .collect();
                    let stroke_c = if *is_global {
                        *col
                    } else {
                        egui::Color32::from_rgba_unmultiplied(col.r(), col.g(), col.b(), 155)
                    };
                    painter.add(egui::Shape::line(
                        poly,
                        egui::Stroke::new(if *is_global { 1.1 } else { 1.4 }, stroke_c),
                    ));
                }

                // Draw sample markers & detect hover
                let ptr = resp.hover_pos();
                let mut hovered_sample: Option<&ScorePoint> = None;
                for sp in &pts {
                    let pos = to_screen(sp.x, sp.y);
                    let gi = unique_groups
                        .iter()
                        .position(|g| g == &study.sample_groups[sp.sample_idx])
                        .unwrap_or(0);
                    let c = group_color(gi);
                    if sp.outlier {
                        painter.circle_stroke(
                            pos,
                            7.2,
                            egui::Stroke::new(1.6, egui::Color32::from_rgb(250, 204, 21)),
                        );
                    }
                    painter.circle_filled(pos, 4.6, c);
                    if let Some(hp) = ptr {
                        if hp.distance(pos) <= 8.0 {
                            hovered_sample = Some(sp);
                        }
                    }
                }

                // Legend in top-right corner
                for (gi, g_name) in unique_groups.iter().enumerate() {
                    let ly = plot_rect.top() + 8.0 + (gi as f32) * 14.0;
                    let lx = plot_rect.right() - 8.0;
                    painter.text(
                        egui::pos2(lx, ly),
                        egui::Align2::RIGHT_CENTER,
                        format!("● {g_name}"),
                        egui::FontId::proportional(11.0),
                        group_color(gi),
                    );
                }

                // Axis captions
                painter.text(
                    egui::pos2(plot_rect.center().x, rect.bottom() - 4.0),
                    egui::Align2::CENTER_BOTTOM,
                    x_lbl,
                    egui::FontId::monospace(10.5),
                    egui::Color32::from_rgb(165, 182, 208),
                );
                painter.text(
                    egui::pos2(rect.left() + 4.0, plot_rect.top() + 2.0),
                    egui::Align2::LEFT_TOP,
                    y_lbl,
                    egui::FontId::monospace(10.0),
                    egui::Color32::from_rgb(165, 182, 208),
                );

                if let Some(sp) = hovered_sample {
                    let si = sp.sample_idx;
                    resp.on_hover_text(format!(
                        "{} ({})\nScore: ({:.3}, {:.3})\nHotelling T²: {:.2} (95% lim {:.2})\nQ-residual (SPE): {:.3} (95% lim {:.3})",
                        study.sample_ids[si],
                        study.sample_groups[si],
                        sp.x,
                        sp.y,
                        study.pca.hotelling_t2[si],
                        study.pca.hotelling_t2_limit_95,
                        study.pca.q_residuals[si],
                        study.pca.q_residual_limit_95
                    ));
                }
            });
    }

    // ========================================================================
    // Quadrant C: Clickable OPLS-DA S-Plot / BH-FDR Volcano Plot
    // ========================================================================

    fn draw_quadrant_c_splot_volcano(&mut self, ui: &mut egui::Ui, size: egui::Vec2) {
        let mut clicked_bucket: Option<usize> = None;

        egui::Frame::new()
            .fill(egui::Color32::from_rgb(14, 18, 26))
            .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(42, 54, 74)))
            .corner_radius(5.0)
            .inner_margin(egui::Margin::same(6))
            .show(ui, |ui| {
                ui.set_min_size(size - egui::vec2(12.0, 12.0));
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("C. Biomarker S-Plot / Volcano")
                            .strong()
                            .color(egui::Color32::from_rgb(135, 215, 255)),
                    );
                    ui.separator();
                    ui.selectable_value(
                        &mut self.feature_mode,
                        FeaturePlotMode::OplsSPlot,
                        "OPLS-DA S-Plot",
                    );
                    ui.selectable_value(
                        &mut self.feature_mode,
                        FeaturePlotMode::VolcanoPlot,
                        "BH-FDR Volcano",
                    );
                    ui.checkbox(&mut self.auto_zoom_on_select, "Auto-Zoom 1D");
                });

                let canvas_h = (ui.available_height() - 4.0).max(120.0);
                let canvas_w = ui.available_width().max(180.0);
                let (rect, resp) = ui.allocate_exact_size(
                    egui::vec2(canvas_w, canvas_h),
                    egui::Sense::click(),
                );
                let painter = ui.painter_at(rect);
                painter.rect_filled(rect, 3.0, egui::Color32::from_rgb(10, 14, 21));

                let Some(ref study) = self.study else {
                    return;
                };
                if study.biomarkers.is_empty() {
                    return;
                }
                let plot_rect = egui::Rect::from_min_max(
                    egui::pos2(rect.left() + 34.0, rect.top() + 10.0),
                    egui::pos2(rect.right() - 12.0, rect.bottom() - 22.0),
                );
                if plot_rect.width() <= 20.0 || plot_rect.height() <= 20.0 {
                    return;
                }

                let use_splot =
                    self.feature_mode == FeaturePlotMode::OplsSPlot && study.opls_da.is_some();

                let mut coords = Vec::with_capacity(study.biomarkers.len());
                for bm in &study.biomarkers {
                    let (vx, vy) = if use_splot {
                        (bm.s_plot_cov, bm.s_plot_corr)
                    } else {
                        let nlogq = (-bm.fdr_q.max(1e-30).log10()).clamp(0.0, 25.0);
                        (bm.log2_fold_change, nlogq)
                    };
                    coords.push((bm.bucket_index, vx, vy, bm));
                }

                let mut x_min = coords.iter().map(|c| c.1).fold(0.0_f64, f64::min);
                let mut x_max = coords.iter().map(|c| c.1).fold(0.0_f64, f64::max);
                let x_abs = x_min.abs().max(x_max.abs()).max(0.15) * 1.12;
                x_min = -x_abs;
                x_max = x_abs;

                let (y_min, y_max, x_lbl, y_lbl) = if use_splot {
                    (-1.08, 1.08, "p(cov) = Cov(tp, Xj)", "p(corr)")
                } else {
                    let ym = coords
                        .iter()
                        .map(|c| c.2)
                        .fold(2.0_f64, f64::max)
                        * 1.12;
                    (-0.08 * ym, ym, "log2 Fold Change (Case / Control)", "-log10(FDR q)")
                };

                let to_screen = |x: f64, y: f64| -> egui::Pos2 {
                    let fx = ((x - x_min) / (x_max - x_min).max(1e-12)) as f32;
                    let fy = ((y - y_min) / (y_max - y_min).max(1e-12)) as f32;
                    egui::pos2(
                        plot_rect.left() + fx * plot_rect.width(),
                        plot_rect.bottom() - fy * plot_rect.height(),
                    )
                };

                // Threshold guide lines
                let x0 = to_screen(0.0, 0.0).x;
                painter.line_segment(
                    [
                        egui::pos2(x0, plot_rect.top()),
                        egui::pos2(x0, plot_rect.bottom()),
                    ],
                    egui::Stroke::new(0.8, egui::Color32::from_rgb(50, 62, 84)),
                );
                if use_splot {
                    for thr in [-0.70, 0.0, 0.70] {
                        let y_line = to_screen(0.0, thr).y;
                        painter.line_segment(
                            [
                                egui::pos2(plot_rect.left(), y_line),
                                egui::pos2(plot_rect.right(), y_line),
                            ],
                            egui::Stroke::new(
                                0.8,
                                if thr == 0.0 {
                                    egui::Color32::from_rgb(50, 62, 84)
                                } else {
                                    egui::Color32::from_rgba_unmultiplied(234, 179, 8, 95)
                                },
                            ),
                        );
                    }
                } else {
                    // FDR q = 0.05 => -log10(0.05) = 1.30103
                    let y_fdr = to_screen(0.0, 1.301_03).y;
                    painter.line_segment(
                        [
                            egui::pos2(plot_rect.left(), y_fdr),
                            egui::pos2(plot_rect.right(), y_fdr),
                        ],
                        egui::Stroke::new(
                            0.9,
                            egui::Color32::from_rgba_unmultiplied(234, 179, 8, 110),
                        ),
                    );
                }

                let ptr = resp.hover_pos();
                let mut hovered_bm = None;
                let mut nearest_click: Option<(usize, f32)> = None;
                let click_pos = resp.clicked().then(|| resp.interact_pointer_pos()).flatten();

                for &(b_idx, vx, vy, bm) in &coords {
                    let pos = to_screen(vx, vy);
                    let is_sel = self.selected_bucket_idx == Some(b_idx);
                    let col = if let Some(ref st) = self.active_stocsy {
                        let r2 = st.bucket_r2.get(b_idx).copied().unwrap_or(0.0);
                        stocsy_r2_color(r2)
                    } else if bm.fdr_q <= 0.05 && bm.vip >= 1.0 {
                        egui::Color32::from_rgb(248, 113, 113)
                    } else {
                        egui::Color32::from_rgb(100, 130, 170)
                    };

                    if is_sel {
                        painter.circle_stroke(
                            pos,
                            7.5,
                            egui::Stroke::new(2.0, egui::Color32::WHITE),
                        );
                    }
                    painter.circle_filled(pos, if is_sel { 5.0 } else { 3.8 }, col);

                    if let Some(hp) = ptr {
                        if hp.distance(pos) <= 7.5 {
                            hovered_bm = Some(bm);
                        }
                    }
                    if let Some(cp) = click_pos {
                        let dist = cp.distance(pos);
                        if dist <= 14.0
                            && nearest_click.map(|(_, d)| dist < d).unwrap_or(true)
                        {
                            nearest_click = Some((b_idx, dist));
                        }
                    }
                }

                if let Some((bi, _)) = nearest_click {
                    clicked_bucket = Some(bi);
                }

                painter.text(
                    egui::pos2(plot_rect.center().x, rect.bottom() - 4.0),
                    egui::Align2::CENTER_BOTTOM,
                    x_lbl,
                    egui::FontId::monospace(10.5),
                    egui::Color32::from_rgb(165, 182, 208),
                );
                painter.text(
                    egui::pos2(rect.left() + 4.0, plot_rect.top() + 2.0),
                    egui::Align2::LEFT_TOP,
                    y_lbl,
                    egui::FontId::monospace(10.0),
                    egui::Color32::from_rgb(165, 182, 208),
                );

                if let Some(bm) = hovered_bm {
                    let met = bm
                        .putative_metabolite
                        .as_deref()
                        .unwrap_or("Unassigned bucket");
                    resp.on_hover_text(format!(
                        "Bucket B{:03} @ {:.3} ppm [{:.3}..{:.3}]\nPutative BMRB: {}\nVIP: {:.2} | p(corr): {:+.3}\nlog2 FC: {:+.2} | BH-FDR q: {:.2e}\n(Click to jump 1D & run live STOCSY)",
                        bm.bucket_index,
                        bm.center_ppm,
                        bm.low_ppm,
                        bm.high_ppm,
                        met,
                        bm.vip,
                        bm.s_plot_corr,
                        bm.log2_fold_change,
                        bm.fdr_q
                    ));
                }
            });

        if let Some(bi) = clicked_bucket {
            let zoom = self.auto_zoom_on_select;
            self.select_bucket_and_run_stocsy(bi, zoom);
        }
    }

    // ========================================================================
    // Quadrant D: Ranked Candidate Biomarker & BMRB Metabolite Table
    // ========================================================================

    fn draw_quadrant_d_biomarker_table(&mut self, ui: &mut egui::Ui, size: egui::Vec2) {
        let mut clicked_bucket: Option<usize> = None;

        egui::Frame::new()
            .fill(egui::Color32::from_rgb(14, 18, 26))
            .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(42, 54, 74)))
            .corner_radius(5.0)
            .inner_margin(egui::Margin::same(6))
            .show(ui, |ui| {
                ui.set_min_size(size - egui::vec2(12.0, 12.0));
                ui.label(
                    egui::RichText::new("D. Ranked Candidate Biomarkers & STOCSY BMRB Matches")
                        .strong()
                        .color(egui::Color32::from_rgb(135, 215, 255)),
                );

                // Active STOCSY spin-system match summary card
                if let Some(ref st) = self.active_stocsy {
                    egui::Frame::new()
                        .fill(egui::Color32::from_rgb(20, 28, 42))
                        .corner_radius(4.0)
                        .inner_margin(egui::Margin::symmetric(6, 4))
                        .show(ui, |ui| {
                            if let Some(top) = st.metabolite_matches.first() {
                                let peaks_str = top
                                    .matched_peaks
                                    .iter()
                                    .map(|(r_ppm, _, r2)| format!("{r_ppm:.2} (r²={r2:.2})"))
                                    .collect::<Vec<_>>()
                                    .join(", ");
                                ui.label(
                                    egui::RichText::new(format!(
                                        "Driver {:.3} ppm -> {} [{}] | Score {:.2} ({}/{} multiplets, stoich cos {:.2}) | {}",
                                        st.driver_ppm,
                                        top.name,
                                        top.bmrb_id,
                                        top.score,
                                        top.matched_multiplets,
                                        top.observable_multiplets,
                                        top.stoichiometry_cosine,
                                        peaks_str
                                    ))
                                    .small()
                                    .monospace()
                                    .color(egui::Color32::from_rgb(190, 242, 214)),
                                );
                            } else {
                                ui.label(
                                    egui::RichText::new(format!(
                                        "Driver {:.3} ppm — no BMRB library resonance within chemical-shift tolerance",
                                        st.driver_ppm
                                    ))
                                    .small()
                                    .monospace()
                                    .color(egui::Color32::from_rgb(165, 180, 204)),
                                );
                            }
                        });
                }

                ui.add_space(3.0);

                let Some(ref study) = self.study else {
                    return;
                };

                egui::ScrollArea::vertical()
                    .id_salt("metabolomics_biomarker_table_scroll")
                    .max_height((ui.available_height() - 2.0).max(80.0))
                    .show(ui, |ui| {
                        egui::Grid::new("metabolomics_biomarker_grid")
                            .striped(true)
                            .spacing([8.0, 3.0])
                            .show(ui, |ui| {
                                for hdr in [
                                    "#",
                                    "δ (ppm)",
                                    "VIP",
                                    "p(corr)",
                                    "log2 FC",
                                    "FDR q",
                                    "STOCSY r²",
                                    "Putative BMRB Metabolite",
                                ] {
                                    ui.label(
                                        egui::RichText::new(hdr)
                                            .small()
                                            .strong()
                                            .color(egui::Color32::from_rgb(148, 184, 226)),
                                    );
                                }
                                ui.end_row();

                                for bm in study.biomarkers.iter().take(60) {
                                    let is_sel = self.selected_bucket_idx == Some(bm.bucket_index);
                                    let r2 = self
                                        .active_stocsy
                                        .as_ref()
                                        .and_then(|s| s.bucket_r2.get(bm.bucket_index).copied())
                                        .unwrap_or(0.0);

                                    let rank_btn = ui.selectable_label(
                                        is_sel,
                                        egui::RichText::new(format!("{:>2}", bm.rank))
                                            .small()
                                            .monospace(),
                                    );
                                    let ppm_btn = ui.selectable_label(
                                        is_sel,
                                        egui::RichText::new(format!("{:.3}", bm.center_ppm))
                                            .small()
                                            .monospace()
                                            .strong(),
                                    );
                                    if rank_btn.clicked() || ppm_btn.clicked() {
                                        clicked_bucket = Some(bm.bucket_index);
                                    }

                                    ui.label(
                                        egui::RichText::new(format!("{:.2}", bm.vip))
                                            .small()
                                            .monospace()
                                            .color(if bm.vip >= 1.0 {
                                                egui::Color32::from_rgb(134, 239, 172)
                                            } else {
                                                egui::Color32::from_rgb(160, 174, 192)
                                            }),
                                    );
                                    ui.label(
                                        egui::RichText::new(format!("{:+.2}", bm.s_plot_corr))
                                            .small()
                                            .monospace(),
                                    );
                                    ui.label(
                                        egui::RichText::new(format!("{:+.2}", bm.log2_fold_change))
                                            .small()
                                            .monospace()
                                            .color(if bm.log2_fold_change > 0.3 {
                                                egui::Color32::from_rgb(251, 146, 60)
                                            } else if bm.log2_fold_change < -0.3 {
                                                egui::Color32::from_rgb(96, 165, 250)
                                            } else {
                                                egui::Color32::from_rgb(180, 192, 210)
                                            }),
                                    );
                                    ui.label(
                                        egui::RichText::new(format!("{:.1e}", bm.fdr_q))
                                            .small()
                                            .monospace()
                                            .color(if bm.fdr_q <= 0.05 {
                                                egui::Color32::from_rgb(167, 243, 208)
                                            } else {
                                                egui::Color32::from_rgb(148, 163, 184)
                                            }),
                                    );
                                    ui.label(
                                        egui::RichText::new(format!("{:.2}", r2))
                                            .small()
                                            .monospace()
                                            .color(stocsy_r2_color(r2)),
                                    );

                                    let met_txt = match (&bm.putative_metabolite, &bm.putative_bmrb_id)
                                    {
                                        (Some(m), Some(id)) => format!("{m} ({id})"),
                                        (Some(m), None) => m.clone(),
                                        _ => "—".to_string(),
                                    };
                                    if ui
                                        .selectable_label(
                                            is_sel,
                                            egui::RichText::new(met_txt).small(),
                                        )
                                        .clicked()
                                    {
                                        clicked_bucket = Some(bm.bucket_index);
                                    }
                                    ui.end_row();
                                }
                            });
                    });
            });

        if let Some(bi) = clicked_bucket {
            let zoom = self.auto_zoom_on_select;
            self.select_bucket_and_run_stocsy(bi, zoom);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metabolomics_studio_headless_egui_render_and_live_stocsy_jump() {
        let ctx = egui::Context::default();
        let mut studio = MetabolomicsStudioPanel::default();
        studio.config.n_permutations = 20;

        // Frame 1: empty state render
        let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
            studio.ui(ui);
        });

        // Load 24-spectrum demo cohort and render OPLS-DA + S-plot + Cohort Overlay
        studio.load_demo_cohort();
        assert!(studio.study.is_some());
        assert!(studio.active_stocsy.is_some());

        let raw_in = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1440.0, 900.0),
            )),
            ..Default::default()
        };
        let _ = ctx.run_ui(raw_in.clone(), |ui| {
            studio.ui(ui);
        });

        // Jump to Lactate 1.33 ppm bucket, switch to STOCSY heat + PCA + Volcano, and render Frame 3
        let lac_idx = studio
            .study
            .as_ref()
            .unwrap()
            .buckets
            .iter()
            .min_by(|a, b| {
                (a.center_ppm - 1.330)
                    .abs()
                    .total_cmp(&(b.center_ppm - 1.330).abs())
            })
            .unwrap()
            .index;
        studio.select_bucket_and_run_stocsy(lac_idx, true);
        assert_eq!(studio.selected_bucket_idx, Some(lac_idx));
        assert!(
            studio.view_ppm_bounds.1 - studio.view_ppm_bounds.0 < 2.0,
            "Auto-zoom should focus Quadrant A around selected bucket"
        );
        assert_eq!(
            studio
                .active_stocsy
                .as_ref()
                .unwrap()
                .metabolite_matches
                .first()
                .map(|m| m.name.as_str()),
            Some("L-Lactate")
        );

        studio.spectrum_mode = SpectrumViewMode::StocsyCovarianceHeat;
        studio.score_mode = ScorePlotMode::Pca;
        studio.feature_mode = FeaturePlotMode::VolcanoPlot;
        let _ = ctx.run_ui(raw_in, |ui| {
            studio.ui(ui);
        });
    }
}
