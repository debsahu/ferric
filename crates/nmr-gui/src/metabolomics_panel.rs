//! Interactive 4-Quadrant 1D NMR Metabolomics, `PCA` / `OPLS-DA` Chemometrics & `STOCSY`
//! Biomarker Discovery Studio (`T14.5`, native desktop + `wasm32-unknown-unknown`).
//!
//! Provenance: Clean-room original human-computer interaction and plotting implementation of
//! `docs/specs/20-metabolomics-chemometrics.md` §9.2 built strictly on `eframe::egui` and
//! `nmr_core::metabolomics` (Savorani et al. *J. Magn. Reson.* 2010; De Meyer et al. *Anal. Chem.*
//! 2008; Dieterle et al. *Anal. Chem.* 2006; Trygg & Wold *J. Chemom.* 2002; Bylesjö et al.
//! *J. Chemom.* 2006; Westerhuis et al. *Metabolomics* 2008; Wiklund et al. *Anal. Chem.* 2008;
//! Cloarec et al. *Anal. Chem.* 2005; `BMRB-BMSE` metabolomics reference library: free access,
//! © UConn Health, acknowledgement requested).

use eframe::egui;
use nmr_core::metabolite_db::{
    extract_stocsy_cluster_query_peaks, npmrd_web_search_url, query_offline_metabolites,
    synthesize_reference_spectrum_1d, MetaboliteEntry, MetaboliteMatch, MetaboliteNucleus,
    QueryPeak, ReferenceSpectrum1D, DEFAULT_STOCSY_CLUSTER_MIN_R,
};
#[cfg(not(target_arch = "wasm32"))]
use nmr_core::metabolite_db::{query_online_bmrb_and_npmrd, OnlineQuerySources};
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

/// Completed background result from an opt-in desktop `BMRB v2` online query.
#[cfg(not(target_arch = "wasm32"))]
struct OnlineIdentifyOutcome {
    generation: u64,
    origin_label: String,
    n_peaks: usize,
    tol_ppm: f64,
    offline_count: usize,
    result: Result<Vec<MetaboliteMatch>, String>,
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
    pub show_biomarker_table: bool,
    /// Toggle for the `Identify Metabolite (BMRB-BMSE)` drawer below the main quadrants.
    pub show_identify_panel: bool,
    /// Query nucleus for metabolite identification (`1H` by default).
    pub identify_nucleus: MetaboliteNucleus,
    /// Chemical-shift matching tolerance in ppm for metabolite identification.
    pub identify_tol_ppm: f64,
    /// Opt-in desktop online search flag (`BMRB v2` REST API); strictly `false` by default.
    pub include_online_search: bool,
    /// Ranked candidate hits from the latest offline (and opt-in online) metabolite search.
    pub identify_results: Vec<MetaboliteMatch>,
    /// Status message for the metabolite identification panel.
    pub identify_status: String,
    /// Non-blocking channel receiver for background desktop `BMRB v2` online queries.
    #[cfg(not(target_arch = "wasm32"))]
    online_query_rx: Option<std::sync::mpsc::Receiver<OnlineIdentifyOutcome>>,
    /// True while a background desktop `BMRB v2` online query thread is in flight.
    #[cfg(not(target_arch = "wasm32"))]
    pub online_query_in_flight: bool,
    /// Monotonic generation counter to discard stale background online responses.
    #[cfg(not(target_arch = "wasm32"))]
    online_query_generation: u64,
    /// Synthesized 1D reference spectrum currently overlaid onto Quadrant A.
    pub active_reference_overlay: Option<ReferenceSpectrum1D>,
    /// Pending `SMILES` string queued by the 1-click `Send to Octet` action.
    pub pending_send_smiles_to_octet: Option<String>,
    /// True when the currently loaded cohort came from the built-in synthetic biofluid generator.
    pub is_synthetic_demo_cohort: bool,

    /// Visible horizontal chemical-shift window `(low_ppm, high_ppm)` in Quadrant A.
    pub view_ppm_bounds: (f64, f64),
    pub status_banner: String,
    pub demo_seed: u64,
    /// Last rendered inner plot rectangles for Quadrants A, B, and C.
    pub last_plot_rects: [Option<egui::Rect>; 3],
    /// Union visualization bounding rectangle across the active metabolomics plots (`UI-B6`).
    pub last_canvas_rect: Option<egui::Rect>,
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
            show_biomarker_table: false,
            show_identify_panel: false,
            identify_nucleus: MetaboliteNucleus::H1,
            identify_tol_ppm: 0.03,
            include_online_search: false,
            identify_results: Vec::new(),
            identify_status:
                "Click 'Query Picked 1D Peaks' or 'Query STOCSY (r >= 0.85)' to search offline BMRB-BMSE index."
                    .to_string(),
            #[cfg(not(target_arch = "wasm32"))]
            online_query_rx: None,
            #[cfg(not(target_arch = "wasm32"))]
            online_query_in_flight: false,
            #[cfg(not(target_arch = "wasm32"))]
            online_query_generation: 0,
            active_reference_overlay: None,
            pending_send_smiles_to_octet: None,
            is_synthetic_demo_cohort: false,
            view_ppm_bounds: view_bounds,
            status_banner:
                "No cohort loaded -- click 'Load 24-Sample Biofluid Demo' to run untargeted 1D metabolomics."
                    .to_string(),
            demo_seed: 20261115,
            last_plot_rects: [None; 3],
            last_canvas_rect: None,
        }
    }

    /// Load an external cohort of 1D samples and execute the full metabolomics + `OPLS-DA` + `STOCSY` pipeline.
    pub fn set_cohort_and_run(&mut self, samples: Vec<CohortSample1D>) {
        self.is_synthetic_demo_cohort = false;
        self.samples = samples;
        self.recompute_pipeline();
    }

    /// Generate the built-in 24-spectrum synthetic Case vs Control human biofluid cohort and solve.
    pub fn load_demo_cohort(&mut self) {
        let cohort = generate_demo_biofluid_cohort(self.demo_seed);
        self.view_ppm_bounds = (self.config.min_ppm, self.config.max_ppm);
        self.is_synthetic_demo_cohort = true;
        self.samples = cohort;
        self.recompute_pipeline();
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
        let demo_prefix = if self.is_synthetic_demo_cohort {
            "[Synthetic Demo Cohort] "
        } else {
            ""
        };
        let summary_txt = if let Some(ref op) = res.opls_da {
            let caveat = if op.permutation.is_validated {
                ""
            } else {
                " [!] Overfit warning"
            };
            format!(
                "{demo_prefix}Analyzed {n_s} spectra -> {n_b} buckets | CV Q² = {:.2} (7-fold), R²Y = {:.2}, permutation p <= {:.3} (n={}){caveat}",
                op.q2,
                op.r2_y,
                op.permutation.p_value_q2,
                op.permutation.n_permutations,
            )
        } else {
            format!("{demo_prefix}Analyzed {n_s} spectra -> {n_b} buckets (PCA only)")
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
            let demo_prefix = if self.is_synthetic_demo_cohort {
                "[Synthetic Demo Cohort] "
            } else {
                ""
            };
            self.status_banner = format!(
                "{demo_prefix}STOCSY driver B{:03} @ {:.3} ppm -> top match: {}",
                bucket_idx, st.driver_ppm, top_hit
            );
        }
    }

    /// Find the cohort-mean peak apex ppm inside a bucket window, falling back to `center_ppm`.
    fn bucket_apex_ppm(
        study: &MetabolomicsStudyResult,
        low_ppm: f64,
        high_ppm: f64,
        fallback: f64,
    ) -> f64 {
        let m = study.common_ppm_grid.len();
        if m == 0 || study.aligned_spectra.is_empty() {
            return fallback;
        }
        let i_start = study.common_ppm_grid.partition_point(|&p| p > high_ppm);
        let i_end = study
            .common_ppm_grid
            .partition_point(|&p| p >= low_ppm)
            .saturating_sub(1)
            .min(m.saturating_sub(1));
        if i_start > i_end || i_end >= m {
            return fallback;
        }
        let mut best_ppm = fallback;
        let mut best_val = f64::NEG_INFINITY;
        for idx in i_start..=i_end {
            let mut sum = 0.0_f64;
            for (si, row) in study.aligned_spectra.iter().enumerate() {
                let df = study
                    .dilution_factors
                    .get(si)
                    .copied()
                    .unwrap_or(1.0)
                    .max(1e-15);
                if let Some(&v) = row.get(idx) {
                    sum += v / df;
                }
            }
            if sum > best_val {
                best_val = sum;
                best_ppm = study.common_ppm_grid[idx];
            }
        }
        best_ppm
    }

    /// Extract genuine 1D local-maxima peaks above robust MAD-estimated SNR threshold from the
    /// dilution-normalized cohort-mean spectrum, coalescing multiplet fine-structure lines into
    /// their intensity-weighted centroid ppm (never sending raw bucket apices).
    pub fn extract_picked_1d_query_peaks(&self) -> Vec<QueryPeak> {
        let Some(ref study) = self.study else {
            return Vec::new();
        };
        let m = study.common_ppm_grid.len();
        let n_samples = study.aligned_spectra.len();
        if m < 3 || n_samples == 0 {
            return Vec::new();
        }

        // 1. Dilution-normalized cohort mean spectrum (masked regions zeroed).
        let mut mean_spec = vec![0.0_f64; m];
        let mut unmasked_vals = Vec::with_capacity(m);
        for (idx, mean_slot) in mean_spec.iter_mut().enumerate() {
            if study.masked_grid_points.get(idx).copied().unwrap_or(false) {
                continue;
            }
            let mut sum = 0.0_f64;
            let mut cnt = 0usize;
            for (si, row) in study.aligned_spectra.iter().enumerate() {
                if let Some(&v) = row.get(idx) {
                    if v.is_finite() {
                        let df = study
                            .dilution_factors
                            .get(si)
                            .copied()
                            .unwrap_or(1.0)
                            .max(1e-15);
                        sum += v / df;
                        cnt += 1;
                    }
                }
            }
            if cnt > 0 {
                let y = sum / (cnt as f64);
                *mean_slot = y;
                unmasked_vals.push(y);
            }
        }
        if unmasked_vals.len() < 8 {
            return Vec::new();
        }

        // 2. Robust baseline & noise floor via median and MAD over unmasked points.
        unmasked_vals.sort_by(|a, b| a.total_cmp(b));
        let baseline = unmasked_vals[unmasked_vals.len() / 2];
        let max_val = *unmasked_vals.last().unwrap_or(&baseline);
        let dynamic_range = (max_val - baseline).max(0.0);
        if dynamic_range <= 1e-15 {
            return Vec::new();
        }

        let mut abs_devs: Vec<f64> = unmasked_vals
            .iter()
            .map(|&v| (v - baseline).abs())
            .collect();
        abs_devs.sort_by(|a, b| a.total_cmp(b));
        let mad = abs_devs[abs_devs.len() / 2];
        let sigma_noise = (1.4826 * mad).max(1e-9 * dynamic_range);

        // Require >= 5*sigma above baseline AND >= 1.5% of tallest peak above baseline.
        let snr_threshold = baseline + (5.0 * sigma_noise).max(0.015 * dynamic_range);
        let min_prominence = (3.0 * sigma_noise).max(0.010 * dynamic_range);

        // 3. Detect strict local maxima with local prominence over +-0.03 ppm neighborhood.
        let grid_step = if m > 1 {
            (study.common_ppm_grid[0] - study.common_ppm_grid[m - 1]).abs() / ((m - 1) as f64)
        } else {
            0.001
        }
        .max(1e-5);
        let prom_half_win = ((0.035 / grid_step).round() as usize).clamp(2, 64);

        let mut raw_peaks: Vec<(f64, f64)> = Vec::new();
        for i in 1..(m - 1) {
            if study.masked_grid_points.get(i).copied().unwrap_or(false)
                || study
                    .masked_grid_points
                    .get(i - 1)
                    .copied()
                    .unwrap_or(false)
                || study
                    .masked_grid_points
                    .get(i + 1)
                    .copied()
                    .unwrap_or(false)
            {
                continue;
            }
            let y = mean_spec[i];
            if y < snr_threshold {
                continue;
            }
            let is_local_max = (y >= mean_spec[i - 1] && y > mean_spec[i + 1])
                || (y > mean_spec[i - 1] && y >= mean_spec[i + 1]);
            if !is_local_max {
                continue;
            }
            // Local prominence: peak must exceed min valley on BOTH sides within window.
            let lo_idx = i.saturating_sub(prom_half_win);
            let hi_idx = (i + prom_half_win).min(m - 1);
            let left_min = mean_spec[lo_idx..i].iter().copied().fold(y, f64::min);
            let right_min = mean_spec[(i + 1)..=hi_idx]
                .iter()
                .copied()
                .fold(y, f64::min);
            let prominence = (y - left_min).min(y - right_min);
            if prominence >= min_prominence {
                let height = (y - baseline).max(0.0);
                raw_peaks.push((study.common_ppm_grid[i], height));
            }
        }
        if raw_peaks.is_empty() {
            return Vec::new();
        }

        // 4. Coalesce adjacent fine-structure multiplet lines (gap <= 0.022 ppm) into one
        //    intensity-weighted centroid peak per multiplet.
        raw_peaks.sort_by(|a, b| a.0.total_cmp(&b.0));
        let coalesce_gap_ppm = 0.022;
        let mut clusters: Vec<Vec<(f64, f64)>> = Vec::new();
        for (ppm, h) in raw_peaks {
            if let Some(last_cluster) = clusters.last_mut() {
                let prev_ppm = last_cluster.last().map(|x| x.0).unwrap_or(ppm);
                if (ppm - prev_ppm).abs() <= coalesce_gap_ppm {
                    last_cluster.push((ppm, h));
                    continue;
                }
            }
            clusters.push(vec![(ppm, h)]);
        }

        let mut picked: Vec<QueryPeak> = clusters
            .into_iter()
            .map(|cl| {
                let sum_h: f64 = cl.iter().map(|&(_, h)| h).sum();
                let centroid = if sum_h > 1e-18 {
                    cl.iter().map(|&(p, h)| p * h).sum::<f64>() / sum_h
                } else {
                    cl[0].0
                };
                let mut qp = QueryPeak::from_ppm(centroid);
                qp.intensity = Some(sum_h);
                qp
            })
            .collect();
        // Order descending by ppm to match NMR axis convention.
        picked.sort_by(|a, b| b.ppm.total_cmp(&a.ppm));
        picked
    }

    /// Extract active `STOCSY` correlated peaks (`r >= min_r`, default `0.85`) refined to apex ppm.
    pub fn extract_stocsy_query_peaks(&self, min_r: f64) -> Vec<QueryPeak> {
        let (Some(ref study), Some(ref st)) = (&self.study, &self.active_stocsy) else {
            return Vec::new();
        };
        let mut peaks = extract_stocsy_cluster_query_peaks(&study.buckets, st, min_r);
        for qp in &mut peaks {
            if let Some(b) = study.buckets.iter().min_by(|a, b| {
                (a.center_ppm - qp.ppm)
                    .abs()
                    .total_cmp(&(b.center_ppm - qp.ppm).abs())
            }) {
                qp.ppm = Self::bucket_apex_ppm(study, b.low_ppm, b.high_ppm, b.center_ppm);
            }
        }
        peaks
    }

    /// Non-blocking poll for any completed background desktop `BMRB v2` online query.
    /// Returns `true` while a background query is still in flight.
    pub fn poll_online_identify_results(&mut self) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use std::sync::mpsc::TryRecvError;
            let outcome = match self.online_query_rx.as_ref() {
                Some(rx) => match rx.try_recv() {
                    Ok(out) => Some(Ok(out)),
                    Err(TryRecvError::Empty) => return self.online_query_in_flight,
                    Err(TryRecvError::Disconnected) => Some(Err(())),
                },
                None => return false,
            };
            self.online_query_rx = None;
            self.online_query_in_flight = false;
            if let Some(Ok(out)) = outcome {
                if out.generation != self.online_query_generation {
                    return false;
                }
                let online_note = match out.result {
                    Ok(online_hits) => {
                        let mut added = 0usize;
                        for hit in online_hits {
                            let dup = self.identify_results.iter().any(|r| {
                                r.entry.source == hit.entry.source
                                    && r.entry.bmse_id == hit.entry.bmse_id
                            });
                            if !dup {
                                self.identify_results.push(hit);
                                added += 1;
                            }
                        }
                        self.identify_results.sort_by(|a, b| {
                            b.score
                                .total_cmp(&a.score)
                                .then_with(|| b.matched_ref_peaks.cmp(&a.matched_ref_peaks))
                                .then_with(|| a.mean_abs_error_ppm.total_cmp(&b.mean_abs_error_ppm))
                                .then_with(|| a.entry.bmse_id.cmp(&b.entry.bmse_id))
                        });
                        self.identify_results.truncate(16);
                        format!(" (+{added} online BMRB v2)")
                    }
                    Err(e) => format!(" | Online unavailable: {e}"),
                };
                self.identify_status = format!(
                    "{} ({} peaks, tol {:.3} ppm): {} offline BMRB-BMSE hits{online_note}",
                    out.origin_label, out.n_peaks, out.tol_ppm, out.offline_count
                );
            }
            false
        }
        #[cfg(target_arch = "wasm32")]
        {
            false
        }
    }

    fn run_identify_query(&mut self, query_peaks: &[QueryPeak], origin_label: &str) {
        if query_peaks.is_empty() {
            self.identify_results.clear();
            self.identify_status = format!("No peaks above SNR threshold for {origin_label}.");
            return;
        }
        let tol = self.identify_tol_ppm.clamp(0.005, 2.0);
        let results = query_offline_metabolites(query_peaks, self.identify_nucleus, tol, 12);
        let offline_count = results.len();
        self.identify_results = results;

        #[cfg(not(target_arch = "wasm32"))]
        if self.include_online_search {
            self.online_query_generation = self.online_query_generation.wrapping_add(1);
            let gen_id = self.online_query_generation;
            let peaks_owned = query_peaks.to_vec();
            let nucleus = self.identify_nucleus;
            let origin_owned = origin_label.to_string();
            let n_peaks = peaks_owned.len();
            let (tx, rx) = std::sync::mpsc::channel();
            self.online_query_rx = Some(rx);
            self.online_query_in_flight = true;
            self.identify_status = format!(
                "{origin_label} ({n_peaks} peaks, tol {tol:.3} ppm): {offline_count} offline BMRB-BMSE hits | Querying BMRB v2 online in background..."
            );
            std::thread::spawn(move || {
                let res = query_online_bmrb_and_npmrd(
                    &peaks_owned,
                    nucleus,
                    tol,
                    OnlineQuerySources::bmrb_only(),
                );
                let _ = tx.send(OnlineIdentifyOutcome {
                    generation: gen_id,
                    origin_label: origin_owned,
                    n_peaks,
                    tol_ppm: tol,
                    offline_count,
                    result: res,
                });
            });
            return;
        }

        self.identify_status = format!(
            "{origin_label} ({} peaks, tol {:.3} ppm): {} offline BMRB-BMSE hits",
            query_peaks.len(),
            tol,
            offline_count
        );
    }

    /// 1-click metabolite identification from picked 1D cohort peaks.
    pub fn identify_from_picked_peaks(&mut self) {
        let peaks = self.extract_picked_1d_query_peaks();
        self.run_identify_query(&peaks, "Picked 1D Peaks");
    }

    /// 1-click metabolite identification from active `STOCSY` correlated peaks (`r >= 0.85`).
    pub fn identify_from_stocsy_cluster(&mut self) {
        let peaks = self.extract_stocsy_query_peaks(DEFAULT_STOCSY_CLUSTER_MIN_R);
        self.run_identify_query(&peaks, "STOCSY Cluster (r >= 0.85)");
    }

    /// Synthesize and activate a 1D reference overlay curve + stick list on Quadrant A.
    pub fn set_reference_overlay(&mut self, entry: &MetaboliteEntry) {
        let obs_mhz = self
            .samples
            .first()
            .map(|s| s.spectrum.axis.obs_mhz)
            .filter(|&m| m.is_finite() && m > 10.0)
            .unwrap_or(600.0);
        let spec = synthesize_reference_spectrum_1d(entry, self.identify_nucleus, obs_mhz, 2048);
        self.status_banner = format!(
            "Reference overlay active: {} ({}) at {:.1} MHz",
            entry.name, entry.bmse_id, obs_mhz
        );
        self.active_reference_overlay = Some(spec);
    }

    /// Clear any active candidate reference spectrum overlay from Quadrant A.
    pub fn clear_reference_overlay(&mut self) {
        self.active_reference_overlay = None;
    }

    /// Render the complete 4-Quadrant Metabolomics & Biomarker Discovery Studio.
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        if self.poll_online_identify_results() {
            ui.ctx()
                .request_repaint_after(std::time::Duration::from_millis(100));
        }
        self.last_plot_rects = [None; 3];
        self.last_canvas_rect = None;
        self.draw_top_toolbar(ui);
        ui.add_space(3.0);

        if self.study.is_none() {
            self.draw_empty_state(ui);
            return;
        }

        let avail = ui.available_size();
        let full_w = avail.x.max(320.0);
        let table_h = if self.show_biomarker_table {
            (avail.y * 0.13).clamp(80.0, 102.0)
        } else {
            0.0
        };
        let identify_h = if self.show_identify_panel {
            (avail.y * 0.23).clamp(132.0, 195.0)
        } else {
            0.0
        };
        let drawer_pad = if self.show_biomarker_table { 4.0 } else { 0.0 }
            + if self.show_identify_panel { 4.0 } else { 0.0 }
            + 2.0;
        let plots_avail_h = (avail.y - table_h - identify_h - drawer_pad).max(260.0);
        let row1_h = (plots_avail_h * 0.52).max(135.0);
        let row2_h = (plots_avail_h - row1_h - 4.0).max(120.0);
        let half_w = ((full_w - 6.0) * 0.5).max(240.0);

        // Row 1: Full-width Quadrant A (1D Spectral Overlay & Live STOCSY Trace)
        ui.allocate_ui_with_layout(
            egui::vec2(full_w, row1_h),
            egui::Layout::top_down(egui::Align::Min),
            |ui| {
                self.render_quadrant_a_overlay(ui, egui::vec2(full_w, row1_h));
            },
        );

        ui.add_space(4.0);

        // Row 2: Quadrant B (PCA / OPLS-DA Score Plot) + Quadrant C (Clickable S-Plot / Volcano)
        ui.horizontal(|ui| {
            ui.allocate_ui_with_layout(
                egui::vec2(half_w, row2_h),
                egui::Layout::top_down(egui::Align::Min),
                |ui| {
                    self.draw_quadrant_b_scores(ui, egui::vec2(half_w, row2_h));
                },
            );
            ui.add_space(6.0);
            ui.allocate_ui_with_layout(
                egui::vec2(half_w, row2_h),
                egui::Layout::top_down(egui::Align::Min),
                |ui| {
                    self.draw_quadrant_c_splot_volcano(ui, egui::vec2(half_w, row2_h));
                },
            );
        });

        // Row 3: Compact full-width Quadrant D (Biomarker & BMRB Table)
        if self.show_biomarker_table {
            ui.add_space(4.0);
            ui.allocate_ui_with_layout(
                egui::vec2(full_w, table_h),
                egui::Layout::top_down(egui::Align::Min),
                |ui| {
                    self.draw_quadrant_d_biomarker_table(ui, egui::vec2(full_w, table_h));
                },
            );
        }

        // Row 4: Identify Metabolite (BMRB-BMSE) Offline + Opt-In Online Drawer
        if self.show_identify_panel {
            ui.add_space(4.0);
            ui.allocate_ui_with_layout(
                egui::vec2(full_w, identify_h),
                egui::Layout::top_down(egui::Align::Min),
                |ui| {
                    self.draw_identify_metabolite_section(ui, egui::vec2(full_w, identify_h));
                },
            );
        }

        self.last_canvas_rect = self
            .last_plot_rects
            .into_iter()
            .flatten()
            .reduce(egui::Rect::union);
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
                    ui.label(
                        egui::RichText::new(
                            "Untargeted 1D NMR Metabolomics, Chemometrics & STOCSY Biomarker Studio",
                        )
                        .size(15.0)
                        .strong()
                        .color(egui::Color32::from_rgb(140, 220, 255)),
                    );
                    ui.add_space(8.0);
                    ui.label(
                        egui::RichText::new(
                            "Interval FFT Peak Alignment | Adaptive Intelligent Binning | PQN Normalization | PCA / OPLS-DA | S-Plot / Volcano | Live STOCSY + BMRB-BMSE Library",
                        )
                        .size(12.5),
                    );
                    ui.add_space(16.0);
                    if ui
                        .add(
                            egui::Button::new(
                                egui::RichText::new("Load 24-Sample Biofluid Demo Cohort")
                                    .size(14.0)
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
            .inner_margin(egui::Margin::symmetric(8, 5))
            .show(ui, |ui| {
                // Primary Toolbar (<= 8 visible primary interactive controls)
                ui.horizontal_wrapped(|ui| {
                    ui.label(
                        egui::RichText::new("1D Metabolomics Studio")
                            .size(15.0)
                            .strong()
                            .color(egui::Color32::from_rgb(185, 230, 255)),
                    );

                    if self.is_synthetic_demo_cohort {
                        egui::Frame::new()
                            .fill(egui::Color32::from_rgb(52, 42, 18))
                            .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(202, 138, 4)))
                            .corner_radius(3.0)
                            .inner_margin(egui::Margin::symmetric(6, 2))
                            .show(ui, |ui| {
                                ui.label(
                                    egui::RichText::new("Synthetic Demo Cohort")
                                        .size(11.0)
                                        .monospace()
                                        .strong()
                                        .color(egui::Color32::from_rgb(253, 224, 71)),
                                );
                            });
                    }

                    if self.study.is_none()
                        && ui
                            .add(
                                egui::Button::new(
                                    egui::RichText::new("Load 24-Sample Demo")
                                        .size(12.5)
                                        .strong()
                                        .color(egui::Color32::from_rgb(185, 245, 255)),
                                )
                                .fill(egui::Color32::from_rgb(24, 82, 128)),
                            )
                            .on_hover_text(
                                "Synthesize 24 human biofluid 1D 1H spectra (12 Control, 12 Case) with sample dilutions, pH jitter, and Lactate/Valine/TMAO biomarkers",
                            )
                            .clicked()
                    {
                        self.demo_seed = self.demo_seed.wrapping_add(1);
                        self.load_demo_cohort();
                    }

                    if !self.samples.is_empty() && ui.button("Re-Run").clicked() {
                        trigger_rerun = true;
                    }

                    ui.separator();

                    let spec_lbl = match self.spectrum_mode {
                        SpectrumViewMode::CohortOverlay => "1D Overlay",
                        SpectrumViewMode::StocsyCovarianceHeat => "STOCSY Cov + r^2",
                    };
                    egui::ComboBox::from_id_salt("metab_spec_mode")
                        .selected_text(spec_lbl)
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut self.spectrum_mode,
                                SpectrumViewMode::CohortOverlay,
                                "1D Overlay",
                            );
                            ui.selectable_value(
                                &mut self.spectrum_mode,
                                SpectrumViewMode::StocsyCovarianceHeat,
                                "STOCSY Cov + r^2",
                            );
                        });

                    let score_lbl = match self.score_mode {
                        ScorePlotMode::OplsDa => "OPLS-DA Scores",
                        ScorePlotMode::Pca => "PCA Scores",
                        ScorePlotMode::PlsDa => "PLS-DA Scores",
                    };
                    egui::ComboBox::from_id_salt("metab_score_mode")
                        .selected_text(score_lbl)
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut self.score_mode,
                                ScorePlotMode::OplsDa,
                                "OPLS-DA Scores",
                            );
                            ui.selectable_value(
                                &mut self.score_mode,
                                ScorePlotMode::Pca,
                                "PCA Scores",
                            );
                            ui.selectable_value(
                                &mut self.score_mode,
                                ScorePlotMode::PlsDa,
                                "PLS-DA Scores",
                            );
                        });

                    let feat_lbl = match self.feature_mode {
                        FeaturePlotMode::OplsSPlot => "OPLS-DA S-Plot",
                        FeaturePlotMode::VolcanoPlot => "BH-FDR Volcano",
                    };
                    egui::ComboBox::from_id_salt("metab_feat_mode")
                        .selected_text(feat_lbl)
                        .show_ui(ui, |ui| {
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
                        });

                    if ui
                        .selectable_label(self.show_biomarker_table, "Biomarker Table")
                        .clicked()
                    {
                        self.show_biomarker_table = !self.show_biomarker_table;
                    }

                    if ui
                        .selectable_label(self.show_identify_panel, "Identify (BMRB-BMSE)")
                        .on_hover_text(
                            "Open offline BMRB-BMSE & opt-in BMRB v2 metabolite identification drawer",
                        )
                        .clicked()
                    {
                        self.show_identify_panel = !self.show_identify_panel;
                    }
                });

                // Progressive-disclosure Advanced Preprocessing & Export
                egui::CollapsingHeader::new(
                    egui::RichText::new("Advanced Preprocessing & Export").size(12.5),
                )
                .default_open(false)
                .show(ui, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        if ui.button("Load 24-Sample Demo").clicked() {
                            self.demo_seed = self.demo_seed.wrapping_add(1);
                            self.load_demo_cohort();
                        }
                        if self.study.is_some() && ui.button("Full Sweep").clicked() {
                            if let Some(ref st) = self.study {
                                self.view_ppm_bounds = (st.config.min_ppm, st.config.max_ppm);
                            }
                        }
                        ui.separator();

                        let is_adaptive = matches!(
                            self.config.bucketing,
                            BucketingMethod::AdaptiveIntelligent { .. }
                        );
                        ui.label(egui::RichText::new("Bucketing:").size(11.0));
                        if ui.selectable_label(is_adaptive, "Adaptive (AIB)").clicked()
                            && !is_adaptive
                        {
                            self.config.bucketing = BucketingMethod::default();
                            trigger_rerun = true;
                        }
                        if ui.selectable_label(!is_adaptive, "Uniform").clicked() && is_adaptive {
                            self.config.bucketing = BucketingMethod::Uniform {
                                bin_width_ppm: 0.04,
                                min_bucket_snr: 3.0,
                            };
                            trigger_rerun = true;
                        }
                        if let BucketingMethod::Uniform {
                            ref mut bin_width_ppm,
                            ..
                        } = self.config.bucketing
                        {
                            ui.horizontal(|ui| {
                                ui.label(egui::RichText::new("d_ppm:").size(11.0));
                                if ui
                                    .add(
                                        egui::DragValue::new(bin_width_ppm)
                                            .range(0.005..=0.20)
                                            .speed(0.005)
                                            .fixed_decimals(3),
                                    )
                                    .changed()
                                {
                                    trigger_rerun = true;
                                }
                            });
                        }

                        ui.separator();

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
                                    self.config.normalization =
                                        NormalizationMethod::TotalSpectralArea;
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
                                    self.config.normalization =
                                        NormalizationMethod::InternalStandard {
                                            target_ppm: 0.0,
                                            half_window_ppm: 0.08,
                                        };
                                    trigger_rerun = true;
                                }
                                if ui
                                    .selectable_label(
                                        matches!(
                                            self.config.normalization,
                                            NormalizationMethod::None
                                        ),
                                        "None",
                                    )
                                    .clicked()
                                {
                                    self.config.normalization = NormalizationMethod::None;
                                    trigger_rerun = true;
                                }
                            });

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
                                    if ui.selectable_label(self.config.scaling == m, lbl).clicked()
                                    {
                                        self.config.scaling = m;
                                        trigger_rerun = true;
                                    }
                                }
                            });

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
                                        self.config.mask_preset
                                            == BiofluidMaskPreset::AqueousMinimal,
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

                        ui.checkbox(&mut self.show_bucket_boundaries, "Buckets");
                        ui.checkbox(&mut self.auto_zoom_on_select, "Auto-Zoom 1D");

                        if let Some(ref st) = self.study {
                            ui.separator();
                            if ui
                                .button("Export CSV")
                                .on_hover_text(
                                    "Copy ranked candidate biomarker table CSV to clipboard",
                                )
                                .clicked()
                            {
                                ui.ctx().copy_text(export_biomarker_ranking_csv(st));
                                self.status_banner =
                                    "Copied ranked biomarker discovery CSV to clipboard."
                                        .to_string();
                            }
                            if ui
                                .button("Copy Buckets CSV")
                                .on_hover_text(
                                    "Copy normalized N x P bucket matrix CSV to clipboard",
                                )
                                .clicked()
                            {
                                ui.ctx().copy_text(export_bucket_table_csv(st));
                                self.status_banner =
                                    "Copied normalized bucket matrix CSV to clipboard.".to_string();
                            }
                            if ui
                                .button("Copy Summary JSON")
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
                });

                // Compact Statistical Reporting Banner (neutral scientific candor, zero "VALIDATED")
                if let Some(ref study) = self.study {
                    ui.add_space(2.0);
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
                                "Cohort: {} spectra x {} buckets | PCA 2-comp R²X={:.1}%",
                                study.sample_ids.len(),
                                study.buckets.len(),
                                pca_r2 * 100.0
                            ))
                            .size(11.0)
                            .monospace()
                            .color(egui::Color32::from_rgb(175, 205, 235)),
                        );

                        if let Some(ref op) = study.opls_da {
                            ui.separator();
                            let stats_txt = format!(
                                "CV Q² = {:.2} (7-fold), R²Y = {:.2}, permutation p <= {:.3} (n={})",
                                op.q2,
                                op.r2_y,
                                op.permutation.p_value_q2,
                                op.permutation.n_permutations
                            );
                            let (badge_txt, fill_c, txt_c) = if op.permutation.is_validated {
                                (
                                    stats_txt,
                                    egui::Color32::from_rgb(24, 52, 68),
                                    egui::Color32::from_rgb(186, 230, 253),
                                )
                            } else {
                                (
                                    format!("[!] Overfit warning | {stats_txt}"),
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
                                            .size(11.0)
                                            .monospace()
                                            .strong()
                                            .color(txt_c),
                                    );
                                });
                        }
                    });
                    if self.status_banner.starts_with("Copied ") {
                        ui.label(
                            egui::RichText::new(&self.status_banner)
                                .size(11.0)
                                .color(egui::Color32::from_rgb(203, 213, 225)),
                        );
                    }
                }
            });

        if trigger_rerun {
            self.recompute_pipeline();
        }
    }

    // ========================================================================
    // Quadrant A: Cohort 1D Spectral Overlay & Live STOCSY r^2 Heat Trace
    // ========================================================================

    pub fn render_quadrant_a_overlay(&mut self, ui: &mut egui::Ui, size: egui::Vec2) {
        let mut clicked_bucket: Option<usize> = None;

        egui::Frame::new()
            .fill(egui::Color32::from_rgb(14, 18, 26))
            .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(42, 54, 74)))
            .corner_radius(5.0)
            .inner_margin(egui::Margin::same(4))
            .show(ui, |ui| {
                ui.set_min_size(size - egui::vec2(8.0, 8.0));
                let canvas_h = (size.y - 8.0).max(120.0);
                let canvas_w = (size.x - 8.0).max(180.0);
                let (rect, resp) = ui.allocate_exact_size(
                    egui::vec2(canvas_w, canvas_h),
                    egui::Sense::click_and_drag(),
                );
                let painter = ui.painter_at(rect);
                painter.rect_filled(rect, 3.0, egui::Color32::from_rgb(10, 14, 21));

                painter.text(
                    egui::pos2(rect.left() + 8.0, rect.top() + 4.0),
                    egui::Align2::LEFT_TOP,
                    "A. Cohort 1D Overlay & STOCSY Trace",
                    egui::FontId::proportional(12.5),
                    egui::Color32::from_rgb(135, 215, 255),
                );

                let Some(ref study) = self.study else {
                    return;
                };
                let plot_rect = egui::Rect::from_min_max(
                    egui::pos2(rect.left() + 8.0, rect.top() + 20.0),
                    egui::pos2(rect.right() - 8.0, rect.bottom() - 28.0),
                );
                if plot_rect.width() <= 16.0 || plot_rect.height() <= 16.0 {
                    return;
                }
                self.last_plot_rects[0] = Some(rect);

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

                // 2b. Optional 1-click candidate reference spectrum overlay (sticks + Lorentzian curve)
                if let Some(ref overlay) = self.active_reference_overlay {
                    if overlay.nucleus == MetaboliteNucleus::H1 {
                        let stick_col = egui::Color32::from_rgba_unmultiplied(52, 211, 153, 145);
                        let curve_col = egui::Color32::from_rgb(52, 211, 153);
                        let max_h = plot_rect.height() * 0.82;

                        for &(sppm, sh) in &overlay.sticks {
                            if sppm < v_lo || sppm > v_hi {
                                continue;
                            }
                            let sx = ppm_to_x(sppm);
                            let sy = plot_rect.bottom() - (sh.clamp(0.0, 1.0) as f32) * max_h;
                            painter.line_segment(
                                [egui::pos2(sx, plot_rect.bottom()), egui::pos2(sx, sy)],
                                egui::Stroke::new(1.2, stick_col),
                            );
                        }

                        let n_ov = overlay.ppm_grid.len();
                        if n_ov > 1 && overlay.intensities.len() == n_ov {
                            let ov_start = overlay.ppm_grid.partition_point(|&p| p > v_hi);
                            let ov_end = overlay
                                .ppm_grid
                                .partition_point(|&p| p >= v_lo)
                                .saturating_sub(1)
                                .min(n_ov.saturating_sub(1));
                            if ov_start < ov_end {
                                let max_cols = (plot_rect.width().ceil() as usize).max(160);
                                let ov_stride = ((ov_end - ov_start) / max_cols).max(1);
                                let mut pts =
                                    Vec::with_capacity((ov_end - ov_start) / ov_stride + 2);
                                let mut idx = ov_start;
                                while idx <= ov_end {
                                    let y_val = overlay.intensities[idx].clamp(0.0, 1.0) as f32;
                                    pts.push(egui::pos2(
                                        ppm_to_x(overlay.ppm_grid[idx]),
                                        plot_rect.bottom() - y_val * max_h,
                                    ));
                                    idx += ov_stride;
                                }
                                if pts.len() >= 2 {
                                    painter.add(egui::Shape::line(
                                        pts,
                                        egui::Stroke::new(1.6, curve_col),
                                    ));
                                }
                            }
                        }

                        painter.text(
                            egui::pos2(rect.right() - 10.0, rect.top() + 4.0),
                            egui::Align2::RIGHT_TOP,
                            format!("Ref Overlay: {} ({})", overlay.name, overlay.bmse_id),
                            egui::FontId::monospace(11.0),
                            egui::Color32::from_rgb(110, 231, 183),
                        );
                    }
                }

                // 3. Draw chemical-shift axis ticks along bottom (>= 60 px spacing + single "1H (ppm)" title)
                painter.line_segment(
                    [
                        egui::pos2(plot_rect.left(), plot_rect.bottom()),
                        egui::pos2(plot_rect.right(), plot_rect.bottom()),
                    ],
                    egui::Stroke::new(1.0, egui::Color32::from_rgb(82, 98, 126)),
                );
                let n_ticks = ((plot_rect.width() / 72.0).floor() as usize).clamp(1, 10);
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
                    let align = if t == 0 {
                        egui::Align2::LEFT_TOP
                    } else if t == n_ticks {
                        egui::Align2::RIGHT_TOP
                    } else {
                        egui::Align2::CENTER_TOP
                    };
                    painter.text(
                        egui::pos2(x, plot_rect.bottom() + 4.0),
                        align,
                        format!("{ppm:.1}"),
                        egui::FontId::monospace(11.0),
                        egui::Color32::from_rgb(165, 182, 208),
                    );
                }
                painter.text(
                    egui::pos2(plot_rect.center().x, rect.bottom() - 1.0),
                    egui::Align2::CENTER_BOTTOM,
                    "1H (ppm)",
                    egui::FontId::proportional(11.0),
                    egui::Color32::from_rgb(186, 202, 224),
                );

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
            .inner_margin(egui::Margin::same(4))
            .show(ui, |ui| {
                ui.set_min_size(size - egui::vec2(8.0, 8.0));
                let canvas_h = (size.y - 8.0).max(120.0);
                let canvas_w = (size.x - 8.0).max(180.0);
                let (rect, resp) = ui.allocate_exact_size(
                    egui::vec2(canvas_w, canvas_h),
                    egui::Sense::hover(),
                );
                let painter = ui.painter_at(rect);
                painter.rect_filled(rect, 3.0, egui::Color32::from_rgb(10, 14, 21));
                painter.text(
                    egui::pos2(rect.left() + 8.0, rect.top() + 4.0),
                    egui::Align2::LEFT_TOP,
                    "B. Score Space & 95% Hotelling T^2",
                    egui::FontId::proportional(12.5),
                    egui::Color32::from_rgb(135, 215, 255),
                );

                let Some(ref study) = self.study else {
                    return;
                };
                let plot_rect = egui::Rect::from_min_max(
                    egui::pos2(rect.left() + 34.0, rect.top() + 20.0),
                    egui::pos2(rect.right() - 10.0, rect.bottom() - 20.0),
                );
                if plot_rect.width() <= 20.0 || plot_rect.height() <= 20.0 {
                    return;
                }
                self.last_plot_rects[1] = Some(rect);

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

                let (x_lbl, y_lbl) = match (&self.score_mode, study.opls_da.as_ref(), study.pls_da.as_ref()) {
                    (ScorePlotMode::OplsDa, Some(op), _) => {
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
                    (ScorePlotMode::PlsDa, _, Some(pl)) => {
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
                        format!("[*] {g_name}"),
                        egui::FontId::proportional(11.0),
                        group_color(gi),
                    );
                }

                // Axis captions
                painter.text(
                    egui::pos2(plot_rect.center().x, rect.bottom() - 3.0),
                    egui::Align2::CENTER_BOTTOM,
                    x_lbl,
                    egui::FontId::monospace(11.0),
                    egui::Color32::from_rgb(165, 182, 208),
                );
                painter.text(
                    egui::pos2(plot_rect.left() + 4.0, plot_rect.top() + 2.0),
                    egui::Align2::LEFT_TOP,
                    y_lbl,
                    egui::FontId::monospace(11.0),
                    egui::Color32::from_rgb(165, 182, 208),
                );

                if let Some(sp) = hovered_sample {
                    let si = sp.sample_idx;
                    resp.on_hover_text(format!(
                        "{} ({})\nScore: ({:.3}, {:.3})\nHotelling T^2: {:.2} (95% lim {:.2})\nQ-residual (SPE): {:.3} (95% lim {:.3})",
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
            .inner_margin(egui::Margin::same(4))
            .show(ui, |ui| {
                ui.set_min_size(size - egui::vec2(8.0, 8.0));
                let canvas_h = (size.y - 8.0).max(120.0);
                let canvas_w = (size.x - 8.0).max(180.0);
                let (rect, resp) = ui.allocate_exact_size(
                    egui::vec2(canvas_w, canvas_h),
                    egui::Sense::click(),
                );
                let painter = ui.painter_at(rect);
                painter.rect_filled(rect, 3.0, egui::Color32::from_rgb(10, 14, 21));
                painter.text(
                    egui::pos2(rect.left() + 8.0, rect.top() + 4.0),
                    egui::Align2::LEFT_TOP,
                    "C. Biomarker S-Plot / Volcano",
                    egui::FontId::proportional(12.5),
                    egui::Color32::from_rgb(135, 215, 255),
                );

                let Some(ref study) = self.study else {
                    return;
                };
                if study.biomarkers.is_empty() {
                    return;
                }
                let plot_rect = egui::Rect::from_min_max(
                    egui::pos2(rect.left() + 34.0, rect.top() + 20.0),
                    egui::pos2(rect.right() - 10.0, rect.bottom() - 20.0),
                );
                if plot_rect.width() <= 20.0 || plot_rect.height() <= 20.0 {
                    return;
                }
                self.last_plot_rects[2] = Some(rect);

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
                    egui::pos2(plot_rect.center().x, rect.bottom() - 3.0),
                    egui::Align2::CENTER_BOTTOM,
                    x_lbl,
                    egui::FontId::monospace(11.0),
                    egui::Color32::from_rgb(165, 182, 208),
                );
                painter.text(
                    egui::pos2(plot_rect.left() + 4.0, plot_rect.top() + 2.0),
                    egui::Align2::LEFT_TOP,
                    y_lbl,
                    egui::FontId::monospace(11.0),
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
    // Quadrant D: Compact Ranked Candidate Biomarker & BMRB Metabolite Table
    // ========================================================================

    fn draw_quadrant_d_biomarker_table(&mut self, ui: &mut egui::Ui, size: egui::Vec2) {
        let mut clicked_bucket: Option<usize> = None;

        egui::Frame::new()
            .fill(egui::Color32::from_rgb(14, 18, 26))
            .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(42, 54, 74)))
            .corner_radius(5.0)
            .inner_margin(egui::Margin::symmetric(8, 4))
            .show(ui, |ui| {
                ui.set_min_size(size - egui::vec2(16.0, 8.0));
                ui.horizontal_wrapped(|ui| {
                    ui.label(
                        egui::RichText::new("D. Candidate Biomarkers & STOCSY BMRB Matches")
                            .size(12.5)
                            .strong()
                            .color(egui::Color32::from_rgb(135, 215, 255)),
                    );
                    if let Some(ref st) = self.active_stocsy {
                        ui.separator();
                        if let Some(top) = st.metabolite_matches.first() {
                            ui.label(
                                egui::RichText::new(format!(
                                    "Driver {:.3} ppm -> {} [{}] (score {:.2}, {}/{} multiplets)",
                                    st.driver_ppm,
                                    top.name,
                                    top.bmrb_id,
                                    top.score,
                                    top.matched_multiplets,
                                    top.observable_multiplets,
                                ))
                                .size(11.0)
                                .monospace()
                                .color(egui::Color32::from_rgb(190, 242, 214)),
                            );
                        } else {
                            ui.label(
                                egui::RichText::new(format!(
                                    "Driver {:.3} ppm -- no BMRB library match within shift tolerance",
                                    st.driver_ppm
                                ))
                                .size(11.0)
                                .monospace()
                                .color(egui::Color32::from_rgb(165, 180, 204)),
                            );
                        }
                    }
                });

                let Some(ref study) = self.study else {
                    return;
                };

                egui::ScrollArea::vertical()
                    .id_salt("metabolomics_biomarker_table_scroll")
                    .max_height((ui.available_height() - 2.0).max(52.0))
                    .show(ui, |ui| {
                        egui::Grid::new("metabolomics_biomarker_grid")
                            .striped(true)
                            .spacing([14.0, 2.0])
                            .show(ui, |ui| {
                                for hdr in [
                                    "#",
                                    "Shift (ppm)",
                                    "VIP",
                                    "p(corr)",
                                    "log2 FC",
                                    "FDR q",
                                    "STOCSY r^2",
                                    "Putative BMRB Metabolite",
                                ] {
                                    ui.label(
                                        egui::RichText::new(hdr)
                                            .size(11.0)
                                            .strong()
                                            .color(egui::Color32::from_rgb(148, 184, 226)),
                                    );
                                }
                                ui.end_row();

                                for bm in study.biomarkers.iter().take(25) {
                                    let is_sel = self.selected_bucket_idx == Some(bm.bucket_index);
                                    let r2 = self
                                        .active_stocsy
                                        .as_ref()
                                        .and_then(|s| s.bucket_r2.get(bm.bucket_index).copied())
                                        .unwrap_or(0.0);

                                    let row_col = if is_sel {
                                        egui::Color32::from_rgb(250, 204, 21)
                                    } else {
                                        egui::Color32::from_rgb(226, 232, 240)
                                    };
                                    let rank_lbl = ui.add(
                                        egui::Label::new(
                                            egui::RichText::new(format!("{:>2}", bm.rank))
                                                .size(11.0)
                                                .monospace()
                                                .color(row_col),
                                        )
                                        .sense(egui::Sense::click()),
                                    );
                                    let ppm_lbl = ui.add(
                                        egui::Label::new(
                                            egui::RichText::new(format!("{:.3}", bm.center_ppm))
                                                .size(11.0)
                                                .monospace()
                                                .strong()
                                                .color(row_col),
                                        )
                                        .sense(egui::Sense::click()),
                                    );
                                    if rank_lbl.clicked() || ppm_lbl.clicked() {
                                        clicked_bucket = Some(bm.bucket_index);
                                    }

                                    ui.label(
                                        egui::RichText::new(format!("{:.2}", bm.vip))
                                            .size(11.0)
                                            .monospace()
                                            .color(if bm.vip >= 1.0 {
                                                egui::Color32::from_rgb(134, 239, 172)
                                            } else {
                                                egui::Color32::from_rgb(160, 174, 192)
                                            }),
                                    );
                                    ui.label(
                                        egui::RichText::new(format!("{:+.2}", bm.s_plot_corr))
                                            .size(11.0)
                                            .monospace(),
                                    );
                                    ui.label(
                                        egui::RichText::new(format!("{:+.2}", bm.log2_fold_change))
                                            .size(11.0)
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
                                            .size(11.0)
                                            .monospace()
                                            .color(if bm.fdr_q <= 0.05 {
                                                egui::Color32::from_rgb(167, 243, 208)
                                            } else {
                                                egui::Color32::from_rgb(148, 163, 184)
                                            }),
                                    );
                                    ui.label(
                                        egui::RichText::new(format!("{:.2}", r2))
                                            .size(11.0)
                                            .monospace()
                                            .color(stocsy_r2_color(r2)),
                                    );

                                    let met_txt =
                                        match (&bm.putative_metabolite, &bm.putative_bmrb_id) {
                                            (Some(m), Some(id)) => format!("{m} ({id})"),
                                            (Some(m), None) => m.clone(),
                                            _ => "--".to_string(),
                                        };
                                    if ui
                                        .add(
                                            egui::Label::new(
                                                egui::RichText::new(met_txt)
                                                    .size(11.0)
                                                    .color(row_col),
                                            )
                                            .sense(egui::Sense::click()),
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

    // ========================================================================
    // Identify Metabolite (BMRB-BMSE) Offline + Opt-In Desktop Online Drawer
    // ========================================================================

    fn draw_identify_metabolite_section(&mut self, ui: &mut egui::Ui, size: egui::Vec2) {
        let mut trigger_picked = false;
        let mut trigger_stocsy = false;
        let mut trigger_clear_overlay = false;
        let mut selected_overlay: Option<MetaboliteEntry> = None;
        let mut selected_octet_smiles: Option<(String, String)> = None;

        let active_overlay_id = self
            .active_reference_overlay
            .as_ref()
            .map(|o| o.bmse_id.clone());

        egui::Frame::new()
            .fill(egui::Color32::from_rgb(14, 18, 26))
            .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(42, 54, 74)))
            .corner_radius(5.0)
            .inner_margin(egui::Margin::symmetric(8, 4))
            .show(ui, |ui| {
                ui.set_min_size(size - egui::vec2(16.0, 8.0));
                ui.horizontal_wrapped(|ui| {
                    ui.label(
                        egui::RichText::new("Identify Metabolite (BMRB-BMSE)")
                            .size(12.5)
                            .strong()
                            .color(egui::Color32::from_rgb(135, 215, 255)),
                    );
                    ui.separator();

                    if ui
                        .button(egui::RichText::new("Query Picked 1D Peaks").size(11.5))
                        .on_hover_text(
                            "Pick genuine 1D local-maxima peaks above SNR threshold and match against the offline BMRB-BMSE index",
                        )
                        .clicked()
                    {
                        trigger_picked = true;
                    }

                    let has_stocsy = self.active_stocsy.is_some();
                    if ui
                        .add_enabled(
                            has_stocsy,
                            egui::Button::new(
                                egui::RichText::new("Query STOCSY (r >= 0.85)").size(11.5),
                            ),
                        )
                        .on_hover_text(
                            "Match active STOCSY correlated peak cluster (r >= 0.85) against BMRB-BMSE",
                        )
                        .clicked()
                    {
                        trigger_stocsy = true;
                    }

                    ui.label(egui::RichText::new("Tol (ppm):").size(11.0));
                    ui.add(
                        egui::DragValue::new(&mut self.identify_tol_ppm)
                            .range(0.005..=0.50)
                            .speed(0.002)
                            .fixed_decimals(3),
                    );

                    #[cfg(not(target_arch = "wasm32"))]
                    ui.checkbox(
                        &mut self.include_online_search,
                        egui::RichText::new("Include Online Search (BMRB v2)").size(11.5),
                    )
                    .on_hover_text(
                        "Opt-in background HTTPS query to api.bmrb.io (curl -q, 5 s timeout, 4 MiB cap; never blocks UI or runs automatically)",
                    );

                    ui.hyperlink_to(
                        egui::RichText::new("NP-MRD Web Search").size(11.0),
                        npmrd_web_search_url(),
                    )
                    .on_hover_text(
                        "Open NP-MRD (CC-BY-4.0) web portal at https://np-mrd.org/ in your browser",
                    );

                    if active_overlay_id.is_some()
                        && ui
                            .button(egui::RichText::new("Clear Overlay").size(11.0))
                            .clicked()
                    {
                        trigger_clear_overlay = true;
                    }
                });

                ui.label(
                    egui::RichText::new(&self.identify_status)
                        .size(11.0)
                        .monospace()
                        .color(egui::Color32::from_rgb(176, 198, 224)),
                );

                egui::ScrollArea::vertical()
                    .id_salt("metabolomics_identify_table_scroll")
                    .max_height((ui.available_height() - 2.0).max(60.0))
                    .show(ui, |ui| {
                        egui::Grid::new("metabolomics_identify_grid")
                            .striped(true)
                            .spacing([12.0, 2.0])
                            .show(ui, |ui| {
                                for hdr in [
                                    "#",
                                    "BMSE ID",
                                    "Metabolite",
                                    "Score",
                                    "Matched/Total",
                                    "Mean d(ppm)",
                                    "Source",
                                    "Actions",
                                ] {
                                    ui.label(
                                        egui::RichText::new(hdr)
                                            .size(11.0)
                                            .strong()
                                            .color(egui::Color32::from_rgb(148, 184, 226)),
                                    );
                                }
                                ui.end_row();

                                for (rank, hit) in self.identify_results.iter().take(12).enumerate()
                                {
                                    let is_overlaid = active_overlay_id.as_deref()
                                        == Some(hit.entry.bmse_id.as_str());
                                    let row_col = if is_overlaid {
                                        egui::Color32::from_rgb(110, 231, 183)
                                    } else {
                                        egui::Color32::from_rgb(226, 232, 240)
                                    };
                                    ui.label(
                                        egui::RichText::new(format!("{:>2}", rank + 1))
                                            .size(11.0)
                                            .monospace()
                                            .color(row_col),
                                    );
                                    ui.label(
                                        egui::RichText::new(&hit.entry.bmse_id)
                                            .size(11.0)
                                            .monospace()
                                            .color(row_col),
                                    );
                                    ui.label(
                                        egui::RichText::new(&hit.entry.name)
                                            .size(11.0)
                                            .strong()
                                            .color(row_col),
                                    );
                                    ui.label(
                                        egui::RichText::new(format!("{:.3}", hit.score))
                                            .size(11.0)
                                            .monospace()
                                            .color(if hit.score >= 0.80 {
                                                egui::Color32::from_rgb(134, 239, 172)
                                            } else {
                                                egui::Color32::from_rgb(203, 213, 225)
                                            }),
                                    );
                                    let matched_lbl = if hit.total_ref_peaks > 0 {
                                        format!(
                                            "{}/{}",
                                            hit.matched_ref_peaks, hit.total_ref_peaks
                                        )
                                    } else {
                                        format!("{}/?", hit.matched_ref_peaks)
                                    };
                                    ui.label(
                                        egui::RichText::new(matched_lbl)
                                            .size(11.0)
                                            .monospace(),
                                    );
                                    ui.label(
                                        egui::RichText::new(format!(
                                            "{:.4}",
                                            hit.mean_abs_error_ppm
                                        ))
                                        .size(11.0)
                                        .monospace(),
                                    );
                                    ui.label(
                                        egui::RichText::new(hit.entry.source.label())
                                            .size(11.0)
                                            .color(egui::Color32::from_rgb(160, 182, 208)),
                                    );

                                    ui.horizontal(|ui| {
                                        let ov_lbl = if is_overlaid {
                                            "Overlay [On]"
                                        } else {
                                            "Overlay"
                                        };
                                        if ui
                                            .button(egui::RichText::new(ov_lbl).size(11.0))
                                            .on_hover_text(
                                                "Synthesize 1D reference sticks and Lorentzian curve onto Quadrant A",
                                            )
                                            .clicked()
                                        {
                                            selected_overlay = Some(hit.entry.clone());
                                        }
                                        let has_smiles = !hit.entry.smiles.trim().is_empty();
                                        if ui
                                            .add_enabled(
                                                has_smiles,
                                                egui::Button::new(
                                                    egui::RichText::new("Send to Octet").size(11.0),
                                                ),
                                            )
                                            .on_hover_text(
                                                "Queue metabolite SMILES to open in the Octet-Draw 2D structure editor",
                                            )
                                            .clicked()
                                        {
                                            selected_octet_smiles = Some((
                                                hit.entry.name.clone(),
                                                hit.entry.smiles.clone(),
                                            ));
                                        }
                                    });
                                    ui.end_row();
                                }
                            });
                    });
            });

        if trigger_picked {
            self.identify_from_picked_peaks();
        }
        if trigger_stocsy {
            self.identify_from_stocsy_cluster();
        }
        if trigger_clear_overlay {
            self.clear_reference_overlay();
        }
        if let Some(entry) = selected_overlay {
            self.set_reference_overlay(&entry);
        }
        if let Some((name, smiles)) = selected_octet_smiles {
            self.pending_send_smiles_to_octet = Some(smiles);
            self.status_banner = format!("Queued {name} SMILES for Octet-Draw workbench.");
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
        let mut out1 = ctx.run_ui(egui::RawInput::default(), |ui| {
            studio.ui(ui);
        });
        out1.textures_delta.clear();

        // Load 24-spectrum demo cohort and render OPLS-DA + S-plot + Cohort Overlay
        studio.load_demo_cohort();
        assert!(studio.is_synthetic_demo_cohort);
        assert!(studio.study.is_some());
        assert!(studio.active_stocsy.is_some());
        assert!(
            !studio.status_banner.to_uppercase().contains("VALIDATED"),
            "status_banner must never claim VALIDATED: {}",
            studio.status_banner
        );
        assert!(studio.status_banner.contains("Synthetic Demo Cohort"));

        let raw_in = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1440.0, 900.0),
            )),
            ..Default::default()
        };
        let mut out2 = ctx.run_ui(raw_in.clone(), |ui| {
            egui::CentralPanel::default().show(ui, |ui| {
                studio.ui(ui);
            });
        });
        out2.textures_delta.clear();

        let plot_area_sum: f32 = studio
            .last_plot_rects
            .iter()
            .filter_map(|r| r.map(|rect| rect.area()))
            .sum();
        assert!(
            plot_area_sum >= 0.65 * 1440.0 * 900.0,
            "Metabolomics plots A+B+C area {:.0} must be >= 65% of 1440x900 ({:.0})",
            plot_area_sum,
            0.65 * 1440.0 * 900.0
        );

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
        let mut out3 = ctx.run_ui(raw_in, |ui| {
            egui::CentralPanel::default().show(ui, |ui| {
                studio.ui(ui);
            });
        });
        out3.textures_delta.clear();
    }

    #[test]
    fn test_identify_metabolite_drawer_stocsy_overlay_and_send_to_octet_headless() {
        let ctx = egui::Context::default();
        let mut studio = MetabolomicsStudioPanel::default();
        assert!(
            !studio.include_online_search,
            "online search must be opt-in (false by default)"
        );
        assert!(studio.pending_send_smiles_to_octet.is_none());

        studio.config.n_permutations = 10;
        studio.load_demo_cohort();
        studio.show_identify_panel = true;

        // 1. Identify from genuine picked 1D local-maxima cohort peaks
        studio.identify_from_picked_peaks();
        assert!(
            !studio.identify_results.is_empty(),
            "picked 1D peak identification should return offline BMRB-BMSE candidates"
        );

        // 2. Select L-Lactate 1.33 ppm driver and identify from active STOCSY cluster (r >= 0.85)
        let lac_idx = studio
            .study
            .as_ref()
            .expect("study loaded")
            .buckets
            .iter()
            .min_by(|a, b| {
                (a.center_ppm - 1.330)
                    .abs()
                    .total_cmp(&(b.center_ppm - 1.330).abs())
            })
            .expect("lactate bucket")
            .index;
        studio.select_bucket_and_run_stocsy(lac_idx, false);
        studio.identify_from_stocsy_cluster();
        let top = studio
            .identify_results
            .first()
            .expect("top STOCSY metabolite hit")
            .clone();
        assert_eq!(top.entry.bmse_id, "bmse000208");
        assert_eq!(top.entry.name, "L-(+) Lactic acid");

        // 3. Activate 1-click reference overlay & simulate Send to Octet handoff
        studio.set_reference_overlay(&top.entry);
        assert!(studio.active_reference_overlay.is_some());
        studio.pending_send_smiles_to_octet = Some(top.entry.smiles.clone());
        assert_eq!(
            studio.pending_send_smiles_to_octet.as_deref(),
            Some("C[C@@H](C(=O)O)O")
        );

        // 4. Render with Identify drawer open + active reference overlay at 1440x900
        let raw_in = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1440.0, 900.0),
            )),
            ..Default::default()
        };
        let mut out = ctx.run_ui(raw_in, |ui| {
            egui::CentralPanel::default().show(ui, |ui| {
                studio.ui(ui);
            });
        });
        out.textures_delta.clear();
        assert!(studio.last_plot_rects[0].is_some());
        assert!(studio.last_canvas_rect.is_some());
    }

    #[test]
    fn test_picked_1d_peaks_uses_local_maxima_above_snr_not_bucket_comb() {
        use nmr_core::nmr1d::{Experiment1DKind, Spectrum1D};
        use nmr_core::rng::Rng;
        use nmr_core::spectrum::AxisPpm;

        // Build a 6-sample cohort containing ONLY L-(+) Lactic acid (1.318 doublet + 4.101 quartet)
        // plus baseline noise, and use dense Uniform 0.04 ppm bucketing (min_bucket_snr = 0.0)
        // which produces ~180+ uniform buckets across 0.5..9.5 ppm.
        let n_pts = 4096usize;
        let axis = AxisPpm::from_bounds("1H", n_pts, 9.5, 0.5, 600.0);
        let mut rng = Rng::new(20261119);
        let mut samples = Vec::new();
        for s in 0..6 {
            let mut real = vec![0.0_f64; n_pts];
            let scale = if s < 3 { 1.0 } else { 2.0 };
            let gamma = 0.0025_f64;
            let g2 = gamma * gamma;
            for (k, slot) in real.iter_mut().enumerate() {
                let ppm = axis.index_to_ppm(k as f64);
                let mut y = rng.normal(0.0, 0.003);
                // Doublet at 1.318 ppm (J = 7.5 Hz @ 600 MHz -> +-0.00625 ppm)
                for &c in &[1.31175, 1.32425] {
                    let d = ppm - c;
                    y += scale * 1.5 * g2 / (d * d + g2);
                }
                // Quartet at 4.101 ppm (J = 7.0 Hz @ 600 MHz -> step 0.01167 ppm)
                for &(c, w) in &[
                    (4.0835, 0.125),
                    (4.0952, 0.375),
                    (4.1068, 0.375),
                    (4.1185, 0.125),
                ] {
                    let d = ppm - c;
                    y += scale * w * 1.0 * g2 / (d * d + g2);
                }
                *slot = y;
            }
            let spec = Spectrum1D::new(
                format!("S{s}"),
                Experiment1DKind::Proton1D,
                axis.clone(),
                real,
                None,
            );
            samples.push(CohortSample1D::new(
                format!("S{s}"),
                if s < 3 { "Control" } else { "Case" },
                spec,
            ));
        }

        let mut studio = MetabolomicsStudioPanel::default();
        studio.config.n_permutations = 5;
        studio.config.alignment = AlignmentMode::Disabled;
        studio.config.bucketing = BucketingMethod::Uniform {
            bin_width_ppm: 0.04,
            min_bucket_snr: 0.0,
        };
        studio.set_cohort_and_run(samples);

        let n_buckets = studio.study.as_ref().expect("study").buckets.len();
        assert!(
            n_buckets > 100,
            "uniform 0.04 ppm bucketing should create >100 buckets (got {n_buckets})"
        );

        let picked = studio.extract_picked_1d_query_peaks();
        assert_eq!(
            picked.len(),
            2,
            "genuine local-maxima peak picker must extract ONLY the 2 real resonances (not {n_buckets} bucket apices!), got {picked:?}"
        );
        assert!(picked.iter().any(|p| (p.ppm - 1.318).abs() < 0.015));
        assert!(picked.iter().any(|p| (p.ppm - 4.101).abs() < 0.015));

        studio.identify_from_picked_peaks();
        let top = studio
            .identify_results
            .first()
            .expect("top picked-peaks hit");
        assert_eq!(top.entry.bmse_id, "bmse000208");
        assert_eq!(top.entry.name, "L-(+) Lactic acid");

        // Pure-noise cohort must extract 0 peaks (never a dense bucket comb)
        let mut noise_samples = Vec::new();
        for s in 0..4 {
            let real: Vec<f64> = (0..n_pts).map(|_| rng.normal(0.0, 0.005)).collect();
            noise_samples.push(CohortSample1D::new(
                format!("N{s}"),
                if s < 2 { "A" } else { "B" },
                Spectrum1D::new(
                    format!("N{s}"),
                    Experiment1DKind::Proton1D,
                    axis.clone(),
                    real,
                    None,
                ),
            ));
        }
        studio.set_cohort_and_run(noise_samples);
        let noise_picked = studio.extract_picked_1d_query_peaks();
        assert!(
            noise_picked.len() <= 2,
            "pure-noise spectrum must not produce a dense bucket comb (got {} peaks)",
            noise_picked.len()
        );
    }
}
