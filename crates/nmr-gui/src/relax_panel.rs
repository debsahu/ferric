//! Interactive `Relaxation / S²` dynamics studio (`T13.1`–`T13.3`) for `nmr-gui` (native & `wasm32`).
//!
//! Renders interactive residue $I(\tau)$ mono-exponential $R_1$ / $R_2$ decay curves with fitted
//! decay models + parametric Monte Carlo $1\sigma$ uncertainty bands, plus synchronized per-residue
//! $R_1$, $R_2$, steady-state $\{^1\text{H}\}\text{–}^{15}\text{N}$ hetNOE, Lipari–Szabo $S^2$, and
//! $R_{\text{ex}}$ sequence bar tracks with 1-click export to the 3D PDB `QuantitativeTrack` putty.
//!
//! # Literature & Clean-Room Provenance
//! - Kay, L. E., Torchia, D. A. & Bax, A. (1989). *Biochemistry* 28(23), 8972–8979.
//! - Farrow, N. A. et al. (1994). *Biochemistry* 33(19), 5984–6003; (1995) *J. Biomol. NMR* 6, 153–162.
//! - Lipari, G. & Szabo, A. (1982a, 1982b). *J. Am. Chem. Soc.* 104(17), 4546–4559 & 4559–4570.
//! - Clore, G. M., Szabo, A., Bax, A., Kay, L. E., Driscoll, P. C. & Gronenborn, A. M. (1990).
//!   *J. Am. Chem. Soc.* 112(12), 4989–4991.
//! - Mandel, A. M., Akke, M. & Palmer, A. G. III (1995). *J. Mol. Biol.* 246(1), 144–163.
//! - Woessner, D. E. (1962). *J. Chem. Phys.* 37(3), 647–654.
//! - d'Auvergne, E. J. & Gooley, P. R. (2003). *J. Biomol. NMR* 25(1), 25–39.

use std::collections::BTreeMap;

use eframe::egui;
use nmr_core::modelfree::{
    extract_nh_unit_vectors_from_chain, fit_modelfree_dataset, group_multifield_rates,
    predict_r1_r2_noe, write_modelfree_tsv_str, DiffusionTensor, DiffusionTensorKind,
    LocalMfParams, ModelFreeConfig, ModelFreeDatasetResult, ModelFreeId, ModelSelectionCriterion,
};
use nmr_core::noesy::parse_pdb_str_first_model;
use nmr_core::relax_fit::{
    compute_hetnoe_batch, decay_inputs_from_series, fit_decay_batch_cpu, DecayCurveFit,
    DecayCurveInput, ExpFitConfig, HetNoeInput, HetNoeResult, OffsetModelGate, RelaxPhysicsConfig,
    ResidueRelaxRates,
};
use nmr_core::relax_io::{
    parse_relax_series_nef_str, parse_relax_series_tsv_str, LockedPeakFootprint,
    RelaxExperimentKind, RelaxSchedule, RelaxSeriesDataset, ResidueRelaxTrace,
};
use nmr_core::rng::Rng;

use crate::mol3d_overlay::QuantitativeTrack;

/// Which decay curve family is displayed in the interactive $I(\tau)$ curve canvas.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ActiveDecayView {
    #[default]
    R1,
    R2,
}

/// Actions emitted by `RelaxPanelState` to synchronize with the rest of the workbench.
#[derive(Clone, Debug, PartialEq)]
pub enum RelaxPanelAction {
    /// User clicked a residue bar (`0`-based index into `sequence` / `seq_ids`).
    SelectResidueIndex(usize),
    /// 1-click export of Lipari–Szabo $S^2$ and $R_{\text{ex}}$ onto the 3D PDB `QuantitativeTrack` putty.
    ExportTo3dTrack {
        track: QuantitativeTrack,
        s2_by_seq_idx: BTreeMap<usize, f64>,
        rex_by_seq_idx: BTreeMap<usize, f64>,
        rex_max_s_inv: f64,
    },
    /// Status banner update for the main workbench.
    StatusMessage(String),
}

/// Interactive state for the `Relaxation / S²` dynamics studio (`wasm32`-safe).
#[derive(Clone, Debug)]
pub struct RelaxPanelState {
    pub active_decay: ActiveDecayView,
    pub mc_replicates: usize,
    pub offset_gate: OffsetModelGate,
    pub criterion: ModelSelectionCriterion,
    pub diffusion_kind: DiffusionTensorKind,

    pub primary_field_mhz: f64,
    pub r1_curves: BTreeMap<i32, DecayCurveInput>,
    pub r2_curves: BTreeMap<i32, DecayCurveInput>,
    pub r1_fits: BTreeMap<i32, DecayCurveFit>,
    pub r2_fits: BTreeMap<i32, DecayCurveFit>,
    pub hetnoe_fits: BTreeMap<i32, HetNoeResult>,
    pub aux_field_rates: Vec<Vec<ResidueRelaxRates>>,
    pub modelfree_result: Option<ModelFreeDatasetResult>,

    pub selected_seq_id: Option<i32>,
    pub show_paste_drawer: bool,
    pub paste_series_buf: String,
    pub exported_tsv_preview: String,
    pub show_tsv_modal: bool,
    pub status_banner: String,
    pub last_canvas_rect: Option<egui::Rect>,
}

impl Default for RelaxPanelState {
    fn default() -> Self {
        Self::new()
    }
}

impl RelaxPanelState {
    pub fn new() -> Self {
        Self {
            active_decay: ActiveDecayView::R1,
            mc_replicates: 200,
            offset_gate: OffsetModelGate::Disabled,
            criterion: ModelSelectionCriterion::Aic,
            diffusion_kind: DiffusionTensorKind::Auto,
            primary_field_mhz: 600.0,
            r1_curves: BTreeMap::new(),
            r2_curves: BTreeMap::new(),
            r1_fits: BTreeMap::new(),
            r2_fits: BTreeMap::new(),
            hetnoe_fits: BTreeMap::new(),
            aux_field_rates: Vec::new(),
            modelfree_result: None,
            selected_seq_id: None,
            show_paste_drawer: false,
            paste_series_buf: String::new(),
            exported_tsv_preview: String::new(),
            show_tsv_modal: false,
            status_banner:
                "Load a relaxation series TSV/NEF or synthesize dual-field 600+800 MHz dynamics."
                    .to_string(),
            last_canvas_rect: None,
        }
    }

    pub fn has_fitted_rates(&self) -> bool {
        !self.r1_fits.is_empty() || !self.r2_fits.is_empty() || !self.hetnoe_fits.is_empty()
    }

    /// Import a relaxation series (`R1`, `R2`/`R1rho`, or `hetNOE`) from TSV or NEF text and fit immediately.
    pub fn import_series_text(&mut self, text: &str) -> Result<String, String> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Err("empty relaxation series text".to_string());
        }
        let ds: RelaxSeriesDataset = if trimmed.contains("save_nmrassign_relaxation_series") {
            parse_relax_series_nef_str(trimmed)?
        } else {
            parse_relax_series_tsv_str(trimmed)?
        };
        self.ingest_series_dataset(&ds)
    }

    /// Fit an integrated `RelaxSeriesDataset` (`R1`, `R2`/`R1rho`, or `hetNOE`) and store curves + MC uncertainties.
    pub fn ingest_series_dataset(&mut self, ds: &RelaxSeriesDataset) -> Result<String, String> {
        let fm = if ds.field_mhz.is_finite() && ds.field_mhz > 0.0 {
            ds.field_mhz
        } else {
            600.0
        };
        self.primary_field_mhz = fm;

        match ds.experiment_kind {
            RelaxExperimentKind::HetNoe => {
                if ds.schedule.len() < 2 {
                    return Err("hetNOE series requires >= 2 planes (unsat, sat)".to_string());
                }
                let vals = ds.schedule.primary_values();
                let (u_idx, s_idx) = if vals[0].abs() <= 1.0e-9 && vals[1] > 0.0 {
                    (0usize, 1usize)
                } else if vals[1].abs() <= 1.0e-9 && vals[0] > 0.0 {
                    (1usize, 0usize)
                } else {
                    (0usize, 1usize)
                };
                let inputs: Vec<HetNoeInput> = ds
                    .traces
                    .iter()
                    .map(|tr| HetNoeInput {
                        seq_id: tr.footprint.seq_id,
                        res_name: tr.footprint.res_name.clone(),
                        i_sat: tr.intensities[s_idx],
                        sigma_sat: tr.sigmas[s_idx],
                        i_unsat: tr.intensities[u_idx],
                        sigma_unsat: tr.sigmas[u_idx],
                    })
                    .collect();
                let res = compute_hetnoe_batch(&inputs, self.mc_replicates, 2026_1302)?;
                self.hetnoe_fits.clear();
                for r in res {
                    if self.selected_seq_id.is_none() {
                        self.selected_seq_id = Some(r.seq_id);
                    }
                    self.hetnoe_fits.insert(r.seq_id, r);
                }
                let msg = format!(
                    "Computed steady-state hetNOE for {} residues ({fm:.1} MHz)",
                    self.hetnoe_fits.len()
                );
                self.status_banner.clone_from(&msg);
                Ok(msg)
            }
            RelaxExperimentKind::R1 | RelaxExperimentKind::R2 | RelaxExperimentKind::R1rho => {
                let curves = decay_inputs_from_series(ds);
                let cfg = ExpFitConfig {
                    offset_gate: self.offset_gate,
                    mc_replicates: self.mc_replicates,
                    seed: 2026_1302,
                    ..ExpFitConfig::default()
                };
                let fits = fit_decay_batch_cpu(&curves, &cfg)?;
                let is_r1 = ds.experiment_kind == RelaxExperimentKind::R1;
                if is_r1 {
                    self.r1_curves.clear();
                    self.r1_fits.clear();
                    for (c, f) in curves.into_iter().zip(fits) {
                        if self.selected_seq_id.is_none() {
                            self.selected_seq_id = Some(f.seq_id);
                        }
                        self.r1_curves.insert(c.seq_id, c);
                        self.r1_fits.insert(f.seq_id, f);
                    }
                    self.active_decay = ActiveDecayView::R1;
                } else {
                    self.r2_curves.clear();
                    self.r2_fits.clear();
                    for (c, f) in curves.into_iter().zip(fits) {
                        if self.selected_seq_id.is_none() {
                            self.selected_seq_id = Some(f.seq_id);
                        }
                        self.r2_curves.insert(c.seq_id, c);
                        self.r2_fits.insert(f.seq_id, f);
                    }
                    self.active_decay = ActiveDecayView::R2;
                }
                let count = if is_r1 {
                    self.r1_fits.len()
                } else {
                    self.r2_fits.len()
                };
                let msg = format!(
                    "Fitted {} curves for {count} residues ({fm:.1} MHz, {} MC trials)",
                    ds.experiment_kind.as_str(),
                    self.mc_replicates
                );
                self.status_banner.clone_from(&msg);
                Ok(msg)
            }
            other => Err(format!(
                "unsupported series kind '{}' for R1/R2/hetNOE panel",
                other.as_str()
            )),
        }
    }

    /// Re-fit currently loaded $R_1$ and $R_2$ decay curves with updated `mc_replicates` / `offset_gate`.
    pub fn refit_loaded_curves(&mut self) -> Result<String, String> {
        let cfg = ExpFitConfig {
            offset_gate: self.offset_gate,
            mc_replicates: self.mc_replicates,
            seed: 2026_1302,
            ..ExpFitConfig::default()
        };
        if !self.r1_curves.is_empty() {
            let curves: Vec<DecayCurveInput> = self.r1_curves.values().cloned().collect();
            let fits = fit_decay_batch_cpu(&curves, &cfg)?;
            self.r1_fits = fits.into_iter().map(|f| (f.seq_id, f)).collect();
        }
        if !self.r2_curves.is_empty() {
            let curves: Vec<DecayCurveInput> = self.r2_curves.values().cloned().collect();
            let fits = fit_decay_batch_cpu(&curves, &cfg)?;
            self.r2_fits = fits.into_iter().map(|f| (f.seq_id, f)).collect();
        }
        let msg = format!(
            "Re-fitted {} R1 and {} R2 curves ({} MC trials)",
            self.r1_fits.len(),
            self.r2_fits.len(),
            self.mc_replicates
        );
        self.status_banner.clone_from(&msg);
        Ok(msg)
    }

    /// Assemble primary-field `ResidueRelaxRates` from fitted $R_1$, $R_2$, and optional hetNOE.
    pub fn primary_rate_table(&self) -> Vec<ResidueRelaxRates> {
        let mut out = Vec::new();
        for (&seq_id, r1_fit) in &self.r1_fits {
            let Some(r2_fit) = self.r2_fits.get(&seq_id) else {
                continue;
            };
            let (hetnoe, sigma_hetnoe) = match self.hetnoe_fits.get(&seq_id) {
                Some(n) => (Some(n.noe), Some(n.sigma_noe.max(1.0e-4))),
                None => (None, None),
            };
            out.push(ResidueRelaxRates {
                seq_id,
                res_name: r1_fit.res_name.clone(),
                field_mhz: self.primary_field_mhz,
                r1: r1_fit.rate,
                sigma_r1: r1_fit.sigma_rate_mc.max(1.0e-4),
                r2: r2_fit.rate,
                sigma_r2: r2_fit.sigma_rate_mc.max(1.0e-4),
                hetnoe,
                sigma_hetnoe,
            });
        }
        out
    }

    /// Run Lipari–Szabo / Mandel–Akke–Palmer `M1`–`M5` model selection and global diffusion optimization.
    pub fn run_modelfree(&mut self, pdb_text: Option<&str>) -> Result<String, String> {
        let primary = self.primary_rate_table();
        if primary.is_empty() {
            return Err("Fit both R1 and R2 rates before running Model-Free M1-M5".to_string());
        }
        let mut tables = vec![primary];
        for aux in &self.aux_field_rates {
            if !aux.is_empty() {
                tables.push(aux.clone());
            }
        }

        let nh_vecs = pdb_text
            .and_then(|txt| parse_pdb_str_first_model(txt).ok())
            .map(|chain| extract_nh_unit_vectors_from_chain(&chain))
            .filter(|v| !v.is_empty());

        let diff_kind = if nh_vecs.is_none()
            && matches!(
                self.diffusion_kind,
                DiffusionTensorKind::Auto
                    | DiffusionTensorKind::AxialSymmetric
                    | DiffusionTensorKind::FullyAnisotropic
            ) {
            DiffusionTensorKind::Isotropic
        } else {
            self.diffusion_kind
        };

        let grouped = group_multifield_rates(&tables, nh_vecs.as_deref());
        let cfg = ModelFreeConfig {
            diffusion_kind: diff_kind,
            selection_criterion: self.criterion,
            mc_replicates: 60,
            seed: 2026_1303,
            ..ModelFreeConfig::default()
        };
        let res = fit_modelfree_dataset(&grouped, &cfg)?;
        let tm_ns = res.diffusion_tensor.effective_tau_m_sec() * 1.0e9;
        let n_res = res.residues.len();
        self.exported_tsv_preview = write_modelfree_tsv_str(&res);
        self.modelfree_result = Some(res);

        let msg = format!(
            "Converged Model-Free M1-M5 on {n_res} residues (effective tau_m = {tm_ns:.2} ns, {:?})",
            self.criterion
        );
        self.status_banner.clone_from(&msg);
        Ok(msg)
    }

    /// Synthesize realistic dual-field (`600 + 800 MHz`) $R_1$, $R_2$, and hetNOE series for the
    /// active sequence (or a 36-residue globular + flexible loop + $R_{\text{ex}}$ showcase), fit
    /// curves with parametric MC uncertainties, and solve Lipari–Szabo `M1`–`M5`.
    pub fn load_demo_dynamics(
        &mut self,
        sequence: &[String],
        seq_ids: &[i32],
        pdb_text: Option<&str>,
    ) -> Result<String, String> {
        let fallback_seq: Vec<String> = (0..36)
            .map(|k| match k % 6 {
                0 => "MET",
                1 => "GLN",
                2 => "ILE",
                3 => "PHE",
                4 => "VAL",
                _ => "LYS",
            })
            .map(String::from)
            .collect();
        let fallback_ids: Vec<i32> = (1..=36).collect();
        let seq = if sequence.is_empty() {
            &fallback_seq
        } else {
            sequence
        };
        let ids = if seq_ids.len() == seq.len() && !seq_ids.is_empty() {
            seq_ids
        } else {
            &fallback_ids[..seq.len().min(fallback_ids.len())]
        };

        let nh_map: BTreeMap<i32, [f64; 3]> = pdb_text
            .and_then(|txt| parse_pdb_str_first_model(txt).ok())
            .map(|c| extract_nh_unit_vectors_from_chain(&c))
            .unwrap_or_default()
            .into_iter()
            .collect();

        let tensor = if !nh_map.is_empty() {
            DiffusionTensor::AxialSymmetric {
                d_perp_s_inv: 3.55e7,
                d_par_s_inv: 5.05e7,
                theta_rad: 0.60,
                phi_rad: 1.00,
            }
        } else {
            DiffusionTensor::Isotropic { tau_m_sec: 4.3e-9 }
        };
        let phys = RelaxPhysicsConfig::default();
        let mut rng = Rng::new(2026_1350);

        let r1_delays = vec![0.01, 0.05, 0.12, 0.25, 0.45, 0.70, 1.00, 1.35];
        let r2_delays = vec![0.008, 0.024, 0.048, 0.080, 0.120, 0.176, 0.240, 0.320];
        let n_len = seq.len().min(ids.len());

        let mut r1_traces = Vec::new();
        let mut r2_traces = Vec::new();
        let mut noe_traces = Vec::new();
        let mut rates_800 = Vec::new();

        for idx in 0..n_len {
            let res_name = &seq[idx];
            if res_name == "PRO" {
                continue;
            }
            let seq_id = ids[idx];
            let frac = (idx as f64) / (n_len.max(1) as f64);

            // Realistic protein profile: flexible termini (`M5`), exchange loop (`M3`/`M4`), rigid core (`M1`/`M2`)
            let local = if idx < 2 || idx + 3 >= n_len {
                LocalMfParams::m5(0.88, 0.58, 950.0e-12)
            } else if (0.28..0.38).contains(&frac) {
                LocalMfParams {
                    s2: 0.81,
                    sf2: 1.0,
                    ss2: 0.81,
                    tau_e_sec: 48.0e-12,
                    rex_ref_s_inv: 3.4,
                }
            } else if (0.62..0.70).contains(&frac) {
                LocalMfParams {
                    s2: 0.85,
                    sf2: 1.0,
                    ss2: 0.85,
                    tau_e_sec: 0.0,
                    rex_ref_s_inv: 2.5,
                }
            } else if idx % 3 == 1 {
                LocalMfParams {
                    s2: 0.83,
                    sf2: 1.0,
                    ss2: 0.83,
                    tau_e_sec: 38.0e-12,
                    rex_ref_s_inv: 0.0,
                }
            } else {
                LocalMfParams {
                    s2: 0.88,
                    sf2: 1.0,
                    ss2: 0.88,
                    tau_e_sec: 0.0,
                    rex_ref_s_inv: 0.0,
                }
            };

            let comps = tensor.rotational_components(nh_map.get(&seq_id).copied());
            let (r1_600, r2_600, noe_600) = predict_r1_r2_noe(&comps, &local, 600.0, 600.0, &phys);
            let (r1_800, r2_800, noe_800) = predict_r1_r2_noe(&comps, &local, 800.0, 600.0, &phys);

            let fp = LockedPeakFootprint {
                seq_id,
                res_name: res_name.clone(),
                atom_h: "H".to_string(),
                atom_x: "N".to_string(),
                delta_h_ppm: 8.20,
                delta_x_ppm: 120.0,
                sigma_h_ppm: 0.022,
                sigma_x_ppm: 0.20,
                eta_h: 0.0,
                eta_x: 0.0,
                overlap_cluster_id: idx,
            };
            let i0 = 1000.0;
            let sig = 7.5;

            let r1_int: Vec<f64> = r1_delays
                .iter()
                .map(|&t| i0 * (-r1_600 * t).exp() + rng.normal(0.0, sig))
                .collect();
            let r2_int: Vec<f64> = r2_delays
                .iter()
                .map(|&t| i0 * (-r2_600 * t).exp() + rng.normal(0.0, sig))
                .collect();
            let noe_int = vec![
                i0 + rng.normal(0.0, 6.0),
                i0 * noe_600 + rng.normal(0.0, 6.0),
            ];

            r1_traces.push(ResidueRelaxTrace {
                footprint: fp.clone(),
                volumes: r1_int.clone(),
                intensities: r1_int,
                sigmas: vec![sig; r1_delays.len()],
            });
            r2_traces.push(ResidueRelaxTrace {
                footprint: fp.clone(),
                volumes: r2_int.clone(),
                intensities: r2_int,
                sigmas: vec![sig; r2_delays.len()],
            });
            noe_traces.push(ResidueRelaxTrace {
                footprint: fp,
                volumes: noe_int.clone(),
                intensities: noe_int,
                sigmas: vec![6.0, 6.0],
            });

            rates_800.push(ResidueRelaxRates {
                seq_id,
                res_name: res_name.clone(),
                field_mhz: 800.0,
                r1: (r1_800 + rng.normal(0.0, 0.015)).max(0.05),
                sigma_r1: 0.020,
                r2: (r2_800 + rng.normal(0.0, 0.08)).max(0.1),
                sigma_r2: 0.10,
                hetnoe: Some(noe_800 + rng.normal(0.0, 0.010)),
                sigma_hetnoe: Some(0.012),
            });
        }

        let ds_r1 = RelaxSeriesDataset {
            name: "demo_r1_600".to_string(),
            experiment_kind: RelaxExperimentKind::R1,
            field_mhz: 600.0,
            schedule: RelaxSchedule::VariableDelaysSec {
                delays_sec: r1_delays.clone(),
            },
            plane_noise_mad: vec![7.5; r1_delays.len()],
            plane_is_duplicate: vec![false; r1_delays.len()],
            duplicate_variance_scale: 1.0,
            duplicate_dof: 0,
            traces: r1_traces,
        };
        let ds_r2 = RelaxSeriesDataset {
            name: "demo_r2_600".to_string(),
            experiment_kind: RelaxExperimentKind::R2,
            field_mhz: 600.0,
            schedule: RelaxSchedule::VariableDelaysSec {
                delays_sec: r2_delays.clone(),
            },
            plane_noise_mad: vec![7.5; r2_delays.len()],
            plane_is_duplicate: vec![false; r2_delays.len()],
            duplicate_variance_scale: 1.0,
            duplicate_dof: 0,
            traces: r2_traces,
        };
        let ds_noe = RelaxSeriesDataset {
            name: "demo_noe_600".to_string(),
            experiment_kind: RelaxExperimentKind::HetNoe,
            field_mhz: 600.0,
            schedule: RelaxSchedule::VariableDelaysSec {
                delays_sec: vec![0.0, 5.0],
            },
            plane_noise_mad: vec![6.0, 6.0],
            plane_is_duplicate: vec![false, false],
            duplicate_variance_scale: 1.0,
            duplicate_dof: 0,
            traces: noe_traces,
        };

        self.selected_seq_id = None;
        self.ingest_series_dataset(&ds_r1)?;
        self.ingest_series_dataset(&ds_r2)?;
        self.ingest_series_dataset(&ds_noe)?;
        self.active_decay = ActiveDecayView::R1;
        self.aux_field_rates = vec![rates_800];
        self.run_modelfree(pdb_text)
    }

    /// Build a 1-click 3D PDB `QuantitativeTrack` export mapping `seq_id -> 0-based seq_idx`.
    pub fn build_3d_track_export(
        &self,
        seq_ids: &[i32],
        track: QuantitativeTrack,
    ) -> Option<RelaxPanelAction> {
        let mf = self.modelfree_result.as_ref()?;
        let id_to_idx: BTreeMap<i32, usize> = seq_ids
            .iter()
            .enumerate()
            .map(|(idx, &sid)| (sid, idx))
            .collect();

        let mut s2_by_seq_idx = BTreeMap::new();
        let mut rex_by_seq_idx = BTreeMap::new();
        let mut rex_max = 1.0_f64;

        for (pos, r) in mf.residues.iter().enumerate() {
            let idx = id_to_idx.get(&r.seq_id).copied().unwrap_or(pos);
            s2_by_seq_idx.insert(idx, r.params.s2.clamp(0.0, 1.0));
            let rex = r.params.rex_ref_s_inv.max(0.0);
            if rex > 1.0e-4 {
                rex_by_seq_idx.insert(idx, rex);
                rex_max = rex_max.max(rex);
            }
        }

        Some(RelaxPanelAction::ExportTo3dTrack {
            track,
            s2_by_seq_idx,
            rex_by_seq_idx,
            rex_max_s_inv: rex_max.max(5.0),
        })
    }

    /// Render the interactive `Relaxation / S²` workbench view.
    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        sequence: &[String],
        seq_ids: &[i32],
        selected_res_idx: Option<usize>,
        pdb_text: Option<&str>,
    ) -> Vec<RelaxPanelAction> {
        let mut actions = Vec::new();

        if let Some(idx) = selected_res_idx {
            if let Some(&sid) = seq_ids.get(idx) {
                if self.r1_fits.contains_key(&sid)
                    || self.r2_fits.contains_key(&sid)
                    || self.hetnoe_fits.contains_key(&sid)
                {
                    self.selected_seq_id = Some(sid);
                }
            }
        }

        // Top Toolbar (<= 8 visible primary interactive controls)
        ui.horizontal_wrapped(|ui| {
            ui.label(
                egui::RichText::new("Relaxation (R1 / R2 / hetNOE) & Lipari-Szabo S^2")
                    .size(15.0)
                    .strong(),
            );
            ui.separator();

            if ui
                .button("Demo 600+800 MHz")
                .on_hover_text(
                    "Synthesize dual-field R1, R2 & hetNOE series, fit MC curves, and solve M1-M5",
                )
                .clicked()
            {
                match self.load_demo_dynamics(sequence, seq_ids, pdb_text) {
                    Ok(msg) => actions.push(RelaxPanelAction::StatusMessage(msg)),
                    Err(e) => self.status_banner = format!("Demo error: {e}"),
                }
            }

            if (!self.r1_curves.is_empty() || !self.r2_curves.is_empty())
                && ui.button("Fit Curves").clicked()
            {
                match self.refit_loaded_curves() {
                    Ok(msg) => actions.push(RelaxPanelAction::StatusMessage(msg)),
                    Err(e) => self.status_banner = format!("Curve fit error: {e}"),
                }
            }

            if (!self.r1_fits.is_empty() && !self.r2_fits.is_empty())
                && ui.button("Run M1-M5").clicked()
            {
                match self.run_modelfree(pdb_text) {
                    Ok(msg) => actions.push(RelaxPanelAction::StatusMessage(msg)),
                    Err(e) => self.status_banner = format!("Model-Free error: {e}"),
                }
            }

            let curve_lbl = match self.active_decay {
                ActiveDecayView::R1 => "Curve: R1(tau)",
                ActiveDecayView::R2 => "Curve: R2(tau)",
            };
            egui::ComboBox::from_id_salt("relax_panel_curve_view")
                .selected_text(curve_lbl)
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.active_decay, ActiveDecayView::R1, "R1(tau)");
                    ui.selectable_value(&mut self.active_decay, ActiveDecayView::R2, "R2(tau)");
                });

            if self.modelfree_result.is_some() {
                if ui
                    .button("Map S^2 -> 3D Putty")
                    .on_hover_text("Color & scale 3D PDB B-factor putty by Lipari-Szabo S^2")
                    .clicked()
                {
                    if let Some(act) =
                        self.build_3d_track_export(seq_ids, QuantitativeTrack::LipariSzaboS2)
                    {
                        actions.push(act);
                    }
                }
                if ui
                    .button("Map R_ex -> 3D Putty")
                    .on_hover_text("Color & scale 3D PDB B-factor putty by chemical exchange R_ex")
                    .clicked()
                {
                    if let Some(act) =
                        self.build_3d_track_export(seq_ids, QuantitativeTrack::ChemicalExchangeRex)
                    {
                        actions.push(act);
                    }
                }
            }
        });

        // Progressive-disclosure Advanced Fitting & Export section (collapsed by default)
        egui::CollapsingHeader::new(egui::RichText::new("Advanced Fitting & Export").size(12.5))
            .default_open(false)
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("Monte Carlo:").size(12.5));
                        ui.add(
                            egui::DragValue::new(&mut self.mc_replicates)
                                .range(0..=2000)
                                .speed(10),
                        );
                    });
                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("Field 1 (MHz):").size(12.5));
                        ui.add(
                            egui::DragValue::new(&mut self.primary_field_mhz)
                                .range(200.0..=1200.0)
                                .speed(10.0)
                                .fixed_decimals(1),
                        );
                    });
                    ui.separator();
                    ui.label(egui::RichText::new("Criterion:").size(12.5));
                    egui::ComboBox::from_id_salt("relax_panel_mf_crit")
                        .selected_text(format!("{:?}", self.criterion))
                        .width(88.0)
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut self.criterion,
                                ModelSelectionCriterion::Aic,
                                "AIC",
                            );
                            ui.selectable_value(
                                &mut self.criterion,
                                ModelSelectionCriterion::Aicc,
                                "AICc",
                            );
                            ui.selectable_value(
                                &mut self.criterion,
                                ModelSelectionCriterion::Bic,
                                "BIC",
                            );
                            ui.selectable_value(
                                &mut self.criterion,
                                ModelSelectionCriterion::PalmerFTest,
                                "Palmer F-test",
                            );
                        });
                    ui.label(egui::RichText::new("Diffusion / Tau_m:").size(12.5));
                    egui::ComboBox::from_id_salt("relax_panel_diff_kind")
                        .selected_text(format!("{:?}", self.diffusion_kind))
                        .width(115.0)
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut self.diffusion_kind,
                                DiffusionTensorKind::Auto,
                                "Auto (AIC)",
                            );
                            ui.selectable_value(
                                &mut self.diffusion_kind,
                                DiffusionTensorKind::Isotropic,
                                "Isotropic",
                            );
                            ui.selectable_value(
                                &mut self.diffusion_kind,
                                DiffusionTensorKind::AxialSymmetric,
                                "AxialSymmetric",
                            );
                            ui.selectable_value(
                                &mut self.diffusion_kind,
                                DiffusionTensorKind::FullyAnisotropic,
                                "FullyAnisotropic",
                            );
                        });

                    ui.separator();
                    if ui
                        .button("Paste Series TSV/NEF")
                        .on_hover_text("Import integrated relaxation series TSV or NEF v1.1 text")
                        .clicked()
                    {
                        self.show_paste_drawer = !self.show_paste_drawer;
                    }
                    if self.modelfree_result.is_some() && ui.button("Export TSV").clicked() {
                        self.show_tsv_modal = !self.show_tsv_modal;
                    }
                });
                ui.label(
                    egui::RichText::new(&self.status_banner)
                        .size(11.0)
                        .color(egui::Color32::from_rgb(155, 195, 225)),
                );
            });

        self.last_canvas_rect = Some(ui.available_rect_before_wrap().intersect(ui.clip_rect()));

        if self.show_paste_drawer {
            ui.group(|ui| {
                ui.label(
                    egui::RichText::new(
                        "Paste Relaxation Series TSV or `save_nmrassign_relaxation_series` NEF:",
                    )
                    .size(12.5),
                );
                ui.add(
                    egui::TextEdit::multiline(&mut self.paste_series_buf)
                        .desired_rows(4)
                        .desired_width(f32::INFINITY)
                        .font(egui::TextStyle::Monospace),
                );
                ui.horizontal(|ui| {
                    if ui.button("Import & Fit Series").clicked() {
                        let txt = self.paste_series_buf.clone();
                        match self.import_series_text(&txt) {
                            Ok(msg) => {
                                self.show_paste_drawer = false;
                                actions.push(RelaxPanelAction::StatusMessage(msg));
                            }
                            Err(e) => self.status_banner = format!("Import error: {e}"),
                        }
                    }
                    if ui.button("Close").clicked() {
                        self.show_paste_drawer = false;
                    }
                });
            });
        }

        if self.show_tsv_modal && !self.exported_tsv_preview.is_empty() {
            ui.group(|ui| {
                ui.horizontal(|ui| {
                    ui.strong("Model-Free M1-M5 TSV Output");
                    if ui.button("Copy").clicked() {
                        ui.ctx().copy_text(self.exported_tsv_preview.clone());
                    }
                    if ui.button("Hide").clicked() {
                        self.show_tsv_modal = false;
                    }
                });
                egui::ScrollArea::vertical()
                    .max_height(110.0)
                    .show(ui, |ui| {
                        ui.monospace(&self.exported_tsv_preview);
                    });
            });
        }

        ui.separator();

        if !self.has_fitted_rates() {
            ui.vertical_centered(|ui| {
                ui.add_space(28.0);
                ui.label(
                    egui::RichText::new(
                        "No relaxation curves loaded yet. Click 'Demo 600+800 MHz' or open 'Advanced Fitting & Export' to paste TSV/NEF.",
                    )
                    .size(12.5),
                );
            });
            return actions;
        }

        let avail_h = ui.available_height().max(420.0);
        let decay_h = (avail_h * 0.28).clamp(148.0, 210.0);

        egui::ScrollArea::vertical().show(ui, |ui| {
            // 1. Interactive I(tau) mono-exponential decay plot with fitted curve + MC 1-sigma envelope
            self.draw_decay_curve_canvas(ui, sequence, seq_ids, decay_h, &mut actions);
            ui.add_space(4.0);

            // 2. Per-residue sequence bar tracks: R1, R2, hetNOE, S^2, R_ex
            let track_h = ((ui.available_height() - 42.0) / 5.0).clamp(44.0, 74.0);
            self.draw_sequence_tracks_canvas(ui, sequence, seq_ids, track_h, &mut actions);
        });

        actions
    }

    fn draw_decay_curve_canvas(
        &mut self,
        ui: &mut egui::Ui,
        sequence: &[String],
        seq_ids: &[i32],
        canvas_height: f32,
        actions: &mut Vec<RelaxPanelAction>,
    ) {
        let (curves, fits, label, accent) = match self.active_decay {
            ActiveDecayView::R1 => (
                &self.r1_curves,
                &self.r1_fits,
                "R1",
                egui::Color32::from_rgb(76, 185, 255),
            ),
            ActiveDecayView::R2 => (
                &self.r2_curves,
                &self.r2_fits,
                "R2",
                egui::Color32::from_rgb(255, 162, 68),
            ),
        };

        let all_sids: Vec<i32> = fits.keys().copied().collect();
        let active_sid = self
            .selected_seq_id
            .filter(|sid| fits.contains_key(sid))
            .or_else(|| all_sids.first().copied());
        self.selected_seq_id = active_sid;

        let Some(sid) = active_sid else {
            ui.label(egui::RichText::new(format!("No {label} curves fitted yet.")).size(11.0));
            return;
        };
        let (Some(curve), Some(fit)) = (curves.get(&sid), fits.get(&sid)) else {
            return;
        };

        // Header & residue stepper
        ui.horizontal_wrapped(|ui| {
            let cur_pos = all_sids.iter().position(|&x| x == sid).unwrap_or(0);
            if ui.button("<").clicked() && cur_pos > 0 {
                let prev_sid = all_sids[cur_pos - 1];
                self.selected_seq_id = Some(prev_sid);
                if let Some(k) = seq_ids.iter().position(|&x| x == prev_sid) {
                    actions.push(RelaxPanelAction::SelectResidueIndex(k));
                }
            }
            if ui.button(">").clicked() && cur_pos + 1 < all_sids.len() {
                let next_sid = all_sids[cur_pos + 1];
                self.selected_seq_id = Some(next_sid);
                if let Some(k) = seq_ids.iter().position(|&x| x == next_sid) {
                    actions.push(RelaxPanelAction::SelectResidueIndex(k));
                }
            }

            let res_label = seq_ids
                .iter()
                .position(|&x| x == sid)
                .and_then(|k| sequence.get(k))
                .map(|s| s.as_str())
                .unwrap_or(&fit.res_name);
            ui.label(
                egui::RichText::new(format!(
                    "{res_label}{sid} -- {label} = {:.3} +- {:.3} s^-1 (I0 = {:.1}, chi^2/dof = {:.2}/{})",
                    fit.rate, fit.sigma_rate_mc, fit.i0, fit.chi2, fit.dof
                ))
                .size(12.5)
                .strong(),
            );

            if let Some(mf) = self
                .modelfree_result
                .as_ref()
                .and_then(|m| m.residues.iter().find(|r| r.seq_id == sid))
            {
                ui.colored_label(
                    egui::Color32::from_rgb(130, 235, 165),
                    egui::RichText::new(format!(
                        "[{}] S^2 = {:.3} +- {:.3}  R_ex = {:.2} s^-1",
                        mf.selected_model.as_str(),
                        mf.params.s2,
                        mf.sigma_s2,
                        mf.params.rex_ref_s_inv
                    ))
                    .size(11.0),
                );
            }
        });

        let desired = egui::vec2(ui.available_width().max(260.0), canvas_height);
        let (rect, resp) = ui.allocate_exact_size(desired, egui::Sense::hover());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 4.0, egui::Color32::from_rgb(16, 20, 28));
        painter.rect_stroke(
            rect,
            4.0,
            egui::Stroke::new(1.0, egui::Color32::from_rgb(44, 54, 72)),
            egui::StrokeKind::Inside,
        );

        let pad_l = 48.0_f32;
        let pad_r = 14.0_f32;
        let pad_t = 12.0_f32;
        let pad_b = 24.0_f32;
        let plot = egui::Rect::from_min_max(
            egui::pos2(rect.left() + pad_l, rect.top() + pad_t),
            egui::pos2(rect.right() - pad_r, rect.bottom() - pad_b),
        );
        if plot.width() <= 10.0 || plot.height() <= 10.0 {
            return;
        }

        let t_max = curve.times_sec.iter().copied().fold(0.05_f64, f64::max) * 1.06;
        let y_max = curve
            .intensities
            .iter()
            .zip(curve.sigmas.iter())
            .map(|(&y, &s)| y + s)
            .fold(fit.i0 + fit.offset_c, f64::max)
            .max(1.0)
            * 1.08;
        let y_min = curve
            .intensities
            .iter()
            .zip(curve.sigmas.iter())
            .map(|(&y, &s)| y - s)
            .fold(0.0_f64, f64::min)
            .min(0.0);

        let map_pt = |t: f64, y: f64| -> egui::Pos2 {
            let fx = (t / t_max).clamp(0.0, 1.0) as f32;
            let fy = ((y - y_min) / (y_max - y_min).max(1.0e-9)).clamp(0.0, 1.0) as f32;
            egui::pos2(
                plot.left() + fx * plot.width(),
                plot.bottom() - fy * plot.height(),
            )
        };

        // Subtle horizontal grid lines + zero baseline
        for g in 0..=4 {
            let frac = g as f64 / 4.0;
            let yv = y_min + frac * (y_max - y_min);
            let py = map_pt(0.0, yv).y;
            painter.line_segment(
                [egui::pos2(plot.left(), py), egui::pos2(plot.right(), py)],
                egui::Stroke::new(0.6, egui::Color32::from_rgb(30, 37, 50)),
            );
            if g == 0 || g == 2 || g == 4 {
                painter.text(
                    egui::pos2(plot.left() - 4.0, py),
                    egui::Align2::RIGHT_CENTER,
                    format!("{yv:.0}"),
                    egui::FontId::monospace(11.0),
                    egui::Color32::from_rgb(138, 150, 170),
                );
            }
        }
        for gx in 0..=4 {
            let tv = t_max * (gx as f64 / 4.0);
            let px = map_pt(tv, y_min).x;
            let align = if gx == 0 {
                egui::Align2::LEFT_CENTER
            } else if gx == 4 {
                egui::Align2::RIGHT_CENTER
            } else {
                egui::Align2::CENTER_CENTER
            };
            painter.text(
                egui::pos2(px, plot.bottom() + 11.0),
                align,
                format!("{tv:.2}s"),
                egui::FontId::monospace(11.0),
                egui::Color32::from_rgb(138, 150, 170),
            );
        }

        // Parametric MC +-1 sigma rate band + smooth fitted curve
        let n_seg = 64usize;
        let r_lo = (fit.rate - fit.sigma_rate_mc).max(0.0);
        let r_hi = fit.rate + fit.sigma_rate_mc;
        let mut fit_pts = Vec::with_capacity(n_seg + 1);
        let mut hi_pts = Vec::with_capacity(n_seg + 1);
        let mut lo_pts = Vec::with_capacity(n_seg + 1);
        for s in 0..=n_seg {
            let t = t_max * (s as f64 / n_seg as f64);
            let y_mid = fit.i0 * (-fit.rate * t).exp() + fit.offset_c;
            let y_top = (fit.i0 + fit.sigma_i0_mc) * (-r_lo * t).exp() + fit.offset_c;
            let y_bot = (fit.i0 - fit.sigma_i0_mc).max(0.0) * (-r_hi * t).exp() + fit.offset_c;
            fit_pts.push(map_pt(t, y_mid));
            hi_pts.push(map_pt(t, y_top));
            lo_pts.push(map_pt(t, y_bot));
        }
        painter.add(egui::Shape::line(
            hi_pts,
            egui::Stroke::new(1.0, accent.gamma_multiply(0.38)),
        ));
        painter.add(egui::Shape::line(
            lo_pts,
            egui::Stroke::new(1.0, accent.gamma_multiply(0.38)),
        ));
        painter.add(egui::Shape::line(fit_pts, egui::Stroke::new(2.0, accent)));

        // Experimental points + vertical +-1 sigma error bars
        for ((&t, &y), &sig) in curve
            .times_sec
            .iter()
            .zip(curve.intensities.iter())
            .zip(curve.sigmas.iter())
        {
            let p_mid = map_pt(t, y);
            let p_top = map_pt(t, y + sig);
            let p_bot = map_pt(t, y - sig);
            painter.line_segment(
                [p_top, p_bot],
                egui::Stroke::new(1.3, egui::Color32::from_rgb(220, 228, 242)),
            );
            painter.line_segment(
                [
                    egui::pos2(p_top.x - 3.0, p_top.y),
                    egui::pos2(p_top.x + 3.0, p_top.y),
                ],
                egui::Stroke::new(1.1, egui::Color32::from_rgb(220, 228, 242)),
            );
            painter.line_segment(
                [
                    egui::pos2(p_bot.x - 3.0, p_bot.y),
                    egui::pos2(p_bot.x + 3.0, p_bot.y),
                ],
                egui::Stroke::new(1.1, egui::Color32::from_rgb(220, 228, 242)),
            );
            painter.circle_filled(p_mid, 3.4, egui::Color32::WHITE);
            painter.circle_stroke(p_mid, 3.4, egui::Stroke::new(1.2, accent));
        }

        if let Some(hp) = resp.hover_pos() {
            if plot.contains(hp) {
                let t_hov = ((hp.x - plot.left()) / plot.width()) as f64 * t_max;
                let y_hat = fit.i0 * (-fit.rate * t_hov).exp() + fit.offset_c;
                resp.on_hover_text(format!("τ = {t_hov:.3} s\nÎ(τ) = {y_hat:.1}"));
            }
        }
    }

    fn draw_sequence_tracks_canvas(
        &mut self,
        ui: &mut egui::Ui,
        sequence: &[String],
        seq_ids: &[i32],
        track_h: f32,
        actions: &mut Vec<RelaxPanelAction>,
    ) {
        // Collect ordered sequence positions
        let ordered_ids: Vec<(usize, i32, String)> =
            if !seq_ids.is_empty() && seq_ids.len() == sequence.len() {
                seq_ids
                    .iter()
                    .enumerate()
                    .map(|(idx, &sid)| (idx, sid, sequence[idx].clone()))
                    .collect()
            } else {
                let mut sids: Vec<i32> = self
                    .r1_fits
                    .keys()
                    .chain(self.r2_fits.keys())
                    .chain(self.hetnoe_fits.keys())
                    .copied()
                    .collect();
                sids.sort_unstable();
                sids.dedup();
                sids.into_iter()
                    .enumerate()
                    .map(|(idx, sid)| (idx, sid, "RES".to_string()))
                    .collect()
            };
        if ordered_ids.is_empty() {
            return;
        }

        let mf_by_sid: BTreeMap<i32, _> = self
            .modelfree_result
            .as_ref()
            .map(|m| m.residues.iter().map(|r| (r.seq_id, r)).collect())
            .unwrap_or_default();

        let max_r1 = self
            .r1_fits
            .values()
            .map(|f| f.rate + f.sigma_rate_mc)
            .fold(2.5_f64, f64::max)
            * 1.08;
        let max_r2 = self
            .r2_fits
            .values()
            .map(|f| f.rate + f.sigma_rate_mc)
            .fold(12.0_f64, f64::max)
            * 1.08;
        let max_rex = mf_by_sid
            .values()
            .map(|r| r.params.rex_ref_s_inv + r.sigma_rex_ref)
            .fold(4.0_f64, f64::max)
            * 1.12;

        struct TrackSpec<'a> {
            title: &'a str,
            y_min: f64,
            y_max: f64,
            base_color: egui::Color32,
        }
        let specs = [
            TrackSpec {
                title: "R₁ (s⁻¹)",
                y_min: 0.0,
                y_max: max_r1,
                base_color: egui::Color32::from_rgb(70, 178, 250),
            },
            TrackSpec {
                title: "R₂ (s⁻¹)",
                y_min: 0.0,
                y_max: max_r2,
                base_color: egui::Color32::from_rgb(248, 158, 66),
            },
            TrackSpec {
                title: "hetNOE",
                y_min: -0.2,
                y_max: 1.05,
                base_color: egui::Color32::from_rgb(176, 128, 248),
            },
            TrackSpec {
                title: "S² (M1–M5)",
                y_min: 0.0,
                y_max: 1.0,
                base_color: egui::Color32::from_rgb(84, 214, 148),
            },
            TrackSpec {
                title: "R_ex (s⁻¹)",
                y_min: 0.0,
                y_max: max_rex,
                base_color: egui::Color32::from_rgb(244, 92, 108),
            },
        ];

        let n_cols = ordered_ids.len();
        let pad_l = 78.0_f32;
        let pad_r = 10.0_f32;

        for (t_idx, spec) in specs.iter().enumerate() {
            let desired = egui::vec2(ui.available_width().max(260.0), track_h);
            let (rect, resp) = ui.allocate_exact_size(desired, egui::Sense::click_and_drag());
            let painter = ui.painter_at(rect);
            painter.rect_filled(rect, 3.0, egui::Color32::from_rgb(18, 23, 32));
            painter.rect_stroke(
                rect,
                3.0,
                egui::Stroke::new(1.0, egui::Color32::from_rgb(38, 48, 64)),
                egui::StrokeKind::Inside,
            );

            painter.text(
                egui::pos2(rect.left() + 6.0, rect.top() + 12.0),
                egui::Align2::LEFT_CENTER,
                spec.title,
                egui::FontId::proportional(11.0),
                egui::Color32::from_rgb(210, 220, 235),
            );
            painter.text(
                egui::pos2(rect.left() + 6.0, rect.bottom() - 11.0),
                egui::Align2::LEFT_CENTER,
                format!("{:.1}..{:.1}", spec.y_min, spec.y_max),
                egui::FontId::monospace(11.0),
                egui::Color32::from_rgb(126, 140, 160),
            );

            let bar_area = egui::Rect::from_min_max(
                egui::pos2(rect.left() + pad_l, rect.top() + 4.0),
                egui::pos2(rect.right() - pad_r, rect.bottom() - 4.0),
            );
            if bar_area.width() <= 8.0 {
                continue;
            }
            let col_w = bar_area.width() / (n_cols as f32);
            let zero_frac =
                ((0.0 - spec.y_min) / (spec.y_max - spec.y_min).max(1.0e-9)).clamp(0.0, 1.0) as f32;
            let zero_y = bar_area.bottom() - zero_frac * bar_area.height();
            painter.line_segment(
                [
                    egui::pos2(bar_area.left(), zero_y),
                    egui::pos2(bar_area.right(), zero_y),
                ],
                egui::Stroke::new(0.7, egui::Color32::from_rgb(48, 58, 76)),
            );

            for (c_idx, &(seq_idx, sid, ref _rname)) in ordered_ids.iter().enumerate() {
                let x0 = bar_area.left() + (c_idx as f32) * col_w;
                let x1 = x0 + (col_w - 1.0).max(1.2);
                let xc = 0.5 * (x0 + x1);

                if self.selected_seq_id == Some(sid) {
                    painter.rect_filled(
                        egui::Rect::from_min_max(
                            egui::pos2(x0, bar_area.top()),
                            egui::pos2(x0 + col_w, bar_area.bottom()),
                        ),
                        0.0,
                        egui::Color32::from_rgba_unmultiplied(255, 255, 255, 22),
                    );
                }

                let val_sig_col: Option<(f64, f64, egui::Color32)> = match t_idx {
                    0 => self
                        .r1_fits
                        .get(&sid)
                        .map(|f| (f.rate, f.sigma_rate_mc, spec.base_color)),
                    1 => self
                        .r2_fits
                        .get(&sid)
                        .map(|f| (f.rate, f.sigma_rate_mc, spec.base_color)),
                    2 => self
                        .hetnoe_fits
                        .get(&sid)
                        .map(|n| (n.noe, n.sigma_noe, spec.base_color)),
                    3 => mf_by_sid.get(&sid).map(|r| {
                        let col = match r.selected_model {
                            ModelFreeId::M1 => egui::Color32::from_rgb(76, 206, 146),
                            ModelFreeId::M2 => egui::Color32::from_rgb(82, 182, 248),
                            ModelFreeId::M3 => egui::Color32::from_rgb(248, 168, 68),
                            ModelFreeId::M4 => egui::Color32::from_rgb(244, 102, 114),
                            ModelFreeId::M5 => egui::Color32::from_rgb(196, 122, 250),
                        };
                        (r.params.s2, r.sigma_s2, col)
                    }),
                    _ => mf_by_sid
                        .get(&sid)
                        .map(|r| (r.params.rex_ref_s_inv, r.sigma_rex_ref, spec.base_color)),
                };

                if let Some((val, sig, col)) = val_sig_col {
                    let f_val = ((val - spec.y_min) / (spec.y_max - spec.y_min).max(1.0e-9))
                        .clamp(0.0, 1.0) as f32;
                    let y_val = bar_area.bottom() - f_val * bar_area.height();
                    let top = y_val.min(zero_y);
                    let bot = y_val.max(zero_y).max(top + 1.0);
                    painter.rect_filled(
                        egui::Rect::from_min_max(egui::pos2(x0, top), egui::pos2(x1, bot)),
                        1.0,
                        col,
                    );
                    if sig > 0.0 {
                        let f_hi = ((val + sig - spec.y_min)
                            / (spec.y_max - spec.y_min).max(1.0e-9))
                        .clamp(0.0, 1.0) as f32;
                        let f_lo = ((val - sig - spec.y_min)
                            / (spec.y_max - spec.y_min).max(1.0e-9))
                        .clamp(0.0, 1.0) as f32;
                        let y_hi = bar_area.bottom() - f_hi * bar_area.height();
                        let y_lo = bar_area.bottom() - f_lo * bar_area.height();
                        painter.line_segment(
                            [egui::pos2(xc, y_hi), egui::pos2(xc, y_lo)],
                            egui::Stroke::new(1.0, egui::Color32::from_rgb(230, 236, 245)),
                        );
                    }
                }
                let _ = seq_idx;
            }

            if let Some(pos) = resp.interact_pointer_pos() {
                if resp.clicked() && bar_area.contains(pos) {
                    let c_idx = (((pos.x - bar_area.left()) / col_w).floor() as usize)
                        .min(n_cols.saturating_sub(1));
                    let (seq_idx, sid, _) = ordered_ids[c_idx];
                    self.selected_seq_id = Some(sid);
                    actions.push(RelaxPanelAction::SelectResidueIndex(seq_idx));
                }
            }
            if let Some(hp) = resp.hover_pos() {
                if bar_area.contains(hp) {
                    let c_idx = (((hp.x - bar_area.left()) / col_w).floor() as usize)
                        .min(n_cols.saturating_sub(1));
                    let (_seq_idx, sid, ref rname) = ordered_ids[c_idx];
                    let mut tip = format!("{rname}{sid}");
                    if let Some(f1) = self.r1_fits.get(&sid) {
                        tip.push_str(&format!(
                            "\nR₁ = {:.3} ± {:.3} s⁻¹",
                            f1.rate, f1.sigma_rate_mc
                        ));
                    }
                    if let Some(f2) = self.r2_fits.get(&sid) {
                        tip.push_str(&format!(
                            "\nR₂ = {:.3} ± {:.3} s⁻¹",
                            f2.rate, f2.sigma_rate_mc
                        ));
                    }
                    if let Some(hn) = self.hetnoe_fits.get(&sid) {
                        tip.push_str(&format!("\nhetNOE = {:.3} ± {:.3}", hn.noe, hn.sigma_noe));
                    }
                    if let Some(mf) = mf_by_sid.get(&sid) {
                        tip.push_str(&format!(
                            "\n{} S² = {:.3} ± {:.3}, R_ex = {:.2} s⁻¹",
                            mf.selected_model.as_str(),
                            mf.params.s2,
                            mf.sigma_s2,
                            mf.params.rex_ref_s_inv
                        ));
                    }
                    resp.on_hover_text(tip);
                }
            }
            ui.add_space(3.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nmr_core::relax_io::{write_relax_series_nef_str, write_relax_series_tsv_str};

    #[test]
    fn demo_dynamics_fits_curves_modelfree_and_exports_3d_tracks() {
        let mut panel = RelaxPanelState::new();
        panel.mc_replicates = 100;
        let seq: Vec<String> = [
            "MET", "GLN", "ILE", "PHE", "VAL", "LYS", "THR", "LEU", "THR", "GLY", "LYS", "THR",
            "ILE", "THR", "LEU", "GLU", "VAL", "GLU", "PRO", "SER",
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
        let seq_ids: Vec<i32> = (1..=20).collect();

        let msg = panel
            .load_demo_dynamics(&seq, &seq_ids, None)
            .expect("demo dynamics must succeed");
        assert!(msg.contains("Converged Model-Free"));
        // 19 non-proline residues
        assert_eq!(panel.r1_fits.len(), 19);
        assert_eq!(panel.r2_fits.len(), 19);
        assert_eq!(panel.hetnoe_fits.len(), 19);
        let mf = panel.modelfree_result.as_ref().expect("modelfree result");
        assert_eq!(mf.residues.len(), 19);
        assert!(panel.exported_tsv_preview.contains("nmrassign_modelfree"));

        // Verify 1-click export to 3D PDB QuantitativeTrack (S^2 and R_ex)
        let export_s2 = panel
            .build_3d_track_export(&seq_ids, QuantitativeTrack::LipariSzaboS2)
            .expect("S^2 export");
        match export_s2 {
            RelaxPanelAction::ExportTo3dTrack {
                track,
                s2_by_seq_idx,
                rex_by_seq_idx,
                rex_max_s_inv,
            } => {
                assert_eq!(track, QuantitativeTrack::LipariSzaboS2);
                assert_eq!(s2_by_seq_idx.len(), 19);
                assert!(!s2_by_seq_idx.contains_key(&18)); // PRO19 (`idx = 18`) skipped
                assert!(!rex_by_seq_idx.is_empty());
                assert!(rex_max_s_inv >= 5.0);
            }
            other => panic!("unexpected action: {other:?}"),
        }
    }

    #[test]
    fn import_series_tsv_and_nef_roundtrip_in_panel() {
        let fp = LockedPeakFootprint {
            seq_id: 7,
            res_name: "THR".to_string(),
            atom_h: "H".to_string(),
            atom_x: "N".to_string(),
            delta_h_ppm: 8.45,
            delta_x_ppm: 115.2,
            sigma_h_ppm: 0.022,
            sigma_x_ppm: 0.20,
            eta_h: 0.0,
            eta_x: 0.0,
            overlap_cluster_id: 0,
        };
        let delays = vec![0.02, 0.10, 0.25, 0.50, 0.90];
        let true_r1 = 1.60_f64;
        let intensities: Vec<f64> = delays
            .iter()
            .map(|&t| 800.0 * (-true_r1 * t).exp())
            .collect();
        let ds = RelaxSeriesDataset {
            name: "unit_r1".to_string(),
            experiment_kind: RelaxExperimentKind::R1,
            field_mhz: 700.0,
            schedule: RelaxSchedule::VariableDelaysSec {
                delays_sec: delays.clone(),
            },
            plane_noise_mad: vec![2.0; delays.len()],
            plane_is_duplicate: vec![false; delays.len()],
            duplicate_variance_scale: 1.0,
            duplicate_dof: 0,
            traces: vec![ResidueRelaxTrace {
                footprint: fp,
                volumes: intensities.clone(),
                intensities,
                sigmas: vec![2.0; delays.len()],
            }],
        };

        let mut panel = RelaxPanelState::new();
        panel.mc_replicates = 120;
        let tsv = write_relax_series_tsv_str(&ds);
        panel.import_series_text(&tsv).expect("import TSV");
        assert!((panel.r1_fits[&7].rate - true_r1).abs() < 1.0e-4);

        let mut ds_r2 = ds;
        ds_r2.experiment_kind = RelaxExperimentKind::R2;
        let nef = write_relax_series_nef_str(&ds_r2);
        panel.import_series_text(&nef).expect("import NEF");
        assert!((panel.r2_fits[&7].rate - true_r1).abs() < 1.0e-4);
    }

    #[test]
    fn headless_egui_renders_empty_and_populated_relax_panel() {
        let ctx = egui::Context::default();
        let mut panel = RelaxPanelState::new();
        panel.mc_replicates = 60;
        let seq: Vec<String> = ["MET", "GLN", "ILE", "PHE", "VAL", "LYS"]
            .iter()
            .map(|s| (*s).to_string())
            .collect();
        let seq_ids: Vec<i32> = (1..=6).collect();

        // 1. Empty state frame
        let mut out1 = ctx.run_ui(egui::RawInput::default(), |ui| {
            let acts = panel.ui(ui, &seq, &seq_ids, Some(0), None);
            assert!(acts.is_empty());
        });
        out1.textures_delta.clear();

        // 2. Populated state frame (`I(tau)` canvas + 5 sequence bar tracks)
        panel.load_demo_dynamics(&seq, &seq_ids, None).unwrap();
        let mut out2 = ctx.run_ui(egui::RawInput::default(), |ui| {
            let _acts = panel.ui(ui, &seq, &seq_ids, Some(2), None);
        });
        out2.textures_delta.clear();
        assert_eq!(panel.selected_seq_id, Some(3));
    }
}
