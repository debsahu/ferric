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
use nmr_core::ccr_relax::{invert_tract_eta_xy_to_tau_m, CcrPhysicsConfig, TractTauMEstimate};
use nmr_core::dynamics_detectors::{
    build_dynamics_detectors, experiments_and_observations_from_rates,
    fit_impact_from_multi_field_rates, log_spaced_grid, select_impact_model_aicc_from_rates,
    DetectorConfig, DetectorDesign, DetectorFitResult, ImpactConfig, ImpactFitResult,
};
use nmr_core::modelfree::{
    extract_nh_unit_vectors_from_chain, fit_modelfree_dataset, group_multifield_rates,
    predict_r1_r2_noe, write_modelfree_tsv_str, DiffusionTensor, DiffusionTensorKind,
    LocalMfParams, ModelFreeConfig, ModelFreeDatasetResult, ModelFreeId, ModelSelectionCriterion,
};
use nmr_core::noesy::parse_pdb_str_first_model;
use nmr_core::peaks::Peak;
use nmr_core::priors::one_to_three;
use nmr_core::relax_fit::{
    compute_hetnoe_batch, decay_inputs_from_series, fit_decay_batch_cpu, DecayCurveFit,
    DecayCurveInput, ExpFitConfig, HetNoeInput, HetNoeResult, OffsetModelGate, RelaxPhysicsConfig,
    ResidueRelaxRates,
};
use nmr_core::relax_io::{
    parse_bruker_delay_token_sec, parse_relax_series_nef_str, parse_relax_series_tsv_str,
    HetNoeOrder, LockedPeakFootprint, RelaxExperimentKind, RelaxSchedule, RelaxSeriesDataset,
    ResidueRelaxTrace, TargetPeakSeed,
};
use nmr_core::rng::Rng;
use nmr_core::titration_2d::Peak2d as TitrationPeak2d;

use crate::mol3d_overlay::QuantitativeTrack;
use crate::workbench_mode::{TYPE_BODY_PX, TYPE_CAPTION_PX};

/// 2D HSQC cross-peak passed from `Assign` mode into [`RelaxPanelState::sync_assigned_hsqc_peaks`]
/// to seed the locked peak footprint reference for pseudo-3D relaxation integration.
#[derive(Clone, Debug, PartialEq)]
pub struct Peak2D {
    /// Direct `1H` chemical shift (ppm).
    pub h_ppm: f64,
    /// Indirect `15N` chemical shift (ppm).
    pub n_ppm: f64,
    /// Optional 1-based sequence residue ID (e.g. `7` for `T7`).
    pub seq_id: Option<i32>,
    /// Optional 3-letter amino-acid code (e.g. `"THR"`).
    pub res_name: Option<String>,
    /// Optional human-readable label (e.g. `"T7"` or `"THR7"`).
    pub label: Option<String>,
    /// Optional peak height/intensity.
    pub intensity: Option<f64>,
}

impl Peak2D {
    pub fn new(h_ppm: f64, n_ppm: f64) -> Self {
        Self {
            h_ppm,
            n_ppm,
            seq_id: None,
            res_name: None,
            label: None,
            intensity: None,
        }
    }

    pub fn assigned(seq_id: i32, res_name: impl Into<String>, h_ppm: f64, n_ppm: f64) -> Self {
        Self {
            h_ppm,
            n_ppm,
            seq_id: Some(seq_id),
            res_name: Some(res_name.into()),
            label: None,
            intensity: None,
        }
    }

    /// Convert a core HSQC [`Peak`] plus optional solver assignment into a [`Peak2D`] footprint seed.
    pub fn from_core_peak(peak: &Peak, seq_id: Option<i32>, res_name: Option<&str>) -> Self {
        Self {
            h_ppm: peak.h,
            n_ppm: peak.n,
            seq_id,
            res_name: res_name.map(String::from),
            label: None,
            intensity: peak.height,
        }
    }
}

impl From<&TitrationPeak2d> for Peak2D {
    fn from(p: &TitrationPeak2d) -> Self {
        Self {
            h_ppm: p.h_ppm,
            n_ppm: p.n_ppm,
            seq_id: None,
            res_name: p.res_name.clone(),
            label: p.label.clone(),
            intensity: p.intensity,
        }
    }
}

/// Active sub-tab inside the `Relaxation / S²` studio body.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum RelaxStudioSubTab {
    #[default]
    CurvesAndModelFree,
    Pseudo3dImport,
    DetectorsImpactTract,
}

impl RelaxStudioSubTab {
    pub const ALL: [Self; 3] = [
        Self::CurvesAndModelFree,
        Self::Pseudo3dImport,
        Self::DetectorsImpactTract,
    ];

    pub const fn label(self) -> &'static str {
        match self {
            Self::CurvesAndModelFree => "Curves & Model-Free",
            Self::Pseudo3dImport => "Pseudo-3D T1/T2/hetNOE",
            Self::DetectorsImpactTract => "Detectors / IMPACT / TRACT",
        }
    }
}

/// UI state for one pseudo-3D relaxation folder slot (`T1`, `T2`, or `hetNOE`).
#[derive(Clone, Debug, Default)]
pub struct Pseudo3dSlotState {
    pub dir_path: String,
    pub delays_text: String,
    pub loaded_planes: usize,
    pub fitted_residues: usize,
    pub warnings: Vec<String>,
}

/// Fail-closed error message when TRACT is requested without experimental TROSY/anti-TROSY rates.
pub const TRACT_MISSING_RATES_ERR: &str =
    "TRACT requires experimental TROSY (Rα) and anti-TROSY (Rβ) decay rates";

/// Compact Phase 22 `TRACT` / CCR ($\eta_{xy}$, $R_{\text{ex}}$-free $\tau_m$, $\kappa_0$) summary.
#[derive(Clone, Debug, PartialEq)]
pub struct TractSummaryReport {
    pub field_mhz: f64,
    pub r_alpha_s_inv: f64,
    pub r_beta_s_inv: f64,
    pub eta_xy_s_inv: f64,
    pub sigma_eta_xy_s_inv: f64,
    pub tau_m_rigid: TractTauMEstimate,
    pub tau_m_s2_corrected: Option<TractTauMEstimate>,
    pub kappa0_sec: f64,
}

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
    pub active_sub_tab: RelaxStudioSubTab,
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

    /// Locked HSQC peak footprints synced from `Assign` mode for pseudo-3D plane integration.
    pub synced_peak_seeds: Vec<TargetPeakSeed>,
    /// Synthetic pick IDs created by unassigned plane-0 auto-picking (never exported onto 3D PDB residues).
    pub unassigned_pick_ids: std::collections::BTreeSet<i32>,
    pub t1_slot: Pseudo3dSlotState,
    pub t2_slot: Pseudo3dSlotState,
    pub hetnoe_slot: Pseudo3dSlotState,
    pub hetnoe_order: HetNoeOrder,
    /// Optional unit CPMG cycle time in milliseconds when `t2_slot.delays_text` holds integer `vclist` loop counts.
    pub t2_vclist_unit_ms: f64,

    /// Phase 22 (`T22.3`) linear optimal-window dynamics detectors design & per-residue responses.
    pub detector_design: Option<DetectorDesign>,
    pub detector_fits: BTreeMap<i32, DetectorFitResult>,
    /// Phase 22 (`T22.4`) `IMPACT` simplex correlation-time decompositions per residue.
    pub impact_fits: BTreeMap<i32, ImpactFitResult>,
    /// Phase 22 (`T22.5`) `TRACT` experimental TROSY ($\alpha$) and anti-TROSY ($\beta$) decay rates
    /// and uncertainties (`None` until explicitly supplied by the user — never placeholder values).
    pub tract_r_alpha_input: Option<f64>,
    pub tract_sigma_alpha_input: Option<f64>,
    pub tract_r_beta_input: Option<f64>,
    pub tract_sigma_beta_input: Option<f64>,
    pub tract_r_alpha_text: String,
    pub tract_sigma_alpha_text: String,
    pub tract_r_beta_text: String,
    pub tract_sigma_beta_text: String,
    pub tract_assumed_s2: f64,
    pub tract_report: Option<TractSummaryReport>,

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
            active_sub_tab: RelaxStudioSubTab::CurvesAndModelFree,
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
            synced_peak_seeds: Vec::new(),
            unassigned_pick_ids: std::collections::BTreeSet::new(),
            t1_slot: Pseudo3dSlotState::default(),
            t2_slot: Pseudo3dSlotState::default(),
            hetnoe_slot: Pseudo3dSlotState::default(),
            hetnoe_order: HetNoeOrder::Auto,
            t2_vclist_unit_ms: 0.0,
            detector_design: None,
            detector_fits: BTreeMap::new(),
            impact_fits: BTreeMap::new(),
            tract_r_alpha_input: None,
            tract_sigma_alpha_input: None,
            tract_r_beta_input: None,
            tract_sigma_beta_input: None,
            tract_r_alpha_text: String::new(),
            tract_sigma_alpha_text: String::new(),
            tract_r_beta_text: String::new(),
            tract_sigma_beta_text: String::new(),
            tract_assumed_s2: 0.85,
            tract_report: None,
            selected_seq_id: None,
            show_paste_drawer: false,
            paste_series_buf: String::new(),
            exported_tsv_preview: String::new(),
            show_tsv_modal: false,
            status_banner:
                "Load a pseudo-3D Bruker T1/T2/hetNOE folder, TSV/NEF, or synthesize dual-field dynamics."
                    .to_string(),
            last_canvas_rect: None,
        }
    }

    pub fn has_fitted_rates(&self) -> bool {
        !self.r1_fits.is_empty() || !self.r2_fits.is_empty() || !self.hetnoe_fits.is_empty()
    }

    /// Number of locked HSQC cross-peak footprints currently synced from `Assign` mode.
    pub fn synced_peak_count(&self) -> usize {
        self.synced_peak_seeds.len()
    }

    /// Auto-populate the locked peak footprint reference (`synced_peak_seeds`) directly from
    /// `Assign` mode's HSQC peaks and protein sequence (accepting 1-letter FASTA/raw strings or
    /// whitespace/comma-separated 3-letter residue codes).
    pub fn sync_assigned_hsqc_peaks(&mut self, peaks: &[Peak2D], sequence: &str) -> usize {
        let seq_3letter = parse_sequence_3letter_tokens(sequence);
        let mut seeds = Vec::with_capacity(peaks.len());
        let mut used_ids = std::collections::BTreeSet::new();
        let mut skipped_unassigned = 0usize;
        let mut skipped_duplicates = 0usize;

        for peak in peaks {
            if !peak.h_ppm.is_finite() || !peak.n_ppm.is_finite() {
                continue;
            }
            let parsed_label_id = peak.label.as_deref().and_then(extract_seq_id_from_label);
            let Some(seq_id) = peak.seq_id.or(parsed_label_id) else {
                skipped_unassigned += 1;
                continue;
            };
            if !used_ids.insert(seq_id) {
                skipped_duplicates += 1;
                continue;
            }

            let res_name = peak
                .res_name
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(normalize_res_name_3letter)
                .or_else(|| {
                    if seq_id >= 1 {
                        seq_3letter.get((seq_id - 1) as usize).cloned()
                    } else {
                        None
                    }
                })
                .unwrap_or_else(|| "RES".to_string());

            seeds.push(TargetPeakSeed {
                seq_id,
                res_name,
                atom_h: "H".to_string(),
                atom_x: "N".to_string(),
                delta_h_ppm: peak.h_ppm,
                delta_x_ppm: peak.n_ppm,
            });
        }

        let n = seeds.len();
        self.synced_peak_seeds = seeds;
        self.unassigned_pick_ids.clear();
        if skipped_duplicates > 0 || skipped_unassigned > 0 {
            self.status_banner = format!(
                "Synced {n} assigned HSQC peak footprints (dropped {skipped_duplicates} duplicate seq_id(s), {skipped_unassigned} unassigned peak(s))."
            );
        } else {
            self.status_banner = format!(
                "Synced {n} locked HSQC peak footprints from Assign mode for pseudo-3D integration."
            );
        }
        n
    }

    /// Parse an optional comma/whitespace/newline-separated `vdlist` or `vclist` delay override
    /// string into seconds. Returns `Ok(None)` when `raw` is blank so Bruker's on-disk schedule is used.
    pub fn parse_delays_override_str(
        raw: &str,
        vclist_unit_ms: Option<f64>,
    ) -> Result<Option<Vec<f64>>, String> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Ok(None);
        }
        let tokens: Vec<&str> = trimmed
            .split(|c: char| c.is_ascii_whitespace() || c == ',' || c == ';')
            .filter(|t| !t.is_empty())
            .collect();
        if tokens.is_empty() {
            return Ok(None);
        }
        let mut out = Vec::with_capacity(tokens.len());
        let all_plain_ints = tokens
            .iter()
            .all(|t| !t.is_empty() && t.chars().all(|c| c.is_ascii_digit()));
        if all_plain_ints {
            let unit_ms = match vclist_unit_ms {
                Some(u) if u.is_finite() && u > 0.0 => u,
                _ => {
                    return Err(
                        "Plain integer vclist loop counts require an explicit positive CPMG cycle duration (cpmg_cycle_ms > 0.0 ms); set CPMG cycle duration or use explicit units ('s'/'ms') / decimal seconds."
                            .to_string(),
                    );
                }
            };
            let u = unit_ms * 1.0e-3;
            for tok in tokens {
                let n: u32 = tok
                    .parse()
                    .map_err(|e| format!("invalid vclist count '{tok}': {e}"))?;
                out.push((n as f64) * u);
            }
        } else {
            for tok in tokens {
                let sec = parse_bruker_delay_token_sec(tok)?;
                out.push(sec);
            }
        }
        Ok(Some(out))
    }

    /// Load a native Bruker pseudo-3D relaxation folder (`T1`/`R1`, `T2`/`R2`/`R1rho`, or
    /// interleaved 2-plane `hetNOE`) via `nmr_core::relax_io`, process all planes with a single
    /// locked recipe, integrate locked HSQC footprints, and fit decay curves / hetNOE immediately.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_bruker_pseudo3d_series(
        &mut self,
        kind: RelaxExperimentKind,
        dir_path: impl AsRef<std::path::Path>,
        hetnoe_order: HetNoeOrder,
        delays_override: Option<&[f64]>,
    ) -> Result<String, String> {
        if !matches!(
            kind,
            RelaxExperimentKind::R1
                | RelaxExperimentKind::R2
                | RelaxExperimentKind::R1rho
                | RelaxExperimentKind::HetNoe
        ) {
            return Err(format!(
                "unsupported relaxation slot kind '{}' (expected R1, R2, R1rho, or hetNOE)",
                kind.as_str()
            ));
        }

        let path = dir_path.as_ref();
        if !path.exists() {
            return Err(format!(
                "Bruker pseudo-3D folder does not exist: {}",
                path.display()
            ));
        }

        let schedule_override = match delays_override {
            Some(vals) if !vals.is_empty() => {
                for (idx, &v) in vals.iter().enumerate() {
                    if !v.is_finite() || v < 0.0 {
                        return Err(format!(
                            "delays_override[{idx}] = {v} must be finite and >= 0 s"
                        ));
                    }
                }
                Some(RelaxSchedule::VariableDelaysSec {
                    delays_sec: vals.to_vec(),
                })
            }
            _ => None,
        };

        let unit_cycle_sec = if matches!(kind, RelaxExperimentKind::R2 | RelaxExperimentKind::R1rho)
            && self.t2_vclist_unit_ms.is_finite()
            && self.t2_vclist_unit_ms > 0.0
        {
            Some(self.t2_vclist_unit_ms * 1.0e-3)
        } else {
            None
        };

        let raw = nmr_core::relax_io::read_bruker_pseudo3d_with_hetnoe_order(
            path,
            None,
            unit_cycle_sec,
            schedule_override,
            hetnoe_order,
        )?;
        let n_planes = raw.planes.len();
        let mut warnings = raw.warnings.clone();
        let (planes, _recipe) = nmr_core::relax_io::process_pseudo3d_fids(&raw.planes, None)?;

        let seeds: Vec<TargetPeakSeed> = if !self.synced_peak_seeds.is_empty() {
            self.unassigned_pick_ids.clear();
            self.synced_peak_seeds.clone()
        } else {
            let Some(ref_plane) = planes.first() else {
                return Err("pseudo-3D folder yielded zero processed 2D planes".to_string());
            };
            let picked = nmr_core::picker::pick_hsqc_2d(
                ref_plane,
                &nmr_core::picker::PickerConfig::default(),
            );
            if picked.is_empty() {
                return Err(
                    "No HSQC peaks synced and auto-pick on plane 0 found 0 peaks. Sync HSQC peaks from Assign mode first."
                        .to_string(),
                );
            }
            warnings.push(format!(
                "No assigned HSQC peaks synced: integrated {} unassigned plane-0 picks (blocked from 3D structure export until assigned).",
                picked.len()
            ));
            let mut auto_seeds = Vec::with_capacity(picked.len());
            for (idx, pk) in picked.iter().enumerate() {
                let synthetic_id = (idx as i32) + 1;
                self.unassigned_pick_ids.insert(synthetic_id);
                auto_seeds.push(TargetPeakSeed {
                    seq_id: synthetic_id,
                    res_name: "PK".to_string(),
                    atom_h: "H".to_string(),
                    atom_x: "N".to_string(),
                    delta_h_ppm: pk.h,
                    delta_x_ppm: pk.n,
                });
            }
            auto_seeds
        };

        let ds = nmr_core::relax_io::integrate_relax_planes(
            raw.name,
            kind,
            raw.schedule,
            &planes,
            &seeds,
            &nmr_core::relax_io::RelaxIntegrationConfig::default(),
        )?;
        let msg = self.ingest_series_dataset(&ds)?;

        let slot = match kind {
            RelaxExperimentKind::R1 => &mut self.t1_slot,
            RelaxExperimentKind::R2 | RelaxExperimentKind::R1rho => &mut self.t2_slot,
            RelaxExperimentKind::HetNoe => &mut self.hetnoe_slot,
            _ => &mut self.t1_slot,
        };
        slot.dir_path = path.display().to_string();
        slot.loaded_planes = n_planes;
        slot.fitted_residues = ds.traces.len();
        slot.warnings = warnings;
        Ok(msg)
    }

    /// Collect all multi-field `ResidueRelaxRates` entries for a given `seq_id` (primary + auxiliary fields).
    fn multifield_rates_for_seq_id(&self, seq_id: i32) -> Vec<ResidueRelaxRates> {
        let mut out = Vec::new();
        if let (Some(r1), Some(r2)) = (self.r1_fits.get(&seq_id), self.r2_fits.get(&seq_id)) {
            let (hetnoe, sigma_hetnoe) = match self.hetnoe_fits.get(&seq_id) {
                Some(n) => (Some(n.noe), Some(n.sigma_noe.max(1.0e-4))),
                None => (None, None),
            };
            out.push(ResidueRelaxRates {
                seq_id,
                res_name: r1.res_name.clone(),
                field_mhz: self.primary_field_mhz,
                r1: r1.rate,
                sigma_r1: r1.sigma_rate_mc.max(1.0e-4),
                r2: r2.rate,
                sigma_r2: r2.sigma_rate_mc.max(1.0e-4),
                hetnoe,
                sigma_hetnoe,
            });
        }
        for aux in &self.aux_field_rates {
            if let Some(entry) = aux.iter().find(|r| r.seq_id == seq_id) {
                out.push(entry.clone());
            }
        }
        out
    }

    /// Run Phase 22 (`T22.3`) linear optimal-window dynamics detectors on the currently fitted
    /// `R1`/`R2`/`hetNOE` rates (including auxiliary fields when loaded).
    pub fn run_phase22_detectors(&mut self) -> Result<String, String> {
        let primary = self.primary_rate_table();
        if primary.is_empty() {
            return Err(
                "Fit R1 and R2 rates before running Dynamics Detectors (T22.3)".to_string(),
            );
        }
        let phys = RelaxPhysicsConfig::default();
        let mut first_design: Option<DetectorDesign> = None;
        let mut fits = BTreeMap::new();

        for row in &primary {
            let mf_rates = self.multifield_rates_for_seq_id(row.seq_id);
            let Ok((exps, obs, cov)) = experiments_and_observations_from_rates(&mf_rates, &phys)
            else {
                continue;
            };
            let design_for_res = match &first_design {
                Some(d) if d.experiments.len() == exps.len() => d.clone(),
                _ => {
                    let cfg = DetectorConfig {
                        physics: phys,
                        grid_points: 120,
                        max_detectors: exps.len().clamp(1, 5),
                        ..DetectorConfig::default()
                    };
                    let d = build_dynamics_detectors(&exps, &cfg)?;
                    if first_design.is_none() {
                        first_design = Some(d.clone());
                    }
                    d
                }
            };
            if let Ok(fit) = design_for_res.estimate(&obs, Some(&cov)) {
                fits.insert(row.seq_id, fit);
            }
        }

        let design = first_design
            .ok_or_else(|| "Unable to build Dynamics Detectors from fitted rates".to_string())?;
        let k = design.num_detectors;
        let n_res = fits.len();
        self.detector_design = Some(design);
        self.detector_fits = fits;
        let msg = format!("Computed {k} Dynamics Detectors (T22.3) across {n_res} residues.");
        self.status_banner.clone_from(&msg);
        Ok(msg)
    }

    /// Run Phase 22 (`T22.4`) `IMPACT` simplex correlation-time decomposition directly on the
    /// fitted `R1`/`R2`/`hetNOE` rates.
    pub fn run_phase22_impact(&mut self) -> Result<String, String> {
        let primary = self.primary_rate_table();
        if primary.is_empty() {
            return Err("Fit R1, R2, and hetNOE rates before running IMPACT (T22.4)".to_string());
        }
        let phys = RelaxPhysicsConfig::default();
        let cfg = ImpactConfig {
            mc_replicates: 40,
            ..ImpactConfig::default()
        };
        let fallback_grid = log_spaced_grid(cfg.tau_min_sec, cfg.tau_max_sec, 3)?;
        let mut out = BTreeMap::new();

        for row in &primary {
            let mf_rates: Vec<ResidueRelaxRates> = self
                .multifield_rates_for_seq_id(row.seq_id)
                .into_iter()
                .filter(|r| r.hetnoe.is_some())
                .collect();
            if mf_rates.is_empty() {
                continue;
            }
            let fit_res = if mf_rates.len() >= 2 {
                select_impact_model_aicc_from_rates(&mf_rates, &[3, 4, 5], &phys, &cfg)
                    .map(|sel| sel.best_fit)
                    .or_else(|_| {
                        fit_impact_from_multi_field_rates(&mf_rates, &fallback_grid, &phys, &cfg)
                    })
            } else {
                fit_impact_from_multi_field_rates(&mf_rates, &fallback_grid, &phys, &cfg)
            };
            if let Ok(fit) = fit_res {
                out.insert(row.seq_id, fit);
            }
        }

        if out.is_empty() {
            return Err(
                "IMPACT (T22.4) requires R1, R2, and hetNOE rates for spectral density stacking"
                    .to_string(),
            );
        }
        let n_res = out.len();
        self.impact_fits = out;
        let msg = format!("Converged IMPACT (T22.4) simplex decomposition for {n_res} residues.");
        self.status_banner.clone_from(&msg);
        Ok(msg)
    }

    /// Record explicit experimental TROSY (`Rα ± σ_α`) and anti-TROSY (`Rβ ± σ_β`) decay rates
    /// (s⁻¹) for Phase 22 (`T22.5`) `TRACT` analysis.
    pub fn set_tract_experimental_rates(
        &mut self,
        r_alpha_s_inv: f64,
        sigma_r_alpha: f64,
        r_beta_s_inv: f64,
        sigma_r_beta: f64,
    ) {
        self.tract_r_alpha_input = Some(r_alpha_s_inv);
        self.tract_sigma_alpha_input = Some(sigma_r_alpha);
        self.tract_r_beta_input = Some(r_beta_s_inv);
        self.tract_sigma_beta_input = Some(sigma_r_beta);
        self.tract_r_alpha_text = format!("{r_alpha_s_inv:.3}");
        self.tract_sigma_alpha_text = format!("{sigma_r_alpha:.3}");
        self.tract_r_beta_text = format!("{r_beta_s_inv:.3}");
        self.tract_sigma_beta_text = format!("{sigma_r_beta:.3}");
    }

    /// Parse experimental TRACT rates from a 4-column TSV/whitespace string
    /// (`r_alpha  sigma_alpha  r_beta  sigma_beta`, optional header/comment lines) and run TRACT.
    pub fn import_tract_rates_tsv(&mut self, raw: &str) -> Result<String, String> {
        let mut parsed: Option<(f64, f64, f64, f64)> = None;
        for (line_idx, raw_line) in raw.lines().enumerate() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let cols: Vec<&str> = line
                .split(|c: char| c.is_ascii_whitespace() || c == ',' || c == ';')
                .filter(|s| !s.is_empty())
                .collect();
            if cols.len() >= 4 && cols[0].eq_ignore_ascii_case("r_alpha") {
                continue;
            }
            if cols.len() < 4 {
                self.tract_report = None;
                return Err(format!(
                    "TRACT TSV line {}: expected 4 numeric columns (r_alpha sigma_alpha r_beta sigma_beta)",
                    line_idx + 1
                ));
            }
            let ra: f64 = cols[0].parse().map_err(|e| {
                self.tract_report = None;
                format!(
                    "TRACT TSV line {}: invalid R_alpha '{}': {e}",
                    line_idx + 1,
                    cols[0]
                )
            })?;
            let sa: f64 = cols[1].parse().map_err(|e| {
                self.tract_report = None;
                format!(
                    "TRACT TSV line {}: invalid sigma_alpha '{}': {e}",
                    line_idx + 1,
                    cols[1]
                )
            })?;
            let rb: f64 = cols[2].parse().map_err(|e| {
                self.tract_report = None;
                format!(
                    "TRACT TSV line {}: invalid R_beta '{}': {e}",
                    line_idx + 1,
                    cols[2]
                )
            })?;
            let sb: f64 = cols[3].parse().map_err(|e| {
                self.tract_report = None;
                format!(
                    "TRACT TSV line {}: invalid sigma_beta '{}': {e}",
                    line_idx + 1,
                    cols[3]
                )
            })?;
            parsed = Some((ra, sa, rb, sb));
            break;
        }
        let Some((ra, sa, rb, sb)) = parsed else {
            self.tract_report = None;
            return Err(TRACT_MISSING_RATES_ERR.to_string());
        };
        self.run_phase22_tract(ra, sa, rb, sb)
    }

    /// Run Phase 22 (`T22.5`) `TRACT` $R_{\text{ex}}$-free $\tau_m$ inversion from experimental
    /// TROSY ($\alpha$) and anti-TROSY ($\beta$) transverse decay rates and explicit uncertainties
    /// (`R_beta > R_alpha > 0`, `sigma >= 0`). Fails closed and clears stale reports on invalid input.
    pub fn run_phase22_tract(
        &mut self,
        r_alpha_s_inv: f64,
        sigma_r_alpha: f64,
        r_beta_s_inv: f64,
        sigma_r_beta: f64,
    ) -> Result<String, String> {
        if !r_alpha_s_inv.is_finite()
            || !r_beta_s_inv.is_finite()
            || r_alpha_s_inv <= 0.0
            || r_beta_s_inv <= r_alpha_s_inv
        {
            self.tract_report = None;
            return Err(
                "TRACT requires finite positive rates with R_beta > R_alpha (eta_xy = (R_beta - R_alpha)/2 > 0)"
                    .to_string(),
            );
        }
        if !sigma_r_alpha.is_finite()
            || !sigma_r_beta.is_finite()
            || sigma_r_alpha < 0.0
            || sigma_r_beta < 0.0
        {
            self.tract_report = None;
            return Err(
                "TRACT requires finite non-negative experimental uncertainties (sigma_alpha >= 0, sigma_beta >= 0)"
                    .to_string(),
            );
        }
        let eta_xy = 0.5 * (r_beta_s_inv - r_alpha_s_inv);
        let sigma_eta_xy = 0.5 * (sigma_r_alpha.powi(2) + sigma_r_beta.powi(2)).sqrt();
        let phys = CcrPhysicsConfig::default();
        let tau_m_rigid = match invert_tract_eta_xy_to_tau_m(
            eta_xy,
            sigma_eta_xy,
            self.primary_field_mhz,
            1.0,
            0.0,
            &phys,
        ) {
            Ok(est) => est,
            Err(e) => {
                self.tract_report = None;
                return Err(e);
            }
        };
        let s2 = self.tract_assumed_s2.clamp(0.2, 1.0);
        let tau_m_s2_corrected = if s2 < 0.999 {
            invert_tract_eta_xy_to_tau_m(
                eta_xy,
                sigma_eta_xy,
                self.primary_field_mhz,
                s2,
                0.0,
                &phys,
            )
            .ok()
        } else {
            None
        };
        let kappa0_sec = match phys.theoretical_kappa0(self.primary_field_mhz) {
            Ok(k) => k,
            Err(e) => {
                self.tract_report = None;
                return Err(e);
            }
        };
        self.set_tract_experimental_rates(r_alpha_s_inv, sigma_r_alpha, r_beta_s_inv, sigma_r_beta);
        self.tract_report = Some(TractSummaryReport {
            field_mhz: self.primary_field_mhz,
            r_alpha_s_inv,
            r_beta_s_inv,
            eta_xy_s_inv: eta_xy,
            sigma_eta_xy_s_inv: sigma_eta_xy,
            tau_m_rigid,
            tau_m_s2_corrected,
            kappa0_sec,
        });
        let msg = format!(
            "TRACT (T22.5): eta_xy = {eta_xy:.2} +- {sigma_eta_xy:.2} s^-1 -> rigid tau_m = {:.2} ns (kappa_0 = {kappa0_sec:.2})",
            tau_m_rigid.tau_m_sec * 1.0e9
        );
        self.status_banner.clone_from(&msg);
        Ok(msg)
    }

    /// Resolve explicit experimental TROSY (`Rα ± σ_α`) and anti-TROSY (`Rβ ± σ_β`) inputs from the
    /// UI text boxes (or programmatic `Option<f64>` fields) and run `TRACT`. Fails closed with
    /// [`TRACT_MISSING_RATES_ERR`] when experimental rates are absent.
    pub fn run_phase22_tract_from_inputs(&mut self) -> Result<String, String> {
        let parse_opt =
            |txt: &str, fallback: Option<f64>, label: &str| -> Result<Option<f64>, String> {
                let trimmed = txt.trim();
                if trimmed.is_empty() {
                    Ok(fallback)
                } else {
                    let v: f64 = trimmed
                        .parse()
                        .map_err(|e| format!("TRACT invalid {label} '{trimmed}': {e}"))?;
                    Ok(Some(v))
                }
            };
        let ra = match parse_opt(
            &self.tract_r_alpha_text,
            self.tract_r_alpha_input,
            "R_alpha",
        ) {
            Ok(v) => v,
            Err(e) => {
                self.tract_report = None;
                return Err(e);
            }
        };
        let rb = match parse_opt(&self.tract_r_beta_text, self.tract_r_beta_input, "R_beta") {
            Ok(v) => v,
            Err(e) => {
                self.tract_report = None;
                return Err(e);
            }
        };
        let (Some(r_alpha), Some(r_beta)) = (ra, rb) else {
            self.tract_report = None;
            return Err(TRACT_MISSING_RATES_ERR.to_string());
        };
        let sa = match parse_opt(
            &self.tract_sigma_alpha_text,
            self.tract_sigma_alpha_input,
            "sigma_alpha",
        ) {
            Ok(v) => v,
            Err(e) => {
                self.tract_report = None;
                return Err(e);
            }
        };
        let sb = match parse_opt(
            &self.tract_sigma_beta_text,
            self.tract_sigma_beta_input,
            "sigma_beta",
        ) {
            Ok(v) => v,
            Err(e) => {
                self.tract_report = None;
                return Err(e);
            }
        };
        let (Some(sigma_alpha), Some(sigma_beta)) = (sa, sb) else {
            self.tract_report = None;
            return Err(format!(
                "{TRACT_MISSING_RATES_ERR} with explicit uncertainties (σ_α, σ_β >= 0)"
            ));
        };
        self.run_phase22_tract(r_alpha, sigma_alpha, r_beta, sigma_beta)
    }

    /// Run Phase 22 analyses (`T22.3` Detectors, `T22.4` IMPACT, and `T22.5` TRACT when
    /// experimental TROSY/anti-TROSY rates are supplied) on the currently fitted `R1`/`R2`/`hetNOE`
    /// rates. Never fabricates placeholder TRACT rates or Model-Free round-trips.
    pub fn run_phase22_suite(&mut self) -> Result<String, String> {
        let det_msg = self.run_phase22_detectors()?;
        let imp_ok = self.run_phase22_impact().is_ok();
        let tract_status = match self.run_phase22_tract_from_inputs() {
            Ok(_) => self
                .tract_report
                .as_ref()
                .map(|t| format!("tau_m = {:.2} ns", t.tau_m_rigid.tau_m_sec * 1.0e9))
                .unwrap_or_else(|| "n/a".to_string()),
            Err(e) => format!("not run ({e})"),
        };
        let msg = format!(
            "{det_msg} IMPACT: {} residues. TRACT: {tract_status}.",
            if imp_ok { self.impact_fits.len() } else { 0 },
        );
        self.status_banner.clone_from(&msg);
        Ok(msg)
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

        for r in &mf.residues {
            if self.unassigned_pick_ids.contains(&r.seq_id) {
                continue;
            }
            let Some(&idx) = id_to_idx.get(&r.seq_id) else {
                continue;
            };
            s2_by_seq_idx.insert(idx, r.params.s2.clamp(0.0, 1.0));
            let rex = r.params.rex_ref_s_inv.max(0.0);
            if rex > 1.0e-4 {
                rex_by_seq_idx.insert(idx, rex);
                rex_max = rex_max.max(rex);
            }
        }
        if s2_by_seq_idx.is_empty() && rex_by_seq_idx.is_empty() {
            return None;
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

        // Connected folder-style sub-tabs inside `last_canvas_rect` so the top toolbar stays
        // well within the <= 12 primary-control budget while giving 1-click access to Curves/M1-M5,
        // Pseudo-3D Bruker T1/T2/hetNOE import, and Phase 22 Detectors / IMPACT / TRACT.
        self.draw_studio_sub_tabs(ui);

        match self.active_sub_tab {
            RelaxStudioSubTab::CurvesAndModelFree => {
                if !self.has_fitted_rates() {
                    ui.vertical_centered(|ui| {
                        ui.add_space(28.0);
                        ui.label(
                            egui::RichText::new(
                                "No relaxation curves loaded yet. Open 'Pseudo-3D T1/T2/hetNOE', click 'Demo 600+800 MHz', or paste TSV/NEF.",
                            )
                            .size(TYPE_BODY_PX),
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
            }
            RelaxStudioSubTab::Pseudo3dImport => {
                self.draw_pseudo3d_import_tab(ui, pdb_text, &mut actions);
            }
            RelaxStudioSubTab::DetectorsImpactTract => {
                self.draw_phase22_dynamics_tab(ui, sequence, seq_ids, &mut actions);
            }
        }

        actions
    }

    fn draw_studio_sub_tabs(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 3.0;
            for tab in RelaxStudioSubTab::ALL {
                let selected = self.active_sub_tab == tab;
                let fill = if selected {
                    egui::Color32::from_rgb(16, 22, 32)
                } else {
                    egui::Color32::from_rgb(24, 31, 44)
                };
                let stroke_col = if selected {
                    egui::Color32::from_rgb(84, 198, 250)
                } else {
                    egui::Color32::from_rgb(56, 70, 92)
                };
                let text_col = if selected {
                    egui::Color32::from_rgb(238, 246, 255)
                } else {
                    egui::Color32::from_rgb(172, 188, 210)
                };
                let btn = egui::Button::new(
                    egui::RichText::new(tab.label())
                        .size(TYPE_CAPTION_PX)
                        .color(text_col),
                )
                .fill(fill)
                .stroke(egui::Stroke::new(
                    if selected { 1.4 } else { 1.0 },
                    stroke_col,
                ))
                .corner_radius(egui::CornerRadius {
                    nw: 5,
                    ne: 5,
                    sw: 0,
                    se: 0,
                });
                if ui.add(btn).clicked() {
                    self.active_sub_tab = tab;
                }
            }
        });
        let sep_y = ui.cursor().top();
        let x0 = ui.available_rect_before_wrap().left();
        let x1 = ui.available_rect_before_wrap().right();
        ui.painter().line_segment(
            [egui::pos2(x0, sep_y), egui::pos2(x1, sep_y)],
            egui::Stroke::new(1.0, egui::Color32::from_rgb(48, 60, 80)),
        );
        ui.add_space(4.0);
    }

    fn draw_pseudo3d_import_tab(
        &mut self,
        ui: &mut egui::Ui,
        pdb_text: Option<&str>,
        actions: &mut Vec<RelaxPanelAction>,
    ) {
        egui::ScrollArea::vertical()
            .id_salt("relax_pseudo3d_import_scroll")
            .show(ui, |ui| {
                // Locked HSQC footprint readiness banner
                egui::Frame::group(ui.style())
                    .fill(egui::Color32::from_rgb(18, 24, 35))
                    .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(46, 60, 82)))
                    .corner_radius(4.0)
                    .inner_margin(egui::Margin::same(8))
                    .show(ui, |ui| {
                        ui.horizontal_wrapped(|ui| {
                            ui.label(
                                egui::RichText::new("Locked HSQC Peak Footprint Reference:")
                                    .size(TYPE_BODY_PX)
                                    .strong(),
                            );
                            if !self.synced_peak_seeds.is_empty() {
                                ui.colored_label(
                                    egui::Color32::from_rgb(86, 228, 154),
                                    egui::RichText::new(format!(
                                        "{} Assign-mode HSQC peaks synced",
                                        self.synced_peak_seeds.len()
                                    ))
                                    .size(TYPE_BODY_PX),
                                );
                            } else {
                                ui.colored_label(
                                    egui::Color32::from_rgb(248, 182, 74),
                                    egui::RichText::new(
                                        "0 synced (will auto-pick plane 0 or sync HSQC peaks from Assign)",
                                    )
                                    .size(TYPE_CAPTION_PX),
                                );
                            }
                        });
                    });

                ui.add_space(6.0);

                #[cfg(target_arch = "wasm32")]
                {
                    ui.colored_label(
                        egui::Color32::from_rgb(248, 182, 74),
                        egui::RichText::new(
                            "Native Bruker folder loading is available on desktop builds; in WebAssembly use TSV/NEF import or Demo 600+800 MHz.",
                        )
                        .size(TYPE_CAPTION_PX),
                    );
                    ui.add_space(4.0);
                }

                // Slot 1: T1 (R1)
                self.draw_pseudo3d_slot_card(
                    ui,
                    "Slot 1 - T1 (R1) Pseudo-3D Folder",
                    RelaxExperimentKind::R1,
                    egui::Color32::from_rgb(76, 185, 255),
                    actions,
                );
                ui.add_space(6.0);

                // Slot 2: T2 (R2)
                self.draw_pseudo3d_slot_card(
                    ui,
                    "Slot 2 - T2 (R2) Pseudo-3D Folder",
                    RelaxExperimentKind::R2,
                    egui::Color32::from_rgb(255, 162, 68),
                    actions,
                );
                ui.add_space(6.0);

                // Slot 3: hetNOE
                self.draw_pseudo3d_slot_card(
                    ui,
                    "Slot 3 - {1H}-15N hetNOE Pseudo-3D Folder",
                    RelaxExperimentKind::HetNoe,
                    egui::Color32::from_rgb(176, 128, 248),
                    actions,
                );
                ui.add_space(8.0);

                // 1-Click Run Model-Free & Phase 22 Bridge
                egui::Frame::group(ui.style())
                    .fill(egui::Color32::from_rgb(16, 26, 36))
                    .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(52, 86, 108)))
                    .corner_radius(4.0)
                    .inner_margin(egui::Margin::same(8))
                    .show(ui, |ui| {
                        ui.horizontal_wrapped(|ui| {
                            let ready = !self.r1_fits.is_empty() && !self.r2_fits.is_empty();
                            let mf_btn = egui::Button::new(
                                egui::RichText::new("Run Model-Free")
                                    .size(TYPE_BODY_PX)
                                    .strong(),
                            )
                            .fill(egui::Color32::from_rgb(26, 114, 86));
                            if ui
                                .add_enabled(ready, mf_btn)
                                .on_hover_text(
                                    "Fit Lipari-Szabo M1-M5 using the fitted R1/R2/hetNOE rates and loaded PDB/AF structure",
                                )
                                .clicked()
                            {
                                match self.run_modelfree(pdb_text) {
                                    Ok(msg) => {
                                        self.active_sub_tab = RelaxStudioSubTab::CurvesAndModelFree;
                                        actions.push(RelaxPanelAction::StatusMessage(msg));
                                    }
                                    Err(e) => self.status_banner = format!("Model-Free error: {e}"),
                                }
                            }
                            if ui
                                .add_enabled(
                                    ready,
                                    egui::Button::new(
                                        egui::RichText::new("Run Detectors / IMPACT / TRACT")
                                            .size(TYPE_BODY_PX),
                                    ),
                                )
                                .clicked()
                            {
                                match self.run_phase22_suite() {
                                    Ok(msg) => {
                                        self.active_sub_tab =
                                            RelaxStudioSubTab::DetectorsImpactTract;
                                        actions.push(RelaxPanelAction::StatusMessage(msg));
                                    }
                                    Err(e) => {
                                        self.status_banner = format!("Phase 22 dynamics error: {e}")
                                    }
                                }
                            }
                            let struct_hint = if pdb_text.is_some() {
                                "PDB/AF N-H vectors active (anisotropic diffusion enabled)"
                            } else {
                                "No PDB loaded (isotropic diffusion)"
                            };
                            ui.label(
                                egui::RichText::new(struct_hint)
                                    .size(TYPE_CAPTION_PX)
                                    .color(egui::Color32::from_rgb(155, 190, 215)),
                            );
                        });
                    });
            });
    }

    fn draw_pseudo3d_slot_card(
        &mut self,
        ui: &mut egui::Ui,
        title: &str,
        kind: RelaxExperimentKind,
        accent: egui::Color32,
        actions: &mut Vec<RelaxPanelAction>,
    ) {
        let fitted_count = match kind {
            RelaxExperimentKind::R1 => self.r1_fits.len(),
            RelaxExperimentKind::R2 | RelaxExperimentKind::R1rho => self.r2_fits.len(),
            RelaxExperimentKind::HetNoe => self.hetnoe_fits.len(),
            _ => 0,
        };
        let (dir_buf, delays_buf, loaded_planes, warnings_len) = match kind {
            RelaxExperimentKind::R1 => (
                &mut self.t1_slot.dir_path,
                &mut self.t1_slot.delays_text,
                self.t1_slot.loaded_planes,
                self.t1_slot.warnings.len(),
            ),
            RelaxExperimentKind::R2 | RelaxExperimentKind::R1rho => (
                &mut self.t2_slot.dir_path,
                &mut self.t2_slot.delays_text,
                self.t2_slot.loaded_planes,
                self.t2_slot.warnings.len(),
            ),
            _ => (
                &mut self.hetnoe_slot.dir_path,
                &mut self.hetnoe_slot.delays_text,
                self.hetnoe_slot.loaded_planes,
                self.hetnoe_slot.warnings.len(),
            ),
        };

        let mut trigger_load = false;
        egui::Frame::group(ui.style())
            .fill(egui::Color32::from_rgb(18, 23, 33))
            .stroke(egui::Stroke::new(1.0, accent.gamma_multiply(0.55)))
            .corner_radius(4.0)
            .inner_margin(egui::Margin::same(8))
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.label(
                        egui::RichText::new(title)
                            .size(TYPE_BODY_PX)
                            .strong()
                            .color(accent),
                    );
                    let badge = if fitted_count > 0 {
                        format!("{fitted_count} residues fitted ({loaded_planes} planes)")
                    } else {
                        "Not loaded".to_string()
                    };
                    ui.label(egui::RichText::new(badge).size(TYPE_CAPTION_PX).color(
                        if fitted_count > 0 {
                            egui::Color32::from_rgb(98, 228, 160)
                        } else {
                            egui::Color32::from_rgb(138, 152, 174)
                        },
                    ));
                    if warnings_len > 0 {
                        ui.label(
                            egui::RichText::new(format!("({warnings_len} warnings)"))
                                .size(TYPE_CAPTION_PX)
                                .color(egui::Color32::from_rgb(248, 182, 74)),
                        );
                    }
                });

                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Bruker Dir:").size(TYPE_CAPTION_PX));
                    ui.add(
                        egui::TextEdit::singleline(dir_buf)
                            .id_salt(format!("relax_dir_{}", kind.as_str()))
                            .desired_width((ui.available_width() - 135.0).max(140.0))
                            .font(egui::FontId::monospace(TYPE_CAPTION_PX))
                            .hint_text("/path/to/bruker/expno (acqus + ser + vdlist/vclist)"),
                    );
                    let btn_lbl = match kind {
                        RelaxExperimentKind::R1 => "Load & Fit T1",
                        RelaxExperimentKind::R2 | RelaxExperimentKind::R1rho => "Load & Fit T2",
                        _ => "Load hetNOE",
                    };
                    if ui
                        .add(egui::Button::new(
                            egui::RichText::new(btn_lbl).size(TYPE_CAPTION_PX),
                        ))
                        .clicked()
                    {
                        trigger_load = true;
                    }
                });

                ui.horizontal_wrapped(|ui| {
                    if kind == RelaxExperimentKind::HetNoe {
                        ui.label(egui::RichText::new("Plane Order:").size(TYPE_CAPTION_PX));
                        for (ord, lbl, tip) in [
                            (
                                HetNoeOrder::Auto,
                                "Auto (pulprog)",
                                "Fail-closed default: prove saturated-vs-reference order from pulseprogram, or fail if ambiguous",
                            ),
                            (
                                HetNoeOrder::RefFirst,
                                "Ref-First",
                                "Explicitly assert plane 0 = reference (unsaturated) and plane 1 = saturated",
                            ),
                            (
                                HetNoeOrder::SatFirst,
                                "Sat-First",
                                "Explicitly assert plane 0 = saturated and plane 1 = reference (unsaturated)",
                            ),
                        ] {
                            if ui
                                .add(egui::Button::selectable(
                                    self.hetnoe_order == ord,
                                    egui::RichText::new(lbl).size(TYPE_CAPTION_PX),
                                ))
                                .on_hover_text(tip)
                                .clicked()
                            {
                                self.hetnoe_order = ord;
                            }
                        }
                    } else {
                        let sched_lbl = if kind == RelaxExperimentKind::R1 {
                            "vdlist delays:"
                        } else {
                            "vdlist / vclist:"
                        };
                        ui.label(egui::RichText::new(sched_lbl).size(TYPE_CAPTION_PX));
                        ui.add(
                            egui::TextEdit::singleline(delays_buf)
                                .id_salt(format!("relax_delays_{}", kind.as_str()))
                                .desired_width(240.0)
                                .font(egui::FontId::monospace(TYPE_CAPTION_PX))
                                .hint_text("optional override, e.g. 0.01, 0.05, 0.15, 0.4, 0.8"),
                        );
                        if matches!(kind, RelaxExperimentKind::R2 | RelaxExperimentKind::R1rho) {
                            ui.label(
                                egui::RichText::new("CPMG cycle (ms):").size(TYPE_CAPTION_PX),
                            );
                            ui.add(
                                egui::DragValue::new(&mut self.t2_vclist_unit_ms)
                                    .range(0.0..=50.0)
                                    .speed(0.2)
                                    .fixed_decimals(2),
                            )
                            .on_hover_text(
                                "Explicit positive CPMG cycle duration (ms) required when vclist contains plain integer loop counts (e.g. 2 4 8 16)",
                            );
                        }
                    }
                });
            });

        if trigger_load {
            let dir_str = match kind {
                RelaxExperimentKind::R1 => self.t1_slot.dir_path.clone(),
                RelaxExperimentKind::R2 | RelaxExperimentKind::R1rho => {
                    self.t2_slot.dir_path.clone()
                }
                _ => self.hetnoe_slot.dir_path.clone(),
            };
            let delays_str = match kind {
                RelaxExperimentKind::R1 => self.t1_slot.delays_text.clone(),
                RelaxExperimentKind::R2 | RelaxExperimentKind::R1rho => {
                    self.t2_slot.delays_text.clone()
                }
                _ => String::new(),
            };
            let unit_ms = matches!(kind, RelaxExperimentKind::R2 | RelaxExperimentKind::R1rho)
                .then_some(self.t2_vclist_unit_ms);
            match Self::parse_delays_override_str(&delays_str, unit_ms) {
                Ok(override_opt) => {
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        match self.load_bruker_pseudo3d_series(
                            kind,
                            &dir_str,
                            self.hetnoe_order,
                            override_opt.as_deref(),
                        ) {
                            Ok(msg) => actions.push(RelaxPanelAction::StatusMessage(msg)),
                            Err(e) => self.status_banner = format!("Pseudo-3D load error: {e}"),
                        }
                    }
                    #[cfg(target_arch = "wasm32")]
                    {
                        let _ = (&dir_str, override_opt, actions);
                        self.status_banner =
                            "Native Bruker folder loading requires desktop build.".to_string();
                    }
                }
                Err(e) => self.status_banner = format!("Delay schedule parse error: {e}"),
            }
        }
    }

    fn draw_phase22_dynamics_tab(
        &mut self,
        ui: &mut egui::Ui,
        sequence: &[String],
        seq_ids: &[i32],
        actions: &mut Vec<RelaxPanelAction>,
    ) {
        egui::ScrollArea::vertical()
            .id_salt("relax_phase22_dynamics_scroll")
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    let has_r1_r2 = !self.r1_fits.is_empty() && !self.r2_fits.is_empty();
                    if ui
                        .add_enabled(
                            has_r1_r2,
                            egui::Button::new(
                                egui::RichText::new("Run Detectors + IMPACT + TRACT")
                                    .size(TYPE_BODY_PX)
                                    .strong(),
                            )
                            .fill(egui::Color32::from_rgb(28, 92, 118)),
                        )
                        .clicked()
                    {
                        match self.run_phase22_suite() {
                            Ok(msg) => actions.push(RelaxPanelAction::StatusMessage(msg)),
                            Err(e) => self.status_banner = format!("Phase 22 error: {e}"),
                        }
                    }
                    if ui
                        .add_enabled(
                            has_r1_r2,
                            egui::Button::new(
                                egui::RichText::new("Detectors (T22.3)").size(TYPE_CAPTION_PX),
                            ),
                        )
                        .clicked()
                    {
                        match self.run_phase22_detectors() {
                            Ok(msg) => actions.push(RelaxPanelAction::StatusMessage(msg)),
                            Err(e) => self.status_banner = format!("Detectors error: {e}"),
                        }
                    }
                    if ui
                        .add_enabled(
                            has_r1_r2,
                            egui::Button::new(
                                egui::RichText::new("IMPACT (T22.4)").size(TYPE_CAPTION_PX),
                            ),
                        )
                        .clicked()
                    {
                        match self.run_phase22_impact() {
                            Ok(msg) => actions.push(RelaxPanelAction::StatusMessage(msg)),
                            Err(e) => self.status_banner = format!("IMPACT error: {e}"),
                        }
                    }
                });

                ui.add_space(6.0);

                // Section A: T22.3 Dynamics Detectors
                egui::Frame::group(ui.style())
                    .fill(egui::Color32::from_rgb(18, 23, 33))
                    .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(48, 66, 90)))
                    .corner_radius(4.0)
                    .inner_margin(egui::Margin::same(8))
                    .show(ui, |ui| {
                        ui.label(
                            egui::RichText::new(
                                "T22.3 - Linear Optimal-Window Dynamics Detectors (rho_n(tau))",
                            )
                            .size(TYPE_BODY_PX)
                            .strong()
                            .color(egui::Color32::from_rgb(92, 210, 255)),
                        );
                        if let Some(design) = &self.detector_design {
                            ui.label(
                                egui::RichText::new(format!(
                                    "{} detectors across {} experiments (max overlap {:.3}, cond {:.2})",
                                    design.num_detectors,
                                    design.experiments.len(),
                                    design.max_pairwise_overlap,
                                    design.transform_condition_number
                                ))
                                .size(TYPE_CAPTION_PX)
                                .color(egui::Color32::from_rgb(172, 192, 214)),
                            );
                            draw_detector_windows_canvas(ui, design);

                            let active_sid = self
                                .selected_seq_id
                                .filter(|sid| self.detector_fits.contains_key(sid))
                                .or_else(|| self.detector_fits.keys().next().copied());
                            if let Some(sid) = active_sid {
                                if let Some(dfit) = self.detector_fits.get(&sid) {
                                    let rname = seq_ids
                                        .iter()
                                        .position(|&x| x == sid)
                                        .and_then(|k| sequence.get(k))
                                        .map(String::as_str)
                                        .unwrap_or("RES");
                                    ui.add_space(4.0);
                                    ui.label(
                                        egui::RichText::new(format!(
                                            "{rname}{sid} detector responses (chi2/dof = {:.2}/{}):",
                                            dfit.chi2, dfit.dof
                                        ))
                                        .size(TYPE_CAPTION_PX)
                                        .strong(),
                                    );
                                    egui::Grid::new("phase22_detector_table")
                                        .num_columns(4)
                                        .spacing([12.0, 2.0])
                                        .striped(true)
                                        .show(ui, |ui| {
                                            ui.label(egui::RichText::new("Detector").size(TYPE_CAPTION_PX).strong());
                                            ui.label(egui::RichText::new("Peak tau (ns)").size(TYPE_CAPTION_PX).strong());
                                            ui.label(egui::RichText::new("Response").size(TYPE_CAPTION_PX).strong());
                                            ui.label(egui::RichText::new("FWHM (dec)").size(TYPE_CAPTION_PX).strong());
                                            ui.end_row();
                                            for (idx, win) in design.windows.iter().enumerate() {
                                                let resp = dfit.responses.get(idx).copied().unwrap_or(0.0);
                                                let sig = dfit.sigmas.get(idx).copied().unwrap_or(0.0);
                                                ui.label(
                                                    egui::RichText::new(format!("rho_{idx}"))
                                                        .monospace()
                                                        .size(TYPE_CAPTION_PX),
                                                );
                                                ui.label(
                                                    egui::RichText::new(format!(
                                                        "{:.3}",
                                                        win.peak_tau_sec * 1.0e9
                                                    ))
                                                    .monospace()
                                                    .size(TYPE_CAPTION_PX),
                                                );
                                                ui.label(
                                                    egui::RichText::new(format!("{resp:.3} +- {sig:.3}"))
                                                        .monospace()
                                                        .size(TYPE_CAPTION_PX),
                                                );
                                                ui.label(
                                                    egui::RichText::new(format!("{:.2}", win.fwhm_decades))
                                                        .monospace()
                                                        .size(TYPE_CAPTION_PX),
                                                );
                                                ui.end_row();
                                            }
                                        });
                                }
                            }
                        } else {
                            ui.label(
                                egui::RichText::new(
                                    "Fit R1/R2/hetNOE rates and click 'Run Detectors + IMPACT + TRACT' to build timescale detector windows.",
                                )
                                .size(TYPE_CAPTION_PX)
                                .color(egui::Color32::from_rgb(148, 162, 184)),
                            );
                        }
                    });

                ui.add_space(6.0);

                // Section B: T22.4 IMPACT Simplex Decomposition
                egui::Frame::group(ui.style())
                    .fill(egui::Color32::from_rgb(18, 23, 33))
                    .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(64, 58, 96)))
                    .corner_radius(4.0)
                    .inner_margin(egui::Margin::same(8))
                    .show(ui, |ui| {
                        ui.label(
                            egui::RichText::new(
                                "T22.4 - IMPACT Simplex Correlation-Time Decomposition",
                            )
                            .size(TYPE_BODY_PX)
                            .strong()
                            .color(egui::Color32::from_rgb(186, 140, 252)),
                        );
                        let active_sid = self
                            .selected_seq_id
                            .filter(|sid| self.impact_fits.contains_key(sid))
                            .or_else(|| self.impact_fits.keys().next().copied());
                        if let Some(sid) = active_sid {
                            if let Some(ifit) = self.impact_fits.get(&sid) {
                                ui.label(
                                    egui::RichText::new(format!(
                                        "Residue {sid}: M = {} timescales | sum(A) = {:.4} | chi2 = {:.2} | KKT = {:.2e}",
                                        ifit.tau_grid_sec.len(),
                                        ifit.sum_amplitudes,
                                        ifit.chi2,
                                        ifit.kkt_violation
                                    ))
                                    .size(TYPE_CAPTION_PX)
                                    .color(egui::Color32::from_rgb(176, 192, 214)),
                                );
                                egui::Grid::new("phase22_impact_table")
                                    .num_columns(3)
                                    .spacing([12.0, 2.0])
                                    .striped(true)
                                    .show(ui, |ui| {
                                        ui.label(egui::RichText::new("tau_m (ns)").size(TYPE_CAPTION_PX).strong());
                                        ui.label(egui::RichText::new("Weight A_m").size(TYPE_CAPTION_PX).strong());
                                        ui.label(egui::RichText::new("MC 1-sigma").size(TYPE_CAPTION_PX).strong());
                                        ui.end_row();
                                        for (m, (&tau, &amp)) in ifit
                                            .tau_grid_sec
                                            .iter()
                                            .zip(ifit.amplitudes.iter())
                                            .enumerate()
                                        {
                                            let sig = ifit
                                                .sigma_amplitudes_mc
                                                .get(m)
                                                .copied()
                                                .unwrap_or(0.0);
                                            ui.label(
                                                egui::RichText::new(format!("{:.4}", tau * 1.0e9))
                                                    .monospace()
                                                    .size(TYPE_CAPTION_PX),
                                            );
                                            ui.label(
                                                egui::RichText::new(format!("{amp:.3}"))
                                                    .monospace()
                                                    .size(TYPE_CAPTION_PX),
                                            );
                                            ui.label(
                                                egui::RichText::new(format!("{sig:.3}"))
                                                    .monospace()
                                                    .size(TYPE_CAPTION_PX),
                                            );
                                            ui.end_row();
                                        }
                                    });
                            }
                        } else {
                            ui.label(
                                egui::RichText::new(
                                    "Requires fitted R1, R2, and hetNOE rates for spectral-density simplex projection.",
                                )
                                .size(TYPE_CAPTION_PX)
                                .color(egui::Color32::from_rgb(148, 162, 184)),
                            );
                        }
                    });

                ui.add_space(6.0);

                // Section C: T22.5 TRACT R_ex-free tau_m Inversion
                egui::Frame::group(ui.style())
                    .fill(egui::Color32::from_rgb(18, 23, 33))
                    .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(44, 82, 68)))
                    .corner_radius(4.0)
                    .inner_margin(egui::Margin::same(8))
                    .show(ui, |ui| {
                        ui.label(
                            egui::RichText::new(
                                "T22.5 - TRACT (TROSY alpha/beta) R_ex-Free tau_m & CCR",
                            )
                            .size(TYPE_BODY_PX)
                            .strong()
                            .color(egui::Color32::from_rgb(94, 228, 160)),
                        );
                        ui.horizontal_wrapped(|ui| {
                            ui.label(egui::RichText::new("Rα (s⁻¹):").size(TYPE_CAPTION_PX));
                            if ui
                                .add(
                                    egui::TextEdit::singleline(&mut self.tract_r_alpha_text)
                                        .id_salt("tract_r_alpha_input")
                                        .desired_width(54.0)
                                        .font(egui::FontId::monospace(TYPE_CAPTION_PX))
                                        .hint_text("req."),
                                )
                                .changed()
                            {
                                self.tract_r_alpha_input =
                                    self.tract_r_alpha_text.trim().parse::<f64>().ok();
                            }
                            ui.label(egui::RichText::new("± σ_α:").size(TYPE_CAPTION_PX));
                            if ui
                                .add(
                                    egui::TextEdit::singleline(&mut self.tract_sigma_alpha_text)
                                        .id_salt("tract_sigma_alpha_input")
                                        .desired_width(50.0)
                                        .font(egui::FontId::monospace(TYPE_CAPTION_PX))
                                        .hint_text("req."),
                                )
                                .changed()
                            {
                                self.tract_sigma_alpha_input =
                                    self.tract_sigma_alpha_text.trim().parse::<f64>().ok();
                            }
                            ui.label(egui::RichText::new("Rβ (s⁻¹):").size(TYPE_CAPTION_PX));
                            if ui
                                .add(
                                    egui::TextEdit::singleline(&mut self.tract_r_beta_text)
                                        .id_salt("tract_r_beta_input")
                                        .desired_width(54.0)
                                        .font(egui::FontId::monospace(TYPE_CAPTION_PX))
                                        .hint_text("req."),
                                )
                                .changed()
                            {
                                self.tract_r_beta_input =
                                    self.tract_r_beta_text.trim().parse::<f64>().ok();
                            }
                            ui.label(egui::RichText::new("± σ_β:").size(TYPE_CAPTION_PX));
                            if ui
                                .add(
                                    egui::TextEdit::singleline(&mut self.tract_sigma_beta_text)
                                        .id_salt("tract_sigma_beta_input")
                                        .desired_width(50.0)
                                        .font(egui::FontId::monospace(TYPE_CAPTION_PX))
                                        .hint_text("req."),
                                )
                                .changed()
                            {
                                self.tract_sigma_beta_input =
                                    self.tract_sigma_beta_text.trim().parse::<f64>().ok();
                            }
                            ui.label(egui::RichText::new("S^2:").size(TYPE_CAPTION_PX));
                            ui.add(
                                egui::DragValue::new(&mut self.tract_assumed_s2)
                                    .range(0.2..=1.0)
                                    .speed(0.01)
                                    .fixed_decimals(2),
                            );
                            if ui
                                .add(egui::Button::new(
                                    egui::RichText::new("Compute TRACT tau_m")
                                        .size(TYPE_CAPTION_PX),
                                ))
                                .clicked()
                            {
                                match self.run_phase22_tract_from_inputs() {
                                    Ok(msg) => actions.push(RelaxPanelAction::StatusMessage(msg)),
                                    Err(e) => self.status_banner = format!("TRACT error: {e}"),
                                }
                            }
                        });
                        if let Some(tr) = &self.tract_report {
                            let s2_txt = tr
                                .tau_m_s2_corrected
                                .map(|est| format!(" | S^2-corr tau_m = {:.2} ns", est.tau_m_sec * 1.0e9))
                                .unwrap_or_default();
                            ui.label(
                                egui::RichText::new(format!(
                                    "eta_xy = {:.2} +- {:.2} s^-1 | rigid tau_m = {:.2} +- {:.2} ns{s2_txt} | Kroenke kappa_0 = {:.2}",
                                    tr.eta_xy_s_inv,
                                    tr.sigma_eta_xy_s_inv,
                                    tr.tau_m_rigid.tau_m_sec * 1.0e9,
                                    tr.tau_m_rigid.sigma_tau_m_sec * 1.0e9,
                                    tr.kappa0_sec
                                ))
                                .monospace()
                                .size(TYPE_CAPTION_PX)
                                .color(egui::Color32::from_rgb(168, 236, 198)),
                            );
                        } else {
                            ui.label(
                                egui::RichText::new(
                                    "TRACT requires experimental TROSY (Rα) and anti-TROSY (Rβ) decay rates (with σ_α, σ_β).",
                                )
                                .size(TYPE_CAPTION_PX)
                                .color(egui::Color32::from_rgb(148, 162, 184)),
                            );
                        }
                    });
            });
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

fn parse_sequence_3letter_tokens(sequence: &str) -> Vec<String> {
    let trimmed = sequence.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }
    let non_header_lines: Vec<&str> = trimmed
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('>') && !l.starts_with(';'))
        .collect();
    let joined = non_header_lines.join(" ");
    let tokens: Vec<&str> = joined
        .split(|c: char| c.is_ascii_whitespace() || c == ',' || c == ';')
        .filter(|t| !t.is_empty())
        .collect();
    if tokens.len() > 1
        && tokens
            .iter()
            .all(|t| t.len() == 3 && t.chars().all(|c| c.is_ascii_alphabetic()))
    {
        return tokens.into_iter().map(|t| t.to_ascii_uppercase()).collect();
    }
    let mut out = Vec::new();
    for line in non_header_lines {
        for ch in line.chars() {
            if ch.is_ascii_alphabetic() {
                out.push(one_to_three(ch.to_ascii_uppercase()).to_string());
            }
        }
    }
    out
}

fn normalize_res_name_3letter(raw: &str) -> String {
    let up = raw.trim().to_ascii_uppercase();
    if up.len() == 1 {
        if let Some(ch) = up.chars().next() {
            return one_to_three(ch).to_string();
        }
    }
    up
}

fn extract_seq_id_from_label(label: &str) -> Option<i32> {
    let digits: String = label.chars().filter(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return None;
    }
    digits.parse::<i32>().ok().filter(|&v| v > 0)
}

fn draw_detector_windows_canvas(ui: &mut egui::Ui, design: &DetectorDesign) {
    let desired = egui::vec2(ui.available_width().max(240.0), 108.0);
    let (rect, _resp) = ui.allocate_exact_size(desired, egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 3.0, egui::Color32::from_rgb(14, 19, 27));
    painter.rect_stroke(
        rect,
        3.0,
        egui::Stroke::new(1.0, egui::Color32::from_rgb(40, 52, 70)),
        egui::StrokeKind::Inside,
    );

    let pad_l = 36.0_f32;
    let pad_r = 12.0_f32;
    let pad_t = 10.0_f32;
    let pad_b = 20.0_f32;
    let plot = egui::Rect::from_min_max(
        egui::pos2(rect.left() + pad_l, rect.top() + pad_t),
        egui::pos2(rect.right() - pad_r, rect.bottom() - pad_b),
    );
    if plot.width() <= 10.0 || plot.height() <= 10.0 || design.log10_tau_grid.len() < 2 {
        return;
    }

    let z_min = design.log10_tau_grid.first().copied().unwrap_or(-13.0);
    let z_max = design.log10_tau_grid.last().copied().unwrap_or(-7.0);
    let z_span = (z_max - z_min).max(1.0e-6);

    let palette = [
        egui::Color32::from_rgb(76, 206, 252),
        egui::Color32::from_rgb(92, 232, 154),
        egui::Color32::from_rgb(248, 178, 68),
        egui::Color32::from_rgb(196, 132, 252),
        egui::Color32::from_rgb(246, 108, 118),
    ];

    for (w_idx, win) in design.windows.iter().enumerate() {
        let col = palette[w_idx % palette.len()];
        let pts: Vec<egui::Pos2> = design
            .log10_tau_grid
            .iter()
            .zip(win.sensitivities.iter())
            .map(|(&z, &rho)| {
                let fx = ((z - z_min) / z_span).clamp(0.0, 1.0) as f32;
                let fy = rho.clamp(0.0, 1.05) as f32 / 1.05;
                egui::pos2(
                    plot.left() + fx * plot.width(),
                    plot.bottom() - fy * plot.height(),
                )
            })
            .collect();
        if pts.len() >= 2 {
            painter.add(egui::Shape::line(pts, egui::Stroke::new(1.6, col)));
        }
    }

    painter.text(
        egui::pos2(plot.left() - 4.0, plot.top() + 4.0),
        egui::Align2::RIGHT_CENTER,
        "1.0",
        egui::FontId::monospace(TYPE_CAPTION_PX),
        egui::Color32::from_rgb(138, 152, 174),
    );
    painter.text(
        egui::pos2(plot.left() - 4.0, plot.bottom()),
        egui::Align2::RIGHT_CENTER,
        "0.0",
        egui::FontId::monospace(TYPE_CAPTION_PX),
        egui::Color32::from_rgb(138, 152, 174),
    );
    painter.text(
        egui::pos2(plot.left(), plot.bottom() + 10.0),
        egui::Align2::LEFT_CENTER,
        format!("10^{z_min:.0}s"),
        egui::FontId::monospace(TYPE_CAPTION_PX),
        egui::Color32::from_rgb(138, 152, 174),
    );
    painter.text(
        egui::pos2(plot.right(), plot.bottom() + 10.0),
        egui::Align2::RIGHT_CENTER,
        format!("10^{z_max:.0}s"),
        egui::FontId::monospace(TYPE_CAPTION_PX),
        egui::Color32::from_rgb(138, 152, 174),
    );
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
