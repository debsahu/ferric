//! Opt-in CLI integration (`T21.4`, `docs/specs/21-alphafold-integrative-nmr.md` section 4)
//! for `ferric assign --af-structure <model.cif|pdb> [--af-pae <pae.json>] [--noesy-15n <peaks.tsv|nef>]`.
//!
//! When `--af-structure` is omitted, `ferric assign` never invokes this module and executes the
//! bit-identical structure-free `specs/03` / `specs/10` / `specs/19` workflow.
//!
//! # Literature & Clean-Room Provenance
//! - Lange, K. L., Little, R. J. A. & Taylor, J. M. G. (*J. Am. Stat. Assoc.* **1989**, 84, 881-896):
//!   heavy-tailed Student-`t(nu = 4)` likelihood clamping.
//! - Jumper, J. et al. (*Nature* **2021**, 596, 583-589) & Varadi, M. et al. (*Nucleic Acids Res.* **2022**,
//!   50, D439-D444): `pLDDT` and Predicted Aligned Error (`PAE`) gating.
//! - Wuthrich, K. (*NMR of Proteins and Nucleic Acids*, Wiley, **1986**) & Grzesiek, S. et al.
//!   (*J. Am. Chem. Soc.* **1995**, 117, 5312-5315): `3D 15N-edited NOESY-HSQC` `d_NN(i, j)` contacts.

use std::path::PathBuf;

use nmr_core::af_assign_priors::{
    AfAssignConfig, AfAssignPriors, AfShiftOffsets, DEFAULT_AF_CLAMP_NATS, DEFAULT_AF_WEIGHT,
    DEFAULT_NOE_WEIGHT,
};
use nmr_core::evo_graph::{assign_evo_graph_from_peaks, EvoGraphConfig};
use nmr_core::peaks::Peak;
use nmr_core::priors::{
    estimate_reference_offsets_detailed, sync_detailed_reference_offsets, Priors, ShiftOffsets,
};
use nmr_core::qc::{
    apply_dataset_qc_to_assignments, evaluate_dataset_qc, EVO_GRAPH_CALIBRATED_RUNS,
    STANDARD_CPU_CALIBRATED_RUNS, STANDARD_GPU_CALIBRATED_RUNS,
};
use nmr_core::solver::{assign_with_af_priors, prepare_scoring_priors, AnnealConfig, ScoreConfig};
use nmr_core::spinsystems::{build_spin_systems, Tolerances};
use nmr_core::torsion_db::HomologFirewallQuery;
use nmr_gpu::GpuEngine;

use crate::{
    real_benchmark, resolve_solver_engine, version_string, EngineKind, RegimeKind, SolverKind,
    SolverWorkflowOutput, ToleranceOverrides,
};

/// Parsed opt-in `--af-*` and `--noesy-15n` flags on `ferric assign`.
#[derive(Clone, Debug, Default)]
pub struct AfAssignCliOpts {
    pub af_structure_path: Option<PathBuf>,
    pub af_pae_path: Option<PathBuf>,
    pub noesy_15n_path: Option<PathBuf>,
    pub af_weight: Option<f64>,
    pub noe_weight: Option<f64>,
    pub af_clamp_nats: Option<f64>,
    pub max_seq_identity: Option<f64>,
    pub firewall_query: HomologFirewallQuery,
    pub has_firewall_flag: bool,
    pub force_bfactor_plddt: bool,
    pub treat_as_model: bool,
}

fn push_csv_items(dst: &mut Vec<String>, raw: &str) {
    for part in raw.split(',') {
        let trimmed = part.trim();
        if !trimmed.is_empty() {
            dst.push(trimmed.to_string());
        }
    }
}

impl AfAssignCliOpts {
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.af_structure_path.is_some()
    }

    /// Attempt to consume an `--af-*` / `--noesy-15n` / homolog-firewall flag at `args[*idx]`.
    /// Returns `Ok(true)` if matched (advancing `*idx` across its argument value), or `Ok(false)`
    /// if the token belongs to standard `cmd_assign`.
    pub fn try_parse_arg(&mut self, args: &[String], idx: &mut usize) -> Result<bool, String> {
        let flag = args[*idx].as_str();
        let next_val = |cursor: &mut usize, name: &str| -> Result<&str, String> {
            *cursor += 1;
            args.get(*cursor)
                .map(String::as_str)
                .ok_or_else(|| format!("missing value after {name}"))
        };

        match flag {
            "--af-structure" => {
                let v = next_val(idx, flag)?;
                self.af_structure_path = Some(PathBuf::from(v));
                Ok(true)
            }
            "--af-pae" => {
                let v = next_val(idx, flag)?;
                self.af_pae_path = Some(PathBuf::from(v));
                Ok(true)
            }
            "--noesy-15n" => {
                let v = next_val(idx, flag)?;
                self.noesy_15n_path = Some(PathBuf::from(v));
                Ok(true)
            }
            "--af-weight" => {
                let v = next_val(idx, flag)?;
                let w: f64 = v
                    .parse()
                    .map_err(|e| format!("invalid --af-weight {v:?}: {e}"))?;
                if !w.is_finite() || !(0.0..=1.0).contains(&w) {
                    return Err(format!("--af-weight {w} out of allowed bounds [0.0, 1.0]"));
                }
                self.af_weight = Some(w);
                Ok(true)
            }
            "--noe-weight" => {
                let v = next_val(idx, flag)?;
                let w: f64 = v
                    .parse()
                    .map_err(|e| format!("invalid --noe-weight {v:?}: {e}"))?;
                if !w.is_finite() || !(0.0..=1.0).contains(&w) {
                    return Err(format!("--noe-weight {w} out of allowed bounds [0.0, 1.0]"));
                }
                self.noe_weight = Some(w);
                Ok(true)
            }
            "--af-clamp-nats" => {
                let v = next_val(idx, flag)?;
                let c: f64 = v
                    .parse()
                    .map_err(|e| format!("invalid --af-clamp-nats {v:?}: {e}"))?;
                if !c.is_finite() || !(0.25..=25.0).contains(&c) {
                    return Err(format!(
                        "--af-clamp-nats {c} out of allowed bounds [0.25, 25.0] nats"
                    ));
                }
                self.af_clamp_nats = Some(c);
                Ok(true)
            }
            "--exclude-pdb" => {
                let v = next_val(idx, flag)?;
                push_csv_items(&mut self.firewall_query.query_pdb_ids, v);
                self.has_firewall_flag = true;
                Ok(true)
            }
            "--exclude-bmrb" => {
                let v = next_val(idx, flag)?;
                push_csv_items(&mut self.firewall_query.query_bmrb_ids, v);
                self.has_firewall_flag = true;
                Ok(true)
            }
            "--exclude-uniprot" => {
                let v = next_val(idx, flag)?;
                let trimmed = v.trim();
                if !trimmed.is_empty() {
                    self.firewall_query.query_uniprot_id = Some(trimmed.to_string());
                }
                self.has_firewall_flag = true;
                Ok(true)
            }
            "--exclude-family" => {
                let v = next_val(idx, flag)?;
                let trimmed = v.trim();
                if !trimmed.is_empty() {
                    self.firewall_query.query_family_id = Some(trimmed.to_string());
                }
                self.has_firewall_flag = true;
                Ok(true)
            }
            "--exclude-topology" => {
                let v = next_val(idx, flag)?;
                let trimmed = v.trim();
                if !trimmed.is_empty() {
                    self.firewall_query.query_topology_id = Some(trimmed.to_string());
                }
                self.has_firewall_flag = true;
                Ok(true)
            }
            "--max-seq-identity" => {
                let v = next_val(idx, flag)?;
                let sid: f64 = v
                    .parse()
                    .map_err(|e| format!("invalid --max-seq-identity {v:?}: {e}"))?;
                if !sid.is_finite() || !(0.0..=1.0).contains(&sid) || sid == 0.0 {
                    return Err(format!(
                        "--max-seq-identity {sid} out of allowed bounds (0.0, 1.0]"
                    ));
                }
                self.max_seq_identity = Some(sid);
                self.has_firewall_flag = true;
                Ok(true)
            }
            "--force-bfactor-plddt" => {
                self.force_bfactor_plddt = true;
                Ok(true)
            }
            "--treat-as-model" => {
                self.treat_as_model = true;
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    /// Validate flag combinations prior to solving.
    pub fn validate(&self, regime: RegimeKind, solver: SolverKind) -> Result<(), String> {
        if !self.is_enabled() {
            if self.af_pae_path.is_some()
                || self.noesy_15n_path.is_some()
                || self.af_weight.is_some()
                || self.noe_weight.is_some()
                || self.af_clamp_nats.is_some()
                || self.has_firewall_flag
                || self.force_bfactor_plddt
                || self.treat_as_model
            {
                return Err(
                    "--af-pae / --noesy-15n / --af-weight / --noe-weight / --af-clamp-nats / --treat-as-model / --exclude-* require explicit opt-in --af-structure <model.cif|pdb> (backbone assignment is strictly structure-free by default)"
                        .to_string(),
                );
            }
            return Ok(());
        }

        if regime != RegimeKind::Globular {
            return Err(
                "--af-structure is only supported with --regime globular in `ferric assign`"
                    .to_string(),
            );
        }
        if solver == SolverKind::BayesSa {
            return Err(
                "--solver bayes-sa is not supported with --af-structure; use --solver evo-graph (default) or --solver standard"
                    .to_string(),
            );
        }
        Ok(())
    }
}

/// Metadata summary returned alongside [`SolverWorkflowOutput`] when `--af-structure` is active.
#[derive(Clone, Debug)]
pub struct AfWorkflowMetadata {
    pub af_priors: AfAssignPriors,
    pub af_offsets: AfShiftOffsets,
}

impl AfWorkflowMetadata {
    pub fn format_header_suffix(&self) -> String {
        let rep = &self.af_priors.shift_report;
        let cs = &self.af_priors.contact_summary;
        format!(
            "af_priors(uncalibrated_af_prior=true, af_weight={:.3}, noe_weight={:.3}, af_clamp_nats={:.2}, max_retained_seq_id={:.4}, exclusion_sha256={}, af_offsets=[dH={:+.3}, dN={:+.3}, dCA={:+.3}, dCB={:+.3}, dC={:+.3}], noesy_peaks={}, admitted_contacts={}, seq_contacts={}, lr_contacts={}, rejected_high_pae={})",
            self.af_priors.cfg.af_weight,
            self.af_priors.cfg.noe_weight,
            self.af_priors.cfg.af_clamp_nats,
            rep.max_retained_seq_identity,
            rep.exclusion_list_sha256,
            self.af_offsets.h,
            self.af_offsets.n,
            self.af_offsets.ca,
            self.af_offsets.cb,
            self.af_offsets.c,
            cs.noesy_peak_count,
            cs.admitted_contacts_total,
            cs.admitted_sequential_contacts,
            cs.admitted_medium_long_contacts,
            cs.rejected_high_pae_contacts,
        )
    }

    pub fn render_json_block(&self) -> String {
        let rep = &self.af_priors.shift_report;
        let cs = &self.af_priors.contact_summary;
        format!(
            "{{\n\
             \x20   \"uncalibrated_af_prior\": true,\n\
             \x20   \"af_weight\": {:.6},\n\
             \x20   \"noe_weight\": {:.6},\n\
             \x20   \"af_clamp_nats\": {:.6},\n\
             \x20   \"treat_as_model\": {},\n\
             \x20   \"model_evidence\": \"{}\",\n\
             \x20   \"exclusion_list_sha256\": \"{}\",\n\
             \x20   \"max_retained_seq_identity\": {:.6},\n\
             \x20   \"af_offsets\": {{ \"h\": {:.6}, \"n\": {:.6}, \"ca\": {:.6}, \"cb\": {:.6}, \"c\": {:.6} }},\n\
             \x20   \"noesy_15n\": {{\n\
             \x20     \"peak_count\": {},\n\
             \x20     \"spatial_amide_pairs_le_5p5\": {},\n\
             \x20     \"admitted_contacts_total\": {},\n\
             \x20     \"admitted_sequential_contacts\": {},\n\
             \x20     \"admitted_medium_long_contacts\": {},\n\
             \x20     \"rejected_low_plddt_contacts\": {},\n\
             \x20     \"rejected_high_pae_contacts\": {}\n\
             \x20   }}\n\
             \x20 }}",
            self.af_priors.cfg.af_weight,
            self.af_priors.cfg.noe_weight,
            self.af_priors.cfg.af_clamp_nats,
            self.af_priors.cfg.treat_as_model,
            real_benchmark::json_escape(&self.af_priors.model_evidence),
            real_benchmark::json_escape(&rep.exclusion_list_sha256),
            rep.max_retained_seq_identity,
            self.af_offsets.h,
            self.af_offsets.n,
            self.af_offsets.ca,
            self.af_offsets.cb,
            self.af_offsets.c,
            cs.noesy_peak_count,
            cs.total_spatial_amide_pairs_le_5p5,
            cs.admitted_contacts_total,
            cs.admitted_sequential_contacts,
            cs.admitted_medium_long_contacts,
            cs.rejected_low_plddt_contacts,
            cs.rejected_high_pae_contacts,
        )
    }
}

/// Execute opt-in `--af-structure` backbone assignment (`evo-graph` or `standard`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_af_assisted_assignment_workflow(
    opts: &AfAssignCliOpts,
    solver: SolverKind,
    engine_opt: Option<EngineKind>,
    peaks: &[Peak],
    sequence: &[String],
    seq_ids: &[i32],
    priors: &Priors,
    score_cfg: &ScoreConfig,
    tol_overrides: ToleranceOverrides,
    runs_opt: Option<usize>,
    steps_opt: Option<usize>,
    default_cpu_runs: usize,
    seed: u64,
) -> Result<(SolverWorkflowOutput, AfWorkflowMetadata), String> {
    let structure_path = opts
        .af_structure_path
        .as_deref()
        .ok_or_else(|| "internal error: missing --af-structure path".to_string())?;

    let mut af_cfg = AfAssignConfig {
        af_weight: opts.af_weight.unwrap_or(DEFAULT_AF_WEIGHT),
        noe_weight: opts.noe_weight.unwrap_or(DEFAULT_NOE_WEIGHT),
        treat_as_model: opts.treat_as_model,
        af_clamp_nats: opts.af_clamp_nats.unwrap_or(DEFAULT_AF_CLAMP_NATS),
        ..AfAssignConfig::default()
    };
    af_cfg.struct_shift_cfg.force_bfactor_plddt = opts.force_bfactor_plddt;
    if let Some(sid) = opts.max_seq_identity {
        af_cfg.struct_shift_cfg.max_seq_identity = sid;
    }
    tol_overrides_to_af_cfg(tol_overrides, &mut af_cfg);
    af_cfg.validate()?;

    let af_priors = AfAssignPriors::from_files(
        structure_path,
        sequence,
        Some(seq_ids),
        opts.af_pae_path.as_deref(),
        opts.noesy_15n_path.as_deref(),
        &opts.firewall_query,
        &af_cfg,
    )?;

    match solver {
        SolverKind::Standard => {
            let has_noesy_contacts = af_priors.contact_summary.noesy_peak_count > 0
                && af_priors.contact_summary.admitted_contacts_total > 0;
            let requested_engine = if has_noesy_contacts && engine_opt.is_none() {
                EngineKind::Cpu
            } else {
                resolve_solver_engine(solver, engine_opt)?
            };
            if has_noesy_contacts && requested_engine == EngineKind::Gpu {
                return Err(
                    "--engine gpu with --solver standard supports --af-structure unary priors only; omit --engine (or pass --engine cpu / --solver evo-graph) when --noesy-15n contacts are supplied"
                        .to_string(),
                );
            }

            let mut tol = Tolerances::default();
            tol_overrides.apply_to(&mut tol);
            let systems = build_spin_systems(peaks, &tol, 3, false)?;
            let base_detailed = estimate_reference_offsets_detailed(&systems, sequence, priors);
            let (effective_priors, offsets) =
                prepare_scoring_priors(&systems, sequence, priors, score_cfg);
            let detailed_offsets = sync_detailed_reference_offsets(&offsets, &base_detailed);
            let af_offsets = AfShiftOffsets::from_shift_offsets(&offsets);

            let (mut result, engine_label, fit_runs, used_runs, used_steps) = match requested_engine
            {
                EngineKind::Gpu => match GpuEngine::new_headless() {
                    Ok(gpu) => {
                        let anneal_cfg = AnnealConfig {
                            runs: runs_opt.unwrap_or(STANDARD_GPU_CALIBRATED_RUNS),
                            steps_per_system: steps_opt.unwrap_or(1500),
                            seed,
                            ..AnnealConfig::default()
                        };
                        let label = format!(
                            "WebGPU ({}, {}, af-priors)",
                            gpu.adapter_name, gpu.backend_name
                        );
                        let res = gpu.assign_with_af_priors_gpu(
                            &systems,
                            sequence,
                            priors,
                            score_cfg,
                            &anneal_cfg,
                            Some(&af_priors),
                        )?;
                        (
                            res,
                            label,
                            STANDARD_GPU_CALIBRATED_RUNS,
                            anneal_cfg.runs,
                            anneal_cfg.steps_per_system,
                        )
                    }
                    Err(err) if engine_opt.is_none() => {
                        eprintln!(
                            "Warning: WebGPU unavailable for --af-structure ({err}); falling back to CPU (f64, af-priors)."
                        );
                        let anneal_cfg = AnnealConfig {
                            runs: runs_opt.unwrap_or(default_cpu_runs),
                            steps_per_system: steps_opt.unwrap_or(1500),
                            seed,
                            ..AnnealConfig::default()
                        };
                        let res = assign_with_af_priors(
                            &systems,
                            sequence,
                            priors,
                            score_cfg,
                            &anneal_cfg,
                            Some(&af_priors),
                        );
                        (
                            res,
                            "CPU (f64, af-priors, auto-fallback)".to_string(),
                            STANDARD_CPU_CALIBRATED_RUNS,
                            anneal_cfg.runs,
                            anneal_cfg.steps_per_system,
                        )
                    }
                    Err(err) => return Err(err),
                },
                EngineKind::Cpu => {
                    let anneal_cfg = AnnealConfig {
                        runs: runs_opt.unwrap_or(default_cpu_runs),
                        steps_per_system: steps_opt.unwrap_or(1500),
                        seed,
                        ..AnnealConfig::default()
                    };
                    let res = assign_with_af_priors(
                        &systems,
                        sequence,
                        priors,
                        score_cfg,
                        &anneal_cfg,
                        Some(&af_priors),
                    );
                    (
                        res,
                        "CPU (f64, af-priors)".to_string(),
                        STANDARD_CPU_CALIBRATED_RUNS,
                        anneal_cfg.runs,
                        anneal_cfg.steps_per_system,
                    )
                }
            };

            let mut qc_report = evaluate_dataset_qc(
                Some(peaks),
                &systems,
                sequence,
                priors,
                score_cfg,
                &result,
                Some(&detailed_offsets),
            );
            if used_runs != fit_runs {
                qc_report.mark_uncalibrated_runs(&engine_label, used_runs, fit_runs);
            }
            qc_report.confidence_calibrated = false;
            let af_warn = format!(
                "uncalibrated_af_prior=true: 3D AlphaFold shift/NOE priors active on {engine_label}; sequence-only isotonic calibration knots were not refit with AF priors (confidence_calibrated=false)"
            );
            qc_report.calibration_warning = Some(match qc_report.calibration_warning.take() {
                Some(prev) => format!("{prev}; {af_warn}"),
                None => af_warn.clone(),
            });
            if !qc_report
                .reasons
                .iter()
                .any(|r| r.contains("uncalibrated_af_prior"))
            {
                qc_report.reasons.push("uncalibrated_af_prior=true".to_string());
            }
            eprintln!("Warning: [{engine_label}] {af_warn}");
            apply_dataset_qc_to_assignments(&mut result, &qc_report);
            qc_report.already_applied = true;

            Ok((
                SolverWorkflowOutput {
                    solver,
                    systems,
                    kept_hsqc_indices: None,
                    result,
                    effective_priors,
                    offsets,
                    detailed_offsets,
                    qc_report,
                    engine_label,
                    replicas: used_runs,
                    steps_per_system: used_steps,
                    bayes_report: None,
                },
                AfWorkflowMetadata {
                    af_priors,
                    af_offsets,
                },
            ))
        }
        SolverKind::EvoGraph => {
            let _ = resolve_solver_engine(solver, engine_opt)?;
            let mut evo_cfg = EvoGraphConfig {
                score_cfg: score_cfg.clone(),
                auto_reference: score_cfg.auto_reference,
                locked_assignments: score_cfg.locked_assignments.clone(),
                structure_torsions: None,
                af_priors: Some(&af_priors),
                seed,
                ..EvoGraphConfig::default()
            };
            tol_overrides.apply_to(&mut evo_cfg.tolerances);
            if let Some(r) = runs_opt {
                evo_cfg.population_size = r.max(4);
                evo_cfg.num_islands = evo_cfg.num_islands.min(evo_cfg.population_size).max(1);
            }
            if let Some(st) = steps_opt {
                evo_cfg.steps_per_system_per_gen = st.max(1);
            }
            let (systems, result, mut rep) =
                assign_evo_graph_from_peaks(peaks, sequence, priors, &evo_cfg);
            let engine_label = "CPU (f64, evo-graph, af-priors)".to_string();
            let requested_runs = runs_opt.unwrap_or(evo_cfg.population_size);
            if requested_runs != EVO_GRAPH_CALIBRATED_RUNS {
                rep.qc_report.mark_uncalibrated_runs(
                    &engine_label,
                    requested_runs,
                    EVO_GRAPH_CALIBRATED_RUNS,
                );
            }
            rep.qc_report.confidence_calibrated = false;
            if let Some(ref w) = rep.qc_report.calibration_warning {
                eprintln!("Warning: [{engine_label}] {w}");
            }
            let detailed_offsets = rep.qc_report.detailed_offsets;
            let offsets = ShiftOffsets {
                h: rep.final_offsets.d_h,
                n: rep.final_offsets.d_n,
                ca: rep.final_offsets.d_ca,
                cb: rep.final_offsets.d_cb,
                c: rep.final_offsets.d_c,
            };
            let af_offsets = AfShiftOffsets::from_shift_offsets(&offsets);
            let effective_priors = rep.shifted_priors(priors);
            let replicas = evo_cfg.population_size;
            let steps_per_system = evo_cfg.generations * evo_cfg.steps_per_system_per_gen;

            Ok((
                SolverWorkflowOutput {
                    solver,
                    systems,
                    kept_hsqc_indices: Some(rep.kept_hsqc_indices),
                    result,
                    effective_priors,
                    offsets,
                    detailed_offsets,
                    qc_report: rep.qc_report,
                    engine_label,
                    replicas,
                    steps_per_system,
                    bayes_report: None,
                },
                AfWorkflowMetadata {
                    af_priors,
                    af_offsets,
                },
            ))
        }
        SolverKind::BayesSa => Err(
            "--solver bayes-sa is not supported with --af-structure; use --solver evo-graph or --solver standard"
                .to_string(),
        ),
    }
}

fn tol_overrides_to_af_cfg(tol: ToleranceOverrides, af_cfg: &mut AfAssignConfig) {
    if let Some(th) = tol.h {
        af_cfg.noesy_strip_tol_hn.0 = th.max(0.02);
        af_cfg.noesy_indirect_cutoff_h_ppm = (th * 1.35).max(0.03);
    }
    if let Some(tn) = tol.n {
        af_cfg.noesy_strip_tol_hn.1 = tn.max(0.15);
    }
}

/// Format `.str` / `.nef` provenance header when `--af-structure` is active.
pub(crate) fn format_af_star_header(
    base_header: Option<String>,
    meta: &AfWorkflowMetadata,
    wf: &SolverWorkflowOutput,
) -> String {
    let suffix = meta.format_header_suffix();
    match base_header {
        Some(hdr) => format!("{hdr} | {suffix}"),
        None => format!(
            "{} | solver={} | qc_verdict={} | confidence_calibrated={} | {suffix}",
            version_string(),
            wf.engine_label,
            wf.qc_report.verdict,
            wf.qc_report.confidence_calibrated,
        ),
    }
}

/// Inject `"af_priors": { ... }` into `render_assign_workflow_json` output when `--af-structure` is active.
pub(crate) fn inject_af_json_metadata(base_json: &str, meta: &AfWorkflowMetadata) -> String {
    let trimmed = base_json.trim_end();
    if let Some(stripped) = trimmed.strip_suffix('}') {
        let stripped = stripped.trim_end();
        format!(
            "{stripped},\n  \"af_priors\": {}\n}}",
            meta.render_json_block()
        )
    } else {
        base_json.to_string()
    }
}
