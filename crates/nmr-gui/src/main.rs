//! Interactive WebGPU + `eframe`/`egui` NMR spectrum and backbone resonance assignment workbench.
//!
//! Provenance: Original scientific GUI built on `eframe`, `wgpu`, `nmr-core`, and `nmr-gpu`
//! (native Metal/Vulkan/DX12 and browser `wasm32-unknown-unknown` + WebGPU).

#![cfg_attr(
    all(target_os = "windows", not(debug_assertions)),
    windows_subsystem = "windows"
)]

#[allow(dead_code)]
mod octet_canvas;

use std::collections::{BTreeMap, BTreeSet};
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

use eframe::{egui, egui_wgpu};
use nmr_core::aromatic::{
    assign_aromatic_rings, estimate_unlabelled_yamazaki_cb_offset,
    merge_aromatic_assignments_preserving_existing, AromaticRingCluster, AromaticSolverConfig,
    AROMATIC_EXPERIMENTS,
};
use nmr_core::bayes_sa::{
    solve_bayes_sa, solve_bayes_sa_from_peaks, BayesSaConfig, BayesSaReport, LockedAssignment,
};
use nmr_core::c13_detected::{
    pick_caco_2d, pick_con_2d, pick_con_anchored_3d, simulate_idp_13c_cubes, C13DetectedExp,
    C13PickedDataset, C13PickerConfig, C13Strip3DPeak, Caco2DPeak, CacoPeakKind, Con2DPeak,
    IdpCubeSimConfig,
};
use nmr_core::classify::{
    DataState, ExperimentDimensionality, ExperimentVendor, ProblemRegime, PulseSequenceClass,
};
use nmr_core::evaluate::{score, to_reference, Metrics};
use nmr_core::evo_graph::{assign_evo_graph_from_peaks, EvoGraphConfig};
use nmr_core::fid::read_bruker_fid_2d_from_bytes;
use nmr_core::idp_propensity::{compute_idp_propensity, PropensityConfig, PropensityReport};
use nmr_core::idp_solver::{
    assign_idp_13c_cpu, idp_to_reference, register_and_build_con_dipeptide_systems, IdpSolverConfig,
};
use nmr_core::nef::read_nef_str;
use nmr_core::neural_picker::{pick_hsqc_2d_neural, NeuralPickerConfig};
#[cfg(not(target_arch = "wasm32"))]
use nmr_core::nmrstar::{read_star, read_star_all_atoms, write_star};
use nmr_core::nmrstar::{read_star_str, read_star_str_all_atoms, Reference};
use nmr_core::octet_analyt::iupac_naming::{iupac_name_to_mol, mol_to_iupac_name};
use nmr_core::octet_analyt::jcamp_spec1d::{
    pick_1d_multiplets, read_jcamp, synthesize_1d_nmr, write_jcamp_nmr, JcampCompression,
    Synthesize1dOptions,
};
use nmr_core::octet_analyt::mass_props::compute_phys_chem_props;
use nmr_core::octet_analyt::nmr_predict::{predict_nmr, NmrNucleus};
use nmr_core::octet_analyt::AnalytMol;
use nmr_core::octet_mol::{
    build_peptide_2d, clean_2d, expand_abbreviations, kekulize, parse_biopolymer_sequence,
    parse_condensed_formula, parse_inchi, parse_reaction_smiles, parse_smiles, perceive_sssr,
    read_cdx, read_cdxml, read_mol, read_oct, read_sdf, write_canonical_smiles, write_cdx,
    write_cdxml, write_inchi, write_inchikey, write_mol_v2000, write_oct, write_reaction_smiles,
    write_sdf, ChemDocument, Molecule, SdfRecord,
};
use nmr_core::octet_render::{
    compile_scene_primitives, flatten_path_commands, render_scene_svg, BondOrder, BondStereo,
    DocumentStyle, RenderScene, RgbaColor, SceneAtom, SceneBond, SceneSpectrum1D, SpectrumSignal1D,
    VectorPrimitive,
};
use nmr_core::paramagnetic::{
    assign_paramagnetic_backbone, assign_paramagnetic_from_peaks, parse_metal_ligands,
    ParamagneticAssignConfig, ParamagneticSolverMode,
};
#[cfg(not(target_arch = "wasm32"))]
use nmr_core::peaks::read_peaks;
use nmr_core::peaks::{read_peaks_str, Peak, EXPERIMENT_NAMES};
use nmr_core::picker::{
    estimate_noise_mad, pick_c13_hsqc_2d, pick_hsqc_2d, pick_triple_3d_two_pass,
    CorroboratedRepickConfig, PickerConfig,
};
use nmr_core::priors::{one_to_three, three_to_one, Priors, ATOMS};
use nmr_core::process::{
    process_fid_2d_mut, register_3d_hn_to_hsqc_2d_with_audit, AxisRescueAuditRecord,
    ProcessingRecipe,
};
#[cfg(not(target_arch = "wasm32"))]
use nmr_core::process::{process_fid_dir_2d, process_fid_dir_3d};
use nmr_core::qc::{evaluate_dataset_qc, DatasetQcReport};
use nmr_core::report::{generate_quality_report_html, predict_csi_dihedrals, ReportInput};
use nmr_core::rng::Rng;
use nmr_core::session::{
    ResidueAssignmentState, SessionContourParams, WorkbenchCommand, WorkbenchSessionState,
};
use nmr_core::sidechain::{
    assign_sidechains, solve_joint_backbone_sidechain, SidechainSolverConfig,
};
use nmr_core::simulate::{
    render_hsqc_2d, render_triple_3d, simulate, SimConfig, SpectrumRenderConfig,
};
use nmr_core::solver::{
    assign_from_matrices, pair_matrix, unary_matrix, AnnealConfig, AssignResult, Matrix2D,
    ScoreConfig, FORBID,
};
#[cfg(not(target_arch = "wasm32"))]
use nmr_core::spectrum::{
    read_bruker_pdata_2d, read_bruker_pdata_3d, read_nmrpipe_2d, read_nmrpipe_3d, read_ucsf_2d,
    read_ucsf_3d, resolve_bruker_pdata_dir,
};
use nmr_core::spectrum::{
    read_bruker_pdata_2d_from_bytes, read_bruker_pdata_3d_from_bytes, read_nmrpipe_2d_from_bytes,
    read_nmrpipe_3d_from_bytes, read_ucsf_2d_from_bytes, read_ucsf_3d_from_bytes, Spectrum2D,
    Spectrum3D,
};
use nmr_core::spinsystems::{build_spin_systems, SpinSystem, Tolerances};
use nmr_core::stereospec::{
    assign_stereospecific_dataset, StereospecConfig, StereospecDatasetInput,
};
use nmr_core::topology::SidechainPriors;
#[cfg(not(target_arch = "wasm32"))]
use nmr_gpu::bayes_sa_gpu::BayesSaGpuSolver;
#[cfg(not(target_arch = "wasm32"))]
use nmr_gpu::GpuEngine;
#[cfg(not(target_arch = "wasm32"))]
use nmr_gui::ingest_panel::{
    ingest_dropped_files_native, load_native_path_1d_or_workspace, sanitize_user_path_input,
};
#[cfg(target_arch = "wasm32")]
use nmr_gui::ingest_panel::{
    ingest_dropped_files_wasm, trigger_browser_files_picker, trigger_browser_folder_picker,
};
#[cfg(target_arch = "wasm32")]
use nmr_gui::session_io::{default_session_nef_filename, trigger_browser_nef_download};
use nmr_gui::{
    command_palette::CommandPalette,
    contour_painter::{
        apply_pinch_zoom_and_pan_2d, clamp_view_ppm_2d, default_overlay_palette,
        ContourPipelineResource, MultiContourCallback, OverlayLayerSpec, SyncCrosshair,
        ViewBounds2dLimits,
    },
    demo_preset::{build_builtin_demo_suite, BUILTIN_DEMO_SUITE_SEED},
    dosy_panel::{render_dosy_panel, DosyPanelAction, DosyPanelState},
    dynamics_panel::{DynamicsStudioOutput, DynamicsStudioPanel, DynamicsStudioTab},
    ingest_panel::{
        ingest_dropped_payloads, install_browser_folder_drop_hook, is_bare_bruker_file_error,
        poll_browser_drop_error, poll_browser_dropped_payloads, try_load_1d_ucsf_or_pipe_bytes,
        IngestAction, IngestDropOutcome, IngestExecutionPlan, IngestPanelState,
    },
    keymap::{poll_global_modifier_shortcuts, ChordDispatcher, CONTOUR_FLOOR_STEP_RATIO},
    mcp_bridge::{render_mcp_bridge_panel, McpBridgeState, McpUiCommand, ViewportFrameBuffer},
    metabolomics_panel::MetabolomicsStudioPanel,
    mol3d_overlay::{Mol3dSyncAction, QuantitativeTrack},
    mol3d_viewer::{Mol3dPipelineResource, Mol3dViewerPanel},
    relax_panel::{RelaxPanelAction, RelaxPanelState},
    session_io::WorkbenchSessionController,
    sidechain_widget::{
        AssignedAtomInfo, SequenceRibbonState, SidechainInspectorPanel, SidechainWorkspaceInput,
    },
    stack1d_panel::{
        build_kinetics_demo_traces, export_stack1d_svg, render_stack1d_panel, AttachedSpectrum1d,
        InspectorTab1d, Stack1dState, Stack1dViewMode, ToolMode1d, Trace1d,
    },
    strip_tiler::{StripSpectra3D, StripTilerAction, StripTilerContext, StripTilerPanel},
    workbook::{
        build_showcase_small_molecule_13c_trace, build_showcase_small_molecule_1h_trace,
        build_showcase_small_molecule_cosy_2d, chrome_icon_button, contour_palette_for_theme,
        estimate_contour_base_and_noise, evaluate_2d_marginals, f1_ppm_to_screen_y,
        f2_ppm_to_screen_x, paint_f1_marginal_strip, paint_f2_marginal_strip, paint_icon,
        rail_tool_button, render_workbook_sidebar, ActiveWorkbookSelection, CanvasTheme2D,
        ToolMode2d, WorkbookIcon, WorkbookSidebarAction, WorkbookState, WorkbookStudioKind,
        CANVAS_TOOL_RAIL_WIDTH, CONTOUR_SCROLL_STEP_RATIO, MARGINAL_2D_STRIP_PX,
    },
    workflow_bar::{
        draw_top_menu_bar, draw_workflow_stepper, render_responsive_workbench, HostWorkspaceMode,
        MobileChemTab, MobileWorkspaceFamily, ResponsiveShellState, ResponsiveWorkbenchHost,
        TabletActiveDock, WorkbenchAction, WorkspaceReadiness, MOBILE_TOUCH_MIN_HEIGHT,
    },
};
use octet_canvas::{IoFormat, MolecularPropertiesSummary, OctetBackendAdapter, OctetDrawPanel};

const UBIQUITIN_SEQ: &str =
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG";

const SYNTHETIC_UBIQUITIN_STR: &str =
    include_str!("../../../data/fixtures/synthetic_ubiquitin.str");

const DEMO_UBIQUITIN_1UBQ_PDB: &str =
    include_str!("../../../data/fixtures/ubiquitin_1ubq_xray.pdb");

struct PerfTimer {
    #[cfg(not(target_arch = "wasm32"))]
    start: Instant,
    #[cfg(target_arch = "wasm32")]
    start_ms: f64,
}

impl PerfTimer {
    fn start() -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
            Self {
                start: Instant::now(),
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            let start_ms = web_sys::window()
                .and_then(|w| w.performance())
                .map(|p| p.now())
                .unwrap_or(0.0);
            Self { start_ms }
        }
    }

    fn elapsed_ms(&self) -> f64 {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.start.elapsed().as_secs_f64() * 1000.0
        }
        #[cfg(target_arch = "wasm32")]
        {
            let now_ms = web_sys::window()
                .and_then(|w| w.performance())
                .map(|p| p.now())
                .unwrap_or(self.start_ms);
            (now_ms - self.start_ms).max(0.0)
        }
    }
}

fn build_synthetic_ubiquitin() -> Reference {
    #[cfg(not(target_arch = "wasm32"))]
    if let Ok(r) = read_star("data/fixtures/synthetic_ubiquitin.str", None) {
        return r;
    }
    if let Ok(r) = read_star_str(SYNTHETIC_UBIQUITIN_STR, None) {
        return r;
    }
    let pri = Priors::provisional();
    let mut rng = Rng::new(1);
    let seq: Vec<String> = UBIQUITIN_SEQ
        .chars()
        .map(|c| one_to_three(c).to_string())
        .collect();
    let seq_ids: Vec<i32> = (1..=(seq.len() as i32)).collect();
    let mut r = Reference::new(seq.clone(), seq_ids);
    for (k, res) in seq.iter().enumerate() {
        if let Some(atoms) = pri.table.get(res) {
            for (atom, &(mu, sd)) in atoms {
                if atom == "H" && (k == 0 || res == "PRO") {
                    continue;
                }
                r.shifts.insert((k, atom.clone()), rng.normal(mu, sd));
            }
        }
    }
    r
}

fn format_star_in_memory(
    ref_data: &Reference,
    confidence: Option<&BTreeMap<usize, f64>>,
) -> String {
    let mut lines = vec![
        "data_ferric_output".to_string(),
        String::new(),
        "save_assigned_chem_shift_list_1".to_string(),
        "   _Assigned_chem_shift_list.Sf_category  assigned_chemical_shifts".to_string(),
        String::new(),
        "   loop_".to_string(),
    ];
    for t in [
        "ID",
        "Entity_ID",
        "Seq_ID",
        "Comp_ID",
        "Atom_ID",
        "Atom_type",
        "Val",
        "Ambiguity_code",
        "Details",
    ] {
        lines.push(format!("      _Atom_chem_shift.{t}"));
    }
    lines.push(String::new());

    let mut entries: Vec<(usize, &str, f64)> = ref_data
        .shifts
        .iter()
        .map(|(&(idx, ref atom), &v)| (idx, atom.as_str(), v))
        .collect();
    entries.sort_by(|&(ia, aa, _), &(ib, ab, _)| {
        let oa = ATOMS.iter().position(|&a| a == aa).unwrap_or(99);
        let ob = ATOMS.iter().position(|&a| a == ab).unwrap_or(99);
        (ia, oa, aa).cmp(&(ib, ob, ab))
    });

    for (i, (idx, atom, v)) in entries.into_iter().enumerate() {
        let det = confidence
            .and_then(|c| c.get(&idx))
            .map(|&c| format!("conf={c:.2}"))
            .unwrap_or_else(|| ".".to_string());
        let atom_type = atom
            .chars()
            .find(|c| c.is_ascii_alphabetic())
            .map(|c| c.to_string())
            .unwrap_or_else(|| "X".to_string());
        lines.push(format!(
            "      {:5} 1 {:5} {:3} {:4} {} {:9.3} 1 {}",
            i + 1,
            ref_data.seq_ids[idx],
            ref_data.sequence[idx],
            atom,
            atom_type,
            v,
            det
        ));
    }
    lines.push("   stop_".to_string());
    lines.push("save_".to_string());
    lines.push(String::new());
    lines.join("\n")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WorkspaceMode {
    ProteinNmr,
    OctetChem,
    Dosy,
    Metabolomics,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum GuiSolverKind {
    #[default]
    EvoGraph,
    Standard,
    BayesSa,
}

impl GuiSolverKind {
    fn label(self) -> &'static str {
        match self {
            Self::EvoGraph => "Evo-Graph",
            Self::Standard => "Standard",
            Self::BayesSa => "Bayesian SA (Cv Adaptive)",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum GuiRegimeKind {
    #[default]
    Globular,
    Idp13C,
    Paramagnetic,
}

impl GuiRegimeKind {
    fn label(self) -> &'static str {
        match self {
            Self::Globular => "Globular",
            Self::Idp13C => "13C-IDP",
            Self::Paramagnetic => "Paramagnetic",
        }
    }
}

fn parse_gui_c13_tsv_str(text: &str) -> Option<C13PickedDataset> {
    let mut ds = C13PickedDataset::default();
    let mut saw_c13_row = false;
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').map(str::trim).collect();
        let cols: Vec<&str> = if cols.len() <= 1 {
            line.split_whitespace().collect()
        } else {
            cols
        };
        if cols.len() < 3 {
            continue;
        }
        if matches!(
            cols[0].to_ascii_lowercase().as_str(),
            "experiment" | "exp" | "spectrum" | "peak_id"
        ) {
            continue;
        }
        let exp = C13DetectedExp::from_name(cols[0])?;
        saw_c13_row = true;
        let rem = &cols[1..];
        let opt_f64 = |s: &str| -> Option<f64> {
            let t = s.trim();
            if t.is_empty() || t == "." || t == "-" {
                None
            } else {
                t.parse::<f64>().ok()
            }
        };
        match exp {
            C13DetectedExp::Con2D => {
                let c_prev = opt_f64(rem.first().copied()?)?;
                let n_curr = opt_f64(rem.get(1).copied()?)?;
                ds.con_roots.push(Con2DPeak {
                    c_prev_ppm: c_prev,
                    n_curr_ppm: n_curr,
                    height: 1.0e5,
                    volume: 1.0e5,
                    is_proline_imino_region: n_curr >= 131.5,
                    deconvolved_from_cluster: false,
                });
            }
            C13DetectedExp::Caco2D => {
                let f0 = opt_f64(rem.first().copied()?)?;
                let ca = opt_f64(rem.get(1).copied()?).or_else(|| opt_f64(rem.get(2).copied()?))?;
                let (c_curr_ppm, ca_curr_ppm) = if f0 < 100.0 && ca > 140.0 {
                    (ca, f0)
                } else {
                    (f0, ca)
                };
                let kind = if ca_curr_ppm <= 47.8 {
                    CacoPeakKind::GlyRegion
                } else if ca_curr_ppm >= 58.0 {
                    CacoPeakKind::SerThrProDownfield
                } else {
                    CacoPeakKind::GeneralAliphatic
                };
                ds.caco_peaks.push(Caco2DPeak {
                    c_curr_ppm,
                    ca_curr_ppm,
                    sign: 1,
                    height: 1.0e5,
                    volume: 1.0e5,
                    kind,
                });
            }
            _ => {
                let c_prime_ppm = opt_f64(rem.first().copied()?)?;
                let n_ppm = opt_f64(rem.get(1).copied()?)?;
                let c_indirect_ppm = opt_f64(rem.get(2).copied()?)?;
                let sign = rem.get(3).and_then(|s| s.parse::<i32>().ok()).unwrap_or(1);
                let anchor_root_idx = rem.get(6).and_then(|s| s.parse::<usize>().ok());
                ds.strip_peaks.push(C13Strip3DPeak {
                    exp,
                    c_prime_ppm,
                    n_ppm,
                    c_indirect_ppm,
                    sign,
                    height: 1.0e5,
                    volume: 1.0e5,
                    anchor_root_idx,
                });
            }
        }
    }
    if !saw_c13_row || ds.con_roots.is_empty() {
        return None;
    }
    // Anchor any unanchored CANCO/CBCACON/CCCON peaks onto nearest CON root
    for p in &mut ds.strip_peaks {
        if p.anchor_root_idx.is_none() && p.exp != C13DetectedExp::Ncaco3D {
            let mut best: Option<(usize, f64)> = None;
            for (r_idx, r) in ds.con_roots.iter().enumerate() {
                let dc = (p.c_prime_ppm - r.c_prev_ppm).abs();
                let dn = (p.n_ppm - r.n_curr_ppm).abs();
                if dc <= 0.25 && dn <= 0.65 {
                    let d2 = (dc / 0.06).powi(2) + (dn / 0.25).powi(2);
                    if best.is_none_or(|(_, b)| d2 < b) {
                        best = Some((r_idx, d2));
                    }
                }
            }
            p.anchor_root_idx = best.map(|(i, _)| i);
        }
    }
    Some(ds)
}

#[derive(Clone, Copy, Debug, Default)]
struct ClusterCuratedEval {
    total_cur_obs: usize,
    total_clu_obs: usize,
    shared_obs: usize,
    conflict_obs: usize,
    cb_contradicts: bool,
    norm_dist_sq: f64,
}

fn eval_cluster_vs_curated_residue(
    clu: &AromaticRingCluster,
    k_cur: usize,
    curated_ref: &Reference,
    cfg: &AromaticSolverConfig,
) -> ClusterCuratedEval {
    let res_name = curated_ref
        .sequence
        .get(k_cur)
        .map(String::as_str)
        .unwrap_or("");
    if clu.kind.res_name() != res_name {
        return ClusterCuratedEval {
            conflict_obs: 1,
            ..ClusterCuratedEval::default()
        };
    }

    let tol_h = (cfg.tol_h * 2.0).max(0.055);
    let tol_c = (cfg.tol_c * 1.6).max(0.45);
    let tol_n = (cfg.tol_n * 1.6).max(0.65);

    let logical_schema: &[(&str, Option<&str>, f64)] = match res_name {
        "PHE" => &[
            ("CD1", Some("CD2"), tol_c),
            ("HD1", Some("HD2"), tol_h),
            ("CE1", Some("CE2"), tol_c),
            ("HE1", Some("HE2"), tol_h),
            ("CZ", None, tol_c),
            ("HZ", None, tol_h),
        ],
        "TYR" => &[
            ("CD1", Some("CD2"), tol_c),
            ("HD1", Some("HD2"), tol_h),
            ("CE1", Some("CE2"), tol_c),
            ("HE1", Some("HE2"), tol_h),
        ],
        "TRP" => &[
            ("CD1", None, tol_c),
            ("HD1", None, tol_h),
            ("CE3", None, tol_c),
            ("HE3", None, tol_h),
            ("CZ3", None, tol_c),
            ("HZ3", None, tol_h),
            ("CH2", None, tol_c),
            ("HH2", None, tol_h),
            ("CZ2", None, tol_c),
            ("HZ2", None, tol_h),
            ("NE1", None, tol_n),
            ("HE1", None, tol_h),
        ],
        "HIS" => &[
            ("CD2", None, tol_c),
            ("HD2", None, tol_h),
            ("CE1", None, tol_c),
            ("HE1", None, tol_h),
            ("ND1", None, tol_n),
            ("NE2", None, tol_n),
        ],
        _ => &[],
    };

    let clu_val_for = |primary: &str| -> Option<f64> {
        match (res_name, primary) {
            ("PHE" | "TYR", "CD1") => clu.ch_slots.get("CD1").map(|s| s.c_ppm),
            ("PHE" | "TYR", "HD1") => clu.ch_slots.get("CD1").map(|s| s.h_ppm),
            ("PHE" | "TYR", "CE1") => clu.ch_slots.get("CE1").map(|s| s.c_ppm),
            ("PHE" | "TYR", "HE1") => clu.ch_slots.get("CE1").map(|s| s.h_ppm),
            ("PHE", "CZ") => clu.ch_slots.get("CZ").map(|s| s.c_ppm),
            ("PHE", "HZ") => clu.ch_slots.get("CZ").map(|s| s.h_ppm),
            ("TRP", "CD1") => clu.ch_slots.get("CD1").map(|s| s.c_ppm),
            ("TRP", "HD1") => clu.ch_slots.get("CD1").map(|s| s.h_ppm),
            ("TRP", "CE3") => clu.ch_slots.get("CE3").map(|s| s.c_ppm),
            ("TRP", "HE3") => clu.ch_slots.get("CE3").map(|s| s.h_ppm),
            ("TRP", "CZ3") => clu.ch_slots.get("CZ3").map(|s| s.c_ppm),
            ("TRP", "HZ3") => clu.ch_slots.get("CZ3").map(|s| s.h_ppm),
            ("TRP", "CH2") => clu.ch_slots.get("CH2").map(|s| s.c_ppm),
            ("TRP", "HH2") => clu.ch_slots.get("CH2").map(|s| s.h_ppm),
            ("TRP", "CZ2") => clu.ch_slots.get("CZ2").map(|s| s.c_ppm),
            ("TRP", "HZ2") => clu.ch_slots.get("CZ2").map(|s| s.h_ppm),
            ("TRP", "NE1") => clu.trp_indole.map(|(n, _, _)| n),
            ("TRP", "HE1") => clu.trp_indole.map(|(_, h, _)| h),
            ("HIS", "CD2") => clu.ch_slots.get("CD2").map(|s| s.c_ppm),
            ("HIS", "HD2") => clu.ch_slots.get("CD2").map(|s| s.h_ppm),
            ("HIS", "CE1") => clu.ch_slots.get("CE1").map(|s| s.c_ppm),
            ("HIS", "HE1") => clu.ch_slots.get("CE1").map(|s| s.h_ppm),
            ("HIS", "ND1") => clu.nd1_ppm.map(|(n, _)| n),
            ("HIS", "NE2") => clu.ne2_ppm.map(|(n, _)| n),
            _ => None,
        }
    };

    let mut out = ClusterCuratedEval::default();
    for &(a1, a2_opt, tol) in logical_schema {
        let cur1 = curated_ref.shifts.get(&(k_cur, a1.to_string())).copied();
        let cur2 = a2_opt.and_then(|a2| curated_ref.shifts.get(&(k_cur, a2.to_string())).copied());
        let has_cur = cur1.is_some() || cur2.is_some();
        if has_cur {
            out.total_cur_obs += 1;
        }
        let Some(cv) = clu_val_for(a1) else {
            continue;
        };
        out.total_clu_obs += 1;
        if !has_cur {
            continue;
        }
        let best_diff = [cur1, cur2]
            .into_iter()
            .flatten()
            .map(|v| (cv - v).abs())
            .fold(f64::INFINITY, f64::min);
        if best_diff <= tol {
            out.shared_obs += 1;
            out.norm_dist_sq += (best_diff / tol).powi(2);
        } else {
            out.conflict_obs += 1;
        }
    }

    if let Some(&cur_cb) = curated_ref.shifts.get(&(k_cur, "CB".to_string())) {
        let cbs: Vec<f64> = [clu.cb_hd, clu.cb_he].into_iter().flatten().collect();
        if !cbs.is_empty()
            && cbs
                .iter()
                .all(|&cb| (cb - cur_cb).abs() > cfg.tol_cb * 1.20)
        {
            out.cb_contradicts = true;
        }
    }

    out
}

fn cluster_belongs_to_curated(ev: &ClusterCuratedEval) -> bool {
    if ev.cb_contradicts || ev.total_clu_obs == 0 {
        return false;
    }
    let min_corroborate = ev.total_cur_obs.clamp(1, 2);
    if ev.conflict_obs == 0 {
        ev.shared_obs == ev.total_clu_obs || ev.shared_obs >= min_corroborate
    } else {
        ev.conflict_obs <= 1 && ev.shared_obs >= 2 && ev.shared_obs > ev.conflict_obs
    }
}

fn compute_yamazaki_cb_offsets(
    backbone_ref: &Reference,
    backbone_conf: Option<&BTreeMap<usize, f64>>,
    aromatic_peaks: &[Peak],
    cfg: &AromaticSolverConfig,
) -> (f64, f64) {
    let d_joint = estimate_unlabelled_yamazaki_cb_offset(
        backbone_ref,
        backbone_conf,
        aromatic_peaks,
        None,
        cfg.min_backbone_conf,
    );
    let d_hd_raw = estimate_unlabelled_yamazaki_cb_offset(
        backbone_ref,
        backbone_conf,
        aromatic_peaks,
        Some("HBCBCGCDHD"),
        cfg.min_backbone_conf,
    );
    let d_he_raw = estimate_unlabelled_yamazaki_cb_offset(
        backbone_ref,
        backbone_conf,
        aromatic_peaks,
        Some("HBCBCGCDCEHE"),
        cfg.min_backbone_conf,
    );
    let d_hd = if d_hd_raw != 0.0 { d_hd_raw } else { d_joint };
    let d_he = if d_he_raw != 0.0 { d_he_raw } else { d_joint };
    (d_hd, d_he)
}

fn peak_matches_aromatic_cluster(
    p: &Peak,
    clu: &AromaticRingCluster,
    cb_offset_hd: f64,
    cb_offset_he: f64,
    cfg: &AromaticSolverConfig,
) -> bool {
    let tol_h = (cfg.tol_h * 1.5).max(0.045);
    let tol_c = (cfg.tol_c * 1.5).max(0.42);
    let tol_n = (cfg.tol_n * 1.5).max(0.55);
    match p.exp.as_str() {
        "C13HSQC" => clu.ch_slots.values().any(|s| {
            (p.h - s.h_ppm).abs() <= tol_h && p.c.is_some_and(|c| (c - s.c_ppm).abs() <= tol_c)
        }),
        "AROM_HCCH_TOCSY" | "HCCHTOCSY" => {
            clu.ch_slots
                .values()
                .any(|s| (p.h - s.h_ppm).abs() <= tol_h)
                && clu
                    .ch_slots
                    .values()
                    .any(|s| p.c.is_some_and(|c| (c - s.c_ppm).abs() <= tol_c))
        }
        "HBCBCGCDHD" => {
            clu.ch_slots
                .get("CD1")
                .or_else(|| clu.ch_slots.get("CD2"))
                .is_some_and(|s| (p.h - s.h_ppm).abs() <= tol_h * 1.25)
                && clu.cb_hd.is_some_and(|cb| {
                    p.c.is_some_and(|c| {
                        ((c - cb_offset_hd) - cb).abs() <= cfg.tol_cb * 1.15
                            || (c - cb).abs() <= cfg.tol_cb * 1.15
                    })
                })
        }
        "HBCBCGCDCEHE" => {
            clu.ch_slots
                .get("CE1")
                .or_else(|| clu.ch_slots.get("CE3"))
                .is_some_and(|s| (p.h - s.h_ppm).abs() <= tol_h * 1.25)
                && clu.cb_he.is_some_and(|cb| {
                    p.c.is_some_and(|c| {
                        ((c - cb_offset_he) - cb).abs() <= cfg.tol_cb * 1.15
                            || (c - cb).abs() <= cfg.tol_cb * 1.15
                    })
                })
        }
        "TRP_IND_N15HSQC" => clu
            .trp_indole
            .is_some_and(|(ne1, he1, _)| (p.h - he1).abs() <= tol_h && (p.n - ne1).abs() <= tol_n),
        "HIS_LR_N15HMQC" => {
            clu.ch_slots
                .values()
                .any(|s| (p.h - s.h_ppm).abs() <= tol_h)
                && (clu.nd1_ppm.is_some_and(|(n, _)| (p.n - n).abs() <= tol_n)
                    || clu.ne2_ppm.is_some_and(|(n, _)| (p.n - n).abs() <= tol_n))
        }
        _ => false,
    }
}

fn put_ring_atom_if_missing(
    merged: &mut Reference,
    k_cur: usize,
    res_name: &str,
    atom: &str,
    val: f64,
) {
    let key = (k_cur, atom.to_string());
    if !merged.shifts.contains_key(&key) {
        merged.shifts.insert(key.clone(), val);
        let code = nmr_core::stereospec::default_atom_ambiguity_code(res_name, atom);
        merged.ambiguity_codes.entry(key).or_insert(code);
    }
}

fn insert_corroborated_missing_ring_atoms(
    merged: &mut Reference,
    k_cur: usize,
    clu: &AromaticRingCluster,
) {
    let res_name = merged.sequence.get(k_cur).cloned().unwrap_or_default();
    let has_any = |r: &Reference, atoms: &[&str]| -> bool {
        atoms
            .iter()
            .any(|a| r.shifts.contains_key(&(k_cur, (*a).to_string())))
    };

    for (&c_atom, slot) in &clu.ch_slots {
        match (res_name.as_str(), c_atom) {
            ("PHE" | "TYR", "CD1") => {
                if !has_any(merged, &["CD1", "CD2"]) {
                    put_ring_atom_if_missing(merged, k_cur, &res_name, "CD1", slot.c_ppm);
                    put_ring_atom_if_missing(merged, k_cur, &res_name, "CD2", slot.c_ppm);
                }
                if !has_any(merged, &["HD1", "HD2"]) {
                    put_ring_atom_if_missing(merged, k_cur, &res_name, "HD1", slot.h_ppm);
                    put_ring_atom_if_missing(merged, k_cur, &res_name, "HD2", slot.h_ppm);
                }
            }
            ("PHE" | "TYR", "CE1") => {
                if !has_any(merged, &["CE1", "CE2"]) {
                    put_ring_atom_if_missing(merged, k_cur, &res_name, "CE1", slot.c_ppm);
                    put_ring_atom_if_missing(merged, k_cur, &res_name, "CE2", slot.c_ppm);
                }
                if !has_any(merged, &["HE1", "HE2"]) {
                    put_ring_atom_if_missing(merged, k_cur, &res_name, "HE1", slot.h_ppm);
                    put_ring_atom_if_missing(merged, k_cur, &res_name, "HE2", slot.h_ppm);
                }
            }
            ("PHE", "CZ") => {
                put_ring_atom_if_missing(merged, k_cur, &res_name, "CZ", slot.c_ppm);
                put_ring_atom_if_missing(merged, k_cur, &res_name, "HZ", slot.h_ppm);
            }
            ("TRP", ca) => {
                let pa = match ca {
                    "CD1" => "HD1",
                    "CE3" => "HE3",
                    "CZ3" => "HZ3",
                    "CH2" => "HH2",
                    "CZ2" => "HZ2",
                    _ => continue,
                };
                put_ring_atom_if_missing(merged, k_cur, &res_name, ca, slot.c_ppm);
                put_ring_atom_if_missing(merged, k_cur, &res_name, pa, slot.h_ppm);
            }
            ("HIS", ca) => {
                let pa = if ca == "CD2" { "HD2" } else { "HE1" };
                put_ring_atom_if_missing(merged, k_cur, &res_name, ca, slot.c_ppm);
                put_ring_atom_if_missing(merged, k_cur, &res_name, pa, slot.h_ppm);
            }
            _ => {}
        }
    }
    if res_name == "TRP" {
        if let Some((ne1, he1, _)) = clu.trp_indole {
            put_ring_atom_if_missing(merged, k_cur, &res_name, "NE1", ne1);
            put_ring_atom_if_missing(merged, k_cur, &res_name, "HE1", he1);
        }
    }
    if res_name == "HIS" {
        if let Some((nd1, _)) = clu.nd1_ppm {
            put_ring_atom_if_missing(merged, k_cur, &res_name, "ND1", nd1);
        }
        if let Some((ne2, _)) = clu.ne2_ppm {
            put_ring_atom_if_missing(merged, k_cur, &res_name, "NE2", ne2);
        }
    }
    if !merged.shifts.contains_key(&(k_cur, "CB".to_string())) {
        if let Some(cb_y) = clu.cb_hd.or(clu.cb_he) {
            put_ring_atom_if_missing(merged, k_cur, &res_name, "CB", cb_y);
        }
    }
}

#[allow(dead_code)]
fn merge_gui_aromatic_preserving_user_rings(
    working_ref: &Reference,
    prior_user_ref: Option<&Reference>,
    prior_amb: &BTreeMap<(usize, String), u8>,
    pred_conf: Option<&BTreeMap<usize, f64>>,
    arom_peaks: &[Peak],
    sc_priors: &SidechainPriors,
) -> Reference {
    merge_gui_aromatic_with_options(
        working_ref,
        prior_user_ref,
        prior_amb,
        pred_conf,
        arom_peaks,
        sc_priors,
        false,
    )
}

fn merge_gui_aromatic_with_options(
    working_ref: &Reference,
    prior_user_ref: Option<&Reference>,
    prior_amb: &BTreeMap<(usize, String), u8>,
    pred_conf: Option<&BTreeMap<usize, f64>>,
    arom_peaks: &[Peak],
    sc_priors: &SidechainPriors,
    overwrite_curated_rings: bool,
) -> Reference {
    merge_gui_aromatic_with_ph(
        working_ref,
        prior_user_ref,
        prior_amb,
        pred_conf,
        arom_peaks,
        sc_priors,
        overwrite_curated_rings,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn merge_gui_aromatic_with_ph(
    working_ref: &Reference,
    prior_user_ref: Option<&Reference>,
    prior_amb: &BTreeMap<(usize, String), u8>,
    pred_conf: Option<&BTreeMap<usize, f64>>,
    arom_peaks: &[Peak],
    sc_priors: &SidechainPriors,
    overwrite_curated_rings: bool,
    sample_ph: Option<f64>,
) -> Reference {
    const RING_ATOMS: &[&str] = &[
        "CG", "CD1", "CD2", "HD1", "HD2", "CE1", "CE2", "HE1", "HE2", "CZ", "HZ", "CE3", "HE3",
        "CZ2", "HZ2", "CZ3", "HZ3", "CH2", "HH2", "ND1", "NE1", "NE2",
    ];
    let is_arom_ring_key = |r: &Reference, k: usize, atom: &str| -> bool {
        matches!(
            r.sequence.get(k).map(String::as_str),
            Some("PHE" | "TYR" | "TRP" | "HIS")
        ) && RING_ATOMS.contains(&atom)
    };

    // Assemble full reference retaining all user-curated ring shifts & ambiguity codes
    let mut curated_ref = working_ref.clone();
    if let Some(user_ref) = prior_user_ref {
        for (&(k, ref atom), &v) in &user_ref.shifts {
            if is_arom_ring_key(user_ref, k, atom) {
                curated_ref.shifts.insert((k, atom.clone()), v);
            }
        }
        for (&(k, ref atom), &code) in &user_ref.ambiguity_codes {
            if is_arom_ring_key(user_ref, k, atom) {
                curated_ref.ambiguity_codes.insert((k, atom.clone()), code);
            }
        }
    }
    for (&(k, ref atom), &code) in prior_amb {
        if is_arom_ring_key(&curated_ref, k, atom) {
            curated_ref.ambiguity_codes.insert((k, atom.clone()), code);
        }
    }

    let cfg = AromaticSolverConfig {
        sample_ph: sample_ph.filter(|v| v.is_finite() && (0.0..=14.0).contains(v)),
        ..AromaticSolverConfig::default()
    };

    if overwrite_curated_rings {
        let mut backbone_for_arom = curated_ref.clone();
        backbone_for_arom
            .shifts
            .retain(|&(k, ref atom), _| !is_arom_ring_key(&curated_ref, k, atom));
        backbone_for_arom
            .ambiguity_codes
            .retain(|&(k, ref atom), _| !is_arom_ring_key(&curated_ref, k, atom));

        let arom_res =
            assign_aromatic_rings(&backbone_for_arom, pred_conf, arom_peaks, sc_priors, &cfg);
        let assigned_ring_residues: BTreeSet<usize> = arom_res
            .rings
            .iter()
            .filter_map(|r| r.cluster_idx.map(|_| r.residue_idx))
            .collect();
        let mut base_for_overwrite = curated_ref.clone();
        base_for_overwrite.shifts.retain(|&(k, ref atom), _| {
            !(assigned_ring_residues.contains(&k) && is_arom_ring_key(&curated_ref, k, atom))
        });
        base_for_overwrite
            .ambiguity_codes
            .retain(|&(k, ref atom), _| {
                !(assigned_ring_residues.contains(&k) && is_arom_ring_key(&curated_ref, k, atom))
            });
        let mut merged = merge_aromatic_assignments_preserving_existing(
            &base_for_overwrite,
            &arom_res.reference,
            &arom_res.atom_confidence,
        );
        for (k, atom) in arom_res.atom_confidence.keys() {
            let res_name = merged.sequence.get(*k).map(String::as_str).unwrap_or("");
            let code = nmr_core::stereospec::default_atom_ambiguity_code(res_name, atom);
            merged.ambiguity_codes.insert((*k, atom.clone()), code);
        }
        return merged;
    }

    // Default (`overwrite_curated_rings == false`):
    // 1. Do NOT strip user-curated ring shifts before conflict/corroboration checking.
    // 2. Identify residues with >= 1 user-curated aromatic ring shift.
    let mut curated_residues: BTreeSet<usize> = BTreeSet::new();
    for &(k, ref atom) in curated_ref.shifts.keys() {
        if is_arom_ring_key(&curated_ref, k, atom) {
            curated_residues.insert(k);
        }
    }

    let probe = assign_aromatic_rings(&curated_ref, pred_conf, arom_peaks, sc_priors, &cfg);
    if curated_residues.is_empty() {
        let mut merged = merge_aromatic_assignments_preserving_existing(
            &curated_ref,
            &probe.reference,
            &probe.atom_confidence,
        );
        for (k, atom) in probe.atom_confidence.keys() {
            let res_name = merged.sequence.get(*k).map(String::as_str).unwrap_or("");
            let code = nmr_core::stereospec::default_atom_ambiguity_code(res_name, atom);
            merged
                .ambiguity_codes
                .entry((*k, atom.clone()))
                .or_insert(code);
        }
        return merged;
    }

    // 3. Match each curated residue against compatible experimental clusters so:
    //    - partial disjoint clusters (e.g. CD-only vs curated CE/CZ) are never mixed into curated rings,
    //    - clusters whose shifts belong to a curated residue (including superset clusters and 1-slot
    //      user-tweaked curated rings) are claimed and cannot jump onto another residue.
    let (cb_offset_hd, cb_offset_he) =
        compute_yamazaki_cb_offsets(&curated_ref, pred_conf, arom_peaks, &cfg);
    let mut claimed_clusters: BTreeMap<usize, (usize, ClusterCuratedEval)> = BTreeMap::new();
    for &k_cur in &curated_residues {
        let mut best: Option<(usize, ClusterCuratedEval)> = None;
        for (ci, clu) in probe.clusters.iter().enumerate() {
            if claimed_clusters.contains_key(&ci) {
                continue;
            }
            let ev = eval_cluster_vs_curated_residue(clu, k_cur, &curated_ref, &cfg);
            if !cluster_belongs_to_curated(&ev) {
                continue;
            }
            let better = match best {
                None => true,
                Some((_, b_ev)) => {
                    ev.conflict_obs < b_ev.conflict_obs
                        || (ev.conflict_obs == b_ev.conflict_obs
                            && (ev.shared_obs > b_ev.shared_obs
                                || (ev.shared_obs == b_ev.shared_obs
                                    && ev.norm_dist_sq < b_ev.norm_dist_sq)))
                }
            };
            if better {
                best = Some((ci, ev));
            }
        }
        if let Some((ci, ev)) = best {
            claimed_clusters.insert(ci, (k_cur, ev));
        }
    }

    // 4. Solve uncurated residues on peaks excluding peaks owned solely by claimed curated clusters,
    //    pre-applying `(cb_offset_hd, cb_offset_he)` so masking curated residues to `ALA` never loses
    //    Yamazaki CB offset calibration on remaining uncurated residues.
    let mut sub_ref = curated_ref.clone();
    for &k_cur in &curated_residues {
        if let Some(slot) = sub_ref.sequence.get_mut(k_cur) {
            *slot = "ALA".to_string();
        }
    }
    let sub_peaks: Vec<Peak> = arom_peaks
        .iter()
        .filter(|p| {
            let matches_claimed = claimed_clusters.keys().any(|&ci| {
                peak_matches_aromatic_cluster(
                    p,
                    &probe.clusters[ci],
                    cb_offset_hd,
                    cb_offset_he,
                    &cfg,
                )
            });
            if !matches_claimed {
                return true;
            }
            probe.clusters.iter().enumerate().any(|(cj, clu)| {
                !claimed_clusters.contains_key(&cj)
                    && peak_matches_aromatic_cluster(p, clu, cb_offset_hd, cb_offset_he, &cfg)
            })
        })
        .map(|p| {
            let mut q = p.clone();
            if q.exp == "HBCBCGCDHD" && cb_offset_hd != 0.0 {
                q.c = q.c.map(|c| c - cb_offset_hd);
            } else if q.exp == "HBCBCGCDCEHE" && cb_offset_he != 0.0 {
                q.c = q.c.map(|c| c - cb_offset_he);
            }
            q
        })
        .collect();

    let sub_res = assign_aromatic_rings(&sub_ref, pred_conf, &sub_peaks, sc_priors, &cfg);

    let mut merged = curated_ref.clone();

    // Corroborated missing-atom completion on curated residues (strictly `conflict_obs == 0` so
    // conflicting or disjoint slots NEVER overwrite or mix into curated rings!)
    for (&ci, &(k_cur, ev)) in &claimed_clusters {
        let min_corroborate = ev.total_cur_obs.clamp(1, 2);
        if ev.conflict_obs == 0 && !ev.cb_contradicts && ev.shared_obs >= min_corroborate {
            insert_corroborated_missing_ring_atoms(&mut merged, k_cur, &probe.clusters[ci]);
        }
    }

    // Merge assignments onto uncurated residues while blocking duplicate curated ring clusters
    for r in &sub_res.rings {
        let k_other = r.residue_idx;
        if curated_residues.contains(&k_other) {
            continue;
        }
        let Some(sub_ci) = r.cluster_idx else {
            continue;
        };
        let sub_clu = &sub_res.clusters[sub_ci];

        let is_duplicate_of_curated = curated_residues.iter().any(|&k_cur| {
            if curated_ref.sequence.get(k_cur) != curated_ref.sequence.get(k_other) {
                return false;
            }
            let dup_ev = eval_cluster_vs_curated_residue(sub_clu, k_cur, &curated_ref, &cfg);
            if !cluster_belongs_to_curated(&dup_ev) {
                return false;
            }
            // Allow only if >= 2 distinct clusters in probe genuinely shared these degenerate shifts,
            // or if sub_clu carries a Yamazaki CB matching k_other's CB while contradicting k_cur's CB.
            let n_probe_matches = probe
                .clusters
                .iter()
                .filter(|c| {
                    let e = eval_cluster_vs_curated_residue(c, k_cur, &curated_ref, &cfg);
                    cluster_belongs_to_curated(&e)
                })
                .count();
            if n_probe_matches >= 2 {
                return false;
            }
            if let (Some(&cb_other), Some(&cb_cur)) = (
                curated_ref.shifts.get(&(k_other, "CB".to_string())),
                curated_ref.shifts.get(&(k_cur, "CB".to_string())),
            ) {
                let cbs: Vec<f64> = [sub_clu.cb_hd, sub_clu.cb_he]
                    .into_iter()
                    .flatten()
                    .collect();
                if !cbs.is_empty()
                    && cbs.iter().any(|&cb| (cb - cb_other).abs() <= cfg.tol_cb)
                    && cbs.iter().all(|&cb| (cb - cb_cur).abs() > cfg.tol_cb)
                {
                    return false;
                }
            }
            true
        });

        if is_duplicate_of_curated {
            continue;
        }

        for key in sub_res.atom_confidence.keys() {
            if key.0 != k_other {
                continue;
            }
            if let Some(&val) = sub_res.reference.shifts.get(key) {
                merged.shifts.entry(key.clone()).or_insert(val);
            }
            let res_name = merged
                .sequence
                .get(k_other)
                .map(String::as_str)
                .unwrap_or("");
            let default_code = nmr_core::stereospec::default_atom_ambiguity_code(res_name, &key.1);
            let code = sub_res
                .reference
                .ambiguity_codes
                .get(key)
                .copied()
                .unwrap_or(default_code);
            merged.ambiguity_codes.entry(key.clone()).or_insert(code);
        }
    }

    merged
}

struct CoreOctetBackendAdapter;

impl CoreOctetBackendAdapter {
    fn encode_hex_pretty(bytes: &[u8]) -> String {
        let mut out = String::with_capacity(bytes.len() * 3 + 64);
        out.push_str("# Binary ChemDraw CDX (VjCD0100) hex stream\n");
        for (i, b) in bytes.iter().enumerate() {
            if i > 0 {
                if i % 24 == 0 {
                    out.push('\n');
                } else {
                    out.push(' ');
                }
            }
            out.push_str(&format!("{b:02X}"));
        }
        out
    }

    fn decode_hex_bytes(input: &str) -> Result<Vec<u8>, String> {
        let mut hex_chars = Vec::new();
        for line in input.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with('#') || trimmed.starts_with("//") {
                continue;
            }
            for ch in trimmed.chars() {
                if ch.is_ascii_hexdigit() {
                    hex_chars.push(ch);
                } else if !ch.is_whitespace() && ch != ',' && ch != ':' && ch != '-' {
                    return Err(format!(
                        "Unexpected non-hex character '{ch}' in CDX hex buffer"
                    ));
                }
            }
        }
        if hex_chars.len() % 2 != 0 {
            return Err("CDX hex stream must have an even number of hex digits".to_string());
        }
        let mut bytes = Vec::with_capacity(hex_chars.len() / 2);
        for pair in hex_chars.chunks_exact(2) {
            let s: String = pair.iter().collect();
            let v =
                u8::from_str_radix(&s, 16).map_err(|e| format!("Invalid hex byte '{s}': {e}"))?;
            bytes.push(v);
        }
        Ok(bytes)
    }

    fn mol_with_cip_to_scene(mut mol: Molecule) -> RenderScene {
        let _ = kekulize(&mut mol);
        let mut scene = RenderScene::from(&mol);
        let [cx, cy] = scene.atom_centroid();
        scene.translate(-cx, -cy);
        scene
    }
}

impl OctetBackendAdapter for CoreOctetBackendAdapter {
    fn clean_2d(&self, scene: &RenderScene) -> Option<RenderScene> {
        if scene.atoms.is_empty() {
            return Some(scene.clone());
        }
        let orig_centroid = scene.atom_centroid();
        let mut mol = Molecule::from(scene);
        let _ = expand_abbreviations(&mut mol);
        clean_2d(&mut mol);
        let mut cleaned = Self::mol_with_cip_to_scene(mol);
        cleaned.translate(orig_centroid[0], orig_centroid[1]);
        // Preserve non-molecular canvas annotations (arrows, brackets, orbitals, TLC, captions, embedded spectra)
        cleaned.reaction_arrows = scene.reaction_arrows.clone();
        cleaned.curved_arrows = scene.curved_arrows.clone();
        cleaned.brackets = scene.brackets.clone();
        cleaned.orbitals = scene.orbitals.clone();
        cleaned.tlc_plates = scene.tlc_plates.clone();
        cleaned.spectra_1d = scene.spectra_1d.clone();
        cleaned.texts = scene.texts.clone();
        Some(cleaned)
    }

    fn name_or_smiles_to_scene(&self, query: &str) -> Option<Result<RenderScene, String>> {
        let q = query.trim();
        if q.is_empty() {
            return Some(Err(
                "Enter a chemical name, SMILES, InChI, or peptide sequence".to_string(),
            ));
        }

        // 1. Reaction SMILES (`A.B>reag>C`)
        if q.contains(">>") || (q.matches('>').count() == 2 && !q.contains('<')) {
            if let Ok(mut doc) = parse_reaction_smiles(q) {
                let mut x_cursor = 0.0_f64;
                for m in &mut doc.molecules {
                    clean_2d(m);
                    for a in &mut m.atoms {
                        a.x += x_cursor * 1.54;
                    }
                    x_cursor += 5.5;
                }
                let mut scene = RenderScene::from(&doc);
                let [cx, cy] = scene.atom_centroid();
                scene.translate(-cx, -cy);
                return Some(Ok(scene));
            }
        }

        // 2. Standard InChI (`InChI=1S/...`)
        if q.starts_with("InChI=") {
            return Some(
                parse_inchi(q)
                    .map(|mut m| {
                        clean_2d(&mut m);
                        Self::mol_with_cip_to_scene(m)
                    })
                    .map_err(|e| format!("InChI parse error: {e}")),
            );
        }

        // 3. Biopolymer HELM / peptide 3-letter notation (`PEPTIDE1{...}` or `H-Ala-Pro-OH`)
        if q.starts_with("PEPTIDE") || q.starts_with("RNA") || q.starts_with("DNA") {
            if let Ok(mut m) = parse_biopolymer_sequence(q) {
                clean_2d(&mut m);
                return Some(Ok(Self::mol_with_cip_to_scene(m)));
            }
        }

        // 4. Offline IUPAC / common chemical name (`aspirin`, `ibuprofen`, `bicyclo[2.2.1]hept-2-ene`, `(2S)-2-aminopropanoic acid`, ...)
        if let Ok(amol) = iupac_name_to_mol(q) {
            let smi = amol.to_smiles();
            if let Ok(mut m) = parse_smiles(&smi) {
                clean_2d(&mut m);
                return Some(Ok(Self::mol_with_cip_to_scene(m)));
            }
        }

        // 5. SMILES (`CC(=O)Oc1ccccc1C(=O)O`)
        if let Ok(mut m) = parse_smiles(q) {
            clean_2d(&mut m);
            return Some(Ok(Self::mol_with_cip_to_scene(m)));
        }

        // 6. Condensed structural formula (`CH3CH2OH`, `(CH3)3COH`, `PhCOOH`) or peptide (`Ala-Pro-Tyr`)
        if let Ok(mut m) = parse_condensed_formula(q) {
            clean_2d(&mut m);
            return Some(Ok(Self::mol_with_cip_to_scene(m)));
        }
        if q.contains('-') {
            if let Ok(m) = build_peptide_2d(q) {
                return Some(Ok(Self::mol_with_cip_to_scene(m)));
            }
        }

        None
    }

    fn scene_to_name_and_smiles(&self, scene: &RenderScene) -> Option<(String, String)> {
        if scene.atoms.is_empty() {
            return Some((String::new(), String::new()));
        }
        let mut mol = Molecule::from(scene);
        let _ = expand_abbreviations(&mut mol);
        let smiles = if !scene.reaction_arrows.is_empty() {
            let doc = ChemDocument::from(scene);
            let rxn = write_reaction_smiles(&doc);
            if rxn.is_empty() {
                write_canonical_smiles(&mol)
            } else {
                rxn
            }
        } else {
            write_canonical_smiles(&mol)
        };
        let amol = AnalytMol::from(&mol);
        let name = mol_to_iupac_name(&amol).unwrap_or_else(|_| {
            compute_phys_chem_props(&amol)
                .map(|p| p.formula)
                .unwrap_or_else(|_| smiles.clone())
        });
        Some((name, smiles))
    }

    fn compute_properties(&self, scene: &RenderScene) -> Option<MolecularPropertiesSummary> {
        if scene.atoms.is_empty() {
            return Some(MolecularPropertiesSummary::default());
        }
        let mut mol = Molecule::from(scene);
        let _ = expand_abbreviations(&mut mol);
        let rings = perceive_sssr(&mol).len();
        let amol = AnalytMol::from(&mol);
        let props = compute_phys_chem_props(&amol).ok()?;
        let heavy_atoms = amol.atoms.iter().filter(|a| a.element != "H").count();
        Some(MolecularPropertiesSummary {
            formula_hill: props.formula,
            molecular_weight: props.molecular_weight,
            exact_mass: props.monoisotopic_mass,
            elemental_analysis: props.elemental_analysis_text,
            heavy_atoms,
            rings,
            h_bond_donors: props.h_bond_donors,
            h_bond_acceptors: props.h_bond_acceptors,
        })
    }

    fn predict_nmr_1h_13c(
        &self,
        scene: &RenderScene,
        _freq_mhz: f64,
    ) -> Option<(Vec<SpectrumSignal1D>, Vec<SpectrumSignal1D>)> {
        if scene.atoms.is_empty() {
            return Some((Vec::new(), Vec::new()));
        }
        let raw_mol = Molecule::from(scene);
        let has_abbrevs = raw_mol.atoms.iter().any(|a| a.superatom_label.is_some());
        let mol = if has_abbrevs {
            let mut m = raw_mol;
            let _ = expand_abbreviations(&mut m);
            m
        } else {
            raw_mol
        };
        let amol = AnalytMol::from(&mol);

        let pred_h = predict_nmr(&amol, NmrNucleus::H1, Some("CDCl3"), None);
        let pred_c = predict_nmr(&amol, NmrNucleus::C13, Some("CDCl3"), None);

        let n_scene_atoms = scene.atoms.len();
        let sigs_1h = pred_h
            .signals
            .into_iter()
            .map(|s| {
                let mult = s.multiplicity();
                let mut couplings_hz = Vec::new();
                for c in &s.couplings {
                    for _ in 0..c.count.min(12) {
                        couplings_hz.push(c.j_hz);
                    }
                }
                let atom_indices: Vec<usize> = if has_abbrevs {
                    Vec::new()
                } else {
                    s.atoms.into_iter().filter(|&i| i < n_scene_atoms).collect()
                };
                SpectrumSignal1D {
                    ppm: s.shift_ppm,
                    multiplicity: mult,
                    couplings_hz,
                    integral: s.nuclei as f64,
                    atom_indices,
                    assignment_label: Some(format!("{}H", s.nuclei)),
                }
            })
            .collect();

        let sigs_13c = pred_c
            .signals
            .into_iter()
            .map(|s| {
                let atom_indices: Vec<usize> = if has_abbrevs {
                    Vec::new()
                } else {
                    s.atoms.into_iter().filter(|&i| i < n_scene_atoms).collect()
                };
                SpectrumSignal1D {
                    ppm: s.shift_ppm,
                    multiplicity: "s".to_string(),
                    couplings_hz: Vec::new(),
                    integral: s.nuclei as f64,
                    atom_indices,
                    assignment_label: Some(format!("{}C", s.nuclei)),
                }
            })
            .collect();

        Some((sigs_1h, sigs_13c))
    }

    fn export_custom(
        &self,
        scene: &RenderScene,
        style: &DocumentStyle,
        format: IoFormat,
    ) -> Option<Result<String, String>> {
        match format {
            IoFormat::Cdxml => {
                let doc = ChemDocument::from(scene);
                Some(Ok(write_cdxml(&doc)))
            }
            IoFormat::CdxHex => {
                let doc = ChemDocument::from(scene);
                let bytes = write_cdx(&doc);
                Some(Ok(Self::encode_hex_pretty(&bytes)))
            }
            IoFormat::OctJson => {
                let doc = ChemDocument::from(scene);
                Some(write_oct(&doc).map_err(|e| format!("OCT JSON export error: {e}")))
            }
            IoFormat::MolV2000 => {
                let mut mol = Molecule::from(scene);
                let _ = expand_abbreviations(&mut mol);
                Some(write_mol_v2000(&mol).map_err(|e| format!("MOL V2000 export error: {e}")))
            }
            IoFormat::Sdf => {
                let mut mol = Molecule::from(scene);
                let _ = expand_abbreviations(&mut mol);
                let amol = AnalytMol::from(&mol);
                let mut props = BTreeMap::new();
                if let Ok(pin) = mol_to_iupac_name(&amol) {
                    props.insert("IUPAC_NAME".to_string(), pin);
                }
                props.insert("SMILES".to_string(), write_canonical_smiles(&mol));
                props.insert("INCHIKEY".to_string(), write_inchikey(&mol));
                if let Ok(pc) = compute_phys_chem_props(&amol) {
                    props.insert("FORMULA".to_string(), pc.formula);
                    props.insert(
                        "EXACT_MASS".to_string(),
                        format!("{:.5}", pc.monoisotopic_mass),
                    );
                }
                Some(Ok(write_sdf(&[SdfRecord {
                    molecule: mol,
                    properties: props,
                }])))
            }
            IoFormat::Smiles => {
                let mut mol = Molecule::from(scene);
                let _ = expand_abbreviations(&mut mol);
                if !scene.reaction_arrows.is_empty() {
                    let doc = ChemDocument::from(scene);
                    let rxn = write_reaction_smiles(&doc);
                    if !rxn.is_empty() {
                        return Some(Ok(rxn));
                    }
                }
                Some(Ok(write_canonical_smiles(&mol)))
            }
            IoFormat::InChI => {
                let mut mol = Molecule::from(scene);
                let _ = expand_abbreviations(&mut mol);
                let inchi = write_inchi(&mol);
                let key = write_inchikey(&mol);
                Some(Ok(format!("{inchi}\nInChIKey={key}")))
            }
            IoFormat::JcampDx => {
                let mut mol = Molecule::from(scene);
                let _ = expand_abbreviations(&mut mol);
                let amol = AnalytMol::from(&mol);
                let pred = predict_nmr(&amol, NmrNucleus::H1, Some("CDCl3"), None);
                let spec = synthesize_1d_nmr(&pred, &Synthesize1dOptions::default());
                Some(Ok(write_jcamp_nmr(&spec, JcampCompression::Affn)))
            }
            IoFormat::Svg => Some(Ok(render_scene_svg(scene, style))),
            IoFormat::Pdf | IoFormat::Png => None,
        }
    }

    fn import_custom(&self, format: IoFormat, input: &str) -> Option<Result<RenderScene, String>> {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return Some(Err("Input buffer is empty".to_string()));
        }
        match format {
            IoFormat::Cdxml => Some(
                read_cdxml(trimmed)
                    .map(|doc| {
                        let mut sc = RenderScene::from(&doc);
                        let [cx, cy] = sc.atom_centroid();
                        sc.translate(-cx, -cy);
                        sc
                    })
                    .map_err(|e| format!("CDXML import error: {e}")),
            ),
            IoFormat::CdxHex => Some(
                Self::decode_hex_bytes(trimmed)
                    .and_then(|bytes| {
                        read_cdx(&bytes).map_err(|e| format!("CDX binary error: {e}"))
                    })
                    .map(|doc| {
                        let mut sc = RenderScene::from(&doc);
                        let [cx, cy] = sc.atom_centroid();
                        sc.translate(-cx, -cy);
                        sc
                    }),
            ),
            IoFormat::OctJson => Some(
                read_oct(trimmed)
                    .map(|doc| {
                        let mut sc = RenderScene::from(&doc);
                        let [cx, cy] = sc.atom_centroid();
                        sc.translate(-cx, -cy);
                        sc
                    })
                    .map_err(|e| format!("OCT JSON import error: {e}")),
            ),
            IoFormat::MolV2000 => Some(
                read_mol(trimmed)
                    .map(Self::mol_with_cip_to_scene)
                    .map_err(|e| format!("MOL import error: {e}")),
            ),
            IoFormat::Sdf => Some(
                read_sdf(trimmed)
                    .map_err(|e| format!("SDF import error: {e}"))
                    .and_then(|recs| {
                        recs.into_iter()
                            .next()
                            .map(|r| Self::mol_with_cip_to_scene(r.molecule))
                            .ok_or_else(|| "SDF contains no molecule records".to_string())
                    }),
            ),
            IoFormat::Smiles => self.name_or_smiles_to_scene(
                trimmed
                    .lines()
                    .next()
                    .unwrap_or(trimmed)
                    .split_whitespace()
                    .next()
                    .unwrap_or(trimmed),
            ),
            IoFormat::InChI => {
                let inchi_line = trimmed
                    .lines()
                    .map(str::trim)
                    .find(|l| l.starts_with("InChI="))
                    .unwrap_or(trimmed);
                Some(
                    parse_inchi(inchi_line)
                        .map(|mut m| {
                            clean_2d(&mut m);
                            Self::mol_with_cip_to_scene(m)
                        })
                        .map_err(|e| format!("InChI import error: {e}")),
                )
            }
            IoFormat::JcampDx => {
                let blocks = match read_jcamp(trimmed) {
                    Ok(b) if !b.is_empty() => b,
                    Ok(_) => return Some(Err("No spectrum blocks found in JCAMP-DX".to_string())),
                    Err(e) => return Some(Err(format!("JCAMP-DX import error: {e}"))),
                };
                let sp = &blocks[0];
                let picked = pick_1d_multiplets(sp, 0.04, 22.0, None);
                let signals: Vec<SpectrumSignal1D> = picked
                    .into_iter()
                    .map(|m| SpectrumSignal1D {
                        ppm: m.center_ppm,
                        multiplicity: m.estimated_multiplicity,
                        couplings_hz: m.estimated_j_hz.into_iter().collect(),
                        integral: m.normalized_integral.max(0.5),
                        atom_indices: Vec::new(),
                        assignment_label: None,
                    })
                    .collect();
                let n_pts = sp.y.len();
                let mut custom_trace = Vec::with_capacity(n_pts.min(2048));
                let max_y =
                    sp.y.iter()
                        .copied()
                        .fold(0.0_f64, |acc, v| acc.max(v.abs()))
                        .max(1e-9);
                let step = (n_pts / 2048).max(1);
                for idx in (0..n_pts).step_by(step) {
                    custom_trace.push([sp.x_at(idx), sp.y[idx] / max_y]);
                }
                let ppm_min = sp.first_x.min(sp.last_x);
                let ppm_max = sp.first_x.max(sp.last_x);
                let mut scene = RenderScene::new();
                scene.spectra_1d.push(SceneSpectrum1D {
                    origin: [-5.5, -2.0],
                    width_mol: 11.0,
                    height_mol: 4.0,
                    nucleus: sp.nucleus.clone().unwrap_or_else(|| "1H".to_string()),
                    spectrometer_mhz: sp.mhz.unwrap_or(400.0).max(60.0),
                    ppm_min: if ppm_min.is_finite() { ppm_min } else { -0.5 },
                    ppm_max: if ppm_max.is_finite() { ppm_max } else { 11.0 },
                    linewidth_hz: 1.4,
                    show_integrals: !signals.is_empty(),
                    show_peak_labels: true,
                    title: Some(if sp.title.is_empty() {
                        "Imported JCAMP-DX 1D NMR".to_string()
                    } else {
                        sp.title.clone()
                    }),
                    signals,
                    custom_trace,
                });
                Some(Ok(scene))
            }
            _ => None,
        }
    }
}

struct NmrWorkbenchApp {
    workspace_mode: WorkspaceMode,
    octet_app: OctetDrawPanel,
    dosy_panel: DosyPanelState,
    metabolomics_panel: MetabolomicsStudioPanel,
    #[cfg(not(target_arch = "wasm32"))]
    gpu_engine: Option<GpuEngine>,
    #[cfg(not(target_arch = "wasm32"))]
    bayes_gpu_solver: Option<BayesSaGpuSolver>,
    adapter_label: Option<String>,
    solver_kind: GuiSolverKind,
    regime_kind: GuiRegimeKind,
    metal_ligands_input: String,
    idp_temp_k: f64,
    idp_ph_input: String,
    c13_picked_dataset: Option<C13PickedDataset>,
    propensity_report: Option<PropensityReport>,
    show_propensity_panel: bool,
    #[cfg(not(target_arch = "wasm32"))]
    use_gpu: bool,
    use_neural_picker: bool,
    enable_axis_rescue: bool,
    axis_rescue_notices: Vec<String>,
    axis_rescue_audits: Vec<AxisRescueAuditRecord>,
    status_msg: String,
    last_solve_ms: Option<f64>,
    last_bayes_sa_telemetry: Option<(f64, f64, f64)>,

    // Active dataset
    priors: Priors,
    sidechain_priors: SidechainPriors,
    reference: Option<Reference>,
    sequence: Vec<String>,
    seq_ids: Vec<i32>,
    peaks: Vec<Peak>,
    truth: Option<Vec<i32>>,
    systems: Vec<SpinSystem>,
    u_mat: Option<Matrix2D>,
    p_mat: Option<Matrix2D>,
    assign_res: Option<AssignResult>,
    pred_ref: Option<Reference>,
    pred_conf: BTreeMap<usize, f64>,
    pred_ambiguity_codes: BTreeMap<(usize, String), u8>,
    metrics: Option<Metrics>,
    sidechain_ribbon_state: SequenceRibbonState,
    show_sidechain_inspector: bool,
    overwrite_curated_rings: bool,
    mol3d_panel: Mol3dViewerPanel,
    show_mol3d_viewport: bool,
    loaded_pdb_text: Option<String>,
    relax_panel: RelaxPanelState,
    show_relax_panel: bool,
    dynamics_panel: DynamicsStudioPanel,
    show_dynamics_panel: bool,
    stack1d_panel: Stack1dState,
    show_stack1d_panel: bool,
    show_strips_dock: bool,
    show_workflow_guide: bool,
    show_advanced_protein_bar: bool,
    show_sequence_ribbon: bool,
    paper_white_canvas: bool,
    workbook: WorkbookState,
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    show_native_global_open_bar: bool,

    // Spectra for GPU contour & 3D strip visualization
    hsqc_spec: Option<Spectrum2D>,
    hsqc_arc: Arc<Vec<f32>>,
    hsqc_gen: u64,
    hncacb_3d: Option<Spectrum3D>,
    hnco_3d: Option<Spectrum3D>,
    extra_3d_count: usize,
    overlay_layers: Vec<(String, OverlayLayerSpec)>,
    primary_2d_label: String,
    primary_2d_visible: bool,
    primary_ppm_nudge: [f32; 2],
    next_layer_id: u64,
    active_overlay_idx: usize,

    // Phase 9 UX, Session, Overlays & Match-Finder State
    ingest_panel: IngestPanelState,
    command_palette: CommandPalette,
    chord_dispatcher: ChordDispatcher,
    session: WorkbenchSessionController,
    crosshair: SyncCrosshair,
    strip_tiler: StripTilerPanel,
    use_match_finder_strips: bool,
    qc_report: Option<DatasetQcReport>,
    raw_fids_total: usize,
    unprocessed_fids_remaining: usize,
    has_structure_or_dynamics: bool,
    exported_in_session: bool,
    show_contour_modal: bool,
    show_overlay_manager: bool,
    show_assign_popover: bool,
    show_qc_modal: bool,
    show_about_modal: bool,
    mcp_bridge: McpBridgeState,
    show_mcp_bridge_window: bool,
    mcp_last_viewport_key: (u64, i32, i32, i32),
    assign_target_res: usize,
    assign_lock_check: bool,

    // 2D viewport & contour settings
    view_h_left: f32,
    view_h_right: f32,
    view_n_top: f32,
    view_n_bottom: f32,
    contour_base: f32,
    contour_noise_ref: f32,
    contour_factor: f32,
    contour_levels: u32,
    show_negative: bool,
    show_peak_labels: bool,
    box_zoom_2d_mode: bool,
    box_zoom_2d_anchor: Option<[f32; 2]>,

    // Solver & simulator controls
    replicas: usize,
    steps_per_sys: usize,
    assign_cost: f64,
    sim_missing: f64,
    sim_spurious: f64,
    sim_seed: u64,
    picker_snr: f32,

    // Selection & File / In-Memory Text I/O
    selected_ss: Option<usize>,
    selected_res: Option<usize>,
    peak_2d_labels: BTreeMap<(i64, i64), String>,
    context_menu_2d_ppm: Option<[f64; 2]>,
    secondary_drag_accum_2d: f32,
    bare_bruker_hint: Option<String>,
    picker_wakeup_until_ms: u64,
    water_undo_2d: Option<(Spectrum2D, Option<Spectrum3D>, Option<Spectrum3D>)>,
    #[cfg(not(target_arch = "wasm32"))]
    path_input: String,
    #[cfg(not(target_arch = "wasm32"))]
    export_prefix: String,
    show_text_io_modal: bool,
    paste_text_buf: String,
    export_text_buf: String,
    responsive_shell: ResponsiveShellState,
    wgpu_render_state: Option<egui_wgpu::RenderState>,
    suppress_hsqc_pan_until_release: bool,
    suppress_octet_pan_until_release: bool,
    #[cfg(target_arch = "wasm32")]
    pending_drop_outcome: Arc<std::sync::Mutex<Option<Result<IngestDropOutcome, String>>>>,
}

#[inline]
pub(crate) fn peak_label_key_2d(h: f64, n: f64) -> (i64, i64) {
    ((h * 10_000.0).round() as i64, (n * 10_000.0).round() as i64)
}

fn fast_median_f32(vals: &mut [f32]) -> f32 {
    if vals.is_empty() {
        return 0.0;
    }
    let mid = vals.len() / 2;
    let (_, m, _) = vals.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    *m
}

fn trimmed_slice_med_mad_max(
    vals: &[f32],
    scratch: &mut Vec<f32>,
    fallback_sigma: f32,
) -> (f32, f32, f32) {
    let mut max_abs = 0.0_f32;
    for &v in vals {
        let a = v.abs();
        if a > max_abs {
            max_abs = a;
        }
    }
    if vals.len() < 8 {
        return (0.0, fallback_sigma, max_abs);
    }
    scratch.clear();
    scratch.extend_from_slice(vals);
    let med0 = fast_median_f32(scratch);
    for x in scratch.iter_mut() {
        *x = (*x - med0).abs();
    }
    let mad0 = (1.4826 * fast_median_f32(scratch)).max(1e-12);
    if vals.len() < 24 {
        return (med0, mad0.max(fallback_sigma), max_abs);
    }
    scratch.clear();
    for &v in vals {
        if (v - med0).abs() <= 3.5 * mad0 {
            scratch.push(v);
        }
    }
    if scratch.len() < 12 {
        return (med0, mad0.max(fallback_sigma), max_abs);
    }
    let med1 = fast_median_f32(scratch);
    for x in scratch.iter_mut() {
        *x = (*x - med1).abs();
    }
    let mad1 = (1.4826 * fast_median_f32(scratch)).max(fallback_sigma);
    (med1, mad1, max_abs)
}

/// Ridge-safe 2D cross-peak picker for homonuclear (`NOESY`, `TOCSY`, `COSY`, `ROESY`) and
/// heteronuclear (`HMBC`, `19F-X`, ...) 2D spectra:
/// 1. Estimates global thermal noise `sigma0` from the 25th-percentile of `64×64` non-zero tile MADs
///    (immune to diagonal ridges, `4.70 ppm` water stripes, and zero-padded margins).
/// 2. Computes per-column (`F1` `t1`-ridge) and per-row (`F2` `t2`-baseline/water) outlier-trimmed
///    median & MAD so intense water ridges spawn zero false cross-peaks.
/// 3. Enforces `|v| >= max(snr * sigma0, contour_base)` so invisible sub-contour noise is never picked.
/// 4. Optionally excludes homonuclear diagonal self-peaks (`|x_ppm - y_ppm| < 0.045 ppm`), applies
///    `±4×±4` 2D non-maximum suppression + 3-point parabolic sub-bin interpolation, and returns up to
///    `3,000` cross-peaks sorted descending by `|height|`.
pub(crate) fn pick_2d_crosspeaks_ridge_safe(
    spec: &Spectrum2D,
    snr_threshold: f32,
    contour_base: f32,
    is_homonuclear: bool,
    allow_diagonal: bool,
    allow_negative: bool,
) -> Vec<Peak> {
    let nx = spec.nx();
    let ny = spec.ny();
    if nx < 5 || ny < 5 || spec.data.len() < nx * ny {
        return Vec::new();
    }

    // 1. 25th-percentile of 64x64 non-zero tile MADs
    let tile_w = 64usize.min(nx);
    let tile_h = 64usize.min(ny);
    let step_in_tile = if nx * ny > 1_048_576 { 2usize } else { 1usize };
    let mut tile_mads: Vec<f32> = Vec::new();
    let mut scratch: Vec<f32> = Vec::with_capacity(tile_w * tile_h);
    let mut ty0 = 0usize;
    while ty0 < ny {
        let ty1 = (ty0 + tile_h).min(ny);
        let mut tx0 = 0usize;
        while tx0 < nx {
            let tx1 = (tx0 + tile_w).min(nx);
            scratch.clear();
            let mut tile_max = 0.0_f32;
            let mut iy = ty0;
            while iy < ty1 {
                let row = &spec.data[iy * nx..iy * nx + tx1];
                let mut ix = tx0;
                while ix < tx1 {
                    let v = row[ix];
                    if v.is_finite() {
                        tile_max = tile_max.max(v.abs());
                        scratch.push(v);
                    }
                    ix += step_in_tile;
                }
                iy += step_in_tile;
            }
            if tile_max > 1e-12 && scratch.len() >= 16 {
                let med = fast_median_f32(&mut scratch);
                for x in &mut scratch {
                    *x = (*x - med).abs();
                }
                let mad = 1.4826 * fast_median_f32(&mut scratch);
                if mad.is_finite() && mad > 1e-12 {
                    tile_mads.push(mad);
                }
            }
            tx0 += tile_w;
        }
        ty0 += tile_h;
    }

    let sigma0 = if tile_mads.len() >= 4 {
        let q25 = tile_mads.len() / 4;
        let (_, s, _) = tile_mads.select_nth_unstable_by(q25, |a, b| a.total_cmp(b));
        (*s).max(1e-12)
    } else {
        estimate_contour_base_and_noise(&spec.data).1.max(1e-12)
    };

    // 2. Per-row (F2) and per-column (F1) median, MAD & max_abs
    let col_stride = (ny / 512).max(1);
    let row_stride = (nx / 512).max(1);
    let mut line_buf: Vec<f32> = Vec::with_capacity(520);
    let mut col_med = vec![0.0_f32; nx];
    let mut col_sigma = vec![sigma0; nx];
    let mut col_max_abs = vec![0.0_f32; nx];
    for ix in 0..nx {
        line_buf.clear();
        let mut iy = 0usize;
        while iy < ny {
            let v = spec.data[iy * nx + ix];
            if v.is_finite() {
                line_buf.push(v);
            }
            iy += col_stride;
        }
        let (m, s, mx) = trimmed_slice_med_mad_max(&line_buf, &mut scratch, sigma0);
        col_med[ix] = m;
        col_sigma[ix] = s;
        col_max_abs[ix] = mx;
    }

    let mut row_med = vec![0.0_f32; ny];
    let mut row_sigma = vec![sigma0; ny];
    if is_homonuclear {
        for iy in 0..ny {
            line_buf.clear();
            let row = &spec.data[iy * nx..(iy + 1) * nx];
            let mut ix = 0usize;
            while ix < nx {
                let v = row[ix];
                if v.is_finite() {
                    line_buf.push(v);
                }
                ix += row_stride;
            }
            let (m, s, _) = trimmed_slice_med_mad_max(&line_buf, &mut scratch, sigma0);
            row_med[iy] = m;
            row_sigma[iy] = s;
        }
    }

    // 3. Local-ridge + contour-floor gated 2D NMS + parabolic sub-bin refinement
    let snr = snr_threshold.max(3.0);
    let abs_floor = (snr * sigma0).max(contour_base.max(0.0));
    let rx = if nx >= 256 {
        4isize
    } else if nx >= 96 {
        2isize
    } else {
        1isize
    };
    let ry = if ny >= 256 {
        4isize
    } else if ny >= 96 {
        2isize
    } else {
        1isize
    };
    let margin_x = (rx as usize).max(1);
    let margin_y = (ry as usize).max(1);
    if nx <= 2 * margin_x || ny <= 2 * margin_y {
        return Vec::new();
    }

    let mut peaks = Vec::new();
    for iy in margin_y..(ny - margin_y) {
        let r_med = row_med[iy];
        let r_sig = row_sigma[iy].max(sigma0);
        let y_approx = spec.axis_y.index_to_ppm(iy as f64);
        for ix in margin_x..(nx - margin_x) {
            let v = spec.data[iy * nx + ix];
            if !v.is_finite() {
                continue;
            }
            let sign = if v >= abs_floor {
                1i32
            } else if allow_negative && -v >= abs_floor {
                -1i32
            } else {
                continue;
            };
            let av = v.abs();
            let c_sig = col_sigma[ix].max(sigma0);
            if (v - col_med[ix]).abs() < snr * c_sig {
                continue;
            }
            if col_max_abs[ix] > 40.0 * sigma0 && av < 0.018 * col_max_abs[ix] {
                continue;
            }
            if is_homonuclear {
                if (v - r_med).abs() < snr * r_sig {
                    continue;
                }
                if !allow_diagonal {
                    let x_approx = spec.axis_x.index_to_ppm(ix as f64);
                    if (x_approx - y_approx).abs() < 0.045 {
                        continue;
                    }
                }
            }

            let mut is_max = true;
            'nms: for dy in -ry..=ry {
                let ny_i = (iy as isize + dy) as usize;
                let nrow = &spec.data[ny_i * nx..(ny_i + 1) * nx];
                for dx in -rx..=rx {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nv = nrow[(ix as isize + dx) as usize];
                    let nav = if allow_negative {
                        nv.abs()
                    } else {
                        nv.max(0.0)
                    };
                    if nav > av || (nav == av && (dy < 0 || (dy == 0 && dx < 0))) {
                        is_max = false;
                        break 'nms;
                    }
                }
            }
            if !is_max {
                continue;
            }

            let sgn = sign as f64;
            let xm = (spec.data[iy * nx + (ix - 1)] as f64) * sgn;
            let x0 = (v as f64) * sgn;
            let xp = (spec.data[iy * nx + (ix + 1)] as f64) * sgn;
            let denom_x = xm - 2.0 * x0 + xp;
            let sub_x = if denom_x < -1e-9 {
                (0.5 * (xm - xp) / denom_x).clamp(-0.5, 0.5)
            } else {
                0.0
            };

            let ym = (spec.data[(iy - 1) * nx + ix] as f64) * sgn;
            let yp = (spec.data[(iy + 1) * nx + ix] as f64) * sgn;
            let denom_y = ym - 2.0 * x0 + yp;
            let sub_y = if denom_y < -1e-9 {
                (0.5 * (ym - yp) / denom_y).clamp(-0.5, 0.5)
            } else {
                0.0
            };

            let x_ppm = spec.axis_x.index_to_ppm(ix as f64 + sub_x);
            let y_ppm = spec.axis_y.index_to_ppm(iy as f64 + sub_y);
            if is_homonuclear && !allow_diagonal && (x_ppm - y_ppm).abs() < 0.045 {
                continue;
            }
            peaks.push(Peak {
                exp: spec.name.clone(),
                h: x_ppm,
                n: y_ppm,
                c: None,
                h2: None,
                sign,
                height: Some(v as f64),
                volume: Some(v as f64),
            });
        }
    }

    peaks.sort_by(|a, b| {
        b.height
            .unwrap_or(0.0)
            .abs()
            .total_cmp(&a.height.unwrap_or(0.0).abs())
    });
    peaks.truncate(3_000);
    peaks
}

impl NmrWorkbenchApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        cc.egui_ctx.set_theme(egui::Theme::Dark);
        let mut style = (*cc.egui_ctx.style_of(egui::Theme::Dark)).clone();
        style.visuals = egui::Visuals::dark();
        style.visuals.panel_fill = egui::Color32::from_rgb(16, 20, 28);
        style.visuals.window_fill = egui::Color32::from_rgb(20, 25, 35);
        cc.egui_ctx.set_style_of(egui::Theme::Dark, style);

        #[cfg(not(target_arch = "wasm32"))]
        let mut gpu_engine = None;
        #[cfg(not(target_arch = "wasm32"))]
        let mut bayes_gpu_solver = None;
        let mut adapter_label = None;

        if let Some(rs) = &cc.wgpu_render_state {
            let contour_res = ContourPipelineResource::new(&rs.device, rs.target_format);
            let mol3d_res = Mol3dPipelineResource::new(&rs.device, Some(rs.target_format));
            {
                let mut rend = rs.renderer.write();
                rend.callback_resources.insert(contour_res);
                rend.callback_resources.insert(mol3d_res);
            }

            let info = rs.adapter.get_info();
            let name = if info.name.trim().is_empty() {
                "WebGPU Adapter".to_string()
            } else {
                info.name.clone()
            };
            adapter_label = Some(format!("{} ({:?})", name, info.backend));

            #[cfg(not(target_arch = "wasm32"))]
            {
                let backend_str = format!("{:?}", info.backend);
                gpu_engine = Some(GpuEngine::from_device(
                    rs.device.clone(),
                    rs.queue.clone(),
                    info.name.clone(),
                    backend_str.clone(),
                ));
                bayes_gpu_solver = Some(BayesSaGpuSolver::with_device_type(
                    GpuEngine::from_device(
                        rs.device.clone(),
                        rs.queue.clone(),
                        info.name.clone(),
                        backend_str,
                    ),
                    info.device_type,
                ));
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        let default_replicas = 64;
        #[cfg(target_arch = "wasm32")]
        let default_replicas = 16;

        let mut octet_app = OctetDrawPanel::new();
        octet_app.set_backend_adapter(Box::new(CoreOctetBackendAdapter));

        let ingest_panel = IngestPanelState {
            open: false,
            ..Default::default()
        };

        let mut app = Self {
            workspace_mode: WorkspaceMode::ProteinNmr,
            octet_app,
            dosy_panel: DosyPanelState::default(),
            metabolomics_panel: MetabolomicsStudioPanel::new(),
            solver_kind: GuiSolverKind::default(),
            regime_kind: GuiRegimeKind::default(),
            metal_ligands_input: String::new(),
            idp_temp_k: 298.15,
            idp_ph_input: "6.5".to_string(),
            c13_picked_dataset: None,
            propensity_report: None,
            show_propensity_panel: false,
            #[cfg(not(target_arch = "wasm32"))]
            use_gpu: gpu_engine.is_some(),
            #[cfg(not(target_arch = "wasm32"))]
            gpu_engine,
            #[cfg(not(target_arch = "wasm32"))]
            bayes_gpu_solver,
            use_neural_picker: true,
            enable_axis_rescue: true,
            axis_rescue_notices: Vec::new(),
            axis_rescue_audits: Vec::new(),
            adapter_label,
            status_msg: String::new(),
            last_solve_ms: None,
            last_bayes_sa_telemetry: None,
            priors: Priors::provisional(),
            sidechain_priors: SidechainPriors::empirical_bmrb(),
            reference: None,
            sequence: Vec::new(),
            seq_ids: Vec::new(),
            peaks: Vec::new(),
            truth: None,
            systems: Vec::new(),
            u_mat: None,
            p_mat: None,
            assign_res: None,
            pred_ref: None,
            pred_conf: BTreeMap::new(),
            pred_ambiguity_codes: BTreeMap::new(),
            metrics: None,
            sidechain_ribbon_state: SequenceRibbonState::new(),
            show_sidechain_inspector: false,
            overwrite_curated_rings: false,
            mol3d_panel: Mol3dViewerPanel::new(),
            show_mol3d_viewport: false,
            loaded_pdb_text: None,
            relax_panel: RelaxPanelState::new(),
            show_relax_panel: false,
            dynamics_panel: DynamicsStudioPanel::new(),
            show_dynamics_panel: false,
            stack1d_panel: Stack1dState::default(),
            show_stack1d_panel: true,
            show_strips_dock: false,
            show_workflow_guide: false,
            show_advanced_protein_bar: false,
            show_sequence_ribbon: true,
            paper_white_canvas: true,
            workbook: WorkbookState::default(),
            show_native_global_open_bar: false,
            hsqc_spec: None,
            hsqc_arc: Arc::new(Vec::new()),
            hsqc_gen: 1,
            hncacb_3d: None,
            hnco_3d: None,
            extra_3d_count: 0,
            overlay_layers: Vec::new(),
            primary_2d_label: "15N-HSQC".to_string(),
            primary_2d_visible: true,
            primary_ppm_nudge: [0.0, 0.0],
            next_layer_id: 2,
            active_overlay_idx: 0,
            ingest_panel,
            command_palette: CommandPalette::new(),
            chord_dispatcher: ChordDispatcher::new(),
            session: WorkbenchSessionController::default(),
            crosshair: SyncCrosshair::default(),
            strip_tiler: StripTilerPanel::new(),
            use_match_finder_strips: true,
            qc_report: None,
            raw_fids_total: 0,
            unprocessed_fids_remaining: 0,
            has_structure_or_dynamics: false,
            exported_in_session: false,
            show_contour_modal: false,
            show_overlay_manager: false,
            show_assign_popover: false,
            show_qc_modal: false,
            show_about_modal: false,
            mcp_bridge: McpBridgeState::default(),
            show_mcp_bridge_window: false,
            mcp_last_viewport_key: (0, 0, 0, -1),
            assign_target_res: 0,
            assign_lock_check: true,
            view_h_left: 10.4,
            view_h_right: 6.1,
            view_n_top: 103.0,
            view_n_bottom: 135.0,
            contour_base: 8.5,
            contour_noise_ref: 1.8,
            contour_factor: 1.42,
            contour_levels: 10,
            show_negative: true,
            show_peak_labels: true,
            box_zoom_2d_mode: false,
            box_zoom_2d_anchor: None,
            replicas: default_replicas,
            steps_per_sys: 1500,
            assign_cost: 0.0,
            sim_missing: 0.06,
            sim_spurious: 0.04,
            sim_seed: 0,
            picker_snr: 5.0,
            selected_ss: None,
            selected_res: None,
            peak_2d_labels: BTreeMap::new(),
            context_menu_2d_ppm: None,
            secondary_drag_accum_2d: 0.0,
            bare_bruker_hint: None,
            picker_wakeup_until_ms: 0,
            water_undo_2d: None,
            #[cfg(not(target_arch = "wasm32"))]
            path_input: "data/fixtures/synthetic_ubiquitin.str".to_string(),
            #[cfg(not(target_arch = "wasm32"))]
            export_prefix: "result_gui".to_string(),
            show_text_io_modal: false,
            paste_text_buf: String::new(),
            export_text_buf: String::new(),
            responsive_shell: ResponsiveShellState::new(),
            wgpu_render_state: cc.wgpu_render_state.clone(),
            suppress_hsqc_pan_until_release: false,
            suppress_octet_pan_until_release: false,
            #[cfg(target_arch = "wasm32")]
            pending_drop_outcome: Arc::new(std::sync::Mutex::new(None)),
        };

        app.load_builtin_demo_suite();
        app
    }

    fn estimate_2d_contour_base(data: &[f32]) -> (f32, f32) {
        estimate_contour_base_and_noise(data)
    }

    fn arm_browser_picker_wakeup(&mut self) {
        self.picker_wakeup_until_ms = u64::MAX - 1;
    }

    fn canonical_2d_axis_nucleus(raw_label: &str, min_ppm: f64, max_ppm: f64) -> &'static str {
        let up = raw_label
            .trim()
            .trim_matches(|c: char| c == '<' || c == '>' || c == '"' || c == '\'')
            .trim()
            .to_ascii_uppercase();
        match up.as_str() {
            "1H" | "H1" | "H" | "HN" | "HA" | "HB" | "PROTON" => "1H",
            "15N" | "N15" | "N" | "NH" => "15N",
            "13C" | "C13" | "C" | "CA" | "CB" | "CO" | "CACB" => "13C",
            "19F" | "F19" | "F" => "19F",
            "31P" | "P31" | "P" => "31P",
            "2H" | "H2" | "D" => "2H",
            "11B" | "B11" => "11B",
            "29SI" | "SI29" => "29Si",
            _ => {
                if max_ppm <= 19.0 && min_ppm >= -6.0 {
                    "1H"
                } else if min_ppm >= 88.0 && max_ppm <= 152.0 && (max_ppm - min_ppm) <= 65.0 {
                    "15N"
                } else if max_ppm > 19.0 {
                    "13C"
                } else {
                    "1H"
                }
            }
        }
    }

    fn pretty_nucleus_unicode(nuc: &str) -> &'static str {
        match nuc {
            "1H" => "¹H",
            "15N" => "¹⁵N",
            "13C" => "¹³C",
            "19F" => "¹⁹F",
            "31P" => "³¹P",
            "2H" => "²H",
            "11B" => "¹¹B",
            "29Si" => "²⁹Si",
            _ => "X",
        }
    }

    fn active_2d_nuclei(&self) -> (&'static str, &'static str) {
        if let Some(spec) = &self.hsqc_spec {
            let nx = Self::canonical_2d_axis_nucleus(
                &spec.axis_x.label,
                spec.axis_x.min_ppm(),
                spec.axis_x.max_ppm(),
            );
            let ny = Self::canonical_2d_axis_nucleus(
                &spec.axis_y.label,
                spec.axis_y.min_ppm(),
                spec.axis_y.max_ppm(),
            );
            (nx, ny)
        } else {
            ("1H", "15N")
        }
    }

    fn view_limits_for_active_2d(&self) -> ViewBounds2dLimits {
        let Some(spec) = &self.hsqc_spec else {
            return ViewBounds2dLimits::default_hn();
        };
        let mut x_lo = spec.axis_x.min_ppm() as f32;
        let mut x_hi = spec.axis_x.max_ppm() as f32;
        let mut y_lo = spec.axis_y.min_ppm() as f32;
        let mut y_hi = spec.axis_y.max_ppm() as f32;

        for (_, ov) in &self.overlay_layers {
            if !ov.visible || ov.dims[0] == 0 || ov.dims[1] == 0 {
                continue;
            }
            let ox0 = ov.axis_ppm[0];
            let ox1 = ox0 + ov.axis_ppm[1] * ((ov.dims[0].saturating_sub(1)) as f32);
            let oy0 = ov.axis_ppm[2];
            let oy1 = oy0 + ov.axis_ppm[3] * ((ov.dims[1].saturating_sub(1)) as f32);
            let ov_x_lo = ox0.min(ox1);
            let ov_x_hi = ox0.max(ox1);
            let ov_y_lo = oy0.min(oy1);
            let ov_y_hi = oy0.max(oy1);
            // Union bounds when overlay shares a compatible ppm scale (e.g. NOESY + TOCSY or HSQC + HMBC)
            if (ov_x_hi - ov_x_lo).is_finite()
                && (ov_y_hi - ov_y_lo).is_finite()
                && (ov_x_lo <= x_hi + 25.0 && ov_x_hi >= x_lo - 25.0)
                && (ov_y_lo <= y_hi + 80.0 && ov_y_hi >= y_lo - 80.0)
            {
                x_lo = x_lo.min(ov_x_lo);
                x_hi = x_hi.max(ov_x_hi);
                y_lo = y_lo.min(ov_y_lo);
                y_hi = y_hi.max(ov_y_hi);
            }
        }

        if !x_lo.is_finite()
            || !x_hi.is_finite()
            || !y_lo.is_finite()
            || !y_hi.is_finite()
            || (x_hi - x_lo).abs() < 1e-4
            || (y_hi - y_lo).abs() < 1e-4
        {
            return ViewBounds2dLimits::default_hn();
        }
        let x_span = (x_hi - x_lo).abs();
        let y_span = (y_hi - y_lo).abs();
        let x_pad = (x_span * 0.15).max(0.20);
        let y_pad = (y_span * 0.15).max(0.20);
        ViewBounds2dLimits {
            x_min_ppm: x_lo - x_pad,
            x_max_ppm: x_hi + x_pad,
            y_min_ppm: y_lo - y_pad,
            y_max_ppm: y_hi + y_pad,
            min_x_span_ppm: (x_span * 0.008).clamp(0.03, 0.35),
            min_y_span_ppm: (y_span * 0.008).clamp(0.03, 1.50),
        }
    }

    fn reset_view_to_spec(&mut self) {
        self.box_zoom_2d_anchor = None;
        if let Some(spec) = &self.hsqc_spec {
            let mut x_lo = spec.axis_x.min_ppm() as f32;
            let mut x_hi = spec.axis_x.max_ppm() as f32;
            let mut y_lo = spec.axis_y.min_ppm() as f32;
            let mut y_hi = spec.axis_y.max_ppm() as f32;

            for (_, ov) in &self.overlay_layers {
                if !ov.visible || ov.dims[0] == 0 || ov.dims[1] == 0 {
                    continue;
                }
                let ox0 = ov.axis_ppm[0];
                let ox1 = ox0 + ov.axis_ppm[1] * ((ov.dims[0].saturating_sub(1)) as f32);
                let oy0 = ov.axis_ppm[2];
                let oy1 = oy0 + ov.axis_ppm[3] * ((ov.dims[1].saturating_sub(1)) as f32);
                let (ov_x_lo, ov_x_hi) = (ox0.min(ox1), ox0.max(ox1));
                let (ov_y_lo, ov_y_hi) = (oy0.min(oy1), oy0.max(oy1));
                if ov_x_lo <= x_hi + 25.0
                    && ov_x_hi >= x_lo - 25.0
                    && ov_y_lo <= y_hi + 80.0
                    && ov_y_hi >= y_lo - 80.0
                {
                    x_lo = x_lo.min(ov_x_lo);
                    x_hi = x_hi.max(ov_x_hi);
                    y_lo = y_lo.min(ov_y_lo);
                    y_hi = y_hi.max(ov_y_hi);
                }
            }

            self.view_h_left = x_hi;
            self.view_h_right = x_lo;
            self.view_n_top = y_lo;
            self.view_n_bottom = y_hi;
            let (base, nref) = Self::estimate_2d_contour_base(&spec.data);
            self.contour_base = base;
            self.contour_noise_ref = nref;
            self.crosshair.snap_to(
                0.5 * (x_lo as f64 + x_hi as f64),
                0.5 * (y_lo as f64 + y_hi as f64),
                None,
            );
        }
    }

    fn clear_stale_protein_demo_state(&mut self, keep_sequence: bool, keep_3d_cubes: bool) {
        if !keep_sequence {
            self.sequence.clear();
            self.seq_ids.clear();
            self.octet_app.set_loaded_protein_sequence(Vec::new());
            let saved_ph = self.sidechain_ribbon_state.sample_ph;
            self.sidechain_ribbon_state = SequenceRibbonState::new();
            self.sidechain_ribbon_state.set_sample_ph(saved_ph);
        }
        self.reference = None;
        self.peaks.clear();
        self.truth = None;
        self.systems.clear();
        self.u_mat = None;
        self.p_mat = None;
        self.assign_res = None;
        self.pred_ref = None;
        self.pred_conf.clear();
        self.pred_ambiguity_codes.clear();
        self.c13_picked_dataset = None;
        self.propensity_report = None;
        self.metrics = None;
        self.qc_report = None;
        self.last_bayes_sa_telemetry = None;
        self.selected_ss = None;
        self.selected_res = None;
        self.loaded_pdb_text = None;
        self.mol3d_panel = Mol3dViewerPanel::new();
        self.has_structure_or_dynamics = false;
        if !keep_3d_cubes {
            self.hncacb_3d = None;
            self.hnco_3d = None;
            self.extra_3d_count = 0;
            self.show_mol3d_viewport = false;
            self.show_strips_dock = false;
            self.responsive_shell.tablet_dock = TabletActiveDock::Hidden;
        }
    }

    fn compute_matrices(&self, sys: &[SpinSystem], scfg: &ScoreConfig) -> (Matrix2D, Matrix2D) {
        #[cfg(not(target_arch = "wasm32"))]
        if self.use_gpu {
            if let Some(gpu) = &self.gpu_engine {
                if let Ok(pair) = gpu.score_matrices_gpu(sys, &self.sequence, &self.priors, scfg) {
                    return pair;
                }
            }
        }
        (
            unary_matrix(sys, &self.sequence, &self.priors, scfg),
            pair_matrix(sys, scfg),
        )
    }

    fn rebuild_systems_and_matrices(&mut self) {
        match build_spin_systems(&self.peaks, &Tolerances::default(), 3, false) {
            Ok(sys) => {
                let locked: Vec<(usize, usize)> = self
                    .session
                    .state
                    .locked_assignments_vec()
                    .into_iter()
                    .filter(|&(s, k)| s < sys.len() && k < self.sequence.len())
                    .collect();
                let scfg = ScoreConfig {
                    assign_cost: self.assign_cost,
                    locked_assignments: locked,
                    ..ScoreConfig::default()
                };
                let (u, p) = self.compute_matrices(&sys, &scfg);
                self.u_mat = Some(u);
                self.p_mat = Some(p);
                self.systems = sys;
                self.selected_ss = match self.selected_ss {
                    Some(s) if s < self.systems.len() => Some(s),
                    _ => (!self.systems.is_empty()).then_some(0),
                };
            }
            Err(e) => {
                self.status_msg = format!("Spin system build error: {e}");
            }
        }
    }

    fn regenerate_synthetic_spectra(&mut self) {
        let rcfg = SpectrumRenderConfig {
            nx_h: 512,
            ny_n: 256,
            nz_c: 96,
            seed: self.sim_seed.wrapping_add(7),
            ..SpectrumRenderConfig::default()
        };
        let hsqc = render_hsqc_2d(&self.peaks, &rcfg);
        self.hsqc_arc = Arc::new(hsqc.data.clone());
        self.hsqc_gen = self.hsqc_gen.wrapping_add(1);
        self.workbook
            .set_single_2d_page("15N-HSQC", hsqc.clone(), true);
        self.hsqc_spec = Some(hsqc);
        self.hncacb_3d = Some(render_triple_3d("HNCACB", &self.peaks, &rcfg));
        self.hnco_3d = Some(render_triple_3d("HNCO", &self.peaks, &rcfg));
        self.extra_3d_count = 0;
        self.overlay_layers.clear();
        self.primary_2d_label = "15N-HSQC".to_string();
        self.primary_2d_visible = true;
        self.primary_ppm_nudge = [0.0, 0.0];
        self.reset_view_to_spec();
    }

    fn sync_session_from_app_fresh(&mut self, protein_name: &str) {
        let mut st = WorkbenchSessionState::new(protein_name, self.sequence.clone());
        if self.seq_ids.len() == self.sequence.len() {
            st.seq_ids = self.seq_ids.clone();
        }
        st.peaks = self.peaks.clone();
        let mut hsqc_layer = SessionContourParams::new("HSQC");
        hsqc_layer.base_level = self.contour_base;
        hsqc_layer.level_factor = self.contour_factor;
        hsqc_layer.num_levels = self.contour_levels;
        hsqc_layer.show_negative = self.show_negative;
        hsqc_layer.ppm_nudge = self.primary_ppm_nudge;
        st.contour_layers = vec![hsqc_layer];
        self.session = WorkbenchSessionController::new(st);
    }

    fn sync_session_assignments_from_solver(&mut self) {
        self.session.state.sequence = self.sequence.clone();
        if self.seq_ids.len() == self.sequence.len() {
            self.session.state.seq_ids = self.seq_ids.clone();
        }
        self.session.state.peaks = self.peaks.clone();
        self.session.state.qc_verdict = self.qc_report.as_ref().map(|q| q.verdict);

        let mut new_assign: BTreeMap<usize, ResidueAssignmentState> = BTreeMap::new();
        if let Some(pred) = &self.pred_ref {
            for (&(res_idx, ref atom), &val) in &pred.shifts {
                if res_idx < self.sequence.len() {
                    let entry =
                        new_assign
                            .entry(res_idx)
                            .or_insert_with(|| ResidueAssignmentState {
                                spin_system: None,
                                shifts: BTreeMap::new(),
                                confidence: self.pred_conf.get(&res_idx).copied().unwrap_or(0.0),
                            });
                    entry.shifts.insert(atom.clone(), val);
                }
            }
        }
        if let Some(res) = &self.assign_res {
            for (ss, &pos) in res.position.iter().enumerate() {
                if pos >= 0 && (pos as usize) < self.sequence.len() {
                    let k = pos as usize;
                    let c = res.confidence.get(ss).copied().unwrap_or(0.0);
                    let entry = new_assign
                        .entry(k)
                        .or_insert_with(|| ResidueAssignmentState {
                            spin_system: Some(ss),
                            shifts: BTreeMap::new(),
                            confidence: c,
                        });
                    entry.spin_system = Some(ss);
                    entry.confidence = c;
                }
            }
        }
        // Keep lock-assignment coherence invariant satisfied
        self.session.state.locked_residues.retain(|&ss, k| {
            new_assign
                .get(k)
                .is_some_and(|st| st.spin_system == Some(ss))
        });
        self.session.state.assignments = new_assign;
        self.sync_mol3d_from_app();
    }

    fn sync_app_from_session(&mut self) {
        self.sequence = self.session.state.sequence.clone();
        self.seq_ids = if self.session.state.seq_ids.len() == self.sequence.len() {
            self.session.state.seq_ids.clone()
        } else {
            (1..=(self.sequence.len() as i32)).collect()
        };
        self.octet_app
            .set_loaded_protein_sequence(self.sequence.clone());

        if self.peaks != self.session.state.peaks {
            self.peaks = self.session.state.peaks.clone();
            self.rebuild_systems_and_matrices();
        } else if !self.sequence.is_empty() && !self.systems.is_empty() {
            let locked = self.session.state.locked_assignments_vec();
            let scfg = ScoreConfig {
                assign_cost: self.assign_cost,
                locked_assignments: locked,
                ..ScoreConfig::default()
            };
            self.u_mat = Some(unary_matrix(
                &self.systems,
                &self.sequence,
                &self.priors,
                &scfg,
            ));
        }

        if let Some(hsqc_layer) = self
            .session
            .state
            .contour_layers
            .iter()
            .find(|l| l.layer_id == "HSQC")
        {
            self.contour_base = hsqc_layer.base_level.max(1e-4);
            self.contour_factor = hsqc_layer.level_factor.max(1.05);
            self.contour_levels = hsqc_layer.num_levels.clamp(1, 24);
            self.show_negative = hsqc_layer.show_negative;
            self.primary_ppm_nudge = hsqc_layer.ppm_nudge;
        }
        if let Some(ref spec) = self.hsqc_spec {
            let (_, nref) = Self::estimate_2d_contour_base(&spec.data);
            self.contour_noise_ref = nref.max(self.contour_base / 6.0).max(1e-6);
        }

        let n_sys = self.systems.len();
        let mut position = vec![-1i32; n_sys];
        let mut confidence = vec![0.0f64; n_sys];
        for (&res_idx, st) in &self.session.state.assignments {
            if let Some(ss) = st.spin_system {
                if ss < n_sys && res_idx < self.sequence.len() {
                    position[ss] = res_idx as i32;
                    confidence[ss] = st.confidence;
                }
            }
        }
        let (mut pred, mut conf) = self.session.state.to_reference_and_confidences();
        if n_sys > 0 {
            let synth_res = AssignResult {
                position: position.clone(),
                score: self.assign_res.as_ref().map_or(0.0, |r| r.score),
                confidence: confidence.clone(),
                run_scores: Vec::new(),
            };
            let (derived_ref, derived_conf) = to_reference(
                &synth_res,
                &self.systems,
                &self.sequence,
                &self.seq_ids,
                0.0,
                Some(&self.priors),
            );
            for (key, val) in derived_ref.shifts {
                pred.shifts.entry(key).or_insert(val);
            }
            for (k, c) in derived_conf {
                conf.entry(k).or_insert(c);
            }
            self.assign_res = Some(synth_res);
        } else {
            self.assign_res = None;
        }
        self.sidechain_ribbon_state
            .sync_inferred_variants_from_reference(&pred);
        self.pred_ref = Some(pred);
        self.pred_conf = conf;
        self.refresh_propensity_report();
        self.sync_mol3d_from_app();
    }

    fn sync_mol3d_from_app(&mut self) {
        let src_ref = self.pred_ref.as_ref().or(self.reference.as_ref());
        self.mol3d_panel
            .sync_nmr_tracks(&self.pred_conf, src_ref, Some(&self.priors));
        self.mol3d_panel
            .highlight_sequence_residue(self.selected_res, false);
        self.mol3d_panel.flush_pending_updates();
    }

    fn load_structure_3d_in_memory(&mut self, text: &str, source_label: &str) {
        match self
            .mol3d_panel
            .load_structure_text(text, &self.sequence, &self.seq_ids)
        {
            Ok(()) => {
                if self.sequence.is_empty() {
                    if let Some(ens) = self.mol3d_panel.ensemble.as_ref() {
                        if !ens.primary_sequence.is_empty() {
                            self.sequence = ens.primary_sequence.clone();
                            self.seq_ids = ens.primary_seq_ids.clone();
                            self.octet_app
                                .set_loaded_protein_sequence(self.sequence.clone());
                            self.session.state.sequence = self.sequence.clone();
                            self.session.state.seq_ids = self.seq_ids.clone();
                        }
                    }
                }
                if text.lines().any(|ln| ln.starts_with("ATOM  ")) {
                    self.loaded_pdb_text = Some(text.to_string());
                }
                self.show_mol3d_viewport = true;
                self.has_structure_or_dynamics = true;
                self.workspace_mode = WorkspaceMode::ProteinNmr;
                self.sync_mol3d_from_app();
                self.status_msg = format!(
                    "Loaded 3D structure from {source_label}: {}",
                    self.mol3d_panel.status_banner
                );
            }
            Err(e) => {
                self.status_msg = format!("3D structure load error ({source_label}): {e}");
            }
        }
    }

    fn apply_relax_panel_actions(&mut self, actions: Vec<RelaxPanelAction>) {
        for act in actions {
            match act {
                RelaxPanelAction::SelectResidueIndex(k) => {
                    if k < self.sequence.len() {
                        self.select_residue_and_sync_strip(k);
                    }
                }
                RelaxPanelAction::ExportTo3dTrack {
                    track,
                    s2_by_seq_idx,
                    rex_by_seq_idx,
                    rex_max_s_inv,
                } => {
                    let count = match track {
                        QuantitativeTrack::LipariSzaboS2 => s2_by_seq_idx.len(),
                        QuantitativeTrack::ChemicalExchangeRex => rex_by_seq_idx.len(),
                        _ => s2_by_seq_idx.len(),
                    };
                    self.mol3d_panel.track_data.lipari_szabo_s2 = s2_by_seq_idx;
                    self.mol3d_panel.track_data.rex_s_inv = rex_by_seq_idx;
                    self.mol3d_panel.track_data.rex_max_s_inv = rex_max_s_inv;
                    self.mol3d_panel.track_data.active_track = track;
                    self.mol3d_panel.track_data.putty_enabled = true;
                    self.show_mol3d_viewport = true;
                    self.has_structure_or_dynamics = true;
                    self.mol3d_panel.rebuild_for_test();
                    let track_label = match track {
                        QuantitativeTrack::LipariSzaboS2 => "Lipari–Szabo S²",
                        QuantitativeTrack::ChemicalExchangeRex => "Chemical Exchange R_ex",
                        _ => "Quantitative Track",
                    };
                    self.status_msg = format!(
                        "Mapped {track_label} ({count} residues) onto 3D PDB B-factor putty"
                    );
                }
                RelaxPanelAction::StatusMessage(msg) => {
                    self.has_structure_or_dynamics = true;
                    self.status_msg = msg;
                }
            }
        }
    }

    fn apply_dynamics_panel_output(&mut self, out: DynamicsStudioOutput) {
        if let Some(sid) = out.clicked_seq_id {
            let res_idx = if !self.seq_ids.is_empty() {
                self.seq_ids.iter().position(|&id| id == sid)
            } else if sid >= 1 {
                Some((sid - 1) as usize)
            } else {
                None
            };
            if let Some(k) = res_idx.filter(|&k| k < self.sequence.len()) {
                self.select_residue_and_sync_strip(k);
            }
        }
        if let Some(proj) = out.overlay_projection {
            let count = proj.values_by_seq_id.len();
            let track_label = match proj.track {
                QuantitativeTrack::LipariSzaboS2 => "Lipari–Szabo S²",
                QuantitativeTrack::ChemicalExchangeRex => "Chemical Exchange R_ex",
                QuantitativeTrack::LigandCsp => "Ligand Titration CSP Δδ_comb",
                _ => "Quantitative Track",
            };
            proj.apply_to(&mut self.mol3d_panel.track_data, &self.seq_ids);
            self.show_mol3d_viewport = true;
            self.has_structure_or_dynamics = true;
            self.mol3d_panel.rebuild_for_test();
            self.status_msg =
                format!("Mapped {track_label} ({count} residues) onto 3D structure putty");
        }
    }

    fn apply_mol3d_sync_action(&mut self, sync: Mol3dSyncAction) {
        if sync.residue_idx < self.sequence.len() {
            self.select_residue_and_sync_strip(sync.residue_idx);
        }
        if sync.open_sidechain_graph {
            self.show_sidechain_inspector = true;
        }
        sync.apply_to_crosshair(&mut self.crosshair);
        let h_ppm = sync.atom_jump.as_ref().and_then(|j| j.backbone_h_ppm);
        let n_ppm = sync.atom_jump.as_ref().and_then(|j| j.backbone_n_ppm);
        if let (Some(h), Some(n)) = (h_ppm, n_ppm) {
            let h_f = h as f32;
            let n_f = n as f32;
            let h_min = self.view_h_left.min(self.view_h_right);
            let h_max = self.view_h_left.max(self.view_h_right);
            let n_min = self.view_n_top.min(self.view_n_bottom);
            let n_max = self.view_n_top.max(self.view_n_bottom);
            if h_f < h_min || h_f > h_max || n_f < n_min || n_f > n_max {
                let half_h = 0.5 * (self.view_h_left - self.view_h_right).abs().max(0.8);
                let half_n = 0.5 * (self.view_n_bottom - self.view_n_top).abs().max(6.0);
                self.view_h_left = h_f + half_h;
                self.view_h_right = h_f - half_h;
                self.view_n_top = n_f - half_n;
                self.view_n_bottom = n_f + half_n;
            }
        }
        let shift_summary = if let Some(ref j) = sync.atom_jump {
            match (j.backbone_h_ppm, j.backbone_n_ppm) {
                (Some(h), Some(n)) => {
                    format!(" ({} {:.3} ppm; ¹H {h:.3}, ¹⁵N {n:.2})", j.atom_name, j.ppm)
                }
                _ => format!(" ({} {:.3} ppm)", j.atom_name, j.ppm),
            }
        } else {
            String::new()
        };
        let atom_lbl = if sync.clicked_atom.is_empty() {
            "CA"
        } else {
            sync.clicked_atom.as_str()
        };
        self.status_msg = format!(
            "3D Pick → {} {} {}{}",
            sync.residue_name, sync.seq_id, atom_lbl, shift_summary
        );
    }

    fn refresh_propensity_report(&mut self) {
        let Some(src_ref) = self.pred_ref.as_ref().or(self.reference.as_ref()) else {
            self.propensity_report = None;
            return;
        };
        if src_ref.sequence.is_empty() {
            self.propensity_report = None;
            return;
        }
        let temp_k = self.idp_temp_k.clamp(250.0, 360.0);
        let mut cfg = PropensityConfig::default().with_temperature_kelvin(temp_k);
        if let Ok(ph) = self.idp_ph_input.trim().parse::<f64>() {
            if (1.0..=12.0).contains(&ph) {
                cfg = cfg.with_sample_ph(ph);
            }
        }
        self.propensity_report = Some(compute_idp_propensity(src_ref, &cfg));
    }

    fn raw_hsqc_locks_for_peaks(&self) -> Vec<(usize, usize)> {
        if self.session.state.locked_residues.is_empty() {
            return Vec::new();
        }
        let hsqc_roots: Vec<(f64, f64)> = self
            .peaks
            .iter()
            .filter(|p| p.exp == "HSQC")
            .map(|p| (p.h, p.n))
            .collect();
        let mut out = Vec::new();
        for (&ss_idx, &res_idx) in &self.session.state.locked_residues {
            if let Some(ss) = self.systems.get(ss_idx) {
                if let Some((raw_i, _)) = hsqc_roots
                    .iter()
                    .enumerate()
                    .filter(|&(_, &(rh, rn))| {
                        (rh - ss.h).abs() <= 0.045 && (rn - ss.n).abs() <= 0.45
                    })
                    .min_by(|&(_, &(ah, an)), &(_, &(bh, bn))| {
                        let da = ((ah - ss.h) / 0.03).powi(2) + ((an - ss.n) / 0.3).powi(2);
                        let db = ((bh - ss.h) / 0.03).powi(2) + ((bn - ss.n) / 0.3).powi(2);
                        da.total_cmp(&db)
                    })
                {
                    out.push((raw_i, res_idx));
                }
            } else if ss_idx < hsqc_roots.len() {
                out.push((ss_idx, res_idx));
            }
        }
        out
    }

    fn load_demo_and_solve(&mut self) {
        let ref_data = build_synthetic_ubiquitin();
        self.load_from_reference(ref_data);
        let _ = self.mol3d_panel.load_structure_text(
            DEMO_UBIQUITIN_1UBQ_PDB,
            &self.sequence,
            &self.seq_ids,
        );
        self.loaded_pdb_text = Some(DEMO_UBIQUITIN_1UBQ_PDB.to_string());
        self.has_structure_or_dynamics = true;
        self.workspace_mode = WorkspaceMode::ProteinNmr;
        self.show_stack1d_panel = false;
        self.show_strips_dock = true;
        self.run_assignment();
    }

    fn load_small_molecule_1h_sample(&mut self) {
        let (tr_1h, rois_1h) = build_showcase_small_molecule_1h_trace();
        let mut tr_13c = build_showcase_small_molecule_13c_trace();
        tr_13c.visible = false;
        let mut stack = Stack1dState::with_traces(vec![tr_1h, tr_13c]);
        stack.ref_trace_idx = 0;
        stack.view_mode = Stack1dViewMode::SuperimposedOverlay;
        stack.roi_regions = rois_1h;
        stack.target_total_protons = 12.0;
        stack.fit_full_ppm_bounds();
        stack.refresh_active_trace_peaks_and_multiplets();
        if !stack.detected_multiplet_integrals.is_empty() {
            stack.normalize_multiplet_integral_to(0, 1.0);
        }
        self.stack1d_panel = stack;
        self.activate_1d_studio_mode();
        self.status_msg =
            "Loaded small-molecule showcase: 600 MHz ¹H + 151 MHz ¹³C{¹H} (CDCl₃)".to_string();
    }

    fn load_kinetics_stack_sample(&mut self) {
        let (traces, rois) = build_kinetics_demo_traces();
        let mut stack = Stack1dState::with_traces(traces);
        stack.view_mode = Stack1dViewMode::StackedOffset;
        stack.roi_regions = rois;
        stack.selected_inspector_tab = InspectorTab1d::MultiTraceStack;
        stack.fit_full_ppm_bounds();
        stack.refresh_active_trace_peaks_and_multiplets();
        self.stack1d_panel = stack;
        self.activate_1d_studio_mode();
        self.status_msg = "Loaded 8-trace reaction kinetics 1D stack (A → B)".to_string();
    }

    fn save_live_2d_into_workbook(&mut self) {
        if let Some(pg) = self.workbook.pages_2d.get_mut(self.workbook.active_2d_idx) {
            pg.view_ppm = [
                self.view_h_left,
                self.view_h_right,
                self.view_n_top,
                self.view_n_bottom,
            ];
            pg.base_level = self.contour_base;
            pg.noise_ref = self.contour_noise_ref;
            pg.level_factor = self.contour_factor;
            pg.num_levels = self.contour_levels;
            pg.show_negative = self.show_negative;
            pg.ppm_nudge = self.primary_ppm_nudge;
            pg.peaks = self.peaks.clone();
            pg.peak_2d_labels = self.peak_2d_labels.clone();
            pg.c13_picked_dataset = self.c13_picked_dataset.clone();
            pg.water_undo_2d = self.water_undo_2d.clone();
            pg.zoom_history = self.workbook.two_d.zoom_history.clone();
            pg.pick_diagonal = self.workbook.two_d.pick_diagonal;
        }
    }

    fn prepare_workbook_for_user_append(&mut self) {
        if !self.show_stack1d_panel
            && self.workspace_mode == WorkspaceMode::ProteinNmr
            && !self.workbook.pages_2d.is_empty()
        {
            self.save_live_2d_into_workbook();
        }
        if !self.stack1d_panel.traces.is_empty() {
            self.stack1d_panel.save_active_trace_curation_state();
        }
        self.workbook.evict_demo_2d_pages();
        self.stack1d_panel.evict_demo_traces();
    }

    fn sync_overlay_layers_from_workbook(&mut self) {
        self.overlay_layers = self
            .workbook
            .build_compatible_overlay_layers(self.paper_white_canvas);
    }

    fn activate_2d_workbook_page(&mut self, idx: usize) {
        if idx >= self.workbook.pages_2d.len() {
            return;
        }
        if !self.show_stack1d_panel
            && self.workspace_mode == WorkspaceMode::ProteinNmr
            && !self.workbook.pages_2d.is_empty()
        {
            self.save_live_2d_into_workbook();
        } else if self.show_stack1d_panel && !self.stack1d_panel.traces.is_empty() {
            self.stack1d_panel.save_active_trace_curation_state();
        }
        self.workbook.active_2d_idx = idx;
        let pg = self.workbook.pages_2d[idx].clone();
        self.primary_2d_label = pg.title.clone();
        self.primary_2d_visible = true;
        self.primary_ppm_nudge = pg.ppm_nudge;
        self.hsqc_arc = Arc::clone(&pg.data_arc);
        self.hsqc_gen = self.hsqc_gen.wrapping_add(1);
        self.hsqc_spec = Some(pg.spec);
        self.view_h_left = pg.view_ppm[0];
        self.view_h_right = pg.view_ppm[1];
        self.view_n_top = pg.view_ppm[2];
        self.view_n_bottom = pg.view_ppm[3];
        self.contour_base = pg.base_level;
        self.contour_noise_ref = pg.noise_ref.max(pg.base_level / 6.0).max(1e-6);
        self.contour_factor = pg.level_factor;
        self.contour_levels = pg.num_levels;
        self.show_negative = pg.show_negative;
        self.peaks = pg.peaks;
        self.peak_2d_labels = pg.peak_2d_labels;
        self.c13_picked_dataset = pg.c13_picked_dataset;
        self.water_undo_2d = pg.water_undo_2d;
        self.workbook.two_d.zoom_history = pg.zoom_history;
        self.workbook.two_d.pick_diagonal = pg.pick_diagonal;
        self.workbook.two_d.selected_peak_idx = None;
        self.workbook.two_d.dragging_peak_idx = None;
        self.workbook.two_d.pick_box_start_pos = None;
        self.box_zoom_2d_anchor = None;
        self.crosshair.snap_to(
            0.5 * (self.view_h_left as f64 + self.view_h_right as f64),
            0.5 * (self.view_n_top as f64 + self.view_n_bottom as f64),
            self.crosshair.c_ppm,
        );
        self.sync_overlay_layers_from_workbook();
        self.workspace_mode = WorkspaceMode::ProteinNmr;
        self.show_stack1d_panel = false;
        self.show_relax_panel = false;
        self.show_dynamics_panel = false;
        if self.active_2d_nuclei() == ("13C", "15N") {
            self.regime_kind = GuiRegimeKind::Idp13C;
            self.systems.clear();
        } else {
            self.regime_kind = GuiRegimeKind::Globular;
            if self.active_2d_nuclei() == ("1H", "15N") && !self.peaks.is_empty() {
                self.rebuild_systems_and_matrices();
            } else {
                self.systems.clear();
            }
        }
        let has_3d = self.hncacb_3d.is_some() || self.hnco_3d.is_some();
        self.show_strips_dock = pg.has_3d_strips && has_3d;
        if !pg.has_3d_strips {
            self.show_mol3d_viewport = false;
            self.responsive_shell.tablet_dock = TabletActiveDock::Hidden;
        }
        self.status_msg = format!("Viewing {}", pg.title);
    }

    fn activate_1d_workbook_page(&mut self, idx: usize) {
        if idx >= self.stack1d_panel.traces.len() {
            return;
        }
        if !self.show_stack1d_panel && self.workspace_mode == WorkspaceMode::ProteinNmr {
            self.save_live_2d_into_workbook();
        }
        self.stack1d_panel.activate_trace_page(idx);
        let lbl = self.stack1d_panel.traces[idx].label.clone();
        self.activate_1d_studio_mode();
        self.status_msg = format!("Viewing {lbl}");
    }

    fn load_builtin_demo_suite(&mut self) {
        let seed = BUILTIN_DEMO_SUITE_SEED.wrapping_add(self.sim_seed);
        match build_builtin_demo_suite(seed) {
            Ok(suite) => {
                let banner = suite.status_summary();
                let ref_data = suite.ubiquitin_ref;
                self.sequence = ref_data.sequence.clone();
                self.seq_ids = ref_data.seq_ids.clone();
                self.octet_app
                    .set_loaded_protein_sequence(self.sequence.clone());
                let saved_ph = self.sidechain_ribbon_state.sample_ph;
                self.sidechain_ribbon_state = SequenceRibbonState::new();
                self.sidechain_ribbon_state.set_sample_ph(saved_ph);
                self.sidechain_ribbon_state
                    .sync_inferred_variants_from_reference(&ref_data);
                self.reference = Some(ref_data);
                self.c13_picked_dataset = None;
                self.peaks = suite.peaks;
                self.truth = Some(suite.truth);
                self.assign_res = None;
                self.pred_ref = None;
                self.pred_conf.clear();
                self.pred_ambiguity_codes.clear();
                self.metrics = None;
                self.qc_report = None;
                self.raw_fids_total = 0;
                self.unprocessed_fids_remaining = 0;
                self.exported_in_session = false;

                let ubiq_hsqc = suite.hsqc_2d;
                self.hsqc_arc = Arc::new(ubiq_hsqc.data.clone());
                self.hsqc_gen = self.hsqc_gen.wrapping_add(1);
                self.hsqc_spec = Some(ubiq_hsqc.clone());
                self.hncacb_3d = Some(suite.hncacb_3d);
                self.hnco_3d = Some(suite.hnco_3d);
                self.extra_3d_count = 0;
                self.overlay_layers.clear();
                self.primary_2d_label = "20 · Ubiquitin 2D ¹H–¹⁵N HSQC".to_string();
                self.primary_2d_visible = true;
                self.primary_ppm_nudge = [0.0, 0.0];
                self.reset_view_to_spec();

                // Register clean multi-page 2D workbook entries:
                // 0: Small-molecule 2D 1H-1H COSY (paired with Page 1 1D 1H)
                // 1: Ubiquitin 2D 1H-15N HSQC (+ 3D HNCACB/HNCO strips)
                self.workbook.clear_2d_pages();
                let cosy_2d = build_showcase_small_molecule_cosy_2d();
                let _ = self
                    .workbook
                    .add_demo_2d_page("12 · 2D ¹H–¹H COSY", cosy_2d, false, false);
                let hsqc_page_idx = self.workbook.add_demo_2d_page(
                    "20 · Ubiquitin 2D ¹H–¹⁵N HSQC",
                    ubiq_hsqc,
                    false,
                    true,
                );
                self.workbook.active_2d_idx = hsqc_page_idx;
                self.save_live_2d_into_workbook();

                self.mol3d_panel = suite.mol3d_panel;
                self.loaded_pdb_text = Some(suite.pdb_text.to_string());

                // Prepend clean small-molecule 1D 1H (Page 1) & 1D 13C (Page 2) for calm default launch
                let (tr_1h, rois_1h) = build_showcase_small_molecule_1h_trace();
                let mut tr_13c = build_showcase_small_molecule_13c_trace();
                tr_13c.visible = false;
                let mut stack = Stack1dState::with_traces(vec![tr_1h, tr_13c]);
                stack.ref_trace_idx = 0;
                stack.view_mode = Stack1dViewMode::SuperimposedOverlay;
                stack.roi_regions = rois_1h;
                stack.target_total_protons = 12.0;
                stack.fit_full_ppm_bounds();
                stack.refresh_active_trace_peaks_and_multiplets();
                if !stack.detected_multiplet_integrals.is_empty() {
                    stack.normalize_multiplet_integral_to(0, 1.0);
                }
                self.stack1d_panel = stack;

                self.dosy_panel = suite.dosy_panel;
                self.metabolomics_panel = suite.metabolomics_panel;
                let _ = self.relax_panel.load_demo_dynamics(
                    &self.sequence,
                    &self.seq_ids,
                    Some(suite.pdb_text),
                );
                let _ = self
                    .dynamics_panel
                    .ensure_demo_exchange_and_csp(&self.sequence, &self.seq_ids);
                self.has_structure_or_dynamics = true;

                self.sync_session_from_app_fresh("Ubiquitin");
                self.rebuild_systems_and_matrices();
                self.refresh_propensity_report();
                self.run_assignment();

                // Calm Default Startup: open cleanly onto Page 1 (1D 1H Small Molecule) with side docks tucked away
                self.workspace_mode = WorkspaceMode::ProteinNmr;
                self.show_stack1d_panel = true;
                self.show_mol3d_viewport = false;
                self.show_strips_dock = false;
                self.show_relax_panel = false;
                self.show_dynamics_panel = false;
                self.status_msg = banner;
            }
            Err(err) => {
                self.status_msg = format!("Built-in demo suite error: {err}");
            }
        }
    }

    fn load_from_reference(&mut self, ref_data: Reference) {
        let (peaks, truth) = simulate(
            &ref_data,
            &SimConfig {
                p_missing_peak: self.sim_missing,
                spurious_rate: self.sim_spurious,
                seed: self.sim_seed,
                ..SimConfig::default()
            },
        );
        self.sequence = ref_data.sequence.clone();
        self.seq_ids = ref_data.seq_ids.clone();
        self.octet_app
            .set_loaded_protein_sequence(self.sequence.clone());
        let saved_ph = self.sidechain_ribbon_state.sample_ph;
        self.sidechain_ribbon_state = SequenceRibbonState::new();
        self.sidechain_ribbon_state.set_sample_ph(saved_ph);
        self.sidechain_ribbon_state
            .sync_inferred_variants_from_reference(&ref_data);
        self.reference = Some(ref_data);
        self.c13_picked_dataset = None;
        self.peaks = peaks;
        self.truth = Some(truth);
        self.assign_res = None;
        self.pred_ref = None;
        self.pred_conf.clear();
        self.pred_ambiguity_codes.clear();
        self.metrics = None;
        self.qc_report = None;
        self.raw_fids_total = 0;
        self.unprocessed_fids_remaining = 0;
        self.has_structure_or_dynamics = false;
        self.exported_in_session = false;
        self.regenerate_synthetic_spectra();
        self.sync_session_from_app_fresh("Ubiquitin");
        self.rebuild_systems_and_matrices();
        self.refresh_propensity_report();
        self.status_msg = format!(
            "Simulated {} peaks & {} spin systems ({} aa) + 2D HSQC / 3D HNCACB & HNCO spectra",
            self.peaks.len(),
            self.systems.len(),
            self.sequence.len()
        );
    }

    fn run_idp_13c_assignment(&mut self) {
        if self.sequence.is_empty() {
            self.status_msg =
                "Load an IDP sequence or NMR-STAR entry before 13C-IDP assignment".to_string();
            return;
        }
        if self.c13_picked_dataset.is_none() {
            if let Some(ref r) = self.reference {
                let ph_val = self
                    .idp_ph_input
                    .trim()
                    .parse::<f64>()
                    .ok()
                    .filter(|p| (1.0..=12.0).contains(p))
                    .unwrap_or(6.5);
                let cubes = simulate_idp_13c_cubes(
                    r,
                    &IdpCubeSimConfig {
                        temperature_k: self.idp_temp_k.clamp(250.0, 360.0),
                        ph: ph_val,
                        con_grid: (512, 512),
                        caco_grid: (256, 256),
                        cube_3d_grid: (128, 128, 128),
                        seed: self.sim_seed.wrapping_add(42),
                        ..IdpCubeSimConfig::default()
                    },
                );
                let pcfg = C13PickerConfig::default();
                let con_roots = pick_con_2d(&cubes.con_2d, &pcfg);
                let caco_peaks = cubes
                    .caco_2d
                    .as_ref()
                    .map(|sp| pick_caco_2d(sp, &pcfg))
                    .unwrap_or_default();
                let mut strip_peaks = Vec::new();
                for (exp, ref sp3) in &cubes.cubes_3d {
                    strip_peaks.extend(pick_con_anchored_3d(sp3, *exp, &con_roots, &pcfg));
                }
                self.hsqc_arc = Arc::new(cubes.con_2d.data.clone());
                self.hsqc_gen = self.hsqc_gen.wrapping_add(1);
                self.hsqc_spec = Some(cubes.con_2d);
                self.reset_view_to_spec();
                self.c13_picked_dataset = Some(C13PickedDataset {
                    con_roots,
                    caco_peaks,
                    strip_peaks,
                });
            }
        }

        let Some(ref ds) = self.c13_picked_dataset else {
            self.status_msg =
                "Load a 13C-detected peak TSV (CON/CACO/CANCO/CBCACON) or NMR-STAR entry first"
                    .to_string();
            return;
        };
        let systems = register_and_build_con_dipeptide_systems(ds);
        if systems.is_empty() {
            self.status_msg =
                "No CON dipeptide spin systems could be built from 13C-detected peaks".to_string();
            return;
        }

        let timer = PerfTimer::start();
        let ph_val = self
            .idp_ph_input
            .trim()
            .parse::<f64>()
            .ok()
            .filter(|p| (1.0..=12.0).contains(p))
            .unwrap_or(6.5);
        #[allow(unused_mut)]
        let mut idp_cfg = IdpSolverConfig {
            temperature_k: self.idp_temp_k.clamp(250.0, 360.0),
            ph: ph_val,
            replicas: self.replicas.max(8),
            steps_per_system: self.steps_per_sys.max(400),
            seed: self.sim_seed,
            ..IdpSolverConfig::default()
        };
        #[cfg(target_arch = "wasm32")]
        {
            idp_cfg.replicas = idp_cfg.replicas.min(16);
            idp_cfg.generations = 2;
            idp_cfg.steps_per_system = idp_cfg.steps_per_system.min(1000);
        }

        #[allow(unused_mut)]
        let mut gpu_out: Option<(nmr_core::idp_solver::IdpAssignResult, &'static str)> = None;
        #[cfg(not(target_arch = "wasm32"))]
        if self.use_gpu {
            if let Some(ref gpu) = self.gpu_engine {
                match nmr_gpu::idp_gpu::assign_idp_13c_gpu(gpu, &systems, &self.sequence, &idp_cfg)
                {
                    Ok(res) => gpu_out = Some((res, "13C-IDP WebGPU")),
                    Err(e) => {
                        self.status_msg =
                            format!("13C-IDP WebGPU error ({e}), falling back to CPU");
                    }
                }
            }
        }

        let default_tag = if cfg!(target_arch = "wasm32") {
            "13C-IDP WASM"
        } else {
            "13C-IDP CPU"
        };
        let (idp_res, tag) = gpu_out.unwrap_or_else(|| {
            (
                assign_idp_13c_cpu(&systems, &self.sequence, &idp_cfg),
                default_tag,
            )
        });
        let dt_ms = timer.elapsed_ms();
        self.last_solve_ms = Some(dt_ms);

        let (pred, conf) = idp_to_reference(
            &idp_res,
            &systems,
            &self.sequence,
            &self.seq_ids,
            0.0,
            false,
        );
        let n_pro = self
            .sequence
            .iter()
            .enumerate()
            .filter(|&(k, r)| {
                r == "PRO"
                    && (pred.shifts.contains_key(&(k, "N".to_string()))
                        || pred.shifts.contains_key(&(k, "CA".to_string())))
            })
            .count();
        let n_pro_tot = self.sequence.iter().filter(|r| r.as_str() == "PRO").count();
        let n_hc = conf.values().filter(|&&c| c >= 0.8).count();

        self.sidechain_ribbon_state
            .sync_inferred_variants_from_reference(&pred);
        self.pred_ref = Some(pred);
        self.pred_conf = conf;
        self.assign_res = Some(idp_res.as_assign_result());
        self.metrics = None;
        self.refresh_propensity_report();
        self.show_propensity_panel = true;
        self.sync_session_assignments_from_solver();
        self.status_msg = format!(
            "[{tag}] Assigned {}/{} residues ({n_hc} >=0.8, {n_pro}/{n_pro_tot} Pro) from {} CON systems in {dt_ms:.0} ms",
            self.pred_conf.len(),
            self.sequence.len(),
            systems.len()
        );
    }

    fn run_paramagnetic_assignment(&mut self) {
        if self.sequence.is_empty()
            || (self.solver_kind == GuiSolverKind::EvoGraph && self.peaks.is_empty())
            || (matches!(
                self.solver_kind,
                GuiSolverKind::Standard | GuiSolverKind::BayesSa
            ) && self.systems.is_empty()
                && self.peaks.is_empty())
        {
            self.status_msg =
                "Need peaks/spin systems and sequence to run paramagnetic assignment".to_string();
            return;
        }
        let explicit_ligands = if self.metal_ligands_input.trim().is_empty() {
            Vec::new()
        } else {
            match parse_metal_ligands(
                self.metal_ligands_input.trim(),
                &self.sequence,
                &self.seq_ids,
            ) {
                Ok(idxs) => idxs,
                Err(e) => {
                    self.status_msg = format!("Invalid metal ligands: {e}");
                    return;
                }
            }
        };

        let timer = PerfTimer::start();
        let scfg = ScoreConfig {
            assign_cost: self.assign_cost,
            locked_assignments: self.session.state.locked_assignments_vec(),
            ..ScoreConfig::default()
        };
        let acfg = AnnealConfig {
            runs: self.replicas.max(4),
            steps_per_system: self.steps_per_sys.max(300),
            seed: self.sim_seed,
            ..AnnealConfig::default()
        };
        let mode = match self.solver_kind {
            GuiSolverKind::Standard | GuiSolverKind::BayesSa => ParamagneticSolverMode::Standard,
            GuiSolverKind::EvoGraph => ParamagneticSolverMode::EvoGraph,
        };
        let para_cfg = ParamagneticAssignConfig {
            mode,
            explicit_ligand_indices: explicit_ligands.clone(),
            auto_detect_motifs: explicit_ligands.is_empty(),
            score_cfg: scfg.clone(),
            anneal_cfg: acfg,
            seed: self.sim_seed,
            ..ParamagneticAssignConfig::default()
        };

        let (res, para_rep, eff_priors, tag) = match self.solver_kind {
            GuiSolverKind::Standard | GuiSolverKind::BayesSa => {
                if let Ok(sys) = build_spin_systems(&self.peaks, &Tolerances::default(), 3, false) {
                    if !sys.is_empty() {
                        self.systems = sys;
                    }
                }
                let (r, rep) = assign_paramagnetic_backbone(
                    &self.systems,
                    &self.sequence,
                    &self.priors,
                    &para_cfg,
                );
                let lbl = if cfg!(target_arch = "wasm32") {
                    "Paramagnetic WASM-SA"
                } else {
                    "Paramagnetic CPU-SA"
                };
                (r, rep, self.priors.clone(), lbl)
            }
            GuiSolverKind::EvoGraph => {
                let (sys, r, rep) = assign_paramagnetic_from_peaks(
                    &self.peaks,
                    &self.sequence,
                    &self.priors,
                    &para_cfg,
                );
                let eff = rep
                    .evo_report
                    .as_ref()
                    .map(|er| er.shifted_priors(&self.priors))
                    .unwrap_or_else(|| self.priors.clone());
                self.systems = sys;
                let lbl = if cfg!(target_arch = "wasm32") {
                    "Paramagnetic WASM-Evo"
                } else {
                    "Paramagnetic CPU-Evo"
                };
                (r, rep, eff, lbl)
            }
        };

        self.u_mat = Some(unary_matrix(
            &self.systems,
            &self.sequence,
            &eff_priors,
            &scfg,
        ));
        self.p_mat = Some(pair_matrix(&self.systems, &scfg));
        let dt_ms = timer.elapsed_ms();
        self.last_solve_ms = Some(dt_ms);

        let (pred, conf) = to_reference(
            &res,
            &self.systems,
            &self.sequence,
            &self.seq_ids,
            0.0,
            Some(&eff_priors),
        );
        if let Some(truth) = &self.truth {
            if truth.len() == res.position.len() {
                self.metrics = Some(score(&res, truth, 0.8));
            } else {
                self.metrics = None;
            }
        }
        let n_hc = conf.values().filter(|&&c| c >= 0.8).count();
        self.sidechain_ribbon_state
            .sync_inferred_variants_from_reference(&pred);
        self.pred_ref = Some(pred);
        self.pred_conf = conf;
        self.assign_res = Some(res);
        self.refresh_propensity_report();
        self.sync_session_assignments_from_solver();
        self.status_msg = format!(
            "[{tag}] Assigned {}/{} residues ({n_hc} >=0.8 | {} metal cluster(s), {} blind-zone aa, {} abstained) in {dt_ms:.0} ms",
            self.pred_conf.len(),
            self.sequence.len(),
            para_rep.detected_clusters.len(),
            para_rep.blind_zone_residues.len(),
            para_rep.n_blind_zone_abstentions,
        );
    }

    fn run_assignment(&mut self) {
        match self.regime_kind {
            GuiRegimeKind::Idp13C => {
                self.run_idp_13c_assignment();
                return;
            }
            GuiRegimeKind::Paramagnetic => {
                self.run_paramagnetic_assignment();
                return;
            }
            GuiRegimeKind::Globular => {}
        }
        if self.sequence.is_empty()
            || (self.solver_kind == GuiSolverKind::EvoGraph && self.peaks.is_empty())
            || (self.solver_kind == GuiSolverKind::Standard && self.systems.is_empty())
            || (self.solver_kind == GuiSolverKind::BayesSa
                && self.peaks.is_empty()
                && self.systems.is_empty())
        {
            self.status_msg = "Need peaks/spin systems and sequence to run assignment".to_string();
            return;
        }
        self.last_bayes_sa_telemetry = None;
        // Snapshot coordinates of user-locked spin systems so locks remap across filtered/rebuilt SpinSystems
        let locked_coords: Vec<(f64, f64, usize)> = self
            .session
            .state
            .locked_residues
            .iter()
            .filter_map(|(&ss_idx, &res_idx)| {
                self.systems.get(ss_idx).map(|ss| (ss.h, ss.n, res_idx))
            })
            .collect();

        let timer = PerfTimer::start();
        let (res, engine_tag, eff_priors, kept_hsqc_opt) = match self.solver_kind {
            GuiSolverKind::EvoGraph => {
                let raw_locks = self.raw_hsqc_locks_for_peaks();
                let scfg_base = ScoreConfig {
                    assign_cost: self.assign_cost,
                    ..ScoreConfig::default()
                };
                let mut evo_cfg = EvoGraphConfig {
                    score_cfg: scfg_base.clone(),
                    auto_reference: scfg_base.auto_reference,
                    structure_torsions: None,
                    locked_assignments: raw_locks,
                    seed: self.sim_seed,
                    ..EvoGraphConfig::default()
                };
                evo_cfg.population_size = self.replicas.max(4);
                evo_cfg.num_islands = evo_cfg.num_islands.min(evo_cfg.population_size).max(1);
                evo_cfg.steps_per_system_per_gen = (self.steps_per_sys * 3 / 10).max(100);
                #[cfg(target_arch = "wasm32")]
                {
                    evo_cfg.generations = 4;
                    evo_cfg.steps_per_system_per_gen = evo_cfg.steps_per_system_per_gen.min(250);
                }
                let (evo_systems, r, rep) = assign_evo_graph_from_peaks(
                    &self.peaks,
                    &self.sequence,
                    &self.priors,
                    &evo_cfg,
                );
                let mut remapped_locks: BTreeMap<usize, usize> = BTreeMap::new();
                for &(lh, ln, lres) in &locked_coords {
                    if let Some((new_ss, _)) = evo_systems
                        .iter()
                        .enumerate()
                        .find(|&(_, s)| (s.h - lh).abs() <= 0.045 && (s.n - ln).abs() <= 0.45)
                    {
                        remapped_locks.insert(new_ss, lres);
                    }
                }
                self.session.state.locked_residues = remapped_locks;

                let scfg_u = ScoreConfig {
                    assign_cost: self.assign_cost,
                    locked_assignments: self.session.state.locked_assignments_vec(),
                    ..ScoreConfig::default()
                };
                let shifted_pri = rep.shifted_priors(&self.priors);
                self.u_mat = Some(unary_matrix(
                    &evo_systems,
                    &self.sequence,
                    &shifted_pri,
                    &scfg_u,
                ));
                self.p_mat = Some(pair_matrix(&evo_systems, &scfg_u));
                self.qc_report = Some(rep.qc_report.clone());
                self.systems = evo_systems;
                self.selected_ss = match self.selected_ss {
                    Some(s) if s < self.systems.len() => Some(s),
                    _ => (!self.systems.is_empty()).then_some(0),
                };
                let tag = if cfg!(target_arch = "wasm32") {
                    "WASM Evo-Graph"
                } else {
                    "CPU Evo-Graph"
                };
                (r, tag, shifted_pri, Some(rep.kept_hsqc_indices))
            }
            GuiSolverKind::Standard => {
                // Rebuild unfiltered spin systems when switching back from Evo-Graph filtered systems
                if let Ok(sys) = build_spin_systems(&self.peaks, &Tolerances::default(), 3, false) {
                    if sys.len() != self.systems.len() {
                        let mut remapped_locks: BTreeMap<usize, usize> = BTreeMap::new();
                        for &(lh, ln, lres) in &locked_coords {
                            if let Some((new_ss, _)) = sys.iter().enumerate().find(|&(_, s)| {
                                (s.h - lh).abs() <= 0.045 && (s.n - ln).abs() <= 0.45
                            }) {
                                remapped_locks.insert(new_ss, lres);
                            }
                        }
                        self.session.state.locked_residues = remapped_locks;
                        self.systems = sys;
                        self.selected_ss = (!self.systems.is_empty()).then_some(0);
                    }
                }
                let scfg = ScoreConfig {
                    assign_cost: self.assign_cost,
                    locked_assignments: self.session.state.locked_assignments_vec(),
                    ..ScoreConfig::default()
                };
                self.p_mat = Some(pair_matrix(&self.systems, &scfg));
                let acfg = AnnealConfig {
                    runs: self.replicas,
                    steps_per_system: self.steps_per_sys,
                    seed: self.sim_seed,
                    ..AnnealConfig::default()
                };
                #[allow(unused_mut)]
                let mut res_opt: Option<(AssignResult, &'static str)> = None;

                #[cfg(not(target_arch = "wasm32"))]
                if self.use_gpu {
                    if let Some(gpu) = &self.gpu_engine {
                        match gpu.assign_gpu(
                            &self.systems,
                            &self.sequence,
                            &self.priors,
                            &scfg,
                            &acfg,
                        ) {
                            Ok(r) => {
                                self.u_mat = Some(unary_matrix(
                                    &self.systems,
                                    &self.sequence,
                                    &self.priors,
                                    &scfg,
                                ));
                                res_opt = Some((r, "WebGPU"));
                            }
                            Err(e) => {
                                self.status_msg =
                                    format!("GPU solver error ({e}), falling back to CPU");
                            }
                        }
                    }
                }

                let default_tag = if cfg!(target_arch = "wasm32") {
                    "WASM"
                } else {
                    "CPU-MT"
                };
                let (r, tag) = res_opt.unwrap_or_else(|| {
                    let u = unary_matrix(&self.systems, &self.sequence, &self.priors, &scfg);
                    let p = self
                        .p_mat
                        .clone()
                        .unwrap_or_else(|| pair_matrix(&self.systems, &scfg));
                    self.u_mat = Some(u.clone());
                    (assign_from_matrices(&u, &p, &acfg), default_tag)
                });
                let qc = evaluate_dataset_qc(
                    Some(&self.peaks),
                    &self.systems,
                    &self.sequence,
                    &self.priors,
                    &scfg,
                    &r,
                    None,
                );
                self.qc_report = Some(qc);
                (r, tag, self.priors.clone(), None)
            }
            GuiSolverKind::BayesSa => {
                let raw_locks: Vec<LockedAssignment> = if !self.peaks.is_empty() {
                    self.raw_hsqc_locks_for_peaks()
                        .into_iter()
                        .map(LockedAssignment::from)
                        .collect()
                } else {
                    self.session
                        .state
                        .locked_assignments_vec()
                        .into_iter()
                        .map(LockedAssignment::from)
                        .collect()
                };
                let mu_override = (self.assign_cost != 0.0).then_some(self.assign_cost);

                #[allow(unused_mut)]
                let mut bayes_out: Option<(
                    Vec<SpinSystem>,
                    AssignResult,
                    BayesSaReport,
                    &'static str,
                )> = None;

                #[cfg(not(target_arch = "wasm32"))]
                if self.use_gpu {
                    if let Some(ref bgpu) = self.bayes_gpu_solver {
                        let mut gcfg = BayesSaConfig {
                            replicas: self.replicas.max(8),
                            seed: self.sim_seed,
                            locked_assignments: raw_locks.clone(),
                            mu_null_override: mu_override,
                            ..BayesSaConfig::default_gpu()
                        };
                        if gcfg.replicas == 64 {
                            // Preserve BayesSaConfig::default_gpu() R=128 unless user adjusted slider away from GUI startup default
                            gcfg.replicas = BayesSaConfig::default_gpu().replicas;
                        }
                        let gpu_res = if !self.peaks.is_empty() {
                            bgpu.solve_from_peaks(&self.peaks, &self.sequence, &self.priors, &gcfg)
                        } else {
                            bgpu.solve(&self.systems, &self.sequence, &self.priors, &gcfg)
                                .map(|(r, rep)| (self.systems.clone(), r, rep))
                        };
                        match gpu_res {
                            Ok((sys, r, rep)) => {
                                bayes_out = Some((sys, r, rep, "WebGPU Bayes-SA"));
                            }
                            Err(e) => {
                                self.status_msg = format!(
                                    "WebGPU Bayes-SA error ({e}), falling back to CPU Bayes-SA"
                                );
                            }
                        }
                    }
                }

                let cpu_tag = if cfg!(target_arch = "wasm32") {
                    "WASM Bayes-SA"
                } else {
                    "CPU Bayes-SA"
                };
                let (bayes_systems, r, rep, tag) = match bayes_out {
                    Some(v) => v,
                    None => {
                        #[allow(unused_mut)]
                        let mut ccfg = BayesSaConfig {
                            replicas: if self.replicas == 64 {
                                BayesSaConfig::default().replicas
                            } else {
                                self.replicas.max(4)
                            },
                            seed: self.sim_seed,
                            locked_assignments: raw_locks,
                            mu_null_override: mu_override,
                            ..BayesSaConfig::default()
                        };
                        #[cfg(target_arch = "wasm32")]
                        {
                            ccfg.replicas = ccfg.replicas.min(16);
                            ccfg.max_stages = ccfg.max_stages.min(24);
                        }
                        if !self.peaks.is_empty() {
                            match solve_bayes_sa_from_peaks(
                                &self.peaks,
                                &self.sequence,
                                &self.priors,
                                &ccfg,
                            ) {
                                Ok((sys, r, rep)) => (sys, r, rep, cpu_tag),
                                Err(e) => {
                                    self.status_msg = format!("Bayes-SA error: {e}");
                                    return;
                                }
                            }
                        } else {
                            let (r, rep) =
                                solve_bayes_sa(&self.systems, &self.sequence, &self.priors, &ccfg);
                            (self.systems.clone(), r, rep, cpu_tag)
                        }
                    }
                };

                let mut remapped_locks: BTreeMap<usize, usize> = BTreeMap::new();
                for &(lh, ln, lres) in &locked_coords {
                    if let Some((new_ss, _)) = bayes_systems
                        .iter()
                        .enumerate()
                        .find(|&(_, s)| (s.h - lh).abs() <= 0.045 && (s.n - ln).abs() <= 0.45)
                    {
                        remapped_locks.insert(new_ss, lres);
                    }
                }
                self.session.state.locked_residues = remapped_locks;

                let scfg_u = ScoreConfig {
                    assign_cost: self.assign_cost,
                    locked_assignments: self.session.state.locked_assignments_vec(),
                    ..ScoreConfig::default()
                };
                let shifted_pri = self.priors.with_offsets(&rep.applied_offsets);
                self.u_mat = Some(unary_matrix(
                    &bayes_systems,
                    &self.sequence,
                    &shifted_pri,
                    &scfg_u,
                ));
                self.p_mat = Some(pair_matrix(&bayes_systems, &scfg_u));
                self.qc_report = Some(rep.qc_report.clone());
                self.last_bayes_sa_telemetry = Some((rep.t_crit, rep.max_cv, rep.mu_null));
                self.systems = bayes_systems;
                self.selected_ss = match self.selected_ss {
                    Some(s) if s < self.systems.len() => Some(s),
                    _ => (!self.systems.is_empty()).then_some(0),
                };
                (r, tag, shifted_pri, None)
            }
        };

        let dt_ms = timer.elapsed_ms();
        self.last_solve_ms = Some(dt_ms);

        let (pred, conf) = to_reference(
            &res,
            &self.systems,
            &self.sequence,
            &self.seq_ids,
            0.0,
            Some(&eff_priors),
        );
        if let Some(truth) = &self.truth {
            if let Some(kept) = &kept_hsqc_opt {
                let eval_truth: Vec<i32> = kept
                    .iter()
                    .map(|&idx| truth.get(idx).copied().unwrap_or(-1))
                    .collect();
                if eval_truth.len() == res.position.len() {
                    self.metrics = Some(Metrics {
                        assignable: truth.iter().filter(|&&t| t >= 0).count(),
                        ..score(&res, &eval_truth, 0.8)
                    });
                } else {
                    self.metrics = None;
                }
            } else if truth.len() == res.position.len() {
                self.metrics = Some(score(&res, truth, 0.8));
            } else {
                self.metrics = None;
            }
        }
        let n_hc = conf.values().filter(|&&c| c >= 0.8).count();
        let n_locks = self.session.state.locked_residues.len();
        let lock_note = if n_locks > 0 {
            format!(", {n_locks} locked")
        } else {
            String::new()
        };
        let bayes_note = if let Some((t_crit, max_cv, mu_null)) = self.last_bayes_sa_telemetry {
            format!(", T_crit={t_crit:.2}, max_Cv={max_cv:.1}, mu_null={mu_null:.2}")
        } else {
            String::new()
        };
        self.status_msg = format!(
            "[{engine_tag}] Assigned {}/{} residues ({} high-conf >=0.8{lock_note}{bayes_note}) in {:.0} ms",
            conf.len(),
            self.sequence.len(),
            n_hc,
            dt_ms
        );
        if let Some(ss) = self.selected_ss {
            if let Some(&k) = res.position.get(ss) {
                if k >= 0 {
                    self.selected_res = Some(k as usize);
                }
            }
        }
        self.sidechain_ribbon_state
            .sync_inferred_variants_from_reference(&pred);
        self.pred_ref = Some(pred);
        self.pred_conf = conf;
        self.assign_res = Some(res);
        self.refresh_propensity_report();
        self.sync_session_assignments_from_solver();
    }

    fn load_text_in_memory(&mut self, text: &str, source_label: &str) {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            self.status_msg =
                "Paste NMR-STAR (.str), NEF (.nef), Peak TSV (.tsv), CDXML, MOL, or SMILES first"
                    .to_string();
            return;
        }
        let lower_label = source_label.to_ascii_lowercase();

        // Check 3D biomolecular structure formats (`.pdb`, `.ent`, `.cif`, `.mmcif`)
        if lower_label.ends_with(".pdb")
            || lower_label.ends_with(".ent")
            || lower_label.ends_with(".cif")
            || lower_label.ends_with(".mmcif")
            || trimmed.contains("_atom_site.")
            || trimmed
                .lines()
                .take(48)
                .any(|ln| ln.starts_with("ATOM  ") || ln.starts_with("HETATM"))
        {
            self.load_structure_3d_in_memory(trimmed, source_label);
            return;
        }

        // Check chemical structure formats first (`CDXML`, `OCT`, `MOL`/`SDF`, `InChI`, `.smi`)
        if trimmed.contains("<CDXML")
            || lower_label.ends_with(".cdxml")
            || lower_label.ends_with(".oct")
            || lower_label.ends_with(".mol")
            || lower_label.ends_with(".sdf")
            || lower_label.ends_with(".smi")
            || trimmed.contains("V2000")
            || trimmed.contains("V3000")
            || trimmed.starts_with("InChI=")
        {
            let fmt = if trimmed.contains("<CDXML") || lower_label.ends_with(".cdxml") {
                IoFormat::Cdxml
            } else if lower_label.ends_with(".oct") || trimmed.starts_with('{') {
                IoFormat::OctJson
            } else if lower_label.ends_with(".sdf") || trimmed.contains("$$$$") {
                IoFormat::Sdf
            } else if trimmed.starts_with("InChI=") {
                IoFormat::InChI
            } else if trimmed.contains("V2000")
                || trimmed.contains("V3000")
                || lower_label.ends_with(".mol")
            {
                IoFormat::MolV2000
            } else {
                IoFormat::Smiles
            };
            if let Some(res) = CoreOctetBackendAdapter.import_custom(fmt, trimmed) {
                match res {
                    Ok(sc) => {
                        self.octet_app.set_scene(sc);
                        self.workspace_mode = WorkspaceMode::OctetChem;
                        self.octet_app.status_msg =
                            format!("Loaded {} from {source_label}", fmt.label());
                        self.status_msg = format!("Loaded {} into Octet Studio", fmt.label());
                    }
                    Err(e) => self.status_msg = format!("Chemical file parse error: {e}"),
                }
                return;
            }
        }

        // Check Relaxation Series TSV / NEF (`save_nmrassign_relaxation_series` or `# nmrassign_relax_series`)
        if trimmed.contains("save_nmrassign_relaxation_series")
            || trimmed.contains("experiment_kind=")
            || trimmed.starts_with("# nmrassign_relax_series")
        {
            match self.relax_panel.import_series_text(trimmed) {
                Ok(msg) => {
                    self.workspace_mode = WorkspaceMode::ProteinNmr;
                    self.show_relax_panel = true;
                    self.show_dynamics_panel = false;
                    self.show_stack1d_panel = false;
                    self.has_structure_or_dynamics = true;
                    self.status_msg = msg;
                }
                Err(e) => {
                    self.status_msg =
                        format!("Relaxation series import error ({source_label}): {e}");
                }
            }
            return;
        }

        // Check CEST / CPMG / Ligand CSP trajectory TSV (`T13.4`–`T13.6`)
        if DynamicsStudioPanel::looks_like_exchange_or_csp_tsv(trimmed) {
            match self.dynamics_panel.import_exchange_or_csp_tsv(trimmed) {
                Ok(msg) => {
                    self.workspace_mode = WorkspaceMode::ProteinNmr;
                    self.show_dynamics_panel = true;
                    self.show_relax_panel = false;
                    self.show_stack1d_panel = false;
                    self.has_structure_or_dynamics = true;
                    self.status_msg = msg;
                }
                Err(e) => {
                    self.status_msg = format!("Exchange / CSP import error ({source_label}): {e}");
                }
            }
            return;
        }

        // Check 1D JCAMP-DX (`.jdx` / `.dx` / `##TITLE=`) or 1D `ppm\treal` trace TSV for Stack1dPanel
        if lower_label.ends_with(".jdx")
            || lower_label.ends_with(".dx")
            || trimmed.starts_with("##TITLE=")
        {
            match read_jcamp(trimmed) {
                Ok(blocks) if !blocks.is_empty() => {
                    let mut added = 0usize;
                    for (b_idx, blk) in blocks.into_iter().enumerate() {
                        if blk.y.len() >= 2 {
                            let idx = self.stack1d_panel.traces.len();
                            let lbl = if blk.title.trim().is_empty() {
                                format!("{source_label}#{}", b_idx + 1)
                            } else {
                                blk.title.clone()
                            };
                            let xs = blk.x_axis();
                            self.stack1d_panel.add_trace(Trace1d::new(
                                format!("jcamp_{idx}"),
                                lbl,
                                idx as f64,
                                xs,
                                blk.y,
                            ));
                            added += 1;
                        }
                    }
                    if added > 0 {
                        self.activate_1d_studio_mode();
                        self.status_msg = format!(
                            "Loaded {added} 1D JCAMP-DX trace(s) from {source_label} into 1D Studio"
                        );
                        return;
                    }
                }
                Ok(_) => {}
                Err(e) => {
                    self.status_msg = format!("JCAMP-DX 1D parse error ({source_label}): {e}");
                    return;
                }
            }
        }
        if let Some(first_non_comment) = trimmed
            .lines()
            .map(str::trim)
            .find(|l| !l.is_empty() && !l.starts_with('#'))
        {
            let hdr_lower = first_non_comment.to_ascii_lowercase();
            let cols: Vec<&str> = hdr_lower.split(['\t', ',']).map(str::trim).collect();
            if cols.len() >= 2 && cols[0] == "ppm" && matches!(cols[1], "real" | "intensity" | "y")
            {
                let mut ppms = Vec::new();
                let mut ys = Vec::new();
                for line in trimmed.lines().map(str::trim) {
                    if line.is_empty() || line.starts_with('#') {
                        continue;
                    }
                    let f: Vec<&str> = line.split(['\t', ',']).map(str::trim).collect();
                    if f.len() >= 2 {
                        if let (Ok(x), Ok(y)) = (f[0].parse::<f64>(), f[1].parse::<f64>()) {
                            ppms.push(x);
                            ys.push(y);
                        }
                    }
                }
                if ppms.len() >= 2 {
                    let first = ppms[0];
                    let last = *ppms.last().unwrap_or(&first);
                    let axis = nmr_core::spectrum::AxisPpm::from_bounds(
                        "1H",
                        ppms.len(),
                        first,
                        last,
                        500.0,
                    );
                    let spec = nmr_core::nmr1d::Spectrum1D::new(
                        source_label.to_string(),
                        nmr_core::nmr1d::Experiment1DKind::Proton1D,
                        axis,
                        ys,
                        None,
                    );
                    self.prepare_workbook_for_user_append();
                    self.stack1d_panel.load_user_spectra(vec![spec]);
                    self.activate_1d_studio_mode();
                    self.status_msg = format!(
                        "Loaded 1D spectrum '{source_label}' into 1D Studio ({} traces total)",
                        self.stack1d_panel.traces.len()
                    );
                    return;
                }
            }
        }

        // Check NEF v1.1 (`data_nef_...`, `save_nef_molecular_system`, or `_nmrassign_workbench`)
        if lower_label.ends_with(".nef")
            || trimmed.contains("save_nef_molecular_system")
            || trimmed.starts_with("data_nef_")
        {
            if trimmed.contains("_nmrassign_workbench") {
                match self
                    .session
                    .load_from_nef_str(trimmed, Some(source_label.to_string()))
                {
                    Ok(summary) => {
                        self.truth = None;
                        self.metrics = None;
                        self.peaks = self.session.state.peaks.clone();
                        self.sequence = self.session.state.sequence.clone();
                        self.seq_ids = self.session.state.seq_ids.clone();
                        if !self.peaks.is_empty() {
                            self.regenerate_synthetic_spectra();
                        }
                        self.rebuild_systems_and_matrices();
                        self.sync_app_from_session();
                        self.workspace_mode = WorkspaceMode::ProteinNmr;
                        self.status_msg = summary;
                        return;
                    }
                    Err(e) => {
                        self.status_msg = format!("NEF session restore error: {e}");
                        return;
                    }
                }
            }

            match read_nef_str(trimmed) {
                Ok(doc) => match doc.to_reference() {
                    Ok(r) => {
                        let nef_pks = doc.to_peaks();
                        self.sequence = r.sequence.clone();
                        self.seq_ids = r.seq_ids.clone();
                        self.octet_app
                            .set_loaded_protein_sequence(self.sequence.clone());
                        self.reference = Some(r.clone());
                        if !nef_pks.is_empty() {
                            self.peaks = nef_pks;
                            self.truth = None;
                            self.metrics = None;
                            self.regenerate_synthetic_spectra();
                            self.sync_session_from_app_fresh(&doc.project_name);
                            self.rebuild_systems_and_matrices();
                            self.run_assignment();
                        } else {
                            self.load_from_reference(r);
                            self.run_assignment();
                        }
                        self.workspace_mode = WorkspaceMode::ProteinNmr;
                    }
                    Err(e) => self.status_msg = format!("NEF molecular system error: {e}"),
                },
                Err(e) => self.status_msg = format!("NEF parse error ({source_label}): {e}"),
            }
            return;
        }

        let looks_like_star = trimmed.contains("loop_")
            || trimmed.contains("_Atom_chem_shift")
            || trimmed.contains("_Chem_shift_value")
            || trimmed.starts_with("data_");

        if looks_like_star {
            match read_star_str_all_atoms(trimmed, None) {
                Ok(r) => {
                    self.load_from_reference(r);
                    self.run_assignment();
                    self.workspace_mode = WorkspaceMode::ProteinNmr;
                }
                Err(e) => self.status_msg = format!("NMR-STAR parse error ({source_label}): {e}"),
            }
        } else {
            match read_peaks_str(trimmed) {
                Ok(pks) => {
                    self.peaks = pks;
                    self.truth = None;
                    self.metrics = None;
                    self.regenerate_synthetic_spectra();
                    self.sync_session_from_app_fresh("in_memory_peaks");
                    self.rebuild_systems_and_matrices();
                    self.workspace_mode = WorkspaceMode::ProteinNmr;
                    self.status_msg = format!(
                        "Loaded {} peaks in memory from {source_label}",
                        self.peaks.len()
                    );
                }
                Err(e) => {
                    if let Some(ds) = parse_gui_c13_tsv_str(trimmed) {
                        let n_con = ds.con_roots.len();
                        let n_strip = ds.strip_peaks.len();
                        self.c13_picked_dataset = Some(ds);
                        self.regime_kind = GuiRegimeKind::Idp13C;
                        self.workspace_mode = WorkspaceMode::ProteinNmr;
                        if !self.sequence.is_empty() {
                            self.run_assignment();
                        } else {
                            self.status_msg = format!(
                                "Loaded 13C-IDP TSV ({n_con} CON roots, {n_strip} 3D peaks) — load sequence/.str to assign"
                            );
                        }
                    } else {
                        self.status_msg = format!("Peak TSV parse error ({source_label}): {e}");
                    }
                }
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn install_single_2d_spectrum(&mut self, spec: Spectrum2D, label: String, status: String) {
        self.bare_bruker_hint = None;
        self.prepare_workbook_for_user_append();
        self.water_undo_2d = None;
        self.peak_2d_labels.clear();
        self.workbook.two_d.selected_peak_idx = None;
        self.workbook.two_d.dragging_peak_idx = None;
        self.workbook.two_d.zoom_history.clear();
        self.clear_stale_protein_demo_state(false, false);
        self.primary_2d_label = label.clone();
        self.primary_2d_visible = true;
        self.primary_ppm_nudge = [0.0, 0.0];
        self.hsqc_arc = Arc::new(spec.data.clone());
        self.hsqc_gen = self.hsqc_gen.wrapping_add(1);
        let new_idx = self
            .workbook
            .add_2d_page(label.clone(), spec.clone(), false, false);
        self.workbook.active_2d_idx = new_idx;
        self.hsqc_spec = Some(spec);
        self.sync_overlay_layers_from_workbook();
        self.reset_view_to_spec();
        self.workspace_mode = WorkspaceMode::ProteinNmr;
        self.show_stack1d_panel = false;
        self.show_relax_panel = false;
        self.show_dynamics_panel = false;
        if self.active_2d_nuclei() == ("13C", "15N") {
            self.regime_kind = GuiRegimeKind::Idp13C;
            if let Some(ref sp) = self.hsqc_spec {
                let pcfg = C13PickerConfig {
                    snr_2d: self.picker_snr,
                    ..C13PickerConfig::default()
                };
                let con_roots = pick_con_2d(sp, &pcfg);
                self.c13_picked_dataset = Some(C13PickedDataset {
                    con_roots,
                    ..C13PickedDataset::default()
                });
                self.sync_con_roots_to_peaks();
            }
        } else {
            self.regime_kind = GuiRegimeKind::Globular;
        }
        self.save_live_2d_into_workbook();
        self.sync_session_from_app_fresh(&label);
        self.status_msg = status;
    }

    fn pick_generic_2d_crosspeaks(&self, spec: &Spectrum2D, snr_threshold: f32) -> Vec<Peak> {
        let (nx, ny) = self.active_2d_nuclei();
        pick_2d_crosspeaks_ridge_safe(
            spec,
            snr_threshold,
            self.contour_base,
            nx == ny,
            self.workbook.two_d.pick_diagonal,
            self.show_negative,
        )
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn load_path_auto(&mut self) {
        let path_str = sanitize_user_path_input(&self.path_input);
        self.path_input = path_str.clone();
        let p = Path::new(&path_str);
        if !p.exists() {
            self.status_msg = format!("Path not found: {}", p.display());
            return;
        }
        let lower_full = path_str.to_ascii_lowercase();
        let fname_low = p
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if p.is_dir()
            || lower_full.ends_with(".zip")
            || lower_full.ends_with(".tar")
            || lower_full.ends_with(".tar.gz")
            || lower_full.ends_with(".tgz")
            || lower_full.ends_with(".jdx")
            || lower_full.ends_with(".dx")
            || lower_full.ends_with(".jcamp")
            || lower_full.ends_with(".ft1")
            || matches!(
                fname_low.as_str(),
                "1r" | "1i" | "procs" | "fid" | "acqus" | "acqu" | "ser"
            )
        {
            if p.is_dir() && p.join("2rr").exists() {
                match read_bruker_pdata_2d(p) {
                    Ok(spec) => {
                        let lbl = spec.name.clone();
                        self.install_single_2d_spectrum(
                            spec,
                            lbl,
                            format!("Loaded Bruker 2D pdata from {}", p.display()),
                        );
                    }
                    Err(e) => self.status_msg = format!("Bruker load error: {e}"),
                }
                return;
            }
            match load_native_path_1d_or_workspace(p) {
                Ok(outcome) => self.apply_drop_outcome(outcome),
                Err(e) => self.status_msg = format!("Load error: {e}"),
            }
            return;
        }

        if p.file_name().and_then(|s| s.to_str()) == Some("2rr") {
            let dir = p.parent().unwrap_or(p);
            match read_bruker_pdata_2d(dir) {
                Ok(spec) => {
                    let lbl = spec.name.clone();
                    self.install_single_2d_spectrum(
                        spec,
                        lbl,
                        format!("Loaded Bruker 2D pdata from {}", dir.display()),
                    );
                }
                Err(e) => self.status_msg = format!("Bruker load error: {e}"),
            }
            return;
        }

        let ext = p
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        match ext.as_str() {
            "pdb" | "ent" | "cif" | "mmcif" => match std::fs::read_to_string(p) {
                Ok(txt) => self.load_structure_3d_in_memory(&txt, &p.display().to_string()),
                Err(e) => self.status_msg = format!("3D structure read error: {e}"),
            },
            "str" | "star" => match read_star_all_atoms(p, None) {
                Ok(r) => {
                    self.load_from_reference(r);
                    self.workspace_mode = WorkspaceMode::ProteinNmr;
                }
                Err(e) => self.status_msg = format!("NMR-STAR error: {e}"),
            },
            "nef" => match std::fs::read_to_string(p) {
                Ok(txt) => self.load_text_in_memory(&txt, &p.display().to_string()),
                Err(e) => self.status_msg = format!("NEF read error: {e}"),
            },
            "fasta" | "fa" | "faa" | "seq" => match std::fs::read_to_string(p) {
                Ok(txt) => {
                    let parsed = nmr_gui::ingest_panel::parse_sequence_three_letter(&txt);
                    if parsed.is_empty() {
                        self.status_msg =
                            format!("No amino-acid sequence found in {}", p.display());
                    } else {
                        self.sequence = parsed;
                        self.seq_ids = (1..=(self.sequence.len() as i32)).collect();
                        self.octet_app
                            .set_loaded_protein_sequence(self.sequence.clone());
                        self.session.state.sequence = self.sequence.clone();
                        self.session.state.seq_ids = self.seq_ids.clone();
                        self.rebuild_systems_and_matrices();
                        self.workspace_mode = WorkspaceMode::ProteinNmr;
                        self.status_msg = format!(
                            "Loaded {} residue sequence from {}",
                            self.sequence.len(),
                            p.display()
                        );
                    }
                }
                Err(e) => self.status_msg = format!("FASTA read error: {e}"),
            },
            "tsv" => match read_peaks(p) {
                Ok(pks) => {
                    self.peaks = pks;
                    self.truth = None;
                    self.metrics = None;
                    self.regenerate_synthetic_spectra();
                    self.sync_session_from_app_fresh("tsv_peaks");
                    self.rebuild_systems_and_matrices();
                    self.workspace_mode = WorkspaceMode::ProteinNmr;
                    self.status_msg =
                        format!("Loaded {} peaks from {}", self.peaks.len(), p.display());
                }
                Err(e) => {
                    if let Ok(raw_txt) = std::fs::read_to_string(p) {
                        if let Some(ds) = parse_gui_c13_tsv_str(&raw_txt) {
                            let n_con = ds.con_roots.len();
                            let n_strip = ds.strip_peaks.len();
                            self.c13_picked_dataset = Some(ds);
                            self.regime_kind = GuiRegimeKind::Idp13C;
                            self.workspace_mode = WorkspaceMode::ProteinNmr;
                            self.status_msg = format!(
                                "Loaded 13C-IDP TSV from {} ({n_con} CON roots, {n_strip} 3D peaks)",
                                p.display()
                            );
                            return;
                        }
                    }
                    self.status_msg = format!("Peak TSV error: {e}");
                }
            },
            "ucsf" | "usf" | "sp2" => match read_ucsf_2d(p) {
                Ok(spec) => {
                    let lbl = p
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .map(str::to_string)
                        .unwrap_or_else(|| spec.name.clone());
                    self.install_single_2d_spectrum(
                        spec,
                        lbl,
                        format!("Loaded Sparky/UCSF 2D spectrum {}", p.display()),
                    );
                }
                Err(_) => match read_ucsf_3d(p, "HNCACB") {
                    Ok(spec3d) => {
                        self.hncacb_3d = Some(spec3d);
                        self.workspace_mode = WorkspaceMode::ProteinNmr;
                        self.status_msg = format!("Loaded UCSF 3D spectrum {}", p.display());
                    }
                    Err(e) => {
                        if let Ok(bytes) = std::fs::read(p) {
                            let stem = p
                                .file_stem()
                                .and_then(|s| s.to_str())
                                .unwrap_or("spectrum_1d");
                            if let Some(sp1d) = try_load_1d_ucsf_or_pipe_bytes(stem, &bytes) {
                                self.stack1d_panel.load_user_spectra(vec![sp1d]);
                                self.activate_1d_studio_mode();
                                self.status_msg = format!(
                                    "Loaded 1D UCSF spectrum {} into 1D Studio",
                                    p.display()
                                );
                                return;
                            }
                        }
                        self.status_msg = format!("UCSF load error: {e}");
                    }
                },
            },
            "ft2" | "dat" | "pipe" | "ft" => match read_nmrpipe_2d(p) {
                Ok(spec) => {
                    let lbl = p
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .map(str::to_string)
                        .unwrap_or_else(|| spec.name.clone());
                    self.install_single_2d_spectrum(
                        spec,
                        lbl,
                        format!("Loaded NMRPipe 2D spectrum {}", p.display()),
                    );
                }
                Err(e) => self.status_msg = format!("NMRPipe load error: {e}"),
            },
            "ft3" => match read_nmrpipe_3d(p, "HNCACB") {
                Ok(spec3d) => {
                    self.hncacb_3d = Some(spec3d);
                    self.workspace_mode = WorkspaceMode::ProteinNmr;
                    self.status_msg = format!("Loaded NMRPipe 3D spectrum {}", p.display());
                }
                Err(e) => self.status_msg = format!("NMRPipe 3D load error: {e}"),
            },
            "cdx" => match std::fs::read(p) {
                Ok(bytes) => match read_cdx(&bytes) {
                    Ok(doc) => {
                        let mut sc = RenderScene::from(&doc);
                        let [cx, cy] = sc.atom_centroid();
                        sc.translate(-cx, -cy);
                        self.octet_app.set_scene(sc);
                        self.workspace_mode = WorkspaceMode::OctetChem;
                        self.octet_app.status_msg =
                            format!("Loaded binary ChemDraw CDX {}", p.display());
                        self.status_msg = format!("Loaded binary CDX {}", p.display());
                    }
                    Err(e) => self.status_msg = format!("CDX parse error: {e}"),
                },
                Err(e) => self.status_msg = format!("CDX read error: {e}"),
            },
            "cdxml" | "oct" | "mol" | "sdf" | "smi" | "jdx" | "dx" => {
                match std::fs::read_to_string(p) {
                    Ok(txt) => {
                        let fmt = match ext.as_str() {
                            "cdxml" => IoFormat::Cdxml,
                            "oct" => IoFormat::OctJson,
                            "mol" => IoFormat::MolV2000,
                            "sdf" => IoFormat::Sdf,
                            "jdx" | "dx" => IoFormat::JcampDx,
                            _ => IoFormat::Smiles,
                        };
                        if let Some(res) = CoreOctetBackendAdapter.import_custom(fmt, &txt) {
                            match res {
                                Ok(sc) => {
                                    self.octet_app.set_scene(sc);
                                    self.workspace_mode = WorkspaceMode::OctetChem;
                                    self.octet_app.status_msg =
                                        format!("Loaded {} from {}", fmt.label(), p.display());
                                    self.status_msg = format!("Loaded {}", p.display());
                                }
                                Err(e) => self.status_msg = format!("Import error: {e}"),
                            }
                        }
                    }
                    Err(e) => self.status_msg = format!("File read error: {e}"),
                }
            }
            _ => {
                if let Ok(bytes) = std::fs::read(p) {
                    if bytes.len() >= 180 && &bytes[0..8] == b"UCSF NMR" {
                        if let Ok(spec) = read_ucsf_2d_from_bytes(&fname_low, &bytes) {
                            let lbl = p
                                .file_stem()
                                .and_then(|s| s.to_str())
                                .map(str::to_string)
                                .unwrap_or_else(|| spec.name.clone());
                            self.install_single_2d_spectrum(
                                spec,
                                lbl,
                                format!("Loaded Sparky/UCSF 2D spectrum {}", p.display()),
                            );
                            return;
                        }
                        if let Some(sp1d) = try_load_1d_ucsf_or_pipe_bytes(&fname_low, &bytes) {
                            self.stack1d_panel.load_user_spectra(vec![sp1d]);
                            self.activate_1d_studio_mode();
                            self.status_msg =
                                format!("Loaded 1D UCSF spectrum {} into 1D Studio", p.display());
                            return;
                        }
                    }
                }
                self.status_msg = format!(
                    "Unrecognized file extension .{ext} (expected folder, .tar.gz, .ucsf/.usf, .str, .nef, .fasta, .tsv, .ft2, .ft3)"
                );
            }
        }
    }

    fn pick_2d_roots(&mut self) {
        let (nuc_x, nuc_y) = self.active_2d_nuclei();
        let Some(spec) = self.hsqc_spec.clone() else {
            self.status_msg = "No 2D spectrum loaded".to_string();
            return;
        };
        if nuc_x == "13C" && nuc_y == "15N" {
            let pcfg = C13PickerConfig {
                snr_2d: self.picker_snr,
                ..C13PickerConfig::default()
            };
            let mut con_roots = pick_con_2d(&spec, &pcfg);
            let min_vis = self.contour_base.max(0.0) as f64;
            if min_vis > 0.0 {
                con_roots.retain(|r| r.height.abs() >= min_vis * 0.85);
            }
            let n_picked = con_roots.len();
            let n_pro = con_roots
                .iter()
                .filter(|r| r.is_proline_imino_region)
                .count();
            let mut ds = self.c13_picked_dataset.take().unwrap_or_default();
            ds.con_roots = con_roots;
            self.c13_picked_dataset = Some(ds);
            self.sync_con_roots_to_peaks();
            self.workbook.two_d.show_peak_table = true;
            self.regime_kind = GuiRegimeKind::Idp13C;
            self.status_msg = format!(
                "13C-IDP CON picker found {n_picked} peptide-bond roots ({n_pro} Pro imino 132–142 ppm)"
            );
            return;
        }
        if !(nuc_x == "1H" && (nuc_y == "15N" || nuc_y == "13C")) {
            let picked = self.pick_generic_2d_crosspeaks(&spec, self.picker_snr);
            let n_picked = picked.len();
            self.peaks = picked;
            self.session.state.peaks = self.peaks.clone();
            self.workbook.two_d.selected_peak_idx = None;
            self.workbook.two_d.show_peak_table = true;
            self.systems.clear();
            self.status_msg = format!(
                "2D peak picker found {n_picked} {}–{} cross-peaks{}",
                Self::pretty_nucleus_unicode(nuc_x),
                Self::pretty_nucleus_unicode(nuc_y),
                if nuc_x == nuc_y && !self.workbook.two_d.pick_diagonal {
                    " (diagonal excluded)"
                } else {
                    ""
                }
            );
            return;
        }
        let (mut picked, method_label) = if nuc_y == "13C" {
            (
                pick_c13_hsqc_2d(
                    &spec,
                    &PickerConfig {
                        snr_threshold: self.picker_snr,
                        ..PickerConfig::default()
                    },
                ),
                "Classical 13C-HSQC picker",
            )
        } else if self.use_neural_picker {
            (
                pick_hsqc_2d_neural(
                    &spec,
                    &NeuralPickerConfig {
                        prob_threshold: (0.25 + 0.035 * (self.picker_snr - 3.0)).clamp(0.20, 0.85),
                        ..NeuralPickerConfig::default()
                    },
                ),
                "Neural 2D CNN picker",
            )
        } else {
            (
                pick_hsqc_2d(
                    &spec,
                    &PickerConfig {
                        snr_threshold: self.picker_snr,
                        ..PickerConfig::default()
                    },
                ),
                "Classical picker",
            )
        };
        picked.sort_by(|a, b| {
            b.height
                .unwrap_or(0.0)
                .abs()
                .total_cmp(&a.height.unwrap_or(0.0).abs())
        });
        let n_picked = picked.len();
        // Keep existing triple-resonance peaks while replacing 2D roots
        let root_exp = if nuc_y == "13C" { "C13HSQC" } else { "HSQC" };
        let mut combined = picked;
        combined.extend(
            self.peaks
                .iter()
                .filter(|p| p.exp != root_exp && p.c.is_some())
                .cloned(),
        );
        self.peaks = combined;
        self.session.state.peaks = self.peaks.clone();
        self.workbook.two_d.selected_peak_idx = None;
        self.workbook.two_d.show_peak_table = true;
        self.truth = None;
        self.metrics = None;
        if nuc_y == "15N" {
            self.rebuild_systems_and_matrices();
            self.status_msg = format!(
                "{method_label} found {n_picked} 2D roots -> rebuilt {} spin systems",
                self.systems.len()
            );
        } else {
            self.systems.clear();
            self.status_msg = format!("{method_label} found {n_picked} ¹H–¹³C cross-peaks");
        }
    }

    fn sync_con_roots_to_peaks(&mut self) {
        let Some(ref ds) = self.c13_picked_dataset else {
            return;
        };
        let mut out: Vec<Peak> = ds
            .con_roots
            .iter()
            .map(|r| Peak {
                exp: "CON".to_string(),
                h: r.c_prev_ppm,
                n: r.n_curr_ppm,
                c: None,
                h2: None,
                sign: if r.height < 0.0 { -1 } else { 1 },
                height: Some(r.height),
                volume: Some(r.volume),
            })
            .collect();
        out.extend(self.peaks.iter().filter(|p| p.c.is_some()).cloned());
        self.peaks = out;
        self.session.state.peaks = self.peaks.clone();
    }

    fn sync_2d_peaks_after_edit(&mut self) {
        self.session.state.peaks = self.peaks.clone();
        let (nx, ny) = self.active_2d_nuclei();
        if nx == "13C" && ny == "15N" {
            let new_con: Vec<Con2DPeak> = self
                .peaks
                .iter()
                .filter(|p| p.c.is_none())
                .map(|p| {
                    let hgt = p.height.unwrap_or(100.0);
                    Con2DPeak {
                        c_prev_ppm: p.h,
                        n_curr_ppm: p.n,
                        height: hgt,
                        volume: p.volume.unwrap_or(hgt),
                        is_proline_imino_region: p.n >= 131.5,
                        deconvolved_from_cluster: false,
                    }
                })
                .collect();
            let mut ds = self.c13_picked_dataset.take().unwrap_or_default();
            for sp in &mut ds.strip_peaks {
                sp.anchor_root_idx = new_con
                    .iter()
                    .enumerate()
                    .filter(|(_, r)| {
                        (r.c_prev_ppm - sp.c_prime_ppm).abs() <= 0.18
                            && (r.n_curr_ppm - sp.n_ppm).abs() <= 0.55
                    })
                    .min_by(|(_, a), (_, b)| {
                        let da = ((a.c_prev_ppm - sp.c_prime_ppm) / 0.08).powi(2)
                            + ((a.n_curr_ppm - sp.n_ppm) / 0.35).powi(2);
                        let db = ((b.c_prev_ppm - sp.c_prime_ppm) / 0.08).powi(2)
                            + ((b.n_curr_ppm - sp.n_ppm) / 0.35).powi(2);
                        da.total_cmp(&db)
                    })
                    .map(|(i, _)| i);
            }
            ds.con_roots = new_con;
            self.c13_picked_dataset = Some(ds);
        } else if nx == "1H" && ny == "15N" {
            self.rebuild_systems_and_matrices();
        }
    }

    fn default_2d_peak_exp_name(&self) -> String {
        let (nx, ny) = self.active_2d_nuclei();
        if nx == "13C" && ny == "15N" {
            "CON".to_string()
        } else if nx == "1H" && ny == "15N" {
            "HSQC".to_string()
        } else if nx == "1H" && ny == "13C" {
            "C13HSQC".to_string()
        } else {
            self.hsqc_spec
                .as_ref()
                .map(|s| s.name.clone())
                .filter(|s| !s.trim().is_empty())
                .unwrap_or_else(|| "2D".to_string())
        }
    }

    fn sample_2d_extremum_or_exact(
        &self,
        f2_ppm: f64,
        f1_ppm: f64,
        snap_radius_ppm: Option<(f64, f64)>,
    ) -> (f64, f64, f64, i32) {
        let Some(ref spec) = self.hsqc_spec else {
            return (f2_ppm, f1_ppm, 100.0, 1);
        };
        let nx = spec.nx();
        let ny = spec.ny();
        if nx < 3 || ny < 3 || spec.data.len() < nx * ny {
            return (f2_ppm, f1_ppm, 100.0, 1);
        }
        if let Some((rx_ppm, ry_ppm)) = snap_radius_ppm {
            let cx = spec
                .axis_x
                .ppm_to_index(f2_ppm)
                .round()
                .clamp(1.0, (nx - 2) as f64) as isize;
            let cy = spec
                .axis_y
                .ppm_to_index(f1_ppm)
                .round()
                .clamp(1.0, (ny - 2) as f64) as isize;
            let rx = ((rx_ppm / spec.axis_x.step_ppm.abs().max(1e-6)).ceil() as isize).clamp(1, 8);
            let ry = ((ry_ppm / spec.axis_y.step_ppm.abs().max(1e-6)).ceil() as isize).clamp(1, 8);
            let mut best_ix = cx as usize;
            let mut best_iy = cy as usize;
            let mut best_score = -1.0_f32;
            for dy in -ry..=ry {
                let iy = (cy + dy).clamp(1, (ny - 2) as isize) as usize;
                for dx in -rx..=rx {
                    let ix = (cx + dx).clamp(1, (nx - 2) as isize) as usize;
                    let v = spec.get(ix, iy);
                    let score = if self.show_negative { v.abs() } else { v };
                    if score > best_score {
                        best_score = score;
                        best_ix = ix;
                        best_iy = iy;
                    }
                }
            }
            let v0 = spec.get(best_ix, best_iy) as f64;
            let sign = if v0 < 0.0 && self.show_negative {
                -1
            } else {
                1
            };
            let sgn = sign as f64;
            let xm = (spec.get(best_ix - 1, best_iy) as f64) * sgn;
            let xc = v0 * sgn;
            let xp = (spec.get(best_ix + 1, best_iy) as f64) * sgn;
            let denom_x = xm - 2.0 * xc + xp;
            let sub_x = if denom_x < -1e-9 {
                (0.5 * (xm - xp) / denom_x).clamp(-0.5, 0.5)
            } else {
                0.0
            };
            let ym = (spec.get(best_ix, best_iy - 1) as f64) * sgn;
            let yp = (spec.get(best_ix, best_iy + 1) as f64) * sgn;
            let denom_y = ym - 2.0 * xc + yp;
            let sub_y = if denom_y < -1e-9 {
                (0.5 * (ym - yp) / denom_y).clamp(-0.5, 0.5)
            } else {
                0.0
            };
            (
                spec.axis_x.index_to_ppm(best_ix as f64 + sub_x),
                spec.axis_y.index_to_ppm(best_iy as f64 + sub_y),
                v0,
                sign,
            )
        } else {
            let ix_f = spec.axis_x.ppm_to_index(f2_ppm).clamp(0.0, (nx - 1) as f64);
            let iy_f = spec.axis_y.ppm_to_index(f1_ppm).clamp(0.0, (ny - 1) as f64);
            let ix0 = (ix_f.floor() as usize).min(nx - 1);
            let ix1 = (ix0 + 1).min(nx - 1);
            let iy0 = (iy_f.floor() as usize).min(ny - 1);
            let iy1 = (iy0 + 1).min(ny - 1);
            let tx = (ix_f - ix0 as f64) as f32;
            let ty = (iy_f - iy0 as f64) as f32;
            let r0 = spec.get(ix0, iy0) * (1.0 - tx) + spec.get(ix1, iy0) * tx;
            let r1 = spec.get(ix0, iy1) * (1.0 - tx) + spec.get(ix1, iy1) * tx;
            let amp = (r0 * (1.0 - ty) + r1 * ty) as f64;
            let sign = if amp < 0.0 && self.show_negative {
                -1
            } else {
                1
            };
            (f2_ppm, f1_ppm, amp, sign)
        }
    }

    fn add_manual_2d_peak(
        &mut self,
        f2_ppm: f64,
        f1_ppm: f64,
        snap_radius_ppm: Option<(f64, f64)>,
    ) -> usize {
        let (h, n, amp, sign) = self.sample_2d_extremum_or_exact(f2_ppm, f1_ppm, snap_radius_ppm);
        let pk = Peak {
            exp: self.default_2d_peak_exp_name(),
            h,
            n,
            c: None,
            h2: None,
            sign,
            height: Some(amp),
            volume: Some(amp),
        };
        let _ = self.session.execute(WorkbenchCommand::AddPeak(pk));
        self.peaks = self.session.state.peaks.clone();
        let new_idx = self.peaks.len().saturating_sub(1);
        self.workbook.two_d.selected_peak_idx = Some(new_idx);
        self.crosshair.snap_to(h, n, self.crosshair.c_ppm);
        self.sync_2d_peaks_after_edit();
        let (nx, ny) = self.active_2d_nuclei();
        self.status_msg = format!(
            "Added 2D peak #{} at {} {:.3}, {} {:.3} ppm",
            new_idx + 1,
            Self::pretty_nucleus_unicode(nx),
            h,
            Self::pretty_nucleus_unicode(ny),
            n
        );
        new_idx
    }

    fn box_pick_2d_region(&mut self, f2_a: f64, f2_b: f64, f1_a: f64, f1_b: f64) -> usize {
        let Some(spec) = self.hsqc_spec.clone() else {
            return 0;
        };
        let f2_lo = f2_a.min(f2_b);
        let f2_hi = f2_a.max(f2_b);
        let f1_lo = f1_a.min(f1_b);
        let f1_hi = f1_a.max(f1_b);
        let (nx_lbl, ny_lbl) = self.active_2d_nuclei();
        let cands = pick_2d_crosspeaks_ridge_safe(
            &spec,
            (self.picker_snr * 0.85).max(2.5),
            self.contour_base,
            nx_lbl == ny_lbl,
            true,
            self.show_negative,
        );
        let tol_x = (spec.axis_x.step_ppm.abs() * 2.2).max(0.008);
        let tol_y = (spec.axis_y.step_ppm.abs() * 2.2).max(0.015);
        let exp_name = self.default_2d_peak_exp_name();
        let mut added = 0usize;
        for mut cand in cands {
            if cand.h < f2_lo || cand.h > f2_hi || cand.n < f1_lo || cand.n > f1_hi {
                continue;
            }
            let dup = self.peaks.iter().any(|p| {
                p.c.is_none() && (p.h - cand.h).abs() <= tol_x && (p.n - cand.n).abs() <= tol_y
            });
            if !dup {
                cand.exp = exp_name.clone();
                self.peaks.push(cand);
                added += 1;
            }
        }
        if added > 0 {
            self.workbook.two_d.selected_peak_idx = Some(self.peaks.len() - 1);
            self.workbook.two_d.show_peak_table = true;
            self.sync_2d_peaks_after_edit();
            self.status_msg = format!("Box-picked {added} 2D cross-peak(s) in region");
        } else {
            self.status_msg =
                "No additional cross-peaks above contour floor inside box (lower contour floor or click directly to add)".to_string();
        }
        added
    }

    fn move_2d_peak_live(&mut self, peak_idx: usize, new_f2: f64, new_f1: f64, finalize: bool) {
        if peak_idx >= self.peaks.len() || !new_f2.is_finite() || !new_f1.is_finite() {
            return;
        }
        let old_key = peak_label_key_2d(self.peaks[peak_idx].h, self.peaks[peak_idx].n);
        let (h, n, amp, sign) = self.sample_2d_extremum_or_exact(new_f2, new_f1, None);
        self.peaks[peak_idx].h = h;
        self.peaks[peak_idx].n = n;
        self.peaks[peak_idx].height = Some(amp);
        self.peaks[peak_idx].sign = sign;
        let new_key = peak_label_key_2d(h, n);
        if old_key != new_key {
            if let Some(lbl) = self.peak_2d_labels.remove(&old_key) {
                self.peak_2d_labels.insert(new_key, lbl);
            }
        }
        self.crosshair.snap_to(h, n, self.crosshair.c_ppm);
        if finalize {
            self.sync_2d_peaks_after_edit();
            self.status_msg = format!("Moved 2D peak #{} to ({h:.3}, {n:.3}) ppm", peak_idx + 1);
        }
    }

    fn snap_2d_peak_by_index(&mut self, peak_idx: usize) {
        if peak_idx >= self.peaks.len() {
            return;
        }
        let (cur_h, cur_n) = (self.peaks[peak_idx].h, self.peaks[peak_idx].n);
        let (rx_ppm, ry_ppm) = if let Some(ref sp) = self.hsqc_spec {
            (
                (sp.axis_x.step_ppm.abs() * 4.0).max(0.015),
                (sp.axis_y.step_ppm.abs() * 4.0).max(0.04),
            )
        } else {
            (0.03, 0.25)
        };
        let old_key = peak_label_key_2d(cur_h, cur_n);
        let (h, n, amp, sign) =
            self.sample_2d_extremum_or_exact(cur_h, cur_n, Some((rx_ppm, ry_ppm)));
        self.peaks[peak_idx].h = h;
        self.peaks[peak_idx].n = n;
        self.peaks[peak_idx].height = Some(amp);
        self.peaks[peak_idx].sign = sign;
        let new_key = peak_label_key_2d(h, n);
        if old_key != new_key {
            if let Some(lbl) = self.peak_2d_labels.remove(&old_key) {
                self.peak_2d_labels.insert(new_key, lbl);
            }
        }
        self.crosshair.snap_to(h, n, self.crosshair.c_ppm);
        self.sync_2d_peaks_after_edit();
        self.status_msg = format!("Snapped 2D peak #{} to ({h:.3}, {n:.3}) ppm", peak_idx + 1);
    }

    fn delete_2d_peak(&mut self, peak_idx: usize) {
        if peak_idx >= self.peaks.len() {
            return;
        }
        let removed = self.peaks[peak_idx].clone();
        self.peak_2d_labels
            .remove(&peak_label_key_2d(removed.h, removed.n));
        self.session.state.peaks = self.peaks.clone();
        let _ = self
            .session
            .execute(WorkbenchCommand::DeletePeaks(vec![peak_idx]));
        self.peaks = self.session.state.peaks.clone();
        self.workbook.two_d.dragging_peak_idx = None;
        if self.workbook.two_d.selected_peak_idx == Some(peak_idx) {
            self.workbook.two_d.selected_peak_idx = None;
        } else if let Some(sel) = self.workbook.two_d.selected_peak_idx {
            if sel > peak_idx {
                self.workbook.two_d.selected_peak_idx = Some(sel - 1);
            }
        }
        self.sync_2d_peaks_after_edit();
        self.status_msg = format!(
            "Deleted 2D peak at ({:.3}, {:.3}) ppm",
            removed.h, removed.n
        );
    }

    fn zoom_to_2d_peak(&mut self, peak_idx: usize) {
        let Some(pk) = self.peaks.get(peak_idx).cloned() else {
            return;
        };
        let prev = [
            self.view_h_left,
            self.view_h_right,
            self.view_n_top,
            self.view_n_bottom,
        ];
        self.workbook.two_d.push_zoom(prev);
        let (nx, ny) = self.active_2d_nuclei();
        let half_x = if nx == "1H" { 0.35_f32 } else { 2.2_f32 };
        let half_y = if ny == "1H" { 0.35_f32 } else { 3.2_f32 };
        let limits = self.view_limits_for_active_2d();
        let clamped = clamp_view_ppm_2d(
            [
                pk.h as f32 + half_x,
                pk.h as f32 - half_x,
                pk.n as f32 - half_y,
                pk.n as f32 + half_y,
            ],
            limits,
        );
        self.view_h_left = clamped[0];
        self.view_h_right = clamped[1];
        self.view_n_top = clamped[2];
        self.view_n_bottom = clamped[3];
        self.workbook.two_d.selected_peak_idx = Some(peak_idx);
        self.crosshair.snap_to(pk.h, pk.n, self.crosshair.c_ppm);
    }

    fn export_2d_peaks_tsv(&self) -> String {
        let (nx, ny) = self.active_2d_nuclei();
        let mut rows = vec![format!(
            "index\texp\tf2_{nx}_ppm\tf1_{ny}_ppm\theight\tsign\tlabel"
        )];
        for (idx, pk) in self.peaks.iter().filter(|p| p.c.is_none()).enumerate() {
            let ord = idx + 1;
            let lbl = self
                .peak_2d_labels
                .get(&peak_label_key_2d(pk.h, pk.n))
                .map(String::as_str)
                .unwrap_or("");
            rows.push(format!(
                "{ord}\t{}\t{:.4}\t{:.4}\t{:.4e}\t{}\t{}",
                pk.exp,
                pk.h,
                pk.n,
                pk.height.unwrap_or(0.0),
                pk.sign,
                lbl
            ));
        }
        rows.join("\n") + "\n"
    }

    fn draw_2d_peaks_bottom_table(&mut self, ui: &mut egui::Ui) {
        let indices: Vec<usize> = self
            .peaks
            .iter()
            .enumerate()
            .filter_map(|(i, p)| p.c.is_none().then_some(i))
            .collect();
        let (nx, ny) = self.active_2d_nuclei();
        let px = Self::pretty_nucleus_unicode(nx);
        let py = Self::pretty_nucleus_unicode(ny);

        ui.add_space(2.0);
        ui.horizontal_wrapped(|ui| {
            let tab_lbl = format!("2D Peaks ({})", indices.len());
            if ui
                .selectable_label(self.workbook.two_d.show_peak_table, &tab_lbl)
                .on_hover_text("Show / hide interactive 2D Cross-Peak Curation Table")
                .clicked()
            {
                self.workbook.two_d.show_peak_table = !self.workbook.two_d.show_peak_table;
            }
            if ui
                .small_button("+ Add Peak")
                .on_hover_text("Add a 2D peak at current crosshair (or Left-Click in [K] mode / Right-Click anywhere on 2D plot)")
                .clicked()
            {
                let rx = self
                    .hsqc_spec
                    .as_ref()
                    .map(|s| (s.axis_x.step_ppm.abs() * 3.0).max(0.01))
                    .unwrap_or(0.02);
                let ry = self
                    .hsqc_spec
                    .as_ref()
                    .map(|s| (s.axis_y.step_ppm.abs() * 3.0).max(0.02))
                    .unwrap_or(0.10);
                let _ = self.add_manual_2d_peak(
                    self.crosshair.h_ppm,
                    self.crosshair.n_ppm,
                    Some((rx, ry)),
                );
                self.workbook.two_d.show_peak_table = true;
            }
            if ui
                .small_button("Auto-Pick")
                .on_hover_text("Run ridge-safe 2D cross-peak picker above current contour floor")
                .clicked()
            {
                self.pick_2d_roots();
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let toggle_lbl = if self.workbook.two_d.show_peak_table {
                    "Hide Table"
                } else {
                    "Show Table"
                };
                if ui.small_button(toggle_lbl).clicked() {
                    self.workbook.two_d.show_peak_table = !self.workbook.two_d.show_peak_table;
                }
            });
        });

        if !self.workbook.two_d.show_peak_table {
            return;
        }

        egui::Frame::group(ui.style())
            .fill(egui::Color32::from_rgb(15, 21, 32))
            .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(45, 58, 78)))
            .inner_margin(egui::Margin::symmetric(8, 5))
            .show(ui, |ui| {
                let mut repick = false;
                ui.horizontal_wrapped(|ui| {
                    ui.strong("2D Peak Picker:");
                    ui.label("Min SNR:");
                    if ui
                        .add(
                            egui::DragValue::new(&mut self.picker_snr)
                                .range(2.0..=50.0)
                                .speed(0.2)
                                .fixed_decimals(1),
                        )
                        .changed()
                    {
                        repick = true;
                    }
                    if ui.button("Re-Pick 2D").clicked() {
                        repick = true;
                    }
                    if nx == ny
                        && ui
                            .checkbox(&mut self.workbook.two_d.pick_diagonal, "Pick Diagonal")
                            .on_hover_text("Include δ₁ ≈ δ₂ homonuclear diagonal self-peaks")
                            .changed()
                    {
                        repick = true;
                    }
                    if ui
                        .checkbox(&mut self.show_negative, "Neg Contours / Peaks")
                        .changed()
                    {
                        repick = true;
                    }
                    ui.checkbox(&mut self.show_peak_labels, "Peak Labels");
                    if !indices.is_empty() && ui.button("Copy TSV").clicked() {
                        ui.ctx().copy_text(self.export_2d_peaks_tsv());
                        self.status_msg =
                            format!("Copied {} 2D peaks (TSV) to clipboard", indices.len());
                    }
                    if !indices.is_empty() && ui.button("Clear All Peaks").clicked() {
                        self.peaks.retain(|p| p.c.is_some());
                        self.peak_2d_labels.clear();
                        self.workbook.two_d.selected_peak_idx = None;
                        self.sync_2d_peaks_after_edit();
                        self.status_msg = "Cleared all 2D peaks".to_string();
                    }
                });
                if repick {
                    self.pick_2d_roots();
                }

                if indices.is_empty() {
                    ui.label(
                        egui::RichText::new(
                            "No 2D peaks picked yet. Click 'Auto-Pick', use [K] on the left tool rail to click/box-pick on the spectrum, or Right-Click anywhere on the 2D plot.",
                        )
                        .weak()
                        .small(),
                    );
                    return;
                }

                let mut jump_req: Option<usize> = None;
                let mut snap_req: Option<usize> = None;
                let mut del_req: Option<usize> = None;
                let mut move_req: Option<(usize, f64, f64)> = None;

                egui::ScrollArea::vertical()
                    .id_salt("two_d_peaks_table_scroll")
                    .max_height(132.0)
                    .show(ui, |ui| {
                        egui::Grid::new("two_d_peaks_grid")
                            .striped(true)
                            .spacing([12.0, 3.0])
                            .show(ui, |ui| {
                                ui.strong("#");
                                ui.strong(format!("F2 ({px} ppm)"));
                                ui.strong(format!("F1 ({py} ppm)"));
                                ui.strong("Height");
                                ui.strong("Sign");
                                ui.strong("Label / Assignment");
                                ui.strong("Actions");
                                ui.end_row();

                                for (disp_ord, &pk_idx) in indices.iter().take(500).enumerate() {
                                    let Some(pk) = self.peaks.get(pk_idx).cloned() else {
                                        continue;
                                    };
                                    let is_sel =
                                        self.workbook.two_d.selected_peak_idx == Some(pk_idx);
                                    if ui
                                        .selectable_label(is_sel, format!("{}", disp_ord + 1))
                                        .on_hover_text("Click to center & zoom 2D viewport onto peak")
                                        .clicked()
                                    {
                                        jump_req = Some(pk_idx);
                                    }
                                    let mut f2_edit = pk.h;
                                    let mut f1_edit = pk.n;
                                    let spd_x = if nx == "1H" { 0.001 } else { 0.01 };
                                    let spd_y = if ny == "1H" { 0.001 } else { 0.01 };
                                    let c_x = ui
                                        .add(
                                            egui::DragValue::new(&mut f2_edit)
                                                .speed(spd_x)
                                                .fixed_decimals(3),
                                        )
                                        .changed();
                                    let c_y = ui
                                        .add(
                                            egui::DragValue::new(&mut f1_edit)
                                                .speed(spd_y)
                                                .fixed_decimals(3),
                                        )
                                        .changed();
                                    if c_x || c_y {
                                        move_req = Some((pk_idx, f2_edit, f1_edit));
                                    }
                                    if ui
                                        .selectable_label(
                                            is_sel,
                                            egui::RichText::new(format!(
                                                "{:.3e}",
                                                pk.height.unwrap_or(0.0)
                                            ))
                                            .monospace(),
                                        )
                                        .clicked()
                                    {
                                        jump_req = Some(pk_idx);
                                    }
                                    ui.monospace(if pk.sign < 0 { "-" } else { "+" });
                                    let key = peak_label_key_2d(pk.h, pk.n);
                                    let mut lbl_buf =
                                        self.peak_2d_labels.get(&key).cloned().unwrap_or_default();
                                    if ui
                                        .add(
                                            egui::TextEdit::singleline(&mut lbl_buf)
                                                .desired_width(135.0)
                                                .hint_text("label / residue"),
                                        )
                                        .changed()
                                    {
                                        if lbl_buf.trim().is_empty() {
                                            self.peak_2d_labels.remove(&key);
                                        } else {
                                            self.peak_2d_labels.insert(key, lbl_buf);
                                        }
                                    }
                                    ui.horizontal(|ui| {
                                        if ui
                                            .small_button("Jump")
                                            .on_hover_text("Zoom 2D canvas onto peak")
                                            .clicked()
                                        {
                                            jump_req = Some(pk_idx);
                                        }
                                        if ui
                                            .small_button("Snap")
                                            .on_hover_text("Snap peak to local 2D parabolic extremum")
                                            .clicked()
                                        {
                                            snap_req = Some(pk_idx);
                                        }
                                        if ui
                                            .small_button("×")
                                            .on_hover_text("Delete peak")
                                            .clicked()
                                        {
                                            del_req = Some(pk_idx);
                                        }
                                    });
                                    ui.end_row();
                                }
                            });
                    });

                if let Some(di) = del_req {
                    self.delete_2d_peak(di);
                } else if let Some((mi, nf2, nf1)) = move_req {
                    self.move_2d_peak_live(mi, nf2, nf1, true);
                } else if let Some(si) = snap_req {
                    self.snap_2d_peak_by_index(si);
                } else if let Some(ji) = jump_req {
                    self.zoom_to_2d_peak(ji);
                }
            });
    }

    fn format_shifts_tsv(&self, pred: &Reference) -> String {
        let mut rows = vec!["seq_id\tresidue\tatom\tshift\tconfidence".to_string()];
        for (&(k, ref atom), &v) in &pred.shifts {
            let c = self.pred_conf.get(&k).copied().unwrap_or(0.0);
            rows.push(format!(
                "{}\t{}\t{}\t{:.3}\t{:.2}",
                self.seq_ids[k], self.sequence[k], atom, v, c
            ));
        }
        rows.join("\n") + "\n"
    }

    fn export_results(&mut self, ctx: &egui::Context) {
        let Some(pred) = &self.pred_ref else {
            self.status_msg = "Run assignment first before exporting".to_string();
            return;
        };
        let tsv_text = self.format_shifts_tsv(pred);
        let star_text = format_star_in_memory(pred, Some(&self.pred_conf));
        self.export_text_buf =
            format!("# --- NMR-STAR v3 ---\n{star_text}\n# --- Shifts TSV ---\n{tsv_text}");
        self.exported_in_session = true;

        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = ctx;
            let prefix = Path::new(self.export_prefix.trim());
            let str_p = prefix.with_extension("str");
            let tsv_p = prefix.with_extension("tsv");
            if let Err(e) = write_star(&str_p, pred, Some(&self.pred_conf)) {
                self.status_msg = format!("Export failed: {e}");
                return;
            }
            let _ = std::fs::write(&tsv_p, &tsv_text);
            self.status_msg = format!("Exported {} and {}", str_p.display(), tsv_p.display());
        }

        #[cfg(target_arch = "wasm32")]
        {
            ctx.copy_text(star_text);
            self.show_text_io_modal = true;
            self.status_msg =
                "Exported NMR-STAR v3 + TSV in memory (NMR-STAR copied to clipboard)".to_string();
        }
    }

    fn is_1d_studio_active(&self) -> bool {
        self.workspace_mode == WorkspaceMode::ProteinNmr && self.show_stack1d_panel
    }

    fn activate_1d_studio_mode(&mut self) {
        self.workspace_mode = WorkspaceMode::ProteinNmr;
        self.show_stack1d_panel = true;
        self.show_relax_panel = false;
        self.show_dynamics_panel = false;
        self.show_mol3d_viewport = false;
        self.show_strips_dock = false;
        self.responsive_shell.tablet_dock = TabletActiveDock::Hidden;
        self.responsive_shell.mobile_family = MobileWorkspaceFamily::ChemStudio;
        self.responsive_shell.chem_tab = MobileChemTab::Multiplets1dQnmr;
    }

    fn current_workbook_selection(&self) -> ActiveWorkbookSelection {
        match self.workspace_mode {
            WorkspaceMode::OctetChem => {
                ActiveWorkbookSelection::Studio(WorkbookStudioKind::ChemicalStructure)
            }
            WorkspaceMode::Dosy => {
                ActiveWorkbookSelection::Studio(WorkbookStudioKind::DosyDiffusion)
            }
            WorkspaceMode::Metabolomics => {
                ActiveWorkbookSelection::Studio(WorkbookStudioKind::MetabolomicsCohort)
            }
            WorkspaceMode::ProteinNmr => {
                if self.show_relax_panel || self.show_dynamics_panel {
                    ActiveWorkbookSelection::Studio(WorkbookStudioKind::RelaxationDynamics)
                } else if self.show_stack1d_panel {
                    let idx = self
                        .stack1d_panel
                        .ref_trace_idx
                        .min(self.stack1d_panel.traces.len().saturating_sub(1));
                    ActiveWorkbookSelection::Trace1D(idx)
                } else {
                    ActiveWorkbookSelection::Spectrum2D(self.workbook.active_2d_idx)
                }
            }
        }
    }

    fn handle_workbook_sidebar_actions(&mut self, actions: Vec<WorkbookSidebarAction>) {
        for action in actions {
            match action {
                WorkbookSidebarAction::Select1dPage(idx) => {
                    self.activate_1d_workbook_page(idx);
                }
                WorkbookSidebarAction::Toggle1dOverlay(idx) => {
                    if let Some(tr) = self.stack1d_panel.traces.get_mut(idx) {
                        tr.overlay_pinned = !tr.overlay_pinned;
                        tr.visible = tr.overlay_pinned || idx == self.stack1d_panel.ref_trace_idx;
                        let now_vis = tr.visible;
                        let lbl = tr.label.clone();
                        let vis_count = self
                            .stack1d_panel
                            .traces
                            .iter()
                            .filter(|t| t.visible)
                            .count();
                        self.status_msg = if now_vis {
                            format!("Superimposed '{lbl}' ({vis_count} visible 1D traces)")
                        } else {
                            format!("Hid '{lbl}' from 1D overlay")
                        };
                    }
                }
                WorkbookSidebarAction::Remove1dPage(idx) => {
                    if idx < self.stack1d_panel.traces.len() {
                        let removed = self.stack1d_panel.traces.remove(idx);
                        if self.stack1d_panel.traces.is_empty() {
                            self.stack1d_panel.ref_trace_idx = 0;
                            if !self.workbook.pages_2d.is_empty() {
                                self.activate_2d_workbook_page(
                                    self.workbook
                                        .active_2d_idx
                                        .min(self.workbook.pages_2d.len() - 1),
                                );
                            }
                        } else {
                            let next_idx = self
                                .stack1d_panel
                                .ref_trace_idx
                                .min(self.stack1d_panel.traces.len() - 1);
                            self.stack1d_panel
                                .restore_or_init_trace_page_without_saving(next_idx);
                            self.activate_1d_studio_mode();
                        }
                        self.status_msg = format!("Closed 1D page '{}'", removed.label);
                    }
                }
                WorkbookSidebarAction::Select2dPage(idx) => {
                    self.activate_2d_workbook_page(idx);
                }
                WorkbookSidebarAction::Toggle2dOverlay(idx) => {
                    if let Some(pg) = self.workbook.pages_2d.get_mut(idx) {
                        pg.overlay_enabled = !pg.overlay_enabled;
                        let enabled = pg.overlay_enabled;
                        let title = pg.title.clone();
                        self.sync_overlay_layers_from_workbook();
                        self.status_msg = if enabled {
                            format!("Enabled 2D GPU contour overlay: '{title}'")
                        } else {
                            format!("Hid 2D contour overlay: '{title}'")
                        };
                    }
                }
                WorkbookSidebarAction::Remove2dPage(idx) => {
                    if idx < self.workbook.pages_2d.len() {
                        let removed = self.workbook.pages_2d.remove(idx);
                        if self.workbook.pages_2d.is_empty() {
                            self.hsqc_spec = None;
                            self.hsqc_arc = Arc::new(Vec::new());
                            self.overlay_layers.clear();
                            if !self.stack1d_panel.traces.is_empty() {
                                self.activate_1d_workbook_page(0);
                            }
                        } else {
                            let next = self
                                .workbook
                                .active_2d_idx
                                .min(self.workbook.pages_2d.len() - 1);
                            self.activate_2d_workbook_page(next);
                        }
                        self.status_msg = format!("Closed 2D page '{}'", removed.title);
                    }
                }
                WorkbookSidebarAction::OpenStudio(kind) => match kind {
                    WorkbookStudioKind::ChemicalStructure => {
                        self.workspace_mode = WorkspaceMode::OctetChem;
                        self.show_stack1d_panel = false;
                    }
                    WorkbookStudioKind::Protein3dStrips => {
                        let protein_idx = self
                            .workbook
                            .pages_2d
                            .iter()
                            .position(|p| p.has_3d_strips)
                            .unwrap_or(self.workbook.active_2d_idx);
                        if !self.workbook.pages_2d.is_empty() {
                            self.activate_2d_workbook_page(protein_idx);
                        } else {
                            self.workspace_mode = WorkspaceMode::ProteinNmr;
                            self.show_stack1d_panel = false;
                        }
                        self.show_strips_dock = !self.show_strips_dock;
                    }
                    WorkbookStudioKind::Biomolecular3dPdb => {
                        if self.show_stack1d_panel
                            || self.workspace_mode != WorkspaceMode::ProteinNmr
                        {
                            let protein_idx = self
                                .workbook
                                .pages_2d
                                .iter()
                                .position(|p| p.has_3d_strips)
                                .unwrap_or(self.workbook.active_2d_idx);
                            if !self.workbook.pages_2d.is_empty() {
                                self.activate_2d_workbook_page(protein_idx);
                            } else {
                                self.workspace_mode = WorkspaceMode::ProteinNmr;
                                self.show_stack1d_panel = false;
                            }
                        }
                        self.show_mol3d_viewport = !self.show_mol3d_viewport;
                    }
                    WorkbookStudioKind::RelaxationDynamics => {
                        self.workspace_mode = WorkspaceMode::ProteinNmr;
                        self.show_stack1d_panel = false;
                        self.show_relax_panel = true;
                        self.show_dynamics_panel = false;
                    }
                    WorkbookStudioKind::DosyDiffusion => {
                        if self.dosy_panel.peaks.is_empty() {
                            self.dosy_panel = DosyPanelState::with_synthetic_demo();
                        }
                        self.workspace_mode = WorkspaceMode::Dosy;
                        self.show_stack1d_panel = false;
                    }
                    WorkbookStudioKind::MetabolomicsCohort => {
                        if self.metabolomics_panel.samples.is_empty() {
                            self.metabolomics_panel.load_demo_cohort();
                        }
                        self.workspace_mode = WorkspaceMode::Metabolomics;
                        self.show_stack1d_panel = false;
                    }
                },
                WorkbookSidebarAction::TriggerOpenFiles => {
                    self.arm_browser_picker_wakeup();
                    #[cfg(target_arch = "wasm32")]
                    trigger_browser_files_picker();
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        self.show_native_global_open_bar = true;
                    }
                }
            }
        }
    }

    fn render_and_sync_stack1d_panel(&mut self, ui: &mut egui::Ui) {
        ui.ctx().data_mut(|d| {
            d.insert_temp::<(bool, bool)>(
                egui::Id::unique("ferric_wb_cfg"),
                (true, self.paper_white_canvas),
            );
        });
        render_stack1d_panel(ui, &mut self.stack1d_panel);
        if self.stack1d_panel.pending_water_suppress_1d {
            self.stack1d_panel.pending_water_suppress_1d = false;
            self.suppress_water_active_1d();
        }
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(p_str) = self.stack1d_panel.pending_open_native_path.take() {
            self.path_input = p_str;
            self.load_path_auto();
        }
        if self.stack1d_panel.pending_send_to_metabolomics {
            self.stack1d_panel.pending_send_to_metabolomics = false;
            let cohort = self.stack1d_panel.build_metabolomics_cohort_samples();
            if cohort.len() >= 3 {
                let n = cohort.len();
                self.metabolomics_panel.set_cohort_and_run(cohort);
                self.workspace_mode = WorkspaceMode::Metabolomics;
                self.responsive_shell.mobile_family = MobileWorkspaceFamily::ChemStudio;
                self.responsive_shell.chem_tab = MobileChemTab::Metabolomics;
                self.status_msg =
                    format!("Sent {n} 1D spectra into Metabolomics PCA / OPLS-DA / STOCSY Studio");
            }
        }
    }

    fn apply_drop_outcome(&mut self, outcome: IngestDropOutcome) {
        self.bare_bruker_hint = None;
        self.picker_wakeup_until_ms = 0;
        match outcome {
            IngestDropOutcome::NefSession {
                source_name,
                nef_bytes,
            } => match String::from_utf8(nef_bytes) {
                Ok(txt) => self.load_text_in_memory(&txt, &source_name),
                Err(e) => self.status_msg = format!("Dropped NEF is not valid UTF-8: {e}"),
            },
            IngestDropOutcome::TextTable { source_name, text } => {
                self.load_text_in_memory(&text, &source_name);
            }
            IngestDropOutcome::Spectra1D {
                source_name,
                spectra,
            } => {
                let n = spectra.len();
                self.prepare_workbook_for_user_append();
                self.stack1d_panel.load_user_spectra(spectra);
                self.activate_1d_studio_mode();
                self.status_msg =
                    format!("Loaded {n} 1D spectrum(a) from '{source_name}' into 1D NMR Studio");
            }
            IngestDropOutcome::Workspace(panel) => {
                let n_exp = panel.inventory.experiments.len();
                let root = panel.inventory.root_name.clone();
                let single_direct_2d = n_exp == 1
                    && panel.inventory.auxiliary_files.is_empty()
                    && matches!(
                        panel.inventory.experiments[0].dimensionality,
                        ExperimentDimensionality::Dim2D | ExperimentDimensionality::Pseudo3D
                    );
                self.ingest_panel = *panel;
                self.workspace_mode = WorkspaceMode::ProteinNmr;
                self.show_stack1d_panel = false;
                if single_direct_2d {
                    self.ingest_panel.open = false;
                    let plan = self.ingest_panel.build_execution_plan();
                    self.execute_ingest_plan(plan, false);
                } else {
                    self.ingest_panel.open = true;
                    self.status_msg = format!(
                        "Classified '{root}' ({n_exp} experiment(s)) — review selection matrix"
                    );
                }
            }
        }
    }

    fn record_ingest_error(&mut self, prefix: &str, err: String) {
        if is_bare_bruker_file_error(&err) {
            self.bare_bruker_hint = Some(err.clone());
        }
        self.status_msg = format!("{prefix}: {err}");
    }

    fn handle_dropped_files(&mut self, ctx: &egui::Context) {
        install_browser_folder_drop_hook();
        if let Some(err) = poll_browser_drop_error() {
            self.record_ingest_error("Browser ingest", err);
            ctx.request_repaint();
        }
        if let Some(batch) = poll_browser_dropped_payloads() {
            match ingest_dropped_payloads(&batch) {
                Ok(outcome) => self.apply_drop_outcome(outcome),
                Err(e) => self.record_ingest_error("Ingest error", e),
            }
            ctx.request_repaint();
        }

        #[cfg(target_arch = "wasm32")]
        {
            let ready = self
                .pending_drop_outcome
                .lock()
                .ok()
                .and_then(|mut g| g.take());
            if let Some(res) = ready {
                match res {
                    Ok(outcome) => self.apply_drop_outcome(outcome),
                    Err(e) => self.record_ingest_error("Drop ingest error", e),
                }
                ctx.request_repaint();
            }
        }

        let dropped = ctx.input(|i| i.raw.dropped_files.clone());
        if !dropped.is_empty() {
            // Intercept 3D biomolecular structure files (`.pdb`, `.ent`, `.cif`, `.mmcif`) directly
            let mut struct_files = Vec::new();
            let mut remaining = Vec::with_capacity(dropped.len());
            for df in dropped {
                let fname = df
                    .path()
                    .file_name()
                    .and_then(|s| s.to_str())
                    .map(str::to_string)
                    .unwrap_or_else(|| df.path().display().to_string());
                let low = fname.to_ascii_lowercase();
                if low.ends_with(".pdb")
                    || low.ends_with(".ent")
                    || low.ends_with(".cif")
                    || low.ends_with(".mmcif")
                {
                    struct_files.push((fname, df));
                } else {
                    remaining.push(df);
                }
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                let _ = ctx;
                for (fname, df) in struct_files {
                    let loaded_txt = df
                        .bytes()
                        .or_else(|_| std::fs::read(df.path()).map_err(|e| e.to_string()))
                        .and_then(|b| {
                            String::from_utf8(b).map_err(|e| format!("UTF-8 error: {e}"))
                        });
                    match loaded_txt {
                        Ok(txt) => self.load_structure_3d_in_memory(&txt, &fname),
                        Err(e) => self.status_msg = format!("Structure drop error: {e}"),
                    }
                }
                if !remaining.is_empty() {
                    match ingest_dropped_files_native(&remaining) {
                        Ok(outcome) => self.apply_drop_outcome(outcome),
                        Err(e) => self.record_ingest_error("Drop ingest error", e),
                    }
                }
            }
            #[cfg(target_arch = "wasm32")]
            {
                if let Some((fname, df)) = struct_files.into_iter().next() {
                    let slot = Arc::clone(&self.pending_drop_outcome);
                    let egui_ctx = ctx.clone();
                    wasm_bindgen_futures::spawn_local(async move {
                        let res = df
                            .bytes_async()
                            .await
                            .and_then(|b| {
                                String::from_utf8(b).map_err(|e| format!("UTF-8 error: {e}"))
                            })
                            .map(|text| IngestDropOutcome::TextTable {
                                source_name: fname,
                                text,
                            });
                        if let Ok(mut g) = slot.lock() {
                            *g = Some(res);
                        }
                        egui_ctx.request_repaint();
                    });
                } else if !remaining.is_empty() {
                    let slot = Arc::clone(&self.pending_drop_outcome);
                    let egui_ctx = ctx.clone();
                    wasm_bindgen_futures::spawn_local(async move {
                        let res = ingest_dropped_files_wasm(remaining).await;
                        if let Ok(mut g) = slot.lock() {
                            *g = Some(res);
                        }
                        egui_ctx.request_repaint();
                    });
                }
            }
        }
    }

    fn draw_text_io_window(&mut self, ctx: &egui::Context) {
        if !self.show_text_io_modal {
            return;
        }
        let mut open = self.show_text_io_modal;
        let max_modal_w = (ctx.content_rect().width() - 16.0).max(240.0);
        egui::Window::new("Paste / In-Memory NMR-STAR & Peak TSV")
            .open(&mut open)
            .default_width(560.0_f32.min(max_modal_w))
            .max_width(max_modal_w)
            .resizable(true)
            .show(ctx, |ui| {
                ui.label(
                    egui::RichText::new(
                        "Paste an NMR-STAR (.str) entry or TSV peak list below (or drag & drop a .str/.tsv file onto the canvas):",
                    )
                    .small(),
                );
                ui.horizontal(|ui| {
                    if ui.small_button("Fill Synthetic Ubiquitin .str").clicked() {
                        self.paste_text_buf = SYNTHETIC_UBIQUITIN_STR.to_string();
                    }
                    if ui.small_button("Load Pasted Text & Solve").clicked() {
                        let buf = self.paste_text_buf.clone();
                        self.load_text_in_memory(&buf, "pasted text");
                    }
                    if ui.small_button("Clear").clicked() {
                        self.paste_text_buf.clear();
                    }
                });
                egui::ScrollArea::vertical()
                    .id_salt("paste_text_scroll")
                    .max_height(180.0)
                    .show(ui, |ui| {
                        ui.add(
                            egui::TextEdit::multiline(&mut self.paste_text_buf)
                                .font(egui::TextStyle::Monospace)
                                .desired_rows(8)
                                .desired_width(f32::INFINITY),
                        );
                    });

                if !self.export_text_buf.is_empty() {
                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("Exported NMR-STAR / TSV:").small().strong());
                        if ui.small_button("Copy All to Clipboard").clicked() {
                            ui.ctx().copy_text(self.export_text_buf.clone());
                        }
                    });
                    egui::ScrollArea::vertical()
                        .id_salt("export_text_scroll")
                        .max_height(180.0)
                        .show(ui, |ui| {
                            let mut preview = self.export_text_buf.as_str();
                            ui.add(
                                egui::TextEdit::multiline(&mut preview)
                                    .font(egui::TextStyle::Monospace)
                                    .desired_rows(8)
                                    .desired_width(f32::INFINITY),
                            );
                        });
                }
            });
        self.show_text_io_modal = open;
    }

    fn is_protein_2d_context(&self) -> bool {
        if self.show_stack1d_panel || self.workspace_mode != WorkspaceMode::ProteinNmr {
            return false;
        }
        if self.show_relax_panel || self.show_dynamics_panel {
            return true;
        }
        let (nx, ny) = self.active_2d_nuclei();
        let page_has_3d = self
            .workbook
            .pages_2d
            .get(self.workbook.active_2d_idx)
            .is_some_and(|p| p.has_3d_strips);
        page_has_3d || (nx == "1H" && ny == "15N") || (nx == "13C" && ny == "15N")
    }

    fn export_active_svg_action(&mut self, ctx: &egui::Context) {
        if self.is_1d_studio_active() && !self.stack1d_panel.traces.is_empty() {
            let svg = export_stack1d_svg(&self.stack1d_panel, 1280, 680);
            self.stack1d_panel.last_exported_svg = Some(svg.clone());
            self.export_text_buf = svg.clone();
            ctx.copy_text(svg);
            self.show_text_io_modal = true;
            self.status_msg =
                "Exported 1D publication vector SVG (copied to clipboard)".to_string();
        } else if self.workspace_mode == WorkspaceMode::OctetChem {
            let svg = render_scene_svg(&self.octet_app.scene, &self.octet_app.style);
            self.export_text_buf = svg.clone();
            ctx.copy_text(svg);
            self.show_text_io_modal = true;
            self.status_msg = "Exported chemical structure vector SVG".to_string();
        } else {
            self.export_results(ctx);
        }
    }

    fn clear_workbook_all(&mut self) {
        self.stack1d_panel = Stack1dState::default();
        self.workbook.clear_2d_pages();
        self.hsqc_spec = None;
        self.hsqc_arc = Arc::new(Vec::new());
        self.overlay_layers.clear();
        self.clear_stale_protein_demo_state(false, false);
        self.workspace_mode = WorkspaceMode::ProteinNmr;
        self.show_stack1d_panel = true;
        self.status_msg = "Cleared workbook — open or drop spectra to begin".to_string();
    }

    fn draw_workspace_mode_header(
        &mut self,
        ui: &mut egui::Ui,
        tablet_dock: Option<&mut TabletActiveDock>,
    ) {
        let ctx = ui.ctx().clone();
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 6.0;

            // Left: Collapsible Spectra Sidebar Toggle | FERRIC Wordmark | Classic Menus
            if chrome_icon_button(
                ui,
                WorkbookIcon::SidebarToggle,
                None,
                self.workbook.show_sidebar,
                "Show / hide left Spectra workbook pages sidebar",
                26.0,
            )
            .clicked()
            {
                self.workbook.show_sidebar = !self.workbook.show_sidebar;
            }

            ui.add_space(2.0);
            nmr_gui::brand_icon::paint_ferric_brand_logo(ui, 22.0)
                .on_hover_text("FERRIC");
            ui.label(
                egui::RichText::new("FERRIC")
                    .size(13.5)
                    .strong()
                    .color(egui::Color32::from_rgb(241, 245, 249)),
            );
            ui.separator();

            // --- Classic Clean Top Dropdown Menus: File | Process | Analyze | View ---
            ui.menu_button("File", |ui| {
                ui.set_min_width(220.0);
                if ui.button("Open Folder…").clicked() {
                    ui.close();
                    self.arm_browser_picker_wakeup();
                    #[cfg(target_arch = "wasm32")]
                    trigger_browser_folder_picker();
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        self.show_native_global_open_bar = true;
                    }
                }
                if ui.button("Open Files / Archive…").clicked() {
                    ui.close();
                    self.arm_browser_picker_wakeup();
                    #[cfg(target_arch = "wasm32")]
                    trigger_browser_files_picker();
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        self.show_native_global_open_bar = true;
                    }
                }
                if ui.button("Experiment Selection Matrix…").clicked() {
                    ui.close();
                    self.ingest_panel.open = true;
                }
                ui.separator();
                ui.menu_button("Samples", |ui| {
                    ui.set_min_width(235.0);
                    if ui.button("1D ¹H Small Molecule (600 MHz)").clicked() {
                        ui.close();
                        self.load_small_molecule_1h_sample();
                    }
                    if ui.button("1D Reaction Kinetics Stack").clicked() {
                        ui.close();
                        self.load_kinetics_stack_sample();
                    }
                    if ui.button("2D Ubiquitin HSQC + 3D HNCACB").clicked() {
                        ui.close();
                        self.load_builtin_demo_suite();
                        let hsqc_idx = self
                            .workbook
                            .pages_2d
                            .iter()
                            .position(|p| p.has_3d_strips)
                            .unwrap_or(0);
                        self.activate_2d_workbook_page(hsqc_idx);
                    }
                    if ui.button("2D DOSY Diffusion Mixture").clicked() {
                        ui.close();
                        if self.dosy_panel.peaks.is_empty() {
                            self.dosy_panel = DosyPanelState::with_synthetic_demo();
                        }
                        self.workspace_mode = WorkspaceMode::Dosy;
                        self.show_stack1d_panel = false;
                    }
                    if ui.button("Metabolomics Cohort (PCA / OPLS-DA)").clicked() {
                        ui.close();
                        if self.metabolomics_panel.samples.is_empty() {
                            self.metabolomics_panel.load_demo_cohort();
                        }
                        self.workspace_mode = WorkspaceMode::Metabolomics;
                        self.show_stack1d_panel = false;
                    }
                });
                ui.separator();
                if ui.button("Paste NMR-STAR / TSV…").clicked() {
                    ui.close();
                    self.show_text_io_modal = true;
                }
                if ui.button("Save Session (.nef)").clicked() {
                    ui.close();
                    self.save_session_nef_action(&ctx);
                }
                if ui.button("Export Vector SVG").clicked() {
                    ui.close();
                    self.export_active_svg_action(&ctx);
                }
                ui.separator();
                if ui.button("Clear Workbook").clicked() {
                    ui.close();
                    self.clear_workbook_all();
                }
            });

            ui.menu_button("Process", |ui| {
                ui.set_min_width(220.0);
                if ui.button("Auto-Phase").clicked() {
                    ui.close();
                    if self.is_1d_studio_active() && !self.stack1d_panel.traces.is_empty() {
                        self.stack1d_panel.auto_phase_active_trace();
                        self.status_msg = self.stack1d_panel.status_message.clone();
                    } else {
                        self.dispatch_workbench_action(&ctx, WorkbenchAction::AutoPhaseDirect1D);
                    }
                }
                if ui.button("Interactive Phase (P)").clicked() {
                    ui.close();
                    if !self.stack1d_panel.traces.is_empty() {
                        self.activate_1d_studio_mode();
                        self.stack1d_panel.show_1d_analysis_drawer = true;
                        self.stack1d_panel.selected_inspector_tab = InspectorTab1d::PhaseBaseline;
                        ctx.input_mut(|i| {
                            i.events.push(egui::Event::Key {
                                key: egui::Key::P,
                                physical_key: Some(egui::Key::P),
                                pressed: true,
                                repeat: false,
                                modifiers: egui::Modifiers::NONE,
                            });
                        });
                    }
                }
                if ui.button("Baseline Correction (Whittaker)").clicked() {
                    ui.close();
                    if self.is_1d_studio_active() && !self.stack1d_panel.traces.is_empty() {
                        self.stack1d_panel.baseline_correct_active_trace();
                        self.status_msg = self.stack1d_panel.status_message.clone();
                    } else {
                        self.dispatch_workbench_action(&ctx, WorkbenchAction::ApplyMedianBaseline);
                    }
                }
                if ui.button("Solvent Reference…").clicked() {
                    ui.close();
                    if self.is_1d_studio_active() && !self.stack1d_panel.traces.is_empty() {
                        self.stack1d_panel.calibrate_solvent_active_trace();
                        self.status_msg = self.stack1d_panel.status_message.clone();
                    } else {
                        self.show_overlay_manager = true;
                    }
                }
                ui.separator();
                if ui.button("Suppress Water Peak").clicked() {
                    ui.close();
                    self.suppress_water_active_spectrum();
                }
                let has_water_undo = if self.is_1d_studio_active() {
                    self.stack1d_panel.water_undo_snapshot.is_some()
                } else {
                    self.water_undo_2d.is_some()
                };
                if ui
                    .add_enabled(has_water_undo, egui::Button::new("Undo Water Suppression"))
                    .clicked()
                {
                    ui.close();
                    self.undo_water_suppression_active();
                }
            });

            ui.menu_button("Analyze", |ui| {
                ui.set_min_width(230.0);
                if ui.button("Auto-Pick Peaks").clicked() {
                    ui.close();
                    if self.is_1d_studio_active() {
                        self.stack1d_panel.refresh_active_trace_peaks_and_multiplets();
                        self.status_msg = format!(
                            "Picked {} 1D peaks",
                            self.stack1d_panel.picked_peaks.len()
                        );
                    } else {
                        self.pick_2d_roots();
                    }
                }
                if ui.button("Multiplet & J-Tree Analysis").clicked() {
                    ui.close();
                    if !self.stack1d_panel.traces.is_empty() {
                        self.activate_1d_studio_mode();
                        self.stack1d_panel.show_multiplet_boxes = true;
                        self.stack1d_panel.show_j_trees = true;
                        self.stack1d_panel.show_1d_analysis_drawer = true;
                        self.stack1d_panel.selected_inspector_tab = InspectorTab1d::Multiplets;
                        self.stack1d_panel.refresh_active_trace_peaks_and_multiplets();
                    }
                }
                if ui.button("Integrate Regions").clicked() {
                    ui.close();
                    if !self.stack1d_panel.traces.is_empty() {
                        self.activate_1d_studio_mode();
                        self.stack1d_panel.tool_mode = ToolMode1d::IntegrateRoi;
                        self.stack1d_panel.show_integral_curves = true;
                    }
                }
                ui.separator();
                if ui.button("Run 3D Backbone Assignment").clicked() {
                    ui.close();
                    let hsqc_idx = self
                        .workbook
                        .pages_2d
                        .iter()
                        .position(|p| p.has_3d_strips)
                        .unwrap_or(self.workbook.active_2d_idx);
                    if !self.workbook.pages_2d.is_empty() {
                        self.activate_2d_workbook_page(hsqc_idx);
                    } else {
                        self.workspace_mode = WorkspaceMode::ProteinNmr;
                        self.show_stack1d_panel = false;
                    }
                    self.run_assignment();
                }
                ui.separator();
                if ui.button("Open Chemical Structure").clicked() {
                    ui.close();
                    self.workspace_mode = WorkspaceMode::OctetChem;
                    self.show_stack1d_panel = false;
                }
                if ui.button("Open DOSY Studio").clicked() {
                    ui.close();
                    if self.dosy_panel.peaks.is_empty() {
                        self.dosy_panel = DosyPanelState::with_synthetic_demo();
                    }
                    self.workspace_mode = WorkspaceMode::Dosy;
                    self.show_stack1d_panel = false;
                }
                if ui.button("Open Metabolomics Studio").clicked() {
                    ui.close();
                    if self.metabolomics_panel.samples.is_empty() {
                        self.metabolomics_panel.load_demo_cohort();
                    }
                    self.workspace_mode = WorkspaceMode::Metabolomics;
                    self.show_stack1d_panel = false;
                }
            });

            ui.menu_button("View", |ui| {
                ui.set_min_width(225.0);
                if ui
                    .radio_value(
                        &mut self.paper_white_canvas,
                        true,
                        "Paper White Canvas",
                    )
                    .clicked()
                {
                    self.sync_overlay_layers_from_workbook();
                    ui.close();
                }
                if ui
                    .radio_value(
                        &mut self.paper_white_canvas,
                        false,
                        "Dark Oscilloscope Canvas",
                    )
                    .clicked()
                {
                    self.sync_overlay_layers_from_workbook();
                    ui.close();
                }
                ui.separator();
                ui.checkbox(&mut self.workbook.show_sidebar, "Spectra Sidebar");
                if self.is_1d_studio_active() {
                    ui.checkbox(&mut self.stack1d_panel.show_peak_labels, "Peak Labels");
                } else {
                    ui.checkbox(&mut self.show_peak_labels, "Peak Labels");
                }
                ui.checkbox(&mut self.stack1d_panel.show_integral_curves, "Integrals");
                let mut mult_j =
                    self.stack1d_panel.show_multiplet_boxes && self.stack1d_panel.show_j_trees;
                if ui
                    .checkbox(&mut mult_j, "Multiplet Boxes & J-Trees")
                    .changed()
                {
                    self.stack1d_panel.show_multiplet_boxes = mult_j;
                    self.stack1d_panel.show_j_trees = mult_j;
                }
                ui.checkbox(&mut self.workbook.two_d.show_params_card, "Parameters Card");
                if self.is_1d_studio_active() {
                    ui.checkbox(&mut self.stack1d_panel.show_1d_analysis_drawer, "Bottom Table");
                } else {
                    ui.checkbox(&mut self.workbook.two_d.show_peak_table, "Bottom Table");
                }
                ui.checkbox(
                    &mut self.workbook.two_d.show_projections,
                    "2D F2/F1 Projections",
                );
                ui.separator();
                ui.checkbox(&mut self.show_mol3d_viewport, "3D Structure Dock");
                ui.checkbox(&mut self.show_strips_dock, "3D Strips Dock");
                ui.checkbox(&mut self.show_workflow_guide, "Workflow Stepper");
            });

            if let Some(dock) = tablet_dock {
                if self.is_protein_2d_context() {
                    ui.separator();
                    ui.selectable_value(dock, TabletActiveDock::StripsRight, "Strips");
                    ui.selectable_value(dock, TabletActiveDock::Mol3dLeft, "3D");
                    ui.selectable_value(dock, TabletActiveDock::Hidden, "Full");
                }
            }

            let now_ms = ctx.input(|i| (i.time.max(0.0) * 1000.0).round() as u64);
            self.chord_dispatcher.draw_status_bar_indicator(ui, now_ms);

            // --- Right Side Segmented Instrument Controls ---
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.spacing_mut().item_spacing.x = 5.0;

                if chrome_icon_button(
                    ui,
                    WorkbookIcon::HelpQuestion,
                    None,
                    self.show_about_modal,
                    "About FERRIC & Keyboard Shortcuts (?)",
                    26.0,
                )
                .clicked()
                {
                    self.show_about_modal = !self.show_about_modal;
                }

                if chrome_icon_button(
                    ui,
                    WorkbookIcon::CommandKey,
                    Some("Cmd+K"),
                    self.command_palette.open,
                    "Open Fuzzy Command Palette (Cmd/Ctrl+K)",
                    26.0,
                )
                .clicked()
                {
                    self.command_palette.open = !self.command_palette.open;
                }

                #[cfg(not(target_arch = "wasm32"))]
                {
                    let bridge_running = self.mcp_bridge.is_running();
                    let mcp_lbl = if bridge_running { "MCP ●" } else { "MCP" };
                    if chrome_icon_button(
                        ui,
                        WorkbookIcon::SlidersTune,
                        Some(mcp_lbl),
                        self.show_mcp_bridge_window || bridge_running,
                        "Local 127.0.0.1 MCP live workbench bridge",
                        26.0,
                    )
                    .clicked()
                    {
                        self.show_mcp_bridge_window = !self.show_mcp_bridge_window;
                    }
                }

                let (theme_icon, theme_lbl) = if self.paper_white_canvas {
                    (WorkbookIcon::SunPaper, "Paper")
                } else {
                    (WorkbookIcon::MoonDark, "Dark")
                };
                if chrome_icon_button(
                    ui,
                    theme_icon,
                    Some(theme_lbl),
                    false,
                    "Toggle spectrum canvas between Publication Paper White and Dark Oscilloscope",
                    26.0,
                )
                .clicked()
                {
                    self.paper_white_canvas = !self.paper_white_canvas;
                    self.sync_overlay_layers_from_workbook();
                }

                if chrome_icon_button(
                    ui,
                    WorkbookIcon::FileDoc,
                    Some("Open Files"),
                    false,
                    "Open .zip/.tar.gz archive, .str/.nef/.tsv, .jdx, .ucsf/.ft2, .pdb/.cif, or .mol/.sdf",
                    26.0,
                )
                .clicked()
                {
                    self.arm_browser_picker_wakeup();
                    #[cfg(target_arch = "wasm32")]
                    trigger_browser_files_picker();
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        self.show_native_global_open_bar = !self.show_native_global_open_bar;
                    }
                }

                if chrome_icon_button(
                    ui,
                    WorkbookIcon::Folder,
                    Some("Open Folder"),
                    false,
                    "Open a Bruker study / expno / pdata folder (1D, 2D, or 3D)",
                    26.0,
                )
                .clicked()
                {
                    self.arm_browser_picker_wakeup();
                    #[cfg(target_arch = "wasm32")]
                    trigger_browser_folder_picker();
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        self.show_native_global_open_bar = !self.show_native_global_open_bar;
                    }
                }
            });
        });

        #[cfg(not(target_arch = "wasm32"))]
        if self.show_native_global_open_bar {
            ui.add_space(3.0);
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Path:").strong());
                let resp = ui.add(
                    egui::TextEdit::singleline(&mut self.path_input)
                        .hint_text("Enter Bruker folder, .zip/.tar.gz, .str/.nef, .jdx, .ucsf/.ft2, .pdb, or .mol path…")
                        .desired_width(460.0),
                );
                let enter = resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                if ui.button("Load").clicked() || enter {
                    self.load_path_auto();
                }
                if ui.button("Paste Text…").clicked() {
                    self.show_text_io_modal = true;
                }
                if ui.small_button("×").clicked() {
                    self.show_native_global_open_bar = false;
                }
            });
        }
    }

    fn draw_top_bar(&mut self, ui: &mut egui::Ui) {
        let mut menu_actions: Vec<WorkbenchAction> = Vec::new();
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing.x = 5.0;
            let on_hsqc = !self.show_relax_panel && !self.show_dynamics_panel;
            let (nx_lbl, ny_lbl) = self.active_2d_nuclei();
            let pill_2d = match (nx_lbl, ny_lbl) {
                ("1H", "15N") => "2D HSQC".to_string(),
                ("13C", "15N") => "2D ¹³C–¹⁵N CON".to_string(),
                (x, y) => format!(
                    "2D {}–{}",
                    Self::pretty_nucleus_unicode(x),
                    Self::pretty_nucleus_unicode(y)
                ),
            };
            if chrome_icon_button(
                ui,
                WorkbookIcon::Contour2D,
                Some(&pill_2d),
                on_hsqc,
                "2D multi-layer WebGPU contour viewport",
                25.0,
            )
            .clicked()
            {
                self.show_relax_panel = false;
                self.show_dynamics_panel = false;
                self.show_stack1d_panel = false;
            }
            if chrome_icon_button(
                ui,
                WorkbookIcon::RelaxDynamics,
                Some("Relaxation S²"),
                self.show_relax_panel,
                "R1, R2, hetNOE & Lipari-Szabo model-free S² dynamics",
                25.0,
            )
            .clicked()
            {
                self.show_relax_panel = true;
                self.show_dynamics_panel = false;
                self.show_stack1d_panel = false;
            }
            if chrome_icon_button(
                ui,
                WorkbookIcon::SlidersTune,
                Some("CEST / CPMG"),
                self.show_dynamics_panel,
                "CEST minor-state saturation, CPMG dispersion & HSQC titration Kd",
                25.0,
            )
            .clicked()
            {
                self.show_dynamics_panel = true;
                self.show_relax_panel = false;
                self.show_stack1d_panel = false;
            }

            ui.separator();
            if chrome_icon_button(
                ui,
                WorkbookIcon::PeakPick,
                Some("Pick Peaks"),
                false,
                "Pick 2D root cross-peaks on the active spectrum",
                25.0,
            )
            .clicked()
            {
                self.pick_2d_roots();
            }
            if self.regime_kind != GuiRegimeKind::Idp13C {
                egui::ComboBox::from_id_salt("gui_solver_kind")
                    .selected_text(format!("Solver: {}", self.solver_kind.label()))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.solver_kind,
                            GuiSolverKind::EvoGraph,
                            "Evo-Graph (CPU f64, default)",
                        );
                        ui.selectable_value(
                            &mut self.solver_kind,
                            GuiSolverKind::Standard,
                            "Standard SA (GPU / CPU)",
                        );
                        ui.selectable_value(
                            &mut self.solver_kind,
                            GuiSolverKind::BayesSa,
                            "Bayesian SA (Cv Adaptive)",
                        );
                    });
            }
            let solve_btn = egui::Button::new(
                egui::RichText::new("Run Assignment")
                    .size(11.5)
                    .strong()
                    .color(egui::Color32::from_rgb(12, 24, 18)),
            )
            .corner_radius(4.0)
            .fill(egui::Color32::from_rgb(82, 224, 152));
            if ui.add_sized([108.0, 25.0], solve_btn).clicked() {
                self.run_assignment();
            }
            if chrome_icon_button(
                ui,
                WorkbookIcon::DownloadSave,
                Some("Export"),
                false,
                "Export NMR-STAR v3 & shift TSV",
                25.0,
            )
            .clicked()
            {
                let ctx = ui.ctx().clone();
                self.export_results(&ctx);
            }

            ui.separator();
            if chrome_icon_button(
                ui,
                WorkbookIcon::Cube3D,
                Some("3D PDB"),
                self.show_mol3d_viewport,
                "Show / hide 3D Biomolecular Structure & Putty dock",
                25.0,
            )
            .clicked()
            {
                self.show_mol3d_viewport = !self.show_mol3d_viewport;
            }
            if chrome_icon_button(
                ui,
                WorkbookIcon::Strips3D,
                Some("3D Strips"),
                self.show_strips_dock,
                "Show / hide right 3D Sequential Strip Inspector dock",
                25.0,
            )
            .clicked()
            {
                self.show_strips_dock = !self.show_strips_dock;
            }
            if on_hsqc
                && !self.sequence.is_empty()
                && chrome_icon_button(
                    ui,
                    WorkbookIcon::Trace1D,
                    Some("Sequence"),
                    self.show_sequence_ribbon,
                    "Show / hide sequence assignment confidence ribbon",
                    25.0,
                )
                .clicked()
            {
                self.show_sequence_ribbon = !self.show_sequence_ribbon;
            }
            if chrome_icon_button(
                ui,
                WorkbookIcon::SlidersTune,
                Some("Tuning"),
                self.show_advanced_protein_bar,
                "Expand Regime, SA replicas/steps, Picker SNR & Simulator",
                25.0,
            )
            .clicked()
            {
                self.show_advanced_protein_bar = !self.show_advanced_protein_bar;
            }
        });

        if self.show_advanced_protein_bar {
            ui.add_space(2.0);
            let readiness = self.workspace_readiness();
            menu_actions = draw_top_menu_bar(ui, &readiness);
            ui.horizontal_wrapped(|ui| {
                if ui.small_button("Paste .str / .nef / .tsv").clicked() {
                    self.show_text_io_modal = !self.show_text_io_modal;
                }
                egui::ComboBox::from_id_salt("gui_regime_kind")
                    .selected_text(format!("Regime: {}", self.regime_kind.label()))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.regime_kind,
                            GuiRegimeKind::Globular,
                            "Globular (¹H–¹⁵N HSQC + 3D)",
                        );
                        ui.selectable_value(
                            &mut self.regime_kind,
                            GuiRegimeKind::Idp13C,
                            "13C-IDP (CON / CACO / CANCO + Pro)",
                        );
                        ui.selectable_value(
                            &mut self.regime_kind,
                            GuiRegimeKind::Paramagnetic,
                            "Paramagnetic (PRE Blind-Zone + Metal)",
                        );
                    });
                if self.regime_kind == GuiRegimeKind::Paramagnetic {
                    ui.label(egui::RichText::new("Ligands:").small());
                    ui.add(
                        egui::TextEdit::singleline(&mut self.metal_ligands_input)
                            .hint_text("auto motif or CYS26,HIS41")
                            .desired_width(135.0),
                    );
                } else if self.regime_kind == GuiRegimeKind::Idp13C {
                    if ui
                        .add(
                            egui::DragValue::new(&mut self.idp_temp_k)
                                .speed(0.5)
                                .range(265.0..=335.0)
                                .prefix("T ")
                                .suffix(" K"),
                        )
                        .changed()
                    {
                        self.refresh_propensity_report();
                    }
                    ui.label(egui::RichText::new("pH:").small());
                    if ui
                        .add(egui::TextEdit::singleline(&mut self.idp_ph_input).desired_width(38.0))
                        .changed()
                    {
                        self.refresh_propensity_report();
                    }
                }
                #[cfg(not(target_arch = "wasm32"))]
                {
                    let gpu_supported = (self.regime_kind == GuiRegimeKind::Globular
                        && matches!(
                            self.solver_kind,
                            GuiSolverKind::Standard | GuiSolverKind::BayesSa
                        ))
                        || self.regime_kind == GuiRegimeKind::Idp13C;
                    ui.add_enabled(
                        gpu_supported,
                        egui::Checkbox::new(&mut self.use_gpu, "WebGPU Compute"),
                    );
                }
                ui.add(egui::Slider::new(&mut self.replicas, 8..=256).text("Replicas"));
                ui.add(egui::Slider::new(&mut self.steps_per_sys, 400..=4000).text("Steps/SS"));
                if ui
                    .add(
                        egui::Slider::new(&mut self.assign_cost, 0.0..=4.0)
                            .text("Assign cost")
                            .fixed_decimals(2),
                    )
                    .changed()
                    && !self.systems.is_empty()
                    && !self.sequence.is_empty()
                {
                    let scfg = ScoreConfig {
                        assign_cost: self.assign_cost,
                        ..ScoreConfig::default()
                    };
                    self.u_mat = Some(unary_matrix(
                        &self.systems,
                        &self.sequence,
                        &self.priors,
                        &scfg,
                    ));
                }
                ui.separator();
                ui.checkbox(&mut self.use_neural_picker, "Neural CNN");
                ui.add(egui::Slider::new(&mut self.picker_snr, 3.0..=12.0).text("Pick SNR"));
                ui.add(
                    egui::Slider::new(&mut self.sim_missing, 0.0..=0.25)
                        .text("Sim Missing")
                        .fixed_decimals(2),
                );
                ui.add(
                    egui::Slider::new(&mut self.sim_spurious, 0.0..=0.20)
                        .text("Sim Spurious")
                        .fixed_decimals(2),
                );
                if ui.small_button("Re-simulate & Solve").clicked() {
                    self.sim_seed = self.sim_seed.wrapping_add(1);
                    if let Some(r) = self.reference.clone() {
                        self.load_from_reference(r);
                        self.run_assignment();
                    }
                }
            });
        }
        let ctx = ui.ctx().clone();
        for act in menu_actions {
            self.dispatch_workbench_action(&ctx, act);
        }
    }

    fn draw_sequence_ribbon(&mut self, ui: &mut egui::Ui) {
        if self.sequence.is_empty() {
            return;
        }
        let mut res_to_ss: BTreeMap<usize, (usize, f64)> = BTreeMap::new();
        if let Some(res) = &self.assign_res {
            for (s, &k) in res.position.iter().enumerate() {
                if k >= 0 {
                    res_to_ss.insert(k as usize, (s, res.confidence[s]));
                }
            }
        }

        egui::ScrollArea::horizontal()
            .id_salt("seq_ribbon_scroll")
            .max_height(52.0)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 3.0;
                    if chrome_icon_button(
                        ui,
                        WorkbookIcon::StructureChem,
                        Some("Sidechain 2D"),
                        self.show_sidechain_inspector,
                        "Toggle live 2D IUPAC covalent sidechain Z-score inspector",
                        28.0,
                    )
                    .clicked()
                    {
                        self.show_sidechain_inspector = !self.show_sidechain_inspector;
                        if self.show_sidechain_inspector && self.selected_res.is_none() {
                            self.selected_res = Some(0);
                        }
                    }
                    if chrome_icon_button(
                        ui,
                        WorkbookIcon::Trace1D,
                        Some("SSP / δ2D"),
                        self.show_propensity_panel,
                        "Toggle per-residue Marsh SSP & Camilloni 4-state δ2D ensemble track",
                        28.0,
                    )
                    .clicked()
                    {
                        self.show_propensity_panel = !self.show_propensity_panel;
                        if self.show_propensity_panel {
                            self.refresh_propensity_report();
                            if self.selected_res.is_none() {
                                self.selected_res = Some(0);
                            }
                        }
                    }
                    if chrome_icon_button(
                        ui,
                        WorkbookIcon::Cube3D,
                        Some("3D Structure"),
                        self.show_mol3d_viewport,
                        "Toggle dockable WebGPU 3D biomolecular structure & putty viewport",
                        28.0,
                    )
                    .clicked()
                    {
                        self.show_mol3d_viewport = !self.show_mol3d_viewport;
                    }
                    ui.add_space(4.0);
                    for (k, three) in self.sequence.iter().enumerate() {
                        let one = three_to_one(three);
                        let seq_id = self.seq_ids.get(k).copied().unwrap_or((k + 1) as i32);
                        let is_pro_or_nterm = k == 0 || three == "PRO";
                        let assigned = res_to_ss.get(&k).copied();
                        let direct_conf = assigned
                            .map(|(_, c)| c)
                            .or_else(|| self.pred_conf.get(&k).copied());
                        let is_selected = self.selected_res == Some(k);

                        let fill = if let Some(conf) = direct_conf {
                            if conf >= 0.80 {
                                egui::Color32::from_rgb(28, 125, 82)
                            } else if conf >= 0.50 {
                                egui::Color32::from_rgb(148, 112, 28)
                            } else {
                                egui::Color32::from_rgb(145, 68, 42)
                            }
                        } else if is_pro_or_nterm {
                            egui::Color32::from_rgb(38, 43, 54)
                        } else {
                            egui::Color32::from_rgb(48, 56, 74)
                        };

                        let is_locked = self.session.state.is_residue_locked(k);
                        let stroke = if is_selected {
                            egui::Stroke::new(2.0, egui::Color32::from_rgb(120, 235, 255))
                        } else if is_locked {
                            egui::Stroke::new(1.8, egui::Color32::from_rgb(255, 215, 90))
                        } else {
                            egui::Stroke::new(1.0, egui::Color32::from_rgb(65, 76, 98))
                        };

                        let label = if is_locked {
                            format!("*{one}{seq_id}")
                        } else {
                            format!("{one}{seq_id}")
                        };
                        let btn = egui::Button::new(
                            egui::RichText::new(label)
                                .monospace()
                                .size(11.5)
                                .color(egui::Color32::WHITE),
                        )
                        .fill(fill)
                        .stroke(stroke)
                        .min_size(egui::vec2(34.0, 28.0));

                        let resp = ui.add(btn);
                        if resp.clicked() {
                            self.selected_res = Some(k);
                            self.sidechain_ribbon_state.selected_residue = k;
                            self.assign_target_res = k;
                            if let Some((ss_idx, _)) = assigned {
                                self.selected_ss = Some(ss_idx);
                                if let Some(ss) = self.systems.get(ss_idx) {
                                    self.crosshair.snap_to(ss.h, ss.n, None);
                                }
                            }
                            self.mol3d_panel.highlight_sequence_residue(Some(k), true);
                        }
                        resp.on_hover_ui(|ui| {
                            ui.label(egui::RichText::new(format!("{three} {seq_id}")).strong());
                            if let Some((ss_idx, conf)) = assigned {
                                ui.label(format!(
                                    "Spin system #{ss_idx} (conf {:.0}%{})",
                                    conf * 100.0,
                                    if is_locked { ", LOCKED" } else { "" }
                                ));
                            } else if let Some(conf) = direct_conf {
                                ui.label(format!(
                                    "Assigned ({}) — conf {:.0}%{}",
                                    self.regime_kind.label(),
                                    conf * 100.0,
                                    if is_locked { ", LOCKED" } else { "" }
                                ));
                            } else if is_pro_or_nterm {
                                ui.label("No amide H root (N-term / Proline)");
                            } else {
                                ui.label("Unassigned");
                            }
                            if let Some(pred) = &self.pred_ref {
                                for atom in ["H", "N", "CA", "CB", "C"] {
                                    if let Some(&v) = pred.shifts.get(&(k, atom.to_string())) {
                                        ui.monospace(format!("  {atom:2}: {v:7.3} ppm"));
                                    }
                                }
                            }
                        });
                    }
                });
            });
    }

    fn draw_hsqc_canvas(&mut self, ui: &mut egui::Ui) -> egui::Rect {
        let is_compact_mobile = ui.ctx().content_rect().width() < 680.0;
        let (nuc_x, nuc_y) = self.active_2d_nuclei();
        let pretty_x = Self::pretty_nucleus_unicode(nuc_x);
        let pretty_y = Self::pretty_nucleus_unicode(nuc_y);
        let theme = CanvasTheme2D::for_mode(self.paper_white_canvas);
        let base_lo = (self.contour_noise_ref * 0.15).max(1e-6);
        let base_hi = (self.contour_noise_ref * 400.0)
            .max(self.contour_base * 6.0)
            .max(60.0);

        if nuc_x == "13C"
            && nuc_y == "15N"
            && !self.peaks.iter().any(|p| p.c.is_none())
            && self
                .c13_picked_dataset
                .as_ref()
                .is_some_and(|ds| !ds.con_roots.is_empty())
        {
            self.sync_con_roots_to_peaks();
        }

        // `Del`/`Backspace` delete selected peak and `Escape` cancels drag boxes (two-letter chords are handled globally by ChordDispatcher)
        let mut key_delete_peak: Option<usize> = None;
        if !is_compact_mobile && !ui.ctx().egui_wants_keyboard_input() {
            ui.input(|i| {
                let plain = !i.modifiers.command && !i.modifiers.ctrl && !i.modifiers.alt;
                if plain
                    && (i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace))
                {
                    key_delete_peak = self.workbook.two_d.selected_peak_idx;
                }
                if i.key_pressed(egui::Key::Escape) {
                    self.box_zoom_2d_anchor = None;
                    self.box_zoom_2d_mode = false;
                    self.workbook.two_d.drag_box_start_pos = None;
                    self.workbook.two_d.pick_box_start_pos = None;
                    self.workbook.two_d.dragging_peak_idx = None;
                }
            });
        }
        if let Some(di) = key_delete_peak {
            self.delete_2d_peak(di);
        }

        if is_compact_mobile {
            egui::ScrollArea::horizontal()
                .id_salt("hsqc_mobile_toolbar_scroll")
                .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysHidden)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 4.0;
                        ui.spacing_mut().interact_size.y = MOBILE_TOUCH_MIN_HEIGHT;
                        if ui
                            .add(
                                egui::Button::new("Levels")
                                    .selected(self.show_contour_modal)
                                    .min_size(egui::vec2(58.0, MOBILE_TOUCH_MIN_HEIGHT)),
                            )
                            .clicked()
                        {
                            self.show_contour_modal = !self.show_contour_modal;
                        }
                        if ui
                            .add(
                                egui::Button::new("−Floor")
                                    .min_size(egui::vec2(52.0, MOBILE_TOUCH_MIN_HEIGHT)),
                            )
                            .clicked()
                        {
                            self.contour_base =
                                (self.contour_base / CONTOUR_SCROLL_STEP_RATIO).max(1e-6);
                        }
                        if ui
                            .add(
                                egui::Button::new("+Floor")
                                    .min_size(egui::vec2(52.0, MOBILE_TOUCH_MIN_HEIGHT)),
                            )
                            .clicked()
                        {
                            self.contour_base =
                                (self.contour_base * CONTOUR_SCROLL_STEP_RATIO).min(base_hi);
                        }
                        if ui
                            .add(
                                egui::Button::new("Pick")
                                    .min_size(egui::vec2(46.0, MOBILE_TOUCH_MIN_HEIGHT)),
                            )
                            .clicked()
                        {
                            self.pick_2d_roots();
                        }
                        if ui
                            .add(
                                egui::Button::new("Water")
                                    .selected(self.water_undo_2d.is_some())
                                    .min_size(egui::vec2(50.0, MOBILE_TOUCH_MIN_HEIGHT)),
                            )
                            .clicked()
                        {
                            self.suppress_water_active_spectrum();
                        }
                        let lbl_btn = egui::Button::new("Labels")
                            .selected(self.show_peak_labels)
                            .min_size(egui::vec2(54.0, MOBILE_TOUCH_MIN_HEIGHT));
                        if ui.add(lbl_btn).clicked() {
                            self.show_peak_labels = !self.show_peak_labels;
                        }
                        let neg_btn = egui::Button::new("Neg")
                            .selected(self.show_negative)
                            .min_size(egui::vec2(44.0, MOBILE_TOUCH_MIN_HEIGHT));
                        if ui.add(neg_btn).clicked() {
                            self.show_negative = !self.show_negative;
                        }
                        if ui
                            .add(
                                egui::Button::new(format!(
                                    "Layers ({})",
                                    1 + self.overlay_layers.len()
                                ))
                                .min_size(egui::vec2(66.0, MOBILE_TOUCH_MIN_HEIGHT)),
                            )
                            .clicked()
                        {
                            self.show_overlay_manager = !self.show_overlay_manager;
                        }
                        let lock_txt = if self.crosshair.locked {
                            "Locked"
                        } else {
                            "XHair"
                        };
                        if ui
                            .add(
                                egui::Button::new(lock_txt)
                                    .selected(self.crosshair.locked)
                                    .min_size(egui::vec2(54.0, MOBILE_TOUCH_MIN_HEIGHT)),
                            )
                            .clicked()
                        {
                            self.crosshair.toggle_lock();
                        }
                        if ui
                            .add(
                                egui::Button::new("Full")
                                    .min_size(egui::vec2(44.0, MOBILE_TOUCH_MIN_HEIGHT)),
                            )
                            .clicked()
                        {
                            self.reset_view_to_spec();
                        }
                    });
                });
        }

        let outer_avail = ui.available_size();
        let table_reserve = if is_compact_mobile {
            2.0
        } else if self.workbook.two_d.show_peak_table {
            226.0_f32.min((outer_avail.y - 180.0).max(28.0))
        } else {
            28.0
        };
        let outer_size = egui::vec2(
            outer_avail.x.max(160.0),
            (outer_avail.y - table_reserve).max(160.0),
        );
        let (outer_rect, _) = ui.allocate_exact_size(outer_size, egui::Sense::hover());

        // 1. Slim `38 px` Left Vertical Tool Rail on Desktop & Tablet
        let stage_rect = if is_compact_mobile {
            outer_rect
        } else {
            let rail_rect = egui::Rect::from_min_max(
                outer_rect.min,
                egui::pos2(
                    outer_rect.left() + CANVAS_TOOL_RAIL_WIDTH,
                    outer_rect.bottom(),
                ),
            );
            ui.painter()
                .rect_filled(rail_rect, 3.0, egui::Color32::from_rgb(15, 20, 29));
            ui.painter().rect_stroke(
                rail_rect,
                3.0,
                egui::Stroke::new(1.0, egui::Color32::from_rgb(36, 46, 62)),
                egui::StrokeKind::Inside,
            );

            let mut rail_ui = ui.new_child(
                egui::UiBuilder::new()
                    .max_rect(rail_rect.shrink2(egui::vec2(4.0, 6.0)))
                    .layout(egui::Layout::top_down_justified(egui::Align::Center)),
            );
            rail_ui.spacing_mut().item_spacing.y = 4.0;

            let tm = self.workbook.two_d.tool_mode;
            if rail_tool_button(
                &mut rail_ui,
                WorkbookIcon::ZoomBox,
                tm == ToolMode2d::ZoomBox,
                "Zoom Box (zm) · Prev Zoom (zp) · Full (zf): Drag or 2-click to zoom (Right-drag pans)",
            )
            .clicked()
            {
                self.workbook.two_d.tool_mode = ToolMode2d::ZoomBox;
            }
            if rail_tool_button(
                &mut rail_ui,
                WorkbookIcon::PanMove,
                tm == ToolMode2d::Pan,
                "Pan 2D (mv): Drag anywhere to translate (F2, F1) chemical-shift window",
            )
            .clicked()
            {
                self.workbook.two_d.tool_mode = ToolMode2d::Pan;
                self.box_zoom_2d_anchor = None;
                self.workbook.two_d.drag_box_start_pos = None;
            }
            if rail_tool_button(
                &mut rail_ui,
                WorkbookIcon::CrosshairSlice,
                tm == ToolMode2d::CrosshairSlice,
                "Crosshair & Live 1D Slice (xh): Inspect orthogonal F2 row & F1 column traces at cursor",
            )
            .clicked()
            {
                self.workbook.two_d.tool_mode = ToolMode2d::CrosshairSlice;
                self.workbook.two_d.show_projections = true;
            }
            if rail_tool_button(
                &mut rail_ui,
                WorkbookIcon::PeakPick,
                tm == ToolMode2d::PickPeaks,
                "Pick 2D Peaks (kr): Click to add, drag box to pick; click again to Auto-Pick all",
            )
            .clicked()
            {
                if tm == ToolMode2d::PickPeaks {
                    self.pick_2d_roots();
                } else {
                    self.workbook.two_d.tool_mode = ToolMode2d::PickPeaks;
                }
            }

            rail_ui.separator();

            if rail_tool_button(
                &mut rail_ui,
                WorkbookIcon::FullSweep,
                false,
                "Full 2D Sweep (zf / ze / oz): Reset (F2, F1) viewport to full spectral bounds",
            )
            .clicked()
            {
                let cur = [
                    self.view_h_left,
                    self.view_h_right,
                    self.view_n_top,
                    self.view_n_bottom,
                ];
                self.workbook.two_d.push_zoom(cur);
                self.reset_view_to_spec();
            }
            if rail_tool_button(
                &mut rail_ui,
                WorkbookIcon::UndoBack,
                !self.workbook.two_d.zoom_history.is_empty(),
                "Zoom Previous (zp): Restore previous (F2, F1) zoom window",
            )
            .clicked()
            {
                if let Some(prev) = self.workbook.two_d.pop_zoom() {
                    self.view_h_left = prev[0];
                    self.view_h_right = prev[1];
                    self.view_n_top = prev[2];
                    self.view_n_bottom = prev[3];
                }
            }
            if rail_tool_button(
                &mut rail_ui,
                WorkbookIcon::Plus,
                false,
                "Lower Contour Floor (cb / -): Reveal weaker cross-peaks (or scroll wheel up)",
            )
            .clicked()
            {
                self.contour_base = (self.contour_base / CONTOUR_SCROLL_STEP_RATIO).max(base_lo);
            }
            if rail_tool_button(
                &mut rail_ui,
                WorkbookIcon::Minus,
                false,
                "Raise Contour Floor (ci / +): Suppress baseline noise (or scroll wheel down)",
            )
            .clicked()
            {
                self.contour_base = (self.contour_base * CONTOUR_SCROLL_STEP_RATIO).min(base_hi);
            }

            rail_ui.separator();

            if rail_tool_button(
                &mut rail_ui,
                WorkbookIcon::Projections2D,
                self.workbook.two_d.show_projections,
                "Toggle aligned Top F2 & Left F1 1D marginal projections (v1 / v2)",
            )
            .clicked()
            {
                self.workbook.two_d.show_projections = !self.workbook.two_d.show_projections;
            }
            if rail_tool_button(
                &mut rail_ui,
                WorkbookIcon::ParamsCard,
                self.workbook.two_d.show_params_card,
                "Toggle floating 2D Parameters & Active Overlay Layers card",
            )
            .clicked()
            {
                self.workbook.two_d.show_params_card = !self.workbook.two_d.show_params_card;
            }
            if rail_tool_button(
                &mut rail_ui,
                WorkbookIcon::SlidersTune,
                self.show_contour_modal,
                "Contour Levels & Threshold Dialog (ct / vt)",
            )
            .clicked()
            {
                self.show_contour_modal = !self.show_contour_modal;
            }
            if rail_tool_button(
                &mut rail_ui,
                WorkbookIcon::TableGrid,
                self.workbook.two_d.show_peak_table,
                "Toggle Bottom 2D Peak List Table (lt)",
            )
            .clicked()
            {
                self.workbook.two_d.show_peak_table = !self.workbook.two_d.show_peak_table;
            }
            if rail_tool_button(
                &mut rail_ui,
                WorkbookIcon::WaterSuppress,
                self.water_undo_2d.is_some(),
                "Suppress Water Peak (ws) / Undo Water Suppression (wu)",
            )
            .clicked()
            {
                self.suppress_water_active_spectrum();
            }

            egui::Rect::from_min_max(
                egui::pos2(rail_rect.right() + 4.0, outer_rect.top()),
                outer_rect.max,
            )
        };

        // 2. Partition `stage_rect` into aligned Top F2 (`64 px`), Left F1 (`64 px`), Plot, and Rulers
        let show_proj = !is_compact_mobile
            && self.workbook.two_d.show_projections
            && stage_rect.width() >= 280.0
            && stage_rect.height() >= 240.0;
        let ruler_bottom_h = if is_compact_mobile { 0.0 } else { 22.0 };
        let ruler_right_w = if is_compact_mobile { 0.0 } else { 44.0 };
        let gap = if show_proj { 4.0 } else { 0.0 };
        let marg_sz = if show_proj { MARGINAL_2D_STRIP_PX } else { 0.0 };

        let plot_left = stage_rect.left() + marg_sz + gap;
        let plot_right = (stage_rect.right() - ruler_right_w).max(plot_left + 80.0);
        let plot_top = stage_rect.top() + marg_sz + gap;
        let plot_bottom = (stage_rect.bottom() - ruler_bottom_h).max(plot_top + 80.0);

        let rect = egui::Rect::from_min_max(
            egui::pos2(plot_left, plot_top),
            egui::pos2(plot_right, plot_bottom),
        );
        let f2_rect = show_proj.then(|| {
            egui::Rect::from_min_max(
                egui::pos2(rect.left(), stage_rect.top()),
                egui::pos2(rect.right(), stage_rect.top() + MARGINAL_2D_STRIP_PX),
            )
        });
        let f1_rect = show_proj.then(|| {
            egui::Rect::from_min_max(
                egui::pos2(stage_rect.left(), rect.top()),
                egui::pos2(stage_rect.left() + MARGINAL_2D_STRIP_PX, rect.bottom()),
            )
        });
        let corner_rect = show_proj.then(|| {
            egui::Rect::from_min_max(
                stage_rect.min,
                egui::pos2(
                    stage_rect.left() + MARGINAL_2D_STRIP_PX,
                    stage_rect.top() + MARGINAL_2D_STRIP_PX,
                ),
            )
        });

        // Interact on main 2D plot rect
        let response = ui.interact(
            rect,
            ui.scope_id().with("nmr_2d_contour_plot_sense"),
            egui::Sense::click_and_drag(),
        );
        let limits = self.view_limits_for_active_2d();

        let apply_box_zoom = |app: &mut Self, p0: egui::Pos2, p1: egui::Pos2| -> bool {
            let x0 = p0.x.min(p1.x).clamp(rect.left(), rect.right());
            let x1 = p0.x.max(p1.x).clamp(rect.left(), rect.right());
            let y0 = p0.y.min(p1.y).clamp(rect.top(), rect.bottom());
            let y1 = p0.y.max(p1.y).clamp(rect.top(), rect.bottom());
            if (x1 - x0) >= 6.0 && (y1 - y0) >= 6.0 {
                let prev = [
                    app.view_h_left,
                    app.view_h_right,
                    app.view_n_top,
                    app.view_n_bottom,
                ];
                let u0 = (x0 - rect.left()) / rect.width().max(1.0);
                let u1 = (x1 - rect.left()) / rect.width().max(1.0);
                let v0 = (y0 - rect.top()) / rect.height().max(1.0);
                let v1 = (y1 - rect.top()) / rect.height().max(1.0);
                let nh_left = app.view_h_left + u0 * (app.view_h_right - app.view_h_left);
                let nh_right = app.view_h_left + u1 * (app.view_h_right - app.view_h_left);
                let nn_top = app.view_n_top + v0 * (app.view_n_bottom - app.view_n_top);
                let nn_bottom = app.view_n_top + v1 * (app.view_n_bottom - app.view_n_top);
                let clamped = clamp_view_ppm_2d([nh_left, nh_right, nn_top, nn_bottom], limits);
                app.workbook.two_d.push_zoom(prev);
                app.view_h_left = clamped[0];
                app.view_h_right = clamped[1];
                app.view_n_top = clamped[2];
                app.view_n_bottom = clamped[3];
                true
            } else {
                false
            }
        };

        // Two-finger multi-touch pinch-to-zoom & translation pan around gesture center
        let multi_touch_active = if let Some(mt) = ui.ctx().input(|i| i.multi_touch()) {
            if mt.num_touches >= 2
                && (rect.contains(mt.start_pos)
                    || rect.contains(mt.center_pos)
                    || response.hovered()
                    || response.is_pointer_button_down_on())
            {
                self.suppress_hsqc_pan_until_release = true;
                let updated = apply_pinch_zoom_and_pan_2d(
                    [
                        self.view_h_left,
                        self.view_h_right,
                        self.view_n_top,
                        self.view_n_bottom,
                    ],
                    rect,
                    mt.center_pos,
                    mt.zoom_delta,
                    mt.translation_delta,
                    limits,
                );
                self.view_h_left = updated[0];
                self.view_h_right = updated[1];
                self.view_n_top = updated[2];
                self.view_n_bottom = updated[3];
                true
            } else {
                false
            }
        } else {
            false
        };

        let tool_mode = self.workbook.two_d.tool_mode;
        let space_down = ui.input(|i| i.key_down(egui::Key::Space));
        let shift_down = ui.input(|i| i.modifiers.shift);
        let hit_radius_px = if is_compact_mobile { 22.0 } else { 12.0 };
        let peak_edit_bar_rect = (!is_compact_mobile
            && rect.width() >= 360.0
            && rect.height() >= 160.0
            && self
                .workbook
                .two_d
                .selected_peak_idx
                .is_some_and(|i| i < self.peaks.len() && self.peaks[i].c.is_none()))
        .then(|| {
            let bw = (rect.width() - 16.0).min(548.0);
            egui::Rect::from_min_size(
                egui::pos2(rect.left() + 8.0, rect.bottom() - 36.0),
                egui::vec2(bw, 28.0),
            )
        });
        let in_peak_edit_bar = |p: egui::Pos2| peak_edit_bar_rect.is_some_and(|r| r.contains(p));

        let screen_to_ppm_cur = |p: egui::Pos2, vh: [f32; 4]| -> (f64, f64) {
            let u = ((p.x - rect.left()) / rect.width().max(1.0)).clamp(0.0, 1.0);
            let v = ((p.y - rect.top()) / rect.height().max(1.0)).clamp(0.0, 1.0);
            (
                (vh[0] + u * (vh[1] - vh[0])) as f64,
                (vh[2] + v * (vh[3] - vh[2])) as f64,
            )
        };
        let nearest_2d_peak =
            |peaks: &[Peak], p: egui::Pos2, vh: [f32; 4], rad: f32| -> Option<(usize, f32)> {
                let mut best: Option<(usize, f32)> = None;
                for (idx, pk) in peaks.iter().enumerate() {
                    if pk.c.is_some() {
                        continue;
                    }
                    let pt = SyncCrosshair::ppm_to_screen_2d(vh, rect, pk.h, pk.n);
                    if !rect.contains(pt) {
                        continue;
                    }
                    let d = pt.distance(p);
                    if d <= rad && best.is_none_or(|(_, bd)| d < bd) {
                        best = Some((idx, d));
                    }
                }
                best
            };

        // Track stationary right-click vs right-drag pan
        if ui.input(|i| i.pointer.button_pressed(egui::PointerButton::Secondary)) {
            if let Some(p) = ui
                .input(|i| i.pointer.interact_pos())
                .or_else(|| response.hover_pos())
                .filter(|p| rect.contains(*p))
            {
                self.secondary_drag_accum_2d = 0.0;
                let (ch, cn) = screen_to_ppm_cur(
                    p,
                    [
                        self.view_h_left,
                        self.view_h_right,
                        self.view_n_top,
                        self.view_n_bottom,
                    ],
                );
                self.context_menu_2d_ppm = Some([ch, cn]);
            }
        }
        if response.dragged_by(egui::PointerButton::Secondary) {
            self.secondary_drag_accum_2d += response.drag_delta().length();
        }

        // Start dragging an existing 2D peak marker if primary drag originates on a peak
        if !multi_touch_active
            && !space_down
            && !shift_down
            && self.workbook.two_d.dragging_peak_idx.is_none()
            && self.workbook.two_d.drag_box_start_pos.is_none()
            && self.workbook.two_d.pick_box_start_pos.is_none()
            && response.dragged_by(egui::PointerButton::Primary)
        {
            if let Some(origin) = ui.input(|i| i.pointer.press_origin()) {
                if rect.contains(origin) && !in_peak_edit_bar(origin) {
                    let cur_v = [
                        self.view_h_left,
                        self.view_h_right,
                        self.view_n_top,
                        self.view_n_bottom,
                    ];
                    if let Some((pk_idx, _)) =
                        nearest_2d_peak(&self.peaks, origin, cur_v, hit_radius_px)
                    {
                        self.workbook.two_d.dragging_peak_idx = Some(pk_idx);
                        self.workbook.two_d.selected_peak_idx = Some(pk_idx);
                        self.box_zoom_2d_anchor = None;
                    }
                }
            }
        }

        if let Some(drag_idx) = self.workbook.two_d.dragging_peak_idx {
            if let Some(cur_pos) = response
                .interact_pointer_pos()
                .or_else(|| ui.input(|i| i.pointer.latest_pos()))
            {
                let (nh, nn) = screen_to_ppm_cur(
                    cur_pos,
                    [
                        self.view_h_left,
                        self.view_h_right,
                        self.view_n_top,
                        self.view_n_bottom,
                    ],
                );
                self.move_2d_peak_live(drag_idx, nh, nn, false);
            }
        }

        let peak_drag_active = self.workbook.two_d.dragging_peak_idx.is_some();
        let primary_pans =
            !peak_drag_active && (is_compact_mobile || tool_mode == ToolMode2d::Pan || space_down);
        let pan_dragged = !multi_touch_active
            && !peak_drag_active
            && (response.dragged_by(egui::PointerButton::Secondary)
                || response.dragged_by(egui::PointerButton::Middle)
                || (primary_pans
                    && response.dragged_by(egui::PointerButton::Primary)
                    && !ui
                        .input(|i| i.pointer.press_origin())
                        .is_some_and(in_peak_edit_bar)));

        if pan_dragged && !self.suppress_hsqc_pan_until_release {
            let delta = response.drag_delta();
            let span_h = self.view_h_right - self.view_h_left;
            let span_n = self.view_n_bottom - self.view_n_top;
            let dh = -delta.x / rect.width().max(1.0) * span_h;
            let dn = -delta.y / rect.height().max(1.0) * span_n;
            let clamped = clamp_view_ppm_2d(
                [
                    self.view_h_left + dh,
                    self.view_h_right + dh,
                    self.view_n_top + dn,
                    self.view_n_bottom + dn,
                ],
                limits,
            );
            self.view_h_left = clamped[0];
            self.view_h_right = clamped[1];
            self.view_n_top = clamped[2];
            self.view_n_bottom = clamped[3];
        }

        // Rubber-band 2D Zoom Box (`ZoomBox` mode or `Shift+Drag`) & Box-Pick (`PickPeaks` mode)
        let zoom_drag_active = !multi_touch_active
            && !is_compact_mobile
            && !primary_pans
            && !peak_drag_active
            && response.dragged_by(egui::PointerButton::Primary)
            && (tool_mode == ToolMode2d::ZoomBox || shift_down);
        if zoom_drag_active && self.workbook.two_d.drag_box_start_pos.is_none() {
            if let Some(origin) = ui
                .input(|i| i.pointer.press_origin())
                .or_else(|| response.interact_pointer_pos())
            {
                if rect.contains(origin) && !in_peak_edit_bar(origin) {
                    self.workbook.two_d.drag_box_start_pos = Some(origin);
                }
            }
        }

        let pick_box_drag_active = !multi_touch_active
            && !is_compact_mobile
            && !primary_pans
            && !peak_drag_active
            && !shift_down
            && tool_mode == ToolMode2d::PickPeaks
            && response.dragged_by(egui::PointerButton::Primary);
        if pick_box_drag_active && self.workbook.two_d.pick_box_start_pos.is_none() {
            if let Some(origin) = ui
                .input(|i| i.pointer.press_origin())
                .or_else(|| response.interact_pointer_pos())
            {
                if rect.contains(origin) && !in_peak_edit_bar(origin) {
                    self.workbook.two_d.pick_box_start_pos = Some(origin);
                }
            }
        }

        if !ui.input(|i| i.pointer.any_down()) {
            if let Some(drag_idx) = self.workbook.two_d.dragging_peak_idx.take() {
                if let Some(pk) = self.peaks.get(drag_idx).cloned() {
                    self.move_2d_peak_live(drag_idx, pk.h, pk.n, true);
                }
            }
            if let Some(p0) = self.workbook.two_d.drag_box_start_pos.take() {
                if let Some(p1) = response
                    .interact_pointer_pos()
                    .or_else(|| ui.input(|i| i.pointer.latest_pos()))
                {
                    if apply_box_zoom(self, p0, p1) {
                        self.box_zoom_2d_anchor = None;
                    }
                }
            }
            if let Some(p0) = self.workbook.two_d.pick_box_start_pos.take() {
                if let Some(p1) = response
                    .interact_pointer_pos()
                    .or_else(|| ui.input(|i| i.pointer.latest_pos()))
                {
                    let x0 = p0.x.min(p1.x).clamp(rect.left(), rect.right());
                    let x1 = p0.x.max(p1.x).clamp(rect.left(), rect.right());
                    let y0 = p0.y.min(p1.y).clamp(rect.top(), rect.bottom());
                    let y1 = p0.y.max(p1.y).clamp(rect.top(), rect.bottom());
                    if (x1 - x0) >= 6.0 && (y1 - y0) >= 6.0 {
                        let cur_v = [
                            self.view_h_left,
                            self.view_h_right,
                            self.view_n_top,
                            self.view_n_bottom,
                        ];
                        let (h0, n0) = screen_to_ppm_cur(egui::pos2(x0, y0), cur_v);
                        let (h1, n1) = screen_to_ppm_cur(egui::pos2(x1, y1), cur_v);
                        let _ = self.box_pick_2d_region(h0, h1, n0, n1);
                    }
                }
            }
            self.suppress_hsqc_pan_until_release = false;
        }

        if response.double_clicked()
            && !multi_touch_active
            && !response
                .interact_pointer_pos()
                .is_some_and(in_peak_edit_bar)
        {
            let cur = [
                self.view_h_left,
                self.view_h_right,
                self.view_n_top,
                self.view_n_bottom,
            ];
            self.workbook.two_d.push_zoom(cur);
            self.reset_view_to_spec();
        }

        // Scroll Wheel over 2D Plot:
        // - Plain Scroll = Contour Threshold (`contour_base` up/down, `×1.18` / `÷1.18`, consuming scroll delta!)
        // - `Ctrl` / `Cmd` + Scroll = 2D Zoom around cursor (consuming scroll delta!)
        if !multi_touch_active {
            if let Some(hover_pos) = response.hover_pos() {
                let (scroll_y, ctrl_or_cmd) = ui.ctx().input_mut(|i| {
                    let sy = i.smooth_scroll_delta.y;
                    if sy.abs() > 0.05 {
                        i.smooth_scroll_delta = egui::Vec2::ZERO;
                    }
                    (sy, i.modifiers.command || i.modifiers.ctrl)
                });
                if scroll_y.abs() > 0.05 {
                    if ctrl_or_cmd {
                        let zoom = (1.0_f32 - scroll_y * 0.003).clamp(0.72, 1.38);
                        let u =
                            ((hover_pos.x - rect.left()) / rect.width().max(1.0)).clamp(0.0, 1.0);
                        let v =
                            ((hover_pos.y - rect.top()) / rect.height().max(1.0)).clamp(0.0, 1.0);
                        let anchor_h =
                            self.view_h_left + u * (self.view_h_right - self.view_h_left);
                        let anchor_n = self.view_n_top + v * (self.view_n_bottom - self.view_n_top);
                        let clamped = clamp_view_ppm_2d(
                            [
                                anchor_h + (self.view_h_left - anchor_h) * zoom,
                                anchor_h + (self.view_h_right - anchor_h) * zoom,
                                anchor_n + (self.view_n_top - anchor_n) * zoom,
                                anchor_n + (self.view_n_bottom - anchor_n) * zoom,
                            ],
                            limits,
                        );
                        self.view_h_left = clamped[0];
                        self.view_h_right = clamped[1];
                        self.view_n_top = clamped[2];
                        self.view_n_bottom = clamped[3];
                    } else {
                        let notches = (scroll_y / 36.0).clamp(-3.0, 3.0);
                        let factor = CONTOUR_SCROLL_STEP_RATIO.powf(-notches);
                        self.contour_base = (self.contour_base * factor).clamp(base_lo, base_hi);
                    }
                }
            }
        }

        let view_ppm = [
            self.view_h_left,
            self.view_h_right,
            self.view_n_top,
            self.view_n_bottom,
        ];

        // Stationary Right-Click Context Menu on 2D Canvas
        let snap_rx_ppm =
            ((4.0 / rect.width().max(1.0) * (view_ppm[1] - view_ppm[0]).abs()) as f64).max(
                self.hsqc_spec
                    .as_ref()
                    .map(|s| s.axis_x.step_ppm.abs() * 2.0)
                    .unwrap_or(0.01),
            );
        let snap_ry_ppm =
            ((4.0 / rect.height().max(1.0) * (view_ppm[3] - view_ppm[2]).abs()) as f64).max(
                self.hsqc_spec
                    .as_ref()
                    .map(|s| s.axis_y.step_ppm.abs() * 2.0)
                    .unwrap_or(0.02),
            );
        response.context_menu(|ui| {
            if self.secondary_drag_accum_2d >= 5.0 {
                ui.close();
                return;
            }
            ui.set_min_width(225.0);
            let [th, tn] = self
                .context_menu_2d_ppm
                .unwrap_or([self.crosshair.h_ppm, self.crosshair.n_ppm]);
            ui.label(
                egui::RichText::new(format!("F2 {pretty_x} {th:.3} | F1 {pretty_y} {tn:.3} ppm"))
                    .small()
                    .monospace()
                    .weak(),
            );
            ui.separator();
            if ui.button("+ Add Peak (Snap to Apex)").clicked() {
                let _ = self.add_manual_2d_peak(th, tn, Some((snap_rx_ppm, snap_ry_ppm)));
                self.workbook.two_d.show_peak_table = true;
                ui.close();
            }
            if ui.button("+ Add Peak (Exact Cursor)").clicked() {
                let _ = self.add_manual_2d_peak(th, tn, None);
                self.workbook.two_d.show_peak_table = true;
                ui.close();
            }
            if let Some(sel) = self
                .workbook
                .two_d
                .selected_peak_idx
                .filter(|&i| i < self.peaks.len() && self.peaks[i].c.is_none())
            {
                ui.separator();
                if ui
                    .button(format!("Snap Peak #{} to Extremum", sel + 1))
                    .clicked()
                {
                    self.snap_2d_peak_by_index(sel);
                    ui.close();
                }
                if ui.button(format!("Delete Peak #{}", sel + 1)).clicked() {
                    self.delete_2d_peak(sel);
                    ui.close();
                }
            }
            ui.separator();
            if ui.button("Auto-Pick All 2D Peaks").clicked() {
                self.pick_2d_roots();
                ui.close();
            }
            let tbl_txt = if self.workbook.two_d.show_peak_table {
                "Hide 2D Peak Table"
            } else {
                "Show 2D Peak Table"
            };
            if ui.button(tbl_txt).clicked() {
                self.workbook.two_d.show_peak_table = !self.workbook.two_d.show_peak_table;
                ui.close();
            }
            if ui.button("Suppress Water Band").clicked() {
                self.suppress_water_active_spectrum();
                ui.close();
            }
            if self.water_undo_2d.is_some() && ui.button("Undo Water Suppression").clicked() {
                self.undo_water_suppression_active();
                ui.close();
            }
            if ui.button("Reset Full View (F)").clicked() {
                self.reset_view_to_spec();
                ui.close();
            }
        });

        // 3. Render Top F2 & Left F1 1D Marginal Strips (pixel-aligned with `rect`)
        if let (Some(r_f2), Some(r_f1), Some(r_corner)) = (f2_rect, f1_rect, corner_rect) {
            // Corner mode pill (`Auto 1D` | `Skyline` | `Slice`)
            let c_resp = ui.interact(
                r_corner,
                ui.scope_id().with("wb_2d_marginal_corner_btn"),
                egui::Sense::click(),
            );
            if c_resp.clicked() {
                if self.workbook.two_d.tool_mode == ToolMode2d::CrosshairSlice {
                    self.workbook.two_d.tool_mode = ToolMode2d::ZoomBox;
                }
                self.workbook.two_d.marginal_mode = self.workbook.two_d.marginal_mode.cycle_next();
            }
            let cp = ui.painter_at(r_corner);
            cp.rect_filled(r_corner, 2.0, theme.marginal_bg);
            cp.rect_stroke(
                r_corner,
                2.0,
                egui::Stroke::new(1.0, theme.marginal_border),
                egui::StrokeKind::Inside,
            );
            cp.text(
                egui::pos2(r_corner.center().x, r_corner.top() + 18.0),
                egui::Align2::CENTER_CENTER,
                format!("{pretty_x} / {pretty_y}"),
                egui::FontId::proportional(10.5),
                theme.axis_text,
            );
            let mode_badge = self.workbook.two_d.effective_marginal_mode().short_badge();
            cp.text(
                egui::pos2(r_corner.center().x, r_corner.bottom() - 15.0),
                egui::Align2::CENTER_CENTER,
                mode_badge,
                egui::FontId::monospace(9.0),
                theme.marginal_trace_ext,
            );
            c_resp.on_hover_text(
                "Click to cycle 2D marginal source: Auto (External 1D / Skyline) -> Skyline Only -> Live Crosshair Slice",
            );

            // Interact on F2 & F1 marginal strips (scroll = vertical marginal gain, drag/click = move crosshair)
            let f2_resp = ui.interact(
                r_f2,
                ui.scope_id().with("wb_2d_f2_marginal_sense"),
                egui::Sense::click_and_drag(),
            );
            if f2_resp.hovered() {
                let sy = ui.ctx().input_mut(|i| {
                    let s = i.smooth_scroll_delta.y;
                    if s.abs() > 0.05 {
                        i.smooth_scroll_delta = egui::Vec2::ZERO;
                    }
                    s
                });
                if sy.abs() > 0.05 {
                    let notches = (sy / 36.0).clamp(-3.0, 3.0);
                    self.workbook.two_d.f2_gain = (self.workbook.two_d.f2_gain
                        * CONTOUR_SCROLL_STEP_RATIO.powf(notches))
                    .clamp(0.08, 64.0);
                }
            }
            if f2_resp.clicked() || f2_resp.dragged() {
                if let Some(p) = f2_resp.interact_pointer_pos() {
                    let u = ((p.x - r_f2.left()) / r_f2.width().max(1.0)).clamp(0.0, 1.0);
                    let h_ppm =
                        (self.view_h_left + u * (self.view_h_right - self.view_h_left)) as f64;
                    self.crosshair.h_ppm = h_ppm;
                }
            }

            let f1_resp = ui.interact(
                r_f1,
                ui.scope_id().with("wb_2d_f1_marginal_sense"),
                egui::Sense::click_and_drag(),
            );
            if f1_resp.hovered() {
                let sy = ui.ctx().input_mut(|i| {
                    let s = i.smooth_scroll_delta.y;
                    if s.abs() > 0.05 {
                        i.smooth_scroll_delta = egui::Vec2::ZERO;
                    }
                    s
                });
                if sy.abs() > 0.05 {
                    let notches = (sy / 36.0).clamp(-3.0, 3.0);
                    self.workbook.two_d.f1_gain = (self.workbook.two_d.f1_gain
                        * CONTOUR_SCROLL_STEP_RATIO.powf(notches))
                    .clamp(0.08, 64.0);
                }
            }
            if f1_resp.clicked() || f1_resp.dragged() {
                if let Some(p) = f1_resp.interact_pointer_pos() {
                    let v = ((p.y - r_f1.top()) / r_f1.height().max(1.0)).clamp(0.0, 1.0);
                    let n_ppm =
                        (self.view_n_top + v * (self.view_n_bottom - self.view_n_top)) as f64;
                    self.crosshair.n_ppm = n_ppm;
                }
            }

            if let Some(ref spec) = self.hsqc_spec {
                let (f2_curve, f1_curve) = evaluate_2d_marginals(
                    &mut self.workbook.two_d,
                    spec,
                    self.hsqc_gen,
                    nuc_x,
                    nuc_y,
                    view_ppm,
                    &self.crosshair,
                    &self.stack1d_panel,
                    rect.width(),
                    rect.height(),
                );
                paint_f2_marginal_strip(
                    ui.painter(),
                    r_f2,
                    view_ppm,
                    &f2_curve,
                    self.workbook.two_d.f2_gain,
                    self.crosshair.h_ppm,
                    &theme,
                );
                paint_f1_marginal_strip(
                    ui.painter(),
                    r_f1,
                    view_ppm,
                    &f1_curve,
                    self.workbook.two_d.f1_gain,
                    self.crosshair.n_ppm,
                    &theme,
                );
            } else {
                ui.painter().rect_filled(r_f2, 2.0, theme.marginal_bg);
                ui.painter().rect_filled(r_f1, 2.0, theme.marginal_bg);
            }
        }

        // 4. Render Main 2D WebGPU Contour Canvas (Paper White OR Dark Oscilloscope)
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 2.0, theme.canvas_bg);

        // Subtle major-tick grid lines on Paper White (and Dark fallback when GPU callback has transparent bg)
        let tick_steps = if is_compact_mobile { 4usize } else { 6usize };
        for i in 1..tick_steps {
            let t = (i as f32) / (tick_steps as f32);
            let gx = rect.left() + t * rect.width();
            let gy = rect.top() + t * rect.height();
            painter.line_segment(
                [egui::pos2(gx, rect.top()), egui::pos2(gx, rect.bottom())],
                egui::Stroke::new(0.8, theme.grid_line),
            );
            painter.line_segment(
                [egui::pos2(rect.left(), gy), egui::pos2(rect.right(), gy)],
                egui::Stroke::new(0.8, theme.grid_line),
            );
        }

        let mut gpu_layers: Vec<OverlayLayerSpec> = Vec::new();
        if self.primary_2d_visible {
            if let Some(spec) = &self.hsqc_spec {
                let mut base_layer = OverlayLayerSpec::from_spectrum_2d(
                    1,
                    self.hsqc_gen,
                    spec,
                    Arc::clone(&self.hsqc_arc),
                    self.contour_base,
                    0,
                );
                let (pos_c, neg_c) = contour_palette_for_theme(0, self.paper_white_canvas);
                base_layer.pos_rgba = pos_c;
                base_layer.neg_rgba = neg_c;
                base_layer.level_factor = self.contour_factor;
                base_layer.num_levels = self.contour_levels;
                base_layer.show_negative = self.show_negative;
                base_layer.ppm_nudge = self.primary_ppm_nudge;
                gpu_layers.push(base_layer);
            }
        }
        for (idx, (_, ov)) in self.overlay_layers.iter().enumerate() {
            if ov.visible {
                let mut layer = ov.clone();
                if self.paper_white_canvas {
                    let (pos_c, neg_c) = contour_palette_for_theme(idx + 1, true);
                    layer.pos_rgba = [pos_c[0], pos_c[1], pos_c[2], ov.pos_rgba[3]];
                    layer.neg_rgba = [neg_c[0], neg_c[1], neg_c[2], ov.neg_rgba[3]];
                }
                gpu_layers.push(layer);
            }
        }

        if !gpu_layers.is_empty() {
            let cb = egui_wgpu::Callback::new_paint_callback(
                rect,
                MultiContourCallback::new(view_ppm, [rect.width(), rect.height()], gpu_layers)
                    .with_opaque_background(!self.paper_white_canvas),
            );
            painter.add(cb);
        }

        let ppm_to_screen = |h: f64, n: f64| -> egui::Pos2 {
            SyncCrosshair::ppm_to_screen_2d(view_ppm, rect, h, n)
        };

        // Crisp subtle dashed δ₁ = δ₂ homonuclear diagonal when direct & indirect nuclei match
        if nuc_x == nuc_y {
            let x_lo = self.view_h_left.min(self.view_h_right) as f64;
            let x_hi = self.view_h_left.max(self.view_h_right) as f64;
            let y_lo = self.view_n_top.min(self.view_n_bottom) as f64;
            let y_hi = self.view_n_top.max(self.view_n_bottom) as f64;
            let d_lo = x_lo.max(y_lo);
            let d_hi = x_hi.min(y_hi);
            if d_hi > d_lo {
                let p_start = ppm_to_screen(d_hi, d_hi);
                let p_end = ppm_to_screen(d_lo, d_lo);
                let total_len = p_start.distance(p_end);
                if total_len >= 2.0 {
                    let dir = (p_end - p_start) / total_len;
                    let dash = 6.0_f32;
                    let gap_d = 5.0_f32;
                    let diag_stroke = egui::Stroke::new(1.15, theme.diagonal_stroke);
                    let mut s = 0.0_f32;
                    while s < total_len {
                        let e = (s + dash).min(total_len);
                        painter.line_segment([p_start + dir * s, p_start + dir * e], diag_stroke);
                        s += dash + gap_d;
                    }
                    let lbl_pt = p_start + dir * (total_len * 0.12).clamp(18.0, total_len - 4.0);
                    if rect.shrink(14.0).contains(lbl_pt) {
                        painter.text(
                            egui::pos2(lbl_pt.x + 5.0, lbl_pt.y + 4.0),
                            egui::Align2::LEFT_TOP,
                            "δ₁ = δ₂",
                            egui::FontId::monospace(9.5),
                            theme.axis_muted,
                        );
                    }
                }
            }
        }

        let is_prot_2d = self.is_protein_2d_context();

        // Tap / Click handling:
        // - `Ctrl/Cmd + Click` quick-adds a 2D peak in ANY mode (`Alt` disables snap)
        // - Clicking an existing peak within `hit_radius_px` selects it (and syncs spin-system / residue on protein 2D)
        // - Left-Click in `PickPeaks [K]` mode adds a 2D peak (`Alt` disables snap)
        // - Otherwise in `ZoomBox` mode: 2-click zoom anchor -> zoom box!
        if response.clicked() && !multi_touch_active && !self.suppress_hsqc_pan_until_release {
            if let Some(click_pos) = response.interact_pointer_pos() {
                if !in_peak_edit_bar(click_pos) {
                    let (ctrl_or_cmd, alt_down) =
                        ui.input(|i| (i.modifiers.command || i.modifiers.ctrl, i.modifiers.alt));
                    let (tap_h, tap_n) = screen_to_ppm_cur(click_pos, view_ppm);
                    let snap_opt = (!alt_down).then_some((snap_rx_ppm, snap_ry_ppm));

                    if let Some(anchor) = self.box_zoom_2d_anchor.take() {
                        if !apply_box_zoom(self, egui::pos2(anchor[0], anchor[1]), click_pos) {
                            self.box_zoom_2d_anchor = Some([click_pos.x, click_pos.y]);
                        }
                    } else if ctrl_or_cmd {
                        let _ = self.add_manual_2d_peak(tap_h, tap_n, snap_opt);
                        self.workbook.two_d.show_peak_table = true;
                    } else {
                        let hit_pk =
                            nearest_2d_peak(&self.peaks, click_pos, view_ppm, hit_radius_px);
                        let mut hit_any = false;

                        if let Some((pk_idx, _)) = hit_pk {
                            self.workbook.two_d.selected_peak_idx = Some(pk_idx);
                            let pk = &self.peaks[pk_idx];
                            self.crosshair.snap_to(pk.h, pk.n, self.crosshair.c_ppm);
                            hit_any = true;
                        }

                        if is_prot_2d {
                            let mut best_ss: Option<(usize, f32)> = None;
                            for (s, ss) in self.systems.iter().enumerate() {
                                let pt = ppm_to_screen(ss.h, ss.n);
                                let dist = pt.distance(click_pos);
                                if dist <= hit_radius_px && best_ss.is_none_or(|(_, bd)| dist < bd)
                                {
                                    best_ss = Some((s, dist));
                                }
                            }
                            if let Some((s, _)) = best_ss {
                                self.selected_ss = Some(s);
                                let ss = &self.systems[s];
                                self.crosshair.snap_to(ss.h, ss.n, None);
                                if let Some(res) = &self.assign_res {
                                    if res.position.get(s).copied().unwrap_or(-1) >= 0 {
                                        let rk = res.position[s] as usize;
                                        self.selected_res = Some(rk);
                                        self.assign_target_res = rk;
                                        self.mol3d_panel.highlight_sequence_residue(Some(rk), true);
                                    }
                                }
                                hit_any = true;
                            }
                        }

                        if !hit_any {
                            self.workbook.two_d.selected_peak_idx = None;
                            if tool_mode == ToolMode2d::PickPeaks {
                                let _ = self.add_manual_2d_peak(tap_h, tap_n, snap_opt);
                                self.workbook.two_d.show_peak_table = true;
                            } else if !is_compact_mobile
                                && (tool_mode == ToolMode2d::ZoomBox || self.box_zoom_2d_mode)
                            {
                                self.box_zoom_2d_anchor = Some([click_pos.x, click_pos.y]);
                            } else {
                                self.crosshair.update_from_2d_hover(tap_h, tap_n);
                            }
                        }
                    }
                }
            }
        }

        // Draw up to `3,000` 2D cross-peak / CON / protein spin-system markers + labels
        let draw_selection_diamond = |pt: egui::Pos2| {
            let r = 8.5_f32;
            let pts = vec![
                egui::pos2(pt.x, pt.y - r),
                egui::pos2(pt.x + r, pt.y),
                egui::pos2(pt.x, pt.y + r),
                egui::pos2(pt.x - r, pt.y),
                egui::pos2(pt.x, pt.y - r),
            ];
            painter.add(egui::Shape::line(
                pts,
                egui::Stroke::new(1.8, theme.zoom_box_stroke),
            ));
        };

        if is_prot_2d && !self.systems.is_empty() && !(nuc_x == "13C" && nuc_y == "15N") {
            // Draw protein spin system peak markers and residue / custom labels
            for (s, ss) in self.systems.iter().enumerate() {
                let pt = ppm_to_screen(ss.h, ss.n);
                if !rect.contains(pt) {
                    continue;
                }
                let matched_pk_sel = self
                    .workbook
                    .two_d
                    .selected_peak_idx
                    .and_then(|pi| self.peaks.get(pi))
                    .is_some_and(|pk| {
                        pk.c.is_none() && (pk.h - ss.h).abs() < 0.025 && (pk.n - ss.n).abs() < 0.25
                    });
                let is_sel = self.selected_ss == Some(s) || matched_pk_sel;
                let is_locked = self.session.state.is_spin_system_locked(s);
                let (assigned_k, conf) = self
                    .assign_res
                    .as_ref()
                    .and_then(|r| Some((*r.position.get(s)?, *r.confidence.get(s)?)))
                    .unwrap_or((-1, 0.0));

                let marker_col = if is_locked {
                    theme.peak_locked
                } else if assigned_k >= 0 {
                    if conf >= 0.80 {
                        theme.peak_hi_conf
                    } else {
                        theme.peak_med_conf
                    }
                } else {
                    theme.peak_unassigned
                };

                let sz = if is_sel { 6.5 } else { 3.8 };
                painter.line_segment(
                    [egui::pos2(pt.x - sz, pt.y), egui::pos2(pt.x + sz, pt.y)],
                    egui::Stroke::new(if is_sel { 2.0 } else { 1.2 }, marker_col),
                );
                painter.line_segment(
                    [egui::pos2(pt.x, pt.y - sz), egui::pos2(pt.x, pt.y + sz)],
                    egui::Stroke::new(if is_sel { 2.0 } else { 1.2 }, marker_col),
                );
                if is_sel {
                    draw_selection_diamond(pt);
                }

                let custom_lbl = self
                    .peak_2d_labels
                    .get(&peak_label_key_2d(ss.h, ss.n))
                    .cloned();
                if self.show_peak_labels || custom_lbl.is_some() {
                    let text = if let Some(cl) = custom_lbl {
                        cl
                    } else if assigned_k >= 0 && (assigned_k as usize) < self.sequence.len() {
                        let k = assigned_k as usize;
                        if is_locked {
                            format!("*{}{}", three_to_one(&self.sequence[k]), self.seq_ids[k])
                        } else {
                            format!("{}{}", three_to_one(&self.sequence[k]), self.seq_ids[k])
                        }
                    } else {
                        format!("#{s}")
                    };
                    painter.text(
                        egui::pos2(pt.x + 6.0, pt.y - 6.0),
                        egui::Align2::LEFT_BOTTOM,
                        text,
                        egui::FontId::monospace(10.5),
                        marker_col,
                    );
                }
            }
        } else {
            let visible_count = self
                .peaks
                .iter()
                .filter(|p| p.c.is_none())
                .take(3_000)
                .filter(|p| rect.contains(ppm_to_screen(p.h, p.n)))
                .count();
            let is_con_view = nuc_x == "13C" && nuc_y == "15N";
            for (ord, (pk_idx, pk)) in self
                .peaks
                .iter()
                .enumerate()
                .filter(|(_, p)| p.c.is_none())
                .take(3_000)
                .enumerate()
            {
                let pt = ppm_to_screen(pk.h, pk.n);
                if !rect.contains(pt) {
                    continue;
                }
                let is_sel = self.workbook.two_d.selected_peak_idx == Some(pk_idx);
                let marker_col = if is_sel {
                    theme.zoom_box_stroke
                } else if is_con_view {
                    if pk.n >= 131.5 {
                        theme.peak_med_conf
                    } else {
                        theme.peak_hi_conf
                    }
                } else if pk.sign < 0 {
                    egui::Color32::from_rgb(244, 63, 94)
                } else {
                    theme.peak_hi_conf
                };
                let sz = if is_sel { 5.8 } else { 3.8 };
                let sw = if is_sel { 1.9 } else { 1.25 };
                painter.line_segment(
                    [egui::pos2(pt.x - sz, pt.y), egui::pos2(pt.x + sz, pt.y)],
                    egui::Stroke::new(sw, marker_col),
                );
                painter.line_segment(
                    [egui::pos2(pt.x, pt.y - sz), egui::pos2(pt.x, pt.y + sz)],
                    egui::Stroke::new(sw, marker_col),
                );
                if is_sel {
                    draw_selection_diamond(pt);
                }
                let custom_lbl = self
                    .peak_2d_labels
                    .get(&peak_label_key_2d(pk.h, pk.n))
                    .cloned();
                if let Some(cl) = custom_lbl {
                    painter.text(
                        egui::pos2(pt.x + 5.0, pt.y - 5.0),
                        egui::Align2::LEFT_BOTTOM,
                        cl,
                        egui::FontId::monospace(10.0),
                        marker_col,
                    );
                } else if (self.show_peak_labels && visible_count <= 260) || is_sel {
                    let lbl = if is_con_view && pk.n >= 131.5 {
                        format!("Pro#{}", ord + 1)
                    } else {
                        format!("#{}", ord + 1)
                    };
                    painter.text(
                        egui::pos2(pt.x + 5.0, pt.y - 5.0),
                        egui::Align2::LEFT_BOTTOM,
                        lbl,
                        egui::FontId::monospace(9.5),
                        marker_col,
                    );
                }
            }
        }

        // Update & draw synchronized crosshair
        if let Some(hp) = response.hover_pos() {
            if !multi_touch_active {
                let u = (hp.x - rect.left()) / rect.width().max(1.0);
                let v = (hp.y - rect.top()) / rect.height().max(1.0);
                let cur_h = (self.view_h_left + u * (self.view_h_right - self.view_h_left)) as f64;
                let cur_n = (self.view_n_top + v * (self.view_n_bottom - self.view_n_top)) as f64;
                self.crosshair.update_from_2d_hover(cur_h, cur_n);
            }
        }
        let ch_pt = ppm_to_screen(self.crosshair.h_ppm, self.crosshair.n_ppm);
        if rect.contains(ch_pt) {
            let ch_stroke = if self.crosshair.locked {
                egui::Stroke::new(1.15, theme.crosshair_locked)
            } else {
                egui::Stroke::new(0.85, theme.crosshair_unlocked)
            };
            painter.line_segment(
                [
                    egui::pos2(rect.left(), ch_pt.y),
                    egui::pos2(rect.right(), ch_pt.y),
                ],
                ch_stroke,
            );
            painter.line_segment(
                [
                    egui::pos2(ch_pt.x, rect.top()),
                    egui::pos2(ch_pt.x, rect.bottom()),
                ],
                ch_stroke,
            );
        }

        // Draw live rubber-band 2D Zoom Box OR Box-Pick preview (`ΔF2 ppm × ΔF1 ppm`)
        let active_box_start = self
            .workbook
            .two_d
            .drag_box_start_pos
            .map(|p| (p, false))
            .or_else(|| self.workbook.two_d.pick_box_start_pos.map(|p| (p, true)))
            .or_else(|| {
                self.box_zoom_2d_anchor
                    .map(|a| (egui::pos2(a[0], a[1]), false))
            });
        if let Some((anchor_pt, is_pick_box)) = active_box_start {
            if let Some(cur_pos) = response
                .hover_pos()
                .or_else(|| ui.input(|i| i.pointer.latest_pos()))
            {
                let box_rect = egui::Rect::from_two_pos(
                    egui::pos2(
                        anchor_pt.x.clamp(rect.left(), rect.right()),
                        anchor_pt.y.clamp(rect.top(), rect.bottom()),
                    ),
                    egui::pos2(
                        cur_pos.x.clamp(rect.left(), rect.right()),
                        cur_pos.y.clamp(rect.top(), rect.bottom()),
                    ),
                );
                let (fill_c, stroke_c, pfx) = if is_pick_box {
                    (
                        egui::Color32::from_rgba_unmultiplied(16, 185, 129, 34),
                        egui::Color32::from_rgb(16, 185, 129),
                        "Box-Pick ",
                    )
                } else {
                    (theme.zoom_box_fill, theme.zoom_box_stroke, "")
                };
                painter.rect_filled(box_rect, 1.5, fill_c);
                painter.rect_stroke(
                    box_rect,
                    1.5,
                    egui::Stroke::new(1.35, stroke_c),
                    egui::StrokeKind::Inside,
                );
                let dh_ppm = ((box_rect.width() / rect.width().max(1.0))
                    * (self.view_h_left - self.view_h_right).abs())
                .abs();
                let dn_ppm = ((box_rect.height() / rect.height().max(1.0))
                    * (self.view_n_bottom - self.view_n_top).abs())
                .abs();
                painter.text(
                    egui::pos2(
                        box_rect.left() + 4.0,
                        (box_rect.top() - 4.0).max(rect.top() + 14.0),
                    ),
                    egui::Align2::LEFT_BOTTOM,
                    format!("{pfx}ΔF2 {dh_ppm:.2} {pretty_x} | ΔF1 {dn_ppm:.2} {pretty_y}"),
                    egui::FontId::monospace(9.5),
                    stroke_c,
                );
            }
        }

        // 5. Crisp Plot Frame + Bottom F2 & Right F1 Chemical-Shift Rulers
        painter.rect_stroke(
            rect,
            2.0,
            egui::Stroke::new(1.0, theme.axis_border),
            egui::StrokeKind::Inside,
        );

        let full_painter = ui.painter();
        let x_span_abs = (self.view_h_left - self.view_h_right).abs();
        let y_span_abs = (self.view_n_bottom - self.view_n_top).abs();
        let x_tick_dec = if x_span_abs < 0.5 {
            3usize
        } else if x_span_abs < 40.0 {
            2usize
        } else {
            1usize
        };
        let y_tick_dec = if y_span_abs < 0.5 {
            3usize
        } else if y_span_abs < 25.0 {
            2usize
        } else {
            1usize
        };

        if is_compact_mobile {
            for i in 0..=tick_steps {
                let t = (i as f32) / (tick_steps as f32);
                let h_val = self.view_h_left + t * (self.view_h_right - self.view_h_left);
                let x = rect.left() + t * rect.width();
                painter.text(
                    egui::pos2(x, rect.bottom() - 3.0),
                    egui::Align2::CENTER_BOTTOM,
                    format!("{h_val:.x_tick_dec$}"),
                    egui::FontId::monospace(9.5),
                    theme.axis_muted,
                );
                let n_val = self.view_n_top + t * (self.view_n_bottom - self.view_n_top);
                let y = rect.top() + t * rect.height();
                painter.text(
                    egui::pos2(rect.left() + 4.0, y),
                    egui::Align2::LEFT_CENTER,
                    format!("{n_val:.y_tick_dec$}"),
                    egui::FontId::monospace(9.5),
                    theme.axis_muted,
                );
            }
        } else {
            let ruler_col = egui::Color32::from_rgb(203, 213, 225);
            let tick_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(100, 116, 139));
            for i in 0..=tick_steps {
                let t = (i as f32) / (tick_steps as f32);
                let h_val = self.view_h_left + t * (self.view_h_right - self.view_h_left);
                let x = f2_ppm_to_screen_x(view_ppm, rect, h_val as f64);
                full_painter.line_segment(
                    [
                        egui::pos2(x, rect.bottom()),
                        egui::pos2(x, rect.bottom() + 4.0),
                    ],
                    tick_stroke,
                );
                let x_lbl = if i == tick_steps / 2 {
                    format!("{h_val:.x_tick_dec$}  f2 ({pretty_x} ppm)")
                } else {
                    format!("{h_val:.x_tick_dec$}")
                };
                let align_x = if i == 0 {
                    egui::Align2::LEFT_TOP
                } else if i == tick_steps {
                    egui::Align2::RIGHT_TOP
                } else {
                    egui::Align2::CENTER_TOP
                };
                full_painter.text(
                    egui::pos2(x, rect.bottom() + 5.0),
                    align_x,
                    x_lbl,
                    egui::FontId::monospace(9.5),
                    ruler_col,
                );

                let n_val = self.view_n_top + t * (self.view_n_bottom - self.view_n_top);
                let y = f1_ppm_to_screen_y(view_ppm, rect, n_val as f64);
                full_painter.line_segment(
                    [
                        egui::pos2(rect.right(), y),
                        egui::pos2(rect.right() + 4.0, y),
                    ],
                    tick_stroke,
                );
                let y_lbl = if i == 0 {
                    format!("{n_val:.y_tick_dec$} {pretty_y}")
                } else {
                    format!("{n_val:.y_tick_dec$}")
                };
                let align_y = if i == 0 {
                    egui::Align2::LEFT_TOP
                } else if i == tick_steps {
                    egui::Align2::LEFT_BOTTOM
                } else {
                    egui::Align2::LEFT_CENTER
                };
                full_painter.text(
                    egui::pos2(rect.right() + 6.0, y),
                    align_y,
                    y_lbl,
                    egui::FontId::monospace(9.5),
                    ruler_col,
                );
            }
        }

        // 6. Floating Compact 2D Parameters & Active Layers Card (Top-Right of `rect`)
        if !is_compact_mobile
            && self.workbook.two_d.show_params_card
            && rect.width() >= 280.0
            && rect.height() >= 180.0
        {
            let n_ov = self.overlay_layers.len();
            let card_h = 84.0 + (n_ov.min(4) as f32) * 18.0;
            let card_w = 214.0_f32.min(rect.width() * 0.46);
            let card_rect = egui::Rect::from_min_size(
                egui::pos2(rect.right() - card_w - 8.0, rect.top() + 8.0),
                egui::vec2(card_w, card_h),
            );
            painter.rect_filled(card_rect, 4.0, theme.card_bg);
            painter.rect_stroke(
                card_rect,
                4.0,
                egui::Stroke::new(1.0, theme.card_border),
                egui::StrokeKind::Inside,
            );

            let mut card_ui = ui.new_child(
                egui::UiBuilder::new()
                    .max_rect(card_rect.shrink(6.0))
                    .layout(egui::Layout::top_down(egui::Align::Min)),
            );
            card_ui.spacing_mut().item_spacing.y = 2.5;

            card_ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new(&self.primary_2d_label)
                        .size(10.5)
                        .strong()
                        .color(theme.card_title),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let close_r =
                        ui.allocate_exact_size(egui::vec2(14.0, 14.0), egui::Sense::click());
                    paint_icon(
                        ui.painter(),
                        close_r.0.shrink(2.0),
                        WorkbookIcon::CloseX,
                        theme.card_body,
                    );
                    if close_r.1.on_hover_text("Hide 2D Parameters card").clicked() {
                        self.workbook.two_d.show_params_card = false;
                    }
                });
            });

            if let Some(ref spec) = self.hsqc_spec {
                card_ui.label(
                    egui::RichText::new(format!(
                        "{}×{}  ·  {:.0}/{:.0} MHz",
                        spec.nx(),
                        spec.ny(),
                        spec.axis_x.obs_mhz,
                        spec.axis_y.obs_mhz
                    ))
                    .size(9.5)
                    .monospace()
                    .color(theme.card_body),
                );
            }
            let snr_floor = if self.contour_noise_ref > 1e-9 {
                self.contour_base / self.contour_noise_ref
            } else {
                0.0
            };
            card_ui.label(
                egui::RichText::new(format!(
                    "Base {:.2} ({snr_floor:.1}σ) · {} lvls",
                    self.contour_base, self.contour_levels
                ))
                .size(9.5)
                .monospace()
                .color(theme.card_body),
            );
            let x_cur_dec = if x_span_abs < 30.0 { 3usize } else { 2usize };
            let y_cur_dec = if y_span_abs < 30.0 { 3usize } else { 2usize };
            card_ui.label(
                egui::RichText::new(format!(
                    "F2 {:.x_cur_dec$} | F1 {:.y_cur_dec$} ppm{}",
                    self.crosshair.h_ppm,
                    self.crosshair.n_ppm,
                    if self.crosshair.locked { " *" } else { "" }
                ))
                .size(9.5)
                .monospace()
                .color(theme.card_title),
            );

            for (lbl, ov) in self.overlay_layers.iter_mut().take(4) {
                card_ui.horizontal(|ui| {
                    ui.checkbox(&mut ov.visible, "");
                    ui.label(
                        egui::RichText::new(lbl.as_str())
                            .size(9.5)
                            .color(theme.card_body),
                    );
                });
            }
        } else {
            let x_cur_dec = if x_span_abs < 30.0 { 3usize } else { 2usize };
            let y_cur_dec = if y_span_abs < 30.0 { 3usize } else { 2usize };
            painter.text(
                egui::pos2(rect.right() - 8.0, rect.top() + 6.0),
                egui::Align2::RIGHT_TOP,
                format!(
                    "F2 {pretty_x} {:.x_cur_dec$} | F1 {pretty_y} {:.y_cur_dec$} ppm",
                    self.crosshair.h_ppm, self.crosshair.n_ppm
                ),
                egui::FontId::monospace(10.0),
                theme.axis_text,
            );
        }

        // 7. Floating Selected-Peak Inline Edit Bar (Bottom-Left of `rect`)
        if let (Some(bar_r), Some(sel_idx)) = (
            peak_edit_bar_rect,
            self.workbook
                .two_d
                .selected_peak_idx
                .filter(|&i| i < self.peaks.len() && self.peaks[i].c.is_none()),
        ) {
            painter.rect_filled(bar_r, 4.0, theme.card_bg);
            painter.rect_stroke(
                bar_r,
                4.0,
                egui::Stroke::new(1.1, theme.zoom_box_stroke),
                egui::StrokeKind::Inside,
            );
            let pk_cur = self.peaks[sel_idx].clone();
            let mut f2_val = pk_cur.h;
            let mut f1_val = pk_cur.n;
            let spd_x = if nuc_x == "1H" { 0.001 } else { 0.01 };
            let spd_y = if nuc_y == "1H" { 0.001 } else { 0.01 };
            let lbl_key = peak_label_key_2d(pk_cur.h, pk_cur.n);
            let mut lbl_buf = self
                .peak_2d_labels
                .get(&lbl_key)
                .cloned()
                .unwrap_or_default();
            let mut do_move = false;
            let mut do_snap = false;
            let mut do_del = false;
            let mut do_desel = false;

            let mut bar_ui = ui.new_child(
                egui::UiBuilder::new()
                    .max_rect(bar_r.shrink2(egui::vec2(6.0, 3.0)))
                    .layout(egui::Layout::left_to_right(egui::Align::Center)),
            );
            bar_ui.spacing_mut().item_spacing.x = 5.0;
            bar_ui.label(
                egui::RichText::new(format!("Peak #{}", sel_idx + 1))
                    .size(10.5)
                    .strong()
                    .color(theme.card_title),
            );
            if bar_ui
                .add(
                    egui::DragValue::new(&mut f2_val)
                        .speed(spd_x)
                        .fixed_decimals(3)
                        .prefix(format!("{pretty_x} ")),
                )
                .changed()
            {
                do_move = true;
            }
            if bar_ui
                .add(
                    egui::DragValue::new(&mut f1_val)
                        .speed(spd_y)
                        .fixed_decimals(3)
                        .prefix(format!("{pretty_y} ")),
                )
                .changed()
            {
                do_move = true;
            }
            if bar_ui
                .add(
                    egui::TextEdit::singleline(&mut lbl_buf)
                        .desired_width(110.0)
                        .hint_text("label / residue"),
                )
                .changed()
            {
                if lbl_buf.trim().is_empty() {
                    self.peak_2d_labels.remove(&lbl_key);
                } else {
                    self.peak_2d_labels.insert(lbl_key, lbl_buf);
                }
            }
            if bar_ui
                .small_button("Snap")
                .on_hover_text("Snap peak to local 2D parabolic extremum")
                .clicked()
            {
                do_snap = true;
            }
            if bar_ui
                .small_button("Delete")
                .on_hover_text("Delete selected peak (Del / Backspace)")
                .clicked()
            {
                do_del = true;
            }
            if bar_ui
                .small_button("×")
                .on_hover_text("Deselect peak")
                .clicked()
            {
                do_desel = true;
            }

            if do_del {
                self.delete_2d_peak(sel_idx);
            } else if do_move {
                self.move_2d_peak_live(sel_idx, f2_val, f1_val, true);
            } else if do_snap {
                self.snap_2d_peak_by_index(sel_idx);
            } else if do_desel {
                self.workbook.two_d.selected_peak_idx = None;
            }
        }

        if is_compact_mobile {
            rect
        } else {
            outer_rect
        }
    }

    fn select_residue_and_sync_strip(&mut self, k: usize) {
        if k >= self.sequence.len() {
            return;
        }
        self.selected_res = Some(k);
        self.sidechain_ribbon_state.selected_residue = k;
        self.assign_target_res = k;
        self.mol3d_panel.highlight_sequence_residue(Some(k), true);
        if let Some(res) = &self.assign_res {
            for (s, &pos) in res.position.iter().enumerate() {
                if pos >= 0 && (pos as usize) == k {
                    self.selected_ss = Some(s);
                    if let Some(ss) = self.systems.get(s) {
                        self.crosshair.snap_to(ss.h, ss.n, self.crosshair.c_ppm);
                    }
                    break;
                }
            }
        }
    }

    fn run_joint_sidechain_in_gui(&mut self) {
        if self.sequence.is_empty() {
            self.status_msg =
                "Load sequence and peaks/spin systems before sidechain assignment".to_string();
            return;
        }
        let timer = PerfTimer::start();
        let seed = self.sim_seed.wrapping_add(2026);
        let prior_user_ref = self.pred_ref.clone().or_else(|| self.reference.clone());
        let prior_amb = self.pred_ambiguity_codes.clone();
        let mut loaded_sc_peaks = Vec::new();
        let mut loaded_arom_peaks = Vec::new();
        for p in &self.peaks {
            if EXPERIMENT_NAMES.contains(&p.exp.as_str()) {
                continue;
            }
            if AROMATIC_EXPERIMENTS.contains(&p.exp.as_str()) || p.exp == "AROM_HB_NOE" {
                loaded_arom_peaks.push(p.clone());
            } else {
                loaded_sc_peaks.push(p.clone());
            }
        }

        let mut working_ref = if !loaded_sc_peaks.is_empty() {
            if !self.systems.is_empty() {
                let locked_pairs: Vec<(usize, usize)> = self
                    .session
                    .state
                    .locked_residues
                    .iter()
                    .map(|(&s, &k)| (s, k))
                    .collect();
                let scfg = ScoreConfig {
                    assign_cost: self.assign_cost,
                    locked_assignments: locked_pairs,
                    ..ScoreConfig::default()
                };
                let acfg = AnnealConfig {
                    runs: self.replicas.clamp(4, 16),
                    steps_per_system: self.steps_per_sys,
                    seed,
                    ..AnnealConfig::default()
                };
                let joint = solve_joint_backbone_sidechain(
                    &self.systems,
                    &self.sequence,
                    &self.seq_ids,
                    &self.priors,
                    &self.sidechain_priors,
                    &loaded_sc_peaks,
                    &scfg,
                    &acfg,
                    &SidechainSolverConfig::default(),
                );
                let (_, bb_conf) = to_reference(
                    &joint.backbone_result,
                    &self.systems,
                    &self.sequence,
                    &self.seq_ids,
                    0.0,
                    Some(&self.priors),
                );
                if let Some(truth) = &self.truth {
                    self.metrics = Some(score(&joint.backbone_result, truth, 0.8));
                }
                self.assign_res = Some(joint.backbone_result);
                self.pred_conf = bb_conf;
                joint.sidechain_result.reference
            } else if let Some(base_ref) = prior_user_ref.clone() {
                let sc_res = assign_sidechains(
                    &base_ref,
                    (!self.pred_conf.is_empty()).then_some(&self.pred_conf),
                    &loaded_sc_peaks,
                    &self.sidechain_priors,
                    &SidechainSolverConfig::default(),
                );
                sc_res.reference
            } else {
                self.status_msg =
                    "Load backbone shifts or spin systems before sidechain assignment".to_string();
                return;
            }
        } else {
            if self.pred_ref.is_none() && !self.systems.is_empty() {
                self.run_assignment();
            }
            if let Some(pred) = &self.pred_ref {
                pred.clone()
            } else if self.systems.is_empty() {
                if let Some(r) = &self.reference {
                    r.clone()
                } else {
                    Reference::new(self.sequence.clone(), self.seq_ids.clone())
                }
            } else {
                Reference::new(self.sequence.clone(), self.seq_ids.clone())
            }
        };

        if !loaded_arom_peaks.is_empty() {
            working_ref = merge_gui_aromatic_with_ph(
                &working_ref,
                prior_user_ref.as_ref(),
                &prior_amb,
                (!self.pred_conf.is_empty()).then_some(&self.pred_conf),
                &loaded_arom_peaks,
                &self.sidechain_priors,
                self.overwrite_curated_rings,
                self.sidechain_ribbon_state.sample_ph,
            );
        }

        let mut amb_codes = working_ref.ambiguity_codes.clone();
        if let Ok(stereo_out) = assign_stereospecific_dataset(&StereospecDatasetInput {
            reference: &working_ref,
            j_couplings_tsv: None,
            intra_noes_tsv: None,
            pdb_path: None,
            config: StereospecConfig::default(),
        }) {
            let preserved_amb = working_ref.ambiguity_codes.clone();
            working_ref = stereo_out.reference;
            amb_codes = stereo_out.ambiguity_codes;
            for (key, code) in preserved_amb {
                amb_codes.entry(key).or_insert(code);
            }
            working_ref.ambiguity_codes = amb_codes.clone();
        }

        // P2: Sync auto-inferred Cys redox & His Pelton tautomer variants cleanly without
        // clobbering user_variant_overrides or leaving stale auto-inferred states.
        self.sidechain_ribbon_state
            .sync_inferred_variants_from_reference(&working_ref);

        let n_shifts = working_ref.shifts.len();
        let dt_ms = timer.elapsed_ms();
        self.pred_ref = Some(working_ref);
        self.pred_ambiguity_codes = amb_codes;
        if !loaded_sc_peaks.is_empty() || !loaded_arom_peaks.is_empty() {
            self.has_structure_or_dynamics = true;
        }
        self.show_sidechain_inspector = true;
        if self.selected_res.is_none() && !self.sequence.is_empty() {
            self.select_residue_and_sync_strip(0);
        }
        self.refresh_propensity_report();
        self.sync_session_assignments_from_solver();
        self.status_msg = if !loaded_sc_peaks.is_empty() && !loaded_arom_peaks.is_empty() {
            format!("Assigned {n_shifts} backbone + aliphatic + aromatic shifts in {dt_ms:.0} ms")
        } else if !loaded_sc_peaks.is_empty() {
            format!("Assigned {n_shifts} backbone + aliphatic sidechain shifts in {dt_ms:.0} ms")
        } else if !loaded_arom_peaks.is_empty() {
            format!("Assigned aromatic rings ({n_shifts} total shifts) in {dt_ms:.0} ms")
        } else {
            format!(
                "No sidechain or aromatic peak lists loaded — inspecting {n_shifts} existing shifts & BMRB Z-scores"
            )
        };
    }

    fn draw_sidechain_inspector_drawer(&mut self, ui: &mut egui::Ui) {
        if self.sequence.is_empty() {
            ui.label("Load a sequence or NMR-STAR file to inspect 2D covalent sidechains.");
            return;
        }
        let cur_k = self
            .selected_res
            .unwrap_or(self.sidechain_ribbon_state.selected_residue)
            .min(self.sequence.len() - 1);
        self.selected_res = Some(cur_k);
        self.sidechain_ribbon_state.selected_residue = cur_k;

        let mut trigger_sidechain_solve = false;
        ui.horizontal(|ui| {
            if ui
                .add_enabled(cur_k > 0, egui::Button::new("Prev").small())
                .on_hover_text("Previous residue")
                .clicked()
            {
                self.select_residue_and_sync_strip(cur_k - 1);
            }
            if ui
                .add_enabled(
                    cur_k + 1 < self.sequence.len(),
                    egui::Button::new("Next").small(),
                )
                .on_hover_text("Next residue")
                .clicked()
            {
                self.select_residue_and_sync_strip(cur_k + 1);
            }
            if ui
                .small_button("Assign Sidechains")
                .on_hover_text(
                    "Run joint aliphatic, aromatic (Pelton His tautomer), and stereospecific assignment",
                )
                .clicked()
            {
                trigger_sidechain_solve = true;
            }
            ui.checkbox(
                &mut self.overwrite_curated_rings,
                egui::RichText::new("Overwrite curated rings").size(10.5),
            )
            .on_hover_text(
                "When unchecked (default), user-curated aromatic ring shifts (CD*, CE*, CZ*, HD*, HE*, HZ*) are protected from overwrite, partial mixing, and duplicate re-assignment",
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .small_button("×")
                    .on_hover_text("Hide 2D covalent sidechain inspector")
                    .clicked()
                {
                    self.show_sidechain_inspector = false;
                }
            });
        });

        if trigger_sidechain_solve {
            self.run_joint_sidechain_in_gui();
        }
        if !self.show_sidechain_inspector {
            return;
        }

        let mut assigned_map: BTreeMap<(usize, String), AssignedAtomInfo> = BTreeMap::new();
        let display_ref = self.pred_ref.as_ref().or({
            if self.systems.is_empty() {
                self.reference.as_ref()
            } else {
                None
            }
        });
        if let Some(pred) = display_ref {
            for (&(rk, ref atom), &ppm) in &pred.shifts {
                let amb = self.pred_ambiguity_codes.get(&(rk, atom.clone())).copied();
                assigned_map.insert(
                    (rk, atom.clone()),
                    AssignedAtomInfo::with_metadata(ppm, None, amb),
                );
            }
        }
        let locked_set = self
            .session
            .state
            .locked_residues
            .values()
            .copied()
            .collect();

        let out = {
            let sc_input = SidechainWorkspaceInput::from_assigned_map(
                &self.sequence,
                &self.seq_ids,
                assigned_map,
                &self.priors,
                &self.sidechain_priors,
            )
            .with_backbone_confidences(self.pred_conf.clone())
            .with_locked_residues(locked_set);
            SidechainInspectorPanel::show_inspector(ui, &mut self.sidechain_ribbon_state, &sc_input)
        };

        if out.selected_residue < self.sequence.len()
            && self.selected_res != Some(out.selected_residue)
        {
            self.select_residue_and_sync_strip(out.selected_residue);
        }
        if let Some(jump) = out.jump_to_atom {
            self.select_residue_and_sync_strip(jump.residue_idx);
            jump.apply_to_crosshair(&mut self.crosshair);
            self.status_msg = format!(
                "Jumped crosshair to {} {} {} ({:.3} ppm)",
                jump.residue_name, jump.seq_id, jump.atom_name, jump.ppm
            );
        }
    }

    fn draw_propensity_inspector_drawer(&mut self, ui: &mut egui::Ui) {
        if self.propensity_report.is_none() && (self.pred_ref.is_some() || self.reference.is_some())
        {
            self.refresh_propensity_report();
        }
        let Some(rep) = self.propensity_report.clone() else {
            ui.label(
                "Load an NMR-STAR shift file or run assignment to compute Marsh SSP & Camilloni 4-state δ2D populations.",
            );
            return;
        };
        if rep.residues.is_empty() {
            ui.label("No residues available for SSP / δ2D conformational analysis.");
            return;
        }

        let sel_k = self
            .selected_res
            .unwrap_or(0)
            .min(rep.residues.len().saturating_sub(1));

        egui::Frame::group(ui.style())
            .fill(egui::Color32::from_rgb(20, 26, 37))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("SSP & 4-State δ2D Ensemble")
                            .strong()
                            .color(egui::Color32::from_rgb(220, 195, 255)),
                    );
                    let badge = if rep.is_predominantly_disordered {
                        ("IDP / Disordered", egui::Color32::from_rgb(186, 132, 255))
                    } else {
                        ("Folded / Structured", egui::Color32::from_rgb(92, 225, 175))
                    };
                    ui.label(egui::RichText::new(badge.0).small().color(badge.1));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .small_button("×")
                            .on_hover_text("Hide SSP / δ2D propensity drawer")
                            .clicked()
                        {
                            self.show_propensity_panel = false;
                        }
                        if ui
                            .small_button("Recompute")
                            .on_hover_text("Recompute SSP / δ2D with current T(K) & pH")
                            .clicked()
                        {
                            self.refresh_propensity_report();
                        }
                    });
                });

                ui.label(
                    egui::RichText::new(format!(
                        "T={:.1} K, pH={:.2} | α {:.0}% · β {:.0}% · PPII {:.0}% · Coil {:.0}% | <|SSP|> {:.2}",
                        rep.temperature_kelvin,
                        rep.effective_ph,
                        rep.mean_p_alpha * 100.0,
                        rep.mean_p_beta * 100.0,
                        rep.mean_p_ppii * 100.0,
                        rep.mean_p_coil * 100.0,
                        rep.mean_abs_ssp
                    ))
                    .monospace()
                    .size(10.5)
                    .color(egui::Color32::from_rgb(175, 192, 218)),
                );

                ui.add_space(4.0);
                let avail_w = ui.available_width().max(220.0);
                let n = rep.residues.len().max(1);
                let col_w = (avail_w / (n as f32)).clamp(2.0, 14.0);
                let plot_w = (col_w * (n as f32)).max(avail_w);

                egui::ScrollArea::horizontal()
                    .id_salt("ssp_d2d_scroll_canvas")
                    .show(ui, |ui| {
                        let (rect, resp) = ui.allocate_exact_size(
                            egui::vec2(plot_w, 114.0),
                            egui::Sense::click_and_drag(),
                        );
                        let painter = ui.painter_at(rect);
                        painter.rect_filled(rect, 3.0, egui::Color32::from_rgb(14, 18, 26));

                        let ssp_top = rect.top() + 4.0;
                        let ssp_mid = rect.top() + 34.0;
                        let ssp_bot = rect.top() + 64.0;
                        let pop_top = rect.top() + 72.0;
                        let pop_bot = rect.bottom() - 4.0;
                        let pop_h = (pop_bot - pop_top).max(12.0);

                        painter.line_segment(
                            [egui::pos2(rect.left(), ssp_mid), egui::pos2(rect.right(), ssp_mid)],
                            egui::Stroke::new(1.0, egui::Color32::from_rgb(62, 74, 96)),
                        );

                        for (k, r) in rep.residues.iter().enumerate() {
                            let x0 = rect.left() + (k as f32) * col_w;
                            let x1 = (x0 + col_w - 0.8).max(x0 + 1.0);
                            if k == sel_k {
                                painter.rect_filled(
                                    egui::Rect::from_min_max(
                                        egui::pos2(x0, rect.top()),
                                        egui::pos2(x0 + col_w, rect.bottom()),
                                    ),
                                    0.0,
                                    egui::Color32::from_rgba_unmultiplied(120, 215, 255, 28),
                                );
                            }

                            if let Some(ssp) = r.ssp_score {
                                let clamped = (ssp as f32).clamp(-1.0, 1.0);
                                let half_h = (ssp_bot - ssp_mid).max(8.0);
                                let (y_min, y_max, col) = if clamped >= 0.0 {
                                    (
                                        ssp_mid - clamped * (ssp_mid - ssp_top),
                                        ssp_mid,
                                        egui::Color32::from_rgb(56, 189, 248),
                                    )
                                } else {
                                    (
                                        ssp_mid,
                                        ssp_mid + (-clamped) * half_h,
                                        egui::Color32::from_rgb(245, 158, 11),
                                    )
                                };
                                painter.rect_filled(
                                    egui::Rect::from_min_max(
                                        egui::pos2(x0, y_min),
                                        egui::pos2(x1, y_max.max(y_min + 1.0)),
                                    ),
                                    1.0,
                                    col,
                                );
                            }

                            // Stacked 4-state δ2D bar: Alpha (cyan), Beta (amber), PPII (violet), Coil (slate)
                            let mut y_cur = pop_bot;
                            for (frac, col) in [
                                (r.p_alpha as f32, egui::Color32::from_rgb(56, 189, 248)),
                                (r.p_beta as f32, egui::Color32::from_rgb(245, 158, 11)),
                                (r.p_ppii as f32, egui::Color32::from_rgb(168, 85, 247)),
                                (r.p_coil as f32, egui::Color32::from_rgb(71, 85, 105)),
                            ] {
                                let h = (frac.clamp(0.0, 1.0) * pop_h).max(0.0);
                                if h > 0.3 {
                                    painter.rect_filled(
                                        egui::Rect::from_min_max(
                                            egui::pos2(x0, (y_cur - h).max(pop_top)),
                                            egui::pos2(x1, y_cur),
                                        ),
                                        0.0,
                                        col,
                                    );
                                    y_cur = (y_cur - h).max(pop_top);
                                }
                            }
                        }

                        if resp.clicked() || resp.dragged() {
                            if let Some(pos) = resp.interact_pointer_pos() {
                                let idx = (((pos.x - rect.left()) / col_w).floor() as isize)
                                    .clamp(0, (n as isize) - 1) as usize;
                                self.select_residue_and_sync_strip(idx);
                            }
                        }
                    });

                if let Some(cur) = rep.residues.get(sel_k) {
                    let ssp_txt = cur
                        .ssp_score
                        .map(|v| format!("{v:+.3}"))
                        .unwrap_or_else(|| "n/a".to_string());
                    ui.label(
                        egui::RichText::new(format!(
                            "{} {} ({}) | SSP {} | α {:.0}% β {:.0}% PPII {:.0}% coil {:.0}%",
                            cur.residue,
                            cur.seq_id,
                            cur.dominant_state.label(),
                            ssp_txt,
                            cur.p_alpha * 100.0,
                            cur.p_beta * 100.0,
                            cur.p_ppii * 100.0,
                            cur.p_coil * 100.0,
                        ))
                        .monospace()
                        .size(10.5)
                        .color(egui::Color32::from_rgb(215, 232, 252)),
                    );
                }
            });
    }

    fn draw_right_inspector(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.strong("3D Strip Inspector");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .small_button("× Hide")
                    .on_hover_text(
                        "Hide Right Strip Inspector dock (reopen via View -> 3D Strips Dock)",
                    )
                    .clicked()
                {
                    self.show_strips_dock = false;
                    self.responsive_shell.tablet_dock = TabletActiveDock::Hidden;
                }
            });
        });
        ui.separator();

        // 1. Metrics summary card if reference truth is present
        if let Some(m) = &self.metrics {
            egui::Frame::group(ui.style())
                .fill(egui::Color32::from_rgb(22, 29, 40))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("Benchmark vs Truth").strong());
                        if let Some(ms) = self.last_solve_ms {
                            ui.label(
                                egui::RichText::new(format!("{ms:.0} ms"))
                                    .monospace()
                                    .color(egui::Color32::from_rgb(120, 225, 255)),
                            );
                        }
                    });
                    ui.columns(3, |cols| {
                        cols[0].vertical_centered(|ui| {
                            ui.label(egui::RichText::new("Accuracy").small().weak());
                            ui.label(
                                egui::RichText::new(format!("{:.1}%", m.accuracy() * 100.0))
                                    .strong()
                                    .size(16.0),
                            );
                        });
                        cols[1].vertical_centered(|ui| {
                            ui.label(egui::RichText::new("Recall").small().weak());
                            ui.label(
                                egui::RichText::new(format!("{:.1}%", m.recall() * 100.0))
                                    .strong()
                                    .size(16.0),
                            );
                        });
                        cols[2].vertical_centered(|ui| {
                            ui.label(egui::RichText::new("HC Prec (>=0.8)").small().weak());
                            ui.label(
                                egui::RichText::new(format!(
                                    "{:.1}% ({}/{})",
                                    m.hc_precision() * 100.0,
                                    m.hc_correct,
                                    m.hc_assigned
                                ))
                                .strong()
                                .size(15.0)
                                .color(egui::Color32::from_rgb(95, 240, 165)),
                            );
                        });
                    });
                });
            ui.add_space(6.0);
        }

        // 2. Strip View Mode Switch: Match-Finder (`st`), Sequential Walk, & Live 2D Sidechain Graph
        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(
                &mut self.use_match_finder_strips,
                true,
                "Match-Finder (i−1/i/i+1)",
            );
            ui.selectable_value(&mut self.use_match_finder_strips, false, "Walk & LLR");
            let sc_btn = egui::Button::new(
                egui::RichText::new("Sidechain 2D")
                    .size(12.0)
                    .color(egui::Color32::WHITE),
            )
            .fill(if self.show_sidechain_inspector {
                egui::Color32::from_rgb(20, 108, 130)
            } else {
                egui::Color32::from_rgb(34, 42, 58)
            });
            if ui
                .add(sc_btn)
                .on_hover_text("Inspect 2D IUPAC covalent atom-bond BMRB Z-score graph")
                .clicked()
            {
                self.show_sidechain_inspector = !self.show_sidechain_inspector;
                if self.show_sidechain_inspector && self.selected_res.is_none() {
                    self.selected_res = Some(0);
                }
            }
            let prop_btn = egui::Button::new(
                egui::RichText::new("SSP / δ2D")
                    .size(12.0)
                    .color(egui::Color32::WHITE),
            )
            .fill(if self.show_propensity_panel {
                egui::Color32::from_rgb(104, 48, 136)
            } else {
                egui::Color32::from_rgb(34, 42, 58)
            });
            if ui
                .add(prop_btn)
                .on_hover_text("Inspect Marsh SSP & Camilloni 4-state δ2D populations")
                .clicked()
            {
                self.show_propensity_panel = !self.show_propensity_panel;
                if self.show_propensity_panel {
                    self.refresh_propensity_report();
                    if self.selected_res.is_none() {
                        self.selected_res = Some(0);
                    }
                }
            }
            let relax_btn = egui::Button::new(
                egui::RichText::new("Relax / S²")
                    .size(12.0)
                    .color(egui::Color32::WHITE),
            )
            .fill(if self.show_relax_panel {
                egui::Color32::from_rgb(28, 122, 92)
            } else {
                egui::Color32::from_rgb(34, 42, 58)
            });
            if ui
                .add(relax_btn)
                .on_hover_text(
                    "Open Relaxation (R1/R2/hetNOE) & Lipari-Szabo Model-Free S² workbench",
                )
                .clicked()
            {
                self.show_relax_panel = !self.show_relax_panel;
                if self.show_relax_panel {
                    self.show_dynamics_panel = false;
                    self.show_stack1d_panel = false;
                }
            }
            let dyn_btn = egui::Button::new(
                egui::RichText::new("CEST / CPMG / CSP")
                    .size(12.0)
                    .color(egui::Color32::WHITE),
            )
            .fill(if self.show_dynamics_panel {
                egui::Color32::from_rgb(138, 86, 28)
            } else {
                egui::Color32::from_rgb(34, 42, 58)
            });
            if ui
                .add(dyn_btn)
                .on_hover_text(
                    "Open CEST/DEST saturation dips, CPMG dispersion & Ligand Titration CSP Kd Studio",
                )
                .clicked()
            {
                self.show_dynamics_panel = !self.show_dynamics_panel;
                if self.show_dynamics_panel {
                    self.show_relax_panel = false;
                    self.show_stack1d_panel = false;
                }
            }
            let stack1d_btn = egui::Button::new(
                egui::RichText::new("1D Stack / Kinetics")
                    .size(12.0)
                    .color(egui::Color32::WHITE),
            )
            .fill(if self.show_stack1d_panel {
                egui::Color32::from_rgb(56, 98, 158)
            } else {
                egui::Color32::from_rgb(34, 42, 58)
            });
            if ui
                .add(stack1d_btn)
                .on_hover_text(
                    "Open multi-spectrum 1D overlay, tilted stack, difference spectrum & kinetics ROI workbench",
                )
                .clicked()
            {
                self.show_stack1d_panel = !self.show_stack1d_panel;
                if self.show_stack1d_panel {
                    self.show_relax_panel = false;
                    self.show_dynamics_panel = false;
                    if self.stack1d_panel.traces.is_empty() {
                        let (traces, rois) = build_kinetics_demo_traces();
                        self.stack1d_panel = Stack1dState::with_traces(traces);
                        self.stack1d_panel.roi_regions = rois;
                    }
                }
            }
            if ui.small_button("Assign/Lock (at)").clicked() {
                self.show_assign_popover = true;
            }
        });

        if self.show_propensity_panel {
            ui.add_space(4.0);
            self.draw_propensity_inspector_drawer(ui);
            ui.add_space(6.0);
            ui.separator();
        }

        if self.show_sidechain_inspector {
            ui.add_space(4.0);
            self.draw_sidechain_inspector_drawer(ui);
            ui.add_space(6.0);
            ui.separator();
        }

        let Some(sel_s) = self.selected_ss.filter(|&s| s < self.systems.len()) else {
            ui.label("Click any HSQC peak or assigned residue to inspect 3D strips.");
            return;
        };

        if self.use_match_finder_strips {
            if let Some(p_mat) = self.p_mat.clone() {
                let u_clone = self.u_mat.clone();
                let assign_pos = self.assign_res.as_ref().map(|r| r.position.clone());
                let locks_snapshot = self.session.state.locked_residues.clone();
                let links_snapshot = self.session.state.confirmed_links.clone();
                let strip_actions = {
                    let tiler_ctx = StripTilerContext {
                        systems: &self.systems,
                        pair_mat: &p_mat,
                        u_mat: u_clone.as_ref(),
                        sequence: &self.sequence,
                        seq_ids: &self.seq_ids,
                        active_root: sel_s,
                        assignments: assign_pos.as_deref(),
                        locks: Some(&locks_snapshot),
                        confirmed_links: Some(&links_snapshot),
                        spectra_3d: StripSpectra3D {
                            aliphatic_3d: self.hncacb_3d.as_ref(),
                            carbonyl_3d: self.hnco_3d.as_ref(),
                        },
                    };
                    self.strip_tiler
                        .show(ui, &tiler_ctx, Some(&mut self.crosshair))
                };
                for act in strip_actions {
                    self.handle_strip_tiler_action(act);
                }
                ui.separator();
            }
        } else {
            ui.label(
                egui::RichText::new(
                    "Cyan = +Cα, Coral = −Cβ, Gold ◆ = (i−1), Dashed = sequential tie",
                )
                .small()
                .weak(),
            );
            let (pred_s, succ_s) = self.find_walk_neighbors(sel_s);
            let strip_list: Vec<(Option<usize>, &str)> = vec![
                (pred_s, "i − 1"),
                (Some(sel_s), "Selected (i)"),
                (succ_s, "i + 1"),
            ];
            let strip_h = 275.0f32;
            let (strip_rect, _) = ui.allocate_exact_size(
                egui::vec2(ui.available_width().max(240.0), strip_h),
                egui::Sense::hover(),
            );
            self.paint_sequential_strips(ui.painter_at(strip_rect), strip_rect, &strip_list);
            ui.add_space(6.0);
            ui.separator();
        }
        ui.add_space(8.0);
        ui.separator();

        // 3. Detailed Spin System Candidate & LLR / Connectivity Inspector
        egui::ScrollArea::vertical()
            .id_salt("ss_inspector_scroll")
            .show(ui, |ui| {
                let ss = &self.systems[sel_s];
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new(format!(
                            "Spin System #{sel_s}  (¹H {:.3}, ¹⁵N {:.2})",
                            ss.h, ss.n
                        ))
                        .strong(),
                    );
                    if let Some(res) = &self.assign_res {
                        let k = res.position[sel_s];
                        if k >= 0 && (k as usize) < self.sequence.len() {
                            let ku = k as usize;
                            ui.label(
                                egui::RichText::new(format!(
                                    "→ {}{} ({:.0}%)",
                                    self.sequence[ku],
                                    self.seq_ids[ku],
                                    res.confidence[sel_s] * 100.0
                                ))
                                .color(egui::Color32::from_rgb(95, 240, 165))
                                .strong(),
                            );
                        }
                    }
                });

                egui::Grid::new("cands_grid")
                    .striped(true)
                    .min_col_width(58.0)
                    .show(ui, |ui| {
                        ui.strong("Atom");
                        ui.strong("Offset");
                        ui.strong("Candidates (ppm × support)");
                        ui.end_row();

                        for atom in ["CA", "CB", "C"] {
                            for (off, label) in [(0, "i"), (-1, "i − 1")] {
                                if let Some(cands) = ss.cands.get(&(atom.to_string(), off)) {
                                    ui.monospace(atom);
                                    ui.monospace(label);
                                    let text = cands
                                        .iter()
                                        .map(|(v, sup)| format!("{v:.2} (×{sup})"))
                                        .collect::<Vec<_>>()
                                        .join(", ");
                                    ui.monospace(text);
                                    ui.end_row();
                                }
                            }
                        }
                    });

                if let Some(u_mat) = &self.u_mat {
                    ui.add_space(6.0);
                    ui.label(
                        egui::RichText::new("Top Residue Typing LLR U(s, k):")
                            .small()
                            .strong(),
                    );
                    let mut ranked: Vec<(usize, f64)> = (0..u_mat.cols)
                        .map(|k| (k, u_mat.get(sel_s, k)))
                        .filter(|&(_, v)| v > FORBID * 0.5)
                        .collect();
                    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
                    let top_txt = if ranked.is_empty() {
                        "0 observed carbons (excluded from backbone)".to_string()
                    } else {
                        ranked
                            .iter()
                            .take(5)
                            .map(|&(k, sc)| {
                                format!("{}{}: {:+.1}", self.sequence[k], self.seq_ids[k], sc)
                            })
                            .collect::<Vec<_>>()
                            .join("  |  ")
                    };
                    ui.monospace(top_txt);
                }

                if let Some(p_mat) = &self.p_mat {
                    ui.add_space(4.0);
                    ui.label(
                        egui::RichText::new("Best Sequential Successors P(s → t):")
                            .small()
                            .strong(),
                    );
                    let mut succs: Vec<(usize, f64)> = (0..p_mat.cols)
                        .filter(|&t| t != sel_s)
                        .map(|t| (t, p_mat.get(sel_s, t)))
                        .filter(|&(_, v)| v > -20.0 && v.abs() > 1e-6)
                        .collect();
                    succs.sort_by(|a, b| b.1.total_cmp(&a.1));
                    for (t, sc) in succs.into_iter().take(4) {
                        if ui
                            .small_button(format!("Jump → SS #{t} (P = {sc:+.2})"))
                            .clicked()
                        {
                            self.selected_ss = Some(t);
                        }
                    }
                }
            });
    }

    fn find_walk_neighbors(&self, sel_s: usize) -> (Option<usize>, Option<usize>) {
        if let Some(res) = &self.assign_res {
            let k = res.position[sel_s];
            if k >= 0 {
                let pred = res.position.iter().position(|&pk| pk == k - 1);
                let succ = res.position.iter().position(|&pk| pk == k + 1);
                return (pred, succ);
            }
        }
        if let Some(p_mat) = &self.p_mat {
            let pred = (0..p_mat.rows)
                .filter(|&a| a != sel_s)
                .max_by(|&a, &b| p_mat.get(a, sel_s).total_cmp(&p_mat.get(b, sel_s)))
                .filter(|&a| p_mat.get(a, sel_s) > 0.0);
            let succ = (0..p_mat.cols)
                .filter(|&b| b != sel_s)
                .max_by(|&a, &b| p_mat.get(sel_s, a).total_cmp(&p_mat.get(sel_s, b)))
                .filter(|&b| p_mat.get(sel_s, b) > 0.0);
            return (pred, succ);
        }
        (None, None)
    }

    fn paint_sequential_strips(
        &self,
        painter: egui::Painter,
        rect: egui::Rect,
        strips: &[(Option<usize>, &str)],
    ) {
        painter.rect_filled(rect, 4.0, egui::Color32::from_rgb(13, 17, 24));
        let c_top = 13.0f64;
        let c_bot = 74.0f64;
        let top_pad = 22.0f32;
        let bot_pad = 8.0f32;
        let left_axis_w = 34.0f32;

        let plot_top = rect.top() + top_pad;
        let plot_bot = rect.bottom() - bot_pad;
        let c_to_y = |c_ppm: f64| -> f32 {
            let t = ((c_ppm - c_top) / (c_bot - c_top)) as f32;
            plot_top + t.clamp(0.0, 1.0) * (plot_bot - plot_top)
        };

        // 13C ppm scale on left
        for c_tick in [15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 72.0] {
            let y = c_to_y(c_tick);
            painter.text(
                egui::pos2(rect.left() + 3.0, y),
                egui::Align2::LEFT_CENTER,
                format!("{c_tick:.0}"),
                egui::FontId::monospace(9.5),
                egui::Color32::from_rgb(145, 165, 190),
            );
            painter.line_segment(
                [
                    egui::pos2(rect.left() + left_axis_w, y),
                    egui::pos2(rect.right() - 4.0, y),
                ],
                egui::Stroke::new(0.5, egui::Color32::from_rgb(28, 35, 48)),
            );
        }

        let total_w = (rect.width() - left_axis_w - 6.0).max(90.0);
        let col_w = total_w / (strips.len() as f32);

        let mut col_centers = Vec::with_capacity(strips.len());

        for (idx, &(ss_opt, role)) in strips.iter().enumerate() {
            let x0 = rect.left() + left_axis_w + (idx as f32) * col_w + 3.0;
            let x1 = x0 + col_w - 6.0;
            let cx = 0.5 * (x0 + x1);
            col_centers.push(cx);

            let strip_box =
                egui::Rect::from_min_max(egui::pos2(x0, plot_top), egui::pos2(x1, plot_bot));
            painter.rect_stroke(
                strip_box,
                2.0,
                egui::Stroke::new(
                    if idx == 1 { 1.5 } else { 1.0 },
                    if idx == 1 {
                        egui::Color32::from_rgb(95, 200, 255)
                    } else {
                        egui::Color32::from_rgb(50, 62, 82)
                    },
                ),
                egui::StrokeKind::Inside,
            );

            let Some(s) = ss_opt else {
                painter.text(
                    egui::pos2(cx, rect.top() + 10.0),
                    egui::Align2::CENTER_CENTER,
                    format!("{role}: —"),
                    egui::FontId::monospace(10.0),
                    egui::Color32::GRAY,
                );
                continue;
            };
            let ss = &self.systems[s];

            // Header title (assigned residue or SS#)
            let title = self
                .assign_res
                .as_ref()
                .and_then(|r| (r.position[s] >= 0).then(|| r.position[s] as usize))
                .map(|k| {
                    format!(
                        "{}{} (#{s})",
                        three_to_one(&self.sequence[k]),
                        self.seq_ids[k]
                    )
                })
                .unwrap_or_else(|| format!("SS #{s}"));
            painter.text(
                egui::pos2(cx, rect.top() + 10.0),
                egui::Align2::CENTER_CENTER,
                title,
                egui::FontId::monospace(10.5),
                egui::Color32::from_rgb(225, 238, 255),
            );

            // Render 3D HNCACB strip slice intensity heatmap if available
            if let Some(s3) = &self.hncacb_3d {
                let slice = s3.extract_strip(ss.h, ss.n, 0.04);
                let max_v = slice.max_abs().max(1.0);
                for ic in 0..slice.ny() {
                    let c_ppm = slice.axis_y.index_to_ppm(ic as f64);
                    if !(c_top..=c_bot).contains(&c_ppm) {
                        continue;
                    }
                    let y = c_to_y(c_ppm);
                    let mid_x = slice.nx() / 2;
                    let val = slice.get(mid_x, ic);
                    let norm = (val.abs() / max_v).clamp(0.0, 1.0);
                    if norm > 0.14 {
                        let col = if val >= 0.0 {
                            egui::Color32::from_rgba_unmultiplied(
                                40,
                                195,
                                255,
                                (norm * 130.0) as u8,
                            )
                        } else {
                            egui::Color32::from_rgba_unmultiplied(
                                255,
                                95,
                                105,
                                (norm * 130.0) as u8,
                            )
                        };
                        let half_bar = norm * (strip_box.width() * 0.36);
                        painter.line_segment(
                            [egui::pos2(cx - half_bar, y), egui::pos2(cx + half_bar, y)],
                            egui::Stroke::new(2.2, col),
                        );
                    }
                }
            }

            // Plot resolved spin-system candidates in strip: (i) solid circles, (i-1) diamonds
            for atom in ["CA", "CB"] {
                let base_col = if atom == "CA" {
                    egui::Color32::from_rgb(80, 230, 255)
                } else {
                    egui::Color32::from_rgb(255, 115, 120)
                };
                if let Some(cands_i) = ss.cands.get(&(atom.to_string(), 0)) {
                    if let Some(&(ppm, sup)) = cands_i.first() {
                        if sup > 0 {
                            let y = c_to_y(ppm);
                            painter.circle_filled(egui::pos2(cx - 8.0, y), 4.2, base_col);
                            painter.text(
                                egui::pos2(cx - 14.0, y),
                                egui::Align2::RIGHT_CENTER,
                                format!("{ppm:.1}"),
                                egui::FontId::monospace(8.8),
                                base_col,
                            );
                        }
                    }
                }
                if let Some(cands_im1) = ss.cands.get(&(atom.to_string(), -1)) {
                    if let Some(&(ppm, _)) = cands_im1.first() {
                        let y = c_to_y(ppm);
                        let d_col = egui::Color32::from_rgb(250, 205, 90);
                        let pts = vec![
                            egui::pos2(cx + 8.0, y - 4.2),
                            egui::pos2(cx + 12.2, y),
                            egui::pos2(cx + 8.0, y + 4.2),
                            egui::pos2(cx + 3.8, y),
                        ];
                        painter.add(egui::Shape::convex_polygon(pts, d_col, egui::Stroke::NONE));
                    }
                }
            }
        }

        // Draw sequential connectivity tie-lines between strip[w] (offset 0) and strip[w+1] (offset -1)
        for w in 0..strips.len().saturating_sub(1) {
            let (Some(sa), Some(sb)) = (strips[w].0, strips[w + 1].0) else {
                continue;
            };
            for atom in ["CA", "CB"] {
                let Some(va) = self.systems[sa]
                    .cands
                    .get(&(atom.to_string(), 0))
                    .and_then(|c| c.first())
                    .filter(|c| c.1 > 0)
                    .map(|c| c.0)
                else {
                    continue;
                };
                let Some(vb) = self.systems[sb]
                    .cands
                    .get(&(atom.to_string(), -1))
                    .and_then(|c| c.first())
                    .map(|c| c.0)
                else {
                    continue;
                };
                if (va - vb).abs() <= 0.45 {
                    let ya = c_to_y(va);
                    let yb = c_to_y(vb);
                    let x_start = col_centers[w] - 4.0;
                    let x_end = col_centers[w + 1] + 4.0;
                    let col = if atom == "CA" {
                        egui::Color32::from_rgba_unmultiplied(110, 245, 255, 210)
                    } else {
                        egui::Color32::from_rgba_unmultiplied(255, 145, 145, 210)
                    };
                    // Dashed horizontal connector
                    let segs = 9;
                    for j in (0..segs).step_by(2) {
                        let t0 = (j as f32) / (segs as f32);
                        let t1 = ((j + 1) as f32) / (segs as f32);
                        painter.line_segment(
                            [
                                egui::pos2(x_start + t0 * (x_end - x_start), ya + t0 * (yb - ya)),
                                egui::pos2(x_start + t1 * (x_end - x_start), ya + t1 * (yb - ya)),
                            ],
                            egui::Stroke::new(1.5, col),
                        );
                    }
                }
            }
        }
    }
    fn compute_workspace_readiness(&self) -> WorkspaceReadiness {
        let spectra_2d_count = usize::from(self.hsqc_spec.is_some()) + self.overlay_layers.len();
        let spectra_3d_count = usize::from(self.hncacb_3d.is_some())
            + usize::from(self.hnco_3d.is_some())
            + self.extra_3d_count;
        let assigned_residues_count = self.pred_conf.len();
        let high_confidence_count = self.pred_conf.values().filter(|&&c| c >= 0.80).count();
        let qc_verdict = self
            .qc_report
            .as_ref()
            .map(|q| q.verdict)
            .or(self.session.state.qc_verdict);
        let qc_summary = self.qc_report.as_ref().map(|q| {
            format!(
                "{:.0}% 3D support, scale {:.2}",
                q.hsqc_3d_support_frac * 100.0,
                q.confidence_scale
            )
        });

        WorkspaceReadiness {
            sequence_loaded: !self.sequence.is_empty(),
            sequence_len: self.sequence.len(),
            raw_fids_count: self.raw_fids_total,
            unprocessed_fids_count: self.unprocessed_fids_remaining,
            spectra_2d_count,
            spectra_3d_count,
            picked_peaks_count: self.peaks.len(),
            spin_systems_count: self.systems.len(),
            assigned_residues_count,
            high_confidence_count,
            locked_residues_count: self.session.state.locked_residues.len(),
            has_structure_or_dynamics: self.has_structure_or_dynamics,
            exported_in_session: self.exported_in_session,
            qc_verdict,
            qc_summary,
        }
    }

    fn handle_strip_tiler_action(&mut self, act: StripTilerAction) {
        match act {
            StripTilerAction::SelectRoot { spin_system_idx } => {
                if let Some(ss) = self.systems.get(spin_system_idx) {
                    self.selected_ss = Some(spin_system_idx);
                    self.crosshair.snap_to(ss.h, ss.n, None);
                    if let Some(res) = &self.assign_res {
                        if res.position.get(spin_system_idx).copied().unwrap_or(-1) >= 0 {
                            let k = res.position[spin_system_idx] as usize;
                            self.selected_res = Some(k);
                            self.assign_target_res = k;
                            self.mol3d_panel.highlight_sequence_residue(Some(k), true);
                        }
                    }
                }
            }
            StripTilerAction::ConfirmSequentialLink { pred_idx, succ_idx } => {
                let pair = (pred_idx, succ_idx);
                if let Some(pos) = self
                    .session
                    .state
                    .confirmed_links
                    .iter()
                    .position(|&lk| lk == pair)
                {
                    self.session.state.confirmed_links.remove(pos);
                    self.session.dirty = true;
                    self.status_msg =
                        format!("Removed sequential link SS #{pred_idx} → SS #{succ_idx}");
                } else {
                    self.session.state.confirmed_links.push(pair);
                    self.session.dirty = true;
                    self.status_msg = format!(
                        "Confirmed sequential link SS #{pred_idx} (i−1) → SS #{succ_idx} (i)"
                    );
                }
            }
            StripTilerAction::LockToResidue {
                spin_system_idx,
                residue_seq_idx,
            } => {
                if residue_seq_idx < self.sequence.len() {
                    let already = self
                        .session
                        .state
                        .locked_residue_for_spin_system(spin_system_idx)
                        == Some(residue_seq_idx);
                    let msg = self.session.execute(WorkbenchCommand::LockResidue {
                        spin_system: spin_system_idx,
                        residue_seq_idx,
                        locked: !already,
                    });
                    self.sync_app_from_session();
                    self.status_msg = msg;
                }
            }
        }
    }

    fn snap_selected_peak_parabolic(&mut self) {
        let Some(sel_s) = self.selected_ss.filter(|&s| s < self.systems.len()) else {
            self.status_msg = "Select a peak / spin system first (`pc`)".to_string();
            return;
        };
        let ss = &self.systems[sel_s];
        let Some((pk_idx, pk)) = self
            .peaks
            .iter()
            .enumerate()
            .filter(|(_, p)| p.exp == "HSQC")
            .min_by(|(_, a), (_, b)| {
                let da = ((a.h - ss.h) / 0.03).powi(2) + ((a.n - ss.n) / 0.3).powi(2);
                let db = ((b.h - ss.h) / 0.03).powi(2) + ((b.n - ss.n) / 0.3).powi(2);
                da.total_cmp(&db)
            })
            .map(|(i, p)| (i, p.clone()))
        else {
            self.status_msg = "No HSQC peak found near active spin system".to_string();
            return;
        };

        let (mut fitted_h, mut fitted_n, mut new_height) = (pk.h, pk.n, pk.height);
        if let Some(spec) = &self.hsqc_spec {
            let ix = spec
                .axis_x
                .ppm_to_index(pk.h)
                .round()
                .clamp(1.0, (spec.nx().saturating_sub(2)) as f64) as usize;
            let iy = spec
                .axis_y
                .ppm_to_index(pk.n)
                .round()
                .clamp(1.0, (spec.ny().saturating_sub(2)) as f64) as usize;
            // Local 3x3 extremum search + 1D parabolic sub-voxel interpolation
            let mut best_ix = ix;
            let mut best_iy = iy;
            let mut best_val = spec.get(ix, iy).abs();
            for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    let cx = (ix as isize + dx).clamp(1, (spec.nx().saturating_sub(2)) as isize)
                        as usize;
                    let cy = (iy as isize + dy).clamp(1, (spec.ny().saturating_sub(2)) as isize)
                        as usize;
                    let v = spec.get(cx, cy).abs();
                    if v > best_val {
                        best_val = v;
                        best_ix = cx;
                        best_iy = cy;
                    }
                }
            }
            let xm = spec.get(best_ix - 1, best_iy) as f64;
            let x0 = spec.get(best_ix, best_iy) as f64;
            let xp = spec.get(best_ix + 1, best_iy) as f64;
            let denom_x = xm - 2.0 * x0 + xp;
            let dx = if denom_x.abs() > 1e-9 {
                (0.5 * (xm - xp) / denom_x).clamp(-0.5, 0.5)
            } else {
                0.0
            };

            let ym = spec.get(best_ix, best_iy - 1) as f64;
            let yp = spec.get(best_ix, best_iy + 1) as f64;
            let denom_y = ym - 2.0 * x0 + yp;
            let dy = if denom_y.abs() > 1e-9 {
                (0.5 * (ym - yp) / denom_y).clamp(-0.5, 0.5)
            } else {
                0.0
            };

            fitted_h = spec.axis_x.index_to_ppm(best_ix as f64 + dx);
            fitted_n = spec.axis_y.index_to_ppm(best_iy as f64 + dy);
            new_height = Some(x0);
        }

        let msg = self.session.execute(WorkbenchCommand::RecenterPeak {
            index: pk_idx,
            fitted_h,
            fitted_n,
            fitted_c: None,
            new_height,
        });
        self.sync_app_from_session();
        self.crosshair.snap_to(fitted_h, fitted_n, None);
        self.status_msg = msg;
    }

    fn save_session_nef_action(&mut self, ctx: &egui::Context) {
        self.session.state.sequence = self.sequence.clone();
        self.session.state.seq_ids = self.seq_ids.clone();
        self.session.state.peaks = self.peaks.clone();
        self.session.state.view_bounds_ppm = Some([
            self.view_h_left as f64,
            self.view_h_right as f64,
            self.view_n_top as f64,
            self.view_n_bottom as f64,
        ]);
        let mut hsqc_params = SessionContourParams::new("HSQC");
        hsqc_params.base_level = self.contour_base;
        hsqc_params.level_factor = self.contour_factor;
        hsqc_params.num_levels = self.contour_levels;
        hsqc_params.show_negative = self.show_negative;
        hsqc_params.ppm_nudge = self.primary_ppm_nudge;
        self.session.state.contour_layers = vec![hsqc_params];

        let nef_text = self.session.save_to_nef_string();
        self.export_text_buf = nef_text.clone();
        self.exported_in_session = true;

        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = ctx;
            let p = Path::new(self.export_prefix.trim()).with_extension("nef");
            match self.session.save_to_path(&p) {
                Ok(msg) => self.status_msg = msg,
                Err(e) => self.status_msg = format!("NEF session save failed: {e}"),
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            let fname = default_session_nef_filename(&self.session.state.protein_name);
            let _ = trigger_browser_nef_download(&fname, nef_text.as_bytes());
            ctx.copy_text(nef_text);
            self.status_msg = format!("Downloaded lossless NEF session '{fname}'");
        }
    }

    fn execute_ingest_plan(&mut self, plan: IngestExecutionPlan, run_full_pipeline: bool) {
        self.prepare_workbook_for_user_append();
        self.workspace_mode = WorkspaceMode::ProteinNmr;
        self.show_stack1d_panel = false;
        self.show_relax_panel = false;
        self.show_dynamics_panel = false;

        let has_user_seq = !plan.sequence_three_letter.is_empty();
        self.clear_stale_protein_demo_state(has_user_seq, false);
        if has_user_seq {
            self.sequence = plan.sequence_three_letter.clone();
            self.seq_ids = (1..=(self.sequence.len() as i32)).collect();
            self.octet_app
                .set_loaded_protein_sequence(self.sequence.clone());
        }
        self.raw_fids_total = plan
            .selected_experiments
            .iter()
            .filter(|s| s.experiment.data_state == DataState::RawFid)
            .count();
        self.unprocessed_fids_remaining = self.raw_fids_total;

        self.axis_rescue_notices.clear();
        self.axis_rescue_audits.clear();
        self.peak_2d_labels.clear();
        self.water_undo_2d = None;
        self.bare_bruker_hint = None;

        let mut ingested_recipes: BTreeMap<String, ProcessingRecipe> = BTreeMap::new();
        let mut load_errors: Vec<String> = Vec::new();
        let mut primary_cls: Option<PulseSequenceClass> = None;
        {
            let mut loaded_2d_list: Vec<(
                String,
                PulseSequenceClass,
                Option<&'static str>,
                Spectrum2D,
            )> = Vec::new();
            let mut cubes_3d: Vec<(String, PulseSequenceClass, Spectrum3D)> = Vec::new();
            let mut processed_fids = 0usize;

            let find_mem_file = |exp_rel: &str, child: &str| -> Option<&[u8]> {
                let prefix = exp_rel.trim_matches('/').trim_start_matches("./");
                let target = if prefix.is_empty() || prefix == "." {
                    child.to_string()
                } else {
                    format!("{prefix}/{child}")
                };
                if let Some(b) = plan.in_memory_files.get(&target) {
                    return Some(b.as_slice());
                }
                let suffix = format!("/{target}");
                plan.in_memory_files
                    .iter()
                    .find(|(k, _)| {
                        k.eq_ignore_ascii_case(&target)
                            || k.to_ascii_lowercase()
                                .ends_with(&suffix.to_ascii_lowercase())
                    })
                    .map(|(_, v)| v.as_slice())
            };

            let find_mem_exact = |rel_path: &str| -> Option<&[u8]> {
                let clean = rel_path.trim_matches('/').trim_start_matches("./");
                if let Some(b) = plan.in_memory_files.get(clean) {
                    return Some(b.as_slice());
                }
                let suffix = format!("/{clean}");
                plan.in_memory_files
                    .iter()
                    .find(|(k, _)| {
                        k.eq_ignore_ascii_case(clean)
                            || k.to_ascii_lowercase()
                                .ends_with(&suffix.to_ascii_lowercase())
                    })
                    .map(|(_, v)| v.as_slice())
            };

            // Discover all Bruker `pdata/<procno>` sub-prefixes present in `in_memory_files` (`1` first, then numeric ascending)
            let find_mem_pdata_prefixes = |exp_rel: &str, bin_file: &str| -> Vec<String> {
                let exp_clean = exp_rel.trim_matches('/').trim_start_matches("./");
                let mut procnos: Vec<(u32, String)> = Vec::new();
                let mut has_direct = false;
                let suffix_bin = format!("/{bin_file}").to_ascii_lowercase();
                for key in plan.in_memory_files.keys() {
                    let k_clean = key.trim_matches('/').trim_start_matches("./");
                    let tail_opt = if exp_clean.is_empty() || exp_clean == "." {
                        Some(k_clean)
                    } else if let Some(rest) = k_clean.strip_prefix(&format!("{exp_clean}/")) {
                        Some(rest)
                    } else {
                        let marker = format!("/{exp_clean}/");
                        k_clean
                            .find(&marker)
                            .map(|idx| &k_clean[idx + marker.len()..])
                    };
                    let Some(tail) = tail_opt else {
                        continue;
                    };
                    if tail.eq_ignore_ascii_case(bin_file) {
                        has_direct = true;
                        continue;
                    }
                    let tail_low = tail.to_ascii_lowercase();
                    if tail_low.starts_with("pdata/") && tail_low.ends_with(&suffix_bin) {
                        let mid = &tail[6..tail.len() - suffix_bin.len()];
                        if !mid.contains('/') {
                            let rank = if mid == "1" {
                                0
                            } else {
                                mid.parse::<u32>().unwrap_or(u32::MAX - 1).saturating_add(1)
                            };
                            let pfx = format!("pdata/{mid}");
                            if !procnos.iter().any(|(_, existing)| existing == &pfx) {
                                procnos.push((rank, pfx));
                            }
                        }
                    }
                }
                procnos.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
                let mut out: Vec<String> = procnos.into_iter().map(|(_, p)| p).collect();
                if out.is_empty() {
                    out.push("pdata/1".to_string());
                }
                if has_direct || !out.iter().any(|p| p.is_empty()) {
                    out.push(String::new());
                }
                out
            };

            for item in &plan.selected_experiments {
                let exp_token = item.solver_exp_name.unwrap_or("HNCACB");
                let cls = item.effective_class;
                let rel = &item.experiment.relative_path;
                let disp = &item.experiment.display_name;

                match item.experiment.dimensionality {
                    ExperimentDimensionality::Dim2D | ExperimentDimensionality::Pseudo3D => {
                        let mut loaded_2d: Option<Spectrum2D> = None;
                        let mut item_err: Option<String> = None;
                        #[cfg(not(target_arch = "wasm32"))]
                        if let Some(ref path) = item.resolved_path {
                            if path.exists() {
                                if item.experiment.data_state == DataState::RawFid {
                                    match process_fid_dir_2d(path, false, false) {
                                        Ok((s2, mut recipe)) => {
                                            recipe.enable_axis_rescue = self.enable_axis_rescue;
                                            processed_fids += 1;
                                            ingested_recipes.insert(disp.clone(), recipe);
                                            loaded_2d = Some(s2);
                                        }
                                        Err(e) => item_err = Some(e),
                                    }
                                } else {
                                    let res_2d = match item.experiment.vendor {
                                        ExperimentVendor::SparkyUcsf => read_ucsf_2d(path),
                                        ExperimentVendor::NmrPipe => read_nmrpipe_2d(path),
                                        ExperimentVendor::Bruker => {
                                            let pdata = resolve_bruker_pdata_dir(path, "2rr");
                                            read_bruker_pdata_2d(&pdata)
                                        }
                                        _ => Err("Unsupported 2D vendor format".to_string()),
                                    };
                                    match res_2d {
                                        Ok(s2) => loaded_2d = Some(s2),
                                        Err(e) => {
                                            if item.experiment.data_state == DataState::Both
                                                && item.experiment.vendor
                                                    == ExperimentVendor::Bruker
                                            {
                                                match process_fid_dir_2d(path, false, false) {
                                                    Ok((s2, mut recipe)) => {
                                                        recipe.enable_axis_rescue =
                                                            self.enable_axis_rescue;
                                                        processed_fids += 1;
                                                        ingested_recipes
                                                            .insert(disp.clone(), recipe);
                                                        loaded_2d = Some(s2);
                                                    }
                                                    Err(_) => item_err = Some(e),
                                                }
                                            } else {
                                                item_err = Some(e);
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // Cross-platform in-memory loader (`wasm32` browser + `.zip`/`.tar.gz` archives)
                        if loaded_2d.is_none() && !plan.in_memory_files.is_empty() {
                            let try_fid_mem =
                                |ingested_recipes: &mut BTreeMap<String, ProcessingRecipe>,
                                 processed_fids: &mut usize|
                                 -> Result<Spectrum2D, String> {
                                    let acqus_b = find_mem_file(rel, "acqus")
                                        .or_else(|| find_mem_file(rel, "acqu"))
                                        .ok_or_else(|| "missing acqus".to_string())?;
                                    let acqu2s_b = find_mem_file(rel, "acqu2s")
                                        .or_else(|| find_mem_file(rel, "acqu2"))
                                        .ok_or_else(|| "missing acqu2s".to_string())?;
                                    let ser_b = find_mem_file(rel, "ser")
                                        .ok_or_else(|| "missing ser".to_string())?;
                                    let acqus_s = String::from_utf8_lossy(acqus_b);
                                    let acqu_s = find_mem_file(rel, "acqu")
                                        .map(|b| String::from_utf8_lossy(b).into_owned());
                                    let acqu2s_s = String::from_utf8_lossy(acqu2s_b);
                                    let acqu2_s = find_mem_file(rel, "acqu2")
                                        .map(|b| String::from_utf8_lossy(b).into_owned());
                                    let procs_s = find_mem_file(rel, "pdata/1/procs")
                                        .or_else(|| find_mem_file(rel, "procs"))
                                        .map(|b| String::from_utf8_lossy(b).into_owned());
                                    let proc2s_s = find_mem_file(rel, "pdata/1/proc2s")
                                        .or_else(|| find_mem_file(rel, "proc2s"))
                                        .map(|b| String::from_utf8_lossy(b).into_owned());
                                    let pp_s = find_mem_file(rel, "pulseprogram")
                                        .or_else(|| find_mem_file(rel, "pulseprogram.precomp"))
                                        .map(|b| String::from_utf8_lossy(b).into_owned());
                                    let nl_s = find_mem_file(rel, "nuslist")
                                        .map(|b| String::from_utf8_lossy(b).into_owned());
                                    let fid2d = read_bruker_fid_2d_from_bytes(
                                        disp,
                                        &acqus_s,
                                        acqu_s.as_deref(),
                                        &acqu2s_s,
                                        acqu2_s.as_deref(),
                                        procs_s.as_deref(),
                                        proc2s_s.as_deref(),
                                        pp_s.as_deref(),
                                        nl_s.as_deref(),
                                        ser_b,
                                        false,
                                    )?;
                                    let mut recipe = ProcessingRecipe::from_fid_2d(&fid2d);
                                    recipe.enable_axis_rescue = self.enable_axis_rescue;
                                    let s2 = process_fid_2d_mut(&fid2d, &mut recipe)?;
                                    *processed_fids += 1;
                                    ingested_recipes.insert(disp.clone(), recipe);
                                    Ok(s2)
                                };

                            if item.experiment.data_state == DataState::RawFid {
                                match try_fid_mem(&mut ingested_recipes, &mut processed_fids) {
                                    Ok(s2) => loaded_2d = Some(s2),
                                    Err(e) => item_err = Some(e),
                                }
                            } else {
                                let mem_res: Result<Spectrum2D, String> =
                                    match item.experiment.vendor {
                                        ExperimentVendor::SparkyUcsf => find_mem_exact(rel)
                                            .ok_or_else(|| format!("missing bytes for '{rel}'"))
                                            .and_then(|b| read_ucsf_2d_from_bytes(disp, b)),
                                        ExperimentVendor::NmrPipe => find_mem_exact(rel)
                                            .ok_or_else(|| format!("missing bytes for '{rel}'"))
                                            .and_then(|b| read_nmrpipe_2d_from_bytes(disp, b)),
                                        ExperimentVendor::Bruker => {
                                            let mut last_e =
                                                "missing Bruker 2D pdata (procs/proc2s/2rr)"
                                                    .to_string();
                                            let mut found_s2 = None;
                                            for pfx in find_mem_pdata_prefixes(rel, "2rr") {
                                                let k_p1 = if pfx.is_empty() {
                                                    "procs".to_string()
                                                } else {
                                                    format!("{pfx}/procs")
                                                };
                                                let k_p2 = if pfx.is_empty() {
                                                    "proc2s".to_string()
                                                } else {
                                                    format!("{pfx}/proc2s")
                                                };
                                                let k_rr = if pfx.is_empty() {
                                                    "2rr".to_string()
                                                } else {
                                                    format!("{pfx}/2rr")
                                                };
                                                if let (Some(p1), Some(p2), Some(rr)) = (
                                                    find_mem_file(rel, &k_p1),
                                                    find_mem_file(rel, &k_p2),
                                                    find_mem_file(rel, &k_rr),
                                                ) {
                                                    match read_bruker_pdata_2d_from_bytes(
                                                        disp,
                                                        &String::from_utf8_lossy(p1),
                                                        &String::from_utf8_lossy(p2),
                                                        rr,
                                                    ) {
                                                        Ok(s2) => {
                                                            found_s2 = Some(s2);
                                                            break;
                                                        }
                                                        Err(e) => last_e = e,
                                                    }
                                                }
                                            }
                                            found_s2.ok_or(last_e)
                                        }
                                        _ => Err("Unsupported 2D vendor format".to_string()),
                                    };
                                match mem_res {
                                    Ok(s2) => loaded_2d = Some(s2),
                                    Err(e) => {
                                        if item.experiment.data_state == DataState::Both
                                            && item.experiment.vendor == ExperimentVendor::Bruker
                                        {
                                            if let Ok(s2) = try_fid_mem(
                                                &mut ingested_recipes,
                                                &mut processed_fids,
                                            ) {
                                                loaded_2d = Some(s2);
                                            } else {
                                                item_err = Some(e);
                                            }
                                        } else {
                                            item_err = Some(e);
                                        }
                                    }
                                }
                            }
                        }

                        if let Some(s2) = loaded_2d {
                            loaded_2d_list.push((disp.clone(), cls, item.solver_exp_name, s2));
                        } else if item.experiment.dimensionality == ExperimentDimensionality::Dim2D
                            || plan.selected_experiments.len() == 1
                        {
                            if let Some(e) = item_err {
                                load_errors.push(format!("{disp}: {e}"));
                            } else {
                                load_errors.push(format!("{disp}: spectrum binary not found"));
                            }
                        }
                    }
                    ExperimentDimensionality::Dim3D => {
                        let mut loaded_3d: Option<Spectrum3D> = None;
                        let mut item_err: Option<String> = None;
                        #[cfg(not(target_arch = "wasm32"))]
                        if let Some(ref path) = item.resolved_path {
                            if path.exists() {
                                if item.experiment.data_state == DataState::RawFid {
                                    match process_fid_dir_3d(path, exp_token, false, false) {
                                        Ok((s3, mut recipe)) => {
                                            recipe.enable_axis_rescue = self.enable_axis_rescue;
                                            processed_fids += 1;
                                            ingested_recipes.insert(disp.clone(), recipe);
                                            loaded_3d = Some(s3);
                                        }
                                        Err(e) => item_err = Some(e),
                                    }
                                } else {
                                    let res_3d = match item.experiment.vendor {
                                        ExperimentVendor::SparkyUcsf => {
                                            read_ucsf_3d(path, exp_token)
                                        }
                                        ExperimentVendor::NmrPipe => {
                                            read_nmrpipe_3d(path, exp_token)
                                        }
                                        ExperimentVendor::Bruker => {
                                            let pdata = resolve_bruker_pdata_dir(path, "3rrr");
                                            read_bruker_pdata_3d(&pdata, exp_token)
                                        }
                                        _ => Err("Unsupported 3D vendor format".to_string()),
                                    };
                                    match res_3d {
                                        Ok(s3) => loaded_3d = Some(s3),
                                        Err(e) => {
                                            if item.experiment.data_state == DataState::Both
                                                && item.experiment.vendor
                                                    == ExperimentVendor::Bruker
                                            {
                                                if let Ok((s3, mut recipe)) = process_fid_dir_3d(
                                                    path, exp_token, false, false,
                                                ) {
                                                    recipe.enable_axis_rescue =
                                                        self.enable_axis_rescue;
                                                    processed_fids += 1;
                                                    ingested_recipes.insert(disp.clone(), recipe);
                                                    loaded_3d = Some(s3);
                                                } else {
                                                    item_err = Some(e);
                                                }
                                            } else {
                                                item_err = Some(e);
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        if loaded_3d.is_none() && !plan.in_memory_files.is_empty() {
                            let mem_res3: Result<Spectrum3D, String> = match item.experiment.vendor
                            {
                                ExperimentVendor::SparkyUcsf => find_mem_exact(rel)
                                    .ok_or_else(|| format!("missing bytes for '{rel}'"))
                                    .and_then(|b| read_ucsf_3d_from_bytes(disp, exp_token, b)),
                                ExperimentVendor::NmrPipe => find_mem_exact(rel)
                                    .ok_or_else(|| format!("missing bytes for '{rel}'"))
                                    .and_then(|b| read_nmrpipe_3d_from_bytes(disp, exp_token, b)),
                                ExperimentVendor::Bruker => {
                                    let mut last_e =
                                        "missing Bruker 3D pdata (procs/proc2s/proc3s/3rrr)"
                                            .to_string();
                                    let mut found_s3 = None;
                                    for pfx in find_mem_pdata_prefixes(rel, "3rrr") {
                                        let k1 = if pfx.is_empty() {
                                            "procs".to_string()
                                        } else {
                                            format!("{pfx}/procs")
                                        };
                                        let k2 = if pfx.is_empty() {
                                            "proc2s".to_string()
                                        } else {
                                            format!("{pfx}/proc2s")
                                        };
                                        let k3 = if pfx.is_empty() {
                                            "proc3s".to_string()
                                        } else {
                                            format!("{pfx}/proc3s")
                                        };
                                        let kr = if pfx.is_empty() {
                                            "3rrr".to_string()
                                        } else {
                                            format!("{pfx}/3rrr")
                                        };
                                        if let (Some(b1), Some(b2), Some(b3), Some(br)) = (
                                            find_mem_file(rel, &k1),
                                            find_mem_file(rel, &k2),
                                            find_mem_file(rel, &k3),
                                            find_mem_file(rel, &kr),
                                        ) {
                                            match read_bruker_pdata_3d_from_bytes(
                                                disp,
                                                exp_token,
                                                &String::from_utf8_lossy(b1),
                                                &String::from_utf8_lossy(b2),
                                                &String::from_utf8_lossy(b3),
                                                br,
                                            ) {
                                                Ok(s3) => {
                                                    found_s3 = Some(s3);
                                                    break;
                                                }
                                                Err(e) => last_e = e,
                                            }
                                        }
                                    }
                                    found_s3.ok_or(last_e)
                                }
                                _ => Err("Unsupported 3D vendor format".to_string()),
                            };
                            match mem_res3 {
                                Ok(s3) => loaded_3d = Some(s3),
                                Err(e) => item_err = Some(e),
                            }
                        }

                        if let Some(s3) = loaded_3d {
                            cubes_3d.push((disp.clone(), cls, s3));
                        } else if let Some(e) = item_err {
                            load_errors.push(format!("{disp}: {e}"));
                        } else {
                            load_errors.push(format!("{disp}: 3D cube binary not found"));
                        }
                    }
                    _ => {}
                }
            }

            self.unprocessed_fids_remaining = self.raw_fids_total.saturating_sub(processed_fids);

            // Promote primary_2d by priority: N15Hsqc -> Con2D -> C13Hsqc/C13Hmqc -> first loaded Dim2D
            let mut caco_2d_specs: Vec<Spectrum2D> = Vec::new();
            if !loaded_2d_list.is_empty() {
                let rank_2d = |cls: PulseSequenceClass, tok: Option<&'static str>| -> u8 {
                    if cls == PulseSequenceClass::N15Hsqc || tok == Some("HSQC") {
                        0
                    } else if cls == PulseSequenceClass::Con2D {
                        1
                    } else if matches!(
                        cls,
                        PulseSequenceClass::C13Hsqc | PulseSequenceClass::C13Hmqc
                    ) {
                        2
                    } else {
                        3
                    }
                };
                let best_idx = loaded_2d_list
                    .iter()
                    .enumerate()
                    .min_by_key(|&(idx, (_, cls, tok, _))| (rank_2d(*cls, *tok), idx))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                let (prim_lbl, prim_c, _, hsqc) = loaded_2d_list.remove(best_idx);
                primary_cls = Some(prim_c);

                let prim_nx = Self::canonical_2d_axis_nucleus(
                    &hsqc.axis_x.label,
                    hsqc.axis_x.min_ppm(),
                    hsqc.axis_x.max_ppm(),
                );
                let prim_ny = Self::canonical_2d_axis_nucleus(
                    &hsqc.axis_y.label,
                    hsqc.axis_y.min_ppm(),
                    hsqc.axis_y.max_ppm(),
                );
                if prim_nx == "1H" && prim_ny == "15N" {
                    for (disp_name, _, s3) in &mut cubes_3d {
                        let rescue_flag = ingested_recipes
                            .get(disp_name)
                            .map(|r| r.enable_axis_rescue)
                            .unwrap_or(self.enable_axis_rescue);
                        let (_, _, audit_opt) = register_3d_hn_to_hsqc_2d_with_audit(
                            s3,
                            &hsqc,
                            0.08,
                            0.60,
                            rescue_flag,
                        );
                        if let Some(audit) = audit_opt {
                            if let Some(warn) = audit.warning_message() {
                                self.axis_rescue_notices.push(warn);
                            }
                            if let Some(rec) = ingested_recipes.get_mut(disp_name) {
                                rec.axis_rescue_audit = Some(audit.clone());
                            }
                            self.axis_rescue_audits.push(audit);
                        }
                    }
                }
                if prim_c == PulseSequenceClass::Caco2D {
                    caco_2d_specs.push(hsqc.clone());
                }
                self.primary_2d_label = prim_lbl.clone();
                self.primary_2d_visible = true;
                self.primary_ppm_nudge = [0.0, 0.0];
                self.hsqc_arc = Arc::new(hsqc.data.clone());
                self.hsqc_gen = self.hsqc_gen.wrapping_add(1);
                let prim_idx =
                    self.workbook
                        .add_2d_page(prim_lbl, hsqc.clone(), false, !cubes_3d.is_empty());
                self.workbook.active_2d_idx = prim_idx;
                self.workbook.two_d.zoom_history.clear();
                self.hsqc_spec = Some(hsqc);
            } else if self.workbook.pages_2d.is_empty() {
                self.hsqc_spec = None;
                self.hsqc_arc = Arc::new(Vec::new());
            }

            let mut any_extra_2d = false;
            for (lbl, cls_ov, _, s2) in loaded_2d_list {
                if cls_ov == PulseSequenceClass::Caco2D {
                    caco_2d_specs.push(s2.clone());
                }
                let _ = self.workbook.add_2d_page(lbl, s2, true, false);
                any_extra_2d = true;
            }
            if any_extra_2d || self.hsqc_spec.is_some() {
                self.sync_overlay_layers_from_workbook();
            }

            if self.hsqc_spec.is_some() {
                self.reset_view_to_spec();
                self.save_live_2d_into_workbook();
            }

            // Regime auto-selection & 13C-IDP CON/CACO/3D root picking
            let is_idp_13c = plan.problem_regime == ProblemRegime::Idp13cDetected
                || primary_cls == Some(PulseSequenceClass::Con2D)
                || self.active_2d_nuclei() == ("13C", "15N");
            if is_idp_13c {
                self.regime_kind = GuiRegimeKind::Idp13C;
                let c13_cfg = C13PickerConfig {
                    snr_2d: self.picker_snr,
                    ..C13PickerConfig::default()
                };
                let con_roots = if self.active_2d_nuclei() == ("13C", "15N") {
                    self.hsqc_spec
                        .as_ref()
                        .map(|sp| pick_con_2d(sp, &c13_cfg))
                        .unwrap_or_default()
                } else {
                    Vec::new()
                };
                let mut caco_peaks = Vec::new();
                for caco_sp in &caco_2d_specs {
                    caco_peaks.extend(pick_caco_2d(caco_sp, &c13_cfg));
                }
                let mut strip_peaks = Vec::new();
                if !con_roots.is_empty() {
                    for (_, cls3, ref sp3) in &cubes_3d {
                        let exp_opt = match cls3 {
                            PulseSequenceClass::Canco3D => Some(C13DetectedExp::Canco3D),
                            PulseSequenceClass::Cbcaco3D => Some(C13DetectedExp::Cbcaco3D),
                            _ => C13DetectedExp::from_name(&sp3.exp),
                        };
                        if let Some(c13_exp) = exp_opt {
                            strip_peaks
                                .extend(pick_con_anchored_3d(sp3, c13_exp, &con_roots, &c13_cfg));
                        }
                    }
                }
                self.c13_picked_dataset = Some(C13PickedDataset {
                    con_roots,
                    caco_peaks,
                    strip_peaks,
                });
                self.sync_con_roots_to_peaks();
            } else if plan.problem_regime == ProblemRegime::Paramagnetic {
                self.regime_kind = GuiRegimeKind::Paramagnetic;
            } else {
                self.regime_kind = GuiRegimeKind::Globular;
            }

            let n_3d_loaded = cubes_3d.len();
            if n_3d_loaded > 0 {
                self.show_strips_dock = true;
            }

            if run_full_pipeline && !is_idp_13c {
                let pcfg = PickerConfig {
                    snr_threshold: self.picker_snr,
                    ..PickerConfig::default()
                };
                let mut all_peaks = Vec::new();
                if let Some(ref hsqc) = self.hsqc_spec {
                    let (nx, ny) = self.active_2d_nuclei();
                    let roots = if nx == "1H" && ny == "13C" {
                        pick_c13_hsqc_2d(hsqc, &pcfg)
                    } else if nx == "1H" && ny == "15N" {
                        if self.use_neural_picker {
                            pick_hsqc_2d_neural(hsqc, &NeuralPickerConfig::default())
                        } else {
                            pick_hsqc_2d(hsqc, &pcfg)
                        }
                    } else {
                        self.pick_generic_2d_crosspeaks(hsqc, self.picker_snr)
                    };
                    let c_refs: Vec<&Spectrum3D> = cubes_3d.iter().map(|(_, _, s)| s).collect();
                    if !c_refs.is_empty() && !roots.is_empty() && nx == "1H" && ny == "15N" {
                        let p3 = pick_triple_3d_two_pass(
                            &c_refs,
                            &roots,
                            &pcfg,
                            &CorroboratedRepickConfig::default(),
                        );
                        all_peaks.extend(roots);
                        all_peaks.extend(p3);
                    } else {
                        all_peaks.extend(roots);
                    }
                }
                if !all_peaks.is_empty() {
                    self.peaks = all_peaks;
                }
            }

            self.hncacb_3d = None;
            self.hnco_3d = None;
            for (_, _, s3) in cubes_3d {
                if s3.exp == "HNCO" || s3.exp == "HNCACO" {
                    if self.hnco_3d.is_none() {
                        self.hnco_3d = Some(s3);
                    }
                } else if self.hncacb_3d.is_none() {
                    self.hncacb_3d = Some(s3);
                }
            }
            let stored_3d =
                usize::from(self.hncacb_3d.is_some()) + usize::from(self.hnco_3d.is_some());
            self.extra_3d_count = n_3d_loaded.saturating_sub(stored_3d);
            if self.hsqc_spec.is_some() {
                self.save_live_2d_into_workbook();
            }
        }

        let n_3d_total = usize::from(self.hncacb_3d.is_some())
            + usize::from(self.hnco_3d.is_some())
            + self.extra_3d_count;
        self.sync_session_from_app_fresh(&plan.root_name);
        self.session.state.processing_recipes = ingested_recipes;
        self.rebuild_systems_and_matrices();

        if self.hsqc_spec.is_none() && n_3d_total == 0 {
            if !load_errors.is_empty() {
                self.status_msg = format!(
                    "Failed to load selected spectra from '{}': {}",
                    plan.root_name,
                    load_errors.join(" | ")
                );
            } else {
                self.status_msg = format!(
                    "No 2D/3D spectra selected or loaded from '{}'",
                    plan.root_name
                );
            }
            return;
        }

        let can_run_idp = self.regime_kind == GuiRegimeKind::Idp13C
            && !self.sequence.is_empty()
            && self
                .c13_picked_dataset
                .as_ref()
                .is_some_and(|ds| !ds.con_roots.is_empty());
        let can_run_hn = self.regime_kind != GuiRegimeKind::Idp13C
            && !self.sequence.is_empty()
            && !self.peaks.is_empty();

        if run_full_pipeline && (can_run_idp || can_run_hn) {
            self.run_assignment();
        } else if let Some(first_warn) = self.axis_rescue_notices.first() {
            self.status_msg = first_warn.clone();
        } else if self.regime_kind == GuiRegimeKind::Idp13C {
            let (n_con, n_pro) = self
                .c13_picked_dataset
                .as_ref()
                .map(|ds| {
                    (
                        ds.con_roots.len(),
                        ds.con_roots
                            .iter()
                            .filter(|r| r.is_proline_imino_region)
                            .count(),
                    )
                })
                .unwrap_or((0, 0));
            let warn_suffix = if load_errors.is_empty() {
                String::new()
            } else {
                format!(" [warn: {}]", load_errors.join("; "))
            };
            self.status_msg = format!(
                "Loaded 13C-IDP '{}' ({} + {} overlay(s), {n_con} CON roots incl. {n_pro} Pro){warn_suffix}",
                plan.root_name,
                self.primary_2d_label,
                self.overlay_layers.len()
            );
        } else {
            let warn_suffix = if load_errors.is_empty() {
                String::new()
            } else {
                format!(" [warn: {}]", load_errors.join("; "))
            };
            self.status_msg = format!(
                "Loaded '{}' (primary: {}{}, {} peaks, {} aa){warn_suffix}",
                plan.root_name,
                self.primary_2d_label,
                if self.overlay_layers.is_empty() {
                    String::new()
                } else {
                    format!(" + {} 2D overlay(s)", self.overlay_layers.len())
                },
                self.peaks.len(),
                self.sequence.len()
            );
        }
    }

    fn dispatch_workbench_action(&mut self, ctx: &egui::Context, action: WorkbenchAction) {
        match action {
            // --- 1. File & Ingest ---
            WorkbenchAction::OpenIngestMatrix | WorkbenchAction::LoadSequenceFasta => {
                self.ingest_panel.open = true;
            }
            WorkbenchAction::LoadPathOrSpectrum => {
                #[cfg(not(target_arch = "wasm32"))]
                self.load_path_auto();
                #[cfg(target_arch = "wasm32")]
                {
                    self.show_text_io_modal = true;
                }
            }
            WorkbenchAction::OpenPasteTextModal => {
                self.show_text_io_modal = true;
            }
            WorkbenchAction::LoadUbiquitinDemo => {
                self.load_demo_and_solve();
            }
            WorkbenchAction::SaveSessionNef | WorkbenchAction::SaveSessionNefAs => {
                self.save_session_nef_action(ctx);
            }
            WorkbenchAction::LoadSessionNef => {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    let p = Path::new(self.export_prefix.trim()).with_extension("nef");
                    if p.exists() {
                        match std::fs::read_to_string(&p) {
                            Ok(txt) => self.load_text_in_memory(&txt, &p.display().to_string()),
                            Err(e) => self.status_msg = format!("NEF load error: {e}"),
                        }
                    } else {
                        self.show_text_io_modal = true;
                        self.status_msg =
                            "Drop a .nef file onto the window or paste NEF session text"
                                .to_string();
                    }
                }
                #[cfg(target_arch = "wasm32")]
                {
                    self.show_text_io_modal = true;
                }
            }
            WorkbenchAction::ExportNmrStar | WorkbenchAction::ExportPeaksTsv => {
                self.export_results(ctx);
            }
            WorkbenchAction::ExportQualityReportHtml => {
                if let Some(pred) = self.pred_ref.clone().or_else(|| self.reference.clone()) {
                    if (self.regime_kind == GuiRegimeKind::Idp13C || self.show_propensity_panel)
                        && self.propensity_report.is_none()
                    {
                        self.refresh_propensity_report();
                    }
                    let mut inp = ReportInput::new(
                        self.session.state.protein_name.clone(),
                        pred.clone(),
                        self.pred_conf.clone(),
                    );
                    inp.n_peaks = Some(self.peaks.len());
                    inp.n_spin_systems = Some(self.systems.len());
                    inp.dihedrals = predict_csi_dihedrals(&pred);
                    inp.axis_rescue_records = self.axis_rescue_audits.clone();
                    if let Some(res) = &self.assign_res {
                        inp.solver_score = Some(res.score);
                        inp.systems_and_assignment = Some((self.systems.clone(), res.clone()));
                    }
                    if let Some((t_crit, max_cv, mu_null)) = self.last_bayes_sa_telemetry {
                        inp.engine_label = Some(format!(
                            "{} [T_crit={t_crit:.4} | max_Cv={max_cv:.3} | mu_null={mu_null:.4}]",
                            self.solver_kind.label()
                        ));
                    } else {
                        inp.engine_label = Some(self.solver_kind.label().to_string());
                    }
                    if self.regime_kind == GuiRegimeKind::Idp13C || self.show_propensity_panel {
                        inp.propensity_report = self.propensity_report.clone();
                    }
                    let html = generate_quality_report_html(&inp);
                    self.export_text_buf = html.clone();
                    self.exported_in_session = true;
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        let html_p = Path::new(self.export_prefix.trim()).with_extension("html");
                        let _ = std::fs::write(&html_p, &html);
                        self.status_msg =
                            format!("Exported HTML quality report to {}", html_p.display());
                    }
                    #[cfg(target_arch = "wasm32")]
                    {
                        self.show_text_io_modal = true;
                        self.status_msg = "Generated HTML quality report in memory".to_string();
                    }
                } else {
                    self.status_msg =
                        "Run assignment before exporting HTML quality report".to_string();
                }
            }
            WorkbenchAction::ResetWorkspace => {
                self.peaks.clear();
                self.systems.clear();
                self.sequence.clear();
                self.seq_ids.clear();
                self.assign_res = None;
                self.pred_ref = None;
                self.pred_conf.clear();
                self.pred_ambiguity_codes.clear();
                let saved_ph = self.sidechain_ribbon_state.sample_ph;
                self.sidechain_ribbon_state = SequenceRibbonState::new();
                self.sidechain_ribbon_state.set_sample_ph(saved_ph);
                self.c13_picked_dataset = None;
                self.propensity_report = None;
                self.metrics = None;
                self.qc_report = None;
                self.last_bayes_sa_telemetry = None;
                self.overlay_layers.clear();
                self.primary_2d_label = "15N-HSQC".to_string();
                self.primary_2d_visible = true;
                self.box_zoom_2d_mode = false;
                self.box_zoom_2d_anchor = None;
                self.hsqc_spec = None;
                self.hncacb_3d = None;
                self.hnco_3d = None;
                self.axis_rescue_notices.clear();
                self.axis_rescue_audits.clear();
                self.session = WorkbenchSessionController::default();
                self.status_msg = "Workspace reset".to_string();
            }

            // --- 2. FID Processing ---
            WorkbenchAction::ProcessAllRawFids | WorkbenchAction::ReconstructNusIstFista => {
                let plan = self.ingest_panel.build_execution_plan();
                if plan.selected_experiments.is_empty() {
                    self.ingest_panel.open = true;
                    self.status_msg =
                        "Open Ingest Matrix to select raw FIDs to process".to_string();
                } else {
                    self.execute_ingest_plan(plan, false);
                }
            }
            WorkbenchAction::Register3dToHsqc2d =>
            {
                #[cfg(not(target_arch = "wasm32"))]
                if let Some(ref hsqc) = self.hsqc_spec {
                    let mut n_reg = 0usize;
                    self.axis_rescue_notices.clear();
                    self.axis_rescue_audits.clear();
                    if let Some(ref mut s3) = self.hncacb_3d {
                        let (_, _, audit_opt) = register_3d_hn_to_hsqc_2d_with_audit(
                            s3,
                            hsqc,
                            0.08,
                            0.60,
                            self.enable_axis_rescue,
                        );
                        if let Some(audit) = audit_opt {
                            if let Some(warn) = audit.warning_message() {
                                self.axis_rescue_notices.push(warn);
                            }
                            self.axis_rescue_audits.push(audit);
                        }
                        n_reg += 1;
                    }
                    if let Some(ref mut s3) = self.hnco_3d {
                        let (_, _, audit_opt) = register_3d_hn_to_hsqc_2d_with_audit(
                            s3,
                            hsqc,
                            0.08,
                            0.60,
                            self.enable_axis_rescue,
                        );
                        if let Some(audit) = audit_opt {
                            if let Some(warn) = audit.warning_message() {
                                self.axis_rescue_notices.push(warn);
                            }
                            self.axis_rescue_audits.push(audit);
                        }
                        n_reg += 1;
                    }
                    self.status_msg = if let Some(first_warn) = self.axis_rescue_notices.first() {
                        first_warn.clone()
                    } else {
                        format!("Registered {n_reg} 3D cube(s) to 2D HSQC reference")
                    };
                }
            }
            WorkbenchAction::ExportProcessingRecipeJson => {
                if let Ok(js) = serde_json::to_string_pretty(&self.session.state.processing_recipes)
                {
                    self.export_text_buf = js;
                    self.show_text_io_modal = true;
                    self.status_msg = "Exported ProcessingRecipe JSON".to_string();
                }
            }
            WorkbenchAction::AutoPhaseDirect1D => {
                if self.is_1d_studio_active() && !self.stack1d_panel.traces.is_empty() {
                    self.stack1d_panel.auto_phase_active_trace();
                    self.status_msg = self.stack1d_panel.status_message.clone();
                } else {
                    self.status_msg = "Configured DSP action: Auto-Phase Direct 1D".to_string();
                }
            }
            WorkbenchAction::ApplyMedianBaseline => {
                if self.is_1d_studio_active() && !self.stack1d_panel.traces.is_empty() {
                    self.stack1d_panel.baseline_correct_active_trace();
                    self.status_msg = self.stack1d_panel.status_message.clone();
                } else {
                    self.status_msg = "Configured DSP action: Baseline Correction".to_string();
                }
            }
            WorkbenchAction::AutoPhaseIndirectBiphasic
            | WorkbenchAction::ToggleLinearPrediction => {
                self.status_msg = format!("Configured DSP action: {}", action.metadata().title);
            }
            WorkbenchAction::SelectPhase1dMode => {
                if !self.stack1d_panel.traces.is_empty() {
                    self.activate_1d_studio_mode();
                    self.stack1d_panel.select_tool_mode(ToolMode1d::Phase);
                    self.stack1d_panel.show_1d_analysis_drawer = true;
                    self.stack1d_panel.selected_inspector_tab = InspectorTab1d::PhaseBaseline;
                    self.status_msg = "1D Live Phase Mode (`ph`)".to_string();
                }
            }
            WorkbenchAction::SelectIntegrate1dMode => {
                if !self.stack1d_panel.traces.is_empty() {
                    self.activate_1d_studio_mode();
                    self.stack1d_panel
                        .select_tool_mode(ToolMode1d::IntegrateRoi);
                    self.stack1d_panel.show_integral_curves = true;
                    self.status_msg = "1D Peak Integration Mode (`pi` / `it`)".to_string();
                }
            }
            WorkbenchAction::SelectMultiplet1dMode => {
                if !self.stack1d_panel.traces.is_empty() {
                    self.activate_1d_studio_mode();
                    self.stack1d_panel.select_tool_mode(ToolMode1d::Multiplet);
                    self.stack1d_panel.show_multiplet_boxes = true;
                    self.stack1d_panel.show_j_trees = true;
                    self.status_msg = "1D Multiplet Analysis Mode (`mp`)".to_string();
                }
            }
            WorkbenchAction::SelectReferenceOrAlignMode => {
                if self.is_1d_studio_active() {
                    self.stack1d_panel.select_tool_mode(ToolMode1d::Reference);
                    self.status_msg = "1D Chemical Shift Referencing Mode (`rf`)".to_string();
                } else {
                    self.show_overlay_manager = true;
                    self.status_msg = "2D Reference / Overlay Alignment (`rf`)".to_string();
                }
            }
            WorkbenchAction::SuppressWaterPeak => {
                self.suppress_water_active_spectrum();
            }
            WorkbenchAction::UndoWaterSuppression => {
                self.undo_water_suppression_active();
            }

            // --- 3. Spectrum View & Contours ---
            WorkbenchAction::OpenContourThresholdPopup => {
                self.show_contour_modal = !self.show_contour_modal;
            }
            WorkbenchAction::ZoomFullSweep => {
                if self.is_1d_studio_active() {
                    self.stack1d_panel.fit_full_ppm_bounds();
                    self.status_msg = "1D Full Sweep (`zf`)".to_string();
                } else {
                    let cur = [
                        self.view_h_left,
                        self.view_h_right,
                        self.view_n_top,
                        self.view_n_bottom,
                    ];
                    self.workbook.two_d.push_zoom(cur);
                    self.reset_view_to_spec();
                }
            }
            WorkbenchAction::ZoomPrevious => {
                if self.is_1d_studio_active() {
                    if self.stack1d_panel.undo_zoom() {
                        self.status_msg = "1D Zoom Previous (`zp`)".to_string();
                    }
                } else if let Some(prev) = self.workbook.two_d.pop_zoom() {
                    self.view_h_left = prev[0];
                    self.view_h_right = prev[1];
                    self.view_n_top = prev[2];
                    self.view_n_bottom = prev[3];
                    self.status_msg = "2D Zoom Previous (`zp`)".to_string();
                }
            }
            WorkbenchAction::ZoomIn | WorkbenchAction::ZoomOut => {
                let factor = if action == WorkbenchAction::ZoomIn {
                    0.80f32
                } else {
                    1.25f32
                };
                if self.is_1d_studio_active() {
                    let lo = self.stack1d_panel.view_ppm_min;
                    let hi = self.stack1d_panel.view_ppm_max;
                    let anchor = self.stack1d_panel.cursor_ppm.unwrap_or(0.5 * (lo + hi));
                    let f64_f = factor as f64;
                    self.stack1d_panel.push_zoom_window(
                        anchor + (lo - anchor) * f64_f,
                        anchor + (hi - anchor) * f64_f,
                    );
                } else {
                    let cur = [
                        self.view_h_left,
                        self.view_h_right,
                        self.view_n_top,
                        self.view_n_bottom,
                    ];
                    self.workbook.two_d.push_zoom(cur);
                    let ah = self.crosshair.h_ppm as f32;
                    let an = self.crosshair.n_ppm as f32;
                    self.view_h_left = ah + (self.view_h_left - ah) * factor;
                    self.view_h_right = ah + (self.view_h_right - ah) * factor;
                    self.view_n_top = an + (self.view_n_top - an) * factor;
                    self.view_n_bottom = an + (self.view_n_bottom - an) * factor;
                }
            }
            WorkbenchAction::SelectZoomBoxMode => {
                if self.is_1d_studio_active() {
                    self.stack1d_panel.select_tool_mode(ToolMode1d::Zoom);
                } else {
                    self.workbook.two_d.tool_mode = ToolMode2d::ZoomBox;
                }
            }
            WorkbenchAction::SelectPanMoveMode => {
                if self.is_1d_studio_active() {
                    self.stack1d_panel.select_tool_mode(ToolMode1d::Pan);
                } else {
                    self.workbook.two_d.tool_mode = ToolMode2d::Pan;
                    self.box_zoom_2d_anchor = None;
                    self.workbook.two_d.drag_box_start_pos = None;
                    self.workbook.two_d.pick_box_start_pos = None;
                }
            }
            WorkbenchAction::SelectCrosshairSliceMode => {
                if self.is_1d_studio_active() {
                    let _ = self.crosshair.toggle_lock();
                } else {
                    self.workbook.two_d.tool_mode = ToolMode2d::CrosshairSlice;
                    self.workbook.two_d.show_projections = true;
                }
            }
            WorkbenchAction::SelectPickPeaksMode => {
                if self.is_1d_studio_active() {
                    self.stack1d_panel.select_tool_mode(ToolMode1d::PeakPick);
                } else {
                    self.workbook.two_d.tool_mode = ToolMode2d::PickPeaks;
                }
            }
            WorkbenchAction::SelectPointerMode => {
                if self.is_1d_studio_active() {
                    self.stack1d_panel.select_tool_mode(ToolMode1d::Zoom);
                } else {
                    self.workbook.two_d.tool_mode = ToolMode2d::ZoomBox;
                }
            }
            WorkbenchAction::ToggleMarginalProjections => {
                if !self.is_1d_studio_active() {
                    self.workbook.two_d.show_projections = !self.workbook.two_d.show_projections;
                }
            }
            WorkbenchAction::TogglePeakListTable => {
                if self.is_1d_studio_active() {
                    self.stack1d_panel.show_1d_analysis_drawer =
                        !self.stack1d_panel.show_1d_analysis_drawer
                            || self.stack1d_panel.selected_inspector_tab != InspectorTab1d::Peaks;
                    self.stack1d_panel.selected_inspector_tab = InspectorTab1d::Peaks;
                } else {
                    self.workbook.two_d.show_peak_table = !self.workbook.two_d.show_peak_table;
                }
            }
            WorkbenchAction::NextWorkbookPage | WorkbenchAction::PrevWorkbookPage => {
                let n1 = self.stack1d_panel.traces.len();
                let n2 = self.workbook.pages_2d.len();
                let total = n1 + n2;
                if total > 1 {
                    let cur = if self.show_stack1d_panel && n1 > 0 {
                        self.stack1d_panel.ref_trace_idx.min(n1 - 1)
                    } else if n2 > 0 {
                        n1 + self.workbook.active_2d_idx.min(n2 - 1)
                    } else {
                        0
                    };
                    let next = if action == WorkbenchAction::NextWorkbookPage {
                        (cur + 1) % total
                    } else {
                        (cur + total - 1) % total
                    };
                    if next < n1 {
                        self.activate_1d_workbook_page(next);
                    } else {
                        self.activate_2d_workbook_page(next - n1);
                    }
                }
            }
            WorkbenchAction::ToggleCanvasTheme => {
                self.paper_white_canvas = !self.paper_white_canvas;
                self.sync_overlay_layers_from_workbook();
            }
            WorkbenchAction::RaiseContourFloor | WorkbenchAction::LowerContourFloor => {
                if self.is_1d_studio_active() {
                    let g_mul = if action == WorkbenchAction::RaiseContourFloor {
                        1.25_f64
                    } else {
                        0.80_f64
                    };
                    self.stack1d_panel.y_gain =
                        (self.stack1d_panel.y_gain * g_mul).clamp(0.001, 100_000.0);
                } else {
                    let new_base = if action == WorkbenchAction::RaiseContourFloor {
                        (self.contour_base * CONTOUR_FLOOR_STEP_RATIO)
                            .min((self.contour_noise_ref * 1e4).max(1e9))
                    } else {
                        (self.contour_base / CONTOUR_FLOOR_STEP_RATIO).max(1e-6)
                    };
                    self.contour_base = new_base;
                    let mut params = SessionContourParams::new("HSQC");
                    params.base_level = self.contour_base;
                    params.level_factor = self.contour_factor;
                    params.num_levels = self.contour_levels;
                    params.show_negative = self.show_negative;
                    params.ppm_nudge = self.primary_ppm_nudge;
                    let msg = self.session.execute(WorkbenchCommand::SetContourParams {
                        layer_id: "HSQC".to_string(),
                        params,
                    });
                    self.status_msg = msg;
                }
            }
            WorkbenchAction::ToggleNegativeContours => {
                self.show_negative = !self.show_negative;
            }
            WorkbenchAction::ToggleCrosshairLock => {
                let locked = self.crosshair.toggle_lock();
                self.status_msg = if locked {
                    format!(
                        "Crosshair locked at ¹H {:.3}, ¹⁵N {:.2} ppm",
                        self.crosshair.h_ppm, self.crosshair.n_ppm
                    )
                } else {
                    "Crosshair unlocked".to_string()
                };
            }
            WorkbenchAction::OpenOverlayStackManager
            | WorkbenchAction::NudgeSpectrumReferenceOffset => {
                self.show_overlay_manager = !self.show_overlay_manager;
            }
            WorkbenchAction::CycleActiveOverlayLayer => {
                let total = 1 + self.overlay_layers.len();
                self.active_overlay_idx = (self.active_overlay_idx + 1) % total;
                self.status_msg = format!("Active contour layer: #{}", self.active_overlay_idx);
            }
            WorkbenchAction::SwitchToProteinNmrWorkspace => {
                self.workspace_mode = WorkspaceMode::ProteinNmr;
            }

            // --- 4. Peak Picking & Editing ---
            WorkbenchAction::PickHsqc2dRoots => {
                self.pick_2d_roots();
            }
            WorkbenchAction::PickAll3dSpectra | WorkbenchAction::RunCorroboratedWeakPeakRescue => {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    let roots: Vec<Peak> = self
                        .peaks
                        .iter()
                        .filter(|p| p.exp == "HSQC")
                        .cloned()
                        .collect();
                    let mut cubes: Vec<&Spectrum3D> = Vec::new();
                    if let Some(ref s3) = self.hncacb_3d {
                        cubes.push(s3);
                    }
                    if let Some(ref s3) = self.hnco_3d {
                        cubes.push(s3);
                    }
                    if !roots.is_empty() && !cubes.is_empty() {
                        let pcfg = PickerConfig {
                            snr_threshold: self.picker_snr,
                            ..PickerConfig::default()
                        };
                        let p3 = pick_triple_3d_two_pass(
                            &cubes,
                            &roots,
                            &pcfg,
                            &CorroboratedRepickConfig::default(),
                        );
                        let n3 = p3.len();
                        let mut combined = roots;
                        combined.extend(p3);
                        self.peaks = combined;
                        self.session.state.peaks = self.peaks.clone();
                        self.rebuild_systems_and_matrices();
                        self.status_msg = format!("Two-pass 3D picker extracted {n3} cross-peaks");
                    } else {
                        self.status_msg =
                            "Load 2D HSQC roots and 3D cubes before 3D strip picking".to_string();
                    }
                }
            }
            WorkbenchAction::ToggleNeuralVsClassicalPicker => {
                self.use_neural_picker = !self.use_neural_picker;
                self.status_msg = format!(
                    "2D Picker: {}",
                    if self.use_neural_picker {
                        "Neural CNN"
                    } else {
                        "Classical MAD"
                    }
                );
            }
            WorkbenchAction::AddPeakAtCrosshair => {
                if self.is_1d_studio_active() {
                    let target = self.stack1d_panel.cursor_ppm.unwrap_or(
                        0.5 * (self.stack1d_panel.view_ppm_min + self.stack1d_panel.view_ppm_max),
                    );
                    let _ = self
                        .stack1d_panel
                        .add_manual_peak_at_or_near_ppm(target, true);
                    self.status_msg = self.stack1d_panel.status_message.clone();
                } else {
                    let rx = self
                        .hsqc_spec
                        .as_ref()
                        .map(|s| (s.axis_x.step_ppm.abs() * 3.0).max(0.01))
                        .unwrap_or(0.02);
                    let ry = self
                        .hsqc_spec
                        .as_ref()
                        .map(|s| (s.axis_y.step_ppm.abs() * 3.0).max(0.02))
                        .unwrap_or(0.10);
                    let _ = self.add_manual_2d_peak(
                        self.crosshair.h_ppm,
                        self.crosshair.n_ppm,
                        Some((rx, ry)),
                    );
                }
            }
            WorkbenchAction::DeleteSelectedPeaks => {
                if self.is_1d_studio_active() {
                    if let Some(sel) = self.stack1d_panel.selected_peak_idx {
                        let _ = self.stack1d_panel.delete_picked_peak(sel);
                        self.status_msg = self.stack1d_panel.status_message.clone();
                    }
                } else if let Some(sel_2d) = self.workbook.two_d.selected_peak_idx {
                    self.delete_2d_peak(sel_2d);
                } else if let Some(sel_s) = self.selected_ss.filter(|&s| s < self.systems.len()) {
                    let ss = &self.systems[sel_s];
                    let idxs: Vec<usize> = self
                        .peaks
                        .iter()
                        .enumerate()
                        .filter(|(_, p)| (p.h - ss.h).abs() <= 0.035 && (p.n - ss.n).abs() <= 0.35)
                        .map(|(i, _)| i)
                        .collect();
                    if !idxs.is_empty() {
                        let msg = self.session.execute(WorkbenchCommand::DeletePeaks(idxs));
                        self.sync_app_from_session();
                        self.status_msg = msg;
                    }
                } else {
                    self.status_msg = "Select a peak / spin system to delete (`pd`)".to_string();
                }
            }
            WorkbenchAction::SnapPeakParabolicCenter => {
                if self.is_1d_studio_active() {
                    if let Some(sel) = self.stack1d_panel.selected_peak_idx {
                        let _ = self.stack1d_panel.snap_picked_peak_to_apex(sel);
                        self.status_msg = self.stack1d_panel.status_message.clone();
                    }
                } else if let Some(sel_2d) = self.workbook.two_d.selected_peak_idx {
                    self.snap_2d_peak_by_index(sel_2d);
                } else {
                    self.snap_selected_peak_parabolic();
                }
            }
            WorkbenchAction::TogglePeakLabels => {
                if self.is_1d_studio_active() {
                    self.stack1d_panel.show_peak_labels = !self.stack1d_panel.show_peak_labels;
                } else {
                    self.show_peak_labels = !self.show_peak_labels;
                }
            }
            WorkbenchAction::FilterArtifactAndFoldedPeaks => {
                self.rebuild_systems_and_matrices();
            }

            // --- 5. Spin-System, Assignment & Session Undo/Redo ---
            WorkbenchAction::BuildSpinSystems => {
                self.rebuild_systems_and_matrices();
            }
            WorkbenchAction::RunBackboneAssignment => {
                self.run_assignment();
            }
            WorkbenchAction::SelectSolverEvoGraph => {
                self.solver_kind = GuiSolverKind::EvoGraph;
            }
            WorkbenchAction::SelectSolverStandardSa => {
                self.solver_kind = GuiSolverKind::Standard;
            }
            WorkbenchAction::SelectSolverBayesSa => {
                self.solver_kind = GuiSolverKind::BayesSa;
            }
            WorkbenchAction::ToggleGpuComputeEngine => {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    self.use_gpu = !self.use_gpu
                        && (self.gpu_engine.is_some() || self.bayes_gpu_solver.is_some());
                }
            }
            WorkbenchAction::TileSequentialStrips => {
                self.workspace_mode = WorkspaceMode::ProteinNmr;
                self.use_match_finder_strips = true;
            }
            WorkbenchAction::OpenResidueAssignmentPopover => {
                self.show_assign_popover = true;
            }
            WorkbenchAction::LockSelectedResidueAssignment => {
                if let (Some(ss), Some(rk)) = (self.selected_ss, self.selected_res) {
                    if ss < self.systems.len()
                        && rk < self.sequence.len()
                        && self.sequence[rk] != "PRO"
                        && rk > 0
                    {
                        let msg = self.session.execute(WorkbenchCommand::LockResidue {
                            spin_system: ss,
                            residue_seq_idx: rk,
                            locked: true,
                        });
                        self.sync_app_from_session();
                        self.status_msg = msg;
                    }
                }
            }
            WorkbenchAction::UnlockSelectedResidueAssignment => {
                if let Some(ss) = self.selected_ss {
                    if let Some(rk) = self.session.state.locked_residue_for_spin_system(ss) {
                        let msg = self.session.execute(WorkbenchCommand::LockResidue {
                            spin_system: ss,
                            residue_seq_idx: rk,
                            locked: false,
                        });
                        self.sync_app_from_session();
                        self.status_msg = msg;
                    }
                }
            }
            WorkbenchAction::UnassignSelectedSpinSystem => {
                if let Some(rk) = self.selected_res {
                    let msg = self.session.execute(WorkbenchCommand::UnassignResidue {
                        residue_seq_idx: rk,
                    });
                    self.sync_app_from_session();
                    self.status_msg = msg;
                }
            }
            WorkbenchAction::EstimateGlobalReferenceOffsets => {
                self.run_assignment();
            }
            WorkbenchAction::InspectDatasetQcReport => {
                self.show_qc_modal = !self.show_qc_modal;
            }
            WorkbenchAction::UndoLastCommand => {
                if let Some(cmd) = self.mcp_bridge.undo_last() {
                    let summary = cmd.summary();
                    self.sync_app_from_mcp_snapshot(&format!("Undid AI action: {summary}"));
                } else if let Some(msg) = self.session.undo() {
                    self.sync_app_from_session();
                    self.status_msg = msg;
                } else {
                    self.status_msg = "Nothing to undo".to_string();
                }
            }
            WorkbenchAction::RedoLastCommand => {
                if let Some(cmd) = self.mcp_bridge.redo_last() {
                    let summary = cmd.summary();
                    self.sync_app_from_mcp_snapshot(&format!("Redid AI action: {summary}"));
                } else if let Some(msg) = self.session.redo() {
                    self.sync_app_from_session();
                    self.status_msg = msg;
                } else {
                    self.status_msg = "Nothing to redo".to_string();
                }
            }

            // --- 6 & 7. Structure, Sidechain & Dynamics ---
            WorkbenchAction::PredictBackboneDihedrals => {
                if let Some(pred) = &self.pred_ref {
                    let dih = predict_csi_dihedrals(pred);
                    self.has_structure_or_dynamics = !dih.is_empty();
                    self.status_msg = format!(
                        "Predicted CSI backbone (φ, ψ) dihedrals for {} residues",
                        dih.len()
                    );
                } else {
                    self.status_msg =
                        "Run backbone assignment before predicting dihedrals".to_string();
                }
            }
            WorkbenchAction::RunJointSidechainAssignment => {
                self.workspace_mode = WorkspaceMode::ProteinNmr;
                self.run_joint_sidechain_in_gui();
            }
            WorkbenchAction::InspectSidechainCovalentGraph => {
                self.workspace_mode = WorkspaceMode::ProteinNmr;
                self.show_sidechain_inspector = true;
                if self.selected_res.is_none() && !self.sequence.is_empty() {
                    self.select_residue_and_sync_strip(0);
                }
                self.status_msg =
                    "Opened live 2D IUPAC covalent sidechain Z-score inspector".to_string();
            }
            WorkbenchAction::FitRelaxationR1R2HetNoe => {
                self.workspace_mode = WorkspaceMode::ProteinNmr;
                self.show_relax_panel = true;
                self.show_dynamics_panel = false;
                self.has_structure_or_dynamics = true;
                let pdb_ref = self.loaded_pdb_text.clone();
                let res = if self.relax_panel.has_fitted_rates() {
                    self.relax_panel.refit_loaded_curves()
                } else {
                    self.relax_panel.load_demo_dynamics(
                        &self.sequence,
                        &self.seq_ids,
                        pdb_ref.as_deref(),
                    )
                };
                match res {
                    Ok(msg) => self.status_msg = msg,
                    Err(e) => self.status_msg = format!("Relaxation fit error: {e}"),
                }
            }
            WorkbenchAction::RunModelFreeLipariSzabo => {
                self.workspace_mode = WorkspaceMode::ProteinNmr;
                self.show_relax_panel = true;
                self.show_dynamics_panel = false;
                self.has_structure_or_dynamics = true;
                let pdb_ref = self.loaded_pdb_text.clone();
                let res = if !self.relax_panel.r1_fits.is_empty()
                    && !self.relax_panel.r2_fits.is_empty()
                {
                    self.relax_panel.run_modelfree(pdb_ref.as_deref())
                } else {
                    self.relax_panel.load_demo_dynamics(
                        &self.sequence,
                        &self.seq_ids,
                        pdb_ref.as_deref(),
                    )
                };
                match res {
                    Ok(msg) => {
                        self.status_msg = msg;
                        if let Some(act) = self
                            .relax_panel
                            .build_3d_track_export(&self.seq_ids, QuantitativeTrack::LipariSzaboS2)
                        {
                            self.apply_relax_panel_actions(vec![act]);
                        }
                    }
                    Err(e) => self.status_msg = format!("Model-Free error: {e}"),
                }
            }
            WorkbenchAction::AnalyzeCestExchange => {
                self.workspace_mode = WorkspaceMode::ProteinNmr;
                self.show_dynamics_panel = true;
                self.show_relax_panel = false;
                self.has_structure_or_dynamics = true;
                self.dynamics_panel.active_tab = DynamicsStudioTab::CestDest;
                match self
                    .dynamics_panel
                    .ensure_demo_exchange_and_csp(&self.sequence, &self.seq_ids)
                {
                    Ok(msg) => self.status_msg = msg,
                    Err(e) => self.status_msg = format!("CEST solver error: {e}"),
                }
            }
            WorkbenchAction::AnalyzeCpmgDispersion => {
                self.workspace_mode = WorkspaceMode::ProteinNmr;
                self.show_dynamics_panel = true;
                self.show_relax_panel = false;
                self.has_structure_or_dynamics = true;
                self.dynamics_panel.active_tab = DynamicsStudioTab::CpmgDispersion;
                match self
                    .dynamics_panel
                    .ensure_demo_exchange_and_csp(&self.sequence, &self.seq_ids)
                {
                    Ok(msg) => {
                        self.status_msg = msg;
                        if let Some(proj) = self
                            .dynamics_panel
                            .build_overlay_projection(QuantitativeTrack::ChemicalExchangeRex)
                        {
                            self.apply_dynamics_panel_output(DynamicsStudioOutput {
                                clicked_seq_id: None,
                                overlay_projection: Some(proj),
                            });
                        }
                    }
                    Err(e) => self.status_msg = format!("CPMG dispersion error: {e}"),
                }
            }
            WorkbenchAction::FitLigandTitrationCspKd => {
                self.workspace_mode = WorkspaceMode::ProteinNmr;
                self.show_dynamics_panel = true;
                self.show_relax_panel = false;
                self.has_structure_or_dynamics = true;
                self.dynamics_panel.active_tab = DynamicsStudioTab::LigandCsp;
                match self
                    .dynamics_panel
                    .ensure_demo_exchange_and_csp(&self.sequence, &self.seq_ids)
                {
                    Ok(msg) => {
                        self.status_msg = msg;
                        if let Some(proj) = self
                            .dynamics_panel
                            .build_overlay_projection(QuantitativeTrack::LigandCsp)
                        {
                            self.apply_dynamics_panel_output(DynamicsStudioOutput {
                                clicked_seq_id: None,
                                overlay_projection: Some(proj),
                            });
                        }
                    }
                    Err(e) => self.status_msg = format!("CSP tight-binding Kd error: {e}"),
                }
            }
            WorkbenchAction::AssignNoesyRestraints
            | WorkbenchAction::FoldStructureEnsemble
            | WorkbenchAction::ValidateEnsembleRmsdRamachandran => {
                self.has_structure_or_dynamics = true;
                self.status_msg = format!("Triggered {}", action.metadata().title);
            }

            // --- 8. Octet Chemical Studio ---
            WorkbenchAction::SwitchToOctetChemStudio => {
                self.workspace_mode = WorkspaceMode::OctetChem;
            }
            WorkbenchAction::ImportLoadedProteinToOctet => {
                self.octet_app
                    .set_loaded_protein_sequence(self.sequence.clone());
                self.workspace_mode = WorkspaceMode::OctetChem;
            }
            WorkbenchAction::Clean2dMolecularLayout
            | WorkbenchAction::Predict1dProtonCarbonNmr
            | WorkbenchAction::ExportChemicalPublicationSvg => {
                self.workspace_mode = WorkspaceMode::OctetChem;
            }

            // --- 9. Help & Diagnostics ---
            WorkbenchAction::OpenCommandPalette => {
                self.command_palette.open_palette();
            }
            WorkbenchAction::OpenKeyboardCheatSheet => {
                self.chord_dispatcher.open_cheat_sheet();
            }
            WorkbenchAction::ReSimulateSyntheticNoise => {
                self.sim_seed = self.sim_seed.wrapping_add(1);
                if let Some(r) = self.reference.clone() {
                    self.load_from_reference(r);
                    self.run_assignment();
                }
            }
            WorkbenchAction::ShowAboutAndProvenance => {
                self.show_about_modal = true;
            }
        }
    }

    fn draw_phase9_modals(&mut self, ctx: &egui::Context) {
        // 1. Ingest Selection Matrix Modal (`T09.2`)
        if let Some(ingest_act) = self.ingest_panel.show(ctx) {
            match ingest_act {
                IngestAction::ScanNativePath(path_str) => {
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        let cleaned = sanitize_user_path_input(&path_str);
                        match IngestPanelState::from_native_path(Path::new(&cleaned)) {
                            Ok(panel) => {
                                self.ingest_panel = panel;
                                self.ingest_panel.open = true;
                            }
                            Err(e) => {
                                self.ingest_panel.status_error = Some(e);
                            }
                        }
                    }
                    #[cfg(target_arch = "wasm32")]
                    {
                        let _ = path_str;
                    }
                }
                IngestAction::LoadSelectedSpectra(plan) => {
                    self.execute_ingest_plan(plan, false);
                }
                IngestAction::ProcessPickAndRun(plan) => {
                    self.execute_ingest_plan(plan, true);
                }
                IngestAction::OpenSelected1dInStudio(plan) => {
                    let mut spectra = Vec::new();
                    #[cfg(not(target_arch = "wasm32"))]
                    for item in &plan.selected_experiments {
                        if matches!(
                            item.experiment.dimensionality,
                            ExperimentDimensionality::Dim1D
                        ) {
                            if let Some(ref p) = item.resolved_path {
                                if let Ok(loaded) = nmr_core::nmr1d::load_1d_input(p, None, false) {
                                    if let Ok(sp) =
                                        nmr_core::nmr1d::prepare_loaded_1d_for_studio(loaded)
                                    {
                                        spectra.push(sp);
                                        continue;
                                    }
                                }
                                if let Ok(bytes) = std::fs::read(p) {
                                    if let Some(sp) = try_load_1d_ucsf_or_pipe_bytes(
                                        &item.experiment.display_name,
                                        &bytes,
                                    ) {
                                        spectra.push(sp);
                                    }
                                }
                            }
                        }
                    }
                    if spectra.is_empty() && !plan.in_memory_files.is_empty() {
                        let sel_prefixes: Vec<String> = plan
                            .selected_experiments
                            .iter()
                            .filter(|s| {
                                matches!(
                                    s.experiment.dimensionality,
                                    ExperimentDimensionality::Dim1D
                                )
                            })
                            .map(|s| {
                                s.experiment
                                    .relative_path
                                    .trim_matches('/')
                                    .trim_start_matches("./")
                                    .to_string()
                            })
                            .collect();
                        let refs: Vec<(&str, &[u8])> = plan
                            .in_memory_files
                            .iter()
                            .filter(|(k, _)| {
                                sel_prefixes.iter().any(|pr| {
                                    pr.is_empty()
                                        || pr == "."
                                        || k.as_str() == pr
                                        || k.starts_with(&format!("{pr}/"))
                                        || k.contains(&format!("/{pr}/"))
                                })
                            })
                            .map(|(k, v)| (k.as_str(), v.as_slice()))
                            .collect();
                        if let Ok(loaded) =
                            nmr_core::nmr1d::load_1d_from_virtual_entries(&refs, None, false)
                        {
                            for it in loaded {
                                if let Ok(sp) = nmr_core::nmr1d::prepare_loaded_1d_for_studio(it) {
                                    spectra.push(sp);
                                }
                            }
                        }
                        if spectra.is_empty() {
                            for (k, v) in &refs {
                                if let Some(sp) = try_load_1d_ucsf_or_pipe_bytes(k, v) {
                                    spectra.push(sp);
                                }
                            }
                        }
                    }
                    if !spectra.is_empty() {
                        let n = spectra.len();
                        self.prepare_workbook_for_user_append();
                        self.stack1d_panel.load_user_spectra(spectra);
                        self.activate_1d_studio_mode();
                        self.status_msg =
                            format!("Opened {n} checked 1D experiment(s) in 1D NMR Studio");
                    } else {
                        self.status_msg =
                            "Could not load checked 1D rows — select valid 1D Bruker/JCAMP experiments.".to_string();
                    }
                }
            }
        }

        // 2. `Cmd/Ctrl+K` Fuzzy Command Palette (`T09.3`)
        if let Some(pal_act) = self.command_palette.show(ctx) {
            self.dispatch_workbench_action(ctx, pal_act);
        }

        // 3. `?` Keyboard Accelerator Cheat-Sheet (`T09.4`)
        if let Some(cs_act) = self.chord_dispatcher.show_cheat_sheet_window(ctx) {
            self.dispatch_workbench_action(ctx, cs_act);
        }

        let max_modal_w = (ctx.content_rect().width() - 16.0).max(240.0);

        // 4. `vt` Contour Threshold Popover
        if self.show_contour_modal {
            let mut open = self.show_contour_modal;
            egui::Window::new("Contour Threshold & Levels (vt)")
                .id(egui::Id::unique("nmr_contour_threshold_modal"))
                .open(&mut open)
                .collapsible(false)
                .default_width(340.0_f32.min(max_modal_w))
                .max_width(max_modal_w)
                .show(ctx, |ui| {
                    let lo = (self.contour_noise_ref * 0.15).max(1e-6);
                    let hi = (self.contour_noise_ref * 400.0)
                        .max(self.contour_base * 4.0)
                        .max(120.0);
                    ui.add(
                        egui::Slider::new(&mut self.contour_base, lo..=hi)
                            .logarithmic(true)
                            .text("Base Floor"),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.contour_factor, 1.12..=2.0).text("Step Ratio"),
                    );
                    ui.add(egui::Slider::new(&mut self.contour_levels, 2..=20).text("Levels"));
                    ui.checkbox(&mut self.show_negative, "Show Negative Contours");
                    if ui.button("Commit to Undo Stack").clicked() {
                        let mut params = SessionContourParams::new("HSQC");
                        params.base_level = self.contour_base;
                        params.level_factor = self.contour_factor;
                        params.num_levels = self.contour_levels;
                        params.show_negative = self.show_negative;
                        params.ppm_nudge = self.primary_ppm_nudge;
                        let msg = self.session.execute(WorkbenchCommand::SetContourParams {
                            layer_id: "HSQC".to_string(),
                            params,
                        });
                        self.status_msg = msg;
                    }
                });
            self.show_contour_modal = open;
        }

        // 5. Multi-Spectrum Overlay Stack Manager (`T09.5`)
        if self.show_overlay_manager {
            let mut open = self.show_overlay_manager;
            egui::Window::new("Multi-Spectrum GPU Contour Overlays")
                .id(egui::Id::unique("nmr_overlay_stack_manager"))
                .open(&mut open)
                .collapsible(false)
                .default_width(520.0_f32.min(max_modal_w))
                .max_width(max_modal_w)
                .show(ctx, |ui| {
                    let (nx_m, ny_m) = self.active_2d_nuclei();
                    let px_m = Self::pretty_nucleus_unicode(nx_m);
                    let py_m = Self::pretty_nucleus_unicode(ny_m);
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut self.primary_2d_visible, "");
                        ui.label(format!(
                            "Primary 2D ({}) Nudge (ppm):",
                            self.primary_2d_label
                        ));
                        ui.add(
                            egui::DragValue::new(&mut self.primary_ppm_nudge[0])
                                .speed(0.002)
                                .prefix(format!("Δ{px_m} ")),
                        );
                        ui.add(
                            egui::DragValue::new(&mut self.primary_ppm_nudge[1])
                                .speed(0.01)
                                .prefix(format!("Δ{py_m} ")),
                        );
                        if ui.small_button("Zero").clicked() {
                            self.primary_ppm_nudge = [0.0, 0.0];
                        }
                    });

                    ui.horizontal(|ui| {
                        if let Some(ref s3) = self.hncacb_3d {
                            if ui
                                .small_button("+ Add 3D HNCACB ¹H–¹⁵N Plane Slice")
                                .clicked()
                            {
                                let c_ppm = self.crosshair.c_ppm.unwrap_or(56.0);
                                let lid = self.next_layer_id;
                                self.next_layer_id += 1;
                                let pal_i = self.overlay_layers.len() + 1;
                                let base = (estimate_noise_mad(&s3.data) * 4.5).max(1.0);
                                let mut ov = OverlayLayerSpec::from_spectrum_3d_hn_plane(
                                    lid, 1, s3, c_ppm, base, pal_i,
                                );
                                let (pos, neg) = default_overlay_palette(pal_i);
                                ov.pos_rgba = pos;
                                ov.neg_rgba = neg;
                                self.overlay_layers
                                    .push((format!("HNCACB @ {c_ppm:.1} ppm"), ov));
                            }
                        }
                        if !self.overlay_layers.is_empty()
                            && ui.small_button("Clear Overlays").clicked()
                        {
                            self.overlay_layers.clear();
                        }
                    });
                    ui.separator();

                    for (idx, (lbl, ov)) in self.overlay_layers.iter_mut().enumerate() {
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut ov.visible, "");
                            ui.monospace(format!("#{} {lbl}", idx + 1));
                            let mut alpha = ov.pos_rgba[3];
                            if ui
                                .add(egui::Slider::new(&mut alpha, 0.15..=1.0).text("α"))
                                .changed()
                            {
                                ov.pos_rgba[3] = alpha;
                                ov.neg_rgba[3] = alpha;
                            }
                            let spd = (ov.base_level * 0.05).max(0.01);
                            ui.add(
                                egui::DragValue::new(&mut ov.base_level)
                                    .speed(spd)
                                    .range(1e-6..=1e12)
                                    .prefix("Base "),
                            );
                            ui.add(
                                egui::DragValue::new(&mut ov.ppm_nudge[0])
                                    .speed(0.002)
                                    .prefix(format!("Δ{px_m} ")),
                            );
                            ui.add(
                                egui::DragValue::new(&mut ov.ppm_nudge[1])
                                    .speed(0.01)
                                    .prefix(format!("Δ{py_m} ")),
                            );
                        });
                    }
                });
            self.show_overlay_manager = open;
        }

        // 6. `at` Residue Assignment & Lock Popover (`T09.6` / `T09.7`)
        if self.show_assign_popover {
            let mut open = self.show_assign_popover;
            egui::Window::new("Assign & Lock Spin System (at)")
                .id(egui::Id::unique("nmr_assign_lock_popover"))
                .open(&mut open)
                .collapsible(false)
                .default_width(360.0_f32.min(max_modal_w))
                .max_width(max_modal_w)
                .show(ctx, |ui| {
                    if let Some(ss) = self.selected_ss.filter(|&s| s < self.systems.len()) {
                        ui.label(format!(
                            "Active Spin System: #{ss} (¹H {:.3}, ¹⁵N {:.2})",
                            self.systems[ss].h, self.systems[ss].n
                        ));
                        if !self.sequence.is_empty() {
                            let max_k = self.sequence.len().saturating_sub(1);
                            self.assign_target_res =
                                self.assign_target_res.clamp(1.min(max_k), max_k);
                            egui::ComboBox::from_id_salt("assign_target_res_combo")
                                .selected_text(format!(
                                    "{}{}",
                                    self.sequence[self.assign_target_res],
                                    self.seq_ids[self.assign_target_res]
                                ))
                                .show_ui(ui, |ui| {
                                    for k in 1..self.sequence.len() {
                                        if self.sequence[k] == "PRO" {
                                            continue;
                                        }
                                        ui.selectable_value(
                                            &mut self.assign_target_res,
                                            k,
                                            format!("{}{}", self.sequence[k], self.seq_ids[k]),
                                        );
                                    }
                                });
                            ui.checkbox(&mut self.assign_lock_check, "Pin as hard solver lock");
                            ui.horizontal(|ui| {
                                if ui.button("Apply Assignment").clicked() {
                                    let target_k = self.assign_target_res;
                                    if self.assign_lock_check {
                                        let msg =
                                            self.session.execute(WorkbenchCommand::LockResidue {
                                                spin_system: ss,
                                                residue_seq_idx: target_k,
                                                locked: true,
                                            });
                                        self.status_msg = msg;
                                    } else {
                                        let msg = self.session.execute(
                                            WorkbenchCommand::AssignSpinSystem {
                                                spin_system: ss,
                                                residue_seq_idx: target_k,
                                            },
                                        );
                                        self.status_msg = msg;
                                    }
                                    self.sync_app_from_session();
                                    self.selected_res = Some(target_k);
                                }
                                if ui.button("Re-Solve with Locks").clicked() {
                                    self.run_assignment();
                                }
                            });
                        }
                    } else {
                        ui.label("Select an HSQC root on the 2D canvas first.");
                    }
                });
            self.show_assign_popover = open;
        }

        // 7. Dataset QC Report Modal
        if self.show_qc_modal {
            let mut open = self.show_qc_modal;
            egui::Window::new("Calibrated Dataset QC Gate")
                .id(egui::Id::unique("nmr_qc_report_modal"))
                .open(&mut open)
                .collapsible(false)
                .default_width(440.0_f32.min(max_modal_w))
                .max_width(max_modal_w)
                .show(ctx, |ui| {
                    if let Some(ref qc) = self.qc_report {
                        ui.strong(format!("Verdict: {}", qc.verdict));
                        ui.monospace(format!(
                            "3D HSQC Support: {:.1}% | Confidence Scale: {:.2}",
                            qc.hsqc_3d_support_frac * 100.0,
                            qc.confidence_scale
                        ));
                        for reason in &qc.reasons {
                            ui.label(format!("• {reason}"));
                        }
                    } else {
                        ui.label("Run backbone assignment to evaluate Dataset QC metrics.");
                    }
                });
            self.show_qc_modal = open;
        }

        // 8. About & Clean-Room Provenance Modal
        if self.show_about_modal {
            let mut open = self.show_about_modal;
            egui::Window::new("About FERRIC")
                .id(egui::Id::unique("nmr_about_modal"))
                .open(&mut open)
                .collapsible(false)
                .default_width(440.0_f32.min(max_modal_w))
                .max_width(max_modal_w)
                .show(ctx, |ui| {
                    ui.strong(format!("FERRIC v{}", nmr_core::VERSION));
                    if let Some(ref gpu_lbl) = self.adapter_label {
                        ui.monospace(format!("GPU Adapter: {gpu_lbl}"));
                    }
                    ui.label(
                        "Clean-room open-source WebGPU protein NMR resonance assignment, raw FID/NUS DSP, and Octet 2D chemical studio (MIT License).",
                    );
                });
            self.show_about_modal = open;
        }
    }

    fn suppress_water_active_1d(&mut self) {
        if self.stack1d_panel.traces.is_empty() {
            self.status_msg = "No 1D trace loaded for water suppression".to_string();
            return;
        }
        let idx = self
            .stack1d_panel
            .ref_trace_idx
            .min(self.stack1d_panel.traces.len() - 1);
        let Some(orig_spec) = self.stack1d_panel.traces[idx]
            .raw_spec
            .as_ref()
            .map(|a| a.0.clone())
        else {
            self.status_msg =
                "Active 1D trace is synthetic/table-only; load a 1D NMR spectrum to suppress water"
                    .to_string();
            return;
        };
        let mut spec = orig_spec.clone();
        if spec.len() < 8 {
            return;
        }
        let cfg = nmr_core::water::WaterSuppressionConfig::default();
        let center_ppm = nmr_core::water::suppress_water_spectrum_1d(&mut spec, &cfg);
        self.stack1d_panel.water_undo_snapshot = Some((idx, orig_spec));
        self.stack1d_panel.traces[idx].raw_spec = Some(AttachedSpectrum1d(spec.clone()));
        self.stack1d_panel.traces[idx].sync_from_spectrum1d(&spec);
        self.stack1d_panel
            .refresh_active_trace_peaks_and_multiplets();
        let msg = format!(
            "Suppressed residual water resonance at {center_ppm:.3} ppm on '{}' (Process -> Undo Water Suppression restores)",
            self.stack1d_panel.traces[idx].label
        );
        self.stack1d_panel.status_message = msg.clone();
        self.status_msg = msg;
    }

    fn suppress_water_active_spectrum(&mut self) {
        if self.is_1d_studio_active() {
            self.suppress_water_active_1d();
            return;
        }
        let Some(orig_2d) = self.hsqc_spec.clone() else {
            if !self.stack1d_panel.traces.is_empty() {
                self.suppress_water_active_1d();
            } else {
                self.status_msg = "No active 2D or 1D spectrum loaded".to_string();
            }
            return;
        };
        let (nx_lbl, ny_lbl) = self.active_2d_nuclei();
        if nx_lbl != "1H" && ny_lbl != "1H" {
            self.status_msg = format!(
                "Active 2D spectrum is {}–{} (no ¹H water axis to suppress)",
                Self::pretty_nucleus_unicode(nx_lbl),
                Self::pretty_nucleus_unicode(ny_lbl)
            );
            return;
        }
        let orig_cb = self.hncacb_3d.clone();
        let orig_co = self.hnco_3d.clone();
        let mut s2 = orig_2d.clone();
        let cfg = nmr_core::water::WaterSuppressionConfig::default();
        let reported_ppm = nmr_core::water::suppress_water_spectrum_2d(&mut s2, &cfg);
        if let Some(ref mut cb3) = self.hncacb_3d {
            let _ = nmr_core::water::suppress_water_spectrum_3d(cb3, &cfg);
        }
        if let Some(ref mut co3) = self.hnco_3d {
            let _ = nmr_core::water::suppress_water_spectrum_3d(co3, &cfg);
        }

        self.water_undo_2d = Some((orig_2d, orig_cb, orig_co));
        let (base, nref) = Self::estimate_2d_contour_base(&s2.data);
        self.contour_noise_ref = nref;
        self.contour_base = base;
        self.hsqc_arc = Arc::new(s2.data.clone());
        self.hsqc_gen = self.hsqc_gen.wrapping_add(1);
        self.workbook.update_active_2d_spectrum(s2.clone());
        self.hsqc_spec = Some(s2);
        self.save_live_2d_into_workbook();
        self.status_msg = format!(
            "Suppressed residual ¹H water & t₁ ridge at {reported_ppm:.3} ppm on '{}' (Process -> Undo Water Suppression restores)",
            self.primary_2d_label
        );
    }

    fn undo_water_suppression_active(&mut self) {
        if self.is_1d_studio_active() {
            if self.stack1d_panel.undo_water_suppression_active_trace() {
                self.status_msg = self.stack1d_panel.status_message.clone();
            } else {
                self.status_msg = "No 1D water suppression snapshot to undo".to_string();
            }
            return;
        }
        let Some((orig_2d, orig_cb, orig_co)) = self.water_undo_2d.take() else {
            if self.stack1d_panel.undo_water_suppression_active_trace() {
                self.status_msg = self.stack1d_panel.status_message.clone();
            } else {
                self.status_msg = "No water suppression snapshot to undo".to_string();
            }
            return;
        };
        let (base, nref) = Self::estimate_2d_contour_base(&orig_2d.data);
        self.contour_noise_ref = nref;
        self.contour_base = base;
        self.hsqc_arc = Arc::new(orig_2d.data.clone());
        self.hsqc_gen = self.hsqc_gen.wrapping_add(1);
        self.workbook.update_active_2d_spectrum(orig_2d.clone());
        self.hsqc_spec = Some(orig_2d);
        self.hncacb_3d = orig_cb;
        self.hnco_3d = orig_co;
        self.save_live_2d_into_workbook();
        self.status_msg = format!(
            "Restored pre-water-suppression 2D spectrum on '{}'",
            self.primary_2d_label
        );
    }

    fn draw_bottom_status_bar(&mut self, ui: &mut egui::Ui, now_ms: u64) {
        ui.horizontal(|ui| {
            self.chord_dispatcher.draw_status_bar_indicator(ui, now_ms);
            let (nx, ny) = self.active_2d_nuclei();
            let px = Self::pretty_nucleus_unicode(nx);
            let py = Self::pretty_nucleus_unicode(ny);
            ui.label(
                egui::RichText::new(format!(
                    "Crosshair: {px} {:.3} | {py} {:.2}{}",
                    self.crosshair.h_ppm,
                    self.crosshair.n_ppm,
                    if self.crosshair.locked {
                        " [Locked]"
                    } else {
                        ""
                    }
                ))
                .small()
                .monospace(),
            );
            ui.separator();
            ui.label(
                egui::RichText::new(format!(
                    "Undo: {} | Redo: {} | Locks: {}",
                    self.session.history.undo_len(),
                    self.session.history.redo_len(),
                    self.session.state.locked_residues.len()
                ))
                .small()
                .monospace()
                .weak(),
            );
            ui.separator();
            ui.label(
                egui::RichText::new(&self.status_msg)
                    .small()
                    .color(egui::Color32::from_rgb(190, 212, 240)),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(
                    egui::RichText::new("Cmd/Ctrl+K Palette  |  ? Shortcuts")
                        .small()
                        .monospace()
                        .weak(),
                );
            });
        });
    }
}

impl eframe::App for NmrWorkbenchApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        let now_ms = ctx.input(|i| (i.time.max(0.0) * 1000.0).round() as u64);
        if self.picker_wakeup_until_ms == u64::MAX - 1 {
            self.picker_wakeup_until_ms = now_ms.saturating_add(5_500);
        }
        if now_ms < self.picker_wakeup_until_ms {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }

        // 1. Global Cmd/Ctrl+K Command Palette toggle
        let toggle_palette = ctx.input_mut(|i| {
            let cmd = i.modifiers.command || i.modifiers.ctrl;
            cmd && !i.modifiers.alt
                && !i.modifiers.shift
                && i.consume_key(i.modifiers, egui::Key::K)
        });
        if toggle_palette {
            self.command_palette.toggle();
        }

        // 2. Global modifier shortcuts (Cmd/Ctrl+Z, Cmd/Ctrl+Shift+Z, Cmd/Ctrl+S, Cmd/Ctrl+O, ...)
        let octet_active = self.workspace_mode == WorkspaceMode::OctetChem;
        let mod_actions = poll_global_modifier_shortcuts(&ctx, octet_active);
        for act in mod_actions {
            self.dispatch_workbench_action(&ctx, act);
        }

        // 3. Modal Two-Key NMR Chord Dispatcher (runs across BOTH 2D and 1D views; suppressed in text inputs & Octet)
        let suppress_chords = ctx.egui_wants_keyboard_input()
            || self.workspace_mode != WorkspaceMode::ProteinNmr
            || self.command_palette.open
            || self.ingest_panel.open
            || self.show_text_io_modal
            || self.show_assign_popover
            || self.show_contour_modal
            || self.show_mcp_bridge_window;
        let chord_actions = self
            .chord_dispatcher
            .handle_egui_input(&ctx, suppress_chords);
        for act in chord_actions {
            self.dispatch_workbench_action(&ctx, act);
        }

        if let Some(rs) = _frame.wgpu_render_state() {
            self.wgpu_render_state = Some(rs.clone());
        }

        // 4. Drag-and-drop + Modals + MCP Live Workbench Bridge Synchronization
        self.apply_pending_mcp_bridge_commands();
        self.sync_mcp_bridge_snapshot_and_viewport();
        self.handle_dropped_files(&ctx);
        self.draw_text_io_window(&ctx);
        self.draw_phase9_modals(&ctx);
        self.draw_mcp_bridge_window(&ctx);

        // 5. Responsive Workbench Shell (`CompactMobile < 680`, `TabletSplit 680..1024`, `DesktopFull >= 1024`)
        let mut shell = std::mem::take(&mut self.responsive_shell);
        let _ = render_responsive_workbench(ui, now_ms, &mut shell, self);
        self.responsive_shell = shell;
    }
}

fn paint_mobile_octet_primitive(
    painter: &egui::Painter,
    pr: &VectorPrimitive,
    pt_to_scr: &dyn Fn([f64; 2]) -> egui::Pos2,
    pt_scale: f32,
) {
    let to_col = |c: RgbaColor| egui::Color32::from_rgba_unmultiplied(c.r, c.g, c.b, c.a);
    match pr {
        VectorPrimitive::Line {
            p1,
            p2,
            width_pt,
            color,
            dash_pt,
        } => {
            let s1 = pt_to_scr(*p1);
            let s2 = pt_to_scr(*p2);
            let stroke = egui::Stroke::new((*width_pt as f32 * pt_scale).max(1.0), to_col(*color));
            if let Some([d1, d2]) = dash_pt {
                let d_on = (*d1 as f32 * pt_scale).max(2.0);
                let d_off = (*d2 as f32 * pt_scale).max(2.0);
                let total = s1.distance(s2);
                if total > 0.5 {
                    let dir = (s2 - s1) / total;
                    let mut t = 0.0;
                    while t < total {
                        let e = (t + d_on).min(total);
                        painter.line_segment([s1 + dir * t, s1 + dir * e], stroke);
                        t += d_on + d_off;
                    }
                }
            } else {
                painter.line_segment([s1, s2], stroke);
            }
        }
        VectorPrimitive::Polygon {
            points,
            fill,
            stroke,
        } => {
            let pts: Vec<egui::Pos2> = points.iter().map(|&p| pt_to_scr(p)).collect();
            if pts.len() >= 3 {
                let fc = fill.map(to_col).unwrap_or(egui::Color32::TRANSPARENT);
                let st = stroke
                    .map(|(w, c)| egui::Stroke::new((w as f32 * pt_scale).max(1.0), to_col(c)))
                    .unwrap_or(egui::Stroke::NONE);
                painter.add(egui::Shape::convex_polygon(pts, fc, st));
            }
        }
        VectorPrimitive::Path {
            cmds, fill, stroke, ..
        } => {
            let subpaths = flatten_path_commands(cmds, 14);
            let st = stroke
                .map(|(w, c)| egui::Stroke::new((w as f32 * pt_scale).max(1.0), to_col(c)))
                .unwrap_or(egui::Stroke::NONE);
            for (pts, closed) in subpaths {
                let spts: Vec<egui::Pos2> = pts.iter().map(|&p| pt_to_scr(p)).collect();
                if spts.len() >= 3 && (closed || fill.is_some()) {
                    if let Some(fc) = fill {
                        painter.add(egui::Shape::convex_polygon(
                            spts.clone(),
                            to_col(*fc),
                            egui::Stroke::NONE,
                        ));
                    }
                }
                if st != egui::Stroke::NONE && spts.len() >= 2 {
                    painter.add(egui::Shape::line(spts, st));
                }
            }
        }
        VectorPrimitive::Circle {
            center,
            radius_pt,
            fill,
            stroke,
            ..
        } => {
            let c = pt_to_scr(*center);
            let r = (*radius_pt as f32 * pt_scale).max(1.0);
            if let Some(fc) = fill {
                painter.circle_filled(c, r, to_col(*fc));
            }
            if let Some((sw, sc)) = *stroke {
                painter.circle_stroke(
                    c,
                    r,
                    egui::Stroke::new((sw as f32 * pt_scale).max(1.0), to_col(sc)),
                );
            }
        }
        VectorPrimitive::Text {
            pos,
            text,
            font_size_pt,
            color,
            ..
        } => {
            let sp = pt_to_scr(*pos);
            let fsz = (*font_size_pt as f32 * pt_scale).clamp(7.5, 42.0);
            painter.text(
                sp,
                egui::Align2::LEFT_BOTTOM,
                text,
                egui::FontId::proportional(fsz),
                to_col(*color),
            );
        }
    }
}

impl NmrWorkbenchApp {
    fn draw_mobile_octet_2d_canvas(&mut self, ui: &mut egui::Ui) -> egui::Rect {
        egui::ScrollArea::horizontal()
            .id_salt("mobile_octet_2d_toolbar")
            .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysHidden)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 4.0;
                    ui.spacing_mut().interact_size.y = MOBILE_TOUCH_MIN_HEIGHT;
                    if ui
                        .add(
                            egui::Button::new("↩ Undo")
                                .min_size(egui::vec2(56.0, MOBILE_TOUCH_MIN_HEIGHT)),
                        )
                        .clicked()
                    {
                        self.octet_app.undo();
                    }
                    if ui
                        .add(
                            egui::Button::new("↪ Redo")
                                .min_size(egui::vec2(56.0, MOBILE_TOUCH_MIN_HEIGHT)),
                        )
                        .clicked()
                    {
                        self.octet_app.redo();
                    }
                    if ui
                        .add(
                            egui::Button::new("Clean 2D")
                                .min_size(egui::vec2(78.0, MOBILE_TOUCH_MIN_HEIGHT)),
                        )
                        .clicked()
                    {
                        self.octet_app.clean_2d_structure();
                    }
                    if ui
                        .add(
                            egui::Button::new("Peptide")
                                .min_size(egui::vec2(72.0, MOBILE_TOUCH_MIN_HEIGHT)),
                        )
                        .clicked()
                    {
                        self.octet_app
                            .set_loaded_protein_sequence(self.sequence.clone());
                        self.octet_app.import_from_loaded_protein();
                    }
                    if ui
                        .add(
                            egui::Button::new("Fit")
                                .min_size(egui::vec2(44.0, MOBILE_TOUCH_MIN_HEIGHT)),
                        )
                        .clicked()
                    {
                        self.octet_app.fit_camera_to_scene();
                    }
                    if ui
                        .add(
                            egui::Button::new("Clear")
                                .min_size(egui::vec2(58.0, MOBILE_TOUCH_MIN_HEIGHT)),
                        )
                        .clicked()
                    {
                        self.octet_app.set_scene(RenderScene::new());
                    }
                });
            });

        let avail = ui.available_size();
        let canvas_size = egui::vec2(avail.x.max(160.0), (avail.y - 4.0).max(160.0));
        let (rect, response) = ui.allocate_exact_size(canvas_size, egui::Sense::click_and_drag());
        let center_scr = rect.center();

        let mol_to_screen = |xm: f64, ym: f64, cam_c: [f64; 2], scale: f32| -> egui::Pos2 {
            egui::pos2(
                center_scr.x + ((xm - cam_c[0]) as f32) * scale,
                center_scr.y - ((ym - cam_c[1]) as f32) * scale,
            )
        };
        let screen_to_mol = |pos: egui::Pos2, cam_c: [f64; 2], scale: f32| -> [f64; 2] {
            [
                cam_c[0] + ((pos.x - center_scr.x) / scale.max(1.0)) as f64,
                cam_c[1] - ((pos.y - center_scr.y) / scale.max(1.0)) as f64,
            ]
        };

        if !ui.input(|i| i.pointer.any_down()) {
            self.suppress_octet_pan_until_release = false;
        }

        let multi_touch_active = if let Some(mt) = ui.ctx().input(|i| i.multi_touch()) {
            if mt.num_touches >= 2
                && (rect.contains(mt.start_pos)
                    || rect.contains(mt.center_pos)
                    || response.hovered()
                    || response.is_pointer_button_down_on())
            {
                self.suppress_octet_pan_until_release = true;
                let before_mol = screen_to_mol(
                    mt.center_pos,
                    self.octet_app.cam_center_mol,
                    self.octet_app.pixels_per_bond,
                );
                if mt.zoom_delta.is_finite() && mt.zoom_delta > 1e-4 {
                    self.octet_app.pixels_per_bond = (self.octet_app.pixels_per_bond
                        * mt.zoom_delta.clamp(0.4, 2.5))
                    .clamp(12.0, 110.0);
                }
                let after_mol = screen_to_mol(
                    mt.center_pos,
                    self.octet_app.cam_center_mol,
                    self.octet_app.pixels_per_bond,
                );
                let ppb = self.octet_app.pixels_per_bond.max(1.0);
                self.octet_app.cam_center_mol[0] +=
                    (before_mol[0] - after_mol[0]) - (mt.translation_delta.x / ppb) as f64;
                self.octet_app.cam_center_mol[1] +=
                    (before_mol[1] - after_mol[1]) + (mt.translation_delta.y / ppb) as f64;
                true
            } else {
                false
            }
        } else {
            false
        };

        if !multi_touch_active && !self.suppress_octet_pan_until_release {
            if response.dragged() {
                let delta = response.drag_delta();
                let ppb = self.octet_app.pixels_per_bond.max(1.0);
                self.octet_app.cam_center_mol[0] -= (delta.x / ppb) as f64;
                self.octet_app.cam_center_mol[1] += (delta.y / ppb) as f64;
            } else if response.clicked() {
                if let Some(pos) = response.interact_pointer_pos() {
                    let pm = screen_to_mol(
                        pos,
                        self.octet_app.cam_center_mol,
                        self.octet_app.pixels_per_bond,
                    );
                    let mut hit_atom: Option<(usize, f64)> = None;
                    for (idx, a) in self.octet_app.scene.atoms.iter().enumerate() {
                        if a.is_multi_center() {
                            continue;
                        }
                        let d = (a.x - pm[0]).hypot(a.y - pm[1]);
                        if d < 0.45 && hit_atom.is_none_or(|(_, best_d)| d < best_d) {
                            hit_atom = Some((idx, d));
                        }
                    }
                    self.octet_app.push_undo();
                    if let Some((ai, _)) = hit_atom {
                        let dir = self.octet_app.next_sprout_direction(ai);
                        let nx = self.octet_app.scene.atoms[ai].x + dir[0];
                        let ny = self.octet_app.scene.atoms[ai].y + dir[1];
                        let new_a = self.octet_app.scene.add_atom(SceneAtom::carbon(nx, ny));
                        let mut sb = SceneBond::new(ai, new_a, BondOrder::Single);
                        sb.stereo = BondStereo::None;
                        self.octet_app.scene.add_bond(sb);
                    } else {
                        let a0 = self
                            .octet_app
                            .scene
                            .add_atom(SceneAtom::carbon(pm[0], pm[1]));
                        let ang = 30.0_f64.to_radians();
                        let a1 = self
                            .octet_app
                            .scene
                            .add_atom(SceneAtom::carbon(pm[0] + ang.cos(), pm[1] + ang.sin()));
                        self.octet_app
                            .scene
                            .add_bond(SceneBond::new(a0, a1, BondOrder::Single));
                    }
                }
            }
        }
        if response.double_clicked() && !multi_touch_active {
            self.octet_app.fit_camera_to_scene();
        }

        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 3.0, egui::Color32::from_rgb(252, 253, 255));

        let cur_cam = self.octet_app.cam_center_mol;
        let cur_ppb = self.octet_app.pixels_per_bond.max(1.0);
        let compiled = compile_scene_primitives(&self.octet_app.scene, &self.octet_app.style);
        let bond_len_pt = self.octet_app.style.bond_length_pt.max(1.0);
        let pt_to_scr = |pt: [f64; 2]| -> egui::Pos2 {
            let raw_x_pt = pt[0] - compiled.shift_x_pt;
            let raw_y_pt = pt[1] - compiled.shift_y_pt;
            let xm = raw_x_pt / bond_len_pt;
            let ym = -raw_y_pt / bond_len_pt;
            mol_to_screen(xm, ym, cur_cam, cur_ppb)
        };
        let pt_scale = cur_ppb / (bond_len_pt as f32);
        for pr in &compiled.primitives {
            paint_mobile_octet_primitive(&painter, pr, &pt_to_scr, pt_scale);
        }

        painter.rect_stroke(
            rect,
            3.0,
            egui::Stroke::new(1.0, egui::Color32::from_rgb(203, 213, 225)),
            egui::StrokeKind::Inside,
        );
        rect
    }

    fn draw_mobile_chem_1d_qnmr(&mut self, ui: &mut egui::Ui) {
        if self.stack1d_panel.traces.is_empty() {
            let (traces, rois) = build_kinetics_demo_traces();
            self.stack1d_panel = Stack1dState::with_traces(traces);
            self.stack1d_panel.roi_regions = rois;
        }
        if let Some(props) = CoreOctetBackendAdapter.compute_properties(&self.octet_app.scene) {
            egui::Frame::group(ui.style())
                .fill(egui::Color32::from_rgb(20, 26, 37))
                .show(ui, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        ui.strong("qNMR Structure Summary:");
                        ui.monospace(&props.formula_hill);
                        ui.monospace(format!("MW {:.2} g/mol", props.molecular_weight));
                        ui.monospace(format!("Exact {:.4}", props.exact_mass));
                    });
                });
            ui.add_space(4.0);
        }
        self.render_and_sync_stack1d_panel(ui);
    }
}

impl ResponsiveWorkbenchHost for NmrWorkbenchApp {
    fn host_workspace_mode(&self) -> HostWorkspaceMode {
        match self.workspace_mode {
            WorkspaceMode::ProteinNmr => HostWorkspaceMode::ProteinNmr,
            WorkspaceMode::OctetChem => HostWorkspaceMode::OctetChem,
            WorkspaceMode::Dosy => HostWorkspaceMode::Dosy,
            WorkspaceMode::Metabolomics => HostWorkspaceMode::Metabolomics,
        }
    }

    fn set_host_workspace_mode(&mut self, mode: HostWorkspaceMode) {
        self.workspace_mode = match mode {
            HostWorkspaceMode::ProteinNmr => WorkspaceMode::ProteinNmr,
            HostWorkspaceMode::OctetChem => WorkspaceMode::OctetChem,
            HostWorkspaceMode::Dosy => WorkspaceMode::Dosy,
            HostWorkspaceMode::Metabolomics => WorkspaceMode::Metabolomics,
        };
    }

    fn workspace_readiness(&self) -> WorkspaceReadiness {
        self.compute_workspace_readiness()
    }

    fn active_residue_badge(&self) -> Option<String> {
        if let Some(k) = self.selected_res.filter(|&i| i < self.sequence.len()) {
            let sid = self.seq_ids.get(k).copied().unwrap_or((k + 1) as i32);
            Some(format!("{}{}", self.sequence[k], sid))
        } else if !self.sequence.is_empty() {
            Some(format!("{} aa", self.sequence.len()))
        } else {
            None
        }
    }

    fn status_banner_text(&self) -> &str {
        &self.status_msg
    }

    fn trigger_builtin_demo(&mut self) {
        self.load_builtin_demo_suite();
    }

    fn dispatch_chrome_action(&mut self, ctx: &egui::Context, action: WorkbenchAction) {
        self.dispatch_workbench_action(ctx, action);
    }

    fn show_mol3d_dock(&self) -> bool {
        self.show_mol3d_viewport && self.is_protein_2d_context()
    }

    fn show_strips_dock(&self) -> bool {
        self.show_strips_dock && self.is_protein_2d_context()
    }

    fn show_workflow_guide(&self) -> bool {
        self.show_workflow_guide
    }

    fn is_1d_full_canvas(&self) -> bool {
        self.is_1d_studio_active() || !self.is_protein_2d_context()
    }

    fn show_workbook_sidebar(&self) -> bool {
        self.workbook.show_sidebar
    }

    fn draw_workbook_sidebar(&mut self, ui: &mut egui::Ui) {
        let sel = self.current_workbook_selection();
        let has_3d = self.hncacb_3d.is_some() || self.hnco_3d.is_some();
        let has_pdb = self.loaded_pdb_text.is_some() || self.mol3d_panel.ensemble.is_some();
        let show_strips = self.show_strips_dock && self.is_protein_2d_context();
        let show_mol3d = self.show_mol3d_viewport && self.is_protein_2d_context();
        let actions = render_workbook_sidebar(
            ui,
            &mut self.workbook,
            &self.stack1d_panel,
            sel,
            has_3d,
            has_pdb,
            show_strips,
            show_mol3d,
        );
        self.handle_workbook_sidebar_actions(actions);
    }

    fn draw_desktop_mode_header(
        &mut self,
        ui: &mut egui::Ui,
        tablet_dock: Option<&mut TabletActiveDock>,
    ) {
        self.draw_workspace_mode_header(ui, tablet_dock);
    }

    fn draw_desktop_status_bar(&mut self, ui: &mut egui::Ui, now_ms: u64) {
        self.draw_bottom_status_bar(ui, now_ms);
    }

    fn draw_protein_top_controls_and_ribbon(&mut self, ui: &mut egui::Ui) {
        self.draw_top_bar(ui);
        if let Some(hint_msg) = self.bare_bruker_hint.clone() {
            ui.add_space(3.0);
            egui::Frame::new()
                .fill(egui::Color32::from_rgb(68, 36, 10))
                .stroke(egui::Stroke::new(
                    1.0,
                    egui::Color32::from_rgb(245, 158, 11),
                ))
                .corner_radius(4.0)
                .inner_margin(egui::Margin::symmetric(8, 5))
                .show(ui, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        ui.label(
                            egui::RichText::new("Bruker Folder Required:")
                                .strong()
                                .color(egui::Color32::from_rgb(254, 240, 138)),
                        );
                        ui.label(
                            egui::RichText::new(&hint_msg)
                                .small()
                                .color(egui::Color32::from_rgb(254, 249, 195)),
                        );
                        if ui.button("Open Folder").clicked() {
                            self.bare_bruker_hint = None;
                            self.arm_browser_picker_wakeup();
                            #[cfg(target_arch = "wasm32")]
                            trigger_browser_folder_picker();
                            #[cfg(not(target_arch = "wasm32"))]
                            {
                                self.show_native_global_open_bar = true;
                            }
                        }
                        if ui.small_button("Dismiss").clicked() {
                            self.bare_bruker_hint = None;
                        }
                    });
                });
        }
        if !self.axis_rescue_notices.is_empty() {
            ui.add_space(3.0);
            egui::Frame::new()
                .fill(egui::Color32::from_rgb(72, 46, 8))
                .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(234, 179, 8)))
                .corner_radius(4.0)
                .inner_margin(egui::Margin::symmetric(8, 4))
                .show(ui, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        ui.label(
                            egui::RichText::new("Axis Rescue Applied:")
                                .strong()
                                .color(egui::Color32::from_rgb(254, 240, 138)),
                        );
                        for msg in &self.axis_rescue_notices {
                            ui.label(
                                egui::RichText::new(msg)
                                    .small()
                                    .monospace()
                                    .color(egui::Color32::from_rgb(254, 249, 195)),
                            );
                        }
                        ui.checkbox(&mut self.enable_axis_rescue, "Enable Axis Rescue");
                        if ui.small_button("Dismiss").clicked() {
                            self.axis_rescue_notices.clear();
                        }
                    });
                });
        }
        let on_hsqc =
            !self.show_relax_panel && !self.show_dynamics_panel && !self.show_stack1d_panel;
        if on_hsqc && self.show_sequence_ribbon {
            ui.add_space(3.0);
            self.draw_sequence_ribbon(ui);
        }
    }

    fn draw_protein_desktop_central(&mut self, ui: &mut egui::Ui) -> egui::Rect {
        let mut relax_actions = Vec::new();
        let mut dyn_output: Option<DynamicsStudioOutput> = None;
        let body_avail = ui.available_rect_before_wrap();
        let viz_rect = if self.show_relax_panel {
            relax_actions = self.relax_panel.ui(
                ui,
                &self.sequence,
                &self.seq_ids,
                self.selected_res,
                self.loaded_pdb_text.as_deref(),
            );
            body_avail
        } else if self.show_dynamics_panel {
            if self.dynamics_panel.selected_seq_id.is_none() {
                if let Some(k) = self.selected_res {
                    self.dynamics_panel.selected_seq_id =
                        Some(self.seq_ids.get(k).copied().unwrap_or((k + 1) as i32));
                }
            }
            dyn_output = Some(self.dynamics_panel.show(ui));
            body_avail
        } else if self.show_stack1d_panel {
            self.render_and_sync_stack1d_panel(ui);
            body_avail
        } else {
            let r = self.draw_hsqc_canvas(ui);
            self.draw_2d_peaks_bottom_table(ui);
            r
        };
        if !relax_actions.is_empty() {
            self.apply_relax_panel_actions(relax_actions);
        }
        if let Some(out) = dyn_output {
            self.apply_dynamics_panel_output(out);
        }
        viz_rect
    }

    fn draw_hsqc_2d_view(&mut self, ui: &mut egui::Ui) -> egui::Rect {
        self.draw_hsqc_canvas(ui)
    }

    fn draw_strips_inspector_view(&mut self, ui: &mut egui::Ui) {
        self.draw_right_inspector(ui);
    }

    fn draw_mol3d_structure_view(&mut self, ui: &mut egui::Ui) -> egui::Rect {
        ui.horizontal(|ui| {
            ui.strong("3D Structure");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .small_button("× Hide")
                    .on_hover_text("Close 3D Structure dock (reopen via View -> 3D Structure Dock)")
                    .clicked()
                {
                    self.show_mol3d_viewport = false;
                    self.responsive_shell.tablet_dock = TabletActiveDock::Hidden;
                }
                if (self.loaded_pdb_text.is_some() || self.mol3d_panel.ensemble.is_some())
                    && ui
                        .small_button("Unload")
                        .on_hover_text("Clear currently loaded 3D PDB/mmCIF structure")
                        .clicked()
                {
                    self.loaded_pdb_text = None;
                    self.mol3d_panel = Mol3dViewerPanel::new();
                    self.status_msg = "Unloaded 3D structure.".to_string();
                }
            });
        });
        ui.separator();
        let display_ref = self.pred_ref.as_ref().or(self.reference.as_ref());
        let wgpu_rs = self.wgpu_render_state.clone();
        let picked = self.mol3d_panel.ui(
            ui,
            wgpu_rs.as_ref(),
            &self.sequence,
            &self.seq_ids,
            display_ref,
        );
        if let Some(sync) = picked {
            self.apply_mol3d_sync_action(sync);
        }
        self.mol3d_panel
            .last_canvas_rect
            .unwrap_or_else(|| ui.min_rect())
    }

    fn draw_relax_s2_view(&mut self, ui: &mut egui::Ui) {
        let actions = self.relax_panel.ui(
            ui,
            &self.sequence,
            &self.seq_ids,
            self.selected_res,
            self.loaded_pdb_text.as_deref(),
        );
        if !actions.is_empty() {
            self.apply_relax_panel_actions(actions);
        }
    }

    fn draw_dynamics_exchange_view(&mut self, ui: &mut egui::Ui) {
        if self.dynamics_panel.selected_seq_id.is_none() {
            if let Some(k) = self.selected_res {
                self.dynamics_panel.selected_seq_id =
                    Some(self.seq_ids.get(k).copied().unwrap_or((k + 1) as i32));
            }
        }
        let out = self.dynamics_panel.show(ui);
        self.apply_dynamics_panel_output(out);
    }

    fn draw_stack1d_view(&mut self, ui: &mut egui::Ui) {
        if self.stack1d_panel.traces.is_empty() {
            let (traces, rois) = build_kinetics_demo_traces();
            self.stack1d_panel = Stack1dState::with_traces(traces);
            self.stack1d_panel.roi_regions = rois;
        }
        self.render_and_sync_stack1d_panel(ui);
    }

    fn draw_mobile_seq_and_tools_view(
        &mut self,
        ui: &mut egui::Ui,
        readiness: &WorkspaceReadiness,
    ) -> Vec<WorkbenchAction> {
        let actions = draw_workflow_stepper(ui, readiness);
        ui.add_space(4.0);
        self.draw_sequence_ribbon(ui);
        ui.add_space(6.0);
        ui.separator();
        self.draw_top_bar(ui);
        actions
    }

    fn draw_octet_desktop_or_tablet(
        &mut self,
        ui: &mut egui::Ui,
        is_tablet: bool,
    ) -> (egui::Rect, Option<egui::Rect>, Option<egui::Rect>) {
        if is_tablet {
            // Keep at most 1 side dock (`octet_left_tool_palette`) open on TabletSplit
            self.octet_app.show_right_drawer = false;
        }
        let outer = ui.available_rect_before_wrap();
        self.octet_app.ui(ui);
        let left_w = 138.0_f32.min(outer.width() * 0.35);
        let left_rect =
            egui::Rect::from_min_max(outer.min, egui::pos2(outer.min.x + left_w, outer.max.y));
        let right_w = if self.octet_app.show_right_drawer {
            315.0_f32.min(outer.width() * 0.35)
        } else {
            0.0
        };
        let right_rect = (right_w > 0.0).then(|| {
            egui::Rect::from_min_max(egui::pos2(outer.max.x - right_w, outer.min.y), outer.max)
        });
        let central_rect = egui::Rect::from_min_max(
            egui::pos2(outer.min.x + left_w, outer.min.y),
            egui::pos2(outer.max.x - right_w, outer.max.y),
        );
        (central_rect, Some(left_rect), right_rect)
    }

    fn draw_mobile_octet_2d_view(&mut self, ui: &mut egui::Ui) -> egui::Rect {
        self.draw_mobile_octet_2d_canvas(ui)
    }

    fn draw_mobile_chem_1d_qnmr_view(&mut self, ui: &mut egui::Ui) {
        self.draw_mobile_chem_1d_qnmr(ui);
    }

    fn draw_dosy_view(&mut self, ui: &mut egui::Ui) {
        for action in render_dosy_panel(ui, &mut self.dosy_panel) {
            match action {
                DosyPanelAction::SelectedPeak { index, ppm } => {
                    self.status_msg =
                        format!("DOSY selected peak #{} at {:.3} ppm", index + 1, ppm);
                }
                DosyPanelAction::ToggledIlt { enabled } => {
                    self.status_msg = format!(
                        "DOSY Tikhonov ILT {}",
                        if enabled { "enabled" } else { "disabled" }
                    );
                }
                DosyPanelAction::LoadedDemoMixture => {
                    self.status_msg = "Loaded synthetic 3-species DOSY mixture".to_string();
                }
            }
        }
    }

    fn draw_metabolomics_view(&mut self, ui: &mut egui::Ui) {
        self.metabolomics_panel.ui(ui);
    }
}

impl NmrWorkbenchApp {
    fn resolve_seq_idx_by_id(&self, seq_id: i32) -> Option<usize> {
        self.seq_ids
            .iter()
            .position(|&sid| sid == seq_id)
            .or_else(|| {
                let idx = (seq_id - 1).max(0) as usize;
                (idx < self.sequence.len()).then_some(idx)
            })
    }

    fn apply_pending_mcp_bridge_commands(&mut self) {
        let pending = self.mcp_bridge.drain_pending_ui_commands();
        if pending.is_empty() {
            return;
        }
        for cmd in pending {
            match cmd {
                McpUiCommand::JumpToResidue { seq_id } => {
                    if let Some(res_idx) = self.resolve_seq_idx_by_id(seq_id) {
                        self.selected_res = Some(res_idx);
                        if let Some(ar) = &self.assign_res {
                            if let Some(ss_idx) =
                                ar.position.iter().position(|&pos| pos == res_idx as i32)
                            {
                                self.selected_ss = Some(ss_idx);
                                if let Some(sys) = self.systems.get(ss_idx) {
                                    self.crosshair.h_ppm = sys.h;
                                    self.crosshair.n_ppm = sys.n;
                                }
                            }
                        }
                    }
                    self.status_msg = format!("MCP jumped to residue Seq_ID {seq_id}");
                }
                McpUiCommand::SetCrosshairPpm {
                    h_ppm,
                    n_ppm,
                    c_ppm,
                } => {
                    self.crosshair.h_ppm = h_ppm;
                    self.crosshair.n_ppm = n_ppm;
                    self.crosshair.c_ppm = c_ppm;
                    self.status_msg =
                        format!("MCP moved crosshair to ({h_ppm:.3}, {n_ppm:.2}) ppm");
                }
                McpUiCommand::AssignSpinSystemToResidue {
                    cluster_id,
                    seq_id,
                    ref reason,
                } => {
                    if let Some(res_idx) = self.resolve_seq_idx_by_id(seq_id) {
                        if cluster_id < self.systems.len() {
                            let _ = self.session.execute(WorkbenchCommand::AssignSpinSystem {
                                spin_system: cluster_id,
                                residue_seq_idx: res_idx,
                            });
                            self.sync_app_from_session();
                        }
                    }
                    self.status_msg =
                        format!("MCP assigned cluster #{cluster_id} -> Seq_ID {seq_id} ({reason})");
                }
                McpUiCommand::UnassignResidue { seq_id, ref reason } => {
                    if let Some(res_idx) = self.resolve_seq_idx_by_id(seq_id) {
                        let _ = self.session.execute(WorkbenchCommand::UnassignResidue {
                            residue_seq_idx: res_idx,
                        });
                        self.sync_app_from_session();
                    }
                    self.status_msg = format!("MCP unassigned Seq_ID {seq_id} ({reason})");
                }
                McpUiCommand::TogglePeakExclusion { peak_id, excluded } => {
                    self.status_msg = format!(
                        "MCP {} peak #{peak_id}",
                        if excluded { "excluded" } else { "restored" }
                    );
                }
                McpUiCommand::HighlightResidues3d { ref seq_ids } => {
                    if let Some(&first_id) = seq_ids.first() {
                        if let Some(res_idx) = self.resolve_seq_idx_by_id(first_id) {
                            self.selected_res = Some(res_idx);
                        }
                    }
                    self.status_msg = format!("MCP highlighted {} 3D residues", seq_ids.len());
                }
            }
        }
    }

    fn sync_app_from_mcp_snapshot(&mut self, banner_msg: &str) {
        let snap = self.mcp_bridge.snapshot();
        self.crosshair = snap.to_sync_crosshair();
        if let Some(seq_id) = snap.active_residue_seq {
            if let Some(res_idx) = self.resolve_seq_idx_by_id(seq_id) {
                self.selected_res = Some(res_idx);
            }
        }
        self.status_msg = banner_msg.to_string();
    }

    fn sync_mcp_bridge_snapshot_and_viewport(&mut self) {
        let active_seq = self
            .selected_res
            .map(|idx| self.seq_ids.get(idx).copied().unwrap_or((idx + 1) as i32));
        let xh = self.crosshair;
        let qc_badge = self
            .qc_report
            .as_ref()
            .map(|r| format!("{:?}", r.verdict).to_ascii_uppercase())
            .unwrap_or_else(|| "UNRUN".to_string());
        let active_exp = match self.workspace_mode {
            WorkspaceMode::ProteinNmr => {
                if self.show_relax_panel {
                    "RELAX".to_string()
                } else if self.show_dynamics_panel {
                    "EXCHANGE".to_string()
                } else if self.show_stack1d_panel {
                    "NMR1D_STACK".to_string()
                } else {
                    "HSQC".to_string()
                }
            }
            WorkspaceMode::OctetChem => "OCTET_1D".to_string(),
            WorkspaceMode::Dosy => "DOSY_2D".to_string(),
            WorkspaceMode::Metabolomics => "METABOLOMICS_1D".to_string(),
        };

        self.mcp_bridge.update_snapshot(|snap| {
            snap.active_residue_seq = active_seq;
            snap.crosshair_ppm = (xh.h_ppm, xh.n_ppm, xh.c_ppm);
            snap.active_experiment = active_exp;
            snap.qc_status = qc_badge;
        });

        let vp_key = (
            self.hsqc_gen,
            (xh.h_ppm * 1000.0) as i32,
            (xh.n_ppm * 100.0) as i32,
            active_seq.unwrap_or(-1),
        );
        if vp_key != self.mcp_last_viewport_key {
            self.mcp_last_viewport_key = vp_key;
            let (w, h, rgba, label) = self.rasterize_mcp_viewport_rgba(192, 144);
            self.mcp_bridge.update_viewport_rgba(w, h, rgba, label);
        }
    }

    fn rasterize_mcp_viewport_rgba(
        &self,
        width: u32,
        height: u32,
    ) -> (u32, u32, Vec<u8>, &'static str) {
        let w = width.max(16);
        let h = height.max(16);
        let xh_tuple = (
            self.crosshair.h_ppm,
            self.crosshair.n_ppm,
            self.crosshair.c_ppm,
        );
        let Some(spec) = &self.hsqc_spec else {
            let card = ViewportFrameBuffer::diagnostic_card(w, h, xh_tuple);
            return (card.width, card.height, card.rgba, "diagnostic_contour_2d");
        };
        let nx = spec.nx();
        let ny = spec.ny();
        if nx == 0 || ny == 0 || spec.data.len() < nx.saturating_mul(ny) {
            let card = ViewportFrameBuffer::diagnostic_card(w, h, xh_tuple);
            return (card.width, card.height, card.rgba, "diagnostic_contour_2d");
        }

        let h_left = f64::from(self.view_h_left);
        let h_right = f64::from(self.view_h_right);
        let n_top = f64::from(self.view_n_top);
        let n_bottom = f64::from(self.view_n_bottom);
        let h_span = (h_left - h_right).abs().max(1e-6);
        let n_span = (n_bottom - n_top).abs().max(1e-6);
        let base_thr = f64::from(self.contour_base.abs().max(1e-6));

        let xh_px = (((h_left - self.crosshair.h_ppm) / h_span).clamp(0.0, 1.0) * f64::from(w - 1))
            .round() as u32;
        let xh_py = (((self.crosshair.n_ppm - n_top) / n_span).clamp(0.0, 1.0) * f64::from(h - 1))
            .round() as u32;

        let mut rgba = vec![0u8; (w as usize) * (h as usize) * 4];
        for py in 0..h {
            let n_ppm = n_top + (f64::from(py) / f64::from((h - 1).max(1))) * (n_bottom - n_top);
            let sy_f = if spec.axis_y.step_ppm.abs() > 1e-12 {
                (n_ppm - spec.axis_y.first_ppm) / spec.axis_y.step_ppm
            } else {
                0.0
            };
            let sy = sy_f.round().clamp(0.0, (ny - 1) as f64) as usize;

            for px in 0..w {
                let h_ppm =
                    h_left - (f64::from(px) / f64::from((w - 1).max(1))) * (h_left - h_right);
                let sx_f = if spec.axis_x.step_ppm.abs() > 1e-12 {
                    (h_ppm - spec.axis_x.first_ppm) / spec.axis_x.step_ppm
                } else {
                    0.0
                };
                let sx = sx_f.round().clamp(0.0, (nx - 1) as f64) as usize;
                let val = f64::from(spec.data[sy * nx + sx]);

                let on_border = px == 0 || py == 0 || px + 1 == w || py + 1 == h;
                let on_crosshair = px == xh_px || py == xh_py;
                let pixel = if on_border {
                    [58, 72, 96, 255]
                } else if on_crosshair {
                    [245, 194, 66, 255]
                } else if val >= base_thr {
                    let lvl = ((val / base_thr).ln().max(0.0) * 42.0).min(115.0) as u8;
                    [36, 140u8.saturating_add(lvl), 250, 255]
                } else if self.show_negative && val <= -base_thr {
                    let lvl = (((-val) / base_thr).ln().max(0.0) * 42.0).min(115.0) as u8;
                    [235, 75u8.saturating_add(lvl / 2), 95, 255]
                } else {
                    [16, 20, 28, 255]
                };
                let out_idx = ((py * w + px) as usize) * 4;
                rgba[out_idx..out_idx + 4].copy_from_slice(&pixel);
            }
        }
        (w, h, rgba, "contour_2d")
    }

    fn draw_mcp_bridge_window(&mut self, ctx: &egui::Context) {
        if !self.show_mcp_bridge_window {
            return;
        }
        let mut open = self.show_mcp_bridge_window;
        let max_modal_w = (ctx.content_rect().width() - 16.0).max(240.0);
        let mut panel_resp = None;
        egui::Window::new("MCP Live Workbench Bridge")
            .open(&mut open)
            .resizable(true)
            .default_width(520.0_f32.min(max_modal_w))
            .max_width(max_modal_w)
            .show(ctx, |ui| {
                panel_resp = Some(render_mcp_bridge_panel(ui, &mut self.mcp_bridge));
            });
        self.show_mcp_bridge_window = open;

        if let Some(resp) = panel_resp {
            if let Some(cmd) = resp.undone_command {
                let summary = cmd.summary();
                self.sync_app_from_mcp_snapshot(&format!("Undid AI action: {summary}"));
            }
            if let Some(cmd) = resp.redone_command {
                let summary = cmd.summary();
                self.sync_app_from_mcp_snapshot(&format!("Redid AI action: {summary}"));
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title(format!(
                "FERRIC v{} — WebGPU Protein NMR & Octet Chemical Studio",
                nmr_core::VERSION
            ))
            .with_icon(Arc::new(nmr_gui::brand_icon::build_ferric_window_icon()))
            .with_app_id("ferric-gui")
            .with_inner_size([1380.0, 860.0])
            .with_min_inner_size([360.0, 560.0]),
        ..Default::default()
    };
    eframe::run_native(
        "FERRIC",
        native_options,
        Box::new(|cc| Ok(Box::new(NmrWorkbenchApp::new(cc)))),
    )
}

#[cfg(target_arch = "wasm32")]
fn main() {
    use wasm_bindgen::JsCast as _;

    wasm_bindgen_futures::spawn_local(async {
        let Some(document) = web_sys::window().and_then(|w| w.document()) else {
            return;
        };

        let Some(canvas) = document
            .get_element_by_id("nmr_canvas")
            .and_then(|el| el.dyn_into::<web_sys::HtmlCanvasElement>().ok())
        else {
            if let Some(loading) = document.get_element_by_id("nmr_loading") {
                loading.set_inner_html(
                    "<p style=\"color:#ff7b7b;max-width:520px;text-align:center\">Missing <code>&lt;canvas id=\"nmr_canvas\"&gt;</code> element in demo page.</p>",
                );
            }
            return;
        };

        let web_options = eframe::WebOptions::default();
        let start_result = eframe::WebRunner::new()
            .start(
                canvas,
                web_options,
                Box::new(|cc| Ok(Box::new(NmrWorkbenchApp::new(cc)))),
            )
            .await;

        if let Some(loading) = document.get_element_by_id("nmr_loading") {
            match start_result {
                Ok(()) => loading.remove(),
                Err(e) => {
                    loading.set_inner_html(&format!(
                        "<p style=\"color:#ff7b7b;max-width:520px;text-align:center\">Failed to start WebGPU workbench: {:?}<br>\
                         Ensure WebGPU is enabled in your browser (Chrome/Edge 113+, Safari 18+, Firefox Nightly).</p>",
                        e
                    ));
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use nmr_core::aromatic::{
        default_his_tautomer_map, simulate_aromatic_peaks, synthesize_aromatic_reference,
        AromaticSimConfig,
    };
    use nmr_core::idp_propensity::{compute_idp_propensity, PropensityConfig};
    use nmr_core::nmrstar::Reference;
    use nmr_core::sidechain::synthesize_all_atom_reference;
    use nmr_core::topology::{ResidueVariant, SidechainPriors};
    use nmr_gui::sidechain_widget::SequenceRibbonState;

    use super::{
        build_synthetic_ubiquitin, merge_gui_aromatic_preserving_user_rings,
        merge_gui_aromatic_with_options, merge_gui_aromatic_with_ph, parse_gui_c13_tsv_str,
        GuiRegimeKind, GuiSolverKind,
    };

    #[test]
    fn test_gui_default_solver_is_evo_graph() {
        assert_eq!(GuiSolverKind::default(), GuiSolverKind::EvoGraph);
        assert_eq!(GuiSolverKind::default().label(), "Evo-Graph");
    }

    #[test]
    fn test_gui_p2_variant_state_separation_across_resolves() {
        let seq = vec!["CYS".to_string(), "HIS".to_string(), "ALA".to_string()];
        let seq_ids = vec![1, 2, 3];
        let mut r1 = Reference::new(seq.clone(), seq_ids.clone());
        r1.shifts.insert((0, "CB".to_string()), 41.2); // Oxidized Cys
        r1.shifts.insert((1, "ND1".to_string()), 249.5); // Ne2H His
        r1.shifts.insert((1, "NE2".to_string()), 167.8);

        let mut state = SequenceRibbonState::new();
        state.set_variant(0, ResidueVariant::CysReduced); // Explicit user override on Cys 1
        state.sync_inferred_variants_from_reference(&r1);

        // User override wins on Cys 1 while auto-inferred state is recorded cleanly underneath
        assert_eq!(
            state.auto_inferred_variants.get(&0),
            Some(&ResidueVariant::CysOxidized)
        );
        assert_eq!(state.variant_for(0), ResidueVariant::CysReduced);
        assert_eq!(state.variant_for(1), ResidueVariant::HisNe2H);

        // Re-solve where His 2 has no ring shifts anymore -> stale auto-inferred His variant clears!
        let mut r2 = Reference::new(seq, seq_ids);
        r2.shifts.insert((0, "CB".to_string()), 28.0);
        state.sync_inferred_variants_from_reference(&r2);
        assert_eq!(state.variant_for(1), ResidueVariant::Canonical);
        assert_eq!(state.variant_for(0), ResidueVariant::CysReduced);
    }

    #[test]
    fn test_gui_p1_12_aromatic_merge_preserves_unassigned_user_rings() {
        let seq = vec![
            "ALA".to_string(),
            "PHE".to_string(),
            "GLY".to_string(),
            "PHE".to_string(),
            "HIS".to_string(),
        ];
        let seq_ids = vec![1, 2, 3, 4, 5];
        let mut base = Reference::new(seq.clone(), seq_ids);
        for (k, ca, cb) in [
            (0, 52.5, 19.1),
            (1, 57.8, 39.4),
            (2, 45.2, 0.0),
            (3, 56.1, 42.3),
            (4, 55.4, 29.8),
        ] {
            base.shifts.insert((k, "CA".to_string()), ca);
            if cb > 0.0 {
                base.shifts.insert((k, "CB".to_string()), cb);
            }
        }
        let sc_priors = SidechainPriors::empirical_bmrb();
        let all_ref = synthesize_all_atom_reference(&base, &sc_priors, 77);
        let his_map = default_his_tautomer_map(&seq, 77);
        let truth_ref = synthesize_aromatic_reference(&all_ref, &sc_priors, &his_map, 77);

        // Simulate peaks ONLY for PHE at index 1 (drop peaks belonging to PHE index 3 and HIS index 4)
        let mut solve_source = truth_ref.clone();
        solve_source.shifts.retain(|&(k, ref a), _| {
            !(k >= 3 && !matches!(a.as_str(), "CA" | "CB" | "HA" | "HB2" | "HB3"))
        });
        let (arom_peaks, _) = simulate_aromatic_peaks(
            &solve_source,
            &BTreeMap::new(),
            &AromaticSimConfig::default(),
        );

        // User-curated shifts on unassigned PHE (idx 3) and HIS (idx 4)
        let mut user_ref = base.clone();
        user_ref.shifts.insert(
            (1, "HB2".to_string()),
            truth_ref.shifts[&(1, "HB2".to_string())],
        );
        user_ref.shifts.insert(
            (1, "HB3".to_string()),
            truth_ref.shifts[&(1, "HB3".to_string())],
        );
        user_ref.shifts.insert((3, "CD1".to_string()), 131.42);
        user_ref.shifts.insert((3, "HD1".to_string()), 7.26);
        user_ref.shifts.insert((4, "CE1".to_string()), 137.85);
        user_ref.shifts.insert((4, "HE1".to_string()), 8.14);
        let mut prior_amb = BTreeMap::new();
        prior_amb.insert((3, "CD1".to_string()), 2u8);
        prior_amb.insert((4, "CE1".to_string()), 1u8);

        let merged = merge_gui_aromatic_preserving_user_rings(
            &user_ref,
            Some(&user_ref),
            &prior_amb,
            None,
            &arom_peaks,
            &sc_priors,
        );

        // Observed PHE 2 is solved, while unobserved PHE 4 and HIS 5 preserve user ring shifts & ambiguity codes!
        assert!(merged.shifts.contains_key(&(1, "CD1".to_string())));
        assert_eq!(
            merged.shifts.get(&(3, "CD1".to_string())).copied(),
            Some(131.42)
        );
        assert_eq!(
            merged.shifts.get(&(3, "HD1".to_string())).copied(),
            Some(7.26)
        );
        assert_eq!(
            merged.shifts.get(&(4, "CE1".to_string())).copied(),
            Some(137.85)
        );
        assert_eq!(
            merged.shifts.get(&(4, "HE1".to_string())).copied(),
            Some(8.14)
        );
        assert_eq!(
            merged.ambiguity_codes.get(&(3, "CD1".to_string())).copied(),
            Some(2)
        );
        assert_eq!(
            merged.ambiguity_codes.get(&(4, "CE1".to_string())).copied(),
            Some(1)
        );
    }

    #[test]
    fn test_gui_phase11_regimes_c13_tsv_and_propensity_track() {
        assert_eq!(GuiRegimeKind::default(), GuiRegimeKind::Globular);
        assert_eq!(GuiRegimeKind::Idp13C.label(), "13C-IDP");
        assert_eq!(GuiRegimeKind::Paramagnetic.label(), "Paramagnetic");

        let tsv = "\
# exp\tdim1\tdim2\tdim3\tsign\theight
CON\t175.40\t122.10\t.\t1\t1200.0
CACO\t175.42\t55.10\t.\t1\t950.0
CANCO\t175.40\t122.11\t55.08\t1\t600.0
CBCACON\t175.41\t122.09\t30.40\t1\t520.0
";
        let ds = parse_gui_c13_tsv_str(tsv).expect("valid 13C-detected TSV");
        assert_eq!(ds.con_roots.len(), 1);
        assert_eq!(ds.caco_peaks.len(), 1);
        assert_eq!(ds.strip_peaks.len(), 2);
        assert_eq!(ds.strip_peaks[0].anchor_root_idx, Some(0));
        assert_eq!(ds.strip_peaks[1].anchor_root_idx, Some(0));

        let ubiq = build_synthetic_ubiquitin();
        let rep = compute_idp_propensity(&ubiq, &PropensityConfig::default());
        assert_eq!(rep.residues.len(), ubiq.sequence.len());
        for r in &rep.residues {
            assert!((r.population_sum() - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_gui_p2_explicit_canonical_override_persists_across_resolves() {
        let seq = vec!["CYS".to_string(), "HIS".to_string()];
        let seq_ids = vec![1, 2];
        let mut ref_shifts = Reference::new(seq, seq_ids);
        ref_shifts.shifts.insert((0, "CB".to_string()), 41.6); // Oxidized Cys signal
        ref_shifts.shifts.insert((1, "ND1".to_string()), 249.4); // His Ne2H signal
        ref_shifts.shifts.insert((1, "NE2".to_string()), 167.5);

        let mut state = SequenceRibbonState::new();
        state.sync_inferred_variants_from_reference(&ref_shifts);
        assert_eq!(state.variant_for(0), ResidueVariant::CysOxidized);
        assert_eq!(state.variant_for(1), ResidueVariant::HisNe2H);

        // User explicitly selects Canonical on both residues: must record explicit Canonical override!
        state.set_variant(0, ResidueVariant::Canonical);
        state.set_variant(1, ResidueVariant::Canonical);
        assert_eq!(
            state.user_variant_overrides.get(&0),
            Some(&ResidueVariant::Canonical)
        );
        assert_eq!(
            state.user_variant_overrides.get(&1),
            Some(&ResidueVariant::Canonical)
        );

        // Subsequent auto-sync must NEVER silently override explicit Canonical choice!
        state.sync_inferred_variants_from_reference(&ref_shifts);
        assert_eq!(state.variant_for(0), ResidueVariant::Canonical);
        assert_eq!(state.variant_for(1), ResidueVariant::Canonical);
    }

    #[test]
    fn test_gui_p1_10_curated_ring_protection_partial_and_duplicate_guards() {
        use nmr_core::peaks::Peak;

        let sc_priors = SidechainPriors::empirical_bmrb();
        let seq = vec![
            "ALA".to_string(),
            "PHE".to_string(),
            "GLY".to_string(),
            "PHE".to_string(),
        ];
        let seq_ids = vec![1, 2, 3, 4];

        // 1. Conflicting solver ring must NOT overwrite curated ring atoms unless overwrite_curated_rings == true
        let mut curated = Reference::new(seq.clone(), seq_ids.clone());
        curated.shifts.insert((1, "CA".to_string()), 57.5);
        curated.shifts.insert((1, "CB".to_string()), 39.2);
        curated.shifts.insert((1, "CD1".to_string()), 131.10);
        curated.shifts.insert((1, "CD2".to_string()), 131.10);
        curated.shifts.insert((1, "HD1".to_string()), 7.120);
        curated.shifts.insert((1, "HD2".to_string()), 7.120);
        curated.shifts.insert((1, "CE1".to_string()), 129.30);
        curated.shifts.insert((1, "CE2".to_string()), 129.30);
        curated.shifts.insert((1, "HE1".to_string()), 7.280);
        curated.shifts.insert((1, "HE2".to_string()), 7.280);
        curated.shifts.insert((1, "CZ".to_string()), 127.60);
        curated.shifts.insert((1, "HZ".to_string()), 7.210);
        let mut prior_amb = BTreeMap::new();
        prior_amb.insert((1, "CD1".to_string()), 1u8);

        let mk_peak = |exp: &str, h: f64, c: f64, height: f64| Peak {
            exp: exp.to_string(),
            h,
            n: 0.0,
            c: Some(c),
            h2: None,
            sign: 1,
            height: Some(height),
            volume: None,
        };

        // Experimental peaks with same CB (39.2) but shifted ring frequencies (7.45 / 7.56 / 7.39 ppm)
        let conflicting_peaks = vec![
            mk_peak("C13HSQC", 7.450, 132.40, 1000.0),
            mk_peak("C13HSQC", 7.560, 130.60, 1000.0),
            mk_peak("C13HSQC", 7.390, 128.90, 900.0),
            mk_peak("HBCBCGCDHD", 7.450, 39.20, 700.0),
            mk_peak("HBCBCGCDCEHE", 7.560, 39.20, 650.0),
        ];

        let kept = merge_gui_aromatic_with_options(
            &curated,
            Some(&curated),
            &prior_amb,
            None,
            &conflicting_peaks,
            &sc_priors,
            false,
        );
        assert_eq!(
            kept.shifts.get(&(1, "CD1".to_string())).copied(),
            Some(131.10)
        );
        assert_eq!(
            kept.shifts.get(&(1, "HD1".to_string())).copied(),
            Some(7.120)
        );
        assert_eq!(
            kept.shifts.get(&(1, "CE1".to_string())).copied(),
            Some(129.30)
        );
        assert_eq!(
            kept.shifts.get(&(1, "CZ".to_string())).copied(),
            Some(127.60)
        );
        assert_eq!(
            kept.ambiguity_codes.get(&(1, "CD1".to_string())).copied(),
            Some(1)
        );

        // Explicit overwrite_curated_rings = true replaces curated shifts with solved shifts
        let replaced = merge_gui_aromatic_with_options(
            &curated,
            Some(&curated),
            &prior_amb,
            None,
            &conflicting_peaks,
            &sc_priors,
            true,
        );
        assert!(
            (replaced.shifts[&(1, "HD1".to_string())] - 7.450).abs() < 0.03,
            "overwrite_curated_rings=true should replace curated HD1"
        );

        // 2. Partial CD-only cluster must NOT be mixed into curated CE/CZ shifts when overwrite_curated_rings == false
        let mut ce_cz_curated = Reference::new(seq.clone(), seq_ids.clone());
        ce_cz_curated.shifts.insert((1, "CA".to_string()), 57.5);
        ce_cz_curated.shifts.insert((1, "CB".to_string()), 39.2);
        ce_cz_curated.shifts.insert((1, "CE1".to_string()), 129.30);
        ce_cz_curated.shifts.insert((1, "CE2".to_string()), 129.30);
        ce_cz_curated.shifts.insert((1, "HE1".to_string()), 7.280);
        ce_cz_curated.shifts.insert((1, "HE2".to_string()), 7.280);
        ce_cz_curated.shifts.insert((1, "CZ".to_string()), 127.60);
        ce_cz_curated.shifts.insert((1, "HZ".to_string()), 7.210);
        let cd_only_peaks = vec![
            mk_peak("C13HSQC", 7.120, 131.10, 1000.0),
            mk_peak("HBCBCGCDHD", 7.120, 39.20, 800.0),
        ];
        let no_partial_mix = merge_gui_aromatic_with_options(
            &ce_cz_curated,
            Some(&ce_cz_curated),
            &BTreeMap::new(),
            None,
            &cd_only_peaks,
            &sc_priors,
            false,
        );
        assert!(
            !no_partial_mix.shifts.contains_key(&(1, "CD1".to_string()))
                && !no_partial_mix.shifts.contains_key(&(1, "HD1".to_string())),
            "Partial CD-only cluster must not be mixed into curated CE/CZ shifts"
        );
        assert_eq!(
            no_partial_mix.shifts.get(&(1, "CE1".to_string())).copied(),
            Some(129.30)
        );
        assert_eq!(
            no_partial_mix.shifts.get(&(1, "CZ".to_string())).copied(),
            Some(127.60)
        );

        // 3. Duplicate ring cluster matching curated PHE #1 (idx 1) must NOT be assigned onto uncurated PHE #2 (idx 3)
        let mut dup_ref = curated.clone();
        dup_ref.shifts.insert((3, "CA".to_string()), 56.8);
        dup_ref.shifts.insert((3, "CB".to_string()), 39.25); // Degenerate CB near PHE #1!
        let dup_peaks_matching_phe1 = vec![
            mk_peak("C13HSQC", 7.120, 131.10, 1000.0),
            mk_peak("C13HSQC", 7.280, 129.30, 1000.0),
            mk_peak("C13HSQC", 7.210, 127.60, 900.0),
            mk_peak("HBCBCGCDHD", 7.120, 39.22, 750.0),
            mk_peak("HBCBCGCDCEHE", 7.280, 39.22, 700.0),
        ];
        let dup_checked = merge_gui_aromatic_with_options(
            &dup_ref,
            Some(&dup_ref),
            &prior_amb,
            None,
            &dup_peaks_matching_phe1,
            &sc_priors,
            false,
        );
        assert_eq!(
            dup_checked.shifts.get(&(1, "HD1".to_string())).copied(),
            Some(7.120)
        );
        assert!(
            !dup_checked.shifts.contains_key(&(3, "CD1".to_string()))
                && !dup_checked.shifts.contains_key(&(3, "HD1".to_string())),
            "Cluster matching curated PHE #1 must not be assigned as duplicate onto PHE #2"
        );
    }

    #[test]
    fn test_gui_p1_10_superset_duplicate_tweaked_conflict_and_yamazaki_cb_offset() {
        use nmr_core::peaks::Peak;

        let sc_priors = SidechainPriors::empirical_bmrb();
        let mk_peak = |exp: &str, h: f64, c: f64, height: f64| Peak {
            exp: exp.to_string(),
            h,
            n: 0.0,
            c: Some(c),
            h2: None,
            sign: 1,
            height: Some(height),
            volume: None,
        };
        let mk_tocsy = |h: f64, c: f64, h2: f64, height: f64| Peak {
            exp: "AROM_HCCH_TOCSY".to_string(),
            h,
            n: 0.0,
            c: Some(c),
            h2: Some(h2),
            sign: 1,
            height: Some(height),
            volume: None,
        };

        let seq = vec![
            "ALA".to_string(),
            "PHE".to_string(),
            "GLY".to_string(),
            "PHE".to_string(),
            "TYR".to_string(),
        ];
        let seq_ids = vec![1, 2, 3, 4, 5];

        // (a) 4-slot curated PHE #1 (`CD1/HD1/CE1/HE1`, missing `CZ/HZ`) vs a 6-slot superset PHE cluster:
        //     must complete `CZ/HZ` on PHE #1 AND never duplicate onto uncurated PHE #2 (even with an
        //     overlapping peak or degenerate CB)!
        let mut ref_a = Reference::new(seq.clone(), seq_ids.clone());
        ref_a.shifts.insert((1, "CA".to_string()), 57.4);
        ref_a.shifts.insert((1, "CB".to_string()), 39.10);
        ref_a.shifts.insert((1, "CD1".to_string()), 131.20);
        ref_a.shifts.insert((1, "CD2".to_string()), 131.20);
        ref_a.shifts.insert((1, "HD1".to_string()), 7.140);
        ref_a.shifts.insert((1, "HD2".to_string()), 7.140);
        ref_a.shifts.insert((1, "CE1".to_string()), 129.40);
        ref_a.shifts.insert((1, "CE2".to_string()), 129.40);
        ref_a.shifts.insert((1, "HE1".to_string()), 7.300);
        ref_a.shifts.insert((1, "HE2".to_string()), 7.300);
        ref_a.shifts.insert((3, "CA".to_string()), 56.9);
        ref_a.shifts.insert((3, "CB".to_string()), 39.15); // Degenerate CB near PHE #1

        let superset_peaks = vec![
            mk_peak("C13HSQC", 7.140, 131.20, 1000.0),
            mk_peak("C13HSQC", 7.300, 129.40, 1000.0),
            mk_peak("C13HSQC", 7.220, 127.70, 900.0),
            mk_tocsy(7.140, 131.20, 7.300, 500.0),
            mk_tocsy(7.300, 129.40, 7.140, 500.0),
            mk_tocsy(7.300, 129.40, 7.220, 500.0),
            mk_tocsy(7.220, 127.70, 7.300, 500.0),
            mk_peak("HBCBCGCDHD", 7.140, 39.12, 750.0),
            mk_peak("HBCBCGCDCEHE", 7.300, 39.12, 700.0),
        ];

        let res_a = merge_gui_aromatic_with_options(
            &ref_a,
            Some(&ref_a),
            &BTreeMap::new(),
            None,
            &superset_peaks,
            &sc_priors,
            false,
        );
        assert_eq!(
            res_a.shifts.get(&(1, "CZ".to_string())).copied(),
            Some(127.70)
        );
        assert_eq!(
            res_a.shifts.get(&(1, "HZ".to_string())).copied(),
            Some(7.220)
        );
        assert!(
            !res_a.shifts.contains_key(&(3, "CD1".to_string()))
                && !res_a.shifts.contains_key(&(3, "HD1".to_string())),
            "6-slot superset cluster matching 4-slot curated PHE #1 must NOT duplicate onto uncurated PHE #2"
        );

        // (b) Curated PHE #1 with 1 user-adjusted slot slightly outside tol (`HE1 = 7.395` vs peak `7.300`,
        //     `conflict_obs == 1, shared_obs == 3`): must NOT overwrite PHE #1, must NOT mix `CZ/HZ` into
        //     PHE #1, AND must NOT leak/dump PHE #1's cluster onto uncurated PHE #2!
        let mut ref_b = ref_a.clone();
        ref_b.shifts.insert((1, "HE1".to_string()), 7.395);
        ref_b.shifts.insert((1, "HE2".to_string()), 7.395);
        let res_b = merge_gui_aromatic_with_options(
            &ref_b,
            Some(&ref_b),
            &BTreeMap::new(),
            None,
            &superset_peaks,
            &sc_priors,
            false,
        );
        assert_eq!(
            res_b.shifts.get(&(1, "HE1".to_string())).copied(),
            Some(7.395),
            "User-adjusted HE1 on PHE #1 must be preserved"
        );
        assert!(
            !res_b.shifts.contains_key(&(1, "CZ".to_string())),
            "Cluster with 1 conflicting slot must not fill missing CZ onto PHE #1"
        );
        assert!(
            !res_b.shifts.contains_key(&(3, "CD1".to_string()))
                && !res_b.shifts.contains_key(&(3, "HD1".to_string())),
            "PHE #1's cluster with 1 tweaked slot must NOT leak onto uncurated PHE #2"
        );

        // (c) Systematic Yamazaki `13CB` referencing offset (`+2.20 ppm`):
        //     `peak_matches_aromatic_cluster` must strip PHE #1's offset-shifted Yamazaki peaks and
        //     still calibrate & assign distinct uncurated PHE #2 (`idx 3`, CB 41.80) + TYR #3 (`idx 4`, CB 37.20)!
        let cb_off = 2.20;
        let mut ref_c = ref_a.clone();
        ref_c.shifts.insert((3, "CB".to_string()), 41.80);
        ref_c.shifts.insert((4, "CA".to_string()), 58.10);
        ref_c.shifts.insert((4, "CB".to_string()), 37.20);

        let offset_peaks = vec![
            // Curated PHE #1 (`idx 1`, CB 39.10 -> raw Yamazaki CB 41.30; notice 41.30 is close to PHE #2's true CB 41.80!)
            mk_peak("C13HSQC", 7.140, 131.20, 1000.0),
            mk_peak("C13HSQC", 7.300, 129.40, 1000.0),
            mk_peak("C13HSQC", 7.220, 127.70, 900.0),
            mk_tocsy(7.140, 131.20, 7.300, 500.0),
            mk_tocsy(7.300, 129.40, 7.140, 500.0),
            mk_tocsy(7.300, 129.40, 7.220, 500.0),
            mk_tocsy(7.220, 127.70, 7.300, 500.0),
            mk_peak("HBCBCGCDHD", 7.140, 39.10 + cb_off, 750.0),
            mk_peak("HBCBCGCDCEHE", 7.300, 39.10 + cb_off, 700.0),
            // Uncurated TYR #3 (`idx 4`, CB 37.20 -> raw Yamazaki CB 39.40) so >= 2 independent pairs establish +2.20 ppm offset!
            mk_peak("C13HSQC", 7.020, 132.60, 1000.0),
            mk_peak("C13HSQC", 6.780, 117.80, 1000.0),
            mk_tocsy(7.020, 132.60, 6.780, 500.0),
            mk_tocsy(6.780, 117.80, 7.020, 500.0),
            mk_peak("HBCBCGCDHD", 7.020, 37.20 + cb_off, 750.0),
            mk_peak("HBCBCGCDCEHE", 6.780, 37.20 + cb_off, 700.0),
        ];

        let res_c = merge_gui_aromatic_with_options(
            &ref_c,
            Some(&ref_c),
            &BTreeMap::new(),
            None,
            &offset_peaks,
            &sc_priors,
            false,
        );
        // Without cb_offset in `peak_matches_aromatic_cluster`, PHE #1's raw Yamazaki peak at 41.30 ppm
        // would NOT be stripped from `sub_peaks` and would falsely match uncurated PHE #2 (`CB = 41.80`)!
        assert!(
            !res_c.shifts.contains_key(&(3, "CD1".to_string()))
                && !res_c.shifts.contains_key(&(3, "HD1".to_string())),
            "Offset-shifted Yamazaki peak from curated PHE #1 (39.10 + 2.20 = 41.30) must NOT leak onto PHE #2 (CB 41.80)"
        );
        assert!(
            res_c
                .shifts
                .get(&(4, "HD1".to_string()))
                .is_some_and(|v| (v - 7.020).abs() < 0.03),
            "Uncurated TYR #3 should still receive its calibrated aromatic assignment"
        );
    }

    #[test]
    fn test_gui_p1_2_sample_ph_plumbing_into_aromatic_and_inspector() {
        let sc_priors = SidechainPriors::empirical_bmrb();
        let seq = vec!["ALA".to_string(), "HIS".to_string()];
        let seq_ids = vec![1, 2];
        let mut base = Reference::new(seq.clone(), seq_ids);
        base.shifts.insert((0, "CA".to_string()), 52.5);
        base.shifts.insert((0, "CB".to_string()), 19.1);
        base.shifts.insert((1, "CA".to_string()), 55.4);
        base.shifts.insert((1, "CB".to_string()), 29.0);
        base.shifts.insert((1, "HB2".to_string()), 3.12);
        base.shifts.insert((1, "HB3".to_string()), 3.04);

        let all_ref = synthesize_all_atom_reference(&base, &sc_priors, 12);
        let mut his_map = BTreeMap::new();
        his_map.insert(1, nmr_core::aromatic::HisTautomer::Ne2H);
        let truth_ref = synthesize_aromatic_reference(&all_ref, &sc_priors, &his_map, 12);
        let (arom_peaks, _) = simulate_aromatic_peaks(
            &truth_ref,
            &his_map,
            &AromaticSimConfig {
                seed: 12,
                ..AromaticSimConfig::default()
            },
        );

        let mut state = SequenceRibbonState::new();
        state.set_sample_ph(Some(6.8));
        let solved = merge_gui_aromatic_with_ph(
            &base,
            None,
            &BTreeMap::new(),
            None,
            &arom_peaks,
            &sc_priors,
            false,
            state.sample_ph,
        );
        state.sync_inferred_variants_from_reference(&solved);
        assert_eq!(state.sample_ph, Some(6.8));
        assert_eq!(state.variant_for(1), ResidueVariant::HisNe2H);
    }

    #[test]
    fn test_gui_solver_kind_bayes_sa_selector_and_solve() {
        use super::*;

        assert_eq!(GuiSolverKind::default(), GuiSolverKind::EvoGraph);
        assert_eq!(GuiSolverKind::BayesSa.label(), "Bayesian SA (Cv Adaptive)");

        let ref_data =
            read_star_str(SYNTHETIC_UBIQUITIN_STR, None).expect("parse ubiquitin fixture");
        let priors = Priors::provisional();
        let (peaks, truth) = simulate(
            &ref_data,
            &SimConfig {
                p_missing_peak: 0.0,
                spurious_rate: 0.0,
                seed: 42,
                ..SimConfig::default()
            },
        );
        let locked_idx = truth
            .iter()
            .enumerate()
            .find_map(|(ss, &k)| (k > 0).then_some((ss, k as usize)))
            .expect("at least one assignable root");
        let cfg = BayesSaConfig {
            replicas: 8,
            max_stages: 18,
            seed: 42,
            locked_assignments: vec![LockedAssignment::from(locked_idx)],
            ..BayesSaConfig::default()
        };
        let (sys, res, rep) = solve_bayes_sa_from_peaks(&peaks, &ref_data.sequence, &priors, &cfg)
            .expect("GUI Bayes-SA from peaks");
        assert!(!sys.is_empty());
        assert_eq!(res.position[locked_idx.0], locked_idx.1 as i32);
        assert!(rep.t_crit.is_finite() && rep.t_crit > 0.0);
        assert!(rep.max_cv.is_finite() && rep.max_cv >= 0.0);
        let m = score(&res, &truth, 0.8);
        assert!(
            m.accuracy() >= 0.90,
            "expected >=90% GUI Bayes-SA accuracy on synthetic ubiquitin, got {:.3}",
            m.accuracy()
        );
    }

    #[test]
    fn test_nmr_workbench_app_responsive_mobile_tablet_desktop_and_touch() {
        use super::*;
        use nmr_gui::workflow_bar::{
            MobileDynamicsSubTab, MobileProteinTab, ResponsiveBreakpoint, TabletActiveDock,
        };

        let mut octet_app = OctetDrawPanel::new();
        octet_app.set_backend_adapter(Box::new(CoreOctetBackendAdapter));
        let mut app = NmrWorkbenchApp {
            workspace_mode: WorkspaceMode::ProteinNmr,
            octet_app,
            dosy_panel: DosyPanelState::default(),
            metabolomics_panel: MetabolomicsStudioPanel::new(),
            solver_kind: GuiSolverKind::default(),
            regime_kind: GuiRegimeKind::default(),
            metal_ligands_input: String::new(),
            idp_temp_k: 298.15,
            idp_ph_input: "6.5".to_string(),
            c13_picked_dataset: None,
            propensity_report: None,
            show_propensity_panel: false,
            #[cfg(not(target_arch = "wasm32"))]
            use_gpu: false,
            #[cfg(not(target_arch = "wasm32"))]
            gpu_engine: None,
            #[cfg(not(target_arch = "wasm32"))]
            bayes_gpu_solver: None,
            use_neural_picker: true,
            enable_axis_rescue: true,
            axis_rescue_notices: Vec::new(),
            axis_rescue_audits: Vec::new(),
            adapter_label: None,
            status_msg: String::new(),
            last_solve_ms: None,
            last_bayes_sa_telemetry: None,
            priors: Priors::provisional(),
            sidechain_priors: SidechainPriors::empirical_bmrb(),
            reference: None,
            sequence: Vec::new(),
            seq_ids: Vec::new(),
            peaks: Vec::new(),
            truth: None,
            systems: Vec::new(),
            u_mat: None,
            p_mat: None,
            assign_res: None,
            pred_ref: None,
            pred_conf: BTreeMap::new(),
            pred_ambiguity_codes: BTreeMap::new(),
            metrics: None,
            sidechain_ribbon_state: SequenceRibbonState::new(),
            show_sidechain_inspector: false,
            overwrite_curated_rings: false,
            mol3d_panel: Mol3dViewerPanel::new(),
            show_mol3d_viewport: false,
            loaded_pdb_text: None,
            relax_panel: RelaxPanelState::new(),
            show_relax_panel: false,
            dynamics_panel: DynamicsStudioPanel::new(),
            show_dynamics_panel: false,
            stack1d_panel: Stack1dState::default(),
            show_stack1d_panel: true,
            show_strips_dock: false,
            show_workflow_guide: false,
            show_advanced_protein_bar: false,
            show_sequence_ribbon: true,
            paper_white_canvas: true,
            workbook: WorkbookState::default(),
            show_native_global_open_bar: false,
            hsqc_spec: None,
            hsqc_arc: Arc::new(Vec::new()),
            hsqc_gen: 1,
            hncacb_3d: None,
            hnco_3d: None,
            extra_3d_count: 0,
            overlay_layers: Vec::new(),
            primary_2d_label: "15N-HSQC".to_string(),
            primary_2d_visible: true,
            primary_ppm_nudge: [0.0, 0.0],
            contour_noise_ref: 1.8,
            box_zoom_2d_mode: false,
            box_zoom_2d_anchor: None,
            next_layer_id: 2,
            active_overlay_idx: 0,
            ingest_panel: IngestPanelState::default(),
            command_palette: CommandPalette::new(),
            chord_dispatcher: ChordDispatcher::new(),
            session: WorkbenchSessionController::default(),
            crosshair: SyncCrosshair::default(),
            strip_tiler: StripTilerPanel::new(),
            use_match_finder_strips: true,
            qc_report: None,
            raw_fids_total: 0,
            unprocessed_fids_remaining: 0,
            has_structure_or_dynamics: false,
            exported_in_session: false,
            show_contour_modal: false,
            show_overlay_manager: false,
            show_assign_popover: false,
            show_qc_modal: false,
            show_about_modal: false,
            mcp_bridge: McpBridgeState::default(),
            show_mcp_bridge_window: false,
            mcp_last_viewport_key: (0, 0, 0, -1),
            assign_target_res: 0,
            assign_lock_check: true,
            view_h_left: 10.4,
            view_h_right: 6.1,
            view_n_top: 103.0,
            view_n_bottom: 135.0,
            contour_base: 8.5,
            contour_factor: 1.42,
            contour_levels: 10,
            show_negative: true,
            show_peak_labels: true,
            replicas: 8,
            steps_per_sys: 400,
            assign_cost: 0.0,
            sim_missing: 0.06,
            sim_spurious: 0.04,
            sim_seed: 0,
            picker_snr: 5.0,
            selected_ss: None,
            selected_res: None,
            peak_2d_labels: BTreeMap::new(),
            context_menu_2d_ppm: None,
            secondary_drag_accum_2d: 0.0,
            bare_bruker_hint: None,
            picker_wakeup_until_ms: 0,
            water_undo_2d: None,
            #[cfg(not(target_arch = "wasm32"))]
            path_input: "data/fixtures/synthetic_ubiquitin.str".to_string(),
            #[cfg(not(target_arch = "wasm32"))]
            export_prefix: "result_gui".to_string(),
            show_text_io_modal: false,
            paste_text_buf: String::new(),
            export_text_buf: String::new(),
            responsive_shell: ResponsiveShellState::new(),
            wgpu_render_state: None,
            suppress_hsqc_pan_until_release: false,
            suppress_octet_pan_until_release: false,
            #[cfg(target_arch = "wasm32")]
            pending_drop_outcome: Arc::new(std::sync::Mutex::new(None)),
        };
        app.load_builtin_demo_suite();
        assert!(
            app.paper_white_canvas
                && app.show_stack1d_panel
                && !app.show_mol3d_viewport
                && !app.show_strips_dock,
            "Calm default startup must open Page 1 (1D 1H) on Paper White with side docks hidden"
        );
        assert_eq!(app.stack1d_panel.traces.len(), 2);
        assert_eq!(app.workbook.pages_2d.len(), 2);
        let mol_pep = nmr_core::octet_mol::build_peptide_2d("APY").expect("APY peptide");
        app.octet_app.scene = RenderScene::from(&mol_pep);

        let ctx = egui::Context::default();
        let run_app_frame = |ctx: &egui::Context,
                             w: f32,
                             h: f32,
                             events: Vec<egui::Event>,
                             app: &mut NmrWorkbenchApp| {
            let raw = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(w, h),
                )),
                events,
                ..Default::default()
            };
            let mut metrics_out = None;
            let _ = ctx.run_ui(raw, |ui| {
                let mut shell = std::mem::take(&mut app.responsive_shell);
                metrics_out = Some(render_responsive_workbench(ui, 100, &mut shell, app));
                app.responsive_shell = shell;
            });
            metrics_out.expect("frame rendered")
        };

        for (w, h) in [(375.0_f32, 667.0_f32), (390.0, 844.0), (412.0, 915.0)] {
            app.responsive_shell.mobile_family = MobileWorkspaceFamily::ProteinNmr;
            for tab in MobileProteinTab::ALL {
                app.responsive_shell.protein_tab = tab;
                if tab == MobileProteinTab::Dynamics1d {
                    for sub in MobileDynamicsSubTab::ALL {
                        app.responsive_shell.dynamics_sub_tab = sub;
                        let m = run_app_frame(&ctx, w, h, Vec::new(), &mut app);
                        assert_eq!(m.breakpoint, ResponsiveBreakpoint::CompactMobile);
                        assert!(m.left_dock_rect.is_none() && m.right_dock_rect.is_none());
                        assert!(m.active_viz_rect.width() >= 340.0);
                        assert!(m.active_viz_rect.height() >= 430.0);
                    }
                } else {
                    let m = run_app_frame(&ctx, w, h, Vec::new(), &mut app);
                    assert_eq!(m.breakpoint, ResponsiveBreakpoint::CompactMobile);
                    assert!(m.left_dock_rect.is_none() && m.right_dock_rect.is_none());
                    assert!(m.active_viz_rect.width() >= 340.0);
                    assert!(m.active_viz_rect.height() >= 430.0);
                }
            }

            app.responsive_shell.mobile_family = MobileWorkspaceFamily::ChemStudio;
            for ctab in MobileChemTab::ALL {
                app.responsive_shell.chem_tab = ctab;
                let m = run_app_frame(&ctx, w, h, Vec::new(), &mut app);
                assert_eq!(m.breakpoint, ResponsiveBreakpoint::CompactMobile);
                assert!(m.left_dock_rect.is_none() && m.right_dock_rect.is_none());
                assert!(m.active_viz_rect.width() >= 340.0);
                assert!(m.active_viz_rect.height() >= 430.0);
            }
        }

        // Switching to small-molecule 2D COSY page (index 0) keeps protein docks tucked away
        app.activate_2d_workbook_page(0);
        assert!(!app.show_stack1d_panel && !app.show_strips_dock);
        let md_cosy = run_app_frame(&ctx, 1440.0, 900.0, Vec::new(), &mut app);
        assert!(md_cosy.left_dock_rect.is_none() && md_cosy.right_dock_rect.is_none());

        // Switching to Ubiquitin 2D HSQC page (index 1) enables 3D Strips dock automatically
        app.activate_2d_workbook_page(1);
        assert!(!app.show_stack1d_panel && app.show_strips_dock);

        // Tablet 768x1024: single dock max + CentralPanel >= 420 px
        app.workspace_mode = WorkspaceMode::ProteinNmr;
        app.responsive_shell.tablet_dock = TabletActiveDock::StripsRight;
        let mt = run_app_frame(&ctx, 768.0, 1024.0, Vec::new(), &mut app);
        assert_eq!(mt.breakpoint, ResponsiveBreakpoint::TabletSplit);
        assert!(mt.left_dock_rect.is_none() && mt.right_dock_rect.is_some());
        assert!(mt.central_panel_rect.width() >= 420.0);

        // Desktop 1440x900: dual docks preserved when both docks are enabled
        app.show_mol3d_viewport = true;
        app.show_strips_dock = true;
        let md = run_app_frame(&ctx, 1440.0, 900.0, Vec::new(), &mut app);
        assert_eq!(md.breakpoint, ResponsiveBreakpoint::DesktopFull);
        assert!(md.left_dock_rect.is_some() && md.right_dock_rect.is_some());

        // 1D Studio full-canvas mode hides both side docks automatically
        app.activate_1d_workbook_page(0);
        let md_1d = run_app_frame(&ctx, 1440.0, 900.0, Vec::new(), &mut app);
        assert!(md_1d.left_dock_rect.is_none() && md_1d.right_dock_rect.is_none());
        app.activate_2d_workbook_page(1);

        // Live 2-finger pinch zoom on NmrWorkbenchApp's own `draw_hsqc_canvas`
        app.responsive_shell.mobile_family = MobileWorkspaceFamily::ProteinNmr;
        app.responsive_shell.protein_tab = MobileProteinTab::Contours2d;
        let m0 = run_app_frame(&ctx, 390.0, 844.0, Vec::new(), &mut app);
        let c = m0.active_viz_rect.center();
        let h_span_before = app.view_h_left - app.view_h_right;
        let d0 = egui::TouchDeviceId(99);
        let t1 = egui::TouchId(1);
        let t2 = egui::TouchId(2);
        let p1 = egui::pos2(c.x - 25.0, c.y - 25.0);
        let p2 = egui::pos2(c.x + 25.0, c.y + 25.0);
        let _ = run_app_frame(
            &ctx,
            390.0,
            844.0,
            vec![egui::Event::PointerMoved(p1)],
            &mut app,
        );
        let _ = run_app_frame(
            &ctx,
            390.0,
            844.0,
            vec![
                egui::Event::PointerButton {
                    pos: p1,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
                egui::Event::Touch {
                    device_id: d0,
                    id: t1,
                    phase: egui::TouchPhase::Start,
                    pos: p1,
                    force: None,
                },
                egui::Event::Touch {
                    device_id: d0,
                    id: t2,
                    phase: egui::TouchPhase::Start,
                    pos: p2,
                    force: None,
                },
            ],
            &mut app,
        );
        let _ = run_app_frame(
            &ctx,
            390.0,
            844.0,
            vec![
                egui::Event::PointerMoved(egui::pos2(c.x - 60.0, c.y - 60.0)),
                egui::Event::Touch {
                    device_id: d0,
                    id: t1,
                    phase: egui::TouchPhase::Move,
                    pos: egui::pos2(c.x - 60.0, c.y - 60.0),
                    force: None,
                },
                egui::Event::Touch {
                    device_id: d0,
                    id: t2,
                    phase: egui::TouchPhase::Move,
                    pos: egui::pos2(c.x + 60.0, c.y + 60.0),
                    force: None,
                },
            ],
            &mut app,
        );
        let h_span_after = app.view_h_left - app.view_h_right;
        assert!(
            h_span_after < h_span_before * 0.70,
            "NmrWorkbenchApp 2D HSQC pinch zoom must zoom in (before {h_span_before:.3}, after {h_span_after:.3})"
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn build_synthetic_ucsf_2d_test_bytes(
        nuc_0: &str,
        n_0: usize,
        tile_0: usize,
        obs_0: f32,
        sw_hz_0: f32,
        ref_ppm_0: f32,
        nuc_1: &str,
        n_1: usize,
        tile_1: usize,
        obs_1: f32,
        sw_hz_1: f32,
        ref_ppm_1: f32,
        peaks_01_idx: &[(usize, usize, f32)],
    ) -> Vec<u8> {
        let mut out = vec![0u8; 180 + 2 * 128];
        out[0..8].copy_from_slice(b"UCSF NMR");
        out[10] = 2; // ndim = 2
        out[11] = 1; // ncomponents = 1
        out[13] = 2; // version = 2
        let write_axis =
            |buf: &mut [u8], nuc: &str, n: usize, tile: usize, obs: f32, sw: f32, rppm: f32| {
                let nb = nuc.as_bytes();
                let clen = nb.len().min(6);
                buf[0..clen].copy_from_slice(&nb[..clen]);
                buf[8..12].copy_from_slice(&(n as u32).to_be_bytes());
                buf[16..20].copy_from_slice(&(tile as u32).to_be_bytes());
                buf[20..24].copy_from_slice(&obs.to_be_bytes());
                buf[24..28].copy_from_slice(&sw.to_be_bytes());
                buf[28..32].copy_from_slice(&rppm.to_be_bytes());
            };
        write_axis(
            &mut out[180..308],
            nuc_0,
            n_0,
            tile_0,
            obs_0,
            sw_hz_0,
            ref_ppm_0,
        );
        write_axis(
            &mut out[308..436],
            nuc_1,
            n_1,
            tile_1,
            obs_1,
            sw_hz_1,
            ref_ppm_1,
        );

        let nt0 = n_0.div_ceil(tile_0);
        let nt1 = n_1.div_ceil(tile_1);
        let mut grid = vec![0.02_f32; n_0 * n_1];
        for (i, slot) in grid.iter_mut().enumerate() {
            *slot = ((i as u32).wrapping_mul(2654435761) % 97) as f32 * 0.001;
        }
        for &(c0, c1, amp) in peaks_01_idx {
            for d0 in -3i32..=3 {
                for d1 in -3i32..=3 {
                    let p0 = c0 as i32 + d0;
                    let p1 = c1 as i32 + d1;
                    if p0 >= 0 && (p0 as usize) < n_0 && p1 >= 0 && (p1 as usize) < n_1 {
                        let g = (-0.45 * ((d0 * d0 + d1 * d1) as f32)).exp();
                        grid[(p0 as usize) * n_1 + (p1 as usize)] += amp * g;
                    }
                }
            }
        }
        for t0 in 0..nt0 {
            for t1 in 0..nt1 {
                for l0 in 0..tile_0 {
                    for l1 in 0..tile_1 {
                        let i0 = t0 * tile_0 + l0;
                        let i1 = t1 * tile_1 + l1;
                        let v = if i0 < n_0 && i1 < n_1 {
                            grid[i0 * n_1 + i1]
                        } else {
                            0.0
                        };
                        out.extend_from_slice(&v.to_be_bytes());
                    }
                }
            }
        }
        out
    }

    #[test]
    fn test_execute_ingest_plan_2d_con_and_homonuclear_noesy_tocsy_overlay() {
        use super::*;
        use nmr_gui::ingest_panel::{ingest_dropped_payloads, DroppedPayload, IngestDropOutcome};

        let mut octet_app = OctetDrawPanel::new();
        octet_app.set_backend_adapter(Box::new(CoreOctetBackendAdapter));
        let mut app = NmrWorkbenchApp {
            workspace_mode: WorkspaceMode::ProteinNmr,
            octet_app,
            dosy_panel: DosyPanelState::default(),
            metabolomics_panel: MetabolomicsStudioPanel::new(),
            solver_kind: GuiSolverKind::default(),
            regime_kind: GuiRegimeKind::default(),
            metal_ligands_input: String::new(),
            idp_temp_k: 298.15,
            idp_ph_input: "6.5".to_string(),
            c13_picked_dataset: None,
            propensity_report: None,
            show_propensity_panel: false,
            #[cfg(not(target_arch = "wasm32"))]
            use_gpu: false,
            #[cfg(not(target_arch = "wasm32"))]
            gpu_engine: None,
            #[cfg(not(target_arch = "wasm32"))]
            bayes_gpu_solver: None,
            use_neural_picker: false,
            enable_axis_rescue: true,
            axis_rescue_notices: Vec::new(),
            axis_rescue_audits: Vec::new(),
            adapter_label: None,
            status_msg: String::new(),
            last_solve_ms: None,
            last_bayes_sa_telemetry: None,
            priors: Priors::provisional(),
            sidechain_priors: SidechainPriors::empirical_bmrb(),
            reference: None,
            sequence: Vec::new(),
            seq_ids: Vec::new(),
            peaks: Vec::new(),
            truth: None,
            systems: Vec::new(),
            u_mat: None,
            p_mat: None,
            assign_res: None,
            pred_ref: None,
            pred_conf: BTreeMap::new(),
            pred_ambiguity_codes: BTreeMap::new(),
            metrics: None,
            sidechain_ribbon_state: SequenceRibbonState::new(),
            show_sidechain_inspector: false,
            overwrite_curated_rings: false,
            mol3d_panel: Mol3dViewerPanel::new(),
            show_mol3d_viewport: true,
            loaded_pdb_text: None,
            relax_panel: RelaxPanelState::new(),
            show_relax_panel: false,
            dynamics_panel: DynamicsStudioPanel::new(),
            show_dynamics_panel: false,
            stack1d_panel: Stack1dState::default(),
            show_stack1d_panel: true,
            show_strips_dock: false,
            show_workflow_guide: false,
            show_advanced_protein_bar: false,
            show_sequence_ribbon: true,
            paper_white_canvas: true,
            workbook: WorkbookState::default(),
            show_native_global_open_bar: false,
            hsqc_spec: None,
            hsqc_arc: Arc::new(Vec::new()),
            hsqc_gen: 1,
            hncacb_3d: None,
            hnco_3d: None,
            extra_3d_count: 0,
            overlay_layers: Vec::new(),
            primary_2d_label: "15N-HSQC".to_string(),
            primary_2d_visible: true,
            primary_ppm_nudge: [0.0, 0.0],
            contour_noise_ref: 1.8,
            box_zoom_2d_mode: false,
            box_zoom_2d_anchor: None,
            next_layer_id: 2,
            active_overlay_idx: 0,
            ingest_panel: IngestPanelState::default(),
            command_palette: CommandPalette::new(),
            chord_dispatcher: ChordDispatcher::new(),
            session: WorkbenchSessionController::default(),
            crosshair: SyncCrosshair::default(),
            strip_tiler: StripTilerPanel::new(),
            use_match_finder_strips: true,
            qc_report: None,
            raw_fids_total: 0,
            unprocessed_fids_remaining: 0,
            has_structure_or_dynamics: false,
            exported_in_session: false,
            show_contour_modal: false,
            show_overlay_manager: false,
            show_assign_popover: false,
            show_qc_modal: false,
            show_about_modal: false,
            mcp_bridge: McpBridgeState::default(),
            show_mcp_bridge_window: false,
            mcp_last_viewport_key: (0, 0, 0, -1),
            assign_target_res: 0,
            assign_lock_check: true,
            view_h_left: 10.4,
            view_h_right: 6.1,
            view_n_top: 103.0,
            view_n_bottom: 135.0,
            contour_base: 8.5,
            contour_factor: 1.42,
            contour_levels: 10,
            show_negative: true,
            show_peak_labels: true,
            replicas: 8,
            steps_per_sys: 400,
            assign_cost: 0.0,
            sim_missing: 0.06,
            sim_spurious: 0.04,
            sim_seed: 0,
            picker_snr: 5.0,
            selected_ss: None,
            selected_res: None,
            peak_2d_labels: BTreeMap::new(),
            context_menu_2d_ppm: None,
            secondary_drag_accum_2d: 0.0,
            bare_bruker_hint: None,
            picker_wakeup_until_ms: 0,
            water_undo_2d: None,
            #[cfg(not(target_arch = "wasm32"))]
            path_input: "data/fixtures/synthetic_ubiquitin.str".to_string(),
            #[cfg(not(target_arch = "wasm32"))]
            export_prefix: "result_gui".to_string(),
            show_text_io_modal: false,
            paste_text_buf: String::new(),
            export_text_buf: String::new(),
            responsive_shell: ResponsiveShellState::new(),
            wgpu_render_state: None,
            suppress_hsqc_pan_until_release: false,
            suppress_octet_pan_until_release: false,
            #[cfg(target_arch = "wasm32")]
            pending_drop_outcome: Arc::new(std::sync::Mutex::new(None)),
        };
        app.load_builtin_demo_suite();
        assert_eq!(app.sequence.len(), 76, "demo starts with 76-aa Ubiquitin");

        // 1. Homonuclear 2D `.usf` multi-overlay (`sample_noesy.usf` + `sample_tocsy.usf`)
        let noesy_usf = build_synthetic_ucsf_2d_test_bytes(
            "1H",
            32,
            16,
            600.0,
            6000.0,
            10.0,
            "1H",
            32,
            16,
            600.0,
            6000.0,
            10.0,
            &[(8, 22, 50.0), (22, 8, 50.0), (8, 8, 90.0)],
        );
        let tocsy_usf = build_synthetic_ucsf_2d_test_bytes(
            "1H",
            32,
            16,
            600.0,
            6000.0,
            10.0,
            "1H",
            32,
            16,
            600.0,
            6000.0,
            10.0,
            &[(10, 18, 45.0), (18, 10, 45.0), (10, 10, 85.0)],
        );
        let out_homo = ingest_dropped_payloads(&[
            DroppedPayload::new("sample_noesy.usf", noesy_usf),
            DroppedPayload::new("sample_tocsy.usf", tocsy_usf),
        ])
        .expect("2D NOESY + TOCSY .usf ingest");
        app.apply_drop_outcome(out_homo);
        assert!(app.ingest_panel.open);
        assert_eq!(
            app.ingest_panel.inventory.problem_regime,
            ProblemRegime::SmallMolecule2D
        );
        let plan_homo = app.ingest_panel.build_execution_plan();
        app.execute_ingest_plan(plan_homo, false);

        assert!(app.hsqc_spec.is_some(), "primary_2d must be promoted");
        assert_eq!(app.active_2d_nuclei(), ("1H", "1H"));
        assert_eq!(
            app.overlay_layers.len(),
            1,
            "second 2D spectrum must be overlaid"
        );
        assert!(app.overlay_layers[0].1.visible);
        assert!(
            app.sequence.is_empty() && app.peaks.is_empty() && app.hncacb_3d.is_none(),
            "stale Ubiquitin sequence/peaks/3D cubes must be cleared"
        );
        assert!(!app.show_mol3d_viewport && !app.show_strips_dock);

        // 2. Bruker `pdata/2` (non-`pdata/1` procno!) + `AXNUC = <off>` `2D CON` (`13C'` x `15N`, incl. Pro 137 ppm)
        let nx = 64usize;
        let ny = 64usize;
        let procs_con = format!(
            "##TITLE= c_con_iasq\n##$SI= {nx}\n##$OFFSET= 182.0\n##$SW_p= 2414.4\n##$SF= 150.90\n##$XDIM= 64\n##$BYTORDP= 0\n##$NC_proc= 0\n##$AXNUC= <off>\n##END=\n"
        );
        let proc2s_con = format!(
            "##TITLE= c_con_iasq\n##$SI= {ny}\n##$OFFSET= 142.0\n##$SW_p= 2431.9\n##$SF= 60.81\n##$XDIM= 64\n##$BYTORDP= 0\n##$NC_proc= 0\n##$AXNUC= <off>\n##END=\n"
        );
        let acqus_con = "##TITLE= con\n##$PULPROG= <c_con_iasq>\n##$NUC1= <13C>\n##END=\n";
        let acqu2s_con = "##TITLE= con\n##$NUC1= <15N>\n##END=\n";
        let mut rr_bytes = Vec::with_capacity(nx * ny * 4);
        // Place one amide root (~175 ppm C', ~120 ppm N) and one Pro imino root (~174 ppm C', ~137 ppm N)
        // OFFSET=182, step = -16/63 = -0.25397 -> ix=28 => ~174.9 ppm
        // OFFSET=142, step = -40/63 = -0.63492 -> iy=8 => ~136.9 ppm (Pro!), iy=34 => ~120.4 ppm
        for iy in 0..ny {
            for ix in 0..nx {
                let mut val: f64 = (((ix * 31 + iy * 17) % 19) as f64) * 5.0;
                for &(px, py, amp) in &[(28i32, 34i32, 85_000.0_f64), (30i32, 8i32, 92_000.0_f64)] {
                    let dx = (ix as i32 - px) as f64;
                    let dy = (iy as i32 - py) as f64;
                    if dx.abs() <= 4.0 && dy.abs() <= 4.0 {
                        val += amp * (-0.35 * (dx * dx + dy * dy)).exp();
                    }
                }
                rr_bytes.extend_from_slice(&(val as i32).to_le_bytes());
            }
        }
        let out_con = ingest_dropped_payloads(&[
            DroppedPayload::new("idp_study/12/acqus", acqus_con.as_bytes().to_vec()),
            DroppedPayload::new("idp_study/12/acqu2s", acqu2s_con.as_bytes().to_vec()),
            DroppedPayload::new("idp_study/12/pdata/2/procs", procs_con.as_bytes().to_vec()),
            DroppedPayload::new(
                "idp_study/12/pdata/2/proc2s",
                proc2s_con.as_bytes().to_vec(),
            ),
            DroppedPayload::new("idp_study/12/pdata/2/2rr", rr_bytes),
        ])
        .expect("Bruker pdata/2 2D CON ingest");
        let panel_con = match out_con {
            IngestDropOutcome::Workspace(p) => p,
            other => panic!("expected Workspace for 2D CON, got {other:?}"),
        };
        assert_eq!(
            panel_con.inventory.problem_regime,
            ProblemRegime::Idp13cDetected
        );
        assert_eq!(
            panel_con.inventory.experiments[0].pulse_class,
            PulseSequenceClass::Con2D
        );
        let plan_con = panel_con.build_execution_plan();
        app.execute_ingest_plan(plan_con, false);

        assert_eq!(app.regime_kind, GuiRegimeKind::Idp13C);
        assert_eq!(app.active_2d_nuclei(), ("13C", "15N"));
        assert!(app.view_h_left > 180.0 && app.view_h_right < 168.0);
        assert!(app.view_n_top < 105.0 && app.view_n_bottom > 140.0);
        let ds = app
            .c13_picked_dataset
            .as_ref()
            .expect("2D CON roots must be picked automatically");
        assert!(
            ds.con_roots.len() >= 2,
            "expected >= 2 picked CON roots, got {}",
            ds.con_roots.len()
        );
        assert!(
            ds.con_roots.iter().any(|r| r.is_proline_imino_region),
            "must detect Proline imino peak at ~137 ppm"
        );
    }

    fn make_headless_test_app() -> super::NmrWorkbenchApp {
        use super::*;
        let mut octet_app = OctetDrawPanel::new();
        octet_app.set_backend_adapter(Box::new(CoreOctetBackendAdapter));
        NmrWorkbenchApp {
            workspace_mode: WorkspaceMode::ProteinNmr,
            octet_app,
            dosy_panel: DosyPanelState::default(),
            metabolomics_panel: MetabolomicsStudioPanel::new(),
            solver_kind: GuiSolverKind::default(),
            regime_kind: GuiRegimeKind::default(),
            metal_ligands_input: String::new(),
            idp_temp_k: 298.15,
            idp_ph_input: "6.5".to_string(),
            c13_picked_dataset: None,
            propensity_report: None,
            show_propensity_panel: false,
            #[cfg(not(target_arch = "wasm32"))]
            use_gpu: false,
            #[cfg(not(target_arch = "wasm32"))]
            gpu_engine: None,
            #[cfg(not(target_arch = "wasm32"))]
            bayes_gpu_solver: None,
            use_neural_picker: false,
            enable_axis_rescue: true,
            axis_rescue_notices: Vec::new(),
            axis_rescue_audits: Vec::new(),
            adapter_label: None,
            status_msg: String::new(),
            last_solve_ms: None,
            last_bayes_sa_telemetry: None,
            priors: Priors::provisional(),
            sidechain_priors: SidechainPriors::empirical_bmrb(),
            reference: None,
            sequence: Vec::new(),
            seq_ids: Vec::new(),
            peaks: Vec::new(),
            truth: None,
            systems: Vec::new(),
            u_mat: None,
            p_mat: None,
            assign_res: None,
            pred_ref: None,
            pred_conf: BTreeMap::new(),
            pred_ambiguity_codes: BTreeMap::new(),
            metrics: None,
            sidechain_ribbon_state: SequenceRibbonState::new(),
            show_sidechain_inspector: false,
            overwrite_curated_rings: false,
            mol3d_panel: Mol3dViewerPanel::new(),
            show_mol3d_viewport: true,
            loaded_pdb_text: None,
            relax_panel: RelaxPanelState::new(),
            show_relax_panel: false,
            dynamics_panel: DynamicsStudioPanel::new(),
            show_dynamics_panel: false,
            stack1d_panel: Stack1dState::default(),
            show_stack1d_panel: false,
            show_strips_dock: false,
            show_workflow_guide: false,
            show_advanced_protein_bar: false,
            show_sequence_ribbon: true,
            paper_white_canvas: true,
            workbook: WorkbookState::default(),
            show_native_global_open_bar: false,
            hsqc_spec: None,
            hsqc_arc: Arc::new(Vec::new()),
            hsqc_gen: 1,
            hncacb_3d: None,
            hnco_3d: None,
            extra_3d_count: 0,
            overlay_layers: Vec::new(),
            primary_2d_label: "15N-HSQC".to_string(),
            primary_2d_visible: true,
            primary_ppm_nudge: [0.0, 0.0],
            contour_noise_ref: 1.8,
            box_zoom_2d_mode: false,
            box_zoom_2d_anchor: None,
            next_layer_id: 2,
            active_overlay_idx: 0,
            ingest_panel: IngestPanelState::default(),
            command_palette: CommandPalette::new(),
            chord_dispatcher: ChordDispatcher::new(),
            session: WorkbenchSessionController::default(),
            crosshair: SyncCrosshair::default(),
            strip_tiler: StripTilerPanel::new(),
            use_match_finder_strips: true,
            qc_report: None,
            raw_fids_total: 0,
            unprocessed_fids_remaining: 0,
            has_structure_or_dynamics: false,
            exported_in_session: false,
            show_contour_modal: false,
            show_overlay_manager: false,
            show_assign_popover: false,
            show_qc_modal: false,
            show_about_modal: false,
            mcp_bridge: McpBridgeState::default(),
            show_mcp_bridge_window: false,
            mcp_last_viewport_key: (0, 0, 0, -1),
            assign_target_res: 0,
            assign_lock_check: true,
            view_h_left: 10.4,
            view_h_right: 6.1,
            view_n_top: 103.0,
            view_n_bottom: 135.0,
            contour_base: 8.5,
            contour_factor: 1.42,
            contour_levels: 10,
            show_negative: true,
            show_peak_labels: true,
            replicas: 8,
            steps_per_sys: 400,
            assign_cost: 0.0,
            sim_missing: 0.06,
            sim_spurious: 0.04,
            sim_seed: 0,
            picker_snr: 5.0,
            selected_ss: None,
            selected_res: None,
            peak_2d_labels: BTreeMap::new(),
            context_menu_2d_ppm: None,
            secondary_drag_accum_2d: 0.0,
            bare_bruker_hint: None,
            picker_wakeup_until_ms: 0,
            water_undo_2d: None,
            #[cfg(not(target_arch = "wasm32"))]
            path_input: "data/fixtures/synthetic_ubiquitin.str".to_string(),
            #[cfg(not(target_arch = "wasm32"))]
            export_prefix: "result_gui".to_string(),
            show_text_io_modal: false,
            paste_text_buf: String::new(),
            export_text_buf: String::new(),
            responsive_shell: ResponsiveShellState::new(),
            wgpu_render_state: None,
            suppress_hsqc_pan_until_release: false,
            suppress_octet_pan_until_release: false,
            #[cfg(target_arch = "wasm32")]
            pending_drop_outcome: Arc::new(std::sync::Mutex::new(None)),
        }
    }

    #[test]
    fn test_path_sanitization_bare_2rr_error_and_single_con_auto_open() {
        use super::*;
        use nmr_gui::ingest_panel::{ingest_dropped_payloads, DroppedPayload};

        // 1. Path sanitization: quotes + macOS terminal backslash-escaped spaces + Windows paths
        #[cfg(not(target_arch = "wasm32"))]
        {
            assert_eq!(
                sanitize_user_path_input("'/Users/test/My\\ Drive/nmrdata/10003/pdata/1'"),
                "/Users/test/My Drive/nmrdata/10003/pdata/1"
            );
            assert_eq!(
                sanitize_user_path_input("\"C:\\nmrdata\\10003\\pdata\\1\""),
                "C:\\nmrdata\\10003\\pdata\\1"
            );
        }

        // 2. Bare `2rr` selection returns actionable folder hint instead of silent 0-experiment modal
        let bare_err = ingest_dropped_payloads(&[DroppedPayload::new("2rr", vec![0u8; 4096])])
            .expect_err("bare 2rr without procs/proc2s must return actionable error");
        assert!(
            is_bare_bruker_file_error(&bare_err),
            "unexpected bare 2rr message: {bare_err}"
        );
        let mut app = make_headless_test_app();
        app.record_ingest_error("Open error", bare_err);
        assert!(
            app.bare_bruker_hint.is_some(),
            "bare Bruker file error must arm prominent Open Folder banner"
        );

        // 3. High-NC_proc (`2.5e9`) + >60% zero-padded 2D array produces finite, non-blackout contour_base & noise_ref
        let mut high_nc = vec![0.0_f32; 128 * 128];
        for (i, cell) in high_nc.iter_mut().enumerate().take(4_500) {
            let sign = if i % 2 == 0 { 1.0_f32 } else { -1.0_f32 };
            *cell = sign * (1.2e7_f32 + ((i * 37) % 1000) as f32 * 8.0e4_f32);
        }
        high_nc[6_200] = 2.5e9_f32;
        let (base_nc, nref_nc) = estimate_contour_base_and_noise(&high_nc);
        assert!(base_nc.is_finite() && base_nc > 1.0e7 && base_nc < 5.0e8);
        assert!(nref_nc.is_finite() && nref_nc >= base_nc / 6.1);

        // 4. Single Bruker `2D CON` folder auto-opens directly onto the 2D canvas without blocking modal
        let nx = 48usize;
        let ny = 48usize;
        let procs = format!(
            "##TITLE= c_hcacon_ia\n##$SI= {nx}\n##$OFFSET= 182.0\n##$SW_p= 2414.4\n##$SF= 201.26\n##$XDIM= 48\n##$BYTORDP= 0\n##$NC_proc= 3\n##END=\n"
        );
        let proc2s = format!(
            "##TITLE= c_hcacon_ia\n##$SI= {ny}\n##$OFFSET= 142.0\n##$SW_p= 3244.0\n##$SF= 81.10\n##$XDIM= 48\n##$BYTORDP= 0\n##$NC_proc= 3\n##END=\n"
        );
        let mut rr = Vec::with_capacity(nx * ny * 4);
        for iy in 0..ny {
            for ix in 0..nx {
                let mut v = (((ix * 19 + iy * 11) % 17) as i32 - 8) * 1_500;
                let dx = ix as f64 - 22.0;
                let dy = iy as f64 - 24.0;
                if dx.abs() <= 3.0 && dy.abs() <= 3.0 {
                    v += (650_000.0 * (-0.45 * (dx * dx + dy * dy)).exp()) as i32;
                }
                rr.extend_from_slice(&v.to_le_bytes());
            }
        }
        let out_single = ingest_dropped_payloads(&[
            DroppedPayload::new("10003/pdata/1/procs", procs.into_bytes()),
            DroppedPayload::new("10003/pdata/1/proc2s", proc2s.into_bytes()),
            DroppedPayload::new("10003/pdata/1/2rr", rr),
        ])
        .expect("single pdata/1 CON ingest");
        app.apply_drop_outcome(out_single);
        assert!(
            !app.ingest_panel.open,
            "single 2D CON drop without sequence must open directly onto 2D canvas"
        );
        assert!(app.hsqc_spec.is_some());
        assert!(app.contour_noise_ref > 100.0);
        assert!(
            app.peaks.iter().any(|p| p.c.is_none()),
            "CON roots must be synced into editable 2D peaks"
        );
    }

    #[test]
    fn test_ridge_safe_homonuclear_2d_picker_and_interactive_2d_curation_plus_water_undo() {
        use super::*;
        use eframe::egui;

        let nx = 128usize;
        let ny = 128usize;
        let axis_x = nmr_core::spectrum::AxisPpm {
            label: "1H".to_string(),
            first_ppm: 10.7,
            step_ppm: -12.0 / (nx - 1) as f64,
            obs_mhz: 500.13,
            size: nx,
        };
        let axis_y = axis_x.clone();
        let mut data = vec![0.0_f32; nx * ny];
        let mut rng_state = 0x9E37_79B9_7F4A_7C15_u64;
        for cell in &mut data {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = ((rng_state >> 32) as f32 / (u32::MAX as f32)) - 0.5;
            *cell = u * 20.0;
        }
        // Inject massive H2O t1 + t2 ridge at 4.70 ppm + diagonal ridge
        let w_idx = axis_x.ppm_to_index(4.70).round() as isize;
        for iy in 0..ny {
            let py = axis_y.index_to_ppm(iy as f64);
            for ix in 0..nx {
                let px = axis_x.index_to_ppm(ix as f64);
                let dwx = (ix as isize - w_idx) as f32;
                let dwy = (iy as isize - w_idx) as f32;
                if dwx.abs() <= 2.0 {
                    data[iy * nx + ix] += 55_000.0 * (-0.7 * dwx * dwx).exp();
                }
                if dwy.abs() <= 2.0 {
                    data[iy * nx + ix] += 42_000.0 * (-0.7 * dwy * dwy).exp();
                }
                let d_diag = ((px - py) / 0.03) as f32;
                if d_diag.abs() <= 2.5 {
                    data[iy * nx + ix] += 80_000.0 * (-0.8 * d_diag * d_diag).exp();
                }
            }
        }
        // Inject 4 genuine off-diagonal cross-peaks
        let true_crosspeaks = [
            (8.20_f64, 4.15_f64, 8_500.0_f32),
            (4.15_f64, 8.20_f64, 8_200.0_f32),
            (7.35_f64, 2.85_f64, 6_400.0_f32),
            (2.85_f64, 7.35_f64, 6_100.0_f32),
        ];
        for &(cx_ppm, cy_ppm, amp) in &true_crosspeaks {
            let cxi = axis_x.ppm_to_index(cx_ppm);
            let cyi = axis_y.ppm_to_index(cy_ppm);
            for iy in 0..ny {
                for ix in 0..nx {
                    let dx = (ix as f64 - cxi) as f32;
                    let dy = (iy as f64 - cyi) as f32;
                    if dx.abs() <= 4.0 && dy.abs() <= 4.0 {
                        data[iy * nx + ix] += amp * (-0.55 * (dx * dx + dy * dy)).exp();
                    }
                }
            }
        }
        let noesy = Spectrum2D {
            name: "NOESY_70".to_string(),
            axis_x,
            axis_y,
            data,
        };

        let (c_base, _) = estimate_contour_base_and_noise(&noesy.data);
        let picked =
            pick_2d_crosspeaks_ridge_safe(&noesy, 5.0, c_base.min(800.0), true, false, false);
        assert_eq!(
            picked.len(),
            4,
            "ridge-safe NOESY picker must recover all 4 cross-peaks with zero water-stripe or diagonal artifacts, got {:?}",
            picked
                .iter()
                .map(|p| (p.h, p.n, p.height))
                .collect::<Vec<_>>()
        );
        assert!(picked[0].height.unwrap_or(0.0) >= picked[1].height.unwrap_or(0.0));
        for &(tx, ty, _) in &true_crosspeaks {
            assert!(
                picked
                    .iter()
                    .any(|p| (p.h - tx).abs() < 0.08 && (p.n - ty).abs() < 0.08),
                "missing cross-peak ({tx}, {ty})"
            );
        }

        // Interactive 2D peak add, move, label, snap, TSV export, delete & water suppression + undo
        let mut app = make_headless_test_app();
        app.install_single_2d_spectrum(
            noesy,
            "NOESY_70.ucsf".to_string(),
            "Loaded NOESY".to_string(),
        );
        app.contour_base = 500.0;
        app.pick_2d_roots();
        assert_eq!(app.peaks.len(), 4);

        let man_idx = app.add_manual_2d_peak(6.50, 1.95, None);
        assert_eq!(app.peaks.len(), 5);
        app.peak_2d_labels.insert(
            peak_label_key_2d(app.peaks[man_idx].h, app.peaks[man_idx].n),
            "Val12-HG".to_string(),
        );
        app.move_2d_peak_live(man_idx, 8.18, 4.17, true);
        app.snap_2d_peak_by_index(man_idx);
        assert!((app.peaks[man_idx].h - 8.20).abs() < 0.08);
        assert!((app.peaks[man_idx].n - 4.15).abs() < 0.08);
        assert!(app.export_2d_peaks_tsv().contains("Val12-HG"));
        app.delete_2d_peak(man_idx);
        assert_eq!(app.peaks.len(), 4);

        // Suppress water ridge on 2D NOESY -> verify >85% attenuation along 4.70 ppm & exact undo
        let w_col = app
            .hsqc_spec
            .as_ref()
            .map(|s| s.axis_x.ppm_to_index(4.70).round() as usize)
            .unwrap_or(0);
        let pre_water_amp = app
            .hsqc_spec
            .as_ref()
            .map(|s| s.get(w_col, 10).abs())
            .unwrap_or(0.0);
        assert!(pre_water_amp > 20_000.0);
        app.suppress_water_active_spectrum();
        assert!(app.water_undo_2d.is_some());
        let post_water_amp = app
            .hsqc_spec
            .as_ref()
            .map(|s| s.get(w_col, 10).abs())
            .unwrap_or(pre_water_amp);
        assert!(
            post_water_amp < pre_water_amp * 0.15,
            "expected >85% water ridge suppression, got {pre_water_amp} -> {post_water_amp}"
        );
        app.undo_water_suppression_active();
        assert!(app.water_undo_2d.is_none());
        let restored_amp = app
            .hsqc_spec
            .as_ref()
            .map(|s| s.get(w_col, 10).abs())
            .unwrap_or(0.0);
        assert!((restored_amp - pre_water_amp).abs() < 1e-3);

        // Headless egui render pass over 2D canvas + bottom 2D peak table
        app.workbook.two_d.selected_peak_idx = Some(0);
        let egui_ctx = egui::Context::default();
        let _ = egui_ctx.run_ui(egui::RawInput::default(), |ui| {
            let _ = app.draw_hsqc_canvas(ui);
            app.draw_2d_peaks_bottom_table(ui);
        });
    }

    #[test]
    fn test_env_driven_readonly_dataset_verification() {
        use super::*;
        let Ok(spec_list) = std::env::var("FERRIC_READONLY_VERIFY_PATHS") else {
            return;
        };
        #[cfg(not(target_arch = "wasm32"))]
        for raw_p in spec_list
            .split('|')
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            let p = Path::new(raw_p);
            assert!(p.exists(), "verification path missing: {}", p.display());
            let mut app = make_headless_test_app();
            let outcome = load_native_path_1d_or_workspace(p)
                .unwrap_or_else(|e| panic!("failed to classify {}: {e}", p.display()));
            app.apply_drop_outcome(outcome);
            if app.is_1d_studio_active() {
                assert!(!app.stack1d_panel.traces.is_empty());
                app.suppress_water_active_spectrum();
                assert!(app.stack1d_panel.water_undo_snapshot.is_some());
                app.undo_water_suppression_active();
                println!(
                    "[VERIFY-1D] {} -> traces={} peaks={}",
                    p.display(),
                    app.stack1d_panel.traces.len(),
                    app.stack1d_panel.picked_peaks.len()
                );
            } else {
                assert!(
                    !app.ingest_panel.open && app.hsqc_spec.is_some(),
                    "single 2D dataset {} must auto-open directly onto canvas (status: {})",
                    p.display(),
                    app.status_msg
                );
                app.pick_2d_roots();
                let (nx, ny) = app.active_2d_nuclei();
                let total_2d = app.peaks.iter().filter(|pk| pk.c.is_none()).count();
                let water_band_hits = app
                    .peaks
                    .iter()
                    .filter(|pk| {
                        pk.c.is_none()
                            && ((4.67..=4.73).contains(&pk.h) || (4.67..=4.73).contains(&pk.n))
                    })
                    .count();
                let water_note = if nx == "1H" || ny == "1H" {
                    app.suppress_water_active_spectrum();
                    assert!(app.water_undo_2d.is_some());
                    let m = app.status_msg.clone();
                    app.undo_water_suppression_active();
                    assert!(app.water_undo_2d.is_none());
                    m
                } else {
                    "n/a".to_string()
                };
                println!(
                    "[VERIFY-2D] {} -> nuclei=({nx},{ny}) base={:.3e} nref={:.3e} peaks={total_2d} water_band_hits={water_band_hits} status='{}' water='{water_note}'",
                    p.display(),
                    app.contour_base,
                    app.contour_noise_ref,
                    app.status_msg
                );
            }
        }
    }

    #[test]
    fn test_workbook_append_preserves_5_open_pages_on_new_drop() {
        use super::*;
        use nmr_core::nmr1d::{Experiment1DKind, Spectrum1D};
        use nmr_gui::ingest_panel::{ingest_dropped_payloads, DroppedPayload};

        let mut app = make_headless_test_app();
        app.load_builtin_demo_suite();
        assert_eq!(app.stack1d_panel.traces.len(), 2);
        assert_eq!(app.workbook.pages_2d.len(), 2);

        // 1. Drop 3 user 1D spectra (built-in demo 1D + 2D pages are evicted once on first user drop)
        let mk_1d = |name: &str, center_ppm: f64| -> Spectrum1D {
            let n = 128usize;
            let axis = nmr_core::spectrum::AxisPpm::from_bounds("1H", n, 10.0, 0.0, 500.0);
            let mut real = vec![0.0_f64; n];
            for (i, y) in real.iter_mut().enumerate() {
                let p = axis.index_to_ppm(i as f64);
                let d = (p - center_ppm) / 0.08;
                *y = 1_000.0 * (-d * d).exp();
            }
            Spectrum1D::new(
                name.to_string(),
                Experiment1DKind::Proton1D,
                axis,
                real,
                None,
            )
        };
        app.apply_drop_outcome(IngestDropOutcome::Spectra1D {
            source_name: "user_1d_batch".to_string(),
            spectra: vec![
                mk_1d("1D-A", 7.50),
                mk_1d("1D-B", 3.80),
                mk_1d("1D-C", 1.25),
            ],
        });
        assert_eq!(app.stack1d_panel.traces.len(), 3);
        assert_eq!(app.workbook.pages_2d.len(), 0);

        // Curate a custom peak on 1D Page #0, switch to Page #1, switch back -> Page #0 preserves peak
        let _ = app
            .stack1d_panel
            .add_manual_peak_at_or_near_ppm(7.50, false);
        let p0_count = app.stack1d_panel.picked_peaks.len();
        app.activate_1d_workbook_page(1);
        app.activate_1d_workbook_page(0);
        assert_eq!(app.stack1d_panel.picked_peaks.len(), p0_count);

        // 2. Append 2 user 2D spectra (Pages #4 & #5 in workbook) with curated peaks & zoomed views
        let usf4 = build_synthetic_ucsf_2d_test_bytes(
            "1H",
            32,
            16,
            500.0,
            5000.0,
            10.0,
            "1H",
            32,
            16,
            500.0,
            5000.0,
            10.0,
            &[(8, 20, 60.0), (20, 8, 60.0)],
        );
        let out4 = ingest_dropped_payloads(&[DroppedPayload::new("NOESY_user.ucsf", usf4)])
            .expect("NOESY_user.ucsf");
        app.apply_drop_outcome(out4);
        let p4_idx = app.add_manual_2d_peak(7.10, 2.90, None);
        app.peak_2d_labels.insert(
            peak_label_key_2d(app.peaks[p4_idx].h, app.peaks[p4_idx].n),
            "Page4-Peak".to_string(),
        );
        app.view_h_left = 8.8;
        app.view_h_right = 1.4;
        app.save_live_2d_into_workbook();

        let usf5 = build_synthetic_ucsf_2d_test_bytes(
            "1H",
            32,
            16,
            500.0,
            5000.0,
            10.0,
            "1H",
            32,
            16,
            500.0,
            5000.0,
            10.0,
            &[(6, 24, 55.0), (24, 6, 55.0)],
        );
        let out5 = ingest_dropped_payloads(&[DroppedPayload::new("TOCSY_user.ucsf", usf5)])
            .expect("TOCSY_user.ucsf");
        app.apply_drop_outcome(out5);
        let p5_idx = app.add_manual_2d_peak(6.40, 2.20, None);
        app.peak_2d_labels.insert(
            peak_label_key_2d(app.peaks[p5_idx].h, app.peaks[p5_idx].n),
            "Page5-Peak".to_string(),
        );
        app.view_h_left = 8.2;
        app.view_h_right = 0.9;
        app.save_live_2d_into_workbook();

        assert_eq!(app.stack1d_panel.traces.len(), 3);
        assert_eq!(app.workbook.pages_2d.len(), 2);

        // 3. Now drop a single 2D Bruker CON folder "3" — MUST append (6 pages total: 3 1D + 3 2D)!
        let nx = 48usize;
        let ny = 48usize;
        let procs = format!(
            "##TITLE= c_hcacon_ia\n##$SI= {nx}\n##$OFFSET= 182.0\n##$SW_p= 2414.4\n##$SF= 201.26\n##$XDIM= 48\n##$BYTORDP= 0\n##$NC_proc= 3\n##END=\n"
        );
        let proc2s = format!(
            "##TITLE= c_hcacon_ia\n##$SI= {ny}\n##$OFFSET= 142.0\n##$SW_p= 3244.0\n##$SF= 81.10\n##$XDIM= 48\n##$BYTORDP= 0\n##$NC_proc= 3\n##END=\n"
        );
        let mut rr = Vec::with_capacity(nx * ny * 4);
        for iy in 0..ny {
            for ix in 0..nx {
                let mut v = (((ix * 19 + iy * 11) % 17) as i32 - 8) * 1_500;
                let dx = ix as f64 - 22.0;
                let dy = iy as f64 - 24.0;
                if dx.abs() <= 3.0 && dy.abs() <= 3.0 {
                    v += (650_000.0 * (-0.45 * (dx * dx + dy * dy)).exp()) as i32;
                }
                rr.extend_from_slice(&v.to_le_bytes());
            }
        }
        let out_con = ingest_dropped_payloads(&[
            DroppedPayload::new("3/pdata/1/procs", procs.into_bytes()),
            DroppedPayload::new("3/pdata/1/proc2s", proc2s.into_bytes()),
            DroppedPayload::new("3/pdata/1/2rr", rr),
        ])
        .expect("single Bruker 2D CON folder 3");
        app.apply_drop_outcome(out_con);

        assert_eq!(
            app.stack1d_panel.traces.len(),
            3,
            "3 user 1D pages must remain open"
        );
        assert_eq!(
            app.workbook.pages_2d.len(),
            3,
            "2 existing user 2D pages + 1 newly dropped CON folder = 3 2D pages (6 total)"
        );
        assert_eq!(app.active_2d_nuclei(), ("13C", "15N"));
        assert!(
            app.overlay_layers.is_empty(),
            "incompatible 1H-1H pages must NOT overlay onto 13C-15N CON"
        );

        // 4. Switch back to Page #4 (index 0) and Page #5 (index 1) and verify full state restoration
        app.activate_2d_workbook_page(0);
        assert_eq!(app.primary_2d_label, "NOESY_user.ucsf");
        assert!((app.view_h_left - 8.8).abs() < 1e-4 && (app.view_h_right - 1.4).abs() < 1e-4);
        assert!(app
            .peaks
            .iter()
            .any(|p| (p.h - 7.10).abs() < 1e-3 && (p.n - 2.90).abs() < 1e-3));
        assert_eq!(
            app.peak_2d_labels
                .get(&peak_label_key_2d(7.10, 2.90))
                .map(String::as_str),
            Some("Page4-Peak")
        );

        app.activate_2d_workbook_page(1);
        assert_eq!(app.primary_2d_label, "TOCSY_user.ucsf");
        assert!((app.view_h_left - 8.2).abs() < 1e-4 && (app.view_h_right - 0.9).abs() < 1e-4);
        assert!(app
            .peaks
            .iter()
            .any(|p| (p.h - 6.40).abs() < 1e-3 && (p.n - 2.20).abs() < 1e-3));
        assert_eq!(
            app.peak_2d_labels
                .get(&peak_label_key_2d(6.40, 2.20))
                .map(String::as_str),
            Some("Page5-Peak")
        );
    }
}
