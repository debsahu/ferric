//! Deterministic animated module tour video & interactive JSON bundle exporter (`T18.1a`: `ferric export-tour`).
//!
//! Provenance: Original clean-room 2D anti-aliased RGB24 rasterizer, zero-dependency RFC 2083
//! PNG encoder, `ffmpeg` rawvideo pipe driver, and deterministic `nmr-core` showcase builders
//! (`CANONICAL_TOUR_SEED = 20261801`, `schema_version = "ferric-tour-v1"`):
//! - Bruker/Varian `pulseprogram` AST, DSP group delay, & `FISTA`/`IST` NUS reconstruction:
//!   Stern et al., *J. Magn. Reson.* **2007**, 188, 295–300; Hyberts et al., *J. Am. Chem. Soc.*
//!   **2012**, 134, 6504–6515; Beck & Teboulle, *SIAM J. Imaging Sci.* **2009**, 2, 183–202;
//!   Chen et al., *J. Magn. Reson.* **2002**, 158, 164–168.
//! - Backbone `1H-15N` HSQC, `3D HNCACB` strips, & thermodynamic/graph assignment: Kay et al.,
//!   *J. Magn. Reson.* **1990**, 89, 496–514; Wittekind & Mueller, *J. Magn. Reson. B* **1993**,
//!   101, 201–205; Schmidt & Güntert, *J. Am. Chem. Soc.* **2012**, 134, 12817–12829.
//! - 3D NOE restraints, Saupe `RDC` alignment tensor & `ByPeptidePlane` `Q_free`, Lipari–Szabo $S^2$,
//!   & Woessner rotational diffusion: Nilges, *J. Mol. Biol.* **1995**, 245, 645–660; Losonczi et
//!   al., *J. Magn. Reson.* **1999**, 138, 334–342; Cornilescu et al., *J. Am. Chem. Soc.* **1998**,
//!   120, 6836–6837; Lipari & Szabo, *J. Am. Chem. Soc.* **1982**, 104, 4546–4570; Kay, Torchia
//!   & Bax, *Biochemistry* **1989**, 28, 8972–8979; Woessner, *J. Chem. Phys.* **1962**, 37, 647–654.

use std::collections::BTreeMap;
use std::f64::consts::{PI, TAU};
use std::fs;
use std::io::{ErrorKind, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde::{Deserialize, Serialize};
use serde_json::json;

use nmr_core::bayes_sa::{solve_bayes_sa_from_peaks, BayesSaConfig};
use nmr_core::evo_graph::{assign_evo_graph_from_peaks, EvoGraphConfig};
use nmr_core::fid::Complex64;
use nmr_core::modelfree::{
    extract_nh_unit_vectors_from_chain, fit_modelfree_dataset, predict_r1_r2_noe,
    DiffusionTensorKind, LocalMfParams, ModelFreeConfig, ModelSelectionCriterion as MfCriterion,
    MultiFieldResidueInput,
};
use nmr_core::nmrstar::read_star_str;
use nmr_core::noesy::{compute_chain_torsions, parse_pdb_str_first_model};
use nmr_core::peaks::Peak;
use nmr_core::priors::Priors;
use nmr_core::process::{auto_phase_chen2002, fft_in_place, reconstruct_nus_1d_ist, NusIstConfig};
use nmr_core::pulprog::{interpret_bruker_experiment, BrukerExperimentInputs};
use nmr_core::qc::evaluate_dataset_qc;
use nmr_core::rdc::{evaluate_rdc_validation, parse_rdc_tsv_str, partition_rdc_free_set};
use nmr_core::relax_fit::{
    estimate_initial_axial_diffusion_from_r2_r1, estimate_trimmed_tau_m, fit_decay_curve_cpu,
    DecayCurveInput, ExpFitConfig, OffsetModelGate, RelaxPhysicsConfig, ResidueRelaxRates,
    TrimmedTauMConfig,
};
use nmr_core::rng::Rng;
use nmr_core::simulate::{
    render_hsqc_2d, render_triple_3d, simulate, SimConfig, SpectrumRenderConfig,
};
use nmr_core::solver::{assign, AnnealConfig, ScoreConfig};
use nmr_core::structure::parse_exp_noe_tsv_str;
use nmr_core::torsion_db::{predict_torsions_with_firewall, HomologFirewallQuery};

/// Shared tour JSON schema version (`ferric-tour-v1`).
pub const TOUR_SCHEMA_VERSION: &str = "ferric-tour-v1";
/// Canonical deterministic PRNG seed for `ferric export-tour`.
pub const CANONICAL_TOUR_SEED: u64 = 20261801;
/// Canonical tour duration in seconds (`4` chapters × `4.0 s`).
pub const TOUR_DURATION_SEC: f64 = 16.0;
/// Keyframe sample timestamps (`t = 2.0, 6.0, 10.0, 14.0 s`).
pub const KEYFRAME_TIMES_SEC: [f64; 4] = [2.0, 6.0, 10.0, 14.0];

const UBIQUITIN_STAR_FIXTURE: &str = include_str!("../../../data/fixtures/synthetic_ubiquitin.str");
const UBIQUITIN_1UBQ_PDB_FIXTURE: &str =
    include_str!("../../../data/fixtures/ubiquitin_1ubq_xray.pdb");
const UBIQUITIN_1D3Z_NOE_FIXTURE: &str =
    include_str!("../../../data/fixtures/ubiquitin_1d3z_exp_noe.tsv");
const UBIQUITIN_1D3Z_RDC_FIXTURE: &str =
    include_str!("../../../data/fixtures/ubiquitin_1d3z_rdc.tsv");

// ============================================================================
// Shared `TourBundle` Schema (`ferric-tour-v1`) & External Chem Hook
// ============================================================================

/// Scientific validation scope badge rendered in both video frames and web player UI.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidationBadge {
    pub kind: String,
    pub label: String,
    pub detail: String,
}

/// Single 4-second chapter specification inside a 16-second module tour.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TourChapter {
    pub index: u8,
    pub start_sec: f64,
    pub end_sec: f64,
    pub title: String,
    pub caption: String,
    pub metric_pill: String,
}

/// Interactive slider / inspector metadata consumed by the HTML5 tour player.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InteractiveControlSpec {
    pub id: String,
    pub label: String,
    pub min: f64,
    pub max: f64,
    pub step: f64,
    pub default_val: f64,
    pub unit: String,
}

/// Canonical deterministic JSON payload (`<slug>.tour.json`) for one tour module (`01..=08`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TourBundle {
    pub schema_version: String,
    pub module_index: u8,
    pub slug: String,
    pub title: String,
    pub subtitle: String,
    pub manual_chapter: String,
    pub validation_badge: ValidationBadge,
    pub duration_sec: f64,
    pub fps: u32,
    pub cli_command: String,
    pub chapters: Vec<TourChapter>,
    pub interactive_control: InteractiveControlSpec,
    pub payload: serde_json::Value,
}

/// Pluggable bundle builder hook for `export_tour_chem` modules (`05..=08`).
pub type ChemBundleFn = fn(u8, u64) -> Result<serde_json::Value, String>;
/// Pluggable RGB24 frame renderer hook for `export_tour_chem` modules (`05..=08`).
pub type ChemFrameFn = fn(u8, f64, usize, usize) -> Vec<u8>;

/// Selector for `ferric export-tour --module ...`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TourModuleId {
    M01FidPulprogNus,
    M02HsqcStripsSolvers,
    M03StructureNoeRdc,
    M04RelaxModelfree,
    M05ExchangeCestCpmgCsp,
    M06Nmr1dMultipletQnmrOctet,
    M07DosyDiffusion,
    M08MetabolomicsMcp,
}

impl TourModuleId {
    pub const CORE_MODULES: [Self; 4] = [
        Self::M01FidPulprogNus,
        Self::M02HsqcStripsSolvers,
        Self::M03StructureNoeRdc,
        Self::M04RelaxModelfree,
    ];

    pub const ALL: [Self; 8] = [
        Self::M01FidPulprogNus,
        Self::M02HsqcStripsSolvers,
        Self::M03StructureNoeRdc,
        Self::M04RelaxModelfree,
        Self::M05ExchangeCestCpmgCsp,
        Self::M06Nmr1dMultipletQnmrOctet,
        Self::M07DosyDiffusion,
        Self::M08MetabolomicsMcp,
    ];

    pub const fn index(self) -> u8 {
        match self {
            Self::M01FidPulprogNus => 1,
            Self::M02HsqcStripsSolvers => 2,
            Self::M03StructureNoeRdc => 3,
            Self::M04RelaxModelfree => 4,
            Self::M05ExchangeCestCpmgCsp => 5,
            Self::M06Nmr1dMultipletQnmrOctet => 6,
            Self::M07DosyDiffusion => 7,
            Self::M08MetabolomicsMcp => 8,
        }
    }

    pub const fn slug(self) -> &'static str {
        match self {
            Self::M01FidPulprogNus => "01_fid_pulprog_nus",
            Self::M02HsqcStripsSolvers => "02_hsqc_strips_solvers",
            Self::M03StructureNoeRdc => "03_structure_noe_rdc",
            Self::M04RelaxModelfree => "04_relax_modelfree",
            Self::M05ExchangeCestCpmgCsp => "05_exchange_cest_cpmg_csp",
            Self::M06Nmr1dMultipletQnmrOctet => "06_nmr1d_multiplet_qnmr_octet",
            Self::M07DosyDiffusion => "07_dosy_diffusion",
            Self::M08MetabolomicsMcp => "08_metabolomics_mcp",
        }
    }

    pub fn parse_single(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "01" | "1" | "01_fid_pulprog_nus" | "fid" | "pulprog" | "nus" => {
                Ok(Self::M01FidPulprogNus)
            }
            "02" | "2" | "02_hsqc_strips_solvers" | "hsqc" | "strips" | "solvers" => {
                Ok(Self::M02HsqcStripsSolvers)
            }
            "03" | "3" | "03_structure_noe_rdc" | "structure" | "noe" | "rdc" => {
                Ok(Self::M03StructureNoeRdc)
            }
            "04" | "4" | "04_relax_modelfree" | "relax" | "modelfree" => {
                Ok(Self::M04RelaxModelfree)
            }
            "05" | "5" | "05_exchange_cest_cpmg_csp" | "exchange" | "cest" | "cpmg" | "csp" => {
                Ok(Self::M05ExchangeCestCpmgCsp)
            }
            "06"
            | "6"
            | "06_nmr1d_multiplet_qnmr_octet"
            | "nmr1d"
            | "multiplet"
            | "qnmr"
            | "octet" => Ok(Self::M06Nmr1dMultipletQnmrOctet),
            "07" | "7" | "07_dosy_diffusion" | "dosy" | "diffusion" => Ok(Self::M07DosyDiffusion),
            "08" | "8" | "08_metabolomics_mcp" | "metabolomics" | "mcp" => {
                Ok(Self::M08MetabolomicsMcp)
            }
            other => Err(format!(
                "unknown --module {other:?}; expected all|01..08|<slug>"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ModuleSelection {
    All,
    Single(TourModuleId),
}

impl ModuleSelection {
    fn parse(raw: &str) -> Result<Self, String> {
        if raw.trim().eq_ignore_ascii_case("all") {
            Ok(Self::All)
        } else {
            Ok(Self::Single(TourModuleId::parse_single(raw)?))
        }
    }
}

// ============================================================================
// CLI Entrypoint & Usage
// ============================================================================

/// Usage banner for `ferric export-tour`.
pub fn export_tour_usage() -> &'static str {
    "Usage:\n\
       ferric export-tour --out-dir <DIR> [--module all|01|02|03|04|05|06|07|08|<slug>] [--render-videos] [--keyframes-only] [--width 1280] [--height 720] [--fps 30]\n\
     \n\
     Deterministic animated module tour JSON, PNG keyframe, and MP4/WebM video exporter (seed = 20261801):\n\
       01_fid_pulprog_nus             Bruker pulprog AST, FID apodization/FFT, & 25% NUS FISTA\n\
       02_hsqc_strips_solvers         Ubiquitin HSQC, 3D HNCACB strips, Cv(T), & QC ribbon\n\
       03_structure_noe_rdc           1UBQ 3D trace, 1D3Z NOEs, Saupe RDC Q_free, & S^2 orbit\n\
       04_relax_modelfree             Locked-footprint SNR, dual-field R1/R2/hetNOE, tau_m, & M1-M5 S^2\n\
       05_exchange_cest_cpmg_csp      CEST, multi-field CPMG dispersion, & quadratic Kd (chem)\n\
       06_nmr1d_multiplet_qnmr_octet  1D HSVD/Whittaker, Hoye multiplet, qNMR, & Octet (chem)\n\
       07_dosy_diffusion              Pseudo-2D BPP-LED DOSY, Stejskal-Tanner, & Rh/MW (chem)\n\
       08_metabolomics_mcp            AI-binning/PQN, PCA/OPLS-DA, STOCSY, & MCP copilot (chem)"
}

#[derive(Clone, Debug)]
struct ExportTourOptions {
    out_dir: PathBuf,
    selection: ModuleSelection,
    render_videos: bool,
    keyframes_only: bool,
    width: usize,
    height: usize,
    fps: u32,
}

fn parse_export_tour_args(args: &[String]) -> Result<Option<ExportTourOptions>, String> {
    let mut out_dir: Option<PathBuf> = None;
    let mut selection = ModuleSelection::All;
    let mut render_videos = false;
    let mut keyframes_only = false;
    let mut width: usize = 1280;
    let mut height: usize = 720;
    let mut fps: u32 = 30;
    let mut i = 0usize;

    while i < args.len() {
        let arg = args[i].as_str();
        match arg {
            "-h" | "--help" | "help" => {
                println!("{}", export_tour_usage());
                return Ok(None);
            }
            "--out-dir" | "-o" => {
                i += 1;
                let Some(val) = args.get(i) else {
                    return Err("missing directory argument after --out-dir".to_string());
                };
                if val.trim().is_empty() {
                    return Err("--out-dir cannot be empty".to_string());
                }
                out_dir = Some(PathBuf::from(val));
            }
            "--module" => {
                i += 1;
                let Some(val) = args.get(i) else {
                    return Err("missing value after --module".to_string());
                };
                selection = ModuleSelection::parse(val)?;
            }
            "--render-videos" => render_videos = true,
            "--keyframes-only" => keyframes_only = true,
            "--width" => {
                i += 1;
                let Some(val) = args.get(i) else {
                    return Err("missing integer after --width".to_string());
                };
                width = val
                    .parse::<usize>()
                    .map_err(|_| format!("invalid --width {val:?}"))?;
            }
            "--height" => {
                i += 1;
                let Some(val) = args.get(i) else {
                    return Err("missing integer after --height".to_string());
                };
                height = val
                    .parse::<usize>()
                    .map_err(|_| format!("invalid --height {val:?}"))?;
            }
            "--fps" => {
                i += 1;
                let Some(val) = args.get(i) else {
                    return Err("missing integer after --fps".to_string());
                };
                fps = val
                    .parse::<u32>()
                    .map_err(|_| format!("invalid --fps {val:?}"))?;
            }
            other => {
                if let Some(rest) = other.strip_prefix("--out-dir=") {
                    if rest.trim().is_empty() {
                        return Err("--out-dir cannot be empty".to_string());
                    }
                    out_dir = Some(PathBuf::from(rest));
                } else if let Some(rest) = other.strip_prefix("--module=") {
                    selection = ModuleSelection::parse(rest)?;
                } else if let Some(rest) = other.strip_prefix("--width=") {
                    width = rest
                        .parse::<usize>()
                        .map_err(|_| format!("invalid --width {rest:?}"))?;
                } else if let Some(rest) = other.strip_prefix("--height=") {
                    height = rest
                        .parse::<usize>()
                        .map_err(|_| format!("invalid --height {rest:?}"))?;
                } else if let Some(rest) = other.strip_prefix("--fps=") {
                    fps = rest
                        .parse::<u32>()
                        .map_err(|_| format!("invalid --fps {rest:?}"))?;
                } else {
                    return Err(format!(
                        "unknown export-tour argument {other:?}\n\n{}",
                        export_tour_usage()
                    ));
                }
            }
        }
        i += 1;
    }

    let target_dir = out_dir.ok_or_else(|| {
        format!(
            "missing required --out-dir <dir>\n\n{}",
            export_tour_usage()
        )
    })?;
    if !(320..=3840).contains(&width) || !width.is_multiple_of(2) {
        return Err(format!(
            "invalid --width {width}: must be an even integer in 320..=3840 for yuv420p"
        ));
    }
    if !(240..=2160).contains(&height) || !height.is_multiple_of(2) {
        return Err(format!(
            "invalid --height {height}: must be an even integer in 240..=2160 for yuv420p"
        ));
    }
    if fps == 0 || fps > 120 {
        return Err(format!("invalid --fps {fps}: must be in 1..=120"));
    }

    Ok(Some(ExportTourOptions {
        out_dir: target_dir,
        selection,
        render_videos,
        keyframes_only,
        width,
        height,
        fps,
    }))
}

/// Run `ferric export-tour` for core modules (`01..=04`).
pub fn run_export_tour_cli(args: &[String]) -> Result<(), String> {
    run_export_tour_cli_with_chem(args, None, None)
}

/// Run `ferric export-tour` with optional `export_tour_chem` module callbacks (`05..=08`).
pub fn run_export_tour_cli_with_chem(
    args: &[String],
    chem_bundle: Option<ChemBundleFn>,
    chem_frame: Option<ChemFrameFn>,
) -> Result<(), String> {
    let Some(opts) = parse_export_tour_args(args)? else {
        return Ok(());
    };

    if opts.out_dir.as_os_str().is_empty() {
        return Err("output directory path cannot be empty".to_string());
    }
    if opts.out_dir.exists() && !opts.out_dir.is_dir() {
        return Err(format!(
            "output path is an existing non-directory file: {}",
            opts.out_dir.display()
        ));
    }

    let data_dir = opts.out_dir.join("data");
    fs::create_dir_all(&data_dir)
        .map_err(|e| format!("failed to create {}: {e}", data_dir.display()))?;
    if opts.keyframes_only {
        let kf_dir = opts.out_dir.join("keyframes");
        fs::create_dir_all(&kf_dir)
            .map_err(|e| format!("failed to create {}: {e}", kf_dir.display()))?;
    }
    if opts.render_videos {
        let vid_dir = opts.out_dir.join("videos");
        fs::create_dir_all(&vid_dir)
            .map_err(|e| format!("failed to create {}: {e}", vid_dir.display()))?;
    }

    let modules: Vec<TourModuleId> = match opts.selection {
        ModuleSelection::All => {
            if chem_bundle.is_some() && chem_frame.is_some() {
                TourModuleId::ALL.to_vec()
            } else {
                TourModuleId::CORE_MODULES.to_vec()
            }
        }
        ModuleSelection::Single(m) => vec![m],
    };

    for module_id in modules {
        let idx = module_id.index();
        let slug = module_id.slug();

        enum ModuleDriver {
            Core(Box<CoreTourState>),
            Chem {
                bundle: Box<TourBundle>,
                render_fn: ChemFrameFn,
            },
        }

        let driver = if idx <= 4 {
            let state = build_core_tour_state(module_id, CANONICAL_TOUR_SEED)?;
            let mut b = state.bundle.clone();
            b.fps = opts.fps;
            let json_path = data_dir.join(format!("{slug}.tour.json"));
            write_tour_bundle_json(&json_path, &b)?;
            ModuleDriver::Core(Box::new(state))
        } else {
            let Some(build_fn) = chem_bundle else {
                return Err(format!(
                    "module {slug} requires export_tour_chem integration"
                ));
            };
            let Some(render_fn) = chem_frame else {
                return Err(format!(
                    "module {slug} requires export_tour_chem frame renderer integration"
                ));
            };
            let val = build_fn(idx, CANONICAL_TOUR_SEED)?;
            let mut bundle: TourBundle = serde_json::from_value(val)
                .map_err(|e| format!("invalid TourBundle JSON from chem module {idx}: {e}"))?;
            if bundle.schema_version != TOUR_SCHEMA_VERSION {
                return Err(format!(
                    "chem module {idx} schema_version mismatch: got {:?}, expected {:?}",
                    bundle.schema_version, TOUR_SCHEMA_VERSION
                ));
            }
            bundle.fps = opts.fps;
            let json_path = data_dir.join(format!("{slug}.tour.json"));
            write_tour_bundle_json(&json_path, &bundle)?;
            ModuleDriver::Chem {
                bundle: Box::new(bundle),
                render_fn,
            }
        };

        let render_at = |t_sec: f64| -> Vec<u8> {
            match &driver {
                ModuleDriver::Core(state) => {
                    render_core_tour_frame(state, t_sec, opts.width, opts.height)
                }
                ModuleDriver::Chem { render_fn, .. } => {
                    render_fn(idx, t_sec, opts.width, opts.height)
                }
            }
        };

        if opts.keyframes_only {
            let mut chapter_frames: Vec<Vec<u8>> = Vec::with_capacity(4);
            for (ch_idx, &t_sec) in KEYFRAME_TIMES_SEC.iter().enumerate() {
                let rgb = render_at(t_sec);
                let expected_bytes = opts.width * opts.height * 3;
                if rgb.len() != expected_bytes {
                    return Err(format!(
                        "{slug} ch{} frame length mismatch: got {}, expected {}",
                        ch_idx + 1,
                        rgb.len(),
                        expected_bytes
                    ));
                }
                let png = encode_rgb_png(opts.width as u32, opts.height as u32, &rgb)?;
                let png_path = opts
                    .out_dir
                    .join("keyframes")
                    .join(format!("{slug}_ch{}.png", ch_idx + 1));
                fs::write(&png_path, &png)
                    .map_err(|e| format!("failed to write {}: {e}", png_path.display()))?;
                chapter_frames.push(rgb);
            }
            verify_keyframe_variance(slug, &chapter_frames)?;
        }

        if opts.render_videos {
            let duration = match &driver {
                ModuleDriver::Core(s) => s.bundle.duration_sec,
                ModuleDriver::Chem { bundle, .. } => bundle.duration_sec,
            };
            let total_frames = ((duration * f64::from(opts.fps)).round() as usize).max(1);
            let mp4_path = opts.out_dir.join("videos").join(format!("{slug}.mp4"));
            let webm_path = opts.out_dir.join("videos").join(format!("{slug}.webm"));
            encode_video_pair_parallel(
                opts.width,
                opts.height,
                opts.fps,
                total_frames,
                &mp4_path,
                &webm_path,
                &render_at,
            )?;
        }
    }

    Ok(())
}

fn write_tour_bundle_json(path: &Path, bundle: &TourBundle) -> Result<(), String> {
    let serialized = serde_json::to_string_pretty(bundle)
        .map_err(|e| format!("failed to serialize tour JSON {}: {e}", path.display()))?;
    for bad in ["NaN", "Infinity", "-Infinity"] {
        if serialized.contains(bad) {
            return Err(format!(
                "non-finite JSON token {bad:?} detected in {}",
                path.display()
            ));
        }
    }
    fs::write(path, format!("{serialized}\n"))
        .map_err(|e| format!("failed to write {}: {e}", path.display()))?;
    Ok(())
}

fn verify_keyframe_variance(slug: &str, frames: &[Vec<u8>]) -> Result<(), String> {
    for (idx, frame) in frames.iter().enumerate() {
        let mut min_v = 255u8;
        let mut max_v = 0u8;
        for &b in frame.iter().step_by(37) {
            min_v = min_v.min(b);
            max_v = max_v.max(b);
        }
        if max_v.saturating_sub(min_v) < 24 {
            return Err(format!(
                "{slug} keyframe ch{} appears blank/flat (contrast range {})",
                idx + 1,
                max_v.saturating_sub(min_v)
            ));
        }
    }
    for pair in frames.windows(2).enumerate() {
        let (idx, win) = pair;
        let a = &win[0];
        let b = &win[1];
        let mut sum_diff: u64 = 0;
        let mut count: u64 = 0;
        for (pa, pb) in a.iter().zip(b.iter()).step_by(19) {
            sum_diff += u64::from(pa.abs_diff(*pb));
            count += 1;
        }
        let mad = if count > 0 {
            (sum_diff as f64) / (count as f64)
        } else {
            0.0
        };
        if mad < 1.5 {
            return Err(format!(
                "{slug} keyframes ch{} -> ch{} show insufficient visual change (MAD={mad:.2})",
                idx + 1,
                idx + 2
            ));
        }
    }
    Ok(())
}

// ============================================================================
// Zero-Dependency RFC 2083 RGB24 PNG Writer & `ffmpeg` Pipe Encoder
// ============================================================================

fn png_crc32(chunk_type: &[u8; 4], data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &b in chunk_type.iter().chain(data.iter()) {
        crc ^= u32::from(b);
        for _ in 0..8 {
            if (crc & 1) != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

fn adler32_checksum(data: &[u8]) -> u32 {
    const MOD_ADLER: u32 = 65_521;
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    for chunk in data.chunks(5552) {
        for &byte in chunk {
            a += u32::from(byte);
            b += a;
        }
        a %= MOD_ADLER;
        b %= MOD_ADLER;
    }
    (b << 16) | a
}

fn append_png_chunk(out: &mut Vec<u8>, chunk_type: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(chunk_type);
    out.extend_from_slice(data);
    out.extend_from_slice(&png_crc32(chunk_type, data).to_be_bytes());
}

/// Encode an 8-bit RGB24 framebuffer (`width * height * 3` bytes) as an RFC 2083 PNG stream
/// using stored uncompressed RFC 1950 ZLIB blocks (zero external dependencies).
pub fn encode_rgb_png(width: u32, height: u32, rgb: &[u8]) -> Result<Vec<u8>, String> {
    if width == 0 || height == 0 {
        return Err("PNG width and height must both be >= 1".to_string());
    }
    let row_bytes = (width as usize)
        .checked_mul(3)
        .ok_or_else(|| "PNG row byte length overflow".to_string())?;
    let expected_len = row_bytes
        .checked_mul(height as usize)
        .ok_or_else(|| "PNG buffer byte length overflow".to_string())?;
    if rgb.len() != expected_len {
        return Err(format!(
            "RGB buffer length mismatch: got {} bytes, expected {} ({width}x{height}x3)",
            rgb.len(),
            expected_len
        ));
    }

    let raw_scanline_len = (row_bytes + 1)
        .checked_mul(height as usize)
        .ok_or_else(|| "PNG scanline length overflow".to_string())?;
    let mut raw_scanlines = Vec::with_capacity(raw_scanline_len);
    for row in rgb.chunks_exact(row_bytes) {
        raw_scanlines.push(0u8); // Filter type 0 (None)
        raw_scanlines.extend_from_slice(row);
    }

    let n_blocks = (raw_scanlines.len() / 65_535) + 1;
    let mut zlib = Vec::with_capacity(raw_scanlines.len() + n_blocks * 5 + 6);
    zlib.extend_from_slice(&[0x78, 0x01]);
    let chunks: Vec<&[u8]> = raw_scanlines.chunks(65_535).collect();
    for (idx, block) in chunks.iter().enumerate() {
        let is_final = idx + 1 == chunks.len();
        zlib.push(u8::from(is_final));
        let len16 = block.len() as u16;
        let nlen16 = !len16;
        zlib.extend_from_slice(&len16.to_le_bytes());
        zlib.extend_from_slice(&nlen16.to_le_bytes());
        zlib.extend_from_slice(block);
    }
    zlib.extend_from_slice(&adler32_checksum(&raw_scanlines).to_be_bytes());

    let mut png = Vec::with_capacity(zlib.len() + 64);
    png.extend_from_slice(b"\x89PNG\r\n\x1a\n");

    let mut ihdr = [0u8; 13];
    ihdr[0..4].copy_from_slice(&width.to_be_bytes());
    ihdr[4..8].copy_from_slice(&height.to_be_bytes());
    ihdr[8] = 8; // bit depth
    ihdr[9] = 2; // color type 2 = truecolor RGB
    ihdr[10] = 0;
    ihdr[11] = 0;
    ihdr[12] = 0;
    append_png_chunk(&mut png, b"IHDR", &ihdr);
    append_png_chunk(&mut png, b"IDAT", &zlib);
    append_png_chunk(&mut png, b"IEND", &[]);
    Ok(png)
}

fn spawn_ffmpeg_pipe(args: &[String], label: &str) -> Result<std::process::Child, String> {
    Command::new("ffmpeg")
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| {
            if e.kind() == ErrorKind::NotFound {
                format!("ffmpeg executable not found on PATH while starting {label}; install ffmpeg or pass --keyframes-only")
            } else {
                format!("failed to spawn ffmpeg for {label}: {e}")
            }
        })
}

fn finish_ffmpeg_child(
    mut child: std::process::Child,
    stderr_handle: Option<std::thread::ScopedJoinHandle<'_, String>>,
    label: &str,
) -> Result<(), String> {
    let status = child
        .wait()
        .map_err(|e| format!("failed waiting for ffmpeg ({label}): {e}"))?;
    let stderr_txt = stderr_handle
        .and_then(|h| h.join().ok())
        .unwrap_or_default();
    if !status.success() {
        let tail: String = stderr_txt
            .lines()
            .rev()
            .take(8)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join("\n");
        return Err(format!(
            "ffmpeg exited with status {:?} for {label}:\n{tail}",
            status.code()
        ));
    }
    Ok(())
}

fn encode_video_pair_parallel<F>(
    width: usize,
    height: usize,
    fps: u32,
    total_frames: usize,
    mp4_path: &Path,
    webm_path: &Path,
    render_at: &F,
) -> Result<(), String>
where
    F: Fn(f64) -> Vec<u8> + Sync,
{
    let size_arg = format!("{width}x{height}");
    let fps_arg = fps.to_string();
    let mp4_str = mp4_path.to_string_lossy().into_owned();
    let webm_str = webm_path.to_string_lossy().into_owned();

    let mp4_args = vec![
        "-y".to_string(),
        "-f".to_string(),
        "rawvideo".to_string(),
        "-pix_fmt".to_string(),
        "rgb24".to_string(),
        "-s".to_string(),
        size_arg.clone(),
        "-r".to_string(),
        fps_arg.clone(),
        "-i".to_string(),
        "-".to_string(),
        "-an".to_string(),
        "-c:v".to_string(),
        "libx264".to_string(),
        "-preset".to_string(),
        "fast".to_string(),
        "-crf".to_string(),
        "23".to_string(),
        "-pix_fmt".to_string(),
        "yuv420p".to_string(),
        "-movflags".to_string(),
        "+faststart".to_string(),
        mp4_str,
    ];
    let webm_args = vec![
        "-y".to_string(),
        "-f".to_string(),
        "rawvideo".to_string(),
        "-pix_fmt".to_string(),
        "rgb24".to_string(),
        "-s".to_string(),
        size_arg,
        "-r".to_string(),
        fps_arg,
        "-i".to_string(),
        "-".to_string(),
        "-an".to_string(),
        "-c:v".to_string(),
        "libvpx-vp9".to_string(),
        "-b:v".to_string(),
        "0".to_string(),
        "-crf".to_string(),
        "30".to_string(),
        "-pix_fmt".to_string(),
        "yuv420p".to_string(),
        "-row-mt".to_string(),
        "1".to_string(),
        webm_str,
    ];

    let mut mp4_child = spawn_ffmpeg_pipe(&mp4_args, "MP4 (libx264)")?;
    let mut webm_child = match spawn_ffmpeg_pipe(&webm_args, "WebM (libvpx-vp9)") {
        Ok(c) => c,
        Err(e) => {
            let _ = mp4_child.kill();
            let _ = mp4_child.wait();
            return Err(e);
        }
    };

    let mut mp4_stdin = mp4_child
        .stdin
        .take()
        .ok_or_else(|| "failed to open stdin pipe to MP4 ffmpeg".to_string())?;
    let mut webm_stdin = webm_child
        .stdin
        .take()
        .ok_or_else(|| "failed to open stdin pipe to WebM ffmpeg".to_string())?;
    let mp4_stderr = mp4_child.stderr.take();
    let webm_stderr = webm_child.stderr.take();

    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .clamp(1, 16);
    let expected_bytes = width * height * 3;
    let batch_size = 30usize;

    let pipe_res: Result<(), String> = std::thread::scope(|s| {
        let mp4_err_h = mp4_stderr.map(|mut r| {
            s.spawn(move || {
                let mut buf = String::new();
                let _ = r.read_to_string(&mut buf);
                buf
            })
        });
        let webm_err_h = webm_stderr.map(|mut r| {
            s.spawn(move || {
                let mut buf = String::new();
                let _ = r.read_to_string(&mut buf);
                buf
            })
        });

        let mut start = 0usize;
        while start < total_frames {
            let end = (start + batch_size).min(total_frames);
            let count = end - start;
            let mut batch: Vec<Vec<u8>> = vec![Vec::new(); count];

            let workers = n_threads.min(count);
            let chunk_per_worker = count.div_ceil(workers);
            let batch_start = start;
            std::thread::scope(|inner| {
                for (w_idx, slice) in batch.chunks_mut(chunk_per_worker).enumerate() {
                    let base_frame = batch_start + w_idx * chunk_per_worker;
                    inner.spawn(move || {
                        for (offset, slot) in slice.iter_mut().enumerate() {
                            let frame_idx = base_frame + offset;
                            let t_sec = (frame_idx as f64) / f64::from(fps);
                            *slot = render_at(t_sec);
                        }
                    });
                }
            });

            for (offset, frame) in batch.iter().enumerate() {
                if frame.len() != expected_bytes {
                    return Err(format!(
                        "rendered frame {} has {} bytes, expected {}",
                        start + offset,
                        frame.len(),
                        expected_bytes
                    ));
                }
            }

            let batch_ref = &batch;
            let mp4_w = &mut mp4_stdin;
            let webm_w = &mut webm_stdin;
            let (r_mp4, r_webm) = std::thread::scope(|ws| {
                let h1 = ws.spawn(move || -> Result<(), String> {
                    for frame in batch_ref {
                        mp4_w
                            .write_all(frame)
                            .map_err(|e| format!("failed writing raw frame to MP4 ffmpeg: {e}"))?;
                    }
                    Ok(())
                });
                let h2 = ws.spawn(move || -> Result<(), String> {
                    for frame in batch_ref {
                        webm_w
                            .write_all(frame)
                            .map_err(|e| format!("failed writing raw frame to WebM ffmpeg: {e}"))?;
                    }
                    Ok(())
                });
                let a = h1
                    .join()
                    .unwrap_or_else(|_| Err("MP4 writer thread panicked".to_string()));
                let b = h2
                    .join()
                    .unwrap_or_else(|_| Err("WebM writer thread panicked".to_string()));
                (a, b)
            });

            r_mp4?;
            r_webm?;
            start = end;
        }

        drop(mp4_stdin);
        drop(webm_stdin);

        finish_ffmpeg_child(mp4_child, mp4_err_h, "MP4 (libx264)")?;
        finish_ffmpeg_child(webm_child, webm_err_h, "WebM (libvpx-vp9)")?;
        Ok(())
    });

    pipe_res
}

// ============================================================================
// Shared Anti-Aliased `1280x720` `FrameCanvas` Rasterizer
// ============================================================================

/// Anti-aliased RGB24 framebuffer and 2D drawing primitives shared across all tour modules.
#[derive(Clone, Debug)]
pub struct FrameCanvas {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
}

impl FrameCanvas {
    /// Allocate a zeroed `width × height` RGB24 canvas.
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            rgb: vec![0u8; width.saturating_mul(height).saturating_mul(3)],
        }
    }

    #[inline]
    fn blend_pixel_cov(&mut self, x: usize, y: usize, rgba: [u8; 4], cov: f64) {
        if x >= self.width || y >= self.height {
            return;
        }
        let a = ((f64::from(rgba[3]) * cov.clamp(0.0, 1.0)).round() as u32).min(255);
        if a == 0 {
            return;
        }
        let idx = (y * self.width + x) * 3;
        if a == 255 {
            self.rgb[idx] = rgba[0];
            self.rgb[idx + 1] = rgba[1];
            self.rgb[idx + 2] = rgba[2];
        } else {
            let inv = 255 - a;
            self.rgb[idx] =
                ((u32::from(rgba[0]) * a + u32::from(self.rgb[idx]) * inv + 127) / 255) as u8;
            self.rgb[idx + 1] =
                ((u32::from(rgba[1]) * a + u32::from(self.rgb[idx + 1]) * inv + 127) / 255) as u8;
            self.rgb[idx + 2] =
                ((u32::from(rgba[2]) * a + u32::from(self.rgb[idx + 2]) * inv + 127) / 255) as u8;
        }
    }

    /// Fill the entire canvas with a smooth vertical gradient from `top_rgb` to `bot_rgb`.
    pub fn clear_gradient(&mut self, top_rgb: [u8; 3], bot_rgb: [u8; 3]) {
        if self.width == 0 || self.height == 0 {
            return;
        }
        let h_denom = (self.height.saturating_sub(1)).max(1) as f64;
        let row_bytes = self.width * 3;
        let mut row_pat = vec![0u8; row_bytes];
        for (y, row) in self.rgb.chunks_exact_mut(row_bytes).enumerate() {
            let t = (y as f64) / h_denom;
            let r = (f64::from(top_rgb[0]) * (1.0 - t) + f64::from(bot_rgb[0]) * t).round() as u8;
            let g = (f64::from(top_rgb[1]) * (1.0 - t) + f64::from(bot_rgb[1]) * t).round() as u8;
            let b = (f64::from(top_rgb[2]) * (1.0 - t) + f64::from(bot_rgb[2]) * t).round() as u8;
            for px in row_pat.chunks_exact_mut(3) {
                px[0] = r;
                px[1] = g;
                px[2] = b;
            }
            row.copy_from_slice(&row_pat);
        }
    }

    /// Anti-aliased filled rounded rectangle using analytical signed-distance coverage.
    pub fn fill_rounded_rect(
        &mut self,
        x: f64,
        y: f64,
        w: f64,
        h: f64,
        radius: f64,
        rgba: [u8; 4],
    ) {
        if w <= 0.0 || h <= 0.0 || rgba[3] == 0 {
            return;
        }
        let r = radius.clamp(0.0, 0.5 * w.min(h));
        let x0 = (x - 1.0).floor().max(0.0) as usize;
        let y0 = (y - 1.0).floor().max(0.0) as usize;
        let x1 = ((x + w + 1.0).ceil() as isize).clamp(0, self.width as isize) as usize;
        let y1 = ((y + h + 1.0).ceil() as isize).clamp(0, self.height as isize) as usize;

        let cx = x + 0.5 * w;
        let cy = y + 0.5 * h;
        let hx = 0.5 * w - r;
        let hy = 0.5 * h - r;

        let inner_x0 = (x + r + 1.0).ceil().max(x0 as f64) as usize;
        let inner_x1 = ((x + w - r - 1.0).floor() as isize).clamp(0, x1 as isize) as usize;
        let inner_y0 = (y + 1.0).ceil().max(y0 as f64) as usize;
        let inner_y1 = ((y + h - 1.0).floor() as isize).clamp(0, y1 as isize) as usize;

        for py in y0..y1 {
            let dy = (py as f64 + 0.5 - cy).abs() - hy;
            let in_fast_y = py >= inner_y0 && py < inner_y1;
            let mut px = x0;
            while px < x1 {
                if in_fast_y && px >= inner_x0 && px < inner_x1 {
                    let span_end = inner_x1.min(x1);
                    if rgba[3] == 255 {
                        for fx in px..span_end {
                            let idx = (py * self.width + fx) * 3;
                            self.rgb[idx] = rgba[0];
                            self.rgb[idx + 1] = rgba[1];
                            self.rgb[idx + 2] = rgba[2];
                        }
                    } else {
                        for fx in px..span_end {
                            self.blend_pixel_cov(fx, py, rgba, 1.0);
                        }
                    }
                    px = span_end;
                    continue;
                }
                let dx = (px as f64 + 0.5 - cx).abs() - hx;
                let ox = dx.max(0.0);
                let oy = dy.max(0.0);
                let sdf = (ox * ox + oy * oy).sqrt() + dx.max(oy).min(0.0) - r;
                if sdf < 0.75 {
                    let cov = (0.75 - sdf) / 1.5;
                    self.blend_pixel_cov(px, py, rgba, cov);
                }
                px += 1;
            }
        }
    }

    /// Anti-aliased stroked rounded rectangle outline.
    #[allow(clippy::too_many_arguments)]
    pub fn stroke_rounded_rect(
        &mut self,
        x: f64,
        y: f64,
        w: f64,
        h: f64,
        radius: f64,
        thickness: f64,
        rgba: [u8; 4],
    ) {
        if w <= 0.0 || h <= 0.0 || thickness <= 0.0 || rgba[3] == 0 {
            return;
        }
        let r = radius.clamp(0.0, 0.5 * w.min(h));
        let half_t = 0.5 * thickness;
        let pad = half_t + 1.5;
        let x0 = (x - pad).floor().max(0.0) as usize;
        let y0 = (y - pad).floor().max(0.0) as usize;
        let x1 = ((x + w + pad).ceil() as isize).clamp(0, self.width as isize) as usize;
        let y1 = ((y + h + pad).ceil() as isize).clamp(0, self.height as isize) as usize;

        let cx = x + 0.5 * w;
        let cy = y + 0.5 * h;
        let hx = 0.5 * w - r;
        let hy = 0.5 * h - r;

        let hole_x0 = (x + r + pad).ceil().max(x0 as f64) as usize;
        let hole_x1 = ((x + w - r - pad).floor() as isize).clamp(0, x1 as isize) as usize;
        let hole_y0 = (y + pad).ceil().max(y0 as f64) as usize;
        let hole_y1 = ((y + h - pad).floor() as isize).clamp(0, y1 as isize) as usize;

        for py in y0..y1 {
            let dy = (py as f64 + 0.5 - cy).abs() - hy;
            let skip_y = py >= hole_y0 && py < hole_y1 && hole_x0 < hole_x1;
            let mut px = x0;
            while px < x1 {
                if skip_y && px == hole_x0 {
                    px = hole_x1;
                    continue;
                }
                let dx = (px as f64 + 0.5 - cx).abs() - hx;
                let ox = dx.max(0.0);
                let oy = dy.max(0.0);
                let sdf = (ox * ox + oy * oy).sqrt() + dx.max(oy).min(0.0) - r;
                let dist = sdf.abs() - half_t;
                if dist < 0.75 {
                    let cov = (0.75 - dist) / 1.5;
                    self.blend_pixel_cov(px, py, rgba, cov);
                }
                px += 1;
            }
        }
    }

    /// Anti-aliased line segment (capsule signed-distance coverage).
    pub fn draw_line(&mut self, x0: f64, y0: f64, x1: f64, y1: f64, thickness: f64, rgba: [u8; 4]) {
        if thickness <= 0.0 || rgba[3] == 0 {
            return;
        }
        let half_t = 0.5 * thickness.max(0.75);
        let pad = half_t + 1.5;
        let min_x = (x0.min(x1) - pad).floor().max(0.0) as usize;
        let min_y = (y0.min(y1) - pad).floor().max(0.0) as usize;
        let max_x = ((x0.max(x1) + pad).ceil() as isize).clamp(0, self.width as isize) as usize;
        let max_y = ((y0.max(y1) + pad).ceil() as isize).clamp(0, self.height as isize) as usize;

        let dx = x1 - x0;
        let dy = y1 - y0;
        let len_sq = dx * dx + dy * dy;

        for py in min_y..max_y {
            let fy = py as f64 + 0.5;
            for px in min_x..max_x {
                let fx = px as f64 + 0.5;
                let t = if len_sq <= 1.0e-12 {
                    0.0
                } else {
                    (((fx - x0) * dx + (fy - y0) * dy) / len_sq).clamp(0.0, 1.0)
                };
                let proj_x = x0 + t * dx;
                let proj_y = y0 + t * dy;
                let dist = ((fx - proj_x).powi(2) + (fy - proj_y).powi(2)).sqrt() - half_t;
                if dist < 0.75 {
                    let cov = (0.75 - dist) / 1.5;
                    self.blend_pixel_cov(px, py, rgba, cov);
                }
            }
        }
    }

    /// Anti-aliased dashed line segment.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_dashed_line(
        &mut self,
        x0: f64,
        y0: f64,
        x1: f64,
        y1: f64,
        dash_len: f64,
        gap_len: f64,
        thickness: f64,
        rgba: [u8; 4],
    ) {
        let dx = x1 - x0;
        let dy = y1 - y0;
        let total = (dx * dx + dy * dy).sqrt();
        if total < 1.0e-6 || dash_len <= 0.0 {
            return;
        }
        let step = (dash_len + gap_len.max(0.5)).max(1.0);
        let ux = dx / total;
        let uy = dy / total;
        let mut s = 0.0;
        while s < total {
            let e = (s + dash_len).min(total);
            self.draw_line(
                x0 + s * ux,
                y0 + s * uy,
                x0 + e * ux,
                y0 + e * uy,
                thickness,
                rgba,
            );
            s += step;
        }
    }

    /// Anti-aliased filled disc.
    pub fn fill_circle(&mut self, cx: f64, cy: f64, radius: f64, rgba: [u8; 4]) {
        if radius <= 0.0 || rgba[3] == 0 {
            return;
        }
        let pad = radius + 1.5;
        let x0 = (cx - pad).floor().max(0.0) as usize;
        let y0 = (cy - pad).floor().max(0.0) as usize;
        let x1 = ((cx + pad).ceil() as isize).clamp(0, self.width as isize) as usize;
        let y1 = ((cy + pad).ceil() as isize).clamp(0, self.height as isize) as usize;

        for py in y0..y1 {
            let dy = py as f64 + 0.5 - cy;
            for px in x0..x1 {
                let dx = px as f64 + 0.5 - cx;
                let dist = (dx * dx + dy * dy).sqrt() - radius;
                if dist < 0.75 {
                    let cov = (0.75 - dist) / 1.5;
                    self.blend_pixel_cov(px, py, rgba, cov);
                }
            }
        }
    }

    /// Anti-aliased stroked circle ring.
    pub fn stroke_circle(&mut self, cx: f64, cy: f64, radius: f64, thickness: f64, rgba: [u8; 4]) {
        if radius <= 0.0 || thickness <= 0.0 || rgba[3] == 0 {
            return;
        }
        let half_t = 0.5 * thickness.max(0.75);
        let pad = radius + half_t + 1.5;
        let x0 = (cx - pad).floor().max(0.0) as usize;
        let y0 = (cy - pad).floor().max(0.0) as usize;
        let x1 = ((cx + pad).ceil() as isize).clamp(0, self.width as isize) as usize;
        let y1 = ((cy + pad).ceil() as isize).clamp(0, self.height as isize) as usize;

        for py in y0..y1 {
            let dy = py as f64 + 0.5 - cy;
            for px in x0..x1 {
                let dx = px as f64 + 0.5 - cx;
                let dist = ((dx * dx + dy * dy).sqrt() - radius).abs() - half_t;
                if dist < 0.75 {
                    let cov = (0.75 - dist) / 1.5;
                    self.blend_pixel_cov(px, py, rgba, cov);
                }
            }
        }
    }

    /// Anti-aliased connected polyline.
    pub fn draw_polyline(&mut self, pts: &[[f64; 2]], thickness: f64, rgba: [u8; 4]) {
        for win in pts.windows(2) {
            self.draw_line(win[0][0], win[0][1], win[1][0], win[1][1], thickness, rgba);
        }
    }

    /// Measure pixel width of `text` rendered at `scale`.
    pub fn measure_text_width(&self, text: &str, scale: f64) -> f64 {
        let count = text.chars().count();
        if count == 0 {
            return 0.0;
        }
        (count as f64) * 6.0 * scale.max(0.5)
    }

    /// Crisp scalable 5x7 ASCII + Greek/math bitmap glyph renderer (`1x`, `1.5x`, `2x`, `3x`, ...).
    pub fn draw_text(&mut self, x: f64, y: f64, text: &str, scale: f64, rgba: [u8; 4]) {
        if rgba[3] == 0 || text.is_empty() {
            return;
        }
        let sc = scale.max(0.5);
        let mut cursor_x = x;
        for ch in text.chars() {
            let glyph = glyph_5x7(ch);
            for (row_idx, &bits) in glyph.iter().enumerate() {
                if bits == 0 {
                    continue;
                }
                for col_idx in 0..5 {
                    if (bits & (1 << (4 - col_idx))) != 0 {
                        let gx0 = cursor_x + (col_idx as f64) * sc;
                        let gy0 = y + (row_idx as f64) * sc;
                        let gx1 = gx0 + sc;
                        let gy1 = gy0 + sc;
                        let px0 = gx0.floor().max(0.0) as usize;
                        let py0 = gy0.floor().max(0.0) as usize;
                        let px1 = (gx1.ceil() as isize).clamp(0, self.width as isize) as usize;
                        let py1 = (gy1.ceil() as isize).clamp(0, self.height as isize) as usize;
                        for py in py0..py1 {
                            let oy = ((py as f64 + 1.0).min(gy1) - (py as f64).max(gy0)).max(0.0);
                            for px in px0..px1 {
                                let ox =
                                    ((px as f64 + 1.0).min(gx1) - (px as f64).max(gx0)).max(0.0);
                                let cov = (ox * oy).clamp(0.0, 1.0);
                                if cov > 0.02 {
                                    self.blend_pixel_cov(px, py, rgba, cov);
                                }
                            }
                        }
                    }
                }
            }
            cursor_x += 6.0 * sc;
        }
    }

    /// Animated spotlight cursor with expanding ripple ring (`t_sec` drives ripple phase).
    pub fn draw_spotlight_cursor(&mut self, x: f64, y: f64, t_sec: f64, active_pulse: f64) {
        let phase = (t_sec * 1.4).fract();
        let ripple_r = 9.0 + phase * 22.0 * active_pulse.clamp(0.4, 1.6);
        let ripple_alpha = ((1.0 - phase) * 185.0).round() as u8;
        self.stroke_circle(x, y, ripple_r, 2.0, [56, 189, 248, ripple_alpha]);
        self.fill_circle(x, y, 5.5, [248, 250, 252, 235]);
        self.fill_circle(x, y, 3.2, [14, 165, 233, 255]);
        // Subtle pointer triangle stem
        self.draw_line(x + 2.0, y + 2.0, x + 13.0, y + 15.0, 2.6, [15, 23, 42, 220]);
        self.draw_line(
            x + 2.0,
            y + 2.0,
            x + 12.0,
            y + 14.0,
            1.6,
            [248, 250, 252, 245],
        );
    }

    /// Render the shared top header + prominent color-coded Validation Scope Badge + bottom cinema HUD bar.
    pub fn draw_tour_chrome(&mut self, bundle: &TourBundle, t_sec: f64) {
        let w = self.width as f64;
        let h = self.height as f64;

        // Top header bar
        self.fill_rounded_rect(14.0, 10.0, w - 28.0, 62.0, 10.0, [15, 23, 42, 232]);
        self.stroke_rounded_rect(14.0, 10.0, w - 28.0, 62.0, 10.0, 1.2, [51, 65, 85, 255]);

        // FERRIC brand + module index pill (#B7410E / #FB923C)
        self.fill_rounded_rect(24.0, 20.0, 48.0, 42.0, 8.0, [183, 65, 14, 245]);
        self.stroke_rounded_rect(24.0, 20.0, 48.0, 42.0, 8.0, 1.5, [251, 146, 60, 255]);
        self.draw_text(29.0, 24.5, "FERRIC", 0.9, [251, 146, 60, 255]);
        let idx_str = format!("{:02}", bundle.module_index);
        self.draw_text(36.0, 37.0, &idx_str, 1.8, [248, 250, 252, 255]);

        // Title & subtitle
        self.draw_text(82.0, 20.0, &bundle.title, 2.0, [248, 250, 252, 255]);
        self.draw_text(82.0, 44.0, &bundle.subtitle, 1.35, [148, 163, 184, 255]);

        // Prominent color-coded Validation Scope Badge
        let (badge_bg, badge_border, badge_ink) = match bundle.validation_badge.kind.as_str() {
            "empirical_calibrated" | "fid_ingest_synthetic_nus" => {
                ([6, 78, 59, 245], [52, 211, 153, 255], [209, 250, 229, 255])
            }
            "cold_gate_failed_withdrawn" | "cold_holdout_disclosed" => (
                [120, 53, 15, 245],
                [251, 191, 36, 255],
                [254, 243, 199, 255],
            ),
            _ => (
                [136, 19, 55, 245],
                [251, 113, 133, 255],
                [255, 228, 230, 255],
            ),
        };
        let badge_scale = 1.45;
        let badge_tw = self.measure_text_width(&bundle.validation_badge.label, badge_scale);
        let badge_w = (badge_tw + 28.0).clamp(210.0, 540.0);
        let badge_x = w - 24.0 - badge_w;
        self.fill_rounded_rect(badge_x, 22.0, badge_w, 36.0, 18.0, badge_bg);
        self.stroke_rounded_rect(badge_x, 22.0, badge_w, 36.0, 18.0, 1.5, badge_border);
        self.fill_circle(badge_x + 14.0, 40.0, 4.5, badge_border);
        self.draw_text(
            badge_x + 24.0,
            34.5,
            &bundle.validation_badge.label,
            badge_scale,
            badge_ink,
        );

        // Bottom cinema HUD bar
        let hud_y = h - 84.0;
        self.fill_rounded_rect(14.0, hud_y, w - 28.0, 74.0, 10.0, [15, 23, 42, 236]);
        self.stroke_rounded_rect(14.0, hud_y, w - 28.0, 74.0, 10.0, 1.2, [51, 65, 85, 255]);

        let ch_idx = active_chapter_index(bundle, t_sec);
        let ch = bundle.chapters.get(ch_idx);
        let ch_title = ch.map(|c| c.title.as_str()).unwrap_or("Overview");
        let ch_caption = ch.map(|c| c.caption.as_str()).unwrap_or("");
        let ch_pill = format!("CH {}/4: {}", ch_idx + 1, ch_title);

        let pill_w = (self.measure_text_width(&ch_pill, 1.5) + 20.0).min(410.0);
        self.fill_rounded_rect(26.0, hud_y + 10.0, pill_w, 24.0, 12.0, [30, 64, 175, 245]);
        self.stroke_rounded_rect(
            26.0,
            hud_y + 10.0,
            pill_w,
            24.0,
            12.0,
            1.0,
            [96, 165, 250, 255],
        );
        self.draw_text(36.0, hud_y + 16.5, &ch_pill, 1.5, [239, 246, 255, 255]);

        // CLI command pill on right (auto-fit width so long commands never clip)
        let max_cli_w = (w - 68.0 - pill_w).max(240.0);
        let raw_cli_tw = self.measure_text_width(&bundle.cli_command, 1.2);
        let cli_scale = if raw_cli_tw + 18.0 > max_cli_w {
            (1.2 * (max_cli_w - 18.0) / raw_cli_tw.max(1.0)).max(0.72)
        } else {
            1.2
        };
        let cli_tw = self.measure_text_width(&bundle.cli_command, cli_scale);
        let cli_w = (cli_tw + 18.0).min(max_cli_w);
        let cli_x = w - 26.0 - cli_w;
        self.fill_rounded_rect(cli_x, hud_y + 10.0, cli_w, 24.0, 6.0, [30, 41, 59, 255]);
        self.stroke_rounded_rect(
            cli_x,
            hud_y + 10.0,
            cli_w,
            24.0,
            6.0,
            1.0,
            [71, 85, 105, 255],
        );
        self.draw_text(
            cli_x + 9.0,
            hud_y + 17.5,
            &bundle.cli_command,
            cli_scale,
            [125, 211, 252, 255],
        );

        // Caption line
        self.draw_text(28.0, hud_y + 39.0, ch_caption, 1.35, [226, 232, 240, 255]);

        // Progress bar 0:00..0:16
        let bar_x = 28.0;
        let bar_y = hud_y + 58.0;
        let bar_w = w - 160.0;
        let frac = (t_sec / bundle.duration_sec.max(1.0)).clamp(0.0, 1.0);
        self.fill_rounded_rect(bar_x, bar_y, bar_w, 7.0, 3.5, [51, 65, 85, 255]);
        if frac > 0.002 {
            self.fill_rounded_rect(
                bar_x,
                bar_y,
                (bar_w * frac).max(7.0),
                7.0,
                3.5,
                [56, 189, 248, 255],
            );
        }
        for tick in 1..4 {
            let tx = bar_x + bar_w * (tick as f64 / 4.0);
            self.draw_line(tx, bar_y - 2.0, tx, bar_y + 9.0, 1.5, [148, 163, 184, 220]);
        }
        let head_x = bar_x + bar_w * frac;
        self.fill_circle(head_x, bar_y + 3.5, 5.5, [248, 250, 252, 255]);

        let time_str = format!("0:{:02.0} / 0:16", t_sec.floor().clamp(0.0, 16.0));
        self.draw_text(
            w - 118.0,
            hud_y + 56.0,
            &time_str,
            1.4,
            [203, 213, 225, 255],
        );
    }
}

fn active_chapter_index(bundle: &TourBundle, t_sec: f64) -> usize {
    for (i, ch) in bundle.chapters.iter().enumerate() {
        if t_sec >= ch.start_sec && t_sec < ch.end_sec {
            return i;
        }
    }
    bundle.chapters.len().saturating_sub(1)
}

#[inline]
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

#[inline]
fn smoothstep(t: f64) -> f64 {
    let x = t.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

#[inline]
fn s2_color_rgba(s2: f64) -> [u8; 4] {
    // Low S^2 (flexible ~0.3) -> rose/amber; rigid core (~0.88) -> cyan/teal
    let u = ((s2 - 0.30) / 0.60).clamp(0.0, 1.0);
    if u < 0.5 {
        let w = u * 2.0;
        [
            lerp(244.0, 251.0, w).round() as u8,
            lerp(63.0, 191.0, w).round() as u8,
            lerp(94.0, 36.0, w).round() as u8,
            255,
        ]
    } else {
        let w = (u - 0.5) * 2.0;
        [
            lerp(251.0, 14.0, w).round() as u8,
            lerp(191.0, 165.0, w).round() as u8,
            lerp(36.0, 233.0, w).round() as u8,
            255,
        ]
    }
}

// ============================================================================
// Compact 5x7 ASCII + NMR Greek/Math Glyph Table
// ============================================================================

fn glyph_5x7(ch: char) -> [u8; 7] {
    match ch {
        ' ' => [0, 0, 0, 0, 0, 0, 0],
        '!' => [0x04, 0x04, 0x04, 0x04, 0x04, 0x00, 0x04],
        '"' => [0x0A, 0x0A, 0x00, 0x00, 0x00, 0x00, 0x00],
        '#' => [0x0A, 0x1F, 0x0A, 0x0A, 0x1F, 0x0A, 0x00],
        '$' => [0x04, 0x0F, 0x14, 0x0E, 0x05, 0x1E, 0x04],
        '%' => [0x18, 0x19, 0x02, 0x04, 0x08, 0x13, 0x03],
        '&' => [0x0C, 0x12, 0x14, 0x08, 0x15, 0x12, 0x0D],
        '\'' => [0x04, 0x04, 0x08, 0x00, 0x00, 0x00, 0x00],
        '(' => [0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02],
        ')' => [0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08],
        '*' => [0x00, 0x0A, 0x04, 0x1F, 0x04, 0x0A, 0x00],
        '+' => [0x00, 0x04, 0x04, 0x1F, 0x04, 0x04, 0x00],
        ',' => [0x00, 0x00, 0x00, 0x00, 0x04, 0x04, 0x08],
        '-' | '–' | '—' => [0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00],
        '.' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x06, 0x06],
        '/' => [0x01, 0x01, 0x02, 0x04, 0x08, 0x10, 0x10],
        '0' => [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E],
        '1' => [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E],
        '2' => [0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F],
        '3' => [0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E],
        '4' => [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02],
        '5' => [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E],
        '6' => [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E],
        '7' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
        '8' => [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E],
        '9' => [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C],
        ':' => [0x00, 0x06, 0x06, 0x00, 0x06, 0x06, 0x00],
        ';' => [0x00, 0x06, 0x06, 0x00, 0x06, 0x02, 0x04],
        '<' => [0x02, 0x04, 0x08, 0x10, 0x08, 0x04, 0x02],
        '=' => [0x00, 0x00, 0x1F, 0x00, 0x1F, 0x00, 0x00],
        '>' => [0x08, 0x04, 0x02, 0x01, 0x02, 0x04, 0x08],
        '?' => [0x0E, 0x11, 0x01, 0x02, 0x04, 0x00, 0x04],
        '@' => [0x0E, 0x11, 0x17, 0x15, 0x17, 0x10, 0x0E],
        'A' => [0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        'B' => [0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E],
        'C' => [0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E],
        'D' => [0x1C, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1C],
        'E' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F],
        'F' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10],
        'G' => [0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F],
        'H' => [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        'I' => [0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E],
        'J' => [0x07, 0x02, 0x02, 0x02, 0x12, 0x12, 0x0C],
        'K' => [0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11],
        'L' => [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F],
        'M' => [0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11],
        'N' => [0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11],
        'O' => [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'P' => [0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10],
        'Q' => [0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D],
        'R' => [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11],
        'S' => [0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E],
        'T' => [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
        'U' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'V' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04],
        'W' => [0x11, 0x11, 0x11, 0x15, 0x15, 0x1B, 0x11],
        'X' => [0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11],
        'Y' => [0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04],
        'Z' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F],
        '[' => [0x0E, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0E],
        '\\' => [0x10, 0x10, 0x08, 0x04, 0x02, 0x01, 0x01],
        ']' => [0x0E, 0x02, 0x02, 0x02, 0x02, 0x02, 0x0E],
        '^' => [0x04, 0x0A, 0x11, 0x00, 0x00, 0x00, 0x00],
        '_' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1F],
        '`' => [0x08, 0x04, 0x02, 0x00, 0x00, 0x00, 0x00],
        'a' => [0x00, 0x00, 0x0E, 0x01, 0x0F, 0x11, 0x0F],
        'b' => [0x10, 0x10, 0x16, 0x19, 0x11, 0x11, 0x1E],
        'c' => [0x00, 0x00, 0x0E, 0x11, 0x10, 0x11, 0x0E],
        'd' => [0x01, 0x01, 0x0D, 0x13, 0x11, 0x11, 0x0F],
        'e' => [0x00, 0x00, 0x0E, 0x11, 0x1F, 0x10, 0x0E],
        'f' => [0x06, 0x09, 0x08, 0x1E, 0x08, 0x08, 0x08],
        'g' => [0x00, 0x0F, 0x11, 0x11, 0x0F, 0x01, 0x0E],
        'h' => [0x10, 0x10, 0x16, 0x19, 0x11, 0x11, 0x11],
        'i' => [0x04, 0x00, 0x0C, 0x04, 0x04, 0x04, 0x0E],
        'j' => [0x02, 0x00, 0x06, 0x02, 0x02, 0x12, 0x0C],
        'k' => [0x10, 0x10, 0x12, 0x14, 0x18, 0x14, 0x12],
        'l' => [0x0C, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E],
        'm' => [0x00, 0x00, 0x1A, 0x15, 0x15, 0x11, 0x11],
        'n' => [0x00, 0x00, 0x16, 0x19, 0x11, 0x11, 0x11],
        'o' => [0x00, 0x00, 0x0E, 0x11, 0x11, 0x11, 0x0E],
        'p' => [0x00, 0x00, 0x1E, 0x11, 0x1E, 0x10, 0x10],
        'q' => [0x00, 0x00, 0x0F, 0x11, 0x0F, 0x01, 0x01],
        'r' => [0x00, 0x00, 0x16, 0x19, 0x10, 0x10, 0x10],
        's' => [0x00, 0x00, 0x0F, 0x10, 0x0E, 0x01, 0x1E],
        't' => [0x08, 0x08, 0x1E, 0x08, 0x08, 0x09, 0x06],
        'u' => [0x00, 0x00, 0x11, 0x11, 0x11, 0x13, 0x0D],
        'v' => [0x00, 0x00, 0x11, 0x11, 0x11, 0x0A, 0x04],
        'w' => [0x00, 0x00, 0x11, 0x11, 0x15, 0x15, 0x0A],
        'x' => [0x00, 0x00, 0x11, 0x0A, 0x04, 0x0A, 0x11],
        'y' => [0x00, 0x00, 0x11, 0x11, 0x0F, 0x01, 0x0E],
        'z' => [0x00, 0x00, 0x1F, 0x02, 0x04, 0x08, 0x1F],
        '{' => [0x02, 0x04, 0x04, 0x08, 0x04, 0x04, 0x02],
        '|' => [0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
        '}' => [0x08, 0x04, 0x04, 0x02, 0x04, 0x04, 0x08],
        '~' => [0x00, 0x08, 0x15, 0x02, 0x00, 0x00, 0x00],
        // Greek & math NMR symbols
        'α' => [0x00, 0x00, 0x0D, 0x12, 0x12, 0x12, 0x0D],
        'β' => [0x0C, 0x12, 0x12, 0x1C, 0x12, 0x12, 0x1C],
        'γ' => [0x00, 0x00, 0x11, 0x0A, 0x04, 0x04, 0x08],
        'δ' => [0x06, 0x08, 0x04, 0x0E, 0x11, 0x11, 0x0E],
        'Δ' => [0x04, 0x04, 0x0A, 0x0A, 0x11, 0x11, 0x1F],
        'τ' => [0x00, 0x00, 0x1F, 0x04, 0x04, 0x04, 0x06],
        'λ' => [0x10, 0x08, 0x04, 0x0A, 0x0A, 0x11, 0x11],
        'σ' => [0x00, 0x00, 0x0F, 0x14, 0x12, 0x12, 0x0C],
        'μ' => [0x00, 0x00, 0x11, 0x11, 0x1E, 0x10, 0x10],
        'ν' => [0x00, 0x00, 0x11, 0x11, 0x0A, 0x0C, 0x08],
        'χ' => [0x00, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11],
        'ω' => [0x00, 0x00, 0x11, 0x11, 0x15, 0x15, 0x0A],
        '°' => [0x06, 0x09, 0x09, 0x06, 0x00, 0x00, 0x00],
        '±' => [0x04, 0x04, 0x1F, 0x04, 0x04, 0x00, 0x1F],
        '×' => [0x00, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x00],
        '·' => [0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00],
        '→' => [0x00, 0x04, 0x02, 0x1F, 0x02, 0x04, 0x00],
        '←' => [0x00, 0x04, 0x08, 0x1F, 0x08, 0x04, 0x00],
        '≥' => [0x08, 0x04, 0x02, 0x04, 0x08, 0x00, 0x1E],
        '≤' => [0x02, 0x04, 0x08, 0x04, 0x02, 0x00, 0x1E],
        '≈' => [0x00, 0x08, 0x15, 0x02, 0x08, 0x15, 0x02],
        'Å' => [0x04, 0x00, 0x0E, 0x11, 0x1F, 0x11, 0x11],
        '¹' => [0x04, 0x0C, 0x04, 0x04, 0x0E, 0x00, 0x00],
        '²' => [0x0C, 0x02, 0x04, 0x08, 0x0E, 0x00, 0x00],
        '³' => [0x0E, 0x02, 0x06, 0x02, 0x0E, 0x00, 0x00],
        '⁵' => [0x0E, 0x08, 0x0C, 0x02, 0x0C, 0x00, 0x00],
        '∞' => [0x00, 0x00, 0x0A, 0x15, 0x15, 0x0A, 0x00],
        _ => [0x1F, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1F],
    }
}

// ============================================================================
// Core Module State (`01..=04`) Built from Real `nmr-core` Physics
// ============================================================================

enum CoreSceneData {
    M01FidPulprogNus(Box<M01SceneData>),
    M02HsqcStripsSolvers(Box<M02SceneData>),
    M03StructureNoeRdc(Box<M03SceneData>),
    M04RelaxModelfree(Box<M04SceneData>),
}

struct CoreTourState {
    bundle: TourBundle,
    scene: CoreSceneData,
}

struct M01SceneData {
    pulprog_name: String,
    aqseq: String,
    f1_quad: String,
    f2_quad: String,
    transfer_summary: String,
    fid_real: Vec<f64>,
    sine_window: Vec<f64>,
    fft_unphased: Vec<f64>,
    fft_phased: Vec<f64>,
    p0_deg: f64,
    p1_deg: f64,
    nus_mask: Vec<bool>,
    nus_dirty_spec: Vec<f64>,
    nus_fista_iters: Vec<Vec<f64>>,
    nus_residual_l1: Vec<f64>,
}

struct StripSliceData {
    seq_id: i32,
    label: String,
    h_ppm: f64,
    n_ppm: f64,
    ca_i: Option<f64>,
    cb_i: Option<f64>,
    ca_im1: Option<f64>,
    cb_im1: Option<f64>,
    pos_pts: Vec<[f64; 2]>,
    neg_pts: Vec<[f64; 2]>,
}

struct M02SceneData {
    hsqc_pos_contours: Vec<Vec<[f64; 2]>>,
    hsqc_neg_contours: Vec<Vec<[f64; 2]>>,
    hsqc_roots: Vec<(f64, f64, String)>,
    strips: Vec<StripSliceData>,
    cv_vals: Vec<f64>,
    t_crit: f64,
    max_cv: f64,
    res_confidences: Vec<f64>,
    evo_acc_pct: f64,
    qc_verdict: String,
    n_high_conf: usize,
}

struct M03SceneData {
    ca_coords: Vec<[f64; 3]>,
    ss_classes: Vec<u8>, // 0=coil, 1=helix, 2=sheet
    s2_vals: Vec<f64>,
    noe_pairs: Vec<(usize, usize, f64)>,
    rdc_points: Vec<(f64, f64, bool, i32)>,
    q_work: f64,
    q_free: f64,
    da_nh_hz: f64,
    rhombicity: f64,
    n_torsions_firewalled: usize,
    n_excluded_templates: usize,
}

struct M04SceneData {
    snr_levels: Vec<f64>,
    locked_ls_bias_pct: Vec<f64>,
    local_max_bias_pct: Vec<f64>,
    times_r1: Vec<f64>,
    curve_600_r1: Vec<f64>,
    curve_800_r1: Vec<f64>,
    fit_600_rate: f64,
    fit_600_se: f64,
    fit_800_rate: f64,
    fit_800_se: f64,
    tau_m_ns: f64,
    d_par_perp: f64,
    seq_ids: Vec<i32>,
    r2_over_r1: Vec<f64>,
    s2_vals: Vec<f64>,
    rex_vals: Vec<f64>,
}

fn build_core_tour_state(module: TourModuleId, seed: u64) -> Result<CoreTourState, String> {
    match module {
        TourModuleId::M01FidPulprogNus => build_m01_state(seed),
        TourModuleId::M02HsqcStripsSolvers => build_m02_state(seed),
        TourModuleId::M03StructureNoeRdc => build_m03_state(seed),
        TourModuleId::M04RelaxModelfree => build_m04_state(seed),
        _ => Err(format!("module {} is not a core module", module.slug())),
    }
}

fn build_m01_state(_seed: u64) -> Result<CoreTourState, String> {
    // 1. Real Bruker pulseprogram AST & quad detection via `nmr_core::pulprog`
    let acqus = "##TITLE= Bruker 3D HNCACB\n##$AQSEQ= 1\n##$NUC1= <1H>\n##$SFO1= 600.13\n##$SW= 14.0\n##$TD= 2048\n##$DSPFVS= 20\n##$DECIM= 24\n##$ZGOPTNS= <-DLABEL_CN -DNUS>\n";
    let acqu2s = "##$NUC1= <15N>\n##$SFO1= 60.81\n##$SW= 32.0\n##$TD= 64\n##$FnMODE= 6\n";
    let acqu3s = "##$NUC1= <13C>\n##$SFO1= 150.90\n##$SW= 64.0\n##$TD= 96\n##$FnMODE= 5\n";
    let pulprog_src = "\
;hncacbgpwg3d\n\
\"d11=30m\"\n\
1 ze\n\
2 d1\n\
  (p1 ph1):f1\n\
  (p3 ph2):f2\n\
  (p5 ph3):f3\n\
  d0\n\
  igrad EA\n\
  go=2 ph31\n\
  d11 mc #0 to 2\n\
    F1EA(igrad EA, id0)\n\
    F2PH(ip3, id10)\n\
exit\n";
    let nuslist =
        "0 0\n0 2\n1 0\n2 1\n3 4\n4 2\n5 7\n6 3\n8 5\n10 8\n12 1\n15 9\n18 4\n22 11\n26 7\n30 14\n";
    let interp = interpret_bruker_experiment(&BrukerExperimentInputs {
        acqus_text: acqus,
        acqu2s_text: Some(acqu2s),
        acqu3s_text: Some(acqu3s),
        acqu4s_text: None,
        pulseprogram_text: Some(pulprog_src),
        nuslist_text: Some(nuslist),
        vdlist_text: None,
        vclist_text: None,
        fqlist_text: None,
        difflist_text: None,
        local_includes: Default::default(),
    })
    .map_err(|e| format!("pulprog interpretation failed: {e}"))?;

    let aqseq_str = format!("{:?}", interp.aqseq);
    let f1_quad = interp
        .indirect_dims
        .first()
        .map(|d| format!("{:?}", d.quad_mode))
        .unwrap_or_else(|| "EchoAntiecho".to_string());
    let f2_quad = interp
        .indirect_dims
        .get(1)
        .map(|d| format!("{:?}", d.quad_mode))
        .unwrap_or_else(|| "StatesTppi".to_string());
    let transfer_summary = format!(
        "{:?} ({}D)",
        interp.transfer_graph.deduced_family, interp.dimensionality
    );

    // 2. Real 1D FID decay + shifted sine-bell apodization + FFT + Chen(2002) phase correction
    let n_td = 64usize;
    let n_fft = 128usize;
    let peaks_spec = [
        (0.18_f64, 1.00_f64, 0.045_f64),
        (0.36_f64, 0.72_f64, 0.038_f64),
        (0.58_f64, 0.24_f64, 0.032_f64), // weaker peak (tests NUS dynamic range!)
        (0.77_f64, 0.85_f64, 0.042_f64),
    ];
    let phase0_true = 0.62_f64;
    let phase1_true = -0.35_f64;

    let mut fid_raw: Vec<Complex64> = Vec::with_capacity(n_td);
    let mut sine_win: Vec<f64> = Vec::with_capacity(n_td);
    for k in 0..n_td {
        let t = (k as f64) / (n_td as f64);
        let w = (PI * (0.15 + 0.85 * t)).sin().powf(1.2);
        sine_win.push(w);
        let mut re = 0.0;
        let mut im = 0.0;
        for &(freq, amp, decay) in &peaks_spec {
            let env = amp * (-t / (0.28 + decay)).exp();
            let ang = TAU * freq * (k as f64) + phase0_true + phase1_true * (freq - 0.5);
            re += env * ang.cos();
            im += env * ang.sin();
        }
        fid_raw.push([re, im]);
    }
    let fid_real: Vec<f64> = fid_raw.iter().map(|c| c[0]).collect();

    let mut spec_buf = vec![[0.0_f64, 0.0_f64]; n_fft];
    for k in 0..n_td {
        spec_buf[k] = [fid_raw[k][0] * sine_win[k], fid_raw[k][1] * sine_win[k]];
    }
    fft_in_place(&mut spec_buf, false);
    let fft_unphased: Vec<f64> = spec_buf.iter().map(|c| c[0]).collect();
    let (p0_rad, p1_rad) = auto_phase_chen2002(&spec_buf);
    let mut fft_phased = Vec::with_capacity(n_fft);
    for (idx, c) in spec_buf.iter().enumerate() {
        let f = (idx as f64) / (n_fft as f64);
        let phi = p0_rad + p1_rad * (f - 0.5);
        fft_phased.push(c[0] * phi.cos() - c[1] * phi.sin());
    }

    // 3. Real 25% Poisson-gap NUS schedule (16/64 points) + FISTA reconstruction across 0..=20 iters
    let mut nus_mask = vec![false; n_td];
    let mut rng = Rng::new(CANONICAL_TOUR_SEED);
    nus_mask[0] = true;
    let mut pos = 0usize;
    while pos < n_td {
        nus_mask[pos] = true;
        let step = 2 + ((rng.next_u64() % 4) as usize);
        pos += step;
    }
    // Ensure exactly 16 sampled points (25%)
    let mut sampled_count = nus_mask.iter().filter(|&&m| m).count();
    for (idx, flag) in nus_mask.iter_mut().enumerate() {
        if sampled_count >= 16 {
            if idx > 0 && sampled_count > 16 && *flag {
                *flag = false;
                sampled_count -= 1;
            }
        } else if !*flag && idx % 4 == 1 {
            *flag = true;
            sampled_count += 1;
        }
    }

    let mut fid_clean: Vec<Complex64> = Vec::with_capacity(n_td);
    for k in 0..n_td {
        let t = (k as f64) / (n_td as f64);
        let mut re = 0.0;
        let mut im = 0.0;
        for &(freq, amp, decay) in &peaks_spec {
            let env = amp * (-t / (0.35 + decay)).exp();
            let ang = TAU * freq * (k as f64);
            re += env * ang.cos();
            im += env * ang.sin();
        }
        fid_clean.push([re, im]);
    }

    let mag_spectrum_from_fid = |td: &[Complex64]| -> Vec<f64> {
        let mut buf = vec![[0.0, 0.0]; n_fft];
        for (k, &v) in td.iter().enumerate().take(n_td) {
            buf[k] = [v[0] * sine_win[k], v[1] * sine_win[k]];
        }
        fft_in_place(&mut buf, false);
        buf.iter().map(|c| c[0].max(0.0)).collect()
    };

    let mut zero_filled = vec![[0.0, 0.0]; n_td];
    for k in 0..n_td {
        if nus_mask[k] {
            zero_filled[k] = fid_clean[k];
        }
    }
    let nus_dirty_spec = mag_spectrum_from_fid(&zero_filled);

    let mut nus_fista_iters = Vec::with_capacity(21);
    let mut nus_residual_l1 = Vec::with_capacity(21);
    let ref_full_spec = mag_spectrum_from_fid(&fid_clean);
    let ref_norm: f64 = ref_full_spec.iter().sum::<f64>().max(1.0e-9);

    for it in 0..=20 {
        let spec = if it == 0 {
            nus_dirty_spec.clone()
        } else {
            let recon = reconstruct_nus_1d_ist(
                &fid_clean,
                &nus_mask,
                &NusIstConfig {
                    iterations: it,
                    initial_thresh_frac: 0.72,
                    final_noise_sigma: 2.2,
                    fista_momentum: true,
                },
            );
            mag_spectrum_from_fid(&recon)
        };
        let l1_err: f64 = spec
            .iter()
            .zip(ref_full_spec.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            / ref_norm;
        nus_fista_iters.push(spec);
        nus_residual_l1.push((l1_err * 1000.0).round() / 1000.0);
    }

    let bundle = TourBundle {
        schema_version: TOUR_SCHEMA_VERSION.to_string(),
        module_index: 1,
        slug: TourModuleId::M01FidPulprogNus.slug().to_string(),
        title: "Raw-FID DSP, Pulprog AST & 25% NUS FISTA".to_string(),
        subtitle: "Bruker pulseprogram physics · Sine-bell FFT + Chen phase · Hypercomplex IST/FISTA".to_string(),
        manual_chapter: "02-custom-pulprog-and-dsp.html".to_string(),
        validation_badge: ValidationBadge {
            kind: "fid_ingest_synthetic_nus".to_string(),
            label: "RAW BRUKER/VARIAN FID INGEST · SYNTHETIC NUS DEMO".to_string(),
            detail: "Real Bruker/Varian raw-FID & pulseprogram AST ingest; 25% Poisson-gap NUS IST/FISTA reconstruction demo is synthetic.".to_string(),
        },
        duration_sec: TOUR_DURATION_SEC,
        fps: 30,
        cli_command: "ferric process --in fid_dir --out hncacb.ucsf --nus-method fista".to_string(),
        chapters: vec![
            TourChapter {
                index: 1,
                start_sec: 0.0,
                end_sec: 4.0,
                title: "Sandboxed Pulprog AST".to_string(),
                caption: "Interprets igrad EA (Echo-Antiecho) vs States-TPPI, AQSEQ 321/123, and magnetization transfer.".to_string(),
                metric_pill: format!("{} · {}", interp.pulse_program_name, transfer_summary),
            },
            TourChapter {
                index: 2,
                start_sec: 4.0,
                end_sec: 8.0,
                title: "FID Apodization & Calibrated FFT".to_string(),
                caption: "Damped time-domain FID multiplied by shifted sine-bell window, zero-filled, FFT'd & auto-phased.".to_string(),
                metric_pill: format!("P0={:.1} deg  P1={:.1} deg", p0_rad.to_degrees(), p1_rad.to_degrees()),
            },
            TourChapter {
                index: 3,
                start_sec: 8.0,
                end_sec: 12.0,
                title: "25% Poisson-Gap NUS & PSF Ridges".to_string(),
                caption: "Only 16/64 indirect samples acquired; zero-fill FFT creates severe point-spread sidelobe artifacts.".to_string(),
                metric_pill: "25% sampling (16/64 pts)".to_string(),
            },
            TourChapter {
                index: 4,
                start_sec: 12.0,
                end_sec: 16.0,
                title: "FISTA Shrinkage (0 -> 20 Iters)".to_string(),
                caption: "Proximal L1 soft-thresholding suppresses NUS ridges and restores the weak 0.58-ppm peak.".to_string(),
                metric_pill: format!(
                    "L1 error {:.3} -> {:.3}",
                    nus_residual_l1.first().copied().unwrap_or(0.0),
                    nus_residual_l1.last().copied().unwrap_or(0.0)
                ),
            },
        ],
        interactive_control: InteractiveControlSpec {
            id: "nus_iter".to_string(),
            label: "FISTA Iteration".to_string(),
            min: 0.0,
            max: 20.0,
            step: 1.0,
            default_val: 20.0,
            unit: "iter".to_string(),
        },
        payload: json!({
            "pulprog_name": interp.pulse_program_name,
            "aqseq": aqseq_str,
            "f1_quad": f1_quad,
            "f2_quad": f2_quad,
            "transfer_summary": transfer_summary,
            "fid_real": round_vec(&fid_real, 4),
            "sine_window": round_vec(&sine_win, 4),
            "fft_unphased": round_vec(&fft_unphased, 4),
            "fft_phased": round_vec(&fft_phased, 4),
            "p0_deg": round4(p0_rad.to_degrees()),
            "p1_deg": round4(p1_rad.to_degrees()),
            "nus_mask": nus_mask,
            "nus_dirty_spec": round_vec(&nus_dirty_spec, 4),
            "nus_fista_iters": nus_fista_iters.iter().map(|v| round_vec(v, 4)).collect::<Vec<_>>(),
            "nus_residual_l1": nus_residual_l1,
        }),
    };

    Ok(CoreTourState {
        bundle,
        scene: CoreSceneData::M01FidPulprogNus(Box::new(M01SceneData {
            pulprog_name: interp.pulse_program_name,
            aqseq: aqseq_str,
            f1_quad,
            f2_quad,
            transfer_summary,
            fid_real,
            sine_window: sine_win,
            fft_unphased,
            fft_phased,
            p0_deg: p0_rad.to_degrees(),
            p1_deg: p1_rad.to_degrees(),
            nus_mask,
            nus_dirty_spec,
            nus_fista_iters,
            nus_residual_l1,
        })),
    })
}

fn build_m02_state(seed: u64) -> Result<CoreTourState, String> {
    let ref_data = read_star_str(UBIQUITIN_STAR_FIXTURE, None)?;
    let sim_cfg = SimConfig {
        noise_h: 0.002,
        noise_n: 0.015,
        noise_c: 0.012,
        p_missing_peak: 0.0,
        p_missing_root: 0.0,
        spurious_rate: 0.0,
        sidechain_nh2: true,
        include_intensities: true,
        seed,
        ..SimConfig::default()
    };
    let (mut peaks, _truth) = simulate(&ref_data, &sim_cfg);

    let priors = Priors::provisional();
    let evo_cfg = EvoGraphConfig {
        population_size: 20,
        num_islands: 4,
        generations: 2,
        steps_per_system_per_gen: 180,
        seed,
        ..EvoGraphConfig::default()
    };
    let (systems, assign_res, _rep) =
        assign_evo_graph_from_peaks(&peaks, &ref_data.sequence, &priors, &evo_cfg);

    let _std_res = assign(
        &systems,
        &ref_data.sequence,
        &priors,
        &ScoreConfig::default(),
        &AnnealConfig {
            steps_per_system: 900,
            runs: 8,
            seed,
            ..AnnealConfig::default()
        },
    );

    let (_b_sys, _b_res, bayes_rep) = solve_bayes_sa_from_peaks(
        &peaks,
        &ref_data.sequence,
        &priors,
        &BayesSaConfig {
            replicas: 8,
            max_stages: 24,
            steps_per_stage_per_active: 32,
            seed,
            ..BayesSaConfig::default()
        },
    )?;

    let qc = evaluate_dataset_qc(
        Some(&peaks),
        &systems,
        &ref_data.sequence,
        &priors,
        &ScoreConfig::default(),
        &assign_res,
        None,
    );

    // Compute per-residue confidence profile 1..=76
    let mut res_confidences = vec![0.0_f64; ref_data.sequence.len()];
    for (s_idx, &pos) in assign_res.position.iter().enumerate() {
        if pos >= 0 && (pos as usize) < res_confidences.len() {
            let c = assign_res.confidence.get(s_idx).copied().unwrap_or(0.0);
            res_confidences[pos as usize] = res_confidences[pos as usize].max(c);
        }
    }
    let res_labels: Vec<String> = ref_data
        .sequence
        .iter()
        .enumerate()
        .map(|(i, r)| format!("{r}{}", i + 1))
        .collect();
    let n_high_conf = res_confidences.iter().filter(|&&c| c >= 0.80).count();
    let mut n_correct = 0usize;
    let mut n_assignable = 0usize;
    for (k, r) in ref_data.sequence.iter().enumerate() {
        if r == "PRO" || k == 0 {
            continue;
        }
        n_assignable += 1;
        if res_confidences[k] >= 0.50 {
            n_correct += 1;
        }
    }
    let evo_acc_pct = if n_assignable > 0 {
        100.0 * (n_correct as f64) / (n_assignable as f64)
    } else {
        0.0
    };

    // Folded Arg negative-phase peaks
    for &(h_ppm, n_ppm, height) in &[
        (7.36, 115.4, 82.0),
        (7.24, 116.8, 76.0),
        (7.49, 114.2, 68.0),
    ] {
        peaks.push(Peak {
            exp: "HSQC".to_string(),
            h: h_ppm,
            n: n_ppm,
            c: None,
            sign: -1,
            height: Some(height),
            volume: Some(height),
            h2: None,
        });
    }

    let hsqc_cfg = SpectrumRenderConfig {
        nx_h: 120,
        ny_n: 96,
        h_high_ppm: 9.85,
        h_low_ppm: 6.35,
        n_high_ppm: 131.5,
        n_low_ppm: 103.5,
        sf_h_mhz: 600.13,
        sf_n_mhz: 60.81,
        lw_h_ppm: 0.040,
        lw_n_ppm: 0.36,
        gaussian_mix: 0.65,
        noise_level: 0.30,
        seed,
        ..SpectrumRenderConfig::default()
    };
    let hsqc_spec = render_hsqc_2d(&peaks, &hsqc_cfg);
    let nx = hsqc_spec.axis_x.size;
    let ny = hsqc_spec.axis_y.size;
    let pos_grid: Vec<f64> = hsqc_spec.data.iter().map(|&v| v as f64).collect();
    let neg_grid: Vec<f64> = hsqc_spec.data.iter().map(|&v| -(v as f64)).collect();
    let max_pos = pos_grid.iter().copied().fold(1.0_f64, f64::max);

    let hsqc_pos_contours = extract_contour_polylines(
        &pos_grid,
        nx,
        ny,
        max_pos * 0.22,
        hsqc_cfg.h_high_ppm,
        hsqc_cfg.h_low_ppm,
        hsqc_cfg.n_low_ppm,
        hsqc_cfg.n_high_ppm,
    );
    let hsqc_neg_contours = extract_contour_polylines(
        &neg_grid,
        nx,
        ny,
        max_pos * 0.22,
        hsqc_cfg.h_high_ppm,
        hsqc_cfg.h_low_ppm,
        hsqc_cfg.n_low_ppm,
        hsqc_cfg.n_high_ppm,
    );

    let mut hsqc_roots = Vec::new();
    for (k, res_name) in ref_data.sequence.iter().enumerate() {
        if let (Some(&h), Some(&n)) = (
            ref_data.shifts.get(&(k, "H".to_string())),
            ref_data.shifts.get(&(k, "N".to_string())),
        ) {
            hsqc_roots.push((h, n, format!("{res_name}{}", k + 1)));
        }
    }

    // 3D HNCACB strips for residues 1..=10 (Met1..Gly10)
    let hncacb_cfg = SpectrumRenderConfig {
        nx_h: 96,
        ny_n: 64,
        nz_c: 120,
        h_high_ppm: 9.8,
        h_low_ppm: 6.6,
        n_high_ppm: 130.0,
        n_low_ppm: 105.0,
        c_aliph_high_ppm: 74.0,
        c_aliph_low_ppm: 14.0,
        lw_h_ppm: 0.042,
        lw_n_ppm: 0.45,
        lw_c_ppm: 0.65,
        gaussian_mix: 0.70,
        noise_level: 0.15,
        seed,
        ..SpectrumRenderConfig::default()
    };
    let cube = render_triple_3d("HNCACB", &peaks, &hncacb_cfg);
    let mut strips = Vec::new();
    for k in 0..10.min(ref_data.sequence.len()) {
        let seq_id = (k + 1) as i32;
        let label = format!("{}{}", ref_data.sequence[k], seq_id);
        let h_i = ref_data
            .shifts
            .get(&(k, "H".to_string()))
            .copied()
            .unwrap_or(8.2);
        let n_i = ref_data
            .shifts
            .get(&(k, "N".to_string()))
            .copied()
            .unwrap_or(120.0);
        let ca_i = ref_data.shifts.get(&(k, "CA".to_string())).copied();
        let cb_i = ref_data.shifts.get(&(k, "CB".to_string())).copied();
        let ca_im1 = if k > 0 {
            ref_data.shifts.get(&(k - 1, "CA".to_string())).copied()
        } else {
            None
        };
        let cb_im1 = if k > 0 {
            ref_data.shifts.get(&(k - 1, "CB".to_string())).copied()
        } else {
            None
        };

        // Extract 1D C trace at (h_i, n_i) for crisp fast strip visualization
        let ix = (cube.axis_x.ppm_to_index(h_i).round() as isize)
            .clamp(0, cube.axis_x.size as isize - 1) as usize;
        let iy = (cube.axis_y.ppm_to_index(n_i).round() as isize)
            .clamp(0, cube.axis_y.size as isize - 1) as usize;
        let mut pos_pts = Vec::with_capacity(cube.axis_z.size);
        let mut neg_pts = Vec::with_capacity(cube.axis_z.size);
        let mut max_c = 1.0_f64;
        let mut trace = Vec::with_capacity(cube.axis_z.size);
        for iz in 0..cube.axis_z.size {
            let c_ppm = cube.axis_z.index_to_ppm(iz as f64);
            let val = cube.data[(iz * cube.axis_y.size + iy) * cube.axis_x.size + ix] as f64;
            max_c = max_c.max(val.abs());
            trace.push((c_ppm, val));
        }
        for (c_ppm, val) in trace {
            pos_pts.push([c_ppm, (val / max_c).max(0.0)]);
            neg_pts.push([c_ppm, (-val / max_c).max(0.0)]);
        }
        strips.push(StripSliceData {
            seq_id,
            label,
            h_ppm: h_i,
            n_ppm: n_i,
            ca_i,
            cb_i,
            ca_im1,
            cb_im1,
            pos_pts,
            neg_pts,
        });
    }

    let cv_temps: Vec<f64> = bayes_rep
        .cooling_trajectory
        .iter()
        .map(|s| round4(s.temperature))
        .collect();
    let cv_vals: Vec<f64> = bayes_rep
        .cooling_trajectory
        .iter()
        .map(|s| round4(s.specific_heat_cv))
        .collect();
    let mean_energies: Vec<f64> = bayes_rep
        .cooling_trajectory
        .iter()
        .map(|s| round4(s.mean_energy))
        .collect();
    let qc_verdict = qc.verdict.as_str().to_string();

    let bundle = TourBundle {
        schema_version: TOUR_SCHEMA_VERSION.to_string(),
        module_index: 2,
        slug: TourModuleId::M02HsqcStripsSolvers.slug().to_string(),
        title: "2D HSQC, 3D HNCACB Strips & Backbone Solvers".to_string(),
        subtitle: "Synthetic UBQ (--realistic) 98.4% · Dev CISD3 64.38%/100% HC · Cold BMRB evo-graph 88.25-95.30% (FAILS >=94%; withdrawn)".to_string(),
        manual_chapter: "03-solvers-and-calibration.html".to_string(),
        validation_badge: ValidationBadge {
            kind: "cold_gate_failed_withdrawn".to_string(),
            label: "SYNTHETIC / DEV · COLD HCP GATE FAILED (WITHDRAWN)".to_string(),
            detail: "Synthetic Ubiquitin (--realistic) 98.4% (310/315); dev CISD3 (BMRB 51439) 64.38% acc / 100% (59/59) HC; default evo-graph unseen cold micro HC-prec 88.25-95.30% (latest single-shot cold cohort 88.25% FAILS >= 94% gate; cold HCP claim formally withdrawn under 13.1:P1-2).".to_string(),
        },
        duration_sec: TOUR_DURATION_SEC,
        fps: 30,
        cli_command: "ferric assign --star data/fixtures/synthetic_ubiquitin.str --solver evo-graph".to_string(),
        chapters: vec![
            TourChapter {
                index: 1,
                start_sec: 0.0,
                end_sec: 4.0,
                title: "2D 1H-15N HSQC & Root Picking".to_string(),
                caption: "Ubiquitin (76 aa, 600 MHz) positive backbone/NH2 contours + aliased Arg Ne-He negative contours.".to_string(),
                metric_pill: format!("{} HSQC roots picked", hsqc_roots.len()),
            },
            TourChapter {
                index: 2,
                start_sec: 4.0,
                end_sec: 8.0,
                title: "3D HNCACB Sequential Strip Walk".to_string(),
                caption: "Walks Phe4 -> Gly10: CA(i)->CA(i-1), Thr7 CB=69.5 > CA inversion, and Gly10 CB-null anchor.".to_string(),
                metric_pill: "Phe4 -> Gly10 sequential walk".to_string(),
            },
            TourChapter {
                index: 3,
                start_sec: 8.0,
                end_sec: 12.0,
                title: "Solver Cv(T) & Benchmark Range".to_string(),
                caption: "bayes-sa Cv(T) alongside synthetic UBQ (98.4%), dev CISD3 (64.38%/100% HC), and cold BMRB evo-graph 88.25-95.30% (FAILS >=94%; withdrawn).".to_string(),
                metric_pill: format!("T_crit={:.2}, max Cv={:.1}", bayes_rep.t_crit, bayes_rep.max_cv),
            },
            TourChapter {
                index: 4,
                start_sec: 12.0,
                end_sec: 16.0,
                title: "76-Residue QC Confidence Ribbon".to_string(),
                caption: "DatasetQcReport posterior confidence across residues 1..76 (>= 0.80 calibrated HC tier).".to_string(),
                metric_pill: format!("QC={qc_verdict} · {n_high_conf}/76 >= 0.80"),
            },
        ],
        interactive_control: InteractiveControlSpec {
            id: "focus_res".to_string(),
            label: "Focus Strip Residue".to_string(),
            min: 1.0,
            max: 10.0,
            step: 1.0,
            default_val: 7.0,
            unit: "res".to_string(),
        },
        payload: json!({
            "hsqc_pos_contours": hsqc_pos_contours,
            "hsqc_neg_contours": hsqc_neg_contours,
            "hsqc_roots": hsqc_roots.iter().map(|(h, n, l)| json!({"h": round4(*h), "n": round4(*n), "label": l})).collect::<Vec<_>>(),
            "strips": strips.iter().map(|s| json!({
                "seq_id": s.seq_id,
                "label": s.label,
                "h_ppm": round4(s.h_ppm),
                "n_ppm": round4(s.n_ppm),
                "ca_i": s.ca_i.map(round4),
                "cb_i": s.cb_i.map(round4),
                "ca_im1": s.ca_im1.map(round4),
                "cb_im1": s.cb_im1.map(round4),
                "pos_pts": s.pos_pts.iter().map(|p| [round4(p[0]), round4(p[1])]).collect::<Vec<_>>(),
                "neg_pts": s.neg_pts.iter().map(|p| [round4(p[0]), round4(p[1])]).collect::<Vec<_>>(),
            })).collect::<Vec<_>>(),
            "cv_temps": cv_temps,
            "cv_vals": cv_vals,
            "mean_energies": mean_energies,
            "t_crit": round4(bayes_rep.t_crit),
            "max_cv": round4(bayes_rep.max_cv),
            "res_confidences": round_vec(&res_confidences, 4),
            "res_labels": res_labels,
            "qc_verdict": qc_verdict,
            "n_high_conf": n_high_conf,
            "benchmarks": {
                "ubiquitin_realistic_pct": 98.4,
                "cisd3_acc_pct": 64.38,
                "cisd3_hc_prec_pct": 100.0,
                "cold_bmrb_hc_min_pct": 86.18,
                "cold_bmrb_hc_max_pct": 96.31,
            }
        }),
    };

    Ok(CoreTourState {
        bundle,
        scene: CoreSceneData::M02HsqcStripsSolvers(Box::new(M02SceneData {
            hsqc_pos_contours,
            hsqc_neg_contours,
            hsqc_roots,
            strips,
            cv_vals,
            t_crit: bayes_rep.t_crit,
            max_cv: bayes_rep.max_cv,
            res_confidences,
            evo_acc_pct,
            qc_verdict,
            n_high_conf,
        })),
    })
}

fn build_m03_state(seed: u64) -> Result<CoreTourState, String> {
    let chain = parse_pdb_str_first_model(UBIQUITIN_1UBQ_PDB_FIXTURE)?;
    let ref_data = read_star_str(UBIQUITIN_STAR_FIXTURE, None)?;

    // Center 1UBQ coordinates
    let n_res = chain.residues.len().max(1);
    let mut cen = [0.0_f64; 3];
    for r in &chain.residues {
        cen[0] += r.ca[0];
        cen[1] += r.ca[1];
        cen[2] += r.ca[2];
    }
    cen[0] /= n_res as f64;
    cen[1] /= n_res as f64;
    cen[2] /= n_res as f64;
    let ca_coords: Vec<[f64; 3]> = chain
        .residues
        .iter()
        .map(|r| [r.ca[0] - cen[0], r.ca[1] - cen[1], r.ca[2] - cen[2]])
        .collect();

    // Secondary structure from real chain torsions
    let torsions = compute_chain_torsions(&chain);
    let ss_classes: Vec<u8> = torsions
        .iter()
        .map(|(_, _, phi_opt, psi_opt)| match (*phi_opt, *psi_opt) {
            (Some(phi), Some(psi))
                if (-95.0..=-35.0).contains(&phi) && (-70.0..=-15.0).contains(&psi) =>
            {
                1 // helix
            }
            (Some(phi), Some(psi))
                if (-165.0..=-75.0).contains(&phi) && (85.0..=175.0).contains(&psi) =>
            {
                2 // sheet
            }
            _ => 0,
        })
        .collect();

    // Real homolog-firewalled torsion prediction (<= 25% sequence identity)
    let fw_query = HomologFirewallQuery {
        query_pdb_ids: vec!["1UBQ".to_string(), "1D3Z".to_string()],
        query_bmrb_ids: vec!["17769".to_string(), "6457".to_string()],
        query_uniprot_id: Some("P0CG48".to_string()),
        query_family_id: None,
        query_topology_id: None,
        query_ca_coords: Some(chain.residues.iter().map(|r| r.ca).collect()),
    };
    let torsion_rep = predict_torsions_with_firewall(&ref_data, &fw_query);
    let n_torsions_firewalled = torsion_rep
        .residues
        .iter()
        .filter(|r| r.phi_deg.is_some())
        .count();
    let n_excluded_templates = torsion_rep.excluded_template_ids.len();
    let mut s2_vals = vec![0.86_f64; chain.residues.len()];
    for (idx, r) in chain.residues.iter().enumerate() {
        if let Some(tr) = torsion_rep.residues.iter().find(|t| t.seq_id == r.seq_id) {
            s2_vals[idx] = tr.rci_s2.clamp(0.25, 0.95);
        }
        if r.seq_id >= 72 {
            let tail = (r.seq_id - 71) as f64;
            s2_vals[idx] = (0.70 - 0.095 * tail).clamp(0.25, 0.72);
        }
    }

    // Real experimental 1D3Z NOE parsing & long-range (|i-j| >= 5) contact extraction
    let seq_names: Vec<String> = chain.residues.iter().map(|r| r.res_name.clone()).collect();
    let seq_ids: Vec<i32> = chain.residues.iter().map(|r| r.seq_id).collect();
    let adrs = parse_exp_noe_tsv_str(UBIQUITIN_1D3Z_NOE_FIXTURE, &seq_names, &seq_ids)?;
    let mut noe_pairs: Vec<(usize, usize, f64)> = Vec::new();
    let mut seen_pairs = std::collections::BTreeSet::new();
    for adr in &adrs {
        if let Some(cand) = adr.candidates.first() {
            let a = cand.res_a.min(cand.res_b);
            let b = cand.res_a.max(cand.res_b);
            if b.saturating_sub(a) >= 5 && adr.upper_bound <= 4.6 && seen_pairs.insert((a, b)) {
                noe_pairs.push((a, b, adr.upper_bound));
                if noe_pairs.len() >= 48 {
                    break;
                }
            }
        }
    }

    // Real experimental 1D3Z RDC parsing, ByPeptidePlane 20% free-set split & Saupe evaluation
    let mut media = parse_rdc_tsv_str(UBIQUITIN_1D3Z_RDC_FIXTURE)?;
    partition_rdc_free_set(&mut media, 0.20, seed);
    let rdc_val = evaluate_rdc_validation(&chain, &media)?;
    let first_med = rdc_val
        .media
        .first()
        .ok_or_else(|| "missing fitted RDC medium".to_string())?;
    let rdc_points: Vec<(f64, f64, bool, i32)> = rdc_val
        .points
        .iter()
        .filter(|p| p.medium_id == first_med.medium_id)
        .map(|p| (p.d_obs_norm_nh_hz, p.d_calc_norm_nh_hz, p.is_free, p.seq_id))
        .collect();

    let bundle = TourBundle {
        schema_version: TOUR_SCHEMA_VERSION.to_string(),
        module_index: 3,
        slug: TourModuleId::M03StructureNoeRdc.slug().to_string(),
        title: "3D Structure Folding, 1D3Z NOEs & Saupe RDCs".to_string(),
        subtitle: "Dev 1D3Z/1UBQ · F8 unseen 1KHM->1ZZK improves NOE-only 5/5 (1.86-4.14 A), 0/5 < 1.80 A".to_string(),
        manual_chapter: "01-quickstart-and-workflows.html".to_string(),
        validation_badge: ValidationBadge {
            kind: "cold_holdout_disclosed".to_string(),
            label: "DEV 1D3Z + UNSEEN 1KHM HOLDOUT DISCLOSED (F8)".to_string(),
            detail: "F8 disclosed: unseen cold 1KHM -> 1ZZK improves NOE-only on 5/5 seeds (1.86-4.14 A), 0/5 < 1.80 A. Dev benchmark uses 1D3Z / 1UBQ.".to_string(),
        },
        duration_sec: TOUR_DURATION_SEC,
        fps: 30,
        cli_command: "ferric structure --star data/fixtures/synthetic_ubiquitin.str --noesy-peaks data/fixtures/ubiquitin_1d3z_exp_noe.tsv --rdc data/fixtures/ubiquitin_1d3z_rdc.tsv --torsion-db".to_string(),
        chapters: vec![
            TourChapter {
                index: 1,
                start_sec: 0.0,
                end_sec: 4.0,
                title: "1UBQ 3D Backbone & Secondary Structure".to_string(),
                caption: "Depth-sorted CA trace colored by alpha-helix, beta-sheet, and loop torsions.".to_string(),
                metric_pill: format!("{} residues (PDB 1UBQ)", ca_coords.len()),
            },
            TourChapter {
                index: 2,
                start_sec: 4.0,
                end_sec: 8.0,
                title: "1D3Z NOE Restraints & <=25% Firewall".to_string(),
                caption: "Long-range |i-j| >= 5 proton NOE distance cylinders + homolog-firewalled (<=25% id) torsions.".to_string(),
                metric_pill: format!("{} long-range NOEs · {} excluded", noe_pairs.len(), n_excluded_templates),
            },
            TourChapter {
                index: 3,
                start_sec: 8.0,
                end_sec: 12.0,
                title: "Saupe Tensor & Q_free + F8 Holdout".to_string(),
                caption: "ByPeptidePlane Q_free RDC correlation on 1D3Z; F8 discloses unseen 1KHM->1ZZK (1.86-4.14 A, 0/5 < 1.80 A).".to_string(),
                metric_pill: format!("Q_work={:.3}  Q_free={:.3}", first_med.q_work, first_med.q_free),
            },
            TourChapter {
                index: 4,
                start_sec: 12.0,
                end_sec: 16.0,
                title: "360 deg S^2 Dynamics Putty Orbit".to_string(),
                caption: "Continuous 360 deg 3D orbit scaling tube radius by internal flexibility (1 - S^2).".to_string(),
                metric_pill: format!("{n_torsions_firewalled} firewalled torsions"),
            },
        ],
        interactive_control: InteractiveControlSpec {
            id: "orbit_deg".to_string(),
            label: "3D Orbit Azimuth".to_string(),
            min: 0.0,
            max: 360.0,
            step: 5.0,
            default_val: 45.0,
            unit: "deg".to_string(),
        },
        payload: json!({
            "ca_coords": ca_coords.iter().map(|c| [round4(c[0]), round4(c[1]), round4(c[2])]).collect::<Vec<_>>(),
            "ss_classes": ss_classes,
            "s2_vals": round_vec(&s2_vals, 4),
            "noe_pairs": noe_pairs.iter().map(|(a, b, d)| json!({"a": a, "b": b, "upper_a": round4(*d)})).collect::<Vec<_>>(),
            "rdc_points": rdc_points.iter().map(|(o, c, f, s)| json!({"d_obs": round4(*o), "d_calc": round4(*c), "is_free": f, "seq_id": s})).collect::<Vec<_>>(),
            "q_work": round4(first_med.q_work),
            "q_free": round4(first_med.q_free),
            "da_nh_hz": round4(first_med.da_nh_hz),
            "rhombicity": round4(first_med.rhombicity),
            "f8_holdout_note": "F8: unseen 1KHM -> 1ZZK improves NOE-only on 5/5 seeds (1.86-4.14 A), 0/5 < 1.80 A",
        }),
    };

    Ok(CoreTourState {
        bundle,
        scene: CoreSceneData::M03StructureNoeRdc(Box::new(M03SceneData {
            ca_coords,
            ss_classes,
            s2_vals,
            noe_pairs,
            rdc_points,
            q_work: first_med.q_work,
            q_free: first_med.q_free,
            da_nh_hz: first_med.da_nh_hz,
            rhombicity: first_med.rhombicity,
            n_torsions_firewalled,
            n_excluded_templates,
        })),
    })
}

fn build_m04_state(seed: u64) -> Result<CoreTourState, String> {
    let chain = parse_pdb_str_first_model(UBIQUITIN_1UBQ_PDB_FIXTURE)?;
    let nh_vecs = extract_nh_unit_vectors_from_chain(&chain);
    let nh_map: BTreeMap<i32, [f64; 3]> = nh_vecs.iter().copied().collect();
    let phys = RelaxPhysicsConfig::default();
    let mut rng = Rng::new(seed);

    // 1. Ch 1: Locked-footprint 2D plane linear LS vs naive local-max bias across SNR 80 -> 4
    let snr_levels = vec![80.0, 40.0, 20.0, 10.0, 6.0, 4.0];
    let mut locked_ls_bias_pct = Vec::with_capacity(snr_levels.len());
    let mut local_max_bias_pct = Vec::with_capacity(snr_levels.len());
    let true_amp = 100.0_f64;
    let mut footprint = Vec::with_capacity(25);
    let mut f_norm2 = 0.0_f64;
    for dy in -2_i32..=2 {
        for dx in -2_i32..=2 {
            let v = (-(f64::from(dx * dx + dy * dy)) / 2.2).exp();
            f_norm2 += v * v;
            footprint.push(v);
        }
    }
    for &snr in &snr_levels {
        let sigma = true_amp / snr;
        let mut ls_sum = 0.0;
        let mut max_sum = 0.0;
        let trials = 40usize;
        for _ in 0..trials {
            let mut dot = 0.0;
            let mut mx = f64::NEG_INFINITY;
            for &fv in &footprint {
                let y = true_amp * fv + rng.normal(0.0, sigma);
                dot += fv * y;
                mx = mx.max(y);
            }
            ls_sum += dot / f_norm2;
            max_sum += mx;
        }
        let ls_mean = ls_sum / (trials as f64);
        let max_mean = max_sum / (trials as f64);
        locked_ls_bias_pct.push(round4(100.0 * (ls_mean - true_amp) / true_amp));
        local_max_bias_pct.push(round4(100.0 * (max_mean - true_amp) / true_amp));
    }

    // 2. Ch 2: Dual-field 600/800 MHz exponential decay fit + MC bootstrap
    let times_r1 = vec![0.02, 0.08, 0.16, 0.30, 0.50, 0.75, 1.05, 1.40];
    let true_r1_600 = 2.15_f64;
    let true_r1_800 = 1.62_f64;
    let curve_600_r1: Vec<f64> = times_r1
        .iter()
        .map(|&t| (100.0 * (-true_r1_600 * t).exp() + rng.normal(0.0, 1.6)).max(1.0))
        .collect();
    let curve_800_r1: Vec<f64> = times_r1
        .iter()
        .map(|&t| (100.0 * (-true_r1_800 * t).exp() + rng.normal(0.0, 1.4)).max(1.0))
        .collect();
    let fit_cfg = ExpFitConfig {
        offset_gate: OffsetModelGate::Disabled,
        mc_replicates: 100,
        seed,
        ..ExpFitConfig::default()
    };
    let fit_600 = fit_decay_curve_cpu(
        &DecayCurveInput {
            seq_id: 23,
            res_name: "ILE".to_string(),
            times_sec: times_r1.clone(),
            intensities: curve_600_r1.clone(),
            sigmas: vec![1.6; times_r1.len()],
        },
        &fit_cfg,
    )?;
    let fit_800 = fit_decay_curve_cpu(
        &DecayCurveInput {
            seq_id: 23,
            res_name: "ILE".to_string(),
            times_sec: times_r1.clone(),
            intensities: curve_800_r1.clone(),
            sigmas: vec![1.4; times_r1.len()],
        },
        &fit_cfg,
    )?;

    // 3 & 4. Ch 3/4: Ubiquitin synthetic dual-field relaxation -> trimmed tau_m, Woessner axial, & M1-M5 S^2
    let tau_m_true = 4.10e-9_f64;
    let rot_comps = [(1.0_f64, tau_m_true)];
    let mut mf_inputs = Vec::new();
    let mut rates_600 = Vec::new();

    for res in &chain.residues {
        if res.res_name == "PRO" || res.seq_id == 1 {
            continue;
        }
        let s_id = res.seq_id;
        let (true_s2, true_te, true_rex) = if s_id >= 72 {
            let tail_step = (s_id - 71) as f64;
            ((0.72 - 0.095 * tail_step).clamp(0.24, 0.70), 85.0e-12, 0.0)
        } else if (8..=11).contains(&s_id) {
            (0.68, 45.0e-12, 0.0)
        } else if s_id == 23 || s_id == 25 {
            (0.84, 15.0e-12, 3.6)
        } else if (36..=40).contains(&s_id) || (62..=65).contains(&s_id) {
            (0.77, 25.0e-12, 0.0)
        } else {
            (0.87, 12.0e-12, 0.0)
        };
        let local = if true_rex > 0.0 {
            LocalMfParams::m3(true_s2, true_rex)
        } else {
            LocalMfParams::m2(true_s2, true_te)
        };

        let mut fields = Vec::with_capacity(2);
        for &mhz in &[600.0_f64, 800.0_f64] {
            let (r1_0, r2_0, noe_0) = predict_r1_r2_noe(&rot_comps, &local, mhz, 600.0, &phys);
            let r1 = (r1_0 * (1.0 + rng.normal(0.0, 0.008))).max(0.1);
            let r2 = (r2_0 * (1.0 + rng.normal(0.0, 0.008))).max(0.2);
            let noe = noe_0 + rng.normal(0.0, 0.010);
            let rr = ResidueRelaxRates {
                seq_id: s_id,
                res_name: res.res_name.clone(),
                field_mhz: mhz,
                r1,
                sigma_r1: 0.018 * r1_0,
                r2,
                sigma_r2: 0.020 * r2_0,
                hetnoe: Some(noe),
                sigma_hetnoe: Some(0.018),
            };
            if (mhz - 600.0).abs() < 1.0 {
                rates_600.push(rr.clone());
            }
            fields.push(rr);
        }
        mf_inputs.push(MultiFieldResidueInput {
            seq_id: s_id,
            res_name: res.res_name.clone(),
            nh_unit_vec: nh_map.get(&s_id).copied(),
            fields,
        });
    }

    let trimmed = estimate_trimmed_tau_m(&rates_600, &TrimmedTauMConfig::default())?;
    let axial = estimate_initial_axial_diffusion_from_r2_r1(
        &rates_600,
        &nh_vecs,
        &TrimmedTauMConfig::default(),
    )?;
    let mf_fit = fit_modelfree_dataset(
        &mf_inputs,
        &ModelFreeConfig {
            diffusion_kind: DiffusionTensorKind::Isotropic,
            selection_criterion: MfCriterion::Aic,
            ref_field_mhz: Some(600.0),
            physics: phys,
            enable_model_elimination: true,
            max_outer_iters: 4,
            mc_replicates: 0,
            seed,
        },
    )?;

    let seq_ids: Vec<i32> = mf_fit.residues.iter().map(|r| r.seq_id).collect();
    let r2_over_r1: Vec<f64> = rates_600
        .iter()
        .map(|r| round4(r.r2 / r.r1.max(1.0e-6)))
        .collect();
    let s2_vals: Vec<f64> = mf_fit
        .residues
        .iter()
        .map(|r| round4(r.params.s2))
        .collect();
    let rex_vals: Vec<f64> = mf_fit
        .residues
        .iter()
        .map(|r| round4(r.params.rex_ref_s_inv))
        .collect();
    let model_ids: Vec<String> = mf_fit
        .residues
        .iter()
        .map(|r| format!("{:?}", r.selected_model))
        .collect();

    let tau_m_ns = trimmed.tau_m_sec * 1.0e9;
    let d_par_perp = axial.anisotropy_ratio;

    let bundle = TourBundle {
        schema_version: TOUR_SCHEMA_VERSION.to_string(),
        module_index: 4,
        slug: TourModuleId::M04RelaxModelfree.slug().to_string(),
        title: "15N Relaxation & Lipari-Szabo Model-Free (M1-M5)".to_string(),
        subtitle: "Synthetic Ubiquitin 600/800 MHz forward-model showcase · Phase 17 F10/F11 blind gates open".to_string(),
        manual_chapter: "04-dynamics-and-exchange.html".to_string(),
        validation_badge: ValidationBadge {
            kind: "synthetic_showcase".to_string(),
            label: "SYNTHETIC FORWARD-MODEL SHOWCASE (F10/F11 OPEN)".to_string(),
            detail: "Synthetic forward-model showcase: 0/8 Phase 17 blind real-data gates passed (F10 relax / F11 modelfree remain open in TODO.md).".to_string(),
        },
        duration_sec: TOUR_DURATION_SEC,
        fps: 30,
        cli_command: "ferric modelfree --r1 r1.tsv --r2 r2.tsv --hetnoe noe.tsv --pdb 1ubq.pdb --diffusion axial".to_string(),
        chapters: vec![
            TourChapter {
                index: 1,
                start_sec: 0.0,
                end_sec: 4.0,
                title: "Locked-Footprint Plane Integration".to_string(),
                caption: "Linear LS fixed-footprint plane integration stays unbiased at SNR=4 vs naive local-max bias.".to_string(),
                metric_pill: format!(
                    "SNR=4 bias: LS {:.1}% vs Max +{:.1}%",
                    locked_ls_bias_pct.last().copied().unwrap_or(0.0),
                    local_max_bias_pct.last().copied().unwrap_or(0.0)
                ),
            },
            TourChapter {
                index: 2,
                start_sec: 4.0,
                end_sec: 8.0,
                title: "Dual-Field R1/R2/hetNOE + MC Fit".to_string(),
                caption: "600 & 800 MHz VarPro+LM exponential fits with 100-trial parametric Monte Carlo SE.".to_string(),
                metric_pill: format!(
                    "600 R1={:.2}+/-{:.2}  800 R1={:.2}+/-{:.2}",
                    fit_600.rate, fit_600.sigma_rate_mc, fit_800.rate, fit_800.sigma_rate_mc
                ),
            },
            TourChapter {
                index: 3,
                start_sec: 8.0,
                end_sec: 12.0,
                title: "Kay-Torchia-Bax tau_m & Woessner".to_string(),
                caption: "Outlier-trimmed R2/R1 global tau_m & Woessner axial D_par/D_perp on 1UBQ N-H vectors.".to_string(),
                metric_pill: format!("tau_m={tau_m_ns:.2} ns  D_par/D_perp={d_par_perp:.2}"),
            },
            TourChapter {
                index: 4,
                start_sec: 12.0,
                end_sec: 16.0,
                title: "Lipari-Szabo M1-M5 S^2 Profile".to_string(),
                caption: "Ubiquitin 1..76 S^2 profile: rigid core (~0.87), Ile23 R_ex (M3), and flexible C-tail 72..76.".to_string(),
                metric_pill: "SYNTHETIC SHOWCASE (F10/F11 OPEN)".to_string(),
            },
        ],
        interactive_control: InteractiveControlSpec {
            id: "residue_seq".to_string(),
            label: "Inspect Residue".to_string(),
            min: 1.0,
            max: 76.0,
            step: 1.0,
            default_val: 23.0,
            unit: "seq".to_string(),
        },
        payload: json!({
            "snr_levels": snr_levels,
            "locked_ls_bias_pct": locked_ls_bias_pct,
            "local_max_bias_pct": local_max_bias_pct,
            "times_r1": times_r1,
            "curve_600_r1": round_vec(&curve_600_r1, 4),
            "curve_800_r1": round_vec(&curve_800_r1, 4),
            "fit_600_rate": round4(fit_600.rate),
            "fit_600_se": round4(fit_600.sigma_rate_mc),
            "fit_800_rate": round4(fit_800.rate),
            "fit_800_se": round4(fit_800.sigma_rate_mc),
            "tau_m_ns": round4(tau_m_ns),
            "d_par_perp": round4(d_par_perp),
            "seq_ids": seq_ids,
            "r2_over_r1": r2_over_r1,
            "s2_vals": s2_vals,
            "rex_vals": rex_vals,
            "model_ids": model_ids,
        }),
    };

    Ok(CoreTourState {
        bundle,
        scene: CoreSceneData::M04RelaxModelfree(Box::new(M04SceneData {
            snr_levels,
            locked_ls_bias_pct,
            local_max_bias_pct,
            times_r1,
            curve_600_r1,
            curve_800_r1,
            fit_600_rate: fit_600.rate,
            fit_600_se: fit_600.sigma_rate_mc,
            fit_800_rate: fit_800.rate,
            fit_800_se: fit_800.sigma_rate_mc,
            tau_m_ns,
            d_par_perp,
            seq_ids,
            r2_over_r1,
            s2_vals,
            rex_vals,
        })),
    })
}

fn round4(v: f64) -> f64 {
    if !v.is_finite() {
        0.0
    } else {
        (v * 10_000.0).round() / 10_000.0
    }
}

fn round_vec(v: &[f64], decimals: i32) -> Vec<f64> {
    let scale = 10.0_f64.powi(decimals);
    v.iter()
        .map(|&x| {
            if x.is_finite() {
                (x * scale).round() / scale
            } else {
                0.0
            }
        })
        .collect()
}

/// Extract compact marching-squares contour polylines in `(h_ppm, n_ppm)` space.
#[allow(clippy::too_many_arguments)]
fn extract_contour_polylines(
    grid: &[f64],
    nx: usize,
    ny: usize,
    level: f64,
    x0_val: f64,
    x1_val: f64,
    y0_val: f64,
    y1_val: f64,
) -> Vec<Vec<[f64; 2]>> {
    if nx < 2 || ny < 2 || grid.len() < nx * ny {
        return Vec::new();
    }
    let sx = (x1_val - x0_val) / ((nx - 1) as f64);
    let sy = (y1_val - y0_val) / ((ny - 1) as f64);
    let interp = |va: f64, vb: f64| -> f64 {
        let d = vb - va;
        if d.abs() < 1.0e-12 {
            0.5
        } else {
            ((level - va) / d).clamp(0.0, 1.0)
        }
    };
    let mut segs: Vec<([f64; 2], [f64; 2])> = Vec::new();
    for iy in 0..(ny - 1) {
        let r0 = iy * nx;
        let r1 = (iy + 1) * nx;
        let yt = y0_val + (iy as f64) * sy;
        let yb = y0_val + ((iy + 1) as f64) * sy;
        for ix in 0..(nx - 1) {
            let v00 = grid[r0 + ix];
            let v10 = grid[r0 + ix + 1];
            let v11 = grid[r1 + ix + 1];
            let v01 = grid[r1 + ix];
            let mut m = 0u8;
            if v00 >= level {
                m |= 1;
            }
            if v10 >= level {
                m |= 2;
            }
            if v11 >= level {
                m |= 4;
            }
            if v01 >= level {
                m |= 8;
            }
            if m == 0 || m == 15 {
                continue;
            }
            let xl = x0_val + (ix as f64) * sx;
            let xr = x0_val + ((ix + 1) as f64) * sx;
            let e = |edge: u8| -> [f64; 2] {
                match edge {
                    0 => [round4(xl + interp(v00, v10) * sx), round4(yt)],
                    1 => [round4(xr), round4(yt + interp(v10, v11) * sy)],
                    2 => [round4(xl + interp(v01, v11) * sx), round4(yb)],
                    _ => [round4(xl), round4(yt + interp(v00, v01) * sy)],
                }
            };
            match m {
                1 | 14 => segs.push((e(3), e(0))),
                2 | 13 => segs.push((e(0), e(1))),
                3 | 12 => segs.push((e(3), e(1))),
                4 | 11 => segs.push((e(1), e(2))),
                6 | 9 => segs.push((e(0), e(2))),
                7 | 8 => segs.push((e(3), e(2))),
                5 | 10 => {
                    segs.push((e(3), e(0)));
                    segs.push((e(1), e(2)));
                }
                _ => {}
            }
        }
    }
    segs.into_iter().map(|(a, b)| vec![a, b]).collect()
}

// ============================================================================
// Animated Frame Renderers for Core Modules (`01..=04`)
// ============================================================================

fn render_core_tour_frame(
    state: &CoreTourState,
    t_sec: f64,
    width: usize,
    height: usize,
) -> Vec<u8> {
    let mut canvas = FrameCanvas::new(width, height);
    canvas.clear_gradient([10, 15, 29], [17, 24, 39]);

    let t = t_sec.clamp(0.0, state.bundle.duration_sec);
    let ch_idx = active_chapter_index(&state.bundle, t);
    let local_t = ((t - 4.0 * (ch_idx as f64)) / 4.0).clamp(0.0, 1.0);

    match &state.scene {
        CoreSceneData::M01FidPulprogNus(d) => render_m01_frame(&mut canvas, d, ch_idx, local_t, t),
        CoreSceneData::M02HsqcStripsSolvers(d) => {
            render_m02_frame(&mut canvas, d, ch_idx, local_t, t)
        }
        CoreSceneData::M03StructureNoeRdc(d) => {
            render_m03_frame(&mut canvas, d, ch_idx, local_t, t)
        }
        CoreSceneData::M04RelaxModelfree(d) => render_m04_frame(&mut canvas, d, ch_idx, local_t, t),
    }

    canvas.draw_tour_chrome(&state.bundle, t);
    canvas.rgb
}

fn render_m01_frame(
    c: &mut FrameCanvas,
    d: &M01SceneData,
    ch_idx: usize,
    local_t: f64,
    t_sec: f64,
) {
    let w = c.width as f64;
    let h = c.height as f64;
    let sx = w / 1280.0;
    let sy = h / 720.0;

    // Main stage card
    c.fill_rounded_rect(
        22.0 * sx,
        84.0 * sy,
        1236.0 * sx,
        540.0 * sy,
        12.0,
        [17, 25, 44, 240],
    );
    c.stroke_rounded_rect(
        22.0 * sx,
        84.0 * sy,
        1236.0 * sx,
        540.0 * sy,
        12.0,
        1.4,
        [51, 65, 85, 255],
    );

    match ch_idx {
        0 => {
            // Ch 1: Sandboxed Bruker Pulseprogram AST & Echo-Antiecho vs States-TPPI
            c.fill_rounded_rect(
                44.0 * sx,
                106.0 * sy,
                580.0 * sx,
                496.0 * sy,
                10.0,
                [15, 23, 42, 255],
            );
            c.stroke_rounded_rect(
                44.0 * sx,
                106.0 * sy,
                580.0 * sx,
                496.0 * sy,
                10.0,
                1.2,
                [56, 189, 248, 190],
            );
            c.draw_text(
                64.0 * sx,
                122.0 * sy,
                "BRUKER PULSEPROGRAM AST SCAN",
                1.7 * sx,
                [125, 211, 252, 255],
            );

            let ast_lines = [
                ("1 ze / 2 d1", "recycle delay loop"),
                ("(p1 ph1):f1 -> (p3 ph2):f2", "1H -> 15N INEPT transfer"),
                ("igrad EA", "Rance-Kay Echo-Antiecho gradient"),
                ("F1EA(igrad EA, id0)", &d.f1_quad),
                ("F2PH(ip3, id10)", &d.f2_quad),
                (
                    "AQSEQ / Transfer",
                    &format!("{} | {}", d.aqseq, d.transfer_summary),
                ),
            ];
            let active_row = ((local_t * 6.0).floor() as usize).min(ast_lines.len() - 1);
            for (i, (lhs, rhs)) in ast_lines.iter().enumerate() {
                let ry = (164.0 + (i as f64) * 68.0) * sy;
                let hi = i == active_row;
                let bg = if hi {
                    [14, 116, 144, 210]
                } else {
                    [30, 41, 59, 220]
                };
                let br = if hi {
                    [56, 189, 248, 255]
                } else {
                    [71, 85, 105, 200]
                };
                c.fill_rounded_rect(64.0 * sx, ry, 540.0 * sx, 54.0 * sy, 8.0, bg);
                c.stroke_rounded_rect(64.0 * sx, ry, 540.0 * sx, 54.0 * sy, 8.0, 1.2, br);
                c.draw_text(
                    80.0 * sx,
                    ry + 11.0 * sy,
                    lhs,
                    1.45 * sx,
                    [248, 250, 252, 255],
                );
                c.draw_text(
                    80.0 * sx,
                    ry + 32.0 * sy,
                    rhs,
                    1.25 * sx,
                    [186, 230, 253, 255],
                );
            }

            // Right card: Magnetization transfer & Rance-Kay vs States-TPPI comparison
            c.fill_rounded_rect(
                646.0 * sx,
                106.0 * sy,
                590.0 * sx,
                496.0 * sy,
                10.0,
                [15, 23, 42, 255],
            );
            c.stroke_rounded_rect(
                646.0 * sx,
                106.0 * sy,
                590.0 * sx,
                496.0 * sy,
                10.0,
                1.2,
                [52, 211, 153, 190],
            );
            c.draw_text(
                668.0 * sx,
                122.0 * sy,
                "QUADRATURE & TRANSFER GRAPH",
                1.7 * sx,
                [110, 231, 183, 255],
            );

            let nodes = [
                (730.0 * sx, 235.0 * sy, "1H", [56, 189, 248, 255]),
                (920.0 * sx, 235.0 * sy, "15N", [52, 211, 153, 255]),
                (1110.0 * sx, 235.0 * sy, "13CA/CB", [251, 191, 36, 255]),
            ];
            c.draw_line(
                765.0 * sx,
                235.0 * sy,
                885.0 * sx,
                235.0 * sy,
                3.0,
                [94, 234, 212, 220],
            );
            c.draw_line(
                955.0 * sx,
                235.0 * sy,
                1070.0 * sx,
                235.0 * sy,
                3.0,
                [253, 224, 71, 220],
            );
            for (nx, ny, lbl, col) in nodes {
                c.fill_circle(nx, ny, 34.0 * sx, [30, 41, 59, 255]);
                c.stroke_circle(nx, ny, 34.0 * sx, 2.5, col);
                c.draw_text(
                    nx - c.measure_text_width(lbl, 1.6 * sx) * 0.5,
                    ny - 6.0 * sy,
                    lbl,
                    1.6 * sx,
                    col,
                );
            }

            c.fill_rounded_rect(
                672.0 * sx,
                330.0 * sy,
                538.0 * sx,
                110.0 * sy,
                8.0,
                [30, 41, 59, 230],
            );
            c.draw_text(
                690.0 * sx,
                348.0 * sy,
                "Echo-Antiecho (igrad EA): S+ / S- recombination",
                1.45 * sx,
                [248, 250, 252, 255],
            );
            c.draw_text(
                690.0 * sx,
                376.0 * sy,
                "P = (S+ + S-),  N = (S+ - S-)  -> pure absorption",
                1.35 * sx,
                [148, 163, 184, 255],
            );
            c.draw_text(
                690.0 * sx,
                404.0 * sy,
                "States-TPPI (F2): alternating (-1)^k shifts axial to edge",
                1.35 * sx,
                [167, 243, 208, 255],
            );

            c.fill_rounded_rect(
                672.0 * sx,
                464.0 * sy,
                538.0 * sx,
                110.0 * sy,
                8.0,
                [6, 78, 59, 175],
            );
            c.stroke_rounded_rect(
                672.0 * sx,
                464.0 * sy,
                538.0 * sx,
                110.0 * sy,
                8.0,
                1.2,
                [52, 211, 153, 230],
            );
            c.draw_text(
                690.0 * sx,
                484.0 * sy,
                &format!("Program: {}", d.pulprog_name),
                1.5 * sx,
                [209, 250, 229, 255],
            );
            c.draw_text(
                690.0 * sx,
                514.0 * sy,
                "Sandbox guard: zero escaping #include paths",
                1.35 * sx,
                [167, 243, 208, 255],
            );

            let cursor_y = (190.0 + (active_row as f64) * 68.0) * sy;
            c.draw_spotlight_cursor(560.0 * sx, cursor_y, t_sec, 1.0);
        }
        1 => {
            // Ch 2: FID decay + sine-bell apodization -> FFT + Chen(2002) phase correction
            let rev = smoothstep(local_t);
            // Left: Time-domain FID & Sine-Bell window
            c.fill_rounded_rect(
                44.0 * sx,
                106.0 * sy,
                580.0 * sx,
                496.0 * sy,
                10.0,
                [15, 23, 42, 255],
            );
            c.stroke_rounded_rect(
                44.0 * sx,
                106.0 * sy,
                580.0 * sx,
                496.0 * sy,
                10.0,
                1.2,
                [56, 189, 248, 180],
            );
            c.draw_text(
                64.0 * sx,
                122.0 * sy,
                "TIME-DOMAIN FID + SINE-BELL WINDOW",
                1.65 * sx,
                [125, 211, 252, 255],
            );

            let px0 = 70.0 * sx;
            let px1 = 600.0 * sx;
            let py_mid = 365.0 * sy;
            c.draw_line(px0, py_mid, px1, py_mid, 1.2, [71, 85, 105, 220]);

            let n = d.fid_real.len();
            let show_n = ((rev * n as f64).ceil() as usize).clamp(2, n);
            let mut fid_pts = Vec::with_capacity(show_n);
            let mut win_pts = Vec::with_capacity(show_n);
            for i in 0..show_n {
                let x = lerp(px0, px1, (i as f64) / ((n - 1).max(1) as f64));
                let w_val = d.sine_window[i];
                let amp = d.fid_real[i] * lerp(1.0, w_val, rev);
                fid_pts.push([x, py_mid - amp * 62.0 * sy]);
                win_pts.push([x, py_mid - w_val * 155.0 * sy]);
            }
            c.draw_polyline(&win_pts, 2.0, [251, 191, 36, 220]);
            c.draw_polyline(&fid_pts, 2.2, [56, 189, 248, 255]);
            c.draw_text(
                74.0 * sx,
                160.0 * sy,
                "Amber: shifted sine-bell window W(t)",
                1.35 * sx,
                [251, 191, 36, 255],
            );
            c.draw_text(
                74.0 * sx,
                182.0 * sy,
                "Cyan: apodized FID Re{s(t)*W(t)}",
                1.35 * sx,
                [56, 189, 248, 255],
            );

            // Right: Unphased vs Phase-Corrected FFT spectrum
            c.fill_rounded_rect(
                646.0 * sx,
                106.0 * sy,
                590.0 * sx,
                496.0 * sy,
                10.0,
                [15, 23, 42, 255],
            );
            c.stroke_rounded_rect(
                646.0 * sx,
                106.0 * sy,
                590.0 * sx,
                496.0 * sy,
                10.0,
                1.2,
                [52, 211, 153, 180],
            );
            c.draw_text(
                668.0 * sx,
                122.0 * sy,
                "ZERO-FILL FFT & CHEN(2002) PHASE",
                1.65 * sx,
                [110, 231, 183, 255],
            );

            let sx0 = 674.0 * sx;
            let sx1 = 1210.0 * sx;
            let base_y = 515.0 * sy;
            c.draw_line(sx0, base_y, sx1, base_y, 1.3, [71, 85, 105, 230]);

            let m = d.fft_phased.len();
            let max_s = d
                .fft_phased
                .iter()
                .chain(d.fft_unphased.iter())
                .map(|v| v.abs())
                .fold(1.0e-6_f64, f64::max);
            let mut spec_pts = Vec::with_capacity(m);
            for i in 0..m {
                let x = lerp(sx0, sx1, (i as f64) / ((m - 1).max(1) as f64));
                let val = lerp(d.fft_unphased[i], d.fft_phased[i], rev);
                spec_pts.push([x, base_y - (val / max_s) * 290.0 * sy]);
            }
            c.draw_polyline(&spec_pts, 2.4, [52, 211, 153, 255]);
            c.draw_text(
                674.0 * sx,
                160.0 * sy,
                &format!(
                    "Auto-phase blend: {:.0}%  (P0={:.1} deg, P1={:.1} deg)",
                    rev * 100.0,
                    d.p0_deg,
                    d.p1_deg
                ),
                1.35 * sx,
                [167, 243, 208, 255],
            );
            c.draw_spotlight_cursor(lerp(sx0, sx1, 0.36), base_y - 185.0 * sy, t_sec, 1.0);
        }
        2 => {
            // Ch 3: 25% Poisson-Gap NUS sampling mask & zero-fill PSF artifacts
            c.fill_rounded_rect(
                44.0 * sx,
                106.0 * sy,
                1192.0 * sx,
                130.0 * sy,
                10.0,
                [15, 23, 42, 255],
            );
            c.stroke_rounded_rect(
                44.0 * sx,
                106.0 * sy,
                1192.0 * sx,
                130.0 * sy,
                10.0,
                1.2,
                [251, 191, 36, 190],
            );
            c.draw_text(
                64.0 * sx,
                120.0 * sy,
                "25% POISSON-GAP NUS SCHEDULE (16 / 64 INDIRECT SAMPLES)",
                1.65 * sx,
                [253, 224, 71, 255],
            );

            let bar_x0 = 66.0 * sx;
            let bar_w = (1146.0 * sx) / (d.nus_mask.len().max(1) as f64);
            let sweep = ((local_t * d.nus_mask.len() as f64).ceil() as usize).min(d.nus_mask.len());
            for (i, &sampled) in d.nus_mask.iter().enumerate() {
                let bx = bar_x0 + (i as f64) * bar_w;
                let col = if i >= sweep {
                    [30, 41, 59, 180]
                } else if sampled {
                    [52, 211, 153, 255]
                } else {
                    [51, 65, 85, 130]
                };
                c.fill_rounded_rect(
                    bx + 1.0,
                    158.0 * sy,
                    (bar_w - 2.0).max(2.0),
                    56.0 * sy,
                    2.0,
                    col,
                );
            }

            // Bottom panel: Zero-fill dirty spectrum vs true peaks showing PSF ridges
            c.fill_rounded_rect(
                44.0 * sx,
                254.0 * sy,
                1192.0 * sx,
                348.0 * sy,
                10.0,
                [15, 23, 42, 255],
            );
            c.stroke_rounded_rect(
                44.0 * sx,
                254.0 * sy,
                1192.0 * sx,
                348.0 * sy,
                10.0,
                1.2,
                [244, 63, 94, 190],
            );
            c.draw_text(
                64.0 * sx,
                270.0 * sy,
                "ZERO-FILL FFT POINT-SPREAD FUNCTION (PSF) ARTIFACT RIDGES",
                1.65 * sx,
                [253, 164, 175, 255],
            );

            let sx0 = 70.0 * sx;
            let sx1 = 1208.0 * sx;
            let base_y = 548.0 * sy;
            c.draw_line(sx0, base_y, sx1, base_y, 1.2, [71, 85, 105, 230]);
            let m = d.nus_dirty_spec.len();
            let max_d = d.nus_dirty_spec.iter().copied().fold(1.0e-6_f64, f64::max);
            let mut pts = Vec::with_capacity(m);
            for (i, &v) in d.nus_dirty_spec.iter().enumerate() {
                let x = lerp(sx0, sx1, (i as f64) / ((m - 1).max(1) as f64));
                pts.push([x, base_y - (v / max_d) * 235.0 * sy]);
            }
            c.draw_polyline(&pts, 2.3, [251, 113, 133, 255]);
            c.draw_text(
                64.0 * sx,
                298.0 * sy,
                "Strong PSF sidelobes mask the weak 0.58-ppm resonance before IST/FISTA",
                1.4 * sx,
                [226, 232, 240, 255],
            );
            c.draw_spotlight_cursor(lerp(sx0, sx1, 0.58), base_y - 75.0 * sy, t_sec, 1.15);
        }
        _ => {
            // Ch 4: Hypercomplex FISTA iteration 0 -> 20 suppressing ridges
            let iter_idx =
                ((local_t * 20.0).round() as usize).min(d.nus_fista_iters.len().saturating_sub(1));
            let cur_spec = &d.nus_fista_iters[iter_idx];
            let l1 = d.nus_residual_l1.get(iter_idx).copied().unwrap_or(0.0);

            c.fill_rounded_rect(
                44.0 * sx,
                106.0 * sy,
                840.0 * sx,
                496.0 * sy,
                10.0,
                [15, 23, 42, 255],
            );
            c.stroke_rounded_rect(
                44.0 * sx,
                106.0 * sy,
                840.0 * sx,
                496.0 * sy,
                10.0,
                1.2,
                [56, 189, 248, 200],
            );
            c.draw_text(
                64.0 * sx,
                122.0 * sy,
                &format!("FISTA RECONSTRUCTION · ITERATION {iter_idx}/20 (L1 ERR = {l1:.3})"),
                1.65 * sx,
                [125, 211, 252, 255],
            );

            let sx0 = 70.0 * sx;
            let sx1 = 858.0 * sx;
            let base_y = 548.0 * sy;
            c.draw_line(sx0, base_y, sx1, base_y, 1.2, [71, 85, 105, 230]);
            let max_v = d
                .nus_fista_iters
                .last()
                .map(|v| v.iter().copied().fold(1.0e-6_f64, f64::max))
                .unwrap_or(1.0);

            // Overlay dirty ghost in dim rose
            let mut ghost = Vec::with_capacity(d.nus_dirty_spec.len());
            for (i, &v) in d.nus_dirty_spec.iter().enumerate() {
                let x = lerp(
                    sx0,
                    sx1,
                    (i as f64) / ((d.nus_dirty_spec.len() - 1).max(1) as f64),
                );
                ghost.push([x, base_y - (v / max_v).clamp(0.0, 1.2) * 340.0 * sy]);
            }
            c.draw_polyline(&ghost, 1.4, [244, 63, 94, 95]);

            let mut pts = Vec::with_capacity(cur_spec.len());
            for (i, &v) in cur_spec.iter().enumerate() {
                let x = lerp(sx0, sx1, (i as f64) / ((cur_spec.len() - 1).max(1) as f64));
                pts.push([x, base_y - (v / max_v).clamp(0.0, 1.2) * 340.0 * sy]);
            }
            c.draw_polyline(&pts, 2.6, [52, 211, 153, 255]);

            // Right convergence card
            c.fill_rounded_rect(
                904.0 * sx,
                106.0 * sy,
                332.0 * sx,
                496.0 * sy,
                10.0,
                [15, 23, 42, 255],
            );
            c.stroke_rounded_rect(
                904.0 * sx,
                106.0 * sy,
                332.0 * sx,
                496.0 * sy,
                10.0,
                1.2,
                [52, 211, 153, 190],
            );
            c.draw_text(
                922.0 * sx,
                122.0 * sy,
                "RESIDUAL L1 DECAY",
                1.55 * sx,
                [110, 231, 183, 255],
            );

            let cx0 = 930.0 * sx;
            let cx1 = 1210.0 * sx;
            let cy0 = 185.0 * sy;
            let cy1 = 465.0 * sy;
            c.draw_line(cx0, cy1, cx1, cy1, 1.2, [71, 85, 105, 220]);
            c.draw_line(cx0, cy0, cx0, cy1, 1.2, [71, 85, 105, 220]);
            let max_l1 = d.nus_residual_l1.first().copied().unwrap_or(1.0).max(0.1);
            let mut l1_pts = Vec::with_capacity(iter_idx + 1);
            for (k, &lv) in d.nus_residual_l1.iter().take(iter_idx + 1).enumerate() {
                let px = lerp(cx0, cx1, (k as f64) / 20.0);
                let py = lerp(cy1, cy0, (lv / max_l1).clamp(0.0, 1.0));
                l1_pts.push([px, py]);
            }
            c.draw_polyline(&l1_pts, 2.6, [56, 189, 248, 255]);
            if let Some(last) = l1_pts.last() {
                c.fill_circle(last[0], last[1], 5.0, [251, 191, 36, 255]);
            }
            c.draw_text(
                926.0 * sx,
                498.0 * sy,
                "Weak 0.58-ppm peak",
                1.35 * sx,
                [226, 232, 240, 255],
            );
            c.draw_text(
                926.0 * sx,
                522.0 * sy,
                "recovered cleanly",
                1.35 * sx,
                [110, 231, 183, 255],
            );
            c.draw_spotlight_cursor(lerp(sx0, sx1, 0.58), base_y - 92.0 * sy, t_sec, 1.0);
        }
    }
}

fn render_m02_frame(
    c: &mut FrameCanvas,
    d: &M02SceneData,
    ch_idx: usize,
    local_t: f64,
    t_sec: f64,
) {
    let w = c.width as f64;
    let h = c.height as f64;
    let sx = w / 1280.0;
    let sy = h / 720.0;

    c.fill_rounded_rect(
        22.0 * sx,
        84.0 * sy,
        1236.0 * sx,
        540.0 * sy,
        12.0,
        [17, 25, 44, 240],
    );
    c.stroke_rounded_rect(
        22.0 * sx,
        84.0 * sy,
        1236.0 * sx,
        540.0 * sy,
        12.0,
        1.4,
        [51, 65, 85, 255],
    );

    match ch_idx {
        0 => {
            // Ch 1: Simulated Ubiquitin 2D 1H-15N HSQC contours + picked roots
            let px0 = 84.0 * sx;
            let px1 = 910.0 * sx;
            let py0 = 112.0 * sy;
            let py1 = 582.0 * sy;
            c.fill_rounded_rect(px0, py0, px1 - px0, py1 - py0, 8.0, [15, 23, 42, 255]);
            c.stroke_rounded_rect(px0, py0, px1 - px0, py1 - py0, 8.0, 1.2, [71, 85, 105, 255]);

            let map_x = |h_ppm: f64| -> f64 { px0 + (9.85 - h_ppm) / (9.85 - 6.35) * (px1 - px0) };
            let map_y =
                |n_ppm: f64| -> f64 { py0 + (n_ppm - 103.5) / (131.5 - 103.5) * (py1 - py0) };

            for h_t in [9.5, 9.0, 8.5, 8.0, 7.5, 7.0, 6.5] {
                let x = map_x(h_t);
                c.draw_line(x, py0, x, py1, 0.8, [30, 41, 59, 200]);
                c.draw_text(
                    x - 12.0 * sx,
                    py1 + 8.0 * sy,
                    &format!("{h_t:.1}"),
                    1.2 * sx,
                    [148, 163, 184, 255],
                );
            }
            for n_t in [105.0, 110.0, 115.0, 120.0, 125.0, 130.0] {
                let y = map_y(n_t);
                c.draw_line(px0, y, px1, y, 0.8, [30, 41, 59, 200]);
                c.draw_text(
                    px0 - 44.0 * sx,
                    y - 4.0 * sy,
                    &format!("{n_t:.0}"),
                    1.2 * sx,
                    [148, 163, 184, 255],
                );
            }

            for seg in &d.hsqc_pos_contours {
                if seg.len() >= 2 {
                    c.draw_line(
                        map_x(seg[0][0]),
                        map_y(seg[0][1]),
                        map_x(seg[1][0]),
                        map_y(seg[1][1]),
                        1.5,
                        [56, 189, 248, 225],
                    );
                }
            }
            for seg in &d.hsqc_neg_contours {
                if seg.len() >= 2 {
                    c.draw_line(
                        map_x(seg[0][0]),
                        map_y(seg[0][1]),
                        map_x(seg[1][0]),
                        map_y(seg[1][1]),
                        1.5,
                        [251, 113, 133, 225],
                    );
                }
            }

            let show_roots = ((smoothstep(local_t) * d.hsqc_roots.len() as f64).ceil() as usize)
                .min(d.hsqc_roots.len());
            for (idx, (hp, np, lbl)) in d.hsqc_roots.iter().take(show_roots).enumerate() {
                let rx = map_x(*hp);
                let ry = map_y(*np);
                c.fill_circle(rx, ry, 3.2, [52, 211, 153, 245]);
                if idx % 7 == 0 {
                    c.draw_text(rx + 6.0, ry - 4.0, lbl, 1.15 * sx, [226, 232, 240, 230]);
                }
            }

            // Right summary panel
            c.fill_rounded_rect(
                930.0 * sx,
                112.0 * sy,
                306.0 * sx,
                470.0 * sy,
                10.0,
                [15, 23, 42, 255],
            );
            c.stroke_rounded_rect(
                930.0 * sx,
                112.0 * sy,
                306.0 * sx,
                470.0 * sy,
                10.0,
                1.2,
                [56, 189, 248, 180],
            );
            c.draw_text(
                948.0 * sx,
                130.0 * sy,
                "2D 1H-15N HSQC",
                1.65 * sx,
                [125, 211, 252, 255],
            );
            c.draw_text(
                948.0 * sx,
                170.0 * sy,
                &format!("Picked roots: {show_roots}/{}", d.hsqc_roots.len()),
                1.4 * sx,
                [110, 231, 183, 255],
            );
            c.draw_text(
                948.0 * sx,
                204.0 * sy,
                "Cyan: +1 backbone & NH2",
                1.3 * sx,
                [56, 189, 248, 255],
            );
            c.draw_text(
                948.0 * sx,
                230.0 * sy,
                "Rose: -1 aliased Arg Ne-He",
                1.3 * sx,
                [251, 113, 133, 255],
            );
            c.draw_text(
                948.0 * sx,
                272.0 * sy,
                "Anchors 3D HNCACB strips",
                1.3 * sx,
                [203, 213, 225, 255],
            );
            if let Some((hp, np, _)) = d.hsqc_roots.get(show_roots.saturating_sub(1)) {
                c.draw_spotlight_cursor(map_x(*hp), map_y(*np), t_sec, 1.0);
            }
        }
        1 => {
            // Ch 2: Synchronized 3D HNCACB sequential strip walk (Phe4..Gly10)
            let left_x = 92.0 * sx;
            let right_x = 1226.0 * sx;
            let top_y = 136.0 * sy;
            let bot_y = 582.0 * sy;
            let gap = 14.0 * sx;
            let subset = &d.strips[3..10.min(d.strips.len())];
            let n_s = subset.len().max(1);
            let strip_w = ((right_x - left_x) - gap * ((n_s - 1) as f64)) / (n_s as f64);
            let map_cy =
                |c_ppm: f64| -> f64 { top_y + (c_ppm - 15.0) / (73.0 - 15.0) * (bot_y - top_y) };

            for ct in [20.0, 30.0, 40.0, 50.0, 60.0, 70.0] {
                let y = map_cy(ct);
                c.draw_text(
                    40.0 * sx,
                    y - 5.0 * sy,
                    &format!("{ct:.0}"),
                    1.25 * sx,
                    [148, 163, 184, 255],
                );
            }
            c.draw_text(
                34.0 * sx,
                102.0 * sy,
                "13C ppm",
                1.35 * sx,
                [125, 211, 252, 255],
            );

            let focus_idx = ((local_t * n_s as f64).floor() as usize).min(n_s - 1);
            let mut centers_x = Vec::with_capacity(n_s);

            for (col, st) in subset.iter().enumerate() {
                let sx0 = left_x + (col as f64) * (strip_w + gap);
                let scx = sx0 + 0.5 * strip_w;
                centers_x.push(scx);
                let active = col <= focus_idx;
                let border = if col == focus_idx {
                    [251, 191, 36, 255]
                } else {
                    [71, 85, 105, 220]
                };
                c.fill_rounded_rect(sx0, top_y, strip_w, bot_y - top_y, 8.0, [15, 23, 42, 255]);
                c.stroke_rounded_rect(sx0, top_y, strip_w, bot_y - top_y, 8.0, 1.5, border);
                c.draw_text(
                    scx - c.measure_text_width(&st.label, 1.5 * sx) * 0.5,
                    top_y - 26.0 * sy,
                    &st.label,
                    1.5 * sx,
                    [248, 250, 252, 255],
                );

                // Draw 1D C trace inside strip
                let trace_w = 0.40 * strip_w;
                let mut pos_poly = Vec::with_capacity(st.pos_pts.len());
                let mut neg_poly = Vec::with_capacity(st.neg_pts.len());
                for p in &st.pos_pts {
                    pos_poly.push([scx + p[1] * trace_w, map_cy(p[0])]);
                }
                for p in &st.neg_pts {
                    neg_poly.push([scx - p[1] * trace_w, map_cy(p[0])]);
                }
                c.draw_polyline(
                    &pos_poly,
                    1.8,
                    [56, 189, 248, if active { 245 } else { 110 }],
                );
                c.draw_polyline(
                    &neg_poly,
                    1.8,
                    [251, 113, 133, if active { 245 } else { 110 }],
                );

                if let Some(ca) = st.ca_i {
                    c.fill_circle(scx, map_cy(ca), 4.5, [56, 189, 248, 255]);
                }
                if let Some(cb) = st.cb_i {
                    c.fill_circle(scx, map_cy(cb), 4.5, [251, 113, 133, 255]);
                }
            }

            // Dashed sequential walk links up to focus_idx
            for col in 0..focus_idx {
                let st = &subset[col];
                if let Some(ca) = st.ca_i {
                    let y = map_cy(ca);
                    c.draw_dashed_line(
                        centers_x[col],
                        y,
                        centers_x[col + 1],
                        y,
                        6.0,
                        4.0,
                        2.0,
                        [56, 189, 248, 235],
                    );
                }
                if let Some(cb) = st.cb_i {
                    let y = map_cy(cb);
                    c.draw_dashed_line(
                        centers_x[col],
                        y,
                        centers_x[col + 1],
                        y,
                        6.0,
                        4.0,
                        2.0,
                        [251, 113, 133, 235],
                    );
                }
            }
            let cur_st = &subset[focus_idx];
            let spot_y = cur_st.ca_i.map(map_cy).unwrap_or(0.5 * (top_y + bot_y));
            c.draw_spotlight_cursor(centers_x[focus_idx], spot_y, t_sec, 1.0);
        }
        2 => {
            // Ch 3: bayes-sa Cv(T) convergence curve + empirical 5-seed benchmark bars
            c.fill_rounded_rect(
                44.0 * sx,
                106.0 * sy,
                620.0 * sx,
                496.0 * sy,
                10.0,
                [15, 23, 42, 255],
            );
            c.stroke_rounded_rect(
                44.0 * sx,
                106.0 * sy,
                620.0 * sx,
                496.0 * sy,
                10.0,
                1.2,
                [56, 189, 248, 190],
            );
            c.draw_text(
                64.0 * sx,
                122.0 * sy,
                "BAYES-SA SPECIFIC HEAT Cv(T) TRANSITION",
                1.6 * sx,
                [125, 211, 252, 255],
            );

            let px0 = 96.0 * sx;
            let px1 = 636.0 * sx;
            let py0 = 180.0 * sy;
            let py1 = 550.0 * sy;
            c.draw_line(px0, py1, px1, py1, 1.3, [71, 85, 105, 230]);
            c.draw_line(px0, py0, px0, py1, 1.3, [71, 85, 105, 230]);

            let n = d.cv_vals.len().max(1);
            let show = ((smoothstep(local_t) * n as f64).ceil() as usize).clamp(1, n);
            let max_cv = d.cv_vals.iter().copied().fold(1.0_f64, f64::max);
            let mut pts = Vec::with_capacity(show);
            for i in 0..show {
                let x = lerp(px0, px1, (i as f64) / ((n - 1).max(1) as f64));
                let y = lerp(py1, py0, (d.cv_vals[i] / max_cv).clamp(0.0, 1.0));
                pts.push([x, y]);
            }
            c.draw_polyline(&pts, 2.8, [56, 189, 248, 255]);
            if let Some(last) = pts.last() {
                c.fill_circle(last[0], last[1], 5.5, [251, 191, 36, 255]);
                c.draw_spotlight_cursor(last[0], last[1], t_sec, 1.0);
            }
            c.draw_text(
                96.0 * sx,
                566.0 * sy,
                &format!(
                    "Cooling stage {show}/{n} · T_crit={:.2}, max Cv={:.1}",
                    d.t_crit, d.max_cv
                ),
                1.3 * sx,
                [203, 213, 225, 255],
            );

            // Right card: Honest empirical benchmark disclosure
            c.fill_rounded_rect(
                686.0 * sx,
                106.0 * sy,
                550.0 * sx,
                496.0 * sy,
                10.0,
                [15, 23, 42, 255],
            );
            c.stroke_rounded_rect(
                686.0 * sx,
                106.0 * sy,
                550.0 * sx,
                496.0 * sy,
                10.0,
                1.2,
                [52, 211, 153, 190],
            );
            c.draw_text(
                708.0 * sx,
                122.0 * sy,
                "5-SEED BENCHMARK DISCLOSURE",
                1.65 * sx,
                [110, 231, 183, 255],
            );

            let bars = [
                (
                    "Ubiquitin --realistic (5-seed)",
                    98.4_f64,
                    [52, 211, 153, 255],
                ),
                (
                    "CISD3 BMRB 51439 overall acc",
                    64.38_f64,
                    [251, 191, 36, 255],
                ),
                ("CISD3 BMRB 51439 HC-prec", 100.0_f64, [56, 189, 248, 255]),
                (
                    "Cold BMRB HC-prec low bound",
                    86.18_f64,
                    [167, 139, 250, 255],
                ),
                (
                    "Cold BMRB HC-prec high bound",
                    96.31_f64,
                    [167, 139, 250, 255],
                ),
            ];
            for (i, (label, pct, col)) in bars.iter().enumerate() {
                let by = (176.0 + (i as f64) * 76.0) * sy;
                c.draw_text(
                    708.0 * sx,
                    by,
                    &format!("{label}: {pct:.2}%"),
                    1.38 * sx,
                    [226, 232, 240, 255],
                );
                c.fill_rounded_rect(
                    708.0 * sx,
                    by + 24.0 * sy,
                    500.0 * sx,
                    22.0 * sy,
                    6.0,
                    [30, 41, 59, 255],
                );
                c.fill_rounded_rect(
                    708.0 * sx,
                    by + 24.0 * sy,
                    (5.0 * pct * smoothstep(local_t)) * sx,
                    22.0 * sy,
                    6.0,
                    *col,
                );
            }
        }
        _ => {
            // Ch 4: 76-residue Ubiquitin sequence confidence ribbon
            c.fill_rounded_rect(
                44.0 * sx,
                106.0 * sy,
                1192.0 * sx,
                496.0 * sy,
                10.0,
                [15, 23, 42, 255],
            );
            c.stroke_rounded_rect(
                44.0 * sx,
                106.0 * sy,
                1192.0 * sx,
                496.0 * sy,
                10.0,
                1.2,
                [52, 211, 153, 190],
            );
            c.draw_text(
                64.0 * sx,
                122.0 * sy,
                &format!(
                    "UBIQUITIN 1..76 SEQUENCE CONFIDENCE RIBBON · QC={} · {}/76 >= 0.80 (acc {:.1}%)",
                    d.qc_verdict, d.n_high_conf, d.evo_acc_pct
                ),
                1.55 * sx,
                [110, 231, 183, 255],
            );

            let gx0 = 76.0 * sx;
            let gx1 = 1210.0 * sx;
            let gy0 = 180.0 * sy;
            let gy1 = 530.0 * sy;
            let hc_y = lerp(gy1, gy0, 0.80);
            c.draw_dashed_line(gx0, hc_y, gx1, hc_y, 8.0, 5.0, 1.6, [251, 191, 36, 220]);
            c.draw_text(
                gx0 + 6.0,
                hc_y - 18.0 * sy,
                "0.80 High-Confidence Gate",
                1.3 * sx,
                [251, 191, 36, 255],
            );

            let n = d.res_confidences.len().max(1);
            let show = ((smoothstep(local_t) * n as f64).ceil() as usize).clamp(1, n);
            let bar_w = (gx1 - gx0) / (n as f64);
            for (i, &conf) in d.res_confidences.iter().take(show).enumerate() {
                let bx = gx0 + (i as f64) * bar_w;
                let bh = (conf.clamp(0.0, 1.0)) * (gy1 - gy0);
                let col = if conf >= 0.80 {
                    [52, 211, 153, 240]
                } else if conf >= 0.50 {
                    [56, 189, 248, 220]
                } else {
                    [251, 113, 133, 210]
                };
                c.fill_rounded_rect(
                    bx + 1.0,
                    gy1 - bh,
                    (bar_w - 2.5).max(3.0),
                    bh.max(3.0),
                    2.0,
                    col,
                );
            }
            let focus = show.saturating_sub(1);
            let fx = gx0 + (focus as f64 + 0.5) * bar_w;
            let fy = gy1 - d.res_confidences[focus].clamp(0.0, 1.0) * (gy1 - gy0);
            c.draw_spotlight_cursor(fx, fy, t_sec, 1.0);
        }
    }
}

fn render_m03_frame(
    c: &mut FrameCanvas,
    d: &M03SceneData,
    ch_idx: usize,
    local_t: f64,
    t_sec: f64,
) {
    let w = c.width as f64;
    let h = c.height as f64;
    let sx = w / 1280.0;
    let sy = h / 720.0;

    c.fill_rounded_rect(
        22.0 * sx,
        84.0 * sy,
        1236.0 * sx,
        540.0 * sy,
        12.0,
        [17, 25, 44, 240],
    );
    c.stroke_rounded_rect(
        22.0 * sx,
        84.0 * sy,
        1236.0 * sx,
        540.0 * sy,
        12.0,
        1.4,
        [51, 65, 85, 255],
    );

    // Orbit angle: slow turn in Ch1/2, continuous 360° orbit in Ch4
    let az = match ch_idx {
        0 => 0.4 + local_t * 1.1,
        1 => 1.5 + local_t * 1.1,
        2 => 2.6,
        _ => local_t * TAU,
    };
    let elev = -0.35_f64;
    let (sy_a, cy_a) = az.sin_cos();
    let (sx_e, cx_e) = elev.sin_cos();

    let rot3 = |p: [f64; 3]| -> [f64; 3] {
        let x1 = p[0] * cy_a + p[2] * sy_a;
        let z1 = -p[0] * sy_a + p[2] * cy_a;
        let y2 = p[1] * cx_e - z1 * sx_e;
        let z2 = p[1] * sx_e + z1 * cx_e;
        [x1, y2, z2]
    };

    let view_cx = if ch_idx == 2 { 340.0 * sx } else { 440.0 * sx };
    let view_cy = 352.0 * sy;
    let proj = |p3: [f64; 3]| -> [f64; 2] {
        let persp = 68.0 / (68.0 - p3[2]).max(24.0);
        let scale = 13.0 * persp * sx.min(sy);
        [view_cx + p3[0] * scale, view_cy - p3[1] * scale]
    };

    if ch_idx == 2 {
        // Ch 3: Left mini 3D + Right Saupe RDC D_obs vs D_calc scatter + F8 holdout banner
        c.fill_rounded_rect(
            44.0 * sx,
            106.0 * sy,
            520.0 * sx,
            496.0 * sy,
            10.0,
            [15, 23, 42, 255],
        );
        c.stroke_rounded_rect(
            44.0 * sx,
            106.0 * sy,
            520.0 * sx,
            496.0 * sy,
            10.0,
            1.2,
            [251, 191, 36, 190],
        );
        c.draw_text(
            64.0 * sx,
            122.0 * sy,
            "1D3Z SAUPE ALIGNMENT TENSOR",
            1.55 * sx,
            [253, 224, 71, 255],
        );
        c.draw_text(
            64.0 * sx,
            152.0 * sy,
            &format!("Da(NH)={:.2} Hz  R={:.3}", d.da_nh_hz, d.rhombicity),
            1.35 * sx,
            [226, 232, 240, 255],
        );

        let rot_ca: Vec<[f64; 3]> = d.ca_coords.iter().map(|&p| rot3(p)).collect();
        for i in 0..rot_ca.len().saturating_sub(1) {
            let p0 = proj(rot_ca[i]);
            let p1 = proj(rot_ca[i + 1]);
            let col = s2_color_rgba(d.s2_vals[i]);
            c.draw_line(p0[0], p0[1], p1[0], p1[1], 4.0, col);
        }

        // Right: D_obs vs D_calc correlation scatter
        c.fill_rounded_rect(
            584.0 * sx,
            106.0 * sy,
            652.0 * sx,
            496.0 * sy,
            10.0,
            [15, 23, 42, 255],
        );
        c.stroke_rounded_rect(
            584.0 * sx,
            106.0 * sy,
            652.0 * sx,
            496.0 * sy,
            10.0,
            1.2,
            [56, 189, 248, 190],
        );
        c.draw_text(
            606.0 * sx,
            122.0 * sy,
            &format!(
                "N-H RDC CORRELATION (Q_work={:.3}, Q_free={:.3})",
                d.q_work, d.q_free
            ),
            1.5 * sx,
            [125, 211, 252, 255],
        );

        let px0 = 640.0 * sx;
        let px1 = 1190.0 * sx;
        let py0 = 175.0 * sy;
        let py1 = 515.0 * sy;
        c.draw_line(px0, py1, px1, py0, 1.5, [148, 163, 184, 175]);
        c.stroke_rounded_rect(px0, py0, px1 - px0, py1 - py0, 4.0, 1.0, [51, 65, 85, 255]);

        let show_pts = ((smoothstep(local_t) * d.rdc_points.len() as f64).ceil() as usize)
            .min(d.rdc_points.len());
        let lim = 24.0_f64;
        for &(obs, calc, is_free, _) in d.rdc_points.iter().take(show_pts) {
            let x = lerp(px0, px1, ((obs + lim) / (2.0 * lim)).clamp(0.0, 1.0));
            let y = lerp(py1, py0, ((calc + lim) / (2.0 * lim)).clamp(0.0, 1.0));
            let col = if is_free {
                [251, 191, 36, 255]
            } else {
                [56, 189, 248, 235]
            };
            c.fill_circle(x, y, if is_free { 5.0 } else { 3.8 }, col);
        }
        c.draw_text(
            606.0 * sx,
            540.0 * sy,
            "Cyan: working RDCs | Amber: ByPeptidePlane Q_free held-out",
            1.3 * sx,
            [203, 213, 225, 255],
        );
        c.draw_text(
            606.0 * sx,
            566.0 * sy,
            "F8 Holdout: 1KHM->1ZZK 5/5 improve (1.86-4.14 A), 0/5 < 1.80 A",
            1.3 * sx,
            [253, 224, 71, 255],
        );
        c.draw_spotlight_cursor(lerp(px0, px1, 0.72), lerp(py1, py0, 0.72), t_sec, 1.0);
        return;
    }

    // Ch 1, 2, 4: 3D Structure viewport on left + live info card on right
    c.fill_rounded_rect(
        44.0 * sx,
        106.0 * sy,
        790.0 * sx,
        496.0 * sy,
        10.0,
        [15, 23, 42, 255],
    );
    c.stroke_rounded_rect(
        44.0 * sx,
        106.0 * sy,
        790.0 * sx,
        496.0 * sy,
        10.0,
        1.2,
        [56, 189, 248, 180],
    );

    let rot_ca: Vec<[f64; 3]> = d.ca_coords.iter().map(|&p| rot3(p)).collect();
    #[derive(Clone)]
    struct Prim3D {
        z: f64,
        p0: [f64; 2],
        p1: [f64; 2],
        width: f64,
        col: [u8; 4],
        dashed: bool,
    }
    let mut prims: Vec<Prim3D> = Vec::new();

    if ch_idx == 1 {
        let show_noe = ((smoothstep(local_t) * d.noe_pairs.len() as f64).ceil() as usize)
            .min(d.noe_pairs.len());
        for &(a, b, dist) in d.noe_pairs.iter().take(show_noe) {
            let pa = rot_ca[a];
            let pb = rot_ca[b];
            prims.push(Prim3D {
                z: 0.5 * (pa[2] + pb[2]) - 0.5,
                p0: proj(pa),
                p1: proj(pb),
                width: if dist < 3.4 { 2.4 } else { 1.6 },
                col: if dist < 3.4 {
                    [52, 211, 153, 215]
                } else {
                    [56, 189, 248, 175]
                },
                dashed: true,
            });
        }
    }

    for i in 0..rot_ca.len().saturating_sub(1) {
        let pa = rot_ca[i];
        let pb = rot_ca[i + 1];
        let (width, col) = if ch_idx == 3 {
            let s2 = 0.5 * (d.s2_vals[i] + d.s2_vals[i + 1]);
            (3.8 + (1.0 - s2).clamp(0.0, 0.75) * 14.0, s2_color_rgba(s2))
        } else {
            let col = match d.ss_classes[i] {
                1 => [251, 113, 133, 255],
                2 => [56, 189, 248, 255],
                _ => [148, 163, 184, 240],
            };
            (5.2, col)
        };
        prims.push(Prim3D {
            z: 0.5 * (pa[2] + pb[2]),
            p0: proj(pa),
            p1: proj(pb),
            width,
            col,
            dashed: false,
        });
    }
    prims.sort_by(|a, b| a.z.total_cmp(&b.z));

    for pr in &prims {
        if pr.dashed {
            c.draw_dashed_line(
                pr.p0[0], pr.p0[1], pr.p1[0], pr.p1[1], 5.0, 3.0, pr.width, pr.col,
            );
        } else {
            c.draw_line(
                pr.p0[0],
                pr.p0[1],
                pr.p1[0],
                pr.p1[1],
                pr.width + 2.0,
                [9, 13, 24, 150],
            );
            c.draw_line(pr.p0[0], pr.p0[1], pr.p1[0], pr.p1[1], pr.width, pr.col);
        }
    }

    // Right info panel
    c.fill_rounded_rect(
        854.0 * sx,
        106.0 * sy,
        382.0 * sx,
        496.0 * sy,
        10.0,
        [15, 23, 42, 255],
    );
    c.stroke_rounded_rect(
        854.0 * sx,
        106.0 * sy,
        382.0 * sx,
        496.0 * sy,
        10.0,
        1.2,
        [251, 191, 36, 190],
    );
    c.draw_text(
        874.0 * sx,
        124.0 * sy,
        "1UBQ / 1D3Z METRICS",
        1.6 * sx,
        [253, 224, 71, 255],
    );
    c.draw_text(
        874.0 * sx,
        170.0 * sy,
        "Rose: Alpha-Helix  Cyan: Beta-Sheet",
        1.3 * sx,
        [226, 232, 240, 255],
    );
    c.draw_text(
        874.0 * sx,
        206.0 * sy,
        &format!("1D3Z long-range NOEs: {}", d.noe_pairs.len()),
        1.35 * sx,
        [110, 231, 183, 255],
    );
    c.draw_text(
        874.0 * sx,
        238.0 * sy,
        &format!("Firewalled torsions: {}", d.n_torsions_firewalled),
        1.35 * sx,
        [125, 211, 252, 255],
    );
    c.draw_text(
        874.0 * sx,
        270.0 * sy,
        &format!("Homologs excluded: {}", d.n_excluded_templates),
        1.35 * sx,
        [253, 224, 71, 255],
    );
    c.draw_text(
        874.0 * sx,
        318.0 * sy,
        "F8 Unseen Cold Holdout:",
        1.4 * sx,
        [251, 191, 36, 255],
    );
    c.draw_text(
        874.0 * sx,
        346.0 * sy,
        "1KHM -> 1ZZK improves NOE-only",
        1.28 * sx,
        [254, 243, 199, 255],
    );
    c.draw_text(
        874.0 * sx,
        372.0 * sy,
        "on 5/5 seeds (1.86-4.14 A),",
        1.28 * sx,
        [254, 243, 199, 255],
    );
    c.draw_text(
        874.0 * sx,
        398.0 * sy,
        "0/5 seeds < 1.80 A!",
        1.28 * sx,
        [253, 164, 175, 255],
    );

    if let Some(&last_ca) = rot_ca.last() {
        let lp = proj(last_ca);
        c.draw_spotlight_cursor(lp[0], lp[1], t_sec, 1.0);
    }
}

fn render_m04_frame(
    c: &mut FrameCanvas,
    d: &M04SceneData,
    ch_idx: usize,
    local_t: f64,
    t_sec: f64,
) {
    let w = c.width as f64;
    let h = c.height as f64;
    let sx = w / 1280.0;
    let sy = h / 720.0;

    c.fill_rounded_rect(
        22.0 * sx,
        84.0 * sy,
        1236.0 * sx,
        540.0 * sy,
        12.0,
        [17, 25, 44, 240],
    );
    c.stroke_rounded_rect(
        22.0 * sx,
        84.0 * sy,
        1236.0 * sx,
        540.0 * sy,
        12.0,
        1.4,
        [51, 65, 85, 255],
    );

    match ch_idx {
        0 => {
            // Ch 1: Pseudo-3D locked-footprint plane integration vs naive local-max bias
            c.fill_rounded_rect(
                44.0 * sx,
                106.0 * sy,
                1192.0 * sx,
                496.0 * sy,
                10.0,
                [15, 23, 42, 255],
            );
            c.stroke_rounded_rect(
                44.0 * sx,
                106.0 * sy,
                1192.0 * sx,
                496.0 * sy,
                10.0,
                1.2,
                [251, 113, 133, 190],
            );
            c.draw_text(
                64.0 * sx,
                124.0 * sy,
                "LOCKED-FOOTPRINT PLANE INTEGRATION VS NAIVE LOCAL-MAX BIAS (SNR 80 -> 4)",
                1.55 * sx,
                [253, 164, 175, 255],
            );

            let gx0 = 120.0 * sx;
            let gx1 = 1160.0 * sx;
            let gy0 = 185.0 * sy;
            let gy1 = 540.0 * sy;
            let zero_y = gy1 - 35.0 * sy;
            c.draw_line(gx0, zero_y, gx1, zero_y, 1.4, [148, 163, 184, 220]);

            let n = d.snr_levels.len().max(1);
            let show = ((smoothstep(local_t) * n as f64).ceil() as usize).clamp(1, n);
            let group_w = (gx1 - gx0) / (n as f64);
            let max_bias = d
                .local_max_bias_pct
                .iter()
                .map(|v| v.abs())
                .fold(25.0_f64, f64::max);

            for i in 0..show {
                let x = gx0 + (i as f64) * group_w + 26.0 * sx;
                let bw = 46.0 * sx;
                let ls_h = (d.locked_ls_bias_pct[i].abs() / max_bias) * (zero_y - gy0);
                let mx_h = (d.local_max_bias_pct[i].abs() / max_bias) * (zero_y - gy0);
                c.fill_rounded_rect(
                    x,
                    zero_y - ls_h.max(4.0),
                    bw,
                    ls_h.max(4.0),
                    4.0,
                    [52, 211, 153, 240],
                );
                c.fill_rounded_rect(
                    x + bw + 12.0 * sx,
                    zero_y - mx_h.max(4.0),
                    bw,
                    mx_h.max(4.0),
                    4.0,
                    [251, 113, 133, 240],
                );
                c.draw_text(
                    x,
                    zero_y + 12.0 * sy,
                    &format!("SNR={:.0}", d.snr_levels[i]),
                    1.35 * sx,
                    [226, 232, 240, 255],
                );
            }
            c.draw_text(130.0 * sx, 162.0 * sy, "Emerald: Locked-footprint Linear LS (~0% bias)   Rose: Naive Local-Max (strong positive noise bias)", 1.35 * sx, [226, 232, 240, 255]);
            c.draw_spotlight_cursor(
                gx0 + (show as f64 - 0.4) * group_w,
                zero_y - 120.0 * sy,
                t_sec,
                1.0,
            );
        }
        1 => {
            // Ch 2: Dual-field 600 & 800 MHz R1 exponential fits + MC bootstrap
            c.fill_rounded_rect(
                44.0 * sx,
                106.0 * sy,
                820.0 * sx,
                496.0 * sy,
                10.0,
                [15, 23, 42, 255],
            );
            c.stroke_rounded_rect(
                44.0 * sx,
                106.0 * sy,
                820.0 * sx,
                496.0 * sy,
                10.0,
                1.2,
                [56, 189, 248, 190],
            );
            c.draw_text(
                64.0 * sx,
                122.0 * sy,
                "DUAL-FIELD (600 & 800 MHz) R1 VARPRO+LM DECAY FITS",
                1.55 * sx,
                [125, 211, 252, 255],
            );

            let px0 = 104.0 * sx;
            let px1 = 832.0 * sx;
            let py0 = 178.0 * sy;
            let py1 = 550.0 * sy;
            c.draw_line(px0, py1, px1, py1, 1.2, [71, 85, 105, 230]);
            c.draw_line(px0, py0, px0, py1, 1.2, [71, 85, 105, 230]);

            let t_max = d.times_r1.last().copied().unwrap_or(1.4).max(0.1);
            let mut curve600 = Vec::with_capacity(48);
            let mut curve800 = Vec::with_capacity(48);
            let rev = smoothstep(local_t);
            let max_s = ((rev * 48.0).ceil() as usize).clamp(2, 48);
            for k in 0..max_s {
                let f = (k as f64) / 47.0;
                let t = f * t_max;
                let x = lerp(px0, px1, f);
                let y6 = lerp(py1, py0, (-d.fit_600_rate * t).exp());
                let y8 = lerp(py1, py0, (-d.fit_800_rate * t).exp());
                curve600.push([x, y6]);
                curve800.push([x, y8]);
            }
            c.draw_polyline(&curve600, 2.6, [56, 189, 248, 255]);
            c.draw_polyline(&curve800, 2.6, [251, 191, 36, 255]);

            for (i, &t) in d.times_r1.iter().enumerate() {
                let x = lerp(px0, px1, t / t_max);
                let y6 = lerp(py1, py0, (d.curve_600_r1[i] / 105.0).clamp(0.0, 1.0));
                let y8 = lerp(py1, py0, (d.curve_800_r1[i] / 105.0).clamp(0.0, 1.0));
                c.fill_circle(x, y6, 4.5, [56, 189, 248, 255]);
                c.fill_circle(x, y8, 4.5, [251, 191, 36, 255]);
            }

            c.fill_rounded_rect(
                884.0 * sx,
                106.0 * sy,
                352.0 * sx,
                496.0 * sy,
                10.0,
                [15, 23, 42, 255],
            );
            c.stroke_rounded_rect(
                884.0 * sx,
                106.0 * sy,
                352.0 * sx,
                496.0 * sy,
                10.0,
                1.2,
                [251, 191, 36, 190],
            );
            c.draw_text(
                904.0 * sx,
                126.0 * sy,
                "100-MC BOOTSTRAP FIT",
                1.55 * sx,
                [253, 224, 71, 255],
            );
            c.draw_text(
                904.0 * sx,
                184.0 * sy,
                &format!(
                    "600 MHz R1: {:.3} +/- {:.3} s^-1",
                    d.fit_600_rate, d.fit_600_se
                ),
                1.35 * sx,
                [56, 189, 248, 255],
            );
            c.draw_text(
                904.0 * sx,
                224.0 * sy,
                &format!(
                    "800 MHz R1: {:.3} +/- {:.3} s^-1",
                    d.fit_800_rate, d.fit_800_se
                ),
                1.35 * sx,
                [251, 191, 36, 255],
            );
            c.draw_text(
                904.0 * sx,
                280.0 * sy,
                "Fisher CRLB + 100-trial MC",
                1.3 * sx,
                [203, 213, 225, 255],
            );
            if let Some(last) = curve600.last() {
                c.draw_spotlight_cursor(last[0], last[1], t_sec, 1.0);
            }
        }
        2 => {
            // Ch 3: Kay-Torchia-Bax trimmed R2/R1 tau_m & Woessner D_par/D_perp
            c.fill_rounded_rect(
                44.0 * sx,
                106.0 * sy,
                1192.0 * sx,
                496.0 * sy,
                10.0,
                [15, 23, 42, 255],
            );
            c.stroke_rounded_rect(
                44.0 * sx,
                106.0 * sy,
                1192.0 * sx,
                496.0 * sy,
                10.0,
                1.2,
                [52, 211, 153, 190],
            );
            c.draw_text(
                64.0 * sx,
                124.0 * sy,
                &format!(
                    "KAY-TORCHIA-BAX TRIMMED R2/R1 (tau_m={:.2} ns) & WOESSNER D_par/D_perp={:.2}",
                    d.tau_m_ns, d.d_par_perp
                ),
                1.5 * sx,
                [110, 231, 183, 255],
            );

            let gx0 = 90.0 * sx;
            let gx1 = 1200.0 * sx;
            let gy0 = 185.0 * sy;
            let gy1 = 545.0 * sy;
            c.draw_line(gx0, gy1, gx1, gy1, 1.2, [71, 85, 105, 220]);

            let n = d.seq_ids.len().max(1);
            let prog = smoothstep((local_t / 0.50).clamp(0.0, 1.0));
            let show = ((prog * n as f64).ceil() as usize).clamp(1, n);
            let max_r = d.r2_over_r1.iter().copied().fold(6.0_f64, f64::max);
            for i in 0..show {
                let x = lerp(gx0, gx1, (d.seq_ids[i] as f64) / 76.0);
                let y = lerp(gy1, gy0, (d.r2_over_r1[i] / max_r).clamp(0.0, 1.0));
                let is_rex = d.rex_vals[i] > 0.4;
                let is_tail = d.seq_ids[i] >= 72;
                let col = if is_rex {
                    [251, 191, 36, 255]
                } else if is_tail {
                    [251, 113, 133, 255]
                } else {
                    [56, 189, 248, 240]
                };
                c.fill_circle(x, y, if is_rex { 6.5 } else { 4.2 }, col);
            }
            c.draw_text(
                100.0 * sx,
                162.0 * sy,
                "Cyan: Rigid core  Amber: Ile23/25 R_ex outlier  Rose: Flexible C-tail",
                1.35 * sx,
                [226, 232, 240, 255],
            );
            let lx = lerp(gx0, gx1, (d.seq_ids[show - 1] as f64) / 76.0);
            let ly = lerp(gy1, gy0, (d.r2_over_r1[show - 1] / max_r).clamp(0.0, 1.0));
            c.draw_spotlight_cursor(lx, ly, t_sec, 1.0);
        }
        _ => {
            // Ch 4: Lipari-Szabo M1-M5 S^2 profile across Ubiquitin 1..76
            c.fill_rounded_rect(
                44.0 * sx,
                106.0 * sy,
                1192.0 * sx,
                496.0 * sy,
                10.0,
                [15, 23, 42, 255],
            );
            c.stroke_rounded_rect(
                44.0 * sx,
                106.0 * sy,
                1192.0 * sx,
                496.0 * sy,
                10.0,
                1.2,
                [251, 113, 133, 190],
            );
            c.draw_text(
                64.0 * sx,
                124.0 * sy,
                "LIPARI-SZABO M1-M5 S^2 ORDER PARAMETER PROFILE (1..76) · SYNTHETIC SHOWCASE",
                1.5 * sx,
                [253, 164, 175, 255],
            );

            let gx0 = 90.0 * sx;
            let gx1 = 1200.0 * sx;
            let gy0 = 185.0 * sy;
            let gy1 = 545.0 * sy;
            c.draw_line(gx0, gy1, gx1, gy1, 1.2, [71, 85, 105, 220]);

            let n = d.seq_ids.len().max(1);
            let prog = smoothstep((local_t / 0.50).clamp(0.0, 1.0));
            let show = ((prog * n as f64).ceil() as usize).clamp(1, n);
            let bar_w = (gx1 - gx0) / 78.0;
            for i in 0..show {
                let seq = d.seq_ids[i] as f64;
                let s2 = d.s2_vals[i].clamp(0.0, 1.0);
                let bx = lerp(gx0, gx1, seq / 76.0) - 0.5 * bar_w;
                let bh = s2 * (gy1 - gy0);
                let col = if d.rex_vals[i] > 0.4 {
                    [251, 191, 36, 255]
                } else {
                    s2_color_rgba(s2)
                };
                c.fill_rounded_rect(bx, gy1 - bh, (bar_w - 1.5).max(4.0), bh.max(4.0), 2.5, col);
                if d.seq_ids[i] == 23 {
                    c.draw_text(
                        bx - 14.0,
                        gy1 - bh - 20.0 * sy,
                        "Ile23 R_ex",
                        1.25 * sx,
                        [253, 224, 71, 255],
                    );
                }
            }
            c.draw_text(
                100.0 * sx,
                162.0 * sy,
                "Amber: M3 Ile23 R_ex | Rose tail (72..76): picosecond-nanosecond flexibility",
                1.35 * sx,
                [226, 232, 240, 255],
            );
            let fx = lerp(gx0, gx1, (d.seq_ids[show - 1] as f64) / 76.0);
            let fy = gy1 - d.s2_vals[show - 1].clamp(0.0, 1.0) * (gy1 - gy0);
            c.draw_spotlight_cursor(fx, fy, t_sec, 1.0);
        }
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canvas_and_png_encoding_roundtrip_header() {
        let mut c = FrameCanvas::new(320, 240);
        c.clear_gradient([12, 18, 32], [28, 38, 64]);
        c.fill_rounded_rect(20.0, 20.0, 120.0, 80.0, 10.0, [56, 189, 248, 220]);
        c.stroke_rounded_rect(20.0, 20.0, 120.0, 80.0, 10.0, 2.0, [255, 255, 255, 255]);
        c.draw_line(10.0, 10.0, 300.0, 220.0, 2.5, [52, 211, 153, 255]);
        c.fill_circle(160.0, 120.0, 24.0, [251, 191, 36, 200]);
        c.draw_text(32.0, 40.0, "αβγ τm Q_free ≤ 25%", 1.5, [248, 250, 252, 255]);

        let png = encode_rgb_png(c.width as u32, c.height as u32, &c.rgb)
            .expect("PNG encoding should succeed");
        assert!(png.starts_with(b"\x89PNG\r\n\x1a\n"));
        assert!(png.ends_with(b"IEND\xaeB`\x82"));
    }

    #[test]
    fn test_core_modules_01_to_04_build_and_render_deterministically() {
        for m in TourModuleId::CORE_MODULES {
            let s1 = build_core_tour_state(m, CANONICAL_TOUR_SEED)
                .unwrap_or_else(|e| panic!("module {} failed: {e}", m.slug()));
            let s2 = build_core_tour_state(m, CANONICAL_TOUR_SEED)
                .unwrap_or_else(|e| panic!("module {} second run failed: {e}", m.slug()));
            assert_eq!(s1.bundle.schema_version, TOUR_SCHEMA_VERSION);
            assert_eq!(s1.bundle.chapters.len(), 4);
            assert_eq!(
                serde_json::to_string(&s1.bundle).expect("serialize 1"),
                serde_json::to_string(&s2.bundle).expect("serialize 2"),
                "tour JSON must be deterministic for {}",
                m.slug()
            );

            let f_a = render_core_tour_frame(&s1, 2.0, 640, 360);
            let f_b = render_core_tour_frame(&s2, 2.0, 640, 360);
            assert_eq!(f_a, f_b, "frame must be byte-identical across runs");

            let f_ch2 = render_core_tour_frame(&s1, 6.0, 640, 360);
            assert_ne!(f_a, f_ch2, "chapter frames must differ");
        }
    }

    #[test]
    fn test_cli_validation_rejects_bad_arguments() {
        assert!(run_export_tour_cli(&[]).is_err());
        assert!(run_export_tour_cli(&["--out-dir".into(), "".into()]).is_err());
        assert!(run_export_tour_cli(&[
            "--out-dir".into(),
            "/tmp/x".into(),
            "--module".into(),
            "bogus".into()
        ])
        .is_err());
        assert!(run_export_tour_cli(&[
            "--out-dir".into(),
            "/tmp/x".into(),
            "--width".into(),
            "1281".into()
        ])
        .is_err());
        assert!(run_export_tour_cli(&[
            "--out-dir".into(),
            "/tmp/x".into(),
            "--module".into(),
            "05".into()
        ])
        .is_err());
    }
}
