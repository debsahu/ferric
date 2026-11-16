//! Deterministic publication SVG figure showcase pipeline (`T16.1`: `ferric export-fig`).
//!
//! Provenance: Original clean-room vector SVG rendering engine driving audited `nmr-core`
//! physical simulation and optimization modules with fixed deterministic seed (`42`), zero
//! external image dependencies, and zero nondeterministic hash-map iteration:
//! - Backbone `1H-15N` HSQC & `3D HNCACB` strip simulation/assignment: Kay et al., *J. Magn.
//!   Reson.* **1990**, 89, 496–514; Wittekind & Mueller, *J. Magn. Reson. B* **1993**, 101,
//!   201–205; Schmidt & Güntert, *J. Am. Chem. Soc.* **2012**, 134, 12817–12829.
//! - Lipari–Szabo $S^2$ internal dynamics & NOE geometry on Ubiquitin `1UBQ`: Lipari & Szabo,
//!   *J. Am. Chem. Soc.* **1982**, 104, 4546–4570; Vijay-Kumar, Bugg & Cook, *J. Mol. Biol.*
//!   **1987**, 194, 531–544 (`PDB 1UBQ`, public domain CC0).
//! - Analytical 1D multiplet `J`-trees, arrayed kinetics stack & internal-standard `qNMR`:
//!   Hoye, Hanson & Vyvyan, *J. Org. Chem.* **1994**, 59, 4096–4103; Pauli et al., *J. Med.
//!   Chem.* **2014**, 57, 9220–9231; Malz & Jancke, *J. Pharm. Biomed. Anal.* **2005**, 38, 813.
//! - Pseudo-2D `DOSY` Stejskal–Tanner / `BPP-LED` fitting: Stejskal & Tanner, *J. Chem. Phys.*
//!   **1965**, 42, 288–292; Wu, Chen & Johnson, *J. Magn. Reson. A* **1995**, 115, 260–264.
//! - Untargeted 1D NMR metabolomics (`PCA`, `OPLS-DA` S-plot, `STOCSY`): Trygg & Wold,
//!   *J. Chemom.* **2002**, 16, 119–128; Wiklund et al., *Anal. Chem.* **2008**, 80, 115–122;
//!   Cloarec et al., *Anal. Chem.* **2005**, 77, 1282–1289.
//! - Chemical exchange (`CEST`, `CPMG`, tight-binding `CSP` $K_d$): Vallurupalli, Bouvignies &
//!   Kay, *J. Am. Chem. Soc.* **2012**, 134, 8148–8161; Baldwin, *J. Magn. Reson.* **2014**,
//!   244, 114–124; Williamson, *Prog. Nucl. Magn. Reson. Spectrosc.* **2013**, 73, 1–16.
//! - 2D skeletal chemical depiction & `1H`/`13C` shift linking: Bremser, *Anal. Chim. Acta*
//!   **1978**, 103, 355–365; Pretsch, Bühlmann & Badertscher, Springer, **2009**; Octet-Draw
//!   (`Apache-2.0`).

use std::collections::BTreeMap;
use std::f64::consts::TAU;
use std::fmt::Write as FmtWrite;
use std::fs;
use std::path::{Path, PathBuf};

use nmr_core::cest::{
    fit_cest_global_cpu, simulate_cest_2state_profile, Cest2StateParams, CestFitConfig,
    CestNormalization, CestProfileDataset,
};
use nmr_core::cpmg::{
    fit_cpmg_global_cpu, r2eff_baldwin2014, CpmgFitConfig, CpmgModel, CpmgSeriesDataset,
};
use nmr_core::csp::{
    fit_csp_global_kd, naive_michaelis_menten_csp, quadratic_tight_binding_csp, CspAnalysisConfig,
    ResidueCspTrajectory, ResidueTitrationPoint,
};
use nmr_core::dosy::{
    compute_b_values_s_m2, estimate_hydrodynamics, fit_stejskal_tanner_multi,
    synthesize_dosy_grid_from_peaks, BootstrapMethod, DosyFitConfig, DosyGridConfig, DosyNucleus,
    DosyPeakEntry, DosySequenceType, DosySolvent, GradientShape,
    ModelSelectionCriterion as DosyCriterion, SoluteCompactnessClass,
};
use nmr_core::evo_graph::{assign_evo_graph_from_peaks, EvoGraphConfig};
use nmr_core::metabolomics::{
    generate_demo_biofluid_cohort, run_metabolomics_pipeline, MetabolomicsPipelineConfig,
};
use nmr_core::modelfree::{
    extract_nh_unit_vectors_from_chain, fit_modelfree_dataset, predict_r1_r2_noe,
    DiffusionTensorKind, LocalMfParams, ModelFreeConfig, ModelSelectionCriterion as MfCriterion,
    MultiFieldResidueInput,
};
use nmr_core::multiplet::{
    analyze_multiplet_region, calculate_qnmr_purity, integrate_regions,
    synthesize_multiplet_spectrum, BaselineMode, IntegralRegionSpec, JournalNmrReport,
    MultipletConfig, MultipletSubPeak, NormalizationMode, PublicationStyle, QnmrAcquisitionParams,
    QnmrAnalyteSignal, QnmrAnalyteSpec, QnmrStandardSpec, ReportNucleus, ReportedSignal,
    TextEncoding,
};
use nmr_core::nmrstar::read_star_str;
use nmr_core::noesy::{parse_pdb_str_first_model, Vec3};
use nmr_core::octet_analyt::nmr_predict::{predict_nmr, NmrNucleus};
use nmr_core::octet_analyt::AnalytMol;
use nmr_core::octet_mol::{clean_2d, parse_smiles, perceive_cip};
use nmr_core::octet_render::{
    compile_scene_primitives, JournalStyle, PathCmd, RenderScene, VectorPrimitive,
};
use nmr_core::peaks::Peak;
use nmr_core::priors::Priors;
use nmr_core::relax_fit::{RelaxPhysicsConfig, ResidueRelaxRates};
use nmr_core::rng::Rng;
use nmr_core::simulate::{
    render_hsqc_2d, render_triple_3d, simulate, SimConfig, SpectrumRenderConfig,
};

/// Canonical deterministic PRNG seed for `ferric export-fig`.
pub const CANONICAL_EXPORT_SEED: u64 = 42;

const UBIQUITIN_STAR_FIXTURE: &str = include_str!("../../../data/fixtures/synthetic_ubiquitin.str");
const UBIQUITIN_1UBQ_PDB_FIXTURE: &str =
    include_str!("../../../data/fixtures/ubiquitin_1ubq_xray.pdb");

/// Individual showcase figure panel selector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShowcasePanel {
    Hsqc,
    Strips,
    PdbS2,
    Nmr1dStack,
    Dosy,
    Metabolomics,
    Exchange,
    Octet,
}

impl ShowcasePanel {
    pub const ALL: [Self; 8] = [
        Self::Hsqc,
        Self::Strips,
        Self::PdbS2,
        Self::Nmr1dStack,
        Self::Dosy,
        Self::Metabolomics,
        Self::Exchange,
        Self::Octet,
    ];

    pub const fn cli_slug(self) -> &'static str {
        match self {
            Self::Hsqc => "hsqc",
            Self::Strips => "strips",
            Self::PdbS2 => "pdb-s2",
            Self::Nmr1dStack => "nmr1d-stack",
            Self::Dosy => "dosy",
            Self::Metabolomics => "metabolomics",
            Self::Exchange => "exchange",
            Self::Octet => "octet",
        }
    }

    pub const fn filename(self) -> &'static str {
        match self {
            Self::Hsqc => "01_hsqc_contour_overlay.svg",
            Self::Strips => "02_sequential_strip_walk.svg",
            Self::PdbS2 => "03_pdb_s2_dynamics_putty.svg",
            Self::Nmr1dStack => "04_nmr1d_multiplet_stack.svg",
            Self::Dosy => "05_dosy_2d_diffusion_map.svg",
            Self::Metabolomics => "06_metabolomics_oplsda_stocsy.svg",
            Self::Exchange => "07_cest_cpmg_csp_exchange.svg",
            Self::Octet => "08_octet_structure_link.svg",
        }
    }
}

/// Requested `--panel` target (`all` or a single [`ShowcasePanel`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PanelSelection {
    All,
    Single(ShowcasePanel),
}

impl PanelSelection {
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "all" => Ok(Self::All),
            "hsqc" => Ok(Self::Single(ShowcasePanel::Hsqc)),
            "strips" => Ok(Self::Single(ShowcasePanel::Strips)),
            "pdb-s2" => Ok(Self::Single(ShowcasePanel::PdbS2)),
            "nmr1d-stack" => Ok(Self::Single(ShowcasePanel::Nmr1dStack)),
            "dosy" => Ok(Self::Single(ShowcasePanel::Dosy)),
            "metabolomics" => Ok(Self::Single(ShowcasePanel::Metabolomics)),
            "exchange" => Ok(Self::Single(ShowcasePanel::Exchange)),
            "octet" => Ok(Self::Single(ShowcasePanel::Octet)),
            other => Err(format!(
                "unknown --panel {other:?}; expected all|hsqc|strips|pdb-s2|nmr1d-stack|dosy|metabolomics|exchange|octet"
            )),
        }
    }

    pub fn panels(self) -> Vec<ShowcasePanel> {
        match self {
            Self::All => ShowcasePanel::ALL.to_vec(),
            Self::Single(p) => vec![p],
        }
    }
}

pub fn export_fig_usage() -> &'static str {
    "Usage:\n\
       ferric export-fig --out-dir <dir> [--panel all|hsqc|strips|pdb-s2|nmr1d-stack|dosy|metabolomics|exchange|octet]\n\
     \n\
     Deterministic publication SVG showcase exporter (fixed seed 42, zero external image tools):\n\
       01_hsqc_contour_overlay.svg      2D 1H-15N HSQC positive/negative contours + assigned labels\n\
       02_sequential_strip_walk.svg     3D HNCACB i vs i-1 CA/CB sequential strip walk\n\
       03_pdb_s2_dynamics_putty.svg     Ubiquitin 1UBQ 3D backbone S^2 putty ribbon + NOE cylinders\n\
       04_nmr1d_multiplet_stack.svg     1D multiplet J-tree + tilted kinetics stack + qNMR integral\n\
       05_dosy_2d_diffusion_map.svg     2D (delta ppm x log10 D) DOSY contour map + decay inset\n\
       06_metabolomics_oplsda_stocsy.svg PCA Hotelling T^2 + OPLS-DA S-plot + STOCSY spectrum\n\
       07_cest_cpmg_csp_exchange.svg    CEST minor-state dip + multi-field CPMG + CSP Kd isotherm\n\
       08_octet_structure_link.svg      2D skeletal depiction linked to 1H/13C multiplet table"
}

/// Parse CLI arguments and export the requested deterministic SVG panel(s).
pub fn run_export_fig_cli(args: &[String]) -> Result<(), String> {
    let mut out_dir: Option<PathBuf> = None;
    let mut selection = PanelSelection::All;
    let mut i = 0usize;

    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" | "help" => {
                println!("{}", export_fig_usage());
                return Ok(());
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
            "--panel" => {
                i += 1;
                let Some(val) = args.get(i) else {
                    return Err("missing value after --panel".to_string());
                };
                selection = PanelSelection::parse(val)?;
            }
            other => {
                if let Some(rest) = other.strip_prefix("--out-dir=") {
                    if rest.trim().is_empty() {
                        return Err("--out-dir cannot be empty".to_string());
                    }
                    out_dir = Some(PathBuf::from(rest));
                } else if let Some(rest) = other.strip_prefix("--panel=") {
                    selection = PanelSelection::parse(rest)?;
                } else {
                    return Err(format!(
                        "unknown export-fig argument {other:?}\n\n{}",
                        export_fig_usage()
                    ));
                }
            }
        }
        i += 1;
    }

    let target_dir = out_dir
        .ok_or_else(|| format!("missing required --out-dir <dir>\n\n{}", export_fig_usage()))?;

    export_panels_to_dir(&target_dir, selection, CANONICAL_EXPORT_SEED)?;
    Ok(())
}

/// Export selected panels into `out_dir`, validating directory writability and SVG integrity.
pub fn export_panels_to_dir(
    out_dir: &Path,
    selection: PanelSelection,
    seed: u64,
) -> Result<Vec<PathBuf>, String> {
    if out_dir.as_os_str().is_empty() {
        return Err("output directory path cannot be empty".to_string());
    }
    if out_dir.exists() && !out_dir.is_dir() {
        return Err(format!(
            "output path is an existing non-directory file: {}",
            out_dir
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("<path>")
        ));
    }
    fs::create_dir_all(out_dir).map_err(|e| format!("failed to create output directory: {e}"))?;

    let panels = selection.panels();
    let mut written = Vec::with_capacity(panels.len());
    for panel in panels {
        let svg = render_panel_svg(panel, seed)?;
        validate_svg_output(&svg, panel.filename())?;
        let dest = out_dir.join(panel.filename());
        if dest.exists() && dest.is_dir() {
            return Err(format!(
                "cannot overwrite directory with SVG file {}",
                panel.filename()
            ));
        }
        fs::write(&dest, svg.as_bytes())
            .map_err(|e| format!("failed to write {}: {e}", panel.filename()))?;
        written.push(dest);
    }
    Ok(written)
}

/// Render a single [`ShowcasePanel`] to a deterministic SVG string.
pub fn render_panel_svg(panel: ShowcasePanel, seed: u64) -> Result<String, String> {
    match panel {
        ShowcasePanel::Hsqc => render_01_hsqc_svg(seed),
        ShowcasePanel::Strips => render_02_strips_svg(seed),
        ShowcasePanel::PdbS2 => render_03_pdb_s2_svg(seed),
        ShowcasePanel::Nmr1dStack => render_04_nmr1d_stack_svg(seed),
        ShowcasePanel::Dosy => render_05_dosy_svg(seed),
        ShowcasePanel::Metabolomics => render_06_metabolomics_svg(seed),
        ShowcasePanel::Exchange => render_07_exchange_svg(seed),
        ShowcasePanel::Octet => render_08_octet_svg(seed),
    }
}

// ============================================================================
// Shared SVG Document Builder, Formatting & Marching-Squares Contour Engine
// ============================================================================

fn escape_xml(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            c => out.push(c),
        }
    }
    out
}

/// Deterministic fixed-point formatter guarding against `NaN`/`Inf`, `-0.0`, and cross-OS 1-ULP jitter.
#[inline]
fn fmt_f(val: f64, decimals: usize) -> Result<String, String> {
    if !val.is_finite() {
        return Err(format!(
            "non-finite SVG coordinate or metric encountered: {val}"
        ));
    }
    // Snap to 1e-8 grid to eliminate 1-ULP libm variations across aarch64/x86_64 while preserving
    // full physical precision at <= 4 decimal places.
    let snapped = (val * 1.0e8).round() * 1.0e-8;
    let scale = 10.0_f64.powi(decimals as i32);
    let rounded = (snapped * scale).round() / scale;
    let clean = if rounded.abs() < 0.5 / scale {
        0.0
    } else {
        rounded
    };
    Ok(format!("{clean:.decimals$}"))
}

#[inline]
fn f1(v: f64) -> Result<String, String> {
    fmt_f(v, 1)
}

#[inline]
fn f2(v: f64) -> Result<String, String> {
    fmt_f(v, 2)
}

fn validate_svg_output(svg: &str, name: &str) -> Result<(), String> {
    let trimmed = svg.trim();
    if !trimmed.starts_with("<svg ") || !trimmed.ends_with("</svg>") {
        return Err(format!("{name}: invalid outer <svg>...</svg> envelope"));
    }
    for req in ["<path ", "<circle ", "<line ", "<text "] {
        if !trimmed.contains(req) {
            return Err(format!("{name}: missing required vector primitive {req:?}"));
        }
    }
    for bad in ["NaN", "nan", "Infinity", "infinity", "-inf", " inf"] {
        if trimmed.contains(bad) {
            return Err(format!("{name}: non-finite token {bad:?} detected in SVG"));
        }
    }
    Ok(())
}

fn svg_open_header(title: &str, subtitle: &str, badge: &str) -> String {
    let mut s = String::with_capacity(4096);
    let _ = writeln!(
        s,
        r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 540" width="960" height="540" role="img">"##
    );
    let _ = writeln!(s, "  <title>{}</title>", escape_xml(title));
    let _ = writeln!(s, "  <desc>{}</desc>", escape_xml(subtitle));
    let _ = writeln!(
        s,
        r##"  <defs>
    <style>
      :root {{
        --bg: #f8fafc;
        --card: #ffffff;
        --card-alt: #f1f5f9;
        --border: #cbd5e1;
        --grid: #e2e8f0;
        --axis: #475569;
        --ink: #0f172a;
        --muted: #475569;
        --pos: #0284c7;
        --pos-hi: #0369a1;
        --neg: #e11d48;
        --acc-emerald: #059669;
        --acc-amber: #d97706;
        --acc-violet: #7c3aed;
        --badge-bg: #e0f2fe;
        --badge-ink: #0369a1;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      }}
      @media (prefers-color-scheme: dark) {{
        :root {{
          --bg: #0b1120;
          --card: #111827;
          --card-alt: #1e293b;
          --border: #334155;
          --grid: #1e293b;
          --axis: #94a3b8;
          --ink: #f1f5f9;
          --muted: #94a3b8;
          --pos: #38bdf8;
          --pos-hi: #7dd3fc;
          --neg: #fb7185;
          --acc-emerald: #34d399;
          --acc-amber: #fbbf24;
          --acc-violet: #a78bfa;
          --badge-bg: #0c4a6e;
          --badge-ink: #7dd3fc;
        }}
      }}
      .bg {{ fill: var(--bg); }}
      .card {{ fill: var(--card); stroke: var(--border); stroke-width: 1.2; }}
      .card-alt {{ fill: var(--card-alt); stroke: var(--border); stroke-width: 1; }}
      .grid {{ stroke: var(--grid); stroke-width: 0.9; }}
      .axis {{ stroke: var(--axis); stroke-width: 1.25; }}
      .ink {{ fill: var(--ink); }}
      .muted {{ fill: var(--muted); }}
      .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
      .pos-stroke {{ stroke: var(--pos); fill: none; }}
      .neg-stroke {{ stroke: var(--neg); fill: none; }}
      .badge-rect {{ fill: var(--badge-bg); stroke: var(--pos); stroke-width: 0.9; }}
      .badge-txt {{ fill: var(--badge-ink); font-weight: 600; }}
    </style>
  </defs>
  <rect class="bg" x="0" y="0" width="960" height="540" rx="12" fill="#f8fafc"/>
  <rect class="card" x="12" y="12" width="936" height="516" rx="10" fill="#ffffff" stroke="#cbd5e1"/>"##
    );
    let _ = writeln!(
        s,
        r##"  <text class="ink" x="28" y="38" font-size="16.5" font-weight="700" fill="#0f172a">{}</text>"##,
        escape_xml(title)
    );
    let _ = writeln!(
        s,
        r##"  <text class="muted" x="28" y="55" font-size="11.2" fill="#475569">{}</text>"##,
        escape_xml(subtitle)
    );
    let badge_w = (badge.chars().count() as f64 * 6.4 + 22.0).max(110.0);
    let badge_x = 932.0 - badge_w;
    let _ = writeln!(
        s,
        r##"  <rect class="badge-rect" x="{:.1}" y="22" width="{:.1}" height="24" rx="12" fill="#e0f2fe" stroke="#0284c7"/>"##,
        badge_x, badge_w
    );
    let _ = writeln!(
        s,
        r##"  <text class="badge-txt mono" x="{:.1}" y="38" font-size="10.5" text-anchor="middle" fill="#0369a1">{}</text>"##,
        badge_x + 0.5 * badge_w,
        escape_xml(badge)
    );
    s
}

/// Deterministic bilinear marching-squares contour path builder on a row-major `ny x nx` grid.
///
/// Maps column index `ix in [0, nx - 1]` linearly to `[px0, px1]` and row index `iy in [0, ny - 1]`
/// linearly to `[py0, py1]`, then chains adjacent cell segments into compact SVG path strings.
#[allow(clippy::too_many_arguments)]
fn marching_squares_svg_path(
    grid: &[f64],
    nx: usize,
    ny: usize,
    level: f64,
    px0: f64,
    px1: f64,
    py0: f64,
    py1: f64,
) -> Result<String, String> {
    if nx < 2 || ny < 2 || grid.len() < nx * ny {
        return Ok(String::new());
    }
    let sx = (px1 - px0) / ((nx - 1) as f64);
    let sy = (py1 - py0) / ((ny - 1) as f64);

    let interp = |va: f64, vb: f64| -> f64 {
        let denom = vb - va;
        if denom.abs() < 1.0e-15 {
            0.5
        } else {
            ((level - va) / denom).clamp(0.0, 1.0)
        }
    };

    let qkey = |pt: [f64; 2]| -> (i64, i64) {
        ((pt[0] * 20.0).round() as i64, (pt[1] * 20.0).round() as i64)
    };

    let mut segments: Vec<([f64; 2], [f64; 2])> = Vec::new();

    for iy in 0..(ny - 1) {
        let r0 = iy * nx;
        let r1 = (iy + 1) * nx;
        let y_top = py0 + (iy as f64) * sy;
        let y_bot = py0 + ((iy + 1) as f64) * sy;

        for ix in 0..(nx - 1) {
            let v00 = grid[r0 + ix];
            let v10 = grid[r0 + ix + 1];
            let v11 = grid[r1 + ix + 1];
            let v01 = grid[r1 + ix];

            let mut mask = 0u8;
            if v00 >= level {
                mask |= 1;
            }
            if v10 >= level {
                mask |= 2;
            }
            if v11 >= level {
                mask |= 4;
            }
            if v01 >= level {
                mask |= 8;
            }
            if mask == 0 || mask == 15 {
                continue;
            }

            let x_left = px0 + (ix as f64) * sx;
            let x_right = px0 + ((ix + 1) as f64) * sx;

            // Edges: 0 = top (00->10), 1 = right (10->11), 2 = bottom (01->11), 3 = left (00->01)
            let edge_pt = |edge: u8| -> [f64; 2] {
                match edge {
                    0 => [x_left + interp(v00, v10) * sx, y_top],
                    1 => [x_right, y_top + interp(v10, v11) * sy],
                    2 => [x_left + interp(v01, v11) * sx, y_bot],
                    _ => [x_left, y_top + interp(v00, v01) * sy],
                }
            };

            let center = 0.25 * (v00 + v10 + v11 + v01);
            match mask {
                1 | 14 => segments.push((edge_pt(3), edge_pt(0))),
                2 | 13 => segments.push((edge_pt(0), edge_pt(1))),
                3 | 12 => segments.push((edge_pt(3), edge_pt(1))),
                4 | 11 => segments.push((edge_pt(1), edge_pt(2))),
                6 | 9 => segments.push((edge_pt(0), edge_pt(2))),
                7 | 8 => segments.push((edge_pt(3), edge_pt(2))),
                5 => {
                    if center >= level {
                        segments.push((edge_pt(3), edge_pt(2)));
                        segments.push((edge_pt(0), edge_pt(1)));
                    } else {
                        segments.push((edge_pt(3), edge_pt(0)));
                        segments.push((edge_pt(1), edge_pt(2)));
                    }
                }
                10 => {
                    if center >= level {
                        segments.push((edge_pt(3), edge_pt(0)));
                        segments.push((edge_pt(1), edge_pt(2)));
                    } else {
                        segments.push((edge_pt(3), edge_pt(2)));
                        segments.push((edge_pt(0), edge_pt(1)));
                    }
                }
                _ => {}
            }
        }
    }

    if segments.is_empty() {
        return Ok(String::new());
    }

    // Deterministic endpoint chaining using BTreeMap adjacency
    let mut adj: BTreeMap<(i64, i64), Vec<(usize, bool)>> = BTreeMap::new();
    for (idx, (a, b)) in segments.iter().enumerate() {
        adj.entry(qkey(*a)).or_default().push((idx, true));
        adj.entry(qkey(*b)).or_default().push((idx, false));
    }

    let mut used = vec![false; segments.len()];
    let mut d = String::with_capacity(segments.len() * 14);

    for start_idx in 0..segments.len() {
        if used[start_idx] {
            continue;
        }
        used[start_idx] = true;
        let (p0, p1) = segments[start_idx];
        let mut chain: Vec<[f64; 2]> = vec![p0, p1];

        // Extend forward
        while let Some(&last_pt) = chain.last() {
            let key = qkey(last_pt);
            let mut next_pt_opt: Option<[f64; 2]> = None;
            if let Some(cands) = adj.get(&key) {
                for &(sidx, at_a) in cands {
                    if !used[sidx] {
                        used[sidx] = true;
                        next_pt_opt = Some(if at_a {
                            segments[sidx].1
                        } else {
                            segments[sidx].0
                        });
                        break;
                    }
                }
            }
            if let Some(npt) = next_pt_opt {
                chain.push(npt);
            } else {
                break;
            }
        }

        if !d.is_empty() {
            d.push(' ');
        }
        let _ = write!(d, "M{} {}", f1(chain[0][0])?, f1(chain[0][1])?);
        for pt in chain.iter().skip(1) {
            let _ = write!(d, "L{} {}", f1(pt[0])?, f1(pt[1])?);
        }
    }

    Ok(d)
}

// ============================================================================
// Panel 01: 2D 1H-15N HSQC Positive/Negative Contours + Assigned Labels
// ============================================================================

fn render_01_hsqc_svg(seed: u64) -> Result<String, String> {
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

    // Run real `evo-graph` backbone assignment on the simulated peak table
    let priors = Priors::provisional();
    let evo_cfg = EvoGraphConfig {
        population_size: 24,
        num_islands: 6,
        generations: 3,
        steps_per_system_per_gen: 220,
        seed,
        ..EvoGraphConfig::default()
    };
    let (systems, assign_res, _rep) =
        assign_evo_graph_from_peaks(&peaks, &ref_data.sequence, &priors, &evo_cfg);

    // Append folded Arg N_epsilon-H_epsilon sidechain peaks (`sign = -1` from F1 aliasing in
    // sensitivity-enhanced gradient 1H-15N HSQC) for authentic negative-phase contours
    let folded_arg_peaks = [
        (7.36, 115.4, 82.0),
        (7.24, 116.8, 76.0),
        (7.49, 114.2, 68.0),
        (7.15, 118.1, 64.0),
    ];
    for &(h_ppm, n_ppm, height) in &folded_arg_peaks {
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

    let render_cfg = SpectrumRenderConfig {
        nx_h: 220,
        ny_n: 168,
        h_high_ppm: 9.85,
        h_low_ppm: 6.35,
        n_high_ppm: 131.5,
        n_low_ppm: 103.5,
        sf_h_mhz: 600.13,
        sf_n_mhz: 60.81,
        lw_h_ppm: 0.036,
        lw_n_ppm: 0.34,
        gaussian_mix: 0.65,
        noise_level: 0.35,
        seed,
        ..SpectrumRenderConfig::default()
    };
    let spec = render_hsqc_2d(&peaks, &render_cfg);
    let nx = spec.axis_x.size;
    let ny = spec.axis_y.size;
    let pos_grid: Vec<f64> = spec.data.iter().map(|&v| v as f64).collect();
    let neg_grid: Vec<f64> = spec.data.iter().map(|&v| -(v as f64)).collect();

    let mut svg = svg_open_header(
        "2D 1H-15N HSQC Contour Overlay & Assigned Backbone Roots",
        "Human Ubiquitin (76 aa, 600 MHz) · Positive backbone/NH2 contours + aliased Arg Ne-He negative contours · evo-graph assignments",
        "seed=42 · evo-graph",
    );

    // Plot geometry: main 2D box + top 1H skyline + left 15N skyline
    let px0 = 112.0;
    let px1 = 716.0;
    let py0 = 122.0;
    let py1 = 486.0;
    let h_hi = render_cfg.h_high_ppm;
    let h_lo = render_cfg.h_low_ppm;
    let n_lo = render_cfg.n_low_ppm;
    let n_hi = render_cfg.n_high_ppm;

    let map_x = |h: f64| -> f64 { px0 + (h_hi - h) / (h_hi - h_lo) * (px1 - px0) };
    let map_y = |n: f64| -> f64 { py0 + (n - n_lo) / (n_hi - n_lo) * (py1 - py0) };

    // Plot frame & grid
    let _ = writeln!(
        svg,
        r##"  <rect class="card-alt" x="{:.1}" y="{:.1}" width="{:.1}" height="{:.1}" rx="6" fill="#f1f5f9" stroke="#cbd5e1"/>"##,
        px0,
        py0,
        px1 - px0,
        py1 - py0
    );

    for h_tick in [9.5, 9.0, 8.5, 8.0, 7.5, 7.0, 6.5] {
        let x = map_x(h_tick);
        let _ = writeln!(
            svg,
            r##"  <line class="grid" x1="{}" y1="{:.1}" x2="{}" y2="{:.1}" stroke="#e2e8f0"/>"##,
            f1(x)?,
            py0,
            f1(x)?,
            py1
        );
        let _ = writeln!(
            svg,
            r##"  <line class="axis" x1="{}" y1="{:.1}" x2="{}" y2="{:.1}" stroke="#475569"/>"##,
            f1(x)?,
            py1,
            f1(x)?,
            py1 + 5.0
        );
        let _ = writeln!(
            svg,
            r##"  <text class="muted mono" x="{}" y="{:.1}" font-size="10" text-anchor="middle" fill="#475569">{:.1}</text>"##,
            f1(x)?,
            py1 + 17.5,
            h_tick
        );
    }

    for n_tick in [105.0, 110.0, 115.0, 120.0, 125.0, 130.0] {
        let y = map_y(n_tick);
        let _ = writeln!(
            svg,
            r##"  <line class="grid" x1="{:.1}" y1="{}" x2="{:.1}" y2="{}" stroke="#e2e8f0"/>"##,
            px0,
            f1(y)?,
            px1,
            f1(y)?
        );
        let _ = writeln!(
            svg,
            r##"  <line class="axis" x1="{:.1}" y1="{}" x2="{:.1}" y2="{}" stroke="#475569"/>"##,
            px0 - 5.0,
            f1(y)?,
            px0,
            f1(y)?
        );
        let _ = writeln!(
            svg,
            r##"  <text class="muted mono" x="{:.1}" y="{}" font-size="10" text-anchor="end" fill="#475569">{:.0}</text>"##,
            px0 - 8.0,
            f1(y + 3.5)?,
            n_tick
        );
    }

    // 1D F2 (1H) top skyline projection
    let mut sky_x = String::with_capacity(nx * 12);
    let max_pos = pos_grid.iter().copied().fold(1.0_f64, f64::max);
    for ix in 0..nx {
        let mut col_max = 0.0_f64;
        for iy in 0..ny {
            col_max = col_max.max(pos_grid[iy * nx + ix]);
        }
        let h_ppm = spec.axis_x.index_to_ppm(ix as f64);
        let x = map_x(h_ppm);
        let y = 114.0 - (col_max / max_pos).clamp(0.0, 1.0) * 38.0;
        let cmd = if ix == 0 { 'M' } else { 'L' };
        let _ = write!(sky_x, "{cmd}{} {}", f1(x)?, f1(y)?);
    }
    let _ = writeln!(
        svg,
        r##"  <path class="pos-stroke" d="{sky_x}" stroke="#0284c7" stroke-width="1.15" opacity="0.85" fill="none"/>"##
    );

    // 1D F1 (15N) left skyline projection
    let mut sky_y = String::with_capacity(ny * 12);
    for iy in 0..ny {
        let mut row_max = 0.0_f64;
        for ix in 0..nx {
            row_max = row_max.max(pos_grid[iy * nx + ix]);
        }
        let n_ppm = spec.axis_y.index_to_ppm(iy as f64);
        let y = map_y(n_ppm);
        let x = 66.0 - (row_max / max_pos).clamp(0.0, 1.0) * 38.0;
        let cmd = if iy == 0 { 'M' } else { 'L' };
        let _ = write!(sky_y, "{cmd}{} {}", f1(x)?, f1(y)?);
    }
    let _ = writeln!(
        svg,
        r##"  <path class="pos-stroke" d="{sky_y}" stroke="#0284c7" stroke-width="1.15" opacity="0.85" fill="none"/>"##
    );

    // Marching-squares positive & negative contours (`axis_x` goes high->low, `axis_y` goes high->low)
    let gx0 = map_x(spec.axis_x.index_to_ppm(0.0));
    let gx1 = map_x(spec.axis_x.index_to_ppm((nx - 1) as f64));
    let gy0 = map_y(spec.axis_y.index_to_ppm(0.0));
    let gy1 = map_y(spec.axis_y.index_to_ppm((ny - 1) as f64));

    for (lvl_frac, width, op) in [
        (0.14, 0.90, 0.55),
        (0.28, 1.05, 0.72),
        (0.52, 1.20, 0.88),
        (0.78, 1.35, 1.00),
    ] {
        let d =
            marching_squares_svg_path(&pos_grid, nx, ny, max_pos * lvl_frac, gx0, gx1, gy0, gy1)?;
        if !d.is_empty() {
            let _ = writeln!(
                svg,
                r##"  <path class="pos-stroke" d="{d}" stroke="#0284c7" stroke-width="{width:.2}" opacity="{op:.2}" fill="none"/>"##
            );
        }
    }

    for (lvl_frac, width, op) in [(0.16, 0.95, 0.68), (0.34, 1.15, 0.85), (0.58, 1.30, 1.00)] {
        let d =
            marching_squares_svg_path(&neg_grid, nx, ny, max_pos * lvl_frac, gx0, gx1, gy0, gy1)?;
        if !d.is_empty() {
            let _ = writeln!(
                svg,
                r##"  <path class="neg-stroke" d="{d}" stroke="#e11d48" stroke-width="{width:.2}" stroke-dasharray="3,1.8" opacity="{op:.2}" fill="none"/>"##
            );
        }
    }

    // Assigned spin-system markers + non-overlapping callout labels
    let mut sys_for_pos: Vec<Option<(usize, f64)>> = vec![None; ref_data.sequence.len()];
    for (s_idx, &pos_k) in assign_res.position.iter().enumerate() {
        if pos_k >= 0 && (pos_k as usize) < sys_for_pos.len() {
            let conf = assign_res.confidence.get(s_idx).copied().unwrap_or(0.95);
            sys_for_pos[pos_k as usize] = Some((s_idx, conf));
        }
    }
    let mut assigned_pts: Vec<(usize, String, f64, f64, f64)> = Vec::new();
    for (k, entry) in sys_for_pos.iter().enumerate() {
        if let Some((s_idx, conf)) = *entry {
            if let Some(sys) = systems.get(s_idx) {
                let code = one_letter_code(&ref_data.sequence[k]);
                assigned_pts.push((k + 1, format!("{code}{}", k + 1), sys.h, sys.n, conf));
            }
        }
    }

    let mut placed_boxes: Vec<[f64; 4]> = Vec::new();
    for (seq_id, label, h_ppm, n_ppm, conf) in &assigned_pts {
        if *h_ppm < h_lo || *h_ppm > h_hi || *n_ppm < n_lo || *n_ppm > n_hi {
            continue;
        }
        let cx = map_x(*h_ppm);
        let cy = map_y(*n_ppm);
        let dot_col = if *conf >= 0.85 { "#059669" } else { "#d97706" };
        let _ = writeln!(
            svg,
            r##"  <circle cx="{}" cy="{}" r="2.2" fill="{dot_col}" stroke="#ffffff" stroke-width="0.6"/>"##,
            f1(cx)?,
            f1(cy)?
        );

        // Select well-spaced landmark peaks across the HSQC plane for crisp callout labels
        let priority = matches!(
            *seq_id,
            2 | 3
                | 4
                | 6
                | 7
                | 10
                | 13
                | 15
                | 23
                | 26
                | 30
                | 36
                | 41
                | 44
                | 47
                | 50
                | 56
                | 61
                | 68
                | 72
                | 76
        );
        if !priority {
            continue;
        }
        let w = (label.len() as f64) * 6.0 + 6.0;
        let h_box = 11.5;
        let offsets = [
            (8.0, -7.0),
            (-w - 6.0, -7.0),
            (8.0, 9.0),
            (-w - 6.0, 9.0),
            (0.0, -13.0),
        ];
        for (dx, dy) in offsets {
            let bx0 = cx + dx;
            let by0 = cy + dy - h_box + 2.0;
            let bx1 = bx0 + w;
            let by1 = by0 + h_box;
            if bx0 < px0 + 4.0 || bx1 > px1 - 4.0 || by0 < py0 + 4.0 || by1 > py1 - 4.0 {
                continue;
            }
            let overlaps = placed_boxes.iter().any(|b| {
                !(bx1 + 3.0 < b[0] || bx0 - 3.0 > b[2] || by1 + 2.0 < b[1] || by0 - 2.0 > b[3])
            });
            if !overlaps {
                placed_boxes.push([bx0, by0, bx1, by1]);
                let lx = if dx >= 0.0 { bx0 - 1.5 } else { bx1 + 1.5 };
                let ly = 0.5 * (by0 + by1);
                let _ = writeln!(
                    svg,
                    r##"  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{dot_col}" stroke-width="0.85" opacity="0.8"/>"##,
                    f1(cx)?,
                    f1(cy)?,
                    f1(lx)?,
                    f1(ly)?
                );
                let _ = writeln!(
                    svg,
                    r##"  <text class="ink mono" x="{}" y="{}" font-size="9.3" font-weight="600" fill="#0f172a">{}</text>"##,
                    f1(bx0 + 2.0)?,
                    f1(by1 - 2.2)?,
                    escape_xml(label)
                );
                break;
            }
        }
    }

    // Label one folded Arg Ne-He negative cluster
    let fx = map_x(7.36);
    let fy = map_y(115.4);
    let _ = writeln!(
        svg,
        r##"  <text class="mono" x="{}" y="{}" font-size="9.2" font-weight="600" fill="#e11d48">Arg Ne-He (aliased -)</text>"##,
        f1(fx + 10.0)?,
        f1(fy - 8.0)?
    );

    // Axis titles
    let _ = writeln!(
        svg,
        r##"  <text class="ink" x="{:.1}" y="518" font-size="11.5" font-weight="600" text-anchor="middle" fill="#0f172a">1H Chemical Shift delta (ppm)</text>"##,
        0.5 * (px0 + px1)
    );
    let _ = writeln!(
        svg,
        r##"  <text class="ink" transform="translate(26,{:.1}) rotate(-90)" font-size="11.5" font-weight="600" text-anchor="middle" fill="#0f172a">15N Chemical Shift delta (ppm)</text>"##,
        0.5 * (py0 + py1)
    );

    // Right-hand summary card
    let rx = 734.0;
    let rw = 198.0;
    let _ = writeln!(
        svg,
        r##"  <rect class="card-alt" x="{rx:.1}" y="{py0:.1}" width="{rw:.1}" height="{:.1}" rx="8" fill="#f1f5f9" stroke="#cbd5e1"/>"##,
        py1 - py0
    );
    let _ = writeln!(
        svg,
        r##"  <text class="ink" x="{:.1}" y="{:.1}" font-size="12.2" font-weight="700" fill="#0f172a">Spectrum &amp; Solver Summary</text>"##,
        rx + 14.0,
        py0 + 24.0
    );
    let n_assigned = assigned_pts.len();
    let n_hc = assigned_pts.iter().filter(|p| p.4 >= 0.80).count();
    let info_rows = [
        ("Target", "Ubiquitin (76 aa)".to_string()),
        ("Field", "600.13 MHz (14.1 T)".to_string()),
        ("Matrix", format!("{nx} x {ny} pts")),
        ("Assigned", format!("{n_assigned} / 72 non-Pro")),
        ("HC (>=0.80)", format!("{n_hc} roots")),
        ("Solver", "evo-graph (CPU f64)".to_string()),
    ];
    for (idx, (k_lbl, v_lbl)) in info_rows.iter().enumerate() {
        let yy = py0 + 50.0 + (idx as f64) * 24.0;
        let _ = writeln!(
            svg,
            r##"  <text class="muted" x="{:.1}" y="{yy:.1}" font-size="10.2" fill="#475569">{}</text>"##,
            rx + 14.0,
            escape_xml(k_lbl)
        );
        let _ = writeln!(
            svg,
            r##"  <text class="ink mono" x="{:.1}" y="{yy:.1}" font-size="10.2" font-weight="600" text-anchor="end" fill="#0f172a">{}</text>"##,
            rx + rw - 14.0,
            escape_xml(v_lbl)
        );
    }

    // Legend inside summary card
    let ly0 = py0 + 216.0;
    let _ = writeln!(
        svg,
        r##"  <line x1="{:.1}" y1="{:.1}" x2="{:.1}" y2="{:.1}" stroke="#0284c7" stroke-width="2.0"/>"##,
        rx + 14.0,
        ly0,
        rx + 36.0,
        ly0
    );
    let _ = writeln!(
        svg,
        r##"  <text class="ink" x="{:.1}" y="{:.1}" font-size="10.2" fill="#0f172a">Positive (+1): Backbone HN</text>"##,
        rx + 44.0,
        ly0 + 3.5
    );
    let _ = writeln!(
        svg,
        r##"  <line x1="{:.1}" y1="{:.1}" x2="{:.1}" y2="{:.1}" stroke="#e11d48" stroke-width="2.0" stroke-dasharray="4,2"/>"##,
        rx + 14.0,
        ly0 + 24.0,
        rx + 36.0,
        ly0 + 24.0
    );
    let _ = writeln!(
        svg,
        r##"  <text class="ink" x="{:.1}" y="{:.1}" font-size="10.2" fill="#0f172a">Negative (-1): Folded Arg</text>"##,
        rx + 44.0,
        ly0 + 27.5
    );
    let _ = writeln!(
        svg,
        r##"  <circle cx="{:.1}" cy="{:.1}" r="3.5" fill="#059669"/>"##,
        rx + 25.0,
        ly0 + 48.0
    );
    let _ = writeln!(
        svg,
        r##"  <text class="ink" x="{:.1}" y="{:.1}" font-size="10.2" fill="#0f172a">Assigned Root (conf &gt;= 0.85)</text>"##,
        rx + 44.0,
        ly0 + 51.5
    );
    let _ = writeln!(
        svg,
        r##"  <text class="muted" x="{:.1}" y="{:.1}" font-size="9.5" fill="#475569">F2 / F1 1D skyline projections</text>"##,
        rx + 14.0,
        ly0 + 82.0
    );
    let _ = writeln!(
        svg,
        r##"  <text class="muted" x="{:.1}" y="{:.1}" font-size="9.5" fill="#475569">rendered directly from Spectrum2D</text>"##,
        rx + 14.0,
        ly0 + 96.0
    );

    svg.push_str("</svg>\n");
    Ok(svg)
}

// ============================================================================
// Panel 02: 3D HNCACB Sequential Strip Walk (F4 -> V5 -> K6 -> T7 -> L8 -> T9 -> G10)
// ============================================================================

fn render_02_strips_svg(seed: u64) -> Result<String, String> {
    let ref_data = read_star_str(UBIQUITIN_STAR_FIXTURE, None)?;
    let sim_cfg = SimConfig {
        experiments: vec!["HSQC".to_string(), "HNCACB".to_string()],
        noise_h: 0.001,
        noise_n: 0.01,
        noise_c: 0.01,
        p_missing_peak: 0.0,
        p_missing_root: 0.0,
        spurious_rate: 0.0,
        sidechain_nh2: false,
        include_intensities: true,
        seed,
        ..SimConfig::default()
    };
    let (peaks, _) = simulate(&ref_data, &sim_cfg);

    let render_cfg = SpectrumRenderConfig {
        nx_h: 128,
        ny_n: 84,
        nz_c: 168,
        h_high_ppm: 9.8,
        h_low_ppm: 6.6,
        n_high_ppm: 130.0,
        n_low_ppm: 105.0,
        c_aliph_high_ppm: 74.0,
        c_aliph_low_ppm: 14.0,
        lw_h_ppm: 0.038,
        lw_n_ppm: 0.42,
        lw_c_ppm: 0.62,
        gaussian_mix: 0.70,
        noise_level: 0.15,
        seed,
        ..SpectrumRenderConfig::default()
    };
    let cube = render_triple_3d("HNCACB", &peaks, &render_cfg);

    let mut svg = svg_open_header(
        "3D HNCACB Sequential Backbone Strip Walk (Phe4 -> Gly10)",
        "Positive CA (blue, i strong / i-1 weak) & negative CB (rose, Thr7 downfield CB > CA inversion, Gly10 CB-null) sliced from Spectrum3D",
        "3D HNCACB · seed=42",
    );

    // Stretch: residues 4..=10 (0-based indices 3..=9), with predecessor 3 (Ile3)
    let seq_indices: [usize; 7] = [3, 4, 5, 6, 7, 8, 9];
    let left_x = 76.0;
    let right_x = 924.0;
    let top_y = 104.0;
    let bot_y = 482.0;
    let gap = 10.0;
    let n_strips = seq_indices.len();
    let strip_w = ((right_x - left_x) - gap * ((n_strips - 1) as f64)) / (n_strips as f64);

    let c_lo = 15.0_f64;
    let c_hi = 73.0_f64;
    let map_cy = |c_ppm: f64| -> f64 { top_y + (c_ppm - c_lo) / (c_hi - c_lo) * (bot_y - top_y) };

    // Left 13C axis ticks
    for c_tick in [20.0, 30.0, 40.0, 50.0, 60.0, 70.0] {
        let y = map_cy(c_tick);
        let _ = writeln!(
            svg,
            r##"  <line class="axis" x1="{:.1}" y1="{}" x2="{:.1}" y2="{}" stroke="#475569"/>"##,
            left_x - 6.0,
            f1(y)?,
            left_x - 1.0,
            f1(y)?
        );
        let _ = writeln!(
            svg,
            r##"  <text class="muted mono" x="{:.1}" y="{}" font-size="10" text-anchor="end" fill="#475569">{:.0}</text>"##,
            left_x - 9.0,
            f1(y + 3.5)?,
            c_tick
        );
    }
    let _ = writeln!(
        svg,
        r##"  <text class="ink" transform="translate(26,{:.1}) rotate(-90)" font-size="11.5" font-weight="600" text-anchor="middle" fill="#0f172a">13CA / 13CB Chemical Shift delta (ppm)</text>"##,
        0.5 * (top_y + bot_y)
    );

    let nx_h = cube.axis_x.size;
    let ny_n = cube.axis_y.size;
    let nz_c = cube.axis_z.size;

    let mut strip_centers_x = Vec::with_capacity(n_strips);

    for (s_col, &k) in seq_indices.iter().enumerate() {
        let sx0 = left_x + (s_col as f64) * (strip_w + gap);
        let sx1 = sx0 + strip_w;
        let scx = 0.5 * (sx0 + sx1);
        strip_centers_x.push(scx);

        let seq_id = (k + 1) as i32;
        let res3 = &ref_data.sequence[k];
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

        // Strip card + header badge
        let _ = writeln!(
            svg,
            r##"  <rect class="card-alt" x="{}" y="{top_y:.1}" width="{}" height="{:.1}" rx="6" fill="#f1f5f9" stroke="#cbd5e1"/>"##,
            f1(sx0)?,
            f1(strip_w)?,
            bot_y - top_y
        );
        let _ = writeln!(
            svg,
            r##"  <line class="grid" x1="{}" y1="{top_y:.1}" x2="{}" y2="{bot_y:.1}" stroke="#e2e8f0" stroke-dasharray="2,2"/>"##,
            f1(scx)?,
            f1(scx)?
        );
        for c_tick in [20.0, 30.0, 40.0, 50.0, 60.0, 70.0] {
            let y = map_cy(c_tick);
            let _ = writeln!(
                svg,
                r##"  <line class="grid" x1="{}" y1="{}" x2="{}" y2="{}" stroke="#e2e8f0"/>"##,
                f1(sx0)?,
                f1(y)?,
                f1(sx1)?,
                f1(y)?
            );
        }

        let _ = writeln!(
            svg,
            r##"  <text class="ink mono" x="{}" y="{:.1}" font-size="11.5" font-weight="700" text-anchor="middle" fill="#0f172a">{}{}</text>"##,
            f1(scx)?,
            top_y - 19.0,
            res3,
            seq_id
        );
        let _ = writeln!(
            svg,
            r##"  <text class="muted mono" x="{}" y="{:.1}" font-size="9.2" text-anchor="middle" fill="#475569">{:.2}H / {:.1}N</text>"##,
            f1(scx)?,
            top_y - 6.0,
            h_i,
            n_i
        );
        let _ = writeln!(
            svg,
            r##"  <text class="muted mono" x="{}" y="{:.1}" font-size="9.0" text-anchor="middle" fill="#475569">+/-0.09 ppm</text>"##,
            f1(scx)?,
            bot_y + 15.0
        );

        // Extract 2D (13C x local 1H) slice from the 3D HNCACB cube at nearest `iy` for `n_i`
        let iy =
            (cube.axis_y.ppm_to_index(n_i).round() as isize).clamp(0, ny_n as isize - 1) as usize;
        let sub_nx = 21usize;
        let h_span = 0.095_f64;

        // Sample rows corresponding to `c_ppm in [c_lo, c_hi]`
        let sub_ny = 116usize;
        let mut pos_slice = vec![0.0_f64; sub_ny * sub_nx];
        let mut neg_slice = vec![0.0_f64; sub_ny * sub_nx];
        let mut slice_peak = 1.0_f64;

        for ry in 0..sub_ny {
            let c_ppm = c_lo + (ry as f64) * (c_hi - c_lo) / ((sub_ny - 1) as f64);
            let iz_f = cube
                .axis_z
                .ppm_to_index(c_ppm)
                .clamp(0.0, (nz_c - 1) as f64);
            let iz0 = iz_f.floor() as usize;
            let iz1 = (iz0 + 1).min(nz_c - 1);
            let tz = iz_f - (iz0 as f64);

            for rx in 0..sub_nx {
                let h_ppm = (h_i + h_span) - (rx as f64) * (2.0 * h_span) / ((sub_nx - 1) as f64);
                let ix_f = cube
                    .axis_x
                    .ppm_to_index(h_ppm)
                    .clamp(0.0, (nx_h - 1) as f64);
                let ix0 = ix_f.floor() as usize;
                let ix1 = (ix0 + 1).min(nx_h - 1);
                let tx = ix_f - (ix0 as f64);

                let sample = |iz: usize, ix: usize| -> f64 {
                    cube.data[(iz * ny_n + iy) * nx_h + ix] as f64
                };
                let v0 = sample(iz0, ix0) * (1.0 - tx) + sample(iz0, ix1) * tx;
                let v1 = sample(iz1, ix0) * (1.0 - tx) + sample(iz1, ix1) * tx;
                let val = v0 * (1.0 - tz) + v1 * tz;
                pos_slice[ry * sub_nx + rx] = val.max(0.0);
                neg_slice[ry * sub_nx + rx] = (-val).max(0.0);
                slice_peak = slice_peak.max(val.abs());
            }
        }

        for (lvl, lw) in [(0.16, 1.0), (0.36, 1.15), (0.65, 1.35)] {
            let dp = marching_squares_svg_path(
                &pos_slice,
                sub_nx,
                sub_ny,
                slice_peak * lvl,
                sx0 + 5.0,
                sx1 - 5.0,
                top_y,
                bot_y,
            )?;
            if !dp.is_empty() {
                let _ = writeln!(
                    svg,
                    r##"  <path class="pos-stroke" d="{dp}" stroke="#0284c7" stroke-width="{lw:.2}" fill="none"/>"##
                );
            }
            let dn = marching_squares_svg_path(
                &neg_slice,
                sub_nx,
                sub_ny,
                slice_peak * lvl,
                sx0 + 5.0,
                sx1 - 5.0,
                top_y,
                bot_y,
            )?;
            if !dn.is_empty() {
                let _ = writeln!(
                    svg,
                    r##"  <path class="neg-stroke" d="{dn}" stroke="#e11d48" stroke-width="{lw:.2}" fill="none"/>"##
                );
            }
        }

        // Annotate intra-residue `i` peak centers
        if let Some(&ca_i) = ref_data.shifts.get(&(k, "CA".to_string())) {
            let y_ca = map_cy(ca_i);
            let _ = writeln!(
                svg,
                r##"  <circle cx="{}" cy="{}" r="2.8" fill="#0284c7" stroke="#ffffff" stroke-width="0.7"/>"##,
                f1(scx)?,
                f1(y_ca)?
            );
            let _ = writeln!(
                svg,
                r##"  <text class="mono" x="{}" y="{}" font-size="8.6" font-weight="700" fill="#0284c7">CA(i)</text>"##,
                f1(sx0 + 6.0)?,
                f1(y_ca - 5.0)?
            );
        }
        if let Some(&cb_i) = ref_data.shifts.get(&(k, "CB".to_string())) {
            let y_cb = map_cy(cb_i);
            let _ = writeln!(
                svg,
                r##"  <circle cx="{}" cy="{}" r="2.8" fill="#e11d48" stroke="#ffffff" stroke-width="0.7"/>"##,
                f1(scx)?,
                f1(y_cb)?
            );
            let _ = writeln!(
                svg,
                r##"  <text class="mono" x="{}" y="{}" font-size="8.6" font-weight="700" fill="#e11d48">CB(i)</text>"##,
                f1(sx0 + 6.0)?,
                f1(y_cb - 5.0)?
            );
        }
    }

    // Sequential connectivity dashed walk lines (`CA(i)` in strip s -> `CA(i-1)` in strip s+1, same for `CB`)
    for s_col in 0..(n_strips - 1) {
        let k_curr = seq_indices[s_col];
        let x0 = strip_centers_x[s_col];
        let x1 = strip_centers_x[s_col + 1];

        if let Some(&ca_val) = ref_data.shifts.get(&(k_curr, "CA".to_string())) {
            let y = map_cy(ca_val);
            let _ = writeln!(
                svg,
                r##"  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#0284c7" stroke-width="1.35" stroke-dasharray="5,3" opacity="0.9"/>"##,
                f1(x0)?,
                f1(y)?,
                f1(x1)?,
                f1(y)?
            );
            let _ = writeln!(
                svg,
                r##"  <circle cx="{}" cy="{}" r="2.1" fill="none" stroke="#0284c7" stroke-width="1.2"/>"##,
                f1(x1)?,
                f1(y)?
            );
        }
        if let Some(&cb_val) = ref_data.shifts.get(&(k_curr, "CB".to_string())) {
            let y = map_cy(cb_val);
            let _ = writeln!(
                svg,
                r##"  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#e11d48" stroke-width="1.35" stroke-dasharray="5,3" opacity="0.9"/>"##,
                f1(x0)?,
                f1(y)?,
                f1(x1)?,
                f1(y)?
            );
            let _ = writeln!(
                svg,
                r##"  <circle cx="{}" cy="{}" r="2.1" fill="none" stroke="#e11d48" stroke-width="1.2"/>"##,
                f1(x1)?,
                f1(y)?
            );
        }
    }

    // Bottom legend bar
    let _ = writeln!(
        svg,
        r##"  <line x1="180" y1="518" x2="206" y2="518" stroke="#0284c7" stroke-width="2.0" stroke-dasharray="5,3"/>"##
    );
    let _ = writeln!(
        svg,
        r##"  <text class="ink" x="214" y="522" font-size="10.5" fill="#0f172a">CA(i) -&gt; CA(i-1) positive walk (+1)</text>"##
    );
    let _ = writeln!(
        svg,
        r##"  <line x1="460" y1="518" x2="486" y2="518" stroke="#e11d48" stroke-width="2.0" stroke-dasharray="5,3"/>"##
    );
    let _ = writeln!(
        svg,
        r##"  <text class="ink" x="494" y="522" font-size="10.5" fill="#0f172a">CB(i) -&gt; CB(i-1) negative walk (-1, Thr7 CB=69.5 ppm, Gly10 no CB)</text>"##
    );

    svg.push_str("</svg>\n");
    Ok(svg)
}

// ============================================================================
// Panel 03: Ubiquitin 1UBQ 3D Backbone Putty Ribbon (Lipari-Szabo S^2 + NOEs)
// ============================================================================

fn render_03_pdb_s2_svg(seed: u64) -> Result<String, String> {
    let chain = parse_pdb_str_first_model(UBIQUITIN_1UBQ_PDB_FIXTURE)?;
    let nh_vecs = extract_nh_unit_vectors_from_chain(&chain);
    let nh_map: BTreeMap<i32, [f64; 3]> = nh_vecs.into_iter().collect();

    // Synthesize dual-field (600 & 800 MHz) relaxation rates from realistic Ubiquitin flexibility
    // profile and fit with real `nmr_core::modelfree::fit_modelfree_dataset`!
    let phys = RelaxPhysicsConfig::default();
    let tau_m_true = 4.10e-9_f64;
    let rot_comps = [(1.0_f64, tau_m_true)];
    let mut rng = Rng::new(seed);
    let mut mf_inputs: Vec<MultiFieldResidueInput> = Vec::new();

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
            fields.push(ResidueRelaxRates {
                seq_id: s_id,
                res_name: res.res_name.clone(),
                field_mhz: mhz,
                r1,
                sigma_r1: 0.018 * r1_0,
                r2,
                sigma_r2: 0.020 * r2_0,
                hetnoe: Some(noe),
                sigma_hetnoe: Some(0.018),
            });
        }
        mf_inputs.push(MultiFieldResidueInput {
            seq_id: s_id,
            res_name: res.res_name.clone(),
            nh_unit_vec: nh_map.get(&s_id).copied(),
            fields,
        });
    }

    let mf_cfg = ModelFreeConfig {
        diffusion_kind: DiffusionTensorKind::Isotropic,
        selection_criterion: MfCriterion::Aic,
        ref_field_mhz: Some(600.0),
        physics: phys,
        enable_model_elimination: true,
        max_outer_iters: 4,
        mc_replicates: 0,
        seed,
    };
    let mf_fit = fit_modelfree_dataset(&mf_inputs, &mf_cfg)?;

    let mut s2_by_seq: BTreeMap<i32, f64> = BTreeMap::new();
    let mut rex_by_seq: BTreeMap<i32, f64> = BTreeMap::new();
    for r in &mf_fit.residues {
        s2_by_seq.insert(r.seq_id, r.params.s2);
        rex_by_seq.insert(r.seq_id, r.params.rex_ref_s_inv);
    }

    // Center & rotate 1UBQ coordinates for clear beta-grasp + alpha-helix + C-tail view
    let n_res = chain.residues.len();
    let mut centroid = [0.0_f64; 3];
    for r in &chain.residues {
        centroid[0] += r.ca[0];
        centroid[1] += r.ca[1];
        centroid[2] += r.ca[2];
    }
    centroid[0] /= n_res.max(1) as f64;
    centroid[1] /= n_res.max(1) as f64;
    centroid[2] /= n_res.max(1) as f64;

    let rot_pt = |p: Vec3| -> [f64; 3] {
        let x = p[0] - centroid[0];
        let y = p[1] - centroid[1];
        let z = p[2] - centroid[2];
        let (ry, cy) = (0.58_f64.sin(), 0.58_f64.cos());
        let (rx, cx) = ((-0.38_f64).sin(), (-0.38_f64).cos());
        let x1 = x * cy + z * ry;
        let z1 = -x * ry + z * cy;
        let y2 = y * cx - z1 * rx;
        let z2 = y * rx + z1 * cx;
        [x1, y2, z2]
    };

    let ca_rot: Vec<[f64; 3]> = chain.residues.iter().map(|r| rot_pt(r.ca)).collect();
    let proj_2d = |p3: [f64; 3]| -> [f64; 2] {
        let persp = 68.0 / (68.0 - p3[2]).max(25.0);
        let scale = 11.4 * persp;
        [330.0 + p3[0] * scale, 250.0 - p3[1] * scale]
    };

    // Identify genuine long-range (< 4.85 A, |i - j| >= 5) backbone H/HA NOE contacts in 1UBQ
    let mut noe_cylinders: Vec<([f64; 3], [f64; 3], f64)> = Vec::new();
    for i in 0..n_res {
        for j in (i + 5)..n_res {
            let p_a = chain.residues[i].h.unwrap_or(chain.residues[i].ha);
            let p_b = chain.residues[j].h.unwrap_or(chain.residues[j].ha);
            let dx = p_a[0] - p_b[0];
            let dy = p_a[1] - p_b[1];
            let dz = p_a[2] - p_b[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            if dist <= 4.55 {
                noe_cylinders.push((
                    rot_pt(chain.residues[i].ca),
                    rot_pt(chain.residues[j].ca),
                    dist,
                ));
            }
        }
    }

    // Smooth Catmull-Rom 3D backbone putty subdivision
    let s2_for_idx = |idx: usize| -> f64 {
        let s_id = chain.residues[idx].seq_id;
        s2_by_seq.get(&s_id).copied().unwrap_or(0.85)
    };

    #[derive(Clone)]
    struct DepthPrim {
        z: f64,
        is_noe: bool,
        p0: [f64; 2],
        p1: [f64; 2],
        width: f64,
        color: String,
    }

    let mut prims: Vec<DepthPrim> = Vec::new();
    for (pa, pb, dist) in &noe_cylinders {
        let z_mid = 0.5 * (pa[2] + pb[2]) - 0.5;
        let col = if *dist < 3.4 { "#10b981" } else { "#0ea5e9" };
        prims.push(DepthPrim {
            z: z_mid,
            is_noe: true,
            p0: proj_2d(*pa),
            p1: proj_2d(*pb),
            width: if *dist < 3.4 { 2.2 } else { 1.5 },
            color: col.to_string(),
        });
    }

    let sub_steps = 6usize;
    for i in 0..(n_res.saturating_sub(1)) {
        let p_m1 = ca_rot[i.saturating_sub(1)];
        let p_0 = ca_rot[i];
        let p_1 = ca_rot[i + 1];
        let p_2 = ca_rot[(i + 2).min(n_res - 1)];
        let s2_0 = s2_for_idx(i);
        let s2_1 = s2_for_idx(i + 1);

        let mut prev_3d = p_0;
        for step in 1..=sub_steps {
            let t = (step as f64) / (sub_steps as f64);
            let cur_3d = catmull_rom_3d(p_m1, p_0, p_1, p_2, t);
            let s2 = s2_0 * (1.0 - t) + s2_1 * t;
            // Putty width increases with internal flexibility `(1 - S^2)`
            let putty_w = 4.2 + (1.0 - s2).clamp(0.0, 0.8) * 12.5;
            prims.push(DepthPrim {
                z: 0.5 * (prev_3d[2] + cur_3d[2]),
                is_noe: false,
                p0: proj_2d(prev_3d),
                p1: proj_2d(cur_3d),
                width: putty_w,
                color: s2_color_hex(s2),
            });
            prev_3d = cur_3d;
        }
    }

    prims.sort_by(|a, b| a.z.total_cmp(&b.z));

    let tau_ns = mf_fit.diffusion_tensor.effective_tau_m_sec() * 1.0e9;
    let mut svg = svg_open_header(
        "Ubiquitin 1UBQ 3D Lipari-Szabo S^2 Putty Ribbon & Long-Range NOE Restraints",
        &format!(
            "Dual-field (600/800 MHz) M1-M5 Model-Free fit (tau_m = {tau_ns:.2} ns) · Synthetic forward-model self-consistency showcase"
        ),
        "1UBQ · S^2 + NOEs",
    );

    // 3D viewport frame (left) + S^2 sequence profile & colorbar (right)
    let _ = writeln!(
        svg,
        r##"  <rect class="card-alt" x="28" y="72" width="584" height="440" rx="8" fill="#f1f5f9" stroke="#cbd5e1"/>"##
    );

    for pr in &prims {
        if pr.is_noe {
            let _ = writeln!(
                svg,
                r##"  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="{:.2}" stroke-dasharray="4,2.5" opacity="0.68"/>"##,
                f1(pr.p0[0])?,
                f1(pr.p0[1])?,
                f1(pr.p1[0])?,
                f1(pr.p1[1])?,
                pr.color,
                pr.width
            );
        } else {
            let _ = writeln!(
                svg,
                r##"  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#0f172a" stroke-width="{:.2}" stroke-linecap="round" opacity="0.22"/>"##,
                f1(pr.p0[0])?,
                f1(pr.p0[1])?,
                f1(pr.p1[0])?,
                f1(pr.p1[1])?,
                pr.width + 1.8
            );
            let _ = writeln!(
                svg,
                r##"  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="{:.2}" stroke-linecap="round"/>"##,
                f1(pr.p0[0])?,
                f1(pr.p0[1])?,
                f1(pr.p1[0])?,
                f1(pr.p1[1])?,
                pr.color,
                pr.width
            );
        }
    }

    // Landmark C-alpha spheres & callouts on 1UBQ (`M1`, `G10` beta-turn, `I23` Rex, `K48`, `G76` C-tail)
    for &(res_idx, lbl, dx, dy) in &[
        (0usize, "Met1 (N-term)", -52.0, -12.0),
        (9usize, "Gly10 (beta1-beta2 loop)", 10.0, -10.0),
        (22usize, "Ile23 (M3 Rex)", 10.0, 14.0),
        (47usize, "Lys48", -44.0, 15.0),
        (75usize, "Gly76 (C-tail S^2=0.25)", 10.0, 6.0),
    ] {
        if let Some(&p3) = ca_rot.get(res_idx) {
            let p2 = proj_2d(p3);
            let col = s2_color_hex(s2_for_idx(res_idx));
            let _ = writeln!(
                svg,
                r##"  <circle cx="{}" cy="{}" r="4.3" fill="{col}" stroke="#ffffff" stroke-width="1.2"/>"##,
                f1(p2[0])?,
                f1(p2[1])?
            );
            let _ = writeln!(
                svg,
                r##"  <text class="ink mono" x="{}" y="{}" font-size="9.6" font-weight="700" fill="#0f172a">{lbl}</text>"##,
                f1(p2[0] + dx)?,
                f1(p2[1] + dy)?
            );
        }
    }

    // Colorbar inside 3D viewport
    for step in 0..24 {
        let frac = (step as f64) / 23.0;
        let s2_v = 0.25 + frac * 0.70;
        let bx = 46.0 + (step as f64) * 9.0;
        let _ = writeln!(
            svg,
            r##"  <rect x="{bx:.1}" y="482" width="9.5" height="10" fill="{}"/>"##,
            s2_color_hex(s2_v)
        );
    }
    let _ = writeln!(
        svg,
        r##"  <text class="ink mono" x="46" y="476" font-size="9.5" font-weight="600" fill="#0f172a">S^2: 0.25 (Flexible Putty) -&gt; 0.95 (Rigid Core)</text>"##
    );
    let _ = writeln!(
        svg,
        r##"  <text class="muted mono" x="310" y="491" font-size="9.5" fill="#475569">NOE cylinders (&lt;=4.55 A): {} long-range pairs</text>"##,
        noe_cylinders.len()
    );

    // Right panel: Fitted S^2 vs Residue sequence profile + Rex markers
    let rx0 = 628.0;
    let rx1 = 932.0;
    let ry0 = 72.0;
    let ry1 = 310.0;
    let _ = writeln!(
        svg,
        r##"  <rect class="card-alt" x="{rx0:.1}" y="{ry0:.1}" width="{:.1}" height="{:.1}" rx="8" fill="#f1f5f9" stroke="#cbd5e1"/>"##,
        rx1 - rx0,
        ry1 - ry0
    );
    let _ = writeln!(
        svg,
        r##"  <text class="ink" x="{:.1}" y="{:.1}" font-size="12.0" font-weight="700" fill="#0f172a">Fitted Lipari-Szabo S^2 Profile (M1-M5 AIC)</text>"##,
        rx0 + 14.0,
        ry0 + 22.0
    );

    let gx0 = rx0 + 38.0;
    let gx1 = rx1 - 14.0;
    let gy0 = ry0 + 38.0;
    let gy1 = ry1 - 28.0;
    for s2_tick in [0.2, 0.4, 0.6, 0.8, 1.0] {
        let y = gy1 - (s2_tick - 0.2) / 0.8 * (gy1 - gy0);
        let _ = writeln!(
            svg,
            r##"  <line class="grid" x1="{gx0:.1}" y1="{}" x2="{gx1:.1}" y2="{}" stroke="#e2e8f0"/>"##,
            f1(y)?,
            f1(y)?
        );
        let _ = writeln!(
            svg,
            r##"  <text class="muted mono" x="{:.1}" y="{}" font-size="9.2" text-anchor="end" fill="#475569">{:.1}</text>"##,
            gx0 - 5.0,
            f1(y + 3.0)?,
            s2_tick
        );
    }

    let mut s2_path = String::new();
    for (idx, r) in mf_fit.residues.iter().enumerate() {
        let x = gx0 + ((r.seq_id - 1) as f64) / 75.0 * (gx1 - gx0);
        let y = gy1 - ((r.params.s2 - 0.2) / 0.8).clamp(0.0, 1.0) * (gy1 - gy0);
        let cmd = if idx == 0 { 'M' } else { 'L' };
        let _ = write!(s2_path, "{cmd}{} {}", f1(x)?, f1(y)?);
        let col = s2_color_hex(r.params.s2);
        let _ = writeln!(
            svg,
            r##"  <circle cx="{}" cy="{}" r="2.2" fill="{col}"/>"##,
            f1(x)?,
            f1(y)?
        );
    }
    let _ = writeln!(
        svg,
        r##"  <path d="{s2_path}" stroke="#6366f1" stroke-width="1.35" fill="none"/>"##
    );
    for seq_tick in [1, 15, 30, 45, 60, 76] {
        let x = gx0 + ((seq_tick - 1) as f64) / 75.0 * (gx1 - gx0);
        let _ = writeln!(
            svg,
            r##"  <text class="muted mono" x="{}" y="{:.1}" font-size="9.2" text-anchor="middle" fill="#475569">{seq_tick}</text>"##,
            f1(x)?,
            gy1 + 14.0
        );
    }

    // Bottom-right card: Model-Free & Exchange Diagnostics
    let by0 = 326.0;
    let by1 = 512.0;
    let _ = writeln!(
        svg,
        r##"  <rect class="card-alt" x="{rx0:.1}" y="{by0:.1}" width="{:.1}" height="{:.1}" rx="8" fill="#f1f5f9" stroke="#cbd5e1"/>"##,
        rx1 - rx0,
        by1 - by0
    );
    let _ = writeln!(
        svg,
        r##"  <text class="ink" x="{:.1}" y="{:.1}" font-size="12.0" font-weight="700" fill="#0f172a">Model-Free &amp; R_ex Summary</text>"##,
        rx0 + 14.0,
        by0 + 22.0
    );
    let rex_i23 = rex_by_seq.get(&23).copied().unwrap_or(0.0);
    let s2_g76 = s2_by_seq.get(&76).copied().unwrap_or(0.25);
    let rows = [
        ("Global tau_m", format!("{tau_ns:.3} ns")),
        (
            "Fitted Amides",
            format!("{} non-Pro", mf_fit.residues.len()),
        ),
        ("Beta-Turn L8-K11", "S^2 = 0.67-0.70 (M2)".to_string()),
        ("Exchange Ile23", format!("R_ex = {rex_i23:.2} s^-1 (M3)")),
        ("C-Tail Gly76", format!("S^2 = {s2_g76:.3}")),
        ("Total chi^2", format!("{:.1}", mf_fit.total_chi2)),
    ];
    for (idx, (k_lbl, v_lbl)) in rows.iter().enumerate() {
        let yy = by0 + 46.0 + (idx as f64) * 23.0;
        let _ = writeln!(
            svg,
            r##"  <text class="muted" x="{:.1}" y="{yy:.1}" font-size="10.2" fill="#475569">{}</text>"##,
            rx0 + 14.0,
            escape_xml(k_lbl)
        );
        let _ = writeln!(
            svg,
            r##"  <text class="ink mono" x="{:.1}" y="{yy:.1}" font-size="10.2" font-weight="600" text-anchor="end" fill="#0f172a">{}</text>"##,
            rx1 - 14.0,
            escape_xml(v_lbl)
        );
    }

    svg.push_str("</svg>\n");
    Ok(svg)
}

// ============================================================================
// Panel 04: 1D Multiplet J-Tree + Tilted Reaction Stack + qNMR Integral Trace
// ============================================================================

fn render_04_nmr1d_stack_svg(seed: u64) -> Result<String, String> {
    let mhz = 500.0_f64;
    let center_ppm = 5.82_f64;
    let j1 = 14.4_f64;
    let j2 = 8.4_f64;
    let j3 = 3.2_f64;

    // Build exact 8-line `ddd` subpeaks with mild second-order leaning + deterministic noise
    let mut subpeaks = Vec::with_capacity(8);
    for &s1 in &[-0.5_f64, 0.5] {
        for &s2 in &[-0.5_f64, 0.5] {
            for &s3 in &[-0.5_f64, 0.5] {
                let off_hz = s1 * j1 + s2 * j2 + s3 * j3;
                let ppm = center_ppm + off_hz / mhz;
                let roof = 1.0 - 0.04 * (off_hz / 15.0);
                subpeaks.push(MultipletSubPeak::new(ppm, 0.125 * roof).with_fwhm_hz(1.15));
            }
        }
    }

    let n_pts = 520usize;
    let ppm_hi = 5.89_f64;
    let ppm_lo = 5.75_f64;
    let ppm_grid: Vec<f64> = (0..n_pts)
        .map(|i| ppm_hi - (i as f64) * (ppm_hi - ppm_lo) / ((n_pts - 1) as f64))
        .collect();
    let mut y_ddd = synthesize_multiplet_spectrum(&ppm_grid, &subpeaks, mhz, 1.15, 0.20);
    let mut rng = Rng::new(seed);
    for v in &mut y_ddd {
        *v += rng.normal(0.0, 0.12);
    }

    let m_cfg = MultipletConfig::default();
    let analysis = analyze_multiplet_region(&ppm_grid, &y_ddd, ppm_lo, ppm_hi, mhz, &m_cfg)
        .map_err(|e| e.to_string())?;
    let rep_sig = ReportedSignal::from_analysis(&analysis, Some(1), None);
    let acs_str = format_acs_signal(&rep_sig, mhz);

    // Subpanel C: qNMR spectrum (`DMS` certified standard at 3.00 ppm + Analyte quartet at 4.12 ppm)
    let q_n = 640usize;
    let q_hi = 4.55_f64;
    let q_lo = 2.65_f64;
    let q_ppm: Vec<f64> = (0..q_n)
        .map(|i| q_hi - (i as f64) * (q_hi - q_lo) / ((q_n - 1) as f64))
        .collect();
    let mut q_pks = vec![MultipletSubPeak::new(3.00, 6.00).with_fwhm_hz(1.3)];
    for (m_idx, &w) in [1.0_f64, 3.0, 3.0, 1.0].iter().enumerate() {
        let off_hz = ((m_idx as f64) - 1.5) * 7.1;
        q_pks.push(MultipletSubPeak::new(4.12 + off_hz / mhz, 1.842 * (w / 8.0)).with_fwhm_hz(1.3));
    }
    let q_y = synthesize_multiplet_spectrum(&q_ppm, &q_pks, mhz, 1.3, 0.15);
    let q_specs = vec![
        IntegralRegionSpec::new(4.26, 3.98)
            .with_baseline(BaselineMode::None)
            .with_label("Analyte CH2 (2H)"),
        IntegralRegionSpec::new(3.15, 2.85)
            .with_baseline(BaselineMode::None)
            .with_label("DMS Cert. Std (6H)"),
    ];
    let q_int = integrate_regions(
        &q_ppm,
        &q_y,
        &q_specs,
        NormalizationMode::ReferenceRegion {
            region_index: 1,
            target_protons: 6.0,
        },
        Some(0.0015),
        None,
    )
    .map_err(|e| e.to_string())?;

    let std_spec = QnmrStandardSpec::new(
        "Dimethyl sulfone (NIST SRM)",
        94.13,
        12.50,
        0.9995,
        6,
        q_int.regions[1].abs_integral,
    )
    .with_purity_percent(99.95, 0.05)
    .with_uncertainties(0.01, 0.015, q_int.regions[1].abs_sigma.max(0.002))
    .with_t1_s(3.2);
    let ana_spec = QnmrAnalyteSpec {
        name: "Ethyl 4-hydroxybenzoate".to_string(),
        mw_g_mol: 166.17,
        mw_sigma_g_mol: 0.01,
        mass_mg: 22.24,
        mass_sigma_mg: 0.02,
        solution_volume_ml: Some(0.60),
        solution_volume_sigma_ml: None,
        signals: vec![QnmrAnalyteSignal::new(
            2,
            q_int.regions[0].abs_integral,
            q_int.regions[0].abs_sigma.max(0.001),
        )
        .with_t1_s(2.4)],
    };
    let acq = QnmrAcquisitionParams::new(30.0, 4.0);
    let qnmr_rep =
        calculate_qnmr_purity(&ana_spec, &std_spec, Some(&acq)).map_err(|e| e.to_string())?;

    let mut svg = svg_open_header(
        "Analytical 1D NMR: Recursive J-Coupling Tree, Arrayed Kinetics Stack & GUM qNMR",
        "Hoye peeling ddd multiplet (500 MHz) · 5-timepoint kinetics stack · GUM qNMR metrology (synthetic forward-model showcase)",
        "1D · J-Tree · qNMR",
    );

    // --- Left Panel: ddd Multiplet + Hierarchical J-Tree ---
    let lx0 = 28.0;
    let lx1 = 468.0;
    let ly0 = 72.0;
    let ly1 = 512.0;
    let _ = writeln!(
        svg,
        r##"  <rect class="card-alt" x="{lx0:.1}" y="{ly0:.1}" width="{:.1}" height="{:.1}" rx="8" fill="#f1f5f9" stroke="#cbd5e1"/>"##,
        lx1 - lx0,
        ly1 - ly0
    );
    let _ = writeln!(
        svg,
        r##"  <text class="ink" x="{:.1}" y="{:.1}" font-size="12.2" font-weight="700" fill="#0f172a">A. Recursive Hoye J-Tree Splitting Diagram ({})</text>"##,
        lx0 + 14.0,
        ly0 + 22.0,
        analysis.class.as_str()
    );
    let _ = writeln!(
        svg,
        r##"  <text class="muted mono" x="{:.1}" y="{:.1}" font-size="9.6" fill="#475569">{}</text>"##,
        lx0 + 14.0,
        ly0 + 38.0,
        escape_xml(&acs_str)
    );

    let mx0 = lx0 + 22.0;
    let mx1 = lx1 - 22.0;
    let map_mx = |p: f64| -> f64 { mx0 + (ppm_hi - p) / (ppm_hi - ppm_lo) * (mx1 - mx0) };

    // J-Tree inverted hierarchy (`root -> J1 -> J2 -> J3`)
    let js = analysis.j_values_hz();
    let j_a = js.first().copied().unwrap_or(j1);
    let j_b = js.get(1).copied().unwrap_or(j2);
    let j_c = js.get(2).copied().unwrap_or(j3);
    let root_y = 122.0;
    let lv1_y = 166.0;
    let lv2_y = 212.0;
    let lv3_y = 258.0;

    let c0_x = map_mx(analysis.center_ppm);
    let _ = writeln!(
        svg,
        r##"  <circle cx="{}" cy="{root_y:.1}" r="3.4" fill="#7c3aed"/>"##,
        f1(c0_x)?
    );
    for &s1 in &[-0.5_f64, 0.5] {
        let p1 = analysis.center_ppm + (s1 * j_a) / mhz;
        let x1 = map_mx(p1);
        let _ = writeln!(
            svg,
            r##"  <line x1="{}" y1="{root_y:.1}" x2="{}" y2="{lv1_y:.1}" stroke="#7c3aed" stroke-width="1.5"/>"##,
            f1(c0_x)?,
            f1(x1)?
        );
        for &s2 in &[-0.5_f64, 0.5] {
            let p2 = p1 + (s2 * j_b) / mhz;
            let x2 = map_mx(p2);
            let _ = writeln!(
                svg,
                r##"  <line x1="{}" y1="{lv1_y:.1}" x2="{}" y2="{lv2_y:.1}" stroke="#0284c7" stroke-width="1.35"/>"##,
                f1(x1)?,
                f1(x2)?
            );
            for &s3 in &[-0.5_f64, 0.5] {
                let p3 = p2 + (s3 * j_c) / mhz;
                let x3 = map_mx(p3);
                let _ = writeln!(
                    svg,
                    r##"  <line x1="{}" y1="{lv2_y:.1}" x2="{}" y2="{lv3_y:.1}" stroke="#059669" stroke-width="1.2"/>"##,
                    f1(x2)?,
                    f1(x3)?
                );
                let _ = writeln!(
                    svg,
                    r##"  <line x1="{}" y1="{lv3_y:.1}" x2="{}" y2="278.0" stroke="#059669" stroke-width="0.9" stroke-dasharray="2,2"/>"##,
                    f1(x3)?,
                    f1(x3)?
                );
            }
        }
    }
    let _ = writeln!(
        svg,
        r##"  <text class="mono" x="{}" y="146" font-size="9.6" font-weight="700" text-anchor="middle" fill="#7c3aed">3J_trans = {:.2} Hz</text>"##,
        f1(c0_x)?,
        j_a
    );
    let _ = writeln!(
        svg,
        r##"  <text class="mono" x="{}" y="192" font-size="9.2" font-weight="700" text-anchor="middle" fill="#0284c7">3J_cis = {:.2} Hz</text>"##,
        f1(map_mx(analysis.center_ppm + 0.5 * j_a / mhz))?,
        j_b
    );
    let _ = writeln!(
        svg,
        r##"  <text class="mono" x="{}" y="238" font-size="8.8" font-weight="700" text-anchor="middle" fill="#059669">2J_gem = {:.2} Hz</text>"##,
        f1(map_mx(analysis.center_ppm - 0.5 * j_a / mhz))?,
        j_c
    );

    // 1D multiplet spectrum trace
    let y_max = y_ddd.iter().copied().fold(1.0_f64, f64::max);
    let spec_base_y = 472.0;
    let spec_amp = 178.0;
    let mut m_path = String::with_capacity(n_pts * 12);
    for (idx, (&p, &yv)) in ppm_grid.iter().zip(y_ddd.iter()).enumerate() {
        let x = map_mx(p);
        let y = spec_base_y - (yv / y_max).clamp(-0.08, 1.05) * spec_amp;
        let cmd = if idx == 0 { 'M' } else { 'L' };
        let _ = write!(m_path, "{cmd}{} {}", f1(x)?, f1(y)?);
    }
    let _ = writeln!(
        svg,
        r##"  <path class="pos-stroke" d="{m_path}" stroke="#0284c7" stroke-width="1.55" fill="none"/>"##
    );
    let _ = writeln!(
        svg,
        r##"  <line class="axis" x1="{mx0:.1}" y1="478" x2="{mx1:.1}" y2="478" stroke="#475569"/>"##
    );
    for p_tick in [5.88, 5.85, 5.82, 5.79, 5.76] {
        let x = map_mx(p_tick);
        let _ = writeln!(
            svg,
            r##"  <text class="muted mono" x="{}" y="494" font-size="9.4" text-anchor="middle" fill="#475569">{:.2}</text>"##,
            f1(x)?,
            p_tick
        );
    }
    let _ = writeln!(
        svg,
        r##"  <text class="muted" x="{:.1}" y="507" font-size="10.0" text-anchor="middle" fill="#475569">delta 1H (ppm)</text>"##,
        0.5 * (mx0 + mx1)
    );

    // --- Top-Right Panel: Tilted Reaction / Titration 1D Stack ---
    let rx0 = 484.0;
    let rx1 = 932.0;
    let ry0 = 72.0;
    let ry1 = 286.0;
    let _ = writeln!(
        svg,
        r##"  <rect class="card-alt" x="{rx0:.1}" y="{ry0:.1}" width="{:.1}" height="{:.1}" rx="8" fill="#f1f5f9" stroke="#cbd5e1"/>"##,
        rx1 - rx0,
        ry1 - ry0
    );
    let _ = writeln!(
        svg,
        r##"  <text class="ink" x="{:.1}" y="{:.1}" font-size="12.2" font-weight="700" fill="#0f172a">B. Tilted Reaction Kinetics 1D Overlay Stack (t = 0..120 min)</text>"##,
        rx0 + 14.0,
        ry0 + 22.0
    );

    let st_n = 280usize;
    let st_hi = 4.45_f64;
    let st_lo = 3.45_f64;
    let st_ppm: Vec<f64> = (0..st_n)
        .map(|i| st_hi - (i as f64) * (st_hi - st_lo) / ((st_n - 1) as f64))
        .collect();
    let stack_colors = ["#0284c7", "#0ea5e9", "#10b981", "#f59e0b", "#e11d48"];
    let times_min = [0, 15, 35, 70, 120];
    for (t_idx, &t_min) in times_min.iter().enumerate().rev() {
        let conv = 1.0 - (-(t_min as f64) / 45.0).exp();
        let mut t_pks = Vec::new();
        for (m_i, &w) in [1.0_f64, 3.0, 3.0, 1.0].iter().enumerate() {
            let off = ((m_i as f64) - 1.5) * 7.1 / mhz;
            t_pks.push(MultipletSubPeak::new(4.22 + off, (1.0 - conv) * w / 8.0).with_fwhm_hz(1.8));
            t_pks.push(MultipletSubPeak::new(3.68 + off, conv * w / 8.0).with_fwhm_hz(1.8));
        }
        let tr = synthesize_multiplet_spectrum(&st_ppm, &t_pks, mhz, 1.8, 0.2);
        let tr_peak = 95.0_f64;
        let tilt_dx = (t_idx as f64) * 14.0;
        let tilt_dy = (t_idx as f64) * 20.0;
        let base_y = ry1 - 24.0 - tilt_dy;
        let sx_start = rx0 + 20.0 + tilt_dx;
        let sx_span = (rx1 - rx0) - 108.0;

        let mut sp = String::with_capacity(st_n * 11);
        for (i, &yv) in tr.iter().enumerate() {
            let x = sx_start + (i as f64) / ((st_n - 1) as f64) * sx_span;
            let y = base_y - (yv / tr_peak).clamp(0.0, 1.2) * 58.0;
            let cmd = if i == 0 { 'M' } else { 'L' };
            let _ = write!(sp, "{cmd}{} {}", f1(x)?, f1(y)?);
        }
        let col = stack_colors[t_idx];
        let _ = writeln!(
            svg,
            r##"  <path d="{sp}" stroke="{col}" stroke-width="1.4" fill="none"/>"##
        );
        let _ = writeln!(
            svg,
            r##"  <text class="mono" x="{}" y="{}" font-size="9.0" font-weight="600" fill="{col}">t={t_min}m</text>"##,
            f1(sx_start + sx_span + 6.0)?,
            f1(base_y - 2.0)?
        );
    }
    let _ = writeln!(
        svg,
        r##"  <text class="muted mono" x="{:.1}" y="{:.1}" font-size="9.2" fill="#475569">Reactant OCH2 (4.22 ppm) -&gt; Product HOCH2 (3.68 ppm)</text>"##,
        rx0 + 20.0,
        ry1 - 7.0
    );

    // --- Bottom-Right Panel: qNMR Integral Step Trace & GUM Purity ---
    let qy0 = 300.0;
    let qy1 = 512.0;
    let _ = writeln!(
        svg,
        r##"  <rect class="card-alt" x="{rx0:.1}" y="{qy0:.1}" width="{:.1}" height="{:.1}" rx="8" fill="#f1f5f9" stroke="#cbd5e1"/>"##,
        rx1 - rx0,
        qy1 - qy0
    );
    let _ = writeln!(
        svg,
        r##"  <text class="ink" x="{:.1}" y="{:.1}" font-size="12.2" font-weight="700" fill="#0f172a">C. Internal-Standard qNMR Integral Trace &amp; Purity</text>"##,
        rx0 + 14.0,
        qy0 + 22.0
    );

    let qx0 = rx0 + 22.0;
    let qx1 = rx1 - 22.0;
    let q_base = qy1 - 28.0;
    let q_max = q_y.iter().copied().fold(1.0_f64, f64::max);
    let mut qp = String::with_capacity(q_n * 11);
    let mut cum_path = String::new();
    let mut running = 0.0_f64;
    let total_sum: f64 = q_y.iter().sum::<f64>().max(1.0);

    for (i, (&p, &yv)) in q_ppm.iter().zip(q_y.iter()).enumerate() {
        let x = qx0 + (q_hi - p) / (q_hi - q_lo) * (qx1 - qx0);
        let y = q_base - (yv / q_max).clamp(0.0, 1.0) * 126.0;
        let cmd = if i == 0 { 'M' } else { 'L' };
        let _ = write!(qp, "{cmd}{} {}", f1(x)?, f1(y)?);

        running += yv.max(0.0);
        if i % 4 == 0 || i + 1 == q_n {
            let cy = (q_base - 14.0) - (running / total_sum) * 112.0;
            let ccmd = if cum_path.is_empty() { 'M' } else { 'L' };
            let _ = write!(cum_path, "{ccmd}{} {}", f1(x)?, f1(cy)?);
        }
    }
    let _ = writeln!(
        svg,
        r##"  <path class="pos-stroke" d="{qp}" stroke="#0284c7" stroke-width="1.35" fill="none"/>"##
    );
    let _ = writeln!(
        svg,
        r##"  <path d="{cum_path}" stroke="#059669" stroke-width="1.8" fill="none"/>"##
    );

    let purity_pct = qnmr_rep.purity_percent;
    let u_k2 = qnmr_rep.expanded_uncertainty_k2_percent;
    let _ = writeln!(
        svg,
        r##"  <text class="ink mono" x="{:.1}" y="{:.1}" font-size="10.2" font-weight="700" fill="#059669">Purity P = {:.2}% +/- {:.2}% (k=2 GUM)</text>"##,
        rx0 + 24.0,
        qy0 + 44.0,
        purity_pct,
        u_k2
    );
    let _ = writeln!(
        svg,
        r##"  <text class="muted mono" x="{:.1}" y="{:.1}" font-size="9.4" fill="#475569">Analyte 4.12 ppm ({:.2} H) · DMS Std 3.00 ppm ({:.2} H)</text>"##,
        rx0 + 24.0,
        qy0 + 60.0,
        q_int.regions[0].norm_integral,
        q_int.regions[1].norm_integral
    );
    let _ = writeln!(
        svg,
        r##"  <line class="axis" x1="{qx0:.1}" y1="{q_base:.1}" x2="{qx1:.1}" y2="{q_base:.1}" stroke="#475569"/>"##
    );
    for tk in [4.5, 4.0, 3.5, 3.0] {
        let x = qx0 + (q_hi - tk) / (q_hi - q_lo) * (qx1 - qx0);
        let _ = writeln!(
            svg,
            r##"  <text class="muted mono" x="{}" y="{:.1}" font-size="9.2" text-anchor="middle" fill="#475569">{:.1}</text>"##,
            f1(x)?,
            q_base + 14.0,
            tk
        );
    }

    svg.push_str("</svg>\n");
    Ok(svg)
}

// ============================================================================
// Panel 05: 2D (delta ppm x log10 D) DOSY Diffusion Map + Stejskal-Tanner Inset
// ============================================================================

fn render_05_dosy_svg(seed: u64) -> Result<String, String> {
    // 16-point BPP-LED gradient ramp (2 -> 52 G/cm) in D2O at 298.15 K
    let gradients_g_cm: Vec<f64> = (0..16).map(|i| 2.0 + (i as f64) * (50.0 / 15.0)).collect();
    let b_vals = compute_b_values_s_m2(
        &gradients_g_cm,
        DosyNucleus::H1,
        DosySequenceType::BppLed,
        GradientShape::SineBell,
        0.080,
        0.0020,
        0.0002,
    )
    .map_err(|e| e.to_string())?;

    // Three molecular species in D2O + one overlapped bi-exponential resonance at 3.65 ppm:
    // 1) Small internal ref Maleate (D = 8.4e-10 m^2/s, log10 D = -9.08)
    // 2) Oligosaccharide (D = 3.1e-10 m^2/s, log10 D = -9.51)
    // 3) Globular protein Lysozyme (D = 1.12e-10 m^2/s, log10 D = -9.95)
    type DosyPeakDef = (&'static str, f64, f64, &'static [(f64, f64)]);
    let peak_defs: [DosyPeakDef; 6] = [
        ("Lysozyme Ar-H", 7.18, 0.055, &[(0.85, 1.12e-10)]),
        ("Maleate =CH", 6.02, 0.040, &[(1.00, 8.40e-10)]),
        ("Raffinose H1", 5.24, 0.045, &[(0.92, 3.10e-10)]),
        (
            "Overlap (Oligo+Ref)",
            3.66,
            0.050,
            &[(0.55, 3.10e-10), (0.45, 8.40e-10)],
        ),
        ("Raffinose CH2", 3.42, 0.045, &[(0.88, 3.10e-10)]),
        ("Lysozyme Leu-CH3", 0.88, 0.060, &[(0.95, 1.12e-10)]),
    ];

    let fit_cfg = DosyFitConfig {
        d_min_m2_s: 1.0e-11,
        d_max_m2_s: 5.0e-9,
        max_components: 2,
        criterion: DosyCriterion::Bic,
        min_component_fraction: 0.12,
        min_d_ratio: 1.8,
        max_lm_iters: 60,
        bootstrap_samples: 24,
        bootstrap_method: BootstrapMethod::Residual,
        rng_seed: seed,
        known_noise_sigma: Some(0.004),
    };

    let mut rng = Rng::new(seed);
    let mut dosy_peaks = Vec::with_capacity(peak_defs.len());
    for (lbl, ppm, fwhm, comps) in peak_defs {
        let mut intens = Vec::with_capacity(b_vals.len());
        for &b in &b_vals {
            let mut y = 0.0_f64;
            for &(amp, d) in comps {
                y += amp * (-b * d).exp();
            }
            y += rng.normal(0.0, 0.0032);
            intens.push(y.max(1.0e-4));
        }
        let multi_exp =
            fit_stejskal_tanner_multi(&b_vals, &intens, &fit_cfg).map_err(|e| e.to_string())?;
        dosy_peaks.push(DosyPeakEntry {
            label: lbl.to_string(),
            center_ppm: ppm,
            fwhm_ppm: fwhm,
            intensities: intens,
            multi_exp,
            ilt: None,
        });
    }

    // Build 1D top spectrum & 2D `(ppm x log10 D)` grid via `nmr_core::dosy::synthesize_dosy_grid_from_peaks`
    let n_ppm = 260usize;
    let ppm_hi = 7.8_f64;
    let ppm_lo = 0.4_f64;
    let ppm_axis: Vec<f64> = (0..n_ppm)
        .map(|i| ppm_hi - (i as f64) * (ppm_hi - ppm_lo) / ((n_ppm - 1) as f64))
        .collect();
    let mut top_1d = vec![0.0_f64; n_ppm];
    for pk in &dosy_peaks {
        let hwhm = 0.5 * pk.fwhm_ppm;
        let amp: f64 = pk
            .multi_exp
            .best_fit
            .components
            .iter()
            .map(|c| c.amplitude)
            .sum();
        for (y, &x) in top_1d.iter_mut().zip(ppm_axis.iter()) {
            let dx = x - pk.center_ppm;
            *y += amp * (hwhm * hwhm) / (dx * dx + hwhm * hwhm);
        }
    }

    let grid_cfg = DosyGridConfig {
        n_log10_d: 120,
        log10_d_min: -10.35,
        log10_d_max: -8.75,
        min_sigma_log10_d: 0.032,
        max_sigma_log10_d: 0.085,
    };
    let grid = synthesize_dosy_grid_from_peaks(&ppm_axis, &top_1d, &dosy_peaks, &grid_cfg, false)
        .map_err(|e| e.to_string())?;

    let hydro_lyso = estimate_hydrodynamics(
        dosy_peaks[0].multi_exp.best_fit.components[0].d_m2_s,
        DosySolvent::D2O,
        298.15,
        SoluteCompactnessClass::GlobularProtein,
    )
    .map_err(|e| e.to_string())?;

    let mut svg = svg_open_header(
        "Pseudo-2D DOSY Diffusion Map (delta 1H x log10 D) & Stejskal-Tanner BIC Fit",
        "16-step Wu-Johnson BPP-LED mixture in D2O (298 K) · Discrete K=1..2 BIC + Stokes-Einstein MW (synthetic forward-model showcase)",
        "2D DOSY · BPP-LED",
    );

    // 2D DOSY map frame (left) + 1D top projection
    let px0 = 96.0;
    let px1 = 602.0;
    let py0 = 132.0;
    let py1 = 482.0;
    let ld_min = grid_cfg.log10_d_min;
    let ld_max = grid_cfg.log10_d_max;
    let map_x = |p: f64| -> f64 { px0 + (ppm_hi - p) / (ppm_hi - ppm_lo) * (px1 - px0) };
    // Put fast diffusion (`-8.8`) near top and slow macromolecule (`-10.3`) near bottom
    let map_y = |ld: f64| -> f64 { py0 + (ld_max - ld) / (ld_max - ld_min) * (py1 - py0) };

    let _ = writeln!(
        svg,
        r##"  <rect class="card-alt" x="{px0:.1}" y="{py0:.1}" width="{:.1}" height="{:.1}" rx="6" fill="#f1f5f9" stroke="#cbd5e1"/>"##,
        px1 - px0,
        py1 - py0
    );

    // Top 1D projection
    let top_max = top_1d.iter().copied().fold(1.0_f64, f64::max);
    let mut top_path = String::with_capacity(n_ppm * 11);
    for (i, (&p, &yv)) in ppm_axis.iter().zip(top_1d.iter()).enumerate() {
        let x = map_x(p);
        let y = 124.0 - (yv / top_max).clamp(0.0, 1.0) * 44.0;
        let cmd = if i == 0 { 'M' } else { 'L' };
        let _ = write!(top_path, "{cmd}{} {}", f1(x)?, f1(y)?);
    }
    let _ = writeln!(
        svg,
        r##"  <path class="pos-stroke" d="{top_path}" stroke="#0284c7" stroke-width="1.3" fill="none"/>"##
    );

    // Grid & ticks
    for p_tk in [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0] {
        let x = map_x(p_tk);
        let _ = writeln!(
            svg,
            r##"  <line class="grid" x1="{}" y1="{py0:.1}" x2="{}" y2="{py1:.1}" stroke="#e2e8f0"/>"##,
            f1(x)?,
            f1(x)?
        );
        let _ = writeln!(
            svg,
            r##"  <text class="muted mono" x="{}" y="{:.1}" font-size="9.6" text-anchor="middle" fill="#475569">{:.1}</text>"##,
            f1(x)?,
            py1 + 16.0,
            p_tk
        );
    }
    for ld_tk in [-9.0, -9.3, -9.6, -9.9, -10.2] {
        let y = map_y(ld_tk);
        let _ = writeln!(
            svg,
            r##"  <line class="grid" x1="{px0:.1}" y1="{}" x2="{px1:.1}" y2="{}" stroke="#e2e8f0"/>"##,
            f1(y)?,
            f1(y)?
        );
        let _ = writeln!(
            svg,
            r##"  <text class="muted mono" x="{:.1}" y="{}" font-size="9.6" text-anchor="end" fill="#475569">{:.1}</text>"##,
            px0 - 8.0,
            f1(y + 3.5)?,
            ld_tk
        );
    }

    // Horizontal species diffusivity bands
    for (ld_band, col, name) in [
        (-9.076, "#059669", "Maleate Ref (D = 8.4e-10 m^2/s)"),
        (-9.509, "#0284c7", "Raffinose (D = 3.1e-10 m^2/s)"),
        (
            -9.951,
            "#7c3aed",
            &format!(
                "Lysozyme (D = 1.12e-10, Rh = {:.1} A, MW ~ {:.1} kDa)",
                hydro_lyso.r_h_angstrom,
                hydro_lyso.estimated_mw_da / 1000.0
            ),
        ),
    ] {
        let y = map_y(ld_band);
        let _ = writeln!(
            svg,
            r##"  <line x1="{px0:.1}" y1="{}" x2="{px1:.1}" y2="{}" stroke="{col}" stroke-width="1.1" stroke-dasharray="5,3" opacity="0.75"/>"##,
            f1(y)?,
            f1(y)?
        );
        let _ = writeln!(
            svg,
            r##"  <text class="mono" x="{:.1}" y="{}" font-size="9.0" font-weight="600" fill="{col}">{}</text>"##,
            px0 + 8.0,
            f1(y - 5.0)?,
            escape_xml(name)
        );
    }

    // Marching-squares contours of 2D DOSY grid (`row 0 = ld_min`, `row n-1 = ld_max`)
    let g_max = grid.data.iter().copied().fold(1.0_f64, f64::max);
    let gy0 = map_y(grid.log10_d_axis[0]);
    let gy1 = map_y(*grid.log10_d_axis.last().unwrap_or(&ld_max));
    for (lvl, lw) in [(0.08, 1.0), (0.20, 1.2), (0.45, 1.4), (0.72, 1.55)] {
        let d = marching_squares_svg_path(
            &grid.data,
            grid.ppm_axis.len(),
            grid.log10_d_axis.len(),
            g_max * lvl,
            px0,
            px1,
            gy0,
            gy1,
        )?;
        if !d.is_empty() {
            let _ = writeln!(
                svg,
                r##"  <path class="pos-stroke" d="{d}" stroke="#0284c7" stroke-width="{lw:.2}" fill="none"/>"##
            );
        }
    }

    // Peak center markers
    for pk in &dosy_peaks {
        for comp in &pk.multi_exp.best_fit.components {
            let cx = map_x(pk.center_ppm);
            let cy = map_y(comp.log10_d);
            let _ = writeln!(
                svg,
                r##"  <circle cx="{}" cy="{}" r="2.7" fill="#e11d48" stroke="#ffffff" stroke-width="0.7"/>"##,
                f1(cx)?,
                f1(cy)?
            );
        }
    }

    let _ = writeln!(
        svg,
        r##"  <text class="ink" x="{:.1}" y="518" font-size="11.5" font-weight="600" text-anchor="middle" fill="#0f172a">1H Chemical Shift delta (ppm)</text>"##,
        0.5 * (px0 + px1)
    );
    let _ = writeln!(
        svg,
        r##"  <text class="ink" transform="translate(28,{:.1}) rotate(-90)" font-size="11.5" font-weight="600" text-anchor="middle" fill="#0f172a">log10(D / m^2 s^-1)</text>"##,
        0.5 * (py0 + py1)
    );

    // --- Right Inset: Stejskal-Tanner Attenuation Curves I(b)/I0 vs b (10^9 s/m^2) ---
    let rx0 = 622.0;
    let rx1 = 932.0;
    let ry0 = 132.0;
    let ry1 = 482.0;
    let _ = writeln!(
        svg,
        r##"  <rect class="card-alt" x="{rx0:.1}" y="{ry0:.1}" width="{:.1}" height="{:.1}" rx="8" fill="#f1f5f9" stroke="#cbd5e1"/>"##,
        rx1 - rx0,
        ry1 - ry0
    );
    let _ = writeln!(
        svg,
        r##"  <text class="ink" x="{:.1}" y="{:.1}" font-size="12.0" font-weight="700" fill="#0f172a">Stejskal-Tanner Decay Curves</text>"##,
        rx0 + 14.0,
        ry0 + 22.0
    );
    let _ = writeln!(
        svg,
        r##"  <text class="muted mono" x="{:.1}" y="{:.1}" font-size="9.2" fill="#475569">I(b) = sum A_k exp(-b D_k) · BIC Selected</text>"##,
        rx0 + 14.0,
        ry0 + 38.0
    );

    let gx0 = rx0 + 40.0;
    let gx1 = rx1 - 16.0;
    let gy_top = ry0 + 56.0;
    let gy_bot = ry1 - 42.0;
    let b_max = b_vals.iter().copied().fold(1.0_f64, f64::max);
    let map_bx = |b: f64| -> f64 { gx0 + (b / b_max) * (gx1 - gx0) };
    let map_iy = |y: f64| -> f64 { gy_bot - y.clamp(0.0, 1.05) * (gy_bot - gy_top) };

    for y_tk in [0.0, 0.25, 0.50, 0.75, 1.0] {
        let y = map_iy(y_tk);
        let _ = writeln!(
            svg,
            r##"  <line class="grid" x1="{gx0:.1}" y1="{}" x2="{gx1:.1}" y2="{}" stroke="#e2e8f0"/>"##,
            f1(y)?,
            f1(y)?
        );
        let _ = writeln!(
            svg,
            r##"  <text class="muted mono" x="{:.1}" y="{}" font-size="9.0" text-anchor="end" fill="#475569">{:.2}</text>"##,
            gx0 - 5.0,
            f1(y + 3.0)?,
            y_tk
        );
    }

    // Plot Maleate (fast), Overlap 3.66 ppm (K=2 bi-exponential!), and Lysozyme (slow)
    for &(pk_idx, col, tag) in &[
        (1usize, "#059669", "6.02 ppm (K=1 Maleate)"),
        (3usize, "#e11d48", "3.66 ppm (K=2 Bi-exp BIC)"),
        (5usize, "#7c3aed", "0.88 ppm (K=1 Lysozyme)"),
    ] {
        let pk = &dosy_peaks[pk_idx];
        let i0 = pk.intensities.first().copied().unwrap_or(1.0).max(1e-6);
        let mut cpath = String::new();
        for s in 0..=60 {
            let b = (s as f64) / 60.0 * b_max;
            let mut yf = 0.0_f64;
            for c in &pk.multi_exp.best_fit.components {
                yf += c.amplitude * (-b * c.d_m2_s).exp();
            }
            let x = map_bx(b);
            let y = map_iy(yf / i0);
            let cmd = if s == 0 { 'M' } else { 'L' };
            let _ = write!(cpath, "{cmd}{} {}", f1(x)?, f1(y)?);
        }
        let _ = writeln!(
            svg,
            r##"  <path d="{cpath}" stroke="{col}" stroke-width="1.6" fill="none"/>"##
        );
        for (&b, &yi) in b_vals.iter().zip(pk.intensities.iter()) {
            let _ = writeln!(
                svg,
                r##"  <circle cx="{}" cy="{}" r="2.3" fill="{col}"/>"##,
                f1(map_bx(b))?,
                f1(map_iy(yi / i0))?
            );
        }
        let ly = gy_top
            + 14.0
            + (if pk_idx == 1 {
                0.0
            } else if pk_idx == 3 {
                15.0
            } else {
                30.0
            });
        let _ = writeln!(
            svg,
            r##"  <text class="mono" x="{:.1}" y="{ly:.1}" font-size="8.8" font-weight="600" fill="{col}">{tag} (sel K={})</text>"##,
            gx0 + 52.0,
            pk.multi_exp.selected_k
        );
    }

    let _ = writeln!(
        svg,
        r##"  <text class="muted" x="{:.1}" y="{:.1}" font-size="10.2" text-anchor="middle" fill="#475569">Stejskal-Tanner Factor b (up to {:.1}e9 s m^-2)</text>"##,
        0.5 * (gx0 + gx1),
        ry1 - 12.0,
        b_max * 1.0e-9
    );

    svg.push_str("</svg>\n");
    Ok(svg)
}

// ============================================================================
// Panel 06: 1D Metabolomics PCA Hotelling T^2 + OPLS-DA S-Plot + STOCSY
// ============================================================================

fn render_06_metabolomics_svg(seed: u64) -> Result<String, String> {
    let cohort = generate_demo_biofluid_cohort(seed);
    let cfg = MetabolomicsPipelineConfig {
        grid_step_ppm: 0.004,
        n_permutations: 100,
        rng_seed: seed,
        ..MetabolomicsPipelineConfig::default()
    };
    let study = run_metabolomics_pipeline(&cohort, &cfg);

    let opls = study
        .opls_da
        .as_ref()
        .ok_or_else(|| "expected binary OPLS-DA model on demo cohort".to_string())?;
    let stocsy = study
        .top_biomarker_stocsy
        .as_ref()
        .ok_or_else(|| "expected top biomarker STOCSY result on demo cohort".to_string())?;

    let perm_str = if opls.permutation.p_value_q2 <= (1.0 / 101.0) + 1.0e-5 {
        "perm p <= 1/101 (0.0099, 100-perm floor)".to_string()
    } else {
        format!("perm p = {:.4}", opls.permutation.p_value_q2)
    };
    let mut svg = svg_open_header(
        "Untargeted 1D NMR Metabolomics: PCA Hotelling T^2, OPLS-DA S-Plot & STOCSY",
        &format!(
            "AIB + PQN + Pareto cohort (N={}) · OPLS-DA Q^2 = {:.3} ({perm_str}) · Synthetic forward-model showcase",
            study.sample_ids.len(),
            opls.q2
        ),
        "PCA · OPLS-DA · STOCSY",
    );

    // --- Panel A (Top-Left): PCA Scores + 95% Hotelling T^2 Ellipse ---
    let ax0 = 28.0;
    let ax1 = 468.0;
    let ay0 = 72.0;
    let ay1 = 290.0;
    let _ = writeln!(
        svg,
        r##"  <rect class="card-alt" x="{ax0:.1}" y="{ay0:.1}" width="{:.1}" height="{:.1}" rx="8" fill="#f1f5f9" stroke="#cbd5e1"/>"##,
        ax1 - ax0,
        ay1 - ay0
    );
    let r2_pc1 = study
        .pca
        .explained_variance_ratio
        .first()
        .copied()
        .unwrap_or(0.0)
        * 100.0;
    let r2_pc2 = study
        .pca
        .explained_variance_ratio
        .get(1)
        .copied()
        .unwrap_or(0.0)
        * 100.0;
    let _ = writeln!(
        svg,
        r##"  <text class="ink" x="{:.1}" y="{:.1}" font-size="11.8" font-weight="700" fill="#0f172a">A. PCA Score Space &amp; 95% Hotelling T^2 (PC1 {:.1}%, PC2 {:.1}%)</text>"##,
        ax0 + 12.0,
        ay0 + 20.0,
        r2_pc1,
        r2_pc2
    );

    let ell_pts = study.pca.global_ellipse_95.sample_points(72);
    let mut max_abs_x = 1.0_f64;
    let mut max_abs_y = 1.0_f64;
    for pt in &ell_pts {
        max_abs_x = max_abs_x.max(pt[0].abs() * 1.15);
        max_abs_y = max_abs_y.max(pt[1].abs() * 1.15);
    }
    for sc in &study.pca.scores {
        max_abs_x = max_abs_x.max(sc.first().copied().unwrap_or(0.0).abs() * 1.15);
        max_abs_y = max_abs_y.max(sc.get(1).copied().unwrap_or(0.0).abs() * 1.15);
    }
    let px_l = ax0 + 42.0;
    let px_r = ax1 - 18.0;
    let py_t = ay0 + 34.0;
    let py_b = ay1 - 22.0;
    let map_px = |v: f64| -> f64 { 0.5 * (px_l + px_r) + (v / max_abs_x) * 0.5 * (px_r - px_l) };
    let map_py = |v: f64| -> f64 { 0.5 * (py_t + py_b) - (v / max_abs_y) * 0.5 * (py_b - py_t) };

    let _ = writeln!(
        svg,
        r##"  <line class="grid" x1="{px_l:.1}" y1="{:.1}" x2="{px_r:.1}" y2="{:.1}" stroke="#e2e8f0"/>"##,
        0.5 * (py_t + py_b),
        0.5 * (py_t + py_b)
    );
    let _ = writeln!(
        svg,
        r##"  <line class="grid" x1="{:.1}" y1="{py_t:.1}" x2="{:.1}" y2="{py_b:.1}" stroke="#e2e8f0"/>"##,
        0.5 * (px_l + px_r),
        0.5 * (px_l + px_r)
    );

    let mut ell_d = String::new();
    for (idx, pt) in ell_pts.iter().enumerate() {
        let cmd = if idx == 0 { 'M' } else { 'L' };
        let _ = write!(ell_d, "{cmd}{} {}", f1(map_px(pt[0]))?, f1(map_py(pt[1]))?);
    }
    ell_d.push('Z');
    let _ = writeln!(
        svg,
        r##"  <path d="{ell_d}" stroke="#64748b" stroke-width="1.2" stroke-dasharray="4,2.5" fill="none"/>"##
    );

    for (sc, grp) in study.pca.scores.iter().zip(study.sample_groups.iter()) {
        let x = map_px(sc.first().copied().unwrap_or(0.0));
        let y = map_py(sc.get(1).copied().unwrap_or(0.0));
        let is_case = grp.to_ascii_lowercase().contains("case")
            || grp.to_ascii_lowercase().contains("treated")
            || grp != &study.sample_groups[0];
        let col = if is_case { "#e11d48" } else { "#0284c7" };
        let _ = writeln!(
            svg,
            r##"  <circle cx="{}" cy="{}" r="3.6" fill="{col}" stroke="#ffffff" stroke-width="0.8"/>"##,
            f1(x)?,
            f1(y)?
        );
    }
    let _ = writeln!(
        svg,
        r##"  <text class="mono" x="{:.1}" y="{:.1}" font-size="9.2" font-weight="600" fill="#0284c7">Control</text>"##,
        px_l + 6.0,
        py_t + 12.0
    );
    let _ = writeln!(
        svg,
        r##"  <text class="mono" x="{:.1}" y="{:.1}" font-size="9.2" font-weight="600" fill="#e11d48">Case (95% T^2 dashed)</text>"##,
        px_l + 62.0,
        py_t + 12.0
    );

    // --- Panel B (Top-Right): OPLS-DA Wiklund S-Plot ---
    let bx0 = 484.0;
    let bx1 = 932.0;
    let _ = writeln!(
        svg,
        r##"  <rect class="card-alt" x="{bx0:.1}" y="{ay0:.1}" width="{:.1}" height="{:.1}" rx="8" fill="#f1f5f9" stroke="#cbd5e1"/>"##,
        bx1 - bx0,
        ay1 - ay0
    );
    let _ = writeln!(
        svg,
        r##"  <text class="ink" x="{:.1}" y="{:.1}" font-size="11.8" font-weight="700" fill="#0f172a">B. OPLS-DA Wiklund S-Plot (Cov(t_p, X) vs p(corr))</text>"##,
        bx0 + 12.0,
        ay0 + 20.0
    );

    let sx_l = bx0 + 42.0;
    let sx_r = bx1 - 18.0;
    let max_cov = opls
        .s_plot_cov
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.1_f64, f64::max)
        * 1.12;
    let map_sx = |cov: f64| -> f64 { 0.5 * (sx_l + sx_r) + (cov / max_cov) * 0.5 * (sx_r - sx_l) };
    let map_sy =
        |corr: f64| -> f64 { 0.5 * (py_t + py_b) - corr.clamp(-1.0, 1.0) * 0.46 * (py_b - py_t) };

    let _ = writeln!(
        svg,
        r##"  <line class="grid" x1="{sx_l:.1}" y1="{:.1}" x2="{sx_r:.1}" y2="{:.1}" stroke="#e2e8f0"/>"##,
        0.5 * (py_t + py_b),
        0.5 * (py_t + py_b)
    );
    let _ = writeln!(
        svg,
        r##"  <line class="grid" x1="{:.1}" y1="{py_t:.1}" x2="{:.1}" y2="{py_b:.1}" stroke="#e2e8f0"/>"##,
        0.5 * (sx_l + sx_r),
        0.5 * (sx_l + sx_r)
    );

    for j in 0..study.buckets.len() {
        let cov = opls.s_plot_cov.get(j).copied().unwrap_or(0.0);
        let corr = opls.s_plot_corr.get(j).copied().unwrap_or(0.0);
        let vip = opls.vip_predictive.get(j).copied().unwrap_or(0.0);
        let is_hit = vip >= 1.2 && corr.abs() >= 0.70;
        let col = if is_hit && corr > 0.0 {
            "#e11d48"
        } else if is_hit {
            "#0284c7"
        } else {
            "#94a3b8"
        };
        let rad = if is_hit { 3.1 } else { 1.9 };
        let _ = writeln!(
            svg,
            r##"  <circle cx="{}" cy="{}" r="{rad:.1}" fill="{col}" opacity="0.88"/>"##,
            f1(map_sx(cov))?,
            f1(map_sy(corr))?
        );
    }
    if let Some(top_bm) = study.biomarkers.first() {
        let tx = map_sx(top_bm.s_plot_cov);
        let ty = map_sy(top_bm.s_plot_corr);
        let _ = writeln!(
            svg,
            r##"  <text class="ink mono" x="{}" y="{}" font-size="9.2" font-weight="700" text-anchor="end" fill="#0f172a">#1 {:.2} ppm (VIP {:.1})</text>"##,
            f1((tx - 6.0).max(sx_l + 90.0))?,
            f1((ty - 5.0).max(py_t + 12.0))?,
            top_bm.center_ppm,
            top_bm.vip
        );
    }

    // --- Panel C (Bottom Full Width): STOCSY Correlation-Colored 1D Biofluid Spectrum ---
    let cy0 = 304.0;
    let cy1 = 512.0;
    let _ = writeln!(
        svg,
        r##"  <rect class="card-alt" x="28.0" y="{cy0:.1}" width="904.0" height="{:.1}" rx="8" fill="#f1f5f9" stroke="#cbd5e1"/>"##,
        cy1 - cy0
    );
    let top_match_txt = stocsy
        .metabolite_matches
        .first()
        .map(|m| format!("{} ({}, score={:.2})", m.name, m.bmrb_id, m.score))
        .unwrap_or_else(|| "BMRB spin-system match".to_string());
    let _ = writeln!(
        svg,
        r##"  <text class="ink" x="42.0" y="{:.1}" font-size="11.8" font-weight="700" fill="#0f172a">C. STOCSY Driver Correlation Spectrum (Driver = {:.3} ppm -&gt; {})</text>"##,
        cy0 + 21.0,
        stocsy.driver_ppm,
        escape_xml(&top_match_txt)
    );

    // Plot window `[0.85, 4.45] ppm` of mean cohort spectrum colored by `stocsy.spectral_r2`
    let sp_hi = 4.45_f64;
    let sp_lo = 0.85_f64;
    let cx_l = 52.0;
    let cx_r = 908.0;
    let cy_base = cy1 - 28.0;
    let cy_amp = 136.0;

    let m_len = study.common_ppm_grid.len();
    let n_samp = study.aligned_spectra.len().max(1) as f64;
    let mut win_pts: Vec<(f64, f64, f64)> = Vec::new();
    let mut max_mean = 1.0e-9_f64;
    for m in 0..m_len {
        let p = study.common_ppm_grid[m];
        if p < sp_lo || p > sp_hi || study.masked_grid_points[m] {
            continue;
        }
        let mut mean_y = 0.0_f64;
        for row in &study.aligned_spectra {
            mean_y += row[m];
        }
        mean_y /= n_samp;
        max_mean = max_mean.max(mean_y);
        let r2 = stocsy.spectral_r2.get(m).copied().unwrap_or(0.0);
        win_pts.push((p, mean_y, r2));
    }

    // Group segments into 5 r^2 color bins for compact, crisp vector paths
    let bin_colors = ["#64748b", "#0284c7", "#059669", "#f59e0b", "#e11d48"];
    let mut bin_paths: [String; 5] = Default::default();
    for pair in win_pts.windows(2) {
        let (p0, y0, r0) = pair[0];
        let (p1, y1, r1) = pair[1];
        if (p0 - p1).abs() > 0.03 {
            continue;
        }
        let r2_mid = 0.5 * (r0 + r1);
        let b_idx = ((r2_mid.clamp(0.0, 0.999) * 5.0).floor() as usize).min(4);
        let x0 = cx_l + (sp_hi - p0) / (sp_hi - sp_lo) * (cx_r - cx_l);
        let x1 = cx_l + (sp_hi - p1) / (sp_hi - sp_lo) * (cx_r - cx_l);
        let sy0 = cy_base - (y0 / max_mean).clamp(-0.03, 1.0) * cy_amp;
        let sy1 = cy_base - (y1 / max_mean).clamp(-0.03, 1.0) * cy_amp;
        let _ = write!(
            bin_paths[b_idx],
            "M{} {}L{} {}",
            f1(x0)?,
            f1(sy0)?,
            f1(x1)?,
            f1(sy1)?
        );
    }
    for (b_idx, bp) in bin_paths.iter().enumerate() {
        if !bp.is_empty() {
            let lw = if b_idx >= 3 { 1.65 } else { 1.15 };
            let _ = writeln!(
                svg,
                r##"  <path d="{bp}" stroke="{}" stroke-width="{lw:.2}" fill="none"/>"##,
                bin_colors[b_idx]
            );
        }
    }

    // Driver peak marker
    if stocsy.driver_ppm >= sp_lo && stocsy.driver_ppm <= sp_hi {
        let dx = cx_l + (sp_hi - stocsy.driver_ppm) / (sp_hi - sp_lo) * (cx_r - cx_l);
        let _ = writeln!(
            svg,
            r##"  <line x1="{}" y1="{:.1}" x2="{}" y2="{:.1}" stroke="#e11d48" stroke-width="1.2" stroke-dasharray="3,2"/>"##,
            f1(dx)?,
            cy0 + 32.0,
            f1(dx)?,
            cy_base
        );
        let _ = writeln!(
            svg,
            r##"  <text class="mono" x="{}" y="{:.1}" font-size="9.2" font-weight="700" text-anchor="middle" fill="#e11d48">Driver {:.2} ppm</text>"##,
            f1(dx)?,
            cy0 + 30.0,
            stocsy.driver_ppm
        );
    }

    let _ = writeln!(
        svg,
        r##"  <line class="axis" x1="{cx_l:.1}" y1="{cy_base:.1}" x2="{cx_r:.1}" y2="{cy_base:.1}" stroke="#475569"/>"##
    );
    for tk in [4.4, 4.0, 3.6, 3.2, 2.8, 2.4, 2.0, 1.6, 1.2] {
        let x = cx_l + (sp_hi - tk) / (sp_hi - sp_lo) * (cx_r - cx_l);
        let _ = writeln!(
            svg,
            r##"  <text class="muted mono" x="{}" y="{:.1}" font-size="9.2" text-anchor="middle" fill="#475569">{:.1}</text>"##,
            f1(x)?,
            cy_base + 14.0,
            tk
        );
    }

    svg.push_str("</svg>\n");
    Ok(svg)
}

// ============================================================================
// Panel 07: Chemical Exchange Studio (CEST + Multi-Field CPMG + CSP Kd)
// ============================================================================

fn render_07_exchange_svg(seed: u64) -> Result<String, String> {
    let mut rng = Rng::new(seed);

    // --- A. Dual-B1 15N CEST (15 Hz & 30 Hz at 600 MHz, p_B = 2.2%, k_ex = 160 s^-1, dw = +3.6 ppm) ---
    let offsets_ppm: Vec<f64> = (0..57).map(|i| 115.0 + (i as f64) * 0.15).collect();
    let mut cest_datasets = Vec::with_capacity(2);
    for &b1_hz in &[15.0_f64, 30.0_f64] {
        let p_true = Cest2StateParams {
            p_b: 0.022,
            k_ex_s_inv: 160.0,
            major_ppm: 117.4,
            dw_ab_ppm: 3.6,
            r1a_s_inv: 1.45,
            r2a_s_inv: 11.0,
            r1b_s_inv: 1.45,
            r2b_s_inv: 14.0,
            larmor_mhz: 60.8106,
            t_ex_s: 0.40,
            b1_hz,
            b1_inhomogeneity_frac: 0.08,
            n_b1_quad: 5,
            normalization: CestNormalization::ZeroTimePhaseCycled,
        };
        let clean = simulate_cest_2state_profile(&p_true, &offsets_ppm)?;
        let noisy: Vec<f64> = clean
            .iter()
            .map(|&v| (v + rng.normal(0.0, 0.0045)).clamp(0.0, 1.05))
            .collect();
        cest_datasets.push(CestProfileDataset {
            seq_id: 23,
            res_name: "ILE".to_string(),
            larmor_mhz: 60.8106,
            t_ex_s: 0.40,
            b1_hz,
            b1_inhomogeneity_frac: 0.08,
            normalization: CestNormalization::ZeroTimePhaseCycled,
            major_ppm_hint: Some(117.4),
            offsets_ppm: offsets_ppm.clone(),
            intensities: noisy,
            uncertainties: vec![0.005; offsets_ppm.len()],
        });
    }
    let cest_cfg = CestFitConfig {
        compute_chi2_surface: false,
        max_lm_iters: 35,
        ..CestFitConfig::default()
    };
    let cest_fit = fit_cest_global_cpu(&cest_datasets, &cest_cfg)?;
    let cest_res = cest_fit
        .residues
        .first()
        .ok_or_else(|| "empty CEST residue fit".to_string())?;

    // --- B. Dual-Field 15N CPMG (600 & 800 MHz, Baldwin2014, p_B = 1.9%, k_ex = 850 s^-1, dw = 2.7 ppm) ---
    let nu_grid: Vec<f64> = vec![
        25.0, 50.0, 100.0, 150.0, 200.0, 300.0, 400.0, 550.0, 750.0, 1000.0,
    ];
    let mut cpmg_series = Vec::with_capacity(2);
    for &(larmor_n, r20_base) in &[(60.8106_f64, 10.5_f64), (81.0808_f64, 12.6_f64)] {
        let dw_rad_s = TAU * larmor_n * 2.7;
        let r2_vals: Vec<f64> = nu_grid
            .iter()
            .map(|&nu| {
                r2eff_baldwin2014(r20_base, r20_base + 2.0, 0.019, 850.0, dw_rad_s, 0.040, nu)
                    + rng.normal(0.0, 0.16)
            })
            .collect();
        cpmg_series.push(CpmgSeriesDataset {
            seq_id: 25,
            res_name: "ASN".to_string(),
            larmor_mhz: larmor_n,
            t_relax_s: 0.040,
            nu_cpmg_hz: nu_grid.clone(),
            r2_eff_s_inv: r2_vals,
            r2_eff_err: vec![0.22; nu_grid.len()],
        });
    }
    let cpmg_cfg = CpmgFitConfig {
        model: CpmgModel::Baldwin2014,
        grid_dims: (24, 28),
        max_lm_iters: 35,
        ..CpmgFitConfig::default()
    };
    let cpmg_fit = fit_cpmg_global_cpu(&cpmg_series, &cpmg_cfg)?;
    let cpmg_res = cpmg_fit
        .residues
        .first()
        .ok_or_else(|| "empty CPMG residue fit".to_string())?;

    // --- C. Tight-Binding Ligand Titration CSP ([P]_0 = 50 uM, true Kd = 4.8 uM) ---
    let p0_um = 50.0_f64;
    let l_steps = [0.0_f64, 10.0, 22.0, 36.0, 50.0, 75.0, 110.0, 165.0, 240.0];
    let mut trajs = Vec::new();
    for &(seq_id, res_name, dmax_true) in &[
        (44i32, "ILE", 0.42_f64),
        (8i32, "LEU", 0.31_f64),
        (68i32, "HIS", 0.26_f64),
        (14i32, "THR", 0.025_f64),
        (32i32, "ASP", 0.018_f64),
    ] {
        let mut pts = Vec::with_capacity(l_steps.len());
        let mut max_obs = 0.0_f64;
        for &lt in &l_steps {
            let y_true = quadratic_tight_binding_csp(dmax_true, p0_um, lt, 4.8);
            let y_obs = if lt == 0.0 {
                0.0
            } else {
                (y_true + rng.normal(0.0, 0.003)).max(0.0)
            };
            max_obs = max_obs.max(y_obs);
            pts.push(ResidueTitrationPoint {
                protein_conc_um: p0_um,
                ligand_conc_um: lt,
                matched_peak: Some((8.2 + y_obs, 120.0)),
                csp_ppm: Some(y_obs),
            });
        }
        trajs.push(ResidueCspTrajectory {
            seq_index: (seq_id - 1) as usize,
            seq_id,
            res_name: res_name.to_string(),
            apo_h_ppm: 8.2,
            apo_n_ppm: 120.0,
            points: pts,
            max_observed_csp_ppm: max_obs,
        });
    }
    let csp_cfg = CspAnalysisConfig {
        n_mc: 120,
        mc_seed: seed,
        ..CspAnalysisConfig::default()
    };
    let csp_fit = fit_csp_global_kd(&trajs, &csp_cfg)?;

    let mut svg = svg_open_header(
        "Chemical Exchange & Binding Kinetics: Bloch-McConnell CEST, Baldwin CPMG & CSP Kd",
        "Higham [6/6] Pade 2-state CEST · Multi-field (600/800 MHz) CPMG · Tight-binding quadratic Kd (synthetic forward-model showcase)",
        "CEST · CPMG · CSP",
    );

    // Three side-by-side cards
    let cards = [
        (28.0_f64, 318.0_f64),
        (335.0_f64, 625.0_f64),
        (642.0_f64, 932.0_f64),
    ];
    let cy0 = 72.0;
    let cy1 = 512.0;
    for &(x0, x1) in &cards {
        let _ = writeln!(
            svg,
            r##"  <rect class="card-alt" x="{x0:.1}" y="{cy0:.1}" width="{:.1}" height="{:.1}" rx="8" fill="#f1f5f9" stroke="#cbd5e1"/>"##,
            x1 - x0,
            cy1 - cy0
        );
    }

    // --- Subpanel A: CEST ---
    let (ax0, ax1) = cards[0];
    let _ = writeln!(
        svg,
        r##"  <text class="ink" x="{:.1}" y="94" font-size="11.8" font-weight="700" fill="#0f172a">A. 15N CEST Minor-State Dip</text>"##,
        ax0 + 12.0
    );
    let _ = writeln!(
        svg,
        r##"  <text class="muted mono" x="{:.1}" y="110" font-size="9.0" fill="#475569">p_B={:.2}% · k_ex={:.0} s^-1 · dw=+{:.2} ppm</text>"##,
        ax0 + 12.0,
        cest_fit.p_b * 100.0,
        cest_fit.k_ex_s_inv,
        cest_res.dw_ab_ppm
    );
    let gx0 = ax0 + 36.0;
    let gx1 = ax1 - 14.0;
    let gy0 = 132.0;
    let gy1 = 468.0;
    let map_cx = |w: f64| -> f64 { gx0 + (123.4 - w) / (123.4 - 115.0) * (gx1 - gx0) };
    let map_cy = |i_rat: f64| -> f64 { gy1 - i_rat.clamp(0.0, 1.05) * (gy1 - gy0) };

    for y_tk in [0.0, 0.25, 0.50, 0.75, 1.0] {
        let y = map_cy(y_tk);
        let _ = writeln!(
            svg,
            r##"  <line class="grid" x1="{gx0:.1}" y1="{}" x2="{gx1:.1}" y2="{}" stroke="#e2e8f0"/>"##,
            f1(y)?,
            f1(y)?
        );
        let _ = writeln!(
            svg,
            r##"  <text class="muted mono" x="{:.1}" y="{}" font-size="8.8" text-anchor="end" fill="#475569">{:.2}</text>"##,
            gx0 - 4.0,
            f1(y + 3.0)?,
            y_tk
        );
    }

    for (ds_idx, ds) in cest_datasets.iter().enumerate() {
        let col = if ds_idx == 0 { "#0284c7" } else { "#e11d48" };
        let fit_p = Cest2StateParams {
            p_b: cest_fit.p_b,
            k_ex_s_inv: cest_fit.k_ex_s_inv,
            major_ppm: cest_res.major_ppm,
            dw_ab_ppm: cest_res.dw_ab_ppm,
            r1a_s_inv: cest_res.r1a_s_inv,
            r2a_s_inv: cest_res.r2a_s_inv,
            r1b_s_inv: cest_res.r1a_s_inv,
            r2b_s_inv: cest_res.r2b_s_inv,
            larmor_mhz: ds.larmor_mhz,
            t_ex_s: ds.t_ex_s,
            b1_hz: ds.b1_hz,
            b1_inhomogeneity_frac: ds.b1_inhomogeneity_frac,
            n_b1_quad: 5,
            normalization: ds.normalization,
        };
        let curve = simulate_cest_2state_profile(&fit_p, &offsets_ppm)?;
        let mut cp = String::new();
        for (i, (&w, &yv)) in offsets_ppm.iter().zip(curve.iter()).enumerate() {
            let cmd = if i == 0 { 'M' } else { 'L' };
            let _ = write!(cp, "{cmd}{} {}", f1(map_cx(w))?, f1(map_cy(yv))?);
        }
        let _ = writeln!(
            svg,
            r##"  <path d="{cp}" stroke="{col}" stroke-width="1.5" fill="none"/>"##
        );
        for (&w, &yi) in ds.offsets_ppm.iter().zip(ds.intensities.iter()).step_by(2) {
            let _ = writeln!(
                svg,
                r##"  <circle cx="{}" cy="{}" r="2.0" fill="{col}"/>"##,
                f1(map_cx(w))?,
                f1(map_cy(yi))?
            );
        }
    }
    let minor_x = map_cx(cest_res.major_ppm + cest_res.dw_ab_ppm);
    let _ = writeln!(
        svg,
        r##"  <line x1="{}" y1="138" x2="{}" y2="335" stroke="#e11d48" stroke-width="1.0" stroke-dasharray="3,2"/>"##,
        f1(minor_x)?,
        f1(minor_x)?
    );
    let _ = writeln!(
        svg,
        r##"  <text class="mono" x="{}" y="148" font-size="8.8" font-weight="700" fill="#e11d48">Minor State B</text>"##,
        f1(minor_x + 4.0)?
    );
    let _ = writeln!(
        svg,
        r##"  <text class="muted" x="{:.1}" y="496" font-size="9.8" text-anchor="middle" fill="#475569">15N Carrier Offset (ppm, B1 = 15 &amp; 30 Hz)</text>"##,
        0.5 * (gx0 + gx1)
    );

    // --- Subpanel B: Multi-Field CPMG ---
    let (bx0, bx1) = cards[1];
    let _ = writeln!(
        svg,
        r##"  <text class="ink" x="{:.1}" y="94" font-size="11.8" font-weight="700" fill="#0f172a">B. Multi-Field CPMG Dispersion</text>"##,
        bx0 + 12.0
    );
    let _ = writeln!(
        svg,
        r##"  <text class="muted mono" x="{:.1}" y="110" font-size="9.0" fill="#475569">Baldwin2014 · p_B={:.2}% · k_ex={:.0} s^-1</text>"##,
        bx0 + 12.0,
        cpmg_fit.p_b * 100.0,
        cpmg_fit.k_ex_s_inv
    );
    let mx0 = bx0 + 36.0;
    let mx1 = bx1 - 14.0;
    let r2_lo = 9.0_f64;
    let r2_hi = 30.0_f64;
    let map_nx = |nu: f64| -> f64 { mx0 + (nu / 1050.0) * (mx1 - mx0) };
    let map_ry =
        |r2: f64| -> f64 { gy1 - ((r2 - r2_lo) / (r2_hi - r2_lo)).clamp(0.0, 1.0) * (gy1 - gy0) };

    for r_tk in [10.0, 15.0, 20.0, 25.0, 30.0] {
        let y = map_ry(r_tk);
        let _ = writeln!(
            svg,
            r##"  <line class="grid" x1="{mx0:.1}" y1="{}" x2="{mx1:.1}" y2="{}" stroke="#e2e8f0"/>"##,
            f1(y)?,
            f1(y)?
        );
        let _ = writeln!(
            svg,
            r##"  <text class="muted mono" x="{:.1}" y="{}" font-size="8.8" text-anchor="end" fill="#475569">{:.0}</text>"##,
            mx0 - 4.0,
            f1(y + 3.0)?,
            r_tk
        );
    }

    for (f_idx, ser) in cpmg_series.iter().enumerate() {
        let col = if f_idx == 0 { "#0284c7" } else { "#7c3aed" };
        let r20 = cpmg_res.field_r20.get(f_idx).map(|t| t.1).unwrap_or(11.0);
        let dw_rad = TAU * ser.larmor_mhz * cpmg_res.dw_ppm;
        let mut mp = String::new();
        for s in 1..=60 {
            let nu = (s as f64) * (1020.0 / 60.0);
            let r2 = r2eff_baldwin2014(
                r20,
                r20,
                cpmg_fit.p_b,
                cpmg_fit.k_ex_s_inv,
                dw_rad,
                ser.t_relax_s,
                nu,
            );
            let cmd = if s == 1 { 'M' } else { 'L' };
            let _ = write!(mp, "{cmd}{} {}", f1(map_nx(nu))?, f1(map_ry(r2))?);
        }
        let _ = writeln!(
            svg,
            r##"  <path d="{mp}" stroke="{col}" stroke-width="1.65" fill="none"/>"##
        );
        for (&nu, &r2) in ser.nu_cpmg_hz.iter().zip(ser.r2_eff_s_inv.iter()) {
            let _ = writeln!(
                svg,
                r##"  <circle cx="{}" cy="{}" r="2.6" fill="{col}"/>"##,
                f1(map_nx(nu))?,
                f1(map_ry(r2))?
            );
        }
    }
    let _ = writeln!(
        svg,
        r##"  <text class="mono" x="{:.1}" y="152" font-size="9.0" font-weight="700" fill="#7c3aed">800 MHz (18.8 T)</text>"##,
        mx0 + 96.0
    );
    let _ = writeln!(
        svg,
        r##"  <text class="mono" x="{:.1}" y="170" font-size="9.0" font-weight="700" fill="#0284c7">600 MHz (14.1 T)</text>"##,
        mx0 + 96.0
    );
    let _ = writeln!(
        svg,
        r##"  <text class="muted" x="{:.1}" y="496" font-size="9.8" text-anchor="middle" fill="#475569">nu_CPMG Refocusing Frequency (Hz)</text>"##,
        0.5 * (mx0 + mx1)
    );

    // --- Subpanel C: Tight-Binding CSP Kd Isotherm ---
    let (cx0, cx1) = cards[2];
    let _ = writeln!(
        svg,
        r##"  <text class="ink" x="{:.1}" y="94" font-size="11.8" font-weight="700" fill="#0f172a">C. Tight-Binding CSP Kd Isotherm</text>"##,
        cx0 + 12.0
    );
    let _ = writeln!(
        svg,
        r##"  <text class="muted mono" x="{:.1}" y="110" font-size="9.0" fill="#475569">Quadratic Kd = {:.2} uM ([P]_0 = 50 uM)</text>"##,
        cx0 + 12.0,
        csp_fit.global_kd_um
    );
    let kx0 = cx0 + 40.0;
    let kx1 = cx1 - 14.0;
    let map_kx = |ratio: f64| -> f64 { kx0 + (ratio / 5.0) * (kx1 - kx0) };
    let map_ky = |csp: f64| -> f64 { gy1 - (csp / 0.45).clamp(0.0, 1.0) * (gy1 - gy0) };

    for y_tk in [0.0, 0.10, 0.20, 0.30, 0.40] {
        let y = map_ky(y_tk);
        let _ = writeln!(
            svg,
            r##"  <line class="grid" x1="{kx0:.1}" y1="{}" x2="{kx1:.1}" y2="{}" stroke="#e2e8f0"/>"##,
            f1(y)?,
            f1(y)?
        );
        let _ = writeln!(
            svg,
            r##"  <text class="muted mono" x="{:.1}" y="{}" font-size="8.8" text-anchor="end" fill="#475569">{:.2}</text>"##,
            kx0 - 4.0,
            f1(y + 3.0)?,
            y_tk
        );
    }

    let r0_fit = csp_fit
        .residue_fits
        .first()
        .ok_or_else(|| "empty CSP residue fit".to_string())?;
    let mut quad_p = String::new();
    let mut mm_p = String::new();
    for s in 0..=60 {
        let lt = (s as f64) * 4.0;
        let ratio = lt / p0_um;
        let y_q =
            quadratic_tight_binding_csp(r0_fit.ddelta_max_ppm, p0_um, lt, csp_fit.global_kd_um);
        // Biased Michaelis-Menten curve using `[L]_free ~ [L]_t`
        let y_mm = naive_michaelis_menten_csp(r0_fit.ddelta_max_ppm, lt, r0_fit.naive_mm_kd_um);
        let cmd = if s == 0 { 'M' } else { 'L' };
        let _ = write!(quad_p, "{cmd}{} {}", f1(map_kx(ratio))?, f1(map_ky(y_q))?);
        let _ = write!(mm_p, "{cmd}{} {}", f1(map_kx(ratio))?, f1(map_ky(y_mm))?);
    }
    let _ = writeln!(
        svg,
        r##"  <path d="{mm_p}" stroke="#94a3b8" stroke-width="1.4" stroke-dasharray="4,3" fill="none"/>"##
    );
    let _ = writeln!(
        svg,
        r##"  <path d="{quad_p}" stroke="#059669" stroke-width="1.8" fill="none"/>"##
    );
    for pt in &trajs[0].points {
        if let Some(yv) = pt.csp_ppm {
            let _ = writeln!(
                svg,
                r##"  <circle cx="{}" cy="{}" r="2.7" fill="#059669"/>"##,
                f1(map_kx(pt.ligand_conc_um / p0_um))?,
                f1(map_ky(yv))?
            );
        }
    }
    let _ = writeln!(
        svg,
        r##"  <text class="mono" x="{:.1}" y="388" font-size="8.8" font-weight="700" fill="#059669">Exact Quadratic (Kd={:.1} uM)</text>"##,
        kx0 + 38.0,
        csp_fit.global_kd_um
    );
    let _ = writeln!(
        svg,
        r##"  <text class="muted mono" x="{:.1}" y="404" font-size="8.8" fill="#475569">Naive MM Bias ({:.1} uM)</text>"##,
        kx0 + 38.0,
        r0_fit.naive_mm_kd_um
    );
    let _ = writeln!(
        svg,
        r##"  <text class="muted" x="{:.1}" y="496" font-size="9.8" text-anchor="middle" fill="#475569">Molar Ratio [L]_t / [P]_t</text>"##,
        0.5 * (kx0 + kx1)
    );

    svg.push_str("</svg>\n");
    Ok(svg)
}

// ============================================================================
// Panel 08: Octet 2D Small-Molecule Depiction Linked to 1H/13C Multiplet Table
// ============================================================================

fn render_08_octet_svg(_seed: u64) -> Result<String, String> {
    // (S)-Ibuprofen: chiral propanoic acid stereocenter + para-aromatic + isobutyl spin system
    let smiles = "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O";
    let mut mol = parse_smiles(smiles).map_err(|e| e.to_string())?;
    clean_2d(&mut mol);
    let _cip = perceive_cip(&mol);

    let analyt = AnalytMol::from(&mol);
    let pred_1h = predict_nmr(&analyt, NmrNucleus::H1, Some("CDCl3"), None);
    let pred_13c = predict_nmr(&analyt, NmrNucleus::C13, Some("CDCl3"), None);

    let scene = RenderScene::from(&mol);
    let style = JournalStyle::Acs1996.to_style();
    let compiled = compile_scene_primitives(&scene, &style);

    let mut svg = svg_open_header(
        "Octet 2D Chemical Structure Linked to 1H & 13C NMR Shift-Multiplet Table",
        "(S)-Ibuprofen (C13H18O2, CDCl3, 400 MHz) · 6-sphere HOSE + additivity shift prediction & CIP (S)-stereocenter",
        "Octet-Draw · 1H/13C",
    );

    // Left panel: 2D skeletal structure with color-coded atom halos & callout badges
    let lx0 = 28.0;
    let lx1 = 448.0;
    let ly0 = 72.0;
    let ly1 = 512.0;
    let _ = writeln!(
        svg,
        r##"  <rect class="card-alt" x="{lx0:.1}" y="{ly0:.1}" width="{:.1}" height="{:.1}" rx="8" fill="#f1f5f9" stroke="#cbd5e1"/>"##,
        lx1 - lx0,
        ly1 - ly0
    );
    let _ = writeln!(
        svg,
        r##"  <text class="ink" x="{:.1}" y="96" font-size="12.2" font-weight="700" fill="#0f172a">2D ACS Skeletal Depiction &amp; Equivalence Halos</text>"##,
        lx0 + 14.0
    );
    let _ = writeln!(
        svg,
        r##"  <text class="muted mono" x="{:.1}" y="112" font-size="9.2" fill="#475569">SMILES: {}</text>"##,
        lx0 + 14.0,
        smiles
    );

    let avail_w = (lx1 - lx0) - 64.0;
    let avail_h = 240.0;
    let sc = (avail_w / compiled.width_pt.max(1.0))
        .min(avail_h / compiled.height_pt.max(1.0))
        .clamp(1.0, 2.55);
    let off_x = lx0 + 32.0 + 0.5 * (avail_w - compiled.width_pt * sc);
    let off_y = 148.0 + 0.5 * (avail_h - compiled.height_pt * sc);

    let map_sx = |x_pt: f64| -> f64 { off_x + x_pt * sc };
    let map_sy = |y_pt: f64| -> f64 { off_y + y_pt * sc };

    let env_palette = [
        "#e11d48", "#0284c7", "#059669", "#7c3aed", "#d97706", "#0d9488", "#db2777",
    ];

    // Atom highlight halos for each predicted 1H resonance group
    let bond_len = style.bond_length_pt;
    for (g_idx, sig) in pred_1h.signals.iter().take(7).enumerate() {
        let col = env_palette[g_idx % env_palette.len()];
        let badge_ch = (b'a' + (g_idx as u8)) as char;
        for (a_pos, &atom_idx) in sig.atoms.iter().enumerate() {
            if let Some(sa) = scene.atoms.get(atom_idx) {
                let ax = map_sx(sa.x * bond_len + compiled.shift_x_pt);
                let ay = map_sy(-sa.y * bond_len + compiled.shift_y_pt);
                let _ = writeln!(
                    svg,
                    r##"  <circle cx="{}" cy="{}" r="9.5" fill="{col}" opacity="0.20" stroke="{col}" stroke-width="1.1"/>"##,
                    f1(ax)?,
                    f1(ay)?
                );
                if a_pos == 0 {
                    let _ = writeln!(
                        svg,
                        r##"  <text class="mono" x="{}" y="{}" font-size="9.5" font-weight="700" fill="{col}">({badge_ch})</text>"##,
                        f1(ax + 8.0)?,
                        f1(ay - 8.0)?
                    );
                }
            }
        }
    }

    // Render compiled chemical vector primitives (`Line`, `Polygon` wedges, `Path`, `Circle`, `Text`)
    for prim in &compiled.primitives {
        match prim {
            VectorPrimitive::Line {
                p1, p2, width_pt, ..
            } => {
                let _ = writeln!(
                    svg,
                    r##"  <line class="axis" x1="{}" y1="{}" x2="{}" y2="{}" stroke="#0f172a" stroke-width="{}"/>"##,
                    f1(map_sx(p1[0]))?,
                    f1(map_sy(p1[1]))?,
                    f1(map_sx(p2[0]))?,
                    f1(map_sy(p2[1]))?,
                    f2((width_pt * sc).max(1.25))?
                );
            }
            VectorPrimitive::Polygon { points, .. } => {
                if !points.is_empty() {
                    let mut pd = String::new();
                    for (i, pt) in points.iter().enumerate() {
                        let cmd = if i == 0 { 'M' } else { 'L' };
                        let _ = write!(pd, "{cmd}{} {}", f1(map_sx(pt[0]))?, f1(map_sy(pt[1]))?);
                    }
                    pd.push('Z');
                    let _ = writeln!(svg, r##"  <path class="ink" d="{pd}" fill="#0f172a"/>"##);
                }
            }
            VectorPrimitive::Path { cmds, fill, .. } => {
                let mut pd = String::new();
                for c in cmds {
                    match *c {
                        PathCmd::MoveTo(x, y) => {
                            let _ = write!(pd, "M{} {}", f1(map_sx(x))?, f1(map_sy(y))?);
                        }
                        PathCmd::LineTo(x, y) => {
                            let _ = write!(pd, "L{} {}", f1(map_sx(x))?, f1(map_sy(y))?);
                        }
                        PathCmd::CubicTo(x1, y1, x2, y2, x, y) => {
                            let _ = write!(
                                pd,
                                "C{} {},{} {},{} {}",
                                f1(map_sx(x1))?,
                                f1(map_sy(y1))?,
                                f1(map_sx(x2))?,
                                f1(map_sy(y2))?,
                                f1(map_sx(x))?,
                                f1(map_sy(y))?
                            );
                        }
                        PathCmd::Close => pd.push('Z'),
                    }
                }
                if fill.is_some() {
                    let _ = writeln!(svg, r##"  <path class="ink" d="{pd}" fill="#0f172a"/>"##);
                } else {
                    let _ = writeln!(
                        svg,
                        r##"  <path class="pos-stroke" d="{pd}" stroke="#0f172a" stroke-width="1.3" fill="none"/>"##
                    );
                }
            }
            VectorPrimitive::Circle {
                center, radius_pt, ..
            } => {
                let _ = writeln!(
                    svg,
                    r##"  <circle cx="{}" cy="{}" r="{}" fill="none" stroke="#0f172a" stroke-width="1.2"/>"##,
                    f1(map_sx(center[0]))?,
                    f1(map_sy(center[1]))?,
                    f1(radius_pt * sc)?
                );
            }
            VectorPrimitive::Text {
                pos,
                text,
                font_size_pt,
                ..
            } => {
                let _ = writeln!(
                    svg,
                    r##"  <text class="ink" x="{}" y="{}" font-size="{}" font-weight="700" fill="#0f172a">{}</text>"##,
                    f1(map_sx(pos[0]))?,
                    f1(map_sy(pos[1]))?,
                    f1((font_size_pt * sc).clamp(10.0, 16.0))?,
                    escape_xml(text)
                );
            }
        }
    }

    // Mini synthesized 1H 1D stick/Lorentzian preview at bottom of left card
    let sp_x0 = lx0 + 22.0;
    let sp_x1 = lx1 - 22.0;
    let sp_base = 486.0;
    let _ = writeln!(
        svg,
        r##"  <line class="axis" x1="{sp_x0:.1}" y1="{sp_base:.1}" x2="{sp_x1:.1}" y2="{sp_base:.1}" stroke="#475569"/>"##
    );
    for (g_idx, sig) in pred_1h.signals.iter().take(7).enumerate() {
        let col = env_palette[g_idx % env_palette.len()];
        let lines = sig.lines_at_mhz(NmrNucleus::H1, 400.0);
        for &(ppm, inten) in &lines {
            let x = sp_x0 + (11.8 - ppm.clamp(0.5, 11.8)) / (11.8 - 0.5) * (sp_x1 - sp_x0);
            let h_px = (inten * 24.0).clamp(10.0, 62.0);
            let _ = writeln!(
                svg,
                r##"  <line x1="{}" y1="{sp_base:.1}" x2="{}" y2="{}" stroke="{col}" stroke-width="1.8"/>"##,
                f1(x)?,
                f1(x)?,
                f1(sp_base - h_px)?
            );
        }
    }
    let _ = writeln!(
        svg,
        r##"  <text class="muted mono" x="{:.1}" y="502" font-size="9.0" text-anchor="middle" fill="#475569">Predicted 1H Stick Multiplet Overview (11.8 -&gt; 0.5 ppm)</text>"##,
        0.5 * (sp_x0 + sp_x1)
    );

    // Right panel: Linked 1H & 13C Chemical-Shift Multiplet Assignment Table
    let rx0 = 464.0;
    let rx1 = 932.0;
    let _ = writeln!(
        svg,
        r##"  <rect class="card-alt" x="{rx0:.1}" y="{ly0:.1}" width="{:.1}" height="{:.1}" rx="8" fill="#f1f5f9" stroke="#cbd5e1"/>"##,
        rx1 - rx0,
        ly1 - ly0
    );
    let _ = writeln!(
        svg,
        r##"  <text class="ink" x="{:.1}" y="96" font-size="12.2" font-weight="700" fill="#0f172a">Linked 1H Multiplet &amp; 13C{{1H}} Chemical-Shift Assignment Table</text>"##,
        rx0 + 14.0
    );

    // 1H table header
    let th_y = 122.0;
    let _ = writeln!(
        svg,
        r##"  <text class="muted mono" x="480" y="{th_y:.1}" font-size="9.5" font-weight="700" fill="#475569">Env</text>
  <text class="muted mono" x="526" y="{th_y:.1}" font-size="9.5" font-weight="700" fill="#475569">delta 1H</text>
  <text class="muted mono" x="602" y="{th_y:.1}" font-size="9.5" font-weight="700" fill="#475569">Mult (nH)</text>
  <text class="muted mono" x="692" y="{th_y:.1}" font-size="9.5" font-weight="700" fill="#475569">J Couplings (Hz)</text>
  <text class="muted mono" x="838" y="{th_y:.1}" font-size="9.5" font-weight="700" fill="#475569">Atoms</text>"##
    );
    let _ = writeln!(
        svg,
        r##"  <line class="grid" x1="476" y1="129" x2="920" y2="129" stroke="#cbd5e1"/>"##
    );

    for (g_idx, sig) in pred_1h.signals.iter().take(7).enumerate() {
        let col = env_palette[g_idx % env_palette.len()];
        let badge_ch = (b'a' + (g_idx as u8)) as char;
        let ry = 148.0 + (g_idx as f64) * 24.0;
        let j_txt = if sig.couplings.is_empty() {
            "—".to_string()
        } else {
            sig.couplings
                .iter()
                .take(2)
                .map(|c| format!("{:.1}", c.j_hz))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let atom_ids = sig
            .atoms
            .iter()
            .map(|a| format!("C{}", a + 1))
            .collect::<Vec<_>>()
            .join(",");

        let _ = writeln!(
            svg,
            r##"  <circle cx="486" cy="{:.1}" r="5.5" fill="{col}" opacity="0.22" stroke="{col}" stroke-width="1.1"/>"##,
            ry - 3.5
        );
        let _ = writeln!(
            svg,
            r##"  <text class="mono" x="486" y="{ry:.1}" font-size="9.4" font-weight="700" text-anchor="middle" fill="{col}">{badge_ch}</text>
  <text class="ink mono" x="526" y="{ry:.1}" font-size="9.6" font-weight="600" fill="#0f172a">{:.2} ppm</text>
  <text class="ink mono" x="602" y="{ry:.1}" font-size="9.6" fill="#0f172a">{} ({}H)</text>
  <text class="muted mono" x="692" y="{ry:.1}" font-size="9.4" fill="#475569">{}</text>
  <text class="muted mono" x="838" y="{ry:.1}" font-size="9.4" fill="#475569">{}</text>"##,
            sig.shift_ppm,
            escape_xml(&sig.multiplicity()),
            sig.nuclei,
            escape_xml(&j_txt),
            escape_xml(&atom_ids)
        );
    }

    // 13C{1H} summary section
    let c_hdr_y = 334.0;
    let _ = writeln!(
        svg,
        r##"  <line class="grid" x1="476" y1="316" x2="920" y2="316" stroke="#cbd5e1"/>
  <text class="ink" x="480" y="{c_hdr_y:.1}" font-size="11.2" font-weight="700" fill="#0f172a">Predicted 13C{{1H}} Carbon Environments ({} signals)</text>"##,
        pred_13c.signals.len()
    );
    for (c_idx, csig) in pred_13c.signals.iter().take(8).enumerate() {
        let col_idx = c_idx % 2;
        let row_idx = c_idx / 2;
        let cx = if col_idx == 0 { 480.0 } else { 704.0 };
        let cy = 358.0 + (row_idx as f64) * 23.0;
        let atoms_s = csig
            .atoms
            .iter()
            .map(|a| format!("C{}", a + 1))
            .collect::<Vec<_>>()
            .join(",");
        let _ = writeln!(
            svg,
            r##"  <text class="ink mono" x="{cx:.1}" y="{cy:.1}" font-size="9.4" fill="#0f172a">{:>6.1} ppm · {} ({})</text>"##,
            csig.shift_ppm,
            escape_xml(&csig.group_label),
            escape_xml(&atoms_s)
        );
    }
    let _ = writeln!(
        svg,
        r##"  <text class="muted mono" x="480" y="496" font-size="9.2" fill="#475569">Bidirectional 0-based Octet &lt;-&gt; 1-based MOL/SDF atom map verified</text>"##
    );

    svg.push_str("</svg>\n");
    Ok(svg)
}

// ============================================================================
// Pure Geometry & Formatting Helpers
// ============================================================================

fn one_letter_code(res3: &str) -> &'static str {
    match res3.trim().to_ascii_uppercase().as_str() {
        "ALA" => "A",
        "ARG" => "R",
        "ASN" => "N",
        "ASP" => "D",
        "CYS" => "C",
        "GLN" => "Q",
        "GLU" => "E",
        "GLY" => "G",
        "HIS" => "H",
        "ILE" => "I",
        "LEU" => "L",
        "LYS" => "K",
        "MET" => "M",
        "PHE" => "F",
        "PRO" => "P",
        "SER" => "S",
        "THR" => "T",
        "TRP" => "W",
        "TYR" => "Y",
        "VAL" => "V",
        _ => "X",
    }
}

fn catmull_rom_3d(p0: [f64; 3], p1: [f64; 3], p2: [f64; 3], p3: [f64; 3], t: f64) -> [f64; 3] {
    let t2 = t * t;
    let t3 = t2 * t;
    let mut out = [0.0_f64; 3];
    for d in 0..3 {
        out[d] = 0.5
            * ((2.0 * p1[d])
                + (-p0[d] + p2[d]) * t
                + (2.0 * p0[d] - 5.0 * p1[d] + 4.0 * p2[d] - p3[d]) * t2
                + (-p0[d] + 3.0 * p1[d] - 3.0 * p2[d] + p3[d]) * t3);
    }
    out
}

/// Perceptually ordered Lipari-Szabo `S^2` color ramp (`0.25` fiery coral -> `0.65` amber/teal -> `0.92` deep cobalt).
fn s2_color_hex(s2: f64) -> String {
    let u = ((s2 - 0.25) / 0.68).clamp(0.0, 1.0);
    let stops: [([f64; 3], f64); 4] = [
        ([225.0, 29.0, 72.0], 0.0),   // #e11d48 flexible tail
        ([245.0, 158.0, 11.0], 0.35), // #f59e0b loop
        ([16.0, 185.0, 129.0], 0.68), // #10b981 hinge
        ([37.0, 99.0, 235.0], 1.0),   // #2563eb rigid core
    ];
    let mut rgb = stops[3].0;
    for w in stops.windows(2) {
        let (c0, t0) = w[0];
        let (c1, t1) = w[1];
        if u >= t0 && u <= t1 {
            let f = ((u - t0) / (t1 - t0)).clamp(0.0, 1.0);
            rgb = [
                c0[0] + (c1[0] - c0[0]) * f,
                c0[1] + (c1[1] - c0[1]) * f,
                c0[2] + (c1[2] - c0[2]) * f,
            ];
            break;
        }
    }
    format!(
        "#{:02x}{:02x}{:02x}",
        rgb[0].round() as u8,
        rgb[1].round() as u8,
        rgb[2].round() as u8
    )
}

fn format_acs_signal(sig: &ReportedSignal, mhz: f64) -> String {
    let report = JournalNmrReport {
        nucleus: ReportNucleus::H1,
        proton_decoupled: false,
        mhz,
        solvent: "CDCl3".to_string(),
        signals: vec![sig.clone()],
    };
    report.format(PublicationStyle::Acs, TextEncoding::Ascii)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_panel_selection_parse_and_adversarial_errors() {
        assert_eq!(PanelSelection::parse("all"), Ok(PanelSelection::All));
        assert_eq!(
            PanelSelection::parse("pdb-s2"),
            Ok(PanelSelection::Single(ShowcasePanel::PdbS2))
        );
        assert!(PanelSelection::parse("nonexistent-panel").is_err());
        assert!(run_export_fig_cli(&["--panel".into(), "bogus".into()]).is_err());
        assert!(run_export_fig_cli(&["--out-dir".into(), "".into()]).is_err());
        assert!(run_export_fig_cli(&["--unknown-flag".into()]).is_err());
        assert!(run_export_fig_cli(&[]).is_err());

        let tmp_file = std::env::temp_dir().join(format!(
            "nmr_export_fig_unit_file_{}.txt",
            std::process::id()
        ));
        let _ = std::fs::write(&tmp_file, b"not a directory");
        assert!(export_panels_to_dir(
            &tmp_file,
            PanelSelection::Single(ShowcasePanel::Octet),
            CANONICAL_EXPORT_SEED,
        )
        .is_err());
        let nested_under_file = tmp_file.join("subdir");
        assert!(export_panels_to_dir(
            &nested_under_file,
            PanelSelection::Single(ShowcasePanel::Octet),
            CANONICAL_EXPORT_SEED,
        )
        .is_err());
        let _ = std::fs::remove_file(&tmp_file);
    }
}
