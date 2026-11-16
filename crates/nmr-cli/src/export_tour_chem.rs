//! Chemical & Analytical NMR Interactive Tour Engine (`T18.1b`, Modules `05`–`08`).
//!
//! Provenance: Original clean-room physics data builders and deterministic `RGB24`
//! `1280x720` frame rasterizer driving audited `nmr-core` simulation, optimization,
//! cheminformatics, and sandboxed protocol modules with canonical seed `20261801`:
//! - Module `05` (Exchange `CEST`, `CPMG`, `CSP`): Vallurupalli, Bouvignies & Kay,
//!   *J. Am. Chem. Soc.* **2012**, 134, 8148–8161 (Higham `[6/6]` Padé Bloch–McConnell);
//!   Baldwin, *J. Magn. Reson.* **2014**, 244, 114–124; Williamson, *Prog. Nucl. Magn.
//!   Reson. Spectrosc.* **2013**, 73, 1–16 (tight-binding quadratic `K_d`).
//! - Module `06` (1D `FID`/`HSVD`/Whittaker/`qNMR` & `Octet`): Barkhuijsen, de Beer &
//!   van Ormondt, *J. Magn. Reson.* **1987**, 73, 553–557 (`HSVD`); Eilers, *Anal. Chem.*
//!   **2003**, 75, 3631–3636 (Whittaker smoother); Hoye, Hanson & Vyvyan, *J. Org. Chem.*
//!   **1994**, 59, 4096–4103; Pauli et al., *J. Med. Chem.* **2014**, 57, 9220–9231;
//!   Bremser, *Anal. Chim. Acta* **1978**, 103, 355–365; Octet-Draw (`Apache-2.0`).
//! - Module `07` (`DOSY` Diffusion & Hydrodynamics): Stejskal & Tanner, *J. Chem. Phys.*
//!   **1965**, 42, 288–292; Wu, Chen & Johnson, *J. Magn. Reson. A* **1995**, 115, 260–264
//!   (`BPP-LED`); Provencher, *Comput. Phys. Commun.* **1982**, 27, 213–227 (Tikhonov `ILT`);
//!   Evans et al., *Angew. Chem. Int. Ed.* **2013**, 52, 3199–3202 (Stokes–Einstein `MW`).
//! - Module `08` (Metabolomics & Sandboxed `MCP`): De Meyer et al., *Anal. Chem.* **2008**,
//!   80, 3783–3790 (`AI-Binning`); Dieterle et al., *Anal. Chem.* **2006**, 78, 4281–4290
//!   (`PQN`); Trygg & Wold, *J. Chemom.* **2002**, 16, 119–128 (`OPLS-DA`); Wiklund et al.,
//!   *Anal. Chem.* **2008**, 80, 115–122 (`S-plot`); Cloarec et al., *Anal. Chem.* **2005**,
//!   77, 1282–1289 (`STOCSY`); JSON-RPC 2.0 `MCP` `ReadOnly` sandbox & `256`-step Undo bridge.

use std::f64::consts::TAU;
use std::fs;
use std::path::PathBuf;
use std::sync::OnceLock;

use serde_json::{json, Value};

use nmr_core::cest::{
    fit_cest_global_cpu, simulate_cest_2state_profile, Cest2StateParams, CestFitConfig,
    CestNormalization, CestProfileDataset,
};
use nmr_core::cpmg::{
    evaluate_exchange_gate, fit_cpmg_global_cpu, r2eff_baldwin2014, CpmgFitConfig, CpmgModel,
    CpmgSeriesDataset, ExchangeGateConfig,
};
use nmr_core::csp::{
    fit_csp_global_kd, naive_michaelis_menten_csp, quadratic_tight_binding_csp,
    track_titration_trajectories, AssignedAmidePeak, CspAnalysisConfig, ResidueCspTrajectory,
    UnassignedHsqcPeak, UnassignedTitrationStep,
};
use nmr_core::dosy::{
    compute_b_values_s_m2, estimate_hydrodynamics, fit_stejskal_tanner_multi, solve_tikhonov_ilt,
    synthesize_dosy_grid_from_peaks, BootstrapMethod, DosyFitConfig, DosyGrid2D, DosyGridConfig,
    DosyNucleus, DosyPeakEntry, DosySequenceType, DosySolvent, GradientShape, HydrodynamicEstimate,
    IltConfig, IltLambdaStrategy, IltResult, ModelSelectionCriterion as DosyCriterion,
    SoluteCompactnessClass, TikhonovRegularizer,
};
use nmr_core::mcp::{McpSecurityPolicy, McpServer};
use nmr_core::metabolomics::{
    compute_stocsy_at_ppm, generate_demo_biofluid_cohort, run_metabolomics_pipeline,
    MetabolomicsPipelineConfig, MetabolomicsStudyResult, StocsyResult,
};
use nmr_core::multiplet::{
    analyze_multiplet_region, calculate_qnmr_purity, integrate_regions,
    synthesize_multiplet_spectrum, BaselineMode, IntegralRegionSpec, JournalNmrReport,
    MultipletAnalysis, MultipletConfig, MultipletSubPeak, NormalizationMode, PublicationStyle,
    QnmrAcquisitionParams, QnmrAnalyteSignal, QnmrAnalyteSpec, QnmrPurityReport, QnmrStandardSpec,
    ReportNucleus, ReportedSignal, TextEncoding,
};
use nmr_core::nmr1d::{
    apply_apodization_1d, correct_baseline_1d, deconvolve_region_1d, fft_1d,
    subtract_solvent_hsvd_1d, synthesize_fid_1d, zero_fill_pow2_1d, Apodization1D,
    BaselineConfig1D, BaselineMethod1D, DeconvolutionClusterResult, DeconvolveConfig1D,
    Experiment1DKind, HsvdConfig, SyntheticLine1D,
};
use nmr_core::octet_analyt::nmr_predict::{predict_nmr, NmrNucleus};
use nmr_core::octet_analyt::AnalytMol;
use nmr_core::octet_mol::{clean_2d, parse_smiles, perceive_cip};
use nmr_core::octet_render::{
    compile_scene_primitives, CompiledScene, JournalStyle, PathCmd, RenderScene, VectorPrimitive,
};
use nmr_core::peaks::Peak;
use nmr_core::rng::Rng;
use nmr_core::session::{CommandHistory, WorkbenchCommand, WorkbenchSessionState};

/// Canonical deterministic seed shared across Phase 18 tours.
pub const CANONICAL_TOUR_SEED: u64 = 20261801;
/// Tour bundle JSON schema tag shared with the CLI exporter and web player.
pub const TOUR_SCHEMA_VERSION: &str = "ferric-tour-v1";
/// Fixed 4-chapter tour duration in seconds (`4 x 4.0 s`).
pub const TOUR_DURATION_SEC: f64 = 16.0;
/// Fixed video frame rate in frames per second (`480` frames total).
pub const TOUR_FPS: u32 = 30;

const UBIQUITIN_STAR_FIXTURE: &str = include_str!("../../../data/fixtures/synthetic_ubiquitin.str");

/// Return the canonical file slug for Modules `05`–`08`.
pub const fn chem_module_slug(module_index: u8) -> Option<&'static str> {
    match module_index {
        5 => Some("05_exchange_cest_cpmg_csp"),
        6 => Some("06_nmr1d_multiplet_qnmr_octet"),
        7 => Some("07_dosy_diffusion"),
        8 => Some("08_metabolomics_mcp"),
        _ => None,
    }
}

/// Build the deterministic `ferric-tour-v1` JSON bundle for Module `5..=8` with canonical seed.
pub fn build_chem_tour_bundle(module_index: u8) -> Result<Value, String> {
    match module_index {
        5 => Ok(get_mod05_data()?.bundle.clone()),
        6 => Ok(get_mod06_data()?.bundle.clone()),
        7 => Ok(get_mod07_data()?.bundle.clone()),
        8 => Ok(get_mod08_data()?.bundle.clone()),
        other => Err(format!(
            "chem tour module_index must be in 5..=8, got {other}"
        )),
    }
}

/// Hook-compatible `ChemBundleFn` signature (`fn(u8, u64) -> Result<Value, String>`).
pub fn build_chem_tour_bundle_with_seed(module_index: u8, _seed: u64) -> Result<Value, String> {
    build_chem_tour_bundle(module_index)
}

/// Build all four deterministic `ferric-tour-v1` JSON bundles (`05`–`08`).
pub fn build_all_chem_tour_bundles() -> Result<Vec<Value>, String> {
    Ok(vec![
        build_chem_tour_bundle(5)?,
        build_chem_tour_bundle(6)?,
        build_chem_tour_bundle(7)?,
        build_chem_tour_bundle(8)?,
    ])
}

/// Render a deterministic `RGB24` frame (`width * height * 3` bytes) for Modules `05`–`08`.
///
/// Any `t_sec` outside `[0.0, 16.0]` is clamped cleanly; non-chem module indices render a
/// safe diagnostic frame without panicking.
pub fn render_chem_tour_frame(
    module_index: u8,
    t_sec: f64,
    width: usize,
    height: usize,
) -> Vec<u8> {
    render_chem_tour_frame_checked(module_index, t_sec, width, height)
        .unwrap_or_else(|msg| render_fallback_frame(width, height, &msg))
}

/// Checked variant of [`render_chem_tour_frame`] returning an error for invalid dimensions
/// or module indices outside `5..=8`.
pub fn render_chem_tour_frame_checked(
    module_index: u8,
    t_sec: f64,
    width: usize,
    height: usize,
) -> Result<Vec<u8>, String> {
    if width < 64 || height < 64 {
        return Err(format!(
            "render_chem_tour_frame requires width,height >= 64, got {width}x{height}"
        ));
    }
    let t = if t_sec.is_finite() {
        t_sec.clamp(0.0, TOUR_DURATION_SEC)
    } else {
        0.0
    };
    let mut canvas = Canvas2d::new(width, height);
    canvas.clear_cinema_backdrop();

    match module_index {
        5 => {
            let d = get_mod05_data()?;
            render_mod05_scene(&mut canvas, d, t);
        }
        6 => {
            let d = get_mod06_data()?;
            render_mod06_scene(&mut canvas, d, t);
        }
        7 => {
            let d = get_mod07_data()?;
            render_mod07_scene(&mut canvas, d, t);
        }
        8 => {
            let d = get_mod08_data()?;
            render_mod08_scene(&mut canvas, d, t);
        }
        other => {
            return Err(format!(
                "chem tour module_index must be in 5..=8, got {other}"
            ));
        }
    }

    Ok(canvas.rgb)
}

fn render_fallback_frame(width: usize, height: usize, msg: &str) -> Vec<u8> {
    if width == 0 || height == 0 {
        return Vec::new();
    }
    let mut canvas = Canvas2d::new(width, height);
    canvas.clear_cinema_backdrop();
    canvas.draw_text(24.0, 24.0, 1.8, msg, [248, 113, 113], true);
    canvas.rgb
}

// ============================================================================
// Cached Deterministic Physics Datasets (Computed Once per Module)
// ============================================================================

struct Mod05Data {
    bundle: Value,
    cest_offsets: Vec<f64>,
    cest_15_obs: Vec<f64>,
    cest_15_fit: Vec<f64>,
    cest_30_obs: Vec<f64>,
    cest_30_fit: Vec<f64>,
    cest_pb: f64,
    cest_kex: f64,
    cest_major_ppm: f64,
    cest_dw_ppm: f64,
    cpmg_nu: Vec<f64>,
    cpmg_600_obs: Vec<f64>,
    cpmg_800_obs: Vec<f64>,
    cpmg_600_curve: Vec<(f64, f64)>,
    cpmg_800_curve: Vec<(f64, f64)>,
    cpmg_pb: f64,
    cpmg_kex: f64,
    cpmg_delta_aic: f64,
    cpmg_p_val: f64,
    hsqc_trajectories: Vec<ResidueCspTrajectory>,
    csp_l_max: f64,
    csp_p0: f64,
    csp_kd_quad: f64,
    csp_kd_mm: f64,
    csp_dmax: f64,
    csp_obs_pts: Vec<(f64, f64)>,
}

struct Mod06Data {
    bundle: Value,
    fid_ppm: Vec<f64>,
    spec_with_water: Vec<f64>,
    spec_hsvd_flat: Vec<f64>,
    spec_baseline_curve: Vec<f64>,
    deconv_cluster: DeconvolutionClusterResult,
    ddd_ppm: Vec<f64>,
    ddd_y: Vec<f64>,
    ddd_analysis: MultipletAnalysis,
    acs_report: String,
    qnmr_ppm: Vec<f64>,
    qnmr_y: Vec<f64>,
    qnmr_report: QnmrPurityReport,
    qnmr_analyte_int: f64,
    qnmr_std_int: f64,
    octet_scene: RenderScene,
    octet_compiled: CompiledScene,
    octet_1h_peaks: Vec<(usize, f64, f64, String)>,
    octet_13c_shifts: Vec<(usize, f64)>,
}

struct Mod07Data {
    bundle: Value,
    gradients_g_cm: Vec<f64>,
    b_vals: Vec<f64>,
    ppm_axis: Vec<f64>,
    top_1d: Vec<f64>,
    stack_slices: Vec<Vec<f64>>,
    dosy_peaks: Vec<DosyPeakEntry>,
    grid: DosyGrid2D,
    ilt_overlap: IltResult,
    hydro_maleate: HydrodynamicEstimate,
    hydro_raffinose: HydrodynamicEstimate,
    hydro_lyso: HydrodynamicEstimate,
}

struct Mod08Data {
    bundle: Value,
    study: MetabolomicsStudyResult,
    stocsy_lactate: StocsyResult,
    mean_ctrl: Vec<f64>,
    mean_case: Vec<f64>,
    mcp_init_resp: String,
    mcp_torsion_resp: String,
    mcp_denied_resp: String,
    undo_steps: Vec<(String, usize, usize, bool)>,
}

static MOD05_CACHE: OnceLock<Result<Mod05Data, String>> = OnceLock::new();
static MOD06_CACHE: OnceLock<Result<Mod06Data, String>> = OnceLock::new();
static MOD07_CACHE: OnceLock<Result<Mod07Data, String>> = OnceLock::new();
static MOD08_CACHE: OnceLock<Result<Mod08Data, String>> = OnceLock::new();

fn get_mod05_data() -> Result<&'static Mod05Data, String> {
    match MOD05_CACHE.get_or_init(compute_mod05_data) {
        Ok(d) => Ok(d),
        Err(e) => Err(e.clone()),
    }
}

fn get_mod06_data() -> Result<&'static Mod06Data, String> {
    match MOD06_CACHE.get_or_init(compute_mod06_data) {
        Ok(d) => Ok(d),
        Err(e) => Err(e.clone()),
    }
}

fn get_mod07_data() -> Result<&'static Mod07Data, String> {
    match MOD07_CACHE.get_or_init(compute_mod07_data) {
        Ok(d) => Ok(d),
        Err(e) => Err(e.clone()),
    }
}

fn get_mod08_data() -> Result<&'static Mod08Data, String> {
    match MOD08_CACHE.get_or_init(compute_mod08_data) {
        Ok(d) => Ok(d),
        Err(e) => Err(e.clone()),
    }
}

// ============================================================================
// Module 05 Builder: CEST, CPMG, HSQC Titration & Tight-Binding CSP Kd
// ============================================================================

fn compute_mod05_data() -> Result<Mod05Data, String> {
    let mut rng = Rng::new(CANONICAL_TOUR_SEED);

    // Ch 1: 2-state Higham [6/6] Pade Bloch-McConnell CEST (pB=1.5%, kex=150 s^-1, B1=15 & 30 Hz)
    let cest_offsets: Vec<f64> = (0..61).map(|i| 114.5 + (i as f64) * 0.15).collect();
    let mut cest_datasets = Vec::with_capacity(2);
    for &b1_hz in &[15.0_f64, 30.0_f64] {
        let p_true = Cest2StateParams {
            p_b: 0.015,
            k_ex_s_inv: 150.0,
            major_ppm: 117.4,
            dw_ab_ppm: 3.8,
            r1a_s_inv: 1.45,
            r2a_s_inv: 10.5,
            r1b_s_inv: 1.45,
            r2b_s_inv: 13.5,
            larmor_mhz: 60.8106,
            t_ex_s: 0.45,
            b1_hz,
            b1_inhomogeneity_frac: 0.08,
            n_b1_quad: 5,
            normalization: CestNormalization::ZeroTimePhaseCycled,
        };
        let clean = simulate_cest_2state_profile(&p_true, &cest_offsets)?;
        let noisy: Vec<f64> = clean
            .iter()
            .map(|&v| (v + rng.normal(0.0, 0.0038)).clamp(0.0, 1.05))
            .collect();
        cest_datasets.push(CestProfileDataset {
            seq_id: 23,
            res_name: "ILE".to_string(),
            larmor_mhz: 60.8106,
            t_ex_s: 0.45,
            b1_hz,
            b1_inhomogeneity_frac: 0.08,
            normalization: CestNormalization::ZeroTimePhaseCycled,
            major_ppm_hint: Some(117.4),
            offsets_ppm: cest_offsets.clone(),
            intensities: noisy,
            uncertainties: vec![0.0045; cest_offsets.len()],
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
    let mut cest_fits = Vec::with_capacity(2);
    for ds in &cest_datasets {
        let fp = Cest2StateParams {
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
        cest_fits.push(simulate_cest_2state_profile(&fp, &cest_offsets)?);
    }

    // Ch 2: Dual-field (600 & 800 MHz) Baldwin2014 CPMG dispersion + AICc exchange gate
    let cpmg_nu: Vec<f64> = vec![
        25.0, 50.0, 100.0, 150.0, 200.0, 300.0, 400.0, 550.0, 750.0, 1000.0,
    ];
    let mut cpmg_series = Vec::with_capacity(2);
    for &(larmor_n, r20_base) in &[(60.8106_f64, 10.4_f64), (81.0808_f64, 12.5_f64)] {
        let dw_rad_s = TAU * larmor_n * 2.7;
        let r2_vals: Vec<f64> = cpmg_nu
            .iter()
            .map(|&nu| {
                r2eff_baldwin2014(r20_base, r20_base + 2.0, 0.019, 850.0, dw_rad_s, 0.040, nu)
                    + rng.normal(0.0, 0.14)
            })
            .collect();
        cpmg_series.push(CpmgSeriesDataset {
            seq_id: 25,
            res_name: "ASN".to_string(),
            larmor_mhz: larmor_n,
            t_relax_s: 0.040,
            nu_cpmg_hz: cpmg_nu.clone(),
            r2_eff_s_inv: r2_vals,
            r2_eff_err: vec![0.20; cpmg_nu.len()],
        });
    }
    let gate_refs: Vec<&CpmgSeriesDataset> = cpmg_series.iter().collect();
    let gate = evaluate_exchange_gate(&gate_refs, &ExchangeGateConfig::default());
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
    let mut cpmg_curves: Vec<Vec<(f64, f64)>> = vec![Vec::new(), Vec::new()];
    for (f_idx, ser) in cpmg_series.iter().enumerate() {
        let r20 = cpmg_res.field_r20.get(f_idx).map(|t| t.1).unwrap_or(11.0);
        let dw_rad = TAU * ser.larmor_mhz * cpmg_res.dw_ppm;
        for s in 1..=50 {
            let nu = (s as f64) * 20.0;
            let r2 = r2eff_baldwin2014(
                r20,
                r20,
                cpmg_fit.p_b,
                cpmg_fit.k_ex_s_inv,
                dw_rad,
                ser.t_relax_s,
                nu,
            );
            cpmg_curves[f_idx].push((nu, r2));
        }
    }

    // Ch 3 & Ch 4: Hungarian HSQC ligand titration ([L] = 0..250 uM, [P]_0 = 100 uM, Kd = 5.0 uM)
    let p0_um = 100.0_f64;
    let true_kd_um = 5.0_f64;
    let l_steps = [0.0_f64, 20.0, 45.0, 75.0, 100.0, 130.0, 175.0, 220.0, 250.0];
    let apo_defs = [
        (44i32, "ILE", 8.42_f64, 121.2_f64, 0.34_f64, -1.65_f64),
        (8i32, "LEU", 8.18, 119.6, -0.26, 1.35),
        (68i32, "HIS", 7.92, 118.1, 0.22, 1.15),
        (14i32, "THR", 8.58, 116.8, 0.02, -0.10),
        (32i32, "ASP", 8.05, 122.7, -0.015, 0.08),
    ];
    let apo_peaks: Vec<AssignedAmidePeak> = apo_defs
        .iter()
        .map(|&(seq_id, res_name, h, n, _, _)| AssignedAmidePeak {
            seq_index: (seq_id - 1) as usize,
            seq_id,
            res_name: res_name.to_string(),
            h_ppm: h,
            n_ppm: n,
        })
        .collect();

    let mut steps = Vec::new();
    for (s_idx, &lt) in l_steps.iter().enumerate() {
        if lt <= 1e-12 {
            continue;
        }
        let mut pks = Vec::new();
        for &(_, _, h0, n0, dh_max, dn_max) in &apo_defs {
            let dmax = (dh_max * dh_max + (0.14 * dn_max).powi(2)).sqrt().max(1e-6);
            let frac = quadratic_tight_binding_csp(1.0, p0_um, lt, true_kd_um);
            let h = h0 + frac * dh_max + rng.normal(0.0, 0.002);
            let n = n0 + frac * dn_max + rng.normal(0.0, 0.012);
            let _ = dmax;
            pks.push(UnassignedHsqcPeak { h_ppm: h, n_ppm: n });
        }
        // Reverse order on alternate steps so Hungarian solver genuinely unscrambles assignment
        if s_idx % 2 == 1 {
            pks.reverse();
        }
        steps.push(UnassignedTitrationStep {
            label: format!("L={lt:.0}uM"),
            protein_conc_um: p0_um,
            ligand_conc_um: lt,
            peaks: pks,
        });
    }
    let csp_cfg = CspAnalysisConfig {
        n_mc: 100,
        mc_seed: CANONICAL_TOUR_SEED,
        ..CspAnalysisConfig::default()
    };
    let hsqc_trajectories = track_titration_trajectories(&apo_peaks, &steps, &csp_cfg)?;
    let csp_fit = fit_csp_global_kd(&hsqc_trajectories, &csp_cfg)?;
    let r0_fit = csp_fit
        .residue_fits
        .first()
        .ok_or_else(|| "empty CSP residue fit".to_string())?;
    let csp_obs_pts: Vec<(f64, f64)> = hsqc_trajectories
        .first()
        .map(|t| {
            t.points
                .iter()
                .filter_map(|pt| pt.csp_ppm.map(|v| (pt.ligand_conc_um, v)))
                .collect()
        })
        .unwrap_or_default();

    let bundle = json!({
        "schema_version": TOUR_SCHEMA_VERSION,
        "module_index": 5,
        "slug": "05_exchange_cest_cpmg_csp",
        "title": "Chemical Exchange & Tight-Binding Kinetics",
        "subtitle": "Higham [6/6] Pade CEST, Baldwin 2014 CPMG, Hungarian HSQC Tracking & Morrison Kd",
        "manual_chapter": "04-dynamics-and-exchange.html",
        "validation_badge": {
            "kind": "synthetic_showcase",
            "label": "SYNTHETIC FORWARD-MODEL SHOWCASE (F13 OPEN)",
            "detail": "SYNTHETIC forward-model showcase; Phase 17 blind real-data gate F13 remains open (0/8 Phase 17 real-data gates passed)."
        },
        "duration_sec": TOUR_DURATION_SEC,
        "fps": TOUR_FPS,
        "cli_command": "ferric exchange --cest cest.tsv --cpmg cpmg.tsv --csp titration.tsv --out exchange_fit.json",
        "chapters": [
            {
                "index": 1,
                "start_sec": 0.0,
                "end_sec": 4.0,
                "title": "Bloch-McConnell CEST Saturation Sweep",
                "caption": "Higham [6/6] Pade propagator across 15 Hz & 30 Hz B1 fields resolves minor dark state (pB = 1.5%, kex = 150 s^-1).",
                "metric_pill": format!("pB={:.2}% · kex={:.0} s^-1 · dw=+{:.2} ppm", cest_fit.p_b * 100.0, cest_fit.k_ex_s_inv, cest_res.dw_ab_ppm)
            },
            {
                "index": 2,
                "start_sec": 4.0,
                "end_sec": 8.0,
                "title": "Dual-Field Baldwin CPMG & AICc Gate",
                "caption": "600 & 800 MHz exact Baldwin (2014) R2,eff(nu_CPMG) dispersion curves pass AICc exchange gating.",
                "metric_pill": format!("pB={:.2}% · kex={:.0} s^-1 · dAICc={:.1}", cpmg_fit.p_b * 100.0, cpmg_fit.k_ex_s_inv, gate.delta_aic)
            },
            {
                "index": 3,
                "start_sec": 8.0,
                "end_sec": 12.0,
                "title": "Hungarian 2D HSQC Titration Tracking",
                "caption": "Linear extrapolation + exact Hungarian minimum-cost matching tracks fast-exchange peaks across [L] = 0..250 uM.",
                "metric_pill": format!("{} trajectories tracked · max CSP {:.3} ppm", hsqc_trajectories.len(), hsqc_trajectories.first().map(|t| t.max_observed_csp_ppm).unwrap_or(0.0))
            },
            {
                "index": 4,
                "start_sec": 12.0,
                "end_sec": 16.0,
                "title": "Morrison Quadratic vs Naive MM Isotherm",
                "caption": "Tight-binding 1:1 quadratic Kd (5.0 uM at [P]0 = 100 uM) avoids severe ligand-depletion bias in naive Michaelis-Menten.",
                "metric_pill": format!("Quad Kd={:.2} uM vs Naive MM={:.2} uM", csp_fit.global_kd_um, r0_fit.naive_mm_kd_um)
            }
        ],
        "interactive_control": {
            "id": "ligand_um",
            "label": "Ligand [L]t",
            "min": 0.0,
            "max": 300.0,
            "step": 5.0,
            "default_val": 50.0,
            "unit": "uM"
        },
        "payload": {
            "cest": {
                "offsets_ppm": cest_offsets,
                "b1_15hz_obs": cest_datasets[0].intensities,
                "b1_15hz_fit": cest_fits[0],
                "b1_30hz_obs": cest_datasets[1].intensities,
                "b1_30hz_fit": cest_fits[1],
                "p_b": cest_fit.p_b,
                "k_ex_s_inv": cest_fit.k_ex_s_inv,
                "major_ppm": cest_res.major_ppm,
                "dw_ab_ppm": cest_res.dw_ab_ppm
            },
            "cpmg": {
                "nu_hz": cpmg_nu,
                "r2_600_obs": cpmg_series[0].r2_eff_s_inv,
                "r2_800_obs": cpmg_series[1].r2_eff_s_inv,
                "p_b": cpmg_fit.p_b,
                "k_ex_s_inv": cpmg_fit.k_ex_s_inv,
                "delta_aicc": gate.delta_aic,
                "significant_exchange": gate.significant_exchange,
                "f_pvalue": gate.f_pvalue
            },
            "hsqc_trajectories": hsqc_trajectories.iter().map(|tr| json!({
                "seq_id": tr.seq_id,
                "res_name": tr.res_name,
                "max_csp": tr.max_observed_csp_ppm,
                "points": tr.points.iter().map(|pt| json!({
                    "ligand_um": pt.ligand_conc_um,
                    "h_ppm": pt.matched_peak.map(|p| p.0),
                    "n_ppm": pt.matched_peak.map(|p| p.1),
                    "csp_ppm": pt.csp_ppm
                })).collect::<Vec<_>>()
            })).collect::<Vec<_>>(),
            "csp_isotherm": {
                "protein_conc_um": p0_um,
                "true_kd_um": true_kd_um,
                "fitted_kd_um": csp_fit.global_kd_um,
                "naive_mm_kd_um": r0_fit.naive_mm_kd_um,
                "ddelta_max_ppm": r0_fit.ddelta_max_ppm,
                "observed_points": csp_obs_pts.iter().map(|&(l, y)| json!({"ligand_um": l, "csp_ppm": y})).collect::<Vec<_>>()
            }
        }
    });

    Ok(Mod05Data {
        bundle,
        cest_offsets,
        cest_15_obs: cest_datasets[0].intensities.clone(),
        cest_15_fit: cest_fits[0].clone(),
        cest_30_obs: cest_datasets[1].intensities.clone(),
        cest_30_fit: cest_fits[1].clone(),
        cest_pb: cest_fit.p_b,
        cest_kex: cest_fit.k_ex_s_inv,
        cest_major_ppm: cest_res.major_ppm,
        cest_dw_ppm: cest_res.dw_ab_ppm,
        cpmg_nu,
        cpmg_600_obs: cpmg_series[0].r2_eff_s_inv.clone(),
        cpmg_800_obs: cpmg_series[1].r2_eff_s_inv.clone(),
        cpmg_600_curve: cpmg_curves[0].clone(),
        cpmg_800_curve: cpmg_curves[1].clone(),
        cpmg_pb: cpmg_fit.p_b,
        cpmg_kex: cpmg_fit.k_ex_s_inv,
        cpmg_delta_aic: gate.delta_aic,
        cpmg_p_val: gate.f_pvalue,
        hsqc_trajectories,
        csp_l_max: 300.0,
        csp_p0: p0_um,
        csp_kd_quad: csp_fit.global_kd_um,
        csp_kd_mm: r0_fit.naive_mm_kd_um,
        csp_dmax: r0_fit.ddelta_max_ppm,
        csp_obs_pts,
    })
}

// ============================================================================
// Module 06 Builder: 1D FID/HSVD/Whittaker/Voigt, Hoye ddd, GUM qNMR & Octet
// ============================================================================

fn compute_mod06_data() -> Result<Mod06Data, String> {
    let mut rng = Rng::new(CANONICAL_TOUR_SEED);
    let mhz = 500.0_f64;

    // Ch 1: Raw 1D FID -> HSVD water suppression -> Whittaker baseline -> Voigt cluster deconvolution
    let fid_lines = vec![
        SyntheticLine1D::lorentzian(4.78, 120.0, 8.0), // Intense H2O solvent peak
        SyntheticLine1D::lorentzian(3.85, 6.2, 2.2),
        SyntheticLine1D::lorentzian(3.81, 5.1, 2.4),
        SyntheticLine1D::lorentzian(3.76, 3.8, 2.1),
        SyntheticLine1D::lorentzian(2.35, 7.5, 2.0),
    ];
    let mut fid_raw = synthesize_fid_1d(
        "synthetic_aq_1h",
        Experiment1DKind::Proton1D,
        1024,
        5000.0,
        mhz,
        4.78,
        &fid_lines,
    );
    for s in &mut fid_raw.data {
        s[0] += rng.normal(0.0, 0.08);
        s[1] += rng.normal(0.0, 0.08);
    }
    let mut fid_water = fid_raw.clone();
    apply_apodization_1d(
        &mut fid_water,
        Apodization1D::Exponential { lb_hz: 0.6 },
        0.5,
    );
    zero_fill_pow2_1d(&mut fid_water, 2);
    let mut spec_water = fft_1d(&fid_water);

    let mut fid_clean = fid_raw.clone();
    let hsvd_cfg = HsvdConfig {
        subspace_rank: 16,
        max_hankel_points: 256,
        solvent_ppm_windows: vec![(4.55, 5.05)],
    };
    let _hsvd_res = subtract_solvent_hsvd_1d(&mut fid_clean, &hsvd_cfg);
    apply_apodization_1d(
        &mut fid_clean,
        Apodization1D::Exponential { lb_hz: 0.6 },
        0.5,
    );
    zero_fill_pow2_1d(&mut fid_clean, 2);
    let mut spec_hsvd = fft_1d(&fid_clean);

    // Inject smooth baseline roll before Whittaker flattening on both traces
    let mut baseline_roll = Vec::with_capacity(spec_hsvd.len());
    let roll_scale = spec_hsvd.real.iter().copied().fold(1.0_f64, f64::max) * 0.16;
    for i in 0..spec_hsvd.len() {
        let p = spec_hsvd.ppm_at(i);
        let roll = roll_scale * (0.45 + 0.55 * ((p - 3.5) * 0.65).sin());
        spec_water.real[i] += roll;
        spec_hsvd.real[i] += roll;
        baseline_roll.push(roll);
    }
    let bl_res = correct_baseline_1d(
        &mut spec_hsvd,
        &BaselineConfig1D {
            method: BaselineMethod1D::Whittaker { lambda: 1.0e5 },
            ..BaselineConfig1D::default()
        },
    );
    let deconv_cluster =
        deconvolve_region_1d(&spec_hsvd, 3.62, 3.98, None, &DeconvolveConfig1D::default());

    // Subsample 2.0..5.5 ppm window for clean JSON payload and crisp rendering
    let mut fid_ppm = Vec::new();
    let mut spec_with_water = Vec::new();
    let mut spec_hsvd_flat = Vec::new();
    let mut spec_baseline_curve = Vec::new();
    for i in (0..spec_hsvd.len()).step_by(3) {
        let p = spec_hsvd.ppm_at(i);
        if (1.8..=5.4).contains(&p) {
            fid_ppm.push(p);
            spec_with_water.push(spec_water.real[i]);
            spec_hsvd_flat.push(spec_hsvd.real[i]);
            spec_baseline_curve.push(bl_res.baseline[i]);
        }
    }

    // Ch 2: Recursive Hoye multiplet peeling & ddd splitting tree (J1=14.2, J2=7.1, J3=2.4 Hz)
    let j1 = 14.2_f64;
    let j2 = 7.1_f64;
    let j3 = 2.4_f64;
    let center_ppm = 5.82_f64;
    let mut subpeaks = Vec::with_capacity(8);
    for &s1 in &[-0.5_f64, 0.5] {
        for &s2 in &[-0.5_f64, 0.5] {
            for &s3 in &[-0.5_f64, 0.5] {
                let off_hz = s1 * j1 + s2 * j2 + s3 * j3;
                let ppm = center_ppm + off_hz / mhz;
                let roof = 1.0 - 0.03 * (off_hz / 15.0);
                subpeaks.push(MultipletSubPeak::new(ppm, 0.125 * roof).with_fwhm_hz(1.05));
            }
        }
    }
    let ddd_n = 360usize;
    let ddd_hi = 5.88_f64;
    let ddd_lo = 5.76_f64;
    let ddd_ppm: Vec<f64> = (0..ddd_n)
        .map(|i| ddd_hi - (i as f64) * (ddd_hi - ddd_lo) / ((ddd_n - 1) as f64))
        .collect();
    let mut ddd_y = synthesize_multiplet_spectrum(&ddd_ppm, &subpeaks, mhz, 1.05, 0.16);
    for v in &mut ddd_y {
        *v += rng.normal(0.0, 0.08);
    }
    let ddd_analysis = analyze_multiplet_region(
        &ddd_ppm,
        &ddd_y,
        ddd_lo,
        ddd_hi,
        mhz,
        &MultipletConfig::default(),
    )
    .map_err(|e| e.to_string())?;
    let rep_sig = ReportedSignal::from_analysis(&ddd_analysis, Some(1), None);
    let acs_report = JournalNmrReport {
        nucleus: ReportNucleus::H1,
        proton_decoupled: false,
        mhz,
        solvent: "CDCl3".to_string(),
        signals: vec![rep_sig],
    }
    .format(PublicationStyle::Acs, TextEncoding::Ascii);

    // Ch 3: Internal-standard GUM qNMR molar purity (~98.40% +- 0.14%, k=2) & 5*T1 guard
    let q_n = 420usize;
    let q_hi = 4.55_f64;
    let q_lo = 2.65_f64;
    let qnmr_ppm: Vec<f64> = (0..q_n)
        .map(|i| q_hi - (i as f64) * (q_hi - q_lo) / ((q_n - 1) as f64))
        .collect();
    let mut q_pks = vec![MultipletSubPeak::new(3.00, 6.00).with_fwhm_hz(1.3)];
    for (m_idx, &w) in [1.0_f64, 3.0, 3.0, 1.0].iter().enumerate() {
        let off_hz = ((m_idx as f64) - 1.5) * 7.1;
        q_pks
            .push(MultipletSubPeak::new(4.12 + off_hz / mhz, 1.8424 * (w / 8.0)).with_fwhm_hz(1.3));
    }
    let qnmr_y = synthesize_multiplet_spectrum(&qnmr_ppm, &q_pks, mhz, 1.3, 0.12);
    let q_specs = vec![
        IntegralRegionSpec::new(4.26, 3.98)
            .with_baseline(BaselineMode::None)
            .with_label("Analyte CH2 (2H)"),
        IntegralRegionSpec::new(3.15, 2.85)
            .with_baseline(BaselineMode::None)
            .with_label("DMS Cert. Std (6H)"),
    ];
    let q_int = integrate_regions(
        &qnmr_ppm,
        &qnmr_y,
        &q_specs,
        NormalizationMode::ReferenceRegion {
            region_index: 1,
            target_protons: 6.0,
        },
        Some(0.0012),
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
    .with_uncertainties(0.01, 0.004, q_int.regions[1].abs_sigma.max(0.0008))
    .with_t1_s(3.2);
    let ana_spec = QnmrAnalyteSpec {
        name: "Ethyl 4-hydroxybenzoate".to_string(),
        mw_g_mol: 166.17,
        mw_sigma_g_mol: 0.01,
        mass_mg: 16.08,
        mass_sigma_mg: 0.006,
        solution_volume_ml: Some(0.60),
        solution_volume_sigma_ml: None,
        signals: vec![QnmrAnalyteSignal::new(
            2,
            q_int.regions[0].abs_integral,
            q_int.regions[0].abs_sigma.max(0.0008),
        )
        .with_t1_s(2.4)],
    };
    let acq = QnmrAcquisitionParams::new(30.0, 4.0);
    let qnmr_report =
        calculate_qnmr_purity(&ana_spec, &std_spec, Some(&acq)).map_err(|e| e.to_string())?;

    // Ch 4: Octet ACS-styled 2D (S)-Ibuprofen depiction + bidirectional 1H/13C peak link
    let smiles = "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O";
    let mut mol = parse_smiles(smiles).map_err(|e| e.to_string())?;
    clean_2d(&mut mol);
    let cip = perceive_cip(&mol);
    let analyt = AnalytMol::from(&mol);
    let pred_1h = predict_nmr(&analyt, NmrNucleus::H1, Some("CDCl3"), None);
    let pred_13c = predict_nmr(&analyt, NmrNucleus::C13, Some("CDCl3"), None);
    let octet_scene = RenderScene::from(&mol);
    let style = JournalStyle::Acs1996.to_style();
    let octet_compiled = compile_scene_primitives(&octet_scene, &style);

    let mut octet_1h_peaks = Vec::new();
    for sig in &pred_1h.signals {
        for &atom_idx in &sig.atoms {
            octet_1h_peaks.push((
                atom_idx,
                sig.shift_ppm,
                sig.nuclei as f64,
                sig.multiplicity(),
            ));
        }
    }
    octet_1h_peaks.sort_by_key(|p| p.0);

    let mut octet_13c_shifts = Vec::new();
    for sig in &pred_13c.signals {
        for &atom_idx in &sig.atoms {
            octet_13c_shifts.push((atom_idx, sig.shift_ppm));
        }
    }
    octet_13c_shifts.sort_by_key(|p| p.0);

    let js = ddd_analysis.j_values_hz();
    let j_a = js.first().copied().unwrap_or(j1);
    let j_b = js.get(1).copied().unwrap_or(j2);
    let j_c = js.get(2).copied().unwrap_or(j3);

    let bundle = json!({
        "schema_version": TOUR_SCHEMA_VERSION,
        "module_index": 6,
        "slug": "06_nmr1d_multiplet_qnmr_octet",
        "title": "1D NMR Processing, Hoye Multiplets, GUM qNMR & Octet Linking",
        "subtitle": "HSVD Water Suppression, Whittaker Baseline, ddd J-Tree, GUM Purity & (S)-Ibuprofen",
        "manual_chapter": "05-nmr1d-multiplets-and-dosy.html",
        "validation_badge": {
            "kind": "synthetic_showcase",
            "label": "SYNTHETIC FORWARD-MODEL SHOWCASE (F14 OPEN)",
            "detail": "SYNTHETIC forward-model showcase; Phase 17 blind real-data gate F14 remains open (0/8 Phase 17 real-data gates passed)."
        },
        "duration_sec": TOUR_DURATION_SEC,
        "fps": TOUR_FPS,
        "cli_command": "ferric nmr1d --fid fid --hsvd --baseline whittaker --deconvolve && ferric nmr1d qnmr --spec spec.tsv",
        "chapters": [
            {
                "index": 1,
                "start_sec": 0.0,
                "end_sec": 4.0,
                "title": "HSVD Solvent Suppression & Whittaker Baseline",
                "caption": "Time-domain Hankel-Lanczos HSVD strips 4.78 ppm H2O, Whittaker pentadiagonal smoother flattens baseline, LM Voigt resolves clusters.",
                "metric_pill": format!("{} Voigt components · RMSD {:.3}", deconv_cluster.peaks.len(), deconv_cluster.residual_rmsd)
            },
            {
                "index": 2,
                "start_sec": 4.0,
                "end_sec": 8.0,
                "title": "Recursive Hoye ddd Splitting Tree & ACS Report",
                "caption": "Symmetric Hoye multiplet peeling extracts J1=14.2, J2=7.1, J3=2.4 Hz and emits publication-ready ACS string.",
                "metric_pill": format!("{} · J = {:.1}, {:.1}, {:.1} Hz", ddd_analysis.class.as_str(), j_a, j_b, j_c)
            },
            {
                "index": 3,
                "start_sec": 8.0,
                "end_sec": 12.0,
                "title": "GUM qNMR Purity & 5*T1 Relaxation Guard",
                "caption": "Internal-standard NIST DMS qNMR propagates balance, MW, and integral noise into k=2 uncertainty with 5*T1 check.",
                "metric_pill": format!("P = {:.2}% +- {:.2}% (k=2) · 5*T1 OK={}", qnmr_report.purity_percent, qnmr_report.expanded_uncertainty_k2_percent, !qnmr_report.t1_recycle_warning)
            },
            {
                "index": 4,
                "start_sec": 12.0,
                "end_sec": 16.0,
                "title": "Octet (S)-Ibuprofen <-> 1H/13C Shift Linking",
                "caption": "ACS 2D skeletal depiction with CIP (S)-perception links every heavy atom interactively to predicted 1H/13C shifts.",
                "metric_pill": format!("C13H18O2 · {} CIP annotations · {} 1H / {} 13C signals", cip.atoms.len(), pred_1h.signals.len(), pred_13c.signals.len())
            }
        ],
        "interactive_control": {
            "id": "atom_idx",
            "label": "(S)-Ibuprofen Atom",
            "min": 0.0,
            "max": 14.0,
            "step": 1.0,
            "default_val": 3.0,
            "unit": "atom"
        },
        "payload": {
            "fid_processing": {
                "ppm": fid_ppm,
                "with_water": spec_with_water,
                "hsvd_flattened": spec_hsvd_flat,
                "baseline_roll": spec_baseline_curve,
                "voigt_peaks": deconv_cluster.peaks.iter().map(|p| json!({
                    "center_ppm": p.center_ppm,
                    "fwhm_hz": p.fwhm_hz,
                    "height": p.height,
                    "lorentz_fraction": p.lorentz_fraction
                })).collect::<Vec<_>>()
            },
            "multiplet": {
                "center_ppm": ddd_analysis.center_ppm,
                "class": ddd_analysis.class.as_str(),
                "j_hz": js,
                "acs_report": acs_report,
                "ppm": ddd_ppm,
                "intensity": ddd_y
            },
            "qnmr": {
                "purity_percent": qnmr_report.purity_percent,
                "u_k2_percent": qnmr_report.expanded_uncertainty_k2_percent,
                "relaxation_guard_passed": !qnmr_report.t1_recycle_warning,
                "analyte_integral": q_int.regions[0].norm_integral,
                "std_integral": q_int.regions[1].norm_integral
            },
            "octet_ibuprofen": {
                "smiles": smiles,
                "n_atoms": octet_scene.atoms.len(),
                "atoms": octet_scene.atoms.iter().enumerate().map(|(i, a)| json!({
                    "idx": i,
                    "element": a.label,
                    "x": a.x,
                    "y": a.y
                })).collect::<Vec<_>>(),
                "bonds": octet_scene.bonds.iter().map(|b| json!({
                    "a": b.a,
                    "b": b.b,
                    "order": b.order
                })).collect::<Vec<_>>(),
                "h1_links": octet_1h_peaks.iter().map(|(idx, ppm, nh, mult)| json!({
                    "atom_idx": idx,
                    "shift_ppm": ppm,
                    "n_protons": nh,
                    "multiplicity": mult
                })).collect::<Vec<_>>(),
                "c13_links": octet_13c_shifts.iter().map(|(idx, ppm)| json!({
                    "atom_idx": idx,
                    "shift_ppm": ppm
                })).collect::<Vec<_>>()
            }
        }
    });

    Ok(Mod06Data {
        bundle,
        fid_ppm,
        spec_with_water,
        spec_hsvd_flat,
        spec_baseline_curve,
        deconv_cluster,
        ddd_ppm,
        ddd_y,
        ddd_analysis,
        acs_report,
        qnmr_ppm,
        qnmr_y,
        qnmr_report,
        qnmr_analyte_int: q_int.regions[0].norm_integral,
        qnmr_std_int: q_int.regions[1].norm_integral,
        octet_scene,
        octet_compiled,
        octet_1h_peaks,
        octet_13c_shifts,
    })
}

// ============================================================================
// Module 07 Builder: Pseudo-2D BPP-LED DOSY, BIC/ILT & Stokes-Einstein Rh/MW
// ============================================================================

fn compute_mod07_data() -> Result<Mod07Data, String> {
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
        rng_seed: CANONICAL_TOUR_SEED,
        known_noise_sigma: Some(0.004),
    };

    let mut rng = Rng::new(CANONICAL_TOUR_SEED);
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

    // Opt-in continuous Tikhonov ILT comparison on the overlapped 3.66 ppm peak
    let ilt_cfg = IltConfig {
        n_grid: 64,
        d_min_m2_s: 1.0e-11,
        d_max_m2_s: 5.0e-9,
        regularizer: TikhonovRegularizer::SecondDerivative,
        lambda_strategy: IltLambdaStrategy::Gcv,
        lambda_min: 1.0e-5,
        lambda_max: 1.0e1,
        n_lambda_sweep: 20,
    };
    let ilt_overlap = solve_tikhonov_ilt(&b_vals, &dosy_peaks[3].intensities, &ilt_cfg)
        .map_err(|e| e.to_string())?;

    // Synthesize 1D slices across gradient steps g_1..g_16
    let n_ppm = 220usize;
    let ppm_hi = 7.8_f64;
    let ppm_lo = 0.4_f64;
    let ppm_axis: Vec<f64> = (0..n_ppm)
        .map(|i| ppm_hi - (i as f64) * (ppm_hi - ppm_lo) / ((n_ppm - 1) as f64))
        .collect();
    let mut stack_slices = Vec::with_capacity(b_vals.len());
    for g_idx in 0..b_vals.len() {
        let mut slice = vec![0.0_f64; n_ppm];
        for pk in &dosy_peaks {
            let hwhm = 0.5 * pk.fwhm_ppm;
            let amp = pk.intensities.get(g_idx).copied().unwrap_or(0.0);
            for (y, &x) in slice.iter_mut().zip(ppm_axis.iter()) {
                let dx = x - pk.center_ppm;
                *y += amp * (hwhm * hwhm) / (dx * dx + hwhm * hwhm);
            }
        }
        stack_slices.push(slice);
    }
    let top_1d = stack_slices
        .first()
        .cloned()
        .unwrap_or_else(|| vec![0.0; n_ppm]);

    let grid_cfg = DosyGridConfig {
        n_log10_d: 96,
        log10_d_min: -10.35,
        log10_d_max: -8.75,
        min_sigma_log10_d: 0.032,
        max_sigma_log10_d: 0.085,
    };
    let grid = synthesize_dosy_grid_from_peaks(&ppm_axis, &top_1d, &dosy_peaks, &grid_cfg, false)
        .map_err(|e| e.to_string())?;

    let d_mal = dosy_peaks[1]
        .multi_exp
        .best_fit
        .components
        .first()
        .map(|c| c.d_m2_s)
        .unwrap_or(8.4e-10);
    let d_raf = dosy_peaks[2]
        .multi_exp
        .best_fit
        .components
        .first()
        .map(|c| c.d_m2_s)
        .unwrap_or(3.1e-10);
    let d_lys = dosy_peaks[0]
        .multi_exp
        .best_fit
        .components
        .first()
        .map(|c| c.d_m2_s)
        .unwrap_or(1.12e-10);

    let hydro_maleate = estimate_hydrodynamics(
        d_mal,
        DosySolvent::D2O,
        298.15,
        SoluteCompactnessClass::SmallMolecule,
    )
    .map_err(|e| e.to_string())?;
    let hydro_raffinose = estimate_hydrodynamics(
        d_raf,
        DosySolvent::D2O,
        298.15,
        SoluteCompactnessClass::SmallMolecule,
    )
    .map_err(|e| e.to_string())?;
    let hydro_lyso = estimate_hydrodynamics(
        d_lys,
        DosySolvent::D2O,
        298.15,
        SoluteCompactnessClass::GlobularProtein,
    )
    .map_err(|e| e.to_string())?;

    let bundle = json!({
        "schema_version": TOUR_SCHEMA_VERSION,
        "module_index": 7,
        "slug": "07_dosy_diffusion",
        "title": "Pseudo-2D DOSY Diffusion & Stokes-Einstein Sizing",
        "subtitle": "BPP-LED Gradient Stack, Stejskal-Tanner BIC/ILT & Hydrodynamic MW",
        "manual_chapter": "05-nmr1d-multiplets-and-dosy.html",
        "validation_badge": {
            "kind": "synthetic_showcase",
            "label": "SYNTHETIC FORWARD-MODEL SHOWCASE (F16 OPEN)",
            "detail": "SYNTHETIC forward-model showcase; Phase 17 blind real-data gate F16 remains open (0/8 Phase 17 real-data gates passed)."
        },
        "duration_sec": TOUR_DURATION_SEC,
        "fps": TOUR_FPS,
        "cli_command": "ferric dosy fit --peaks dosy_peaks.tsv --seq bpp-led --shape sine100 --solvent d2o --ilt",
        "chapters": [
            {
                "index": 1,
                "start_sec": 0.0,
                "end_sec": 4.0,
                "title": "BPP-LED Gradient Stack (g1 -> g16)",
                "caption": "16-step Wu-Johnson BPP-LED ramp in D2O (298 K) with SINE.100 2/pi shape factor separates fast vs slow decay.",
                "metric_pill": format!("16 gradients (2..52 G/cm) · b_max = {:.2}e9 s/m^2", b_vals.last().copied().unwrap_or(0.0) * 1.0e-9)
            },
            {
                "index": 2,
                "start_sec": 4.0,
                "end_sec": 8.0,
                "title": "Stejskal-Tanner Decay & BIC K=1 vs K=2",
                "caption": "Overlapped 3.66 ppm signal resolves Raffinose + Maleate via BIC K=2 selection over K=1 mono-exponential.",
                "metric_pill": format!("3.66 ppm BIC K={} · Maleate D={:.2e}", dosy_peaks[3].multi_exp.selected_k, d_mal)
            },
            {
                "index": 3,
                "start_sec": 8.0,
                "end_sec": 12.0,
                "title": "2D (ppm x log10 D) Map & Tikhonov ILT",
                "caption": "Pseudo-2D DOSY map resolves Maleate, Raffinose, and Lysozyme alongside GCV-selected continuous Tikhonov ILT.",
                "metric_pill": format!("ILT GCV lambda={:.2e} · {} modes", ilt_overlap.selected_lambda, ilt_overlap.modes.len())
            },
            {
                "index": 4,
                "start_sec": 12.0,
                "end_sec": 16.0,
                "title": "Evans Stokes-Einstein Rh & MW Sizing",
                "caption": "Diffusivity maps directly to hydrodynamic radius Rh and compactness-calibrated molecular weight in D2O.",
                "metric_pill": format!("Lysozyme Rh={:.1} A (~{:.1} kDa) · Raffinose Rh={:.1} A", hydro_lyso.r_h_angstrom, hydro_lyso.estimated_mw_da / 1000.0, hydro_raffinose.r_h_angstrom)
            }
        ],
        "interactive_control": {
            "id": "grad_step",
            "label": "BPP-LED Gradient Step",
            "min": 0.0,
            "max": 15.0,
            "step": 1.0,
            "default_val": 4.0,
            "unit": "step"
        },
        "payload": {
            "gradients_g_cm": gradients_g_cm,
            "b_values_s_m2": b_vals,
            "ppm_axis": ppm_axis,
            "peaks": dosy_peaks.iter().map(|pk| json!({
                "label": pk.label,
                "center_ppm": pk.center_ppm,
                "selected_k": pk.multi_exp.selected_k,
                "intensities": pk.intensities,
                "components": pk.multi_exp.best_fit.components.iter().map(|c| json!({
                    "amplitude": c.amplitude,
                    "d_m2_s": c.d_m2_s,
                    "log10_d": c.log10_d
                })).collect::<Vec<_>>()
            })).collect::<Vec<_>>(),
            "ilt_overlap": {
                "log10_d_grid": ilt_overlap.log10_d_grid,
                "spectrum": ilt_overlap.spectrum,
                "selected_lambda": ilt_overlap.selected_lambda,
                "modes": ilt_overlap.modes.iter().map(|m| json!({
                    "log10_d": m.log10_d,
                    "d_m2_s": m.d_m2_s,
                    "area_fraction": m.area_fraction
                })).collect::<Vec<_>>()
            },
            "hydrodynamics": [
                {"name": "Maleate", "r_h_angstrom": hydro_maleate.r_h_angstrom, "mw_da": hydro_maleate.estimated_mw_da},
                {"name": "Raffinose", "r_h_angstrom": hydro_raffinose.r_h_angstrom, "mw_da": hydro_raffinose.estimated_mw_da},
                {"name": "Lysozyme", "r_h_angstrom": hydro_lyso.r_h_angstrom, "mw_da": hydro_lyso.estimated_mw_da}
            ]
        }
    });

    Ok(Mod07Data {
        bundle,
        gradients_g_cm,
        b_vals,
        ppm_axis,
        top_1d,
        stack_slices,
        dosy_peaks,
        grid,
        ilt_overlap,
        hydro_maleate,
        hydro_raffinose,
        hydro_lyso,
    })
}

// ============================================================================
// Module 08 Builder: 24-Sample Metabolomics, OPLS-DA, STOCSY & MCP + Undo(256)
// ============================================================================

fn compute_mod08_data() -> Result<Mod08Data, String> {
    let cohort = generate_demo_biofluid_cohort(CANONICAL_TOUR_SEED);
    let cfg = MetabolomicsPipelineConfig {
        grid_step_ppm: 0.005,
        n_permutations: 100,
        rng_seed: CANONICAL_TOUR_SEED,
        ..MetabolomicsPipelineConfig::default()
    };
    let study = run_metabolomics_pipeline(&cohort, &cfg);
    let opls = study
        .opls_da
        .as_ref()
        .ok_or_else(|| "expected OPLS-DA model on 24-sample demo cohort".to_string())?;
    let stocsy_lactate = compute_stocsy_at_ppm(1.33, &study)
        .or_else(|| study.top_biomarker_stocsy.clone())
        .ok_or_else(|| "expected STOCSY result at 1.33 ppm".to_string())?;

    // Mean Control vs Case spectra along common_ppm_grid
    let m_len = study.common_ppm_grid.len();
    let mut mean_ctrl = vec![0.0_f64; m_len];
    let mut mean_case = vec![0.0_f64; m_len];
    let mut n_ctrl = 0usize;
    let mut n_case = 0usize;
    for (row, grp) in study.aligned_spectra.iter().zip(study.sample_groups.iter()) {
        let is_case = grp.to_ascii_lowercase().contains("case")
            || grp.to_ascii_lowercase().contains("treated")
            || grp != &study.sample_groups[0];
        if is_case {
            n_case += 1;
            for (acc, &v) in mean_case.iter_mut().zip(row.iter()) {
                *acc += v;
            }
        } else {
            n_ctrl += 1;
            for (acc, &v) in mean_ctrl.iter_mut().zip(row.iter()) {
                *acc += v;
            }
        }
    }
    if n_ctrl > 0 {
        for v in &mut mean_ctrl {
            *v /= n_ctrl as f64;
        }
    }
    if n_case > 0 {
        for v in &mut mean_case {
            *v /= n_case as f64;
        }
    }

    // Live Sandboxed MCP JSON-RPC 2.0 ReadOnly session over temp fixture dir
    let mcp_dir: PathBuf = std::env::temp_dir().join(format!(
        "ferric_tour_mcp_{}_{}",
        CANONICAL_TOUR_SEED,
        std::process::id()
    ));
    fs::create_dir_all(&mcp_dir).map_err(|e| format!("create mcp tmp dir: {e}"))?;
    let star_file = mcp_dir.join("synthetic_ubiquitin.str");
    fs::write(&star_file, UBIQUITIN_STAR_FIXTURE)
        .map_err(|e| format!("write mcp tmp star: {e}"))?;

    let policy = McpSecurityPolicy::read_only([&mcp_dir])
        .map_err(|e| format!("mcp readonly policy error: {e}"))?;
    let mut server = McpServer::new(policy);

    let req_init = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "ferric-tour", "version": nmr_core::VERSION}
        }
    })
    .to_string();
    let mcp_init_resp = server
        .handle_json_request(&req_init)
        .unwrap_or_else(|| "{}".to_string());
    let notif_init = json!({
        "jsonrpc": "2.0",
        "method": "notifications/initialized"
    })
    .to_string();
    let _ = server.handle_json_request(&notif_init);

    let req_torsions = json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "predict_homolog_free_torsions",
            "arguments": {
                "star_path": star_file.to_string_lossy(),
                "exclude_pdb_ids": ["1UBQ", "1D3Z"],
                "max_seq_identity": 0.25
            }
        }
    })
    .to_string();
    let mcp_torsion_resp = server
        .handle_json_request(&req_torsions)
        .unwrap_or_else(|| "{}".to_string());

    let req_denied = json!({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "write_curated_peak_tsv",
            "arguments": {
                "path": mcp_dir.join("blocked.tsv").to_string_lossy(),
                "peaks_tsv": "exp\th\tn\nHSQC\t8.20\t120.0\n"
            }
        }
    })
    .to_string();
    let mcp_denied_resp = server
        .handle_json_request(&req_denied)
        .unwrap_or_else(|| "{}".to_string());

    let _ = fs::remove_dir_all(&mcp_dir);

    // 256-step reversible Workbench Undo/Redo bridge verification
    let mut sess = WorkbenchSessionState::from_fasta_sequence("Ubiquitin", "MQIFVKTLTG");
    let baseline_sess = sess.clone();
    let mut history = CommandHistory::with_capacity(256);
    let mut undo_steps = Vec::new();
    let d1 = history.execute(
        &mut sess,
        WorkbenchCommand::AddPeak(Peak {
            exp: "HSQC".to_string(),
            h: 8.42,
            n: 121.3,
            c: None,
            sign: 1,
            height: Some(1.0),
            volume: None,
            h2: None,
        }),
    );
    undo_steps.push((d1, history.undo_len(), history.redo_len(), false));
    let d2 = history.execute(
        &mut sess,
        WorkbenchCommand::AssignSpinSystem {
            spin_system: 1,
            residue_seq_idx: 3,
        },
    );
    undo_steps.push((d2, history.undo_len(), history.redo_len(), false));
    let d3 = history.execute(
        &mut sess,
        WorkbenchCommand::LockResidue {
            spin_system: 1,
            residue_seq_idx: 3,
            locked: true,
        },
    );
    undo_steps.push((d3, history.undo_len(), history.redo_len(), false));
    while let Some(u_desc) = history.undo(&mut sess) {
        undo_steps.push((
            format!("Undo: {u_desc}"),
            history.undo_len(),
            history.redo_len(),
            sess.bitwise_eq(&baseline_sess),
        ));
    }

    let perm_p = opls.permutation.p_value_q2;
    let top_match = stocsy_lactate
        .metabolite_matches
        .first()
        .map(|m| format!("{} ({}, score={:.2})", m.name, m.bmrb_id, m.score))
        .unwrap_or_else(|| "L-Lactate (bmse000288)".to_string());
    let bitwise_ok = sess.bitwise_eq(&baseline_sess);

    let bundle = json!({
        "schema_version": TOUR_SCHEMA_VERSION,
        "module_index": 8,
        "slug": "08_metabolomics_mcp",
        "title": "1D NMR Metabolomics, STOCSY & Sandboxed MCP Copilot",
        "subtitle": "AI-Binning + PQN, PCA/OPLS-DA (100-Perm), 1.33 ppm STOCSY & ReadOnly MCP + Undo(256)",
        "manual_chapter": "08-metabolomics-and-chemometrics.html",
        "validation_badge": {
            "kind": "synthetic_and_protocol_showcase",
            "label": "SYNTHETIC & PROTOCOL SHOWCASE (F17/F18 OPEN)",
            "detail": "SYNTHETIC cohort & MCP protocol showcase; Phase 17 blind empirical gates F17/F18 remain open (0/8 Phase 17 real-data gates passed)."
        },
        "duration_sec": TOUR_DURATION_SEC,
        "fps": TOUR_FPS,
        "cli_command": "ferric metabolomics run --cohort biofluid_24.tsv --bucketing ai --norm pqn --oplsda --stocsy 1.33",
        "chapters": [
            {
                "index": 1,
                "start_sec": 0.0,
                "end_sec": 4.0,
                "title": "24-Sample AI-Binning & PQN Normalization",
                "caption": "De Meyer (2008) valley-seeking adaptive intelligent binning + Dieterle PQN dilution correction across 12 Control vs 12 Case spectra.",
                "metric_pill": format!("N={} samples · {} AI buckets · PQN", study.sample_ids.len(), study.buckets.len())
            },
            {
                "index": 2,
                "start_sec": 4.0,
                "end_sec": 8.0,
                "title": "PCA Hotelling T^2 & 100-Perm OPLS-DA",
                "caption": "Leak-free stratified k-fold OPLS-DA achieves Q^2 with 100-permutation null floor and Wiklund S-plot biomarker ranking.",
                "metric_pill": format!("OPLS-DA Q^2={:.3} · perm p={:.4}", opls.q2, perm_p)
            },
            {
                "index": 3,
                "start_sec": 8.0,
                "end_sec": 12.0,
                "title": "STOCSY 1.33 ppm Driver Spin-System",
                "caption": "Statistical Total Correlation Spectroscopy from 1.33 ppm doublet highlights correlated quartet at 4.11 ppm (L-Lactate 3:1).",
                "metric_pill": format!("Driver {:.2} ppm -> {}", stocsy_lactate.driver_ppm, top_match)
            },
            {
                "index": 4,
                "start_sec": 12.0,
                "end_sec": 16.0,
                "title": "Sandboxed ReadOnly MCP & 256-Step Undo",
                "caption": "MCP JSON-RPC 2.0 enforces <=25% HomologFirewall and blocks writes in ReadOnly mode; CommandHistory restores bitwise equality.",
                "metric_pill": format!("ReadOnly MCP verified · Undo(256) bitwise_eq={bitwise_ok}")
            }
        ],
        "interactive_control": {
            "id": "driver_bucket",
            "label": "STOCSY Driver Bucket",
            "min": 0.0,
            "max": 20.0,
            "step": 1.0,
            "default_val": 5.0,
            "unit": "bin"
        },
        "payload": {
            "n_samples": study.sample_ids.len(),
            "n_buckets": study.buckets.len(),
            "pca": {
                "explained_var": study.pca.explained_variance_ratio,
                "scores": study.pca.scores.iter().zip(study.sample_groups.iter()).map(|(sc, g)| json!({
                    "t1": sc.first().copied().unwrap_or(0.0),
                    "t2": sc.get(1).copied().unwrap_or(0.0),
                    "group": g
                })).collect::<Vec<_>>(),
                "ellipse_95": study.pca.global_ellipse_95.sample_points(48)
            },
            "oplsda": {
                "r2y": opls.r2_y,
                "q2": opls.q2,
                "perm_p_q2": perm_p,
                "perm_null_q2": opls.permutation.trials.iter().map(|t| t.q2).collect::<Vec<_>>(),
                "s_plot": study.buckets.iter().enumerate().take(120).map(|(j, b)| json!({
                    "ppm": b.center_ppm,
                    "cov": opls.s_plot_cov.get(j).copied().unwrap_or(0.0),
                    "corr": opls.s_plot_corr.get(j).copied().unwrap_or(0.0),
                    "vip": opls.vip_predictive.get(j).copied().unwrap_or(0.0)
                })).collect::<Vec<_>>()
            },
            "stocsy": {
                "driver_ppm": stocsy_lactate.driver_ppm,
                "top_match": top_match,
                "ppm": study.common_ppm_grid.iter().step_by(3).copied().collect::<Vec<_>>(),
                "mean_ctrl": mean_ctrl.iter().step_by(3).copied().collect::<Vec<_>>(),
                "mean_case": mean_case.iter().step_by(3).copied().collect::<Vec<_>>(),
                "r2": stocsy_lactate.spectral_r2.iter().step_by(3).copied().collect::<Vec<_>>()
            },
            "mcp": {
                "access_mode": "ReadOnly",
                "homolog_firewall_max_seq_id": 0.25,
                "undo_capacity": 256,
                "undo_bitwise_eq": bitwise_ok,
                "initialize_response": serde_json::from_str::<Value>(&mcp_init_resp).unwrap_or(Value::Null),
                "torsion_call_response": serde_json::from_str::<Value>(&mcp_torsion_resp).unwrap_or(Value::Null),
                "readonly_denied_response": serde_json::from_str::<Value>(&mcp_denied_resp).unwrap_or(Value::Null)
            }
        }
    });

    Ok(Mod08Data {
        bundle,
        study,
        stocsy_lactate,
        mean_ctrl,
        mean_case,
        mcp_init_resp,
        mcp_torsion_resp,
        mcp_denied_resp,
        undo_steps,
    })
}

// ============================================================================
// Shared Cinema Chrome & Per-Module 4-Chapter Frame Renderers (1280x720 base)
// ============================================================================

fn chapter_and_phase(t_sec: f64) -> (usize, f64) {
    let t = t_sec.clamp(0.0, TOUR_DURATION_SEC - 1e-6);
    let ch = ((t / 4.0).floor() as usize).min(3);
    let local = ((t - (ch as f64) * 4.0) / 4.0).clamp(0.0, 1.0);
    (ch, local)
}

fn ease_in_out(u: f64) -> f64 {
    let x = u.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

struct TourChromeSpec<'a> {
    title: &'a str,
    badge_label: &'a str,
    badge_color: [u8; 3],
    ch_idx: usize,
    ch_title: &'a str,
    ch_metric: &'a str,
    cli_cmd: &'a str,
    t_sec: f64,
}

fn draw_tour_chrome(c: &mut Canvas2d, spec: &TourChromeSpec<'_>) {
    let title = spec.title;
    let badge_label = spec.badge_label;
    let badge_color = spec.badge_color;
    let ch_idx = spec.ch_idx;
    let ch_title = spec.ch_title;
    let ch_metric = spec.ch_metric;
    let cli_cmd = spec.cli_cmd;
    let t_sec = spec.t_sec;
    let sx = c.width as f64 / 1280.0;
    let sy = c.height as f64 / 720.0;

    // Top Header Bar
    c.fill_rounded_rect(
        18.0 * sx,
        14.0 * sy,
        1244.0 * sx,
        54.0 * sy,
        10.0 * sx,
        [15, 23, 42, 235],
    );
    c.stroke_rounded_rect(
        [18.0 * sx, 14.0 * sy, 1244.0 * sx, 54.0 * sy],
        10.0 * sx,
        1.5,
        [51, 65, 85, 255],
    );
    // Top-left FERRIC brand badge (#B7410E / #FB923C)
    c.fill_rounded_rect(
        28.0 * sx,
        22.0 * sy,
        74.0 * sx,
        38.0 * sy,
        8.0 * sx,
        [183, 65, 14, 245],
    );
    c.stroke_rounded_rect(
        [28.0 * sx, 22.0 * sy, 74.0 * sx, 38.0 * sy],
        8.0 * sx,
        1.5,
        [251, 146, 60, 255],
    );
    c.draw_text(
        37.0 * sx,
        35.0 * sy,
        1.45 * sy,
        "FERRIC",
        [248, 250, 252],
        true,
    );

    c.draw_text(
        114.0 * sx,
        28.0 * sy,
        2.0 * sy,
        title,
        [241, 245, 249],
        true,
    );
    c.draw_text(
        114.0 * sx,
        50.0 * sy,
        1.25 * sy,
        "0/8 PHASE 17 BLIND REAL-DATA GATES PASSED · DETERMINISTIC nmr-core OUTPUT",
        [148, 163, 184],
        false,
    );

    // Prominent Validation Scope Badge (Top-Right)
    let bw = (badge_label.len() as f64 * 7.8 + 26.0).min(560.0) * sx;
    let bx = (1262.0 * sx) - bw - 12.0 * sx;
    c.fill_rounded_rect(
        bx,
        22.0 * sy,
        bw,
        36.0 * sy,
        8.0 * sx,
        [badge_color[0], badge_color[1], badge_color[2], 42],
    );
    c.stroke_rounded_rect(
        [bx, 22.0 * sy, bw, 36.0 * sy],
        8.0 * sx,
        1.6,
        [badge_color[0], badge_color[1], badge_color[2], 235],
    );
    c.draw_text(
        bx + 12.0 * sx,
        34.0 * sy,
        1.45 * sy,
        badge_label,
        badge_color,
        true,
    );

    // Bottom Cinema HUD Bar
    let hud_y = 638.0 * sy;
    c.fill_rounded_rect(
        18.0 * sx,
        hud_y,
        1244.0 * sx,
        68.0 * sy,
        10.0 * sx,
        [15, 23, 42, 240],
    );
    c.stroke_rounded_rect(
        [18.0 * sx, hud_y, 1244.0 * sx, 68.0 * sy],
        10.0 * sx,
        1.5,
        [51, 65, 85, 255],
    );

    // Chapter pill
    c.fill_rounded_rect(
        32.0 * sx,
        hud_y + 12.0 * sy,
        98.0 * sx,
        26.0 * sy,
        6.0 * sx,
        [14, 165, 233, 48],
    );
    c.stroke_rounded_rect(
        [32.0 * sx, hud_y + 12.0 * sy, 98.0 * sx, 26.0 * sy],
        6.0 * sx,
        1.2,
        [56, 189, 248, 220],
    );
    c.draw_text(
        46.0 * sx,
        hud_y + 19.0 * sy,
        1.55 * sy,
        &format!("CH {}/4", ch_idx + 1),
        [125, 211, 252],
        true,
    );

    c.draw_text(
        144.0 * sx,
        hud_y + 15.0 * sy,
        1.75 * sy,
        ch_title,
        [248, 250, 252],
        true,
    );
    c.draw_text(
        144.0 * sx,
        hud_y + 34.0 * sy,
        1.35 * sy,
        &format!("{ch_metric}  |  $ {cli_cmd}"),
        [148, 163, 184],
        false,
    );

    // Timecode & progress bar
    let time_str = format!("{:05.2}s / 16.00s", t_sec.clamp(0.0, TOUR_DURATION_SEC));
    c.draw_text(
        1118.0 * sx,
        hud_y + 16.0 * sy,
        1.5 * sy,
        &time_str,
        [226, 232, 240],
        true,
    );
    let bar_x = 32.0 * sx;
    let bar_w = 1216.0 * sx;
    let bar_y = hud_y + 52.0 * sy;
    c.fill_rounded_rect(bar_x, bar_y, bar_w, 7.0 * sy, 3.5 * sy, [30, 41, 59, 255]);
    let frac = (t_sec / TOUR_DURATION_SEC).clamp(0.0, 1.0);
    c.fill_rounded_rect(
        bar_x,
        bar_y,
        (bar_w * frac).max(6.0 * sx),
        7.0 * sy,
        3.5 * sy,
        [56, 189, 248, 255],
    );
    for tick in 1..4 {
        let tx = bar_x + bar_w * (tick as f64 / 4.0);
        c.draw_line_aa(
            tx,
            bar_y - 2.0 * sy,
            tx,
            bar_y + 9.0 * sy,
            1.5,
            [148, 163, 184, 220],
        );
    }
}

fn draw_stage_card(c: &mut Canvas2d, x: f64, y: f64, w: f64, h: f64, header: &str, sub: &str) {
    let sx = c.width as f64 / 1280.0;
    let sy = c.height as f64 / 720.0;
    c.fill_rounded_rect(x * sx, y * sy, w * sx, h * sy, 12.0 * sx, [15, 23, 42, 220]);
    c.stroke_rounded_rect(
        [x * sx, y * sy, w * sx, h * sy],
        12.0 * sx,
        1.5,
        [51, 65, 85, 235],
    );
    c.draw_text(
        (x + 20.0) * sx,
        (y + 18.0) * sy,
        1.85 * sy,
        header,
        [248, 250, 252],
        true,
    );
    c.draw_text(
        (x + 20.0) * sx,
        (y + 40.0) * sy,
        1.35 * sy,
        sub,
        [148, 163, 184],
        false,
    );
}

fn draw_animated_cursor(c: &mut Canvas2d, x_1280: f64, y_720: f64, local_t: f64, label: &str) {
    let sx = c.width as f64 / 1280.0;
    let sy = c.height as f64 / 720.0;
    let cx = x_1280 * sx;
    let cy = y_720 * sy;
    let pulse = (local_t * TAU * 1.5).sin() * 0.5 + 0.5;
    c.stroke_circle_aa(
        cx,
        cy,
        (11.0 + 9.0 * pulse) * sx,
        1.6,
        [251, 191, 36, (180.0 - 110.0 * pulse) as u8],
    );
    c.fill_circle_aa(cx, cy, 4.8 * sx, [251, 191, 36, 255]);
    if !label.is_empty() {
        let lw = (label.len() as f64 * 7.6 + 16.0) * sx;
        let lx = (cx + 12.0 * sx).min(c.width as f64 - lw - 24.0 * sx);
        let ly = (cy - 26.0 * sy).max(80.0 * sy);
        c.fill_rounded_rect(lx, ly, lw, 22.0 * sy, 5.0 * sx, [15, 23, 42, 230]);
        c.stroke_rounded_rect([lx, ly, lw, 22.0 * sy], 5.0 * sx, 1.1, [251, 191, 36, 220]);
        c.draw_text(
            lx + 8.0 * sx,
            ly + 6.0 * sy,
            1.3 * sy,
            label,
            [253, 224, 71],
            true,
        );
    }
}

fn draw_axes_grid(
    c: &mut Canvas2d,
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
    x_label: &str,
    y_label: &str,
) {
    let sx = c.width as f64 / 1280.0;
    let sy = c.height as f64 / 720.0;
    for k in 1..4 {
        let gy = y0 + (y1 - y0) * (k as f64 / 4.0);
        c.draw_line_aa(x0 * sx, gy * sy, x1 * sx, gy * sy, 1.0, [30, 41, 59, 210]);
        let gx = x0 + (x1 - x0) * (k as f64 / 4.0);
        c.draw_line_aa(gx * sx, y0 * sy, gx * sx, y1 * sy, 1.0, [30, 41, 59, 210]);
    }
    c.draw_line_aa(
        x0 * sx,
        y1 * sy,
        x1 * sx,
        y1 * sy,
        1.8,
        [100, 116, 139, 240],
    );
    c.draw_line_aa(
        x0 * sx,
        y0 * sy,
        x0 * sx,
        y1 * sy,
        1.8,
        [100, 116, 139, 240],
    );
    c.draw_text(
        (0.5 * (x0 + x1) - x_label.len() as f64 * 3.8) * sx,
        (y1 + 14.0) * sy,
        1.4 * sy,
        x_label,
        [148, 163, 184],
        false,
    );
    c.draw_text(
        (x0 + 8.0) * sx,
        (y0 + 6.0) * sy,
        1.35 * sy,
        y_label,
        [148, 163, 184],
        false,
    );
}

// ============================================================================
// Module 05 Scene Renderer (CEST -> CPMG -> Hungarian HSQC -> Morrison Kd)
// ============================================================================

fn render_mod05_scene(c: &mut Canvas2d, d: &Mod05Data, t_sec: f64) {
    let sx = c.width as f64 / 1280.0;
    let sy = c.height as f64 / 720.0;
    let (ch, local) = chapter_and_phase(t_sec);
    let rev = ease_in_out((local * 1.35).min(1.0));

    let (ch_title, ch_metric) = match ch {
        0 => (
            "15N CEST Saturation Sweep (B1 = 15 & 30 Hz)",
            format!(
                "Higham [6/6] Pade · pB={:.2}% · kex={:.0} s^-1 · dw=+{:.2} ppm",
                d.cest_pb * 100.0,
                d.cest_kex,
                d.cest_dw_ppm
            ),
        ),
        1 => (
            "Dual-Field Baldwin 2014 CPMG Dispersion & AICc Gate",
            format!(
                "600 & 800 MHz · pB={:.2}% · kex={:.0} s^-1 · dAICc={:.1} (p={:.2e})",
                d.cpmg_pb * 100.0,
                d.cpmg_kex,
                d.cpmg_delta_aic,
                d.cpmg_p_val
            ),
        ),
        2 => (
            "2D 1H-15N HSQC Hungarian Ligand Titration Tracking",
            format!(
                "{} trajectories tracked ([L] = 0..250 uM) · Hungarian min-cost",
                d.hsqc_trajectories.len()
            ),
        ),
        _ => (
            "Morrison Tight-Binding Quadratic Kd vs Biased Naive MM",
            format!(
                "[P]0={:.0} uM · Quad Kd={:.2} uM vs Naive MM={:.2} uM",
                d.csp_p0, d.csp_kd_quad, d.csp_kd_mm
            ),
        ),
    };

    match ch {
        0 => {
            draw_stage_card(
                c,
                26.0,
                80.0,
                1228.0,
                544.0,
                "A. 2-State Bloch-McConnell 15N CEST Saturation Profile (ILE23)",
                "Higham [6/6] Pade matrix exponential · B1 = 15 Hz (cyan) & 30 Hz (rose) revealing minor dark-state dip",
            );
            let (px0, px1, py0, py1) = (108.0, 1196.0, 152.0, 566.0);
            draw_axes_grid(
                c,
                px0,
                py0,
                px1,
                py1,
                "15N Saturation Offset (ppm, 123.5 -> 114.5)",
                "I / I0",
            );
            let w_lo = d.cest_offsets.first().copied().unwrap_or(114.5);
            let w_hi = d.cest_offsets.last().copied().unwrap_or(123.5);
            let map_x = |w: f64| px0 + (w_hi - w) / (w_hi - w_lo).max(1e-6) * (px1 - px0);
            let map_y = |v: f64| py1 - v.clamp(0.0, 1.06) * (py1 - py0);

            let minor_ppm = d.cest_major_ppm + d.cest_dw_ppm;
            let mx = map_x(minor_ppm);
            c.draw_dashed_line_aa(
                [mx * sx, py0 * sy],
                [mx * sx, py1 * sy],
                1.8,
                8.0,
                [251, 113, 133, 210],
            );
            c.draw_text(
                (mx + 8.0) * sx,
                (py0 + 18.0) * sy,
                1.5 * sy,
                &format!(
                    "Minor State B ({:.1} ppm, pB={:.2}%)",
                    minor_ppm,
                    d.cest_pb * 100.0
                ),
                [253, 164, 175],
                true,
            );

            let n_show = ((d.cest_offsets.len() as f64 * rev).ceil() as usize).max(2);
            for (curve, obs, col) in [
                (&d.cest_15_fit, &d.cest_15_obs, [56_u8, 189, 248]),
                (&d.cest_30_fit, &d.cest_30_obs, [244_u8, 63, 94]),
            ] {
                let pts: Vec<(f64, f64)> = d
                    .cest_offsets
                    .iter()
                    .zip(curve.iter())
                    .take(n_show)
                    .map(|(&w, &y)| (map_x(w) * sx, map_y(y) * sy))
                    .collect();
                c.draw_polyline_aa(&pts, 2.6, [col[0], col[1], col[2], 245]);
                for (&w, &yo) in d
                    .cest_offsets
                    .iter()
                    .zip(obs.iter())
                    .take(n_show)
                    .step_by(2)
                {
                    c.fill_circle_aa(
                        map_x(w) * sx,
                        map_y(yo) * sy,
                        3.6 * sx,
                        [col[0], col[1], col[2], 220],
                    );
                }
            }
            let scan_w = w_hi - (w_hi - w_lo) * local;
            let scan_x = map_x(scan_w);
            let scan_y = map_y(
                d.cest_30_fit
                    .get(
                        ((local * (d.cest_offsets.len() - 1) as f64).round() as usize)
                            .min(d.cest_offsets.len() - 1),
                    )
                    .copied()
                    .unwrap_or(1.0),
            );
            draw_animated_cursor(
                c,
                scan_x,
                scan_y,
                local,
                &format!("Offset {:.2} ppm", scan_w),
            );
        }
        1 => {
            draw_stage_card(
                c,
                26.0,
                80.0,
                1228.0,
                544.0,
                "B. Multi-Field Baldwin (2014) 15N CPMG Dispersion & AICc Exchange Gate",
                "600 MHz (14.1 T, cyan) & 800 MHz (18.8 T, violet) global R2,eff(nu_CPMG) fit",
            );
            let (px0, px1, py0, py1) = (108.0, 1196.0, 152.0, 566.0);
            draw_axes_grid(
                c,
                px0,
                py0,
                px1,
                py1,
                "nu_CPMG Refocusing Frequency (Hz)",
                "R2,eff (s^-1)",
            );
            let r_lo = 9.0_f64;
            let r_hi = 30.0_f64;
            let map_x = |nu: f64| px0 + (nu / 1040.0) * (px1 - px0);
            let map_y = |r2: f64| py1 - ((r2 - r_lo) / (r_hi - r_lo)).clamp(0.0, 1.0) * (py1 - py0);

            for (curve, obs, col, lbl) in [
                (
                    &d.cpmg_600_curve,
                    &d.cpmg_600_obs,
                    [56_u8, 189, 248],
                    "600 MHz (14.1 T)",
                ),
                (
                    &d.cpmg_800_curve,
                    &d.cpmg_800_obs,
                    [167_u8, 139, 250],
                    "800 MHz (18.8 T)",
                ),
            ] {
                let n_c = ((curve.len() as f64 * rev).ceil() as usize).max(2);
                let pts: Vec<(f64, f64)> = curve
                    .iter()
                    .take(n_c)
                    .map(|&(nu, r2)| (map_x(nu) * sx, map_y(r2) * sy))
                    .collect();
                c.draw_polyline_aa(&pts, 2.8, [col[0], col[1], col[2], 250]);
                for (&nu, &r2) in d.cpmg_nu.iter().zip(obs.iter()) {
                    if nu <= 1040.0 * rev {
                        c.fill_circle_aa(
                            map_x(nu) * sx,
                            map_y(r2) * sy,
                            4.5 * sx,
                            [col[0], col[1], col[2], 240],
                        );
                    }
                }
                let ly = if col[0] == 56 { py0 + 48.0 } else { py0 + 24.0 };
                c.draw_text((px1 - 240.0) * sx, ly * sy, 1.6 * sy, lbl, col, true);
            }
            // AICc gate pill
            c.fill_rounded_rect(
                760.0 * sx,
                168.0 * sy,
                390.0 * sx,
                54.0 * sy,
                8.0 * sx,
                [16, 185, 129, 36],
            );
            c.stroke_rounded_rect(
                [760.0 * sx, 168.0 * sy, 390.0 * sx, 54.0 * sy],
                8.0 * sx,
                1.4,
                [52, 211, 153, 220],
            );
            c.draw_text(
                776.0 * sx,
                182.0 * sy,
                1.55 * sy,
                &format!("AICc GATE: PASS (dAICc = {:.1})", d.cpmg_delta_aic),
                [110, 231, 183],
                true,
            );
            c.draw_text(
                776.0 * sx,
                202.0 * sy,
                1.3 * sy,
                &format!(
                    "pB = {:.2}% · kex = {:.0} s^-1",
                    d.cpmg_pb * 100.0,
                    d.cpmg_kex
                ),
                [203, 213, 225],
                false,
            );
            let cur_nu = 25.0 + 975.0 * local;
            let cur_r2 = d
                .cpmg_800_curve
                .iter()
                .min_by(|a, b| (a.0 - cur_nu).abs().total_cmp(&(b.0 - cur_nu).abs()))
                .map(|p| p.1)
                .unwrap_or(15.0);
            draw_animated_cursor(
                c,
                map_x(cur_nu),
                map_y(cur_r2),
                local,
                &format!("nu={:.0} Hz", cur_nu),
            );
        }
        2 => {
            draw_stage_card(
                c,
                26.0,
                80.0,
                1228.0,
                544.0,
                "C. 2D 1H-15N HSQC Ligand Titration Hungarian Trajectory Tracking",
                "Fast-exchange crosspeak trajectories tracked across [L] = 0 -> 250 uM without manual re-assignment",
            );
            let (px0, px1, py0, py1) = (108.0, 1196.0, 152.0, 566.0);
            draw_axes_grid(
                c,
                px0,
                py0,
                px1,
                py1,
                "1H Chemical Shift (ppm, 8.85 -> 7.75)",
                "15N (ppm)",
            );
            let map_h = |h: f64| px0 + (8.85 - h) / (8.85 - 7.75) * (px1 - px0);
            let map_n = |n: f64| py0 + (n - 115.5) / (124.0 - 115.5) * (py1 - py0);

            let pal = [
                [56_u8, 189, 248],
                [52, 211, 153],
                [251, 191, 36],
                [167, 139, 250],
                [244, 114, 182],
            ];
            let mut focus_xy = (640.0, 340.0, 0.0_f64);
            for (t_idx, tr) in d.hsqc_trajectories.iter().enumerate() {
                let col = pal[t_idx % pal.len()];
                let n_pts = tr.points.len().max(1);
                let prog = (local * (n_pts - 1) as f64).clamp(0.0, (n_pts - 1) as f64);
                let max_step = (prog.ceil() as usize + 1).min(n_pts);
                let mut line_pts = Vec::new();
                for (p_i, pt) in tr.points.iter().take(max_step).enumerate() {
                    if let Some((hp, np)) = pt.matched_peak {
                        let x = map_h(hp);
                        let y = map_n(np);
                        line_pts.push((x * sx, y * sy));
                        let r = 4.0 + (p_i as f64 / n_pts as f64) * 3.5;
                        c.stroke_circle_aa(
                            x * sx,
                            y * sy,
                            r * sx,
                            1.5,
                            [col[0], col[1], col[2], 230],
                        );
                        if t_idx == 0 && p_i + 1 == max_step {
                            focus_xy = (x, y, pt.ligand_conc_um);
                        }
                    }
                }
                c.draw_polyline_aa(&line_pts, 2.2, [col[0], col[1], col[2], 235]);
                c.draw_text(
                    (map_h(tr.apo_h_ppm) + 10.0) * sx,
                    (map_n(tr.apo_n_ppm) - 10.0) * sy,
                    1.45 * sy,
                    &format!("{}{}", tr.res_name, tr.seq_id),
                    col,
                    true,
                );
            }
            draw_animated_cursor(
                c,
                focus_xy.0,
                focus_xy.1,
                local,
                &format!("ILE44 @ [L]={:.0} uM", focus_xy.2),
            );
        }
        _ => {
            draw_stage_card(
                c,
                26.0,
                80.0,
                1228.0,
                544.0,
                "D. Tight-Binding Morrison/Williamson 1:1 Kd Isotherm vs Naive MM",
                "[P]0 = 100 uM, true Kd = 5.0 uM: quadratic mass-balance exact vs biased naive Michaelis-Menten",
            );
            let (px0, px1, py0, py1) = (108.0, 1196.0, 152.0, 566.0);
            draw_axes_grid(
                c,
                px0,
                py0,
                px1,
                py1,
                "Total Ligand [L]t (uM, 0..300)",
                "Combined CSP (ppm)",
            );
            let y_max = (d.csp_dmax * 1.15).max(0.45);
            let map_x = |l: f64| px0 + (l / d.csp_l_max) * (px1 - px0);
            let map_y = |csp: f64| py1 - (csp / y_max).clamp(0.0, 1.0) * (py1 - py0);

            let mut quad_pts = Vec::new();
            let mut mm_pts = Vec::new();
            for s in 0..=60 {
                let l = (s as f64) * (d.csp_l_max / 60.0);
                let yq = quadratic_tight_binding_csp(d.csp_dmax, d.csp_p0, l, d.csp_kd_quad);
                let ymm = naive_michaelis_menten_csp(d.csp_dmax, l, d.csp_kd_mm);
                quad_pts.push((map_x(l) * sx, map_y(yq) * sy));
                mm_pts.push((map_x(l) * sx, map_y(ymm) * sy));
            }
            c.draw_polyline_aa(&mm_pts, 2.2, [244, 63, 94, 210]);
            c.draw_polyline_aa(&quad_pts, 3.0, [52, 211, 153, 250]);
            for &(l, y) in &d.csp_obs_pts {
                c.fill_circle_aa(map_x(l) * sx, map_y(y) * sy, 5.0 * sx, [52, 211, 153, 255]);
            }

            let cur_l = 300.0 * local;
            let cur_yq = quadratic_tight_binding_csp(d.csp_dmax, d.csp_p0, cur_l, d.csp_kd_quad);
            let cur_ymm = naive_michaelis_menten_csp(d.csp_dmax, cur_l, d.csp_kd_mm);
            let cx = map_x(cur_l);
            c.draw_dashed_line_aa(
                [cx * sx, py0 * sy],
                [cx * sx, py1 * sy],
                1.5,
                6.0,
                [251, 191, 36, 200],
            );
            c.fill_circle_aa(cx * sx, map_y(cur_ymm) * sy, 5.5 * sx, [244, 63, 94, 255]);
            c.draw_text(
                830.0 * sx,
                476.0 * sy,
                1.55 * sy,
                &format!("Exact Morrison Kd = {:.2} uM", d.csp_kd_quad),
                [110, 231, 183],
                true,
            );
            c.draw_text(
                830.0 * sx,
                500.0 * sy,
                1.55 * sy,
                &format!("Biased Naive MM Kd = {:.2} uM", d.csp_kd_mm),
                [253, 164, 175],
                true,
            );
            draw_animated_cursor(
                c,
                cx,
                map_y(cur_yq),
                local,
                &format!("[L]={:.0} uM · d={:.3}", cur_l, cur_yq),
            );
        }
    }

    draw_tour_chrome(
        c,
        &TourChromeSpec {
            title: "05 · EXCHANGE & BINDING: CEST · CPMG · HSQC TITRATION · Kd",
            badge_label: "SYNTHETIC FORWARD-MODEL SHOWCASE (F13 OPEN)",
            badge_color: [251, 191, 36],
            ch_idx: ch,
            ch_title,
            ch_metric: &ch_metric,
            cli_cmd: "ferric exchange --cest cest.tsv --cpmg cpmg.tsv --csp titration.tsv",
            t_sec,
        },
    );
}

// ============================================================================
// Module 06 Scene Renderer (FID/HSVD/Whittaker -> Hoye ddd -> qNMR -> Octet)
// ============================================================================

fn render_mod06_scene(c: &mut Canvas2d, d: &Mod06Data, t_sec: f64) {
    let sx = c.width as f64 / 1280.0;
    let sy = c.height as f64 / 720.0;
    let (ch, local) = chapter_and_phase(t_sec);

    let (ch_title, ch_metric) = match ch {
        0 => (
            "1D FID -> HSVD Water Suppression -> Whittaker Baseline -> Voigt",
            format!(
                "{} Voigt peaks deconvolved · RMSD {:.3}",
                d.deconv_cluster.peaks.len(),
                d.deconv_cluster.residual_rmsd
            ),
        ),
        1 => (
            "Recursive Hoye ddd Splitting Tree & ACS Report",
            format!(
                "{} · {}",
                d.ddd_analysis.class.as_str(),
                d.acs_report.chars().take(62).collect::<String>()
            ),
        ),
        2 => (
            "Internal-Standard GUM qNMR Purity & 5*T1 Relaxation Guard",
            format!(
                "Purity = {:.2}% +- {:.2}% (k=2) · 5*T1 Guard: {}",
                d.qnmr_report.purity_percent,
                d.qnmr_report.expanded_uncertainty_k2_percent,
                if !d.qnmr_report.t1_recycle_warning {
                    "PASS"
                } else {
                    "WARN"
                }
            ),
        ),
        _ => (
            "Octet ACS 2D (S)-Ibuprofen <-> Bidirectional 1H/13C Peak Link",
            format!(
                "SMILES -> clean_2d -> perceive_cip -> {} primitives",
                d.octet_compiled.primitives.len()
            ),
        ),
    };

    match ch {
        0 => {
            draw_stage_card(
                c,
                26.0,
                80.0,
                1228.0,
                544.0,
                "A. Raw 1D FID Processing: HSVD Water Removal, Whittaker Baseline & Voigt Fit",
                "Top: Raw FID FFT with 4.78 ppm H2O (gray) · Bottom: HSVD + Whittaker flattened + Voigt clusters (cyan)",
            );
            let (px0, px1, py0, py1) = (108.0, 1196.0, 152.0, 566.0);
            draw_axes_grid(
                c,
                px0,
                py0,
                px1,
                py1,
                "1H Chemical Shift (ppm, 5.4 -> 1.8)",
                "Normalized Intensity",
            );
            let p_hi = d.fid_ppm.first().copied().unwrap_or(5.4);
            let p_lo = d.fid_ppm.last().copied().unwrap_or(1.8);
            let map_x = |p: f64| px0 + (p_hi - p) / (p_hi - p_lo).max(1e-6) * (px1 - px0);
            let max_raw = d
                .spec_with_water
                .iter()
                .copied()
                .fold(1.0_f64, f64::max)
                .max(1e-6);
            let max_flat = d
                .spec_hsvd_flat
                .iter()
                .copied()
                .fold(1.0_f64, f64::max)
                .max(1e-6);

            let morph = ease_in_out(((local - 0.15) * 1.4).clamp(0.0, 1.0));
            let mut raw_pts = Vec::with_capacity(d.fid_ppm.len());
            let mut bl_pts = Vec::with_capacity(d.fid_ppm.len());
            let mut clean_pts = Vec::with_capacity(d.fid_ppm.len());
            for i in 0..d.fid_ppm.len() {
                let x = map_x(d.fid_ppm[i]) * sx;
                let yr = py1 - (d.spec_with_water[i] / max_raw).clamp(-0.05, 1.0) * (py1 - py0);
                let yb = py1 - (d.spec_baseline_curve[i] / max_raw).clamp(-0.05, 1.0) * (py1 - py0);
                let yc = py1 - (d.spec_hsvd_flat[i] / max_flat).clamp(-0.05, 1.0) * (py1 - py0);
                raw_pts.push((x, yr * sy));
                bl_pts.push((x, yb * sy));
                clean_pts.push((x, (yr * (1.0 - morph) + yc * morph) * sy));
            }
            c.draw_polyline_aa(&raw_pts, 1.4, [100, 116, 139, 150]);
            c.draw_polyline_aa(&bl_pts, 1.6, [251, 146, 60, 180]);
            c.draw_polyline_aa(&clean_pts, 2.5, [56, 189, 248, 250]);

            // Highlight Voigt fitted centers
            for vp in &d.deconv_cluster.peaks {
                let vx = map_x(vp.center_ppm);
                c.draw_dashed_line_aa(
                    [vx * sx, (py0 + 20.0) * sy],
                    [vx * sx, py1 * sy],
                    1.2,
                    5.0,
                    [52, 211, 153, 180],
                );
            }
            c.draw_text(
                (map_x(4.78) - 65.0) * sx,
                (py0 + 20.0) * sy,
                1.5 * sy,
                "4.78 ppm H2O suppressed by HSVD",
                [253, 164, 175],
                true,
            );
            draw_animated_cursor(
                c,
                map_x(3.81),
                py1 - 220.0 * morph,
                local,
                &format!("Voigt Cluster ({} pks)", d.deconv_cluster.peaks.len()),
            );
        }
        1 => {
            draw_stage_card(
                c,
                26.0,
                80.0,
                1228.0,
                544.0,
                "B. Recursive Hoye Multiplet Peeling & ddd Splitting Tree (500 MHz)",
                &d.acs_report,
            );
            let (px0, px1, py0, py1) = (108.0, 1196.0, 152.0, 566.0);
            draw_axes_grid(
                c,
                px0,
                py0,
                px1,
                py1,
                "1H Chemical Shift (ppm, 5.88 -> 5.76)",
                "ddd Signal",
            );
            let p_hi = d.ddd_ppm.first().copied().unwrap_or(5.88);
            let p_lo = d.ddd_ppm.last().copied().unwrap_or(5.76);
            let map_x = |p: f64| px0 + (p_hi - p) / (p_hi - p_lo).max(1e-6) * (px1 - px0);
            let y_max = d.ddd_y.iter().copied().fold(1.0_f64, f64::max).max(1e-6);

            // Animated J-tree hierarchy unfolding across local time
            let js = d.ddd_analysis.j_values_hz();
            let j1 = js.first().copied().unwrap_or(14.2);
            let j2 = js.get(1).copied().unwrap_or(7.1);
            let j3 = js.get(2).copied().unwrap_or(2.4);
            let mhz = 500.0;
            let c0x = map_x(d.ddd_analysis.center_ppm);
            let (r_y, l1_y, l2_y, l3_y) = (164.0, 222.0, 282.0, 340.0);
            c.fill_circle_aa(c0x * sx, r_y * sy, 5.0 * sx, [167, 139, 250, 255]);
            for &s1 in &[-0.5_f64, 0.5] {
                let p1 = d.ddd_analysis.center_ppm + s1 * j1 / mhz;
                let x1 = map_x(p1);
                if local > 0.10 {
                    c.draw_line_aa(
                        c0x * sx,
                        r_y * sy,
                        x1 * sx,
                        l1_y * sy,
                        2.2,
                        [167, 139, 250, 240],
                    );
                }
                for &s2 in &[-0.5_f64, 0.5] {
                    let p2 = p1 + s2 * j2 / mhz;
                    let x2 = map_x(p2);
                    if local > 0.32 {
                        c.draw_line_aa(
                            x1 * sx,
                            l1_y * sy,
                            x2 * sx,
                            l2_y * sy,
                            2.0,
                            [56, 189, 248, 240],
                        );
                    }
                    for &s3 in &[-0.5_f64, 0.5] {
                        let p3 = p2 + s3 * j3 / mhz;
                        let x3 = map_x(p3);
                        if local > 0.55 {
                            c.draw_line_aa(
                                x2 * sx,
                                l2_y * sy,
                                x3 * sx,
                                l3_y * sy,
                                1.8,
                                [52, 211, 153, 240],
                            );
                            c.draw_dashed_line_aa(
                                [x3 * sx, l3_y * sy],
                                [x3 * sx, 385.0 * sy],
                                1.2,
                                4.0,
                                [52, 211, 153, 175],
                            );
                        }
                    }
                }
            }
            c.draw_text(
                (c0x - 48.0) * sx,
                192.0 * sy,
                1.5 * sy,
                &format!("J1={j1:.2} Hz"),
                [196, 181, 253],
                true,
            );
            c.draw_text(
                (map_x(d.ddd_analysis.center_ppm + 0.5 * j1 / mhz) - 48.0) * sx,
                252.0 * sy,
                1.45 * sy,
                &format!("J2={j2:.2} Hz"),
                [125, 211, 252],
                true,
            );
            c.draw_text(
                (map_x(d.ddd_analysis.center_ppm - 0.5 * j1 / mhz) - 48.0) * sx,
                312.0 * sy,
                1.4 * sy,
                &format!("J3={j3:.2} Hz"),
                [110, 231, 183],
                true,
            );

            let spec_pts: Vec<(f64, f64)> = d
                .ddd_ppm
                .iter()
                .zip(d.ddd_y.iter())
                .map(|(&p, &v)| {
                    (
                        map_x(p) * sx,
                        (py1 - (v / y_max).clamp(-0.04, 1.0) * 190.0) * sy,
                    )
                })
                .collect();
            c.draw_polyline_aa(&spec_pts, 2.5, [56, 189, 248, 250]);
            draw_animated_cursor(
                c,
                c0x,
                l2_y,
                local,
                &format!("{} (8 lines)", d.ddd_analysis.class.as_str()),
            );
        }
        2 => {
            draw_stage_card(
                c,
                26.0,
                80.0,
                1228.0,
                544.0,
                "C. Internal-Standard GUM qNMR Molar Purity & 5*T1 Relaxation Guard",
                "Analyte CH2 (4.12 ppm, 2H) vs NIST SRM Dimethyl Sulfone (3.00 ppm, 6H) · D1 + AQ = 34.0 s > 5*T1",
            );
            let (px0, px1, py0, py1) = (108.0, 820.0, 152.0, 566.0);
            draw_axes_grid(
                c,
                px0,
                py0,
                px1,
                py1,
                "1H Chemical Shift (ppm, 4.55 -> 2.65)",
                "qNMR Integral Step",
            );
            let p_hi = d.qnmr_ppm.first().copied().unwrap_or(4.55);
            let p_lo = d.qnmr_ppm.last().copied().unwrap_or(2.65);
            let map_x = |p: f64| px0 + (p_hi - p) / (p_hi - p_lo).max(1e-6) * (px1 - px0);
            let q_max = d.qnmr_y.iter().copied().fold(1.0_f64, f64::max).max(1e-6);
            let total_sum: f64 = d.qnmr_y.iter().sum::<f64>().max(1.0);
            let mut running = 0.0_f64;
            let mut spec_pts = Vec::with_capacity(d.qnmr_ppm.len());
            let mut cum_pts = Vec::with_capacity(d.qnmr_ppm.len());
            let rev = ease_in_out((local * 1.25).min(1.0));
            for (&p, &v) in d.qnmr_ppm.iter().zip(d.qnmr_y.iter()) {
                let x = map_x(p) * sx;
                let y = py1 - (v / q_max).clamp(0.0, 1.0) * 280.0;
                running += v.max(0.0);
                let cy = py1 - 20.0 - (running / total_sum) * 320.0 * rev;
                spec_pts.push((x, y * sy));
                cum_pts.push((x, cy * sy));
            }
            c.draw_polyline_aa(&spec_pts, 2.2, [56, 189, 248, 245]);
            c.draw_polyline_aa(&cum_pts, 2.8, [52, 211, 153, 250]);

            // Right summary metrology card
            c.fill_rounded_rect(
                848.0 * sx,
                152.0 * sy,
                380.0 * sx,
                414.0 * sy,
                10.0 * sx,
                [30, 41, 59, 210],
            );
            c.stroke_rounded_rect(
                [848.0 * sx, 152.0 * sy, 380.0 * sx, 414.0 * sy],
                10.0 * sx,
                1.4,
                [52, 211, 153, 200],
            );
            c.draw_text(
                870.0 * sx,
                178.0 * sy,
                1.75 * sy,
                "GUM METROLOGY BUDGET",
                [110, 231, 183],
                true,
            );
            c.draw_text(
                870.0 * sx,
                218.0 * sy,
                2.25 * sy,
                &format!(
                    "{:.2}% +- {:.2}%",
                    d.qnmr_report.purity_percent, d.qnmr_report.expanded_uncertainty_k2_percent
                ),
                [248, 250, 252],
                true,
            );
            c.draw_text(
                870.0 * sx,
                254.0 * sy,
                1.45 * sy,
                "Coverage factor k = 2 (95% CI)",
                [148, 163, 184],
                false,
            );
            c.draw_text(
                870.0 * sx,
                294.0 * sy,
                1.5 * sy,
                &format!("Analyte 4.12 ppm: {:.3} H", d.qnmr_analyte_int),
                [125, 211, 252],
                false,
            );
            c.draw_text(
                870.0 * sx,
                322.0 * sy,
                1.5 * sy,
                &format!("DMS Std 3.00 ppm: {:.3} H", d.qnmr_std_int),
                [125, 211, 252],
                false,
            );
            c.draw_text(
                870.0 * sx,
                362.0 * sy,
                1.55 * sy,
                &format!(
                    "5*T1 Guard: {} (T_rec = 34.0s >= 16.0s)",
                    if !d.qnmr_report.t1_recycle_warning {
                        "PASSED"
                    } else {
                        "FAILED"
                    }
                ),
                [110, 231, 183],
                true,
            );
            draw_animated_cursor(c, map_x(4.12), py1 - 140.0, local, "Analyte CH2 (2H)");
        }
        _ => {
            draw_stage_card(
                c,
                26.0,
                80.0,
                1228.0,
                544.0,
                "D. Octet ACS 2D (S)-Ibuprofen Depiction <-> Bidirectional 1H/13C Peak Link",
                "parse_smiles -> clean_2d -> perceive_cip -> compile_scene_primitives · Live atom-to-shift cursor",
            );
            let n_atoms = d.octet_scene.atoms.len().max(1);
            let active_atom = ((local * n_atoms as f64).floor() as usize).min(n_atoms - 1);

            // Left half: 2D Molecular Scene
            let (mx0, my0, mw, mh) = (56.0, 154.0, 560.0, 420.0);
            c.fill_rounded_rect(
                mx0 * sx,
                my0 * sy,
                mw * sx,
                mh * sy,
                10.0 * sx,
                [241, 245, 249, 250],
            );
            let sc = ((mw - 80.0) / d.octet_compiled.width_pt.max(1.0))
                .min((mh - 80.0) / d.octet_compiled.height_pt.max(1.0))
                .clamp(1.2, 3.2);
            let off_x = mx0 + 40.0 + 0.5 * (mw - 80.0 - d.octet_compiled.width_pt * sc);
            let off_y = my0 + 40.0 + 0.5 * (mh - 80.0 - d.octet_compiled.height_pt * sc);
            let map_sx = |x_pt: f64| off_x + x_pt * sc;
            let map_sy = |y_pt: f64| off_y + y_pt * sc;

            // Active atom highlight halo
            let bond_len = JournalStyle::Acs1996.to_style().bond_length_pt;
            let mut active_xy = (mx0 + 180.0, my0 + 180.0);
            if let Some(sa) = d.octet_scene.atoms.get(active_atom) {
                let ax = map_sx(sa.x * bond_len + d.octet_compiled.shift_x_pt);
                let ay = map_sy(-sa.y * bond_len + d.octet_compiled.shift_y_pt);
                active_xy = (ax, ay);
                c.fill_circle_aa(ax * sx, ay * sy, 18.0 * sx, [251, 191, 36, 95]);
                c.stroke_circle_aa(ax * sx, ay * sy, 18.0 * sx, 2.2, [217, 119, 6, 245]);
            }

            for prim in &d.octet_compiled.primitives {
                match prim {
                    VectorPrimitive::Line {
                        p1, p2, width_pt, ..
                    } => {
                        c.draw_line_aa(
                            map_sx(p1[0]) * sx,
                            map_sy(p1[1]) * sy,
                            map_sx(p2[0]) * sx,
                            map_sy(p2[1]) * sy,
                            (width_pt * sc).max(1.8),
                            [15, 23, 42, 255],
                        );
                    }
                    VectorPrimitive::Polygon { points, .. } => {
                        if points.len() >= 3 {
                            for w in points.windows(2) {
                                c.draw_line_aa(
                                    map_sx(w[0][0]) * sx,
                                    map_sy(w[0][1]) * sy,
                                    map_sx(w[1][0]) * sx,
                                    map_sy(w[1][1]) * sy,
                                    2.5,
                                    [15, 23, 42, 255],
                                );
                            }
                            if let (Some(first), Some(last)) = (points.first(), points.last()) {
                                c.draw_line_aa(
                                    map_sx(last[0]) * sx,
                                    map_sy(last[1]) * sy,
                                    map_sx(first[0]) * sx,
                                    map_sy(first[1]) * sy,
                                    2.5,
                                    [15, 23, 42, 255],
                                );
                            }
                        }
                    }
                    VectorPrimitive::Path { cmds, .. } => {
                        let mut cur = [0.0_f64, 0.0];
                        for cmd in cmds {
                            match *cmd {
                                PathCmd::MoveTo(x, y) => cur = [x, y],
                                PathCmd::LineTo(x, y) => {
                                    c.draw_line_aa(
                                        map_sx(cur[0]) * sx,
                                        map_sy(cur[1]) * sy,
                                        map_sx(x) * sx,
                                        map_sy(y) * sy,
                                        1.8,
                                        [15, 23, 42, 255],
                                    );
                                    cur = [x, y];
                                }
                                PathCmd::CubicTo(_, _, _, _, x, y) => {
                                    c.draw_line_aa(
                                        map_sx(cur[0]) * sx,
                                        map_sy(cur[1]) * sy,
                                        map_sx(x) * sx,
                                        map_sy(y) * sy,
                                        1.8,
                                        [15, 23, 42, 255],
                                    );
                                    cur = [x, y];
                                }
                                PathCmd::Close => {}
                            }
                        }
                    }
                    VectorPrimitive::Circle {
                        center, radius_pt, ..
                    } => {
                        c.stroke_circle_aa(
                            map_sx(center[0]) * sx,
                            map_sy(center[1]) * sy,
                            radius_pt * sc * sx,
                            1.8,
                            [15, 23, 42, 255],
                        );
                    }
                    VectorPrimitive::Text { pos, text, .. } => {
                        c.draw_text(
                            map_sx(pos[0]) * sx,
                            (map_sy(pos[1]) - 6.0) * sy,
                            1.65 * sy,
                            text,
                            [15, 23, 42],
                            false,
                        );
                    }
                }
            }

            // Right half: Linked 1H Stick Spectrum + 13C Table
            let (rx0, rx1, ry0, ry1) = (650.0, 1216.0, 166.0, 420.0);
            draw_axes_grid(
                c,
                rx0,
                ry0,
                rx1,
                ry1,
                "Predicted 1H Stick Multiplet (12.0 -> 0.5 ppm)",
                "Intensity",
            );
            let mut active_h_shift = None;
            for &(atom_idx, ppm, nh, _) in &d.octet_1h_peaks {
                let x = rx0 + (12.0 - ppm.clamp(0.5, 12.0)) / 11.5 * (rx1 - rx0);
                let h_px = (nh * 48.0).clamp(45.0, 210.0);
                let is_sel = atom_idx == active_atom;
                let col = if is_sel {
                    [251_u8, 191, 36, 255]
                } else {
                    [56_u8, 189, 248, 210]
                };
                c.draw_line_aa(
                    x * sx,
                    ry1 * sy,
                    x * sx,
                    (ry1 - h_px) * sy,
                    if is_sel { 4.0 } else { 2.0 },
                    col,
                );
                if is_sel {
                    active_h_shift = Some(ppm);
                }
            }
            let active_c_shift = d
                .octet_13c_shifts
                .iter()
                .find(|x| x.0 == active_atom)
                .map(|x| x.1);
            c.fill_rounded_rect(
                650.0 * sx,
                466.0 * sy,
                566.0 * sx,
                108.0 * sy,
                10.0 * sx,
                [30, 41, 59, 215],
            );
            c.stroke_rounded_rect(
                [650.0 * sx, 466.0 * sy, 566.0 * sx, 108.0 * sy],
                10.0 * sx,
                1.4,
                [251, 191, 36, 210],
            );
            c.draw_text(
                670.0 * sx,
                486.0 * sy,
                1.75 * sy,
                &format!(
                    "LINKED ATOM #{active_atom} ({})",
                    d.octet_scene
                        .atoms
                        .get(active_atom)
                        .map(|a| a.label.as_str())
                        .unwrap_or("C")
                ),
                [253, 224, 71],
                true,
            );
            c.draw_text(
                670.0 * sx,
                516.0 * sy,
                1.55 * sy,
                &format!(
                    "1H Shift: {}  |  13C{{1H}} Shift: {}",
                    active_h_shift
                        .map(|v| format!("{v:.2} ppm"))
                        .unwrap_or_else(|| "quaternary / non-CH".to_string()),
                    active_c_shift
                        .map(|v| format!("{v:.1} ppm"))
                        .unwrap_or_else(|| "heteroatom".to_string())
                ),
                [226, 232, 240],
                false,
            );
            draw_animated_cursor(
                c,
                active_xy.0,
                active_xy.1,
                local,
                &format!("Atom #{active_atom}"),
            );
        }
    }

    draw_tour_chrome(
        c,
        &TourChromeSpec {
            title: "06 · 1D NMR: HSVD/WHITTAKER · HOYE ddd · GUM qNMR · OCTET",
            badge_label: "SYNTHETIC FORWARD-MODEL SHOWCASE (F14 OPEN)",
            badge_color: [251, 191, 36],
            ch_idx: ch,
            ch_title,
            ch_metric: &ch_metric,
            cli_cmd: "ferric nmr1d --fid fid --hsvd --baseline whittaker && ferric nmr1d qnmr",
            t_sec,
        },
    );
}

// ============================================================================
// Module 07 Scene Renderer (BPP-LED Stack -> Stejskal-Tanner -> 2D/ILT -> Rh)
// ============================================================================

fn render_mod07_scene(c: &mut Canvas2d, d: &Mod07Data, t_sec: f64) {
    let sx = c.width as f64 / 1280.0;
    let sy = c.height as f64 / 720.0;
    let (ch, local) = chapter_and_phase(t_sec);

    let (ch_title, ch_metric) = match ch {
        0 => (
            "Pseudo-2D BPP-LED Gradient Stack (g1 -> g16, SINE.100)",
            format!(
                "16 gradient planes (2.0..52.0 G/cm) · 2/pi shape factor · b_max={:.2}e9 s/m^2",
                d.b_vals.last().copied().unwrap_or(0.0) * 1.0e-9
            ),
        ),
        1 => (
            "Stejskal-Tanner Attenuation & BIC K=1 vs K=2 Resolution",
            format!(
                "3.66 ppm overlap resolved to K={} (Raffinose + Maleate)",
                d.dosy_peaks[3].multi_exp.selected_k
            ),
        ),
        2 => (
            "2D (ppm x log10 D) Contour Map & Tikhonov ILT Comparison",
            format!(
                "ILT GCV lambda={:.2e} ({} continuous modes) vs discrete BIC",
                d.ilt_overlap.selected_lambda,
                d.ilt_overlap.modes.len()
            ),
        ),
        _ => (
            "Evans Stokes-Einstein Hydrodynamic Radius Rh & MW Sizing",
            format!(
                "Maleate Rh={:.1}A · Raffinose Rh={:.1}A · Lysozyme Rh={:.1}A (~{:.1} kDa)",
                d.hydro_maleate.r_h_angstrom,
                d.hydro_raffinose.r_h_angstrom,
                d.hydro_lyso.r_h_angstrom,
                d.hydro_lyso.estimated_mw_da / 1000.0
            ),
        ),
    };

    match ch {
        0 => {
            draw_stage_card(
                c,
                26.0,
                80.0,
                1228.0,
                544.0,
                "A. Pseudo-2D BPP-LED Gradient-Attenuated 1D Stack (g1 -> g16)",
                "Fast-diffusing Maleate (6.02 ppm) decays rapidly while macromolecular Lysozyme (0.88 ppm) persists",
            );
            let (px0, px1, py0, py1) = (108.0, 1196.0, 152.0, 566.0);
            draw_axes_grid(
                c,
                px0,
                py0,
                px1,
                py1,
                "1H Chemical Shift (ppm, 7.8 -> 0.4)",
                "Stacked Gradient Planes g1..g16",
            );
            let p_hi = d.ppm_axis.first().copied().unwrap_or(7.8);
            let p_lo = d.ppm_axis.last().copied().unwrap_or(0.4);
            let active_g =
                ((local * 15.0).round() as usize).min(d.stack_slices.len().saturating_sub(1));

            for g in (0..d.stack_slices.len()).step_by(2).rev() {
                let slice = &d.stack_slices[g];
                let tilt_x = (g as f64) * 5.0;
                let base_y = py1 - 14.0 - (g as f64) * 21.0;
                let span_x = (px1 - px0) - 90.0;
                let is_sel = g == active_g || g + 1 == active_g;
                let col = if is_sel {
                    [251_u8, 191, 36, 255]
                } else {
                    [56_u8, 189, 248, 185]
                };
                let pts: Vec<(f64, f64)> = d
                    .ppm_axis
                    .iter()
                    .zip(slice.iter())
                    .map(|(&p, &yv)| {
                        let x = px0 + tilt_x + (p_hi - p) / (p_hi - p_lo) * span_x;
                        let y = base_y - yv.clamp(0.0, 1.2) * 85.0;
                        (x * sx, y * sy)
                    })
                    .collect();
                c.draw_polyline_aa(&pts, if is_sel { 2.6 } else { 1.3 }, col);
            }
            let cur_y = py1 - 14.0 - (active_g as f64) * 21.0 - 45.0;
            draw_animated_cursor(
                c,
                px0 + (active_g as f64) * 5.0 + 220.0,
                cur_y,
                local,
                &format!("g{} ({:.1} G/cm)", active_g + 1, d.gradients_g_cm[active_g]),
            );
        }
        1 => {
            draw_stage_card(
                c,
                26.0,
                80.0,
                1228.0,
                544.0,
                "B. Stejskal-Tanner Decay Curves & BIC K=1 vs K=2 Mixture Resolution",
                "I(b) = sum A_k exp(-b D_k): 3.66 ppm overlap selects K=2 bi-exponential via BIC",
            );
            let (px0, px1, py0, py1) = (108.0, 1196.0, 152.0, 566.0);
            draw_axes_grid(
                c,
                px0,
                py0,
                px1,
                py1,
                "Stejskal-Tanner b-value (10^9 s/m^2)",
                "Normalized I(b) / I0",
            );
            let b_max = d.b_vals.iter().copied().fold(1.0_f64, f64::max);
            let map_x = |b: f64| px0 + (b / b_max) * (px1 - px0);
            let map_y = |v: f64| py1 - v.clamp(0.0, 1.05) * (py1 - py0);
            let rev = ease_in_out((local * 1.25).min(1.0));

            for &(pk_idx, col, tag) in &[
                (1usize, [52_u8, 211, 153], "6.02 ppm Maleate (K=1)"),
                (3usize, [244_u8, 63, 94], "3.66 ppm Overlap (K=2 BIC)"),
                (5usize, [167_u8, 139, 250], "0.88 ppm Lysozyme (K=1)"),
            ] {
                let pk = &d.dosy_peaks[pk_idx];
                let i0 = pk.intensities.first().copied().unwrap_or(1.0).max(1e-6);
                let n_steps = ((60.0 * rev).ceil() as usize).max(2);
                let mut pts = Vec::with_capacity(n_steps);
                for s in 0..=n_steps {
                    let b = (s as f64) / 60.0 * b_max;
                    let mut yf = 0.0;
                    for comp in &pk.multi_exp.best_fit.components {
                        yf += comp.amplitude * (-b * comp.d_m2_s).exp();
                    }
                    pts.push((map_x(b) * sx, map_y(yf / i0) * sy));
                }
                c.draw_polyline_aa(&pts, 2.8, [col[0], col[1], col[2], 245]);
                for (&b, &yi) in d.b_vals.iter().zip(pk.intensities.iter()) {
                    if b <= b_max * rev + 1e-6 {
                        c.fill_circle_aa(
                            map_x(b) * sx,
                            map_y(yi / i0) * sy,
                            4.2 * sx,
                            [col[0], col[1], col[2], 240],
                        );
                    }
                }
                let ly = if pk_idx == 1 {
                    py0 + 66.0
                } else if pk_idx == 3 {
                    py0 + 42.0
                } else {
                    py0 + 18.0
                };
                c.draw_text(830.0 * sx, ly * sy, 1.55 * sy, tag, col, true);
            }
            let cur_b = b_max * local;
            draw_animated_cursor(
                c,
                map_x(cur_b),
                map_y((-cur_b * 3.1e-10).exp()),
                local,
                &format!("b={:.2}e9", cur_b * 1e-9),
            );
        }
        2 => {
            draw_stage_card(
                c,
                26.0,
                80.0,
                1228.0,
                544.0,
                "C. 2D (ppm x log10 D) DOSY Map & Continuous Tikhonov ILT Comparison",
                "Left: 2D DOSY grid separating 3 species · Right: 3.66 ppm overlap continuous ILT f(D) vs BIC sticks",
            );
            // Left 2D map
            let (px0, px1, py0, py1) = (96.0, 810.0, 152.0, 566.0);
            draw_axes_grid(
                c,
                px0,
                py0,
                px1,
                py1,
                "1H Chemical Shift (ppm, 7.8 -> 0.4)",
                "log10 D (m^2/s)",
            );
            let p_hi = 7.8_f64;
            let p_lo = 0.4_f64;
            let ld_min = -10.35_f64;
            let ld_max = -8.75_f64;
            let map_x = |p: f64| px0 + (p_hi - p) / (p_hi - p_lo) * (px1 - px0);
            let map_y = |ld: f64| py0 + (ld_max - ld) / (ld_max - ld_min) * (py1 - py0);

            for (ld_band, col, label) in [
                (-9.076, [52_u8, 211, 153], "Maleate (-9.08)"),
                (-9.509, [56_u8, 189, 248], "Raffinose (-9.51)"),
                (-9.951, [167_u8, 139, 250], "Lysozyme (-9.95)"),
            ] {
                let y = map_y(ld_band);
                c.draw_dashed_line_aa(
                    [px0 * sx, y * sy],
                    [px1 * sx, y * sy],
                    1.4,
                    6.0,
                    [col[0], col[1], col[2], 185],
                );
                c.draw_text(
                    (px0 + 10.0) * sx,
                    (y - 14.0) * sy,
                    1.35 * sy,
                    label,
                    col,
                    true,
                );
            }
            for pk in &d.dosy_peaks {
                for comp in &pk.multi_exp.best_fit.components {
                    let cx = map_x(pk.center_ppm);
                    let cy = map_y(comp.log10_d);
                    for (rad, alpha) in [(16.0, 90_u8), (10.0, 160_u8), (5.0, 240_u8)] {
                        c.stroke_circle_aa(cx * sx, cy * sy, rad * sx, 1.5, [56, 189, 248, alpha]);
                    }
                    c.fill_circle_aa(cx * sx, cy * sy, 3.8 * sx, [244, 63, 94, 255]);
                }
            }

            // Right Tikhonov ILT comparison
            let (ix0, ix1, iy0, iy1) = (846.0, 1226.0, 152.0, 566.0);
            draw_axes_grid(c, ix0, iy0, ix1, iy1, "ILT Spectrum f(D)", "log10 D");
            let ilt_max = d
                .ilt_overlap
                .spectrum
                .iter()
                .copied()
                .fold(1e-9_f64, f64::max);
            let ilt_pts: Vec<(f64, f64)> = d
                .ilt_overlap
                .log10_d_grid
                .iter()
                .zip(d.ilt_overlap.spectrum.iter())
                .map(|(&ld, &val)| {
                    let x = ix0 + (val / ilt_max).clamp(0.0, 1.0) * (ix1 - ix0) * 0.85;
                    let y = iy0 + (ld_max - ld) / (ld_max - ld_min) * (iy1 - iy0);
                    (x * sx, y * sy)
                })
                .collect();
            let _ = (&d.grid.data, &d.top_1d);
            c.draw_polyline_aa(&ilt_pts, 2.5, [251, 191, 36, 250]);
            for comp in &d.dosy_peaks[3].multi_exp.best_fit.components {
                let y = iy0 + (ld_max - comp.log10_d) / (ld_max - ld_min) * (iy1 - iy0);
                c.draw_line_aa(
                    ix0 * sx,
                    y * sy,
                    (ix1 - 24.0) * sx,
                    y * sy,
                    2.0,
                    [244, 63, 94, 220],
                );
            }
            c.draw_text(
                (ix0 + 14.0) * sx,
                (iy0 + 18.0) * sy,
                1.4 * sy,
                &format!(
                    "Tikhonov ILT (GCV lambda={:.1e})",
                    d.ilt_overlap.selected_lambda
                ),
                [253, 224, 71],
                true,
            );
            draw_animated_cursor(c, map_x(3.66), map_y(-9.509), local, "3.66 ppm Bi-modal");
        }
        _ => {
            draw_stage_card(
                c,
                26.0,
                80.0,
                1228.0,
                544.0,
                "D. Evans Stokes-Einstein Hydrodynamic Radius Rh & MW Calibration",
                "Solvent viscosity eta(D2O, 298.15 K) maps measured diffusivity D to Rh (Angstrom) and MW (Da)",
            );
            let cards = [
                (
                    "Maleate (Internal Ref)",
                    &d.hydro_maleate,
                    [52_u8, 211, 153],
                    64.0,
                ),
                (
                    "Raffinose (Trisaccharide)",
                    &d.hydro_raffinose,
                    [56_u8, 189, 248],
                    454.0,
                ),
                (
                    "Lysozyme (Globular Protein)",
                    &d.hydro_lyso,
                    [167_u8, 139, 250],
                    844.0,
                ),
            ];
            let pulse = 0.88 + 0.12 * (local * TAU).sin();
            for (idx, (name, est, col, cx0)) in cards.iter().enumerate() {
                c.fill_rounded_rect(
                    *cx0 * sx,
                    162.0 * sy,
                    368.0 * sx,
                    412.0 * sy,
                    12.0 * sx,
                    [30, 41, 59, 200],
                );
                c.stroke_rounded_rect(
                    [*cx0 * sx, 162.0 * sy, 368.0 * sx, 412.0 * sy],
                    12.0 * sx,
                    1.6,
                    [col[0], col[1], col[2], 210],
                );
                c.draw_text((*cx0 + 20.0) * sx, 184.0 * sy, 1.75 * sy, name, *col, true);
                let sphere_r = (est.r_h_angstrom * 3.8).clamp(16.0, 82.0) * pulse;
                let scx = *cx0 + 184.0;
                let scy = 324.0;
                c.fill_circle_aa(
                    scx * sx,
                    scy * sy,
                    sphere_r * sx,
                    [col[0], col[1], col[2], 55],
                );
                c.stroke_circle_aa(
                    scx * sx,
                    scy * sy,
                    sphere_r * sx,
                    2.5,
                    [col[0], col[1], col[2], 245],
                );
                c.draw_text(
                    (*cx0 + 20.0) * sx,
                    450.0 * sy,
                    1.85 * sy,
                    &format!("Rh = {:.2} A", est.r_h_angstrom),
                    [248, 250, 252],
                    true,
                );
                c.draw_text(
                    (*cx0 + 20.0) * sx,
                    482.0 * sy,
                    1.6 * sy,
                    &format!("Est. MW ~ {:.0} Da", est.estimated_mw_da),
                    [203, 213, 225],
                    false,
                );
                c.draw_text(
                    (*cx0 + 20.0) * sx,
                    510.0 * sy,
                    1.45 * sy,
                    &format!("D = {:.2e} m^2/s", est.d_m2_s),
                    [148, 163, 184],
                    false,
                );
                if idx == ((local * 3.0).floor() as usize).min(2) {
                    draw_animated_cursor(
                        c,
                        scx,
                        scy,
                        local,
                        &format!("Rh={:.1} A", est.r_h_angstrom),
                    );
                }
            }
        }
    }

    draw_tour_chrome(
        c,
        &TourChromeSpec {
            title: "07 · DOSY DIFFUSION: BPP-LED · STEJSKAL-TANNER · ILT · Rh",
            badge_label: "SYNTHETIC FORWARD-MODEL SHOWCASE (F16 OPEN)",
            badge_color: [251, 191, 36],
            ch_idx: ch,
            ch_title,
            ch_metric: &ch_metric,
            cli_cmd: "ferric dosy fit --peaks dosy_peaks.tsv --seq bpp-led --ilt",
            t_sec,
        },
    );
}

// ============================================================================
// Module 08 Scene Renderer (Cohort -> PCA/OPLS-DA -> STOCSY -> MCP + Undo)
// ============================================================================

fn render_mod08_scene(c: &mut Canvas2d, d: &Mod08Data, t_sec: f64) {
    let sx = c.width as f64 / 1280.0;
    let sy = c.height as f64 / 720.0;
    let (ch, local) = chapter_and_phase(t_sec);
    let opls = d.study.opls_da.as_ref();
    let q2 = opls.map(|o| o.q2).unwrap_or(0.0);
    let perm_p = opls.map(|o| o.permutation.p_value_q2).unwrap_or(1.0);

    let (ch_title, ch_metric) = match ch {
        0 => (
            "24-Sample Biofluid Cohort: AI-Binning & PQN Normalization",
            format!(
                "N={} (12 Control vs 12 Case) · {} De Meyer AI-Bins · PQN",
                d.study.sample_ids.len(),
                d.study.buckets.len()
            ),
        ),
        1 => (
            "PCA 95% Hotelling T^2 & 100-Perm Leak-Free OPLS-DA",
            format!(
                "Stratified CV OPLS-DA Q^2 = {:.3} · perm p = {:.4}",
                q2, perm_p
            ),
        ),
        2 => (
            "STOCSY 1.33 ppm Driver Spin-System Identification",
            format!(
                "Driver {:.2} ppm -> {}",
                d.stocsy_lactate.driver_ppm,
                d.stocsy_lactate
                    .metabolite_matches
                    .first()
                    .map(|m| format!("{} (score={:.2})", m.name, m.score))
                    .unwrap_or_else(|| "L-Lactate".to_string())
            ),
        ),
        _ => (
            "Sandboxed ReadOnly MCP JSON-RPC 2.0 & 256-Step Undo Bridge",
            format!(
                "HomologFirewall <= 25% · ReadOnly write denied · Undo bitwise_eq={}",
                d.undo_steps.last().map(|s| s.3).unwrap_or(false)
            ),
        ),
    };

    match ch {
        0 => {
            draw_stage_card(
                c,
                26.0,
                80.0,
                1228.0,
                544.0,
                "A. 24-Sample Biofluid Cohort: De Meyer Valley-Seeking AI-Binning + PQN",
                "Mean Control (cyan) vs Case (rose) PQN-normalized spectra avoiding TSA dilution distortion",
            );
            let (px0, px1, py0, py1) = (108.0, 1196.0, 152.0, 566.0);
            draw_axes_grid(
                c,
                px0,
                py0,
                px1,
                py1,
                "1H Chemical Shift (ppm, 4.45 -> 0.85)",
                "PQN Intensity",
            );
            let sp_hi = 4.45_f64;
            let sp_lo = 0.85_f64;
            let map_x = |p: f64| px0 + (sp_hi - p) / (sp_hi - sp_lo) * (px1 - px0);
            let mut max_y = 1e-9_f64;
            let mut ctrl_pts = Vec::new();
            let mut case_pts = Vec::new();
            for m in 0..d.study.common_ppm_grid.len() {
                let p = d.study.common_ppm_grid[m];
                if p < sp_lo || p > sp_hi || d.study.masked_grid_points[m] {
                    continue;
                }
                max_y = max_y.max(d.mean_ctrl[m]).max(d.mean_case[m]);
            }
            for m in 0..d.study.common_ppm_grid.len() {
                let p = d.study.common_ppm_grid[m];
                if p < sp_lo || p > sp_hi || d.study.masked_grid_points[m] {
                    continue;
                }
                let x = map_x(p) * sx;
                let yc = (py1 - (d.mean_ctrl[m] / max_y).clamp(0.0, 1.0) * (py1 - py0 - 20.0)) * sy;
                let yk = (py1 - (d.mean_case[m] / max_y).clamp(0.0, 1.0) * (py1 - py0 - 20.0)) * sy;
                ctrl_pts.push((x, yc));
                case_pts.push((x, yk));
            }
            c.draw_polyline_aa(&ctrl_pts, 2.0, [56, 189, 248, 235]);
            c.draw_polyline_aa(&case_pts, 2.2, [244, 63, 94, 245]);

            // Show AI-Binning vertical boundaries swept by cursor
            let sweep_ppm = sp_hi - (sp_hi - sp_lo) * local;
            for b in d.study.buckets.iter().step_by(3) {
                if b.center_ppm >= sp_lo && b.center_ppm <= sp_hi {
                    let bx = map_x(b.center_ppm);
                    c.draw_line_aa(
                        bx * sx,
                        (py1 - 24.0) * sy,
                        bx * sx,
                        py1 * sy,
                        1.0,
                        [52, 211, 153, 140],
                    );
                }
            }
            draw_animated_cursor(
                c,
                map_x(sweep_ppm),
                py0 + 180.0,
                local,
                &format!("AI-Bin @ {sweep_ppm:.2} ppm"),
            );
        }
        1 => {
            draw_stage_card(
                c,
                26.0,
                80.0,
                1228.0,
                544.0,
                "B. PCA Scores (95% Hotelling T^2), OPLS-DA Permutation Guard & Wiklund S-Plot",
                &format!(
                    "Leak-free stratified CV Q^2 = {q2:.3} · 100-permutation null p = {perm_p:.4}"
                ),
            );
            // Left: PCA + 95% Hotelling T^2
            let (ax0, ax1, ay0, ay1) = (86.0, 620.0, 156.0, 560.0);
            draw_axes_grid(c, ax0, ay0, ax1, ay1, "PC1 Score", "PC2 Score (95% T^2)");
            let ell = d.study.pca.global_ellipse_95.sample_points(64);
            let mut max_ax = 1.0_f64;
            let mut max_ay = 1.0_f64;
            for pt in &ell {
                max_ax = max_ax.max(pt[0].abs() * 1.15);
                max_ay = max_ay.max(pt[1].abs() * 1.15);
            }
            for sc in &d.study.pca.scores {
                max_ax = max_ax.max(sc.first().copied().unwrap_or(0.0).abs() * 1.15);
                max_ay = max_ay.max(sc.get(1).copied().unwrap_or(0.0).abs() * 1.15);
            }
            let map_px = |v: f64| 0.5 * (ax0 + ax1) + (v / max_ax) * 0.5 * (ax1 - ax0);
            let map_py = |v: f64| 0.5 * (ay0 + ay1) - (v / max_ay) * 0.5 * (ay1 - ay0);
            let ell_pts: Vec<(f64, f64)> = ell
                .iter()
                .map(|pt| (map_px(pt[0]) * sx, map_py(pt[1]) * sy))
                .collect();
            c.draw_polyline_aa(&ell_pts, 1.8, [251, 191, 36, 210]);
            for (idx, (sc, grp)) in d
                .study
                .pca
                .scores
                .iter()
                .zip(d.study.sample_groups.iter())
                .enumerate()
            {
                if (idx as f64) <= local * 24.0 + 1.0 {
                    let is_case = grp.to_ascii_lowercase().contains("case")
                        || grp != &d.study.sample_groups[0];
                    let col = if is_case {
                        [244_u8, 63, 94, 245]
                    } else {
                        [56_u8, 189, 248, 245]
                    };
                    c.fill_circle_aa(
                        map_px(sc.first().copied().unwrap_or(0.0)) * sx,
                        map_py(sc.get(1).copied().unwrap_or(0.0)) * sy,
                        5.2 * sx,
                        col,
                    );
                }
            }

            // Right: Wiklund S-plot
            let (bx0, bx1, by0, by1) = (676.0, 1216.0, 156.0, 560.0);
            draw_axes_grid(
                c,
                bx0,
                by0,
                bx1,
                by1,
                "OPLS-DA Covariance Cov(tp, X)",
                "Reliability p(corr)",
            );
            if let Some(op) = opls {
                let max_cov = op
                    .s_plot_cov
                    .iter()
                    .copied()
                    .map(f64::abs)
                    .fold(0.1_f64, f64::max)
                    * 1.1;
                let map_sx = |v: f64| 0.5 * (bx0 + bx1) + (v / max_cov) * 0.5 * (bx1 - bx0);
                let map_sy = |v: f64| 0.5 * (by0 + by1) - v.clamp(-1.0, 1.0) * 0.45 * (by1 - by0);
                for j in 0..d.study.buckets.len() {
                    let cov = op.s_plot_cov.get(j).copied().unwrap_or(0.0);
                    let corr = op.s_plot_corr.get(j).copied().unwrap_or(0.0);
                    let vip = op.vip_predictive.get(j).copied().unwrap_or(0.0);
                    let hit = vip >= 1.2 && corr.abs() >= 0.7;
                    let col = if hit && corr > 0.0 {
                        [244_u8, 63, 94, 245]
                    } else if hit {
                        [56_u8, 189, 248, 245]
                    } else {
                        [148_u8, 163, 184, 150]
                    };
                    c.fill_circle_aa(
                        map_sx(cov) * sx,
                        map_sy(corr) * sy,
                        if hit { 4.5 * sx } else { 2.5 * sx },
                        col,
                    );
                }
            }
            draw_animated_cursor(c, 1120.0, 210.0, local, &format!("OPLS-DA Q2={q2:.3}"));
        }
        2 => {
            draw_stage_card(
                c,
                26.0,
                80.0,
                1228.0,
                544.0,
                "C. STOCSY 1.33 ppm Driver Correlation Spectrum -> L-Lactate (3:1 Stoichiometry)",
                "Intramolecular r^2 correlation lights up coupled 1.33 ppm (CH3) & 4.11 ppm (CH) L-Lactate spin system",
            );
            let (px0, px1, py0, py1) = (108.0, 1196.0, 152.0, 566.0);
            draw_axes_grid(
                c,
                px0,
                py0,
                px1,
                py1,
                "1H Chemical Shift (ppm, 4.45 -> 0.85)",
                "Correlation-Colored Mean Spectrum",
            );
            let sp_hi = 4.45_f64;
            let sp_lo = 0.85_f64;
            let map_x = |p: f64| px0 + (sp_hi - p) / (sp_hi - sp_lo) * (px1 - px0);
            let mut max_y = 1e-9_f64;
            let mut pts: Vec<(f64, f64, f64)> = Vec::new();
            for m in 0..d.study.common_ppm_grid.len() {
                let p = d.study.common_ppm_grid[m];
                if p < sp_lo || p > sp_hi || d.study.masked_grid_points[m] {
                    continue;
                }
                let y = d.mean_case[m];
                max_y = max_y.max(y);
                let r2 = d.stocsy_lactate.spectral_r2.get(m).copied().unwrap_or(0.0);
                pts.push((p, y, r2));
            }
            for pair in pts.windows(2) {
                let (p0, y0, r0) = pair[0];
                let (p1, y1, r1) = pair[1];
                if (p0 - p1).abs() > 0.03 {
                    continue;
                }
                let r2 = 0.5 * (r0 + r1);
                let col = if r2 > 0.82 {
                    [244_u8, 63, 94, 255]
                } else if r2 > 0.55 {
                    [251_u8, 191, 36, 240]
                } else if r2 > 0.30 {
                    [52_u8, 211, 153, 215]
                } else {
                    [100_u8, 116, 139, 175]
                };
                c.draw_line_aa(
                    map_x(p0) * sx,
                    (py1 - (y0 / max_y).clamp(0.0, 1.0) * (py1 - py0 - 24.0)) * sy,
                    map_x(p1) * sx,
                    (py1 - (y1 / max_y).clamp(0.0, 1.0) * (py1 - py0 - 24.0)) * sy,
                    if r2 > 0.75 { 2.8 } else { 1.6 },
                    col,
                );
            }
            let drv_x = map_x(d.stocsy_lactate.driver_ppm);
            c.draw_dashed_line_aa(
                [drv_x * sx, py0 * sy],
                [drv_x * sx, py1 * sy],
                1.6,
                6.0,
                [244, 63, 94, 235],
            );
            draw_animated_cursor(
                c,
                drv_x,
                py0 + 120.0,
                local,
                &format!("Driver {:.2} ppm (L-Lactate)", d.stocsy_lactate.driver_ppm),
            );
        }
        _ => {
            draw_stage_card(
                c,
                26.0,
                80.0,
                1228.0,
                544.0,
                "D. Sandboxed MCP JSON-RPC 2.0 ReadOnly Copilot & 256-Step Undo Bridge",
                "Live nmr_core::mcp server enforces <=25% HomologFirewall & blocks mutating writes; CommandHistory restores bitwise equality",
            );
            let stages = [
                (
                    "1. JSON-RPC 2.0 Init",
                    "McpSecurityPolicy::read_only",
                    [56_u8, 189, 248],
                ),
                (
                    "2. HomologFirewall",
                    "predict_homolog_free_torsions <=25%",
                    [52_u8, 211, 153],
                ),
                (
                    "3. ReadOnly Guardrail",
                    "write_curated_peak_tsv -> DENIED",
                    [244_u8, 63, 94],
                ),
                (
                    "4. 256-Step Undo Stack",
                    "CommandHistory -> bitwise_eq = true",
                    [167_u8, 139, 250],
                ),
            ];
            let active = ((local * 4.0).floor() as usize).min(3);
            for (i, (head, sub, col)) in stages.iter().enumerate() {
                let bx = 56.0 + (i as f64) * 296.0;
                let is_on = i <= active;
                c.fill_rounded_rect(
                    bx * sx,
                    168.0 * sy,
                    276.0 * sx,
                    140.0 * sy,
                    10.0 * sx,
                    if is_on {
                        [col[0], col[1], col[2], 38]
                    } else {
                        [30, 41, 59, 180]
                    },
                );
                c.stroke_rounded_rect(
                    [bx * sx, 168.0 * sy, 276.0 * sx, 140.0 * sy],
                    10.0 * sx,
                    if i == active { 2.4 } else { 1.3 },
                    if is_on {
                        [col[0], col[1], col[2], 240]
                    } else {
                        [71, 85, 105, 200]
                    },
                );
                c.draw_text((bx + 14.0) * sx, 192.0 * sy, 1.65 * sy, head, *col, true);
                c.draw_text(
                    (bx + 14.0) * sx,
                    224.0 * sy,
                    1.3 * sy,
                    sub,
                    [226, 232, 240],
                    false,
                );
                if i < 3 {
                    c.draw_line_aa(
                        (bx + 276.0) * sx,
                        238.0 * sy,
                        (bx + 296.0) * sx,
                        238.0 * sy,
                        2.2,
                        [148, 163, 184, 220],
                    );
                }
            }

            // Live JSON-RPC + Undo Telemetry Log Box
            c.fill_rounded_rect(
                56.0 * sx,
                336.0 * sy,
                1168.0 * sx,
                248.0 * sy,
                10.0 * sx,
                [2, 6, 23, 225],
            );
            c.stroke_rounded_rect(
                [56.0 * sx, 336.0 * sy, 1168.0 * sx, 248.0 * sy],
                10.0 * sx,
                1.4,
                [51, 65, 85, 240],
            );
            let snip1: String = d.mcp_init_resp.chars().take(102).collect();
            let snip2: String = d.mcp_torsion_resp.chars().take(102).collect();
            let snip3: String = d.mcp_denied_resp.chars().take(102).collect();
            c.draw_text(
                74.0 * sx,
                356.0 * sy,
                1.45 * sy,
                &format!("init_resp    : {snip1}"),
                [125, 211, 252],
                false,
            );
            c.draw_text(
                74.0 * sx,
                384.0 * sy,
                1.45 * sy,
                &format!("torsion_resp : {snip2}"),
                [110, 231, 183],
                false,
            );
            c.draw_text(
                74.0 * sx,
                412.0 * sy,
                1.45 * sy,
                &format!("write_denied : {snip3}"),
                [253, 164, 175],
                false,
            );
            for (u_i, (desc, ulen, rlen, beq)) in d.undo_steps.iter().take(6).enumerate() {
                c.draw_text(
                    74.0 * sx,
                    (448.0 + u_i as f64 * 22.0) * sy,
                    1.35 * sy,
                    &format!("undo_ring[{u_i}] undo_len={ulen} redo_len={rlen} bitwise_eq={beq} :: {desc}"),
                    [196, 181, 253],
                    false,
                );
            }
            draw_animated_cursor(
                c,
                194.0 + (active as f64) * 296.0,
                272.0,
                local,
                "Live Protocol",
            );
        }
    }

    draw_tour_chrome(
        c,
        &TourChromeSpec {
            title: "08 · METABOLOMICS: AI-BIN/PQN · OPLS-DA · STOCSY · MCP + UNDO",
            badge_label: "SYNTHETIC & PROTOCOL SHOWCASE (F17/F18 OPEN)",
            badge_color: [167, 139, 250],
            ch_idx: ch,
            ch_title,
            ch_metric: &ch_metric,
            cli_cmd: "ferric metabolomics run --bucketing ai --norm pqn --oplsda --stocsy 1.33",
            t_sec,
        },
    );
}

// ============================================================================
// Self-Contained Anti-Aliased RGB24 Canvas2d Rasterizer + Scalable 5x7 Font
// ============================================================================

struct Canvas2d {
    width: usize,
    height: usize,
    rgb: Vec<u8>,
}

impl Canvas2d {
    fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            rgb: vec![0; width.saturating_mul(height).saturating_mul(3)],
        }
    }

    fn clear_cinema_backdrop(&mut self) {
        let w = self.width.max(1) as f64;
        let h = self.height.max(1) as f64;
        for y in 0..self.height {
            let v = y as f64 / h;
            for x in 0..self.width {
                let u = x as f64 / w;
                let glow = (1.0 - ((u - 0.5).powi(2) + (v - 0.35).powi(2)).sqrt() * 1.15).max(0.0);
                let r = (7.0 + 9.0 * v + 10.0 * glow).clamp(0.0, 255.0) as u8;
                let g = (11.0 + 13.0 * v + 14.0 * glow).clamp(0.0, 255.0) as u8;
                let b = (22.0 + 21.0 * v + 22.0 * glow).clamp(0.0, 255.0) as u8;
                let idx = (y * self.width + x) * 3;
                if idx + 2 < self.rgb.len() {
                    self.rgb[idx] = r;
                    self.rgb[idx + 1] = g;
                    self.rgb[idx + 2] = b;
                }
            }
        }
    }

    #[inline]
    fn blend_pixel(&mut self, x: isize, y: isize, rgba: [u8; 4], cov: f64) {
        if x < 0 || y < 0 || (x as usize) >= self.width || (y as usize) >= self.height {
            return;
        }
        let a = (rgba[3] as f64 / 255.0) * cov.clamp(0.0, 1.0);
        if a <= 1e-4 {
            return;
        }
        let idx = ((y as usize) * self.width + (x as usize)) * 3;
        if idx + 2 >= self.rgb.len() {
            return;
        }
        let inv = 1.0 - a;
        self.rgb[idx] = (self.rgb[idx] as f64 * inv + rgba[0] as f64 * a).round() as u8;
        self.rgb[idx + 1] = (self.rgb[idx + 1] as f64 * inv + rgba[1] as f64 * a).round() as u8;
        self.rgb[idx + 2] = (self.rgb[idx + 2] as f64 * inv + rgba[2] as f64 * a).round() as u8;
    }

    fn fill_rounded_rect(&mut self, x: f64, y: f64, w: f64, h: f64, r: f64, rgba: [u8; 4]) {
        if w <= 0.0 || h <= 0.0 {
            return;
        }
        let rad = r.min(0.5 * w).min(0.5 * h).max(0.0);
        let x0 = (x.floor() as isize - 1).max(0);
        let y0 = (y.floor() as isize - 1).max(0);
        let x1 = ((x + w).ceil() as isize + 1).min(self.width as isize - 1);
        let y1 = ((y + h).ceil() as isize + 1).min(self.height as isize - 1);
        let cx = x + 0.5 * w;
        let cy = y + 0.5 * h;
        let bx = 0.5 * w - rad;
        let by = 0.5 * h - rad;

        for py in y0..=y1 {
            let dy = (py as f64 + 0.5 - cy).abs() - by;
            for px in x0..=x1 {
                let dx = (px as f64 + 0.5 - cx).abs() - bx;
                let sd = (dx.max(0.0).hypot(dy.max(0.0))) + dx.max(dy).min(0.0) - rad;
                let cov = (0.5 - sd).clamp(0.0, 1.0);
                if cov > 0.0 {
                    self.blend_pixel(px, py, rgba, cov);
                }
            }
        }
    }

    fn stroke_rounded_rect(&mut self, rect: [f64; 4], r: f64, thickness: f64, rgba: [u8; 4]) {
        let [x, y, w, h] = rect;
        if w <= 0.0 || h <= 0.0 {
            return;
        }
        let rad = r.min(0.5 * w).min(0.5 * h).max(0.0);
        let half_t = 0.5 * thickness.max(0.5);
        let x0 = ((x - half_t).floor() as isize - 1).max(0);
        let y0 = ((y - half_t).floor() as isize - 1).max(0);
        let x1 = ((x + w + half_t).ceil() as isize + 1).min(self.width as isize - 1);
        let y1 = ((y + h + half_t).ceil() as isize + 1).min(self.height as isize - 1);
        let cx = x + 0.5 * w;
        let cy = y + 0.5 * h;
        let bx = 0.5 * w - rad;
        let by = 0.5 * h - rad;

        for py in y0..=y1 {
            let dy = (py as f64 + 0.5 - cy).abs() - by;
            for px in x0..=x1 {
                let dx = (px as f64 + 0.5 - cx).abs() - bx;
                let sd = (dx.max(0.0).hypot(dy.max(0.0))) + dx.max(dy).min(0.0) - rad;
                let cov = (half_t + 0.5 - sd.abs()).clamp(0.0, 1.0);
                if cov > 0.0 {
                    self.blend_pixel(px, py, rgba, cov);
                }
            }
        }
    }

    fn draw_line_aa(&mut self, x0: f64, y0: f64, x1: f64, y1: f64, thickness: f64, rgba: [u8; 4]) {
        let half_t = 0.5 * thickness.max(0.6);
        let pad = half_t + 1.5;
        let min_x = (x0.min(x1) - pad).floor() as isize;
        let max_x = (x0.max(x1) + pad).ceil() as isize;
        let min_y = (y0.min(y1) - pad).floor() as isize;
        let max_y = (y0.max(y1) + pad).ceil() as isize;

        let dx = x1 - x0;
        let dy = y1 - y0;
        let len2 = dx * dx + dy * dy;

        for py in min_y.max(0)..=max_y.min(self.height as isize - 1) {
            for px in min_x.max(0)..=max_x.min(self.width as isize - 1) {
                let cx = px as f64 + 0.5;
                let cy = py as f64 + 0.5;
                let dist = if len2 <= 1e-12 {
                    (cx - x0).hypot(cy - y0)
                } else {
                    let t = (((cx - x0) * dx + (cy - y0) * dy) / len2).clamp(0.0, 1.0);
                    (cx - (x0 + t * dx)).hypot(cy - (y0 + t * dy))
                };
                let cov = (half_t + 0.5 - dist).clamp(0.0, 1.0);
                if cov > 0.0 {
                    self.blend_pixel(px, py, rgba, cov);
                }
            }
        }
    }

    fn draw_dashed_line_aa(
        &mut self,
        p0: [f64; 2],
        p1: [f64; 2],
        thickness: f64,
        dash_len: f64,
        rgba: [u8; 4],
    ) {
        let [x0, y0] = p0;
        let [x1, y1] = p1;
        let total = (x1 - x0).hypot(y1 - y0);
        if total <= 1e-6 || dash_len <= 0.5 {
            self.draw_line_aa(x0, y0, x1, y1, thickness, rgba);
            return;
        }
        let step = dash_len * 1.75;
        let mut s = 0.0;
        while s < total {
            let t0 = s / total;
            let t1 = ((s + dash_len) / total).min(1.0);
            self.draw_line_aa(
                x0 + t0 * (x1 - x0),
                y0 + t0 * (y1 - y0),
                x0 + t1 * (x1 - x0),
                y0 + t1 * (y1 - y0),
                thickness,
                rgba,
            );
            s += step;
        }
    }

    fn draw_polyline_aa(&mut self, pts: &[(f64, f64)], thickness: f64, rgba: [u8; 4]) {
        for pair in pts.windows(2) {
            self.draw_line_aa(pair[0].0, pair[0].1, pair[1].0, pair[1].1, thickness, rgba);
        }
    }

    fn fill_circle_aa(&mut self, cx: f64, cy: f64, radius: f64, rgba: [u8; 4]) {
        let pad = radius + 1.5;
        let x0 = ((cx - pad).floor() as isize).max(0);
        let x1 = ((cx + pad).ceil() as isize).min(self.width as isize - 1);
        let y0 = ((cy - pad).floor() as isize).max(0);
        let y1 = ((cy + pad).ceil() as isize).min(self.height as isize - 1);
        for py in y0..=y1 {
            let dy = py as f64 + 0.5 - cy;
            for px in x0..=x1 {
                let dx = px as f64 + 0.5 - cx;
                let cov = (radius + 0.5 - dx.hypot(dy)).clamp(0.0, 1.0);
                if cov > 0.0 {
                    self.blend_pixel(px, py, rgba, cov);
                }
            }
        }
    }

    fn stroke_circle_aa(&mut self, cx: f64, cy: f64, radius: f64, thickness: f64, rgba: [u8; 4]) {
        let half_t = 0.5 * thickness.max(0.6);
        let pad = radius + half_t + 1.5;
        let x0 = ((cx - pad).floor() as isize).max(0);
        let x1 = ((cx + pad).ceil() as isize).min(self.width as isize - 1);
        let y0 = ((cy - pad).floor() as isize).max(0);
        let y1 = ((cy + pad).ceil() as isize).min(self.height as isize - 1);
        for py in y0..=y1 {
            let dy = py as f64 + 0.5 - cy;
            for px in x0..=x1 {
                let dx = px as f64 + 0.5 - cx;
                let dist = (dx.hypot(dy) - radius).abs();
                let cov = (half_t + 0.5 - dist).clamp(0.0, 1.0);
                if cov > 0.0 {
                    self.blend_pixel(px, py, rgba, cov);
                }
            }
        }
    }

    fn draw_text(&mut self, x: f64, y: f64, scale: f64, text: &str, rgb: [u8; 3], bold: bool) {
        let sc = scale.max(1.0);
        let advance = 6.0 * sc;
        let mut cursor_x = x;
        for ch in text.chars() {
            let glyph = glyph_5x7(ch);
            for (row, &bits) in glyph.iter().enumerate() {
                for col in 0..5 {
                    if (bits >> (4 - col)) & 1 == 1 {
                        let px = cursor_x + col as f64 * sc;
                        let py = y + row as f64 * sc;
                        self.fill_rect_fast(px, py, sc, sc, [rgb[0], rgb[1], rgb[2], 245]);
                        if bold {
                            self.fill_rect_fast(
                                px + 0.45 * sc,
                                py,
                                sc,
                                sc,
                                [rgb[0], rgb[1], rgb[2], 140],
                            );
                        }
                    }
                }
            }
            cursor_x += advance;
            if cursor_x + advance > self.width as f64 - 16.0 {
                break;
            }
        }
    }

    fn fill_rect_fast(&mut self, x: f64, y: f64, w: f64, h: f64, rgba: [u8; 4]) {
        let x0 = (x.floor() as isize).max(0);
        let y0 = (y.floor() as isize).max(0);
        let x1 = ((x + w).ceil() as isize).min(self.width as isize);
        let y1 = ((y + h).ceil() as isize).min(self.height as isize);
        for py in y0..y1 {
            for px in x0..x1 {
                self.blend_pixel(px, py, rgba, 1.0);
            }
        }
    }
}

const fn glyph_5x7(ch: char) -> [u8; 7] {
    let uc = match ch {
        'a'..='z' => ((ch as u8) - b'a' + b'A') as char,
        _ => ch,
    };
    match uc {
        'A' => [
            0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
        ],
        'B' => [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110,
        ],
        'C' => [
            0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110,
        ],
        'D' => [
            0b11100, 0b10010, 0b10001, 0b10001, 0b10001, 0b10010, 0b11100,
        ],
        'E' => [
            0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111,
        ],
        'F' => [
            0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000,
        ],
        'G' => [
            0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110,
        ],
        'H' => [
            0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
        ],
        'I' => [
            0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
        ],
        'J' => [
            0b00111, 0b00010, 0b00010, 0b00010, 0b10010, 0b10010, 0b01100,
        ],
        'K' => [
            0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001,
        ],
        'L' => [
            0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111,
        ],
        'M' => [
            0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001,
        ],
        'N' => [
            0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001,
        ],
        'O' => [
            0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
        ],
        'P' => [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000,
        ],
        'Q' => [
            0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101,
        ],
        'R' => [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001,
        ],
        'S' => [
            0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110,
        ],
        'T' => [
            0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100,
        ],
        'U' => [
            0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
        ],
        'V' => [
            0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100,
        ],
        'W' => [
            0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001,
        ],
        'X' => [
            0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001,
        ],
        'Y' => [
            0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100,
        ],
        'Z' => [
            0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111,
        ],
        '0' => [
            0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110,
        ],
        '1' => [
            0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
        ],
        '2' => [
            0b01110, 0b10001, 0b00001, 0b00110, 0b01000, 0b10000, 0b11111,
        ],
        '3' => [
            0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110,
        ],
        '4' => [
            0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010,
        ],
        '5' => [
            0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110,
        ],
        '6' => [
            0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110,
        ],
        '7' => [
            0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000,
        ],
        '8' => [
            0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110,
        ],
        '9' => [
            0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100,
        ],
        '.' => [
            0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b01100,
        ],
        ',' => [
            0b00000, 0b00000, 0b00000, 0b00000, 0b00100, 0b00100, 0b01000,
        ],
        ':' => [
            0b00000, 0b01100, 0b01100, 0b00000, 0b01100, 0b01100, 0b00000,
        ],
        ';' => [
            0b00000, 0b01100, 0b01100, 0b00000, 0b01100, 0b00100, 0b01000,
        ],
        '-' => [
            0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000,
        ],
        '+' => [
            0b00000, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0b00000,
        ],
        '=' => [
            0b00000, 0b00000, 0b11111, 0b00000, 0b11111, 0b00000, 0b00000,
        ],
        '/' => [
            0b00001, 0b00010, 0b00010, 0b00100, 0b01000, 0b01000, 0b10000,
        ],
        '(' => [
            0b00010, 0b00100, 0b01000, 0b01000, 0b01000, 0b00100, 0b00010,
        ],
        ')' => [
            0b01000, 0b00100, 0b00010, 0b00010, 0b00010, 0b00100, 0b01000,
        ],
        '[' => [
            0b01110, 0b01000, 0b01000, 0b01000, 0b01000, 0b01000, 0b01110,
        ],
        ']' => [
            0b01110, 0b00010, 0b00010, 0b00010, 0b00010, 0b00010, 0b01110,
        ],
        '{' => [
            0b00010, 0b00100, 0b00100, 0b01000, 0b00100, 0b00100, 0b00010,
        ],
        '}' => [
            0b01000, 0b00100, 0b00100, 0b00010, 0b00100, 0b00100, 0b01000,
        ],
        '%' => [
            0b11001, 0b11010, 0b00010, 0b00100, 0b01000, 0b01011, 0b10011,
        ],
        '&' => [
            0b01100, 0b10010, 0b10100, 0b01000, 0b10101, 0b10010, 0b01101,
        ],
        '*' => [
            0b00000, 0b10101, 0b01110, 0b11111, 0b01110, 0b10101, 0b00000,
        ],
        '^' => [
            0b00100, 0b01010, 0b10001, 0b00000, 0b00000, 0b00000, 0b00000,
        ],
        '_' => [
            0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b11111,
        ],
        '<' => [
            0b00010, 0b00100, 0b01000, 0b10000, 0b01000, 0b00100, 0b00010,
        ],
        '>' => [
            0b01000, 0b00100, 0b00010, 0b00001, 0b00010, 0b00100, 0b01000,
        ],
        '|' => [
            0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100,
        ],
        '#' => [
            0b01010, 0b11111, 0b01010, 0b01010, 0b11111, 0b01010, 0b00000,
        ],
        '$' => [
            0b00100, 0b01111, 0b10100, 0b01110, 0b00101, 0b11110, 0b00100,
        ],
        '·' => [
            0b00000, 0b00000, 0b00000, 0b00100, 0b00000, 0b00000, 0b00000,
        ],
        '±' => [
            0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0b00000, 0b11111,
        ],
        _ => [
            0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000,
        ],
    }
}

// ============================================================================
// Self-Contained Deterministic Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chem_tour_bundles_schema_and_determinism() {
        let bundles = build_all_chem_tour_bundles().expect("build 05..08 bundles");
        assert_eq!(bundles.len(), 4);
        for (offset, b) in bundles.iter().enumerate() {
            let expected_idx = (offset + 5) as u8;
            assert_eq!(b["schema_version"], TOUR_SCHEMA_VERSION);
            assert_eq!(b["module_index"], expected_idx);
            assert_eq!(b["slug"], chem_module_slug(expected_idx).expect("slug"));
            assert_eq!(b["duration_sec"], 16.0);
            assert_eq!(b["fps"], 30);

            let badge_lbl = b["validation_badge"]["label"]
                .as_str()
                .expect("badge label");
            assert!(
                badge_lbl.contains("SYNTHETIC"),
                "module {expected_idx} must carry explicit SYNTHETIC showcase badge: {badge_lbl}"
            );
            let chapters = b["chapters"].as_array().expect("chapters");
            assert_eq!(chapters.len(), 4);
            assert_eq!(chapters[0]["start_sec"], 0.0);
            assert_eq!(chapters[3]["end_sec"], 16.0);
            assert!(b["interactive_control"]["id"].as_str().is_some());
            assert!(b["payload"].is_object());
        }

        // Byte-for-byte determinism across repeated calls
        let b5_a = serde_json::to_string(&build_chem_tour_bundle(5).expect("b5 a")).expect("ser");
        let b5_b = serde_json::to_string(&build_chem_tour_bundle(5).expect("b5 b")).expect("ser");
        assert_eq!(b5_a, b5_b);
    }

    #[test]
    fn test_chem_tour_physics_values_real_nmr_core() {
        let d5 = get_mod05_data().expect("mod05");
        assert!((d5.cest_pb - 0.015).abs() < 0.006);
        assert!((d5.csp_kd_quad - 5.0).abs() < 1.5);
        assert!(d5.csp_kd_mm > d5.csp_kd_quad * 2.0);
        assert!(d5.cpmg_delta_aic > 50.0);
        assert_eq!(d5.hsqc_trajectories.len(), 5);

        let d6 = get_mod06_data().expect("mod06");
        assert!(!d6.deconv_cluster.peaks.is_empty());
        assert!(!d6.qnmr_report.t1_recycle_warning);
        assert!(
            (d6.qnmr_report.purity_percent - 98.4).abs() < 0.5,
            "qNMR purity_percent={} u_k2={}",
            d6.qnmr_report.purity_percent,
            d6.qnmr_report.expanded_uncertainty_k2_percent
        );
        assert!(d6.octet_compiled.primitives.len() > 10);
        assert!(d6.acs_report.contains("5.82"));

        let d7 = get_mod07_data().expect("mod07");
        assert_eq!(d7.dosy_peaks[3].multi_exp.selected_k, 2);
        assert!(!d7.ilt_overlap.modes.is_empty());
        assert!(d7.hydro_lyso.r_h_angstrom > d7.hydro_raffinose.r_h_angstrom);
        assert!(d7.hydro_raffinose.r_h_angstrom > d7.hydro_maleate.r_h_angstrom);

        let d8 = get_mod08_data().expect("mod08");
        assert_eq!(d8.study.sample_ids.len(), 24);
        assert!(d8.study.opls_da.as_ref().expect("opls").q2 > 0.85);
        assert!(d8.mcp_init_resp.contains("jsonrpc"));
        assert!(d8.mcp_torsion_resp.contains("result"));
        assert!(
            d8.mcp_denied_resp.contains("read_only_mode_denied")
                || d8.mcp_denied_resp.contains("ReadOnly")
        );
        assert!(d8.undo_steps.last().expect("undo").3);
    }

    #[test]
    fn test_render_chem_tour_frames_1280x720_variance_and_determinism() {
        for m in 5..=8 {
            let mut frames = Vec::with_capacity(4);
            for &t in &[2.0_f64, 6.0, 10.0, 14.0] {
                let f = render_chem_tour_frame(m, t, 1280, 720);
                assert_eq!(f.len(), 1280 * 720 * 3);
                frames.push(f);
            }
            // Verify determinism on identical call
            let f_dup = render_chem_tour_frame(m, 6.0, 1280, 720);
            assert_eq!(frames[1], f_dup);

            // Verify every chapter pair has substantial visual pixel variance
            for i in 0..4 {
                for j in (i + 1)..4 {
                    let diff_bytes: usize = frames[i]
                        .iter()
                        .zip(frames[j].iter())
                        .filter(|(a, b)| a.abs_diff(**b) > 8)
                        .count();
                    assert!(
                        diff_bytes > 20_000,
                        "module {m} chapter {i} vs {j} rendered too similarly ({diff_bytes} diff bytes)"
                    );
                }
            }
        }
    }
}
