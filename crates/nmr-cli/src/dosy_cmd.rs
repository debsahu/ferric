//! CLI subcommands `nmrassign dosy fit`, `nmrassign dosy ilt`, and `nmrassign dosy mw` (`T14.4`):
//! pseudo-2D pulsed-field-gradient `DOSY` ingestion (Bruker `difflist`/`acqus` `STE` & `BPP-LED`,
//! Varian `procpar` `gzlvl1`, and tabular decay matrices), default reproducible 1–3 component
//! Stejskal–Tanner / Wu–Johnson Levenberg–Marquardt fitting on CPU (`f64`) and WebGPU (`f32`) with
//! `BIC`/`AIC` model selection and bootstrap 95% confidence intervals, opt-in `CONTIN`-style
//! second-derivative Tikhonov non-negative Inverse Laplace Transform (`ILT`) with discrepancy /
//! `GCV` / L-curve regularization and full $\alpha$-sweep diagnostics, and Stokes–Einstein
//! hydrodynamic radius ($R_h$) / empirical power-law molecular-weight (`MW`) estimation.
//!
//! # Literature & Clean-Room Provenance
//! - Stejskal, E. O. & Tanner, J. E. (*J. Chem. Phys.* **1965**, 42, 288–292).
//! - Wu, D. H., Chen, A. D. & Johnson, C. S. Jr. (*J. Magn. Reson. A* **1995**, 115, 260–264).
//! - Pelta, M. D., Morris, G. A., Stchedroff, M. J. & Hammond, S. J. (*Magn. Reson. Chem.* **2002**, 40, S147–S152).
//! - Lawson, C. L. & Hanson, R. J. (*Solving Least Squares Problems*, Prentice-Hall, **1974**).
//! - Provencher, S. W. (*Comput. Phys. Commun.* **1982**, 27, 213–227 & 229–242).
//! - Golub, G. H., Heath, M. & Wahba, G. (*Technometrics* **1979**, 21, 215–223).
//! - Hansen, P. C. (*SIAM Rev.* **1992**, 34, 561–580).
//! - Evans, R., Deng, Z., Rogerson, A. K., McLachlan, A. S., Richards, J. J., Nilsson, M. &
//!   Morris, G. A. (*Angew. Chem. Int. Ed.* **2013**, 52, 3199–3202).
//!
//! Implemented strictly from the peer-reviewed equations and `docs/specs/17-nmr1d-and-dosy.md` §4
//! without consulting external proprietary or copyleft diffusion software.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use nmr_core::dosy::{
    compute_b_values_s_m2, estimate_hydrodynamics, fit_stejskal_tanner_k,
    fit_stejskal_tanner_multi, parse_bruker_difflist, read_bruker_dosy_dir, read_varian_dosy_dir,
    solve_tikhonov_ilt, stokes_einstein_radius_m, BootstrapMethod, BrukerDosyOverrides,
    DosyFitConfig, DosyNucleus, DosySequenceType, DosySolvent, GradientShape, HydrodynamicEstimate,
    IltConfig, IltLambdaStrategy, IltModePeak, IltSweepPoint, ModelSelectionCriterion,
    PowerLawCalibration, SoluteCompactnessClass, TikhonovRegularizer, VarianDosyOverrides,
    ILT_ARTIFACT_CAVEAT,
};
use nmr_core::rng::Rng;
use nmr_gpu::dosy_gpu::{fit_dosy_columns_gpu, DosyGpuColumnFit};
use nmr_gpu::GpuEngine;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DosyEngineChoice {
    Cpu,
    Gpu,
    Auto,
}

impl DosyEngineChoice {
    fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "cpu" => Ok(Self::Cpu),
            "gpu" | "wgpu" | "webgpu" => Ok(Self::Gpu),
            "auto" => Ok(Self::Auto),
            other => Err(format!(
                "invalid --engine {other:?} (expected cpu, gpu, or auto)"
            )),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DosyInputPeakRecord {
    pub label: String,
    pub center_ppm: f64,
    #[serde(default = "default_fwhm_ppm")]
    pub fwhm_ppm: f64,
    pub intensities: Vec<f64>,
}

fn default_fwhm_ppm() -> f64 {
    0.04
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DosyInputDatasetJson {
    #[serde(default)]
    pub b_values_s_m2: Vec<f64>,
    #[serde(default)]
    pub gradients_g_per_cm: Vec<f64>,
    #[serde(default)]
    pub nucleus: Option<String>,
    #[serde(default)]
    pub sequence_type: Option<String>,
    #[serde(default)]
    pub gradient_shape: Option<String>,
    #[serde(default)]
    pub big_delta_s: Option<f64>,
    #[serde(default)]
    pub little_delta_s: Option<f64>,
    #[serde(default)]
    pub tau_s: Option<f64>,
    pub peaks: Vec<DosyInputPeakRecord>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DosyComponentSummaryJson {
    pub component_index: usize,
    pub amplitude: f64,
    pub fraction: f64,
    pub d_m2_s: f64,
    pub log10_d: f64,
    pub ci95_d_m2_s: (f64, f64),
    pub sd_log10_d: f64,
    pub r_h_angstrom: f64,
    pub r_h_nm: f64,
    pub estimated_mw_da: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DosyPeakFitReportJson {
    pub label: String,
    pub center_ppm: f64,
    pub selected_k: usize,
    pub criterion: String,
    pub rss: f64,
    pub rmse: f64,
    pub aic: f64,
    pub bic: f64,
    pub is_well_resolved: bool,
    pub components: Vec<DosyComponentSummaryJson>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_column_fit: Option<DosyGpuColumnSummaryJson>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DosyGpuColumnSummaryJson {
    pub selected_k: usize,
    pub mono_d_m2_s: f64,
    pub bi_d_1_m2_s: f64,
    pub bi_d_2_m2_s: f64,
    pub mono_bic: f64,
    pub bi_bic: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DosyFitCliReport {
    pub engine_used: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_adapter: Option<String>,
    pub solvent: String,
    pub temperature_k: f64,
    pub viscosity_pa_s: f64,
    pub compactness_class: String,
    pub b_values_s_m2: Vec<f64>,
    pub peaks: Vec<DosyPeakFitReportJson>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DosyIltModeSummaryJson {
    pub mode_index: usize,
    pub d_m2_s: f64,
    pub log10_d: f64,
    pub area_fraction: f64,
    pub sd_log10_d: f64,
    pub r_h_angstrom: f64,
    pub estimated_mw_da: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DosyIltPeakReportJson {
    pub label: String,
    pub center_ppm: f64,
    pub selected_alpha: f64,
    pub modes: Vec<DosyIltModeSummaryJson>,
    pub d_grid_m2_s: Vec<f64>,
    pub log10_d_grid: Vec<f64>,
    pub spectrum: Vec<f64>,
    pub alpha_sweep: Vec<IltSweepPoint>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DosyIltCliReport {
    pub caveat: String,
    pub regularizer: String,
    pub strategy: String,
    pub solvent: String,
    pub temperature_k: f64,
    pub b_values_s_m2: Vec<f64>,
    pub peaks: Vec<DosyIltPeakReportJson>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DosyMwCliReport {
    pub solvent: String,
    pub temperature_k: f64,
    pub viscosity_pa_s: f64,
    pub compactness_class: String,
    pub power_law_alpha: f64,
    pub power_law_beta: f64,
    pub calibrated_from_standards: bool,
    pub estimates: Vec<HydrodynamicEstimate>,
}

pub fn dosy_usage() -> &'static str {
    "nmrassign dosy — Pseudo-2D Diffusion-Ordered Spectroscopy (DOSY) solver & hydrodynamics (T14.4)\n\
     Usage:\n\
       nmrassign dosy fit (--in <decay.tsv|json> | --demo) [--exp-dir <dir>] [--engine cpu|gpu|auto] [-o <out.tsv|json>]\n\
       nmrassign dosy ilt (--in <decay.tsv|json> | --demo) [--strategy gcv|lcurve|discrepancy|fixed] [-o <out.tsv|json>]\n\
       nmrassign dosy mw  --d <m2_s> [--d <m2_s> ...] [--solvent D2O] [--temp-k 298.15] [-o <out.tsv|json>]\n\n\
     Subcommands:\n\
       fit   Discrete 1-3 component Stejskal-Tanner / Wu-Johnson CPU/WebGPU LM + BIC/AIC + bootstrap CIs [default]\n\
       ilt   Opt-in CONTIN-style second-derivative Tikhonov non-negative ILT + discrepancy/GCV/L-curve alpha-sweep\n\
       mw    Stokes-Einstein hydrodynamic radius (Rh) & empirical power-law molecular-weight (MW) estimator\n\n\
     Common Input & Physics Options (fit / ilt):\n\
       --in | -i <path>          Input decay matrix (.json or wide/long .tsv)\n\
       --demo                    Use built-in 3-peak synthetic BPP-LED diffusion mixture\n\
       --exp-dir <dir>           Bruker (difflist + acqus) or Varian (procpar gzlvl1) directory for b-factors\n\
       --difflist <file>         Explicit Bruker difflist file (G/cm)\n\
       --seq-type <ste|bppled>   Diffusion pulse sequence family [default: bppled]\n\
       --shape <rect|sine|smsq>  Gradient pulse envelope shape [default: sine]\n\
       --nucleus <1H|19F|31P|13C> Observed nucleus for gyromagnetic ratio gamma [default: 1H]\n\
       --big-delta-s <sec>       Diffusion delay Delta in seconds [default: 0.100]\n\
       --little-delta-s <sec>    Total diffusion-encoding gradient width delta in seconds [default: 0.0024]\n\
       --tau-s <sec>             Bipolar inter-pulse recovery delay tau in seconds [default: 0.0002]\n\
       --solvent <name>          D2O | H2O | CDCl3 | DMSO-d6 | CD3OD | CD3CN [default: D2O]\n\
       --temp-k <K>              Sample temperature in Kelvin [default: 298.15]\n\
       --class <kind>            small | protein | polymer [default: small]\n\n\
     Discrete Fit Options (dosy fit):\n\
       --engine <cpu|gpu|auto>   Compute backend (WebGPU parallel K=1..2 WGSL vs CPU f64) [default: cpu]\n\
       --criterion <bic|aic>     Model-order selection criterion [default: bic]\n\
       --max-components <1|2|3>  Maximum exponential components K_max [default: 3]\n\
       --bootstrap <N>           Bootstrap replicates for 95% CIs (0 disables) [default: 200]\n\
       --bootstrap-method <m>    residual | parametric [default: residual]\n\
       --seed <u64>              Deterministic RNG seed [default: 20261404]\n\n\
     Continuous ILT Options (dosy ilt):\n\
       --strategy <name>         gcv | lcurve | discrepancy | fixed [default: gcv]\n\
       --noise-sigma <sigma>     Baseline noise sigma (required for --strategy discrepancy)\n\
       --alpha | --lambda <val>  Fixed regularization weight (required for --strategy fixed)\n\
       --regularizer <kind>      second-deriv (CONTIN-style) | identity [default: second-deriv]\n\
       --n-grid <N>              Log-spaced diffusivity grid points [default: 64]\n\
       --alpha-min <val>         Minimum alpha in sweep [default: 1e-5]\n\
       --alpha-max <val>         Maximum alpha in sweep [default: 10.0]\n\
       --alpha-steps <N>         Number of log-spaced alpha sweep points [default: 25]\n\
       --alpha-sweep-out <tsv>   Optional TSV path exporting the full regularization alpha-sweep\n\n\
     Hydrodynamics Options (dosy mw):\n\
       --d <m2_s>                Diffusion coefficient D in m^2/s (repeatable)\n\
       --standard <mw_da:d_m2_s> Internal calibration standard (repeatable, >= 2 fits log-log power law)\n\n\
     Outputs:\n\
       --out | -o <path>         Output TSV or JSON path (.json writes structured JSON; omit for stdout)\n\
       --json-out <path.json>    Additional structured JSON output path"
}

fn parse_seq_type(s: &str) -> Result<DosySequenceType, String> {
    match s.trim().to_ascii_lowercase().replace('-', "").as_str() {
        "ste" | "led" | "monopolar" => Ok(DosySequenceType::Ste),
        "bppled" | "bppste" | "dstebpgp" | "convcomp" | "bipolar" => Ok(DosySequenceType::BppLed),
        other => Err(format!(
            "invalid --seq-type {other:?} (expected ste or bppled)"
        )),
    }
}

fn parse_grad_shape(s: &str) -> Result<GradientShape, String> {
    let low = s.trim().to_ascii_lowercase();
    match low.as_str() {
        "rect" | "rectangular" | "square" => Ok(GradientShape::Rectangular),
        "sine" | "sinebell" | "sine.100" | "sin" => Ok(GradientShape::SineBell),
        "smsq" | "smsq10.100" | "trapezoid" => {
            Ok(GradientShape::SmoothedSquare { area_factor: 0.90 })
        }
        _ => {
            if let Ok(f) = low.parse::<f64>() {
                Ok(GradientShape::CustomFactor(f))
            } else {
                Err(format!(
                    "invalid --shape {s:?} (expected rect, sine, smsq, or numeric factor)"
                ))
            }
        }
    }
}

fn parse_compactness(s: &str) -> Result<SoluteCompactnessClass, String> {
    match s
        .trim()
        .to_ascii_lowercase()
        .replace(['-', '_'], "")
        .as_str()
    {
        "small" | "smallmolecule" | "organic" => Ok(SoluteCompactnessClass::SmallMolecule),
        "protein" | "globular" | "globularprotein" => Ok(SoluteCompactnessClass::GlobularProtein),
        "polymer" | "idp" | "expanded" | "expandedpolymer" => {
            Ok(SoluteCompactnessClass::ExpandedPolymer)
        }
        other => Err(format!(
            "invalid --class {other:?} (expected small, protein, or polymer)"
        )),
    }
}

fn parse_ppm_from_header(token: &str, fallback_idx: usize) -> f64 {
    let trimmed = token.trim();
    if let Ok(v) = trimmed.parse::<f64>() {
        return v;
    }
    let cleaned = trimmed
        .trim_end_matches("ppm")
        .trim_end_matches('_')
        .rsplit(['_', '@', ':'])
        .next()
        .unwrap_or(trimmed);
    cleaned
        .parse::<f64>()
        .unwrap_or_else(|_| (fallback_idx + 1) as f64)
}

#[derive(Clone, Debug)]
struct DosyCliPhysArgs {
    exp_dir: Option<PathBuf>,
    difflist: Option<PathBuf>,
    nucleus: DosyNucleus,
    seq_type: Option<DosySequenceType>,
    shape: Option<GradientShape>,
    big_delta_s: Option<f64>,
    little_delta_s: Option<f64>,
    tau_s: Option<f64>,
    dac_to_gcm: Option<f64>,
}

impl Default for DosyCliPhysArgs {
    fn default() -> Self {
        Self {
            exp_dir: None,
            difflist: None,
            nucleus: DosyNucleus::H1,
            seq_type: None,
            shape: None,
            big_delta_s: None,
            little_delta_s: None,
            tau_s: None,
            dac_to_gcm: None,
        }
    }
}

impl DosyCliPhysArgs {
    fn resolve_b_from_dir_or_grads(
        &self,
        explicit_grads_gcm: Option<&[f64]>,
    ) -> Result<Vec<f64>, String> {
        if let Some(dir) = &self.exp_dir {
            if dir.join("procpar").is_file() {
                let ovr = VarianDosyOverrides {
                    nucleus: Some(self.nucleus),
                    sequence_type: self.seq_type,
                    gradient_shape: self.shape,
                    big_delta_s: self.big_delta_s,
                    little_delta_s: self.little_delta_s,
                    tau_s: self.tau_s,
                    dac_to_g_per_cm: self.dac_to_gcm,
                };
                return read_varian_dosy_dir(dir, &ovr)
                    .map(|p| p.b_values_s_m2)
                    .map_err(|e| e.to_string());
            }
            let ovr = BrukerDosyOverrides {
                nucleus: Some(self.nucleus),
                sequence_type: self.seq_type,
                gradient_shape: self.shape,
                p30_mode: None,
                big_delta_s: self.big_delta_s,
                little_delta_s: self.little_delta_s,
                tau_s: self.tau_s,
                gpnam_index: None,
            };
            return read_bruker_dosy_dir(dir, &ovr)
                .map(|p| p.b_values_s_m2)
                .map_err(|e| e.to_string());
        }

        let grads_owned: Vec<f64>;
        let grads: &[f64] = if let Some(g) = explicit_grads_gcm {
            g
        } else if let Some(dl_path) = &self.difflist {
            let txt = fs::read_to_string(dl_path)
                .map_err(|e| format!("failed to read difflist {}: {e}", dl_path.display()))?;
            grads_owned = parse_bruker_difflist(&txt).map_err(|e| e.to_string())?;
            &grads_owned
        } else {
            return Err(
                "missing Stejskal-Tanner b-values (provide b_s_m2 column, --exp-dir, or --difflist)"
                    .to_string(),
            );
        };

        let seq = self.seq_type.unwrap_or(DosySequenceType::BppLed);
        let shape = self.shape.unwrap_or(GradientShape::SineBell);
        let big_d = self.big_delta_s.unwrap_or(0.100);
        let lit_d = self.little_delta_s.unwrap_or(0.0024);
        let tau = self.tau_s.unwrap_or(match seq {
            DosySequenceType::Ste => 0.0,
            DosySequenceType::BppLed => 0.0002,
        });
        compute_b_values_s_m2(grads, self.nucleus, seq, shape, big_d, lit_d, tau)
            .map_err(|e| e.to_string())
    }
}

fn build_demo_dataset() -> (Vec<f64>, Vec<DosyInputPeakRecord>) {
    let grads: Vec<f64> = (0..24).map(|i| 2.0 + (i as f64) * (72.0 / 23.0)).collect();
    let b_vals = compute_b_values_s_m2(
        &grads,
        DosyNucleus::H1,
        DosySequenceType::BppLed,
        GradientShape::SineBell,
        0.120,
        0.0024,
        0.0002,
    )
    .expect("valid demo BPP-LED b-values");

    type DemoPeakDef<'a> = (&'a str, f64, f64, &'a [(f64, f64)]);
    let d_slow = 2.0e-10_f64;
    let d_fast = 8.5e-10_f64;
    let defs: &[DemoPeakDef<'_>] = &[
        ("Ar_Slow_7.45", 7.45, 0.045, &[(100.0, d_slow)]),
        (
            "Mix_Overlap_3.68",
            3.68,
            0.050,
            &[(50.0, d_slow), (50.0, d_fast)],
        ),
        ("Aliph_Fast_1.22", 1.22, 0.040, &[(100.0, d_fast)]),
    ];

    let mut peaks = Vec::with_capacity(defs.len());
    for (idx, &(label, ppm, fwhm, comps)) in defs.iter().enumerate() {
        let mut rng = Rng::new(9001 + idx as u64);
        let intensities: Vec<f64> = b_vals
            .iter()
            .map(|&b| {
                let clean: f64 = comps.iter().map(|&(a, d)| a * (-b * d).exp()).sum();
                clean + rng.normal(0.0, 0.15)
            })
            .collect();
        peaks.push(DosyInputPeakRecord {
            label: label.to_string(),
            center_ppm: ppm,
            fwhm_ppm: fwhm,
            intensities,
        });
    }
    (b_vals, peaks)
}

fn load_dosy_dataset(
    in_path: Option<&Path>,
    demo: bool,
    phys: &DosyCliPhysArgs,
) -> Result<(Vec<f64>, Vec<DosyInputPeakRecord>), String> {
    if demo {
        return Ok(build_demo_dataset());
    }
    let path = in_path
        .ok_or_else(|| "missing required --in <decay.tsv|json> (or pass --demo)".to_string())?;
    let raw = fs::read_to_string(path)
        .map_err(|e| format!("failed to read DOSY input {}: {e}", path.display()))?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(format!("DOSY input {} is empty", path.display()));
    }

    // 1. JSON format
    if trimmed.starts_with('{') {
        let ds: DosyInputDatasetJson = serde_json::from_str(trimmed)
            .map_err(|e| format!("invalid DOSY JSON {}: {e}", path.display()))?;
        if ds.peaks.is_empty() {
            return Err("DOSY JSON contains zero peaks".to_string());
        }
        let b_vals = if !ds.b_values_s_m2.is_empty() {
            ds.b_values_s_m2
        } else if !ds.gradients_g_per_cm.is_empty() {
            let mut local_phys = phys.clone();
            if let Some(ref nuc) = ds.nucleus {
                local_phys.nucleus = DosyNucleus::from_label(nuc)
                    .ok_or_else(|| format!("unknown nucleus {nuc:?} in JSON"))?;
            }
            if let Some(ref st) = ds.sequence_type {
                local_phys.seq_type = Some(parse_seq_type(st)?);
            }
            if let Some(ref sh) = ds.gradient_shape {
                local_phys.shape = Some(parse_grad_shape(sh)?);
            }
            if ds.big_delta_s.is_some() {
                local_phys.big_delta_s = ds.big_delta_s;
            }
            if ds.little_delta_s.is_some() {
                local_phys.little_delta_s = ds.little_delta_s;
            }
            if ds.tau_s.is_some() {
                local_phys.tau_s = ds.tau_s;
            }
            local_phys.resolve_b_from_dir_or_grads(Some(&ds.gradients_g_per_cm))?
        } else {
            phys.resolve_b_from_dir_or_grads(None)?
        };
        for pk in &ds.peaks {
            if pk.intensities.len() != b_vals.len() {
                return Err(format!(
                    "peak {:?} has {} intensities, expected {} matching b_values_s_m2",
                    pk.label,
                    pk.intensities.len(),
                    b_vals.len()
                ));
            }
        }
        return Ok((b_vals, ds.peaks));
    }

    // 2. TSV / whitespace-delimited formats
    let mut header_b_vals: Option<Vec<f64>> = None;
    let mut data_lines: Vec<&str> = Vec::new();
    for line in trimmed.lines() {
        let s = line.trim();
        if s.is_empty() {
            continue;
        }
        if let Some(rest) = s.strip_prefix('#') {
            let meta = rest.trim();
            if let Some(b_str) = meta
                .strip_prefix("b_values_s_m2:")
                .or_else(|| meta.strip_prefix("b_s_m2:"))
            {
                let parsed: Result<Vec<f64>, _> = b_str
                    .split(|c: char| c.is_whitespace() || c == ',')
                    .filter(|t| !t.is_empty())
                    .map(|t| t.parse::<f64>())
                    .collect();
                if let Ok(v) = parsed {
                    if !v.is_empty() {
                        header_b_vals = Some(v);
                    }
                }
            }
            continue;
        }
        data_lines.push(s);
    }
    if data_lines.len() < 2 {
        return Err(format!(
            "DOSY table {} requires a header row and >= 1 data row",
            path.display()
        ));
    }

    let sep = if data_lines[0].contains('\t') {
        '\t'
    } else if data_lines[0].contains(',') {
        ','
    } else {
        ' '
    };
    let split_row = |row: &str| -> Vec<String> {
        if sep == ' ' {
            row.split_whitespace().map(str::to_string).collect()
        } else {
            row.split(sep).map(|c| c.trim().to_string()).collect()
        }
    };

    let cols0 = split_row(data_lines[0]);
    let c0_low = cols0
        .first()
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();

    // Case A: Long format (`label`, `center_ppm`, `b_s_m2` | `grad_g_cm`, `intensity`)
    if cols0.len() == 4
        && (c0_low == "label" || c0_low == "peak")
        && (cols0[2].eq_ignore_ascii_case("b_s_m2")
            || cols0[2].eq_ignore_ascii_case("b")
            || cols0[2].eq_ignore_ascii_case("grad_g_cm"))
    {
        let is_grad = cols0[2].eq_ignore_ascii_case("grad_g_cm");
        let mut grouped: BTreeMap<String, (f64, Vec<(f64, f64)>)> = BTreeMap::new();
        let mut order: Vec<String> = Vec::new();
        for (ln, row_s) in data_lines[1..].iter().enumerate() {
            let toks = split_row(row_s);
            if toks.len() < 4 {
                return Err(format!("short row at data line {}", ln + 2));
            }
            let lbl = toks[0].clone();
            let ppm: f64 = toks[1]
                .parse()
                .map_err(|_| format!("invalid center_ppm {:?} at line {}", toks[1], ln + 2))?;
            let x: f64 = toks[2]
                .parse()
                .map_err(|_| format!("invalid b/grad {:?} at line {}", toks[2], ln + 2))?;
            let y: f64 = toks[3]
                .parse()
                .map_err(|_| format!("invalid intensity {:?} at line {}", toks[3], ln + 2))?;
            if !grouped.contains_key(&lbl) {
                order.push(lbl.clone());
            }
            let entry = grouped.entry(lbl).or_insert_with(|| (ppm, Vec::new()));
            entry.1.push((x, y));
        }
        let first_key = order.first().ok_or("no peaks in long DOSY TSV")?;
        let x_ref: Vec<f64> = grouped[first_key].1.iter().map(|&(x, _)| x).collect();
        let b_vals = if is_grad {
            phys.resolve_b_from_dir_or_grads(Some(&x_ref))?
        } else {
            x_ref
        };
        let mut peaks = Vec::with_capacity(order.len());
        for key in order {
            let (ppm, pairs) = grouped.remove(&key).unwrap();
            if pairs.len() != b_vals.len() {
                return Err(format!(
                    "peak {key:?} has {} steps, expected {}",
                    pairs.len(),
                    b_vals.len()
                ));
            }
            peaks.push(DosyInputPeakRecord {
                label: key,
                center_ppm: ppm,
                fwhm_ppm: 0.04,
                intensities: pairs.into_iter().map(|(_, y)| y).collect(),
            });
        }
        return Ok((b_vals, peaks));
    }

    // Case B: Wide column-per-peak format (`b_s_m2` or `grad_g_cm` in col 0, each subsequent column is a peak)
    if c0_low == "b_s_m2"
        || c0_low == "b"
        || c0_low == "b_value"
        || c0_low == "grad_g_cm"
        || c0_low == "g_gcm"
        || c0_low == "gradient_g_cm"
    {
        if cols0.len() < 2 {
            return Err(
                "wide DOSY TSV requires >= 1 peak column after b_s_m2/grad_g_cm".to_string(),
            );
        }
        let is_grad = c0_low.starts_with("grad") || c0_low == "g_gcm";
        let n_peaks = cols0.len() - 1;
        let mut x_col = Vec::with_capacity(data_lines.len() - 1);
        let mut peak_cols = vec![Vec::with_capacity(data_lines.len() - 1); n_peaks];
        for (ln, row_s) in data_lines[1..].iter().enumerate() {
            let toks = split_row(row_s);
            if toks.len() != cols0.len() {
                return Err(format!(
                    "column count mismatch at data row {} (got {}, expected {})",
                    ln + 2,
                    toks.len(),
                    cols0.len()
                ));
            }
            let x: f64 = toks[0]
                .parse()
                .map_err(|_| format!("invalid numeric {:?} at row {}", toks[0], ln + 2))?;
            x_col.push(x);
            for p in 0..n_peaks {
                let y: f64 = toks[p + 1].parse().map_err(|_| {
                    format!("invalid intensity {:?} at row {}", toks[p + 1], ln + 2)
                })?;
                peak_cols[p].push(y);
            }
        }
        let b_vals = if is_grad {
            phys.resolve_b_from_dir_or_grads(Some(&x_col))?
        } else {
            x_col
        };
        let peaks = cols0[1..]
            .iter()
            .enumerate()
            .zip(peak_cols)
            .map(|((idx, hdr), intensities)| DosyInputPeakRecord {
                label: hdr.clone(),
                center_ppm: parse_ppm_from_header(hdr, idx),
                fwhm_ppm: 0.04,
                intensities,
            })
            .collect();
        return Ok((b_vals, peaks));
    }

    // Case C: Row-per-peak format (`label\tcenter_ppm\tI_0\tI_1...` with `# b_values_s_m2:` or `--exp-dir` / `--difflist`)
    let b_vals = match header_b_vals {
        Some(v) => v,
        None => phys.resolve_b_from_dir_or_grads(None)?,
    };
    let mut peaks = Vec::new();
    for (ln, row_s) in data_lines[1..].iter().enumerate() {
        let toks = split_row(row_s);
        if toks.len() != b_vals.len() + 2 {
            return Err(format!(
                "row-per-peak DOSY TSV line {} has {} columns, expected {} (label, center_ppm, {} steps)",
                ln + 2,
                toks.len(),
                b_vals.len() + 2,
                b_vals.len()
            ));
        }
        let ppm: f64 = toks[1]
            .parse()
            .map_err(|_| format!("invalid center_ppm {:?} at line {}", toks[1], ln + 2))?;
        let mut ints = Vec::with_capacity(b_vals.len());
        for tok in &toks[2..] {
            let v: f64 = tok
                .parse()
                .map_err(|_| format!("invalid intensity {:?} at line {}", tok, ln + 2))?;
            ints.push(v);
        }
        peaks.push(DosyInputPeakRecord {
            label: toks[0].clone(),
            center_ppm: ppm,
            fwhm_ppm: 0.04,
            intensities: ints,
        });
    }
    Ok((b_vals, peaks))
}

fn emit_output(
    out_path: Option<&Path>,
    json_out: Option<&Path>,
    tsv_text: &str,
    json_text: &str,
) -> Result<(), String> {
    if let Some(jp) = json_out {
        if let Some(parent) = jp.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
        }
        fs::write(jp, json_text).map_err(|e| format!("failed to write {}: {e}", jp.display()))?;
    }
    if let Some(op) = out_path {
        if let Some(parent) = op.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
        }
        let is_json = op
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("json"))
            .unwrap_or(false);
        let body = if is_json { json_text } else { tsv_text };
        fs::write(op, body).map_err(|e| format!("failed to write {}: {e}", op.display()))?;
    } else if json_out.is_none() {
        print!("{tsv_text}");
    }
    Ok(())
}

fn run_dosy_fit(args: &[String]) -> Result<(), String> {
    let mut in_path: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut json_out: Option<PathBuf> = None;
    let mut demo = false;
    let mut phys = DosyCliPhysArgs::default();
    let mut engine_choice = DosyEngineChoice::Cpu;
    let mut fit_cfg = DosyFitConfig::default();
    let mut fixed_k: Option<usize> = None;
    let mut solvent = DosySolvent::D2O;
    let mut temp_k = 298.15_f64;
    let mut comp_class = SoluteCompactnessClass::SmallMolecule;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                println!("{}", dosy_usage());
                return Ok(());
            }
            "--demo" => demo = true,
            "--in" | "-i" => {
                i += 1;
                in_path = Some(PathBuf::from(args.get(i).ok_or("missing value for --in")?));
            }
            "--out" | "-o" => {
                i += 1;
                out_path = Some(PathBuf::from(args.get(i).ok_or("missing value for --out")?));
            }
            "--json-out" => {
                i += 1;
                json_out = Some(PathBuf::from(
                    args.get(i).ok_or("missing value for --json-out")?,
                ));
            }
            "--exp-dir" => {
                i += 1;
                phys.exp_dir = Some(PathBuf::from(
                    args.get(i).ok_or("missing value for --exp-dir")?,
                ));
            }
            "--difflist" => {
                i += 1;
                phys.difflist = Some(PathBuf::from(
                    args.get(i).ok_or("missing value for --difflist")?,
                ));
            }
            "--seq-type" => {
                i += 1;
                phys.seq_type = Some(parse_seq_type(
                    args.get(i).ok_or("missing value for --seq-type")?,
                )?);
            }
            "--shape" => {
                i += 1;
                phys.shape = Some(parse_grad_shape(
                    args.get(i).ok_or("missing value for --shape")?,
                )?);
            }
            "--nucleus" => {
                i += 1;
                let raw = args.get(i).ok_or("missing value for --nucleus")?;
                phys.nucleus = DosyNucleus::from_label(raw)
                    .ok_or_else(|| format!("unsupported --nucleus {raw:?}"))?;
            }
            "--big-delta-s" => {
                i += 1;
                phys.big_delta_s = Some(
                    args.get(i)
                        .ok_or("missing --big-delta-s")?
                        .parse()
                        .map_err(|_| "invalid --big-delta-s")?,
                );
            }
            "--little-delta-s" => {
                i += 1;
                phys.little_delta_s = Some(
                    args.get(i)
                        .ok_or("missing --little-delta-s")?
                        .parse()
                        .map_err(|_| "invalid --little-delta-s")?,
                );
            }
            "--tau-s" => {
                i += 1;
                phys.tau_s = Some(
                    args.get(i)
                        .ok_or("missing --tau-s")?
                        .parse()
                        .map_err(|_| "invalid --tau-s")?,
                );
            }
            "--dac-to-gcm" => {
                i += 1;
                phys.dac_to_gcm = Some(
                    args.get(i)
                        .ok_or("missing --dac-to-gcm")?
                        .parse()
                        .map_err(|_| "invalid --dac-to-gcm")?,
                );
            }
            "--engine" => {
                i += 1;
                engine_choice =
                    DosyEngineChoice::parse(args.get(i).ok_or("missing value for --engine")?)?;
            }
            "--criterion" => {
                i += 1;
                let raw = args.get(i).ok_or("missing value for --criterion")?;
                fit_cfg.criterion = match raw.to_ascii_lowercase().as_str() {
                    "bic" => ModelSelectionCriterion::Bic,
                    "aic" => ModelSelectionCriterion::Aic,
                    _ => return Err(format!("invalid --criterion {raw:?} (expected bic or aic)")),
                };
            }
            "--max-components" => {
                i += 1;
                let k: usize = args
                    .get(i)
                    .ok_or("missing --max-components")?
                    .parse()
                    .map_err(|_| "invalid --max-components")?;
                if !(1..=3).contains(&k) {
                    return Err("--max-components must be 1, 2, or 3".to_string());
                }
                fit_cfg.max_components = k;
            }
            "--fit" => {
                i += 1;
                let mode = args.get(i).ok_or("missing --fit")?.to_ascii_lowercase();
                match mode.as_str() {
                    "discrete-bic" => {
                        fit_cfg.criterion = ModelSelectionCriterion::Bic;
                        fixed_k = None;
                    }
                    "discrete-aic" => {
                        fit_cfg.criterion = ModelSelectionCriterion::Aic;
                        fixed_k = None;
                    }
                    "mono" | "1" => fixed_k = Some(1),
                    "bi" | "2" => fixed_k = Some(2),
                    "tri" | "3" => fixed_k = Some(3),
                    other => {
                        return Err(format!(
                            "unsupported --fit {other:?} in `dosy fit` (use `dosy ilt` for continuous ILT)"
                        ))
                    }
                }
            }
            "--bootstrap" => {
                i += 1;
                fit_cfg.bootstrap_samples = args
                    .get(i)
                    .ok_or("missing --bootstrap")?
                    .parse()
                    .map_err(|_| "invalid --bootstrap")?;
            }
            "--bootstrap-method" => {
                i += 1;
                let raw = args.get(i).ok_or("missing --bootstrap-method")?;
                fit_cfg.bootstrap_method = match raw.to_ascii_lowercase().as_str() {
                    "residual" => BootstrapMethod::Residual,
                    "parametric" => BootstrapMethod::Parametric,
                    _ => {
                        return Err(format!(
                            "invalid --bootstrap-method {raw:?} (expected residual or parametric)"
                        ))
                    }
                };
            }
            "--seed" => {
                i += 1;
                fit_cfg.rng_seed = args
                    .get(i)
                    .ok_or("missing --seed")?
                    .parse()
                    .map_err(|_| "invalid --seed")?;
            }
            "--d-min" => {
                i += 1;
                fit_cfg.d_min_m2_s = args
                    .get(i)
                    .ok_or("missing --d-min")?
                    .parse()
                    .map_err(|_| "invalid --d-min")?;
            }
            "--d-max" => {
                i += 1;
                fit_cfg.d_max_m2_s = args
                    .get(i)
                    .ok_or("missing --d-max")?
                    .parse()
                    .map_err(|_| "invalid --d-max")?;
            }
            "--solvent" => {
                i += 1;
                let raw = args.get(i).ok_or("missing --solvent")?;
                solvent = DosySolvent::from_name(raw).ok_or_else(|| {
                    format!(
                        "unknown --solvent {raw:?} (expected D2O, H2O, CDCl3, DMSO-d6, CD3OD, CD3CN)"
                    )
                })?;
            }
            "--temp-k" => {
                i += 1;
                temp_k = args
                    .get(i)
                    .ok_or("missing --temp-k")?
                    .parse()
                    .map_err(|_| "invalid --temp-k")?;
            }
            "--class" => {
                i += 1;
                comp_class = parse_compactness(args.get(i).ok_or("missing --class")?)?;
            }
            other => {
                return Err(format!(
                    "unknown `nmrassign dosy fit` option {other:?}\n\n{}",
                    dosy_usage()
                ))
            }
        }
        i += 1;
    }

    let (b_vals, peaks) = load_dosy_dataset(in_path.as_deref(), demo, &phys)?;
    let viscosity_pa_s = solvent.viscosity_pa_s(temp_k).map_err(|e| e.to_string())?;

    // Optional WebGPU parallel K=1..2 column solver
    let mut gpu_adapter: Option<String> = None;
    let mut gpu_fits: Option<Vec<DosyGpuColumnFit>> = None;
    match engine_choice {
        DosyEngineChoice::Cpu => {}
        DosyEngineChoice::Gpu => {
            let eng = GpuEngine::new_headless()?;
            gpu_adapter = Some(format!("{} ({})", eng.adapter_name, eng.backend_name));
            let cols: Vec<Vec<f64>> = peaks.iter().map(|p| p.intensities.clone()).collect();
            gpu_fits = Some(fit_dosy_columns_gpu(&eng, &b_vals, &cols, &fit_cfg, 0.0)?);
        }
        DosyEngineChoice::Auto => {
            if let Ok(eng) = GpuEngine::new_headless() {
                let cols: Vec<Vec<f64>> = peaks.iter().map(|p| p.intensities.clone()).collect();
                if let Ok(gf) = fit_dosy_columns_gpu(&eng, &b_vals, &cols, &fit_cfg, 0.0) {
                    gpu_adapter = Some(format!("{} ({})", eng.adapter_name, eng.backend_name));
                    gpu_fits = Some(gf);
                }
            }
        }
    }

    let engine_used = if gpu_fits.is_some() {
        "gpu".to_string()
    } else {
        "cpu".to_string()
    };

    let crit_str = match fit_cfg.criterion {
        ModelSelectionCriterion::Bic => "BIC",
        ModelSelectionCriterion::Aic => "AIC",
    };

    let mut peak_reports = Vec::with_capacity(peaks.len());
    let mut tsv = String::from(
        "label\tcenter_ppm\tselected_k\tcriterion\tcomponent\tamplitude\tfraction\td_m2_s\tlog10_d\tci95_low_m2_s\tci95_high_m2_s\tr_h_angstrom\testimated_mw_da\trmse\taic\tbic\tengine\n",
    );

    for (idx, pk) in peaks.iter().enumerate() {
        let mut local_cfg = fit_cfg.clone();
        local_cfg.rng_seed = fit_cfg.rng_seed.wrapping_add(idx as u64 * 7919);

        let gpu_col = gpu_fits.as_ref().and_then(|v| v.get(idx));
        let best_fit = if let Some(k_req) = fixed_k {
            fit_stejskal_tanner_k(
                &b_vals,
                &pk.intensities,
                k_req,
                &local_cfg,
                local_cfg.bootstrap_samples > 0,
            )
            .map_err(|e| e.to_string())?
        } else {
            fit_stejskal_tanner_multi(&b_vals, &pk.intensities, &local_cfg)
                .map_err(|e| e.to_string())?
                .best_fit
        };

        // When WebGPU ran and selected K in 1..=2 (and K_req != 3), adopt the GPU's primary D/I0
        // values while keeping bootstrap 95% CIs so `--engine gpu` reports genuine GPU solver output.
        let mut comps_json = Vec::with_capacity(best_fit.components.len());
        let selected_k = if let (Some(g), None) = (gpu_col, fixed_k) {
            if best_fit.k_components <= 2 && g.selected_k >= 1 {
                g.selected_k
            } else {
                best_fit.k_components
            }
        } else {
            best_fit.k_components
        };

        if let (Some(g), true) = (
            gpu_col,
            fixed_k.is_none()
                && best_fit.k_components <= 2
                && gpu_col.is_some_and(|c| c.selected_k == selected_k),
        ) {
            if selected_k == 1 {
                let d = g.mono_d_m2_s;
                let amp = g.mono_i0;
                let ci = best_fit
                    .components
                    .first()
                    .map(|c| c.ci95_d_m2_s)
                    .unwrap_or((d, d));
                let sd_ld = best_fit
                    .components
                    .first()
                    .map(|c| c.sd_log10_d)
                    .unwrap_or(0.0);
                let hydro = estimate_hydrodynamics(d, solvent, temp_k, comp_class)
                    .map_err(|e| e.to_string())?;
                comps_json.push(DosyComponentSummaryJson {
                    component_index: 1,
                    amplitude: amp,
                    fraction: 1.0,
                    d_m2_s: d,
                    log10_d: d.log10(),
                    ci95_d_m2_s: ci,
                    sd_log10_d: sd_ld,
                    r_h_angstrom: hydro.r_h_angstrom,
                    r_h_nm: hydro.r_h_nm,
                    estimated_mw_da: hydro.estimated_mw_da,
                });
            } else {
                let tot = (g.bi_i0_1 + g.bi_i0_2).max(1.0e-30);
                let pairs = [(g.bi_i0_1, g.bi_d_1_m2_s), (g.bi_i0_2, g.bi_d_2_m2_s)];
                for (ci_idx, &(amp, d)) in pairs.iter().enumerate() {
                    let ci = best_fit
                        .components
                        .get(ci_idx)
                        .map(|c| c.ci95_d_m2_s)
                        .unwrap_or((d, d));
                    let sd_ld = best_fit
                        .components
                        .get(ci_idx)
                        .map(|c| c.sd_log10_d)
                        .unwrap_or(0.0);
                    let hydro = estimate_hydrodynamics(d, solvent, temp_k, comp_class)
                        .map_err(|e| e.to_string())?;
                    comps_json.push(DosyComponentSummaryJson {
                        component_index: ci_idx + 1,
                        amplitude: amp,
                        fraction: amp / tot,
                        d_m2_s: d,
                        log10_d: d.log10(),
                        ci95_d_m2_s: ci,
                        sd_log10_d: sd_ld,
                        r_h_angstrom: hydro.r_h_angstrom,
                        r_h_nm: hydro.r_h_nm,
                        estimated_mw_da: hydro.estimated_mw_da,
                    });
                }
            }
        } else {
            for (ci_idx, c) in best_fit.components.iter().enumerate() {
                let hydro = estimate_hydrodynamics(c.d_m2_s, solvent, temp_k, comp_class)
                    .map_err(|e| e.to_string())?;
                comps_json.push(DosyComponentSummaryJson {
                    component_index: ci_idx + 1,
                    amplitude: c.amplitude,
                    fraction: c.fraction,
                    d_m2_s: c.d_m2_s,
                    log10_d: c.log10_d,
                    ci95_d_m2_s: c.ci95_d_m2_s,
                    sd_log10_d: c.sd_log10_d,
                    r_h_angstrom: hydro.r_h_angstrom,
                    r_h_nm: hydro.r_h_nm,
                    estimated_mw_da: hydro.estimated_mw_da,
                });
            }
        }

        for c in &comps_json {
            tsv.push_str(&format!(
                "{}\t{:.4}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6e}\t{:.5}\t{:.6e}\t{:.6e}\t{:.3}\t{:.1}\t{:.6}\t{:.3}\t{:.3}\t{}\n",
                pk.label,
                pk.center_ppm,
                selected_k,
                crit_str,
                c.component_index,
                c.amplitude,
                c.fraction,
                c.d_m2_s,
                c.log10_d,
                c.ci95_d_m2_s.0,
                c.ci95_d_m2_s.1,
                c.r_h_angstrom,
                c.estimated_mw_da,
                best_fit.rmse,
                best_fit.aic,
                best_fit.bic,
                engine_used,
            ));
        }

        let gpu_summary = gpu_col.map(|g| DosyGpuColumnSummaryJson {
            selected_k: g.selected_k,
            mono_d_m2_s: g.mono_d_m2_s,
            bi_d_1_m2_s: g.bi_d_1_m2_s,
            bi_d_2_m2_s: g.bi_d_2_m2_s,
            mono_bic: g.mono_bic,
            bi_bic: g.bi_bic,
        });

        peak_reports.push(DosyPeakFitReportJson {
            label: pk.label.clone(),
            center_ppm: pk.center_ppm,
            selected_k,
            criterion: crit_str.to_string(),
            rss: best_fit.rss,
            rmse: best_fit.rmse,
            aic: best_fit.aic,
            bic: best_fit.bic,
            is_well_resolved: best_fit.is_well_resolved,
            components: comps_json,
            gpu_column_fit: gpu_summary,
        });
    }

    let report = DosyFitCliReport {
        engine_used,
        gpu_adapter,
        solvent: solvent.label().to_string(),
        temperature_k: temp_k,
        viscosity_pa_s,
        compactness_class: comp_class.label().to_string(),
        b_values_s_m2: b_vals,
        peaks: peak_reports,
    };
    let json_text = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?;
    emit_output(out_path.as_deref(), json_out.as_deref(), &tsv, &json_text)
}

fn run_dosy_ilt(args: &[String]) -> Result<(), String> {
    let mut in_path: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut json_out: Option<PathBuf> = None;
    let mut sweep_out: Option<PathBuf> = None;
    let mut demo = false;
    let mut phys = DosyCliPhysArgs::default();
    let mut ilt_cfg = IltConfig::default();
    let mut strat_kind = "gcv".to_string();
    let mut noise_sigma: Option<f64> = None;
    let mut fixed_alpha: Option<f64> = None;
    let mut solvent = DosySolvent::D2O;
    let mut temp_k = 298.15_f64;
    let mut comp_class = SoluteCompactnessClass::SmallMolecule;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                println!("{}", dosy_usage());
                return Ok(());
            }
            "--demo" => demo = true,
            "--in" | "-i" => {
                i += 1;
                in_path = Some(PathBuf::from(args.get(i).ok_or("missing value for --in")?));
            }
            "--out" | "-o" => {
                i += 1;
                out_path = Some(PathBuf::from(args.get(i).ok_or("missing value for --out")?));
            }
            "--json-out" => {
                i += 1;
                json_out = Some(PathBuf::from(
                    args.get(i).ok_or("missing value for --json-out")?,
                ));
            }
            "--alpha-sweep-out" | "--lambda-sweep-out" => {
                i += 1;
                sweep_out = Some(PathBuf::from(
                    args.get(i).ok_or("missing value for --alpha-sweep-out")?,
                ));
            }
            "--exp-dir" => {
                i += 1;
                phys.exp_dir = Some(PathBuf::from(
                    args.get(i).ok_or("missing value for --exp-dir")?,
                ));
            }
            "--difflist" => {
                i += 1;
                phys.difflist = Some(PathBuf::from(
                    args.get(i).ok_or("missing value for --difflist")?,
                ));
            }
            "--seq-type" => {
                i += 1;
                phys.seq_type = Some(parse_seq_type(
                    args.get(i).ok_or("missing value for --seq-type")?,
                )?);
            }
            "--shape" => {
                i += 1;
                phys.shape = Some(parse_grad_shape(
                    args.get(i).ok_or("missing value for --shape")?,
                )?);
            }
            "--nucleus" => {
                i += 1;
                let raw = args.get(i).ok_or("missing value for --nucleus")?;
                phys.nucleus = DosyNucleus::from_label(raw)
                    .ok_or_else(|| format!("unsupported --nucleus {raw:?}"))?;
            }
            "--big-delta-s" => {
                i += 1;
                phys.big_delta_s = Some(
                    args.get(i)
                        .ok_or("missing --big-delta-s")?
                        .parse()
                        .map_err(|_| "invalid --big-delta-s")?,
                );
            }
            "--little-delta-s" => {
                i += 1;
                phys.little_delta_s = Some(
                    args.get(i)
                        .ok_or("missing --little-delta-s")?
                        .parse()
                        .map_err(|_| "invalid --little-delta-s")?,
                );
            }
            "--tau-s" => {
                i += 1;
                phys.tau_s = Some(
                    args.get(i)
                        .ok_or("missing --tau-s")?
                        .parse()
                        .map_err(|_| "invalid --tau-s")?,
                );
            }
            "--strategy" => {
                i += 1;
                strat_kind = args
                    .get(i)
                    .ok_or("missing --strategy")?
                    .trim()
                    .to_ascii_lowercase();
            }
            "--noise-sigma" => {
                i += 1;
                noise_sigma = Some(
                    args.get(i)
                        .ok_or("missing --noise-sigma")?
                        .parse()
                        .map_err(|_| "invalid --noise-sigma")?,
                );
            }
            "--alpha" | "--lambda" => {
                i += 1;
                fixed_alpha = Some(
                    args.get(i)
                        .ok_or("missing --alpha")?
                        .parse()
                        .map_err(|_| "invalid --alpha")?,
                );
            }
            "--regularizer" => {
                i += 1;
                let raw = args
                    .get(i)
                    .ok_or("missing --regularizer")?
                    .to_ascii_lowercase();
                ilt_cfg.regularizer = match raw.as_str() {
                    "second-deriv" | "second" | "contin" | "l2" => {
                        TikhonovRegularizer::SecondDerivative
                    }
                    "identity" | "zero" | "l0" => TikhonovRegularizer::Identity,
                    _ => {
                        return Err(format!(
                            "invalid --regularizer {raw:?} (expected second-deriv or identity)"
                        ))
                    }
                };
            }
            "--n-grid" => {
                i += 1;
                ilt_cfg.n_grid = args
                    .get(i)
                    .ok_or("missing --n-grid")?
                    .parse()
                    .map_err(|_| "invalid --n-grid")?;
            }
            "--d-min" => {
                i += 1;
                ilt_cfg.d_min_m2_s = args
                    .get(i)
                    .ok_or("missing --d-min")?
                    .parse()
                    .map_err(|_| "invalid --d-min")?;
            }
            "--d-max" => {
                i += 1;
                ilt_cfg.d_max_m2_s = args
                    .get(i)
                    .ok_or("missing --d-max")?
                    .parse()
                    .map_err(|_| "invalid --d-max")?;
            }
            "--alpha-min" | "--lambda-min" => {
                i += 1;
                ilt_cfg.lambda_min = args
                    .get(i)
                    .ok_or("missing --alpha-min")?
                    .parse()
                    .map_err(|_| "invalid --alpha-min")?;
            }
            "--alpha-max" | "--lambda-max" => {
                i += 1;
                ilt_cfg.lambda_max = args
                    .get(i)
                    .ok_or("missing --alpha-max")?
                    .parse()
                    .map_err(|_| "invalid --alpha-max")?;
            }
            "--alpha-steps" | "--n-sweep" => {
                i += 1;
                ilt_cfg.n_lambda_sweep = args
                    .get(i)
                    .ok_or("missing --alpha-steps")?
                    .parse()
                    .map_err(|_| "invalid --alpha-steps")?;
            }
            "--solvent" => {
                i += 1;
                let raw = args.get(i).ok_or("missing --solvent")?;
                solvent = DosySolvent::from_name(raw)
                    .ok_or_else(|| format!("unknown --solvent {raw:?}"))?;
            }
            "--temp-k" => {
                i += 1;
                temp_k = args
                    .get(i)
                    .ok_or("missing --temp-k")?
                    .parse()
                    .map_err(|_| "invalid --temp-k")?;
            }
            "--class" => {
                i += 1;
                comp_class = parse_compactness(args.get(i).ok_or("missing --class")?)?;
            }
            "--fit" => {
                i += 1;
                let _ = args.get(i).ok_or("missing --fit")?;
            }
            other => {
                return Err(format!(
                    "unknown `nmrassign dosy ilt` option {other:?}\n\n{}",
                    dosy_usage()
                ))
            }
        }
        i += 1;
    }

    ilt_cfg.lambda_strategy = match strat_kind.as_str() {
        "gcv" => IltLambdaStrategy::Gcv,
        "lcurve" | "l-curve" => IltLambdaStrategy::LCurve,
        "discrepancy" | "morozov" => {
            let sig = noise_sigma
                .ok_or("--strategy discrepancy requires --noise-sigma <sigma>".to_string())?;
            IltLambdaStrategy::DiscrepancyPrinciple { noise_sigma: sig }
        }
        "fixed" => {
            let a = fixed_alpha.ok_or("--strategy fixed requires --alpha <value>".to_string())?;
            IltLambdaStrategy::Fixed(a)
        }
        other => {
            return Err(format!(
                "invalid --strategy {other:?} (expected gcv, lcurve, discrepancy, or fixed)"
            ))
        }
    };

    let (b_vals, peaks) = load_dosy_dataset(in_path.as_deref(), demo, &phys)?;
    eprintln!("note: [dosy-ilt] {ILT_ARTIFACT_CAVEAT}");

    let mut tsv = format!(
        "# caveat: {ILT_ARTIFACT_CAVEAT}\nlabel\tcenter_ppm\tselected_alpha\tmode_index\td_m2_s\tlog10_d\tarea_fraction\tsd_log10_d\tr_h_angstrom\testimated_mw_da\n"
    );
    let mut sweep_tsv = String::from(
        "label\tcenter_ppm\talpha\tresidual_norm\troughness_norm\teffective_dof\tgcv_score\n",
    );
    let mut peak_reports = Vec::with_capacity(peaks.len());

    for pk in &peaks {
        let res =
            solve_tikhonov_ilt(&b_vals, &pk.intensities, &ilt_cfg).map_err(|e| e.to_string())?;
        for sp in &res.lambda_sweep {
            sweep_tsv.push_str(&format!(
                "{}\t{:.4}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.4}\t{:.6e}\n",
                pk.label,
                pk.center_ppm,
                sp.lambda,
                sp.residual_norm,
                sp.roughness_norm,
                sp.effective_dof,
                sp.gcv_score
            ));
        }

        let mut modes_json = Vec::with_capacity(res.modes.len());
        for (m_idx, m) in res.modes.iter().enumerate() {
            let IltModePeak {
                d_m2_s,
                log10_d,
                area_fraction,
                sd_log10_d,
            } = *m;
            let hydro = estimate_hydrodynamics(d_m2_s, solvent, temp_k, comp_class)
                .map_err(|e| e.to_string())?;
            tsv.push_str(&format!(
                "{}\t{:.4}\t{:.6e}\t{}\t{:.6e}\t{:.5}\t{:.5}\t{:.5}\t{:.3}\t{:.1}\n",
                pk.label,
                pk.center_ppm,
                res.selected_lambda,
                m_idx + 1,
                d_m2_s,
                log10_d,
                area_fraction,
                sd_log10_d,
                hydro.r_h_angstrom,
                hydro.estimated_mw_da,
            ));
            modes_json.push(DosyIltModeSummaryJson {
                mode_index: m_idx + 1,
                d_m2_s,
                log10_d,
                area_fraction,
                sd_log10_d,
                r_h_angstrom: hydro.r_h_angstrom,
                estimated_mw_da: hydro.estimated_mw_da,
            });
        }

        peak_reports.push(DosyIltPeakReportJson {
            label: pk.label.clone(),
            center_ppm: pk.center_ppm,
            selected_alpha: res.selected_lambda,
            modes: modes_json,
            d_grid_m2_s: res.d_grid_m2_s,
            log10_d_grid: res.log10_d_grid,
            spectrum: res.spectrum,
            alpha_sweep: res.lambda_sweep,
        });
    }

    if let Some(sp_path) = sweep_out.as_deref() {
        if let Some(parent) = sp_path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
        }
        fs::write(sp_path, &sweep_tsv)
            .map_err(|e| format!("failed to write {}: {e}", sp_path.display()))?;
    }

    let reg_str = match ilt_cfg.regularizer {
        TikhonovRegularizer::SecondDerivative => "SecondDerivative",
        TikhonovRegularizer::Identity => "Identity",
    };
    let report = DosyIltCliReport {
        caveat: ILT_ARTIFACT_CAVEAT.to_string(),
        regularizer: reg_str.to_string(),
        strategy: strat_kind,
        solvent: solvent.label().to_string(),
        temperature_k: temp_k,
        b_values_s_m2: b_vals,
        peaks: peak_reports,
    };
    let json_text = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?;
    emit_output(out_path.as_deref(), json_out.as_deref(), &tsv, &json_text)
}

fn run_dosy_mw(args: &[String]) -> Result<(), String> {
    let mut d_list: Vec<f64> = Vec::new();
    let mut standards: Vec<(f64, f64)> = Vec::new();
    let mut solvent = DosySolvent::D2O;
    let mut temp_k = 298.15_f64;
    let mut comp_class = SoluteCompactnessClass::SmallMolecule;
    let mut out_path: Option<PathBuf> = None;
    let mut json_out: Option<PathBuf> = None;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                println!("{}", dosy_usage());
                return Ok(());
            }
            "--d" => {
                i += 1;
                let raw = args.get(i).ok_or("missing value for --d")?;
                for tok in raw.split(',').map(str::trim).filter(|t| !t.is_empty()) {
                    let d: f64 = tok
                        .parse()
                        .map_err(|_| format!("invalid diffusion coefficient {tok:?}"))?;
                    d_list.push(d);
                }
            }
            "--standard" => {
                i += 1;
                let raw = args
                    .get(i)
                    .ok_or("missing value for --standard <mw_da:d_m2_s>")?;
                let (mw_s, d_s) = raw
                    .split_once(':')
                    .ok_or_else(|| format!("invalid --standard {raw:?} (expected mw_da:d_m2_s)"))?;
                let mw: f64 = mw_s
                    .parse()
                    .map_err(|_| format!("invalid MW {mw_s:?} in --standard"))?;
                let d: f64 = d_s
                    .parse()
                    .map_err(|_| format!("invalid D {d_s:?} in --standard"))?;
                standards.push((mw, d));
            }
            "--solvent" => {
                i += 1;
                let raw = args.get(i).ok_or("missing --solvent")?;
                solvent = DosySolvent::from_name(raw)
                    .ok_or_else(|| format!("unknown --solvent {raw:?}"))?;
            }
            "--temp-k" => {
                i += 1;
                temp_k = args
                    .get(i)
                    .ok_or("missing --temp-k")?
                    .parse()
                    .map_err(|_| "invalid --temp-k")?;
            }
            "--class" => {
                i += 1;
                comp_class = parse_compactness(args.get(i).ok_or("missing --class")?)?;
            }
            "--out" | "-o" => {
                i += 1;
                out_path = Some(PathBuf::from(args.get(i).ok_or("missing value for --out")?));
            }
            "--json-out" => {
                i += 1;
                json_out = Some(PathBuf::from(
                    args.get(i).ok_or("missing value for --json-out")?,
                ));
            }
            other => {
                return Err(format!(
                    "unknown `nmrassign dosy mw` option {other:?}\n\n{}",
                    dosy_usage()
                ))
            }
        }
        i += 1;
    }

    if d_list.is_empty() {
        return Err("`nmrassign dosy mw` requires at least one `--d <m2_s>` value".to_string());
    }
    let eta = solvent.viscosity_pa_s(temp_k).map_err(|e| e.to_string())?;
    let calibrated = !standards.is_empty();
    let cal = if calibrated {
        PowerLawCalibration::fit_from_standards(&standards).map_err(|e| e.to_string())?
    } else {
        PowerLawCalibration::for_solvent(solvent, temp_k, comp_class).map_err(|e| e.to_string())?
    };

    let mut tsv = String::from(
        "d_m2_s\tlog10_d\tsolvent\ttemp_k\tviscosity_pa_s\tr_h_m\tr_h_nm\tr_h_angstrom\testimated_mw_da\n",
    );
    let mut estimates = Vec::with_capacity(d_list.len());
    for &d in &d_list {
        let rh_m = stokes_einstein_radius_m(d, temp_k, eta).map_err(|e| e.to_string())?;
        let mw_da = cal.estimate_mw_da(d).map_err(|e| e.to_string())?;
        let est = HydrodynamicEstimate {
            d_m2_s: d,
            temp_k,
            viscosity_pa_s: eta,
            r_h_m: rh_m,
            r_h_angstrom: rh_m * 1.0e10,
            r_h_nm: rh_m * 1.0e9,
            estimated_mw_da: mw_da,
        };
        tsv.push_str(&format!(
            "{:.6e}\t{:.5}\t{}\t{:.2}\t{:.6e}\t{:.6e}\t{:.4}\t{:.3}\t{:.2}\n",
            est.d_m2_s,
            est.d_m2_s.log10(),
            solvent.label(),
            est.temp_k,
            est.viscosity_pa_s,
            est.r_h_m,
            est.r_h_nm,
            est.r_h_angstrom,
            est.estimated_mw_da
        ));
        estimates.push(est);
    }

    let report = DosyMwCliReport {
        solvent: solvent.label().to_string(),
        temperature_k: temp_k,
        viscosity_pa_s: eta,
        compactness_class: comp_class.label().to_string(),
        power_law_alpha: cal.alpha,
        power_law_beta: cal.beta,
        calibrated_from_standards: calibrated,
        estimates,
    };
    let json_text = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?;
    emit_output(out_path.as_deref(), json_out.as_deref(), &tsv, &json_text)
}

pub fn run_dosy_cli(args: &[String]) -> Result<(), String> {
    if args.is_empty() || matches!(args[0].as_str(), "-h" | "--help") {
        println!("{}", dosy_usage());
        return Ok(());
    }
    match args[0].as_str() {
        "fit" => run_dosy_fit(&args[1..]),
        "ilt" => run_dosy_ilt(&args[1..]),
        "mw" | "hydro" => run_dosy_mw(&args[1..]),
        first if first.starts_with('-') => {
            // Allow `nmrassign dosy --in ... [--fit contin-ilt]` shorthand per roadmap
            let wants_ilt = args.windows(2).any(|w| {
                w[0] == "--fit"
                    && matches!(
                        w[1].to_ascii_lowercase().as_str(),
                        "contin-ilt" | "ilt" | "contin"
                    )
            });
            if wants_ilt {
                run_dosy_ilt(args)
            } else {
                run_dosy_fit(args)
            }
        }
        other => Err(format!(
            "unknown `nmrassign dosy` subcommand {other:?} (expected fit, ilt, or mw)\n\n{}",
            dosy_usage()
        )),
    }
}
