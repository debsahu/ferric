//! CLI subcommands `ferric metabolomics bin`, `ferric metabolomics pca`,
//! `ferric metabolomics oplsda`, and `ferric metabolomics stocsy` (`T14.5`):
//! untargeted 1D NMR biofluid cohort ingestion, solvent/water/urea masking, interval FFT local
//! alignment, Adaptive Intelligent Binning (`AIB`) & uniform bucketing, Probabilistic Quotient
//! Normalization (`PQN`), Pareto/UV/Mean scaling, PCA Hotelling $T^2$ & $Q$-residual outlier
//! diagnostics, leak-free NIPALS `OPLS-DA` with stratified $k$-fold $Q^2$ & permutation
//! anti-overfitting validation, `VIP` / Wiklund `S-plot` / `BH-FDR` Volcano biomarker ranking,
//! and 1D/2D `STOCSY` structural correlation matched against the open-access `BMRB` library.
//!
//! # Literature & Clean-Room Provenance
//! Original clean-room implementation of `docs/specs/20-metabolomics-chemometrics.md` (`T14.5`)
//! derived strictly from peer-reviewed literature and `BMRB` reference data (free access,
//! © UConn Health, acknowledgement requested; zero proprietary or copyleft chemometrics
//! packages consulted):
//! - Savorani, F., Tomasi, G. & Engelsen, S. B. (*J. Magn. Reson.* **2010**, 202, 190–202).
//! - Holmes, E. et al. (*Anal. Biochem.* **1994**, 220, 284–296).
//! - De Meyer, T. et al. (*Anal. Chem.* **2008**, 80, 3783–3790).
//! - Dieterle, F., Ross, A., Schlotterbeck, G. & Senn, H. (*Anal. Chem.* **2006**, 78, 4281–4290).
//! - van den Berg, R. A. et al. (*BMC Genomics* **2006**, 7, 142).
//! - Hotelling, H. (*Ann. Math. Stat.* **1931**, 2, 360–378) & Jackson, J. E. & Mudholkar, G. S.
//!   (*Technometrics* **1979**, 21, 341–349).
//! - Trygg, J. & Wold, S. (*J. Chemom.* **2002**, 16, 119–128); Bylesjö, M. et al.
//!   (*J. Chemom.* **2006**, 20, 341–351); Westerhuis, J. A. et al. (*Metabolomics* **2008**, 4, 81–89).
//! - Wiklund, S. et al. (*Anal. Chem.* **2008**, 80, 115–122) & Benjamini, Y. & Hochberg, Y.
//!   (*J. R. Stat. Soc. B* **1995**, 57, 289–300).
//! - Cloarec, O. et al. (*Anal. Chem.* **2005**, 77, 1282–1289) & Ulrich, E. L. et al.
//!   (*Nucleic Acids Res.* **2008**, 36, D402–D408, `BMRB` archive: free access, © UConn Health).

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use nmr_core::metabolite_db::{
    npmrd_web_search_url, query_offline_metabolites, query_online_bmrb_and_npmrd, MetaboliteMatch,
    MetaboliteNucleus, OnlineQuerySources, QueryPeak, MAX_METABOLITE_TOL_PPM,
};
use nmr_core::metabolomics::{
    align_cohort_interval_fft, bucket_cohort_spectra, compute_stocsy, compute_stocsy_at_ppm,
    export_biomarker_ranking_csv, export_bucket_table_csv, fit_opls_da_with_options, fit_pca,
    fit_pls_da_with_transform, generate_demo_biofluid_cohort, normalize_bucket_matrix,
    preprocess_cohort_1d, rank_biomarkers, run_metabolomics_pipeline,
    transform_and_scale_bucket_matrix, AlignmentMode, AlignmentReference, BiofluidMaskPreset,
    BiomarkerCandidate, BucketingMethod, CohortSample1D, ConfidenceEllipse2D, ExclusionZonePpm,
    MetaboliteMatchScore, MetabolomicsPipelineConfig, MetabolomicsStudyResult, NormalizationMethod,
    OplsDaResult, PcaResult, ScalingMethod, SpectralBucket, VarianceTransform,
};
use nmr_core::nmr1d::{Experiment1DKind, Spectrum1D};
use nmr_core::octet_analyt::jcamp_spec1d::read_jcamp;
use nmr_core::rng::Rng;
use nmr_core::spectrum::AxisPpm;

/// Loaded dataset ready for chemometric modeling (`PCA`, `OPLS-DA`, `STOCSY`), sourced either
/// from full 1D cohort spectra (`Study`) or from an already-exported bucket table (`BucketTable`).
enum LoadedMetabolomicsDataset {
    Study(Box<MetabolomicsStudyResult>),
    BucketTable {
        sample_ids: Vec<String>,
        sample_groups: Vec<String>,
        buckets: Vec<SpectralBucket>,
        normalized_bucket_matrix: Vec<Vec<f64>>,
        exclusion_zones: Vec<ExclusionZonePpm>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BinningCliReportJson {
    pub n_samples: usize,
    pub n_buckets: usize,
    pub sample_ids: Vec<String>,
    pub sample_groups: Vec<String>,
    pub dilution_factors: Vec<f64>,
    pub normalization: String,
    pub scaling: String,
    pub bucketing: String,
    pub mask_zones_ppm: Vec<[f64; 2]>,
    pub buckets: Vec<SpectralBucket>,
    pub normalized_bucket_matrix: Vec<Vec<f64>>,
    pub scaled_bucket_matrix: Vec<Vec<f64>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PcaSampleRowJson {
    pub sample_id: String,
    pub group: String,
    pub scores: Vec<f64>,
    pub hotelling_t2: f64,
    pub q_residual: f64,
    pub is_t2_outlier_95: bool,
    pub is_q_outlier_95: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PcaCliReportJson {
    pub n_samples: usize,
    pub n_buckets: usize,
    pub n_components: usize,
    pub scaling: String,
    pub explained_variance_ratio: Vec<f64>,
    pub cumulative_variance_ratio: Vec<f64>,
    pub hotelling_t2_limit_95: f64,
    pub hotelling_t2_limit_99: f64,
    pub q_residual_limit_95: f64,
    pub t2_outliers_95: Vec<String>,
    pub q_outliers_95: Vec<String>,
    pub global_ellipse_95: ConfidenceEllipse2D,
    pub group_ellipses_95: Vec<ConfidenceEllipse2D>,
    pub samples: Vec<PcaSampleRowJson>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OplsDaSampleRowJson {
    pub sample_id: String,
    pub group: String,
    pub y_binary: f64,
    pub t_predictive: f64,
    pub t_orthogonal_1: f64,
    pub y_fitted: f64,
    pub y_cv_predicted: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OplsDaCliReportJson {
    pub control_group: String,
    pub case_group: String,
    pub n_active_samples: usize,
    pub n_buckets: usize,
    pub n_orthogonal: usize,
    pub cv_folds: usize,
    pub n_permutations: usize,
    pub r2_x_predictive: f64,
    pub r2_x_orthogonal: f64,
    pub r2_x_total: f64,
    pub r2_y: f64,
    pub q2: f64,
    pub permutation_p_value_q2: f64,
    pub permutation_p_value_r2_y: f64,
    pub permutation_mean_null_q2: f64,
    pub permutation_q2_intercept: f64,
    pub is_validated: bool,
    pub overfit_warning: Option<String>,
    pub samples: Vec<OplsDaSampleRowJson>,
    pub top_biomarkers: Vec<BiomarkerCandidate>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StocsyBucketHitJson {
    pub bucket_index: usize,
    pub center_ppm: f64,
    pub low_ppm: f64,
    pub high_ppm: f64,
    pub covariance: f64,
    pub correlation: f64,
    pub r2: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StocsyCliReportJson {
    pub requested_driver_ppm: f64,
    pub driver_bucket_index: usize,
    pub driver_bucket_ppm: f64,
    pub n_samples: usize,
    pub n_buckets: usize,
    pub min_r2_filter: f64,
    pub covarying_buckets: Vec<StocsyBucketHitJson>,
    pub metabolite_matches: Vec<MetaboliteMatchScore>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Stocsy2dMatrixJson {
    pub bucket_centers_ppm: Vec<f64>,
    pub correlation_matrix: Vec<Vec<f64>>,
    pub covariance_matrix: Vec<Vec<f64>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IdentifyCliReportJson {
    pub nucleus: String,
    pub tol_ppm: f64,
    pub query_peaks_ppm: Vec<f64>,
    pub online_requested: bool,
    pub online_error: Option<String>,
    pub matches: Vec<MetaboliteMatch>,
}

pub fn metabolomics_usage() -> &'static str {
    "\
ferric metabolomics — 1D NMR Untargeted Metabolomics, Chemometrics & STOCSY (T14.5)

USAGE:
  ferric metabolomics bin      [COHORT_INPUTS] [PREPROCESS_OPTIONS] [-o buckets.csv] [--study-out study.json]
  ferric metabolomics pca      [--in study.json|buckets.csv | COHORT_INPUTS] [--components 5] [-o pca.json]
  ferric metabolomics oplsda   [--in study.json|buckets.csv | COHORT_INPUTS] [--ortho 1] [--cv-folds 7] [--permutations 100] [-o oplsda.json]
  ferric metabolomics stocsy   [--in study.json|buckets.csv | COHORT_INPUTS] --driver-ppm <PPM> [-o stocsy.json]
  ferric metabolomics identify --peaks <PPM1,PPM2,...> [--nucleus 1h|13c] [--tol 0.03] [--online] [-o hits.json]

IDENTIFY OPTIONS (BMRB-BMSE Offline + Opt-In BMRB v2 Online; NP-MRD web: https://np-mrd.org/):
  --peaks <PPM1,PPM2,...>      Comma-separated chemical shifts in ppm (repeatable)
  --nucleus <1h|13c>           Target NMR nucleus: 1h (default) or 13c
  --tol <PPM>                  Shift matching tolerance in ppm in (0.0, 2.0] (default: 0.03 for 1H, 0.50 for 13C)
  --online                     Opt-in live HTTPS query to api.bmrb.io (BMRB v2 REST API)
  --top <K>                    Maximum number of ranked metabolite matches to output (default: 10)
  -o, --out <PATH>             Optional output path (.json, .csv, or .tsv)

COHORT INPUTS (for bin, or direct on-the-fly cohort execution in pca/oplsda/stocsy):
  --cohort <PATH.json>         JSON array of CohortSample1D records
  --manifest <PATH.tsv|csv>    Cohort table: sample_id<TAB/COMMA>group<TAB/COMMA>spectrum_path
  --sample <ID:GROUP=PATH>     Repeatable 1D sample (2-col ppm/intensity TSV/CSV, JCAMP-DX, or Spectrum1D JSON)
  --demo [--seed <U64>]        Built-in 24-sample synthetic biofluid cohort (12 Control vs 12 Case)
  --permute-cohort-labels      Deterministically shuffle sample group labels (null-model stress test)
  --obs-mhz <MHZ>              Fallback spectrometer frequency for 2-column ASCII spectra (default: 600.0)

PREPROCESS & BINNING OPTIONS:
  --min-ppm <PPM>              Lower chemical-shift bound (default: 0.50)
  --max-ppm <PPM>              Upper chemical-shift bound (default: 9.50)
  --grid-step <PPM>            Common resampling step in ppm (default: 0.001)
  --mask <PRESET>              Solvent/matrix mask: urine (default), serum, aqueous, none, custom
  --exclude <LOW:HIGH>         Repeatable custom exclusion window in ppm (e.g. 4.65:5.05)
  --align <MODE>               Local interval FFT peak alignment: auto (default) or none
  --interval-width <PPM>       Interval width for FFT alignment (default: 0.24)
  --max-shift <PPM>            Maximum allowed alignment shift in ppm (default: 0.05)
  --align-ref <REF>            Alignment target: median (default) or mean
  --method <METHOD>            Bucketing: aib (Adaptive Intelligent, default) or uniform
  --bin-width <PPM>            Uniform bucket width in ppm (default: 0.04)
  --resolution <R>             AIB valley-seeking resolution exponent in (0,1) (default: 0.50)
  --min-width <PPM>            AIB minimum bucket width in ppm (default: 0.020)
  --max-width <PPM>            AIB maximum bucket width in ppm (default: 0.100)
  --coalesce-hz <HZ>           AIB multiplet-coalescing envelope in Hz (default: 6.5)
  --min-snr <SNR>              Minimum cohort peak SNR to retain a bucket (default: 3.0)
  --norm <NORM>                Dilution normalization: pqn (default), tsa, tsp, none
  --tsp-ppm <PPM>              Internal standard target ppm when --norm tsp (default: 0.00)
  --tsp-half-window <PPM>      Internal standard integration half-window (default: 0.05)
  --transform <MODE>           Pre-scaling variance transform: none (default) or log10
  --scale <SCALE>              Variable scaling: pareto (default), uv, mean, none

CHEMOMETRICS & STOCSY OPTIONS:
  -i, --in <PATH>              Load exported study.json or buckets.csv from `metabolomics bin`
  --components <A>             Number of PCA components (default: 5)
  --ortho <A_O>                Number of OPLS-DA orthogonal components (default: 1)
  --max-ortho <MAX_A_O>        CV-selected OPLS-DA orthogonal components in 1..=MAX_A_O (default: 1)
  --cv-folds <K>               Stratified cross-validation folds (default: 7)
  --permutations <N>           Label permutation trials for OPLS-DA validation (default: 100)
  --seed <U64>                 Deterministic PRNG seed (default: 20261115)
  --control <GROUP>            Explicit control class label (y = 0)
  --case <GROUP>               Explicit case class label (y = 1)
  --top <K>                    Number of top biomarkers / STOCSY hits to report (default: 20)
  --biomarkers-out <PATH.csv>  Write full ranked biomarker table CSV (`VIP`, `S-plot`, `Welch t`, `BH-FDR`, `BMRB`)
  --driver-ppm <PPM>           STOCSY driver resonance chemical shift in ppm
  --min-r2 <R2>                Minimum bucket r^2 to list in STOCSY output (default: 0.25)
  --matrix-2d-out <PATH.json>  Export full P x P 2D STOCSY correlation & covariance matrix JSON
  -o, --out <PATH>             Output report path (.csv, .tsv, or .json; prints summary to stdout)
  --study-out <PATH.json>      Save full serialized MetabolomicsStudyResult JSON
"
}

fn next_arg<'a>(args: &'a [String], i: &mut usize, flag: &str) -> Result<&'a str, String> {
    if *i + 1 >= args.len() {
        return Err(format!("missing value after {flag}"));
    }
    *i += 1;
    Ok(args[*i].as_str())
}

fn parse_f64_flag(raw: &str, flag: &str) -> Result<f64, String> {
    let v: f64 = raw
        .trim()
        .parse()
        .map_err(|_| format!("invalid floating-point number for {flag}: {raw:?}"))?;
    if !v.is_finite() {
        return Err(format!("non-finite value for {flag}: {raw:?}"));
    }
    Ok(v)
}

fn parse_usize_flag(raw: &str, flag: &str) -> Result<usize, String> {
    raw.trim()
        .parse::<usize>()
        .map_err(|_| format!("invalid non-negative integer for {flag}: {raw:?}"))
}

fn parse_u64_flag(raw: &str, flag: &str) -> Result<u64, String> {
    raw.trim()
        .parse::<u64>()
        .map_err(|_| format!("invalid u64 seed for {flag}: {raw:?}"))
}

fn parse_scaling(raw: &str) -> Result<ScalingMethod, String> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "pareto" => Ok(ScalingMethod::Pareto),
        "uv" | "unit-variance" | "unit_variance" | "autoscale" => Ok(ScalingMethod::UnitVariance),
        "mean" | "center" | "mean-center" | "mean_center" => Ok(ScalingMethod::MeanCenter),
        "none" | "raw" => Ok(ScalingMethod::None),
        other => Err(format!(
            "invalid --scale {other:?} (expected pareto, uv, mean, or none)"
        )),
    }
}

fn parse_variance_transform(raw: &str) -> Result<VarianceTransform, String> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "none" | "raw" | "identity" => Ok(VarianceTransform::None),
        "log10" | "log10-pseudocount" | "log10_pseudocount" | "log" => {
            Ok(VarianceTransform::Log10Pseudocount)
        }
        other => Err(format!(
            "invalid --transform {other:?} (expected none or log10)"
        )),
    }
}

fn format_scaling(s: ScalingMethod) -> &'static str {
    match s {
        ScalingMethod::Pareto => "pareto",
        ScalingMethod::UnitVariance => "uv",
        ScalingMethod::MeanCenter => "mean",
        ScalingMethod::None => "none",
    }
}

fn format_normalization(n: NormalizationMethod) -> String {
    match n {
        NormalizationMethod::ProbabilisticQuotient => "pqn".to_string(),
        NormalizationMethod::TotalSpectralArea => "tsa".to_string(),
        NormalizationMethod::InternalStandard {
            target_ppm,
            half_window_ppm,
        } => format!("tsp({target_ppm:.3}+-{half_window_ppm:.3}ppm)"),
        NormalizationMethod::None => "none".to_string(),
    }
}

fn format_bucketing(b: BucketingMethod) -> String {
    match b {
        BucketingMethod::Uniform {
            bin_width_ppm,
            min_bucket_snr,
        } => format!("uniform(width={bin_width_ppm:.4}ppm,min_snr={min_bucket_snr:.1})"),
        BucketingMethod::AdaptiveIntelligent {
            resolution,
            min_width_ppm,
            max_width_ppm,
            multiplet_coalesce_hz,
            min_bucket_snr,
        } => format!(
            "aib(r={resolution:.2},w=[{min_width_ppm:.3},{max_width_ppm:.3}]ppm,coalesce={multiplet_coalesce_hz:.1}Hz,min_snr={min_bucket_snr:.1})"
        ),
    }
}

fn parse_zone_pair(raw: &str) -> Result<ExclusionZonePpm, String> {
    let parts: Vec<&str> = raw.split([':', ',']).collect();
    if parts.len() != 2 {
        return Err(format!(
            "invalid --exclude {raw:?} (expected LOW:HIGH in ppm, e.g. 4.65:5.05)"
        ));
    }
    let a = parse_f64_flag(parts[0], "--exclude low")?;
    let b = parse_f64_flag(parts[1], "--exclude high")?;
    Ok(ExclusionZonePpm::new(a, b))
}

#[derive(Clone, Debug)]
struct CommonMetabCliArgs {
    in_path: Option<PathBuf>,
    cohort_json: Option<PathBuf>,
    manifest_path: Option<PathBuf>,
    sample_specs: Vec<String>,
    demo: bool,
    permute_cohort_labels: bool,
    fallback_obs_mhz: f64,
    config: MetabolomicsPipelineConfig,
    transform_override: Option<VarianceTransform>,
    scale_override: Option<ScalingMethod>,
    pca_components_override: Option<usize>,
    ortho_override: Option<usize>,
    max_ortho_override: Option<usize>,
    cv_folds_override: Option<usize>,
    permutations_override: Option<usize>,
    seed_override: Option<u64>,
    control_override: Option<String>,
    case_override: Option<String>,
    top_k: usize,
    driver_ppm: Option<f64>,
    min_r2: f64,
    out_path: Option<PathBuf>,
    study_out_path: Option<PathBuf>,
    biomarkers_out_path: Option<PathBuf>,
    matrix_2d_out_path: Option<PathBuf>,
}

impl Default for CommonMetabCliArgs {
    fn default() -> Self {
        Self {
            in_path: None,
            cohort_json: None,
            manifest_path: None,
            sample_specs: Vec::new(),
            demo: false,
            permute_cohort_labels: false,
            fallback_obs_mhz: 600.0,
            config: MetabolomicsPipelineConfig::default(),
            transform_override: None,
            scale_override: None,
            pca_components_override: None,
            ortho_override: None,
            max_ortho_override: None,
            cv_folds_override: None,
            permutations_override: None,
            seed_override: None,
            control_override: None,
            case_override: None,
            top_k: 20,
            driver_ppm: None,
            min_r2: 0.25,
            out_path: None,
            study_out_path: None,
            biomarkers_out_path: None,
            matrix_2d_out_path: None,
        }
    }
}

fn parse_common_args(args: &[String]) -> Result<Option<CommonMetabCliArgs>, String> {
    let mut parsed = CommonMetabCliArgs::default();
    let mut custom_zones: Vec<ExclusionZonePpm> = Vec::new();
    let mut mask_name: Option<String> = None;
    let mut method_name = String::from("aib");
    let mut uniform_width = 0.04;
    let mut aib_res = 0.50;
    let mut aib_min_w = 0.020;
    let mut aib_max_w = 0.100;
    let mut aib_coalesce_hz = 6.5;
    let mut min_snr: Option<f64> = None;

    let mut align_enabled = true;
    let mut align_interval = 0.24;
    let mut align_max_shift = 0.05;
    let mut align_ref = AlignmentReference::Median;

    let mut norm_kind = String::from("pqn");
    let mut tsp_ppm = 0.0;
    let mut tsp_half_win = 0.05;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" | "help" => return Ok(None),
            "-i" | "--in" | "--input" => {
                let flag = args[i].as_str();
                parsed.in_path = Some(PathBuf::from(next_arg(args, &mut i, flag)?));
            }
            "--cohort" => {
                parsed.cohort_json = Some(PathBuf::from(next_arg(args, &mut i, "--cohort")?));
            }
            "--manifest" => {
                parsed.manifest_path = Some(PathBuf::from(next_arg(args, &mut i, "--manifest")?));
            }
            "--sample" => {
                parsed
                    .sample_specs
                    .push(next_arg(args, &mut i, "--sample")?.to_string());
            }
            "--demo" => {
                parsed.demo = true;
            }
            "--permute-cohort-labels" | "--null-labels" => {
                parsed.permute_cohort_labels = true;
            }
            "--obs-mhz" => {
                parsed.fallback_obs_mhz =
                    parse_f64_flag(next_arg(args, &mut i, "--obs-mhz")?, "--obs-mhz")?;
                if parsed.fallback_obs_mhz <= 0.0 {
                    return Err("--obs-mhz must be > 0".to_string());
                }
            }
            "--min-ppm" => {
                parsed.config.min_ppm =
                    parse_f64_flag(next_arg(args, &mut i, "--min-ppm")?, "--min-ppm")?;
            }
            "--max-ppm" => {
                parsed.config.max_ppm =
                    parse_f64_flag(next_arg(args, &mut i, "--max-ppm")?, "--max-ppm")?;
            }
            "--grid-step" => {
                parsed.config.grid_step_ppm =
                    parse_f64_flag(next_arg(args, &mut i, "--grid-step")?, "--grid-step")?;
                if parsed.config.grid_step_ppm <= 0.0 {
                    return Err("--grid-step must be > 0".to_string());
                }
            }
            "--calibrate-tsp" => {
                parsed.config.calibrate_internal_standard = true;
            }
            "--mask" => {
                mask_name = Some(next_arg(args, &mut i, "--mask")?.to_ascii_lowercase());
            }
            "--exclude" => {
                let z = parse_zone_pair(next_arg(args, &mut i, "--exclude")?)?;
                custom_zones.push(z);
            }
            "--align" => {
                let v = next_arg(args, &mut i, "--align")?.to_ascii_lowercase();
                match v.as_str() {
                    "auto" | "fft" | "interval" | "on" | "true" => align_enabled = true,
                    "none" | "off" | "false" | "disabled" => align_enabled = false,
                    other => {
                        return Err(format!("invalid --align {other:?} (expected auto or none)"));
                    }
                }
            }
            "--interval-width" => {
                align_interval = parse_f64_flag(
                    next_arg(args, &mut i, "--interval-width")?,
                    "--interval-width",
                )?;
            }
            "--max-shift" => {
                align_max_shift =
                    parse_f64_flag(next_arg(args, &mut i, "--max-shift")?, "--max-shift")?;
            }
            "--align-ref" => {
                let v = next_arg(args, &mut i, "--align-ref")?.to_ascii_lowercase();
                align_ref = match v.as_str() {
                    "median" => AlignmentReference::Median,
                    "mean" => AlignmentReference::Mean,
                    other => {
                        return Err(format!(
                            "invalid --align-ref {other:?} (expected median or mean)"
                        ));
                    }
                };
            }
            "--method" | "--binning" => {
                method_name = next_arg(args, &mut i, "--method")?.to_ascii_lowercase();
            }
            "--bin-width" => {
                uniform_width =
                    parse_f64_flag(next_arg(args, &mut i, "--bin-width")?, "--bin-width")?;
            }
            "--resolution" => {
                aib_res = parse_f64_flag(next_arg(args, &mut i, "--resolution")?, "--resolution")?;
            }
            "--min-width" => {
                aib_min_w = parse_f64_flag(next_arg(args, &mut i, "--min-width")?, "--min-width")?;
            }
            "--max-width" => {
                aib_max_w = parse_f64_flag(next_arg(args, &mut i, "--max-width")?, "--max-width")?;
            }
            "--coalesce-hz" => {
                aib_coalesce_hz =
                    parse_f64_flag(next_arg(args, &mut i, "--coalesce-hz")?, "--coalesce-hz")?;
            }
            "--min-snr" => {
                min_snr = Some(parse_f64_flag(
                    next_arg(args, &mut i, "--min-snr")?,
                    "--min-snr",
                )?);
            }
            "--norm" | "--normalization" => {
                norm_kind = next_arg(args, &mut i, "--norm")?.to_ascii_lowercase();
            }
            "--tsp-ppm" => {
                tsp_ppm = parse_f64_flag(next_arg(args, &mut i, "--tsp-ppm")?, "--tsp-ppm")?;
            }
            "--tsp-half-window" => {
                tsp_half_win = parse_f64_flag(
                    next_arg(args, &mut i, "--tsp-half-window")?,
                    "--tsp-half-window",
                )?;
            }
            "--transform" | "--variance-transform" => {
                let vt = parse_variance_transform(next_arg(args, &mut i, "--transform")?)?;
                parsed.config.variance_transform = vt;
                parsed.transform_override = Some(vt);
            }
            "--scale" | "--scaling" => {
                let sc = parse_scaling(next_arg(args, &mut i, "--scale")?)?;
                parsed.config.scaling = sc;
                parsed.scale_override = Some(sc);
            }
            "--components" | "--pca-components" => {
                let k = parse_usize_flag(next_arg(args, &mut i, "--components")?, "--components")?;
                if k == 0 {
                    return Err("--components must be >= 1".to_string());
                }
                parsed.config.pca_components = k;
                parsed.pca_components_override = Some(k);
            }
            "--ortho" | "--orthogonal" => {
                let k = parse_usize_flag(next_arg(args, &mut i, "--ortho")?, "--ortho")?;
                parsed.config.opls_orthogonal_components = k;
                parsed.ortho_override = Some(k);
            }
            "--max-ortho" | "--max-orthogonal" => {
                let k = parse_usize_flag(next_arg(args, &mut i, "--max-ortho")?, "--max-ortho")?;
                if k == 0 {
                    return Err("--max-ortho must be >= 1".to_string());
                }
                parsed.config.max_opls_orthogonal_components = Some(k);
                parsed.max_ortho_override = Some(k);
            }
            "--cv-folds" | "--folds" => {
                let k = parse_usize_flag(next_arg(args, &mut i, "--cv-folds")?, "--cv-folds")?;
                if k < 2 {
                    return Err("--cv-folds must be >= 2".to_string());
                }
                parsed.config.cv_folds = k;
                parsed.cv_folds_override = Some(k);
            }
            "--permutations" => {
                let n =
                    parse_usize_flag(next_arg(args, &mut i, "--permutations")?, "--permutations")?;
                parsed.config.n_permutations = n;
                parsed.permutations_override = Some(n);
            }
            "--seed" => {
                let s = parse_u64_flag(next_arg(args, &mut i, "--seed")?, "--seed")?;
                parsed.config.rng_seed = s;
                parsed.seed_override = Some(s);
            }
            "--control" => {
                let g = next_arg(args, &mut i, "--control")?.to_string();
                parsed.config.control_group = Some(g.clone());
                parsed.control_override = Some(g);
            }
            "--case" => {
                let g = next_arg(args, &mut i, "--case")?.to_string();
                parsed.config.case_group = Some(g.clone());
                parsed.case_override = Some(g);
            }
            "--top" => {
                parsed.top_k = parse_usize_flag(next_arg(args, &mut i, "--top")?, "--top")?.max(1);
            }
            "--driver-ppm" | "--driver" => {
                parsed.driver_ppm = Some(parse_f64_flag(
                    next_arg(args, &mut i, "--driver-ppm")?,
                    "--driver-ppm",
                )?);
            }
            "--min-r2" => {
                parsed.min_r2 = parse_f64_flag(next_arg(args, &mut i, "--min-r2")?, "--min-r2")?
                    .clamp(0.0, 1.0);
            }
            "-o" | "--out" | "--output" => {
                let flag = args[i].as_str();
                parsed.out_path = Some(PathBuf::from(next_arg(args, &mut i, flag)?));
            }
            "--study-out" | "--json-out" => {
                let flag = args[i].as_str();
                parsed.study_out_path = Some(PathBuf::from(next_arg(args, &mut i, flag)?));
            }
            "--biomarkers-out" => {
                parsed.biomarkers_out_path =
                    Some(PathBuf::from(next_arg(args, &mut i, "--biomarkers-out")?));
            }
            "--matrix-2d-out" => {
                parsed.matrix_2d_out_path =
                    Some(PathBuf::from(next_arg(args, &mut i, "--matrix-2d-out")?));
            }
            other => {
                return Err(format!(
                    "unrecognized metabolomics flag {other:?}\n\n{}",
                    metabolomics_usage()
                ));
            }
        }
        i += 1;
    }

    if parsed.config.min_ppm >= parsed.config.max_ppm {
        return Err(format!(
            "--min-ppm ({}) must be strictly less than --max-ppm ({})",
            parsed.config.min_ppm, parsed.config.max_ppm
        ));
    }

    // Mask preset + optional custom exclusions
    let base_preset = match mask_name.as_deref().unwrap_or("urine") {
        "urine" => BiofluidMaskPreset::Urine,
        "serum" | "plasma" | "serum-plasma" | "serum_plasma" => BiofluidMaskPreset::SerumPlasma,
        "aqueous" | "water" | "minimal" => BiofluidMaskPreset::AqueousMinimal,
        "none" | "off" => BiofluidMaskPreset::None,
        "custom" => BiofluidMaskPreset::Custom(Vec::new()),
        other => {
            return Err(format!(
                "invalid --mask {other:?} (expected urine, serum, aqueous, none, or custom)"
            ));
        }
    };
    parsed.config.mask_preset = if custom_zones.is_empty() {
        base_preset
    } else {
        let mut merged = base_preset.zones();
        merged.extend(custom_zones);
        BiofluidMaskPreset::Custom(merged)
    };

    parsed.config.alignment = if align_enabled {
        AlignmentMode::AutomaticIntervals {
            interval_width_ppm: align_interval,
            max_shift_ppm: align_max_shift,
            reference: align_ref,
        }
    } else {
        AlignmentMode::Disabled
    };

    parsed.config.bucketing = match method_name.as_str() {
        "aib" | "adaptive" | "intelligent" | "adaptive-intelligent" => {
            BucketingMethod::AdaptiveIntelligent {
                resolution: aib_res,
                min_width_ppm: aib_min_w,
                max_width_ppm: aib_max_w,
                multiplet_coalesce_hz: aib_coalesce_hz,
                min_bucket_snr: min_snr.unwrap_or(3.0),
            }
        }
        "uniform" | "fixed" => {
            if uniform_width <= 0.0 {
                return Err("--bin-width must be > 0".to_string());
            }
            BucketingMethod::Uniform {
                bin_width_ppm: uniform_width,
                min_bucket_snr: min_snr.unwrap_or(0.0),
            }
        }
        other => {
            return Err(format!(
                "invalid --method {other:?} (expected aib or uniform)"
            ));
        }
    };

    parsed.config.normalization = match norm_kind.as_str() {
        "pqn" | "probabilistic-quotient" => NormalizationMethod::ProbabilisticQuotient,
        "tsa" | "total-area" | "total_area" | "area" => NormalizationMethod::TotalSpectralArea,
        "tsp" | "dss" | "is" | "internal-standard" => NormalizationMethod::InternalStandard {
            target_ppm: tsp_ppm,
            half_window_ppm: tsp_half_win,
        },
        "none" | "raw" => NormalizationMethod::None,
        other => {
            return Err(format!(
                "invalid --norm {other:?} (expected pqn, tsa, tsp, or none)"
            ));
        }
    };

    Ok(Some(parsed))
}

fn read_text_capped(path: &Path, label: &str) -> Result<String, String> {
    let meta = fs::metadata(path)
        .map_err(|e| format!("failed to stat {label} {}: {e}", path.display()))?;
    if meta.len() > 64 * 1024 * 1024 {
        return Err(format!(
            "{label} {} ({} bytes) exceeds 64 MiB limit",
            path.display(),
            meta.len()
        ));
    }
    fs::read_to_string(path).map_err(|e| format!("failed reading {label} {}: {e}", path.display()))
}

fn load_spectrum_1d_from_path(
    path: &Path,
    sample_id: &str,
    fallback_obs_mhz: f64,
) -> Result<Spectrum1D, String> {
    let raw = read_text_capped(path, "1D spectrum")?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(format!("spectrum file {} is empty", path.display()));
    }

    // 1. JSON serialized Spectrum1D
    if trimmed.starts_with('{') {
        let spec = serde_json::from_str::<Spectrum1D>(trimmed)
            .map_err(|e| format!("failed parsing Spectrum1D JSON {}: {e}", path.display()))?;
        spec.validate()?;
        return Ok(spec);
    }

    // 2. JCAMP-DX 1D NMR (`##TITLE=`)
    if trimmed.starts_with("##") {
        let jdx_blocks = read_jcamp(trimmed)
            .map_err(|e| format!("failed parsing JCAMP-DX {}: {e:?}", path.display()))?;
        let first = jdx_blocks
            .first()
            .ok_or_else(|| format!("JCAMP-DX {} contained zero spectrum blocks", path.display()))?;
        let xs = first.x_axis();
        if xs.len() < 8 || xs.len() != first.y.len() {
            return Err(format!(
                "JCAMP-DX {} has invalid 1D trace length {}",
                path.display(),
                xs.len()
            ));
        }
        if !xs.iter().all(|v| v.is_finite()) || !first.y.iter().all(|v| v.is_finite()) {
            return Err(format!(
                "JCAMP-DX {} contains non-finite values",
                path.display()
            ));
        }
        let obs = first
            .mhz
            .filter(|&v| v.is_finite() && v > 1.0)
            .unwrap_or(fallback_obs_mhz);
        return build_uniform_spectrum_from_xy(sample_id, &xs, &first.y, obs);
    }

    // 3. 2-column ASCII table (`ppm\tintensity` or `ppm,intensity`)
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut obs_mhz = fallback_obs_mhz;
    let mut header_seen = false;
    for (line_no, line) in trimmed.lines().enumerate() {
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        if let Some(rest) = t.strip_prefix('#') {
            if let Some(idx) = rest.to_ascii_lowercase().find("obs_mhz=") {
                let tail = &rest[idx + 8..];
                let tok = tail.split_whitespace().next().unwrap_or("");
                if let Ok(v) = tok.parse::<f64>() {
                    if v.is_finite() && v > 1.0 {
                        obs_mhz = v;
                    }
                }
            }
            continue;
        }
        let cols: Vec<&str> = t
            .split(['\t', ',', ';', ' '])
            .filter(|s| !s.is_empty())
            .collect();
        if cols.len() < 2 {
            return Err(format!(
                "short row at {}:{}: {t:?}",
                path.display(),
                line_no + 1
            ));
        }
        let x_res = cols[0].parse::<f64>();
        let y_res = cols[1].parse::<f64>();
        match (x_res, y_res) {
            (Ok(x), Ok(y)) if x.is_finite() && y.is_finite() => {
                xs.push(x);
                ys.push(y);
            }
            (Err(_), _) if xs.is_empty() && !header_seen => {
                // Allow a single non-numeric column header line before data rows
                header_seen = true;
                continue;
            }
            _ => {
                return Err(format!(
                    "invalid or non-finite numeric (ppm, intensity) row at {}:{}: {t:?}",
                    path.display(),
                    line_no + 1
                ));
            }
        }
    }
    if xs.len() < 8 {
        return Err(format!(
            "spectrum {} contained fewer than 8 numeric (ppm, intensity) points",
            path.display()
        ));
    }
    build_uniform_spectrum_from_xy(sample_id, &xs, &ys, obs_mhz)
}

fn build_uniform_spectrum_from_xy(
    name: &str,
    xs: &[f64],
    ys: &[f64],
    obs_mhz: f64,
) -> Result<Spectrum1D, String> {
    let mut pairs: Vec<(f64, f64)> = xs.iter().copied().zip(ys.iter().copied()).collect();
    // Sort descending in ppm (`first_ppm > last_ppm`, `step_ppm < 0`)
    pairs.sort_by(|a, b| b.0.total_cmp(&a.0));
    pairs.dedup_by(|a, b| (a.0 - b.0).abs() <= 1e-12);
    if pairs.len() < 8 {
        return Err(format!(
            "spectrum {name:?} has fewer than 8 distinct chemical-shift points"
        ));
    }
    let hi_ppm = pairs.first().map(|p| p.0).unwrap_or(10.0);
    let lo_ppm = pairs.last().map(|p| p.0).unwrap_or(0.0);
    let span = hi_ppm - lo_ppm;
    if span <= 1e-6 {
        return Err(format!(
            "spectrum {name:?} has degenerate chemical-shift span [{lo_ppm}, {hi_ppm}]"
        ));
    }
    let n = pairs.len();
    let step_ppm = -span / ((n - 1) as f64);

    // Check if already uniformly spaced to within 1e-6 ppm
    let is_uniform = pairs
        .iter()
        .enumerate()
        .all(|(k, &(ppm, _))| ((hi_ppm + (k as f64) * step_ppm) - ppm).abs() <= 2e-5);
    let real = if is_uniform {
        pairs.iter().map(|p| p.1).collect()
    } else {
        // Linear interpolation onto uniform descending grid
        let mut out = Vec::with_capacity(n);
        let mut cursor = 0usize;
        for k in 0..n {
            let target = hi_ppm + (k as f64) * step_ppm;
            while cursor + 2 < pairs.len() && pairs[cursor + 1].0 >= target {
                cursor += 1;
            }
            let (x0, y0) = pairs[cursor];
            let (x1, y1) = pairs[cursor + 1];
            let denom = x0 - x1;
            let val = if denom.abs() <= 1e-12 {
                0.5 * (y0 + y1)
            } else {
                let frac = ((x0 - target) / denom).clamp(0.0, 1.0);
                y0 + frac * (y1 - y0)
            };
            out.push(val);
        }
        out
    };

    let axis = AxisPpm {
        label: "1H".to_string(),
        obs_mhz,
        size: n,
        first_ppm: hi_ppm,
        step_ppm,
    };
    Ok(Spectrum1D::new(
        name.to_string(),
        Experiment1DKind::Proton1D,
        axis,
        real,
        None,
    ))
}

fn load_cohort_samples(cli: &CommonMetabCliArgs) -> Result<Vec<CohortSample1D>, String> {
    let mut samples = Vec::new();

    if let Some(ref cohort_path) = cli.cohort_json {
        let txt = read_text_capped(cohort_path, "--cohort")?;
        let mut loaded: Vec<CohortSample1D> = serde_json::from_str(&txt).map_err(|e| {
            format!(
                "failed parsing CohortSample1D JSON {}: {e}",
                cohort_path.display()
            )
        })?;
        for s in &loaded {
            s.spectrum.validate()?;
        }
        samples.append(&mut loaded);
    }

    if let Some(ref manifest_path) = cli.manifest_path {
        let txt = read_text_capped(manifest_path, "--manifest")?;
        let base_dir = manifest_path.parent().unwrap_or_else(|| Path::new("."));
        for (line_idx, line) in txt.lines().enumerate() {
            let t = line.trim();
            if t.is_empty() || t.starts_with('#') {
                continue;
            }
            let sep = if t.contains('\t') { '\t' } else { ',' };
            let cols: Vec<&str> = t.split(sep).map(|s| s.trim()).collect();
            if cols.len() < 3 {
                return Err(format!(
                    "manifest {}:{} requires 3 columns (sample_id, group, path)",
                    manifest_path.display(),
                    line_idx + 1
                ));
            }
            if line_idx == 0 {
                let c0 = cols[0].to_ascii_lowercase();
                if c0 == "id" || c0 == "sample" || c0 == "sample_id" {
                    continue;
                }
            }
            let id = cols[0];
            let group = cols[1];
            let raw_p = Path::new(cols[2]);
            let resolved = if raw_p.is_absolute() {
                raw_p.to_path_buf()
            } else {
                base_dir.join(raw_p)
            };
            let spec = load_spectrum_1d_from_path(&resolved, id, cli.fallback_obs_mhz)?;
            samples.push(CohortSample1D::new(id, group, spec));
        }
    }

    for spec_arg in &cli.sample_specs {
        let Some((lhs, rhs_path)) = spec_arg.split_once('=') else {
            return Err(format!(
                "invalid --sample {spec_arg:?} (expected ID:GROUP=PATH)"
            ));
        };
        let Some((id, group)) = lhs.split_once(':') else {
            return Err(format!(
                "invalid --sample {spec_arg:?} (expected ID:GROUP=PATH)"
            ));
        };
        let id = id.trim();
        let group = group.trim();
        let path = Path::new(rhs_path.trim());
        if id.is_empty() || group.is_empty() {
            return Err(format!("empty sample id or group in --sample {spec_arg:?}"));
        }
        let spec = load_spectrum_1d_from_path(path, id, cli.fallback_obs_mhz)?;
        samples.push(CohortSample1D::new(id, group, spec));
    }

    if cli.demo {
        let mut demo_samples = generate_demo_biofluid_cohort(cli.config.rng_seed);
        samples.append(&mut demo_samples);
    }

    if samples.len() < 2 {
        return Err(
            "at least 2 cohort samples are required (supply --cohort, --manifest, repeatable --sample, or --demo)"
                .to_string(),
        );
    }

    if cli.permute_cohort_labels {
        let mut rng = Rng::new(cli.config.rng_seed ^ 0x9E37_79B9_7F4A_7C15);
        let mut groups: Vec<String> = samples.iter().map(|s| s.group.clone()).collect();
        for idx in (1..groups.len()).rev() {
            let j = rng.usize(idx + 1);
            groups.swap(idx, j);
        }
        for (sample, new_g) in samples.iter_mut().zip(groups) {
            sample.group = new_g;
        }
    }

    Ok(samples)
}

fn parse_bucket_header_ppm(col_header: &str, idx: usize) -> f64 {
    let trimmed = col_header.trim().trim_matches('"');
    let without_suffix = trimmed.strip_suffix("ppm").unwrap_or(trimmed);
    if let Some((_, rhs)) = without_suffix.rsplit_once('_') {
        if let Ok(v) = rhs.parse::<f64>() {
            if v.is_finite() {
                return v;
            }
        }
    }
    if let Ok(v) = without_suffix.parse::<f64>() {
        if v.is_finite() {
            return v;
        }
    }
    9.50 - (idx as f64) * 0.04
}

fn parse_bucket_table_csv_text(
    text: &str,
    exclusion_zones: Vec<ExclusionZonePpm>,
) -> Result<LoadedMetabolomicsDataset, String> {
    let mut lines = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'));
    let Some(header_line) = lines.next() else {
        return Err("bucket table CSV is empty".to_string());
    };
    let sep = if header_line.contains('\t') && !header_line.contains(',') {
        '\t'
    } else {
        ','
    };
    let headers: Vec<&str> = header_line.split(sep).collect();
    if headers.len() < 4 {
        return Err(
            "bucket table CSV requires sample_id,group,dilution_factor,<bucket_columns...>"
                .to_string(),
        );
    }
    let has_dilution_col = headers[2].trim().to_ascii_lowercase().contains("dilution");
    let bucket_start_col = if has_dilution_col { 3 } else { 2 };
    let n_buckets = headers.len() - bucket_start_col;
    if n_buckets == 0 {
        return Err("bucket table CSV has zero spectral bucket columns".to_string());
    }

    let centers: Vec<f64> = headers[bucket_start_col..]
        .iter()
        .enumerate()
        .map(|(j, h)| parse_bucket_header_ppm(h, j))
        .collect();

    let mut buckets = Vec::with_capacity(n_buckets);
    for (j, &c_ppm) in centers.iter().enumerate() {
        let half_w = if n_buckets >= 2 {
            let left_d = if j > 0 {
                (centers[j - 1] - c_ppm).abs()
            } else {
                (c_ppm - centers[j + 1]).abs()
            };
            let right_d = if j + 1 < n_buckets {
                (c_ppm - centers[j + 1]).abs()
            } else {
                left_d
            };
            0.5 * left_d.min(right_d).clamp(0.01, 0.08)
        } else {
            0.02
        };
        buckets.push(SpectralBucket {
            index: j,
            low_ppm: c_ppm - half_w,
            high_ppm: c_ppm + half_w,
            center_ppm: c_ppm,
            width_ppm: 2.0 * half_w,
            start_grid_idx: j * 2,
            end_grid_idx: j * 2 + 1,
            max_snr: 10.0,
        });
    }

    let mut sample_ids = Vec::new();
    let mut sample_groups = Vec::new();
    let mut dilution_factors = Vec::new();
    let mut matrix = Vec::new();

    for (row_idx, line) in lines.enumerate() {
        let cols: Vec<&str> = line.split(sep).collect();
        if cols.len() != headers.len() {
            return Err(format!(
                "bucket table row {} has {} columns (expected {})",
                row_idx + 2,
                cols.len(),
                headers.len()
            ));
        }
        sample_ids.push(cols[0].trim().trim_matches('"').to_string());
        sample_groups.push(cols[1].trim().trim_matches('"').to_string());
        let dil = if has_dilution_col {
            let d = cols[2]
                .trim()
                .parse::<f64>()
                .map_err(|_| format!("invalid dilution_factor at row {}", row_idx + 2))?;
            if !d.is_finite() || d <= 0.0 {
                return Err(format!(
                    "non-positive or non-finite dilution_factor at row {}: {d}",
                    row_idx + 2
                ));
            }
            d
        } else {
            1.0
        };
        dilution_factors.push(dil);
        let mut row = Vec::with_capacity(n_buckets);
        for (j, raw_v) in cols[bucket_start_col..].iter().enumerate() {
            let v = raw_v.trim().parse::<f64>().map_err(|_| {
                format!(
                    "invalid bucket intensity {:?} at row {}, bucket {}",
                    raw_v,
                    row_idx + 2,
                    j
                )
            })?;
            if !v.is_finite() {
                return Err(format!(
                    "non-finite bucket intensity {:?} at row {}, bucket {}",
                    raw_v,
                    row_idx + 2,
                    j
                ));
            }
            row.push(v);
        }
        matrix.push(row);
    }

    if sample_ids.len() < 2 {
        return Err("bucket table CSV must contain at least 2 sample rows".to_string());
    }

    let _ = dilution_factors;
    Ok(LoadedMetabolomicsDataset::BucketTable {
        sample_ids,
        sample_groups,
        buckets,
        normalized_bucket_matrix: matrix,
        exclusion_zones,
    })
}

fn validate_bucket_matrix_dims(
    n_samples: usize,
    n_groups: usize,
    n_buckets: usize,
    matrix: &[Vec<f64>],
    source: &Path,
) -> Result<(), String> {
    if n_samples < 2 || n_groups != n_samples || n_buckets == 0 || matrix.len() != n_samples {
        return Err(format!(
            "invalid metabolomics dataset dimensions in {} (samples={n_samples}, groups={n_groups}, buckets={n_buckets}, rows={})",
            source.display(),
            matrix.len()
        ));
    }
    for (r_idx, row) in matrix.iter().enumerate() {
        if row.len() != n_buckets || !row.iter().all(|v| v.is_finite()) {
            return Err(format!(
                "invalid or non-finite bucket row {r_idx} in {} (len={}, expected {n_buckets})",
                source.display(),
                row.len()
            ));
        }
    }
    Ok(())
}

fn load_dataset_for_chemometrics(
    cli: &CommonMetabCliArgs,
) -> Result<LoadedMetabolomicsDataset, String> {
    if let Some(ref in_path) = cli.in_path {
        let txt = read_text_capped(in_path, "--in")?;
        let trimmed = txt.trim();
        if trimmed.starts_with('{') {
            if let Ok(study) = serde_json::from_str::<MetabolomicsStudyResult>(trimmed) {
                validate_bucket_matrix_dims(
                    study.sample_ids.len(),
                    study.sample_groups.len(),
                    study.buckets.len(),
                    &study.normalized_bucket_matrix,
                    in_path,
                )?;
                return Ok(LoadedMetabolomicsDataset::Study(Box::new(study)));
            }
            if let Ok(bin_rep) = serde_json::from_str::<BinningCliReportJson>(trimmed) {
                validate_bucket_matrix_dims(
                    bin_rep.sample_ids.len(),
                    bin_rep.sample_groups.len(),
                    bin_rep.buckets.len(),
                    &bin_rep.normalized_bucket_matrix,
                    in_path,
                )?;
                let zones = bin_rep
                    .mask_zones_ppm
                    .iter()
                    .map(|z| ExclusionZonePpm::new(z[0], z[1]))
                    .collect();
                return Ok(LoadedMetabolomicsDataset::BucketTable {
                    sample_ids: bin_rep.sample_ids,
                    sample_groups: bin_rep.sample_groups,
                    buckets: bin_rep.buckets,
                    normalized_bucket_matrix: bin_rep.normalized_bucket_matrix,
                    exclusion_zones: zones,
                });
            }
            return Err(format!(
                "JSON file {} is neither a MetabolomicsStudyResult nor a BinningCliReportJson",
                in_path.display()
            ));
        }
        if trimmed.starts_with('[') {
            let mut cohort: Vec<CohortSample1D> = serde_json::from_str(trimmed).map_err(|e| {
                format!(
                    "failed parsing CohortSample1D JSON array {}: {e}",
                    in_path.display()
                )
            })?;
            if cohort.len() < 2 {
                return Err(format!(
                    "CohortSample1D JSON array {} must contain >= 2 samples",
                    in_path.display()
                ));
            }
            for s in &cohort {
                s.spectrum.validate()?;
            }
            if cli.permute_cohort_labels {
                let mut rng = Rng::new(cli.config.rng_seed ^ 0x9E37_79B9_7F4A_7C15);
                let mut groups: Vec<String> = cohort.iter().map(|s| s.group.clone()).collect();
                for idx in (1..groups.len()).rev() {
                    let j = rng.usize(idx + 1);
                    groups.swap(idx, j);
                }
                for (sample, new_g) in cohort.iter_mut().zip(groups) {
                    sample.group = new_g;
                }
            }
            let study = run_metabolomics_pipeline(&cohort, &cli.config);
            return Ok(LoadedMetabolomicsDataset::Study(Box::new(study)));
        }
        return parse_bucket_table_csv_text(trimmed, cli.config.mask_preset.zones());
    }

    let samples = load_cohort_samples(cli)?;
    let study = run_metabolomics_pipeline(&samples, &cli.config);
    Ok(LoadedMetabolomicsDataset::Study(Box::new(study)))
}

fn write_parent_and_file(path: &Path, content: &str) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("failed creating directory {}: {e}", parent.display()))?;
        }
    }
    fs::write(path, content).map_err(|e| format!("failed writing {}: {e}", path.display()))
}

fn cmd_bin(args: &[String]) -> Result<(), String> {
    let Some(cli) = parse_common_args(args)? else {
        print!("{}", metabolomics_usage());
        return Ok(());
    };
    let samples = load_cohort_samples(&cli)?;
    let study = run_metabolomics_pipeline(&samples, &cli.config);

    // Alternatively verify preprocess + align + bucket + norm + scale already populated in `study`
    let _ = (
        preprocess_cohort_1d,
        align_cohort_interval_fft,
        bucket_cohort_spectra,
        normalize_bucket_matrix,
    );

    if let Some(ref study_out) = cli.study_out_path {
        let study_json = serde_json::to_string_pretty(&study)
            .map_err(|e| format!("failed serializing study JSON: {e}"))?;
        write_parent_and_file(study_out, &study_json)?;
    }

    let mask_zones_ppm: Vec<[f64; 2]> = cli
        .config
        .mask_preset
        .zones()
        .into_iter()
        .map(|z| [z.low_ppm, z.high_ppm])
        .collect();
    let bin_report = BinningCliReportJson {
        n_samples: study.sample_ids.len(),
        n_buckets: study.buckets.len(),
        sample_ids: study.sample_ids.clone(),
        sample_groups: study.sample_groups.clone(),
        dilution_factors: study.dilution_factors.clone(),
        normalization: format_normalization(cli.config.normalization),
        scaling: format_scaling(cli.config.scaling).to_string(),
        bucketing: format_bucketing(cli.config.bucketing),
        mask_zones_ppm,
        buckets: study.buckets.clone(),
        normalized_bucket_matrix: study.normalized_bucket_matrix.clone(),
        scaled_bucket_matrix: study.scaled_bucket_matrix.clone(),
    };

    if let Some(ref out_path) = cli.out_path {
        let ext = out_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if ext == "json" {
            let js = serde_json::to_string_pretty(&bin_report)
                .map_err(|e| format!("failed serializing binning JSON: {e}"))?;
            write_parent_and_file(out_path, &js)?;
        } else {
            let mut csv = export_bucket_table_csv(&study);
            if ext == "tsv" {
                csv = csv.replace(',', "\t");
            }
            write_parent_and_file(out_path, &csv)?;
        }
    }

    let mut group_counts: BTreeMap<&str, usize> = BTreeMap::new();
    for g in &study.sample_groups {
        *group_counts.entry(g.as_str()).or_insert(0) += 1;
    }
    let groups_summary: Vec<String> = group_counts
        .into_iter()
        .map(|(g, n)| format!("{g}:{n}"))
        .collect();

    println!(
        "Metabolomics Binning Summary: samples={} ({}) buckets={} bucketing={} norm={} scale={}",
        study.sample_ids.len(),
        groups_summary.join(", "),
        study.buckets.len(),
        bin_report.bucketing,
        bin_report.normalization,
        bin_report.scaling,
    );
    if !study.dilution_factors.is_empty() {
        let d_min = study
            .dilution_factors
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let d_max = study
            .dilution_factors
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        println!("Dilution factors: min={d_min:.4} max={d_max:.4}");
    }
    Ok(())
}

fn cmd_pca(args: &[String]) -> Result<(), String> {
    let Some(cli) = parse_common_args(args)? else {
        print!("{}", metabolomics_usage());
        return Ok(());
    };
    let ds = load_dataset_for_chemometrics(&cli)?;

    let (sample_ids, sample_groups, n_buckets, scaling_used, pca): (
        Vec<String>,
        Vec<String>,
        usize,
        ScalingMethod,
        PcaResult,
    ) = match ds {
        LoadedMetabolomicsDataset::Study(study) => {
            let vt = cli
                .transform_override
                .unwrap_or(study.config.variance_transform);
            let sc = cli.scale_override.unwrap_or(study.config.scaling);
            let n_comp = cli
                .pca_components_override
                .unwrap_or(study.config.pca_components);
            let pca_res = if cli.transform_override.is_none()
                && cli.scale_override.is_none()
                && cli.pca_components_override.is_none()
            {
                study.pca
            } else {
                let scaled =
                    transform_and_scale_bucket_matrix(&study.normalized_bucket_matrix, vt, sc);
                fit_pca(&scaled, &study.sample_groups, n_comp)
            };
            (
                study.sample_ids,
                study.sample_groups,
                study.buckets.len(),
                sc,
                pca_res,
            )
        }
        LoadedMetabolomicsDataset::BucketTable {
            sample_ids,
            sample_groups,
            buckets,
            normalized_bucket_matrix,
            ..
        } => {
            let vt = cli.config.variance_transform;
            let sc = cli.config.scaling;
            let scaled = transform_and_scale_bucket_matrix(&normalized_bucket_matrix, vt, sc);
            let pca_res = fit_pca(&scaled, &sample_groups, cli.config.pca_components);
            (sample_ids, sample_groups, buckets.len(), sc, pca_res)
        }
    };

    let mut samples_rows = Vec::with_capacity(sample_ids.len());
    let mut t2_outliers = Vec::new();
    let mut q_outliers = Vec::new();
    for i in 0..sample_ids.len() {
        let t2 = pca.hotelling_t2.get(i).copied().unwrap_or(0.0);
        let q = pca.q_residuals.get(i).copied().unwrap_or(0.0);
        let is_t2 = t2 > pca.hotelling_t2_limit_95;
        let is_q = q > pca.q_residual_limit_95;
        if is_t2 {
            t2_outliers.push(sample_ids[i].clone());
        }
        if is_q {
            q_outliers.push(sample_ids[i].clone());
        }
        samples_rows.push(PcaSampleRowJson {
            sample_id: sample_ids[i].clone(),
            group: sample_groups[i].clone(),
            scores: pca.scores.get(i).cloned().unwrap_or_default(),
            hotelling_t2: t2,
            q_residual: q,
            is_t2_outlier_95: is_t2,
            is_q_outlier_95: is_q,
        });
    }

    let report = PcaCliReportJson {
        n_samples: sample_ids.len(),
        n_buckets,
        n_components: pca.n_components,
        scaling: format_scaling(scaling_used).to_string(),
        explained_variance_ratio: pca.explained_variance_ratio.clone(),
        cumulative_variance_ratio: pca.cumulative_variance_ratio.clone(),
        hotelling_t2_limit_95: pca.hotelling_t2_limit_95,
        hotelling_t2_limit_99: pca.hotelling_t2_limit_99,
        q_residual_limit_95: pca.q_residual_limit_95,
        t2_outliers_95: t2_outliers.clone(),
        q_outliers_95: q_outliers.clone(),
        global_ellipse_95: pca.global_ellipse_95,
        group_ellipses_95: pca.group_ellipses_95,
        samples: samples_rows,
    };

    if let Some(ref out_path) = cli.out_path {
        let ext = out_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if ext == "tsv" || ext == "csv" {
            let sep = if ext == "csv" { ',' } else { '\t' };
            let mut tbl = format!("sample_id{sep}group");
            for c in 1..=report.n_components {
                tbl.push_str(&format!("{sep}PC{c}"));
            }
            tbl.push_str(&format!(
                "{sep}hotelling_t2{sep}q_residual{sep}t2_outlier_95{sep}q_outlier_95\n"
            ));
            for row in &report.samples {
                tbl.push_str(&format!("{}{sep}{}", row.sample_id, row.group));
                for &sc_val in &row.scores {
                    tbl.push_str(&format!("{sep}{sc_val:.6}"));
                }
                tbl.push_str(&format!(
                    "{sep}{:.6}{sep}{:.6e}{sep}{}{sep}{}\n",
                    row.hotelling_t2, row.q_residual, row.is_t2_outlier_95, row.is_q_outlier_95
                ));
            }
            write_parent_and_file(out_path, &tbl)?;
        } else {
            let js = serde_json::to_string_pretty(&report)
                .map_err(|e| format!("failed serializing PCA JSON: {e}"))?;
            write_parent_and_file(out_path, &js)?;
        }
    }

    let pc1 = report
        .explained_variance_ratio
        .first()
        .copied()
        .unwrap_or(0.0)
        * 100.0;
    let pc2 = report
        .explained_variance_ratio
        .get(1)
        .copied()
        .unwrap_or(0.0)
        * 100.0;
    let cum = report
        .cumulative_variance_ratio
        .last()
        .copied()
        .unwrap_or(0.0)
        * 100.0;
    println!(
        "PCA Summary: samples={} buckets={} components={} scale={} PC1={pc1:.2}% PC2={pc2:.2}% Cum={cum:.2}%",
        report.n_samples, report.n_buckets, report.n_components, report.scaling
    );
    println!(
        "Outlier Gates (95%): Hotelling_T2_crit={:.4} (outliers={}) Q_SPE_crit={:.4e} (outliers={})",
        report.hotelling_t2_limit_95,
        report.t2_outliers_95.len(),
        report.q_residual_limit_95,
        report.q_outliers_95.len()
    );
    Ok(())
}

fn cmd_oplsda(args: &[String]) -> Result<(), String> {
    let Some(cli) = parse_common_args(args)? else {
        print!("{}", metabolomics_usage());
        return Ok(());
    };
    let ds = load_dataset_for_chemometrics(&cli)?;

    type OplsUnpacked = (
        Vec<String>,
        Vec<String>,
        Vec<SpectralBucket>,
        OplsDaResult,
        Vec<BiomarkerCandidate>,
        usize,
        Option<Box<MetabolomicsStudyResult>>,
    );
    let (
        sample_ids,
        sample_groups,
        buckets,
        opls,
        biomarkers,
        cv_folds_used,
        full_study_opt,
    ): OplsUnpacked = match ds {
        LoadedMetabolomicsDataset::Study(study) => {
            let vt = cli
                .transform_override
                .unwrap_or(study.config.variance_transform);
            let sc = cli.scale_override.unwrap_or(study.config.scaling);
            let n_ortho = cli
                .ortho_override
                .unwrap_or(study.config.opls_orthogonal_components);
            let max_ortho = cli
                .max_ortho_override
                .or(study.config.max_opls_orthogonal_components);
            let folds = cli.cv_folds_override.unwrap_or(study.config.cv_folds);
            let n_perm = cli
                .permutations_override
                .unwrap_or(study.config.n_permutations);
            let seed = cli.seed_override.unwrap_or(study.config.rng_seed);
            let ctrl = cli
                .control_override
                .as_deref()
                .or(study.config.control_group.as_deref());
            let case = cli
                .case_override
                .as_deref()
                .or(study.config.case_group.as_deref());

            let no_overrides = cli.transform_override.is_none()
                && cli.scale_override.is_none()
                && cli.ortho_override.is_none()
                && cli.max_ortho_override.is_none()
                && cli.cv_folds_override.is_none()
                && cli.permutations_override.is_none()
                && cli.seed_override.is_none()
                && cli.control_override.is_none()
                && cli.case_override.is_none();

            if no_overrides {
                let Some(existing_opls) = study.opls_da.clone() else {
                    return Err(
                        "OPLS-DA requires at least 2 sample groups with >= 2 samples each"
                            .to_string(),
                    );
                };
                let bm = study.biomarkers.clone();
                let s_ids = study.sample_ids.clone();
                let s_grps = study.sample_groups.clone();
                let bks = study.buckets.clone();
                (s_ids, s_grps, bks, existing_opls, bm, folds, Some(study))
            } else {
                let opls_res = fit_opls_da_with_options(
                    &study.normalized_bucket_matrix,
                    &study.sample_groups,
                    vt,
                    sc,
                    n_ortho,
                    max_ortho,
                    folds,
                    n_perm,
                    seed,
                    ctrl,
                    case,
                )
                .ok_or_else(|| {
                    "OPLS-DA requires at least 2 sample groups with >= 2 samples each".to_string()
                })?;
                let pls_res = fit_pls_da_with_transform(
                    &study.normalized_bucket_matrix,
                    &study.sample_groups,
                    vt,
                    sc,
                    study.config.pls_components,
                    folds,
                    ctrl,
                    case,
                );
                let zones = study.config.mask_preset.zones();
                let bm = rank_biomarkers(
                    &study.buckets,
                    &study.normalized_bucket_matrix,
                    Some(&opls_res),
                    pls_res.as_ref(),
                    &zones,
                );
                let mut updated = study;
                updated.opls_da = Some(opls_res.clone());
                updated.biomarkers = bm.clone();
                let s_ids = updated.sample_ids.clone();
                let s_grps = updated.sample_groups.clone();
                let bks = updated.buckets.clone();
                (s_ids, s_grps, bks, opls_res, bm, folds, Some(updated))
            }
        }
        LoadedMetabolomicsDataset::BucketTable {
            sample_ids,
            sample_groups,
            buckets,
            normalized_bucket_matrix,
            exclusion_zones,
            ..
        } => {
            let folds = cli.config.cv_folds;
            let opls_res = fit_opls_da_with_options(
                &normalized_bucket_matrix,
                &sample_groups,
                cli.config.variance_transform,
                cli.config.scaling,
                cli.config.opls_orthogonal_components,
                cli.config.max_opls_orthogonal_components,
                folds,
                cli.config.n_permutations,
                cli.config.rng_seed,
                cli.config.control_group.as_deref(),
                cli.config.case_group.as_deref(),
            )
            .ok_or_else(|| {
                "OPLS-DA requires at least 2 sample groups with >= 2 samples each".to_string()
            })?;
            let pls_res = fit_pls_da_with_transform(
                &normalized_bucket_matrix,
                &sample_groups,
                cli.config.variance_transform,
                cli.config.scaling,
                cli.config.pls_components,
                folds,
                cli.config.control_group.as_deref(),
                cli.config.case_group.as_deref(),
            );
            let bm = rank_biomarkers(
                &buckets,
                &normalized_bucket_matrix,
                Some(&opls_res),
                pls_res.as_ref(),
                &exclusion_zones,
            );
            (
                sample_ids,
                sample_groups,
                buckets,
                opls_res,
                bm,
                folds,
                None,
            )
        }
    };

    let overfit_warning = if !opls.permutation.is_validated {
        Some(format!(
            "OVERFIT WARNING: OPLS-DA model did NOT pass permutation/CV validation (Q2={:.4}, perm_p={:.4}; require Q2 >= 0.40 and perm_p <= 0.05)",
            opls.q2, opls.permutation.p_value_q2
        ))
    } else {
        None
    };

    let mut sample_rows = Vec::with_capacity(opls.active_sample_indices.len());
    for (local_i, &orig_i) in opls.active_sample_indices.iter().enumerate() {
        let to1 = opls
            .orthogonal_scores
            .first()
            .and_then(|v| v.get(local_i))
            .copied()
            .unwrap_or(0.0);
        sample_rows.push(OplsDaSampleRowJson {
            sample_id: sample_ids.get(orig_i).cloned().unwrap_or_default(),
            group: sample_groups.get(orig_i).cloned().unwrap_or_default(),
            y_binary: opls.y_binary.get(local_i).copied().unwrap_or(0.0),
            t_predictive: opls.predictive_scores.get(local_i).copied().unwrap_or(0.0),
            t_orthogonal_1: to1,
            y_fitted: opls.y_fitted.get(local_i).copied().unwrap_or(0.0),
            y_cv_predicted: opls.y_cv_predicted.get(local_i).copied().unwrap_or(0.0),
        });
    }

    let top_biomarkers: Vec<BiomarkerCandidate> =
        biomarkers.iter().take(cli.top_k).cloned().collect();
    let report = OplsDaCliReportJson {
        control_group: opls.control_group.clone(),
        case_group: opls.case_group.clone(),
        n_active_samples: opls.active_sample_indices.len(),
        n_buckets: buckets.len(),
        n_orthogonal: opls.n_orthogonal,
        cv_folds: cv_folds_used,
        n_permutations: opls.permutation.n_permutations,
        r2_x_predictive: opls.r2_x_predictive,
        r2_x_orthogonal: opls.r2_x_orthogonal,
        r2_x_total: opls.r2_x_total,
        r2_y: opls.r2_y,
        q2: opls.q2,
        permutation_p_value_q2: opls.permutation.p_value_q2,
        permutation_p_value_r2_y: opls.permutation.p_value_r2_y,
        permutation_mean_null_q2: opls.permutation.mean_null_q2,
        permutation_q2_intercept: opls.permutation.q2_intercept,
        is_validated: opls.permutation.is_validated,
        overfit_warning: overfit_warning.clone(),
        samples: sample_rows,
        top_biomarkers: top_biomarkers.clone(),
    };

    if let Some(ref bm_out) = cli.biomarkers_out_path {
        if let Some(ref st) = full_study_opt {
            write_parent_and_file(bm_out, &export_biomarker_ranking_csv(st))?;
        } else {
            let mut csv = String::from(
                "rank,bucket_index,center_ppm,low_ppm,high_ppm,control_mean,case_mean,fold_change,log2_fc,welch_t,p_value,fdr_q,vip,s_plot_cov,s_plot_corr,composite_score,putative_metabolite,bmrb_id\n",
            );
            for c in &biomarkers {
                csv.push_str(&format!(
                    "{},{},{:.4},{:.4},{:.4},{:.6e},{:.6e},{:.4},{:.4},{:.4},{:.4e},{:.4e},{:.4},{:.6e},{:.4},{:.4},{},{}\n",
                    c.rank,
                    c.bucket_index,
                    c.center_ppm,
                    c.low_ppm,
                    c.high_ppm,
                    c.control_mean,
                    c.case_mean,
                    c.fold_change,
                    c.log2_fold_change,
                    c.welch_t,
                    c.p_value,
                    c.fdr_q,
                    c.vip,
                    c.s_plot_cov,
                    c.s_plot_corr,
                    c.composite_score,
                    c.putative_metabolite.as_deref().unwrap_or(""),
                    c.putative_bmrb_id.as_deref().unwrap_or(""),
                ));
            }
            write_parent_and_file(bm_out, &csv)?;
        }
    }

    if let Some(ref out_path) = cli.out_path {
        let ext = out_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if ext == "csv" || ext == "tsv" {
            let mut csv = String::from(
                "rank,bucket_index,center_ppm,vip,s_plot_cov,s_plot_corr,log2_fc,welch_t,fdr_q,putative_metabolite,bmrb_id\n",
            );
            for c in &top_biomarkers {
                csv.push_str(&format!(
                    "{},{},{:.4},{:.4},{:.6e},{:.4},{:.4},{:.4},{:.4e},{},{}\n",
                    c.rank,
                    c.bucket_index,
                    c.center_ppm,
                    c.vip,
                    c.s_plot_cov,
                    c.s_plot_corr,
                    c.log2_fold_change,
                    c.welch_t,
                    c.fdr_q,
                    c.putative_metabolite.as_deref().unwrap_or(""),
                    c.putative_bmrb_id.as_deref().unwrap_or(""),
                ));
            }
            if ext == "tsv" {
                csv = csv.replace(',', "\t");
            }
            write_parent_and_file(out_path, &csv)?;
        } else {
            let js = serde_json::to_string_pretty(&report)
                .map_err(|e| format!("failed serializing OPLS-DA JSON: {e}"))?;
            write_parent_and_file(out_path, &js)?;
        }
    }

    println!(
        "OPLS-DA Summary ({} vs {}): N={} P={} Ao={} R2Xp={:.4} R2Xo={:.4} R2Y={:.4} Q2={:.4} perm_p(Q2)={:.4} validated={}",
        report.control_group,
        report.case_group,
        report.n_active_samples,
        report.n_buckets,
        report.n_orthogonal,
        report.r2_x_predictive,
        report.r2_x_orthogonal,
        report.r2_y,
        report.q2,
        report.permutation_p_value_q2,
        report.is_validated
    );
    if let Some(ref warn) = report.overfit_warning {
        println!("{warn}");
    }
    for c in top_biomarkers.iter().take(5) {
        println!(
            "  Rank #{:<2} {:.3} ppm  VIP={:.3}  p(corr)={:+.3}  FDR_q={:.3e}  match={} ({})",
            c.rank,
            c.center_ppm,
            c.vip,
            c.s_plot_corr,
            c.fdr_q,
            c.putative_metabolite.as_deref().unwrap_or("-"),
            c.putative_bmrb_id.as_deref().unwrap_or("-")
        );
    }
    Ok(())
}

fn cmd_stocsy(args: &[String]) -> Result<(), String> {
    let Some(cli) = parse_common_args(args)? else {
        print!("{}", metabolomics_usage());
        return Ok(());
    };
    let Some(driver_ppm) = cli.driver_ppm else {
        return Err(
            "missing required --driver-ppm <PPM> for `ferric metabolomics stocsy`".to_string(),
        );
    };
    let ds = load_dataset_for_chemometrics(&cli)?;

    let (n_samples, buckets, normalized_matrix, stocsy) = match ds {
        LoadedMetabolomicsDataset::Study(study) => {
            let st = compute_stocsy_at_ppm(driver_ppm, &study).ok_or_else(|| {
                "failed computing STOCSY: cohort has insufficient samples or buckets".to_string()
            })?;
            (
                study.sample_ids.len(),
                study.buckets,
                study.normalized_bucket_matrix,
                st,
            )
        }
        LoadedMetabolomicsDataset::BucketTable {
            sample_ids,
            buckets,
            normalized_bucket_matrix,
            exclusion_zones,
            ..
        } => {
            let best_idx = buckets
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    (a.center_ppm - driver_ppm)
                        .abs()
                        .total_cmp(&(b.center_ppm - driver_ppm).abs())
                })
                .map(|(idx, _)| idx)
                .ok_or_else(|| "bucket table has zero buckets".to_string())?;
            let st = compute_stocsy(
                best_idx,
                &buckets,
                &normalized_bucket_matrix,
                None,
                &exclusion_zones,
            )
            .ok_or_else(|| {
                "failed computing STOCSY from bucket table: require >= 3 samples".to_string()
            })?;
            (sample_ids.len(), buckets, normalized_bucket_matrix, st)
        }
    };

    let mut hits: Vec<StocsyBucketHitJson> = buckets
        .iter()
        .enumerate()
        .filter_map(|(j, b)| {
            let r2 = stocsy.bucket_r2.get(j).copied().unwrap_or(0.0);
            if r2 >= cli.min_r2 {
                Some(StocsyBucketHitJson {
                    bucket_index: j,
                    center_ppm: b.center_ppm,
                    low_ppm: b.low_ppm,
                    high_ppm: b.high_ppm,
                    covariance: stocsy.bucket_covariance.get(j).copied().unwrap_or(0.0),
                    correlation: stocsy.bucket_correlation.get(j).copied().unwrap_or(0.0),
                    r2,
                })
            } else {
                None
            }
        })
        .collect();
    hits.sort_by(|a, b| {
        b.r2.total_cmp(&a.r2)
            .then_with(|| a.bucket_index.cmp(&b.bucket_index))
    });
    hits.truncate(cli.top_k);

    let matches: Vec<MetaboliteMatchScore> = stocsy
        .metabolite_matches
        .iter()
        .take(cli.top_k)
        .cloned()
        .collect();

    let report = StocsyCliReportJson {
        requested_driver_ppm: driver_ppm,
        driver_bucket_index: stocsy.driver_bucket_index,
        driver_bucket_ppm: stocsy.driver_ppm,
        n_samples,
        n_buckets: buckets.len(),
        min_r2_filter: cli.min_r2,
        covarying_buckets: hits,
        metabolite_matches: matches,
    };

    if let Some(ref mat2d_path) = cli.matrix_2d_out_path {
        let p = buckets.len();
        let mut corr_mat = vec![vec![0.0; p]; p];
        let mut cov_mat = vec![vec![0.0; p]; p];
        let empty_zones = Vec::new();
        for i in 0..p {
            if let Some(row_st) =
                compute_stocsy(i, &buckets, &normalized_matrix, None, &empty_zones)
            {
                corr_mat[i] = row_st.bucket_correlation;
                cov_mat[i] = row_st.bucket_covariance;
            }
        }
        let mat2d = Stocsy2dMatrixJson {
            bucket_centers_ppm: buckets.iter().map(|b| b.center_ppm).collect(),
            correlation_matrix: corr_mat,
            covariance_matrix: cov_mat,
        };
        let js = serde_json::to_string_pretty(&mat2d)
            .map_err(|e| format!("failed serializing 2D STOCSY matrix JSON: {e}"))?;
        write_parent_and_file(mat2d_path, &js)?;
    }

    if let Some(ref out_path) = cli.out_path {
        let ext = out_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if ext == "tsv" || ext == "csv" {
            let sep = if ext == "csv" { ',' } else { '\t' };
            let mut tbl = format!(
                "bucket_index{sep}center_ppm{sep}low_ppm{sep}high_ppm{sep}covariance{sep}correlation{sep}r2\n"
            );
            for h in &report.covarying_buckets {
                tbl.push_str(&format!(
                    "{}{sep}{:.4}{sep}{:.4}{sep}{:.4}{sep}{:.6e}{sep}{:+.4}{sep}{:.4}\n",
                    h.bucket_index,
                    h.center_ppm,
                    h.low_ppm,
                    h.high_ppm,
                    h.covariance,
                    h.correlation,
                    h.r2
                ));
            }
            write_parent_and_file(out_path, &tbl)?;
        } else {
            let js = serde_json::to_string_pretty(&report)
                .map_err(|e| format!("failed serializing STOCSY JSON: {e}"))?;
            write_parent_and_file(out_path, &js)?;
        }
    }

    println!(
        "STOCSY Summary: driver_req={:.3}ppm -> bucket #{} ({:.3}ppm) samples={} buckets={} hits(r2>={:.2})={}",
        report.requested_driver_ppm,
        report.driver_bucket_index,
        report.driver_bucket_ppm,
        report.n_samples,
        report.n_buckets,
        report.min_r2_filter,
        report.covarying_buckets.len()
    );
    for h in report.covarying_buckets.iter().take(6) {
        println!(
            "  Bucket #{:<3} {:>7.3} ppm  r={:+.4}  r2={:.4}  cov={:.4e}",
            h.bucket_index, h.center_ppm, h.correlation, h.r2, h.covariance
        );
    }
    for m in report.metabolite_matches.iter().take(5) {
        println!(
            "  BMRB Match: {} ({}) score={:.3} multiplets={}/{} mean_r2={:.3}",
            m.name,
            m.bmrb_id,
            m.score,
            m.matched_multiplets,
            m.observable_multiplets,
            m.mean_stocsy_r2
        );
    }
    Ok(())
}

fn parse_peaks_csv(raw: &str) -> Result<Vec<f64>, String> {
    let mut out = Vec::new();
    for tok in raw.split([',', ';', ' ']) {
        let t = tok.trim();
        if t.is_empty() {
            continue;
        }
        let v = parse_f64_flag(t, "--peaks")?;
        out.push(v);
    }
    if out.is_empty() {
        return Err(
            "expected at least one numeric ppm value in --peaks <ppm1,ppm2,...>".to_string(),
        );
    }
    Ok(out)
}

fn escape_delimited_field(raw: &str, sep: char) -> String {
    if sep == ',' {
        if raw.contains([',', '"', '\n', '\r']) {
            format!("\"{}\"", raw.replace('"', "\"\""))
        } else {
            raw.to_string()
        }
    } else {
        raw.replace(['\t', '\n', '\r'], " ")
    }
}

fn cmd_identify(args: &[String]) -> Result<(), String> {
    let mut peaks_ppm: Vec<f64> = Vec::new();
    let mut nucleus = MetaboliteNucleus::H1;
    let mut tol_override: Option<f64> = None;
    let mut online = false;
    let mut top_k: usize = 10;
    let mut out_path: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" | "help" => {
                print!("{}", metabolomics_usage());
                return Ok(());
            }
            "--peaks" => {
                let raw = next_arg(args, &mut i, "--peaks")?;
                peaks_ppm.extend(parse_peaks_csv(raw)?);
            }
            "--nucleus" => {
                let raw = next_arg(args, &mut i, "--nucleus")?;
                nucleus = MetaboliteNucleus::parse(raw)?;
            }
            "--tol" => {
                let tol = parse_f64_flag(next_arg(args, &mut i, "--tol")?, "--tol")?;
                if tol <= 0.0 || tol > MAX_METABOLITE_TOL_PPM {
                    return Err(format!(
                        "--tol must be in (0.0, {MAX_METABOLITE_TOL_PPM}] ppm (got {tol})"
                    ));
                }
                tol_override = Some(tol);
            }
            "--online" => {
                online = true;
            }
            "--top" => {
                top_k = parse_usize_flag(next_arg(args, &mut i, "--top")?, "--top")?.max(1);
            }
            "-o" | "--out" | "--output" => {
                let flag = args[i].as_str();
                out_path = Some(PathBuf::from(next_arg(args, &mut i, flag)?));
            }
            other => {
                return Err(format!(
                    "unrecognized flag {other:?} for `ferric metabolomics identify`\n\n{}",
                    metabolomics_usage()
                ));
            }
        }
        i += 1;
    }

    if peaks_ppm.is_empty() {
        return Err(
            "missing required --peaks <ppm1,ppm2,...> for `ferric metabolomics identify`"
                .to_string(),
        );
    }

    let tol_ppm = tol_override.unwrap_or_else(|| nucleus.default_tol_ppm());
    let query_peaks: Vec<QueryPeak> = peaks_ppm.iter().copied().map(QueryPeak::from_ppm).collect();

    let mut matches = query_offline_metabolites(&query_peaks, nucleus, tol_ppm, top_k);
    let mut online_error: Option<String> = None;

    if online {
        match query_online_bmrb_and_npmrd(
            &query_peaks,
            nucleus,
            tol_ppm,
            OnlineQuerySources::bmrb_only(),
        ) {
            Ok(online_hits) => {
                for hit in online_hits {
                    let dup = matches.iter().any(|m| {
                        m.entry.source == hit.entry.source && m.entry.bmse_id == hit.entry.bmse_id
                    });
                    if !dup {
                        matches.push(hit);
                    }
                }
                matches.sort_by(|a, b| {
                    b.score
                        .total_cmp(&a.score)
                        .then_with(|| b.matched_ref_peaks.cmp(&a.matched_ref_peaks))
                        .then_with(|| a.mean_abs_error_ppm.total_cmp(&b.mean_abs_error_ppm))
                        .then_with(|| a.entry.bmse_id.cmp(&b.entry.bmse_id))
                });
                matches.truncate(top_k);
            }
            Err(e) => {
                eprintln!("Warning: online BMRB v2 metabolite search unavailable ({e})");
                online_error = Some(e);
            }
        }
    }

    let report = IdentifyCliReportJson {
        nucleus: nucleus.as_str().to_string(),
        tol_ppm,
        query_peaks_ppm: peaks_ppm,
        online_requested: online,
        online_error,
        matches,
    };

    if let Some(ref p) = out_path {
        let ext = p
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if ext == "csv" || ext == "tsv" {
            let sep = if ext == "csv" { ',' } else { '\t' };
            let mut tbl = format!(
                "rank{sep}bmse_id{sep}name{sep}formula{sep}score{sep}matched_peaks{sep}total_ref_peaks{sep}mean_abs_err_ppm{sep}source{sep}smiles\n"
            );
            for (idx, m) in report.matches.iter().enumerate() {
                tbl.push_str(&format!(
                    "{}{sep}{}{sep}{}{sep}{}{sep}{:.4}{sep}{}{sep}{}{sep}{:.4}{sep}{}{sep}{}\n",
                    idx + 1,
                    escape_delimited_field(&m.entry.bmse_id, sep),
                    escape_delimited_field(&m.entry.name, sep),
                    escape_delimited_field(&m.entry.formula, sep),
                    m.score,
                    m.matched_ref_peaks,
                    m.total_ref_peaks,
                    m.mean_abs_error_ppm,
                    escape_delimited_field(m.entry.source.label(), sep),
                    escape_delimited_field(&m.entry.smiles, sep),
                ));
            }
            write_parent_and_file(p, &tbl)?;
        } else {
            let js = serde_json::to_string_pretty(&report)
                .map_err(|e| format!("failed serializing identify JSON: {e}"))?;
            write_parent_and_file(p, &js)?;
        }
    }

    println!(
        "Metabolite Identification ({}, tol={:.4} ppm, query_peaks={}, online={}): {} matches (NP-MRD web search: {})",
        report.nucleus,
        report.tol_ppm,
        report.query_peaks_ppm.len(),
        report.online_requested,
        report.matches.len(),
        npmrd_web_search_url(),
    );
    for (idx, m) in report.matches.iter().enumerate() {
        println!(
            "  #{:<2} {:<12} {:<24} score={:.3}  matched={}/{}  mean_err={:.4}ppm  [{}]",
            idx + 1,
            m.entry.bmse_id,
            m.entry.name,
            m.score,
            m.matched_ref_peaks,
            m.total_ref_peaks,
            m.mean_abs_error_ppm,
            m.entry.source.label()
        );
    }
    Ok(())
}

pub fn run_metabolomics_cli(args: &[String]) -> Result<(), String> {
    if args.is_empty() || matches!(args[0].as_str(), "-h" | "--help" | "help") {
        print!("{}", metabolomics_usage());
        return Ok(());
    }
    match args[0].as_str() {
        "bin" | "bucket" => cmd_bin(&args[1..]),
        "pca" => cmd_pca(&args[1..]),
        "oplsda" | "opls-da" | "opls" => cmd_oplsda(&args[1..]),
        "stocsy" => cmd_stocsy(&args[1..]),
        "identify" | "id" => cmd_identify(&args[1..]),
        other => Err(format!(
            "unknown `ferric metabolomics` subcommand {other:?} (expected bin, pca, oplsda, stocsy, or identify)\n\n{}",
            metabolomics_usage()
        )),
    }
}

#[cfg(test)]
mod identify_tests {
    use super::*;

    #[test]
    fn test_cli_metabolomics_identify_offline_1h_and_13c_and_error_cases() {
        let tmp = std::env::temp_dir().join(format!(
            "ferric_metab_identify_test_{}.json",
            std::process::id()
        ));
        let args: Vec<String> = vec![
            "identify".into(),
            "--peaks".into(),
            "1.318,4.101".into(),
            "--nucleus".into(),
            "1h".into(),
            "--tol".into(),
            "0.03".into(),
            "-o".into(),
            tmp.to_string_lossy().into_owned(),
        ];
        run_metabolomics_cli(&args).expect("identify 1H L-(+) Lactic acid succeeds");
        let body = fs::read_to_string(&tmp).expect("read identify json");
        let _ = fs::remove_file(&tmp);
        let parsed: IdentifyCliReportJson =
            serde_json::from_str(&body).expect("deserialize IdentifyCliReportJson");
        assert_eq!(parsed.nucleus, "1H");
        assert!(!parsed.online_requested);
        assert_eq!(
            parsed.matches.first().map(|m| m.entry.bmse_id.as_str()),
            Some("bmse000208")
        );
        assert_eq!(
            parsed.matches.first().map(|m| m.entry.name.as_str()),
            Some("L-(+) Lactic acid")
        );

        // 13C query (L-Alanine bmse000028)
        let args_c13: Vec<String> = vec![
            "identify".into(),
            "--peaks".into(),
            "18.83,53.20,178.56".into(),
            "--nucleus".into(),
            "13c".into(),
            "--tol".into(),
            "0.5".into(),
        ];
        run_metabolomics_cli(&args_c13).expect("identify 13C Alanine succeeds");

        // CSV escaping check (compound names like "N,N-Dimethylglycine" contain commas!)
        assert_eq!(
            escape_delimited_field("N,N-Dimethylglycine", ','),
            "\"N,N-Dimethylglycine\""
        );
        assert_eq!(escape_delimited_field("a\tb\nc", '\t'), "a b c");

        // Missing --peaks must error cleanly
        let bad: Vec<String> = vec!["identify".into()];
        assert!(run_metabolomics_cli(&bad).is_err());

        // Invalid or unbounded --tol (> 2.0 ppm) must error cleanly
        for bad_val in ["-0.01", "0.0", "2.5", "1e300"] {
            let bad_tol: Vec<String> = vec![
                "identify".into(),
                "--peaks".into(),
                "1.318".into(),
                "--tol".into(),
                bad_val.into(),
            ];
            assert!(
                run_metabolomics_cli(&bad_tol).is_err(),
                "expected --tol {bad_val} to be rejected"
            );
        }
    }
}
