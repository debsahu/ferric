//! Frozen real-protein raw time-domain (FID) benchmark runner (`nmrassign real-benchmark`).
//!
//! Provenance: Original clean-room implementation of `docs/specs/05-benchmark.md` and
//! `docs/specs/11-fid-processing.md` sections 5.2–5.3, citing:
//! - Public spectrometer parameter specifications: Bruker JCAMP-DX v5.0 (`acqus`/`acqu2s`/`acqu3s`,
//!   `pulseprogram`, `pdata/1/title`) and Varian/Agilent VnmrJ Data File Reference (`procpar`
//!   ASCII dictionary: `pslabel`, `seqfil`, `comment`, `sw`/`sw1`/`sw2`, `sfrq`/`dfrq`/`dfrq2`).
//! - IUPAC/IUBMB indirect chemical-shift ranges for triple-resonance protein NMR experiments:
//!   Wishart, D. S. et al., *J. Biomol. NMR* **1995**, 6, 135–140.
//! - Strictly unlabelled empirical global reference-offset estimation (`fit_unlabelled_global_offsets`
//!   in `crates/nmr-cli/src/main.rs`) — fitting ONE scalar offset per atom type (`H`, `N`, `CA`,
//!   `CB`, `C`) per entry from unlabelled marginal/2D point clouds without ever inspecting matched
//!   `(predicted, truth)` residue pairs.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use nmr_core::bayes_sa::{condition_peaks_preserving_hsqc_roots, solve_bayes_sa_with_peaks};
use nmr_core::evaluate::to_reference;
use nmr_core::peaks::Peak;
use nmr_core::picker::{
    corroborated_re_pick_3d, pick_hsqc_2d, CorroboratedRepickConfig, PickerConfig, StripBank3D,
};
use nmr_core::priors::{DetailedReferenceOffsets, NucleusOffsetFit, OffsetFitStatus, ShiftOffsets};
#[cfg(test)]
use nmr_core::process::{process_fid_dir_2d, process_fid_dir_3d};
use nmr_core::process::{process_fid_dir_2d_opts, process_fid_dir_3d_opts, AxisRescueAuditRecord};
use nmr_core::qc::{
    apply_truncated_fid_qc_warning, DatasetQcReport, ProteinReliabilityCurve,
    BAYES_SA_CPU_CALIBRATED_RUNS, BAYES_SA_GPU_CALIBRATED_RUNS, STANDARD_CPU_CALIBRATED_RUNS,
};
use nmr_core::solver::ScoreConfig;
use nmr_core::spectrum::{Spectrum2D, Spectrum3D};
use nmr_core::spinsystems::{build_spin_systems, Tolerances};
use nmr_gpu::bayes_sa_gpu::BayesSaGpuSolver;
use nmr_gpu::GpuEngine;

use crate::{
    build_bayes_sa_config_cli, evaluate_prediction_against_truth,
    execute_assignment_workflow_with_options, load_priors, parse_bounded_runs, parse_bounded_steps,
    parse_engine, parse_solver, parse_truth_auto, register_3d_hn_to_hsqc_adaptive_with_audit,
    resolve_solver_engine, version_string, EngineKind, SolverKind, ToleranceOverrides,
};

// ============================================================================
// Manifest Types & Parsing (`data/benchmark/real_manifest.tsv`)
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BenchmarkSplitFilter {
    Dev,
    Heldout,
    All,
}

impl BenchmarkSplitFilter {
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "dev" => Ok(Self::Dev),
            "heldout" | "holdout" | "held-out" => Ok(Self::Heldout),
            "all" => Ok(Self::All),
            other => Err(format!(
                "unknown --split {other:?}; expected dev or heldout (or all)"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Dev => "dev",
            Self::Heldout => "heldout",
            Self::All => "all",
        }
    }

    pub fn matches(self, entry_split: &str) -> bool {
        match self {
            Self::All => true,
            Self::Dev => entry_split.trim().eq_ignore_ascii_case("dev"),
            Self::Heldout => {
                let s = entry_split.trim();
                s.eq_ignore_ascii_case("heldout")
                    || s.eq_ignore_ascii_case("holdout")
                    || s.eq_ignore_ascii_case("held-out")
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RealManifestEntry {
    pub bmrb_id: String,
    pub protein: String,
    pub residues: usize,
    pub field_mhz: f64,
    pub vendor: String,
    pub experiments: Vec<String>,
    pub split: String,
    pub cohort: String,
}

pub fn parse_real_manifest_str(text: &str) -> Result<Vec<RealManifestEntry>, String> {
    let mut out = Vec::new();
    for (line_idx, raw_line) in text.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols
            .first()
            .is_some_and(|c| c.trim().eq_ignore_ascii_case("bmrb_id"))
        {
            continue;
        }
        if cols.len() < 7 {
            return Err(format!(
                "real_manifest.tsv line {} has {} columns (expected 7 or 8: bmrb_id, protein, residues, field_mhz, vendor, experiments, split[, cohort])",
                line_idx + 1,
                cols.len()
            ));
        }
        let bmrb_id = cols[0].trim().to_string();
        let protein = cols[1].trim().to_string();
        let residues: usize = cols[2].trim().parse().map_err(|e| {
            format!(
                "invalid residues {:?} on line {} ({bmrb_id}): {e}",
                cols[2],
                line_idx + 1
            )
        })?;
        let field_first = cols[3].split([',', '/']).next().unwrap_or("").trim();
        let field_mhz: f64 = field_first.parse().map_err(|e| {
            format!(
                "invalid field_mhz {:?} on line {} ({bmrb_id}): {e}",
                cols[3],
                line_idx + 1
            )
        })?;
        let vendor = cols[4].trim().to_string();
        let experiments: Vec<String> = cols[5]
            .split([',', ';', '+'])
            .map(canonicalize_exp_token)
            .filter(|s| !s.is_empty() && s != "*" && s != "AUTO")
            .collect();
        let split = cols[6].trim().to_ascii_lowercase();
        let cohort = cols
            .get(7)
            .map(|c| c.trim().to_ascii_lowercase())
            .filter(|c| !c.is_empty())
            .unwrap_or_else(|| split.clone());
        out.push(RealManifestEntry {
            bmrb_id,
            protein,
            residues,
            field_mhz,
            vendor,
            experiments,
            split,
            cohort,
        });
    }
    Ok(out)
}

fn canonicalize_exp_token(raw: &str) -> String {
    let cleaned: String = raw
        .trim()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_uppercase();
    match cleaned.as_str() {
        "15NHSQC" | "NHSQC" | "HNHSQC" | "HSQC" | "TROSY" | "15NTROSY" => "HSQC".to_string(),
        "HNCOCACB" | "CBCACONH" | "CBCACOCANH" => "CBCACONH".to_string(),
        "CBCANH" | "HNCACB" => "HNCACB".to_string(),
        other => other.to_string(),
    }
}

// ============================================================================
// Linux Case-Safe Filesystem & Local Entry Resolution Helpers
// ============================================================================

/// Deterministically list directory children sorted by exact UTF-8/byte filename.
fn sorted_dir_entries(dir: &Path) -> Vec<(String, PathBuf)> {
    let Ok(rd) = fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut items: Vec<(String, PathBuf)> = rd
        .filter_map(|r| r.ok())
        .map(|e| (e.file_name().to_string_lossy().into_owned(), e.path()))
        .collect();
    items.sort_by(|a, b| a.0.cmp(&b.0));
    items
}

/// Case-insensitive child lookup returning the exact filesystem path (safe on Linux `ext4`).
pub fn find_child_ci(dir: &Path, name_ci: &str) -> Option<PathBuf> {
    for (fname, path) in sorted_dir_entries(dir) {
        if fname.eq_ignore_ascii_case(name_ci) {
            return Some(path);
        }
    }
    None
}

/// Resolve the main git repository checkout root when running inside a git worktree or a crate
/// subdirectory (such as `crates/nmr-cli` during `cargo test`) so gitignored `data/ubiquitin`
/// and `data/private/` are discovered when `--data-root` uses its default.
fn resolve_git_main_checkout_root() -> Option<PathBuf> {
    let start = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    for anc in start.ancestors().take(6) {
        let dot_git = anc.join(".git");
        if dot_git.is_file() {
            let Ok(txt) = fs::read_to_string(&dot_git) else {
                continue;
            };
            for line in txt.lines() {
                if let Some(rest) = line.strip_prefix("gitdir:") {
                    let gd = PathBuf::from(rest.trim());
                    // Worktree gitdir has shape `<main_repo>/.git/worktrees/<worktree_name>`.
                    let Some(worktrees_dir) = gd.parent() else {
                        continue;
                    };
                    if worktrees_dir.file_name().and_then(|s| s.to_str()) == Some("worktrees") {
                        if let Some(main_git_dir) = worktrees_dir.parent() {
                            if main_git_dir.file_name().and_then(|s| s.to_str()) == Some(".git") {
                                return main_git_dir.parent().map(Path::to_path_buf);
                            }
                        }
                    }
                }
            }
        } else if dot_git.is_dir() {
            return if anc != start {
                Some(anc.to_path_buf())
            } else {
                None
            };
        }
    }
    None
}

#[derive(Clone, Debug)]
pub struct ResolvedEntryLocations {
    pub entry_dir: PathBuf,
    pub truth_path: Option<PathBuf>,
}

/// Resolve `(entry_dir, optional_truth_path)` for a manifest entry.
/// Returns `Err(expected_display_path)` when the local dataset directory is absent on disk.
pub fn resolve_local_entry_paths(
    bmrb_id: &str,
    data_root: &Path,
    allow_repo_fallbacks: bool,
) -> Result<ResolvedEntryLocations, String> {
    let id_trim = bmrb_id.trim();
    let mut base_roots: Vec<PathBuf> = Vec::new();
    // When `--data-root` has shape `<repo>/data/private/bmrb` or `<repo>/data/real_bmrb`, include
    // `<repo>/data` so sibling `<repo>/data/ubiquitin` and `<repo>/data/private/bmrb51439_cisd3` resolve cleanly too.
    if data_root
        .file_name()
        .and_then(|s| s.to_str())
        .is_some_and(|s| s.eq_ignore_ascii_case("bmrb"))
    {
        if let Some(priv_parent) = data_root.parent() {
            if priv_parent
                .file_name()
                .and_then(|s| s.to_str())
                .is_some_and(|s| s.eq_ignore_ascii_case("private"))
            {
                if let Some(data_anc) = priv_parent.parent() {
                    base_roots.push(data_anc.to_path_buf());
                }
            }
        }
    } else if data_root
        .file_name()
        .and_then(|s| s.to_str())
        .is_some_and(|s| s.eq_ignore_ascii_case("real_bmrb"))
    {
        if let Some(data_anc) = data_root.parent() {
            base_roots.push(data_anc.to_path_buf());
        }
    }
    if allow_repo_fallbacks {
        base_roots.push(PathBuf::from("data"));
        if let Some(env_data) = env::var_os("NMR_REAL_DATA_DIR") {
            base_roots.push(PathBuf::from(env_data));
        }
        if let Some(main_repo) = resolve_git_main_checkout_root() {
            base_roots.push(main_repo.join("data"));
        }
    }

    if id_trim.eq_ignore_ascii_case("ubiquitin_local") {
        // 1. Check `<data-root>/ubiquitin_local` first (supports custom isolated test roots)
        if let Some(custom_ubq) = find_child_ci(data_root, "ubiquitin_local") {
            if custom_ubq.is_dir() {
                let truth = find_truth_file_in_or_near(&custom_ubq, "ubiquitin_local");
                return Ok(ResolvedEntryLocations {
                    entry_dir: custom_ubq,
                    truth_path: truth,
                });
            }
        }
        for droot in &base_roots {
            if let Some(ubq_dir) = find_child_ci(droot, "ubiquitin") {
                if ubq_dir.is_dir() {
                    let truth = find_child_ci(droot, "ubiquitin_backbone_assignment_191104.txt")
                        .or_else(|| find_truth_file_in_or_near(&ubq_dir, "ubiquitin_local"));
                    return Ok(ResolvedEntryLocations {
                        entry_dir: ubq_dir,
                        truth_path: truth,
                    });
                }
            }
        }
        return Err("data/ubiquitin".to_string());
    }

    let id_digits = id_trim
        .strip_prefix("bmr")
        .or_else(|| id_trim.strip_prefix("BMR"))
        .unwrap_or(id_trim);

    // Special local CISD3 (`51439`) paths + `<data-root>/<id>`
    let mut dir_candidates: Vec<PathBuf> = Vec::new();
    if let Some(c1) = find_child_ci(data_root, id_digits) {
        dir_candidates.push(c1);
    }
    if let Some(c2) = find_child_ci(data_root, &format!("bmr{id_digits}")) {
        dir_candidates.push(c2);
    }

    if !base_roots.is_empty() {
        if id_digits == "51439" {
            for droot in &base_roots {
                if let Some(priv_dir) = find_child_ci(droot, "private") {
                    if let Some(cisd3) = find_child_ci(&priv_dir, "bmrb51439_cisd3") {
                        dir_candidates.push(cisd3);
                    }
                    if let Some(bmrb_sub) = find_child_ci(&priv_dir, "bmrb") {
                        if let Some(c) = find_child_ci(&bmrb_sub, "51439") {
                            dir_candidates.push(c);
                        }
                    }
                }
                if let Some(real_bmrb) = find_child_ci(droot, "real_bmrb") {
                    if let Some(c) = find_child_ci(&real_bmrb, "51439") {
                        dir_candidates.push(c);
                    }
                }
            }
        } else {
            for droot in &base_roots {
                if let Some(priv_dir) = find_child_ci(droot, "private") {
                    if let Some(bmrb_sub) = find_child_ci(&priv_dir, "bmrb") {
                        if let Some(c1) = find_child_ci(&bmrb_sub, id_digits) {
                            dir_candidates.push(c1);
                        }
                        if let Some(c2) = find_child_ci(&bmrb_sub, &format!("bmr{id_digits}")) {
                            dir_candidates.push(c2);
                        }
                    }
                }
                if let Some(real_bmrb) = find_child_ci(droot, "real_bmrb") {
                    if let Some(c1) = find_child_ci(&real_bmrb, id_digits) {
                        dir_candidates.push(c1);
                    }
                    if let Some(c2) = find_child_ci(&real_bmrb, &format!("bmr{id_digits}")) {
                        dir_candidates.push(c2);
                    }
                }
            }
        }
    }

    for cand in dir_candidates {
        if cand.is_dir() {
            let truth = find_truth_file_in_or_near(&cand, id_digits);
            return Ok(ResolvedEntryLocations {
                entry_dir: cand,
                truth_path: truth,
            });
        }
    }

    Err(data_root.join(id_digits).display().to_string())
}

fn find_truth_file_in_or_near(entry_dir: &Path, id_digits: &str) -> Option<PathBuf> {
    let target_3 = format!("bmr{id_digits}_3.str");
    let target_plain = format!("bmr{id_digits}.str");

    if let Some(p) = find_child_ci(entry_dir, &target_3) {
        if p.is_file() {
            return Some(p);
        }
    }
    if let Some(p) = find_child_ci(entry_dir, &target_plain) {
        if p.is_file() {
            return Some(p);
        }
    }
    if let Some(p) = find_child_ci(entry_dir, "ubiquitin_backbone_assignment_191104.txt") {
        if p.is_file() {
            return Some(p);
        }
    }

    // Any `.str` at top level or 1 subdirectory deep
    for (fname, path) in sorted_dir_entries(entry_dir) {
        if path.is_file() && fname.to_ascii_lowercase().ends_with(".str") {
            return Some(path);
        }
    }
    for (_, sub) in sorted_dir_entries(entry_dir) {
        if sub.is_dir() {
            for (fname, path) in sorted_dir_entries(&sub) {
                if path.is_file() && fname.to_ascii_lowercase().ends_with(".str") {
                    return Some(path);
                }
            }
        }
    }

    // Fallback to public `data/benchmark/bmr<id>.str` if present
    let bench_dir = Path::new("data/benchmark");
    if let Some(p) = find_child_ci(bench_dir, &target_plain) {
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

// ============================================================================
// Deterministic Raw FID Experiment Auto-Discovery (Bruker & Varian)
// ============================================================================

#[derive(Clone, Debug)]
pub struct DiscoveredFidExperiment {
    pub exp_type: String,
    pub dir: PathBuf,
    pub rel_path: String,
    pub vendor: String,
    pub dims: usize,
    pub pulse_program: String,
    pub title: String,
    pub h_mhz: f64,
    pub indirect_td_product: usize,
    pub raw_file_bytes: u64,
    pub quad_mode_summary: String,
    pub nus_summary: String,
}

#[derive(Clone, Debug)]
pub struct CandidateFidInspection {
    pub dir: PathBuf,
    pub rel_path: String,
    pub vendor: String,
    pub dims: usize,
    pub pulse_program: String,
    pub title: String,
    pub h_mhz: f64,
    pub direct_nuc: String,
    pub indirect_nucs: Vec<String>,
    pub c_carrier_ppm: Option<f64>,
    pub c_sw_ppm: Option<f64>,
    pub n_sw_ppm: Option<f64>,
    pub indirect_td_product: usize,
    pub raw_file_bytes: u64,
    pub quad_mode_summary: String,
    pub nus_summary: String,
    pub classified_exp: Option<String>,
}

/// Infer canonical experiment type (`HSQC`, `HNCO`, `HNCACO`, `HNCA`, `HNCOCA`, `HNCACB`, `CBCACONH`)
/// from a single Bruker (`acqus`/`ser`) or Varian (`procpar`/`fid`) directory.
pub fn infer_single_fid_experiment(dir: &Path) -> Option<String> {
    let rel = dir.file_name().and_then(|s| s.to_str()).unwrap_or(".");
    inspect_single_raw_fid_dir(dir, rel).and_then(|c| c.classified_exp)
}

/// Inspect `<entry_dir>` recursively (up to depth 10, skipping `pdata` and hidden folders) and
/// deterministically classify raw Bruker (`acqus`/`ser`) and Varian (`procpar`/`fid`) backbone
/// experiments (`HSQC`, `HNCO`, `HNCACO`, `HNCA`, `HNCOCA`, `HNCACB`, `CBCACONH`).
pub fn discover_entry_experiments(
    entry_dir: &Path,
    hint_field_mhz: Option<f64>,
) -> Result<(Vec<DiscoveredFidExperiment>, Vec<CandidateFidInspection>), String> {
    let mut raw_dirs = Vec::new();
    collect_raw_fid_dirs_recursive(entry_dir, 0, 10, &mut raw_dirs);

    let mut inspections: Vec<CandidateFidInspection> = Vec::new();
    for dir in raw_dirs {
        let rel = dir
            .strip_prefix(entry_dir)
            .unwrap_or(&dir)
            .to_string_lossy()
            .replace('\\', "/");
        let rel_clean = if rel.is_empty() { ".".to_string() } else { rel };
        if let Some(insp) = inspect_single_raw_fid_dir(&dir, &rel_clean) {
            inspections.push(insp);
        }
    }

    // Group classified candidates by canonical experiment type
    let mut by_exp: BTreeMap<String, Vec<CandidateFidInspection>> = BTreeMap::new();
    for insp in &inspections {
        if let Some(exp) = &insp.classified_exp {
            if insp.raw_file_bytes > 0 {
                by_exp.entry(exp.clone()).or_default().push(insp.clone());
            }
        }
    }

    // Anchor session parent directories: parent directories of uniquely-identified 3D experiments
    // (or all 3D triple-resonance experiments when unambiguous) plus median 3D `1H` spectrometer MHz.
    let mut anchor_parents = BTreeSet::new();
    let mut triple_mhz_vals: Vec<f64> = Vec::new();
    for (exp, cands) in &by_exp {
        if exp != "HSQC" {
            for c in cands {
                if c.h_mhz > 100.0 {
                    triple_mhz_vals.push(c.h_mhz);
                }
            }
            if cands.len() == 1 {
                if let Some(parent) = Path::new(&cands[0].rel_path).parent() {
                    let ps = parent.to_string_lossy().into_owned();
                    if !ps.is_empty() && ps != "." {
                        anchor_parents.insert(ps);
                    }
                }
            }
        }
    }
    triple_mhz_vals.sort_by(|a, b| a.total_cmp(b));
    let target_mhz = if !triple_mhz_vals.is_empty() {
        Some(triple_mhz_vals[triple_mhz_vals.len() / 2])
    } else {
        hint_field_mhz.filter(|&m| m > 100.0)
    };

    let exp_order = [
        "HSQC", "HNCO", "HNCACO", "HNCA", "HNCOCA", "HNCACB", "CBCACONH",
    ];
    let mut selected = Vec::new();
    for &exp_name in &exp_order {
        let Some(mut cands) = by_exp.remove(exp_name) else {
            continue;
        };
        cands.sort_by(|a, b| {
            let sa = candidate_selection_score(a, &anchor_parents, target_mhz);
            let sb = candidate_selection_score(b, &anchor_parents, target_mhz);
            sb.cmp(&sa).then_with(|| a.rel_path.cmp(&b.rel_path))
        });
        let best = &cands[0];
        // Once a 3D experiment picks its best session directory, add its parent to anchors too
        // so HSQC / companion 3Ds co-locate deterministically.
        if exp_name != "HSQC" {
            if let Some(parent) = Path::new(&best.rel_path).parent() {
                let ps = parent.to_string_lossy().into_owned();
                if !ps.is_empty() && ps != "." {
                    anchor_parents.insert(ps);
                }
            }
        }
        selected.push(DiscoveredFidExperiment {
            exp_type: exp_name.to_string(),
            dir: best.dir.clone(),
            rel_path: best.rel_path.clone(),
            vendor: best.vendor.clone(),
            dims: best.dims,
            pulse_program: best.pulse_program.clone(),
            title: best.title.clone(),
            h_mhz: best.h_mhz,
            indirect_td_product: best.indirect_td_product,
            raw_file_bytes: best.raw_file_bytes,
            quad_mode_summary: best.quad_mode_summary.clone(),
            nus_summary: best.nus_summary.clone(),
        });
    }

    // Re-evaluate 2D HSQC selection once all 3D experiments are finalized (`F6b`):
    // incorporates any newly anchored session parents plus acquisition-series / spectrometer-field /
    // stem-prefix / date-cohort alignment against the selected 3D experiments.
    if let Some(hsqc_pos) = selected.iter().position(|s| s.exp_type == "HSQC") {
        let selected_3ds: Vec<DiscoveredFidExperiment> = selected
            .iter()
            .filter(|s| s.exp_type != "HSQC")
            .cloned()
            .collect();
        let mut hsqc_cands: Vec<CandidateFidInspection> = inspections
            .iter()
            .filter(|c| c.classified_exp.as_deref() == Some("HSQC") && c.raw_file_bytes > 0)
            .cloned()
            .collect();
        if hsqc_cands.len() > 1 {
            hsqc_cands.sort_by(|a, b| {
                let sa = hsqc_series_selection_score(a, &anchor_parents, target_mhz, &selected_3ds);
                let sb = hsqc_series_selection_score(b, &anchor_parents, target_mhz, &selected_3ds);
                sb.cmp(&sa).then_with(|| a.rel_path.cmp(&b.rel_path))
            });
            let best = &hsqc_cands[0];
            selected[hsqc_pos] = DiscoveredFidExperiment {
                exp_type: "HSQC".to_string(),
                dir: best.dir.clone(),
                rel_path: best.rel_path.clone(),
                vendor: best.vendor.clone(),
                dims: best.dims,
                pulse_program: best.pulse_program.clone(),
                title: best.title.clone(),
                h_mhz: best.h_mhz,
                indirect_td_product: best.indirect_td_product,
                raw_file_bytes: best.raw_file_bytes,
                quad_mode_summary: best.quad_mode_summary.clone(),
                nus_summary: best.nus_summary.clone(),
            };
        }
    }

    Ok((selected, inspections))
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct AcquisitionSeriesMeta {
    series_prefix: Option<String>,
    sample_code: Option<String>,
    batch_num: Option<i64>,
    year_month: Option<(i32, u32)>,
}

fn strip_fid_or_mhz_ext(leaf: &str) -> &str {
    if let Some(s) = leaf
        .strip_suffix(".fid")
        .or_else(|| leaf.strip_suffix(".FID"))
    {
        return s;
    }
    if let Some((base, ext)) = leaf.rsplit_once('.') {
        let ext_low = ext.to_ascii_lowercase();
        if let Some(digits) = ext_low.strip_suffix("mhz") {
            if !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit()) {
                return base;
            }
        }
    }
    leaf
}

fn strip_spins_archive_prefix(stem: &str) -> &str {
    // Strip leading SPINS archive identifiers such as `JMA.I.7125_` or `MCB.I.005_` so archive
    // accession numbers never masquerade as sample/series batch tags.
    let Some((head, rest)) = stem.split_once('_') else {
        return stem;
    };
    let parts: Vec<&str> = head.split('.').collect();
    if parts.len() == 3
        && parts[0].len() <= 4
        && parts[0].chars().all(|c| c.is_ascii_alphabetic())
        && !parts[1].is_empty()
        && parts[1]
            .chars()
            .all(|c| matches!(c.to_ascii_uppercase(), 'I' | 'V' | 'X'))
        && !parts[2].is_empty()
        && parts[2].chars().all(|c| c.is_ascii_digit())
    {
        return rest;
    }
    stem
}

fn is_exp_or_acq_modifier_token(tok_low: &str) -> bool {
    if tok_low.is_empty() || tok_low.chars().all(|c| c.is_ascii_digit()) {
        return true;
    }
    if tok_low.ends_with("mhz")
        && tok_low[..tok_low.len() - 3]
            .chars()
            .all(|c| c.is_ascii_digit())
    {
        return true;
    }
    if tok_low.starts_with("sw")
        && tok_low.len() >= 4
        && tok_low.bytes().any(|b| b.is_ascii_digit())
    {
        return true;
    }
    for kw in [
        "hsqc",
        "nhsqc",
        "fhsqc",
        "sofast",
        "trosy",
        "hnco",
        "hncaco",
        "hnca",
        "hncoca",
        "hncacb",
        "cbcanh",
        "cbcaconh",
        "cbcaconnh",
        "hncocacb",
        "noesy",
        "tocsy",
        "cconh",
        "hcconh",
    ] {
        if tok_low.contains(kw) {
            return true;
        }
    }
    matches!(
        tok_low,
        "ghn"
            | "gcbca"
            | "cbca"
            | "cacb"
            | "co"
            | "ca"
            | "nh"
            | "open"
            | "short"
            | "long"
            | "hires"
            | "diamagnetic"
            | "paramagnetic"
            | "d2o"
            | "30min"
            | "150ppm"
            | "nc5"
    )
}

fn extract_year_month_from_tokens(tokens: &[&str]) -> Option<(i32, u32)> {
    // 1. Check trailing / embedded `MM_DD_YY` or `MM_DD_YYYY` triplets (e.g. `10_23_08`, `5_12_09`)
    if tokens.len() >= 3 {
        for idx in (0..=tokens.len() - 3).rev() {
            if let (Ok(mm), Ok(dd), Ok(yy)) = (
                tokens[idx].parse::<u32>(),
                tokens[idx + 1].parse::<u32>(),
                tokens[idx + 2].parse::<i32>(),
            ) {
                if (1..=12).contains(&mm) && (1..=31).contains(&dd) {
                    let full_year = if (0..=49).contains(&yy) {
                        2000 + yy
                    } else if (70..=99).contains(&yy) {
                        1900 + yy
                    } else if (1980..=2050).contains(&yy) {
                        yy
                    } else {
                        continue;
                    };
                    return Some((full_year, mm));
                }
            }
        }
    }
    // 2. Check compact 8-digit `YYYYMMDD` or `MMDDYYYY` or 6-digit `MMDDYY` tokens
    for tok in tokens.iter().rev() {
        if tok.len() == 8 && tok.chars().all(|c| c.is_ascii_digit()) {
            if let (Ok(y1), Ok(m1), Ok(d1)) = (
                tok[0..4].parse::<i32>(),
                tok[4..6].parse::<u32>(),
                tok[6..8].parse::<u32>(),
            ) {
                if (1990..=2050).contains(&y1) && (1..=12).contains(&m1) && (1..=31).contains(&d1) {
                    return Some((y1, m1));
                }
            }
            if let (Ok(m2), Ok(d2), Ok(y2)) = (
                tok[0..2].parse::<u32>(),
                tok[2..4].parse::<u32>(),
                tok[4..8].parse::<i32>(),
            ) {
                if (1990..=2050).contains(&y2) && (1..=12).contains(&m2) && (1..=31).contains(&d2) {
                    return Some((y2, m2));
                }
            }
        }
    }
    None
}

fn extract_acquisition_series_meta(rel_path: &str) -> AcquisitionSeriesMeta {
    let p = Path::new(rel_path);
    let raw_leaf = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
    let leaf_is_expno = !raw_leaf.is_empty() && raw_leaf.chars().all(|c| c.is_ascii_digit());
    let primary = if leaf_is_expno {
        p.parent()
            .and_then(|par| par.file_name())
            .and_then(|s| s.to_str())
            .unwrap_or("")
    } else {
        strip_fid_or_mhz_ext(raw_leaf)
    };
    let cleaned = strip_spins_archive_prefix(primary);
    if cleaned.is_empty() {
        return AcquisitionSeriesMeta::default();
    }

    let raw_tokens: Vec<&str> = cleaned
        .split(['_', '-'])
        .filter(|t| !t.is_empty())
        .collect();
    let year_month = extract_year_month_from_tokens(&raw_tokens);

    let mut sample_code = None;
    let mut batch_num = None;
    if let Some(&first_tok) = raw_tokens.first() {
        if let Some((scode, bstr)) = first_tok.split_once('.') {
            if !scode.is_empty()
                && scode.chars().any(|c| c.is_ascii_alphabetic())
                && !bstr.is_empty()
                && bstr.chars().all(|c| c.is_ascii_digit())
            {
                if let Ok(bnum) = bstr.parse::<i64>() {
                    sample_code = Some(scode.to_ascii_lowercase());
                    batch_num = Some(bnum);
                }
            }
        }
    }

    let mut prefix_toks = Vec::new();
    for &tok in &raw_tokens {
        let low = tok.to_ascii_lowercase();
        if is_exp_or_acq_modifier_token(&low) {
            if prefix_toks.is_empty() && low.chars().all(|c| c.is_ascii_digit()) {
                continue;
            }
            break;
        }
        prefix_toks.push(low);
    }
    let series_prefix = (!prefix_toks.is_empty()).then(|| prefix_toks.join("_"));

    AcquisitionSeriesMeta {
        series_prefix,
        sample_code,
        batch_num,
        year_month,
    }
}

fn hsqc_series_selection_score(
    c: &CandidateFidInspection,
    anchor_parents: &BTreeSet<String>,
    target_mhz: Option<f64>,
    selected_3ds: &[DiscoveredFidExperiment],
) -> i64 {
    if selected_3ds.is_empty() {
        return candidate_selection_score(c, anchor_parents, target_mhz);
    }

    let mut score = 0i64;
    if let Some(parent) = Path::new(&c.rel_path).parent() {
        let ps = parent.to_string_lossy();
        if anchor_parents.contains(ps.as_ref()) {
            score += 10_000;
        }
    }

    let min_3d_mhz_diff = selected_3ds
        .iter()
        .filter(|s| c.h_mhz > 100.0 && s.h_mhz > 100.0)
        .map(|s| (c.h_mhz - s.h_mhz).abs())
        .fold(f64::INFINITY, f64::min);

    if let Some(tmhz) = target_mhz {
        let primary_diff = (c.h_mhz - tmhz).abs();
        if primary_diff <= 5.0 {
            score += 5_000;
        } else if min_3d_mhz_diff <= 5.0 {
            // Matches an actual selected secondary 3D batch's spectrometer field (e.g. `16238`
            // `vcr52.010` at 600 MHz matching `vcr52.011` `HNCA`+`HNCOCA`), beating an orphan
            // third-field HSQC (`vcr52.006` at 750 MHz) while staying below a primary-field HSQC.
            score += 2_500;
        } else {
            let eff_diff = primary_diff.min(min_3d_mhz_diff);
            if eff_diff <= 25.0 {
                score += 1_000;
            } else {
                score -= (eff_diff as i64) * 10;
            }
        }
    }

    let h_meta = extract_acquisition_series_meta(&c.rel_path);
    let mut best_series_bonus = 0i64;
    let mut min_month_gap: Option<i32> = None;
    for s3d in selected_3ds {
        let same_field =
            c.h_mhz <= 100.0 || s3d.h_mhz <= 100.0 || (c.h_mhz - s3d.h_mhz).abs() <= 5.0;
        let s_meta = extract_acquisition_series_meta(&s3d.rel_path);
        if let (Some((hy, hm)), Some((sy, sm))) = (h_meta.year_month, s_meta.year_month) {
            let gap = ((hy * 12 + hm as i32) - (sy * 12 + sm as i32)).abs();
            min_month_gap = Some(min_month_gap.map_or(gap, |prev| prev.min(gap)));
        }
        let mut bonus = 0i64;
        if let (Some(hp), Some(sp)) = (&h_meta.series_prefix, &s_meta.series_prefix) {
            if hp == sp {
                bonus += if same_field { 1_200 } else { 300 };
            }
        }
        if same_field {
            if let (Some(hs), Some(hb), Some(ss), Some(sb)) = (
                &h_meta.sample_code,
                h_meta.batch_num,
                &s_meta.sample_code,
                s_meta.batch_num,
            ) {
                if hs == ss {
                    let bdiff = (hb - sb).abs();
                    if bdiff == 0 {
                        bonus += 1_200;
                    } else if bdiff <= 2 {
                        bonus += 900;
                    }
                }
            }
            if let (Some(h_ym), Some(s_ym)) = (h_meta.year_month, s_meta.year_month) {
                if h_ym == s_ym {
                    bonus += 600;
                }
            }
        }
        if bonus > best_series_bonus {
            best_series_bonus = bonus;
        }
    }
    score += best_series_bonus;

    // Campaign / sample-era alignment across all selected 3Ds:
    // reward co-eval acquisitions (<= 2 months from a 3D), and penalize post-hoc / different-year
    // re-acquisitions (>= 6 months removed from every selected 3D, especially on an orphan field).
    if let Some(gap) = min_month_gap {
        if gap <= 2 {
            score += 900;
        } else if gap >= 6 {
            score -= 6_000;
            if min_3d_mhz_diff > 10.0 {
                score -= 2_500;
            }
        }
    }

    // Wide-band `15N` HSQCs (`SW_15N > 100 ppm`, e.g. `150 ppm` Arg/His+amide scout spectra) have
    // ~4-5x coarser 15N digital resolution than dedicated narrow-band backbone amide HSQCs (`24-45 ppm`),
    // so prefer any same-campaign narrow-band amide HSQC while allowing a co-eval wide-band HSQC
    // (`16238` `vcr52.010` Oct-2008 600 MHz) to beat an out-of-campaign/orphan-field HSQC (`vcr52.006` May-2009 750 MHz).
    if c.n_sw_ppm.is_some_and(|nsw| nsw > 100.0) {
        score -= 10_500;
    }

    let t_low = c.title.to_ascii_lowercase();
    let p_low = c.pulse_program.to_ascii_lowercase();
    let rel_low = c.rel_path.to_ascii_lowercase();
    if t_low.contains("align")
        || t_low.contains("allign")
        || t_low.contains("acrylamid")
        || t_low.contains("polyphosphazine")
        || t_low.contains("bicelle")
        || t_low.contains("phage")
        || t_low.contains("paramag")
        || t_low.contains("d2o")
        || t_low.contains("hdex")
        || rel_low.contains("d2o")
        || rel_low.contains("hdex")
        || rel_low.contains("nh2only")
        || p_low.contains("bird")
    {
        score -= 3_000;
    }
    if c.classified_exp.as_deref() == Some("HSQC") && !p_low.contains("trosy") {
        score += 200;
    }
    score += (c.indirect_td_product.min(4_096)) as i64;
    score
}

fn candidate_selection_score(
    c: &CandidateFidInspection,
    anchor_parents: &BTreeSet<String>,
    target_mhz: Option<f64>,
) -> i64 {
    let mut score = 0i64;
    if let Some(parent) = Path::new(&c.rel_path).parent() {
        let ps = parent.to_string_lossy();
        if anchor_parents.contains(ps.as_ref()) {
            score += 10_000;
        }
    }
    if let Some(tmhz) = target_mhz {
        let diff = (c.h_mhz - tmhz).abs();
        if diff <= 5.0 {
            score += 5_000;
        } else if diff <= 25.0 {
            score += 1_000;
        } else {
            score -= (diff as i64) * 10;
        }
    }
    let t_low = c.title.to_ascii_lowercase();
    let p_low = c.pulse_program.to_ascii_lowercase();
    let rel_low = c.rel_path.to_ascii_lowercase();
    // Deprioritize aligned media / RDC / paramagnetic / BIRD / D2O H-D exchange / NH2-only variants
    // when an isotropic H2O backbone spectrum exists
    if t_low.contains("align")
        || t_low.contains("allign")
        || t_low.contains("acrylamid")
        || t_low.contains("polyphosphazine")
        || t_low.contains("bicelle")
        || t_low.contains("phage")
        || t_low.contains("paramag")
        || t_low.contains("d2o")
        || t_low.contains("hdex")
        || rel_low.contains("d2o")
        || rel_low.contains("hdex")
        || rel_low.contains("nh2only")
        || p_low.contains("bird")
    {
        score -= 3_000;
    }
    // Prefer standard HSQC over TROSY when at moderate field unless TROSY is the only 2D
    if c.classified_exp.as_deref() == Some("HSQC") && !p_low.contains("trosy") {
        score += 200;
    }
    score += (c.indirect_td_product.min(4_096)) as i64;
    score
}

fn collect_raw_fid_dirs_recursive(
    dir: &Path,
    depth: usize,
    max_depth: usize,
    out: &mut Vec<PathBuf>,
) {
    if depth > max_depth {
        return;
    }
    let has_bruker_acqu =
        find_child_ci(dir, "acqus").is_some() || find_child_ci(dir, "acqu").is_some();
    let has_bruker_raw = find_child_ci(dir, "ser").is_some_and(|p| p.is_file())
        || find_child_ci(dir, "fid").is_some_and(|p| p.is_file());
    let has_varian_procpar = find_child_ci(dir, "procpar").is_some_and(|p| p.is_file());
    let has_varian_fid = find_child_ci(dir, "fid").is_some_and(|p| p.is_file());

    if (has_bruker_acqu && has_bruker_raw) || (has_varian_procpar && has_varian_fid) {
        out.push(dir.to_path_buf());
        return;
    }

    for (fname, path) in sorted_dir_entries(dir) {
        if !path.is_dir() {
            continue;
        }
        if fname.starts_with('.')
            || fname.eq_ignore_ascii_case("pdata")
            || fname.eq_ignore_ascii_case("__macosx")
        {
            continue;
        }
        collect_raw_fid_dirs_recursive(&path, depth + 1, max_depth, out);
    }
}

fn inspect_single_raw_fid_dir(dir: &Path, rel_path: &str) -> Option<CandidateFidInspection> {
    if find_child_ci(dir, "acqus").is_some() || find_child_ci(dir, "acqu").is_some() {
        inspect_bruker_dir(dir, rel_path)
    } else if find_child_ci(dir, "procpar").is_some() {
        inspect_varian_dir(dir, rel_path)
    } else {
        None
    }
}

// ============================================================================
// Bruker JCAMP-DX Acquisition Metadata Inspector
// ============================================================================

fn parse_jcamp_map(path: &Path) -> HashMap<String, String> {
    let mut map = HashMap::new();
    let Ok(bytes) = fs::read(path) else {
        return map;
    };
    let text = String::from_utf8_lossy(&bytes);
    let mut pending_array_key: Option<String> = None;
    for raw_line in text.lines() {
        let line = raw_line.trim();
        if let Some(key) = pending_array_key.take() {
            if !line.starts_with("##") {
                map.insert(key, line.trim_matches(['<', '>']).trim().to_string());
                continue;
            }
        }
        if let Some(rest) = line.strip_prefix("##$") {
            if let Some((k, v)) = rest.split_once('=') {
                let key = k.trim().to_string();
                let val = v.trim();
                if val.starts_with('(') && val.contains("..") && !val.ends_with('>') {
                    // JCAMP array header `##$KEY= (0..7)` with values on next line
                    pending_array_key = Some(key);
                } else {
                    map.insert(key, val.trim_matches(['<', '>']).trim().to_string());
                }
            }
        }
    }
    map
}

fn read_bruker_dim_params(
    dir: &Path,
    primary: &str,
    fallback: &str,
) -> Option<HashMap<String, String>> {
    let p1 = find_child_ci(dir, primary);
    let p2 = find_child_ci(dir, fallback);
    if p1.is_none() && p2.is_none() {
        return None;
    }
    let mut merged = p2.map(|p| parse_jcamp_map(&p)).unwrap_or_default();
    if let Some(p) = p1 {
        for (k, v) in parse_jcamp_map(&p) {
            merged.insert(k, v);
        }
    }
    Some(merged)
}

fn normalize_nucleus_code(raw: &str, sfo_mhz: f64, h_mhz: f64) -> String {
    let up: String = raw
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_uppercase();
    match up.as_str() {
        "1H" | "H1" | "PROTON" => return "1H".to_string(),
        "15N" | "N15" => return "15N".to_string(),
        "13C" | "C13" => return "13C".to_string(),
        "2H" | "H2" | "D" => return "2H".to_string(),
        "31P" | "P31" => return "31P".to_string(),
        "19F" | "F19" => return "19F".to_string(),
        _ => {}
    }
    if h_mhz > 100.0 && sfo_mhz > 1.0 {
        let ratio = sfo_mhz / h_mhz;
        if (0.90..=1.10).contains(&ratio) {
            return "1H".to_string();
        }
        if (0.085..=0.125).contains(&ratio) {
            return "15N".to_string();
        }
        if (0.23..=0.27).contains(&ratio) {
            return "13C".to_string();
        }
    } else if sfo_mhz >= 250.0 {
        return "1H".to_string();
    }
    up
}

fn read_bruker_title_and_pp_header(dir: &Path) -> (String, String) {
    let mut title = String::new();
    if let Some(pdata) = find_child_ci(dir, "pdata") {
        if let Some(p1) = find_child_ci(&pdata, "1") {
            if let Some(tp) = find_child_ci(&p1, "title") {
                if let Ok(bytes) = fs::read(tp) {
                    title = String::from_utf8_lossy(&bytes)
                        .lines()
                        .map(str::trim)
                        .filter(|l| !l.is_empty())
                        .collect::<Vec<_>>()
                        .join(" | ");
                }
            }
        }
    }
    let mut pp_comment = String::new();
    if let Some(pp) =
        find_child_ci(dir, "pulseprogram").or_else(|| find_child_ci(dir, "pulseprogram.precomp"))
    {
        if let Ok(bytes) = fs::read(pp) {
            let head = String::from_utf8_lossy(&bytes[..bytes.len().min(2048)]);
            pp_comment = head
                .lines()
                .take(20)
                .filter(|l| l.trim_start().starts_with(';'))
                .collect::<Vec<_>>()
                .join(" ");
        }
    }
    (title, pp_comment)
}

fn inspect_bruker_dir(dir: &Path, rel_path: &str) -> Option<CandidateFidInspection> {
    let d1 = read_bruker_dim_params(dir, "acqus", "acqu")?;
    let d2_opt = read_bruker_dim_params(dir, "acqu2s", "acqu2");
    let d3_opt = read_bruker_dim_params(dir, "acqu3s", "acqu3");

    let ser_file = find_child_ci(dir, "ser").filter(|p| p.is_file());
    let fid_file = find_child_ci(dir, "fid").filter(|p| p.is_file());
    let raw_file_bytes = ser_file
        .as_ref()
        .or(fid_file.as_ref())
        .and_then(|p| fs::metadata(p).ok())
        .map(|m| m.len())
        .unwrap_or(0);

    let get_usize = |m: &HashMap<String, String>, k: &str| -> usize {
        m.get(k).and_then(|v| v.parse::<usize>().ok()).unwrap_or(0)
    };
    let get_f64 = |m: &HashMap<String, String>, k: &str| -> f64 {
        m.get(k).and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0)
    };
    let get_i32 = |m: &HashMap<String, String>, k: &str| -> Option<i32> {
        m.get(k).and_then(|v| v.parse::<i32>().ok())
    };

    let pulprog = d1.get("PULPROG").cloned().unwrap_or_default();
    let sfo1 = get_f64(&d1, "SFO1");
    let bf1 = get_f64(&d1, "BF1");
    let h_mhz_guess = if sfo1 >= 200.0 {
        sfo1
    } else if bf1 >= 200.0 {
        bf1
    } else {
        sfo1
    };
    let direct_nuc = normalize_nucleus_code(
        d1.get("NUC1").map(String::as_str).unwrap_or(""),
        sfo1,
        h_mhz_guess,
    );

    let td2 = d2_opt.as_ref().map(|m| get_usize(m, "TD")).unwrap_or(0);
    let nustd2 = d2_opt.as_ref().map(|m| get_usize(m, "NusTD")).unwrap_or(0);
    let td3 = d3_opt.as_ref().map(|m| get_usize(m, "TD")).unwrap_or(0);
    let nustd3 = d3_opt.as_ref().map(|m| get_usize(m, "NusTD")).unwrap_or(0);

    let dims = if ser_file.is_some() && d3_opt.is_some() && (td3 > 1 || nustd3 > 1) {
        3
    } else if ser_file.is_some() && d2_opt.is_some() && (td2 > 1 || nustd2 > 1) {
        2
    } else {
        1
    };

    let h_mhz = if direct_nuc == "1H" {
        sfo1.max(bf1)
    } else {
        0.0
    };
    let mut indirect_nucs = Vec::new();
    let mut c_carrier_ppm = None;
    let mut c_sw_ppm = None;
    let mut n_sw_ppm = None;

    let mut inspect_ind = |m: &HashMap<String, String>| {
        let sfo = get_f64(m, "SFO1");
        let bf = get_f64(m, "BF1");
        let sw_ppm = {
            let sw = get_f64(m, "SW");
            if sw > 0.0 {
                sw
            } else if sfo > 0.0 {
                get_f64(m, "SW_h") / sfo
            } else {
                0.0
            }
        };
        let nuc =
            normalize_nucleus_code(m.get("NUC1").map(String::as_str).unwrap_or(""), sfo, h_mhz);
        if nuc == "13C" {
            c_sw_ppm = Some(sw_ppm);
            if bf > 0.0 && sfo > 0.0 {
                c_carrier_ppm = Some(((sfo - bf) / bf) * 1.0e6);
            }
        } else if nuc == "15N" {
            n_sw_ppm = Some(sw_ppm);
        }
        indirect_nucs.push(nuc);
    };

    if dims >= 2 {
        if let Some(m2) = &d2_opt {
            inspect_ind(m2);
        }
    }
    if dims >= 3 {
        if let Some(m3) = &d3_opt {
            inspect_ind(m3);
        }
    }

    let aq_mod = get_i32(&d1, "AQ_mod").unwrap_or(-1);
    let fn2 = d2_opt.as_ref().and_then(|m| get_i32(m, "FnMODE"));
    let fn3 = d3_opt.as_ref().and_then(|m| get_i32(m, "FnMODE"));
    let quad_mode_summary = match dims {
        3 => format!(
            "AQ_mod={aq_mod},F2:FnMODE={},F1:FnMODE={}",
            fn2.map(|v| v.to_string()).unwrap_or_else(|| "?".into()),
            fn3.map(|v| v.to_string()).unwrap_or_else(|| "?".into())
        ),
        2 => format!(
            "AQ_mod={aq_mod},F1:FnMODE={}",
            fn2.map(|v| v.to_string()).unwrap_or_else(|| "?".into())
        ),
        _ => format!("AQ_mod={aq_mod}"),
    };

    let fn_type = get_i32(&d1, "FnTYPE").unwrap_or(0);
    let has_nuslist = find_child_ci(dir, "nuslist").is_some_and(|p| p.is_file());
    let nus_summary = if fn_type == 2 || (fn_type > 0 && has_nuslist) {
        format!(
            "NUS(FnTYPE={fn_type},nuslist={})",
            if has_nuslist { "yes" } else { "missing" }
        )
    } else {
        format!("uniform(FnTYPE={fn_type})")
    };

    let eff_td2 = td2.max(nustd2).max(1);
    let eff_td3 = if dims == 3 { td3.max(nustd3).max(1) } else { 1 };
    let indirect_td_product = if dims >= 2 { eff_td2 * eff_td3 } else { 0 };

    let (title, pp_comment) = read_bruker_title_and_pp_header(dir);
    let dir_leaf = dir.file_name().and_then(|s| s.to_str()).unwrap_or("");
    let classified_exp = classify_backbone_experiment(
        dims,
        &direct_nuc,
        &indirect_nucs,
        &pulprog,
        &pp_comment,
        &title,
        dir_leaf,
        c_carrier_ppm,
        c_sw_ppm,
        n_sw_ppm,
    );

    Some(CandidateFidInspection {
        dir: dir.to_path_buf(),
        rel_path: rel_path.to_string(),
        vendor: "Bruker".to_string(),
        dims,
        pulse_program: pulprog,
        title,
        h_mhz,
        direct_nuc,
        indirect_nucs,
        c_carrier_ppm,
        c_sw_ppm,
        n_sw_ppm,
        indirect_td_product,
        raw_file_bytes,
        quad_mode_summary,
        nus_summary,
        classified_exp,
    })
}

// ============================================================================
// Varian / Agilent `procpar` Acquisition Metadata Inspector
// ============================================================================

#[derive(Default)]
struct VarianProcparSummary {
    strings: HashMap<String, Vec<String>>,
    reals: HashMap<String, Vec<f64>>,
}

impl VarianProcparSummary {
    fn first_str(&self, key: &str) -> Option<&str> {
        self.strings
            .get(key)
            .and_then(|v| v.first())
            .map(String::as_str)
    }

    fn first_f64(&self, key: &str) -> Option<f64> {
        self.reals.get(key).and_then(|v| v.first()).copied()
    }
}

fn parse_varian_procpar_summary(path: &Path) -> VarianProcparSummary {
    let mut out = VarianProcparSummary::default();
    let Ok(bytes) = fs::read(path) else {
        return out;
    };
    let text = String::from_utf8_lossy(&bytes);
    let lines: Vec<&str> = text.lines().collect();
    let mut idx = 0usize;

    while idx < lines.len() {
        let hdr = lines[idx].trim();
        if hdr.is_empty() {
            idx += 1;
            continue;
        }
        let hdr_toks: Vec<&str> = hdr.split_whitespace().collect();
        // Standard VnmrJ `procpar` parameter header has 11 tokens: `name subtype basictype ...`
        if hdr_toks.len() >= 11
            && hdr_toks[1].parse::<i32>().is_ok()
            && matches!(hdr_toks[2], "0" | "1" | "2")
        {
            let name = hdr_toks[0].to_string();
            let basictype = hdr_toks[2];
            idx += 1;
            if idx >= lines.len() {
                break;
            }
            let val_line = lines[idx].trim();
            if basictype == "2" {
                // String array: `<count> "first" ...` followed by `count - 1` quoted lines
                let (cnt_str, rest) = val_line
                    .split_once(char::is_whitespace)
                    .unwrap_or((val_line, ""));
                let count = cnt_str.parse::<usize>().unwrap_or(1);
                let mut vals = Vec::with_capacity(count);
                vals.push(unquote_procpar_str(rest.trim()));
                idx += 1;
                for _ in 1..count {
                    if idx < lines.len() {
                        vals.push(unquote_procpar_str(lines[idx].trim()));
                        idx += 1;
                    }
                }
                out.strings.insert(name, vals);
            } else if basictype == "1" {
                let toks: Vec<&str> = val_line.split_whitespace().collect();
                let count = toks
                    .first()
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(0);
                let mut nums: Vec<f64> = toks
                    .iter()
                    .skip(1)
                    .filter_map(|s| s.parse::<f64>().ok())
                    .collect();
                idx += 1;
                while nums.len() < count && idx < lines.len() {
                    let more: Vec<f64> = lines[idx]
                        .split_whitespace()
                        .filter_map(|s| s.parse::<f64>().ok())
                        .collect();
                    if more.is_empty() {
                        break;
                    }
                    nums.extend(more);
                    idx += 1;
                }
                out.reals.insert(name, nums);
            } else {
                idx += 1;
            }
            // Skip trailing enum specification line (`0` or `N "opt1" "opt2" ...`)
            if idx < lines.len() {
                idx += 1;
            }
        } else {
            idx += 1;
        }
    }
    out
}

fn unquote_procpar_str(s: &str) -> String {
    let t = s.trim();
    if t.len() >= 2 && t.starts_with('"') && t.ends_with('"') {
        t[1..t.len() - 1].to_string()
    } else {
        t.trim_matches('"').to_string()
    }
}

fn inspect_varian_dir(dir: &Path, rel_path: &str) -> Option<CandidateFidInspection> {
    let procpar_path = find_child_ci(dir, "procpar")?;
    let fid_path = find_child_ci(dir, "fid").filter(|p| p.is_file());
    let raw_file_bytes = fid_path
        .as_ref()
        .and_then(|p| fs::metadata(p).ok())
        .map(|m| m.len())
        .unwrap_or(0);

    let pp = parse_varian_procpar_summary(&procpar_path);
    let pslabel = pp.first_str("pslabel").unwrap_or("");
    let seqfil = pp.first_str("seqfil").unwrap_or("");
    let pulse_program = if !pslabel.is_empty() {
        pslabel.to_string()
    } else {
        seqfil.to_string()
    };
    let comment = pp.first_str("comment").unwrap_or("").to_string();

    let sfrq = pp.first_f64("sfrq").unwrap_or(0.0);
    let dfrq = pp.first_f64("dfrq").unwrap_or(0.0);
    let dfrq2 = pp.first_f64("dfrq2").unwrap_or(0.0);
    let dfrq3 = pp.first_f64("dfrq3").unwrap_or(0.0);

    let tn = pp.first_str("tn").unwrap_or("");
    let dn = pp.first_str("dn").unwrap_or("");
    let dn2 = pp.first_str("dn2").unwrap_or("");
    let dn3 = pp.first_str("dn3").unwrap_or("");

    let direct_nuc = normalize_nucleus_code(tn, sfrq, sfrq);
    let h_mhz = if direct_nuc == "1H" { sfrq } else { 0.0 };

    let ni = pp.first_f64("ni").unwrap_or(0.0) as usize;
    let ni2 = pp.first_f64("ni2").unwrap_or(0.0) as usize;
    let ni3 = pp.first_f64("ni3").unwrap_or(0.0) as usize;

    let mut active_ind_slots: Vec<usize> = Vec::new();
    if ni > 1 {
        active_ind_slots.push(1);
    }
    if ni2 > 1 {
        active_ind_slots.push(2);
    }
    if ni3 > 1 && active_ind_slots.len() < 2 {
        active_ind_slots.push(3);
    }
    let dims = 1 + active_ind_slots.len();

    let axis_str = pp.first_str("axis").unwrap_or("");
    let axis_chars: Vec<char> = axis_str.chars().collect();

    let mut indirect_nucs = Vec::new();
    let mut c_carrier_ppm = None;
    let mut c_sw_ppm = None;
    let mut n_sw_ppm = None;

    for &slot in &active_ind_slots {
        let sw_key = match slot {
            1 => "sw1",
            2 => "sw2",
            _ => "sw3",
        };
        let reffrq_key = match slot {
            1 => "reffrq1",
            2 => "reffrq2",
            _ => "reffrq3",
        };
        let sw_hz = pp.first_f64(sw_key).unwrap_or(0.0);
        let reffrq = pp.first_f64(reffrq_key).unwrap_or(0.0);
        let axis_ch = axis_chars.get(slot).copied();

        // Determine channel (`d` -> dn/dfrq, `2` -> dn2/dfrq2, `3` -> dn3/dfrq3).
        // Prefer hardware synthesizer frequency `chan_frq` (`dfrq`/`dfrq2`) over `reffrq*`,
        // which in legacy VnmrJ `procpar` files is occasionally left at the 1H frequency.
        let (chan_nuc_raw, chan_frq) = match axis_ch {
            Some('d') | Some('1') => (dn, dfrq),
            Some('2') => (dn2, dfrq2),
            Some('3') => (dn3, dfrq3),
            _ => {
                // BioPack standard: in 2D `gNhsqc`, `ni` (`sw1 ~ 1200..3500 Hz`) is `N15` (`dn2`) when `dn=C13`,
                // whereas in 3D (`ghncacb`, `ghnco`), `ni` (`sw1`) is `C13` (`dn`) and `ni2` (`sw2`) is `N15` (`dn2`).
                if dims == 2 {
                    let n_cand = normalize_nucleus_code(dn2, dfrq2, h_mhz);
                    let d_cand = normalize_nucleus_code(dn, dfrq, h_mhz);
                    let pp_low = pulse_program.to_ascii_lowercase();
                    let explicit_nhsqc = pp_low.contains("nhsqc")
                        || pp_low.contains("fhsqc")
                        || pp_low.contains("trosy");
                    if d_cand == "15N" {
                        (dn, dfrq)
                    } else if n_cand == "15N"
                        && (sw_hz <= 4800.0 || (explicit_nhsqc && sw_hz <= 15_000.0))
                    {
                        (dn2, dfrq2)
                    } else {
                        (dn, dfrq)
                    }
                } else if slot == 1 {
                    (dn, dfrq)
                } else {
                    (dn2, dfrq2)
                }
            }
        };

        let nuc = normalize_nucleus_code(
            chan_nuc_raw,
            if chan_frq > 1.0 { chan_frq } else { reffrq },
            h_mhz,
        );
        let obs = if chan_frq > 1.0 {
            chan_frq
        } else if reffrq > 1.0 {
            reffrq
        } else if nuc == "13C" {
            h_mhz * 0.25144953
        } else if nuc == "15N" {
            h_mhz * 0.10132912
        } else {
            0.0
        };
        let sw_ppm = if obs > 0.0 { sw_hz / obs } else { 0.0 };
        if nuc == "13C" {
            c_sw_ppm = Some(sw_ppm);
            if h_mhz > 100.0 && chan_frq > 1.0 {
                let nu0_h = h_mhz / (1.0 + 4.766e-6);
                let nu0_c = nu0_h * 0.25144953;
                c_carrier_ppm = Some(((chan_frq - nu0_c) / nu0_c) * 1.0e6);
            }
        } else if nuc == "15N" {
            n_sw_ppm = Some(sw_ppm);
        }
        indirect_nucs.push(nuc);
    }

    let fmt_phase = |key: &str| -> String {
        match pp.reals.get(key) {
            Some(v) if !v.is_empty() => v
                .iter()
                .map(|x| format!("{}", *x as i32))
                .collect::<Vec<_>>()
                .join(","),
            _ => "none".to_string(),
        }
    };
    let fcoef_info = if pp.first_str("fcoef").is_some_and(|s| !s.is_empty())
        || pp.first_str("f1coef").is_some_and(|s| !s.is_empty())
        || pp.first_str("f2coef").is_some_and(|s| !s.is_empty())
    {
        "+fcoef"
    } else {
        ""
    };
    let quad_mode_summary = match dims {
        3 => format!(
            "phase=[{}],phase2=[{}]{fcoef_info}",
            fmt_phase("phase"),
            fmt_phase("phase2")
        ),
        2 => format!("phase=[{}]{fcoef_info}", fmt_phase("phase")),
        _ => "1D".to_string(),
    };

    let sampling = pp.first_str("sampling").unwrap_or("");
    let sparse = pp.first_str("sparse").unwrap_or("");
    let has_sch = find_child_ci(dir, "sampling.sch").is_some_and(|p| p.is_file())
        || find_child_ci(dir, "nuslist").is_some_and(|p| p.is_file());
    let explicit_uniform = sampling.eq_ignore_ascii_case("s") && !sparse.eq_ignore_ascii_case("y");
    let nus_summary = if sampling.eq_ignore_ascii_case("sparse")
        || sparse.eq_ignore_ascii_case("y")
        || (!explicit_uniform && sampling.is_empty() && has_sch)
    {
        format!(
            "NUS(sampling={sampling:?},sparse={sparse:?},sch={})",
            if has_sch { "yes" } else { "missing" }
        )
    } else {
        "uniform".to_string()
    };

    let indirect_td_product = active_ind_slots
        .iter()
        .map(|&s| match s {
            1 => ni.max(1),
            2 => ni2.max(1),
            _ => ni3.max(1),
        })
        .product::<usize>();

    let dir_leaf = dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .trim_end_matches(".fid");
    let seq_both = if !pslabel.is_empty() && !seqfil.is_empty() && pslabel != seqfil {
        format!("{pslabel} {seqfil}")
    } else {
        pulse_program.clone()
    };

    let classified_exp = classify_backbone_experiment(
        dims,
        &direct_nuc,
        &indirect_nucs,
        &seq_both,
        "",
        &comment,
        dir_leaf,
        c_carrier_ppm,
        c_sw_ppm,
        n_sw_ppm,
    );

    Some(CandidateFidInspection {
        dir: dir.to_path_buf(),
        rel_path: rel_path.to_string(),
        vendor: "Varian".to_string(),
        dims,
        pulse_program,
        title: comment,
        h_mhz,
        direct_nuc,
        indirect_nucs,
        c_carrier_ppm,
        c_sw_ppm,
        n_sw_ppm,
        indirect_td_product,
        raw_file_bytes,
        quad_mode_summary,
        nus_summary,
        classified_exp,
    })
}

// ============================================================================
// Pure Deterministic Experiment Classifier (`HSQC` & 3D Triple-Resonance)
// ============================================================================

#[allow(clippy::too_many_arguments)]
fn classify_backbone_experiment(
    dims: usize,
    direct_nuc: &str,
    indirect_nucs: &[String],
    primary_prog: &str,
    pp_header_comment: &str,
    title_or_comment: &str,
    dir_leaf_fallback: &str,
    c_carrier_ppm: Option<f64>,
    c_sw_ppm: Option<f64>,
    n_sw_ppm: Option<f64>,
) -> Option<String> {
    if direct_nuc != "1H" {
        return None;
    }

    let norm_alnum = |s: &str| -> String {
        s.chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .collect::<String>()
            .to_ascii_lowercase()
    };
    let p_norm = norm_alnum(primary_prog);
    let t_norm = norm_alnum(title_or_comment);
    let d_norm = norm_alnum(dir_leaf_fallback);
    let cmt_norm = norm_alnum(pp_header_comment);

    if dims == 2 {
        if indirect_nucs.first().map(String::as_str) != Some("15N") {
            return None;
        }
        if let Some(nsw) = n_sw_ppm {
            // Standard backbone amide HSQC uses ~20..45 ppm 15N SW; wide-band Arg/His+amide
            // `swN150p` / `150ppm` HSQCs use up to ~160 ppm 15N SW.
            if nsw > 0.0 && !(5.0..=165.0).contains(&nsw) {
                return None;
            }
        }
        // Reject heteronuclear relaxation / T1 / T2 / NOE / CPMG / CEST / TOCSY / HMBC / CON 2D experiments
        for bad in [
            "hsqct1",
            "hsqct2",
            "hsqcnoe",
            "hsqnoef",
            "invit1",
            "invit2",
            "invinoe",
            "trosyt1",
            "trosyt2",
            "trosynoe",
            "gnhsqct1",
            "gnhsqct2",
            "gnhsqcnoe",
            "cpmg",
            "t1rho",
            "hetnoe",
            "relax",
            "cest",
            "tocsy",
            "dipsi",
            "roesy",
            "cosy",
            "hmbc",
            "ipap",
        ] {
            if p_norm.contains(bad) || t_norm.contains(bad) {
                return None;
            }
        }
        if title_or_comment.to_ascii_lowercase().contains("(1h) noe")
            || title_or_comment.to_ascii_lowercase().contains("15n-noe")
        {
            return None;
        }
        for src in [&p_norm, &cmt_norm, &t_norm, &d_norm] {
            if src.contains("hsqc")
                || src.contains("trosy")
                || src.contains("sfhmqc")
                || src.contains("sofast")
            {
                return Some("HSQC".to_string());
            }
        }
        return None;
    }

    if dims == 3 {
        let has_15n = indirect_nucs.iter().any(|n| n == "15N");
        let has_13c = indirect_nucs.iter().any(|n| n == "13C");
        if !has_15n || !has_13c {
            return None;
        }

        // Reject sidechain / NOESY / TOCSY 3D sequences first
        for sidechain_or_noe in [
            "hccconh",
            "ccconh",
            "hcch",
            "noesy",
            "tocsy",
            "dipsi",
            "hbhaconh",
            "hbhacbcanh",
            "hnha",
            "hnhb",
            "hncannh",
            "hncocannh",
        ] {
            if p_norm.contains(sidechain_or_noe) {
                return None;
            }
        }

        let match_3d_token = |tok: &str| -> Option<&'static str> {
            if tok.contains("cbcaconh")
                || tok.contains("cbcacocanh")
                || tok.contains("cbcacogp")
                || tok.contains("cbcacoca")
                || tok.contains("hncocacb")
                || tok.contains("cbcaconnh")
            {
                Some("CBCACONH")
            } else if tok.contains("hncacb") || tok.contains("cbcanh") {
                Some("HNCACB")
            } else if tok.contains("hncoca") {
                Some("HNCOCA")
            } else if tok.contains("hncaco") {
                Some("HNCACO")
            } else if tok.contains("hnca") {
                Some("HNCA")
            } else if tok.contains("hnco") {
                Some("HNCO")
            } else {
                None
            }
        };

        let cand = match_3d_token(&p_norm)
            .or_else(|| match_3d_token(&cmt_norm))
            .or_else(|| match_3d_token(&t_norm))
            .or_else(|| match_3d_token(&d_norm))?;

        // Physics sanity check against 13C carrier / spectral width when present
        if let Some(cc) = c_carrier_ppm {
            match cand {
                // Carbonyl C' carrier ~ 170..182 ppm (allow generous 135..215 ppm unless folded)
                "HNCO" | "HNCACO"
                    if !(135.0..=215.0).contains(&cc) && c_sw_ppm.is_some_and(|sw| sw > 42.0) =>
                {
                    return None;
                }
                // Aliphatic CA / CACB carrier ~ 35..60 ppm (never at 176 ppm carbonyl!)
                "HNCACB" | "CBCACONH" | "HNCA" | "HNCOCA" if cc > 130.0 => {
                    return None;
                }
                _ => {}
            }
        }
        return Some(cand.to_string());
    }

    None
}

// ============================================================================
// Benchmark Execution, Macro/Micro Aggregation & JSON Reporting
// ============================================================================

#[derive(Clone, Debug)]
pub struct EntryFailureDiagnostic {
    pub stage: String,
    pub experiment: Option<String>,
    pub vendor: String,
    pub quad_mode: String,
    pub nus_status: String,
    pub reason: String,
}

#[derive(Clone, Debug)]
pub struct EntrySuccessMetrics {
    pub seeds: usize,
    pub discovered_experiments: Vec<(String, String)>,
    pub axis_rescue_audits: Vec<AxisRescueAuditRecord>,
    pub n_hsqc_roots: usize,
    pub n_total_peaks: usize,
    pub assignable_per_seed: usize,
    pub total_assignable: usize,
    pub total_assigned: usize,
    pub total_correct: usize,
    pub total_hc_assigned: usize,
    pub total_hc_correct: usize,
    pub mean_accuracy: f64,
    pub mean_recall: f64,
    pub mean_hc_precision: f64,
    pub mean_hc_recall: f64,
    pub global_offsets: ShiftOffsets,
    pub detailed_offsets: DetailedReferenceOffsets,
    pub prior_detailed_offsets: DetailedReferenceOffsets,
    pub qc_report: DatasetQcReport,
    pub reliability: ProteinReliabilityCurve,
    pub predictions: Vec<(f64, bool)>,
    pub elapsed_secs: f64,
}

#[derive(Clone, Debug)]
pub enum EntryBenchmarkOutcome {
    SkippedMissingData { expected_path: String },
    Failed(EntryFailureDiagnostic),
    Succeeded(Box<EntrySuccessMetrics>),
}

#[derive(Clone, Debug)]
pub struct EntryBenchmarkRecord {
    pub manifest_entry: RealManifestEntry,
    pub outcome: EntryBenchmarkOutcome,
}

#[derive(Clone, Debug)]
pub struct RealBenchmarkSummary {
    pub split: String,
    pub solver_label: String,
    pub seeds: usize,
    pub hc_threshold: f64,
    pub records: Vec<EntryBenchmarkRecord>,
    pub n_hc_active_proteins: usize,
    pub n_hc_abstained_proteins: usize,
    pub worst_protein_hc_precision: Option<(String, String, f64)>,
    pub median_protein_hc_precision: Option<f64>,
    pub mean_protein_ece: f64,
    pub mean_protein_ece_10: f64,
    pub mean_protein_debiased_ece_10: f64,
    pub mean_protein_ece_10_bernoulli_floor: f64,
    pub mean_protein_ece_12: f64,
    pub pooled_reliability: ProteinReliabilityCurve,
    pub macro_accuracy: f64,
    pub macro_recall: f64,
    pub macro_hc_precision: f64,
    pub macro_hc_recall: f64,
    pub micro_assignable: usize,
    pub micro_assigned: usize,
    pub micro_correct: usize,
    pub micro_hc_assigned: usize,
    pub micro_hc_correct: usize,
    pub micro_accuracy: f64,
    pub micro_recall: f64,
    pub micro_hc_precision: f64,
    pub micro_hc_recall: f64,
}

#[derive(Clone, Debug)]
pub struct RealBenchmarkConfig {
    pub manifest_path: PathBuf,
    pub split: BenchmarkSplitFilter,
    pub cohort_opt: Option<String>,
    pub discover_only: bool,
    pub solver: SolverKind,
    pub engine_opt: Option<EngineKind>,
    pub seeds: usize,
    pub base_seed: u64,
    pub data_root: PathBuf,
    pub allow_repo_fallbacks: bool,
    pub priors_path: Option<String>,
    pub runs_opt: Option<usize>,
    pub steps_opt: Option<usize>,
    pub hc_threshold: f64,
    pub no_crop: bool,
    pub lp: bool,
    pub no_axis_rescue: bool,
    pub allow_truncated_fid: bool,
    pub json_out: Option<PathBuf>,
    pub verbose: bool,
}

// ============================================================================
// Header-Only Discovery & Compatibility Preflight (`--discover-only`, `F5`)
// ============================================================================

/// Structured per-entry status emitted by `nmrassign real-benchmark --discover-only` (`F5`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiscoverOnlyStatus {
    Ok,
    HeaderIncompatible,
    MissingExperiments,
    Skipped,
}

impl DiscoverOnlyStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ok => "OK",
            Self::HeaderIncompatible => "HEADER_INCOMPATIBLE",
            Self::MissingExperiments => "MISSING_EXPERIMENTS",
            Self::Skipped => "SKIPPED",
        }
    }
}

#[derive(Clone, Debug)]
pub struct DiscoverOnlyExpCheck {
    pub exp_type: String,
    pub rel_path: String,
    pub vendor: String,
    pub dims: usize,
    pub h_mhz: f64,
    pub required: bool,
    pub header_ok: bool,
    pub error: Option<String>,
}

#[derive(Clone, Debug)]
pub struct DiscoverOnlyEntryRecord {
    pub manifest_entry: RealManifestEntry,
    pub status: DiscoverOnlyStatus,
    pub entry_dir: Option<PathBuf>,
    pub truth_path: Option<PathBuf>,
    pub discovered_experiments: Vec<DiscoverOnlyExpCheck>,
    pub missing_experiments: Vec<String>,
    pub reason: Option<String>,
}

#[derive(Clone, Debug)]
pub struct DiscoverOnlySummary {
    pub split: String,
    pub cohort_opt: Option<String>,
    pub records: Vec<DiscoverOnlyEntryRecord>,
    pub n_ok: usize,
    pub n_header_incompatible: usize,
    pub n_missing_experiments: usize,
    pub n_skipped: usize,
}

const MAX_HEADER_TEXT_BYTES: u64 = 4 * 1024 * 1024;

fn read_header_text_capped(path: &Path) -> Result<String, String> {
    let meta = fs::metadata(path).map_err(|e| format!("failed to stat {}: {e}", path.display()))?;
    if meta.len() > MAX_HEADER_TEXT_BYTES {
        return Err(format!(
            "header file {} ({} bytes) exceeds {} byte safety cap",
            path.display(),
            meta.len(),
            MAX_HEADER_TEXT_BYTES
        ));
    }
    let bytes = fs::read(path).map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

fn verify_bruker_pulseprogram_containment_header_only(dir: &Path) -> Result<(), String> {
    let Ok(canon_dir) = fs::canonicalize(dir) else {
        return Ok(());
    };
    for name in ["pulseprogram", "pulseprogram.precomp"] {
        if let Some(pp_path) = find_child_ci(dir, name) {
            let canon_pp = fs::canonicalize(&pp_path).map_err(|e| {
                format!(
                    "failed to resolve Bruker pulseprogram {}: {e}",
                    pp_path.display()
                )
            })?;
            if !canon_pp.starts_with(&canon_dir) {
                return Err(format!(
                    "refusing Bruker pulseprogram symlink escaping acquisition directory {}: {} -> {}",
                    dir.display(),
                    pp_path.display(),
                    canon_pp.display()
                ));
            }
        }
    }
    Ok(())
}

fn read_optional_bruker_proc_header_only(
    dir: &Path,
    proc_name: &str,
) -> Option<HashMap<String, String>> {
    let is_processed = |pm: &HashMap<String, String>| -> bool {
        pm.get("FT_mod")
            .and_then(|s| s.parse::<i32>().ok())
            .map(|ft_mod| ft_mod != 0)
            .unwrap_or(true)
    };
    if let Some(pdata_dir) = find_child_ci(dir, "pdata") {
        if let Some(d1) = find_child_ci(&pdata_dir, "1") {
            if let Some(p1) = find_child_ci(&d1, proc_name).filter(|p| p.is_file()) {
                let m = parse_jcamp_map(&p1);
                if !m.is_empty() && is_processed(&m) {
                    return Some(m);
                }
                return None;
            }
        }
    }
    if let Some(p_direct) = find_child_ci(dir, proc_name).filter(|p| p.is_file()) {
        let m = parse_jcamp_map(&p_direct);
        if !m.is_empty() && is_processed(&m) {
            return Some(m);
        }
    }
    None
}

fn find_bruker_nuslist_header_only(dir: &Path, acqus: &HashMap<String, String>) -> Option<PathBuf> {
    if let Some(nl_name) = acqus.get("NUSLIST") {
        let trimmed = nl_name.trim();
        if !trimmed.is_empty()
            && !trimmed.eq_ignore_ascii_case("automatic")
            && !trimmed.eq_ignore_ascii_case("none")
            && !trimmed.contains("..")
            && !trimmed.starts_with('/')
            && !trimmed.starts_with('\\')
        {
            if let Some(cand) = find_child_ci(dir, trimmed).filter(|p| p.is_file()) {
                if fs::metadata(&cand).map(|m| m.len()).unwrap_or(0) > 0 {
                    return Some(cand);
                }
            }
        }
    }
    if let Some(default_nl) = find_child_ci(dir, "nuslist").filter(|p| p.is_file()) {
        if fs::metadata(&default_nl).map(|m| m.len()).unwrap_or(0) > 0 {
            return Some(default_nl);
        }
    }
    None
}

fn determine_bruker_ser_stride_header_only(
    file_len: usize,
    n_traces: usize,
    td_words: usize,
    word_bytes: usize,
    allow_trailing_blocks: bool,
    allow_truncated_fid: bool,
    dir: &Path,
) -> Result<(usize, usize), String> {
    let raw_trace_bytes = td_words * word_bytes;
    let stride_1024 = raw_trace_bytes.div_ceil(1024) * 1024;
    let stride_256w = td_words.div_ceil(256) * 256 * word_bytes;

    if file_len == n_traces * stride_1024 {
        Ok((stride_1024, n_traces))
    } else if file_len == n_traces * stride_256w {
        Ok((stride_256w, n_traces))
    } else if allow_trailing_blocks
        && stride_1024 > 0
        && file_len > n_traces * stride_1024
        && file_len.is_multiple_of(stride_1024)
    {
        Ok((stride_1024, n_traces))
    } else if allow_truncated_fid && file_len < n_traces * stride_1024 {
        let stride = if stride_256w != stride_1024
            && stride_256w > 0
            && file_len >= stride_256w
            && file_len.is_multiple_of(stride_256w)
            && !file_len.is_multiple_of(stride_1024.max(1))
        {
            stride_256w
        } else if (raw_trace_bytes < stride_1024
            && file_len < stride_1024
            && file_len >= raw_trace_bytes)
            || (raw_trace_bytes > 0
                && raw_trace_bytes != stride_1024
                && !file_len.is_multiple_of(stride_1024.max(1))
                && file_len.is_multiple_of(raw_trace_bytes)
                && (file_len / raw_trace_bytes) < n_traces)
        {
            raw_trace_bytes
        } else {
            stride_1024
        };
        let observed = file_len.checked_div(stride).unwrap_or(0);
        if observed == 0 {
            return Err(format!(
                "Bruker ser in {} is truncated before completing 1 trace ({file_len} B < {raw_trace_bytes} B)",
                dir.display()
            ));
        }
        Ok((stride, observed.min(n_traces)))
    } else if file_len < n_traces * raw_trace_bytes {
        Err(format!(
            "Bruker ser in {} is smaller than expected trace count ({file_len} B < {n_traces} traces x {stride_1024} B)",
            dir.display()
        ))
    } else {
        Err(format!(
            "Bruker ser size mismatch in {}: got {file_len} B for {n_traces} traces (expected {} B with 1024-B block stride or {} B with 256-word block stride)",
            dir.display(),
            n_traces * stride_1024,
            n_traces * stride_256w
        ))
    }
}

fn check_bruker_indirect_axis_header_only(
    dir: &Path,
    acqu_ind: &HashMap<String, String>,
    proc_ind: Option<&HashMap<String, String>>,
    ind_phys: Option<&nmr_core::pulprog::IndirectDimPhysics>,
    td_full: usize,
    dim_label: &str,
) -> Result<(nmr_core::fid::QuadMode, usize), String> {
    use nmr_core::fid::QuadMode;
    if td_full <= 1 {
        return Err(format!(
            "missing or degenerate indirect TD={td_full} on Bruker {dim_label} in {}",
            dir.display()
        ));
    }
    let fnmode_opt: Option<i32> = match acqu_ind.get("FnMODE") {
        Some(raw) => Some(raw.trim().parse::<i32>().map_err(|e| {
            format!(
                "invalid FnMODE {raw:?} on Bruker {dim_label} in {}: {e}",
                dir.display()
            )
        })?),
        None => None,
    };
    let manual_loop_quad = ind_phys
        .filter(|ip| {
            matches!(
                ip.quad_source,
                nmr_core::pulprog::QuadSource::SynthesizedManualLoop { .. }
            )
        })
        .and_then(|ip| ip.quad_mode.to_fid_quad_mode());
    let quad_mode = match fnmode_opt {
        Some(1) => manual_loop_quad.unwrap_or(QuadMode::Qf),
        Some(4) => QuadMode::States,
        Some(5) => QuadMode::StatesTppi,
        Some(6) => QuadMode::EchoAntiecho,
        None | Some(0) => {
            if let Some(qm) = manual_loop_quad {
                qm
            } else if let Some(mc2_raw) = proc_ind.and_then(|p| p.get("MC2")) {
                let mc2 = mc2_raw.trim().parse::<i32>().map_err(|e| {
                    format!(
                        "invalid ##$MC2 {mc2_raw:?} on Bruker {dim_label} in {}: {e}",
                        dir.display()
                    )
                })?;
                match mc2 {
                    0 => QuadMode::Qf,
                    3 => QuadMode::States,
                    4 => QuadMode::StatesTppi,
                    5 => QuadMode::EchoAntiecho,
                    other => {
                        return Err(format!(
                            "Bruker {dim_label} FnMODE is Undefined (0) and proc*s MC2={other} is unsupported in {}",
                            dir.display()
                        ));
                    }
                }
            } else {
                return Err(format!(
                    "Bruker {dim_label} FnMODE is Undefined (0) and neither pulseprogram nor proc*s MC2 specifies quadrature in {}",
                    dir.display()
                ));
            }
        }
        Some(other) => {
            return Err(format!(
                "unsupported Bruker {dim_label} FnMODE={other} in {} (expected 1=QF, 4=States, 5=States-TPPI, 6=Echo-Antiecho)",
                dir.display()
            ));
        }
    };

    let complex_points = if quad_mode == QuadMode::Qf {
        td_full
    } else {
        if td_full < 2 {
            return Err(format!(
                "Bruker hypercomplex {dim_label} TD={td_full} < 2 in {}",
                dir.display()
            ));
        }
        td_full / 2
    };
    let sfo1 = acqu_ind
        .get("SFO1")
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.0);
    let bf1 = acqu_ind
        .get("BF1")
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.0);
    if sfo1 <= 0.0 && bf1 <= 0.0 {
        return Err(format!(
            "missing positive SFO1/BF1 on Bruker {dim_label} in {}",
            dir.display()
        ));
    }
    let sw = acqu_ind
        .get("SW")
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.0);
    let sw_h = acqu_ind
        .get("SW_h")
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.0);
    if sw <= 0.0 && sw_h <= 0.0 {
        return Err(format!(
            "missing positive SW/SW_h on Bruker {dim_label} in {}",
            dir.display()
        ));
    }
    Ok((quad_mode, complex_points))
}

#[allow(clippy::too_many_lines)]
fn check_bruker_header_compatibility(
    dir: &Path,
    expected_dims: usize,
    allow_truncated_fid: bool,
) -> Result<(), String> {
    verify_bruker_pulseprogram_containment_header_only(dir)?;
    let pulprog_interp = match nmr_core::pulprog::interpret_bruker_dir(dir) {
        Ok(pi) => Some(pi),
        Err(nmr_core::pulprog::PulprogError::UnsafeIncludePath(msg)) => {
            return Err(format!(
                "unsafe Bruker pulseprogram #include in {}: {msg}",
                dir.display()
            ));
        }
        Err(_) => None,
    };

    let acqus = read_bruker_dim_params(dir, "acqus", "acqu").ok_or_else(|| {
        format!(
            "missing Bruker parameter file acqus (or acqu) in {}",
            dir.display()
        )
    })?;
    let acqu2s = read_bruker_dim_params(dir, "acqu2s", "acqu2").ok_or_else(|| {
        format!(
            "missing Bruker indirect parameter file acqu2s (or acqu2) in {}",
            dir.display()
        )
    })?;
    let acqu3s_opt = if expected_dims >= 3 {
        Some(
            read_bruker_dim_params(dir, "acqu3s", "acqu3").ok_or_else(|| {
                format!(
                    "missing Bruker 3D indirect parameter file acqu3s (or acqu3) in {}",
                    dir.display()
                )
            })?,
        )
    } else {
        None
    };

    // Reject GFT / reduced-dimensionality or unsupported Bruker FnTYPE modes
    let mut is_fntype_nus = false;
    let mut check_maps: Vec<&HashMap<String, String>> = vec![&acqus, &acqu2s];
    if let Some(ref a3) = acqu3s_opt {
        check_maps.push(a3);
    }
    for m in &check_maps {
        for gft_key in ["RDphase", "RD2phase"] {
            if m.get(gft_key).is_some_and(|v| !v.trim().is_empty()) {
                return Err(format!(
                    "unsupported Bruker GFT / reduced-dimensionality `{gft_key}` header in {}",
                    dir.display()
                ));
            }
        }
        if let Some(ft_raw) = m.get("FnTYPE") {
            let ft: i32 = ft_raw.trim().parse().map_err(|e| {
                format!("invalid Bruker FnTYPE={ft_raw:?} in {}: {e}", dir.display())
            })?;
            match ft {
                0 => {}
                2 => is_fntype_nus = true,
                other => {
                    return Err(format!(
                        "unsupported Bruker non-traditional sampling mode in {}: FnTYPE={other} (expected 0=uniform or 2=NUS)",
                        dir.display()
                    ));
                }
            }
        }
    }

    // Direct axis checks (`acqus`)
    let td_x: usize = acqus
        .get("TD")
        .ok_or_else(|| format!("missing ##$TD in Bruker acqus ({})", dir.display()))?
        .trim()
        .parse()
        .map_err(|e| format!("invalid TD in Bruker acqus ({}): {e}", dir.display()))?;
    if td_x < 2 {
        return Err(format!(
            "invalid Bruker direct TD={td_x} in {}",
            dir.display()
        ));
    }
    let dtypa: i32 = acqus
        .get("DTYPA")
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(0);
    if !matches!(dtypa, 0..=2) {
        return Err(format!(
            "unsupported Bruker DTYPA={dtypa} in {} (expected 0=int32, 1=float32, 2=float64)",
            dir.display()
        ));
    }
    let aq_mod: i32 = acqus
        .get("AQ_mod")
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(3);
    if !matches!(aq_mod, 1 | 3 | 4) {
        return Err(format!(
            "unsupported Bruker direct quadrature mode AQ_mod={aq_mod} in {} (expected 1=QF, 3=DQD, 4=QSIM)",
            dir.display()
        ));
    }
    let sfo1_x = acqus
        .get("SFO1")
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.0);
    let bf1_x = acqus
        .get("BF1")
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.0);
    if sfo1_x <= 0.0 && bf1_x <= 0.0 {
        return Err(format!(
            "missing positive SFO1/BF1 in Bruker acqus ({})",
            dir.display()
        ));
    }
    let sw_x = acqus
        .get("SW")
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.0);
    let sw_h_x = acqus
        .get("SW_h")
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.0);
    if sw_x <= 0.0 && sw_h_x <= 0.0 {
        return Err(format!(
            "missing positive SW/SW_h in Bruker acqus ({})",
            dir.display()
        ));
    }
    let procs_raw = read_optional_bruker_proc_header_only(dir, "procs");
    let proc2s_raw = read_optional_bruker_proc_header_only(dir, "proc2s");
    let proc3s_raw = read_optional_bruker_proc_header_only(dir, "proc3s");
    let (procs, proc2s, proc3s) =
        if expected_dims >= 3 && proc2s_raw.is_some() && proc3s_raw.is_none() {
            (None, None, None)
        } else {
            (procs_raw, proc2s_raw, proc3s_raw)
        };

    let pknl_enabled = procs
        .as_ref()
        .and_then(|p| p.get("PKNL"))
        .map(|v| !v.eq_ignore_ascii_case("no") && v != "0")
        .unwrap_or(true);
    let grpdly_raw = acqus
        .get("GRPDLY")
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(-1.0);
    if pknl_enabled && !(grpdly_raw > 0.0 && grpdly_raw.is_finite()) {
        let dspfvs: i32 = acqus
            .get("DSPFVS")
            .and_then(|v| v.parse::<i32>().ok())
            .unwrap_or(0);
        let decim: i32 = acqus
            .get("DECIM")
            .and_then(|v| v.parse::<i32>().ok())
            .unwrap_or(1);
        let _ = nmr_core::fid::bruker_dsp_group_delay(dspfvs, decim)?;
    }

    let ser_path = find_child_ci(dir, "ser")
        .filter(|p| p.is_file())
        .ok_or_else(|| format!("missing Bruker ser binary file in {}", dir.display()))?;
    let ser_len = fs::metadata(&ser_path)
        .map_err(|e| format!("failed to stat {}: {e}", ser_path.display()))?
        .len() as usize;
    let word_bytes = if dtypa == 2 { 8 } else { 4 };
    let raw_stride_1024 = (td_x * word_bytes).div_ceil(1024) * 1024;

    let parse_td_indirect_entry = |idx: usize| -> Option<usize> {
        acqus
            .get("TD_INDIRECT")
            .and_then(|s| s.split_whitespace().nth(idx))
            .and_then(|tok| tok.parse::<usize>().ok())
            .filter(|&v| v > 0)
    };

    let sched_file = find_bruker_nuslist_header_only(dir, &acqus);

    let td_acq_y: usize = acqu2s
        .get("TD")
        .ok_or_else(|| format!("missing ##$TD in Bruker acqu2s ({})", dir.display()))?
        .trim()
        .parse()
        .map_err(|e| format!("invalid TD in Bruker acqu2s ({}): {e}", dir.display()))?;

    if expected_dims == 2 {
        let nus_td_y_opt = acqu2s
            .get("NusTD")
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .or_else(|| parse_td_indirect_entry(1));
        let implicit_nus = !is_fntype_nus
            && sched_file.is_some()
            && (nus_td_y_opt.is_some_and(|ntd| ntd > td_acq_y)
                || ser_len < td_acq_y * raw_stride_1024);
        let is_nus = is_fntype_nus || implicit_nus;
        let td_full_y = if is_nus {
            nus_td_y_opt
                .or_else(|| (ser_len < td_acq_y * raw_stride_1024 && td_acq_y > 1).then_some(td_acq_y))
                .ok_or_else(|| {
                    format!(
                        "Bruker 2D NUS dataset in {} is missing NusTD (and TD_INDIRECT[1]) full-grid size on F1 (acqu2s)",
                        dir.display()
                    )
                })?
        } else {
            td_acq_y
        };
        let ind_phys_y = pulprog_interp
            .as_ref()
            .and_then(|pi| pi.indirect_dims.first());
        let (qm_y, n1) = check_bruker_indirect_axis_header_only(
            dir,
            &acqu2s,
            proc2s.as_ref(),
            ind_phys_y,
            td_full_y,
            "F1 (acqu2s)",
        )?;
        let q1_mult = if qm_y == nmr_core::fid::QuadMode::Qf {
            1
        } else {
            2
        };
        if is_nus {
            let sp = sched_file.ok_or_else(|| {
                format!(
                    "Bruker 2D NUS dataset in {} is missing a non-empty nuslist schedule file",
                    dir.display()
                )
            })?;
            let sched_txt = read_header_text_capped(&sp)?;
            let decoded = nmr_core::pulprog::decode_nuslist_str(
                &sched_txt,
                &[n1],
                nmr_core::pulprog::AcquisitionOrder::Dim2D,
            )
            .map_err(|e| format!("invalid Bruker 2D NUS schedule {}: {e}", sp.display()))?;
            let n_acq_traces = decoded.points.len() * q1_mult;
            if td_acq_y != n_acq_traces && td_acq_y != td_full_y {
                return Err(format!(
                    "Bruker 2D NUS trace count mismatch in {}: acqu2s.TD={td_acq_y}, but nuslist has {} points ({n_acq_traces} traces)",
                    dir.display(),
                    decoded.points.len()
                ));
            }
            let _ = determine_bruker_ser_stride_header_only(
                ser_len,
                n_acq_traces,
                td_x,
                word_bytes,
                false,
                allow_truncated_fid,
                dir,
            )?;
        } else {
            let _ = determine_bruker_ser_stride_header_only(
                ser_len,
                td_acq_y,
                td_x,
                word_bytes,
                true,
                allow_truncated_fid,
                dir,
            )?;
        }
        return Ok(());
    }

    // 3D Bruker header checks
    let Some(ref acqu3s) = acqu3s_opt else {
        return Err(format!("missing Bruker acqu3s in {}", dir.display()));
    };
    let header_aqseq: i32 = match acqus.get("AQSEQ") {
        Some(v) => v.trim().parse::<i32>().unwrap_or(-1),
        None => 0,
    };
    if !matches!(header_aqseq, 0 | 1) {
        return Err(format!(
            "unsupported Bruker 3D AQSEQ={header_aqseq} in {} (expected 0 for 3-2-1 or 1 for 3-1-2)",
            dir.display()
        ));
    }
    let aqseq: i32 = match pulprog_interp.as_ref().map(|pi| pi.aqseq) {
        Some(nmr_core::pulprog::AcquisitionOrder::Aqseq312) => 1,
        Some(nmr_core::pulprog::AcquisitionOrder::Aqseq321) => 0,
        _ => header_aqseq,
    };
    let td_acq_z: usize = acqu3s
        .get("TD")
        .ok_or_else(|| format!("missing ##$TD in Bruker acqu3s ({})", dir.display()))?
        .trim()
        .parse()
        .map_err(|e| format!("invalid TD in Bruker acqu3s ({}): {e}", dir.display()))?;
    let total_acq_param = td_acq_y * td_acq_z;
    let nus_td_y_opt = acqu2s
        .get("NusTD")
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .or_else(|| parse_td_indirect_entry(2));
    let nus_td_z_opt = acqu3s
        .get("NusTD")
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .or_else(|| parse_td_indirect_entry(1));
    let full_grid_est = nus_td_y_opt.unwrap_or(td_acq_y) * nus_td_z_opt.unwrap_or(td_acq_z);
    let implicit_nus = !is_fntype_nus
        && sched_file.is_some()
        && (full_grid_est > total_acq_param || ser_len < full_grid_est * raw_stride_1024);
    let is_nus = is_fntype_nus || implicit_nus;

    let (td_full_y, td_full_z) = if is_nus {
        let unflat = td_acq_y > 1 && td_acq_z > 1;
        let fy = nus_td_y_opt
            .or_else(|| unflat.then_some(td_acq_y))
            .ok_or_else(|| {
                format!(
                    "Bruker 3D NUS dataset in {} is missing NusTD (and TD_INDIRECT[2]) full-grid size on F2 (acqu2s)",
                    dir.display()
                )
            })?;
        let fz = nus_td_z_opt
            .or_else(|| unflat.then_some(td_acq_z))
            .ok_or_else(|| {
                format!(
                    "Bruker 3D NUS dataset in {} is missing NusTD (and TD_INDIRECT[1]) full-grid size on F1 (acqu3s)",
                    dir.display()
                )
            })?;
        (fy, fz)
    } else {
        (td_acq_y, td_acq_z)
    };

    let ind_phys_y = pulprog_interp
        .as_ref()
        .and_then(|pi| pi.indirect_dims.first());
    let ind_phys_z = pulprog_interp
        .as_ref()
        .and_then(|pi| pi.indirect_dims.get(1));
    let (qm_y, n1) = check_bruker_indirect_axis_header_only(
        dir,
        &acqu2s,
        proc2s.as_ref(),
        ind_phys_y,
        td_full_y,
        "F2 (acqu2s)",
    )?;
    let (qm_z, n2) = check_bruker_indirect_axis_header_only(
        dir,
        acqu3s,
        proc3s.as_ref(),
        ind_phys_z,
        td_full_z,
        "F1 (acqu3s)",
    )?;
    let q1_mult = if qm_y == nmr_core::fid::QuadMode::Qf {
        1
    } else {
        2
    };
    let q2_mult = if qm_z == nmr_core::fid::QuadMode::Qf {
        1
    } else {
        2
    };
    if is_nus {
        let sp = sched_file.ok_or_else(|| {
            format!(
                "Bruker 3D NUS dataset in {} is missing a non-empty nuslist schedule file",
                dir.display()
            )
        })?;
        let sched_txt = read_header_text_capped(&sp)?;
        let sched_order = if aqseq == 0 {
            nmr_core::pulprog::AcquisitionOrder::Aqseq321
        } else {
            nmr_core::pulprog::AcquisitionOrder::Aqseq312
        };
        let decoded = nmr_core::pulprog::decode_nuslist_str(&sched_txt, &[n1, n2], sched_order)
            .map_err(|e| format!("invalid Bruker 3D NUS schedule {}: {e}", sp.display()))?;
        let traces_per_point = q1_mult * q2_mult;
        let n_acq_traces = decoded.points.len() * traces_per_point;
        let full_grid_traces = td_full_y * td_full_z;
        if total_acq_param != n_acq_traces && total_acq_param != full_grid_traces {
            return Err(format!(
                "Bruker 3D NUS trace count mismatch in {}: acqu2s.TD*acqu3s.TD={total_acq_param}, but nuslist has {} points ({n_acq_traces} traces)",
                dir.display(),
                decoded.points.len()
            ));
        }
        let _ = determine_bruker_ser_stride_header_only(
            ser_len,
            n_acq_traces,
            td_x,
            word_bytes,
            false,
            allow_truncated_fid,
            dir,
        )?;
    } else {
        let n_traces = n1 * q1_mult * n2 * q2_mult;
        let _ = determine_bruker_ser_stride_header_only(
            ser_len,
            n_traces,
            td_x,
            word_bytes,
            true,
            allow_truncated_fid,
            dir,
        )?;
    }
    Ok(())
}

fn read_varian_fid_header_only(
    path: &Path,
    allow_truncated_fid: bool,
) -> Result<(usize, usize), String> {
    const VARIAN_S_DATA: u16 = 0x0001;
    const VARIAN_S_SPEC: u16 = 0x0002;
    const VARIAN_S_32: u16 = 0x0004;
    const VARIAN_S_FLOAT: u16 = 0x0008;

    let file_len = fs::metadata(path)
        .map_err(|e| format!("failed to stat {}: {e}", path.display()))?
        .len() as usize;
    if file_len < 32 {
        return Err(format!(
            "Varian fid file {} too short ({file_len} bytes < 32-byte datafilehead)",
            path.display()
        ));
    }

    use std::io::Read as _;
    let mut f =
        fs::File::open(path).map_err(|e| format!("failed to open {}: {e}", path.display()))?;
    let read_len = file_len.min(36);
    let mut hdr = [0u8; 36];
    f.read_exact(&mut hdr[..read_len])
        .map_err(|e| format!("failed to read Varian header {}: {e}", path.display()))?;

    let nblocks = i32::from_be_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]).max(0) as usize;
    let ntraces = i32::from_be_bytes([hdr[4], hdr[5], hdr[6], hdr[7]]).max(0) as usize;
    let np = i32::from_be_bytes([hdr[8], hdr[9], hdr[10], hdr[11]]).max(0) as usize;
    let ebytes = i32::from_be_bytes([hdr[12], hdr[13], hdr[14], hdr[15]]).max(0) as usize;
    let bbytes = i32::from_be_bytes([hdr[20], hdr[21], hdr[22], hdr[23]]).max(0) as usize;
    let status = u16::from_be_bytes([hdr[26], hdr[27]]);
    let nbheaders = i32::from_be_bytes([hdr[28], hdr[29], hdr[30], hdr[31]]).max(0) as usize;

    if (status & VARIAN_S_DATA) == 0 {
        return Err(format!(
            "Varian fid {} has S_DATA=0 (empty acquisition buffer)",
            path.display()
        ));
    }
    if (status & VARIAN_S_SPEC) != 0 {
        return Err(format!(
            "Varian file {} has S_SPEC=1 (already Fourier-transformed spectrum, expected raw FID)",
            path.display()
        ));
    }
    if nblocks == 0 || ntraces == 0 {
        return Err(format!(
            "invalid Varian datafilehead nblocks={nblocks}, ntraces={ntraces} in {}",
            path.display()
        ));
    }
    if np < 2 || !np.is_multiple_of(2) {
        return Err(format!("invalid Varian np={np} in {}", path.display()));
    }
    let blk_status = if nbheaders >= 1 && read_len >= 36 {
        u16::from_be_bytes([hdr[34], hdr[35]])
    } else {
        status
    };
    let is_float = (blk_status & VARIAN_S_FLOAT) != 0 || (status & VARIAN_S_FLOAT) != 0;
    let is_32 = (blk_status & VARIAN_S_32) != 0 || (status & VARIAN_S_32) != 0 || ebytes == 4;
    if !((is_float && ebytes == 4) || (is_32 && ebytes == 4) || ebytes == 2) {
        return Err(format!(
            "unsupported Varian sample format (status=0x{status:04x}, ebytes={ebytes}) in {}",
            path.display()
        ));
    }

    let hdr_bytes_per_block = nbheaders * 28;
    let trace_bytes = np * ebytes;
    let min_bbytes = hdr_bytes_per_block + ntraces * trace_bytes;
    if bbytes < min_bbytes || trace_bytes == 0 {
        return Err(format!(
            "corrupt Varian bbytes={bbytes} (< {min_bbytes} required for {ntraces} traces x {trace_bytes} B) in {}",
            path.display()
        ));
    }
    let total_needed = 32 + nblocks * bbytes;
    if file_len < total_needed && !allow_truncated_fid {
        return Err(format!(
            "Varian fid {} truncated: header declares {total_needed} B ({nblocks} blocks x {bbytes} B), got {file_len} B",
            path.display()
        ));
    }
    let observed_traces = if file_len >= total_needed {
        nblocks * ntraces
    } else {
        let body = file_len.saturating_sub(32);
        let full_blocks = (body / bbytes).min(nblocks);
        let rem = body % bbytes;
        let partial_traces = if full_blocks < nblocks && rem > hdr_bytes_per_block {
            ((rem - hdr_bytes_per_block) / trace_bytes).min(ntraces)
        } else {
            0
        };
        full_blocks * ntraces + partial_traces
    };
    if observed_traces == 0 {
        return Err(format!(
            "Varian fid {} truncated before completing 1 trace: header declares {total_needed} B, got {file_len} B",
            path.display()
        ));
    }
    Ok((np, observed_traces))
}

fn parse_varian_fcoef_header_only(raw: Option<&str>) -> Result<Option<[f64; 8]>, String> {
    let Some(s) = raw.map(str::trim).filter(|s| !s.is_empty()) else {
        return Ok(None);
    };
    let vals: Result<Vec<f64>, _> = s.split_whitespace().map(str::parse::<f64>).collect();
    let vals = vals.map_err(|e| format!("invalid Varian fcoef '{s}': {e}"))?;
    if vals.len() != 8 {
        return Err(format!(
            "invalid Varian fcoef '{s}': expected 8 floats, got {}",
            vals.len()
        ));
    }
    let mut out = [0.0; 8];
    out.copy_from_slice(&vals);
    Ok(Some(out))
}

fn parse_varian_phase_header_only(
    pp: &VarianProcparSummary,
    key: &str,
    fcoef: Option<[f64; 8]>,
) -> Result<nmr_core::fid::QuadMode, String> {
    use nmr_core::fid::QuadMode;
    let arr = pp.reals.get(key).filter(|v| !v.is_empty()).ok_or_else(|| {
        format!("missing `{key}` quadrature array in Varian procpar (never guessing)")
    })?;
    if arr.len() == 1 && (arr[0] - 1.0).abs() < 1e-6 {
        return Ok(QuadMode::Qf);
    }
    if arr.len() == 2 && (arr[0] - 1.0).abs() < 1e-6 && (arr[1] - 2.0).abs() < 1e-6 {
        if let Some(c) = fcoef {
            let is_ea =
                c[1].abs() > 0.0 || c[3].abs() > 0.0 || c[5].abs() > 0.0 || c[7].abs() > 0.0;
            return Ok(if is_ea {
                QuadMode::EchoAntiecho
            } else {
                QuadMode::States
            });
        }
        return Ok(QuadMode::States);
    }
    Err(format!(
        "unsupported Varian `{key}` quadrature array {:?} (expected [1, 2] hypercomplex or [1] QF)",
        arr
    ))
}

fn find_varian_schedule_header_only(dir: &Path, pp: &VarianProcparSummary) -> Option<PathBuf> {
    for key in ["ni_file", "sparse_file", "schedfile"] {
        if let Some(s) = pp.first_str(key).map(str::trim).filter(|s| !s.is_empty()) {
            if let Some(leaf) = Path::new(s).file_name().and_then(|n| n.to_str()) {
                if let Some(cand) = find_child_ci(dir, leaf).filter(|p| p.is_file()) {
                    if fs::metadata(&cand).map(|m| m.len()).unwrap_or(0) > 0 {
                        return Some(cand);
                    }
                }
            }
        }
    }
    for name in ["sampling.sch", "nuslist"] {
        if let Some(cand) = find_child_ci(dir, name).filter(|p| p.is_file()) {
            if fs::metadata(&cand).map(|m| m.len()).unwrap_or(0) > 0 {
                return Some(cand);
            }
        }
    }
    None
}

#[allow(clippy::too_many_lines)]
fn check_varian_header_compatibility(
    dir: &Path,
    expected_dims: usize,
    allow_truncated_fid: bool,
) -> Result<(), String> {
    use nmr_core::fid::QuadMode;

    let procpar_path = find_child_ci(dir, "procpar")
        .filter(|p| p.is_file())
        .ok_or_else(|| format!("missing Varian procpar in {}", dir.display()))?;
    let _ = read_header_text_capped(&procpar_path)?;
    let pp = parse_varian_procpar_summary(&procpar_path);
    if pp.strings.is_empty() && pp.reals.is_empty() {
        return Err(format!(
            "empty or malformed Varian procpar in {}",
            procpar_path.display()
        ));
    }

    // Explicit GFT / reduced-dimensionality header check (`15089` `array="phase2,phase,RDphase"` / `RDphase=[1,2]`)
    for gft_key in ["RDphase", "RD2phase"] {
        if let Some(rd) = pp.reals.get(gft_key) {
            if rd.len() > 1 {
                return Err(format!(
                    "unsupported Varian GFT / reduced-dimensionality `{gft_key}`={rd:?} (array={:?}) in {}",
                    pp.first_str("array").unwrap_or(""),
                    dir.display()
                ));
            }
        }
    }
    if let Some(raw_arr) = pp.first_str("array") {
        let arr_low = raw_arr.to_ascii_lowercase();
        if arr_low.contains("rdphase") || arr_low.contains("rd2phase") || arr_low.contains("gft") {
            return Err(format!(
                "unsupported Varian GFT / reduced-dimensionality `array`='{raw_arr}' in {}",
                dir.display()
            ));
        }
    }

    let sfrq = pp
        .first_f64("sfrq")
        .filter(|&v| v > 0.0 && v.is_finite())
        .ok_or_else(|| {
            format!(
                "missing positive `sfrq` in Varian procpar ({})",
                dir.display()
            )
        })?;
    let sw = pp
        .first_f64("sw")
        .filter(|&v| v > 0.0 && v.is_finite())
        .ok_or_else(|| {
            format!(
                "missing positive `sw` in Varian procpar ({})",
                dir.display()
            )
        })?;
    let _ = (sfrq, sw);

    let fid_path = find_child_ci(dir, "origfid")
        .filter(|p| p.is_file())
        .and_then(|orig| {
            let fid_meta = find_child_ci(dir, "fid").and_then(|f| fs::metadata(f).ok())?;
            let orig_meta = fs::metadata(&orig).ok()?;
            (orig_meta.len() == fid_meta.len() && orig_meta.len() >= 32).then_some(orig)
        })
        .or_else(|| find_child_ci(dir, "fid").filter(|p| p.is_file()))
        .ok_or_else(|| format!("missing Varian fid binary in {}", dir.display()))?;
    let (_np, observed_traces) = read_varian_fid_header_only(&fid_path, allow_truncated_fid)?;

    let ni_raw = pp.first_f64("ni").unwrap_or(0.0).round().max(0.0) as usize;
    let nimax = pp
        .first_f64("nimax")
        .map(|v| v.round().max(0.0) as usize)
        .filter(|&v| v > 1);
    let ni2_raw = pp.first_f64("ni2").unwrap_or(0.0).round().max(0.0) as usize;
    let ni2max = pp
        .first_f64("ni2max")
        .map(|v| v.round().max(0.0) as usize)
        .filter(|&v| v > 1);

    let explicit_nus = pp
        .first_str("sampling")
        .is_some_and(|s| s.eq_ignore_ascii_case("sparse"))
        || pp
            .first_str("sparse")
            .is_some_and(|s| s.eq_ignore_ascii_case("y"));
    let explicit_uniform = !explicit_nus
        && pp
            .first_str("sampling")
            .is_some_and(|s| s.eq_ignore_ascii_case("s") || s.eq_ignore_ascii_case("standard"));
    let sched_file = if explicit_uniform {
        None
    } else {
        find_varian_schedule_header_only(dir, &pp)
    };
    let prelim_has_nus = explicit_nus || sched_file.is_some();
    let ni = if prelim_has_nus {
        nimax.unwrap_or(ni_raw)
    } else {
        ni_raw
    };
    let ni2 = if prelim_has_nus {
        ni2max.unwrap_or(ni2_raw)
    } else {
        ni2_raw
    };

    if ni <= 1 {
        return Err(format!(
            "missing or degenerate Varian indirect `ni`={ni} in {} (expected > 1 for {expected_dims}D)",
            dir.display()
        ));
    }
    let sw1 = pp
        .first_f64("sw1")
        .filter(|&v| v > 0.0 && v.is_finite())
        .ok_or_else(|| {
            format!(
                "missing positive `sw1` in Varian procpar ({})",
                dir.display()
            )
        })?;
    let _ = sw1;

    let non_empty_str =
        |k: &str| -> Option<&str> { pp.first_str(k).map(str::trim).filter(|s| !s.is_empty()) };
    let raw_fcoef_1 = if ni2 > 1 {
        non_empty_str("f1coef").or_else(|| non_empty_str("fcoef"))
    } else {
        non_empty_str("fcoef").or_else(|| non_empty_str("f1coef"))
    };
    let fcoef_1 = parse_varian_fcoef_header_only(raw_fcoef_1)?;
    let qm1 = parse_varian_phase_header_only(&pp, "phase", fcoef_1)?;
    let p1_mult = if qm1 == QuadMode::Qf { 1 } else { 2 };

    if expected_dims == 2 {
        if ni2 > 1 {
            return Err(format!(
                "dimensionality mismatch in {}: expected 2D (`ni2 <= 1`), got `ni2`={ni2}",
                dir.display()
            ));
        }
        let array_str = pp
            .first_str("array")
            .unwrap_or("")
            .replace([' ', '(', ')'], "");
        if !array_str.is_empty() && array_str != "phase" && qm1 != QuadMode::Qf {
            return Err(format!(
                "unsupported Varian 2D `array`='{array_str}' in {} (expected '' or 'phase')",
                dir.display()
            ));
        }
        let uniform_traces = ni * p1_mult;
        let is_nus = explicit_nus || (sched_file.is_some() && observed_traces < uniform_traces);
        if is_nus {
            let sp = sched_file.ok_or_else(|| {
                format!(
                    "Varian 2D NUS dataset in {} is missing sampling.sch / nuslist schedule file",
                    dir.display()
                )
            })?;
            let sched_txt = read_header_text_capped(&sp)?;
            let decoded = nmr_core::pulprog::decode_nuslist_str(
                &sched_txt,
                &[ni],
                nmr_core::pulprog::AcquisitionOrder::Dim2D,
            )
            .map_err(|e| format!("invalid Varian 2D NUS schedule {}: {e}", sp.display()))?;
            let expected_acq = decoded.points.len() * p1_mult;
            if observed_traces != expected_acq
                && !(allow_truncated_fid && observed_traces > 0 && observed_traces < expected_acq)
            {
                return Err(format!(
                    "Varian 2D NUS trace count mismatch in {}: fid has {observed_traces} traces, schedule {} expects {expected_acq}",
                    dir.display(),
                    sp.display()
                ));
            }
        } else {
            if let Some(sp) = sched_file {
                let sched_txt = read_header_text_capped(&sp)?;
                let decoded = nmr_core::pulprog::decode_nuslist_str(
                    &sched_txt,
                    &[ni],
                    nmr_core::pulprog::AcquisitionOrder::Dim2D,
                )
                .map_err(|e| format!("invalid Varian 2D schedule {}: {e}", sp.display()))?;
                if decoded.points.len() * p1_mult != observed_traces {
                    return Err(format!(
                        "Varian 2D schedule {} has {} points ({} traces) inconsistent with fid trace count {observed_traces}",
                        sp.display(),
                        decoded.points.len(),
                        decoded.points.len() * p1_mult
                    ));
                }
            }
            if observed_traces < uniform_traces && !allow_truncated_fid {
                return Err(format!(
                    "Varian 2D fid in {} has {observed_traces} traces, expected {uniform_traces}",
                    dir.display()
                ));
            }
        }
        return Ok(());
    }

    // 3D Varian header checks
    if ni2 <= 1 {
        return Err(format!(
            "missing or degenerate Varian 3D indirect `ni2`={ni2} in {} (expected > 1)",
            dir.display()
        ));
    }
    let sw2 = pp
        .first_f64("sw2")
        .filter(|&v| v > 0.0 && v.is_finite())
        .ok_or_else(|| {
            format!(
                "missing positive `sw2` in Varian 3D procpar ({})",
                dir.display()
            )
        })?;
    let _ = sw2;

    let fcoef_2 = parse_varian_fcoef_header_only(non_empty_str("f2coef"))?;
    let qm2 = parse_varian_phase_header_only(&pp, "phase2", fcoef_2)?;
    let p2_mult = if qm2 == QuadMode::Qf { 1 } else { 2 };

    let array_str = pp
        .first_str("array")
        .unwrap_or("phase,phase2")
        .replace([' ', '(', ')'], "");
    if !array_str.is_empty()
        && array_str != "phase,phase2"
        && array_str != "phase2,phase"
        && qm1 != QuadMode::Qf
        && qm2 != QuadMode::Qf
    {
        return Err(format!(
            "unsupported Varian 3D `array`='{array_str}' in {} (expected 'phase,phase2' or 'phase2,phase')",
            dir.display()
        ));
    }

    let traces_per_point = p1_mult * p2_mult;
    let uniform_traces = ni * p1_mult * ni2 * p2_mult;
    let is_nus = explicit_nus || (sched_file.is_some() && observed_traces < uniform_traces);
    if is_nus {
        let sp = sched_file.ok_or_else(|| {
            format!(
                "Varian 3D NUS dataset in {} is missing sampling.sch / nuslist schedule file",
                dir.display()
            )
        })?;
        let sched_txt = read_header_text_capped(&sp)?;
        let decoded = nmr_core::pulprog::decode_nuslist_str(
            &sched_txt,
            &[ni, ni2],
            nmr_core::pulprog::AcquisitionOrder::Aqseq321,
        )
        .map_err(|e| format!("invalid Varian 3D NUS schedule {}: {e}", sp.display()))?;
        let expected_acq = decoded.points.len() * traces_per_point;
        if observed_traces != expected_acq
            && !(allow_truncated_fid && observed_traces > 0 && observed_traces < expected_acq)
        {
            return Err(format!(
                "Varian 3D NUS trace count mismatch in {}: fid has {observed_traces} traces, schedule {} expects {expected_acq}",
                dir.display(),
                sp.display()
            ));
        }
    } else {
        if let Some(sp) = sched_file {
            let sched_txt = read_header_text_capped(&sp)?;
            let decoded = nmr_core::pulprog::decode_nuslist_str(
                &sched_txt,
                &[ni, ni2],
                nmr_core::pulprog::AcquisitionOrder::Aqseq321,
            )
            .map_err(|e| format!("invalid Varian 3D schedule {}: {e}", sp.display()))?;
            if decoded.points.len() * traces_per_point != observed_traces {
                return Err(format!(
                    "Varian 3D schedule {} has {} points ({} traces) inconsistent with fid trace count {observed_traces}",
                    sp.display(),
                    decoded.points.len(),
                    decoded.points.len() * traces_per_point
                ));
            }
        }
        if observed_traces < uniform_traces && !allow_truncated_fid {
            return Err(format!(
                "Varian 3D fid in {} has {observed_traces} traces, expected {uniform_traces}",
                dir.display()
            ));
        }
    }
    Ok(())
}

/// Header-only compatibility preflight (`F5`): validates Bruker (`acqus`/`acqu2s`/`acqu3s`,
/// `pulseprogram`, `nuslist`, `ser` byte stride) or Varian (`procpar`, `sampling.sch`, 32-byte
/// `fid`/`origfid` `datafilehead` + file size, GFT `RDphase`/`array` invariants) with **zero**
/// sample-payload reads, **zero** FFT, **zero** peak picking, and **zero** solving.
pub fn check_fid_dir_header_compatibility(
    dir: &Path,
    expected_dims: usize,
    allow_truncated_fid: bool,
) -> Result<(), String> {
    if !matches!(expected_dims, 2 | 3) {
        return Err(format!(
            "unsupported expected_dims={expected_dims} for {} (expected 2 or 3)",
            dir.display()
        ));
    }
    if find_child_ci(dir, "acqus").is_some() || find_child_ci(dir, "acqu").is_some() {
        check_bruker_header_compatibility(dir, expected_dims, allow_truncated_fid)
    } else if find_child_ci(dir, "procpar").is_some() {
        check_varian_header_compatibility(dir, expected_dims, allow_truncated_fid)
    } else {
        Err(format!(
            "unrecognized raw FID directory {} (missing Bruker acqus/acqu and Varian procpar)",
            dir.display()
        ))
    }
}

/// Run discovery and header-only compatibility checks across manifest entries (`--discover-only`, `F5`)
/// with strictly zero FFT, zero peak picking, and zero solving.
pub fn run_real_benchmark_discover_only(
    cfg: &RealBenchmarkConfig,
) -> Result<DiscoverOnlySummary, String> {
    let manifest_txt = fs::read_to_string(&cfg.manifest_path)
        .map_err(|e| format!("failed to read {}: {e}", cfg.manifest_path.display()))?;
    let all_entries = parse_real_manifest_str(&manifest_txt)?;
    let entries: Vec<RealManifestEntry> = all_entries
        .into_iter()
        .filter(|e| cfg.split.matches(&e.split))
        .filter(|e| {
            cfg.cohort_opt
                .as_deref()
                .is_none_or(|c| e.cohort.trim().eq_ignore_ascii_case(c.trim()))
        })
        .collect();

    if entries.is_empty() {
        return Err(match cfg.cohort_opt.as_deref() {
            Some(c) => format!(
                "no entries matching --split {} --cohort {c} in {}",
                cfg.split.as_str(),
                cfg.manifest_path.display()
            ),
            None => format!(
                "no entries matching --split {} in {}",
                cfg.split.as_str(),
                cfg.manifest_path.display()
            ),
        });
    }

    let mut records = Vec::with_capacity(entries.len());
    let mut n_ok = 0usize;
    let mut n_header_incompatible = 0usize;
    let mut n_missing_experiments = 0usize;
    let mut n_skipped = 0usize;

    for entry in entries {
        let loc = match resolve_local_entry_paths(
            &entry.bmrb_id,
            &cfg.data_root,
            cfg.allow_repo_fallbacks,
        ) {
            Ok(l) => l,
            Err(expected_path) => {
                n_skipped += 1;
                records.push(DiscoverOnlyEntryRecord {
                    manifest_entry: entry,
                    status: DiscoverOnlyStatus::Skipped,
                    entry_dir: None,
                    truth_path: None,
                    discovered_experiments: Vec::new(),
                    missing_experiments: Vec::new(),
                    reason: Some(format!("missing local data ({expected_path})")),
                });
                continue;
            }
        };

        // Verify truth file exists and parses cleanly (header/shift table only)
        let Some(ref truth_p) = loc.truth_path else {
            n_missing_experiments += 1;
            records.push(DiscoverOnlyEntryRecord {
                manifest_entry: entry,
                status: DiscoverOnlyStatus::MissingExperiments,
                entry_dir: Some(loc.entry_dir.clone()),
                truth_path: None,
                discovered_experiments: Vec::new(),
                missing_experiments: vec!["TRUTH_STR".to_string()],
                reason: Some(format!(
                    "missing NMR-STAR / shift truth file in {}",
                    loc.entry_dir.display()
                )),
            });
            continue;
        };
        if let Err(e) = parse_truth_auto(truth_p) {
            n_header_incompatible += 1;
            records.push(DiscoverOnlyEntryRecord {
                manifest_entry: entry,
                status: DiscoverOnlyStatus::HeaderIncompatible,
                entry_dir: Some(loc.entry_dir),
                truth_path: Some(truth_p.clone()),
                discovered_experiments: Vec::new(),
                missing_experiments: Vec::new(),
                reason: Some(format!(
                    "truth shift file {} failed header/table parse: {e}",
                    truth_p.display()
                )),
            });
            continue;
        }

        let (discovered, all_inspected) =
            match discover_entry_experiments(&loc.entry_dir, Some(entry.field_mhz)) {
                Ok(d) => d,
                Err(e) => {
                    n_missing_experiments += 1;
                    records.push(DiscoverOnlyEntryRecord {
                        manifest_entry: entry,
                        status: DiscoverOnlyStatus::MissingExperiments,
                        entry_dir: Some(loc.entry_dir),
                        truth_path: Some(truth_p.clone()),
                        discovered_experiments: Vec::new(),
                        missing_experiments: Vec::new(),
                        reason: Some(e),
                    });
                    continue;
                }
            };

        let required_set: BTreeSet<&str> = entry.experiments.iter().map(String::as_str).collect();
        let mut missing_exps: Vec<String> = Vec::new();
        if !discovered.iter().any(|d| d.exp_type == "HSQC") {
            missing_exps.push("HSQC".to_string());
        }
        for &req in &required_set {
            if !discovered.iter().any(|d| d.exp_type == req)
                && !missing_exps.iter().any(|m| m == req)
            {
                missing_exps.push(req.to_string());
            }
        }
        let has_3d = discovered.iter().any(|d| d.exp_type != "HSQC");
        if !has_3d && missing_exps.is_empty() {
            missing_exps.push("3D_TRIPLE_RESONANCE".to_string());
        }

        let mut exp_checks = Vec::with_capacity(discovered.len());
        let mut header_errors = Vec::new();
        for d in &discovered {
            let expected_dims = if d.exp_type == "HSQC" { 2 } else { 3 };
            let is_req = d.exp_type == "HSQC"
                || required_set.is_empty()
                || required_set.contains(d.exp_type.as_str());
            match check_fid_dir_header_compatibility(&d.dir, expected_dims, cfg.allow_truncated_fid)
            {
                Ok(()) => exp_checks.push(DiscoverOnlyExpCheck {
                    exp_type: d.exp_type.clone(),
                    rel_path: d.rel_path.clone(),
                    vendor: d.vendor.clone(),
                    dims: d.dims,
                    h_mhz: d.h_mhz,
                    required: is_req,
                    header_ok: true,
                    error: None,
                }),
                Err(err) => {
                    if is_req {
                        header_errors.push(format!("{} ({}): {err}", d.exp_type, d.rel_path));
                    }
                    exp_checks.push(DiscoverOnlyExpCheck {
                        exp_type: d.exp_type.clone(),
                        rel_path: d.rel_path.clone(),
                        vendor: d.vendor.clone(),
                        dims: d.dims,
                        h_mhz: d.h_mhz,
                        required: is_req,
                        header_ok: false,
                        error: Some(err),
                    });
                }
            }
        }

        if !missing_exps.is_empty() {
            n_missing_experiments += 1;
            let found_summary = if discovered.is_empty() {
                format!(
                    "0 backbone FIDs ({} candidate dirs inspected)",
                    all_inspected.len()
                )
            } else {
                discovered
                    .iter()
                    .map(|d| format!("{}:{}", d.exp_type, d.rel_path))
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            records.push(DiscoverOnlyEntryRecord {
                manifest_entry: entry,
                status: DiscoverOnlyStatus::MissingExperiments,
                entry_dir: Some(loc.entry_dir.clone()),
                truth_path: Some(truth_p.clone()),
                discovered_experiments: exp_checks,
                reason: Some(format!(
                    "missing required experiment(s) [{}] in {} (found: {found_summary})",
                    missing_exps.join(", "),
                    loc.entry_dir.display()
                )),
                missing_experiments: missing_exps,
            });
            continue;
        }

        if !header_errors.is_empty() {
            n_header_incompatible += 1;
            records.push(DiscoverOnlyEntryRecord {
                manifest_entry: entry,
                status: DiscoverOnlyStatus::HeaderIncompatible,
                entry_dir: Some(loc.entry_dir),
                truth_path: Some(truth_p.clone()),
                discovered_experiments: exp_checks,
                missing_experiments: Vec::new(),
                reason: Some(header_errors.join("; ")),
            });
            continue;
        }

        n_ok += 1;
        records.push(DiscoverOnlyEntryRecord {
            manifest_entry: entry,
            status: DiscoverOnlyStatus::Ok,
            entry_dir: Some(loc.entry_dir),
            truth_path: Some(truth_p.clone()),
            discovered_experiments: exp_checks,
            missing_experiments: Vec::new(),
            reason: None,
        });
    }

    let summary = DiscoverOnlySummary {
        split: cfg.split.as_str().to_string(),
        cohort_opt: cfg.cohort_opt.clone(),
        records,
        n_ok,
        n_header_incompatible,
        n_missing_experiments,
        n_skipped,
    };

    print!("{}", format_discover_only_report(&summary));

    if let Some(json_p) = &cfg.json_out {
        if let Some(parent) = json_p.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)
                    .map_err(|e| format!("failed creating {}: {e}", parent.display()))?;
            }
        }
        let json = render_discover_only_json(&summary);
        fs::write(json_p, json).map_err(|e| format!("failed writing {}: {e}", json_p.display()))?;
        println!("Wrote JSON discovery summary -> {}", json_p.display());
    }

    let n_failed = summary.n_header_incompatible + summary.n_missing_experiments;
    if n_failed > 0 {
        return Err(format!(
            "{n_failed} present real-benchmark entry(s) failed --discover-only preflight ({} HEADER_INCOMPATIBLE, {} MISSING_EXPERIMENTS)",
            summary.n_header_incompatible, summary.n_missing_experiments
        ));
    }

    Ok(summary)
}

pub fn format_discover_only_report(summary: &DiscoverOnlySummary) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let coh_lbl = summary.cohort_opt.as_deref().unwrap_or("all");
    let _ = writeln!(
        out,
        "=== Real Time-Domain Benchmark Discovery Preflight (split={}, cohort={}, mode=discover-only) ===",
        summary.split, coh_lbl
    );
    let _ = writeln!(
        out,
        "{:<16} {:<14} {:>4} {:<7} {:<20} | Discovered Experiments / Diagnostic",
        "BMRB", "Protein", "Res", "Vendor", "Status"
    );
    let _ = writeln!(out, "{}", "-".repeat(120));
    for rec in &summary.records {
        let e = &rec.manifest_entry;
        let detail = match rec.status {
            DiscoverOnlyStatus::Ok => rec
                .discovered_experiments
                .iter()
                .map(|d| format!("{}:{}", d.exp_type, d.rel_path))
                .collect::<Vec<_>>()
                .join(", "),
            DiscoverOnlyStatus::HeaderIncompatible
            | DiscoverOnlyStatus::MissingExperiments
            | DiscoverOnlyStatus::Skipped => rec.reason.clone().unwrap_or_default(),
        };
        let _ = writeln!(
            out,
            "{:<16} {:<14} {:>4} {:<7} {:<20} | {}",
            e.bmrb_id,
            truncate_col(&e.protein, 14),
            e.residues,
            truncate_col(&e.vendor, 7),
            rec.status.as_str(),
            detail
        );
    }
    let _ = writeln!(out, "{}", "-".repeat(120));
    let _ = writeln!(
        out,
        "Summary: OK={}, HEADER_INCOMPATIBLE={}, MISSING_EXPERIMENTS={}, SKIPPED={}",
        summary.n_ok,
        summary.n_header_incompatible,
        summary.n_missing_experiments,
        summary.n_skipped
    );
    out
}

pub fn render_discover_only_json(summary: &DiscoverOnlySummary) -> String {
    let coh_json = match &summary.cohort_opt {
        Some(c) => format!("\"{}\"", json_escape(c)),
        None => "null".to_string(),
    };
    let mut j = String::new();
    j.push_str(&format!(
        "{{\n  \"version\": \"{}\",\n  \"mode\": \"discover-only\",\n  \"split\": \"{}\",\n  \"cohort\": {},\n  \"counts\": {{ \"ok\": {}, \"header_incompatible\": {}, \"missing_experiments\": {}, \"skipped\": {} }},\n  \"entries\": [\n",
        json_escape(&version_string()),
        json_escape(&summary.split),
        coh_json,
        summary.n_ok,
        summary.n_header_incompatible,
        summary.n_missing_experiments,
        summary.n_skipped,
    ));
    for (idx, rec) in summary.records.iter().enumerate() {
        let comma = if idx + 1 == summary.records.len() {
            ""
        } else {
            ","
        };
        let e = &rec.manifest_entry;
        let reason_json = match &rec.reason {
            Some(r) => format!("\"{}\"", json_escape(r)),
            None => "null".to_string(),
        };
        let missing_json = rec
            .missing_experiments
            .iter()
            .map(|m| format!("\"{}\"", json_escape(m)))
            .collect::<Vec<_>>()
            .join(", ");
        let exps_json = rec
            .discovered_experiments
            .iter()
            .map(|d| {
                let err_j = match &d.error {
                    Some(er) => format!("\"{}\"", json_escape(er)),
                    None => "null".to_string(),
                };
                format!(
                    "{{ \"exp\": \"{}\", \"path\": \"{}\", \"vendor\": \"{}\", \"dims\": {}, \"h_mhz\": {:.3}, \"required\": {}, \"header_ok\": {}, \"error\": {} }}",
                    json_escape(&d.exp_type),
                    json_escape(&d.rel_path),
                    json_escape(&d.vendor),
                    d.dims,
                    d.h_mhz,
                    d.required,
                    d.header_ok,
                    err_j
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        j.push_str(&format!(
            "    {{\n      \"bmrb_id\": \"{}\",\n      \"protein\": \"{}\",\n      \"residues\": {},\n      \"vendor\": \"{}\",\n      \"split\": \"{}\",\n      \"cohort\": \"{}\",\n      \"status\": \"{}\",\n      \"missing_experiments\": [{}],\n      \"discovered_experiments\": [{}],\n      \"reason\": {}\n    }}{comma}\n",
            json_escape(&e.bmrb_id),
            json_escape(&e.protein),
            e.residues,
            json_escape(&e.vendor),
            json_escape(&e.split),
            json_escape(&e.cohort),
            rec.status.as_str(),
            missing_json,
            exps_json,
            reason_json,
        ));
    }
    j.push_str("  ]\n}\n");
    j
}

/// Execute `process -> pick -> assign -> score` across all manifest entries matching `cfg.split`
/// (and optional `cfg.cohort_opt`).
pub fn run_real_benchmark_with_loaders(
    cfg: &RealBenchmarkConfig,
    load_fid_2d: &dyn Fn(&Path) -> Result<Spectrum2D, String>,
    load_fid_3d: &dyn Fn(&Path, &str) -> Result<Spectrum3D, String>,
) -> Result<RealBenchmarkSummary, String> {
    run_real_benchmark_with_loaders_and_warnings(
        cfg,
        &|dir| load_fid_2d(dir).map(|s| (s, None)),
        &|dir, exp| load_fid_3d(dir, exp).map(|s| (s, None)),
    )
}

/// 2D raw-FID loader callback returning `(Spectrum2D, optional_truncated_fid_warning)`.
pub type Fid2DWarningLoader<'a> =
    dyn Fn(&Path) -> Result<(Spectrum2D, Option<String>), String> + 'a;

/// 3D raw-FID loader callback returning `(Spectrum3D, optional_truncated_fid_warning)`.
pub type Fid3DWarningLoader<'a> =
    dyn Fn(&Path, &str) -> Result<(Spectrum3D, Option<String>), String> + 'a;

/// Variant of [`run_real_benchmark_with_loaders`] whose 2D/3D loaders also return optional
/// truncated-FID zero-filling warnings (`F2`) so they propagate into stderr and `DatasetQcReport`.
pub fn run_real_benchmark_with_loaders_and_warnings(
    cfg: &RealBenchmarkConfig,
    load_fid_2d: &Fid2DWarningLoader<'_>,
    load_fid_3d: &Fid3DWarningLoader<'_>,
) -> Result<RealBenchmarkSummary, String> {
    let resolved_engine = resolve_solver_engine(cfg.solver, cfg.engine_opt)?;
    let manifest_txt = fs::read_to_string(&cfg.manifest_path)
        .map_err(|e| format!("failed to read {}: {e}", cfg.manifest_path.display()))?;
    let all_entries = parse_real_manifest_str(&manifest_txt)?;
    let entries: Vec<RealManifestEntry> = all_entries
        .into_iter()
        .filter(|e| cfg.split.matches(&e.split))
        .filter(|e| {
            cfg.cohort_opt
                .as_deref()
                .is_none_or(|c| e.cohort.trim().eq_ignore_ascii_case(c.trim()))
        })
        .collect();

    if entries.is_empty() {
        return Err(match cfg.cohort_opt.as_deref() {
            Some(c) => format!(
                "no entries matching --split {} --cohort {c} in {}",
                cfg.split.as_str(),
                cfg.manifest_path.display()
            ),
            None => format!(
                "no entries matching --split {} in {}",
                cfg.split.as_str(),
                cfg.manifest_path.display()
            ),
        });
    }

    let priors = load_priors(cfg.priors_path.as_deref())?;
    let score_cfg = ScoreConfig::default();
    let picker_cfg = PickerConfig::default();

    let shared_gpu = if cfg.solver == SolverKind::Standard && resolved_engine == EngineKind::Gpu {
        match GpuEngine::new_headless() {
            Ok(g) => Some(g),
            Err(err) if cfg.engine_opt.is_none() => {
                eprintln!(
                    "Warning: WebGPU unavailable for --solver standard ({err}); falling back to CPU (f64)."
                );
                None
            }
            Err(err) => return Err(err),
        }
    } else {
        None
    };
    let shared_bayes_gpu = if cfg.solver == SolverKind::BayesSa
        && resolved_engine == EngineKind::Gpu
    {
        match BayesSaGpuSolver::new_headless() {
            Ok(bgpu) => Some(bgpu),
            Err(err) if cfg.engine_opt.is_none() => {
                eprintln!(
                        "Warning: WebGPU unavailable for --solver bayes-sa ({err}); falling back to CPU (f64, bayes-sa)."
                    );
                None
            }
            Err(err) => return Err(err),
        }
    } else {
        None
    };

    let mut records: Vec<EntryBenchmarkRecord> = Vec::with_capacity(entries.len());
    let mut solver_label = match cfg.solver {
        SolverKind::Standard => "standard".to_string(),
        SolverKind::EvoGraph => "evo-graph".to_string(),
        SolverKind::BayesSa => "bayes-sa".to_string(),
    };

    for entry in entries {
        let loc = match resolve_local_entry_paths(
            &entry.bmrb_id,
            &cfg.data_root,
            cfg.allow_repo_fallbacks,
        ) {
            Ok(l) => l,
            Err(expected_path) => {
                records.push(EntryBenchmarkRecord {
                    manifest_entry: entry,
                    outcome: EntryBenchmarkOutcome::SkippedMissingData { expected_path },
                });
                continue;
            }
        };

        let t_entry = Instant::now();

        // 1. Load deposited ground-truth shifts
        let Some(truth_path) = loc.truth_path else {
            records.push(EntryBenchmarkRecord {
                outcome: EntryBenchmarkOutcome::Failed(EntryFailureDiagnostic {
                    stage: "truth".to_string(),
                    experiment: None,
                    vendor: entry.vendor.clone(),
                    quad_mode: "n/a".to_string(),
                    nus_status: "n/a".to_string(),
                    reason: format!(
                        "missing NMR-STAR / shift truth file in {}",
                        loc.entry_dir.display()
                    ),
                }),
                manifest_entry: entry,
            });
            continue;
        };

        let (truth_ref, _excluded_q, _fmt) = match parse_truth_auto(&truth_path) {
            Ok(t) => t,
            Err(e) => {
                records.push(EntryBenchmarkRecord {
                    outcome: EntryBenchmarkOutcome::Failed(EntryFailureDiagnostic {
                        stage: "truth".to_string(),
                        experiment: None,
                        vendor: entry.vendor.clone(),
                        quad_mode: "n/a".to_string(),
                        nus_status: "n/a".to_string(),
                        reason: e,
                    }),
                    manifest_entry: entry,
                });
                continue;
            }
        };

        // 2. Auto-discover raw FID experiments inside `entry_dir`
        let (discovered, all_inspected) =
            match discover_entry_experiments(&loc.entry_dir, Some(entry.field_mhz)) {
                Ok(d) => d,
                Err(e) => {
                    records.push(EntryBenchmarkRecord {
                        outcome: EntryBenchmarkOutcome::Failed(EntryFailureDiagnostic {
                            stage: "discovery".to_string(),
                            experiment: None,
                            vendor: entry.vendor.clone(),
                            quad_mode: "n/a".to_string(),
                            nus_status: "n/a".to_string(),
                            reason: e,
                        }),
                        manifest_entry: entry,
                    });
                    continue;
                }
            };

        let fallback_vendor = all_inspected
            .first()
            .map(|c| c.vendor.clone())
            .unwrap_or_else(|| entry.vendor.clone());
        let fallback_quad = all_inspected
            .first()
            .map(|c| c.quad_mode_summary.clone())
            .unwrap_or_else(|| "unknown".to_string());
        let fallback_nus = all_inspected
            .first()
            .map(|c| c.nus_summary.clone())
            .unwrap_or_else(|| "unknown".to_string());

        let Some(hsqc_exp) = discovered.iter().find(|d| d.exp_type == "HSQC") else {
            records.push(EntryBenchmarkRecord {
                outcome: EntryBenchmarkOutcome::Failed(EntryFailureDiagnostic {
                    stage: "discovery".to_string(),
                    experiment: Some("HSQC".to_string()),
                    vendor: fallback_vendor,
                    quad_mode: fallback_quad,
                    nus_status: fallback_nus,
                    reason: format!(
                        "no 2D 1H-15N HSQC raw FID discovered in {} ({} candidate FID dirs inspected)",
                        loc.entry_dir.display(),
                        all_inspected.len()
                    ),
                }),
                manifest_entry: entry,
            });
            continue;
        };

        // Check that any experiment explicitly listed in `real_manifest.tsv` is present
        let required_set: BTreeSet<&str> = entry.experiments.iter().map(String::as_str).collect();
        let mut missing_required: Vec<String> = Vec::new();
        for &req in &required_set {
            if !discovered.iter().any(|d| d.exp_type == req) {
                missing_required.push(req.to_string());
            }
        }
        if !missing_required.is_empty() {
            records.push(EntryBenchmarkRecord {
                outcome: EntryBenchmarkOutcome::Failed(EntryFailureDiagnostic {
                    stage: "discovery".to_string(),
                    experiment: Some(missing_required.join(",")),
                    vendor: hsqc_exp.vendor.clone(),
                    quad_mode: hsqc_exp.quad_mode_summary.clone(),
                    nus_status: hsqc_exp.nus_summary.clone(),
                    reason: format!(
                        "manifest-declared experiment(s) [{}] not discovered in {} (found: {})",
                        missing_required.join(", "),
                        loc.entry_dir.display(),
                        discovered
                            .iter()
                            .map(|d| format!("{}:{}", d.exp_type, d.rel_path))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                }),
                manifest_entry: entry,
            });
            continue;
        }

        let triple_exps: Vec<&DiscoveredFidExperiment> =
            discovered.iter().filter(|d| d.exp_type != "HSQC").collect();
        if triple_exps.is_empty() {
            records.push(EntryBenchmarkRecord {
                outcome: EntryBenchmarkOutcome::Failed(EntryFailureDiagnostic {
                    stage: "discovery".to_string(),
                    experiment: None,
                    vendor: hsqc_exp.vendor.clone(),
                    quad_mode: hsqc_exp.quad_mode_summary.clone(),
                    nus_status: hsqc_exp.nus_summary.clone(),
                    reason: format!(
                        "no 3D triple-resonance raw FIDs discovered alongside HSQC in {}",
                        loc.entry_dir.display()
                    ),
                }),
                manifest_entry: entry,
            });
            continue;
        }

        // 3. Process 2D HSQC raw FID & pick roots
        let mut entry_trunc_warnings: Vec<String> = Vec::new();
        let hsqc_spec = match load_fid_2d(&hsqc_exp.dir) {
            Ok((s, maybe_w)) => {
                if let Some(w) = maybe_w {
                    eprintln!("Warning: [{}] HSQC — {w}", entry.bmrb_id);
                    entry_trunc_warnings.push(format!("HSQC: {w}"));
                }
                s
            }
            Err(reason) => {
                records.push(EntryBenchmarkRecord {
                    outcome: EntryBenchmarkOutcome::Failed(EntryFailureDiagnostic {
                        stage: "process".to_string(),
                        experiment: Some("HSQC".to_string()),
                        vendor: hsqc_exp.vendor.clone(),
                        quad_mode: hsqc_exp.quad_mode_summary.clone(),
                        nus_status: hsqc_exp.nus_summary.clone(),
                        reason,
                    }),
                    manifest_entry: entry,
                });
                continue;
            }
        };

        let roots = pick_hsqc_2d(&hsqc_spec, &picker_cfg);
        let mut all_peaks: Vec<Peak> = roots.clone();
        let mut used_exps: Vec<(String, String)> =
            vec![("HSQC".to_string(), hsqc_exp.rel_path.clone())];
        let mut entry_failed_diag: Option<EntryFailureDiagnostic> = None;

        // 4. Process each 3D triple-resonance raw FID one cube at a time & pick strips
        let mut strip_bank = StripBank3D::new();
        let mut axis_rescue_audits: Vec<AxisRescueAuditRecord> = Vec::new();
        for texp in &triple_exps {
            let is_required =
                required_set.is_empty() || required_set.contains(texp.exp_type.as_str());
            match load_fid_3d(&texp.dir, &texp.exp_type) {
                Ok((mut spec3d, maybe_w)) => {
                    if let Some(w) = maybe_w {
                        eprintln!("Warning: [{}] {} — {w}", entry.bmrb_id, texp.exp_type);
                        entry_trunc_warnings.push(format!("{}: {w}", texp.exp_type));
                    }
                    // Align 3D HN plane to HSQC via two-stage rigid (dH, dN) registration + audited `{sign, fold}` check (`13.3:P1-1`)
                    let (dh, dn, audit_opt) = register_3d_hn_to_hsqc_adaptive_with_audit(
                        &mut spec3d,
                        &hsqc_spec,
                        !cfg.no_axis_rescue,
                    );
                    if let Some(audit) = audit_opt {
                        axis_rescue_audits.push(audit);
                    }
                    if dh.abs() > 1e-6 || dn.abs() > 1e-6 {
                        eprintln!(
                            "[register] {}: aligned {} to HSQC: dH={:+.4} ppm, dN={:+.3} ppm",
                            entry.bmrb_id, texp.exp_type, dh, dn
                        );
                    }
                    let pks = strip_bank.pick_and_record_3d(&spec3d, &roots, &picker_cfg);
                    all_peaks.extend(pks);
                    used_exps.push((texp.exp_type.clone(), texp.rel_path.clone()));
                }
                Err(reason) => {
                    if is_required {
                        entry_failed_diag = Some(EntryFailureDiagnostic {
                            stage: "process".to_string(),
                            experiment: Some(texp.exp_type.clone()),
                            vendor: texp.vendor.clone(),
                            quad_mode: texp.quad_mode_summary.clone(),
                            nus_status: texp.nus_summary.clone(),
                            reason,
                        });
                        break;
                    } else if cfg.verbose {
                        eprintln!(
                            "[real-benchmark] {}: skipping optional unlisted {} ({}) after error: {}",
                            entry.bmrb_id, texp.exp_type, texp.rel_path, reason
                        );
                    }
                }
            }
        }

        if let Some(diag) = entry_failed_diag {
            records.push(EntryBenchmarkRecord {
                manifest_entry: entry,
                outcome: EntryBenchmarkOutcome::Failed(diag),
            });
            continue;
        }

        if used_exps.len() >= 3 && !roots.is_empty() {
            all_peaks.extend(corroborated_re_pick_3d(
                &strip_bank,
                &roots,
                &all_peaks,
                &picker_cfg,
                &CorroboratedRepickConfig::default(),
            ));
        }

        let all_systems_for_eval =
            match build_spin_systems(&all_peaks, &Tolerances::default(), 3, false) {
                Ok(s) => s,
                Err(reason) => {
                    records.push(EntryBenchmarkRecord {
                        outcome: EntryBenchmarkOutcome::Failed(EntryFailureDiagnostic {
                            stage: "pick".to_string(),
                            experiment: None,
                            vendor: hsqc_exp.vendor.clone(),
                            quad_mode: hsqc_exp.quad_mode_summary.clone(),
                            nus_status: hsqc_exp.nus_summary.clone(),
                            reason,
                        }),
                        manifest_entry: entry,
                    });
                    continue;
                }
            };

        // 5. Run solver across `0..cfg.seeds` & score strictly with `fit_unlabelled_global_offsets_detailed`
        let num_seeds = cfg.seeds.max(1);
        let mut tot_assignable = 0usize;
        let mut tot_assigned = 0usize;
        let mut tot_correct = 0usize;
        let mut tot_hc_assigned = 0usize;
        let mut tot_hc_correct = 0usize;
        let mut assignable_per_seed = 0usize;
        let mut reported_offsets = ShiftOffsets::default();
        let mut reported_detailed_offsets = DetailedReferenceOffsets::default();
        let mut reported_prior_detailed_offsets = DetailedReferenceOffsets::default();
        let mut reported_qc_report = DatasetQcReport::default();
        let mut entry_predictions: Vec<(f64, bool)> = Vec::new();
        let mut assign_err: Option<String> = None;

        let bayes_prebuilt = if cfg.solver == SolverKind::BayesSa {
            let use_gpu = shared_bayes_gpu.is_some();
            let base_bcfg = build_bayes_sa_config_cli(
                use_gpu,
                &truth_ref.sequence,
                &score_cfg,
                ToleranceOverrides::default(),
                cfg.runs_opt,
                cfg.steps_opt,
                cfg.base_seed,
            );
            let conditioned = if base_bcfg.enable_weak_root_penalty {
                condition_peaks_preserving_hsqc_roots(&all_peaks, &base_bcfg.tolerances)
            } else {
                all_peaks.clone()
            };
            match build_spin_systems(
                &conditioned,
                &base_bcfg.tolerances,
                base_bcfg.max_cands,
                false,
            ) {
                Ok(bsys) => Some((bsys, use_gpu)),
                Err(reason) => {
                    records.push(EntryBenchmarkRecord {
                        outcome: EntryBenchmarkOutcome::Failed(EntryFailureDiagnostic {
                            stage: "pick".to_string(),
                            experiment: None,
                            vendor: hsqc_exp.vendor.clone(),
                            quad_mode: hsqc_exp.quad_mode_summary.clone(),
                            nus_status: hsqc_exp.nus_summary.clone(),
                            reason,
                        }),
                        manifest_entry: entry,
                    });
                    continue;
                }
            }
        } else {
            None
        };

        for s_idx in 0..num_seeds {
            let cur_seed = cfg.base_seed + (s_idx as u64);
            let (systems_for_seed, mut res_for_seed, eff_priors, det_off, mut qcr, elbl) =
                if let Some((ref bsys, use_gpu)) = bayes_prebuilt {
                    let bcfg = build_bayes_sa_config_cli(
                        use_gpu,
                        &truth_ref.sequence,
                        &score_cfg,
                        ToleranceOverrides::default(),
                        cfg.runs_opt,
                        cfg.steps_opt,
                        cur_seed,
                    );
                    let (mut res, mut rep, lbl) = if let Some(ref bgpu) = shared_bayes_gpu {
                        let lbl = format!(
                            "WebGPU ({}, {}, bayes-sa)",
                            bgpu.engine.adapter_name, bgpu.engine.backend_name
                        );
                        match bgpu.solve_with_peaks(
                            Some(&all_peaks),
                            bsys,
                            &truth_ref.sequence,
                            &priors,
                            &bcfg,
                        ) {
                            Ok((res, rep)) => (res, rep, lbl),
                            Err(e) => {
                                assign_err = Some(e);
                                break;
                            }
                        }
                    } else {
                        let lbl = if resolved_engine == EngineKind::Gpu {
                            "CPU (f64, bayes-sa, auto-fallback)".to_string()
                        } else {
                            "CPU (f64, bayes-sa)".to_string()
                        };
                        let (res, rep) = solve_bayes_sa_with_peaks(
                            Some(&all_peaks),
                            bsys,
                            &truth_ref.sequence,
                            &priors,
                            &bcfg,
                        );
                        (res, rep, lbl)
                    };
                    let fit_runs = if use_gpu {
                        BAYES_SA_GPU_CALIBRATED_RUNS
                    } else {
                        BAYES_SA_CPU_CALIBRATED_RUNS
                    };
                    let req_runs = cfg
                        .runs_opt
                        .unwrap_or_else(|| rep.replica_best_energies.len().max(1));
                    if req_runs != fit_runs {
                        rep.qc_report
                            .mark_uncalibrated_runs(&lbl, req_runs, fit_runs);
                        nmr_core::qc::apply_dataset_qc_to_assignments(&mut res, &rep.qc_report);
                    }
                    if s_idx == 0 && cfg.verbose {
                        eprintln!(
                            "[bayes-sa] {}: T_crit={:.4}, max_Cv={:.4}, mu_null={:+.4}, best_energy={:+.3}",
                            entry.bmrb_id, rep.t_crit, rep.max_cv, rep.mu_null, rep.best_energy
                        );
                    }
                    let eff = priors.with_offsets(&rep.applied_offsets);
                    let det = rep.detailed_offsets;
                    let qc = rep.qc_report;
                    (bsys.clone(), res, eff, det, qc, lbl)
                } else {
                    let eff_engine_opt = if cfg.solver == SolverKind::Standard
                        && resolved_engine == EngineKind::Gpu
                        && shared_gpu.is_none()
                    {
                        Some(EngineKind::Cpu)
                    } else {
                        cfg.engine_opt
                    };
                    let wf = match execute_assignment_workflow_with_options(
                        cfg.solver,
                        eff_engine_opt,
                        &all_peaks,
                        &truth_ref.sequence,
                        &priors,
                        &score_cfg,
                        ToleranceOverrides::default(),
                        cfg.runs_opt,
                        cfg.steps_opt,
                        STANDARD_CPU_CALIBRATED_RUNS,
                        cur_seed,
                        shared_gpu.as_ref(),
                        shared_bayes_gpu.as_ref(),
                    ) {
                        Ok(w) => w,
                        Err(e) => {
                            assign_err = Some(e);
                            break;
                        }
                    };
                    (
                        wf.systems,
                        wf.result,
                        wf.effective_priors,
                        wf.detailed_offsets,
                        wf.qc_report,
                        wf.engine_label,
                    )
                };
            if !entry_trunc_warnings.is_empty() {
                apply_truncated_fid_qc_warning(&mut res_for_seed, &mut qcr, &entry_trunc_warnings);
            }
            solver_label = elbl;

            let (pred_ref, conf_by_k) = to_reference(
                &res_for_seed,
                &systems_for_seed,
                &truth_ref.sequence,
                &truth_ref.seq_ids,
                0.0,
                Some(&eff_priors),
            );
            let mut conf_by_seq_id = BTreeMap::new();
            for (k, c) in conf_by_k {
                conf_by_seq_id.insert(truth_ref.seq_ids[k], c);
            }

            let eval = evaluate_prediction_against_truth(
                &pred_ref,
                &truth_ref,
                &conf_by_seq_id,
                Some(&all_systems_for_eval),
                cfg.hc_threshold,
                true,
            );
            if s_idx == 0 {
                assignable_per_seed = eval.assignable;
                reported_offsets = eval.global_offsets;
                reported_detailed_offsets = eval.detailed_offsets;
                reported_prior_detailed_offsets = det_off;
                reported_qc_report = qcr;
            }
            entry_predictions.extend_from_slice(&eval.placed_predictions);
            tot_assignable += eval.assignable;
            tot_assigned += eval.assigned;
            tot_correct += eval.correct;
            tot_hc_assigned += eval.hc_assigned;
            tot_hc_correct += eval.hc_correct;
        }

        if let Some(reason) = assign_err {
            records.push(EntryBenchmarkRecord {
                outcome: EntryBenchmarkOutcome::Failed(EntryFailureDiagnostic {
                    stage: "assign".to_string(),
                    experiment: None,
                    vendor: hsqc_exp.vendor.clone(),
                    quad_mode: hsqc_exp.quad_mode_summary.clone(),
                    nus_status: hsqc_exp.nus_summary.clone(),
                    reason,
                }),
                manifest_entry: entry,
            });
            continue;
        }

        let mean_accuracy = if tot_assigned > 0 {
            (tot_correct as f64) / (tot_assigned as f64)
        } else {
            0.0
        };
        let mean_recall = if tot_assignable > 0 {
            (tot_correct as f64) / (tot_assignable as f64)
        } else {
            0.0
        };
        let mean_hc_precision = if tot_hc_assigned > 0 {
            (tot_hc_correct as f64) / (tot_hc_assigned as f64)
        } else {
            0.0
        };
        let mean_hc_recall = if tot_assignable > 0 {
            (tot_hc_correct as f64) / (tot_assignable as f64)
        } else {
            0.0
        };
        let reliability = ProteinReliabilityCurve::from_predictions_with_repeats(
            &entry_predictions,
            cfg.hc_threshold,
            num_seeds as f64,
        );

        records.push(EntryBenchmarkRecord {
            manifest_entry: entry,
            outcome: EntryBenchmarkOutcome::Succeeded(Box::new(EntrySuccessMetrics {
                seeds: num_seeds,
                discovered_experiments: used_exps,
                axis_rescue_audits,
                n_hsqc_roots: roots.len(),
                n_total_peaks: all_peaks.len(),
                assignable_per_seed,
                total_assignable: tot_assignable,
                total_assigned: tot_assigned,
                total_correct: tot_correct,
                total_hc_assigned: tot_hc_assigned,
                total_hc_correct: tot_hc_correct,
                mean_accuracy,
                mean_recall,
                mean_hc_precision,
                mean_hc_recall,
                global_offsets: reported_offsets,
                detailed_offsets: reported_detailed_offsets,
                prior_detailed_offsets: reported_prior_detailed_offsets,
                qc_report: reported_qc_report,
                reliability,
                predictions: entry_predictions,
                elapsed_secs: t_entry.elapsed().as_secs_f64(),
            })),
        });
    }

    let summary = summarize_real_benchmark_records(
        cfg.split.as_str(),
        solver_label,
        cfg.seeds.max(1),
        cfg.hc_threshold,
        records,
    );

    print_real_benchmark_report(&summary);

    if let Some(json_p) = &cfg.json_out {
        if let Some(parent) = json_p.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)
                    .map_err(|e| format!("failed creating {}: {e}", parent.display()))?;
            }
        }
        let json = render_summary_json(&summary);
        fs::write(json_p, json).map_err(|e| format!("failed writing {}: {e}", json_p.display()))?;
        println!("Wrote JSON benchmark report -> {}", json_p.display());
    }

    let n_failed = summary
        .records
        .iter()
        .filter(|r| matches!(r.outcome, EntryBenchmarkOutcome::Failed(_)))
        .count();
    if n_failed > 0 {
        return Err(format!(
            "{n_failed} present real-benchmark entry(s) failed processing or scoring (see diagnostics above)"
        ));
    }

    Ok(summary)
}

/// Aggregate macro/micro metrics, per-protein HC abstention vs wrong-HC statistics,
/// worst/median per-protein HC precision, and pooled + per-protein reliability curves (`P1-1`).
pub fn summarize_real_benchmark_records(
    split: &str,
    solver_label: String,
    seeds: usize,
    hc_threshold: f64,
    records: Vec<EntryBenchmarkRecord>,
) -> RealBenchmarkSummary {
    let mut n_ok = 0usize;
    let mut n_hc_active_proteins = 0usize;
    let mut n_hc_abstained_proteins = 0usize;
    let mut sum_acc = 0.0f64;
    let mut sum_rec = 0.0f64;
    let mut sum_active_hcp = 0.0f64;
    let mut sum_hcr = 0.0f64;
    let mut sum_ece = 0.0f64;
    let mut sum_ece_10 = 0.0f64;
    let mut sum_debiased_ece_10 = 0.0f64;
    let mut sum_ece_10_floor = 0.0f64;
    let mut sum_ece_12 = 0.0f64;

    let mut active_hc_precisions: Vec<(String, String, f64)> = Vec::new();
    let mut pooled_predictions: Vec<(f64, bool)> = Vec::new();

    let mut micro_assignable = 0usize;
    let mut micro_assigned = 0usize;
    let mut micro_correct = 0usize;
    let mut micro_hc_assigned = 0usize;
    let mut micro_hc_correct = 0usize;

    for rec in &records {
        if let EntryBenchmarkOutcome::Succeeded(m) = &rec.outcome {
            n_ok += 1;
            sum_acc += m.mean_accuracy;
            sum_rec += m.mean_recall;
            sum_hcr += m.mean_hc_recall;
            sum_ece += m.reliability.ece;
            sum_ece_10 += m.reliability.ece_10;
            sum_debiased_ece_10 += m.reliability.debiased_ece_10;
            sum_ece_10_floor += m.reliability.ece_10_bernoulli_floor;
            sum_ece_12 += m.reliability.ece_12;
            pooled_predictions.extend_from_slice(&m.predictions);

            if let Some(hcp) = m.reliability.hc_precision {
                n_hc_active_proteins += 1;
                sum_active_hcp += hcp;
                active_hc_precisions.push((
                    rec.manifest_entry.bmrb_id.clone(),
                    rec.manifest_entry.protein.clone(),
                    hcp,
                ));
            } else {
                n_hc_abstained_proteins += 1;
            }

            micro_assignable += m.total_assignable;
            micro_assigned += m.total_assigned;
            micro_correct += m.total_correct;
            micro_hc_assigned += m.total_hc_assigned;
            micro_hc_correct += m.total_hc_correct;
        }
    }

    active_hc_precisions.sort_by(|a, b| a.2.total_cmp(&b.2).then_with(|| a.0.cmp(&b.0)));
    let worst_protein_hc_precision = active_hc_precisions.first().cloned();
    let median_protein_hc_precision = if active_hc_precisions.is_empty() {
        None
    } else {
        let mid = active_hc_precisions.len() / 2;
        if active_hc_precisions.len().is_multiple_of(2) {
            Some(0.5 * (active_hc_precisions[mid - 1].2 + active_hc_precisions[mid].2))
        } else {
            Some(active_hc_precisions[mid].2)
        }
    };

    let denom_ok = if n_ok > 0 { n_ok as f64 } else { 1.0 };
    let pooled_reliability = ProteinReliabilityCurve::from_predictions_with_repeats(
        &pooled_predictions,
        hc_threshold,
        seeds.max(1) as f64,
    );

    RealBenchmarkSummary {
        split: split.to_string(),
        solver_label,
        seeds,
        hc_threshold,
        records,
        n_hc_active_proteins,
        n_hc_abstained_proteins,
        worst_protein_hc_precision,
        median_protein_hc_precision,
        mean_protein_ece: if n_ok > 0 { sum_ece / denom_ok } else { 0.0 },
        mean_protein_ece_10: if n_ok > 0 { sum_ece_10 / denom_ok } else { 0.0 },
        mean_protein_debiased_ece_10: if n_ok > 0 {
            sum_debiased_ece_10 / denom_ok
        } else {
            0.0
        },
        mean_protein_ece_10_bernoulli_floor: if n_ok > 0 {
            sum_ece_10_floor / denom_ok
        } else {
            0.0
        },
        mean_protein_ece_12: if n_ok > 0 { sum_ece_12 / denom_ok } else { 0.0 },
        pooled_reliability,
        macro_accuracy: if n_ok > 0 { sum_acc / denom_ok } else { 0.0 },
        macro_recall: if n_ok > 0 { sum_rec / denom_ok } else { 0.0 },
        macro_hc_precision: if n_hc_active_proteins > 0 {
            sum_active_hcp / (n_hc_active_proteins as f64)
        } else {
            0.0
        },
        macro_hc_recall: if n_ok > 0 { sum_hcr / denom_ok } else { 0.0 },
        micro_assignable,
        micro_assigned,
        micro_correct,
        micro_hc_assigned,
        micro_hc_correct,
        micro_accuracy: if micro_assigned > 0 {
            (micro_correct as f64) / (micro_assigned as f64)
        } else {
            0.0
        },
        micro_recall: if micro_assignable > 0 {
            (micro_correct as f64) / (micro_assignable as f64)
        } else {
            0.0
        },
        micro_hc_precision: if micro_hc_assigned > 0 {
            (micro_hc_correct as f64) / (micro_hc_assigned as f64)
        } else {
            0.0
        },
        micro_hc_recall: if micro_assignable > 0 {
            (micro_hc_correct as f64) / (micro_assignable as f64)
        } else {
            0.0
        },
    }
}

fn format_reliability_bins_inline(rel: &ProteinReliabilityCurve) -> String {
    rel.bins
        .iter()
        .enumerate()
        .map(|(idx, b)| {
            let close = if idx + 1 == rel.bins.len() { ']' } else { ')' };
            if b.count > 0 {
                format!(
                    "[{:.1}-{:.1}{close}: {}/{} ({:.1}%)",
                    b.lower,
                    b.upper,
                    b.correct,
                    b.count,
                    b.empirical_acc * 100.0
                )
            } else {
                format!("[{:.1}-{:.1}{close}: 0/0 (-)", b.lower, b.upper)
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

pub fn format_real_benchmark_report(summary: &RealBenchmarkSummary) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let _ = writeln!(
        out,
        "=== Real Time-Domain Benchmark (split={}, solver={}, seeds={}) ===",
        summary.split, summary.solver_label, summary.seeds
    );
    let _ = writeln!(
        out,
        "{:<16} {:<14} {:>4} {:<7} {:<7} | {:>8} {:>8} {:>8} | {:<40}",
        "BMRB",
        "Protein",
        "Res",
        "Vendor",
        "QC",
        "Accuracy",
        "Recall",
        "HC-Prec",
        "Global Offset (pred - truth) / Status"
    );
    let _ = writeln!(out, "{}", "-".repeat(132));

    let mut n_ok = 0usize;
    let mut n_skip = 0usize;
    let mut n_fail = 0usize;

    for rec in &summary.records {
        let e = &rec.manifest_entry;
        match &rec.outcome {
            EntryBenchmarkOutcome::SkippedMissingData { expected_path } => {
                n_skip += 1;
                let _ = writeln!(
                    out,
                    "{:<16} {:<14} {:>4} {:<7} {:<7} | {:>8} {:>8} {:>8} | SKIPPED (missing local data: {})",
                    e.bmrb_id,
                    truncate_col(&e.protein, 14),
                    e.residues,
                    truncate_col(&e.vendor, 7),
                    "-",
                    "-",
                    "-",
                    "-",
                    expected_path
                );
            }
            EntryBenchmarkOutcome::Failed(diag) => {
                n_fail += 1;
                let _ = writeln!(
                    out,
                    "{:<16} {:<14} {:>4} {:<7} {:<7} | {:>8} {:>8} {:>8} | FAILED [{}]",
                    e.bmrb_id,
                    truncate_col(&e.protein, 14),
                    e.residues,
                    truncate_col(&diag.vendor, 7),
                    "ERR",
                    "ERR",
                    "ERR",
                    "ERR",
                    diag.stage
                );
            }
            EntryBenchmarkOutcome::Succeeded(m) => {
                n_ok += 1;
                let hc_col = if m.reliability.is_abstained() {
                    format!("{:>8}", "ABSTAIN")
                } else {
                    format!("{:>7.1}%", m.mean_hc_precision * 100.0)
                };
                let off_str = format!(
                    "{} ({:.1}s)",
                    m.detailed_offsets.format_compact(),
                    m.elapsed_secs
                );
                let _ = writeln!(
                    out,
                    "{:<16} {:<14} {:>4} {:<7} {:<7} | {:>7.1}% {:>7.1}% {} | {}",
                    e.bmrb_id,
                    truncate_col(&e.protein, 14),
                    e.residues,
                    truncate_col(&e.vendor, 7),
                    m.qc_report.verdict.as_str(),
                    m.mean_accuracy * 100.0,
                    m.mean_recall * 100.0,
                    hc_col,
                    off_str
                );
            }
        }
    }

    let _ = writeln!(out, "{}", "-".repeat(132));
    if n_ok > 0 {
        let macro_hc_col = if summary.n_hc_active_proteins == 0 {
            format!("{:>8}", "ABSTAIN")
        } else {
            format!("{:>7.1}%", summary.macro_hc_precision * 100.0)
        };
        let micro_hc_col = if summary.micro_hc_assigned == 0 {
            format!("{:>8}", "ABSTAIN")
        } else {
            format!("{:>7.1}%", summary.micro_hc_precision * 100.0)
        };
        let _ = writeln!(
            out,
            "MACRO MEAN ({n_ok} evaluated)                           | {:>7.1}% {:>7.1}% {} | HC-Recall={:.1}% (HC-active={}/{n_ok}, abstained={})",
            summary.macro_accuracy * 100.0,
            summary.macro_recall * 100.0,
            macro_hc_col,
            summary.macro_hc_recall * 100.0,
            summary.n_hc_active_proteins,
            summary.n_hc_abstained_proteins,
        );
        let _ = writeln!(
            out,
            "MICRO POOL ({}/{:<5} correct/assigned)              | {:>7.1}% {:>7.1}% {} | HC={}/{} (assignable={})",
            summary.micro_correct,
            summary.micro_assigned,
            summary.micro_accuracy * 100.0,
            summary.micro_recall * 100.0,
            micro_hc_col,
            summary.micro_hc_correct,
            summary.micro_hc_assigned,
            summary.micro_assignable,
        );

        let mut cohort_order: Vec<String> = Vec::new();
        for rec in &summary.records {
            if !cohort_order.contains(&rec.manifest_entry.cohort) {
                cohort_order.push(rec.manifest_entry.cohort.clone());
            }
        }
        if cohort_order.len() > 1 {
            let _ = writeln!(out, "\nCohort Breakdown ({}):", cohort_order.join(" vs "));
            for coh in &cohort_order {
                let sub_recs: Vec<EntryBenchmarkRecord> = summary
                    .records
                    .iter()
                    .filter(|r| &r.manifest_entry.cohort == coh)
                    .cloned()
                    .collect();
                let sub = summarize_real_benchmark_records(
                    coh,
                    summary.solver_label.clone(),
                    summary.seeds,
                    summary.hc_threshold,
                    sub_recs,
                );
                let sub_ok = sub
                    .records
                    .iter()
                    .filter(|r| matches!(r.outcome, EntryBenchmarkOutcome::Succeeded(_)))
                    .count();
                let sub_hc = if sub.n_hc_active_proteins == 0 {
                    "ABSTAIN".to_string()
                } else {
                    format!("{:.1}%", sub.macro_hc_precision * 100.0)
                };
                let _ = writeln!(
                    out,
                    "  - cohort={:<12} ({}/{} evaluated): Macro Acc={:.1}%, Recall={:.1}%, HC-Prec={}, HC-Recall={:.1}%, Mean ECE4/10/12={:.3}/{:.3}/{:.3}, Pooled ECE4/10/12={:.3}/{:.3}/{:.3}",
                    coh,
                    sub_ok,
                    sub.records.len(),
                    sub.macro_accuracy * 100.0,
                    sub.macro_recall * 100.0,
                    sub_hc,
                    sub.macro_hc_recall * 100.0,
                    sub.mean_protein_ece,
                    sub.mean_protein_ece_10,
                    sub.mean_protein_ece_12,
                    sub.pooled_reliability.ece,
                    sub.pooled_reliability.ece_10,
                    sub.pooled_reliability.ece_12,
                );
            }
        }

        let _ = writeln!(
            out,
            "\nPer-Protein Confidence Calibration & Reliability Curves (threshold >= {:.2}):",
            summary.hc_threshold
        );
        match (
            &summary.worst_protein_hc_precision,
            summary.median_protein_hc_precision,
        ) {
            (Some((w_id, w_prot, w_hcp)), Some(med_hcp)) => {
                let _ = writeln!(
                    out,
                    "  Per-protein HC-Prec : worst={} ({}) {:.1}%, median={:.1}% across {} HC-active protein(s); abstained={}/{n_ok}",
                    w_prot,
                    w_id,
                    w_hcp * 100.0,
                    med_hcp * 100.0,
                    summary.n_hc_active_proteins,
                    summary.n_hc_abstained_proteins,
                );
            }
            _ => {
                let _ = writeln!(
                    out,
                    "  Per-protein HC-Prec : all {n_ok} evaluated protein(s) ABSTAINED (0 HC calls >= {:.2})",
                    summary.hc_threshold
                );
            }
        }
        let _ = writeln!(
            out,
            "  Calibration (ECE)   : mean per-protein ECE4/10/12={:.3}/{:.3}/{:.3} (debiased ECE10={:.3}, Bernoulli floor={:.3}), pooled ECE4/10/12={:.3}/{:.3}/{:.3}",
            summary.mean_protein_ece,
            summary.mean_protein_ece_10,
            summary.mean_protein_ece_12,
            summary.mean_protein_debiased_ece_10,
            summary.mean_protein_ece_10_bernoulli_floor,
            summary.pooled_reliability.ece,
            summary.pooled_reliability.ece_10,
            summary.pooled_reliability.ece_12,
        );
        let _ = writeln!(
            out,
            "  Pooled reliability  : {}",
            format_reliability_bins_inline(&summary.pooled_reliability)
        );
        for rec in &summary.records {
            if let EntryBenchmarkOutcome::Succeeded(m) = &rec.outcome {
                let qc_note = if m.qc_report.reasons.is_empty() {
                    String::new()
                } else {
                    format!(" | QC notes: {}", m.qc_report.reasons.join("; "))
                };
                let _ = writeln!(
                    out,
                    "  - {:<16} ({:<12}): QC={:<7} | HC={:<22} | ECE={:.3} (ECE10={:.3}, ECE12={:.3}) | {}{}",
                    rec.manifest_entry.bmrb_id,
                    truncate_col(&rec.manifest_entry.protein, 12),
                    m.qc_report.verdict.as_str(),
                    m.reliability.hc_status,
                    m.reliability.ece,
                    m.reliability.ece_10,
                    m.reliability.ece_12,
                    format_reliability_bins_inline(&m.reliability),
                    qc_note
                );
            }
        }
    } else {
        let _ = writeln!(
            out,
            "SUMMARY: 0 evaluated, {n_skip} SKIPPED (missing local data), {n_fail} failed"
        );
    }

    if n_fail > 0 {
        let _ = writeln!(out, "\nFailure Diagnostics:");
        for rec in &summary.records {
            if let EntryBenchmarkOutcome::Failed(d) = &rec.outcome {
                let _ = writeln!(
                    out,
                    "  - {} ({}): stage={}, exp={}, vendor={}, quad_mode={}, nus={} -> {}",
                    rec.manifest_entry.bmrb_id,
                    rec.manifest_entry.protein,
                    d.stage,
                    d.experiment.as_deref().unwrap_or("-"),
                    d.vendor,
                    d.quad_mode,
                    d.nus_status,
                    d.reason
                );
            }
        }
    }
    out
}

pub fn print_real_benchmark_report(summary: &RealBenchmarkSummary) {
    print!("{}", format_real_benchmark_report(summary));
}

fn truncate_col(s: &str, width: usize) -> String {
    if s.chars().count() <= width {
        s.to_string()
    } else {
        s.chars().take(width).collect()
    }
}

pub(crate) fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 4);
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn offset_status_slug(status: OffsetFitStatus) -> &'static str {
    match status {
        OffsetFitStatus::Fitted => "fitted",
        OffsetFitStatus::NotSignificant => "not_significant",
        OffsetFitStatus::TooFewValues => "too_few_values",
    }
}

fn render_nucleus_offset_json(label: &str, fit: &NucleusOffsetFit) -> String {
    format!(
        "{{ \"status\": \"{}\", \"compact\": \"{}\", \"applied_ppm\": {:.4}, \"candidate_ppm\": {:.4}, \"log_likelihood_gain\": {:.3}, \"n_values\": {} }}",
        offset_status_slug(fit.status),
        json_escape(&fit.format_compact(label)),
        fit.applied_ppm,
        fit.candidate_ppm,
        fit.log_likelihood_gain,
        fit.n_values,
    )
}

pub(crate) fn render_detailed_offsets_json(d: &DetailedReferenceOffsets, indent: usize) -> String {
    let pad = " ".repeat(indent);
    let pad2 = " ".repeat(indent + 2);
    format!(
        "{{\n\
         {pad2}\"compact\": \"{}\",\n\
         {pad2}\"H\": {},\n\
         {pad2}\"N\": {},\n\
         {pad2}\"CA\": {},\n\
         {pad2}\"CB\": {},\n\
         {pad2}\"C\": {}\n\
         {pad}}}",
        json_escape(&d.format_compact()),
        render_nucleus_offset_json("dH", &d.h),
        render_nucleus_offset_json("dN", &d.n),
        render_nucleus_offset_json("dCA", &d.ca),
        render_nucleus_offset_json("dCB", &d.cb),
        render_nucleus_offset_json("dC", &d.c),
    )
}

pub(crate) fn render_qc_report_json(qc: &DatasetQcReport, indent: usize) -> String {
    let pad = " ".repeat(indent);
    let pad2 = " ".repeat(indent + 2);
    let reasons_json = qc
        .reasons
        .iter()
        .map(|r| format!("\"{}\"", json_escape(r)))
        .collect::<Vec<_>>()
        .join(", ");
    let cal_warn_json = match &qc.calibration_warning {
        Some(w) => format!("\"{}\"", json_escape(w)),
        None => "null".to_string(),
    };
    format!(
        "{{\n\
         {pad2}\"verdict\": \"{}\",\n\
         {pad2}\"confidence_calibrated\": {},\n\
         {pad2}\"warn_sub_hc_uncalibrated\": {},\n\
         {pad2}\"calibration_warning\": {},\n\
         {pad2}\"confidence_scale\": {:.4},\n\
         {pad2}\"confidence_cap\": {:.4},\n\
         {pad2}\"already_applied\": {},\n\
         {pad2}\"n_assignable_amides\": {},\n\
         {pad2}\"n_raw_hsqc_roots\": {},\n\
         {pad2}\"n_amide_hsqc_roots\": {},\n\
         {pad2}\"hsqc_3d_support_frac\": {:.6},\n\
         {pad2}\"supported_roots_per_assignable\": {:.6},\n\
         {pad2}\"aliphatic_support_frac\": {:.6},\n\
         {pad2}\"resolved_bidir_aliphatic_frac\": {:.6},\n\
         {pad2}\"hsqc_3d_rms_dh_ppm\": {:.6},\n\
         {pad2}\"hsqc_3d_rms_dn_ppm\": {:.6},\n\
         {pad2}\"hsqc_multiplicity_ratio\": {:.6},\n\
         {pad2}\"raw_hsqc_multiplicity_ratio\": {:.6},\n\
         {pad2}\"offset_consistency_ok\": {},\n\
         {pad2}\"connectivity_frac\": {:.6},\n\
         {pad2}\"corroborated_connectivity_frac\": {:.6},\n\
         {pad2}\"chain_corroborated_frac\": {:.6},\n\
         {pad2}\"unary_specificity\": {:.6},\n\
         {pad2}\"solution_margin\": {:.6},\n\
         {pad2}\"hc_call_fraction\": {:.6},\n\
         {pad2}\"protein_hc_gate_triggered\": {},\n\
         {pad2}\"truncated_fid_zero_filled\": {},\n\
         {pad2}\"inter_exp_c13_offset_warn\": {},\n\
         {pad2}\"max_inter_exp_ca_drift_ppm\": {:.6},\n\
         {pad2}\"reasons\": [{}],\n\
         {pad2}\"detailed_offsets\": {}\n\
         {pad}}}",
        qc.verdict.as_str(),
        qc.confidence_calibrated,
        qc.warn_sub_hc_uncalibrated,
        cal_warn_json,
        qc.confidence_scale,
        qc.confidence_cap,
        qc.already_applied,
        qc.n_assignable_amides,
        qc.n_raw_hsqc_roots,
        qc.n_amide_hsqc_roots,
        qc.hsqc_3d_support_frac,
        qc.supported_roots_per_assignable,
        qc.aliphatic_support_frac,
        qc.resolved_bidir_aliphatic_frac,
        qc.hsqc_3d_rms_dh_ppm,
        qc.hsqc_3d_rms_dn_ppm,
        qc.hsqc_multiplicity_ratio,
        qc.raw_hsqc_multiplicity_ratio,
        qc.offset_consistency_ok,
        qc.connectivity_frac,
        qc.corroborated_connectivity_frac,
        qc.chain_corroborated_frac,
        qc.unary_specificity,
        qc.solution_margin,
        qc.hc_call_fraction,
        qc.protein_hc_gate_triggered,
        qc.truncated_fid_zero_filled,
        qc.inter_exp_c13_offset_warn,
        qc.max_inter_exp_ca_drift_ppm,
        reasons_json,
        render_detailed_offsets_json(&qc.detailed_offsets, indent + 2),
    )
}

pub(crate) fn render_reliability_curve_json(
    rel: &ProteinReliabilityCurve,
    indent: usize,
) -> String {
    let pad = " ".repeat(indent);
    let pad2 = " ".repeat(indent + 2);
    let pad4 = " ".repeat(indent + 4);
    let hc_prec_json = match rel.hc_precision {
        Some(v) => format!("{v:.6}"),
        None => "null".to_string(),
    };
    let bins_json = rel
        .bins
        .iter()
        .enumerate()
        .map(|(idx, b)| {
            let close = if idx + 1 == rel.bins.len() { ']' } else { ')' };
            format!(
                "{pad4}{{ \"range\": \"[{:.1}, {:.1}{close}\", \"lower\": {:.2}, \"upper\": {:.2}, \"count\": {}, \"correct\": {}, \"mean_conf\": {:.6}, \"empirical_acc\": {:.6} }}",
                b.lower, b.upper, b.lower, b.upper, b.count, b.correct, b.mean_conf, b.empirical_acc
            )
        })
        .collect::<Vec<_>>()
        .join(",\n");
    format!(
        "{{\n\
         {pad2}\"total_predictions\": {},\n\
         {pad2}\"ece\": {:.6},\n\
         {pad2}\"ece_10\": {:.6},\n\
         {pad2}\"debiased_ece_10\": {:.6},\n\
         {pad2}\"ece_10_bernoulli_floor\": {:.6},\n\
         {pad2}\"ece_12\": {:.6},\n\
         {pad2}\"hc_threshold\": {:.4},\n\
         {pad2}\"abstained\": {},\n\
         {pad2}\"hc_assigned\": {},\n\
         {pad2}\"hc_correct\": {},\n\
         {pad2}\"hc_precision\": {},\n\
         {pad2}\"hc_status\": \"{}\",\n\
         {pad2}\"bins\": [\n{bins_json}\n{pad2}]\n\
         {pad}}}",
        rel.total_predictions,
        rel.ece,
        rel.ece_10,
        rel.debiased_ece_10,
        rel.ece_10_bernoulli_floor,
        rel.ece_12,
        rel.hc_threshold,
        rel.is_abstained(),
        rel.hc_assigned,
        rel.hc_correct,
        hc_prec_json,
        json_escape(&rel.hc_status),
    )
}

pub fn render_summary_json(summary: &RealBenchmarkSummary) -> String {
    let n_total = summary.records.len();
    let n_ok = summary
        .records
        .iter()
        .filter(|r| matches!(r.outcome, EntryBenchmarkOutcome::Succeeded(_)))
        .count();
    let n_skip = summary
        .records
        .iter()
        .filter(|r| matches!(r.outcome, EntryBenchmarkOutcome::SkippedMissingData { .. }))
        .count();
    let n_fail = summary
        .records
        .iter()
        .filter(|r| matches!(r.outcome, EntryBenchmarkOutcome::Failed(_)))
        .count();

    let macro_hcp_json = if summary.n_hc_active_proteins > 0 {
        format!("{:.6}", summary.macro_hc_precision)
    } else {
        "null".to_string()
    };
    let micro_hcp_json = if summary.micro_hc_assigned > 0 {
        format!("{:.6}", summary.micro_hc_precision)
    } else {
        "null".to_string()
    };
    let worst_hcp_json = match &summary.worst_protein_hc_precision {
        Some((bid, prot, val)) => format!(
            "{{ \"bmrb_id\": \"{}\", \"protein\": \"{}\", \"hc_precision\": {:.6} }}",
            json_escape(bid),
            json_escape(prot),
            val
        ),
        None => "null".to_string(),
    };
    let median_hcp_json = match summary.median_protein_hc_precision {
        Some(val) => format!("{val:.6}"),
        None => "null".to_string(),
    };

    let mut j = String::new();
    j.push_str("{\n");
    j.push_str(&format!(
        "  \"tool\": \"{}\",\n  \"split\": \"{}\",\n  \"solver\": \"{}\",\n  \"seeds\": {},\n  \"hc_threshold\": {:.4},\n",
        json_escape(&version_string()),
        json_escape(&summary.split),
        json_escape(&summary.solver_label),
        summary.seeds,
        summary.hc_threshold,
    ));
    j.push_str(&format!(
        "  \"counts\": {{ \"total\": {n_total}, \"evaluated\": {n_ok}, \"hc_active_proteins\": {}, \"hc_abstained_proteins\": {}, \"skipped_missing_local_data\": {n_skip}, \"failed\": {n_fail} }},\n",
        summary.n_hc_active_proteins,
        summary.n_hc_abstained_proteins,
    ));
    j.push_str(&format!(
        "  \"macro_average\": {{ \"accuracy\": {:.6}, \"recall\": {:.6}, \"hc_precision\": {}, \"hc_recall\": {:.6}, \"hc_active_proteins\": {}, \"hc_abstained_proteins\": {} }},\n",
        summary.macro_accuracy,
        summary.macro_recall,
        macro_hcp_json,
        summary.macro_hc_recall,
        summary.n_hc_active_proteins,
        summary.n_hc_abstained_proteins,
    ));
    j.push_str(&format!(
        "  \"micro_average\": {{ \"accuracy\": {:.6}, \"recall\": {:.6}, \"hc_precision\": {}, \"hc_recall\": {:.6}, \"assignable\": {}, \"assigned\": {}, \"correct\": {}, \"hc_assigned\": {}, \"hc_correct\": {} }},\n",
        summary.micro_accuracy,
        summary.micro_recall,
        micro_hcp_json,
        summary.micro_hc_recall,
        summary.micro_assignable,
        summary.micro_assigned,
        summary.micro_correct,
        summary.micro_hc_assigned,
        summary.micro_hc_correct,
    ));
    j.push_str(&format!(
        "  \"calibration_summary\": {{\n    \"hc_active_proteins\": {},\n    \"hc_abstained_proteins\": {},\n    \"worst_protein_hc_precision\": {},\n    \"median_protein_hc_precision\": {},\n    \"mean_protein_ece\": {:.6},\n    \"mean_protein_ece_10\": {:.6},\n    \"mean_protein_debiased_ece_10\": {:.6},\n    \"mean_protein_ece_10_bernoulli_floor\": {:.6},\n    \"mean_protein_ece_12\": {:.6},\n    \"pooled_reliability_curve\": {}\n  }},\n",
        summary.n_hc_active_proteins,
        summary.n_hc_abstained_proteins,
        worst_hcp_json,
        median_hcp_json,
        summary.mean_protein_ece,
        summary.mean_protein_ece_10,
        summary.mean_protein_debiased_ece_10,
        summary.mean_protein_ece_10_bernoulli_floor,
        summary.mean_protein_ece_12,
        render_reliability_curve_json(&summary.pooled_reliability, 4),
    ));
    let mut cohort_order: Vec<String> = Vec::new();
    for rec in &summary.records {
        if !cohort_order.contains(&rec.manifest_entry.cohort) {
            cohort_order.push(rec.manifest_entry.cohort.clone());
        }
    }
    let cohort_json_items: Vec<String> = cohort_order
        .iter()
        .map(|coh| {
            let sub_recs: Vec<EntryBenchmarkRecord> = summary
                .records
                .iter()
                .filter(|r| &r.manifest_entry.cohort == coh)
                .cloned()
                .collect();
            let sub = summarize_real_benchmark_records(
                coh,
                summary.solver_label.clone(),
                summary.seeds,
                summary.hc_threshold,
                sub_recs,
            );
            let sub_ok = sub
                .records
                .iter()
                .filter(|r| matches!(r.outcome, EntryBenchmarkOutcome::Succeeded(_)))
                .count();
            let sub_hcp = if sub.n_hc_active_proteins > 0 {
                format!("{:.6}", sub.macro_hc_precision)
            } else {
                "null".to_string()
            };
            format!(
                "    {{ \"cohort\": \"{}\", \"total\": {}, \"evaluated\": {}, \"macro_accuracy\": {:.6}, \"macro_recall\": {:.6}, \"macro_hc_precision\": {}, \"macro_hc_recall\": {:.6}, \"hc_active_proteins\": {}, \"hc_abstained_proteins\": {}, \"mean_protein_ece\": {:.6}, \"mean_protein_ece_10\": {:.6}, \"mean_protein_debiased_ece_10\": {:.6}, \"mean_protein_ece_10_bernoulli_floor\": {:.6}, \"mean_protein_ece_12\": {:.6}, \"pooled_ece\": {:.6}, \"pooled_ece_10\": {:.6}, \"pooled_ece_12\": {:.6} }}",
                json_escape(coh),
                sub.records.len(),
                sub_ok,
                sub.macro_accuracy,
                sub.macro_recall,
                sub_hcp,
                sub.macro_hc_recall,
                sub.n_hc_active_proteins,
                sub.n_hc_abstained_proteins,
                sub.mean_protein_ece,
                sub.mean_protein_ece_10,
                sub.mean_protein_debiased_ece_10,
                sub.mean_protein_ece_10_bernoulli_floor,
                sub.mean_protein_ece_12,
                sub.pooled_reliability.ece,
                sub.pooled_reliability.ece_10,
                sub.pooled_reliability.ece_12,
            )
        })
        .collect();
    j.push_str(&format!(
        "  \"cohort_summaries\": [\n{}\n  ],\n",
        cohort_json_items.join(",\n")
    ));
    j.push_str("  \"entries\": [\n");
    for (idx, rec) in summary.records.iter().enumerate() {
        let comma = if idx + 1 < summary.records.len() {
            ","
        } else {
            ""
        };
        let e = &rec.manifest_entry;
        j.push_str("    {\n");
        j.push_str(&format!(
            "      \"bmrb_id\": \"{}\",\n      \"protein\": \"{}\",\n      \"residues\": {},\n      \"field_mhz\": {:.2},\n      \"vendor\": \"{}\",\n      \"split\": \"{}\",\n      \"cohort\": \"{}\",\n",
            json_escape(&e.bmrb_id),
            json_escape(&e.protein),
            e.residues,
            e.field_mhz,
            json_escape(&e.vendor),
            json_escape(&e.split),
            json_escape(&e.cohort),
        ));
        match &rec.outcome {
            EntryBenchmarkOutcome::SkippedMissingData { expected_path } => {
                j.push_str(&format!(
                    "      \"status\": \"SKIPPED (missing local data)\",\n      \"expected_path\": \"{}\"\n",
                    json_escape(expected_path)
                ));
            }
            EntryBenchmarkOutcome::Failed(d) => {
                let exp_json = match &d.experiment {
                    Some(ex) => format!("\"{}\"", json_escape(ex)),
                    None => "null".to_string(),
                };
                j.push_str(&format!(
                    "      \"status\": \"FAILED\",\n      \"failure\": {{ \"stage\": \"{}\", \"experiment\": {}, \"vendor\": \"{}\", \"quad_mode\": \"{}\", \"nus\": \"{}\", \"reason\": \"{}\" }}\n",
                    json_escape(&d.stage),
                    exp_json,
                    json_escape(&d.vendor),
                    json_escape(&d.quad_mode),
                    json_escape(&d.nus_status),
                    json_escape(&d.reason),
                ));
            }
            EntryBenchmarkOutcome::Succeeded(m) => {
                let exps_json = m
                    .discovered_experiments
                    .iter()
                    .map(|(k, p)| {
                        format!(
                            "{{\"exp\":\"{}\",\"path\":\"{}\"}}",
                            json_escape(k),
                            json_escape(p)
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                let audits_json = m
                    .axis_rescue_audits
                    .iter()
                    .map(|a| {
                        format!(
                            "{{\"experiment\":\"{}\",\"selected\":\"{:?}\",\"enabled\":{},\"applied\":{},\"polarity_flipped\":{},\"ln_bayes_factor_vs_canonical\":{:.4},\"canonical_matched_fraction\":{:.6},\"selected_matched_fraction\":{:.6}}}",
                            json_escape(&a.experiment),
                            a.selected,
                            a.enabled,
                            a.applied,
                            a.polarity_flipped,
                            a.ln_bayes_factor_vs_canonical,
                            a.canonical_matched_fraction,
                            a.selected_matched_fraction,
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                let entry_hcp_json = match m.reliability.hc_precision {
                    Some(v) => format!("{v:.6}"),
                    None => "null".to_string(),
                };
                j.push_str(&format!(
                    "      \"status\": \"OK\",\n      \"qc_verdict\": \"{}\",\n      \"discovered_experiments\": [{}],\n      \"axis_rescue_audits\": [{}],\n      \"hsqc_roots\": {},\n      \"total_peaks\": {},\n      \"assignable_per_seed\": {},\n      \"accuracy\": {:.6},\n      \"recall\": {:.6},\n      \"hc_abstained\": {},\n      \"hc_precision\": {},\n      \"hc_status\": \"{}\",\n      \"hc_recall\": {:.6},\n      \"global_offsets_ppm\": {{ \"H\": {:.4}, \"N\": {:.4}, \"CA\": {:.4}, \"CB\": {:.4}, \"C\": {:.4} }},\n      \"detailed_offsets_compact\": \"{}\",\n      \"detailed_offsets\": {},\n      \"prior_detailed_offsets\": {},\n      \"qc_report\": {},\n      \"reliability_curve\": {},\n      \"elapsed_secs\": {:.3}\n",
                    m.qc_report.verdict.as_str(),
                    exps_json,
                    audits_json,
                    m.n_hsqc_roots,
                    m.n_total_peaks,
                    m.assignable_per_seed,
                    m.mean_accuracy,
                    m.mean_recall,
                    m.reliability.is_abstained(),
                    entry_hcp_json,
                    json_escape(&m.reliability.hc_status),
                    m.mean_hc_recall,
                    m.global_offsets.h,
                    m.global_offsets.n,
                    m.global_offsets.ca,
                    m.global_offsets.cb,
                    m.global_offsets.c,
                    json_escape(&m.detailed_offsets.format_compact()),
                    render_detailed_offsets_json(&m.detailed_offsets, 6),
                    render_detailed_offsets_json(&m.prior_detailed_offsets, 6),
                    render_qc_report_json(&m.qc_report, 6),
                    render_reliability_curve_json(&m.reliability, 6),
                    m.elapsed_secs,
                ));
            }
        }
        j.push_str(&format!("    }}{comma}\n"));
    }
    j.push_str("  ]\n}\n");
    j
}

pub fn parse_real_benchmark_args(args: &[String]) -> Result<RealBenchmarkConfig, String> {
    let mut manifest_path = PathBuf::from("data/benchmark/real_manifest.tsv");
    let mut split_opt: Option<BenchmarkSplitFilter> = None;
    let mut cohort_opt: Option<String> = None;
    let mut solver = SolverKind::EvoGraph;
    let mut engine_opt: Option<EngineKind> = None;
    let mut seeds = 5usize;
    let mut base_seed = 0u64;
    let mut data_root: Option<PathBuf> = None;
    let mut priors_path: Option<String> = None;
    let mut runs_opt: Option<usize> = None;
    let mut steps_opt: Option<usize> = None;
    let mut hc_threshold = 0.8f64;
    let mut no_crop = false;
    let mut lp = false;
    let mut no_axis_rescue = false;
    let mut allow_truncated_fid = false;
    let mut discover_only = false;
    let mut json_out: Option<PathBuf> = None;
    let mut verbose = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--manifest" => {
                i += 1;
                manifest_path = PathBuf::from(args.get(i).ok_or("missing value for --manifest")?);
            }
            "--split" => {
                i += 1;
                split_opt = Some(BenchmarkSplitFilter::parse(
                    args.get(i).ok_or("missing value for --split")?,
                )?);
            }
            "--cohort" => {
                i += 1;
                let raw_coh = args.get(i).ok_or("missing value for --cohort")?.trim();
                if raw_coh.is_empty() {
                    return Err("--cohort must be non-empty".to_string());
                }
                cohort_opt = Some(raw_coh.to_string());
            }
            "--solver" => {
                i += 1;
                solver = parse_solver(args.get(i).ok_or("missing value for --solver")?)?;
            }
            "--engine" => {
                i += 1;
                engine_opt = Some(parse_engine(
                    args.get(i).ok_or("missing value for --engine")?,
                )?);
            }
            "--seeds" => {
                i += 1;
                seeds = args
                    .get(i)
                    .ok_or("missing value for --seeds")?
                    .parse::<usize>()
                    .map_err(|e| format!("{e}"))?
                    .max(1);
            }
            "--seed" => {
                i += 1;
                base_seed = args
                    .get(i)
                    .ok_or("missing value for --seed")?
                    .parse::<u64>()
                    .map_err(|e| format!("{e}"))?;
            }
            "--data-root" => {
                i += 1;
                data_root = Some(PathBuf::from(
                    args.get(i).ok_or("missing value for --data-root")?,
                ));
            }
            "--priors" => {
                i += 1;
                priors_path = args.get(i).cloned();
            }
            "--runs" => {
                i += 1;
                runs_opt = Some(parse_bounded_runs(
                    args.get(i).ok_or("missing value for --runs")?,
                )?);
            }
            "--steps" => {
                i += 1;
                steps_opt = Some(parse_bounded_steps(
                    args.get(i).ok_or("missing value for --steps")?,
                )?);
            }
            "--hc" => {
                i += 1;
                hc_threshold = args
                    .get(i)
                    .ok_or("missing value for --hc")?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
            }
            "--json-out" => {
                i += 1;
                json_out = Some(PathBuf::from(
                    args.get(i).ok_or("missing value for --json-out")?,
                ));
            }
            "--no-crop" => no_crop = true,
            "--lp" => lp = true,
            "--no-axis-rescue" => no_axis_rescue = true,
            "--allow-truncated-fid" => allow_truncated_fid = true,
            "--discover-only" => discover_only = true,
            "-v" | "--verbose" => verbose = true,
            other => return Err(format!("unknown real-benchmark flag {other:?}")),
        }
        i += 1;
    }

    let split = split_opt.ok_or("real-benchmark requires --split dev|heldout")?;
    let allow_repo_fallbacks = data_root.is_none();
    let effective_data_root = data_root.unwrap_or_else(|| PathBuf::from("data/private/bmrb"));

    Ok(RealBenchmarkConfig {
        manifest_path,
        split,
        cohort_opt,
        solver,
        engine_opt,
        seeds,
        base_seed,
        data_root: effective_data_root,
        allow_repo_fallbacks,
        priors_path,
        runs_opt,
        steps_opt,
        hc_threshold,
        no_crop,
        lp,
        no_axis_rescue,
        allow_truncated_fid,
        discover_only,
        json_out,
        verbose,
    })
}

pub fn cmd_real_benchmark(args: &[String]) -> Result<(), String> {
    let cfg = parse_real_benchmark_args(args)?;
    if cfg.discover_only {
        return run_real_benchmark_discover_only(&cfg).map(|_| ());
    }
    let no_crop = cfg.no_crop;
    let lp = cfg.lp;
    let allow_trunc = cfg.allow_truncated_fid;
    run_real_benchmark_with_loaders_and_warnings(
        &cfg,
        &|dir| {
            process_fid_dir_2d_opts(dir, no_crop, lp, allow_trunc)
                .map(|(s, r)| (s, r.truncated_fid_warning))
        },
        &|dir, exp| {
            process_fid_dir_3d_opts(dir, exp, no_crop, lp, allow_trunc)
                .map(|(s, r)| (s, r.truncated_fid_warning))
        },
    )
    .map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nmr_core::nmrstar::read_star;
    use nmr_core::simulate::{
        render_hsqc_2d, render_triple_3d, simulate, SimConfig, SpectrumRenderConfig,
    };
    use std::process;

    #[test]
    fn test_manifest_parsing_and_missing_local_data_skips_cleanly() {
        let dir = env::temp_dir().join(format!("nmr_real_bench_skip_{}", process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        let manifest = dir.join("real_manifest.tsv");
        fs::write(
            &manifest,
            "bmrb_id\tprotein\tresidues\tfield_mhz\tvendor\texperiments\tsplit\n\
             ubiquitin_local\tUbiquitin\t76\t500\tBruker\tHSQC,HNCACB,CBCACONH,HNCO,HNCACO\tdev\n\
             51439\tCISD3\t63\t900\tBruker\tHSQC,HNCO,HNCA,HNCOCA,CBCACONH\tdev\n\
             15477\tTargetX\t90\t600\tVarian\tHSQC,HNCO,HNCACB,CBCACONH\theldout\n",
        )
        .unwrap();

        let json_out = dir.join("out.json");
        let empty_root = dir.join("empty_data_root");
        let args = vec![
            "--manifest".to_string(),
            manifest.to_string_lossy().into_owned(),
            "--split".to_string(),
            "dev".to_string(),
            "--solver".to_string(),
            "evo-graph".to_string(),
            "--data-root".to_string(),
            empty_root.to_string_lossy().into_owned(),
            "--json-out".to_string(),
            json_out.to_string_lossy().into_owned(),
        ];

        // When local directories are absent, `real-benchmark` marks them `SKIPPED (missing local data)` and returns `Ok(())`.
        cmd_real_benchmark(&args).unwrap();

        let json_txt = fs::read_to_string(&json_out).unwrap();
        assert!(json_txt.contains("\"skipped_missing_local_data\": 2"));
        assert!(json_txt.contains("\"failed\": 0"));
        assert!(json_txt.contains("SKIPPED (missing local data)"));

        // Verify sibling `<repo>/data/ubiquitin` & `<repo>/data/private/bmrb51439_cisd3`
        // resolve when `--data-root` has shape `<repo>/data/private/bmrb`
        let fake_repo_data = dir.join("fake_repo/data");
        let fake_bmrb_root = fake_repo_data.join("private/bmrb");
        fs::create_dir_all(fake_repo_data.join("ubiquitin")).unwrap();
        fs::write(
            fake_repo_data.join("ubiquitin_backbone_assignment_191104.txt"),
            "truth",
        )
        .unwrap();
        fs::create_dir_all(fake_repo_data.join("private/bmrb51439_cisd3")).unwrap();
        fs::create_dir_all(&fake_bmrb_root).unwrap();
        let ubq_sib = resolve_local_entry_paths("ubiquitin_local", &fake_bmrb_root, false).unwrap();
        assert_eq!(ubq_sib.entry_dir, fake_repo_data.join("ubiquitin"));
        assert_eq!(
            ubq_sib.truth_path,
            Some(fake_repo_data.join("ubiquitin_backbone_assignment_191104.txt"))
        );
        let cisd_sib = resolve_local_entry_paths("51439", &fake_bmrb_root, false).unwrap();
        assert_eq!(
            cisd_sib.entry_dir,
            fake_repo_data.join("private/bmrb51439_cisd3")
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_varian_and_uppercase_bruker_case_safe_fid_discovery_and_e2e_scoring() {
        let dir = env::temp_dir().join(format!("nmr_real_bench_disc_{}", process::id()));
        let _ = fs::remove_dir_all(&dir);
        let data_root = dir.join("bmrb");
        let varian_entry = data_root.join("77777");
        fs::create_dir_all(&varian_entry).unwrap();

        let fixture_star = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/fixtures/synthetic_ubiquitin.str");
        fs::copy(&fixture_star, varian_entry.join("bmr77777_3.str")).unwrap();

        // Write Varian BioPack `procpar` + non-empty `fid` subdirectories (`gNhsqc`, `ghnco`, `ghncacb`, `ghn_co_cacb`)
        let write_varian_exp =
            |sub: &str, pslabel: &str, ni: usize, ni2: usize, sw1: f64, sw2: f64| {
                let p = varian_entry.join(sub);
                fs::create_dir_all(&p).unwrap();
                let procpar = format!(
                    "pslabel 2 2 8192 0 0 2 1 1 1 64\n\
                 1 \"{pslabel}\"\n\
                 0\n\
                 tn 2 2 8192 0 0 2 1 1 1 64\n\
                 1 \"H1\"\n\
                 0\n\
                 dn 2 2 8192 0 0 2 1 1 1 64\n\
                 1 \"C13\"\n\
                 0\n\
                 dn2 2 2 8192 0 0 2 1 1 1 64\n\
                 1 \"N15\"\n\
                 0\n\
                 sfrq 1 1 10000 0 0 2 1 1 1 64\n\
                 1 600.133\n\
                 0\n\
                 dfrq 1 1 10000 0 0 2 1 1 1 64\n\
                 1 150.903\n\
                 0\n\
                 dfrq2 1 1 10000 0 0 2 1 1 1 64\n\
                 1 60.818\n\
                 0\n\
                 ni 1 1 10000 0 0 2 1 1 1 64\n\
                 1 {ni}\n\
                 0\n\
                 ni2 1 1 10000 0 0 2 1 1 1 64\n\
                 1 {ni2}\n\
                 0\n\
                 sw1 1 1 100000 0 0 2 1 1 1 64\n\
                 1 {sw1}\n\
                 0\n\
                 sw2 1 1 100000 0 0 2 1 1 1 64\n\
                 1 {sw2}\n\
                 0\n\
                 phase 1 1 10 0 0 2 1 1 1 64\n\
                 2 1 2\n\
                 0\n\
                 phase2 1 1 10 0 0 2 1 1 1 64\n\
                 2 1 2\n\
                 0\n"
                );
                fs::write(p.join("procpar"), procpar).unwrap();
                fs::write(p.join("fid"), b"dummy_raw_fid_payload").unwrap();
            };

        write_varian_exp("exp1_root.fid", "gNhsqc", 128, 1, 2000.0, 0.0);
        write_varian_exp("exp2_co.fid", "ghnco", 48, 32, 2200.0, 2000.0);
        write_varian_exp("exp3_cacb.fid", "ghncacb", 64, 32, 11000.0, 2000.0);
        write_varian_exp("exp4_inter.fid", "ghn_co_cacb", 64, 32, 11000.0, 2000.0);
        // Relaxation distractor that must be ignored
        write_varian_exp("exp5_t1.fid", "gNhsqcT1", 128, 1, 2000.0, 0.0);

        let (disc, _) = discover_entry_experiments(&varian_entry, Some(600.0)).unwrap();
        let names: Vec<&str> = disc.iter().map(|d| d.exp_type.as_str()).collect();
        assert_eq!(names, vec!["HSQC", "HNCO", "HNCACB", "CBCACONH"]);

        // Also verify uppercase UXNMR Bruker (`ACQUS`, `ACQU2S`, `SER`) case-safe discovery
        let bruker_up = data_root.join("88888/EXP_UPPER");
        fs::create_dir_all(&bruker_up).unwrap();
        fs::write(
            bruker_up.join("ACQUS"),
            "##$PULPROG= <hsqcetfpf3gpsi>\n##$NUC1= <1H>\n##$SFO1= 500.13\n##$SW= 14.0\n##$TD= 1024\n##$AQ_mod= 3\n",
        )
        .unwrap();
        fs::write(
            bruker_up.join("ACQU2S"),
            "##$NUC1= <15N>\n##$SFO1= 50.68\n##$SW= 34.0\n##$TD= 128\n##$FnMODE= 6\n",
        )
        .unwrap();
        fs::write(bruker_up.join("SER"), b"raw_bruker_ser").unwrap();
        let (disc_b, _) =
            discover_entry_experiments(&data_root.join("88888"), Some(500.0)).unwrap();
        assert_eq!(disc_b.len(), 1);
        assert_eq!(disc_b[0].exp_type, "HSQC");

        // End-to-end `run_real_benchmark_with_loaders` test with systematic `-2.75 ppm` 13C offset
        let ref_data = read_star(&fixture_star, None).unwrap();
        let (sim_peaks, _) = simulate(
            &ref_data,
            &SimConfig {
                experiments: vec![
                    "HSQC".into(),
                    "HNCO".into(),
                    "HNCACB".into(),
                    "CBCACONH".into(),
                ],
                p_missing_peak: 0.0,
                spurious_rate: 0.0,
                ref_offset_ca: -2.75,
                ref_offset_cb: -2.75,
                ref_offset_c: -2.75,
                seed: 11,
                ..SimConfig::default()
            },
        );
        let rcfg = SpectrumRenderConfig::default();
        let s_hsqc = render_hsqc_2d(&sim_peaks, &rcfg);
        let s_hnco = render_triple_3d("HNCO", &sim_peaks, &rcfg);
        let s_hncacb = render_triple_3d("HNCACB", &sim_peaks, &rcfg);
        let s_cbcaconh = render_triple_3d("CBCACONH", &sim_peaks, &rcfg);

        let manifest = dir.join("manifest.tsv");
        fs::write(
            &manifest,
            "bmrb_id\tprotein\tresidues\tfield_mhz\tvendor\texperiments\tsplit\n\
             77777\tUbqVarian\t76\t600\tVarian\tHSQC,HNCO,HNCACB,CBCACONH\tdev\n",
        )
        .unwrap();

        let cfg = RealBenchmarkConfig {
            manifest_path: manifest,
            split: BenchmarkSplitFilter::Dev,
            cohort_opt: None,
            solver: SolverKind::Standard,
            engine_opt: Some(EngineKind::Cpu),
            seeds: 2,
            base_seed: 0,
            data_root: data_root.clone(),
            allow_repo_fallbacks: false,
            priors_path: None,
            runs_opt: Some(8),
            steps_opt: Some(30_000),
            hc_threshold: 0.8,
            no_crop: false,
            lp: false,
            no_axis_rescue: false,
            allow_truncated_fid: false,
            discover_only: false,
            json_out: Some(dir.join("summary.json")),
            verbose: false,
        };

        let summary =
            run_real_benchmark_with_loaders(&cfg, &|_| Ok(s_hsqc.clone()), &|_, exp| match exp {
                "HNCO" => Ok(s_hnco.clone()),
                "HNCACB" => Ok(s_hncacb.clone()),
                "CBCACONH" => Ok(s_cbcaconh.clone()),
                other => Err(format!("unexpected exp {other}")),
            })
            .unwrap();

        assert_eq!(summary.records.len(), 1);
        match &summary.records[0].outcome {
            EntryBenchmarkOutcome::Succeeded(m) => {
                assert!(m.mean_accuracy >= 0.85, "acc={}", m.mean_accuracy);
                assert!(
                    (m.global_offsets.ca - (-2.75)).abs() < 0.35,
                    "expected ~ -2.75 ppm unlabelled dCA offset, got {}",
                    m.global_offsets.ca
                );
                assert_eq!(m.detailed_offsets.ca.status, OffsetFitStatus::Fitted);
                assert!(!m.reliability.is_abstained());
            }
            other => panic!("expected Succeeded, got {other:?}"),
        }

        let summary_json_txt = fs::read_to_string(dir.join("summary.json")).unwrap();
        for required_key in [
            "\"warn_sub_hc_uncalibrated\":",
            "\"aliphatic_support_frac\":",
            "\"corroborated_connectivity_frac\":",
            "\"hc_call_fraction\":",
            "\"protein_hc_gate_triggered\":",
            "\"debiased_ece_10\":",
            "\"ece_10_bernoulli_floor\":",
            "\"mean_protein_debiased_ece_10\":",
            "\"mean_protein_ece_10_bernoulli_floor\":",
        ] {
            assert!(
                summary_json_txt.contains(required_key),
                "summary.json missing Phase 13 key {required_key}"
            );
        }

        // Also smoke-check discovery on local Ubiquitin & CISD3 if present on developer machine
        if let Ok(ubq_loc) = resolve_local_entry_paths("ubiquitin_local", &data_root, true) {
            let (ubq_disc, _) =
                discover_entry_experiments(&ubq_loc.entry_dir, Some(500.0)).unwrap();
            let ubq_map: BTreeMap<String, String> = ubq_disc
                .into_iter()
                .map(|d| (d.exp_type, d.rel_path))
                .collect();
            assert_eq!(
                ubq_map.get("HSQC").map(String::as_str),
                Some("Ubiquitin_090309/2")
            );
            assert_eq!(
                ubq_map.get("HNCACB").map(String::as_str),
                Some("Ubiquitin_090309/3")
            );
            assert_eq!(
                ubq_map.get("CBCACONH").map(String::as_str),
                Some("Ubiquitin_090309/4")
            );
            assert_eq!(
                ubq_map.get("HNCO").map(String::as_str),
                Some("Ubiquitin_111909/10")
            );
            assert_eq!(
                ubq_map.get("HNCACO").map(String::as_str),
                Some("Ubiquitin_111909/11")
            );
        }
        if let Ok(cisd_loc) = resolve_local_entry_paths("51439", &data_root, true) {
            let (cisd_disc, _) =
                discover_entry_experiments(&cisd_loc.entry_dir, Some(900.0)).unwrap();
            let cisd_exps: Vec<&str> = cisd_disc.iter().map(|d| d.exp_type.as_str()).collect();
            assert_eq!(
                cisd_exps,
                vec!["HSQC", "HNCO", "HNCA", "HNCOCA", "CBCACONH"]
            );
        }

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_committed_real_manifest_tsv_schema_splits_and_disjointness() {
        let manifest_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/benchmark/real_manifest.tsv");
        let raw = fs::read_to_string(&manifest_path).unwrap();
        let entries = parse_real_manifest_str(&raw).unwrap();
        let n_dev = entries.iter().filter(|e| e.split == "dev").count();
        let n_held = entries.iter().filter(|e| e.split == "heldout").count();
        assert_eq!(
            n_dev, 13,
            "expected 13 dev entries (10 dev + 3 dev_from_p10)"
        );

        // Verify unique BMRB IDs across all entries
        let unique_ids: BTreeSet<&str> = entries.iter().map(|e| e.bmrb_id.as_str()).collect();
        assert_eq!(unique_ids.len(), entries.len());

        // Verify Q1 + P1-7 governance transitions:
        // - 5329, 6318, 15476, 16385, 53280 in `dev` (`cohort=dev`),
        // - 16072, 16782, 17092 promoted from `cold_p10` to `dev` (`cohort=dev_from_p10`),
        // - 5845, 17030, 18145, 51351, 16099, 18489 tagged `seen_p8_p9` (`heldout`),
        // - Former `cold_p12` entries relabeled `seen_p12` (`heldout`, spent at `v0.8.0` per `P1-7`).
        for promoted_id in ["5329", "6318", "15476", "16385", "53280"] {
            let e = entries
                .iter()
                .find(|e| e.bmrb_id == promoted_id)
                .unwrap_or_else(|| panic!("missing promoted entry {promoted_id}"));
            assert_eq!(e.split, "dev", "expected {promoted_id} in dev split");
            assert_eq!(e.cohort, "dev", "expected {promoted_id} in dev cohort");
        }
        for dev_p10_id in ["16072", "16782", "17092"] {
            let e = entries
                .iter()
                .find(|e| e.bmrb_id == dev_p10_id)
                .unwrap_or_else(|| panic!("missing dev_from_p10 entry {dev_p10_id}"));
            assert_eq!(e.split, "dev", "expected {dev_p10_id} in dev split");
            assert_eq!(
                e.cohort, "dev_from_p10",
                "expected {dev_p10_id} in dev_from_p10 cohort"
            );
            assert!((50..=160).contains(&e.residues));
        }
        let n_seen = entries
            .iter()
            .filter(|e| e.split == "heldout" && e.cohort == "seen_p8_p9")
            .count();
        let n_seen_p12 = entries
            .iter()
            .filter(|e| e.split == "heldout" && e.cohort == "seen_p12")
            .count();
        let n_seen_p13 = entries
            .iter()
            .filter(|e| e.split == "heldout" && e.cohort == "seen_p13")
            .count();
        let n_seen_p13_w2 = entries
            .iter()
            .filter(|e| e.split == "heldout" && e.cohort == "seen_p13_w2")
            .count();
        let n_cold_p12 = entries.iter().filter(|e| e.cohort == "cold_p12").count();
        let n_cold_p13 = entries.iter().filter(|e| e.cohort == "cold_p13").count();
        let n_cold_p13_w2 = entries.iter().filter(|e| e.cohort == "cold_p13_w2").count();
        assert_eq!(n_seen, 6);
        assert_eq!(
            n_cold_p12, 0,
            "cold_p12 was evaluated at v0.8.0 and must be relabeled seen_p12 per P1-7"
        );
        assert_eq!(
            n_cold_p13, 0,
            "cold_p13 was evaluated and must be relabeled seen_p13 per W1 F4"
        );
        assert_eq!(
            n_cold_p13_w2, 0,
            "cold_p13_w2 was evaluated in 13.W2 and must be relabeled seen_p13_w2 per 13.1:P1-2"
        );
        assert!(
            (6..=8).contains(&n_seen_p12),
            "expected 6..=8 seen_p12 entries, got {n_seen_p12}"
        );
        assert!(
            (6..=8).contains(&n_seen_p13),
            "expected 6..=8 seen_p13 entries, got {n_seen_p13}"
        );
        assert!(
            (6..=8).contains(&n_seen_p13_w2),
            "expected 6..=8 seen_p13_w2 entries, got {n_seen_p13_w2}"
        );
        assert_eq!(n_held, n_seen + n_seen_p12 + n_seen_p13 + n_seen_p13_w2);
        for seen_id in ["5845", "17030", "18145", "51351", "16099", "18489"] {
            let e = entries
                .iter()
                .find(|e| e.bmrb_id == seen_id)
                .unwrap_or_else(|| panic!("missing seen heldout entry {seen_id}"));
            assert_eq!(e.split, "heldout");
            assert_eq!(e.cohort, "seen_p8_p9");
        }
        for e in entries.iter().filter(|e| {
            e.split == "heldout"
                && matches!(e.cohort.as_str(), "seen_p12" | "seen_p13" | "seen_p13_w2")
        }) {
            let entry_id = e.bmrb_id.as_str();
            let coh = e.cohort.as_str();
            assert!((40..=200).contains(&e.residues));
            // Header-only verification (strictly zero FID FFT, peak picking, or solver runs!)
            if let Ok(loc) =
                resolve_local_entry_paths(entry_id, Path::new("data/private/bmrb"), true)
            {
                assert!(
                    loc.truth_path.as_ref().is_some_and(|p| p.is_file()),
                    "{coh} entry {entry_id} must have NMR-STAR file present"
                );
                let ws = nmr_core::classify::classify_nmr_workspace(&loc.entry_dir).unwrap();
                assert!(
                    ws.problem_regime == nmr_core::classify::ProblemRegime::GlobularBackbone
                        || ws
                            .secondary_regimes
                            .contains(&nmr_core::classify::ProblemRegime::GlobularBackbone),
                    "{coh} entry {entry_id} must support GlobularBackbone (got primary {:?}, secondary {:?})",
                    ws.problem_regime,
                    ws.secondary_regimes
                );
                let (disc, _) =
                    discover_entry_experiments(&loc.entry_dir, Some(e.field_mhz)).unwrap();
                let disc_exps: BTreeSet<&str> = disc.iter().map(|d| d.exp_type.as_str()).collect();
                println!("  [{coh} header-only] {entry_id} -> discovered={disc_exps:?}");
                for req_exp in &e.experiments {
                    assert!(
                        disc_exps.contains(req_exp.as_str()),
                        "{coh} entry {entry_id} missing manifest experiment {req_exp}; discovered {disc_exps:?}"
                    );
                }
            }
        }

        // Verify zero overlap with Phase-1 `data/benchmark/manifest.tsv`
        let p1_raw = fs::read_to_string(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/benchmark/manifest.tsv"),
        )
        .unwrap();
        for line in p1_raw.lines().skip(1) {
            if let Some(col0) = line.split('\t').next() {
                let id_num = col0.trim().trim_start_matches("bmr");
                assert!(
                    !entries.iter().any(|e| e.bmrb_id == id_num),
                    "real_manifest entry {id_num} overlaps Phase-1 manifest.tsv"
                );
            }
        }

        // Verify zero overlap with empirical prior training set in `data/priors/PROVENANCE.md`
        let prov_raw = fs::read_to_string(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/priors/PROVENANCE.md"),
        )
        .unwrap();
        for tok in prov_raw
            .split(|c: char| !c.is_ascii_alphanumeric())
            .filter(|t| t.starts_with("bmr") && t.len() > 3)
        {
            let id_num = tok.trim_start_matches("bmr");
            if id_num.chars().all(|c| c.is_ascii_digit()) {
                assert!(
                    !entries.iter().any(|e| e.bmrb_id == id_num),
                    "real_manifest entry {id_num} overlaps empirical prior training set in data/priors/PROVENANCE.md"
                );
            }
        }

        // Verify `data/benchmark/PROVENANCE.md` pre-registration record embeds the exact
        // FIPS 180-4 SHA-256 digest of `data/benchmark/real_manifest.tsv` (Q1 governance;
        // when `n_cold_p13_w2 == 0` prior to Wave 2 cold pre-registration, also accept the exact
        // `c0dfc76` W1-F4 `cold_p13 -> seen_p13` relabelled digest alongside the W1 `PROVENANCE.md` record).
        const W1_F4_SEEN_P13_MANIFEST_SHA256: &str =
            "0ab592d8fecc1e5374da0dbac5ecc84f0068bcdb256152dbfa4fce8477ef7dec";
        let (manifest_sha256, _) = crate::fetch_bmrb::sha256_file_and_size(&manifest_path).unwrap();
        let bench_prov_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/benchmark/PROVENANCE.md");
        let bench_prov_raw = fs::read_to_string(&bench_prov_path)
            .expect("data/benchmark/PROVENANCE.md must exist with pre-registered manifest SHA-256");
        let digest_ok = bench_prov_raw.contains(&manifest_sha256)
            || (n_cold_p13_w2 == 0 && manifest_sha256 == W1_F4_SEEN_P13_MANIFEST_SHA256);
        assert!(
            digest_ok,
            "data/benchmark/PROVENANCE.md does not contain current real_manifest.tsv SHA-256 ({manifest_sha256})"
        );

        // Verify P1-6 default solver is `evo-graph` while `--solver standard` remains selectable
        let cfg_default =
            parse_real_benchmark_args(&["--split".to_string(), "dev".to_string()]).unwrap();
        assert_eq!(cfg_default.solver, SolverKind::EvoGraph);
        let cfg_std = parse_real_benchmark_args(&[
            "--split".to_string(),
            "dev".to_string(),
            "--solver".to_string(),
            "standard".to_string(),
        ])
        .unwrap();
        assert_eq!(cfg_std.solver, SolverKind::Standard);
        let cfg_bayes = parse_real_benchmark_args(&[
            "--split".to_string(),
            "dev".to_string(),
            "--solver".to_string(),
            "bayes-sa".to_string(),
        ])
        .unwrap();
        assert_eq!(cfg_bayes.solver, SolverKind::BayesSa);
        let cfg_bayes_alias = parse_real_benchmark_args(&[
            "--split".to_string(),
            "dev".to_string(),
            "--solver".to_string(),
            "bayes_sa".to_string(),
        ])
        .unwrap();
        assert_eq!(cfg_bayes_alias.solver, SolverKind::BayesSa);
    }

    #[test]
    fn test_calibration_summary_distinguishes_abstain_from_zero_percent_hc_and_detailed_offsets() {
        use nmr_core::qc::DatasetQcVerdict;

        let mk_entry = |id: &str, name: &str| RealManifestEntry {
            bmrb_id: id.to_string(),
            protein: name.to_string(),
            residues: 70,
            field_mhz: 600.0,
            vendor: "Bruker".to_string(),
            experiments: vec!["HSQC".into(), "HNCO".into(), "HNCACB".into()],
            split: "dev".to_string(),
            cohort: "dev".to_string(),
        };

        let det_mixed = DetailedReferenceOffsets {
            h: NucleusOffsetFit {
                status: OffsetFitStatus::NotSignificant,
                applied_ppm: 0.0,
                candidate_ppm: 0.01,
                log_likelihood_gain: 0.4,
                n_values: 60,
            },
            n: NucleusOffsetFit {
                status: OffsetFitStatus::NotSignificant,
                applied_ppm: 0.0,
                candidate_ppm: -0.05,
                log_likelihood_gain: 0.6,
                n_values: 60,
            },
            ca: NucleusOffsetFit {
                status: OffsetFitStatus::Fitted,
                applied_ppm: -2.75,
                candidate_ppm: -2.75,
                log_likelihood_gain: 42.5,
                n_values: 58,
            },
            cb: NucleusOffsetFit {
                status: OffsetFitStatus::TooFewValues,
                applied_ppm: 0.0,
                candidate_ppm: 0.0,
                log_likelihood_gain: 0.0,
                n_values: 0,
            },
            c: NucleusOffsetFit {
                status: OffsetFitStatus::NotSignificant,
                applied_ppm: 0.0,
                candidate_ppm: 0.08,
                log_likelihood_gain: 1.1,
                n_values: 55,
            },
        };

        // 1. Clean high-accuracy protein (`PASS`, 19/20 = 95.0% HC precision)
        let mut preds_good: Vec<(f64, bool)> = (0..19).map(|_| (0.95, true)).collect();
        preds_good.push((0.85, false));
        let rel_good = ProteinReliabilityCurve::from_predictions(&preds_good, 0.8);
        let rec_good = EntryBenchmarkRecord {
            manifest_entry: mk_entry("11111", "CleanProt"),
            outcome: EntryBenchmarkOutcome::Succeeded(Box::new(EntrySuccessMetrics {
                seeds: 1,
                discovered_experiments: vec![("HSQC".into(), "1".into())],
                n_hsqc_roots: 62,
                n_total_peaks: 310,
                assignable_per_seed: 20,
                total_assignable: 20,
                total_assigned: 20,
                total_correct: 19,
                total_hc_assigned: 20,
                total_hc_correct: 19,
                mean_accuracy: 0.95,
                mean_recall: 0.95,
                mean_hc_precision: 0.95,
                mean_hc_recall: 0.95,
                global_offsets: det_mixed.to_shift_offsets(),
                detailed_offsets: det_mixed,
                prior_detailed_offsets: det_mixed,
                qc_report: DatasetQcReport {
                    verdict: DatasetQcVerdict::Pass,
                    detailed_offsets: det_mixed,
                    ..DatasetQcReport::default()
                },
                axis_rescue_audits: Vec::new(),
                reliability: rel_good,
                predictions: preds_good,
                elapsed_secs: 0.5,
            })),
        };

        // 2. Abstained dataset (`ABSTAIN`, all confidences capped <= 0.45 -> 0 HC calls)
        let preds_abstain: Vec<(f64, bool)> = (0..10).map(|i| (0.35, i < 2)).collect();
        let rel_abstain = ProteinReliabilityCurve::from_predictions(&preds_abstain, 0.8);
        assert!(rel_abstain.is_abstained());
        let rec_abstain = EntryBenchmarkRecord {
            manifest_entry: mk_entry("22222", "AbstainProt"),
            outcome: EntryBenchmarkOutcome::Succeeded(Box::new(EntrySuccessMetrics {
                seeds: 1,
                discovered_experiments: vec![("HSQC".into(), "1".into())],
                n_hsqc_roots: 60,
                n_total_peaks: 80,
                assignable_per_seed: 20,
                total_assignable: 20,
                total_assigned: 10,
                total_correct: 2,
                total_hc_assigned: 0,
                total_hc_correct: 0,
                mean_accuracy: 0.20,
                mean_recall: 0.10,
                mean_hc_precision: 0.0,
                mean_hc_recall: 0.0,
                global_offsets: ShiftOffsets::default(),
                detailed_offsets: DetailedReferenceOffsets::default(),
                prior_detailed_offsets: DetailedReferenceOffsets::default(),
                qc_report: DatasetQcReport {
                    verdict: DatasetQcVerdict::Abstain,
                    confidence_scale: 0.0,
                    confidence_cap: 0.45,
                    reasons: vec!["catastrophic 3D strip support (12.0% < 25.0%)".into()],
                    ..DatasetQcReport::default()
                },
                axis_rescue_audits: Vec::new(),
                reliability: rel_abstain,
                predictions: preds_abstain,
                elapsed_secs: 0.3,
            })),
        };

        // 3. Miscalibrated dataset with wrong HC calls (`WARN`, 0/5 = 0.0% HC precision, NOT abstained!)
        let preds_wrong_hc: Vec<(f64, bool)> = (0..5).map(|_| (0.86, false)).collect();
        let rel_wrong_hc = ProteinReliabilityCurve::from_predictions(&preds_wrong_hc, 0.8);
        assert!(!rel_wrong_hc.is_abstained());
        assert_eq!(rel_wrong_hc.hc_precision, Some(0.0));
        let rec_wrong_hc = EntryBenchmarkRecord {
            manifest_entry: mk_entry("33333", "WrongHcProt"),
            outcome: EntryBenchmarkOutcome::Succeeded(Box::new(EntrySuccessMetrics {
                seeds: 1,
                discovered_experiments: vec![("HSQC".into(), "1".into())],
                n_hsqc_roots: 60,
                n_total_peaks: 200,
                assignable_per_seed: 10,
                total_assignable: 10,
                total_assigned: 5,
                total_correct: 0,
                total_hc_assigned: 5,
                total_hc_correct: 0,
                mean_accuracy: 0.0,
                mean_recall: 0.0,
                mean_hc_precision: 0.0,
                mean_hc_recall: 0.0,
                global_offsets: ShiftOffsets::default(),
                detailed_offsets: DetailedReferenceOffsets::default(),
                prior_detailed_offsets: DetailedReferenceOffsets::default(),
                qc_report: DatasetQcReport {
                    verdict: DatasetQcVerdict::Warn,
                    confidence_scale: 0.78,
                    confidence_cap: 1.0,
                    reasons: vec!["elevated HSQC root multiplicity (1.60x)".into()],
                    ..DatasetQcReport::default()
                },
                axis_rescue_audits: Vec::new(),
                reliability: rel_wrong_hc,
                predictions: preds_wrong_hc,
                elapsed_secs: 0.4,
            })),
        };

        let summary = summarize_real_benchmark_records(
            "dev",
            "CPU (f64, evo-graph)".to_string(),
            1,
            0.8,
            vec![rec_good, rec_abstain, rec_wrong_hc],
        );

        assert_eq!(summary.n_hc_active_proteins, 2);
        assert_eq!(summary.n_hc_abstained_proteins, 1);
        assert!(
            (summary.macro_hc_precision - 0.475).abs() < 1e-9,
            "macro HC precision must average over the 2 HC-active proteins (0.95 and 0.0 -> 0.475), got {}",
            summary.macro_hc_precision
        );
        assert_eq!(
            summary.worst_protein_hc_precision,
            Some(("33333".to_string(), "WrongHcProt".to_string(), 0.0))
        );
        assert!(
            (summary.median_protein_hc_precision.unwrap() - 0.475).abs() < 1e-9,
            "got {:?}",
            summary.median_protein_hc_precision
        );

        let table = format_real_benchmark_report(&summary);
        assert!(table.contains("PASS") && table.contains("ABSTAIN") && table.contains("WARN"));
        assert!(table.contains("dCA=-2.75(ΔLL=42.5)"), "table:\n{table}");
        assert!(table.contains("dCB=n<5(0)"), "table:\n{table}");
        assert!(table.contains("dC=ns(ΔLL=1.1)"), "table:\n{table}");
        assert!(
            table.contains("Abstained (0 HC calls)"),
            "expected explicit abstained status in table:\n{table}"
        );
        assert!(
            table.contains("0.0% (0/5)"),
            "expected explicit 0.0% (0/5) wrong-HC status in table:\n{table}"
        );
        assert!(
            table.contains("[0.0-0.5):")
                && table.contains("[0.5-0.8):")
                && table.contains("[0.8-0.9):")
                && table.contains("[0.9-1.0]:"),
            "expected 4-bin reliability curves in table:\n{table}"
        );

        let json = render_summary_json(&summary);
        assert!(json.contains("\"hc_active_proteins\": 2"));
        assert!(json.contains("\"hc_abstained_proteins\": 1"));
        assert!(json.contains("\"hc_abstained\": true,\n      \"hc_precision\": null"));
        assert!(json.contains("\"hc_abstained\": false,\n      \"hc_precision\": 0.000000"));
        assert!(json.contains("\"status\": \"fitted\""));
        assert!(json.contains("\"status\": \"not_significant\""));
        assert!(json.contains("\"status\": \"too_few_values\""));
        assert!(json.contains("\"calibration_summary\":"));
    }

    #[test]
    #[ignore]
    fn dev_split_lopo_calibration_and_minor_conformer_audit() {
        use nmr_core::evo_graph::{assign_evo_graph_from_peaks_with_minor_damping, EvoGraphConfig};
        use nmr_core::priors::sync_detailed_reference_offsets;
        use nmr_core::qc::{
            apply_evo_graph_calibration_map, apply_standard_calibration_map,
            apply_standard_gpu_calibration_map, evaluate_dataset_qc,
            evaluate_protein_hc_gate_objective_multi_engine, evaluate_seed_predictions_with_gate,
            fit_isotonic_monotone_knots, fit_isotonic_monotone_knots_beta,
            fit_isotonic_monotone_knots_shrunk_toward_beta,
            fit_protein_hc_gate_config_multi_engine, fit_protein_hc_gate_config_with_grids,
            is_unconverged_solution_margin, select_gpu_shrinkage_weight_nested_lopo,
            DatasetQcVerdict, ProteinHcGateConfig, ProteinHcTrainingSeed, ProteinReliabilityCurve,
            CANONICAL_CALIBRATION_GRID_12, CANONICAL_CALIBRATION_LAPLACE_PRIOR,
            CANONICAL_PROTEIN_HC_GATE, QC_STRONG_BELIEF_THRESHOLD, STANDARD_CPU_PRIOR_ALPHA,
            STANDARD_CPU_PRIOR_BETA, STANDARD_GPU_PRIOR_ALPHA, STANDARD_GPU_PRIOR_BETA,
        };
        use nmr_core::solver::{assign_uncalibrated, prepare_scoring_priors, AnnealConfig};
        use nmr_gpu::GpuEngine;

        let gpu_engine = GpuEngine::new_headless()
            .expect("WebGPU headless adapter required for STANDARD_GPU LOPO calibration");
        println!(
            "  [GPU ADAPTER] name={:?}, backend={:?}",
            gpu_engine.adapter_name, gpu_engine.backend_name
        );

        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        env::set_current_dir(&repo_root).expect("chdir repo_root");
        let manifest_path = PathBuf::from("data/benchmark/real_manifest.tsv");
        let manifest_txt = fs::read_to_string(&manifest_path).expect("read real_manifest.tsv");
        let entries: Vec<RealManifestEntry> = parse_real_manifest_str(&manifest_txt)
            .expect("parse real_manifest.tsv")
            .into_iter()
            .filter(|e| e.split == "dev")
            .collect();
        assert_eq!(entries.len(), 13, "expected exact 13 dev entries");
        for e in &entries {
            assert_eq!(e.split, "dev", "STRICT HELDOUT FIREWALL");
        }

        let priors = load_priors(None).expect("load priors");
        let score_cfg = ScoreConfig::default();
        let picker_cfg = PickerConfig::default();
        let data_root = env::var("NMR_REAL_DATA_DIR")
            .map(|p| PathBuf::from(p).join("private/bmrb"))
            .unwrap_or_else(|_| PathBuf::from("data/private/bmrb"));

        let make_training_seed =
            |qc: &DatasetQcReport, raw_predictions: Vec<(f64, bool)>| -> ProteinHcTrainingSeed {
                ProteinHcTrainingSeed::from_qc_report(qc, true, raw_predictions)
            };

        // Fixed dataset convergence specification (`(0.65, 0.07, 1.60, 0.16)`) identical to
        // `nmr_core::qc::is_unconverged_solution_margin` / `evaluate_dataset_qc`: excludes
        // datasets whose solver ensemble failed to converge (`Abstain`) from the PAV training pool.
        let is_trainable_prot = |seeds: &[ProteinHcTrainingSeed]| -> bool {
            let tot_n: usize = seeds.iter().map(|s| s.raw_predictions.len()).sum();
            if tot_n == 0 {
                return false;
            }
            let s0 = &seeds[0];
            let strong: usize = seeds
                .iter()
                .map(|s| {
                    s.raw_predictions
                        .iter()
                        .filter(|&&(p, _)| p >= QC_STRONG_BELIEF_THRESHOLD)
                        .count()
                })
                .sum();
            let m = (strong as f64) / (tot_n as f64);
            let n_per_seed = tot_n / seeds.len().max(1);
            !(s0.n_amide_hsqc_roots == 0
                || (s0.n_amide_hsqc_roots > 0 && s0.hsqc_3d_support_frac < 0.25)
                || s0.aliphatic_support_frac < 0.20
                || (n_per_seed >= 6 && s0.connectivity_frac < 0.15)
                || is_unconverged_solution_margin(tot_n, m, s0.hsqc_multiplicity_ratio)
                || ((s0.hsqc_3d_rms_dh_ppm > 0.065 || s0.hsqc_3d_rms_dn_ppm > 0.65)
                    && s0.hsqc_3d_support_frac < 0.45)
                || (s0.n_amide_hsqc_roots > 0
                    && s0.hsqc_3d_support_frac < 0.45
                    && s0.supported_roots_per_assignable < 0.35
                    && m < 0.20))
        };

        type ProtRecord = (String, String, DatasetQcVerdict, Vec<ProteinHcTrainingSeed>);
        let mut evo_by_prot: Vec<ProtRecord> = Vec::new();
        let mut std_by_prot: Vec<ProtRecord> = Vec::new();
        let mut std_gpu_by_prot: Vec<ProtRecord> = Vec::new();

        // P1-5 minor-conformer audit accumulators across all 10 dev proteins
        let mut tot_true_roots_1seed = 0usize;
        let mut tot_weak_true_roots_1seed = 0usize;
        let mut tot_weak_ht_true_roots_1seed = 0usize;
        let mut weak_with_assigned = 0usize;
        let mut weak_with_correct = 0usize;
        let mut weak_without_assigned = 0usize;
        let mut weak_without_correct = 0usize;
        let mut weak_total_5seed = 0usize;
        let mut all_with_assigned = 0usize;
        let mut all_with_correct = 0usize;
        let mut all_without_assigned = 0usize;
        let mut all_without_correct = 0usize;
        let mut per_prot_audit_lines: Vec<String> = Vec::new();

        let only_filter = env::var("ONLY_ENTRY").ok();
        for entry in &entries {
            if let Some(ref f) = only_filter {
                if !f.split(',').any(|tok| tok.trim() == entry.bmrb_id) {
                    continue;
                }
            }
            let loc = resolve_local_entry_paths(&entry.bmrb_id, &data_root, true)
                .unwrap_or_else(|p| panic!("missing dev entry {}: {p:?}", entry.bmrb_id));
            let truth_path = loc.truth_path.expect("truth STAR path");
            let (truth_ref, _, _) = parse_truth_auto(&truth_path).expect("parse truth STAR");
            let (discovered, _) = discover_entry_experiments(&loc.entry_dir, Some(entry.field_mhz))
                .expect("discover");
            let hsqc_exp = discovered
                .iter()
                .find(|d| d.exp_type == "HSQC")
                .expect("HSQC");
            let triple_exps: Vec<&DiscoveredFidExperiment> =
                discovered.iter().filter(|d| d.exp_type != "HSQC").collect();

            let (hsqc_spec, _) =
                process_fid_dir_2d(&hsqc_exp.dir, false, false).expect("load 2D HSQC");
            let roots = pick_hsqc_2d(&hsqc_spec, &picker_cfg);
            let mut all_peaks: Vec<Peak> = roots.clone();
            let mut used_exps = vec![("HSQC".to_string(), hsqc_exp.rel_path.clone())];
            let mut strip_bank = StripBank3D::new();
            for texp in &triple_exps {
                if let Ok((mut spec3d, _)) =
                    process_fid_dir_3d(&texp.dir, &texp.exp_type, false, false)
                {
                    let _ =
                        register_3d_hn_to_hsqc_adaptive_with_audit(&mut spec3d, &hsqc_spec, true);
                    let pks = strip_bank.pick_and_record_3d(&spec3d, &roots, &picker_cfg);
                    all_peaks.extend(pks);
                    used_exps.push((texp.exp_type.clone(), texp.rel_path.clone()));
                }
            }
            if used_exps.len() >= 3 && !roots.is_empty() {
                all_peaks.extend(corroborated_re_pick_3d(
                    &strip_bank,
                    &roots,
                    &all_peaks,
                    &picker_cfg,
                    &CorroboratedRepickConfig::default(),
                ));
            }
            let all_systems_for_eval =
                build_spin_systems(&all_peaks, &Tolerances::default(), 3, false)
                    .expect("build_spin_systems");

            let mut evo_samples = Vec::new();
            let mut std_samples = Vec::new();
            let mut std_gpu_samples = Vec::new();
            let mut evo_verdict = DatasetQcVerdict::Pass;
            let mut std_verdict = DatasetQcVerdict::Pass;
            let mut std_gpu_verdict = DatasetQcVerdict::Pass;
            let mut evo_qc_reasons = String::new();
            let mut std_qc_reasons = String::new();

            let mut p_weak_1seed = 0usize;
            let mut p_weak_ht_1seed = 0usize;
            let mut p_true_1seed = 0usize;
            let mut p_weak_w_ass = 0usize;
            let mut p_weak_w_cor = 0usize;
            let mut p_weak_wo_ass = 0usize;
            let mut p_weak_wo_cor = 0usize;

            // Precompute standard spin systems & offsets once per protein
            let std_base_det = nmr_core::priors::estimate_reference_offsets_detailed(
                &all_systems_for_eval,
                &truth_ref.sequence,
                &priors,
            );
            let (std_eff_priors, std_offs) = prepare_scoring_priors(
                &all_systems_for_eval,
                &truth_ref.sequence,
                &priors,
                &score_cfg,
            );
            let std_synced_det = sync_detailed_reference_offsets(&std_offs, &std_base_det);

            for seed in 0..5u64 {
                // 1. Evo-graph WITH minor-conformer damping (`apply_qc_calibration = false`;
                // zero empirical knot tables exist upstream of Pass 2!)
                let evo_cfg_raw = EvoGraphConfig {
                    score_cfg: score_cfg.clone(),
                    auto_reference: true,
                    apply_qc_calibration: false,
                    seed,
                    ..EvoGraphConfig::default()
                };
                let (evo_sys, evo_res_raw, evo_rep) =
                    assign_evo_graph_from_peaks_with_minor_damping(
                        &all_peaks,
                        &truth_ref.sequence,
                        &priors,
                        &evo_cfg_raw,
                        true,
                    );
                let qc_evo = &evo_rep.qc_report;
                if seed == 0 {
                    // Display seed-0 diagnostic QC verdict/notes on calibrated confidences (`P2`)
                    // while keeping `qc_evo` / `evo_res_raw` completely knot-free for LOPO training.
                    let mut cal_evo_res = evo_res_raw.clone();
                    for c in &mut cal_evo_res.confidence {
                        *c = apply_evo_graph_calibration_map(*c);
                    }
                    let cal_qc_evo = evaluate_dataset_qc(
                        Some(&all_peaks),
                        &evo_sys,
                        &truth_ref.sequence,
                        &priors,
                        &score_cfg,
                        &cal_evo_res,
                        Some(&qc_evo.detailed_offsets),
                    );
                    evo_verdict = cal_qc_evo.verdict;
                    evo_qc_reasons = cal_qc_evo.reasons.join("; ");
                }
                let eff_evo_priors = evo_rep.shifted_priors(&priors);
                let (pred_evo, conf_k_evo) = to_reference(
                    &evo_res_raw,
                    &evo_sys,
                    &truth_ref.sequence,
                    &truth_ref.seq_ids,
                    0.0,
                    Some(&eff_evo_priors),
                );
                let mut conf_seq_evo = BTreeMap::new();
                for (k, c) in conf_k_evo {
                    conf_seq_evo.insert(truth_ref.seq_ids[k], c);
                }
                let eval_evo = evaluate_prediction_against_truth(
                    &pred_evo,
                    &truth_ref,
                    &conf_seq_evo,
                    Some(&all_systems_for_eval),
                    0.80,
                    true,
                );
                evo_samples.push(make_training_seed(
                    qc_evo,
                    eval_evo.placed_predictions.clone(),
                ));
                all_with_assigned += eval_evo.assigned;
                all_with_correct += eval_evo.correct;

                // Map `evo_sys` 1-to-1 to `eval_evo.truth_for_cmp` (`tol_h = 0.05, tol_n = 0.50`)
                let mut true_roots = Vec::new();
                for k in 1..eval_evo.truth_for_cmp.sequence.len() {
                    if eval_evo.truth_for_cmp.sequence[k] == "PRO" {
                        continue;
                    }
                    if let (Some(&th), Some(&tn)) = (
                        eval_evo.truth_for_cmp.shifts.get(&(k, "H".to_string())),
                        eval_evo.truth_for_cmp.shifts.get(&(k, "N".to_string())),
                    ) {
                        true_roots.push((k, th, tn));
                    }
                }
                let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
                for (si, ss) in evo_sys.iter().enumerate() {
                    for (ti, &(_, th, tn)) in true_roots.iter().enumerate() {
                        let dh = (ss.h - th).abs();
                        let dn = (ss.n - tn).abs();
                        if dh <= 0.05 && dn <= 0.50 {
                            pairs.push(((dh / 0.05).powi(2) + (dn / 0.50).powi(2), si, ti));
                        }
                    }
                }
                pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
                let mut used_s = vec![false; evo_sys.len()];
                let mut used_t = vec![false; true_roots.len()];
                let mut evo_ss_truth = vec![-1i32; evo_sys.len()];
                for &(_, si, ti) in &pairs {
                    if !used_s[si] && !used_t[ti] {
                        used_s[si] = true;
                        used_t[ti] = true;
                        evo_ss_truth[si] = true_roots[ti].0 as i32;
                    }
                }

                // 2. Evo-graph WITHOUT minor-conformer damping (`enable_minor_damping = false`)
                let (_evo_sys_no, evo_res_no, evo_rep_no) =
                    assign_evo_graph_from_peaks_with_minor_damping(
                        &all_peaks,
                        &truth_ref.sequence,
                        &priors,
                        &evo_cfg_raw,
                        false,
                    );
                let eff_no_priors = evo_rep_no.shifted_priors(&priors);
                let (pred_no, conf_k_no) = to_reference(
                    &evo_res_no,
                    &evo_sys,
                    &truth_ref.sequence,
                    &truth_ref.seq_ids,
                    0.0,
                    Some(&eff_no_priors),
                );
                let mut conf_seq_no = BTreeMap::new();
                for (k, c) in conf_k_no {
                    conf_seq_no.insert(truth_ref.seq_ids[k], c);
                }
                let eval_no = evaluate_prediction_against_truth(
                    &pred_no,
                    &truth_ref,
                    &conf_seq_no,
                    Some(&all_systems_for_eval),
                    0.80,
                    true,
                );
                all_without_assigned += eval_no.assigned;
                all_without_correct += eval_no.correct;

                for (si, ss) in evo_sys.iter().enumerate() {
                    let tk = evo_ss_truth[si];
                    if tk < 0 {
                        continue;
                    }
                    if seed == 0 {
                        p_true_1seed += 1;
                        if ss.rel_root_height.is_some_and(|rh| rh < 0.15) {
                            p_weak_ht_1seed += 1;
                        }
                    }
                    if ss
                        .rel_root_intensity
                        .or(ss.rel_root_height)
                        .is_some_and(|r| r < 0.15)
                    {
                        if seed == 0 {
                            p_weak_1seed += 1;
                        }
                        weak_total_5seed += 1;
                        if evo_res_raw.position[si] >= 0 {
                            p_weak_w_ass += 1;
                            if evo_res_raw.position[si] == tk {
                                p_weak_w_cor += 1;
                            }
                        }
                        if evo_res_no.position[si] >= 0 {
                            p_weak_wo_ass += 1;
                            if evo_res_no.position[si] == tk {
                                p_weak_wo_cor += 1;
                            }
                        }
                    }
                }

                // 3. Standard CPU (`runs = 16`) raw pre-QC confidences
                let anneal_cfg = AnnealConfig {
                    runs: 16,
                    steps_per_system: 1500,
                    seed,
                    ..AnnealConfig::default()
                };
                let std_res_raw = assign_uncalibrated(
                    &all_systems_for_eval,
                    &truth_ref.sequence,
                    &priors,
                    &score_cfg,
                    &anneal_cfg,
                );
                let qc_std = evaluate_dataset_qc(
                    Some(&all_peaks),
                    &all_systems_for_eval,
                    &truth_ref.sequence,
                    &priors,
                    &score_cfg,
                    &std_res_raw,
                    Some(&std_synced_det),
                );
                if seed == 0 {
                    let mut cal_std_res = std_res_raw.clone();
                    for c in &mut cal_std_res.confidence {
                        *c = apply_standard_calibration_map(*c);
                    }
                    let cal_qc_std = evaluate_dataset_qc(
                        Some(&all_peaks),
                        &all_systems_for_eval,
                        &truth_ref.sequence,
                        &priors,
                        &score_cfg,
                        &cal_std_res,
                        Some(&std_synced_det),
                    );
                    std_verdict = cal_qc_std.verdict;
                    std_qc_reasons = cal_qc_std.reasons.join("; ");
                }
                let (pred_std, conf_k_std) = to_reference(
                    &std_res_raw,
                    &all_systems_for_eval,
                    &truth_ref.sequence,
                    &truth_ref.seq_ids,
                    0.0,
                    Some(&std_eff_priors),
                );
                let mut conf_seq_std = BTreeMap::new();
                for (k, c) in conf_k_std {
                    conf_seq_std.insert(truth_ref.seq_ids[k], c);
                }
                let eval_std = evaluate_prediction_against_truth(
                    &pred_std,
                    &truth_ref,
                    &conf_seq_std,
                    Some(&all_systems_for_eval),
                    0.80,
                    true,
                );
                std_samples.push(make_training_seed(
                    &qc_std,
                    eval_std.placed_predictions.clone(),
                ));
                if only_filter.is_some() && seed == 0 {
                    let n_hi_ok = eval_std
                        .placed_predictions
                        .iter()
                        .filter(|&&(p, ok)| p >= 0.90 && ok)
                        .count();
                    let n_hi_bad = eval_std
                        .placed_predictions
                        .iter()
                        .filter(|&&(p, ok)| p >= 0.90 && !ok)
                        .count();
                    println!(
                        "    [SEED0 {}] acc={}/{} ({:.1}%), eval_offs={}, qc={}, conn={:.3}, margin={:.3}, p>=0.90 ok={} bad={}",
                        entry.bmrb_id,
                        eval_std.correct,
                        eval_std.assigned,
                        100.0 * (eval_std.correct as f64) / (eval_std.assigned.max(1) as f64),
                        eval_std.global_offsets,
                        qc_std.verdict.as_str(),
                        qc_std.connectivity_frac,
                        qc_std.solution_margin,
                        n_hi_ok,
                        n_hi_bad,
                    );
                }
                // 4. Standard WebGPU (`runs = 64`, matching `execute_assignment_workflow` on GPU)
                let gpu_anneal_cfg = AnnealConfig {
                    runs: 64,
                    steps_per_system: 1500,
                    seed,
                    ..AnnealConfig::default()
                };
                let std_gpu_res_raw = gpu_engine
                    .assign_gpu_uncalibrated(
                        &all_systems_for_eval,
                        &truth_ref.sequence,
                        &priors,
                        &score_cfg,
                        &gpu_anneal_cfg,
                    )
                    .expect("assign_gpu_uncalibrated");
                let qc_std_gpu = evaluate_dataset_qc(
                    Some(&all_peaks),
                    &all_systems_for_eval,
                    &truth_ref.sequence,
                    &priors,
                    &score_cfg,
                    &std_gpu_res_raw,
                    Some(&std_synced_det),
                );
                if seed == 0 {
                    let mut cal_gpu_res = std_gpu_res_raw.clone();
                    for c in &mut cal_gpu_res.confidence {
                        *c = apply_standard_gpu_calibration_map(*c);
                    }
                    let cal_qc_gpu = evaluate_dataset_qc(
                        Some(&all_peaks),
                        &all_systems_for_eval,
                        &truth_ref.sequence,
                        &priors,
                        &score_cfg,
                        &cal_gpu_res,
                        Some(&std_synced_det),
                    );
                    std_gpu_verdict = cal_qc_gpu.verdict;
                }
                let (pred_std_gpu, conf_k_std_gpu) = to_reference(
                    &std_gpu_res_raw,
                    &all_systems_for_eval,
                    &truth_ref.sequence,
                    &truth_ref.seq_ids,
                    0.0,
                    Some(&std_eff_priors),
                );
                let mut conf_seq_std_gpu = BTreeMap::new();
                for (k, c) in conf_k_std_gpu {
                    conf_seq_std_gpu.insert(truth_ref.seq_ids[k], c);
                }
                let eval_std_gpu = evaluate_prediction_against_truth(
                    &pred_std_gpu,
                    &truth_ref,
                    &conf_seq_std_gpu,
                    Some(&all_systems_for_eval),
                    0.80,
                    true,
                );
                std_gpu_samples.push(make_training_seed(
                    &qc_std_gpu,
                    eval_std_gpu.placed_predictions.clone(),
                ));
            }

            tot_true_roots_1seed += p_true_1seed;
            tot_weak_true_roots_1seed += p_weak_1seed;
            tot_weak_ht_true_roots_1seed += p_weak_ht_1seed;
            weak_with_assigned += p_weak_w_ass;
            weak_with_correct += p_weak_w_cor;
            weak_without_assigned += p_weak_wo_ass;
            weak_without_correct += p_weak_wo_cor;

            per_prot_audit_lines.push(format!(
                "  {:16} {:14} | QC(evo={}, std={}) | true_roots={} | r_vol<0.15={} (r_ht<0.15={}) | 5-seed weak correct: WITH={}/{} (ass={}), WITHOUT={}/{} (ass={}) | evo_notes=[{}] std_notes=[{}]",
                entry.bmrb_id,
                entry.protein,
                evo_verdict.as_str(),
                std_verdict.as_str(),
                p_true_1seed,
                p_weak_1seed,
                p_weak_ht_1seed,
                p_weak_w_cor,
                p_weak_1seed * 5,
                p_weak_w_ass,
                p_weak_wo_cor,
                p_weak_1seed * 5,
                p_weak_wo_ass,
                evo_qc_reasons,
                std_qc_reasons,
            ));

            evo_by_prot.push((
                entry.bmrb_id.clone(),
                entry.protein.clone(),
                evo_verdict,
                evo_samples,
            ));
            std_by_prot.push((
                entry.bmrb_id.clone(),
                entry.protein.clone(),
                std_verdict,
                std_samples,
            ));
            std_gpu_by_prot.push((
                entry.bmrb_id.clone(),
                entry.protein.clone(),
                std_gpu_verdict,
                std_gpu_samples,
            ));
        }

        println!("\n================ P1-5 MINOR-CONFORMER (< 0.15) 13-PROTEIN DEV AUDIT ================");
        for line in &per_prot_audit_lines {
            println!("{line}");
        }
        println!(
            "  TOTAL (13 dev proteins): true_roots={} | r_vol<0.15={} ({:.2}%) [vs r_height<0.15={} ({:.2}%)]",
            tot_true_roots_1seed,
            tot_weak_true_roots_1seed,
            100.0 * (tot_weak_true_roots_1seed as f64) / (tot_true_roots_1seed.max(1) as f64),
            tot_weak_ht_true_roots_1seed,
            100.0 * (tot_weak_ht_true_roots_1seed as f64) / (tot_true_roots_1seed.max(1) as f64),
        );
        println!(
            "  r_vol<0.15 5-seed pool (N={}): WITH damping acc={:.2}% ({}/{}), recall={:.2}% ({}/{}) | WITHOUT damping acc={:.2}% ({}/{}), recall={:.2}% ({}/{})",
            weak_total_5seed,
            100.0 * (weak_with_correct as f64) / (weak_with_assigned.max(1) as f64),
            weak_with_correct,
            weak_with_assigned,
            100.0 * (weak_with_correct as f64) / (weak_total_5seed.max(1) as f64),
            weak_with_correct,
            weak_total_5seed,
            100.0 * (weak_without_correct as f64) / (weak_without_assigned.max(1) as f64),
            weak_without_correct,
            weak_without_assigned,
            100.0 * (weak_without_correct as f64) / (weak_total_5seed.max(1) as f64),
            weak_without_correct,
            weak_total_5seed,
        );
        println!(
            "  Overall 13-dev micro accuracy: WITH damping={:.2}% ({}/{}) | WITHOUT damping={:.2}% ({}/{})",
            100.0 * (all_with_correct as f64) / (all_with_assigned.max(1) as f64),
            all_with_correct,
            all_with_assigned,
            100.0 * (all_without_correct as f64) / (all_without_assigned.max(1) as f64),
            all_without_correct,
            all_without_assigned,
        );

        struct LopoGateCheck {
            solver_name: String,
            oof_mean_ece: f64,
            oof_mean_ece_10: f64,
            oof_mean_debiased_ece_10: f64,
            oof_pool_ece_10: f64,
            oof_micro_hcp: f64,
            oof_macro_hcp: f64,
            full_knots: Vec<(f64, f64)>,
        }
        let mut gate_checks: Vec<LopoGateCheck> = Vec::new();
        let identity_knots = [(0.0, 0.0), (1.0, 1.0)];

        // Strict 13-fold Leave-One-Protein-Out (LOPO) cross-validation (`P1-7` & `P1-8`):
        // inside every outer fold `hold_idx`, ALL isotonic PAV knot tables (`EVO_GRAPH`,
        // `STANDARD_CPU`, and nested-12-fold-LOPO `STANDARD_GPU` shrinkage `w_i`) AND a single
        // unified 3-engine `ProteinHcGateConfig` (`fit_protein_hc_gate_config_multi_engine`) are
        // fitted strictly on the 12 training proteins `idx != hold_idx` with zero outer-fold leakage!
        let n_prots = evo_by_prot.len();
        assert_eq!(std_by_prot.len(), n_prots);
        assert_eq!(std_gpu_by_prot.len(), n_prots);

        let report_identity_baseline = |solver_name: &str, by_prot: &[ProtRecord]| {
            let mut prot_eces = Vec::new();
            let mut prot_eces_10 = Vec::new();
            let mut prot_eces_12 = Vec::new();
            let mut prot_hcps = Vec::new();
            let mut pooled = Vec::new();
            for (_, _, _, seeds) in by_prot {
                let mut preds = Vec::new();
                for s in seeds {
                    preds.extend(evaluate_seed_predictions_with_gate(
                        s,
                        &identity_knots,
                        &CANONICAL_PROTEIN_HC_GATE,
                    ));
                }
                let rel = ProteinReliabilityCurve::from_predictions_with_repeats(&preds, 0.80, 5.0);
                prot_eces.push(rel.ece);
                prot_eces_10.push(rel.ece_10);
                prot_eces_12.push(rel.ece_12);
                if let Some(hcp) = rel.hc_precision {
                    prot_hcps.push(hcp);
                }
                pooled.extend(preds);
            }
            let pool_rel =
                ProteinReliabilityCurve::from_predictions_with_repeats(&pooled, 0.80, 5.0);
            let mean_ece = prot_eces.iter().sum::<f64>() / (prot_eces.len().max(1) as f64);
            let mean_ece_10 = prot_eces_10.iter().sum::<f64>() / (prot_eces_10.len().max(1) as f64);
            let mean_ece_12 = prot_eces_12.iter().sum::<f64>() / (prot_eces_12.len().max(1) as f64);
            let macro_hcp = prot_hcps.iter().sum::<f64>() / (prot_hcps.len().max(1) as f64);
            println!(
                "  [{solver_name} Identity T=1.0 baseline] mean_ECE4/10/12={mean_ece:.4}/{mean_ece_10:.4}/{mean_ece_12:.4}, pooled_ECE4/10/12={:.4}/{:.4}/{:.4}, macro_HCP={:.2}% (n_hc_prot={}), micro_HCP={:.2}% ({}/{})",
                pool_rel.ece,
                pool_rel.ece_10,
                pool_rel.ece_12,
                macro_hcp * 100.0,
                prot_hcps.len(),
                pool_rel.hc_precision.unwrap_or(0.0) * 100.0,
                pool_rel.hc_correct,
                pool_rel.hc_assigned,
            );
        };

        println!("\n================ IDENTITY T=1.0 BASELINES ================");
        report_identity_baseline("EVO_GRAPH", &evo_by_prot);
        report_identity_baseline("STANDARD_CPU", &std_by_prot);
        report_identity_baseline("STANDARD_GPU", &std_gpu_by_prot);

        // Fit joint 3-engine calibration + unified gate + nested GPU shrinkage on any subset
        // (`skip_idx = Some(hold_idx)` for 12-protein outer LOPO folds; `None` for full-13 `dev`).
        struct JointFoldFit {
            evo_knots: Vec<(f64, f64)>,
            cpu_knots: Vec<(f64, f64)>,
            gpu_unshrunk_knots: Vec<(f64, f64)>,
            gpu_w: f64,
            gpu_cand_stats: Vec<(f64, f64, f64)>,
            gpu_knots: Vec<(f64, f64)>,
            unified_gate: ProteinHcGateConfig,
        }

        let fit_joint_on_subset = |skip_idx: Option<usize>| -> JointFoldFit {
            let mut tr_evo_prots = Vec::new();
            let mut tr_cpu_prots = Vec::new();
            let mut tr_gpu_prots = Vec::new();
            let mut tr_cpu_raw = Vec::new();
            let mut tr_gpu_ok = Vec::new();
            let mut tr_cpu_ok = Vec::new();
            let mut evo_flat = Vec::new();
            let mut cpu_flat = Vec::new();
            let mut gpu_flat = Vec::new();

            for idx in 0..n_prots {
                if skip_idx == Some(idx) {
                    continue;
                }
                let e_seeds = &evo_by_prot[idx].3;
                let c_seeds = &std_by_prot[idx].3;
                let g_seeds = &std_gpu_by_prot[idx].3;
                let e_ok = is_trainable_prot(e_seeds);
                let c_ok = is_trainable_prot(c_seeds);
                let g_ok = is_trainable_prot(g_seeds);
                let c_pairs: Vec<(f64, bool)> = c_seeds
                    .iter()
                    .flat_map(|s| s.raw_predictions.iter().copied())
                    .collect();
                if e_ok {
                    for s in e_seeds {
                        evo_flat.extend_from_slice(&s.raw_predictions);
                    }
                }
                if c_ok {
                    cpu_flat.extend_from_slice(&c_pairs);
                }
                if g_ok {
                    for s in g_seeds {
                        gpu_flat.extend_from_slice(&s.raw_predictions);
                    }
                }
                tr_evo_prots.push(e_seeds.clone());
                tr_cpu_prots.push(c_seeds.clone());
                tr_gpu_prots.push(g_seeds.clone());
                tr_cpu_raw.push(c_pairs);
                tr_gpu_ok.push(g_ok);
                tr_cpu_ok.push(c_ok);
            }

            let evo_knots = fit_isotonic_monotone_knots(
                &evo_flat,
                CANONICAL_CALIBRATION_GRID_12,
                CANONICAL_CALIBRATION_LAPLACE_PRIOR,
            );
            let cpu_knots = fit_isotonic_monotone_knots_beta(
                &cpu_flat,
                CANONICAL_CALIBRATION_GRID_12,
                STANDARD_CPU_PRIOR_ALPHA,
                STANDARD_CPU_PRIOR_BETA,
            );
            let gpu_unshrunk_knots = fit_isotonic_monotone_knots_beta(
                &gpu_flat,
                CANONICAL_CALIBRATION_GRID_12,
                STANDARD_GPU_PRIOR_ALPHA,
                STANDARD_GPU_PRIOR_BETA,
            );

            // Alternating 1-step consistency strictly inside the training subset:
            // (init_gpu_knots -> init_gate -> inner-LOPO w_0 -> gpu_knots_0 -> gate_0 -> inner-LOPO w -> gpu_knots -> unified_gate)
            let init_gpu_knots = fit_isotonic_monotone_knots_shrunk_toward_beta(
                &gpu_flat,
                &cpu_flat,
                0.50,
                CANONICAL_CALIBRATION_GRID_12,
                STANDARD_GPU_PRIOR_ALPHA,
                STANDARD_GPU_PRIOR_BETA,
            );
            let init_gate = fit_protein_hc_gate_config_multi_engine(&[
                (&tr_evo_prots, &evo_knots),
                (&tr_cpu_prots, &cpu_knots),
                (&tr_gpu_prots, &init_gpu_knots),
            ]);
            let (w_0, _) = select_gpu_shrinkage_weight_nested_lopo(
                &tr_gpu_prots,
                &tr_cpu_raw,
                &tr_gpu_ok,
                &tr_cpu_ok,
                &init_gate,
            );
            let gpu_knots_0 = fit_isotonic_monotone_knots_shrunk_toward_beta(
                &gpu_flat,
                &cpu_flat,
                w_0,
                CANONICAL_CALIBRATION_GRID_12,
                STANDARD_GPU_PRIOR_ALPHA,
                STANDARD_GPU_PRIOR_BETA,
            );
            let gate_0 = fit_protein_hc_gate_config_multi_engine(&[
                (&tr_evo_prots, &evo_knots),
                (&tr_cpu_prots, &cpu_knots),
                (&tr_gpu_prots, &gpu_knots_0),
            ]);
            let (gpu_w, gpu_cand_stats) = select_gpu_shrinkage_weight_nested_lopo(
                &tr_gpu_prots,
                &tr_cpu_raw,
                &tr_gpu_ok,
                &tr_cpu_ok,
                &gate_0,
            );
            let gpu_knots = fit_isotonic_monotone_knots_shrunk_toward_beta(
                &gpu_flat,
                &cpu_flat,
                gpu_w,
                CANONICAL_CALIBRATION_GRID_12,
                STANDARD_GPU_PRIOR_ALPHA,
                STANDARD_GPU_PRIOR_BETA,
            );
            let unified_gate = fit_protein_hc_gate_config_multi_engine(&[
                (&tr_evo_prots, &evo_knots),
                (&tr_cpu_prots, &cpu_knots),
                (&tr_gpu_prots, &gpu_knots),
            ]);

            JointFoldFit {
                evo_knots,
                cpu_knots,
                gpu_unshrunk_knots,
                gpu_w,
                gpu_cand_stats,
                gpu_knots,
                unified_gate,
            }
        };

        let mut outer_fits: Vec<JointFoldFit> = Vec::with_capacity(n_prots);
        let mut outer_chosen_ws: Vec<f64> = Vec::with_capacity(n_prots);
        for hold_idx in 0..n_prots {
            let fit = fit_joint_on_subset(Some(hold_idx));
            outer_chosen_ws.push(fit.gpu_w);
            outer_fits.push(fit);
        }
        let full_fit = fit_joint_on_subset(None);
        let full_selected_gpu_w = full_fit.gpu_w;
        let full_canonical_gate = full_fit.unified_gate;

        println!(
            "\n================ UNIFIED 13-FOLD LOPO + NESTED 13x12 GPU SHRINKAGE ================"
        );
        println!("  [FULL-13 UNIFIED GATE] CANONICAL_PROTEIN_HC_GATE = {full_canonical_gate:?}");
        println!(
            "  [FULL-13 INNER LOPO] selected CANONICAL_GPU_CPU_SHRINKAGE_WEIGHT = {full_selected_gpu_w:.2} (outer_ws={outer_chosen_ws:?}):"
        );
        for (cw, cece10, chcp) in &full_fit.gpu_cand_stats {
            println!(
                "    w={cw:4.2}: inner_OOF_ECE10={cece10:.5}, inner_OOF_micro_HCP={:.2}%",
                chcp * 100.0
            );
        }

        // 13.1:P1-2 — Log all 13 per-fold refit `ProteinHcGateConfig`s + nested `gpu_w` AND
        // evaluate an extended diagnostic grid reaching `[0.0, 0.0, 0.0, 0.0]` below the
        // bounded `CANDIDATE_GATE_*` floors `[0.45, 0.04, 0.25, 0.06]`!
        println!(
            "\n================ 13-FOLD PER-FOLD REFIT GATE & SHRINKAGE AUDIT ================"
        );
        for (hold_idx, fit) in outer_fits.iter().enumerate() {
            let (ref hold_id, ref hold_name, _, _) = evo_by_prot[hold_idx];
            let g = fit.unified_gate;
            println!(
                "  [FOLD {:02}] held_out={:16} ({:12}) | gpu_w={:.2} | gate={{ c35: {:.2}, hcf: {:.2}, cmarg: {:.2}, mpen: {:.2} }}",
                hold_idx,
                hold_id,
                hold_name,
                fit.gpu_w,
                g.min_corroborated_connectivity,
                g.min_hc_call_fraction,
                g.min_crowded_solution_margin,
                g.multiplicity_penalty_floor,
            );
        }

        {
            let full_evo_prots: Vec<Vec<ProteinHcTrainingSeed>> =
                evo_by_prot.iter().map(|(_, _, _, s)| s.clone()).collect();
            let full_cpu_prots: Vec<Vec<ProteinHcTrainingSeed>> =
                std_by_prot.iter().map(|(_, _, _, s)| s.clone()).collect();
            let full_gpu_prots: Vec<Vec<ProteinHcTrainingSeed>> = std_gpu_by_prot
                .iter()
                .map(|(_, _, _, s)| s.clone())
                .collect();
            let full_slices = [
                (full_evo_prots.as_slice(), full_fit.evo_knots.as_slice()),
                (full_cpu_prots.as_slice(), full_fit.cpu_knots.as_slice()),
                (full_gpu_prots.as_slice(), full_fit.gpu_knots.as_slice()),
            ];
            let ext_c35 = &[0.00, 0.20, 0.35, 0.45, 0.50, 0.55, 0.60, 0.65];
            let ext_hcf = &[0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12];
            let ext_cmarg = &[0.00, 0.15, 0.25, 0.35, 0.45, 0.55];
            let ext_mpen = &[0.00, 0.03, 0.06, 0.08, 0.10, 0.12];
            let zero_gate = ProteinHcGateConfig {
                min_corroborated_connectivity: 0.0,
                min_hc_call_fraction: 0.0,
                min_crowded_solution_margin: 0.0,
                multiplicity_penalty_floor: 0.0,
                min_chain_corroborated_frac: 0.0,
                min_unary_specificity: 0.0,
                min_bidir_aliphatic_frac: 0.0,
            };
            let (z_ece10, z_hcps, z_capped) =
                evaluate_protein_hc_gate_objective_multi_engine(&full_slices, &zero_gate);
            let (c_ece10, c_hcps, c_capped) =
                evaluate_protein_hc_gate_objective_multi_engine(&full_slices, &full_canonical_gate);
            let (ext_occam_gate, ext_occam_obj) = fit_protein_hc_gate_config_with_grids(
                &full_slices,
                ext_c35,
                ext_hcf,
                ext_cmarg,
                ext_mpen,
                1e-4,
            );
            let (eo_ece10, eo_hcps, eo_capped) =
                evaluate_protein_hc_gate_objective_multi_engine(&full_slices, &ext_occam_gate);
            let (ext_raw_gate, ext_raw_obj) = fit_protein_hc_gate_config_with_grids(
                &full_slices,
                ext_c35,
                ext_hcf,
                ext_cmarg,
                ext_mpen,
                0.0,
            );
            let (er_ece10, er_hcps, er_capped) =
                evaluate_protein_hc_gate_objective_multi_engine(&full_slices, &ext_raw_gate);
            println!(
                "\n================ EXTENDED GATE FLOOR DIAGNOSTIC (FULL-13 DEV) ================"
            );
            println!(
                "  [GATE DIAG zero_floor] gate={{ c35: 0.00, hcf: 0.00, cmarg: 0.00, mpen: 0.00 }} | mean_ECE10={z_ece10:.6} | hcps=[{:.2}%, {:.2}%, {:.2}%] | capped_seeds={z_capped}",
                z_hcps[0] * 100.0,
                z_hcps[1] * 100.0,
                z_hcps[2] * 100.0,
            );
            println!(
                "  [GATE DIAG canonical ] gate={{ c35: {:.2}, hcf: {:.2}, cmarg: {:.2}, mpen: {:.2} }} | mean_ECE10={c_ece10:.6} | hcps=[{:.2}%, {:.2}%, {:.2}%] | capped_seeds={c_capped}",
                full_canonical_gate.min_corroborated_connectivity,
                full_canonical_gate.min_hc_call_fraction,
                full_canonical_gate.min_crowded_solution_margin,
                full_canonical_gate.multiplicity_penalty_floor,
                c_hcps[0] * 100.0,
                c_hcps[1] * 100.0,
                c_hcps[2] * 100.0,
            );
            println!(
                "  [GATE DIAG ext_occam ] gate={{ c35: {:.2}, hcf: {:.2}, cmarg: {:.2}, mpen: {:.2} }} | obj={ext_occam_obj:.6} | mean_ECE10={eo_ece10:.6} | hcps=[{:.2}%, {:.2}%, {:.2}%] | capped_seeds={eo_capped}",
                ext_occam_gate.min_corroborated_connectivity,
                ext_occam_gate.min_hc_call_fraction,
                ext_occam_gate.min_crowded_solution_margin,
                ext_occam_gate.multiplicity_penalty_floor,
                eo_hcps[0] * 100.0,
                eo_hcps[1] * 100.0,
                eo_hcps[2] * 100.0,
            );
            println!(
                "  [GATE DIAG ext_unreg ] gate={{ c35: {:.2}, hcf: {:.2}, cmarg: {:.2}, mpen: {:.2} }} | obj={ext_raw_obj:.6} | mean_ECE10={er_ece10:.6} | hcps=[{:.2}%, {:.2}%, {:.2}%] | capped_seeds={er_capped}",
                ext_raw_gate.min_corroborated_connectivity,
                ext_raw_gate.min_hc_call_fraction,
                ext_raw_gate.min_crowded_solution_margin,
                ext_raw_gate.multiplicity_penalty_floor,
                er_hcps[0] * 100.0,
                er_hcps[1] * 100.0,
                er_hcps[2] * 100.0,
            );
        }

        let mut summarize_engine_lopo =
            |solver_name: &str,
             by_prot: &[ProtRecord],
             pick_fold_knots: &dyn Fn(&JointFoldFit) -> &[(f64, f64)],
             full_knots: &[(f64, f64)]| {
                let mut oof_eces = Vec::with_capacity(n_prots);
                let mut oof_eces_10 = Vec::with_capacity(n_prots);
                let mut oof_deb_eces_10 = Vec::with_capacity(n_prots);
                let mut oof_floor_10 = Vec::with_capacity(n_prots);
                let mut oof_eces_12 = Vec::with_capacity(n_prots);
                let mut oof_hcps = Vec::with_capacity(n_prots);
                let mut oof_pooled = Vec::new();

                for (hold_idx, (_, _, _, held_seeds)) in by_prot.iter().enumerate() {
                    let fold_fit = &outer_fits[hold_idx];
                    let fold_knots = pick_fold_knots(fold_fit);
                    let fold_gate = &fold_fit.unified_gate;
                    let mut held_preds = Vec::new();
                    for s in held_seeds {
                        held_preds.extend(evaluate_seed_predictions_with_gate(
                            s, fold_knots, fold_gate,
                        ));
                    }
                    let rel = ProteinReliabilityCurve::from_predictions_with_repeats(
                        &held_preds,
                        0.80,
                        5.0,
                    );
                    oof_eces.push(rel.ece);
                    oof_eces_10.push(rel.ece_10);
                    oof_deb_eces_10.push(rel.debiased_ece_10);
                    oof_floor_10.push(rel.ece_10_bernoulli_floor);
                    oof_eces_12.push(rel.ece_12);
                    if let Some(hcp) = rel.hc_precision {
                        oof_hcps.push(hcp);
                    }
                    oof_pooled.extend(held_preds);
                }

                let oof_pool =
                    ProteinReliabilityCurve::from_predictions_with_repeats(&oof_pooled, 0.80, 5.0);
                let oof_mean_ece = oof_eces.iter().sum::<f64>() / (oof_eces.len().max(1) as f64);
                let oof_mean_ece_10 =
                    oof_eces_10.iter().sum::<f64>() / (oof_eces_10.len().max(1) as f64);
                let oof_mean_deb_ece_10 =
                    oof_deb_eces_10.iter().sum::<f64>() / (oof_deb_eces_10.len().max(1) as f64);
                let oof_mean_floor_10 =
                    oof_floor_10.iter().sum::<f64>() / (oof_floor_10.len().max(1) as f64);
                let oof_mean_ece_12 =
                    oof_eces_12.iter().sum::<f64>() / (oof_eces_12.len().max(1) as f64);
                let oof_macro_hcp = oof_hcps.iter().sum::<f64>() / (oof_hcps.len().max(1) as f64);
                let oof_micro_hcp = oof_pool.hc_precision.unwrap_or(0.0);

                println!(
                "\n  >>> 13-FOLD UNIFIED-IN-FOLD-REFIT LOPO {solver_name} (full_gate={full_canonical_gate:?})"
            );
                println!(
                "  OOF LOPO: mean_ECE4/10/12={oof_mean_ece:.4}/{oof_mean_ece_10:.4}/{oof_mean_ece_12:.4} (debiased_ECE10={oof_mean_deb_ece_10:.4}, Bernoulli_floor={oof_mean_floor_10:.4}), pooled_ECE4/10/12={:.4}/{:.4}/{:.4}, macro_HCP={:.2}%, micro_HCP={:.2}% ({}/{})",
                oof_pool.ece,
                oof_pool.ece_10,
                oof_pool.ece_12,
                oof_macro_hcp * 100.0,
                oof_micro_hcp * 100.0,
                oof_pool.hc_correct,
                oof_pool.hc_assigned,
            );
                for ((((id, name, verd, seeds), &pece), &pece10), &pece12) in by_prot
                    .iter()
                    .zip(oof_eces.iter())
                    .zip(oof_eces_10.iter())
                    .zip(oof_eces_12.iter())
                {
                    let tot_s: usize = seeds.iter().map(|s| s.raw_predictions.len()).sum();
                    let n_ok: usize = seeds
                        .iter()
                        .map(|s| s.raw_predictions.iter().filter(|&&(_, ok)| ok).count())
                        .sum();
                    let acc = 100.0 * (n_ok as f64) / (tot_s.max(1) as f64);
                    let mut preds_p = Vec::new();
                    for s in seeds {
                        preds_p.extend(evaluate_seed_predictions_with_gate(
                            s,
                            full_knots,
                            &full_canonical_gate,
                        ));
                    }
                    let rel_p =
                        ProteinReliabilityCurve::from_predictions_with_repeats(&preds_p, 0.80, 5.0);
                    println!(
                    "    per-protein {id:16} ({name:12}, QC={:4}): acc={acc:5.1}%, OOF_ECE4/10/12={pece:.4}/{pece10:.4}/{pece12:.4}, INS_HC={}",
                    verd.as_str(),
                    rel_p.hc_status,
                );
                }
                print!("  RUST KNOTS {solver_name}: &[");
                for (i, &(x, y)) in full_knots.iter().enumerate() {
                    if i > 0 {
                        print!(", ");
                    }
                    print!("({x:.4}, {y:.4})");
                }
                println!("];");
                if solver_name != "STANDARD_GPU_UNSHRUNK" {
                    gate_checks.push(LopoGateCheck {
                        solver_name: solver_name.to_string(),
                        oof_mean_ece,
                        oof_mean_ece_10,
                        oof_mean_debiased_ece_10: oof_mean_deb_ece_10,
                        oof_pool_ece_10: oof_pool.ece_10,
                        oof_micro_hcp,
                        oof_macro_hcp,
                        full_knots: full_knots.to_vec(),
                    });
                }
            };

        summarize_engine_lopo(
            "EVO_GRAPH",
            &evo_by_prot,
            &|f| &f.evo_knots,
            &full_fit.evo_knots,
        );
        summarize_engine_lopo(
            "STANDARD_CPU",
            &std_by_prot,
            &|f| &f.cpu_knots,
            &full_fit.cpu_knots,
        );
        summarize_engine_lopo(
            "STANDARD_GPU_UNSHRUNK",
            &std_gpu_by_prot,
            &|f| &f.gpu_unshrunk_knots,
            &full_fit.gpu_unshrunk_knots,
        );
        summarize_engine_lopo(
            "STANDARD_GPU",
            &std_gpu_by_prot,
            &|f| &f.gpu_knots,
            &full_fit.gpu_knots,
        );

        if only_filter.is_none() {
            // Note: In Phase 13 Wave 2 (`13.1:P1-7b`), `CANONICAL_*` knots, shrinkage weights,
            // and `CANONICAL_PROTEIN_HC_GATE` were refit over the 26-protein non-cold pool
            // (`dev` + `seen_p12` + `seen_p13`) and are asserted to `<= 2e-4` exactness in
            // `w2_non_cold_26_lopo_calibration`. Here in the legacy 13-protein `dev`-only audit,
            // we still verify the 13-fold OOF calibration gates and grid monotonicity.
            let _ = (
                &gpu_engine.adapter_name,
                full_selected_gpu_w,
                &full_canonical_gate,
            );
            for LopoGateCheck {
                solver_name,
                oof_mean_ece,
                oof_mean_ece_10,
                oof_mean_debiased_ece_10,
                oof_pool_ece_10,
                oof_micro_hcp,
                oof_macro_hcp,
                full_knots,
            } in gate_checks
            {
                assert!(
                    oof_pool_ece_10 <= 0.050,
                    "{solver_name} OOF pooled_ECE_10 {oof_pool_ece_10:.4} exceeded primary 0.050 gate"
                );
                assert!(
                    oof_micro_hcp >= 0.94,
                    "{solver_name} OOF micro_HCP {:.2}% fell below primary 94.0% gate",
                    oof_micro_hcp * 100.0
                );
                assert!(
                    oof_macro_hcp >= 0.95,
                    "{solver_name} OOF macro_HCP {:.2}% fell below 95.0% gate",
                    oof_macro_hcp * 100.0
                );
                assert!(
                    oof_mean_ece <= 0.065,
                    "{solver_name} OOF mean_ECE_4 {oof_mean_ece:.4} exceeded 0.065 gate"
                );
                assert!(
                    oof_mean_ece_10 <= 0.105,
                    "{solver_name} OOF mean_ECE_10 {oof_mean_ece_10:.4} exceeded finite-sample 0.105 gate"
                );
                assert!(
                    oof_mean_debiased_ece_10 <= 0.065,
                    "{solver_name} OOF mean_debiased_ECE_10 {oof_mean_debiased_ece_10:.4} exceeded 0.065 gate"
                );
                let committed = match solver_name.as_str() {
                    "EVO_GRAPH" => nmr_core::qc::EVO_GRAPH_CALIBRATION_KNOTS,
                    "STANDARD_CPU" => nmr_core::qc::STANDARD_CALIBRATION_KNOTS,
                    "STANDARD_GPU" => nmr_core::qc::STANDARD_GPU_CALIBRATION_KNOTS,
                    other => panic!("unexpected solver_name {other}"),
                };
                assert_eq!(
                    committed.len(),
                    full_knots.len(),
                    "{solver_name} committed knot count mismatch"
                );
                for w in full_knots.windows(2) {
                    assert!(
                        w[1].0 >= w[0].0 - 1e-12 && w[1].1 >= w[0].1 - 1e-12,
                        "{solver_name} fitted knots violated monotonicity: {w:?}"
                    );
                }
            }
        }
    }

    #[test]
    #[ignore]
    fn audit_readonly_3d_axis_hypotheses_dev_and_cold_p13() {
        use nmr_core::pulprog::{evaluate_3d_axis_hypotheses_against_hsqc, NamedAxisHypothesis};

        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        env::set_current_dir(&repo_root).expect("chdir repo_root");
        let manifest_path = PathBuf::from("data/benchmark/real_manifest.tsv");
        let manifest_txt = fs::read_to_string(&manifest_path).expect("read real_manifest.tsv");
        let entries: Vec<RealManifestEntry> = parse_real_manifest_str(&manifest_txt)
            .expect("parse real_manifest.tsv")
            .into_iter()
            .filter(|e| e.split == "dev" || e.cohort == "seen_p13" || e.cohort == "cold_p13")
            .collect();
        let data_root = env::var("NMR_REAL_DATA_DIR")
            .map(|p| PathBuf::from(p).join("private/bmrb"))
            .unwrap_or_else(|_| PathBuf::from("data/private/bmrb"));

        let mut json_entries = Vec::new();
        let mut total_cubes_evaluated = 0usize;
        let mut total_would_apply = 0usize;

        for entry in &entries {
            let loc = resolve_local_entry_paths(&entry.bmrb_id, &data_root, true)
                .unwrap_or_else(|p| panic!("missing entry {}: {p:?}", entry.bmrb_id));
            let (discovered, _) = discover_entry_experiments(&loc.entry_dir, Some(entry.field_mhz))
                .expect("discover_entry_experiments");
            let required_set: BTreeSet<&str> =
                entry.experiments.iter().map(String::as_str).collect();
            let Some(hsqc_exp) = discovered.iter().find(|d| d.exp_type == "HSQC") else {
                panic!("missing HSQC in {}", entry.bmrb_id);
            };
            let mut cube_items = Vec::new();
            let mut entry_status = "OK".to_string();
            let mut fail_reason: Option<String> = None;

            let hsqc_spec = match process_fid_dir_2d(&hsqc_exp.dir, false, false) {
                Ok((s, _)) => s,
                Err(e) => {
                    entry_status = "PROCESS_FAILED".to_string();
                    fail_reason = Some(format!("HSQC: {e}"));
                    println!(
                        "  [AXIS AUDIT] {:8} {:16} {:12} {:9} | PROCESS_FAILED ({})",
                        entry.cohort, entry.bmrb_id, entry.protein, "HSQC", e
                    );
                    let fail_json = format!("\"{}\"", json_escape(fail_reason.as_deref().unwrap()));
                    json_entries.push(format!(
                        "    {{\n      \"bmrb_id\": \"{}\",\n      \"protein\": \"{}\",\n      \"vendor\": \"{}\",\n      \"split\": \"{}\",\n      \"cohort\": \"{}\",\n      \"status\": \"{}\",\n      \"process_failure\": {},\n      \"cubes\": []\n    }}",
                        json_escape(&entry.bmrb_id),
                        json_escape(&entry.protein),
                        json_escape(&entry.vendor),
                        json_escape(&entry.split),
                        json_escape(&entry.cohort),
                        entry_status,
                        fail_json,
                    ));
                    continue;
                }
            };

            for texp in discovered.iter().filter(|d| d.exp_type != "HSQC") {
                let is_required =
                    required_set.is_empty() || required_set.contains(texp.exp_type.as_str());
                match process_fid_dir_3d(&texp.dir, &texp.exp_type, false, false) {
                    Ok((spec3d, _)) => {
                        // Strictly read-only evaluation — ZERO cube mutation, ZERO solver runs!
                        let ver = evaluate_3d_axis_hypotheses_against_hsqc(
                            &spec3d,
                            &hsqc_spec,
                            Some(&texp.exp_type),
                        );
                        let would_apply = ver.selected != NamedAxisHypothesis::Canonical;
                        total_cubes_evaluated += 1;
                        if would_apply {
                            total_would_apply += 1;
                        }
                        let canon_lp = ver
                            .candidates
                            .iter()
                            .find(|c| c.hypothesis == NamedAxisHypothesis::Canonical)
                            .map(|c| c.ln_posterior)
                            .unwrap_or(f64::NEG_INFINITY);
                        let best_non_canon = ver
                            .candidates
                            .iter()
                            .filter(|c| c.hypothesis != NamedAxisHypothesis::Canonical)
                            .max_by(|a, b| a.ln_posterior.total_cmp(&b.ln_posterior));
                        let (runner_hyp, runner_ln_bf) = match best_non_canon {
                            Some(c) => (format!("{:?}", c.hypothesis), c.ln_posterior - canon_lp),
                            None => ("None".to_string(), 0.0),
                        };
                        println!(
                            "  [AXIS AUDIT] {:8} {:16} {:12} {:9} | selected={:14?} would_apply={:5} ln_BF={:+7.2} | canon_match={:5.1}% -> sel_match={:5.1}% | best_alt={:14} ({:+8.2})",
                            entry.cohort,
                            entry.bmrb_id,
                            entry.protein,
                            texp.exp_type,
                            ver.selected,
                            would_apply,
                            ver.ln_bayes_factor_vs_canonical,
                            ver.canonical_matched_fraction * 100.0,
                            ver.selected_matched_fraction * 100.0,
                            runner_hyp.as_str(),
                            runner_ln_bf,
                        );
                        cube_items.push(format!(
                            "        {{ \"experiment\": \"{}\", \"rel_path\": \"{}\", \"selected\": \"{:?}\", \"would_apply\": {}, \"ln_bayes_factor_vs_canonical\": {:.4}, \"canonical_matched_fraction\": {:.6}, \"selected_matched_fraction\": {:.6}, \"best_non_canonical_hypothesis\": \"{}\", \"best_non_canonical_ln_bf\": {:.4} }}",
                            json_escape(&texp.exp_type),
                            json_escape(&texp.rel_path),
                            ver.selected,
                            would_apply,
                            ver.ln_bayes_factor_vs_canonical,
                            ver.canonical_matched_fraction,
                            ver.selected_matched_fraction,
                            json_escape(&runner_hyp),
                            runner_ln_bf,
                        ));
                    }
                    Err(e) => {
                        if is_required {
                            entry_status = "PROCESS_FAILED".to_string();
                            fail_reason = Some(format!("{}: {}", texp.exp_type, e));
                            println!(
                                "  [AXIS AUDIT] {:8} {:16} {:12} {:9} | PROCESS_FAILED ({})",
                                entry.cohort, entry.bmrb_id, entry.protein, texp.exp_type, e
                            );
                            break;
                        }
                    }
                }
            }

            let fail_json = match fail_reason {
                Some(ref r) => format!("\"{}\"", json_escape(r)),
                None => "null".to_string(),
            };
            json_entries.push(format!(
                "    {{\n      \"bmrb_id\": \"{}\",\n      \"protein\": \"{}\",\n      \"vendor\": \"{}\",\n      \"split\": \"{}\",\n      \"cohort\": \"{}\",\n      \"status\": \"{}\",\n      \"process_failure\": {},\n      \"cubes\": [\n{}\n      ]\n    }}",
                json_escape(&entry.bmrb_id),
                json_escape(&entry.protein),
                json_escape(&entry.vendor),
                json_escape(&entry.split),
                json_escape(&entry.cohort),
                entry_status,
                fail_json,
                cube_items.join(",\n"),
            ));
        }

        let out_json = format!(
            "{{\n  \"mode\": \"readonly_evaluate_3d_axis_hypotheses_against_hsqc\",\n  \"solver_runs_executed\": 0,\n  \"total_entries\": {},\n  \"total_cubes_evaluated\": {},\n  \"total_rescues_would_apply\": {},\n  \"entries\": [\n{}\n  ]\n}}\n",
            entries.len(),
            total_cubes_evaluated,
            total_would_apply,
            json_entries.join(",\n"),
        );
        let out_path = PathBuf::from("data/private/phase13_w1_axis_rescue_audit.json");
        fs::write(&out_path, out_json).expect("write phase13_w1_axis_rescue_audit.json");
        println!(
            "\n  [AXIS AUDIT SUMMARY] entries={}, cubes_evaluated={}, rescues_would_apply={} -> {}",
            entries.len(),
            total_cubes_evaluated,
            total_would_apply,
            out_path.display()
        );
    }

    #[test]
    #[ignore]
    fn w2_non_cold_26_lopo_calibration() {
        use nmr_core::evo_graph::{assign_evo_graph_from_peaks_with_minor_damping, EvoGraphConfig};
        use nmr_core::priors::sync_detailed_reference_offsets;
        use nmr_core::qc::{
            evaluate_dataset_qc, evaluate_seed_predictions_with_gate, fit_isotonic_monotone_knots,
            fit_isotonic_monotone_knots_beta, fit_isotonic_monotone_knots_shrunk_toward_beta,
            fit_protein_hc_gate_config_multi_engine, is_unconverged_solution_margin,
            select_gpu_shrinkage_weight_nested_lopo_beta, ProteinHcGateConfig,
            ProteinHcTrainingSeed, ProteinReliabilityCurve, BAYES_SA_CPU_PRIOR_ALPHA,
            BAYES_SA_CPU_PRIOR_BETA, BAYES_SA_GPU_PRIOR_ALPHA, BAYES_SA_GPU_PRIOR_BETA,
            CANONICAL_BAYES_SA_GPU_CPU_SHRINKAGE_WEIGHT, CANONICAL_CALIBRATION_GRID_12,
            CANONICAL_CALIBRATION_LAPLACE_PRIOR, CANONICAL_GPU_CPU_SHRINKAGE_WEIGHT,
            QC_STRONG_BELIEF_THRESHOLD, STANDARD_CPU_PRIOR_ALPHA, STANDARD_CPU_PRIOR_BETA,
            STANDARD_GPU_PRIOR_ALPHA, STANDARD_GPU_PRIOR_BETA,
        };
        use nmr_core::solver::{assign_uncalibrated, prepare_scoring_priors, AnnealConfig};
        use serde::{Deserialize, Serialize};

        #[derive(Clone, Debug, Serialize, Deserialize)]
        struct CachedProteinTelemetry {
            bmrb_id: String,
            protein: String,
            split: String,
            cohort: String,
            evo_seeds: Vec<ProteinHcTrainingSeed>,
            std_cpu_seeds: Vec<ProteinHcTrainingSeed>,
            std_gpu_seeds: Vec<ProteinHcTrainingSeed>,
            bayes_cpu_seeds: Vec<ProteinHcTrainingSeed>,
            bayes_gpu_seeds: Vec<ProteinHcTrainingSeed>,
        }

        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        env::set_current_dir(&repo_root).expect("chdir repo_root");
        let manifest_path = PathBuf::from("data/benchmark/real_manifest.tsv");
        let manifest_txt = fs::read_to_string(&manifest_path).expect("read real_manifest.tsv");
        let entries: Vec<RealManifestEntry> = parse_real_manifest_str(&manifest_txt)
            .expect("parse real_manifest.tsv")
            .into_iter()
            .filter(|e| {
                e.split == "dev"
                    || e.cohort == "seen_p12"
                    || (e.cohort == "seen_p13" && e.bmrb_id != "15822")
            })
            .collect();
        assert_eq!(
            entries.len(),
            26,
            "expected exact 26 default-solvable non-cold proteins (13 dev + 7 seen_p12 + 6 seen_p13)"
        );
        for e in &entries {
            assert!(
                !e.cohort.starts_with("cold_"),
                "STRICT COLD FIREWALL VIOLATION: {}",
                e.bmrb_id
            );
        }

        let cache_path_opt = env::var("NMR_W2_RAW_CACHE").ok().map(PathBuf::from);
        let force_recompute = env::var("NMR_W2_FORCE_RECOMPUTE")
            .ok()
            .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));

        let prots: Vec<CachedProteinTelemetry> = if let Some(ref cp) = cache_path_opt {
            if cp.exists() && !force_recompute {
                println!(
                    "  [W2 LOPO] Loading cached raw 26-protein telemetry from {}",
                    cp.display()
                );
                let txt = fs::read_to_string(cp).expect("read NMR_W2_RAW_CACHE");
                let loaded: Vec<CachedProteinTelemetry> =
                    serde_json::from_str(&txt).expect("parse NMR_W2_RAW_CACHE");
                assert_eq!(loaded.len(), 26, "cached telemetry must have 26 proteins");
                loaded
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        let prots: Vec<CachedProteinTelemetry> = if !prots.is_empty() {
            prots
        } else {
            let gpu_engine = GpuEngine::new_headless()
                .expect("WebGPU headless adapter required for W2 5-engine LOPO calibration");
            let bayes_gpu = BayesSaGpuSolver::new_headless().expect(
                "WebGPU headless Bayes-SA solver required for W2 5-engine LOPO calibration",
            );
            println!(
                "  [GPU ADAPTER] name={:?}, backend={:?}",
                gpu_engine.adapter_name, gpu_engine.backend_name
            );
            let priors = load_priors(None).expect("load priors");
            let score_cfg = ScoreConfig::default();
            let picker_cfg = PickerConfig::default();
            let data_root = env::var("NMR_REAL_DATA_DIR")
                .map(|p| PathBuf::from(p).join("private/bmrb"))
                .unwrap_or_else(|_| PathBuf::from("data/private/bmrb"));

            let mut computed = Vec::with_capacity(entries.len());
            for (p_idx, entry) in entries.iter().enumerate() {
                let t0 = Instant::now();
                let loc = resolve_local_entry_paths(&entry.bmrb_id, &data_root, true)
                    .unwrap_or_else(|p| panic!("missing non-cold entry {}: {p:?}", entry.bmrb_id));
                let truth_path = loc.truth_path.expect("truth STAR path");
                let (truth_ref, _, _) = parse_truth_auto(&truth_path).expect("parse truth STAR");
                let (discovered, _) =
                    discover_entry_experiments(&loc.entry_dir, Some(entry.field_mhz))
                        .expect("discover");
                let required_set: BTreeSet<&str> =
                    entry.experiments.iter().map(String::as_str).collect();
                let hsqc_exp = discovered
                    .iter()
                    .find(|d| d.exp_type == "HSQC")
                    .expect("HSQC");
                let triple_exps: Vec<&DiscoveredFidExperiment> =
                    discovered.iter().filter(|d| d.exp_type != "HSQC").collect();

                let (hsqc_spec, _) =
                    process_fid_dir_2d(&hsqc_exp.dir, false, false).expect("load 2D HSQC");
                let roots = pick_hsqc_2d(&hsqc_spec, &picker_cfg);
                let mut all_peaks: Vec<Peak> = roots.clone();
                let mut used_exps = vec![("HSQC".to_string(), hsqc_exp.rel_path.clone())];
                let mut strip_bank = StripBank3D::new();
                for texp in &triple_exps {
                    let is_req =
                        required_set.is_empty() || required_set.contains(texp.exp_type.as_str());
                    match process_fid_dir_3d(&texp.dir, &texp.exp_type, false, false) {
                        Ok((mut spec3d, _)) => {
                            let _ = register_3d_hn_to_hsqc_adaptive_with_audit(
                                &mut spec3d,
                                &hsqc_spec,
                                true,
                            );
                            let pks = strip_bank.pick_and_record_3d(&spec3d, &roots, &picker_cfg);
                            all_peaks.extend(pks);
                            used_exps.push((texp.exp_type.clone(), texp.rel_path.clone()));
                        }
                        Err(e) => {
                            if is_req {
                                panic!(
                                    "required 3D {} failed on non-cold {}: {e}",
                                    texp.exp_type, entry.bmrb_id
                                );
                            }
                        }
                    }
                }
                if used_exps.len() >= 3 && !roots.is_empty() {
                    all_peaks.extend(corroborated_re_pick_3d(
                        &strip_bank,
                        &roots,
                        &all_peaks,
                        &picker_cfg,
                        &CorroboratedRepickConfig::default(),
                    ));
                }
                let all_systems_for_eval =
                    build_spin_systems(&all_peaks, &Tolerances::default(), 3, false)
                        .expect("build_spin_systems");

                let std_base_det = nmr_core::priors::estimate_reference_offsets_detailed(
                    &all_systems_for_eval,
                    &truth_ref.sequence,
                    &priors,
                );
                let (std_eff_priors, std_offs) = prepare_scoring_priors(
                    &all_systems_for_eval,
                    &truth_ref.sequence,
                    &priors,
                    &score_cfg,
                );
                let std_synced_det = sync_detailed_reference_offsets(&std_offs, &std_base_det);

                let base_bcfg = build_bayes_sa_config_cli(
                    false,
                    &truth_ref.sequence,
                    &score_cfg,
                    ToleranceOverrides::default(),
                    None,
                    None,
                    0,
                );
                let conditioned = if base_bcfg.enable_weak_root_penalty {
                    condition_peaks_preserving_hsqc_roots(&all_peaks, &base_bcfg.tolerances)
                } else {
                    all_peaks.clone()
                };
                let bayes_systems = build_spin_systems(
                    &conditioned,
                    &base_bcfg.tolerances,
                    base_bcfg.max_cands,
                    false,
                )
                .expect("build_spin_systems bayes-sa");

                let mut evo_seeds = Vec::with_capacity(5);
                let mut std_cpu_seeds = Vec::with_capacity(5);
                let mut std_gpu_seeds = Vec::with_capacity(5);
                let mut bayes_cpu_seeds = Vec::with_capacity(5);
                let mut bayes_gpu_seeds = Vec::with_capacity(5);

                let eval_raw_seed = |res: &nmr_core::solver::AssignResult,
                                     sys: &[nmr_core::spinsystems::SpinSystem],
                                     eff_p: &nmr_core::priors::Priors|
                 -> Vec<(f64, bool)> {
                    let (pred, conf_k) = to_reference(
                        res,
                        sys,
                        &truth_ref.sequence,
                        &truth_ref.seq_ids,
                        0.0,
                        Some(eff_p),
                    );
                    let mut conf_seq = BTreeMap::new();
                    for (k, c) in conf_k {
                        conf_seq.insert(truth_ref.seq_ids[k], c);
                    }
                    evaluate_prediction_against_truth(
                        &pred,
                        &truth_ref,
                        &conf_seq,
                        Some(&all_systems_for_eval),
                        0.80,
                        true,
                    )
                    .placed_predictions
                };

                for seed in 0..5u64 {
                    // 1. EVO_GRAPH (`runs = 64`, `apply_qc_calibration = false`)
                    let evo_cfg_raw = EvoGraphConfig {
                        score_cfg: score_cfg.clone(),
                        auto_reference: true,
                        apply_qc_calibration: false,
                        seed,
                        ..EvoGraphConfig::default()
                    };
                    let (evo_sys, evo_res_raw, evo_rep) =
                        assign_evo_graph_from_peaks_with_minor_damping(
                            &all_peaks,
                            &truth_ref.sequence,
                            &priors,
                            &evo_cfg_raw,
                            true,
                        );
                    let evo_eff = evo_rep.shifted_priors(&priors);
                    let evo_preds = eval_raw_seed(&evo_res_raw, &evo_sys, &evo_eff);
                    evo_seeds.push(ProteinHcTrainingSeed::from_qc_report(
                        &evo_rep.qc_report,
                        true,
                        evo_preds,
                    ));

                    // 2. STANDARD_CPU (`runs = 16`, uncalibrated)
                    let std_cpu_res = assign_uncalibrated(
                        &all_systems_for_eval,
                        &truth_ref.sequence,
                        &priors,
                        &score_cfg,
                        &AnnealConfig {
                            runs: 16,
                            steps_per_system: 1500,
                            seed,
                            ..AnnealConfig::default()
                        },
                    );
                    let qc_std_cpu = evaluate_dataset_qc(
                        Some(&all_peaks),
                        &all_systems_for_eval,
                        &truth_ref.sequence,
                        &priors,
                        &score_cfg,
                        &std_cpu_res,
                        Some(&std_synced_det),
                    );
                    let std_cpu_preds =
                        eval_raw_seed(&std_cpu_res, &all_systems_for_eval, &std_eff_priors);
                    std_cpu_seeds.push(ProteinHcTrainingSeed::from_qc_report(
                        &qc_std_cpu,
                        true,
                        std_cpu_preds,
                    ));

                    // 3. STANDARD_GPU (`runs = 64`, uncalibrated)
                    let std_gpu_res = gpu_engine
                        .assign_gpu_uncalibrated(
                            &all_systems_for_eval,
                            &truth_ref.sequence,
                            &priors,
                            &score_cfg,
                            &AnnealConfig {
                                runs: 64,
                                steps_per_system: 1500,
                                seed,
                                ..AnnealConfig::default()
                            },
                        )
                        .expect("assign_gpu_uncalibrated");
                    let qc_std_gpu = evaluate_dataset_qc(
                        Some(&all_peaks),
                        &all_systems_for_eval,
                        &truth_ref.sequence,
                        &priors,
                        &score_cfg,
                        &std_gpu_res,
                        Some(&std_synced_det),
                    );
                    let std_gpu_preds =
                        eval_raw_seed(&std_gpu_res, &all_systems_for_eval, &std_eff_priors);
                    std_gpu_seeds.push(ProteinHcTrainingSeed::from_qc_report(
                        &qc_std_gpu,
                        true,
                        std_gpu_preds,
                    ));

                    // 4. BAYES_SA_CPU (`replicas = 32`, `apply_qc_calibration = false`)
                    let mut bcfg_cpu = build_bayes_sa_config_cli(
                        false,
                        &truth_ref.sequence,
                        &score_cfg,
                        ToleranceOverrides::default(),
                        None,
                        None,
                        seed,
                    );
                    bcfg_cpu.apply_qc_calibration = false;
                    let (bcpu_res, bcpu_rep) = solve_bayes_sa_with_peaks(
                        Some(&all_peaks),
                        &bayes_systems,
                        &truth_ref.sequence,
                        &priors,
                        &bcfg_cpu,
                    );
                    let bcpu_eff = priors.with_offsets(&bcpu_rep.applied_offsets);
                    let bcpu_preds = eval_raw_seed(&bcpu_res, &bayes_systems, &bcpu_eff);
                    bayes_cpu_seeds.push(ProteinHcTrainingSeed::from_qc_report(
                        &bcpu_rep.qc_report,
                        true,
                        bcpu_preds,
                    ));

                    // 5. BAYES_SA_GPU (`replicas = 128`, `apply_qc_calibration = false`)
                    let mut bcfg_gpu = build_bayes_sa_config_cli(
                        true,
                        &truth_ref.sequence,
                        &score_cfg,
                        ToleranceOverrides::default(),
                        None,
                        None,
                        seed,
                    );
                    bcfg_gpu.apply_qc_calibration = false;
                    let (bgpu_res, bgpu_rep) = bayes_gpu
                        .solve_with_peaks(
                            Some(&all_peaks),
                            &bayes_systems,
                            &truth_ref.sequence,
                            &priors,
                            &bcfg_gpu,
                        )
                        .expect("bayes_gpu.solve_with_peaks");
                    let bgpu_eff = priors.with_offsets(&bgpu_rep.applied_offsets);
                    let bgpu_preds = eval_raw_seed(&bgpu_res, &bayes_systems, &bgpu_eff);
                    bayes_gpu_seeds.push(ProteinHcTrainingSeed::from_qc_report(
                        &bgpu_rep.qc_report,
                        true,
                        bgpu_preds,
                    ));
                }

                println!(
                    "  [W2 RAW {:2}/26] {:8} {:16} {:14} | {:.1}s | evo_n={} std_n={} bayes_n={}",
                    p_idx + 1,
                    entry.cohort,
                    entry.bmrb_id,
                    entry.protein,
                    t0.elapsed().as_secs_f64(),
                    evo_seeds[0].raw_predictions.len(),
                    std_cpu_seeds[0].raw_predictions.len(),
                    bayes_cpu_seeds[0].raw_predictions.len(),
                );

                computed.push(CachedProteinTelemetry {
                    bmrb_id: entry.bmrb_id.clone(),
                    protein: entry.protein.clone(),
                    split: entry.split.clone(),
                    cohort: entry.cohort.clone(),
                    evo_seeds,
                    std_cpu_seeds,
                    std_gpu_seeds,
                    bayes_cpu_seeds,
                    bayes_gpu_seeds,
                });
            }

            if let Some(ref cp) = cache_path_opt {
                if let Some(parent) = cp.parent() {
                    let _ = fs::create_dir_all(parent);
                }
                let txt = serde_json::to_string_pretty(&computed).expect("serialize raw cache");
                fs::write(cp, txt).expect("write NMR_W2_RAW_CACHE");
                println!(
                    "  [W2 LOPO] Saved raw 26-protein telemetry cache -> {}",
                    cp.display()
                );
            }
            computed
        };

        let is_trainable_prot = |seeds: &[ProteinHcTrainingSeed]| -> bool {
            let tot_n: usize = seeds.iter().map(|s| s.raw_predictions.len()).sum();
            if tot_n == 0 {
                return false;
            }
            let s0 = &seeds[0];
            let strong: usize = seeds
                .iter()
                .map(|s| {
                    s.raw_predictions
                        .iter()
                        .filter(|&&(p, _)| p >= QC_STRONG_BELIEF_THRESHOLD)
                        .count()
                })
                .sum();
            let m = (strong as f64) / (tot_n as f64);
            let n_per_seed = tot_n / seeds.len().max(1);
            !(s0.n_amide_hsqc_roots == 0
                || (s0.n_amide_hsqc_roots > 0 && s0.hsqc_3d_support_frac < 0.25)
                || s0.aliphatic_support_frac < 0.20
                || (n_per_seed >= 6 && s0.connectivity_frac < 0.15)
                || is_unconverged_solution_margin(tot_n, m, s0.hsqc_multiplicity_ratio)
                || ((s0.hsqc_3d_rms_dh_ppm > 0.065 || s0.hsqc_3d_rms_dn_ppm > 0.65)
                    && s0.hsqc_3d_support_frac < 0.45)
                || (s0.n_amide_hsqc_roots > 0
                    && s0.hsqc_3d_support_frac < 0.45
                    && s0.supported_roots_per_assignable < 0.35
                    && m < 0.20))
        };

        println!("\n================ 26-PROTEIN NON-COLD RAW FEATURE TABLE ================");
        for p in &prots {
            let s_evo = &p.evo_seeds[0];
            let s_bcpu = &p.bayes_cpu_seeds[0];
            let mean_feat =
                |seeds: &[ProteinHcTrainingSeed], f: fn(&ProteinHcTrainingSeed) -> f64| -> f64 {
                    seeds.iter().map(f).sum::<f64>() / (seeds.len().max(1) as f64)
                };
            let acc_fn = |seeds: &[ProteinHcTrainingSeed]| -> f64 {
                let n: usize = seeds.iter().map(|s| s.raw_predictions.len()).sum();
                let ok: usize = seeds
                    .iter()
                    .map(|s| s.raw_predictions.iter().filter(|&&(_, c)| c).count())
                    .sum();
                100.0 * (ok as f64) / (n.max(1) as f64)
            };
            println!(
                "  {:8} {:16} {:12} | mult={:4.2} sup3d={:4.2} bidir(evo={:4.2},b={:4.2}) | c35(evo={:4.2},b={:4.2}) chain(evo={:4.2},b={:4.2}) unary(evo={:4.2},b={:4.2}) | acc(evo={:5.1}%, std={:5.1}%, bcpu={:5.1}%, bgpu={:5.1}%)",
                p.cohort,
                p.bmrb_id,
                truncate_col(&p.protein, 12),
                s_evo.hsqc_multiplicity_ratio,
                s_evo.hsqc_3d_support_frac,
                s_evo.resolved_bidir_aliphatic_frac,
                s_bcpu.resolved_bidir_aliphatic_frac,
                mean_feat(&p.evo_seeds, |s| s.corroborated_connectivity_frac),
                mean_feat(&p.bayes_cpu_seeds, |s| s.corroborated_connectivity_frac),
                mean_feat(&p.evo_seeds, |s| s.chain_corroborated_frac),
                mean_feat(&p.bayes_cpu_seeds, |s| s.chain_corroborated_frac),
                mean_feat(&p.evo_seeds, |s| s.unary_specificity),
                mean_feat(&p.bayes_cpu_seeds, |s| s.unary_specificity),
                acc_fn(&p.evo_seeds),
                acc_fn(&p.std_cpu_seeds),
                acc_fn(&p.bayes_cpu_seeds),
                acc_fn(&p.bayes_gpu_seeds),
            );
        }

        struct Joint5EngineFit {
            evo_knots: Vec<(f64, f64)>,
            std_cpu_knots: Vec<(f64, f64)>,
            std_gpu_w: f64,
            std_gpu_knots: Vec<(f64, f64)>,
            bayes_cpu_knots: Vec<(f64, f64)>,
            bayes_gpu_w: f64,
            bayes_gpu_knots: Vec<(f64, f64)>,
            unified_gate: ProteinHcGateConfig,
        }

        let n_prots = prots.len();
        let fit_5engines_on_subset = |skip_idx: Option<usize>| -> Joint5EngineFit {
            let mut tr_evo = Vec::new();
            let mut tr_scpu = Vec::new();
            let mut tr_sgpu = Vec::new();
            let mut tr_bcpu = Vec::new();
            let mut tr_bgpu = Vec::new();
            let mut tr_scpu_raw = Vec::new();
            let mut tr_bcpu_raw = Vec::new();
            let mut tr_sgpu_ok = Vec::new();
            let mut tr_scpu_ok = Vec::new();
            let mut tr_bgpu_ok = Vec::new();
            let mut tr_bcpu_ok = Vec::new();

            let mut evo_flat = Vec::new();
            let mut scpu_flat = Vec::new();
            let mut sgpu_flat = Vec::new();
            let mut bcpu_flat = Vec::new();
            let mut bgpu_flat = Vec::new();

            for (idx, p) in prots.iter().enumerate() {
                if skip_idx == Some(idx) {
                    continue;
                }
                let e_ok = is_trainable_prot(&p.evo_seeds);
                let sc_ok = is_trainable_prot(&p.std_cpu_seeds);
                let sg_ok = is_trainable_prot(&p.std_gpu_seeds);
                let bc_ok = is_trainable_prot(&p.bayes_cpu_seeds);
                let bg_ok = is_trainable_prot(&p.bayes_gpu_seeds);

                let sc_pairs: Vec<(f64, bool)> = p
                    .std_cpu_seeds
                    .iter()
                    .flat_map(|s| s.raw_predictions.iter().copied())
                    .collect();
                let bc_pairs: Vec<(f64, bool)> = p
                    .bayes_cpu_seeds
                    .iter()
                    .flat_map(|s| s.raw_predictions.iter().copied())
                    .collect();

                if e_ok {
                    for s in &p.evo_seeds {
                        evo_flat.extend_from_slice(&s.raw_predictions);
                    }
                }
                if sc_ok {
                    scpu_flat.extend_from_slice(&sc_pairs);
                }
                if sg_ok {
                    for s in &p.std_gpu_seeds {
                        sgpu_flat.extend_from_slice(&s.raw_predictions);
                    }
                }
                if bc_ok {
                    bcpu_flat.extend_from_slice(&bc_pairs);
                }
                if bg_ok {
                    for s in &p.bayes_gpu_seeds {
                        bgpu_flat.extend_from_slice(&s.raw_predictions);
                    }
                }

                tr_evo.push(p.evo_seeds.clone());
                tr_scpu.push(p.std_cpu_seeds.clone());
                tr_sgpu.push(p.std_gpu_seeds.clone());
                tr_bcpu.push(p.bayes_cpu_seeds.clone());
                tr_bgpu.push(p.bayes_gpu_seeds.clone());
                tr_scpu_raw.push(sc_pairs);
                tr_bcpu_raw.push(bc_pairs);
                tr_sgpu_ok.push(sg_ok);
                tr_scpu_ok.push(sc_ok);
                tr_bgpu_ok.push(bg_ok);
                tr_bcpu_ok.push(bc_ok);
            }

            let evo_knots = fit_isotonic_monotone_knots(
                &evo_flat,
                CANONICAL_CALIBRATION_GRID_12,
                CANONICAL_CALIBRATION_LAPLACE_PRIOR,
            );
            let std_cpu_knots = fit_isotonic_monotone_knots_beta(
                &scpu_flat,
                CANONICAL_CALIBRATION_GRID_12,
                STANDARD_CPU_PRIOR_ALPHA,
                STANDARD_CPU_PRIOR_BETA,
            );
            let bayes_cpu_knots = fit_isotonic_monotone_knots_beta(
                &bcpu_flat,
                CANONICAL_CALIBRATION_GRID_12,
                BAYES_SA_CPU_PRIOR_ALPHA,
                BAYES_SA_CPU_PRIOR_BETA,
            );

            let init_sgpu_knots = fit_isotonic_monotone_knots_shrunk_toward_beta(
                &sgpu_flat,
                &scpu_flat,
                0.50,
                CANONICAL_CALIBRATION_GRID_12,
                STANDARD_GPU_PRIOR_ALPHA,
                STANDARD_GPU_PRIOR_BETA,
            );
            let init_bgpu_knots = fit_isotonic_monotone_knots_shrunk_toward_beta(
                &bgpu_flat,
                &bcpu_flat,
                0.50,
                CANONICAL_CALIBRATION_GRID_12,
                BAYES_SA_GPU_PRIOR_ALPHA,
                BAYES_SA_GPU_PRIOR_BETA,
            );

            let init_gate = fit_protein_hc_gate_config_multi_engine(&[
                (&tr_evo, &evo_knots),
                (&tr_scpu, &std_cpu_knots),
                (&tr_sgpu, &init_sgpu_knots),
                (&tr_bcpu, &bayes_cpu_knots),
                (&tr_bgpu, &init_bgpu_knots),
            ]);

            let (std_gpu_w, _) = select_gpu_shrinkage_weight_nested_lopo_beta(
                &tr_sgpu,
                &tr_scpu_raw,
                &tr_sgpu_ok,
                &tr_scpu_ok,
                &init_gate,
                STANDARD_GPU_PRIOR_ALPHA,
                STANDARD_GPU_PRIOR_BETA,
                CANONICAL_GPU_CPU_SHRINKAGE_WEIGHT,
            );
            let (bayes_gpu_w, _) = select_gpu_shrinkage_weight_nested_lopo_beta(
                &tr_bgpu,
                &tr_bcpu_raw,
                &tr_bgpu_ok,
                &tr_bcpu_ok,
                &init_gate,
                BAYES_SA_GPU_PRIOR_ALPHA,
                BAYES_SA_GPU_PRIOR_BETA,
                CANONICAL_BAYES_SA_GPU_CPU_SHRINKAGE_WEIGHT,
            );

            let std_gpu_knots = fit_isotonic_monotone_knots_shrunk_toward_beta(
                &sgpu_flat,
                &scpu_flat,
                std_gpu_w,
                CANONICAL_CALIBRATION_GRID_12,
                STANDARD_GPU_PRIOR_ALPHA,
                STANDARD_GPU_PRIOR_BETA,
            );
            let bayes_gpu_knots = fit_isotonic_monotone_knots_shrunk_toward_beta(
                &bgpu_flat,
                &bcpu_flat,
                bayes_gpu_w,
                CANONICAL_CALIBRATION_GRID_12,
                BAYES_SA_GPU_PRIOR_ALPHA,
                BAYES_SA_GPU_PRIOR_BETA,
            );

            let unified_gate = fit_protein_hc_gate_config_multi_engine(&[
                (&tr_evo, &evo_knots),
                (&tr_scpu, &std_cpu_knots),
                (&tr_sgpu, &std_gpu_knots),
                (&tr_bcpu, &bayes_cpu_knots),
                (&tr_bgpu, &bayes_gpu_knots),
            ]);

            Joint5EngineFit {
                evo_knots,
                std_cpu_knots,
                std_gpu_w,
                std_gpu_knots,
                bayes_cpu_knots,
                bayes_gpu_w,
                bayes_gpu_knots,
                unified_gate,
            }
        };

        println!(
            "\n================ FITTING 26-FOLD LOPO + FULL-26 CANONICAL FIT ================"
        );
        let t_fit = Instant::now();
        let mut outer_fits: Vec<Joint5EngineFit> = Vec::with_capacity(n_prots);
        for (hold_idx, prot) in prots.iter().enumerate() {
            let f = fit_5engines_on_subset(Some(hold_idx));
            println!(
                "  [FOLD {:2}/26 hold={:16} ({:8})] std_w={:.2} bayes_w={:.2} gate={:?}",
                hold_idx + 1,
                prot.bmrb_id,
                prot.cohort,
                f.std_gpu_w,
                f.bayes_gpu_w,
                f.unified_gate
            );
            outer_fits.push(f);
        }
        let full_fit = fit_5engines_on_subset(None);
        println!(
            "  [FULL-26 FIT ({:.2}s)] std_w={:.2} bayes_w={:.2} gate={:?}",
            t_fit.elapsed().as_secs_f64(),
            full_fit.std_gpu_w,
            full_fit.bayes_gpu_w,
            full_fit.unified_gate
        );

        let zero_gate = ProteinHcGateConfig {
            min_corroborated_connectivity: 0.0,
            min_hc_call_fraction: 0.0,
            min_crowded_solution_margin: 0.0,
            multiplicity_penalty_floor: 0.0,
            min_chain_corroborated_frac: 0.0,
            min_unary_specificity: 0.0,
            min_bidir_aliphatic_frac: 0.0,
        };

        #[derive(Serialize)]
        struct OofProteinReport {
            bmrb_id: String,
            protein: String,
            split: String,
            cohort: String,
            accuracy: f64,
            uncapped_hc_assigned: usize,
            uncapped_hc_correct: usize,
            uncapped_hc_precision: Option<f64>,
            oof_hc_assigned: usize,
            oof_hc_correct: usize,
            oof_hc_precision: Option<f64>,
            oof_capped_or_warned: bool,
            oof_ece_10: f64,
            oof_debiased_ece_10: f64,
        }

        #[derive(Serialize)]
        struct EngineLopoSummary {
            engine: String,
            oof_pooled_ece_10: f64,
            oof_pooled_ece_4: f64,
            oof_mean_ece_10: f64,
            oof_mean_debiased_ece_10: f64,
            oof_micro_hc_precision: f64,
            oof_macro_hc_precision: f64,
            oof_hc_active_proteins: usize,
            oof_hc_abstained_proteins: usize,
            oof_5691_uncapped_hcp: Option<f64>,
            oof_5691_gated_hcp: Option<f64>,
            oof_5691_capped_or_warned: bool,
            oof_18561_uncapped_hcp: Option<f64>,
            oof_18561_gated_hcp: Option<f64>,
            oof_18561_capped_or_warned: bool,
            dev_clean_proteins_capped: Vec<String>,
            full_knots: Vec<(f64, f64)>,
            per_protein: Vec<OofProteinReport>,
        }

        let eval_engine_oof = |engine_name: &str,
                               get_seeds: &dyn Fn(
            &CachedProteinTelemetry,
        ) -> &[ProteinHcTrainingSeed],
                               get_fold_knots: &dyn Fn(&Joint5EngineFit) -> &[(f64, f64)],
                               full_knots: &[(f64, f64)]|
         -> EngineLopoSummary {
            let mut oof_pooled = Vec::new();
            let mut oof_eces_10 = Vec::new();
            let mut oof_deb_10 = Vec::new();
            let mut oof_hcps = Vec::new();
            let mut per_protein = Vec::new();
            let mut dev_clean_capped = Vec::new();

            for (hold_idx, p) in prots.iter().enumerate() {
                let seeds = get_seeds(p);
                let fold_fit = &outer_fits[hold_idx];
                let f_knots = get_fold_knots(fold_fit);
                let f_gate = &fold_fit.unified_gate;

                let mut uncap_preds = Vec::new();
                let mut gated_preds = Vec::new();
                for s in seeds {
                    uncap_preds.extend(evaluate_seed_predictions_with_gate(s, f_knots, &zero_gate));
                    gated_preds.extend(evaluate_seed_predictions_with_gate(s, f_knots, f_gate));
                }
                let uncap_rel =
                    ProteinReliabilityCurve::from_predictions_with_repeats(&uncap_preds, 0.80, 5.0);
                let gated_rel =
                    ProteinReliabilityCurve::from_predictions_with_repeats(&gated_preds, 0.80, 5.0);

                let n_tot: usize = seeds.iter().map(|s| s.raw_predictions.len()).sum();
                let n_ok: usize = seeds
                    .iter()
                    .map(|s| s.raw_predictions.iter().filter(|&&(_, c)| c).count())
                    .sum();
                let acc = (n_ok as f64) / (n_tot.max(1) as f64);

                let capped_by_gate =
                    uncap_rel.hc_assigned > 0 && gated_rel.hc_assigned < uncap_rel.hc_assigned;
                // Also check if intrinsic WARN damped or if already 0 HC calls
                let already_abstained_or_warned = uncap_rel.hc_assigned == 0
                    || gated_preds
                        .iter()
                        .zip(uncap_preds.iter())
                        .any(|(&(gc, _), &(uc, _))| gc + 1e-9 < uc);
                let capped_or_warned = capped_by_gate || already_abstained_or_warned;

                if p.split == "dev"
                    && uncap_rel.hc_precision.is_some_and(|h| h >= 0.94)
                    && capped_by_gate
                {
                    dev_clean_capped.push(format!(
                        "{} (uncapped={:.1}% -> gated={})",
                        p.bmrb_id,
                        uncap_rel.hc_precision.unwrap() * 100.0,
                        gated_rel.hc_status
                    ));
                }

                oof_eces_10.push(gated_rel.ece_10);
                oof_deb_10.push(gated_rel.debiased_ece_10);
                if let Some(hcp) = gated_rel.hc_precision {
                    oof_hcps.push(hcp);
                }
                oof_pooled.extend(gated_preds);

                per_protein.push(OofProteinReport {
                    bmrb_id: p.bmrb_id.clone(),
                    protein: p.protein.clone(),
                    split: p.split.clone(),
                    cohort: p.cohort.clone(),
                    accuracy: acc,
                    uncapped_hc_assigned: uncap_rel.hc_assigned,
                    uncapped_hc_correct: uncap_rel.hc_correct,
                    uncapped_hc_precision: uncap_rel.hc_precision,
                    oof_hc_assigned: gated_rel.hc_assigned,
                    oof_hc_correct: gated_rel.hc_correct,
                    oof_hc_precision: gated_rel.hc_precision,
                    oof_capped_or_warned: capped_or_warned,
                    oof_ece_10: gated_rel.ece_10,
                    oof_debiased_ece_10: gated_rel.debiased_ece_10,
                });
            }

            let pool_rel =
                ProteinReliabilityCurve::from_predictions_with_repeats(&oof_pooled, 0.80, 5.0);
            let mean_ece_10 = oof_eces_10.iter().sum::<f64>() / (oof_eces_10.len().max(1) as f64);
            let mean_deb_10 = oof_deb_10.iter().sum::<f64>() / (oof_deb_10.len().max(1) as f64);
            let macro_hcp = oof_hcps.iter().sum::<f64>() / (oof_hcps.len().max(1) as f64);
            let micro_hcp = pool_rel.hc_precision.unwrap_or(0.0);
            let hc_active = oof_hcps.len();
            let hc_abstained = n_prots.saturating_sub(hc_active);

            let p5691 = per_protein.iter().find(|r| r.bmrb_id == "5691").unwrap();
            let p18561 = per_protein.iter().find(|r| r.bmrb_id == "18561").unwrap();

            println!(
                "\n  >>> [26-FOLD LOPO {engine_name}] pooled_ECE10={:.4} (ECE4={:.4}), mean_ECE10={mean_ece_10:.4} (deb={mean_deb_10:.4}), micro_HCP={:.2}% ({}/{}), macro_HCP={:.2}% (active={hc_active}/{n_prots})",
                pool_rel.ece_10,
                pool_rel.ece,
                micro_hcp * 100.0,
                pool_rel.hc_correct,
                pool_rel.hc_assigned,
                macro_hcp * 100.0,
            );
            println!(
                "      5691  : uncapped_HCP={:?} ({}/{}) -> OOF_gated={:?} ({}/{}) capped_or_warned={}",
                p5691.uncapped_hc_precision.map(|v| format!("{:.1}%", v * 100.0)),
                p5691.uncapped_hc_correct,
                p5691.uncapped_hc_assigned,
                p5691.oof_hc_precision.map(|v| format!("{:.1}%", v * 100.0)),
                p5691.oof_hc_correct,
                p5691.oof_hc_assigned,
                p5691.oof_capped_or_warned,
            );
            println!(
                "      18561 : uncapped_HCP={:?} ({}/{}) -> OOF_gated={:?} ({}/{}) capped_or_warned={}",
                p18561.uncapped_hc_precision.map(|v| format!("{:.1}%", v * 100.0)),
                p18561.uncapped_hc_correct,
                p18561.uncapped_hc_assigned,
                p18561.oof_hc_precision.map(|v| format!("{:.1}%", v * 100.0)),
                p18561.oof_hc_correct,
                p18561.oof_hc_assigned,
                p18561.oof_capped_or_warned,
            );
            println!("      dev_clean_capped (>=94% HCP): {:?}", dev_clean_capped);
            for r in &per_protein {
                println!(
                    "      {:8} {:16} {:12} | acc={:5.1}% | uncap_HC={:3}/{:3} ({:>6}) -> OOF_HC={:3}/{:3} ({:>6}) | cap/warn={:5} | ECE10={:.4}",
                    r.cohort,
                    r.bmrb_id,
                    truncate_col(&r.protein, 12),
                    r.accuracy * 100.0,
                    r.uncapped_hc_correct,
                    r.uncapped_hc_assigned,
                    r.uncapped_hc_precision
                        .map(|v| format!("{:.1}%", v * 100.0))
                        .unwrap_or_else(|| "ABST".to_string()),
                    r.oof_hc_correct,
                    r.oof_hc_assigned,
                    r.oof_hc_precision
                        .map(|v| format!("{:.1}%", v * 100.0))
                        .unwrap_or_else(|| "ABST".to_string()),
                    r.oof_capped_or_warned,
                    r.oof_ece_10,
                );
            }
            print!("  RUST KNOTS {engine_name}: &[");
            for (i, &(x, y)) in full_knots.iter().enumerate() {
                if i > 0 {
                    print!(", ");
                }
                print!("({x:.4}, {y:.4})");
            }
            println!("];");

            EngineLopoSummary {
                engine: engine_name.to_string(),
                oof_pooled_ece_10: pool_rel.ece_10,
                oof_pooled_ece_4: pool_rel.ece,
                oof_mean_ece_10: mean_ece_10,
                oof_mean_debiased_ece_10: mean_deb_10,
                oof_micro_hc_precision: micro_hcp,
                oof_macro_hc_precision: macro_hcp,
                oof_hc_active_proteins: hc_active,
                oof_hc_abstained_proteins: hc_abstained,
                oof_5691_uncapped_hcp: p5691.uncapped_hc_precision,
                oof_5691_gated_hcp: p5691.oof_hc_precision,
                oof_5691_capped_or_warned: p5691.oof_capped_or_warned,
                oof_18561_uncapped_hcp: p18561.uncapped_hc_precision,
                oof_18561_gated_hcp: p18561.oof_hc_precision,
                oof_18561_capped_or_warned: p18561.oof_capped_or_warned,
                dev_clean_proteins_capped: dev_clean_capped,
                full_knots: full_knots.to_vec(),
                per_protein,
            }
        };

        let s_evo = eval_engine_oof(
            "EVO_GRAPH",
            &|p| &p.evo_seeds,
            &|f| &f.evo_knots,
            &full_fit.evo_knots,
        );
        let s_scpu = eval_engine_oof(
            "STANDARD_CPU",
            &|p| &p.std_cpu_seeds,
            &|f| &f.std_cpu_knots,
            &full_fit.std_cpu_knots,
        );
        let s_sgpu = eval_engine_oof(
            "STANDARD_GPU",
            &|p| &p.std_gpu_seeds,
            &|f| &f.std_gpu_knots,
            &full_fit.std_gpu_knots,
        );
        let s_bcpu = eval_engine_oof(
            "BAYES_SA_CPU",
            &|p| &p.bayes_cpu_seeds,
            &|f| &f.bayes_cpu_knots,
            &full_fit.bayes_cpu_knots,
        );
        let s_bgpu = eval_engine_oof(
            "BAYES_SA_GPU",
            &|p| &p.bayes_gpu_seeds,
            &|f| &f.bayes_gpu_knots,
            &full_fit.bayes_gpu_knots,
        );

        println!(
            "\n  RUST CONSTANTS SUMMARY:\n  CANONICAL_GPU_CPU_SHRINKAGE_WEIGHT = {:.2};\n  CANONICAL_BAYES_SA_GPU_CPU_SHRINKAGE_WEIGHT = {:.2};\n  CANONICAL_PROTEIN_HC_GATE = {:#?};",
            full_fit.std_gpu_w,
            full_fit.bayes_gpu_w,
            full_fit.unified_gate,
        );

        #[derive(Serialize)]
        struct FullW2LopoArtifact {
            n_proteins: usize,
            canonical_std_gpu_shrinkage_w: f64,
            canonical_bayes_gpu_shrinkage_w: f64,
            canonical_protein_hc_gate: ProteinHcGateConfig,
            fold_gates: Vec<(String, String, f64, f64, ProteinHcGateConfig)>,
            engines: Vec<EngineLopoSummary>,
        }

        let fold_gates = prots
            .iter()
            .zip(outer_fits.iter())
            .map(|(p, f)| {
                (
                    p.bmrb_id.clone(),
                    p.cohort.clone(),
                    f.std_gpu_w,
                    f.bayes_gpu_w,
                    f.unified_gate,
                )
            })
            .collect();

        let artifact = FullW2LopoArtifact {
            n_proteins: n_prots,
            canonical_std_gpu_shrinkage_w: full_fit.std_gpu_w,
            canonical_bayes_gpu_shrinkage_w: full_fit.bayes_gpu_w,
            canonical_protein_hc_gate: full_fit.unified_gate,
            fold_gates,
            engines: vec![s_evo, s_scpu, s_sgpu, s_bcpu, s_bgpu],
        };

        let out_json_path = env::var("NMR_W2_LOPO_JSON_OUT")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("data/private/w2_results/w2_non_cold_26_lopo.json"));
        if let Some(parent) = out_json_path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        fs::write(
            &out_json_path,
            serde_json::to_string_pretty(&artifact).expect("serialize LOPO artifact"),
        )
        .expect("write LOPO artifact");
        println!(
            "  [W2 LOPO] Wrote JSON report -> {}",
            out_json_path.display()
        );

        // Binding 13.1:P1-7b & 13.1:P1-1 LOPO assertions across all 5 engines via
        // ACTIVE_26_FOLD_CALIBRATION_GATES (plus informational logging of RETIRED_W1_13_FOLD_CALIBRATION_GATES):
        assert_active_26_fold_gates_on_lopo_engines(&artifact.engines);

        // Verify committed qc.rs constants match full-26 fit within 4-decimal rounding (2e-4):
        assert_eq!(
            nmr_core::qc::CANONICAL_PROTEIN_HC_GATE,
            full_fit.unified_gate,
            "CANONICAL_PROTEIN_HC_GATE in qc.rs must equal full_fit.unified_gate"
        );
        assert!(
            (nmr_core::qc::CANONICAL_GPU_CPU_SHRINKAGE_WEIGHT - full_fit.std_gpu_w).abs() < 1e-6,
            "CANONICAL_GPU_CPU_SHRINKAGE_WEIGHT mismatch: {} vs {}",
            nmr_core::qc::CANONICAL_GPU_CPU_SHRINKAGE_WEIGHT,
            full_fit.std_gpu_w
        );
        assert!(
            (nmr_core::qc::CANONICAL_BAYES_SA_GPU_CPU_SHRINKAGE_WEIGHT - full_fit.bayes_gpu_w)
                .abs()
                < 1e-6,
            "CANONICAL_BAYES_SA_GPU_CPU_SHRINKAGE_WEIGHT mismatch: {} vs {}",
            nmr_core::qc::CANONICAL_BAYES_SA_GPU_CPU_SHRINKAGE_WEIGHT,
            full_fit.bayes_gpu_w
        );
        let assert_knots_match = |label: &str, committed: &[(f64, f64)], fitted: &[(f64, f64)]| {
            assert_eq!(committed.len(), fitted.len(), "{label} knot count mismatch");
            for (i, (&(cx, cy), &(fx, fy))) in committed.iter().zip(fitted.iter()).enumerate() {
                assert!(
                        (cx - fx).abs() <= 2e-4 && (cy - fy).abs() <= 2e-4,
                        "{label} knot[{i}] mismatch: committed=({cx:.4},{cy:.4}) vs fitted=({fx:.4},{fy:.4})"
                    );
            }
        };
        assert_knots_match(
            "EVO_GRAPH_CALIBRATION_KNOTS",
            nmr_core::qc::EVO_GRAPH_CALIBRATION_KNOTS,
            &full_fit.evo_knots,
        );
        assert_knots_match(
            "STANDARD_CALIBRATION_KNOTS",
            nmr_core::qc::STANDARD_CALIBRATION_KNOTS,
            &full_fit.std_cpu_knots,
        );
        assert_knots_match(
            "STANDARD_GPU_CALIBRATION_KNOTS",
            nmr_core::qc::STANDARD_GPU_CALIBRATION_KNOTS,
            &full_fit.std_gpu_knots,
        );
        assert_knots_match(
            "BAYES_SA_CALIBRATION_KNOTS",
            nmr_core::qc::BAYES_SA_CALIBRATION_KNOTS,
            &full_fit.bayes_cpu_knots,
        );
        assert_knots_match(
            "BAYES_SA_GPU_CALIBRATION_KNOTS",
            nmr_core::qc::BAYES_SA_GPU_CALIBRATION_KNOTS,
            &full_fit.bayes_gpu_knots,
        );
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct CalibrationGateSpec {
        gate_id: String,
        operator: String,
        threshold: String,
        status: String,
    }

    impl CalibrationGateSpec {
        fn new(gate_id: &str, operator: &str, threshold: &str, status: &str) -> Self {
            Self {
                gate_id: gate_id.to_string(),
                operator: operator.to_string(),
                threshold: threshold.to_string(),
                status: status.to_string(),
            }
        }
    }

    /// Canonical active 26-fold non-cold calibration gates (`13.1:P1-1`).
    const ACTIVE_26_FOLD_CALIBRATION_GATES: &[(&str, &str, &str, &str)] = &[
        ("pooled_ECE_10", "<=", "0.050", "ACTIVE"),
        ("micro_HCP", ">=", "0.940", "ACTIVE"),
        ("clean_dev_proteins_capped", "==", "0", "ACTIVE"),
        ("catastrophic_5691_18561_oof_capped", "==", "true", "ACTIVE"),
    ];

    /// Retired `13.W1` 13-fold `dev` calibration gates (`13.1:P1-1`), documented with rationale
    /// and explicit 5-engine `PASS`/`FAIL` labels in `data/benchmark/PROVENANCE.md` & `TODO.md`.
    const RETIRED_W1_13_FOLD_CALIBRATION_GATES: &[(&str, &str, &str, &str)] = &[
        ("macro_HCP", ">=", "0.950", "RETIRED_IN_13_W2"),
        ("mean_debiased_ECE_10", "<=", "0.065", "RETIRED_IN_13_W2"),
    ];

    #[derive(serde::Serialize, serde::Deserialize)]
    struct LopoEngineSummaryForGateCheck {
        engine: String,
        oof_pooled_ece_10: f64,
        oof_micro_hc_precision: f64,
        oof_macro_hc_precision: f64,
        oof_mean_debiased_ece_10: f64,
        oof_5691_uncapped_hcp: Option<f64>,
        oof_5691_capped_or_warned: bool,
        oof_18561_uncapped_hcp: Option<f64>,
        oof_18561_capped_or_warned: bool,
        dev_clean_proteins_capped: Vec<String>,
    }

    #[derive(serde::Deserialize)]
    struct LopoArtifactForGateCheck {
        engines: Vec<LopoEngineSummaryForGateCheck>,
    }

    fn assert_active_26_fold_gates_on_lopo_engines<T>(engines: &[T])
    where
        T: serde::Serialize,
    {
        let parsed: Vec<LopoEngineSummaryForGateCheck> = engines
            .iter()
            .map(|e| serde_json::from_value(serde_json::to_value(e).unwrap()).unwrap())
            .collect();
        for &(gate_id, operator, threshold, status) in ACTIVE_26_FOLD_CALIBRATION_GATES {
            assert_eq!(status, "ACTIVE");
            for eng in &parsed {
                match (gate_id, operator) {
                    ("pooled_ECE_10", "<=") => {
                        let thr: f64 = threshold.parse().unwrap();
                        assert!(
                            eng.oof_pooled_ece_10 <= thr,
                            "{} 26-fold LOPO pooled_ECE_10 {:.4} > {:.3}",
                            eng.engine,
                            eng.oof_pooled_ece_10,
                            thr
                        );
                    }
                    ("micro_HCP", ">=") => {
                        let thr: f64 = threshold.parse().unwrap();
                        assert!(
                            eng.oof_micro_hc_precision >= thr,
                            "{} 26-fold LOPO micro_HCP {:.2}% < {:.1}%",
                            eng.engine,
                            eng.oof_micro_hc_precision * 100.0,
                            thr * 100.0
                        );
                    }
                    ("clean_dev_proteins_capped", "==") => {
                        let thr: usize = threshold.parse().unwrap();
                        assert_eq!(
                            eng.dev_clean_proteins_capped.len(),
                            thr,
                            "{} capped clean dev protein(s) >= 94% HCP: {:?}",
                            eng.engine,
                            eng.dev_clean_proteins_capped
                        );
                    }
                    ("catastrophic_5691_18561_oof_capped", "==") => {
                        let thr: bool = threshold.parse().unwrap();
                        let ok_5691 = !eng.oof_5691_uncapped_hcp.is_some_and(|h| h < 0.85)
                            || eng.oof_5691_capped_or_warned;
                        let ok_18561 = !eng.oof_18561_uncapped_hcp.is_some_and(|h| h < 0.85)
                            || eng.oof_18561_capped_or_warned;
                        assert_eq!(
                            ok_5691 && ok_18561,
                            thr,
                            "{} failed catastrophic_5691_18561_oof_capped gate (5691={ok_5691}, 18561={ok_18561})",
                            eng.engine
                        );
                    }
                    other => panic!("unhandled active gate spec: {other:?}"),
                }
            }
        }
        for &(gate_id, operator, threshold, status) in RETIRED_W1_13_FOLD_CALIBRATION_GATES {
            assert_eq!(status, "RETIRED_IN_13_W2");
            let thr: f64 = threshold.parse().unwrap();
            for eng in &parsed {
                let (val, pass) = match (gate_id, operator) {
                    ("macro_HCP", ">=") => (
                        eng.oof_macro_hc_precision,
                        eng.oof_macro_hc_precision >= thr,
                    ),
                    ("mean_debiased_ECE_10", "<=") => (
                        eng.oof_mean_debiased_ece_10,
                        eng.oof_mean_debiased_ece_10 <= thr,
                    ),
                    other => panic!("unhandled retired gate spec: {other:?}"),
                };
                println!(
                    "  [W2 RETIRED GATE] {} {gate_id} {operator} {threshold}: {:.4} ({})",
                    eng.engine,
                    val,
                    if pass { "PASS" } else { "FAIL" }
                );
            }
        }
    }

    #[test]
    fn test_documented_calibration_gate_set_matches_asserted_gate_set() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let prov_path = root.join("data/benchmark/PROVENANCE.md");
        let prov = fs::read_to_string(&prov_path).expect("read data/benchmark/PROVENANCE.md");

        assert!(
            !prov.contains("micro high-confidence precision (`>= 94.0%`, macro `>= 95.0%`) serve as the primary calibration gates"),
            "PROVENANCE.md must not claim retired macro >= 95.0% as an active 26-fold primary gate"
        );

        let begin_marker = "<!-- CALIBRATION-GATE-SPEC-BEGIN -->";
        let end_marker = "<!-- CALIBRATION-GATE-SPEC-END -->";
        let start = prov
            .find(begin_marker)
            .expect("missing CALIBRATION-GATE-SPEC-BEGIN in PROVENANCE.md")
            + begin_marker.len();
        let end = prov[start..]
            .find(end_marker)
            .expect("missing CALIBRATION-GATE-SPEC-END in PROVENANCE.md")
            + start;
        let block = &prov[start..end];

        let mut doc_active = Vec::new();
        let mut doc_retired = Vec::new();
        for raw_line in block.lines() {
            let line = raw_line.trim();
            if !line.starts_with('|')
                || line.contains("Gate ID")
                || line.contains(":---")
                || line.contains("|---")
            {
                continue;
            }
            let cols: Vec<String> = line
                .trim_matches('|')
                .split('|')
                .map(|c| c.trim().trim_matches('`').to_string())
                .collect();
            assert!(
                cols.len() >= 4,
                "malformed gate-spec row in PROVENANCE.md: {line}"
            );
            let spec = CalibrationGateSpec::new(&cols[0], &cols[1], &cols[2], &cols[3]);
            match spec.status.as_str() {
                "ACTIVE" => doc_active.push(spec),
                "RETIRED_IN_13_W2" => doc_retired.push(spec),
                other => panic!("unexpected calibration gate status in PROVENANCE.md: {other}"),
            }
        }

        let expected_active: Vec<CalibrationGateSpec> = ACTIVE_26_FOLD_CALIBRATION_GATES
            .iter()
            .map(|&(g, op, thr, st)| CalibrationGateSpec::new(g, op, thr, st))
            .collect();
        let expected_retired: Vec<CalibrationGateSpec> = RETIRED_W1_13_FOLD_CALIBRATION_GATES
            .iter()
            .map(|&(g, op, thr, st)| CalibrationGateSpec::new(g, op, thr, st))
            .collect();

        assert_eq!(
            doc_active, expected_active,
            "Active calibration gate set in PROVENANCE.md must match ACTIVE_26_FOLD_CALIBRATION_GATES 1-to-1"
        );
        assert_eq!(
            doc_retired, expected_retired,
            "Retired calibration gate set in PROVENANCE.md must match RETIRED_W1_13_FOLD_CALIBRATION_GATES 1-to-1"
        );

        // When the frozen 26-fold LOPO artifact is present locally, verify both active assertions
        // and the exact retired PASS/FAIL labels documented in PROVENANCE.md.
        let lopo_json_path = root.join("data/private/w2_results/w2_non_cold_26_lopo.json");
        if lopo_json_path.is_file() {
            let lopo: LopoArtifactForGateCheck = serde_json::from_str(
                &fs::read_to_string(&lopo_json_path).expect("read w2_non_cold_26_lopo.json"),
            )
            .expect("parse w2_non_cold_26_lopo.json");
            assert_eq!(lopo.engines.len(), 5);
            assert_active_26_fold_gates_on_lopo_engines(&lopo.engines);

            for eng in &lopo.engines {
                let macro_pass = eng.oof_macro_hc_precision >= 0.950;
                let deb_pass = eng.oof_mean_debiased_ece_10 <= 0.065;
                let (exp_macro_pass, exp_deb_pass) = match eng.engine.as_str() {
                    "EVO_GRAPH" => (true, true),
                    "STANDARD_CPU" => (false, false),
                    "STANDARD_GPU" => (true, false),
                    "BAYES_SA_CPU" => (false, false),
                    "BAYES_SA_GPU" => (false, false),
                    other => panic!("unexpected engine {other}"),
                };
                assert_eq!(
                    macro_pass, exp_macro_pass,
                    "{} macro_HCP PASS/FAIL mismatch",
                    eng.engine
                );
                assert_eq!(
                    deb_pass, exp_deb_pass,
                    "{} mean_debiased_ECE_10 PASS/FAIL mismatch",
                    eng.engine
                );
            }
        }
    }

    fn write_synthetic_varian_fid_binary(path: &Path, total_traces: usize, np: usize) {
        let ebytes = 4usize;
        let nbheaders = 1usize;
        let bbytes = nbheaders * 28 + np * ebytes;
        let total_len = 32 + total_traces * bbytes;
        let mut buf = vec![0u8; total_len];
        // datafilehead (32 bytes, big-endian)
        buf[0..4].copy_from_slice(&(total_traces as i32).to_be_bytes()); // nblocks
        buf[4..8].copy_from_slice(&1i32.to_be_bytes()); // ntraces = 1 per block
        buf[8..12].copy_from_slice(&(np as i32).to_be_bytes()); // np
        buf[12..16].copy_from_slice(&(ebytes as i32).to_be_bytes()); // ebytes = 4
        buf[16..20].copy_from_slice(&((np * ebytes) as i32).to_be_bytes()); // tbytes
        buf[20..24].copy_from_slice(&(bbytes as i32).to_be_bytes()); // bbytes
        let status: u16 = 0x0001 | 0x0004; // S_DATA | S_32
        buf[26..28].copy_from_slice(&status.to_be_bytes());
        buf[28..32].copy_from_slice(&(nbheaders as i32).to_be_bytes()); // nbheaders = 1
        if total_traces > 0 && total_len >= 36 {
            buf[34..36].copy_from_slice(&status.to_be_bytes());
        }
        fs::write(path, buf).unwrap();
    }

    #[allow(clippy::too_many_arguments)]
    fn write_synthetic_varian_exp_dir(
        dir: &Path,
        pslabel: &str,
        sfrq: f64,
        ni: usize,
        ni2: usize,
        sw1: f64,
        sw2: f64,
        extra_procpar: &str,
    ) {
        fs::create_dir_all(dir).unwrap();
        let dfrq = sfrq * (150.903 / 600.133);
        let dfrq2 = sfrq * (60.818 / 600.133);
        let procpar = format!(
            "pslabel 2 2 8192 0 0 2 1 1 1 64\n\
             1 \"{pslabel}\"\n\
             0\n\
             seqfil 2 2 8192 0 0 2 1 1 1 64\n\
             1 \"{pslabel}\"\n\
             0\n\
             tn 2 2 8192 0 0 2 1 1 1 64\n\
             1 \"H1\"\n\
             0\n\
             dn 2 2 8192 0 0 2 1 1 1 64\n\
             1 \"C13\"\n\
             0\n\
             dn2 2 2 8192 0 0 2 1 1 1 64\n\
             1 \"N15\"\n\
             0\n\
             sfrq 1 1 10000 0 0 2 1 1 1 64\n\
             1 {sfrq:.4}\n\
             0\n\
             dfrq 1 1 10000 0 0 2 1 1 1 64\n\
             1 {dfrq:.4}\n\
             0\n\
             dfrq2 1 1 10000 0 0 2 1 1 1 64\n\
             1 {dfrq2:.4}\n\
             0\n\
             sw 1 1 100000 0 0 2 1 1 1 64\n\
             1 8000.0\n\
             0\n\
             ni 1 1 10000 0 0 2 1 1 1 64\n\
             1 {ni}\n\
             0\n\
             ni2 1 1 10000 0 0 2 1 1 1 64\n\
             1 {ni2}\n\
             0\n\
             sw1 1 1 100000 0 0 2 1 1 1 64\n\
             1 {sw1}\n\
             0\n\
             sw2 1 1 100000 0 0 2 1 1 1 64\n\
             1 {sw2}\n\
             0\n\
             phase 1 1 10 0 0 2 1 1 1 64\n\
             2 1 2\n\
             0\n\
             phase2 1 1 10 0 0 2 1 1 1 64\n\
             2 1 2\n\
             0\n\
             {extra_procpar}"
        );
        fs::write(dir.join("procpar"), procpar).unwrap();
        let total_traces = if ni2 > 1 {
            ni.max(1) * 2 * ni2 * 2
        } else {
            ni.max(1) * 2
        };
        write_synthetic_varian_fid_binary(&dir.join("fid"), total_traces, 8);
    }

    fn write_synthetic_bruker_exp_dir(
        dir: &Path,
        pulprog: &str,
        sfo1_h: f64,
        td_y: usize,
        sw_y_ppm: f64,
        nuc_y: &str,
        td_z_opt: Option<(usize, f64, &str)>,
    ) {
        fs::create_dir_all(dir).unwrap();
        let td_x = 16usize;
        let sfo_y = if nuc_y == "15N" {
            sfo1_h * 0.101329
        } else {
            sfo1_h * 0.251450
        };
        fs::write(
            dir.join("acqus"),
            format!(
                "##$PULPROG= <{pulprog}>\n\
                 ##$NUC1= <1H>\n\
                 ##$SFO1= {sfo1_h:.4}\n\
                 ##$SW= 14.0\n\
                 ##$TD= {td_x}\n\
                 ##$DTYPA= 0\n\
                 ##$AQ_mod= 3\n\
                 ##$GRPDLY= 67.98\n"
            ),
        )
        .unwrap();
        fs::write(
            dir.join("acqu2s"),
            format!(
                "##$NUC1= <{nuc_y}>\n\
                 ##$SFO1= {sfo_y:.4}\n\
                 ##$SW= {sw_y_ppm:.2}\n\
                 ##$TD= {td_y}\n\
                 ##$FnMODE= 5\n"
            ),
        )
        .unwrap();
        let n_traces = if let Some((td_z, sw_z_ppm, nuc_z)) = td_z_opt {
            let sfo_z = if nuc_z == "15N" {
                sfo1_h * 0.101329
            } else {
                sfo1_h * 0.251450
            };
            fs::write(
                dir.join("acqu3s"),
                format!(
                    "##$NUC1= <{nuc_z}>\n\
                     ##$SFO1= {sfo_z:.4}\n\
                     ##$SW= {sw_z_ppm:.2}\n\
                     ##$TD= {td_z}\n\
                     ##$FnMODE= 6\n"
                ),
            )
            .unwrap();
            td_y.max(1) * td_z.max(1)
        } else {
            td_y.max(1)
        };
        // 1024-byte block stride per trace
        fs::write(dir.join("ser"), vec![0u8; n_traces * 1024]).unwrap();
    }

    #[test]
    fn test_f5_discover_only_cli_headers_depth10_and_gft_rejection() {
        let dir = env::temp_dir().join(format!("nmr_f5_discover_only_{}", process::id()));
        let _ = fs::remove_dir_all(&dir);
        let data_root = dir.join("bmrb");
        fs::create_dir_all(&data_root).unwrap();

        let fixture_star = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/fixtures/synthetic_ubiquitin.str");

        // 1. Clean Bruker entry (`90001`) + 7-level nested Varian entry (`90002`, modeling `5844` > depth 6)
        let e_bruker = data_root.join("90001");
        fs::create_dir_all(&e_bruker).unwrap();
        fs::copy(&fixture_star, e_bruker.join("bmr90001_3.str")).unwrap();
        write_synthetic_bruker_exp_dir(
            &e_bruker.join("2"),
            "hsqcetfpf3gpsi",
            600.13,
            8,
            32.0,
            "15N",
            None,
        );
        write_synthetic_bruker_exp_dir(
            &e_bruker.join("3"),
            "hncogp3d",
            600.13,
            8,
            14.0,
            "13C",
            Some((8, 32.0, "15N")),
        );
        write_synthetic_bruker_exp_dir(
            &e_bruker.join("4"),
            "hncacbgp3d",
            600.13,
            8,
            65.0,
            "13C",
            Some((8, 32.0, "15N")),
        );

        let e_deep = data_root.join("90002");
        fs::create_dir_all(&e_deep).unwrap();
        fs::copy(&fixture_star, e_deep.join("bmr90002_3.str")).unwrap();
        // Nest `.fid` 7 levels deep under `e_deep` (exceeds old depth-6 cap, within new depth-10 cap)
        let deep_base = e_deep.join("l1/l2/l3/l4/l5/l6");
        write_synthetic_varian_exp_dir(
            &deep_base.join("gNhsqc.fid"),
            "gNhsqc",
            600.13,
            4,
            1,
            2000.0,
            0.0,
            "",
        );
        write_synthetic_varian_exp_dir(
            &deep_base.join("ghnco.fid"),
            "ghnco",
            600.13,
            4,
            4,
            2200.0,
            2000.0,
            "array 2 2 64 0 0 2 1 1 1 64\n1 \"phase,phase2\"\n0\n",
        );
        write_synthetic_varian_exp_dir(
            &deep_base.join("ghncacb.fid"),
            "ghncacb",
            600.13,
            4,
            4,
            11000.0,
            2000.0,
            "array 2 2 64 0 0 2 1 1 1 64\n1 \"phase,phase2\"\n0\n",
        );

        // 1c. Bruker entry (`90006`) with `FnMODE=1` (QF) + NUS + manual `States-TPPI` `ip3`/`ip31*2`
        //     `pulseprogram` (`QuadSource::SynthesizedManualLoop`, `q1_mult = 2`, 2 NUS points => 4 traces)
        //     and `GRPDLY=0` + `DSPFVS=99`/`DECIM=4` bypassed by `pdata/1/procs` `##$PKNL= no`.
        let e_manloop = data_root.join("90006");
        fs::create_dir_all(&e_manloop).unwrap();
        fs::copy(&fixture_star, e_manloop.join("bmr90006_3.str")).unwrap();
        let manloop_hsqc = e_manloop.join("2");
        fs::create_dir_all(manloop_hsqc.join("pdata/1")).unwrap();
        fs::write(
            manloop_hsqc.join("acqus"),
            "##$PULPROG= <hsqcetfpf3gpsi>\n\
             ##$NUC1= <1H>\n\
             ##$SFO1= 600.1300\n\
             ##$SW= 14.0\n\
             ##$TD= 16\n\
             ##$DTYPA= 0\n\
             ##$AQ_mod= 3\n\
             ##$GRPDLY= 0\n\
             ##$DSPFVS= 99\n\
             ##$DECIM= 4\n",
        )
        .unwrap();
        fs::write(
            manloop_hsqc.join("pdata/1/procs"),
            "##$FT_mod= 6\n##$PKNL= no\n",
        )
        .unwrap();
        fs::write(
            manloop_hsqc.join("acqu2s"),
            format!(
                "##$NUC1= <15N>\n\
                 ##$SFO1= {:.4}\n\
                 ##$SW= 32.00\n\
                 ##$TD= 4\n\
                 ##$NusTD= 8\n\
                 ##$FnTYPE= 2\n\
                 ##$FnMODE= 1\n",
                600.13 * 0.101329
            ),
        )
        .unwrap();
        fs::write(
            manloop_hsqc.join("pulseprogram"),
            "; custom HSQC with manual ip3/ip31*2 States-TPPI loop while FnMODE=1 (QF)\n\
             1 ze\n\
             2 d1 do:f3\n\
             3 (p1 ph1)\n\
               d4\n\
               (p21 ph3):f3\n\
               d0\n\
               (p2 ph2)\n\
               d0\n\
               (p21 ph4):f3\n\
               go=3 ph31 cpd3:f3\n\
               d1 do:f3 wr #0 if #0 zd ip3\n\
               lo to 3 times 2\n\
               d1 id0 ip3*2 ip31*2\n\
               lo to 2 times l0\n\
             exit\n\
             ph1=0\n\
             ph2=0\n\
             ph3=0 2\n\
             ph4=0\n\
             ph31=0 2\n",
        )
        .unwrap();
        fs::write(manloop_hsqc.join("nuslist"), "0\n2\n").unwrap();
        fs::write(manloop_hsqc.join("ser"), vec![0u8; 4 * 1024]).unwrap();
        write_synthetic_bruker_exp_dir(
            &e_manloop.join("3"),
            "hncogp3d",
            600.13,
            8,
            14.0,
            "13C",
            Some((8, 32.0, "15N")),
        );

        let clean_manifest = dir.join("clean_manifest.tsv");
        fs::write(
            &clean_manifest,
            "bmrb_id\tprotein\tresidues\tfield_mhz\tvendor\texperiments\tsplit\tcohort\n\
             90001\tCleanBruker\t76\t600\tBruker\tHSQC,HNCO,HNCACB\tdev\tdev\n\
             90002\tDeepVarian7\t76\t600\tVarian\tHSQC,HNCO,HNCACB\tdev\tdev\n\
             90006\tManLoopNus\t76\t600\tBruker\tHSQC,HNCO\tdev\tdev\n\
             90099\tAbsentEntry\t76\t600\tBruker\tHSQC,HNCO\tdev\tdev\n",
        )
        .unwrap();

        let json_out = dir.join("discover_clean.json");
        let args_ok = vec![
            "--manifest".to_string(),
            clean_manifest.to_string_lossy().into_owned(),
            "--split".to_string(),
            "dev".to_string(),
            "--data-root".to_string(),
            data_root.to_string_lossy().into_owned(),
            "--discover-only".to_string(),
            "--json-out".to_string(),
            json_out.to_string_lossy().into_owned(),
        ];
        cmd_real_benchmark(&args_ok)
            .expect("clean + manual-loop NUS + skipped entries should succeed in --discover-only");
        let json_txt = fs::read_to_string(&json_out).unwrap();
        assert!(json_txt.contains("\"mode\": \"discover-only\""));
        assert!(json_txt.contains("\"ok\": 3"));
        assert!(json_txt.contains("\"skipped\": 1"));
        assert!(json_txt.contains("\"l1/l2/l3/l4/l5/l6/ghncacb.fid\""));

        // Direct parity checks for strict `FnMODE`/`MC2` parsing and `GRPDLY=0` / `PKNL` / 3D `xfb` plane guard
        let bogus_fnmode = dir.join("unit_bogus_fnmode");
        write_synthetic_bruker_exp_dir(
            &bogus_fnmode,
            "hsqcetfpf3gpsi",
            600.13,
            8,
            32.0,
            "15N",
            None,
        );
        fs::write(
            bogus_fnmode.join("acqu2s"),
            "##$NUC1= <15N>\n##$SFO1= 60.81\n##$SW= 32.0\n##$TD= 8\n##$FnMODE= <bogus>\n",
        )
        .unwrap();
        let err_fn = check_bruker_header_compatibility(&bogus_fnmode, 2, false)
            .expect_err("malformed non-integer FnMODE must fail closed");
        assert!(err_fn.contains("invalid FnMODE"), "got: {err_fn}");

        fs::write(
            bogus_fnmode.join("acqu2s"),
            "##$NUC1= <15N>\n##$SFO1= 60.81\n##$SW= 32.0\n##$TD= 8\n##$FnMODE= 0\n",
        )
        .unwrap();
        fs::create_dir_all(bogus_fnmode.join("pdata/1")).unwrap();
        fs::write(
            bogus_fnmode.join("pdata/1/proc2s"),
            "##$FT_mod= 6\n##$MC2= not_an_int\n",
        )
        .unwrap();
        let err_mc2 = check_bruker_header_compatibility(&bogus_fnmode, 2, false)
            .expect_err("malformed non-integer MC2 must fail closed");
        assert!(err_mc2.contains("invalid ##$MC2"), "got: {err_mc2}");

        let grpdly_zero_3d = dir.join("unit_grpdly_zero_3d");
        write_synthetic_bruker_exp_dir(
            &grpdly_zero_3d,
            "hncogp3d",
            600.13,
            8,
            14.0,
            "13C",
            Some((8, 32.0, "15N")),
        );
        fs::write(
            grpdly_zero_3d.join("acqus"),
            "##$PULPROG= <hncogp3d>\n\
             ##$NUC1= <1H>\n\
             ##$SFO1= 600.1300\n\
             ##$SW= 14.0\n\
             ##$TD= 16\n\
             ##$DTYPA= 0\n\
             ##$AQ_mod= 3\n\
             ##$GRPDLY= 0\n\
             ##$DSPFVS= 99\n\
             ##$DECIM= 4\n",
        )
        .unwrap();
        assert!(
            check_bruker_header_compatibility(&grpdly_zero_3d, 3, false).is_err(),
            "GRPDLY=0 with unsupported DSPFVS=99/DECIM=4 must fail closed when PKNL is enabled"
        );
        fs::create_dir_all(grpdly_zero_3d.join("pdata/1")).unwrap();
        fs::write(
            grpdly_zero_3d.join("pdata/1/procs"),
            "##$FT_mod= 6\n##$PKNL= no\n",
        )
        .unwrap();
        fs::write(grpdly_zero_3d.join("pdata/1/proc2s"), "##$FT_mod= 6\n").unwrap();
        assert!(
            check_bruker_header_compatibility(&grpdly_zero_3d, 3, false).is_err(),
            "2D xfb scout plane (proc2s without proc3s on 3D) must clear procs so PKNL=no is ignored"
        );
        fs::write(grpdly_zero_3d.join("pdata/1/proc3s"), "##$FT_mod= 6\n").unwrap();
        check_bruker_header_compatibility(&grpdly_zero_3d, 3, false)
            .expect("3D cube with proc3s + procs PKNL=no must bypass GRPDLY=0 DSP lookup");

        // 2. GFT `RDphase` Varian entry (`90003`, modeling `15089` `array="phase2,phase,RDphase"`, `RDphase=[1,2]`)
        let e_gft = data_root.join("90003");
        fs::create_dir_all(&e_gft).unwrap();
        fs::copy(&fixture_star, e_gft.join("bmr90003_3.str")).unwrap();
        write_synthetic_varian_exp_dir(
            &e_gft.join("nhsqc.fid"),
            "gNhsqc",
            600.13,
            4,
            1,
            2000.0,
            0.0,
            "",
        );
        write_synthetic_varian_exp_dir(
            &e_gft.join("hncacb_gft.fid"),
            "ghncacb",
            600.13,
            4,
            4,
            11000.0,
            2000.0,
            "array 2 2 64 0 0 2 1 1 1 64\n\
             1 \"phase2,phase,RDphase\"\n\
             0\n\
             RDphase 1 1 10 0 0 2 1 1 1 64\n\
             2 1 2\n\
             0\n",
        );

        // 3. Degenerate Bruker indirect `TD=1` (`90004`) + Missing experiment (`90005`)
        let e_bad_td = data_root.join("90004");
        fs::create_dir_all(&e_bad_td).unwrap();
        fs::copy(&fixture_star, e_bad_td.join("bmr90004_3.str")).unwrap();
        write_synthetic_bruker_exp_dir(
            &e_bad_td.join("2"),
            "hsqcetfpf3gpsi",
            600.13,
            8,
            32.0,
            "15N",
            None,
        );
        write_synthetic_bruker_exp_dir(
            &e_bad_td.join("3"),
            "hncogp3d",
            600.13,
            1, // degenerate indirect TD=1!
            14.0,
            "13C",
            Some((8, 32.0, "15N")),
        );

        let e_missing = data_root.join("90005");
        fs::create_dir_all(&e_missing).unwrap();
        fs::copy(&fixture_star, e_missing.join("bmr90005_3.str")).unwrap();
        write_synthetic_bruker_exp_dir(
            &e_missing.join("2"),
            "hsqcetfpf3gpsi",
            600.13,
            8,
            32.0,
            "15N",
            None,
        );
        write_synthetic_bruker_exp_dir(
            &e_missing.join("3"),
            "hncogp3d",
            600.13,
            8,
            14.0,
            "13C",
            Some((8, 32.0, "15N")),
        );

        let bad_manifest = dir.join("bad_manifest.tsv");
        fs::write(
            &bad_manifest,
            "bmrb_id\tprotein\tresidues\tfield_mhz\tvendor\texperiments\tsplit\tcohort\n\
             90003\tGftVarian\t76\t600\tVarian\tHSQC,HNCACB\tdev\tdev\n\
             90004\tBadTdBruker\t76\t600\tBruker\tHSQC,HNCO\tdev\tdev\n\
             90005\tMissingExp\t76\t600\tBruker\tHSQC,HNCO,HNCACB\tdev\tdev\n",
        )
        .unwrap();
        let bad_json = dir.join("discover_bad.json");
        let args_bad = vec![
            "--manifest".to_string(),
            bad_manifest.to_string_lossy().into_owned(),
            "--split".to_string(),
            "dev".to_string(),
            "--data-root".to_string(),
            data_root.to_string_lossy().into_owned(),
            "--discover-only".to_string(),
            "--json-out".to_string(),
            bad_json.to_string_lossy().into_owned(),
        ];
        let err = cmd_real_benchmark(&args_bad)
            .expect_err("GFT / degenerate TD / missing experiment must fail --discover-only");
        assert!(
            err.contains("2 HEADER_INCOMPATIBLE") && err.contains("1 MISSING_EXPERIMENTS"),
            "unexpected error: {err}"
        );
        let bad_json_txt = fs::read_to_string(&bad_json).unwrap();
        assert!(bad_json_txt.contains("RDphase"));
        assert!(bad_json_txt.contains("\"status\": \"HEADER_INCOMPATIBLE\""));
        assert!(bad_json_txt.contains("\"status\": \"MISSING_EXPERIMENTS\""));

        // 4. Local header-only verification on real `dev` + `seen_p13_w2` `5844` & `15089` when present
        //    (strictly header-only: zero FFT, zero peak picking, zero solver execution!)
        let real_manifest_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/benchmark/real_manifest.tsv");
        if resolve_local_entry_paths("ubiquitin_local", Path::new("data/private/bmrb"), true)
            .is_ok()
        {
            let dev_cfg = parse_real_benchmark_args(&[
                "--manifest".to_string(),
                real_manifest_path.to_string_lossy().into_owned(),
                "--split".to_string(),
                "dev".to_string(),
                "--discover-only".to_string(),
            ])
            .unwrap();
            let dev_sum = run_real_benchmark_discover_only(&dev_cfg)
                .expect("all local dev entries must pass --discover-only");
            assert_eq!(dev_sum.n_header_incompatible, 0);
            assert_eq!(dev_sum.n_missing_experiments, 0);
            assert!(dev_sum.n_ok > 0);
        }
        if resolve_local_entry_paths("5844", Path::new("data/private/bmrb"), true).is_ok() {
            let m5844 = dir.join("m5844.tsv");
            fs::write(
                &m5844,
                "bmrb_id\tprotein\tresidues\tfield_mhz\tvendor\texperiments\tsplit\tcohort\n\
                 5844\tOP1841\t100\t750\tVarian\tHSQC,HNCO,HNCA,HNCACB,CBCACONH,HNCOCA\theldout\tseen_p13_w2\n",
            )
            .unwrap();
            let cfg_5844 = parse_real_benchmark_args(&[
                "--manifest".to_string(),
                m5844.to_string_lossy().into_owned(),
                "--split".to_string(),
                "heldout".to_string(),
                "--cohort".to_string(),
                "seen_p13_w2".to_string(),
                "--discover-only".to_string(),
            ])
            .unwrap();
            let sum_5844 = run_real_benchmark_discover_only(&cfg_5844)
                .expect("5844 6-level Varian .fid directories must pass --discover-only");
            assert_eq!(sum_5844.n_ok, 1);
            assert_eq!(sum_5844.records[0].discovered_experiments.len(), 6);
        }
        if resolve_local_entry_paths("15089", Path::new("data/private/bmrb"), true).is_ok() {
            let m15089 = dir.join("m15089.tsv");
            fs::write(
                &m15089,
                "bmrb_id\tprotein\tresidues\tfield_mhz\tvendor\texperiments\tsplit\tcohort\n\
                 15089\tStR65\t91\t600\tVarian\tHSQC,HNCO,HNCA,HNCACB,CBCACONH,HNCOCA,HNCACO\theldout\tseen_p13_w2\n",
            )
            .unwrap();
            let cfg_15089 = parse_real_benchmark_args(&[
                "--manifest".to_string(),
                m15089.to_string_lossy().into_owned(),
                "--split".to_string(),
                "heldout".to_string(),
                "--cohort".to_string(),
                "seen_p13_w2".to_string(),
                "--discover-only".to_string(),
            ])
            .unwrap();
            let err_15089 = run_real_benchmark_discover_only(&cfg_15089)
                .expect_err("15089 GFT RDphase procpar must fail --discover-only");
            assert!(err_15089.contains("1 HEADER_INCOMPATIBLE"));
        }

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_f6b_series_matched_hsqc_selection_synthetic_and_zero_dev_change() {
        let dir = env::temp_dir().join(format!("nmr_f6b_hsqc_series_{}", process::id()));
        let _ = fs::remove_dir_all(&dir);

        // Case A: Multi-field / multi-year acquisition series modeling `16238` (`seen_p13_w2`):
        // - Primary 3Ds (`vcr52.007` HNCO, HNCACB, CBCACONH) at 850 MHz (2008)
        // - Secondary 3Ds (`vcr52.011` HNCA, HNCOCA) at 600 MHz (2008-10)
        // - Two candidate 2D HSQCs:
        //   * `vcr52.006_nhsqc_750mhz_5_12_09.fid` (750 MHz, May 2009 — closer to 850 MHz, wrong batch/year)
        //   * `vcr52.010_Nhsqc_swN150p_10_23_08.fid` (600 MHz, Oct 2008 — matches `vcr52.011` 600 MHz + Oct 2008!)
        let e_multi = dir.join("multi_batch_16238_like/fids");
        write_synthetic_bruker_exp_dir(
            &e_multi.join("vcr52.007_hnco_9_10_08.850mhz"),
            "hncogp3d",
            850.13,
            8,
            14.0,
            "13C",
            Some((8, 32.0, "15N")),
        );
        write_synthetic_bruker_exp_dir(
            &e_multi.join("vcr52.007_hncacb_9_12_08.850mhz"),
            "hncacbgp3d",
            850.13,
            8,
            65.0,
            "13C",
            Some((8, 32.0, "15N")),
        );
        write_synthetic_bruker_exp_dir(
            &e_multi.join("vcr52.007_cbcaconh_9_14_08.850mhz"),
            "cbcaconhgp3d",
            850.13,
            8,
            65.0,
            "13C",
            Some((8, 32.0, "15N")),
        );
        write_synthetic_varian_exp_dir(
            &e_multi.join("vcr52.011_hnca_10_25_08.fid"),
            "ghnca",
            599.85,
            8,
            8,
            4800.0,
            2000.0,
            "",
        );
        write_synthetic_varian_exp_dir(
            &e_multi.join("vcr52.011_hncoca_10_27_08.fid"),
            "ghn_co_ca",
            599.85,
            8,
            8,
            4800.0,
            2000.0,
            "",
        );
        write_synthetic_varian_exp_dir(
            &e_multi.join("vcr52.006_nhsqc_750mhz_5_12_09.fid"),
            "gNhsqc",
            749.22,
            128,
            1,
            2400.0,
            0.0,
            "",
        );
        write_synthetic_varian_exp_dir(
            &e_multi.join("vcr52.010_Nhsqc_swN150p_10_23_08.fid"),
            "gNhsqc",
            599.85,
            128,
            1,
            2000.0,
            0.0,
            "",
        );
        let (disc_multi, _) =
            discover_entry_experiments(&dir.join("multi_batch_16238_like"), Some(850.0)).unwrap();
        let hsqc_multi = disc_multi
            .iter()
            .find(|d| d.exp_type == "HSQC")
            .expect("HSQC discovered");
        assert_eq!(
            hsqc_multi.rel_path, "fids/vcr52.010_Nhsqc_swN150p_10_23_08.fid",
            "series-matched 2008-10 600 MHz HSQC must beat orphan 2009 750 MHz HSQC"
        );

        // Case B: Same-field stem-prefix tie-break (`protA_batch2` 3Ds vs `protA_batch1` HSQC with slightly larger ni)
        let e_stem = dir.join("same_field_stem/data");
        write_synthetic_varian_exp_dir(
            &e_stem.join("protA_batch1_nhsqc.fid"),
            "gNhsqc",
            600.13,
            160,
            1,
            2000.0,
            0.0,
            "",
        );
        write_synthetic_varian_exp_dir(
            &e_stem.join("protA_batch2_nhsqc.fid"),
            "gNhsqc",
            600.13,
            128,
            1,
            2000.0,
            0.0,
            "",
        );
        write_synthetic_varian_exp_dir(
            &e_stem.join("protA_batch2_hnco.fid"),
            "ghnco",
            600.13,
            32,
            32,
            2200.0,
            2000.0,
            "",
        );
        write_synthetic_varian_exp_dir(
            &e_stem.join("protA_batch2_hncacb.fid"),
            "ghncacb",
            600.13,
            48,
            32,
            11000.0,
            2000.0,
            "",
        );
        let (disc_stem, _) =
            discover_entry_experiments(&dir.join("same_field_stem"), Some(600.0)).unwrap();
        let hsqc_stem = disc_stem
            .iter()
            .find(|d| d.exp_type == "HSQC")
            .expect("HSQC discovered");
        assert_eq!(hsqc_stem.rel_path, "data/protA_batch2_nhsqc.fid");

        // Case C: Verify ZERO selection change across all 13 real `dev` entries + verify `16238` (`seen_p13_w2`)
        let manifest_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/benchmark/real_manifest.tsv");
        let entries =
            parse_real_manifest_str(&fs::read_to_string(&manifest_path).unwrap()).unwrap();
        for e in entries.iter().filter(|e| e.split == "dev") {
            if let Ok(loc) =
                resolve_local_entry_paths(&e.bmrb_id, Path::new("data/private/bmrb"), true)
            {
                let (disc, inspections) =
                    discover_entry_experiments(&loc.entry_dir, Some(e.field_mhz)).unwrap();
                let new_hsqc = disc
                    .iter()
                    .find(|d| d.exp_type == "HSQC")
                    .expect("dev HSQC")
                    .rel_path
                    .clone();
                // Compute what legacy `candidate_selection_score` selected given the same 3D anchor parents
                let target_mhz = Some(e.field_mhz);
                let mut anchor_parents = BTreeSet::new();
                for d in disc.iter().filter(|d| d.exp_type != "HSQC") {
                    let same_type_n = inspections
                        .iter()
                        .filter(|c| {
                            c.classified_exp.as_deref() == Some(d.exp_type.as_str())
                                && c.raw_file_bytes > 0
                        })
                        .count();
                    if same_type_n == 1 || anchor_parents.is_empty() {
                        if let Some(par) = Path::new(&d.rel_path).parent() {
                            let ps = par.to_string_lossy().into_owned();
                            if !ps.is_empty() {
                                anchor_parents.insert(ps);
                            }
                        }
                    }
                }
                let mut legacy_cands: Vec<CandidateFidInspection> = inspections
                    .into_iter()
                    .filter(|c| {
                        c.classified_exp.as_deref() == Some("HSQC")
                            && c.raw_file_bytes > 0
                            && c.n_sw_ppm.is_none_or(|nsw| nsw <= 100.0)
                    })
                    .collect();
                legacy_cands.sort_by(|a, b| {
                    let sa = candidate_selection_score(a, &anchor_parents, target_mhz);
                    let sb = candidate_selection_score(b, &anchor_parents, target_mhz);
                    sb.cmp(&sa).then_with(|| a.rel_path.cmp(&b.rel_path))
                });
                let legacy_hsqc = legacy_cands[0].rel_path.clone();
                assert_eq!(
                    new_hsqc, legacy_hsqc,
                    "dev entry {} HSQC selection changed from {} to {}",
                    e.bmrb_id, legacy_hsqc, new_hsqc
                );
            }
        }
        if let Ok(loc_16238) =
            resolve_local_entry_paths("16238", Path::new("data/private/bmrb"), true)
        {
            let (disc_16238, _) =
                discover_entry_experiments(&loc_16238.entry_dir, Some(850.0)).unwrap();
            let hsqc_16238 = disc_16238
                .iter()
                .find(|d| d.exp_type == "HSQC")
                .expect("16238 HSQC");
            assert_eq!(
                hsqc_16238.rel_path,
                "nesgVcR52_bmrb16238_pdb2khd_fids/vcr52.010_Nhsqc_swN150p_10_23_08.fid"
            );
        }

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_f6a_qc_json_serialization_and_shifted_ca_batch_warn() {
        let fixture_star = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/fixtures/synthetic_ubiquitin.str");
        let ref_data = read_star(&fixture_star, None).unwrap();
        let (mut peaks, _) = simulate(
            &ref_data,
            &SimConfig {
                experiments: vec![
                    "HSQC".into(),
                    "HNCO".into(),
                    "HNCACO".into(),
                    "HNCA".into(),
                    "HNCACB".into(),
                    "CBCACONH".into(),
                ],
                p_missing_peak: 0.0,
                spurious_rate: 0.0,
                seed: 23,
                ..SimConfig::default()
            },
        );
        let priors = nmr_core::priors::Priors::bmrb();
        let score_cfg = ScoreConfig::default();
        let map_ss_to_truth = |systems: &[nmr_core::spinsystems::SpinSystem]| -> Vec<i32> {
            systems
                .iter()
                .map(|ss| {
                    let mut best = 0i32;
                    let mut best_d2 = f64::INFINITY;
                    for seq_id in 1..=ref_data.sequence.len() {
                        if let (Some(&th), Some(&tn)) = (
                            ref_data.shifts.get(&(seq_id, "H".to_string())),
                            ref_data.shifts.get(&(seq_id, "N".to_string())),
                        ) {
                            let d2 = ((ss.h - th) / 0.03).powi(2) + ((ss.n - tn) / 0.30).powi(2);
                            if d2 < best_d2 && d2 < 4.0 {
                                best_d2 = d2;
                                best = seq_id as i32;
                            }
                        }
                    }
                    best
                })
                .collect()
        };
        let ss_clean = build_spin_systems(
            &peaks,
            &Tolerances::default(),
            ref_data.sequence.len(),
            false,
        )
        .unwrap();
        let res_clean = nmr_core::solver::AssignResult {
            position: map_ss_to_truth(&ss_clean),
            confidence: vec![0.92; ss_clean.len()],
            score: 120.0,
            run_scores: vec![120.0, 118.0],
        };
        let qc_clean = nmr_core::qc::evaluate_dataset_qc(
            Some(&peaks),
            &ss_clean,
            &ref_data.sequence,
            &priors,
            &score_cfg,
            &res_clean,
            None,
        );
        assert_eq!(qc_clean.verdict, nmr_core::qc::DatasetQcVerdict::Pass);
        assert!(!qc_clean.inter_exp_c13_offset_warn);
        assert!(qc_clean.max_inter_exp_ca_drift_ppm < 0.15);
        let json_clean = render_qc_report_json(&qc_clean, 0);
        assert!(json_clean.contains("\"verdict\": \"PASS\""));
        assert!(json_clean.contains("\"inter_exp_c13_offset_warn\": false"));

        // Shift `HNCA` 13C referencing by `+1.15 ppm` relative to `HNCACB` / `CBCACONH`
        for pk in &mut peaks {
            if pk.exp == "HNCA" {
                if let Some(ref mut c) = pk.c {
                    *c += 1.15;
                }
            }
        }
        let ss_shift = build_spin_systems(
            &peaks,
            &Tolerances::default(),
            ref_data.sequence.len(),
            false,
        )
        .unwrap();
        let res_shift = nmr_core::solver::AssignResult {
            position: map_ss_to_truth(&ss_shift),
            confidence: vec![0.92; ss_shift.len()],
            score: 120.0,
            run_scores: vec![120.0, 118.0],
        };
        let qc_shift = nmr_core::qc::evaluate_dataset_qc(
            Some(&peaks),
            &ss_shift,
            &ref_data.sequence,
            &priors,
            &score_cfg,
            &res_shift,
            None,
        );
        assert!(qc_shift.inter_exp_c13_offset_warn);
        assert!(
            (qc_shift.max_inter_exp_ca_drift_ppm - 1.15).abs() < 0.12,
            "expected ~1.15 ppm drift, got {}",
            qc_shift.max_inter_exp_ca_drift_ppm
        );
        assert_eq!(qc_shift.verdict, nmr_core::qc::DatasetQcVerdict::Warn);
        let json_shift = render_qc_report_json(&qc_shift, 0);
        assert!(json_shift.contains("\"verdict\": \"WARN\""));
        assert!(json_shift.contains("\"inter_exp_c13_offset_warn\": true"));
    }

    #[test]
    #[ignore = "empirical raw-FID FFT verification across local dev (13) + 16238 seen_p13_w2"]
    fn verify_f6a_empirical_c13_ca_consistency_on_local_dev_and_16238() {
        let manifest_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/benchmark/real_manifest.tsv");
        let entries =
            parse_real_manifest_str(&fs::read_to_string(&manifest_path).unwrap()).unwrap();
        let picker_cfg = PickerConfig::default();

        let eval_entry_ca_drift =
            |entry: &RealManifestEntry| -> nmr_core::qc::InterExpC13CaConsistency {
                let loc =
                    resolve_local_entry_paths(&entry.bmrb_id, Path::new("data/private/bmrb"), true)
                        .expect("local entry dir present");
                let (discovered, _) =
                    discover_entry_experiments(&loc.entry_dir, Some(entry.field_mhz)).unwrap();
                let hsqc_exp = discovered.iter().find(|d| d.exp_type == "HSQC").unwrap();
                let (hsqc_spec, _) =
                    process_fid_dir_2d_opts(&hsqc_exp.dir, false, false, false).unwrap();
                let roots = pick_hsqc_2d(&hsqc_spec, &picker_cfg);
                let mut all_peaks: Vec<Peak> = roots.clone();
                let mut strip_bank = StripBank3D::new();
                for texp in discovered
                    .iter()
                    .filter(|d| matches!(d.exp_type.as_str(), "HNCA" | "HNCACB" | "CBCACONH"))
                {
                    let (mut spec3d, _) =
                        process_fid_dir_3d_opts(&texp.dir, &texp.exp_type, false, false, false)
                            .unwrap();
                    let _ =
                        register_3d_hn_to_hsqc_adaptive_with_audit(&mut spec3d, &hsqc_spec, true);
                    let pks = strip_bank.pick_and_record_3d(&spec3d, &roots, &picker_cfg);
                    all_peaks.extend(pks);
                }
                nmr_core::qc::check_inter_experiment_c13_ca_consistency(&all_peaks)
            };

        for e in entries.iter().filter(|e| e.split == "dev") {
            let diag = eval_entry_ca_drift(e);
            println!(
                "[dev F6a] {:<16} warn={} max_drift={:.3} ppm diag={:?}",
                e.bmrb_id, diag.warn_triggered, diag.max_abs_drift_ppm, diag.diagnostic
            );
            assert!(
                !diag.warn_triggered,
                "dev entry {} triggered false inter-exp 13C CA warning: {:?}",
                e.bmrb_id, diag
            );
        }

        let e_16238 = entries
            .iter()
            .find(|e| e.bmrb_id == "16238" && e.cohort == "seen_p13_w2")
            .expect("16238 seen_p13_w2 in manifest");
        let diag_16238 = eval_entry_ca_drift(e_16238);
        println!(
            "[seen_p13_w2 F6a] 16238 warn={} max_drift={:.3} ppm diag={:?}",
            diag_16238.warn_triggered, diag_16238.max_abs_drift_ppm, diag_16238.diagnostic
        );
        assert!(
            diag_16238.warn_triggered,
            "16238 multi-batch 13C referencing mismatch must trigger F6a WARN, got {:?}",
            diag_16238
        );
    }
}
