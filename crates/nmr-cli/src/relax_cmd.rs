//! CLI subcommands `ferric relax` and `ferric modelfree` for Slice `13.5` (`T13.1`–`T13.3`):
//! pseudo-3D relaxation series integration, batched CPU/WebGPU $R_1$, $R_2$, $R_{1\rho}$ &
//! steady-state $\{^1\text{H}\}\text{–}^{15}\text{N}$ hetNOE exponential fitting, and
//! Lipari–Szabo / Mandel–Akke–Palmer Model-Free (`M1`–`M5`) + anisotropic rotational diffusion.
//!
//! # Literature & Clean-Room Provenance
//! - Kay, L. E., Torchia, D. A. & Bax, A. (1989). *Biochemistry* 28(23), 8972–8979.
//! - Farrow, N. A. et al. (1994). *Biochemistry* 33(19), 5984–6003.
//! - Farrow, N. A., Zhang, O., Szabo, A., Torchia, D. A. & Kay, L. E. (1995).
//!   *J. Biomol. NMR* 6(2), 153–162.
//! - Lipari, G. & Szabo, A. (1982a, 1982b). *J. Am. Chem. Soc.* 104(17), 4546–4559 & 4559–4570.
//! - Clore, G. M., Szabo, A., Bax, A., Kay, L. E., Driscoll, P. C. & Gronenborn, A. M. (1990).
//!   *J. Am. Chem. Soc.* 112(12), 4989–4991.
//! - Mandel, A. M., Akke, M. & Palmer, A. G. III (1995). *J. Mol. Biol.* 246(1), 144–163.
//! - Woessner, D. E. (1962). *J. Chem. Phys.* 37(3), 647–654.
//! - d'Auvergne, E. J. & Gooley, P. R. (2003). *J. Biomol. NMR* 25(1), 25–39.
//!
//! Implemented from the published equations and `docs/specs/16-dynamics-and-csp.md` §§1–2 without
//! consulting external relaxation/dynamics software source code.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use nmr_core::modelfree::{
    extract_nh_unit_vectors_from_pdb, fit_modelfree_dataset, group_multifield_rates,
    write_modelfree_tsv_str, DiffusionTensor, DiffusionTensorKind, ModelFreeConfig, ModelFreeId,
    ModelSelectionCriterion,
};
use nmr_core::nef::read_nef_file;
use nmr_core::nmrstar::{read_star, Reference};
use nmr_core::peaks::read_peaks;
use nmr_core::picker::{pick_hsqc_2d, PickerConfig};
use nmr_core::relax_fit::{
    compute_hetnoe_batch, decay_inputs_from_series, estimate_trimmed_tau_m, fit_decay_batch_cpu,
    map_reduced_spectral_density_batch, DecayCurveFit, ExpFitConfig, HetNoeInput, HetNoeResult,
    OffsetModelGate, RelaxPhysicsConfig, ResidueRelaxRates, TrimmedTauMConfig,
};
use nmr_core::relax_io::{
    integrate_relax_planes, parse_bruker_delay_token_sec, parse_bruker_vdlist,
    parse_relax_series_nef_str, parse_relax_series_tsv_str, process_pseudo3d_fids,
    read_bruker_pseudo3d_with_hetnoe_order, read_varian_pseudo3d, write_relax_series_nef_str,
    write_relax_series_tsv_str, HetNoeOrder, RelaxExperimentKind, RelaxIntegrationConfig,
    RelaxSchedule, RelaxSeriesDataset, TargetPeakSeed,
};
use nmr_core::spectrum::{read_bruker_pdata_2d, read_nmrpipe_2d, read_ucsf_2d, Spectrum2D};
use nmr_gpu::relax_gpu::fit_decay_batch_gpu;
use nmr_gpu::GpuEngine;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RelaxEngineChoice {
    Cpu,
    Gpu,
    Auto,
}

impl RelaxEngineChoice {
    fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "cpu" => Ok(Self::Cpu),
            "gpu" | "webgpu" => Ok(Self::Gpu),
            "auto" => Ok(Self::Auto),
            other => Err(format!(
                "invalid --engine {other:?} (expected cpu, gpu, or auto)"
            )),
        }
    }
}

pub fn relax_usage() -> &'static str {
    "ferric relax — Pseudo-3D relaxation series integration & CPU/WebGPU R1/R2/hetNOE curve fitting\n\
     Usage:\n\
       ferric relax (--series <series.tsv|series.nef|dir> | --planes <p1,p2,...>) --out <rates.tsv> [options]\n\n\
     Inputs & Integration:\n\
       --series | --in <path>   Relaxation series TSV/NEF, Bruker/Varian pseudo-3D folder, or 2D plane folder\n\
       --planes <p1,p2,...>     Comma-separated 2D spectrum paths (.ft2, .ucsf, or Bruker pdata dir)\n\
       --vdlist <file>          Variable-delay schedule file (seconds / ms / us suffixes; overrides raw ser/fid schedule)\n\
       --delays <csv|file>      Comma-separated delay list or schedule file (overrides raw ser/fid schedule)\n\
       --peaks <peaks.tsv>      Target 2D HSQC crosspeak seeds for locked-footprint integration\n\
       --star | --shifts <p>    Target NMR-STAR (.str) or NEF (.nef) shift list for footprint seeds\n\
       --series-out <path>      Optional path to save integrated series (.tsv or .nef)\n\n\
     Curve Fitting:\n\
       --kind <r1|r2|r1rho|hetnoe>  Experiment type (overrides header metadata when provided)\n\
       --field-mhz <f64>        1H spectrometer frequency in MHz (overrides header when provided)\n\
       --engine <cpu|gpu|auto>  Compute backend for exponential fitting + MC bootstrap [default: cpu]\n\
       --mc-trials <N>          Parametric Monte Carlo replicates per residue [default: 500]\n\
       --offset-gate <mode>     3-parameter offset gate: none|aic|aicc|bic|always [default: none]\n\
       --unsat-plane <idx>      Reference unsaturated plane index for hetNOE (auto-detected by default)\n\
       --sat-plane <idx>        Steady-state saturated plane index for hetNOE (auto-detected by default)\n\
       --hetnoe-order <sat-first|ref-first>  Plane order of a schedule-less 2-plane Bruker hetNOE ser; REQUIRED when the pulseprogram does not prove it (otherwise `ferric relax` fails closed)\n\
       --seed <u64>             Deterministic PRNG seed [default: 20261302]\n\
       --out | -o <rates.tsv>   Output fitted relaxation rate / hetNOE TSV table"
}

pub fn modelfree_usage() -> &'static str {
    "ferric modelfree — Lipari-Szabo / Mandel-Akke-Palmer M1-M5 & anisotropic rotational diffusion\n\
     Usage:\n\
       ferric modelfree (--r1 <r1.tsv> --r2 <r2.tsv> [--hetnoe <noe.tsv>] ... | --rates <rates.tsv|bmr.str> ...) --out <modelfree.tsv> [options]\n\n\
     Inputs (repeatable for multi-field 600 + 800 + 900 MHz datasets):\n\
       --r1 <r1.tsv>            R1 rate table from `ferric relax` (repeatable per static B0 field)\n\
       --r2 <r2.tsv>            R2 rate table from `ferric relax` (repeatable per static B0 field)\n\
       --hetnoe | --noe <tsv>   Steady-state {1H}-15N hetNOE table from `ferric relax` (repeatable)\n\
       --rates | --star <path>  Single-kind/combined rate TSV or NMR-STAR (.str) relaxation entry (repeatable)\n\
       --pdb <model.pdb>        Optional 3D PDB structure for N-H unit vectors & axial/anisotropic tensor\n\n\
     Model Selection & Physics:\n\
       --diffusion <mode>       auto|isotropic|axial|anisotropic [default: auto with --pdb, else isotropic]\n\
       --criterion <mode>       aic|aicc|bic|ftest per-residue M1-M5 model selection [default: aic]\n\
       --ref-mhz <f64>          Reference 1H field (MHz) for Rex quadratic B0^2 reporting [default: min field]\n\
       --mc-trials <N>          Parametric MC replicates for S^2 / tau_e / Rex uncertainties [default: 0]\n\
       --r-nh <angstrom>        Effective N-H bond length in Angstroms [default: 1.02]\n\
       --csa-ppm <ppm>          15N chemical shift anisotropy in ppm (e.g. -172.0 or legacy -160.0) [default: -172.0]\n\
       --default-rel-sigma <f>  Explicit opt-in fractional uncertainty (0 < f < 1) for legacy null Val_err (.) [default: off]\n\
       --seed <u64>             Deterministic PRNG seed [default: 20261303]\n\
       --rsd-out <j_map.tsv>    Optional Farrow (1995) reduced spectral density J(0), J(wN), J(0.87wH) TSV\n\
       --out | -o <mf.tsv>      Output Lipari-Szabo Model-Free M1-M5 TSV table"
}

fn parse_offset_gate(raw: &str) -> Result<OffsetModelGate, String> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "none" | "2p" | "two" => Ok(OffsetModelGate::Disabled),
        "aic" => Ok(OffsetModelGate::Aic),
        "aicc" => Ok(OffsetModelGate::Aicc),
        "bic" => Ok(OffsetModelGate::Bic),
        "always" | "3p" | "three" => Ok(OffsetModelGate::AlwaysThreeParam),
        other => Err(format!(
            "invalid --offset-gate {other:?} (expected none, aic, aicc, bic, or always)"
        )),
    }
}

fn parse_criterion(raw: &str) -> Result<ModelSelectionCriterion, String> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "aic" => Ok(ModelSelectionCriterion::Aic),
        "aicc" => Ok(ModelSelectionCriterion::Aicc),
        "bic" => Ok(ModelSelectionCriterion::Bic),
        "ftest" | "f-test" | "palmer" => Ok(ModelSelectionCriterion::PalmerFTest),
        other => Err(format!(
            "invalid --criterion {other:?} (expected aic, aicc, bic, or ftest)"
        )),
    }
}

fn parse_diffusion_kind(raw: &str) -> Result<DiffusionTensorKind, String> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "auto" => Ok(DiffusionTensorKind::Auto),
        "isotropic" | "iso" => Ok(DiffusionTensorKind::Isotropic),
        "axial" | "axial_symmetric" | "axially_symmetric" => {
            Ok(DiffusionTensorKind::AxialSymmetric)
        }
        "anisotropic" | "fully_anisotropic" | "aniso" => Ok(DiffusionTensorKind::FullyAnisotropic),
        other => Err(format!(
            "invalid --diffusion {other:?} (expected auto, isotropic, axial, or anisotropic)"
        )),
    }
}

fn load_spectrum_2d_any(path: &Path) -> Result<Spectrum2D, String> {
    if path.is_dir() {
        let pdata1 = path.join("pdata").join("1");
        if pdata1.join("2rr").is_file() {
            return read_bruker_pdata_2d(&pdata1);
        }
        if path.join("2rr").is_file() {
            return read_bruker_pdata_2d(path);
        }
        return Err(format!(
            "2D spectrum directory '{}' does not contain Bruker 2rr",
            path.display()
        ));
    }
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if ext == "ucsf" {
        read_ucsf_2d(path)
    } else {
        read_nmrpipe_2d(path)
    }
}

fn discover_2d_planes_in_dir(dir: &Path) -> Result<Vec<Spectrum2D>, String> {
    let mut entries: Vec<PathBuf> = fs::read_dir(dir)
        .map_err(|e| format!("failed to list series directory '{}': {e}", dir.display()))?
        .filter_map(|e| e.ok().map(|de| de.path()))
        .collect();

    // 1. Check numeric Bruker experiment sub-directories (`1/pdata/1/2rr`, `2/pdata/1/2rr`, ...).
    let mut numeric_dirs: Vec<(u64, PathBuf)> = entries
        .iter()
        .filter_map(|p| {
            if !p.is_dir() {
                return None;
            }
            let num: u64 = p.file_name()?.to_str()?.parse().ok()?;
            let pdata = p.join("pdata").join("1");
            if pdata.join("2rr").is_file() {
                Some((num, pdata))
            } else if p.join("2rr").is_file() {
                Some((num, p.clone()))
            } else {
                None
            }
        })
        .collect();
    if !numeric_dirs.is_empty() {
        numeric_dirs.sort_by_key(|(n, _)| *n);
        return numeric_dirs
            .into_iter()
            .map(|(_, p)| read_bruker_pdata_2d(&p))
            .collect();
    }

    // 2. Otherwise collect `.ft2`, `.dat`, or `.ucsf` 2D plane files in lexicographical order.
    entries.retain(|p| {
        p.is_file()
            && matches!(
                p.extension()
                    .and_then(|s| s.to_str())
                    .map(|s| s.to_ascii_lowercase())
                    .as_deref(),
                Some("ft2" | "ucsf" | "pipe")
            )
    });
    entries.sort();
    if entries.is_empty() {
        return Err(format!(
            "no raw Bruker `ser`, Varian `fid`, Bruker `<n>/pdata/1/2rr`, or 2D `.ft2`/`.ucsf` planes found in '{}'",
            dir.display()
        ));
    }
    entries.iter().map(|p| load_spectrum_2d_any(p)).collect()
}

fn build_seeds_from_reference(reference: &Reference) -> Vec<TargetPeakSeed> {
    let mut out = Vec::new();
    for (idx, res_name) in reference.sequence.iter().enumerate() {
        if res_name == "PRO" {
            continue;
        }
        let h = reference
            .shifts
            .get(&(idx, "H".to_string()))
            .or_else(|| reference.shifts.get(&(idx, "HN".to_string())));
        let n = reference.shifts.get(&(idx, "N".to_string()));
        if let (Some(&dh), Some(&dn)) = (h, n) {
            let seq_id = reference
                .seq_ids
                .get(idx)
                .copied()
                .unwrap_or((idx + 1) as i32);
            out.push(TargetPeakSeed {
                seq_id,
                res_name: res_name.clone(),
                atom_h: "H".to_string(),
                atom_x: "N".to_string(),
                delta_h_ppm: dh,
                delta_x_ppm: dn,
            });
        }
    }
    out
}

fn resolve_target_seeds(
    peaks_path: Option<&Path>,
    shifts_path: Option<&Path>,
    ref_plane: &Spectrum2D,
) -> Result<Vec<TargetPeakSeed>, String> {
    if let Some(sp) = shifts_path {
        let ext = sp
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let reference = if ext == "nef" {
            read_nef_file(sp)?.to_reference()?
        } else {
            read_star(sp, None)?
        };
        let seeds = build_seeds_from_reference(&reference);
        if seeds.is_empty() {
            return Err(format!(
                "shift list '{}' contains no (H, N) backbone chemical shift pairs",
                sp.display()
            ));
        }
        return Ok(seeds);
    }

    if let Some(pp) = peaks_path {
        let peaks = read_peaks(pp)?;
        let hsqc_only: Vec<_> = peaks
            .iter()
            .filter(|p| p.exp.eq_ignore_ascii_case("HSQC"))
            .collect();
        let chosen = if hsqc_only.is_empty() {
            peaks.iter().collect::<Vec<_>>()
        } else {
            hsqc_only
        };
        if chosen.is_empty() {
            return Err(format!("peak list '{}' is empty", pp.display()));
        }
        let seeds = chosen
            .iter()
            .enumerate()
            .map(|(idx, p)| TargetPeakSeed {
                seq_id: (idx + 1) as i32,
                res_name: "UNK".to_string(),
                atom_h: "H".to_string(),
                atom_x: "N".to_string(),
                delta_h_ppm: p.h,
                delta_x_ppm: p.n,
            })
            .collect();
        return Ok(seeds);
    }

    let picked = pick_hsqc_2d(ref_plane, &PickerConfig::default());
    if picked.is_empty() {
        return Err(
            "auto-picking reference plane 0 yielded 0 crosspeaks; supply --peaks or --star"
                .to_string(),
        );
    }
    Ok(picked
        .iter()
        .enumerate()
        .map(|(idx, p)| TargetPeakSeed {
            seq_id: (idx + 1) as i32,
            res_name: "UNK".to_string(),
            atom_h: "H".to_string(),
            atom_x: "N".to_string(),
            delta_h_ppm: p.h,
            delta_x_ppm: p.n,
        })
        .collect())
}

fn parse_explicit_schedule_override(
    vdlist_path: Option<&Path>,
    delays_csv: Option<&str>,
) -> Result<Option<RelaxSchedule>, String> {
    if let Some(spec) = delays_csv {
        let p = Path::new(spec.trim());
        let delays = if p.is_file() {
            let txt = fs::read_to_string(p)
                .map_err(|e| format!("failed to read --delays file '{}': {e}", p.display()))?;
            parse_bruker_vdlist(&txt)?
        } else {
            let mut v = Vec::new();
            for tok in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                v.push(parse_bruker_delay_token_sec(tok)?);
            }
            if v.len() < 2 {
                return Err(format!(
                    "--delays requires at least 2 delay values (got {})",
                    v.len()
                ));
            }
            v
        };
        return Ok(Some(RelaxSchedule::VariableDelaysSec {
            delays_sec: delays,
        }));
    }
    if let Some(vp) = vdlist_path {
        let txt = fs::read_to_string(vp)
            .map_err(|e| format!("failed to read --vdlist '{}': {e}", vp.display()))?;
        let delays = parse_bruker_vdlist(&txt)?;
        return Ok(Some(RelaxSchedule::VariableDelaysSec {
            delays_sec: delays,
        }));
    }
    Ok(None)
}

fn parse_schedule_for_planes(
    vdlist_path: Option<&Path>,
    delays_csv: Option<&str>,
    fallback_dir: Option<&Path>,
    expected_planes: usize,
) -> Result<RelaxSchedule, String> {
    if let Some(sched) = parse_explicit_schedule_override(vdlist_path, delays_csv)? {
        if sched.len() != expected_planes {
            return Err(format!(
                "explicit schedule count ({}) does not match plane count ({expected_planes})",
                sched.len()
            ));
        }
        return Ok(sched);
    }
    if let Some(dir) = fallback_dir {
        let cand = dir.join("vdlist");
        if cand.is_file() {
            let txt = fs::read_to_string(&cand)
                .map_err(|e| format!("failed to read '{}': {e}", cand.display()))?;
            let delays = parse_bruker_vdlist(&txt)?;
            if delays.len() != expected_planes {
                return Err(format!(
                    "vdlist '{}' count ({}) does not match plane count ({expected_planes})",
                    cand.display(),
                    delays.len()
                ));
            }
            return Ok(RelaxSchedule::VariableDelaysSec { delays_sec: delays });
        }
    }
    Err(
        "2D plane stack requires --vdlist <file>, --delays <t1,t2,...>, or <dir>/vdlist"
            .to_string(),
    )
}

#[allow(clippy::too_many_arguments)]
fn load_or_integrate_series(
    series_path: Option<&Path>,
    planes_csv: Option<&str>,
    vdlist_path: Option<&Path>,
    delays_csv: Option<&str>,
    peaks_path: Option<&Path>,
    shifts_path: Option<&Path>,
    kind_override: Option<RelaxExperimentKind>,
    field_override: Option<f64>,
    hetnoe_order: HetNoeOrder,
) -> Result<RelaxSeriesDataset, String> {
    if let Some(p_csv) = planes_csv {
        let plane_paths: Vec<PathBuf> = p_csv
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(PathBuf::from)
            .collect();
        if plane_paths.len() < 2 {
            return Err("--planes requires at least 2 2D spectrum paths".to_string());
        }
        let planes: Vec<Spectrum2D> = plane_paths
            .iter()
            .map(|p| load_spectrum_2d_any(p))
            .collect::<Result<_, _>>()?;
        let schedule = parse_schedule_for_planes(vdlist_path, delays_csv, None, planes.len())?;
        let seeds = resolve_target_seeds(peaks_path, shifts_path, &planes[0])?;
        let exp_kind = kind_override.unwrap_or(RelaxExperimentKind::R1);
        let mut ds = integrate_relax_planes(
            "plane_stack",
            exp_kind,
            schedule,
            &planes,
            &seeds,
            &RelaxIntegrationConfig::default(),
        )?;
        if let Some(fm) = field_override {
            ds.field_mhz = fm;
        }
        return Ok(ds);
    }

    let sp = series_path.ok_or_else(|| {
        format!(
            "missing --series <path> or --planes <csv>\n\n{}",
            relax_usage()
        )
    })?;
    if sp.is_file() {
        let raw = fs::read_to_string(sp)
            .map_err(|e| format!("failed to read series file '{}': {e}", sp.display()))?;
        let ext = sp
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let mut ds = if ext == "nef" || raw.contains("save_nmrassign_relaxation_series") {
            parse_relax_series_nef_str(&raw)?
        } else {
            parse_relax_series_tsv_str(&raw)?
        };
        if let Some(sched) = parse_explicit_schedule_override(vdlist_path, delays_csv)? {
            if sched.len() != ds.schedule.len() {
                return Err(format!(
                    "explicit schedule count ({}) does not match series plane count ({})",
                    sched.len(),
                    ds.schedule.len()
                ));
            }
            ds.schedule = sched;
        }
        if let Some(k) = kind_override {
            ds.experiment_kind = k;
        }
        if let Some(fm) = field_override {
            ds.field_mhz = fm;
        }
        return Ok(ds);
    }

    if !sp.is_dir() {
        return Err(format!("series path '{}' does not exist", sp.display()));
    }

    let ds_name = sp
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("relax_series");

    let explicit_sched = parse_explicit_schedule_override(vdlist_path, delays_csv)?;

    let mut ds = if sp.join("ser").is_file() && sp.join("acqus").is_file() {
        let raw_p3d = if let Some(ref sched_ovr) = explicit_sched {
            match read_bruker_pseudo3d_with_hetnoe_order(sp, None, None, None, hetnoe_order) {
                Ok(mut p3d) => {
                    if sched_ovr.len() != p3d.planes.len() {
                        return Err(format!(
                            "explicit schedule count ({}) does not match Bruker ser plane count ({})",
                            sched_ovr.len(),
                            p3d.planes.len()
                        ));
                    }
                    // The explicit schedule supersedes any inferred hetNOE delay/order.
                    p3d.warnings.clear();
                    p3d.schedule = sched_ovr.clone();
                    p3d
                }
                Err(_) => read_bruker_pseudo3d_with_hetnoe_order(
                    sp,
                    None,
                    None,
                    Some(sched_ovr.clone()),
                    hetnoe_order,
                )?,
            }
        } else {
            read_bruker_pseudo3d_with_hetnoe_order(sp, None, None, None, hetnoe_order)?
        };
        for w in &raw_p3d.warnings {
            eprintln!("warning: {w}");
        }
        let (planes, _recipe) = process_pseudo3d_fids(&raw_p3d.planes, None)?;
        let seeds = resolve_target_seeds(peaks_path, shifts_path, &planes[0])?;
        let exp_kind = kind_override.unwrap_or(raw_p3d.experiment_kind);
        integrate_relax_planes(
            ds_name,
            exp_kind,
            raw_p3d.schedule,
            &planes,
            &seeds,
            &RelaxIntegrationConfig::default(),
        )?
    } else if sp.join("fid").is_file() && sp.join("procpar").is_file() {
        let mut raw_p3d = read_varian_pseudo3d(sp, None, None)?;
        if let Some(sched_ovr) = explicit_sched {
            if sched_ovr.len() != raw_p3d.planes.len() {
                return Err(format!(
                    "explicit schedule count ({}) does not match Varian fid plane count ({})",
                    sched_ovr.len(),
                    raw_p3d.planes.len()
                ));
            }
            raw_p3d.schedule = sched_ovr;
        }
        let (planes, _recipe) = process_pseudo3d_fids(&raw_p3d.planes, None)?;
        let seeds = resolve_target_seeds(peaks_path, shifts_path, &planes[0])?;
        let exp_kind = kind_override.unwrap_or(raw_p3d.experiment_kind);
        integrate_relax_planes(
            ds_name,
            exp_kind,
            raw_p3d.schedule,
            &planes,
            &seeds,
            &RelaxIntegrationConfig::default(),
        )?
    } else {
        let planes = discover_2d_planes_in_dir(sp)?;
        let schedule = parse_schedule_for_planes(vdlist_path, delays_csv, Some(sp), planes.len())?;
        let seeds = resolve_target_seeds(peaks_path, shifts_path, &planes[0])?;
        let exp_kind = kind_override.unwrap_or(RelaxExperimentKind::R1);
        integrate_relax_planes(
            ds_name,
            exp_kind,
            schedule,
            &planes,
            &seeds,
            &RelaxIntegrationConfig::default(),
        )?
    };

    if let Some(k) = kind_override {
        ds.experiment_kind = k;
    }
    if let Some(fm) = field_override {
        ds.field_mhz = fm;
    }
    Ok(ds)
}

fn format_decay_rates_tsv(
    kind: RelaxExperimentKind,
    field_mhz: f64,
    engine_label: &str,
    mc_trials: usize,
    fits: &[DecayCurveFit],
) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "# nmrassign_relax kind={} field_mhz={field_mhz:.4} engine={engine_label} mc_trials={mc_trials}\n",
        kind.as_str()
    ));
    out.push_str(
        "seq_id\tres_name\tfield_mhz\tkind\tvalue\tsigma\tsigma_fisher\tci68_lo\tci68_hi\tci95_lo\tci95_hi\ti0\tsigma_i0\toffset_c\tused_three_param\tchi2\tdof\taic\tbic\n",
    );
    for f in fits {
        out.push_str(&format!(
            "{}\t{}\t{:.4}\t{}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.6}\t{:.6}\t{:.6}\t{}\t{:.6}\t{}\t{:.6}\t{:.6}\n",
            f.seq_id,
            f.res_name,
            field_mhz,
            kind.as_str(),
            f.rate,
            f.sigma_rate_mc,
            f.sigma_rate_fisher,
            f.rate_ci68[0],
            f.rate_ci68[1],
            f.rate_ci95[0],
            f.rate_ci95[1],
            f.i0,
            f.sigma_i0_mc,
            f.offset_c,
            f.used_three_param,
            f.chi2,
            f.dof,
            f.aic,
            f.bic,
        ));
    }
    out
}

fn format_hetnoe_tsv(field_mhz: f64, mc_trials: usize, results: &[HetNoeResult]) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "# nmrassign_relax kind=hetNOE field_mhz={field_mhz:.4} engine=cpu mc_trials={mc_trials}\n"
    ));
    out.push_str(
        "seq_id\tres_name\tfield_mhz\tkind\tvalue\tsigma\tsigma_mc\tci68_lo\tci68_hi\tci95_lo\tci95_hi\n",
    );
    for r in results {
        let sigma = if mc_trials > 0 && r.sigma_noe_mc > 0.0 {
            r.sigma_noe_mc
        } else {
            r.sigma_noe
        };
        out.push_str(&format!(
            "{}\t{}\t{:.4}\thetNOE\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\n",
            r.seq_id,
            r.res_name,
            field_mhz,
            r.noe,
            sigma,
            r.sigma_noe_mc,
            r.noe_ci68[0],
            r.noe_ci68[1],
            r.noe_ci95[0],
            r.noe_ci95[1],
        ));
    }
    out
}

fn infer_hetnoe_plane_indices(
    ds: &RelaxSeriesDataset,
    unsat_override: Option<usize>,
    sat_override: Option<usize>,
) -> Result<(usize, usize), String> {
    let n_planes = ds.schedule.len();
    if n_planes < 2 {
        return Err(format!(
            "hetNOE series '{}' has {n_planes} plane(s); at least 2 planes (unsat, sat) are required",
            ds.name
        ));
    }
    if let (Some(u), Some(s)) = (unsat_override, sat_override) {
        if u >= n_planes || s >= n_planes || u == s {
            return Err(format!(
                "invalid --unsat-plane ({u}) / --sat-plane ({s}) for {n_planes}-plane series"
            ));
        }
        return Ok((u, s));
    }
    let vals = ds.schedule.primary_values();
    let default_pair = if vals[0].abs() <= 1.0e-9 && vals[1] > 0.0 {
        (0usize, 1usize)
    } else if vals[1].abs() <= 1.0e-9 && vals[0] > 0.0 {
        (1usize, 0usize)
    } else {
        (0usize, 1usize)
    };
    let u = unsat_override.unwrap_or(default_pair.0);
    let s = sat_override.unwrap_or(default_pair.1);
    if u >= n_planes || s >= n_planes || u == s {
        return Err(format!(
            "invalid hetNOE plane pair (unsat={u}, sat={s}) for {n_planes}-plane series"
        ));
    }
    Ok((u, s))
}

fn median_f64(mut vals: Vec<f64>) -> f64 {
    vals.retain(|v| v.is_finite());
    if vals.is_empty() {
        return 0.0;
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = vals.len() / 2;
    if vals.len() % 2 == 1 {
        vals[mid]
    } else {
        0.5 * (vals[mid - 1] + vals[mid])
    }
}

/// Entry point for `ferric relax`.
pub fn run_relax_cli(args: &[String]) -> Result<(), String> {
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        println!("{}", relax_usage());
        return Ok(());
    }

    let mut series_path: Option<PathBuf> = None;
    let mut planes_csv: Option<String> = None;
    let mut vdlist_path: Option<PathBuf> = None;
    let mut delays_csv: Option<String> = None;
    let mut peaks_path: Option<PathBuf> = None;
    let mut shifts_path: Option<PathBuf> = None;
    let mut series_out_path: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut kind_override: Option<RelaxExperimentKind> = None;
    let mut field_override: Option<f64> = None;
    let mut engine = RelaxEngineChoice::Cpu;
    let mut mc_trials: usize = 500;
    let mut offset_gate = OffsetModelGate::Disabled;
    let mut unsat_plane: Option<usize> = None;
    let mut sat_plane: Option<usize> = None;
    let mut hetnoe_order = HetNoeOrder::Auto;
    let mut seed: u64 = 2026_1302;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--series" | "--in" => {
                i += 1;
                let v = args.get(i).ok_or("--series requires a path")?;
                series_path = Some(PathBuf::from(v));
            }
            "--planes" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or("--planes requires a comma-separated list")?;
                planes_csv = Some(v.clone());
            }
            "--vdlist" => {
                i += 1;
                let v = args.get(i).ok_or("--vdlist requires a file path")?;
                vdlist_path = Some(PathBuf::from(v));
            }
            "--delays" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or("--delays requires a comma-separated delay list")?;
                delays_csv = Some(v.clone());
            }
            "--peaks" => {
                i += 1;
                let v = args.get(i).ok_or("--peaks requires a file path")?;
                peaks_path = Some(PathBuf::from(v));
            }
            "--star" | "--shifts" => {
                i += 1;
                let v = args.get(i).ok_or("--star/--shifts requires a file path")?;
                shifts_path = Some(PathBuf::from(v));
            }
            "--series-out" => {
                i += 1;
                let v = args.get(i).ok_or("--series-out requires a file path")?;
                series_out_path = Some(PathBuf::from(v));
            }
            "--out" | "-o" => {
                i += 1;
                let v = args.get(i).ok_or("--out requires a file path")?;
                out_path = Some(PathBuf::from(v));
            }
            "--kind" => {
                i += 1;
                let v = args.get(i).ok_or("--kind requires r1|r2|r1rho|hetnoe")?;
                kind_override = Some(RelaxExperimentKind::parse_label(v)?);
            }
            "--field-mhz" | "--mhz" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or("--field-mhz requires a positive number")?;
                let fm: f64 = v
                    .parse()
                    .map_err(|_| format!("invalid --field-mhz {v:?}"))?;
                if !fm.is_finite() || fm <= 0.0 {
                    return Err(format!("--field-mhz must be positive finite, got {fm}"));
                }
                field_override = Some(fm);
            }
            "--engine" => {
                i += 1;
                let v = args.get(i).ok_or("--engine requires cpu|gpu|auto")?;
                engine = RelaxEngineChoice::parse(v)?;
            }
            "--mc-trials" | "--mc" => {
                i += 1;
                let v = args.get(i).ok_or("--mc-trials requires an integer")?;
                mc_trials = v
                    .parse()
                    .map_err(|_| format!("invalid --mc-trials {v:?}"))?;
            }
            "--offset-gate" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or("--offset-gate requires none|aic|aicc|bic|always")?;
                offset_gate = parse_offset_gate(v)?;
            }
            "--unsat-plane" => {
                i += 1;
                let v = args.get(i).ok_or("--unsat-plane requires an index")?;
                unsat_plane = Some(
                    v.parse()
                        .map_err(|_| format!("invalid --unsat-plane {v:?}"))?,
                );
            }
            "--sat-plane" => {
                i += 1;
                let v = args.get(i).ok_or("--sat-plane requires an index")?;
                sat_plane = Some(
                    v.parse()
                        .map_err(|_| format!("invalid --sat-plane {v:?}"))?,
                );
            }
            "--hetnoe-order" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or("--hetnoe-order requires sat-first|ref-first")?;
                hetnoe_order = HetNoeOrder::parse_label(v)?;
            }
            "--seed" => {
                i += 1;
                let v = args.get(i).ok_or("--seed requires an integer")?;
                seed = v.parse().map_err(|_| format!("invalid --seed {v:?}"))?;
            }
            other => {
                return Err(format!(
                    "unknown `ferric relax` argument {other:?}\n\n{}",
                    relax_usage()
                ));
            }
        }
        i += 1;
    }

    let out_file = out_path
        .ok_or_else(|| format!("missing required --out <rates.tsv>\n\n{}", relax_usage()))?;

    let t0 = Instant::now();
    let ds = load_or_integrate_series(
        series_path.as_deref(),
        planes_csv.as_deref(),
        vdlist_path.as_deref(),
        delays_csv.as_deref(),
        peaks_path.as_deref(),
        shifts_path.as_deref(),
        kind_override,
        field_override,
        hetnoe_order,
    )?;

    if let Some(ref ser_out) = series_out_path {
        if let Some(parent) = ser_out.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent).map_err(|e| {
                format!(
                    "failed to create parent directory '{}': {e}",
                    parent.display()
                )
            })?;
        }
        let ext = ser_out
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let body = if ext == "nef" {
            write_relax_series_nef_str(&ds)
        } else {
            write_relax_series_tsv_str(&ds)
        };
        fs::write(ser_out, body)
            .map_err(|e| format!("failed to write --series-out '{}': {e}", ser_out.display()))?;
    }

    if let Some(parent) = out_file.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent).map_err(|e| {
            format!(
                "failed to create output directory '{}': {e}",
                parent.display()
            )
        })?;
    }

    let field_mhz = if ds.field_mhz.is_finite() && ds.field_mhz > 0.0 {
        ds.field_mhz
    } else {
        600.0
    };

    if ds.experiment_kind == RelaxExperimentKind::HetNoe {
        let (u_idx, s_idx) = infer_hetnoe_plane_indices(&ds, unsat_plane, sat_plane)?;
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
        let results = compute_hetnoe_batch(&inputs, mc_trials, seed)?;
        let tsv = format_hetnoe_tsv(field_mhz, mc_trials, &results);
        fs::write(&out_file, tsv)
            .map_err(|e| format!("failed to write '{}': {e}", out_file.display()))?;
        let med_noe = median_f64(results.iter().map(|r| r.noe).collect());
        let med_sig = median_f64(results.iter().map(|r| r.sigma_noe).collect());
        println!(
            "[relax] hetNOE ({field_mhz:.2} MHz): {} residues (unsat plane {u_idx}, sat plane {s_idx}), median hetNOE = {med_noe:.4} +- {med_sig:.4} -> '{}' ({:.1} ms)",
            results.len(),
            out_file.display(),
            t0.elapsed().as_secs_f64() * 1000.0
        );
        return Ok(());
    }

    if !matches!(
        ds.experiment_kind,
        RelaxExperimentKind::R1 | RelaxExperimentKind::R2 | RelaxExperimentKind::R1rho
    ) {
        return Err(format!(
            "`ferric relax` expects R1, R2, R1rho, or hetNOE (got {}); pass --kind r1|r2|r1rho|hetnoe",
            ds.experiment_kind.as_str()
        ));
    }

    let curves = decay_inputs_from_series(&ds);
    let fit_cfg = ExpFitConfig {
        offset_gate,
        mc_replicates: mc_trials,
        seed,
        ..ExpFitConfig::default()
    };

    let (fits, engine_used) = match engine {
        RelaxEngineChoice::Cpu => (fit_decay_batch_cpu(&curves, &fit_cfg)?, "cpu".to_string()),
        RelaxEngineChoice::Gpu => {
            let gpu = GpuEngine::new_headless()
                .map_err(|e| format!("WebGPU initialization failed for --engine gpu: {e}"))?;
            let res = fit_decay_batch_gpu(&gpu, &curves, &fit_cfg)?;
            (res, format!("gpu({})", gpu.backend_name))
        }
        RelaxEngineChoice::Auto => match GpuEngine::new_headless() {
            Ok(gpu) => match fit_decay_batch_gpu(&gpu, &curves, &fit_cfg) {
                Ok(res) => (res, format!("gpu({})", gpu.backend_name)),
                Err(_) => (
                    fit_decay_batch_cpu(&curves, &fit_cfg)?,
                    "cpu(fallback)".to_string(),
                ),
            },
            Err(_) => (
                fit_decay_batch_cpu(&curves, &fit_cfg)?,
                "cpu(fallback)".to_string(),
            ),
        },
    };

    let tsv = format_decay_rates_tsv(
        ds.experiment_kind,
        field_mhz,
        &engine_used,
        mc_trials,
        &fits,
    );
    fs::write(&out_file, tsv)
        .map_err(|e| format!("failed to write '{}': {e}", out_file.display()))?;

    let med_rate = median_f64(fits.iter().map(|f| f.rate).collect());
    let med_sig = median_f64(fits.iter().map(|f| f.sigma_rate_mc).collect());
    println!(
        "[relax] {} ({field_mhz:.2} MHz, {engine_used}): {} residues x {} planes ({} MC trials), median R = {med_rate:.4} +- {med_sig:.4} s^-1 -> '{}' ({:.1} ms)",
        ds.experiment_kind.as_str(),
        fits.len(),
        ds.schedule.len(),
        mc_trials,
        out_file.display(),
        t0.elapsed().as_secs_f64() * 1000.0
    );
    Ok(())
}

#[derive(Clone, Debug, Default)]
struct PartialFieldRateRow {
    seq_id: i32,
    res_name: String,
    field_mhz: f64,
    r1: Option<(f64, f64)>,
    r2: Option<(f64, f64)>,
    hetnoe: Option<(f64, f64)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ForcedRateKind {
    R1,
    R2,
    HetNoe,
    Auto,
}

#[inline]
fn fallback_sigma_for_observable(val: f64, kind: ForcedRateKind, rel_sigma: f64) -> f64 {
    match kind {
        ForcedRateKind::HetNoe => rel_sigma * val.abs().max(1.0),
        _ => (rel_sigma * val.abs()).max(1.0e-6),
    }
}

fn tokenize_star_line_local(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let bytes = line.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() || bytes[i] == b'#' {
            break;
        }
        if bytes[i] == b'\'' || bytes[i] == b'"' {
            let q = bytes[i];
            i += 1;
            let s0 = i;
            while i < bytes.len() {
                if bytes[i] == q && (i + 1 == bytes.len() || bytes[i + 1].is_ascii_whitespace()) {
                    break;
                }
                i += 1;
            }
            out.push(line[s0..i].to_string());
            if i < bytes.len() && bytes[i] == q {
                i += 1;
            }
        } else {
            let s0 = i;
            while i < bytes.len() && !bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            out.push(line[s0..i].to_string());
        }
    }
    out
}

fn ingest_nmrstar_relax_str(
    path: &Path,
    star_text: &str,
    default_rel_sigma: Option<f64>,
    acc: &mut BTreeMap<(i32, i64), PartialFieldRateRow>,
) -> Result<(), String> {
    #[derive(Clone, Debug)]
    struct StarSaveframe {
        tags: BTreeMap<String, String>,
        loops: Vec<(Vec<String>, Vec<Vec<String>>)>,
    }

    let mut saveframes: Vec<StarSaveframe> = Vec::new();
    let mut cur_sf: Option<StarSaveframe> = None;
    let mut in_loop = false;
    let mut reading_loop_tags = false;
    let mut cur_loop_tags: Vec<String> = Vec::new();
    let mut cur_loop_vals: Vec<String> = Vec::new();

    let flush_loop = |sf: &mut StarSaveframe, tags: &mut Vec<String>, vals: &mut Vec<String>| {
        if !tags.is_empty() && !vals.is_empty() {
            let ncols = tags.len();
            let nrows = vals.len() / ncols;
            let mut rows = Vec::with_capacity(nrows);
            for r in 0..nrows {
                rows.push(vals[r * ncols..(r + 1) * ncols].to_vec());
            }
            sf.loops.push((std::mem::take(tags), rows));
        } else {
            tags.clear();
        }
        vals.clear();
    };

    let lines: Vec<&str> = star_text.lines().collect();
    let mut idx = 0usize;
    while idx < lines.len() {
        let raw = lines[idx];
        idx += 1;
        if raw.starts_with(';') {
            while idx < lines.len() && !lines[idx].starts_with(';') {
                idx += 1;
            }
            if idx < lines.len() {
                idx += 1;
            }
            continue;
        }
        let trimmed = raw.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let toks = tokenize_star_line_local(trimmed);
        if toks.is_empty() {
            continue;
        }
        let t0_low = toks[0].to_ascii_lowercase();
        if t0_low.starts_with("save_") && toks[0].len() > 5 {
            if let Some(mut prev) = cur_sf.take() {
                if in_loop {
                    flush_loop(&mut prev, &mut cur_loop_tags, &mut cur_loop_vals);
                }
                saveframes.push(prev);
            }
            in_loop = false;
            reading_loop_tags = false;
            cur_sf = Some(StarSaveframe {
                tags: BTreeMap::new(),
                loops: Vec::new(),
            });
            continue;
        }
        if t0_low == "save_" {
            if let Some(mut prev) = cur_sf.take() {
                if in_loop {
                    flush_loop(&mut prev, &mut cur_loop_tags, &mut cur_loop_vals);
                }
                saveframes.push(prev);
            }
            in_loop = false;
            reading_loop_tags = false;
            continue;
        }
        let Some(ref mut sf) = cur_sf else {
            continue;
        };
        if t0_low == "loop_" {
            if in_loop {
                flush_loop(sf, &mut cur_loop_tags, &mut cur_loop_vals);
            }
            in_loop = true;
            reading_loop_tags = true;
            continue;
        }
        if t0_low == "stop_" {
            if in_loop {
                flush_loop(sf, &mut cur_loop_tags, &mut cur_loop_vals);
            }
            in_loop = false;
            reading_loop_tags = false;
            continue;
        }
        if in_loop {
            if reading_loop_tags && toks[0].starts_with('_') {
                for tk in toks {
                    cur_loop_tags.push(tk);
                }
            } else {
                reading_loop_tags = false;
                for tk in toks {
                    cur_loop_vals.push(tk);
                }
            }
        } else if toks[0].starts_with('_') && toks.len() >= 2 {
            let short = toks[0]
                .rsplit_once('.')
                .map(|(_, s)| s)
                .unwrap_or(&toks[0])
                .to_ascii_lowercase();
            sf.tags.insert(short, toks[1].clone());
        }
    }
    if let Some(mut last) = cur_sf.take() {
        if in_loop {
            flush_loop(&mut last, &mut cur_loop_tags, &mut cur_loop_vals);
        }
        saveframes.push(last);
    }

    // Map spectrometer IDs (`_Spectrometer.ID` -> `Field_strength` MHz)
    let mut spec_mhz: BTreeMap<String, f64> = BTreeMap::new();
    for sf in &saveframes {
        let cat = sf
            .tags
            .get("sf_category")
            .map(String::as_str)
            .unwrap_or("")
            .to_ascii_lowercase();
        if cat.contains("spectrometer") {
            if let (Some(id), Some(fs_s)) = (sf.tags.get("id"), sf.tags.get("field_strength")) {
                if let Ok(fm) = fs_s.parse::<f64>() {
                    if fm.is_finite() && fm > 0.0 {
                        spec_mhz.insert(id.clone(), fm);
                    }
                }
            }
        }
    }

    #[allow(clippy::type_complexity)]
    let mut parsed_sets: Vec<(ForcedRateKind, f64, Vec<(i32, String, f64, f64)>)> = Vec::new();
    for sf in &saveframes {
        let cat = sf
            .tags
            .get("sf_category")
            .map(String::as_str)
            .unwrap_or("")
            .to_ascii_lowercase();
        let kind_opt = if cat.contains("heteronucl_t1") || cat == "t1_relaxation" {
            Some(ForcedRateKind::R1)
        } else if cat.contains("heteronucl_t2") || cat == "t2_relaxation" {
            Some(ForcedRateKind::R2)
        } else if cat.contains("heteronucl_noe") {
            Some(ForcedRateKind::HetNoe)
        } else if cat.contains("auto_relaxation") {
            let tc = sf
                .tags
                .get("temp_calibration_method")
                .or_else(|| sf.tags.get("common_relaxation_type_name"))
                .or_else(|| sf.tags.get("sf_framecode"))
                .map(String::as_str)
                .unwrap_or("")
                .to_ascii_lowercase();
            if tc.contains("r2") || tc.contains("t2") || tc.contains("r1rho") {
                Some(ForcedRateKind::R2)
            } else {
                Some(ForcedRateKind::R1)
            }
        } else {
            None
        };
        let Some(kind) = kind_opt else {
            continue;
        };

        let fm = sf
            .tags
            .get("spectrometer_frequency_1h")
            .and_then(|s| s.parse::<f64>().ok())
            .filter(|&v| v.is_finite() && v > 0.0)
            .or_else(|| {
                sf.tags
                    .get("spectrometer_id")
                    .and_then(|id| spec_mhz.get(id).copied())
            })
            .unwrap_or(600.0);

        let units_raw = sf
            .tags
            .get("t1_val_units")
            .or_else(|| sf.tags.get("t2_val_units"))
            .or_else(|| sf.tags.get("auto_relaxation_val_units"))
            .map(|s| s.to_ascii_lowercase())
            .unwrap_or_default();
        let units = if !units_raw.is_empty() && units_raw != "." && units_raw != "?" {
            units_raw
        } else if cat.contains("auto_relax") {
            "s-1".to_string()
        } else {
            // In NMR-STAR v2/v3 `_Heteronucl_T1/T2_relaxation`, omitted/`.` `T1_val_units` defaults to seconds (`s`).
            "s".to_string()
        };

        for (ltags, lrows) in &sf.loops {
            let short_tags: Vec<String> = ltags
                .iter()
                .map(|t| {
                    t.rsplit_once('.')
                        .map(|(_, s)| s)
                        .unwrap_or(t)
                        .to_ascii_lowercase()
                })
                .collect();
            let seq_cols: Vec<usize> = [
                "seq_id",
                "comp_index_id",
                "auth_seq_id",
                "seq_id_1",
                "comp_index_id_1",
                "auth_seq_id_1",
            ]
            .iter()
            .filter_map(|cand| short_tags.iter().position(|s| s == cand))
            .collect();
            let val_col = short_tags.iter().position(|s| {
                matches!(
                    s.as_str(),
                    "val"
                        | "t1_val"
                        | "t2_val"
                        | "noe_val"
                        | "auto_relaxation_val"
                        | "heteronucl_noe_val"
                )
            });
            let err_col = short_tags.iter().position(|s| {
                matches!(
                    s.as_str(),
                    "val_err"
                        | "t1_val_err"
                        | "t2_val_err"
                        | "noe_val_err"
                        | "auto_relaxation_val_err"
                        | "heteronucl_noe_val_err"
                )
            });
            let res_cols: Vec<usize> = ["comp_id", "comp_id_1", "auth_comp_id", "auth_comp_id_1"]
                .iter()
                .filter_map(|cand| short_tags.iter().position(|s| s == cand))
                .collect();
            let atom_col = short_tags
                .iter()
                .position(|s| matches!(s.as_str(), "atom_id" | "atom_id_1" | "auth_atom_id"));

            let Some(vc) = val_col else {
                continue;
            };
            if seq_cols.is_empty() {
                continue;
            }
            let mut rows_out = Vec::new();
            for r in lrows {
                if let Some(ac) = atom_col {
                    let at = r[ac].to_ascii_uppercase();
                    if !matches!(at.as_str(), "N" | "HN" | "H" | "15N" | "N15" | "." | "?") {
                        continue;
                    }
                }
                let Some(seq_id) = seq_cols
                    .iter()
                    .find_map(|&sc| r.get(sc).and_then(|s| s.trim().parse::<i32>().ok()))
                else {
                    continue;
                };
                let v_str = r[vc].trim();
                if v_str.is_empty() || v_str == "." || v_str == "?" {
                    continue;
                }
                let Ok(raw_v) = v_str.parse::<f64>() else {
                    continue;
                };
                if !raw_v.is_finite() {
                    continue;
                }
                let res_name = res_cols
                    .iter()
                    .find_map(|&rc| {
                        r.get(rc)
                            .map(|s| s.trim())
                            .filter(|s| !s.is_empty() && *s != "." && *s != "?")
                            .map(str::to_string)
                    })
                    .unwrap_or_else(|| "UNK".to_string());

                let raw_err_opt = err_col.and_then(|ec| {
                    let es = r[ec].trim();
                    if es.is_empty() || es == "." || es == "?" || es.eq_ignore_ascii_case("na") {
                        None
                    } else {
                        es.parse::<f64>().ok().filter(|&e| e.is_finite() && e > 0.0)
                    }
                });
                let raw_err = match raw_err_opt {
                    Some(e) => e,
                    None => {
                        let rel = default_rel_sigma.ok_or_else(|| {
                            format!(
                                "NMR-STAR relaxation saveframe in '{}' has null/missing Val_err ('.') at seq_id={seq_id}; pass explicit --default-rel-sigma <frac> (e.g. 0.03) to opt into relative uncertainty assignment",
                                path.display()
                            )
                        })?;
                        fallback_sigma_for_observable(raw_v, kind, rel)
                    }
                };

                let (final_val, final_sig) = if kind == ForcedRateKind::HetNoe {
                    (raw_v, raw_err)
                } else if units == "s" {
                    if raw_v <= 0.0 {
                        continue;
                    }
                    (1.0 / raw_v, raw_err / (raw_v * raw_v))
                } else if units == "ms" {
                    if raw_v <= 0.0 {
                        continue;
                    }
                    let t_sec = raw_v * 1.0e-3;
                    let sig_sec = raw_err * 1.0e-3;
                    (1.0 / t_sec, sig_sec / (t_sec * t_sec))
                } else {
                    if raw_v <= 0.0 {
                        continue;
                    }
                    (raw_v, raw_err)
                };
                rows_out.push((seq_id, res_name, final_val, final_sig));
            }
            if !rows_out.is_empty() {
                parsed_sets.push((kind, fm, rows_out));
            }
        }
    }

    if parsed_sets.is_empty() {
        return Err(format!(
            "NMR-STAR file '{}' contains no 15N R1/R2/hetNOE relaxation saveframes",
            path.display()
        ));
    }

    // Disambiguate duplicate nominal spectrometer frequencies ONLY when R1 medians differ
    // substantially (`|med0 - med1| / max(med0, med1) >= 0.10`, indicating distinct B0 fields);
    // when `< 0.10` (technical replicates at the same B0 field, e.g. BMRB:6470 set_1 & set_2),
    // pool replicates at their shared field via inverse-variance weighting!
    let mut alt_other_mhz: Option<f64> = None;
    for &v in spec_mhz.values() {
        if (v - 600.0).abs() > 25.0 {
            alt_other_mhz = Some(v);
        }
    }
    let r1_indices: Vec<usize> = parsed_sets
        .iter()
        .enumerate()
        .filter(|(_, (k, _, _))| *k == ForcedRateKind::R1)
        .map(|(i, _)| i)
        .collect();
    if r1_indices.len() == 2 {
        let (r0, r1) = (r1_indices[0], r1_indices[1]);
        if (parsed_sets[r0].1 - parsed_sets[r1].1).abs() < 1.0 {
            let med0 = median_f64(parsed_sets[r0].2.iter().map(|r| r.2).collect());
            let med1 = median_f64(parsed_sets[r1].2.iter().map(|r| r.2).collect());
            let rel_diff = (med0 - med1).abs() / med0.max(med1).max(1.0e-9);
            if rel_diff >= 0.10 {
                let base_mhz = parsed_sets[r0].1;
                let other_mhz = alt_other_mhz.unwrap_or(360.13);
                let second_is_other = if other_mhz < base_mhz {
                    med1 > med0 // Lower B0 -> larger R1 (`1/T1`)
                } else {
                    med1 < med0 // Higher B0 -> smaller R1 (`1/T1`)
                };
                for check_kind in [
                    ForcedRateKind::R1,
                    ForcedRateKind::R2,
                    ForcedRateKind::HetNoe,
                ] {
                    let idxs: Vec<usize> = parsed_sets
                        .iter()
                        .enumerate()
                        .filter(|(_, (k, _, _))| *k == check_kind)
                        .map(|(i, _)| i)
                        .collect();
                    if idxs.len() == 2 {
                        if second_is_other {
                            parsed_sets[idxs[1]].1 = other_mhz;
                        } else {
                            parsed_sets[idxs[0]].1 = other_mhz;
                        }
                    }
                }
            }
        }
    }

    let pool_pair = |cur: Option<(f64, f64)>, val: f64, sig: f64| -> Option<(f64, f64)> {
        match cur {
            None => Some((val, sig)),
            Some((v0, s0)) => {
                let w0 = 1.0 / (s0 * s0).max(1.0e-18);
                let w1 = 1.0 / (sig * sig).max(1.0e-18);
                let wsum = w0 + w1;
                Some(((w0 * v0 + w1 * val) / wsum, (1.0 / wsum).sqrt()))
            }
        }
    };

    for (kind, fm, rows) in parsed_sets {
        let field_key = (fm * 100.0).round() as i64;
        for (seq_id, res_name, val, sig) in rows {
            let entry = acc
                .entry((seq_id, field_key))
                .or_insert_with(|| PartialFieldRateRow {
                    seq_id,
                    res_name: res_name.clone(),
                    field_mhz: fm,
                    ..PartialFieldRateRow::default()
                });
            if entry.res_name == "UNK" && res_name != "UNK" {
                entry.res_name = res_name;
            }
            match kind {
                ForcedRateKind::R1 => entry.r1 = pool_pair(entry.r1, val, sig),
                ForcedRateKind::R2 => entry.r2 = pool_pair(entry.r2, val, sig),
                ForcedRateKind::HetNoe => entry.hetnoe = pool_pair(entry.hetnoe, val, sig),
                ForcedRateKind::Auto => {}
            }
        }
    }
    Ok(())
}

fn ingest_rate_tsv_file(
    path: &Path,
    forced_kind: ForcedRateKind,
    default_rel_sigma: Option<f64>,
    acc: &mut BTreeMap<(i32, i64), PartialFieldRateRow>,
) -> Result<(), String> {
    let text = fs::read_to_string(path)
        .map_err(|e| format!("failed to read rate table '{}': {e}", path.display()))?;
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if ext == "str"
        || ext == "star"
        || text.trim_start().starts_with("data_")
        || text.contains("_Heteronucl_T1_relaxation")
    {
        return ingest_nmrstar_relax_str(path, &text, default_rel_sigma, acc);
    }

    let mut header_field_mhz: Option<f64> = None;
    let mut header_kind: Option<ForcedRateKind> = None;
    let mut col_idx: BTreeMap<String, usize> = BTreeMap::new();

    for (line_no, raw_line) in text.lines().enumerate() {
        let trimmed = raw_line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(comment) = trimmed.strip_prefix('#') {
            for tok in comment.split_whitespace() {
                if let Some(v) = tok.strip_prefix("field_mhz=") {
                    if let Ok(fm) = v.parse::<f64>() {
                        if fm.is_finite() && fm > 0.0 {
                            header_field_mhz = Some(fm);
                        }
                    }
                } else if let Some(k) = tok.strip_prefix("kind=") {
                    header_kind = match k.to_ascii_lowercase().as_str() {
                        "r1" | "t1" => Some(ForcedRateKind::R1),
                        "r2" | "t2" | "r1rho" => Some(ForcedRateKind::R2),
                        "hetnoe" | "noe" => Some(ForcedRateKind::HetNoe),
                        _ => None,
                    };
                }
            }
            continue;
        }

        let cols: Vec<&str> = trimmed.split('\t').map(str::trim).collect();
        if col_idx.is_empty() {
            for (idx, name) in cols.iter().enumerate() {
                col_idx.insert(name.to_ascii_lowercase(), idx);
            }
            if !col_idx.contains_key("seq_id") {
                return Err(format!(
                    "rate TSV '{}' line {} is missing required `seq_id` column header",
                    path.display(),
                    line_no + 1
                ));
            }
            continue;
        }

        let get_str = |name: &str| -> Option<&str> {
            col_idx.get(name).and_then(|&idx| cols.get(idx)).copied()
        };
        let get_f64 = |name: &str| -> Option<f64> {
            get_str(name).and_then(|s| {
                if s.is_empty()
                    || s == "."
                    || s.eq_ignore_ascii_case("na")
                    || s.eq_ignore_ascii_case("nan")
                {
                    None
                } else {
                    s.parse::<f64>().ok().filter(|v| v.is_finite())
                }
            })
        };

        let seq_id: i32 = get_str("seq_id")
            .ok_or_else(|| {
                format!(
                    "rate TSV '{}' line {}: missing seq_id",
                    path.display(),
                    line_no + 1
                )
            })?
            .parse()
            .map_err(|_| {
                format!(
                    "rate TSV '{}' line {}: invalid integer seq_id",
                    path.display(),
                    line_no + 1
                )
            })?;
        let res_name = get_str("res_name").unwrap_or("UNK").to_string();
        let field_mhz = get_f64("field_mhz").or(header_field_mhz).unwrap_or(600.0);
        let field_key = (field_mhz * 100.0).round() as i64;

        let entry = acc
            .entry((seq_id, field_key))
            .or_insert_with(|| PartialFieldRateRow {
                seq_id,
                res_name: res_name.clone(),
                field_mhz,
                ..PartialFieldRateRow::default()
            });
        if entry.res_name == "UNK" && res_name != "UNK" {
            entry.res_name = res_name;
        }

        // Combined multi-column format (`r1`, `sigma_r1`, `r2`, `sigma_r2`, `hetnoe`, `sigma_hetnoe`)
        if col_idx.contains_key("r1") || col_idx.contains_key("r2") {
            if let Some(r1) = get_f64("r1") {
                let sr1 = match get_f64("sigma_r1").filter(|&s| s > 0.0) {
                    Some(s) => s,
                    None => {
                        let rel = default_rel_sigma.ok_or_else(|| {
                            format!(
                                "rate TSV '{}' line {}: null/missing sigma_r1; pass explicit --default-rel-sigma <frac> to opt in",
                                path.display(),
                                line_no + 1
                            )
                        })?;
                        fallback_sigma_for_observable(r1, ForcedRateKind::R1, rel)
                    }
                };
                entry.r1 = Some((r1, sr1));
            }
            if let Some(r2) = get_f64("r2") {
                let sr2 = match get_f64("sigma_r2").filter(|&s| s > 0.0) {
                    Some(s) => s,
                    None => {
                        let rel = default_rel_sigma.ok_or_else(|| {
                            format!(
                                "rate TSV '{}' line {}: null/missing sigma_r2; pass explicit --default-rel-sigma <frac> to opt in",
                                path.display(),
                                line_no + 1
                            )
                        })?;
                        fallback_sigma_for_observable(r2, ForcedRateKind::R2, rel)
                    }
                };
                entry.r2 = Some((r2, sr2));
            }
            if let Some(n) = get_f64("hetnoe").or_else(|| get_f64("noe")) {
                let sn = match get_f64("sigma_hetnoe")
                    .or_else(|| get_f64("sigma_noe"))
                    .filter(|&s| s > 0.0)
                {
                    Some(s) => s,
                    None => {
                        let rel = default_rel_sigma.ok_or_else(|| {
                            format!(
                                "rate TSV '{}' line {}: null/missing sigma_hetnoe; pass explicit --default-rel-sigma <frac> to opt in",
                                path.display(),
                                line_no + 1
                            )
                        })?;
                        fallback_sigma_for_observable(n, ForcedRateKind::HetNoe, rel)
                    }
                };
                entry.hetnoe = Some((n, sn));
            }
            continue;
        }

        // Single-observable `ferric relax` TSV (`value`/`rate`, `sigma`)
        let val = get_f64("value")
            .or_else(|| get_f64("rate"))
            .or_else(|| get_f64("noe"))
            .ok_or_else(|| {
                format!(
                    "rate TSV '{}' line {}: missing `value` column",
                    path.display(),
                    line_no + 1
                )
            })?;

        let row_kind = if forced_kind != ForcedRateKind::Auto {
            forced_kind
        } else if let Some(k_str) = get_str("kind") {
            match k_str.to_ascii_lowercase().as_str() {
                "r1" | "t1" => ForcedRateKind::R1,
                "r2" | "t2" | "r1rho" => ForcedRateKind::R2,
                "hetnoe" | "noe" => ForcedRateKind::HetNoe,
                other => {
                    return Err(format!(
                        "rate TSV '{}' line {}: unrecognized kind {other:?}",
                        path.display(),
                        line_no + 1
                    ))
                }
            }
        } else if let Some(hk) = header_kind {
            hk
        } else {
            return Err(format!(
                "rate TSV '{}' has no `kind` column or header; pass via --r1, --r2, or --hetnoe",
                path.display()
            ));
        };

        let sig = match get_f64("sigma")
            .or_else(|| get_f64("sigma_rate"))
            .or_else(|| get_f64("sigma_noe"))
            .filter(|&s| s > 0.0)
        {
            Some(s) => s,
            None => {
                let rel = default_rel_sigma.ok_or_else(|| {
                    format!(
                        "rate TSV '{}' line {}: missing/null `sigma` value; pass explicit --default-rel-sigma <frac> to opt in",
                        path.display(),
                        line_no + 1
                    )
                })?;
                fallback_sigma_for_observable(val, row_kind, rel)
            }
        };

        match row_kind {
            ForcedRateKind::R1 => entry.r1 = Some((val, sig)),
            ForcedRateKind::R2 => entry.r2 = Some((val, sig)),
            ForcedRateKind::HetNoe => entry.hetnoe = Some((val, sig)),
            ForcedRateKind::Auto => unreachable!(),
        }
    }
    Ok(())
}

fn format_rsd_tsv(
    rate_tables: &[Vec<ResidueRelaxRates>],
    phys: &RelaxPhysicsConfig,
) -> Result<String, String> {
    let mut out = String::from(
        "# Farrow et al. (1995) Reduced Spectral Density Mapping & Kay-Torchia-Bax Trimmed tau_m\n",
    );
    for table in rate_tables {
        if table.is_empty() {
            continue;
        }
        let fm = table[0].field_mhz;
        if let Ok(tm) = estimate_trimmed_tau_m(table, &TrimmedTauMConfig::default()) {
            let n_kept = tm.retained_seq_ids.len();
            let n_total = n_kept + tm.excluded_seq_ids.len();
            out.push_str(&format!(
                "# field_mhz={fm:.2} trimmed_tau_m_ns={:.4} sigma_tau_m_ns={:.4} n_kept={n_kept}/{n_total}\n",
                tm.tau_m_sec * 1.0e9,
                tm.sigma_tau_m_sec * 1.0e9,
            ));
        }
    }
    out.push_str(
        "seq_id\tres_name\tfield_mhz\tj0_ns_rad\tsigma_j0\tj_wn_ns_rad\tsigma_j_wn\tj_087wh_ps_rad\tsigma_j_087wh\n",
    );
    for table in rate_tables {
        let mapped = map_reduced_spectral_density_batch(table, phys)?;
        for r in mapped {
            out.push_str(&format!(
                "{}\t{}\t{:.4}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\n",
                r.seq_id,
                r.res_name,
                r.field_mhz,
                r.j_0 * 1.0e9,
                r.sigma_j_0 * 1.0e9,
                r.j_omega_n * 1.0e9,
                r.sigma_j_omega_n * 1.0e9,
                r.j_087_omega_h * 1.0e12,
                r.sigma_j_087_omega_h * 1.0e12,
            ));
        }
    }
    Ok(out)
}

fn describe_diffusion_tensor(tensor: &DiffusionTensor) -> String {
    let tm_ns = tensor.effective_tau_m_sec() * 1.0e9;
    match tensor {
        DiffusionTensor::Isotropic { .. } => {
            format!("Isotropic(tau_m = {tm_ns:.3} ns)")
        }
        DiffusionTensor::AxialSymmetric {
            d_perp_s_inv,
            d_par_s_inv,
            theta_rad,
            phi_rad,
        } => {
            format!(
                "AxialSymmetric(tau_m = {tm_ns:.3} ns, D_par/D_perp = {:.4}, theta = {:.1} deg, phi = {:.1} deg)",
                d_par_s_inv / d_perp_s_inv.max(1.0),
                theta_rad.to_degrees(),
                phi_rad.to_degrees()
            )
        }
        DiffusionTensor::FullyAnisotropic {
            dx_s_inv,
            dz_s_inv,
            alpha_rad,
            beta_rad,
            gamma_rad,
            ..
        } => {
            format!(
                "FullyAnisotropic(tau_m = {tm_ns:.3} ns, Dz/Dx = {:.4}, euler = [{:.1}, {:.1}, {:.1}] deg)",
                dz_s_inv / dx_s_inv.max(1.0),
                alpha_rad.to_degrees(),
                beta_rad.to_degrees(),
                gamma_rad.to_degrees()
            )
        }
    }
}

/// Entry point for `ferric modelfree`.
pub fn run_modelfree_cli(args: &[String]) -> Result<(), String> {
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        println!("{}", modelfree_usage());
        return Ok(());
    }

    let mut input_files: Vec<(PathBuf, ForcedRateKind)> = Vec::new();
    let mut pdb_path: Option<PathBuf> = None;
    let mut diffusion_opt: Option<DiffusionTensorKind> = None;
    let mut criterion = ModelSelectionCriterion::Aic;
    let mut ref_field_mhz: Option<f64> = None;
    let mut mc_trials: usize = 0;
    let mut r_nh: f64 = 1.02;
    let mut csa_ppm: f64 = -172.0;
    let mut default_rel_sigma: Option<f64> = None;
    let mut seed: u64 = 2026_1303;
    let mut rsd_out_path: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--r1" => {
                i += 1;
                let v = args.get(i).ok_or("--r1 requires a TSV file path")?;
                input_files.push((PathBuf::from(v), ForcedRateKind::R1));
            }
            "--r2" => {
                i += 1;
                let v = args.get(i).ok_or("--r2 requires a TSV file path")?;
                input_files.push((PathBuf::from(v), ForcedRateKind::R2));
            }
            "--hetnoe" | "--noe" => {
                i += 1;
                let v = args.get(i).ok_or("--hetnoe requires a TSV file path")?;
                input_files.push((PathBuf::from(v), ForcedRateKind::HetNoe));
            }
            "--rates" | "--in" | "--star" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or("--rates/--star requires a TSV or NMR-STAR file path")?;
                input_files.push((PathBuf::from(v), ForcedRateKind::Auto));
            }
            "--pdb" => {
                i += 1;
                let v = args.get(i).ok_or("--pdb requires a PDB file path")?;
                pdb_path = Some(PathBuf::from(v));
            }
            "--diffusion" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or("--diffusion requires auto|isotropic|axial|anisotropic")?;
                diffusion_opt = Some(parse_diffusion_kind(v)?);
            }
            "--criterion" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or("--criterion requires aic|aicc|bic|ftest")?;
                criterion = parse_criterion(v)?;
            }
            "--ref-mhz" => {
                i += 1;
                let v = args.get(i).ok_or("--ref-mhz requires a positive number")?;
                let fm: f64 = v.parse().map_err(|_| format!("invalid --ref-mhz {v:?}"))?;
                if !fm.is_finite() || fm <= 0.0 {
                    return Err(format!("--ref-mhz must be positive finite, got {fm}"));
                }
                ref_field_mhz = Some(fm);
            }
            "--mc-trials" | "--mc" => {
                i += 1;
                let v = args.get(i).ok_or("--mc-trials requires an integer")?;
                mc_trials = v
                    .parse()
                    .map_err(|_| format!("invalid --mc-trials {v:?}"))?;
            }
            "--r-nh" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or("--r-nh requires bond length in Angstroms")?;
                r_nh = v.parse().map_err(|_| format!("invalid --r-nh {v:?}"))?;
                if !r_nh.is_finite() || r_nh <= 0.5 || r_nh >= 2.0 {
                    return Err(format!("--r-nh out of physical range (0.5..2.0 A): {r_nh}"));
                }
            }
            "--csa-ppm" => {
                i += 1;
                let v = args.get(i).ok_or("--csa-ppm requires 15N CSA in ppm")?;
                csa_ppm = v.parse().map_err(|_| format!("invalid --csa-ppm {v:?}"))?;
                if !csa_ppm.is_finite() || csa_ppm.abs() < 10.0 || csa_ppm.abs() > 400.0 {
                    return Err(format!("--csa-ppm out of physical range: {csa_ppm}"));
                }
            }
            "--default-rel-sigma" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or("--default-rel-sigma requires a fractional uncertainty in (0, 1)")?;
                let rel: f64 = v
                    .parse()
                    .map_err(|_| format!("invalid --default-rel-sigma {v:?}"))?;
                if !rel.is_finite() || rel <= 0.0 || rel >= 1.0 {
                    return Err(format!(
                        "--default-rel-sigma must be strictly in (0.0, 1.0), got {rel}"
                    ));
                }
                default_rel_sigma = Some(rel);
            }
            "--seed" => {
                i += 1;
                let v = args.get(i).ok_or("--seed requires an integer")?;
                seed = v.parse().map_err(|_| format!("invalid --seed {v:?}"))?;
            }
            "--rsd-out" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or("--rsd-out requires an output file path")?;
                rsd_out_path = Some(PathBuf::from(v));
            }
            "--out" | "-o" => {
                i += 1;
                let v = args.get(i).ok_or("--out requires an output file path")?;
                out_path = Some(PathBuf::from(v));
            }
            other => {
                return Err(format!(
                    "unknown `ferric modelfree` argument {other:?}\n\n{}",
                    modelfree_usage()
                ));
            }
        }
        i += 1;
    }

    if input_files.is_empty() {
        return Err(format!(
            "missing relaxation rate inputs (--r1 / --r2 / --hetnoe or --rates)\n\n{}",
            modelfree_usage()
        ));
    }
    let out_file = out_path.ok_or_else(|| {
        format!(
            "missing required --out <modelfree.tsv>\n\n{}",
            modelfree_usage()
        )
    })?;

    let t0 = Instant::now();
    let mut acc: BTreeMap<(i32, i64), PartialFieldRateRow> = BTreeMap::new();
    for (path, fkind) in &input_files {
        ingest_rate_tsv_file(path, *fkind, default_rel_sigma, &mut acc)?;
    }

    let mut by_field: BTreeMap<i64, Vec<ResidueRelaxRates>> = BTreeMap::new();
    for ((_seq_id, field_key), row) in acc {
        if let (Some((r1, sr1)), Some((r2, sr2))) = (row.r1, row.r2) {
            let (hetnoe, sigma_hetnoe) = match row.hetnoe {
                Some((n, sn)) => (Some(n), Some(sn)),
                None => (None, None),
            };
            by_field
                .entry(field_key)
                .or_default()
                .push(ResidueRelaxRates {
                    seq_id: row.seq_id,
                    res_name: row.res_name,
                    field_mhz: row.field_mhz,
                    r1,
                    sigma_r1: sr1,
                    r2,
                    sigma_r2: sr2,
                    hetnoe,
                    sigma_hetnoe,
                });
        }
    }

    if by_field.is_empty() {
        return Err(
            "no residues have both R1 and R2 rates at a shared field_mhz; provide matching --r1 and --r2 tables"
                .to_string(),
        );
    }
    let rate_tables: Vec<Vec<ResidueRelaxRates>> = by_field.into_values().collect();

    let nh_vectors = if let Some(ref pdb) = pdb_path {
        let vecs = extract_nh_unit_vectors_from_pdb(pdb)?;
        if vecs.is_empty() {
            return Err(format!(
                "PDB '{}' yielded 0 backbone N-H vectors",
                pdb.display()
            ));
        }
        Some(vecs)
    } else {
        None
    };

    let diffusion_kind = diffusion_opt.unwrap_or(if nh_vectors.is_some() {
        DiffusionTensorKind::Auto
    } else {
        DiffusionTensorKind::Isotropic
    });

    let phys = RelaxPhysicsConfig {
        r_nh_angstrom: r_nh,
        csa_n_ppm: csa_ppm,
        ..RelaxPhysicsConfig::default()
    };

    if let Some(ref rsd_out) = rsd_out_path {
        if let Some(parent) = rsd_out.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent).map_err(|e| {
                format!(
                    "failed to create parent directory '{}': {e}",
                    parent.display()
                )
            })?;
        }
        let rsd_tsv = format_rsd_tsv(&rate_tables, &phys)?;
        fs::write(rsd_out, rsd_tsv)
            .map_err(|e| format!("failed to write --rsd-out '{}': {e}", rsd_out.display()))?;
    }

    let grouped = group_multifield_rates(&rate_tables, nh_vectors.as_deref());
    let mf_cfg = ModelFreeConfig {
        diffusion_kind,
        selection_criterion: criterion,
        physics: phys,
        ref_field_mhz,
        mc_replicates: mc_trials,
        seed,
        ..ModelFreeConfig::default()
    };

    let mf_res = fit_modelfree_dataset(&grouped, &mf_cfg)?;

    if let Some(parent) = out_file.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent).map_err(|e| {
            format!(
                "failed to create output directory '{}': {e}",
                parent.display()
            )
        })?;
    }
    let tsv = write_modelfree_tsv_str(&mf_res);
    fs::write(&out_file, tsv)
        .map_err(|e| format!("failed to write '{}': {e}", out_file.display()))?;

    let mut counts = [0usize; 5];
    for r in &mf_res.residues {
        let idx = match r.selected_model {
            ModelFreeId::M1 => 0,
            ModelFreeId::M2 => 1,
            ModelFreeId::M3 => 2,
            ModelFreeId::M4 => 3,
            ModelFreeId::M5 => 4,
        };
        counts[idx] += 1;
    }
    let med_s2 = median_f64(mf_res.residues.iter().map(|r| r.params.s2).collect());
    println!(
        "[modelfree] {} residues across {} field(s), {}, criterion={:?}, models=[M1:{}, M2:{}, M3:{}, M4:{}, M5:{}], median S^2 = {med_s2:.4} -> '{}' ({:.1} ms)",
        mf_res.residues.len(),
        rate_tables.len(),
        describe_diffusion_tensor(&mf_res.diffusion_tensor),
        criterion,
        counts[0],
        counts[1],
        counts[2],
        counts[3],
        counts[4],
        out_file.display(),
        t0.elapsed().as_secs_f64() * 1000.0
    );
    Ok(())
}
