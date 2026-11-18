//! CLI subcommands `ferric cest`, `ferric cpmg`, and `ferric csp` for Slice `13.6`
//! (`T13.4`–`T13.6`): 2-/3-state Bloch–McConnell `CEST`/`DEST` matrix-exponential fitting,
//! Carver–Richards / Baldwin `CPMG` & Trott–Palmer off-resonance $R_{1\rho}$ relaxation
//! dispersion, and Hungarian fast-exchange `HSQC`/`HMQC` ligand titration `CSP` + quadratic
//! tight-binding $1:1$ $K_d$ isotherm analysis.
//!
//! # Literature & Clean-Room Provenance
//! - McConnell, H. M. (*J. Chem. Phys.* 1958, 28, 430–431).
//! - Helgstrand, M., Härd, T. & Allard, P. (*J. Biomol. NMR* 2000, 18, 47–61).
//! - Vallurupalli, P., Bouvignies, G. & Kay, L. E. (*J. Am. Chem. Soc.* 2012, 134, 8148–8161).
//! - Fawzi, N. L., Ying, J., Ghirlando, R., Torchia, D. A. & Clore, G. M. (*Nature* 2011, 480, 268–272).
//! - Higham, N. J. (*SIAM J. Matrix Anal. Appl.* 2005, 26, 1179–1193).
//! - Luz, Z. & Meiboom, S. (*J. Chem. Phys.* 1963, 39, 366–370).
//! - Carver, J. P. & Richards, R. E. (*J. Magn. Reson.* 1972, 6, 89–105).
//! - Baldwin, A. J. (*J. Magn. Reson.* 2014, 244, 114–124).
//! - Trott, O. & Palmer, A. G. III (*J. Magn. Reson.* 2002, 154, 157–160).
//! - Williamson, M. P. (*Prog. Nucl. Magn. Reson. Spectrosc.* 2013, 73, 1–16).
//!
//! Implemented from the published equations and `docs/specs/16-dynamics-and-csp.md` §§3–5 without
//! consulting external chemical-exchange or titration software source code.

use std::collections::{BTreeMap, BTreeSet};
use std::f64::consts::TAU;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use nmr_core::cest::{
    fit_cest_3state_profile_cpu, fit_cest_global_cpu, format_cest_3state_fit_json,
    format_cest_fit_json, format_cest_fit_tsv, read_cest_profiles_tsv, Cest3StateFitResult,
    CestFitConfig, CestProfileDataset,
};
use nmr_core::cpmg::{
    compute_r2eff_from_intensities, fit_cpmg_global_cpu, fit_r1rho_trott_palmer_cpu,
    format_cpmg_fit_json, format_cpmg_fit_tsv, format_r1rho_fit_json, read_cpmg_series_tsv,
    write_cpmg_series_tsv, CpmgFitConfig, CpmgModel, CpmgSeriesDataset, ExchangeGateConfig,
    R1RhoFitResult, R1RhoOffResPoint, RawCpmgIntensityPlane,
};
use nmr_core::csp::{
    diluted_protein_conc_for_target_ligand, fit_csp_global_kd, format_csp_json, format_csp_tsv,
    track_methyl_hmqc_trajectories, track_titration_trajectories, write_csp_trajectories_tsv,
    AssignedAmidePeak, CspAnalysisConfig, ResidueCspTrajectory, ResidueTitrationPoint,
    UnassignedHsqcPeak, UnassignedTitrationStep,
};
use nmr_core::excited_state::{
    classify_excited_state_with_firewall, format_excited_state_json, format_excited_state_tsv,
    parse_exchange_observations_tsv_with_meta, parse_ground_shifts_str, AltConformerInput,
    ExchangeShiftObservation, ExcitedStateConfig,
};
use nmr_core::nmrstar::{read_star, read_star_all_atoms};
use nmr_core::peaks::read_peaks;
use nmr_core::picker::{pick_c13_hsqc_2d, pick_hsqc_2d, PickerConfig};
use nmr_core::spectrum::{
    is_ucsf_file, read_bruker_pdata_2d, read_nmrpipe_2d, read_ucsf_2d, Spectrum2D,
};
use nmr_core::struct_shifts::StructureModel;
use nmr_core::torsion_db::HomologFirewallQuery;
use nmr_gpu::cest_gpu::fit_cest_global_gpu;
use nmr_gpu::cpmg_gpu::fit_cpmg_global_gpu;
use nmr_gpu::GpuEngine;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExchangeEngineChoice {
    Cpu,
    Gpu,
    Auto,
}

impl ExchangeEngineChoice {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TitrationExperimentKind {
    AmideHsqc,
    MethylHmqc,
}

impl TitrationExperimentKind {
    fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "hsqc" | "amide" | "15n" | "nh" => Ok(Self::AmideHsqc),
            "hmqc" | "methyl" | "13c" | "ch3" => Ok(Self::MethylHmqc),
            other => Err(format!(
                "invalid --experiment {other:?} (expected hsqc or hmqc)"
            )),
        }
    }
}

pub fn cest_usage() -> &'static str {
    "ferric cest — 2-/3-state Bloch-McConnell CEST & DEST Padé [6/6] matrix-exponential solver\n\
     Usage:\n\
       ferric cest --in <profiles.tsv> [--in <profiles_b2.tsv> ...] [--out <fit.tsv>] [options]\n\n\
     Inputs & Physics:\n\
       --in | --profiles <tsv>  CEST/DEST saturation profile TSV (repeatable across B1 fields)\n\
       --states <2|3>           Chemical exchange topology: 2 (global A<->B) or 3 (A<->B<->C) [default: 2]\n\
       --triangular             For --states 3, co-fit direct k_ex,AC alongside k_ex,AB & k_ex,BC\n\
       --b1-quad <N>            Gauss-Hermite B1-inhomogeneity quadrature nodes (1,3,5,7,9) [default: 5]\n\
       --engine <cpu|gpu|auto>  Compute backend for 2-state global Padé [6/6] LM [default: cpu]\n\
       --max-iters <N>          Maximum Levenberg-Marquardt iterations [default: 45]\n\
       --no-refine-major        Hold major-state chemical shift hints fixed during global LM\n\
       --chi2-surface           Compute 1D & 2D (p_B, k_ex) profile chi^2 confidence surfaces\n\
       --json-out <fit.json>    Optional JSON output path (includes chi^2 surface when enabled)\n\
       --out | -o <fit.tsv>     Output fitted CEST/DEST parameter TSV table"
}

pub fn cpmg_usage() -> &'static str {
    "ferric cpmg — Multi-field CPMG (LM63 / CR72 / Baldwin2014) & Trott-Palmer off-resonance R1rho solver\n\
     Usage:\n\
       ferric cpmg (--in <cpmg.tsv> ... | --raw-planes <planes.tsv> ... | --r1rho <r1rho.tsv> --major-ppm <ppm> --larmor-mhz <mhz>) [--out <fit.tsv>] [options]\n\n\
     CPMG Inputs & Dispersion Models:\n\
       --in | --series <tsv>    Processed R2,eff(nu_CPMG) series TSV (repeatable for 600+800+900 MHz)\n\
       --raw-planes <tsv>       Raw constant-time CPMG intensity plane TSV (nu_cpmg_hz=0 for I0)\n\
       --series-out <tsv>       Optional path to export R2,eff(nu_CPMG) series computed from --raw-planes\n\
       --sigma-floor <s_inv>    Minimum R2,eff uncertainty floor in s^-1 for --raw-planes [default: 0.15]\n\
       --model <name>           baldwin2014 | cr72 | lm63 | numerical2x2 | flat [default: baldwin2014]\n\
       --engine <cpu|gpu|auto>  Compute backend for global (p_B, k_ex) grid + Schur LM [default: cpu]\n\
       --min-rex <s_inv>        Exchange gate minimum Delta R2,eff step in s^-1 [default: 2.0]\n\
       --min-daic <val>         Exchange gate minimum AICc improvement over Flat null [default: 2.0]\n\
       --max-fp <p>             Exchange gate maximum nested F-test p-value [default: 0.05]\n\
       --no-gate                Include all residues in global fit without Flat null gating\n\n\
     Off-Resonance R1rho Mode (Trott & Palmer 2002):\n\
       --r1rho <r1rho.tsv>      Off-resonance R1rho table (carrier_ppm, spin_lock_hz, r1rho_s_inv, r1rho_err)\n\
       --major-ppm <ppm>        Major-state chemical shift varpi_A in ppm (required with --r1rho)\n\
       --larmor-mhz <mhz>       Heteronuclear Larmor frequency |nu_0| in MHz (required with --r1rho)\n\n\
     Outputs:\n\
       --json-out <fit.json>    Optional JSON output path\n\
       --out | -o <fit.tsv>     Output fitted CPMG or R1rho TSV table"
}

pub fn csp_usage() -> &'static str {
    "ferric csp — Hungarian HSQC/HMQC titration peak tracking, Williamson 2-sigma pocket & quadratic 1:1 Kd\n\
     Usage:\n\
       ferric csp (--apo-peaks <apo.tsv> | --star <apo.str>) --step <[L]_uM[:[P]_uM]=path> ... [--out <csp.tsv>] [options]\n\
       ferric csp --trajectories <tracked.tsv> [--out <csp.tsv>] [options]\n\n\
     Inputs & Titration Tracking:\n\
       --apo-peaks <apo.tsv>    Assigned apo crosspeak TSV (seq_id, res_name, h_ppm, n_or_c_ppm)\n\
       --star | --apo-star <p>  Assigned apo NMR-STAR (.str) chemical shift list\n\
       --step <SPEC>            Repeatable titration step `<L_uM>=<peaks_or_2d>` or `<L_uM>:<P_uM>=<peaks_or_2d>`\n\
       --trajectories <tsv>     Pre-tracked CSP trajectory TSV (bypasses peak matching)\n\
       --experiment <kind>      hsqc (1H-15N amide, alpha_N=0.14/0.20 Gly) | hmqc (1H-13C methyl, alpha_C=0.30) [default: hsqc]\n\
       --protein-um <uM>        Initial apo total protein concentration [P]_0 in uM [default: 100.0]\n\
       --ligand-stock-um <uM>   Optional ligand stock concentration [L]_stock in uM for automatic dilution correction\n\
       --max-step-csp <ppm>     Maximum single-step shift jump before slack broadening [default: 0.45]\n\
       --snr <f32>              2D peak-picker SNR threshold when --step points to a 2D spectrum [default: 5.0]\n\n\
     Isotherm Fitting & Outputs:\n\
       --csp-sigma <ppm>        1-sigma combined CSP measurement uncertainty in ppm [default: 0.004]\n\
       --mc-trials <N>          Parametric Monte Carlo replicates for 95% Kd CIs [default: 400]\n\
       --seed <u64>             Deterministic PRNG seed [default: 20261306]\n\
       --trajectories-out <p>   Optional path to export tracked per-step peak trajectories TSV\n\
       --json-out <csp.json>    Optional JSON output path\n\
       --out | -o <csp.tsv>     Output per-residue & global tight-binding Kd TSV table"
}

fn format_cest_3state_tsv(results: &[Cest3StateFitResult]) -> String {
    let mut out = String::from(
        "seq_id\tres_name\tp_b\tp_c\tk_ex_ab_s_inv\tk_ex_bc_s_inv\tk_ex_ac_s_inv\tmajor_ppm\tdw_ab_ppm\tdw_ac_ppm\tr1a_s_inv\tr2a_s_inv\tr2b_s_inv\tr2c_s_inv\tchi2\treduced_chi2\tdof\tconverged\n",
    );
    for r in results {
        out.push_str(&format!(
            "{}\t{}\t{:.6}\t{:.6}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{}\t{}\n",
            r.seq_id,
            r.res_name,
            r.p_b,
            r.p_c,
            r.k_ex_ab_s_inv,
            r.k_ex_bc_s_inv,
            r.k_ex_ac_s_inv,
            r.major_ppm,
            r.dw_ab_ppm,
            r.dw_ac_ppm,
            r.r1a_s_inv,
            r.r2a_s_inv,
            r.r2b_s_inv,
            r.r2c_s_inv,
            r.chi2,
            r.reduced_chi2,
            r.dof,
            r.converged,
        ));
    }
    out
}

pub fn run_cest_cli(args: &[String]) -> Result<(), String> {
    if args.is_empty() || args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{}", cest_usage());
        return Ok(());
    }

    let mut in_paths: Vec<PathBuf> = Vec::new();
    let mut states: usize = 2;
    let mut triangular = false;
    let mut b1_quad: usize = 5;
    let mut engine_choice = ExchangeEngineChoice::Cpu;
    let mut max_iters: usize = 45;
    let mut refine_major = true;
    let mut chi2_surface = false;
    let mut json_out: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--in" | "--profiles" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for --in")?;
                in_paths.push(PathBuf::from(v));
            }
            "--states" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for --states")?;
                states = v
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --states {v:?}: {e}"))?;
                if states != 2 && states != 3 {
                    return Err(format!("invalid --states {states}: expected 2 or 3"));
                }
            }
            "--triangular" => {
                triangular = true;
            }
            "--b1-quad" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for --b1-quad")?;
                b1_quad = v
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --b1-quad {v:?}: {e}"))?;
                if !matches!(b1_quad, 1 | 3 | 5 | 7 | 9 | 11 | 15) {
                    return Err(format!(
                        "unsupported --b1-quad {b1_quad}: expected one of 1, 3, 5, 7, 9, 11, or 15"
                    ));
                }
            }
            "--engine" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for --engine")?;
                engine_choice = ExchangeEngineChoice::parse(v)?;
            }
            "--max-iters" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for --max-iters")?;
                max_iters = v
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --max-iters {v:?}: {e}"))?;
                if !(1..=500).contains(&max_iters) {
                    return Err(format!("--max-iters {max_iters} out of bounds (1..=500)"));
                }
            }
            "--no-refine-major" => {
                refine_major = false;
            }
            "--chi2-surface" => {
                chi2_surface = true;
            }
            "--json-out" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for --json-out")?;
                json_out = Some(PathBuf::from(v));
            }
            "--out" | "-o" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for --out")?;
                out_path = Some(PathBuf::from(v));
            }
            other => {
                return Err(format!("unknown cest flag {other:?}\n\n{}", cest_usage()));
            }
        }
        i += 1;
    }

    if in_paths.is_empty() {
        return Err("ferric cest requires at least one --in <profiles.tsv>".to_string());
    }

    let mut profiles: Vec<CestProfileDataset> = Vec::new();
    for p in &in_paths {
        let raw = fs::read_to_string(p)
            .map_err(|e| format!("failed to read CEST profile TSV '{}': {e}", p.display()))?;
        let mut parsed = read_cest_profiles_tsv(&raw)
            .map_err(|e| format!("failed to parse '{}': {e}", p.display()))?;
        profiles.append(&mut parsed);
    }
    if profiles.is_empty() {
        return Err("no CEST/DEST profiles found in input TSV(s)".to_string());
    }

    let t0 = Instant::now();
    if states == 3 {
        if engine_choice == ExchangeEngineChoice::Gpu {
            return Err(
                "--states 3 optimization currently runs on CPU f64; omit --engine or use --engine cpu|auto"
                    .to_string(),
            );
        }
        let mut by_res: BTreeMap<i32, Vec<CestProfileDataset>> = BTreeMap::new();
        for p in profiles {
            by_res.entry(p.seq_id).or_default().push(p);
        }
        let mut results = Vec::with_capacity(by_res.len());
        for (_, res_profs) in by_res {
            results.push(fit_cest_3state_profile_cpu(
                &res_profs, b1_quad, triangular,
            )?);
        }
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let tsv = format_cest_3state_tsv(&results);
        if let Some(ref p) = out_path {
            fs::write(p, &tsv).map_err(|e| format!("failed to write '{}': {e}", p.display()))?;
        }
        if let Some(ref jp) = json_out {
            let js = format_cest_3state_fit_json(&results)?;
            fs::write(jp, js).map_err(|e| format!("failed to write '{}': {e}", jp.display()))?;
        }
        println!(
            "CEST 3-state ({}) fit complete across {} residue(s) in {:.1} ms",
            if triangular { "triangular" } else { "linear" },
            results.len(),
            elapsed_ms
        );
        if out_path.is_none() {
            print!("{tsv}");
        }
        return Ok(());
    }

    let cfg = CestFitConfig {
        n_b1_quad: b1_quad,
        max_lm_iters: max_iters,
        refine_major_ppm: refine_major,
        compute_chi2_surface: chi2_surface,
        ..CestFitConfig::default()
    };

    let (fit, backend_label) = match engine_choice {
        ExchangeEngineChoice::Cpu => (
            fit_cest_global_cpu(&profiles, &cfg)?,
            "CPU (f64)".to_string(),
        ),
        ExchangeEngineChoice::Gpu => {
            let gpu = GpuEngine::new_headless()?;
            let label = format!("WebGPU ({}, {})", gpu.adapter_name, gpu.backend_name);
            (fit_cest_global_gpu(&gpu, &profiles, &cfg)?, label)
        }
        ExchangeEngineChoice::Auto => match GpuEngine::new_headless() {
            Ok(gpu) => {
                let label = format!("WebGPU ({}, {})", gpu.adapter_name, gpu.backend_name);
                match fit_cest_global_gpu(&gpu, &profiles, &cfg) {
                    Ok(res) => (res, label),
                    Err(_) => (
                        fit_cest_global_cpu(&profiles, &cfg)?,
                        "CPU (f64 fallback)".to_string(),
                    ),
                }
            }
            Err(_) => (
                fit_cest_global_cpu(&profiles, &cfg)?,
                "CPU (f64)".to_string(),
            ),
        },
    };
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let tsv = format_cest_fit_tsv(&fit);
    if let Some(ref p) = out_path {
        fs::write(p, &tsv).map_err(|e| format!("failed to write '{}': {e}", p.display()))?;
    }
    if let Some(ref jp) = json_out {
        let js = format_cest_fit_json(&fit)?;
        fs::write(jp, js).map_err(|e| format!("failed to write '{}': {e}", jp.display()))?;
    }

    println!(
        "CEST 2-state global fit [{backend_label}] ({} residues, {:.1} ms): p_B = {:.3}% +- {:.3}%, k_ex = {:.2} +- {:.2} s^-1, chi^2_nu = {:.3}",
        fit.residues.len(),
        elapsed_ms,
        fit.p_b * 100.0,
        fit.p_b_err * 100.0,
        fit.k_ex_s_inv,
        fit.k_ex_err,
        fit.reduced_chi2,
    );
    if out_path.is_none() {
        print!("{tsv}");
    }
    Ok(())
}

fn parse_raw_cpmg_planes_tsv(
    tsv: &str,
    sigma_floor_s_inv: f64,
) -> Result<Vec<CpmgSeriesDataset>, String> {
    type Key = (i32, String, u64, u64);
    let mut groups: BTreeMap<Key, (f64, f64, Vec<RawCpmgIntensityPlane>)> = BTreeMap::new();

    for (line_no, raw) in tsv.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols[0] == "seq_id" {
            continue;
        }
        if cols.len() < 6 {
            return Err(format!(
                "line {}: expected >= 6 TSV columns (seq_id, res_name, larmor_mhz, t_relax_s, nu_cpmg_hz, intensity[, baseline_noise]), got {}",
                line_no + 1,
                cols.len()
            ));
        }
        let seq_id: i32 = cols[0]
            .parse()
            .map_err(|e| format!("line {}: invalid seq_id: {e}", line_no + 1))?;
        let res_name = cols[1].trim().to_string();
        let larmor_mhz: f64 = cols[2]
            .parse()
            .map_err(|e| format!("line {}: invalid larmor_mhz: {e}", line_no + 1))?;
        let t_relax_s: f64 = cols[3]
            .parse()
            .map_err(|e| format!("line {}: invalid t_relax_s: {e}", line_no + 1))?;
        let nu_cpmg_hz: f64 = cols[4]
            .parse()
            .map_err(|e| format!("line {}: invalid nu_cpmg_hz: {e}", line_no + 1))?;
        let intensity: f64 = cols[5]
            .parse()
            .map_err(|e| format!("line {}: invalid intensity: {e}", line_no + 1))?;
        let baseline_noise =
            if let Some(c6) = cols.get(6) {
                let s = c6.trim();
                if s.is_empty() || s.eq_ignore_ascii_case("na") {
                    None
                } else {
                    Some(s.parse::<f64>().map_err(|e| {
                        format!("line {}: invalid baseline_noise: {e}", line_no + 1)
                    })?)
                }
            } else {
                None
            };

        let key = (
            seq_id,
            res_name.clone(),
            (larmor_mhz * 1e5).round() as u64,
            (t_relax_s * 1e6).round() as u64,
        );
        let entry = groups
            .entry(key)
            .or_insert_with(|| (larmor_mhz, t_relax_s, Vec::new()));
        entry.2.push(RawCpmgIntensityPlane {
            nu_cpmg_hz,
            intensity,
            baseline_noise,
        });
    }

    let mut series = Vec::with_capacity(groups.len());
    for ((seq_id, res_name, _, _), (larmor_mhz, t_relax_s, planes)) in groups {
        series.push(compute_r2eff_from_intensities(
            seq_id,
            res_name,
            larmor_mhz,
            t_relax_s,
            &planes,
            sigma_floor_s_inv,
        )?);
    }
    Ok(series)
}

fn parse_r1rho_tsv(
    tsv: &str,
    major_ppm: f64,
    larmor_mhz: f64,
) -> Result<Vec<R1RhoOffResPoint>, String> {
    let mut pts = Vec::new();
    let scale = TAU * larmor_mhz;
    for (line_no, raw) in tsv.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols[0] == "carrier_ppm" || cols[0] == "seq_id" {
            continue;
        }
        let base = if cols.len() >= 6 {
            2usize
        } else if cols.len() >= 4 {
            0usize
        } else {
            return Err(format!(
                "line {}: expected 4 or 6 TSV columns in R1rho table, got {}",
                line_no + 1,
                cols.len()
            ));
        };
        let carrier_ppm: f64 = cols[base]
            .parse()
            .map_err(|e| format!("line {}: invalid carrier_ppm: {e}", line_no + 1))?;
        let spin_lock_hz: f64 = cols[base + 1]
            .parse()
            .map_err(|e| format!("line {}: invalid spin_lock_hz: {e}", line_no + 1))?;
        let r1rho_s_inv: f64 = cols[base + 2]
            .parse()
            .map_err(|e| format!("line {}: invalid r1rho_s_inv: {e}", line_no + 1))?;
        let r1rho_err: f64 = cols[base + 3]
            .parse()
            .map_err(|e| format!("line {}: invalid r1rho_err: {e}", line_no + 1))?;

        let om_a = scale * (major_ppm - carrier_ppm);
        let w1 = TAU * spin_lock_hz;
        let omega_eff_rad_s = om_a.hypot(w1);
        let theta_rad = w1.atan2(om_a);
        pts.push(R1RhoOffResPoint {
            carrier_ppm,
            spin_lock_hz,
            omega_eff_rad_s,
            theta_rad,
            r1rho_s_inv,
            r1rho_err,
        });
    }
    Ok(pts)
}

fn format_r1rho_fit_tsv(
    fit: &R1RhoFitResult,
    major_ppm: f64,
    larmor_mhz: f64,
    n_points: usize,
) -> String {
    let dof = n_points.saturating_sub(5).max(1);
    let red_chi2 = fit.chi2 / (dof as f64);
    format!(
        "# model=trott_palmer_2002\tmajor_ppm={major_ppm:.4}\tlarmor_mhz={larmor_mhz:.5}\n\
         p_b\tk_ex_s_inv\tdw_ab_ppm\tr1_s_inv\tr20_s_inv\tchi2\treduced_chi2\tn_points\n\
         {:.6}\t{:.4}\t{:+.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{}\n",
        fit.p_b,
        fit.k_ex_s_inv,
        fit.dw_ab_ppm,
        fit.r1_s_inv,
        fit.r20_s_inv,
        fit.chi2,
        red_chi2,
        n_points,
    )
}

pub fn run_cpmg_cli(args: &[String]) -> Result<(), String> {
    if args.is_empty() || args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{}", cpmg_usage());
        return Ok(());
    }

    let mut series_paths: Vec<PathBuf> = Vec::new();
    let mut raw_plane_paths: Vec<PathBuf> = Vec::new();
    let mut series_out: Option<PathBuf> = None;
    let mut sigma_floor: f64 = 0.15;
    let mut model = CpmgModel::Baldwin2014;
    let mut engine_choice = ExchangeEngineChoice::Cpu;
    let mut min_rex: f64 = 2.0;
    let mut min_daic: f64 = 2.0;
    let mut max_fp: f64 = 0.05;
    let mut filter_gate = true;

    let mut r1rho_path: Option<PathBuf> = None;
    let mut major_ppm: Option<f64> = None;
    let mut larmor_mhz: Option<f64> = None;

    let mut json_out: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--in" | "--series" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for --in")?;
                series_paths.push(PathBuf::from(v));
            }
            "--raw-planes" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for --raw-planes")?;
                raw_plane_paths.push(PathBuf::from(v));
            }
            "--series-out" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for --series-out")?;
                series_out = Some(PathBuf::from(v));
            }
            "--sigma-floor" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for --sigma-floor")?;
                sigma_floor = v
                    .parse::<f64>()
                    .map_err(|e| format!("invalid --sigma-floor {v:?}: {e}"))?;
            }
            "--model" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for --model")?;
                model = CpmgModel::from_str_label(v)?;
            }
            "--engine" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for --engine")?;
                engine_choice = ExchangeEngineChoice::parse(v)?;
            }
            "--min-rex" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for --min-rex")?;
                min_rex = v
                    .parse::<f64>()
                    .map_err(|e| format!("invalid --min-rex {v:?}: {e}"))?;
            }
            "--min-daic" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for --min-daic")?;
                min_daic = v
                    .parse::<f64>()
                    .map_err(|e| format!("invalid --min-daic {v:?}: {e}"))?;
            }
            "--max-fp" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for --max-fp")?;
                max_fp = v
                    .parse::<f64>()
                    .map_err(|e| format!("invalid --max-fp {v:?}: {e}"))?;
            }
            "--no-gate" => {
                filter_gate = false;
            }
            "--r1rho" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for --r1rho")?;
                r1rho_path = Some(PathBuf::from(v));
            }
            "--major-ppm" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for --major-ppm")?;
                major_ppm = Some(
                    v.parse::<f64>()
                        .map_err(|e| format!("invalid --major-ppm {v:?}: {e}"))?,
                );
            }
            "--larmor-mhz" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for --larmor-mhz")?;
                larmor_mhz = Some(
                    v.parse::<f64>()
                        .map_err(|e| format!("invalid --larmor-mhz {v:?}: {e}"))?,
                );
            }
            "--json-out" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for --json-out")?;
                json_out = Some(PathBuf::from(v));
            }
            "--out" | "-o" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for --out")?;
                out_path = Some(PathBuf::from(v));
            }
            other => {
                return Err(format!("unknown cpmg flag {other:?}\n\n{}", cpmg_usage()));
            }
        }
        i += 1;
    }

    if let Some(ref r1p) = r1rho_path {
        let m_ppm = major_ppm.ok_or("--r1rho requires --major-ppm <ppm>")?;
        let l_mhz = larmor_mhz.ok_or("--r1rho requires --larmor-mhz <mhz>")?;
        let raw = fs::read_to_string(r1p)
            .map_err(|e| format!("failed to read R1rho TSV '{}': {e}", r1p.display()))?;
        let pts = parse_r1rho_tsv(&raw, m_ppm, l_mhz)?;
        let t0 = Instant::now();
        let fit = fit_r1rho_trott_palmer_cpu(m_ppm, l_mhz, &pts)?;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let tsv = format_r1rho_fit_tsv(&fit, m_ppm, l_mhz, pts.len());
        if let Some(ref p) = out_path {
            fs::write(p, &tsv).map_err(|e| format!("failed to write '{}': {e}", p.display()))?;
        }
        if let Some(ref jp) = json_out {
            let js = format_r1rho_fit_json(&fit)?;
            fs::write(jp, js).map_err(|e| format!("failed to write '{}': {e}", jp.display()))?;
        }
        println!(
            "Off-resonance R1rho Trott-Palmer (2002) fit ({} points, {:.1} ms): p_B = {:.3}%, k_ex = {:.1} s^-1, signed dw_AB = {:+.3} ppm",
            pts.len(),
            elapsed_ms,
            fit.p_b * 100.0,
            fit.k_ex_s_inv,
            fit.dw_ab_ppm,
        );
        if out_path.is_none() {
            print!("{tsv}");
        }
        return Ok(());
    }

    if series_paths.is_empty() && raw_plane_paths.is_empty() {
        return Err(
            "ferric cpmg requires --in <cpmg.tsv>, --raw-planes <planes.tsv>, or --r1rho <r1rho.tsv>"
                .to_string(),
        );
    }

    let mut all_series: Vec<CpmgSeriesDataset> = Vec::new();
    for p in &series_paths {
        let raw = fs::read_to_string(p)
            .map_err(|e| format!("failed to read CPMG series TSV '{}': {e}", p.display()))?;
        let mut parsed = read_cpmg_series_tsv(&raw)
            .map_err(|e| format!("failed to parse '{}': {e}", p.display()))?;
        all_series.append(&mut parsed);
    }
    for p in &raw_plane_paths {
        let raw = fs::read_to_string(p)
            .map_err(|e| format!("failed to read raw CPMG planes TSV '{}': {e}", p.display()))?;
        let mut computed = parse_raw_cpmg_planes_tsv(&raw, sigma_floor)?;
        all_series.append(&mut computed);
    }
    if all_series.is_empty() {
        return Err("no CPMG dispersion curves found in input TSV(s)".to_string());
    }

    if let Some(ref so) = series_out {
        let stsv = write_cpmg_series_tsv(&all_series);
        fs::write(so, stsv).map_err(|e| format!("failed to write '{}': {e}", so.display()))?;
    }

    let cfg = CpmgFitConfig {
        model,
        gate: ExchangeGateConfig {
            min_delta_r2_eff_s_inv: min_rex,
            min_delta_aic: min_daic,
            max_f_pvalue: max_fp,
        },
        filter_by_exchange_gate: filter_gate,
        ..CpmgFitConfig::default()
    };

    let t0 = Instant::now();
    let (fit, backend_label) = match engine_choice {
        ExchangeEngineChoice::Cpu => (
            fit_cpmg_global_cpu(&all_series, &cfg)?,
            "CPU (f64)".to_string(),
        ),
        ExchangeEngineChoice::Gpu => {
            if model == CpmgModel::Numerical2x2 {
                return Err(
                    "--model numerical2x2 is CPU-only; use --model baldwin2014|cr72|lm63 with --engine gpu"
                        .to_string(),
                );
            }
            let gpu = GpuEngine::new_headless()?;
            let label = format!("WebGPU ({}, {})", gpu.adapter_name, gpu.backend_name);
            (fit_cpmg_global_gpu(&gpu, &all_series, &cfg)?, label)
        }
        ExchangeEngineChoice::Auto => {
            if model == CpmgModel::Numerical2x2 {
                (
                    fit_cpmg_global_cpu(&all_series, &cfg)?,
                    "CPU (f64)".to_string(),
                )
            } else {
                match GpuEngine::new_headless() {
                    Ok(gpu) => {
                        let label = format!("WebGPU ({}, {})", gpu.adapter_name, gpu.backend_name);
                        match fit_cpmg_global_gpu(&gpu, &all_series, &cfg) {
                            Ok(res) => (res, label),
                            Err(_) => (
                                fit_cpmg_global_cpu(&all_series, &cfg)?,
                                "CPU (f64 fallback)".to_string(),
                            ),
                        }
                    }
                    Err(_) => (
                        fit_cpmg_global_cpu(&all_series, &cfg)?,
                        "CPU (f64)".to_string(),
                    ),
                }
            }
        }
    };
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let tsv = format_cpmg_fit_tsv(&fit);
    if let Some(ref p) = out_path {
        fs::write(p, &tsv).map_err(|e| format!("failed to write '{}': {e}", p.display()))?;
    }
    if let Some(ref jp) = json_out {
        let js = format_cpmg_fit_json(&fit)?;
        fs::write(jp, js).map_err(|e| format!("failed to write '{}': {e}", jp.display()))?;
    }

    println!(
        "CPMG global fit [{} | {backend_label}] ({} exchanging residues, {:.1} ms): p_B = {:.3}% +- {:.3}%, k_ex = {:.1} +- {:.1} s^-1, chi^2_nu = {:.3}",
        fit.model.as_str(),
        fit.residues.len(),
        elapsed_ms,
        fit.p_b * 100.0,
        fit.p_b_err * 100.0,
        fit.k_ex_s_inv,
        fit.k_ex_err,
        fit.reduced_chi2,
    );
    if out_path.is_none() {
        print!("{tsv}");
    }
    Ok(())
}

fn parse_apo_peaks_tsv(path: &Path) -> Result<Vec<AssignedAmidePeak>, String> {
    let raw = fs::read_to_string(path)
        .map_err(|e| format!("failed to read apo peaks TSV '{}': {e}", path.display()))?;
    let mut out = Vec::new();
    for (line_no, raw_line) in raw.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols[0] == "seq_id" || cols[0] == "seq_index" {
            continue;
        }
        let (seq_index, seq_id, res_name, h_ppm, n_ppm) = if cols.len() >= 5 {
            let idx: usize = cols[0]
                .parse()
                .map_err(|e| format!("line {}: invalid seq_index: {e}", line_no + 1))?;
            let sid: i32 = cols[1]
                .parse()
                .map_err(|e| format!("line {}: invalid seq_id: {e}", line_no + 1))?;
            let rn = cols[2].trim().to_string();
            let h: f64 = cols[3]
                .parse()
                .map_err(|e| format!("line {}: invalid h_ppm: {e}", line_no + 1))?;
            let n: f64 = cols[4]
                .parse()
                .map_err(|e| format!("line {}: invalid n_ppm: {e}", line_no + 1))?;
            (idx, sid, rn, h, n)
        } else if cols.len() >= 4 {
            let sid: i32 = cols[0]
                .parse()
                .map_err(|e| format!("line {}: invalid seq_id: {e}", line_no + 1))?;
            let rn = cols[1].trim().to_string();
            let h: f64 = cols[2]
                .parse()
                .map_err(|e| format!("line {}: invalid h_ppm: {e}", line_no + 1))?;
            let n: f64 = cols[3]
                .parse()
                .map_err(|e| format!("line {}: invalid n_ppm: {e}", line_no + 1))?;
            (out.len(), sid, rn, h, n)
        } else {
            return Err(format!(
                "line {}: expected 4 or 5 TSV columns in apo peaks file, got {}",
                line_no + 1,
                cols.len()
            ));
        };
        out.push(AssignedAmidePeak {
            seq_index,
            seq_id,
            res_name,
            h_ppm,
            n_ppm,
        });
    }
    if out.is_empty() {
        return Err(format!(
            "no assigned apo crosspeaks parsed from '{}'",
            path.display()
        ));
    }
    Ok(out)
}

fn load_apo_peaks_from_star(
    path: &Path,
    exp_kind: TitrationExperimentKind,
) -> Result<Vec<AssignedAmidePeak>, String> {
    match exp_kind {
        TitrationExperimentKind::AmideHsqc => {
            let r = read_star(path, None)?;
            let mut out = Vec::new();
            for (k, rn) in r.sequence.iter().enumerate() {
                if rn.eq_ignore_ascii_case("PRO") {
                    continue;
                }
                let h = r.shifts.get(&(k, "H".to_string())).copied();
                let n = r.shifts.get(&(k, "N".to_string())).copied();
                if let (Some(h_ppm), Some(n_ppm)) = (h, n) {
                    let seq_id = r.seq_ids.get(k).copied().unwrap_or((k + 1) as i32);
                    out.push(AssignedAmidePeak {
                        seq_index: k,
                        seq_id,
                        res_name: rn.clone(),
                        h_ppm,
                        n_ppm,
                    });
                }
            }
            if out.is_empty() {
                return Err(format!(
                    "no assigned (H, N) amide pairs found in '{}'",
                    path.display()
                ));
            }
            Ok(out)
        }
        TitrationExperimentKind::MethylHmqc => {
            let r = read_star_all_atoms(path, None)?;
            const METHYL_ATOM_PAIRS: &[(&str, &str)] = &[
                ("HD1", "CD1"),
                ("HD2", "CD2"),
                ("HG1", "CG1"),
                ("HG2", "CG2"),
                ("HB", "CB"),
                ("HE", "CE"),
            ];
            let mut out = Vec::new();
            for (k, rn) in r.sequence.iter().enumerate() {
                let base_sid = r.seq_ids.get(k).copied().unwrap_or((k + 1) as i32);
                for &(h_atom, c_atom) in METHYL_ATOM_PAIRS {
                    let h = r.shifts.get(&(k, h_atom.to_string())).copied();
                    let c = r.shifts.get(&(k, c_atom.to_string())).copied();
                    if let (Some(h_ppm), Some(c_ppm)) = (h, c) {
                        out.push(AssignedAmidePeak {
                            seq_index: k,
                            seq_id: base_sid,
                            res_name: rn.clone(),
                            h_ppm,
                            n_ppm: c_ppm,
                        });
                        break;
                    }
                }
            }
            if out.is_empty() {
                return Err(format!(
                    "no assigned (1H, 13C) methyl pairs found in '{}'",
                    path.display()
                ));
            }
            Ok(out)
        }
    }
}

fn load_spectrum_2d_for_csp(path: &Path) -> Result<Spectrum2D, String> {
    if path.is_dir() {
        let pdata1 = path.join("pdata").join("1");
        if pdata1.join("2rr").is_file() {
            return read_bruker_pdata_2d(&pdata1);
        }
        if path.join("2rr").is_file() {
            return read_bruker_pdata_2d(path);
        }
        return Err(format!(
            "2D titration directory '{}' does not contain Bruker 2rr",
            path.display()
        ));
    }
    if is_ucsf_file(path) {
        read_ucsf_2d(path)
    } else {
        read_nmrpipe_2d(path)
    }
}

fn load_unassigned_step_peaks(
    path: &Path,
    exp_kind: TitrationExperimentKind,
    snr: f32,
) -> Result<Vec<UnassignedHsqcPeak>, String> {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let is_spectrum = path.is_dir()
        || matches!(ext.as_str(), "ucsf" | "ft2" | "pipe" | "2rr")
        || is_ucsf_file(path);

    if is_spectrum {
        let spec = load_spectrum_2d_for_csp(path)?;
        let pcfg = PickerConfig {
            snr_threshold: snr,
            ..PickerConfig::default()
        };
        let picked = match exp_kind {
            TitrationExperimentKind::AmideHsqc => pick_hsqc_2d(&spec, &pcfg),
            TitrationExperimentKind::MethylHmqc => pick_c13_hsqc_2d(&spec, &pcfg),
        };
        return Ok(picked
            .into_iter()
            .map(|pk| UnassignedHsqcPeak {
                h_ppm: pk.h,
                n_ppm: match exp_kind {
                    TitrationExperimentKind::AmideHsqc => pk.n,
                    TitrationExperimentKind::MethylHmqc => pk.c.unwrap_or(pk.n),
                },
            })
            .collect());
    }

    let raw = fs::read_to_string(path).map_err(|e| {
        format!(
            "failed to read titration step file '{}': {e}",
            path.display()
        )
    })?;

    // Try standard `ferric` peak TSV first if it carries an `exp` header or known experiment column
    if let Ok(std_peaks) = read_peaks(path) {
        if !std_peaks.is_empty() {
            return Ok(std_peaks
                .into_iter()
                .map(|pk| UnassignedHsqcPeak {
                    h_ppm: pk.h,
                    n_ppm: match exp_kind {
                        TitrationExperimentKind::AmideHsqc => pk.n,
                        TitrationExperimentKind::MethylHmqc => pk.c.unwrap_or(pk.n),
                    },
                })
                .collect());
        }
    }

    // Fallback: simple 2-column (`h_ppm \t n_or_c_ppm`) or 4-column (`seq_id \t res \t h \t n`) TSV
    let mut peaks = Vec::new();
    for (line_no, raw_line) in raw.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols[0] == "h_ppm" || cols[0] == "h" || cols[0] == "seq_id" {
            continue;
        }
        let (h_s, n_s) = if cols.len() >= 4 {
            (cols[2], cols[3])
        } else if cols.len() >= 2 {
            (cols[0], cols[1])
        } else {
            return Err(format!(
                "{}:{}: expected >= 2 TSV columns for 2D peak coordinates",
                path.display(),
                line_no + 1
            ));
        };
        let h_ppm: f64 = h_s
            .parse()
            .map_err(|e| format!("{}:{}: invalid h_ppm: {e}", path.display(), line_no + 1))?;
        let n_ppm: f64 = n_s.parse().map_err(|e| {
            format!(
                "{}:{}: invalid n_or_c_ppm: {e}",
                path.display(),
                line_no + 1
            )
        })?;
        peaks.push(UnassignedHsqcPeak { h_ppm, n_ppm });
    }
    Ok(peaks)
}

fn parse_step_spec(
    spec: &str,
    default_protein_um: f64,
    ligand_stock_um: Option<f64>,
    exp_kind: TitrationExperimentKind,
    snr: f32,
) -> Result<UnassignedTitrationStep, String> {
    let (lhs, rhs) = spec.split_once('=').ok_or_else(|| {
        format!("invalid --step {spec:?}: expected `<L_uM>=<path>` or `<L_uM>:<P_uM>=<path>`")
    })?;
    let (ligand_conc_um, protein_conc_um) = if let Some((l_str, p_str)) = lhs.split_once(':') {
        let l: f64 = l_str
            .trim()
            .parse()
            .map_err(|e| format!("invalid ligand concentration {l_str:?} in --step: {e}"))?;
        let p: f64 = p_str
            .trim()
            .parse()
            .map_err(|e| format!("invalid protein concentration {p_str:?} in --step: {e}"))?;
        (l, p)
    } else {
        let l: f64 = lhs
            .trim()
            .parse()
            .map_err(|e| format!("invalid ligand concentration {lhs:?} in --step: {e}"))?;
        let p = if let Some(stock) = ligand_stock_um {
            diluted_protein_conc_for_target_ligand(default_protein_um, stock, l)?
        } else {
            default_protein_um
        };
        (l, p)
    };
    let path = PathBuf::from(rhs.trim());
    let peaks = load_unassigned_step_peaks(&path, exp_kind, snr)?;
    Ok(UnassignedTitrationStep {
        label: format!("{ligand_conc_um:.2}uM"),
        protein_conc_um,
        ligand_conc_um,
        peaks,
    })
}

pub fn parse_csp_trajectories_tsv(tsv: &str) -> Result<Vec<ResidueCspTrajectory>, String> {
    let mut by_res: BTreeMap<(usize, i32), ResidueCspTrajectory> = BTreeMap::new();
    for (line_no, raw_line) in tsv.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols[0] == "seq_index" {
            continue;
        }
        if cols.len() < 8 {
            return Err(format!(
                "line {}: expected 8 TSV columns in CSP trajectories table, got {}",
                line_no + 1,
                cols.len()
            ));
        }
        let seq_index: usize = cols[0]
            .parse()
            .map_err(|e| format!("line {}: invalid seq_index: {e}", line_no + 1))?;
        let seq_id: i32 = cols[1]
            .parse()
            .map_err(|e| format!("line {}: invalid seq_id: {e}", line_no + 1))?;
        let res_name = cols[2].trim().to_string();
        let protein_conc_um: f64 = cols[3]
            .parse()
            .map_err(|e| format!("line {}: invalid protein_conc_um: {e}", line_no + 1))?;
        let ligand_conc_um: f64 = cols[4]
            .parse()
            .map_err(|e| format!("line {}: invalid ligand_conc_um: {e}", line_no + 1))?;
        let matched_peak =
            if cols[5].eq_ignore_ascii_case("na") || cols[6].eq_ignore_ascii_case("na") {
                None
            } else {
                let h: f64 = cols[5]
                    .parse()
                    .map_err(|e| format!("line {}: invalid h_ppm: {e}", line_no + 1))?;
                let n: f64 = cols[6]
                    .parse()
                    .map_err(|e| format!("line {}: invalid n_ppm: {e}", line_no + 1))?;
                Some((h, n))
            };
        let csp_ppm = if cols[7].eq_ignore_ascii_case("na") {
            None
        } else {
            Some(
                cols[7]
                    .parse::<f64>()
                    .map_err(|e| format!("line {}: invalid csp_ppm: {e}", line_no + 1))?,
            )
        };

        let entry = by_res
            .entry((seq_index, seq_id))
            .or_insert_with(|| ResidueCspTrajectory {
                seq_index,
                seq_id,
                res_name,
                apo_h_ppm: matched_peak.map(|p| p.0).unwrap_or(0.0),
                apo_n_ppm: matched_peak.map(|p| p.1).unwrap_or(0.0),
                points: Vec::new(),
                max_observed_csp_ppm: 0.0,
            });
        if ligand_conc_um <= 1e-12 {
            if let Some((h, n)) = matched_peak {
                entry.apo_h_ppm = h;
                entry.apo_n_ppm = n;
            }
        }
        if let Some(c) = csp_ppm {
            entry.max_observed_csp_ppm = entry.max_observed_csp_ppm.max(c);
        }
        entry.points.push(ResidueTitrationPoint {
            protein_conc_um,
            ligand_conc_um,
            matched_peak,
            csp_ppm,
        });
    }
    if by_res.is_empty() {
        return Err("no CSP trajectories found in TSV".to_string());
    }
    Ok(by_res.into_values().collect())
}

pub fn run_csp_cli(args: &[String]) -> Result<(), String> {
    if args.is_empty() || args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{}", csp_usage());
        return Ok(());
    }

    let mut apo_peaks_path: Option<PathBuf> = None;
    let mut apo_star_path: Option<PathBuf> = None;
    let mut step_specs: Vec<String> = Vec::new();
    let mut trajectories_in: Option<PathBuf> = None;
    let mut exp_kind = TitrationExperimentKind::AmideHsqc;
    let mut protein_um: f64 = 100.0;
    let mut ligand_stock_um: Option<f64> = None;
    let mut max_step_csp: f64 = 0.45;
    let mut snr: f32 = 5.0;
    let mut csp_sigma: f64 = 0.004;
    let mut mc_trials: usize = 400;
    let mut seed: u64 = 20261306;
    let mut trajectories_out: Option<PathBuf> = None;
    let mut json_out: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--apo-peaks" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for --apo-peaks")?;
                apo_peaks_path = Some(PathBuf::from(v));
            }
            "--star" | "--apo-star" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for --star")?;
                apo_star_path = Some(PathBuf::from(v));
            }
            "--step" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for --step")?;
                step_specs.push(v.clone());
            }
            "--trajectories" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for --trajectories")?;
                trajectories_in = Some(PathBuf::from(v));
            }
            "--experiment" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for --experiment")?;
                exp_kind = TitrationExperimentKind::parse(v)?;
            }
            "--protein-um" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for --protein-um")?;
                protein_um = v
                    .parse::<f64>()
                    .map_err(|e| format!("invalid --protein-um {v:?}: {e}"))?;
                if !protein_um.is_finite() || protein_um <= 0.0 {
                    return Err("--protein-um must be > 0".to_string());
                }
            }
            "--ligand-stock-um" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for --ligand-stock-um")?;
                let s = v
                    .parse::<f64>()
                    .map_err(|e| format!("invalid --ligand-stock-um {v:?}: {e}"))?;
                if !s.is_finite() || s <= 0.0 {
                    return Err("--ligand-stock-um must be > 0".to_string());
                }
                ligand_stock_um = Some(s);
            }
            "--max-step-csp" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for --max-step-csp")?;
                max_step_csp = v
                    .parse::<f64>()
                    .map_err(|e| format!("invalid --max-step-csp {v:?}: {e}"))?;
            }
            "--snr" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for --snr")?;
                snr = v
                    .parse::<f32>()
                    .map_err(|e| format!("invalid --snr {v:?}: {e}"))?;
            }
            "--csp-sigma" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for --csp-sigma")?;
                csp_sigma = v
                    .parse::<f64>()
                    .map_err(|e| format!("invalid --csp-sigma {v:?}: {e}"))?;
            }
            "--mc-trials" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for --mc-trials")?;
                mc_trials = v
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --mc-trials {v:?}: {e}"))?;
            }
            "--seed" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for --seed")?;
                seed = v
                    .parse::<u64>()
                    .map_err(|e| format!("invalid --seed {v:?}: {e}"))?;
            }
            "--trajectories-out" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for --trajectories-out")?;
                trajectories_out = Some(PathBuf::from(v));
            }
            "--json-out" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for --json-out")?;
                json_out = Some(PathBuf::from(v));
            }
            "--out" | "-o" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for --out")?;
                out_path = Some(PathBuf::from(v));
            }
            other => {
                return Err(format!("unknown csp flag {other:?}\n\n{}", csp_usage()));
            }
        }
        i += 1;
    }

    let cfg = CspAnalysisConfig {
        max_step_csp_ppm: max_step_csp,
        csp_sigma_ppm: csp_sigma,
        n_mc: mc_trials,
        mc_seed: seed,
        ..CspAnalysisConfig::default()
    };

    let t0 = Instant::now();
    let trajectories = if let Some(ref tr_p) = trajectories_in {
        let raw = fs::read_to_string(tr_p)
            .map_err(|e| format!("failed to read trajectories TSV '{}': {e}", tr_p.display()))?;
        parse_csp_trajectories_tsv(&raw)?
    } else {
        let apo_peaks = if let Some(ref ap) = apo_peaks_path {
            parse_apo_peaks_tsv(ap)?
        } else if let Some(ref sp) = apo_star_path {
            load_apo_peaks_from_star(sp, exp_kind)?
        } else {
            return Err(
                "ferric csp requires --apo-peaks <apo.tsv>, --star <apo.str>, or --trajectories <traj.tsv>"
                    .to_string(),
            );
        };
        if step_specs.is_empty() {
            return Err(
                "ferric csp requires at least 3 --step <[L]_uM=path> titration points".to_string(),
            );
        }
        let mut steps = Vec::with_capacity(step_specs.len() + 1);
        steps.push(UnassignedTitrationStep {
            label: "0.00uM".to_string(),
            protein_conc_um: protein_um,
            ligand_conc_um: 0.0,
            peaks: Vec::new(),
        });
        for spec in &step_specs {
            steps.push(parse_step_spec(
                spec,
                protein_um,
                ligand_stock_um,
                exp_kind,
                snr,
            )?);
        }
        match exp_kind {
            TitrationExperimentKind::AmideHsqc => {
                track_titration_trajectories(&apo_peaks, &steps, &cfg)?
            }
            TitrationExperimentKind::MethylHmqc => {
                track_methyl_hmqc_trajectories(&apo_peaks, &steps, &cfg)?
            }
        }
    };

    if let Some(ref tp) = trajectories_out {
        let ttsv = write_csp_trajectories_tsv(&trajectories);
        fs::write(tp, ttsv).map_err(|e| format!("failed to write '{}': {e}", tp.display()))?;
    }

    let fit = fit_csp_global_kd(&trajectories, &cfg)?;
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let tsv = format_csp_tsv(&fit);
    if let Some(ref p) = out_path {
        fs::write(p, &tsv).map_err(|e| format!("failed to write '{}': {e}", p.display()))?;
    }
    if let Some(ref jp) = json_out {
        let js = format_csp_json(&fit)?;
        fs::write(jp, js).map_err(|e| format!("failed to write '{}': {e}", jp.display()))?;
    }

    println!(
        "CSP titration global 1:1 tight-binding fit ({} residues, {} >=2sigma pocket sites, {:.1} ms): Kd = {:.3} +- {:.3} uM (95% CI [{:.3}, {:.3}] uM; naive MM = {:.3} uM)",
        fit.residue_fits.len(),
        fit.thresholds.significant_seq_ids.len(),
        elapsed_ms,
        fit.global_kd_um,
        fit.global_kd_sd_um,
        fit.global_kd_ci95_low_um,
        fit.global_kd_ci95_high_um,
        fit.global_naive_mm_kd_um,
    );
    if out_path.is_none() {
        print!("{tsv}");
    }
    Ok(())
}

pub fn exchange_state_usage() -> &'static str {
    "ferric exchange-state — Bayesian excited-state (CEST / CPMG / R1rho) structural hypothesis engine\n\
     Usage:\n\
       ferric exchange-state --shifts <ground.str|nef> --structure <ground.cif|pdb> --exchange <cest_or_cpmg.tsv> [--alt-structures <m1.cif,m2.pdb>] [--ph 6.5] [--temp-k 298.15] [-o <excited_state.json|tsv>]\n\n\
     Inputs & Biophysical Hypotheses (H0..H4):\n\
       --shifts | --star <p>       Ground-state assigned chemical shifts (.str or .nef)\n\
       --structure | --pdb <p>     Ground-state 3D structure coordinates (.cif, .mmcif, or .pdb)\n\
       --exchange | --in <tsv>     CEST (signed), R1rho (signed), or CPMG (unsigned |dw|) TSV (repeatable; duplicate (chain, seq_id, atom) rows rejected)\n\
       --alt-structures <list>     Comma-separated candidate alternative 3D conformers (.cif/.pdb) for H4\n\
       --ph <f64>                  Sample pH for ncIDP coil & His titration baselines [default: 6.5]\n\
       --temp-k <f64>              Sample temperature in Kelvin [default: 298.15]\n\
       --exclude-pdb <ids>         Comma-separated PDB IDs excluded by Layer 1 of HomologFirewall\n\
       --exclude-bmrb <ids>        Comma-separated BMRB IDs excluded by Layer 1 of HomologFirewall\n\
       --exclude-topology <id>     CATH/ECOD fold topology excluded by Layer 2 of HomologFirewall\n\
       --force-bfactor-plddt       Treat experimental PDB/CIF B_iso column as pLDDT\n\
       --json-out <out.json>       Optional explicit JSON output path\n\
       --out | -o <out.json|tsv>   Output hypothesis ranking report (.json or .tsv)\n\n\
     Scope & Statistical Notes:\n\
       Models 100%-switched endpoint differential shifts Delta_varpi_AB (no fractional partial-unfolding\n\
       or population scaling parameter inside mu_Delta). Marginalizes over discrete candidates (ln mean L_m),\n\
       counts n = distinct residues in BIC, and requires >= 4 distinct residues to commit."
}

fn split_comma_items(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

pub fn run_exchange_state_cli(args: &[String]) -> Result<(), String> {
    if args.is_empty() || args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{}", exchange_state_usage());
        return Ok(());
    }

    let mut shifts_path: Option<PathBuf> = None;
    let mut structure_path: Option<PathBuf> = None;
    let mut exchange_paths: Vec<PathBuf> = Vec::new();
    let mut alt_paths: Vec<PathBuf> = Vec::new();
    let mut sample_ph: f64 = 6.5;
    let mut temp_k: f64 = 298.15;
    let mut exclude_pdb: Vec<String> = Vec::new();
    let mut exclude_bmrb: Vec<String> = Vec::new();
    let mut exclude_topology: Option<String> = None;
    let mut force_bfactor_plddt = false;
    let mut json_out: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--shifts" | "--star" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or("missing value for --shifts <ground.str|nef>")?;
                shifts_path = Some(PathBuf::from(v));
            }
            "--structure" | "--pdb" | "--cif" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or("missing value for --structure <ground.cif|pdb>")?;
                structure_path = Some(PathBuf::from(v));
            }
            "--exchange" | "--in" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or("missing value for --exchange <cest_or_cpmg.tsv>")?;
                exchange_paths.push(PathBuf::from(v));
            }
            "--alt-structures" | "--alt-structure" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or("missing value for --alt-structures <m1.cif,m2.pdb>")?;
                for item in split_comma_items(v) {
                    alt_paths.push(PathBuf::from(item));
                }
            }
            "--ph" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for --ph")?;
                sample_ph = v
                    .parse::<f64>()
                    .map_err(|e| format!("invalid --ph {v:?}: {e}"))?;
                if !sample_ph.is_finite() || !(0.0..=14.0).contains(&sample_ph) {
                    return Err(format!("--ph {sample_ph} out of bounds [0.0, 14.0]"));
                }
            }
            "--temp-k" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for --temp-k")?;
                temp_k = v
                    .parse::<f64>()
                    .map_err(|e| format!("invalid --temp-k {v:?}: {e}"))?;
                if !temp_k.is_finite() || !(250.0..=380.0).contains(&temp_k) {
                    return Err(format!("--temp-k {temp_k} out of bounds [250.0, 380.0] K"));
                }
            }
            "--exclude-pdb" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for --exclude-pdb")?;
                exclude_pdb.extend(split_comma_items(v));
            }
            "--exclude-bmrb" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for --exclude-bmrb")?;
                exclude_bmrb.extend(split_comma_items(v));
            }
            "--exclude-topology" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for --exclude-topology")?;
                exclude_topology = Some(v.trim().to_string());
            }
            "--force-bfactor-plddt" => {
                force_bfactor_plddt = true;
            }
            "--json-out" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for --json-out")?;
                json_out = Some(PathBuf::from(v));
            }
            "--out" | "-o" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for --out")?;
                out_path = Some(PathBuf::from(v));
            }
            other => {
                return Err(format!(
                    "unknown exchange-state flag {other:?}\n\n{}",
                    exchange_state_usage()
                ));
            }
        }
        i += 1;
    }

    let sp = shifts_path.ok_or("ferric exchange-state requires --shifts <ground.str|nef>")?;
    let stp =
        structure_path.ok_or("ferric exchange-state requires --structure <ground.cif|pdb>")?;
    if exchange_paths.is_empty() {
        return Err("ferric exchange-state requires --exchange <cest_or_cpmg.tsv>".to_string());
    }

    let t0 = Instant::now();
    let shifts_raw = fs::read_to_string(&sp)
        .map_err(|e| format!("failed to read ground shifts '{}': {e}", sp.display()))?;
    let shifts_ext = sp.extension().and_then(|s| s.to_str());
    let ground_ref = parse_ground_shifts_str(&shifts_raw, shifts_ext)
        .map_err(|e| format!("failed to parse ground shifts '{}': {e}", sp.display()))?;

    let ground_model = StructureModel::parse_file(&stp, force_bfactor_plddt)?;

    let mut all_obs: Vec<ExchangeShiftObservation> = Vec::new();
    let mut seen_cli_keys: BTreeSet<(String, i32, String)> = BTreeSet::new();
    let mut cpmg_prefiltered = false;
    for ep in &exchange_paths {
        let raw_tsv = fs::read_to_string(ep)
            .map_err(|e| format!("failed to read exchange TSV '{}': {e}", ep.display()))?;
        let parsed = parse_exchange_observations_tsv_with_meta(&raw_tsv)
            .map_err(|e| format!("failed to parse exchange TSV '{}': {e}", ep.display()))?;
        cpmg_prefiltered |= parsed.cpmg_prefiltered_significant_only;
        for obs in parsed.observations {
            let key = (obs.chain_id.clone(), obs.seq_id, obs.atom.clone());
            if !seen_cli_keys.insert(key) {
                return Err(format!(
                    "duplicate exchange observation across --exchange input(s) in '{}': (chain_id='{}', seq_id={}, atom='{}')",
                    ep.display(),
                    obs.chain_id,
                    obs.seq_id,
                    obs.atom
                ));
            }
            all_obs.push(obs);
        }
    }

    let mut alt_models: Vec<AltConformerInput> = Vec::with_capacity(alt_paths.len());
    for ap in &alt_paths {
        let model = StructureModel::parse_file(ap, force_bfactor_plddt)?;
        let label = ap
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("alt_model")
            .to_string();
        alt_models.push(AltConformerInput { label, model });
    }

    let fw_query = HomologFirewallQuery {
        query_pdb_ids: exclude_pdb,
        query_bmrb_ids: exclude_bmrb,
        query_topology_id: exclude_topology,
        ..HomologFirewallQuery::default()
    };

    let cfg = ExcitedStateConfig {
        sample_ph,
        temperature_kelvin: temp_k,
        cpmg_prefiltered_significant_only: cpmg_prefiltered,
        ..ExcitedStateConfig::default()
    };

    let report = classify_excited_state_with_firewall(
        Some(&ground_ref),
        &ground_model,
        &all_obs,
        &alt_models,
        &fw_query,
        &cfg,
    )?;
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let tsv_str = format_excited_state_tsv(&report);
    let json_str = format_excited_state_json(&report)?;

    if let Some(ref jp) = json_out {
        fs::write(jp, &json_str).map_err(|e| format!("failed to write '{}': {e}", jp.display()))?;
    }

    if let Some(ref op) = out_path {
        let is_json = op
            .extension()
            .and_then(|s| s.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("json"));
        let payload = if is_json { &json_str } else { &tsv_str };
        fs::write(op, payload).map_err(|e| format!("failed to write '{}': {e}", op.display()))?;
        println!(
            "Excited-state hypothesis engine ({} obs: {} signed, {} unsigned; {:.1} ms): {} [2lnBF_12={:.2}, 2lnBF_null={:.2}, rchi2={:.3}] -> {}",
            report.n_observations,
            report.n_signed_observations,
            report.n_unsigned_observations,
            elapsed_ms,
            report.verdict.as_str(),
            report.two_ln_bf_top1_vs_top2,
            report.two_ln_bf_top1_vs_null,
            report.top1_reduced_chi2,
            op.display()
        );
    } else {
        print!("{tsv_str}");
    }

    Ok(())
}

#[cfg(test)]
mod exchange_state_cli_unit_tests {
    use super::*;
    use nmr_core::excited_state::{
        write_exchange_observations_tsv, ExchangeSignMode, ExcitedStateHypothesisKind,
        ExcitedStateReport, ExcitedStateVerdict,
    };

    #[test]
    fn test_run_exchange_state_cli_help_errors_and_e2e_json_tsv() {
        // 1. `--help` succeeds
        run_exchange_state_cli(&["--help".to_string()]).expect("help");

        // 2. Missing required flags or out-of-range pH/temp fail cleanly
        assert!(run_exchange_state_cli(&["--unknown".to_string()]).is_err());
        assert!(run_exchange_state_cli(&[
            "--shifts".to_string(),
            "missing.str".to_string(),
            "--structure".to_string(),
            "missing.pdb".to_string()
        ])
        .is_err());
        assert!(run_exchange_state_cli(&[
            "--shifts".to_string(),
            "a.str".to_string(),
            "--structure".to_string(),
            "a.pdb".to_string(),
            "--exchange".to_string(),
            "a.tsv".to_string(),
            "--ph".to_string(),
            "19.5".to_string()
        ])
        .is_err());

        // 3. End-to-end execution on Ubiquitin (`1d3z.pdb` + `bmr17769_ubiquitin_allatom.str`)
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .expect("repo root");
        let tmp = std::env::temp_dir().join(format!(
            "ferric_exchange_state_cli_test_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).expect("mkdir tmp");

        let obs = vec![
            ExchangeShiftObservation {
                chain_id: "A".to_string(),
                seq_id: 19,
                res_name: Some("PRO".to_string()),
                atom: "CB".to_string(),
                dw_ppm: 2.40,
                dw_err_ppm: 0.08,
                sign_mode: ExchangeSignMode::UnsignedMagnitude,
                major_ppm: None,
            },
            ExchangeShiftObservation {
                chain_id: "A".to_string(),
                seq_id: 19,
                res_name: Some("PRO".to_string()),
                atom: "CG".to_string(),
                dw_ppm: 2.60,
                dw_err_ppm: 0.08,
                sign_mode: ExchangeSignMode::UnsignedMagnitude,
                major_ppm: None,
            },
            ExchangeShiftObservation {
                chain_id: "A".to_string(),
                seq_id: 18,
                res_name: Some("GLU".to_string()),
                atom: "CA".to_string(),
                dw_ppm: 1.15,
                dw_err_ppm: 0.08,
                sign_mode: ExchangeSignMode::UnsignedMagnitude,
                major_ppm: None,
            },
            ExchangeShiftObservation {
                chain_id: "A".to_string(),
                seq_id: 18,
                res_name: Some("GLU".to_string()),
                atom: "C".to_string(),
                dw_ppm: 1.35,
                dw_err_ppm: 0.08,
                sign_mode: ExchangeSignMode::UnsignedMagnitude,
                major_ppm: None,
            },
            ExchangeShiftObservation {
                chain_id: "A".to_string(),
                seq_id: 20,
                res_name: Some("SER".to_string()),
                atom: "N".to_string(),
                dw_ppm: 1.85,
                dw_err_ppm: 0.12,
                sign_mode: ExchangeSignMode::UnsignedMagnitude,
                major_ppm: None,
            },
            ExchangeShiftObservation {
                chain_id: "A".to_string(),
                seq_id: 21,
                res_name: Some("ASP".to_string()),
                atom: "N".to_string(),
                dw_ppm: 0.45,
                dw_err_ppm: 0.08,
                sign_mode: ExchangeSignMode::UnsignedMagnitude,
                major_ppm: None,
            },
        ];
        let ex_tsv = tmp.join("cpmg_pro19.tsv");
        fs::write(&ex_tsv, write_exchange_observations_tsv(&obs)).expect("write ex_tsv");

        let out_json = tmp.join("report.json");
        let out_tsv = tmp.join("report.tsv");
        run_exchange_state_cli(&[
            "--shifts".to_string(),
            repo_root
                .join("data/fixtures/bmr17769_ubiquitin_allatom.str")
                .to_string_lossy()
                .into_owned(),
            "--structure".to_string(),
            repo_root
                .join("data/pdb_refs/1d3z.pdb")
                .to_string_lossy()
                .into_owned(),
            "--exchange".to_string(),
            ex_tsv.to_string_lossy().into_owned(),
            "--exclude-pdb".to_string(),
            "1D3Z,1UBQ".to_string(),
            "--exclude-bmrb".to_string(),
            "17769,6457".to_string(),
            "--ph".to_string(),
            "6.5".to_string(),
            "--temp-k".to_string(),
            "298.15".to_string(),
            "--json-out".to_string(),
            out_json.to_string_lossy().into_owned(),
            "-o".to_string(),
            out_tsv.to_string_lossy().into_owned(),
        ])
        .expect("run_exchange_state_cli");

        let parsed_json: ExcitedStateReport =
            serde_json::from_str(&fs::read_to_string(&out_json).expect("read json"))
                .expect("deserialize report");
        assert_eq!(
            parsed_json.verdict,
            ExcitedStateVerdict::Committed(ExcitedStateHypothesisKind::XxxProCisTrans)
        );
        assert!(parsed_json.ground_max_retained_seq_identity <= 0.25);
        let tsv_txt = fs::read_to_string(&out_tsv).expect("read tsv");
        assert!(tsv_txt.contains("Committed(XxxProCisTrans)"));

        let _ = fs::remove_dir_all(&tmp);
    }
}
