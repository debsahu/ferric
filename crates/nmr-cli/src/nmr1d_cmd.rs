//! CLI subcommands (`nmrassign nmr1d process`, `nmrassign nmr1d multiplet`, `nmrassign nmr1d qnmr`)
//! for analytical 1D small-molecule & biomolecular NMR digital signal processing, recursive
//! multiplet `J`-coupling tree analysis, publication journal characterization formatting,
//! `Octet` 2D chemical-structure atom-assignment linking, and internal-standard quantitative
//! NMR (`qNMR`) purity metrology.
//!
//! # Clean-Room Provenance
//! Original CLI driver implementing `docs/specs/17-nmr1d-and-dosy.md` §§1–2 from peer-reviewed
//! algorithms in `nmr_core::nmr1d` and `nmr_core::multiplet`:
//! - Bartholdi, E. & Ernst, R. R., *J. Magn. Reson.* **1973**, 11, 9–19 (causality, zero-filling & Hilbert transform).
//! - Ferrige, A. G. & Lindon, J. C., *J. Magn. Reson.* **1978**, 31, 337–340;
//!   Doddrell, D. M., Pegg, D. T. & Bendall, M. R., *J. Magn. Reson.* **1982**, 48, 323–327 (`DEPT-90`/`DEPT-135`).
//! - Chen, L., Weng, Z., Goh, L. & Garland, M., *J. Magn. Reson.* **2002**, 158, 164–168 (minimum-entropy autophasing).
//! - Eilers, P. H. C., *Anal. Chem.* **2003**, 75, 3631–3636;
//!   Cobas, J. C. et al., *J. Magn. Reson.* **2006**, 183, 145–151 (Whittaker & Bernstein baseline flattening).
//! - Barkhuijsen, H., de Beer, R. & van Ormondt, D., *J. Magn. Reson.* **1987**, 73, 553–557 (`HSVD` solvent subtraction).
//! - Fulmer, G. R. et al., *Organometallics* **2010**, 29, 2176–2179 (residual solvent & trace impurity shifts).
//! - Thompson, P., Cox, D. E. & Hastings, J. B., *J. Appl. Crystallogr.* **1987**, 20, 79–83 (pseudo-Voigt deconvolution).
//! - Hoye, T. R., Hanson, P. R. & Vyvyan, J. R., *J. Org. Chem.* **1994**, 59, 4096–4103;
//!   Hoye, T. R. & Zhao, H., *J. Org. Chem.* **2002**, 67, 4014–4016 (recursive multiplet `J`-coupling trees & AB roofing).
//! - Pauli, G. F. et al., *J. Med. Chem.* **2014**, 57, 9220–9231;
//!   Malz, F. & Jancke, H., *J. Pharm. Biomed. Anal.* **2005**, 38, 813–823 (internal-standard `qNMR` purity & GUM uncertainty).

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use nmr_core::fid::read_bruker_fid_1d;
use nmr_core::multiplet::{
    analyze_multiplet_region, auto_assign_1h_multiplets, calculate_qnmr_purity,
    calculate_qnmr_replicates, format_journal_nmr_report, integrate_regions,
    qnmr_certified_standards, BaselineMode, IntegralRegionSpec, IntegratedRegion, JournalNmrReport,
    MultipletAnalysis, MultipletAssignmentTable, MultipletConfig, NormalizationMode,
    PublicationStyle, QnmrAcquisitionParams, QnmrAnalyteSignal, QnmrAnalyteSpec, QnmrPurityReport,
    QnmrStandardSpec, ReportNucleus, ReportedSignal, TextEncoding,
};
use nmr_core::nmr1d::{
    apply_apodization_1d, apply_phase_1d, auto_calibrate_solvent_1d, auto_phase_and_apply_1d,
    correct_baseline_1d, deconvolve_region_1d, deconvolve_spectrum_1d, estimate_noise_sigma_1d,
    eval_pseudo_voigt_sum_1d, fft_1d, match_trace_impurities, pick_peaks_1d,
    subtract_solvent_hsvd_1d, suppress_solvent_notch_1d, zero_fill_1d, zero_fill_pow2_1d,
    Apodization1D, AutoPhase1DConfig, BaselineConfig1D, BaselineMethod1D, DeconvolutionResult1D,
    DeconvolveConfig1D, Experiment1DKind, HsvdConfig, HsvdPole, NotchConfig1D, Nucleus1D,
    Peak1DPicked, PeakPickConfig1D, PhaseResult1D, Pivot1D, Spectrum1D, TimeDomain1D, VoigtPeak1D,
};
use nmr_core::octet_analyt::jcamp_spec1d::{read_bruker_1d, read_jcamp, Spectrum1d};
use nmr_core::octet_mol::{parse_smiles, read_mol, read_sdf, Molecule};
use nmr_core::spectrum::AxisPpm;

/// Usage text for `nmrassign nmr1d`.
pub fn nmr1d_usage() -> &'static str {
    "Usage:\n\
     \x20 nmrassign nmr1d process   --in <fid_dir|pdata_dir|spec.jdx|spec.tsv|spec.json> --out <spec.tsv|spec.json>\n\
     \x20                           [--experiment 1h|13c|dept90|dept135|19f|31p] [--mhz <f64>] [--solvent <SOLV>]\n\
     \x20                           [--hsvd [<lo:hi,...>]] [--hsvd-rank 24] [--apod em|gm|lg|cos|cos2|ssb|none]\n\
     \x20                           [--lb <hz>] [--gb <hz>] [--lg-center 0.0] [--first-point 0.5] [--zf 2 | --zf-points <N>]\n\
     \x20                           [--auto-phase | --phase <PHC0:PHC1>] [--pivot <frac|ppm:val|idx:k|max>]\n\
     \x20                           [--exclude-ppm <lo:hi,...>] [--biphasic-pos <lo:hi>] [--notch <lo:hi,...>]\n\
     \x20                           [--baseline whittaker|asymmetric-whittaker|bernstein|none] [--lambda 1e6] [--bernstein-degree 6]\n\
     \x20                           [--calibrate-solvent <SOLV>] [--cal-window 0.25] [--ref-offset <ppm>]\n\
     \x20                           [--min-snr 6.0] [--shoulders] [--match-impurities] [--deconvolve] [--deconv-region <lo:hi>]\n\
     \x20                           [--peaks-out <peaks.tsv>] [--json-out <summary.json>]\n\
     \x20 nmrassign nmr1d multiplet --in <spec.tsv|spec.jdx|spec.json|pdata_dir> [--out <multiplets.tsv>]\n\
     \x20                           [--mhz <f64>] [--solvent CDCl3] [--nucleus 1H|13C|19F|31P|15N] [--proton-decoupled]\n\
     \x20                           [--region <hi:lo[:label]> ...] [--regions <hi1:lo1[:lbl],...>]\n\
     \x20                           [--norm-ref <idx:protons> | --total-protons <H> | --norm-scale <factor>]\n\
     \x20                           [--region-baseline auto|none|<bias:slope>] [--style acs|joc|rsc|nature]\n\
     \x20                           [--encoding unicode|ascii|latex] [--mol <file.mol|file.sdf> | --smiles <SMILES>]\n\
     \x20                           [--links-out <links.tsv|links.json>] [--report-out <report.txt>] [--json-out <out.json>]\n\
     \x20 nmrassign nmr1d qnmr      [--in <spec.tsv|spec.jdx|spec.json|pdata_dir> --std-region <hi:lo> --analyte-region <hi:lo:H[:lbl][:t1]> ...]\n\
     \x20                           [--std-crm DMSO2|MA|TMB|DMT|BA|DNBA | --std-name <name> --std-mw <g_mol> --std-protons <N>]\n\
     \x20                           [--std-mass-mg <mg>] [--std-mass-sigma-mg 0.0] [--std-purity <0..1> | --std-purity-pct <0..100>]\n\
     \x20                           [--std-purity-sigma <0..1> | --std-purity-sigma-pct <0..100>] [--std-mw-sigma 0.0] [--std-t1 <s>]\n\
     \x20                           [--std-integral <I> [--std-integral-sigma <u_I>]]\n\
     \x20                           [--analyte-name <name>] [--analyte-mw <g_mol>] [--analyte-mw-sigma 0.0]\n\
     \x20                           [--analyte-mass-mg <mg>] [--analyte-mass-sigma-mg 0.0] [--volume-ml <mL>] [--volume-sigma-ml <mL>]\n\
     \x20                           [--signal <H:integral[:sigma][:label][:t1]> ...] [--d1-s <s>] [--aq-s <s>] [--flip-deg 90.0]\n\
     \x20                           [--replicates-json <tubes.json>] [--out <qnmr.tsv>] [--json-out <qnmr.json>]\n"
}

/// Top-level dispatcher for `nmrassign nmr1d <subcommand>`.
pub fn run_nmr1d_cli(args: &[String]) -> Result<(), String> {
    if args.is_empty() || args[0] == "-h" || args[0] == "--help" {
        print!("{}", nmr1d_usage());
        return Ok(());
    }
    match args[0].as_str() {
        "process" => run_nmr1d_process(&args[1..]),
        "multiplet" | "multiplets" => run_nmr1d_multiplet(&args[1..]),
        "qnmr" => run_nmr1d_qnmr(&args[1..]),
        other => Err(format!(
            "unknown `nmr1d` subcommand {other:?}\n\n{}",
            nmr1d_usage()
        )),
    }
}

// ============================================================================
// 1D Input / Output Helpers (Bruker FID, Bruker pdata, JCAMP-DX, JSON, TSV)
// ============================================================================

enum Loaded1DInput {
    TimeDomain(TimeDomain1D),
    FrequencyDomain(Spectrum1D),
}

fn parse_experiment_kind(label: &str) -> Result<Experiment1DKind, String> {
    let norm = label
        .trim()
        .to_ascii_lowercase()
        .replace(['-', '_', '{', '}', ' '], "");
    match norm.as_str() {
        "1h" | "h1" | "proton" | "proton1d" => Ok(Experiment1DKind::Proton1D),
        "13c" | "c13" | "13c1h" | "carbon" | "carbon13" | "carbon13decoupled" => {
            Ok(Experiment1DKind::Carbon13Decoupled)
        }
        "dept90" => Ok(Experiment1DKind::Dept90),
        "dept135" | "deptq135" => Ok(Experiment1DKind::Dept135),
        "19f" | "f19" | "fluorine19" => Ok(Experiment1DKind::Fluorine19),
        "31p" | "p31" | "phosphorus31" => Ok(Experiment1DKind::Phosphorus31),
        _ => Ok(Experiment1DKind::Custom(Nucleus1D::parse(label))),
    }
}

fn spectrum1d_from_octet_jcamp(
    spec: &Spectrum1d,
    mhz_override: Option<f64>,
) -> Result<Spectrum1D, String> {
    if spec.y.len() < 2 {
        return Err("1D spectrum contains fewer than 2 points".to_string());
    }
    let nuc_str = spec.nucleus.as_deref().unwrap_or("1H");
    let exp = Experiment1DKind::from_pulse_program("", nuc_str);
    let mhz = mhz_override.or(spec.mhz).unwrap_or(500.0).max(1e-6);

    let mut xs = spec.x_axis();
    let mut ys = spec.y.clone();
    // Standardize to descending ppm (`first_ppm = max_ppm`, `step_ppm < 0`) as required by `Spectrum1D`.
    if xs.first().copied().unwrap_or(0.0) < xs.last().copied().unwrap_or(0.0) {
        xs.reverse();
        ys.reverse();
    }
    let high_ppm = xs[0];
    let low_ppm = *xs.last().unwrap_or(&high_ppm);
    let axis = AxisPpm::from_bounds(exp.nucleus().as_str(), ys.len(), high_ppm, low_ppm, mhz);
    let mut out = Spectrum1D::new(spec.title.clone(), exp, axis, ys, None);
    out.solvent = spec.solvent.clone();
    Ok(out)
}

fn try_load_bruker_pdata_dir(
    dir: &Path,
    mhz_override: Option<f64>,
) -> Result<Option<Spectrum1D>, String> {
    // Check direct `dir/procs` + `dir/1r` or `dir/pdata/1/procs` + `dir/pdata/1/1r`
    let candidates = [dir.to_path_buf(), dir.join("pdata").join("1")];
    for pdir in &candidates {
        let procs_path = pdir.join("procs");
        let data_1r_path = pdir.join("1r");
        if procs_path.is_file() && data_1r_path.is_file() {
            let procs = fs::read_to_string(&procs_path)
                .map_err(|e| format!("failed to read {}: {e}", procs_path.display()))?;
            let data_1r = fs::read(&data_1r_path)
                .map_err(|e| format!("failed to read {}: {e}", data_1r_path.display()))?;
            let acqus = pdir
                .parent()
                .and_then(|p| p.parent())
                .map(|exp_dir| exp_dir.join("acqus"))
                .filter(|p| p.is_file())
                .or_else(|| {
                    let direct_acqus = dir.join("acqus");
                    direct_acqus.is_file().then_some(direct_acqus)
                })
                .and_then(|p| fs::read_to_string(p).ok())
                .unwrap_or_default();
            let name = dir
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("bruker_1d");
            let spec_j = read_bruker_1d(&procs, &data_1r, &acqus, name)?;
            let mut spec = spectrum1d_from_octet_jcamp(&spec_j, mhz_override)?;

            // Load optional imaginary file `1i` if present with matching word count.
            let data_1i_path = pdir.join("1i");
            if data_1i_path.is_file() {
                if let Ok(bytes_1i) = fs::read(&data_1i_path) {
                    if let Ok(spec_im_j) = read_bruker_1d(&procs, &bytes_1i, &acqus, name) {
                        let mut im_y = spec_im_j.y;
                        if spec_j.first_x < spec_j.last_x {
                            im_y.reverse();
                        }
                        if im_y.len() == spec.real.len() {
                            spec.imag = Some(im_y);
                        }
                    }
                }
            }
            return Ok(Some(spec));
        }
    }
    Ok(None)
}

fn parse_tsv_1d(
    text: &str,
    name: &str,
    mhz_override: Option<f64>,
) -> Result<Loaded1DInput, String> {
    let mut meta: BTreeMap<String, String> = BTreeMap::new();
    let mut header_cols: Vec<String> = Vec::new();
    let mut rows: Vec<Vec<f64>> = Vec::new();

    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(comment) = line.strip_prefix('#') {
            for part in comment.split([';', ',']) {
                if let Some((k, v)) = part.split_once('=') {
                    meta.insert(k.trim().to_ascii_lowercase(), v.trim().to_string());
                }
            }
            continue;
        }
        let fields: Vec<&str> = line
            .split(['\t', ','])
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        if fields.len() < 2 {
            continue;
        }
        if rows.is_empty() && fields[0].parse::<f64>().is_err() {
            header_cols = fields.iter().map(|s| s.to_ascii_lowercase()).collect();
            continue;
        }
        let parsed: Result<Vec<f64>, _> = fields.iter().map(|s| s.parse::<f64>()).collect();
        let vals = parsed.map_err(|e| format!("invalid numeric value in 1D TSV {line:?}: {e}"))?;
        rows.push(vals);
    }

    if rows.len() < 2 {
        return Err("1D TSV input contains fewer than 2 numeric rows".to_string());
    }

    let exp = meta
        .get("experiment")
        .or_else(|| meta.get("nucleus"))
        .map(|s| parse_experiment_kind(s))
        .transpose()?
        .unwrap_or(Experiment1DKind::Proton1D);
    let mhz = mhz_override
        .or_else(|| meta.get("obs_mhz").and_then(|s| s.parse::<f64>().ok()))
        .or_else(|| meta.get("mhz").and_then(|s| s.parse::<f64>().ok()))
        .unwrap_or(500.0)
        .max(1e-6);

    let first_col_name = header_cols.first().map(String::as_str).unwrap_or("");
    let is_fid = meta
        .get("domain")
        .is_some_and(|d| d.eq_ignore_ascii_case("time") || d.eq_ignore_ascii_case("fid"))
        || matches!(first_col_name, "t" | "t_s" | "time" | "time_s" | "sample");

    if is_fid {
        let dt_from_col = if rows.len() >= 2 && (rows[1][0] - rows[0][0]).abs() > 1e-12 {
            Some((rows[1][0] - rows[0][0]).abs())
        } else {
            None
        };
        let sw_hz = meta
            .get("sw_hz")
            .and_then(|s| s.parse::<f64>().ok())
            .or_else(|| dt_from_col.map(|dt| 1.0 / dt))
            .unwrap_or(6000.0);
        let carrier_ppm = meta
            .get("carrier_ppm")
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(4.70);
        let mut data = Vec::with_capacity(rows.len());
        for r in &rows {
            if r.len() >= 3 {
                data.push([r[1], r[2]]);
            } else {
                data.push([r[0], r[1]]);
            }
        }
        let mut td = TimeDomain1D::new(name.to_string(), exp, sw_hz, mhz, carrier_ppm, data);
        if let Some(gd) = meta
            .get("group_delay_pts")
            .and_then(|s| s.parse::<f64>().ok())
        {
            td.group_delay_pts = gd;
        }
        return Ok(Loaded1DInput::TimeDomain(td));
    }

    // Frequency-domain `ppm \t real [\t imag]`
    let mut pts: Vec<(f64, f64, Option<f64>)> = rows
        .into_iter()
        .map(|r| (r[0], r[1], r.get(2).copied()))
        .collect();
    // Ensure descending ppm order (`first_ppm` = downfield max ppm, `last_ppm` = upfield min ppm).
    if pts.first().map(|p| p.0).unwrap_or(0.0) < pts.last().map(|p| p.0).unwrap_or(0.0) {
        pts.reverse();
    }
    let high_ppm = pts[0].0;
    let low_ppm = pts.last().map(|p| p.0).unwrap_or(high_ppm);
    let real: Vec<f64> = pts.iter().map(|p| p.1).collect();
    let has_imag = pts.iter().all(|p| p.2.is_some());
    let imag: Option<Vec<f64>> = has_imag.then(|| pts.iter().filter_map(|p| p.2).collect());

    let axis = AxisPpm::from_bounds(exp.nucleus().as_str(), real.len(), high_ppm, low_ppm, mhz);
    let mut spec = Spectrum1D::new(name.to_string(), exp, axis, real, imag);
    if let Some(solv) = meta.get("solvent") {
        if !solv.is_empty() && solv != "none" {
            spec.solvent = Some(solv.clone());
        }
    }
    Ok(Loaded1DInput::FrequencyDomain(spec))
}

fn load_1d_input(
    path: &Path,
    mhz_override: Option<f64>,
    prefer_fid: bool,
) -> Result<Loaded1DInput, String> {
    let name = path
        .file_stem()
        .or_else(|| path.file_name())
        .and_then(|s| s.to_str())
        .unwrap_or("spectrum_1d");

    if path.is_dir() {
        let fid_file = path.join("fid");
        if prefer_fid && fid_file.is_file() {
            let fid1d = read_bruker_fid_1d(path)?;
            let mut td = TimeDomain1D::from_fid1d(&fid1d);
            if let Some(m) = mhz_override {
                td.obs_mhz = m;
            }
            return Ok(Loaded1DInput::TimeDomain(td));
        }
        if let Some(spec) = try_load_bruker_pdata_dir(path, mhz_override)? {
            return Ok(Loaded1DInput::FrequencyDomain(spec));
        }
        if fid_file.is_file() {
            let fid1d = read_bruker_fid_1d(path)?;
            let mut td = TimeDomain1D::from_fid1d(&fid1d);
            if let Some(m) = mhz_override {
                td.obs_mhz = m;
            }
            return Ok(Loaded1DInput::TimeDomain(td));
        }
        return Err(format!(
            "directory {} contains neither Bruker 1D `fid`+`acqus` nor `procs`+`1r`",
            path.display()
        ));
    }

    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    let text = fs::read_to_string(path)
        .map_err(|e| format!("failed to read 1D input {}: {e}", path.display()))?;

    if ext == "json" {
        if let Ok(mut spec) = serde_json::from_str::<Spectrum1D>(&text) {
            if let Some(m) = mhz_override {
                spec.axis.obs_mhz = m;
            }
            return Ok(Loaded1DInput::FrequencyDomain(spec));
        }
        if let Ok(mut td) = serde_json::from_str::<TimeDomain1D>(&text) {
            if let Some(m) = mhz_override {
                td.obs_mhz = m;
            }
            return Ok(Loaded1DInput::TimeDomain(td));
        }
        return Err(format!(
            "JSON file {} matched neither `Spectrum1D` nor `TimeDomain1D` schema",
            path.display()
        ));
    }

    if ext == "jdx" || ext == "dx" || ext == "jcamp" || text.trim_start().starts_with("##TITLE=") {
        let blocks = read_jcamp(&text)?;
        let first = blocks.first().ok_or_else(|| {
            format!(
                "JCAMP-DX file {} contained zero spectrum blocks",
                path.display()
            )
        })?;
        let spec = spectrum1d_from_octet_jcamp(first, mhz_override)?;
        return Ok(Loaded1DInput::FrequencyDomain(spec));
    }

    parse_tsv_1d(&text, name, mhz_override)
}

fn load_processed_spectrum_1d(
    path: &Path,
    mhz_override: Option<f64>,
) -> Result<Spectrum1D, String> {
    match load_1d_input(path, mhz_override, false)? {
        Loaded1DInput::FrequencyDomain(spec) => Ok(spec),
        Loaded1DInput::TimeDomain(mut td) => {
            let lb = td.experiment.default_line_broadening_hz();
            apply_apodization_1d(&mut td, Apodization1D::Exponential { lb_hz: lb }, 0.5);
            zero_fill_pow2_1d(&mut td, 2);
            let mut spec = fft_1d(&td);
            let _ = auto_phase_and_apply_1d(&mut spec, &AutoPhase1DConfig::default());
            Ok(spec)
        }
    }
}

fn write_spectrum_1d_tsv(
    path: &Path,
    spec: &Spectrum1D,
    fitted: Option<&[f64]>,
    residual: Option<&[f64]>,
) -> Result<(), String> {
    let mut out = String::with_capacity(spec.len() * 48 + 256);
    out.push_str(&format!(
        "# name={}; nucleus={}; obs_mhz={:.6}; solvent={}; phc0_deg={:.4}; phc1_deg={:.4}; pivot_frac={:.4}\n",
        spec.name,
        spec.experiment.nucleus().as_str(),
        spec.axis.obs_mhz,
        spec.solvent.as_deref().unwrap_or("none"),
        spec.phase_applied.0,
        spec.phase_applied.1,
        spec.phase_applied.2
    ));
    let has_im = spec.imag.is_some();
    let has_fit = fitted.is_some_and(|f| f.len() == spec.len())
        && residual.is_some_and(|r| r.len() == spec.len());

    match (has_im, has_fit) {
        (true, true) => out.push_str("ppm\treal\timag\tfitted\tresidual\n"),
        (true, false) => out.push_str("ppm\treal\timag\n"),
        (false, true) => out.push_str("ppm\treal\tfitted\tresidual\n"),
        (false, false) => out.push_str("ppm\treal\n"),
    }

    for i in 0..spec.len() {
        let ppm = spec.ppm_at(i);
        let re = spec.real[i];
        match (spec.imag.as_ref(), fitted, residual) {
            (Some(im), Some(fit), Some(res)) if has_fit => {
                out.push_str(&format!(
                    "{ppm:.6}\t{re:.8e}\t{:.8e}\t{:.8e}\t{:.8e}\n",
                    im[i], fit[i], res[i]
                ));
            }
            (Some(im), _, _) => {
                out.push_str(&format!("{ppm:.6}\t{re:.8e}\t{:.8e}\n", im[i]));
            }
            (None, Some(fit), Some(res)) if has_fit => {
                out.push_str(&format!(
                    "{ppm:.6}\t{re:.8e}\t{:.8e}\t{:.8e}\n",
                    fit[i], res[i]
                ));
            }
            (None, _, _) => {
                out.push_str(&format!("{ppm:.6}\t{re:.8e}\n"));
            }
        }
    }
    fs::write(path, out).map_err(|e| format!("failed to write {}: {e}", path.display()))
}

fn parse_ppm_pair(spec: &str, flag: &str) -> Result<(f64, f64), String> {
    let (a, b) = spec
        .split_once(':')
        .ok_or_else(|| format!("{flag} expects `<ppm_a>:<ppm_b>`, got {spec:?}"))?;
    let va = a
        .trim()
        .parse::<f64>()
        .map_err(|_| format!("invalid first ppm in {flag} {spec:?}"))?;
    let vb = b
        .trim()
        .parse::<f64>()
        .map_err(|_| format!("invalid second ppm in {flag} {spec:?}"))?;
    if !va.is_finite() || !vb.is_finite() {
        return Err(format!("{flag} bounds must be finite numbers"));
    }
    Ok((va.min(vb), va.max(vb)))
}

fn parse_ppm_pair_list(csv: &str, flag: &str) -> Result<Vec<(f64, f64)>, String> {
    let mut out = Vec::new();
    for item in csv.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        out.push(parse_ppm_pair(item, flag)?);
    }
    if out.is_empty() {
        return Err(format!("{flag} requires at least one `<lo>:<hi>` window"));
    }
    Ok(out)
}

fn parse_pivot(spec: &str) -> Result<Pivot1D, String> {
    let s = spec.trim();
    if s.eq_ignore_ascii_case("max") || s.eq_ignore_ascii_case("maxpeak") {
        return Ok(Pivot1D::MaxPeak);
    }
    if let Some(rest) = s.strip_prefix("ppm:") {
        let v = rest
            .trim()
            .parse::<f64>()
            .map_err(|_| format!("invalid `--pivot ppm:<val>`: {s:?}"))?;
        return Ok(Pivot1D::Ppm(v));
    }
    if let Some(rest) = s.strip_prefix("idx:") {
        let k = rest
            .trim()
            .parse::<usize>()
            .map_err(|_| format!("invalid `--pivot idx:<k>`: {s:?}"))?;
        return Ok(Pivot1D::Index(k));
    }
    let frac = s.parse::<f64>().map_err(|_| {
        format!("invalid `--pivot` {s:?} (expected fraction 0..1, `ppm:<v>`, `idx:<k>`, or `max`)")
    })?;
    Ok(Pivot1D::Fraction(frac))
}

// ============================================================================
// Subcommand 1: `nmrassign nmr1d process`
// ============================================================================

/// Owned JSON-serializable laboratory trace-impurity match (Fulmer et al. 2010).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliImpurityMatch {
    pub observed_ppm: f64,
    pub compound: String,
    pub site: String,
    pub solvent: String,
    pub nucleus: String,
    pub reference_ppm: f64,
    pub multiplicity: String,
    pub delta_ppm: f64,
}

/// Structured JSON summary emitted by `nmrassign nmr1d process --json-out`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Process1dCliReport {
    pub name: String,
    pub nucleus: String,
    pub obs_mhz: f64,
    pub num_points: usize,
    pub first_ppm: f64,
    pub last_ppm: f64,
    pub solvent: Option<String>,
    pub hsvd_poles_subtracted: usize,
    pub hsvd_subtracted: Vec<HsvdPole>,
    pub phase_applied_deg: (f64, f64, f64),
    pub auto_phase: Option<PhaseResult1D>,
    pub baseline_noise_sigma: f64,
    pub solvent_calibration_delta_ppm: Option<f64>,
    pub picked_peaks: Vec<Peak1DPicked>,
    pub impurity_matches: Vec<CliImpurityMatch>,
    pub deconvolved_peaks: Vec<VoigtPeak1D>,
    pub deconvolve_residual_rmsd: Option<f64>,
}

fn run_nmr1d_process(args: &[String]) -> Result<(), String> {
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        print!("{}", nmr1d_usage());
        return Ok(());
    }

    let mut in_path: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut peaks_out: Option<PathBuf> = None;
    let mut json_out: Option<PathBuf> = None;
    let mut exp_override: Option<Experiment1DKind> = None;
    let mut mhz_override: Option<f64> = None;
    let mut solvent_override: Option<String> = None;

    let mut hsvd_windows: Option<Vec<(f64, f64)>> = None;
    let mut hsvd_rank: usize = 24;
    let mut hsvd_pts: usize = 1024;

    let mut apod_kind = String::from("em");
    let mut lb_hz: Option<f64> = None;
    let mut gb_hz: f64 = 1.0;
    let mut lg_center: f64 = 0.0;
    let mut ssb_offset: f64 = 0.5;
    let mut first_point_scale: f64 = 0.5;
    let mut zf_factor: usize = 2;
    let mut zf_points: Option<usize> = None;

    let mut auto_phase = false;
    let mut manual_phase: Option<(f64, f64)> = None;
    let mut pivot = Pivot1D::Fraction(0.5);
    let mut exclude_ppm: Vec<(f64, f64)> = Vec::new();
    let mut biphasic_pos: Option<(f64, f64)> = None;
    let mut notch_windows: Vec<(f64, f64)> = Vec::new();

    let mut baseline_mode = String::from("none");
    let mut whittaker_lambda: f64 = 1.0e6;
    let mut bernstein_degree: usize = 6;

    let mut calibrate_solvent: Option<String> = None;
    let mut cal_window_ppm: f64 = 0.25;
    let mut ref_offset_ppm: Option<f64> = None;

    let mut min_snr: f64 = 6.0;
    let mut detect_shoulders = false;
    let mut match_impurities_flag = false;
    let mut deconvolve_flag = false;
    let mut deconv_region: Option<(f64, f64)> = None;
    let mut max_lm_iters: usize = 120;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--in" | "-i" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--in`")?;
                in_path = Some(PathBuf::from(v));
            }
            "--out" | "-o" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--out`")?;
                out_path = Some(PathBuf::from(v));
            }
            "--peaks-out" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--peaks-out`")?;
                peaks_out = Some(PathBuf::from(v));
            }
            "--json-out" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--json-out`")?;
                json_out = Some(PathBuf::from(v));
            }
            "--experiment" | "--exp" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--experiment`")?;
                exp_override = Some(parse_experiment_kind(v)?);
            }
            "--mhz" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--mhz`")?;
                mhz_override = Some(
                    v.parse::<f64>()
                        .map_err(|_| format!("invalid `--mhz` {v:?}"))?,
                );
            }
            "--solvent" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--solvent`")?;
                solvent_override = Some(v.clone());
            }
            "--hsvd" => {
                if let Some(next) = args.get(i + 1) {
                    if !next.starts_with('-') && next.contains(':') {
                        i += 1;
                        hsvd_windows = Some(parse_ppm_pair_list(next, "--hsvd")?);
                    } else {
                        hsvd_windows = Some(vec![(4.55, 5.05)]);
                    }
                } else {
                    hsvd_windows = Some(vec![(4.55, 5.05)]);
                }
            }
            "--hsvd-rank" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--hsvd-rank`")?;
                hsvd_rank = v
                    .parse::<usize>()
                    .map_err(|_| format!("invalid `--hsvd-rank` {v:?}"))?;
            }
            "--hsvd-pts" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--hsvd-pts`")?;
                hsvd_pts = v
                    .parse::<usize>()
                    .map_err(|_| format!("invalid `--hsvd-pts` {v:?}"))?;
            }
            "--apod" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--apod`")?;
                apod_kind = v.to_ascii_lowercase();
            }
            "--lb" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--lb`")?;
                lb_hz = Some(
                    v.parse::<f64>()
                        .map_err(|_| format!("invalid `--lb` {v:?}"))?,
                );
            }
            "--gb" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--gb`")?;
                gb_hz = v
                    .parse::<f64>()
                    .map_err(|_| format!("invalid `--gb` {v:?}"))?;
            }
            "--lg-center" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--lg-center`")?;
                lg_center = v
                    .parse::<f64>()
                    .map_err(|_| format!("invalid `--lg-center` {v:?}"))?;
            }
            "--ssb-offset" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--ssb-offset`")?;
                ssb_offset = v
                    .parse::<f64>()
                    .map_err(|_| format!("invalid `--ssb-offset` {v:?}"))?;
            }
            "--first-point" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--first-point`")?;
                first_point_scale = v
                    .parse::<f64>()
                    .map_err(|_| format!("invalid `--first-point` {v:?}"))?;
            }
            "--zf" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--zf`")?;
                zf_factor = v
                    .parse::<usize>()
                    .map_err(|_| format!("invalid `--zf` {v:?}"))?;
            }
            "--zf-points" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--zf-points`")?;
                zf_points = Some(
                    v.parse::<usize>()
                        .map_err(|_| format!("invalid `--zf-points` {v:?}"))?,
                );
            }
            "--auto-phase" => {
                auto_phase = true;
            }
            "--phase" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--phase`")?;
                let (p0_s, p1_s) = v
                    .split_once(':')
                    .ok_or_else(|| format!("`--phase` expects `<PHC0>:<PHC1>`, got {v:?}"))?;
                let p0 = p0_s
                    .parse::<f64>()
                    .map_err(|_| format!("invalid PHC0 in `--phase` {v:?}"))?;
                let p1 = p1_s
                    .parse::<f64>()
                    .map_err(|_| format!("invalid PHC1 in `--phase` {v:?}"))?;
                manual_phase = Some((p0, p1));
            }
            "--pivot" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--pivot`")?;
                pivot = parse_pivot(v)?;
            }
            "--exclude-ppm" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--exclude-ppm`")?;
                exclude_ppm.extend(parse_ppm_pair_list(v, "--exclude-ppm")?);
            }
            "--biphasic-pos" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--biphasic-pos`")?;
                biphasic_pos = Some(parse_ppm_pair(v, "--biphasic-pos")?);
            }
            "--notch" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--notch`")?;
                notch_windows.extend(parse_ppm_pair_list(v, "--notch")?);
            }
            "--baseline" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--baseline`")?;
                baseline_mode = v.to_ascii_lowercase();
            }
            "--lambda" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--lambda`")?;
                whittaker_lambda = v
                    .parse::<f64>()
                    .map_err(|_| format!("invalid `--lambda` {v:?}"))?;
            }
            "--bernstein-degree" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or("missing value for `--bernstein-degree`")?;
                bernstein_degree = v
                    .parse::<usize>()
                    .map_err(|_| format!("invalid `--bernstein-degree` {v:?}"))?;
            }
            "--calibrate-solvent" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or("missing value for `--calibrate-solvent`")?;
                calibrate_solvent = Some(v.clone());
            }
            "--cal-window" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--cal-window`")?;
                cal_window_ppm = v
                    .parse::<f64>()
                    .map_err(|_| format!("invalid `--cal-window` {v:?}"))?;
            }
            "--ref-offset" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--ref-offset`")?;
                ref_offset_ppm = Some(
                    v.parse::<f64>()
                        .map_err(|_| format!("invalid `--ref-offset` {v:?}"))?,
                );
            }
            "--min-snr" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--min-snr`")?;
                min_snr = v
                    .parse::<f64>()
                    .map_err(|_| format!("invalid `--min-snr` {v:?}"))?;
            }
            "--shoulders" => {
                detect_shoulders = true;
            }
            "--match-impurities" => {
                match_impurities_flag = true;
            }
            "--deconvolve" => {
                deconvolve_flag = true;
            }
            "--deconv-region" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--deconv-region`")?;
                deconv_region = Some(parse_ppm_pair(v, "--deconv-region")?);
                deconvolve_flag = true;
            }
            "--max-lm-iters" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--max-lm-iters`")?;
                max_lm_iters = v
                    .parse::<usize>()
                    .map_err(|_| format!("invalid `--max-lm-iters` {v:?}"))?;
            }
            other => {
                return Err(format!(
                    "unknown `nmr1d process` option {other:?}\n\n{}",
                    nmr1d_usage()
                ));
            }
        }
        i += 1;
    }

    let in_path = in_path.ok_or("`nmr1d process` requires `--in <path>`")?;
    let out_path = out_path.ok_or("`nmr1d process` requires `--out <path>`")?;

    let loaded = load_1d_input(&in_path, mhz_override, true)?;
    let mut hsvd_subtracted: Vec<HsvdPole> = Vec::new();

    let mut spec = match loaded {
        Loaded1DInput::TimeDomain(mut td) => {
            if let Some(ref exp) = exp_override {
                td.experiment = exp.clone();
            }
            if let Some(ref wins) = hsvd_windows {
                let hsvd_res = subtract_solvent_hsvd_1d(
                    &mut td,
                    &HsvdConfig {
                        subspace_rank: hsvd_rank,
                        max_hankel_points: hsvd_pts,
                        solvent_ppm_windows: wins.clone(),
                    },
                );
                hsvd_subtracted = hsvd_res.subtracted_poles;
            }

            let default_lb = td.experiment.default_line_broadening_hz();
            let window = match apod_kind.as_str() {
                "none" | "rect" => Apodization1D::None,
                "em" | "exp" | "exponential" => Apodization1D::Exponential {
                    lb_hz: lb_hz.unwrap_or(default_lb),
                },
                "gm" | "gauss" | "gaussian" => Apodization1D::Gaussian { gb_hz },
                "lg" | "lorentz-gauss" => Apodization1D::LorentzToGauss {
                    narrow_lb_hz: lb_hz.unwrap_or(default_lb).abs().max(0.1),
                    gb_hz,
                    center_frac: lg_center,
                },
                "cos" | "cosine" => Apodization1D::CosineBell { power: 1.0 },
                "cos2" | "cosine2" => Apodization1D::CosineBell { power: 2.0 },
                "ssb" | "sine" => Apodization1D::ShiftedSineBell {
                    offset_pi_frac: ssb_offset,
                    power: 2.0,
                },
                other => {
                    return Err(format!(
                        "unsupported `--apod` window {other:?} (expected em|gm|lg|cos|cos2|ssb|none)"
                    ));
                }
            };
            apply_apodization_1d(&mut td, window, first_point_scale);

            if let Some(target_n) = zf_points {
                zero_fill_1d(&mut td, target_n);
            } else if zf_factor > 0 {
                zero_fill_pow2_1d(&mut td, zf_factor);
            }
            fft_1d(&td)
        }
        Loaded1DInput::FrequencyDomain(mut sp) => {
            if let Some(ref exp) = exp_override {
                sp.experiment = exp.clone();
                sp.axis.label = exp.nucleus().as_str().to_string();
            }
            sp
        }
    };

    if let Some(solv) = solvent_override {
        spec.solvent = Some(solv);
    }

    let mut auto_phase_res: Option<PhaseResult1D> = None;
    if let Some((p0, p1)) = manual_phase {
        apply_phase_1d(&mut spec, p0, p1, pivot);
    } else if auto_phase {
        let mut ap_cfg = AutoPhase1DConfig {
            pivot,
            excluded_ppm_regions: exclude_ppm,
            biphasic_positive_ppm: biphasic_pos,
            ..AutoPhase1DConfig::default()
        };
        if spec.experiment == Experiment1DKind::Dept135 && ap_cfg.biphasic_positive_ppm.is_none() {
            ap_cfg.polarity = Some(nmr_core::nmr1d::Polarity1D::Biphasic);
        }
        auto_phase_res = Some(auto_phase_and_apply_1d(&mut spec, &ap_cfg));
    }

    for &(lo, hi) in &notch_windows {
        suppress_solvent_notch_1d(&mut spec, &NotchConfig1D::new(lo, hi));
    }

    let mut baseline_noise_sigma = estimate_noise_sigma_1d(&spec);
    match baseline_mode.as_str() {
        "none" | "off" => {}
        "whittaker" | "als" | "pls" => {
            let bres = correct_baseline_1d(
                &mut spec,
                &BaselineConfig1D {
                    method: BaselineMethod1D::Whittaker {
                        lambda: whittaker_lambda,
                    },
                    ..BaselineConfig1D::default()
                },
            );
            baseline_noise_sigma = bres.baseline_noise_sigma;
        }
        "asymmetric-whittaker" | "asym-whittaker" => {
            let bres = correct_baseline_1d(
                &mut spec,
                &BaselineConfig1D {
                    method: BaselineMethod1D::AsymmetricWhittaker {
                        lambda: whittaker_lambda,
                        asymmetry_p: 0.005,
                    },
                    ..BaselineConfig1D::default()
                },
            );
            baseline_noise_sigma = bres.baseline_noise_sigma;
        }
        "bernstein" | "poly" => {
            let bres = correct_baseline_1d(
                &mut spec,
                &BaselineConfig1D {
                    method: BaselineMethod1D::Bernstein {
                        degree: bernstein_degree,
                    },
                    ..BaselineConfig1D::default()
                },
            );
            baseline_noise_sigma = bres.baseline_noise_sigma;
        }
        other => {
            return Err(format!(
                "unknown `--baseline` method {other:?} (expected whittaker|asymmetric-whittaker|bernstein|none)"
            ));
        }
    }

    let mut solvent_cal_delta: Option<f64> = None;
    if let Some(ref solv_name) = calibrate_solvent {
        let delta = auto_calibrate_solvent_1d(&mut spec, solv_name, cal_window_ppm)?;
        solvent_cal_delta = Some(delta);
    }
    if let Some(off) = ref_offset_ppm {
        spec.apply_reference_offset(off);
        solvent_cal_delta = Some(solvent_cal_delta.unwrap_or(0.0) + off);
    }

    let pick_cfg = PeakPickConfig1D {
        min_snr,
        detect_shoulders,
        ..PeakPickConfig1D::default()
    };
    let picked_peaks = pick_peaks_1d(&spec, &pick_cfg);

    let impurity_matches = if match_impurities_flag {
        if let Some(ref solv) = spec.solvent {
            let nuc = spec.experiment.nucleus();
            let mut found = Vec::new();
            for pk in &picked_peaks {
                for m in match_trace_impurities(solv, &nuc, pk.ppm, 0.04) {
                    found.push(CliImpurityMatch {
                        observed_ppm: pk.ppm,
                        compound: m.entry.compound.to_string(),
                        site: m.entry.site.to_string(),
                        solvent: m.entry.solvent.to_string(),
                        nucleus: m.entry.nucleus.to_string(),
                        reference_ppm: m.entry.ppm,
                        multiplicity: m.entry.multiplicity.to_string(),
                        delta_ppm: m.delta_ppm,
                    });
                }
            }
            found
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };

    let deconv_res = if deconvolve_flag {
        let dcfg = DeconvolveConfig1D {
            pick_config: pick_cfg,
            max_lm_iters,
            ..DeconvolveConfig1D::default()
        };
        if let Some((lo, hi)) = deconv_region {
            let cluster = deconvolve_region_1d(&spec, lo, hi, None, &dcfg);
            let fitted_spectrum: Vec<f64> = (0..spec.len())
                .map(|j| eval_pseudo_voigt_sum_1d(spec.ppm_at(j), &cluster.peaks))
                .collect();
            let residual_spectrum: Vec<f64> = spec
                .real
                .iter()
                .zip(fitted_spectrum.iter())
                .map(|(&y, &f)| y - f)
                .collect();
            Some(DeconvolutionResult1D {
                peaks: cluster.peaks.clone(),
                fitted_spectrum,
                residual_spectrum,
                residual_rmsd: cluster.residual_rmsd,
                clusters: vec![cluster],
            })
        } else {
            Some(deconvolve_spectrum_1d(&spec, &dcfg))
        }
    } else {
        None
    };

    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("failed to create {}: {e}", parent.display()))?;
        }
    }

    let out_ext = out_path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if out_ext == "json" {
        let js = serde_json::to_string_pretty(&spec)
            .map_err(|e| format!("failed to serialize Spectrum1D JSON: {e}"))?;
        fs::write(&out_path, js)
            .map_err(|e| format!("failed to write {}: {e}", out_path.display()))?;
    } else {
        write_spectrum_1d_tsv(
            &out_path,
            &spec,
            deconv_res.as_ref().map(|d| d.fitted_spectrum.as_slice()),
            deconv_res.as_ref().map(|d| d.residual_spectrum.as_slice()),
        )?;
    }

    if let Some(ref pk_path) = peaks_out {
        if let Some(parent) = pk_path.parent() {
            if !parent.as_os_str().is_empty() {
                let _ = fs::create_dir_all(parent);
            }
        }
        let mut pk_tsv = String::new();
        if let Some(ref d) = deconv_res {
            pk_tsv.push_str(
                "index\tcenter_ppm\tfwhm_ppm\tfwhm_hz\tlorentz_fraction\tarea_ppm\tarea_hz\theight\n",
            );
            for (idx, vp) in d.peaks.iter().enumerate() {
                pk_tsv.push_str(&format!(
                    "{idx}\t{:.6}\t{:.6}\t{:.4}\t{:.4}\t{:.8e}\t{:.8e}\t{:.8e}\n",
                    vp.center_ppm,
                    vp.fwhm_ppm,
                    vp.fwhm_hz,
                    vp.lorentz_fraction,
                    vp.area_ppm,
                    vp.area_hz,
                    vp.height
                ));
            }
        } else {
            pk_tsv.push_str("index\tppm\theight\tsnr\test_fwhm_ppm\test_fwhm_hz\tis_shoulder\n");
            for (idx, pk) in picked_peaks.iter().enumerate() {
                pk_tsv.push_str(&format!(
                    "{idx}\t{:.6}\t{:.8e}\t{:.2}\t{:.6}\t{:.4}\t{}\n",
                    pk.ppm, pk.height, pk.snr, pk.est_fwhm_ppm, pk.est_fwhm_hz, pk.is_shoulder
                ));
            }
        }
        fs::write(pk_path, pk_tsv)
            .map_err(|e| format!("failed to write {}: {e}", pk_path.display()))?;
    }

    let summary = Process1dCliReport {
        name: spec.name.clone(),
        nucleus: spec.experiment.nucleus().as_str().to_string(),
        obs_mhz: spec.axis.obs_mhz,
        num_points: spec.len(),
        first_ppm: spec.axis.first_ppm,
        last_ppm: spec.axis.last_ppm(),
        solvent: spec.solvent.clone(),
        hsvd_poles_subtracted: hsvd_subtracted.len(),
        hsvd_subtracted,
        phase_applied_deg: spec.phase_applied,
        auto_phase: auto_phase_res,
        baseline_noise_sigma,
        solvent_calibration_delta_ppm: solvent_cal_delta,
        picked_peaks,
        impurity_matches,
        deconvolved_peaks: deconv_res
            .as_ref()
            .map(|d| d.peaks.clone())
            .unwrap_or_default(),
        deconvolve_residual_rmsd: deconv_res.as_ref().map(|d| d.residual_rmsd),
    };

    if let Some(ref js_path) = json_out {
        if let Some(parent) = js_path.parent() {
            if !parent.as_os_str().is_empty() {
                let _ = fs::create_dir_all(parent);
            }
        }
        let js = serde_json::to_string_pretty(&summary)
            .map_err(|e| format!("failed to serialize process summary JSON: {e}"))?;
        fs::write(js_path, js)
            .map_err(|e| format!("failed to write {}: {e}", js_path.display()))?;
    }

    println!(
        "nmr1d process: {} ({}, {:.2} MHz, {} pts, {:.3}..{:.3} ppm) -> {} (picked={}, deconvolved={})",
        summary.name,
        summary.nucleus,
        summary.obs_mhz,
        summary.num_points,
        summary.first_ppm,
        summary.last_ppm,
        out_path.display(),
        summary.picked_peaks.len(),
        summary.deconvolved_peaks.len()
    );
    Ok(())
}

// ============================================================================
// Subcommand 2: `nmrassign nmr1d multiplet`
// ============================================================================

/// Structured JSON output of `nmrassign nmr1d multiplet --json-out`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultipletCliReport {
    pub spectrum_name: String,
    pub nucleus: ReportNucleus,
    pub mhz: f64,
    pub solvent: String,
    pub noise_rms: f64,
    pub scale_factor: f64,
    pub integrated_regions: Vec<IntegratedRegion>,
    pub multiplets: Vec<MultipletAnalysis>,
    pub journal_report: JournalNmrReport,
    pub formatted_primary: String,
    pub formatted_styles: BTreeMap<String, String>,
    pub assignment_table: Option<MultipletAssignmentTable>,
}

fn parse_region_arg(spec: &str, baseline: BaselineMode) -> Result<IntegralRegionSpec, String> {
    let parts: Vec<&str> = spec.split(':').map(str::trim).collect();
    if parts.len() < 2 || parts.len() > 3 {
        return Err(format!(
            "region specification expects `<ppm_a>:<ppm_b>[:<label>]`, got {spec:?}"
        ));
    }
    let a = parts[0]
        .parse::<f64>()
        .map_err(|_| format!("invalid first ppm in region {spec:?}"))?;
    let b = parts[1]
        .parse::<f64>()
        .map_err(|_| format!("invalid second ppm in region {spec:?}"))?;
    let mut r = IntegralRegionSpec::new(a, b).with_baseline(baseline);
    if let Some(lbl) = parts.get(2) {
        if !lbl.is_empty() {
            r = r.with_label(*lbl);
        }
    }
    Ok(r)
}

fn auto_discover_multiplet_regions(
    spec: &Spectrum1D,
    baseline: BaselineMode,
) -> Vec<IntegralRegionSpec> {
    let pks = pick_peaks_1d(spec, &PeakPickConfig1D::default());
    if pks.is_empty() {
        return Vec::new();
    }
    let mhz = spec.axis.obs_mhz.max(1e-6);
    let merge_ppm = (45.0 / mhz).max(0.06);
    let pad_ppm = (18.0 / mhz).max(0.035);

    // `pks` are sorted high-to-low ppm.
    let mut clusters: Vec<(f64, f64)> = Vec::new();
    for pk in &pks {
        let lo = pk.ppm - pad_ppm;
        let hi = pk.ppm + pad_ppm;
        if let Some(last) = clusters.last_mut() {
            if last.0 - pk.ppm <= merge_ppm {
                last.0 = last.0.min(lo);
                last.1 = last.1.max(hi);
                continue;
            }
        }
        clusters.push((lo, hi));
    }

    let min_ax = spec.axis.min_ppm();
    let max_ax = spec.axis.max_ppm();
    clusters
        .into_iter()
        .filter_map(|(lo, hi)| {
            let clo = lo.max(min_ax);
            let chi = hi.min(max_ax);
            (chi - clo > 1e-5).then(|| IntegralRegionSpec::new(chi, clo).with_baseline(baseline))
        })
        .collect()
}

fn parse_publication_style(s: &str) -> Result<PublicationStyle, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "acs" => Ok(PublicationStyle::Acs),
        "joc" => Ok(PublicationStyle::Joc),
        "rsc" => Ok(PublicationStyle::Rsc),
        "nature" | "nat" => Ok(PublicationStyle::Nature),
        other => Err(format!(
            "unknown `--style` {other:?} (expected acs|joc|rsc|nature)"
        )),
    }
}

fn parse_text_encoding(s: &str) -> Result<TextEncoding, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "unicode" | "utf8" | "utf-8" => Ok(TextEncoding::Unicode),
        "ascii" | "plain" => Ok(TextEncoding::Ascii),
        "latex" | "tex" => Ok(TextEncoding::Latex),
        other => Err(format!(
            "unknown `--encoding` {other:?} (expected unicode|ascii|latex)"
        )),
    }
}

fn load_octet_molecule_arg(
    mol_path: Option<&Path>,
    smiles: Option<&str>,
) -> Result<Option<Molecule>, String> {
    if let Some(smi) = smiles {
        let mol =
            parse_smiles(smi).map_err(|e| format!("failed to parse `--smiles` {smi:?}: {e}"))?;
        return Ok(Some(mol));
    }
    if let Some(path) = mol_path {
        let text = fs::read_to_string(path)
            .map_err(|e| format!("failed to read structure {}: {e}", path.display()))?;
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if ext == "sdf" || text.contains("$$$$") {
            let recs = read_sdf(&text)
                .map_err(|e| format!("failed to parse SDF {}: {e}", path.display()))?;
            let first = recs
                .into_iter()
                .next()
                .ok_or_else(|| format!("SDF {} contained zero records", path.display()))?;
            return Ok(Some(first.molecule));
        }
        let mol =
            read_mol(&text).map_err(|e| format!("failed to parse MOL {}: {e}", path.display()))?;
        return Ok(Some(mol));
    }
    Ok(None)
}

fn run_nmr1d_multiplet(args: &[String]) -> Result<(), String> {
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        print!("{}", nmr1d_usage());
        return Ok(());
    }

    let mut in_path: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut json_out: Option<PathBuf> = None;
    let mut links_out: Option<PathBuf> = None;
    let mut report_out: Option<PathBuf> = None;

    let mut mhz_override: Option<f64> = None;
    let mut solvent_override: Option<String> = None;
    let mut nucleus_override: Option<ReportNucleus> = None;
    let mut proton_decoupled = false;

    let mut raw_regions: Vec<String> = Vec::new();
    let mut region_baseline = BaselineMode::default();
    let mut norm_mode: Option<NormalizationMode> = None;

    let mut style = PublicationStyle::Acs;
    let mut encoding = TextEncoding::Unicode;
    let mut mol_path: Option<PathBuf> = None;
    let mut smiles_arg: Option<String> = None;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--in" | "-i" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--in`")?;
                in_path = Some(PathBuf::from(v));
            }
            "--out" | "-o" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--out`")?;
                out_path = Some(PathBuf::from(v));
            }
            "--json-out" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--json-out`")?;
                json_out = Some(PathBuf::from(v));
            }
            "--links-out" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--links-out`")?;
                links_out = Some(PathBuf::from(v));
            }
            "--report-out" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--report-out`")?;
                report_out = Some(PathBuf::from(v));
            }
            "--mhz" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--mhz`")?;
                mhz_override = Some(
                    v.parse::<f64>()
                        .map_err(|_| format!("invalid `--mhz` {v:?}"))?,
                );
            }
            "--solvent" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--solvent`")?;
                solvent_override = Some(v.clone());
            }
            "--nucleus" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--nucleus`")?;
                nucleus_override = Some(
                    ReportNucleus::from_ascii_token(v)
                        .ok_or_else(|| format!("unsupported `--nucleus` {v:?}"))?,
                );
            }
            "--proton-decoupled" => {
                proton_decoupled = true;
            }
            "--region" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--region`")?;
                raw_regions.push(v.clone());
            }
            "--regions" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--regions`")?;
                for item in v.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                    raw_regions.push(item.to_string());
                }
            }
            "--region-baseline" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--region-baseline`")?;
                if v.eq_ignore_ascii_case("auto") {
                    region_baseline = BaselineMode::default();
                } else if v.eq_ignore_ascii_case("none") {
                    region_baseline = BaselineMode::None;
                } else if let Some((b_s, sl_s)) = v.split_once(':') {
                    let bias = b_s
                        .parse::<f64>()
                        .map_err(|_| format!("invalid bias in `--region-baseline` {v:?}"))?;
                    let slope = sl_s
                        .parse::<f64>()
                        .map_err(|_| format!("invalid slope in `--region-baseline` {v:?}"))?;
                    region_baseline = BaselineMode::Explicit { bias, slope };
                } else {
                    return Err(format!(
                        "invalid `--region-baseline` {v:?} (expected auto|none|<bias:slope>)"
                    ));
                }
            }
            "--norm-ref" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--norm-ref`")?;
                let (idx_s, h_s) = v.split_once(':').ok_or_else(|| {
                    format!("`--norm-ref` expects `<region_idx>:<protons>`, got {v:?}")
                })?;
                let region_index = idx_s
                    .parse::<usize>()
                    .map_err(|_| format!("invalid region_idx in `--norm-ref` {v:?}"))?;
                let target_protons = h_s
                    .parse::<f64>()
                    .map_err(|_| format!("invalid protons in `--norm-ref` {v:?}"))?;
                norm_mode = Some(NormalizationMode::ReferenceRegion {
                    region_index,
                    target_protons,
                });
            }
            "--total-protons" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--total-protons`")?;
                let total_protons = v
                    .parse::<f64>()
                    .map_err(|_| format!("invalid `--total-protons` {v:?}"))?;
                norm_mode = Some(NormalizationMode::TotalProtons { total_protons });
            }
            "--norm-scale" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--norm-scale`")?;
                let factor = v
                    .parse::<f64>()
                    .map_err(|_| format!("invalid `--norm-scale` {v:?}"))?;
                norm_mode = Some(NormalizationMode::ExplicitScale { factor });
            }
            "--style" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--style`")?;
                style = parse_publication_style(v)?;
            }
            "--encoding" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--encoding`")?;
                encoding = parse_text_encoding(v)?;
            }
            "--mol" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--mol`")?;
                mol_path = Some(PathBuf::from(v));
            }
            "--smiles" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--smiles`")?;
                smiles_arg = Some(v.clone());
            }
            other => {
                return Err(format!(
                    "unknown `nmr1d multiplet` option {other:?}\n\n{}",
                    nmr1d_usage()
                ));
            }
        }
        i += 1;
    }

    let in_path = in_path.ok_or("`nmr1d multiplet` requires `--in <path>`")?;
    let spec = load_processed_spectrum_1d(&in_path, mhz_override)?;
    let mhz = mhz_override.unwrap_or(spec.axis.obs_mhz).max(1e-6);
    let solvent = solvent_override
        .or_else(|| spec.solvent.clone())
        .unwrap_or_else(|| "CDCl3".to_string());
    let rep_nucleus = nucleus_override
        .or_else(|| ReportNucleus::from_ascii_token(spec.experiment.nucleus().as_str()))
        .unwrap_or(ReportNucleus::H1);
    if matches!(spec.experiment, Experiment1DKind::Carbon13Decoupled) {
        proton_decoupled = true;
    }

    let mol_opt = load_octet_molecule_arg(mol_path.as_deref(), smiles_arg.as_deref())?;

    let region_specs: Vec<IntegralRegionSpec> = if raw_regions.is_empty() {
        auto_discover_multiplet_regions(&spec, region_baseline)
    } else {
        raw_regions
            .iter()
            .map(|s| parse_region_arg(s, region_baseline))
            .collect::<Result<Vec<_>, _>>()?
    };

    if region_specs.is_empty() {
        return Err("no signal regions detected or specified for `nmr1d multiplet`".to_string());
    }

    // Default normalization: if a molecule was supplied and no explicit normalization flag was passed,
    // normalize to the molecule's total non-exchangeable/implicit+explicit 1H count on 1H spectra.
    let active_norm = if let Some(nm) = norm_mode {
        nm
    } else if rep_nucleus == ReportNucleus::H1 {
        if let Some(ref mol) = mol_opt {
            let total_h: usize = (0..mol.atoms.len())
                .map(|idx| {
                    if mol.atoms[idx].z == 1 {
                        1usize
                    } else {
                        mol.implicit_hydrogens(idx) as usize
                    }
                })
                .sum();
            if total_h > 0 {
                NormalizationMode::TotalProtons {
                    total_protons: total_h as f64,
                }
            } else {
                NormalizationMode::None
            }
        } else {
            NormalizationMode::None
        }
    } else {
        NormalizationMode::None
    };

    let ppm_vec: Vec<f64> = (0..spec.len()).map(|idx| spec.ppm_at(idx)).collect();
    let int_report =
        integrate_regions(&ppm_vec, &spec.real, &region_specs, active_norm, None, None)
            .map_err(|e| format!("integration failed: {e}"))?;

    let mcfg = MultipletConfig::default();
    let is_normalized = !matches!(active_norm, NormalizationMode::None);

    let mut analyses: Vec<MultipletAnalysis> = Vec::with_capacity(int_report.regions.len());
    let mut reported_signals: Vec<ReportedSignal> = Vec::with_capacity(int_report.regions.len());

    for reg in &int_report.regions {
        let mut ma =
            analyze_multiplet_region(&ppm_vec, &spec.real, reg.ppm_low, reg.ppm_high, mhz, &mcfg)
                .map_err(|e| {
                format!(
                    "multiplet analysis failed in [{:.3}, {:.3}] ppm: {e}",
                    reg.ppm_low, reg.ppm_high
                )
            })?;
        let h_count = if is_normalized && reg.nearest_integer_h > 0 {
            Some(reg.nearest_integer_h)
        } else {
            None
        };
        ma.proton_count = h_count;
        let sig = if rep_nucleus == ReportNucleus::C13 && proton_decoupled {
            let mut ds = ReportedSignal::new_decoupled(ma.center_ppm);
            ds.assignment = reg.label.clone();
            ds
        } else {
            ReportedSignal::from_analysis(&ma, h_count, reg.label.clone())
        };
        reported_signals.push(sig);
        analyses.push(ma);
    }

    let assignment_table = mol_opt.as_ref().map(|mol| {
        let tbl = auto_assign_1h_multiplets(mol, &analyses, Some(&solvent));
        // Backfill assignment labels onto `reported_signals` when unlabeled.
        for link in &tbl.links {
            if let Some(sig) = reported_signals.get_mut(link.multiplet_id) {
                if sig.assignment.is_none() {
                    sig.assignment = link.annotation.clone();
                }
            }
        }
        tbl
    });

    let journal_report = JournalNmrReport {
        nucleus: rep_nucleus,
        proton_decoupled,
        mhz,
        solvent: solvent.clone(),
        signals: reported_signals,
    };

    let formatted_primary = format_journal_nmr_report(&journal_report, style, encoding);
    let mut formatted_styles = BTreeMap::new();
    for (lbl, st) in [
        ("acs", PublicationStyle::Acs),
        ("joc", PublicationStyle::Joc),
        ("rsc", PublicationStyle::Rsc),
        ("nature", PublicationStyle::Nature),
    ] {
        formatted_styles.insert(
            lbl.to_string(),
            format_journal_nmr_report(&journal_report, st, encoding),
        );
    }

    if let Some(ref tsv_path) = out_path {
        if let Some(parent) = tsv_path.parent() {
            if !parent.as_os_str().is_empty() {
                let _ = fs::create_dir_all(parent);
            }
        }
        let mut tsv = String::from(
            "index\tppm_high\tppm_low\tcenter_ppm\tclass\tj_hz\tabs_integral\tnorm_integral\tnorm_sigma\tnearest_h\tis_roofed\tinner_outer_ratio\tlabel\n",
        );
        for (reg, ma) in int_report.regions.iter().zip(analyses.iter()) {
            let j_csv: Vec<String> = ma
                .j_values_hz()
                .into_iter()
                .map(|j| format!("{j:.2}"))
                .collect();
            tsv.push_str(&format!(
                "{}\t{:.4}\t{:.4}\t{:.4}\t{}\t{}\t{:.6e}\t{:.4}\t{:.4}\t{}\t{}\t{:.3}\t{}\n",
                reg.index,
                reg.ppm_high,
                reg.ppm_low,
                ma.center_ppm,
                ma.label,
                j_csv.join(","),
                reg.abs_integral,
                reg.norm_integral,
                reg.norm_sigma,
                reg.nearest_integer_h,
                ma.leaning.is_roofed,
                ma.leaning.inner_to_outer_ratio,
                reg.label.as_deref().unwrap_or("")
            ));
        }
        fs::write(tsv_path, tsv)
            .map_err(|e| format!("failed to write {}: {e}", tsv_path.display()))?;
    }

    if let Some(ref lpath) = links_out {
        let tbl = assignment_table
            .as_ref()
            .ok_or("`--links-out` requires `--mol <file>` or `--smiles <SMILES>`")?;
        if let Some(parent) = lpath.parent() {
            if !parent.as_os_str().is_empty() {
                let _ = fs::create_dir_all(parent);
            }
        }
        let lext = lpath
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if lext == "json" {
            let js = serde_json::to_string_pretty(tbl)
                .map_err(|e| format!("failed to serialize assignment table JSON: {e}"))?;
            fs::write(lpath, js)
                .map_err(|e| format!("failed to write {}: {e}", lpath.display()))?;
        } else {
            fs::write(lpath, format!("{}\n", tbl.to_tsv()))
                .map_err(|e| format!("failed to write {}: {e}", lpath.display()))?;
        }
    }

    if let Some(ref rpath) = report_out {
        if let Some(parent) = rpath.parent() {
            if !parent.as_os_str().is_empty() {
                let _ = fs::create_dir_all(parent);
            }
        }
        fs::write(rpath, format!("{formatted_primary}\n"))
            .map_err(|e| format!("failed to write {}: {e}", rpath.display()))?;
    }

    let full_report = MultipletCliReport {
        spectrum_name: spec.name,
        nucleus: rep_nucleus,
        mhz,
        solvent,
        noise_rms: int_report.noise_rms,
        scale_factor: int_report.scale_factor,
        integrated_regions: int_report.regions,
        multiplets: analyses,
        journal_report,
        formatted_primary: formatted_primary.clone(),
        formatted_styles,
        assignment_table,
    };

    if let Some(ref js_path) = json_out {
        if let Some(parent) = js_path.parent() {
            if !parent.as_os_str().is_empty() {
                let _ = fs::create_dir_all(parent);
            }
        }
        let js = serde_json::to_string_pretty(&full_report)
            .map_err(|e| format!("failed to serialize multiplet JSON: {e}"))?;
        fs::write(js_path, js)
            .map_err(|e| format!("failed to write {}: {e}", js_path.display()))?;
    }

    println!("{formatted_primary}");
    Ok(())
}

// ============================================================================
// Subcommand 3: `nmrassign nmr1d qnmr`
// ============================================================================

/// Multi-tube replicate schema accepted by `nmrassign nmr1d qnmr --replicates-json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QnmrReplicateTubeInput {
    pub analyte: QnmrAnalyteSpec,
    pub standard: QnmrStandardSpec,
}

#[derive(Debug, Clone)]
struct AnalyteRegionCliSpec {
    ppm_a: f64,
    ppm_b: f64,
    protons_n: usize,
    label: Option<String>,
    t1_s: Option<f64>,
}

fn parse_analyte_region_arg(s: &str) -> Result<AnalyteRegionCliSpec, String> {
    // `<ppm_a>:<ppm_b>:<protons>[:<label>][:<t1_s>]`
    let parts: Vec<&str> = s.split(':').map(str::trim).collect();
    if parts.len() < 3 || parts.len() > 5 {
        return Err(format!(
            "`--analyte-region` expects `<ppm_a>:<ppm_b>:<protons>[:<label>][:<t1_s>]`, got {s:?}"
        ));
    }
    let ppm_a = parts[0]
        .parse::<f64>()
        .map_err(|_| format!("invalid ppm_a in `--analyte-region` {s:?}"))?;
    let ppm_b = parts[1]
        .parse::<f64>()
        .map_err(|_| format!("invalid ppm_b in `--analyte-region` {s:?}"))?;
    let protons_n = parts[2]
        .parse::<usize>()
        .map_err(|_| format!("invalid protons in `--analyte-region` {s:?}"))?;
    let label = parts
        .get(3)
        .filter(|v| !v.is_empty())
        .map(|v| (*v).to_string());
    let t1_s = match parts.get(4).filter(|v| !v.is_empty()) {
        Some(v) => Some(
            v.parse::<f64>()
                .map_err(|_| format!("invalid t1_s in `--analyte-region` {s:?}"))?,
        ),
        None => None,
    };
    Ok(AnalyteRegionCliSpec {
        ppm_a,
        ppm_b,
        protons_n,
        label,
        t1_s,
    })
}

fn parse_direct_signal_arg(s: &str) -> Result<QnmrAnalyteSignal, String> {
    // `<protons>:<integral>[:<sigma>][:<label>][:<t1_s>]`
    let parts: Vec<&str> = s.split(':').map(str::trim).collect();
    if parts.len() < 2 || parts.len() > 5 {
        return Err(format!(
            "`--signal` expects `<protons>:<integral>[:<sigma>][:<label>][:<t1_s>]`, got {s:?}"
        ));
    }
    let protons_n = parts[0]
        .parse::<usize>()
        .map_err(|_| format!("invalid protons in `--signal` {s:?}"))?;
    let integral = parts[1]
        .parse::<f64>()
        .map_err(|_| format!("invalid integral in `--signal` {s:?}"))?;
    let sigma = match parts.get(2).filter(|v| !v.is_empty()) {
        Some(v) => v
            .parse::<f64>()
            .map_err(|_| format!("invalid sigma in `--signal` {s:?}"))?,
        None => 0.0,
    };
    let mut sig = QnmrAnalyteSignal::new(protons_n, integral, sigma);
    sig.label = parts
        .get(3)
        .filter(|v| !v.is_empty())
        .map(|v| (*v).to_string());
    if let Some(t1_str) = parts.get(4).filter(|v| !v.is_empty()) {
        let t1 = t1_str
            .parse::<f64>()
            .map_err(|_| format!("invalid t1_s in `--signal` {s:?}"))?;
        sig.t1_s = Some(t1);
    }
    Ok(sig)
}

fn run_nmr1d_qnmr(args: &[String]) -> Result<(), String> {
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        print!("{}", nmr1d_usage());
        return Ok(());
    }

    let mut in_path: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut json_out: Option<PathBuf> = None;
    let mut replicates_json: Option<PathBuf> = None;

    let mut std_crm: Option<String> = None;
    let mut std_name: Option<String> = None;
    let mut std_mw: Option<f64> = None;
    let mut std_mw_sigma: f64 = 0.0;
    let mut std_mass_mg: Option<f64> = None;
    let mut std_mass_sigma_mg: f64 = 0.0;
    let mut std_purity_frac: Option<f64> = None;
    let mut std_purity_sigma_frac: f64 = 0.0;
    let mut std_protons: Option<usize> = None;
    let mut std_integral: Option<f64> = None;
    let mut std_integral_sigma: f64 = 0.0;
    let mut std_t1_s: Option<f64> = None;
    let mut std_region: Option<(f64, f64)> = None;

    let mut analyte_name = String::from("Analyte");
    let mut analyte_mw: Option<f64> = None;
    let mut analyte_mw_sigma: f64 = 0.0;
    let mut analyte_mass_mg: Option<f64> = None;
    let mut analyte_mass_sigma_mg: f64 = 0.0;
    let mut volume_ml: Option<f64> = None;
    let mut volume_sigma_ml: Option<f64> = None;

    let mut direct_signals: Vec<QnmrAnalyteSignal> = Vec::new();
    let mut analyte_regions: Vec<AnalyteRegionCliSpec> = Vec::new();

    let mut d1_s: Option<f64> = None;
    let mut aq_s: f64 = 3.0;
    let mut flip_deg: f64 = 90.0;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--in" | "-i" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--in`")?;
                in_path = Some(PathBuf::from(v));
            }
            "--out" | "-o" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--out`")?;
                out_path = Some(PathBuf::from(v));
            }
            "--json-out" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--json-out`")?;
                json_out = Some(PathBuf::from(v));
            }
            "--replicates-json" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--replicates-json`")?;
                replicates_json = Some(PathBuf::from(v));
            }
            "--std-crm" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--std-crm`")?;
                std_crm = Some(v.clone());
            }
            "--std-name" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--std-name`")?;
                std_name = Some(v.clone());
            }
            "--std-mw" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--std-mw`")?;
                std_mw = Some(
                    v.parse::<f64>()
                        .map_err(|_| format!("invalid `--std-mw` {v:?}"))?,
                );
            }
            "--std-mw-sigma" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--std-mw-sigma`")?;
                std_mw_sigma = v
                    .parse::<f64>()
                    .map_err(|_| format!("invalid `--std-mw-sigma` {v:?}"))?;
            }
            "--std-mass-mg" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--std-mass-mg`")?;
                std_mass_mg = Some(
                    v.parse::<f64>()
                        .map_err(|_| format!("invalid `--std-mass-mg` {v:?}"))?,
                );
            }
            "--std-mass-sigma-mg" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or("missing value for `--std-mass-sigma-mg`")?;
                std_mass_sigma_mg = v
                    .parse::<f64>()
                    .map_err(|_| format!("invalid `--std-mass-sigma-mg` {v:?}"))?;
            }
            "--std-purity" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--std-purity`")?;
                let p = v
                    .parse::<f64>()
                    .map_err(|_| format!("invalid `--std-purity` {v:?}"))?;
                std_purity_frac = Some(if p > 1.000_001 { p / 100.0 } else { p });
            }
            "--std-purity-pct" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--std-purity-pct`")?;
                let p = v
                    .parse::<f64>()
                    .map_err(|_| format!("invalid `--std-purity-pct` {v:?}"))?;
                std_purity_frac = Some(p / 100.0);
            }
            "--std-purity-sigma" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or("missing value for `--std-purity-sigma`")?;
                std_purity_sigma_frac = v
                    .parse::<f64>()
                    .map_err(|_| format!("invalid `--std-purity-sigma` {v:?}"))?;
            }
            "--std-purity-sigma-pct" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or("missing value for `--std-purity-sigma-pct`")?;
                let p = v
                    .parse::<f64>()
                    .map_err(|_| format!("invalid `--std-purity-sigma-pct` {v:?}"))?;
                std_purity_sigma_frac = p / 100.0;
            }
            "--std-protons" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--std-protons`")?;
                std_protons = Some(
                    v.parse::<usize>()
                        .map_err(|_| format!("invalid `--std-protons` {v:?}"))?,
                );
            }
            "--std-integral" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--std-integral`")?;
                std_integral = Some(
                    v.parse::<f64>()
                        .map_err(|_| format!("invalid `--std-integral` {v:?}"))?,
                );
            }
            "--std-integral-sigma" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or("missing value for `--std-integral-sigma`")?;
                std_integral_sigma = v
                    .parse::<f64>()
                    .map_err(|_| format!("invalid `--std-integral-sigma` {v:?}"))?;
            }
            "--std-t1" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--std-t1`")?;
                std_t1_s = Some(
                    v.parse::<f64>()
                        .map_err(|_| format!("invalid `--std-t1` {v:?}"))?,
                );
            }
            "--std-region" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--std-region`")?;
                std_region = Some(parse_ppm_pair(v, "--std-region")?);
            }
            "--analyte-name" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--analyte-name`")?;
                analyte_name = v.clone();
            }
            "--analyte-mw" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--analyte-mw`")?;
                analyte_mw = Some(
                    v.parse::<f64>()
                        .map_err(|_| format!("invalid `--analyte-mw` {v:?}"))?,
                );
            }
            "--analyte-mw-sigma" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or("missing value for `--analyte-mw-sigma`")?;
                analyte_mw_sigma = v
                    .parse::<f64>()
                    .map_err(|_| format!("invalid `--analyte-mw-sigma` {v:?}"))?;
            }
            "--analyte-mass-mg" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--analyte-mass-mg`")?;
                analyte_mass_mg = Some(
                    v.parse::<f64>()
                        .map_err(|_| format!("invalid `--analyte-mass-mg` {v:?}"))?,
                );
            }
            "--analyte-mass-sigma-mg" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or("missing value for `--analyte-mass-sigma-mg`")?;
                analyte_mass_sigma_mg = v
                    .parse::<f64>()
                    .map_err(|_| format!("invalid `--analyte-mass-sigma-mg` {v:?}"))?;
            }
            "--volume-ml" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--volume-ml`")?;
                volume_ml = Some(
                    v.parse::<f64>()
                        .map_err(|_| format!("invalid `--volume-ml` {v:?}"))?,
                );
            }
            "--volume-sigma-ml" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--volume-sigma-ml`")?;
                volume_sigma_ml = Some(
                    v.parse::<f64>()
                        .map_err(|_| format!("invalid `--volume-sigma-ml` {v:?}"))?,
                );
            }
            "--signal" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--signal`")?;
                direct_signals.push(parse_direct_signal_arg(v)?);
            }
            "--analyte-region" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--analyte-region`")?;
                analyte_regions.push(parse_analyte_region_arg(v)?);
            }
            "--d1-s" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--d1-s`")?;
                d1_s = Some(
                    v.parse::<f64>()
                        .map_err(|_| format!("invalid `--d1-s` {v:?}"))?,
                );
            }
            "--aq-s" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--aq-s`")?;
                aq_s = v
                    .parse::<f64>()
                    .map_err(|_| format!("invalid `--aq-s` {v:?}"))?;
            }
            "--flip-deg" => {
                i += 1;
                let v = args.get(i).ok_or("missing value for `--flip-deg`")?;
                flip_deg = v
                    .parse::<f64>()
                    .map_err(|_| format!("invalid `--flip-deg` {v:?}"))?;
            }
            other => {
                return Err(format!(
                    "unknown `nmr1d qnmr` option {other:?}\n\n{}",
                    nmr1d_usage()
                ));
            }
        }
        i += 1;
    }

    let acq_opt = d1_s.map(|d1| QnmrAcquisitionParams {
        recycle_delay_d1_s: d1,
        acquisition_time_aq_s: aq_s,
        pulse_flip_angle_deg: flip_deg,
    });

    let report: QnmrPurityReport = if let Some(ref rep_path) = replicates_json {
        let txt = fs::read_to_string(rep_path)
            .map_err(|e| format!("failed to read {}: {e}", rep_path.display()))?;
        let tubes_vec: Vec<QnmrReplicateTubeInput> = serde_json::from_str(&txt)
            .map_err(|e| format!("invalid replicates JSON in {}: {e}", rep_path.display()))?;
        let pairs: Vec<(QnmrAnalyteSpec, QnmrStandardSpec)> = tubes_vec
            .into_iter()
            .map(|t| (t.analyte, t.standard))
            .collect();
        calculate_qnmr_replicates(&pairs, acq_opt.as_ref())
            .map_err(|e| format!("qNMR replicate calculation failed: {e}"))?
    } else {
        if let Some(ref crm_key) = std_crm {
            let crm = qnmr_certified_standards()
                .iter()
                .find(|c| {
                    c.abbreviation.eq_ignore_ascii_case(crm_key)
                        || c.name.eq_ignore_ascii_case(crm_key)
                })
                .ok_or_else(|| {
                    format!(
                        "unknown `--std-crm` {crm_key:?} (supported: DMSO2, MA, TMB, DMT, BA, DNBA)"
                    )
                })?;
            if std_name.is_none() {
                std_name = Some(crm.name.to_string());
            }
            if std_mw.is_none() {
                std_mw = Some(crm.mw_g_mol);
            }
            if std_protons.is_none() {
                std_protons = Some(crm.protons_n);
            }
            if std_t1_s.is_none() {
                std_t1_s = Some(crm.typical_t1_s);
            }
        }

        // Integrate directly from 1D spectrum if `--in` is provided.
        if let Some(ref sp_path) = in_path {
            let spec = load_processed_spectrum_1d(sp_path, None)?;
            let ppm_vec: Vec<f64> = (0..spec.len()).map(|idx| spec.ppm_at(idx)).collect();
            let mut reg_specs = Vec::new();
            let has_std_reg = if let Some((lo, hi)) = std_region {
                reg_specs.push(IntegralRegionSpec::new(hi, lo).with_label("InternalStd"));
                true
            } else {
                false
            };
            for ar in &analyte_regions {
                let mut rs = IntegralRegionSpec::new(ar.ppm_a, ar.ppm_b);
                if let Some(ref lbl) = ar.label {
                    rs = rs.with_label(lbl.clone());
                }
                reg_specs.push(rs);
            }
            if !reg_specs.is_empty() {
                let irep = integrate_regions(
                    &ppm_vec,
                    &spec.real,
                    &reg_specs,
                    NormalizationMode::None,
                    None,
                    None,
                )
                .map_err(|e| format!("qNMR spectral integration failed: {e}"))?;
                let mut offset = 0usize;
                if has_std_reg {
                    let sreg = &irep.regions[0];
                    std_integral = Some(sreg.abs_integral);
                    std_integral_sigma = sreg.abs_sigma;
                    offset = 1;
                }
                for (ar, reg) in analyte_regions.iter().zip(irep.regions.iter().skip(offset)) {
                    let mut sig =
                        QnmrAnalyteSignal::new(ar.protons_n, reg.abs_integral, reg.abs_sigma);
                    sig.label = ar.label.clone();
                    sig.shift_ppm = Some(0.5 * (reg.ppm_low + reg.ppm_high));
                    sig.t1_s = ar.t1_s;
                    direct_signals.push(sig);
                }
            }
        }

        let s_name = std_name.unwrap_or_else(|| "Internal Standard".to_string());
        let s_mw = std_mw.ok_or("`nmr1d qnmr` requires `--std-mw <g_mol>` or `--std-crm <CRM>`")?;
        let s_mass = std_mass_mg.ok_or("`nmr1d qnmr` requires `--std-mass-mg <mg>`")?;
        let s_pur = std_purity_frac.unwrap_or(1.0);
        let s_prot =
            std_protons.ok_or("`nmr1d qnmr` requires `--std-protons <N>` or `--std-crm <CRM>`")?;
        let s_int = std_integral
            .ok_or("`nmr1d qnmr` requires `--std-integral <I>` or (`--in <spec>` + `--std-region <lo:hi>`)")?;

        let a_mw = analyte_mw.ok_or("`nmr1d qnmr` requires `--analyte-mw <g_mol>`")?;
        let a_mass = analyte_mass_mg.ok_or("`nmr1d qnmr` requires `--analyte-mass-mg <mg>`")?;
        if direct_signals.is_empty() {
            return Err(
                "`nmr1d qnmr` requires at least one `--signal <H:integral...>` or `--analyte-region <lo:hi:H...>`"
                    .to_string(),
            );
        }

        let standard = QnmrStandardSpec {
            name: s_name,
            mw_g_mol: s_mw,
            mw_sigma_g_mol: std_mw_sigma,
            mass_mg: s_mass,
            mass_sigma_mg: std_mass_sigma_mg,
            purity_fraction: s_pur,
            purity_sigma_fraction: std_purity_sigma_frac,
            protons_n: s_prot,
            integral: s_int,
            integral_sigma: std_integral_sigma,
            t1_s: std_t1_s,
        };
        let analyte = QnmrAnalyteSpec {
            name: analyte_name.clone(),
            mw_g_mol: a_mw,
            mw_sigma_g_mol: analyte_mw_sigma,
            mass_mg: a_mass,
            mass_sigma_mg: analyte_mass_sigma_mg,
            signals: direct_signals,
            solution_volume_ml: volume_ml,
            solution_volume_sigma_ml: volume_sigma_ml,
        };

        calculate_qnmr_purity(&analyte, &standard, acq_opt.as_ref())
            .map_err(|e| format!("qNMR calculation failed: {e}"))?
    };

    if let Some(ref tsv_path) = out_path {
        if let Some(parent) = tsv_path.parent() {
            if !parent.as_os_str().is_empty() {
                let _ = fs::create_dir_all(parent);
            }
        }
        let mut tsv = String::from(
            "analyte\tpurity_percent\tcombined_uncertainty_percent\texpanded_uncertainty_k2_percent\tmolar_concentration_mm\treplicate_count\treplicate_rsd_percent\tt1_recycle_warning\n",
        );
        tsv.push_str(&format!(
            "{}\t{:.4}\t{:.4}\t{:.4}\t{}\t{}\t{}\t{}\n",
            analyte_name,
            report.purity_percent,
            report.combined_uncertainty_percent,
            report.expanded_uncertainty_k2_percent,
            report
                .molar_concentration_mm
                .map(|c| format!("{c:.4}"))
                .unwrap_or_default(),
            report.replicate_count,
            report
                .replicate_rsd_percent
                .map(|r| format!("{r:.4}"))
                .unwrap_or_default(),
            report.t1_recycle_warning
        ));
        fs::write(tsv_path, tsv)
            .map_err(|e| format!("failed to write {}: {e}", tsv_path.display()))?;
    }

    if let Some(ref js_path) = json_out {
        if let Some(parent) = js_path.parent() {
            if !parent.as_os_str().is_empty() {
                let _ = fs::create_dir_all(parent);
            }
        }
        let js = serde_json::to_string_pretty(&report)
            .map_err(|e| format!("failed to serialize qNMR report JSON: {e}"))?;
        fs::write(js_path, js)
            .map_err(|e| format!("failed to write {}: {e}", js_path.display()))?;
    }

    println!(
        "nmr1d qnmr: {} purity = {:.3}% +/- {:.3}% (k=2, 95% CI; u_c = {:.3}%){}{}",
        analyte_name,
        report.purity_percent,
        report.expanded_uncertainty_k2_percent,
        report.combined_uncertainty_percent,
        report
            .molar_concentration_mm
            .map(|c| format!(", [C] = {c:.3} mM"))
            .unwrap_or_default(),
        if report.t1_recycle_warning {
            " [WARNING: T_rep < 5*T1]"
        } else {
            ""
        }
    );
    Ok(())
}
