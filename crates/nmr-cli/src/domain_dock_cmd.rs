//! CLI implementation of `ferric structure --pae-domains` (`T21.5`,
//! `docs/specs/21-alphafold-integrative-nmr.md` §5.3):
//! Symmetrized `PAE` normalized-Laplacian spectral domain clustering, Tolman $\vartheta_d / \vartheta_1$
//! Generalized Degree of Order (`GDO`) inter-domain motion testing, and batched WebGPU / CPU
//! multi-domain `RDC` / `PCS` / `PRE` / inter-domain `NOE` covalent linker docking.
//!
//! Provenance: Clean-room CLI integration citing Ng, Jordan & Weiss (*NIPS* **2001**, 14, 849-856),
//! Tolman et al. (*J. Am. Chem. Soc.* **2001**, 123, 1416-1424), Braddock et al. (*J. Am. Chem. Soc.*
//! **2001**, 123, 8634-8635), Clore & Schwieters (*J. Am. Chem. Soc.* **2004**, 126, 2923-2938),
//! and Bertini et al. (*J. Am. Chem. Soc.* **2007**, 129, 12786-12794).

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use nmr_core::af_concordance::{
    parse_af_structure_model, parse_pae_json_file, AfIngestOptions, AfResidueRecord,
};
use nmr_core::domain_dock::{
    dock_domains_multi_chain_cpu, format_domain_dock_json, format_domain_dock_tsv,
    multi_chain_backbone_rmsd, read_domain_noe_tsv, read_domain_pcs_tsv,
    read_domain_pre_tsv_with_defaults, read_domain_rdc_tsv, write_docked_mmcif_ensemble,
    write_docked_pdb_ensemble, DomainDockConfig, DomainDockResult, DomainPcsTagSet,
    DomainPreSiteSet, DomainRdcMediumSet,
};
use nmr_core::nef::read_nef_file;
use nmr_core::noesy::AmbiguousDistanceRestraint;
use nmr_gpu::domain_dock_gpu::dock_domains_multi_chain_gpu;
use nmr_gpu::GpuEngine;

use crate::{parse_engine_request, EngineRequest};

pub const STRUCTURE_PAE_DOMAINS_USAGE: &str = "Usage: ferric structure --pae-domains --af-structure <model.cif|pdb> --af-pae <pae.json> [--chain <ID[,ID2,...]>] [--rdc <rdcs.tsv|nef>] [--pcs <pcs.tsv>] [--pre <pre.tsv>] [--pre-tau-c-ns <ns>] [--pre-proton-mhz <MHz>] [--mtsl-tether-allowance-a <A>] [--refine-metal-pos] [--metal-search-radius <A>] [--noesy <inter_noe.tsv>] [--rdc-free-frac 0.20] [--pcs-free-frac 0.20] [--allow-poor-intra-domain-fit] [--allow-underdetermined-single-medium] [--ensemble-size 10] [--replicas 64] [--engine auto|gpu|cpu] [--pdb-ref <ref.pdb|cif>] [--json-out <report.json>] [--tsv-out <report.tsv>] [--seed 42] -o <docked.pdb|docked.cif>";

fn load_rdc_file_auto(path: &Path) -> Result<Vec<DomainRdcMediumSet>, String> {
    let is_nef = path
        .extension()
        .and_then(|s| s.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("nef"));
    let media = if is_nef {
        DomainRdcMediumSet::from_rdc_media(&read_nef_file(path)?.rdc_media()?)
    } else {
        read_domain_rdc_tsv(path)?
    };
    if media.is_empty() {
        return Err(format!(
            "RDC input '{}' contains zero alignment media / restraints",
            path.display()
        ));
    }
    Ok(media)
}

fn default_sidecar_report_paths(out_coord: &Path) -> (PathBuf, PathBuf) {
    let parent = out_coord.parent().unwrap_or_else(|| Path::new("."));
    let stem = out_coord
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("docked");
    (
        parent.join(format!("{stem}.domain_dock.json")),
        parent.join(format!("{stem}.domain_dock.tsv")),
    )
}

fn discover_pdb_polymer_chain_ids(path: &Path) -> Vec<String> {
    let Ok(text) = fs::read_to_string(path) else {
        return Vec::new();
    };
    let mut chains: Vec<String> = Vec::new();
    for line in text.lines() {
        if line.starts_with("ENDMDL") {
            break;
        }
        if line.starts_with("ATOM  ") && line.len() >= 22 {
            let b = line.as_bytes()[21];
            let cid = if b == b' ' || !b.is_ascii_graphic() {
                "A".to_string()
            } else {
                (b as char).to_string()
            };
            if !chains.iter().any(|c| c == &cid) {
                chains.push(cid);
            }
        }
    }
    chains
}

/// Execute `ferric structure --pae-domains ...`.
pub fn cmd_structure_pae_domains(args: &[String]) -> Result<(), String> {
    let mut af_struct_path: Option<PathBuf> = None;
    let mut af_pae_path: Option<PathBuf> = None;
    let mut rdc_paths: Vec<PathBuf> = Vec::new();
    let mut pcs_paths: Vec<PathBuf> = Vec::new();
    let mut pre_paths: Vec<PathBuf> = Vec::new();
    let mut noesy_paths: Vec<PathBuf> = Vec::new();
    let mut out_path: Option<PathBuf> = None;
    let mut json_out_path: Option<PathBuf> = None;
    let mut tsv_out_path: Option<PathBuf> = None;
    let mut pdb_ref_path: Option<PathBuf> = None;
    let mut chain_ids_opt: Vec<String> = Vec::new();
    let mut force_bfactor_plddt = false;
    let mut cli_pre_tau_c_ns: Option<f64> = None;
    let mut cli_pre_proton_mhz: Option<f64> = None;
    let mut cli_mtsl_tether_a: Option<f64> = None;
    let mut cli_refine_metal_pos = false;
    let mut cli_metal_search_radius_a: Option<f64> = None;
    let mut engine_req = EngineRequest::Auto;
    let mut cfg = DomainDockConfig::default();

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                println!("{STRUCTURE_PAE_DOMAINS_USAGE}");
                return Ok(());
            }
            "--pae-domains" => {}
            "--af-structure" | "--structure" => {
                i += 1;
                af_struct_path = Some(PathBuf::from(
                    args.get(i)
                        .ok_or("missing value for --af-structure <model.cif|pdb>")?,
                ));
            }
            "--af-pae" | "--pae" => {
                i += 1;
                af_pae_path = Some(PathBuf::from(
                    args.get(i).ok_or("missing value for --af-pae <pae.json>")?,
                ));
            }
            "--chain" | "--chains" => {
                i += 1;
                let raw = args
                    .get(i)
                    .ok_or("missing value for --chain <ID[,ID2,...]>")?;
                for part in raw.split(',') {
                    let t = part.trim();
                    if !t.is_empty() && !chain_ids_opt.iter().any(|c| c == t) {
                        chain_ids_opt.push(t.to_string());
                    }
                }
            }
            "--force-bfactor-plddt" => {
                force_bfactor_plddt = true;
            }
            "--rdc" => {
                i += 1;
                rdc_paths.push(PathBuf::from(
                    args.get(i)
                        .ok_or("missing value for --rdc <rdcs.tsv|nef>")?,
                ));
            }
            "--pcs" => {
                i += 1;
                pcs_paths.push(PathBuf::from(
                    args.get(i).ok_or("missing value for --pcs <pcs.tsv>")?,
                ));
            }
            "--pre" => {
                i += 1;
                pre_paths.push(PathBuf::from(
                    args.get(i).ok_or("missing value for --pre <pre.tsv>")?,
                ));
            }
            "--pre-tau-c-ns" | "--tau-c-ns" => {
                i += 1;
                let v: f64 = args
                    .get(i)
                    .ok_or("missing value for --pre-tau-c-ns <ns>")?
                    .parse()
                    .map_err(|e| format!("invalid --pre-tau-c-ns: {e}"))?;
                if !v.is_finite() || !(0.05..=1000.0).contains(&v) {
                    return Err(format!(
                        "invalid --pre-tau-c-ns {v}: expected finite rotational correlation time in 0.05..=1000.0 ns"
                    ));
                }
                cli_pre_tau_c_ns = Some(v);
            }
            "--pre-proton-mhz" | "--proton-mhz" => {
                i += 1;
                let v: f64 = args
                    .get(i)
                    .ok_or("missing value for --pre-proton-mhz <MHz>")?
                    .parse()
                    .map_err(|e| format!("invalid --pre-proton-mhz: {e}"))?;
                if !v.is_finite() || !(40.0..=1500.0).contains(&v) {
                    return Err(format!(
                        "invalid --pre-proton-mhz {v}: expected finite 1H Larmor frequency in 40.0..=1500.0 MHz"
                    ));
                }
                cli_pre_proton_mhz = Some(v);
            }
            "--mtsl-tether-allowance-a" | "--pre-tether-allowance" => {
                i += 1;
                let v: f64 = args
                    .get(i)
                    .ok_or("missing value for --mtsl-tether-allowance-a <A>")?
                    .parse()
                    .map_err(|e| format!("invalid --mtsl-tether-allowance-a: {e}"))?;
                if !v.is_finite() || !(0.0..=15.0).contains(&v) {
                    return Err(format!(
                        "invalid --mtsl-tether-allowance-a {v}: expected finite allowance in 0.0..=15.0 A"
                    ));
                }
                cli_mtsl_tether_a = Some(v);
            }
            "--refine-metal-pos" => {
                cli_refine_metal_pos = true;
            }
            "--metal-search-radius" => {
                i += 1;
                let v: f64 = args
                    .get(i)
                    .ok_or("missing value for --metal-search-radius <A>")?
                    .parse()
                    .map_err(|e| format!("invalid --metal-search-radius: {e}"))?;
                if !v.is_finite() || !(0.5..=20.0).contains(&v) {
                    return Err(format!(
                        "invalid --metal-search-radius {v}: expected finite radius in 0.5..=20.0 A"
                    ));
                }
                cli_metal_search_radius_a = Some(v);
            }
            "--noesy" | "--noesy-peaks" | "--noe" => {
                i += 1;
                noesy_paths.push(PathBuf::from(
                    args.get(i)
                        .ok_or("missing value for --noesy <inter_noe.tsv>")?,
                ));
            }
            "--rdc-free-frac" => {
                i += 1;
                let frac: f64 = args
                    .get(i)
                    .ok_or("missing value for --rdc-free-frac")?
                    .parse()
                    .map_err(|e| format!("invalid --rdc-free-frac: {e}"))?;
                if !frac.is_finite() || !(0.0..=0.50).contains(&frac) {
                    return Err(format!(
                        "invalid --rdc-free-frac {frac}: expected finite fraction in 0.0..=0.50"
                    ));
                }
                cfg.rdc_free_fraction = frac;
            }
            "--pcs-free-frac" => {
                i += 1;
                let frac: f64 = args
                    .get(i)
                    .ok_or("missing value for --pcs-free-frac")?
                    .parse()
                    .map_err(|e| format!("invalid --pcs-free-frac: {e}"))?;
                if !frac.is_finite() || !(0.0..=0.50).contains(&frac) {
                    return Err(format!(
                        "invalid --pcs-free-frac {frac}: expected finite fraction in 0.0..=0.50"
                    ));
                }
                cfg.pcs_free_fraction = frac;
            }
            "--allow-poor-intra-domain-fit" => {
                cfg.allow_poor_intra_domain_fit = true;
            }
            "--allow-underdetermined-single-medium" => {
                cfg.allow_underdetermined_single_medium = true;
            }
            "--rdc-weight" => {
                i += 1;
                let w: f64 = args
                    .get(i)
                    .ok_or("missing value for --rdc-weight")?
                    .parse()
                    .map_err(|e| format!("invalid --rdc-weight: {e}"))?;
                if !w.is_finite() || !(0.0..=1000.0).contains(&w) {
                    return Err(format!(
                        "invalid --rdc-weight {w}: expected finite weight in 0.0..=1000.0"
                    ));
                }
                cfg.weights.w_rdc = w;
            }
            "--ensemble-size" | "--conformers" => {
                i += 1;
                let n: usize = args
                    .get(i)
                    .ok_or("missing value for --ensemble-size")?
                    .parse()
                    .map_err(|e| format!("invalid --ensemble-size: {e}"))?;
                if !(1..=256).contains(&n) {
                    return Err(format!(
                        "invalid --ensemble-size {n}: expected integer in 1..=256"
                    ));
                }
                cfg.ensemble_size = n;
            }
            "--replicas" | "--runs" => {
                i += 1;
                let r: usize = args
                    .get(i)
                    .ok_or("missing value for --replicas")?
                    .parse()
                    .map_err(|e| format!("invalid --replicas: {e}"))?;
                if !(1..=1024).contains(&r) {
                    return Err(format!(
                        "invalid --replicas {r}: expected integer in 1..=1024"
                    ));
                }
                cfg.n_replicas = r;
            }
            "--engine" => {
                i += 1;
                engine_req = parse_engine_request(args.get(i).ok_or("missing --engine")?)?;
            }
            "--pdb-ref" => {
                i += 1;
                pdb_ref_path = Some(PathBuf::from(
                    args.get(i).ok_or("missing value for --pdb-ref")?,
                ));
            }
            "-o" | "--out" | "--out-pdb" => {
                i += 1;
                out_path = Some(PathBuf::from(
                    args.get(i)
                        .ok_or("missing value for -o <docked.pdb|docked.cif>")?,
                ));
            }
            "--json-out" | "--report-json" => {
                i += 1;
                json_out_path = Some(PathBuf::from(
                    args.get(i).ok_or("missing value for --json-out")?,
                ));
            }
            "--tsv-out" | "--report-tsv" => {
                i += 1;
                tsv_out_path = Some(PathBuf::from(
                    args.get(i).ok_or("missing value for --tsv-out")?,
                ));
            }
            "--seed" => {
                i += 1;
                cfg.seed = args
                    .get(i)
                    .ok_or("missing value for --seed")?
                    .parse()
                    .map_err(|e| format!("invalid --seed: {e}"))?;
            }
            other => {
                return Err(format!(
                    "unknown `ferric structure --pae-domains` flag {other:?}\n{STRUCTURE_PAE_DOMAINS_USAGE}"
                ));
            }
        }
        i += 1;
    }

    let struct_p = af_struct_path.ok_or_else(|| {
        format!("`ferric structure --pae-domains` requires --af-structure <model.cif|pdb>\n{STRUCTURE_PAE_DOMAINS_USAGE}")
    })?;
    let pae_p = af_pae_path.ok_or_else(|| {
        format!("`ferric structure --pae-domains` requires --af-pae <pae.json>\n{STRUCTURE_PAE_DOMAINS_USAGE}")
    })?;
    let out_p = out_path.ok_or_else(|| {
        format!("`ferric structure --pae-domains` requires -o <docked.pdb|docked.cif>\n{STRUCTURE_PAE_DOMAINS_USAGE}")
    })?;

    let (models, pae) = if chain_ids_opt.len() >= 2 {
        let mut ms = Vec::with_capacity(chain_ids_opt.len());
        let mut total_res = 0usize;
        for cid in &chain_ids_opt {
            let opts = AfIngestOptions {
                target_chain_id: Some(cid.clone()),
                force_bfactor_plddt,
            };
            let m = parse_af_structure_model(&struct_p, &opts)?;
            total_res += m.residues.len();
            ms.push(m);
        }
        let p = parse_pae_json_file(&pae_p, None, Some(total_res))?;
        (ms, p)
    } else if let Some(single_cid) = chain_ids_opt.first() {
        let opts = AfIngestOptions {
            target_chain_id: Some(single_cid.clone()),
            force_bfactor_plddt,
        };
        let m = parse_af_structure_model(&struct_p, &opts)?;
        let p = parse_pae_json_file(&pae_p, Some(single_cid.as_str()), Some(m.residues.len()))?;
        (vec![m], p)
    } else {
        let discovered = discover_pdb_polymer_chain_ids(&struct_p);
        if discovered.len() >= 2 {
            let mut ms = Vec::with_capacity(discovered.len());
            let mut total_res = 0usize;
            let mut all_ok = true;
            for cid in &discovered {
                let opts = AfIngestOptions {
                    target_chain_id: Some(cid.clone()),
                    force_bfactor_plddt,
                };
                match parse_af_structure_model(&struct_p, &opts) {
                    Ok(m) => {
                        total_res += m.residues.len();
                        ms.push(m);
                    }
                    Err(_) => {
                        all_ok = false;
                        break;
                    }
                }
            }
            if all_ok {
                if let Ok(p) = parse_pae_json_file(&pae_p, None, Some(total_res)) {
                    (ms, p)
                } else {
                    let opts = AfIngestOptions {
                        target_chain_id: None,
                        force_bfactor_plddt,
                    };
                    let m = parse_af_structure_model(&struct_p, &opts)?;
                    let p = parse_pae_json_file(&pae_p, None, Some(m.residues.len()))?;
                    (vec![m], p)
                }
            } else {
                let opts = AfIngestOptions {
                    target_chain_id: None,
                    force_bfactor_plddt,
                };
                let m = parse_af_structure_model(&struct_p, &opts)?;
                let p = parse_pae_json_file(&pae_p, None, Some(m.residues.len()))?;
                (vec![m], p)
            }
        } else {
            let opts = AfIngestOptions {
                target_chain_id: None,
                force_bfactor_plddt,
            };
            let m = parse_af_structure_model(&struct_p, &opts)?;
            let p = parse_pae_json_file(&pae_p, None, Some(m.residues.len()))?;
            (vec![m], p)
        }
    };

    let mut rdc_media: Vec<DomainRdcMediumSet> = Vec::new();
    for rp in &rdc_paths {
        rdc_media.extend(load_rdc_file_auto(rp)?);
    }
    let mut pcs_tags: Vec<DomainPcsTagSet> = Vec::new();
    for pp in &pcs_paths {
        let mut parsed = read_domain_pcs_tsv(pp)?;
        if cli_refine_metal_pos || cli_metal_search_radius_a.is_some() {
            for tag in &mut parsed {
                if cli_refine_metal_pos {
                    tag.refine_metal_pos = true;
                }
                if let Some(rad) = cli_metal_search_radius_a {
                    tag.metal_search_radius_a = Some(rad);
                }
            }
        }
        pcs_tags.extend(parsed);
    }
    let mut pre_sites: Vec<DomainPreSiteSet> = Vec::new();
    for prp in &pre_paths {
        pre_sites.extend(read_domain_pre_tsv_with_defaults(
            prp,
            cli_pre_tau_c_ns,
            cli_pre_proton_mhz,
            cli_mtsl_tether_a,
        )?);
    }
    let mut all_af_residues: Vec<AfResidueRecord> = Vec::new();
    for m in &models {
        let cid = if m.chain_id.trim().is_empty() {
            "A".to_string()
        } else {
            m.chain_id.trim().to_string()
        };
        for r in &m.residues {
            let mut rec = r.clone();
            rec.chain_id = cid.clone();
            all_af_residues.push(rec);
        }
    }
    let mut noe_restraints: Vec<AmbiguousDistanceRestraint> = Vec::new();
    for np in &noesy_paths {
        noe_restraints.extend(read_domain_noe_tsv(np, &all_af_residues)?);
    }

    let t0 = Instant::now();
    let mut dock_res: DomainDockResult = match engine_req {
        EngineRequest::Cpu => {
            let mut r = dock_domains_multi_chain_cpu(
                &models,
                &pae,
                &rdc_media,
                &pcs_tags,
                &pre_sites,
                &noe_restraints,
                &cfg,
            )?;
            r.report.elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
            r
        }
        EngineRequest::Gpu => {
            let gpu = GpuEngine::new_headless()?;
            dock_domains_multi_chain_gpu(
                &gpu,
                &models,
                &pae,
                &rdc_media,
                &pcs_tags,
                &pre_sites,
                &noe_restraints,
                &cfg,
            )?
        }
        EngineRequest::Auto => match GpuEngine::new_headless() {
            Ok(gpu) => match dock_domains_multi_chain_gpu(
                &gpu,
                &models,
                &pae,
                &rdc_media,
                &pcs_tags,
                &pre_sites,
                &noe_restraints,
                &cfg,
            ) {
                Ok(r) => r,
                Err(gpu_err) => {
                    let mut r = dock_domains_multi_chain_cpu(
                        &models,
                        &pae,
                        &rdc_media,
                        &pcs_tags,
                        &pre_sites,
                        &noe_restraints,
                        &cfg,
                    )?;
                    eprintln!(
                        "Warning: WebGPU domain docking failed ({gpu_err}); completed on CPU (f64)."
                    );
                    r.report.engine_used = "CPU (f64, auto-fallback)".to_string();
                    r.report.elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
                    r
                }
            },
            Err(_) => {
                let mut r = dock_domains_multi_chain_cpu(
                    &models,
                    &pae,
                    &rdc_media,
                    &pcs_tags,
                    &pre_sites,
                    &noe_restraints,
                    &cfg,
                )?;
                r.report.elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
                r
            }
        },
    };

    for note in &dock_res.report.candor_notes {
        eprintln!("Warning: {note}");
    }

    let ext = out_p
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();
    let entry_label = models
        .first()
        .and_then(|m| m.entry_id.as_deref())
        .unwrap_or("ferric_pae_dock");

    if ext == "json" {
        let js = format_domain_dock_json(&dock_res.report)?;
        fs::write(&out_p, js).map_err(|e| format!("failed to write '{}': {e}", out_p.display()))?;
    } else if ext == "tsv" {
        let tv = format_domain_dock_tsv(&dock_res.report);
        fs::write(&out_p, tv).map_err(|e| format!("failed to write '{}': {e}", out_p.display()))?;
    } else {
        if ext == "cif" || ext == "mmcif" {
            write_docked_mmcif_ensemble(&out_p, &dock_res, entry_label)?;
        } else {
            write_docked_pdb_ensemble(&out_p, &dock_res, entry_label)?;
        }
        let (def_json, def_tsv) = default_sidecar_report_paths(&out_p);
        let target_json = json_out_path.as_ref().unwrap_or(&def_json);
        let target_tsv = tsv_out_path.as_ref().unwrap_or(&def_tsv);
        let js = format_domain_dock_json(&dock_res.report)?;
        fs::write(target_json, js)
            .map_err(|e| format!("failed to write '{}': {e}", target_json.display()))?;
        let tv = format_domain_dock_tsv(&dock_res.report);
        fs::write(target_tsv, tv)
            .map_err(|e| format!("failed to write '{}': {e}", target_tsv.display()))?;
    }

    if let Some(ref_p) = &pdb_ref_path {
        let ref_model = parse_af_structure_model(ref_p, &AfIngestOptions::default())?;
        let ref_chain = ref_model.to_pdb_chain();
        let total_res = all_af_residues.len();
        let mut dom_mask = vec![false; total_res];
        for dom in &dock_res.report.partition.domains {
            for &idx in &dom.residue_indices {
                if idx < dom_mask.len() {
                    dom_mask[idx] = true;
                }
            }
        }
        let mask_opt = dom_mask.iter().any(|&b| b).then_some(dom_mask.as_slice());
        let bb_rmsd = multi_chain_backbone_rmsd(&dock_res.chains[0], &ref_chain, mask_opt);
        println!("  All-domain BB RMSD vs reference: {bb_rmsd:.3} A");
    }

    let pcs_q_work_str = dock_res
        .report
        .pcs_q_work
        .map_or_else(|| "NA".to_string(), |q| format!("{q:.4}"));
    let pcs_q_free_str = dock_res
        .report
        .pcs_q_free
        .map_or_else(|| "NA".to_string(), |q| format!("{q:.4}"));

    println!(
        "PAE multi-domain docking complete ({}, {:.1} ms): D={} rigid domain(s), {} linker(s), verdict={:?}, regime={:?}, Tolman={:?}, Q_work={:.4}, ByPeptidePlane Q_free={:.4}, PCS Q_work={}, PCS Q_free={}, max intra-domain distortion={:.3} A -> {}",
        dock_res.report.engine_used,
        dock_res.report.elapsed_ms,
        dock_res.report.partition.domains.len(),
        dock_res.report.partition.linkers.len(),
        dock_res.report.quality_verdict,
        dock_res.report.observability_regime,
        dock_res.report.tolman_gdo.regime,
        dock_res.report.global_rdc_q_work,
        dock_res.report.global_rdc_q_free,
        pcs_q_work_str,
        pcs_q_free_str,
        dock_res.report.max_intra_domain_rmsd_angstrom,
        out_p.display()
    );

    let _ = &mut dock_res;
    Ok(())
}
