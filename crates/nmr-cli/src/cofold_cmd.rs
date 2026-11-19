//! CLI subcommand `ferric cofold (export | score)` (`T21.6a`,
//! `docs/specs/21-alphafold-integrative-nmr.md` section 6):
//! schema-honest `AlphaFold 3` JSON (`dialect: "alphafold3"`, `version: 2`) and `Boltz-2` YAML
//! (`version: 1`) interface restraint exporter, plus multi-pose co-folded complex NMR re-scorer.
//!
//! # Literature & Clean-Room Provenance
//! - Abramson, J. et al. (*Nature* **2024**, 630, 493-500) -- AlphaFold 3 JSON v2 schema.
//! - Wohlwend, J. et al. (*bioRxiv* **2024**, DOI: `10.1101/2024.11.19.624167`) & Passaro, S. et al.
//!   (*bioRxiv* **2025**, DOI: `10.1101/2025.06.14.659707`) -- MIT `Boltz-2` YAML v1 schema.
//! - Shrake, A. & Rupley, J. A. (*J. Mol. Biol.* **1973**, 79, 351-371) -- 92-point Fibonacci `SASA`.
//! - Williamson, M. P. (*Prog. Nucl. Magn. Reson. Spectrosc.* **2013**, 73, 1-16) -- `2-sigma` `CSP`.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use nmr_core::cofold_bridge::{
    export_af3_json, export_boltz2_yaml, export_nmr_restraints_sidecar_json,
    extract_nmr_interface_restraints, format_pose_score_json, format_pose_score_tsv,
    ligand_spec_from_octet_file, ligand_spec_from_octet_str, parse_cofold_fasta_str,
    parse_cofold_pcs_tsv_str, parse_cofold_rdc_tsv_str, parse_csp_restraint_str,
    parse_intermolecular_noe_tsv_str, parse_intermolecular_pre_tsv_str_with_overrides,
    score_cofold_poses, Af3ExportConfig, Boltz2ExportConfig, BondedAtomPairSpec, BondedAtomRef,
    CofoldConfidenceMetadata, CofoldEntitySpec, CofoldLigandSpec, CofoldPolymerKind,
    CofoldRestraintBundle, CofoldRestraintExtractConfig, CofoldRestraintInput, CspScoringMode,
    PoseScoreConfig, PoseVerdict, StructurePoseInput, BOLTZ2_MAX_DISTANCE_A,
    BOLTZ2_MIN_DISTANCE_A, MAX_COFOLD_INPUT_BYTES,
};
use nmr_core::struct_shifts::StructureModel;

pub fn cofold_usage() -> &'static str {
    "ferric cofold -- AlphaFold 3 JSON / Boltz-2 NMR-restrained co-folding bridge and multi-pose NMR re-scorer\n\
     Usage:\n\
       ferric cofold export [--fasta <seqs.fasta>] [--protein <ID:SEQ>] [--rna <ID:SEQ>] [--dna <ID:SEQ>]\n\
                            [--ligand-smiles <ID:SMILES>] [--ligand-ccd <ID:CCD>] [--ligand-file <[ID:]path>]\n\
                            [--apo-structure <apo.cif|pdb>] [--allow-unfiltered-no-apo] [--min-rsasa 0.20]\n\
                            [--csp <csp.tsv|json>] [--noesy <inter_noe.tsv>] [--pre <pre.tsv>]\n\
                            [--pre-anchor <CHAIN:SEQ:ATOM>] [--pre-tau-c <ns>] [--pre-proton-mhz <MHz>]\n\
                            [--pre-tether-pad 6.0] [--pre-rigid-metal] [--pre-spin 0.5]\n\
                            [--rdc <rdcs.tsv>] [--pcs <pcs.tsv>] [--receptor-chain <ID>] [--binder-chain <ID>]\n\
                            [--pocket-max-dist 6.0] [--max-contact-dist 14.0] [--affinity] [--seeds 1,2,3] [--name <job>]\n\
                            [--format af3|boltz2|both] [-o <out.json|out.yaml> | --out-prefix <prefix>]\n\
                            [--restraints-out <sidecar.nmr_restraints.json>]\n\n\
       ferric cofold score  (--poses <p1.cif,p2.pdb,...> | --pose <path> ... | --pose-dir <dir>)\n\
                            [--restraints <sidecar.nmr_restraints.json>] [--csp <csp.tsv|json>]\n\
                            [--apo-structure <apo.cif|pdb>] [--allow-unfiltered-no-apo]\n\
                            [--noesy <inter_noe.tsv>] [--pre <pre.tsv>] [--pre-anchor <CHAIN:SEQ:ATOM>]\n\
                            [--pre-tau-c <ns>] [--pre-proton-mhz <MHz>] [--pre-tether-pad 6.0] [--pre-rigid-metal]\n\
                            [--rdc <rdcs.tsv>] [--pcs <pcs.tsv>] [--ligand-smiles <ID:SMILES>]\n\
                            [--ligand-file <[ID:]path>] [--receptor-chain <ID>] [--binder-chain <ID>]\n\
                            [--reference-pose <native.cif|pdb>] [--confidence <pose_id=conf.json>]\n\
                            [--csp-scoring burial|forward] [--w-csp 4.0] [--w-noe 1.5] [--w-rdc 2.5] [--w-pcs 2.5]\n\
                            [--rdc-free-frac 0.20] [--exclude-pdb <ids>] [--exclude-bmrb <ids>] [--max-seq-identity 0.25]\n\
                            [--format tsv|json] [-o <out.tsv|out.json>]"
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExportFormat {
    Af3,
    Boltz2,
    Both,
}

impl ExportFormat {
    fn from_token(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "af3" | "alphafold3" | "json" => Ok(Self::Af3),
            "boltz2" | "boltz" | "boltz-2" | "yaml" | "yml" => Ok(Self::Boltz2),
            "both" | "all" => Ok(Self::Both),
            other => Err(format!(
                "unsupported --format {other:?}; expected af3, boltz2, or both"
            )),
        }
    }

    fn from_path(path: &Path) -> Result<Self, String> {
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        match ext.as_str() {
            "json" => Ok(Self::Af3),
            "yaml" | "yml" => Ok(Self::Boltz2),
            other => Err(format!(
                "unable to infer export format from extension '.{other}' on '{}'; pass --format af3|boltz2",
                path.display()
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScoreFormat {
    Tsv,
    Json,
}

impl ScoreFormat {
    fn from_token(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "tsv" | "tab" => Ok(Self::Tsv),
            "json" => Ok(Self::Json),
            other => Err(format!(
                "unsupported --format {other:?}; expected tsv or json"
            )),
        }
    }

    fn from_path(path: &Path) -> Result<Self, String> {
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        match ext.as_str() {
            "json" => Ok(Self::Json),
            "tsv" | "txt" => Ok(Self::Tsv),
            other => Err(format!(
                "unsupported score output extension '.{other}' on '{}'; expected .tsv or .json",
                path.display()
            )),
        }
    }
}

fn read_utf8_bounded(path: &Path, label: &str) -> Result<String, String> {
    let meta = fs::metadata(path)
        .map_err(|e| format!("failed to stat {label} '{}': {e}", path.display()))?;
    if meta.len() > MAX_COFOLD_INPUT_BYTES as u64 {
        return Err(format!("{label} '{}' exceeds 64 MiB limit", path.display()));
    }
    let bytes =
        fs::read(path).map_err(|e| format!("failed to read {label} '{}': {e}", path.display()))?;
    if bytes.len() > MAX_COFOLD_INPUT_BYTES || bytes.contains(&0u8) {
        return Err(format!(
            "{label} '{}' exceeds size cap or contains NUL bytes",
            path.display()
        ));
    }
    String::from_utf8(bytes)
        .map_err(|e| format!("invalid UTF-8 in {label} '{}': {e}", path.display()))
}

fn write_file_creating_dirs(path: &Path, content: &str) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("failed to create directory '{}': {e}", parent.display()))?;
        }
    }
    fs::write(path, content).map_err(|e| format!("failed to write '{}': {e}", path.display()))
}

fn split_optional_id_prefix<'a>(raw: &'a str, default_id: &'a str) -> (&'a str, &'a str) {
    if let Some((left, right)) = raw.split_once(':') {
        let l = left.trim();
        // Avoid splitting Windows drive letters `C:\...` when right starts with backslash
        let looks_like_id = !l.is_empty()
            && l.len() <= 8
            && l.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
            && !(l.len() == 1 && right.starts_with('\\'));
        if looks_like_id {
            return (l, right.trim());
        }
    }
    (default_id, raw.trim())
}

fn parse_chain_seq_atom_triple(
    raw: &str,
    flag_name: &str,
) -> Result<(String, i32, String), String> {
    let parts: Vec<&str> = raw.split(':').map(str::trim).collect();
    if parts.len() != 3 {
        return Err(format!(
            "invalid {flag_name} {raw:?}; expected CHAIN:SEQ_ID:ATOM (e.g. B:42:SG)"
        ));
    }
    let sid: i32 = parts[1]
        .parse()
        .map_err(|e| format!("invalid residue number in {flag_name} {raw:?}: {e}"))?;
    Ok((parts[0].to_string(), sid, parts[2].to_ascii_uppercase()))
}

fn parse_bonded_pair_spec(raw: &str) -> Result<BondedAtomPairSpec, String> {
    let (lhs, rhs) = raw.split_once('=').ok_or_else(|| {
        format!("invalid --bonded-pair {raw:?}; expected C1:IDX1:ATOM1=C2:IDX2:ATOM2")
    })?;
    let parse_side = |s: &str| -> Result<BondedAtomRef, String> {
        let parts: Vec<&str> = s.split(':').map(str::trim).collect();
        if parts.len() != 3 {
            return Err(format!(
                "invalid bonded atom endpoint {s:?}; expected CHAIN:1_BASED_IDX:ATOM"
            ));
        }
        let idx1: usize = parts[1]
            .parse()
            .map_err(|e| format!("invalid 1-based index in {s:?}: {e}"))?;
        if idx1 == 0 {
            return Err(format!(
                "bonded atom endpoint {s:?} must use 1-based residue index >= 1"
            ));
        }
        Ok(BondedAtomRef {
            chain_id: parts[0].to_string(),
            one_based_res_idx: idx1,
            atom_name: parts[2].to_ascii_uppercase(),
        })
    };
    Ok(BondedAtomPairSpec {
        atom1: parse_side(lhs)?,
        atom2: parse_side(rhs)?,
    })
}

fn default_sidecar_path_for(out_path: &Path) -> PathBuf {
    let parent = out_path.parent().unwrap_or_else(|| Path::new(""));
    let stem = out_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("cofold");
    // Strip trailing `.af3` or `.boltz2` if present so `job.af3.json` -> `job.nmr_restraints.json`
    let base = stem
        .strip_suffix(".af3")
        .or_else(|| stem.strip_suffix(".boltz2"))
        .unwrap_or(stem);
    parent.join(format!("{base}.nmr_restraints.json"))
}

pub fn run_cofold_cli(args: &[String]) -> Result<(), String> {
    if args.is_empty() || args[0] == "-h" || args[0] == "--help" {
        println!("{}", cofold_usage());
        return Ok(());
    }
    match args[0].as_str() {
        "export" => run_cofold_export_cli(&args[1..]),
        "score" => run_cofold_score_cli(&args[1..]),
        other => Err(format!(
            "unknown `ferric cofold` subcommand {other:?}; expected `export` or `score`\n\n{}",
            cofold_usage()
        )),
    }
}

fn run_cofold_export_cli(args: &[String]) -> Result<(), String> {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("{}", cofold_usage());
        return Ok(());
    }

    let mut input = CofoldRestraintInput::default();
    let mut extract_cfg = CofoldRestraintExtractConfig::default();
    let mut af3_cfg = Af3ExportConfig::default();
    let mut boltz_cfg = Boltz2ExportConfig::default();

    let mut fasta_path: Option<PathBuf> = None;
    let mut apo_path: Option<PathBuf> = None;
    let mut csp_path: Option<PathBuf> = None;
    let mut noesy_path: Option<PathBuf> = None;
    let mut pre_path: Option<PathBuf> = None;
    let mut pre_anchor: Option<(String, i32, String)> = None;
    let mut pre_tau_c_ns: Option<f64> = None;
    let mut pre_proton_mhz: Option<f64> = None;
    let mut pre_tether_pad_a: Option<f64> = None;
    let mut pre_spin_s: Option<f64> = None;
    let mut rdc_path: Option<PathBuf> = None;
    let mut pcs_path: Option<PathBuf> = None;
    let mut explicit_format: Option<ExportFormat> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut out_prefix: Option<PathBuf> = None;
    let mut sidecar_out: Option<PathBuf> = None;

    let mut i = 0usize;
    while i < args.len() {
        let flag = args[i].as_str();
        let next_val = |idx: &mut usize, name: &str| -> Result<&str, String> {
            *idx += 1;
            args.get(*idx)
                .map(|s| s.as_str())
                .ok_or_else(|| format!("missing value after {name}"))
        };

        match flag {
            "--fasta" | "--seq" => {
                fasta_path = Some(PathBuf::from(next_val(&mut i, flag)?));
            }
            "--protein" => {
                let v = next_val(&mut i, flag)?;
                let (cid, seq) = split_optional_id_prefix(v, "A");
                input.entities.push(CofoldEntitySpec::Polymer {
                    kind: CofoldPolymerKind::Protein,
                    chain_id: cid.to_string(),
                    sequence: seq.to_string(),
                });
            }
            "--rna" => {
                let v = next_val(&mut i, flag)?;
                let (cid, seq) = split_optional_id_prefix(v, "R");
                input.entities.push(CofoldEntitySpec::Polymer {
                    kind: CofoldPolymerKind::Rna,
                    chain_id: cid.to_string(),
                    sequence: seq.to_string(),
                });
            }
            "--dna" => {
                let v = next_val(&mut i, flag)?;
                let (cid, seq) = split_optional_id_prefix(v, "D");
                input.entities.push(CofoldEntitySpec::Polymer {
                    kind: CofoldPolymerKind::Dna,
                    chain_id: cid.to_string(),
                    sequence: seq.to_string(),
                });
            }
            "--ligand-smiles" | "--smiles" => {
                let v = next_val(&mut i, flag)?;
                let (cid, smi) = split_optional_id_prefix(v, "L");
                let (spec, mol) =
                    ligand_spec_from_octet_str(smi, Some("smi")).map_err(String::from)?;
                input.ligand_molecules.insert(cid.to_string(), mol);
                input.entities.push(CofoldEntitySpec::Ligand {
                    chain_id: cid.to_string(),
                    spec,
                });
            }
            "--ligand-ccd" | "--ccd" => {
                let v = next_val(&mut i, flag)?;
                let (cid, raw_codes) = split_optional_id_prefix(v, "L");
                let codes: Vec<String> = raw_codes
                    .split(',')
                    .map(|c| c.trim().to_ascii_uppercase())
                    .filter(|c| !c.is_empty())
                    .collect();
                input.entities.push(CofoldEntitySpec::Ligand {
                    chain_id: cid.to_string(),
                    spec: CofoldLigandSpec::CcdCodes(codes),
                });
            }
            "--ligand-file" => {
                let v = next_val(&mut i, flag)?;
                let (cid, p_str) = split_optional_id_prefix(v, "L");
                let (spec, mol) =
                    ligand_spec_from_octet_file(Path::new(p_str)).map_err(String::from)?;
                input.ligand_molecules.insert(cid.to_string(), mol);
                input.entities.push(CofoldEntitySpec::Ligand {
                    chain_id: cid.to_string(),
                    spec,
                });
            }
            "--receptor-chain" => {
                input.receptor_chain_id = Some(next_val(&mut i, flag)?.trim().to_string());
            }
            "--binder-chain" => {
                input.binder_chain_id = Some(next_val(&mut i, flag)?.trim().to_string());
            }
            "--apo-structure" | "--apo" => {
                apo_path = Some(PathBuf::from(next_val(&mut i, flag)?));
            }
            "--allow-unfiltered-no-apo" => {
                extract_cfg.allow_unfiltered_no_apo = true;
            }
            "--min-rsasa" => {
                let v = next_val(&mut i, flag)?;
                let rs: f64 = v
                    .parse()
                    .map_err(|e| format!("invalid --min-rsasa {v:?}: {e}"))?;
                if !rs.is_finite() || !(0.0..=1.0).contains(&rs) {
                    return Err(format!("--min-rsasa {rs} must be in [0.0, 1.0]"));
                }
                extract_cfg.min_pocket_rsasa = rs;
            }
            "--csp" => {
                csp_path = Some(PathBuf::from(next_val(&mut i, flag)?));
            }
            "--noesy" | "--inter-noe" => {
                noesy_path = Some(PathBuf::from(next_val(&mut i, flag)?));
            }
            "--pre" => {
                pre_path = Some(PathBuf::from(next_val(&mut i, flag)?));
            }
            "--pre-anchor" => {
                let v = next_val(&mut i, flag)?;
                pre_anchor = Some(parse_chain_seq_atom_triple(v, flag)?);
            }
            "--pre-tau-c" => {
                let v = next_val(&mut i, flag)?;
                let tau: f64 = v
                    .parse()
                    .map_err(|e| format!("invalid --pre-tau-c {v:?}: {e}"))?;
                if !tau.is_finite() || !(0.05..=1.0e4).contains(&tau) {
                    return Err(format!(
                        "--pre-tau-c {tau} must be finite in [0.05, 10000] ns"
                    ));
                }
                pre_tau_c_ns = Some(tau);
            }
            "--pre-proton-mhz" | "--proton-mhz" => {
                let v = next_val(&mut i, flag)?;
                let mhz: f64 = v
                    .parse()
                    .map_err(|e| format!("invalid --pre-proton-mhz {v:?}: {e}"))?;
                if !mhz.is_finite() || !(1.0..=5.0e3).contains(&mhz) {
                    return Err(format!(
                        "--pre-proton-mhz {mhz} must be finite in [1, 5000] MHz"
                    ));
                }
                pre_proton_mhz = Some(mhz);
            }
            "--pre-tether-pad" => {
                let v = next_val(&mut i, flag)?;
                let pad: f64 = v
                    .parse()
                    .map_err(|e| format!("invalid --pre-tether-pad {v:?}: {e}"))?;
                if !pad.is_finite() || !(0.0..=15.0).contains(&pad) {
                    return Err(format!(
                        "--pre-tether-pad {pad} must be finite in [0.0, 15.0] A"
                    ));
                }
                pre_tether_pad_a = Some(pad);
            }
            "--pre-rigid-metal" => {
                pre_tether_pad_a = Some(0.0);
            }
            "--pre-spin" => {
                let v = next_val(&mut i, flag)?;
                let s: f64 = v
                    .parse()
                    .map_err(|e| format!("invalid --pre-spin {v:?}: {e}"))?;
                if !s.is_finite() || !(0.5..=8.0).contains(&s) {
                    return Err(format!("--pre-spin {s} must be finite in [0.5, 8.0]"));
                }
                pre_spin_s = Some(s);
            }
            "--rdc" => {
                rdc_path = Some(PathBuf::from(next_val(&mut i, flag)?));
            }
            "--pcs" => {
                pcs_path = Some(PathBuf::from(next_val(&mut i, flag)?));
            }
            "--bonded-pair" => {
                let v = next_val(&mut i, flag)?;
                input.bonded_atom_pairs.push(parse_bonded_pair_spec(v)?);
            }
            "--pocket-max-dist" => {
                let v = next_val(&mut i, flag)?;
                let d: f64 = v
                    .parse()
                    .map_err(|e| format!("invalid --pocket-max-dist {v:?}: {e}"))?;
                if !d.is_finite() || !(BOLTZ2_MIN_DISTANCE_A..=BOLTZ2_MAX_DISTANCE_A).contains(&d) {
                    return Err(format!(
                        "--pocket-max-dist ({d}) must be finite in [{BOLTZ2_MIN_DISTANCE_A}, {BOLTZ2_MAX_DISTANCE_A}] A"
                    ));
                }
                extract_cfg.pocket_max_distance_a = d;
                boltz_cfg.pocket_max_distance_override = Some(d);
            }
            "--max-contact-dist" => {
                let v = next_val(&mut i, flag)?;
                let d: f64 = v
                    .parse()
                    .map_err(|e| format!("invalid --max-contact-dist {v:?}: {e}"))?;
                if !d.is_finite() || !(BOLTZ2_MIN_DISTANCE_A..=BOLTZ2_MAX_DISTANCE_A).contains(&d) {
                    return Err(format!(
                        "--max-contact-dist ({d}) must be finite in [{BOLTZ2_MIN_DISTANCE_A}, {BOLTZ2_MAX_DISTANCE_A}] A"
                    ));
                }
                extract_cfg.max_contact_distance_a = d;
            }
            "--force-constraints" => {
                extract_cfg.force_constraints = true;
                boltz_cfg.force_override = Some(true);
            }
            "--affinity" => {
                extract_cfg.request_boltz2_affinity = true;
                boltz_cfg.include_affinity_override = Some(true);
            }
            "--seeds" => {
                let v = next_val(&mut i, flag)?;
                let mut seeds = Vec::new();
                for part in v.split(',') {
                    let t = part.trim();
                    if !t.is_empty() {
                        let s: u32 = t
                            .parse()
                            .map_err(|e| format!("invalid integer seed {t:?} in --seeds: {e}"))?;
                        seeds.push(s);
                    }
                }
                if seeds.is_empty() {
                    return Err("--seeds must contain at least one integer".to_string());
                }
                af3_cfg.model_seeds = seeds;
            }
            "--name" => {
                input.job_name = next_val(&mut i, flag)?.trim().to_string();
            }
            "--format" => {
                explicit_format = Some(ExportFormat::from_token(next_val(&mut i, flag)?)?);
            }
            "-o" | "--out" => {
                out_path = Some(PathBuf::from(next_val(&mut i, flag)?));
            }
            "--out-prefix" => {
                out_prefix = Some(PathBuf::from(next_val(&mut i, flag)?));
            }
            "--restraints-out" => {
                sidecar_out = Some(PathBuf::from(next_val(&mut i, flag)?));
            }
            other => {
                return Err(format!(
                    "unknown flag {other:?} for `ferric cofold export`\n\n{}",
                    cofold_usage()
                ));
            }
        }
        i += 1;
    }

    if let Some(ref fp) = fasta_path {
        let fasta_txt = read_utf8_bounded(fp, "FASTA")?;
        let mut parsed_ents = parse_cofold_fasta_str(&fasta_txt).map_err(String::from)?;
        parsed_ents.extend(input.entities);
        input.entities = parsed_ents;
    }

    if let Some(ref ap) = apo_path {
        let model = StructureModel::parse_file(ap, false)?;
        input.apo_receptor_model = Some(model);
    }

    if let Some(ref cp) = csp_path {
        let txt = read_utf8_bounded(cp, "CSP file")?;
        let (rows, kd) = parse_csp_restraint_str(&txt).map_err(String::from)?;
        input.raw_csp_rows = rows;
        if input.experimental_kd_um.is_none() {
            input.experimental_kd_um = kd;
        }
    }

    if let Some(ref np) = noesy_path {
        let txt = read_utf8_bounded(np, "NOE TSV")?;
        let noes = parse_intermolecular_noe_tsv_str(&txt).map_err(String::from)?;
        input.raw_distance_restraints.extend(noes);
    }

    let default_rec = input
        .receptor_chain_id
        .clone()
        .or_else(|| {
            input
                .entities
                .iter()
                .find(|e| e.is_polymer())
                .map(|e| e.chain_id().to_string())
        })
        .unwrap_or_else(|| "A".to_string());

    if let Some(ref pp) = pre_path {
        let txt = read_utf8_bounded(pp, "PRE TSV")?;
        let anc_ref = pre_anchor
            .as_ref()
            .map(|(c, s, a)| (c.as_str(), *s, a.as_str()));
        let pres = parse_intermolecular_pre_tsv_str_with_overrides(
            &txt,
            anc_ref,
            &default_rec,
            pre_tau_c_ns,
            pre_proton_mhz,
            pre_spin_s,
            pre_tether_pad_a,
            extract_cfg.pre_gamma2_upper_margin_a,
        )
        .map_err(String::from)?;
        input.raw_distance_restraints.extend(pres);
    }

    if let Some(ref rp) = rdc_path {
        let txt = read_utf8_bounded(rp, "RDC TSV")?;
        input.rdc_media = parse_cofold_rdc_tsv_str(&txt, &default_rec).map_err(String::from)?;
    }

    if let Some(ref pp) = pcs_path {
        let txt = read_utf8_bounded(pp, "PCS TSV")?;
        input.pcs_tags = parse_cofold_pcs_tsv_str(&txt, &default_rec).map_err(String::from)?;
    }

    if input.job_name.is_empty() {
        if let Some(ref op) = out_prefix {
            if let Some(st) = op.file_stem().and_then(|s| s.to_str()) {
                input.job_name = st.to_string();
            }
        } else if let Some(ref o) = out_path {
            if let Some(st) = o.file_stem().and_then(|s| s.to_str()) {
                input.job_name = st
                    .strip_suffix(".af3")
                    .or_else(|| st.strip_suffix(".boltz2"))
                    .unwrap_or(st)
                    .to_string();
            }
        }
    }

    let bundle = extract_nmr_interface_restraints(&input, &extract_cfg).map_err(String::from)?;
    for diag in &bundle.diagnostics {
        if diag.starts_with("WARNING:") || diag.starts_with("Solomon Gamma2 PRE") {
            eprintln!("[cofold] {diag}");
        }
    }

    let fmt = match (explicit_format, out_path.as_deref(), out_prefix.as_deref()) {
        (Some(f), _, _) => f,
        (None, Some(p), _) => ExportFormat::from_path(p)?,
        (None, None, Some(_)) => ExportFormat::Both,
        (None, None, None) => ExportFormat::Boltz2,
    };

    if let Some(ref prefix) = out_prefix {
        let base_str = prefix.to_string_lossy();
        if matches!(fmt, ExportFormat::Af3 | ExportFormat::Both) {
            let af3_str = export_af3_json(&bundle, &af3_cfg).map_err(String::from)?;
            let af3_p = PathBuf::from(format!("{base_str}.af3.json"));
            write_file_creating_dirs(&af3_p, &af3_str)?;
        }
        if matches!(fmt, ExportFormat::Boltz2 | ExportFormat::Both) {
            let boltz_str = export_boltz2_yaml(&bundle, &boltz_cfg).map_err(String::from)?;
            let boltz_p = PathBuf::from(format!("{base_str}.boltz2.yaml"));
            write_file_creating_dirs(&boltz_p, &boltz_str)?;
        }
        let sc_path =
            sidecar_out.unwrap_or_else(|| PathBuf::from(format!("{base_str}.nmr_restraints.json")));
        let sc_str = export_nmr_restraints_sidecar_json(&bundle).map_err(String::from)?;
        write_file_creating_dirs(&sc_path, &sc_str)?;
        eprintln!(
            "Exported co-folding restraints (pocket={}, distal_surface={}, allosteric_core={}, contacts={}, dropped={}) -> prefix '{}'",
            bundle.pocket_contacts.len(),
            bundle.distal_surface_csp_hits.len(),
            bundle.allosteric_core_perturbations.len(),
            bundle
                .distance_restraints
                .iter()
                .filter(|r| r.exported_contact_max_distance_a.is_some())
                .count(),
            bundle.dropped_weak_restraints_count,
            prefix.display()
        );
        return Ok(());
    }

    match fmt {
        ExportFormat::Af3 => {
            let af3_str = export_af3_json(&bundle, &af3_cfg).map_err(String::from)?;
            if let Some(ref out_p) = out_path {
                write_file_creating_dirs(out_p, &af3_str)?;
                let sc_p = sidecar_out.unwrap_or_else(|| default_sidecar_path_for(out_p));
                let sc_str = export_nmr_restraints_sidecar_json(&bundle).map_err(String::from)?;
                write_file_creating_dirs(&sc_p, &sc_str)?;
                eprintln!(
                    "Exported AlphaFold 3 JSON -> {} (sidecar -> {})",
                    out_p.display(),
                    sc_p.display()
                );
            } else {
                if let Some(ref sc_p) = sidecar_out {
                    let sc_str =
                        export_nmr_restraints_sidecar_json(&bundle).map_err(String::from)?;
                    write_file_creating_dirs(sc_p, &sc_str)?;
                }
                print!("{af3_str}");
            }
        }
        ExportFormat::Boltz2 => {
            let boltz_str = export_boltz2_yaml(&bundle, &boltz_cfg).map_err(String::from)?;
            if let Some(ref out_p) = out_path {
                write_file_creating_dirs(out_p, &boltz_str)?;
                if let Some(ref sc_p) = sidecar_out {
                    let sc_str =
                        export_nmr_restraints_sidecar_json(&bundle).map_err(String::from)?;
                    write_file_creating_dirs(sc_p, &sc_str)?;
                }
                eprintln!(
                    "Exported Boltz-2 YAML (pocket={}, distal_surface={}, contacts={}, dropped={}) -> {}",
                    bundle.pocket_contacts.len(),
                    bundle.distal_surface_csp_hits.len(),
                    bundle
                        .distance_restraints
                        .iter()
                        .filter(|r| r.exported_contact_max_distance_a.is_some())
                        .count(),
                    bundle.dropped_weak_restraints_count,
                    out_p.display()
                );
            } else {
                if let Some(ref sc_p) = sidecar_out {
                    let sc_str =
                        export_nmr_restraints_sidecar_json(&bundle).map_err(String::from)?;
                    write_file_creating_dirs(sc_p, &sc_str)?;
                }
                print!("{boltz_str}");
            }
        }
        ExportFormat::Both => {
            return Err(
                "--format both requires --out-prefix <prefix> so both .af3.json and .boltz2.yaml can be written"
                    .to_string(),
            );
        }
    }

    Ok(())
}

fn push_csv_items(dst: &mut Vec<String>, raw: &str) {
    for part in raw.split(',') {
        let t = part.trim();
        if !t.is_empty() {
            dst.push(t.to_string());
        }
    }
}

fn run_cofold_score_cli(args: &[String]) -> Result<(), String> {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("{}", cofold_usage());
        return Ok(());
    }

    let mut pose_paths: Vec<PathBuf> = Vec::new();
    let mut pose_dir: Option<PathBuf> = None;
    let mut conf_map: BTreeMap<String, PathBuf> = BTreeMap::new();
    let mut restraints_json_path: Option<PathBuf> = None;

    let mut raw_input = CofoldRestraintInput::default();
    let mut extract_cfg = CofoldRestraintExtractConfig {
        scoring_only: true,
        ..CofoldRestraintExtractConfig::default()
    };
    let mut apo_path: Option<PathBuf> = None;
    let mut csp_path: Option<PathBuf> = None;
    let mut noesy_path: Option<PathBuf> = None;
    let mut pre_path: Option<PathBuf> = None;
    let mut pre_anchor: Option<(String, i32, String)> = None;
    let mut pre_tau_c_ns: Option<f64> = None;
    let mut pre_proton_mhz: Option<f64> = None;
    let mut pre_tether_pad_a: Option<f64> = None;
    let mut pre_spin_s: Option<f64> = None;
    let mut rdc_path: Option<PathBuf> = None;
    let mut pcs_path: Option<PathBuf> = None;
    let mut ref_pose_path: Option<PathBuf> = None;

    let mut score_cfg = PoseScoreConfig::default();
    let mut explicit_format: Option<ScoreFormat> = None;
    let mut out_path: Option<PathBuf> = None;

    let mut i = 0usize;
    while i < args.len() {
        let flag = args[i].as_str();
        let next_val = |idx: &mut usize, name: &str| -> Result<&str, String> {
            *idx += 1;
            args.get(*idx)
                .map(|s| s.as_str())
                .ok_or_else(|| format!("missing value after {name}"))
        };

        match flag {
            "--poses" => {
                let v = next_val(&mut i, flag)?;
                for part in v.split(',') {
                    let t = part.trim();
                    if !t.is_empty() {
                        pose_paths.push(PathBuf::from(t));
                    }
                }
            }
            "--pose" => {
                pose_paths.push(PathBuf::from(next_val(&mut i, flag)?));
            }
            "--pose-dir" => {
                pose_dir = Some(PathBuf::from(next_val(&mut i, flag)?));
            }
            "--restraints" => {
                restraints_json_path = Some(PathBuf::from(next_val(&mut i, flag)?));
            }
            "--confidence" | "--confidences" => {
                let v = next_val(&mut i, flag)?;
                for item in v.split(',') {
                    let t = item.trim();
                    if t.is_empty() {
                        continue;
                    }
                    let (pid, cpath) = t.split_once('=').ok_or_else(|| {
                        format!("invalid --confidence {t:?}; expected <pose_id>=<conf.json>")
                    })?;
                    conf_map.insert(pid.trim().to_string(), PathBuf::from(cpath.trim()));
                }
            }
            "--apo-structure" | "--apo" => {
                apo_path = Some(PathBuf::from(next_val(&mut i, flag)?));
            }
            "--allow-unfiltered-no-apo" => {
                extract_cfg.allow_unfiltered_no_apo = true;
            }
            "--max-contact-dist" => {
                let v = next_val(&mut i, flag)?;
                let d: f64 = v
                    .parse()
                    .map_err(|e| format!("invalid --max-contact-dist {v:?}: {e}"))?;
                if !d.is_finite() || !(BOLTZ2_MIN_DISTANCE_A..=BOLTZ2_MAX_DISTANCE_A).contains(&d) {
                    return Err(format!(
                        "--max-contact-dist ({d}) must be finite in [{BOLTZ2_MIN_DISTANCE_A}, {BOLTZ2_MAX_DISTANCE_A}] A"
                    ));
                }
                extract_cfg.max_contact_distance_a = d;
            }
            "--csp" => {
                csp_path = Some(PathBuf::from(next_val(&mut i, flag)?));
            }
            "--noesy" | "--inter-noe" => {
                noesy_path = Some(PathBuf::from(next_val(&mut i, flag)?));
            }
            "--pre" => {
                pre_path = Some(PathBuf::from(next_val(&mut i, flag)?));
            }
            "--pre-anchor" => {
                let v = next_val(&mut i, flag)?;
                pre_anchor = Some(parse_chain_seq_atom_triple(v, flag)?);
            }
            "--pre-tau-c" => {
                let v = next_val(&mut i, flag)?;
                let tau: f64 = v
                    .parse()
                    .map_err(|e| format!("invalid --pre-tau-c {v:?}: {e}"))?;
                if !tau.is_finite() || !(0.05..=1.0e4).contains(&tau) {
                    return Err(format!(
                        "--pre-tau-c {tau} must be finite in [0.05, 10000] ns"
                    ));
                }
                pre_tau_c_ns = Some(tau);
            }
            "--pre-proton-mhz" | "--proton-mhz" => {
                let v = next_val(&mut i, flag)?;
                let mhz: f64 = v
                    .parse()
                    .map_err(|e| format!("invalid --pre-proton-mhz {v:?}: {e}"))?;
                if !mhz.is_finite() || !(1.0..=5.0e3).contains(&mhz) {
                    return Err(format!(
                        "--pre-proton-mhz {mhz} must be finite in [1, 5000] MHz"
                    ));
                }
                pre_proton_mhz = Some(mhz);
            }
            "--pre-tether-pad" => {
                let v = next_val(&mut i, flag)?;
                let pad: f64 = v
                    .parse()
                    .map_err(|e| format!("invalid --pre-tether-pad {v:?}: {e}"))?;
                if !pad.is_finite() || !(0.0..=15.0).contains(&pad) {
                    return Err(format!(
                        "--pre-tether-pad {pad} must be finite in [0.0, 15.0] A"
                    ));
                }
                pre_tether_pad_a = Some(pad);
            }
            "--pre-rigid-metal" => {
                pre_tether_pad_a = Some(0.0);
            }
            "--pre-spin" => {
                let v = next_val(&mut i, flag)?;
                let s: f64 = v
                    .parse()
                    .map_err(|e| format!("invalid --pre-spin {v:?}: {e}"))?;
                if !s.is_finite() || !(0.5..=8.0).contains(&s) {
                    return Err(format!("--pre-spin {s} must be finite in [0.5, 8.0]"));
                }
                pre_spin_s = Some(s);
            }
            "--csp-scoring" => {
                let v = next_val(&mut i, flag)?;
                score_cfg.csp_scoring_mode = match v.trim().to_ascii_lowercase().as_str() {
                    "burial" | "burial-contact" | "contact" => CspScoringMode::BurialAndContactOnly,
                    "forward" | "forward-shift" | "full" => {
                        CspScoringMode::FullForwardShiftAndBurial
                    }
                    other => {
                        return Err(format!(
                            "unsupported --csp-scoring {other:?}; expected 'burial' or 'forward'"
                        ));
                    }
                };
            }
            "--rdc" => {
                rdc_path = Some(PathBuf::from(next_val(&mut i, flag)?));
            }
            "--pcs" => {
                pcs_path = Some(PathBuf::from(next_val(&mut i, flag)?));
            }
            "--ligand-smiles" | "--smiles" => {
                let v = next_val(&mut i, flag)?;
                let (cid, smi) = split_optional_id_prefix(v, "L");
                let (spec, mol) =
                    ligand_spec_from_octet_str(smi, Some("smi")).map_err(String::from)?;
                raw_input.ligand_molecules.insert(cid.to_string(), mol);
                raw_input.entities.push(CofoldEntitySpec::Ligand {
                    chain_id: cid.to_string(),
                    spec,
                });
            }
            "--ligand-file" => {
                let v = next_val(&mut i, flag)?;
                let (cid, p_str) = split_optional_id_prefix(v, "L");
                let (spec, mol) =
                    ligand_spec_from_octet_file(Path::new(p_str)).map_err(String::from)?;
                raw_input.ligand_molecules.insert(cid.to_string(), mol);
                raw_input.entities.push(CofoldEntitySpec::Ligand {
                    chain_id: cid.to_string(),
                    spec,
                });
            }
            "--receptor-chain" => {
                raw_input.receptor_chain_id = Some(next_val(&mut i, flag)?.trim().to_string());
            }
            "--binder-chain" => {
                raw_input.binder_chain_id = Some(next_val(&mut i, flag)?.trim().to_string());
            }
            "--reference-pose" | "--native" => {
                ref_pose_path = Some(PathBuf::from(next_val(&mut i, flag)?));
            }
            "--w-csp" => {
                let v = next_val(&mut i, flag)?;
                score_cfg.w_csp = v.parse().map_err(|e| format!("invalid --w-csp: {e}"))?;
            }
            "--w-noe" => {
                let v = next_val(&mut i, flag)?;
                score_cfg.w_noe_pre = v.parse().map_err(|e| format!("invalid --w-noe: {e}"))?;
            }
            "--w-rdc" => {
                let v = next_val(&mut i, flag)?;
                score_cfg.w_rdc = v.parse().map_err(|e| format!("invalid --w-rdc: {e}"))?;
            }
            "--w-pcs" => {
                let v = next_val(&mut i, flag)?;
                score_cfg.w_pcs = v.parse().map_err(|e| format!("invalid --w-pcs: {e}"))?;
            }
            "--rdc-free-frac" => {
                let v = next_val(&mut i, flag)?;
                let f: f64 = v
                    .parse()
                    .map_err(|e| format!("invalid --rdc-free-frac: {e}"))?;
                if !f.is_finite() || !(0.0..=0.5).contains(&f) {
                    return Err("--rdc-free-frac must be in [0.0, 0.5]".to_string());
                }
                score_cfg.rdc_free_fraction = f;
            }
            "--exclude-pdb" => {
                push_csv_items(
                    &mut score_cfg.firewall_query.query_pdb_ids,
                    next_val(&mut i, flag)?,
                );
            }
            "--exclude-bmrb" => {
                push_csv_items(
                    &mut score_cfg.firewall_query.query_bmrb_ids,
                    next_val(&mut i, flag)?,
                );
            }
            "--exclude-topology" => {
                let v = next_val(&mut i, flag)?.trim();
                if !v.is_empty() {
                    score_cfg.firewall_query.query_topology_id = Some(v.to_string());
                }
            }
            "--max-seq-identity" => {
                let v = next_val(&mut i, flag)?;
                let sid: f64 = v
                    .parse()
                    .map_err(|e| format!("invalid --max-seq-identity: {e}"))?;
                if !sid.is_finite() || !(0.0..=1.0).contains(&sid) {
                    return Err("--max-seq-identity must be in [0.0, 1.0]".to_string());
                }
                score_cfg.struct_shift_config.max_seq_identity = sid;
            }
            "--format" => {
                explicit_format = Some(ScoreFormat::from_token(next_val(&mut i, flag)?)?);
            }
            "-o" | "--out" => {
                out_path = Some(PathBuf::from(next_val(&mut i, flag)?));
            }
            other => {
                return Err(format!(
                    "unknown flag {other:?} for `ferric cofold score`\n\n{}",
                    cofold_usage()
                ));
            }
        }
        i += 1;
    }

    if let Some(ref dir) = pose_dir {
        let mut dir_files: Vec<PathBuf> = fs::read_dir(dir)
            .map_err(|e| format!("failed to read --pose-dir '{}': {e}", dir.display()))?
            .filter_map(|ent| ent.ok().map(|e| e.path()))
            .filter(|p| {
                p.is_file()
                    && p.extension().and_then(|s| s.to_str()).is_some_and(|ext| {
                        matches!(ext.to_ascii_lowercase().as_str(), "cif" | "mmcif" | "pdb")
                    })
            })
            .collect();
        dir_files.sort();
        pose_paths.extend(dir_files);
    }

    if pose_paths.is_empty() {
        return Err(format!(
            "missing candidate poses; pass --poses <p1,p2>, --pose <path>, or --pose-dir <dir>\n\n{}",
            cofold_usage()
        ));
    }

    let mut poses = Vec::with_capacity(pose_paths.len());
    for p in &pose_paths {
        let mut pose_in = StructurePoseInput::from_file(p).map_err(String::from)?;
        if let Some(conf_p) = conf_map.get(&pose_in.pose_id) {
            let c_txt = read_utf8_bounded(conf_p, "confidence JSON")?;
            let conf = CofoldConfidenceMetadata::parse_json_str(&c_txt).map_err(String::from)?;
            pose_in = pose_in.with_confidence(conf);
        }
        poses.push(pose_in);
    }

    if let Some(ref ap) = apo_path {
        let txt = read_utf8_bounded(ap, "apo structure")?;
        let ext = ap
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_ascii_lowercase());
        let model = StructureModel::parse_str(&txt, ext.as_deref(), false)?;
        score_cfg.apo_receptor_coord_text = Some(txt);
        score_cfg.apo_receptor_format_hint = ext;
        raw_input.apo_receptor_model = Some(model);
    }

    if let Some(ref rp_ref) = ref_pose_path {
        let txt = read_utf8_bounded(rp_ref, "reference pose")?;
        let ext = rp_ref
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_ascii_lowercase());
        score_cfg.reference_complex_pose = Some((txt, ext));
    }

    let mut bundle: CofoldRestraintBundle = if let Some(ref sc_p) = restraints_json_path {
        let txt = read_utf8_bounded(sc_p, "NMR restraints sidecar JSON")?;
        serde_json::from_str(&txt).map_err(|e| {
            format!(
                "failed to parse restraints sidecar '{}': {e}",
                sc_p.display()
            )
        })?
    } else {
        if raw_input.apo_receptor_model.is_none() {
            // Use first pose's receptor chain coordinates for residue indexing when no apo file was supplied
            if let Some(first_pose) = poses.first() {
                if let Ok(m) = StructureModel::parse_str(
                    &first_pose.coord_text,
                    first_pose.format_hint.as_deref(),
                    false,
                ) {
                    raw_input.apo_receptor_model = Some(m);
                }
            }
        }
        if let Some(ref cp) = csp_path {
            let txt = read_utf8_bounded(cp, "CSP file")?;
            let (rows, kd) = parse_csp_restraint_str(&txt).map_err(String::from)?;
            raw_input.raw_csp_rows = rows;
            raw_input.experimental_kd_um = kd;
        }
        if let Some(ref np) = noesy_path {
            let txt = read_utf8_bounded(np, "NOE TSV")?;
            let noes = parse_intermolecular_noe_tsv_str(&txt).map_err(String::from)?;
            raw_input.raw_distance_restraints.extend(noes);
        }
        let default_rec = raw_input
            .receptor_chain_id
            .clone()
            .unwrap_or_else(|| "A".to_string());
        if let Some(ref pp) = pre_path {
            let txt = read_utf8_bounded(pp, "PRE TSV")?;
            let anc_ref = pre_anchor
                .as_ref()
                .map(|(c, s, a)| (c.as_str(), *s, a.as_str()));
            let pres = parse_intermolecular_pre_tsv_str_with_overrides(
                &txt,
                anc_ref,
                &default_rec,
                pre_tau_c_ns,
                pre_proton_mhz,
                pre_spin_s,
                pre_tether_pad_a,
                extract_cfg.pre_gamma2_upper_margin_a,
            )
            .map_err(String::from)?;
            raw_input.raw_distance_restraints.extend(pres);
        }
        if let Some(ref rdc_p) = rdc_path {
            let txt = read_utf8_bounded(rdc_p, "RDC TSV")?;
            raw_input.rdc_media =
                parse_cofold_rdc_tsv_str(&txt, &default_rec).map_err(String::from)?;
        }
        if let Some(ref pcs_p) = pcs_path {
            let txt = read_utf8_bounded(pcs_p, "PCS TSV")?;
            raw_input.pcs_tags =
                parse_cofold_pcs_tsv_str(&txt, &default_rec).map_err(String::from)?;
        }
        extract_nmr_interface_restraints(&raw_input, &extract_cfg).map_err(String::from)?
    };

    // Allow augmenting a loaded `.nmr_restraints.json` bundle with extra CLI `--rdc`/`--pcs`/`--ligand-*` overrides
    if restraints_json_path.is_some() {
        for (k, m) in raw_input.ligand_molecules {
            bundle.ligand_molecules.insert(k, m);
        }
        if let Some(ref rdc_p) = rdc_path {
            let txt = read_utf8_bounded(rdc_p, "RDC TSV")?;
            bundle.rdc_media =
                parse_cofold_rdc_tsv_str(&txt, &bundle.receptor_chain_id).map_err(String::from)?;
        }
        if let Some(ref pcs_p) = pcs_path {
            let txt = read_utf8_bounded(pcs_p, "PCS TSV")?;
            bundle.pcs_tags =
                parse_cofold_pcs_tsv_str(&txt, &bundle.receptor_chain_id).map_err(String::from)?;
        }
    }

    let report = score_cofold_poses(&poses, &bundle, &score_cfg).map_err(String::from)?;
    let fmt = match (explicit_format, out_path.as_deref()) {
        (Some(f), _) => f,
        (None, Some(p)) => ScoreFormat::from_path(p)?,
        (None, None) => ScoreFormat::Tsv,
    };
    let rendered = match fmt {
        ScoreFormat::Tsv => format_pose_score_tsv(&report),
        ScoreFormat::Json => format_pose_score_json(&report).map_err(String::from)?,
    };

    if let Some(ref out_p) = out_path {
        write_file_creating_dirs(out_p, &rendered)?;
        let verdict_label = match report.verdict {
            PoseVerdict::RankedConclusive => "RankedConclusive",
            PoseVerdict::InconclusiveOrAllViolated => "InconclusiveOrAllViolated",
        };
        eprintln!(
            "Scored {} poses (verdict={}, top_pose={}) -> {}",
            report.n_poses_evaluated,
            verdict_label,
            report.top_pose_id.as_deref().unwrap_or("ABSTAINED"),
            out_p.display()
        );
    } else {
        print!("{rendered}");
    }

    Ok(())
}
