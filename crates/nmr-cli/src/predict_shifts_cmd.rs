//! CLI subcommand `ferric predict-shifts` (`T21.2`, `docs/specs/21-alphafold-integrative-nmr.md` section 2.1(7)):
//! clean-room 3D protein structure (`.pdb` / `.cif`) to backbone (`H^N`, `15N`, `13CA`, `13CB`,
//! `13C'`, `1HA`) and `ILVMAT` sidechain methyl (`1H^m`, `13C^m`) chemical-shift forward predictor.
//!
//! # Literature & Clean-Room Provenance
//! - Shen, Y. & Bax, A. (*J. Biomol. NMR* **2010**, 48, 13-22).
//! - Haigh, C. W. & Mallion, R. B. (*Mol. Phys.* **1972**, 24, 955-970) & Case, D. A. (*J. Biomol. NMR* **1995**, 6, 341-346).
//! - Wagner, G., Pardi, A. & Bax, A. (*J. Am. Chem. Soc.* **1983**, 105, 5948-5949).
//! - McConnell, H. M. (*J. Chem. Phys.* **1957**, 27, 226-229); Buckingham, A. D. (*Can. J. Chem.* **1960**, 38, 300-307);
//!   Sitkoff, D. & Case, D. A. (*Prog. Nucl. Magn. Reson. Spectrosc.* **1998**, 32, 165-190).
//! - Sahakyan, A. B., Vranken, W. F., Cavalli, A. & Vendruscolo, M. (*Angew. Chem. Int. Ed.* **2011**, 50, 9620-9623).

use std::fs;
use std::path::{Path, PathBuf};

use nmr_core::struct_shifts::{
    predict_structure_shifts_with_firewall, StructShiftConfig, StructShiftPredictionReport,
    StructureModel,
};
use nmr_core::torsion_db::HomologFirewallQuery;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OutputFormat {
    Star,
    Nef,
    Tsv,
    Json,
}

impl OutputFormat {
    fn from_token(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "str" | "star" | "nmrstar" | "nmr-star" => Ok(Self::Star),
            "nef" => Ok(Self::Nef),
            "tsv" | "tab" => Ok(Self::Tsv),
            "json" => Ok(Self::Json),
            other => Err(format!(
                "unsupported output format {other:?}; expected str, nef, tsv, or json"
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
            "str" | "star" => Ok(Self::Star),
            "nef" => Ok(Self::Nef),
            "tsv" | "txt" => Ok(Self::Tsv),
            "json" => Ok(Self::Json),
            other => Err(format!(
                "unsupported output extension '.{other}' on '{}'; expected .str, .nef, .tsv, or .json",
                path.display()
            )),
        }
    }
}

pub fn predict_shifts_usage() -> &'static str {
    "ferric predict-shifts -- Clean-room 3D coordinate (.cif/.pdb) to NMR chemical-shift forward predictor\n\
     Usage:\n\
       ferric predict-shifts --structure <model.cif|model.pdb> [--ph 6.5] [--temp-k 298.15] \\\n\
                             [--methyl-ilvmat] [--exclude-pdb <ids>] [--exclude-bmrb <ids>] \\\n\
                             [--exclude-topology <id>] [-o <out.str|out.nef|out.tsv|out.json>]\n\n\
     Options:\n\
       --structure | --pdb | --cif <path>  Input 3D protein model (.cif/.mmcif or .pdb)\n\
       --ph <f64>                          Sample pH for Henderson-Hasselbalch coil titration [default: 6.5]\n\
       --temp-k <f64>                      Sample temperature in Kelvin [default: 298.15]\n\
       --methyl-ilvmat                     Also predict ILVMAT sidechain methyl (13C, 1H) shifts\n\
       --exclude-pdb <ids>                 Comma-separated PDB IDs excluded by Layer 1 HomologFirewall\n\
       --exclude-bmrb <ids>                Comma-separated BMRB IDs excluded by Layer 1 HomologFirewall\n\
       --exclude-uniprot <id>              UniProt accession excluded by Layer 1 HomologFirewall\n\
       --exclude-family <id>               CATH/superfamily ID excluded by Layer 1 HomologFirewall\n\
       --exclude-topology <id>             CATH C.A.T / ECOD fold group excluded by Layer 2 HomologFirewall\n\
       --max-seq-identity <f64>            Layer 3 sequence-identity cutoff in (0, 1] [default: 0.25]\n\
       --force-bfactor-plddt               Treat PDB tempFactor / mmCIF B_iso_or_equiv as pLDDT\n\
       --plddt <f64>                       Override uniform per-residue pLDDT in [0, 100]\n\
       --no-ring-current                   Disable Haigh-Mallion aromatic ring-current term (ablation)\n\
       --format <str|nef|tsv|json>         Explicit output format (auto-detected from -o extension)\n\
       -o | --out <path>                   Output shift path (.str, .nef, .tsv, or .json; stdout if omitted)"
}

fn push_csv_items(dst: &mut Vec<String>, raw: &str) {
    for part in raw.split(',') {
        let trimmed = part.trim();
        if !trimmed.is_empty() {
            dst.push(trimmed.to_string());
        }
    }
}

fn render_report(
    report: &StructShiftPredictionReport,
    fmt: OutputFormat,
    entry_name: &str,
) -> Result<String, String> {
    match fmt {
        OutputFormat::Star => Ok(report.to_nmrstar_str()),
        OutputFormat::Nef => Ok(report.to_nef_str(entry_name)),
        OutputFormat::Tsv => Ok(report.to_tsv_str()),
        OutputFormat::Json => report.to_json_str(),
    }
}

pub fn run_predict_shifts_cli(args: &[String]) -> Result<(), String> {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("{}", predict_shifts_usage());
        return Ok(());
    }

    let mut structure_path: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut explicit_format: Option<OutputFormat> = None;
    let mut cfg = StructShiftConfig::default();
    let mut fw_query = HomologFirewallQuery::default();

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
            "--structure" | "--pdb" | "--cif" | "--in" => {
                let v = next_val(&mut i, flag)?;
                structure_path = Some(PathBuf::from(v));
            }
            "-o" | "--out" => {
                let v = next_val(&mut i, flag)?;
                out_path = Some(PathBuf::from(v));
            }
            "--format" => {
                let v = next_val(&mut i, flag)?;
                explicit_format = Some(OutputFormat::from_token(v)?);
            }
            "--ph" => {
                let v = next_val(&mut i, flag)?;
                let ph: f64 = v.parse().map_err(|e| format!("invalid --ph {v:?}: {e}"))?;
                if !ph.is_finite() || !(0.5..=13.5).contains(&ph) {
                    return Err(format!("--ph {ph} out of physical bounds [0.5, 13.5]"));
                }
                cfg.sample_ph = ph;
            }
            "--temp-k" | "--temp" => {
                let v = next_val(&mut i, flag)?;
                let tk: f64 = v
                    .parse()
                    .map_err(|e| format!("invalid --temp-k {v:?}: {e}"))?;
                if !tk.is_finite() || !(250.0..=373.15).contains(&tk) {
                    return Err(format!(
                        "--temp-k {tk} out of physical bounds [250.0, 373.15] K"
                    ));
                }
                cfg.temperature_kelvin = tk;
            }
            "--methyl-ilvmat" => {
                cfg.include_methyl_ilvmat = true;
            }
            "--exclude-pdb" => {
                let v = next_val(&mut i, flag)?;
                push_csv_items(&mut fw_query.query_pdb_ids, v);
            }
            "--exclude-bmrb" => {
                let v = next_val(&mut i, flag)?;
                push_csv_items(&mut fw_query.query_bmrb_ids, v);
            }
            "--exclude-uniprot" => {
                let v = next_val(&mut i, flag)?;
                let trimmed = v.trim();
                if !trimmed.is_empty() {
                    fw_query.query_uniprot_id = Some(trimmed.to_string());
                }
            }
            "--exclude-family" => {
                let v = next_val(&mut i, flag)?;
                let trimmed = v.trim();
                if !trimmed.is_empty() {
                    fw_query.query_family_id = Some(trimmed.to_string());
                }
            }
            "--exclude-topology" => {
                let v = next_val(&mut i, flag)?;
                let trimmed = v.trim();
                if !trimmed.is_empty() {
                    fw_query.query_topology_id = Some(trimmed.to_string());
                }
            }
            "--max-seq-identity" => {
                let v = next_val(&mut i, flag)?;
                let sid: f64 = v
                    .parse()
                    .map_err(|e| format!("invalid --max-seq-identity {v:?}: {e}"))?;
                if !sid.is_finite() || !(0.0..=1.0).contains(&sid) {
                    return Err(format!(
                        "--max-seq-identity {sid} must be a finite fraction in [0.0, 1.0]"
                    ));
                }
                cfg.max_seq_identity = sid;
            }
            "--force-bfactor-plddt" => {
                cfg.force_bfactor_plddt = true;
            }
            "--plddt" => {
                let v = next_val(&mut i, flag)?;
                let p: f64 = v
                    .parse()
                    .map_err(|e| format!("invalid --plddt {v:?}: {e}"))?;
                if !p.is_finite() || !(0.0..=100.0).contains(&p) {
                    return Err(format!("--plddt {p} must be finite in [0.0, 100.0]"));
                }
                cfg.override_uniform_plddt = Some(p);
            }
            "--no-ring-current" => {
                cfg.enable_ring_current = false;
            }
            other => {
                return Err(format!(
                    "unknown flag {other:?} for `ferric predict-shifts`\n\n{}",
                    predict_shifts_usage()
                ));
            }
        }
        i += 1;
    }

    let Some(struct_path) = structure_path else {
        return Err(format!(
            "missing required --structure <model.cif|model.pdb>\n\n{}",
            predict_shifts_usage()
        ));
    };

    let fmt = match (explicit_format, out_path.as_deref()) {
        (Some(f), _) => f,
        (None, Some(p)) => OutputFormat::from_path(p)?,
        (None, None) => OutputFormat::Tsv,
    };

    let model = StructureModel::parse_file(&struct_path, cfg.force_bfactor_plddt)?;
    let report = predict_structure_shifts_with_firewall(&model, &fw_query, &cfg)?;

    let entry_name = struct_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("predicted_shifts");
    let rendered = render_report(&report, fmt, entry_name)?;

    if let Some(ref out_p) = out_path {
        if let Some(parent) = out_p.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|e| {
                    format!(
                        "failed to create output directory '{}': {e}",
                        parent.display()
                    )
                })?;
            }
        }
        fs::write(out_p, rendered)
            .map_err(|e| format!("failed to write '{}': {e}", out_p.display()))?;
        eprintln!(
            "Predicted shifts for {} residues (retained_templates={}, max_seq_id={:.3}, excl_sha256={}) -> {}",
            report.residues.len(),
            report.retained_template_count,
            report.max_retained_seq_identity,
            &report.exclusion_list_sha256[..12.min(report.exclusion_list_sha256.len())],
            out_p.display()
        );
    } else {
        print!("{rendered}");
    }

    Ok(())
}
