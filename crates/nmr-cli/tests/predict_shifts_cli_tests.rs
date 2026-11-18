//! End-to-end integration tests for `ferric predict-shifts` (`crates/nmr-cli/src/predict_shifts_cmd.rs`
//! and top-level `ferric` binary subcommand dispatch).
//!
//! # Clean-Room Provenance
//! - Shen, Y. & Bax, A. (*J. Biomol. NMR* **2010**, 48, 13-22).
//! - Haigh, C. W. & Mallion, R. B. (*Mol. Phys.* **1972**, 24, 955-970) & Case, D. A. (*J. Biomol. NMR* **1995**, 6, 341-346).
//! - Empirical inverse-cube H-bond shift term inspired by Wagner, G., Pardi, A. & Bax, A. (*J. Am. Chem. Soc.* **1983**, 105, 5948-5949).
//! - McConnell, H. M. (*J. Chem. Phys.* **1957**, 27, 226-229); Buckingham, A. D. (*Can. J. Chem.* **1960**, 38, 300-307);
//!   MacKerell, A. D. Jr. et al. (*J. Phys. Chem. B* **1998**, 102, 3586-3616, CHARMM22 partial charges);
//!   Sitkoff, D. & Case, D. A. (*Prog. Nucl. Magn. Reson. Spectrosc.* **1998**, 32, 165-190).
//! - Sahakyan, A. B., Vranken, W. F., Cavalli, A. & Vendruscolo, M. (*Angew. Chem. Int. Ed.* **2011**, 50, 9620-9623).

#[path = "../src/predict_shifts_cmd.rs"]
mod predict_shifts_cmd;

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::{self, Command};

use nmr_core::nef::read_nef_file;
use nmr_core::nmrstar::read_star_all_atoms;
use nmr_core::struct_shifts::StructShiftPredictionReport;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

#[test]
fn test_predict_shifts_cli_help_and_argument_validation() {
    predict_shifts_cmd::run_predict_shifts_cli(&["--help".to_string()])
        .expect("--help should succeed");
    assert!(predict_shifts_cmd::run_predict_shifts_cli(&[]).is_err());
    assert!(predict_shifts_cmd::run_predict_shifts_cli(&["--bogus-flag".to_string()]).is_err());
    assert!(predict_shifts_cmd::run_predict_shifts_cli(&[
        "--structure".to_string(),
        "nonexistent_model.pdb".to_string()
    ])
    .is_err());
    assert!(predict_shifts_cmd::run_predict_shifts_cli(&[
        "--structure".to_string(),
        repo_root()
            .join("data/pdb_refs/1d3z.pdb")
            .to_string_lossy()
            .into_owned(),
        "--ph".to_string(),
        "99.0".to_string(),
    ])
    .is_err());
}

#[test]
fn test_predict_shifts_cli_star_nef_tsv_json_and_mmcif_end_to_end() {
    let root = repo_root();
    let pdb_1d3z = root.join("data/pdb_refs/1d3z.pdb");

    let dir = env::temp_dir().join(format!("ferric_predict_shifts_cli_{}", process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    // 1. Export NMR-STAR v3 (.str) with --methyl-ilvmat and explicit firewall exclusions
    let out_star = dir.join("ubq_pred.str");
    predict_shifts_cmd::run_predict_shifts_cli(&[
        "--structure".to_string(),
        pdb_1d3z.to_string_lossy().into_owned(),
        "--ph".to_string(),
        "6.5".to_string(),
        "--temp-k".to_string(),
        "298.15".to_string(),
        "--methyl-ilvmat".to_string(),
        "--exclude-pdb".to_string(),
        "1D3Z,1UBQ".to_string(),
        "--exclude-bmrb".to_string(),
        "17769,5387".to_string(),
        "--exclude-topology".to_string(),
        "3.10.20".to_string(),
        "-o".to_string(),
        out_star.to_string_lossy().into_owned(),
    ])
    .expect("predict-shifts .str export");

    let star_ref = read_star_all_atoms(&out_star, None).expect("parse exported .str");
    assert_eq!(star_ref.sequence.len(), 76);
    assert!(star_ref.shifts.contains_key(&(1, "CA".to_string())));
    assert!(star_ref.shifts.contains_key(&(49, "CD1".to_string())));
    assert!(star_ref.shifts.contains_key(&(49, "HD11".to_string())));

    // 2. Export CCPN NEF v1.1 (.nef)
    let out_nef = dir.join("ubq_pred.nef");
    predict_shifts_cmd::run_predict_shifts_cli(&[
        "--structure".to_string(),
        pdb_1d3z.to_string_lossy().into_owned(),
        "--methyl-ilvmat".to_string(),
        "-o".to_string(),
        out_nef.to_string_lossy().into_owned(),
    ])
    .expect("predict-shifts .nef export");

    let nef_doc = read_nef_file(&out_nef).expect("parse exported .nef");
    assert_eq!(nef_doc.shift_lists.len(), 1);
    assert!(nef_doc.shift_lists[0]
        .shifts
        .iter()
        .any(|s| s.atom_name == "CA" && s.value_uncertainty.is_some()));
    assert!(nef_doc.shift_lists[0]
        .shifts
        .iter()
        .any(|s| s.atom_name == "HD2%"));

    // 3. Export JSON (.json) via the compiled `ferric` binary and verify schema + HomologFirewall audit
    let out_json = dir.join("ubq_pred.json");
    let status = Command::new(env!("CARGO_BIN_EXE_ferric"))
        .args([
            "predict-shifts",
            "--structure",
            pdb_1d3z.to_str().unwrap(),
            "--ph",
            "6.5",
            "--temp-k",
            "298.15",
            "--methyl-ilvmat",
            "--exclude-pdb",
            "1D3Z,1UBQ",
            "--exclude-bmrb",
            "17769,5387",
            "--exclude-topology",
            "3.10.20",
            "-o",
            out_json.to_str().unwrap(),
        ])
        .status()
        .expect("spawn ferric predict-shifts");
    assert!(status.success(), "ferric predict-shifts binary failed");

    let json_text = fs::read_to_string(&out_json).unwrap();
    let report: StructShiftPredictionReport =
        serde_json::from_str(&json_text).expect("deserialize StructShiftPredictionReport");
    assert_eq!(report.residues.len(), 76);
    assert_eq!(report.exclusion_list_sha256.len(), 64);
    assert!(report.max_retained_seq_identity <= 0.25 + 1e-12);
    assert!(report.include_methyl_ilvmat);

    // 4. ModelArchive mmCIF (.cif) input (`>= 8` CA atoms for Layer 4 TM-score) + TSV output check,
    //    AND verify `< 8` CA query fails closed (`21.W1:P1-4`)!
    let short4_cif_path = dir.join("short4_af.cif");
    let short4_cif = "\
data_SHORT4_AF
loop_
_ma_qa_metric.id
_ma_qa_metric.name
_ma_qa_metric.mode
_ma_qa_metric.type
1 pLDDT local pLDDT
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.B_iso_or_equiv
_atom_site.pdbx_PDB_model_num
ATOM 1 N  . ALA A 1 0.000 0.000 0.000 88.0 1
ATOM 2 CA . ALA A 1 1.458 0.000 0.000 88.0 1
ATOM 3 C  . ALA A 1 2.009 1.420 0.000 88.0 1
ATOM 4 N  . VAL A 2 1.250 2.510 0.000 91.0 1
ATOM 5 CA . VAL A 2 1.820 3.850 0.000 91.0 1
ATOM 6 C  . VAL A 2 3.340 3.820 0.000 91.0 1
ATOM 7 N  . GLY A 3 3.950 2.640 0.000 12.0 1
ATOM 8 CA . GLY A 3 5.400 2.550 0.000 12.0 1
ATOM 9 C  . GLY A 3 6.020 3.940 0.000 12.0 1
ATOM 10 N . ALA A 4 7.340 4.020 0.000 0.0 1
ATOM 11 CA . ALA A 4 8.050 5.290 0.000 0.0 1
ATOM 12 C  . ALA A 4 9.570 5.120 0.000 0.0 1
#
loop_
_ma_qa_metric_local.label_asym_id
_ma_qa_metric_local.label_seq_id
_ma_qa_metric_local.metric_id
_ma_qa_metric_local.metric_value
A 1 1 88.0
A 2 1 91.0
A 3 1 12.0
A 4 1 0.0
";
    fs::write(&short4_cif_path, short4_cif).unwrap();
    let short_err = predict_shifts_cmd::run_predict_shifts_cli(&[
        "--structure".to_string(),
        short4_cif_path.to_string_lossy().into_owned(),
    ])
    .unwrap_err();
    assert!(
        short_err.contains("Layer 4") && short_err.contains(">= 8 CA atoms"),
        "expected Layer 4 fail-closed error on 4-residue model, got: {short_err}"
    );

    // Real committed AFDB v6 `AF-P0CG48-F1-model_v6.cif` -> TSV output check
    let real_af_cif = root.join("data/afdb/AF-P0CG48-F1-model_v6.cif");
    let out_tsv = dir.join("afdb_p0cg48.tsv");
    predict_shifts_cmd::run_predict_shifts_cli(&[
        "--structure".to_string(),
        real_af_cif.to_string_lossy().into_owned(),
        "--exclude-uniprot".to_string(),
        "P0CG48".to_string(),
        "-o".to_string(),
        out_tsv.to_string_lossy().into_owned(),
    ])
    .expect("predict-shifts real AFDB .cif -> .tsv");

    let tsv_text = fs::read_to_string(&out_tsv).unwrap();
    assert!(tsv_text.contains("# exclusion_list_sha256:"));
    assert!(tsv_text.contains("provenance: Computational"));

    let _ = fs::remove_dir_all(&dir);
}
