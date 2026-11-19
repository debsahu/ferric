//! Shared `21F:B1` pLDDT-provenance probe matrix (`FIX3` slice A).
//!
//! Included via `#[path]` from `crates/nmr-core/tests/af_concordance_tests.rs` (three core gates)
//! and `crates/nmr-gui/tests/mol3d_viewer_tests.rs` (all four gates incl. the GUI `mol3d` parser),
//! so every gate is probed with byte-identical inputs.
//!
//! Rule under test (single source: `nmr_core::af_concordance::PlddtProvenanceEvidence`):
//! B-factors / QA values are read as `pLDDT` only when
//! 1. a `_ma_qa_metric` whose name/type names `pLDDT` and which is local (no `global` in
//!    mode/name/type) has at least one matching `_ma_qa_metric_local` row, or
//! 2. a recognised predictor header (`data_AF-*` / `_entry.id AF-*` / `_software.name` naming
//!    AlphaFold, ColabFold or Boltz) is present AND the file declares no `_ma_qa_metric` and no
//!    `_ma_qa_metric_local` rows at all (any declared QA schema must supply local pLDDT rows).
//!
//! Generic ModelArchive blocks (`_ma_protocol_step`, `_ma_model_list`, `_ma_data`) and PDB-format
//! files (no machine-readable provenance) never qualify. Synthetic fixtures; the only real file is
//! the AFDB v6 ubiquitin model `data/afdb/AF-P0CG48-F1-model_v6.cif` (`CC-BY-4.0`).

#![allow(dead_code)]

use std::path::PathBuf;

/// One probe case: coordinate text and the expected unanimous gate decision.
pub struct AfProvenanceProbe {
    pub name: &'static str,
    pub text: String,
    /// `"cif"` or `"pdb"`.
    pub format: &'static str,
    /// `true` when every gate must accept the file as a predicted model with per-residue pLDDT.
    pub expect_predicted: bool,
    /// When accepted from synthetic data: the exact per-residue pLDDT the gates must report.
    pub expected_plddt: Option<Vec<f64>>,
}

/// Real-shaped SWISS-MODEL QMEANDisCo per-residue scores in `[0, 1]` (stored in `B_iso`).
pub const QMEAN_B: [f64; N] = [
    0.512, 0.833, 0.701, 0.640, 0.922, 0.588, 0.771, 0.690, 0.455, 0.804,
];
/// Synthetic per-residue pLDDT in `[0, 100]`.
/// Deliberately NOT proportional to [`QMEAN_B`], so laundering QMEAN x100 is detectable.
pub const PLDDT: [f64; N] = [92.4, 78.1, 61.2, 70.5, 88.0, 90.3, 95.1, 66.7, 50.2, 83.6];

/// Residues per synthetic probe (`>= 8` CA atoms so the `af_assign_priors` homolog firewall runs).
pub const N: usize = 10;
const RES: [&str; N] = [
    "MET", "GLN", "ILE", "PHE", "VAL", "LYS", "THR", "LEU", "THR", "GLY",
];

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root")
}

/// Residue frame origin on an ideal alpha-helix axis (2.3 A radius, 100 deg/residue, 1.5 A rise).
fn helix_origin(i: usize) -> (f64, f64, f64) {
    let th = (100.0f64 * i as f64).to_radians();
    (2.3 * th.cos(), 2.3 * th.sin(), 1.5 * i as f64)
}

/// Ten-residue backbone (`N`, `CA`, `C`, `O`) `_atom_site` loop with per-residue `B_iso`.
fn atom_site_cif(b: &[f64; N]) -> String {
    let mut s = String::from(
        "loop_\n_atom_site.group_PDB\n_atom_site.id\n_atom_site.type_symbol\n_atom_site.label_atom_id\n\
         _atom_site.label_comp_id\n_atom_site.label_asym_id\n_atom_site.label_seq_id\n\
         _atom_site.auth_seq_id\n_atom_site.auth_asym_id\n_atom_site.Cartn_x\n_atom_site.Cartn_y\n\
         _atom_site.Cartn_z\n_atom_site.occupancy\n_atom_site.B_iso_or_equiv\n\
         _atom_site.pdbx_PDB_model_num\n",
    );
    let mut serial = 1;
    for (i, res) in RES.iter().enumerate() {
        let (x0, y0, z0) = helix_origin(i);
        for (atom, el, dx, dy) in [
            ("N", "N", 0.0, 0.0),
            ("CA", "C", 1.46, 0.0),
            ("C", "C", 2.0, 1.4),
            ("O", "O", 1.6, 2.5),
        ] {
            s.push_str(&format!(
                "ATOM {serial} {el} {atom} {res} A {seq} {seq} A {x:.3} {y:.3} {z:.3} 1.00 {b:.3} 1\n",
                seq = i + 1,
                x = x0 + dx,
                y = y0 + dy,
                z = z0,
                b = b[i],
            ));
            serial += 1;
        }
    }
    s.push_str("#\n");
    s
}

/// Ten-residue backbone PDB `ATOM` records with per-residue `tempFactor`.
fn atom_pdb(b: &[f64; N]) -> String {
    let mut s = String::new();
    let mut serial = 1;
    for (i, res) in RES.iter().enumerate() {
        let (x0, y0, z0) = helix_origin(i);
        for (atom, el, dx, dy) in [
            ("N", "N", 0.0, 0.0),
            ("CA", "C", 1.46, 0.0),
            ("C", "C", 2.0, 1.4),
            ("O", "O", 1.6, 2.5),
        ] {
            s.push_str(&format!(
                "ATOM  {serial:>5}  {atom:<3} {res} A{seq:>4}    {x:>8.3}{y:>8.3}{z:>8.3}{occ:>6.2}{b:>6.2}          {el:>2}\n",
                seq = i + 1,
                x = x0 + dx,
                y = y0 + dy,
                z = z0,
                occ = 1.0,
                b = b[i],
            ));
            serial += 1;
        }
    }
    s.push_str("TER\nEND\n");
    s
}

const MA_GENERIC_BLOCKS: &str = "\
_ma_data.id 1
_ma_data.name 'SWISS-MODEL entry'
_ma_model_list.ordinal_id 1
_ma_model_list.model_name 'SWISS-MODEL QMEANDisCo'
loop_
_ma_protocol_step.ordinal_id
_ma_protocol_step.protocol_id
_ma_protocol_step.step_id
_ma_protocol_step.method_type
1 1 1 'template search'
2 1 2 'modeling'
#
";

const QMEAN_METRIC: &str = "\
loop_
_ma_qa_metric.id
_ma_qa_metric.name
_ma_qa_metric.type
_ma_qa_metric.mode
1 QMEANDisCo other local
#
";

fn local_rows(metric_id: &str, vals: &[f64; N]) -> String {
    let mut s = String::from(
        "loop_\n_ma_qa_metric_local.label_asym_id\n_ma_qa_metric_local.label_comp_id\n\
         _ma_qa_metric_local.label_seq_id\n_ma_qa_metric_local.metric_id\n\
         _ma_qa_metric_local.metric_value\n_ma_qa_metric_local.model_id\n\
         _ma_qa_metric_local.ordinal_id\n",
    );
    for (i, v) in vals.iter().enumerate() {
        s.push_str(&format!(
            "A {} {} {metric_id} {v:.3} 1 {}\n",
            RES[i],
            i + 1,
            i + 1
        ));
    }
    s.push_str("#\n");
    s
}

fn local_rows_two_metrics(id_a: &str, a: &[f64; N], id_b: &str, b: &[f64; N]) -> String {
    let mut s = String::from(
        "loop_\n_ma_qa_metric_local.label_asym_id\n_ma_qa_metric_local.label_comp_id\n\
         _ma_qa_metric_local.label_seq_id\n_ma_qa_metric_local.metric_id\n\
         _ma_qa_metric_local.metric_value\n_ma_qa_metric_local.model_id\n\
         _ma_qa_metric_local.ordinal_id\n",
    );
    let mut ord = 1;
    for i in 0..N {
        for (mid, v) in [(id_a, a[i]), (id_b, b[i])] {
            s.push_str(&format!("A {} {} {mid} {v:.3} 1 {ord}\n", RES[i], i + 1));
            ord += 1;
        }
    }
    s.push_str("#\n");
    s
}

const AF_SOFTWARE_LOOP: &str = "\
loop_
_software.pdbx_ordinal
_software.name
_software.version
_software.type
1 AlphaFold 2.3.2 package
#
";

/// Build the full `21F:B1` probe matrix.
pub fn af_provenance_probe_matrix() -> Vec<AfProvenanceProbe> {
    let mut out = Vec::new();

    // (1) Real-shaped SWISS-MODEL QMEANDisCo ModelArchive entry (B in 0..1).
    out.push(AfProvenanceProbe {
        name: "qmean_modelarchive",
        text: format!(
            "data_MA-BAK-CEPC-0001\n_entry.id MA-BAK-CEPC-0001\n{MA_GENERIC_BLOCKS}{QMEAN_METRIC}{}{}",
            atom_site_cif(&QMEAN_B),
            local_rows("1", &QMEAN_B)
        ),
        format: "cif",
        expect_predicted: false,
        expected_plddt: None,
    });
    // (2) Same + `_software.name AlphaFold` (template-based model citing AlphaFold).
    out.push(AfProvenanceProbe {
        name: "qmean_modelarchive_plus_alphafold_software",
        text: format!(
            "data_MA-BAK-CEPC-0002\n_entry.id MA-BAK-CEPC-0002\n{AF_SOFTWARE_LOOP}{MA_GENERIC_BLOCKS}{QMEAN_METRIC}{}{}",
            atom_site_cif(&QMEAN_B),
            local_rows("1", &QMEAN_B)
        ),
        format: "cif",
        expect_predicted: false,
        expected_plddt: None,
    });
    // (3) `data_AF-*` header but the only declared metric is QMEANDisCo.
    out.push(AfProvenanceProbe {
        name: "af_header_plus_qmean_metric",
        text: format!(
            "data_AF-P0CG48-F1\n_entry.id AF-P0CG48-F1\n{QMEAN_METRIC}{}{}",
            atom_site_cif(&QMEAN_B),
            local_rows("1", &QMEAN_B)
        ),
        format: "cif",
        expect_predicted: false,
        expected_plddt: None,
    });
    // (4) Generic ModelArchive blocks, no `_ma_qa_metric`, B in 0..1.
    out.push(AfProvenanceProbe {
        name: "modelarchive_no_metric",
        text: format!(
            "data_MA-BAK-CEPC-0003\n_entry.id MA-BAK-CEPC-0003\n{MA_GENERIC_BLOCKS}{}",
            atom_site_cif(&QMEAN_B)
        ),
        format: "cif",
        expect_predicted: false,
        expected_plddt: None,
    });
    // (5) SWISS-MODEL PDB export (QMEANDisCo in B column, REMARK text only).
    out.push(AfProvenanceProbe {
        name: "swissmodel_pdb",
        text: format!(
            "TITLE     SWISS-MODEL SERVER (https://swissmodel.expasy.org)\n\
             REMARK   3 MODEL INFORMATION\nREMARK   3  ENGIN   PROMOD3\n\
             REMARK   3  QMEANDISCO GLOBAL 0.77\nREMARK   3  TEMPLATE ALPHAFOLD DB MODEL\n{}",
            atom_pdb(&QMEAN_B)
        ),
        format: "pdb",
        expect_predicted: false,
        expected_plddt: None,
    });
    // (6) AF header + only a GLOBAL pLDDT metric (no per-residue rows), B in 0..100.
    out.push(AfProvenanceProbe {
        name: "plddt_global_only",
        text: format!(
            "data_AF-P0CG48-F1\n_entry.id AF-P0CG48-F1\nloop_\n_ma_qa_metric.id\n_ma_qa_metric.mode\n\
             _ma_qa_metric.name\n_ma_qa_metric.type\n1 global pLDDT pLDDT\n#\n\
             _ma_qa_metric_global.metric_id 1\n_ma_qa_metric_global.metric_value 77.23\n\
             _ma_qa_metric_global.model_id 1\n#\n{}",
            atom_site_cif(&PLDDT)
        ),
        format: "cif",
        expect_predicted: false,
        expected_plddt: None,
    });
    // (7a/7b) Lower-case `plddt` and upper-case `PLDDT` local metric with local rows (no header).
    for (nm, name) in [
        ("plddt", "local_plddt_lowercase"),
        ("PLDDT", "local_plddt_uppercase"),
    ] {
        out.push(AfProvenanceProbe {
            name,
            text: format!(
                "data_model_1\nloop_\n_ma_qa_metric.id\n_ma_qa_metric.mode\n_ma_qa_metric.name\n\
                 _ma_qa_metric.type\n1 local {nm} {nm}\n#\n{}{}",
                atom_site_cif(&PLDDT),
                local_rows("1", &PLDDT)
            ),
            format: "cif",
            expect_predicted: true,
            expected_plddt: Some(PLDDT.to_vec()),
        });
    }
    // (8) Mixed pLDDT + QMEANDisCo local rows; B carries QMEAN. pLDDT rows must win.
    out.push(AfProvenanceProbe {
        name: "mixed_plddt_and_qmean_rows",
        text: format!(
            "data_model_mixed\nloop_\n_ma_qa_metric.id\n_ma_qa_metric.mode\n_ma_qa_metric.name\n\
             _ma_qa_metric.type\n1 local pLDDT pLDDT\n2 local QMEANDisCo other\n#\n{}{}",
            atom_site_cif(&QMEAN_B),
            local_rows_two_metrics("1", &PLDDT, "2", &QMEAN_B)
        ),
        format: "cif",
        expect_predicted: true,
        expected_plddt: Some(PLDDT.to_vec()),
    });
    // (f3) pLDDT local + QMEANDisCo local declared, but only QMEAN local rows; B = QMEAN.
    out.push(AfProvenanceProbe {
        name: "f3_plddt_declared_only_qmean_rows",
        text: format!(
            "data_model_f3\nloop_\n_ma_qa_metric.id\n_ma_qa_metric.mode\n_ma_qa_metric.name\n\
             _ma_qa_metric.type\n1 local pLDDT pLDDT\n2 local QMEANDisCo other\n#\n{}{}",
            atom_site_cif(&QMEAN_B),
            local_rows("2", &QMEAN_B)
        ),
        format: "cif",
        expect_predicted: false,
        expected_plddt: None,
    });
    // (f5) pLDDT local metric declared, no `_ma_qa_metric_local` loop at all; B in 0..1.
    out.push(AfProvenanceProbe {
        name: "f5_plddt_declared_no_local_loop",
        text: format!(
            "data_model_f5\nloop_\n_ma_qa_metric.id\n_ma_qa_metric.mode\n_ma_qa_metric.name\n\
             _ma_qa_metric.type\n1 local pLDDT pLDDT\n#\n{}",
            atom_site_cif(&QMEAN_B)
        ),
        format: "cif",
        expect_predicted: false,
        expected_plddt: None,
    });
    // (e2) `global` only in the `type` column (no `mode` column): a global metric, not local.
    out.push(AfProvenanceProbe {
        name: "e2_global_in_type_no_mode",
        text: format!(
            "data_model_e2\nloop_\n_ma_qa_metric.id\n_ma_qa_metric.name\n_ma_qa_metric.type\n\
             1 pLDDT global\n#\n{}{}",
            atom_site_cif(&QMEAN_B),
            local_rows("1", &QMEAN_B)
        ),
        format: "cif",
        expect_predicted: false,
        expected_plddt: None,
    });
    // (9) Real AFDB v6 ubiquitin (`AF-P0CG48-F1`, local pLDDT rows) must still be accepted.
    let afdb = workspace_root().join("data/afdb/AF-P0CG48-F1-model_v6.cif");
    out.push(AfProvenanceProbe {
        name: "real_afdb_AF-P0CG48-F1_v6",
        text: std::fs::read_to_string(&afdb).expect("read data/afdb/AF-P0CG48-F1-model_v6.cif"),
        format: "cif",
        expect_predicted: true,
        expected_plddt: None,
    });
    // (10) ColabFold PDB output (B = pLDDT 0..100, no machine-readable provenance in PDB format).
    out.push(AfProvenanceProbe {
        name: "colabfold_pdb",
        text: format!(
            "REMARK   1 CREATED WITH COLABFOLD 1.5.5 (ALPHAFOLD2_PTM)\n{}",
            atom_pdb(&PLDDT)
        ),
        format: "pdb",
        expect_predicted: false,
        expected_plddt: None,
    });
    // (11) `_software.name ColabFold` mmCIF, no QA metric schema, B = pLDDT: header path accepts.
    out.push(AfProvenanceProbe {
        name: "colabfold_software_cif_no_metric",
        text: format!(
            "data_colabfold_model\nloop_\n_software.pdbx_ordinal\n_software.name\n_software.version\n\
             _software.type\n1 ColabFold 1.5.5 package\n#\n{}",
            atom_site_cif(&PLDDT)
        ),
        format: "cif",
        expect_predicted: true,
        expected_plddt: Some(PLDDT.to_vec()),
    });
    out
}

/// Residue names / seq ids of the synthetic probes (for `AfAssignPriors::from_strs`).
pub fn probe_sequence() -> (Vec<String>, Vec<i32>) {
    (
        RES.iter().map(|s| s.to_string()).collect(),
        (1..=N as i32).collect(),
    )
}

/// Decision and reported per-residue pLDDT of one gate.
#[derive(Clone, Debug, PartialEq)]
pub struct GateVerdict {
    pub gate: &'static str,
    pub predicted: bool,
    pub plddt: Option<Vec<f64>>,
    pub detail: String,
}

/// Evaluate the three nmr-core gates (`af_concordance`, `struct_shifts`, `af_assign_priors`).
pub fn core_gate_verdicts(p: &AfProvenanceProbe) -> Vec<GateVerdict> {
    use nmr_core::af_assign_priors::{
        text_has_predicted_model_evidence, AfAssignConfig, AfAssignPriors,
    };
    use nmr_core::af_concordance::{
        parse_af_structure_model_str, require_predicted_model, AfIngestOptions,
    };
    use nmr_core::struct_shifts::{require_predicted_structure_model, StructureModel};
    use nmr_core::torsion_db::HomologFirewallQuery;

    let mut out = Vec::new();

    // Gate 1: af_concordance (`ferric af-concordance`).
    let af = parse_af_structure_model_str(&p.text, &AfIngestOptions::default())
        .unwrap_or_else(|e| panic!("[{}] af_concordance parse failed: {e}", p.name));
    let af_gate = require_predicted_model(&af, false);
    out.push(GateVerdict {
        gate: "af_concordance",
        predicted: af_gate.is_ok(),
        plddt: af.plddt_per_residue.clone(),
        detail: format!(
            "provenance={:?} source={:?} diag={:?} gate={:?}",
            af.provenance, af.plddt_source, af.diagnostics, af_gate
        ),
    });

    // Gate 2: struct_shifts (`ferric struct-shifts`).
    let sm = StructureModel::parse_str(&p.text, Some(p.format), false)
        .unwrap_or_else(|e| panic!("[{}] struct_shifts parse failed: {e}", p.name));
    let ss_gate = require_predicted_structure_model(&sm, false);
    let ss_plddt: Option<Vec<f64>> = sm.residues.iter().map(|r| r.plddt).collect();
    out.push(GateVerdict {
        gate: "struct_shifts",
        predicted: ss_gate.is_ok(),
        plddt: ss_plddt,
        detail: format!(
            "provenance={:?} diag={:?} gate={:?}",
            sm.provenance, sm.diagnostics, ss_gate
        ),
    });

    // Gate 3: af_assign_priors (`ferric assign --af-structure`).
    let seq: Vec<String> = sm.residues.iter().map(|r| r.res_name.clone()).collect();
    let ids: Vec<i32> = sm.residues.iter().map(|r| r.seq_id).collect();
    let text_ev = text_has_predicted_model_evidence(&p.text);
    out.push(GateVerdict {
        gate: "af_assign_priors::text_has_predicted_model_evidence",
        predicted: text_ev,
        plddt: None,
        detail: String::new(),
    });
    let pri = AfAssignPriors::from_strs(
        &p.text,
        Some(p.format),
        &seq,
        Some(&ids),
        None,
        None,
        &HomologFirewallQuery::default(),
        &AfAssignConfig::default(),
    );
    out.push(GateVerdict {
        gate: "af_assign_priors",
        predicted: text_ev && pri.is_ok(),
        plddt: None,
        detail: format!(
            "text_evidence={text_ev} from_strs={:?}",
            pri.as_ref().map(|_| "Ok").map_err(|e| e.clone())
        ),
    });
    out
}

/// Check a gate verdict against the probe's expectation (decision and, if given, exact pLDDT).
pub fn check_gate_matches(p: &AfProvenanceProbe, v: &GateVerdict) -> Result<(), String> {
    if v.predicted != p.expect_predicted {
        return Err(format!(
            "[{}] gate {} predicted={} but expected {} ({})",
            p.name, v.gate, v.predicted, p.expect_predicted, v.detail
        ));
    }
    if !p.expect_predicted && v.plddt.is_some() && !v.gate.starts_with("af_assign_priors") {
        return Err(format!(
            "[{}] gate {} rejected the model but still exposes pLDDT {:?} ({})",
            p.name, v.gate, v.plddt, v.detail
        ));
    }
    if p.expect_predicted && !v.gate.starts_with("af_assign_priors") && v.plddt.is_none() {
        return Err(format!(
            "[{}] gate {} accepted the model but exposes no pLDDT ({})",
            p.name, v.gate, v.detail
        ));
    }
    if let (Some(want), Some(got)) = (&p.expected_plddt, &v.plddt) {
        let same = got.len() == want.len()
            // 1e-4 absorbs the GUI's `f32` B-factor storage (~1e-5 at 100); laundering QMEAN
            // x100 or substituting B for pLDDT rows differs by >= 1.
            && got.iter().zip(want).all(|(g, w)| (g - w).abs() < 1e-4);
        if !same {
            return Err(format!(
                "[{}] gate {} pLDDT {:?} != expected {:?} (QA values laundered?) ({})",
                p.name, v.gate, got, want, v.detail
            ));
        }
    }
    Ok(())
}

/// Run `verdicts` over the whole matrix and panic with every mismatch at once.
pub fn assert_matrix<F>(verdicts: F)
where
    F: Fn(&AfProvenanceProbe) -> Vec<GateVerdict>,
{
    let mut failures = Vec::new();
    for p in af_provenance_probe_matrix() {
        for v in verdicts(&p) {
            if let Err(e) = check_gate_matches(&p, &v) {
                failures.push(e);
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} provenance-gate mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
