//! Integration and regression tests for `T21.1` (`crates/nmr-core/src/af_concordance.rs`,
//! `crates/nmr-core/src/kabsch_sander_ss.rs`):
//! - Hostile mmCIF / PDB / `PAE` JSON `Err` guards (`<= 64 MiB`, `<= 4,096` pre-allocation cap,
//!   `NaN`/`Inf`, truncated `_ma_qa_metric_local` / `_atom_site` loops, $N \ne L$),
//! - Fail-closed coordinate provenance (`require_predicted_model`, `21.W1:P0-2`, `P2-6`): rejecting
//!   experimental X-ray/NMR/cryo-EM structures, headerless PDBs with `B <= 100`, `REMARK ALPHAFOLD`
//!   strings, constant/all-zero `B_iso`, and non-`pLDDT` ModelArchive metrics (`QMEANDisCo`),
//! - Clean-room Kabsch–Sander (1983) `8 -> 4`-state self-consistency check between `1D3Z` (NMR) and
//!   `1UBQ` (X-ray) Ubiquitin without reading `HELIX` or `SHEET` records, plus `PPII` (Adzhubei 2013),
//! - Original FERRIC `pLDDT`-vs-$S^2$ rigidity concordance heuristic (inspired by Fowler, Sljoka &
//!   Williamson 2020/2022, which used coordinate `FIRST` rigidity rather than `pLDDT`) & gap-free
//!   6-class concordance taxonomy (`AfNmrConcordanceClass` including `Indeterminate` and
//!   `ModelOrderedSolutionDisordered`, `21.W1:P1-1`, `A5`):
//!   - Real downloaded AFDB v6 `AF-P0CG48-F1-model_v6.cif` + `AF-P0CG48-F1-predicted_aligned_error_v6.json`
//!     verified by SHA-256 against `data/afdb/PROVENANCE.md` (`21.W1:P0-1`; pre-cutoff `<= 2018-04-30`
//!     fixture, not a cold post-cutoff generalization claim),
//!   - Synthetic unit / self-consistency checks (`21.W1:P1-2`) for disordered coil (`bmr6968`),
//!     planted high-`pLDDT` MoRF segments (`bmr15176`, `bmr51984`), and planted $\alpha \leftrightarrow \beta$
//!     conformational switches.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use nmr_core::af_concordance::{
    block_bootstrap_spearman_rho_ci95, compute_berjanskii_wishart_rci_s2,
    evaluate_af_nmr_concordance, format_af_concordance_json, format_af_concordance_tsv,
    parse_af_structure_model, parse_af_structure_model_str, parse_modelfree_s2_tsv_str,
    parse_pae_json_file, parse_pae_json_str, parse_pae_json_str_for_chain, require_predicted_model,
    tie_adjusted_spearman_rho, AfConcordanceConfig, AfIngestDiagnostic, AfIngestOptions,
    AfNmrConcordanceClass, ConcordanceSummary, CoordinateProvenance, ExperimentalRigiditySource,
    PaeJsonDialect, PlddtSource, MAX_AF_STRUCTURE_BYTES,
};
use nmr_core::kabsch_sander_ss::{
    classify_kabsch_sander, BackboneResidueCoords, KabschSanderState8, SecondaryStructure4,
};
use nmr_core::nmrstar::{read_star, Reference};
use nmr_core::noesy::PdbChain;
use nmr_core::report::{generate_quality_report_html, render_af_concordance_svg, ReportInput};
use nmr_core::structure::build_backbone_from_torsions;
use nmr_core::torsion_db::sha256_hex;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn strip_pdb_ss_headers(pdb_text: &str) -> String {
    pdb_text
        .lines()
        .filter(|l| !l.starts_with("HELIX ") && !l.starts_with("SHEET "))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Serialize a `PdbChain` + per-residue `pLDDT` vector into a canonical ModelArchive mmCIF string
/// with `_ma_qa_metric` + `_ma_qa_metric_local` (and decoy `B_iso = 15.0` so tests also prove
/// `_ma_qa_metric_local` takes strict precedence over `_atom_site.B_iso_or_equiv`).
fn build_af_mmcif_from_chain(entry_id: &str, chain: &PdbChain, plddts: &[f64]) -> String {
    assert_eq!(chain.residues.len(), plddts.len());
    let mut cif = String::with_capacity(chain.residues.len() * 320);
    cif.push_str(&format!("data_{entry_id}\n"));
    cif.push_str(&format!("_entry.id {entry_id}\n"));
    cif.push_str("_ma_protocol_step.step_id 1\n");
    cif.push_str(
        "loop_\n_ma_qa_metric.id\n_ma_qa_metric.name\n_ma_qa_metric.mode\n1 pLDDT local\n#\n",
    );
    cif.push_str("loop_\n_atom_site.group_PDB\n_atom_site.id\n_atom_site.type_symbol\n_atom_site.label_atom_id\n_atom_site.label_alt_id\n_atom_site.label_comp_id\n_atom_site.label_asym_id\n_atom_site.label_seq_id\n_atom_site.auth_seq_id\n_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n_atom_site.occupancy\n_atom_site.B_iso_or_equiv\n_atom_site.pdbx_pdb_model_num\n");

    let mut serial = 1usize;
    for (idx, r) in chain.residues.iter().enumerate() {
        let lseq = idx + 1;
        let aseq = r.seq_id;
        let mut push_atom = |sym: &str, aname: &str, xyz: [f64; 3]| {
            cif.push_str(&format!(
                "ATOM {serial} {sym} {aname} . {} A {lseq} {aseq} {:.3} {:.3} {:.3} 1.00 15.00 1\n",
                r.res_name, xyz[0], xyz[1], xyz[2]
            ));
            serial += 1;
        };
        push_atom("N", "N", r.n);
        push_atom("C", "CA", r.ca);
        push_atom("C", "C", r.c);
        push_atom("O", "O", r.o);
        if let Some(cb) = r.cb {
            push_atom("C", "CB", cb);
        }
    }
    cif.push_str("#\nloop_\n_ma_qa_metric_local.label_asym_id\n_ma_qa_metric_local.label_seq_id\n_ma_qa_metric_local.label_comp_id\n_ma_qa_metric_local.metric_id\n_ma_qa_metric_local.metric_value\n");
    for (idx, (r, &pl)) in chain.residues.iter().zip(plddts.iter()).enumerate() {
        cif.push_str(&format!("A {} {} 1 {:.2}\n", idx + 1, r.res_name, pl));
    }
    cif.push_str("#\n");
    cif
}

#[test]
fn test_hostile_mmcif_pdb_and_pae_inputs_return_err_without_panic() {
    let opts = AfIngestOptions::default();

    // 1. Empty, binary NUL, and > 64 MiB coordinate size cap
    assert!(parse_af_structure_model_str("", &opts).is_err());
    assert!(parse_af_structure_model_str("ATOM      1  N   ALA A   1\0", &opts).is_err());
    let oversized = " ".repeat(MAX_AF_STRUCTURE_BYTES + 1);
    let err_cap = parse_af_structure_model_str(&oversized, &opts).unwrap_err();
    assert!(
        err_cap.contains("MAX_AF_STRUCTURE_BYTES"),
        "expected size cap error, got: {err_cap}"
    );

    // 2. Truncated `_ma_qa_metric_local` loop (mid-row before `#`/EOF)
    let trunc_ma = "\
data_trunc_ma
loop_
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.auth_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
N ALA 1 0.0 0.0 0.0
CA ALA 1 1.46 0.0 0.0
C ALA 1 2.0 1.4 0.0
O ALA 1 1.5 2.4 0.0
loop_
_ma_qa_metric_local.label_asym_id
_ma_qa_metric_local.label_seq_id
_ma_qa_metric_local.metric_id
_ma_qa_metric_local.metric_value
A 1 1
";
    let err_trunc_ma = parse_af_structure_model_str(trunc_ma, &opts).unwrap_err();
    assert!(
        err_trunc_ma.contains("truncated mmCIF loop"),
        "got: {err_trunc_ma}"
    );

    // 3. Truncated `_atom_site` loop
    let trunc_atom = "\
data_trunc_atom
loop_
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.auth_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
N ALA 1 0.0 0.0
";
    assert!(parse_af_structure_model_str(trunc_atom, &opts).is_err());

    // 4. NaN / Inf in coordinates and pLDDT
    let nan_coord_cif = "\
data_nan
loop_
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.auth_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
N ALA 1 NaN 0.0 0.0
CA ALA 1 1.46 0.0 0.0
C ALA 1 2.0 1.4 0.0
";
    assert!(parse_af_structure_model_str(nan_coord_cif, &opts).is_err());

    let inf_plddt_cif = "\
data_inf_plddt
loop_
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.auth_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
N ALA 1 0.0 0.0 0.0
CA ALA 1 1.46 0.0 0.0
C ALA 1 2.0 1.4 0.0
loop_
_ma_qa_metric_local.label_asym_id
_ma_qa_metric_local.label_seq_id
_ma_qa_metric_local.metric_value
A 1 inf
";
    assert!(parse_af_structure_model_str(inf_plddt_cif, &opts).is_err());

    // 5. Crystallographic & solution NMR B-factor guard (`X-RAY DIFFRACTION`, `SOLUTION NMR`, `ELECTRON MICROSCOPY`)
    let ubq_xray = parse_af_structure_model(
        repo_root().join("data/fixtures/ubiquitin_1ubq_xray.pdb"),
        &opts,
    )
    .expect("1UBQ X-ray PDB must parse cleanly");
    assert_eq!(ubq_xray.provenance, CoordinateProvenance::Experimental);
    assert_eq!(ubq_xray.plddt_source, PlddtSource::Unavailable);
    assert!(
        ubq_xray.plddt_per_residue.is_none(),
        "X-ray thermal B-factors must never be misinterpreted as pLDDT"
    );
    assert!(ubq_xray
        .diagnostics
        .iter()
        .any(|d| matches!(d, AfIngestDiagnostic::ExperimentalBfactorSuppressed { .. })));

    let ubq_nmr = parse_af_structure_model(repo_root().join("data/pdb_refs/1d3z.pdb"), &opts)
        .expect("1D3Z NMR PDB must parse cleanly");
    assert_eq!(ubq_nmr.provenance, CoordinateProvenance::Experimental);
    assert!(ubq_nmr.plddt_per_residue.is_none());

    let cryoem_cif = "\
data_cryo
_exptl.method 'ELECTRON MICROSCOPY'
loop_
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.auth_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.B_iso_or_equiv
N ALA 1 0.0 0.0 0.0 82.0
CA ALA 1 1.46 0.0 0.0 82.0
C ALA 1 2.0 1.4 0.0 82.0
O ALA 1 1.5 2.4 0.0 82.0
";
    let cryo = parse_af_structure_model_str(cryoem_cif, &opts).unwrap();
    assert_eq!(cryo.provenance, CoordinateProvenance::Experimental);
    assert!(cryo.plddt_per_residue.is_none());

    let cryo_forced = parse_af_structure_model_str(
        cryoem_cif,
        &AfIngestOptions {
            target_chain_id: None,
            force_bfactor_plddt: true,
        },
    )
    .unwrap();
    assert_eq!(
        cryo_forced.plddt_source,
        PlddtSource::ForcedExperimentalBFactor
    );
    assert_eq!(cryo_forced.plddt_per_residue, Some(vec![82.0]));
    assert!(require_predicted_model(&cryo_forced, true).is_ok());
    assert!(require_predicted_model(&cryo_forced, false).is_err());

    // 6. Unit-interval `[0, 1] -> [0, 100]` auto-rescaling with audit diagnostic
    //    (requires a recognized AFDB/ColabFold header; bare `_ma_protocol_step` no longer qualifies!)
    let unit_b_cif = "\
data_AF-P00001-F1-model_v4
_entry.id AF-P00001-F1-model_v4
loop_
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.auth_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.B_iso_or_equiv
N ALA 1 0.0 0.0 0.0 0.92
CA ALA 1 1.46 0.0 0.0 0.92
C ALA 1 2.0 1.4 0.0 0.92
N GLY 2 3.2 1.2 0.0 0.45
CA GLY 2 4.5 1.8 0.0 0.45
C GLY 2 5.2 0.6 0.0 0.45
";
    let unit_m = parse_af_structure_model_str(unit_b_cif, &opts).unwrap();
    assert_eq!(unit_m.provenance, CoordinateProvenance::ComputationalModel);
    assert!(unit_m
        .diagnostics
        .contains(&AfIngestDiagnostic::RescaledUnitPlddtTo100));
    let plddts = unit_m.plddt_per_residue.unwrap();
    assert!((plddts[0] - 92.0).abs() < 1.0e-6);
    assert!((plddts[1] - 45.0).abs() < 1.0e-6);

    // 7. Hostile PAE JSON inputs: N = 0, N = 4097 (> 4096 pre-alloc cap), ragged, NaN, out-of-range, N != L
    assert!(parse_pae_json_str("").is_err());
    assert!(parse_pae_json_str(r#"{"pae": []}"#).is_err());
    assert!(parse_pae_json_str(r#"{"pae": [[1.0, 2.0], [3.0]]}"#).is_err());
    assert!(parse_pae_json_str(r#"{"pae": [[-0.5, 1.0], [1.0, 0.2]]}"#).is_err());
    assert!(parse_pae_json_str(r#"{"pae": [[65.0, 1.0], [1.0, 0.2]]}"#).is_err());
    assert!(parse_pae_json_str(r#"{"pae": [[NaN, 1.0], [1.0, 0.2]]}"#).is_err());

    let row_4097 = vec!["0.5"; 4097].join(",");
    let hostile_pae_4097 = format!("{{\"pae\": [[{row_4097}]]}}");
    let err_4097 = parse_pae_json_str(&hostile_pae_4097).unwrap_err();
    assert!(
        err_4097.contains("MAX_PAE_RESIDUES"),
        "expected pre-alloc cap error, got: {err_4097}"
    );
    let hostile_sparse_4097 = r#"[{"residue1": [4097], "residue2": [1], "distance": [1.2], "max_predicted_aligned_error": 31.75}]"#;
    assert!(parse_pae_json_str(hostile_sparse_4097).is_err());

    // 8. All 4 supported PAE JSON dialects + N != L mismatch check
    let afdb_v4 = r#"[{"predicted_aligned_error": [[0.2, 3.4], [3.8, 0.2]], "max_predicted_aligned_error": 31.75}]"#;
    let m_v4 = parse_pae_json_str(afdb_v4).unwrap();
    assert_eq!(m_v4.dialect, PaeJsonDialect::AfdbV3V4);
    assert_eq!(m_v4.n_residues, 2);
    assert!((m_v4.symmetrized(0, 1).unwrap() - 3.6).abs() < 1.0e-5);
    assert!(m_v4.validate_chain_length(3).is_err());

    let afdb_v1 = r#"[{"residue1": [1, 1, 2, 2], "residue2": [1, 2, 1, 2], "distance": [0.3, 4.0, 4.4, 0.3]}]"#;
    let m_v1 = parse_pae_json_str(afdb_v1).unwrap();
    assert_eq!(m_v1.dialect, PaeJsonDialect::AfdbLegacySparse);
    assert_eq!(m_v1.n_residues, 2);
    assert!((m_v1.symmetrized(0, 1).unwrap() - 4.2).abs() < 1.0e-5);

    let colab = r#"{"pae": [[0.2, 1.5], [1.7, 0.2]], "max_pae": 31.75, "plddt": [94.0, 88.5]}"#;
    let m_cf = parse_pae_json_str(colab).unwrap();
    assert_eq!(m_cf.dialect, PaeJsonDialect::ColabFoldAf2);
    assert_eq!(m_cf.plddt_from_json, Some(vec![94.0, 88.5]));

    // AF3 confidences.json: Chain A has 2 residues where residue 1 has 2 atom tokens, plus Chain B token
    let af3 = r#"{
        "pae": [
            [0.2, 0.4, 2.0, 12.0],
            [0.4, 0.2, 2.4, 12.5],
            [2.2, 2.6, 0.2, 11.0],
            [13.0, 13.5, 11.5, 0.2]
        ],
        "token_chain_ids": ["A", "A", "A", "B"],
        "token_res_ids": [1, 1, 2, 1]
    }"#;
    let m_af3 = parse_pae_json_str_for_chain(af3, Some("A"), Some(2)).unwrap();
    assert_eq!(m_af3.dialect, PaeJsonDialect::AlphaFold3Confidences);
    assert_eq!(m_af3.n_residues, 2);
    assert!((m_af3.get(0, 1).unwrap() - 2.2).abs() < 1.0e-5);
}

#[test]
fn test_kabsch_sander_ubiquitin_1d3z_and_1ubq_ge_96_pct_without_headers() {
    let opts = AfIngestOptions::default();

    let raw_1d3z =
        fs::read_to_string(repo_root().join("data/pdb_refs/1d3z.pdb")).expect("read 1d3z.pdb");
    let raw_1ubq = fs::read_to_string(repo_root().join("data/fixtures/ubiquitin_1ubq_xray.pdb"))
        .expect("read 1ubq.pdb");

    // Strip all HELIX and SHEET records and prove byte-identical classification
    let stripped_1d3z = strip_pdb_ss_headers(&raw_1d3z);
    let stripped_1ubq = strip_pdb_ss_headers(&raw_1ubq);
    assert!(!stripped_1d3z.contains("\nHELIX "));
    assert!(!stripped_1d3z.contains("\nSHEET "));

    let m_1d3z_with = parse_af_structure_model_str(&raw_1d3z, &opts).unwrap();
    let m_1d3z_no_hdr = parse_af_structure_model_str(&stripped_1d3z, &opts).unwrap();
    let m_1ubq_no_hdr = parse_af_structure_model_str(&stripped_1ubq, &opts).unwrap();

    let ss_1d3z_with = classify_kabsch_sander(&m_1d3z_with.backbone_coords_for_ss()).unwrap();
    let ss_1d3z = classify_kabsch_sander(&m_1d3z_no_hdr.backbone_coords_for_ss()).unwrap();
    let ss_1ubq = classify_kabsch_sander(&m_1ubq_no_hdr.backbone_coords_for_ss()).unwrap();

    assert_eq!(ss_1d3z_with.state8_string(), ss_1d3z.state8_string());
    assert_eq!(ss_1d3z_with.state4_string(), ss_1d3z.state4_string());
    assert_eq!(ss_1d3z.residues.len(), 76);
    assert_eq!(ss_1ubq.residues.len(), 76);

    // Canonical published Kabsch-Sander (1983) 8-state & 4-state secondary structure of 1UBQ (Vijay-Kumar 1987):
    //   beta1: 2..=6 (E), turn: 7..=10 (TT..), beta2: 12..=16 (E) + bulge 17,
    //   alpha1: 23..=34 (H), 3_10: 38..=40 (G), beta3: 41..=45 (E), beta4: 48..=49 (E/B),
    //   3_10: 57..=59 (G), beta5: 66..=71 (E).
    // Compare 1D3Z vs 1UBQ 3-class (Helix / Sheet / Coil+Ppii) and 4-state agreement:
    let agree_4state = ss_1d3z
        .residues
        .iter()
        .zip(ss_1ubq.residues.iter())
        .filter(|(a, b)| a.state4 == b.state4)
        .count();
    let pct_1d3z_vs_1ubq = 100.0 * (agree_4state as f64) / 76.0;

    // Canonical Ubiquitin Kabsch-Sander reference 4-state map on 1..=76:
    // E: 2..=7, 11..=17, 41..=45, 48..=49, 66..=71 (note: exact H-bond boundaries checked below)
    println!("1D3Z state8: {}", ss_1d3z.state8_string());
    println!("1D3Z state4: {}", ss_1d3z.state4_string());
    println!("1UBQ state8: {}", ss_1ubq.state8_string());
    println!("1UBQ state4: {}", ss_1ubq.state4_string());
    println!("1D3Z vs 1UBQ 4-state agreement: {agree_4state}/76 ({pct_1d3z_vs_1ubq:.2}%)");

    assert!(
        pct_1d3z_vs_1ubq >= 96.0,
        "expected >= 96.0% 4-state agreement between 1D3Z and 1UBQ without headers, got {pct_1d3z_vs_1ubq:.2}% ({agree_4state}/76)\n1D3Z={}\n1UBQ={}",
        ss_1d3z.state4_string(),
        ss_1ubq.state4_string()
    );

    // Also verify exact core elements on both 1UBQ and 1D3Z:
    // - Main alpha-helix 23..=34 is H
    for seq_id in 23..=34 {
        let idx = (seq_id - 1) as usize;
        assert_eq!(ss_1ubq.residues[idx].state8, KabschSanderState8::H);
        assert_eq!(ss_1d3z.residues[idx].state8, KabschSanderState8::H);
        assert_eq!(ss_1ubq.residues[idx].state4, SecondaryStructure4::Helix);
    }
    // - 3_10 helices 38..=40 and 57..=59 are G -> Helix
    for seq_id in [38, 39, 40, 57, 58, 59] {
        let idx = (seq_id - 1) as usize;
        assert_eq!(ss_1ubq.residues[idx].state8, KabschSanderState8::G);
        assert_eq!(ss_1ubq.residues[idx].state4, SecondaryStructure4::Helix);
    }
    // - Core beta-strands 2..=6, 12..=16, 41..=45, 66..=70 are Sheet (E/B)
    for seq_id in (2..=6).chain(12..=16).chain(41..=45).chain(66..=70) {
        let idx = (seq_id - 1) as usize;
        assert_eq!(
            ss_1ubq.residues[idx].state4,
            SecondaryStructure4::Sheet,
            "1UBQ residue {seq_id} must be Sheet"
        );
        assert_eq!(
            ss_1d3z.residues[idx].state4,
            SecondaryStructure4::Sheet,
            "1D3Z residue {seq_id} must be Sheet"
        );
    }

    // Verify Adzhubei et al. (2013) PPII helix (`phi = -75 deg, psi = +145 deg`, zero backbone H-bonds)
    let ppii_seq = vec![
        "ALA".to_string(),
        "PRO".to_string(),
        "ALA".to_string(),
        "PRO".to_string(),
        "ALA".to_string(),
        "ALA".to_string(),
    ];
    let ppii_ids: Vec<i32> = (1..=6).collect();
    let ppii_chain =
        build_backbone_from_torsions(&ppii_seq, &ppii_ids, &[-75.0; 6], &[145.0; 6], &[180.0; 6]);
    let ppii_ss =
        classify_kabsch_sander(&BackboneResidueCoords::from_pdb_chain(&ppii_chain, true)).unwrap();
    for idx in 1..5 {
        assert_eq!(
            ppii_ss.residues[idx].state4,
            SecondaryStructure4::Ppii,
            "extended (-75, +145) polyproline-II segment residue {} must be classified as Ppii",
            idx + 1
        );
    }
}

#[test]
fn test_real_afdb_p0cg48_v6_sha256_provenance_and_ubiquitin_concordance() {
    // `21.W1:P0-1`: Load the real downloaded AFDB v6 Polyubiquitin-C (`P0CG48`) mmCIF + PAE JSON
    // committed under `data/afdb/` (replacing the deleted hand-typed `AF-P62988-F1` pLDDT array),
    // verify exact SHA-256 digests against `data/afdb/PROVENANCE.md`, and compute concordance
    // metrics directly from the real AFDB coordinates and `_ma_qa_metric_local` pLDDT values.
    // Candor note (`A1`): Human Ubiquitin (`1UBQ` 1987, `1D3Z` 1998, `5B83` 2016) predates
    // AlphaFold2's `2018-04-30` PDB training cutoff; this test validates AFDB ingestion, provenance,
    // and concordance computation on real AFDB files, NOT cold post-cutoff generalization.
    let cif_path = repo_root().join("data/afdb/AF-P0CG48-F1-model_v6.cif");
    let pae_path = repo_root().join("data/afdb/AF-P0CG48-F1-predicted_aligned_error_v6.json");
    let prov_path = repo_root().join("data/afdb/PROVENANCE.md");

    let cif_bytes = fs::read(&cif_path).expect("read real AF-P0CG48-F1-model_v6.cif");
    let pae_bytes =
        fs::read(&pae_path).expect("read real AF-P0CG48-F1-predicted_aligned_error_v6.json");
    let prov_text = fs::read_to_string(&prov_path).expect("read data/afdb/PROVENANCE.md");

    let cif_sha = sha256_hex(&cif_bytes);
    let pae_sha = sha256_hex(&pae_bytes);
    assert_eq!(
        cif_sha,
        "02b8ea3d522fabee58abed221fad1a2101e9ad633ddecbe30bc96d158fc2c408"
    );
    assert_eq!(
        pae_sha,
        "edb163b4dcd75ecd253ef482c15555b524fda3fb91f75ef965ad5426762b1c64"
    );
    assert!(prov_text.contains(&cif_sha));
    assert!(prov_text.contains(&pae_sha));
    assert!(prov_text.contains("CC-BY-4.0"));
    assert!(prov_text.contains("2018-04-30"));

    let full_af_model =
        parse_af_structure_model(&cif_path, &AfIngestOptions::default()).expect("parse AFDB cif");
    require_predicted_model(&full_af_model, false)
        .expect("real AFDB v6 model must pass fail-closed require_predicted_model");
    assert_eq!(
        full_af_model.provenance,
        CoordinateProvenance::ComputationalModel
    );
    assert_eq!(full_af_model.plddt_source, PlddtSource::MaQaMetricLocal);
    assert_eq!(full_af_model.detected_uniprot_id.as_deref(), Some("P0CG48"));
    assert_eq!(full_af_model.residues.len(), 685);

    let full_pae =
        parse_pae_json_file(&pae_path, Some("A"), Some(685)).expect("parse AFDB v6 685x685 PAE");
    full_pae
        .validate_against_structure(&full_af_model)
        .expect("685x685 PAE matches AF-P0CG48-F1-model_v6");

    let shifts = read_star(
        repo_root().join("data/fixtures/bmr17769_ubiquitin_allatom.str"),
        None,
    )
    .expect("read bmr17769 ubiquitin shifts");

    let cfg = AfConcordanceConfig {
        exclude_pdb_ids: vec!["1D3Z".to_string(), "1UBQ".to_string()],
        exclude_bmrb_ids: vec!["17769".to_string(), "6457".to_string(), "5387".to_string()],
        exclude_uniprot_id: Some("P0CG48".to_string()),
        ..AfConcordanceConfig::default()
    };

    // 1. Evaluate directly on the unmodified 685-aa AFDB polyubiquitin model + 685x685 PAE
    let rep_full =
        evaluate_af_nmr_concordance(&full_af_model, &shifts, Some(&full_pae), None, &cfg).unwrap();
    assert_eq!(
        rep_full.rigidity_source,
        ExperimentalRigiditySource::BerjanskiiWishartRciS2
    );
    assert!(rep_full.max_retained_seq_identity <= 0.25);
    assert_eq!(rep_full.exclusion_list_sha256.len(), 64);
    assert!(!rep_full.excluded_template_ids.is_empty());
    assert_eq!(rep_full.pae_n_residues, Some(685));
    assert!(rep_full.spearman_rho.is_some());
    assert!(rep_full.spearman_rho_ci95.is_some());

    // 2. Slice the exact N-terminal monomeric Ubiquitin repeat (`1..=76`) directly from the real
    //    `AF-P0CG48-F1-model_v6.cif` coordinates and real `_ma_qa_metric_local` pLDDT values (zero
    //    hand-typed numbers!) so N/C termini (`Met1`, `Gly76`) match monomeric solution Ubiquitin.
    let full_chain = full_af_model.to_pdb_chain();
    let mono_chain = PdbChain {
        residues: full_chain.residues[..76].to_vec(),
    };
    let mono_plddts: Vec<f64> = full_af_model.plddt_per_residue.as_ref().unwrap()[..76].to_vec();
    let mono_cif = build_af_mmcif_from_chain("AF-P0CG48-F1_1_76", &mono_chain, &mono_plddts);
    let mono_model = parse_af_structure_model_str(&mono_cif, &AfIngestOptions::default()).unwrap();
    require_predicted_model(&mono_model, false).unwrap();

    let rep = evaluate_af_nmr_concordance(&mono_model, &shifts, None, None, &cfg).unwrap();
    let rho_whole = rep
        .spearman_rho
        .expect("whole-chain Spearman rho must be defined on real AF-P0CG48-F1[1..=76]");
    let ci_whole = rep
        .spearman_rho_ci95
        .expect("whole-chain block-bootstrap 95% CI must be defined");
    let rho_core = rep
        .ordered_core_spearman_rho
        .expect("ordered-core Spearman rho must be defined");
    let ci_core = rep
        .ordered_core_spearman_rho_ci95
        .expect("ordered-core block-bootstrap 95% CI must be defined");

    let assigned_total = rep.residues.len() - rep.n_unassigned_nmr;
    let frac_ordered = 100.0 * (rep.n_concordant_ordered as f64) / (assigned_total as f64);
    println!(
        "Real AFDB AF-P0CG48-F1[1..=76] vs BMRB 17769 (pre-cutoff fixture): whole rho_s={rho_whole:.4} [{:.4}, {:.4}], ordered-core (n={}) rho_s={rho_core:.4} [{:.4}, {:.4}], ordered={}/{} ({frac_ordered:.1}%), indeterminate={}",
        ci_whole[0],
        ci_whole[1],
        rep.n_ordered_core,
        ci_core[0],
        ci_core[1],
        rep.n_concordant_ordered,
        assigned_total,
        rep.n_indeterminate
    );

    assert!(ci_whole[0] <= ci_whole[1] && (-1.0..=1.0).contains(&rho_whole));
    assert!(ci_core[0] <= ci_core[1] && (-1.0..=1.0).contains(&rho_core));
    assert!(
        frac_ordered >= 85.0,
        "expected >= 85.0% ConcordantOrdered on real AF-P0CG48-F1[1..=76] Ubiquitin, got {frac_ordered:.1}%"
    );
    // Notice candor (`A1`/`P1-1`): in polyubiquitin `AF-P0CG48-F1-model_v6`, the core pLDDT is high
    // (`84.6..98.7`) and repeat-1 Gly75-Gly76 links into Met77 (`pLDDT ~73..79`), so whole-chain
    // and ordered-core Spearman rho_s are modest positive rank correlations (`~0.35..0.55`) rather
    // than the inflated `0.78` produced by the deleted hand-typed array!
    assert!(
        rho_whole > 0.20,
        "expected positive Spearman rho_s on real AF-P0CG48-F1[1..=76], got {rho_whole:.4}"
    );

    // Also verify Lipari-Szabo `--s2 <modelfree.tsv>` override path & HTML/SVG/JSON/TSV rendering
    let mut mf_tsv = String::from("seq_id\tres_name\tmodel\ts2\tsigma_s2\n");
    for (idx, r) in mono_chain.residues.iter().enumerate() {
        let s2_ls = if idx >= 72 {
            0.28
        } else if (7..=10).contains(&idx) {
            0.70
        } else {
            0.86
        };
        mf_tsv.push_str(&format!(
            "{}\t{}\tM1\t{s2_ls:.4}\t0.020\n",
            r.seq_id, r.res_name
        ));
    }
    let s2_rows = parse_modelfree_s2_tsv_str(&mf_tsv).unwrap();
    let rep_ls =
        evaluate_af_nmr_concordance(&mono_model, &shifts, None, Some(&s2_rows), &cfg).unwrap();
    assert_eq!(
        rep_ls.rigidity_source,
        ExperimentalRigiditySource::LipariSzaboModelFree
    );

    let svg = render_af_concordance_svg(&rep);
    assert!(svg.contains("<svg id=\"svg-af-concordance\""));
    let html = generate_quality_report_html(
        &ReportInput::new("Ubiquitin", shifts, BTreeMap::new())
            .with_af_concordance_report(rep.clone()),
    );
    assert!(html.contains("id=\"section-af-concordance\""));
    let tsv = format_af_concordance_tsv(&rep);
    assert!(tsv.contains("ConcordantOrdered"));
    assert!(tsv.contains("ordered_core_rho_s="));
    assert!(tsv.contains("shift-derived secondary-structure-coupled RCI-S^2"));
    let json = format_af_concordance_json(&rep).unwrap();
    assert!(json.contains("\"spearman_rho\""));
    assert!(json.contains("\"ordered_core_spearman_rho\""));
}

#[test]
fn test_alpha_synuclein_bmr6968_concordant_disordered_ge_85_pct() {
    let shifts = read_star(repo_root().join("data/benchmark/bmr6968.str"), None)
        .expect("read bmr6968 alpha-synuclein shifts");
    let l = shifts.sequence.len();
    assert_eq!(l, 140);

    // Synthetic coil unit check (`21.W1:P1-2`): build an unstructured coil trace (`pLDDT` in `38..54`)
    let mut phi = Vec::with_capacity(l);
    let mut psi = Vec::with_capacity(l);
    let mut plddts = Vec::with_capacity(l);
    for i in 0..l {
        phi.push(-75.0 - ((i % 7) as f64) * 5.0);
        psi.push(138.0 + ((i % 5) as f64) * 4.0);
        plddts.push(38.0 + ((i % 9) as f64) * 1.8);
    }
    let chain = build_backbone_from_torsions(
        &shifts.sequence,
        &shifts.seq_ids,
        &phi,
        &psi,
        &vec![180.0; l],
    );
    let cif = build_af_mmcif_from_chain("SYNTHETIC_COIL_BMR6968", &chain, &plddts);
    let model = parse_af_structure_model_str(&cif, &AfIngestOptions::default()).unwrap();

    let cfg = AfConcordanceConfig {
        exclude_bmrb_ids: vec!["6968".to_string()],
        ..AfConcordanceConfig::default()
    };
    let rep = evaluate_af_nmr_concordance(&model, &shifts, None, None, &cfg).unwrap();
    let assigned = rep.residues.len() - rep.n_unassigned_nmr;
    let pct_disordered = 100.0 * (rep.n_concordant_disordered as f64) / (assigned as f64);
    println!(
        "bmr6968 alpha-synuclein: disordered={}/{} ({pct_disordered:.1}%)",
        rep.n_concordant_disordered, assigned
    );
    assert!(
        pct_disordered >= 85.0,
        "expected >= 85.0% ConcordantDisordered on bmr6968 alpha-synuclein, got {pct_disordered:.1}%"
    );
}

/// Synthetic unit / self-consistency check (`21.W1:P1-2`): plant a high-`pLDDT` 3D helix onto
/// solution-disordered NMR chemical shifts and return `(evaluated, model_ordered_sol_disordered, indeterminate, concordant_ordered)`.
fn evaluate_planted_morf_unit_check(
    shifts: &Reference,
    morf_start_idx: usize,
    morf_end_idx: usize,
    bmrb_id: &str,
) -> (usize, usize, usize, usize) {
    let l = shifts.sequence.len();
    let mut phi = vec![-82.0_f64; l];
    let mut psi = vec![142.0_f64; l];
    let mut plddts = vec![41.0_f64; l];

    let pad_lo = morf_start_idx.saturating_sub(2);
    let pad_hi = (morf_end_idx + 2).min(l - 1);
    for k in pad_lo..=pad_hi {
        phi[k] = -62.0;
        psi[k] = -42.0;
        plddts[k] = 89.0;
    }

    let chain = build_backbone_from_torsions(
        &shifts.sequence,
        &shifts.seq_ids,
        &phi,
        &psi,
        &vec![180.0; l],
    );
    let cif = build_af_mmcif_from_chain(&format!("PLANTED_MORF_{bmrb_id}"), &chain, &plddts);
    let model = parse_af_structure_model_str(&cif, &AfIngestOptions::default()).unwrap();

    let cfg = AfConcordanceConfig {
        exclude_bmrb_ids: vec![bmrb_id.to_string()],
        ..AfConcordanceConfig::default()
    };
    let rep = evaluate_af_nmr_concordance(&model, shifts, None, None, &cfg).unwrap();

    let mut evaluated_morf = 0usize;
    let mut detected_morf = 0usize;
    let mut indeterminate_morf = 0usize;
    let mut ordered_morf = 0usize;
    for k in morf_start_idx..=morf_end_idx {
        let rec = &rep.residues[k];
        if rec.concordance_class == AfNmrConcordanceClass::UnassignedNmr {
            continue;
        }
        assert_eq!(
            rec.state4_af,
            SecondaryStructure4::Helix,
            "planted MoRF residue {} ({}{}) must be 3D Helix",
            k,
            rec.res_name,
            rec.seq_id
        );
        evaluated_morf += 1;
        match rec.concordance_class {
            AfNmrConcordanceClass::ModelOrderedSolutionDisordered => detected_morf += 1,
            AfNmrConcordanceClass::Indeterminate => indeterminate_morf += 1,
            AfNmrConcordanceClass::ConcordantOrdered => ordered_morf += 1,
            _ => {}
        }
    }
    (
        evaluated_morf,
        detected_morf,
        indeterminate_morf,
        ordered_morf,
    )
}

#[test]
fn test_planted_idp_morf_synthetic_unit_check_on_bmr15176_and_bmr51984() {
    let darpp32 = read_star(repo_root().join("data/benchmark/bmr15176.str"), None)
        .expect("read bmr15176 DARPP-32");
    let p53_tad = read_star(
        repo_root().join("data/fixtures/bmr51984_idp_p53_tad_prr.str"),
        None,
    )
    .expect("read bmr51984 p53 TAD-PRR");

    // 1. DARPP-32 (`bmr15176`, random-coil C-terminal segment indices 82..=93): 12/12 `ModelOrderedSolutionDisordered`
    let (ev1, det1, ind1, ord1) = evaluate_planted_morf_unit_check(&darpp32, 82, 93, "15176");
    assert_eq!((ev1, det1, ind1, ord1), (12, 12, 0, 0));

    // 2. p53 TAD1 (`bmr51984`, Seq_ID 18..=26, indices 17..=25): after fixing `21.W1:P2-7`
    //    (stopping Pro27 gap-smearing from artificially depressing Trp25/Lys26 RCI-S2 below 0.60),
    //    `7/9` (`Thr18..Leu24`) are `ModelOrderedSolutionDisordered`, the partially pre-structured
    //    nascent-helical C-cap `Trp25..Lys26` (`RCI-S2 ~ 0.609` in `[0.60, 0.65)`) is honestly
    //    classified as `Indeterminate` (`2/9`), and `0/9` are `ConcordantOrdered`!
    let (ev2, det2, ind2, ord2) = evaluate_planted_morf_unit_check(&p53_tad, 17, 25, "51984");
    assert_eq!(ev2, 9);
    assert_eq!(ord2, 0);
    assert_eq!(det2 + ind2, 9);
    assert!(det2 >= 7);
}

#[test]
fn test_planted_alpha_beta_switch_dynamic_hinge_or_alt_state_ge_90_pct_recall() {
    let shifts = read_star(
        repo_root().join("data/fixtures/bmr17769_ubiquitin_allatom.str"),
        None,
    )
    .expect("read bmr17769 ubiquitin shifts");
    let raw_1d3z =
        fs::read_to_string(repo_root().join("data/pdb_refs/1d3z.pdb")).expect("read 1d3z.pdb");
    let base_model = parse_af_structure_model_str(
        &strip_pdb_ss_headers(&raw_1d3z),
        &AfIngestOptions::default(),
    )
    .unwrap();
    let base_ss = classify_kabsch_sander(&base_model.backbone_coords_for_ss()).unwrap();

    // Construct a chimeric 3D conformer of Ubiquitin where:
    // - Segment A (`indices 1..=5`, Ubiquitin's native beta1 strand `Gln2..Lys6` in NMR) AND
    //   Segment B (`indices 65..=70`, Ubiquitin's native beta5 strand `Thr66..Leu71` in NMR)
    //   are folded into high-confidence (`pLDDT = 92`) right-handed alpha-helices (`phi = -62, psi = -42`)!
    //   In solution NMR (`bmr17769`), both segments are rigid beta-strands (`RCI-S2 >= 0.78`, `SSP < -0.4`),
    //   representing a classic fold-switching alpha <-> beta conformational switch (`12` switched residues).
    let mut phi: Vec<f64> = base_ss
        .residues
        .iter()
        .map(|r| r.phi_deg.unwrap_or(-65.0))
        .collect();
    let mut psi: Vec<f64> = base_ss
        .residues
        .iter()
        .map(|r| r.psi_deg.unwrap_or(135.0))
        .collect();
    let mut plddts = vec![92.0_f64; 76];
    // Also test branch (b): low-pLDDT (< 70) prediction on a rigidly ordered NMR core residue (Ile30, idx 29)
    plddts[29] = 44.0;

    for k in 0..=8 {
        phi[k] = -62.0;
        psi[k] = -42.0;
    }
    for k in 63..=73 {
        phi[k] = -62.0;
        psi[k] = -42.0;
    }

    let switched_chain = build_backbone_from_torsions(
        &shifts.sequence,
        &shifts.seq_ids,
        &phi,
        &psi,
        &vec![180.0; 76],
    );
    let cif = build_af_mmcif_from_chain("UBQ_ALPHA_BETA_SWITCH", &switched_chain, &plddts);
    let model = parse_af_structure_model_str(&cif, &AfIngestOptions::default()).unwrap();

    let cfg = AfConcordanceConfig {
        exclude_pdb_ids: vec!["1D3Z".to_string(), "1UBQ".to_string()],
        exclude_bmrb_ids: vec!["17769".to_string()],
        ..AfConcordanceConfig::default()
    };
    let rep = evaluate_af_nmr_concordance(&model, &shifts, None, None, &cfg).unwrap();

    let switched_indices: Vec<usize> = (2..=5).chain(65..=70).collect(); // 10 interior switched residues
    let recalled = switched_indices
        .iter()
        .filter(|&&k| {
            rep.residues[k].concordance_class == AfNmrConcordanceClass::DynamicHingeOrAltState
        })
        .count();
    let recall_pct = 100.0 * (recalled as f64) / (switched_indices.len() as f64);
    println!(
        "Planted alpha <-> beta switch recall: {recalled}/{} ({recall_pct:.1}%)",
        switched_indices.len()
    );
    assert!(
        recall_pct >= 90.0,
        "expected >= 90.0% DynamicHingeOrAltState recall on planted alpha <-> beta switch, got {recall_pct:.1}% ({recalled}/{})",
        switched_indices.len()
    );
    assert_eq!(
        rep.residues[29].concordance_class,
        AfNmrConcordanceClass::DynamicHingeOrAltState,
        "low-pLDDT (44) vs rigid NMR core residue Ile30 must flag DynamicHingeOrAltState"
    );
}

#[test]
fn test_p0_2_and_p2_6_fail_closed_provenance_headerless_1ubq_remark_zero_b_and_qmeandisco() {
    let opts = AfIngestOptions::default();

    // 1. `21.W1:P0-2`: Strip EXPDTA, HEADER, TITLE, COMPND, REMARK, CRYST1 from `1ubq.pdb`.
    //    On `6328a1e`, raw `ATOM` lines with `B <= 100` were silently classified as `ComputationalModel`
    //    with crystallographic B-factors interpreted as pLDDT! Now must be `Unknown` + `None` + `Err`.
    let raw_1ubq = fs::read_to_string(repo_root().join("data/fixtures/ubiquitin_1ubq_xray.pdb"))
        .expect("read ubiquitin_1ubq_xray.pdb");
    let headerless_1ubq: String = raw_1ubq
        .lines()
        .filter(|l| l.starts_with("ATOM  ") || l.starts_with("TER") || l.starts_with("END"))
        .collect::<Vec<_>>()
        .join("\n");
    let m_stripped = parse_af_structure_model_str(&headerless_1ubq, &opts).unwrap();
    assert_eq!(m_stripped.provenance, CoordinateProvenance::Unknown);
    assert_eq!(m_stripped.plddt_source, PlddtSource::Unavailable);
    assert!(m_stripped.plddt_per_residue.is_none());
    assert!(require_predicted_model(&m_stripped, false).is_err());

    // 2. `21.W1:P0-2`: Free-text `REMARK` / `HEADER` mentioning `ALPHAFOLD` in a PDB must NOT
    //    confer `ComputationalModel` provenance without `--treat-as-model`!
    let remark_af_pdb = format!("REMARK   3 MODEL GENERATED BY ALPHAFOLD\n{headerless_1ubq}");
    let m_remark = parse_af_structure_model_str(&remark_af_pdb, &opts).unwrap();
    assert_eq!(m_remark.provenance, CoordinateProvenance::Unknown);
    assert!(m_remark.plddt_per_residue.is_none());
    assert!(require_predicted_model(&m_remark, false).is_err());

    // 3. `21.W1:P0-2`: All-zero (`0.00`) and constant (`0.50`) B-factors in mmCIF must be rejected
    //    (`ConstantOrZeroBfactorRejected`), NEVER rescaled to `50.0` pLDDT, and fail closed even
    //    when `treat_as_model == true`!
    for const_b in ["0.00", "0.50", "85.00"] {
        let const_b_cif = format!(
            "data_AF-P00000-F1\n_entry.id AF-P00000-F1\n_ma_protocol_step.step_id 1\n\
             loop_\n_atom_site.label_atom_id\n_atom_site.label_comp_id\n_atom_site.auth_seq_id\n\
             _atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n_atom_site.B_iso_or_equiv\n\
             N ALA 1 0.0 0.0 0.0 {const_b}\nCA ALA 1 1.46 0.0 0.0 {const_b}\nC ALA 1 2.0 1.4 0.0 {const_b}\n\
             N GLY 2 3.2 1.2 0.0 {const_b}\nCA GLY 2 4.5 1.8 0.0 {const_b}\nC GLY 2 5.2 0.6 0.0 {const_b}\n"
        );
        let m_cb = parse_af_structure_model_str(
            &const_b_cif,
            &AfIngestOptions {
                target_chain_id: None,
                force_bfactor_plddt: true,
            },
        )
        .unwrap();
        assert!(
            m_cb.diagnostics
                .contains(&AfIngestDiagnostic::ConstantOrZeroBfactorRejected),
            "constant B_iso={const_b} must emit ConstantOrZeroBfactorRejected"
        );
        assert!(
            m_cb.plddt_per_residue.is_none(),
            "constant B_iso={const_b} must not populate plddt_per_residue"
        );
        assert!(require_predicted_model(&m_cb, true).is_err());
    }

    // 4. `21.W1:P2-6`: Non-pLDDT `_ma_qa_metric` (`QMEANDisCo`) must NOT be treated as pLDDT,
    //    and when both `pLDDT` (id=1) and `QMEANDisCo` (id=2) exist, `QMEANDisCo` must NOT overwrite `pLDDT`!
    let qmean_only_cif = "\
data_SWISSMODEL_QMEAN
_ma_protocol_step.step_id 1
loop_
_ma_qa_metric.id
_ma_qa_metric.name
_ma_qa_metric.mode
1 QMEANDisCo local
#
loop_
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.auth_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
N ALA A 1 1 0.0 0.0 0.0
CA ALA A 1 1 1.46 0.0 0.0
C ALA A 1 1 2.0 1.4 0.0
N GLY A 2 2 3.2 1.2 0.0
CA GLY A 2 2 4.5 1.8 0.0
C GLY A 2 2 5.2 0.6 0.0
#
loop_
_ma_qa_metric_local.label_asym_id
_ma_qa_metric_local.label_seq_id
_ma_qa_metric_local.metric_id
_ma_qa_metric_local.metric_value
A 1 1 0.82
A 2 1 0.74
#
";
    let m_qm = parse_af_structure_model_str(qmean_only_cif, &opts).unwrap();
    assert!(m_qm.diagnostics.iter().any(|d| matches!(
        d,
        AfIngestDiagnostic::NonPlddtQaMetricIgnored { metric_name }
            if metric_name.to_ascii_lowercase().contains("qmeandisco")
    )));
    assert!(m_qm.plddt_per_residue.is_none());
    assert!(require_predicted_model(&m_qm, false).is_err());

    // `21F:B1` regression: real-shaped ModelArchive QMEANDisCo mmCIF carrying `_ma_protocol_step`,
    // `_ma_data`, `_ma_model_list`, AND non-constant `_atom_site.B_iso_or_equiv` in `[0, 1]`
    // (must fail closed and NOT rescale 0.924 -> pLDDT 92.4!). Fails at `f8408b9`.
    let qmean_real_shaped_cif = "\
data_MA-BAK-CEPC-0001
_entry.id MA-BAK-CEPC-0001
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
loop_
_ma_qa_metric.id
_ma_qa_metric.name
_ma_qa_metric.type
_ma_qa_metric.mode
1 QMEANDisCo other local
#
loop_
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.auth_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.B_iso_or_equiv
N ALA A 1 1 0.0 0.0 0.0 0.924
CA ALA A 1 1 1.46 0.0 0.0 0.924
C ALA A 1 1 2.0 1.4 0.0 0.924
N GLY A 2 2 3.2 1.2 0.0 0.781
CA GLY A 2 2 4.5 1.8 0.0 0.781
C GLY A 2 2 5.2 0.6 0.0 0.781
#
loop_
_ma_qa_metric_local.label_asym_id
_ma_qa_metric_local.label_seq_id
_ma_qa_metric_local.metric_id
_ma_qa_metric_local.metric_value
A 1 1 0.924
A 2 1 0.781
#
";
    let m_qm_real = parse_af_structure_model_str(qmean_real_shaped_cif, &opts).unwrap();
    assert_eq!(
        m_qm_real.provenance,
        CoordinateProvenance::Unknown,
        "ModelArchive QMEANDisCo mmCIF must NOT be promoted to ComputationalModel"
    );
    assert_eq!(m_qm_real.plddt_source, PlddtSource::Unavailable);
    assert!(
        m_qm_real.plddt_per_residue.is_none(),
        "QMEANDisCo B_iso_or_equiv must NOT be treated as pLDDT or rescaled x100"
    );
    assert!(
        !m_qm_real
            .diagnostics
            .contains(&AfIngestDiagnostic::RescaledUnitPlddtTo100),
        "must not emit RescaledUnitPlddtTo100 for QMEANDisCo"
    );
    assert!(
        require_predicted_model(&m_qm_real, false).is_err(),
        "require_predicted_model must reject real-shaped ModelArchive QMEANDisCo mmCIF"
    );

    let mixed_qa_cif = "\
data_MIXED_QA
_ma_protocol_step.step_id 1
loop_
_ma_qa_metric.id
_ma_qa_metric.name
_ma_qa_metric.mode
1 pLDDT local
2 QMEANDisCo local
#
loop_
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.auth_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
N ALA A 1 1 0.0 0.0 0.0
CA ALA A 1 1 1.46 0.0 0.0
C ALA A 1 1 2.0 1.4 0.0
N GLY A 2 2 3.2 1.2 0.0
CA GLY A 2 2 4.5 1.8 0.0
C GLY A 2 2 5.2 0.6 0.0
#
loop_
_ma_qa_metric_local.label_asym_id
_ma_qa_metric_local.label_seq_id
_ma_qa_metric_local.metric_id
_ma_qa_metric_local.metric_value
A 1 1 94.5
A 1 2 0.61
A 2 1 88.0
A 2 2 0.52
#
";
    let m_mix = parse_af_structure_model_str(mixed_qa_cif, &opts).unwrap();
    require_predicted_model(&m_mix, false).unwrap();
    assert_eq!(m_mix.plddt_per_residue, Some(vec![94.5, 88.0]));
}

#[test]
fn test_p1_1_constant_plddt_spearman_none_indeterminate_taxonomy_and_block_bootstrap_ci() {
    // 1. Constant pLDDT (`85.0` across all 76 residues in `_ma_qa_metric_local`) must return
    //    `spearman_rho == None` (`"undefined"` in TSV, `null` in JSON) — NOT `0.0` / `C_corr = 50.0`!
    assert_eq!(
        tie_adjusted_spearman_rho(
            &[85.0; 20],
            &[
                0.82, 0.85, 0.79, 0.90, 0.45, 0.30, 0.88, 0.86, 0.84, 0.81, 0.77, 0.83, 0.89, 0.91,
                0.76, 0.74, 0.68, 0.55, 0.42, 0.35
            ]
        ),
        None
    );
    assert_eq!(
        block_bootstrap_spearman_rho_ci95(&[85.0; 20], &[0.8; 20], 5, 200),
        None
    );

    let shifts = read_star(
        repo_root().join("data/fixtures/bmr17769_ubiquitin_allatom.str"),
        None,
    )
    .unwrap();
    let raw_1d3z = fs::read_to_string(repo_root().join("data/pdb_refs/1d3z.pdb")).unwrap();
    let chain = parse_af_structure_model_str(&raw_1d3z, &AfIngestOptions::default())
        .unwrap()
        .to_pdb_chain();
    let const_cif = build_af_mmcif_from_chain("CONST_PLDDT", &chain, &[85.0; 76]);
    let const_model =
        parse_af_structure_model_str(&const_cif, &AfIngestOptions::default()).unwrap();
    let rep_const = evaluate_af_nmr_concordance(
        &const_model,
        &shifts,
        None,
        None,
        &AfConcordanceConfig::default(),
    )
    .unwrap();
    assert_eq!(rep_const.spearman_rho, None);
    assert_eq!(rep_const.spearman_rho_ci95, None);
    assert_eq!(rep_const.summary.spearman_rho(), None);
    assert!(matches!(
        rep_const.summary,
        ConcordanceSummary::InsufficientCoverage { .. }
    ));
    let tsv_const = format_af_concordance_tsv(&rep_const);
    assert!(tsv_const.contains("spearman_rho_s=undefined"));
    let json_const: serde_json::Value =
        serde_json::from_str(&format_af_concordance_json(&rep_const).unwrap()).unwrap();
    assert!(json_const["spearman_rho"].is_null());

    // 2. Verify all 3 former threshold-gap regimes classify as `AfNmrConcordanceClass::Indeterminate`:
    //    - Regime A: `pLDDT = 88.0 >= 70` with `0.60 <= S^2_exp = 0.62 < 0.65`
    //    - Regime B: `pLDDT = 55.0 < 70` with `0.65 <= S^2_exp = 0.68 < 0.72`
    //    - Regime C: `pLDDT = 88.0 >= 70` with `S^2_exp = 0.50 < 0.60`, where NMR shifts are strongly
    //      helical (`p_coil + p_ppii < 0.65`, e.g. Ubiquitin helix residue Ile30 idx 29)
    let mut mixed_plddts = vec![92.0_f64; 76];
    mixed_plddts[5] = 88.0; // Regime A (seq_id 6)
    mixed_plddts[10] = 55.0; // Regime B (seq_id 11)
    mixed_plddts[29] = 88.0; // Regime C (seq_id 30, strongly helical in BMRB 17769)
    let mix_cif = build_af_mmcif_from_chain("GAP_TEST", &chain, &mixed_plddts);
    let mix_model = parse_af_structure_model_str(&mix_cif, &AfIngestOptions::default()).unwrap();
    let mut mf_tsv = String::from("seq_id\tres_name\ts2\n");
    for (idx, r) in chain.residues.iter().enumerate() {
        let s2 = match idx {
            5 => 0.62,
            10 => 0.68,
            29 => 0.50,
            _ => 0.86,
        };
        mf_tsv.push_str(&format!("{}\t{}\t{s2:.3}\n", r.seq_id, r.res_name));
    }
    let s2_rows = parse_modelfree_s2_tsv_str(&mf_tsv).unwrap();
    let rep_gap = evaluate_af_nmr_concordance(
        &mix_model,
        &shifts,
        None,
        Some(&s2_rows),
        &AfConcordanceConfig::default(),
    )
    .unwrap();
    assert_eq!(
        rep_gap.residues[5].concordance_class,
        AfNmrConcordanceClass::Indeterminate
    );
    assert_eq!(
        rep_gap.residues[10].concordance_class,
        AfNmrConcordanceClass::Indeterminate
    );
    assert_eq!(
        rep_gap.residues[29].concordance_class,
        AfNmrConcordanceClass::Indeterminate
    );
    assert!(rep_gap.n_indeterminate >= 3);

    // 3. Serde alias round-trip for renamed `ModelOrderedSolutionDisordered`
    let parsed_legacy: AfNmrConcordanceClass =
        serde_json::from_str("\"ConditionalFoldingOrCrystalArtifact\"").unwrap();
    assert_eq!(
        parsed_legacy,
        AfNmrConcordanceClass::ModelOrderedSolutionDisordered
    );
    assert_eq!(
        serde_json::to_string(&parsed_legacy).unwrap(),
        "\"ModelOrderedSolutionDisordered\""
    );
}

#[test]
fn test_p2_7_p2_9_p2_10_gap_aware_rci_smoothing_berjanskii_17_7_and_pae_crosschecks() {
    // 1. `21.W1:P2-7` & `P2-10`: Unassigned gap residues (`[None; 6]`) must NOT smear default
    //    flexible `RCI = 0.22` into assigned neighbors, and `S^2_RCI = 1 - 0.4 * ln(1 + 17.7 * RCI)`.
    let helix_row = [
        Some(-0.35),
        Some(-1.80),
        Some(3.10),
        Some(-0.40),
        Some(2.40),
        Some(-0.38),
    ];
    let intact = vec![helix_row; 15];
    let mut gapped = intact.clone();
    gapped[6] = [None; 6];
    gapped[7] = [None; 6];
    gapped[8] = [None; 6];

    let (_rci_intact, s2_intact) = compute_berjanskii_wishart_rci_s2(&intact);
    let (rci_gapped, s2_gapped) = compute_berjanskii_wishart_rci_s2(&gapped);
    for nb in [4usize, 5, 9, 10] {
        assert!(
            (s2_gapped[nb] - s2_intact[nb]).abs() < 1.0e-12,
            "neighbor residue {nb} next to unassigned gap had RCI-S2 dragged from {:.4} to {:.4}",
            s2_intact[nb],
            s2_gapped[nb]
        );
        let expected_17_7 = (1.0 - 0.4 * (1.0 + 17.7 * rci_gapped[nb]).ln()).clamp(0.0, 1.0);
        assert!((s2_gapped[nb] - expected_17_7).abs() < 1.0e-12);
    }

    // 2. `21.W1:P2-9`: Multi-chain AF3 `confidences.json` without `target_chain_id` must fail closed (`Err`)
    let af3_multi = r#"{
        "pae": [
            [0.2, 1.4, 12.0, 13.0],
            [1.6, 0.2, 11.5, 12.5],
            [12.0, 11.5, 0.2, 1.2],
            [13.0, 12.5, 1.4, 0.2]
        ],
        "token_chain_ids": ["A", "A", "B", "B"],
        "token_res_ids": [1, 2, 1, 2]
    }"#;
    let err_no_chain = parse_pae_json_str_for_chain(af3_multi, None, None).unwrap_err();
    assert!(
        err_no_chain.contains("--chain"),
        "expected explicit --chain requirement error, got: {err_no_chain}"
    );
    let m_b = parse_pae_json_str_for_chain(af3_multi, Some("B"), Some(2)).unwrap();
    assert_eq!(m_b.n_residues, 2);
    assert_eq!(m_b.chain_id.as_deref(), Some("B"));

    // 3. `21.W1:P2-9`: ColabFold PAE JSON `plddt` must be cross-checked against `AfStructureModel`
    let colab_cif = "\
data_CF_TEST
_ma_protocol_step.step_id 1
loop_
_ma_qa_metric.id
_ma_qa_metric.name
_ma_qa_metric.mode
1 pLDDT local
#
loop_
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.auth_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
N ALA A 1 1 0.0 0.0 0.0
CA ALA A 1 1 1.46 0.0 0.0
C ALA A 1 1 2.0 1.4 0.0
N GLY A 2 2 3.2 1.2 0.0
CA GLY A 2 2 4.5 1.8 0.0
C GLY A 2 2 5.2 0.6 0.0
#
loop_
_ma_qa_metric_local.label_asym_id
_ma_qa_metric_local.label_seq_id
_ma_qa_metric_local.metric_id
_ma_qa_metric_local.metric_value
A 1 1 94.0
A 2 1 88.5
#
";
    let cf_struct = parse_af_structure_model_str(colab_cif, &AfIngestOptions::default()).unwrap();
    let pae_ok =
        parse_pae_json_str(r#"{"pae": [[0.2, 1.5], [1.7, 0.2]], "plddt": [94.0, 88.5]}"#).unwrap();
    assert!(pae_ok.validate_against_structure(&cf_struct).is_ok());

    let pae_mismatch =
        parse_pae_json_str(r#"{"pae": [[0.2, 1.5], [1.7, 0.2]], "plddt": [42.0, 88.5]}"#).unwrap();
    let err_cf = pae_mismatch
        .validate_against_structure(&cf_struct)
        .unwrap_err();
    assert!(
        err_cf.contains("plddt mismatch"),
        "expected ColabFold plddt mismatch error, got: {err_cf}"
    );
}

#[path = "support/af_provenance_probes.rs"]
mod af_provenance_probes;

/// `21F:B1` (FIX3 slice A): the shared probe matrix must be decided identically by all three
/// nmr-core gates — `af_concordance::require_predicted_model`, `struct_shifts::
/// require_predicted_structure_model`, and the `af_assign_priors` text-evidence + `from_strs`
/// gate. SWISS-MODEL QMEANDisCo files (with/without `_software` AlphaFold or `data_AF-` header),
/// a declared pLDDT metric lacking local rows (f3/f5), and `global` in the type column (e2) must
/// fail closed; real AFDB v6 and verified local-pLDDT rows must still be accepted. Fails at
/// `b5e87a3` on f3, f5 and e2.
#[test]
fn test_21f_b1_provenance_probe_matrix_all_core_gates_agree() {
    af_provenance_probes::assert_matrix(af_provenance_probes::core_gate_verdicts);
}
