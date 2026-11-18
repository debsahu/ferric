//! Integration and quantitative gate tests for `T21.1` (`docs/specs/21-alphafold-integrative-nmr.md` §1, §8):
//! - Hostile mmCIF / PDB / `PAE` JSON `Err` guards (`<= 64 MiB`, `<= 4,096` pre-allocation cap,
//!   `NaN`/`Inf`, truncated `_ma_qa_metric_local` / `_atom_site` loops, $N \ne L$, crystallographic
//!   `B_iso_or_equiv` / `tempFactor` guard, `[0, 1] -> [0, 100]` auto-rescale),
//! - Clean-room Kabsch–Sander (1983) `8 -> 4`-state classifier (`>= 96.0%` agreement on `1D3Z` and
//!   `1UBQ` Ubiquitin without reading `HELIX` or `SHEET` records, plus `PPII` Adzhubei 2013),
//! - Fowler–Sljoka–Williamson (2020, 2022) rigidity concordance & 4-class discordance taxonomy:
//!   - `>= 85%` `ConcordantOrdered` and Spearman $\rho_s \ge 0.70$ on folded Ubiquitin,
//!   - `>= 85%` `ConcordantDisordered` on $\alpha$-Synuclein (`bmr6968`),
//!   - `100%` `ConditionalFoldingOrCrystalArtifact` detection on planted high-`pLDDT` IDP $\alpha$-MoRFs
//!     (`bmr15176` DARPP-32 and `bmr51984` p53 TAD-PRR),
//!   - `>= 90%` `DynamicHingeOrAltState` recall on planted $\alpha \leftrightarrow \beta$ conformational switches.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use nmr_core::af_concordance::{
    evaluate_af_nmr_concordance, format_af_concordance_json, format_af_concordance_tsv,
    parse_af_structure_model, parse_af_structure_model_str, parse_modelfree_s2_tsv_str,
    parse_pae_json_str, parse_pae_json_str_for_chain, AfConcordanceConfig, AfIngestDiagnostic,
    AfIngestOptions, AfNmrConcordanceClass, ConcordanceSummary, CoordinateProvenance,
    ExperimentalRigiditySource, PaeJsonDialect, PlddtSource, MAX_AF_STRUCTURE_BYTES,
};
use nmr_core::kabsch_sander_ss::{
    classify_kabsch_sander, BackboneResidueCoords, KabschSanderState8, SecondaryStructure4,
};
use nmr_core::nmrstar::{read_star, Reference};
use nmr_core::noesy::PdbChain;
use nmr_core::report::{generate_quality_report_html, render_af_concordance_svg, ReportInput};
use nmr_core::structure::build_backbone_from_torsions;

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

    // 6. Unit-interval `[0, 1] -> [0, 100]` auto-rescaling with audit diagnostic
    let unit_b_cif = "\
data_unit_af
_ma_protocol_step.step_id 1
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
fn test_ubiquitin_concordant_ordered_ge_85_pct_and_spearman_ge_070() {
    let raw_1d3z =
        fs::read_to_string(repo_root().join("data/pdb_refs/1d3z.pdb")).expect("read 1d3z.pdb");
    let base_model = parse_af_structure_model_str(
        &strip_pdb_ss_headers(&raw_1d3z),
        &AfIngestOptions::default(),
    )
    .unwrap();
    let chain = base_model.to_pdb_chain();

    // Published AlphaFold2 Human Ubiquitin (`AF-P62988-F1-model_v4`, Varadi et al. 2024, CC-BY-4.0)
    // per-residue pLDDT profile (76 residues, Met1..Gly76): high confidence (94..98.8) across
    // beta1..beta5 & alpha1/3_10 helices, moderate dips in beta1-beta2 hairpin (Leu8-Lys11),
    // Asp39-Gln40, Lys48-Gly53, and Asn60-Ser65 loops, and steep C-terminal RLRGG tail decay (72..76).
    let plddts: Vec<f64> = vec![
        89.4, 95.1, 97.8, 98.5, 98.4, 96.8, 91.2, 81.4, 76.2, 75.8, // 1..10
        78.9, 91.5, 96.4, 97.9, 98.3, 96.7, 94.8, 92.6, 91.8, 90.9, // 11..20
        92.1, 94.6, 97.4, 98.1, 98.5, 98.7, 98.6, 98.4, 98.5, 98.3, // 21..30
        97.6, 95.8, 93.4, 91.2, 88.6, 89.4, 90.2, 90.8, 88.5, 86.4, // 31..40
        91.8, 95.6, 97.9, 98.4, 97.2, 88.9, 85.4, 83.2, 80.6, 79.8, // 41..50
        80.4, 82.6, 86.5, 91.2, 94.8, 96.2, 95.9, 94.1, 91.6, 87.8, // 51..60
        86.9, 87.5, 88.4, 89.6, 91.4, 95.2, 97.6, 98.3, 97.8, 95.4, // 61..70
        84.6, 68.4, 55.2, 44.6, 38.4, 35.1, // 71..76
    ];

    let af_cif = build_af_mmcif_from_chain("AF-P62988-F1", &chain, &plddts);
    let af_model = parse_af_structure_model_str(&af_cif, &AfIngestOptions::default()).unwrap();
    assert_eq!(
        af_model.provenance,
        CoordinateProvenance::ComputationalModel
    );
    assert_eq!(af_model.plddt_source, PlddtSource::MaQaMetricLocal);
    assert_eq!(af_model.detected_uniprot_id.as_deref(), Some("P62988"));

    let shifts = read_star(
        repo_root().join("data/fixtures/bmr17769_ubiquitin_allatom.str"),
        None,
    )
    .expect("read bmr17769 ubiquitin shifts");

    let cfg = AfConcordanceConfig {
        exclude_pdb_ids: vec!["1D3Z".to_string(), "1UBQ".to_string()],
        exclude_bmrb_ids: vec!["17769".to_string(), "6457".to_string()],
        ..AfConcordanceConfig::default()
    };
    let rep = evaluate_af_nmr_concordance(&af_model, &shifts, None, None, &cfg).unwrap();
    assert_eq!(
        rep.rigidity_source,
        ExperimentalRigiditySource::BerjanskiiWishartRciS2
    );

    // HomologFirewall invariants: <= 25% retained seq identity + non-empty SHA-256 audit digest
    assert!(
        rep.max_retained_seq_identity <= 0.25,
        "HomologFirewall violated: max_retained_seq_identity = {}",
        rep.max_retained_seq_identity
    );
    assert_eq!(rep.exclusion_list_sha256.len(), 64);
    assert!(!rep.excluded_template_ids.is_empty());

    let ConcordanceSummary::Evaluated {
        n_eval,
        spearman_rho_s,
        rmsd_rigidity,
        ..
    } = rep.summary
    else {
        panic!(
            "expected Evaluated concordance summary on Ubiquitin, got {:?}",
            rep.summary
        );
    };

    let assigned_total = rep.residues.len() - rep.n_unassigned_nmr;
    let frac_ordered = 100.0 * (rep.n_concordant_ordered as f64) / (assigned_total as f64);
    println!(
        "Ubiquitin RCI-S2 concordance: n_eval={n_eval}, rho_s={spearman_rho_s:.4}, rmsd_rig={rmsd_rigidity:.4}, ordered={}/{} ({frac_ordered:.1}%), disordered={}",
        rep.n_concordant_ordered, assigned_total, rep.n_concordant_disordered
    );

    assert!(
        frac_ordered >= 85.0,
        "expected >= 85.0% ConcordantOrdered on folded Ubiquitin, got {frac_ordered:.1}%"
    );
    assert!(
        spearman_rho_s >= 0.70,
        "expected Spearman rho_s >= 0.70 on folded Ubiquitin, got {spearman_rho_s:.4}"
    );
    // Flexible C-terminal tail Gly75 / Gly76 must be ConcordantDisordered
    assert_eq!(
        rep.residues[74].concordance_class,
        AfNmrConcordanceClass::ConcordantDisordered
    );
    assert_eq!(
        rep.residues[75].concordance_class,
        AfNmrConcordanceClass::ConcordantDisordered
    );

    // Also verify Lipari-Szabo `--s2 <modelfree.tsv>` override path & HTML/SVG/JSON/TSV rendering
    let mut mf_tsv = String::from("seq_id\tres_name\tmodel\ts2\tsigma_s2\n");
    for (idx, r) in chain.residues.iter().enumerate() {
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
        evaluate_af_nmr_concordance(&af_model, &shifts, None, Some(&s2_rows), &cfg).unwrap();
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
    assert!(format_af_concordance_tsv(&rep).contains("ConcordantOrdered"));
    assert!(format_af_concordance_json(&rep)
        .unwrap()
        .contains("spearman_rho_s"));
}

#[test]
fn test_alpha_synuclein_bmr6968_concordant_disordered_ge_85_pct() {
    let shifts = read_star(repo_root().join("data/benchmark/bmr6968.str"), None)
        .expect("read bmr6968 alpha-synuclein shifts");
    let l = shifts.sequence.len();
    assert_eq!(l, 140);

    // Build an unstructured disordered AlphaFold coil trace (`pLDDT` in `34..54`)
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
    let cif = build_af_mmcif_from_chain("AF-P37840-F1", &chain, &plddts);
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

fn verify_planted_morf_100_pct_detection(
    shifts: &Reference,
    morf_start_idx: usize,
    morf_end_idx: usize,
    bmrb_id: &str,
) {
    let l = shifts.sequence.len();
    let mut phi = vec![-82.0_f64; l];
    let mut psi = vec![142.0_f64; l];
    let mut plddts = vec![41.0_f64; l];

    // Plant a high-confidence AF alpha-helix (`phi = -62 deg, psi = -42 deg, pLDDT = 89.0`)
    // spanning `(morf_start_idx - 2) ..= (morf_end_idx + 2)` so Kabsch-Sander 4-turns cover
    // the entire core MoRF window `morf_start_idx ..= morf_end_idx`.
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
        if rec.concordance_class == AfNmrConcordanceClass::ConditionalFoldingOrCrystalArtifact {
            detected_morf += 1;
        } else {
            println!(
                "MISS on bmr{bmrb_id} idx={k} {}{}: class={:?}, s2_exp={:?}, p_coil+p_ppii={:.3}, ssp={:?}",
                rec.res_name,
                rec.seq_id,
                rec.concordance_class,
                rec.s2_exp,
                rec.p_coil_nmr + rec.p_ppii_nmr,
                rec.ssp_nmr
            );
        }
    }
    assert!(evaluated_morf >= 8);
    assert_eq!(
        detected_morf, evaluated_morf,
        "expected 100% ConditionalFoldingOrCrystalArtifact detection on planted MoRF in bmr{bmrb_id}, got {detected_morf}/{evaluated_morf}"
    );
}

#[test]
fn test_planted_idp_morf_100_pct_conditional_folding_on_bmr15176_and_bmr51984() {
    let darpp32 = read_star(repo_root().join("data/benchmark/bmr15176.str"), None)
        .expect("read bmr15176 DARPP-32");
    let p53_tad = read_star(
        repo_root().join("data/fixtures/bmr51984_idp_p53_tad_prr.str"),
        None,
    )
    .expect("read bmr51984 p53 TAD-PRR");

    // 1. DARPP-32 (`bmr15176`): proline-free C-terminal disordered segment indices 82..=93 (Seq_ID 84..=95)
    verify_planted_morf_100_pct_detection(&darpp32, 82, 93, "15176");

    // 2. p53 TAD (`bmr51984`): canonical MDM2-binding TAD1 alpha-MoRF (Seq_ID 18..=26, indices 17..=25)
    verify_planted_morf_100_pct_detection(&p53_tad, 17, 25, "51984");
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
