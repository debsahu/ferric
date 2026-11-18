//! Integration and quantitative gate tests for `T21.6a` (`nmr-core::cofold_bridge`,
//! `docs/specs/21-alphafold-integrative-nmr.md` sections 6 & 8).

use std::collections::BTreeMap;
use std::f64::consts::PI;
use std::path::PathBuf;

use nmr_core::cofold_bridge::{
    compute_interface_rmsd_angstrom, compute_shrake_rupley_atom_sasa, compute_shrake_rupley_rsasa,
    export_af3_json, export_boltz2_yaml, export_nmr_restraints_sidecar_json,
    extract_nmr_interface_restraints, ligand_spec_from_octet_str, parse_cofold_fasta_str,
    parse_cofold_pcs_tsv_str, parse_cofold_rdc_tsv_str, parse_csp_restraint_str,
    parse_intermolecular_noe_tsv_str, parse_intermolecular_pre_tsv_str, score_cofold_poses,
    Af3ExportConfig, Boltz2ExportConfig, BondedAtomPairSpec, BondedAtomRef, CofoldEntitySpec,
    CofoldLigandSpec, CofoldPcsObservation, CofoldPcsTagSet, CofoldRestraintBundle,
    CofoldRestraintExtractConfig, CofoldRestraintInput, CspPerturbationClass, NmrErrorKind,
    PoseScoreConfig, PoseVerdict, PoseViolationFlag, StructurePoseInput,
};
use nmr_core::csp::amide_alpha_n;
use nmr_core::noesy::{v_add, v_cross, v_dist, v_dot, v_scale, v_sub, v_unit, Vec3};
use nmr_core::octet_mol::{write_mol_v2000, write_oct, ChemDocument};
use nmr_core::paramagnetic::{predict_pcs_spherical_ppm, DeltaChiTensor};
use nmr_core::rdc::{calc_rdc_from_unit_vec, saupe_matrix_from_pas, RdcBondType, DMAX_NH_HZ};
use nmr_core::rng::Rng;
use nmr_core::struct_shifts::{
    evaluate_haigh_mallion_ring_current, predict_structure_shifts_with_firewall,
    ConjugatedAromaticRing, StructShiftConfig, StructureModel,
};
use nmr_core::torsion_db::HomologFirewallQuery;

fn ubq_fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/fixtures/ubiquitin_1ubq_xray.pdb")
}

/// Build a 3D planar indole-amide small-molecule ligand (`c1ccc2c(c1)[nH]c(c2)C(=O)N`, 13 heavy atoms)
/// positioned in a local frame `(origin, e1, e2, normal)` for realistic Haigh-Mallion + H-bond testing.
fn build_indole_amide_heavy_atoms(
    center: Vec3,
    e1: Vec3,
    e2: Vec3,
) -> Vec<(&'static str, &'static str, Vec3)> {
    // Planar fused 6-ring + 5-ring + exocyclic carboxamide `C(=O)N` (bond lengths ~1.39 A)
    let local_xy: [(&str, &str, f64, f64); 13] = [
        // Benzene 6-ring (C1..C6)
        ("C1", "C", -1.40, 1.21),
        ("C2", "C", -2.60, 0.52),
        ("C3", "C", -2.60, -0.87),
        ("C4", "C", -1.40, -1.56),
        ("C5", "C", -0.20, -0.87),
        ("C6", "C", -0.20, 0.52),
        // Fused pyrrole 5-ring (N7, C8, C9 bonded to C5-C6)
        ("N7", "N", 1.12, -1.15),
        ("C8", "C", 1.85, 0.00),
        ("C9", "C", 1.08, 1.08),
        // Exocyclic carboxamide on C8: C10(=O11)N12 + methyl C13
        ("C10", "C", 3.32, 0.05),
        ("O11", "O", 3.96, 1.08),
        ("N12", "N", 3.98, -1.12),
        ("C13", "C", 5.42, -1.18),
    ];
    local_xy
        .into_iter()
        .map(|(aname, el, x, y)| {
            let pos = v_add(center, v_add(v_scale(e1, x), v_scale(e2, y)));
            (aname, el, pos)
        })
        .collect()
}

const INDOLE_AMIDE_SMILES: &str = "CNC(=O)c1cc2ccccc2[nH]1";

/// Render a receptor `StructureModel` (`chain A`) + small-molecule `HETATM` ligand (`chain L`)
/// as an `AlphaFold 3` / `Boltz-2` style `.cif` (where ligand `HETATM` rows carry `label_seq_id = .`!).
fn render_complex_mmcif_with_ligand(
    receptor_pdb_text: &str,
    ligand_atoms: &[(&str, &str, Vec3)],
) -> String {
    let mut cif = String::from(
        "data_cofold_complex\n\
         loop_\n\
         _atom_site.group_PDB\n\
         _atom_site.id\n\
         _atom_site.type_symbol\n\
         _atom_site.label_atom_id\n\
         _atom_site.label_comp_id\n\
         _atom_site.label_asym_id\n\
         _atom_site.label_seq_id\n\
         _atom_site.auth_asym_id\n\
         _atom_site.auth_seq_id\n\
         _atom_site.Cartn_x\n\
         _atom_site.Cartn_y\n\
         _atom_site.Cartn_z\n\
         _atom_site.B_iso_or_equiv\n\
         _atom_site.pdbx_PDB_model_num\n",
    );
    let mut serial = 1usize;
    for line in receptor_pdb_text.lines() {
        if !line.starts_with("ATOM  ") || line.len() < 54 {
            continue;
        }
        let atom_name = line[12..16].trim();
        let res_name = line[17..20].trim();
        if res_name == "HOH" {
            continue;
        }
        let seq_id = line[22..26].trim();
        let x = line[30..38].trim();
        let y = line[38..46].trim();
        let z = line[46..54].trim();
        let el: String = atom_name
            .chars()
            .filter(|c| c.is_ascii_alphabetic())
            .take(1)
            .collect();
        cif.push_str(&format!(
            "ATOM {serial} {el} {atom_name} {res_name} A {seq_id} A {seq_id} {x} {y} {z} 92.5 1\n"
        ));
        serial += 1;
    }
    for &(aname, el, pos) in ligand_atoms {
        // Note `_atom_site.label_seq_id = .` and `auth_seq_id = .` matching AF3 & Boltz-2 ligand CIFs!
        cif.push_str(&format!(
            "HETATM {serial} {el} {aname} LIG L . L . {:.3} {:.3} {:.3} 88.0 1\n",
            pos[0], pos[1], pos[2]
        ));
        serial += 1;
    }
    cif
}

#[test]
fn shrake_rupley_92pt_sasa_separates_surface_pocket_from_buried_core_on_ubiquitin() {
    // 1. Exact single-sphere analytical check: 4 * pi * (1.70 + 1.40)^2 = 120.76282 A^2
    let single = compute_shrake_rupley_atom_sasa(&[([0.0, 0.0, 0.0], 1.70)], 1.40, 92);
    let exact = 4.0 * PI * (3.10_f64).powi(2);
    assert!(
        (single[0] - exact).abs() < 1e-10,
        "single-atom Shrake-Rupley SASA {} != exact {}",
        single[0],
        exact
    );

    // 2. Real 1UBQ Ubiquitin structure: buried core vs surface Ile44-patch residues
    let ubq = StructureModel::parse_file(ubq_fixture_path(), false).expect("parse 1UBQ");
    let rsasa = compute_shrake_rupley_rsasa(&ubq, &[]);
    let by_seq: BTreeMap<i32, f64> = rsasa.iter().map(|r| (r.seq_id, r.rsasa)).collect();

    // Canonical buried hydrophobic core residues in Ubiquitin must satisfy rSASA < 0.20
    for buried_sid in [3, 17, 26, 30] {
        let v = by_seq[&buried_sid];
        assert!(
            v < 0.20,
            "expected Ubiquitin core residue {buried_sid} to be buried (rSASA < 0.20), got {v:.3}"
        );
    }
    // Canonical surface-exposed Ile44/loop binding residues must satisfy rSASA >= 0.20
    let surface_hits = [8, 9, 42, 46, 47, 49, 68];
    for &surf_sid in &surface_hits {
        let v = by_seq[&surf_sid];
        assert!(
            v >= 0.20,
            "expected Ubiquitin surface residue {surf_sid} to be exposed (rSASA >= 0.20), got {v:.3}"
        );
    }

    // 3. Plant 2-sigma CSP hits on 6 surface residues + 2 buried core allosteric relay residues (26, 30)
    let mut raw_csp = Vec::new();
    for r in &ubq.residues {
        let csp = if [8, 9, 42, 46, 49, 68].contains(&r.seq_id) {
            0.34
        } else if [26, 30].contains(&r.seq_id) {
            // Buried allosteric core shift!
            0.31
        } else {
            0.015 + 0.004 * ((r.seq_id % 5) as f64)
        };
        raw_csp.push((r.seq_id, r.res_name.clone(), csp, None, None));
    }

    let (lig_spec, lig_mol) =
        ligand_spec_from_octet_str(INDOLE_AMIDE_SMILES, Some("smi")).expect("parse SMILES");
    let mut lig_map = BTreeMap::new();
    lig_map.insert("L".to_string(), lig_mol);

    let seq_a: String = ubq.residues.iter().map(|r| char::from(r.aa1)).collect();
    let input = CofoldRestraintInput {
        job_name: "ubq_indole_test".to_string(),
        entities: vec![
            CofoldEntitySpec::Polymer {
                kind: nmr_core::cofold_bridge::CofoldPolymerKind::Protein,
                chain_id: "A".to_string(),
                sequence: seq_a,
            },
            CofoldEntitySpec::Ligand {
                chain_id: "L".to_string(),
                spec: lig_spec,
            },
        ],
        ligand_molecules: lig_map,
        receptor_chain_id: Some("A".to_string()),
        binder_chain_id: Some("L".to_string()),
        apo_receptor_model: Some(ubq),
        raw_csp_rows: raw_csp,
        ..Default::default()
    };

    let bundle = extract_nmr_interface_restraints(&input, &CofoldRestraintExtractConfig::default())
        .expect("extract restraints");

    let pocket_sids: Vec<i32> = bundle.pocket_contacts.iter().map(|p| p.seq_id).collect();
    let core_sids: Vec<i32> = bundle
        .allosteric_core_perturbations
        .iter()
        .map(|p| p.seq_id)
        .collect();

    assert_eq!(pocket_sids, vec![8, 9, 42, 46, 49, 68]);
    assert_eq!(core_sids, vec![26, 30]);
    assert!(bundle
        .allosteric_core_perturbations
        .iter()
        .all(|p| p.classification == CspPerturbationClass::AllostericCorePerturbation));
}

#[test]
fn deterministic_af3_json_v2_and_boltz2_yaml_v1_schemas_and_octet_ligands() {
    // Multi-chain FASTA (`protein` + `rna` + `dna`)
    let fasta = "\
>A|protein
MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG
>R|rna
GGGAAAUCC
>D|dna
ACGTACGT
";
    let mut entities = parse_cofold_fasta_str(fasta).expect("parse multi-chain FASTA");
    assert_eq!(entities.len(), 3);

    // Verify `Octet` SMILES -> MOL V2000 & `.oct` round-trip ingestion into `CofoldLigandSpec`
    let (_, orig_mol) =
        ligand_spec_from_octet_str(INDOLE_AMIDE_SMILES, Some("smi")).expect("parse SMILES");
    let mol_v2000 = write_mol_v2000(&orig_mol).expect("write MOL V2000");
    let (mol_spec, _) = ligand_spec_from_octet_str(&mol_v2000, Some("mol")).expect("read MOL");
    let mut doc = ChemDocument::new();
    doc.molecules.push(orig_mol.clone());
    let oct_text = write_oct(&doc).expect("write .oct");
    let (oct_spec, oct_mol) =
        ligand_spec_from_octet_str(&oct_text, Some("oct")).expect("read .oct");
    assert_eq!(mol_spec, oct_spec);

    let mut lig_map = BTreeMap::new();
    lig_map.insert("L".to_string(), oct_mol);

    entities.push(CofoldEntitySpec::Ligand {
        chain_id: "L".to_string(),
        spec: oct_spec,
    });
    entities.push(CofoldEntitySpec::Ligand {
        chain_id: "M".to_string(),
        spec: CofoldLigandSpec::CcdCodes(vec!["ZN".to_string()]),
    });

    let ubq = StructureModel::parse_file(ubq_fixture_path(), false).expect("parse 1UBQ");
    let mut csp_tsv = String::from("seq_id\tres_name\tcsp_ppm\n");
    for r in &ubq.residues {
        let v = if [8, 9, 42, 46, 49, 68].contains(&r.seq_id) {
            0.360
        } else if [26, 30].contains(&r.seq_id) {
            0.340
        } else {
            0.015 + 0.003 * ((r.seq_id % 4) as f64)
        };
        csp_tsv.push_str(&format!("{}\t{}\t{v:.4}\n", r.seq_id, r.res_name));
    }
    let (raw_csp, _) = parse_csp_restraint_str(&csp_tsv).expect("parse CSP TSV");

    // Intermolecular NOE (3.2 A -> clamped to 4.5 A) & PRE (bleach 11.0 A, plus 18.0 A -> clamped to 14.0 A)
    let noe_tsv = "\
chain_1\tseq_id_1\tatom_1\tchain_2\tatom_2\tupper_a\tlower_a\tsigma_a
A\t42\tH\tL\tO11\t3.20\t1.80\t0.40
A\t68\tH\tR\t5\tN1\t7.50\t1.80\t0.50
";
    let mut dist_restraints = parse_intermolecular_noe_tsv_str(noe_tsv).expect("parse NOE TSV");
    let pre_tsv = "\
# anchor: L 1 C8
A\t8\tH\tbleach\t11.0
# mode: bounds
A\t46\tH\tL\tC1\t1.80\t16.50\t1.00
";
    let pre_restraints =
        parse_intermolecular_pre_tsv_str(pre_tsv, None, "A", 1.5).expect("parse PRE TSV");
    dist_restraints.extend(pre_restraints);

    let extract_cfg = CofoldRestraintExtractConfig {
        request_boltz2_affinity: true,
        ..CofoldRestraintExtractConfig::default()
    };

    let input = CofoldRestraintInput {
        job_name: "ubq_multimodal_complex".to_string(),
        entities,
        ligand_molecules: lig_map,
        receptor_chain_id: Some("A".to_string()),
        binder_chain_id: Some("L".to_string()),
        apo_receptor_model: Some(ubq),
        raw_csp_rows: raw_csp,
        raw_distance_restraints: dist_restraints,
        bonded_atom_pairs: vec![BondedAtomPairSpec {
            atom1: BondedAtomRef {
                chain_id: "A".to_string(),
                one_based_res_idx: 68,
                atom_name: "NE2".to_string(),
            },
            atom2: BondedAtomRef {
                chain_id: "M".to_string(),
                one_based_res_idx: 1,
                atom_name: "ZN".to_string(),
            },
        }],
        experimental_kd_um: Some(12.5),
        ..Default::default()
    };

    let bundle =
        extract_nmr_interface_restraints(&input, &extract_cfg).expect("extract multimodal bundle");

    // 1. Validate AlphaFold 3 JSON v2 schema honesty & byte-exact determinism
    let af3_cfg = Af3ExportConfig {
        model_seeds: vec![1, 2, 3],
    };
    let af3_a = export_af3_json(&bundle, &af3_cfg).expect("export AF3 JSON #1");
    let af3_b = export_af3_json(&bundle, &af3_cfg).expect("export AF3 JSON #2");
    assert_eq!(af3_a, af3_b, "AF3 JSON export must be deterministic");

    let af3_val: serde_json::Value = serde_json::from_str(&af3_a).expect("valid AF3 JSON");
    let obj = af3_val.as_object().expect("JSON object");
    assert_eq!(
        obj.get("dialect").and_then(|v| v.as_str()),
        Some("alphafold3")
    );
    assert_eq!(obj.get("version").and_then(|v| v.as_i64()), Some(2));
    assert_eq!(
        obj.get("name").and_then(|v| v.as_str()),
        Some("ubq_multimodal_complex")
    );
    assert_eq!(
        obj.get("modelSeeds").unwrap(),
        &serde_json::json!([1, 2, 3])
    );
    assert_eq!(obj.get("sequences").unwrap().as_array().unwrap().len(), 5);
    assert_eq!(
        obj.get("bondedAtomPairs")
            .unwrap()
            .as_array()
            .unwrap()
            .len(),
        1
    );
    // Schema-honesty check: ZERO invented pocket/constraint keys in AF3 JSON v2!
    for forbidden_key in ["pocket", "constraints", "restraints", "nmr", "contacts"] {
        assert!(
            !obj.contains_key(forbidden_key),
            "AF3 JSON v2 must not invent non-existent key '{forbidden_key}'"
        );
    }

    // Sidecar JSON lossless round-trip
    let sidecar_str = export_nmr_restraints_sidecar_json(&bundle).expect("export sidecar");
    let roundtrip: CofoldRestraintBundle =
        serde_json::from_str(&sidecar_str).expect("parse sidecar JSON");
    assert_eq!(roundtrip.pocket_contacts, bundle.pocket_contacts);
    assert_eq!(
        roundtrip.allosteric_core_perturbations,
        bundle.allosteric_core_perturbations
    );

    // 2. Validate Boltz-2 YAML v1 determinism, 1-based pocket & [4.5, 14.0] A contact clamping, and affinity
    let boltz_a =
        export_boltz2_yaml(&bundle, &Boltz2ExportConfig::default()).expect("export Boltz-2 #1");
    let boltz_b =
        export_boltz2_yaml(&bundle, &Boltz2ExportConfig::default()).expect("export Boltz-2 #2");
    assert_eq!(
        boltz_a, boltz_b,
        "Boltz-2 YAML export must be deterministic"
    );

    assert!(boltz_a.starts_with("version: 1\nsequences:\n"));
    assert!(boltz_a.contains("  - protein:\n      id: A\n"));
    assert!(boltz_a.contains("  - rna:\n      id: R\n      sequence: GGGAAAUCC\n"));
    assert!(boltz_a.contains("  - dna:\n      id: D\n      sequence: ACGTACGT\n"));
    assert!(boltz_a.contains("  - ligand:\n      id: L\n      smiles: '"));
    assert!(boltz_a.contains("  - ligand:\n      id: M\n      ccd: ZN\n"));
    assert!(boltz_a.contains("  - pocket:\n      binder: L\n      contacts:\n"));
    // Surface 1-based indices [8, 9, 42, 46, 49, 68] present; buried [26, 30] absent!
    for idx1 in [8, 9, 42, 46, 49, 68] {
        assert!(
            boltz_a.contains(&format!("        - [A, {idx1}]\n")),
            "missing 1-based surface pocket contact [A, {idx1}] in Boltz-2 YAML:\n{boltz_a}"
        );
    }
    for buried_idx1 in [26, 30] {
        assert!(
            !boltz_a.contains(&format!("        - [A, {buried_idx1}]\n")),
            "buried allosteric core hit [A, {buried_idx1}] leaked into Boltz-2 pocket!"
        );
    }
    assert!(boltz_a.contains("      max_distance: 6.0\n      force: false\n"));
    // Clamped 3.20 -> 4.50 and 16.50 -> 14.00!
    assert!(boltz_a.contains("      max_distance: 4.50\n"));
    assert!(boltz_a.contains("      max_distance: 14.00\n"));
    assert!(boltz_a.contains("      max_distance: 11.00\n"));
    assert!(boltz_a.contains("properties:\n  - affinity:\n      binder: L\n"));
}

#[test]
fn ten_pose_decoy_benchmark_achieves_top1_native_spearman_ge_080_and_decoy_only_abstention() {
    let ubq_pdb_text = std::fs::read_to_string(ubq_fixture_path()).expect("read 1UBQ PDB");
    let ubq = StructureModel::parse_str(&ubq_pdb_text, Some("pdb"), false).expect("parse 1UBQ");

    // Compute Ubiquitin centroid and anchor the indole-amide ligand right above the surface Ile44/His68/Leu8 pocket!
    let n_res = ubq.residues.len() as f64;
    let prot_centroid = ubq.residues.iter().fold([0.0; 3], |acc, r| {
        [
            acc[0] + r.ca[0] / n_res,
            acc[1] + r.ca[1] / n_res,
            acc[2] + r.ca[2] / n_res,
        ]
    });

    let by_sid: BTreeMap<i32, usize> = ubq
        .residues
        .iter()
        .enumerate()
        .map(|(i, r)| (r.seq_id, i))
        .collect();
    let p_ile44 = ubq.residues[by_sid[&44]]
        .cb
        .unwrap_or(ubq.residues[by_sid[&44]].ca);
    let p_his68 = ubq.residues[by_sid[&68]].ca;
    let p_leu8 = ubq.residues[by_sid[&8]].ca;
    let pocket_center = v_scale(v_add(p_ile44, v_add(p_his68, p_leu8)), 1.0 / 3.0);
    let outward = v_unit(v_sub(pocket_center, prot_centroid));
    let e1 = v_unit(v_sub(p_his68, p_leu8));
    let normal = v_unit(v_cross(e1, outward));
    let e2 = v_unit(v_cross(normal, e1));

    // Place native ligand in van der Waals contact (~3.2..4.3 A heavy-atom distance) with the Ile44/His68/Leu8 pocket
    let native_lig_center = v_add(pocket_center, v_scale(outward, 2.3));
    let native_lig = build_indole_amide_heavy_atoms(native_lig_center, e1, e2);
    let native_cif = render_complex_mmcif_with_ligand(&ubq_pdb_text, &native_lig);

    // Synthesize experimental CSPs + 3 intermolecular NOEs/PREs from the true native complex (+ 4 ppb measurement noise)
    let (lig_spec, lig_mol) =
        ligand_spec_from_octet_str(INDOLE_AMIDE_SMILES, Some("smi")).expect("parse ligand");
    let mut lig_map = BTreeMap::new();
    lig_map.insert("L".to_string(), lig_mol);

    // Compute true physical differential shifts on the exact native reference
    let apo_shifts = predict_structure_shifts_with_firewall(
        &ubq,
        &HomologFirewallQuery::default(),
        &StructShiftConfig::default().with_uniform_plddt(100.0),
    )
    .expect("predict apo shifts");

    let benz_verts: Vec<Vec3> = native_lig[0..6].iter().map(|a| a.2).collect();
    let pyr_verts: Vec<Vec3> = vec![
        native_lig[4].2,
        native_lig[5].2,
        native_lig[6].2,
        native_lig[7].2,
        native_lig[8].2,
    ];
    let ring_norm = v_unit(v_cross(e1, e2));
    let native_rings = vec![
        ConjugatedAromaticRing {
            chain_id: "L".to_string(),
            seq_id: 1,
            residue_index: usize::MAX,
            res_name: "LIG".to_string(),
            ring_label: "BENZ".to_string(),
            center: native_lig_center,
            normal: ring_norm,
            vertices: benz_verts,
            intensity_factor: 1.00,
        },
        ConjugatedAromaticRing {
            chain_id: "L".to_string(),
            seq_id: 1,
            residue_index: usize::MAX,
            res_name: "LIG".to_string(),
            ring_label: "PYR".to_string(),
            center: native_lig_center,
            normal: ring_norm,
            vertices: pyr_verts,
            intensity_factor: 0.56,
        },
    ];
    let lig_heavy_occ: Vec<(Vec3, u8)> = native_lig
        .iter()
        .map(|&(_, el, p)| {
            (
                p,
                if el == "N" {
                    7
                } else if el == "O" {
                    8
                } else {
                    6
                },
            )
        })
        .collect();
    let sasa_apo = compute_shrake_rupley_rsasa(&ubq, &[]);
    let sasa_nat = compute_shrake_rupley_rsasa(&ubq, &lig_heavy_occ);

    let mut rng = Rng::new(20262106);
    let mut raw_csp_rows = Vec::new();
    let _ = apo_shifts;
    for (idx, r) in ubq.residues.iter().enumerate() {
        let mut dh = 0.0;
        let mut dn = 0.0;
        if let Some(h_pos) = r.h {
            let (rc_h, _) =
                evaluate_haigh_mallion_ring_current(&native_rings, h_pos, usize::MAX, false, 1.0);
            dh += rc_h;
            // Direct H-bond to ligand carbonyl O11
            let o11 = native_lig[10].2;
            let d_ho = v_dist(h_pos, o11);
            if (1.55..=3.20).contains(&d_ho) {
                let cos_ang = v_dot(v_unit(v_sub(r.n, h_pos)), v_unit(v_sub(o11, h_pos)));
                if cos_ang <= -0.34202014 {
                    let ex = (1.0 / d_ho.max(1.65).powi(3) - 1.0 / 2.60_f64.powi(3)).max(0.0);
                    dh += (4.15 * ex * cos_ang * cos_ang).min(1.35);
                    dn += (8.80 * ex * cos_ang * cos_ang).min(3.00);
                }
            }
        }
        if r.res_name != "PRO" {
            let (rc_n, _) =
                evaluate_haigh_mallion_ring_current(&native_rings, r.n, usize::MAX, false, 0.65);
            dn += rc_n;
        }
        let d_min = r
            .atoms
            .values()
            .flat_map(|&ap| native_lig.iter().map(move |&(_, _, lp)| v_dist(ap, lp)))
            .fold(f64::INFINITY, f64::min);
        let d_rsasa = (sasa_apo[idx].rsasa - sasa_nat[idx].rsasa).max(0.0);
        let prox = if d_min < 6.5 {
            let u = ((6.5 - d_min.max(3.2)) / 3.3).clamp(0.0, 1.0);
            u * u * (3.0 - 2.0 * u)
        } else {
            0.0
        };
        let true_csp = dh
            .hypot(amide_alpha_n(&r.res_name) * dn)
            .hypot(0.12 * d_rsasa + 0.08 * prox);
        let noisy_csp = (true_csp + 0.004 + rng.normal(0.0, 0.0025)).max(0.001);
        raw_csp_rows.push((r.seq_id, r.res_name.clone(), noisy_csp, Some(dh), Some(dn)));
    }

    // Two intermolecular NOEs + one PRE upper bound from true native geometry (+ 1.5 A upper margin)
    let h_44 = ubq.residues[by_sid[&44]].h.unwrap();
    let h_68 = ubq.residues[by_sid[&68]].h.unwrap();
    let h_8 = ubq.residues[by_sid[&8]].h.unwrap();
    let noe_tsv = format!(
        "chain_1\tseq_id_1\tatom_1\tchain_2\tatom_2\tupper_a\tlower_a\tsigma_a\n\
         A\t44\tH\tL\tC5\t{:.2}\t1.80\t0.50\n\
         A\t68\tH\tL\tC1\t{:.2}\t1.80\t0.50\n",
        v_dist(h_44, native_lig[4].2) + 0.85,
        v_dist(h_68, native_lig[0].2) + 0.85,
    );
    let pre_tsv = format!(
        "# anchor: L 1 C6\n\
         A\t8\tH\tbleach\t{:.2}\n",
        (v_dist(h_8, native_lig[5].2) + 1.20).clamp(5.0, 12.0),
    );
    let mut dist_restraints = parse_intermolecular_noe_tsv_str(&noe_tsv).unwrap();
    dist_restraints.extend(parse_intermolecular_pre_tsv_str(&pre_tsv, None, "A", 1.5).unwrap());

    let seq_a: String = ubq.residues.iter().map(|r| char::from(r.aa1)).collect();
    let restraint_input = CofoldRestraintInput {
        job_name: "ubq_10pose_benchmark".to_string(),
        entities: vec![
            CofoldEntitySpec::Polymer {
                kind: nmr_core::cofold_bridge::CofoldPolymerKind::Protein,
                chain_id: "A".to_string(),
                sequence: seq_a,
            },
            CofoldEntitySpec::Ligand {
                chain_id: "L".to_string(),
                spec: lig_spec,
            },
        ],
        ligand_molecules: lig_map,
        receptor_chain_id: Some("A".to_string()),
        binder_chain_id: Some("L".to_string()),
        apo_receptor_model: Some(ubq.clone()),
        raw_csp_rows,
        raw_distance_restraints: dist_restraints,
        ..Default::default()
    };
    let bundle = extract_nmr_interface_restraints(
        &restraint_input,
        &CofoldRestraintExtractConfig::default(),
    )
    .expect("extract benchmark restraints");
    assert!(
        bundle.pocket_contacts.len() >= 5,
        "expected >= 5 surface CSP hits on Ile44 pocket, got {}",
        bundle.pocket_contacts.len()
    );

    // Construct 10 candidate poses:
    // - `pose_near_native`: small 0.65 A shift (`iRMSD < 1.5 A`)
    // - `pose_decoy_01`..`pose_decoy_09`: 9 decoys spanning `4.0 .. 18.0 A` iRMSD (tangential slides,
    //   flipped orientation, opposite-face docking, and solvent-detached poses)
    let mut poses: Vec<StructurePoseInput> = Vec::with_capacity(10);

    let near_center = v_add(
        native_lig_center,
        v_add(v_scale(e1, 0.55), v_scale(outward, 0.35)),
    );
    let near_lig = build_indole_amide_heavy_atoms(near_center, e1, e2);
    let near_cif = render_complex_mmcif_with_ligand(&ubq_pdb_text, &near_lig);
    let near_irmsd =
        compute_interface_rmsd_angstrom(&near_cif, Some("cif"), &native_cif, Some("cif"), "A")
            .expect("near-native iRMSD");
    assert!(
        near_irmsd > 0.10 && near_irmsd < 1.50,
        "near-native pose iRMSD ({near_irmsd:.3} A) must be < 1.50 A"
    );
    poses.push(StructurePoseInput::from_coord_str(
        "pose_near_native",
        near_cif,
        Some("cif"),
    ));

    // 9 decoys calibrated across `4.0..=18.0 A` pocket-aligned iRMSD
    let decoy_displacements: [(f64, f64, f64, bool); 9] = [
        (4.3, 1.5, 1.0, false),  // ~4.66 A iRMSD
        (-5.2, 2.0, 1.2, false), // ~5.70 A iRMSD
        (3.0, 5.8, 2.0, false),  // ~6.83 A iRMSD
        (-6.0, -4.5, 2.5, true), // ~8.7 A iRMSD
        (7.5, -5.0, 3.0, true),  // ~10.2 A iRMSD
        (0.0, 0.0, 11.5, false), // ~11.5 A iRMSD (detached into solvent)
        (-9.5, 6.5, 4.0, true),  // ~13.0 A iRMSD (distal surface patch)
        (10.5, -8.0, 4.5, true), // ~14.8 A iRMSD (distal surface patch)
        (0.0, 0.0, 15.2, true),  // ~16.1 A iRMSD (far detached)
    ];

    let mut decoy_poses_only = Vec::with_capacity(9);
    for (d_idx, &(dx, dy, dz, flip_e1)) in decoy_displacements.iter().enumerate() {
        let c = v_add(
            native_lig_center,
            v_add(
                v_scale(e1, dx),
                v_add(v_scale(e2, dy), v_scale(outward, dz)),
            ),
        );
        let dir1 = if flip_e1 { v_scale(e1, -1.0) } else { e1 };
        let dir2 = if flip_e1 { v_scale(e2, -1.0) } else { e2 };
        let lig = build_indole_amide_heavy_atoms(c, dir1, dir2);
        let cif = render_complex_mmcif_with_ligand(&ubq_pdb_text, &lig);
        let irmsd =
            compute_interface_rmsd_angstrom(&cif, Some("cif"), &native_cif, Some("cif"), "A")
                .expect("decoy iRMSD");
        assert!(
            (4.0..=18.0).contains(&irmsd),
            "decoy {} iRMSD = {irmsd:.3} A must lie inside [4.0, 18.0] A",
            d_idx + 1
        );
        let p = StructurePoseInput::from_coord_str(
            format!("pose_decoy_{:02}", d_idx + 1),
            cif,
            Some("cif"),
        );
        decoy_poses_only.push(p.clone());
        poses.push(p);
    }

    let score_cfg = PoseScoreConfig {
        apo_receptor_coord_text: Some(ubq_pdb_text),
        apo_receptor_format_hint: Some("pdb".to_string()),
        reference_complex_pose: Some((native_cif, Some("cif".to_string()))),
        ..PoseScoreConfig::default()
    };

    // Gate 1: Full 10-pose benchmark (`1` near-native + `9` `4-18 A` decoys) -> Top-1 = 100%, Spearman >= 0.80
    let report = score_cofold_poses(&poses, &bundle, &score_cfg).expect("score 10 poses");
    assert_eq!(report.verdict, PoseVerdict::RankedConclusive);
    assert_eq!(
        report.top_pose_id.as_deref(),
        Some("pose_near_native"),
        "expected pose_near_native at Rank #1, got {:?}",
        report.poses.first().map(|p| (&p.pose_id, p.total_score))
    );
    let top = &report.poses[0];
    assert_eq!(top.rank, 1);
    assert!(top.passes_nmr_guards);
    assert_eq!(top.exclusion_list_sha256.len(), 64);

    let top_csp_rho = top.csp_spearman_rho.expect("near-native CSP Spearman rho");
    assert!(
        top_csp_rho >= 0.80,
        "expected near-native CSP Spearman rho >= 0.80, got {top_csp_rho:.4}"
    );
    let funnel_rho = report
        .funnel_spearman_rho_vs_neg_irmsd
        .expect("10-pose funnel Spearman rho");
    assert!(
        funnel_rho >= 0.80,
        "expected 10-pose funnel Spearman rho(-iRMSD, total_score) >= 0.80, got {funnel_rho:.4}"
    );

    println!(
        "[T21.6a 10-pose gate] near_irmsd={near_irmsd:.3} A, top_csp_rho={top_csp_rho:.4}, top_auc={:.4}, funnel_rho={funnel_rho:.4}",
        top.csp_contact_roc_auc.unwrap_or(0.0)
    );

    // Gate 2: Decoy-only ensemble -> `PoseVerdict::InconclusiveOrAllViolated`!
    let decoy_report =
        score_cofold_poses(&decoy_poses_only, &bundle, &score_cfg).expect("score 9 decoys only");
    assert_eq!(decoy_report.verdict, PoseVerdict::InconclusiveOrAllViolated);
    assert_eq!(decoy_report.top_pose_id, None);
    assert_eq!(decoy_report.n_admissible_poses, 0);
}

#[test]
fn multi_subunit_rdc_by_peptide_plane_qfree_and_pcs_discriminate_complex_orientations() {
    let ubq_pdb_text = std::fs::read_to_string(ubq_fixture_path()).expect("read 1UBQ PDB");
    let ubq = StructureModel::parse_str(&ubq_pdb_text, Some("pdb"), false).expect("parse 1UBQ");

    // Build a two-chain protein-protein complex: Chain A = Ubiquitin (1..76), Chain B = Ubiquitin (1..35)
    // translated by `[+28.0, 0.0, 0.0] A` (overlapping `seq_id = 1..35` across chains A & B!).
    let make_two_chain_pdb = |rot_b_90deg_z: bool| -> String {
        let mut out = String::new();
        for line in ubq_pdb_text.lines() {
            if !line.starts_with("ATOM  ") || line.len() < 54 {
                continue;
            }
            if &line[17..20] == "HOH" {
                continue;
            }
            out.push_str(line);
            out.push('\n');
        }
        out.push_str("TER\n");
        for line in ubq_pdb_text.lines() {
            if !line.starts_with("ATOM  ") || line.len() < 54 {
                continue;
            }
            if &line[17..20] == "HOH" {
                continue;
            }
            let x: f64 = line[30..38].trim().parse().unwrap();
            let y: f64 = line[38..46].trim().parse().unwrap();
            let z: f64 = line[46..54].trim().parse().unwrap();
            let (bx, by, bz) = if rot_b_90deg_z {
                // Proper 90-deg X-axis rotation `(x, y, z) -> (x + 34, -z, y)` swapping Y <-> Z
                (x + 34.0, -z, y)
            } else {
                (x + 34.0, y, z)
            };
            let mut mod_line = String::with_capacity(80);
            mod_line.push_str(&line[..21]);
            mod_line.push('B');
            mod_line.push_str(&line[22..30]);
            mod_line.push_str(&format!("{bx:8.3}{by:8.3}{bz:8.3}"));
            if line.len() > 54 {
                mod_line.push_str(&line[54..]);
            }
            out.push_str(&mod_line);
            out.push('\n');
        }
        out.push_str("END\n");
        out
    };

    let native_ab_pdb = make_two_chain_pdb(false);
    let misoriented_ab_pdb = make_two_chain_pdb(true);

    let native_model =
        StructureModel::parse_str(&native_ab_pdb, Some("pdb"), false).expect("parse AB native");

    // Synthesize shared-Saupe-tensor multi-subunit 1D_NH RDCs across BOTH Chain A and Chain B (`seq_id 1..76` on both!)
    let s_mat = saupe_matrix_from_pas(12.0, 2.0 / 3.0, [0.0, 0.0, 0.0]);
    let mut rdc_tsv = String::from(
        "medium_id\tchain_1\tseq_id_1\tres_name_1\tatom_1\tseq_id_2\tres_name_2\tatom_2\td_obs_hz\tsigma_hz\n",
    );
    for r in &native_model.residues {
        let Some(h_pos) = r.h else { continue };
        let u_nh = v_unit(v_sub(h_pos, r.n));
        let d_hz = calc_rdc_from_unit_vec(u_nh, DMAX_NH_HZ, &s_mat);
        rdc_tsv.push_str(&format!(
            "pf1\t{}\t{}\t{}\tN\t{}\t{}\tH\t{:.4}\t0.35\n",
            r.chain_id, r.seq_id, r.res_name, r.seq_id, r.res_name, d_hz
        ));
    }
    let rdc_media = parse_cofold_rdc_tsv_str(&rdc_tsv, "A").expect("parse multi-chain RDC TSV");

    // Synthesize lanthanide PCSs anchored on Chain A His68 CB evaluated across BOTH Chain A and Chain B!
    let metal_pos = ubq.residues.iter().find(|r| r.seq_id == 68).unwrap().ca;
    let euler_paf = [0.4, 0.9, 0.3];
    let mut pcs_obs = Vec::new();
    for r in &native_model.residues {
        let Some(h_pos) = r.h else { continue };
        if v_dist(h_pos, metal_pos) < 8.0 {
            continue;
        }
        let pcs_val = predict_pcs_spherical_ppm(h_pos, metal_pos, 18.0, 6.0, euler_paf);
        pcs_obs.push(CofoldPcsObservation {
            chain_id: r.chain_id.clone(),
            seq_id: r.seq_id,
            atom_name: "H".to_string(),
            pcs_obs_ppm: pcs_val,
            sigma_ppm: 0.03,
            is_free: r.chain_id == "B",
        });
    }
    let pcs_tag = CofoldPcsTagSet {
        tag_id: "Tm3_His68".to_string(),
        metal_pos: None,
        metal_anchor: Some(("A".to_string(), 68, "CA".to_string())),
        host_chain_id: Some("A".to_string()),
        pre_fitted_tensor: None as Option<DeltaChiTensor>,
        observations: pcs_obs,
    };

    // Also test PCS TSV parser round-trip
    let pcs_tsv_sample = "# metal_anchor: A 68 CA\n# host_chain: A\nA\t10\tH\t0.42\t0.03\nB\t12\tH\t-0.18\t0.03\t1\n";
    let parsed_pcs = parse_cofold_pcs_tsv_str(pcs_tsv_sample, "A").expect("parse PCS TSV");
    assert_eq!(parsed_pcs[0].observations.len(), 2);

    let input = CofoldRestraintInput {
        job_name: "ab_rdc_pcs_test".to_string(),
        receptor_chain_id: Some("A".to_string()),
        binder_chain_id: Some("B".to_string()),
        apo_receptor_model: Some(ubq),
        rdc_media,
        pcs_tags: vec![pcs_tag],
        ..Default::default()
    };
    let bundle = extract_nmr_interface_restraints(&input, &CofoldRestraintExtractConfig::default())
        .expect("extract AB bundle");
    let _ = RdcBondType::NH;

    let poses = vec![
        StructurePoseInput::from_coord_str(
            "ab_misoriented_90deg",
            misoriented_ab_pdb.clone(),
            Some("pdb"),
        ),
        StructurePoseInput::from_coord_str("ab_native", native_ab_pdb, Some("pdb")),
    ];
    let report = score_cofold_poses(&poses, &bundle, &PoseScoreConfig::default())
        .expect("score AB complex poses");

    assert_eq!(report.verdict, PoseVerdict::RankedConclusive);
    assert_eq!(report.top_pose_id.as_deref(), Some("ab_native"));
    let p_nat = &report.poses[0];
    let p_bad = &report.poses[1];
    assert!(
        p_nat.rdc_q_free.unwrap() < 0.05,
        "native complex Q_free should be ~0, got {:?}",
        p_nat.rdc_q_free
    );
    assert!(
        p_nat.pcs_q_factor.unwrap() < 0.05,
        "native complex PCS Q should be ~0, got {:?}",
        p_nat.pcs_q_factor
    );
    assert!(
        p_bad.rdc_q_free.unwrap() > 0.65,
        "90-deg misoriented Chain B must trigger HighRdcQFree (> 0.65), got {:?}",
        p_bad.rdc_q_free
    );
    assert!(p_bad
        .violation_flags
        .contains(&PoseViolationFlag::HighRdcQFree));

    // When ONLY the 90-deg misoriented pose is supplied, `HighRdcQFree` triggers `InconclusiveOrAllViolated`!
    let bad_only = score_cofold_poses(&poses[0..1], &bundle, &PoseScoreConfig::default())
        .expect("score misoriented only");
    assert_eq!(bad_only.verdict, PoseVerdict::InconclusiveOrAllViolated);
}

#[test]
fn hostile_inputs_and_zero_based_indices_return_structured_errors() {
    let empty_bundle = extract_nmr_interface_restraints(
        &CofoldRestraintInput::default(),
        &CofoldRestraintExtractConfig::default(),
    )
    .unwrap();
    let err = score_cofold_poses(&[], &empty_bundle, &PoseScoreConfig::default()).unwrap_err();
    assert_eq!(err.kind, NmrErrorKind::EmptyInput);

    // NUL byte in pose coordinate input must fail cleanly
    let bad_pose =
        StructurePoseInput::from_coord_str("nul_pose", "ATOM      1  N   ALA A   1\0", Some("pdb"));
    let err_nul =
        score_cofold_poses(&[bad_pose], &empty_bundle, &PoseScoreConfig::default()).unwrap_err();
    assert_eq!(err_nul.kind, NmrErrorKind::ParseError);

    // Zero-based `bondedAtomPairs` rejected in AF3 export
    let mut bad_af3_bundle = empty_bundle;
    bad_af3_bundle.entities.push(CofoldEntitySpec::Polymer {
        kind: nmr_core::cofold_bridge::CofoldPolymerKind::Protein,
        chain_id: "A".to_string(),
        sequence: "MQIFVKTLTG".to_string(),
    });
    bad_af3_bundle.bonded_atom_pairs.push(BondedAtomPairSpec {
        atom1: BondedAtomRef {
            chain_id: "A".to_string(),
            one_based_res_idx: 0,
            atom_name: "SG".to_string(),
        },
        atom2: BondedAtomRef {
            chain_id: "A".to_string(),
            one_based_res_idx: 5,
            atom_name: "SG".to_string(),
        },
    });
    assert!(export_af3_json(&bad_af3_bundle, &Af3ExportConfig::default()).is_err());
}
