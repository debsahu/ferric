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
    parse_intermolecular_noe_tsv_str, parse_intermolecular_pre_tsv_str,
    parse_intermolecular_pre_tsv_str_with_params, score_cofold_poses,
    solomon_distance_from_gamma2_a, solomon_gamma2_to_distance_bounds, solomon_pre_k_a6_s2,
    solomon_pre_k_eff_a6_s1, Af3ExportConfig, Boltz2ExportConfig, BondedAtomPairSpec,
    BondedAtomRef, CofoldEntitySpec, CofoldLigandSpec, CofoldPcsObservation, CofoldPcsTagSet,
    CofoldPolymerKind, CofoldRestraintBundle, CofoldRestraintExtractConfig, CofoldRestraintInput,
    CspPerturbationClass, CspScoringMode, NmrErrorKind, PoseScoreConfig, PoseVerdict,
    PoseViolationFlag, PreSolomonParams, RestraintNumbering, StructurePoseInput,
    BOLTZ2_MAX_DISTANCE_A, BOLTZ2_MIN_DISTANCE_A, DEFAULT_MIN_CSP_ROC_AUC,
    DEFAULT_MTSL_TETHER_PAD_A,
};
use nmr_core::csp::amide_alpha_n;
use nmr_core::noesy::{v_add, v_cross, v_dist, v_dot, v_scale, v_sub, v_unit, Vec3};
use nmr_core::octet_mol::{write_mol_v2000, write_oct, ChemDocument};
use nmr_core::paramagnetic::{predict_pcs_spherical_ppm, DeltaChiTensor};
use nmr_core::rdc::{calc_rdc_from_unit_vec, saupe_matrix_from_pas, RdcBondType, DMAX_NH_HZ};
use nmr_core::rng::Rng;
use nmr_core::struct_shifts::StructureModel;

fn ubq_fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/fixtures/ubiquitin_1ubq_xray.pdb")
}

fn ubq_1d3z_nmr_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/pdb_refs/1d3z.pdb")
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

    // Intermolecular NOE on CCD/polymer chains (3.20 A -> clamped up to 4.50 A floor, 7.50 A kept)
    // and PRE (bleach 11.0 A kept; weak 16.50 A > 14.0 A ceiling DROPPED rather than clamped!).
    let noe_tsv = "\
chain_1\tseq_id_1\tatom_1\tchain_2\tatom_2\tupper_a\tlower_a\tsigma_a
A\t42\tH\tM\tZN\t3.20\t1.80\t0.40
A\t68\tH\tR\t5\tN1\t7.50\t1.80\t0.50
";
    let mut dist_restraints = parse_intermolecular_noe_tsv_str(noe_tsv).expect("parse NOE TSV");
    let pre_tsv = "\
# anchor: M 1 ZN
A\t8\tH\tbleach\t11.0
# mode: bounds
A\t46\tH\tM\tZN\t1.80\t16.50\t1.00
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
    // 3.20 A clamped up to 4.50 A floor, 11.00 A and 7.50 A kept, and 16.50 A DROPPED (never clamped to 14.00 A)!
    assert_eq!(bundle.dropped_weak_restraints_count, 1);
    assert!(boltz_a.contains("      max_distance: 4.50\n"));
    assert!(boltz_a.contains("      max_distance: 7.50\n"));
    assert!(boltz_a.contains("      max_distance: 11.00\n"));
    assert!(
        !boltz_a.contains("      max_distance: 14.00\n"),
        "weak 16.50 A restraint must be dropped, never clamped down to 14.00 A:\n{boltz_a}"
    );
    assert!(boltz_a.contains("properties:\n  - affinity:\n      binder: L\n"));
}

/// Independent physical forward model for synthetic CSP generation (`21.W3:P0-3`):
/// uses classical Johnson-Bovey (1958) two-loop magnetic dipole ring-current shielding
/// (`+/- 0.64 A` pi-cloud separation) + Buckingham (1960) linear electric-field (`Stark`)
/// polarization from ligand partial charges + Gaussian contact desolvation.
/// Intentionally shares ZERO helper calls or functional forms with `cofold_bridge`'s scorer.
fn independent_physical_perturbation_csp(
    res_name: &str,
    n_pos: Vec3,
    h_opt: Option<Vec3>,
    atoms: &BTreeMap<String, Vec3>,
    ligand_atoms: &[(&str, &str, Vec3)],
    ring_normal: Vec3,
    rng: &mut Rng,
) -> (f64, f64, f64) {
    // Benzene center (atoms 0..6) and perimeter-ordered pyrrole center ([4, 6, 7, 8, 5])
    let benz_c = (0..6).fold([0.0; 3], |acc, k| {
        v_add(acc, v_scale(ligand_atoms[k].2, 1.0 / 6.0))
    });
    let pyr_idx = [4usize, 6, 7, 8, 5];
    let pyr_c = pyr_idx.iter().fold([0.0; 3], |acc, &k| {
        v_add(acc, v_scale(ligand_atoms[k].2, 1.0 / 5.0))
    });

    // Johnson-Bovey two-loop point-dipole approximation at `+/- 0.64 A` above/below pi plane
    let loop_dipole_ppm = |pt: Vec3, center: Vec3, scale_a3: f64| -> f64 {
        let mut sum = 0.0;
        for sign in [-1.0_f64, 1.0_f64] {
            let lc = v_add(center, v_scale(ring_normal, sign * 0.64));
            let rv = v_sub(pt, lc);
            let r = v_dist(pt, lc).max(1.6);
            let cos_th = v_dot(rv, ring_normal) / r;
            sum += 0.5 * scale_a3 * (1.0 - 3.0 * cos_th * cos_th) / (r * r * r);
        }
        sum
    };

    // Ligand partial charges (e) on carboxamide & indole N-H for Buckingham E-field shift
    let partial_q: [(usize, f64); 5] = [
        (6, -0.32),  // N7 indole NH
        (9, 0.52),   // C10 carbonyl C
        (10, -0.55), // O11 carbonyl O
        (11, -0.38), // N12 amide N
        (12, 0.18),  // C13 methyl
    ];
    let e_field_along_bond = |pt: Vec3, bond_unit: Vec3| -> f64 {
        let mut e_proj = 0.0;
        for &(idx, q) in &partial_q {
            let lp = ligand_atoms[idx].2;
            let disp = v_sub(pt, lp);
            let r = v_dist(pt, lp).max(1.75);
            // Screened Coulomb field `q / (eps_r * r^2)` projected onto N-H bond vector
            let u_r = v_scale(disp, 1.0 / r);
            e_proj += (q / (4.0 * r * r)) * v_dot(u_r, bond_unit);
        }
        e_proj
    };

    let mut dh = rng.normal(0.0, 0.012);
    let mut dn = rng.normal(0.0, 0.080);

    if let Some(h_pos) = h_opt {
        // Physical Johnson-Bovey benzene (~5.5 A^3 -> ~1.0 ppm at 3.0 A above ring) & pyrrole (~3.5 A^3)
        dh += loop_dipole_ppm(h_pos, benz_c, 5.5) + loop_dipole_ppm(h_pos, pyr_c, 3.5);
        let u_nh = v_unit(v_sub(h_pos, n_pos));
        let ef = e_field_along_bond(h_pos, u_nh);
        dh += (12.5 * ef).clamp(-0.9, 1.2);
        dn += (28.0 * ef).clamp(-2.5, 3.2);
    }
    if res_name != "PRO" {
        dn += 0.55 * (loop_dipole_ppm(n_pos, benz_c, 5.5) + loop_dipole_ppm(n_pos, pyr_c, 3.5));
    }

    // Independent Gaussian steric/hydration shell perturbation around heavy-atom contact (`<= 5.2 A`)
    let d_min = atoms
        .values()
        .flat_map(|&ap| ligand_atoms.iter().map(move |&(_, _, lp)| v_dist(ap, lp)))
        .fold(f64::INFINITY, f64::min);
    let desolv = if d_min < 6.0 {
        let z: f64 = (d_min - 3.3_f64).max(0.0) / 1.45_f64;
        0.16 * (-0.5_f64 * z * z).exp()
    } else {
        0.0
    };
    dh += 0.65 * desolv;
    dn += 3.80 * desolv;

    let alpha_n = amide_alpha_n(res_name);
    let csp = dh.hypot(alpha_n * dn).max(0.001);
    (csp, dh, dn)
}

#[test]
fn synthetic_self_consistency_benchmark_and_ablations_across_1d3z_apo_and_1ubq_poses() {
    // Distinct experimental structures (`A7`): solution NMR `1D3Z` Model 1 as apo receptor vs X-ray `1UBQ` poses!
    let apo_1d3z_raw = std::fs::read_to_string(ubq_1d3z_nmr_path()).expect("read 1D3Z NMR PDB");
    let apo_1d3z_text: String = match apo_1d3z_raw.find("\nENDMDL") {
        Some(pos) => format!("{}\nENDMDL\n", &apo_1d3z_raw[..pos]),
        None => apo_1d3z_raw,
    };
    let apo_1d3z =
        StructureModel::parse_str(&apo_1d3z_text, Some("pdb"), false).expect("parse 1D3Z Model 1");
    let ubq_pdb_text = std::fs::read_to_string(ubq_fixture_path()).expect("read 1UBQ X-ray PDB");
    let ubq = StructureModel::parse_str(&ubq_pdb_text, Some("pdb"), false).expect("parse 1UBQ");
    assert_eq!(apo_1d3z.residues.len(), ubq.residues.len());

    // Verify genuine frame-invariant internal CA-CA distance RMSD (`dRMSD`) between 1D3Z (NMR) and 1UBQ (X-ray)
    let mut drmsd_sum = 0.0_f64;
    let mut n_pairs = 0usize;
    for i in 0..ubq.residues.len() {
        for j in (i + 1)..ubq.residues.len() {
            let d_nmr = v_dist(apo_1d3z.residues[i].ca, apo_1d3z.residues[j].ca);
            let d_xray = v_dist(ubq.residues[i].ca, ubq.residues[j].ca);
            let diff = d_nmr - d_xray;
            drmsd_sum += diff * diff;
            n_pairs += 1;
        }
    }
    let ca_drmsd_apo_vs_xray = (drmsd_sum / (n_pairs as f64)).sqrt();
    assert!(
        (0.25..=2.50).contains(&ca_drmsd_apo_vs_xray),
        "1D3Z vs 1UBQ must have realistic distinct backbone conformation, got CA dRMSD {ca_drmsd_apo_vs_xray:.3} A"
    );

    // Anchor native indole-amide ligand above the Ile44/His68/Leu8 hydrophobic patch
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
    let ring_norm = v_unit(v_cross(e1, e2));

    let native_lig_center = v_add(pocket_center, v_scale(outward, 2.3));
    let native_lig = build_indole_amide_heavy_atoms(native_lig_center, e1, e2);
    let native_cif = render_complex_mmcif_with_ligand(&ubq_pdb_text, &native_lig);

    // Synthesize CSPs from the INDEPENDENT Johnson-Bovey + Buckingham Stark model (NOT cofold_bridge!)
    let mut rng = Rng::new(20261118);
    let mut raw_csp_rows = Vec::new();
    for r in &ubq.residues {
        let (csp, dh, dn) = independent_physical_perturbation_csp(
            &r.res_name,
            r.n,
            r.h,
            &r.atoms,
            &native_lig,
            ring_norm,
            &mut rng,
        );
        raw_csp_rows.push((r.seq_id, r.res_name.clone(), csp, Some(dh), Some(dn)));
    }

    // Discrete Wuethrich NOE classes (`2.70 / 3.30 / 5.00 A`) instead of `true_dist + 0.85 A` oracle!
    let wuethrich_class = |true_d: f64, downgrade_one_bin: bool| -> Option<&'static str> {
        if true_d <= 2.7 {
            Some(if downgrade_one_bin {
                "medium"
            } else {
                "strong"
            })
        } else if true_d <= 3.3 {
            Some(if downgrade_one_bin { "weak" } else { "medium" })
        } else if true_d <= 5.0 {
            Some("weak")
        } else {
            None
        }
    };
    let mut contacts_under_48a: Vec<(f64, i32, &'static str, &'static str)> = Vec::new();
    for r in &ubq.residues {
        for &prot_atom in &["H", "CA", "CB"] {
            let p_opt = if prot_atom == "H" {
                r.h
            } else {
                r.atoms.get(prot_atom).copied()
            };
            if let Some(p_pos) = p_opt {
                for &(lig_aname, _, lig_pos) in &native_lig {
                    let d = v_dist(p_pos, lig_pos);
                    if d <= 4.75 {
                        contacts_under_48a.push((d, r.seq_id, prot_atom, lig_aname));
                    }
                }
            }
        }
    }
    contacts_under_48a.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut used_res = std::collections::BTreeSet::new();
    let mut noe_tsv = String::from("chain_1\tseq_id_1\tatom_1\tchain_2\tatom_2\tupper_a\n");
    for (d, sid, patom, latom) in contacts_under_48a {
        if used_res.len() >= 4 || !used_res.insert(sid) {
            continue;
        }
        let downgrade = used_res.len() == 2; // ~25% downgraded one Wuethrich bin
        if let Some(cls_tok) = wuethrich_class(d, downgrade) {
            noe_tsv.push_str(&format!("A\t{sid}\t{patom}\tL\t{latom}\t{cls_tok}\n"));
        }
    }
    let discrete_noes = parse_intermolecular_noe_tsv_str(&noe_tsv).expect("parse discrete NOEs");
    assert!(
        !discrete_noes.is_empty(),
        "expected >= 1 discrete Wuethrich NOE<=5.0 A in pocket"
    );
    for r in &discrete_noes {
        assert!(
            [2.70, 3.30, 5.00]
                .iter()
                .any(|&b| (r.upper_a - b).abs() < 1e-6),
            "NOE upper bound {} must be quantized to standard Wuethrich 2.7/3.3/5.0 A",
            r.upper_a
        );
    }

    let (lig_spec, lig_mol) =
        ligand_spec_from_octet_str(INDOLE_AMIDE_SMILES, Some("smi")).expect("parse ligand");
    let mut lig_map = BTreeMap::new();
    lig_map.insert("L".to_string(), lig_mol);
    let seq_a: String = apo_1d3z
        .residues
        .iter()
        .map(|r| char::from(r.aa1))
        .collect();

    // Build TWO restraint bundles using `1D3Z` (NMR) as apo structure:
    // (1) `bundle_csp_only`: ZERO NOEs, ZERO PREs (realistic CSP-only screening!)
    // (2) `bundle_csp_noe`: CSPs + discrete Wuethrich NOE classes (`scoring_only: true` for SMILES ligand)
    let base_input = CofoldRestraintInput {
        job_name: "ubq_synthetic_self_consistency_benchmark".to_string(),
        entities: vec![
            CofoldEntitySpec::Polymer {
                kind: CofoldPolymerKind::Protein,
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
        apo_receptor_model: Some(apo_1d3z.clone()),
        raw_csp_rows,
        raw_distance_restraints: Vec::new(),
        ..Default::default()
    };
    let scoring_extract_cfg = CofoldRestraintExtractConfig {
        scoring_only: true,
        ..CofoldRestraintExtractConfig::default()
    };
    let bundle_csp_only = extract_nmr_interface_restraints(&base_input, &scoring_extract_cfg)
        .expect("extract CSP-only bundle");
    let mut input_with_noe = base_input;
    input_with_noe.raw_distance_restraints = discrete_noes;
    let bundle_csp_noe = extract_nmr_interface_restraints(&input_with_noe, &scoring_extract_cfg)
        .expect("extract CSP+NOE bundle");
    assert!(
        bundle_csp_only.pocket_contacts.len() >= 3,
        "expected >= 3 surface CSP hits from independent model, got {}",
        bundle_csp_only.pocket_contacts.len()
    );

    // Construct 10 candidate poses (`1` near-native + `9` decoys across `4.0..=18.0 A` iRMSD)
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
    assert!(near_irmsd > 0.10 && near_irmsd < 1.50);
    poses.push(StructurePoseInput::from_coord_str(
        "pose_near_native",
        near_cif,
        Some("cif"),
    ));

    let decoy_displacements: [(f64, f64, f64, bool); 9] = [
        (4.3, 1.5, 1.0, false),
        (-5.2, 2.0, 1.2, false),
        (3.0, 5.8, 2.0, false),
        (-6.0, -4.5, 2.5, true),
        (7.5, -5.0, 3.0, true),
        (0.0, 0.0, 11.5, false),
        (-9.5, 6.5, 4.0, true),
        (10.5, -8.0, 4.5, true),
        (0.0, 0.0, 15.2, true),
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
        assert!((4.0..=18.0).contains(&irmsd));
        let p = StructurePoseInput::from_coord_str(
            format!("pose_decoy_{:02}", d_idx + 1),
            cif,
            Some("cif"),
        );
        decoy_poses_only.push(p.clone());
        poses.push(p);
    }

    // Verify conventional `DEFAULT_MIN_CSP_ROC_AUC` == 0.65 (Mann-Whitney enrichment guard)
    assert!((DEFAULT_MIN_CSP_ROC_AUC - 0.65).abs() < 1e-12);

    let cfg_burial = PoseScoreConfig {
        apo_receptor_coord_text: Some(apo_1d3z_text.clone()),
        apo_receptor_format_hint: Some("pdb".to_string()),
        reference_complex_pose: Some((native_cif.clone(), Some("cif".to_string()))),
        csp_scoring_mode: CspScoringMode::BurialAndContactOnly,
        ..PoseScoreConfig::default()
    };
    let cfg_forward = PoseScoreConfig {
        csp_scoring_mode: CspScoringMode::FullForwardShiftAndBurial,
        ..cfg_burial.clone()
    };

    // Ablation (a) + (b): CSP-only (`0` NOEs, `0` PREs) comparing `BurialAndContactOnly` (`A7` default)
    // against `FullForwardShiftAndBurial` across distinct `1D3Z` (NMR apo) vs `1UBQ` (X-ray pose) coordinates!
    let rep_csp_burial =
        score_cofold_poses(&poses, &bundle_csp_only, &cfg_burial).expect("score CSP-only burial");
    let rep_csp_forward = score_cofold_poses(
        &[poses[0].clone(), poses[1].clone(), poses[5].clone()],
        &bundle_csp_only,
        &cfg_forward,
    )
    .expect("score CSP-only forward");

    for p in &rep_csp_burial.poses {
        println!(
            "  [CSP-only burial] #{:2} {:16} total={:7.3} s_csp={:6.3} rho={:?} auc={:?} p={:?} ex_d={:.2} flags={:?}",
            p.rank, p.pose_id, p.total_score, p.s_csp, p.csp_spearman_rho, p.csp_contact_roc_auc, p.csp_contact_p_value, p.mean_pocket_excess_dist_a, p.violation_flags
        );
    }

    assert_eq!(
        rep_csp_burial.csp_scoring_mode,
        CspScoringMode::BurialAndContactOnly
    );
    assert_eq!(rep_csp_burial.verdict, PoseVerdict::RankedConclusive);
    assert_eq!(
        rep_csp_burial.top_pose_id.as_deref(),
        Some("pose_near_native"),
        "CSP-only burial mode must rank near-native pose #1 without any NOE/PRE help"
    );
    let top_csp_only = &rep_csp_burial.poses[0];
    let auc_csp_only = top_csp_only.csp_contact_roc_auc.expect("CSP-only ROC AUC");
    let pval_csp_only = top_csp_only
        .csp_contact_p_value
        .expect("CSP-only Mann-Whitney p");
    let rho_burial = top_csp_only.csp_spearman_rho.expect("burial Spearman rho");
    let rho_forward = rep_csp_forward
        .poses
        .iter()
        .find(|p| p.pose_id == "pose_near_native")
        .and_then(|p| p.csp_spearman_rho)
        .unwrap_or(0.0);
    let funnel_csp_only = rep_csp_burial
        .funnel_spearman_rho_vs_neg_irmsd
        .expect("CSP-only funnel rho");
    assert!(
        auc_csp_only >= DEFAULT_MIN_CSP_ROC_AUC,
        "CSP-only near-native ROC AUC ({auc_csp_only:.4}) must exceed DEFAULT_MIN_CSP_ROC_AUC (0.65)"
    );
    assert!(
        pval_csp_only < 0.05,
        "CSP-only contact enrichment Mann-Whitney p ({pval_csp_only:.4e}) must be < 0.05"
    );
    assert!(
        rho_burial > rho_forward,
        "A7 ablation: BurialAndContactOnly rho ({rho_burial:.4}) should exceed noisy cross-conformer FullForwardShiftAndBurial rho ({rho_forward:.4}) when 1D3Z apo != 1UBQ complex"
    );
    assert!(
        funnel_csp_only >= 0.65,
        "CSP-only 10-pose funnel Spearman rho ({funnel_csp_only:.4}) must be >= 0.65"
    );

    // Ablation (c): CSP + discrete Wuethrich NOE classes (`2.70 / 3.30 / 5.00 A`)
    let rep_csp_noe =
        score_cofold_poses(&poses, &bundle_csp_noe, &cfg_burial).expect("score CSP+discrete NOE");
    assert_eq!(rep_csp_noe.verdict, PoseVerdict::RankedConclusive);
    assert_eq!(rep_csp_noe.top_pose_id.as_deref(), Some("pose_near_native"));
    let top_noe = &rep_csp_noe.poses[0];
    assert!(top_noe.passes_nmr_guards);
    assert_eq!(top_noe.exclusion_list_sha256.len(), 64);
    let funnel_csp_noe = rep_csp_noe
        .funnel_spearman_rho_vs_neg_irmsd
        .expect("CSP+NOE funnel rho");
    assert!(
        funnel_csp_noe >= 0.75,
        "CSP + discrete Wuethrich NOE funnel Spearman rho ({funnel_csp_noe:.4}) must be >= 0.75"
    );

    // Decoy-only ablation (b): CSP-only (`0` NOEs, `0` PREs) rejects `3/9` solvent-detached decoys
    // (`LowCspContactEnrichment`), while `6/9` pocket-adjacent decoys (`4.7..10.2 A` iRMSD) remain
    // admissible because coarse `8 A` CSP patch proximity alone provides no hard atom-pair exclusion!
    let decoy_csp_only_report =
        score_cofold_poses(&decoy_poses_only, &bundle_csp_only, &cfg_burial)
            .expect("score 9 decoys CSP-only");
    assert_eq!(decoy_csp_only_report.verdict, PoseVerdict::RankedConclusive);
    assert_eq!(decoy_csp_only_report.n_admissible_poses, 6);
    let n_detached_rejected = decoy_csp_only_report
        .poses
        .iter()
        .filter(|p| {
            p.violation_flags
                .contains(&PoseViolationFlag::LowCspContactEnrichment)
        })
        .count();
    assert_eq!(n_detached_rejected, 3);

    // Decoy-only ablation (c): CSP + discrete Wuethrich NOE classes -> `PoseVerdict::InconclusiveOrAllViolated` (`0/9` admissible)!
    let decoy_report = score_cofold_poses(&decoy_poses_only, &bundle_csp_noe, &cfg_burial)
        .expect("score 9 decoys with discrete NOEs");
    assert_eq!(decoy_report.verdict, PoseVerdict::InconclusiveOrAllViolated);
    assert_eq!(decoy_report.top_pose_id, None);
    assert_eq!(decoy_report.n_admissible_poses, 0);

    println!(
        "[S4 synthetic self-consistency ablations] 1D3Z-vs-1UBQ CA_dRMSD={ca_drmsd_apo_vs_xray:.3} A | \
         (a) rho_burial={rho_burial:.4} vs rho_forward={rho_forward:.4} | \
         (b) CSP-only top={} auc={auc_csp_only:.4} funnel_rho={funnel_csp_only:.4} decoy_only_admissible={}/9 | \
         (c) CSP+discrete-NOE top={} funnel_rho={funnel_csp_noe:.4} decoy_only_admissible={}/9",
        rep_csp_burial.top_pose_id.as_deref().unwrap_or("NONE"),
        decoy_csp_only_report.n_admissible_poses,
        rep_csp_noe.top_pose_id.as_deref().unwrap_or("NONE"),
        decoy_report.n_admissible_poses,
    );
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

    // When `rdc_free_fraction == 0.0` (and no preset free rows), working `rdc_q_work` is NEVER aliased as `rdc_q_free`!
    let no_free_rep = score_cofold_poses(
        &poses[1..2],
        &bundle,
        &PoseScoreConfig {
            rdc_free_fraction: 0.0,
            ..PoseScoreConfig::default()
        },
    )
    .expect("score with rdc_free_fraction = 0.0");
    assert!(no_free_rep.poses[0].rdc_q_work.is_some());
    assert_eq!(no_free_rep.poses[0].rdc_q_free, None);
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

#[test]
fn p0_4_solomon_bloembergen_pre_constant_and_mandatory_tau_c() {
    // 1. Exact SI dipole-dipole Solomon constant for S = 1/2 (`~1.2339e16 A^6 s^-2`, NOT `1.23e7`!)
    let k_s12 = solomon_pre_k_a6_s2(0.5);
    assert!(
        (k_s12 - 1.2339e16).abs() < 0.002e16,
        "expected K_Solomon(S=1/2) ~ 1.234e16 A^6 s^-2, got {k_s12:.6e}"
    );

    // 2. At tau_c = 10 ns, 600 MHz, Gamma_2 = 20 s^-1 -> paramagnetic-to-proton distance r ~ 17.0 A
    let k_eff = solomon_pre_k_eff_a6_s1(10.0, 600.0, 0.5).expect("valid Solomon params");
    let r_20 = solomon_distance_from_gamma2_a(20.0, k_eff);
    let r_18 = solomon_distance_from_gamma2_a(18.0, k_eff);
    let r_22 = solomon_distance_from_gamma2_a(22.0, k_eff);
    assert!(
        (r_20 - 17.02).abs() < 0.15 && r_22 < r_20 && r_20 < r_18,
        "Gamma_2 = 20 +/- 2 s^-1 at tau_c = 10 ns must give r ~ 17.0 A, got [{r_22:.2}, {r_20:.2}, {r_18:.2}] A"
    );

    // 3. Configurable MTSL tether pad (6.0 A default vs 0.0 A rigid paramagnetic metal)
    let (_, hi_mtsl) =
        solomon_gamma2_to_distance_bounds(20.0, 1.0, k_eff, DEFAULT_MTSL_TETHER_PAD_A, 1.5);
    let (_, hi_metal) = solomon_gamma2_to_distance_bounds(20.0, 1.0, k_eff, 0.0, 1.5);
    assert!(
        ((hi_mtsl - hi_metal) - 6.0).abs() < 1e-9,
        "MTSL 6.0 A tether pad must widen upper distance bound by +6.0 A ({hi_mtsl:.2} vs {hi_metal:.2})"
    );

    // 4. Missing `tau_c_ns` MUST fail closed (`Err`) on `Gamma2Rate` PRE rows!
    let pre_missing_tau = "\
# anchor: M 1 ZN
# mode: gamma2
A\t44\tH\t20.0\t1.0
";
    let err = parse_intermolecular_pre_tsv_str(pre_missing_tau, None, "A", 1.5).unwrap_err();
    assert_eq!(err.kind, NmrErrorKind::InvalidInput);
    assert!(
        err.message.contains("tau_c"),
        "expected missing tau_c error message, got: {}",
        err.message
    );

    // Supplying `# tau_c_ns: 10.0` + `# tether_pad_a: 0.0` (rigid metal) succeeds and gives ~19.1 A upper bound
    let pre_with_tau = "\
# anchor: M 1 ZN
# tau_c_ns: 10.0
# proton_mhz: 600.0
# tether_pad_a: 0.0
# mode: gamma2
A\t44\tH\t20.0\t1.0
";
    let parsed_metal =
        parse_intermolecular_pre_tsv_str(pre_with_tau, None, "A", 1.5).expect("parse rigid PRE");
    assert_eq!(parsed_metal.len(), 1);
    assert!(
        (parsed_metal[0].upper_a - 19.01).abs() < 0.35,
        "expected rigid-metal Gamma_2=20+/-2 upper_a ~ 19.0 A, got {:.2}",
        parsed_metal[0].upper_a
    );

    // Supplying `PreSolomonParams` with default 6.0 A MTSL tether gives ~25.0 A upper bound
    let mtsl_params = PreSolomonParams {
        tau_c_ns: 10.0,
        proton_mhz: 600.0,
        spin_s: 0.5,
        tether_pad_a: 6.0,
        upper_margin_a: 1.5,
    };
    let parsed_mtsl = parse_intermolecular_pre_tsv_str_with_params(
        pre_missing_tau,
        None,
        "A",
        Some(&mtsl_params),
    )
    .expect("parse MTSL PRE with explicit params");
    assert!(((parsed_mtsl[0].upper_a - parsed_metal[0].upper_a) - 6.0).abs() < 1e-6);
}

#[test]
fn p1_7_drop_weak_restraints_above_ceiling_never_clamp() {
    let ubq = StructureModel::parse_file(ubq_fixture_path(), false).expect("parse 1UBQ");
    let seq_a: String = ubq.residues.iter().map(|r| char::from(r.aa1)).collect();
    let pre_tsv = "\
# mode: bounds
A\t8\tH\tM\t1\tZN\t1.80\t3.20\t0.40
A\t44\tH\tM\t1\tZN\t1.80\t11.50\t0.80
A\t46\tH\tM\t1\tZN\t1.80\t16.50\t1.00
A\t68\tH\tM\t1\tZN\t1.80\t22.50\t1.50
";
    let restraints = parse_intermolecular_pre_tsv_str(pre_tsv, None, "A", 1.5).unwrap();
    let input = CofoldRestraintInput {
        job_name: "drop_weak_test".to_string(),
        entities: vec![
            CofoldEntitySpec::Polymer {
                kind: CofoldPolymerKind::Protein,
                chain_id: "A".to_string(),
                sequence: seq_a,
            },
            CofoldEntitySpec::Ligand {
                chain_id: "M".to_string(),
                spec: CofoldLigandSpec::CcdCodes(vec!["ZN".to_string()]),
            },
        ],
        receptor_chain_id: Some("A".to_string()),
        binder_chain_id: Some("M".to_string()),
        apo_receptor_model: Some(ubq),
        raw_distance_restraints: restraints,
        ..Default::default()
    };

    // At default `max_contact_distance_a = 14.0 A`, BOTH 16.50 A and 22.50 A are dropped (NEVER clamped to 14.0!)
    let bundle_14 =
        extract_nmr_interface_restraints(&input, &CofoldRestraintExtractConfig::default()).unwrap();
    assert_eq!(bundle_14.dropped_weak_restraints_count, 2);
    assert_eq!(
        bundle_14.distance_restraints[0].exported_contact_max_distance_a,
        Some(4.50)
    );
    assert_eq!(
        bundle_14.distance_restraints[1].exported_contact_max_distance_a,
        Some(11.50)
    );
    assert_eq!(
        bundle_14.distance_restraints[2].exported_contact_max_distance_a,
        None
    );
    assert_eq!(
        bundle_14.distance_restraints[3].exported_contact_max_distance_a,
        None
    );
    let yaml_14 = export_boltz2_yaml(&bundle_14, &Boltz2ExportConfig::default()).unwrap();
    assert!(!yaml_14.contains("14.00"));
    assert!(!yaml_14.contains("16.50"));

    // At `max_contact_distance_a = 20.0 A` (Boltz-2 hard limit), 16.50 A is kept while 22.50 A is still dropped!
    let cfg_20 = CofoldRestraintExtractConfig {
        max_contact_distance_a: BOLTZ2_MAX_DISTANCE_A,
        ..CofoldRestraintExtractConfig::default()
    };
    let bundle_20 = extract_nmr_interface_restraints(&input, &cfg_20).unwrap();
    assert_eq!(bundle_20.dropped_weak_restraints_count, 1);
    assert_eq!(
        bundle_20.distance_restraints[2].exported_contact_max_distance_a,
        Some(16.50)
    );
    assert_eq!(
        bundle_20.distance_restraints[3].exported_contact_max_distance_a,
        None
    );
    let yaml_20 = export_boltz2_yaml(&bundle_20, &Boltz2ExportConfig::default()).unwrap();
    assert!(yaml_20.contains("      max_distance: 16.50\n"));
    assert!(!yaml_20.contains("20.00"));
}

#[test]
fn p1_8_fail_closed_without_apo_and_distal_exposed_allosteric_cluster_filter() {
    let ubq = StructureModel::parse_file(ubq_fixture_path(), false).expect("parse 1UBQ");
    let sasa = compute_shrake_rupley_rsasa(&ubq, &[]);
    let sasa_by_sid: BTreeMap<i32, f64> = sasa.iter().map(|s| (s.seq_id, s.rsasa)).collect();
    let ca_by_sid: BTreeMap<i32, Vec3> = ubq.residues.iter().map(|r| (r.seq_id, r.ca)).collect();

    // Pick a genuinely surface-exposed (`rSASA >= 0.20`) distal residue (`> 15 A` from Ile44/His68/Leu8/Arg42/Gln49)
    let pocket_cluster = [8, 9, 42, 44, 46, 49, 68];
    let distal_sid = ubq
        .residues
        .iter()
        .map(|r| r.seq_id)
        .find(|&sid| {
            sasa_by_sid[&sid] >= 0.25
                && pocket_cluster
                    .iter()
                    .all(|&ps| v_dist(ca_by_sid[&sid], ca_by_sid[&ps]) > 16.0)
        })
        .expect("find distal surface-exposed Ubiquitin residue");

    let mut csp_tsv = String::from("seq_id\tres_name\tcsp_ppm\n");
    for r in &ubq.residues {
        let v = if pocket_cluster.contains(&r.seq_id) || r.seq_id == distal_sid {
            0.380
        } else {
            0.014 + 0.002 * ((r.seq_id % 5) as f64)
        };
        csp_tsv.push_str(&format!("{}\t{}\t{v:.4}\n", r.seq_id, r.res_name));
    }
    let (raw_csp, _) = parse_csp_restraint_str(&csp_tsv).unwrap();
    let seq_a: String = ubq.residues.iter().map(|r| char::from(r.aa1)).collect();

    // 1. Without apo structure, `extract_nmr_interface_restraints` MUST fail closed (`Err`) by default!
    let no_apo_input = CofoldRestraintInput {
        job_name: "no_apo_job".to_string(),
        entities: vec![
            CofoldEntitySpec::Polymer {
                kind: CofoldPolymerKind::Protein,
                chain_id: "A".to_string(),
                sequence: seq_a.clone(),
            },
            CofoldEntitySpec::Ligand {
                chain_id: "M".to_string(),
                spec: CofoldLigandSpec::CcdCodes(vec!["ZN".to_string()]),
            },
        ],
        receptor_chain_id: Some("A".to_string()),
        binder_chain_id: Some("M".to_string()),
        apo_receptor_model: None,
        raw_csp_rows: raw_csp.clone(),
        ..Default::default()
    };
    let err =
        extract_nmr_interface_restraints(&no_apo_input, &CofoldRestraintExtractConfig::default())
            .unwrap_err();
    assert_eq!(err.kind, NmrErrorKind::InvalidInput);
    assert!(err.message.contains("allow-unfiltered-no-apo"));

    // Explicit `allow_unfiltered_no_apo: true` opt-in succeeds and emits explicit diagnostic warning
    let opt_in_bundle = extract_nmr_interface_restraints(
        &no_apo_input,
        &CofoldRestraintExtractConfig {
            allow_unfiltered_no_apo: true,
            ..CofoldRestraintExtractConfig::default()
        },
    )
    .expect("opt-in unfiltered no-apo");
    assert!(!opt_in_bundle.pocket_contacts.is_empty());
    assert!(opt_in_bundle
        .diagnostics
        .iter()
        .any(|d| d.contains("WARNING") && d.contains("apo")));

    // 2. With apo structure, the distal surface-exposed (`rSASA >= 0.25`) 2-sigma hit is demoted to
    // `DistalSurfaceOutlier`, excluded from `pocket.contacts`, logged with WARNING, and `force: false`!
    let mut with_apo = no_apo_input;
    with_apo.apo_receptor_model = Some(ubq);
    let bundle =
        extract_nmr_interface_restraints(&with_apo, &CofoldRestraintExtractConfig::default())
            .expect("extract with apo cluster filter");

    assert_eq!(bundle.distal_surface_csp_hits.len(), 1);
    assert_eq!(bundle.distal_surface_csp_hits[0].seq_id, distal_sid);
    assert_eq!(
        bundle.distal_surface_csp_hits[0].classification,
        CspPerturbationClass::DistalSurfaceOutlier
    );
    assert!(
        bundle
            .pocket_contacts
            .iter()
            .all(|p| p.seq_id != distal_sid),
        "isolated distal surface hit {distal_sid} must not pollute Boltz-2 pocket contacts"
    );
    assert!(bundle
        .diagnostics
        .iter()
        .any(|d| d.contains("WARNING") && d.contains("surface-exposed")));
    let yaml = export_boltz2_yaml(&bundle, &Boltz2ExportConfig::default()).unwrap();
    assert!(yaml.contains("      force: false\n"));
    assert!(!yaml.contains(&format!("        - [A, {distal_sid}]\n")));
}

#[test]
fn p1_9_sequence_aligned_indexing_gaps_offset_insertion_codes_and_strict_pdb() {
    let ubq_pdb_text = std::fs::read_to_string(ubq_fixture_path()).expect("read 1UBQ");
    let ubq = StructureModel::parse_str(&ubq_pdb_text, Some("pdb"), false).expect("parse 1UBQ");
    let wt_seq: String = ubq.residues.iter().map(|r| char::from(r.aa1)).collect();

    // 1. Construct a FASTA with a 10-aa N-terminal His-tag (`MGSSHHHHHH` + 76-aa Ubiquitin = 86 aa),
    // and an apo PDB with author offset `+100` (`101..176`) AND missing loop density at `20..=24` (`120..=124`)!
    let n_tag = "MGSSHHHHHH";
    let full_fasta_seq = format!("{n_tag}{wt_seq}");
    assert_eq!(full_fasta_seq.len(), 86);

    let mut gapped_offset_apo_pdb = String::new();
    for line in ubq_pdb_text.lines() {
        if !line.starts_with("ATOM  ") || line.len() < 54 || &line[17..20] == "HOH" {
            continue;
        }
        let sid: i32 = line[22..26].trim().parse().unwrap();
        if (20..=24).contains(&sid) {
            continue; // missing electron density gap!
        }
        let new_sid = sid + 100;
        gapped_offset_apo_pdb.push_str(&format!("{}{:4}{}\n", &line[..22], new_sid, &line[26..]));
    }
    let apo_gapped = StructureModel::parse_str(&gapped_offset_apo_pdb, Some("pdb"), false)
        .expect("parse gapped +100 apo PDB");

    // Verify gap preservation even WITHOUT FASTA (`infer_target_from_observed` inserts `'X'` for 120..=124!)
    let no_fasta_in = CofoldRestraintInput {
        job_name: "gap_infer_no_fasta".to_string(),
        apo_receptor_model: Some(apo_gapped.clone()),
        receptor_chain_id: Some("A".to_string()),
        binder_chain_id: Some("M".to_string()),
        entities: vec![CofoldEntitySpec::Ligand {
            chain_id: "M".to_string(),
            spec: CofoldLigandSpec::CcdCodes(vec!["ZN".to_string()]),
        }],
        ..Default::default()
    };
    let inferred_bundle =
        extract_nmr_interface_restraints(&no_fasta_in, &CofoldRestraintExtractConfig::default())
            .expect("infer gapped sequence");
    let inferred_seq = match &inferred_bundle.entities[0] {
        CofoldEntitySpec::Polymer { sequence, .. } => sequence.clone(),
        _ => panic!("expected inferred receptor polymer"),
    };
    assert_eq!(
        inferred_seq.len(),
        76,
        "missing residues 120..=124 must be preserved as 5 'X' placeholders (76 total, NOT compacted to 71)"
    );
    assert_eq!(&inferred_seq[19..24], "XXXXX");

    // 2. With 86-aa FASTA + author-numbered (`+100`) CSPs (`108, 142, 144, 146, 149, 168`),
    // Ile44 (`author 144`) MUST map to 1-based FASTA index `10 + 44 = 54` (preserving both tag + gap)!
    let mut csp_rows = Vec::new();
    for r in &apo_gapped.residues {
        let v = if [108, 142, 144, 146, 149, 168].contains(&r.seq_id) {
            0.350
        } else {
            0.015
        };
        csp_rows.push((r.seq_id, r.res_name.clone(), v, None, None));
    }
    let aligned_in = CofoldRestraintInput {
        job_name: "tag_gap_offset_job".to_string(),
        entities: vec![
            CofoldEntitySpec::Polymer {
                kind: CofoldPolymerKind::Protein,
                chain_id: "A".to_string(),
                sequence: full_fasta_seq.clone(),
            },
            CofoldEntitySpec::Ligand {
                chain_id: "M".to_string(),
                spec: CofoldLigandSpec::CcdCodes(vec!["ZN".to_string()]),
            },
        ],
        receptor_chain_id: Some("A".to_string()),
        binder_chain_id: Some("M".to_string()),
        apo_receptor_model: Some(apo_gapped.clone()),
        raw_csp_rows: csp_rows,
        raw_distance_restraints: parse_intermolecular_noe_tsv_str(
            "chain_1\tseq_id_1\tatom_1\tchain_2\tatom_2\tupper_a\nA\t144\tH\tM\tZN\t5.00\n",
        )
        .unwrap(),
        ..Default::default()
    };
    let bundle = extract_nmr_interface_restraints(
        &aligned_in,
        &CofoldRestraintExtractConfig {
            numbering: RestraintNumbering::ApoAuthor,
            ..CofoldRestraintExtractConfig::default()
        },
    )
    .expect("align +100 gapped apo to 86-aa FASTA");

    let ala146_hit = bundle
        .pocket_contacts
        .iter()
        .find(|p| p.seq_id == 146)
        .expect("Ala146 surface pocket hit");
    assert_eq!(
        ala146_hit.one_based_seq_idx, 56,
        "author Ala146 after 10-aa tag and 5-aa missing loop must map to FASTA 1-based idx 56"
    );
    let ile144_core = bundle
        .allosteric_core_perturbations
        .iter()
        .find(|p| p.seq_id == 144)
        .expect("Ile144 buried core hit");
    assert_eq!(ile144_core.one_based_seq_idx, 54);
    let yaml = export_boltz2_yaml(&bundle, &Boltz2ExportConfig::default()).unwrap();
    assert!(yaml.contains("        - [A, 56]\n"));
    assert!(yaml.contains("      token1: [A, 54]\n"));

    // 3. `score_cofold_poses` re-indexes AF3/Boltz-2 `1..N` (`11..86` for Ubiquitin domain) poses
    // back to experimental author `seq_id`s (`101..176`) so NOE `A:144:H` and pocket hits evaluate accurately!
    let ile44_h = ubq
        .residues
        .iter()
        .find(|r| r.seq_id == 44)
        .and_then(|r| r.h)
        .unwrap();
    let make_af3_numbered_pose = |zn_pos: Vec3| -> String {
        let mut p = String::new();
        for line in ubq_pdb_text.lines() {
            if !line.starts_with("ATOM  ") || line.len() < 54 || &line[17..20] == "HOH" {
                continue;
            }
            let sid: i32 = line[22..26].trim().parse().unwrap();
            let af3_idx = sid + 10; // 11..86 in 86-aa FASTA numbering!
            p.push_str(&format!("{}{:4}{}\n", &line[..22], af3_idx, &line[26..]));
        }
        p.push_str(&format!(
            "HETATM 9999 ZN    ZN M   1    {:8.3}{:8.3}{:8.3}  1.00 90.00          ZN\n",
            zn_pos[0], zn_pos[1], zn_pos[2]
        ));
        p
    };
    let near_pose_pdb = make_af3_numbered_pose(v_add(ile44_h, [2.2, 0.4, 0.3]));
    let far_pose_pdb = make_af3_numbered_pose(v_add(ile44_h, [18.0, -12.0, 9.0]));
    let poses = vec![
        StructurePoseInput::from_coord_str("af3_far", far_pose_pdb, Some("pdb")),
        StructurePoseInput::from_coord_str("af3_near", near_pose_pdb, Some("pdb")),
    ];
    let score_rep = score_cofold_poses(
        &poses,
        &bundle,
        &PoseScoreConfig {
            apo_receptor_coord_text: Some(gapped_offset_apo_pdb),
            apo_receptor_format_hint: Some("pdb".to_string()),
            ..PoseScoreConfig::default()
        },
    )
    .expect("score 1..N AF3 poses against +100 author restraints");
    assert_eq!(score_rep.top_pose_id.as_deref(), Some("af3_near"));
    assert_eq!(score_rep.poses[0].n_violated_distance_restraints, 0);
    assert_eq!(score_rep.poses[1].n_violated_distance_restraints, 1);

    // 4. Strict PDB `seq_id` parsing (`XXXX` -> `Err`, NEVER `unwrap_or(1)`) & insertion code (`100A`) preservation!
    let corrupt_pdb =
        "ATOM      1  N   ALA AXXXX       1.000   2.000   3.000  1.00 90.00           N\n";
    let bad_pose = StructurePoseInput::from_coord_str("bad_seq", corrupt_pdb, Some("pdb"));
    let empty_bundle = extract_nmr_interface_restraints(
        &CofoldRestraintInput::default(),
        &CofoldRestraintExtractConfig::default(),
    )
    .unwrap();
    let parse_err =
        score_cofold_poses(&[bad_pose], &empty_bundle, &PoseScoreConfig::default()).unwrap_err();
    assert_eq!(parse_err.kind, NmrErrorKind::ParseError);

    // Insertion codes `100` vs `100A` kept distinct in `compute_interface_rmsd_angstrom`
    let ref_ins_pdb = "\
ATOM      1  N   ALA A 100       0.000   0.000   0.000  1.00 90.00           N
ATOM      2  CA  ALA A 100       1.450   0.000   0.000  1.00 90.00           C
ATOM      3  C   ALA A 100       2.000   1.400   0.000  1.00 90.00           C
ATOM      4  O   ALA A 100       1.300   2.400   0.000  1.00 90.00           O
ATOM      5  N   GLY A 100A      3.300   1.500   0.000  1.00 90.00           N
ATOM      6  CA  GLY A 100A      3.950   2.800   0.000  1.00 90.00           C
ATOM      7  C   GLY A 100A      5.450   2.700   0.000  1.00 90.00           C
ATOM      8  O   GLY A 100A      6.050   1.650   0.000  1.00 90.00           O
HETATM    9 ZN    ZN M   1       2.500   1.500   3.200  1.00 90.00          ZN
";
    // Displace `100A` by 6.0 A in candidate: if `100A` were collapsed into `100`, `cand_map.or_insert`
    // would ignore `100A` completely and falsely report `0.0 A` iRMSD!
    let cand_ins_pdb = "\
ATOM      1  N   ALA A 100       0.000   0.000   0.000  1.00 90.00           N
ATOM      2  CA  ALA A 100       1.450   0.000   0.000  1.00 90.00           C
ATOM      3  C   ALA A 100       2.000   1.400   0.000  1.00 90.00           C
ATOM      4  O   ALA A 100       1.300   2.400   0.000  1.00 90.00           O
ATOM      5  N   GLY A 100A      3.300   1.500   6.000  1.00 90.00           N
ATOM      6  CA  GLY A 100A      3.950   2.800   6.000  1.00 90.00           C
ATOM      7  C   GLY A 100A      5.450   2.700   6.000  1.00 90.00           C
ATOM      8  O   GLY A 100A      6.050   1.650   6.000  1.00 90.00           O
HETATM    9 ZN    ZN M   1       2.500   1.500   3.200  1.00 90.00          ZN
";
    let irmsd_ins =
        compute_interface_rmsd_angstrom(cand_ins_pdb, Some("pdb"), ref_ins_pdb, Some("pdb"), "A")
            .expect("insertion-code iRMSD");
    assert!(
        irmsd_ins > 1.5,
        "insertion-code residue 100A displacement must change pocket superposition iRMSD, got {irmsd_ins:.4} A"
    );
}

#[test]
fn p1_10_11_p2_schema_hygiene_smiles_bonded_yaml_af3_chains_noncanonical_aa_five_ring() {
    let ubq = StructureModel::parse_file(ubq_fixture_path(), false).expect("parse 1UBQ");
    let seq_a: String = ubq.residues.iter().map(|r| char::from(r.aa1)).collect();
    let (smi_spec, smi_mol) = ligand_spec_from_octet_str(INDOLE_AMIDE_SMILES, Some("smi")).unwrap();
    let mut lig_map = BTreeMap::new();
    lig_map.insert("L".to_string(), smi_mol);

    // 1. Refuse atom-level contacts on SMILES ligands during export (`scoring_only == false`)
    let smi_atom_noe = parse_intermolecular_noe_tsv_str(
        "chain_1\tseq_id_1\tatom_1\tchain_2\tatom_2\tupper_a\nA\t44\tH\tL\tO11\t4.50\n",
    )
    .unwrap();
    let smi_contact_in = CofoldRestraintInput {
        job_name: "smi_atom_reject".to_string(),
        entities: vec![
            CofoldEntitySpec::Polymer {
                kind: CofoldPolymerKind::Protein,
                chain_id: "A".to_string(),
                sequence: seq_a.clone(),
            },
            CofoldEntitySpec::Ligand {
                chain_id: "L".to_string(),
                spec: smi_spec.clone(),
            },
        ],
        ligand_molecules: lig_map,
        receptor_chain_id: Some("A".to_string()),
        binder_chain_id: Some("L".to_string()),
        apo_receptor_model: Some(ubq.clone()),
        raw_distance_restraints: smi_atom_noe,
        ..Default::default()
    };
    assert!(
        extract_nmr_interface_restraints(&smi_contact_in, &CofoldRestraintExtractConfig::default())
            .is_err(),
        "atom-level contact on SMILES ligand must fail closed on export"
    );

    // 2. Reject `bondedAtomPairs` referencing SMILES ligand or out-of-range polymer index
    let mut bad_bonded = smi_contact_in.clone();
    bad_bonded.raw_distance_restraints.clear();
    bad_bonded.bonded_atom_pairs = vec![BondedAtomPairSpec {
        atom1: BondedAtomRef {
            chain_id: "A".to_string(),
            one_based_res_idx: 44,
            atom_name: "SG".to_string(),
        },
        atom2: BondedAtomRef {
            chain_id: "L".to_string(),
            one_based_res_idx: 1,
            atom_name: "C1".to_string(),
        },
    }];
    assert!(
        extract_nmr_interface_restraints(&bad_bonded, &CofoldRestraintExtractConfig::default())
            .is_err(),
        "bondedAtomPairs against SMILES ligand must fail closed"
    );

    // 3. Reject multi-CCD ligand in Boltz-2 YAML (`ccd` accepts single string code only)
    let multi_ccd_in = CofoldRestraintInput {
        job_name: "multi_ccd".to_string(),
        entities: vec![
            CofoldEntitySpec::Polymer {
                kind: CofoldPolymerKind::Protein,
                chain_id: "A".to_string(),
                sequence: seq_a.clone(),
            },
            CofoldEntitySpec::Ligand {
                chain_id: "M".to_string(),
                spec: CofoldLigandSpec::CcdCodes(vec!["MG".to_string(), "ZN".to_string()]),
            },
        ],
        ..Default::default()
    };
    let multi_ccd_bundle =
        extract_nmr_interface_restraints(&multi_ccd_in, &CofoldRestraintExtractConfig::default())
            .unwrap();
    assert!(
        export_af3_json(&multi_ccd_bundle, &Af3ExportConfig::default()).is_ok(),
        "AF3 supports ccdCodes array"
    );
    assert!(
        export_boltz2_yaml(&multi_ccd_bundle, &Boltz2ExportConfig::default()).is_err(),
        "Boltz-2 YAML must reject multi-CCD array on single ligand entity"
    );

    // 4. Range-check `pocket.max_distance` to `[4.0, 20.0] A`
    for bad_d in [2.5, 3.99, 20.01, 35.0] {
        assert!(
            export_boltz2_yaml(
                &multi_ccd_bundle,
                &Boltz2ExportConfig {
                    pocket_max_distance_override: Some(bad_d),
                    ..Boltz2ExportConfig::default()
                }
            )
            .is_err(),
            "Boltz-2 pocket.max_distance={bad_d} outside [{BOLTZ2_MIN_DISTANCE_A}, {BOLTZ2_MAX_DISTANCE_A}] must fail"
        );
    }

    // 5. Enforce uppercase AF3 chain IDs (`"a"` rejected) & quote numeric YAML chain IDs (`"1"` -> `'1'`)
    let mut lower_af3 = multi_ccd_bundle.clone();
    lower_af3.entities = vec![CofoldEntitySpec::Polymer {
        kind: CofoldPolymerKind::Protein,
        chain_id: "a".to_string(),
        sequence: seq_a.clone(),
    }];
    assert!(
        export_af3_json(&lower_af3, &Af3ExportConfig::default()).is_err(),
        "lowercase AF3 chain ID must be rejected"
    );

    let num_chain_in = CofoldRestraintInput {
        job_name: "numeric_chain".to_string(),
        entities: vec![
            CofoldEntitySpec::Polymer {
                kind: CofoldPolymerKind::Protein,
                chain_id: "1".to_string(),
                sequence: seq_a,
            },
            CofoldEntitySpec::Ligand {
                chain_id: "2".to_string(),
                spec: CofoldLigandSpec::CcdCodes(vec!["ZN".to_string()]),
            },
        ],
        receptor_chain_id: Some("1".to_string()),
        binder_chain_id: Some("2".to_string()),
        raw_distance_restraints: parse_intermolecular_noe_tsv_str(
            "chain_1\tseq_id_1\tatom_1\tchain_2\tatom_2\tupper_a\n1\t8\tH\t2\tZN\t6.00\n",
        )
        .unwrap(),
        ..Default::default()
    };
    let num_bundle =
        extract_nmr_interface_restraints(&num_chain_in, &CofoldRestraintExtractConfig::default())
            .unwrap();
    let num_yaml = export_boltz2_yaml(&num_bundle, &Boltz2ExportConfig::default()).unwrap();
    assert!(
        num_yaml.contains("id: '1'\n") && num_yaml.contains("id: '2'\n"),
        "numeric YAML chain IDs must be single-quoted so YAML parsers keep string type:\n{num_yaml}"
    );
    assert!(
        num_yaml.contains("token1: ['1', 8]\n") && num_yaml.contains("token2: ['2', ZN]\n"),
        "numeric YAML chain IDs in contact tokens must also be quoted:\n{num_yaml}"
    );

    // 6. Reject non-canonical `B/J/O/U/Z` in standard 20-aa protein sequences
    for bad_aa in ['B', 'J', 'O', 'U', 'Z'] {
        let fasta = format!(">A|protein\nMQIFVK{bad_aa}TLTG\n");
        assert!(
            parse_cofold_fasta_str(&fasta).is_err(),
            "non-canonical protein code '{bad_aa}' must be rejected"
        );
    }
    assert!(
        parse_cofold_fasta_str(">A|protein\nMQIFVKXXTLTG\n").is_ok(),
        "'X' missing-residue gap placeholder remains valid"
    );

    // 7. Verify non-self-intersecting pyrrole vertex order `[C5, N7, C8, C9, C6]` (`[4, 6, 7, 8, 5]`)
    // AND distinct 5-ring Haigh-Mallion intensity factors (`Trp` 1-N `~0.90` vs `His` 2-N `~0.43`)!
    let lig = build_indole_amide_heavy_atoms([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    let pyr_idx = [4usize, 6, 7, 8, 5];
    let pyr_verts: Vec<Vec3> = pyr_idx.iter().map(|&i| lig[i].2).collect();
    // Check strictly positive consistent signed triangle fans around polygon centroid (no bowtie crossing!)
    let pc = pyr_verts
        .iter()
        .fold([0.0; 3], |a, &p| v_add(a, v_scale(p, 0.2)));
    for k in 0..5 {
        let a = v_sub(pyr_verts[k], pc);
        let b = v_sub(pyr_verts[(k + 1) % 5], pc);
        let cz = v_cross(a, b)[2];
        assert!(
            cz > 0.20,
            "pyrrole edge {k}->{} signed area z={cz:.4} must be > 0 (non-self-intersecting CCW perimeter)",
            (k + 1) % 5
        );
    }

    // Score a probe proton 3.0 A above a 1-nitrogen pyrrole (`Trp`-like, `0.90`) vs 2-nitrogen imidazole (`His`-like, `0.43`)
    // (`>= 8` CA atoms in `probe_pdb` to satisfy `21.W1:P1-4` Layer 4 `TM-score` firewall)
    let mut probe_pdb = String::from(
        "ATOM      1  N   ALA A   1       0.000   0.000   3.800  1.00 90.00           N\n\
         ATOM      2  H   ALA A   1       0.000   0.000   2.800  1.00 90.00           H\n\
         ATOM      3  CA  ALA A   1       1.200   0.000   4.500  1.00 90.00           C\n\
         ATOM      4  C   ALA A   1       1.200   1.400   5.000  1.00 90.00           C\n\
         ATOM      5  O   ALA A   1       0.200   2.100   5.000  1.00 90.00           O\n",
    );
    for r in 2..=8 {
        let z = 4.5 + (r as f64 - 1.0) * 3.8;
        let base_serial = (r - 1) * 5;
        probe_pdb.push_str(&format!(
            "ATOM   {:4}  N   ALA A {:3}       1.200   1.800 {:7.3}  1.00 90.00           N\n\
             ATOM   {:4}  CA  ALA A {:3}       1.800   0.600 {:7.3}  1.00 90.00           C\n\
             ATOM   {:4}  C   ALA A {:3}       1.200  -0.600 {:7.3}  1.00 90.00           C\n\
             ATOM   {:4}  O   ALA A {:3}       0.100  -0.600 {:7.3}  1.00 90.00           O\n",
            base_serial + 1,
            r,
            z - 1.2,
            base_serial + 2,
            r,
            z,
            base_serial + 3,
            r,
            z + 1.2,
            base_serial + 4,
            r,
            z + 1.2,
        ));
    }
    let (_, pyrrole_mol) = ligand_spec_from_octet_str("c1cc[nH]c1", Some("smi")).unwrap();
    let (_, imidaz_mol) = ligand_spec_from_octet_str("c1c[nH]cn1", Some("smi")).unwrap();
    let ring_coords = [
        ("C1", "C", 1.15, 0.00),
        ("C2", "C", 0.36, 1.09),
        ("C3", "C", -0.93, 0.68),
        ("N4", "N", -0.93, -0.68),
        ("C5", "C", 0.36, -1.09),
    ];
    let mut pyr_pose = probe_pdb.clone();
    for (idx, &(an, el, x, y)) in ring_coords.iter().enumerate() {
        pyr_pose.push_str(&format!(
            "HETATM {:4} {:<4} LIG L   1    {:8.3}{:8.3}   0.000  1.00 90.00          {:>2}\n",
            100 + idx,
            an,
            x,
            y,
            el
        ));
    }
    let mut imi_pose = probe_pdb.clone();
    for (idx, &(an, el, x, y)) in ring_coords.iter().enumerate() {
        let (an2, el2) = if idx == 1 { ("N2", "N") } else { (an, el) };
        imi_pose.push_str(&format!(
            "HETATM {:4} {:<4} LIG L   1    {:8.3}{:8.3}   0.000  1.00 90.00          {:>2}\n",
            100 + idx,
            an2,
            x,
            y,
            el2
        ));
    }
    let make_ring_bundle = |mol: nmr_core::octet_mol::Molecule| {
        let mut lm = BTreeMap::new();
        lm.insert("L".to_string(), mol);
        extract_nmr_interface_restraints(
            &CofoldRestraintInput {
                entities: vec![CofoldEntitySpec::Polymer {
                    kind: CofoldPolymerKind::Protein,
                    chain_id: "A".to_string(),
                    sequence: "AAAAAAAA".to_string(),
                }],
                ligand_molecules: lm,
                receptor_chain_id: Some("A".to_string()),
                binder_chain_id: Some("L".to_string()),
                raw_csp_rows: vec![(1, "ALA".to_string(), 0.25, None, None)],
                ..Default::default()
            },
            &CofoldRestraintExtractConfig {
                allow_unfiltered_no_apo: true,
                ..CofoldRestraintExtractConfig::default()
            },
        )
        .unwrap()
    };
    let fwd_cfg = PoseScoreConfig {
        apo_receptor_coord_text: Some(probe_pdb),
        apo_receptor_format_hint: Some("pdb".to_string()),
        csp_scoring_mode: CspScoringMode::FullForwardShiftAndBurial,
        ..PoseScoreConfig::default()
    };
    let rep_pyr = score_cofold_poses(
        &[StructurePoseInput::from_coord_str(
            "pyr",
            pyr_pose,
            Some("pdb"),
        )],
        &make_ring_bundle(pyrrole_mol),
        &fwd_cfg,
    )
    .unwrap();
    let rep_imi = score_cofold_poses(
        &[StructurePoseInput::from_coord_str(
            "imi",
            imi_pose,
            Some("pdb"),
        )],
        &make_ring_bundle(imidaz_mol),
        &fwd_cfg,
    )
    .unwrap();
    let dh_pyr = rep_pyr.poses[0].residue_csp_points[0]
        .delta_h_calc_ppm
        .abs();
    let dh_imi = rep_imi.poses[0].residue_csp_points[0]
        .delta_h_calc_ppm
        .abs();
    let ratio = dh_pyr / dh_imi.max(1e-9);
    assert!(
        (ratio - (0.90 / 0.43)).abs() < 0.15,
        "1-N pyrrole (0.90) vs 2-N imidazole (0.43) ring-current ratio must be ~2.09, got {ratio:.3} ({dh_pyr:.4} vs {dh_imi:.4})"
    );
}

#[test]
fn test_21f_b18_pre_requires_explicit_proton_mhz_and_warns_on_14a_mtsl_drop() {
    // Regression test for `21F:B18`:
    // 1. `Gamma_2` PRE TSV with `# tau_c_ns:` but NO `# proton_mhz:` MUST fail closed (no silent 600 MHz default!).
    // 2. Explicit `# proton_mhz: 800.0` succeeds and echoes `800.0 MHz` in both the restraint and bundle diagnostics.
    // 3. Default `14.0 A` Boltz-2 contact export ceiling emits an explicit `WARNING:` diagnostic when
    //    tether-padded MTSL `Gamma_2` PRE restraints are dropped from Boltz-2 export!
    let pre_no_mhz = "# anchor: B 12 CB\n# tau_c_ns: 10.0\n# mode: gamma2\nA\t8\tH\t500.0\t10.0\n";
    let err = parse_intermolecular_pre_tsv_str(pre_no_mhz, None, "A", 1.5)
        .expect_err("21F:B18: Gamma_2 PRE without explicit proton_mhz must fail closed");
    let err_msg = err.to_string();
    assert!(
        err_msg.contains("proton_mhz"),
        "error must mention proton_mhz, got: {err_msg}"
    );

    let pre_800 =
        "# anchor: B 12 CB\n# tau_c_ns: 10.0\n# proton_mhz: 800.0\n# mode: gamma2\nA\t8\tH\t500.0\t10.0\n";
    let pres = parse_intermolecular_pre_tsv_str(pre_800, None, "A", 1.5)
        .expect("explicit tau_c_ns + proton_mhz must succeed");
    assert_eq!(pres.len(), 1);
    assert_eq!(pres[0].pre_proton_mhz, Some(800.0));
    assert!(
        pres[0].upper_a > 14.0,
        "even 500 s^-1 MTSL PRE with 6.0 A tether + 1.5 A margin exceeds 14.0 A (got {:.2} A)",
        pres[0].upper_a
    );

    let ubq_pdb = std::fs::read_to_string(ubq_fixture_path()).expect("read 1UBQ");
    let apo = StructureModel::parse_str(&ubq_pdb, Some("pdb"), false).expect("parse 1UBQ");
    let apo_seq: String = apo
        .residues
        .iter()
        .filter(|r| r.chain_id == "A")
        .map(|r| match r.res_name.as_str() {
            "ALA" => 'A', "ARG" => 'R', "ASN" => 'N', "ASP" => 'D', "CYS" => 'C',
            "GLN" => 'Q', "GLU" => 'E', "GLY" => 'G', "HIS" => 'H', "ILE" => 'I',
            "LEU" => 'L', "LYS" => 'K', "MET" => 'M', "PHE" => 'F', "PRO" => 'P',
            "SER" => 'S', "THR" => 'T', "TRP" => 'W', "TYR" => 'Y', "VAL" => 'V',
            _ => 'X',
        })
        .collect();
    let bundle = extract_nmr_interface_restraints(
        &CofoldRestraintInput {
            job_name: "b18_mtsl_pre_drop_diag".to_string(),
            entities: vec![
                CofoldEntitySpec::Polymer {
                    kind: CofoldPolymerKind::Protein,
                    chain_id: "A".to_string(),
                    sequence: apo_seq,
                },
                CofoldEntitySpec::Polymer {
                    kind: CofoldPolymerKind::Protein,
                    chain_id: "B".to_string(),
                    sequence: "AAAAAAAAAAAAAAAAAAAA".to_string(),
                },
            ],
            receptor_chain_id: Some("A".to_string()),
            binder_chain_id: Some("B".to_string()),
            apo_receptor_model: Some(apo),
            raw_distance_restraints: pres,
            ..CofoldRestraintInput::default()
        },
        &CofoldRestraintExtractConfig::default(),
    )
    .expect("extract bundle");

    assert_eq!(bundle.distance_restraints[0].exported_contact_max_distance_a, None);
    assert_eq!(bundle.distance_restraints[0].pre_proton_mhz, Some(800.0));
    assert!(
        bundle
            .diagnostics
            .iter()
            .any(|d| d.contains("proton_mhz=800.0 MHz")),
        "bundle diagnostics must echo explicit proton_mhz=800.0 MHz, got: {:?}",
        bundle.diagnostics
    );
    assert!(
        bundle
            .diagnostics
            .iter()
            .any(|d| d.contains("WARNING:") && d.contains("Solomon Gamma2 PRE") && d.contains("14.0 A")),
        "bundle diagnostics must emit explicit WARNING when default 14.0 A ceiling drops MTSL PRE restraints, got: {:?}",
        bundle.diagnostics
    );
}
