//! End-to-end CLI integration tests for `ferric cofold (export | score)` (`T21.6a`).

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::{self, Command};

use nmr_core::cofold_bridge::{compute_shrake_rupley_rsasa, PoseScoreReport, PoseVerdict};
use nmr_core::csp::amide_alpha_n;
use nmr_core::noesy::{v_add, v_cross, v_dist, v_scale, v_sub, v_unit, Vec3};
use nmr_core::octet_mol::{parse_smiles, write_mol_v2000};
use nmr_core::struct_shifts::{
    evaluate_haigh_mallion_ring_current, ConjugatedAromaticRing, StructureModel,
};

fn ferric_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_ferric"))
}

fn ubq_fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/fixtures/ubiquitin_1ubq_xray.pdb")
}

fn build_indole_amide_heavy_atoms(
    center: Vec3,
    e1: Vec3,
    e2: Vec3,
) -> Vec<(&'static str, &'static str, Vec3)> {
    let local_xy: [(&str, &str, f64, f64); 13] = [
        ("C1", "C", -1.40, 1.21),
        ("C2", "C", -2.60, 0.52),
        ("C3", "C", -2.60, -0.87),
        ("C4", "C", -1.40, -1.56),
        ("C5", "C", -0.20, -0.87),
        ("C6", "C", -0.20, 0.52),
        ("N7", "N", 1.12, -1.15),
        ("C8", "C", 1.85, 0.00),
        ("C9", "C", 1.08, 1.08),
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

fn render_complex_pdb_with_ligand(
    receptor_pdb_text: &str,
    ligand_atoms: &[(&str, &str, Vec3)],
) -> String {
    let mut pdb = String::new();
    for line in receptor_pdb_text.lines() {
        if line.starts_with("ATOM  ") && line.len() >= 54 && &line[17..20] != "HOH" {
            pdb.push_str(line);
            pdb.push('\n');
        }
    }
    pdb.push_str("TER\n");
    for (idx, &(aname, el, pos)) in ligand_atoms.iter().enumerate() {
        pdb.push_str(&format!(
            "HETATM{:5} {:<4} LIG L   1    {:8.3}{:8.3}{:8.3}  1.00 85.00          {:>2}\n",
            2000 + idx + 1,
            aname,
            pos[0],
            pos[1],
            pos[2],
            el
        ));
    }
    pdb.push_str("END\n");
    pdb
}

#[test]
fn cli_cofold_export_and_score_e2e_pipeline() {
    let dir = env::temp_dir().join(format!("ferric_cofold_cli_e2e_{}", process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create temp dir");

    // 1. Verify `--help` exits 0
    let help_out = Command::new(ferric_bin())
        .args(["cofold", "--help"])
        .output()
        .expect("run ferric cofold --help");
    assert!(help_out.status.success());
    let stdout_help = String::from_utf8_lossy(&help_out.stdout);
    assert!(stdout_help.contains("ferric cofold export"));
    assert!(stdout_help.contains("ferric cofold score"));

    // 2. Prepare multi-chain FASTA, Octet `.mol` ligand, CSP TSV, and NOE TSV
    let ubq_pdb_path = ubq_fixture_path();
    let ubq_pdb_text = fs::read_to_string(&ubq_pdb_path).expect("read 1UBQ");
    let ubq = StructureModel::parse_str(&ubq_pdb_text, Some("pdb"), false).expect("parse 1UBQ");
    let seq_a: String = ubq.residues.iter().map(|r| char::from(r.aa1)).collect();

    let fasta_path = dir.join("complex.fasta");
    fs::write(&fasta_path, format!(">A|protein\n{seq_a}\n")).expect("write FASTA");

    let lig_mol = parse_smiles("CNC(=O)c1cc2ccccc2[nH]1").expect("parse SMILES");
    let mol_v2000 = write_mol_v2000(&lig_mol).expect("write MOL");
    let lig_mol_path = dir.join("ligand.mol");
    fs::write(&lig_mol_path, mol_v2000).expect("write ligand.mol");

    // Position native ligand above Ubiquitin Ile44/His68/Leu8 pocket
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

    let native_lig_center = v_add(pocket_center, v_scale(outward, 2.3));
    let native_lig = build_indole_amide_heavy_atoms(native_lig_center, e1, e2);
    let native_pdb = render_complex_pdb_with_ligand(&ubq_pdb_text, &native_lig);
    let native_ref_path = dir.join("native_ref.pdb");
    fs::write(&native_ref_path, &native_pdb).expect("write native_ref.pdb");

    // Generate realistic CSP TSV (+ one planted buried Val26 core shift to test CLI rSASA filtering!)
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
            vertices: native_lig[0..6].iter().map(|a| a.2).collect(),
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
            vertices: vec![
                native_lig[4].2,
                native_lig[5].2,
                native_lig[6].2,
                native_lig[7].2,
                native_lig[8].2,
            ],
            intensity_factor: 0.56,
        },
    ];
    let lig_occ: Vec<(Vec3, u8)> = native_lig
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
    let sasa_nat = compute_shrake_rupley_rsasa(&ubq, &lig_occ);

    let mut csp_tsv = String::from("seq_id\tres_name\tcsp_ppm\n");
    for (idx, r) in ubq.residues.iter().enumerate() {
        let mut dh = 0.0;
        let mut dn = 0.0;
        if let Some(h_pos) = r.h {
            let (rc_h, _) =
                evaluate_haigh_mallion_ring_current(&native_rings, h_pos, usize::MAX, false, 1.0);
            dh += rc_h;
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
        let mut val = dh
            .hypot(amide_alpha_n(&r.res_name) * dn)
            .hypot(0.12 * d_rsasa + 0.08 * prox)
            + 0.005;
        if r.seq_id == 26 {
            val = 0.350; // Buried core allosteric relay!
        }
        csp_tsv.push_str(&format!("{}\t{}\t{:.5}\n", r.seq_id, r.res_name, val));
    }
    let csp_path = dir.join("titration_csp.tsv");
    fs::write(&csp_path, csp_tsv).expect("write CSP TSV");

    let h_44 = ubq.residues[by_sid[&44]].h.unwrap();
    let noe_tsv = format!(
        "chain_1\tseq_id_1\tatom_1\tchain_2\tatom_2\tupper_a\n\
         A\t44\tH\tL\tC5\t{:.2}\n",
        v_dist(h_44, native_lig[4].2) + 0.85
    );
    let noe_path = dir.join("inter_noe.tsv");
    fs::write(&noe_path, noe_tsv).expect("write NOE TSV");

    // 3. Run `ferric cofold export --out-prefix <prefix>`
    let prefix = dir.join("ubq_lig_job");
    let exp_out = Command::new(ferric_bin())
        .args([
            "cofold",
            "export",
            "--fasta",
            fasta_path.to_str().unwrap(),
            "--ligand-file",
            &format!("L:{}", lig_mol_path.display()),
            "--apo-structure",
            ubq_pdb_path.to_str().unwrap(),
            "--csp",
            csp_path.to_str().unwrap(),
            "--noesy",
            noe_path.to_str().unwrap(),
            "--affinity",
            "--seeds",
            "1,2",
            "--out-prefix",
            prefix.to_str().unwrap(),
        ])
        .output()
        .expect("run ferric cofold export");
    assert!(
        exp_out.status.success(),
        "ferric cofold export failed: {}",
        String::from_utf8_lossy(&exp_out.stderr)
    );

    let af3_file = dir.join("ubq_lig_job.af3.json");
    let boltz_file = dir.join("ubq_lig_job.boltz2.yaml");
    let sidecar_file = dir.join("ubq_lig_job.nmr_restraints.json");
    assert!(af3_file.exists());
    assert!(boltz_file.exists());
    assert!(sidecar_file.exists());

    let af3_json: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&af3_file).unwrap()).unwrap();
    assert_eq!(af3_json["dialect"], "alphafold3");
    assert_eq!(af3_json["version"], 2);
    assert!(af3_json.get("pocket").is_none());

    let boltz_txt = fs::read_to_string(&boltz_file).unwrap();
    assert!(boltz_txt.starts_with("version: 1\n"));
    assert!(!boltz_txt.contains("        - [A, 26]\n"));
    assert!(boltz_txt.contains("properties:\n  - affinity:\n      binder: L\n"));

    // 4. Write candidate poses (1 near-native + 3 decoys) and run `ferric cofold score`
    let poses_dir = dir.join("poses");
    fs::create_dir_all(&poses_dir).unwrap();
    let near_lig =
        build_indole_amide_heavy_atoms(v_add(native_lig_center, v_scale(e1, 0.50)), e1, e2);
    fs::write(
        poses_dir.join("00_near_native.pdb"),
        render_complex_pdb_with_ligand(&ubq_pdb_text, &near_lig),
    )
    .unwrap();

    let decoy_shifts = [(10.0, 4.0, 2.0), (-12.0, -8.0, 4.0), (0.0, 0.0, 22.0)];
    let mut decoy_paths = Vec::new();
    for (k, &(dx, dy, dz)) in decoy_shifts.iter().enumerate() {
        let dl = build_indole_amide_heavy_atoms(
            v_add(
                native_lig_center,
                v_add(
                    v_scale(e1, dx),
                    v_add(v_scale(e2, dy), v_scale(outward, dz)),
                ),
            ),
            v_scale(e1, -1.0),
            v_scale(e2, -1.0),
        );
        let dp = poses_dir.join(format!("0{}_decoy.pdb", k + 1));
        fs::write(&dp, render_complex_pdb_with_ligand(&ubq_pdb_text, &dl)).unwrap();
        decoy_paths.push(dp);
    }

    let score_json_path = dir.join("scores.json");
    let score_out = Command::new(ferric_bin())
        .args([
            "cofold",
            "score",
            "--pose-dir",
            poses_dir.to_str().unwrap(),
            "--restraints",
            sidecar_file.to_str().unwrap(),
            "--apo-structure",
            ubq_pdb_path.to_str().unwrap(),
            "--reference-pose",
            native_ref_path.to_str().unwrap(),
            "-o",
            score_json_path.to_str().unwrap(),
        ])
        .output()
        .expect("run ferric cofold score");
    assert!(
        score_out.status.success(),
        "ferric cofold score failed: {}",
        String::from_utf8_lossy(&score_out.stderr)
    );

    let report: PoseScoreReport =
        serde_json::from_str(&fs::read_to_string(&score_json_path).unwrap()).unwrap();
    assert_eq!(report.verdict, PoseVerdict::RankedConclusive);
    assert_eq!(report.top_pose_id.as_deref(), Some("00_near_native"));

    // 5. Decoy-only CLI invocation -> `InconclusiveOrAllViolated`!
    let decoy_csv = decoy_paths
        .iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join(",");
    let decoy_json_path = dir.join("decoy_only.json");
    let decoy_out = Command::new(ferric_bin())
        .args([
            "cofold",
            "score",
            "--poses",
            &decoy_csv,
            "--restraints",
            sidecar_file.to_str().unwrap(),
            "--apo-structure",
            ubq_pdb_path.to_str().unwrap(),
            "-o",
            decoy_json_path.to_str().unwrap(),
        ])
        .output()
        .expect("run ferric cofold score on decoys only");
    assert!(decoy_out.status.success());
    let decoy_report: PoseScoreReport =
        serde_json::from_str(&fs::read_to_string(&decoy_json_path).unwrap()).unwrap();
    assert_eq!(decoy_report.verdict, PoseVerdict::InconclusiveOrAllViolated);
    assert_eq!(decoy_report.top_pose_id, None);

    let _ = fs::remove_dir_all(&dir);
}
