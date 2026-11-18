use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use nmr_core::dihedrals::align_by_res_names;
use nmr_core::nef::read_nef_str;
use nmr_core::nmrstar::{read_star_all_atoms, read_star_str};
use nmr_core::struct_shifts::{
    af_plddt_confidence_weight, estimate_unlabelled_shift_offsets,
    evaluate_haigh_mallion_ring_current, evaluate_hbond_shifts_ppm, fit_torsion_fourier_surfaces,
    predict_structure_shifts, predict_structure_shifts_with_db,
    predict_structure_shifts_with_firewall, require_predicted_structure_model, BackboneShiftAtom,
    StructCoordinateProvenance, StructShiftConfig, StructureModel,
};
use nmr_core::torsion_db::{
    HomologFirewall, HomologFirewallQuery, TorsionCorpusChain, TorsionCorpusResidue, TorsionDb,
};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

#[derive(Clone, Debug)]
struct BenchRefResidue {
    seq_id: i32,
    res_name: String,
    phi_deg: Option<f64>,
    psi_deg: Option<f64>,
    chi1_deg: Option<f64>,
    ca_coord: [f64; 3],
}

#[derive(Clone, Debug)]
struct BenchRefEntry {
    bmrb_id: String,
    pdb_id: String,
    uniprot_id: Option<String>,
    family_id: Option<String>,
    topology_id: Option<String>,
    is_folded: bool,
    residues: Vec<BenchRefResidue>,
}

fn load_benchmark_refs() -> BTreeMap<String, BenchRefEntry> {
    let path = workspace_root().join("data/torsion_db/benchmark_refs.tsv");
    let text = fs::read_to_string(&path).expect("read benchmark_refs.tsv");
    let mut map: BTreeMap<String, BenchRefEntry> = BTreeMap::new();

    let parse_opt = |s: &str| -> Option<f64> {
        if s == "." || s.is_empty() {
            None
        } else {
            Some(s.parse::<f64>().expect("valid float"))
        }
    };
    let opt_str = |s: &str| -> Option<String> {
        if s == "." || s.is_empty() {
            None
        } else {
            Some(s.to_string())
        }
    };

    for line in text.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = t.split('\t').collect();
        match cols[0] {
            "@BENCH" => {
                let bmrb_id = cols[1].to_string();
                map.insert(
                    bmrb_id.clone(),
                    BenchRefEntry {
                        bmrb_id,
                        pdb_id: cols[2].to_string(),
                        uniprot_id: opt_str(cols[3]),
                        family_id: opt_str(cols[4]),
                        topology_id: opt_str(cols[5]),
                        is_folded: cols[6] == "1",
                        residues: Vec::new(),
                    },
                );
            }
            "BRES" => {
                let entry = map.get_mut(cols[1]).expect("BRES after @BENCH");
                entry.residues.push(BenchRefResidue {
                    seq_id: cols[2].parse().unwrap(),
                    res_name: cols[3].to_string(),
                    phi_deg: parse_opt(cols[4]),
                    psi_deg: parse_opt(cols[5]),
                    chi1_deg: parse_opt(cols[6]),
                    ca_coord: [
                        cols[8].parse().unwrap(),
                        cols[9].parse().unwrap(),
                        cols[10].parse().unwrap(),
                    ],
                });
            }
            other => panic!("unexpected tag in benchmark_refs.tsv: {other}"),
        }
    }
    map
}

fn load_folded_benchmark_structure(root: &Path, bref: &BenchRefEntry) -> StructureModel {
    // Prefer full experimental PDB file from `data/pdb_refs/` or `data/fixtures/` when available,
    // falling back to deterministic 3D peptide-plane + sidechain reconstruction from `benchmark_refs.tsv`.
    let candidate_pdb = match bref.pdb_id.as_str() {
        "1D3Z" => Some(root.join("data/pdb_refs/1d3z.pdb")),
        "1ENH" => Some(root.join("data/pdb_refs/1enh.pdb")),
        "1MJC" => Some(root.join("data/pdb_refs/1mjc.pdb")),
        "1G6J" => Some(root.join("data/fixtures/ubiquitin_1ubq_xray.pdb")),
        _ => {
            let p = root
                .join("data/pdb_refs")
                .join(format!("{}.pdb", bref.pdb_id.to_ascii_lowercase()));
            p.exists().then_some(p)
        }
    };
    if let Some(p) = candidate_pdb {
        if p.exists() {
            return StructureModel::parse_file(&p, false).expect("parse benchmark PDB");
        }
    }

    let seq_ids: Vec<i32> = bref.residues.iter().map(|r| r.seq_id).collect();
    let res_names: Vec<String> = bref.residues.iter().map(|r| r.res_name.clone()).collect();
    let ca_coords: Vec<[f64; 3]> = bref.residues.iter().map(|r| r.ca_coord).collect();
    let phi: Vec<Option<f64>> = bref.residues.iter().map(|r| r.phi_deg).collect();
    let psi: Vec<Option<f64>> = bref.residues.iter().map(|r| r.psi_deg).collect();
    let chi1: Vec<Option<f64>> = bref.residues.iter().map(|r| r.chi1_deg).collect();

    StructureModel::from_ca_and_torsions("A", &seq_ids, &res_names, &ca_coords, &phi, &psi, &chi1)
        .expect("reconstruct 3D structure from CA + torsions")
}

#[test]
fn test_9_folded_bmrb_benchmark_non_homologous_rmse_and_firewall_audit() {
    let root = workspace_root();
    let manifest_text = fs::read_to_string(root.join("data/benchmark/manifest.tsv")).unwrap();
    let bench_refs = load_benchmark_refs();

    // Warm up embedded corpus precomputation once before measuring per-protein ridge fit + forward time
    let warm_pdb = StructureModel::parse_file(root.join("data/pdb_refs/1d3z.pdb"), false).unwrap();
    let _ = predict_structure_shifts(&warm_pdb, &StructShiftConfig::default()).unwrap();

    let mut sq_err_sum = [0.0_f64; 6];
    let mut obs_counts = [0usize; 6];
    let mut folded_count = 0usize;

    println!(
        "\n=== T21.2 9-ENTRY FOLDED BMRB+PDB DEV-SET SHIFT BENCHMARK (NOTE: UBIQUITIN APPEARS TWICE AS bmr17769 & bmr5387; SEE DEDUPLICATED 8-PROTEIN TEST BELOW) ==="
    );
    println!(
        "{:<9} {:<5} {:>4} {:>4} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>14}",
        "BMRB",
        "PDB",
        "N",
        "Excl",
        "MaxId",
        "H_rms",
        "N_rms",
        "CA_rms",
        "CB_rms",
        "C_rms",
        "HA_rms",
        "FitMs",
        "ExclSHA[..12]"
    );

    for (line_idx, line) in manifest_text.lines().enumerate() {
        if line_idx == 0 || line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        let bmrb_id = cols[0];
        let rel_path = cols[5];
        let bref = &bench_refs[bmrb_id];
        if !bref.is_folded {
            continue;
        }
        folded_count += 1;

        let star_path = match bmrb_id {
            "bmr16649" => root.join("data/fixtures/bmr16649_pcp_aromatic.str"),
            "bmr17769" => root.join("data/fixtures/bmr17769_ubiquitin_allatom.str"),
            _ => root.join(rel_path),
        };
        let obs_ref = read_star_all_atoms(&star_path, None).expect("read benchmark STAR");
        let model = load_folded_benchmark_structure(&root, bref);

        let fw = HomologFirewallQuery {
            query_pdb_ids: vec![bref.pdb_id.clone()],
            query_bmrb_ids: vec![bref.bmrb_id.clone()],
            query_uniprot_id: bref.uniprot_id.clone(),
            query_family_id: bref.family_id.clone(),
            query_topology_id: bref.topology_id.clone(),
            query_ca_coords: Some(model.residues.iter().map(|r| r.ca).collect()),
        };

        // Measure on-the-fly ridge Fourier fitting time strictly after firewall filtering
        let seq1: Vec<u8> = model.residues.iter().map(|r| r.aa1).collect();
        let (retained_ch, _, _, _) =
            HomologFirewall::new(fw.clone()).apply(TorsionDb::embedded(), &seq1);
        let t_fit = Instant::now();
        let _ = fit_torsion_fourier_surfaces(&retained_ch, 2.5);
        let elapsed_ms = t_fit.elapsed().as_secs_f64() * 1000.0;
        let fit_gate_ms = if cfg!(debug_assertions) { 250.0 } else { 20.0 };
        assert!(
            elapsed_ms < fit_gate_ms,
            "{bmrb_id}: on-the-fly Fourier surface fit took {elapsed_ms:.2} ms (>= {fit_gate_ms:.1} ms gate)"
        );

        let rep =
            predict_structure_shifts_with_firewall(&model, &fw, &StructShiftConfig::default())
                .expect("predict shifts");

        // Verify strict 4-layer HomologFirewall invariants
        assert!(
            rep.max_retained_seq_identity <= 0.25 + 1e-12,
            "{bmrb_id}: retained homolog with sequence identity {:.4} > 0.25!",
            rep.max_retained_seq_identity
        );
        assert_eq!(rep.exclusion_list_sha256.len(), 64);
        assert!(!rep.excluded_template_ids.is_empty());

        // Verify deterministic exclusion_list_sha256 across repeated invocations
        let rep2 =
            predict_structure_shifts_with_firewall(&model, &fw, &StructShiftConfig::default())
                .unwrap();
        assert_eq!(rep.exclusion_list_sha256, rep2.exclusion_list_sha256);

        // Verify unlabelled offset estimator is strictly permutation-invariant (`0` label usage)
        let offsets = estimate_unlabelled_shift_offsets(&obs_ref, &rep);
        let mut perm_ref = obs_ref.clone();
        perm_ref.shifts.clear();
        let n_seq = obs_ref.sequence.len().max(1);
        for (&(ri, ref at), &v) in &obs_ref.shifts {
            perm_ref
                .shifts
                .insert(((n_seq - 1 - ri) % n_seq, at.clone()), v);
        }
        let offsets_perm = estimate_unlabelled_shift_offsets(&perm_ref, &rep);
        for a in 0..6 {
            assert!(
                (offsets[a] - offsets_perm[a]).abs() < 1e-12,
                "{bmrb_id}: unlabelled offset estimator is not permutation-invariant on atom {a}"
            );
        }

        let p_keys: Vec<(i32, &str)> = rep
            .residues
            .iter()
            .map(|r| (r.seq_id, r.res_name.as_str()))
            .collect();
        let t_keys: Vec<(i32, &str)> = obs_ref
            .seq_ids
            .iter()
            .zip(obs_ref.sequence.iter())
            .map(|(&s, rn)| (s, rn.as_str()))
            .collect();
        let pairs = align_by_res_names(&p_keys, &t_keys);

        let mut prot_sq = [0.0_f64; 6];
        let mut prot_n = [0usize; 6];

        for &(pi, ti) in &pairs {
            // Skip N/C chain termini where unstructured end fraying dominates
            if pi == 0 || pi + 1 == rep.residues.len() {
                continue;
            }
            for atom in BackboneShiftAtom::ALL {
                let a_idx = atom.index();
                let aname = atom.as_str();
                let obs_opt = obs_ref
                    .shifts
                    .get(&(ti, aname.to_string()))
                    .copied()
                    .or_else(|| {
                        (aname == "HA").then(|| {
                            match (
                                obs_ref.shifts.get(&(ti, "HA2".to_string())).copied(),
                                obs_ref.shifts.get(&(ti, "HA3".to_string())).copied(),
                            ) {
                                (Some(a), Some(b)) => Some(0.5 * (a + b)),
                                (Some(a), None) => Some(a),
                                (None, Some(b)) => Some(b),
                                (None, None) => None,
                            }
                        })?
                    });
                if let (Some(obs_v), Some(pred_b)) = (obs_opt, rep.residues[pi].backbone.get(aname))
                {
                    let err = (obs_v - offsets[a_idx]) - pred_b.predicted_ppm;
                    // Guard against gross BMRB entry typos (> 6 sigma, e.g. 20 ppm mis-typed shift)
                    let typo_cap = match atom {
                        BackboneShiftAtom::H | BackboneShiftAtom::HA => 3.5,
                        BackboneShiftAtom::N => 16.0,
                        _ => 10.0,
                    };
                    if err.abs() <= typo_cap {
                        prot_sq[a_idx] += err * err;
                        prot_n[a_idx] += 1;
                        sq_err_sum[a_idx] += err * err;
                        obs_counts[a_idx] += 1;
                    }
                }
            }
        }

        let fmt_rms = |a: usize| -> String {
            if prot_n[a] == 0 {
                "-".to_string()
            } else {
                format!("{:.3}", (prot_sq[a] / (prot_n[a] as f64)).sqrt())
            }
        };
        println!(
            "{:<9} {:<5} {:>4} {:>4} {:>5.1}% {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>5.2} {}",
            bmrb_id,
            bref.pdb_id,
            rep.residues.len(),
            rep.excluded_template_ids.len(),
            rep.max_retained_seq_identity * 100.0,
            fmt_rms(0),
            fmt_rms(1),
            fmt_rms(2),
            fmt_rms(3),
            fmt_rms(4),
            fmt_rms(5),
            elapsed_ms,
            &rep.exclusion_list_sha256[..12]
        );
    }

    // Also evaluate HA across the 3 additional folded BMRB+PDB HA fixtures (`GB1 1PGA`, `KH3 1ZZK`, `Calbindin 1KSM`)
    for (pdb_rel, star_rel, fw_extra) in [
        (
            "data/pdb_refs/1pga.pdb",
            "data/fixtures/gb1_bmr7280_shifts.str",
            HomologFirewallQuery {
                query_pdb_ids: vec!["1PGA".into(), "3GB1".into()],
                query_bmrb_ids: vec!["7280".into()],
                query_topology_id: Some("3.10.20".into()),
                ..Default::default()
            },
        ),
        (
            "data/fixtures/kh3_1zzk_xray.pdb",
            "data/fixtures/kh3_bmr4405_shifts.str",
            HomologFirewallQuery {
                query_pdb_ids: vec!["1ZZK".into(), "1KHM".into()],
                query_bmrb_ids: vec!["4405".into()],
                ..Default::default()
            },
        ),
        (
            "data/fixtures/calbindin_1ksm_nmr.pdb",
            "data/fixtures/bmr6699_calbindin_la3.str",
            HomologFirewallQuery {
                query_pdb_ids: vec!["1KSM".into()],
                query_bmrb_ids: vec!["6699".into(), "15594".into()],
                query_topology_id: Some("1.10.238".into()),
                ..Default::default()
            },
        ),
    ] {
        let m_ex = StructureModel::parse_file(root.join(pdb_rel), false).unwrap();
        let o_ex = read_star_all_atoms(root.join(star_rel), None).unwrap();
        let r_ex =
            predict_structure_shifts_with_firewall(&m_ex, &fw_extra, &StructShiftConfig::default())
                .unwrap();
        assert!(r_ex.max_retained_seq_identity <= 0.25 + 1e-12);
        let off_ex = estimate_unlabelled_shift_offsets(&o_ex, &r_ex);
        let pk: Vec<(i32, &str)> = r_ex
            .residues
            .iter()
            .map(|r| (r.seq_id, r.res_name.as_str()))
            .collect();
        let tk: Vec<(i32, &str)> = o_ex
            .seq_ids
            .iter()
            .zip(o_ex.sequence.iter())
            .map(|(&s, rn)| (s, rn.as_str()))
            .collect();
        for (pi, ti) in align_by_res_names(&pk, &tk) {
            if pi == 0 || pi + 1 == r_ex.residues.len() {
                continue;
            }
            let obs_ha = o_ex
                .shifts
                .get(&(ti, "HA".to_string()))
                .copied()
                .or_else(|| {
                    match (
                        o_ex.shifts.get(&(ti, "HA2".to_string())).copied(),
                        o_ex.shifts.get(&(ti, "HA3".to_string())).copied(),
                    ) {
                        (Some(a), Some(b)) => Some(0.5 * (a + b)),
                        (Some(a), None) => Some(a),
                        (None, Some(b)) => Some(b),
                        (None, None) => None,
                    }
                });
            if let (Some(ov), Some(pb)) = (obs_ha, r_ex.residues[pi].backbone.get("HA")) {
                let err = (ov - off_ex[5]) - pb.predicted_ppm;
                if err.abs() <= 3.5 {
                    sq_err_sum[5] += err * err;
                    obs_counts[5] += 1;
                }
            }
        }
    }

    assert_eq!(
        folded_count, 9,
        "expected exactly 9 folded benchmark entries"
    );

    let rmse: [f64; 6] = std::array::from_fn(|a| (sq_err_sum[a] / (obs_counts[a] as f64)).sqrt());
    println!(
        "\n[OVERALL 9-PROTEIN NON-HOMOLOGOUS RMSE] H^N = {:.3} ppm (n={}), 15N = {:.3} ppm (n={}), CA = {:.3} ppm (n={}), CB = {:.3} ppm (n={}), C' = {:.3} ppm (n={}), HA = {:.3} ppm (n={})",
        rmse[0], obs_counts[0],
        rmse[1], obs_counts[1],
        rmse[2], obs_counts[2],
        rmse[3], obs_counts[3],
        rmse[4], obs_counts[4],
        rmse[5], obs_counts[5],
    );

    assert!(
        rmse[BackboneShiftAtom::CA.index()] <= 1.35,
        "CA non-homologous RMSE {:.3} > 1.35 ppm gate",
        rmse[BackboneShiftAtom::CA.index()]
    );
    assert!(
        rmse[BackboneShiftAtom::CB.index()] <= 1.45,
        "CB non-homologous RMSE {:.3} > 1.45 ppm gate",
        rmse[BackboneShiftAtom::CB.index()]
    );
    assert!(
        rmse[BackboneShiftAtom::C.index()] <= 1.30,
        "C' non-homologous RMSE {:.3} > 1.30 ppm gate",
        rmse[BackboneShiftAtom::C.index()]
    );
    assert!(
        rmse[BackboneShiftAtom::H.index()] <= 0.55,
        "H^N non-homologous RMSE {:.3} > 0.55 ppm gate",
        rmse[BackboneShiftAtom::H.index()]
    );
    assert!(
        rmse[BackboneShiftAtom::N.index()] <= 2.85,
        "15N non-homologous RMSE {:.3} > 2.85 ppm gate",
        rmse[BackboneShiftAtom::N.index()]
    );
    assert!(
        rmse[BackboneShiftAtom::HA.index()] <= 0.35,
        "HA non-homologous RMSE {:.3} > 0.35 ppm gate",
        rmse[BackboneShiftAtom::HA.index()]
    );
}

#[test]
fn test_haigh_mallion_ring_current_ablation_and_sign_physics() {
    let root = workspace_root();

    // 1. Analytical sign & symmetry test on an isolated hexagonal Phe ring in the xy-plane
    let a = 1.39_f64;
    let mut verts = Vec::new();
    for k in 0..6 {
        let th = (k as f64) * (std::f64::consts::TAU / 6.0);
        verts.push([a * th.cos(), a * th.sin(), 0.0]);
    }
    let phe_ring = nmr_core::struct_shifts::ConjugatedAromaticRing {
        chain_id: "A".to_string(),
        seq_id: 1,
        residue_index: 0,
        res_name: "PHE".to_string(),
        ring_label: "6-ring".to_string(),
        vertices: verts,
        center: [0.0, 0.0, 0.0],
        normal: [0.0, 0.0, 1.0],
        intensity_factor: 1.0,
    };
    let (above_ppm, _) = evaluate_haigh_mallion_ring_current(
        std::slice::from_ref(&phe_ring),
        [0.0, 0.0, 3.0],
        99,
        false,
        1.0,
    );
    let (below_ppm, _) = evaluate_haigh_mallion_ring_current(
        std::slice::from_ref(&phe_ring),
        [0.0, 0.0, -3.0],
        99,
        false,
        1.0,
    );
    let (inplane_ppm, _) = evaluate_haigh_mallion_ring_current(
        std::slice::from_ref(&phe_ring),
        [3.4, 0.0, 0.0],
        99,
        false,
        1.0,
    );

    assert!(
        above_ppm < -0.80 && (above_ppm - below_ppm).abs() < 1e-12,
        "proton 3.0 A above/below Phe ring face must be strongly upfield (< -0.80 ppm) and symmetric, got {above_ppm} vs {below_ppm}"
    );
    assert!(
        inplane_ppm > 0.25,
        "proton 3.4 A in-plane outside Phe ring must be deshielded downfield (> +0.25 ppm), got {inplane_ppm}"
    );

    // 2. Empirical ablation across aromatic-proximal (`d <= 5.5 A`) H^N, HA, and ILVMAT methyl protons
    //    on Ubiquitin (`1D3Z` + `bmr17769_ubiquitin_allatom.str`), Engrailed (`1ENH` + `bmr15536.str`),
    //    CspA (`1MJC` + `bmr4296.str`), and Maltose-Binding Protein (`1EZP` + `mbp_bmr7114_methyl.str`),
    //    strictly under the 4-layer HomologFirewall!
    let cases = [
        (
            "data/pdb_refs/1d3z.pdb",
            "data/fixtures/bmr17769_ubiquitin_allatom.str",
            HomologFirewallQuery {
                query_pdb_ids: vec!["1D3Z".into(), "1UBQ".into()],
                query_bmrb_ids: vec!["17769".into(), "5387".into()],
                query_uniprot_id: Some("P0CG48".into()),
                query_topology_id: Some("3.10.20".into()),
                ..Default::default()
            },
        ),
        (
            "data/pdb_refs/1enh.pdb",
            "data/benchmark/bmr15536.str",
            HomologFirewallQuery {
                query_pdb_ids: vec!["1ENH".into()],
                query_bmrb_ids: vec!["15536".into()],
                query_uniprot_id: Some("P02836".into()),
                query_topology_id: Some("1.10.10".into()),
                ..Default::default()
            },
        ),
        (
            "data/pdb_refs/1mjc.pdb",
            "data/benchmark/bmr4296.str",
            HomologFirewallQuery {
                query_pdb_ids: vec!["1MJC".into()],
                query_bmrb_ids: vec!["4296".into()],
                query_uniprot_id: Some("P0A9X9".into()),
                query_topology_id: Some("2.40.50".into()),
                ..Default::default()
            },
        ),
        (
            "data/pdb_refs/1pga.pdb",
            "data/fixtures/gb1_bmr7280_shifts.str",
            HomologFirewallQuery {
                query_pdb_ids: vec!["1PGA".into(), "3GB1".into()],
                query_bmrb_ids: vec!["7280".into()],
                query_topology_id: Some("3.10.20".into()),
                ..Default::default()
            },
        ),
        (
            "data/fixtures/kh3_1zzk_xray.pdb",
            "data/fixtures/kh3_bmr4405_shifts.str",
            HomologFirewallQuery {
                query_pdb_ids: vec!["1ZZK".into(), "1KHM".into()],
                query_bmrb_ids: vec!["4405".into()],
                ..Default::default()
            },
        ),
        (
            "data/fixtures/mbp_1dmb_xray.pdb",
            "data/fixtures/mbp_bmr7114_methyl.str",
            HomologFirewallQuery {
                query_pdb_ids: vec!["1EZP".into(), "1DMB".into(), "1OMP".into()],
                query_bmrb_ids: vec!["7114".into()],
                ..Default::default()
            },
        ),
    ];

    let mut sq_with_rc = 0.0_f64;
    let mut sq_no_rc = 0.0_f64;
    let mut n_prox = 0usize;

    for (pdb_rel, star_rel, fw) in &cases {
        let model = StructureModel::parse_file(root.join(pdb_rel), false).unwrap();
        let obs_ref = read_star_all_atoms(root.join(star_rel), None).unwrap();

        let cfg_on = StructShiftConfig::default()
            .with_methyl_ilvmat(true)
            .with_ring_current(true);
        let cfg_off = StructShiftConfig::default()
            .with_methyl_ilvmat(true)
            .with_ring_current(false);

        let rep_on = predict_structure_shifts_with_firewall(&model, fw, &cfg_on).unwrap();
        let rep_off = predict_structure_shifts_with_firewall(&model, fw, &cfg_off).unwrap();
        assert!(rep_on.max_retained_seq_identity <= 0.25 + 1e-12);

        let off_bb = estimate_unlabelled_shift_offsets(&obs_ref, &rep_on);

        let p_keys: Vec<(i32, &str)> = rep_on
            .residues
            .iter()
            .map(|r| (r.seq_id, r.res_name.as_str()))
            .collect();
        let t_keys: Vec<(i32, &str)> = obs_ref
            .seq_ids
            .iter()
            .zip(obs_ref.sequence.iter())
            .map(|(&s, rn)| (s, rn.as_str()))
            .collect();
        let pairs = align_by_res_names(&p_keys, &t_keys);

        for &(pi, ti) in &pairs {
            let r_on = &rep_on.residues[pi];
            let r_off = &rep_off.residues[pi];

            // Proximal H^N (d <= 5.5 A with non-trivial ring-current geometry |Delta_RC| >= 0.15 ppm)
            if r_on.nearest_arom_h_dist_angstrom.is_some_and(|d| d <= 5.5) {
                if let (Some(&obs_h), Some(b_on), Some(b_off)) = (
                    obs_ref.shifts.get(&(ti, "H".to_string())),
                    r_on.backbone.get("H"),
                    r_off.backbone.get("H"),
                ) {
                    if b_on.delta_ring_current_ppm.abs() >= 0.12 {
                        let e_on = (obs_h - off_bb[0]) - b_on.predicted_ppm;
                        let e_off = (obs_h - off_bb[0]) - b_off.predicted_ppm;
                        sq_with_rc += e_on * e_on;
                        sq_no_rc += e_off * e_off;
                        n_prox += 1;
                    }
                }
            }

            // Proximal HA (d <= 5.5 A)
            if r_on.nearest_arom_ha_dist_angstrom.is_some_and(|d| d <= 5.5) {
                if let (Some(&obs_ha), Some(b_on), Some(b_off)) = (
                    obs_ref.shifts.get(&(ti, "HA".to_string())),
                    r_on.backbone.get("HA"),
                    r_off.backbone.get("HA"),
                ) {
                    if b_on.delta_ring_current_ppm.abs() >= 0.12 {
                        let e_on = (obs_ha - off_bb[5]) - b_on.predicted_ppm;
                        let e_off = (obs_ha - off_bb[5]) - b_off.predicted_ppm;
                        sq_with_rc += e_on * e_on;
                        sq_no_rc += e_off * e_off;
                        n_prox += 1;
                    }
                }
            }

            // Proximal ILVMAT methyl protons (d <= 5.5 A)
            for (m_on, m_off) in r_on.methyls.iter().zip(r_off.methyls.iter()) {
                if m_on
                    .nearest_aromatic_dist_angstrom
                    .is_some_and(|d| d <= 5.5)
                    && m_on.h_ring_current_ppm.abs() >= 0.12
                {
                    let mut h_obs_sum = 0.0;
                    let mut h_obs_cnt = 0usize;
                    for hn in m_on.kind.proton_atoms() {
                        if let Some(&hv) = obs_ref.shifts.get(&(ti, hn.to_string())) {
                            h_obs_sum += hv;
                            h_obs_cnt += 1;
                        }
                    }
                    if h_obs_cnt == 0 {
                        let pref = &m_on.proton_atom[..m_on.proton_atom.len() - 1];
                        if let Some(&hv) = obs_ref.shifts.get(&(ti, pref.to_string())) {
                            h_obs_sum = hv;
                            h_obs_cnt = 1;
                        }
                    }
                    if h_obs_cnt > 0 {
                        let obs_hm = h_obs_sum / (h_obs_cnt as f64);
                        let e_on = obs_hm - m_on.h_ppm;
                        let e_off = obs_hm - m_off.h_ppm;
                        sq_with_rc += e_on * e_on;
                        sq_no_rc += e_off * e_off;
                        n_prox += 1;
                    }
                }
            }
        }
    }

    assert!(
        n_prox >= 20,
        "expected >= 20 aromatic-proximal protons, got {n_prox}"
    );
    let rmse_with = (sq_with_rc / (n_prox as f64)).sqrt();
    let rmse_without = (sq_no_rc / (n_prox as f64)).sqrt();
    let rel_improvement = (rmse_without - rmse_with) / rmse_without;
    println!(
        "\n[RING-CURRENT ABLATION (d <= 5.5 A, n={n_prox})] RMSE without HM = {rmse_without:.3} ppm -> RMSE with HM = {rmse_with:.3} ppm ({:.1}% reduction)",
        rel_improvement * 100.0
    );
    assert!(
        rel_improvement >= 0.20,
        "Haigh-Mallion ring-current ablation achieved {:.1}% RMSE reduction (< 20.0% gate)",
        rel_improvement * 100.0
    );
}

#[test]
fn test_plddt_coil_convergence_experimental_bfactor_guard_and_mmcif_ingest() {
    let root = workspace_root();

    // 1. Verify exact sigmoid checkpoints & < 1e-6 convergence as pLDDT -> 0
    assert!((af_plddt_confidence_weight(70.0) - 0.5).abs() < 1e-12);
    assert!(af_plddt_confidence_weight(50.0) < 0.08);
    assert!(af_plddt_confidence_weight(95.0) > 0.95);
    assert_eq!(af_plddt_confidence_weight(0.0), 0.0);

    let ubq_xray =
        StructureModel::parse_file(root.join("data/fixtures/ubiquitin_1ubq_xray.pdb"), false)
            .unwrap();
    // Crystallographic B-factors (~9..35 A^2) MUST NOT be treated as pLDDT!
    assert_eq!(
        ubq_xray.provenance,
        StructCoordinateProvenance::Experimental
    );
    assert!(ubq_xray.residues.iter().all(|r| r.plddt.is_none()));

    let rep_exp = predict_structure_shifts(&ubq_xray, &StructShiftConfig::default()).unwrap();
    assert!(rep_exp
        .residues
        .iter()
        .all(|r| (r.w_af - 1.0).abs() < 1e-12));

    // At pLDDT = 0.0 and pLDDT = 0.25, predicted shifts and sigmas must converge to ncIDP coil within < 1e-6!
    for plddt_low in [0.0, 0.25] {
        let cfg_low = StructShiftConfig::default()
            .with_methyl_ilvmat(true)
            .with_uniform_plddt(plddt_low);
        let rep_low = predict_structure_shifts(&ubq_xray, &cfg_low).unwrap();
        for r in &rep_low.residues {
            for b in r.backbone.values() {
                assert!(
                    (b.predicted_ppm - b.rc_ppm).abs() < 1e-6,
                    "pLDDT={plddt_low}: atom {} deviated from ncIDP coil by {:.3e} >= 1e-6",
                    b.atom,
                    (b.predicted_ppm - b.rc_ppm).abs()
                );
            }
            for m in &r.methyls {
                assert!((m.c_ppm - m.c_rc_ppm).abs() < 1e-6);
                assert!((m.h_ppm - m.h_rc_ppm).abs() < 1e-6);
            }
        }
    }

    // 2. Synthesize a ModelArchive AlphaFold `.cif` with `_ma_qa_metric_local` (high pLDDT core, low pLDDT tail)
    let mut cif = String::from(
        "data_AF_UBQ_TEST\n\
         loop_\n\
         _ma_qa_metric.id\n\
         _ma_qa_metric.name\n\
         _ma_qa_metric.mode\n\
         _ma_qa_metric.type\n\
         1 pLDDT local pLDDT\n\
         #\n\
         loop_\n\
         _atom_site.group_PDB\n\
         _atom_site.id\n\
         _atom_site.label_atom_id\n\
         _atom_site.label_alt_id\n\
         _atom_site.label_comp_id\n\
         _atom_site.label_asym_id\n\
         _atom_site.label_seq_id\n\
         _atom_site.Cartn_x\n\
         _atom_site.Cartn_y\n\
         _atom_site.Cartn_z\n\
         _atom_site.B_iso_or_equiv\n\
         _atom_site.pdbx_PDB_model_num\n",
    );
    let mut atom_id = 1usize;
    for (idx, r) in ubq_xray.residues.iter().take(16).enumerate() {
        let seq_id = (idx + 1) as i32;
        for aname in ["N", "CA", "C", "O", "CB"] {
            if let Some(&xyz) = r.atoms.get(aname) {
                cif.push_str(&format!(
                    "ATOM {} {} . {} A {} {:.3} {:.3} {:.3} 50.00 1\n",
                    atom_id, aname, r.res_name, seq_id, xyz[0], xyz[1], xyz[2]
                ));
                atom_id += 1;
            }
        }
    }
    cif.push_str(
        "#\nloop_\n_ma_qa_metric_local.label_asym_id\n_ma_qa_metric_local.label_seq_id\n_ma_qa_metric_local.metric_id\n_ma_qa_metric_local.metric_value\n",
    );
    for idx in 0..16 {
        let plddt = if idx < 12 { 92.5 } else { 0.0 };
        cif.push_str(&format!("A {} 1 {:.2}\n", idx + 1, plddt));
    }

    let af_model = StructureModel::parse_cif_str(&cif, false).expect("parse AF mmCIF");
    assert_eq!(
        af_model.provenance,
        StructCoordinateProvenance::Computational
    );
    assert_eq!(af_model.residues.len(), 16);
    assert_eq!(af_model.residues[0].plddt, Some(92.5));
    assert_eq!(af_model.residues[15].plddt, Some(0.0));
    // Missing amide protons MUST be deterministically reconstructed on heavy-atom-only `.cif`!
    assert!(af_model.residues[2].h.is_some());

    let rep_af = predict_structure_shifts(&af_model, &StructShiftConfig::default()).unwrap();
    assert!(rep_af.residues[4].w_af > 0.94);
    assert_eq!(rep_af.residues[15].w_af, 0.0);
    for b in rep_af.residues[15].backbone.values() {
        assert!((b.predicted_ppm - b.rc_ppm).abs() < 1e-12);
    }

    // 3. Hostile inputs must fail cleanly without panicking
    assert!(StructureModel::parse_cif_str(
        "data_bad\nloop_\n_atom_site.Cartn_x\n_atom_site.Cartn_y\n1.0\n",
        false
    )
    .is_err());
    assert!(StructureModel::parse_pdb_str(
        "ATOM      1  CA  ALA A   1       NaN   1.000   2.000  1.00 20.00           C\n",
        false
    )
    .is_err());
    assert!(StructureModel::parse_pdb_str("", false).is_err());
}

#[test]
fn test_heavy_atom_proton_reconstruction_ph_temp_and_export_roundtrips() {
    let root = workspace_root();
    let raw_1d3z = fs::read_to_string(root.join("data/pdb_refs/1d3z.pdb")).unwrap();

    // Strip all hydrogen records from 1D3Z Model 1 to mimic heavy-atom X-ray / AF coordinates
    let heavy_only_pdb: String = raw_1d3z
        .lines()
        .take_while(|l| !l.starts_with("ENDMDL"))
        .filter(|l| {
            if !l.starts_with("ATOM  ") {
                return true;
            }
            let aname = l[12..16].trim();
            !aname.starts_with('H')
                && !aname
                    .bytes()
                    .next()
                    .is_some_and(|b| b.is_ascii_digit() && aname.contains('H'))
        })
        .collect::<Vec<_>>()
        .join("\n");

    let full_model = StructureModel::parse_pdb_str(&raw_1d3z, false).unwrap();
    let heavy_model = StructureModel::parse_pdb_str(&heavy_only_pdb, false).unwrap();

    let cfg = StructShiftConfig::default().with_methyl_ilvmat(true);
    let rep_full = predict_structure_shifts(&full_model, &cfg).unwrap();
    let rep_heavy = predict_structure_shifts(&heavy_model, &cfg).unwrap();

    let mut h_diffs_sq = 0.0;
    let mut h_cnt = 0usize;
    for (rf, rh) in rep_full.residues.iter().zip(rep_heavy.residues.iter()) {
        if let (Some(bf), Some(bh)) = (rf.backbone.get("H"), rh.backbone.get("H")) {
            let d = bf.predicted_ppm - bh.predicted_ppm;
            h_diffs_sq += d * d;
            h_cnt += 1;
        }
    }
    let h_rec_rmsd = (h_diffs_sq / (h_cnt as f64)).sqrt();
    assert!(
        h_rec_rmsd < 0.12,
        "NeRF amide proton reconstruction shift RMSD vs NMR protons too large: {h_rec_rmsd:.4} ppm"
    );

    // Temperature & pH coil sensitivity checks
    let rep_cold = predict_structure_shifts(
        &heavy_model,
        &StructShiftConfig::default().with_temperature_kelvin(278.15),
    )
    .unwrap();
    let rep_warm = predict_structure_shifts(
        &heavy_model,
        &StructShiftConfig::default().with_temperature_kelvin(318.15),
    )
    .unwrap();
    // 40 K increase shifts amide 1H^N by 40 * (-0.0046) = -0.184 ppm
    let dh_temp = rep_warm.residues[5].backbone["H"].predicted_ppm
        - rep_cold.residues[5].backbone["H"].predicted_ppm;
    assert!(
        (dh_temp - (-0.184)).abs() < 0.01,
        "expected -0.184 ppm Kjaergaard 40 K amide shift, got {dh_temp:.4}"
    );

    let rep_acid =
        predict_structure_shifts(&heavy_model, &StructShiftConfig::default().with_ph(2.5)).unwrap();
    let asp_idx = rep_full
        .residues
        .iter()
        .position(|r| r.res_name == "ASP")
        .unwrap();
    let dcb_acid = rep_acid.residues[asp_idx].backbone["CB"].predicted_ppm
        - rep_full.residues[asp_idx].backbone["CB"].predicted_ppm;
    assert!(
        dcb_acid < -2.2,
        "acidic pH 2.5 must shift ASP CB upfield by > 2.2 ppm (Platzer 2014), got {dcb_acid:.3}"
    );

    // Round-trip exports: NMR-STAR v3, NEF v1.1, TSV, JSON
    let star_s = rep_full.to_nmrstar_str();
    assert!(star_s.contains("# exclusion_list_sha256:"));
    let parsed_star = read_star_str(&star_s, None).unwrap();
    assert_eq!(parsed_star.sequence.len(), 76);
    assert!(parsed_star.shifts.contains_key(&(49, "CA".to_string())));
    assert!(star_s.contains("CD1"));

    let nef_s = rep_full.to_nef_str("ubq_pred");
    assert!(nef_s.contains("# exclusion_list_sha256:"));
    let parsed_nef = read_nef_str(&nef_s).unwrap();
    assert_eq!(parsed_nef.shift_lists.len(), 1);
    assert!(parsed_nef.shift_lists[0]
        .shifts
        .iter()
        .any(|s| s.atom_name == "HD1%" && s.value_uncertainty.is_some()));

    let tsv_s = rep_full.to_tsv_str();
    assert!(tsv_s.contains(&rep_full.exclusion_list_sha256));
    assert!(tsv_s.contains("delta_ring_current_ppm"));

    let json_s = rep_full.to_json_str().unwrap();
    assert!(json_s.contains(&rep_full.exclusion_list_sha256));
    assert!(json_s.contains("\"max_retained_seq_identity\""));
}

#[test]
fn test_ilvmat_methyl_gamma_gauche_rotamer_physics() {
    use nmr_core::methyl_trosy::MethylGroupKind;
    use nmr_core::structure::nerf_place_atom;

    let root = workspace_root();
    let model_1d3z =
        StructureModel::parse_file(root.join("data/pdb_refs/1d3z.pdb"), false).unwrap();
    let rep_ubq = predict_structure_shifts(
        &model_1d3z,
        &StructShiftConfig::default().with_methyl_ilvmat(true),
    )
    .unwrap();

    // Verify all 6 ILVMAT residue classes (Ile, Leu, Val, Met, Ala, Thr) emit methyl predictions on Ubiquitin
    for kind in MethylGroupKind::ALL {
        assert!(
            rep_ubq
                .residues
                .iter()
                .flat_map(|r| r.methyls.iter())
                .any(|m| m.kind == kind),
            "Ubiquitin ILVMAT prediction missing MethylGroupKind::{kind:?}"
        );
    }

    // Verify Sahakyan (2011) Ile CD1 gamma-gauche ~ -4.2 ppm upfield shielding when chi2 flips from trans (180 deg) to gauche- (-60 deg)
    let ile_idx = model_1d3z
        .residues
        .iter()
        .position(|r| r.res_name == "ILE" && r.atoms.contains_key("CG1"))
        .unwrap();
    let mut m_trans = model_1d3z.clone();
    let mut m_gauche = model_1d3z;
    let ca = m_trans.residues[ile_idx].ca;
    let cb = m_trans.residues[ile_idx].cb.unwrap();
    let cg1 = m_trans.residues[ile_idx].atoms["CG1"];
    let cd1_trans = nerf_place_atom(
        ca,
        cb,
        cg1,
        1.516,
        113.8_f64.to_radians(),
        std::f64::consts::PI,
    );
    let cd1_gm = nerf_place_atom(
        ca,
        cb,
        cg1,
        1.516,
        113.8_f64.to_radians(),
        (-60.0_f64).to_radians(),
    );
    m_trans.residues[ile_idx]
        .atoms
        .insert("CD1".to_string(), cd1_trans);
    m_gauche.residues[ile_idx]
        .atoms
        .insert("CD1".to_string(), cd1_gm);

    let cfg_no_rc = StructShiftConfig::default()
        .with_methyl_ilvmat(true)
        .with_ring_current(false);
    let r_t = predict_structure_shifts(&m_trans, &cfg_no_rc).unwrap();
    let r_g = predict_structure_shifts(&m_gauche, &cfg_no_rc).unwrap();
    let c_trans = r_t.residues[ile_idx].methyls[0].c_ppm;
    let c_gm = r_g.residues[ile_idx].methyls[0].c_ppm;
    assert!(
        c_trans - c_gm > 3.5,
        "Ile CD1 trans ({c_trans:.2} ppm) vs g- ({c_gm:.2} ppm) must exhibit > 3.5 ppm gamma-gauche shielding"
    );
}

#[test]
fn test_p1_4_layer4_tm_score_firewall_never_bypassed_by_short_db_chain_or_short_query() {
    let root = workspace_root();
    let model_1d3z =
        StructureModel::parse_file(root.join("data/pdb_refs/1d3z.pdb"), false).unwrap();

    // 1. Short query (< 8 CA atoms) MUST fail closed (`Err`) so Layer 4 (`TM-score < 0.50`) can
    //    never be silently bypassed!
    let mut short_query = model_1d3z.clone();
    short_query.residues.truncate(5);
    let err_short_q =
        predict_structure_shifts(&short_query, &StructShiftConfig::default()).unwrap_err();
    assert!(
        err_short_q.contains("Layer 4") && err_short_q.contains(">= 8 CA atoms"),
        "expected fail-closed Layer 4 error on 5-residue query, got: {err_short_q}"
    );

    // 2. Database containing a 5-residue short chain (`SHORT5_A`, `L = 5 < 8`) PLUS a disguised
    //    76-residue Ubiquitin-fold decoy (`DECOY_UBQ_Z`, exact `1D3Z` CA trace => `TM-score = 1.0`,
    //    but all-Alanine sequence `< 10%` seq-id and unrelated IDs so ONLY Layer 4 catches it!):
    //    - On `6328a1e`, `SHORT5_A` set `eff_query.query_ca_coords = None` for the ENTIRE database,
    //      letting `DECOY_UBQ_Z` leak straight into training!
    //    - Now ONLY `SHORT5_A` is skipped (`l4_short_templates_skipped == 1`), and `DECOY_UBQ_Z`
    //      is caught and excluded by Layer 4 (`TM-score >= 0.50`)!
    let mut custom_chains = TorsionDb::embedded().chains.clone();
    let short_chain = TorsionCorpusChain {
        chain_key: "SHORT5_A".to_string(),
        pdb_id: "9ZZ1".to_string(),
        chain_id: "A".to_string(),
        bmrb_id: "99991".to_string(),
        uniprot_id: "Q00001".to_string(),
        family_id: "9.99.99.1".to_string(),
        topology_id: "9.99.99".to_string(),
        seq1: b"AAAAA".to_vec(),
        residues: (0..5_i32)
            .map(|k| TorsionCorpusResidue {
                seq_id: k + 1,
                res_name: "ALA".to_string(),
                aa1: b'A',
                sec_shifts: [Some(0.0); 6],
                phi_deg: Some(-62.0),
                psi_deg: Some(-42.0),
                chi1_deg: None,
                ca_coord: [(k as f64) * 3.8, 0.0, 0.0],
            })
            .collect(),
    };
    let decoy_ubq_fold = TorsionCorpusChain {
        chain_key: "DECOY_UBQ_Z".to_string(),
        pdb_id: "9ZZ2".to_string(),
        chain_id: "Z".to_string(),
        bmrb_id: "99992".to_string(),
        uniprot_id: "Q00002".to_string(),
        family_id: "9.88.88.1".to_string(),
        topology_id: "9.88.88".to_string(),
        seq1: vec![b'A'; model_1d3z.residues.len()],
        residues: model_1d3z
            .residues
            .iter()
            .enumerate()
            .map(|(k, r)| TorsionCorpusResidue {
                seq_id: (k + 1) as i32,
                res_name: "ALA".to_string(),
                aa1: b'A',
                sec_shifts: [Some(0.0); 6],
                phi_deg: r.phi_deg,
                psi_deg: r.psi_deg,
                chi1_deg: None,
                ca_coord: r.ca,
            })
            .collect(),
    };
    custom_chains.push(short_chain);
    custom_chains.push(decoy_ubq_fold);
    let custom_db = TorsionDb {
        chains: custom_chains,
    };

    let rep = predict_structure_shifts_with_db(
        &custom_db,
        &model_1d3z,
        &HomologFirewallQuery::default(),
        &StructShiftConfig::default(),
    )
    .unwrap();
    assert_eq!(
        rep.l4_short_templates_skipped, 1,
        "expected exactly 1 short (<8 CA) template skipped"
    );
    assert!(
        rep.excluded_template_ids.iter().any(|id| id == "DECOY_UBQ_Z"),
        "Layer 4 TM-score firewall failed to exclude disguised same-fold decoy DECOY_UBQ_Z! Excluded: {:?}",
        rep.excluded_template_ids
    );
    let decoy_detail = rep
        .excluded_details
        .iter()
        .find(|d| d.chain_key == "DECOY_UBQ_Z")
        .expect("DECOY_UBQ_Z detail present");
    assert!(
        decoy_detail
            .reasons
            .iter()
            .any(|r| r.contains("L4:tm_score")),
        "expected L4:tm_score exclusion reason on DECOY_UBQ_Z, got: {:?}",
        decoy_detail.reasons
    );
}

#[test]
fn test_p2_1_predictive_sigma_strictly_increases_at_low_plddt_and_uses_oos_residuals() {
    let root = workspace_root();
    let model_1d3z =
        StructureModel::parse_file(root.join("data/pdb_refs/1d3z.pdb"), false).unwrap();

    // Evaluate across monotonically decreasing pLDDT levels `[95, 80, 70, 50, 25, 0]`:
    // On `6328a1e`, `coil_sigma_ppm < struct_sigma_ppm` (`15N: 1.45 < 2.45`), causing predictive
    // sigma to SHRINK as pLDDT dropped! Now predictive sigma must monotonically INCREASE (never shrink)
    // across every residue class (`General`, `Gly`, `Pro`, `PrePro`, `BetaBranched`) and every nucleus!
    let plddt_grid = [95.0, 80.0, 70.0, 50.0, 25.0, 0.0];
    let reports: Vec<_> = plddt_grid
        .iter()
        .map(|&pl| {
            predict_structure_shifts(
                &model_1d3z,
                &StructShiftConfig::default()
                    .with_methyl_ilvmat(true)
                    .with_uniform_plddt(pl),
            )
            .unwrap()
        })
        .collect();

    for res_idx in 0..model_1d3z.residues.len() {
        for atom in BackboneShiftAtom::ALL {
            let aname = atom.as_str();
            if !reports[0].residues[res_idx].backbone.contains_key(aname) {
                continue;
            }
            for w in 0..(plddt_grid.len() - 1) {
                let sig_hi_plddt = reports[w].residues[res_idx].backbone[aname].predicted_sigma_ppm;
                let sig_lo_plddt =
                    reports[w + 1].residues[res_idx].backbone[aname].predicted_sigma_ppm;
                assert!(
                    sig_lo_plddt + 1.0e-12 >= sig_hi_plddt,
                    "predictive sigma shrank at residue {} atom {} when pLDDT dropped {} -> {}: {:.4} -> {:.4}",
                    res_idx + 1,
                    aname,
                    plddt_grid[w],
                    plddt_grid[w + 1],
                    sig_hi_plddt,
                    sig_lo_plddt
                );
            }
            let sig_95 = reports[0].residues[res_idx].backbone[aname].predicted_sigma_ppm;
            let sig_25 = reports[4].residues[res_idx].backbone[aname].predicted_sigma_ppm;
            assert!(
                sig_25 > sig_95 + 0.02,
                "predictive sigma at pLDDT=25 ({sig_25:.4}) must strictly exceed pLDDT=95 ({sig_95:.4}) on residue {} atom {}",
                res_idx + 1,
                aname
            );
        }
        for m_idx in 0..reports[0].residues[res_idx].methyls.len() {
            for w in 0..(plddt_grid.len() - 1) {
                let c_hi = reports[w].residues[res_idx].methyls[m_idx].c_sigma_ppm;
                let c_lo = reports[w + 1].residues[res_idx].methyls[m_idx].c_sigma_ppm;
                let h_hi = reports[w].residues[res_idx].methyls[m_idx].h_sigma_ppm;
                let h_lo = reports[w + 1].residues[res_idx].methyls[m_idx].h_sigma_ppm;
                assert!(c_lo + 1.0e-12 >= c_hi);
                assert!(h_lo + 1.0e-12 >= h_hi);
            }
        }
    }
}

#[test]
fn test_p2_3_harmonized_training_vs_query_hbond_geometry_on_1ubq() {
    let root = workspace_root();
    let ubq_pdb =
        StructureModel::parse_file(root.join("data/fixtures/ubiquitin_1ubq_xray.pdb"), false)
            .unwrap();
    let ubq_corpus = TorsionDb::embedded()
        .chains
        .iter()
        .find(|c| c.chain_key == "1UBQ_A")
        .expect("1UBQ_A present in embedded TorsionDb");

    let seq_ids: Vec<i32> = ubq_corpus.residues.iter().map(|r| r.seq_id).collect();
    let res_names: Vec<String> = ubq_corpus
        .residues
        .iter()
        .map(|r| r.res_name.clone())
        .collect();
    let ca_coords: Vec<[f64; 3]> = ubq_corpus.residues.iter().map(|r| r.ca_coord).collect();
    let phi: Vec<Option<f64>> = ubq_corpus.residues.iter().map(|r| r.phi_deg).collect();
    let psi: Vec<Option<f64>> = ubq_corpus.residues.iter().map(|r| r.psi_deg).collect();
    let chi1: Vec<Option<f64>> = ubq_corpus.residues.iter().map(|r| r.chi1_deg).collect();
    let ubq_rebuilt = StructureModel::from_ca_and_torsions(
        "A", &seq_ids, &res_names, &ca_coords, &phi, &psi, &chi1,
    )
    .unwrap();

    let hb_query = evaluate_hbond_shifts_ppm(&ubq_pdb);
    let hb_train = evaluate_hbond_shifts_ppm(&ubq_rebuilt);
    assert_eq!(hb_query.len(), 76);
    assert_eq!(hb_train.len(), 76);

    // On `6328a1e`, 15N H-bond mismatch between `1ubq.pdb` and `1UBQ_A` was `0.154 ppm` RMS
    // (`0.89 ppm` max) due to `OD1/OD2/OE1/OE2` sidechain oxygens and crystallographic bond geometry!
    // After canonicalization (`21.W1:P2-3`), residual difference is solely from 2-decimal TSV rounding (< 0.02 ppm)!
    let n_idx = BackboneShiftAtom::N.index();
    let mut sq_sum = 0.0_f64;
    let mut max_abs = 0.0_f64;
    for i in 0..76 {
        let diff = (hb_query[i][n_idx] - hb_train[i][n_idx]).abs();
        sq_sum += diff * diff;
        max_abs = max_abs.max(diff);
    }
    let rms_n_hb = (sq_sum / 76.0).sqrt();
    println!(
        "1UBQ query vs corpus training H-bond 15N mismatch: RMS = {rms_n_hb:.5} ppm, max = {max_abs:.5} ppm"
    );
    assert!(
        rms_n_hb < 0.02 && max_abs < 0.08,
        "expected harmonized H-bond 15N mismatch RMS < 0.02 ppm & max < 0.08 ppm, got RMS={rms_n_hb:.4}, max={max_abs:.4}"
    );
}

#[test]
fn test_p0_2_require_predicted_structure_model_fail_closed() {
    let root = workspace_root();
    let raw_1ubq = fs::read_to_string(root.join("data/fixtures/ubiquitin_1ubq_xray.pdb")).unwrap();
    let m_xray = StructureModel::parse_pdb_str(&raw_1ubq, false).unwrap();
    assert!(require_predicted_structure_model(&m_xray, false).is_err());

    // Headerless 1UBQ (`B <= 100`) + REMARK ALPHAFOLD must remain `Unknown` with `plddt = None` and fail closed!
    let stripped: String = raw_1ubq
        .lines()
        .filter(|l| l.starts_with("ATOM  ") || l.starts_with("END"))
        .collect::<Vec<_>>()
        .join("\n");
    let m_stripped =
        StructureModel::parse_pdb_str(&format!("REMARK ALPHAFOLD\n{stripped}"), false).unwrap();
    assert_eq!(m_stripped.provenance, StructCoordinateProvenance::Unknown);
    assert!(m_stripped.residues.iter().all(|r| r.plddt.is_none()));
    assert!(require_predicted_structure_model(&m_stripped, false).is_err());

    // QMEANDisCo mmCIF must be rejected as pLDDT (`21.W1:P2-6`)
    let qmean_cif = "\
data_QM
_ma_protocol_step.step_id 1
loop_
_ma_qa_metric.id
_ma_qa_metric.name
_ma_qa_metric.mode
1 QMEANDisCo local
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
ATOM 1 N ALA A 1 0.0 0.0 0.0
ATOM 2 CA ALA A 1 1.46 0.0 0.0
ATOM 3 C ALA A 1 2.0 1.4 0.0
ATOM 4 N GLY A 2 3.2 1.2 0.0
ATOM 5 CA GLY A 2 4.5 1.8 0.0
ATOM 6 C GLY A 2 5.2 0.6 0.0
#
loop_
_ma_qa_metric_local.label_asym_id
_ma_qa_metric_local.label_seq_id
_ma_qa_metric_local.metric_id
_ma_qa_metric_local.metric_value
A 1 1 0.80
A 2 1 0.72
#
";
    let m_qm = StructureModel::parse_cif_str(qmean_cif, false).unwrap();
    assert!(m_qm.residues.iter().all(|r| r.plddt.is_none()));
    assert!(require_predicted_structure_model(&m_qm, false).is_err());

    // Real committed AFDB v6 `AF-P0CG48-F1-model_v6.cif` must pass `require_predicted_structure_model`
    let m_afdb =
        StructureModel::parse_file(root.join("data/afdb/AF-P0CG48-F1-model_v6.cif"), false)
            .unwrap();
    assert_eq!(m_afdb.provenance, StructCoordinateProvenance::Computational);
    assert_eq!(m_afdb.residues.len(), 685);
    assert!(require_predicted_structure_model(&m_afdb, false).is_ok());
}

fn bootstrap_rmse_ci95(sq_errors: &[f64], n_boot: usize) -> [f64; 2] {
    let n = sq_errors.len();
    assert!(n >= 10);
    let mut state: u64 = 0xD1B5_4A32_D192_ED03_u64 ^ ((n as u64) << 19);
    let mut next_u64 = || -> u64 {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        state.wrapping_mul(0x2545_F491_4F6C_DD1D_u64)
    };
    let mut samples = Vec::with_capacity(n_boot);
    for _ in 0..n_boot {
        let mut s = 0.0_f64;
        for _ in 0..n {
            let idx = (next_u64() as usize) % n;
            s += sq_errors[idx];
        }
        samples.push((s / (n as f64)).sqrt());
    }
    samples.sort_by(f64::total_cmp);
    let lo = samples[((0.025 * ((n_boot - 1) as f64)).floor() as usize).min(n_boot - 1)];
    let hi = samples[((0.975 * ((n_boot - 1) as f64)).ceil() as usize).min(n_boot - 1)];
    [lo, hi]
}

#[test]
fn test_a4_deduplicated_8_protein_dev_set_3d_vs_sequence_only_baseline_with_bootstrap_ci() {
    // `A4` & `21.W1:P1-3`: Deduplicated folded dev-set evaluation (`8` unique folded proteins in
    // `manifest.tsv` — skipping `bmr5387` so Human Ubiquitin `P0CG48` is counted ONCE via `bmr17769`!),
    // comparing sequence-only baseline (`ncIDP` random coil, `w_AF = 0`) vs `struct_shifts` 3D model
    // with bootstrap 95% confidence intervals across all 6 backbone nuclei.
    let root = workspace_root();
    let manifest_text = fs::read_to_string(root.join("data/benchmark/manifest.tsv")).unwrap();
    let bench_refs = load_benchmark_refs();

    let mut sq_3d: [Vec<f64>; 6] = std::array::from_fn(|_| Vec::new());
    let mut sq_seq: [Vec<f64>; 6] = std::array::from_fn(|_| Vec::new());
    let mut unique_folded = 0usize;

    for (line_idx, line) in manifest_text.lines().enumerate() {
        if line_idx == 0 || line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        let bmrb_id = cols[0];
        // Deduplicate Ubiquitin (`bmr5387` 1G6J duplicates `bmr17769` 1D3Z on Human Ubiquitin P0CG48!)
        if bmrb_id == "bmr5387" {
            continue;
        }
        let rel_path = cols[5];
        let bref = &bench_refs[bmrb_id];
        if !bref.is_folded {
            continue;
        }
        unique_folded += 1;

        let star_path = match bmrb_id {
            "bmr16649" => root.join("data/fixtures/bmr16649_pcp_aromatic.str"),
            "bmr17769" => root.join("data/fixtures/bmr17769_ubiquitin_allatom.str"),
            _ => root.join(rel_path),
        };
        let obs_ref = read_star_all_atoms(&star_path, None).unwrap();
        let model = load_folded_benchmark_structure(&root, bref);
        let fw = HomologFirewallQuery {
            query_pdb_ids: vec![bref.pdb_id.clone()],
            query_bmrb_ids: vec![bref.bmrb_id.clone()],
            query_uniprot_id: bref.uniprot_id.clone(),
            query_family_id: bref.family_id.clone(),
            query_topology_id: bref.topology_id.clone(),
            query_ca_coords: Some(model.residues.iter().map(|r| r.ca).collect()),
        };

        let rep_3d =
            predict_structure_shifts_with_firewall(&model, &fw, &StructShiftConfig::default())
                .unwrap();
        let rep_seq = predict_structure_shifts_with_firewall(
            &model,
            &fw,
            &StructShiftConfig::default().with_uniform_plddt(0.0),
        )
        .unwrap();
        let off_3d = estimate_unlabelled_shift_offsets(&obs_ref, &rep_3d);
        let off_seq = estimate_unlabelled_shift_offsets(&obs_ref, &rep_seq);

        let p_keys: Vec<(i32, &str)> = rep_3d
            .residues
            .iter()
            .map(|r| (r.seq_id, r.res_name.as_str()))
            .collect();
        let t_keys: Vec<(i32, &str)> = obs_ref
            .seq_ids
            .iter()
            .zip(obs_ref.sequence.iter())
            .map(|(&s, rn)| (s, rn.as_str()))
            .collect();
        for (pi, ti) in align_by_res_names(&p_keys, &t_keys) {
            if pi == 0 || pi + 1 == rep_3d.residues.len() {
                continue;
            }
            for atom in BackboneShiftAtom::ALL {
                let a_idx = atom.index();
                let aname = atom.as_str();
                let obs_opt = obs_ref
                    .shifts
                    .get(&(ti, aname.to_string()))
                    .copied()
                    .or_else(|| {
                        (aname == "HA").then(|| {
                            match (
                                obs_ref.shifts.get(&(ti, "HA2".to_string())).copied(),
                                obs_ref.shifts.get(&(ti, "HA3".to_string())).copied(),
                            ) {
                                (Some(a), Some(b)) => Some(0.5 * (a + b)),
                                (Some(a), None) => Some(a),
                                (None, Some(b)) => Some(b),
                                (None, None) => None,
                            }
                        })?
                    });
                if let (Some(obs_v), Some(b3), Some(bs)) = (
                    obs_opt,
                    rep_3d.residues[pi].backbone.get(aname),
                    rep_seq.residues[pi].backbone.get(aname),
                ) {
                    let e3 = (obs_v - off_3d[a_idx]) - b3.predicted_ppm;
                    let es = (obs_v - off_seq[a_idx]) - bs.predicted_ppm;
                    let typo_cap = match atom {
                        BackboneShiftAtom::H | BackboneShiftAtom::HA => 3.5,
                        BackboneShiftAtom::N => 16.0,
                        _ => 10.0,
                    };
                    if e3.abs() <= typo_cap && es.abs() <= typo_cap {
                        sq_3d[a_idx].push(e3 * e3);
                        sq_seq[a_idx].push(es * es);
                    }
                }
            }
        }
    }

    // Also include the 3 unique non-Ubiquitin HA fixtures (`GB1 1PGA`, `KH3 1ZZK`, `Calbindin 1KSM`) for HA
    for (pdb_rel, star_rel, fw_extra) in [
        (
            "data/pdb_refs/1pga.pdb",
            "data/fixtures/gb1_bmr7280_shifts.str",
            HomologFirewallQuery {
                query_pdb_ids: vec!["1PGA".into(), "3GB1".into()],
                query_bmrb_ids: vec!["7280".into()],
                query_topology_id: Some("3.10.20".into()),
                ..Default::default()
            },
        ),
        (
            "data/fixtures/kh3_1zzk_xray.pdb",
            "data/fixtures/kh3_bmr4405_shifts.str",
            HomologFirewallQuery {
                query_pdb_ids: vec!["1ZZK".into(), "1KHM".into()],
                query_bmrb_ids: vec!["4405".into()],
                ..Default::default()
            },
        ),
        (
            "data/fixtures/calbindin_1ksm_nmr.pdb",
            "data/fixtures/bmr6699_calbindin_la3.str",
            HomologFirewallQuery {
                query_pdb_ids: vec!["1KSM".into()],
                query_bmrb_ids: vec!["6699".into(), "15594".into()],
                query_topology_id: Some("1.10.238".into()),
                ..Default::default()
            },
        ),
    ] {
        let m_ex = StructureModel::parse_file(root.join(pdb_rel), false).unwrap();
        let o_ex = read_star_all_atoms(root.join(star_rel), None).unwrap();
        let r3 =
            predict_structure_shifts_with_firewall(&m_ex, &fw_extra, &StructShiftConfig::default())
                .unwrap();
        let rs = predict_structure_shifts_with_firewall(
            &m_ex,
            &fw_extra,
            &StructShiftConfig::default().with_uniform_plddt(0.0),
        )
        .unwrap();
        let off3 = estimate_unlabelled_shift_offsets(&o_ex, &r3);
        let offs = estimate_unlabelled_shift_offsets(&o_ex, &rs);
        let pk: Vec<(i32, &str)> = r3
            .residues
            .iter()
            .map(|r| (r.seq_id, r.res_name.as_str()))
            .collect();
        let tk: Vec<(i32, &str)> = o_ex
            .seq_ids
            .iter()
            .zip(o_ex.sequence.iter())
            .map(|(&s, rn)| (s, rn.as_str()))
            .collect();
        for (pi, ti) in align_by_res_names(&pk, &tk) {
            if pi == 0 || pi + 1 == r3.residues.len() {
                continue;
            }
            let obs_ha = o_ex
                .shifts
                .get(&(ti, "HA".to_string()))
                .copied()
                .or_else(|| {
                    match (
                        o_ex.shifts.get(&(ti, "HA2".to_string())).copied(),
                        o_ex.shifts.get(&(ti, "HA3".to_string())).copied(),
                    ) {
                        (Some(a), Some(b)) => Some(0.5 * (a + b)),
                        (Some(a), None) => Some(a),
                        (None, Some(b)) => Some(b),
                        (None, None) => None,
                    }
                });
            if let (Some(ov), Some(b3), Some(bs)) = (
                obs_ha,
                r3.residues[pi].backbone.get("HA"),
                rs.residues[pi].backbone.get("HA"),
            ) {
                let e3 = (ov - off3[5]) - b3.predicted_ppm;
                let es = (ov - offs[5]) - bs.predicted_ppm;
                if e3.abs() <= 3.5 && es.abs() <= 3.5 {
                    sq_3d[5].push(e3 * e3);
                    sq_seq[5].push(es * es);
                }
            }
        }
    }

    assert_eq!(
        unique_folded, 8,
        "expected 8 deduplicated unique folded proteins (Ubiquitin counted once)"
    );

    println!(
        "\n=== DEDUPLICATED 8-PROTEIN DEV-SET: SEQUENCE-ONLY (ncIDP) vs 3D STRUCT_SHIFTS (WITH 95% BOOTSTRAP CI) ==="
    );
    for atom in BackboneShiftAtom::ALL {
        let a = atom.index();
        let n = sq_3d[a].len();
        let rmse_3d = (sq_3d[a].iter().sum::<f64>() / (n as f64)).sqrt();
        let rmse_seq = (sq_seq[a].iter().sum::<f64>() / (n as f64)).sqrt();
        let ci_3d = bootstrap_rmse_ci95(&sq_3d[a], 1000);
        let ci_seq = bootstrap_rmse_ci95(&sq_seq[a], 1000);
        let gain_pct = 100.0 * (rmse_seq - rmse_3d) / rmse_seq;
        println!(
            "  {:<3} (n={:>3}): seq-only = {:.3} [{:.3}, {:.3}] ppm -> 3D struct_shifts = {:.3} [{:.3}, {:.3}] ppm ({:+.1}% gain)",
            atom.as_str(),
            n,
            rmse_seq,
            ci_seq[0],
            ci_seq[1],
            rmse_3d,
            ci_3d[0],
            ci_3d[1],
            gain_pct
        );
        assert!(
            rmse_3d < rmse_seq,
            "3D struct_shifts ({rmse_3d:.3}) must beat sequence-only ncIDP baseline ({rmse_seq:.3}) on {}",
            atom.as_str()
        );
    }
}
