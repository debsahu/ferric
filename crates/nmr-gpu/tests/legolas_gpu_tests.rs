//! WebGPU (`wgpu` + WGSL) `LEGOLAS` ANI-AEV & ensemble MLP numerical parity tests against
//! `nmr_core::legolas_shifts` CPU `f64` reference (`T22.12`).
//!
//! All tests in this file are `[synthetic self-consistency]` numerical parity validations.

use std::collections::BTreeMap;
use std::path::PathBuf;

use nmr_core::legolas_shifts::{
    compute_aevs_for_indices, extract_atoms_from_structure_model, train_nucleus_ensemble,
    ActivationKind, AevConfig, LegolasAtom, LegolasError, LegolasMlp, LegolasModel,
    LegolasPredictOptions, LegolasTrainConfig, NucleusEnsemble, Species, TargetNucleus,
    MAX_LEGOLAS_ATOMS, MAX_LEGOLAS_NEIGHBORS_PER_ATOM, UNTRAINED_WEIGHTS_PROVENANCE,
};
use nmr_core::struct_shifts::StructureModel;
use nmr_gpu::legolas_gpu::LegolasGpuEngine;
use nmr_gpu::GpuEngine;

fn fixture_path(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("data/fixtures")
        .join(rel)
}

fn rodrigues_rotation(axis: [f64; 3], angle_rad: f64) -> [[f64; 3]; 3] {
    let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    let (kx, ky, kz) = (axis[0] / norm, axis[1] / norm, axis[2] / norm);
    let c = angle_rad.cos();
    let s = angle_rad.sin();
    let v = 1.0 - c;
    [
        [c + kx * kx * v, kx * ky * v - kz * s, kx * kz * v + ky * s],
        [ky * kx * v + kz * s, c + ky * ky * v, ky * kz * v - kx * s],
        [kz * kx * v - ky * s, kz * ky * v + kx * s, c + kz * kz * v],
    ]
}

fn mat_vec(m: &[[f64; 3]; 3], p: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * p[0] + m[0][1] * p[1] + m[0][2] * p[2],
        m[1][0] * p[0] + m[1][1] * p[1] + m[1][2] * p[2],
        m[2][0] * p[0] + m[2][1] * p[1] + m[2][2] * p[2],
    ]
}

#[test]
fn test_legolas_gpu_aev_and_ensemble_mlp_parity_all_activations() {
    // [synthetic self-consistency] CPU f64 vs WebGPU f32 (< 1e-4) parity on real 1UBQ protein
    // coordinates across all 6 backbone nuclei and CELU / GELU / SiLU activations.
    let engine = match GpuEngine::new_headless() {
        Ok(e) => e,
        Err(err) => {
            eprintln!("Skipping WebGPU LEGOLAS test (no GPU adapter available): {err}");
            return;
        }
    };
    let legolas_gpu = LegolasGpuEngine::new(&engine);

    let pdb = fixture_path("ubiquitin_1ubq_xray.pdb");
    let struct_model = StructureModel::parse_file(&pdb, false).expect("parse 1UBQ PDB");
    let atoms = extract_atoms_from_structure_model(&struct_model).expect("extract 1UBQ atoms");
    assert!(atoms.len() > 600, "expected full 1UBQ all-atom structure");

    let cfg = AevConfig::compact();
    let query_indices: Vec<usize> = atoms
        .iter()
        .enumerate()
        .filter_map(|(i, a)| a.target_nucleus.map(|_| i))
        .take(120)
        .collect();
    assert_eq!(query_indices.len(), 120);

    // 1. Direct AEV descriptor parity (CPU f64 vs WebGPU f32)
    let cpu_aevs = compute_aevs_for_indices(&atoms, &query_indices, &cfg).expect("CPU AEVs");
    let gpu_aevs = legolas_gpu
        .compute_aevs_gpu(&atoms, &query_indices, &cfg)
        .expect("GPU AEVs");
    assert_eq!(cpu_aevs.len(), gpu_aevs.len());

    let mut max_aev_diff = 0.0_f64;
    for (ca, ga) in cpu_aevs.iter().zip(gpu_aevs.iter()) {
        assert_eq!(ca.len(), ga.len());
        for (&c, &g) in ca.iter().zip(ga.iter()) {
            assert!(g.is_finite(), "GPU AEV produced non-finite value");
            max_aev_diff = max_aev_diff.max((c - (g as f64)).abs());
        }
    }
    println!(
        "LEGOLAS AEV CPU f64 vs WebGPU f32 on 1UBQ ({} atoms, {} queries, {} dims): max_diff = {max_aev_diff:.3e}",
        atoms.len(),
        query_indices.len(),
        cfg.total_dim()
    );
    assert!(
        max_aev_diff < 1e-4,
        "GPU AEV max abs diff {max_aev_diff:.3e} >= 1e-4"
    );

    // 2. End-to-end AEV + Ensemble MLP parity for CELU, GELU, and SiLU across all 6 nuclei
    let test_opts = LegolasPredictOptions {
        allow_untrained_for_testing: true,
    };
    let activations = [
        ActivationKind::Celu { alpha: 0.1 },
        ActivationKind::Gelu,
        ActivationKind::Silu,
    ];
    for (idx, &act) in activations.iter().enumerate() {
        let model = LegolasModel::new_seeded_all_backbone(
            cfg.clone(),
            &[32, 16],
            act,
            4,
            2026_1200 + idx as u64,
        )
        .expect("seeded model");

        // Regression guard (22.12:P1-1 & Round-2 P2-1): default GPU predict_shifts_gpu AND
        // low-level predict_ensemble_aevs_gpu MUST fail closed on untrained weights!
        let err = legolas_gpu
            .predict_shifts_gpu(&model, &atoms)
            .expect_err("untrained model must fail closed on default predict_shifts_gpu");
        assert_eq!(
            err,
            LegolasError::UntrainedWeights {
                provenance: UNTRAINED_WEIGHTS_PROVENANCE.to_string()
            }
        );
        let err_ens = legolas_gpu
            .predict_ensemble_aevs_gpu(&model.ensembles[&TargetNucleus::Ca13], &gpu_aevs[0], 1)
            .expect_err("untrained ensemble must fail closed on predict_ensemble_aevs_gpu");
        assert_eq!(
            err_ens,
            LegolasError::UntrainedWeights {
                provenance: UNTRAINED_WEIGHTS_PROVENANCE.to_string()
            }
        );
        let opt_ens = legolas_gpu
            .predict_ensemble_aevs_gpu_with_options(
                &model.ensembles[&TargetNucleus::Ca13],
                &gpu_aevs[0],
                1,
                &test_opts,
            )
            .expect("explicit testing opt-in on predict_ensemble_aevs_gpu_with_options");
        assert_eq!(opt_ens.len(), 1);

        let cpu_preds = model
            .predict_atoms_with_options(&atoms, &test_opts)
            .expect("CPU predict_atoms");
        let gpu_preds = legolas_gpu
            .predict_atoms_gpu_with_options(&model, &atoms, &test_opts)
            .expect("GPU predict_atoms_gpu");
        assert_eq!(cpu_preds.len(), gpu_preds.len());
        assert!(
            cpu_preds.len() > 300,
            "expected >300 backbone nuclei on 1UBQ"
        );

        let mut max_mean_diff = 0.0_f64;
        let mut max_sigma_diff = 0.0_f64;
        let mut max_member_diff = 0.0_f64;
        for (cp, gp) in cpu_preds.iter().zip(gpu_preds.iter()) {
            assert_eq!(cp.atom_index, gp.atom_index);
            assert_eq!(cp.estimate.nucleus, gp.estimate.nucleus);
            max_mean_diff = max_mean_diff.max((cp.estimate.mean_ppm - gp.estimate.mean_ppm).abs());
            max_sigma_diff =
                max_sigma_diff.max((cp.estimate.sigma_ens_ppm - gp.estimate.sigma_ens_ppm).abs());
            for (&cm, &gm) in cp
                .estimate
                .member_predictions_ppm
                .iter()
                .zip(gp.estimate.member_predictions_ppm.iter())
            {
                max_member_diff = max_member_diff.max((cm - gm).abs());
            }
        }
        println!(
            "LEGOLAS {act:?} CPU vs WebGPU (1UBQ, {} predictions, E=4): mean_diff={max_mean_diff:.3e}, sigma_diff={max_sigma_diff:.3e}, member_diff={max_member_diff:.3e}",
            cpu_preds.len()
        );
        assert!(
            max_mean_diff < 1e-4,
            "GPU mean_ppm diff {max_mean_diff:.3e} >= 1e-4 for {act:?}"
        );
        assert!(
            max_sigma_diff < 1e-4,
            "GPU sigma_ens_ppm diff {max_sigma_diff:.3e} >= 1e-4 for {act:?}"
        );
        assert!(
            max_member_diff < 1e-4,
            "GPU member_ppm diff {max_member_diff:.3e} >= 1e-4 for {act:?}"
        );
    }
}

#[test]
fn test_legolas_gpu_se3_large_translation_and_rotation_parity() {
    // [synthetic self-consistency] Verify WebGPU centroid-recentering preserves < 1e-4 parity
    // with CPU f64 even under a 50,000 A translation + arbitrary SO(3) rotation.
    let engine = match GpuEngine::new_headless() {
        Ok(e) => e,
        Err(err) => {
            eprintln!("Skipping WebGPU LEGOLAS SE(3) test (no GPU adapter available): {err}");
            return;
        }
    };
    let legolas_gpu = LegolasGpuEngine::new(&engine);

    let base_atoms = vec![
        LegolasAtom::with_target(Species::N, [0.00, 0.00, 0.00], 1, "N", TargetNucleus::N15),
        LegolasAtom::with_target(Species::H, [-0.42, 0.91, 0.08], 1, "H", TargetNucleus::Hn),
        LegolasAtom::with_target(
            Species::C,
            [1.46, 0.04, -0.09],
            1,
            "CA",
            TargetNucleus::Ca13,
        ),
        LegolasAtom::with_target(
            Species::H,
            [1.79, -0.90, -0.48],
            1,
            "HA",
            TargetNucleus::Ha1,
        ),
        LegolasAtom::with_target(
            Species::C,
            [1.96, 1.14, -1.07],
            1,
            "CB",
            TargetNucleus::Cb13,
        ),
        LegolasAtom::with_target(Species::C, [2.11, 0.21, 1.25], 1, "C", TargetNucleus::Co13),
        LegolasAtom::new(Species::O, [1.56, 0.87, 2.15]),
        LegolasAtom::new(Species::S, [3.65, 1.48, -1.21]),
    ];

    let rot = rodrigues_rotation([-0.52, 0.64, 0.56], 2.17);
    let big_shift = [50_000.0, -35_000.0, 20_000.0];
    let moved_atoms: Vec<LegolasAtom> = base_atoms
        .iter()
        .map(|a| {
            let rp = mat_vec(&rot, a.position);
            let mut b = a.clone();
            b.position = [
                rp[0] + big_shift[0],
                rp[1] + big_shift[1],
                rp[2] + big_shift[2],
            ];
            b
        })
        .collect();

    let cfg = AevConfig::ani1_standard(); // Full 560-dim ANI-1 standard config
    let model = LegolasModel::new_seeded_all_backbone(
        cfg,
        &[24, 12],
        ActivationKind::Celu { alpha: 0.1 },
        3,
        9001,
    )
    .expect("model");

    let test_opts = LegolasPredictOptions {
        allow_untrained_for_testing: true,
    };
    let cpu_base = model
        .predict_atoms_with_options(&base_atoms, &test_opts)
        .expect("CPU base");
    let gpu_moved = legolas_gpu
        .predict_atoms_gpu_with_options(&model, &moved_atoms, &test_opts)
        .expect("GPU moved");

    assert_eq!(cpu_base.len(), gpu_moved.len());
    let mut max_diff = 0.0_f64;
    for (c, g) in cpu_base.iter().zip(gpu_moved.iter()) {
        max_diff = max_diff.max((c.estimate.mean_ppm - g.estimate.mean_ppm).abs());
    }
    assert!(
        max_diff < 1e-4,
        "GPU large-translation SE(3) parity diff {max_diff:.3e} >= 1e-4"
    );
}

#[test]
fn test_legolas_gpu_trained_bundle_and_heterogeneous_ensemble_parity() {
    // [synthetic self-consistency] Verify trained ensemble after LEGOLAS-V1 bundle roundtrip
    // and heterogeneous member architectures match CPU f64 to < 1e-4 on WebGPU.
    let engine = match GpuEngine::new_headless() {
        Ok(e) => e,
        Err(err) => {
            eprintln!(
                "Skipping WebGPU LEGOLAS trained/hetero test (no GPU adapter available): {err}"
            );
            return;
        }
    };
    let legolas_gpu = LegolasGpuEngine::new(&engine);

    let pdb = fixture_path("ubiquitin_1ubq_xray.pdb");
    let struct_model = StructureModel::parse_file(&pdb, false).expect("parse 1UBQ PDB");
    let mut atoms = extract_atoms_from_structure_model(&struct_model).expect("extract 1UBQ");
    atoms.truncate(220);

    let cfg = AevConfig::compact();
    let aev_dim = cfg.total_dim();
    let ca_indices: Vec<usize> = atoms
        .iter()
        .enumerate()
        .filter_map(|(i, a)| (a.target_nucleus == Some(TargetNucleus::Ca13)).then_some(i))
        .collect();
    let ca_aevs = compute_aevs_for_indices(&atoms, &ca_indices, &cfg).expect("CA AEVs");
    let targets: Vec<f64> = ca_aevs
        .iter()
        .enumerate()
        .map(|(i, x)| 54.0 + 0.5 * x.iter().take(10).sum::<f64>() + 0.1 * (i as f64))
        .collect();

    // Build a heterogeneous ensemble (member 0: [16, 1], member 1: [24, 12, 1])
    let m0 = LegolasMlp::new_seeded(aev_dim, &[16], 301).expect("m0");
    let m1 = LegolasMlp::new_seeded(aev_dim, &[24, 12], 302).expect("m1");
    let mut ens = NucleusEnsemble::from_members(
        TargetNucleus::Ca13,
        ActivationKind::Celu { alpha: 0.1 },
        55.0,
        4.0,
        vec![m0, m1],
    )
    .expect("hetero ensemble");
    assert!(!ens.is_homogeneous_architecture());
    assert!(!ens.is_trained());

    let ca_aevs_f32: Vec<f32> = ca_aevs.iter().flatten().map(|&v| v as f32).collect();
    assert!(matches!(
        legolas_gpu.predict_ensemble_aevs_gpu(&ens, &ca_aevs_f32, ca_aevs.len()),
        Err(LegolasError::UntrainedWeights { .. })
    ));

    let train_cfg = LegolasTrainConfig {
        learning_rate: 1e-2,
        ..LegolasTrainConfig::default()
    };
    let rep =
        train_nucleus_ensemble(&mut ens, &ca_aevs, &targets, 25, &train_cfg, 303).expect("train");
    assert!(rep.final_rmse_ppm < rep.initial_rmse_ppm);
    assert!(ens.is_trained());
    let gpu_ens_after = legolas_gpu
        .predict_ensemble_aevs_gpu(&ens, &ca_aevs_f32, ca_aevs.len())
        .expect("trained ensemble GPU predict");
    assert_eq!(gpu_ens_after.len(), ca_aevs.len());

    let mut ensembles = BTreeMap::new();
    ensembles.insert(TargetNucleus::Ca13, ens);
    let model = LegolasModel::from_trained_ensembles(
        cfg,
        ensembles,
        "synthetic-aev-oracle-parity-test [synthetic self-consistency; not BMRB empirical validation]",
    )
    .expect("trained model");

    // Roundtrip through both JSON and binary LEGOLAS-V1 bundles
    let json = model.to_json_bundle().expect("json bundle");
    let reloaded = LegolasModel::from_json_bundle(&json).expect("reload json");
    let bin = reloaded.to_binary_bundle().expect("bin bundle");
    let reloaded_bin = LegolasModel::from_binary_bundle(&bin).expect("reload bin");

    let cpu_preds = reloaded_bin.predict_shifts(&atoms).expect("CPU preds");
    let gpu_preds = legolas_gpu
        .predict_shifts_gpu(&reloaded_bin, &atoms)
        .expect("GPU preds");
    assert_eq!(cpu_preds.len(), gpu_preds.len());
    assert!(!cpu_preds.is_empty());

    let mut max_diff = 0.0_f64;
    for (c, g) in cpu_preds.iter().zip(gpu_preds.iter()) {
        max_diff = max_diff.max((c.estimate.mean_ppm - g.estimate.mean_ppm).abs());
        max_diff = max_diff.max((c.estimate.sigma_ens_ppm - g.estimate.sigma_ens_ppm).abs());
    }
    assert!(
        max_diff < 1e-4,
        "GPU trained/heterogeneous bundle parity diff {max_diff:.3e} >= 1e-4"
    );
}

#[test]
fn test_legolas_gpu_torchani_acos095_parity_and_neighbor_cap() {
    // [synthetic self-consistency] Verify WebGPU parity (< 1e-4) for TorchANI's
    // acos(0.95 * cos(theta)) angular convention AND fail-closed neighbor/cell DoS caps (22.12:P2-2).
    let engine = match GpuEngine::new_headless() {
        Ok(e) => e,
        Err(err) => {
            eprintln!("Skipping WebGPU LEGOLAS TorchANI test (no GPU adapter available): {err}");
            return;
        }
    };
    let legolas_gpu = LegolasGpuEngine::new(&engine);

    let pdb = fixture_path("ubiquitin_1ubq_xray.pdb");
    let struct_model = StructureModel::parse_file(&pdb, false).expect("parse 1UBQ PDB");
    let atoms = extract_atoms_from_structure_model(&struct_model).expect("extract 1UBQ");
    let query_indices: Vec<usize> = atoms
        .iter()
        .enumerate()
        .filter_map(|(i, a)| a.target_nucleus.map(|_| i))
        .take(80)
        .collect();

    let cfg_torch = AevConfig::torchani_standard();
    let cpu_aevs = compute_aevs_for_indices(&atoms, &query_indices, &cfg_torch).expect("CPU AEVs");
    let gpu_aevs = legolas_gpu
        .compute_aevs_gpu(&atoms, &query_indices, &cfg_torch)
        .expect("GPU AEVs");

    let mut max_diff = 0.0_f64;
    for (ca, ga) in cpu_aevs.iter().zip(gpu_aevs.iter()) {
        for (&c, &g) in ca.iter().zip(ga.iter()) {
            assert!(g.is_finite());
            max_diff = max_diff.max((c - (g as f64)).abs());
        }
    }
    println!(
        "LEGOLAS TorchANI acos(0.95*cos) CPU f64 vs WebGPU f32 (1UBQ, {} queries, 560 dims): max_diff = {max_diff:.3e}",
        query_indices.len()
    );
    assert!(
        max_diff < 1e-4,
        "GPU TorchANI AEV max diff {max_diff:.3e} >= 1e-4"
    );

    // Dense > MAX_LEGOLAS_NEIGHBORS_PER_ATOM cluster must fail closed before GPU dispatch
    let dense_n = MAX_LEGOLAS_NEIGHBORS_PER_ATOM + 16;
    let dense_cluster: Vec<LegolasAtom> = (0..dense_n)
        .map(|i| {
            let t = i as f64;
            LegolasAtom::new(
                Species::C,
                [
                    0.01 * (t % 17.0),
                    0.01 * ((t / 17.0).floor() % 17.0),
                    0.01 * (t / 289.0).floor(),
                ],
            )
        })
        .collect();
    let dense_err = legolas_gpu
        .compute_aevs_gpu(&dense_cluster, &[0], &AevConfig::compact())
        .expect_err("GPU must fail closed on dense > MAX_LEGOLAS_NEIGHBORS_PER_ATOM input");
    assert!(
        dense_err.contains("MAX_LEGOLAS_NEIGHBORS_PER_ATOM"),
        "unexpected GPU error: {dense_err}"
    );

    // Dense 65,536-atom hostile cluster must fail closed before GPU dispatch
    let hostile_65k = vec![LegolasAtom::new(Species::C, [0.0, 0.0, 0.0]); MAX_LEGOLAS_ATOMS];
    let err_65k = legolas_gpu
        .compute_aevs_gpu(&hostile_65k, &[0], &AevConfig::compact())
        .expect_err("GPU must fail closed on dense 65k-atom hostile input");
    assert!(
        err_65k.contains("MAX_LEGOLAS_ATOMS_PER_CELL"),
        "unexpected GPU error: {err_65k}"
    );
}
