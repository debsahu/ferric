//! Integration tests for `nmr_core::legolas_shifts` (`T22.12`).
//!
//! All synthetic training, conformer perturbation, and self-consistency checks in this file
//! are explicitly labelled `[synthetic self-consistency]` / `[dev-set]`.

use std::path::PathBuf;

use nmr_core::legolas_shifts::{
    compute_aevs_bruteforce_reference, compute_aevs_for_indices, compute_all_aevs, cosine_cutoff,
    extract_atoms_from_structure_model, mlp_forward_and_gradients, train_nucleus_ensemble,
    ActivationKind, AevConfig, LegolasAtom, LegolasMlp, LegolasModel, LegolasTrainConfig,
    NucleusEnsemble, Species, TargetNucleus, MAX_LEGOLAS_ATOMS,
};
use nmr_core::rng::Rng;
use nmr_core::struct_shifts::StructureModel;

fn fixture_path(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("data/fixtures")
        .join(rel)
}

/// Build a proper 3D rotation matrix `Q in SO(3)` via Rodrigues' rotation formula.
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

/// Construct a compact 5-species (`H, C, N, O, S`) synthetic tripeptide-like cluster with all 6
/// backbone target nuclei (`[synthetic self-consistency]`).
fn synthetic_five_species_cluster() -> Vec<LegolasAtom> {
    vec![
        LegolasAtom::with_target(Species::N, [0.00, 0.00, 0.00], 1, "N", TargetNucleus::N15),
        LegolasAtom::with_target(Species::H, [-0.45, 0.89, 0.12], 1, "H", TargetNucleus::Hn),
        LegolasAtom::with_target(
            Species::C,
            [1.45, 0.05, -0.10],
            1,
            "CA",
            TargetNucleus::Ca13,
        ),
        LegolasAtom::with_target(
            Species::H,
            [1.78, -0.92, -0.45],
            1,
            "HA",
            TargetNucleus::Ha1,
        ),
        LegolasAtom::with_target(
            Species::C,
            [1.95, 1.12, -1.08],
            1,
            "CB",
            TargetNucleus::Cb13,
        ),
        LegolasAtom::with_target(Species::C, [2.10, 0.22, 1.26], 1, "C", TargetNucleus::Co13),
        LegolasAtom::new(Species::O, [1.55, 0.88, 2.14]),
        LegolasAtom::new(Species::S, [3.68, 1.44, -1.25]), // Cysteine/Methionine sulfur
        LegolasAtom::new(Species::H, [3.85, 2.32, -0.35]),
        LegolasAtom::with_target(Species::N, [3.32, -0.30, 1.42], 2, "N", TargetNucleus::N15),
        LegolasAtom::with_target(Species::H, [3.74, -0.82, 0.66], 2, "H", TargetNucleus::Hn),
        LegolasAtom::with_target(
            Species::C,
            [4.08, -0.15, 2.66],
            2,
            "CA",
            TargetNucleus::Ca13,
        ),
        LegolasAtom::with_target(Species::H, [3.62, 0.61, 3.28], 2, "HA", TargetNucleus::Ha1),
        LegolasAtom::with_target(Species::C, [5.52, 0.25, 2.34], 2, "CB", TargetNucleus::Cb13),
        LegolasAtom::with_target(Species::C, [4.02, -1.46, 3.44], 2, "C", TargetNucleus::Co13),
        LegolasAtom::new(Species::O, [4.71, -2.41, 3.08]),
        LegolasAtom::new(Species::S, [6.45, -1.12, 1.70]),
    ]
}

#[test]
fn test_aev_cell_list_matches_bruteforce_and_species_pair_symmetry() {
    // [synthetic self-consistency] Verify species pair index symmetry and bijection onto 0..15.
    let mut seen = [false; 15];
    for &a in &Species::ALL {
        for &b in &Species::ALL {
            let p_ab = Species::pair_index(a, b);
            let p_ba = Species::pair_index(b, a);
            assert_eq!(
                p_ab, p_ba,
                "pair_index must be symmetric for ({a:?}, {b:?})"
            );
            assert!(p_ab < 15);
            seen[p_ab] = true;
        }
    }
    assert!(
        seen.iter().all(|&s| s),
        "all 15 unordered species pairs must be covered"
    );

    let cfg = AevConfig::ani1_standard();
    assert_eq!(cfg.radial_dim(), 80);
    assert_eq!(cfg.angular_dim(), 480);
    assert_eq!(cfg.total_dim(), 560);

    let atoms = synthetic_five_species_cluster();
    let indices: Vec<usize> = (0..atoms.len()).collect();
    let cell_aevs = compute_aevs_for_indices(&atoms, &indices, &cfg).expect("cell-list AEV");
    let brute_aevs =
        compute_aevs_bruteforce_reference(&atoms, &indices, &cfg).expect("brute-force AEV");

    assert_eq!(cell_aevs.len(), brute_aevs.len());
    let mut max_diff = 0.0_f64;
    let mut nonzero_radial = 0usize;
    let mut nonzero_angular = 0usize;
    for (ca, ba) in cell_aevs.iter().zip(brute_aevs.iter()) {
        assert_eq!(ca.len(), 560);
        for (k, (&v1, &v2)) in ca.iter().zip(ba.iter()).enumerate() {
            assert!(v1.is_finite());
            max_diff = max_diff.max((v1 - v2).abs());
            if v1 > 1e-9 {
                if k < 80 {
                    nonzero_radial += 1;
                } else {
                    nonzero_angular += 1;
                }
            }
        }
    }
    assert!(nonzero_radial > 50, "expected active radial channels");
    assert!(nonzero_angular > 50, "expected active angular channels");
    assert!(
        max_diff < 1e-13,
        "cell-list vs brute-force AEV max diff {max_diff:.3e} >= 1e-13"
    );
}

#[test]
fn test_aev_and_shifts_exact_se3_and_permutation_invariance() {
    // [synthetic self-consistency] Verify exact SE(3) translation, rotation, and atom-permutation
    // invariance (< 1e-12 CPU f64) on both real 1UBQ coordinates and 5-species cluster.
    let pdb = fixture_path("ubiquitin_1ubq_xray.pdb");
    let struct_model = StructureModel::parse_file(&pdb, false).expect("parse 1UBQ PDB");
    let mut atoms = extract_atoms_from_structure_model(&struct_model).expect("extract 1UBQ atoms");
    // Use first 180 atoms (contains full local neighborhood of ~15 residues) plus sulfur atoms
    atoms.truncate(180);
    atoms.extend(synthetic_five_species_cluster());

    let cfg = AevConfig::compact();
    let model = LegolasModel::new_seeded_all_backbone(
        cfg.clone(),
        &[24, 12],
        ActivationKind::Celu { alpha: 0.1 },
        4,
        2026_1119,
    )
    .expect("seeded model");

    let base_aevs = compute_all_aevs(&atoms, &cfg).expect("base AEVs");
    let base_preds = model.predict_atoms(&atoms).expect("base predictions");
    assert!(base_preds.len() >= 40, "expected backbone predictions");

    // 1. Arbitrary 3D translation + SO(3) rotation
    let rot = rodrigues_rotation([0.37, -0.71, 0.60], 1.832_595_714_594_046);
    let trans = [137.42, -291.85, 84.19];
    let transformed: Vec<LegolasAtom> = atoms
        .iter()
        .map(|a| {
            let rp = mat_vec(&rot, a.position);
            let mut b = a.clone();
            b.position = [rp[0] + trans[0], rp[1] + trans[1], rp[2] + trans[2]];
            b
        })
        .collect();

    let se3_aevs = compute_all_aevs(&transformed, &cfg).expect("SE(3) AEVs");
    let se3_preds = model
        .predict_atoms(&transformed)
        .expect("SE(3) predictions");

    let mut max_aev_se3 = 0.0_f64;
    for (a, b) in base_aevs.iter().zip(se3_aevs.iter()) {
        for (&va, &vb) in a.iter().zip(b.iter()) {
            max_aev_se3 = max_aev_se3.max((va - vb).abs());
        }
    }
    assert!(
        max_aev_se3 < 1e-12,
        "AEV SE(3) invariance violation: {max_aev_se3:.3e} >= 1e-12"
    );

    let mut max_shift_se3 = 0.0_f64;
    let mut max_sigma_se3 = 0.0_f64;
    for (p0, p1) in base_preds.iter().zip(se3_preds.iter()) {
        assert_eq!(p0.estimate.nucleus, p1.estimate.nucleus);
        max_shift_se3 = max_shift_se3.max((p0.estimate.mean_ppm - p1.estimate.mean_ppm).abs());
        max_sigma_se3 =
            max_sigma_se3.max((p0.estimate.sigma_ens_ppm - p1.estimate.sigma_ens_ppm).abs());
    }
    assert!(
        max_shift_se3 < 1e-12,
        "shift SE(3) invariance violation: {max_shift_se3:.3e} >= 1e-12"
    );
    assert!(
        max_sigma_se3 < 1e-12,
        "sigma_ens SE(3) invariance violation: {max_sigma_se3:.3e} >= 1e-12"
    );

    // 2. Arbitrary atom-index permutation
    let mut rng = Rng::new(987_654_321);
    let perm = rng.permutation(atoms.len());
    let mut inv_perm = vec![0usize; atoms.len()];
    for (new_idx, &old_idx) in perm.iter().enumerate() {
        inv_perm[old_idx] = new_idx;
    }
    let permuted_atoms: Vec<LegolasAtom> = perm.iter().map(|&old| atoms[old].clone()).collect();
    let perm_aevs = compute_all_aevs(&permuted_atoms, &cfg).expect("permuted AEVs");
    let perm_preds = model
        .predict_atoms(&permuted_atoms)
        .expect("permuted predictions");

    let mut max_aev_perm = 0.0_f64;
    for old_idx in 0..atoms.len() {
        let new_idx = inv_perm[old_idx];
        for (&va, &vb) in base_aevs[old_idx].iter().zip(perm_aevs[new_idx].iter()) {
            max_aev_perm = max_aev_perm.max((va - vb).abs());
        }
    }
    assert!(
        max_aev_perm < 1e-12,
        "AEV permutation invariance violation: {max_aev_perm:.3e} >= 1e-12"
    );

    let mut max_shift_perm = 0.0_f64;
    for p0 in &base_preds {
        let expected_new_idx = inv_perm[p0.atom_index];
        let p1 = perm_preds
            .iter()
            .find(|p| p.atom_index == expected_new_idx)
            .expect("matched permuted prediction");
        max_shift_perm = max_shift_perm.max((p0.estimate.mean_ppm - p1.estimate.mean_ppm).abs());
    }
    assert!(
        max_shift_perm < 1e-12,
        "shift permutation invariance violation: {max_shift_perm:.3e} >= 1e-12"
    );
}

#[test]
fn test_cosine_cutoff_c1_continuity_and_boundary_behavior() {
    // [synthetic self-consistency] Smooth C^1 continuity of f_c(R) and full AEV across R_c.
    let rc_r = 5.2_f64;
    let rc_a = 3.5_f64;

    assert!((cosine_cutoff(0.0, rc_r) - 1.0).abs() < 1e-15);
    assert_eq!(cosine_cutoff(rc_r, rc_r), 0.0);
    assert_eq!(cosine_cutoff(rc_r + 1e-9, rc_r), 0.0);

    // Check left-hand value and derivative vanish smoothly as R -> R_c^-
    let h1 = 1e-3;
    let h2 = 1e-5;
    let slope1 = (cosine_cutoff(rc_r - h1, rc_r) - 0.0) / h1;
    let slope2 = (cosine_cutoff(rc_r - h2, rc_r) - 0.0) / h2;
    assert!(slope2 < slope1 * 0.02, "expected zero derivative at R_c");
    assert!(slope2 < 1e-5, "slope near R_c should vanish: {slope2:.3e}");

    // Full AEV radial & angular cutoff continuity
    let cfg = AevConfig::compact();
    let eval_at = |r_rad: f64, r_ang: f64| -> Vec<f64> {
        let atoms = vec![
            LegolasAtom::new(Species::C, [0.0, 0.0, 0.0]),
            LegolasAtom::new(Species::N, [1.5, 0.0, 0.0]),
            LegolasAtom::new(Species::O, [0.0, r_ang, 0.0]),
            LegolasAtom::new(Species::S, [0.0, 0.0, r_rad]),
        ];
        compute_aevs_for_indices(&atoms, &[0], &cfg).expect("cutoff AEV")[0].clone()
    };

    let eps = 1e-4;
    let aev_rad_in = eval_at(rc_r - eps, 2.0);
    let aev_rad_at = eval_at(rc_r, 2.0);
    let aev_rad_out = eval_at(rc_r + eps, 2.0);
    let diff_rad_in: f64 = aev_rad_in
        .iter()
        .zip(aev_rad_at.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    let diff_rad_out: f64 = aev_rad_out
        .iter()
        .zip(aev_rad_at.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(
        diff_rad_in < 1e-8,
        "radial AEV discontinuity at R_c: {diff_rad_in:.3e}"
    );
    assert_eq!(
        diff_rad_out, 0.0,
        "radial AEV must be identically 0 past R_c"
    );

    let aev_ang_in = eval_at(4.0, rc_a - eps);
    let aev_ang_at = eval_at(4.0, rc_a);
    let aev_ang_out = eval_at(4.0, rc_a + eps);
    let diff_ang_in: f64 = aev_ang_in[cfg.radial_dim()..]
        .iter()
        .zip(aev_ang_at[cfg.radial_dim()..].iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    let diff_ang_out: f64 = aev_ang_out[cfg.radial_dim()..]
        .iter()
        .zip(aev_ang_at[cfg.radial_dim()..].iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(
        diff_ang_in < 1e-7,
        "angular AEV discontinuity at R_c^A: {diff_ang_in:.3e}"
    );
    assert_eq!(
        diff_ang_out, 0.0,
        "angular AEV must be identically 0 past R_c^A"
    );
}

#[test]
fn test_mlp_backprop_finite_difference_gradients_all_activations() {
    // [synthetic self-consistency] Central finite-difference check (< 1e-5) for analytical
    // backpropagation across CELU, GELU, and SiLU activations.
    let mut rng = Rng::new(4242);
    let in_dim = 10usize;
    let batch = 5usize;
    let xs_owned: Vec<Vec<f64>> = (0..batch)
        .map(|_| (0..in_dim).map(|_| rng.normal(0.0, 0.8)).collect())
        .collect();
    let xs: Vec<&[f64]> = xs_owned.iter().map(Vec::as_slice).collect();
    let ys: Vec<f64> = (0..batch).map(|_| rng.normal(55.0, 4.0)).collect();

    let activations = [
        ActivationKind::Celu { alpha: 0.1 },
        ActivationKind::Gelu,
        ActivationKind::Silu,
    ];
    let h = 1e-6_f64;

    for (ai, &act) in activations.iter().enumerate() {
        let mut mlp = LegolasMlp::new_seeded(in_dim, &[8, 6], 1000 + ai as u64).expect("mlp");
        let mean = 55.5;
        let scale = 4.2;

        let (_, analytic) =
            mlp_forward_and_gradients(&mlp, act, mean, scale, &xs, &ys).expect("analytic grads");

        let mut max_err = 0.0_f64;
        for li in 0..mlp.layers.len() {
            for wi in 0..mlp.layers[li].weights.len() {
                let orig = mlp.layers[li].weights[wi];
                mlp.layers[li].weights[wi] = orig + h;
                let (lp, _) =
                    mlp_forward_and_gradients(&mlp, act, mean, scale, &xs, &ys).expect("lp");
                mlp.layers[li].weights[wi] = orig - h;
                let (lm, _) =
                    mlp_forward_and_gradients(&mlp, act, mean, scale, &xs, &ys).expect("lm");
                mlp.layers[li].weights[wi] = orig;

                let fd = (lp - lm) / (2.0 * h);
                let g = analytic.layer_d_weights[li][wi];
                max_err = max_err.max((fd - g).abs());
            }
            for bi in 0..mlp.layers[li].biases.len() {
                let orig = mlp.layers[li].biases[bi];
                mlp.layers[li].biases[bi] = orig + h;
                let (lp, _) =
                    mlp_forward_and_gradients(&mlp, act, mean, scale, &xs, &ys).expect("lp");
                mlp.layers[li].biases[bi] = orig - h;
                let (lm, _) =
                    mlp_forward_and_gradients(&mlp, act, mean, scale, &xs, &ys).expect("lm");
                mlp.layers[li].biases[bi] = orig;

                let fd = (lp - lm) / (2.0 * h);
                let g = analytic.layer_d_biases[li][bi];
                max_err = max_err.max((fd - g).abs());
            }
        }
        assert!(
            max_err < 1e-5,
            "backprop FD gradient mismatch for {act:?}: {max_err:.3e} >= 1e-5"
        );
    }
}

#[test]
fn test_online_adam_training_and_ensemble_epistemic_uncertainty() {
    // [synthetic self-consistency] / [dev-set]:
    // Train a compact E=5 LEGOLAS CELU ensemble on multi-conformer 1UBQ CA geometries using real
    // analytical-backprop Adam, verifying MSE & RMSE reduction and epistemic uncertainty widening
    // on perturbed out-of-distribution geometries.
    let pdb = fixture_path("ubiquitin_1ubq_xray.pdb");
    let struct_model = StructureModel::parse_file(&pdb, false).expect("parse 1UBQ PDB");
    let base_atoms = extract_atoms_from_structure_model(&struct_model).expect("extract 1UBQ atoms");

    // Take first 160 atoms (~14 residues) and extract CA query atoms
    let atoms_sub: Vec<LegolasAtom> = base_atoms.into_iter().take(160).collect();
    let ca_indices: Vec<usize> = atoms_sub
        .iter()
        .enumerate()
        .filter_map(|(i, a)| (a.target_nucleus == Some(TargetNucleus::Ca13)).then_some(i))
        .collect();
    assert!(
        ca_indices.len() >= 10,
        "expected >= 10 CA atoms in 1UBQ slice"
    );

    let cfg = AevConfig::compact();
    let aev_dim = cfg.total_dim();

    // Deterministic synthetic ground-truth shift oracle over local AEV features
    let mut oracle_rng = Rng::new(77_001);
    let oracle_w1: Vec<f64> = (0..aev_dim).map(|_| oracle_rng.normal(0.0, 0.35)).collect();
    let oracle_target = |aev: &[f64]| -> f64 {
        let lin: f64 = aev.iter().zip(oracle_w1.iter()).map(|(x, w)| x * w).sum();
        55.0 + 4.0 * (0.4 * lin).tanh() + 0.15 * lin
    };

    // Build 4 slightly jittered in-distribution conformers (3 training, 1 held-out ID)
    let mut conf_rng = Rng::new(77_002);
    let mut train_aevs = Vec::new();
    let mut train_targets = Vec::new();
    let mut id_test_aevs = Vec::new();
    let mut id_test_targets = Vec::new();

    for conf in 0..4 {
        let jittered: Vec<LegolasAtom> = atoms_sub
            .iter()
            .map(|a| {
                let mut b = a.clone();
                for axis in 0..3 {
                    b.position[axis] += conf_rng.normal(0.0, 0.04);
                }
                b
            })
            .collect();
        let aevs = compute_aevs_for_indices(&jittered, &ca_indices, &cfg).expect("conformer AEVs");
        for aev in aevs {
            let y = oracle_target(&aev);
            if conf < 3 {
                train_aevs.push(aev);
                train_targets.push(y);
            } else {
                id_test_aevs.push(aev);
                id_test_targets.push(y);
            }
        }
    }

    // Build strongly perturbed out-of-distribution conformer (compressed/disordered local packing)
    let ood_atoms: Vec<LegolasAtom> = atoms_sub
        .iter()
        .map(|a| {
            let mut b = a.clone();
            for axis in 0..3 {
                b.position[axis] = b.position[axis] * 0.72 + conf_rng.normal(0.0, 0.9);
            }
            b
        })
        .collect();
    let ood_aevs = compute_aevs_for_indices(&ood_atoms, &ca_indices, &cfg).expect("OOD AEVs");

    let mut ensemble = NucleusEnsemble::new_seeded(
        TargetNucleus::Ca13,
        aev_dim,
        &[24, 12],
        ActivationKind::Celu { alpha: 0.1 },
        5,
        55.0,
        4.0,
        77_003,
    )
    .expect("init ensemble");

    let train_cfg = LegolasTrainConfig {
        learning_rate: 1.5e-2,
        weight_decay: 1e-4,
        grad_clip_norm: Some(10.0),
        ..LegolasTrainConfig::default()
    };
    let report = train_nucleus_ensemble(
        &mut ensemble,
        &train_aevs,
        &train_targets,
        120,
        &train_cfg,
        77_004,
    )
    .expect("train ensemble");

    assert_eq!(report.mean_epoch_mse.len(), 120);
    assert!(
        report.final_rmse_ppm < report.initial_rmse_ppm * 0.25,
        "expected >= 4x training RMSE reduction: initial={:.3}, final={:.3}",
        report.initial_rmse_ppm,
        report.final_rmse_ppm
    );
    assert!(
        report.final_rmse_ppm < 0.5,
        "expected tight final training RMSE (< 0.5 ppm), got {:.3}",
        report.final_rmse_ppm
    );

    // Held-out in-distribution conformer RMSE and epistemic uncertainty
    let mut id_sq = 0.0_f64;
    let mut id_sigma_sum = 0.0_f64;
    for (x, &y) in id_test_aevs.iter().zip(id_test_targets.iter()) {
        let est = ensemble.predict_aev(x).expect("ID predict");
        let d = est.mean_ppm - y;
        id_sq += d * d;
        id_sigma_sum += est.sigma_ens_ppm;
    }
    let id_rmse = (id_sq / (id_test_aevs.len() as f64)).sqrt();
    let id_mean_sigma = id_sigma_sum / (id_test_aevs.len() as f64);

    let mut ood_sigma_sum = 0.0_f64;
    for x in &ood_aevs {
        let est = ensemble.predict_aev(x).expect("OOD predict");
        ood_sigma_sum += est.sigma_ens_ppm;
    }
    let ood_mean_sigma = ood_sigma_sum / (ood_aevs.len() as f64);

    assert!(
        id_rmse < report.initial_rmse_ppm * 0.45,
        "expected held-out ID conformer RMSE improvement: id_rmse={id_rmse:.3}, init={:.3}",
        report.initial_rmse_ppm
    );
    assert!(
        ood_mean_sigma > id_mean_sigma * 1.5,
        "expected epistemic uncertainty widening on OOD geometries: ood_sigma={ood_mean_sigma:.4} vs id_sigma={id_mean_sigma:.4}"
    );
}

#[test]
fn test_legolas_v1_json_and_binary_bundle_roundtrip_and_hostile_rejection() {
    // [synthetic self-consistency] Verify LEGOLAS-V1 JSON & binary SHA-256 roundtrip, tamper
    // rejection, dimension-mismatch rejection, NaN/Inf rejection, and MAX_LEGOLAS_ATOMS cap.
    let cfg = AevConfig::compact();
    let model = LegolasModel::new_seeded_all_backbone(
        cfg.clone(),
        &[16, 8],
        ActivationKind::Silu,
        3,
        55_123,
    )
    .expect("model");
    let atoms = synthetic_five_species_cluster();
    let ref_preds = model.predict_atoms(&atoms).expect("ref preds");

    // 1. JSON roundtrip (both pretty-printed and minified preserve SHA-256 integrity)
    let json = model.to_json_bundle().expect("serialize JSON");
    let loaded_json = LegolasModel::from_json_bundle(&json).expect("deserialize JSON");
    let json_preds = loaded_json.predict_atoms(&atoms).expect("json preds");
    for (a, b) in ref_preds.iter().zip(json_preds.iter()) {
        assert!((a.estimate.mean_ppm - b.estimate.mean_ppm).abs() < 1e-13);
        assert!((a.estimate.sigma_ens_ppm - b.estimate.sigma_ens_ppm).abs() < 1e-13);
    }

    // 2. Binary roundtrip (bit-exact IEEE-754 f64 + SHA-256)
    let bin = model.to_binary_bundle().expect("serialize binary");
    let loaded_bin = LegolasModel::from_binary_bundle(&bin).expect("deserialize binary");
    assert_eq!(model, loaded_bin);
    assert_eq!(
        model.payload_sha256().expect("sha1"),
        loaded_bin.payload_sha256().expect("sha2")
    );

    // 3. Tamper detection (JSON)
    let tampered_json = json.replacen("\"output_mean\": 8.25", "\"output_mean\": 8.26", 1);
    assert_ne!(tampered_json, json, "expected tamper replacement to apply");
    let err = LegolasModel::from_json_bundle(&tampered_json)
        .expect_err("tampered JSON must fail SHA-256 verification");
    assert!(err.contains("SHA-256"), "unexpected error: {err}");

    // 4. Tamper detection (binary)
    let mut tampered_bin = bin.clone();
    let last = tampered_bin.len() - 1;
    tampered_bin[last] ^= 0x01;
    let err = LegolasModel::from_binary_bundle(&tampered_bin)
        .expect_err("tampered binary must fail SHA-256 verification");
    assert!(err.contains("SHA-256"), "unexpected error: {err}");

    // 5. Wrong format version / wrong binary magic
    let bad_format_json = json.replacen("\"LEGOLAS-V1\"", "\"LEGOLAS-V9\"", 1);
    assert!(LegolasModel::from_json_bundle(&bad_format_json).is_err());
    let mut bad_magic = bin.clone();
    bad_magic[0] = b'X';
    assert!(LegolasModel::from_binary_bundle(&bad_magic).is_err());

    // 6. Dimension mismatch & NaN/Inf rejection
    let mut dim_bad = model.clone();
    if let Some(ens) = dim_bad.ensembles.get_mut(&TargetNucleus::Ca13) {
        ens.members[0].layers[0].in_dim += 1;
    }
    assert!(dim_bad.validate().is_err());

    let mut nan_bad = model.clone();
    if let Some(ens) = nan_bad.ensembles.get_mut(&TargetNucleus::Hn) {
        ens.members[0].layers[0].weights[0] = f64::NAN;
    }
    assert!(nan_bad.validate().is_err());
    assert!(nan_bad.to_json_bundle().is_err());

    let mut inf_bad = model.clone();
    if let Some(ens) = inf_bad.ensembles.get_mut(&TargetNucleus::N15) {
        ens.members[0].layers[0].biases[0] = f64::INFINITY;
    }
    assert!(inf_bad.validate().is_err());

    // 7. Hostile atom count & coordinate caps
    let mut nan_atoms = atoms.clone();
    nan_atoms[0].position[1] = f64::NAN;
    assert!(model.predict_atoms(&nan_atoms).is_err());

    let mut huge_coord_atoms = atoms.clone();
    huge_coord_atoms[0].position[0] = 1e9;
    assert!(model.predict_atoms(&huge_coord_atoms).is_err());

    let oversized = vec![LegolasAtom::new(Species::C, [0.0, 0.0, 0.0]); MAX_LEGOLAS_ATOMS + 1];
    let cap_err = compute_aevs_for_indices(&oversized, &[0], &cfg)
        .expect_err("oversized atom slice must fail MAX_LEGOLAS_ATOMS cap");
    assert!(cap_err.contains("MAX_LEGOLAS_ATOMS"));
}
