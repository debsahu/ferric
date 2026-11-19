//! Integration and quantitative verification tests for `nmr_core::pcs_transfer` (`T22.10`).
//!
//! All tests in this suite are **`[synthetic self-consistency]`** benchmarks using open public
//! `1UBQ` (`ubiquitin_1ubq_xray.pdb`) geometry and `synthetic_ubiquitin.str` diamagnetic shifts
//! with deterministic seeded noise (zero private/unpublished data, zero flaky wall-clock checks).

use std::collections::BTreeMap;
use std::f64::consts::PI;
use std::fs;
use std::path::{Path, PathBuf};

use nmr_core::nmrstar::read_star;
use nmr_core::noesy::{parse_pdb_str_first_model, v_dist, Vec3};
use nmr_core::paramagnetic::{predict_pcs_from_tensor, PcsObservation};
use nmr_core::pcs_transfer::{
    chi_iso_from_mu_eff_bohr, cornilescu_q_factor, delta_chi_tensor_from_spherical,
    fit_curie_ccr_chi_iso, fit_joint_pcs_ccr, iterative_pcs_assignment_transfer,
    match_pcs_assignment_transfer, mu_eff_bohr_from_chi_iso, predict_curie_dd_ccr_s_inv,
    sinkhorn_partial_assignment, CcrObservation, CcrPhysicsParams, CcrSpinPair, CcrTensorModel,
    DiamagneticPeak2D, IterativePcsTransferConfig, JointPcsCcrFitOptions, ParamagneticPeak2D,
    PcsMatchStatus, PcsTransferConfig, CODATA_GAMMA_15N_RAD_S_T, CODATA_GAMMA_1H_RAD_S_T,
};
use nmr_core::rng::Rng;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .to_path_buf()
}

/// Load 50 backbone amide `(1H^N, 15N)` residues from public `1UBQ` geometry paired with
/// `synthetic_ubiquitin.str` diamagnetic shifts.
fn load_1ubq_50_amides() -> Vec<DiamagneticPeak2D> {
    let root = workspace_root();
    let pdb_txt = fs::read_to_string(root.join("data/fixtures/ubiquitin_1ubq_xray.pdb"))
        .expect("read 1UBQ PDB fixture");

    let chain = parse_pdb_str_first_model(&pdb_txt).expect("parse 1UBQ first model");
    let star_ref = read_star(root.join("data/fixtures/synthetic_ubiquitin.str"), None)
        .expect("parse synthetic_ubiquitin.str");

    let mut shifts_by_seq: BTreeMap<i32, (Option<f64>, Option<f64>)> = BTreeMap::new();
    for (idx, &sid) in star_ref.seq_ids.iter().enumerate() {
        let h_val = star_ref.shifts.get(&(idx, "H".to_string())).copied();
        let n_val = star_ref.shifts.get(&(idx, "N".to_string())).copied();
        shifts_by_seq.insert(sid, (h_val, n_val));
    }

    let mut amides = Vec::new();
    for res in &chain.residues {
        let Some(h_coord) = res.h else {
            continue;
        };
        let Some(&(Some(h_ppm), Some(n_ppm))) = shifts_by_seq.get(&res.seq_id) else {
            continue;
        };
        amides.push(DiamagneticPeak2D {
            seq_id: res.seq_id,
            label: format!("{}{}", res.res_name, res.seq_id),
            h_ppm,
            x_ppm: n_ppm,
            coord_h_angstrom: Some(h_coord),
            coord_x_angstrom: Some(res.n),
        });
        if amides.len() == 50 {
            break;
        }
    }
    assert_eq!(amides.len(), 50, "expected 50 1UBQ backbone amide residues");
    amides
}

/// Choose a realistic `Tm3+` lanthanide tag site near `1UBQ` surface such that exactly
/// `target_bleached` residues fall inside `r <= r_bleach_angstrom` and every residue has `r >= 4.0 A`.
fn choose_metal_pos_with_exact_bleached_count(
    amides: &[DiamagneticPeak2D],
    target_bleached: usize,
    r_bleach_angstrom: f64,
) -> Vec3 {
    let mut centroid = [0.0_f64; 3];
    for a in amides {
        let c = a.coord_h_angstrom.expect("coord_h");
        centroid[0] += c[0] / (amides.len() as f64);
        centroid[1] += c[1] / (amides.len() as f64);
        centroid[2] += c[2] / (amides.len() as f64);
    }
    for a_anchor in amides {
        let anchor = a_anchor.coord_h_angstrom.expect("anchor coord_h");
        let dir_raw = [
            anchor[0] - centroid[0],
            anchor[1] - centroid[1],
            anchor[2] - centroid[2],
        ];
        let dir_norm = (dir_raw[0].powi(2) + dir_raw[1].powi(2) + dir_raw[2].powi(2)).sqrt();
        if dir_norm < 1e-6 {
            continue;
        }
        let dir = [
            dir_raw[0] / dir_norm,
            dir_raw[1] / dir_norm,
            dir_raw[2] / dir_norm,
        ];

        for step in 0..200 {
            let t = 1.0 + 0.15 * (step as f64);
            let cand = [
                anchor[0] + t * dir[0],
                anchor[1] + t * dir[1],
                anchor[2] + t * dir[2],
            ];
            let mut n_bleach = 0usize;
            let mut min_r = f64::INFINITY;
            for a in amides {
                let rh = v_dist(a.coord_h_angstrom.expect("coord_h"), cand);
                let rn = v_dist(a.coord_x_angstrom.expect("coord_n"), cand);
                min_r = min_r.min(rh).min(rn);
                if rh <= r_bleach_angstrom {
                    n_bleach += 1;
                }
            }
            let boundary_clear = amides.iter().all(|a| {
                let rh = v_dist(a.coord_h_angstrom.expect("coord_h"), cand);
                (rh - r_bleach_angstrom).abs() > 0.25
            });
            if n_bleach == target_bleached && min_r >= 3.5 && boundary_clear {
                return cand;
            }
        }
    }
    panic!(
        "failed to locate 1UBQ lanthanide site with exactly {target_bleached} bleached residues"
    );
}

#[test]
fn synthetic_1ubq_tm3_50res_structure_guided_and_em_pcs_transfer() {
    // [synthetic self-consistency]
    // 50-residue 1UBQ-geometry Tm3+ paramagnetic 2D 1H-15N HSQC with 8 PRE-bleached missing peaks
    // near the metal + realistic Gaussian noise (0.02 ppm 1H / 0.10 ppm 15N).
    let amides = load_1ubq_50_amides();
    let r_bleach = 9.0;
    let metal_pos = choose_metal_pos_with_exact_bleached_count(&amides, 8, r_bleach);

    let true_dchi_ax = 32.0; // 10^-32 m^3 (strong Tm3+ lanthanide tag)
    let true_dchi_rh = 11.0; // 10^-32 m^3 (rhombicity = 0.344 <= 2/3)
    let true_euler = [0.65, 1.05, 2.15];
    let true_tensor =
        delta_chi_tensor_from_spherical(metal_pos, true_dchi_ax, true_dchi_rh, true_euler)
            .expect("construct true Tm3+ tensor");

    let mut rng = Rng::new(2026_1022);
    let sigma_h = 0.02;
    let sigma_n = 0.10;

    let mut is_bleached = vec![false; amides.len()];
    let mut true_para_for_dia: Vec<Option<usize>> = vec![None; amides.len()];
    let mut raw_para: Vec<(usize, ParamagneticPeak2D)> = Vec::new();

    for (i, a) in amides.iter().enumerate() {
        let ch = a.coord_h_angstrom.expect("coord_h");
        let cn = a.coord_x_angstrom.expect("coord_n");
        let r_h = v_dist(ch, metal_pos);
        if r_h <= r_bleach {
            is_bleached[i] = true;
            continue;
        }
        let pcs_h = predict_pcs_from_tensor(ch, &true_tensor);
        let pcs_n = predict_pcs_from_tensor(cn, &true_tensor);
        let obs_h = a.h_ppm + pcs_h + rng.normal(0.0, sigma_h);
        let obs_n = a.x_ppm + pcs_n + rng.normal(0.0, sigma_n);
        raw_para.push((
            i,
            ParamagneticPeak2D {
                peak_id: 1000 + i,
                h_ppm: obs_h,
                x_ppm: obs_n,
                intensity: Some(1.0),
            },
        ));
    }
    assert_eq!(is_bleached.iter().filter(|&&b| b).count(), 8);
    assert_eq!(raw_para.len(), 42);

    // Deterministically shuffle paramagnetic peaks so peak ordering carries zero information
    for k in (1..raw_para.len()).rev() {
        let j = (rng.next_u64() as usize) % (k + 1);
        raw_para.swap(k, j);
    }
    let mut para_peaks = Vec::with_capacity(raw_para.len());
    for (new_idx, (dia_idx, pk)) in raw_para.into_iter().enumerate() {
        true_para_for_dia[dia_idx] = Some(new_idx);
        para_peaks.push(pk);
    }

    let cfg = PcsTransferConfig {
        sigma_h_ppm: sigma_h,
        sigma_x_ppm: sigma_n,
        model_sigma_ppm: 0.02,
        r_bleach_angstrom: r_bleach,
        ..Default::default()
    };

    // ------------------------------------------------------------------------
    // (a) Structure + exact DeltaChi-guided matching
    // ------------------------------------------------------------------------
    let guided = match_pcs_assignment_transfer(
        &amides,
        &para_peaks,
        Some(&true_tensor),
        Some(metal_pos),
        &cfg,
    )
    .expect("structure-guided PCS transfer");

    let mut guided_correct = 0usize;
    let mut guided_bleached_ok = 0usize;
    for tr in &guided.transfers {
        let i = tr.dia_index;
        if is_bleached[i] {
            assert_eq!(
                tr.status,
                PcsMatchStatus::UnmatchedBleached,
                "bleached residue {} (r={:?}) should map to UnmatchedBleached",
                tr.seq_id,
                tr.metal_distance_angstrom
            );
            guided_bleached_ok += 1;
        } else {
            // Verify exact geometric residual delta_geom = PCS(r_H) - PCS(r_N)
            let ph = tr.predicted_pcs_h_ppm.expect("pred H");
            let px = tr.predicted_pcs_x_ppm.expect("pred N");
            let gres = tr.geometric_residual_ppm.expect("geom res");
            assert!((gres - (ph - px)).abs() < 1e-12);
            if let PcsMatchStatus::Matched { para_index, .. } = tr.status {
                if true_para_for_dia[i] == Some(para_index) {
                    guided_correct += 1;
                }
            }
        }
    }
    let guided_acc = (guided_correct as f64) / 42.0;
    assert_eq!(guided_bleached_ok, 8);
    assert!(
        guided_acc >= 0.95,
        "structure-guided PCS accuracy {guided_acc:.4} ({guided_correct}/42) < 0.95"
    );

    // ------------------------------------------------------------------------
    // (b) Iterative EM transfer from a rough perturbed DeltaChi_0 (~30% amplitude & ~18 deg rotation)
    // ------------------------------------------------------------------------
    let rough_tensor = delta_chi_tensor_from_spherical(
        metal_pos,
        true_dchi_ax * 0.72,
        true_dchi_rh * 1.35,
        [
            true_euler[0] + 0.28,
            true_euler[1] - 0.22,
            true_euler[2] + 0.25,
        ],
    )
    .expect("rough initial tensor");

    let em_cfg = IterativePcsTransferConfig {
        transfer_cfg: cfg,
        anchor_confidence_threshold: 0.85,
        huber_k: 1.5,
        initial_model_sigma_ppm: 0.45,
        max_em_iters: 20,
        ..Default::default()
    };

    let em_rough = iterative_pcs_assignment_transfer(
        &amides,
        &para_peaks,
        metal_pos,
        Some(&rough_tensor),
        &[],
        &em_cfg,
    )
    .expect("iterative EM from rough initial tensor");

    assert!(em_rough.converged, "EM from rough tensor should converge");
    let mut em_correct = 0usize;
    let mut em_bleached_ok = 0usize;
    for tr in &em_rough.transfer.transfers {
        let i = tr.dia_index;
        if is_bleached[i] {
            assert_eq!(
                tr.status,
                PcsMatchStatus::UnmatchedBleached,
                "EM: bleached residue {} should map to UnmatchedBleached",
                tr.seq_id
            );
            em_bleached_ok += 1;
        } else if let PcsMatchStatus::Matched { para_index, .. } = tr.status {
            if true_para_for_dia[i] == Some(para_index) {
                em_correct += 1;
            }
        }
    }
    let em_acc = (em_correct as f64) / 42.0;
    assert_eq!(em_bleached_ok, 8);
    assert!(
        em_acc >= 0.95,
        "iterative EM (rough tensor) accuracy {em_acc:.4} ({em_correct}/42) < 0.95"
    );

    let ax_rel_err = (em_rough.fitted_tensor.dchi_ax - true_dchi_ax).abs() / true_dchi_ax.abs();
    let rh_rel_err = (em_rough.fitted_tensor.dchi_rh - true_dchi_rh).abs() / true_dchi_rh.abs();
    assert!(
        ax_rel_err < 0.02,
        "EM refitted dchi_ax relative error {ax_rel_err:.4} >= 0.02 (got {:.3} vs {:.3})",
        em_rough.fitted_tensor.dchi_ax,
        true_dchi_ax
    );
    assert!(
        rh_rel_err < 0.02,
        "EM refitted dchi_rh relative error {rh_rel_err:.4} >= 0.02 (got {:.3} vs {:.3})",
        em_rough.fitted_tensor.dchi_rh,
        true_dchi_rh
    );
    assert!(
        em_rough.fitted_tensor.q_factor < 0.03,
        "EM refitted tensor Q = {:.4} >= 0.03",
        em_rough.fitted_tensor.q_factor
    );

    // ------------------------------------------------------------------------
    // (c) Iterative EM transfer from 6 seed anchors (initial_tensor = None)
    // ------------------------------------------------------------------------
    let unbleached_indices: Vec<usize> = (0..amides.len()).filter(|&i| !is_bleached[i]).collect();
    let seed_anchors: Vec<(i32, usize)> = [0usize, 7, 14, 21, 28, 35]
        .iter()
        .map(|&k| {
            let di = unbleached_indices[k];
            let pi = true_para_for_dia[di].expect("unbleached para idx");
            (amides[di].seq_id, pi)
        })
        .collect();

    let em_seeds = iterative_pcs_assignment_transfer(
        &amides,
        &para_peaks,
        metal_pos,
        None,
        &seed_anchors,
        &em_cfg,
    )
    .expect("iterative EM from 6 seed anchors");

    assert!(em_seeds.converged, "EM from 6 seed anchors should converge");
    let mut seed_correct = 0usize;
    let mut seed_bleached_ok = 0usize;
    for tr in &em_seeds.transfer.transfers {
        let i = tr.dia_index;
        if is_bleached[i] {
            assert_eq!(tr.status, PcsMatchStatus::UnmatchedBleached);
            seed_bleached_ok += 1;
        } else if let PcsMatchStatus::Matched { para_index, .. } = tr.status {
            if true_para_for_dia[i] == Some(para_index) {
                seed_correct += 1;
            }
        }
    }
    let seed_acc = (seed_correct as f64) / 42.0;
    assert_eq!(seed_bleached_ok, 8);
    assert!(
        seed_acc >= 0.95,
        "seed-anchor EM accuracy {seed_acc:.4} ({seed_correct}/42) < 0.95"
    );
    assert!(em_seeds.fitted_tensor.q_factor < 0.03);
    let seed_ax_err = (em_seeds.fitted_tensor.dchi_ax - true_dchi_ax).abs() / true_dchi_ax.abs();
    assert!(seed_ax_err < 0.02);

    // Serde round-trip check
    let json = serde_json::to_string(&em_rough).expect("serialize IterativePcsTransferResult");
    let decoded: nmr_core::pcs_transfer::IterativePcsTransferResult =
        serde_json::from_str(&json).expect("deserialize IterativePcsTransferResult");
    assert_eq!(decoded.n_anchors_used, em_rough.n_anchors_used);
    assert_eq!(decoded.transfer.n_matched, em_rough.transfer.n_matched);
}

#[test]
fn synthetic_diagonal_mode_without_tensor_and_artifact_peak_rejection() {
    // [synthetic self-consistency]
    // Structure-free diagonal PCS matching along Delta_delta_H ~ Delta_delta_N plus an off-diagonal artifact peak.
    let dia = vec![
        DiamagneticPeak2D {
            seq_id: 1,
            label: "ALA1".to_string(),
            h_ppm: 7.50,
            x_ppm: 118.00,
            coord_h_angstrom: None,
            coord_x_angstrom: None,
        },
        DiamagneticPeak2D {
            seq_id: 2,
            label: "GLY2".to_string(),
            h_ppm: 8.60,
            x_ppm: 109.00,
            coord_h_angstrom: None,
            coord_x_angstrom: None,
        },
        DiamagneticPeak2D {
            seq_id: 3,
            label: "VAL3".to_string(),
            h_ppm: 9.20,
            x_ppm: 124.50,
            coord_h_angstrom: None,
            coord_x_angstrom: None,
        },
    ];
    let para = vec![
        // Shifted diagonally by +0.85 ppm from ALA1
        ParamagneticPeak2D {
            peak_id: 10,
            h_ppm: 8.36,
            x_ppm: 118.82,
            intensity: Some(1.0),
        },
        // Shifted diagonally by -0.60 ppm from GLY2
        ParamagneticPeak2D {
            peak_id: 11,
            h_ppm: 7.99,
            x_ppm: 108.42,
            intensity: Some(1.0),
        },
        // Shifted diagonally by +1.10 ppm from VAL3
        ParamagneticPeak2D {
            peak_id: 12,
            h_ppm: 10.31,
            x_ppm: 125.58,
            intensity: Some(1.0),
        },
        // Off-diagonal noise/artifact peak far from all diagonal rays
        ParamagneticPeak2D {
            peak_id: 99,
            h_ppm: 6.20,
            x_ppm: 131.00,
            intensity: Some(0.3),
        },
    ];

    let cfg = PcsTransferConfig {
        r_bleach_angstrom: 0.0,
        ..Default::default()
    };
    let res = match_pcs_assignment_transfer(&dia, &para, None, None, &cfg)
        .expect("diagonal PCS matching");

    assert_eq!(res.n_matched, 3);
    assert_eq!(res.unassigned_para_indices, vec![3]);
    assert!(res.sinkhorn.unassigned_peak_probs[3] > 0.99);
    for (i, tr) in res.transfers.iter().enumerate() {
        assert_eq!(
            tr.status,
            PcsMatchStatus::Matched {
                para_index: i,
                para_peak_id: para[i].peak_id
            }
        );
        assert!(tr.posterior_confidence > 0.95);
    }
}

#[test]
fn synthetic_sinkhorn_marginal_calibration_doublets_vs_isolated() {
    // [synthetic self-consistency]
    // Verify calibrated Sinkhorn-Knopp posteriors:
    // - Ambiguous close doublet peaks receive substantially lower posterior confidence & margin
    //   than well-isolated peaks;
    // - Exact degenerate 2x2 doublet yields exact 50/50 marginal split and zero margin;
    // - Row & column marginals satisfy exact probabilistic conservation.
    let metal_pos = [0.0, 0.0, 0.0];
    let tensor = delta_chi_tensor_from_spherical(metal_pos, 20.0, 5.0, [0.3, 0.8, 1.2])
        .expect("build tensor");

    let dia = vec![
        // Isolated peak #0
        DiamagneticPeak2D {
            seq_id: 1,
            label: "ISO1".to_string(),
            h_ppm: 7.20,
            x_ppm: 112.00,
            coord_h_angstrom: Some([18.0, 2.0, 1.0]),
            coord_x_angstrom: Some([18.8, 2.3, 1.2]),
        },
        // Isolated peak #1
        DiamagneticPeak2D {
            seq_id: 2,
            label: "ISO2".to_string(),
            h_ppm: 9.40,
            x_ppm: 127.00,
            coord_h_angstrom: Some([-16.0, 5.0, 4.0]),
            coord_x_angstrom: Some([-16.7, 5.4, 4.3]),
        },
        // Close ambiguous doublet pair (#2 and #3) whose predicted para positions differ by only 0.012 ppm H / 0.04 ppm N
        DiamagneticPeak2D {
            seq_id: 3,
            label: "DBL_A".to_string(),
            h_ppm: 8.100,
            x_ppm: 119.50,
            coord_h_angstrom: Some([2.0, 19.0, 3.0]),
            coord_x_angstrom: Some([2.2, 19.9, 3.3]),
        },
        DiamagneticPeak2D {
            seq_id: 4,
            label: "DBL_B".to_string(),
            h_ppm: 8.112,
            x_ppm: 119.54,
            coord_h_angstrom: Some([2.0, 19.0, 3.0]),
            coord_x_angstrom: Some([2.2, 19.9, 3.3]),
        },
    ];

    let mut para = Vec::new();
    for (i, d) in dia.iter().enumerate() {
        let ph = predict_pcs_from_tensor(d.coord_h_angstrom.expect("h"), &tensor);
        let pn = predict_pcs_from_tensor(d.coord_x_angstrom.expect("n"), &tensor);
        para.push(ParamagneticPeak2D {
            peak_id: i,
            h_ppm: d.h_ppm + ph,
            x_ppm: d.x_ppm + pn,
            intensity: Some(1.0),
        });
    }

    let cfg = PcsTransferConfig {
        sigma_h_ppm: 0.02,
        sigma_x_ppm: 0.10,
        model_sigma_ppm: 0.0,
        r_bleach_angstrom: 9.0,
        sinkhorn_temperature: 1.0,
        ..Default::default()
    };

    let res = match_pcs_assignment_transfer(&dia, &para, Some(&tensor), Some(metal_pos), &cfg)
        .expect("Sinkhorn calibration match");

    assert!(res.sinkhorn.converged);
    assert!(res.sinkhorn.max_marginal_error < 1e-8);

    // Verify exact row & column conservation
    for i in 0..dia.len() {
        let row_sum: f64 = res.sinkhorn.match_probs[i].iter().sum::<f64>()
            + res.sinkhorn.unmatched_residue_probs[i];
        assert!((row_sum - 1.0).abs() < 1e-8);
    }
    for j in 0..para.len() {
        let col_sum: f64 = (0..dia.len())
            .map(|i| res.sinkhorn.match_probs[i][j])
            .sum::<f64>()
            + res.sinkhorn.unassigned_peak_probs[j];
        assert!((col_sum - 1.0).abs() < 1e-8);
    }

    let iso0_p = res.transfers[0].posterior_confidence;
    let iso1_p = res.transfers[1].posterior_confidence;
    let dbl_a_p = res.transfers[2].posterior_confidence;
    let dbl_b_p = res.transfers[3].posterior_confidence;

    assert!(
        iso0_p > 0.99 && iso1_p > 0.99,
        "isolated peaks must have posterior > 0.99 (got {iso0_p:.4}, {iso1_p:.4})"
    );
    assert!(
        dbl_a_p < 0.70 && dbl_b_p < 0.70,
        "close doublet peaks must have reduced posterior < 0.70 (got {dbl_a_p:.4}, {dbl_b_p:.4})"
    );
    assert!(
        res.transfers[2].posterior_margin < 0.40 && res.transfers[0].posterior_margin > 0.98,
        "doublet margin ({:.4}) must be much smaller than isolated margin ({:.4})",
        res.transfers[2].posterior_margin,
        res.transfers[0].posterior_margin
    );

    // Direct exact symmetric 2x2 degenerate Sinkhorn test
    let sym_cost = vec![vec![0.5, 0.5], vec![0.5, 0.5]];
    let sym_sk =
        sinkhorn_partial_assignment(&sym_cost, &[15.0, 15.0], &[0.0, 0.0], 1.0, 200, 1e-11)
            .expect("exact symmetric Sinkhorn");
    assert!((sym_sk.match_probs[0][0] - sym_sk.match_probs[0][1]).abs() < 1e-10);
    assert!((sym_sk.match_probs[0][0] - 0.5).abs() < 1e-3);
}

#[test]
fn synthetic_curie_dd_ccr_exact_physics_scaling_and_spin_pairs() {
    // [synthetic self-consistency]
    // Verify exact Curie-spin / dipolar cross-correlated relaxation physics:
    // 1. P2(cos theta) angular dependence (+1 at 0 deg, 0 at magic angle 54.7356 deg, -0.5 at 90 deg)
    // 2. r_M^-3 inverse-cubic metal-distance scaling
    // 3. Exact B0 * (4 J(0) + 3 J(omega_S)) field scaling & linear B0 scaling in extreme limits
    // 4. Exact CODATA-2018 Curie chi_iso <-> mu_eff (Tm3+) conversion & 1/T temperature scaling
    // 5. All four spin pairs (N15H1, H1N15, C13H1, H1C13) including negative gamma_15N sign
    let metal = [0.0, 0.0, 0.0];
    let chi_tm = chi_iso_from_mu_eff_bohr(7.56, 298.0).expect("Tm3+ chi_iso");
    assert!(
        (chi_tm - 50.0).abs() < 0.5,
        "Tm3+ (7.56 uB, 298 K) chi_iso should be ~50.0 x 10^-32 m^3 (got {chi_tm:.3})"
    );
    let mu_back = mu_eff_bohr_from_chi_iso(chi_tm, 298.0).expect("roundtrip mu_eff");
    assert!((mu_back - 7.56).abs() < 1e-12);

    // Exact 1/T Curie law scaling
    let chi_310 = chi_iso_from_mu_eff_bohr(7.56, 310.0).expect("chi at 310 K");
    assert!(((chi_310 / chi_tm) - (298.0 / 310.0)).abs() < 1e-12);

    let params_800 =
        CcrPhysicsParams::new(5.0, 800.0, 298.0, CcrSpinPair::N15H1).expect("800 MHz N15H1 params");

    let r_m = 15.0;
    let coord_s = [0.0, 0.0, r_m]; // metal -> S along +z
    let rate_at_angle = |theta_rad: f64| -> f64 {
        let coord_i = [
            coord_s[0] + 1.02 * theta_rad.sin(),
            coord_s[1],
            coord_s[2] + 1.02 * theta_rad.cos(),
        ];
        predict_curie_dd_ccr_s_inv(coord_s, coord_i, metal, chi_tm, None, &params_800, 1.0)
            .expect("predict CCR")
    };

    let g_0 = rate_at_angle(0.0);
    let g_30 = rate_at_angle(PI / 6.0);
    let magic = (1.0_f64 / 3.0_f64).sqrt().acos();
    let g_magic = rate_at_angle(magic);
    let g_90 = rate_at_angle(PI / 2.0);
    let g_180 = rate_at_angle(PI);

    // Physical magnitude check for Tm3+ at 15 A, 800 MHz, 5 ns (~1.9 s^-1 for 15N)
    assert!(
        g_0 > 1.5 && g_0 < 2.5,
        "15N Curie-DD CCR at 15 A / 800 MHz / 5 ns should be ~1.9 s^-1 (got {g_0:.4})"
    );
    assert!((g_30 / g_0 - 0.625).abs() < 1e-12, "P2(cos 30) = 5/8");
    assert!(
        g_magic.abs() < 1e-14,
        "magic angle must give exact zero CCR (got {g_magic:.3e})"
    );
    assert!((g_90 / g_0 - (-0.5)).abs() < 1e-12, "P2(cos 90) = -1/2");
    assert!((g_180 - g_0).abs() < 1e-12, "P2 is even under 180 deg flip");

    // Exact r_M^-3 scaling: doubling r_M divides Gamma_CCR by exactly 8.0
    let coord_s_2x = [0.0, 0.0, 2.0 * r_m];
    let coord_i_2x = [0.0, 0.0, 2.0 * r_m + 1.02];
    let g_2x = predict_curie_dd_ccr_s_inv(
        coord_s_2x,
        coord_i_2x,
        metal,
        chi_tm,
        None,
        &params_800,
        1.0,
    )
    .expect("predict at 2*r_M");
    assert!(((g_0 / g_2x) - 8.0).abs() < 1e-12);

    // Exact B0 * (4 J(0) + 3 J(omega_S)) scaling across spectrometer fields
    let coord_i_0 = [0.0, 0.0, r_m + 1.02];
    for mhz in [500.0, 600.0, 1000.0] {
        let p_f = CcrPhysicsParams::new(5.0, mhz, 298.0, CcrSpinPair::N15H1).expect("field params");
        let g_f = predict_curie_dd_ccr_s_inv(coord_s, coord_i_0, metal, chi_tm, None, &p_f, 1.0)
            .expect("CCR at field");
        let expected_ratio = (p_f.b0_tesla() * p_f.spectral_density_combination_s())
            / (params_800.b0_tesla() * params_800.spectral_density_combination_s());
        assert!(((g_f / g_0) - expected_ratio).abs() < 1e-12);
    }

    // Strict linear B0 scaling in the macromolecular slow-tumbling limit (omega_S * tau_m >> 1, 4 J(0) dominated)
    // and in the extreme-narrowing fast-tumbling limit (omega_S * tau_m << 1, 7 tau_m)
    let p_slow_500 =
        CcrPhysicsParams::new(300.0, 500.0, 298.0, CcrSpinPair::N15H1).expect("slow 500");
    let p_slow_1000 =
        CcrPhysicsParams::new(300.0, 1000.0, 298.0, CcrSpinPair::N15H1).expect("slow 1000");
    let g_s500 =
        predict_curie_dd_ccr_s_inv(coord_s, coord_i_0, metal, chi_tm, None, &p_slow_500, 1.0)
            .expect("slow 500 rate");
    let g_s1000 =
        predict_curie_dd_ccr_s_inv(coord_s, coord_i_0, metal, chi_tm, None, &p_slow_1000, 1.0)
            .expect("slow 1000 rate");
    assert!(
        ((g_s1000 / g_s500) - 2.0).abs() < 5.0e-4,
        "linear B0 scaling in slow-tumbling limit: got ratio {:.6}",
        g_s1000 / g_s500
    );

    let p_fast_500 =
        CcrPhysicsParams::new(0.01, 500.0, 298.0, CcrSpinPair::N15H1).expect("fast 500");
    let p_fast_1000 =
        CcrPhysicsParams::new(0.01, 1000.0, 298.0, CcrSpinPair::N15H1).expect("fast 1000");
    let g_f500 =
        predict_curie_dd_ccr_s_inv(coord_s, coord_i_0, metal, chi_tm, None, &p_fast_500, 1.0)
            .expect("fast 500 rate");
    let g_f1000 =
        predict_curie_dd_ccr_s_inv(coord_s, coord_i_0, metal, chi_tm, None, &p_fast_1000, 1.0)
            .expect("fast 1000 rate");
    assert!(
        ((g_f1000 / g_f500) - 2.0).abs() < 5.0e-5,
        "linear B0 scaling in extreme-narrowing limit: got ratio {:.6}",
        g_f1000 / g_f500
    );

    // Order-parameter S^2 linear scaling
    let g_s2 =
        predict_curie_dd_ccr_s_inv(coord_s, coord_i_0, metal, chi_tm, None, &params_800, 0.85)
            .expect("S^2 scaled rate");
    assert!(((g_s2 / g_0) - 0.85).abs() < 1e-12);

    // H1N15 vs N15H1 spin-pair sign & gyromagnetic scaling (extreme narrowing for exact gamma ratio)
    let p_h1_fast =
        CcrPhysicsParams::new(0.001, 800.0, 298.0, CcrSpinPair::H1N15).expect("H1N15 fast");
    let p_n15_fast =
        CcrPhysicsParams::new(0.001, 800.0, 298.0, CcrSpinPair::N15H1).expect("N15H1 fast");
    let g_h1 = predict_curie_dd_ccr_s_inv(coord_s, coord_i_0, metal, chi_tm, None, &p_h1_fast, 1.0)
        .expect("H1N15 rate");
    let g_n15 =
        predict_curie_dd_ccr_s_inv(coord_s, coord_i_0, metal, chi_tm, None, &p_n15_fast, 1.0)
            .expect("N15H1 rate");
    // Ratio should equal (gamma_N * gamma_H^2 * J_comb_H) / (gamma_H * gamma_N^2 * J_comb_N)
    let expected_gamma_ratio = CODATA_GAMMA_1H_RAD_S_T / CODATA_GAMMA_15N_RAD_S_T;
    let expected_exact_ratio = expected_gamma_ratio
        * (p_h1_fast.spectral_density_combination_s()
            / p_n15_fast.spectral_density_combination_s());
    assert!(
        ((g_h1 / g_n15) - expected_exact_ratio).abs() < 1e-11,
        "H1N15/N15H1 exact ratio {:.6} vs expected {:.6}",
        g_h1 / g_n15,
        expected_exact_ratio
    );
    assert!(
        ((g_h1 / g_n15) - expected_gamma_ratio).abs() < 5e-4,
        "H1N15/N15H1 extreme-narrowing ratio {:.4} vs gamma_H/gamma_N {:.4}",
        g_h1 / g_n15,
        expected_gamma_ratio
    );

    // C13H1 & H1C13 both positive at theta = 0 (since gamma_13C > 0 and gamma_1H > 0)
    let p_c13 = CcrPhysicsParams::new(5.0, 800.0, 298.0, CcrSpinPair::C13H1).expect("C13H1");
    let p_hc = CcrPhysicsParams::new(5.0, 800.0, 298.0, CcrSpinPair::H1C13).expect("H1C13");
    let g_c13 = predict_curie_dd_ccr_s_inv(coord_s, coord_i_0, metal, chi_tm, None, &p_c13, 1.0)
        .expect("C13H1 rate");
    let g_hc = predict_curie_dd_ccr_s_inv(coord_s, coord_i_0, metal, chi_tm, None, &p_hc, 1.0)
        .expect("H1C13 rate");
    assert!(g_c13 > 0.0 && g_hc > g_c13);
}

#[test]
fn synthetic_joint_pcs_ccr_recovers_chi_iso_and_delta_chi_with_q_free() {
    // [synthetic self-consistency]
    // Joint PCS + Gamma_CCR recovery of (chi_iso, dchi_ax, dchi_rh) on 1UBQ geometry under both
    // IsotropicChiOnly and FullChiTensor models (noiseless exact + realistic noisy + Huber IRLS).
    let amides = load_1ubq_50_amides();
    let r_bleach = 9.0;
    let metal_pos = choose_metal_pos_with_exact_bleached_count(&amides, 8, r_bleach);

    let true_chi_iso = 50.0; // 10^-32 m^3 (~7.56 uB at 298 K)
    let true_dchi_ax = 32.0; // 10^-32 m^3
    let true_dchi_rh = 11.0; // 10^-32 m^3
    let true_euler = [0.65, 1.05, 2.15];
    let true_tensor =
        delta_chi_tensor_from_spherical(metal_pos, true_dchi_ax, true_dchi_rh, true_euler)
            .expect("true tensor");

    let params =
        CcrPhysicsParams::new(4.5, 800.0, 298.0, CcrSpinPair::N15H1).expect("valid CCR params");

    let mut pcs_clean = Vec::new();
    let mut ccr_iso_clean = Vec::new();
    let mut ccr_full_clean = Vec::new();

    for a in &amides {
        let ch = a.coord_h_angstrom.expect("h");
        let cn = a.coord_x_angstrom.expect("n");
        if v_dist(ch, metal_pos) <= r_bleach {
            continue;
        }
        let pcs_h = predict_pcs_from_tensor(ch, &true_tensor);
        pcs_clean.push(PcsObservation::new(a.seq_id, ch, pcs_h).with_uncertainty(0.02));

        let g_iso = predict_curie_dd_ccr_s_inv(cn, ch, metal_pos, true_chi_iso, None, &params, 1.0)
            .expect("iso CCR");
        let g_full = predict_curie_dd_ccr_s_inv(
            cn,
            ch,
            metal_pos,
            true_chi_iso,
            Some(&true_tensor),
            &params,
            1.0,
        )
        .expect("full-chi CCR");
        ccr_iso_clean.push(CcrObservation::new(a.seq_id, cn, ch, g_iso, 0.05));
        ccr_full_clean.push(CcrObservation::new(a.seq_id, cn, ch, g_full, 0.05));
    }

    // 1. Noiseless exact recovery — IsotropicChiOnly
    let fit_iso_exact = fit_joint_pcs_ccr(
        &pcs_clean,
        &ccr_iso_clean,
        metal_pos,
        &params,
        &JointPcsCcrFitOptions {
            tensor_model: CcrTensorModel::IsotropicChiOnly,
            cv_folds: 5,
            ..Default::default()
        },
    )
    .expect("exact isotropic joint fit");

    assert!((fit_iso_exact.chi_iso_1e32_m3 - true_chi_iso).abs() < 1e-9);
    assert!((fit_iso_exact.delta_chi.dchi_ax - true_dchi_ax).abs() < 1e-9);
    assert!((fit_iso_exact.delta_chi.dchi_rh - true_dchi_rh).abs() < 1e-9);
    assert!(fit_iso_exact.pcs_q_factor < 1e-10);
    assert!(fit_iso_exact.ccr_q_factor < 1e-10);
    assert!(fit_iso_exact.pcs_q_free.expect("pcs q_free") < 1e-10);
    assert!(fit_iso_exact.ccr_q_free.expect("ccr q_free") < 1e-10);
    assert!(fit_iso_exact.joint_q_free.expect("joint q_free") < 1e-10);

    // 2. Noiseless exact recovery — FullChiTensor (coupled DeltaChi + chi_iso in Gamma_CCR)
    let fit_full_exact = fit_joint_pcs_ccr(
        &pcs_clean,
        &ccr_full_clean,
        metal_pos,
        &params,
        &JointPcsCcrFitOptions {
            tensor_model: CcrTensorModel::FullChiTensor,
            cv_folds: 5,
            ..Default::default()
        },
    )
    .expect("exact full-chi joint fit");

    assert!((fit_full_exact.chi_iso_1e32_m3 - true_chi_iso).abs() < 1e-9);
    assert!((fit_full_exact.delta_chi.dchi_ax - true_dchi_ax).abs() < 1e-9);
    assert!((fit_full_exact.delta_chi.dchi_rh - true_dchi_rh).abs() < 1e-9);
    assert!(fit_full_exact.ccr_q_factor < 1e-10);
    assert!(fit_full_exact.ccr_q_free.expect("full ccr q_free") < 1e-10);
    // Verify trace(full_chi_matrix_mol) / 3 == chi_iso
    let tr_full = fit_full_exact.full_chi_matrix_mol[0][0]
        + fit_full_exact.full_chi_matrix_mol[1][1]
        + fit_full_exact.full_chi_matrix_mol[2][2];
    assert!((tr_full / 3.0 - true_chi_iso).abs() < 1e-9);

    // 3. Noisy recovery + Huber IRLS robustness against an outlier CCR & PCS peak
    let mut rng = Rng::new(777_2210);
    let mut pcs_noisy: Vec<PcsObservation> = pcs_clean
        .iter()
        .map(|o| {
            PcsObservation::new(
                o.seq_id,
                o.coord_angstrom,
                o.pcs_ppm + rng.normal(0.0, 0.02),
            )
            .with_uncertainty(0.02)
        })
        .collect();
    let mut ccr_noisy: Vec<CcrObservation> = ccr_full_clean
        .iter()
        .map(|o| {
            CcrObservation::new(
                o.seq_id,
                o.coord_s_angstrom,
                o.coord_i_angstrom,
                o.ccr_rate_s_inv + rng.normal(0.0, 0.04),
                0.04,
            )
        })
        .collect();

    let fit_noisy = fit_joint_pcs_ccr(
        &pcs_noisy,
        &ccr_noisy,
        metal_pos,
        &params,
        &JointPcsCcrFitOptions {
            tensor_model: CcrTensorModel::FullChiTensor,
            cv_folds: 5,
            ..Default::default()
        },
    )
    .expect("noisy full-chi joint fit");

    let chi_err = (fit_noisy.chi_iso_1e32_m3 - true_chi_iso).abs() / true_chi_iso;
    let ax_err = (fit_noisy.delta_chi.dchi_ax - true_dchi_ax).abs() / true_dchi_ax;
    let rh_err = (fit_noisy.delta_chi.dchi_rh - true_dchi_rh).abs() / true_dchi_rh;
    assert!(chi_err < 0.02, "chi_iso rel err {chi_err:.4} >= 0.02");
    assert!(ax_err < 0.02, "dchi_ax rel err {ax_err:.4} >= 0.02");
    assert!(rh_err < 0.02, "dchi_rh rel err {rh_err:.4} >= 0.02");
    assert!(fit_noisy.pcs_q_factor < 0.03);
    assert!(fit_noisy.pcs_q_free.expect("noisy pcs q_free") < 0.04);
    assert!(fit_noisy.ccr_q_factor < 0.08);
    assert!(fit_noisy.ccr_q_free.expect("noisy ccr q_free") < 0.10);
    assert!(
        fit_noisy.chi_iso_uncertainty_1e32_m3 > 0.0 && fit_noisy.chi_iso_uncertainty_1e32_m3 < 2.0
    );

    // Inject severe outliers into both PCS and CCR and verify Huber IRLS recovers accurate tensor
    pcs_noisy[0].pcs_ppm += 2.5;
    ccr_noisy[0].ccr_rate_s_inv += 6.0;
    let fit_huber = fit_joint_pcs_ccr(
        &pcs_noisy,
        &ccr_noisy,
        metal_pos,
        &params,
        &JointPcsCcrFitOptions {
            tensor_model: CcrTensorModel::FullChiTensor,
            huber_k: Some(1.5),
            cv_folds: 5,
            ..Default::default()
        },
    )
    .expect("Huber IRLS joint fit");
    assert!((fit_huber.chi_iso_1e32_m3 - true_chi_iso).abs() / true_chi_iso < 0.03);
    assert!((fit_huber.delta_chi.dchi_ax - true_dchi_ax).abs() / true_dchi_ax < 0.03);

    // 4. Standalone CCR-only chi_iso fit
    let ccr_only =
        fit_curie_ccr_chi_iso(&ccr_iso_clean, metal_pos, None, &params, 5).expect("CCR-only fit");
    assert!((ccr_only.chi_iso_1e32_m3 - true_chi_iso).abs() < 1e-9);
    assert!(ccr_only.q_factor < 1e-10);
    assert!(ccr_only.q_free.expect("ccr_only q_free") < 1e-10);

    // JSON serde round-trip
    let s = serde_json::to_string(&fit_noisy).expect("serialize JointPcsCcrFitResult");
    let back: nmr_core::pcs_transfer::JointPcsCcrFitResult =
        serde_json::from_str(&s).expect("deserialize JointPcsCcrFitResult");
    assert!((back.chi_iso_1e32_m3 - fit_noisy.chi_iso_1e32_m3).abs() < 1e-12);
}

#[test]
fn hostile_inputs_fail_closed_across_all_entry_points() {
    // [synthetic self-consistency]
    // Verify strict fail-closed rejection on non-finite, unphysical, underdetermined, or degenerate inputs.

    // 1. Mandatory CcrPhysicsParams validation (zero silent defaults)
    assert!(CcrPhysicsParams::new(0.0, 800.0, 298.0, CcrSpinPair::N15H1).is_err());
    assert!(CcrPhysicsParams::new(-5.0, 800.0, 298.0, CcrSpinPair::N15H1).is_err());
    assert!(CcrPhysicsParams::new(f64::NAN, 800.0, 298.0, CcrSpinPair::N15H1).is_err());
    assert!(CcrPhysicsParams::new(5.0, 0.0, 298.0, CcrSpinPair::N15H1).is_err());
    assert!(CcrPhysicsParams::new(5.0, f64::INFINITY, 298.0, CcrSpinPair::N15H1).is_err());
    assert!(CcrPhysicsParams::new(5.0, 800.0, 0.0, CcrSpinPair::N15H1).is_err());
    assert!(CcrPhysicsParams::new(5.0, 800.0, -10.0, CcrSpinPair::N15H1).is_err());

    let valid_params = CcrPhysicsParams::new(5.0, 800.0, 298.0, CcrSpinPair::N15H1).expect("ok");
    assert!(valid_params.with_bond_length_angstrom(0.0).is_err());
    assert!(valid_params.with_bond_length_angstrom(-1.0).is_err());

    // 2. Curie chi_iso / mu_eff conversions
    assert!(chi_iso_from_mu_eff_bohr(-1.0, 298.0).is_err());
    assert!(chi_iso_from_mu_eff_bohr(7.5, 0.0).is_err());
    assert!(mu_eff_bohr_from_chi_iso(-5.0, 298.0).is_err());
    assert!(cornilescu_q_factor(&[], &[]).is_err());
    assert!(cornilescu_q_factor(&[0.0, 0.0], &[0.0, 0.0]).is_err());

    // 3. Forward CCR geometric/physical fail-closed checks
    assert!(predict_curie_dd_ccr_s_inv(
        [0.5, 0.0, 0.0], // < 1.2 A from metal
        [1.5, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        50.0,
        None,
        &valid_params,
        1.0
    )
    .is_err());
    assert!(predict_curie_dd_ccr_s_inv(
        [15.0, 0.0, 0.0],
        [15.0, 0.0, 0.0], // zero bond vector
        [0.0, 0.0, 0.0],
        50.0,
        None,
        &valid_params,
        1.0
    )
    .is_err());
    assert!(predict_curie_dd_ccr_s_inv(
        [15.0, 0.0, 0.0],
        [16.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        50.0,
        None,
        &valid_params,
        1.5 // invalid S^2 > 1
    )
    .is_err());

    // 4. Magic-angle degenerate CCR geometry (P2 = 0 everywhere -> zero Fisher info for chi_iso)
    let magic = (1.0_f64 / 3.0_f64).sqrt().acos();
    let magic_ccr: Vec<CcrObservation> = (1..=6)
        .map(|k| {
            let r = 12.0 + (k as f64);
            CcrObservation::new(
                k,
                [0.0, 0.0, r],
                [1.02 * magic.sin(), 0.0, r + 1.02 * magic.cos()],
                0.01,
                0.05,
            )
        })
        .collect();
    assert!(fit_curie_ccr_chi_iso(&magic_ccr, [0.0, 0.0, 0.0], None, &valid_params, 3).is_err());

    // 5. PCS transfer & Sinkhorn hostile configs/inputs
    assert!(
        match_pcs_assignment_transfer(&[], &[], None, None, &PcsTransferConfig::default()).is_err()
    );
    let bad_cfg = PcsTransferConfig {
        sigma_h_ppm: 0.0,
        ..Default::default()
    };
    let good_dia = vec![DiamagneticPeak2D {
        seq_id: 1,
        label: "A1".to_string(),
        h_ppm: 8.0,
        x_ppm: 120.0,
        coord_h_angstrom: Some([15.0, 0.0, 0.0]),
        coord_x_angstrom: Some([16.0, 0.0, 0.0]),
    }];
    let good_para = vec![ParamagneticPeak2D {
        peak_id: 0,
        h_ppm: 8.1,
        x_ppm: 120.1,
        intensity: None,
    }];
    assert!(match_pcs_assignment_transfer(&good_dia, &good_para, None, None, &bad_cfg).is_err());

    let nan_dia = vec![DiamagneticPeak2D {
        seq_id: 1,
        label: "A1".to_string(),
        h_ppm: f64::NAN,
        x_ppm: 120.0,
        coord_h_angstrom: Some([15.0, 0.0, 0.0]),
        coord_x_angstrom: Some([16.0, 0.0, 0.0]),
    }];
    assert!(match_pcs_assignment_transfer(
        &nan_dia,
        &good_para,
        None,
        None,
        &PcsTransferConfig::default()
    )
    .is_err());

    assert!(sinkhorn_partial_assignment(&[vec![f64::NAN]], &[1.0], &[0.0], 1.0, 50, 1e-6).is_err());
    assert!(sinkhorn_partial_assignment(&[vec![1.0]], &[1.0], &[0.0], 0.0, 50, 1e-6).is_err());

    // 6. Iterative EM requires initial_tensor OR >= 5 seed anchors
    assert!(iterative_pcs_assignment_transfer(
        &good_dia,
        &good_para,
        [0.0, 0.0, 0.0],
        None,
        &[(1, 0)], // only 1 seed (< 5)
        &IterativePcsTransferConfig::default()
    )
    .is_err());

    // 7. Joint PCS+CCR underdetermined / invalid observations
    let few_pcs = vec![PcsObservation::new(1, [15.0, 0.0, 0.0], 0.5)];
    let one_ccr = vec![CcrObservation::new(
        1,
        [15.0, 0.0, 0.0],
        [16.0, 0.0, 0.0],
        1.2,
        0.05,
    )];
    assert!(fit_joint_pcs_ccr(
        &few_pcs,
        &one_ccr,
        [0.0, 0.0, 0.0],
        &valid_params,
        &JointPcsCcrFitOptions::default()
    )
    .is_err());
    assert!(fit_joint_pcs_ccr(
        &few_pcs,
        &[],
        [0.0, 0.0, 0.0],
        &valid_params,
        &JointPcsCcrFitOptions::default()
    )
    .is_err());
}
