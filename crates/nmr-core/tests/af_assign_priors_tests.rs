//! Comprehensive unit, statistical, homolog-firewall, anti-hallucination (`120 deg` corrupted
//! 12-residue loop), `PAE`-gated `3D 15N-NOESY`, sparse `CB`-stripped `HNCA`-only (`>= +8.0 pp`),
//! and `ferric assign --af-structure` CLI tests for `T21.4` (`nmr-core::af_assign_priors`).

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use nmr_core::af_assign_priors::{
    ell_student_t4, estimate_af_shift_offsets, estimate_af_shift_offsets_from_predictions,
    parse_noesy_15n_peaks_str, AfAssignConfig, AfAssignPriors, Noesy15nPeak,
    AF_NOESY_MAX_SEQ_BONUS_NATS, DEFAULT_AF_CLAMP_NATS, DEFAULT_AF_WEIGHT,
};
use nmr_core::af_concordance::parse_pae_json_str;
use nmr_core::bayes_sa::{solve_bayes_sa_from_peaks, BayesSaConfig};
use nmr_core::evaluate::score;
use nmr_core::evo_graph::{assign_evo_graph_from_peaks, EvoGraphConfig};
use nmr_core::nmrstar::{read_star, Reference};
use nmr_core::noesy::v_dist;
use nmr_core::peaks::write_peaks;
use nmr_core::priors::Priors;
use nmr_core::rng::Rng;
use nmr_core::simulate::{simulate, SimConfig};
use nmr_core::solver::{
    assign, assign_with_af_priors, pair_matrix, unary_matrix, unary_matrix_with_af_priors,
    AnnealConfig, ScoreConfig, FORBID,
};
use nmr_core::spinsystems::{build_spin_systems, SpinSystem, Tolerances};
use nmr_core::struct_shifts::StructureModel;
use nmr_core::torsion_db::HomologFirewallQuery;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn ubiquitin_firewall() -> HomologFirewallQuery {
    HomologFirewallQuery {
        query_pdb_ids: vec![
            "1D3Z".to_string(),
            "1UBQ".to_string(),
            "1G6J".to_string(),
            "2NR2".to_string(),
        ],
        query_bmrb_ids: vec![
            "6457".to_string(),
            "17769".to_string(),
            "5387".to_string(),
            "4769".to_string(),
        ],
        query_uniprot_id: Some("P0CG48".to_string()),
        query_family_id: Some("2.60.98.10".to_string()),
        query_topology_id: Some("2.60.98".to_string()),
        query_ca_coords: None,
    }
}

fn load_ubiquitin_af_like_model(root: &Path) -> StructureModel {
    let mut model =
        StructureModel::parse_file(root.join("data/pdb_refs/1d3z.pdb"), false).expect("parse 1d3z");
    for (idx, r) in model.residues.iter_mut().enumerate() {
        // Ordered globular core (1..=71) with high pLDDT and flexible RLRGG C-tail (72..=76) with low pLDDT
        r.plddt = Some(if idx >= 71 { 44.0 } else { 92.0 });
    }
    model
}

fn shift_spin_systems(
    systems: &[SpinSystem],
    dh: f64,
    dn: f64,
    dca: f64,
    dcb: f64,
    dc: f64,
) -> Vec<SpinSystem> {
    let mut out = systems.to_vec();
    for ss in &mut out {
        ss.h += dh;
        ss.n += dn;
        for ((atom, _off), cands) in &mut ss.cands {
            let delta = match atom.as_str() {
                "CA" => dca,
                "CB" => dcb,
                "C" => dc,
                _ => 0.0,
            };
            for cand in cands.iter_mut() {
                cand.0 += delta;
            }
        }
    }
    out
}

/// Simulate realistic `3D 15N-edited NOESY-HSQC` amide-amide (`H^N_i, 15N_i, H^N_j`) crosspeaks
/// from ground-truth experimental shifts and 3D `H^N-H^N` distances (`d_NN <= 5.1 A`), plus
/// realistic frequency jitter, missing crosspeaks, and random spurious proton crosspeaks.
fn simulate_amide_15n_noesy_peaks(
    ref_data: &Reference,
    model: &StructureModel,
    seed: u64,
    missing_rate: f64,
    spurious_fraction: f64,
) -> Vec<Noesy15nPeak> {
    let mut rng = Rng::new(seed ^ 0xA551_6E3D);
    let n_res = ref_data.sequence.len().min(model.residues.len());
    let mut peaks = Vec::new();

    for i in 1..n_res {
        if ref_data.sequence[i] == "PRO" {
            continue;
        }
        let Some(&hi_shift) = ref_data.shifts.get(&(i, "H".to_string())) else {
            continue;
        };
        let Some(&ni_shift) = ref_data.shifts.get(&(i, "N".to_string())) else {
            continue;
        };
        let Some(hi_xyz) = model.residues[i].h else {
            continue;
        };

        // Diagonal peak (automatically rejected by self-diagonal filter)
        peaks.push(Noesy15nPeak {
            h_direct_ppm: hi_shift + rng.normal(0.0, 0.008),
            n15_ppm: ni_shift + rng.normal(0.0, 0.08),
            h_indirect_ppm: hi_shift + rng.normal(0.0, 0.010),
            volume: 2.5e6,
        });

        for j in 1..n_res {
            if i == j || ref_data.sequence[j] == "PRO" {
                continue;
            }
            let Some(&hj_shift) = ref_data.shifts.get(&(j, "H".to_string())) else {
                continue;
            };
            let Some(hj_xyz) = model.residues[j].h else {
                continue;
            };
            let d_nn = v_dist(hi_xyz, hj_xyz);
            if !(1.6..=5.10).contains(&d_nn) {
                continue;
            }
            if rng.f64() < missing_rate {
                continue;
            }
            let intensity = 1.0e6 * (3.0 / d_nn.max(2.1)).powi(6);
            peaks.push(Noesy15nPeak {
                h_direct_ppm: hi_shift + rng.normal(0.0, 0.010),
                n15_ppm: ni_shift + rng.normal(0.0, 0.10),
                h_indirect_ppm: hj_shift + rng.normal(0.0, 0.015),
                volume: intensity,
            });
        }
    }

    let n_spur = ((peaks.len() as f64) * spurious_fraction).round() as usize;
    for _ in 0..n_spur {
        let anchor_k = 1 + rng.usize(n_res - 1);
        let (h_a, n_a) = match (
            ref_data.shifts.get(&(anchor_k, "H".to_string())),
            ref_data.shifts.get(&(anchor_k, "N".to_string())),
        ) {
            (Some(&h), Some(&n)) => (h, n),
            _ => (7.5 + rng.f64() * 2.0, 115.0 + rng.f64() * 12.0),
        };
        peaks.push(Noesy15nPeak {
            h_direct_ppm: h_a + rng.normal(0.0, 0.015),
            n15_ppm: n_a + rng.normal(0.0, 0.15),
            h_indirect_ppm: 6.4 + rng.f64() * 3.4,
            volume: 4.0e4,
        });
    }

    peaks
}

fn format_noesy_15n_tsv(peaks: &[Noesy15nPeak]) -> String {
    let mut txt = String::from("experiment\th\tn\th2\tsign\theight\tvolume\n");
    for p in peaks {
        txt.push_str(&format!(
            "NOESY_N15\t{:.4}\t{:.4}\t{:.4}\t1\t{:.1}\t{:.1}\n",
            p.h_direct_ppm, p.n15_ppm, p.h_indirect_ppm, p.volume, p.volume
        ));
    }
    txt
}

fn build_pae_json_for_ubiquitin(n_res: usize) -> String {
    let mut rows = Vec::with_capacity(n_res);
    for i in 0..n_res {
        let mut row = Vec::with_capacity(n_res);
        for j in 0..n_res {
            let v = if i == j {
                0.5
            } else if i >= 71 || j >= 71 {
                // Disordered C-terminal tail has high PAE (> 18 A) relative to folded core
                19.5
            } else {
                2.4 + 0.02 * ((i as f64 - j as f64).abs())
            };
            row.push(format!("{v:.2}"));
        }
        rows.push(format!("[{}]", row.join(",")));
    }
    format!("{{\"pae\":[{}]}}", rows.join(","))
}

#[test]
fn test_opt_in_zero_mutation_and_specs_03_unchanged() {
    let root = workspace_root();

    // Gate 1a: `docs/specs/03-assignment.md` must have 0-byte git diff.
    let diff_out = Command::new("git")
        .arg("-C")
        .arg(&root)
        .args(["diff", "--", "docs/specs/03-assignment.md"])
        .output()
        .expect("git diff specs/03-assignment.md");
    assert!(diff_out.status.success());
    assert!(
        diff_out.stdout.is_empty(),
        "docs/specs/03-assignment.md must remain 0-byte diff"
    );

    // Gate 1b: `af_priors: None` must execute bit-identically (`0.0 < 1e-15` diff) across U, P,
    // `standard`, `evo-graph`, and `bayes-sa`.
    let ref_data = read_star(root.join("data/fixtures/synthetic_ubiquitin.str"), None).unwrap();
    let (peaks, _truth) = simulate(
        &ref_data,
        &SimConfig {
            seed: 0,
            ..SimConfig::realistic_bruker()
        },
    );
    let systems = build_spin_systems(&peaks, &Tolerances::default(), 3, false).unwrap();
    let priors = Priors::default();
    let score_cfg = ScoreConfig::default();

    let u_plain = unary_matrix(&systems, &ref_data.sequence, &priors, &score_cfg);
    let u_none =
        unary_matrix_with_af_priors(&systems, &ref_data.sequence, &priors, &score_cfg, None);
    let p_plain = pair_matrix(&systems, &score_cfg);
    let p_none = pair_matrix(&systems, &score_cfg);

    let max_u_diff = u_plain
        .data
        .iter()
        .zip(u_none.data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    let max_p_diff = p_plain
        .data
        .iter()
        .zip(p_none.data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert_eq!(
        max_u_diff, 0.0,
        "U diff when af_priors=None must be 0.0 (< 1e-15)"
    );
    assert_eq!(
        max_p_diff, 0.0,
        "P diff when af_priors=None must be 0.0 (< 1e-15)"
    );

    let anneal_cfg = AnnealConfig {
        runs: 16,
        steps_per_system: 600,
        seed: 42,
        ..AnnealConfig::default()
    };
    let res_std_a = assign(
        &systems,
        &ref_data.sequence,
        &priors,
        &score_cfg,
        &anneal_cfg,
    );
    let res_std_b = assign_with_af_priors(
        &systems,
        &ref_data.sequence,
        &priors,
        &score_cfg,
        &anneal_cfg,
        None,
    );
    assert_eq!(res_std_a.position, res_std_b.position);
    assert!((res_std_a.score - res_std_b.score).abs() < 1e-15);
    for (ca, cb) in res_std_a.confidence.iter().zip(res_std_b.confidence.iter()) {
        assert!((ca - cb).abs() < 1e-15);
    }

    let evo_cfg = EvoGraphConfig {
        population_size: 16,
        generations: 3,
        steps_per_system_per_gen: 200,
        seed: 42,
        af_priors: None,
        ..EvoGraphConfig::default()
    };
    let (_, res_evo_a, _) =
        assign_evo_graph_from_peaks(&peaks, &ref_data.sequence, &priors, &evo_cfg);
    let (_, res_evo_b, _) =
        assign_evo_graph_from_peaks(&peaks, &ref_data.sequence, &priors, &evo_cfg);
    assert_eq!(res_evo_a.position, res_evo_b.position);
    assert!((res_evo_a.score - res_evo_b.score).abs() < 1e-15);

    let bayes_cfg = BayesSaConfig {
        replicas: 8,
        max_stages: 12,
        steps_per_stage_per_active: 80,
        seed: 42,
        ..BayesSaConfig::default()
    };
    let (_, res_bsa_a, _) =
        solve_bayes_sa_from_peaks(&peaks, &ref_data.sequence, &priors, &bayes_cfg).unwrap();
    let (_, res_bsa_b, _) =
        solve_bayes_sa_from_peaks(&peaks, &ref_data.sequence, &priors, &bayes_cfg).unwrap();
    assert_eq!(res_bsa_a.position, res_bsa_b.position);
    assert!((res_bsa_a.score - res_bsa_b.score).abs() < 1e-15);
}

#[test]
fn test_unlabelled_offset_registration_permutation_invariance_and_homolog_firewall() {
    let root = workspace_root();
    let ref_data = read_star(root.join("data/benchmark/bmr17769.str"), None).unwrap();
    let (peaks, _truth) = simulate(
        &ref_data,
        &SimConfig {
            seed: 2,
            ..SimConfig::realistic_bruker()
        },
    );
    let base_systems = build_spin_systems(&peaks, &Tolerances::default(), 3, false).unwrap();

    let model = load_ubiquitin_af_like_model(&root);
    let fw = ubiquitin_firewall();
    let af_cfg = AfAssignConfig::default();
    let af_priors = AfAssignPriors::from_structure_model(
        &model,
        &ref_data.sequence,
        Some(&ref_data.seq_ids),
        None,
        Vec::new(),
        &fw,
        &af_cfg,
    )
    .expect("build AfAssignPriors");

    // Verify 4-layer HomologFirewall provenance
    assert!(
        af_priors.shift_report.max_retained_seq_identity <= 0.25 + 1e-12,
        "HomologFirewall must enforce <= 25% sequence identity, got {}",
        af_priors.shift_report.max_retained_seq_identity
    );
    assert_eq!(af_priors.shift_report.exclusion_list_sha256.len(), 64);
    assert!(af_priors
        .shift_report
        .exclusion_list_sha256
        .chars()
        .all(|c| c.is_ascii_hexdigit()));
    assert!(
        !af_priors.shift_report.excluded_template_ids.is_empty(),
        "Ubiquitin itself (1D3Z / bmr6457) must be excluded from corpus"
    );

    // Plant realistic spectrometer referencing offsets onto spin systems
    let planted = (0.19, -1.55, 1.15, -0.88, 1.42);
    let base_off = estimate_af_shift_offsets(&base_systems, &af_priors);
    let shifted_systems = shift_spin_systems(
        &base_systems,
        planted.0,
        planted.1,
        planted.2,
        planted.3,
        planted.4,
    );
    let est = estimate_af_shift_offsets(&shifted_systems, &af_priors);

    // Shifting all observed shifts by `+delta` shifts the estimated offset by `+delta`
    assert!(
        ((est.h - base_off.h) - planted.0).abs() < 0.05,
        "H shift equivariance failed: {} vs {}",
        est.h - base_off.h,
        planted.0
    );
    assert!(
        ((est.n - base_off.n) - planted.1).abs() < 0.25,
        "N shift equivariance failed: {} vs {}",
        est.n - base_off.n,
        planted.1
    );
    assert!(
        ((est.ca - base_off.ca) - planted.2).abs() < 0.20,
        "CA shift equivariance failed: {} vs {}",
        est.ca - base_off.ca,
        planted.2
    );
    assert!(
        ((est.cb - base_off.cb) - planted.3).abs() < 0.25,
        "CB shift equivariance failed: {} vs {}",
        est.cb - base_off.cb,
        planted.3
    );
    assert!(
        ((est.c - base_off.c) - planted.4).abs() < 0.20,
        "C shift equivariance failed: {} vs {}",
        est.c - base_off.c,
        planted.4
    );

    // Verify exact `< 1e-12` permutation invariance across simultaneous random permutations
    // of BOTH spin systems AND predicted residue records (`0` ground-truth labels).
    let base_preds = af_priors.shift_report.residues.clone();
    let ref_est = estimate_af_shift_offsets_from_predictions(&shifted_systems, &base_preds);
    let mut rng = Rng::new(20261118);
    for trial in 0..8 {
        let mut perm_sys = shifted_systems.clone();
        for i in (1..perm_sys.len()).rev() {
            let j = rng.usize(i + 1);
            perm_sys.swap(i, j);
        }
        let mut perm_preds = base_preds.clone();
        for i in (1..perm_preds.len()).rev() {
            let j = rng.usize(i + 1);
            perm_preds.swap(i, j);
        }
        let perm_est = estimate_af_shift_offsets_from_predictions(&perm_sys, &perm_preds);
        let max_perm_diff = (ref_est.h - perm_est.h)
            .abs()
            .max((ref_est.n - perm_est.n).abs())
            .max((ref_est.ca - perm_est.ca).abs())
            .max((ref_est.cb - perm_est.cb).abs())
            .max((ref_est.c - perm_est.c).abs());
        assert!(
            max_perm_diff < 1e-12,
            "trial {trial}: unlabelled offset estimator violated < 1e-12 permutation invariance (diff={max_perm_diff:.3e})"
        );
    }
}

#[test]
fn test_student_t4_clamping_plddt_decay_and_120deg_corrupted_12aa_loop_gate() {
    // 1. Exact analytical checks on `ell_student_t4`, `w_AF(pLDDT)`, and hard per-residue clamp
    assert!((ell_student_t4(0.0) - 0.0).abs() < 1e-15);
    assert!((ell_student_t4(2.0) - (-2.5 * 2.0_f64.ln())).abs() < 1e-15);
    assert!((ell_student_t4(-4.0) - (-2.5 * 5.0_f64.ln())).abs() < 1e-15);

    let root = workspace_root();
    let ref_data = read_star(root.join("data/benchmark/bmr17769.str"), None).unwrap();
    let priors = Priors::default();
    let fw = ubiquitin_firewall();

    // Build baseline model and a severely hallucinated model where a contiguous 12-residue loop
    // (residues 40..52, spanning Ubiquitin's beta3-beta4-beta5 loop/turn) is corrupted by +120 deg
    // on BOTH phi and psi AND stamped with overconfident pLDDT = 95.0!
    let clean_model = load_ubiquitin_af_like_model(&root);
    let ca_coords: Vec<[f64; 3]> = clean_model.residues.iter().map(|r| r.ca).collect();
    let seq_ids: Vec<i32> = clean_model.residues.iter().map(|r| r.seq_id).collect();
    let res_names: Vec<String> = clean_model
        .residues
        .iter()
        .map(|r| r.res_name.clone())
        .collect();
    let mut phi: Vec<Option<f64>> = clean_model.residues.iter().map(|r| r.phi_deg).collect();
    let mut psi: Vec<Option<f64>> = clean_model.residues.iter().map(|r| r.psi_deg).collect();
    let chi1: Vec<Option<f64>> = clean_model.residues.iter().map(|r| r.chi1_deg).collect();

    for k in 40..52 {
        phi[k] = Some(match phi[k] {
            Some(v) => {
                let s = v + 120.0;
                if s > 180.0 {
                    s - 360.0
                } else {
                    s
                }
            }
            None => 60.0,
        });
        psi[k] = Some(match psi[k] {
            Some(v) => {
                let s = v + 120.0;
                if s > 180.0 {
                    s - 360.0
                } else {
                    s
                }
            }
            None => 60.0,
        });
    }
    let mut rebuilt_corrupted = StructureModel::from_ca_and_torsions(
        "A", &seq_ids, &res_names, &ca_coords, &phi, &psi, &chi1,
    )
    .expect("rebuild 3D coordinates with 120-deg corrupted 12-residue loop");
    for (idx, r) in rebuilt_corrupted.residues.iter_mut().enumerate() {
        r.plddt = Some(if (40..52).contains(&idx) {
            95.0 // overconfident hallucinated AF loop!
        } else if idx >= 71 {
            35.0 // disordered tail (`w_AF < 0.08`)
        } else {
            92.0
        });
    }

    let af_cfg = AfAssignConfig::default();
    assert_eq!(af_cfg.af_weight, DEFAULT_AF_WEIGHT);
    assert_eq!(af_cfg.af_clamp_nats, DEFAULT_AF_CLAMP_NATS);
    let af_corrupted = AfAssignPriors::from_structure_model(
        &rebuilt_corrupted,
        &ref_data.sequence,
        Some(&ref_data.seq_ids),
        None,
        Vec::new(),
        &fw,
        &af_cfg,
    )
    .expect("build corrupted AfAssignPriors");

    // Verify pLDDT < 50 damping (`w_AF < 0.08`)
    for k in 71..76 {
        assert!(
            af_corrupted.w_af_per_residue[k] < 0.08,
            "pLDDT < 50 tail must decay w_AF below 0.08, got {}",
            af_corrupted.w_af_per_residue[k]
        );
    }

    // Evaluate across 5 seeds on full triple-resonance experimental Ubiquitin (`bmr17769`)
    // across BOTH `standard` and `evo-graph`
    let mut base_std_correct = 0usize;
    let mut corr_std_correct = 0usize;
    let mut base_evo_correct = 0usize;
    let mut corr_evo_correct = 0usize;
    let mut total_assignable = 0usize;
    let mut max_abs_delta_u = 0.0_f64;

    for seed in 0..5u64 {
        let (peaks, truth) = simulate(
            &ref_data,
            &SimConfig {
                seed,
                p_missing_root: 0.0,
                ..SimConfig::default()
            },
        );
        let systems = build_spin_systems(&peaks, &Tolerances::default(), 3, false).unwrap();
        let score_cfg = ScoreConfig::default();

        if seed == 0 {
            let u_plain = unary_matrix(&systems, &ref_data.sequence, &priors, &score_cfg);
            let u_corr = unary_matrix_with_af_priors(
                &systems,
                &ref_data.sequence,
                &priors,
                &score_cfg,
                Some(&af_corrupted),
            );
            for s in 0..u_plain.rows {
                for k in 0..u_plain.cols {
                    if u_plain.get(s, k) > FORBID * 0.5 {
                        max_abs_delta_u =
                            max_abs_delta_u.max((u_corr.get(s, k) - u_plain.get(s, k)).abs());
                    }
                }
            }
        }

        let anneal_cfg = AnnealConfig {
            runs: 64,
            steps_per_system: 1500,
            seed,
            ..AnnealConfig::default()
        };
        let res_std_base = assign(
            &systems,
            &ref_data.sequence,
            &priors,
            &score_cfg,
            &anneal_cfg,
        );
        let res_std_corr = assign_with_af_priors(
            &systems,
            &ref_data.sequence,
            &priors,
            &score_cfg,
            &anneal_cfg,
            Some(&af_corrupted),
        );
        let m_std_base = score(&res_std_base, &truth, 0.8);
        let m_std_corr = score(&res_std_corr, &truth, 0.8);
        base_std_correct += m_std_base.correct;
        corr_std_correct += m_std_corr.correct;
        total_assignable += m_std_base.assignable;

        let evo_base_cfg = EvoGraphConfig {
            seed,
            af_priors: None,
            ..EvoGraphConfig::default()
        };
        let evo_corr_cfg = EvoGraphConfig {
            af_priors: Some(&af_corrupted),
            ..evo_base_cfg.clone()
        };
        let (_, res_evo_base, rep_base) =
            assign_evo_graph_from_peaks(&peaks, &ref_data.sequence, &priors, &evo_base_cfg);
        let (_, res_evo_corr, rep_corr) =
            assign_evo_graph_from_peaks(&peaks, &ref_data.sequence, &priors, &evo_corr_cfg);
        let truth_evo_base: Vec<i32> = rep_base
            .kept_hsqc_indices
            .iter()
            .map(|&i| truth[i])
            .collect();
        let truth_evo_corr: Vec<i32> = rep_corr
            .kept_hsqc_indices
            .iter()
            .map(|&i| truth[i])
            .collect();
        base_evo_correct += score(&res_evo_base, &truth_evo_base, 0.8).correct;
        corr_evo_correct += score(&res_evo_corr, &truth_evo_corr, 0.8).correct;
    }

    let ceiling = af_cfg.af_weight * af_cfg.af_clamp_nats;
    println!(
        "[120-deg 12-aa Loop Gate] max |Delta U_AF| = {max_abs_delta_u:.4} nats (ceiling={ceiling:.4} nats) | standard: {corr_std_correct}/{total_assignable} vs baseline {base_std_correct}/{total_assignable} | evo-graph: {corr_evo_correct}/{total_assignable} vs baseline {base_evo_correct}/{total_assignable}"
    );
    assert!(
        max_abs_delta_u <= ceiling + 1e-9,
        "max |Delta U_AF| ({max_abs_delta_u}) exceeded af_weight * af_clamp_nats ({ceiling})"
    );
    assert!(
        corr_std_correct >= base_std_correct,
        "120-deg corrupted 12-aa AF loop caused standard accuracy loss ({corr_std_correct} < {base_std_correct})"
    );
    assert!(
        corr_evo_correct >= base_evo_correct,
        "120-deg corrupted 12-aa AF loop caused evo-graph accuracy loss ({corr_evo_correct} < {base_evo_correct})"
    );
}

#[test]
fn test_pae_gated_15n_noesy_zero_false_interdomain_bonuses_and_nef_tsv_parsing() {
    let root = workspace_root();
    let ref_data = read_star(root.join("data/benchmark/bmr17769.str"), None).unwrap();
    let mut model = load_ubiquitin_af_like_model(&root);
    let n_res = model.residues.len();
    for r in &mut model.residues {
        r.plddt = Some(92.0);
    }
    // Mark residue 10 as low-pLDDT (64.0 < 70.0) to verify pLDDT contact rejection
    model.residues[10].plddt = Some(64.0);

    // Force an artificial inter-domain close amide contact between residue 14 (domain 1: 0..38)
    // and residue 56 (domain 2: 38..76) at d_NN = 2.85 A, while setting inter-domain PAE = 18.0 A (> 6.0 A)!
    let h14 = model.residues[14].h.expect("amide H on res 14");
    model.residues[56].h = Some([h14[0] + 2.85, h14[1], h14[2]]);
    assert!((v_dist(h14, model.residues[56].h.unwrap()) - 2.85).abs() < 1e-9);

    let mut pae_rows = Vec::with_capacity(n_res);
    for i in 0..n_res {
        let dom_i = usize::from(i >= 38);
        let mut row = Vec::with_capacity(n_res);
        for j in 0..n_res {
            let dom_j = usize::from(j >= 38);
            let v = if i == j {
                0.5
            } else if dom_i == dom_j {
                2.2
            } else {
                18.0 // unaligned inter-domain PAE > 6.0 A!
            };
            row.push(format!("{v:.1}"));
        }
        pae_rows.push(format!("[{}]", row.join(",")));
    }
    let two_dom_pae_json = format!("{{\"pae\":[{}]}}", pae_rows.join(","));
    let pae = parse_pae_json_str(&two_dom_pae_json).expect("parse 2-domain PAE");

    // Construct reciprocal 3D 15N-NOESY crosspeaks in both TSV and NEF formats, including the
    // false inter-domain (14 <-> 56) crosspeak!
    let noesy_peaks = simulate_amide_15n_noesy_peaks(&ref_data, &model, 99, 0.0, 0.0);
    let tsv_txt = format_noesy_15n_tsv(&noesy_peaks);
    let parsed_tsv = parse_noesy_15n_peaks_str(&tsv_txt).expect("parse NOESY TSV");
    assert_eq!(parsed_tsv.len(), noesy_peaks.len());

    let nef_txt = "\
data_nef_noesy_test
save_nef_nmr_spectrum_n15_noesy
   _nef_nmr_spectrum.sf_category          nef_nmr_spectrum
   _nef_nmr_spectrum.sf_framecode         nef_nmr_spectrum_n15_noesy
   _nef_nmr_spectrum.num_dimensions       3
   _nef_nmr_spectrum.experiment_type      '3D 15N-edited NOESY-HSQC'
   loop_
      _nef_spectrum_dimension.dimension_id
      _nef_spectrum_dimension.axis_unit
      _nef_spectrum_dimension.axis_code
      1 ppm 1H
      2 ppm 15N
      3 ppm 1H
   stop_
   loop_
      _nef_peak.index
      _nef_peak.position_1
      _nef_peak.position_2
      _nef_peak.position_3
      _nef_peak.height
      _nef_peak.volume
      1 8.250 121.40 7.910 150000.0 160000.0
      2 7.910 118.20 8.250 145000.0 155000.0
   stop_
save_
";
    let parsed_nef = parse_noesy_15n_peaks_str(nef_txt).expect("parse NOESY NEF");
    assert_eq!(parsed_nef.len(), 2);
    assert!((parsed_nef[0].h_direct_ppm - 8.250).abs() < 1e-6);
    assert!((parsed_nef[0].n15_ppm - 121.40).abs() < 1e-6);
    assert!((parsed_nef[0].h_indirect_ppm - 7.910).abs() < 1e-6);

    let fw = ubiquitin_firewall();
    let af_cfg = AfAssignConfig::default();

    // 1. With 2-domain PAE matrix: ZERO inter-domain (`PAE > 6 A`) or low-pLDDT (`pLDDT < 70`) contacts admitted!
    let af_with_pae = AfAssignPriors::from_structure_model(
        &model,
        &ref_data.sequence,
        Some(&ref_data.seq_ids),
        Some(&pae),
        parsed_tsv.clone(),
        &fw,
        &af_cfg,
    )
    .expect("build AfAssignPriors with 2-domain PAE");

    assert!(af_with_pae.contact_summary.admitted_contacts_total > 20);
    assert!(af_with_pae.contact_summary.rejected_high_pae_contacts >= 1);
    assert!(af_with_pae.contact_summary.rejected_low_plddt_contacts >= 1);
    for c in &af_with_pae.admitted_contacts {
        let dom_j = usize::from(c.seq_j >= 38);
        let dom_k = usize::from(c.seq_k >= 38);
        assert_eq!(
            dom_j, dom_k,
            "FALSE INTER-DOMAIN CONTACT ADMITTED across PAE=18 A boundary: ({}, {})",
            c.seq_j, c.seq_k
        );
        assert!(c.symmetrized_pae_angstrom.unwrap() <= 6.0 + 1e-9);
        assert!(c.plddt_j >= 70.0 && c.plddt_k >= 70.0);
        assert_ne!(c.seq_j, 10);
        assert_ne!(c.seq_k, 10);
    }

    // Check pairwise bonus table directly: every inter-domain pair `(j < 38, k >= 38)` gets EXACT 0.0 bonus!
    let (peaks, _truth) = simulate(
        &ref_data,
        &SimConfig {
            seed: 0,
            ..SimConfig::realistic_bruker()
        },
    );
    let systems = build_spin_systems(&peaks, &Tolerances::default(), 3, false).unwrap();
    let noe_table = af_with_pae
        .build_noesy_pairwise_table(&systems)
        .expect("non-empty AfNoesyPairwiseTable");

    let mut pos_seq_bonuses = 0usize;
    for k in 2..n_res {
        for s1 in 0..systems.len() {
            for s2 in 0..systems.len() {
                if s1 == s2 {
                    continue;
                }
                let b = noe_table.sequential_bonus(s1, s2, k);
                assert!(
                    (0.0..=AF_NOESY_MAX_SEQ_BONUS_NATS + 1e-9).contains(&b),
                    "sequential NOE bonus {b} out of [0, 2.50] nats"
                );
                if b > 0.0 {
                    pos_seq_bonuses += 1;
                }
            }
        }
    }
    assert!(
        pos_seq_bonuses > 0,
        "expected positive sequential NOE bonuses"
    );

    // Verify 0.0 inter-domain bonus across ALL spin system pairs!
    let mut at = vec![-1i32; n_res];
    at[14] = 0;
    at[56] = 1.min((systems.len() - 1) as i32);
    assert_eq!(
        noe_table.total_bonus(&at),
        0.0,
        "inter-domain pair (14, 56) with PAE = 18 A must receive exact 0.0 NOE bonus"
    );

    // 2. Without PAE JSON (`None`): ONLY `|j - k| <= 5` on the same contiguous chain may be admitted!
    let af_without_pae = AfAssignPriors::from_structure_model(
        &model,
        &ref_data.sequence,
        Some(&ref_data.seq_ids),
        None,
        parsed_tsv,
        &fw,
        &af_cfg,
    )
    .expect("build AfAssignPriors without PAE");
    assert!(af_without_pae.contact_summary.admitted_contacts_total > 10);
    for c in &af_without_pae.admitted_contacts {
        assert!(
            c.seq_k - c.seq_j <= 5,
            "without PAE JSON, contact ({}, {}) with |j - k| = {} > 5 must never be admitted",
            c.seq_j,
            c.seq_k,
            c.seq_k - c.seq_j
        );
    }
}

#[test]
fn test_sparse_cb_stripped_hnca_plus_15n_noesy_improves_accuracy_ge_8pp_and_cli() {
    let root = workspace_root();
    let ref_data = read_star(root.join("data/benchmark/bmr17769.str"), None).unwrap();
    let priors = Priors::default();
    let model = load_ubiquitin_af_like_model(&root);
    let pae_json = build_pae_json_for_ubiquitin(model.residues.len());
    let pae = parse_pae_json_str(&pae_json).expect("parse Ubiquitin PAE");
    let fw = ubiquitin_firewall();
    let af_cfg = AfAssignConfig::default();

    // Sparse, CB-stripped perdeuterated `HSQC + HNCA`-only regime (`0` CB, `0` CO, `0` HNCOCA/CBCACONH):
    // In large perdeuterated proteins where only HNCA + 3D 15N-NOESY are collected, sequential CA-only
    // matching suffers severe degeneracy and missing-bridge fragmentation without 3D shift priors +
    // PAE-gated 15N-NOESY amide-amide contacts.
    let sparse_sim_base = SimConfig {
        experiments: vec!["HSQC".to_string(), "HNCA".to_string()],
        p_missing_peak: 0.14,
        spurious_rate: 0.06,
        noise_h: 0.015,
        noise_n: 0.15,
        noise_c: 0.12,
        ..SimConfig::default()
    };

    let seeds = [0u64, 1, 2];
    let mut base_evo_correct = 0usize;
    let mut af_evo_correct = 0usize;
    let mut base_std_correct = 0usize;
    let mut af_std_correct = 0usize;
    let mut total_assignable = 0usize;

    for &seed in &seeds {
        let (peaks, truth) = simulate(
            &ref_data,
            &SimConfig {
                seed,
                ..sparse_sim_base.clone()
            },
        );
        let noesy_peaks = simulate_amide_15n_noesy_peaks(&ref_data, &model, seed, 0.10, 0.08);
        let af_priors = AfAssignPriors::from_structure_model(
            &model,
            &ref_data.sequence,
            Some(&ref_data.seq_ids),
            Some(&pae),
            noesy_peaks,
            &fw,
            &af_cfg,
        )
        .expect("build sparse HNCA + 15N-NOESY AfAssignPriors");

        // Verify zero high-PAE tail contacts (`>= 71`) were admitted
        for c in &af_priors.admitted_contacts {
            assert!(
                c.seq_j < 71 && c.seq_k < 71,
                "PAE=19.5 A C-tail contact ({}, {}) must be rejected",
                c.seq_j,
                c.seq_k
            );
        }

        // 1. Standard solver comparison
        let systems = build_spin_systems(&peaks, &Tolerances::default(), 3, false).unwrap();
        let score_cfg = ScoreConfig::default();
        let anneal_cfg = AnnealConfig {
            runs: 64,
            steps_per_system: 2500,
            seed,
            ..AnnealConfig::default()
        };
        let res_std_base = assign(
            &systems,
            &ref_data.sequence,
            &priors,
            &score_cfg,
            &anneal_cfg,
        );
        let res_std_af = assign_with_af_priors(
            &systems,
            &ref_data.sequence,
            &priors,
            &score_cfg,
            &anneal_cfg,
            Some(&af_priors),
        );
        let m_std_base = score(&res_std_base, &truth, 0.8);
        let m_std_af = score(&res_std_af, &truth, 0.8);
        base_std_correct += m_std_base.correct;
        af_std_correct += m_std_af.correct;

        // 2. Evo-graph solver comparison
        let evo_base = EvoGraphConfig {
            population_size: 32,
            generations: 5,
            steps_per_system_per_gen: 300,
            seed,
            af_priors: None,
            ..EvoGraphConfig::default()
        };
        let evo_af = EvoGraphConfig {
            af_priors: Some(&af_priors),
            ..evo_base.clone()
        };
        let (_, res_evo_base, rep_base) =
            assign_evo_graph_from_peaks(&peaks, &ref_data.sequence, &priors, &evo_base);
        let (_, res_evo_af, rep_af) =
            assign_evo_graph_from_peaks(&peaks, &ref_data.sequence, &priors, &evo_af);
        let truth_base: Vec<i32> = rep_base
            .kept_hsqc_indices
            .iter()
            .map(|&i| truth[i])
            .collect();
        let truth_af: Vec<i32> = rep_af.kept_hsqc_indices.iter().map(|&i| truth[i]).collect();
        let m_evo_base = score(&res_evo_base, &truth_base, 0.8);
        let m_evo_af = score(&res_evo_af, &truth_af, 0.8);
        base_evo_correct += m_evo_base.correct;
        af_evo_correct += m_evo_af.correct;
        total_assignable += m_evo_base.assignable;
    }

    let base_std_acc = 100.0 * (base_std_correct as f64) / (total_assignable as f64);
    let af_std_acc = 100.0 * (af_std_correct as f64) / (total_assignable as f64);
    let gain_std_pp = af_std_acc - base_std_acc;

    let base_evo_acc = 100.0 * (base_evo_correct as f64) / (total_assignable as f64);
    let af_evo_acc = 100.0 * (af_evo_correct as f64) / (total_assignable as f64);
    let gain_evo_pp = af_evo_acc - base_evo_acc;

    println!(
        "[Sparse CB-Stripped HNCA + 15N-NOESY Gate ({} seeds)] standard: {:.2}% -> {:.2}% ({:+.2} pp) | evo-graph: {:.2}% -> {:.2}% ({:+.2} pp)",
        seeds.len(),
        base_std_acc,
        af_std_acc,
        gain_std_pp,
        base_evo_acc,
        af_evo_acc,
        gain_evo_pp,
    );
    assert!(
        gain_std_pp >= 8.0,
        "standard solver gain on sparse CB-stripped HNCA + 15N-NOESY ({gain_std_pp:+.2} pp) must be >= +8.0 pp"
    );
    assert!(
        gain_evo_pp >= 8.0,
        "evo-graph solver gain on sparse CB-stripped HNCA + 15N-NOESY ({gain_evo_pp:+.2} pp) must be >= +8.0 pp"
    );

    // 3. End-to-end `ferric assign --af-structure ... --af-pae ... --noesy-15n ...` CLI integration check
    let tmp_dir =
        std::env::temp_dir().join(format!("ferric_t21_4_cli_test_{}", std::process::id()));
    let _ = fs::remove_dir_all(&tmp_dir);
    fs::create_dir_all(&tmp_dir).expect("create temp dir");

    let (cli_peaks, _cli_truth) = simulate(
        &ref_data,
        &SimConfig {
            seed: 0,
            ..sparse_sim_base
        },
    );
    let cli_noesy = simulate_amide_15n_noesy_peaks(&ref_data, &model, 0, 0.10, 0.05);
    let peaks_path = tmp_dir.join("hnca_sparse.tsv");
    let noesy_path = tmp_dir.join("noesy_15n.tsv");
    let pae_path = tmp_dir.join("pae.json");
    let out_prefix = tmp_dir.join("assigned_af");
    let json_out = tmp_dir.join("assigned_af.json");
    let nef_out = tmp_dir.join("assigned_af.nef");

    write_peaks(&peaks_path, &cli_peaks).unwrap();
    fs::write(&noesy_path, format_noesy_15n_tsv(&cli_noesy)).unwrap();
    fs::write(&pae_path, &pae_json).unwrap();

    // Invoke pre-built `ferric` binary when present, falling back to `cargo run` with matching profile
    let run_ferric = |cli_args: &[&str]| -> std::process::Output {
        let profile_dir = if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        };
        let bin_name = if cfg!(windows) {
            "ferric.exe"
        } else {
            "ferric"
        };
        let candidate = root.join("target").join(profile_dir).join(bin_name);
        if candidate.is_file() {
            Command::new(&candidate)
                .current_dir(&root)
                .args(cli_args)
                .output()
                .expect("run ferric binary")
        } else {
            let mut cmd = Command::new("cargo");
            cmd.current_dir(&root).arg("run").arg("--quiet");
            if !cfg!(debug_assertions) {
                cmd.arg("--release");
            }
            cmd.args(["-p", "nmr-cli", "--bin", "ferric", "--"])
                .args(cli_args)
                .output()
                .expect("cargo run ferric")
        }
    };

    let status = run_ferric(&[
        "assign",
        "--peaks",
        peaks_path.to_str().unwrap(),
        "--star",
        root.join("data/benchmark/bmr17769.str").to_str().unwrap(),
        "--af-structure",
        root.join("data/pdb_refs/1d3z.pdb").to_str().unwrap(),
        "--af-pae",
        pae_path.to_str().unwrap(),
        "--noesy-15n",
        noesy_path.to_str().unwrap(),
        "--force-bfactor-plddt",
        "--exclude-pdb",
        "1D3Z,1UBQ,1G6J",
        "--exclude-bmrb",
        "6457,17769,5387",
        "--exclude-topology",
        "2.60.98",
        "--af-weight",
        "0.65",
        "--af-clamp-nats",
        "4.0",
        "--runs",
        "16",
        "--steps",
        "250",
        "--out",
        out_prefix.to_str().unwrap(),
        "--json-out",
        json_out.to_str().unwrap(),
        "--nef-out",
        nef_out.to_str().unwrap(),
    ]);
    assert!(
        status.status.success(),
        "ferric assign --af-structure failed: stdout={}, stderr={}",
        String::from_utf8_lossy(&status.stdout),
        String::from_utf8_lossy(&status.stderr)
    );

    let json_body = fs::read_to_string(&json_out).expect("read json_out");
    assert!(json_body.contains("\"af_priors\""));
    assert!(json_body.contains("\"exclusion_list_sha256\""));
    assert!(json_body.contains("\"admitted_contacts_total\""));
    let str_body = fs::read_to_string(out_prefix.with_extension("str")).expect("read str_out");
    assert!(str_body.contains("af_priors("));
    assert!(nef_out.exists());

    // Also verify passing `--noesy-15n` WITHOUT `--af-structure` fails cleanly!
    let err_status = run_ferric(&[
        "assign",
        "--peaks",
        peaks_path.to_str().unwrap(),
        "--star",
        root.join("data/benchmark/bmr17769.str").to_str().unwrap(),
        "--noesy-15n",
        noesy_path.to_str().unwrap(),
        "--out",
        tmp_dir.join("should_fail").to_str().unwrap(),
    ]);
    assert!(
        !err_status.status.success(),
        "--noesy-15n without --af-structure must be rejected"
    );

    let _ = fs::remove_dir_all(&tmp_dir);
}
