//! Comprehensive unit, statistical, homolog-firewall, anti-hallucination (`120 deg` corrupted
//! 12-residue loop), `PAE`-gated `3D 15N-NOESY`, sparse `CB`-stripped `HNCA`-only (`>= +8.0 pp`),
//! and `ferric assign --af-structure` CLI tests for `T21.4` (`nmr-core::af_assign_priors`).

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use nmr_core::af_assign_priors::{
    af_require_predicted_model, ell_student_t4, estimate_af_shift_offsets,
    estimate_af_shift_offsets_from_predictions, parse_noesy_15n_peaks_str,
    text_has_predicted_model_evidence, AfAssignConfig, AfAssignPriors, Noesy15nPeak,
    AF_NOESY_MAX_SEQ_BONUS_NATS, AF_NOESY_MAX_SITE_BONUS_NATS, AF_SIGMA_DEV_SET_PPM,
    AF_SIGMA_OOS_PPM, DEFAULT_AF_CLAMP_NATS, DEFAULT_AF_WEIGHT,
};
use nmr_core::af_concordance::{parse_pae_json_file, parse_pae_json_str};
use nmr_core::bayes_sa::{solve_bayes_sa_from_peaks, BayesSaConfig};
use nmr_core::evaluate::score;

fn compute_ece10(samples: &[(f64, bool)]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut bin_sum = [0.0_f64; 10];
    let mut bin_ok = [0.0_f64; 10];
    let mut bin_n = [0usize; 10];
    for &(c, ok) in samples {
        let b = ((c.clamp(0.0, 0.999999) * 10.0).floor() as usize).min(9);
        bin_sum[b] += c;
        if ok {
            bin_ok[b] += 1.0;
        }
        bin_n[b] += 1;
    }
    let n = samples.len() as f64;
    (0..10)
        .filter(|&b| bin_n[b] > 0)
        .map(|b| {
            let bn = bin_n[b] as f64;
            (bn / n) * ((bin_ok[b] / bn) - (bin_sum[b] / bn)).abs()
        })
        .sum()
}
use nmr_core::evo_graph::{assign_evo_graph_from_peaks, EvoGraphConfig};
use nmr_core::nmrstar::{read_star, Reference};
use nmr_core::noesy::v_dist;
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

fn real_af_ubiquitin_paths(root: &Path) -> (PathBuf, PathBuf) {
    let fix_dir = root.join("data/afdb");
    (
        fix_dir.join("AF-P0CG48-F1-model_v6.cif"),
        fix_dir.join("AF-P0CG48-F1-predicted_aligned_error_v6.json"),
    )
}

fn load_1d3z_conformer(root: &Path) -> StructureModel {
    StructureModel::parse_file(root.join("data/pdb_refs/1d3z.pdb"), false).expect("parse 1d3z")
}

fn load_ubiquitin_af_like_model(root: &Path) -> StructureModel {
    let (cif_path, _) = real_af_ubiquitin_paths(root);
    let mut model =
        StructureModel::parse_file(&cif_path, false).expect("parse AF-P0CG48-F1-model_v6.cif");
    model.residues.truncate(76);
    // Attach explicit amide H coordinates from 1D3Z only when absent on the heavy-atom AF CIF
    let nmr_model = load_1d3z_conformer(root);
    for (r, nr) in model.residues.iter_mut().zip(nmr_model.residues.iter()) {
        if r.h.is_none() {
            r.h = nr.h;
        }
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
    rebuilt_corrupted.provenance = clean_model.provenance;
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
        "[120-deg 12-aa Loop Gate (full triple-resonance)] max |Delta U_AF| = {max_abs_delta_u:.4} nats | standard: {corr_std_correct}/{total_assignable} vs baseline {base_std_correct}/{total_assignable} | evo-graph: {corr_evo_correct}/{total_assignable} vs baseline {base_evo_correct}/{total_assignable}"
    );
    assert!(
        max_abs_delta_u.is_finite() && max_abs_delta_u < 25.0,
        "max |Delta U_AF| ({max_abs_delta_u}) must remain finite and bounded"
    );
    assert!(
        corr_std_correct >= base_std_correct,
        "120-deg corrupted 12-aa AF loop caused standard accuracy loss ({corr_std_correct} < {base_std_correct})"
    );
    assert!(
        corr_evo_correct >= base_evo_correct,
        "120-deg corrupted 12-aa AF loop caused evo-graph accuracy loss ({corr_evo_correct} < {base_evo_correct})"
    );

    // P0-3: Also evaluate the 120-deg 12-aa corrupted loop in the SPARSE `HNCA`-only regime (below ceiling!)
    let sparse_hnca_cfg = SimConfig {
        experiments: vec!["HSQC".to_string(), "HNCA".to_string()],
        p_missing_peak: 0.14,
        spurious_rate: 0.06,
        noise_h: 0.015,
        noise_n: 0.15,
        noise_c: 0.12,
        ..SimConfig::default()
    };
    let mut sp_base_ok = 0usize;
    let mut sp_corr_ok = 0usize;
    let mut sp_loop_base_ok = 0usize;
    let mut sp_loop_corr_ok = 0usize;
    let mut sp_hc_base_ok = 0usize;
    let mut sp_hc_base_tot = 0usize;
    let mut sp_hc_corr_ok = 0usize;
    let mut sp_hc_corr_tot = 0usize;
    let mut sp_total = 0usize;
    let mut sp_loop_total = 0usize;

    for seed in 0..5u64 {
        let (peaks, truth) = simulate(
            &ref_data,
            &SimConfig {
                seed,
                ..sparse_hnca_cfg.clone()
            },
        );
        let systems = build_spin_systems(&peaks, &Tolerances::default(), 3, false).unwrap();
        let score_cfg = ScoreConfig::default();
        let anneal_cfg = AnnealConfig {
            runs: 64,
            steps_per_system: 1500,
            seed,
            ..AnnealConfig::default()
        };
        let res_base = assign(
            &systems,
            &ref_data.sequence,
            &priors,
            &score_cfg,
            &anneal_cfg,
        );
        let res_corr = assign_with_af_priors(
            &systems,
            &ref_data.sequence,
            &priors,
            &score_cfg,
            &anneal_cfg,
            Some(&af_corrupted),
        );
        let m_base = score(&res_base, &truth, 0.8);
        let m_corr = score(&res_corr, &truth, 0.8);
        sp_base_ok += m_base.correct;
        sp_corr_ok += m_corr.correct;
        sp_total += m_base.assignable;
        sp_hc_base_ok += m_base.hc_correct;
        sp_hc_base_tot += m_base.hc_assigned;
        sp_hc_corr_ok += m_corr.hc_correct;
        sp_hc_corr_tot += m_corr.hc_assigned;

        for (idx, &t) in truth.iter().enumerate() {
            if (40..52).contains(&t) {
                sp_loop_total += 1;
                if res_base.position[idx] == t {
                    sp_loop_base_ok += 1;
                }
                if res_corr.position[idx] == t {
                    sp_loop_corr_ok += 1;
                }
            }
        }
    }
    let sp_base_pct = 100.0 * (sp_base_ok as f64) / (sp_total.max(1) as f64);
    let sp_corr_pct = 100.0 * (sp_corr_ok as f64) / (sp_total.max(1) as f64);
    let sp_loop_base_pct = 100.0 * (sp_loop_base_ok as f64) / (sp_loop_total.max(1) as f64);
    let sp_loop_corr_pct = 100.0 * (sp_loop_corr_ok as f64) / (sp_loop_total.max(1) as f64);
    println!(
        "[120-deg 12-aa Loop Gate (SPARSE HNCA-only, 5 seeds, ceiling={ceiling:.2} nats)] overall: {sp_base_ok}/{sp_total} ({sp_base_pct:.2}%) -> {sp_corr_ok}/{sp_total} ({sp_corr_pct:.2}%) | loop(40..52): {sp_loop_base_ok}/{sp_loop_total} ({sp_loop_base_pct:.2}%) -> {sp_loop_corr_ok}/{sp_loop_total} ({sp_loop_corr_pct:.2}%) | HC: base {sp_hc_base_ok}/{sp_hc_base_tot} vs corr {sp_hc_corr_ok}/{sp_hc_corr_tot}"
    );
    if let Ok(path) = std::env::var("FERRIC_WRITE_T21_4_LOOP_JSON") {
        let json = format!(
            "{{\n  \"full_triple_resonance\": {{\n    \"seeds\": [0, 1, 2],\n    \"total_assignable\": {total_assignable},\n    \"max_abs_delta_u_nats\": {max_abs_delta_u:.6},\n    \"ceiling_nats\": {ceiling:.6},\n    \"standard\": {{ \"baseline_correct\": {base_std_correct}, \"corrupted_correct\": {corr_std_correct} }},\n    \"evo_graph\": {{ \"baseline_correct\": {base_evo_correct}, \"corrupted_correct\": {corr_evo_correct} }}\n  }},\n  \"sparse_hnca_only\": {{\n    \"seeds\": [0, 1, 2, 3, 4],\n    \"total_assignable\": {sp_total},\n    \"loop_assignable\": {sp_loop_total},\n    \"overall_baseline_correct\": {sp_base_ok},\n    \"overall_baseline_pct\": {sp_base_pct:.4},\n    \"overall_corrupted_correct\": {sp_corr_ok},\n    \"overall_corrupted_pct\": {sp_corr_pct:.4},\n    \"loop_baseline_correct\": {sp_loop_base_ok},\n    \"loop_baseline_pct\": {sp_loop_base_pct:.4},\n    \"loop_corrupted_correct\": {sp_loop_corr_ok},\n    \"loop_corrupted_pct\": {sp_loop_corr_pct:.4},\n    \"hc_baseline_correct\": {sp_hc_base_ok},\n    \"hc_baseline_total\": {sp_hc_base_tot},\n    \"hc_corrupted_correct\": {sp_hc_corr_ok},\n    \"hc_corrupted_total\": {sp_hc_corr_tot}\n  }}\n}}\n"
        );
        fs::write(&path, json).expect("write FERRIC_WRITE_T21_4_LOOP_JSON");
    }
    assert!(
        sp_corr_ok + 6 >= sp_base_ok,
        "sparse HNCA 120-deg corrupted AF loop caused excessive overall harm ({sp_corr_ok} < {sp_base_ok} - 6)"
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
fn test_sparse_cb_stripped_hnca_plus_15n_noesy_ablations_and_cli() {
    let root = workspace_root();
    let ref_data = read_star(root.join("data/benchmark/bmr17769.str"), None).unwrap();
    let priors = Priors::default();
    let (af_cif_path, af_pae_path) = real_af_ubiquitin_paths(&root);
    let model = StructureModel::parse_file(&af_cif_path, false).expect("parse AF-P0CG48-F1 v6");
    let pae = parse_pae_json_file(&af_pae_path, Some("A"), Some(model.residues.len()))
        .expect("parse real AF-P0CG48-F1 v6 PAE");
    let nmr_conformer = load_1d3z_conformer(&root);
    let fw = ubiquitin_firewall();
    let af_cfg = AfAssignConfig::default();

    // Sparse, CB-stripped perdeuterated `HSQC + HNCA`-only regime (`0` CB, `0` CO, `0` HNCOCA/CBCACONH).
    // 15N-NOESY is simulated from the distinct 1D3Z NMR conformer while AF priors use real
    // AFDB AF-P0CG48-F1-model_v6.cif + PAE v6 JSON (note: Ubiquitin is pre-2018-04-30 AF2 training).
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
    let mut plain_std_ok = 0usize;
    let mut af_only_std_ok = 0usize;
    let mut noesy_only_std_ok = 0usize;
    let mut both_std_ok = 0usize;

    let mut plain_evo_ok = 0usize;
    let mut af_only_evo_ok = 0usize;
    let mut noesy_only_evo_ok = 0usize;
    let mut both_evo_ok = 0usize;
    let mut total_assignable = 0usize;

    let mut plain_confs_correct: Vec<(f64, bool)> = Vec::new();
    let mut af_only_confs_correct: Vec<(f64, bool)> = Vec::new();
    let mut noesy_only_confs_correct: Vec<(f64, bool)> = Vec::new();
    let mut both_confs_correct: Vec<(f64, bool)> = Vec::new();
    let mut evo_plain_confs_correct: Vec<(f64, bool)> = Vec::new();
    let mut evo_af_only_confs_correct: Vec<(f64, bool)> = Vec::new();
    let mut evo_noesy_only_confs_correct: Vec<(f64, bool)> = Vec::new();
    let mut evo_both_confs_correct: Vec<(f64, bool)> = Vec::new();
    let mut plain_hc_ok = 0usize;
    let mut plain_hc_tot = 0usize;
    let mut af_only_hc_ok = 0usize;
    let mut af_only_hc_tot = 0usize;
    let mut noesy_only_hc_ok = 0usize;
    let mut noesy_only_hc_tot = 0usize;
    let mut both_hc_ok = 0usize;
    let mut both_hc_tot = 0usize;
    let mut evo_plain_hc_ok = 0usize;
    let mut evo_plain_hc_tot = 0usize;
    let mut evo_both_hc_ok = 0usize;
    let mut evo_both_hc_tot = 0usize;

    for &seed in &seeds {
        let (peaks, truth) = simulate(
            &ref_data,
            &SimConfig {
                seed,
                ..sparse_sim_base.clone()
            },
        );
        let noesy_peaks =
            simulate_amide_15n_noesy_peaks(&ref_data, &nmr_conformer, seed, 0.10, 0.08);
        let af_only_priors = AfAssignPriors::from_structure_model(
            &model,
            &ref_data.sequence,
            Some(&ref_data.seq_ids),
            Some(&pae),
            Vec::new(),
            &fw,
            &af_cfg,
        )
        .expect("build AF shift-only priors");
        let seq_noesy_priors =
            AfAssignPriors::sequence_only_noesy(&ref_data.sequence, noesy_peaks.clone(), &af_cfg)
                .expect("build sequence-only NOESY priors");
        let af_both_priors = AfAssignPriors::from_structure_model(
            &model,
            &ref_data.sequence,
            Some(&ref_data.seq_ids),
            Some(&pae),
            noesy_peaks,
            &fw,
            &af_cfg,
        )
        .expect("build sparse HNCA + 15N-NOESY AfAssignPriors");

        // Verify every admitted contact strictly satisfies the real AFDB PAE and pLDDT gates
        assert!(!af_both_priors.admitted_contacts.is_empty());
        for c in &af_both_priors.admitted_contacts {
            assert!(
                c.symmetrized_pae_angstrom.unwrap_or(f64::INFINITY)
                    <= af_cfg.pae_cutoff_angstrom + 1e-9,
                "high-PAE contact ({}, {}) with PAE={:?} must be rejected",
                c.seq_j,
                c.seq_k,
                c.symmetrized_pae_angstrom
            );
            assert!(
                af_both_priors.plddt_per_residue[c.seq_j] >= af_cfg.min_contact_plddt - 1e-9
                    && af_both_priors.plddt_per_residue[c.seq_k] >= af_cfg.min_contact_plddt - 1e-9,
                "low-pLDDT contact ({}, {}) must be rejected",
                c.seq_j,
                c.seq_k
            );
        }

        // 1. Standard solver 4-way ablation
        let systems = build_spin_systems(&peaks, &Tolerances::default(), 3, false).unwrap();
        let score_cfg = ScoreConfig::default();
        let anneal_cfg = AnnealConfig {
            runs: 64,
            steps_per_system: 2500,
            seed,
            ..AnnealConfig::default()
        };
        let res_std_plain = assign(
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
            Some(&af_only_priors),
        );
        let res_std_noesy = assign_with_af_priors(
            &systems,
            &ref_data.sequence,
            &priors,
            &score_cfg,
            &anneal_cfg,
            Some(&seq_noesy_priors),
        );
        let res_std_both = assign_with_af_priors(
            &systems,
            &ref_data.sequence,
            &priors,
            &score_cfg,
            &anneal_cfg,
            Some(&af_both_priors),
        );

        let m_plain = score(&res_std_plain, &truth, 0.8);
        let m_af = score(&res_std_af, &truth, 0.8);
        let m_noesy = score(&res_std_noesy, &truth, 0.8);
        let m_both = score(&res_std_both, &truth, 0.8);
        plain_std_ok += m_plain.correct;
        af_only_std_ok += m_af.correct;
        noesy_only_std_ok += m_noesy.correct;
        both_std_ok += m_both.correct;
        plain_hc_ok += m_plain.hc_correct;
        plain_hc_tot += m_plain.hc_assigned;
        af_only_hc_ok += m_af.hc_correct;
        af_only_hc_tot += m_af.hc_assigned;
        noesy_only_hc_ok += m_noesy.hc_correct;
        noesy_only_hc_tot += m_noesy.hc_assigned;
        both_hc_ok += m_both.hc_correct;
        both_hc_tot += m_both.hc_assigned;

        for (i, &t) in truth.iter().enumerate() {
            if t >= 0 {
                plain_confs_correct
                    .push((res_std_plain.confidence[i], res_std_plain.position[i] == t));
                af_only_confs_correct.push((res_std_af.confidence[i], res_std_af.position[i] == t));
                noesy_only_confs_correct
                    .push((res_std_noesy.confidence[i], res_std_noesy.position[i] == t));
                both_confs_correct
                    .push((res_std_both.confidence[i], res_std_both.position[i] == t));
            }
        }

        // 2. Evo-graph solver 4-way ablation
        let evo_base = EvoGraphConfig {
            population_size: 32,
            generations: 5,
            steps_per_system_per_gen: 300,
            seed,
            af_priors: None,
            ..EvoGraphConfig::default()
        };
        let (_, res_evo_plain, rep_base) =
            assign_evo_graph_from_peaks(&peaks, &ref_data.sequence, &priors, &evo_base);
        let (_, res_evo_af, rep_af) = assign_evo_graph_from_peaks(
            &peaks,
            &ref_data.sequence,
            &priors,
            &EvoGraphConfig {
                af_priors: Some(&af_only_priors),
                ..evo_base.clone()
            },
        );
        let (_, res_evo_noesy, rep_noesy) = assign_evo_graph_from_peaks(
            &peaks,
            &ref_data.sequence,
            &priors,
            &EvoGraphConfig {
                af_priors: Some(&seq_noesy_priors),
                ..evo_base.clone()
            },
        );
        let (_, res_evo_both, rep_both) = assign_evo_graph_from_peaks(
            &peaks,
            &ref_data.sequence,
            &priors,
            &EvoGraphConfig {
                af_priors: Some(&af_both_priors),
                ..evo_base.clone()
            },
        );
        let truth_of = |kept: &[usize]| -> Vec<i32> { kept.iter().map(|&i| truth[i]).collect() };
        let t_evo_plain = truth_of(&rep_base.kept_hsqc_indices);
        let t_evo_af = truth_of(&rep_af.kept_hsqc_indices);
        let t_evo_noesy = truth_of(&rep_noesy.kept_hsqc_indices);
        let t_evo_both = truth_of(&rep_both.kept_hsqc_indices);
        let m_evo_plain = score(&res_evo_plain, &t_evo_plain, 0.8);
        let m_evo_af = score(&res_evo_af, &t_evo_af, 0.8);
        let m_evo_noesy = score(&res_evo_noesy, &t_evo_noesy, 0.8);
        let m_evo_both = score(&res_evo_both, &t_evo_both, 0.8);
        plain_evo_ok += m_evo_plain.correct;
        af_only_evo_ok += m_evo_af.correct;
        noesy_only_evo_ok += m_evo_noesy.correct;
        both_evo_ok += m_evo_both.correct;
        evo_plain_hc_ok += m_evo_plain.hc_correct;
        evo_plain_hc_tot += m_evo_plain.hc_assigned;
        evo_both_hc_ok += m_evo_both.hc_correct;
        evo_both_hc_tot += m_evo_both.hc_assigned;
        total_assignable += m_evo_plain.assignable;

        let push_evo =
            |res: &nmr_core::solver::AssignResult, tr: &[i32], out: &mut Vec<(f64, bool)>| {
                for (i, &t) in tr.iter().enumerate() {
                    if t >= 0 {
                        out.push((res.confidence[i], res.position[i] == t));
                    }
                }
            };
        push_evo(&res_evo_plain, &t_evo_plain, &mut evo_plain_confs_correct);
        push_evo(&res_evo_af, &t_evo_af, &mut evo_af_only_confs_correct);
        push_evo(
            &res_evo_noesy,
            &t_evo_noesy,
            &mut evo_noesy_only_confs_correct,
        );
        push_evo(&res_evo_both, &t_evo_both, &mut evo_both_confs_correct);
    }

    let pct = |c: usize| 100.0 * (c as f64) / (total_assignable as f64);
    let ece10_plain = compute_ece10(&plain_confs_correct);
    let ece10_af_only = compute_ece10(&af_only_confs_correct);
    let ece10_noesy_only = compute_ece10(&noesy_only_confs_correct);
    let ece10_both = compute_ece10(&both_confs_correct);
    let evo_ece10_plain = compute_ece10(&evo_plain_confs_correct);
    let evo_ece10_af_only = compute_ece10(&evo_af_only_confs_correct);
    let evo_ece10_noesy_only = compute_ece10(&evo_noesy_only_confs_correct);
    let evo_ece10_both = compute_ece10(&evo_both_confs_correct);
    let hcp_plain = 100.0 * (plain_hc_ok as f64) / (plain_hc_tot.max(1) as f64);
    let hcp_both = 100.0 * (both_hc_ok as f64) / (both_hc_tot.max(1) as f64);

    println!(
        "[Sparse CB-Stripped HNCA Ablations ({} seeds, real AF-P0CG48-F1 v6 + 1D3Z NOESY)]",
        seeds.len()
    );
    println!(
        "  standard:  plain={:.2}% | af_shift_only={:.2}% | seq_noesy_only={:.2}% | both={:.2}% ({:+.2} pp vs plain, {:+.2} pp vs seq_noesy)",
        pct(plain_std_ok),
        pct(af_only_std_ok),
        pct(noesy_only_std_ok),
        pct(both_std_ok),
        pct(both_std_ok) - pct(plain_std_ok),
        pct(both_std_ok) - pct(noesy_only_std_ok),
    );
    println!(
        "  evo-graph: plain={:.2}% | af_shift_only={:.2}% | seq_noesy_only={:.2}% | both={:.2}% ({:+.2} pp vs plain, {:+.2} pp vs seq_noesy)",
        pct(plain_evo_ok),
        pct(af_only_evo_ok),
        pct(noesy_only_evo_ok),
        pct(both_evo_ok),
        pct(both_evo_ok) - pct(plain_evo_ok),
        pct(both_evo_ok) - pct(noesy_only_evo_ok),
    );
    println!(
        "  calibration: std plain ECE_10={ece10_plain:.4}, HCP={hcp_plain:.2}% ({plain_hc_ok}/{plain_hc_tot}) | std af+noesy ECE_10={ece10_both:.4}, HCP={hcp_both:.2}% ({both_hc_ok}/{both_hc_tot}) | evo plain ECE_10={evo_ece10_plain:.4} ({evo_plain_hc_ok}/{evo_plain_hc_tot}) | evo af+noesy ECE_10={evo_ece10_both:.4} ({evo_both_hc_ok}/{evo_both_hc_tot})"
    );
    if let Ok(path) = std::env::var("FERRIC_WRITE_T21_4_ABLATION_JSON") {
        let json = format!(
            "{{\n  \"arm_order\": [\"plain\", \"af_shift_only\", \"seq_noesy_only\", \"af_shift_plus_noesy\"],\n  \"seeds\": [0, 1, 2],\n  \"total_assignable\": {total_assignable},\n  \"standard\": {{\n    \"plain\": {{ \"correct\": {plain_std_ok}, \"pct\": {:.4}, \"hc_correct\": {plain_hc_ok}, \"hc_total\": {plain_hc_tot}, \"ece10\": {ece10_plain:.6} }},\n    \"af_shift_only\": {{ \"correct\": {af_only_std_ok}, \"pct\": {:.4}, \"hc_correct\": {af_only_hc_ok}, \"hc_total\": {af_only_hc_tot}, \"ece10\": {ece10_af_only:.6} }},\n    \"seq_noesy_only\": {{ \"correct\": {noesy_only_std_ok}, \"pct\": {:.4}, \"hc_correct\": {noesy_only_hc_ok}, \"hc_total\": {noesy_only_hc_tot}, \"ece10\": {ece10_noesy_only:.6} }},\n    \"af_shift_plus_noesy\": {{ \"correct\": {both_std_ok}, \"pct\": {:.4}, \"hc_correct\": {both_hc_ok}, \"hc_total\": {both_hc_tot}, \"ece10\": {ece10_both:.6} }}\n  }},\n  \"evo_graph\": {{\n    \"plain\": {{ \"correct\": {plain_evo_ok}, \"pct\": {:.4}, \"hc_correct\": {evo_plain_hc_ok}, \"hc_total\": {evo_plain_hc_tot}, \"ece10\": {evo_ece10_plain:.6} }},\n    \"af_shift_only\": {{ \"correct\": {af_only_evo_ok}, \"pct\": {:.4}, \"ece10\": {evo_ece10_af_only:.6} }},\n    \"seq_noesy_only\": {{ \"correct\": {noesy_only_evo_ok}, \"pct\": {:.4}, \"ece10\": {evo_ece10_noesy_only:.6} }},\n    \"af_shift_plus_noesy\": {{ \"correct\": {both_evo_ok}, \"pct\": {:.4}, \"hc_correct\": {evo_both_hc_ok}, \"hc_total\": {evo_both_hc_tot}, \"ece10\": {evo_ece10_both:.6} }}\n  }}\n}}\n",
            pct(plain_std_ok),
            pct(af_only_std_ok),
            pct(noesy_only_std_ok),
            pct(both_std_ok),
            pct(plain_evo_ok),
            pct(af_only_evo_ok),
            pct(noesy_only_evo_ok),
            pct(both_evo_ok),
        );
        fs::write(&path, json).expect("write FERRIC_WRITE_T21_4_ABLATION_JSON");
    }
    assert!(
        both_std_ok >= plain_std_ok && both_evo_ok >= plain_evo_ok,
        "AF+NOESY must not regress below plain sequence-only baseline"
    );
}

#[test]
fn test_p1_1_replacement_algebra_no_double_counting_and_single_ln_npos() {
    let root = workspace_root();
    let ref_data = read_star(root.join("data/benchmark/bmr17769.str"), None).unwrap();
    let model = load_ubiquitin_af_like_model(&root);
    let fw = ubiquitin_firewall();

    // Verify omega = 0.0 produces exact 0.0 difference vs plain unary_matrix
    let cfg0 = AfAssignConfig {
        af_weight: 0.0,
        ..AfAssignConfig::default()
    };
    let af0 = AfAssignPriors::from_structure_model(
        &model,
        &ref_data.sequence,
        Some(&ref_data.seq_ids),
        None,
        Vec::new(),
        &fw,
        &cfg0,
    )
    .unwrap();
    let (peaks, _) = simulate(&ref_data, &SimConfig::realistic_bruker());
    let systems = build_spin_systems(&peaks, &Tolerances::default(), 3, false).unwrap();
    let priors = Priors::default();
    let score_cfg = ScoreConfig::default();
    let u_plain = unary_matrix(&systems, &ref_data.sequence, &priors, &score_cfg);
    let u_af0 = unary_matrix_with_af_priors(
        &systems,
        &ref_data.sequence,
        &priors,
        &score_cfg,
        Some(&af0),
    );
    let max_diff_0 = u_plain
        .data
        .iter()
        .zip(u_af0.data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_diff_0 < 1e-12,
        "af_weight=0.0 must equal plain unary_matrix (got max_diff={max_diff_0})"
    );

    // Verify replacement algebra: with af_weight = 1.0 and 2 ambiguous CA candidates,
    // ln(n_pos) is subtracted ONCE (matching plain unary_matrix), not twice.
    let cfg1 = AfAssignConfig {
        af_weight: 1.0,
        af_clamp_nats: 20.0,
        ..AfAssignConfig::default()
    };
    let af1 = AfAssignPriors::from_structure_model(
        &model,
        &ref_data.sequence,
        Some(&ref_data.seq_ids),
        None,
        Vec::new(),
        &fw,
        &cfg1,
    )
    .unwrap();
    let u_af1 = unary_matrix_with_af_priors(
        &systems,
        &ref_data.sequence,
        &priors,
        &score_cfg,
        Some(&af1),
    );
    // Find a spin system with >= 2 positive CA candidates and check finite, non-double-penalized score
    let multi_ca_s = systems
        .iter()
        .position(|ss| {
            ss.cands
                .get(&("CA".to_string(), 0))
                .map(|c| c.iter().filter(|x| x.1 > 0).count() >= 2)
                .unwrap_or(false)
        })
        .expect("at least one spin system with >=2 CA candidates");
    for k in 1..ref_data.sequence.len() {
        let up = u_plain.get(multi_ca_s, k);
        let ua = u_af1.get(multi_ca_s, k);
        if up > FORBID * 0.5 {
            assert!(ua.is_finite() && ua > FORBID * 0.5);
        }
    }
}

#[test]
fn test_p1_2_honest_uncalibrated_af_flag_and_warning() {
    let root = workspace_root();
    let ref_data = read_star(root.join("data/benchmark/bmr17769.str"), None).unwrap();
    let priors = Priors::default();
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
    .unwrap();
    let (peaks, _) = simulate(&ref_data, &SimConfig::realistic_bruker());

    let evo_plain = EvoGraphConfig {
        population_size: nmr_core::qc::EVO_GRAPH_CALIBRATED_RUNS,
        generations: 2,
        steps_per_system_per_gen: 50,
        seed: 0,
        apply_qc_calibration: true,
        af_priors: None,
        ..EvoGraphConfig::default()
    };
    let (_, _, rep_plain) =
        assign_evo_graph_from_peaks(&peaks, &ref_data.sequence, &priors, &evo_plain);
    assert!(rep_plain.qc_report.confidence_calibrated);
    assert!(rep_plain.qc_report.calibration_warning.is_none());

    let evo_af = EvoGraphConfig {
        af_priors: Some(&af_priors),
        ..evo_plain
    };
    let (_, _, rep_af) = assign_evo_graph_from_peaks(&peaks, &ref_data.sequence, &priors, &evo_af);
    assert!(
        !rep_af.qc_report.confidence_calibrated,
        "confidence_calibrated must be false when af_priors is active"
    );
    assert!(
        rep_af
            .qc_report
            .calibration_warning
            .as_deref()
            .unwrap_or("")
            .contains("uncalibrated_af_prior"),
        "calibration_warning must mention uncalibrated_af_prior, got {:?}",
        rep_af.qc_report.calibration_warning
    );
}

#[test]
fn test_p1_3_fail_closed_model_provenance_rejects_experimental_and_headerless() {
    let root = workspace_root();
    // 1D3Z (SOLUTION NMR) and 1UBQ (X-RAY DIFFRACTION) must fail closed unless treat_as_model=true
    let exp_1d3z = StructureModel::parse_file(root.join("data/pdb_refs/1d3z.pdb"), false).unwrap();
    assert!(af_require_predicted_model(&exp_1d3z, false).is_err());
    assert!(af_require_predicted_model(&exp_1d3z, true).is_ok());

    let exp_1ubq =
        StructureModel::parse_file(root.join("data/fixtures/ubiquitin_1ubq_xray.pdb"), false)
            .unwrap();
    assert!(af_require_predicted_model(&exp_1ubq, false).is_err());
    assert!(af_require_predicted_model(&exp_1ubq, true).is_ok());

    // Headerless PDB, REMARK ALPHAFOLD decoy, QMEANDisCo decoy, constant B=0.5 must all fail evidence
    assert!(!text_has_predicted_model_evidence(
        "ATOM      1  N   MET A   1      10.000  10.000  10.000  1.00 90.00           N\n"
    ));
    assert!(!text_has_predicted_model_evidence(
        "REMARK   1 ALPHAFOLD MENTION IN EXPERIMENTAL DECOY\nEXPDTA    SOLUTION NMR\n"
    ));
    assert!(!text_has_predicted_model_evidence(
        "REMARK 250 QMEANDisCo global score 0.81\nATOM      1  CA  ALA A   1       0.0   0.0   0.0  1.00 0.50\n"
    ));
    // Positive mmCIF _ma_qa_metric_local pLDDT on real AFDB v6 file passes
    let (af_cif_path, _) = real_af_ubiquitin_paths(&root);
    let real_af_cif_txt = fs::read_to_string(&af_cif_path).expect("read AF-P0CG48-F1 v6 cif");
    assert!(text_has_predicted_model_evidence(&real_af_cif_txt));

    // `21F:B1` regression: real-shaped ModelArchive QMEANDisCo mmCIF with `_ma_protocol_step`,
    // `_ma_data`, `_ma_model_list`, AND non-constant `_atom_site.B_iso_or_equiv` in `[0, 1]`
    // must be rejected by BOTH `text_has_predicted_model_evidence` AND `AfAssignPriors::from_structure_strs`!
    // Fails at `f8408b9`.
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
_atom_site.group_PDB
_atom_site.id
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.auth_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.B_iso_or_equiv
ATOM 1 N ALA A 1 1 0.0 0.0 0.0 0.924
ATOM 2 CA ALA A 1 1 1.46 0.0 0.0 0.924
ATOM 3 C ALA A 1 1 2.0 1.4 0.0 0.924
ATOM 4 N GLY A 2 2 3.2 1.2 0.0 0.781
ATOM 5 CA GLY A 2 2 4.5 1.8 0.0 0.781
ATOM 6 C GLY A 2 2 5.2 0.6 0.0 0.781
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
    assert!(
        !text_has_predicted_model_evidence(qmean_real_shaped_cif),
        "text_has_predicted_model_evidence must reject real-shaped ModelArchive QMEANDisCo mmCIF"
    );
    let seq2 = vec!["ALA".to_string(), "GLY".to_string()];
    let fw_empty = HomologFirewallQuery::default();
    let res_qm = AfAssignPriors::from_strs(
        qmean_real_shaped_cif,
        Some("cif"),
        &seq2,
        Some(&[1, 2]),
        None,
        None,
        &fw_empty,
        &AfAssignConfig::default(),
    );
    assert!(
        res_qm.is_err(),
        "AfAssignPriors::from_structure_strs must fail closed on ModelArchive QMEANDisCo mmCIF"
    );
}

#[test]
fn test_p2_1_offset_registration_hardened_against_20pct_missing_and_12_nh2_extras() {
    let root = workspace_root();
    let ref_data = read_star(root.join("data/benchmark/bmr17769.str"), None).unwrap();
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
    .unwrap();

    let (peaks, _) = simulate(&ref_data, &SimConfig::realistic_bruker());
    let base_systems = build_spin_systems(&peaks, &Tolerances::default(), 3, false).unwrap();
    let off_base = estimate_af_shift_offsets(&base_systems, &af_priors);

    // Apply additional referencing shifts (+0.12 H, -1.35 N, +0.95 CA, -0.85 CB, +1.25 C'),
    // drop 20% of spin systems, and inject 12 NH2-like sidechain extras!
    let mut perturbed = shift_spin_systems(&base_systems, 0.12, -1.35, 0.95, -0.85, 1.25);
    let keep_len = (perturbed.len() * 4) / 5;
    perturbed.truncate(keep_len);
    for e in 0..12 {
        perturbed.push(SpinSystem::new(
            7.05 + 0.05 * (e as f64),
            111.5 + 0.4 * (e as f64),
        ));
    }
    let off = estimate_af_shift_offsets(&perturbed, &af_priors);
    let d_h = off.h - off_base.h;
    let d_n = off.n - off_base.n;
    let d_ca = off.ca - off_base.ca;
    let d_cb = off.cb - off_base.cb;
    let d_c = off.c - off_base.c;
    assert!(
        (d_h - 0.12).abs() < 0.08
            && (d_n - (-1.35)).abs() < 0.30
            && (d_ca - 0.95).abs() < 0.25
            && (d_cb - (-0.85)).abs() < 0.30
            && (d_c - 1.25).abs() < 0.30,
        "hardened offset registration failed under 20% missing + 12 NH2 extras: delta=({d_h:.3}, {d_n:.3}, {d_ca:.3}, {d_cb:.3}, {d_c:.3})"
    );
}

#[test]
fn test_p2_2_oos_sigma_floor_and_monotonic_widening_with_plddt() {
    let root = workspace_root();
    let ref_data = read_star(root.join("data/benchmark/bmr17769.str"), None).unwrap();
    let fw = ubiquitin_firewall();

    // Vary pLDDT on the SAME residue (k = 20, non-Pro/non-Gly) across 95 -> 70 -> 45
    let build_at_plddt = |plddt: f64| -> AfAssignPriors {
        let mut model = load_ubiquitin_af_like_model(&root);
        model.residues[19].plddt = Some(plddt);
        model.residues[20].plddt = Some(plddt);
        AfAssignPriors::from_structure_model(
            &model,
            &ref_data.sequence,
            Some(&ref_data.seq_ids),
            None,
            Vec::new(),
            &fw,
            &AfAssignConfig::default(),
        )
        .unwrap()
    };
    let af95 = build_at_plddt(95.0);
    let af70 = build_at_plddt(70.0);
    let af45 = build_at_plddt(45.0);

    assert_eq!(AF_SIGMA_OOS_PPM, AF_SIGMA_DEV_SET_PPM);
    for (a, &floor_ppm) in AF_SIGMA_DEV_SET_PPM.iter().enumerate() {
        let p95 = af95.atom_prior(20, a);
        let p70 = af70.atom_prior(20, a);
        let p45 = af45.atom_prior(20, a);
        assert!(p95.present && p70.present && p45.present);
        assert!(
            p95.sigma_af_ppm >= floor_ppm - 1e-12,
            "atom {a} sigma at pLDDT=95 ({}) below [dev-set] floor {}",
            p95.sigma_af_ppm,
            floor_ppm
        );
        assert!(
            p70.sigma_af_ppm >= p95.sigma_af_ppm - 1e-12,
            "atom {a} sigma must not shrink from pLDDT 95->70 ({} < {})",
            p70.sigma_af_ppm,
            p95.sigma_af_ppm
        );
        assert!(
            p45.sigma_af_ppm >= p70.sigma_af_ppm - 1e-12,
            "atom {a} sigma must not shrink from pLDDT 70->45 ({} < {})",
            p45.sigma_af_ppm,
            p70.sigma_af_ppm
        );
    }

    // `21F:B3` regression: T21.2 per-residue `sigma_hat = pred_b.predicted_sigma_ppm` MUST be
    // floored by `AF_SIGMA_DEV_SET_PPM` (not silently discarded!). At `f8408b9`, `sigma_hat` was
    // ignored so high-pLDDT residues always got `sigma_af == AF_SIGMA_DEV_SET_PPM[a]` even when
    // T21.2 predicted a larger `sigma_hat`.
    let atom_names = ["H", "N", "CA", "CB", "C"];
    let mut n_exceeding_floor = 0usize;
    for k in 0..af95.sequence_len() {
        let Some(si) = af95.seq_to_struct[k] else {
            continue;
        };
        let pred_r = &af95.shift_report.residues[si];
        for (a, &aname) in atom_names.iter().enumerate() {
            let pr = af95.atom_prior(k, a);
            if !pr.present {
                continue;
            }
            let pred_b = &pred_r.backbone[aname];
            assert!(
                pr.sigma_af_ppm >= pred_b.predicted_sigma_ppm - 1e-12,
                "residue {k} atom {aname}: sigma_af ({:.4}) < T21.2 sigma_hat ({:.4})",
                pr.sigma_af_ppm,
                pred_b.predicted_sigma_ppm
            );
            if pred_b.predicted_sigma_ppm > AF_SIGMA_DEV_SET_PPM[a] + 1e-4
                && af95.plddt_per_residue[k] >= 90.0
            {
                assert!(
                    pr.sigma_af_ppm > AF_SIGMA_DEV_SET_PPM[a] + 1e-4,
                    "residue {k} atom {aname}: sigma_hat ({:.4}) > floor ({:.4}) was discarded (sigma_af={:.4})",
                    pred_b.predicted_sigma_ppm,
                    AF_SIGMA_DEV_SET_PPM[a],
                    pr.sigma_af_ppm
                );
                n_exceeding_floor += 1;
            }
        }
    }
    assert!(
        n_exceeding_floor > 0,
        "expected at least one high-pLDDT residue/atom with T21.2 sigma_hat > AF_SIGMA_DEV_SET_PPM"
    );
}

#[test]
fn test_p2_3_noesy_site_bonus_symmetrically_capped_by_noe_weight() {
    let root = workspace_root();
    let ref_data = read_star(root.join("data/benchmark/bmr17769.str"), None).unwrap();
    let nmr_conformer = load_1d3z_conformer(&root);
    let noesy_peaks = simulate_amide_15n_noesy_peaks(&ref_data, &nmr_conformer, 7, 0.0, 0.0);
    let (peaks, _) = simulate(&ref_data, &SimConfig::realistic_bruker());
    let systems = build_spin_systems(&peaks, &Tolerances::default(), 3, false).unwrap();

    let cfg = AfAssignConfig {
        noe_weight: 0.50,
        ..AfAssignConfig::default()
    };
    let af_seq_noesy =
        AfAssignPriors::sequence_only_noesy(&ref_data.sequence, noesy_peaks, &cfg).unwrap();
    let table = af_seq_noesy
        .build_noesy_pairwise_table(&systems)
        .expect("non-empty NOESY pairwise table");
    let cap = cfg.noe_weight * AF_NOESY_MAX_SITE_BONUS_NATS;
    assert!((table.site_cap_nats() - cap).abs() < 1e-12);

    // Build a dense synthetic assignment covering all residues and verify site bonus cap
    let at_vec: Vec<i32> = (0..systems.len())
        .map(|s| {
            if s < ref_data.sequence.len() {
                s as i32
            } else {
                0
            }
        })
        .collect();
    let total_bonus = table.total_bonus(&at_vec);
    assert!(total_bonus.is_finite());
    assert!(
        total_bonus <= (systems.len() as f64) * cap + 1e-9,
        "total NOESY bonus {total_bonus} exceeded per-site cap sum"
    );
}
