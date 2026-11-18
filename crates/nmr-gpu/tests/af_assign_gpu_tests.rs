//! WebGPU vs CPU `f64` `U`-matrix parity, Dawn/Tint const-eval portability, `max_buffer_size`
//! pre-check, and multi-replica GPU annealing tests for opt-in `T21.4` Student-`t(nu = 4)` 3D priors
//! (`docs/specs/21-alphafold-integrative-nmr.md` sections 4.2, 4.4, and 8).

use std::path::PathBuf;

use nmr_core::af_assign_priors::{AfAssignConfig, AfAssignPriors};
use nmr_core::evaluate::score;
use nmr_core::nmrstar::read_star;
use nmr_core::priors::Priors;
use nmr_core::simulate::{simulate, SimConfig};
use nmr_core::solver::{
    pair_matrix, unary_matrix, unary_matrix_with_af_priors, AnnealConfig, ScoreConfig, FORBID,
};
use nmr_core::spinsystems::{build_spin_systems, SpinSystem, Tolerances};
use nmr_core::struct_shifts::StructureModel;
use nmr_core::torsion_db::HomologFirewallQuery;
use nmr_gpu::GpuEngine;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
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

#[test]
fn test_scoring_wgsl_dawn_tint_portability_and_zero_bitcast() {
    let wgsl_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/shaders/scoring.wgsl");
    let wgsl_src = std::fs::read_to_string(&wgsl_path).expect("read scoring.wgsl");

    assert!(
        !wgsl_src.contains("bitcast"),
        "scoring.wgsl must contain zero bitcasts for Dawn/Tint WebGPU const-eval portability"
    );
    let lower = wgsl_src.to_ascii_lowercase();
    assert!(
        !lower.contains("0x7fc00000")
            && !lower.contains("0x7f800000")
            && !lower.contains("0xff800000"),
        "scoring.wgsl must contain zero raw IEEE-754 NaN/Inf bit patterns"
    );
    assert!(
        wgsl_src.contains("fn compute_unary_af(")
            && wgsl_src.contains("raw_delta_a + raw_delta_b")
            && wgsl_src.contains("-af_params.af_clamp_total"),
        "scoring.wgsl must isolate AF scoring in compute_unary_af and clamp raw delta across observed nuclei"
    );
    let seq_only_prefix = wgsl_src
        .split("// ====")
        .next()
        .expect("seq-only prefix before AF section");
    assert!(
        seq_only_prefix.contains("fn compute_unary(")
            && seq_only_prefix.contains("fn compute_pair(")
            && !seq_only_prefix.contains("af_params")
            && !seq_only_prefix.contains("ln_accurate"),
        "sequence-only compute_unary and compute_pair must remain untouched 5c4e87d code"
    );

    // Validate shader compilation under strict default WebGPU limits (`max_storage_buffers_per_shader_stage == 8`)
    let default_limits = wgpu::Limits::default();
    assert_eq!(default_limits.max_storage_buffers_per_shader_stage, 8);
    let gpu = GpuEngine::new_headless().expect("headless WebGPU adapter");
    assert!(!gpu.adapter_name.is_empty());
}

#[test]
fn test_gpu_af_buffer_limits_precheck() {
    let gpu = GpuEngine::new_headless().expect("headless WebGPU adapter");
    gpu.validate_af_scoring_gpu_limits(76, 76, 8, true)
        .expect("normal Ubiquitin dimensions fit comfortably within WebGPU limits");

    // Oversized S x S pair matrix exceeding device max_buffer_size / max_storage_buffer_binding_size
    let err_huge = gpu.validate_af_scoring_gpu_limits(32_000, 32_000, 8, true);
    assert!(
        err_huge.is_err(),
        "oversized matrix exceeding max_storage_buffer_binding_size / max_buffer_size must fail pre-check before allocation"
    );

    // Arithmetic overflow pre-check
    let err_overflow = gpu.validate_af_scoring_gpu_limits(usize::MAX, 76, 8, true);
    assert!(err_overflow.is_err());
}

#[test]
fn test_gpu_af_priors_none_is_bit_identical_zero_diff() {
    let root = workspace_root();
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

    let gpu = GpuEngine::new_headless().expect("headless WebGPU adapter");
    let (u_base, p_base) = gpu
        .score_matrices_gpu(&systems, &ref_data.sequence, &priors, &score_cfg)
        .unwrap();
    let (u_opt_none, p_opt_none) = gpu
        .score_matrices_with_af_priors_gpu(&systems, &ref_data.sequence, &priors, &score_cfg, None)
        .unwrap();

    let mut max_u_diff = 0.0_f64;
    for i in 0..u_base.data.len() {
        max_u_diff = max_u_diff.max((u_base.data[i] - u_opt_none.data[i]).abs());
    }
    let mut max_p_diff = 0.0_f64;
    for i in 0..p_base.data.len() {
        max_p_diff = max_p_diff.max((p_base.data[i] - p_opt_none.data[i]).abs());
    }
    assert_eq!(
        max_u_diff, 0.0,
        "score_matrices_with_af_priors_gpu(..., None) must produce exact 0.0 diff on U"
    );
    assert_eq!(
        max_p_diff, 0.0,
        "score_matrices_with_af_priors_gpu(..., None) must produce exact 0.0 diff on P"
    );
}

fn read_f64_le_bin(path: &std::path::Path) -> Vec<f64> {
    let bytes = std::fs::read(path).expect("read golden f64le binary");
    assert_eq!(bytes.len() % 8, 0);
    bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn read_f32_le_bin(path: &std::path::Path) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("read golden f32le binary");
    assert_eq!(bytes.len() % 4, 0);
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn test_gpu_sequence_only_matches_5c4e87d_golden_snapshots_exact_zero_diff() {
    let root = workspace_root();
    let fix_dir = root.join("crates/nmr-gpu/tests/fixtures");
    let manifest = std::fs::read_to_string(fix_dir.join("MANIFEST.tsv"))
        .expect("read golden 5c4e87d MANIFEST.tsv");
    let mut expected_dims: std::collections::BTreeMap<(String, String), (usize, usize, String)> =
        std::collections::BTreeMap::new();
    for line in manifest.lines().skip(1) {
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() >= 6 {
            expected_dims.insert(
                (cols[0].to_string(), cols[1].to_string()),
                (
                    cols[2].parse().expect("rows"),
                    cols[3].parse().expect("cols"),
                    cols[5].to_string(),
                ),
            );
        }
    }

    let gpu = GpuEngine::new_headless().expect("headless WebGPU adapter");
    let priors = Priors::default();
    let score_cfg = ScoreConfig::default();

    let cases = [
        (
            "synth_ubq_realistic_s0",
            "data/fixtures/synthetic_ubiquitin.str",
            "realistic_bruker",
            0u64,
        ),
        (
            "bmr17769_realistic_s1",
            "data/benchmark/bmr17769.str",
            "realistic_bruker",
            1u64,
        ),
        (
            "synth_ubq_default_s3",
            "data/fixtures/synthetic_ubiquitin.str",
            "default",
            3u64,
        ),
    ];

    for &(case, star_rel, preset, seed) in &cases {
        let ref_data = read_star(root.join(star_rel), None).unwrap();
        let mut sim_cfg = match preset {
            "realistic_bruker" => SimConfig::realistic_bruker(),
            "default" => SimConfig::default(),
            other => panic!("unknown preset {other}"),
        };
        sim_cfg.seed = seed;
        let (peaks, _truth) = simulate(&ref_data, &sim_cfg);
        let systems = build_spin_systems(&peaks, &Tolerances::default(), 3, false).unwrap();

        let cpu_u = unary_matrix(&systems, &ref_data.sequence, &priors, &score_cfg);
        let cpu_p = pair_matrix(&systems, &score_cfg);
        let (gpu_u, gpu_p) = gpu
            .score_matrices_gpu(&systems, &ref_data.sequence, &priors, &score_cfg)
            .unwrap();

        let (r_cu, c_cu, f_cu) = &expected_dims[&(case.to_string(), "cpu_u".to_string())];
        let (r_cp, c_cp, f_cp) = &expected_dims[&(case.to_string(), "cpu_p".to_string())];
        let (r_gu, c_gu, f_gu) = &expected_dims[&(case.to_string(), "gpu_u".to_string())];
        let (r_gp, c_gp, f_gp) = &expected_dims[&(case.to_string(), "gpu_p".to_string())];

        assert_eq!((cpu_u.rows, cpu_u.cols), (*r_cu, *c_cu));
        assert_eq!((cpu_p.rows, cpu_p.cols), (*r_cp, *c_cp));
        assert_eq!((gpu_u.rows, gpu_u.cols), (*r_gu, *c_gu));
        assert_eq!((gpu_p.rows, gpu_p.cols), (*r_gp, *c_gp));

        let ref_cpu_u = read_f64_le_bin(&fix_dir.join(f_cu));
        let ref_cpu_p = read_f64_le_bin(&fix_dir.join(f_cp));
        let ref_gpu_u = read_f32_le_bin(&fix_dir.join(f_gu));
        let ref_gpu_p = read_f32_le_bin(&fix_dir.join(f_gp));

        let max_cpu_u_diff = cpu_u
            .data
            .iter()
            .zip(ref_cpu_u.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let max_cpu_p_diff = cpu_p
            .data
            .iter()
            .zip(ref_cpu_p.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let max_gpu_u_diff = gpu_u
            .data
            .iter()
            .zip(ref_gpu_u.iter())
            .map(|(a, b)| ((*a as f32) - *b).abs())
            .fold(0.0_f32, f32::max);
        let max_gpu_p_diff = gpu_p
            .data
            .iter()
            .zip(ref_gpu_p.iter())
            .map(|(a, b)| ((*a as f32) - *b).abs())
            .fold(0.0_f32, f32::max);

        assert_eq!(max_cpu_u_diff, 0.0, "{case} CPU U drifted from 5c4e87d");
        assert_eq!(max_cpu_p_diff, 0.0, "{case} CPU P drifted from 5c4e87d");
        assert_eq!(max_gpu_u_diff, 0.0, "{case} GPU U drifted from 5c4e87d");
        assert_eq!(max_gpu_p_diff, 0.0, "{case} GPU P drifted from 5c4e87d");
    }
}

#[test]
fn test_gpu_u_matrix_parity_with_student_t4_af_priors_and_clamping() {
    let root = workspace_root();
    // Use real experimental Human Ubiquitin BMRB 17769 shifts paired with 1D3Z structure
    let ref_data = read_star(root.join("data/benchmark/bmr17769.str"), None).unwrap();
    let (peaks, truth) = simulate(
        &ref_data,
        &SimConfig {
            seed: 1,
            ..SimConfig::realistic_bruker()
        },
    );
    let base_systems = build_spin_systems(&peaks, &Tolerances::default(), 3, false).unwrap();
    let priors = Priors::default();

    // Load real downloaded AlphaFoldDB v6 Polyubiquitin-C structure (AF-P0CG48-F1-model_v6.cif)
    // with native _ma_qa_metric_local pLDDT and enforce strict 4-layer HomologFirewall.
    let af_cif = root.join("data/afdb/AF-P0CG48-F1-model_v6.cif");
    let model = StructureModel::parse_file(&af_cif, false).unwrap();

    let fw = ubiquitin_firewall();
    let mut af_cfg = AfAssignConfig::default();
    let af_priors = AfAssignPriors::from_structure_model(
        &model,
        &ref_data.sequence,
        Some(&ref_data.seq_ids),
        None,
        Vec::new(),
        &fw,
        &af_cfg,
    )
    .expect("build firewalled AfAssignPriors");
    assert!(af_priors.shift_report.max_retained_seq_identity <= 0.25 + 1e-12);

    let gpu = GpuEngine::new_headless().expect("headless WebGPU adapter");

    // Case A: Unshifted realistic Bruker spin systems
    let score_cfg = ScoreConfig::default();
    let u_cpu = unary_matrix_with_af_priors(
        &base_systems,
        &ref_data.sequence,
        &priors,
        &score_cfg,
        Some(&af_priors),
    );
    let u_cpu_plain = unary_matrix(&base_systems, &ref_data.sequence, &priors, &score_cfg);
    let p_cpu = pair_matrix(&base_systems, &score_cfg);
    let (u_gpu, p_gpu) = gpu
        .score_matrices_with_af_priors_gpu(
            &base_systems,
            &ref_data.sequence,
            &priors,
            &score_cfg,
            Some(&af_priors),
        )
        .expect("GPU scoring with AF priors");

    let mut max_u_diff = 0.0_f64;
    let mut max_af_delta = 0.0_f64;
    let mut admissible_cells = 0usize;
    for s in 0..u_cpu.rows {
        for k in 0..u_cpu.cols {
            let c = u_cpu.get(s, k);
            let g = u_gpu.get(s, k);
            if c <= FORBID * 0.5 {
                assert!(
                    g <= FORBID * 0.5,
                    "GPU cell ({s}, {k}) must be FORBID when CPU is FORBID (got {g})"
                );
            } else {
                admissible_cells += 1;
                max_u_diff = max_u_diff.max((c - g).abs());
                max_af_delta = max_af_delta.max((c - u_cpu_plain.get(s, k)).abs());
            }
        }
    }
    let mut max_p_diff = 0.0_f64;
    for s1 in 0..p_cpu.rows {
        for s2 in 0..p_cpu.cols {
            max_p_diff = max_p_diff.max((p_cpu.get(s1, s2) - p_gpu.get(s1, s2)).abs());
        }
    }
    println!(
        "[Case A] admissible_cells={admissible_cells}, max_af_delta={max_af_delta:.4} nats, max |U_cpu - U_gpu| = {max_u_diff:.3e}, max |P_cpu - P_gpu| = {max_p_diff:.3e}"
    );
    assert!(admissible_cells > 1000);
    assert!(
        max_af_delta > 0.25 && max_af_delta < 25.0,
        "AF replacement delta U must be active and finite (got {max_af_delta})"
    );
    assert!(
        max_u_diff < 1.0e-5,
        "Case A: max |U_cpu - U_gpu| = {max_u_diff:.3e} exceeded 1.0e-5 threshold"
    );
    assert!(
        max_p_diff < 1.0e-3,
        "Case A: max |P_cpu - P_gpu| = {max_p_diff:.3e} exceeded 1.0e-3 threshold"
    );

    // Case B: Shifted spectrometer carrier offsets (+0.18 H, -1.65 N, +1.10 CA, -0.95 CB, +1.75 C')
    // plus a locked assignment and corrupted 12-residue loop hitting the +-4.0 nats clamp!
    let shifted_systems = shift_spin_systems(&base_systems, 0.18, -1.65, 1.10, -0.95, 1.75);
    let mut corrupted_model = model.clone();
    for r in &mut corrupted_model.residues[22..34] {
        r.phi_deg = r.phi_deg.map(|p| {
            let v = p + 120.0;
            if v > 180.0 {
                v - 360.0
            } else {
                v
            }
        });
        r.psi_deg = r.psi_deg.map(|p| {
            let v = p + 120.0;
            if v > 180.0 {
                v - 360.0
            } else {
                v
            }
        });
        r.plddt = Some(95.0);
    }
    af_cfg.af_weight = 0.85;
    af_cfg.af_clamp_nats = 4.0;
    let af_corrupted = AfAssignPriors::from_structure_model(
        &corrupted_model,
        &ref_data.sequence,
        Some(&ref_data.seq_ids),
        None,
        Vec::new(),
        &fw,
        &af_cfg,
    )
    .expect("build corrupted-loop AfAssignPriors");

    let lock_pair = shifted_systems
        .iter()
        .enumerate()
        .find_map(|(s, _)| {
            let t = truth.get(s).copied().unwrap_or(-1);
            if t >= 2 {
                Some((s, t as usize))
            } else {
                None
            }
        })
        .expect("at least one true spin system");
    let score_cfg_locked = ScoreConfig {
        locked_assignments: vec![lock_pair],
        ..ScoreConfig::default()
    };

    let u_cpu_b = unary_matrix_with_af_priors(
        &shifted_systems,
        &ref_data.sequence,
        &priors,
        &score_cfg_locked,
        Some(&af_corrupted),
    );
    let (u_gpu_b, _) = gpu
        .score_matrices_with_af_priors_gpu(
            &shifted_systems,
            &ref_data.sequence,
            &priors,
            &score_cfg_locked,
            Some(&af_corrupted),
        )
        .expect("GPU scoring Case B");

    let u_cpu_b_plain = unary_matrix(
        &shifted_systems,
        &ref_data.sequence,
        &priors,
        &score_cfg_locked,
    );
    let (u_gpu_b_plain, _) = gpu
        .score_matrices_with_af_priors_gpu(
            &shifted_systems,
            &ref_data.sequence,
            &priors,
            &score_cfg_locked,
            None,
        )
        .expect("GPU scoring Case B plain");

    let mut max_u_diff_b = 0.0_f64;
    let mut worst_info = String::new();
    for s in 0..u_cpu_b.rows {
        for k in 0..u_cpu_b.cols {
            let c = u_cpu_b.get(s, k);
            let g = u_gpu_b.get(s, k);
            if c <= FORBID * 0.5 {
                assert!(g <= FORBID * 0.5);
            } else {
                let d = (c - g).abs();
                if d > max_u_diff_b {
                    max_u_diff_b = d;
                    let dc_plain = u_cpu_b_plain.get(s, k) - u_gpu_b_plain.get(s, k);
                    let af_cpu = c - u_cpu_b_plain.get(s, k);
                    let af_gpu = g - u_gpu_b_plain.get(s, k);
                    worst_info = format!(
                        "s={s}, k={k} ({}): c={c:.7}, g={g:.7}, plain_diff={dc_plain:.3e}, af_cpu={af_cpu:.7}, af_gpu={af_gpu:.7}, af_diff={:.3e}",
                        ref_data.sequence[k],
                        (af_cpu - af_gpu).abs()
                    );
                }
            }
        }
    }
    println!("[Case B shifted + corrupted + locked] max |U_cpu - U_gpu| = {max_u_diff_b:.3e} ({worst_info})");
    assert!(
        max_u_diff_b < 1.0e-5,
        "Case B: max |U_cpu - U_gpu| = {max_u_diff_b:.3e} exceeded 1.0e-5 threshold"
    );

    // End-to-end WebGPU multi-replica simulated annealing with AF priors enabled
    let anneal_cfg = AnnealConfig {
        runs: 256,
        steps_per_system: 1500,
        seed: 7,
        ..AnnealConfig::default()
    };
    let res_gpu_base = gpu
        .assign_gpu(
            &base_systems,
            &ref_data.sequence,
            &priors,
            &score_cfg,
            &anneal_cfg,
        )
        .expect("assign_gpu baseline");
    let res_gpu = gpu
        .assign_with_af_priors_gpu(
            &base_systems,
            &ref_data.sequence,
            &priors,
            &score_cfg,
            &anneal_cfg,
            Some(&af_priors),
        )
        .expect("assign_with_af_priors_gpu");
    let m_base = score(&res_gpu_base, &truth, 0.8);
    let m = score(&res_gpu, &truth, 0.8);
    println!(
        "[GPU SA realistic_bruker] base: acc={:.2}% ({}/{}), hc_prec={:.2}% | +AF priors: acc={:.2}% ({}/{}), hc_prec={:.2}%",
        m_base.accuracy() * 100.0,
        m_base.correct,
        m_base.assignable,
        m_base.hc_precision() * 100.0,
        m.accuracy() * 100.0,
        m.correct,
        m.assignable,
        m.hc_precision() * 100.0,
    );
    assert!(
        m.correct >= m_base.correct && m.hc_precision() >= m_base.hc_precision() - 1e-6,
        "WebGPU SA with AF priors must match/exceed baseline accuracy and HC precision on experimental BMRB 17769 Ubiquitin"
    );
}
