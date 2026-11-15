//! WebGPU (`wgpu` + WGSL) batched multi-conformer RDC torsion energy & analytical gradient parity
//! (`< 1e-5` vs `f64` CPU reference), baseline browser `wgpu::Limits::default()` portability, and
//! cross-validated experimental `1D3Z` Ubiquitin GPU ensemble refinement tests (`Phase 12`, `T12.4`).
//!
//! Provenance: Clean-room verification suite citing:
//! - Tjandra, N. & Bax, A. (*Science* 1997, 278, 1111–1114).
//! - Cornilescu, G., Marquardt, J. L., Ottiger, M. & Bax, A. (*J. Am. Chem. Soc.* 1998, 120, 6836–6837).
//! - Losonczi, J. A., Andrec, M., Fischer, M. W. F. & Prestegard, J. H. (*J. Magn. Reson.* 1999, 138, 334–342).
//! - Parsons, J. et al. (*J. Comput. Chem.* 2005, 26, 1063–1068 — `NeRF`).
//! - Abe, H., Braun, W., Noguti, T. & Go, N. (*Comput. Chem.* 1984, 8, 239–247).

use std::f64::consts::PI;
use std::path::PathBuf;

use nmr_core::noesy::{
    compute_chain_torsions, parse_pdb_first_model, v_norm, v_scale, v_sub, PdbChain,
};
use nmr_core::rdc::{
    calc_rdc_from_unit_vec, evaluate_rdc_validation, partition_rdc_free_set,
    partition_rdc_free_set_with_mode, read_rdc_tsv, saupe_matrix_from_pas, RdcBondType,
    RdcFreeStratification, RdcMediumSet, RdcRestraint,
};
use nmr_core::rng::Rng;
use nmr_core::structure::{
    build_backbone_from_torsions, evaluate_rdc_torsion_energy_and_gradients_fixed_saupe,
    refine_ensemble_with_rdcs, superpose_chain_onto, StructureConformer,
};
use nmr_gpu::rdc_gpu::{
    evaluate_rdc_batch_gpu, refine_ensemble_with_rdcs_gpu, validate_rdc_gpu_dispatch_limits,
    RdcGpuConformerInput, MAX_RDC_CONFORMERS_PER_DISPATCH,
};
use nmr_gpu::GpuEngine;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn require_headless_gpu(custom_limits: Option<wgpu::Limits>) -> Option<GpuEngine> {
    let res = match custom_limits {
        Some(lim) => GpuEngine::new_headless_with_limits(lim),
        None => GpuEngine::new_headless(),
    };
    match res {
        Ok(engine) => Some(engine),
        Err(err) => {
            if std::env::var("NMR_ALLOW_NO_GPU").as_deref() == Ok("1") {
                println!(
                    "NMR_ALLOW_NO_GPU=1 set; opting out of headless WebGPU test cleanly: {err}"
                );
                None
            } else {
                panic!(
                    "Headless WebGPU initialization failed (set NMR_ALLOW_NO_GPU=1 only when running without any GPU/software Vulkan driver): {err}"
                );
            }
        }
    }
}

fn build_all5_medium(chain: &PdbChain, medium_id: &str, s_mat: &[[f64; 3]; 3]) -> RdcMediumSet {
    let mut restraints = Vec::new();
    let l = chain.residues.len();
    for i in 0..l {
        let r = &chain.residues[i];
        if let Some(h) = r.h {
            let u = v_scale(v_sub(h, r.n), 1.0 / v_norm(v_sub(h, r.n)));
            let dmax = RdcBondType::NH.default_d_max_hz();
            restraints.push(RdcRestraint {
                medium_id: medium_id.to_string(),
                seq_id_1: r.seq_id,
                res_name_1: r.res_name.clone(),
                atom_name_1: "N".to_string(),
                seq_id_2: r.seq_id,
                res_name_2: r.res_name.clone(),
                atom_name_2: "H".to_string(),
                bond_type: RdcBondType::NH,
                d_obs_hz: calc_rdc_from_unit_vec(u, dmax, s_mat),
                sigma_hz: 0.5,
                d_max_hz: dmax,
                is_free: i % 5 == 4,
            });
        }
        if r.res_name.eq_ignore_ascii_case("GLY") {
            let dmax = RdcBondType::CAHA.default_d_max_hz();
            for (aname, pos_opt) in [("HA2", r.ha2), ("HA3", r.ha3)] {
                let ha_pos = pos_opt.expect("GLY must carry explicit HA2 and HA3 coordinates");
                let u = v_scale(v_sub(ha_pos, r.ca), 1.0 / v_norm(v_sub(ha_pos, r.ca)));
                restraints.push(RdcRestraint {
                    medium_id: medium_id.to_string(),
                    seq_id_1: r.seq_id,
                    res_name_1: r.res_name.clone(),
                    atom_name_1: "CA".to_string(),
                    seq_id_2: r.seq_id,
                    res_name_2: r.res_name.clone(),
                    atom_name_2: aname.to_string(),
                    bond_type: RdcBondType::CAHA,
                    d_obs_hz: calc_rdc_from_unit_vec(u, dmax, s_mat),
                    sigma_hz: 1.0,
                    d_max_hz: dmax,
                    is_free: false,
                });
            }
        } else {
            let u = v_scale(v_sub(r.ha, r.ca), 1.0 / v_norm(v_sub(r.ha, r.ca)));
            let dmax = RdcBondType::CAHA.default_d_max_hz();
            restraints.push(RdcRestraint {
                medium_id: medium_id.to_string(),
                seq_id_1: r.seq_id,
                res_name_1: r.res_name.clone(),
                atom_name_1: "CA".to_string(),
                seq_id_2: r.seq_id,
                res_name_2: r.res_name.clone(),
                atom_name_2: "HA".to_string(),
                bond_type: RdcBondType::CAHA,
                d_obs_hz: calc_rdc_from_unit_vec(u, dmax, s_mat),
                sigma_hz: 1.0,
                d_max_hz: dmax,
                is_free: false,
            });
        }
        {
            let u = v_scale(v_sub(r.c, r.ca), 1.0 / v_norm(v_sub(r.c, r.ca)));
            let dmax = RdcBondType::CACO.default_d_max_hz();
            restraints.push(RdcRestraint {
                medium_id: medium_id.to_string(),
                seq_id_1: r.seq_id,
                res_name_1: r.res_name.clone(),
                atom_name_1: "CA".to_string(),
                seq_id_2: r.seq_id,
                res_name_2: r.res_name.clone(),
                atom_name_2: "C".to_string(),
                bond_type: RdcBondType::CACO,
                d_obs_hz: calc_rdc_from_unit_vec(u, dmax, s_mat),
                sigma_hz: 0.25,
                d_max_hz: dmax,
                is_free: false,
            });
        }
        if i > 0 {
            let prev = &chain.residues[i - 1];
            let u_nc = v_scale(v_sub(r.n, prev.c), 1.0 / v_norm(v_sub(r.n, prev.c)));
            let dmax_nc = RdcBondType::NCPrime.default_d_max_hz();
            restraints.push(RdcRestraint {
                medium_id: medium_id.to_string(),
                seq_id_1: prev.seq_id,
                res_name_1: prev.res_name.clone(),
                atom_name_1: "C".to_string(),
                seq_id_2: r.seq_id,
                res_name_2: r.res_name.clone(),
                atom_name_2: "N".to_string(),
                bond_type: RdcBondType::NCPrime,
                d_obs_hz: calc_rdc_from_unit_vec(u_nc, dmax_nc, s_mat),
                sigma_hz: 0.20,
                d_max_hz: dmax_nc,
                is_free: false,
            });
            if let Some(h) = r.h {
                let u_hc = v_scale(v_sub(h, prev.c), 1.0 / v_norm(v_sub(h, prev.c)));
                let dmax_hc = RdcBondType::HNCPrime.default_d_max_hz();
                restraints.push(RdcRestraint {
                    medium_id: medium_id.to_string(),
                    seq_id_1: prev.seq_id,
                    res_name_1: prev.res_name.clone(),
                    atom_name_1: "C".to_string(),
                    seq_id_2: r.seq_id,
                    res_name_2: r.res_name.clone(),
                    atom_name_2: "H".to_string(),
                    bond_type: RdcBondType::HNCPrime,
                    d_obs_hz: calc_rdc_from_unit_vec(u_hc, dmax_hc, s_mat),
                    sigma_hz: 0.35,
                    d_max_hz: dmax_hc,
                    is_free: false,
                });
            }
        }
    }
    RdcMediumSet {
        medium_id: medium_id.to_string(),
        restraints,
    }
}

#[test]
fn test_rdc_gpu_vs_cpu_batch_energy_and_gradient_parity_and_default_limits() {
    let Some(engine) = require_headless_gpu(Some(wgpu::Limits::default())) else {
        return;
    };

    let seq: Vec<String> = [
        "MET", "GLN", "ILE", "PHE", "VAL", "LYS", "THR", "LEU", "THR", "GLY", "LYS", "THR", "ILE",
        "THR", "LEU", "GLU", "VAL", "GLU", "PRO", "SER", "ASP", "THR", "ILE", "GLU",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let l = seq.len();
    let seq_ids: Vec<i32> = (1..=(l as i32)).collect();

    let mut rng = Rng::new(987654);
    let ref_phi: Vec<f64> = (0..l).map(|_| rng.uniform(-140.0, -55.0)).collect();
    let ref_psi: Vec<f64> = (0..l).map(|_| rng.uniform(-50.0, 150.0)).collect();
    let ref_omg = vec![180.0; l];
    let ref_chain = build_backbone_from_torsions(&seq, &seq_ids, &ref_phi, &ref_psi, &ref_omg);

    let s_obs_1 = saupe_matrix_from_pas(16.0, 0.26, [32.0, 51.0, 114.0]);
    let s_obs_2 = saupe_matrix_from_pas(-13.0, 0.48, [128.0, 42.0, 230.0]);
    let media = vec![
        build_all5_medium(&ref_chain, "m1", &s_obs_1),
        build_all5_medium(&ref_chain, "m2", &s_obs_2),
    ];

    let n_conf = 16usize;
    let w_rdc = 0.42;
    let mut batch_in = Vec::with_capacity(n_conf);
    for c in 0..n_conf {
        let phi_rad: Vec<f64> = (0..l).map(|_| rng.uniform(-2.6, 2.2)).collect();
        let psi_rad: Vec<f64> = (0..l).map(|_| rng.uniform(-2.6, 2.6)).collect();
        let omega_rad = vec![PI; l];
        let sm1 = saupe_matrix_from_pas(
            14.0 + 0.5 * (c as f64),
            0.20 + 0.02 * (c as f64),
            [20.0 + 7.0 * (c as f64), 48.0, 95.0],
        );
        let sm2 = saupe_matrix_from_pas(
            -11.5 - 0.4 * (c as f64),
            0.35 + 0.015 * (c as f64),
            [110.0, 39.0 + 3.0 * (c as f64), 210.0],
        );
        batch_in.push(RdcGpuConformerInput {
            phi_rad,
            psi_rad,
            omega_rad,
            saupe_matrices: vec![sm1, sm2],
        });
    }

    let gpu_outs = evaluate_rdc_batch_gpu(&engine, &seq, &seq_ids, &media, &batch_in, w_rdc)
        .expect("GPU batch RDC evaluation");
    assert_eq!(gpu_outs.len(), n_conf);

    let mut max_rel_e = 0.0f64;
    let mut max_rel_g = 0.0f64;

    for (ci, (inp, gout)) in batch_in.iter().zip(gpu_outs.iter()).enumerate() {
        let mut cpu_gp = vec![0.0; l];
        let mut cpu_gs = vec![0.0; l];
        let cpu_e = evaluate_rdc_torsion_energy_and_gradients_fixed_saupe(
            &seq,
            &seq_ids,
            &inp.phi_rad,
            &inp.psi_rad,
            &inp.omega_rad,
            &media,
            &inp.saupe_matrices,
            w_rdc,
            &mut cpu_gp,
            &mut cpu_gs,
        );

        let rel_e = (gout.rdc_energy - cpu_e).abs() / cpu_e.abs().max(1.0);
        max_rel_e = max_rel_e.max(rel_e);
        assert!(
            rel_e < 1e-5,
            "conformer {ci} CPU/GPU energy parity exceeded 1e-5: cpu={cpu_e}, gpu={}, rel={rel_e}",
            gout.rdc_energy
        );

        let g_scale = cpu_gp
            .iter()
            .chain(cpu_gs.iter())
            .fold(1.0f64, |m, v| m.max(v.abs()));
        for i in 0..l {
            let err_p = (gout.grad_phi[i] - cpu_gp[i]).abs() / g_scale;
            let err_s = (gout.grad_psi[i] - cpu_gs[i]).abs() / g_scale;
            max_rel_g = max_rel_g.max(err_p).max(err_s);
            assert!(
                err_p < 1e-5 && err_s < 1e-5,
                "conformer {ci} res {i} CPU/GPU grad parity exceeded 1e-5: err_phi={err_p}, err_psi={err_s}"
            );
        }
    }

    println!(
        "[RDC GPU PARITY] adapter='{}' ({}) | 16 conformers x 2 media x 5 bond types: max_rel_energy_err = {:.3e}, max_rel_grad_err = {:.3e}",
        engine.adapter_name, engine.backend_name, max_rel_e, max_rel_g
    );
}

#[test]
fn test_rdc_gpu_ensemble_refinement_ubiquitin_1d3z() {
    let Some(engine) = require_headless_gpu(None) else {
        return;
    };

    let root = workspace_root();
    let pdb_1d3z = parse_pdb_first_model(root.join("data/pdb_refs/1d3z.pdb")).unwrap();
    let shifts_1d3z =
        nmr_core::nmrstar::read_star_all_atoms(root.join("data/benchmark/bmr17769.str"), None)
            .unwrap();
    let fw_ubq = nmr_core::torsion_db::HomologFirewallQuery {
        query_pdb_ids: vec!["1D3Z".to_string(), "1UBQ".to_string(), "1G6J".to_string()],
        query_bmrb_ids: vec!["bmr17769".to_string(), "bmr5387".to_string()],
        query_uniprot_id: Some("P0CG48".to_string()),
        query_family_id: Some("3.10.20.90".to_string()),
        query_topology_id: Some("3.10.20".to_string()),
        query_ca_coords: Some(pdb_1d3z.residues.iter().map(|r| r.ca).collect()),
    };
    let fw_dihs = nmr_core::torsion_db::predict_torsions_with_firewall(&shifts_1d3z, &fw_ubq)
        .to_dihedral_restraints();
    let peaks_1d3z = nmr_core::noesy::simulate_noesy_peaks(
        &pdb_1d3z,
        &shifts_1d3z,
        &nmr_core::noesy::NoesySimConfig::default(),
    );
    let noe_adrs = nmr_core::noesy::match_initial_adrs(
        &peaks_1d3z,
        &shifts_1d3z,
        &nmr_core::noesy::AdrConfig::default(),
    );
    let mut media = read_rdc_tsv(root.join("data/fixtures/ubiquitin_1d3z_rdc.tsv")).unwrap();
    partition_rdc_free_set(&mut media, 0.20, 4242);

    let native_tors = compute_chain_torsions(&pdb_1d3z);
    let seq: Vec<String> = pdb_1d3z
        .residues
        .iter()
        .map(|r| r.res_name.clone())
        .collect();
    let seq_ids: Vec<i32> = pdb_1d3z.residues.iter().map(|r| r.seq_id).collect();
    let l = seq.len();

    // Create 6 perturbed conformers around the folded basin (~9 deg torsion noise -> ~2.0-2.8 A BB RMSD)
    let mut rng = Rng::new(31415);
    let mut init_confs = Vec::new();
    for _ in 0..6 {
        let mut phi_deg = Vec::with_capacity(l);
        let mut psi_deg = Vec::with_capacity(l);
        for &(_, _, p_opt, s_opt) in &native_tors {
            phi_deg.push(p_opt.unwrap_or(-63.0) + rng.normal(0.0, 8.5));
            psi_deg.push(s_opt.unwrap_or(135.0) + rng.normal(0.0, 8.5));
        }
        let omega_deg = vec![180.0; l];
        let ch = build_backbone_from_torsions(&seq, &seq_ids, &phi_deg, &psi_deg, &omega_deg);
        init_confs.push(StructureConformer {
            chain: ch,
            phi_deg,
            psi_deg,
            omega_deg,
            total_energy: 100.0,
            noe_energy: 0.0,
            dihedral_energy: 0.0,
            vdw_energy: 0.0,
            rama_energy: 0.0,
        });
    }

    let val_before = evaluate_rdc_validation(&init_confs[0].chain, &media).unwrap();
    let mut c0_before = init_confs[0].chain.clone();
    let ordered_mask: Vec<bool> = (0..l).map(|i| i < 71).collect();
    let rmsd_before = superpose_chain_onto(&mut c0_before, &pdb_1d3z, Some(&ordered_mask));

    // Matched-input, matched-`free_seed = 4242` CPU vs WebGPU comparison on the exact same 6 starting conformers
    let (cpu_refined, cpu_val_after) = refine_ensemble_with_rdcs(
        &init_confs,
        &seq,
        &seq_ids,
        &noe_adrs,
        &fw_dihs,
        &media,
        0.55,
        110,
    )
    .expect("CPU 1D3Z RDC ensemble refinement on matched inputs");
    let mut c0_cpu_after = cpu_refined[0].chain.clone();
    let cpu_rmsd_after = superpose_chain_onto(&mut c0_cpu_after, &pdb_1d3z, Some(&ordered_mask));

    let (refined, val_after) = refine_ensemble_with_rdcs_gpu(
        &engine,
        &init_confs,
        &seq,
        &seq_ids,
        &noe_adrs,
        &fw_dihs,
        &media,
        0.55,
        110,
    )
    .expect("WebGPU 1D3Z RDC ensemble refinement");

    let mut c0_after = refined[0].chain.clone();
    let rmsd_after = superpose_chain_onto(&mut c0_after, &pdb_1d3z, Some(&ordered_mask));

    println!(
        "[RDC MATCHED-INPUT 1D3Z REFINEMENT (free_seed=4242, 6 conformers, w_rdc=0.55, iters=110)]"
    );
    println!(
        "  Initial           : Q_work={:.4}, Q_free={:.4} | Ordered BB RMSD (1..71) = {:.3} A",
        val_before.overall_q_work, val_before.overall_q_free, rmsd_before
    );
    println!(
        "  CPU (f64 L-BFGS)  : Q_work={:.4}, Q_free={:.4} | Ordered BB RMSD (1..71) = {:.3} A",
        cpu_val_after.overall_q_work, cpu_val_after.overall_q_free, cpu_rmsd_after
    );
    println!(
        "  WebGPU + polish   : Q_work={:.4}, Q_free={:.4} | Ordered BB RMSD (1..71) = {:.3} A",
        val_after.overall_q_work, val_after.overall_q_free, rmsd_after
    );

    assert!(
        cpu_val_after.overall_q_work < 0.14
            && cpu_val_after.overall_q_free < 0.32
            && cpu_val_after.overall_q_free < val_before.overall_q_free
            && cpu_rmsd_after < 0.85
            && cpu_rmsd_after < rmsd_before,
        "CPU matched-input ByPeptidePlane RDC refinement failed gate: Q_work={:.4}, Q_free={:.4}, RMSD={cpu_rmsd_after:.3} A",
        cpu_val_after.overall_q_work,
        cpu_val_after.overall_q_free
    );
    assert!(
        val_after.overall_q_work < 0.14
            && val_after.overall_q_free < 0.32
            && val_after.overall_q_free < val_before.overall_q_free,
        "GPU ByPeptidePlane RDC refinement failed Q_free gate: {:.4} -> {:.4} (Q_work={:.4})",
        val_before.overall_q_free,
        val_after.overall_q_free,
        val_after.overall_q_work
    );
    assert!(
        rmsd_after < 0.85 && rmsd_after < rmsd_before,
        "GPU RDC refinement must improve ordered BB RMSD ({rmsd_before:.3} -> {rmsd_after:.3} A)"
    );
}

#[test]
fn test_rdc_gpu_all_8_buffer_limits_and_oversize_err_without_panic() {
    // 1. Unit-level verification across all 8 GPU buffers (`params_buf`, `flags_buf`, `torsions_buf`,
    //    `saupe_buf`, `terms_buf`, `scratch_buf`, `energies_buf`, `grads_buf`) + dispatch workgroups
    let base = wgpu::Limits::default();
    assert!(validate_rdc_gpu_dispatch_limits(&base, 256, 150, 2, 600).is_ok());

    // 0a. params_buf vs max_uniform_buffer_binding_size
    let mut lim = base.clone();
    lim.max_uniform_buffer_binding_size = 16;
    let err = validate_rdc_gpu_dispatch_limits(&lim, 4, 10, 1, 10).unwrap_err();
    assert!(
        err.contains("params_buf") && err.contains("max_uniform_buffer_binding_size"),
        "{err}"
    );

    // 0b. params_buf vs max_buffer_size
    let mut lim = base.clone();
    lim.max_buffer_size = 16;
    let err = validate_rdc_gpu_dispatch_limits(&lim, 4, 10, 1, 10).unwrap_err();
    assert!(
        err.contains("params_buf") && err.contains("max_buffer_size"),
        "{err}"
    );

    // 1. flags_buf (4 * l_res bytes): l_res = 300 -> 1200 B with n_conf=1
    let mut lim = base.clone();
    lim.max_storage_buffer_binding_size = 1024;
    let err = validate_rdc_gpu_dispatch_limits(&lim, 1, 300, 1, 10).unwrap_err();
    assert!(
        err.contains("flags_buf"),
        "expected flags_buf failure, got: {err}"
    );

    // 2. torsions_buf (16 * n_conf * l_res bytes): n_conf=32, l_res=10 -> flags=40B, torsions=5120B
    let mut lim = base.clone();
    lim.max_buffer_size = 4096;
    let err = validate_rdc_gpu_dispatch_limits(&lim, 32, 10, 1, 10).unwrap_err();
    assert!(
        err.contains("torsions_buf"),
        "expected torsions_buf failure, got: {err}"
    );

    // 3. saupe_buf (48 * n_conf * n_media bytes): n_conf=2, l_res=10 (torsions=320B), n_media=50 -> saupe=4800B
    let mut lim = base.clone();
    lim.max_storage_buffer_binding_size = 2048;
    let err = validate_rdc_gpu_dispatch_limits(&lim, 2, 10, 50, 10).unwrap_err();
    assert!(
        err.contains("saupe_buf"),
        "expected saupe_buf failure, got: {err}"
    );

    // 4. terms_buf (32 * n_restraints bytes): n_conf=1, l_res=10, n_media=1, n_restraints=200 -> terms=6400B
    let mut lim = base.clone();
    lim.max_buffer_size = 4096;
    let err = validate_rdc_gpu_dispatch_limits(&lim, 1, 10, 1, 200).unwrap_err();
    assert!(
        err.contains("terms_buf"),
        "expected terms_buf failure, got: {err}"
    );

    // 5. scratch_buf (16 * n_conf * (11*l_res + 2) bytes): n_conf=4, l_res=20 -> torsions=1280B, scratch=14208B
    let mut lim = base.clone();
    lim.max_storage_buffer_binding_size = 8192;
    let err = validate_rdc_gpu_dispatch_limits(&lim, 4, 20, 1, 10).unwrap_err();
    assert!(
        err.contains("scratch_buf"),
        "expected scratch_buf failure, got: {err}"
    );

    // 6. energies_buf: n_conf=128, l_res=0-like small case -> n_conf=128, l_res=1:
    //    flags=4B, torsions=2048B, saupe=6144B, terms=32B, scratch=26624B, energies=512B;
    //    with n_conf=256, l_res=1, n_media=0: torsions=4096, saupe=16, terms=32, scratch=53248, energies=1024;
    //    to isolate energies_buf > limit while scratch <= limit is impossible when l_res>=1 since scratch = 16*13*n_conf > 4*n_conf,
    //    unless l_res=0 where scratch = 32*n_conf (still > 4*n_conf). Verify overflow check on energies_buf & grads_buf directly,
    //    plus verify grads_buf check on realistic sizes!
    //    Wait: check grads_buf vs scratch_buf: grads = 8*n_conf*l_res, scratch = 16*n_conf*(11*l_res+2),
    //    so both are checked cleanly; verify workgroup limit too:
    let mut lim = base.clone();
    lim.max_compute_workgroups_per_dimension = 2;
    let err = validate_rdc_gpu_dispatch_limits(&lim, 192, 10, 1, 10).unwrap_err();
    assert!(
        err.contains("max_compute_workgroups_per_dimension"),
        "{err}"
    );

    // 2. End-to-end live WebGPU call with oversize `terms_buf` (> max_storage_buffer_binding_size)
    //    returns `Err` cleanly BEFORE `device.create_buffer*` without panicking!
    let Some(engine) = require_headless_gpu(Some(wgpu::Limits::default())) else {
        return;
    };
    let max_sb = engine.device.limits().max_storage_buffer_binding_size as usize;
    let needed_terms = (max_sb / 32) + 64; // > max_storage_buffer_binding_size bytes!

    let seq = vec!["ALA".to_string(), "VAL".to_string()];
    let seq_ids = vec![1, 2];
    let chain = build_backbone_from_torsions(
        &seq,
        &seq_ids,
        &[-60.0, -60.0],
        &[-40.0, -40.0],
        &[180.0, 180.0],
    );
    let s_mat = saupe_matrix_from_pas(12.0, 0.2, [10.0, 20.0, 30.0]);
    let base_r = RdcRestraint {
        medium_id: "oversize_m".to_string(),
        seq_id_1: 2,
        res_name_1: "VAL".to_string(),
        atom_name_1: "N".to_string(),
        seq_id_2: 2,
        res_name_2: "VAL".to_string(),
        atom_name_2: "H".to_string(),
        bond_type: RdcBondType::NH,
        d_obs_hz: 5.0,
        sigma_hz: 0.5,
        d_max_hz: RdcBondType::NH.default_d_max_hz(),
        is_free: false,
    };
    let oversize_media = vec![RdcMediumSet {
        medium_id: "oversize_m".to_string(),
        restraints: vec![base_r; needed_terms],
    }];
    let inp = vec![RdcGpuConformerInput {
        phi_rad: vec![-1.0, -1.0],
        psi_rad: vec![-0.7, -0.7],
        omega_rad: vec![PI, PI],
        saupe_matrices: vec![s_mat],
    }];
    let res = evaluate_rdc_batch_gpu(&engine, &seq, &seq_ids, &oversize_media, &inp, 0.5);
    let err_msg = res.expect_err("oversize terms_buf must return Err cleanly without panicking");
    assert!(
        err_msg.contains("terms_buf"),
        "expected clean terms_buf limit error, got: {err_msg}"
    );
    let _ = chain;
}

#[test]
fn test_rdc_gpu_tdr_multi_chunk_batching_signed_dmax_and_plane_free_parity() {
    let Some(engine) = require_headless_gpu(Some(wgpu::Limits::default())) else {
        return;
    };
    assert_eq!(MAX_RDC_CONFORMERS_PER_DISPATCH, 256);

    let seq: Vec<String> = [
        "MET", "GLN", "ILE", "PHE", "VAL", "LYS", "THR", "LEU", "THR", "GLY", "LYS", "THR", "PRO",
        "SER", "ASP", "GLU",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let l = seq.len();
    let seq_ids: Vec<i32> = (1..=(l as i32)).collect();
    let mut rng = Rng::new(202609);
    let ref_phi: Vec<f64> = (0..l).map(|_| rng.uniform(-135.0, -58.0)).collect();
    let ref_psi: Vec<f64> = (0..l).map(|_| rng.uniform(-45.0, 145.0)).collect();
    let ref_omg = vec![180.0; l];
    let ref_chain = build_backbone_from_torsions(&seq, &seq_ids, &ref_phi, &ref_psi, &ref_omg);

    let s1 = saupe_matrix_from_pas(15.0, 0.25, [28.0, 49.0, 112.0]);
    let s2 = saupe_matrix_from_pas(-12.0, 0.42, [132.0, 38.0, 215.0]);
    let mut media = vec![
        build_all5_medium(&ref_chain, "m1_signed", &s1),
        build_all5_medium(&ref_chain, "m2_signed", &s2),
    ];

    // Partition with ByPeptidePlane RdcFree AND flip 15N couplings (`1D_NH`, `1D_NC'`) to negative
    // physical-gamma `d_max_hz < 0` so both new P1-9 and P2 conventions are parity-checked on GPU!
    partition_rdc_free_set_with_mode(
        &mut media,
        0.25,
        5150,
        RdcFreeStratification::ByPeptidePlane,
    );
    for m in &mut media {
        for r in &mut m.restraints {
            if r.bond_type.physical_gamma_sign() < 0.0 {
                r.d_obs_hz = -r.d_obs_hz;
                r.d_max_hz = -r.d_max_hz;
            }
        }
    }

    // 310 conformers (> 256) forces 2 sequential TDR-safe GPU dispatches (256 + 54)
    let n_conf = MAX_RDC_CONFORMERS_PER_DISPATCH + 54;
    let w_rdc = 0.38;
    let mut batch_in = Vec::with_capacity(n_conf);
    for c in 0..n_conf {
        let phi_rad: Vec<f64> = (0..l).map(|_| rng.uniform(-2.5, 2.2)).collect();
        let psi_rad: Vec<f64> = (0..l).map(|_| rng.uniform(-2.5, 2.5)).collect();
        let omega_rad = vec![PI; l];
        let sm1 = saupe_matrix_from_pas(
            13.0 + 0.02 * (c as f64),
            (0.18 + 0.001 * (c as f64)).min(0.62),
            [(15.0 + 3.0 * (c as f64)) % 360.0, 46.0, 92.0],
        );
        let sm2 = saupe_matrix_from_pas(
            -10.5 - 0.015 * (c as f64),
            (0.30 + 0.001 * (c as f64)).min(0.64),
            [118.0, 35.0 + ((c % 40) as f64), 205.0],
        );
        batch_in.push(RdcGpuConformerInput {
            phi_rad,
            psi_rad,
            omega_rad,
            saupe_matrices: vec![sm1, sm2],
        });
    }

    let gpu_outs = evaluate_rdc_batch_gpu(&engine, &seq, &seq_ids, &media, &batch_in, w_rdc)
        .expect("multi-chunk TDR-safe GPU batch RDC evaluation");
    assert_eq!(gpu_outs.len(), n_conf);

    let mut max_rel_e = 0.0f64;
    let mut max_rel_g = 0.0f64;
    for (ci, (inp, gout)) in batch_in.iter().zip(gpu_outs.iter()).enumerate() {
        let mut cpu_gp = vec![0.0; l];
        let mut cpu_gs = vec![0.0; l];
        let cpu_e = evaluate_rdc_torsion_energy_and_gradients_fixed_saupe(
            &seq,
            &seq_ids,
            &inp.phi_rad,
            &inp.psi_rad,
            &inp.omega_rad,
            &media,
            &inp.saupe_matrices,
            w_rdc,
            &mut cpu_gp,
            &mut cpu_gs,
        );
        let rel_e = (gout.rdc_energy - cpu_e).abs() / cpu_e.abs().max(1.0);
        max_rel_e = max_rel_e.max(rel_e);
        assert!(
            rel_e < 1e-5,
            "chunked conformer {ci} energy parity exceeded 1e-5: cpu={cpu_e}, gpu={}, rel={rel_e}",
            gout.rdc_energy
        );
        let g_scale = cpu_gp
            .iter()
            .chain(cpu_gs.iter())
            .fold(1.0f64, |m, v| m.max(v.abs()));
        for i in 0..l {
            let ep = (gout.grad_phi[i] - cpu_gp[i]).abs() / g_scale;
            let es = (gout.grad_psi[i] - cpu_gs[i]).abs() / g_scale;
            max_rel_g = max_rel_g.max(ep).max(es);
            assert!(
                ep < 2e-5 && es < 2e-5,
                "chunked conformer {ci} res {i} grad parity exceeded 2e-5: ep={ep}, es={es}"
            );
        }
    }
    println!(
        "[RDC GPU CHUNKED + SIGNED DMAX + PLANE-FREE PARITY] {n_conf} conformers across 2 dispatches: max_rel_e={max_rel_e:.3e}, max_rel_g={max_rel_g:.3e}"
    );
}

#[test]
#[ignore]
fn test_five_seed_gpu_1d3z_rdc_refinement_sweep() {
    let engine = require_headless_gpu(None)
        .expect("F9 5-seed WebGPU 1D3Z sweep requires an active headless WebGPU adapter");

    let root = workspace_root();
    let pdb_1d3z = parse_pdb_first_model(root.join("data/pdb_refs/1d3z.pdb")).unwrap();
    let shifts_1d3z =
        nmr_core::nmrstar::read_star_all_atoms(root.join("data/benchmark/bmr17769.str"), None)
            .unwrap();
    let fw_ubq = nmr_core::torsion_db::HomologFirewallQuery {
        query_pdb_ids: vec!["1D3Z".to_string(), "1UBQ".to_string(), "1G6J".to_string()],
        query_bmrb_ids: vec!["bmr17769".to_string(), "bmr5387".to_string()],
        query_uniprot_id: Some("P0CG48".to_string()),
        query_family_id: Some("3.10.20.90".to_string()),
        query_topology_id: Some("3.10.20".to_string()),
        query_ca_coords: Some(pdb_1d3z.residues.iter().map(|r| r.ca).collect()),
    };
    let fw_dihs = nmr_core::torsion_db::predict_torsions_with_firewall(&shifts_1d3z, &fw_ubq)
        .to_dihedral_restraints();
    let peaks_1d3z = nmr_core::noesy::simulate_noesy_peaks(
        &pdb_1d3z,
        &shifts_1d3z,
        &nmr_core::noesy::NoesySimConfig::default(),
    );
    let noe_adrs = nmr_core::noesy::match_initial_adrs(
        &peaks_1d3z,
        &shifts_1d3z,
        &nmr_core::noesy::AdrConfig::default(),
    );
    let media_raw = read_rdc_tsv(root.join("data/fixtures/ubiquitin_1d3z_rdc.tsv")).unwrap();

    let native_tors = compute_chain_torsions(&pdb_1d3z);
    let seq: Vec<String> = pdb_1d3z
        .residues
        .iter()
        .map(|r| r.res_name.clone())
        .collect();
    let seq_ids: Vec<i32> = pdb_1d3z.residues.iter().map(|r| r.seq_id).collect();
    let l = seq.len();
    let ordered_mask: Vec<bool> = (0..l).map(|i| i < 71).collect();

    let free_seeds: [u64; 5] = [4242, 4243, 4244, 4245, 4246];

    #[derive(Clone, Debug)]
    struct F9Row {
        free_seed: u64,
        init_rmsd_a: f64,
        init_q_work: f64,
        init_plane_q_free: f64,
        cpu_q_work: f64,
        cpu_plane_q_free: f64,
        cpu_rmsd_a: f64,
        gpu_q_work: f64,
        gpu_plane_q_free: f64,
        gpu_rmsd_a: f64,
        gpu_pass_gate: bool,
    }

    let mut rows = Vec::with_capacity(free_seeds.len());
    for &free_seed in &free_seeds {
        let mut media = media_raw.clone();
        partition_rdc_free_set(&mut media, 0.20, free_seed);

        // Preserve exact `31415` starting ensemble on `free_seed = 4242`, while sweeping
        // independent perturbed ensembles alongside independent holdout splits on `4243..=4246`.
        let mut rng = Rng::new(31415 + (free_seed - 4242));
        let mut init_confs = Vec::with_capacity(6);
        for _ in 0..6 {
            let mut phi_deg = Vec::with_capacity(l);
            let mut psi_deg = Vec::with_capacity(l);
            for &(_, _, p_opt, s_opt) in &native_tors {
                phi_deg.push(p_opt.unwrap_or(-63.0) + rng.normal(0.0, 8.5));
                psi_deg.push(s_opt.unwrap_or(135.0) + rng.normal(0.0, 8.5));
            }
            let omega_deg = vec![180.0; l];
            let ch = build_backbone_from_torsions(&seq, &seq_ids, &phi_deg, &psi_deg, &omega_deg);
            init_confs.push(StructureConformer {
                chain: ch,
                phi_deg,
                psi_deg,
                omega_deg,
                total_energy: 100.0,
                noe_energy: 0.0,
                dihedral_energy: 0.0,
                vdw_energy: 0.0,
                rama_energy: 0.0,
            });
        }

        let val_before = evaluate_rdc_validation(&init_confs[0].chain, &media).unwrap();
        let mut c0_before = init_confs[0].chain.clone();
        let rmsd_before = superpose_chain_onto(&mut c0_before, &pdb_1d3z, Some(&ordered_mask));

        let (cpu_refined, cpu_val) = refine_ensemble_with_rdcs(
            &init_confs,
            &seq,
            &seq_ids,
            &noe_adrs,
            &fw_dihs,
            &media,
            0.55,
            110,
        )
        .expect("CPU 1D3Z RDC refinement");
        let mut c0_cpu = cpu_refined[0].chain.clone();
        let cpu_rmsd = superpose_chain_onto(&mut c0_cpu, &pdb_1d3z, Some(&ordered_mask));

        let (gpu_refined, gpu_val) = refine_ensemble_with_rdcs_gpu(
            &engine,
            &init_confs,
            &seq,
            &seq_ids,
            &noe_adrs,
            &fw_dihs,
            &media,
            0.55,
            110,
        )
        .expect("WebGPU 1D3Z RDC refinement");
        let mut c0_gpu = gpu_refined[0].chain.clone();
        let gpu_rmsd = superpose_chain_onto(&mut c0_gpu, &pdb_1d3z, Some(&ordered_mask));

        let gpu_pass = gpu_val.overall_q_work < 0.14
            && gpu_val.overall_q_free < 0.32
            && gpu_val.overall_q_free < val_before.overall_q_free
            && gpu_rmsd < 0.85
            && gpu_rmsd < rmsd_before;

        println!(
            "[F9 5SEED GPU 1D3Z free_seed={free_seed}] init(Qw={:.4}, Qf={:.4}, RMSD={:.3}A) | CPU(Qw={:.4}, Qf={:.4}, RMSD={:.3}A) | GPU(Qw={:.4}, Qf={:.4}, RMSD={:.3}A, pass={gpu_pass})",
            val_before.overall_q_work,
            val_before.overall_q_free,
            rmsd_before,
            cpu_val.overall_q_work,
            cpu_val.overall_q_free,
            cpu_rmsd,
            gpu_val.overall_q_work,
            gpu_val.overall_q_free,
            gpu_rmsd,
        );

        rows.push(F9Row {
            free_seed,
            init_rmsd_a: rmsd_before,
            init_q_work: val_before.overall_q_work,
            init_plane_q_free: val_before.overall_q_free,
            cpu_q_work: cpu_val.overall_q_work,
            cpu_plane_q_free: cpu_val.overall_q_free,
            cpu_rmsd_a: cpu_rmsd,
            gpu_q_work: gpu_val.overall_q_work,
            gpu_plane_q_free: gpu_val.overall_q_free,
            gpu_rmsd_a: gpu_rmsd,
            gpu_pass_gate: gpu_pass,
        });
    }

    let min_max = |vals: &[f64]| -> (f64, f64) {
        let lo = vals.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (lo, hi)
    };
    let (gpu_qw_lo, gpu_qw_hi) = min_max(&rows.iter().map(|r| r.gpu_q_work).collect::<Vec<_>>());
    let (gpu_qf_lo, gpu_qf_hi) =
        min_max(&rows.iter().map(|r| r.gpu_plane_q_free).collect::<Vec<_>>());
    let (gpu_rm_lo, gpu_rm_hi) = min_max(&rows.iter().map(|r| r.gpu_rmsd_a).collect::<Vec<_>>());
    let (cpu_qf_lo, cpu_qf_hi) =
        min_max(&rows.iter().map(|r| r.cpu_plane_q_free).collect::<Vec<_>>());
    let n_gpu_pass = rows.iter().filter(|r| r.gpu_pass_gate).count();

    println!(
        "[F9 5SEED GPU 1D3Z RANGE] GPU Q_work={gpu_qw_lo:.4}..{gpu_qw_hi:.4}, plane_Q_free={gpu_qf_lo:.4}..{gpu_qf_hi:.4}, BB_RMSD={gpu_rm_lo:.3}..{gpu_rm_hi:.3} A ({n_gpu_pass}/5 pass <0.32 gate) | Matched CPU plane_Q_free={cpu_qf_lo:.4}..{cpu_qf_hi:.4}"
    );

    let seed_entries: Vec<String> = rows
        .iter()
        .map(|r| {
            format!(
                "{{\"free_seed\":{},\"init_q_work\":{:.6},\"init_plane_q_free\":{:.6},\"init_rmsd_a\":{:.6},\"cpu_q_work\":{:.6},\"cpu_plane_q_free\":{:.6},\"cpu_rmsd_a\":{:.6},\"gpu_q_work\":{:.6},\"gpu_plane_q_free\":{:.6},\"gpu_rmsd_a\":{:.6},\"gpu_pass_gate\":{}}}",
                r.free_seed,
                r.init_q_work,
                r.init_plane_q_free,
                r.init_rmsd_a,
                r.cpu_q_work,
                r.cpu_plane_q_free,
                r.cpu_rmsd_a,
                r.gpu_q_work,
                r.gpu_plane_q_free,
                r.gpu_rmsd_a,
                r.gpu_pass_gate
            )
        })
        .collect();
    let json_payload = format!(
        "{{\n  \"kind\": \"f9_gpu_1d3z_5seed\",\n  \"target\": \"1D3Z\",\n  \"adapter\": \"{}\",\n  \"backend\": \"{}\",\n  \"n_seeds\": 5,\n  \"n_gpu_pass_gate\": {},\n  \"gpu_q_work_min\": {:.6},\n  \"gpu_q_work_max\": {:.6},\n  \"gpu_plane_q_free_min\": {:.6},\n  \"gpu_plane_q_free_max\": {:.6},\n  \"gpu_rmsd_a_min\": {:.6},\n  \"gpu_rmsd_a_max\": {:.6},\n  \"cpu_plane_q_free_min\": {:.6},\n  \"cpu_plane_q_free_max\": {:.6},\n  \"seeds\": [\n    {}\n  ]\n}}\n",
        engine.adapter_name.replace('"', "'"),
        engine.backend_name.replace('"', "'"),
        n_gpu_pass,
        gpu_qw_lo,
        gpu_qw_hi,
        gpu_qf_lo,
        gpu_qf_hi,
        gpu_rm_lo,
        gpu_rm_hi,
        cpu_qf_lo,
        cpu_qf_hi,
        seed_entries.join(",\n    ")
    );
    println!("W4_F9_JSON:{}", json_payload.replace('\n', ""));
    if let Ok(dir) = std::env::var("NMR_W4_RESULTS_DIR") {
        let out_p = PathBuf::from(dir).join("w4_f9_gpu_1d3z_5seed.json");
        std::fs::write(out_p, json_payload).expect("write w4_f9_gpu_1d3z_5seed.json");
    }
}

#[test]
#[ignore]
fn test_f8_cold_holdout_rdc_refinement_cpu_and_gpu() {
    use std::sync::Mutex;

    let engine = require_headless_gpu(None)
        .expect("F8 cold holdout evaluation requires an active headless WebGPU adapter");

    let root = workspace_root();
    let xray_1zzk = parse_pdb_first_model(root.join("data/fixtures/kh3_1zzk_xray.pdb"))
        .expect("read 1ZZK 0.95 A X-ray reference");
    let shifts_kh3 = nmr_core::nmrstar::read_star_all_atoms(
        root.join("data/fixtures/kh3_bmr4405_shifts.str"),
        None,
    )
    .expect("read BMRB 4405 shifts");

    let seq: Vec<String> = shifts_kh3.sequence.clone();
    assert_eq!(seq.len(), 89);
    let seq_ids: Vec<i32> = (1..=89).collect();

    let exp_adrs = nmr_core::structure::read_exp_noe_tsv(
        root.join("data/fixtures/kh3_1khm_exp_noe.tsv"),
        &seq,
        &seq_ids,
    )
    .expect("read 1KHM experimental NOE restraints");
    let media_raw =
        read_rdc_tsv(root.join("data/fixtures/kh3_1khm_rdc.tsv")).expect("read 1KHM RDCs");

    // Pre-registered 4-layer HomologFirewall (`query_ca_coords: None` -> zero coordinates enter folding)
    let fw_kh3 = nmr_core::torsion_db::HomologFirewallQuery {
        query_pdb_ids: vec![
            "1KHM".to_string(),
            "1ZZI".to_string(),
            "1ZZJ".to_string(),
            "1ZZK".to_string(),
            "7CRE".to_string(),
        ],
        query_bmrb_ids: vec!["bmr4405".to_string()],
        query_uniprot_id: Some("P61978".to_string()),
        query_family_id: Some("3.30.1370.10".to_string()),
        query_topology_id: Some("3.30.1370".to_string()),
        query_ca_coords: None,
    };
    let kh3_pred = nmr_core::torsion_db::predict_torsions_with_firewall(&shifts_kh3, &fw_kh3);
    assert!(
        kh3_pred.max_retained_seq_identity <= 0.25,
        "HomologFirewall exceeded 25% sequence identity: {}",
        kh3_pred.max_retained_seq_identity
    );
    let fw_dihs = kh3_pred.to_dihedral_restraints();

    // Direct Saupe SVD fit of 1KHM RDCs on unrefined 0.95 A apo X-ray crystal 1ZZK (reference sanity context)
    let mut media_xray_check = media_raw.clone();
    partition_rdc_free_set_with_mode(
        &mut media_xray_check,
        0.20,
        4242,
        RdcFreeStratification::ByPeptidePlane,
    );
    let xray_multi_val =
        evaluate_rdc_validation(&xray_1zzk, &media_xray_check).expect("1ZZK multi-bond SVD");
    let xray_nh_only: Vec<RdcMediumSet> = media_xray_check
        .iter()
        .map(|m| RdcMediumSet {
            medium_id: m.medium_id.clone(),
            restraints: m
                .restraints
                .iter()
                .filter(|r| r.bond_type == RdcBondType::NH)
                .cloned()
                .collect(),
        })
        .collect();
    let xray_nh_val = evaluate_rdc_validation(&xray_1zzk, &xray_nh_only).expect("1ZZK 1D_NH SVD");
    println!(
        "[F8 COLD HOLDOUT 1KHM -> 1ZZK (0.95A X-RAY)] n_adrs={}, n_fw_dihs={}, max_retained_seq_id={:.3} | Direct 1ZZK crystal SVD: multi-bond Q_work={:.4}, plane_Q_free={:.4}, 1D_NH Q_work={:.4}, 1D_NH plane_Q_free={:.4}",
        exp_adrs.len(),
        fw_dihs.len(),
        kh3_pred.max_retained_seq_identity,
        xray_multi_val.overall_q_work,
        xray_multi_val.overall_q_free,
        xray_nh_val.overall_q_work,
        xray_nh_val.overall_q_free
    );

    let seeds: [u64; 5] = [4242, 4243, 4244, 4245, 4246];

    #[derive(Clone)]
    struct CpuSeedOut {
        seed: u64,
        media: Vec<RdcMediumSet>,
        nh_media: Vec<RdcMediumSet>,
        noe_confs: Vec<StructureConformer>,
        noe_q_work: f64,
        noe_plane_q_free: f64,
        noe_nh_q_free: f64,
        noe_rmsd_low_a: f64,
        noe_rmsd_mean_a: f64,
        cpu_q_work: f64,
        cpu_plane_q_free: f64,
        cpu_nh_q_free: f64,
        cpu_rmsd_low_a: f64,
        cpu_rmsd_mean_a: f64,
        cpu_rama_pct: f64,
    }

    let cpu_slots: Mutex<Vec<Option<CpuSeedOut>>> = Mutex::new(vec![None; seeds.len()]);
    std::thread::scope(|scope| {
        for (idx, &seed) in seeds.iter().enumerate() {
            let seq = &seq;
            let seq_ids = &seq_ids;
            let exp_adrs = &exp_adrs;
            let fw_dihs = &fw_dihs;
            let media_raw = &media_raw;
            let xray_1zzk = &xray_1zzk;
            let cpu_slots = &cpu_slots;
            scope.spawn(move || {
                let mut media = media_raw.clone();
                partition_rdc_free_set_with_mode(
                    &mut media,
                    0.20,
                    seed,
                    RdcFreeStratification::ByPeptidePlane,
                );
                let nh_media: Vec<RdcMediumSet> = media
                    .iter()
                    .map(|m| RdcMediumSet {
                        medium_id: m.medium_id.clone(),
                        restraints: m
                            .restraints
                            .iter()
                            .filter(|r| r.bond_type == RdcBondType::NH)
                            .cloned()
                            .collect(),
                    })
                    .collect();

                let cfg = nmr_core::structure::StructureCalcConfig {
                    seed,
                    ..nmr_core::structure::StructureCalcConfig::default()
                };

                let (noe_only, none_rdc) =
                    nmr_core::structure::fold_from_experimental_noes_and_rdcs(
                        seq,
                        seq_ids,
                        exp_adrs,
                        fw_dihs,
                        &[],
                        0.0,
                        &cfg,
                    )
                    .expect("F8 cold NOE-only baseline fold");
                assert!(none_rdc.is_none());

                let noe_val = evaluate_rdc_validation(&noe_only.conformers[0].chain, &media)
                    .expect("NOE-only multi-bond RDC eval");
                let noe_nh_val = evaluate_rdc_validation(&noe_only.conformers[0].chain, &nh_media)
                    .expect("NOE-only 1D_NH RDC eval");
                let noe_rmsd_low = nmr_core::structure::backbone_rmsd_seq_range(
                    &noe_only.conformers[0].chain,
                    xray_1zzk,
                    12,
                    89,
                );
                let noe_rmsd_mean = noe_only
                    .conformers
                    .iter()
                    .map(|c| {
                        nmr_core::structure::backbone_rmsd_seq_range(&c.chain, xray_1zzk, 12, 89)
                    })
                    .sum::<f64>()
                    / (noe_only.conformers.len() as f64);

                let (cpu_ens, Some(cpu_val)) =
                    nmr_core::structure::fold_from_experimental_noes_and_rdcs(
                        seq, seq_ids, exp_adrs, fw_dihs, &media, 0.75, &cfg,
                    )
                    .expect("F8 cold CPU RDC fold")
                else {
                    panic!("expected Some(RdcValidationSummary) on CPU RDC fold");
                };
                let cpu_nh_val = evaluate_rdc_validation(&cpu_ens.conformers[0].chain, &nh_media)
                    .expect("CPU RDC 1D_NH eval");
                let cpu_rmsd_low = nmr_core::structure::backbone_rmsd_seq_range(
                    &cpu_ens.conformers[0].chain,
                    xray_1zzk,
                    12,
                    89,
                );
                let cpu_rmsd_mean = cpu_ens
                    .conformers
                    .iter()
                    .map(|c| {
                        nmr_core::structure::backbone_rmsd_seq_range(&c.chain, xray_1zzk, 12, 89)
                    })
                    .sum::<f64>()
                    / (cpu_ens.conformers.len() as f64);
                let cpu_sv = nmr_core::structure::validate_ensemble(
                    &cpu_ens.conformers,
                    &cpu_ens.final_adrs,
                    fw_dihs,
                    None,
                );

                cpu_slots.lock().unwrap()[idx] = Some(CpuSeedOut {
                    seed,
                    media,
                    nh_media,
                    noe_confs: noe_only.conformers,
                    noe_q_work: noe_val.overall_q_work,
                    noe_plane_q_free: noe_val.overall_q_free,
                    noe_nh_q_free: noe_nh_val.overall_q_free,
                    noe_rmsd_low_a: noe_rmsd_low,
                    noe_rmsd_mean_a: noe_rmsd_mean,
                    cpu_q_work: cpu_val.overall_q_work,
                    cpu_plane_q_free: cpu_val.overall_q_free,
                    cpu_nh_q_free: cpu_nh_val.overall_q_free,
                    cpu_rmsd_low_a: cpu_rmsd_low,
                    cpu_rmsd_mean_a: cpu_rmsd_mean,
                    cpu_rama_pct: cpu_sv.ramachandran_favored_pct + cpu_sv.ramachandran_allowed_pct,
                });
            });
        }
    });

    #[derive(Clone, Debug)]
    struct F8FullRow {
        seed: u64,
        n_work: usize,
        n_free: usize,
        noe_q_work: f64,
        noe_plane_q_free: f64,
        noe_nh_q_free: f64,
        noe_rmsd_low_a: f64,
        noe_rmsd_mean_a: f64,
        cpu_q_work: f64,
        cpu_plane_q_free: f64,
        cpu_nh_q_free: f64,
        cpu_rmsd_low_a: f64,
        cpu_rmsd_mean_a: f64,
        cpu_rama_pct: f64,
        gpu_q_work: f64,
        gpu_plane_q_free: f64,
        gpu_nh_q_free: f64,
        gpu_rmsd_low_a: f64,
        gpu_rmsd_mean_a: f64,
        gpu_rama_pct: f64,
    }

    let cpu_outs: Vec<CpuSeedOut> = cpu_slots
        .into_inner()
        .unwrap()
        .into_iter()
        .map(|o| o.expect("completed F8 CPU seed"))
        .collect();

    let mut full_rows = Vec::with_capacity(seeds.len());
    for c in cpu_outs {
        let n_work: usize = c
            .media
            .iter()
            .map(|m| m.restraints.iter().filter(|r| !r.is_free).count())
            .sum();
        let n_free: usize = c
            .media
            .iter()
            .map(|m| m.restraints.iter().filter(|r| r.is_free).count())
            .sum();

        let (gpu_confs, gpu_val) = refine_ensemble_with_rdcs_gpu(
            &engine,
            &c.noe_confs,
            &seq,
            &seq_ids,
            &exp_adrs,
            &fw_dihs,
            &c.media,
            0.55,
            110,
        )
        .expect("F8 cold WebGPU RDC refinement");
        let gpu_nh_val = evaluate_rdc_validation(&gpu_confs[0].chain, &c.nh_media)
            .expect("WebGPU RDC 1D_NH eval");
        let gpu_rmsd_low =
            nmr_core::structure::backbone_rmsd_seq_range(&gpu_confs[0].chain, &xray_1zzk, 12, 89);
        let gpu_rmsd_mean = gpu_confs
            .iter()
            .map(|cf| nmr_core::structure::backbone_rmsd_seq_range(&cf.chain, &xray_1zzk, 12, 89))
            .sum::<f64>()
            / (gpu_confs.len() as f64);
        let gpu_sv = nmr_core::structure::validate_ensemble(&gpu_confs, &exp_adrs, &fw_dihs, None);
        let gpu_rama_pct = gpu_sv.ramachandran_favored_pct + gpu_sv.ramachandran_allowed_pct;

        assert!(
            c.cpu_q_work.is_finite()
                && c.cpu_plane_q_free.is_finite()
                && gpu_val.overall_q_work.is_finite()
                && gpu_val.overall_q_free.is_finite(),
            "non-finite Q in F8 seed {}",
            c.seed
        );

        println!(
            "[F8 COLD 1KHM->1ZZK seed={}] work={n_work} free={n_free} | NOE(Qw={:.4}, Qf={:.4}, NH_Qf={:.4}, lowE={:.3}A, mean={:.3}A) | CPU(Qw={:.4}, Qf={:.4}, NH_Qf={:.4}, lowE={:.3}A, mean={:.3}A, Rama={:.1}%) | GPU(Qw={:.4}, Qf={:.4}, NH_Qf={:.4}, lowE={:.3}A, mean={:.3}A, Rama={:.1}%)",
            c.seed,
            c.noe_q_work,
            c.noe_plane_q_free,
            c.noe_nh_q_free,
            c.noe_rmsd_low_a,
            c.noe_rmsd_mean_a,
            c.cpu_q_work,
            c.cpu_plane_q_free,
            c.cpu_nh_q_free,
            c.cpu_rmsd_low_a,
            c.cpu_rmsd_mean_a,
            c.cpu_rama_pct,
            gpu_val.overall_q_work,
            gpu_val.overall_q_free,
            gpu_nh_val.overall_q_free,
            gpu_rmsd_low,
            gpu_rmsd_mean,
            gpu_rama_pct
        );

        full_rows.push(F8FullRow {
            seed: c.seed,
            n_work,
            n_free,
            noe_q_work: c.noe_q_work,
            noe_plane_q_free: c.noe_plane_q_free,
            noe_nh_q_free: c.noe_nh_q_free,
            noe_rmsd_low_a: c.noe_rmsd_low_a,
            noe_rmsd_mean_a: c.noe_rmsd_mean_a,
            cpu_q_work: c.cpu_q_work,
            cpu_plane_q_free: c.cpu_plane_q_free,
            cpu_nh_q_free: c.cpu_nh_q_free,
            cpu_rmsd_low_a: c.cpu_rmsd_low_a,
            cpu_rmsd_mean_a: c.cpu_rmsd_mean_a,
            cpu_rama_pct: c.cpu_rama_pct,
            gpu_q_work: gpu_val.overall_q_work,
            gpu_plane_q_free: gpu_val.overall_q_free,
            gpu_nh_q_free: gpu_nh_val.overall_q_free,
            gpu_rmsd_low_a: gpu_rmsd_low,
            gpu_rmsd_mean_a: gpu_rmsd_mean,
            gpu_rama_pct,
        });
    }

    let min_max = |vals: &[f64]| -> (f64, f64) {
        let lo = vals.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (lo, hi)
    };
    let (noe_qf_lo, noe_qf_hi) = min_max(
        &full_rows
            .iter()
            .map(|r| r.noe_plane_q_free)
            .collect::<Vec<_>>(),
    );
    let (noe_rm_lo, noe_rm_hi) = min_max(
        &full_rows
            .iter()
            .map(|r| r.noe_rmsd_low_a)
            .collect::<Vec<_>>(),
    );
    let (cpu_qw_lo, cpu_qw_hi) =
        min_max(&full_rows.iter().map(|r| r.cpu_q_work).collect::<Vec<_>>());
    let (cpu_qf_lo, cpu_qf_hi) = min_max(
        &full_rows
            .iter()
            .map(|r| r.cpu_plane_q_free)
            .collect::<Vec<_>>(),
    );
    let (cpu_nh_qf_lo, cpu_nh_qf_hi) = min_max(
        &full_rows
            .iter()
            .map(|r| r.cpu_nh_q_free)
            .collect::<Vec<_>>(),
    );
    let (cpu_rm_lo, cpu_rm_hi) = min_max(
        &full_rows
            .iter()
            .map(|r| r.cpu_rmsd_low_a)
            .collect::<Vec<_>>(),
    );
    let (cpu_mean_lo, cpu_mean_hi) = min_max(
        &full_rows
            .iter()
            .map(|r| r.cpu_rmsd_mean_a)
            .collect::<Vec<_>>(),
    );
    let (gpu_qw_lo, gpu_qw_hi) =
        min_max(&full_rows.iter().map(|r| r.gpu_q_work).collect::<Vec<_>>());
    let (gpu_qf_lo, gpu_qf_hi) = min_max(
        &full_rows
            .iter()
            .map(|r| r.gpu_plane_q_free)
            .collect::<Vec<_>>(),
    );
    let (gpu_nh_qf_lo, gpu_nh_qf_hi) = min_max(
        &full_rows
            .iter()
            .map(|r| r.gpu_nh_q_free)
            .collect::<Vec<_>>(),
    );
    let (gpu_rm_lo, gpu_rm_hi) = min_max(
        &full_rows
            .iter()
            .map(|r| r.gpu_rmsd_low_a)
            .collect::<Vec<_>>(),
    );
    let (gpu_mean_lo, gpu_mean_hi) = min_max(
        &full_rows
            .iter()
            .map(|r| r.gpu_rmsd_mean_a)
            .collect::<Vec<_>>(),
    );

    let cpu_qfree_improved = full_rows
        .iter()
        .filter(|r| r.cpu_plane_q_free < r.noe_plane_q_free)
        .count();
    let gpu_qfree_improved = full_rows
        .iter()
        .filter(|r| r.gpu_plane_q_free < r.noe_plane_q_free)
        .count();
    let cpu_rmsd_improved = full_rows
        .iter()
        .filter(|r| r.cpu_rmsd_low_a < r.noe_rmsd_low_a)
        .count();
    let gpu_rmsd_improved = full_rows
        .iter()
        .filter(|r| r.gpu_rmsd_low_a < r.noe_rmsd_low_a)
        .count();

    println!(
        "[F8 COLD 5SEED SUMMARY 1KHM->1ZZK] NOE plane_Q_free={noe_qf_lo:.4}..{noe_qf_hi:.4}, lowE={noe_rm_lo:.3}..{noe_rm_hi:.3}A | CPU Q_work={cpu_qw_lo:.4}..{cpu_qw_hi:.4}, plane_Q_free={cpu_qf_lo:.4}..{cpu_qf_hi:.4} (NH {cpu_nh_qf_lo:.4}..{cpu_nh_qf_hi:.4}), lowE={cpu_rm_lo:.3}..{cpu_rm_hi:.3}A (Qf_improved={cpu_qfree_improved}/5, RMSD_improved={cpu_rmsd_improved}/5) | GPU Q_work={gpu_qw_lo:.4}..{gpu_qw_hi:.4}, plane_Q_free={gpu_qf_lo:.4}..{gpu_qf_hi:.4} (NH {gpu_nh_qf_lo:.4}..{gpu_nh_qf_hi:.4}), lowE={gpu_rm_lo:.3}..{gpu_rm_hi:.3}A (Qf_improved={gpu_qfree_improved}/5, RMSD_improved={gpu_rmsd_improved}/5)"
    );

    let seed_json: Vec<String> = full_rows
        .iter()
        .map(|r| {
            format!(
                "{{\"seed\":{},\"n_work\":{},\"n_free\":{},\"noe_q_work\":{:.6},\"noe_plane_q_free\":{:.6},\"noe_nh_q_free\":{:.6},\"noe_rmsd_low_a\":{:.6},\"noe_rmsd_mean_a\":{:.6},\"cpu_q_work\":{:.6},\"cpu_plane_q_free\":{:.6},\"cpu_nh_q_free\":{:.6},\"cpu_rmsd_low_a\":{:.6},\"cpu_rmsd_mean_a\":{:.6},\"cpu_rama_pct\":{:.3},\"gpu_q_work\":{:.6},\"gpu_plane_q_free\":{:.6},\"gpu_nh_q_free\":{:.6},\"gpu_rmsd_low_a\":{:.6},\"gpu_rmsd_mean_a\":{:.6},\"gpu_rama_pct\":{:.3}}}",
                r.seed,
                r.n_work,
                r.n_free,
                r.noe_q_work,
                r.noe_plane_q_free,
                r.noe_nh_q_free,
                r.noe_rmsd_low_a,
                r.noe_rmsd_mean_a,
                r.cpu_q_work,
                r.cpu_plane_q_free,
                r.cpu_nh_q_free,
                r.cpu_rmsd_low_a,
                r.cpu_rmsd_mean_a,
                r.cpu_rama_pct,
                r.gpu_q_work,
                r.gpu_plane_q_free,
                r.gpu_nh_q_free,
                r.gpu_rmsd_low_a,
                r.gpu_rmsd_mean_a,
                r.gpu_rama_pct
            )
        })
        .collect();

    let json_payload = format!(
        "{{\n  \"kind\": \"f8_cold_holdout_rdc\",\n  \"nmr_pdb_id\": \"1KHM\",\n  \"bmrb_id\": \"4405\",\n  \"xray_pdb_id\": \"1ZZK\",\n  \"xray_resolution_a\": 0.95,\n  \"ordered_seq_range\": [12, 89],\n  \"n_exp_noe_adrs\": {},\n  \"n_firewall_dihedrals\": {},\n  \"max_retained_seq_identity\": {:.6},\n  \"xray_direct_q_work\": {:.6},\n  \"xray_direct_plane_q_free_seed4242\": {:.6},\n  \"xray_direct_nh_q_work\": {:.6},\n  \"xray_direct_nh_plane_q_free_seed4242\": {:.6},\n  \"adapter\": \"{}\",\n  \"backend\": \"{}\",\n  \"noe_plane_q_free_min\": {:.6},\n  \"noe_plane_q_free_max\": {:.6},\n  \"noe_rmsd_low_a_min\": {:.6},\n  \"noe_rmsd_low_a_max\": {:.6},\n  \"cpu_q_work_min\": {:.6},\n  \"cpu_q_work_max\": {:.6},\n  \"cpu_plane_q_free_min\": {:.6},\n  \"cpu_plane_q_free_max\": {:.6},\n  \"cpu_nh_q_free_min\": {:.6},\n  \"cpu_nh_q_free_max\": {:.6},\n  \"cpu_rmsd_low_a_min\": {:.6},\n  \"cpu_rmsd_low_a_max\": {:.6},\n  \"cpu_rmsd_mean_a_min\": {:.6},\n  \"cpu_rmsd_mean_a_max\": {:.6},\n  \"cpu_qfree_improved_count\": {},\n  \"cpu_rmsd_improved_count\": {},\n  \"gpu_q_work_min\": {:.6},\n  \"gpu_q_work_max\": {:.6},\n  \"gpu_plane_q_free_min\": {:.6},\n  \"gpu_plane_q_free_max\": {:.6},\n  \"gpu_nh_q_free_min\": {:.6},\n  \"gpu_nh_q_free_max\": {:.6},\n  \"gpu_rmsd_low_a_min\": {:.6},\n  \"gpu_rmsd_low_a_max\": {:.6},\n  \"gpu_rmsd_mean_a_min\": {:.6},\n  \"gpu_rmsd_mean_a_max\": {:.6},\n  \"gpu_qfree_improved_count\": {},\n  \"gpu_rmsd_improved_count\": {},\n  \"seeds\": [\n    {}\n  ]\n}}\n",
        exp_adrs.len(),
        fw_dihs.len(),
        kh3_pred.max_retained_seq_identity,
        xray_multi_val.overall_q_work,
        xray_multi_val.overall_q_free,
        xray_nh_val.overall_q_work,
        xray_nh_val.overall_q_free,
        engine.adapter_name.replace('"', "'"),
        engine.backend_name.replace('"', "'"),
        noe_qf_lo,
        noe_qf_hi,
        noe_rm_lo,
        noe_rm_hi,
        cpu_qw_lo,
        cpu_qw_hi,
        cpu_qf_lo,
        cpu_qf_hi,
        cpu_nh_qf_lo,
        cpu_nh_qf_hi,
        cpu_rm_lo,
        cpu_rm_hi,
        cpu_mean_lo,
        cpu_mean_hi,
        cpu_qfree_improved,
        cpu_rmsd_improved,
        gpu_qw_lo,
        gpu_qw_hi,
        gpu_qf_lo,
        gpu_qf_hi,
        gpu_nh_qf_lo,
        gpu_nh_qf_hi,
        gpu_rm_lo,
        gpu_rm_hi,
        gpu_mean_lo,
        gpu_mean_hi,
        gpu_qfree_improved,
        gpu_rmsd_improved,
        seed_json.join(",\n    ")
    );
    println!("W4_F8_JSON:{}", json_payload.replace('\n', ""));
    if let Ok(dir) = std::env::var("NMR_W4_RESULTS_DIR") {
        let out_p = PathBuf::from(dir).join("w4_f8_cold_holdout_1khm_1zzk.json");
        std::fs::write(out_p, json_payload).expect("write w4_f8_cold_holdout_1khm_1zzk.json");
    }
}
