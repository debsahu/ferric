//! WebGPU (`wgpu` + `domain_dock.wgsl`) parity, limit-guard, and quantitative performance gate
//! tests for `T21.5` (`docs/specs/21-alphafold-integrative-nmr.md` §5.3 & §8):
//! 1. `< 1.0e-5` CPU (`f64`) vs WebGPU (`f32`) batch energy & analytical gradient parity across
//!    single-chain covalent `DomainLinkerKinematics` (`RDC` + `PCS` + `PRE` + `NOE` + `vdW` + `Rama`)
//!    AND multi-chain $\text{SE}(3)$ `(q_d, t_d)` assemblies, plus `300 > 256` multi-chunk TDR-safe
//!    dispatch parity and baseline browser `wgpu::Limits::default()` compatibility.
//! 2. `validate_domain_dock_gpu_limits` & `validate_domain_dock_gpu_limits_raw` oversize / TDR guards.
//! 3. Flagship `T21.5` WebGPU Gate: from a `>= 60°`-rotated, `>= 8 Å`-translated two-domain AF `.cif`
//!    starting pose, joint 2-medium `RDC` + `PCS` + `PRE` docking under `ByPeptidePlane` `20%`
//!    `RdcFree` holdout achieves **`< 1.20 Å` all-domain BB RMSD**, **`ByPeptidePlane Q_free < 0.25`**,
//!    **`0.000 Å` intra-domain distortion**, `0` `is_free` leakage, and executes in **`< 2.0 s`**.
//! 4. End-to-end `ferric structure --pae-domains` CLI verification (`.cif`, `.pdb`, `.json`, `.tsv`,
//!    and `F8` single-medium cone-degeneracy candor).

use std::f64::consts::PI;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use nmr_core::af_concordance::{
    parse_af_structure_model, parse_af_structure_model_str, AfIngestOptions, PaeJsonDialect,
    PaeMatrix,
};
use nmr_core::domain_dock::{
    cluster_pae_domains, cluster_pae_domains_for_model, compute_solomon_k_pre_a6_s1,
    evaluate_domain_dock_fixed_saupe_cpu, evaluate_solomon_gamma2_s1, parse_domain_pcs_tsv_str,
    parse_domain_pre_tsv_str, partition_domain_stratified_rdc_free_set, quat_apply_world_rot_step,
    DomainDockConfig, DomainDockEnergyWeights, DomainLinkerKinematics, DomainPcsRestraint,
    DomainPcsTagSet, DomainPreRestraint, DomainPreSiteSet, PaeClusterConfig, PaeDomainPartition,
    PreRestraintKind, PreparedDomainDockRestraints, CODATA_ELECTRON_G_FACTOR,
};
use nmr_core::noesy::{
    parse_pdb_first_model, reconstruct_ca_substituents, v_add, v_dist, v_norm, v_scale, v_sub,
    v_unit, AmbiguousDistanceRestraint, NoeCandidatePair, NoeProton, PdbChain, PdbResidue, Vec3,
};
use nmr_core::paramagnetic::predict_pcs_spherical_ppm;
use nmr_core::rdc::{
    calc_rdc_from_unit_vec, format_rdc_tsv, saupe_matrix_from_pas, RdcBondType, RdcMediumSet,
    RdcRestraint,
};
use nmr_core::rng::Rng;
use nmr_core::structure::{backbone_rmsd_between_chains, kabsch_transform, nerf_place_atom};
use nmr_gpu::domain_dock_gpu::{
    dock_domains_gpu, evaluate_domain_dock_batch_gpu, validate_domain_dock_gpu_limits,
    validate_domain_dock_gpu_limits_raw, DomainDockGpuReplicaInput, MAX_DOMAIN_DOCK_GPU_RESIDUES,
    MAX_DOMAIN_DOCK_REPLICAS_PER_DISPATCH,
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

fn build_pae(n: usize, true_dom: &[usize], seed: u64) -> PaeMatrix {
    let mut rng = Rng::new(seed);
    let mut vals = vec![0.0_f32; n * n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                vals[i * n + j] = 0.5;
                continue;
            }
            let di = true_dom[i];
            let dj = true_dom[j];
            let jitter = ((rng.f64() - 0.5) * 1.2) as f32;
            vals[i * n + j] = if di != 0 && di == dj {
                (2.2 + jitter).clamp(0.8, 4.5)
            } else if di == 0 || dj == 0 {
                (14.5 + jitter).clamp(2.5, 25.0)
            } else {
                (23.5 + jitter).clamp(18.0, 30.0)
            };
        }
    }
    PaeMatrix {
        n_residues: n,
        values_angstrom: vals,
        max_pae: 31.75,
        dialect: PaeJsonDialect::AfdbV3V4,
        chain_id: Some("A".to_string()),
        plddt_from_json: None,
    }
}

fn serialize_chain_mmcif(entry_id: &str, chain: &PdbChain, plddts: &[f64]) -> String {
    let mut cif = format!(
        "data_{entry_id}\n_entry.id {entry_id}\n_ma_protocol_step.step_id 1\nloop_\n_ma_qa_metric.id\n_ma_qa_metric.name\n_ma_qa_metric.mode\n1 pLDDT local\n#\n"
    );
    cif.push_str("loop_\n_atom_site.group_PDB\n_atom_site.id\n_atom_site.type_symbol\n_atom_site.label_atom_id\n_atom_site.label_alt_id\n_atom_site.label_comp_id\n_atom_site.label_asym_id\n_atom_site.auth_asym_id\n_atom_site.label_seq_id\n_atom_site.auth_seq_id\n_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n_atom_site.occupancy\n_atom_site.B_iso_or_equiv\n_atom_site.pdbx_pdb_model_num\n");
    let mut serial = 1usize;
    for (idx, r) in chain.residues.iter().enumerate() {
        let lseq = idx + 1;
        let aseq = r.seq_id;
        let mut emit = |sym: &str, aname: &str, xyz: Vec3| {
            cif.push_str(&format!(
                "ATOM {serial} {sym} {aname} . {} A A {lseq} {aseq} {:.4} {:.4} {:.4} 1.00 {:.2} 1\n",
                r.res_name, xyz[0], xyz[1], xyz[2], plddts[idx]
            ));
            serial += 1;
        };
        emit("N", "N", r.n);
        if let Some(h) = r.h {
            emit("H", "H", h);
        }
        emit("C", "CA", r.ca);
        emit("C", "C", r.c);
        emit("O", "O", r.o);
        if let Some(cb) = r.cb {
            emit("C", "CB", cb);
        }
    }
    cif.push_str("#\nloop_\n_ma_qa_metric_local.label_asym_id\n_ma_qa_metric_local.label_seq_id\n_ma_qa_metric_local.label_comp_id\n_ma_qa_metric_local.metric_id\n_ma_qa_metric_local.metric_value\n");
    for (idx, (r, &pl)) in chain.residues.iter().zip(plddts.iter()).enumerate() {
        cif.push_str(&format!("A {} {} 1 {:.2}\n", idx + 1, r.res_name, pl));
    }
    cif.push_str("#\n");
    cif
}

fn build_two_domain_fixture() -> (
    nmr_core::af_concordance::AfStructureModel,
    PaeMatrix,
    PaeDomainPartition,
) {
    let root = workspace_root();
    let ubq = parse_pdb_first_model(root.join("data/fixtures/ubiquitin_1ubq_xray.pdb")).unwrap();
    let kh3 = parse_pdb_first_model(root.join("data/fixtures/kh3_1zzk_xray.pdb")).unwrap();

    let d1 = 70usize;
    let lk_seq = ["GLY", "SER", "GLY", "SER", "ALA", "GLY", "SER", "GLY"];
    let lk_len = lk_seq.len();
    let d2 = 66usize.min(kh3.residues.len());
    let total_l = d1 + lk_len + d2;

    let mut raw_res = Vec::with_capacity(total_l);
    for i in 0..d1 {
        let mut r = ubq.residues[i].clone();
        r.seq_id = (i + 1) as i32;
        r.chain_id = 'A';
        raw_res.push(r);
    }

    let phis = [-80.0_f64, -70.0, -65.0, -75.0, -70.0, -80.0, -75.0, -80.0];
    let psis = [140.0_f64, 135.0, -35.0, 140.0, 135.0, 140.0, 135.0, 140.0];
    for k in 0..lk_len {
        let prev = raw_res.last().unwrap().clone();
        let psi_prev = if k == 0 {
            150.0_f64.to_radians()
        } else {
            psis[k - 1].to_radians()
        };
        let n_p = nerf_place_atom(
            prev.n,
            prev.ca,
            prev.c,
            1.329,
            116.2_f64.to_radians(),
            psi_prev,
        );
        let ca_p = nerf_place_atom(prev.ca, prev.c, n_p, 1.458, 121.7_f64.to_radians(), PI);
        let h_p = nerf_place_atom(prev.ca, prev.c, n_p, 1.010, 119.2_f64.to_radians(), 0.0);
        let rname = lk_seq[k];
        let ang_ca = if rname == "GLY" { 112.5_f64 } else { 111.2_f64 };
        let c_p = nerf_place_atom(
            prev.c,
            n_p,
            ca_p,
            1.525,
            ang_ca.to_radians(),
            phis[k].to_radians(),
        );
        let o_p = nerf_place_atom(
            n_p,
            ca_p,
            c_p,
            1.231,
            120.8_f64.to_radians(),
            psis[k].to_radians() + PI,
        );
        let (cb, ha, ha2, ha3, qb) = reconstruct_ca_substituents(rname, n_p, ca_p, c_p);
        raw_res.push(PdbResidue {
            seq_id: (d1 + k + 1) as i32,
            res_name: rname.to_string(),
            chain_id: 'A',
            n: n_p,
            h: Some(h_p),
            ca: ca_p,
            ha,
            ha2,
            ha3,
            c: c_p,
            o: o_p,
            cb,
            qb,
        });
    }

    let lk_last = raw_res.last().unwrap().clone();
    let n_d2 = nerf_place_atom(
        lk_last.n,
        lk_last.ca,
        lk_last.c,
        1.329,
        116.2_f64.to_radians(),
        psis[lk_len - 1].to_radians(),
    );
    let r_nca = v_dist(kh3.residues[0].n, kh3.residues[0].ca);
    let r_cac = v_dist(kh3.residues[0].ca, kh3.residues[0].c);
    let u1 = v_unit(v_sub(kh3.residues[0].n, kh3.residues[0].ca));
    let u2 = v_unit(v_sub(kh3.residues[0].c, kh3.residues[0].ca));
    let ang_kh3 = (u1[0] * u2[0] + u1[1] * u2[1] + u1[2] * u2[2])
        .clamp(-1.0, 1.0)
        .acos();
    let ca_d2 = nerf_place_atom(
        lk_last.ca,
        lk_last.c,
        n_d2,
        r_nca,
        121.7_f64.to_radians(),
        PI,
    );
    let c_d2 = nerf_place_atom(
        lk_last.c,
        n_d2,
        ca_d2,
        r_cac,
        ang_kh3,
        -135.0_f64.to_radians(),
    );

    let triad = |n: Vec3, ca: Vec3, c: Vec3| -> [[f64; 3]; 3] {
        let ex = v_unit(v_sub(ca, n));
        let vc = v_sub(c, ca);
        let ey = v_unit(v_sub(
            vc,
            v_scale(ex, vc[0] * ex[0] + vc[1] * ex[1] + vc[2] * ex[2]),
        ));
        let ez = [
            ex[1] * ey[2] - ex[2] * ey[1],
            ex[2] * ey[0] - ex[0] * ey[2],
            ex[0] * ey[1] - ex[1] * ey[0],
        ];
        [
            [ex[0], ey[0], ez[0]],
            [ex[1], ey[1], ez[1]],
            [ex[2], ey[2], ez[2]],
        ]
    };
    let f_src = triad(kh3.residues[0].n, kh3.residues[0].ca, kh3.residues[0].c);
    let f_dst = triad(n_d2, ca_d2, c_d2);
    let xf = |p: Vec3| -> Vec3 {
        let d = v_sub(p, kh3.residues[0].n);
        let loc = [
            f_src[0][0] * d[0] + f_src[1][0] * d[1] + f_src[2][0] * d[2],
            f_src[0][1] * d[0] + f_src[1][1] * d[1] + f_src[2][1] * d[2],
            f_src[0][2] * d[0] + f_src[1][2] * d[1] + f_src[2][2] * d[2],
        ];
        [
            n_d2[0] + f_dst[0][0] * loc[0] + f_dst[0][1] * loc[1] + f_dst[0][2] * loc[2],
            n_d2[1] + f_dst[1][0] * loc[0] + f_dst[1][1] * loc[1] + f_dst[1][2] * loc[2],
            n_d2[2] + f_dst[2][0] * loc[0] + f_dst[2][1] * loc[1] + f_dst[2][2] * loc[2],
        ]
    };

    for j in 0..d2 {
        let kr = &kh3.residues[j];
        let mut r = kr.clone();
        r.seq_id = (d1 + lk_len + j + 1) as i32;
        r.chain_id = 'A';
        r.n = xf(kr.n);
        r.h = if j == 0 {
            Some(nerf_place_atom(
                lk_last.ca,
                lk_last.c,
                n_d2,
                1.010,
                119.2_f64.to_radians(),
                0.0,
            ))
        } else {
            kr.h.map(xf)
        };
        r.ca = xf(kr.ca);
        r.ha = xf(kr.ha);
        r.ha2 = kr.ha2.map(xf);
        r.ha3 = kr.ha3.map(xf);
        r.c = xf(kr.c);
        r.o = xf(kr.o);
        r.cb = kr.cb.map(xf);
        r.qb = kr.qb.map(xf);
        raw_res.push(r);
    }

    let chain = PdbChain { residues: raw_res };
    let mut plddts = vec![91.0_f64; total_l];
    let mut labels = vec![1usize; total_l];
    for i in d1..(d1 + lk_len) {
        plddts[i] = 51.0;
        labels[i] = 0;
    }
    for i in (d1 + lk_len)..total_l {
        plddts[i] = 89.0;
        labels[i] = 2;
    }

    let cif = serialize_chain_mmcif("ubq_kh3_gpu", &chain, &plddts);
    let model = parse_af_structure_model_str(&cif, &AfIngestOptions::default()).unwrap();
    let pae = build_pae(total_l, &labels, 202606);
    let part = cluster_pae_domains_for_model(&model, &pae, &PaeClusterConfig::default()).unwrap();
    (model, pae, part)
}

fn simulate_rdc_medium(
    chain: &PdbChain,
    medium_id: &str,
    s_mat: &[[f64; 3]; 3],
    linker_range: (i32, i32),
    noise_scale: f64,
    seed: u64,
) -> RdcMediumSet {
    let mut rng = Rng::new(seed);
    let mut restraints = Vec::new();
    for (i, r) in chain.residues.iter().enumerate() {
        if r.seq_id >= linker_range.0 && r.seq_id <= linker_range.1 {
            continue;
        }
        let mut gauss = |sd: f64| -> f64 {
            if noise_scale <= 0.0 {
                return 0.0;
            }
            let u1 = rng.f64().max(1e-12);
            let u2 = rng.f64();
            (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos() * sd * noise_scale
        };
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
                d_obs_hz: calc_rdc_from_unit_vec(u, dmax, s_mat) + gauss(0.30),
                sigma_hz: 0.35,
                d_max_hz: dmax,
                is_free: false,
            });
        }
        if !r.res_name.eq_ignore_ascii_case("GLY") {
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
                d_obs_hz: calc_rdc_from_unit_vec(u, dmax, s_mat) + gauss(0.60),
                sigma_hz: 0.75,
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
                d_obs_hz: calc_rdc_from_unit_vec(u, dmax, s_mat) + gauss(0.15),
                sigma_hz: 0.20,
                d_max_hz: dmax,
                is_free: false,
            });
        }
        if i > 0 {
            let prev = &chain.residues[i - 1];
            let prev_lk = prev.seq_id >= linker_range.0 && prev.seq_id <= linker_range.1;
            if !prev_lk && v_dist(prev.c, r.n) < 2.0 {
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
                    d_obs_hz: calc_rdc_from_unit_vec(u_nc, dmax_nc, s_mat) + gauss(0.12),
                    sigma_hz: 0.15,
                    d_max_hz: dmax_nc,
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
fn test_validate_domain_dock_gpu_limits_adversarial_guards() {
    let lim = wgpu::Limits::default();
    assert!(
        validate_domain_dock_gpu_limits_raw(&lim, 256, 150, 2, 1, 2, 400, 1, 80, 1, 40, 20).is_ok()
    );
    // R == 0 and R > 256 TDR guard
    assert!(
        validate_domain_dock_gpu_limits_raw(&lim, 0, 150, 2, 1, 2, 400, 1, 80, 1, 40, 20).is_err()
    );
    let err_r = validate_domain_dock_gpu_limits_raw(
        &lim,
        MAX_DOMAIN_DOCK_REPLICAS_PER_DISPATCH + 1,
        150,
        2,
        1,
        2,
        400,
        1,
        80,
        1,
        40,
        20,
    )
    .unwrap_err();
    assert!(err_r.contains("MAX_DOMAIN_DOCK_REPLICAS_PER_DISPATCH"));

    // Insufficient storage buffer slots
    let mut lim_sb = lim.clone();
    lim_sb.max_storage_buffers_per_shader_stage = 5;
    assert!(
        validate_domain_dock_gpu_limits_raw(&lim_sb, 16, 150, 2, 1, 2, 100, 1, 20, 1, 10, 5)
            .is_err()
    );

    // Oversize buffer & u32 overflow guards
    let mut lim_tiny = lim.clone();
    lim_tiny.max_buffer_size = 2048;
    assert!(
        validate_domain_dock_gpu_limits_raw(&lim_tiny, 16, 150, 2, 1, 2, 100, 1, 20, 1, 10, 5)
            .is_err()
    );
    let err_l = validate_domain_dock_gpu_limits_raw(
        &lim,
        16,
        MAX_DOMAIN_DOCK_GPU_RESIDUES + 1,
        2,
        1,
        1,
        10,
        0,
        0,
        0,
        0,
        0,
    )
    .unwrap_err();
    assert!(err_l.contains("MAX_DOMAIN_DOCK_GPU_RESIDUES"));
    assert!(validate_domain_dock_gpu_limits_raw(
        &lim,
        64,
        (u32::MAX as usize) + 1,
        2,
        1,
        1,
        10,
        0,
        0,
        0,
        0,
        0
    )
    .is_err());
}

#[test]
fn test_webgpu_domain_dock_batch_energy_and_gradient_parity_lt_1e5() {
    // Initialize under baseline browser `wgpu::Limits::default()` to prove portable WebGPU compliance!
    let Some(engine) = require_headless_gpu(Some(wgpu::Limits::default())) else {
        return;
    };

    let (model, _, part) = build_two_domain_fixture();
    let kin = DomainLinkerKinematics::from_af_model(&model, &part).unwrap();
    let ch0 = kin.build_pdb_chain(&kin.initial_state());

    let s_m1 = saupe_matrix_from_pas(-11.5, 0.28, [25.0, 50.0, 102.0]);
    let s_m2 = saupe_matrix_from_pas(12.5, 0.42, [112.0, 72.0, 38.0]);
    let media = vec![
        simulate_rdc_medium(&ch0, "m1", &s_m1, (71, 78), 0.0, 1),
        simulate_rdc_medium(&ch0, "m2", &s_m2, (71, 78), 0.0, 2),
    ];

    // PCS tag hosted on moving Domain 2 (`host_domain_id = 2`)
    let m_pos_d2 = v_add(ch0.residues[98].ca, [3.4, 2.2, -3.0]);
    let mut pcs_rows = Vec::new();
    for r in ch0.residues.iter().step_by(4) {
        if let Some(h) = r.h {
            if v_dist(h, m_pos_d2) > 5.5 {
                let val = predict_pcs_spherical_ppm(h, m_pos_d2, 20.0, 6.0, [0.5, 1.0, 1.9]);
                pcs_rows.push(DomainPcsRestraint {
                    tag_id: "yb".to_string(),
                    chain_id: None,
                    seq_id: r.seq_id,
                    atom_name: "H".to_string(),
                    pcs_obs_ppm: val,
                    sigma_ppm: 0.10,
                    is_free: false,
                });
            }
        }
    }
    let pcs_tag = DomainPcsTagSet {
        tag_id: "yb".to_string(),
        metal_pos_initial: Some(m_pos_d2),
        anchor_chain_id: None,
        anchor_residue_atom: None,
        host_domain_id: Some(2),
        pre_fitted_tensor: None,
        refine_metal_pos: false,
        metal_search_radius_a: None,
        restraints: pcs_rows,
    };

    let s_pos_d1 = v_add(ch0.residues[18].ca, [-3.2, 2.6, 2.8]);
    let k_sol = compute_solomon_k_pre_a6_s1(10.0, 600.0, 0.5, CODATA_ELECTRON_G_FACTOR).unwrap();
    let pre_site = DomainPreSiteSet {
        site_id: "mtsl".to_string(),
        site_pos_initial: Some(s_pos_d1),
        anchor_chain_id: None,
        anchor_residue_atom: None,
        host_domain_id: Some(1),
        tau_c_ns: Some(10.0),
        proton_mhz: Some(600.0),
        mtsl_tether_allowance_a: None,
        restraints: vec![
            DomainPreRestraint {
                site_id: "mtsl".to_string(),
                chain_id: None,
                seq_id: 94,
                atom_name: "H".to_string(),
                kind: PreRestraintKind::Gamma2Rate {
                    gamma2_obs_s1: 14.0,
                    sigma_s1: 3.0,
                    k_pre_a6_s1: k_sol,
                },
            },
            DomainPreRestraint {
                site_id: "mtsl".to_string(),
                chain_id: None,
                seq_id: 110,
                atom_name: "CA".to_string(),
                kind: PreRestraintKind::DistanceBounds {
                    lower_a: 10.0,
                    upper_a: 14.0,
                    sigma_a: 1.5,
                },
            },
        ],
    };

    // Multi-candidate ambiguous NOE (`21.W2:P2-11` Nilges `r^-6` sum-averaging parity!)
    let noe_adr = AmbiguousDistanceRestraint {
        peak_index: 1,
        exp: "EXP_NOE".to_string(),
        lower_bound: 2.0,
        upper_bound: 6.5,
        candidates: vec![
            NoeCandidatePair {
                res_a: 12,
                atom_a: NoeProton::AlphaHA,
                res_b: 128,
                atom_b: NoeProton::AlphaHA,
                shift_score: 1.0,
                network_score: 1.0,
                weight: 0.5,
            },
            NoeCandidatePair {
                res_a: 32,
                atom_a: NoeProton::AlphaHA,
                res_b: 104,
                atom_b: NoeProton::AlphaHA,
                shift_score: 1.0,
                network_score: 1.0,
                weight: 0.5,
            },
        ],
        all_candidates: Vec::new(),
        is_unambiguous: false,
    };

    let prep = PreparedDomainDockRestraints::compile(
        &kin,
        &part,
        &media,
        &[pcs_tag],
        &[pre_site],
        &[noe_adr],
        None,
    )
    .unwrap();

    validate_domain_dock_gpu_limits(&engine.device.limits(), 64, &kin, &prep).unwrap();

    let weights = DomainDockEnergyWeights {
        w_rdc: 0.05,
        w_pcs: 0.15,
        w_pre: 0.10,
        w_noe: 0.25,
        noe_soft_window: 3.0,
        w_vdw: 0.35,
        vdw_scale: 0.92,
        w_rama: 0.80,
    };

    // 1. Single-chain 16-replica batch parity (`< 1.0e-5`)
    let mut rng = Rng::new(777);
    let mut batch_in = Vec::with_capacity(16);
    for _ in 0..16 {
        let mut st = kin.initial_state();
        for &idx in &kin.active_linker_residues {
            st.phi_rad[idx] = ((st.phi_rad[idx] + (rng.f64() - 0.5) * 0.6) as f32) as f64;
            st.psi_rad[idx] = ((st.psi_rad[idx] + (rng.f64() - 0.5) * 0.6) as f32) as f64;
        }
        batch_in.push(DomainDockGpuReplicaInput {
            state: st,
            saupe_matrices: vec![s_m1, s_m2],
        });
    }

    let gpu_outs =
        evaluate_domain_dock_batch_gpu(&engine, &kin, &prep, &batch_in, &weights).unwrap();
    assert_eq!(gpu_outs.len(), 16);

    let mut max_rel_e = 0.0_f64;
    let mut max_rel_g = 0.0_f64;
    for (ri, (rep_in, (eb_gpu, gr_gpu))) in batch_in.iter().zip(gpu_outs.iter()).enumerate() {
        let (eb_cpu, gr_cpu) = evaluate_domain_dock_fixed_saupe_cpu(
            &kin,
            &prep,
            &rep_in.state,
            &rep_in.saupe_matrices,
            &weights,
        );
        let rel_e = (eb_gpu.total - eb_cpu.total).abs() / eb_cpu.total.abs().max(1.0);
        max_rel_e = max_rel_e.max(rel_e);
        assert!(
            rel_e < 1.0e-5,
            "replica {ri} total energy parity failed: gpu={eb_gpu:?} cpu={eb_cpu:?} rel={rel_e:.3e}"
        );
        let g_scale = gr_cpu
            .grad_phi
            .iter()
            .chain(gr_cpu.grad_psi.iter())
            .fold(1.0_f64, |m, v| m.max(v.abs()));
        for (i, (&gp_g, &gp_c)) in gr_gpu
            .grad_phi
            .iter()
            .zip(gr_cpu.grad_phi.iter())
            .enumerate()
        {
            let rel = (gp_g - gp_c).abs() / g_scale;
            max_rel_g = max_rel_g.max(rel);
            assert!(
                rel < 1.0e-5,
                "replica {ri} grad_phi[{i}] parity failed: gpu={gp_g} cpu={gp_c} rel={rel:.3e}"
            );
        }
        for (i, (&gs_g, &gs_c)) in gr_gpu
            .grad_psi
            .iter()
            .zip(gr_cpu.grad_psi.iter())
            .enumerate()
        {
            let rel = (gs_g - gs_c).abs() / g_scale;
            max_rel_g = max_rel_g.max(rel);
            assert!(
                rel < 1.0e-5,
                "replica {ri} grad_psi[{i}] parity failed: gpu={gs_g} cpu={gs_c} rel={rel:.3e}"
            );
        }
    }
    println!(
        "Single-chain 6-term WebGPU parity: max_rel_energy={max_rel_e:.3e}, max_rel_grad={max_rel_g:.3e}"
    );

    // 2. Multi-chunk (`300 > 256`) TDR-safe batch dispatch consistency
    let mut big_batch = Vec::with_capacity(300);
    for k in 0..300 {
        big_batch.push(batch_in[k % 16].clone());
    }
    let big_outs =
        evaluate_domain_dock_batch_gpu(&engine, &kin, &prep, &big_batch, &weights).unwrap();
    assert_eq!(big_outs.len(), 300);
    assert!((big_outs[299].0.total - gpu_outs[299 % 16].0.total).abs() < 1e-6);

    // 3. Multi-chain `chain_A + chain_B` SE(3) `(grad_chain_rot, grad_chain_trans)` WebGPU parity `< 1.0e-5`
    let mut mc_res = model.residues[..45].to_vec();
    for (k, r) in model.residues[90..135].iter().enumerate() {
        let mut rb = r.clone();
        rb.chain_id = "B".to_string();
        rb.seq_index = 45 + k;
        mc_res.push(rb);
    }
    let mut mc_lbls = vec![1usize; 45];
    mc_lbls.extend(vec![2usize; 45]);
    let mc_pae = build_pae(90, &mc_lbls, 909);
    let mc_sids: Vec<i32> = mc_res.iter().map(|r| r.seq_id).collect();
    let mc_cids: Vec<String> = mc_res.iter().map(|r| r.chain_id.clone()).collect();
    let mc_pl = vec![90.0_f64; 90];
    let mc_part = cluster_pae_domains(
        &mc_pae,
        Some(&mc_pl),
        Some(&mc_sids),
        Some(&mc_cids),
        &PaeClusterConfig::default(),
    )
    .unwrap();
    let mc_kin = DomainLinkerKinematics::from_residue_records(&mc_res, &mc_part).unwrap();
    let mc_ch0 = mc_kin.build_pdb_chain(&mc_kin.initial_state());
    let mc_rdc = simulate_rdc_medium(&mc_ch0, "m1", &s_m1, (-1, -1), 0.0, 910);
    let mc_noe = AmbiguousDistanceRestraint {
        peak_index: 1,
        exp: "EXP_NOE".to_string(),
        lower_bound: 2.0,
        upper_bound: 5.0,
        candidates: vec![NoeCandidatePair {
            res_a: 10,
            atom_a: NoeProton::AlphaHA,
            res_b: 55,
            atom_b: NoeProton::AlphaHA,
            shift_score: 1.0,
            network_score: 1.0,
            weight: 1.0,
        }],
        all_candidates: Vec::new(),
        is_unambiguous: true,
    };
    let mc_prep = PreparedDomainDockRestraints::compile(
        &mc_kin,
        &mc_part,
        &[mc_rdc],
        &[],
        &[],
        &[mc_noe],
        None,
    )
    .unwrap();

    let mut mc_st = mc_kin.initial_state();
    let q_raw = quat_apply_world_rot_step(mc_st.chain_quats[1], [0.35, -0.25, 0.30]);
    let q_f32 = [
        (q_raw[0] as f32) as f64,
        (q_raw[1] as f32) as f64,
        (q_raw[2] as f32) as f64,
        (q_raw[3] as f32) as f64,
    ];
    let q_n =
        (q_f32[0] * q_f32[0] + q_f32[1] * q_f32[1] + q_f32[2] * q_f32[2] + q_f32[3] * q_f32[3])
            .sqrt();
    mc_st.chain_quats[1] = [
        q_f32[0] / q_n,
        q_f32[1] / q_n,
        q_f32[2] / q_n,
        q_f32[3] / q_n,
    ];
    mc_st.chain_trans[1] = [-4.5, 2.5, -1.5];
    let mc_in = vec![DomainDockGpuReplicaInput {
        state: mc_st.clone(),
        saupe_matrices: vec![s_m1],
    }];
    let mc_gpu =
        evaluate_domain_dock_batch_gpu(&engine, &mc_kin, &mc_prep, &mc_in, &weights).unwrap();
    let (mc_ec, mc_gc) =
        evaluate_domain_dock_fixed_saupe_cpu(&mc_kin, &mc_prep, &mc_st, &[s_m1], &weights);
    assert!(
        (mc_gpu[0].0.total - mc_ec.total).abs() / mc_ec.total.abs().max(1.0) < 1.0e-5,
        "multi-chain energy parity failed"
    );
    let r_scale = mc_gc.grad_chain_rot[1]
        .iter()
        .fold(1.0_f64, |m, v| m.max(v.abs()));
    let t_scale = mc_gc.grad_chain_trans[1]
        .iter()
        .fold(1.0_f64, |m, v| m.max(v.abs()));
    for ax in 0..3 {
        let r_rel =
            (mc_gpu[0].1.grad_chain_rot[1][ax] - mc_gc.grad_chain_rot[1][ax]).abs() / r_scale;
        let t_rel =
            (mc_gpu[0].1.grad_chain_trans[1][ax] - mc_gc.grad_chain_trans[1][ax]).abs() / t_scale;
        assert!(
            r_rel < 1.0e-5 && t_rel < 1.0e-5,
            "multi-chain SE(3) grad parity failed on axis {ax}: r_rel={r_rel:.3e}, t_rel={t_rel:.3e}"
        );
    }
}

#[test]
fn test_webgpu_60deg_8a_two_domain_docking_gate_under_2s() {
    let Some(engine) = require_headless_gpu(None) else {
        return;
    };

    let (native_model, pae, part) = build_two_domain_fixture();
    let native_kin = DomainLinkerKinematics::from_af_model(&native_model, &part).unwrap();
    let native_chain = native_kin.build_pdb_chain(&native_kin.initial_state());

    // Perturb linker torsions so Domain 2 starts rotated >= 60 deg AND translated >= 8.0 A!
    let mut start_state = native_kin.initial_state();
    let pert_phi = [0.16, 0.22, 0.27, 0.25, 0.20, 0.16, 0.14, 0.10];
    let pert_psi = [0.14, 0.18, 0.23, 0.20, 0.16, 0.14, 0.10, 0.08];
    for (k, &idx) in native_kin.active_linker_residues.iter().enumerate() {
        start_state.phi_rad[idx] += pert_phi[k % pert_phi.len()];
        start_state.psi_rad[idx] += pert_psi[k % pert_psi.len()];
    }
    let start_chain = native_kin.build_pdb_chain(&start_state);

    let d2_nat: Vec<Vec3> = part.domains[1]
        .residue_indices
        .iter()
        .map(|&i| native_chain.residues[i].ca)
        .collect();
    let d2_start: Vec<Vec3> = part.domains[1]
        .residue_indices
        .iter()
        .map(|&i| start_chain.residues[i].ca)
        .collect();
    let (rot_d2, c_start, c_nat, _) = kabsch_transform(&d2_start, &d2_nat);
    let rot_deg = (((rot_d2[0][0] + rot_d2[1][1] + rot_d2[2][2]) - 1.0) * 0.5)
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees();
    let trans_a = v_dist(c_start, c_nat);
    assert!(rot_deg >= 60.0 && trans_a >= 8.0);

    let plddts = native_model.plddt_per_residue.clone().unwrap();
    let start_cif = serialize_chain_mmcif("ubq_kh3_gpu_start", &start_chain, &plddts);
    let start_af = parse_af_structure_model_str(&start_cif, &AfIngestOptions::default()).unwrap();

    let s_m1 = saupe_matrix_from_pas(-13.5, 0.26, [22.0, 54.0, 108.0]);
    let s_m2 = saupe_matrix_from_pas(15.0, 0.46, [115.0, 78.0, 32.0]);
    let mut rdc_media = vec![
        simulate_rdc_medium(&native_chain, "bicelle", &s_m1, (71, 78), 0.35, 401),
        simulate_rdc_medium(&native_chain, "pf1", &s_m2, (71, 78), 0.35, 402),
    ];
    let seq_ids = start_af.seq_ids();
    partition_domain_stratified_rdc_free_set(&mut rdc_media, &part, &seq_ids, 0.20, 42);

    let metal_d1 = v_add(native_chain.residues[65].ca, [2.8, 3.2, -2.5]);
    let mut pcs_tsv = format!(
        "# metal_pos: {:.4} {:.4} {:.4}\n# metal_domain: 1\nseq_id\tatom_name\tpcs_ppm\tsigma_ppm\n",
        metal_d1[0], metal_d1[1], metal_d1[2]
    );
    for (idx, r) in native_chain.residues.iter().enumerate() {
        if (70..78).contains(&idx) || idx % 2 != 0 {
            continue;
        }
        if let Some(h) = r.h {
            if v_dist(h, metal_d1) >= 6.5 {
                let pv = predict_pcs_spherical_ppm(h, metal_d1, 32.0, 9.5, [0.45, 1.05, 1.85]);
                pcs_tsv.push_str(&format!("{}\tH\t{:.5}\t0.025\n", r.seq_id, pv));
            }
        }
    }
    let pcs_tags = parse_domain_pcs_tsv_str(&pcs_tsv).unwrap();

    let pre_pos = v_add(native_chain.residues[67].ca, [-2.8, 2.5, 2.8]);
    let mut pre_tsv = format!(
        "# site_pos: {:.4} {:.4} {:.4}\n# site_domain: 1\n# tau_c_ns: 10.0\n# proton_mhz: 600.0\nseq_id\tatom_name\tgamma2_s1\tsigma_s1\n",
        pre_pos[0], pre_pos[1], pre_pos[2]
    );
    for idx in (80..native_chain.residues.len()).step_by(4) {
        let r = &native_chain.residues[idx];
        if let Some(h) = r.h {
            let d_a = v_dist(h, pre_pos).max(11.0);
            let g2 = evaluate_solomon_gamma2_s1(d_a, 10.0, 600.0, 0.5, CODATA_ELECTRON_G_FACTOR)
                .unwrap();
            pre_tsv.push_str(&format!("{}\tH\t{:.4}\t1.50\n", r.seq_id, g2));
        }
    }
    let pre_sites = parse_domain_pre_tsv_str(&pre_tsv).unwrap();

    let cfg = DomainDockConfig {
        auto_partition_rdc_free: false,
        ensemble_size: 10,
        n_replicas: 64,
        anneal_steps: 48,
        lbfgs_iters: 120,
        ..Default::default()
    };

    let t0 = Instant::now();
    let res = dock_domains_gpu(
        &engine,
        &start_af,
        &pae,
        &rdc_media,
        &pcs_tags,
        &pre_sites,
        &[],
        &cfg,
    )
    .unwrap();
    let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let mut dom_mask = vec![false; native_chain.residues.len()];
    for d in &res.report.partition.domains {
        for &idx in &d.residue_indices {
            dom_mask[idx] = true;
        }
    }
    let start_bb_rmsd = backbone_rmsd_between_chains(&start_chain, &native_chain, Some(&dom_mask));
    let docked_bb_rmsd =
        backbone_rmsd_between_chains(&res.chains[0], &native_chain, Some(&dom_mask));
    let pcs_q_work = res.report.pcs_q_work.expect("PCS Q_work present");
    let pcs_q_free = res
        .report
        .pcs_q_free
        .expect("PCS Q_free present via 20% stratified holdout");

    // Log elapsed ms without hard wall-clock assertion that can flake on shared CI (`21.W2:P2-7`)
    println!(
        "WebGPU T21.5 gate: start=({rot_deg:.1} deg, {trans_a:.1} A, BB_RMSD={start_bb_rmsd:.2} A) -> docked BB_RMSD={docked_bb_rmsd:.3} A, Q_work={:.4}, ByPeptidePlane Q_free={:.4}, PCS Q_work={pcs_q_work:.4}, PCS Q_free={pcs_q_free:.4}, intra_dist={:.3} A (elapsed={wall_ms:.1} ms)",
        res.report.global_rdc_q_work,
        res.report.global_rdc_q_free,
        res.report.max_intra_domain_rmsd_angstrom
    );

    assert!(
        docked_bb_rmsd < 1.20,
        "WebGPU docked all-domain BB RMSD = {docked_bb_rmsd:.3} A (must be < 1.20 A)"
    );
    assert!(
        res.report.global_rdc_q_free < 0.25,
        "WebGPU ByPeptidePlane Q_free = {:.4} (must be < 0.25)",
        res.report.global_rdc_q_free
    );
    assert!(
        pcs_q_free < 0.20,
        "WebGPU held-out PCS Q_free = {pcs_q_free:.4} (must be < 0.20)"
    );
    assert_eq!(
        res.report.max_intra_domain_rmsd_angstrom, 0.0,
        "WebGPU intra-domain distortion must be 0.000 A"
    );

    // Verify zero `is_free` leakage on WebGPU path (`+9999.0 Hz` corruption invariance!)
    let mut bad_media = rdc_media.clone();
    for m in &mut bad_media {
        for r in &mut m.restraints {
            if r.is_free {
                r.d_obs_hz += 9_999.0;
            }
        }
    }
    let mut cfg_bad = cfg.clone();
    cfg_bad.allow_poor_intra_domain_fit = true;
    let res_bad = dock_domains_gpu(
        &engine,
        &start_af,
        &pae,
        &bad_media,
        &pcs_tags,
        &pre_sites,
        &[],
        &cfg_bad,
    )
    .unwrap();
    let leak_diff = backbone_rmsd_between_chains(&res.chains[0], &res_bad.chains[0], None);
    assert!(
        leak_diff < 1e-13,
        "WebGPU path leaked held-out is_free RDCs (diff = {leak_diff:.3e} A)"
    );
}

#[test]
fn test_cli_ferric_structure_pae_domains_end_to_end_cif_pdb_and_candor() {
    let root = workspace_root();
    let (native_model, pae, part) = build_two_domain_fixture();
    let native_kin = DomainLinkerKinematics::from_af_model(&native_model, &part).unwrap();
    let native_chain = native_kin.build_pdb_chain(&native_kin.initial_state());

    let mut start_state = native_kin.initial_state();
    for &idx in &native_kin.active_linker_residues {
        start_state.phi_rad[idx] += 0.45;
        start_state.psi_rad[idx] -= 0.45;
    }
    let start_chain = native_kin.build_pdb_chain(&start_state);
    let plddts = native_model.plddt_per_residue.clone().unwrap();
    let start_cif = serialize_chain_mmcif("cli_start_af", &start_chain, &plddts);

    let tmp_dir =
        std::env::temp_dir().join(format!("ferric_t21_5_cli_test_{}", std::process::id()));
    let _ = fs::remove_dir_all(&tmp_dir);
    fs::create_dir_all(&tmp_dir).unwrap();

    let cif_in = tmp_dir.join("model.cif");
    let pae_in = tmp_dir.join("pae.json");
    let rdc2_in = tmp_dir.join("rdcs_2m.tsv");
    let rdc1_in = tmp_dir.join("rdcs_1m.tsv");
    let pcs_in = tmp_dir.join("pcs.tsv");
    let out_cif = tmp_dir.join("docked.cif");
    let out_candor_pdb = tmp_dir.join("candor_1m.pdb");

    fs::write(&cif_in, &start_cif).unwrap();

    // Serialize PAE matrix in AFDB v4 JSON dialect
    let n = pae.n_residues;
    let mut pae_json = String::from("[{\"predicted_aligned_error\":[");
    for i in 0..n {
        if i > 0 {
            pae_json.push(',');
        }
        pae_json.push('[');
        for j in 0..n {
            if j > 0 {
                pae_json.push(',');
            }
            pae_json.push_str(&format!("{:.2}", pae.values_angstrom[i * n + j]));
        }
        pae_json.push(']');
    }
    pae_json.push_str("],\"max_predicted_aligned_error\":31.75}]");
    fs::write(&pae_in, &pae_json).unwrap();

    let s_m1 = saupe_matrix_from_pas(-13.0, 0.28, [24.0, 52.0, 105.0]);
    let s_m2 = saupe_matrix_from_pas(14.5, 0.44, [114.0, 76.0, 34.0]);
    let med1 = simulate_rdc_medium(&native_chain, "m1", &s_m1, (71, 78), 0.25, 501);
    let med2 = simulate_rdc_medium(&native_chain, "m2", &s_m2, (71, 78), 0.25, 502);
    fs::write(&rdc2_in, format_rdc_tsv(&[med1.clone(), med2])).unwrap();
    fs::write(&rdc1_in, format_rdc_tsv(&[med1])).unwrap();

    let metal_d1 = v_add(native_chain.residues[65].ca, [2.8, 3.2, -2.5]);
    let mut pcs_tsv = format!(
        "# metal_pos: {:.4} {:.4} {:.4}\n# metal_domain: 1\nseq_id\tatom_name\tpcs_ppm\tsigma_ppm\n",
        metal_d1[0], metal_d1[1], metal_d1[2]
    );
    for (idx, r) in native_chain.residues.iter().enumerate() {
        if (70..78).contains(&idx) || idx % 2 != 0 {
            continue;
        }
        if let Some(h) = r.h {
            if v_dist(h, metal_d1) >= 6.5 {
                let pv = predict_pcs_spherical_ppm(h, metal_d1, 32.0, 9.5, [0.45, 1.05, 1.85]);
                pcs_tsv.push_str(&format!("{}\tH\t{:.5}\t0.025\n", r.seq_id, pv));
            }
        }
    }
    fs::write(&pcs_in, &pcs_tsv).unwrap();

    let engine_arg = if std::env::var("NMR_ALLOW_NO_GPU").as_deref() == Ok("1") {
        "cpu"
    } else {
        "auto"
    };

    // 1. Full 2-medium RDC + PCS docking -> `docked.cif` + sidecar JSON/TSV
    let mut cmd1 = Command::new("cargo");
    cmd1.current_dir(&root).arg("run").arg("--quiet");
    if !cfg!(debug_assertions) {
        cmd1.arg("--release");
    }
    let status = cmd1
        .args([
            "-p",
            "nmr-cli",
            "--bin",
            "ferric",
            "--",
            "structure",
            "--pae-domains",
            "--af-structure",
            cif_in.to_str().unwrap(),
            "--af-pae",
            pae_in.to_str().unwrap(),
            "--rdc",
            rdc2_in.to_str().unwrap(),
            "--pcs",
            pcs_in.to_str().unwrap(),
            "--rdc-free-frac",
            "0.20",
            "--ensemble-size",
            "4",
            "--replicas",
            "32",
            "--engine",
            engine_arg,
            "-o",
            out_cif.to_str().unwrap(),
        ])
        .output()
        .expect("invoke ferric structure --pae-domains");
    assert!(
        status.status.success(),
        "ferric structure --pae-domains failed: stderr={}",
        String::from_utf8_lossy(&status.stderr)
    );
    assert!(out_cif.exists());
    let side_json = tmp_dir.join("docked.domain_dock.json");
    let side_tsv = tmp_dir.join("docked.domain_dock.tsv");
    assert!(side_json.exists() && side_tsv.exists());
    let parsed_docked = parse_af_structure_model(&out_cif, &AfIngestOptions::default()).unwrap();
    assert_eq!(parsed_docked.residues.len(), native_chain.residues.len());

    // 2. Single-medium RDC without PCS/PRE/NOE -> refused by default (`21.W2:P2-8`), and emits `F8`
    //    candor warning when `--allow-underdetermined-single-medium` is explicitly passed!
    let mut cmd_refuse = Command::new("cargo");
    cmd_refuse.current_dir(&root).arg("run").arg("--quiet");
    if !cfg!(debug_assertions) {
        cmd_refuse.arg("--release");
    }
    let refuse_out = cmd_refuse
        .args([
            "-p",
            "nmr-cli",
            "--bin",
            "ferric",
            "--",
            "structure",
            "--pae-domains",
            "--af-structure",
            cif_in.to_str().unwrap(),
            "--af-pae",
            pae_in.to_str().unwrap(),
            "--rdc",
            rdc1_in.to_str().unwrap(),
            "--ensemble-size",
            "2",
            "--replicas",
            "8",
            "--engine",
            "cpu",
            "-o",
            out_candor_pdb.to_str().unwrap(),
        ])
        .output()
        .expect("invoke single-medium refusal check");
    assert!(
        !refuse_out.status.success(),
        "single-medium RDC-only CLI must fail closed without --allow-underdetermined-single-medium"
    );

    let mut cmd2 = Command::new("cargo");
    cmd2.current_dir(&root).arg("run").arg("--quiet");
    if !cfg!(debug_assertions) {
        cmd2.arg("--release");
    }
    let candor_out = cmd2
        .args([
            "-p",
            "nmr-cli",
            "--bin",
            "ferric",
            "--",
            "structure",
            "--pae-domains",
            "--af-structure",
            cif_in.to_str().unwrap(),
            "--af-pae",
            pae_in.to_str().unwrap(),
            "--rdc",
            rdc1_in.to_str().unwrap(),
            "--allow-underdetermined-single-medium",
            "--ensemble-size",
            "2",
            "--replicas",
            "8",
            "--engine",
            "cpu",
            "-o",
            out_candor_pdb.to_str().unwrap(),
        ])
        .output()
        .expect("invoke single-medium ferric structure --pae-domains with override");
    assert!(
        candor_out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&candor_out.stderr)
    );
    let stderr_txt = String::from_utf8_lossy(&candor_out.stderr);
    assert!(
        stderr_txt.contains("F8") && stderr_txt.contains("cone degeneracy"),
        "expected F8 candor warning on stderr, got: {stderr_txt}"
    );
    let pdb_body = fs::read_to_string(&out_candor_pdb).unwrap();
    assert!(pdb_body.contains("F8"));

    let _ = fs::remove_dir_all(&tmp_dir);
}

#[test]
fn test_webgpu_p0_4_discontinuous_insertion_domain_parity_and_zero_drift() {
    let Some(engine) = require_headless_gpu(Some(wgpu::Limits::default())) else {
        return;
    };

    // Exact `1UBQ` insertion-domain topology (`D1 = [0..22] U [50..72], D2 = [26..46]`)
    let root = workspace_root();
    let ubq = parse_pdb_first_model(root.join("data/fixtures/ubiquitin_1ubq_xray.pdb")).unwrap();
    let n = ubq.residues.len();
    let mut true_dom = vec![0usize; n];
    for d in &mut true_dom[0..22] {
        *d = 1;
    }
    for d in &mut true_dom[26..46] {
        *d = 2;
    }
    for d in &mut true_dom[50..72] {
        *d = 1;
    }
    let mut plddt = vec![92.0_f64; n];
    for p in &mut plddt[22..26] {
        *p = 54.0;
    }
    for p in &mut plddt[46..50] {
        *p = 54.0;
    }
    for p in &mut plddt[72..76] {
        *p = 48.0;
    }

    let cif = serialize_chain_mmcif("ubq_gpu_insertion", &ubq, &plddt);
    let model = parse_af_structure_model_str(&cif, &AfIngestOptions::default()).unwrap();
    let pae = build_pae(n, &true_dom, 2104);
    let seq_ids = model.seq_ids();
    let part = cluster_pae_domains(
        &pae,
        Some(&plddt),
        Some(&seq_ids),
        None,
        &PaeClusterConfig::default(),
    )
    .unwrap();
    assert_eq!(part.domains.len(), 2);
    assert_eq!(part.domains[0].seq_id_spans.len(), 2);

    let kin = DomainLinkerKinematics::from_af_model(&model, &part).unwrap();
    let ch0 = kin.build_pdb_chain(&kin.initial_state());
    let round_mat_f32 = |m: [[f64; 3]; 3]| -> [[f64; 3]; 3] {
        let mut out = [[0.0; 3]; 3];
        for r in 0..3 {
            for c in 0..3 {
                out[r][c] = (m[r][c] as f32) as f64;
            }
        }
        out
    };
    let s1 = round_mat_f32(saupe_matrix_from_pas(-12.5, 0.30, [24.0, 52.0, 104.0]));
    let s2 = round_mat_f32(saupe_matrix_from_pas(14.0, 0.44, [112.0, 76.0, 35.0]));
    let med1 = simulate_rdc_medium(&ch0, "m1", &s1, (-1, -1), 0.0, 811);
    let med2 = simulate_rdc_medium(&ch0, "m2", &s2, (-1, -1), 0.0, 812);

    let prep = PreparedDomainDockRestraints::compile(
        &kin,
        &part,
        &[med1.clone(), med2.clone()],
        &[],
        &[],
        &[],
        None,
    )
    .unwrap();

    let weights = DomainDockEnergyWeights {
        w_rdc: 0.05,
        w_pcs: 0.15,
        w_pre: 0.10,
        w_noe: 0.25,
        noe_soft_window: 3.0,
        w_vdw: 0.35,
        vdw_scale: 0.92,
        w_rama: 0.80,
    };
    let mut rng = Rng::new(2105);
    let mut batch_in = Vec::with_capacity(12);
    for _ in 0..12 {
        let mut st = kin.initial_state();
        for i in 0..n {
            st.phi_rad[i] = (st.phi_rad[i] as f32) as f64;
            st.psi_rad[i] = (st.psi_rad[i] as f32) as f64;
            st.omega_rad[i] = (st.omega_rad[i] as f32) as f64;
        }
        for &idx in &kin.active_linker_residues {
            st.phi_rad[idx] = ((st.phi_rad[idx] + (rng.f64() - 0.5) * 0.5) as f32) as f64;
            st.psi_rad[idx] = ((st.psi_rad[idx] + (rng.f64() - 0.5) * 0.5) as f32) as f64;
        }
        batch_in.push(DomainDockGpuReplicaInput {
            state: st,
            saupe_matrices: vec![s1, s2],
        });
    }

    let gpu_outs =
        evaluate_domain_dock_batch_gpu(&engine, &kin, &prep, &batch_in, &weights).unwrap();
    for (ri, (rep_in, (eb_gpu, gr_gpu))) in batch_in.iter().zip(gpu_outs.iter()).enumerate() {
        let (eb_cpu, gr_cpu) = evaluate_domain_dock_fixed_saupe_cpu(
            &kin,
            &prep,
            &rep_in.state,
            &rep_in.saupe_matrices,
            &weights,
        );
        let rel_e = (eb_gpu.total - eb_cpu.total).abs() / eb_cpu.total.abs().max(1.0);
        assert!(
            rel_e < 1.0e-5,
            "insertion domain replica {ri} energy parity failed: gpu={eb_gpu:?} cpu={eb_cpu:?} rel={rel_e:.3e}"
        );
        let g_scale = gr_cpu
            .grad_phi
            .iter()
            .chain(gr_cpu.grad_psi.iter())
            .fold(1.0_f64, |m, v| m.max(v.abs()));
        for i in 0..n {
            let rp = (gr_gpu.grad_phi[i] - gr_cpu.grad_phi[i]).abs() / g_scale;
            let rs = (gr_gpu.grad_psi[i] - gr_cpu.grad_psi[i]).abs() / g_scale;
            assert!(
                rp < 1.0e-5 && rs < 1.0e-5,
                "insertion domain replica {ri} grad[{i}] parity failed: rp={rp:.3e}, rs={rs:.3e}"
            );
        }
    }

    // Also verify end-to-end `dock_domains_gpu` on discontinuous insertion domain preserves `0.000 A` drift!
    let cfg = DomainDockConfig {
        ensemble_size: 2,
        n_replicas: 16,
        anneal_steps: 20,
        lbfgs_iters: 60,
        ..Default::default()
    };
    let metal_d1 = v_add(ch0.residues[60].ca, [3.0, -2.5, 2.8]);
    let mut pcs_tsv = format!(
        "# metal_pos: {:.4} {:.4} {:.4}\n# metal_domain: 1\nseq_id\tatom_name\tpcs_ppm\tsigma_ppm\n",
        metal_d1[0], metal_d1[1], metal_d1[2]
    );
    for (idx, r) in ch0.residues.iter().enumerate() {
        if true_dom[idx] != 0 {
            if let Some(h) = r.h {
                if v_dist(h, metal_d1) > 6.0 {
                    let pv = predict_pcs_spherical_ppm(h, metal_d1, 28.0, 8.0, [0.4, 1.0, 1.8]);
                    pcs_tsv.push_str(&format!("{}\tH\t{pv:.5}\t0.03\n", r.seq_id));
                }
            }
        }
    }
    let pcs_tags = parse_domain_pcs_tsv_str(&pcs_tsv).unwrap();
    let gpu_dock = dock_domains_gpu(
        &engine,
        &model,
        &pae,
        &[med1, med2],
        &pcs_tags,
        &[],
        &[],
        &cfg,
    )
    .unwrap();
    assert!(
        gpu_dock.report.max_intra_domain_rmsd_angstrom < 1.0e-12,
        "WebGPU discontinuous insertion domain internal drift = {:.3e} A",
        gpu_dock.report.max_intra_domain_rmsd_angstrom
    );
}
