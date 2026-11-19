//! Integration and quantitative verification gate tests for `T21.5` (`nmr-core::domain_dock`,
//! `docs/specs/21-alphafold-integrative-nmr.md` §5 & §8):
//! 1. Symmetrized `PAE` normalized-Laplacian spectral clustering (`ARI >= 0.98` on 2- and 3-domain
//!    benchmarks, discontinuous domain insertions, `D = 1` single-domain control, hostile guards).
//! 2. Tolman et al. (2001) per-domain Generalized Degree of Order ($\vartheta_d / \vartheta_1$)
//!    motion test (`|\Delta\vartheta|/\vartheta_1 < 0.05` rigid co-tumbling vs `100%` detection of
//!    `2x`-attenuated `DecoupledInterDomainMotion`).
//! 3. Exact `0.000 Å` intra-domain rigid preservation & `< 1e-6` central finite-difference
//!    analytical gradient verification across `RDC`, `PCS` (including moving-tag torque), `PRE`,
//!    inter-domain `NOE`, `GridHash3D` `vdW`, Lovell `Rama`, and multi-chain $\mathfrak{se}(3)$.
//! 4. Joint 2-medium `RDC` + `PCS` + `PRE` covalent linker docking from a `>= 60°`-rotated,
//!    `>= 8 Å`-translated two-domain AF starting pose under `ByPeptidePlane` `20%` `RdcFree` holdout
//!    (`< 1.20 Å` all-domain BB RMSD, `Q_free < 0.25`, `0.000 Å` intra-domain distortion, `0`
//!    `is_free` leakage, plus `F8` single-medium cone candor & `.pdb`/`.cif` round-trips).

use std::f64::consts::PI;
use std::path::PathBuf;

use nmr_core::af_concordance::{
    parse_af_structure_model, parse_af_structure_model_str, AfIngestOptions, PaeJsonDialect,
    PaeMatrix,
};
use nmr_core::domain_dock::{
    adjusted_rand_index, cluster_pae_domains, compute_solomon_k_pre_a6_s1, dock_domains_cpu,
    dock_domains_multi_chain_cpu, evaluate_domain_dock_fixed_saupe_cpu, evaluate_solomon_gamma2_s1,
    evaluate_tolman_domain_gdo, format_docked_mmcif_ensemble, format_docked_pdb_ensemble,
    format_domain_dock_json, format_domain_dock_tsv, parse_domain_noe_tsv_str,
    parse_domain_pcs_tsv_str, parse_domain_pre_tsv_str, parse_domain_pre_tsv_str_with_defaults,
    parse_domain_rdc_tsv_str, partition_domain_stratified_pcs_free_set,
    partition_domain_stratified_rdc_free_set, prepare_domain_dock_inputs_full,
    quat_apply_world_rot_step, read_domain_rdc_tsv, solomon_gamma2_to_distance_bounds,
    ChainSeqIndex, DipolarObservabilityRegime, DomainDockConfig, DomainDockEnergyWeights,
    DomainDockQualityVerdict, DomainLinkerKinematics, DomainMotionRegime, DomainPcsRestraint,
    DomainPcsTagSet, DomainPreRestraint, DomainPreSiteSet, PaeClusterConfig, PaeDomainPartition,
    PreRestraintKind, PreparedDomainDockRestraints, ResidueDomainRole, CODATA_ELECTRON_G_FACTOR,
    DEFAULT_MTSL_TETHER_ALLOWANCE_ANGSTROM, K_SOLOMON_PROTON_NITROXIDE_A6_S2,
    MAX_PEPTIDE_JOIN_C_N_ANGSTROM,
};
use nmr_core::noesy::{
    parse_pdb_first_model, v_add, v_dist, v_norm, v_scale, v_sub, v_unit,
    AmbiguousDistanceRestraint, NoeCandidatePair, NoeProton, PdbChain, Vec3,
};
use nmr_core::paramagnetic::predict_pcs_spherical_ppm;
use nmr_core::rdc::{
    calc_rdc_from_unit_vec, rdc_peptide_plane_id, saupe_matrix_from_pas, RdcBondType, RdcMediumSet,
    RdcRestraint,
};
use nmr_core::rng::Rng;
use nmr_core::structure::{backbone_rmsd_between_chains, kabsch_transform};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn build_synthetic_pae(
    n: usize,
    true_dom: &[usize],
    seed: u64,
    intra_base: f32,
    inter_base: f32,
) -> PaeMatrix {
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
            let jitter = ((rng.f64() - 0.5) * 1.4) as f32;
            let v = if di != 0 && di == dj {
                let seq_sep = (i.abs_diff(j) as f32).min(40.0) * 0.035;
                (intra_base + seq_sep + jitter).clamp(0.8, 4.6)
            } else if di == 0 || dj == 0 {
                let local = if i.abs_diff(j) <= 2 { 3.5 } else { 15.5 };
                (local + jitter).clamp(1.5, 28.0)
            } else {
                // Asymmetric cross-domain PAE (`20..26 Å`)
                let asym = if i < j { 2.0 } else { -1.5 };
                (inter_base + asym + jitter).clamp(16.0, 31.0)
            };
            vals[i * n + j] = v;
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

fn serialize_chain_to_af_mmcif(
    entry_id: &str,
    chain: &PdbChain,
    plddts: &[f64],
    chain_ids_override: Option<&[&str]>,
) -> String {
    assert_eq!(chain.residues.len(), plddts.len());
    let mut cif = String::with_capacity(chain.residues.len() * 340);
    cif.push_str(&format!(
        "data_{entry_id}\n_entry.id {entry_id}\n_ma_protocol_step.step_id 1\n"
    ));
    cif.push_str(
        "loop_\n_ma_qa_metric.id\n_ma_qa_metric.name\n_ma_qa_metric.mode\n1 pLDDT local\n#\n",
    );
    cif.push_str("loop_\n_atom_site.group_PDB\n_atom_site.id\n_atom_site.type_symbol\n_atom_site.label_atom_id\n_atom_site.label_alt_id\n_atom_site.label_comp_id\n_atom_site.label_asym_id\n_atom_site.auth_asym_id\n_atom_site.label_seq_id\n_atom_site.auth_seq_id\n_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n_atom_site.occupancy\n_atom_site.B_iso_or_equiv\n_atom_site.pdbx_pdb_model_num\n");

    let mut serial = 1usize;
    for (idx, r) in chain.residues.iter().enumerate() {
        let asym = chain_ids_override
            .and_then(|v| v.get(idx).copied())
            .unwrap_or("A");
        let lseq = idx + 1;
        let aseq = r.seq_id;
        let mut emit = |sym: &str, aname: &str, xyz: Vec3| {
            cif.push_str(&format!(
                "ATOM {serial} {sym} {aname} . {} {asym} {asym} {lseq} {aseq} {:.4} {:.4} {:.4} 1.00 {:.2} 1\n",
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
        let asym = chain_ids_override
            .and_then(|v| v.get(idx).copied())
            .unwrap_or("A");
        cif.push_str(&format!("{asym} {} {} 1 {:.2}\n", idx + 1, r.res_name, pl));
    }
    cif.push_str("#\n");
    cif
}

/// Build a covalent two-domain fusion construct (`Dom1`: Ubiquitin `1..70` + `8`-residue flexible
/// linker `GSGSAGSG` + `Dom2`: KH3 `1..66` renumbered `79..144`, `L = 144` total) from public
/// PDB fixtures `ubiquitin_1ubq_xray.pdb` and `kh3_1zzk_xray.pdb`, returning `(native_af_model, pae, partition)`.
fn build_two_domain_ubq_kh3_benchmark() -> (
    nmr_core::af_concordance::AfStructureModel,
    PaeMatrix,
    PaeDomainPartition,
) {
    let root = repo_root();
    let ubq = parse_pdb_first_model(root.join("data/fixtures/ubiquitin_1ubq_xray.pdb"))
        .expect("load ubiquitin_1ubq_xray.pdb");
    let kh3 = parse_pdb_first_model(root.join("data/fixtures/kh3_1zzk_xray.pdb"))
        .expect("load kh3_1zzk_xray.pdb");

    let dom1_len = 70usize;
    let link_seq = ["GLY", "SER", "GLY", "SER", "ALA", "GLY", "SER", "GLY"];
    let link_len = link_seq.len();
    let dom2_len = 66usize.min(kh3.residues.len());
    let total_l = dom1_len + link_len + dom2_len;

    // Seed raw chain by placing Dom1 at its native coords, extending an 8-aa linker from C_{69},
    // and rigidly attaching KH3 at the end of the linker so every peptide bond & rigid domain is exact!
    let mut raw_res = Vec::with_capacity(total_l);
    for i in 0..dom1_len {
        let mut r = ubq.residues[i].clone();
        r.seq_id = (i + 1) as i32;
        r.chain_id = 'A';
        raw_res.push(r);
    }

    let linker_phis_deg = [-80.0_f64, -70.0, -65.0, -75.0, -70.0, -80.0, -75.0, -80.0];
    let linker_psis_deg = [140.0_f64, 135.0, -35.0, 140.0, 135.0, 140.0, 135.0, 140.0];

    for k in 0..link_len {
        let prev = raw_res.last().expect("prev res").clone();
        let psi_prev = if k == 0 {
            150.0_f64.to_radians()
        } else {
            linker_psis_deg[k - 1].to_radians()
        };
        let phi_cur = linker_phis_deg[k].to_radians();
        let psi_cur = linker_psis_deg[k].to_radians();
        let n_p = nmr_core::structure::nerf_place_atom(
            prev.n,
            prev.ca,
            prev.c,
            1.329,
            116.2_f64.to_radians(),
            psi_prev,
        );
        let ca_p = nmr_core::structure::nerf_place_atom(
            prev.ca,
            prev.c,
            n_p,
            1.458,
            121.7_f64.to_radians(),
            PI,
        );
        let h_p = nmr_core::structure::nerf_place_atom(
            prev.ca,
            prev.c,
            n_p,
            1.010,
            119.2_f64.to_radians(),
            0.0,
        );
        let rname = link_seq[k];
        let ang_ca = if rname == "GLY" { 112.5_f64 } else { 111.2_f64 };
        let c_p = nmr_core::structure::nerf_place_atom(
            prev.c,
            n_p,
            ca_p,
            1.525,
            ang_ca.to_radians(),
            phi_cur,
        );
        let o_p = nmr_core::structure::nerf_place_atom(
            n_p,
            ca_p,
            c_p,
            1.231,
            120.8_f64.to_radians(),
            psi_cur + PI,
        );
        let (cb, ha, ha2, ha3, qb) =
            nmr_core::noesy::reconstruct_ca_substituents(rname, n_p, ca_p, c_p);
        raw_res.push(nmr_core::noesy::PdbResidue {
            seq_id: (dom1_len + k + 1) as i32,
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

    // Place KH3 N_0, CA_0, C_0 covalently at the end of linker residue `link_len - 1`, then rigidly
    // transform all `dom2_len` KH3 residues onto that frame!
    let lk_last = raw_res.last().expect("lk_last").clone();
    let psi_last = linker_psis_deg[link_len - 1].to_radians();
    let n_d2 = nmr_core::structure::nerf_place_atom(
        lk_last.n,
        lk_last.ca,
        lk_last.c,
        1.329,
        116.2_f64.to_radians(),
        psi_last,
    );
    let r_n_ca_kh3 = v_dist(kh3.residues[0].n, kh3.residues[0].ca);
    let r_ca_c_kh3 = v_dist(kh3.residues[0].ca, kh3.residues[0].c);
    let u_ba = v_unit(v_sub(kh3.residues[0].n, kh3.residues[0].ca));
    let u_bc = v_unit(v_sub(kh3.residues[0].c, kh3.residues[0].ca));
    let ang_kh3 = (u_ba[0] * u_bc[0] + u_ba[1] * u_bc[1] + u_ba[2] * u_bc[2])
        .clamp(-1.0, 1.0)
        .acos();
    let ca_d2 = nmr_core::structure::nerf_place_atom(
        lk_last.ca,
        lk_last.c,
        n_d2,
        r_n_ca_kh3,
        121.7_f64.to_radians(),
        PI,
    );
    let c_d2 = nmr_core::structure::nerf_place_atom(
        lk_last.c,
        n_d2,
        ca_d2,
        r_ca_c_kh3,
        ang_kh3,
        -135.0_f64.to_radians(),
    );

    let triad_axes = |n: Vec3, ca: Vec3, c: Vec3| -> [[f64; 3]; 3] {
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
    let f_src = triad_axes(kh3.residues[0].n, kh3.residues[0].ca, kh3.residues[0].c);
    let f_dst = triad_axes(n_d2, ca_d2, c_d2);
    let xform_kh3 = |p: Vec3| -> Vec3 {
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

    for j in 0..dom2_len {
        let kr = &kh3.residues[j];
        let mut r = kr.clone();
        r.seq_id = (dom1_len + link_len + j + 1) as i32;
        r.chain_id = 'A';
        r.n = xform_kh3(kr.n);
        r.h = if j == 0 {
            Some(nmr_core::structure::nerf_place_atom(
                lk_last.ca,
                lk_last.c,
                n_d2,
                1.010,
                119.2_f64.to_radians(),
                0.0,
            ))
        } else {
            kr.h.map(xform_kh3)
        };
        r.ca = xform_kh3(kr.ca);
        r.ha = xform_kh3(kr.ha);
        r.ha2 = kr.ha2.map(xform_kh3);
        r.ha3 = kr.ha3.map(xform_kh3);
        r.c = xform_kh3(kr.c);
        r.o = xform_kh3(kr.o);
        r.cb = kr.cb.map(xform_kh3);
        r.qb = kr.qb.map(xform_kh3);
        raw_res.push(r);
    }

    let chain = PdbChain { residues: raw_res };
    let mut plddts = vec![91.5_f64; total_l];
    let mut true_labels = vec![1usize; total_l];
    for i in dom1_len..(dom1_len + link_len) {
        plddts[i] = 52.0;
        true_labels[i] = 0;
    }
    for i in (dom1_len + link_len)..total_l {
        plddts[i] = 89.0;
        true_labels[i] = 2;
    }

    let cif_str = serialize_chain_to_af_mmcif("ubq_kh3_native", &chain, &plddts, None);
    let model = parse_af_structure_model_str(&cif_str, &AfIngestOptions::default())
        .expect("parse native two-domain mmCIF");
    let pae = build_synthetic_pae(total_l, &true_labels, 202605, 2.2, 23.0);
    let part = nmr_core::domain_dock::cluster_pae_domains_for_model(
        &model,
        &pae,
        &PaeClusterConfig::default(),
    )
    .expect("cluster two-domain PAE");
    (model, pae, part)
}

#[allow(clippy::too_many_arguments)]
fn simulate_domain_rdc_medium(
    chain: &PdbChain,
    medium_id: &str,
    s_dom1: &[[f64; 3]; 3],
    s_dom2: &[[f64; 3]; 3],
    dom2_start_seq_id: i32,
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
        let s_mat = if r.seq_id >= dom2_start_seq_id {
            s_dom2
        } else {
            s_dom1
        };
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
            let prev_in_linker = prev.seq_id >= linker_range.0 && prev.seq_id <= linker_range.1;
            if !prev_in_linker && v_dist(prev.c, r.n) < 2.0 {
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
fn test_pae_spectral_clustering_2_and_3_domain_ari_and_hostile_guards() {
    let cfg = PaeClusterConfig::default();

    // 1. Single-domain control (`L = 76`): must pick D = 1 with 0 linkers
    let single_true = vec![1usize; 76];
    let single_plddt = vec![92.0; 76];
    let single_pae = build_synthetic_pae(76, &single_true, 11, 2.1, 22.0);
    let single_part =
        cluster_pae_domains(&single_pae, Some(&single_plddt), None, None, &cfg).unwrap();
    assert_eq!(single_part.domains.len(), 1);
    assert_eq!(single_part.linkers.len(), 0);
    assert!(
        (adjusted_rand_index(&single_part.domain_labels, &single_true).unwrap() - 1.0).abs()
            < 1e-12
    );

    // 2. 2-domain benchmark (`L = 158`: 4 N-tail + 65 Dom1 + 10 Linker + 74 Dom2 + 5 C-tail),
    //    including a 2-residue low-pLDDT surface turn inside Dom1 (`pLDDT = 66`) and 2 ordered
    //    helical linker residues (`pLDDT = 72`, `PAE ~ 16 Å` to both domains)!
    let l2 = 158usize;
    let mut true2 = vec![0usize; l2];
    let mut plddt2 = vec![44.0_f64; l2];
    for i in 4..69 {
        true2[i] = 1;
        plddt2[i] = 90.0;
    }
    plddt2[31] = 65.5;
    plddt2[32] = 66.0;
    for i in 69..79 {
        true2[i] = 0;
        plddt2[i] = 52.0;
    }
    plddt2[73] = 72.5;
    plddt2[74] = 73.0;
    for i in 79..153 {
        true2[i] = 2;
        plddt2[i] = 88.5;
    }
    let pae2 = build_synthetic_pae(l2, &true2, 22, 2.3, 24.0);
    let part2 = cluster_pae_domains(&pae2, Some(&plddt2), None, None, &cfg).unwrap();
    assert_eq!(part2.domains.len(), 2, "expected D = 2 rigid domains");
    assert_eq!(part2.linkers.len(), 1, "expected 1 inter-domain linker");
    assert_eq!(part2.residue_roles[0], ResidueDomainRole::NTermTail);
    assert_eq!(part2.residue_roles[l2 - 1], ResidueDomainRole::CTermTail);
    assert!(
        matches!(
            part2.residue_roles[74],
            ResidueDomainRole::FlexibleLinker { linker_id: 1 }
        ),
        "ordered helical linker residue with high PAE must be classified into FlexibleLinker"
    );
    let ari2 = adjusted_rand_index(&part2.domain_labels, &true2).unwrap();
    assert!(
        ari2 >= 0.98,
        "2-domain PAE spectral clustering ARI = {ari2:.4} < 0.98"
    );
    for dom in &part2.domains {
        assert!(dom.median_intra_pae_angstrom <= 5.0);
        assert!(dom.residue_indices.len() >= 20);
    }

    // 3. 3-domain benchmark (`L = 196`: 3 N-tail + 55 Dom1 + 8 Link1 + 61 Dom2 + 9 Link2 + 56 Dom3 + 4 C-tail)
    let l3 = 196usize;
    let mut true3 = vec![0usize; l3];
    let mut plddt3 = vec![46.0_f64; l3];
    for i in 3..58 {
        true3[i] = 1;
        plddt3[i] = 91.0;
    }
    for i in 66..127 {
        true3[i] = 2;
        plddt3[i] = 87.5;
    }
    for i in 136..192 {
        true3[i] = 3;
        plddt3[i] = 89.5;
    }
    let pae3 = build_synthetic_pae(l3, &true3, 33, 2.2, 23.5);
    let part3 = cluster_pae_domains(&pae3, Some(&plddt3), None, None, &cfg).unwrap();
    assert_eq!(part3.domains.len(), 3, "expected D = 3 rigid domains");
    assert_eq!(part3.linkers.len(), 2, "expected 2 inter-domain linkers");
    let ari3 = adjusted_rand_index(&part3.domain_labels, &true3).unwrap();
    assert!(
        ari3 >= 0.98,
        "3-domain PAE spectral clustering ARI = {ari3:.4} < 0.98"
    );

    // 4. Discontinuous domain insertion (`Dom1 = 0..32 U 86..118`, `Dom2 = 40..78`)
    let l_ins = 118usize;
    let mut true_ins = vec![0usize; l_ins];
    let mut plddt_ins = vec![50.0_f64; l_ins];
    for i in (0..32).chain(86..118) {
        true_ins[i] = 1;
        plddt_ins[i] = 90.0;
    }
    for i in 40..78 {
        true_ins[i] = 2;
        plddt_ins[i] = 88.0;
    }
    let pae_ins = build_synthetic_pae(l_ins, &true_ins, 44, 2.1, 24.0);
    let part_ins = cluster_pae_domains(&pae_ins, Some(&plddt_ins), None, None, &cfg).unwrap();
    assert_eq!(part_ins.domains.len(), 2);
    assert_eq!(part_ins.domains[0].seq_id_spans.len(), 2);
    let ari_ins = adjusted_rand_index(&part_ins.domain_labels, &true_ins).unwrap();
    assert!(
        ari_ins >= 0.98,
        "discontinuous insertion PAE ARI = {ari_ins:.4} < 0.98"
    );

    println!(
        "PAE Spectral Clustering ARI: 1-dom={:.4}, 2-dom={ari2:.4}, 3-dom={ari3:.4}, discont={ari_ins:.4}",
        adjusted_rand_index(&single_part.domain_labels, &single_true).unwrap()
    );
    // 5. Bit-exact determinism across repeated calls & hostile input Err guards
    let part3_repeat = cluster_pae_domains(&pae3, Some(&plddt3), None, None, &cfg).unwrap();
    assert_eq!(part3, part3_repeat);

    let mut bad_pae = pae2.clone();
    bad_pae.values_angstrom[10] = f32::NAN;
    assert!(cluster_pae_domains(&bad_pae, Some(&plddt2), None, None, &cfg).is_err());
    assert!(cluster_pae_domains(&pae2, Some(&plddt2[..10]), None, None, &cfg).is_err());
}

#[test]
fn test_tolman_gdo_motion_test_rigid_vs_decoupled_and_work_guard() {
    let (model, _, part) = build_two_domain_ubq_kh3_benchmark();
    let chain = model.to_pdb_chain();
    let seq_ids = model.seq_ids();

    let s1_m1 = saupe_matrix_from_pas(-12.5, 0.28, [25.0, 52.0, 110.0]);
    let s1_m2 = saupe_matrix_from_pas(14.0, 0.44, [118.0, 74.0, 35.0]);

    // 1. Rigid co-tumbling (`S_dom2 == S_dom1` + experimental noise)
    let mut media_rigid = vec![
        simulate_domain_rdc_medium(&chain, "bicelle", &s1_m1, &s1_m1, 79, (71, 78), 1.0, 101),
        simulate_domain_rdc_medium(&chain, "pf1", &s1_m2, &s1_m2, 79, (71, 78), 1.0, 102),
    ];
    partition_domain_stratified_rdc_free_set(&mut media_rigid, &part, &seq_ids, 0.20, 42);
    let rep_rigid = evaluate_tolman_domain_gdo(&chain, &part, &media_rigid, 200, 42);
    assert_eq!(rep_rigid.regime, DomainMotionRegime::RigidlyCoupledDomains);
    assert!(
        rep_rigid.max_abs_delta_over_theta1 < 0.05,
        "expected |Delta theta|/theta_1 < 0.05 on rigid co-tumbling, got {:.4}",
        rep_rigid.max_abs_delta_over_theta1
    );
    assert!(rep_rigid.warning_message.is_none());

    // 2. `2x`-attenuated decoupled domain motion (`S_dom2 = 0.50 * S_dom1`) across 10 independent seeds
    let mut s2_half_m1 = s1_m1;
    let mut s2_half_m2 = s1_m2;
    for r in 0..3 {
        for c in 0..3 {
            s2_half_m1[r][c] *= 0.50;
            s2_half_m2[r][c] *= 0.50;
        }
    }
    let mut flagged_count = 0usize;
    for trial_seed in 0..10u64 {
        let mut media_dec = vec![
            simulate_domain_rdc_medium(
                &chain,
                "bicelle",
                &s1_m1,
                &s2_half_m1,
                79,
                (71, 78),
                1.0,
                500 + trial_seed,
            ),
            simulate_domain_rdc_medium(
                &chain,
                "pf1",
                &s1_m2,
                &s2_half_m2,
                79,
                (71, 78),
                1.0,
                700 + trial_seed,
            ),
        ];
        partition_domain_stratified_rdc_free_set(&mut media_dec, &part, &seq_ids, 0.20, trial_seed);
        let rep_dec = evaluate_tolman_domain_gdo(&chain, &part, &media_dec, 200, trial_seed);
        if rep_dec.regime == DomainMotionRegime::DecoupledInterDomainMotion {
            flagged_count += 1;
        }
        assert!(
            (rep_dec.min_rho_gdo - 0.50).abs() < 0.06,
            "expected rho_GDO ~ 0.50, got {:.4}",
            rep_dec.min_rho_gdo
        );
        assert!(rep_dec.warning_message.is_some());
    }
    assert_eq!(
        flagged_count, 10,
        "2x-attenuated decoupled domain motion must be flagged 100% (10/10)"
    );
    println!(
        "Tolman GDO gate: rigid max_|dtheta|/theta_1={:.4} (< 0.05), decoupled 2x flagged={flagged_count}/10 (100%)",
        rep_rigid.max_abs_delta_over_theta1
    );

    // 3. `< 8` working RDC guard on Domain 2 -> `InsufficientDomainRdcs`
    let mut sparse_med = media_rigid[0].clone();
    let mut kept_d2 = 0usize;
    sparse_med.restraints.retain(|r| {
        if r.seq_id_1 >= 79 {
            kept_d2 += 1;
            kept_d2 <= 5
        } else {
            true
        }
    });
    let rep_sparse = evaluate_tolman_domain_gdo(&chain, &part, &[sparse_med], 100, 1);
    assert_eq!(
        rep_sparse.regime,
        DomainMotionRegime::InsufficientDomainRdcs
    );
}

#[test]
fn test_domain_linker_kinematics_analytical_gradients_and_zero_distortion() {
    let (model, _, part) = build_two_domain_ubq_kh3_benchmark();
    let kin = DomainLinkerKinematics::from_af_model(&model, &part).unwrap();

    // 1. Exact reproduction at initial_state (`< 1e-13 Å`) and `0.000 Å` intra-domain distortion under large perturbations
    let st0 = kin.initial_state();
    let ch0 = kin.build_pdb_chain(&st0);
    let init_rmsd = backbone_rmsd_between_chains(&ch0, &kin.ref_chain, None);
    assert!(
        init_rmsd < 1e-12,
        "initial_state must reproduce input AF coordinates identically, got {init_rmsd:.3e} A"
    );
    assert_eq!(kin.max_intra_domain_rmsd(&ch0, &part), 0.0);

    let mut st_pert = st0.clone();
    for (k, &idx) in kin.active_linker_residues.iter().enumerate() {
        st_pert.phi_rad[idx] += 0.55 * ((k as f64 + 1.0).sin());
        st_pert.psi_rad[idx] -= 0.65 * ((k as f64 + 2.0).cos());
    }
    let ch_pert = kin.build_pdb_chain(&st_pert);
    assert_eq!(
        kin.max_intra_domain_rmsd(&ch_pert, &part),
        0.0,
        "intra-domain distortion must remain 0.000 A under arbitrary linker torsions"
    );

    // 2. Analytical gradient vs central finite-difference check across ALL 6 terms (`RDC`, `PCS` on moving
    //    domain 2 to test rotational DeltaChi torque!, `PRE` Gamma2 + bounds, `NOE`, `vdW`, `Rama`)
    let s_m1 = saupe_matrix_from_pas(-11.0, 0.30, [30.0, 48.0, 95.0]);
    let rdc_med = simulate_domain_rdc_medium(&ch0, "m1", &s_m1, &s_m1, 79, (71, 78), 0.0, 7);

    // Place a PCS tag inside moving Domain 2 (`host_domain_id = 2`) so BOTH translational force and
    // rotational `(M v) x v` torque on Domain 2 are exercised!
    let m_pos_d2 = v_add(
        ch0.residues[100].cb.unwrap_or(ch0.residues[100].ca),
        [3.2, 2.1, -2.8],
    );
    let euler_pcs = [0.6, 1.1, 2.0];
    let mut pcs_rows = Vec::new();
    for (idx, r) in ch0.residues.iter().enumerate() {
        if idx % 5 == 0 {
            if let Some(h) = r.h {
                if v_dist(h, m_pos_d2) > 5.0 {
                    let val = predict_pcs_spherical_ppm(h, m_pos_d2, 22.0, 6.5, euler_pcs);
                    pcs_rows.push(DomainPcsRestraint {
                        tag_id: "yb_d2".to_string(),
                        chain_id: None,
                        seq_id: r.seq_id,
                        atom_name: "H".to_string(),
                        pcs_obs_ppm: val,
                        sigma_ppm: 0.05,
                        is_free: false,
                    });
                }
            }
        }
    }
    let pcs_tag = DomainPcsTagSet {
        tag_id: "yb_d2".to_string(),
        metal_pos_initial: Some(m_pos_d2),
        anchor_chain_id: None,
        anchor_residue_atom: None,
        host_domain_id: Some(2),
        pre_fitted_tensor: None,
        refine_metal_pos: false,
        metal_search_radius_a: None,
        restraints: pcs_rows,
    };

    let s_pos_d1 = v_add(ch0.residues[20].ca, [-3.0, 2.5, 2.5]);
    let k_solomon_10ns_600 =
        compute_solomon_k_pre_a6_s1(10.0, 600.0, 0.5, CODATA_ELECTRON_G_FACTOR).unwrap();
    let pre_site = DomainPreSiteSet {
        site_id: "mtsl_d1".to_string(),
        site_pos_initial: Some(s_pos_d1),
        anchor_chain_id: None,
        anchor_residue_atom: None,
        host_domain_id: Some(1),
        tau_c_ns: Some(10.0),
        proton_mhz: Some(600.0),
        mtsl_tether_allowance_a: None,
        restraints: vec![
            DomainPreRestraint {
                site_id: "mtsl_d1".to_string(),
                chain_id: None,
                seq_id: 95,
                atom_name: "H".to_string(),
                kind: PreRestraintKind::Gamma2Rate {
                    gamma2_obs_s1: 18.0,
                    sigma_s1: 2.5,
                    k_pre_a6_s1: k_solomon_10ns_600,
                },
            },
            DomainPreRestraint {
                site_id: "mtsl_d1".to_string(),
                chain_id: None,
                seq_id: 112,
                atom_name: "CA".to_string(),
                kind: PreRestraintKind::DistanceBounds {
                    lower_a: 12.0,
                    upper_a: 16.0,
                    sigma_a: 1.0,
                },
            },
        ],
    };

    let noe_adr = AmbiguousDistanceRestraint {
        peak_index: 1,
        exp: "EXP_NOE".to_string(),
        lower_bound: 2.0,
        upper_bound: 6.0,
        candidates: vec![NoeCandidatePair {
            res_a: 35,
            atom_a: NoeProton::AlphaHA,
            res_b: 105,
            atom_b: NoeProton::AlphaHA,
            shift_score: 1.0,
            network_score: 1.0,
            weight: 1.0,
        }],
        all_candidates: Vec::new(),
        is_unambiguous: true,
    };

    let prep = PreparedDomainDockRestraints::compile(
        &kin,
        &part,
        &[rdc_med],
        &[pcs_tag],
        &[pre_site],
        &[noe_adr],
        None,
    )
    .unwrap();

    // Pick a compact conformation where vdW clashes, NOE, PRE, PCS, RDC, and Rama are ALL active!
    let mut test_st = st0.clone();
    for &idx in &kin.active_linker_residues {
        test_st.phi_rad[idx] = -1.15;
        test_st.psi_rad[idx] = -0.55;
    }
    let fixed_saupe = vec![s_m1];
    let weights = DomainDockEnergyWeights {
        w_rdc: 0.4,
        w_pcs: 0.5,
        w_pre: 0.4,
        w_noe: 1.5,
        noe_soft_window: 3.0,
        w_vdw: 1.2,
        vdw_scale: 1.15,
        w_rama: 4.0,
    };

    let (_, grad) =
        evaluate_domain_dock_fixed_saupe_cpu(&kin, &prep, &test_st, &fixed_saupe, &weights);
    let eps = 1.0e-6_f64;
    for &idx in &kin.active_linker_residues {
        if kin.residues[idx].phi_active {
            let mut sp = test_st.clone();
            let mut sm = test_st.clone();
            sp.phi_rad[idx] += eps;
            sm.phi_rad[idx] -= eps;
            let (ep, _) =
                evaluate_domain_dock_fixed_saupe_cpu(&kin, &prep, &sp, &fixed_saupe, &weights);
            let (em, _) =
                evaluate_domain_dock_fixed_saupe_cpu(&kin, &prep, &sm, &fixed_saupe, &weights);
            let fd = (ep.total - em.total) / (2.0 * eps);
            let rel = (grad.grad_phi[idx] - fd).abs() / fd.abs().max(1.0);
            assert!(
                rel < 1.0e-6,
                "phi[{idx}] analytical grad {:.6e} != fd {:.6e} (rel {rel:.2e})",
                grad.grad_phi[idx],
                fd
            );
        }
        if kin.residues[idx].psi_active {
            let mut sp = test_st.clone();
            let mut sm = test_st.clone();
            sp.psi_rad[idx] += eps;
            sm.psi_rad[idx] -= eps;
            let (ep, _) =
                evaluate_domain_dock_fixed_saupe_cpu(&kin, &prep, &sp, &fixed_saupe, &weights);
            let (em, _) =
                evaluate_domain_dock_fixed_saupe_cpu(&kin, &prep, &sm, &fixed_saupe, &weights);
            let fd = (ep.total - em.total) / (2.0 * eps);
            let rel = (grad.grad_psi[idx] - fd).abs() / fd.abs().max(1.0);
            assert!(
                rel < 1.0e-6,
                "psi[{idx}] analytical grad {:.6e} != fd {:.6e} (rel {rel:.2e})",
                grad.grad_psi[idx],
                fd
            );
        }
    }

    // 3. Multi-chain `chain_A + chain_B` SE(3) `(grad_chain_rot, grad_chain_trans)` finite-difference check
    let mut mc_residues = model.residues[..45].to_vec();
    for (k, r) in model.residues[90..135].iter().enumerate() {
        let mut rb = r.clone();
        rb.chain_id = "B".to_string();
        rb.seq_index = 45 + k;
        mc_residues.push(rb);
    }
    let mut mc_true = vec![1usize; 45];
    mc_true.extend(vec![2usize; 45]);
    let mc_pae = build_synthetic_pae(90, &mc_true, 99, 2.0, 25.0);
    let mc_sids: Vec<i32> = mc_residues.iter().map(|r| r.seq_id).collect();
    let mc_cids: Vec<String> = mc_residues.iter().map(|r| r.chain_id.clone()).collect();
    let mc_plddt = vec![90.0_f64; 90];
    let mc_part = cluster_pae_domains(
        &mc_pae,
        Some(&mc_plddt),
        Some(&mc_sids),
        Some(&mc_cids),
        &PaeClusterConfig::default(),
    )
    .unwrap();
    let mc_kin = DomainLinkerKinematics::from_residue_records(&mc_residues, &mc_part).unwrap();
    assert_eq!(mc_kin.chain_ranges.len(), 2);

    let mc_ch0 = mc_kin.build_pdb_chain(&mc_kin.initial_state());
    let mc_rdc = simulate_domain_rdc_medium(&mc_ch0, "m1", &s_m1, &s_m1, 90, (-1, -1), 0.0, 88);
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
    mc_st.chain_quats[1] = quat_apply_world_rot_step(mc_st.chain_quats[1], [0.4, -0.3, 0.25]);
    mc_st.chain_trans[1] = [-6.0, 3.0, -2.0];

    let (_, mc_gr) =
        evaluate_domain_dock_fixed_saupe_cpu(&mc_kin, &mc_prep, &mc_st, &fixed_saupe, &weights);
    let eps_mc = 5.0e-6_f64;
    for ax in 0..3 {
        let mut dw = [0.0; 3];
        dw[ax] = eps_mc;
        let mut sp = mc_st.clone();
        let mut sm = mc_st.clone();
        sp.chain_quats[1] = quat_apply_world_rot_step(mc_st.chain_quats[1], dw);
        sm.chain_quats[1] =
            quat_apply_world_rot_step(mc_st.chain_quats[1], [-dw[0], -dw[1], -dw[2]]);
        let (ep, _) =
            evaluate_domain_dock_fixed_saupe_cpu(&mc_kin, &mc_prep, &sp, &fixed_saupe, &weights);
        let (em, _) =
            evaluate_domain_dock_fixed_saupe_cpu(&mc_kin, &mc_prep, &sm, &fixed_saupe, &weights);
        let fd_rot = (ep.total - em.total) / (2.0 * eps_mc);
        assert!(
            (mc_gr.grad_chain_rot[1][ax] - fd_rot).abs() / fd_rot.abs().max(1.0) < 1.0e-6,
            "chain_B rot grad[{ax}] mismatch: {} vs {fd_rot}",
            mc_gr.grad_chain_rot[1][ax]
        );

        let mut st_p = mc_st.clone();
        let mut st_m = mc_st.clone();
        st_p.chain_trans[1][ax] += eps_mc;
        st_m.chain_trans[1][ax] -= eps_mc;
        let (etp, _) =
            evaluate_domain_dock_fixed_saupe_cpu(&mc_kin, &mc_prep, &st_p, &fixed_saupe, &weights);
        let (etm, _) =
            evaluate_domain_dock_fixed_saupe_cpu(&mc_kin, &mc_prep, &st_m, &fixed_saupe, &weights);
        let fd_tr = (etp.total - etm.total) / (2.0 * eps_mc);
        assert!(
            (mc_gr.grad_chain_trans[1][ax] - fd_tr).abs() / fd_tr.abs().max(1.0) < 1.0e-6,
            "chain_B trans grad[{ax}] mismatch: {} vs {fd_tr}",
            mc_gr.grad_chain_trans[1][ax]
        );
    }
}

#[test]
fn test_cpu_60deg_8a_two_domain_docking_leak_free_and_f8_candor_roundtrip() {
    let (native_model, pae, part) = build_two_domain_ubq_kh3_benchmark();
    let native_kin = DomainLinkerKinematics::from_af_model(&native_model, &part).unwrap();
    let native_chain = native_kin.build_pdb_chain(&native_kin.initial_state());

    // Construct a covalent starting AF pose where Domain 2 is rotated by >= 60 deg AND its centroid
    // is translated by >= 8.0 A relative to `native_chain` (with Domain 1 fixed at 0.000 A)!
    let mut start_state = native_kin.initial_state();
    let pert_phi = [0.16, 0.22, 0.27, 0.25, 0.20, 0.16, 0.14, 0.10];
    let pert_psi = [0.14, 0.18, 0.23, 0.20, 0.16, 0.14, 0.10, 0.08];
    for (k, &idx) in native_kin.active_linker_residues.iter().enumerate() {
        start_state.phi_rad[idx] += pert_phi[k % pert_phi.len()];
        start_state.psi_rad[idx] += pert_psi[k % pert_psi.len()];
    }
    let start_chain = native_kin.build_pdb_chain(&start_state);

    // Measure exact initial Domain 2 rotation angle & centroid translation (with Domain 1 untouched!)
    let dom2_pts_nat: Vec<Vec3> = part.domains[1]
        .residue_indices
        .iter()
        .map(|&i| native_chain.residues[i].ca)
        .collect();
    let dom2_pts_start: Vec<Vec3> = part.domains[1]
        .residue_indices
        .iter()
        .map(|&i| start_chain.residues[i].ca)
        .collect();
    let (rot_d2, c_start, c_nat, _) = kabsch_transform(&dom2_pts_start, &dom2_pts_nat);
    let tr_rot = rot_d2[0][0] + rot_d2[1][1] + rot_d2[2][2];
    let rot_angle_deg = ((tr_rot - 1.0) * 0.5).clamp(-1.0, 1.0).acos().to_degrees();
    let trans_shift_a = v_dist(c_start, c_nat);
    assert!(
        rot_angle_deg >= 60.0,
        "starting AF pose must rotate Domain 2 by >= 60 deg, got {rot_angle_deg:.2} deg"
    );
    assert!(
        trans_shift_a >= 8.0,
        "starting AF pose must translate Domain 2 by >= 8.0 A, got {trans_shift_a:.2} A"
    );

    let plddts: Vec<f64> = native_model
        .plddt_per_residue
        .clone()
        .unwrap_or_else(|| vec![90.0; native_chain.residues.len()]);
    let start_cif = serialize_chain_to_af_mmcif("ubq_kh3_start_af", &start_chain, &plddts, None);
    let start_af_model =
        parse_af_structure_model_str(&start_cif, &AfIngestOptions::default()).unwrap();

    // Simulate 2 non-coaxial alignment media + lanthanide PCS tag in Domain 1 + Solomon PRE in Domain 1
    // from the TRUE native 3D architecture (`native_chain`), with realistic experimental noise!
    let s_m1 = saupe_matrix_from_pas(-13.5, 0.26, [22.0, 54.0, 108.0]);
    let s_m2 = saupe_matrix_from_pas(15.0, 0.46, [115.0, 78.0, 32.0]);
    let mut rdc_media = vec![
        simulate_domain_rdc_medium(
            &native_chain,
            "bicelle",
            &s_m1,
            &s_m1,
            79,
            (71, 78),
            0.35,
            301,
        ),
        simulate_domain_rdc_medium(&native_chain, "pf1", &s_m2, &s_m2, 79, (71, 78), 0.35, 302),
    ];
    let seq_ids = start_af_model.seq_ids();
    partition_domain_stratified_rdc_free_set(&mut rdc_media, &part, &seq_ids, 0.20, 42);

    // Verify ByPeptidePlane stratification invariant: zero split peptide planes across bond types or media!
    let mut plane_free_map = std::collections::BTreeMap::new();
    for m in &rdc_media {
        for r in &m.restraints {
            let pid = rdc_peptide_plane_id(r);
            let prev = plane_free_map.insert(pid, r.is_free);
            if let Some(was_free) = prev {
                assert_eq!(
                    was_free, r.is_free,
                    "ByPeptidePlane stratification violated on plane {pid}"
                );
            }
        }
    }

    // Paramagnetic lanthanide PCS tag in Domain 1 (`metal_pos` fixed in Domain 1's frame)
    // Lanthanide PCS tag anchored near Domain 1's C-terminal interface (`residue 66`) + 10% held-out PCSs on Domain 2
    let metal_d1 = v_add(native_chain.residues[65].ca, [2.8, 3.2, -2.5]);
    let euler_pcs = [0.45, 1.05, 1.85];
    let mut pcs_tsv = format!(
        "# metal_pos: {:.4} {:.4} {:.4}\n# metal_domain: 1\nseq_id\tatom_name\tpcs_ppm\tsigma_ppm\tis_free\n",
        metal_d1[0], metal_d1[1], metal_d1[2]
    );
    for (idx, r) in native_chain.residues.iter().enumerate() {
        if (70..78).contains(&idx) {
            continue;
        }
        if idx % 2 == 0 {
            if let Some(h) = r.h {
                if v_dist(h, metal_d1) >= 6.5 {
                    let pcs_val = predict_pcs_spherical_ppm(h, metal_d1, 32.0, 9.5, euler_pcs);
                    let is_fr = if idx >= 78 && idx % 10 == 0 { 1 } else { 0 };
                    pcs_tsv.push_str(&format!(
                        "{}\tH\t{:.5}\t0.025\t{is_fr}\n",
                        r.seq_id, pcs_val
                    ));
                }
            }
        }
    }
    let pcs_tags = parse_domain_pcs_tsv_str(&pcs_tsv).unwrap();

    // Solomon PRE Gamma2 site near Domain 1's interface (`residue 68`, first-principles SI `r^-6` at `tau_c = 10 ns, 600 MHz`)
    let pre_site_pos = v_add(native_chain.residues[67].ca, [-2.8, 2.5, 2.8]);
    let mut pre_tsv = format!(
        "# site_pos: {:.4} {:.4} {:.4}\n# site_domain: 1\n# tau_c_ns: 10.0\n# proton_mhz: 600.0\nseq_id\tatom_name\tgamma2_s1\tsigma_s1\n",
        pre_site_pos[0], pre_site_pos[1], pre_site_pos[2]
    );
    for idx in (80..native_chain.residues.len()).step_by(4) {
        let r = &native_chain.residues[idx];
        if let Some(h) = r.h {
            let dist_a = v_dist(h, pre_site_pos).max(11.0);
            let g2 = evaluate_solomon_gamma2_s1(dist_a, 10.0, 600.0, 0.5, CODATA_ELECTRON_G_FACTOR)
                .unwrap();
            pre_tsv.push_str(&format!("{}\tH\t{:.4}\t1.50\n", r.seq_id, g2));
        }
    }
    let pre_sites = parse_domain_pre_tsv_str(&pre_tsv).unwrap();

    let cfg = DomainDockConfig {
        auto_partition_rdc_free: false, // preserve our pre-partitioned set for exact corruption comparison
        auto_partition_pcs_free: false,
        ensemble_size: 5,
        n_replicas: 48,
        anneal_steps: 40,
        lbfgs_iters: 120,
        ..Default::default()
    };

    let res = dock_domains_cpu(
        &start_af_model,
        &pae,
        &rdc_media,
        &pcs_tags,
        &pre_sites,
        &[],
        &cfg,
    )
    .unwrap();

    let mut dom_mask = vec![false; native_chain.residues.len()];
    for d in &res.report.partition.domains {
        for &idx in &d.residue_indices {
            dom_mask[idx] = true;
        }
    }
    let all_dom_bb_rmsd =
        backbone_rmsd_between_chains(&res.chains[0], &native_chain, Some(&dom_mask));
    let pcs_q_free_val = res.report.pcs_q_free.expect("held-out PCS Q_free present");
    let pcs_q_work_val = res.report.pcs_q_work.expect("working PCS Q_work present");

    println!(
        "CPU T21.5 gate: start=({rot_angle_deg:.1} deg, {trans_shift_a:.1} A) -> docked BB_RMSD={all_dom_bb_rmsd:.3} A, Q_work={:.4}, ByPeptidePlane Q_free={:.4}, PCS Q_work={pcs_q_work_val:.4}, PCS Q_free={pcs_q_free_val:.4}, intra_dist={:.3} A",
        res.report.global_rdc_q_work,
        res.report.global_rdc_q_free,
        res.report.max_intra_domain_rmsd_angstrom
    );

    assert!(
        all_dom_bb_rmsd < 1.20,
        "CPU docked all-domain BB RMSD = {all_dom_bb_rmsd:.3} A (must be < 1.20 A from {rot_angle_deg:.1} deg / {trans_shift_a:.1} A start)"
    );
    assert!(
        res.report.global_rdc_q_free < 0.25,
        "ByPeptidePlane Q_free = {:.4} (must be < 0.25)",
        res.report.global_rdc_q_free
    );
    assert_eq!(
        res.report.max_intra_domain_rmsd_angstrom, 0.0,
        "intra-domain distortion must be 0.000 A"
    );
    assert_eq!(
        res.report.observability_regime,
        DipolarObservabilityRegime::FullyRestrainedTranslationalAndOrientational
    );

    // Anti-leakage proof: corrupt EVERY `is_free == true` RDC by `+9999.0 Hz` and verify docked
    // coordinates and working energies change by `0.0` (`< 1e-14`)!
    let mut corrupted_media = rdc_media.clone();
    let mut n_corrupted = 0usize;
    for m in &mut corrupted_media {
        for r in &mut m.restraints {
            if r.is_free {
                r.d_obs_hz += 9_999.0;
                n_corrupted += 1;
            }
        }
    }
    assert!(n_corrupted >= 20);
    // By default, pre-dock `21.W2:P1-6` catches `held-out Q_free == 1.0` and refuses; with
    // `allow_poor_intra_domain_fit = true`, docked coordinates & working energies change by `0.0`!
    assert!(dock_domains_cpu(
        &start_af_model,
        &pae,
        &corrupted_media,
        &pcs_tags,
        &pre_sites,
        &[],
        &cfg,
    )
    .is_err());
    let mut cfg_corrupt = cfg.clone();
    cfg_corrupt.allow_poor_intra_domain_fit = true;
    let res_corrupt = dock_domains_cpu(
        &start_af_model,
        &pae,
        &corrupted_media,
        &pcs_tags,
        &pre_sites,
        &[],
        &cfg_corrupt,
    )
    .unwrap();
    let leak_rmsd = backbone_rmsd_between_chains(&res.chains[0], &res_corrupt.chains[0], None);
    assert!(
        leak_rmsd < 1e-13,
        "held-out is_free RDC corruption altered docked coordinates by {leak_rmsd:.3e} A (must be < 1e-13 A)"
    );
    assert!((res.report.best_energy.total - res_corrupt.report.best_energy.total).abs() < 1e-12);

    // Single-medium `F8` cone-degeneracy refusal by default (`21.W2:P2-8`), plus explicit override & `.pdb` / `.cif` / `.json` / `.tsv` roundtrip
    let mut cfg_1med = cfg.clone();
    cfg_1med.n_replicas = 8;
    cfg_1med.anneal_steps = 8;
    cfg_1med.lbfgs_iters = 15;
    let err_1med = dock_domains_cpu(
        &start_af_model,
        &pae,
        &rdc_media[..1],
        &[],
        &[],
        &[],
        &cfg_1med,
    )
    .expect_err("single-medium RDC-only multi-domain docking must fail closed by default");
    assert!(
        err_1med.contains("F8") || err_1med.contains("allow_underdetermined_single_medium"),
        "unexpected single-medium error: {err_1med}"
    );

    cfg_1med.allow_underdetermined_single_medium = true;
    let res_1med = dock_domains_cpu(
        &start_af_model,
        &pae,
        &rdc_media[..1],
        &[],
        &[],
        &[],
        &cfg_1med,
    )
    .unwrap();
    assert_eq!(
        res_1med.report.observability_regime,
        DipolarObservabilityRegime::UnderdeterminedSingleMediumRdcOnly
    );
    assert_eq!(
        res_1med.report.quality_verdict,
        DomainDockQualityVerdict::UnderdeterminedSingleMediumWarning
    );
    assert!(
        res_1med
            .report
            .candor_notes
            .iter()
            .any(|n| n.contains("F8") && n.contains("cone degeneracy")),
        "expected explicit F8 single-medium cone degeneracy candor note"
    );

    let pdb_txt = format_docked_pdb_ensemble(&res_1med, "candor_test");
    let cif_txt = format_docked_mmcif_ensemble(&res_1med, "candor_test");
    let json_txt = format_domain_dock_json(&res_1med.report).unwrap();
    let tsv_txt = format_domain_dock_tsv(&res_1med.report);
    assert!(
        pdb_txt.contains("F8")
            && cif_txt.contains("F8")
            && json_txt.contains("F8")
            && tsv_txt.contains("F8")
    );

    let parsed_pdb = parse_af_structure_model_str(&pdb_txt, &AfIngestOptions::default()).unwrap();
    let parsed_cif = parse_af_structure_model_str(&cif_txt, &AfIngestOptions::default()).unwrap();
    assert_eq!(parsed_pdb.residues.len(), native_chain.residues.len());
    assert_eq!(parsed_cif.residues.len(), native_chain.residues.len());
}

// ============================================================================
// Phase 21 Slice S3 Regression Suite (`21.W2:P0-4`, `P1-4..P1-8`, `21.W3:P0-4`, `P2-8..P2-12`)
// ============================================================================

#[test]
fn test_p0_4_discontinuous_insertion_domain_zero_internal_drift_and_grad_parity() {
    // Exact insertion-domain topology on `1UBQ` (`21.W2:P0-4` probe):
    //   Domain 1 (discontinuous host): `[0..22] U [50..72]` (`seq_id 1..=22 U 51..=72`)
    //   Linker 1 (entry):              `[22..26]`
    //   Domain 2 (inserted guest):     `[26..46]` (`seq_id 27..=46`)
    //   Linker 2 (re-entry):           `[46..50]`
    //   C-tail:                        `[72..76]`
    let root = repo_root();
    let ubq = parse_pdb_first_model(root.join("data/fixtures/ubiquitin_1ubq_xray.pdb")).unwrap();
    let n = ubq.residues.len();
    assert_eq!(n, 76);

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

    let cif = serialize_chain_to_af_mmcif("ubq_insertion_p0_4", &ubq, &plddt, None);
    let model = parse_af_structure_model_str(&cif, &AfIngestOptions::default()).unwrap();
    let pae = build_synthetic_pae(n, &true_dom, 2104, 2.0, 24.5);
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
    assert_eq!(
        part.domains[0].seq_id_spans.len(),
        2,
        "D1 must have 2 discontinuous spans"
    );

    let kin = DomainLinkerKinematics::from_af_model(&model, &part).unwrap();
    assert_eq!(
        kin.rigid_blocks.len(),
        3,
        "expected 3 contiguous sequence blocks"
    );
    assert_eq!(
        kin.rigid_blocks[2].master_block_id, kin.rigid_blocks[0].master_block_id,
        "re-entry span [50..72] of D1 MUST lock to D1's master rigid-body frame!"
    );

    // 1. Single-residue `+0.5 rad` perturbation (`2.605 A` internal D1 drift on `6328a1e` -> `0.000 A`)
    let mut st_single = kin.initial_state();
    st_single.psi_rad[22] += 0.50;
    let ch_single = kin.build_pdb_chain(&st_single);
    let drift_single = kin.max_intra_domain_rmsd(&ch_single, &part);
    assert!(
        drift_single < 1.0e-12,
        "single-linker perturbation caused {drift_single:.3e} A internal drift in discontinuous D1 (must be < 1e-12 A)"
    );

    // 2. Large multi-residue perturbation across BOTH linkers (`13.65 A` internal D1 drift on `6328a1e` -> `0.000 A`)
    let mut st_large = kin.initial_state();
    for &idx in &kin.active_linker_residues {
        st_large.phi_rad[idx] += 0.65;
        st_large.psi_rad[idx] -= 0.55;
    }
    let ch_large = kin.build_pdb_chain(&st_large);
    let drift_large = kin.max_intra_domain_rmsd(&ch_large, &part);
    // Also explicitly compute Kabsch CA RMSD across ALL residues `[0..22] U [50..72]` of D1 directly!
    let d1_ref_ca: Vec<Vec3> = part.domains[0]
        .residue_indices
        .iter()
        .map(|&i| kin.ref_chain.residues[i].ca)
        .collect();
    let d1_pert_ca: Vec<Vec3> = part.domains[0]
        .residue_indices
        .iter()
        .map(|&i| ch_large.residues[i].ca)
        .collect();
    let (_, _, _, d1_kabsch_rmsd) = kabsch_transform(&d1_pert_ca, &d1_ref_ca);
    println!(
        "P0-4 insertion domain D1 ([0..22] U [50..72]) internal drift: max_intra_rmsd={drift_large:.3e} A, direct_kabsch_ca_rmsd={d1_kabsch_rmsd:.3e} A"
    );
    assert!(
        drift_large < 1.0e-12 && d1_kabsch_rmsd < 1.0e-12,
        "discontinuous D1 internal RMSD = {d1_kabsch_rmsd:.3e} A (must be < 1e-12 A)"
    );

    // 3. Analytical gradient vs central finite-difference check across BOTH linkers (`[22..26]` &
    //    `[46..50]`, including cyclic re-entry loop-closure restraint + cross-span RDC/PCS/PRE/NOE!)
    let ch0 = kin.build_pdb_chain(&kin.initial_state());
    let s1 = saupe_matrix_from_pas(-13.0, 0.31, [24.0, 52.0, 104.0]);
    let s2 = saupe_matrix_from_pas(14.5, 0.45, [112.0, 76.0, 35.0]);
    let med1 = simulate_domain_rdc_medium(&ch0, "m1", &s1, &s1, 77, (-1, -1), 0.0, 401);
    let med2 = simulate_domain_rdc_medium(&ch0, "m2", &s2, &s2, 77, (-1, -1), 0.0, 402);

    let metal_d1 = v_add(ch0.residues[60].ca, [3.0, -2.5, 2.8]);
    let mut pcs_rows = Vec::new();
    for (idx, r) in ch0.residues.iter().enumerate() {
        if true_dom[idx] != 0 {
            if let Some(h) = r.h {
                if v_dist(h, metal_d1) > 6.0 {
                    pcs_rows.push(DomainPcsRestraint {
                        tag_id: "tb_d1".to_string(),
                        chain_id: None,
                        seq_id: r.seq_id,
                        atom_name: "H".to_string(),
                        pcs_obs_ppm: predict_pcs_spherical_ppm(
                            h,
                            metal_d1,
                            28.0,
                            8.0,
                            [0.4, 1.0, 1.8],
                        ),
                        sigma_ppm: 0.03,
                        is_free: false,
                    });
                }
            }
        }
    }
    let pcs_tag = DomainPcsTagSet {
        tag_id: "tb_d1".to_string(),
        metal_pos_initial: Some(metal_d1),
        anchor_chain_id: None,
        anchor_residue_atom: None,
        host_domain_id: Some(1),
        pre_fitted_tensor: None,
        refine_metal_pos: false,
        metal_search_radius_a: None,
        restraints: pcs_rows,
    };

    let prep = PreparedDomainDockRestraints::compile(
        &kin,
        &part,
        &[med1.clone(), med2.clone()],
        std::slice::from_ref(&pcs_tag),
        &[],
        &[],
        None,
    )
    .unwrap();
    let weights = DomainDockEnergyWeights::default();
    let fixed_saupe = vec![s1, s2];
    let (_, grad) =
        evaluate_domain_dock_fixed_saupe_cpu(&kin, &prep, &st_single, &fixed_saupe, &weights);
    let eps = 1.0e-6_f64;
    for &idx in &kin.active_linker_residues {
        if kin.residues[idx].phi_active {
            let mut sp = st_single.clone();
            let mut sm = st_single.clone();
            sp.phi_rad[idx] += eps;
            sm.phi_rad[idx] -= eps;
            let (ep, _) =
                evaluate_domain_dock_fixed_saupe_cpu(&kin, &prep, &sp, &fixed_saupe, &weights);
            let (em, _) =
                evaluate_domain_dock_fixed_saupe_cpu(&kin, &prep, &sm, &fixed_saupe, &weights);
            let fd = (ep.total - em.total) / (2.0 * eps);
            let rel = (grad.grad_phi[idx] - fd).abs() / fd.abs().max(1.0);
            assert!(
                rel < 1.0e-6,
                "insertion phi[{idx}] grad {:.6e} != fd {:.6e} (rel {rel:.2e})",
                grad.grad_phi[idx],
                fd
            );
        }
        if kin.residues[idx].psi_active {
            let mut sp = st_single.clone();
            let mut sm = st_single.clone();
            sp.psi_rad[idx] += eps;
            sm.psi_rad[idx] -= eps;
            let (ep, _) =
                evaluate_domain_dock_fixed_saupe_cpu(&kin, &prep, &sp, &fixed_saupe, &weights);
            let (em, _) =
                evaluate_domain_dock_fixed_saupe_cpu(&kin, &prep, &sm, &fixed_saupe, &weights);
            let fd = (ep.total - em.total) / (2.0 * eps);
            let rel = (grad.grad_psi[idx] - fd).abs() / fd.abs().max(1.0);
            assert!(
                rel < 1.0e-6,
                "insertion psi[{idx}] grad {:.6e} != fd {:.6e} (rel {rel:.2e})",
                grad.grad_psi[idx],
                fd
            );
        }
    }

    // 4. End-to-end docking on insertion topology preserves `0.000 A` D1 internal rigidity!
    let mut st_start = kin.initial_state();
    st_start.psi_rad[23] += 0.35;
    st_start.phi_rad[24] -= 0.30;
    let start_chain = kin.build_pdb_chain(&st_start);
    let start_cif = serialize_chain_to_af_mmcif("ubq_insertion_start", &start_chain, &plddt, None);
    let start_af = parse_af_structure_model_str(&start_cif, &AfIngestOptions::default()).unwrap();
    let cfg = DomainDockConfig {
        ensemble_size: 2,
        n_replicas: 16,
        anneal_steps: 24,
        lbfgs_iters: 80,
        ..Default::default()
    };
    let docked =
        dock_domains_cpu(&start_af, &pae, &[med1, med2], &[pcs_tag], &[], &[], &cfg).unwrap();
    assert!(
        docked.report.max_intra_domain_rmsd_angstrom < 1.0e-12,
        "docked insertion domain max_intra_domain_rmsd = {:.3e} A",
        docked.report.max_intra_domain_rmsd_angstrom
    );
    let d1_docked_ca: Vec<Vec3> = docked.report.partition.domains[0]
        .residue_indices
        .iter()
        .map(|&i| docked.chains[0].residues[i].ca)
        .collect();
    let (_, _, _, d1_vs_ubq_rmsd) = kabsch_transform(&d1_docked_ca, &d1_ref_ca);
    assert!(
        d1_vs_ubq_rmsd < 1.0e-10,
        "docked discontinuous D1 drifted by {d1_vs_ubq_rmsd:.3e} A vs original 1UBQ!"
    );
}

#[test]
fn test_p1_4_flexible_linker_and_tail_rdcs_excluded_from_saupe_and_free_set() {
    let (mut native_model, pae, _) = build_two_domain_ubq_kh3_benchmark();
    // Mark C-terminal tail residues (`140..144`) as low-confidence (`pLDDT = 52 < 70`)
    if let Some(ref mut pl) = native_model.plddt_per_residue {
        for p in pl.iter_mut().skip(140) {
            *p = 52.0;
        }
    }
    for r in native_model.residues.iter_mut().skip(140) {
        r.plddt = Some(52.0);
    }
    let part = nmr_core::domain_dock::cluster_pae_domains_for_model(
        &native_model,
        &pae,
        &PaeClusterConfig::default(),
    )
    .unwrap();

    let kin = DomainLinkerKinematics::from_af_model(&native_model, &part).unwrap();
    let ch0 = kin.build_pdb_chain(&kin.initial_state());
    let s1 = saupe_matrix_from_pas(-13.5, 0.26, [22.0, 54.0, 108.0]);
    let s2 = saupe_matrix_from_pas(15.0, 0.46, [115.0, 78.0, 32.0]);

    // Clean rigid-only RDCs (`seq_id 1..=70` and `79..=140`)
    let clean_m1 = simulate_domain_rdc_medium(&ch0, "m1", &s1, &s1, 79, (71, 78), 0.20, 501);
    let clean_m2 = simulate_domain_rdc_medium(&ch0, "m2", &s2, &s2, 79, (71, 78), 0.20, 502);
    let mut clean_media = vec![clean_m1, clean_m2];
    for m in &mut clean_media {
        m.restraints
            .retain(|r| r.seq_id_1 <= 140 && r.seq_id_2 <= 140);
    }

    // Contaminated media: append 0.2x motionally averaged + sign-corrupted linker (`71..=78`),
    // low-pLDDT tail (`141..=144`), and cross-boundary RDCs!
    let mut dirty_media = clean_media.clone();
    for (mi, m) in dirty_media.iter_mut().enumerate() {
        let s = if mi == 0 { &s1 } else { &s2 };
        for idx in (70..78).chain(140..144) {
            let r = &ch0.residues[idx];
            if let Some(h) = r.h {
                let u = v_unit(v_sub(h, r.n));
                let d_static = calc_rdc_from_unit_vec(u, RdcBondType::NH.default_d_max_hz(), s);
                // Strongly motionally attenuated (`0.2x`) + offset that would corrupt a static Saupe fit!
                m.restraints.push(RdcRestraint {
                    medium_id: m.medium_id.clone(),
                    seq_id_1: r.seq_id,
                    res_name_1: r.res_name.clone(),
                    atom_name_1: "N".to_string(),
                    seq_id_2: r.seq_id,
                    res_name_2: r.res_name.clone(),
                    atom_name_2: "H".to_string(),
                    bond_type: RdcBondType::NH,
                    d_obs_hz: -0.20 * d_static + 12.0,
                    sigma_hz: 0.25,
                    d_max_hz: RdcBondType::NH.default_d_max_hz(),
                    is_free: true, // even if user pre-marked `is_free = true`, non-rigid RDCs must be excluded!
                });
            }
        }
    }

    let seq_ids = native_model.seq_ids();
    partition_domain_stratified_rdc_free_set(&mut clean_media, &part, &seq_ids, 0.20, 42);
    partition_domain_stratified_rdc_free_set(&mut dirty_media, &part, &seq_ids, 0.20, 42);

    // Zero linker or low-pLDDT RDCs may remain marked `is_free` after partitioning
    for m in &dirty_media {
        for r in &m.restraints {
            if (71..=78).contains(&r.seq_id_1) {
                assert!(
                    !r.is_free,
                    "flexible linker RDC at seq_id {} must never enter ByPeptidePlane RdcFree set",
                    r.seq_id_1
                );
            }
        }
    }

    let prep_clean =
        PreparedDomainDockRestraints::compile(&kin, &part, &clean_media, &[], &[], &[], None)
            .unwrap();
    let prep_dirty =
        PreparedDomainDockRestraints::compile(&kin, &part, &dirty_media, &[], &[], &[], None)
            .unwrap();
    assert_eq!(prep_clean.rdc_terms.len(), prep_dirty.rdc_terms.len());

    // Full docking invariance (`0.000 A` coordinate difference, identical Q_work & Q_free)
    let metal_d1 = v_add(ch0.residues[65].ca, [2.8, 3.2, -2.5]);
    let mut pcs_tsv = format!(
        "# metal_pos: {:.4} {:.4} {:.4}\n# metal_domain: 1\nseq_id\tatom_name\tpcs_ppm\tsigma_ppm\n",
        metal_d1[0], metal_d1[1], metal_d1[2]
    );
    for (idx, r) in ch0.residues.iter().enumerate() {
        if !(70..78).contains(&idx) && idx < 140 && idx % 3 == 0 {
            if let Some(h) = r.h {
                if v_dist(h, metal_d1) >= 6.5 {
                    let pv = predict_pcs_spherical_ppm(h, metal_d1, 30.0, 9.0, [0.4, 1.0, 1.8]);
                    pcs_tsv.push_str(&format!("{}\tH\t{:.5}\t0.03\n", r.seq_id, pv));
                }
            }
        }
    }
    let pcs_tags = parse_domain_pcs_tsv_str(&pcs_tsv).unwrap();

    let cfg = DomainDockConfig {
        auto_partition_rdc_free: false,
        ensemble_size: 2,
        n_replicas: 12,
        anneal_steps: 16,
        lbfgs_iters: 40,
        ..Default::default()
    };
    let r_clean =
        dock_domains_cpu(&native_model, &pae, &clean_media, &pcs_tags, &[], &[], &cfg).unwrap();
    let r_dirty =
        dock_domains_cpu(&native_model, &pae, &dirty_media, &pcs_tags, &[], &[], &cfg).unwrap();
    assert!(
        r_dirty.report.n_excluded_flexible_rdcs >= 20,
        "expected >= 20 non-rigid/low-pLDDT RDCs excluded, got {}",
        r_dirty.report.n_excluded_flexible_rdcs
    );
    let diff_rmsd = backbone_rmsd_between_chains(&r_clean.chains[0], &r_dirty.chains[0], None);
    assert!(
        diff_rmsd < 1.0e-13,
        "motionally averaged linker/tail RDCs perturbed docked structure by {diff_rmsd:.3e} A!"
    );
    assert!((r_clean.report.global_rdc_q_work - r_dirty.report.global_rdc_q_work).abs() < 1e-14);
    assert!((r_clean.report.global_rdc_q_free - r_dirty.report.global_rdc_q_free).abs() < 1e-14);
}

#[test]
fn test_p1_5_honest_pcs_q_free_stratified_holdout_and_none_when_no_free() {
    let (native_model, pae, part) = build_two_domain_ubq_kh3_benchmark();
    let kin = DomainLinkerKinematics::from_af_model(&native_model, &part).unwrap();
    let ch0 = kin.build_pdb_chain(&kin.initial_state());

    let s1 = saupe_matrix_from_pas(-13.0, 0.28, [24.0, 52.0, 105.0]);
    let s2 = saupe_matrix_from_pas(14.5, 0.44, [114.0, 76.0, 34.0]);
    let media = vec![
        simulate_domain_rdc_medium(&ch0, "m1", &s1, &s1, 79, (71, 78), 0.25, 601),
        simulate_domain_rdc_medium(&ch0, "m2", &s2, &s2, 79, (71, 78), 0.25, 602),
    ];

    let metal_d1 = v_add(ch0.residues[65].ca, [2.8, 3.2, -2.5]);
    let mut pcs_tsv = format!(
        "# metal_pos: {:.4} {:.4} {:.4}\n# metal_domain: 1\nseq_id\tatom_name\tpcs_ppm\tsigma_ppm\n",
        metal_d1[0], metal_d1[1], metal_d1[2]
    );
    for (idx, r) in ch0.residues.iter().enumerate() {
        if !(70..78).contains(&idx) && idx % 2 == 0 {
            if let Some(h) = r.h {
                if v_dist(h, metal_d1) >= 6.5 {
                    let pv = predict_pcs_spherical_ppm(h, metal_d1, 30.0, 9.0, [0.4, 1.0, 1.8]);
                    pcs_tsv.push_str(&format!("{}\tH\t{:.5}\t0.03\n", r.seq_id, pv));
                }
            }
        }
    }
    let pcs_tags = parse_domain_pcs_tsv_str(&pcs_tsv).unwrap();

    // Case A: `pcs_free_fraction = 0.0` (0 held-out PCSs) -> `pcs_q_free` MUST be `None`, NEVER `pcs_q_work`!
    let cfg_no_free = DomainDockConfig {
        pcs_free_fraction: 0.0,
        auto_partition_pcs_free: false,
        ensemble_size: 2,
        n_replicas: 8,
        anneal_steps: 10,
        lbfgs_iters: 25,
        ..Default::default()
    };
    let res_no_free = dock_domains_cpu(
        &native_model,
        &pae,
        &media,
        &pcs_tags,
        &[],
        &[],
        &cfg_no_free,
    )
    .unwrap();
    assert_eq!(
        res_no_free.report.pcs_q_free, None,
        "pcs_q_free MUST be None when n_pcs_free == 0 (never alias pcs_q_work!)"
    );
    assert!(res_no_free.report.pcs_q_work.is_some());
    let js_no_free = format_domain_dock_json(&res_no_free.report).unwrap();
    assert!(
        js_no_free.contains("\"pcs_q_free\": null") || js_no_free.contains("\"pcs_q_free\":null")
    );
    assert!(js_no_free.contains("\"pcs_q_work\":"));

    // Case B: Default stratified 20% PCS holdout + `+999.0 ppm` corruption invariance proof!
    let cfg_holdout = DomainDockConfig {
        pcs_free_fraction: 0.20,
        auto_partition_pcs_free: true,
        ensemble_size: 2,
        n_replicas: 10,
        anneal_steps: 12,
        lbfgs_iters: 30,
        ..Default::default()
    };
    let res_hold = dock_domains_cpu(
        &native_model,
        &pae,
        &media,
        &pcs_tags,
        &[],
        &[],
        &cfg_holdout,
    )
    .unwrap();
    let qf_clean = res_hold
        .report
        .pcs_q_free
        .expect("20% held-out PCS Q_free must be Some");
    assert!(
        qf_clean < 0.15,
        "clean PCS Q_free = {qf_clean:.4} must be < 0.15"
    );

    // Pre-partition explicitly so we can corrupt ONLY `is_free == true` PCSs by `+999.0 ppm`
    let mut part_tags = pcs_tags.clone();
    let cseq = ChainSeqIndex::from_kin(&kin).unwrap();
    partition_domain_stratified_pcs_free_set(&mut part_tags, &cseq, &part, None, 70.0, 0.20, 42)
        .unwrap();
    let mut cfg_pre_part = cfg_holdout.clone();
    cfg_pre_part.auto_partition_pcs_free = false;
    let res_base = dock_domains_cpu(
        &native_model,
        &pae,
        &media,
        &part_tags,
        &[],
        &[],
        &cfg_pre_part,
    )
    .unwrap();
    let mut n_corrupt_pcs = 0usize;
    for t in &mut part_tags {
        for r in &mut t.restraints {
            if r.is_free {
                r.pcs_obs_ppm += 999.0;
                n_corrupt_pcs += 1;
            }
        }
    }
    assert!(n_corrupt_pcs >= 8);
    let res_corrupt = dock_domains_cpu(
        &native_model,
        &pae,
        &media,
        &part_tags,
        &[],
        &[],
        &cfg_pre_part,
    )
    .unwrap();
    let pcs_leak_rmsd =
        backbone_rmsd_between_chains(&res_base.chains[0], &res_corrupt.chains[0], None);
    assert!(
        pcs_leak_rmsd < 1.0e-13,
        "corrupting held-out PCSs altered coordinates by {pcs_leak_rmsd:.3e} A"
    );
    assert!(res_corrupt.report.pcs_q_free.unwrap() > 0.95);
    assert!(
        (res_base.report.pcs_q_work.unwrap() - res_corrupt.report.pcs_q_work.unwrap()).abs()
            < 1.0e-13
    );
}

#[test]
fn test_p1_6_pre_dock_intra_domain_check_refusal_override_and_tolman_coord_noise() {
    let (native_model, pae, part) = build_two_domain_ubq_kh3_benchmark();
    let kin = DomainLinkerKinematics::from_af_model(&native_model, &part).unwrap();
    let ch0 = kin.build_pdb_chain(&kin.initial_state());

    let s1 = saupe_matrix_from_pas(-13.5, 0.26, [22.0, 54.0, 108.0]);
    let s2 = saupe_matrix_from_pas(15.0, 0.46, [115.0, 78.0, 32.0]);

    // 1. Corrupt Domain 2 RDCs (`seq_id >= 79`) with random sign-flipped values so Domain 2 fails
    //    its pre-dock intra-domain `ByPeptidePlane Q_free > 0.50` check (`21.W2:P1-6`)!
    let mut bad_m1 = simulate_domain_rdc_medium(&ch0, "m1", &s1, &s1, 79, (71, 78), 0.20, 701);
    let mut bad_m2 = simulate_domain_rdc_medium(&ch0, "m2", &s2, &s2, 79, (71, 78), 0.20, 702);
    let mut rng = Rng::new(703);
    for m in [&mut bad_m1, &mut bad_m2] {
        for r in &mut m.restraints {
            if r.seq_id_1 >= 79 {
                r.d_obs_hz = -r.d_obs_hz + (rng.f64() - 0.5) * 28.0;
            }
        }
    }
    let metal_d1 = v_add(ch0.residues[65].ca, [2.8, 3.2, -2.5]);
    let mut pcs_tsv = format!(
        "# metal_pos: {:.4} {:.4} {:.4}\n# metal_domain: 1\nseq_id\tatom_name\tpcs_ppm\tsigma_ppm\n",
        metal_d1[0], metal_d1[1], metal_d1[2]
    );
    for (idx, r) in ch0.residues.iter().enumerate() {
        if !(70..78).contains(&idx) && idx % 3 == 0 {
            if let Some(h) = r.h {
                if v_dist(h, metal_d1) >= 6.5 {
                    let pv = predict_pcs_spherical_ppm(h, metal_d1, 30.0, 9.0, [0.4, 1.0, 1.8]);
                    pcs_tsv.push_str(&format!("{}\tH\t{:.5}\t0.03\n", r.seq_id, pv));
                }
            }
        }
    }
    let pcs_tags = parse_domain_pcs_tsv_str(&pcs_tsv).unwrap();

    let cfg_default = DomainDockConfig {
        ensemble_size: 2,
        n_replicas: 8,
        anneal_steps: 8,
        lbfgs_iters: 20,
        ..Default::default()
    };
    let err_bad = dock_domains_cpu(
        &native_model,
        &pae,
        &[bad_m1.clone(), bad_m2.clone()],
        &pcs_tags,
        &[],
        &[],
        &cfg_default,
    )
    .expect_err("corrupted Domain 2 intra-domain geometry/RDCs must be refused by default");
    assert!(
        err_bad.contains("21.W2:P1-6") && err_bad.contains("Q_free"),
        "unexpected error on poor intra-domain fit: {err_bad}"
    );

    let mut cfg_override = cfg_default.clone();
    cfg_override.allow_poor_intra_domain_fit = true;
    let res_override = dock_domains_cpu(
        &native_model,
        &pae,
        &[bad_m1, bad_m2],
        &pcs_tags,
        &[],
        &[],
        &cfg_override,
    )
    .unwrap();
    assert_eq!(
        res_override.report.quality_verdict,
        DomainDockQualityVerdict::IntraDomainMismatchWarning
    );
    assert!(
        res_override
            .report
            .pre_dock_domain_checks
            .iter()
            .any(|c| c.domain_id == 2 && !c.passed),
        "expected Domain 2 pre-dock check to record passed == false"
    );

    // 2. Tolman GDO coordinate-noise floor (`A8`): perturb individual N/H coordinates by `0.25 A`
    //    isotropic Gaussian noise under strictly RIGID co-tumbling (`S_D2 == S_D1`), while declaring
    //    tiny `sigma_hz = 0.05 Hz` spectroscopic error — without the coordinate-noise + excess chi^2_nu
    //    floor, MC uncertainty would be under-estimated by ~10x!
    let mut noisy_ch = ch0.clone();
    for r in &mut noisy_ch.residues {
        if let Some(ref mut h) = r.h {
            h[0] += (rng.f64() - 0.5) * 0.14;
            h[1] += (rng.f64() - 0.5) * 0.14;
            h[2] += (rng.f64() - 0.5) * 0.14;
        }
    }
    let mut m_tight = simulate_domain_rdc_medium(&ch0, "m1", &s1, &s1, 79, (71, 78), 0.05, 709);
    for r in &mut m_tight.restraints {
        r.sigma_hz = 0.05;
    }
    let gdo_noisy = evaluate_tolman_domain_gdo(&noisy_ch, &part, &[m_tight], 200, 710);
    assert_eq!(
        gdo_noisy.regime,
        DomainMotionRegime::RigidlyCoupledDomains,
        "0.25 A intra-domain coordinate noise on rigid co-tumbling domains must NOT trigger false DecoupledInterDomainMotion: {gdo_noisy:?}"
    );
}

#[test]
fn test_p1_8_multi_chain_overlapping_seq_id_homodimer_rdc_pcs_pre_noe() {
    // Two-chain homodimer (`A:1..=70` and `B:1..=70` — identical overlapping residue numbers!)
    let root = repo_root();
    let ubq = parse_pdb_first_model(root.join("data/fixtures/ubiquitin_1ubq_xray.pdb")).unwrap();
    let mut chain_a = PdbChain {
        residues: ubq.residues[..70].to_vec(),
    };
    for (i, r) in chain_a.residues.iter_mut().enumerate() {
        r.chain_id = 'A';
        r.seq_id = (i + 1) as i32;
    }

    // Native Chain B: rotated by 35 deg and translated `[28.0, 6.0, -4.0]` away from Chain A so zero clashes,
    // with overlapping `seq_id = 1..=70`!
    let c_a = {
        let n_f = chain_a.residues.len() as f64;
        let mut s = [0.0; 3];
        for r in &chain_a.residues {
            s = v_add(s, r.ca);
        }
        v_scale(s, 1.0 / n_f)
    };
    let c_b_nat = v_add(c_a, [34.0, 4.0, -2.0]);
    let q_b_nat = quat_apply_world_rot_step([1.0, 0.0, 0.0, 0.0], [0.35, -0.42, 0.28]);
    let rot_m = {
        let [w, x, y, z] = q_b_nat;
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
            ],
            [
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - w * x),
            ],
            [
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ]
    };
    let map_pt = |p: Vec3| -> Vec3 {
        let u = v_sub(p, c_a);
        [
            rot_m[0][0] * u[0] + rot_m[0][1] * u[1] + rot_m[0][2] * u[2] + c_b_nat[0],
            rot_m[1][0] * u[0] + rot_m[1][1] * u[1] + rot_m[1][2] * u[2] + c_b_nat[1],
            rot_m[2][0] * u[0] + rot_m[2][1] * u[1] + rot_m[2][2] * u[2] + c_b_nat[2],
        ]
    };
    let mut chain_b_nat = chain_a.clone();
    for (i, r) in chain_b_nat.residues.iter_mut().enumerate() {
        r.chain_id = 'B';
        r.seq_id = (i + 1) as i32; // OVERLAPPING `1..=70`!
        r.n = map_pt(r.n);
        r.h = r.h.map(map_pt);
        r.ca = map_pt(r.ca);
        r.c = map_pt(r.c);
        r.o = map_pt(r.o);
        r.cb = r.cb.map(map_pt);
    }

    let plddt70 = vec![92.0_f64; 70];
    let cif_a =
        serialize_chain_to_af_mmcif("homodimer_a", &chain_a, &plddt70, Some(&vec!["A"; 70]));
    let cif_b_nat = serialize_chain_to_af_mmcif(
        "homodimer_b_nat",
        &chain_b_nat,
        &plddt70,
        Some(&vec!["B"; 70]),
    );
    let model_a = parse_af_structure_model_str(&cif_a, &AfIngestOptions::default()).unwrap();
    let model_b_nat =
        parse_af_structure_model_str(&cif_b_nat, &AfIngestOptions::default()).unwrap();

    let mut true_dom = vec![1usize; 70];
    true_dom.extend(vec![2usize; 70]);
    let pae = build_synthetic_pae(140, &true_dom, 801, 2.0, 25.0);

    // 1. Un-chained RDC on overlapping `seq_id = 12` MUST fail closed (`Err`) instead of colliding!
    let unchained_rdc_tsv = "medium_id\tseq_id_1\tres_name_1\tatom_name_1\tseq_id_2\tres_name_2\tatom_name_2\tbond_type\td_obs_hz\tsigma_hz\td_max_hz\nm1\t12\tTHR\tN\t12\tTHR\tH\t1D_NH\t5.0\t0.5\t21655.82\n";
    let unchained_media = parse_domain_rdc_tsv_str(unchained_rdc_tsv).unwrap();
    let ambig_err = dock_domains_multi_chain_cpu(
        &[model_a.clone(), model_b_nat.clone()],
        &pae,
        &unchained_media,
        &[],
        &[],
        &[],
        &DomainDockConfig::default(),
    )
    .expect_err("un-chained restraint on overlapping multi-chain seq_id must fail closed");
    assert!(
        ambig_err.contains("ambiguous") || ambig_err.contains("multiple chains"),
        "expected explicit multi-chain ambiguity error, got: {ambig_err}"
    );

    // 2. Chain-qualified `A:` and `B:` RDC + PCS + PRE + NOE restraints resolve cleanly and dock
    //    a perturbed Chain B starting pose (`rot >= 45 deg, trans >= 7 A`) back to `< 1.2 A`!
    let s1 = saupe_matrix_from_pas(-13.0, 0.28, [24.0, 52.0, 105.0]);
    let s2 = saupe_matrix_from_pas(14.5, 0.44, [114.0, 76.0, 34.0]);
    let mut rdc_tsv = String::from(
        "medium_id\tchain_id_1\tseq_id_1\tres_name_1\tatom_name_1\tchain_id_2\tseq_id_2\tres_name_2\tatom_name_2\tbond_type\td_obs_hz\tsigma_hz\td_max_hz\n",
    );
    for (mname, smat) in [("m1", &s1), ("m2", &s2)] {
        for (cid, ch) in [("A", &chain_a), ("B", &chain_b_nat)] {
            for r in &ch.residues {
                if let Some(h) = r.h {
                    let d = calc_rdc_from_unit_vec(
                        v_unit(v_sub(h, r.n)),
                        RdcBondType::NH.default_d_max_hz(),
                        smat,
                    );
                    rdc_tsv.push_str(&format!(
                        "{mname}\t{cid}\t{}\t{}\tN\t{cid}\t{}\t{}\tH\t1D_NH\t{d:.4}\t0.25\t21655.82\n",
                        r.seq_id, r.res_name, r.seq_id, r.res_name
                    ));
                }
            }
        }
    }
    let rdc_media = parse_domain_rdc_tsv_str(&rdc_tsv).unwrap();

    // Find the `+x`-most interface residue on Chain A so lanthanide PCS tag, Solomon PRE site, and
    // inter-chain NOEs sit right at the A:B dimer interface (`8..22 A`, breaking all 4-fold PAS symmetry!)
    let iface_a = chain_a
        .residues
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.ca[0].total_cmp(&b.ca[0]))
        .unwrap()
        .0;
    let metal_a = v_add(chain_a.residues[iface_a].ca, [2.2, 2.5, -2.0]);
    let mut pcs_tsv = format!(
        "# metal_pos: {:.4} {:.4} {:.4}\n# metal_domain: 1\nchain_id\tseq_id\tatom_name\tpcs_ppm\tsigma_ppm\n",
        metal_a[0], metal_a[1], metal_a[2]
    );
    for (cid, ch) in [("A", &chain_a), ("B", &chain_b_nat)] {
        for r in ch.residues.iter().step_by(2) {
            if let Some(h) = r.h {
                if v_dist(h, metal_a) >= 6.5 {
                    let pv = predict_pcs_spherical_ppm(h, metal_a, 34.0, 10.0, [0.4, 1.0, 1.8]);
                    pcs_tsv.push_str(&format!("{cid}\t{}\tH\t{pv:.5}\t0.015\n", r.seq_id));
                }
            }
        }
    }
    let pcs_tags = parse_domain_pcs_tsv_str(&pcs_tsv).unwrap();

    let pre_pos_a = v_add(chain_a.residues[iface_a].ca, [1.8, -2.5, 2.2]);
    let mut pre_tsv = format!(
        "# site_pos: {:.4} {:.4} {:.4}\n# site_domain: 1\n# tau_c_ns: 10.0\n# proton_mhz: 600.0\nseq_id\tatom_name\tgamma2_s1\tsigma_s1\n",
        pre_pos_a[0], pre_pos_a[1], pre_pos_a[2]
    );
    for r in chain_b_nat.residues.iter().step_by(3) {
        if let Some(h) = r.h {
            let d = v_dist(h, pre_pos_a).max(11.0);
            if d <= 24.0 {
                let g2 = evaluate_solomon_gamma2_s1(d, 10.0, 600.0, 0.5, CODATA_ELECTRON_G_FACTOR)
                    .unwrap();
                pre_tsv.push_str(&format!("B:{}\tH\t{g2:.4}\t1.00\n", r.seq_id));
            }
        }
    }
    let pre_sites = parse_domain_pre_tsv_str(&pre_tsv).unwrap();

    // Select top 6 genuinely close interfacial `(A:ia, B:ib)` pairs across distinct residues
    let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
    for (ia, ra) in chain_a.residues.iter().enumerate().step_by(3) {
        for (ib, rb) in chain_b_nat.residues.iter().enumerate().step_by(3) {
            pairs.push((v_dist(ra.ca, rb.ca), ia, ib));
        }
    }
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut noe_tsv = String::from("res_a\tatom_a\tres_b\tatom_b\tlower_a\tupper_a\n");
    for &(d_ab, ia, ib) in pairs.iter().take(6) {
        noe_tsv.push_str(&format!(
            "A:{}\tHA\tB:{}\tHA\t{:.2}\t{:.2}\n",
            chain_a.residues[ia].seq_id,
            chain_b_nat.residues[ib].seq_id,
            (d_ab - 2.2).max(2.0),
            d_ab + 2.2
        ));
    }
    let mut all_recs = model_a.residues.clone();
    for r in &model_b_nat.residues {
        let mut rb = r.clone();
        rb.chain_id = "B".to_string();
        all_recs.push(rb);
    }
    let noe_adrs = parse_domain_noe_tsv_str(&noe_tsv, &all_recs).unwrap();

    // Displace Chain B by ~50 deg rotation and ~9 A translation for starting pose
    let q_b_start = quat_apply_world_rot_step([1.0, 0.0, 0.0, 0.0], [-0.55, 0.48, -0.35]);
    let rot_s = {
        let [w, x, y, z] = q_b_start;
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
            ],
            [
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - w * x),
            ],
            [
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ]
    };
    let c_b_start = v_add(c_b_nat, [7.0, 6.0, -4.0]);
    let mut chain_b_start = chain_b_nat.clone();
    for r in &mut chain_b_start.residues {
        let map_s = |p: Vec3| -> Vec3 {
            let u = v_sub(p, c_b_nat);
            [
                rot_s[0][0] * u[0] + rot_s[0][1] * u[1] + rot_s[0][2] * u[2] + c_b_start[0],
                rot_s[1][0] * u[0] + rot_s[1][1] * u[1] + rot_s[1][2] * u[2] + c_b_start[1],
                rot_s[2][0] * u[0] + rot_s[2][1] * u[1] + rot_s[2][2] * u[2] + c_b_start[2],
            ]
        };
        r.n = map_s(r.n);
        r.h = r.h.map(map_s);
        r.ca = map_s(r.ca);
        r.c = map_s(r.c);
        r.o = map_s(r.o);
        r.cb = r.cb.map(map_s);
    }
    let cif_b_start = serialize_chain_to_af_mmcif(
        "homodimer_b_start",
        &chain_b_start,
        &plddt70,
        Some(&vec!["B"; 70]),
    );
    let model_b_start =
        parse_af_structure_model_str(&cif_b_start, &AfIngestOptions::default()).unwrap();

    let cfg = DomainDockConfig {
        ensemble_size: 3,
        n_replicas: 48,
        anneal_steps: 40,
        lbfgs_iters: 120,
        ..Default::default()
    };
    let docked = dock_domains_multi_chain_cpu(
        &[model_a, model_b_start],
        &pae,
        &rdc_media,
        &pcs_tags,
        &pre_sites,
        &noe_adrs,
        &cfg,
    )
    .unwrap();

    let mut native_combined = chain_a.clone();
    native_combined
        .residues
        .extend(chain_b_nat.residues.clone());
    let homodimer_rmsd =
        nmr_core::domain_dock::multi_chain_backbone_rmsd(&docked.chains[0], &native_combined, None);
    println!(
        "P1-8 homodimer A:1..70 + B:1..70 docked BB RMSD = {homodimer_rmsd:.3} A, Q_free = {:.4}, ens_rmsd = {:.3} A",
        docked.report.global_rdc_q_free,
        docked.report.ensemble_pairwise_bb_rmsd_angstrom
    );
    assert!(
        homodimer_rmsd < 1.25,
        "overlapping-seq_id homodimer docked BB RMSD = {homodimer_rmsd:.3} A (must be < 1.25 A)"
    );
    assert!(docked.report.max_intra_domain_rmsd_angstrom < 1.0e-12);
}

#[test]
fn test_w3_p0_4_and_p2_9_solomon_si_constant_mandatory_tau_c_tether_and_unsoftened_kernel() {
    // 1. Exact CODATA SI constant `K_SOLOMON_PROTON_NITROXIDE_A6_S2 = 1.2339e16 A^6 s^-2`
    assert!(
        (K_SOLOMON_PROTON_NITROXIDE_A6_S2 - 1.2339e16).abs() / 1.2339e16 < 1.0e-3,
        "K_SOLOMON_PROTON_NITROXIDE_A6_S2 = {K_SOLOMON_PROTON_NITROXIDE_A6_S2:.6e} (expected ~1.2339e16 A^6 s^-2)"
    );
    // General spin S scaling: Mn2+/Gd3+ (`S = 5/2`, `S(S+1) = 35/4`) vs nitroxide (`S = 1/2`, `S(S+1) = 3/4`) = `35/3`x
    let k_s12 = compute_solomon_k_pre_a6_s1(10.0, 600.0, 0.5, CODATA_ELECTRON_G_FACTOR).unwrap();
    let k_s52 = compute_solomon_k_pre_a6_s1(10.0, 600.0, 2.5, CODATA_ELECTRON_G_FACTOR).unwrap();
    assert!(((k_s52 / k_s12) - (35.0 / 3.0)).abs() < 1.0e-12);

    // 2. Mandatory physics unit test (`21.W3:P0-4`): `Gamma_2 = 20 s^-1` at `tau_c = 10 ns`, `600 MHz`
    //    MUST invert to `r ≈ 17.07 A` (NOT `9.22 A` from `1.23e7` on `6328a1e`!)
    let (r_20, r_lo_20, r_hi_20) =
        solomon_gamma2_to_distance_bounds(20.0, 2.0, k_s12, 0.0).unwrap();
    println!("W3:P0-4 Gamma2=20+-2 s^-1 @ tau_c=10 ns, 600 MHz -> r = {r_20:.3} A [{r_lo_20:.3}..{r_hi_20:.3} A]");
    assert!(
        (r_20 - 17.07).abs() < 0.15,
        "Gamma2 = 20 s^-1 at tau_c = 10 ns, 600 MHz must give r ~ 17.07 A, got {r_20:.3} A"
    );

    // 3. Fail-closed `Err` when `tau_c_ns` or `proton_mhz` is omitted on Gamma2 PRE TSV, or when
    //    retired `# k_pre:` header is supplied!
    let missing_tau_tsv = "# site_pos: 0.0 0.0 0.0\n# proton_mhz: 600.0\nseq_id\tatom_name\tgamma2_s1\tsigma_s1\n10\tH\t20.0\t2.0\n";
    assert!(parse_domain_pre_tsv_str(missing_tau_tsv).is_err());
    let missing_mhz_tsv = "# site_pos: 0.0 0.0 0.0\n# tau_c_ns: 10.0\nseq_id\tatom_name\tgamma2_s1\tsigma_s1\n10\tH\t20.0\t2.0\n";
    assert!(parse_domain_pre_tsv_str(missing_mhz_tsv).is_err());
    let retired_k_tsv = "# site_pos: 0.0 0.0 0.0\n# k_pre: 1.23e7\nseq_id\tatom_name\tgamma2_s1\tsigma_s1\n10\tH\t20.0\t2.0\n";
    assert!(parse_domain_pre_tsv_str(retired_k_tsv).is_err());

    // CLI defaults fill missing header cleanly when supplied via `parse_domain_pre_tsv_str_with_defaults`
    let with_cli_defaults = parse_domain_pre_tsv_str_with_defaults(
        "# site_pos: 0.0 0.0 0.0\n# mtsl_tether_allowance_a: 6.0\nseq_id\tatom_name\tgamma2_s1\tsigma_s1\n10\tH\t20.0\t2.0\n",
        Some(10.0),
        Some(600.0),
        None,
    )
    .unwrap();
    assert_eq!(with_cli_defaults[0].mtsl_tether_allowance_a, Some(6.0));

    // 4. `P2-9` zero softening bias at `r = 10.000 A` AND `C^1` core continuity at `r_core = 3.5 A`,
    //    plus `mtsl_tether_allowance_a` allowance check!
    let (model, _, part) = build_two_domain_ubq_kh3_benchmark();
    let kin = DomainLinkerKinematics::from_af_model(&model, &part).unwrap();
    let ch0 = kin.build_pdb_chain(&kin.initial_state());
    let (slots0, _) = kin.forward_slots_and_block_frames(&kin.initial_state());
    let h95 = slots0[94 * 7 + 1].0;
    let r_true = 10.000_f64;
    // Place paramagnetic center on Domain 1 exactly `10.000 A` from `H_95` along `+x`
    let site_exact = [h95[0] - r_true, h95[1], h95[2]];
    let g2_exact_10a =
        evaluate_solomon_gamma2_s1(r_true, 10.0, 600.0, 0.5, CODATA_ELECTRON_G_FACTOR).unwrap();
    let pre_exact = DomainPreSiteSet {
        site_id: "exact10".to_string(),
        site_pos_initial: Some(site_exact),
        anchor_chain_id: None,
        anchor_residue_atom: None,
        host_domain_id: Some(1),
        tau_c_ns: Some(10.0),
        proton_mhz: Some(600.0),
        mtsl_tether_allowance_a: None,
        restraints: vec![DomainPreRestraint {
            site_id: "exact10".to_string(),
            chain_id: None,
            seq_id: ch0.residues[94].seq_id,
            atom_name: "H".to_string(),
            kind: PreRestraintKind::Gamma2Rate {
                gamma2_obs_s1: g2_exact_10a,
                sigma_s1: 1.0,
                k_pre_a6_s1: k_s12,
            },
        }],
    };
    let prep_10a =
        PreparedDomainDockRestraints::compile(&kin, &part, &[], &[], &[pre_exact], &[], None)
            .unwrap();
    let w_pre_only = DomainDockEnergyWeights {
        w_rdc: 0.0,
        w_pcs: 0.0,
        w_pre: 1.0,
        w_noe: 0.0,
        noe_soft_window: 3.0,
        w_vdw: 0.0,
        vdw_scale: 1.0,
        w_rama: 0.0,
    };
    let (eb_10a, gr_10a) = evaluate_domain_dock_fixed_saupe_cpu(
        &kin,
        &prep_10a,
        &kin.initial_state(),
        &[],
        &w_pre_only,
    );
    // At exact `r = 10.000 A`, unsoftened `r^-6` PRE energy MUST be `0.0` (`< 1e-24`, vs huge bias on `r^2 + 4`!)
    assert!(
        eb_10a.pre < 1.0e-22,
        "unsoftened PRE energy at exact r = 10.000 A must vanish, got {:.6e}",
        eb_10a.pre
    );
    assert!(
        gr_10a.grad_phi.iter().all(|g| g.abs() < 1.0e-8),
        "PRE gradient at exact r = 10.000 A minimum must vanish"
    );

    // Verify `mtsl_tether_allowance_a = Some(6.0)`: a DistanceBounds `[12.0, 15.0]` evaluated at
    // `r = 19.5 A` (`<= 15.0 + 6.0 = 21.0 A`) incurs `0.0` false upper-bound penalty!
    let site_tether = [h95[0] - 19.5, h95[1], h95[2]];
    let pre_tether = DomainPreSiteSet {
        site_id: "mtsl_cb".to_string(),
        site_pos_initial: Some(site_tether),
        anchor_chain_id: None,
        anchor_residue_atom: None,
        host_domain_id: Some(1),
        tau_c_ns: None,
        proton_mhz: None,
        mtsl_tether_allowance_a: Some(6.0),
        restraints: vec![DomainPreRestraint {
            site_id: "mtsl_cb".to_string(),
            chain_id: None,
            seq_id: ch0.residues[94].seq_id,
            atom_name: "H".to_string(),
            kind: PreRestraintKind::DistanceBounds {
                lower_a: 12.0,
                upper_a: 15.0,
                sigma_a: 1.0,
            },
        }],
    };
    let prep_tether =
        PreparedDomainDockRestraints::compile(&kin, &part, &[], &[], &[pre_tether], &[], None)
            .unwrap();
    let (eb_tether, _) = evaluate_domain_dock_fixed_saupe_cpu(
        &kin,
        &prep_tether,
        &kin.initial_state(),
        &[],
        &w_pre_only,
    );
    assert_eq!(
        eb_tether.pre, 0.0,
        "6.0 A MTSL tether allowance must admit r = 19.5 A <= 15 + 6 A"
    );
}

#[test]
fn test_p2_8_p2_10_p2_11_p2_12_observability_8param_metal_ambiguous_noe_and_pae_slope() {
    let (native_model, pae, part) = build_two_domain_ubq_kh3_benchmark();
    let kin = DomainLinkerKinematics::from_af_model(&native_model, &part).unwrap();
    let ch0 = kin.build_pdb_chain(&kin.initial_state());

    // --- P2-8: Coaxial 2-medium (`cos > 0.85`), intra-domain-only PCS, and 1-PRE refusal ---
    let s_coax_1 = saupe_matrix_from_pas(-13.0, 0.25, [20.0, 50.0, 100.0]);
    let s_coax_2 = saupe_matrix_from_pas(-14.0, 0.27, [22.0, 52.0, 103.0]); // nearly collinear (`|cos| > 0.98`)
    let coax_m1 =
        simulate_domain_rdc_medium(&ch0, "c1", &s_coax_1, &s_coax_1, 79, (71, 78), 0.2, 901);
    let coax_m2 =
        simulate_domain_rdc_medium(&ch0, "c2", &s_coax_2, &s_coax_2, 79, (71, 78), 0.2, 902);
    let cfg_fast = DomainDockConfig {
        ensemble_size: 1,
        n_replicas: 4,
        anneal_steps: 4,
        lbfgs_iters: 10,
        ..Default::default()
    };
    assert!(
        dock_domains_cpu(
            &native_model,
            &pae,
            &[coax_m1.clone(), coax_m2],
            &[],
            &[],
            &[],
            &cfg_fast
        )
        .is_err(),
        "nearly coaxial 2-medium RDC-only run must be refused by default (`|cos| >= 0.85`)"
    );

    // Intra-domain-only PCS (`seq_id <= 70` on Domain 1 ONLY, `0` PCSs on Domain 2) + 1 RDC medium must STILL be refused!
    let metal_d1_intra = v_add(ch0.residues[30].ca, [4.0, 3.0, -3.0]);
    let mut pcs_intra_only = format!(
        "# metal_pos: {:.4} {:.4} {:.4}\n# metal_domain: 1\nseq_id\tatom_name\tpcs_ppm\tsigma_ppm\n",
        metal_d1_intra[0], metal_d1_intra[1], metal_d1_intra[2]
    );
    for r in &ch0.residues[..70] {
        if let Some(h) = r.h {
            if v_dist(h, metal_d1_intra) >= 6.5 {
                let pv = predict_pcs_spherical_ppm(h, metal_d1_intra, 25.0, 7.0, [0.3, 0.9, 1.5]);
                pcs_intra_only.push_str(&format!("{}\tH\t{pv:.5}\t0.03\n", r.seq_id));
            }
        }
    }
    let intra_pcs_tags = parse_domain_pcs_tsv_str(&pcs_intra_only).unwrap();
    assert!(
        dock_domains_cpu(
            &native_model,
            &pae,
            std::slice::from_ref(&coax_m1),
            &intra_pcs_tags,
            &[],
            &[],
            &cfg_fast,
        )
        .is_err(),
        "intra-domain-only PCS + single RDC medium leaves Domain 2 unrestrained and must fail closed"
    );

    // --- P2-10: Default `7.0 A` `# metal_anchor` offset + 8-parameter VarPro metal refinement ---
    let s_orth_2 = saupe_matrix_from_pas(15.0, 0.46, [115.0, 78.0, 32.0]);
    let orth_m2 =
        simulate_domain_rdc_medium(&ch0, "m2", &s_orth_2, &s_orth_2, 79, (71, 78), 0.15, 903);

    // True lanthanide metal sits `7.2 A` along `CA_66 -> CB_66` + `[1.8, -1.5, 1.2]` (`~2.7 A` off initial anchor guess!)
    let ca66 = ch0.residues[65].ca;
    let cb66 = ch0.residues[65].cb.unwrap();
    let anchor_guess_7a = v_add(ca66, v_scale(v_unit(v_sub(cb66, ca66)), 7.0));
    let true_metal = v_add(anchor_guess_7a, [1.8, -1.5, 1.2]);
    let mut pcs_8p_tsv = String::from(
        "# metal_anchor: 66:CB\n# refine_metal_pos: true\n# metal_search_radius_a: 6.0\nseq_id\tatom_name\tpcs_ppm\tsigma_ppm\n",
    );
    for (idx, r) in ch0.residues.iter().enumerate() {
        if !(70..78).contains(&idx) {
            if let Some(h) = r.h {
                if v_dist(h, true_metal) >= 6.5 {
                    let pv =
                        predict_pcs_spherical_ppm(h, true_metal, 32.0, 9.5, [0.45, 1.05, 1.85]);
                    pcs_8p_tsv.push_str(&format!("{}\tH\t{pv:.5}\t0.015\n", r.seq_id));
                }
            }
        }
    }
    let pcs_8p_tags = parse_domain_pcs_tsv_str(&pcs_8p_tsv).unwrap();
    assert!(pcs_8p_tags[0].refine_metal_pos);
    let prep_8p =
        PreparedDomainDockRestraints::compile(&kin, &part, &[], &pcs_8p_tags, &[], &[], None)
            .unwrap();
    let fitted_metal = prep_8p.pcs_tags[0].fitted_tensor_initial.metal_pos_angstrom;
    let res_8p = dock_domains_cpu(
        &native_model,
        &pae,
        &[coax_m1, orth_m2],
        &pcs_8p_tags,
        &[],
        &[],
        &cfg_fast,
    )
    .unwrap();
    let metal_err_a = v_dist(fitted_metal, true_metal);
    println!(
        "P2-10 8-param VarPro metal refinement: guess_err={:.3} A -> refined_err={metal_err_a:.3} A, PCS Q_work={:.4}",
        v_dist(anchor_guess_7a, true_metal),
        res_8p.report.pcs_q_work.unwrap()
    );
    assert!(
        metal_err_a < 0.35,
        "8-param VarPro metal refinement error = {metal_err_a:.3} A (must be < 0.35 A)"
    );

    // --- P2-11: Ambiguous 3-candidate NOE Nilges `r^-6` sum-averaging + FD gradient check ---
    // Candidate 0 is FAR (`> 25 A`, would violate `[2.0, 12.0]` if only `candidates[0]` were used!),
    // while Candidate 2 is CLOSE (`< 11 A`) so `(r0^-6 + r1^-6 + r2^-6)^(-1/6) < 12.0 A`!
    let d_far = v_dist(ch0.residues[5].ca, ch0.residues[130].ca);
    let d_close = v_dist(ch0.residues[68].ca, ch0.residues[80].ca);
    assert!(d_far > 20.0);
    let upper_ambig = d_close - 2.0; // slightly violated so gradient is non-zero and exercises all 3 candidates!
    let ambig_noe = AmbiguousDistanceRestraint {
        peak_index: 7,
        exp: "NOE_AMBIG".to_string(),
        lower_bound: 2.0,
        upper_bound: upper_ambig.max(4.0),
        candidates: vec![
            NoeCandidatePair {
                res_a: ch0.residues[5].seq_id as usize,
                atom_a: NoeProton::AlphaHA,
                res_b: ch0.residues[130].seq_id as usize,
                atom_b: NoeProton::AlphaHA,
                shift_score: 1.0,
                network_score: 1.0,
                weight: 0.5,
            },
            NoeCandidatePair {
                res_a: ch0.residues[35].seq_id as usize,
                atom_a: NoeProton::AlphaHA,
                res_b: ch0.residues[105].seq_id as usize,
                atom_b: NoeProton::AlphaHA,
                shift_score: 1.0,
                network_score: 1.0,
                weight: 0.3,
            },
            NoeCandidatePair {
                res_a: ch0.residues[68].seq_id as usize,
                atom_a: NoeProton::AlphaHA,
                res_b: ch0.residues[80].seq_id as usize,
                atom_b: NoeProton::AlphaHA,
                shift_score: 1.0,
                network_score: 1.0,
                weight: 0.2,
            },
        ],
        all_candidates: Vec::new(),
        is_unambiguous: false,
    };
    let single_cand0_noe = AmbiguousDistanceRestraint {
        candidates: vec![ambig_noe.candidates[0].clone()],
        ..ambig_noe.clone()
    };
    let prep_ambig =
        PreparedDomainDockRestraints::compile(&kin, &part, &[], &[], &[], &[ambig_noe], None)
            .unwrap();
    let prep_cand0 = PreparedDomainDockRestraints::compile(
        &kin,
        &part,
        &[],
        &[],
        &[],
        &[single_cand0_noe],
        None,
    )
    .unwrap();
    let w_noe_only = DomainDockEnergyWeights {
        w_rdc: 0.0,
        w_pcs: 0.0,
        w_pre: 0.0,
        w_noe: 2.0,
        noe_soft_window: 3.0,
        w_vdw: 0.0,
        vdw_scale: 1.0,
        w_rama: 0.0,
    };
    let st0 = kin.initial_state();
    let (e_ambig, g_ambig) =
        evaluate_domain_dock_fixed_saupe_cpu(&kin, &prep_ambig, &st0, &[], &w_noe_only);
    let (e_cand0, _) =
        evaluate_domain_dock_fixed_saupe_cpu(&kin, &prep_cand0, &st0, &[], &w_noe_only);
    assert!(
        e_ambig.noe < 0.25 * e_cand0.noe,
        "r^-6 sum-averaging must be dominated by the shorter candidate ({:.4} vs single-far {:.4})",
        e_ambig.noe,
        e_cand0.noe
    );
    let eps = 1.0e-6;
    for &idx in &kin.active_linker_residues[..3] {
        let mut sp = st0.clone();
        let mut sm = st0.clone();
        sp.psi_rad[idx] += eps;
        sm.psi_rad[idx] -= eps;
        let (ep, _) =
            evaluate_domain_dock_fixed_saupe_cpu(&kin, &prep_ambig, &sp, &[], &w_noe_only);
        let (em, _) =
            evaluate_domain_dock_fixed_saupe_cpu(&kin, &prep_ambig, &sm, &[], &w_noe_only);
        let fd = (ep.total - em.total) / (2.0 * eps);
        assert!(
            (g_ambig.grad_psi[idx] - fd).abs() / fd.abs().max(1.0) < 1.0e-6,
            "ambiguous NOE grad_psi[{idx}] mismatch: {} vs {fd}",
            g_ambig.grad_psi[idx]
        );
    }

    // --- P2-12: Elongated single-domain PAE slope (`base + 0.12 * |i - j|`, `L = 150`) ---
    let l_slope = 150usize;
    for base in [0.5_f32, 1.0, 1.5, 2.0] {
        let mut vals = vec![0.0_f32; l_slope * l_slope];
        for i in 0..l_slope {
            for j in 0..l_slope {
                vals[i * l_slope + j] = if i == j {
                    0.5
                } else {
                    base + 0.12 * (i.abs_diff(j) as f32)
                };
            }
        }
        let pae_slope = PaeMatrix {
            n_residues: l_slope,
            values_angstrom: vals,
            max_pae: 25.0,
            dialect: PaeJsonDialect::AfdbV3V4,
            chain_id: Some("A".to_string()),
            plddt_from_json: None,
        };
        let p_slope =
            cluster_pae_domains(&pae_slope, None, None, None, &PaeClusterConfig::default())
                .unwrap();
        assert_eq!(
            p_slope.domains.len(),
            1,
            "base={base}: gentle intra-domain PAE slope must NOT be split 75/75 (got D={})",
            p_slope.domains.len()
        );
        for d in &p_slope.domains {
            assert!(
                d.median_intra_pae_angstrom <= 5.0,
                "domain {} violated median_intra_pae <= 5.0 A ({:.2} A)",
                d.domain_id,
                d.median_intra_pae_angstrom
            );
        }
    }
}

#[test]
fn test_p1_7_non_circular_real_experimental_rdcs_and_intra_domain_noise_docking() {
    // 1. Non-circular experimental RDC evaluation on REAL crystal structures (`1UBQ` vs `1D3Z` RDCs,
    //    `1ZZK` vs `1KHM` RDCs — independent X-ray vs solution NMR!)
    let root = repo_root();
    let ubq_model = parse_af_structure_model(
        root.join("data/fixtures/ubiquitin_1ubq_xray.pdb"),
        &AfIngestOptions::default(),
    )
    .unwrap();
    let mut ubq_plddt = vec![92.0_f64; ubq_model.residues.len()];
    for p in ubq_plddt.iter_mut().skip(71) {
        *p = 52.0; // flexible RLRGG C-terminal tail (`72..=76`) excluded via `21.W2:P1-4`!
    }
    let ubq_cif = serialize_chain_to_af_mmcif(
        "ubq_1ubq_real_rdc",
        &ubq_model.to_pdb_chain(),
        &ubq_plddt,
        None,
    );
    let ubq_af = parse_af_structure_model_str(&ubq_cif, &AfIngestOptions::default()).unwrap();
    let mut ubq_dom_mask = vec![1usize; 71];
    ubq_dom_mask.extend(vec![0usize; ubq_af.residues.len() - 71]);
    let ubq_pae = build_synthetic_pae(ubq_af.residues.len(), &ubq_dom_mask, 177, 1.8, 22.0);
    let ubq_real_rdcs =
        read_domain_rdc_tsv(root.join("data/fixtures/ubiquitin_1d3z_rdc.tsv")).unwrap();
    assert_eq!(
        ubq_real_rdcs.len(),
        2,
        "1D3Z has 2 real experimental alignment media"
    );

    let ubq_real_res = dock_domains_multi_chain_cpu(
        &[ubq_af],
        &ubq_pae,
        &ubq_real_rdcs,
        &[],
        &[],
        &[],
        &DomainDockConfig {
            ensemble_size: 1,
            n_replicas: 4,
            anneal_steps: 4,
            lbfgs_iters: 10,
            ..Default::default()
        },
    )
    .unwrap();
    println!(
        "P1-7 real experimental 1D3Z RDCs on 1UBQ 1.8 A X-ray: Q_work={:.4}, ByPeptidePlane Q_free={:.4}",
        ubq_real_res.report.global_rdc_q_work, ubq_real_res.report.global_rdc_q_free
    );
    // Non-circular: real X-ray vs solution NMR Q_work is non-zero (`> 0.08`) and passes `< 0.40`!
    assert!(
        ubq_real_res.report.global_rdc_q_work > 0.08
            && ubq_real_res.report.global_rdc_q_free < 0.40,
        "unexpected 1UBQ vs 1D3Z real RDC Q_work={:.4}, Q_free={:.4}",
        ubq_real_res.report.global_rdc_q_work,
        ubq_real_res.report.global_rdc_q_free
    );

    // 2. Non-circular two-domain docking with REAL experimental `1D3Z` RDCs on Domain 1 AND
    //    genuine `~0.35 A` intra-domain structural discrepancy inside BOTH domains (not zero-noise
    //    self-consistency!)
    let (native_model, pae, part) = build_two_domain_ubq_kh3_benchmark();
    let native_kin = DomainLinkerKinematics::from_af_model(&native_model, &part).unwrap();
    let native_chain = native_kin.build_pdb_chain(&native_kin.initial_state());

    // Build a noisy-intra-domain starting structure: perturb internal coordinates of BOTH rigid
    // domains smoothly (`~0.32 A` intra-domain backbone RMSD vs `native_chain`, matching typical
    // AF2 vs crystal/NMR domain discrepancy!) AND rotate/translate Domain 2 by `> 60 deg, > 8 A`!
    let mut start_state = native_kin.initial_state();
    let pert_phi = [0.16, 0.22, 0.27, 0.25, 0.20, 0.16, 0.14, 0.10];
    let pert_psi = [0.14, 0.18, 0.23, 0.20, 0.16, 0.14, 0.10, 0.08];
    for (k, &idx) in native_kin.active_linker_residues.iter().enumerate() {
        start_state.phi_rad[idx] += pert_phi[k % pert_phi.len()];
        start_state.psi_rad[idx] += pert_psi[k % pert_psi.len()];
    }
    let mut noisy_start_chain = native_kin.build_pdb_chain(&start_state);
    for (idx, r) in noisy_start_chain.residues.iter_mut().enumerate() {
        if !(70..78).contains(&idx) {
            // Smooth sinusoidal internal coordinate distortion (`~0.30..0.38 A` intra-domain RMSD)
            let phase = (idx as f64) * 0.37;
            let dx = [
                0.24 * phase.sin(),
                0.22 * (1.3 * phase).cos(),
                0.20 * (0.8 * phase).sin(),
            ];
            r.n = v_add(r.n, dx);
            r.ca = v_add(r.ca, dx);
            r.c = v_add(r.c, dx);
            r.o = v_add(r.o, dx);
            r.cb = r.cb.map(|p| v_add(p, dx));
            if let Some(h) = r.h {
                // Slightly perturb N-H bond vector orientation (`~3..4 deg`, like AF2 vs NMR!)
                let tilt = [
                    0.055 * (1.7 * phase).cos(),
                    -0.050 * (1.1 * phase).sin(),
                    0.045 * (2.1 * phase).cos(),
                ];
                let nh = v_unit(v_add(v_unit(v_sub(h, r.n)), tilt));
                r.h = Some(v_add(r.n, v_scale(nh, 1.010)));
            }
        }
    }
    let d1_nat_ca: Vec<Vec3> = (0..70).map(|i| native_chain.residues[i].ca).collect();
    let d1_noisy_ca: Vec<Vec3> = (0..70).map(|i| noisy_start_chain.residues[i].ca).collect();
    let (_, _, _, intra_d1_noise_a) = kabsch_transform(&d1_noisy_ca, &d1_nat_ca);
    assert!(
        (0.20..=0.55).contains(&intra_d1_noise_a),
        "expected realistic ~0.3 A intra-domain coordinate noise, got {intra_d1_noise_a:.3} A"
    );

    let plddts = native_model.plddt_per_residue.clone().unwrap();
    let noisy_cif =
        serialize_chain_to_af_mmcif("ubq_kh3_noisy_af", &noisy_start_chain, &plddts, None);
    let noisy_af = parse_af_structure_model_str(&noisy_cif, &AfIngestOptions::default()).unwrap();

    // Restraints come from `native_chain` (distinct intra-domain coordinates!), NOT `noisy_af`!
    let s1 = saupe_matrix_from_pas(-13.5, 0.26, [22.0, 54.0, 108.0]);
    let s2 = saupe_matrix_from_pas(15.0, 0.46, [115.0, 78.0, 32.0]);
    let rdc_media = vec![
        simulate_domain_rdc_medium(&native_chain, "bicelle", &s1, &s1, 79, (71, 78), 0.45, 951),
        simulate_domain_rdc_medium(&native_chain, "pf1", &s2, &s2, 79, (71, 78), 0.45, 952),
    ];
    let metal_d1 = v_add(native_chain.residues[65].ca, [2.8, 3.2, -2.5]);
    let mut pcs_tsv = format!(
        "# metal_pos: {:.4} {:.4} {:.4}\n# metal_domain: 1\nseq_id\tatom_name\tpcs_ppm\tsigma_ppm\n",
        metal_d1[0], metal_d1[1], metal_d1[2]
    );
    for (idx, r) in native_chain.residues.iter().enumerate() {
        if !(70..78).contains(&idx) && idx % 2 == 0 {
            if let Some(h) = r.h {
                if v_dist(h, metal_d1) >= 6.5 {
                    let pv = predict_pcs_spherical_ppm(h, metal_d1, 32.0, 9.5, [0.45, 1.05, 1.85]);
                    pcs_tsv.push_str(&format!("{}\tH\t{pv:.5}\t0.03\n", r.seq_id));
                }
            }
        }
    }
    let pcs_tags = parse_domain_pcs_tsv_str(&pcs_tsv).unwrap();

    let res_noisy = dock_domains_cpu(
        &noisy_af,
        &pae,
        &rdc_media,
        &pcs_tags,
        &[],
        &[],
        &DomainDockConfig {
            ensemble_size: 4,
            n_replicas: 48,
            anneal_steps: 40,
            lbfgs_iters: 120,
            ..Default::default()
        },
    )
    .unwrap();

    let mut dom_mask = vec![false; native_chain.residues.len()];
    for d in &res_noisy.report.partition.domains {
        for &idx in &d.residue_indices {
            dom_mask[idx] = true;
        }
    }
    let noisy_dock_rmsd =
        backbone_rmsd_between_chains(&res_noisy.chains[0], &native_chain, Some(&dom_mask));
    println!(
        "P1-7 non-circular two-domain dock (intra_noise={intra_d1_noise_a:.3} A): BB_RMSD={noisy_dock_rmsd:.3} A, Q_work={:.4}, Q_free={:.4}",
        res_noisy.report.global_rdc_q_work, res_noisy.report.global_rdc_q_free
    );
    assert!(
        noisy_dock_rmsd < 1.60 && res_noisy.report.global_rdc_q_free < 0.35,
        "non-circular noisy-domain docking BB_RMSD={noisy_dock_rmsd:.3} A, Q_free={:.4}",
        res_noisy.report.global_rdc_q_free
    );
    assert_eq!(
        res_noisy.report.quality_verdict,
        DomainDockQualityVerdict::PassedAllChecks
    );
}

#[test]
fn test_21f_b4_gd3_spin_7_2_preserves_per_site_k_pre_cpu() {
    // Regression test for `21F:B4`: `prepare_domain_dock_pipeline` MUST NOT overwrite per-site
    // Solomon `k_pre_a6_s1` (e.g. Gd3+ `S=7/2`, 21x larger than S=1/2 nitroxide) with S=1/2!
    // Fails at `f8408b9` where `*k_pre_a6_s1 < 1.0e12` always fired.
    let (native_model, _pae, part) = build_two_domain_ubq_kh3_benchmark();
    let kin = DomainLinkerKinematics::from_af_model(&native_model, &part).unwrap();
    let (slots0, _) = kin.forward_slots_and_block_frames(&kin.initial_state());

    let h95 = slots0[94 * 7 + 1].0;
    let r_true = 16.0_f64;
    let gd_pos = [h95[0] - r_true, h95[1], h95[2]];
    let k_nitro = compute_solomon_k_pre_a6_s1(10.0, 600.0, 0.5, CODATA_ELECTRON_G_FACTOR).unwrap();
    let k_gd = compute_solomon_k_pre_a6_s1(10.0, 600.0, 3.5, CODATA_ELECTRON_G_FACTOR).unwrap();
    assert!(
        ((k_gd / k_nitro) - 21.0).abs() < 1e-10,
        "Gd3+ S=7/2 K_PRE must be exactly 21x nitroxide S=1/2"
    );

    let g2_gd =
        evaluate_solomon_gamma2_s1(r_true, 10.0, 600.0, 3.5, CODATA_ELECTRON_G_FACTOR).unwrap();
    let tsv = format!(
        "# site_pos: {:.4} {:.4} {:.4}\n# site_domain: 1\n# tau_c_ns: 10.0\n# proton_mhz: 600.0\n# spin_s: 3.5\nseq_id\tatom_name\tgamma2_s1\tsigma_s1\n{}\tH\t{:.6}\t1.0\n",
        gd_pos[0], gd_pos[1], gd_pos[2], kin.residues[94].seq_id, g2_gd
    );
    let parsed_sites = parse_domain_pre_tsv_str(&tsv).unwrap();
    assert_eq!(parsed_sites.len(), 1);

    let cfg = DomainDockConfig {
        allow_underdetermined_single_medium: true,
        ..Default::default()
    };
    let mut empty_rdcs = Vec::new();
    let pipe = prepare_domain_dock_inputs_full(
        &kin,
        &part,
        &mut empty_rdcs,
        &[],
        &parsed_sites,
        &[],
        &cfg,
    )
    .unwrap();
    assert_eq!(pipe.prep.pre_terms.len(), 1);
    let kept_k = pipe.prep.pre_terms[0].k_pre_a6_s1;
    let rel_err = (kept_k - k_gd).abs() / k_gd;
    assert!(
        rel_err < 1e-10,
        "21F:B4 regression: Gd3+ S=7/2 k_pre_a6_s1 was overwritten (got {kept_k:.6e}, expected {k_gd:.6e}, nitroxide={k_nitro:.6e})"
    );

    let w_pre_only = DomainDockEnergyWeights {
        w_rdc: 0.0,
        w_pcs: 0.0,
        w_pre: 1.0,
        w_noe: 0.0,
        noe_soft_window: 3.0,
        w_vdw: 0.0,
        vdw_scale: 1.0,
        w_rama: 0.0,
    };
    let (eb, _) = evaluate_domain_dock_fixed_saupe_cpu(
        &kin,
        &pipe.prep,
        &kin.initial_state(),
        &[],
        &w_pre_only,
    );
    assert!(
        eb.pre < 1.0e-10,
        "21F:B4 regression: Gd3+ PRE energy at exact r = {r_true} A must vanish, got {:.6e}",
        eb.pre
    );
}

#[test]
fn test_21f_b5_discontinuous_insertion_enforces_reentry_peptide_bond_closure() {
    // Regression test for `21F:B5`: docked discontinuous insertion domains MUST keep the re-entry
    // peptide bond C'_{s_1-1}-N_{s_1} closed (`|d(C, N) - 1.33 A| <= 0.15 A`) BOTH with default
    // `w_rama` AND when `w_rama == 0.0`, while reporting `max_reentry_cn_err_a` and preserving
    // `0.000 A` intra-domain rigidity. Fails at `f8408b9` (1.29 A C-N bond break).
    let ubq = parse_pdb_first_model(repo_root().join("data/fixtures/ubiquitin_1ubq_xray.pdb"))
        .expect("parse 1UBQ");
    let n = ubq.residues.len();
    let mut true_dom = vec![0usize; n];
    for d in true_dom.iter_mut().take(22) {
        *d = 1;
    }
    for d in true_dom.iter_mut().take(46).skip(26) {
        *d = 2;
    }
    for d in true_dom.iter_mut().take(72).skip(50) {
        *d = 1;
    }
    let plddt: Vec<f64> = true_dom
        .iter()
        .map(|&d| if d == 0 { 60.0 } else { 93.0 })
        .collect();
    let pae = build_synthetic_pae(n, &true_dom, 2105, 2.0, 24.5);
    let cif = serialize_chain_to_af_mmcif("ubq_insertion_b5", &ubq, &plddt, None);
    let model = parse_af_structure_model_str(&cif, &AfIngestOptions::default()).unwrap();
    let seq_ids = model.seq_ids();
    let part = cluster_pae_domains(
        &pae,
        Some(&plddt),
        Some(&seq_ids),
        None,
        &PaeClusterConfig::default(),
    )
    .unwrap();
    let kin = DomainLinkerKinematics::from_af_model(&model, &part).unwrap();
    assert!(kin.has_reentry_blocks());

    let ch0 = kin.build_pdb_chain(&kin.initial_state());
    let s1 = saupe_matrix_from_pas(-13.0, 0.31, [24.0, 52.0, 104.0]);
    let s2 = saupe_matrix_from_pas(14.5, 0.45, [112.0, 76.0, 35.0]);
    let med1 = simulate_domain_rdc_medium(&ch0, "ins_m1", &s1, &s1, 77, (-1, -1), 0.0, 811);
    let med2 = simulate_domain_rdc_medium(&ch0, "ins_m2", &s2, &s2, 77, (-1, -1), 0.0, 812);

    let metal_d1 = v_add(ubq.residues[60].ca, [3.0, -2.5, 2.8]);
    let mut pcs_rows = Vec::new();
    for (idx, r) in ubq.residues.iter().enumerate() {
        if true_dom[idx] != 0 {
            if let Some(h) = r.h {
                if v_dist(h, metal_d1) > 6.0 {
                    let pv = predict_pcs_spherical_ppm(h, metal_d1, 28.0, 8.0, [0.4, 1.0, 1.8]);
                    pcs_rows.push(DomainPcsRestraint {
                        tag_id: "d1_ln".to_string(),
                        chain_id: None,
                        seq_id: r.seq_id,
                        atom_name: "H".to_string(),
                        pcs_obs_ppm: pv,
                        sigma_ppm: 0.03,
                        is_free: false,
                    });
                }
            }
        }
    }
    let pcs_tag = DomainPcsTagSet {
        tag_id: "d1_ln".to_string(),
        metal_pos_initial: Some(metal_d1),
        anchor_chain_id: None,
        anchor_residue_atom: None,
        host_domain_id: Some(1),
        pre_fitted_tensor: None,
        refine_metal_pos: false,
        metal_search_radius_a: None,
        restraints: pcs_rows,
    };

    // Perturb linker so re-entry C'_{49}-N_{50} starts broken before docking
    let mut st_start = kin.initial_state();
    st_start.psi_rad[23] += 0.40;
    st_start.phi_rad[24] -= 0.35;
    st_start.psi_rad[46] += 0.30;
    let start_chain = kin.build_pdb_chain(&st_start);
    let start_cif =
        serialize_chain_to_af_mmcif("ubq_insertion_b5_start", &start_chain, &plddt, None);
    let start_af = parse_af_structure_model_str(&start_cif, &AfIngestOptions::default()).unwrap();

    for (label, w_rama) in [("default_w_rama", 0.45_f64), ("zero_w_rama", 0.0_f64)] {
        let weights = DomainDockEnergyWeights {
            w_rama,
            ..Default::default()
        };
        let cfg = DomainDockConfig {
            weights,
            ensemble_size: 2,
            n_replicas: 16,
            anneal_steps: 24,
            lbfgs_iters: 100,
            ..Default::default()
        };
        let docked = dock_domains_cpu(
            &start_af,
            &pae,
            &[med1.clone(), med2.clone()],
            std::slice::from_ref(&pcs_tag),
            &[],
            &[],
            &cfg,
        )
        .unwrap();
        let cn_dist = v_dist(
            docked.chains[0].residues[49].c,
            docked.chains[0].residues[50].n,
        );
        let cn_err = (cn_dist - 1.33).abs();
        let ens_max_cn_err = docked
            .conformers
            .iter()
            .map(|c| c.max_reentry_cn_err_a)
            .fold(0.0_f64, f64::max);
        eprintln!(
            "21F:B5 ({label}): max_reentry_cn_err_a model 1 = {:.4} A, max over {} ensemble members = {ens_max_cn_err:.4} A (per model: {:?})",
            docked.conformers[0].max_reentry_cn_err_a,
            docked.conformers.len(),
            docked
                .conformers
                .iter()
                .map(|c| format!("{:.4}", c.max_reentry_cn_err_a))
                .collect::<Vec<_>>()
        );
        assert!(
            cn_err <= 0.15,
            "21F:B5 ({label}): re-entry peptide bond C49-N50 = {cn_dist:.4} A (|d - 1.33| = {cn_err:.4} A > 0.15 A)"
        );
        assert!(
            docked.report.max_reentry_cn_err_a <= 0.15,
            "21F:B5 ({label}): report.max_reentry_cn_err_a = {:.4} A > 0.15 A",
            docked.report.max_reentry_cn_err_a
        );
        assert!(
            docked.conformers[0].max_reentry_cn_err_a <= 0.15,
            "21F:B5 ({label}): conformer.max_reentry_cn_err_a = {:.4} A > 0.15 A",
            docked.conformers[0].max_reentry_cn_err_a
        );
        assert!(
            docked.report.max_intra_domain_rmsd_angstrom < 1.0e-12,
            "21F:B5 ({label}): intra-domain drift = {:.3e} A",
            docked.report.max_intra_domain_rmsd_angstrom
        );
        assert!(
            format_domain_dock_tsv(&docked.report).contains("max_reentry_cn_err_a"),
            "TSV report must include max_reentry_cn_err_a"
        );
    }
}

#[test]
fn test_21f_b18_default_mtsl_tether_allowance_applied_to_anchored_sites() {
    // Regression test for `21F:B18`: `DomainDockConfig::default().mtsl_tether_allowance_a` must
    // equal `DEFAULT_MTSL_TETHER_ALLOWANCE_ANGSTROM` (`6.0 A`), and `# site_anchor` PRE sites
    // without an explicit per-site override must inherit the `6.0 A` MTSL tether allowance!
    // Fails at `f8408b9` where `DomainDockConfig::default().mtsl_tether_allowance_a == 0.0`.
    assert!(
        (DomainDockConfig::default().mtsl_tether_allowance_a
            - DEFAULT_MTSL_TETHER_ALLOWANCE_ANGSTROM)
            .abs()
            < 1e-12,
        "DomainDockConfig::default().mtsl_tether_allowance_a must be DEFAULT_MTSL_TETHER_ALLOWANCE_ANGSTROM (6.0 A)"
    );

    let (native_model, _pae, part) = build_two_domain_ubq_kh3_benchmark();
    let kin = DomainLinkerKinematics::from_af_model(&native_model, &part).unwrap();
    let (slots0, _) = kin.forward_slots_and_block_frames(&kin.initial_state());
    let anchor_ca = slots0[20 * 7 + 2].0; // residue index 20 (seq_id 21) in Domain 1
    let h95 = slots0[94 * 7 + 1].0; // residue index 94 (seq_id 95) in Domain 2
    let r_actual = v_dist(anchor_ca, h95);
    // Set DistanceBounds upper bound 4.5 A below `r_actual`: without 6.0 A MTSL tether allowance,
    // `r_actual > upper_a` incurs a large penalty; with the 6.0 A default tether pad, `r_actual <= upper_a + 6.0` -> 0.0 energy!
    let pre_anchored = DomainPreSiteSet {
        site_id: "mtsl_anchor_default".to_string(),
        site_pos_initial: None,
        anchor_chain_id: None,
        anchor_residue_atom: Some((kin.residues[20].seq_id, "CA".to_string())),
        host_domain_id: Some(1),
        tau_c_ns: None,
        proton_mhz: None,
        mtsl_tether_allowance_a: None,
        restraints: vec![DomainPreRestraint {
            site_id: "mtsl_anchor_default".to_string(),
            chain_id: None,
            seq_id: kin.residues[94].seq_id,
            atom_name: "H".to_string(),
            kind: PreRestraintKind::DistanceBounds {
                lower_a: (r_actual - 8.0).max(2.0),
                upper_a: (r_actual - 4.5).max(3.0),
                sigma_a: 1.0,
            },
        }],
    };

    let prep =
        PreparedDomainDockRestraints::compile(&kin, &part, &[], &[], &[pre_anchored], &[], None)
            .unwrap();
    let w_pre_only = DomainDockEnergyWeights {
        w_rdc: 0.0,
        w_pcs: 0.0,
        w_pre: 1.0,
        w_noe: 0.0,
        noe_soft_window: 3.0,
        w_vdw: 0.0,
        vdw_scale: 1.0,
        w_rama: 0.0,
    };
    let (eb, _) =
        evaluate_domain_dock_fixed_saupe_cpu(&kin, &prep, &kin.initial_state(), &[], &w_pre_only);
    assert_eq!(
        eb.pre, 0.0,
        "21F:B18: anchored MTSL PRE site must inherit 6.0 A default tether allowance (r={r_actual:.2} A <= upper+6.0 A), got E_pre={:.4}",
        eb.pre
    );
}

// ---- FIX3 slice D: 21F:N1 sequentially numbered chain gaps, B4/B5 follow-ups ----

/// Translate residues `from..` of `chain` rigidly along the `C'_{from-1} -> N_{from}` direction so
/// that the input C-N distance across the cut becomes exactly `gap_a` while `seq_id`s stay
/// consecutive (a renumbered model with missing residues).
fn ubq_with_sequential_gap(chain: &PdbChain, from: usize, gap_a: f64) -> PdbChain {
    let mut out = chain.clone();
    let c_prev = chain.residues[from - 1].c;
    let n_cur = chain.residues[from].n;
    let d0 = v_dist(c_prev, n_cur);
    let shift = v_scale(v_unit(v_sub(n_cur, c_prev)), gap_a - d0);
    let mv = |p: Vec3| v_add(p, shift);
    for r in out.residues.iter_mut().skip(from) {
        r.n = mv(r.n);
        r.ca = mv(r.ca);
        r.c = mv(r.c);
        r.o = mv(r.o);
        r.ha = mv(r.ha);
        r.h = r.h.map(mv);
        r.ha2 = r.ha2.map(mv);
        r.ha3 = r.ha3.map(mv);
        r.cb = r.cb.map(mv);
        r.qb = r.qb.map(mv);
    }
    for w in out.residues.windows(2) {
        assert_eq!(
            w[1].seq_id,
            w[0].seq_id + 1,
            "gapped model must stay sequential"
        );
    }
    out
}

/// `1UBQ` insertion-domain topology of `21F:B5` (`D1 = [0..22] ∪ [50..72]`, `D2 = [26..46]`)
/// rebuilt on `chain`, returning `(af_model, pae, partition, rdc_media, pcs_tag)` with restraints
/// simulated from `chain` itself (the docking truth is the gapped input).
fn build_insertion_case_on_chain(
    chain: &PdbChain,
    label: &str,
) -> (
    nmr_core::af_concordance::AfStructureModel,
    PaeMatrix,
    PaeDomainPartition,
    Vec<RdcMediumSet>,
    DomainPcsTagSet,
) {
    let n = chain.residues.len();
    let mut true_dom = vec![0usize; n];
    for d in true_dom.iter_mut().take(22) {
        *d = 1;
    }
    for d in true_dom.iter_mut().take(46).skip(26) {
        *d = 2;
    }
    for d in true_dom.iter_mut().take(72).skip(50) {
        *d = 1;
    }
    let plddt: Vec<f64> = true_dom
        .iter()
        .map(|&d| if d == 0 { 60.0 } else { 93.0 })
        .collect();
    let pae = build_synthetic_pae(n, &true_dom, 2105, 2.0, 24.5);
    let cif = serialize_chain_to_af_mmcif(label, chain, &plddt, None);
    let model = parse_af_structure_model_str(&cif, &AfIngestOptions::default()).unwrap();
    let seq_ids = model.seq_ids();
    let part = cluster_pae_domains(
        &pae,
        Some(&plddt),
        Some(&seq_ids),
        None,
        &PaeClusterConfig::default(),
    )
    .unwrap();
    let ch_in = model.to_pdb_chain();
    let s1 = saupe_matrix_from_pas(-13.0, 0.31, [24.0, 52.0, 104.0]);
    let s2 = saupe_matrix_from_pas(14.5, 0.45, [112.0, 76.0, 35.0]);
    let med1 = simulate_domain_rdc_medium(&ch_in, "gap_m1", &s1, &s1, 77, (-1, -1), 0.0, 811);
    let med2 = simulate_domain_rdc_medium(&ch_in, "gap_m2", &s2, &s2, 77, (-1, -1), 0.0, 812);
    let metal_d1 = v_add(ch_in.residues[60].ca, [3.0, -2.5, 2.8]);
    let mut pcs_rows = Vec::new();
    for (idx, r) in ch_in.residues.iter().enumerate() {
        if true_dom[idx] != 0 {
            if let Some(h) = r.h {
                if v_dist(h, metal_d1) > 6.0 {
                    let pv = predict_pcs_spherical_ppm(h, metal_d1, 28.0, 8.0, [0.4, 1.0, 1.8]);
                    pcs_rows.push(DomainPcsRestraint {
                        tag_id: "d1_ln".to_string(),
                        chain_id: None,
                        seq_id: r.seq_id,
                        atom_name: "H".to_string(),
                        pcs_obs_ppm: pv,
                        sigma_ppm: 0.03,
                        is_free: false,
                    });
                }
            }
        }
    }
    let pcs_tag = DomainPcsTagSet {
        tag_id: "d1_ln".to_string(),
        metal_pos_initial: Some(metal_d1),
        anchor_chain_id: None,
        anchor_residue_atom: None,
        host_domain_id: Some(1),
        pre_fitted_tensor: None,
        refine_metal_pos: false,
        metal_search_radius_a: None,
        restraints: pcs_rows,
    };
    (model, pae, part, vec![med1, med2], pcs_tag)
}

fn assert_initial_state_reproduces_input(kin: &DomainLinkerKinematics, label: &str) {
    let rebuilt = kin.build_pdb_chain(&kin.initial_state());
    let mut max_d = 0.0_f64;
    let mut worst = 0usize;
    for (i, (a, b)) in rebuilt
        .residues
        .iter()
        .zip(kin.ref_chain.residues.iter())
        .enumerate()
    {
        for (p, q) in [(a.n, b.n), (a.ca, b.ca), (a.c, b.c)] {
            let d = v_dist(p, q);
            if d > max_d {
                max_d = d;
                worst = i;
            }
        }
    }
    assert!(
        max_d < 1.0e-6,
        "21F:N1 ({label}): initial state displaced backbone atom of residue index {worst} by {max_d:.4} A from the input coordinates"
    );
}

fn run_sequential_gap_case(gap_from: usize, label: &str) {
    let ubq = parse_pdb_first_model(repo_root().join("data/fixtures/ubiquitin_1ubq_xray.pdb"))
        .expect("parse 1UBQ");
    let gapped = ubq_with_sequential_gap(&ubq, gap_from, 15.0);
    let (model, pae, part, media, pcs_tag) = build_insertion_case_on_chain(&gapped, label);
    let kin = DomainLinkerKinematics::from_af_model(&model, &part).unwrap();
    let d_in = v_dist(
        kin.ref_chain.residues[gap_from - 1].c,
        kin.ref_chain.residues[gap_from].n,
    );
    assert!((d_in - 15.0).abs() < 1e-2, "probe gap = {d_in:.3} A");

    // (1) No fabricated peptide bond: the gap is a chain break in the kinematic topology.
    assert!(
        kin.residues[gap_from].is_chain_start,
        "21F:N1 ({label}): residue index {gap_from} (C-N = {d_in:.2} A, consecutive seq_id) was joined by a fabricated peptide bond (chain_ranges = {:?})",
        kin.chain_ranges
    );
    assert_eq!(
        kin.chain_ranges.len(),
        2,
        "{label}: expected exactly one kept chain break"
    );
    // No closure restraint across the broken boundary (initial-state closure error is 0).
    assert!(
        kin.max_reentry_cn_err_angstrom(&kin.initial_state()) < 1.0e-9,
        "21F:N1 ({label}): closure restraint spans the kept chain break"
    );
    // (2) Initial state reproduces the input coordinates exactly (no clamp "bond" displacement).
    assert_initial_state_reproduces_input(&kin, label);

    // (3) Warning names both residues and the distance, and reaches the docking report.
    let s_prev = kin.residues[gap_from - 1].seq_id;
    let s_cur = kin.residues[gap_from].seq_id;
    let has_warning = |notes: &[String]| {
        notes.iter().any(|w| {
            w.contains("21F:N1")
                && w.contains(&format!("{s_prev} and"))
                && w.contains(&format!("{s_cur} are numbered consecutively"))
                && w.contains("15.00 A")
        })
    };
    assert!(
        has_warning(&kin.chain_topology_warnings),
        "21F:N1 ({label}): missing chain-break warning, got {:?}",
        kin.chain_topology_warnings
    );
    let cfg = DomainDockConfig {
        ensemble_size: 1,
        n_replicas: 8,
        anneal_steps: 12,
        lbfgs_iters: 40,
        ..Default::default()
    };
    let docked = dock_domains_cpu(
        &model,
        &pae,
        &media,
        std::slice::from_ref(&pcs_tag),
        &[],
        &[],
        &cfg,
    )
    .unwrap();
    assert!(
        has_warning(&docked.report.candor_notes),
        "21F:N1 ({label}): chain-break warning missing from report.candor_notes: {:?}",
        docked.report.candor_notes
    );
    let d_docked = v_dist(
        docked.chains[0].residues[gap_from - 1].c,
        docked.chains[0].residues[gap_from].n,
    );
    assert!(
        d_docked > MAX_PEPTIDE_JOIN_C_N_ANGSTROM,
        "21F:N1 ({label}): docking closed the 15 A gap into a {d_docked:.3} A C-N 'bond'"
    );
}

#[test]
fn test_fix3_n1_sequential_gap_at_reentry_boundary_is_not_bonded() {
    // 15 A gap between seq 50 and 51 (index 49 -> 50 = D1 re-entry start), numbering consecutive.
    // Fails at `b5e87a3`: joined into one chain and closed with an ideal 1.329 A re-entry bond.
    run_sequential_gap_case(50, "reentry_gap");
}

#[test]
fn test_fix3_n1_sequential_gap_in_linker_is_not_bonded() {
    // 15 A gap between seq 24 and 25 (index 23 -> 24, inside linker 22..26), numbering consecutive.
    // Fails at `b5e87a3`: joined with a 2.50 A clamp "bond" that moves the initial-state CA ~13 A.
    run_sequential_gap_case(24, "linker_gap");
}

fn gd_pre_tsv_rows(seq_a: i32, seq_b: i32) -> String {
    format!("site_id\tseq_id\tatom_name\tgamma2_s1\tsigma_s1\ngd1\t{seq_a}\tH\t20.0\t1.0\ngd1\t{seq_b}\tH\t15.0\t1.0\n")
}

fn pre_k_values(sites: &[DomainPreSiteSet]) -> Vec<f64> {
    sites
        .iter()
        .flat_map(|s| s.restraints.iter())
        .map(|r| match r.kind {
            PreRestraintKind::Gamma2Rate { k_pre_a6_s1, .. } => k_pre_a6_s1,
            _ => panic!("expected Gamma2Rate"),
        })
        .collect()
}

#[test]
fn test_fix3_b4_pre_unknown_per_site_spin_or_g_tag_fails_closed() {
    // A typo in a per-site `# spin_s[tag]` / `# g_factor[tag]` must not silently fall back to the
    // S = 1/2 nitroxide default (re-creating the 21F:B4 21x K_PRE bias). Fails at `b5e87a3`.
    let head = "# site_pos: 1.0 2.0 3.0\n# site_domain: 1\n# tau_c_ns: 10.0\n# proton_mhz: 600.0\n";
    let rows = gd_pre_tsv_rows(30, 31);
    let ok = format!("{head}# spin_s[gd1]: 3.5\n{rows}");
    let k_ok = pre_k_values(&parse_domain_pre_tsv_str(&ok).unwrap());
    let k_gd = compute_solomon_k_pre_a6_s1(10.0, 600.0, 3.5, CODATA_ELECTRON_G_FACTOR).unwrap();
    assert!(k_ok.iter().all(|k| ((k - k_gd) / k_gd).abs() < 1e-12));

    for (directive, label) in [
        ("# spin_s[gdl]: 3.5\n", "spin_s"),
        ("# g_factor[Gd1]: 1.99\n", "g_factor"),
        ("# tau_c_ns[gd2]: 12.0\n", "tau_c_ns"),
    ] {
        let typo = format!("{head}{directive}{rows}");
        let err = parse_domain_pre_tsv_str(&typo).expect_err(&format!(
            "21F:B4 follow-up: unknown per-site `{label}` tag must fail closed"
        ));
        assert!(
            err.contains("does not match any PRE site_id") && err.contains("gd1"),
            "{label}: error must name the bad tag and the known site IDs, got: {err}"
        );
    }
}

#[test]
fn test_fix3_b4_pre_global_spin_directives_are_order_independent() {
    // `# spin_s:` (and the other global/per-site directives) must apply to every row regardless of
    // its position in the file. Fails at `b5e87a3`, where rows parsed before the directive got S = 1/2.
    let head = "# site_pos: 1.0 2.0 3.0\n# site_domain: 1\n# tau_c_ns: 10.0\n# proton_mhz: 600.0\n";
    let rows = gd_pre_tsv_rows(30, 31);
    let k_gd = compute_solomon_k_pre_a6_s1(10.0, 600.0, 3.5, CODATA_ELECTRON_G_FACTOR).unwrap();
    for (label, text) in [
        ("before", format!("{head}# spin_s: 3.5\n{rows}")),
        ("after", format!("{head}{rows}# spin_s: 3.5\n")),
        ("tagged_after", format!("{head}{rows}# spin_s[gd1]: 3.5\n")),
        (
            "all_after",
            format!("{rows}# spin_s: 3.5\n# tau_c_ns: 10.0\n# proton_mhz: 600.0\n# site_pos: 1.0 2.0 3.0\n# site_domain: 1\n"),
        ),
    ] {
        let sites = parse_domain_pre_tsv_str(&text).unwrap();
        let ks = pre_k_values(&sites);
        assert_eq!(ks.len(), 2);
        for k in ks {
            assert!(
                ((k - k_gd) / k_gd).abs() < 1e-12,
                "21F:B4 follow-up ({label}): K_PRE = {k:.6e}, expected Gd3+ S=7/2 {k_gd:.6e}"
            );
        }
    }
    // Conflicting duplicate directives are ambiguous under order independence: fail closed.
    let dup = format!("{head}# spin_s: 3.5\n{rows}# spin_s: 0.5\n");
    let err = parse_domain_pre_tsv_str(&dup).expect_err("conflicting duplicate # spin_s must fail");
    assert!(err.contains("conflicting"), "got: {err}");
}

#[test]
fn test_fix3_b4_high_spin_pre_site_reports_effective_tau_c_caveat() {
    // Gd3+/Mn2+ PRE sites must carry a report note that tau_c is the effective correlation time
    // (1/tau_c = 1/tau_r + 1/T1e) with no automatic T1e model.
    let (native_model, _pae, part) = build_two_domain_ubq_kh3_benchmark();
    let kin = DomainLinkerKinematics::from_af_model(&native_model, &part).unwrap();
    let tsv = format!(
        "# site_pos: 1.0 2.0 3.0\n# site_domain: 1\n# tau_c_ns: 10.0\n# proton_mhz: 600.0\n# spin_s: 3.5\nseq_id\tatom_name\tgamma2_s1\tsigma_s1\n{}\tH\t20.0\t1.0\n",
        kin.residues[94].seq_id
    );
    let sites = parse_domain_pre_tsv_str(&tsv).unwrap();
    let cfg = DomainDockConfig {
        allow_underdetermined_single_medium: true,
        ..Default::default()
    };
    let mut empty_rdcs = Vec::new();
    let pipe =
        prepare_domain_dock_inputs_full(&kin, &part, &mut empty_rdcs, &[], &sites, &[], &cfg)
            .unwrap();
    assert!(
        pipe.extra_candor_notes
            .iter()
            .any(|n| n.contains("effective") && n.contains("T1e") && n.contains("pre1")),
        "missing effective-tau_c / T1e caveat for S=7/2 site: {:?}",
        pipe.extra_candor_notes
    );
    // A plain S = 1/2 nitroxide site does not get the high-spin caveat.
    let tsv_half = tsv.replace("# spin_s: 3.5\n", "");
    let sites_half = parse_domain_pre_tsv_str(&tsv_half).unwrap();
    let pipe_half =
        prepare_domain_dock_inputs_full(&kin, &part, &mut empty_rdcs, &[], &sites_half, &[], &cfg)
            .unwrap();
    assert!(!pipe_half
        .extra_candor_notes
        .iter()
        .any(|n| n.contains("T1e")));
}

#[test]
fn test_fix3_b5_reentry_closure_energy_has_own_breakdown_field() {
    // The always-on re-entry closure energy must not be booked into `rama`: at `w_rama = 0` the
    // breakdown shows `rama == 0` and the closure in `reentry_closure`, with an unchanged total.
    // Fails at `b5e87a3` (closure booked into `eb.rama`).
    let ubq = parse_pdb_first_model(repo_root().join("data/fixtures/ubiquitin_1ubq_xray.pdb"))
        .expect("parse 1UBQ");
    let (model0, _pae0, part0, _m0, _p0) = build_insertion_case_on_chain(&ubq, "b5_ref");
    let kin0 = DomainLinkerKinematics::from_af_model(&model0, &part0).unwrap();
    let mut st = kin0.initial_state();
    st.psi_rad[23] += 0.40;
    st.phi_rad[24] -= 0.35;
    st.psi_rad[46] += 0.30;
    let start_chain = kin0.build_pdb_chain(&st);
    let d_start = v_dist(start_chain.residues[49].c, start_chain.residues[50].n);
    eprintln!("21F:B5 perturbed start model C49-N50 = {d_start:.3} A");
    let (model, _pae, part, media, pcs_tag) = build_insertion_case_on_chain(&start_chain, "b5");
    let kin = DomainLinkerKinematics::from_af_model(&model, &part).unwrap();
    assert!(kin.has_reentry_blocks());
    assert!(
        !kin.residues[50].is_chain_start,
        "stretched {d_start:.2} A re-entry link must stay joined"
    );
    let mut rdc_media = media;
    let w0 = DomainDockEnergyWeights {
        w_rama: 0.0,
        ..Default::default()
    };
    let cfg = DomainDockConfig {
        weights: w0,
        ..Default::default()
    };
    let pipe = prepare_domain_dock_inputs_full(
        &kin,
        &part,
        &mut rdc_media,
        std::slice::from_ref(&pcs_tag),
        &[],
        &[],
        &cfg,
    )
    .unwrap();
    let st0 = kin.initial_state();
    assert!(kin.max_reentry_cn_err_angstrom(&st0) > 0.05);
    let (eb, _) =
        nmr_core::domain_dock::evaluate_domain_dock_state_cpu(&kin, &pipe.prep, &st0, &w0);
    assert_eq!(
        eb.rama, 0.0,
        "21F:B5 follow-up: w_rama = 0 must give rama = 0 (closure booked into rama: {:.4})",
        eb.rama
    );
    assert!(
        eb.reentry_closure > 0.0,
        "closure energy must be reported separately"
    );
    let sum = eb.rdc + eb.pcs + eb.pre + eb.noe + eb.vdw + eb.rama + eb.reentry_closure;
    assert!((eb.total - sum).abs() <= 1e-9 * eb.total.abs().max(1.0));
}
