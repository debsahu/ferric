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
    parse_af_structure_model_str, AfIngestOptions, PaeJsonDialect, PaeMatrix,
};
use nmr_core::domain_dock::{
    adjusted_rand_index, cluster_pae_domains, dock_domains_cpu,
    evaluate_domain_dock_fixed_saupe_cpu, evaluate_tolman_domain_gdo, format_docked_mmcif_ensemble,
    format_docked_pdb_ensemble, format_domain_dock_json, format_domain_dock_tsv,
    parse_domain_pcs_tsv_str, parse_domain_pre_tsv_str, partition_domain_stratified_rdc_free_set,
    quat_apply_world_rot_step, DipolarObservabilityRegime, DomainDockConfig,
    DomainDockEnergyWeights, DomainLinkerKinematics, DomainMotionRegime, DomainPcsRestraint,
    DomainPcsTagSet, DomainPreRestraint, DomainPreSiteSet, PaeClusterConfig, PaeDomainPartition,
    PreRestraintKind, PreparedDomainDockRestraints, ResidueDomainRole,
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
        anchor_residue_atom: None,
        host_domain_id: Some(2),
        pre_fitted_tensor: None,
        restraints: pcs_rows,
    };

    let s_pos_d1 = v_add(ch0.residues[20].ca, [-3.0, 2.5, 2.5]);
    let pre_site = DomainPreSiteSet {
        site_id: "mtsl_d1".to_string(),
        site_pos_initial: Some(s_pos_d1),
        anchor_residue_atom: None,
        host_domain_id: Some(1),
        restraints: vec![
            DomainPreRestraint {
                site_id: "mtsl_d1".to_string(),
                seq_id: 95,
                atom_name: "H".to_string(),
                kind: PreRestraintKind::Gamma2Rate {
                    gamma2_obs_s1: 18.0,
                    sigma_s1: 2.5,
                    k_pre_a6_s1: 1.23e7,
                },
            },
            DomainPreRestraint {
                site_id: "mtsl_d1".to_string(),
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

    // Solomon PRE Gamma2 site near Domain 1's interface (`residue 68`, `r^-6` in 12..25 A range)
    let pre_site_pos = v_add(native_chain.residues[67].ca, [-2.8, 2.5, 2.8]);
    let k_pre = 1.23e7_f64;
    let mut pre_tsv = format!(
        "# site_pos: {:.4} {:.4} {:.4}\n# site_domain: 1\n# k_pre: {:.6e}\nseq_id\tatom_name\tgamma2_s1\tsigma_s1\n",
        pre_site_pos[0], pre_site_pos[1], pre_site_pos[2], k_pre
    );
    for idx in (80..native_chain.residues.len()).step_by(4) {
        let r = &native_chain.residues[idx];
        if let Some(h) = r.h {
            let dist2 = v_dist(h, pre_site_pos).powi(2) + 4.0;
            let g2 = k_pre / (dist2 * dist2 * dist2);
            pre_tsv.push_str(&format!("{}\tH\t{:.4}\t1.50\n", r.seq_id, g2));
        }
    }
    let pre_sites = parse_domain_pre_tsv_str(&pre_tsv).unwrap();

    let cfg = DomainDockConfig {
        auto_partition_rdc_free: false, // preserve our pre-partitioned set for exact corruption comparison
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

    println!(
        "CPU T21.5 gate: start=({rot_angle_deg:.1} deg, {trans_shift_a:.1} A) -> docked BB_RMSD={all_dom_bb_rmsd:.3} A, Q_work={:.4}, ByPeptidePlane Q_free={:.4}, PCS Q_free={:.4}, intra_dist={:.3} A",
        res.report.global_rdc_q_work,
        res.report.global_rdc_q_free,
        res.report.pcs_q_free,
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
    let res_corrupt = dock_domains_cpu(
        &start_af_model,
        &pae,
        &corrupted_media,
        &pcs_tags,
        &pre_sites,
        &[],
        &cfg,
    )
    .unwrap();
    let leak_rmsd = backbone_rmsd_between_chains(&res.chains[0], &res_corrupt.chains[0], None);
    assert!(
        leak_rmsd < 1e-13,
        "held-out is_free RDC corruption altered docked coordinates by {leak_rmsd:.3e} A (must be < 1e-13 A)"
    );
    assert!((res.report.best_energy.total - res_corrupt.report.best_energy.total).abs() < 1e-12);

    // Single-medium `F8` cone-degeneracy candor check + `.pdb` / `.cif` / `.json` / `.tsv` roundtrip
    let mut cfg_1med = cfg.clone();
    cfg_1med.n_replicas = 8;
    cfg_1med.anneal_steps = 8;
    cfg_1med.lbfgs_iters = 15;
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
