//! Quantitative verification suite for `nmr_core::excited_state` (`T21.6b`,
//! `docs/specs/21-alphafold-integrative-nmr.md` sections 7.1 & 8).
//!
//! # Provenance & Clean-Room Statement
//! Clean-room test suite verifying:
//! 1. Exact normalization of signed Student-`t_5` (`(-inf, +inf)`) and unsigned folded
//!    Student-`t_5` (`[0, +inf)`) densities via high-order Simpson quadrature (`|I - 1| < 1e-4`).
//! 2. `100%` (`4/4` hypothesis classes) top-1 `BIC` recovery (`2 ln BF_{1,2} > 10.0`,
//!    `2 ln BF_{1,H0} > 10.0`, `reduced_chi2 <= 2.5`) across BOTH signed (`CEST` / `R1rho`) and
//!    unsigned (`CPMG` `|Delta_varpi|`) exchange tables on `1D3Z` Ubiquitin under the 4-layer
//!    `HomologFirewall` (`max_retained_seq_identity <= 0.25`):
//!    - `H1: LocalUnfolding` (`ncIDP` random-coil switch across `ILE23..GLU34`),
//!    - `H2: XxxProCisTrans` (`GLU18-PRO19` `trans -> cis`, Schubert `Delta_beta_gamma: 4.5 -> 9.6 ppm`),
//!    - `H3a: RotamerOrHisTautomerFlip` (steric-clash-checked `PHE` `chi1` Haigh-Mallion ring-current flip),
//!    - `H3b: RotamerOrHisTautomerFlip` (Pelton 1993 `HIS68` `+-82.0 ppm` `ND1`/`NE2` tautomer flip),
//!    - `H4: Alternate3DConformer` (candidate 3D conformer ensemble scored via `T21.2` `struct_shifts`).
//! 3. Mixed signed (`CEST`) + unsigned (`CPMG`) multi-nucleus tables and `ferric cest` / `ferric cpmg`
//!    TSV round-trip compatibility.
//! 4. Honest `AmbiguousOrUnmodeledState` abstention on pure noise (`10` seeds, signed & unsigned),
//!    sparse observations (`N_obs < 4`), unmodeled high-`chi^2` shifts, and hostile-input guards.

use std::path::PathBuf;

use nmr_core::excited_state::{
    classify_excited_state_with_firewall, format_excited_state_json, format_excited_state_tsv,
    parse_exchange_observations_tsv, parse_ground_shifts_str, signed_student_t5_log_likelihood,
    unsigned_folded_student_t5_log_likelihood, write_exchange_observations_tsv, AltConformerInput,
    ExchangeShiftObservation, ExchangeSignMode, ExcitedStateConfig, ExcitedStateHypothesisKind,
    ExcitedStateVerdict, HisExchangeTransition, HypothesisBestParameters, ProCisTransDirection,
};
use nmr_core::nmrstar::{read_star_all_atoms, Reference};
use nmr_core::rng::Rng;
use nmr_core::struct_shifts::{
    evaluate_haigh_mallion_ring_current, predict_structure_shifts_with_firewall,
    ConjugatedAromaticRing, StructShiftConfig, StructureModel,
};
use nmr_core::structure::nerf_place_atom;
use nmr_core::torsion_db::HomologFirewallQuery;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root")
}

fn ubiquitin_firewall() -> HomologFirewallQuery {
    HomologFirewallQuery {
        query_pdb_ids: vec!["1D3Z".to_string(), "1UBQ".to_string(), "1G6J".to_string()],
        query_bmrb_ids: vec!["6457".to_string(), "17769".to_string(), "5387".to_string()],
        query_uniprot_id: Some("P0CG48".to_string()),
        query_topology_id: Some("3.10.20.90".to_string()),
        ..HomologFirewallQuery::default()
    }
}

fn load_ubiquitin_ground() -> (StructureModel, Reference) {
    let root = workspace_root();
    let model =
        StructureModel::parse_file(root.join("data/pdb_refs/1d3z.pdb"), false).expect("1d3z.pdb");
    let shifts = read_star_all_atoms(
        root.join("data/fixtures/bmr17769_ubiquitin_allatom.str"),
        None,
    )
    .expect("bmr17769 all-atom shifts");
    (model, shifts)
}

fn simpson_integrate<F: Fn(f64) -> f64>(a: f64, b: f64, n_even: usize, f: F) -> f64 {
    assert!(n_even >= 4 && n_even.is_multiple_of(2));
    let h = (b - a) / (n_even as f64);
    let mut sum = f(a) + f(b);
    for k in 1..n_even {
        let x = a + (k as f64) * h;
        let w = if k % 2 == 1 { 4.0 } else { 2.0 };
        sum += w * f(x);
    }
    sum * (h / 3.0)
}

#[test]
fn test_student_t5_signed_and_folded_unsigned_density_normalization() {
    for &(mu, sigma) in &[
        (0.0, 0.42),
        (2.40, 0.45),
        (-3.15, 0.68),
        (0.85, 0.14),
        (82.0, 3.40),
    ] {
        let span = 120.0 * sigma;
        let int_signed = simpson_integrate(mu - span, mu + span, 8_000, |x| {
            signed_student_t5_log_likelihood(x, mu, sigma)
                .expect("signed ll")
                .exp()
        });
        assert!(
            (int_signed - 1.0).abs() < 1.0e-4,
            "signed Student-t5 PDF failed to integrate to 1.0 for (mu={mu}, sigma={sigma}): got {int_signed:.8}"
        );

        let upper_y = mu.abs() + span;
        let int_unsigned = simpson_integrate(0.0, upper_y, 8_000, |y| {
            unsigned_folded_student_t5_log_likelihood(y, mu, sigma)
                .expect("unsigned ll")
                .exp()
        });
        assert!(
            (int_unsigned - 1.0).abs() < 1.0e-4,
            "folded unsigned Student-t5 PDF failed to integrate to 1.0 on [0, inf) for (mu={mu}, sigma={sigma}): got {int_unsigned:.8}"
        );

        // Exact sign symmetry `f_fold(y | +mu) == f_fold(y | -mu)`
        for &y in &[0.0, 0.25 * sigma, mu.abs(), mu.abs() + 1.5 * sigma] {
            let ll_pos = unsigned_folded_student_t5_log_likelihood(y, mu, sigma).unwrap();
            let ll_neg = unsigned_folded_student_t5_log_likelihood(y, -mu, sigma).unwrap();
            assert!(
                (ll_pos - ll_neg).abs() < 1.0e-14,
                "folded unsigned Student-t5 must be invariant to sign(mu)"
            );
        }
    }

    // At mu == 0, folded density on [0, inf) equals `2 * f_signed(y | 0, sigma)`, i.e. `+ln(2)`
    let l_s = signed_student_t5_log_likelihood(0.75, 0.0, 0.42).unwrap();
    let l_u = unsigned_folded_student_t5_log_likelihood(0.75, 0.0, 0.42).unwrap();
    assert!(
        ((l_u - l_s) - std::f64::consts::LN_2).abs() < 1.0e-14,
        "at mu=0, folded density must carry exact +ln(2) normalization (no dropped constant)"
    );

    // Hostile parameter checks
    assert!(unsigned_folded_student_t5_log_likelihood(-0.01, 1.2, 0.4).is_err());
    assert!(signed_student_t5_log_likelihood(f64::NAN, 0.0, 0.4).is_err());
    assert!(signed_student_t5_log_likelihood(0.5, 0.0, 0.0).is_err());
    assert!(signed_student_t5_log_likelihood(0.5, 0.0, -0.2).is_err());
}

fn to_unsigned_table(signed_obs: &[ExchangeShiftObservation]) -> Vec<ExchangeShiftObservation> {
    signed_obs
        .iter()
        .map(|o| ExchangeShiftObservation {
            dw_ppm: o.dw_ppm.abs(),
            sign_mode: ExchangeSignMode::UnsignedMagnitude,
            major_ppm: None,
            ..o.clone()
        })
        .collect()
}

#[test]
fn test_four_hypothesis_bic_top1_recovery_signed_cest_and_unsigned_cpmg() {
    let (ubq_model, ubq_shifts) = load_ubiquitin_ground();
    let fw = ubiquitin_firewall();
    let cfg = ExcitedStateConfig::default();

    let struct_cfg = StructShiftConfig {
        sample_ph: cfg.sample_ph,
        temperature_kelvin: cfg.temperature_kelvin,
        include_methyl_ilvmat: true,
        ..StructShiftConfig::default()
    };
    let pred_a =
        predict_structure_shifts_with_firewall(&ubq_model, &fw, &struct_cfg).expect("pred_a");
    assert!(
        pred_a.max_retained_seq_identity <= 0.25,
        "HomologFirewall violated: max_retained_seq_identity={}",
        pred_a.max_retained_seq_identity
    );

    let mut rng = Rng::new(20262162);

    // =========================================================================
    // 1. H1: LocalUnfolding (Ubiquitin alpha-helix ILE23..GLU34 -> ncIDP coil)
    // =========================================================================
    let mut h1_signed = Vec::new();
    for (r_idx, r) in ubq_model.residues.iter().enumerate() {
        let sid = r.seq_id;
        // Include exchanging helix [22..=35] plus folded beta-sheet spectators [3..=7, 42..=45]
        if !(22..=35).contains(&sid) && !(3..=7).contains(&sid) && !(42..=45).contains(&sid) {
            continue;
        }
        let w = if (23..=34).contains(&sid) {
            1.0
        } else if sid == 22 || sid == 35 {
            0.5
        } else {
            0.0
        };
        let ref_idx = ubq_shifts
            .seq_ids
            .iter()
            .position(|&s| s == sid)
            .expect("seq_id in ubq_shifts");
        for &(atom, exp_sd) in &[("CA", 0.08), ("C", 0.08), ("N", 0.12), ("H", 0.025)] {
            if let (Some(&g_ppm), Some(bb)) = (
                ubq_shifts.shifts.get(&(ref_idx, atom.to_string())),
                pred_a.residues[r_idx].backbone.get(atom),
            ) {
                let true_dw = w * (bb.rc_ppm - g_ppm);
                let noisy_dw = true_dw + rng.normal(0.0, exp_sd);
                h1_signed.push(ExchangeShiftObservation {
                    chain_id: "A".to_string(),
                    seq_id: sid,
                    res_name: Some(r.res_name.clone()),
                    atom: atom.to_string(),
                    dw_ppm: noisy_dw,
                    dw_err_ppm: exp_sd,
                    sign_mode: ExchangeSignMode::Signed,
                    major_ppm: Some(g_ppm),
                });
            }
        }
    }
    let h1_unsigned = to_unsigned_table(&h1_signed);

    for (mode_label, table) in [("signed_CEST", &h1_signed), ("unsigned_CPMG", &h1_unsigned)] {
        let rep = classify_excited_state_with_firewall(
            Some(&ubq_shifts),
            &ubq_model,
            table,
            &[],
            &fw,
            &cfg,
        )
        .expect("classify H1");
        assert_eq!(
            rep.verdict,
            ExcitedStateVerdict::Committed(ExcitedStateHypothesisKind::LocalUnfolding),
            "H1 ({mode_label}) failed to commit LocalUnfolding: reason={}, ranked={:?}",
            rep.verdict_reason,
            rep.ranked_hypotheses
                .iter()
                .map(|h| (h.kind, h.bic, h.reduced_chi2))
                .collect::<Vec<_>>()
        );
        assert!(
            rep.two_ln_bf_top1_vs_top2 > 10.0,
            "H1 ({mode_label}) expected 2 ln BF_12 > 10.0, got {:.2}",
            rep.two_ln_bf_top1_vs_top2
        );
        println!(
            "EXCITED_STATE_H1_{mode_label}: 2lnBF_12={:.2} 2lnBF_null={:.2} rchi2={:.3}",
            rep.two_ln_bf_top1_vs_top2, rep.two_ln_bf_top1_vs_null, rep.top1_reduced_chi2
        );
        assert!(rep.two_ln_bf_top1_vs_null > 10.0);
        assert!(rep.top1_reduced_chi2 <= 2.5);
        match &rep.ranked_hypotheses[0].parameters {
            HypothesisBestParameters::LocalUnfolding {
                start_seq_id,
                end_seq_id,
                ..
            } => {
                assert!(
                    (*start_seq_id - 23).abs() <= 1 && (*end_seq_id - 34).abs() <= 1,
                    "H1 ({mode_label}) recovered unexpected window {start_seq_id}..{end_seq_id}"
                );
            }
            other => panic!("expected LocalUnfolding parameters, got {other:?}"),
        }
    }

    // =========================================================================
    // 2. H2: XxxProCisTrans (GLU18-PRO19 trans -> cis, Delta_bg: 4.5 -> 9.6 ppm)
    // =========================================================================
    let mut h2_signed = vec![
        // PRO19: Schubert 2002 canonical trans -> cis shifts
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 19,
            res_name: Some("PRO".to_string()),
            atom: "CB".to_string(),
            dw_ppm: 2.42 + rng.normal(0.0, 0.07),
            dw_err_ppm: 0.08,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: None,
        },
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 19,
            res_name: Some("PRO".to_string()),
            atom: "CG".to_string(),
            dw_ppm: -2.58 + rng.normal(0.0, 0.07),
            dw_err_ppm: 0.08,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: None,
        },
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 19,
            res_name: Some("PRO".to_string()),
            atom: "CA".to_string(),
            dw_ppm: -0.54 + rng.normal(0.0, 0.07),
            dw_err_ppm: 0.08,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: None,
        },
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 19,
            res_name: Some("PRO".to_string()),
            atom: "C".to_string(),
            dw_ppm: -0.84 + rng.normal(0.0, 0.07),
            dw_err_ppm: 0.08,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: None,
        },
        // Preceding GLU18
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 18,
            res_name: Some("GLU".to_string()),
            atom: "CA".to_string(),
            dw_ppm: -1.16 + rng.normal(0.0, 0.07),
            dw_err_ppm: 0.08,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: None,
        },
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 18,
            res_name: Some("GLU".to_string()),
            atom: "C".to_string(),
            dw_ppm: -1.33 + rng.normal(0.0, 0.07),
            dw_err_ppm: 0.08,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: None,
        },
        // Following SER20
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 20,
            res_name: Some("SER".to_string()),
            atom: "N".to_string(),
            dw_ppm: -1.80 + rng.normal(0.0, 0.12),
            dw_err_ppm: 0.12,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: None,
        },
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 20,
            res_name: Some("SER".to_string()),
            atom: "H".to_string(),
            dw_ppm: -0.30 + rng.normal(0.0, 0.03),
            dw_err_ppm: 0.03,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: None,
        },
    ];
    // Add unperturbed spectators at Pro37/Pro38 and folded core
    for &(sid, rn, at) in &[
        (37, "PRO", "CB"),
        (37, "PRO", "CG"),
        (38, "PRO", "CB"),
        (26, "VAL", "CA"),
    ] {
        h2_signed.push(ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: sid,
            res_name: Some(rn.to_string()),
            atom: at.to_string(),
            dw_ppm: rng.normal(0.0, 0.05),
            dw_err_ppm: 0.08,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: None,
        });
    }
    let h2_unsigned = to_unsigned_table(&h2_signed);

    for (mode_label, table) in [("signed_CEST", &h2_signed), ("unsigned_CPMG", &h2_unsigned)] {
        let rep = classify_excited_state_with_firewall(
            Some(&ubq_shifts),
            &ubq_model,
            table,
            &[],
            &fw,
            &cfg,
        )
        .expect("classify H2");
        assert_eq!(
            rep.verdict,
            ExcitedStateVerdict::Committed(ExcitedStateHypothesisKind::XxxProCisTrans),
            "H2 ({mode_label}) failed to commit XxxProCisTrans: reason={}, ranked={:?}",
            rep.verdict_reason,
            rep.ranked_hypotheses
                .iter()
                .map(|h| (h.kind, h.bic, h.reduced_chi2))
                .collect::<Vec<_>>()
        );
        assert!(
            rep.two_ln_bf_top1_vs_top2 > 10.0,
            "H2 ({mode_label}) expected 2 ln BF_12 > 10.0, got {:.2}",
            rep.two_ln_bf_top1_vs_top2
        );
        println!(
            "EXCITED_STATE_H2_{mode_label}: 2lnBF_12={:.2} 2lnBF_null={:.2} rchi2={:.3}",
            rep.two_ln_bf_top1_vs_top2, rep.two_ln_bf_top1_vs_null, rep.top1_reduced_chi2
        );
        match &rep.ranked_hypotheses[0].parameters {
            HypothesisBestParameters::XxxProCisTrans {
                pro_seq_id,
                direction,
                ..
            } => {
                assert_eq!(*pro_seq_id, 19);
                assert_eq!(*direction, ProCisTransDirection::TransToCis);
            }
            other => panic!("expected XxxProCisTrans parameters, got {other:?}"),
        }
    }

    // =========================================================================
    // 3a. H3: RotamerOrHisTautomerFlip -- PHE45 chi1 Haigh-Mallion ring-current flip
    // =========================================================================
    let phe_idx = ubq_model
        .residues
        .iter()
        .position(|r| r.seq_id == 45 && r.res_name == "PHE")
        .expect("PHE45");
    let r_phe = &ubq_model.residues[phe_idx];
    let cb_phe = r_phe.cb.expect("PHE45 CB");
    // Build steric-clash-free rotated PHE45 ring at chi1 = +60 deg (g+, flipping out into loop 45-49 solvent pocket), chi2 = +90 deg
    let chi1_deg_target = 60.0_f64;
    let chi1_rot = chi1_deg_target.to_radians();
    let chi2_rot = 90.0_f64.to_radians();
    let cg = nerf_place_atom(
        r_phe.n,
        r_phe.ca,
        cb_phe,
        1.502,
        113.8_f64.to_radians(),
        chi1_rot,
    );
    let cd1 = nerf_place_atom(
        r_phe.ca,
        cb_phe,
        cg,
        1.390,
        120.0_f64.to_radians(),
        chi2_rot,
    );
    let cd2 = nerf_place_atom(
        r_phe.ca,
        cb_phe,
        cg,
        1.390,
        120.0_f64.to_radians(),
        chi2_rot + std::f64::consts::PI,
    );
    let ce1 = nerf_place_atom(
        cb_phe,
        cg,
        cd1,
        1.390,
        120.0_f64.to_radians(),
        std::f64::consts::PI,
    );
    let ce2 = nerf_place_atom(
        cb_phe,
        cg,
        cd2,
        1.390,
        120.0_f64.to_radians(),
        std::f64::consts::PI,
    );
    let cz = nerf_place_atom(cg, cd1, ce1, 1.390, 120.0_f64.to_radians(), 0.0);
    let verts = vec![cg, cd1, ce1, cz, ce2, cd2];
    let cen = [
        verts.iter().map(|v| v[0]).sum::<f64>() / 6.0,
        verts.iter().map(|v| v[1]).sum::<f64>() / 6.0,
        verts.iter().map(|v| v[2]).sum::<f64>() / 6.0,
    ];
    let mut n_acc = [0.0_f64; 3];
    for k in 0..6 {
        let u = [
            verts[k][0] - cen[0],
            verts[k][1] - cen[1],
            verts[k][2] - cen[2],
        ];
        let v = [
            verts[(k + 1) % 6][0] - cen[0],
            verts[(k + 1) % 6][1] - cen[1],
            verts[(k + 1) % 6][2] - cen[2],
        ];
        n_acc[0] += u[1] * v[2] - u[2] * v[1];
        n_acc[1] += u[2] * v[0] - u[0] * v[2];
        n_acc[2] += u[0] * v[1] - u[1] * v[0];
    }
    let n_len = (n_acc[0] * n_acc[0] + n_acc[1] * n_acc[1] + n_acc[2] * n_acc[2]).sqrt();
    let mut rot_rings: Vec<ConjugatedAromaticRing> = ubq_model
        .aromatic_rings
        .iter()
        .filter(|rg| rg.residue_index != phe_idx)
        .cloned()
        .collect();
    rot_rings.push(ConjugatedAromaticRing {
        chain_id: "A".to_string(),
        seq_id: 45,
        residue_index: phe_idx,
        res_name: "PHE".to_string(),
        ring_label: "6-ring".to_string(),
        vertices: verts,
        center: cen,
        normal: [n_acc[0] / n_len, n_acc[1] / n_len, n_acc[2] / n_len],
        intensity_factor: 1.00,
    });

    let mut h3a_signed = Vec::new();
    // Self gamma-gauche CB shift on PHE45 (`g- -> g+`: `-1.65 ppm`)
    h3a_signed.push(ExchangeShiftObservation {
        chain_id: "A".to_string(),
        seq_id: 45,
        res_name: Some("PHE".to_string()),
        atom: "CB".to_string(),
        dw_ppm: -1.65 + rng.normal(0.0, 0.06),
        dw_err_ppm: 0.08,
        sign_mode: ExchangeSignMode::Signed,
        major_ppm: None,
    });
    for (idx, r) in ubq_model.residues.iter().enumerate() {
        let d_cb = ((r.ca[0] - cb_phe[0]).powi(2)
            + (r.ca[1] - cb_phe[1]).powi(2)
            + (r.ca[2] - cb_phe[2]).powi(2))
        .sqrt();
        if d_cb > 8.8 {
            continue;
        }
        if let Some(h_pt) = r.h {
            let (rc0, _) = evaluate_haigh_mallion_ring_current(
                &ubq_model.aromatic_rings,
                h_pt,
                idx,
                true,
                1.0,
            );
            let (rc1, _) = evaluate_haigh_mallion_ring_current(&rot_rings, h_pt, idx, true, 1.0);
            let d_h = rc1.clamp(-1.80, 1.40) - rc0.clamp(-1.80, 1.40);
            h3a_signed.push(ExchangeShiftObservation {
                chain_id: "A".to_string(),
                seq_id: r.seq_id,
                res_name: Some(r.res_name.clone()),
                atom: "H".to_string(),
                dw_ppm: d_h + rng.normal(0.0, 0.015),
                dw_err_ppm: 0.02,
                sign_mode: ExchangeSignMode::Signed,
                major_ppm: None,
            });
        }
        let (ha0, _) =
            evaluate_haigh_mallion_ring_current(&ubq_model.aromatic_rings, r.ha, idx, true, 1.0);
        let (ha1, _) = evaluate_haigh_mallion_ring_current(&rot_rings, r.ha, idx, true, 1.0);
        let d_ha = ha1.clamp(-1.50, 1.20) - ha0.clamp(-1.50, 1.20);
        h3a_signed.push(ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: r.seq_id,
            res_name: Some(r.res_name.clone()),
            atom: "HA".to_string(),
            dw_ppm: d_ha + rng.normal(0.0, 0.015),
            dw_err_ppm: 0.02,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: None,
        });
        // Include surrounding ILVMAT methyl protons (which sit right above/beside PHE45's aromatic face!)
        for m in &pred_a.residues[idx].methyls {
            let h_names = m.kind.proton_atoms();
            let h_pts: Vec<[f64; 3]> = h_names
                .iter()
                .filter_map(|an| r.atoms.get(*an).copied())
                .collect();
            if h_pts.len() == 3 {
                let is_self_arom = matches!(r.res_name.as_str(), "PHE" | "TYR" | "HIS" | "TRP");
                let rc0: f64 = h_pts
                    .iter()
                    .map(|&hp| {
                        evaluate_haigh_mallion_ring_current(
                            &ubq_model.aromatic_rings,
                            hp,
                            idx,
                            is_self_arom,
                            1.0,
                        )
                        .0
                    })
                    .sum::<f64>()
                    / 3.0;
                let rc1: f64 = h_pts
                    .iter()
                    .map(|&hp| {
                        evaluate_haigh_mallion_ring_current(&rot_rings, hp, idx, is_self_arom, 1.0)
                            .0
                    })
                    .sum::<f64>()
                    / 3.0;
                let d_hm = rc1.clamp(-1.85, 1.25) - rc0.clamp(-1.85, 1.25);
                h3a_signed.push(ExchangeShiftObservation {
                    chain_id: "A".to_string(),
                    seq_id: r.seq_id,
                    res_name: Some(r.res_name.clone()),
                    atom: m.proton_atom.clone(),
                    dw_ppm: d_hm + rng.normal(0.0, 0.012),
                    dw_err_ppm: 0.018,
                    sign_mode: ExchangeSignMode::Signed,
                    major_ppm: None,
                });
            }
        }
        if idx != phe_idx {
            // Secondary structure backbone CA remains intact (`~0 ppm` change, ruling out LocalUnfolding)
            let (ca0, _) = evaluate_haigh_mallion_ring_current(
                &ubq_model.aromatic_rings,
                r.ca,
                idx,
                true,
                0.55,
            );
            let (ca1, _) = evaluate_haigh_mallion_ring_current(&rot_rings, r.ca, idx, true, 0.55);
            let d_ca_ppm = ca1.clamp(-1.0, 1.0) - ca0.clamp(-1.0, 1.0);
            h3a_signed.push(ExchangeShiftObservation {
                chain_id: "A".to_string(),
                seq_id: r.seq_id,
                res_name: Some(r.res_name.clone()),
                atom: "CA".to_string(),
                dw_ppm: d_ca_ppm + rng.normal(0.0, 0.05),
                dw_err_ppm: 0.07,
                sign_mode: ExchangeSignMode::Signed,
                major_ppm: None,
            });
        }
    }
    let h3a_unsigned = to_unsigned_table(&h3a_signed);

    for (mode_label, table) in [
        ("signed_CEST", &h3a_signed),
        ("unsigned_CPMG", &h3a_unsigned),
    ] {
        let rep = classify_excited_state_with_firewall(
            Some(&ubq_shifts),
            &ubq_model,
            table,
            &[],
            &fw,
            &cfg,
        )
        .expect("classify H3a PHE45");
        assert_eq!(
            rep.verdict,
            ExcitedStateVerdict::Committed(ExcitedStateHypothesisKind::RotamerOrHisTautomerFlip),
            "H3a PHE45 ({mode_label}) failed to commit RotamerOrHisTautomerFlip: reason={}, ranked={:?}",
            rep.verdict_reason,
            rep.ranked_hypotheses
                .iter()
                .map(|h| (h.kind, h.bic, h.reduced_chi2, &h.best_locus_summary))
                .collect::<Vec<_>>()
        );
        assert!(
            rep.two_ln_bf_top1_vs_top2 > 10.0,
            "H3a PHE45 ({mode_label}) expected 2 ln BF_12 > 10.0, got {:.2}",
            rep.two_ln_bf_top1_vs_top2
        );
        println!(
            "EXCITED_STATE_H3a_{mode_label}: 2lnBF_12={:.2} 2lnBF_null={:.2} rchi2={:.3}",
            rep.two_ln_bf_top1_vs_top2, rep.two_ln_bf_top1_vs_null, rep.top1_reduced_chi2
        );
        match &rep.ranked_hypotheses[0].parameters {
            HypothesisBestParameters::AromaticRotamerFlip {
                seq_id,
                res_name,
                min_steric_dist_angstrom,
                ..
            } => {
                assert_eq!(*seq_id, 45);
                assert_eq!(res_name, "PHE");
                assert!(*min_steric_dist_angstrom >= 2.00);
            }
            other => panic!("expected AromaticRotamerFlip on PHE45, got {other:?}"),
        }
    }

    // =========================================================================
    // 3b. H3: RotamerOrHisTautomerFlip -- Pelton 1993 HIS68 +-82 ppm ND1/NE2 flip
    // =========================================================================
    let h3b_signed = vec![
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 68,
            res_name: Some("HIS".to_string()),
            atom: "ND1".to_string(),
            dw_ppm: -81.6 + rng.normal(0.0, 0.50),
            dw_err_ppm: 0.60,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: Some(249.5),
        },
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 68,
            res_name: Some("HIS".to_string()),
            atom: "NE2".to_string(),
            dw_ppm: 81.9 + rng.normal(0.0, 0.50),
            dw_err_ppm: 0.60,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: Some(168.5),
        },
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 68,
            res_name: Some("HIS".to_string()),
            atom: "CD2".to_string(),
            dw_ppm: 7.15 + rng.normal(0.0, 0.15),
            dw_err_ppm: 0.20,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: Some(118.2),
        },
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 68,
            res_name: Some("HIS".to_string()),
            atom: "CG".to_string(),
            dw_ppm: -4.75 + rng.normal(0.0, 0.15),
            dw_err_ppm: 0.20,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: Some(135.5),
        },
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 68,
            res_name: Some("HIS".to_string()),
            atom: "CB".to_string(),
            dw_ppm: -2.15 + rng.normal(0.0, 0.10),
            dw_err_ppm: 0.15,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: None,
        },
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 67,
            res_name: Some("LEU".to_string()),
            atom: "CA".to_string(),
            dw_ppm: 0.04,
            dw_err_ppm: 0.08,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: None,
        },
    ];
    let h3b_unsigned = to_unsigned_table(&h3b_signed);

    for (mode_label, table) in [
        ("signed_CEST", &h3b_signed),
        ("unsigned_CPMG", &h3b_unsigned),
    ] {
        let rep = classify_excited_state_with_firewall(
            Some(&ubq_shifts),
            &ubq_model,
            table,
            &[],
            &fw,
            &cfg,
        )
        .expect("classify H3b HIS68");
        assert_eq!(
            rep.verdict,
            ExcitedStateVerdict::Committed(ExcitedStateHypothesisKind::RotamerOrHisTautomerFlip),
            "H3b HIS68 ({mode_label}) failed to commit: reason={}",
            rep.verdict_reason
        );
        assert!(
            rep.two_ln_bf_top1_vs_top2 > 10.0,
            "H3b HIS68 ({mode_label}) expected 2 ln BF_12 > 10.0, got {:.2}",
            rep.two_ln_bf_top1_vs_top2
        );
        println!(
            "EXCITED_STATE_H3b_{mode_label}: 2lnBF_12={:.2} 2lnBF_null={:.2} rchi2={:.3}",
            rep.two_ln_bf_top1_vs_top2, rep.two_ln_bf_top1_vs_null, rep.top1_reduced_chi2
        );
        match &rep.ranked_hypotheses[0].parameters {
            HypothesisBestParameters::HisTautomerOrProtonation {
                seq_id, transition, ..
            } => {
                assert_eq!(*seq_id, 68);
                assert!(matches!(
                    transition,
                    HisExchangeTransition::Ne2HToNd1H | HisExchangeTransition::Nd1HToNe2H
                ));
            }
            other => panic!("expected HisTautomerOrProtonation on HIS68, got {other:?}"),
        }
    }

    // =========================================================================
    // 4. H4: Alternate3DConformer (1 true switched 3D conformer + 3 decoys)
    // =========================================================================
    let seq_ids: Vec<i32> = ubq_model.residues.iter().map(|r| r.seq_id).collect();
    let res_names: Vec<String> = ubq_model
        .residues
        .iter()
        .map(|r| r.res_name.clone())
        .collect();
    let ca_coords: Vec<[f64; 3]> = ubq_model.residues.iter().map(|r| r.ca).collect();
    let base_phi: Vec<Option<f64>> = ubq_model.residues.iter().map(|r| r.phi_deg).collect();
    let base_psi: Vec<Option<f64>> = ubq_model.residues.iter().map(|r| r.psi_deg).collect();
    let base_chi1: Vec<Option<f64>> = ubq_model.residues.iter().map(|r| r.chi1_deg).collect();

    // True excited conformer: beta-turn/strand segment 7..=15 flips into alpha-helical (-62, -42) basin
    // (opposite direction from random coil, so H1 LocalUnfolding cannot mimic it!)
    let make_switched_model =
        |range: std::ops::RangeInclusive<i32>, target_phi: f64, target_psi: f64| {
            let mut phi = base_phi.clone();
            let mut psi = base_psi.clone();
            for (k, &sid) in seq_ids.iter().enumerate() {
                if range.contains(&sid) && phi[k].is_some() && psi[k].is_some() {
                    phi[k] = Some(target_phi);
                    psi[k] = Some(target_psi);
                }
            }
            StructureModel::from_ca_and_torsions(
                "A", &seq_ids, &res_names, &ca_coords, &phi, &psi, &base_chi1,
            )
            .expect("reconstruct switched conformer")
        };

    let ground_recon = StructureModel::from_ca_and_torsions(
        "A", &seq_ids, &res_names, &ca_coords, &base_phi, &base_psi, &base_chi1,
    )
    .expect("reconstruct ground");
    let alt_true = make_switched_model(7..=15, -62.0, -42.0);
    let alt_decoy1 = make_switched_model(48..=56, -62.0, -42.0);
    let alt_decoy2 = make_switched_model(25..=32, -125.0, 135.0);
    let alt_decoy3 = make_switched_model(62..=70, -60.0, -40.0);

    let pred_g_recon =
        predict_structure_shifts_with_firewall(&ground_recon, &fw, &struct_cfg).unwrap();
    let pred_b_true = predict_structure_shifts_with_firewall(&alt_true, &fw, &struct_cfg).unwrap();

    let mut h4_signed = Vec::new();
    for (idx, r) in ground_recon.residues.iter().enumerate() {
        if !(5..=17).contains(&r.seq_id) && !(26..=30).contains(&r.seq_id) {
            continue;
        }
        for &(atom, exp_sd) in &[("CA", 0.08), ("CB", 0.09), ("C", 0.08), ("N", 0.15)] {
            if let (Some(ga), Some(gb)) = (
                pred_g_recon.residues[idx].backbone.get(atom),
                pred_b_true.residues[idx].backbone.get(atom),
            ) {
                let true_dw = gb.struct_ppm - ga.struct_ppm;
                h4_signed.push(ExchangeShiftObservation {
                    chain_id: "A".to_string(),
                    seq_id: r.seq_id,
                    res_name: Some(r.res_name.clone()),
                    atom: atom.to_string(),
                    dw_ppm: true_dw + rng.normal(0.0, exp_sd),
                    dw_err_ppm: exp_sd,
                    sign_mode: ExchangeSignMode::Signed,
                    major_ppm: Some(ga.struct_ppm),
                });
            }
        }
    }
    let h4_unsigned = to_unsigned_table(&h4_signed);
    let alt_bundle = vec![
        AltConformerInput {
            label: "decoy_loop48_56.pdb".to_string(),
            model: alt_decoy1,
        },
        AltConformerInput {
            label: "native_excited_hairpin7_15.pdb".to_string(),
            model: alt_true,
        },
        AltConformerInput {
            label: "decoy_helix25_32_strand.pdb".to_string(),
            model: alt_decoy2,
        },
        AltConformerInput {
            label: "decoy_cterm62_70.pdb".to_string(),
            model: alt_decoy3,
        },
    ];

    for (mode_label, table) in [("signed_CEST", &h4_signed), ("unsigned_CPMG", &h4_unsigned)] {
        let rep = classify_excited_state_with_firewall(
            None,
            &ground_recon,
            table,
            &alt_bundle,
            &fw,
            &cfg,
        )
        .expect("classify H4");
        assert_eq!(
            rep.verdict,
            ExcitedStateVerdict::Committed(ExcitedStateHypothesisKind::Alternate3DConformer),
            "H4 ({mode_label}) failed to commit Alternate3DConformer: reason={}, ranked={:?}",
            rep.verdict_reason,
            rep.ranked_hypotheses
                .iter()
                .map(|h| (h.kind, h.bic, h.reduced_chi2))
                .collect::<Vec<_>>()
        );
        println!(
            "EXCITED_STATE_H4_{mode_label}: 2lnBF_12={:.2} 2lnBF_null={:.2} rchi2={:.3}",
            rep.two_ln_bf_top1_vs_top2, rep.two_ln_bf_top1_vs_null, rep.top1_reduced_chi2
        );
        assert!(
            rep.two_ln_bf_top1_vs_top2 > 10.0,
            "H4 ({mode_label}) expected 2 ln BF_12 > 10.0, got {:.2}",
            rep.two_ln_bf_top1_vs_top2
        );
        match &rep.ranked_hypotheses[0].parameters {
            HypothesisBestParameters::Alternate3DConformer {
                model_index,
                model_label,
                max_retained_seq_identity,
                exclusion_list_sha256,
            } => {
                assert_eq!(*model_index, 1);
                assert_eq!(model_label, "native_excited_hairpin7_15.pdb");
                assert!(*max_retained_seq_identity <= 0.25);
                assert_eq!(exclusion_list_sha256.len(), 64);
            }
            other => panic!("expected Alternate3DConformer parameters, got {other:?}"),
        }
    }
}

#[test]
fn test_mixed_cest_signed_and_cpmg_unsigned_table_and_tsv_parsers() {
    let (ubq_model, ubq_shifts) = load_ubiquitin_ground();
    let fw = ubiquitin_firewall();
    let cfg = ExcitedStateConfig::default();

    // Mixed table: signed 15N CEST + unsigned 13CA/13CB/13CG CPMG on GLU18-PRO19 cis-Pro switch
    let mixed = vec![
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 19,
            res_name: Some("PRO".to_string()),
            atom: "CB".to_string(),
            dw_ppm: 2.40,
            dw_err_ppm: 0.08,
            sign_mode: ExchangeSignMode::UnsignedMagnitude,
            major_ppm: None,
        },
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 19,
            res_name: Some("PRO".to_string()),
            atom: "CG".to_string(),
            dw_ppm: 2.60,
            dw_err_ppm: 0.08,
            sign_mode: ExchangeSignMode::UnsignedMagnitude,
            major_ppm: None,
        },
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 18,
            res_name: Some("GLU".to_string()),
            atom: "CA".to_string(),
            dw_ppm: 1.15,
            dw_err_ppm: 0.08,
            sign_mode: ExchangeSignMode::UnsignedMagnitude,
            major_ppm: None,
        },
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 18,
            res_name: Some("GLU".to_string()),
            atom: "C".to_string(),
            dw_ppm: -1.35,
            dw_err_ppm: 0.08,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: None,
        },
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 20,
            res_name: Some("SER".to_string()),
            atom: "N".to_string(),
            dw_ppm: -1.85,
            dw_err_ppm: 0.12,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: Some(116.2),
        },
    ];

    let tsv = write_exchange_observations_tsv(&mixed);
    let reparsed = parse_exchange_observations_tsv(&tsv).expect("reparse canonical TSV");
    assert_eq!(reparsed.len(), mixed.len());
    assert_eq!(reparsed[0].sign_mode, ExchangeSignMode::UnsignedMagnitude);
    assert_eq!(reparsed[3].sign_mode, ExchangeSignMode::Signed);

    let rep = classify_excited_state_with_firewall(
        Some(&ubq_shifts),
        &ubq_model,
        &reparsed,
        &[],
        &fw,
        &cfg,
    )
    .expect("classify mixed");
    assert_eq!(rep.n_unsigned_observations, 3);
    assert_eq!(rep.n_signed_observations, 2);
    assert_eq!(
        rep.verdict,
        ExcitedStateVerdict::Committed(ExcitedStateHypothesisKind::XxxProCisTrans)
    );

    // Verify JSON and TSV report formatters serialize deterministically
    let out_tsv = format_excited_state_tsv(&rep);
    assert!(out_tsv.contains("Committed(XxxProCisTrans)"));
    let out_json = format_excited_state_json(&rep).expect("json");
    assert!(out_json.contains("\"XxxProCisTrans\""));

    // Direct `ferric cest` and `ferric cpmg` output TSV compatibility
    let cest_fit_tsv = "# p_b=0.020000\n\
seq_id\tres_name\tmajor_ppm\tmajor_ppm_err\tdw_ab_ppm\tdw_ab_ppm_err\tr1a_s_inv\tr1a_err\tr2a_s_inv\tr2a_err\tr2b_s_inv\tr2b_err\tchi2\tn_points\n\
20\tSER\t116.2000\t0.0200\t-1.8500\t0.0900\t1.4000\t0.0200\t12.0000\t0.2000\t18.0000\t0.8000\t22.1000\t36\n";
    let cest_rows = parse_exchange_observations_tsv(cest_fit_tsv).expect("parse cest fit tsv");
    assert_eq!(cest_rows.len(), 1);
    assert_eq!(cest_rows[0].atom, "N");
    assert_eq!(cest_rows[0].sign_mode, ExchangeSignMode::Signed);
    assert!((cest_rows[0].dw_ppm - (-1.85)).abs() < 1e-6);

    let cpmg_fit_tsv = "# model=baldwin2014\n\
seq_id\tres_name\tdw_ppm\tdw_ppm_err\talpha_rex\tsignificant_exchange\tdelta_aic\tf_pvalue\tchi2\tn_points\n\
20\tSER\t1.8500\t0.0900\t1.120\ttrue\t18.400\t1.2000e-4\t11.2000\t24\n\
25\tASN\t0.0800\t0.1200\tNA\tfalse\t-1.800\t8.5000e-1\t9.4000\t24\n";
    let cpmg_rows = parse_exchange_observations_tsv(cpmg_fit_tsv).expect("parse cpmg fit tsv");
    assert_eq!(
        cpmg_rows.len(),
        1,
        "non-significant CPMG rows must be filtered"
    );
    assert_eq!(cpmg_rows[0].sign_mode, ExchangeSignMode::UnsignedMagnitude);

    // Parse ground shifts via `parse_ground_shifts_str` (.str auto-detect)
    let raw_star = std::fs::read_to_string(
        workspace_root().join("data/fixtures/bmr17769_ubiquitin_allatom.str"),
    )
    .unwrap();
    let parsed_ref = parse_ground_shifts_str(&raw_star, None).expect("parse star shifts");
    assert_eq!(parsed_ref.sequence.len(), 76);
}

#[test]
fn test_pure_noise_sparse_and_unmodeled_state_abstention_and_hostile_inputs() {
    let (ubq_model, ubq_shifts) = load_ubiquitin_ground();
    let fw = ubiquitin_firewall();
    let cfg = ExcitedStateConfig::default();

    // 1. Pure Gaussian zero-mean noise across 10 independent seeds (both Signed and Unsigned)
    for seed in 0..10u64 {
        let mut rng = Rng::new(9000 + seed);
        let mut noise_signed = Vec::new();
        for sid in [4, 12, 18, 19, 20, 25, 28, 31, 45, 68] {
            let r = ubq_model
                .residues
                .iter()
                .find(|x| x.seq_id == sid)
                .expect("res");
            noise_signed.push(ExchangeShiftObservation {
                chain_id: "A".to_string(),
                seq_id: sid,
                res_name: Some(r.res_name.clone()),
                atom: "N".to_string(),
                dw_ppm: rng.normal(0.0, 0.12),
                dw_err_ppm: 0.12,
                sign_mode: ExchangeSignMode::Signed,
                major_ppm: None,
            });
            noise_signed.push(ExchangeShiftObservation {
                chain_id: "A".to_string(),
                seq_id: sid,
                res_name: Some(r.res_name.clone()),
                atom: "CA".to_string(),
                dw_ppm: rng.normal(0.0, 0.08),
                dw_err_ppm: 0.08,
                sign_mode: ExchangeSignMode::Signed,
                major_ppm: None,
            });
        }
        let noise_unsigned = to_unsigned_table(&noise_signed);

        let rep_s = classify_excited_state_with_firewall(
            Some(&ubq_shifts),
            &ubq_model,
            &noise_signed,
            &[],
            &fw,
            &cfg,
        )
        .expect("noise signed");
        assert_eq!(
            rep_s.verdict,
            ExcitedStateVerdict::AmbiguousOrUnmodeledState,
            "seed {seed} signed pure noise falsely committed: {}",
            rep_s.verdict_reason
        );

        let rep_u = classify_excited_state_with_firewall(
            Some(&ubq_shifts),
            &ubq_model,
            &noise_unsigned,
            &[],
            &fw,
            &cfg,
        )
        .expect("noise unsigned");
        assert_eq!(
            rep_u.verdict,
            ExcitedStateVerdict::AmbiguousOrUnmodeledState,
            "seed {seed} unsigned pure noise falsely committed: {}",
            rep_u.verdict_reason
        );
    }

    // 2. Sparse observations (`N_obs = 3 < 4`) must abstain even on exact cis-Pro signal
    let sparse = vec![
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 19,
            res_name: Some("PRO".to_string()),
            atom: "CB".to_string(),
            dw_ppm: 2.40,
            dw_err_ppm: 0.06,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: None,
        },
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 19,
            res_name: Some("PRO".to_string()),
            atom: "CG".to_string(),
            dw_ppm: -2.60,
            dw_err_ppm: 0.06,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: None,
        },
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 18,
            res_name: Some("GLU".to_string()),
            atom: "CA".to_string(),
            dw_ppm: -1.15,
            dw_err_ppm: 0.06,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: None,
        },
    ];
    let rep_sparse = classify_excited_state_with_firewall(
        Some(&ubq_shifts),
        &ubq_model,
        &sparse,
        &[],
        &fw,
        &cfg,
    )
    .expect("sparse");
    assert_eq!(
        rep_sparse.verdict,
        ExcitedStateVerdict::AmbiguousOrUnmodeledState
    );

    // 3. Unmodeled extreme chemical shift perturbations (`+8.5 ppm` on CA across alpha-helix,
    //    violating `reduced_chi2 <= 2.5` across all H0..H3) must abstain honestly
    let unmodeled: Vec<ExchangeShiftObservation> = (24..=30)
        .map(|sid| ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: sid,
            res_name: None,
            atom: "CA".to_string(),
            dw_ppm: 8.50,
            dw_err_ppm: 0.08,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: None,
        })
        .collect();
    let rep_unmod = classify_excited_state_with_firewall(
        Some(&ubq_shifts),
        &ubq_model,
        &unmodeled,
        &[],
        &fw,
        &cfg,
    )
    .expect("unmodeled");
    assert_eq!(
        rep_unmod.verdict,
        ExcitedStateVerdict::AmbiguousOrUnmodeledState
    );
    assert!(rep_unmod.top1_reduced_chi2 > 2.5);

    // 4. Hostile inputs return clean `Err`
    assert!(parse_exchange_observations_tsv("").is_err());
    assert!(parse_exchange_observations_tsv(
        "chain_id\tseq_id\tres_name\tatom\tdw_ppm\tdw_err_ppm\tsign_mode\nA\t19\tPRO\tCB\t-1.2\t0.1\tunsigned\n"
    )
    .is_err());
    assert!(parse_exchange_observations_tsv(
        "chain_id\tseq_id\tres_name\tatom\tdw_ppm\tdw_err_ppm\tsign_mode\nA\t19\tPRO\tCB\tNaN\t0.1\tsigned\n"
    )
    .is_err());
    assert!(classify_excited_state_with_firewall(
        Some(&ubq_shifts),
        &ubq_model,
        &[ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 19,
            res_name: Some("ALA".to_string()), // residue name mismatch vs PRO19
            atom: "CB".to_string(),
            dw_ppm: 1.0,
            dw_err_ppm: 0.1,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: None,
        }],
        &[],
        &fw,
        &cfg,
    )
    .is_err());
}
