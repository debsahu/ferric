//! Quantitative synthetic self-consistency & adversarial verification suite for
//! `nmr_core::excited_state` (`T21.6b`, `docs/specs/21-alphafold-integrative-nmr.md` sections 7.1 & 8).
//!
//! # Provenance & Clean-Room Statement
//! Clean-room synthetic self-consistency suite verifying:
//! 1. Exact normalization of signed Student-`t_5` (`(-inf, +inf)`) and unsigned folded
//!    Student-`t_5` (`[0, +inf)`) densities via high-order Simpson quadrature (`|I - 1| < 1e-4`).
//! 2. Synthetic self-consistency across `4/4` endpoint-only hypothesis classes with realistic
//!    forward-model perturbation noise (`reduced_chi2` near `1.0`), discrete-scan marginalized
//!    `BIC` (`ln mean L_m`, `n = n_distinct_residues`), and head-to-head non-null competition
//!    (`two_ln_bf_top1_vs_runnerup_non_null >= 6.0`, `two_ln_bf_top1_vs_null >= 10.0`) across
//!    signed (`CEST` / `R1rho`) and unsigned (`CPMG` `|Delta_varpi|`) tables on `1D3Z` Ubiquitin
//!    under the 4-layer `HomologFirewall` (`max_retained_seq_identity <= 0.25`):
//!    - `H1: LocalUnfolding` (`ncIDP` random-coil switch across `ILE23..GLU34`),
//!    - `H2: XxxProCisTrans` (`GLU18-PRO19` `trans -> cis`, Schubert `Delta_beta_gamma: 4.5 -> 9.6 ppm`),
//!    - `H3a: RotamerOrHisTautomerFlip` (steric-clash-checked `PHE` `chi1` Haigh-Mallion ring-current flip),
//!    - `H3b: RotamerOrHisTautomerFlip` (Pelton 1993 `HIS68` `+-82.0 ppm` `ND1`/`NE2`, `+-8.3 ppm` `CG` tautomer flip),
//!    - `H4: Alternate3DConformer` (candidate 3D conformer ensemble scored via `T21.2` `struct_shifts`).
//! 3. Duplicate `(chain_id, seq_id, atom)` rejection, `>= 4` distinct exchanging residue gate,
//!    experimental-only `H0` noise parameterization (`UnmodeledSignificantExchange` on 3 sites at
//!    `2.5 ppm`), `His` sidechain gate & unsigned direction ambiguity, and debug-build `u_best == 0` safety.

use std::path::PathBuf;

use nmr_core::excited_state::{
    classify_excited_state_with_firewall, format_excited_state_json, format_excited_state_tsv,
    parse_exchange_observations_tsv, parse_exchange_observations_tsv_with_meta,
    parse_ground_shifts_str, signed_student_t5_log_likelihood,
    unsigned_folded_student_t5_log_likelihood, write_exchange_observations_tsv, AltConformerInput,
    ExchangeShiftObservation, ExchangeSignMode, ExcitedStateAbstentionKind, ExcitedStateConfig,
    ExcitedStateHypothesisKind, ExcitedStateVerdict, HisExchangeTransition,
    HypothesisBestParameters, ProCisTransDirection,
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

    // Build H4 candidate 3D conformer ensemble upfront so ALL 5 benchmarks (H1, H2, H3a, H3b, H4)
    // pit every non-null mechanism (H1, H2, H3, H4) head-to-head simultaneously (`21.W3:P1-5`).
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
        for &(atom, exp_sd, fwd_sd) in &[
            ("CA", 0.08, 0.45),
            ("C", 0.08, 0.50),
            ("N", 0.12, 1.15),
            ("H", 0.025, 0.20),
        ] {
            if let (Some(&g_ppm), Some(bb)) = (
                ubq_shifts.shifts.get(&(ref_idx, atom.to_string())),
                pred_a.residues[r_idx].backbone.get(atom),
            ) {
                let true_dw = w * (bb.rc_ppm - g_ppm);
                let fwd_noise = if w > 0.0 {
                    rng.normal(0.0, w * fwd_sd)
                } else {
                    0.0
                };
                let noisy_dw = true_dw + fwd_noise + rng.normal(0.0, exp_sd);
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
            &alt_bundle,
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
        let bf_non_null = rep
            .two_ln_bf_top1_vs_runnerup_non_null
            .expect("non-null runnerup");
        assert!(
            rep.two_ln_bf_top1_vs_top2 > 10.0 && bf_non_null > 10.0,
            "H1 ({mode_label}) expected 2 ln BF_12 > 10.0 and vs non-null > 10.0, got {:.2} / {:.2}",
            rep.two_ln_bf_top1_vs_top2,
            bf_non_null
        );
        println!(
            "EXCITED_STATE_H1_{mode_label}: 2lnBF_12={:.2} 2lnBF_non_null={:.2} 2lnBF_null={:.2} rchi2={:.3}",
            rep.two_ln_bf_top1_vs_top2,
            bf_non_null,
            rep.two_ln_bf_top1_vs_null,
            rep.top1_reduced_chi2
        );
        assert!(rep.two_ln_bf_top1_vs_null > 10.0);
        assert!(
            (0.30..=2.5).contains(&rep.top1_reduced_chi2),
            "H1 ({mode_label}) expected realistic reduced_chi2 in [0.30, 2.5], got {:.3}",
            rep.top1_reduced_chi2
        );
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
        // PRO19: Schubert 2002 canonical trans -> cis shifts + realistic forward-model discrepancy
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 19,
            res_name: Some("PRO".to_string()),
            atom: "CB".to_string(),
            dw_ppm: 2.40 + rng.normal(0.0, 0.45) + rng.normal(0.0, 0.08),
            dw_err_ppm: 0.08,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: None,
        },
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 19,
            res_name: Some("PRO".to_string()),
            atom: "CG".to_string(),
            dw_ppm: -2.60 + rng.normal(0.0, 0.40) + rng.normal(0.0, 0.08),
            dw_err_ppm: 0.08,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: None,
        },
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 19,
            res_name: Some("PRO".to_string()),
            atom: "CA".to_string(),
            dw_ppm: -0.55 + rng.normal(0.0, 0.35) + rng.normal(0.0, 0.08),
            dw_err_ppm: 0.08,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: None,
        },
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 19,
            res_name: Some("PRO".to_string()),
            atom: "C".to_string(),
            dw_ppm: -0.85 + rng.normal(0.0, 0.40) + rng.normal(0.0, 0.08),
            dw_err_ppm: 0.08,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: None,
        },
        // Preceding VAL17 & GLU18
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 17,
            res_name: Some("VAL".to_string()),
            atom: "N".to_string(),
            dw_ppm: -0.52 + rng.normal(0.0, 0.35) + rng.normal(0.0, 0.10),
            dw_err_ppm: 0.10,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: None,
        },
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 18,
            res_name: Some("GLU".to_string()),
            atom: "CA".to_string(),
            dw_ppm: -1.15 + rng.normal(0.0, 0.38) + rng.normal(0.0, 0.08),
            dw_err_ppm: 0.08,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: None,
        },
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 18,
            res_name: Some("GLU".to_string()),
            atom: "C".to_string(),
            dw_ppm: -1.35 + rng.normal(0.0, 0.42) + rng.normal(0.0, 0.08),
            dw_err_ppm: 0.08,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: None,
        },
        // Following SER20 & ASP21
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 20,
            res_name: Some("SER".to_string()),
            atom: "N".to_string(),
            dw_ppm: -1.85 + rng.normal(0.0, 0.70) + rng.normal(0.0, 0.12),
            dw_err_ppm: 0.12,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: None,
        },
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 20,
            res_name: Some("SER".to_string()),
            atom: "H".to_string(),
            dw_ppm: -0.32 + rng.normal(0.0, 0.14) + rng.normal(0.0, 0.03),
            dw_err_ppm: 0.03,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: None,
        },
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 21,
            res_name: Some("ASP".to_string()),
            atom: "N".to_string(),
            dw_ppm: -0.48 + rng.normal(0.0, 0.35) + rng.normal(0.0, 0.10),
            dw_err_ppm: 0.10,
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
            dw_ppm: rng.normal(0.0, 0.08),
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
            &alt_bundle,
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
        let bf_non_null = rep
            .two_ln_bf_top1_vs_runnerup_non_null
            .expect("non-null runnerup");
        assert!(
            rep.two_ln_bf_top1_vs_top2 >= 6.0 && bf_non_null >= 6.0,
            "H2 ({mode_label}) expected 2 ln BF >= 6.0, got {:.2} / {:.2}",
            rep.two_ln_bf_top1_vs_top2,
            bf_non_null
        );
        assert!(
            (0.25..=2.5).contains(&rep.top1_reduced_chi2),
            "H2 ({mode_label}) expected realistic reduced_chi2 in [0.25, 2.5], got {:.3}",
            rep.top1_reduced_chi2
        );
        println!(
            "EXCITED_STATE_H2_{mode_label}: 2lnBF_12={:.2} 2lnBF_non_null={:.2} 2lnBF_null={:.2} rchi2={:.3}",
            rep.two_ln_bf_top1_vs_top2,
            bf_non_null,
            rep.two_ln_bf_top1_vs_null,
            rep.top1_reduced_chi2
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
    // Self gamma-gauche CB shift on PHE45 (`g- -> g+`: `-1.65 ppm`) + forward-model perturbation noise
    h3a_signed.push(ExchangeShiftObservation {
        chain_id: "A".to_string(),
        seq_id: 45,
        res_name: Some("PHE".to_string()),
        atom: "CB".to_string(),
        dw_ppm: -1.65 + rng.normal(0.0, 0.45) + rng.normal(0.0, 0.08),
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
            let w_fwd = (d_h.abs() / 0.12).clamp(0.0, 1.0);
            h3a_signed.push(ExchangeShiftObservation {
                chain_id: "A".to_string(),
                seq_id: r.seq_id,
                res_name: Some(r.res_name.clone()),
                atom: "H".to_string(),
                dw_ppm: d_h + rng.normal(0.0, 0.14 * w_fwd) + rng.normal(0.0, 0.02),
                dw_err_ppm: 0.02,
                sign_mode: ExchangeSignMode::Signed,
                major_ppm: None,
            });
        }
        let (ha0, _) =
            evaluate_haigh_mallion_ring_current(&ubq_model.aromatic_rings, r.ha, idx, true, 1.0);
        let (ha1, _) = evaluate_haigh_mallion_ring_current(&rot_rings, r.ha, idx, true, 1.0);
        let d_ha = ha1.clamp(-1.50, 1.20) - ha0.clamp(-1.50, 1.20);
        let w_fwd_ha = (d_ha.abs() / 0.10).clamp(0.0, 1.0);
        h3a_signed.push(ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: r.seq_id,
            res_name: Some(r.res_name.clone()),
            atom: "HA".to_string(),
            dw_ppm: d_ha + rng.normal(0.0, 0.11 * w_fwd_ha) + rng.normal(0.0, 0.02),
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
                let w_fwd_m = (d_hm.abs() / 0.10).clamp(0.0, 1.0);
                h3a_signed.push(ExchangeShiftObservation {
                    chain_id: "A".to_string(),
                    seq_id: r.seq_id,
                    res_name: Some(r.res_name.clone()),
                    atom: m.proton_atom.clone(),
                    dw_ppm: d_hm + rng.normal(0.0, 0.10 * w_fwd_m) + rng.normal(0.0, 0.018),
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
            let w_fwd_ca = (d_ca_ppm.abs() / 0.25).clamp(0.0, 1.0);
            h3a_signed.push(ExchangeShiftObservation {
                chain_id: "A".to_string(),
                seq_id: r.seq_id,
                res_name: Some(r.res_name.clone()),
                atom: "CA".to_string(),
                dw_ppm: d_ca_ppm + rng.normal(0.0, 0.30 * w_fwd_ca) + rng.normal(0.0, 0.07),
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
            &alt_bundle,
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
        let bf_non_null = rep
            .two_ln_bf_top1_vs_runnerup_non_null
            .expect("non-null runnerup");
        assert!(
            rep.two_ln_bf_top1_vs_top2 >= 6.0 && bf_non_null >= 6.0,
            "H3a PHE45 ({mode_label}) expected 2 ln BF >= 6.0, got {:.2} / {:.2}",
            rep.two_ln_bf_top1_vs_top2,
            bf_non_null
        );
        assert!(
            (0.20..=2.5).contains(&rep.top1_reduced_chi2),
            "H3a ({mode_label}) expected realistic reduced_chi2 in [0.20, 2.5], got {:.3}",
            rep.top1_reduced_chi2
        );
        println!(
            "EXCITED_STATE_H3a_{mode_label}: 2lnBF_12={:.2} 2lnBF_non_null={:.2} 2lnBF_null={:.2} rchi2={:.3}",
            rep.two_ln_bf_top1_vs_top2,
            bf_non_null,
            rep.two_ln_bf_top1_vs_null,
            rep.top1_reduced_chi2
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
    // 3b. H3: RotamerOrHisTautomerFlip -- Pelton 1993 HIS68 +-82 ppm ND1/NE2 & +-8.3 ppm CG flip
    // =========================================================================
    let h3b_signed = vec![
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 68,
            res_name: Some("HIS".to_string()),
            atom: "ND1".to_string(),
            dw_ppm: -82.0 + rng.normal(0.0, 2.60) + rng.normal(0.0, 0.60),
            dw_err_ppm: 0.60,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: Some(249.5),
        },
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 68,
            res_name: Some("HIS".to_string()),
            atom: "NE2".to_string(),
            dw_ppm: 82.0 + rng.normal(0.0, 2.60) + rng.normal(0.0, 0.60),
            dw_err_ppm: 0.60,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: Some(168.5),
        },
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 68,
            res_name: Some("HIS".to_string()),
            atom: "CD2".to_string(),
            dw_ppm: 7.20 + rng.normal(0.0, 0.90) + rng.normal(0.0, 0.20),
            dw_err_ppm: 0.20,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: Some(118.2),
        },
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 68,
            res_name: Some("HIS".to_string()),
            atom: "CG".to_string(),
            dw_ppm: -8.30 + rng.normal(0.0, 0.90) + rng.normal(0.0, 0.20),
            dw_err_ppm: 0.20,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: Some(135.5),
        },
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 68,
            res_name: Some("HIS".to_string()),
            atom: "CB".to_string(),
            dw_ppm: -2.20 + rng.normal(0.0, 0.45) + rng.normal(0.0, 0.15),
            dw_err_ppm: 0.15,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: None,
        },
        // Proximal amides (< 6.5 A of HIS68 ring: LYS6, ILE44, LEU67, LEU69) sensing H-bond/electrostatic switch
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 6,
            res_name: Some("LYS".to_string()),
            atom: "H".to_string(),
            dw_ppm: 0.11 + rng.normal(0.0, 0.08),
            dw_err_ppm: 0.02,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: None,
        },
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 44,
            res_name: Some("ILE".to_string()),
            atom: "H".to_string(),
            dw_ppm: 0.09 + rng.normal(0.0, 0.08),
            dw_err_ppm: 0.02,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: None,
        },
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 67,
            res_name: Some("LEU".to_string()),
            atom: "N".to_string(),
            dw_ppm: 0.38 + rng.normal(0.0, 0.25),
            dw_err_ppm: 0.08,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: None,
        },
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 69,
            res_name: Some("LEU".to_string()),
            atom: "N".to_string(),
            dw_ppm: 0.36 + rng.normal(0.0, 0.25),
            dw_err_ppm: 0.08,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: None,
        },
    ];
    let h3b_unsigned = to_unsigned_table(&h3b_signed);

    for (is_unsigned, mode_label, table) in [
        (false, "signed_CEST", &h3b_signed),
        (true, "unsigned_CPMG", &h3b_unsigned),
    ] {
        let rep = classify_excited_state_with_firewall(
            Some(&ubq_shifts),
            &ubq_model,
            table,
            &alt_bundle,
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
        let bf_non_null = rep
            .two_ln_bf_top1_vs_runnerup_non_null
            .expect("non-null runnerup");
        assert!(
            rep.two_ln_bf_top1_vs_top2 > 10.0 && bf_non_null > 10.0,
            "H3b HIS68 ({mode_label}) expected 2 ln BF > 10.0, got {:.2} / {:.2}",
            rep.two_ln_bf_top1_vs_top2,
            bf_non_null
        );
        assert!(
            (0.15..=2.5).contains(&rep.top1_reduced_chi2),
            "H3b ({mode_label}) expected realistic reduced_chi2 in [0.15, 2.5], got {:.3}",
            rep.top1_reduced_chi2
        );
        println!(
            "EXCITED_STATE_H3b_{mode_label}: 2lnBF_12={:.2} 2lnBF_non_null={:.2} 2lnBF_null={:.2} rchi2={:.3}",
            rep.two_ln_bf_top1_vs_top2,
            bf_non_null,
            rep.two_ln_bf_top1_vs_null,
            rep.top1_reduced_chi2
        );
        match &rep.ranked_hypotheses[0].parameters {
            HypothesisBestParameters::HisTautomerOrProtonation {
                seq_id,
                transition,
                delta_cg_ppm,
                ..
            } => {
                assert_eq!(*seq_id, 68);
                assert!(
                    (delta_cg_ppm.abs() - 8.3).abs() < 1e-6,
                    "expected |delta_cg_ppm| == 8.3 ppm matching topology::HIS_*_PRIORS, got {delta_cg_ppm}"
                );
                if is_unsigned {
                    assert_eq!(
                        *transition,
                        HisExchangeTransition::AmbiguousDirectionOnUnsignedCpmg,
                        "unsigned CPMG without ground ND1/NE2 shifts must report AmbiguousDirectionOnUnsignedCpmg"
                    );
                } else {
                    assert_eq!(*transition, HisExchangeTransition::Ne2HToNd1H);
                }
            }
            other => panic!("expected HisTautomerOrProtonation on HIS68, got {other:?}"),
        }
    }

    // =========================================================================
    // 4. H4: Alternate3DConformer (1 true switched 3D conformer + 3 decoys)
    // =========================================================================
    let mut h4_signed = Vec::new();
    for (idx, r) in ground_recon.residues.iter().enumerate() {
        if !(5..=17).contains(&r.seq_id) && !(26..=30).contains(&r.seq_id) {
            continue;
        }
        for &(atom, exp_sd, fwd_sd) in &[
            ("CA", 0.08, 0.44),
            ("CB", 0.09, 0.52),
            ("C", 0.08, 0.48),
            ("N", 0.15, 1.10),
        ] {
            if let (Some(ga), Some(gb)) = (
                pred_g_recon.residues[idx].backbone.get(atom),
                pred_b_true.residues[idx].backbone.get(atom),
            ) {
                let true_dw = gb.struct_ppm - ga.struct_ppm;
                let fwd_noise = if true_dw.abs() > 0.10 {
                    rng.normal(0.0, fwd_sd)
                } else {
                    0.0
                };
                h4_signed.push(ExchangeShiftObservation {
                    chain_id: "A".to_string(),
                    seq_id: r.seq_id,
                    res_name: Some(r.res_name.clone()),
                    atom: atom.to_string(),
                    dw_ppm: true_dw + fwd_noise + rng.normal(0.0, exp_sd),
                    dw_err_ppm: exp_sd,
                    sign_mode: ExchangeSignMode::Signed,
                    major_ppm: Some(ga.struct_ppm),
                });
            }
        }
    }
    let h4_unsigned = to_unsigned_table(&h4_signed);

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
        let bf_non_null = rep
            .two_ln_bf_top1_vs_runnerup_non_null
            .expect("non-null runnerup");
        println!(
            "EXCITED_STATE_H4_{mode_label}: 2lnBF_12={:.2} 2lnBF_non_null={:.2} 2lnBF_null={:.2} rchi2={:.3}",
            rep.two_ln_bf_top1_vs_top2,
            bf_non_null,
            rep.two_ln_bf_top1_vs_null,
            rep.top1_reduced_chi2
        );
        assert!(
            rep.two_ln_bf_top1_vs_top2 > 10.0 && bf_non_null > 10.0,
            "H4 ({mode_label}) expected 2 ln BF > 10.0, got {:.2} / {:.2}",
            rep.two_ln_bf_top1_vs_top2,
            bf_non_null
        );
        assert!(
            (0.30..=2.5).contains(&rep.top1_reduced_chi2),
            "H4 ({mode_label}) expected realistic reduced_chi2 in [0.30, 2.5], got {:.3}",
            rep.top1_reduced_chi2
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
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 21,
            res_name: Some("ASP".to_string()),
            atom: "N".to_string(),
            dw_ppm: 0.48,
            dw_err_ppm: 0.10,
            sign_mode: ExchangeSignMode::UnsignedMagnitude,
            major_ppm: None,
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
    assert_eq!(rep.n_unsigned_observations, 4);
    assert_eq!(rep.n_signed_observations, 2);
    assert_eq!(rep.n_distinct_residues, 4);
    assert_eq!(rep.n_distinct_exchanging_residues, 4);
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

#[test]
fn test_s5_p0_1_duplicate_observation_rejected_and_single_residue_abstains() {
    let (ubq_model, ubq_shifts) = load_ubiquitin_ground();
    let fw = ubiquitin_firewall();
    let cfg = ExcitedStateConfig::default();

    // 1. Exact Claude review probe (`21.W3:P0-1`): Ubiquitin S20 N = -1.85 ppm repeated 4x, 8x, 30x
    //    must be rejected with `Err` in BOTH `parse_exchange_observations_tsv` AND `classify_excited_state_with_firewall`!
    for &(copies, mode) in &[
        (4usize, ExchangeSignMode::Signed),
        (8usize, ExchangeSignMode::UnsignedMagnitude),
        (30usize, ExchangeSignMode::Signed),
    ] {
        let single_row = ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 20,
            res_name: Some("SER".to_string()),
            atom: "N".to_string(),
            dw_ppm: if mode == ExchangeSignMode::Signed {
                -1.85
            } else {
                1.85
            },
            dw_err_ppm: 0.10,
            sign_mode: mode,
            major_ppm: Some(116.2),
        };
        let repeated: Vec<ExchangeShiftObservation> = vec![single_row; copies];
        let tsv = write_exchange_observations_tsv(&repeated);
        let parse_err = parse_exchange_observations_tsv(&tsv)
            .expect_err("duplicate (chain_id, seq_id, atom) rows in TSV must fail closed");
        assert!(
            parse_err.contains("duplicate"),
            "expected 'duplicate' in parse error, got: {parse_err}"
        );

        let classify_err = classify_excited_state_with_firewall(
            Some(&ubq_shifts),
            &ubq_model,
            &repeated,
            &[],
            &fw,
            &cfg,
        )
        .expect_err(
            "duplicate (chain_id, seq_id, atom) slice in classify_excited_state must fail closed",
        );
        assert!(
            classify_err.contains("duplicate"),
            "expected 'duplicate' in classify error, got: {classify_err}"
        );
    }

    // 2. Atom-alias duplicates (`HN` vs `H`, empty chain vs `"A"`) are also normalized and rejected!
    let alias_tsv = "chain_id\tseq_id\tres_name\tatom\tdw_ppm\tdw_err_ppm\tsign_mode\n\
                     A\t20\tSER\tH\t-0.30\t0.03\tsigned\n\
                     \t20\tSER\tHN\t-0.31\t0.03\tsigned\n";
    assert!(
        parse_exchange_observations_tsv(alias_tsv).is_err(),
        "normalized H vs HN duplicate on same (chain, seq_id) must be rejected"
    );

    // 3. Single-residue multi-atom (`N_obs = 5 >= 4`, `N_res = 1 < 4`) MUST abstain!
    let single_res_five_atoms = vec![
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 19,
            res_name: Some("PRO".to_string()),
            atom: "CB".to_string(),
            dw_ppm: 2.40,
            dw_err_ppm: 0.08,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: None,
        },
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 19,
            res_name: Some("PRO".to_string()),
            atom: "CG".to_string(),
            dw_ppm: -2.60,
            dw_err_ppm: 0.08,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: None,
        },
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 19,
            res_name: Some("PRO".to_string()),
            atom: "CA".to_string(),
            dw_ppm: -0.55,
            dw_err_ppm: 0.08,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: None,
        },
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 19,
            res_name: Some("PRO".to_string()),
            atom: "C".to_string(),
            dw_ppm: -0.85,
            dw_err_ppm: 0.08,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: None,
        },
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 19,
            res_name: Some("PRO".to_string()),
            atom: "HA".to_string(),
            dw_ppm: 0.28,
            dw_err_ppm: 0.03,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: None,
        },
    ];
    let rep_single = classify_excited_state_with_firewall(
        Some(&ubq_shifts),
        &ubq_model,
        &single_res_five_atoms,
        &[],
        &fw,
        &cfg,
    )
    .expect("single residue 5 atoms");
    assert_eq!(rep_single.n_observations, 5);
    assert_eq!(rep_single.n_distinct_residues, 1);
    assert_eq!(
        rep_single.verdict,
        ExcitedStateVerdict::AmbiguousOrUnmodeledState
    );
    assert_eq!(
        rep_single.abstention_kind,
        Some(ExcitedStateAbstentionKind::InsufficientDistinctResidues)
    );

    // 4. Spectator-padding probe: 1 exchanging residue (`S20 N = -1.85`) + 5 zero-shift spectators
    //    (`N_res = 6 >= 4`, `N_exch_res = 1 < 4`) MUST NOT commit!
    let mut padded = vec![ExchangeShiftObservation {
        chain_id: "A".to_string(),
        seq_id: 20,
        res_name: Some("SER".to_string()),
        atom: "N".to_string(),
        dw_ppm: -1.85,
        dw_err_ppm: 0.08,
        sign_mode: ExchangeSignMode::Signed,
        major_ppm: Some(116.2),
    }];
    for &(sid, rn) in &[
        (5, "VAL"),
        (13, "ILE"),
        (26, "VAL"),
        (43, "LEU"),
        (56, "LEU"),
    ] {
        padded.push(ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: sid,
            res_name: Some(rn.to_string()),
            atom: "N".to_string(),
            dw_ppm: 0.0,
            dw_err_ppm: 0.12,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: None,
        });
    }
    let rep_padded = classify_excited_state_with_firewall(
        Some(&ubq_shifts),
        &ubq_model,
        &padded,
        &[],
        &fw,
        &cfg,
    )
    .expect("spectator padded");
    assert_eq!(rep_padded.n_distinct_residues, 6);
    assert_eq!(rep_padded.n_distinct_exchanging_residues, 1);
    assert_eq!(
        rep_padded.verdict,
        ExcitedStateVerdict::AmbiguousOrUnmodeledState
    );
}

#[test]
fn test_s5_p0_2_nterminal_u_best_zero_no_debug_underflow_panic() {
    let (ubq_model, ubq_shifts) = load_ubiquitin_ground();
    let fw = ubiquitin_firewall();
    let cfg = ExcitedStateConfig::default();
    let pred_a =
        predict_structure_shifts_with_firewall(&ubq_model, &fw, &StructShiftConfig::default())
            .expect("pred_a");

    // N-terminal unfolding starting at residue index 0 (`MET1..VAL5`, `u_best == 0`!)
    // Previously panicked in debug builds at `(u_best > 0 && ...).then_some(u_best - 1)`.
    let mut nterm_obs = Vec::new();
    for (r_idx, r) in ubq_model.residues.iter().enumerate() {
        if !(1..=6).contains(&r.seq_id) && !(25..=28).contains(&r.seq_id) {
            continue;
        }
        let w = if (1..=5).contains(&r.seq_id) {
            1.0
        } else if r.seq_id == 6 {
            0.5
        } else {
            0.0
        };
        let ref_idx = ubq_shifts
            .seq_ids
            .iter()
            .position(|&s| s == r.seq_id)
            .expect("seq_id");
        for &(atom, exp_sd) in &[("CA", 0.08), ("CB", 0.09), ("C", 0.08), ("N", 0.12)] {
            if let (Some(&g_ppm), Some(bb)) = (
                ubq_shifts.shifts.get(&(ref_idx, atom.to_string())),
                pred_a.residues[r_idx].backbone.get(atom),
            ) {
                nterm_obs.push(ExchangeShiftObservation {
                    chain_id: "A".to_string(),
                    seq_id: r.seq_id,
                    res_name: Some(r.res_name.clone()),
                    atom: atom.to_string(),
                    dw_ppm: w * (bb.rc_ppm - g_ppm),
                    dw_err_ppm: exp_sd,
                    sign_mode: ExchangeSignMode::Signed,
                    major_ppm: Some(g_ppm),
                });
            }
        }
    }

    let rep = classify_excited_state_with_firewall(
        Some(&ubq_shifts),
        &ubq_model,
        &nterm_obs,
        &[],
        &fw,
        &cfg,
    )
    .expect("u_best == 0 must never underflow in debug builds");
    assert_eq!(
        rep.verdict,
        ExcitedStateVerdict::Committed(ExcitedStateHypothesisKind::LocalUnfolding)
    );
    match &rep.ranked_hypotheses[0].parameters {
        HypothesisBestParameters::LocalUnfolding {
            start_seq_id,
            end_seq_id,
            ..
        } => {
            assert_eq!(*start_seq_id, 1);
            assert_eq!(*end_seq_id, 5);
            assert_eq!(
                rep.ranked_hypotheses[0].affected_seq_ids,
                vec![1, 2, 3, 4, 5, 6]
            );
        }
        other => panic!("expected LocalUnfolding at N-terminus, got {other:?}"),
    }
}

#[test]
fn test_s5_p1_3_marginalized_discrete_scan_bic_and_distinct_residue_n() {
    let (ubq_model, ubq_shifts) = load_ubiquitin_ground();
    let fw = ubiquitin_firewall();
    let cfg = ExcitedStateConfig::default();

    // 4 distinct residues (`18, 19, 20, 21`), 8 total observations (2 nuclei per residue):
    // BIC MUST use `n = 4` (`ln(4)`), NOT `n = 8` (`ln(8)`), and `marginal_log_likelihood`
    // MUST equal `ln(1/M_H * sum_m L_m) < max_m ln L_m` (discrete-scan multiplicity penalty).
    let obs = vec![
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 19,
            res_name: Some("PRO".to_string()),
            atom: "CB".to_string(),
            dw_ppm: 2.40,
            dw_err_ppm: 0.08,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: None,
        },
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 19,
            res_name: Some("PRO".to_string()),
            atom: "CG".to_string(),
            dw_ppm: -2.60,
            dw_err_ppm: 0.08,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: None,
        },
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 18,
            res_name: Some("GLU".to_string()),
            atom: "CA".to_string(),
            dw_ppm: -1.15,
            dw_err_ppm: 0.08,
            sign_mode: ExchangeSignMode::Signed,
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
            major_ppm: None,
        },
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 20,
            res_name: Some("SER".to_string()),
            atom: "H".to_string(),
            dw_ppm: -0.32,
            dw_err_ppm: 0.03,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: None,
        },
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 21,
            res_name: Some("ASP".to_string()),
            atom: "N".to_string(),
            dw_ppm: -0.45,
            dw_err_ppm: 0.10,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: None,
        },
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 21,
            res_name: Some("ASP".to_string()),
            atom: "CA".to_string(),
            dw_ppm: -0.22,
            dw_err_ppm: 0.08,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: None,
        },
    ];

    let rep =
        classify_excited_state_with_firewall(Some(&ubq_shifts), &ubq_model, &obs, &[], &fw, &cfg)
            .expect("classify P1-3");
    assert_eq!(rep.n_observations, 8);
    assert_eq!(rep.n_distinct_residues, 4);

    for h in &rep.ranked_hypotheses {
        assert_eq!(h.n_distinct_residues, 4);
        let expected_bic =
            -2.0 * h.marginal_log_likelihood + (h.num_params_k as f64) * (4.0_f64).ln();
        assert!(
            (h.bic - expected_bic).abs() < 1e-10,
            "BIC for {:?} must equal -2*marginal_ll + k*ln(4): got {} vs {}",
            h.kind,
            h.bic,
            expected_bic
        );
        if h.n_candidates_evaluated > 1 {
            assert!(
                h.marginal_log_likelihood < h.max_log_likelihood,
                "discrete candidate scan (M_H={}) for {:?} must carry marginalization penalty (< max ll)",
                h.n_candidates_evaluated,
                h.kind
            );
        }
    }
    // Ubiquitin has 3 Prolines (PRO19, PRO37, PRO38); PRO19 dominates -> marginal_ll ≈ max_ll - ln(3)
    let h2 = rep
        .ranked_hypotheses
        .iter()
        .find(|h| h.kind == ExcitedStateHypothesisKind::XxxProCisTrans)
        .expect("H2");
    assert_eq!(h2.n_candidates_evaluated, 3);
    let penalty_nats = h2.max_log_likelihood - h2.marginal_log_likelihood;
    assert!(
        (penalty_nats - (3.0_f64).ln()).abs() < 0.05,
        "when 1 of 3 Prolines dominates, marginalization penalty must be ~ln(3)={:.4}, got {:.4}",
        (3.0_f64).ln(),
        penalty_nats
    );
}

#[test]
fn test_s5_p1_4_h0_experimental_noise_only_rejects_3_sites_at_2p5_ppm() {
    let (ubq_model, ubq_shifts) = load_ubiquitin_ground();
    let fw = ubiquitin_firewall();
    let cfg = ExcitedStateConfig::default();

    // Exact Claude review probe (`21.W3:P1-4`): 3 scattered sites at |dw| = 2.50 ppm (`20x`
    // measurement error `dw_err = 0.125 ppm`) on a background of 12 pure-noise residues across
    // 20 independent seeds (both Signed and Unsigned) MUST strongly reject H0 and report
    // `UnmodeledSignificantExchange` (NEVER claiming `"within experimental noise"` or `"no significant exchange"`!).
    let background_sids = [4, 7, 11, 15, 24, 27, 31, 41, 48, 52, 61, 71];
    let outlier_sids = [(13, "ILE", 2.50), (33, "LYS", -2.50), (58, "ASP", 2.50)];

    for seed in 0..20u64 {
        let mut rng = Rng::new(77_000 + seed);
        let mut signed_obs = Vec::new();
        for &(sid, rn, dw) in &outlier_sids {
            signed_obs.push(ExchangeShiftObservation {
                chain_id: "A".to_string(),
                seq_id: sid,
                res_name: Some(rn.to_string()),
                atom: "N".to_string(),
                dw_ppm: dw,
                dw_err_ppm: 0.125,
                sign_mode: ExchangeSignMode::Signed,
                major_ppm: None,
            });
        }
        for &sid in &background_sids {
            let r = ubq_model
                .residues
                .iter()
                .find(|x| x.seq_id == sid)
                .expect("bg res");
            signed_obs.push(ExchangeShiftObservation {
                chain_id: "A".to_string(),
                seq_id: sid,
                res_name: Some(r.res_name.clone()),
                atom: "N".to_string(),
                dw_ppm: rng.normal(0.0, 0.125),
                dw_err_ppm: 0.125,
                sign_mode: ExchangeSignMode::Signed,
                major_ppm: None,
            });
        }
        let unsigned_obs = to_unsigned_table(&signed_obs);

        for (label, tbl) in [("signed", &signed_obs), ("unsigned", &unsigned_obs)] {
            let rep = classify_excited_state_with_firewall(
                Some(&ubq_shifts),
                &ubq_model,
                tbl,
                &[],
                &fw,
                &cfg,
            )
            .expect("classify P1-4");
            let h0 = rep
                .ranked_hypotheses
                .iter()
                .find(|h| h.kind == ExcitedStateHypothesisKind::NullOrUnstructuredPerturbation)
                .expect("H0 present");
            assert!(
                h0.reduced_chi2 > 20.0,
                "seed {seed} ({label}): H0 parameterized with sigma_exp must reject 3x 20-sigma outliers (got rchi2={:.3})",
                h0.reduced_chi2
            );
            assert!(
                rep.unmodeled_exchange_two_ln_bf_vs_null > 10.0,
                "seed {seed} ({label}): expected unmodeled_exchange_two_ln_bf_vs_null > 10.0, got {:.2}",
                rep.unmodeled_exchange_two_ln_bf_vs_null
            );
            assert_eq!(rep.verdict, ExcitedStateVerdict::AmbiguousOrUnmodeledState);
            assert_eq!(
                rep.abstention_kind,
                Some(ExcitedStateAbstentionKind::UnmodeledSignificantExchange),
                "seed {seed} ({label}): expected UnmodeledSignificantExchange, got {:?} ({})",
                rep.abstention_kind,
                rep.verdict_reason
            );
            assert!(
                rep.verdict_reason
                    .starts_with("UnmodeledSignificantExchange:"),
                "unexpected verdict_reason: {}",
                rep.verdict_reason
            );
            assert!(
                !rep.verdict_reason
                    .to_ascii_lowercase()
                    .contains("within experimental noise"),
                "must never falsely claim consistency with noise: {}",
                rep.verdict_reason
            );
        }
    }

    // Verify `cpmg_prefiltered_significant_only` is recorded explicitly when parsing CPMG fit TSV
    let cpmg_tsv = "seq_id\tres_name\tdw_ppm\tdw_ppm_err\tsignificant_exchange\n\
                    18\tGLU\t1.15\t0.08\ttrue\n\
                    19\tPRO\t0.55\t0.08\ttrue\n\
                    20\tSER\t1.85\t0.10\ttrue\n\
                    21\tASP\t0.48\t0.09\ttrue\n\
                    25\tASN\t0.04\t0.10\tfalse\n";
    let parsed_meta = parse_exchange_observations_tsv_with_meta(cpmg_tsv).expect("parse meta");
    assert!(parsed_meta.cpmg_prefiltered_significant_only);
    assert_eq!(parsed_meta.observations.len(), 4);

    let cfg_prefilt = ExcitedStateConfig {
        cpmg_prefiltered_significant_only: parsed_meta.cpmg_prefiltered_significant_only,
        ..ExcitedStateConfig::default()
    };
    let rep_pf = classify_excited_state_with_firewall(
        Some(&ubq_shifts),
        &ubq_model,
        &parsed_meta.observations,
        &[],
        &fw,
        &cfg_prefilt,
    )
    .expect("classify prefiltered");
    assert!(rep_pf.cpmg_prefiltered_significant_only);
    assert!(rep_pf
        .warnings
        .iter()
        .any(|w| w.contains("significant_exchange=true")));
    let tsv_pf = format_excited_state_tsv(&rep_pf);
    assert!(tsv_pf.contains("cpmg_prefiltered_significant_only: true"));
}

#[test]
fn test_s5_his_p2_cg_shift_backbone_gate_and_unsigned_direction_ambiguity() {
    let (ubq_model, ubq_shifts) = load_ubiquitin_ground();
    let fw = ubiquitin_firewall();
    let cfg = ExcitedStateConfig::default();

    // 1. Backbone-only data near HIS68 (ZERO sidechain ND1/NE2/CD2/CG/CE1 observations) MUST NEVER
    //    select `HisTautomerOrProtonation` (`His` P2 item 2).
    let backbone_near_his68 = vec![
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 68,
            res_name: Some("HIS".to_string()),
            atom: "CB".to_string(),
            dw_ppm: -2.20,
            dw_err_ppm: 0.12,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: None,
        },
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 68,
            res_name: Some("HIS".to_string()),
            atom: "CA".to_string(),
            dw_ppm: -0.55,
            dw_err_ppm: 0.08,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: None,
        },
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 68,
            res_name: Some("HIS".to_string()),
            atom: "N".to_string(),
            dw_ppm: 0.85,
            dw_err_ppm: 0.12,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: None,
        },
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 67,
            res_name: Some("LEU".to_string()),
            atom: "N".to_string(),
            dw_ppm: 0.35,
            dw_err_ppm: 0.10,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: None,
        },
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 69,
            res_name: Some("LEU".to_string()),
            atom: "N".to_string(),
            dw_ppm: 0.35,
            dw_err_ppm: 0.10,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: None,
        },
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 44,
            res_name: Some("ILE".to_string()),
            atom: "H".to_string(),
            dw_ppm: 0.12,
            dw_err_ppm: 0.02,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: None,
        },
    ];
    let rep_bb = classify_excited_state_with_firewall(
        Some(&ubq_shifts),
        &ubq_model,
        &backbone_near_his68,
        &[],
        &fw,
        &cfg,
    )
    .expect("classify backbone near HIS68");
    for h in &rep_bb.ranked_hypotheses {
        assert!(
            !matches!(
                h.parameters,
                HypothesisBestParameters::HisTautomerOrProtonation { .. }
            ),
            "HisTautomerOrProtonation must be ineligible on backbone-only data, got: {:?}",
            h.parameters
        );
    }

    // 2. Unsigned CPMG data WITH disambiguating ground-state `ND1 = 249.5, NE2 = 168.5` (`major_ppm`)
    //    resolves `Ne2HToNd1H` (`delta_cg_ppm == -8.3`), whereas WITHOUT ground `ND1`/`NE2` it reports
    //    `AmbiguousDirectionOnUnsignedCpmg` (`|delta_cg_ppm| == 8.3`).
    let unsigned_with_ground = vec![
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 68,
            res_name: Some("HIS".to_string()),
            atom: "ND1".to_string(),
            dw_ppm: 82.0,
            dw_err_ppm: 0.60,
            sign_mode: ExchangeSignMode::UnsignedMagnitude,
            major_ppm: Some(249.5),
        },
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 68,
            res_name: Some("HIS".to_string()),
            atom: "NE2".to_string(),
            dw_ppm: 82.0,
            dw_err_ppm: 0.60,
            sign_mode: ExchangeSignMode::UnsignedMagnitude,
            major_ppm: Some(168.5),
        },
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 68,
            res_name: Some("HIS".to_string()),
            atom: "CG".to_string(),
            dw_ppm: 8.30,
            dw_err_ppm: 0.20,
            sign_mode: ExchangeSignMode::UnsignedMagnitude,
            major_ppm: None,
        },
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 6,
            res_name: Some("LYS".to_string()),
            atom: "H".to_string(),
            dw_ppm: 0.10,
            dw_err_ppm: 0.02,
            sign_mode: ExchangeSignMode::UnsignedMagnitude,
            major_ppm: None,
        },
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 44,
            res_name: Some("ILE".to_string()),
            atom: "H".to_string(),
            dw_ppm: 0.09,
            dw_err_ppm: 0.02,
            sign_mode: ExchangeSignMode::UnsignedMagnitude,
            major_ppm: None,
        },
        ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: 67,
            res_name: Some("LEU".to_string()),
            atom: "N".to_string(),
            dw_ppm: 0.32,
            dw_err_ppm: 0.08,
            sign_mode: ExchangeSignMode::UnsignedMagnitude,
            major_ppm: None,
        },
    ];
    let rep_ug = classify_excited_state_with_firewall(
        Some(&ubq_shifts),
        &ubq_model,
        &unsigned_with_ground,
        &[],
        &fw,
        &cfg,
    )
    .expect("unsigned with ground ND1/NE2");
    match &rep_ug.ranked_hypotheses[0].parameters {
        HypothesisBestParameters::HisTautomerOrProtonation {
            transition,
            delta_cg_ppm,
            ..
        } => {
            assert_eq!(*transition, HisExchangeTransition::Ne2HToNd1H);
            assert!((*delta_cg_ppm - (-8.3)).abs() < 1e-6);
        }
        other => panic!("expected HisTautomerOrProtonation, got {other:?}"),
    }

    let unsigned_no_ground: Vec<ExchangeShiftObservation> = unsigned_with_ground
        .into_iter()
        .map(|mut o| {
            o.major_ppm = None;
            o
        })
        .collect();
    let rep_un = classify_excited_state_with_firewall(
        Some(&ubq_shifts),
        &ubq_model,
        &unsigned_no_ground,
        &[],
        &fw,
        &cfg,
    )
    .expect("unsigned without ground ND1/NE2");
    match &rep_un.ranked_hypotheses[0].parameters {
        HypothesisBestParameters::HisTautomerOrProtonation {
            transition,
            delta_cg_ppm,
            ..
        } => {
            assert_eq!(
                *transition,
                HisExchangeTransition::AmbiguousDirectionOnUnsignedCpmg
            );
            assert!((*delta_cg_ppm - 8.3).abs() < 1e-6);
        }
        other => panic!("expected HisTautomerOrProtonation, got {other:?}"),
    }
}

#[test]
fn test_fix2_b19_misestimated_error_warning_2x_3x_pure_noise() {
    let (ubq_model, ubq_shifts) = load_ubiquitin_ground();
    let fw = ubiquitin_firewall();
    let cfg = ExcitedStateConfig::default();

    let probe_sids = [4, 12, 18, 19, 20, 25, 28, 31, 45, 68];

    // 1. Well-calibrated 1x pure noise: when ConsistentWithNoise & within chi2 bound, NO false alarm!
    let mut rng_cal = Rng::new(77_001);
    let mut cal_obs = Vec::new();
    for &sid in &probe_sids {
        let r = ubq_model
            .residues
            .iter()
            .find(|x| x.seq_id == sid)
            .expect("res");
        cal_obs.push(ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: sid,
            res_name: Some(r.res_name.clone()),
            atom: "N".to_string(),
            dw_ppm: rng_cal.normal(0.0, 0.08),
            dw_err_ppm: 0.12,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: None,
        });
        cal_obs.push(ExchangeShiftObservation {
            chain_id: "A".to_string(),
            seq_id: sid,
            res_name: Some(r.res_name.clone()),
            atom: "CA".to_string(),
            dw_ppm: rng_cal.normal(0.0, 0.05),
            dw_err_ppm: 0.08,
            sign_mode: ExchangeSignMode::Signed,
            major_ppm: None,
        });
    }
    let rep_cal = classify_excited_state_with_firewall(
        Some(&ubq_shifts),
        &ubq_model,
        &cal_obs,
        &[],
        &fw,
        &cfg,
    )
    .expect("calibrated 1x noise");
    assert_eq!(
        rep_cal.abstention_kind,
        Some(ExcitedStateAbstentionKind::ConsistentWithNoise)
    );
    assert!(
        rep_cal.error_scale_warning.is_none(),
        "well-calibrated noise within chi2 bound should not emit error_scale_warning: {:?}",
        rep_cal.error_scale_warning
    );

    // 2. 2x and 3x misestimated-error pure-noise probes (10 seeds each):
    //    Whenever H0 is rejected, UnmodeledSignificantExchange is emitted, or chi2_red exceeds
    //    the consistency bound, `error_scale_warning` and `chi2_consistency_diagnostic` MUST be
    //    emitted in the report, in `report.warnings`, and in the formatted TSV!
    for &scale in &[2.0_f64, 3.0_f64] {
        let mut n_h0_rejected = 0usize;
        let mut n_warned = 0usize;
        for seed in 0..10u64 {
            let mut rng = Rng::new(21_900 + (scale as u64) * 100 + seed);
            let mut obs = Vec::new();
            for &sid in &probe_sids {
                let r = ubq_model
                    .residues
                    .iter()
                    .find(|x| x.seq_id == sid)
                    .expect("res");
                obs.push(ExchangeShiftObservation {
                    chain_id: "A".to_string(),
                    seq_id: sid,
                    res_name: Some(r.res_name.clone()),
                    atom: "N".to_string(),
                    dw_ppm: rng.normal(0.0, 0.12 * scale),
                    dw_err_ppm: 0.12,
                    sign_mode: ExchangeSignMode::Signed,
                    major_ppm: None,
                });
                obs.push(ExchangeShiftObservation {
                    chain_id: "A".to_string(),
                    seq_id: sid,
                    res_name: Some(r.res_name.clone()),
                    atom: "CA".to_string(),
                    dw_ppm: rng.normal(0.0, 0.08 * scale),
                    dw_err_ppm: 0.08,
                    sign_mode: ExchangeSignMode::Signed,
                    major_ppm: None,
                });
            }
            let rep = classify_excited_state_with_firewall(
                Some(&ubq_shifts),
                &ubq_model,
                &obs,
                &[],
                &fw,
                &cfg,
            )
            .expect("misestimated noise");
            let diag = rep
                .chi2_consistency_diagnostic
                .as_ref()
                .expect("chi2_consistency_diagnostic must be populated");
            assert!(diag.h0_reduced_chi2.is_finite() && diag.h0_reduced_chi2 > 0.0);
            let h0_rejected = rep.abstention_kind
                == Some(ExcitedStateAbstentionKind::UnmodeledSignificantExchange)
                || diag.h0_exceeds_consistency_bound
                || diag.top1_exceeds_consistency_bound;
            if h0_rejected {
                n_h0_rejected += 1;
                let warn = rep.error_scale_warning.as_deref().unwrap_or_else(|| {
                    panic!(
                        "scale={scale}x seed={seed}: H0 rejected (rchi2={:.3}) without error_scale_warning!",
                        diag.h0_reduced_chi2
                    )
                });
                assert!(
                    warn.contains("2x-3x") && warn.contains("UnmodeledSignificantExchange"),
                    "warning must explain 2x-3x misestimation risk: {warn}"
                );
                assert!(
                    rep.warnings.iter().any(|w| w.contains("Error-scale / chi^2 consistency diagnostic")),
                    "report.warnings must include error_scale_warning"
                );
                let tsv = format_excited_state_tsv(&rep);
                assert!(tsv.contains("# error_scale_warning:"), "TSV missing error_scale_warning");
                assert!(
                    tsv.contains("# chi2_consistency_diagnostic:"),
                    "TSV missing chi2_consistency_diagnostic"
                );
                n_warned += 1;
            }
        }
        assert!(
            n_h0_rejected >= 7,
            "expected >=7/10 H0 rejections at {scale}x misestimated error, got {n_h0_rejected}/10"
        );
        assert_eq!(
            n_warned, n_h0_rejected,
            "every H0 rejection at {scale}x must emit error_scale_warning"
        );
    }
}

#[test]
fn test_fix2_b19_multiseed_10_recovery_and_rchi2_statistics() {
    let (ubq_model, ubq_shifts) = load_ubiquitin_ground();
    let fw = ubiquitin_firewall();
    let cfg = ExcitedStateConfig::default();

    let struct_cfg = StructShiftConfig {
        sample_ph: cfg.sample_ph,
        temperature_kelvin: cfg.temperature_kelvin,
        include_methyl_ilvmat: true,
        ..StructShiftConfig::default()
    };

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

    let pred_a =
        predict_structure_shifts_with_firewall(&ubq_model, &fw, &struct_cfg).expect("pred_a");
    let pred_g_recon =
        predict_structure_shifts_with_firewall(&ground_recon, &fw, &struct_cfg).unwrap();
    let pred_b_true = predict_structure_shifts_with_firewall(&alt_true, &fw, &struct_cfg).unwrap();

    let alt_bundle = vec![
        AltConformerInput {
            label: "decoy_loop48_56.pdb".to_string(),
            model: alt_decoy1,
        },
        AltConformerInput {
            label: "native_excited_hairpin7_15.pdb".to_string(),
            model: alt_true,
        },
    ];

    let mut h1_ok = 0usize;
    let mut h2_ok = 0usize;
    let mut h3_ok = 0usize;
    let mut h4_ok = 0usize;
    let mut h0_clean = 0usize;

    let mut h1_rchi2 = Vec::new();
    let mut h2_rchi2 = Vec::new();
    let mut h3_rchi2 = Vec::new();
    let mut h4_rchi2 = Vec::new();
    let mut h0_rchi2 = Vec::new();

    // Multi-seed (`seeds 0..10`) evaluation across H1, H2, H3, H4, and H0!
    for seed in 0..10u64 {
        let mut rng = Rng::new(20_260_000 + seed);

        // H1: LocalUnfolding (ILE23..GLU34 -> ncIDP coil)
        let mut h1_obs = Vec::new();
        for (r_idx, r) in ubq_model.residues.iter().enumerate() {
            let sid = r.seq_id;
            if !(22..=35).contains(&sid) && !(3..=7).contains(&sid) {
                continue;
            }
            let w: f64 = if (23..=34).contains(&sid) {
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
                .expect("seq_id");
            for &(atom, exp_sd, fwd_sd) in
                &[("CA", 0.08_f64, 0.45_f64), ("C", 0.08, 0.50), ("N", 0.12, 0.90)]
            {
                if let (Some(&obs_a), Some(bb)) = (
                    ubq_shifts.shifts.get(&(ref_idx, atom.to_string())),
                    pred_a.residues[r_idx].backbone.get(atom),
                ) {
                    let mean_dw = w * (bb.rc_ppm - obs_a);
                    let sd = (exp_sd * exp_sd + (w * fwd_sd * 0.4).powi(2)).sqrt();
                    h1_obs.push(ExchangeShiftObservation {
                        chain_id: "A".to_string(),
                        seq_id: sid,
                        res_name: Some(r.res_name.clone()),
                        atom: atom.to_string(),
                        dw_ppm: mean_dw + rng.normal(0.0, sd),
                        dw_err_ppm: exp_sd,
                        sign_mode: ExchangeSignMode::Signed,
                        major_ppm: Some(obs_a),
                    });
                }
            }
        }
        let rep1 = classify_excited_state_with_firewall(
            Some(&ubq_shifts),
            &ubq_model,
            &h1_obs,
            &alt_bundle,
            &fw,
            &cfg,
        )
        .expect("H1 multi-seed");
        if rep1.verdict == ExcitedStateVerdict::Committed(ExcitedStateHypothesisKind::LocalUnfolding)
        {
            h1_ok += 1;
        }
        h1_rchi2.push(rep1.top1_reduced_chi2);

        // H2: Xxx-Pro Cis/Trans at PRO19 (trans -> cis, Schubert 2002)
        let mut h2_obs = Vec::new();
        for &(sid, res_name, atom, mean_dw, exp_sd) in &[
            (19, "PRO", "CB", 2.40, 0.08),
            (19, "PRO", "CG", -2.60, 0.08),
            (19, "PRO", "CA", -0.55, 0.08),
            (19, "PRO", "C", -0.85, 0.08),
            (17, "VAL", "N", -0.52, 0.10),
            (18, "GLU", "CA", -1.15, 0.08),
            (18, "GLU", "C", -1.35, 0.08),
            (18, "GLU", "N", -1.40, 0.12),
            (20, "SER", "N", -1.80, 0.12),
            (20, "SER", "H", -0.25, 0.03),
            (45, "PHE", "N", 0.00, 0.12),
            (55, "THR", "CA", 0.00, 0.08),
        ] {
            h2_obs.push(ExchangeShiftObservation {
                chain_id: "A".to_string(),
                seq_id: sid,
                res_name: Some(res_name.to_string()),
                atom: atom.to_string(),
                dw_ppm: mean_dw + rng.normal(0.0, exp_sd),
                dw_err_ppm: exp_sd,
                sign_mode: ExchangeSignMode::Signed,
                major_ppm: None,
            });
        }
        let rep2 = classify_excited_state_with_firewall(
            Some(&ubq_shifts),
            &ubq_model,
            &h2_obs,
            &alt_bundle,
            &fw,
            &cfg,
        )
        .expect("H2 multi-seed");
        if rep2.verdict == ExcitedStateVerdict::Committed(ExcitedStateHypothesisKind::XxxProCisTrans)
        {
            h2_ok += 1;
        }
        h2_rchi2.push(rep2.top1_reduced_chi2);

        // H3: Histidine tautomer flip at HIS68 (NE2-H -> ND1-H)
        let mut h3_obs = Vec::new();
        for &(sid, res_name, atom, mean_dw, exp_sd, maj) in &[
            (68, "HIS", "ND1", -82.0, 0.60, Some(249.5)),
            (68, "HIS", "NE2", 82.0, 0.60, Some(168.5)),
            (68, "HIS", "CG", -8.30, 0.20, None),
            (6, "LYS", "H", 0.10, 0.02, None),
            (44, "ILE", "H", 0.09, 0.02, None),
            (67, "LEU", "N", 0.32, 0.08, None),
            (25, "ASN", "N", 0.00, 0.10, None),
            (50, "LEU", "CA", 0.00, 0.08, None),
        ] {
            h3_obs.push(ExchangeShiftObservation {
                chain_id: "A".to_string(),
                seq_id: sid,
                res_name: Some(res_name.to_string()),
                atom: atom.to_string(),
                dw_ppm: mean_dw + rng.normal(0.0, exp_sd),
                dw_err_ppm: exp_sd,
                sign_mode: ExchangeSignMode::Signed,
                major_ppm: maj,
            });
        }
        let rep3 = classify_excited_state_with_firewall(
            Some(&ubq_shifts),
            &ubq_model,
            &h3_obs,
            &alt_bundle,
            &fw,
            &cfg,
        )
        .expect("H3 multi-seed");
        if rep3.verdict
            == ExcitedStateVerdict::Committed(ExcitedStateHypothesisKind::RotamerOrHisTautomerFlip)
        {
            h3_ok += 1;
        }
        h3_rchi2.push(rep3.top1_reduced_chi2);

        // H4: Alternate 3D conformer (`alt_true`: hairpin 7..=15 -> helix)
        let mut h4_obs = Vec::new();
        for sid in (6..=16).chain([48, 52, 65]) {
            let ra = pred_g_recon.residues.iter().find(|r| r.seq_id == sid);
            let rb = pred_b_true.residues.iter().find(|r| r.seq_id == sid);
            if let (Some(a), Some(b)) = (ra, rb) {
                for &(atom, exp_sd) in &[("CA", 0.10), ("CB", 0.10), ("C", 0.10), ("N", 0.15)] {
                    if let (Some(ba), Some(bb)) = (a.backbone.get(atom), b.backbone.get(atom)) {
                        let sa = ba.predicted_ppm;
                        let sb = bb.predicted_ppm;
                        h4_obs.push(ExchangeShiftObservation {
                            chain_id: "A".to_string(),
                            seq_id: sid,
                            res_name: Some(a.res_name.clone()),
                            atom: atom.to_string(),
                            dw_ppm: (sb - sa) + rng.normal(0.0, exp_sd),
                            dw_err_ppm: exp_sd,
                            sign_mode: ExchangeSignMode::Signed,
                            major_ppm: Some(sa),
                        });
                    }
                }
            }
        }
        let rep4 = classify_excited_state_with_firewall(
            Some(&ubq_shifts),
            &ground_recon,
            &h4_obs,
            &alt_bundle,
            &fw,
            &cfg,
        )
        .expect("H4 multi-seed");
        if rep4.verdict
            == ExcitedStateVerdict::Committed(ExcitedStateHypothesisKind::Alternate3DConformer)
        {
            h4_ok += 1;
        }
        h4_rchi2.push(rep4.top1_reduced_chi2);

        // H0: Well-calibrated 1x pure noise
        let mut h0_obs = Vec::new();
        for sid in [4, 12, 18, 19, 20, 25, 28, 31, 45, 68] {
            let r = ubq_model
                .residues
                .iter()
                .find(|x| x.seq_id == sid)
                .expect("res");
            h0_obs.push(ExchangeShiftObservation {
                chain_id: "A".to_string(),
                seq_id: sid,
                res_name: Some(r.res_name.clone()),
                atom: "N".to_string(),
                dw_ppm: rng.normal(0.0, 0.12),
                dw_err_ppm: 0.12,
                sign_mode: ExchangeSignMode::Signed,
                major_ppm: None,
            });
            h0_obs.push(ExchangeShiftObservation {
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
        let rep0 = classify_excited_state_with_firewall(
            Some(&ubq_shifts),
            &ubq_model,
            &h0_obs,
            &alt_bundle,
            &fw,
            &cfg,
        )
        .expect("H0 multi-seed");
        if rep0.verdict == ExcitedStateVerdict::AmbiguousOrUnmodeledState {
            h0_clean += 1;
        }
        h0_rchi2.push(rep0.top1_reduced_chi2);
    }

    let range_str = |vals: &[f64]| -> (f64, f64, f64) {
        let mut s = vals.to_vec();
        s.sort_by(|a, b| a.total_cmp(b));
        (s[0], s[s.len() / 2], *s.last().unwrap())
    };
    let (h1_lo, h1_med, h1_hi) = range_str(&h1_rchi2);
    let (h2_lo, h2_med, h2_hi) = range_str(&h2_rchi2);
    let (h3_lo, h3_med, h3_hi) = range_str(&h3_rchi2);
    let (h4_lo, h4_med, h4_hi) = range_str(&h4_rchi2);
    let (h0_lo, h0_med, h0_hi) = range_str(&h0_rchi2);

    eprintln!(
        "B19 10-seed stats (seeds 0..9):\n\
         H1 LocalUnfolding: {h1_ok}/10, rchi2 min={h1_lo:.3} med={h1_med:.3} max={h1_hi:.3}\n\
         H2 XxxProCisTrans: {h2_ok}/10, rchi2 min={h2_lo:.3} med={h2_med:.3} max={h2_hi:.3}\n\
         H3 RotamerOrHis:   {h3_ok}/10, rchi2 min={h3_lo:.3} med={h3_med:.3} max={h3_hi:.3}\n\
         H4 AltConformer:   {h4_ok}/10, rchi2 min={h4_lo:.3} med={h4_med:.3} max={h4_hi:.3}\n\
         H0 PureNoise(1x):  {h0_clean}/10 abstain, rchi2 min={h0_lo:.3} med={h0_med:.3} max={h0_hi:.3}"
    );

    assert!(h1_ok >= 9, "H1 10-seed recovery {h1_ok}/10 < 9/10");
    assert!(h2_ok >= 9, "H2 10-seed recovery {h2_ok}/10 < 9/10");
    assert!(h3_ok >= 9, "H3 10-seed recovery {h3_ok}/10 < 9/10");
    assert!(h4_ok >= 9, "H4 10-seed recovery {h4_ok}/10 < 9/10");
    assert_eq!(h0_clean, 10, "H0 1x noise false-commit detected ({h0_clean}/10)");
}
