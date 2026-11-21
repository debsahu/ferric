//! Clean-room unit tests and realistic Pro/Ser-rich IDP ground-truth benchmark for the
//! unified joint `HN + 2D CON + Proline-Bridge + N–N + CAS` backbone solver (`T23.2`).

use std::collections::BTreeMap;

use nmr_core::c13_detected::{
    n15_obs_mhz, C13DetectedExp, C13PickedDataset, C13Strip3DPeak, CasMode, CasPeak, CasSpectrum,
    Con2DPeak,
};
use nmr_core::idp_solver::{IdpShiftOffsets, MAX_IDP_MATRIX_CELLS};
use nmr_core::joint_idp_solver::{
    apply_gly_cb_bleedthrough_guard, assign_joint_idp, build_calibrated_cas_tables,
    calibrate_trosy_hn_offset, joint_idp_unary_matrix, predict_joint_ncidp_shifts,
    prune_f1_c_prime_nus_sidelobes, try_joint_idp_unary_matrix, DirectionalNnPeak, HncaconhPeak,
    JointIdpConfig, JointIdpDataset, JointIdpNode, JointNodeKind, JointRouteSupport,
    TROSY_HALF_J_HN_HZ,
};
use nmr_core::peaks::Peak;
use nmr_core::priors::one_to_three;
use nmr_core::qc::DatasetQcVerdict;
use nmr_core::rng::Rng;
use nmr_core::spinsystems::HncacbPolarity;

/// Construct a deterministic, clean-room 140-aa Pro/Ser-rich disordered sequence (`15.0% Pro` = 21 Pro,
/// including `-S-P-`, `-G-P-`, `-P-P-` motifs, Ser/Gly-rich stretches, and a non-Pro C-terminus).
fn synthetic_pro_ser_rich_idp_sequence() -> Vec<String> {
    // 140 aa, 21 Pro (15.0%), 22 Ser (15.7%), 14 Gly (10.0%), varied IDP composition
    let seq_1 = concat!(
        "MSGAPQSTPV", // 0..10  (P4, P8)
        "KLDPNSRGAF", // 10..20 (P13)
        "QTPVSLENKP", // 20..30 (P22, P29)
        "PGSVTDRQIA", // 30..40 (P30 -> di-Pro 29-30!)
        "HSMGPLKVSE", // 40..50 (P44)
        "YQNPTASPLR", // 50..60 (P53, P57)
        "DFSVPKQMET", // 60..70 (P64)
        "GSPHLAVNSP", // 70..80 (P72, P79)
        "RKEGTPSQID", // 80..90 (P85)
        "VNSALPGFYT", // 90..100 (P95)
        "QSPKRMGSVP", // 100..110 (P102, P109)
        "PAENHTSLDG", // 110..120 (P110 -> di-Pro 109-110!)
        "KPVSQAFPRN", // 120..130 (P121, P127)
        "TSPLVEPKSA"  // 130..140 (P132, P136, C-term ALA139)
    );
    let seq: Vec<String> = seq_1.chars().map(|c| one_to_three(c).to_string()).collect();
    assert_eq!(seq.len(), 140);
    let n_pro = seq.iter().filter(|r| r.as_str() == "PRO").count();
    assert_eq!(n_pro, 21, "Expected exactly 21 Prolines (15.0%)");
    seq
}

type ResidueTruthShifts = (Option<f64>, f64, f64, f64, Option<f64>, Vec<f64>);

#[derive(Clone)]
struct SyntheticGroundTruth {
    seq: Vec<String>,
    /// Ground-truth DSS shifts per residue `k`: `(H_opt, N, C', CA, CB_opt, sidechain_c)`.
    shifts: Vec<ResidueTruthShifts>,
    /// True if residue `k` is exchange-broadened (missing from all `1H`-detected spectra).
    exchange_broadened: Vec<bool>,
    /// Ground-truth sequence index for each `CON` root (`None` if spurious noise root).
    con_root_truth: Vec<Option<usize>>,
    /// Ground-truth sequence index for each `HNCO` peak (`None` if spurious).
    hnco_truth: Vec<Option<usize>>,
}

/// Synthesize a realistic multi-experiment Pro/Ser-rich IDP dataset with:
/// - Uniform TMS `-2.70 ppm` `13C` offset across all `13C` dimensions,
/// - `15%` Prolines (no `1H^N`, bridged via `2D CON` + `H(CC)CON` + `CANCO` + `(H)N(CA)CON`),
/// - `~14%` exchange-broadened non-Proline amides (absent in `HN`, present in `CON`),
/// - Field-scaled `800 MHz` TROSY `(-46/800, +46/81.06) ppm` offset on `BT_HNCACONH`,
/// - `F1(C')` NUS sidelobes in `HN(CA)CO`,
/// - `Gly CB` bleed-through near `Gly CA` (`41..46.5 ppm` DSS),
/// - Platzer C-terminal `COO-` (`~179.2 ppm` DSS) on residue `L - 1`,
/// - `CAS` (`cacon`/`canco`) subspectra with `~6%` breakthrough peaks,
/// - Inverted-sign (`cb-positive`) `HNCACB` to exercise automatic polarity normalization.
fn synthesize_realistic_idp_benchmark(seed: u64) -> (JointIdpDataset, SyntheticGroundTruth) {
    let mut rng = Rng::new(seed);
    let seq = synthetic_pro_ser_rich_idp_sequence();
    let len = seq.len();
    let cfg_pred = JointIdpConfig::default();
    let preds = predict_joint_ncidp_shifts(&seq, &cfg_pred);

    let tms_c13_offset = -2.70_f64;
    let proton_mhz = 800.0_f64;
    let trosy_dh = -TROSY_HALF_J_HN_HZ / proton_mhz;
    let trosy_dn = TROSY_HALF_J_HN_HZ / n15_obs_mhz(proton_mhz);

    // 1. Ground-truth DSS backbone shifts with realistic IDP micro-heterogeneity
    let mut shifts = Vec::with_capacity(len);
    for k in 0..len {
        let p = &preds[k];
        let is_pro = seq[k] == "PRO";
        let h = if is_pro || k == 0 {
            None
        } else {
            let base_h = match seq[k].as_str() {
                "GLY" => 8.32,
                "SER" | "THR" => 8.24,
                "ALA" => 8.18,
                "VAL" | "ILE" | "LEU" => 8.12,
                "ASP" | "GLU" => 8.34,
                _ => 8.22,
            };
            Some(base_h + rng.normal(0.0, 0.22))
        };
        let n = p.n_ppm + rng.normal(0.0, if is_pro { 0.55 } else { 0.45 });
        let c = if k + 1 == len {
            179.25 + rng.normal(0.0, 0.15) // Platzer C-terminal COO- (~179 ppm DSS)
        } else {
            p.c_ppm + rng.normal(0.0, 0.22)
        };
        let ca = p.ca_ppm + rng.normal(0.0, 0.22);
        let cb = p.cb_ppm.map(|m| m + rng.normal(0.0, 0.28));
        let sc: Vec<f64> = p
            .sidechain_c
            .iter()
            .map(|&m| m + rng.normal(0.0, 0.25))
            .collect();
        shifts.push((h, n, c, ca, cb, sc));
    }

    // 2. Mark ~14% of non-Proline residues (excluding C-term) as exchange-broadened in `HN`
    let mut exchange_broadened = vec![false; len];
    for k in 2..(len - 1) {
        if seq[k] != "PRO" && (k % 9 == 4 || k % 17 == 11) {
            exchange_broadened[k] = true;
        }
    }

    // 3. Build `HN` peaks (`HSQC`, `HNCO`, `HNCACO` + NUS sidelobes, inverted `HNCACB` + `Gly CB`
    //    bleed-through, `CBCACONH`, and TROSY-shifted `BT_HNCACONH`)
    let mut hn_peaks: Vec<Peak> = Vec::new();
    let mut hncaconh_peaks: Vec<HncaconhPeak> = Vec::new();
    let mut hnco_truth: Vec<Option<usize>> = Vec::new();

    for k in 1..len {
        if seq[k] == "PRO" || exchange_broadened[k] {
            continue;
        }
        let h = shifts[k].0.unwrap();
        let n = shifts[k].1;
        let c_prev_obs = shifts[k - 1].2 + tms_c13_offset + rng.normal(0.0, 0.014);
        let c_curr_obs = shifts[k].2 + tms_c13_offset + rng.normal(0.0, 0.015);
        let ca_curr_obs = shifts[k].3 + tms_c13_offset + rng.normal(0.0, 0.022);
        let ca_prev_obs = shifts[k - 1].3 + tms_c13_offset + rng.normal(0.0, 0.024);
        let cb_curr_obs = shifts[k]
            .4
            .map(|v| v + tms_c13_offset + rng.normal(0.0, 0.028));
        let cb_prev_obs = shifts[k - 1]
            .4
            .map(|v| v + tms_c13_offset + rng.normal(0.0, 0.030));

        // HSQC & HNCO
        hn_peaks.push(
            Peak::hsqc(h + rng.normal(0.0, 0.003), n + rng.normal(0.0, 0.02))
                .with_intensity(1.0e5, None),
        );
        hn_peaks.push(
            Peak::triple(
                "HNCO",
                h + rng.normal(0.0, 0.003),
                n + rng.normal(0.0, 0.02),
                c_prev_obs,
                1,
            )
            .with_intensity(9.5e4, None),
        );
        hnco_truth.push(Some(k));

        // HN(CA)CO: both `C'_i` and `C'_{i-1}` + realistic `F1(C')` NUS sidelobes on ~30% of strips!
        hn_peaks.push(
            Peak::triple(
                "HNCACO",
                h + rng.normal(0.0, 0.004),
                n + rng.normal(0.0, 0.025),
                c_curr_obs,
                1,
            )
            .with_intensity(7.5e4, None),
        );
        hn_peaks.push(
            Peak::triple(
                "HNCACO",
                h + rng.normal(0.0, 0.004),
                n + rng.normal(0.0, 0.025),
                c_prev_obs + rng.normal(0.0, 0.012),
                1,
            )
            .with_intensity(5.2e4, None),
        );
        if k % 3 == 0 {
            // Weak `F1(C')` NUS sidelobe (~11% height at +0.95 ppm)
            hn_peaks.push(
                Peak::triple(
                    "HNCACO",
                    h + rng.normal(0.0, 0.004),
                    n + rng.normal(0.0, 0.025),
                    c_curr_obs + 0.95,
                    1,
                )
                .with_intensity(8.2e3, None),
            );
        }

        // HNCACB with INVERTED polarity (`CA` negative `-1`, `CB` positive `+1`) to test `RD-1` auto-flip!
        hn_peaks.push(
            Peak::triple(
                "HNCACB",
                h + rng.normal(0.0, 0.004),
                n + rng.normal(0.0, 0.025),
                ca_curr_obs,
                -1,
            )
            .with_intensity(8.0e4, None),
        );
        hn_peaks.push(
            Peak::triple(
                "HNCACB",
                h + rng.normal(0.0, 0.004),
                n + rng.normal(0.0, 0.025),
                ca_prev_obs,
                -1,
            )
            .with_intensity(4.8e4, None),
        );
        if let Some(cb) = cb_curr_obs {
            hn_peaks.push(
                Peak::triple(
                    "HNCACB",
                    h + rng.normal(0.0, 0.004),
                    n + rng.normal(0.0, 0.025),
                    cb,
                    1,
                )
                .with_intensity(6.5e4, None),
            );
        } else if seq[k] == "GLY" {
            // Inject `Gly CB` bleed-through near `Gly CA` (inverted polarity = `+1` here!)
            hn_peaks.push(
                Peak::triple(
                    "HNCACB",
                    h + rng.normal(0.0, 0.004),
                    n + rng.normal(0.0, 0.025),
                    ca_curr_obs + 0.28,
                    1,
                )
                .with_intensity(1.4e4, None),
            );
        }
        if let Some(cb) = cb_prev_obs {
            hn_peaks.push(
                Peak::triple(
                    "HNCACB",
                    h + rng.normal(0.0, 0.004),
                    n + rng.normal(0.0, 0.025),
                    cb,
                    1,
                )
                .with_intensity(4.2e4, None),
            );
        }

        // CBCACONH (`i - 1` `CA` and `CB`, unipolar `+1`)
        hn_peaks.push(
            Peak::triple(
                "CBCACONH",
                h + rng.normal(0.0, 0.004),
                n + rng.normal(0.0, 0.025),
                ca_prev_obs + rng.normal(0.0, 0.015),
                1,
            )
            .with_intensity(6.0e4, None),
        );
        if let Some(cb) = cb_prev_obs {
            hn_peaks.push(
                Peak::triple(
                    "CBCACONH",
                    h + rng.normal(0.0, 0.004),
                    n + rng.normal(0.0, 0.025),
                    cb + rng.normal(0.0, 0.018),
                    1,
                )
                .with_intensity(5.5e4, None),
            );
        }

        // BT_HNCACONH with `800 MHz` TROSY `(-46 Hz H, +46 Hz N)` offset!
        if k >= 2 && seq[k - 1] != "PRO" && !exchange_broadened[k - 1] {
            hncaconh_peaks.push(HncaconhPeak {
                h_ppm: h + trosy_dh + rng.normal(0.0, 0.004),
                n_curr_ppm: n + trosy_dn + rng.normal(0.0, 0.025),
                n_partner_ppm: shifts[k - 1].1 + rng.normal(0.0, 0.03),
                height: 4.5e4,
            });
        }
        if k + 1 < len && seq[k + 1] != "PRO" && !exchange_broadened[k + 1] {
            hncaconh_peaks.push(HncaconhPeak {
                h_ppm: h + trosy_dh + rng.normal(0.0, 0.004),
                n_curr_ppm: n + trosy_dn + rng.normal(0.0, 0.025),
                n_partner_ppm: shifts[k + 1].1 + rng.normal(0.0, 0.03),
                height: 4.2e4,
            });
        }
    }

    // 4. Build `13C`-detected `2D CON` + realistic lower-sensitivity `H(CC)CON`/`CBCACON` +
    //    `CANCO` on Prolines/exch-broadened + sparse `(H)N(CA)CON` + `CAS` subspectra!
    //    Realistic `13C` acquisition physics:
    //    - ~10% of non-Proline amides drop out / overlap in `2D CON` (still resolved in `HN`!),
    //    - `H(CC)CON`/`CBCACON` has ~28% strip dropout (lower `13C` sensitivity vs `1H`),
    //    - `(H)N(CA)CON` has ~45% coverage (essential across Pro/exchange gaps, sparser elsewhere),
    //    - `NCACO` is omitted so `CON`-only lacks intraresidue `C'_i` & `CA_i`/`CB_i` on most amides,
    //      while `idp-joint` pools `HN(CA)CO` + `HNCACB` + `2D CON` seamlessly!
    let mut con_roots: Vec<Con2DPeak> = Vec::new();
    let mut strip_peaks: Vec<C13Strip3DPeak> = Vec::new();
    let mut hncacon_nn_edges: Vec<DirectionalNnPeak> = Vec::new();
    let mut con_root_truth: Vec<Option<usize>> = Vec::new();

    for k in 1..len {
        // ~10% of non-Pro, non-exchange-broadened amides drop out in crowded `2D CON`
        let con_dropped = seq[k] != "PRO" && !exchange_broadened[k] && k + 1 < len && (k % 11 == 3);
        if con_dropped {
            continue;
        }
        let c_prev_obs = shifts[k - 1].2 + tms_c13_offset + rng.normal(0.0, 0.012);
        let n_curr_obs = shifts[k].1 + rng.normal(0.0, 0.022);
        let r_idx = con_roots.len();
        con_roots.push(Con2DPeak {
            c_prev_ppm: c_prev_obs,
            n_curr_ppm: n_curr_obs,
            height: 8.5e4,
            volume: 8.5e4,
            is_proline_imino_region: n_curr_obs >= 131.5,
            deconvolved_from_cluster: false,
        });
        con_root_truth.push(Some(k));

        // Preceding `(CA, CB, sidechain)` via `CBCACON`/`H(CC)CON` (~72% coverage; always on Pro/exch)
        if seq[k] == "PRO" || exchange_broadened[k] || k % 4 != 1 {
            let ca_prev = shifts[k - 1].3 + tms_c13_offset + rng.normal(0.0, 0.026);
            strip_peaks.push(C13Strip3DPeak {
                exp: C13DetectedExp::Cbcaco3D,
                c_prime_ppm: c_prev_obs,
                n_ppm: n_curr_obs,
                c_indirect_ppm: ca_prev,
                sign: 1,
                height: 5.5e4,
                volume: 5.5e4,
                anchor_root_idx: Some(r_idx),
            });
            if let Some(cb) = shifts[k - 1].4 {
                strip_peaks.push(C13Strip3DPeak {
                    exp: C13DetectedExp::Cbcaco3D,
                    c_prime_ppm: c_prev_obs,
                    n_ppm: n_curr_obs,
                    c_indirect_ppm: cb + tms_c13_offset + rng.normal(0.0, 0.030),
                    sign: -1,
                    height: 4.5e4,
                    volume: 4.5e4,
                    anchor_root_idx: Some(r_idx),
                });
            }
            for &sc in &shifts[k - 1].5 {
                strip_peaks.push(C13Strip3DPeak {
                    exp: C13DetectedExp::Cccon3D,
                    c_prime_ppm: c_prev_obs,
                    n_ppm: n_curr_obs,
                    c_indirect_ppm: sc + tms_c13_offset + rng.normal(0.0, 0.032),
                    sign: 1,
                    height: 3.5e4,
                    volume: 3.5e4,
                    anchor_root_idx: Some(r_idx),
                });
            }
        }

        // `CANCO` on Prolines and exchange-broadened `CON`-only nodes (bridging `CA_i` across Pro!)
        if seq[k] == "PRO" || exchange_broadened[k] {
            strip_peaks.push(C13Strip3DPeak {
                exp: C13DetectedExp::Canco3D,
                c_prime_ppm: c_prev_obs,
                n_ppm: n_curr_obs,
                c_indirect_ppm: shifts[k - 1].3 + tms_c13_offset + rng.normal(0.0, 0.024),
                sign: 1,
                height: 6.0e4,
                volume: 6.0e4,
                anchor_root_idx: Some(r_idx),
            });
            strip_peaks.push(C13Strip3DPeak {
                exp: C13DetectedExp::Canco3D,
                c_prime_ppm: c_prev_obs,
                n_ppm: n_curr_obs,
                c_indirect_ppm: shifts[k].3 + tms_c13_offset + rng.normal(0.0, 0.028),
                sign: 1,
                height: 4.6e4,
                volume: 4.6e4,
                anchor_root_idx: Some(r_idx),
            });
        }

        // Sparse directional `(H)N(CA)CON` sequential `N(i-1)` edges (~45% overall, covering Pro/exch gaps)
        if k >= 2
            && (seq[k] == "PRO"
                || seq[k - 1] == "PRO"
                || exchange_broadened[k]
                || exchange_broadened[k - 1]
                || k % 5 == 0)
        {
            hncacon_nn_edges.push(DirectionalNnPeak {
                c_prev_ppm: c_prev_obs,
                n_curr_ppm: n_curr_obs,
                n_prev_ppm: shifts[k - 1].1 + rng.normal(0.0, 0.055),
                snr: 6.5,
            });
        }
    }

    // Add 3 spurious noise roots in `2D CON` (no 3D support)
    for sp_i in 0..3 {
        con_roots.push(Con2DPeak {
            c_prev_ppm: 171.2 + (sp_i as f64) * 1.1,
            n_curr_ppm: 117.5 + (sp_i as f64) * 2.3,
            height: 2.0e4,
            volume: 2.0e4,
            is_proline_imino_region: false,
            deconvolved_from_cluster: false,
        });
        con_root_truth.push(None);
    }

    // 5. Build `5` `CAS` subspectra (`cacon` on `i-1`, `canco` on `i`) WITH ~6% breakthrough false peaks
    let make_cas = |name: &str, mode: CasMode, res: &[&str], rng: &mut Rng| -> CasSpectrum {
        let mut peaks = Vec::new();
        for (r_idx, cr) in con_roots.iter().enumerate() {
            let Some(k) = con_root_truth[r_idx] else {
                continue;
            };
            let target = match mode {
                CasMode::Cacon => seq[k - 1].as_str(),
                CasMode::Canco => seq[k].as_str(),
            };
            let is_true = res.contains(&target);
            if is_true && rng.f64() < 0.90 {
                peaks.push(CasPeak {
                    c_prime_ppm: cr.c_prev_ppm + rng.normal(0.0, 0.012),
                    n_ppm: cr.n_curr_ppm + rng.normal(0.0, 0.03),
                    height: 6.5e4,
                    volume: 6.5e4,
                });
            } else if !is_true && rng.f64() < 0.055 {
                // Weak incomplete-suppression breakthrough peak (~22% height)
                peaks.push(CasPeak {
                    c_prime_ppm: cr.c_prev_ppm + rng.normal(0.0, 0.015),
                    n_ppm: cr.n_curr_ppm + rng.normal(0.0, 0.04),
                    height: 1.4e4,
                    volume: 1.4e4,
                });
            }
        }
        CasSpectrum {
            name: name.to_string(),
            mode,
            residues: res.iter().map(|s| s.to_string()).collect(),
            peaks,
        }
    };

    let cas_spectra = vec![
        make_cas("cacon_ser", CasMode::Cacon, &["SER"], &mut rng),
        make_cas("cacon_ala", CasMode::Cacon, &["ALA"], &mut rng),
        make_cas("cacon_gly", CasMode::Cacon, &["GLY"], &mut rng),
        make_cas("canco_ser", CasMode::Canco, &["SER"], &mut rng),
        make_cas(
            "canco_tavi",
            CasMode::Canco,
            &["THR", "ALA", "VAL", "ILE"],
            &mut rng,
        ),
    ];

    let dataset = JointIdpDataset {
        hn_peaks,
        c13_dataset: C13PickedDataset {
            con_roots,
            caco_peaks: Vec::new(),
            strip_peaks,
        },
        cas_spectra,
        hncacon_nn_edges,
        hncaconh_peaks,
        reciprocal_nn_pairs: Vec::new(),
        proton_mhz,
        user_locks: Vec::new(),
    };

    let truth = SyntheticGroundTruth {
        seq,
        shifts,
        exchange_broadened,
        con_root_truth,
        hnco_truth,
    };

    (dataset, truth)
}

/// Evaluate node-to-ground-truth accuracy, recall over all `L - 1` assignable bonds, Proline-bridge
/// recall, exchange-broadened `ConOnlyAmide` recall, and HC (`>= 0.80`) precision.
fn evaluate_against_truth(
    res: &nmr_core::joint_idp_solver::JointIdpAssignResult,
    truth: &SyntheticGroundTruth,
) -> (f64, f64, f64, f64, f64, usize, usize, usize, usize) {
    let tms_c13_offset = -2.70_f64;
    let len = truth.seq.len();
    let total_assignable = len - 1;
    let total_pro = (1..len).filter(|&k| truth.seq[k] == "PRO").count();
    let total_exch = (1..len).filter(|&k| truth.exchange_broadened[k]).count();

    // Map each node to its unique true sequence index `k` by matching `(c_prev, n_curr)` to ground truth
    let mut node_true_k: Vec<Option<usize>> = vec![None; res.nodes.len()];
    for (s_idx, node) in res.nodes.iter().enumerate() {
        let mut best: Option<(usize, f64)> = None;
        for k in 1..len {
            let true_n = truth.shifts[k].1;
            let true_cp = truth.shifts[k - 1].2 + tms_c13_offset;
            let dn = (node.n_curr - true_n).abs();
            let dc = node.c_prev.map_or(0.0, |c| (c - true_cp).abs());
            if dn <= 0.16 && dc <= 0.08 {
                let d2 = (dn / 0.05).powi(2) + (dc / 0.025).powi(2);
                if best.is_none_or(|b| d2 < b.1) {
                    best = Some((k, d2));
                }
            }
        }
        node_true_k[s_idx] = best.map(|b| b.0);
    }

    let mut assigned = 0usize;
    let mut correct = 0usize;
    let mut hc_assigned = 0usize;
    let mut hc_correct = 0usize;
    let mut pro_correct = 0usize;
    let mut exch_correct = 0usize;

    let mut seen_true_k: BTreeMap<usize, bool> = BTreeMap::new();
    for (s, &true_k) in node_true_k.iter().enumerate() {
        let pred_k = res.position[s];
        if pred_k < 0 {
            continue;
        }
        assigned += 1;
        let is_ok = true_k == Some(pred_k as usize);
        if is_ok {
            correct += 1;
            let tk = pred_k as usize;
            if seen_true_k.insert(tk, true).is_none() {
                if truth.seq[tk] == "PRO" {
                    pro_correct += 1;
                }
                if truth.exchange_broadened[tk] {
                    exch_correct += 1;
                }
            }
        }
        if res.confidence[s] >= 0.80 {
            hc_assigned += 1;
            if is_ok {
                hc_correct += 1;
            }
        }
    }

    let acc = if assigned > 0 {
        (correct as f64) / (assigned as f64)
    } else {
        0.0
    };
    let rec = (correct as f64) / (total_assignable as f64);
    let pro_rec = if total_pro > 0 {
        (pro_correct as f64) / (total_pro as f64)
    } else {
        0.0
    };
    let exch_rec = if total_exch > 0 {
        (exch_correct as f64) / (total_exch as f64)
    } else {
        0.0
    };
    let hc_prec = if hc_assigned > 0 {
        (hc_correct as f64) / (hc_assigned as f64)
    } else {
        1.0
    };

    (
        acc,
        rec,
        pro_rec,
        exch_rec,
        hc_prec,
        correct,
        total_assignable,
        pro_correct,
        total_pro,
    )
}

#[test]
fn test_synthetic_pro_ser_rich_idp_joint_beats_hn_and_con_only() {
    let (full_dataset, truth) = synthesize_realistic_idp_benchmark(20261121);
    let seq = &truth.seq;
    assert_eq!(
        truth.con_root_truth.len(),
        full_dataset.c13_dataset.con_roots.len()
    );
    assert!(!truth.hnco_truth.is_empty());

    let cfg = JointIdpConfig {
        replicas: 24,
        islands: 4,
        generations: 3,
        steps_per_system: 1_600,
        seed: 42,
        ..JointIdpConfig::default()
    };

    // Arm 1: `HN`-only (no 2D CON, no 13C strips, no CAS, no (H)N(CA)CON)
    let hn_only_dataset = JointIdpDataset {
        hn_peaks: full_dataset.hn_peaks.clone(),
        c13_dataset: C13PickedDataset::default(),
        cas_spectra: Vec::new(),
        hncacon_nn_edges: Vec::new(),
        hncaconh_peaks: full_dataset.hncaconh_peaks.clone(),
        reciprocal_nn_pairs: Vec::new(),
        proton_mhz: full_dataset.proton_mhz,
        user_locks: Vec::new(),
    };
    let res_hn = assign_joint_idp(&hn_only_dataset, seq, &cfg).expect("HN-only solve");
    let (hn_acc, hn_rec, hn_pro_rec, hn_exch_rec, hn_hcp, hn_ok, n_tot, hn_pro_ok, n_pro) =
        evaluate_against_truth(&res_hn, &truth);

    // Arm 2: `CON`-only (`2D CON` + `H(CC)CON`/`CBCACON` + `CAS` + `(H)N(CA)CON`, no `HN` spectra)
    let con_only_dataset = JointIdpDataset {
        hn_peaks: Vec::new(),
        c13_dataset: full_dataset.c13_dataset.clone(),
        cas_spectra: full_dataset.cas_spectra.clone(),
        hncacon_nn_edges: full_dataset.hncacon_nn_edges.clone(),
        hncaconh_peaks: Vec::new(),
        reciprocal_nn_pairs: Vec::new(),
        proton_mhz: full_dataset.proton_mhz,
        user_locks: Vec::new(),
    };
    let res_con = assign_joint_idp(&con_only_dataset, seq, &cfg).expect("CON-only solve");
    let (con_acc, con_rec, con_pro_rec, con_exch_rec, con_hcp, con_ok, _, con_pro_ok, _) =
        evaluate_against_truth(&res_con, &truth);

    // Arm 3: Unified `idp-joint` (pooled `HN + 2D CON + Proline-Bridge + N-N + CAS`)
    let res_joint = assign_joint_idp(&full_dataset, seq, &cfg).expect("Joint IDP solve");
    let (jt_acc, jt_rec, jt_pro_rec, jt_exch_rec, jt_hcp, jt_ok, _, jt_pro_ok, _) =
        evaluate_against_truth(&res_joint, &truth);

    eprintln!(
        "\n=== Synthetic 140-aa Pro/Ser-Rich IDP Benchmark (15% Pro, 14% Exch, -2.7 TMS, 800 MHz TROSY) ==="
    );
    eprintln!(
        "HN-only  : acc={:5.1}% rec={:5.1}% ({:3}/{:3}) Pro={:5.1}% ({:2}/{:2}) Exch={:5.1}% HCP={:5.1}%",
        100.0 * hn_acc, 100.0 * hn_rec, hn_ok, n_tot, 100.0 * hn_pro_rec, hn_pro_ok, n_pro, 100.0 * hn_exch_rec, 100.0 * hn_hcp
    );
    eprintln!(
        "CON-only : acc={:5.1}% rec={:5.1}% ({:3}/{:3}) Pro={:5.1}% ({:2}/{:2}) Exch={:5.1}% HCP={:5.1}%",
        100.0 * con_acc, 100.0 * con_rec, con_ok, n_tot, 100.0 * con_pro_rec, con_pro_ok, n_pro, 100.0 * con_exch_rec, 100.0 * con_hcp
    );
    eprintln!(
        "idp-joint: acc={:5.1}% rec={:5.1}% ({:3}/{:3}) Pro={:5.1}% ({:2}/{:2}) Exch={:5.1}% HCP={:5.1}% | locks={} sidelobes={} gly_guard={}",
        100.0 * jt_acc, 100.0 * jt_rec, jt_ok, n_tot, 100.0 * jt_pro_rec, jt_pro_ok, n_pro, 100.0 * jt_exch_rec, 100.0 * jt_hcp,
        res_joint.pass1_locked_anchors.len(), res_joint.n_sidelobes_pruned, res_joint.n_gly_cb_guarded
    );
    eprintln!(
        "Offsets  : dN={:+.3}, dC'={:+.3}, dCA={:+.3}, dCB={:+.3} | TROSY=({:+.4} H, {:+.4} N, detected={}) | routes: XVal={} ProBr={} ConOnly={} HnOnly={}",
        res_joint.fitted_offsets.d_n,
        res_joint.fitted_offsets.d_c,
        res_joint.fitted_offsets.d_ca,
        res_joint.fitted_offsets.d_cb,
        res_joint.trosy_calibration.dh_ppm,
        res_joint.trosy_calibration.dn_ppm,
        res_joint.trosy_calibration.trosy_detected,
        res_joint.n_cross_validated_hn_con,
        res_joint.n_con_pro_bridges,
        res_joint.n_con_route_only,
        res_joint.n_hn_route_only
    );

    // 1. Unified `idp-joint` significantly outperforms BOTH `HN`-only and `CON`-only!
    assert!(
        jt_rec >= 0.88,
        "Expected idp-joint recall >= 88%, got {:.1}% ({jt_ok}/{n_tot})",
        100.0 * jt_rec
    );
    assert!(
        jt_acc >= 0.88,
        "Expected idp-joint accuracy >= 88%, got {:.1}%",
        100.0 * jt_acc
    );
    assert!(
        jt_rec >= hn_rec + 0.18,
        "Expected idp-joint recall ({:.1}%) to beat HN-only ({:.1}%) by >= 18 pp",
        100.0 * jt_rec,
        100.0 * hn_rec
    );
    assert!(
        jt_rec >= con_rec + 0.12 && jt_acc > con_acc && jt_acc > hn_acc,
        "Expected idp-joint recall ({:.1}%) & acc ({:.1}%) to significantly beat CON-only ({:.1}% / {:.1}%) and HN-only ({:.1}% / {:.1}%)",
        100.0 * jt_rec,
        100.0 * jt_acc,
        100.0 * con_rec,
        100.0 * con_acc,
        100.0 * hn_rec,
        100.0 * hn_acc
    );
    assert!(
        jt_hcp >= 0.92,
        "Expected idp-joint HC precision >= 92%, got {:.1}%",
        100.0 * jt_hcp
    );

    // 2. Proline-bridge recovery & exchange-broadened `ConOnlyAmide` recovery
    assert_eq!(hn_pro_ok, 0, "HN-only cannot assign Prolines");
    assert!(
        jt_pro_rec >= 0.85,
        "Expected idp-joint Proline-bridge recall >= 85%, got {:.1}% ({jt_pro_ok}/{n_pro})",
        100.0 * jt_pro_rec
    );
    assert!(
        jt_exch_rec >= 0.70,
        "Expected idp-joint exchange-broadened ConOnlyAmide recall >= 70%, got {:.1}%",
        100.0 * jt_exch_rec
    );
    assert!(res_joint.n_cross_validated_hn_con >= 70);
    assert!(res_joint.n_con_pro_bridges >= 18);
    assert!(res_joint.n_con_route_only >= 10);
    assert!(
        res_joint.n_hn_route_only >= 5,
        "Expected HnRoute-only recovery for CON-dropped amides, got {}",
        res_joint.n_hn_route_only
    );

    // 3. Auto-calibration of uniform `-2.70 ppm` TMS `13C` offset and `800 MHz` TROSY `±46 Hz / MHz`
    assert!(
        (res_joint.fitted_offsets.d_c - (-2.70)).abs() <= 0.25,
        "Expected dC' ≈ -2.70 ppm TMS, got {:+.3}",
        res_joint.fitted_offsets.d_c
    );
    assert!(
        (res_joint.fitted_offsets.d_ca - (-2.70)).abs() <= 0.25,
        "Expected dCA ≈ -2.70 ppm TMS, got {:+.3}",
        res_joint.fitted_offsets.d_ca
    );
    assert!(
        (res_joint.fitted_offsets.d_cb - (-2.70)).abs() <= 0.30,
        "Expected dCB ≈ -2.70 ppm TMS, got {:+.3}",
        res_joint.fitted_offsets.d_cb
    );
    assert!(
        res_joint.auto_ref_report.tms_reference_hint.is_some(),
        "Expected TMS reference hint on uniform -2.7 ppm 13C offset"
    );

    let exp_dh = -TROSY_HALF_J_HN_HZ / 800.0;
    let exp_dn = TROSY_HALF_J_HN_HZ / n15_obs_mhz(800.0);
    assert!(res_joint.trosy_calibration.trosy_detected);
    assert!(
        (res_joint.trosy_calibration.dh_ppm - exp_dh).abs() <= 0.015,
        "Expected TROSY dH ≈ {exp_dh:+.4}, got {:+.4}",
        res_joint.trosy_calibration.dh_ppm
    );
    assert!(
        (res_joint.trosy_calibration.dn_ppm - exp_dn).abs() <= 0.10,
        "Expected TROSY dN ≈ {exp_dn:+.4}, got {:+.4}",
        res_joint.trosy_calibration.dn_ppm
    );

    // 4. Strip hygiene, C-terminal COO- prior, and 2-pass anchor-lock telemetry
    assert!(
        res_joint.n_sidelobes_pruned >= 20,
        "Expected F1(C') NUS sidelobes pruned"
    );
    assert!(
        res_joint.n_gly_cb_guarded >= 5,
        "Expected Gly CB bleed-through guarded"
    );
    assert!(
        !res_joint.c_term_coo_candidates.is_empty(),
        "Expected C-terminal COO- (~179 ppm DSS) candidate detected"
    );
    assert!(
        !res_joint.pass1_locked_anchors.is_empty(),
        "Expected Pass-1 unequivocal anchors locked before Pass-2"
    );
    assert!(
        res_joint.residue_assignments[seq.len() - 1].is_cterm_coo,
        "Expected C-terminal residue flagged as COO- anchored"
    );
    assert_eq!(res_joint.qc_verdict, DatasetQcVerdict::Pass);
}

#[test]
fn test_trosy_half_j_hn_calibration_600_and_800_mhz() {
    let mut rng = Rng::new(77);
    let decoupled: Vec<(f64, f64)> = (0..35)
        .map(|i| (7.6 + (i as f64) * 0.04, 112.0 + (i as f64) * 0.55))
        .collect();

    // 1. Canonical TROSY at 600 MHz (`-46/600 = -0.0767 H, +46/60.80 = +0.7566 N`)
    let dh600 = -TROSY_HALF_J_HN_HZ / 600.0;
    let dn600 = TROSY_HALF_J_HN_HZ / n15_obs_mhz(600.0);
    let trosy600: Vec<(f64, f64)> = decoupled
        .iter()
        .map(|&(h, n)| {
            (
                h + dh600 + rng.normal(0.0, 0.003),
                n + dn600 + rng.normal(0.0, 0.02),
            )
        })
        .collect();
    let cal600 = calibrate_trosy_hn_offset(&trosy600, &decoupled, 600.0);
    assert!(cal600.trosy_detected);
    assert!((cal600.dh_ppm - dh600).abs() < 0.01);
    assert!((cal600.dn_ppm - dn600).abs() < 0.06);

    // 2. Anti-TROSY / inverted-sign quadrant at 800 MHz (`+46/800 H, -46/81.06 N`)
    let dh800 = TROSY_HALF_J_HN_HZ / 800.0;
    let dn800 = -TROSY_HALF_J_HN_HZ / n15_obs_mhz(800.0);
    let trosy800: Vec<(f64, f64)> = decoupled
        .iter()
        .map(|&(h, n)| {
            (
                h + dh800 + rng.normal(0.0, 0.003),
                n + dn800 + rng.normal(0.0, 0.02),
            )
        })
        .collect();
    let cal800 = calibrate_trosy_hn_offset(&trosy800, &decoupled, 800.0);
    assert!(cal800.trosy_detected);
    assert!((cal800.dh_ppm - dh800).abs() < 0.01);
    assert!((cal800.dn_ppm - dn800).abs() < 0.06);

    // 3. Decoupled null (zero shift): must NOT trigger a false TROSY shift!
    let dec_noisy: Vec<(f64, f64)> = decoupled
        .iter()
        .map(|&(h, n)| (h + rng.normal(0.0, 0.003), n + rng.normal(0.0, 0.02)))
        .collect();
    let cal_null = calibrate_trosy_hn_offset(&dec_noisy, &decoupled, 800.0);
    assert!(!cal_null.trosy_detected);
    assert_eq!(cal_null.dh_ppm, 0.0);
    assert_eq!(cal_null.dn_ppm, 0.0);
}

#[test]
fn test_f1_c_prime_nus_sidelobe_pruning_and_gly_cb_bleedthrough_guard() {
    // 1. `F1(C')` NUS sidelobe pruning retains real `C'_i` + `C'_{i-1}` while dropping weak sidelobes
    let raw_strip = vec![
        (175.40, 100_000.0), // true C'_i
        (174.10, 68_000.0),  // true C'_{i-1}
        (176.35, 11_000.0),  // NUS sidelobe (11% within 2.2 ppm)
        (174.90, 9_500.0),   // NUS sidelobe (9.5% within 2.2 ppm)
        (150.00, 80_000.0),  // out-of-window noise
    ];
    let (kept, pruned) = prune_f1_c_prime_nus_sidelobes(&raw_strip, 3);
    assert_eq!(kept.len(), 2);
    assert_eq!(pruned, 3);
    assert!((kept[0].0 - 175.40).abs() < 1e-6);
    assert!((kept[1].0 - 174.10).abs() < 1e-6);

    // 2. `Gly CB` bleed-through guard under BOTH DSS (0.0) and TMS (-2.7 ppm) offsets
    let make_node = |id: usize, ca_c: f64, cb_c: f64| JointIdpNode {
        id,
        kind: JointNodeKind::AmideFused,
        hn_matched: true,
        con_matched: true,
        h_curr: Some(8.2),
        n_curr: 112.0,
        hn_n_curr: Some(112.0),
        c_prev: Some(174.0),
        root_height: 1.0e5,
        ca_curr: vec![ca_c],
        cb_curr: vec![cb_c],
        ca_prev: Vec::new(),
        cb_prev: Vec::new(),
        c_curr: Vec::new(),
        sidechain_c_prev: Vec::new(),
        nn_prev_cands: Vec::new(),
        recip_nn_partners: Vec::new(),
        prev_gly_opposite_phase: false,
        has_prev_cb_exp: true,
        has_curr_cb_exp: true,
        gly_cb_curr_suppressed: false,
        gly_cb_prev_suppressed: false,
        f1_sidelobes_pruned: 0,
        registration_warnings: Vec::new(),
    };

    // Node 0: Gly with CA=45.1 - 2.7 = 42.4, bleed-through CB=42.7; Node 1: Ser with CA=55.6, CB=61.1
    let mut nodes = vec![make_node(0, 42.4, 42.7), make_node(1, 55.6, 61.1)];
    let tms_off = IdpShiftOffsets {
        d_n: 0.0,
        d_c: -2.7,
        d_ca: -2.7,
        d_cb: -2.7,
    };
    let n_guarded = apply_gly_cb_bleedthrough_guard(&mut nodes, &tms_off);
    assert_eq!(n_guarded, 1);
    assert!(nodes[0].cb_curr.is_empty());
    assert!(nodes[0].gly_cb_curr_suppressed);
    assert_eq!(nodes[1].cb_curr, vec![61.1]);
    assert!(!nodes[1].gly_cb_curr_suppressed);
}

#[test]
fn test_platzer_cterm_coo_and_pre_proline_priors() {
    let seq: Vec<String> = ["ALA", "SER", "PRO", "GLY", "LEU", "ALA"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let cfg = JointIdpConfig::default();
    let preds = predict_joint_ncidp_shifts(&seq, &cfg);

    // Pre-Proline (`SER` at index 1 before `PRO` at index 2) vs internal `SER`
    let seq_nopro: Vec<String> = ["ALA", "SER", "ALA", "GLY", "LEU", "ALA"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let preds_nopro = predict_joint_ncidp_shifts(&seq_nopro, &cfg);
    let dc_pre_pro = preds[1].c_ppm - preds_nopro[1].c_ppm;
    let dca_pre_pro = preds[1].ca_ppm - preds_nopro[1].ca_ppm;
    assert!(
        dc_pre_pro < -1.7 && dc_pre_pro > -2.3,
        "Expected pre-Pro C' shift ≈ -1.9 ppm, got {dc_pre_pro:+.2}"
    );
    assert!(
        dca_pre_pro < -1.7 && dca_pre_pro > -2.3,
        "Expected pre-Pro CA shift ≈ -1.8 ppm, got {dca_pre_pro:+.2}"
    );

    // C-terminal `COO-` at `k = L - 1` (`~179 ppm` DSS at pH 6.5) vs acidic pH 2.0 protonation
    let last = preds.len() - 1;
    assert!(
        preds[last].c_ppm >= 178.4 && preds[last].c_ppm <= 181.0,
        "Expected Platzer C-term COO- ≈ 179 ppm DSS, got {:.2}",
        preds[last].c_ppm
    );
    let cfg_acid = JointIdpConfig {
        ph: 2.0,
        ..JointIdpConfig::default()
    };
    let preds_acid = predict_joint_ncidp_shifts(&seq, &cfg_acid);
    assert!(
        preds_acid[last].c_ppm < preds[last].c_ppm - 2.0,
        "Expected acidic protonation of C-term COO- -> COOH"
    );

    // Verify unary matrix rewards a `179.1 ppm` C-term `c_curr` node at `k = L - 1` over internal `ALA`
    let node_coo = JointIdpNode {
        id: 0,
        kind: JointNodeKind::AmideFused,
        hn_matched: true,
        con_matched: true,
        h_curr: Some(8.05),
        n_curr: preds[last].n_ppm,
        hn_n_curr: Some(preds[last].n_ppm),
        c_prev: Some(preds[last - 1].c_ppm),
        root_height: 1.0e5,
        ca_curr: vec![preds[last].ca_ppm],
        cb_curr: vec![preds[last].cb_ppm.unwrap()],
        ca_prev: vec![preds[last - 1].ca_ppm],
        cb_prev: vec![preds[last - 1].cb_ppm.unwrap()],
        c_curr: vec![179.2],
        sidechain_c_prev: Vec::new(),
        nn_prev_cands: Vec::new(),
        recip_nn_partners: Vec::new(),
        prev_gly_opposite_phase: false,
        has_prev_cb_exp: true,
        has_curr_cb_exp: true,
        gly_cb_curr_suppressed: false,
        gly_cb_prev_suppressed: false,
        f1_sidelobes_pruned: 0,
        registration_warnings: Vec::new(),
    };
    let u = joint_idp_unary_matrix(
        &[node_coo],
        &seq_nopro,
        &[],
        &cfg,
        &IdpShiftOffsets::default(),
        &[],
    );
    assert!(
        u.get(0, last) > u.get(0, 2) + 3.0,
        "Expected C-term COO- node to strongly prefer k = L - 1 ({:.2}) over internal ALA ({:.2})",
        u.get(0, last),
        u.get(0, 2)
    );
}

#[test]
fn test_calibrated_cas_overlap_and_breakthrough_damping() {
    let seq: Vec<String> = ["ALA", "SER", "GLY", "VAL", "LEU", "SER", "ALA", "GLU"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let cfg = JointIdpConfig::default();

    let make_root = |id: usize, cp: f64, n: f64| JointIdpNode {
        id,
        kind: JointNodeKind::AmideFused,
        hn_matched: true,
        con_matched: true,
        h_curr: Some(8.2),
        n_curr: n,
        hn_n_curr: Some(n),
        c_prev: Some(cp),
        root_height: 1.0e5,
        ca_curr: Vec::new(),
        cb_curr: Vec::new(),
        ca_prev: Vec::new(),
        cb_prev: Vec::new(),
        c_curr: Vec::new(),
        sidechain_c_prev: Vec::new(),
        nn_prev_cands: Vec::new(),
        recip_nn_partners: Vec::new(),
        prev_gly_opposite_phase: false,
        has_prev_cb_exp: false,
        has_curr_cb_exp: false,
        gly_cb_curr_suppressed: false,
        gly_cb_prev_suppressed: false,
        f1_sidelobes_pruned: 0,
        registration_warnings: Vec::new(),
    };

    // Node 0 isolated at (174.50, 116.0); Nodes 1 & 2 crowded/overlapping at (175.00, 120.0) & (175.02, 120.05)
    let nodes = vec![
        make_root(0, 174.50, 116.0),
        make_root(1, 175.00, 120.0),
        make_root(2, 175.02, 120.05),
        make_root(3, 176.20, 122.0),
        make_root(4, 176.90, 124.0),
        make_root(5, 177.50, 126.0),
    ];

    let cas_clean = CasSpectrum {
        name: "cacon_ser".to_string(),
        mode: CasMode::Cacon,
        residues: vec!["SER".to_string()],
        peaks: vec![
            CasPeak {
                c_prime_ppm: 174.50,
                n_ppm: 116.0,
                height: 6.0e4,
                volume: 6.0e4,
            },
            CasPeak {
                c_prime_ppm: 175.01,
                n_ppm: 120.02,
                height: 6.0e4,
                volume: 6.0e4,
            },
        ],
    };

    // Breakthrough-heavy CAS hitting 5 of 6 roots when only 2 SER exist in sequence
    let cas_breakthrough = CasSpectrum {
        name: "canco_ser_bt".to_string(),
        mode: CasMode::Canco,
        residues: vec!["SER".to_string()],
        peaks: nodes
            .iter()
            .take(5)
            .map(|nd| CasPeak {
                c_prime_ppm: nd.c_prev.unwrap(),
                n_ppm: nd.n_curr,
                height: 6.0e4,
                volume: 6.0e4,
            })
            .collect(),
    };

    let tables = build_calibrated_cas_tables(&[cas_clean, cas_breakthrough], &nodes, &seq, &cfg);
    assert_eq!(tables.len(), 2);

    // Isolated hit (node 0) must carry strictly stronger weight & LLR than overlapped hit (node 1)
    assert!(
        tables[0].hit_weight[0] > tables[0].hit_weight[1],
        "Expected overlap damping: isolated {:.3} > crowded {:.3}",
        tables[0].hit_weight[0],
        tables[0].hit_weight[1]
    );
    assert!(tables[0].llr_for(0, "SER") > 1.2);
    assert!(tables[0].llr_for(0, "ALA") < -1.0);
    // Breakthrough spectrum must inflate `p_fp_eff` and yield a smaller per-hit LLR than clean spectrum
    assert!(tables[1].p_fp_eff > tables[0].p_fp_eff);
    assert!(tables[1].llr_for(0, "SER") < tables[0].llr_for(0, "SER"));
}

#[test]
fn test_hostile_empty_and_degenerate_inputs_zero_nan_or_panic() {
    let cfg = JointIdpConfig {
        replicas: 6,
        islands: 2,
        generations: 1,
        steps_per_system: 100,
        ..JointIdpConfig::default()
    };

    // 1. Empty sequence & empty dataset
    let r_empty = assign_joint_idp(&JointIdpDataset::default(), &[], &cfg).unwrap();
    assert_eq!(r_empty.qc_verdict, DatasetQcVerdict::Abstain);
    assert!(r_empty.score.is_finite());

    // 2. Single-residue sequence
    let r_one = assign_joint_idp(&JointIdpDataset::default(), &["ALA".to_string()], &cfg).unwrap();
    assert_eq!(r_one.qc_verdict, DatasetQcVerdict::Abstain);
    assert_eq!(r_one.residue_assignments.len(), 1);

    // 3. Hostile NaN / Inf / out-of-range coordinates, NaN proton_mhz, hostile CAS & N-N peaks
    let hostile = JointIdpDataset {
        hn_peaks: vec![
            Peak::hsqc(f64::NAN, 120.0),
            Peak::triple("HNCO", 8.2, f64::INFINITY, 175.0, 1),
            Peak::triple("HNCACB", 8.2, 120.0, f64::NEG_INFINITY, -1),
            Peak::triple("HNCACO", 8.2, 120.0, 9999.0, 1).with_intensity(f64::NAN, None),
            Peak::triple("HNCO", 8.25, 120.5, 175.2, 1).with_intensity(f64::INFINITY, None),
        ],
        c13_dataset: C13PickedDataset {
            con_roots: vec![
                Con2DPeak {
                    c_prev_ppm: f64::NAN,
                    n_curr_ppm: 136.0,
                    height: 100.0,
                    volume: 100.0,
                    is_proline_imino_region: true,
                    deconvolved_from_cluster: false,
                },
                Con2DPeak {
                    c_prev_ppm: 174.0,
                    n_curr_ppm: 136.5,
                    height: 100.0,
                    volume: 100.0,
                    is_proline_imino_region: true,
                    deconvolved_from_cluster: false,
                },
            ],
            caco_peaks: Vec::new(),
            strip_peaks: vec![C13Strip3DPeak {
                exp: C13DetectedExp::Canco3D,
                c_prime_ppm: 174.0,
                n_ppm: 136.5,
                c_indirect_ppm: f64::NAN,
                sign: 1,
                height: f64::INFINITY,
                volume: 0.0,
                anchor_root_idx: Some(999),
            }],
        },
        cas_spectra: vec![CasSpectrum {
            name: "hostile_cas".to_string(),
            mode: CasMode::Cacon,
            residues: vec!["SER".to_string()],
            peaks: vec![CasPeak {
                c_prime_ppm: f64::NAN,
                n_ppm: f64::INFINITY,
                height: f64::NAN,
                volume: 0.0,
            }],
        }],
        hncacon_nn_edges: vec![DirectionalNnPeak {
            c_prev_ppm: f64::NAN,
            n_curr_ppm: 120.0,
            n_prev_ppm: f64::INFINITY,
            snr: f64::NAN,
        }],
        hncaconh_peaks: vec![HncaconhPeak {
            h_ppm: f64::NAN,
            n_curr_ppm: 120.0,
            n_partner_ppm: 118.0,
            height: f64::NEG_INFINITY,
        }],
        reciprocal_nn_pairs: vec![(f64::NAN, 120.0), (120.0, f64::INFINITY)],
        proton_mhz: f64::NAN,
        user_locks: vec![(999, 999), (0, 0)],
    };
    let seq_short: Vec<String> = ["ALA", "PRO", "SER", "GLY", "PRO", "VAL"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let r_hostile = assign_joint_idp(&hostile, &seq_short, &cfg).unwrap();
    assert!(r_hostile.score.is_finite());
    assert!(r_hostile.pass1_score.is_finite());
    assert!(r_hostile.pass2_score_gain.is_finite());
    assert!(r_hostile
        .confidence
        .iter()
        .all(|c| c.is_finite() && (0.0..=1.0).contains(c)));
    assert!(r_hostile
        .raw_posterior
        .iter()
        .all(|c| c.is_finite() && (0.0..=1.0).contains(c)));
    assert!(r_hostile.fitted_offsets.d_n.is_finite());
    assert!(r_hostile.fitted_offsets.d_c.is_finite());
    assert!(r_hostile.trosy_calibration.dh_ppm.is_finite());
    assert!(r_hostile.trosy_calibration.dn_ppm.is_finite());

    // 4. All-Proline sequence with HN-only amide root must cleanly abstain/unassign (`PRO` forbids `AmideFused`)
    let all_pro = vec!["PRO".to_string(); 8];
    let hn_on_pro = JointIdpDataset {
        hn_peaks: vec![Peak::triple("HNCO", 8.2, 120.0, 175.0, 1)],
        proton_mhz: 600.0,
        ..JointIdpDataset::default()
    };
    let r_all_pro = assign_joint_idp(&hn_on_pro, &all_pro, &cfg).unwrap();
    assert!(r_all_pro.position.iter().all(|&p| p == -1));

    // 5. Matrix allocation cap guard (`S * K > MAX_IDP_MATRIX_CELLS`) returns `Err` cleanly
    let huge_seq = vec!["ALA".to_string(); 6000];
    let many_nodes: Vec<JointIdpNode> = (0..5000)
        .map(|i| JointIdpNode {
            id: i,
            kind: JointNodeKind::ConOnlyAmide,
            hn_matched: false,
            con_matched: true,
            h_curr: None,
            n_curr: 120.0,
            hn_n_curr: None,
            c_prev: Some(175.0),
            root_height: 1.0,
            ca_curr: Vec::new(),
            cb_curr: Vec::new(),
            ca_prev: Vec::new(),
            cb_prev: Vec::new(),
            c_curr: Vec::new(),
            sidechain_c_prev: Vec::new(),
            nn_prev_cands: Vec::new(),
            recip_nn_partners: Vec::new(),
            prev_gly_opposite_phase: false,
            has_prev_cb_exp: false,
            has_curr_cb_exp: false,
            gly_cb_curr_suppressed: false,
            gly_cb_prev_suppressed: false,
            f1_sidelobes_pruned: 0,
            registration_warnings: Vec::new(),
        })
        .collect();
    assert!(many_nodes.len() * huge_seq.len() > MAX_IDP_MATRIX_CELLS);
    assert!(try_joint_idp_unary_matrix(
        &many_nodes,
        &huge_seq,
        &[],
        &cfg,
        &IdpShiftOffsets::default(),
        &[],
    )
    .is_err());

    // 6. Verify explicit `JointRouteSupport::HnRoute` on a clean HN-only mini-dataset
    let mini_seq: Vec<String> = ["MET", "ALA", "SER", "GLY", "VAL", "LEU"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let preds = predict_joint_ncidp_shifts(&mini_seq, &cfg);
    let mut mini_hn = Vec::new();
    for k in 1..mini_seq.len() {
        let h = 8.0 + (k as f64) * 0.1;
        let n = preds[k].n_ppm;
        mini_hn.push(Peak::triple("HNCO", h, n, preds[k - 1].c_ppm, 1).with_intensity(1.0e5, None));
        mini_hn.push(Peak::triple("HNCACO", h, n, preds[k].c_ppm, 1).with_intensity(8.0e4, None));
        mini_hn.push(Peak::triple("HNCACB", h, n, preds[k].ca_ppm, 1).with_intensity(8.0e4, None));
        if let Some(cb) = preds[k].cb_ppm {
            mini_hn.push(Peak::triple("HNCACB", h, n, cb, -1).with_intensity(7.0e4, None));
        }
        mini_hn.push(
            Peak::triple("CBCACONH", h, n, preds[k - 1].ca_ppm, 1).with_intensity(6.0e4, None),
        );
        if let Some(cb) = preds[k - 1].cb_ppm {
            mini_hn.push(Peak::triple("CBCACONH", h, n, cb, 1).with_intensity(5.0e4, None));
        }
    }
    let r_mini = assign_joint_idp(
        &JointIdpDataset {
            hn_peaks: mini_hn,
            proton_mhz: 600.0,
            ..JointIdpDataset::default()
        },
        &mini_seq,
        &JointIdpConfig {
            auto_reference_offset: false,
            hncacb_polarity: HncacbPolarity::CbNegative,
            ..cfg
        },
    )
    .unwrap();
    assert!(
        r_mini
            .route_support
            .contains(&Some(JointRouteSupport::HnRoute)),
        "Expected HnRoute tag on HN-only assignment"
    );
}

#[test]
fn test_two_pass_anchor_lock_propagates_and_exercises_pass2() {
    // 20-aa IDP with an unambiguous N-terminal Proline-bridged region (residues 1..12) and a
    // sparse/repetitive Ser-Gln tail (residues 12..20) where Pass 1 locks the unambiguous anchors
    // and Pass 2 re-anneals the remaining unlocked subset.
    let seq: Vec<String> = [
        "MET", "ALA", "GLY", "THR", "PRO", "VAL", "ILE", "SER", "PRO", "ASP", "LYS", "LEU", "SER",
        "GLN", "SER", "GLN", "SER", "GLN", "GLU", "ALA",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let cfg_base = JointIdpConfig {
        auto_reference_offset: false,
        replicas: 16,
        islands: 2,
        generations: 2,
        steps_per_system: 600,
        seed: 123,
        ..JointIdpConfig::default()
    };
    let preds = predict_joint_ncidp_shifts(&seq, &cfg_base);

    let mut hn_peaks = Vec::new();
    let mut con_roots = Vec::new();
    let mut strip_peaks = Vec::new();
    let mut nn_edges = Vec::new();

    for k in 1..seq.len() {
        let r_idx = con_roots.len();
        con_roots.push(Con2DPeak {
            c_prev_ppm: preds[k - 1].c_ppm,
            n_curr_ppm: preds[k].n_ppm,
            height: 8.0e4,
            volume: 8.0e4,
            is_proline_imino_region: seq[k] == "PRO",
            deconvolved_from_cluster: false,
        });
        if k <= 11 {
            // Strong multi-channel region for Pass-1 locking (including Pro4 & Pro8 bridges!)
            strip_peaks.push(C13Strip3DPeak {
                exp: C13DetectedExp::Canco3D,
                c_prime_ppm: preds[k - 1].c_ppm,
                n_ppm: preds[k].n_ppm,
                c_indirect_ppm: preds[k - 1].ca_ppm,
                sign: 1,
                height: 6.0e4,
                volume: 6.0e4,
                anchor_root_idx: Some(r_idx),
            });
            strip_peaks.push(C13Strip3DPeak {
                exp: C13DetectedExp::Canco3D,
                c_prime_ppm: preds[k - 1].c_ppm,
                n_ppm: preds[k].n_ppm,
                c_indirect_ppm: preds[k].ca_ppm,
                sign: 1,
                height: 5.0e4,
                volume: 5.0e4,
                anchor_root_idx: Some(r_idx),
            });
            if k >= 2 {
                nn_edges.push(DirectionalNnPeak {
                    c_prev_ppm: preds[k - 1].c_ppm,
                    n_curr_ppm: preds[k].n_ppm,
                    n_prev_ppm: preds[k - 1].n_ppm,
                    snr: 8.0,
                });
            }
        }
        if seq[k] != "PRO" {
            let h = 7.9 + (k as f64) * 0.05;
            let n = preds[k].n_ppm;
            hn_peaks.push(
                Peak::triple("HNCO", h, n, preds[k - 1].c_ppm, 1).with_intensity(9.0e4, None),
            );
            if k <= 11 {
                hn_peaks.push(
                    Peak::triple("HNCACO", h, n, preds[k].c_ppm, 1).with_intensity(7.0e4, None),
                );
                hn_peaks.push(
                    Peak::triple("HNCACB", h, n, preds[k].ca_ppm, 1).with_intensity(7.5e4, None),
                );
                if let Some(cb) = preds[k].cb_ppm {
                    hn_peaks.push(Peak::triple("HNCACB", h, n, cb, -1).with_intensity(6.5e4, None));
                }
                hn_peaks.push(
                    Peak::triple("CBCACONH", h, n, preds[k - 1].ca_ppm, 1)
                        .with_intensity(5.5e4, None),
                );
                if let Some(cb) = preds[k - 1].cb_ppm {
                    hn_peaks
                        .push(Peak::triple("CBCACONH", h, n, cb, 1).with_intensity(5.0e4, None));
                }
            }
        }
    }

    let ds = JointIdpDataset {
        hn_peaks,
        c13_dataset: C13PickedDataset {
            con_roots,
            caco_peaks: Vec::new(),
            strip_peaks,
        },
        hncacon_nn_edges: nn_edges,
        proton_mhz: 600.0,
        ..JointIdpDataset::default()
    };

    let res = assign_joint_idp(
        &ds,
        &seq,
        &JointIdpConfig {
            enable_two_pass_anchor_lock: true,
            hncacb_polarity: HncacbPolarity::CbNegative,
            ..cfg_base
        },
    )
    .expect("2-pass solve");

    assert!(
        !res.pass1_locked_anchors.is_empty() && res.pass1_locked_anchors.len() < seq.len() - 1,
        "Expected partial Pass-1 anchor locking to trigger Pass 2 (locked {} of {})",
        res.pass1_locked_anchors.len(),
        seq.len() - 1
    );
    assert!(
        res.score >= res.pass1_score - 1e-6,
        "Pass-2 score ({:.2}) must not regress Pass-1 ({:.2})",
        res.score,
        res.pass1_score
    );
    assert_eq!(res.proline_bridge_recovery, (2, 2));
}
