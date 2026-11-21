//! Clean-room unit tests, multi-seed adversarial calibration/QC grid, real-solver baseline
//! comparison, and `NMR_REAL_DATA_DIR`-gated Clec16a replay for the unified joint
//! `HN + 2D CON + Proline-Bridge + N–N + CAS` backbone solver (`T23.2`, `v0.18.0`).
//!
//! All synthetic benchmarks attach `3D` strip peaks strictly via physical `(C', N)` coordinates
//! (`anchor_root_idx: None`) with realistic jitter, random peak dropout, and spurious noise roots.

use std::collections::BTreeMap;
use std::path::PathBuf;

use nmr_core::c13_detected::{
    n15_obs_mhz, pick_con_2d, C13DetectedExp, C13PickedDataset, C13PickerConfig, C13Strip3DPeak,
    CasMode, CasPeak, CasSpectrum, Con2DPeak,
};
use nmr_core::evo_graph::{assign_evo_graph_from_peaks, EvoGraphConfig};
use nmr_core::idp_solver::{
    register_and_build_con_dipeptide_systems, try_assign_idp_13c_cpu, IdpShiftOffsets,
    IdpSolverConfig, MAX_IDP_MATRIX_CELLS,
};
use nmr_core::joint_idp_solver::{
    apply_gly_cb_bleedthrough_guard, assign_joint_idp, assign_joint_idp_cpu,
    build_calibrated_cas_tables, calibrate_joint_posterior, calibrate_trosy_hn_offset,
    joint_idp_to_reference, joint_idp_unary_matrix, motif_repeat_metrics,
    pre_pro_shift_corrections, predict_joint_ncidp_shifts, prune_f1_c_prime_nus_sidelobes,
    trosy_expected_shift_ppm, try_joint_idp_unary_matrix, DirectionalNnPeak, HncaconhPeak,
    JointIdpAssignResult, JointIdpConfig, JointIdpDataset, JointIdpNode, JointNodeKind,
    JointRouteSupport, CANONICAL_TROSY_FALLBACK_MHZ, JOINT_IDP_CALIBRATION_KNOTS,
    MIN_TROSY_MATCHED_ROOTS, PRE_PRO_CA_CORRECTION_PPM, PRE_PRO_CB_CORRECTION_PPM,
    PRE_PRO_C_PRIME_CORRECTION_PPM, PRE_PRO_N_CORRECTION_PPM, PROLINE_N15_MAX_PPM,
    PROLINE_N15_MIN_PPM, TROSY_HALF_J_HN_HZ,
};
use nmr_core::peaks::Peak;
use nmr_core::picker::{pick_hsqc_2d, pick_triple_3d, PickerConfig};
use nmr_core::priors::{one_to_three, Priors};
use nmr_core::qc::DatasetQcVerdict;
use nmr_core::rng::Rng;
use nmr_core::spectrum::{read_bruker_pdata_2d, read_bruker_pdata_3d};
use nmr_core::spinsystems::HncacbPolarity;

/// Held-out 140-aa Pro/Ser-rich disordered sequence (`15.0% Pro` = 21 Pro, including `-S-P-`,
/// `-G-P-`, `-P-P-` motifs, Ser/Gly stretches, and a non-Pro C-terminus).
fn synthetic_pro_ser_rich_idp_sequence() -> Vec<String> {
    let seq_1 = concat!(
        "MSGAPQSTPV", // 0..10  (P4, P8)
        "KLDPNSRGAF", // 10..20 (P13)
        "QTPVSLENKP", // 20..30 (P22, P29)
        "PGSVTDRQIA", // 30..40 (P30 -> di-Pro 29-30)
        "HSMGPLKVSE", // 40..50 (P44)
        "YQNPTASPLR", // 50..60 (P53, P57)
        "DFSVPKQMET", // 60..70 (P64)
        "GSPHLAVNSP", // 70..80 (P72, P79)
        "RKEGTPSQID", // 80..90 (P85)
        "VNSALPGFYT", // 90..100 (P95)
        "QSPKRMGSVP", // 100..110 (P102, P109)
        "PAENHTSLDG", // 110..120 (P110 -> di-Pro 109-110)
        "KPVSQAFPRN", // 120..130 (P121, P127)
        "TSPLVEPKSA"  // 130..140 (P132, P136, C-term ALA139)
    );
    let seq: Vec<String> = seq_1.chars().map(|c| one_to_three(c).to_string()).collect();
    assert_eq!(seq.len(), 140);
    assert_eq!(seq.iter().filter(|r| r.as_str() == "PRO").count(), 21);
    seq
}

/// Low-complexity Ser/Pro/Gln repeat IDP sequence (120 aa) for `180.core:P1-2` QC guard testing.
fn synthetic_low_complexity_repeat_idp_sequence() -> Vec<String> {
    let seq_1 = concat!(
        "MSPQSPQSPQ",
        "SSSPPPQQQS",
        "PQSPQSPQSP",
        "QSSSPPPQQQ",
        "SPQSPQSPQS",
        "PQSSSPPPQQ",
        "QSPQSPQSPQ",
        "SPQSSSPPPQ",
        "QQSPQSPQSP",
        "QSPQSSSPPP",
        "QQQSPQSPQS",
        "PQSSSPQQSA"
    );
    let seq: Vec<String> = seq_1.chars().map(|c| one_to_three(c).to_string()).collect();
    assert_eq!(seq.len(), 120);
    seq
}

/// Disjoint calibration sequence A (115 aa, NOT used in held-out adversarial assertions).
fn disjoint_calibration_seq_a() -> Vec<String> {
    let seq_1 = concat!(
        "MGEKPTSQAV",
        "LNDPRSFGKT",
        "HPVQEMSNPA",
        "DLRKGTYPVS",
        "EQANPFHKSD",
        "MTGPALVNRQ",
        "FPSKEIGTDL",
        "ANPVQSRHME",
        "KTPGDLSFVA",
        "NRPQETSKMG",
        "ILHPDAVNRS",
        "FQPTA"
    );
    seq_1.chars().map(|c| one_to_three(c).to_string()).collect()
}

/// Disjoint calibration sequence B (130 aa, NOT used in held-out adversarial assertions).
fn disjoint_calibration_seq_b() -> Vec<String> {
    let seq_1 = concat!(
        "MASTPLGKQD",
        "NVREPSAFHT",
        "IKLPGQSMED",
        "VRANPFTKSL",
        "QHGDPVEMAR",
        "TKSNPILFQG",
        "ADEPVSTRKH",
        "LMNPQGSFAD",
        "EVRPTKILNS",
        "QGHPAFMDRE",
        "SKTPVGQLAN",
        "DFRPEHSMKT",
        "GVQPASLNET"
    );
    seq_1.chars().map(|c| one_to_three(c).to_string()).collect()
}

/// Generate a fresh deterministic Pro/Ser/Gly/Glu-rich disordered sequence of length `len`
/// from `seq_seed` (for non-circular OOF predictor-bias calibration and testing).
fn generate_random_idp_sequence(len: usize, seq_seed: u64) -> Vec<String> {
    // Realistic IDP composition (~13% PRO, ~13% SER, ~10% GLY, ~10% GLU, ~9% ALA, ~8% GLN, ~7% LYS, etc.)
    const IDP_ALPHABET: &[char] = &[
        'P', 'P', 'P', 'P', 'P', 'P', 'S', 'S', 'S', 'S', 'S', 'S', 'G', 'G', 'G', 'G', 'E', 'E',
        'E', 'E', 'A', 'A', 'A', 'A', 'Q', 'Q', 'Q', 'K', 'K', 'K', 'T', 'T', 'T', 'D', 'D', 'D',
        'N', 'N', 'V', 'V', 'L', 'L', 'R', 'R', 'I', 'H', 'M', 'F', 'Y',
    ];
    let mut rng = Rng::new(seq_seed);
    let mut out = Vec::with_capacity(len);
    out.push("MET".to_string());
    for i in 1..len {
        let mut ch = IDP_ALPHABET[rng.usize(IDP_ALPHABET.len())];
        // Avoid accidental C-terminal Proline so C-term COO- is observable
        if i + 1 == len && ch == 'P' {
            ch = 'A';
        }
        out.push(one_to_three(ch).to_string());
    }
    out
}

/// Generate a realistic 9-mer motif-repeat IDP sequence of length `len` with `25–35%` random
/// point mutations (`180r3.core:P2-2`: e.g. degenerate tandem repeats like CTD/FUS/tau motifs).
fn generate_motif_repeat_idp_sequence(len: usize, seq_seed: u64) -> Vec<String> {
    const MOTIF: &[char] = &['S', 'P', 'T', 'S', 'P', 'S', 'Y', 'G', 'Q']; // 9-mer repeat
    const MUT_ALPHABET: &[char] = &[
        'A', 'E', 'D', 'K', 'N', 'T', 'V', 'L', 'R', 'H', 'G', 'S', 'Q',
    ];
    let mut rng = Rng::new(seq_seed);
    let mut out = Vec::with_capacity(len);
    out.push("MET".to_string());
    for i in 1..len {
        let mut ch = MOTIF[(i - 1) % MOTIF.len()];
        // ~30% point mutation rate across copies (within 25–35%)
        if rng.f64() < 0.30 {
            let mut m = MUT_ALPHABET[rng.usize(MUT_ALPHABET.len())];
            if m == ch {
                m = MUT_ALPHABET[(rng.usize(MUT_ALPHABET.len()) + 1) % MUT_ALPHABET.len()];
            }
            ch = m;
        }
        if i + 1 == len && ch == 'P' {
            ch = 'A';
        }
        out.push(one_to_three(ch).to_string());
    }
    out
}

#[derive(Clone, Debug)]
struct IdpStressSpec {
    jitter_h: f64,
    jitter_n: f64,
    jitter_c: f64,
    rc_dev_scale: f64,
    /// Systematic per-residue-type predictor bias amplitude on `15N` (ppm).
    predictor_bias_n: f64,
    /// Systematic per-residue-type predictor bias amplitude on `13C'`, `13CA`, `13CB` (ppm).
    predictor_bias_c: f64,
    /// Transient helix amplitude on `k in [30, 50) U [85, 103)` (`1.0` = `CA+1.2, C'+0.9, CB-0.3, N-0.6`).
    helix_amp: f64,
    /// Systematic neighbor-dependent bias amplitude keyed on `seq[k-1]` for `15N` (ppm).
    neighbor_bias_n: f64,
    /// Systematic neighbor-dependent bias amplitude keyed on `seq[k-1]` for `13C'/CA/CB` (ppm).
    neighbor_bias_c: f64,
    random_drop_frac: f64,
    n_decoy_con_roots: usize,
    n_decoy_hn_roots: usize,
    tms_c13_offset: f64,
    invert_hncacb: bool,
    include_trosy_hncaconh: bool,
    include_cas: bool,
    exchange_broaden: bool,
}

impl Default for IdpStressSpec {
    fn default() -> Self {
        Self {
            jitter_h: 0.006,
            jitter_n: 0.04,
            jitter_c: 0.018,
            rc_dev_scale: 1.0,
            predictor_bias_n: 0.0,
            predictor_bias_c: 0.0,
            helix_amp: 0.0,
            neighbor_bias_n: 0.0,
            neighbor_bias_c: 0.0,
            random_drop_frac: 0.08,
            n_decoy_con_roots: 15,
            n_decoy_hn_roots: 6,
            tms_c13_offset: -2.70,
            invert_hncacb: true,
            include_trosy_hncaconh: true,
            include_cas: true,
            exchange_broaden: true,
        }
    }
}

type ResidueTruthShifts = (Option<f64>, f64, f64, f64, Option<f64>, Vec<f64>);

#[derive(Clone)]
struct SyntheticGroundTruth {
    seq: Vec<String>,
    /// Ground-truth DSS shifts per residue `k`: `(H_opt, N, C', CA, CB_opt, sidechain_c)`.
    shifts: Vec<ResidueTruthShifts>,
    /// True if residue `k` is exchange-broadened (missing from all `1H`-detected spectra).
    exchange_broadened: Vec<bool>,
    /// True synthetic uniform `13C` referencing offset applied to generated spectra (e.g. `-2.70` ppm TMS).
    tms_c13_offset: f64,
}

/// Synthesize a realistic, clean-room IDP dataset under `IdpStressSpec` with ZERO ground-truth
/// `anchor_root_idx` (`anchor_root_idx: None` on every 3D strip peak).
fn synthesize_idp_stress_dataset(
    seq: &[String],
    seed: u64,
    spec: &IdpStressSpec,
) -> (JointIdpDataset, SyntheticGroundTruth) {
    let mut rng = Rng::new(seed);
    let len = seq.len();
    let cfg_pred = JointIdpConfig::default();
    let preds = predict_joint_ncidp_shifts(seq, &cfg_pred);

    let all_aas = [
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET",
        "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    ];
    let mut aa_bias: BTreeMap<String, (f64, f64, f64, f64)> = BTreeMap::new();
    if spec.predictor_bias_n > 0.0 || spec.predictor_bias_c > 0.0 {
        let mut bias_rng = Rng::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xBEEF_CAFE);
        for aa in all_aas {
            let bn = if spec.predictor_bias_n > 0.0 {
                bias_rng.uniform(-spec.predictor_bias_n, spec.predictor_bias_n)
            } else {
                0.0
            };
            let bc = if spec.predictor_bias_c > 0.0 {
                bias_rng.uniform(-spec.predictor_bias_c, spec.predictor_bias_c)
            } else {
                0.0
            };
            let bca = if spec.predictor_bias_c > 0.0 {
                bias_rng.uniform(-spec.predictor_bias_c, spec.predictor_bias_c)
            } else {
                0.0
            };
            let bcb = if spec.predictor_bias_c > 0.0 {
                bias_rng.uniform(-spec.predictor_bias_c, spec.predictor_bias_c)
            } else {
                0.0
            };
            aa_bias.insert(aa.to_string(), (bn, bc, bca, bcb));
        }
    }

    let mut prev_nb_bias: BTreeMap<String, (f64, f64, f64, f64)> = BTreeMap::new();
    if spec.neighbor_bias_n > 0.0 || spec.neighbor_bias_c > 0.0 {
        let mut nb_rng = Rng::new(seed.wrapping_mul(0xBF58_476D_1CE4_E5B9) ^ 0xCAFE_BABE);
        for aa in all_aas {
            let bn = if spec.neighbor_bias_n > 0.0 {
                nb_rng.uniform(-spec.neighbor_bias_n, spec.neighbor_bias_n)
            } else {
                0.0
            };
            let bc = if spec.neighbor_bias_c > 0.0 {
                nb_rng.uniform(-spec.neighbor_bias_c, spec.neighbor_bias_c)
            } else {
                0.0
            };
            let bca = if spec.neighbor_bias_c > 0.0 {
                nb_rng.uniform(-spec.neighbor_bias_c, spec.neighbor_bias_c)
            } else {
                0.0
            };
            let bcb = if spec.neighbor_bias_c > 0.0 {
                nb_rng.uniform(-spec.neighbor_bias_c, spec.neighbor_bias_c)
            } else {
                0.0
            };
            prev_nb_bias.insert(aa.to_string(), (bn, bc, bca, bcb));
        }
    }

    let proton_mhz = 800.0_f64;
    let trosy_dh = -TROSY_HALF_J_HN_HZ / proton_mhz;
    let trosy_dn = TROSY_HALF_J_HN_HZ / n15_obs_mhz(proton_mhz);

    // 1. Ground-truth DSS backbone shifts with coil spread + per-AA + segmental helix + neighbor bias
    let mut shifts = Vec::with_capacity(len);
    for k in 0..len {
        let p = &preds[k];
        let is_pro = seq[k] == "PRO";
        let (bn, bc, bca, bcb) = aa_bias
            .get(&seq[k])
            .copied()
            .unwrap_or((0.0, 0.0, 0.0, 0.0));
        let (nbn, nbc, nbca, nbcb) = if k >= 1 {
            prev_nb_bias
                .get(&seq[k - 1])
                .copied()
                .unwrap_or((0.0, 0.0, 0.0, 0.0))
        } else {
            (0.0, 0.0, 0.0, 0.0)
        };
        let in_helix = spec.helix_amp != 0.0 && ((30..50).contains(&k) || (85..103).contains(&k));
        let (hn_s, hc_s, hca_s, hcb_s) = if in_helix {
            (
                -0.6 * spec.helix_amp,
                0.9 * spec.helix_amp,
                1.2 * spec.helix_amp,
                -0.3 * spec.helix_amp,
            )
        } else {
            (0.0, 0.0, 0.0, 0.0)
        };

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
            Some(base_h + rng.normal(0.0, 0.22 * spec.rc_dev_scale))
        };
        let n = p.n_ppm
            + bn
            + nbn
            + hn_s
            + rng.normal(0.0, if is_pro { 0.55 } else { 0.48 } * spec.rc_dev_scale);
        let c = if k + 1 == len {
            179.25 + rng.normal(0.0, 0.16 * spec.rc_dev_scale.min(1.5))
        } else {
            p.c_ppm + bc + nbc + hc_s + rng.normal(0.0, 0.25 * spec.rc_dev_scale)
        };
        let ca = p.ca_ppm + bca + nbca + hca_s + rng.normal(0.0, 0.26 * spec.rc_dev_scale);
        let cb = p
            .cb_ppm
            .map(|m| m + bcb + nbcb + hcb_s + rng.normal(0.0, 0.30 * spec.rc_dev_scale));
        let sc: Vec<f64> = p
            .sidechain_c
            .iter()
            .map(|&m| m + rng.normal(0.0, 0.28 * spec.rc_dev_scale))
            .collect();
        shifts.push((h, n, c, ca, cb, sc));
    }

    // 2. Mark ~14% of non-Proline residues (excluding C-term) as exchange-broadened in `HN`
    let mut exchange_broadened = vec![false; len];
    if spec.exchange_broaden {
        for k in 2..(len.saturating_sub(1)) {
            if seq[k] != "PRO" && (k % 9 == 4 || k % 17 == 11) {
                exchange_broadened[k] = true;
            }
        }
    }

    let ca_sign = if spec.invert_hncacb { -1 } else { 1 };
    let cb_sign = -ca_sign;

    // 3. Build `HN` peaks (`HSQC`, `HNCO`, `HNCACO`, `HNCACB`, `CBCACONH`, `BT_HNCACONH`)
    let mut hn_peaks: Vec<Peak> = Vec::new();
    let mut hncaconh_peaks: Vec<HncaconhPeak> = Vec::new();

    for k in 1..len {
        if seq[k] == "PRO" || exchange_broadened[k] {
            continue;
        }
        // Independent random HN root dropout (except C-term)
        if k + 1 < len && rng.f64() < spec.random_drop_frac * 0.6 {
            continue;
        }
        let h = shifts[k].0.unwrap();
        let n = shifts[k].1;
        let c_prev_true = shifts[k - 1].2 + spec.tms_c13_offset;
        let c_curr_true = shifts[k].2 + spec.tms_c13_offset;
        let ca_curr_true = shifts[k].3 + spec.tms_c13_offset;
        let ca_prev_true = shifts[k - 1].3 + spec.tms_c13_offset;
        let cb_curr_true = shifts[k].4.map(|v| v + spec.tms_c13_offset);
        let cb_prev_true = shifts[k - 1].4.map(|v| v + spec.tms_c13_offset);

        let h_hsqc = h + rng.normal(0.0, spec.jitter_h * 0.5);
        let n_hsqc = n + rng.normal(0.0, spec.jitter_n * 0.5);
        hn_peaks.push(Peak::hsqc(h_hsqc, n_hsqc).with_intensity(1.0e5, None));

        // HNCO
        if rng.f64() >= spec.random_drop_frac * 0.5 {
            hn_peaks.push(
                Peak::triple(
                    "HNCO",
                    h + rng.normal(0.0, spec.jitter_h * 0.6),
                    n + rng.normal(0.0, spec.jitter_n * 0.6),
                    c_prev_true + rng.normal(0.0, spec.jitter_c),
                    1,
                )
                .with_intensity(9.5e4, None),
            );
        }

        // HNCACO (`C'_i` & `C'_{i-1}`) + realistic symmetric NUS sidelobes
        if rng.f64() >= spec.random_drop_frac {
            let c_i = c_curr_true + rng.normal(0.0, spec.jitter_c);
            hn_peaks.push(
                Peak::triple(
                    "HNCACO",
                    h + rng.normal(0.0, spec.jitter_h * 0.7),
                    n + rng.normal(0.0, spec.jitter_n * 0.7),
                    c_i,
                    1,
                )
                .with_intensity(7.5e4, None),
            );
            if rng.f64() >= spec.random_drop_frac * 0.8 {
                hn_peaks.push(
                    Peak::triple(
                        "HNCACO",
                        h + rng.normal(0.0, spec.jitter_h * 0.7),
                        n + rng.normal(0.0, spec.jitter_n * 0.7),
                        c_prev_true + rng.normal(0.0, spec.jitter_c),
                        1,
                    )
                    .with_intensity(5.0e4, None),
                );
            }
            if k % 4 == 0 {
                // Symmetric weak F1(C') NUS sidelobe pair around C'_i
                hn_peaks
                    .push(Peak::triple("HNCACO", h, n, c_i + 0.92, 1).with_intensity(7.8e3, None));
                hn_peaks
                    .push(Peak::triple("HNCACO", h, n, c_i - 0.92, 1).with_intensity(7.5e3, None));
            }
        }

        // HNCACB
        if rng.f64() >= spec.random_drop_frac {
            let ca_i = ca_curr_true + rng.normal(0.0, spec.jitter_c * 1.15);
            hn_peaks.push(
                Peak::triple(
                    "HNCACB",
                    h + rng.normal(0.0, spec.jitter_h * 0.7),
                    n + rng.normal(0.0, spec.jitter_n * 0.7),
                    ca_i,
                    ca_sign,
                )
                .with_intensity(8.0e4, None),
            );
            if rng.f64() >= spec.random_drop_frac {
                hn_peaks.push(
                    Peak::triple(
                        "HNCACB",
                        h + rng.normal(0.0, spec.jitter_h * 0.7),
                        n + rng.normal(0.0, spec.jitter_n * 0.7),
                        ca_prev_true + rng.normal(0.0, spec.jitter_c * 1.2),
                        ca_sign,
                    )
                    .with_intensity(4.8e4, None),
                );
            }
            if let Some(cb) = cb_curr_true {
                if rng.f64() >= spec.random_drop_frac {
                    hn_peaks.push(
                        Peak::triple(
                            "HNCACB",
                            h + rng.normal(0.0, spec.jitter_h * 0.7),
                            n + rng.normal(0.0, spec.jitter_n * 0.7),
                            cb + rng.normal(0.0, spec.jitter_c * 1.25),
                            cb_sign,
                        )
                        .with_intensity(6.5e4, None),
                    );
                }
            } else if seq[k] == "GLY" {
                hn_peaks.push(
                    Peak::triple("HNCACB", h, n, ca_i + 0.28, cb_sign).with_intensity(1.3e4, None),
                );
            }
            if let Some(cb) = cb_prev_true {
                if rng.f64() >= spec.random_drop_frac {
                    hn_peaks.push(
                        Peak::triple(
                            "HNCACB",
                            h + rng.normal(0.0, spec.jitter_h * 0.7),
                            n + rng.normal(0.0, spec.jitter_n * 0.7),
                            cb + rng.normal(0.0, spec.jitter_c * 1.3),
                            cb_sign,
                        )
                        .with_intensity(4.2e4, None),
                    );
                }
            }
        }

        // CBCACONH (`i - 1` CA/CB)
        if rng.f64() >= spec.random_drop_frac {
            hn_peaks.push(
                Peak::triple(
                    "CBCACONH",
                    h + rng.normal(0.0, spec.jitter_h * 0.7),
                    n + rng.normal(0.0, spec.jitter_n * 0.7),
                    ca_prev_true + rng.normal(0.0, spec.jitter_c * 1.1),
                    1,
                )
                .with_intensity(6.0e4, None),
            );
            if let Some(cb) = cb_prev_true {
                if rng.f64() >= spec.random_drop_frac {
                    hn_peaks.push(
                        Peak::triple(
                            "CBCACONH",
                            h + rng.normal(0.0, spec.jitter_h * 0.7),
                            n + rng.normal(0.0, spec.jitter_n * 0.7),
                            cb + rng.normal(0.0, spec.jitter_c * 1.2),
                            1,
                        )
                        .with_intensity(5.5e4, None),
                    );
                }
            }
        }

        // BT_HNCACONH with 800 MHz TROSY offset
        if spec.include_trosy_hncaconh {
            if k >= 2
                && seq[k - 1] != "PRO"
                && !exchange_broadened[k - 1]
                && rng.f64() >= spec.random_drop_frac
            {
                hncaconh_peaks.push(HncaconhPeak {
                    h_ppm: h + trosy_dh + rng.normal(0.0, spec.jitter_h * 0.7),
                    n_curr_ppm: n + trosy_dn + rng.normal(0.0, spec.jitter_n * 0.7),
                    n_partner_ppm: shifts[k - 1].1 + rng.normal(0.0, spec.jitter_n * 0.8),
                    height: 4.5e4,
                });
            }
            if k + 1 < len
                && seq[k + 1] != "PRO"
                && !exchange_broadened[k + 1]
                && rng.f64() >= spec.random_drop_frac
            {
                hncaconh_peaks.push(HncaconhPeak {
                    h_ppm: h + trosy_dh + rng.normal(0.0, spec.jitter_h * 0.7),
                    n_curr_ppm: n + trosy_dn + rng.normal(0.0, spec.jitter_n * 0.7),
                    n_partner_ppm: shifts[k + 1].1 + rng.normal(0.0, spec.jitter_n * 0.8),
                    height: 4.2e4,
                });
            }
        }
    }

    // Spurious HN decoy roots (HSQC + HNCO noise)
    for _ in 0..spec.n_decoy_hn_roots {
        let h_d = 7.75 + rng.f64() * 1.15;
        let n_d = 109.0 + rng.f64() * 18.0;
        let c_d = 171.5 + rng.f64() * 6.5 + spec.tms_c13_offset;
        hn_peaks.push(Peak::hsqc(h_d, n_d).with_intensity(1.8e4, None));
        hn_peaks.push(Peak::triple("HNCO", h_d, n_d, c_d, 1).with_intensity(1.6e4, None));
    }

    // 4. Build 13C-detected `2D CON` + `CBCACON`/`H(CC)CON` + `CANCO` + `(H)N(CA)CON` + `CAS`
    //    CRITICAL (`180.core:P1-3`): `anchor_root_idx: None` on EVERY 3D strip peak!
    let mut con_roots: Vec<Con2DPeak> = Vec::new();
    let mut strip_peaks: Vec<C13Strip3DPeak> = Vec::new();
    let mut hncacon_nn_edges: Vec<DirectionalNnPeak> = Vec::new();

    for k in 1..len {
        let natural_drop =
            seq[k] != "PRO" && !exchange_broadened[k] && k + 1 < len && (k % 11 == 3);
        if natural_drop || (k + 1 < len && rng.f64() < spec.random_drop_frac * 0.5) {
            continue;
        }
        let c_prev_obs = shifts[k - 1].2 + spec.tms_c13_offset + rng.normal(0.0, spec.jitter_c);
        let n_curr_obs = shifts[k].1 + rng.normal(0.0, spec.jitter_n);
        con_roots.push(Con2DPeak {
            c_prev_ppm: c_prev_obs,
            n_curr_ppm: n_curr_obs,
            height: 8.5e4,
            volume: 8.5e4,
            is_proline_imino_region: n_curr_obs >= PROLINE_N15_MIN_PPM,
            deconvolved_from_cluster: false,
        });

        // CBCACON / H(CC)CON (independent coordinate jitter, NO anchor_root_idx!)
        if (seq[k] == "PRO" || exchange_broadened[k] || k % 4 != 1)
            && rng.f64() >= spec.random_drop_frac
        {
            let c_3d = c_prev_obs + rng.normal(0.0, spec.jitter_c * 0.65);
            let n_3d = n_curr_obs + rng.normal(0.0, spec.jitter_n * 0.65);
            strip_peaks.push(C13Strip3DPeak {
                exp: C13DetectedExp::Cbcaco3D,
                c_prime_ppm: c_3d,
                n_ppm: n_3d,
                c_indirect_ppm: shifts[k - 1].3
                    + spec.tms_c13_offset
                    + rng.normal(0.0, spec.jitter_c * 1.2),
                sign: 1,
                height: 5.5e4,
                volume: 5.5e4,
                anchor_root_idx: None,
            });
            if let Some(cb) = shifts[k - 1].4 {
                if rng.f64() >= spec.random_drop_frac {
                    strip_peaks.push(C13Strip3DPeak {
                        exp: C13DetectedExp::Cbcaco3D,
                        c_prime_ppm: c_3d,
                        n_ppm: n_3d,
                        c_indirect_ppm: cb
                            + spec.tms_c13_offset
                            + rng.normal(0.0, spec.jitter_c * 1.3),
                        sign: -1,
                        height: 4.5e4,
                        volume: 4.5e4,
                        anchor_root_idx: None,
                    });
                }
            }
            for &sc in &shifts[k - 1].5 {
                if rng.f64() >= spec.random_drop_frac {
                    strip_peaks.push(C13Strip3DPeak {
                        exp: C13DetectedExp::Cccon3D,
                        c_prime_ppm: c_3d,
                        n_ppm: n_3d,
                        c_indirect_ppm: sc
                            + spec.tms_c13_offset
                            + rng.normal(0.0, spec.jitter_c * 1.3),
                        sign: 1,
                        height: 3.5e4,
                        volume: 3.5e4,
                        anchor_root_idx: None,
                    });
                }
            }
        }

        // CANCO on Prolines & exchange-broadened nodes (and 30% of other nodes), NO anchor_root_idx!
        if (seq[k] == "PRO" || exchange_broadened[k] || k % 3 == 0)
            && rng.f64() >= spec.random_drop_frac
        {
            let c_3d = c_prev_obs + rng.normal(0.0, spec.jitter_c * 0.65);
            let n_3d = n_curr_obs + rng.normal(0.0, spec.jitter_n * 0.65);
            strip_peaks.push(C13Strip3DPeak {
                exp: C13DetectedExp::Canco3D,
                c_prime_ppm: c_3d,
                n_ppm: n_3d,
                c_indirect_ppm: shifts[k - 1].3
                    + spec.tms_c13_offset
                    + rng.normal(0.0, spec.jitter_c * 1.15),
                sign: 1,
                height: 6.0e4,
                volume: 6.0e4,
                anchor_root_idx: None,
            });
            strip_peaks.push(C13Strip3DPeak {
                exp: C13DetectedExp::Canco3D,
                c_prime_ppm: c_3d,
                n_ppm: n_3d,
                c_indirect_ppm: shifts[k].3
                    + spec.tms_c13_offset
                    + rng.normal(0.0, spec.jitter_c * 1.25),
                sign: 1,
                height: 4.6e4,
                volume: 4.6e4,
                anchor_root_idx: None,
            });
        }

        // Directional `(H)N(CA)CON` N–N links (critical across Proline & exchange-broadened gaps)
        let pro_or_exch_adj = seq[k] == "PRO"
            || seq[k - 1] == "PRO"
            || exchange_broadened[k]
            || (k >= 2 && exchange_broadened[k - 1]);
        if k >= 2 && (pro_or_exch_adj || k % 2 == 0) && rng.f64() >= spec.random_drop_frac {
            hncacon_nn_edges.push(DirectionalNnPeak {
                c_prev_ppm: c_prev_obs + rng.normal(0.0, spec.jitter_c * 0.7),
                n_curr_ppm: n_curr_obs + rng.normal(0.0, spec.jitter_n * 0.7),
                n_prev_ppm: shifts[k - 1].1 + rng.normal(0.0, spec.jitter_n * 0.8),
                snr: 18.0,
            });
        }
    }

    // Spurious noise `CON` decoy roots (including some in the Proline imino region!)
    for d in 0..spec.n_decoy_con_roots {
        let is_pro_decoy = d % 7 == 0;
        let c_d = 171.0 + rng.f64() * 7.5 + spec.tms_c13_offset;
        let n_d = if is_pro_decoy {
            133.0 + rng.f64() * 9.0
        } else {
            107.0 + rng.f64() * 22.0
        };
        con_roots.push(Con2DPeak {
            c_prev_ppm: c_d,
            n_curr_ppm: n_d,
            height: 1.5e4 + rng.f64() * 1.2e4,
            volume: 1.5e4,
            is_proline_imino_region: n_d >= PROLINE_N15_MIN_PPM,
            deconvolved_from_cluster: false,
        });
    }

    // 5. `CAS` subspectra (`cacon` on `i-1`, `canco` on `i`) with realistic ~6% breakthrough peaks
    let mut cas_spectra: Vec<CasSpectrum> = Vec::new();
    if spec.include_cas {
        let cas_defs: &[(&str, CasMode, &[&str])] = &[
            ("cacon_ser", CasMode::Cacon, &["SER"]),
            ("canco_ser", CasMode::Canco, &["SER"]),
            ("cacon_ala", CasMode::Cacon, &["ALA"]),
            ("cacon_gly", CasMode::Cacon, &["GLY"]),
            ("canco_gly", CasMode::Canco, &["GLY"]),
            ("cacon_tavi", CasMode::Cacon, &["THR", "ALA", "VAL", "ILE"]),
        ];
        for &(label, mode, residues) in cas_defs {
            let mut peaks = Vec::new();
            for k in 1..len {
                let target_aa = match mode {
                    CasMode::Cacon => seq[k - 1].as_str(),
                    CasMode::Canco => seq[k].as_str(),
                };
                let is_true = residues.contains(&target_aa);
                let is_bt = !is_true && rng.f64() < 0.06;
                if (is_true && rng.f64() >= spec.random_drop_frac) || is_bt {
                    let h = if is_true { 6.0e4 } else { 1.4e4 };
                    peaks.push(CasPeak {
                        c_prime_ppm: shifts[k - 1].2
                            + spec.tms_c13_offset
                            + rng.normal(0.0, spec.jitter_c),
                        n_ppm: shifts[k].1 + rng.normal(0.0, spec.jitter_n),
                        height: h,
                        volume: h,
                    });
                }
            }
            cas_spectra.push(CasSpectrum {
                name: label.to_string(),
                mode,
                residues: residues.iter().map(|s| s.to_string()).collect(),
                peaks,
            });
        }
    }

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
        seq: seq.to_vec(),
        shifts,
        exchange_broadened,
        tms_c13_offset: spec.tms_c13_offset,
    };
    (dataset, truth)
}

#[derive(Clone, Debug, Default)]
struct EvalMetrics {
    n_slots: usize,
    n_assigned: usize,
    n_correct: usize,
    n_pro_total: usize,
    n_pro_correct: usize,
    n_hc: usize,
    n_hc_correct: usize,
    n_abstained: usize,
    n_nodes: usize,
}

impl EvalMetrics {
    fn acc(&self) -> f64 {
        if self.n_slots == 0 {
            0.0
        } else {
            self.n_correct as f64 / self.n_slots as f64
        }
    }
    fn pro_acc(&self) -> f64 {
        if self.n_pro_total == 0 {
            1.0
        } else {
            self.n_pro_correct as f64 / self.n_pro_total as f64
        }
    }
    fn abstain_rate(&self) -> f64 {
        if self.n_slots == 0 {
            0.0
        } else {
            self.n_abstained as f64 / self.n_slots as f64
        }
    }
}

/// Check whether `node` physically originated from sequence slot `k` using the true synthetic
/// referencing offset `truth.tms_c13_offset` (and `1H` amide shift when present to disambiguate
/// overlapping `2D CON` roots).
fn is_node_match_for_slot(
    node: &JointIdpNode,
    _offsets: &IdpShiftOffsets,
    k: usize,
    truth: &SyntheticGroundTruth,
) -> bool {
    let len = truth.seq.len();
    let n_obs = node.n_curr;
    let slot_dist = |kp: usize| -> f64 {
        let dn = (n_obs - truth.shifts[kp].1) / 0.55;
        match (node.c_prev, node.h_curr) {
            (Some(c_obs), Some(h_obs)) => {
                let dc = ((c_obs - truth.tms_c13_offset) - truth.shifts[kp - 1].2) / 0.14;
                if let Some(h_true) = truth.shifts[kp].0 {
                    let dh = (h_obs - h_true) / 0.05;
                    (dc * dc + dn * dn + dh * dh).sqrt()
                } else {
                    1e6
                }
            }
            (Some(c_obs), None) => {
                let dc = ((c_obs - truth.tms_c13_offset) - truth.shifts[kp - 1].2) / 0.14;
                dc.hypot(dn)
            }
            (None, Some(h_obs)) => {
                if let Some(h_true) = truth.shifts[kp].0 {
                    ((h_obs - h_true) / 0.05).hypot(dn)
                } else {
                    1e6
                }
            }
            (None, None) => 1e6,
        }
    };

    let dist_k = slot_dist(k);
    if dist_k > 2.8 {
        return false;
    }
    let mut best_slot = k;
    let mut best_dist = dist_k;
    for kp in 1..len {
        let d = slot_dist(kp);
        if d < best_dist {
            best_dist = d;
            best_slot = kp;
        }
    }
    best_slot == k
}

/// Evaluate a `JointIdpAssignResult` against ground truth across `k = 1..L`.
fn evaluate_joint_result(res: &JointIdpAssignResult, truth: &SyntheticGroundTruth) -> EvalMetrics {
    let len = truth.seq.len();
    let mut m = EvalMetrics {
        n_slots: len.saturating_sub(1),
        n_nodes: res.nodes.len(),
        ..EvalMetrics::default()
    };
    for k in 1..len {
        let ra = &res.residue_assignments[k];
        let is_pro = truth.seq[k] == "PRO";
        if is_pro {
            m.n_pro_total += 1;
        }
        let Some(node_idx) = ra.node_idx else {
            m.n_abstained += 1;
            continue;
        };
        m.n_assigned += 1;
        let is_correct =
            is_node_match_for_slot(&res.nodes[node_idx], &res.fitted_offsets, k, truth);
        if is_correct {
            m.n_correct += 1;
            if is_pro {
                m.n_pro_correct += 1;
            }
        }
        if ra.confidence >= 0.80 {
            m.n_hc += 1;
            if is_correct {
                m.n_hc_correct += 1;
            }
        }
    }
    m
}

/// Disjoint empirical PAV calibration fitter (`180.core:P1-1`).
///
/// Run explicitly with:
/// Helper: compute 4-bin reliability curve `[0.0-0.5), [0.5-0.8), [0.8-0.9), [0.9-1.0]` and ECE
/// over `(predicted_conf, is_correct)` pairs.
fn compute_reliability_and_ece(samples: &[(f64, bool)]) -> (f64, [(usize, f64, f64); 4]) {
    let bounds = [(0.0, 0.5), (0.5, 0.8), (0.8, 0.9), (0.9, 1.000_001)];
    let mut bins = [(0usize, 0.0f64, 0.0f64); 4];
    let mut ece = 0.0f64;
    let total = samples.len().max(1) as f64;
    for (bi, &(lo, hi)) in bounds.iter().enumerate() {
        let mut n = 0usize;
        let mut sum_p = 0.0;
        let mut sum_y = 0.0;
        for &(p, y) in samples {
            if p >= lo && p < hi {
                n += 1;
                sum_p += p;
                if y {
                    sum_y += 1.0;
                }
            }
        }
        if n > 0 {
            let mp = sum_p / (n as f64);
            let my = sum_y / (n as f64);
            bins[bi] = (n, mp, my);
            ece += ((n as f64) / total) * (mp - my).abs();
        }
    }
    (ece, bins)
}

/// Exact tie-aware Mann–Whitney U AUROC for `(predicted_confidence, is_correct)` pairs.
fn compute_auroc(samples: &[(f64, bool)]) -> f64 {
    let n_pos = samples.iter().filter(|&&(_, y)| y).count();
    let n_neg = samples.len() - n_pos;
    if n_pos == 0 {
        return 0.5;
    }
    if n_neg == 0 {
        return 1.0;
    }
    let mut sorted: Vec<(f64, bool)> = samples.to_vec();
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut rank_sum_pos = 0.0_f64;
    let mut i = 0usize;
    while i < sorted.len() {
        let mut j = i + 1;
        while j < sorted.len() && (sorted[j].0 - sorted[i].0).abs() <= 1e-12 {
            j += 1;
        }
        // 1-indexed average rank across the tie group [i, j)
        let avg_rank = 0.5 * ((i + 1) as f64 + j as f64);
        let pos_in_group = sorted[i..j].iter().filter(|&&(_, y)| y).count();
        rank_sum_pos += avg_rank * (pos_in_group as f64);
        i = j;
    }
    let u = rank_sum_pos - (n_pos as f64) * ((n_pos + 1) as f64) / 2.0;
    (u / ((n_pos as f64) * (n_neg as f64))).clamp(0.0, 1.0)
}

/// Regularized isotonic PAV knot fitter (`180r3.core:P2-4`):
/// Uses a sharp Gaussian kernel (`bw = 0.045`) plus a weak identity prior (`n_prior = 30` total
/// pseudo-observations along `y = x` concentrated on low bins `x < 0.55`) so sparse low-posterior
/// knots never extrapolate high from the dense `raw > 0.90` mass.
/// Returns `(x, fitted_y, nearest_empirical_n)` for every knot.
fn fit_unbiased_calibration_knots_with_prior(
    samples: &[(f64, bool)],
    knot_xs: &[f64],
) -> Vec<(f64, f64, usize)> {
    let m = knot_xs.len();
    let bw = 0.045_f64;
    let mut nearest_n = vec![0usize; m];
    for &(p, _) in samples {
        let mut best_i = 0usize;
        let mut best_d = (p - knot_xs[0]).abs();
        for (i, &xk) in knot_xs.iter().enumerate().skip(1) {
            let d = (p - xk).abs();
            if d < best_d {
                best_d = d;
                best_i = i;
            }
        }
        nearest_n[best_i] += 1;
    }

    let low_indices: Vec<usize> = knot_xs
        .iter()
        .enumerate()
        .filter(|(_, &x)| x > 0.0 && x < 0.55)
        .map(|(i, _)| i)
        .collect();
    let prior_per_low = if low_indices.is_empty() {
        0.0
    } else {
        30.0 / (low_indices.len() as f64)
    };

    // Weighted kernel mean + weak identity prior on low bins
    let mut y = vec![0.0_f64; m];
    let mut w = vec![0.0_f64; m];
    for (i, &xk) in knot_xs.iter().enumerate() {
        if xk <= 0.0 {
            y[i] = 0.0;
            w[i] = 1e6;
            continue;
        }
        let mut sw = 0.0_f64;
        let mut sy = 0.0_f64;
        for &(p, ok) in samples {
            let d = (p - xk) / bw;
            if d.abs() <= 4.0 {
                let wt = (-0.5 * d * d).exp();
                sw += wt;
                if ok {
                    sy += wt;
                }
            }
        }
        let prior_w = if xk < 0.55 {
            prior_per_low
        } else if xk <= 0.65 {
            2.0
        } else {
            0.5
        };
        sw += prior_w;
        sy += prior_w * xk;
        y[i] = (sy / sw.max(1e-9)).clamp(0.0, 0.995);
        w[i] = sw.max(1e-6);
    }

    // Weighted Pool-Adjacent-Violators (PAV) for non-decreasing isotonicity
    let mut blocks: Vec<(usize, usize, f64, f64)> = Vec::with_capacity(m);
    for i in 0..m {
        blocks.push((i, i, y[i] * w[i], w[i]));
        while blocks.len() >= 2 {
            let len = blocks.len();
            let prev = blocks[len - 2];
            let curr = blocks[len - 1];
            if prev.2 / prev.3 > curr.2 / curr.3 {
                blocks.pop();
                blocks[len - 2] = (prev.0, curr.1, prev.2 + curr.2, prev.3 + curr.3);
            } else {
                break;
            }
        }
    }
    let mut out = vec![(0.0, 0.0, 0usize); m];
    for (lo, hi, swy, sw) in blocks {
        let val = (swy / sw).clamp(0.0, 0.995);
        for i in lo..=hi {
            let v = if knot_xs[i] <= 0.0 { 0.0 } else { val };
            out[i] = (knot_xs[i], v, nearest_n[i]);
        }
    }
    out
}

/// Anchor unanchored 3D strips (`CANCO`, `CBCACON`, `CCCON`) to nearest `CON` roots using the
/// exact `(0.14 ppm C', 0.50 ppm N)` elliptical tolerance used by `joint_idp_solver::sanitize_c13_dataset`
/// (`180r2.core:P1-2`: ensures a fair `idp-13c` baseline that is NOT starved of 3D CA/CB strips).
fn anchor_c13_strips_fair(ds: &C13PickedDataset) -> C13PickedDataset {
    let mut out = ds.clone();
    for sp in &mut out.strip_peaks {
        if sp.anchor_root_idx.is_some() {
            continue;
        }
        let mut best_r = None;
        let mut best_d2 = 1.0_f64; // Elliptical gate: (dc/0.14)^2 + (dn/0.50)^2 <= 1.0
        for (ri, r) in out.con_roots.iter().enumerate() {
            let dc = (sp.c_prime_ppm - r.c_prev_ppm) / 0.14;
            let dn = (sp.n_ppm - r.n_curr_ppm) / 0.50;
            let d2 = dc * dc + dn * dn;
            if d2 <= best_d2 {
                best_d2 = d2;
                best_r = Some(ri);
            }
        }
        sp.anchor_root_idx = best_r;
    }
    out
}

/// Strictly-unbiased disjoint empirical PAV calibration fitter (`180r3.core:P1-1`, `P2-4`).
///
/// Run explicitly with:
/// `cargo test --release -p nmr-core --test joint_idp_solver_tests fit_disjoint_joint_idp_calibration_knots -- --ignored --nocapture`
/// Fits `JOINT_IDP_CALIBRATION_KNOTS` strictly on:
/// - 4 disjoint calibration sequences (`cal_seq_a`, `cal_seq_b`, `rand_cal_9001`, `rand_cal_9002`),
/// - **STRICTLY UNBIASED** regimes (`predictor_bias_n = 0`, `predictor_bias_c = 0`,
///   `helix_amp = 0`, `neighbor_bias_* = 0`) so EVERY bias family (`B0..B5`) stays 100% OOD,
/// - Non-`ABSTAIN` (`PASS` / `WARN`) runs ONLY, with weak identity prior (`n_prior = 30`) on
///   low bins (`x < 0.55`) and printed per-knot empirical sample counts (`P2-4`).
#[test]
#[ignore]
fn fit_disjoint_joint_idp_calibration_knots() {
    let seqs = [
        disjoint_calibration_seq_a(),
        disjoint_calibration_seq_b(),
        generate_random_idp_sequence(125, 9001),
        generate_random_idp_sequence(135, 9002),
    ];
    let seeds = [1001_u64, 1002];
    let scenarios: &[(&str, IdpStressSpec)] = &[
        (
            "cal_clean_1x",
            IdpStressSpec {
                jitter_h: 0.004,
                jitter_n: 0.03,
                jitter_c: 0.014,
                rc_dev_scale: 1.0,
                random_drop_frac: 0.04,
                n_decoy_con_roots: 5,
                n_decoy_hn_roots: 2,
                ..IdpStressSpec::default()
            },
        ),
        ("cal_mod_1x", IdpStressSpec::default()),
        (
            "cal_real_2.2x",
            IdpStressSpec {
                rc_dev_scale: 2.2,
                ..IdpStressSpec::default()
            },
        ),
        (
            "cal_noise_mid",
            IdpStressSpec {
                jitter_h: 0.01,
                jitter_n: 0.08,
                jitter_c: 0.03,
                rc_dev_scale: 1.8,
                ..IdpStressSpec::default()
            },
        ),
        (
            "cal_drop20_2.0x",
            IdpStressSpec {
                rc_dev_scale: 2.0,
                random_drop_frac: 0.20,
                ..IdpStressSpec::default()
            },
        ),
        (
            "cal_noise_hi",
            IdpStressSpec {
                jitter_h: 0.02,
                jitter_n: 0.15,
                jitter_c: 0.05,
                rc_dev_scale: 1.8,
                random_drop_frac: 0.18,
                ..IdpStressSpec::default()
            },
        ),
    ];

    let mut samples: Vec<(f64, bool)> = Vec::new();
    let mut cal_samples: Vec<(f64, bool)> = Vec::new();
    for (label, spec) in scenarios {
        assert_eq!(spec.predictor_bias_n, 0.0);
        assert_eq!(spec.predictor_bias_c, 0.0);
        assert_eq!(spec.helix_amp, 0.0);
        assert_eq!(spec.neighbor_bias_n, 0.0);
        assert_eq!(spec.neighbor_bias_c, 0.0);
        let mut tot_ok = 0usize;
        let mut tot_n = 0usize;
        let mut skipped_abstain = 0usize;
        for (si, seq) in seqs.iter().enumerate() {
            for &seed in &seeds {
                let (ds, truth) = synthesize_idp_stress_dataset(seq, seed + (si as u64) * 97, spec);
                let cfg = JointIdpConfig {
                    replicas: 16,
                    steps_per_system: 1100,
                    seed: seed + 17,
                    ..JointIdpConfig::default()
                };
                let res = assign_joint_idp_cpu(&ds, seq, &cfg).unwrap();
                if res.qc_verdict == DatasetQcVerdict::Abstain {
                    skipped_abstain += 1;
                    continue;
                }
                for k in 1..seq.len() {
                    let ra = &res.residue_assignments[k];
                    let Some(node_idx) = ra.node_idx else {
                        continue;
                    };
                    let ok = is_node_match_for_slot(
                        &res.nodes[node_idx],
                        &res.fitted_offsets,
                        k,
                        &truth,
                    );
                    if ok {
                        tot_ok += 1;
                    }
                    tot_n += 1;
                    samples.push((ra.raw_posterior, ok));
                    cal_samples.push((ra.confidence, ok));
                }
            }
        }
        println!(
            "UNBIASED CAL {label:<16}: assigned acc = {tot_ok}/{tot_n} ({:.1}%) | abstained_runs={skipped_abstain}",
            100.0 * tot_ok as f64 / tot_n.max(1) as f64
        );
    }

    let knot_xs = [
        0.00, 0.15, 0.30, 0.45, 0.60, 0.72, 0.82, 0.88, 0.94, 0.98, 1.00,
    ];
    let fitted = fit_unbiased_calibration_knots_with_prior(&samples, &knot_xs);
    println!(
        "\nFitted STRICTLY-UNBIASED PASS/WARN PAV knots ({} samples, n_prior=30 on x<0.55):",
        samples.len()
    );
    for (x, y, n_bin) in &fitted {
        println!("    ({x:.2}, {y:.3}), // n_bin={n_bin}");
    }
    println!("\nEmpirical unsmoothed raw_posterior bins:");
    for &(lo, hi) in &[
        (0.00, 0.20),
        (0.20, 0.40),
        (0.40, 0.60),
        (0.60, 0.75),
        (0.75, 0.85),
        (0.85, 0.90),
        (0.90, 0.95),
        (0.95, 0.98),
        (0.98, 1.001),
    ] {
        let mut n = 0usize;
        let mut ok_n = 0usize;
        for &(p, ok) in &samples {
            if p >= lo && p < hi {
                n += 1;
                if ok {
                    ok_n += 1;
                }
            }
        }
        if n > 0 {
            println!(
                "  raw [{lo:.2}, {hi:.2}): n={n:4}  emp_acc={:.3}",
                ok_n as f64 / n as f64
            );
        }
    }
    let (ece, bins) = compute_reliability_and_ece(&cal_samples);
    let auroc = compute_auroc(&cal_samples);
    println!(
        "Current-knots cal ECE={ece:.3} | AUROC={auroc:.3} | bins: [0-0.5): n={} acc={:.3}, [0.5-0.8): n={} acc={:.3}, [0.8-0.9): n={} acc={:.3}, [0.9-1.0]: n={} acc={:.3}",
        bins[0].0, bins[0].2, bins[1].0, bins[1].2, bins[2].0, bins[2].2, bins[3].0, bins[3].2
    );
}

struct BiasProbeStats {
    n_runs: usize,
    n_pass: usize,
    n_warn: usize,
    n_abstain: usize,
    n_mismatch_triggered: usize,
    n_aa_trig: usize,
    n_seg_trig: usize,
    n_nb_trig: usize,
    tot_ok: usize,
    tot_slots: usize,
    tot_hc: usize,
    tot_hc_ok: usize,
    mean_dc: f64,
    ece: f64,
    auroc: f64,
    bins: [(usize, f64, f64); 4],
}

fn run_stress_probe(
    seqs: &[Vec<String>],
    seeds: &[u64],
    spec: &IdpStressSpec,
    seed_stride: u64,
) -> BiasProbeStats {
    let mut samples: Vec<(f64, bool)> = Vec::new();
    let mut n_runs = 0usize;
    let mut n_pass = 0usize;
    let mut n_warn = 0usize;
    let mut n_abstain = 0usize;
    let mut n_mismatch_triggered = 0usize;
    let mut n_aa_trig = 0usize;
    let mut n_seg_trig = 0usize;
    let mut n_nb_trig = 0usize;
    let mut tot_ok = 0usize;
    let mut tot_slots = 0usize;
    let mut tot_hc = 0usize;
    let mut tot_hc_ok = 0usize;
    let mut sum_dc = 0.0_f64;

    for (si, seq) in seqs.iter().enumerate() {
        for &seed in seeds {
            n_runs += 1;
            let (ds, truth) =
                synthesize_idp_stress_dataset(seq, seed + (si as u64) * seed_stride, spec);
            let cfg = JointIdpConfig {
                replicas: 12,
                steps_per_system: 800,
                seed: seed * 17 + (si as u64),
                ..JointIdpConfig::default()
            };
            let res = assign_joint_idp_cpu(&ds, seq, &cfg).unwrap();
            match res.qc_verdict {
                DatasetQcVerdict::Pass => n_pass += 1,
                DatasetQcVerdict::Warn => n_warn += 1,
                DatasetQcVerdict::Abstain => n_abstain += 1,
            }
            if res.predictor_mismatch.triggered {
                n_mismatch_triggered += 1;
            }
            if res.predictor_mismatch.aa_bias_triggered {
                n_aa_trig += 1;
            }
            if res.predictor_mismatch.segmental_bias_triggered {
                n_seg_trig += 1;
            }
            if res.predictor_mismatch.neighbor_bias_triggered {
                n_nb_trig += 1;
            }
            sum_dc += res.fitted_offsets.d_c;
            for k in 1..seq.len() {
                tot_slots += 1;
                let ra = &res.residue_assignments[k];
                let Some(node_idx) = ra.node_idx else {
                    continue;
                };
                let ok =
                    is_node_match_for_slot(&res.nodes[node_idx], &res.fitted_offsets, k, &truth);
                if ok {
                    tot_ok += 1;
                }
                if ra.confidence >= 0.80 {
                    tot_hc += 1;
                    if ok {
                        tot_hc_ok += 1;
                    }
                }
                samples.push((ra.confidence, ok));
            }
        }
    }

    let (ece, bins) = compute_reliability_and_ece(&samples);
    let auroc = compute_auroc(&samples);
    BiasProbeStats {
        n_runs,
        n_pass,
        n_warn,
        n_abstain,
        n_mismatch_triggered,
        n_aa_trig,
        n_seg_trig,
        n_nb_trig,
        tot_ok,
        tot_slots,
        tot_hc,
        tot_hc_ok,
        mean_dc: sum_dc / n_runs.max(1) as f64,
        ece,
        auroc,
        bins,
    }
}

/// Specificity probe on unbiased datasets `A1` (default 1x), `A2` (clean 1x), `A3` (unbiased 2.2x)
/// (`180r3.core:P1-2`): verifies the scale-invariant predictor-mismatch detector maintains high
/// specificity (`<= 1/8` false triggers on every unbiased regime), preserves `QC=PASS >= 6/8` on
/// clean/moderate `1x`, and ranks `[0.9, 1.0] > [0.8, 0.9)` with honest `ECE` & `AUROC` on `2.2x`.
#[test]
fn test_v180_r4_specificity_probe_unbiased_a1_a2_a3() {
    let oof_seqs = [
        generate_random_idp_sequence(130, 501),
        generate_random_idp_sequence(128, 502),
        generate_random_idp_sequence(132, 503),
        generate_random_idp_sequence(130, 504),
    ];
    let seeds = [41_u64, 83];
    let scenarios: &[(&str, IdpStressSpec)] = &[
        ("A1_default_1x", IdpStressSpec::default()),
        (
            "A2_clean_1x",
            IdpStressSpec {
                jitter_h: 0.004,
                jitter_n: 0.03,
                jitter_c: 0.014,
                rc_dev_scale: 1.0,
                random_drop_frac: 0.04,
                n_decoy_con_roots: 5,
                n_decoy_hn_roots: 2,
                ..IdpStressSpec::default()
            },
        ),
        (
            "A3_unbiased_2.2x",
            IdpStressSpec {
                rc_dev_scale: 2.2,
                ..IdpStressSpec::default()
            },
        ),
    ];

    println!("\n=== Round-4 Unbiased Specificity Probe (A1-A3, 4 fresh ~130-aa IDPs x 2 seeds = 8 runs each) ===");
    println!(
        "| Scenario | Acc% | #HC | HC_Prec% | MismatchTrig | QC(P/W/A) | ECE | AUROC | [0.8-0.9) | [0.9-1.0] |"
    );
    println!("|---|---:|---:|---:|---:|---|---:|---:|---:|---:|");

    let mut failures: Vec<String> = Vec::new();
    for (label, spec) in scenarios {
        let st = run_stress_probe(&oof_seqs, &seeds, spec, 313);
        let acc = 100.0 * st.tot_ok as f64 / st.tot_slots.max(1) as f64;
        let hc_prec = if st.tot_hc == 0 {
            100.0
        } else {
            100.0 * st.tot_hc_ok as f64 / st.tot_hc as f64
        };
        println!(
            "| {label:<16} | {acc:5.1}% | {:4} | {hc_prec:7.1}% | {:2}/{:<2} ({}/{}/{}) | {}/{}/{}     | {:.3} | {:.3} | {:.3} ({:3}) | {:.3} ({:3}) |",
            st.tot_hc,
            st.n_mismatch_triggered,
            st.n_runs,
            st.n_aa_trig,
            st.n_seg_trig,
            st.n_nb_trig,
            st.n_pass,
            st.n_warn,
            st.n_abstain,
            st.ece,
            st.auroc,
            st.bins[2].2,
            st.bins[2].0,
            st.bins[3].2,
            st.bins[3].0,
        );

        if st.n_mismatch_triggered > 1 {
            failures.push(format!(
                "180r3.core:P1-2 specificity breach on {label}: predictor_mismatch triggered {}/{} (must be <= 1/8)",
                st.n_mismatch_triggered, st.n_runs
            ));
        }
        if (*label == "A1_default_1x" || *label == "A2_clean_1x") && st.n_pass < 6 {
            failures.push(format!(
                "180r3.core:P1-2 specificity breach on {label}: QC=PASS {}/{} (must be >= 6/8)",
                st.n_pass, st.n_runs
            ));
        }
        if *label == "A3_unbiased_2.2x" {
            if st.ece > 0.08 {
                failures.push(format!(
                    "180r3.core:P1-1: unbiased 2.2x ECE ({:.3}) exceeded 0.08",
                    st.ece
                ));
            }
            if st.auroc < 0.65 {
                failures.push(format!(
                    "180r3.core:P1-1: unbiased 2.2x AUROC ({:.3}) below 0.65",
                    st.auroc
                ));
            }
            if st.bins[2].0 >= 5 && st.bins[3].0 >= 5 && st.bins[3].2 <= st.bins[2].2 {
                failures.push(format!(
                    "180r3.core:P1-1: unbiased 2.2x [0.9, 1.0] ({:.3}, n={}) must strictly beat [0.8, 0.9) ({:.3}, n={})",
                    st.bins[3].2, st.bins[3].0, st.bins[2].2, st.bins[2].0
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "A1-A3 specificity failures:\n{}",
        failures.join("\n")
    );
}

/// Held-out OOD predictor-bias suite (`B0a..B5`, 7 distinct bias families/magnitudes/seeds,
/// 100% excluded from calibration training; `180r3.core:P1-1`, `P1-2`).
#[test]
fn test_v180_r4_heldout_ood_bias_suite_b0_to_b5() {
    let oof_seqs = [
        generate_random_idp_sequence(130, 501),
        generate_random_idp_sequence(128, 502),
        generate_random_idp_sequence(132, 503),
        generate_random_idp_sequence(130, 504),
    ];
    let base_22x = IdpStressSpec {
        jitter_h: 0.006,
        jitter_n: 0.04,
        jitter_c: 0.018,
        rc_dev_scale: 2.2,
        random_drop_frac: 0.08,
        n_decoy_con_roots: 12,
        n_decoy_hn_roots: 5,
        tms_c13_offset: -2.70,
        invert_hncacb: true,
        include_trosy_hncaconh: true,
        include_cas: true,
        exchange_broaden: true,
        ..IdpStressSpec::default()
    };

    let cases: &[(&str, [u64; 2], IdpStressSpec, usize, bool)] = &[
        (
            "B0a_perAA_41_83",
            [41, 83],
            IdpStressSpec {
                predictor_bias_n: 1.0,
                predictor_bias_c: 0.40,
                ..base_22x
            },
            1,
            false,
        ),
        (
            "B0b_perAA_137_293",
            [137, 293],
            IdpStressSpec {
                predictor_bias_n: 1.0,
                predictor_bias_c: 0.40,
                ..base_22x
            },
            1,
            false,
        ),
        (
            "B1_helix_1.0",
            [41, 83],
            IdpStressSpec {
                helix_amp: 1.0,
                ..base_22x
            },
            0,
            true,
        ),
        (
            "B2_helix_1.5",
            [41, 83],
            IdpStressSpec {
                helix_amp: 1.5,
                ..base_22x
            },
            0,
            true,
        ),
        (
            "B3_strong_perAA",
            [41, 83],
            IdpStressSpec {
                predictor_bias_n: 2.0,
                predictor_bias_c: 0.80,
                ..base_22x
            },
            0,
            false,
        ),
        (
            "B4_helix_plus_AA",
            [41, 83],
            IdpStressSpec {
                helix_amp: 1.0,
                predictor_bias_n: 1.0,
                predictor_bias_c: 0.40,
                ..base_22x
            },
            0,
            false,
        ),
        (
            "B5_neighbor_bias",
            [41, 83],
            IdpStressSpec {
                neighbor_bias_n: 0.90,
                neighbor_bias_c: 0.35,
                ..base_22x
            },
            1,
            false,
        ),
        (
            "B6_bias_drop25",
            [41, 83],
            IdpStressSpec {
                predictor_bias_n: 2.0,
                predictor_bias_c: 0.80,
                random_drop_frac: 0.25,
                ..base_22x
            },
            0,
            false,
        ),
    ];

    println!(
        "\n=== Round-4 Held-Out 100%-OOD Bias Suite (B0a-B5, 8 runs each; true TMS dC=-2.70) ==="
    );
    println!(
        "| Scenario | Acc% | #HC | HC_Prec% | MismatchTrig | QC(P/W/A) | Mean dC | ECE | AUROC |"
    );
    println!("|---|---:|---:|---:|---:|---|---:|---:|---:|");

    let mut failures: Vec<String> = Vec::new();
    for (label, seeds, spec, max_pass_allowed, check_tms_dc) in cases {
        let st = run_stress_probe(&oof_seqs, seeds, spec, 313);
        let acc = 100.0 * st.tot_ok as f64 / st.tot_slots.max(1) as f64;
        let hc_prec = if st.tot_hc == 0 {
            100.0
        } else {
            100.0 * st.tot_hc_ok as f64 / st.tot_hc as f64
        };
        println!(
            "| {label:<17} | {acc:5.1}% | {:4} | {hc_prec:7.1}% | {:2}/{:<2} ({}/{}/{}) | {}/{}/{}     | {:+6.3} | {:.3} | {:.3} |",
            st.tot_hc,
            st.n_mismatch_triggered,
            st.n_runs,
            st.n_aa_trig,
            st.n_seg_trig,
            st.n_nb_trig,
            st.n_pass,
            st.n_warn,
            st.n_abstain,
            st.mean_dc,
            st.ece,
            st.auroc,
        );

        if st.n_pass > *max_pass_allowed {
            failures.push(format!(
                "180r3.core:P1-2 OOD detection miss on {label}: QC=PASS {}/{} (max allowed {max_pass_allowed})",
                st.n_pass, st.n_runs
            ));
        }
        if st.tot_hc > 20 && hc_prec < 80.0 {
            failures.push(format!(
                "180r3.core:P1-1 OOD HC precision breach on {label}: {}/{} ({hc_prec:.1}%)",
                st.tot_hc_ok, st.tot_hc
            ));
        }
        if st.ece > 0.12 {
            failures.push(format!(
                "180r3.core:P1-1 OOD ECE breach on {label}: {:.3} > 0.12",
                st.ece
            ));
        }
        if st.auroc < 0.60 {
            failures.push(format!(
                "180r3.core:P1-1 OOD AUROC breach on {label}: {:.3} < 0.60",
                st.auroc
            ));
        }
        if *check_tms_dc && (st.mean_dc - (-2.70)).abs() > 0.10 {
            failures.push(format!(
                "180r3.core:P1-1(b) segmental offset drift on {label}: mean dC={:+.3} vs true -2.700 (|err|={:.3} > 0.10)",
                st.mean_dc, (st.mean_dc - (-2.70)).abs()
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "B0a-B5 OOD failures:\n{}",
        failures.join("\n")
    );
}

/// Periodic motif-repeat probe (`180r3.core:P2-2`): 3 fresh 130-aa IDPs containing imperfect
/// 9-mer tandem repeats (alpha-synuclein KTKEGV / Pol II CTD style, which pass high-Shannon-entropy
/// checks!) tested at `1.0x` and `2.2x` coil spread.
#[test]
fn test_v180_r4_motif_repeat_probe_p2_2() {
    let motif_seqs = [
        generate_motif_repeat_idp_sequence(130, 801),
        generate_motif_repeat_idp_sequence(130, 802),
        generate_motif_repeat_idp_sequence(130, 803),
    ];
    for (i, seq) in motif_seqs.iter().enumerate() {
        let (period, id, z) = motif_repeat_metrics(seq);
        assert!(
            id >= 0.28 && z >= 4.5,
            "motif_seq[{i}] should trigger motif_repeat_metrics (got period={period}, id={id:.3}, z={z:.2})"
        );
    }

    let regimes: &[(&str, IdpStressSpec)] = &[
        ("motif_1.0x", IdpStressSpec::default()),
        (
            "motif_2.2x",
            IdpStressSpec {
                rc_dev_scale: 2.2,
                ..IdpStressSpec::default()
            },
        ),
    ];

    println!("\n=== Round-4 Periodic Motif-Repeat Probe (P2-2; 3 x 130-aa 9-mer repeat IDPs) ===");
    println!(
        "| Regime | Acc% | #HC | HC_Prec% | QC(P/W/A) | ECE | AUROC | [0.8-0.9) | [0.9-1.0] |"
    );
    println!("|---|---:|---:|---:|---|---:|---:|---:|---:|");

    for (label, spec) in regimes {
        let mut samples: Vec<(f64, bool)> = Vec::new();
        let mut n_pass = 0usize;
        let mut n_warn = 0usize;
        let mut n_abstain = 0usize;
        let mut tot_ok = 0usize;
        let mut tot_slots = 0usize;
        let mut tot_hc = 0usize;
        let mut tot_hc_ok = 0usize;

        for (si, seq) in motif_seqs.iter().enumerate() {
            let (ds, truth) = synthesize_idp_stress_dataset(seq, 601 + (si as u64) * 37, spec);
            let cfg = JointIdpConfig {
                replicas: 12,
                steps_per_system: 800,
                seed: 73 + (si as u64),
                ..JointIdpConfig::default()
            };
            let res = assign_joint_idp_cpu(&ds, seq, &cfg).unwrap();
            let has_repeat_reason = res.qc_reasons.iter().any(|r| {
                let lc = r.to_ascii_lowercase();
                lc.contains("motif") || lc.contains("repeat") || lc.contains("low-complexity")
            });
            assert!(
                has_repeat_reason,
                "180r3.core:P2-2 {label} seq[{si}] missing motif/repeat QC reason: {:?}",
                res.qc_reasons
            );
            match res.qc_verdict {
                DatasetQcVerdict::Pass => n_pass += 1,
                DatasetQcVerdict::Warn => n_warn += 1,
                DatasetQcVerdict::Abstain => n_abstain += 1,
            }
            for k in 1..seq.len() {
                tot_slots += 1;
                let ra = &res.residue_assignments[k];
                let Some(node_idx) = ra.node_idx else {
                    continue;
                };
                let ok =
                    is_node_match_for_slot(&res.nodes[node_idx], &res.fitted_offsets, k, &truth);
                if ok {
                    tot_ok += 1;
                }
                if ra.confidence >= 0.80 {
                    tot_hc += 1;
                    if ok {
                        tot_hc_ok += 1;
                    }
                }
                samples.push((ra.confidence, ok));
            }
        }
        let (ece, bins) = compute_reliability_and_ece(&samples);
        let auroc = compute_auroc(&samples);
        let acc = 100.0 * tot_ok as f64 / tot_slots.max(1) as f64;
        let hc_prec = if tot_hc == 0 {
            100.0
        } else {
            100.0 * tot_hc_ok as f64 / tot_hc as f64
        };
        println!(
            "| {label:<10} | {acc:5.1}% | {tot_hc:4} | {hc_prec:7.1}% | {n_pass}/{n_warn}/{n_abstain}     | {ece:.3} | {auroc:.3} | {:.3} ({:3}) | {:.3} ({:3}) |",
            bins[2].2, bins[2].0, bins[3].2, bins[3].0,
        );
        assert_eq!(
            n_pass, 0,
            "180r3.core:P2-2: periodic motif-repeat {label} must not return QC=PASS"
        );
        assert!(
            tot_hc <= 20 || hc_prec >= 80.0,
            "180r3.core:P2-2: {label} HC precision ({hc_prec:.1}%, {tot_hc_ok}/{tot_hc}) must be honest"
        );
        assert!(
            auroc >= 0.60,
            "180r3.core:P2-2: {label} AUROC ({auroc:.3}) below 0.60"
        );
        if bins[2].0 >= 5 && bins[3].0 >= 5 {
            assert!(
                bins[3].2 >= bins[2].2 - 0.01,
                "180r3.core:P2-2: {label} [0.9, 1.0] ({:.3}) should rank >= [0.8, 0.9) ({:.3})",
                bins[3].2,
                bins[2].2
            );
        }
    }
}

/// 9-scenario multi-seed (`3` seeds) held-out adversarial calibration & QC grid (`180.core:P1-1`,
/// `P1-2`, `P1-3`, `180r2.core:P1-4`, `180r3.core:P2-2`). Zero ground-truth `anchor_root_idx`.
#[test]
fn test_v180_adversarial_8_scenario_multiseed_grid() {
    let base_seq = synthetic_pro_ser_rich_idp_sequence();
    let repeat_seq = synthetic_low_complexity_repeat_idp_sequence();
    let motif_seq = generate_motif_repeat_idp_sequence(130, 777);
    let seeds = [7_u64, 13, 29];

    let scenarios: &[(&str, u8, IdpStressSpec)] = &[
        ("1_moderate_base", 0, IdpStressSpec::default()),
        (
            "2_noise_01_08_03",
            0,
            IdpStressSpec {
                jitter_h: 0.01,
                jitter_n: 0.08,
                jitter_c: 0.03,
                ..IdpStressSpec::default()
            },
        ),
        (
            "3_noise_02_15_05",
            0,
            IdpStressSpec {
                jitter_h: 0.02,
                jitter_n: 0.15,
                jitter_c: 0.05,
                ..IdpStressSpec::default()
            },
        ),
        (
            "4_drop_20pct",
            0,
            IdpStressSpec {
                random_drop_frac: 0.20,
                ..IdpStressSpec::default()
            },
        ),
        (
            "5_drop20_140dec",
            0,
            IdpStressSpec {
                random_drop_frac: 0.20,
                n_decoy_con_roots: 140,
                n_decoy_hn_roots: 25,
                ..IdpStressSpec::default()
            },
        ),
        (
            "6_rc_dev_2.2x",
            0,
            IdpStressSpec {
                rc_dev_scale: 2.2,
                ..IdpStressSpec::default()
            },
        ),
        (
            "7_combined_stress",
            0,
            IdpStressSpec {
                jitter_h: 0.02,
                jitter_n: 0.15,
                jitter_c: 0.05,
                rc_dev_scale: 2.2,
                random_drop_frac: 0.20,
                n_decoy_con_roots: 160,
                n_decoy_hn_roots: 30,
                ..IdpStressSpec::default()
            },
        ),
        (
            "8_repeat_low_cx",
            1,
            IdpStressSpec {
                jitter_h: 0.008,
                jitter_n: 0.06,
                jitter_c: 0.025,
                random_drop_frac: 0.10,
                n_decoy_con_roots: 20,
                n_decoy_hn_roots: 8,
                ..IdpStressSpec::default()
            },
        ),
        (
            "9_motif_repeat",
            2,
            IdpStressSpec {
                rc_dev_scale: 2.2,
                ..IdpStressSpec::default()
            },
        ),
    ];

    println!(
        "\n| Scenario | Acc% | Pro% | Nodes | #HC (>=0.8) | HC_Prec% | MaxConf | Abstain% | QC (3 seeds) |"
    );
    println!("|---|---:|---:|---:|---:|---:|---:|---:|---|");

    for &(label, seq_kind, ref spec) in scenarios {
        let seq = match seq_kind {
            1 => &repeat_seq,
            2 => &motif_seq,
            _ => &base_seq,
        };
        let mut sum_acc = 0.0;
        let mut sum_pro = 0.0;
        let mut sum_hc = 0.0;
        let mut tot_hc = 0usize;
        let mut tot_hc_ok = 0usize;
        let mut sum_abstain = 0.0;
        let mut sum_nodes = 0.0;
        let mut max_conf_overall = 0.0f64;
        let mut qc_list: Vec<DatasetQcVerdict> = Vec::new();

        for &seed in &seeds {
            let (ds, truth) = synthesize_idp_stress_dataset(seq, seed, spec);
            let cfg = JointIdpConfig {
                replicas: 12,
                steps_per_system: 800,
                seed: seed * 101 + 3,
                ..JointIdpConfig::default()
            };
            let res = assign_joint_idp_cpu(&ds, seq, &cfg).unwrap();
            let ev = evaluate_joint_result(&res, &truth);
            sum_acc += ev.acc();
            sum_pro += ev.pro_acc();
            sum_hc += ev.n_hc as f64;
            tot_hc += ev.n_hc;
            tot_hc_ok += ev.n_hc_correct;
            sum_abstain += ev.abstain_rate();
            sum_nodes += ev.n_nodes as f64;
            for &c in &res.confidence {
                max_conf_overall = max_conf_overall.max(c);
            }
            qc_list.push(res.qc_verdict);

            assert!(res.score.is_finite(), "{label} produced non-finite score");
            assert!(
                res.confidence
                    .iter()
                    .all(|c| c.is_finite() && *c >= 0.0 && *c <= 1.0),
                "{label} produced out-of-range confidence"
            );
        }

        let k_n = seeds.len() as f64;
        let mean_acc = 100.0 * sum_acc / k_n;
        let mean_pro = 100.0 * sum_pro / k_n;
        let mean_hc = sum_hc / k_n;
        let agg_hc_prec = if tot_hc == 0 {
            100.0
        } else {
            100.0 * tot_hc_ok as f64 / tot_hc as f64
        };
        let mean_abstain = 100.0 * sum_abstain / k_n;
        let mean_nodes = sum_nodes / k_n;
        let qc_str: Vec<String> = qc_list.iter().map(|q| q.to_string()).collect();
        println!(
            "| {label:<17} | {mean_acc:5.1}% | {mean_pro:5.1}% | {mean_nodes:5.1} | {mean_hc:5.1} | {agg_hc_prec:6.1}% | {max_conf_overall:7.3} | {mean_abstain:6.1}% | {} |",
            qc_str.join("/")
        );

        match label {
            "1_moderate_base" => {
                assert!(
                    mean_acc >= 75.0,
                    "moderate baseline mean accuracy {mean_acc:.1}% too low"
                );
                assert!(
                    mean_pro >= 65.0,
                    "moderate baseline Proline recovery {mean_pro:.1}% too low"
                );
                assert!(
                    agg_hc_prec >= 88.0,
                    "moderate baseline HC precision {agg_hc_prec:.1}% below 88%"
                );
            }
            "7_combined_stress" => {
                assert!(
                    qc_list.iter().all(|q| *q == DatasetQcVerdict::Abstain),
                    "180r2.core:P1-4: combined stress MUST return QC=Abstain (got {qc_list:?})"
                );
                assert_eq!(tot_hc, 0, "combined stress must emit zero HC calls");
            }
            "8_repeat_low_cx" => {
                assert!(
                    qc_list.iter().all(|q| *q == DatasetQcVerdict::Abstain),
                    "180r2.core:P1-4: near-random low-complexity repeat MUST escalate to QC=Abstain (got {qc_list:?})"
                );
                assert!(
                    max_conf_overall <= 0.15 + 1e-9,
                    "180r2.core:P1-4: repeat ABSTAIN max confidence ({max_conf_overall:.3}) must be <= 0.15"
                );
                assert_eq!(tot_hc, 0, "repeat ABSTAIN must emit zero HC calls");
            }
            _ => {
                let slots = (seq.len() - 1) as f64;
                assert!(
                    mean_hc <= slots * 0.30 || agg_hc_prec >= 78.0,
                    "180r2.core:P1-3: {label} overconfident ({mean_hc:.1} HC @ {agg_hc_prec:.1}% prec)"
                );
            }
        }
    }
}

/// Fair real-solver multi-seed (`3` seeds) baseline comparison (`180r2.core:P1-2`):
/// Compares `assign_joint_idp_cpu` (`idp-joint`), FAIR `idp_solver::try_assign_idp_13c_cpu`
/// (with 3D strips anchored via the identical `(0.14 ppm C', 0.50 ppm N)` gate so `idp-13c`
/// is NOT starved of `CA`/`CB` strips!), and `evo_graph::assign_evo_graph_from_peaks` (`evo-graph`),
/// reporting Accuracy, Recall, Proline recovery, #HC, HC precision, Abstain rate, and exact
/// theoretical ceilings side-by-side.
#[test]
fn test_v180_real_solver_multiseed_baseline_comparison() {
    let seq = synthetic_pro_ser_rich_idp_sequence();
    let priors = Priors::bmrb();
    let seeds = [11_u64, 23, 37];
    let spec = IdpStressSpec {
        jitter_h: 0.006,
        jitter_n: 0.04,
        jitter_c: 0.018,
        rc_dev_scale: 1.0,
        random_drop_frac: 0.08,
        n_decoy_con_roots: 15,
        n_decoy_hn_roots: 6,
        tms_c13_offset: 0.0,
        invert_hncacb: false,
        include_trosy_hncaconh: true,
        include_cas: true,
        exchange_broaden: true,
        ..IdpStressSpec::default()
    };

    let n_assignable = seq.len() - 1;
    let mut joint_acc_sum = 0.0;
    let mut joint_pro_sum = 0.0;
    let mut joint_abstain_sum = 0.0;
    let mut joint_hc = 0usize;
    let mut joint_hc_ok = 0usize;

    let mut c13_acc_sum = 0.0;
    let mut c13_pro_sum = 0.0;
    let mut c13_abstain_sum = 0.0;
    let mut c13_hc = 0usize;
    let mut c13_hc_ok = 0usize;

    let mut hn_acc_sum = 0.0;
    let mut hn_abstain_sum = 0.0;
    let mut hn_hc = 0usize;
    let mut hn_hc_ok = 0usize;
    let mut hn_ceiling_sum = 0.0;

    for &seed in &seeds {
        let (ds, truth) = synthesize_idp_stress_dataset(&seq, seed, &spec);

        // Exact theoretical ceiling for 1H-detected `evo-graph`: non-Proline, non-exchange-broadened slots
        let hn_observable = (1..seq.len())
            .filter(|&k| seq[k] != "PRO" && !truth.exchange_broadened[k])
            .count();
        hn_ceiling_sum += (hn_observable as f64) / (n_assignable as f64);

        // 1. Unified Joint solver (`assign_joint_idp_cpu`)
        let j_cfg = JointIdpConfig {
            replicas: 16,
            steps_per_system: 900,
            seed: seed * 31 + 1,
            ..JointIdpConfig::default()
        };
        let j_res = assign_joint_idp_cpu(&ds, &seq, &j_cfg).unwrap();
        let j_ev = evaluate_joint_result(&j_res, &truth);
        joint_acc_sum += j_ev.acc();
        joint_pro_sum += j_ev.pro_acc();
        joint_abstain_sum += j_ev.abstain_rate();
        joint_hc += j_ev.n_hc;
        joint_hc_ok += j_ev.n_hc_correct;

        // 2. FAIR Real 13C-detected IDP solver (`idp_solver::try_assign_idp_13c_cpu` + CAS)
        //    Anchor unanchored 3D strips via the exact same `(0.14 ppm C', 0.50 ppm N)` rule!
        let fair_c13_ds = anchor_c13_strips_fair(&ds.c13_dataset);
        let con_systems = register_and_build_con_dipeptide_systems(&fair_c13_ds);
        let n_with_ca = con_systems
            .iter()
            .filter(|s| !s.ca_curr.is_empty() || !s.ca_prev.is_empty())
            .count();
        assert!(
            n_with_ca * 2 >= con_systems.len(),
            "180r2.core:P1-2: fair idp-13c baseline must receive anchored 3D CA strips (got {n_with_ca}/{})",
            con_systems.len()
        );
        let c13_cfg = IdpSolverConfig {
            replicas: 16,
            steps_per_system: 900,
            seed: seed * 31 + 1,
            cas_spectra: ds.cas_spectra.clone(),
            ..IdpSolverConfig::default()
        };
        let c13_res = try_assign_idp_13c_cpu(&con_systems, &seq, &c13_cfg).unwrap();
        let mut c13_slot_to_sys: BTreeMap<usize, usize> = BTreeMap::new();
        for (si, &pos) in c13_res.position.iter().enumerate() {
            if pos >= 1 && (pos as usize) < seq.len() {
                c13_slot_to_sys.insert(pos as usize, si);
            }
        }
        let mut c13_ok = 0usize;
        let mut c13_pro_ok = 0usize;
        let mut c13_abstained = 0usize;
        let mut n_pro = 0usize;
        for k in 1..seq.len() {
            if seq[k] == "PRO" {
                n_pro += 1;
            }
            let Some(&si) = c13_slot_to_sys.get(&k) else {
                c13_abstained += 1;
                continue;
            };
            let sys = &con_systems[si];
            let c_corr = sys.c_prev - c13_res.fitted_offsets.d_c;
            let n_corr = sys.n_curr - c13_res.fitted_offsets.d_n;
            let dist_k = ((c_corr - truth.shifts[k - 1].2) / 0.14)
                .hypot((n_corr - truth.shifts[k].1) / 0.55);
            let mut best_slot = k;
            let mut best_dist = dist_k;
            for kp in 1..seq.len() {
                let d = ((c_corr - truth.shifts[kp - 1].2) / 0.14)
                    .hypot((n_corr - truth.shifts[kp].1) / 0.55);
                if d < best_dist {
                    best_dist = d;
                    best_slot = kp;
                }
            }
            let ok = best_slot == k && dist_k <= 2.8;
            if ok {
                c13_ok += 1;
                if seq[k] == "PRO" {
                    c13_pro_ok += 1;
                }
            }
            if c13_res.confidence[si] >= 0.80 {
                c13_hc += 1;
                if ok {
                    c13_hc_ok += 1;
                }
            }
        }
        c13_acc_sum += c13_ok as f64 / n_assignable as f64;
        c13_pro_sum += c13_pro_ok as f64 / n_pro.max(1) as f64;
        c13_abstain_sum += c13_abstained as f64 / n_assignable as f64;

        // 3. Real 1H-detected backbone solver (`evo_graph::assign_evo_graph_from_peaks`)
        let evo_cfg = EvoGraphConfig {
            steps_per_system_per_gen: 900,
            seed: seed * 31 + 1,
            ..EvoGraphConfig::default()
        };
        let (hn_systems, hn_res, _) =
            assign_evo_graph_from_peaks(&ds.hn_peaks, &seq, &priors, &evo_cfg);
        let mut hn_slot_to_sys: BTreeMap<usize, usize> = BTreeMap::new();
        for (si, &pos) in hn_res.position.iter().enumerate() {
            if pos >= 1 && (pos as usize) < seq.len() {
                hn_slot_to_sys.insert(pos as usize, si);
            }
        }
        let mut hn_ok = 0usize;
        let mut hn_abstained = 0usize;
        for (k, aa) in seq.iter().enumerate().skip(1) {
            if aa == "PRO" {
                hn_abstained += 1;
                continue; // 1H-only cannot assign Prolines (no amide 1H)
            }
            let Some(&si) = hn_slot_to_sys.get(&k) else {
                hn_abstained += 1;
                continue;
            };
            let sys = &hn_systems[si];
            let ok = if let Some(h_true) = truth.shifts[k].0 {
                (sys.h - h_true).abs() <= 0.08 && (sys.n - truth.shifts[k].1).abs() <= 0.55
            } else {
                false
            };
            if ok {
                hn_ok += 1;
            }
            if hn_res.confidence[si] >= 0.80 {
                hn_hc += 1;
                if ok {
                    hn_hc_ok += 1;
                }
            }
        }
        hn_acc_sum += hn_ok as f64 / n_assignable as f64;
        hn_abstain_sum += hn_abstained as f64 / n_assignable as f64;
    }

    let n_s = seeds.len() as f64;
    let joint_acc = 100.0 * joint_acc_sum / n_s;
    let joint_pro = 100.0 * joint_pro_sum / n_s;
    let joint_abs = 100.0 * joint_abstain_sum / n_s;
    let joint_hc_mean = joint_hc as f64 / n_s;
    let joint_hc_prec = if joint_hc == 0 {
        100.0
    } else {
        100.0 * joint_hc_ok as f64 / joint_hc as f64
    };

    let c13_acc = 100.0 * c13_acc_sum / n_s;
    let c13_pro = 100.0 * c13_pro_sum / n_s;
    let c13_abs = 100.0 * c13_abstain_sum / n_s;
    let c13_hc_mean = c13_hc as f64 / n_s;
    let c13_hc_prec = if c13_hc == 0 {
        100.0
    } else {
        100.0 * c13_hc_ok as f64 / c13_hc as f64
    };

    let hn_acc = 100.0 * hn_acc_sum / n_s;
    let hn_abs = 100.0 * hn_abstain_sum / n_s;
    let hn_hc_mean = hn_hc as f64 / n_s;
    let hn_hc_prec = if hn_hc == 0 {
        100.0
    } else {
        100.0 * hn_hc_ok as f64 / hn_hc as f64
    };
    let hn_ceiling = 100.0 * hn_ceiling_sum / n_s;

    println!(
        "\n=== Fair Real-Solver 3-Seed Baseline Comparison (140-aa Pro/Ser IDP; equal 0.14/0.50 strip anchoring) ==="
    );
    println!(
        "| Solver | Acc/Recall% | Pro% | #HC/seed | HC_Prec% | Abstain% | Theoretical Ceiling% |"
    );
    println!("|---|---:|---:|---:|---:|---:|---:|");
    println!(
        "| Joint (`idp-joint`)       | {joint_acc:10.1}% | {joint_pro:5.1}% | {joint_hc_mean:8.1} | {joint_hc_prec:7.1}% | {joint_abs:7.1}% |               100.0% |"
    );
    println!(
        "| Fair 13C (`idp-13c`+CAS)  | {c13_acc:10.1}% | {c13_pro:5.1}% | {c13_hc_mean:8.1} | {c13_hc_prec:7.1}% | {c13_abs:7.1}% |               100.0% |"
    );
    println!(
        "| Real 1H (`evo-graph`)     | {hn_acc:10.1}% |   0.0% | {hn_hc_mean:8.1} | {hn_hc_prec:7.1}% | {hn_abs:7.1}% | {hn_ceiling:19.1}% |"
    );

    assert!(
        joint_acc > c13_acc + 5.0,
        "Joint ({joint_acc:.1}%) should decisively beat fair real 13C-only ({c13_acc:.1}%)"
    );
    assert!(
        joint_acc > hn_acc + 5.0,
        "Joint ({joint_acc:.1}%) should decisively beat real 1H-only ({hn_acc:.1}%, ceiling {hn_ceiling:.1}%)"
    );
    assert!(
        joint_pro >= 70.0,
        "Joint Proline recovery ({joint_pro:.1}%) must decisively exceed 1H-only (0.0%)"
    );
}

/// Real experimental Clec16a IDPR2 (`CON + CAS + HN`) regression test (`180.core:P1-3`), gated on
/// `NMR_REAL_DATA_DIR`. Runs real Bruker `pdata/1` spectra end-to-end through `assign_joint_idp_cpu`
/// and verifies physical invariants & honest QC — with NO false ground-truth accuracy claim.
#[test]
fn test_v180_clec16a_real_data_joint_idp_regression() {
    let Ok(root_str) = std::env::var("NMR_REAL_DATA_DIR") else {
        return;
    };
    let clec = PathBuf::from(root_str).join("private/clec16a");
    let fasta = clec.join("clec16a_mouse_892-1050.fasta");
    let con_dir = clec.join("13C15N_Clec16aIDPR2_MG_091119/10007/pdata/1");
    let hsqc_dir = clec.join("13C15N_Clec16aIDPR2_MG_021320/9/pdata/1");
    let hnco_dir = clec.join("13C15N_Clec16aIDPR2_MG_021320/7/pdata/1");
    let hncacb_dir = clec.join("13C15N_Clec16aIDPR2_MG_021320/10/pdata/1");
    let hncocacb_dir = clec.join("13C15N_Clec16aIDPR2_MG_021320/11/pdata/1");
    let cas_ala_dir = clec.join("13C15N_Clec16aIDPR2_MG_091119/10013/pdata/1");
    let cas_gly_dir = clec.join("13C15N_Clec16aIDPR2_MG_091119/10015/pdata/1");
    let cas_ser_dir = clec.join("13C15N_Clec16aIDPR2_MG_021320/10013/pdata/1");
    if !fasta.exists() || !con_dir.exists() || !hsqc_dir.exists() {
        return;
    }

    let seq_txt = std::fs::read_to_string(&fasta).unwrap();
    let seq: Vec<String> = seq_txt
        .lines()
        .filter(|l| !l.starts_with('>'))
        .flat_map(|l| l.trim().chars())
        .filter(|c| c.is_ascii_alphabetic())
        .map(|c| one_to_three(c).to_string())
        .collect();
    assert!(
        seq.len() >= 100,
        "unexpected Clec16a sequence length {}",
        seq.len()
    );

    // Pick 2D CON + CAS
    let c13_cfg = C13PickerConfig::default();
    let sp_con = read_bruker_pdata_2d(&con_dir).unwrap();
    let con_roots = pick_con_2d(&sp_con, &c13_cfg);

    let mut cas_spectra = Vec::new();
    for (label, mode, res_vec, dir) in [
        (
            "cacon_ala",
            CasMode::Cacon,
            vec!["ALA".to_string()],
            &cas_ala_dir,
        ),
        (
            "cacon_gly",
            CasMode::Cacon,
            vec!["GLY".to_string()],
            &cas_gly_dir,
        ),
        (
            "cacon_ser",
            CasMode::Cacon,
            vec!["SER".to_string()],
            &cas_ser_dir,
        ),
    ] {
        if dir.exists() {
            let sp = read_bruker_pdata_2d(dir).unwrap();
            let peaks: Vec<CasPeak> = pick_con_2d(&sp, &c13_cfg)
                .into_iter()
                .map(|p| CasPeak {
                    c_prime_ppm: p.c_prev_ppm,
                    n_ppm: p.n_curr_ppm,
                    height: p.height,
                    volume: p.volume,
                })
                .collect();
            cas_spectra.push(CasSpectrum {
                name: label.to_string(),
                mode,
                residues: res_vec,
                peaks,
            });
        }
    }

    // Pick 2D BEST-HSQC + 3D HNCO / HNCACB / HNCOCACB
    let pcfg = PickerConfig::default();
    let sp_hsqc = read_bruker_pdata_2d(&hsqc_dir).unwrap();
    let hsqc_roots = pick_hsqc_2d(&sp_hsqc, &pcfg);
    let mut hn_peaks = hsqc_roots.clone();
    for (exp, dir) in [
        ("HNCO", &hnco_dir),
        ("HNCACB", &hncacb_dir),
        ("HNCOCACB", &hncocacb_dir),
    ] {
        if dir.exists() {
            let sp3 = read_bruker_pdata_3d(dir, exp).unwrap();
            hn_peaks.extend(pick_triple_3d(&sp3, &hsqc_roots, &pcfg));
        }
    }

    let ds = JointIdpDataset {
        hn_peaks,
        c13_dataset: C13PickedDataset {
            con_roots,
            caco_peaks: Vec::new(),
            strip_peaks: Vec::new(),
        },
        cas_spectra,
        hncacon_nn_edges: Vec::new(),
        hncaconh_peaks: Vec::new(),
        reciprocal_nn_pairs: Vec::new(),
        proton_mhz: 800.0,
        user_locks: Vec::new(),
    };

    let cfg = JointIdpConfig {
        replicas: 12,
        steps_per_system: 800,
        seed: 0,
        ..JointIdpConfig::default()
    };
    let res = assign_joint_idp_cpu(&ds, &seq, &cfg).unwrap();
    let n_fused = res
        .nodes
        .iter()
        .filter(|n| n.kind == JointNodeKind::AmideFused && n.hn_matched && n.con_matched)
        .count();
    let n_pro_nodes = res
        .nodes
        .iter()
        .filter(|n| n.kind == JointNodeKind::ProlineConBridge)
        .count();
    let n_assigned = res
        .residue_assignments
        .iter()
        .filter(|r| r.node_idx.is_some())
        .count();
    let n_hc = res
        .residue_assignments
        .iter()
        .filter(|r| r.node_idx.is_some() && r.confidence >= 0.80)
        .count();
    let (pro_ok, pro_tot) = res.proline_bridge_recovery;

    println!(
        "\n=== Clec16a Real-Data Replay (L={}, CON={}, HSQC={}) ===",
        seq.len(),
        ds.c13_dataset.con_roots.len(),
        hsqc_roots.len()
    );
    println!(
        "Nodes={} (CrossValFused={}, ProBridge={}) | Assigned={} | HC(>=0.8)={} | Pro={pro_ok}/{pro_tot} | Offsets=(dC={:+.2}, dCA={:+.2}, dCB={:+.2}, dN={:+.2}) | QC={} ({:?})",
        res.nodes.len(),
        n_fused,
        n_pro_nodes,
        n_assigned,
        n_hc,
        res.fitted_offsets.d_c,
        res.fitted_offsets.d_ca,
        res.fitted_offsets.d_cb,
        res.fitted_offsets.d_n,
        res.qc_verdict,
        res.qc_reasons,
    );

    assert!(res.score.is_finite(), "Clec16a score must be finite");
    assert!(
        res.confidence
            .iter()
            .all(|c| c.is_finite() && *c >= 0.0 && *c <= 1.0),
        "Clec16a confidences must be finite in [0, 1]"
    );
    assert!(
        n_pro_nodes > 0,
        "Clec16a must detect Proline CON imino nodes"
    );
    assert!(n_fused > 0, "Clec16a must fuse HNCO + CON amide nodes");
    assert!(n_assigned > 0, "Clec16a must place backbone nodes");
}

/// `P2-1`: Verify exact Hungarian bipartite HN↔CON fusion keeps node count bounded under high
/// `0.02 H / 0.15 N / 0.05 C` jitter (no Round-1 greedy explosion to 188–202 nodes!).
#[test]
fn test_p2_1_hungarian_fusion_stable_under_high_noise() {
    let seq = synthetic_pro_ser_rich_idp_sequence();
    let spec = IdpStressSpec {
        jitter_h: 0.02,
        jitter_n: 0.15,
        jitter_c: 0.05,
        random_drop_frac: 0.05,
        n_decoy_con_roots: 0,
        n_decoy_hn_roots: 0,
        tms_c13_offset: 0.0,
        ..IdpStressSpec::default()
    };
    let (ds, _) = synthesize_idp_stress_dataset(&seq, 42, &spec);
    let cfg = JointIdpConfig {
        replicas: 4,
        steps_per_system: 150,
        seed: 42,
        ..JointIdpConfig::default()
    };
    let res = assign_joint_idp_cpu(&ds, &seq, &cfg).unwrap();
    let n_fused = res
        .nodes
        .iter()
        .filter(|n| n.kind == JointNodeKind::AmideFused && n.hn_matched && n.con_matched)
        .count();
    assert!(
        res.nodes.len() <= 165,
        "P2-1: Hungarian fusion must not explode node count under 0.02/0.15/0.05 noise (got {})",
        res.nodes.len()
    );
    assert!(
        n_fused >= 65,
        "P2-1: Hungarian fusion must still fuse >= 65 amide pairs under 0.02/0.15/0.05 noise (got {n_fused})"
    );
}

/// `P2-2`: Verify partner-checked near-diagonal `BT_HNCACONH` (`|N_i - N_partner| <= 0.35 ppm`)
/// links are retained with smooth down-weighting (`0.45..1.0`) rather than hard-dropped.
#[test]
fn test_p2_2_near_diagonal_bt_hncaconh_retained() {
    let seq: Vec<String> = ["MET", "ALA", "SER", "GLY", "GLU", "ALA"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let hn_peaks = vec![
        Peak::hsqc(8.20, 120.00).with_intensity(1.0e5, None),
        Peak::triple("HNCO", 8.20, 120.00, 174.10, 1).with_intensity(1.0e5, None),
        Peak::hsqc(8.45, 120.20).with_intensity(1.0e5, None),
        Peak::triple("HNCO", 8.45, 120.20, 175.80, 1).with_intensity(1.0e5, None),
    ];
    let hncaconh_peaks = vec![
        HncaconhPeak {
            h_ppm: 8.20,
            n_curr_ppm: 120.00,
            n_partner_ppm: 120.20,
            height: 5.0e4,
        },
        HncaconhPeak {
            h_ppm: 8.45,
            n_curr_ppm: 120.20,
            n_partner_ppm: 120.00,
            height: 5.0e4,
        },
    ];
    let ds = JointIdpDataset {
        hn_peaks,
        c13_dataset: C13PickedDataset::default(),
        cas_spectra: Vec::new(),
        hncacon_nn_edges: Vec::new(),
        hncaconh_peaks,
        reciprocal_nn_pairs: Vec::new(),
        proton_mhz: 800.0,
        user_locks: Vec::new(),
    };
    let cfg = JointIdpConfig {
        replicas: 2,
        steps_per_system: 50,
        auto_trosy_calibration: false,
        auto_reference_offset: false,
        hncacb_polarity: HncacbPolarity::Auto,
        ..JointIdpConfig::default()
    };
    let res = assign_joint_idp_cpu(&ds, &seq, &cfg).unwrap();
    let total_recip: usize = res.nodes.iter().map(|n| n.recip_nn_partners.len()).sum();
    assert!(
        total_recip >= 2,
        "P2-2: near-diagonal (0.20 ppm) partner-checked BT_HNCACONH edges must be retained, got {total_recip}"
    );
    for node in &res.nodes {
        for &(_, w) in &node.recip_nn_partners {
            assert!(
                w > 0.40 && w < 0.95,
                "P2-2: near-diagonal BT_HNCACONH weight should be smoothly down-weighted, got {w}"
            );
        }
    }
}

/// `P2-4`: Verify `HNCOCACB`, `CBCA(CO)NH`, `HN(CO)CACB`, `BEST-HSQC` aliases attach cleanly.
#[test]
fn test_p2_4_hn_experiment_name_aliases() {
    let seq: Vec<String> = ["MET", "ALA", "SER", "VAL"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    for exp_name in ["CBCACONH", "HNCOCACB", "CBCA(CO)NH", "HN(CO)CACB"] {
        let ds = JointIdpDataset {
            hn_peaks: vec![
                Peak::triple("BEST-HSQC", 8.25, 118.4, 0.0, 1).with_intensity(1.0e5, None),
                Peak::triple("HNCO", 8.25, 118.4, 175.1, 1).with_intensity(9.0e4, None),
                Peak::triple(exp_name, 8.25, 118.4, 52.4, 1).with_intensity(6.0e4, None),
                Peak::triple(exp_name, 8.25, 118.4, 19.2, 1).with_intensity(5.5e4, None),
            ],
            c13_dataset: C13PickedDataset::default(),
            cas_spectra: Vec::new(),
            hncacon_nn_edges: Vec::new(),
            hncaconh_peaks: Vec::new(),
            reciprocal_nn_pairs: Vec::new(),
            proton_mhz: 800.0,
            user_locks: Vec::new(),
        };
        let cfg = JointIdpConfig {
            replicas: 2,
            steps_per_system: 50,
            auto_reference_offset: false,
            hncacb_polarity: HncacbPolarity::Auto,
            ..JointIdpConfig::default()
        };
        let res = assign_joint_idp_cpu(&ds, &seq, &cfg).unwrap();
        assert_eq!(res.nodes.len(), 1, "expected 1 node for {exp_name}");
        assert!(
            !res.nodes[0].ca_prev.is_empty() && !res.nodes[0].cb_prev.is_empty(),
            "P2-4: alias {exp_name} failed to attach ca_prev/cb_prev"
        );
    }
}

/// `P2-5`: Verify crowded 140-aa IDP HSQC TROSY auto-calibration detects `800 MHz` TROSY cleanly
/// while rejecting null decoupled `(0, 0)` without false triggering.
#[test]
fn test_p2_5_trosy_crowded_idp_hsqc_and_null_rejection() {
    let mut rng = Rng::new(777);
    let mut decoupled_roots = Vec::new();
    for i in 0..120 {
        let h = 7.80 + (i % 15) as f64 * 0.065 + rng.normal(0.0, 0.012);
        let n = 110.0 + (i / 15) as f64 * 2.1 + rng.normal(0.0, 0.15);
        decoupled_roots.push((h, n));
    }

    let (exp_dh, exp_dn) = trosy_expected_shift_ppm(800.0);
    let trosy_roots: Vec<(f64, f64)> = decoupled_roots
        .iter()
        .take(60)
        .map(|&(h, n)| {
            (
                h + exp_dh + rng.normal(0.0, 0.004),
                n + exp_dn + rng.normal(0.0, 0.025),
            )
        })
        .collect();
    let cal_trosy = calibrate_trosy_hn_offset(&trosy_roots, &decoupled_roots, 800.0);
    assert!(
        cal_trosy.trosy_detected && cal_trosy.matched_roots >= MIN_TROSY_MATCHED_ROOTS,
        "P2-5: 800 MHz TROSY must be detected on crowded IDP HSQC: {cal_trosy:?}"
    );
    assert!(
        (cal_trosy.dh_ppm - exp_dh).abs() < 0.015 && (cal_trosy.dn_ppm - exp_dn).abs() < 0.12,
        "P2-5: TROSY shift estimate inaccurate: {cal_trosy:?}"
    );

    // Decoupled null (0, 0) on the exact same crowded HSQC must NOT trigger TROSY detection!
    let null_roots: Vec<(f64, f64)> = decoupled_roots
        .iter()
        .take(60)
        .map(|&(h, n)| (h + rng.normal(0.0, 0.004), n + rng.normal(0.0, 0.025)))
        .collect();
    let cal_null = calibrate_trosy_hn_offset(&null_roots, &decoupled_roots, 800.0);
    assert!(
        !cal_null.trosy_detected,
        "P2-5: decoupled null on crowded IDP HSQC must NOT falsely trigger TROSY: {cal_null:?}"
    );
    // Canonical 800 MHz fallback when non-positive MHz is passed
    assert_eq!(
        trosy_expected_shift_ppm(0.0),
        trosy_expected_shift_ppm(CANONICAL_TROSY_FALLBACK_MHZ)
    );
}

/// `P2-13`: Verify `prune_f1_c_prime_nus_sidelobes` preserves a weak `0.12x` real `C'_{i-1}` peak
/// in a 2-peak doublet at `1.5 ppm` while pruning symmetric 4-peak NUS sidelobes.
#[test]
fn test_p2_13_f1_c_prime_nus_sidelobe_pruning_preserves_weak_doublet() {
    let doublet = vec![(175.50, 100_000.0), (174.00, 12_000.0)];
    let (kept_d, pruned_d) = prune_f1_c_prime_nus_sidelobes(&doublet, 2);
    assert_eq!(
        kept_d.len(),
        2,
        "P2-13: weak 0.12x real C'_(i-1) doublet peak must be preserved, got {kept_d:?}"
    );
    assert_eq!(pruned_d, 0);

    let with_sidelobes = vec![
        (175.50, 100_000.0),
        (173.40, 55_000.0),
        (176.45, 9_000.0),
        (174.55, 8_800.0),
    ];
    let (kept_s, pruned_s) = prune_f1_c_prime_nus_sidelobes(&with_sidelobes, 2);
    assert_eq!(
        kept_s.len(),
        2,
        "P2-13: expected 2 real C' peaks kept, got {kept_s:?}"
    );
    assert_eq!(
        pruned_s, 2,
        "P2-13: expected 2 symmetric NUS sidelobes pruned"
    );
    assert!((kept_s[0].0 - 175.50).abs() < 1e-6);
    assert!((kept_s[1].0 - 173.40).abs() < 1e-6);
}

/// `P2-14`: Verify ALL 4 pre-Proline shift corrections (`C' -1.90, CA -1.80, N +1.65, CB -0.35`)
/// are applied in `predict_joint_ncidp_shifts` and `pre_pro_shift_corrections`.
#[test]
fn test_p2_14_all_four_pre_proline_shifts_applied() {
    let (dn, dc, dca, dcb) = pre_pro_shift_corrections("ALA");
    assert!((dn - PRE_PRO_N_CORRECTION_PPM).abs() < 1e-9);
    assert!((dc - PRE_PRO_C_PRIME_CORRECTION_PPM).abs() < 1e-9);
    assert!((dca - PRE_PRO_CA_CORRECTION_PPM).abs() < 1e-9);
    assert!((dcb - PRE_PRO_CB_CORRECTION_PPM).abs() < 1e-9);
    let (g_dn, g_dc, g_dca, _) = pre_pro_shift_corrections("GLY");
    assert!(g_dn > 0.0 && g_dc < 0.0 && g_dca < 0.0);

    let cfg = JointIdpConfig::default();
    let seq_non_pro: Vec<String> = ["MET", "ALA", "SER", "ALA", "GLU"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let seq_pre_pro: Vec<String> = ["MET", "ALA", "SER", "PRO", "GLU"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let pred_a = predict_joint_ncidp_shifts(&seq_non_pro, &cfg);
    let pred_p = predict_joint_ncidp_shifts(&seq_pre_pro, &cfg);

    let diff_c = pred_p[2].c_ppm - pred_a[2].c_ppm;
    let diff_ca = pred_p[2].ca_ppm - pred_a[2].ca_ppm;
    let diff_n = pred_p[2].n_ppm - pred_a[2].n_ppm;
    let diff_cb = pred_p[2].cb_ppm.unwrap() - pred_a[2].cb_ppm.unwrap();
    assert!(
        diff_c < -1.0,
        "expected pre-Pro C' shift < -1.0 ppm, got {diff_c:+.2}"
    );
    assert!(
        diff_ca < -1.0,
        "expected pre-Pro CA shift < -1.0 ppm, got {diff_ca:+.2}"
    );
    assert!(
        diff_n > 0.8,
        "expected pre-Pro N shift > +0.8 ppm, got {diff_n:+.2}"
    );
    assert!(
        diff_cb < -0.1,
        "expected pre-Pro CB shift < -0.1 ppm, got {diff_cb:+.2}"
    );
}

fn make_minimal_node(
    id: usize,
    kind: JointNodeKind,
    c_prev: Option<f64>,
    n_curr: f64,
) -> JointIdpNode {
    JointIdpNode {
        id,
        kind,
        hn_matched: kind == JointNodeKind::AmideFused,
        con_matched: true,
        h_curr: if kind == JointNodeKind::AmideFused {
            Some(8.15)
        } else {
            None
        },
        n_curr,
        hn_n_curr: if kind == JointNodeKind::AmideFused {
            Some(n_curr)
        } else {
            None
        },
        c_prev,
        root_height: 5.0e4,
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
    }
}

#[test]
fn test_gly_cb_bleedthrough_guard_and_cterm_coo_anchor() {
    let mut gly_node = make_minimal_node(0, JointNodeKind::AmideFused, Some(174.0), 110.0);
    gly_node.ca_curr = vec![45.1];
    gly_node.cb_curr = vec![45.4];
    let mut ser_node = make_minimal_node(1, JointNodeKind::AmideFused, Some(174.5), 116.0);
    ser_node.ca_curr = vec![58.2];
    ser_node.cb_curr = vec![63.8];

    let mut nodes = vec![gly_node, ser_node];
    let guarded = apply_gly_cb_bleedthrough_guard(&mut nodes, &IdpShiftOffsets::default());
    assert_eq!(guarded, 1);
    assert!(nodes[0].cb_curr.is_empty() && nodes[0].gly_cb_curr_suppressed);
    assert_eq!(nodes[1].cb_curr, vec![63.8]);

    // C-terminal COO- anchor (`~179.3 ppm` DSS) must strongly prefer final residue `L-1`
    let seq: Vec<String> = ["MET", "ALA", "SER", "GLU", "ALA"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let cfg = JointIdpConfig::default();
    let preds = predict_joint_ncidp_shifts(&seq, &cfg);
    let mut node_cterm = make_minimal_node(
        0,
        JointNodeKind::AmideFused,
        Some(preds[3].c_ppm),
        preds[4].n_ppm,
    );
    node_cterm.c_curr = vec![179.30];
    node_cterm.ca_curr = vec![preds[4].ca_ppm];
    node_cterm.cb_curr = preds[4].cb_ppm.into_iter().collect();
    node_cterm.ca_prev = vec![preds[3].ca_ppm];
    node_cterm.cb_prev = preds[3].cb_ppm.into_iter().collect();

    let u = joint_idp_unary_matrix(
        &[node_cterm],
        &seq,
        &[],
        &cfg,
        &IdpShiftOffsets::default(),
        &[],
    );
    assert!(
        u.get(0, 4) > u.get(0, 1) + 2.0,
        "C-terminal COO- (179.3 ppm) must score strongly higher at L-1 ({:.2}) vs internal ALA ({:.2})",
        u.get(0, 4),
        u.get(0, 1)
    );
}

#[test]
fn test_calibrated_cas_overlap_and_breakthrough_damping() {
    let seq: Vec<String> = ["MET", "SER", "ALA", "SER", "GLY", "GLU", "VAL", "ALA"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let nodes = vec![
        make_minimal_node(0, JointNodeKind::ConOnlyAmide, Some(174.00), 118.00),
        make_minimal_node(1, JointNodeKind::ConOnlyAmide, Some(175.00), 120.00),
        make_minimal_node(2, JointNodeKind::ConOnlyAmide, Some(175.02), 120.05),
        make_minimal_node(3, JointNodeKind::ConOnlyAmide, Some(176.00), 122.00),
        make_minimal_node(4, JointNodeKind::ConOnlyAmide, Some(177.00), 124.00),
    ];
    let clean_cas = vec![CasSpectrum {
        name: "cacon_ser".into(),
        mode: CasMode::Cacon,
        residues: vec!["SER".into()],
        peaks: vec![
            CasPeak {
                c_prime_ppm: 174.00,
                n_ppm: 118.00,
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
    }];
    let cfg = JointIdpConfig::default();
    let cal_clean = build_calibrated_cas_tables(&clean_cas, &nodes, &seq, &cfg);
    assert!(
        cal_clean[0].hit_weight[0] > cal_clean[0].hit_weight[1],
        "overlapped CAS peak must be damped vs isolated peak"
    );

    let noisy_cas = vec![CasSpectrum {
        name: "cacon_ser_noisy".into(),
        mode: CasMode::Cacon,
        residues: vec!["SER".into()],
        peaks: vec![
            CasPeak {
                c_prime_ppm: 174.00,
                n_ppm: 118.00,
                height: 6.0e4,
                volume: 6.0e4,
            },
            CasPeak {
                c_prime_ppm: 175.00,
                n_ppm: 120.00,
                height: 6.0e4,
                volume: 6.0e4,
            },
            CasPeak {
                c_prime_ppm: 175.02,
                n_ppm: 120.05,
                height: 6.0e4,
                volume: 6.0e4,
            },
            CasPeak {
                c_prime_ppm: 176.00,
                n_ppm: 122.00,
                height: 6.0e4,
                volume: 6.0e4,
            },
            CasPeak {
                c_prime_ppm: 177.00,
                n_ppm: 124.00,
                height: 6.0e4,
                volume: 6.0e4,
            },
        ],
    }];
    let cal_noisy = build_calibrated_cas_tables(&noisy_cas, &nodes, &seq, &cfg);
    assert!(
        cal_noisy[0].p_fp_eff > cal_clean[0].p_fp_eff,
        "breakthrough CAS must raise p_fp_eff ({:.3} vs {:.3})",
        cal_noisy[0].p_fp_eff,
        cal_clean[0].p_fp_eff
    );
}

#[test]
fn test_hostile_empty_nan_and_matrix_cap_fail_closed() {
    let empty = JointIdpDataset::default();
    let cfg = JointIdpConfig::default();
    let res_empty = assign_joint_idp(&empty, &[], &cfg).unwrap();
    assert!(res_empty.residue_assignments.is_empty());
    assert_eq!(res_empty.qc_verdict, DatasetQcVerdict::Abstain);

    let seq: Vec<String> = ["MET", "ALA", "PRO", "SER"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let hostile = JointIdpDataset {
        hn_peaks: vec![
            Peak::hsqc(f64::NAN, 120.0),
            Peak::triple("HNCO", 8.2, f64::INFINITY, 175.0, 1),
        ],
        c13_dataset: C13PickedDataset {
            con_roots: vec![Con2DPeak {
                c_prev_ppm: f64::NAN,
                n_curr_ppm: 136.0,
                height: f64::NEG_INFINITY,
                volume: 0.0,
                is_proline_imino_region: true,
                deconvolved_from_cluster: false,
            }],
            caco_peaks: Vec::new(),
            strip_peaks: Vec::new(),
        },
        cas_spectra: Vec::new(),
        hncacon_nn_edges: vec![DirectionalNnPeak {
            c_prev_ppm: f64::NAN,
            n_curr_ppm: 120.0,
            n_prev_ppm: 118.0,
            snr: 1.0,
        }],
        hncaconh_peaks: vec![HncaconhPeak {
            h_ppm: f64::NAN,
            n_curr_ppm: 120.0,
            n_partner_ppm: 118.0,
            height: 1.0,
        }],
        reciprocal_nn_pairs: Vec::new(),
        proton_mhz: f64::NAN,
        user_locks: Vec::new(),
    };
    let res_h = assign_joint_idp_cpu(&hostile, &seq, &cfg).unwrap();
    assert!(res_h.score.is_finite());
    assert!(res_h.confidence.iter().all(|c| c.is_finite()));

    let huge_seq = vec!["ALA".to_string(); 5001];
    let dummy_node = make_minimal_node(0, JointNodeKind::ConOnlyAmide, Some(175.0), 120.0);
    let huge_nodes = vec![dummy_node; (MAX_IDP_MATRIX_CELLS / 5001) + 2];
    assert!(
        try_joint_idp_unary_matrix(
            &huge_nodes,
            &huge_seq,
            &[],
            &cfg,
            &IdpShiftOffsets::default(),
            &[]
        )
        .is_err(),
        "matrix cell overflow must fail closed"
    );
}

#[test]
fn test_gui_cli_canonical_exports_and_summaries() {
    assert!((PROLINE_N15_MIN_PPM - 131.5).abs() < 1e-9);
    assert!((PROLINE_N15_MAX_PPM - 144.5).abs() < 1e-9);
    assert!(JOINT_IDP_CALIBRATION_KNOTS.len() >= 5);
    for w in JOINT_IDP_CALIBRATION_KNOTS.windows(2) {
        assert!(
            w[1].0 > w[0].0,
            "calibration knot x must be strictly increasing"
        );
        assert!(
            w[1].1 >= w[0].1,
            "calibration knot y must be monotone non-decreasing"
        );
    }
    assert!(calibrate_joint_posterior(0.0) <= calibrate_joint_posterior(0.5));
    assert!(calibrate_joint_posterior(0.5) <= calibrate_joint_posterior(1.0));

    let seq = synthetic_pro_ser_rich_idp_sequence();
    let (ds, _) = synthesize_idp_stress_dataset(&seq, 5, &IdpStressSpec::default());
    let cfg = JointIdpConfig {
        replicas: 6,
        steps_per_system: 300,
        seed: 5,
        ..JointIdpConfig::default()
    };
    let res = assign_joint_idp_cpu(&ds, &seq, &cfg).unwrap();
    let bridges = res.proline_bridge_summaries();
    assert_eq!(
        bridges.len(),
        21,
        "expected 21 Proline bridge summary entries"
    );
    let cterms = res.cterm_anchor_summaries();
    assert!(
        cterms.iter().any(|c| c.assigned_to_cterm),
        "expected C-terminal COO- anchor summary at final residue, got {cterms:?}"
    );
    let seq_ids: Vec<i32> = (1..=seq.len() as i32).collect();
    let (ref_out, conf_map): (_, BTreeMap<usize, f64>) =
        joint_idp_to_reference(&res, &seq, &seq_ids);
    assert_eq!(ref_out.sequence.len(), seq.len());
    assert!(!ref_out.shifts.is_empty());
    assert!(conf_map.values().all(|&v| (0.0..=1.0).contains(&v)));
    assert!(res.route_support.iter().any(|r| {
        matches!(
            r,
            Some(
                JointRouteSupport::CrossValidatedHnAndCon
                    | JointRouteSupport::HnRoute
                    | JointRouteSupport::ConProBridge
                    | JointRouteSupport::ConRouteOnly
            )
        )
    }));
}
