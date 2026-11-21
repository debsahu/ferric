//! Integration tests for `nmr_core::classify` and `POST /api/v1/inspect` (`T09.1`).
//!
//! Covers:
//! - Synthetic Bruker & Varian header trees (including misnamed folders, `pulsecal` 1D decoy,
//!   `FnTYPE=0` template false-alarm guard, NUS density from `nuslist`)
//! - `Pseudo3D` relaxation & exchange (`vdlist` -> `RelaxR1`, `vclist` -> `Cpmg`, `fqlist` -> `Cest`,
//!   Varian `array = "d2"`)
//! - Protonless `13C`-detected IDP suite (`Con2D`, `Caco2D`, `Canco3D`, `Cbcaco3D` -> `Idp13cDetected`)
//! - Methyl-TROSY `13C-HMQC` (`MethylTrosy`), paramagnetic sweep (`Paramagnetic`), and ligand
//!   titration series (`TitrationCsp` + `TitrationSeries`)
//! - Processed Sparky `.ucsf` and multi-plane NMRPipe `.ft3` binary headers
//! - POSIX `.tar` and RFC 1952 `.tar.gz` (stored & fixed-Huffman LZ77) archives with multi-MB `ser`
//!   skipping (`< 50 ms`) and `wasm32`-compatible `classify_nmr_entries` parity
//! - `POST /api/v1/inspect` HTTP endpoint (JSON `path`, JSON `entries`, and binary `.tar.gz` body)
//! - Conditional real header benchmark (`< 50 ms`, `100%` accuracy) on `Ubiquitin`, `CISD3` (`51439`),
//!   `53371`, `18465`, `5329`, and `6318` when `NMR_REAL_DATA_DIR` or local `data/` is present.

use std::env;
use std::fs;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process;
use std::sync::atomic::{AtomicUsize, Ordering};

use nmr_core::classify::{
    classify_nmr_entries, classify_nmr_workspace, extract_archive_full_entries_bounded,
    extract_archive_virtual_entries, extract_archive_virtual_entries_bounded, CasSelectionScope,
    DataState, ExperimentDimensionality, ExperimentVendor, ProblemRegime, PulseSequenceClass,
    QuadratureKind, VirtualEntry, WorkspaceInventory,
};
use nmr_core::service::{
    spawn_background_server, spawn_background_server_with_options, ServiceOptions,
};

const MAX_HEADER_SCAN_MS: f64 = if cfg!(debug_assertions) {
    2500.0
} else {
    500.0
};

static NEXT_TMP_ID: AtomicUsize = AtomicUsize::new(1);

fn make_temp_dir(prefix: &str) -> PathBuf {
    let n = NEXT_TMP_ID.fetch_add(1, Ordering::SeqCst);
    let dir = env::temp_dir().join(format!("nmr_classify_{prefix}_{}_{n}", process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn write_text(root: &Path, rel: &str, content: &str) {
    let p = root.join(rel);
    if let Some(parent) = p.parent() {
        fs::create_dir_all(parent).expect("create parent dir");
    }
    fs::write(p, content.as_bytes()).expect("write text file");
}

fn write_bytes(root: &Path, rel: &str, bytes: &[u8]) {
    let p = root.join(rel);
    if let Some(parent) = p.parent() {
        fs::create_dir_all(parent).expect("create parent dir");
    }
    fs::write(p, bytes).expect("write binary file");
}

fn bruker_acqus(
    pulprog: &str,
    nuc1: &str,
    sfo1: f64,
    sw_ppm: f64,
    td: usize,
    fn_type: usize,
) -> String {
    format!(
        "##TITLE= Parameter file, TopSpin 4.1\n\
         ##$PULPROG= <{pulprog}>\n\
         ##$NUC1= <{nuc1}>\n\
         ##$SFO1= {sfo1:.6}\n\
         ##$BF1= {sfo1:.6}\n\
         ##$SW= {sw_ppm:.4}\n\
         ##$SW_h= {:.3}\n\
         ##$TD= {td}\n\
         ##$FnMODE= 0\n\
         ##$FnTYPE= {fn_type}\n\
         ##$NusAMOUNT= 25\n\
         ##END=\n",
        sw_ppm * sfo1
    )
}

fn bruker_indirect_acqus(nuc1: &str, sfo1: f64, sw_ppm: f64, td: usize, fn_mode: usize) -> String {
    format!(
        "##TITLE= Indirect parameter file\n\
         ##$NUC1= <{nuc1}>\n\
         ##$SFO1= {sfo1:.6}\n\
         ##$BF1= {sfo1:.6}\n\
         ##$SW= {sw_ppm:.4}\n\
         ##$SW_h= {:.3}\n\
         ##$TD= {td}\n\
         ##$FnMODE= {fn_mode}\n\
         ##END=\n",
        sw_ppm * sfo1
    )
}

#[allow(clippy::too_many_arguments)]
fn varian_procpar_3d(
    seqfil: &str,
    sfrq: f64,
    dfrq: f64,
    dfrq2: f64,
    dn: &str,
    dn2: &str,
    sw: f64,
    sw1: f64,
    sw2: f64,
    ni: usize,
    ni2: usize,
) -> String {
    format!(
        "seqfil 2 2 8192 0 0 2 1 1 1 64\n\
         1 \"{seqfil}\"\n\
         0\n\
         pslabel 2 2 8192 0 0 2 1 1 1 64\n\
         1 \"{seqfil}\"\n\
         0\n\
         tn 2 2 8192 0 0 2 1 1 1 64\n\
         1 \"H1\"\n\
         0\n\
         dn 2 2 8192 0 0 2 1 1 1 64\n\
         1 \"{dn}\"\n\
         0\n\
         dn2 2 2 8192 0 0 2 1 1 1 64\n\
         1 \"{dn2}\"\n\
         0\n\
         sfrq 1 1 10000 0 0 2 1 1 1 64\n\
         1 {sfrq:.6}\n\
         0\n\
         dfrq 1 1 10000 0 0 2 1 1 1 64\n\
         1 {dfrq:.6}\n\
         0\n\
         dfrq2 1 1 10000 0 0 2 1 1 1 64\n\
         1 {dfrq2:.6}\n\
         0\n\
         np 1 1 100000 0 0 2 1 1 1 64\n\
         1 2048\n\
         0\n\
         ni 1 1 100000 0 0 2 1 1 1 64\n\
         1 {ni}\n\
         0\n\
         ni2 1 1 100000 0 0 2 1 1 1 64\n\
         1 {ni2}\n\
         0\n\
         sw 1 1 1000000 0 0 2 1 1 1 64\n\
         1 {sw:.2}\n\
         0\n\
         sw1 1 1 1000000 0 0 2 1 1 1 64\n\
         1 {sw1:.2}\n\
         0\n\
         sw2 1 1 1000000 0 0 2 1 1 1 64\n\
         1 {sw2:.2}\n\
         0\n\
         phase 1 1 10 0 0 2 1 1 1 64\n\
         2 1 2\n\
         0\n\
         phase2 1 1 10 0 0 2 1 1 1 64\n\
         2 1 2\n\
         0\n"
    )
}

#[test]
fn test_synthetic_misnamed_bruker_and_varian_headers_and_nus() {
    let root = make_temp_dir("misnamed_headers");

    // 1. Misnamed Bruker folder ("10_hnco_wrong_label") that is physically a 50% NUS 3D HNCACB
    //    with BOTH raw `ser` and processed `pdata/1/3rrr`.
    write_text(
        &root,
        "10_hnco_wrong_label/acqus",
        &bruker_acqus("hncacbgp3d", "1H", 800.13, 14.0, 2048, 2),
    );
    write_text(
        &root,
        "10_hnco_wrong_label/acqu2s",
        &bruker_indirect_acqus("13C", 201.19, 72.0, 160, 5), // States-TPPI, 80 hypercomplex
    );
    write_text(
        &root,
        "10_hnco_wrong_label/acqu3s",
        &bruker_indirect_acqus("15N", 81.08, 32.0, 80, 6), // Echo-Antiecho, 40 hypercomplex => 3200 grid
    );
    let mut nuslist = String::new();
    for idx in 0..1600 {
        nuslist.push_str(&format!("{} {}\n", idx % 80, idx / 80));
    }
    write_text(&root, "10_hnco_wrong_label/nuslist", &nuslist);
    write_bytes(&root, "10_hnco_wrong_label/ser", &[0u8; 512]);
    write_text(
        &root,
        "10_hnco_wrong_label/pdata/1/procs",
        "##$SI= 1024\n##END=\n",
    );
    write_bytes(&root, "10_hnco_wrong_label/pdata/1/3rrr", &[0u8; 1024]);

    // 2. Uniform 2D 15N-HSQC where Bruker template left `NusAMOUNT=25` even though `FnTYPE=0`.
    write_text(
        &root,
        "20_uniform_hsqc/acqus",
        &bruker_acqus("fhsqcf3gpph", "1H", 800.13, 14.0, 2048, 0),
    );
    write_text(
        &root,
        "20_uniform_hsqc/acqu2s",
        &bruker_indirect_acqus("15N", 81.08, 34.0, 256, 6),
    );
    write_bytes(&root, "20_uniform_hsqc/ser", &[0u8; 256]);

    // 3. 1D `pulsecal` folder `99999` with a stale cloned title "3D HNCO" — must stay 1D Unknown & unchecked!
    write_text(
        &root,
        "99999/acqus",
        &bruker_acqus("pulsecal", "1H", 800.13, 14.0, 4096, 0),
    );
    write_text(&root, "99999/pdata/1/title", "Cloned 3D HNCO template\n");
    write_bytes(&root, "99999/fid", &[0u8; 128]);

    // 4. Misnamed Varian folder ("mystery_99.fid") containing `gcbca_co_nh`
    write_text(
        &root,
        "mystery_99.fid/procpar",
        &varian_procpar_3d(
            "gcbca_co_nh",
            799.85,
            201.12,
            81.05,
            "C13",
            "N15",
            11000.0,
            14500.0,
            2500.0,
            64,
            32,
        ),
    );
    write_text(&root, "mystery_99.fid/text", "Misnamed folder test\n");
    write_bytes(&root, "mystery_99.fid/fid", &[0u8; 256]);

    // 5. Auxiliary FASTA + NMR-STAR
    write_text(
        &root,
        "target.fasta",
        ">target_protein\nMQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG\n",
    );
    write_text(&root, "reference.str", "data_synthetic\n");

    let inv = classify_nmr_workspace(&root).expect("classify synthetic tree");
    assert_eq!(inv.problem_regime, ProblemRegime::GlobularBackbone);
    assert!(
        inv.missing_input_hints.is_empty(),
        "expected zero missing hints, got {:?}",
        inv.missing_input_hints
    );
    assert_eq!(inv.auxiliary_files.len(), 2);

    let hncacb = inv
        .experiments
        .iter()
        .find(|e| e.relative_path == "10_hnco_wrong_label")
        .expect("find 10_hnco_wrong_label");
    assert_eq!(hncacb.vendor, ExperimentVendor::Bruker);
    assert_eq!(hncacb.dimensionality, ExperimentDimensionality::Dim3D);
    assert_eq!(hncacb.data_state, DataState::Both);
    assert_eq!(hncacb.pulse_class, PulseSequenceClass::Hncacb);
    assert!(hncacb.nus.is_nus);
    assert_eq!(hncacb.nus.sampled_points, Some(1600));
    assert_eq!(hncacb.nus.full_grid_points, Some(3200));
    assert!((hncacb.nus.sampling_density.unwrap() - 0.50).abs() < 1e-6);
    assert!(
        hncacb.nus.badge.contains("NUS") && hncacb.nus.badge.contains("50"),
        "unexpected NUS badge: {}",
        hncacb.nus.badge
    );
    assert_eq!(hncacb.axes[1].quadrature, QuadratureKind::StatesTppi);
    assert_eq!(hncacb.axes[2].quadrature, QuadratureKind::EchoAntiecho);
    assert_eq!(hncacb.raw_signal_bytes, 512);
    assert_eq!(hncacb.processed_bytes, 1024);
    assert!(hncacb.selected_by_default);

    let hsqc = inv
        .experiments
        .iter()
        .find(|e| e.relative_path == "20_uniform_hsqc")
        .expect("find 20_uniform_hsqc");
    assert_eq!(hsqc.pulse_class, PulseSequenceClass::N15Hsqc);
    assert!(
        !hsqc.nus.is_nus,
        "FnTYPE=0 must override template NusAMOUNT"
    );
    assert_eq!(hsqc.nus.badge, "Uniform");

    let pcal = inv
        .experiments
        .iter()
        .find(|e| e.relative_path == "99999")
        .expect("find pulsecal");
    assert_eq!(pcal.dimensionality, ExperimentDimensionality::Dim1D);
    assert_eq!(pcal.pulse_class, PulseSequenceClass::Unknown);
    assert!(!pcal.selected_by_default);

    let varian = inv
        .experiments
        .iter()
        .find(|e| e.relative_path == "mystery_99.fid")
        .expect("find mystery_99.fid");
    assert_eq!(varian.vendor, ExperimentVendor::Varian);
    assert_eq!(varian.dimensionality, ExperimentDimensionality::Dim3D);
    assert_eq!(varian.pulse_class, PulseSequenceClass::Cbcaconh);
    assert!((varian.spectrometer_mhz.unwrap() - 799.85).abs() < 0.1);

    let _ = fs::remove_dir_all(&root);
}

#[test]
fn test_pseudo3d_relaxation_cpmg_cest_and_varian_array() {
    let root = make_temp_dir("pseudo3d_dynamics");

    // Bruker R1 via vdlist
    write_text(
        &root,
        "10_r1/acqus",
        &bruker_acqus("hsqct1etf3gpsi3d", "1H", 600.13, 14.0, 2048, 0),
    );
    write_text(
        &root,
        "10_r1/acqu2s",
        &bruker_indirect_acqus("15N", 60.81, 32.0, 128, 6),
    );
    write_text(
        &root,
        "10_r1/acqu3s",
        &bruker_indirect_acqus("off", 600.13, 1.0, 10, 1),
    );
    write_text(
        &root,
        "10_r1/vdlist",
        "0.01\n0.05\n0.10\n0.20\n0.40\n0.60\n0.80\n1.00\n1.20\n1.50\n",
    );
    write_bytes(&root, "10_r1/ser", &[0u8; 128]);

    // Bruker CPMG via vclist
    write_text(
        &root,
        "20_cpmg/acqus",
        &bruker_acqus("hsqccpmg", "1H", 600.13, 14.0, 2048, 0),
    );
    write_text(
        &root,
        "20_cpmg/acqu2s",
        &bruker_indirect_acqus("15N", 60.81, 32.0, 128, 6),
    );
    write_text(
        &root,
        "20_cpmg/acqu3s",
        &bruker_indirect_acqus("off", 600.13, 1.0, 12, 1),
    );
    write_text(
        &root,
        "20_cpmg/vclist",
        "0\n2\n4\n6\n8\n10\n12\n16\n20\n24\n32\n40\n",
    );
    write_bytes(&root, "20_cpmg/ser", &[0u8; 128]);

    // Bruker CEST via fqlist
    write_text(
        &root,
        "30_cest/acqus",
        &bruker_acqus("hsqc_cest_3d", "1H", 600.13, 14.0, 2048, 0),
    );
    write_text(
        &root,
        "30_cest/acqu2s",
        &bruker_indirect_acqus("15N", 60.81, 32.0, 128, 6),
    );
    write_text(
        &root,
        "30_cest/acqu3s",
        &bruker_indirect_acqus("off", 600.13, 1.0, 8, 1),
    );
    write_text(
        &root,
        "30_cest/fqlist",
        "105.0\n110.0\n115.0\n120.0\n125.0\n130.0\n135.0\n140.0\n",
    );
    write_bytes(&root, "30_cest/ser", &[0u8; 128]);

    // Varian R1 array = "d2"
    write_text(
        &root,
        "40_varian_r1.fid/procpar",
        "seqfil 2 2 8192 0 0 2 1 1 1 64\n1 \"gNhsqc\"\n0\n\
         tn 2 2 8192 0 0 2 1 1 1 64\n1 \"H1\"\n0\n\
         dn 2 2 8192 0 0 2 1 1 1 64\n1 \"N15\"\n0\n\
         dn2 2 2 8192 0 0 2 1 1 1 64\n1 \"\"\n0\n\
         sfrq 1 1 10000 0 0 2 1 1 1 64\n1 599.91\n0\n\
         dfrq 1 1 10000 0 0 2 1 1 1 64\n1 60.79\n0\n\
         dfrq2 1 1 10000 0 0 2 1 1 1 64\n1 60.79\n0\n\
         np 1 1 100000 0 0 2 1 1 1 64\n1 2048\n0\n\
         ni 1 1 100000 0 0 2 1 1 1 64\n1 128\n0\n\
         ni2 1 1 100000 0 0 2 1 1 1 64\n1 1\n0\n\
         sw 1 1 1000000 0 0 2 1 1 1 64\n1 8400.0\n0\n\
         sw1 1 1 1000000 0 0 2 1 1 1 64\n1 2000.0\n0\n\
         phase 1 1 10 0 0 2 1 1 1 64\n2 1 2\n0\n\
         array 2 2 8192 0 0 2 1 1 1 64\n1 \"d2\"\n0\n\
         d2 1 1 10000 0 0 2 1 1 1 64\n6 0.01 0.05 0.1 0.2 0.5 1.0\n0\n",
    );
    write_bytes(&root, "40_varian_r1.fid/fid", &[0u8; 128]);

    let inv = classify_nmr_workspace(&root).expect("classify dynamics workspace");
    assert_eq!(inv.problem_regime, ProblemRegime::RelaxationDynamics);

    let by_path = |p: &str| {
        inv.experiments
            .iter()
            .find(|e| e.relative_path == p)
            .unwrap_or_else(|| panic!("missing {p}"))
    };
    assert_eq!(
        by_path("10_r1").dimensionality,
        ExperimentDimensionality::Pseudo3D
    );
    assert_eq!(by_path("10_r1").pulse_class, PulseSequenceClass::RelaxR1);

    assert_eq!(
        by_path("20_cpmg").dimensionality,
        ExperimentDimensionality::Pseudo3D
    );
    assert_eq!(by_path("20_cpmg").pulse_class, PulseSequenceClass::Cpmg);

    assert_eq!(
        by_path("30_cest").dimensionality,
        ExperimentDimensionality::Pseudo3D
    );
    assert_eq!(by_path("30_cest").pulse_class, PulseSequenceClass::Cest);

    assert_eq!(
        by_path("40_varian_r1.fid").dimensionality,
        ExperimentDimensionality::Pseudo3D
    );
    assert_eq!(
        by_path("40_varian_r1.fid").pulse_class,
        PulseSequenceClass::RelaxR1
    );

    let _ = fs::remove_dir_all(&root);
}

#[test]
fn test_13c_detected_idp_methyl_trosy_paramagnetic_and_titration_regimes() {
    // A. Protonless 13C-detected IDP workspace (`Con2D`, `Caco2D`, `Canco3D`, `Cbcaco3D`)
    let idp_dir = make_temp_dir("idp_13c_detected");
    write_text(
        &idp_dir,
        "1_con/acqus",
        &bruker_acqus("c_con_iasq", "13C", 201.20, 40.0, 2048, 0),
    );
    write_text(
        &idp_dir,
        "1_con/acqu2s",
        &bruker_indirect_acqus("15N", 81.08, 36.0, 256, 5),
    );
    write_bytes(&idp_dir, "1_con/ser", &[0u8; 64]);

    write_text(
        &idp_dir,
        "2_caco/acqus",
        &bruker_acqus("c_caco_ia", "13C", 201.20, 40.0, 2048, 0),
    );
    write_text(
        &idp_dir,
        "2_caco/acqu2s",
        &bruker_indirect_acqus("13C", 201.20, 35.0, 256, 5),
    );
    write_bytes(&idp_dir, "2_caco/ser", &[0u8; 64]);

    write_text(
        &idp_dir,
        "3_canco/acqus",
        &bruker_acqus("c_canco_ia3d", "13C", 201.20, 40.0, 2048, 0),
    );
    write_text(
        &idp_dir,
        "3_canco/acqu2s",
        &bruker_indirect_acqus("15N", 81.08, 36.0, 96, 5),
    );
    write_text(
        &idp_dir,
        "3_canco/acqu3s",
        &bruker_indirect_acqus("13C", 201.20, 35.0, 128, 5),
    );
    write_bytes(&idp_dir, "3_canco/ser", &[0u8; 64]);

    write_text(
        &idp_dir,
        "4_cbcaco/acqus",
        &bruker_acqus("c_cbcaco_ia3d", "13C", 201.20, 40.0, 2048, 0),
    );
    write_text(
        &idp_dir,
        "4_cbcaco/acqu2s",
        &bruker_indirect_acqus("13C", 201.20, 70.0, 128, 5),
    );
    write_text(
        &idp_dir,
        "4_cbcaco/acqu3s",
        &bruker_indirect_acqus("13C", 201.20, 35.0, 96, 5),
    );
    write_bytes(&idp_dir, "4_cbcaco/ser", &[0u8; 64]);

    let idp_inv = classify_nmr_workspace(&idp_dir).expect("classify 13C-detected IDP");
    assert_eq!(idp_inv.problem_regime, ProblemRegime::Idp13cDetected);
    let classes: Vec<PulseSequenceClass> =
        idp_inv.experiments.iter().map(|e| e.pulse_class).collect();
    assert_eq!(
        classes,
        vec![
            PulseSequenceClass::Con2D,
            PulseSequenceClass::Caco2D,
            PulseSequenceClass::Canco3D,
            PulseSequenceClass::Cbcaco3D
        ]
    );
    // Spectrometer base 1H MHz inferred from 13C 201.20 MHz (~800 MHz)
    let mhz = idp_inv.experiments[0].spectrometer_mhz.unwrap();
    assert!((mhz - 800.0).abs() < 2.0, "expected ~800 MHz, got {mhz}");
    let _ = fs::remove_dir_all(&idp_dir);

    // B. Methyl-TROSY 13C-HMQC workspace (`MethylTrosy`)
    let me_dir = make_temp_dir("methyl_trosy");
    write_text(
        &me_dir,
        "1_methyl/acqus",
        &bruker_acqus("hmqcphpr", "1H", 900.21, 12.0, 2048, 0),
    );
    write_text(
        &me_dir,
        "1_methyl/acqu2s",
        &bruker_indirect_acqus("13C", 226.38, 22.0, 256, 5),
    );
    write_bytes(&me_dir, "1_methyl/ser", &[0u8; 64]);
    let me_inv = classify_nmr_workspace(&me_dir).expect("classify methyl-TROSY");
    assert_eq!(
        me_inv.experiments[0].pulse_class,
        PulseSequenceClass::C13Hmqc
    );
    assert_eq!(me_inv.problem_regime, ProblemRegime::MethylTrosy);
    let _ = fs::remove_dir_all(&me_dir);

    // C. Paramagnetic ultra-wide 1H sweep (`SW = 45 ppm`)
    let para_dir = make_temp_dir("paramagnetic");
    write_text(
        &para_dir,
        "1_para/acqus",
        &bruker_acqus("hsqcetfpf3gpsi", "1H", 600.13, 45.0, 4096, 0),
    );
    write_text(
        &para_dir,
        "1_para/acqu2s",
        &bruker_indirect_acqus("15N", 60.81, 40.0, 256, 6),
    );
    write_bytes(&para_dir, "1_para/ser", &[0u8; 64]);
    let para_inv = classify_nmr_workspace(&para_dir).expect("classify paramagnetic");
    assert_eq!(para_inv.problem_regime, ProblemRegime::Paramagnetic);
    let _ = fs::remove_dir_all(&para_dir);

    // D. Ligand titration CSP series (`0uM`, `50uM`, `150uM`, `500uM`)
    let tit_dir = make_temp_dir("titration_csp");
    for conc in ["01_0uM", "02_50uM", "03_150uM", "04_500uM"] {
        write_text(
            &tit_dir,
            &format!("{conc}/acqus"),
            &bruker_acqus("hsqcetfpf3gpsi", "1H", 700.13, 14.0, 2048, 0),
        );
        write_text(
            &tit_dir,
            &format!("{conc}/acqu2s"),
            &bruker_indirect_acqus("15N", 70.95, 32.0, 256, 6),
        );
        write_bytes(&tit_dir, &format!("{conc}/ser"), &[0u8; 64]);
    }
    let tit_inv = classify_nmr_workspace(&tit_dir).expect("classify titration series");
    assert_eq!(tit_inv.problem_regime, ProblemRegime::TitrationCsp);
    assert!(tit_inv
        .experiments
        .iter()
        .all(|e| e.pulse_class == PulseSequenceClass::TitrationSeries));
    let _ = fs::remove_dir_all(&tit_dir);
}

fn build_synthetic_ucsf_3d(
    n0: (&str, u32, f32, f32, f32),
    n1: (&str, u32, f32, f32, f32),
    n2: (&str, u32, f32, f32, f32),
) -> Vec<u8> {
    let mut buf = vec![0u8; 180 + 3 * 128];
    buf[..8].copy_from_slice(b"UCSF NMR");
    buf[10] = 3; // ndim
    buf[13] = 2; // version 2
    for (idx, &(nuc, size, mhz, sw_hz, orig_ppm)) in [n0, n1, n2].iter().enumerate() {
        let base = 180 + idx * 128;
        let nb = nuc.as_bytes();
        buf[base..base + nb.len()].copy_from_slice(nb);
        buf[base + 8..base + 12].copy_from_slice(&size.to_be_bytes());
        buf[base + 16..base + 20].copy_from_slice(&8u32.to_be_bytes());
        buf[base + 20..base + 24].copy_from_slice(&mhz.to_be_bytes());
        buf[base + 24..base + 28].copy_from_slice(&sw_hz.to_be_bytes());
        buf[base + 28..base + 32].copy_from_slice(&orig_ppm.to_be_bytes());
    }
    buf
}

fn build_synthetic_nmrpipe_plane_header(z_planes: f32) -> Vec<u8> {
    let mut fdata = [0.0f32; 512];
    fdata[0] = 0.0;
    fdata[2] = 2.345; // IEEE-754 little-endian signature
    fdata[9] = 3.0; // FDDIMCOUNT = 3
    fdata[24] = 2.0; // DimOrder0 = F2 (direct 1H)
    fdata[25] = 1.0; // DimOrder1 = F1 (indirect 13C)
    fdata[26] = 3.0; // DimOrder2 = F3 (indirect 15N)

    // F2 (slot index 0 => X): 1H 600 MHz, 1024 pts, 8400 Hz
    let mut label_buf = [0u8; 32];
    label_buf[0..2].copy_from_slice(b"HN");
    label_buf[8..11].copy_from_slice(b"13C");
    label_buf[16..19].copy_from_slice(b"15N");
    for (k, chunk) in label_buf.chunks_exact(4).enumerate() {
        fdata[16 + k] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }

    fdata[99] = 1024.0; // F2 size
    fdata[100] = 8400.0; // F2 SW Hz
    fdata[119] = 600.13; // F2 OBS MHz
    fdata[56] = 1.0; // FT_FLAG

    // F1 (slot index 1): 13C 150.9 MHz, 128 pts, 10800 Hz (~71.5 ppm => HNCACB / CBCACONH window)
    fdata[219] = 128.0;
    fdata[229] = 10800.0;
    fdata[218] = 150.90;
    fdata[55] = 1.0;

    // F3 (slot index 2): 15N 60.81 MHz, z_planes pts, 1945 Hz (~32 ppm)
    fdata[15] = z_planes;
    fdata[11] = 1945.0;
    fdata[10] = 60.81;
    fdata[13] = 1.0;

    let mut bytes = Vec::with_capacity(2048);
    for v in fdata {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

#[test]
fn test_processed_ucsf_and_nmrpipe_plane_series_headers() {
    let root = make_temp_dir("processed_headers");

    let ucsf = build_synthetic_ucsf_3d(
        ("1H", 1024, 600.13, 8400.0, 4.7),
        ("15N", 64, 60.81, 1945.0, 118.0),
        ("13C", 96, 150.90, 4800.0, 54.0), // ~31.8 ppm 13CA window
    );
    write_bytes(&root, "spectra/HNCA.ucsf", &ucsf);

    let pipe_hdr = build_synthetic_nmrpipe_plane_header(4.0);
    for plane in 1..=4 {
        write_bytes(&root, &format!("pipe/HNCACB{plane:03}.ft3"), &pipe_hdr);
    }

    let inv = classify_nmr_workspace(&root).expect("classify processed workspace");
    assert_eq!(inv.experiments.len(), 2);

    let ucsf_exp = inv
        .experiments
        .iter()
        .find(|e| e.vendor == ExperimentVendor::SparkyUcsf)
        .expect("find UCSF experiment");
    assert_eq!(ucsf_exp.dimensionality, ExperimentDimensionality::Dim3D);
    assert_eq!(ucsf_exp.data_state, DataState::Processed);
    assert_eq!(ucsf_exp.pulse_class, PulseSequenceClass::Hnca);
    assert!((ucsf_exp.spectrometer_mhz.unwrap() - 600.13).abs() < 0.05);

    let pipe_exp = inv
        .experiments
        .iter()
        .find(|e| e.vendor == ExperimentVendor::NmrPipe)
        .expect("find NMRPipe series");
    assert_eq!(pipe_exp.dimensionality, ExperimentDimensionality::Dim3D);
    assert_eq!(pipe_exp.data_state, DataState::Processed);
    assert_eq!(pipe_exp.pulse_class, PulseSequenceClass::Hncacb);
    assert_eq!(pipe_exp.processed_bytes, 4 * 2048);

    let _ = fs::remove_dir_all(&root);
}

fn append_tar_entry(tar: &mut Vec<u8>, name: &str, payload: &[u8]) {
    let mut hdr = [0u8; 512];
    let nb = name.as_bytes();
    hdr[..nb.len()].copy_from_slice(nb);
    hdr[100..107].copy_from_slice(b"0000644");
    hdr[108..115].copy_from_slice(b"0001000");
    hdr[116..123].copy_from_slice(b"0001000");
    let size_oct = format!("{:011o}", payload.len());
    hdr[124..135].copy_from_slice(size_oct.as_bytes());
    hdr[136..147].copy_from_slice(b"14600000000");
    hdr[148..156].fill(b' ');
    hdr[156] = b'0';
    hdr[257..263].copy_from_slice(b"ustar\0");
    hdr[263..265].copy_from_slice(b"00");
    let csum: u32 = hdr.iter().map(|&b| b as u32).sum();
    let csum_oct = format!("{csum:06o}\0 ");
    hdr[148..156].copy_from_slice(csum_oct.as_bytes());

    tar.extend_from_slice(&hdr);
    tar.extend_from_slice(payload);
    let rem = payload.len() % 512;
    if rem != 0 {
        tar.extend(std::iter::repeat_n(0u8, 512 - rem));
    }
}

fn crc32_ieee(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            if (crc & 1) != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

/// Encode bytes as a valid RFC 1952 GZIP stream using RFC 1951 stored blocks (`BTYPE=00`).
fn gzip_stored(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    // GZIP 10-byte header: ID1=0x1f, ID2=0x8b, CM=8, FLG=0, MTIME=0, XFL=0, OS=255
    out.extend_from_slice(&[0x1f, 0x8b, 0x08, 0x00, 0, 0, 0, 0, 0, 255]);
    let chunks: Vec<&[u8]> = if data.is_empty() {
        vec![&[][..]]
    } else {
        data.chunks(65_535).collect()
    };
    for (idx, chunk) in chunks.iter().enumerate() {
        let is_last = idx + 1 == chunks.len();
        out.push(if is_last { 0x01 } else { 0x00 }); // BFINAL + BTYPE=00
        let len = chunk.len() as u16;
        let nlen = !len;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&nlen.to_le_bytes());
        out.extend_from_slice(chunk);
    }
    // CRC32 + ISIZE (8-byte trailer)
    out.extend_from_slice(&crc32_ieee(data).to_le_bytes());
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out
}

/// Encode bytes as a valid RFC 1952 GZIP stream using RFC 1951 fixed-Huffman (`BTYPE=01`)
/// with run-length LZ77 distance=1 compression for zero runs so a 4 MiB `ser` compresses to ~20 KiB!
fn gzip_fixed_huffman_rle_zeros(data: &[u8]) -> Vec<u8> {
    struct BitWriter {
        bytes: Vec<u8>,
        buf: u64,
        cnt: u8,
    }
    impl BitWriter {
        fn new() -> Self {
            Self {
                bytes: Vec::new(),
                buf: 0,
                cnt: 0,
            }
        }
        fn write_bits_lsb(&mut self, val: u32, n: u8) {
            self.buf |= (val as u64) << self.cnt;
            self.cnt += n;
            while self.cnt >= 8 {
                self.bytes.push((self.buf & 0xFF) as u8);
                self.buf >>= 8;
                self.cnt -= 8;
            }
        }
        fn write_huff_msb(&mut self, code: u16, len: u8) {
            for i in (0..len).rev() {
                let bit = ((code >> i) & 1) as u32;
                self.write_bits_lsb(bit, 1);
            }
        }
        fn emit_fixed_lit(&mut self, b: u8) {
            if b <= 143 {
                self.write_huff_msb(0x30 + (b as u16), 8);
            } else {
                self.write_huff_msb(0x190 + ((b - 144) as u16), 9);
            }
        }
        fn emit_match_258_dist1(&mut self) {
            // Length 258 = symbol 285 -> fixed code 280..287 => 0xC0 + (285 - 280) = 0xC5 (8 bits)
            self.write_huff_msb(0xC5, 8);
            // Distance 1 = dist symbol 0 -> 5-bit MSB code 00000
            self.write_huff_msb(0, 5);
        }
        fn finish(mut self) -> Vec<u8> {
            // End-of-block symbol 256 -> 7-bit MSB code 0000000
            self.write_huff_msb(0, 7);
            if self.cnt > 0 {
                self.bytes.push((self.buf & 0xFF) as u8);
            }
            self.bytes
        }
    }

    let mut bw = BitWriter::new();
    // BFINAL=1, BTYPE=01 (fixed Huffman) -> LSB bits: 1, then 01 => 0b011 = 3 in 3 bits
    bw.write_bits_lsb(0b011, 3);
    let mut i = 0usize;
    while i < data.len() {
        if i > 0
            && data[i - 1] == 0
            && i + 258 <= data.len()
            && data[i..i + 258].iter().all(|&x| x == 0)
        {
            bw.emit_match_258_dist1();
            i += 258;
        } else {
            bw.emit_fixed_lit(data[i]);
            i += 1;
        }
    }
    let deflate = bw.finish();

    let mut out = Vec::with_capacity(18 + deflate.len());
    out.extend_from_slice(&[0x1f, 0x8b, 0x08, 0x00, 0, 0, 0, 0, 0, 255]);
    out.extend_from_slice(&deflate);
    out.extend_from_slice(&crc32_ieee(data).to_le_bytes());
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out
}

#[test]
fn test_tar_and_targz_archive_streaming_and_wasm_entries_parity() {
    let root = make_temp_dir("archives");

    let acqus = bruker_acqus("hncogp3d", "1H", 700.13, 14.0, 2048, 0);
    let acqu2s = bruker_indirect_acqus("13C", 176.05, 14.0, 96, 5);
    let acqu3s = bruker_indirect_acqus("15N", 70.95, 32.0, 64, 6);
    let fasta =
        b">ubq\nMQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG\n";
    let huge_ser = vec![0u8; 4 * 1024 * 1024]; // 4 MiB FID payload

    let mut tar_bytes = Vec::new();
    append_tar_entry(&mut tar_bytes, "bundle/10_hnco/acqus", acqus.as_bytes());
    append_tar_entry(&mut tar_bytes, "bundle/10_hnco/acqu2s", acqu2s.as_bytes());
    append_tar_entry(&mut tar_bytes, "bundle/10_hnco/acqu3s", acqu3s.as_bytes());
    append_tar_entry(&mut tar_bytes, "bundle/10_hnco/ser", &huge_ser);
    append_tar_entry(&mut tar_bytes, "bundle/ubq.fasta", fasta);
    tar_bytes.extend(std::iter::repeat_n(0u8, 1024)); // Two 512-byte zero EOF blocks

    let tar_path = root.join("bundle.tar");
    fs::write(&tar_path, &tar_bytes).unwrap();

    let tgz_stored = gzip_stored(&tar_bytes);
    let tgz_stored_path = root.join("bundle_stored.tar.gz");
    fs::write(&tgz_stored_path, &tgz_stored).unwrap();

    let tgz_huff = gzip_fixed_huffman_rle_zeros(&tar_bytes);
    assert!(
        tgz_huff.len() < 60_000,
        "4 MiB zero-heavy tar should RLE-compress under 60 KB, got {}",
        tgz_huff.len()
    );
    let tgz_huff_path = root.join("bundle_huff.tgz");
    fs::write(&tgz_huff_path, &tgz_huff).unwrap();

    // 1. Native `.tar` scan (must seek past 4 MiB `ser` without reading payload)
    let inv_tar = classify_nmr_workspace(&tar_path).expect("classify .tar");
    assert!(
        inv_tar.elapsed_ms < MAX_HEADER_SCAN_MS,
        ".tar header scan took {:.2} ms (> {MAX_HEADER_SCAN_MS} ms)",
        inv_tar.elapsed_ms
    );
    assert_eq!(inv_tar.experiments.len(), 1);
    assert_eq!(inv_tar.experiments[0].pulse_class, PulseSequenceClass::Hnco);
    assert_eq!(inv_tar.experiments[0].raw_signal_bytes, 4 * 1024 * 1024);

    // 2. Native `.tar.gz` stored + Huffman scans
    let inv_tgz_stored = classify_nmr_workspace(&tgz_stored_path).expect("classify stored .tar.gz");
    let inv_tgz_huff = classify_nmr_workspace(&tgz_huff_path).expect("classify huffman .tgz");
    assert!(
        inv_tgz_huff.elapsed_ms < MAX_HEADER_SCAN_MS,
        ".tgz header scan took {:.2} ms (> {MAX_HEADER_SCAN_MS} ms)",
        inv_tgz_huff.elapsed_ms
    );
    assert_eq!(inv_tgz_stored.experiments, inv_tar.experiments);
    assert_eq!(inv_tgz_huff.experiments, inv_tar.experiments);
    assert_eq!(inv_tgz_huff.auxiliary_files, inv_tar.auxiliary_files);

    // 3. `wasm32`-compatible `classify_nmr_entries` in-memory parity
    let wasm_entries = vec![
        VirtualEntry::new("bundle/10_hnco/acqus", acqus.into_bytes()),
        VirtualEntry::new("bundle/10_hnco/acqu2s", acqu2s.into_bytes()),
        VirtualEntry::new("bundle/10_hnco/acqu3s", acqu3s.into_bytes()),
        VirtualEntry::header_only("bundle/10_hnco/ser", Vec::new(), 4 * 1024 * 1024),
        VirtualEntry::new("bundle/ubq.fasta", fasta.to_vec()),
    ];
    let inv_wasm = classify_nmr_entries(&wasm_entries);
    assert_eq!(inv_wasm.experiments, inv_tar.experiments);
    assert_eq!(inv_wasm.auxiliary_files, inv_tar.auxiliary_files);
    assert_eq!(inv_wasm.problem_regime, inv_tar.problem_regime);

    // 4. `extract_archive_virtual_entries` + inline `.tgz` expansion inside `classify_nmr_entries`
    let extracted =
        extract_archive_virtual_entries("bundle_huff.tgz", &tgz_huff).expect("archive bytes gz");
    let inv_mem_gz = classify_nmr_entries(&extracted);
    assert_eq!(inv_mem_gz.experiments, inv_tar.experiments);
    let inv_inline_drop = classify_nmr_entries(&[VirtualEntry::new("bundle_huff.tgz", tgz_huff)]);
    assert_eq!(inv_inline_drop.experiments, inv_tar.experiments);

    let _ = fs::remove_dir_all(&root);
}

fn send_http_post_with_headers(
    addr: std::net::SocketAddr,
    path: &str,
    ctype: &str,
    extra_headers: &[(&str, &str)],
    body: &[u8],
) -> (u16, String, Vec<u8>) {
    let mut stream = TcpStream::connect(addr).expect("connect to service");
    let mut req_hdr = format!(
        "POST {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\n",
        body.len()
    );
    for (k, v) in extra_headers {
        req_hdr.push_str(&format!("{k}: {v}\r\n"));
    }
    req_hdr.push_str("Connection: close\r\n\r\n");
    stream.write_all(req_hdr.as_bytes()).unwrap();
    stream.write_all(body).unwrap();
    stream.flush().unwrap();

    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).unwrap();
    let sep = raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .expect("HTTP header separator");
    let hdr_str = String::from_utf8_lossy(&raw[..sep]).into_owned();
    let status: u16 = hdr_str
        .lines()
        .next()
        .unwrap()
        .split_whitespace()
        .nth(1)
        .unwrap()
        .parse()
        .unwrap();
    (status, hdr_str, raw[sep + 4..].to_vec())
}

fn send_http_post(
    addr: std::net::SocketAddr,
    path: &str,
    ctype: &str,
    body: &[u8],
) -> (u16, Vec<u8>) {
    let (status, hdrs, resp) = send_http_post_with_headers(addr, path, ctype, &[], body);
    assert!(
        !hdrs
            .to_ascii_lowercase()
            .contains("access-control-allow-origin: *"),
        "wildcard CORS must never be emitted: {hdrs}"
    );
    (status, resp)
}

#[test]
fn test_service_post_api_v1_inspect_modes_and_security_guards() {
    let root = make_temp_dir("service_inspect");
    let ws_dir = root.join("sample_ws");
    write_text(
        &ws_dir,
        "1_hsqc/acqus",
        &bruker_acqus("hsqcetfpf3gpsi", "1H", 800.13, 14.0, 2048, 0),
    );
    write_text(
        &ws_dir,
        "1_hsqc/acqu2s",
        &bruker_indirect_acqus("15N", 81.08, 32.0, 256, 6),
    );
    write_bytes(&ws_dir, "1_hsqc/ser", &[0u8; 128]);

    let outside_dir = root.join("outside_secret");
    write_text(
        &outside_dir,
        "1_secret/acqus",
        &bruker_acqus("hsqcetfpf3gpsi", "1H", 800.13, 14.0, 2048, 0),
    );

    let storage_dir = root.join("srv_storage");
    let mut srv =
        spawn_background_server("127.0.0.1:0", storage_dir).expect("spawn background server");
    let addr = srv.local_addr();

    // Security Guard 1: Default server rejects `{ "path": ... }` with 403 Forbidden
    let body_path = serde_json::json!({
        "path": ws_dir.to_string_lossy()
    });
    let body_path_bytes = serde_json::to_vec(&body_path).unwrap();
    let (st_forbidden, _) = send_http_post(
        addr,
        "/api/v1/inspect",
        "application/json",
        &body_path_bytes,
    );
    assert_eq!(
        st_forbidden, 403,
        "server-side path must be disabled by default"
    );

    // Security Guard 2: Browser simple-request `text/plain` POST is rejected with 415
    let entries = vec![
        VirtualEntry::new(
            "drop/acqus",
            bruker_acqus("hncagp3d", "1H", 800.13, 14.0, 2048, 0).into_bytes(),
        ),
        VirtualEntry::new(
            "drop/acqu2s",
            bruker_indirect_acqus("13C", 201.19, 32.0, 96, 5).into_bytes(),
        ),
        VirtualEntry::new(
            "drop/acqu3s",
            bruker_indirect_acqus("15N", 81.08, 32.0, 64, 6).into_bytes(),
        ),
        VirtualEntry::header_only("drop/ser", Vec::new(), 65_536),
    ];
    let body_entries = serde_json::json!({
        "root_name": "browser_drop",
        "entries": entries
    });
    let body_entries_bytes = serde_json::to_vec(&body_entries).unwrap();
    let (st_plain, _) = send_http_post(addr, "/api/v1/inspect", "text/plain", &body_entries_bytes);
    assert_eq!(
        st_plain, 415,
        "text/plain JSON CSRF bypass must be rejected"
    );

    // Security Guard 3: Untrusted cross-origin `Origin` header is rejected with 403
    let (st_evil_origin, hdr_evil, _) = send_http_post_with_headers(
        addr,
        "/api/v1/inspect",
        "application/json",
        &[("Origin", "https://evil.example")],
        &body_entries_bytes,
    );
    assert_eq!(st_evil_origin, 403);
    assert!(!hdr_evil
        .to_ascii_lowercase()
        .contains("access-control-allow-origin"));

    // Mode 2: JSON `{ "root_name": "browser_drop", "entries": [...] }` (`application/json`)
    let (st2, resp2) = send_http_post(
        addr,
        "/api/v1/inspect",
        "application/json",
        &body_entries_bytes,
    );
    assert_eq!(st2, 200);
    let inv2: WorkspaceInventory = serde_json::from_slice(&resp2).unwrap();
    assert_eq!(inv2.root_name, "browser_drop");
    assert_eq!(inv2.experiments.len(), 1);
    assert_eq!(inv2.experiments[0].pulse_class, PulseSequenceClass::Hnca);

    // Mode 3: Binary `.tar.gz` upload (`Content-Type: application/gzip`)
    let mut tar = Vec::new();
    append_tar_entry(
        &mut tar,
        "arch/1_con/acqus",
        bruker_acqus("c_con_iasq", "13C", 201.20, 40.0, 2048, 0).as_bytes(),
    );
    append_tar_entry(
        &mut tar,
        "arch/1_con/acqu2s",
        bruker_indirect_acqus("15N", 81.08, 36.0, 256, 5).as_bytes(),
    );
    append_tar_entry(&mut tar, "arch/1_con/ser", &[0u8; 512]);
    tar.extend(std::iter::repeat_n(0u8, 1024));
    let tgz = gzip_fixed_huffman_rle_zeros(&tar);
    let (st3, resp3) = send_http_post(addr, "/api/v1/inspect", "application/gzip", &tgz);
    assert_eq!(st3, 200);
    let inv3: WorkspaceInventory = serde_json::from_slice(&resp3).unwrap();
    assert_eq!(inv3.problem_regime, ProblemRegime::Idp13cDetected);
    assert_eq!(inv3.experiments[0].pulse_class, PulseSequenceClass::Con2D);

    srv.shutdown();

    // Mode 1 (Scoped): Explicit `inspect_root` allows inside-root and rejects outside-root
    let storage_dir2 = root.join("srv_storage_scoped");
    let mut srv_scoped = spawn_background_server_with_options(
        "127.0.0.1:0",
        storage_dir2,
        ServiceOptions {
            inspect_root: Some(ws_dir.clone()),
            allowed_origins: vec!["https://workbench.local".to_string()],
        },
    )
    .expect("spawn scoped background server");
    let addr_scoped = srv_scoped.local_addr();

    let (st_ok, hdr_ok, resp_ok) = send_http_post_with_headers(
        addr_scoped,
        "/api/v1/inspect",
        "application/json",
        &[("Origin", "https://workbench.local")],
        &body_path_bytes,
    );
    assert_eq!(st_ok, 200, "got: {}", String::from_utf8_lossy(&resp_ok));
    assert!(hdr_ok.contains("Access-Control-Allow-Origin: https://workbench.local"));
    let inv1: WorkspaceInventory = serde_json::from_slice(&resp_ok).unwrap();
    assert_eq!(inv1.experiments.len(), 1);
    assert_eq!(inv1.experiments[0].pulse_class, PulseSequenceClass::N15Hsqc);

    let body_outside = serde_json::json!({
        "path": outside_dir.to_string_lossy()
    });
    let (st_out, _) = send_http_post(
        addr_scoped,
        "/api/v1/inspect",
        "application/json",
        &serde_json::to_vec(&body_outside).unwrap(),
    );
    assert_eq!(st_out, 403, "path outside inspect_root must be rejected");

    srv_scoped.shutdown();
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn test_archive_hardening_gnu_tar_crc32_checksum_and_bomb_limits() {
    let acqus = bruker_acqus("hncogp3d", "1H", 700.13, 14.0, 2048, 0);
    let acqu2s = bruker_indirect_acqus("13C", 176.05, 14.0, 96, 5);
    let acqu3s = bruker_indirect_acqus("15N", 70.95, 32.0, 64, 6);

    // 1. GNU tar header (`ustar  \0`) with non-zero atime/ctime bytes in 345..369 must NOT corrupt path
    let mut gnu_tar = Vec::new();
    for (name, payload) in [
        ("gnu_ws/10/acqus", acqus.as_bytes()),
        ("gnu_ws/10/acqu2s", acqu2s.as_bytes()),
        ("gnu_ws/10/acqu3s", acqu3s.as_bytes()),
    ] {
        let mut hdr = [0u8; 512];
        hdr[..name.len()].copy_from_slice(name.as_bytes());
        hdr[100..107].copy_from_slice(b"0000644");
        hdr[108..115].copy_from_slice(b"0001000");
        hdr[116..123].copy_from_slice(b"0001000");
        hdr[124..135].copy_from_slice(format!("{:011o}", payload.len()).as_bytes());
        hdr[136..147].copy_from_slice(b"14600000000");
        hdr[148..156].fill(b' ');
        hdr[156] = b'0';
        hdr[257..265].copy_from_slice(b"ustar  \0"); // GNU magic, NOT POSIX ustar\0
        hdr[345..357].copy_from_slice(b"14600012345\0"); // GNU atime in POSIX prefix slot!
        hdr[357..369].copy_from_slice(b"14600012346\0"); // GNU ctime
        let csum: u32 = hdr.iter().map(|&b| b as u32).sum();
        hdr[148..156].copy_from_slice(format!("{csum:06o}\0 ").as_bytes());
        gnu_tar.extend_from_slice(&hdr);
        gnu_tar.extend_from_slice(payload);
        let rem = payload.len() % 512;
        if rem != 0 {
            gnu_tar.extend(std::iter::repeat_n(0u8, 512 - rem));
        }
    }
    gnu_tar.extend(std::iter::repeat_n(0u8, 1024));

    let gnu_entries = extract_archive_virtual_entries("gnu.tar", &gnu_tar).expect("parse GNU tar");
    assert_eq!(gnu_entries[0].relative_path, "gnu_ws/10/acqus");
    let gnu_inv = classify_nmr_entries(&gnu_entries);
    assert_eq!(gnu_inv.experiments.len(), 1);
    assert_eq!(gnu_inv.experiments[0].pulse_class, PulseSequenceClass::Hnco);

    // 2. Corrupted tar header checksum must return Err
    let mut bad_tar = gnu_tar.clone();
    bad_tar[0] ^= 0xFF;
    assert!(extract_archive_virtual_entries("bad.tar", &bad_tar).is_err());

    // 3. Corrupted gzip CRC32 trailer must return Err
    let mut bad_gz = gzip_stored(&gnu_tar);
    let crc_pos = bad_gz.len() - 8;
    bad_gz[crc_pos] ^= 0x01;
    assert!(extract_archive_virtual_entries("bad.tar.gz", &bad_gz).is_err());

    // 4. Out-of-history DEFLATE LZ77 distance (match before any literal) must return Err
    // BFINAL=1, BTYPE=01 (0b011), length 258 (0xC5 8-bit MSB), dist 1 (00000 5-bit MSB), EOB (0000000)
    let bad_deflate_gz: [u8; 21] = [
        0x1f,
        0x8b,
        0x08,
        0x00,
        0,
        0,
        0,
        0,
        0,
        255, // header
        0b1010_0011,
        0b0000_0001,
        0x00, // match258 dist=1 at pos 0 + EOB
        0,
        0,
        0,
        0,
        2,
        1,
        0,
        0, // trailer
    ];
    assert!(extract_archive_virtual_entries("bad_dist.tgz", &bad_deflate_gz).is_err());

    // 5. > 130 MiB `ser` `.tar.gz`:
    //    (a) `extract_archive_virtual_entries` (browser drag-and-drop) MUST SUCCEED by streaming/discarding
    //        `ser` into its 8 KiB scratch buffer without retaining payload memory;
    //    (b) `extract_archive_virtual_entries_bounded(..., 128 MiB, 32 MiB)` AND untrusted HTTP
    //        `POST /api/v1/inspect` MUST REJECT it!
    let mut bomb_tar = Vec::new();
    let bomb_hdr_only = {
        let mut hdr = [0u8; 512];
        let name = b"bomb/ser";
        hdr[..name.len()].copy_from_slice(name);
        hdr[100..107].copy_from_slice(b"0000644");
        hdr[108..115].copy_from_slice(b"0001000");
        hdr[116..123].copy_from_slice(b"0001000");
        let sz = 130usize * 1024 * 1024; // 130 MiB > 128 MiB HTTP cap
        hdr[124..135].copy_from_slice(format!("{sz:011o}").as_bytes());
        hdr[136..147].copy_from_slice(b"14600000000");
        hdr[148..156].fill(b' ');
        hdr[156] = b'0';
        hdr[257..263].copy_from_slice(b"ustar\0");
        hdr[263..265].copy_from_slice(b"00");
        let csum: u32 = hdr.iter().map(|&b| b as u32).sum();
        hdr[148..156].copy_from_slice(format!("{csum:06o}\0 ").as_bytes());
        hdr
    };
    bomb_tar.extend_from_slice(&bomb_hdr_only);
    bomb_tar.resize(512 + 130 * 1024 * 1024, 0u8);
    let bomb_tgz = gzip_fixed_huffman_rle_zeros(&bomb_tar);
    assert!(
        bomb_tgz.len() < 1_000_000,
        "RLE-compressed 130 MiB archive should be < 1 MB, got {}",
        bomb_tgz.len()
    );
    let ok_entries = extract_archive_virtual_entries("large_bmrb.tgz", &bomb_tgz)
        .expect("130 MiB ser archive must succeed for browser drag-and-drop");
    assert_eq!(ok_entries.len(), 1);
    assert!(
        ok_entries[0].header_bytes.is_empty(),
        "discarded ser payload must retain 0 bytes in header_bytes"
    );
    assert_eq!(ok_entries[0].total_bytes, 130 * 1024 * 1024);

    let bomb_err = extract_archive_virtual_entries_bounded(
        "bomb.tgz",
        &bomb_tgz,
        128 * 1024 * 1024,
        32 * 1024 * 1024,
    )
    .expect_err("130 MiB gzip bomb must be rejected under 128 MiB bound");
    assert!(
        bomb_err.contains("maximum uncompressed byte limit"),
        "unexpected error: {bomb_err}"
    );

    let retained_err =
        extract_archive_virtual_entries_bounded("gnu.tar", &gnu_tar, 16 * 1024 * 1024, 32)
            .expect_err("exceeding max_retained_header_bytes must be rejected");
    assert!(
        retained_err.contains("maximum retained header byte limit"),
        "unexpected retained error: {retained_err}"
    );

    let root = make_temp_dir("bomb_http_inspect");
    let mut srv = spawn_background_server("127.0.0.1:0", root.join("srv"))
        .expect("spawn background server for bomb test");
    let (st_bomb, resp_bomb) = send_http_post(
        srv.local_addr(),
        "/api/v1/inspect",
        "application/gzip",
        &bomb_tgz,
    );
    assert_eq!(
        st_bomb,
        400,
        "POST /api/v1/inspect must reject > 128 MiB gzip bomb: {}",
        String::from_utf8_lossy(&resp_bomb)
    );
    srv.shutdown();
    let _ = fs::remove_dir_all(&root);
}

fn find_real_data_dir() -> Option<PathBuf> {
    if let Ok(val) = env::var("NMR_REAL_DATA_DIR") {
        let p = PathBuf::from(val);
        if p.is_dir() {
            return Some(p);
        }
    }
    let fallback = PathBuf::from("/Users/debsahu/Workspace/nmr_software/data");
    if fallback.is_dir() {
        return Some(fallback);
    }
    None
}

#[test]
fn test_conditional_real_dev_datasets_accuracy_and_speed() {
    let Some(data_dir) = find_real_data_dir() else {
        eprintln!("Skipping real dataset header tests (NMR_REAL_DATA_DIR not present)");
        return;
    };

    // 1. Ubiquitin (`data/Ubiquitin`) — multi-modal Bruker + Sparky UCSF + 13C-detected + relaxation + pulsecal decoy
    let ubq_path = data_dir.join("Ubiquitin");
    if ubq_path.is_dir() {
        let inv = classify_nmr_workspace(&ubq_path).expect("classify Ubiquitin");
        eprintln!(
            "Real dataset Ubiquitin: {} experiments, regime={}, elapsed={:.2} ms",
            inv.experiments.len(),
            inv.problem_regime,
            inv.elapsed_ms
        );
        assert!(
            inv.elapsed_ms < MAX_HEADER_SCAN_MS,
            "Ubiquitin header scan took {:.2} ms (limit < {MAX_HEADER_SCAN_MS} ms)",
            inv.elapsed_ms
        );
        assert!(matches!(
            inv.problem_regime,
            ProblemRegime::SidechainNoesy | ProblemRegime::GlobularBackbone
        ));
        assert!(inv
            .secondary_regimes
            .contains(&ProblemRegime::Idp13cDetected));
        assert!(inv
            .secondary_regimes
            .contains(&ProblemRegime::RelaxationDynamics));

        // Check `99999` pulsecal 1D decoy is NOT misclassified as 3D HNCO
        let pcal = inv
            .experiments
            .iter()
            .find(|e| e.relative_path.ends_with("/99999") || e.relative_path == "99999")
            .expect("Ubiquitin pulsecal 99999");
        assert_eq!(pcal.dimensionality, ExperimentDimensionality::Dim1D);
        assert_eq!(pcal.pulse_class, PulseSequenceClass::Unknown);
        assert!(!pcal.selected_by_default);

        // Check Pseudo3D R1 (`Ubiquitin_apo_083109/4`)
        let pseudo_r1 = inv
            .experiments
            .iter()
            .find(|e| e.relative_path == "Ubiquitin_apo_083109/4")
            .expect("Ubiquitin_apo_083109/4");
        assert_eq!(pseudo_r1.dimensionality, ExperimentDimensionality::Pseudo3D);
        assert_eq!(pseudo_r1.pulse_class, PulseSequenceClass::RelaxR1);

        // Check 13C-detected `Con2D`, `Caco2D`, `Canco3D`, `Cbcaco3D` AND Sparky `.ucsf` files
        for expected in [
            PulseSequenceClass::N15Hsqc,
            PulseSequenceClass::Hnco,
            PulseSequenceClass::Hncaco,
            PulseSequenceClass::Hncacb,
            PulseSequenceClass::Cbcaconh,
            PulseSequenceClass::Con2D,
            PulseSequenceClass::Caco2D,
            PulseSequenceClass::Canco3D,
            PulseSequenceClass::Cbcaco3D,
            PulseSequenceClass::RelaxR1,
            PulseSequenceClass::RelaxR2,
            PulseSequenceClass::HetNoe,
        ] {
            assert!(
                inv.experiments.iter().any(|e| e.pulse_class == expected),
                "Ubiquitin missing expected pulse class {expected:?}"
            );
        }
        assert!(inv
            .experiments
            .iter()
            .any(|e| e.vendor == ExperimentVendor::SparkyUcsf));
    }

    // 2. CISD3 (`BMRB 51439`) — Bruker 900 MHz with 50% NUS CBCACONH
    let cisd3_path = data_dir.join("private/bmrb51439_cisd3");
    if cisd3_path.is_dir() {
        let inv = classify_nmr_workspace(&cisd3_path).expect("classify CISD3 51439");
        eprintln!(
            "Real dataset CISD3 (51439): {} experiments, regime={}, elapsed={:.2} ms",
            inv.experiments.len(),
            inv.problem_regime,
            inv.elapsed_ms
        );
        assert!(
            inv.elapsed_ms < MAX_HEADER_SCAN_MS,
            "CISD3 51439 header scan took {:.2} ms (limit < {MAX_HEADER_SCAN_MS} ms)",
            inv.elapsed_ms
        );
        assert_eq!(inv.problem_regime, ProblemRegime::GlobularBackbone);
        assert!(inv.missing_input_hints.is_empty());

        let expected_cisd3 = [
            (
                "CISD3_15N_HSQC_diamagnetic",
                PulseSequenceClass::N15Hsqc,
                false,
            ),
            ("CISD3_CBCACONH", PulseSequenceClass::Cbcaconh, true),
            ("CISD3_HNCA", PulseSequenceClass::Hnca, false),
            ("CISD3_HNCO", PulseSequenceClass::Hnco, false),
            ("CISD3_HNCOCA", PulseSequenceClass::Hncoca, false),
        ];
        for (sub, exp_cls, exp_nus) in expected_cisd3 {
            let e = inv
                .experiments
                .iter()
                .find(|x| x.relative_path.contains(sub))
                .unwrap_or_else(|| panic!("CISD3 missing {sub}"));
            assert_eq!(e.pulse_class, exp_cls, "mismatch on {sub}");
            assert_eq!(e.nus.is_nus, exp_nus, "NUS mismatch on {sub}");
            if exp_nus {
                assert!((e.nus.sampling_density.unwrap() - 0.50).abs() < 0.02);
            }
        }
    }

    // 3. BMRB 53371 — Bruker NUS triple-resonance + Uniform `fhsqc` (`FnTYPE=0` with template `NusAMOUNT=25`)
    let p53371 = data_dir.join("private/bmrb/53371");
    if p53371.is_dir() {
        let inv = classify_nmr_workspace(&p53371).expect("classify 53371");
        eprintln!(
            "Real dataset 53371: {} experiments, regime={}, elapsed={:.2} ms",
            inv.experiments.len(),
            inv.problem_regime,
            inv.elapsed_ms
        );
        assert!(
            inv.elapsed_ms < MAX_HEADER_SCAN_MS,
            "53371 header scan took {:.2} ms (limit < {MAX_HEADER_SCAN_MS} ms)",
            inv.elapsed_ms
        );
        assert_eq!(inv.problem_regime, ProblemRegime::GlobularBackbone);
        let fhsqc = inv
            .experiments
            .iter()
            .find(|e| e.pulse_class == PulseSequenceClass::N15Hsqc)
            .expect("53371 N15Hsqc");
        assert!(
            !fhsqc.nus.is_nus,
            "53371 fHSQC has FnTYPE=0 and must be Uniform"
        );
        for cls in [
            PulseSequenceClass::Cbcaconh,
            PulseSequenceClass::Hncacb,
            PulseSequenceClass::Hncaco,
            PulseSequenceClass::Hnco,
        ] {
            let e = inv
                .experiments
                .iter()
                .find(|x| x.pulse_class == cls)
                .unwrap_or_else(|| panic!("53371 missing {cls:?}"));
            assert!(e.nus.is_nus, "53371 {cls:?} should be NUS");
        }
    }

    // 4. BMRB 18465 — Mixed Bruker (`b8/`) + Varian (`v6/` with `refsource1="dfrq2"` 15N-HSQC!)
    let p18465 = data_dir.join("private/bmrb/18465");
    if p18465.is_dir() {
        let inv = classify_nmr_workspace(&p18465).expect("classify 18465");
        eprintln!(
            "Real dataset 18465: {} experiments, regime={}, elapsed={:.2} ms",
            inv.experiments.len(),
            inv.problem_regime,
            inv.elapsed_ms
        );
        assert!(
            inv.elapsed_ms < MAX_HEADER_SCAN_MS,
            "18465 header scan took {:.2} ms (limit < {MAX_HEADER_SCAN_MS} ms)",
            inv.elapsed_ms
        );
        assert_eq!(inv.problem_regime, ProblemRegime::GlobularBackbone);
        assert!(inv
            .experiments
            .iter()
            .any(|e| e.vendor == ExperimentVendor::Bruker
                && e.pulse_class == PulseSequenceClass::N15Hsqc));
        for cls in [
            PulseSequenceClass::N15Hsqc,
            PulseSequenceClass::Hnco,
            PulseSequenceClass::Hnca,
            PulseSequenceClass::Hncacb,
            PulseSequenceClass::Cbcaconh,
        ] {
            assert!(
                inv.experiments
                    .iter()
                    .any(|e| e.vendor == ExperimentVendor::Varian && e.pulse_class == cls),
                "18465 Varian missing {cls:?}"
            );
        }
        assert!(
            inv.experiments
                .iter()
                .all(|e| e.pulse_class != PulseSequenceClass::Unknown),
            "18465 had Unknown experiments: {:?}",
            inv.experiments
        );
    }

    // 5. BMRB 5329 — Varian backbone + sidechain TOCSY + 3D/4D NOESY (`SidechainNoesy`)
    let p5329 = data_dir.join("private/bmrb/5329");
    if p5329.is_dir() {
        let inv = classify_nmr_workspace(&p5329).expect("classify 5329");
        eprintln!(
            "Real dataset 5329: {} experiments, regime={}, elapsed={:.2} ms",
            inv.experiments.len(),
            inv.problem_regime,
            inv.elapsed_ms
        );
        assert!(
            inv.elapsed_ms < MAX_HEADER_SCAN_MS,
            "5329 header scan took {:.2} ms (limit < {MAX_HEADER_SCAN_MS} ms)",
            inv.elapsed_ms
        );
        assert_eq!(inv.problem_regime, ProblemRegime::SidechainNoesy);
        for cls in [
            PulseSequenceClass::N15Hsqc,
            PulseSequenceClass::C13Hsqc,
            PulseSequenceClass::Hnco,
            PulseSequenceClass::Hncacb,
            PulseSequenceClass::Cbcaconh,
            PulseSequenceClass::Ccconh,
            PulseSequenceClass::Hccconh,
            PulseSequenceClass::HcchTocsy,
            PulseSequenceClass::N15Noesy,
            PulseSequenceClass::C13Noesy,
        ] {
            assert!(
                inv.experiments.iter().any(|e| e.pulse_class == cls),
                "5329 missing {cls:?}"
            );
        }
        assert!(
            inv.experiments.iter().any(|e| {
                e.pulse_class == PulseSequenceClass::C13Noesy
                    && e.dimensionality == ExperimentDimensionality::Dim4D
            }),
            "5329 missing 4D C13Noesy"
        );
        assert!(
            inv.experiments.iter().all(|e| {
                e.pulse_class != PulseSequenceClass::Unknown
                    || e.pulse_program.as_deref() == Some("ghnha")
            }),
            "5329 had unexpected Unknown experiments: {:?}",
            inv.experiments
        );
    }

    // 6. BMRB 6318 — Bruker XWIN-NMR backbone + sidechain TOCSY + NOESY (`SidechainNoesy`)
    let p6318 = data_dir.join("private/bmrb/6318");
    if p6318.is_dir() {
        let inv = classify_nmr_workspace(&p6318).expect("classify 6318");
        eprintln!(
            "Real dataset 6318: {} experiments, regime={}, elapsed={:.2} ms",
            inv.experiments.len(),
            inv.problem_regime,
            inv.elapsed_ms
        );
        assert!(
            inv.elapsed_ms < MAX_HEADER_SCAN_MS,
            "6318 header scan took {:.2} ms (limit < {MAX_HEADER_SCAN_MS} ms)",
            inv.elapsed_ms
        );
        assert_eq!(inv.problem_regime, ProblemRegime::SidechainNoesy);
        for cls in [
            PulseSequenceClass::N15Hsqc,
            PulseSequenceClass::C13Hsqc,
            PulseSequenceClass::Hnco,
            PulseSequenceClass::Hncaco,
            PulseSequenceClass::Hnca,
            PulseSequenceClass::Hncoca,
            PulseSequenceClass::Hncacb,
            PulseSequenceClass::Ccconh,
            PulseSequenceClass::Hccconh,
            PulseSequenceClass::HcchTocsy,
            PulseSequenceClass::N15Noesy,
            PulseSequenceClass::C13Noesy,
        ] {
            assert!(
                inv.experiments.iter().any(|e| e.pulse_class == cls),
                "6318 missing {cls:?}"
            );
        }
        assert!(
            inv.experiments.iter().all(|e| {
                e.pulse_class != PulseSequenceClass::Unknown
                    || e.dimensionality == ExperimentDimensionality::Dim1D
            }),
            "6318 had unexpected 2D/3D Unknown experiments: {:?}",
            inv.experiments
        );
    }
}

fn bruker_indirect_acqus_with_carrier(
    nuc1: &str,
    bf1: f64,
    carrier_ppm: f64,
    sw_ppm: f64,
    td: usize,
    fn_mode: usize,
) -> String {
    let sfo1 = bf1 * (1.0 + carrier_ppm * 1.0e-6);
    format!(
        "##TITLE= Indirect parameter file\n\
         ##$NUC1= <{nuc1}>\n\
         ##$SFO1= {sfo1:.8}\n\
         ##$BF1= {bf1:.8}\n\
         ##$SW= {sw_ppm:.4}\n\
         ##$SW_h= {:.3}\n\
         ##$TD= {td}\n\
         ##$FnMODE= {fn_mode}\n\
         ##END=\n",
        sw_ppm * sfo1
    )
}

#[test]
fn test_q1b_methyl_carrier_sw_and_interleaved_acqu2s_qf_cest() {
    // 1. Methyl CT-HSQC (`PULPROG = hsqcetgpsisp2.2`, 13C carrier = 19 ppm, SW = 24 ppm) -> `C13Hmqc` (`MethylTrosy`)
    let ct_me_dir = make_temp_dir("q1b_methyl_ct_hsqc");
    write_text(
        &ct_me_dir,
        "10_methyl_cthsqc/acqus",
        &bruker_acqus("hsqcetgpsisp2.2", "1H", 800.13, 12.0, 2048, 0),
    );
    write_text(
        &ct_me_dir,
        "10_methyl_cthsqc/acqu2s",
        &bruker_indirect_acqus_with_carrier("13C", 201.19, 19.0, 24.0, 256, 5),
    );
    write_bytes(&ct_me_dir, "10_methyl_cthsqc/ser", &[0u8; 64]);
    let ct_inv = classify_nmr_workspace(&ct_me_dir).expect("classify methyl CT-HSQC");
    assert_eq!(
        ct_inv.experiments[0].pulse_class,
        PulseSequenceClass::C13Hmqc,
        "Methyl CT-HSQC (carrier=19 ppm, SW=24 ppm) must classify as C13Hmqc"
    );
    assert_eq!(ct_inv.problem_regime, ProblemRegime::MethylTrosy);
    let _ = fs::remove_dir_all(&ct_me_dir);

    // 2. Full-aliphatic `13C-HMQC` (`PULPROG = hmqcetgp`, carrier = 42 ppm, SW = 65 ppm) -> `C13Hsqc`
    //    AND aromatic `13C-HMQC` (`PULPROG = hmqcetgp`, carrier = 125 ppm, SW = 30 ppm) -> `C13Hsqc`
    let non_me_dir = make_temp_dir("q1b_non_methyl_hmqc");
    write_text(
        &non_me_dir,
        "10_aliph_hmqc/acqus",
        &bruker_acqus("hmqcetgp", "1H", 800.13, 12.0, 2048, 0),
    );
    write_text(
        &non_me_dir,
        "10_aliph_hmqc/acqu2s",
        &bruker_indirect_acqus_with_carrier("13C", 201.19, 42.0, 65.0, 512, 5),
    );
    write_bytes(&non_me_dir, "10_aliph_hmqc/ser", &[0u8; 64]);

    write_text(
        &non_me_dir,
        "20_arom_hmqc/acqus",
        &bruker_acqus("hmqcetgp", "1H", 800.13, 12.0, 2048, 0),
    );
    write_text(
        &non_me_dir,
        "20_arom_hmqc/acqu2s",
        &bruker_indirect_acqus_with_carrier("13C", 201.19, 125.0, 30.0, 256, 5),
    );
    write_bytes(&non_me_dir, "20_arom_hmqc/ser", &[0u8; 64]);

    let non_me_inv = classify_nmr_workspace(&non_me_dir).expect("classify non-methyl 13C-HMQC");
    assert_eq!(non_me_inv.experiments.len(), 2);
    assert_eq!(
        non_me_inv.experiments[0].pulse_class,
        PulseSequenceClass::C13Hsqc,
        "Full-aliphatic SW=65 ppm 13C-HMQC must classify as C13Hsqc, not C13Hmqc"
    );
    assert_eq!(
        non_me_inv.experiments[1].pulse_class,
        PulseSequenceClass::C13Hsqc,
        "Aromatic carrier=125 ppm 13C-HMQC must classify as C13Hsqc, not C13Hmqc"
    );
    assert_ne!(non_me_inv.problem_regime, ProblemRegime::MethylTrosy);
    let _ = fs::remove_dir_all(&non_me_dir);

    // 3. Interleaved `acqu2s`-stepped 3D `CEST` (`fq1list`, `acqu2s` FnMODE=1 QF with TD=10,
    //    `acqu3s` 15N Echo-Antiecho with TD=128, generic HSQC PULPROG without "cest" substring) -> `Pseudo3D` (`Cest`)
    let cest_dir = make_temp_dir("q1b_interleaved_acqu2s_cest");
    write_text(
        &cest_dir,
        "30_interleaved_cest/acqus",
        &bruker_acqus("hsqcetf3gpsi3d_sat", "1H", 700.13, 14.0, 2048, 0),
    );
    write_text(
        &cest_dir,
        "30_interleaved_cest/acqu2s",
        &bruker_indirect_acqus("off", 700.13, 1.0, 10, 1), // QF (FnMODE == 1), TD == fq1list length (10)
    );
    write_text(
        &cest_dir,
        "30_interleaved_cest/acqu3s",
        &bruker_indirect_acqus("15N", 70.95, 32.0, 128, 6), // Echo-Antiecho (FnMODE == 6)
    );
    write_text(
        &cest_dir,
        "30_interleaved_cest/fq1list",
        "10000.0\n7100.0\n7250.0\n7400.0\n7550.0\n7700.0\n7850.0\n8000.0\n8150.0\n8300.0\n",
    );
    write_bytes(&cest_dir, "30_interleaved_cest/ser", &[0u8; 256]);

    let cest_inv = classify_nmr_workspace(&cest_dir).expect("classify interleaved acqu2s CEST");
    assert_eq!(cest_inv.experiments.len(), 1);
    assert_eq!(
        cest_inv.experiments[0].dimensionality,
        ExperimentDimensionality::Pseudo3D,
        "Interleaved acqu2s-QF 3D CEST must be classified as Pseudo3D"
    );
    assert_eq!(
        cest_inv.experiments[0].pulse_class,
        PulseSequenceClass::Cest,
        "Interleaved acqu2s-QF 3D fq1list must be classified as Cest"
    );
    assert_eq!(cest_inv.problem_regime, ProblemRegime::RelaxationDynamics);
    let _ = fs::remove_dir_all(&cest_dir);
}

#[test]
fn test_p1_5_n2_trailing_zero_padded_tgz_succeeds_and_rejects_corrupt_nonzero_tail() {
    let root = make_temp_dir("p1_5_n2_pad_tgz");
    let acqus = bruker_acqus("hncogp3d", "1H", 700.13, 14.0, 2048, 0);
    let acqu2s = bruker_indirect_acqus("13C", 176.05, 14.0, 96, 5);
    let acqu3s = bruker_indirect_acqus("15N", 70.95, 32.0, 64, 6);

    let mut tar = Vec::new();
    append_tar_entry(&mut tar, "pad_ws/10_hnco/acqus", acqus.as_bytes());
    append_tar_entry(&mut tar, "pad_ws/10_hnco/acqu2s", acqu2s.as_bytes());
    append_tar_entry(&mut tar, "pad_ws/10_hnco/acqu3s", acqu3s.as_bytes());
    append_tar_entry(&mut tar, "pad_ws/10_hnco/ser", &[0u8; 256]);
    tar.extend(std::iter::repeat_n(0u8, 1024));

    // Valid `.tar.gz` + 1024 trailing 0x00 bytes (`pad.tgz`)
    let mut pad_tgz = gzip_fixed_huffman_rle_zeros(&tar);
    pad_tgz.extend(std::iter::repeat_n(0u8, 1024));

    let pad_tgz_path = root.join("pad.tgz");
    fs::write(&pad_tgz_path, &pad_tgz).expect("write pad.tgz");

    let entries =
        extract_archive_virtual_entries("pad.tgz", &pad_tgz).expect("extract zero-padded pad.tgz");
    assert_eq!(entries.len(), 4);
    let mem_inv = classify_nmr_entries(&entries);
    assert_eq!(mem_inv.experiments.len(), 1);
    assert_eq!(mem_inv.experiments[0].pulse_class, PulseSequenceClass::Hnco);

    let fs_inv =
        classify_nmr_workspace(&pad_tgz_path).expect("classify_nmr_workspace on zero-padded .tgz");
    assert_eq!(fs_inv.experiments, mem_inv.experiments);

    // Non-zero garbage after trailing 0x00 must still be rejected as corrupt gzip magic
    let mut corrupt_tail = pad_tgz.clone();
    corrupt_tail.push(0x42);
    assert!(
        extract_archive_virtual_entries("corrupt_tail.tgz", &corrupt_tail).is_err(),
        "non-zero corrupt byte after zero padding must be rejected"
    );

    // Excessive zero padding exceeding `max_inflate_bytes` must also be rejected
    let mut huge_pad = gzip_fixed_huffman_rle_zeros(&tar);
    huge_pad.extend(std::iter::repeat_n(0u8, 8192));
    assert!(
        extract_archive_virtual_entries_bounded("huge_pad.tgz", &huge_pad, 4096, 4096).is_err(),
        "zero padding exceeding max_inflate_bytes must be rejected"
    );

    let _ = fs::remove_dir_all(&root);
}

#[test]
fn test_p1_5_n3_loopback_dns_rebinding_host_header_validation() {
    let root = make_temp_dir("p1_5_n3_dns_rebind");
    let mut srv = spawn_background_server("127.0.0.1:0", root.join("srv"))
        .expect("spawn background server for DNS rebinding test");
    let addr = srv.local_addr();
    let port = addr.port();

    let send_raw = |raw_req: &str| -> u16 {
        let mut stream = TcpStream::connect(addr).expect("connect");
        stream.write_all(raw_req.as_bytes()).unwrap();
        stream.flush().unwrap();
        let mut raw = Vec::new();
        stream.read_to_end(&mut raw).unwrap();
        let hdr = String::from_utf8_lossy(&raw);
        hdr.lines()
            .next()
            .unwrap()
            .split_whitespace()
            .nth(1)
            .unwrap()
            .parse()
            .unwrap()
    };

    // 1. Exact probe: `Host: attacker-rebind.example:8080` + `Origin: http://attacker-rebind.example:8080` -> 403 Forbidden
    let rebind_8080 = "GET /api/v1/health HTTP/1.1\r\n\
                       Host: attacker-rebind.example:8080\r\n\
                       Origin: http://attacker-rebind.example:8080\r\n\
                       Connection: close\r\n\r\n";
    assert_eq!(send_raw(rebind_8080), 403);

    // 2. DNS rebinding matching the actual bound port `attacker-rebind.example:<port>` (with or without Origin) -> 403 Forbidden
    let rebind_bound = format!(
        "GET /api/v1/health HTTP/1.1\r\n\
         Host: attacker-rebind.example:{port}\r\n\
         Origin: http://attacker-rebind.example:{port}\r\n\
         Connection: close\r\n\r\n"
    );
    assert_eq!(send_raw(&rebind_bound), 403);

    // 3. Missing `Host` header -> 403 Forbidden
    let no_host = "GET /api/v1/health HTTP/1.1\r\nConnection: close\r\n\r\n";
    assert_eq!(send_raw(no_host), 403);

    // 4. Loopback `Host` with mismatched or missing port on non-80/443 listener -> 403 Forbidden
    let wrong_port = format!(
        "GET /api/v1/health HTTP/1.1\r\n\
         Host: 127.0.0.1:{}\r\n\
         Connection: close\r\n\r\n",
        port.wrapping_add(1).max(1)
    );
    assert_eq!(send_raw(&wrong_port), 403);
    if !matches!(port, 80 | 443) {
        let bare_loopback = "GET /api/v1/health HTTP/1.1\r\n\
                             Host: localhost\r\n\
                             Origin: http://localhost\r\n\
                             Connection: close\r\n\r\n";
        assert_eq!(send_raw(bare_loopback), 403);
    }

    // 5. Valid loopback `Host: 127.0.0.1:<port>`, `localhost:<port>`, `LOCALHOST:<port>`, and `[::1]:<port>` -> 200 OK
    for ok_host in [
        format!("127.0.0.1:{port}"),
        format!("localhost:{port}"),
        format!("LOCALHOST:{port}"),
        format!("[::1]:{port}"),
    ] {
        let valid_req = format!(
            "GET /api/v1/health HTTP/1.1\r\n\
             Host: {ok_host}\r\n\
             Origin: http://{ok_host}\r\n\
             Connection: close\r\n\r\n"
        );
        assert_eq!(
            send_raw(&valid_req),
            200,
            "expected 200 OK for Host: {ok_host}"
        );
    }

    srv.shutdown();
    let _ = fs::remove_dir_all(&root);
}

// =============================================================================================
// Slice S6 Regression Tests: v0150:P1-8, v0150:P1-9, and Archive / Classifier P2 Hardening
// =============================================================================================

struct RawZipSpec<'a> {
    cd_name: &'a str,
    local_name: &'a str,
    method: u16,
    raw_comp_data: &'a [u8],
    declared_uncomp_size: u32,
    crc32: u32,
    local_offset_override: Option<u32>,
}

fn build_custom_zip(specs: &[RawZipSpec<'_>]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut offsets = Vec::with_capacity(specs.len());

    for sp in specs {
        if let Some(ov) = sp.local_offset_override {
            offsets.push(ov);
            continue;
        }
        let off = out.len() as u32;
        offsets.push(off);
        let lname = sp.local_name.as_bytes();
        out.extend_from_slice(b"PK\x03\x04");
        out.extend_from_slice(&20u16.to_le_bytes()); // version needed
        out.extend_from_slice(&0u16.to_le_bytes()); // flags
        out.extend_from_slice(&sp.method.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // mtime
        out.extend_from_slice(&0u16.to_le_bytes()); // mdate
        out.extend_from_slice(&sp.crc32.to_le_bytes());
        out.extend_from_slice(&(sp.raw_comp_data.len() as u32).to_le_bytes());
        out.extend_from_slice(&sp.declared_uncomp_size.to_le_bytes());
        out.extend_from_slice(&(lname.len() as u16).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // extra len
        out.extend_from_slice(lname);
        out.extend_from_slice(sp.raw_comp_data);
    }

    let cd_offset = out.len() as u32;
    for (sp, &loff) in specs.iter().zip(offsets.iter()) {
        let cname = sp.cd_name.as_bytes();
        out.extend_from_slice(b"PK\x01\x02");
        out.extend_from_slice(&20u16.to_le_bytes()); // version made by
        out.extend_from_slice(&20u16.to_le_bytes()); // version needed
        out.extend_from_slice(&0u16.to_le_bytes()); // flags
        out.extend_from_slice(&sp.method.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // mtime
        out.extend_from_slice(&0u16.to_le_bytes()); // mdate
        out.extend_from_slice(&sp.crc32.to_le_bytes());
        out.extend_from_slice(&(sp.raw_comp_data.len() as u32).to_le_bytes());
        out.extend_from_slice(&sp.declared_uncomp_size.to_le_bytes());
        out.extend_from_slice(&(cname.len() as u16).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // extra len
        out.extend_from_slice(&0u16.to_le_bytes()); // comment len
        out.extend_from_slice(&0u16.to_le_bytes()); // disk start
        out.extend_from_slice(&0u16.to_le_bytes()); // internal attr
        out.extend_from_slice(&0u32.to_le_bytes()); // external attr
        out.extend_from_slice(&loff.to_le_bytes());
        out.extend_from_slice(cname);
    }
    let cd_size = (out.len() as u32) - cd_offset;

    out.extend_from_slice(b"PK\x05\x06");
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&(specs.len() as u16).to_le_bytes());
    out.extend_from_slice(&(specs.len() as u16).to_le_bytes());
    out.extend_from_slice(&cd_size.to_le_bytes());
    out.extend_from_slice(&cd_offset.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes()); // comment len
    out
}

fn deflate_stored_block(payload: &[u8], bfinal: bool) -> Vec<u8> {
    assert!(payload.len() <= u16::MAX as usize);
    let mut out = Vec::with_capacity(5 + payload.len());
    out.push(if bfinal { 0x01 } else { 0x00 });
    let len = payload.len() as u16;
    let nlen = !len;
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(&nlen.to_le_bytes());
    out.extend_from_slice(payload);
    out
}

#[test]
fn test_v0150_p1_8_and_p2_zip_tar_integrity_streaming_alloc_and_path_parity() {
    // 1. In-memory archive cap: a 2 GiB synthetic tar buffer header or > 1 GiB uncompressed tar
    //    must NEVER be accepted under the 16 GiB disk cap when extracted in memory!
    {
        let mut two_gib_hdr_tar = Vec::with_capacity(1536);
        let mut hdr = [0u8; 512];
        let name = b"1/ser";
        hdr[..name.len()].copy_from_slice(name);
        hdr[100..107].copy_from_slice(b"0000644");
        hdr[108..115].copy_from_slice(b"0001000");
        hdr[116..123].copy_from_slice(b"0001000");
        let two_gib: u64 = 2 * 1024 * 1024 * 1024;
        hdr[124] = 0x80;
        hdr[128..136].copy_from_slice(&two_gib.to_be_bytes());
        hdr[136..147].copy_from_slice(b"14600000000");
        hdr[148..156].fill(b' ');
        hdr[156] = b'0';
        hdr[257..263].copy_from_slice(b"ustar\0");
        hdr[263..265].copy_from_slice(b"00");
        let csum: u32 = hdr.iter().map(|&b| b as u32).sum();
        hdr[148..156].copy_from_slice(format!("{csum:06o}\0 ").as_bytes());
        two_gib_hdr_tar.extend_from_slice(&hdr);
        two_gib_hdr_tar.extend_from_slice(&[0u8; 1024]);
        let gz_2gib_claim = {
            // Two-entry ZIP claiming > 1 GiB uncompressed retained header bytes
            build_custom_zip(&[RawZipSpec {
                cd_name: "1/acqus",
                local_name: "1/acqus",
                method: 0,
                raw_comp_data: b"##$PULPROG= <zg>\n",
                declared_uncomp_size: 1500 * 1024 * 1024,
                crc32: 0,
                local_offset_override: None,
            }])
        };
        assert!(extract_archive_virtual_entries("huge_claim.zip", &gz_2gib_claim).is_err());
    }

    let acqus = bruker_acqus("hsqcetf3gpsi", "1H", 600.13, 14.0, 2048, 0);
    let acqu2s = bruker_indirect_acqus("15N", 60.81, 32.0, 256, 6);
    let acqus_b = acqus.as_bytes();
    let acqu2s_b = acqu2s.as_bytes();

    // 2. Path normalization parity:
    //    - Benign `./` segments (`./1/acqus`, `bundle/./1/acqu2s`) succeed in BOTH `.tar` and `.zip`.
    //    - `..` traversal segments (`../1/acqus`, `bundle/../../evil/acqus`) fail in BOTH `.tar` and `.zip`.
    let mut benign_tar = Vec::new();
    append_tar_entry(&mut benign_tar, "./1/acqus", acqus_b);
    append_tar_entry(&mut benign_tar, "./1/./acqu2s", acqu2s_b);
    benign_tar.extend(std::iter::repeat_n(0u8, 1024));
    let tar_ok = extract_archive_virtual_entries("benign.tar", &benign_tar)
        .expect("tar with benign ./ segments must succeed");
    assert_eq!(tar_ok.len(), 2);
    assert_eq!(tar_ok[0].relative_path, "1/acqus");
    assert_eq!(tar_ok[1].relative_path, "1/acqu2s");

    let benign_zip = build_custom_zip(&[
        RawZipSpec {
            cd_name: "./1/acqus",
            local_name: "./1/acqus",
            method: 0,
            raw_comp_data: acqus_b,
            declared_uncomp_size: acqus_b.len() as u32,
            crc32: crc32_ieee(acqus_b),
            local_offset_override: None,
        },
        RawZipSpec {
            cd_name: "./1/./acqu2s",
            local_name: "./1/./acqu2s",
            method: 8,
            raw_comp_data: &deflate_stored_block(acqu2s_b, true),
            declared_uncomp_size: acqu2s_b.len() as u32,
            crc32: crc32_ieee(acqu2s_b),
            local_offset_override: None,
        },
    ]);
    let zip_ok = extract_archive_virtual_entries("benign.zip", &benign_zip)
        .expect("ZIP with benign ./ segments must succeed");
    assert_eq!(zip_ok.len(), 2);
    assert_eq!(zip_ok[0].relative_path, "1/acqus");
    assert_eq!(zip_ok[1].relative_path, "1/acqu2s");

    for bad_path in ["../1/acqus", "bundle/../../evil/acqus", "a/../1/acqus"] {
        let mut trav_tar = Vec::new();
        append_tar_entry(&mut trav_tar, bad_path, acqus_b);
        trav_tar.extend(std::iter::repeat_n(0u8, 1024));
        let err_t = extract_archive_virtual_entries("traversal.tar", &trav_tar)
            .expect_err("tar '..' path traversal must be rejected");
        assert!(err_t.contains("traversal"), "{err_t}");

        let trav_zip = build_custom_zip(&[RawZipSpec {
            cd_name: bad_path,
            local_name: bad_path,
            method: 0,
            raw_comp_data: acqus_b,
            declared_uncomp_size: acqus_b.len() as u32,
            crc32: crc32_ieee(acqus_b),
            local_offset_override: None,
        }]);
        let err_z = extract_archive_virtual_entries("traversal.zip", &trav_zip)
            .expect_err("ZIP '..' path traversal must be rejected");
        assert!(err_z.contains("traversal"), "{err_z}");
    }

    // 3. ZIP shared / overlapping `local_offset` (classic shared-payload bomb) -> rejected!
    let overlap_zip = build_custom_zip(&[
        RawZipSpec {
            cd_name: "1/acqus",
            local_name: "1/acqus",
            method: 0,
            raw_comp_data: acqus_b,
            declared_uncomp_size: acqus_b.len() as u32,
            crc32: crc32_ieee(acqus_b),
            local_offset_override: None,
        },
        RawZipSpec {
            cd_name: "2/acqus",
            local_name: "1/acqus",
            method: 0,
            raw_comp_data: acqus_b,
            declared_uncomp_size: acqus_b.len() as u32,
            crc32: crc32_ieee(acqus_b),
            local_offset_override: Some(0),
        },
    ]);
    let ov_err = extract_archive_virtual_entries("overlap.zip", &overlap_zip)
        .expect_err("shared/overlapping ZIP local_offset must be rejected");
    assert!(ov_err.contains("overlapping"), "{ov_err}");

    // 4. ZIP local-file-header filename mismatch vs Central Directory filename -> rejected!
    let mismatch_name_zip = build_custom_zip(&[RawZipSpec {
        cd_name: "1/acqus",
        local_name: "1/spoof",
        method: 0,
        raw_comp_data: acqus_b,
        declared_uncomp_size: acqus_b.len() as u32,
        crc32: crc32_ieee(acqus_b),
        local_offset_override: None,
    }]);
    let nm_err = extract_archive_virtual_entries("mismatch.zip", &mismatch_name_zip)
        .expect_err("local vs CD filename mismatch must be rejected");
    assert!(nm_err.contains("local header name"), "{nm_err}");

    // 5. Duplicate normalized entry paths (`"1/acqus"` and `"./1/acqus"`) -> rejected in BOTH `.zip` & `.tar`!
    let dup_zip = build_custom_zip(&[
        RawZipSpec {
            cd_name: "1/acqus",
            local_name: "1/acqus",
            method: 0,
            raw_comp_data: acqus_b,
            declared_uncomp_size: acqus_b.len() as u32,
            crc32: crc32_ieee(acqus_b),
            local_offset_override: None,
        },
        RawZipSpec {
            cd_name: "./1/acqus",
            local_name: "./1/acqus",
            method: 0,
            raw_comp_data: acqus_b,
            declared_uncomp_size: acqus_b.len() as u32,
            crc32: crc32_ieee(acqus_b),
            local_offset_override: None,
        },
    ]);
    let dup_z_err = extract_archive_virtual_entries("dup.zip", &dup_zip)
        .expect_err("duplicate normalized ZIP paths must be rejected");
    assert!(dup_z_err.contains("duplicate"), "{dup_z_err}");

    let mut dup_tar = Vec::new();
    append_tar_entry(&mut dup_tar, "1/acqus", acqus_b);
    append_tar_entry(&mut dup_tar, "./1/acqus", acqus_b);
    dup_tar.extend(std::iter::repeat_n(0u8, 1024));
    let dup_t_err = extract_archive_virtual_entries("dup.tar", &dup_tar)
        .expect_err("duplicate normalized tar paths must be rejected");
    assert!(dup_t_err.contains("duplicate"), "{dup_t_err}");

    // 6. Verify BFINAL + end-of-deflate stream when `read_limit == uncomp_size`:
    //    (a) Stored DEFLATE block with `BFINAL = 0` and no final block after `uncomp_size` -> Err.
    let non_bfinal_deflate = deflate_stored_block(acqus_b, false);
    let non_bfinal_zip = build_custom_zip(&[RawZipSpec {
        cd_name: "1/acqus",
        local_name: "1/acqus",
        method: 8,
        raw_comp_data: &non_bfinal_deflate,
        declared_uncomp_size: acqus_b.len() as u32,
        crc32: crc32_ieee(acqus_b),
        local_offset_override: None,
    }]);
    assert!(
        extract_archive_virtual_entries("non_bfinal.zip", &non_bfinal_zip).is_err(),
        "DEFLATE stream without BFINAL termination must be rejected"
    );

    //    (b) DEFLATE stream that inflates BEYOND declared `uncomp_size` (matching CRC of prefix!) -> Err.
    let mut extended_payload = acqus_b.to_vec();
    extended_payload.extend_from_slice(b"EXTRA_TRAILING_BYTES");
    let overlong_deflate = deflate_stored_block(&extended_payload, true);
    let overlong_zip = build_custom_zip(&[RawZipSpec {
        cd_name: "1/acqus",
        local_name: "1/acqus",
        method: 8,
        raw_comp_data: &overlong_deflate,
        declared_uncomp_size: acqus_b.len() as u32,
        crc32: crc32_ieee(acqus_b),
        local_offset_override: None,
    }]);
    assert!(
        extract_archive_virtual_entries("overlong.zip", &overlong_zip).is_err(),
        "DEFLATE stream inflating beyond declared uncomp_size must be rejected"
    );

    //    (c) Valid `BFINAL = 1` stored DEFLATE block (and fixed-Huffman block) followed by
    //        trailing unread bytes inside `comp_size` -> Err (`has_trailing_bytes()`).
    let mut trailing_stored = deflate_stored_block(acqus_b, true);
    trailing_stored.extend_from_slice(b"TRAILING_GARBAGE");
    let trailing_stored_zip = build_custom_zip(&[RawZipSpec {
        cd_name: "1/acqus",
        local_name: "1/acqus",
        method: 8,
        raw_comp_data: &trailing_stored,
        declared_uncomp_size: acqus_b.len() as u32,
        crc32: crc32_ieee(acqus_b),
        local_offset_override: None,
    }]);
    let trailing_err = extract_archive_virtual_entries("trailing_stored.zip", &trailing_stored_zip)
        .expect_err("stored DEFLATE block with trailing bytes inside comp_size must be rejected");
    assert!(trailing_err.contains("trailing"), "{trailing_err}");

    let trailing_huff_zip = build_custom_zip(&[RawZipSpec {
        cd_name: "1/acqus",
        local_name: "1/acqus",
        method: 8,
        raw_comp_data: &[0x03, 0x00, 0xAA],
        declared_uncomp_size: 0,
        crc32: 0,
        local_offset_override: None,
    }]);
    let trailing_huff_err =
        extract_archive_virtual_entries("trailing_huff.zip", &trailing_huff_zip).expect_err(
            "Huffman DEFLATE block with trailing bytes inside comp_size must be rejected",
        );
    assert!(
        trailing_huff_err.contains("trailing"),
        "{trailing_huff_err}"
    );

    // 7. Streaming allocation in ZIP & tar inflate:
    //    Tiny ~120 B ZIP and ~528 B tar declaring `400 MiB` (`419_430_400` bytes) over a 4-byte
    //    truncated stream must fail in < 50 ms without pre-allocating 400 MiB!
    let t0 = std::time::Instant::now();
    let tiny_400m_zip = build_custom_zip(&[RawZipSpec {
        cd_name: "1/acqus",
        local_name: "1/acqus",
        method: 8,
        raw_comp_data: &[0x01, 0x00, 0x00, 0xFF],
        declared_uncomp_size: 400 * 1024 * 1024,
        crc32: 0,
        local_offset_override: None,
    }]);
    assert!(tiny_400m_zip.len() < 160);
    assert!(extract_archive_full_entries_bounded(
        "tiny_400m.zip",
        &tiny_400m_zip,
        512 * 1024 * 1024,
        512 * 1024 * 1024,
    )
    .is_err());

    let mut tiny_400m_tar = Vec::with_capacity(528);
    {
        let mut hdr = [0u8; 512];
        let name = b"1/acqus";
        hdr[..name.len()].copy_from_slice(name);
        hdr[100..107].copy_from_slice(b"0000644");
        hdr[108..115].copy_from_slice(b"0001000");
        hdr[116..123].copy_from_slice(b"0001000");
        let sz = 400usize * 1024 * 1024;
        hdr[124..135].copy_from_slice(format!("{sz:011o}").as_bytes());
        hdr[136..147].copy_from_slice(b"14600000000");
        hdr[148..156].fill(b' ');
        hdr[156] = b'0';
        hdr[257..263].copy_from_slice(b"ustar\0");
        hdr[263..265].copy_from_slice(b"00");
        let csum: u32 = hdr.iter().map(|&b| b as u32).sum();
        hdr[148..156].copy_from_slice(format!("{csum:06o}\0 ").as_bytes());
        tiny_400m_tar.extend_from_slice(&hdr);
        tiny_400m_tar.extend_from_slice(&[0u8; 16]);
    }
    assert!(extract_archive_full_entries_bounded(
        "tiny_400m.tar",
        &tiny_400m_tar,
        512 * 1024 * 1024,
        512 * 1024 * 1024,
    )
    .is_err());
    assert!(
        t0.elapsed().as_millis() < 50,
        "streaming allocation rejection took {:?}",
        t0.elapsed()
    );
}

#[test]
fn test_v0150_p1_9_hcchdi_hbhacon_t1ir_and_no_invented_noesy_cosy_fallback() {
    // 1. `hcchdigp3d` (`HCCH-TOCSY` with 1H, 13C, 1H axes) -> `HcchTocsy`, NEVER `C13Noesy`!
    let inv_hcch = classify_nmr_entries(&[
        VirtualEntry::new(
            "10_hcch/acqus",
            bruker_acqus("hcchdigp3d", "1H", 600.13, 12.0, 2048, 0).into_bytes(),
        ),
        VirtualEntry::new(
            "10_hcch/acqu2s",
            bruker_indirect_acqus("1H", 600.13, 10.0, 128, 5).into_bytes(),
        ),
        VirtualEntry::new(
            "10_hcch/acqu3s",
            bruker_indirect_acqus("13C", 150.90, 70.0, 64, 6).into_bytes(),
        ),
        VirtualEntry::header_only("10_hcch/ser", Vec::new(), 65536),
    ]);
    assert_eq!(
        inv_hcch.experiments[0].pulse_class,
        PulseSequenceClass::HcchTocsy,
        "hcchdigp3d must classify as HcchTocsy, never C13Noesy"
    );

    // 2. `hbhaconhgp3d` & Varian `ghbha_co_nh` (`HBHA(CO)NH` with 1H, 15N, 1H axes) -> `Unknown`, NEVER `N15Noesy`!
    let inv_hbha = classify_nmr_entries(&[
        VirtualEntry::new(
            "20_hbha/acqus",
            bruker_acqus("hbhaconhgp3d", "1H", 600.13, 14.0, 2048, 0).into_bytes(),
        ),
        VirtualEntry::new(
            "20_hbha/acqu2s",
            bruker_indirect_acqus("1H", 600.13, 8.0, 128, 5).into_bytes(),
        ),
        VirtualEntry::new(
            "20_hbha/acqu3s",
            bruker_indirect_acqus("15N", 60.81, 32.0, 64, 6).into_bytes(),
        ),
        VirtualEntry::header_only("20_hbha/ser", Vec::new(), 65536),
        VirtualEntry::new(
            "21_varian_hbha.fid/procpar",
            varian_procpar_3d(
                "ghbha_co_nh",
                600.13,
                600.13,
                60.81,
                "H1",
                "N15",
                8400.0,
                4800.0,
                1945.0,
                128,
                64,
            )
            .into_bytes(),
        ),
        VirtualEntry::header_only("21_varian_hbha.fid/fid", Vec::new(), 65536),
    ]);
    assert_eq!(inv_hbha.experiments.len(), 2);
    for exp in &inv_hbha.experiments {
        assert_eq!(
            exp.pulse_class,
            PulseSequenceClass::Unknown,
            "HBHA(CO)NH ({}) must return Unknown, never N15Noesy",
            exp.relative_path
        );
        assert!(!exp.selected_by_default);
    }

    // 3. `t1ir` + `vdlist` inversion-recovery T1 -> `Pseudo3D` `RelaxR1`, NEVER `Cosy2D`!
    let inv_t1ir = classify_nmr_entries(&[
        VirtualEntry::new(
            "30_t1ir/acqus",
            bruker_acqus("t1ir", "1H", 500.13, 14.0, 16384, 0).into_bytes(),
        ),
        VirtualEntry::new(
            "30_t1ir/acqu2s",
            bruker_indirect_acqus("off", 500.13, 10.0, 8, 1).into_bytes(),
        ),
        VirtualEntry::new(
            "30_t1ir/vdlist",
            b"0.01\n0.05\n0.1\n0.25\n0.5\n1.0\n2.0\n5.0\n".to_vec(),
        ),
        VirtualEntry::header_only("30_t1ir/ser", Vec::new(), 65536),
    ]);
    assert_eq!(
        inv_t1ir.experiments[0].dimensionality,
        ExperimentDimensionality::Pseudo3D
    );
    assert_eq!(
        inv_t1ir.experiments[0].pulse_class,
        PulseSequenceClass::RelaxR1,
        "t1ir + vdlist must classify as RelaxR1, never Cosy2D"
    );

    // 4. Axis-physics fallback on unlabelled / unknown pulse sequences MUST return `Unknown`,
    //    NEVER inventing `Cosy2D`, `Noesy2D`, `N15Noesy`, or `C13Noesy`!
    let unnamed_3d_nhh = build_synthetic_ucsf_3d(
        ("1H", 512, 600.13, 7200.0, 4.7),
        ("15N", 64, 60.81, 1945.0, 118.0),
        ("1H", 128, 600.13, 7200.0, 4.7),
    );
    let unnamed_3d_chh = build_synthetic_ucsf_3d(
        ("1H", 512, 600.13, 7200.0, 4.7),
        ("13C", 64, 150.90, 9000.0, 40.0),
        ("1H", 128, 600.13, 7200.0, 4.7),
    );
    let inv_ambig = classify_nmr_entries(&[
        // Unknown 2D 1H-1H Bruker experiment
        VirtualEntry::new(
            "40_unknown_hh/acqus",
            bruker_acqus("mystery_lab_seq_2d", "1H", 600.13, 12.0, 2048, 0).into_bytes(),
        ),
        VirtualEntry::new(
            "40_unknown_hh/acqu2s",
            bruker_indirect_acqus("1H", 600.13, 12.0, 256, 5).into_bytes(),
        ),
        VirtualEntry::header_only("40_unknown_hh/ser", Vec::new(), 32768),
        // Unknown 2D 1H-Delay QF array
        VirtualEntry::new(
            "41_unknown_qf/acqus",
            bruker_acqus("mystery_kinetics_2d", "1H", 600.13, 12.0, 2048, 0).into_bytes(),
        ),
        VirtualEntry::new(
            "41_unknown_qf/acqu2s",
            bruker_indirect_acqus("off", 600.13, 10.0, 16, 1).into_bytes(),
        ),
        VirtualEntry::header_only("41_unknown_qf/ser", Vec::new(), 32768),
        // Unnamed 3D (1H, 15N, 1H) and (1H, 13C, 1H) UCSF cubes
        VirtualEntry::new("42_unnamed_nhh.ucsf", unnamed_3d_nhh),
        VirtualEntry::new("43_unnamed_chh.ucsf", unnamed_3d_chh),
    ]);
    assert_eq!(inv_ambig.experiments.len(), 4);
    for exp in &inv_ambig.experiments {
        assert_eq!(
            exp.pulse_class,
            PulseSequenceClass::Unknown,
            "ambiguous axes on '{}' must return Unknown, got {:?}",
            exp.relative_path,
            exp.pulse_class
        );
    }
}

#[test]
fn test_v0150_p2_nucleus_inference_31p_1h_vs_13c_15n_and_fqlist_indirect_loop_gating() {
    // 1. Disambiguate `31P/1H` (~0.4048, e.g. 202.46 / 500.13 MHz) vs `13C/15N` (~0.40298, e.g. 150.90 / 60.81 MHz)
    //    when Bruker processed `procs`/`proc2s` have `AXNUC = <off>`.
    let p1_1h = b"##$SI= 2048\n##$SF= 500.13\n##$SW_p= 6000.0\n##$OFFSET= 10.0\n##$AXNUC= <off>\n";
    let p2_31p = b"##$SI= 256\n##$SF= 202.46\n##$SW_p= 8098.0\n##$OFFSET= 20.0\n##$AXNUC= <off>\n";
    let inv_hp = classify_nmr_entries(&[
        VirtualEntry::new("hp/procs", p1_1h.to_vec()),
        VirtualEntry::new("hp/proc2s", p2_31p.to_vec()),
        VirtualEntry::header_only("hp/2rr", Vec::new(), 2048 * 256 * 4),
    ]);
    assert_eq!(inv_hp.experiments[0].axes[0].nucleus, "1H");
    assert_eq!(inv_hp.experiments[0].axes[1].nucleus, "31P");
    assert_eq!(
        inv_hp.experiments[0].pulse_class,
        PulseSequenceClass::Hetero2D,
        "500/202 MHz AXNUC=<off> must be 1H-31P Hetero2D, never 13C-15N Con2D"
    );

    // Reversed direct `31P` (202.46 MHz) x indirect `1H` (500.13 MHz), ratio = 2.4703
    let inv_ph = classify_nmr_entries(&[
        VirtualEntry::new("ph/procs", p2_31p.to_vec()),
        VirtualEntry::new("ph/proc2s", p1_1h.to_vec()),
        VirtualEntry::header_only("ph/2rr", Vec::new(), 2048 * 256 * 4),
    ]);
    assert_eq!(inv_ph.experiments[0].axes[0].nucleus, "31P");
    assert_eq!(inv_ph.experiments[0].axes[1].nucleus, "1H");
    assert_eq!(
        inv_ph.experiments[0].pulse_class,
        PulseSequenceClass::Hetero2D
    );

    // Genuine 600 MHz `13C/15N` CON (`150.90 / 60.81 MHz`, ratio = 0.40298) remains `13C`/`15N` `Con2D`
    let p1_13c =
        b"##$SI= 1024\n##$SF= 150.90\n##$SW_p= 3018.0\n##$OFFSET= 185.0\n##$AXNUC= <off>\n";
    let p2_15n = b"##$SI= 512\n##$SF= 60.81\n##$SW_p= 2189.0\n##$OFFSET= 138.0\n##$AXNUC= <off>\n";
    let inv_con = classify_nmr_entries(&[
        VirtualEntry::new("con/procs", p1_13c.to_vec()),
        VirtualEntry::new("con/proc2s", p2_15n.to_vec()),
        VirtualEntry::header_only("con/2rr", Vec::new(), 1024 * 512 * 4),
    ]);
    assert_eq!(inv_con.experiments[0].axes[0].nucleus, "13C");
    assert_eq!(inv_con.experiments[0].axes[1].nucleus, "15N");
    assert_eq!(
        inv_con.experiments[0].pulse_class,
        PulseSequenceClass::Con2D
    );

    // 2. `fq*list` gating: selective-pulse offset list (`fq1:f2` without `ifq*` or QF TD match)
    //    must NOT promote a 2D spectrum to `Pseudo3D`!
    let fq_offsets = b"bf ppm\n170.0\n55.0\n175.0\n52.0\n172.0\n58.0\n174.0\n54.0\n";
    let inv_sel_fq = classify_nmr_entries(&[
        VirtualEntry::new(
            "sel_2d/acqus",
            b"##$PULPROG= <custom_band_sel_2d>\n##$NUC1= <1H>\n##$SFO1= 600.13\n##$SW= 14.0\n##$TD= 2048\n##$FQ1LIST= <sel_offsets>\n"
                .to_vec(),
        ),
        VirtualEntry::new(
            "sel_2d/acqu2s",
            b"##$NUC1= <15N>\n##$SFO1= 60.81\n##$SW= 32.0\n##$TD= 128\n##$FnMODE= 6\n".to_vec(),
        ),
        VirtualEntry::new(
            "sel_2d/pulseprogram",
            b";band-selective 2D HSQC\nd1 fq1:f2\np1 ph1\nd0\n10u fq2:f2\ngo=2 ph31\n".to_vec(),
        ),
        VirtualEntry::new("sel_2d/fq1list", fq_offsets.to_vec()),
        VirtualEntry::header_only("sel_2d/ser", Vec::new(), 65536),
    ]);
    assert_eq!(
        inv_sel_fq.experiments[0].dimensionality,
        ExperimentDimensionality::Dim2D,
        "intra-scan selective-pulse fq1list without ifq* or QF loop must remain Dim2D"
    );
    assert_ne!(
        inv_sel_fq.experiments[0].pulse_class,
        PulseSequenceClass::Cest
    );
}

#[test]
fn test_rd3_clec16a_real_data_pulse_program_classes_and_axis_validation() {
    #[allow(clippy::too_many_arguments)]
    fn mk_acq(
        pp: Option<&str>,
        nuc: &str,
        sfo1: f64,
        bf1: f64,
        sw: f64,
        td: usize,
        fnmode: usize,
        nustd: usize,
        fntype: usize,
    ) -> Vec<u8> {
        let pp_line = pp
            .map(|p| format!("##$PULPROG= <{p}>\n##$FnTYPE= {fntype}\n"))
            .unwrap_or_default();
        format!(
            "{pp_line}##$NUC1= <{nuc}>\n##$SFO1= {sfo1}\n##$BF1= {bf1}\n##$SW= {sw}\n##$TD= {td}\n##$FnMODE= {fnmode}\n##$NusTD= {nustd}\n##END=\n"
        )
        .into_bytes()
    }

    let mut entries = Vec::new();
    // Generic CON must stay Con2D alongside CAS experiments (no duplicate Con2D merge!)
    entries.push(VirtualEntry::new(
        "01_con/acqus",
        mk_acq(
            Some("c_hcacon_ia"),
            "13C",
            201.258,
            201.223,
            11.83,
            1024,
            0,
            0,
            0,
        ),
    ));
    entries.push(VirtualEntry::new(
        "01_con/acqu2s",
        mk_acq(None, "15N", 81.0987, 81.0886, 36.0, 128, 5, 128, 0),
    ));
    entries.push(VirtualEntry::header_only("01_con/ser", Vec::new(), 65536));

    // 1. Amino-acid-selective CAS-CACON (ala, gly, ser, tavi) & CAS-CANCO (ala, gly, ser, tavi)
    let cas_cases = [
        (
            "02_cacon_ala",
            "c_cacon_ala_iasq_official.sas",
            PulseSequenceClass::CasCacon2D,
            CasSelectionScope::Predecessor,
            "ALA",
            vec!["ALA"],
        ),
        (
            "03_cacon_gly",
            "c_cacon_gly_iasq_official.sas",
            PulseSequenceClass::CasCacon2D,
            CasSelectionScope::Predecessor,
            "GLY",
            vec!["GLY"],
        ),
        (
            "04_cacon_ser",
            "c_cacon_ser_iasq_official.sas",
            PulseSequenceClass::CasCacon2D,
            CasSelectionScope::Predecessor,
            "SER",
            vec!["SER"],
        ),
        (
            "05_cacon_tavi",
            "c_cacon_tavi_iasq_official.sas",
            PulseSequenceClass::CasCacon2D,
            CasSelectionScope::Predecessor,
            "TAVI",
            vec!["THR", "ALA", "VAL", "ILE"],
        ),
        (
            "06_canco_ala",
            "c_canco_ala_iasq_official.sas",
            PulseSequenceClass::CasCanco2D,
            CasSelectionScope::IntraAndPredecessor,
            "ALA",
            vec!["ALA"],
        ),
        (
            "07_canco_gly",
            "c_canco_gly_iasq_official.sas",
            PulseSequenceClass::CasCanco2D,
            CasSelectionScope::IntraAndPredecessor,
            "GLY",
            vec!["GLY"],
        ),
        (
            "08_canco_ser",
            "c_canco_ser_iasq_official.sas",
            PulseSequenceClass::CasCanco2D,
            CasSelectionScope::IntraAndPredecessor,
            "SER",
            vec!["SER"],
        ),
        (
            "09_canco_tavi",
            "c_canco_tavi_iasq_official.sas",
            PulseSequenceClass::CasCanco2D,
            CasSelectionScope::IntraAndPredecessor,
            "TAVI",
            vec!["THR", "ALA", "VAL", "ILE"],
        ),
    ];
    for (dir, pp, expected_cls, expected_scope, expected_tag, expected_aas) in &cas_cases {
        entries.push(VirtualEntry::new(
            format!("{dir}/acqus"),
            mk_acq(Some(pp), "13C", 201.258, 201.223, 11.83, 1024, 0, 0, 0),
        ));
        entries.push(VirtualEntry::new(
            format!("{dir}/acqu2s"),
            mk_acq(None, "15N", 81.0987, 81.0886, 36.0, 128, 5, 128, 0),
        ));
        entries.push(VirtualEntry::header_only(
            format!("{dir}/ser"),
            Vec::new(),
            65536,
        ));
        let _ = (expected_cls, expected_scope, expected_tag, expected_aas);
    }

    // 2. 13C-detected (H)N(CA)CON / (HACA)N(CA)CON N-N sequential correlation (`13C, 15N, 15N`) -> Hncacon3D
    for (dir, pp) in [
        ("10_hncacon", "c_hncacon_iare3d"),
        ("11_hacancacon", "c_hacancacon_official_ia3d.sas"),
    ] {
        entries.push(VirtualEntry::new(
            format!("{dir}/acqus"),
            mk_acq(Some(pp), "13C", 201.258, 201.223, 11.83, 1024, 0, 0, 2),
        ));
        entries.push(VirtualEntry::new(
            format!("{dir}/acqu2s"),
            mk_acq(None, "15N", 81.0987, 81.0886, 36.0, 1280, 5, 128, 0),
        ));
        entries.push(VirtualEntry::new(
            format!("{dir}/acqu3s"),
            mk_acq(None, "15N", 81.0987, 81.0886, 36.0, 1, 5, 40, 0),
        ));
        entries.push(VirtualEntry::header_only(
            format!("{dir}/ser"),
            Vec::new(),
            65536,
        ));
    }

    // Negative control: 2D plane of `c_hncacon_iare3d` (`13C, 15N`) stays Con2D
    entries.push(VirtualEntry::new(
        "11b_hncacon_2d/acqus",
        mk_acq(
            Some("c_hncacon_iare3d"),
            "13C",
            201.258,
            201.223,
            11.83,
            1024,
            0,
            0,
            0,
        ),
    ));
    entries.push(VirtualEntry::new(
        "11b_hncacon_2d/acqu2s",
        mk_acq(None, "15N", 81.0987, 81.0886, 36.0, 128, 5, 128, 0),
    ));
    entries.push(VirtualEntry::header_only(
        "11b_hncacon_2d/ser",
        Vec::new(),
        65536,
    ));

    // 3. 13C-detected `(H)CC(CO)N` (`13C, 15N, 13C`) -> Cccon3D (NOT 1H-detected Hccconh!)
    entries.push(VirtualEntry::new(
        "12_hcccon/acqus",
        mk_acq(
            Some("c_hcccon_ia3d"),
            "13C",
            201.258,
            201.223,
            8.01,
            1024,
            0,
            0,
            2,
        ),
    ));
    entries.push(VirtualEntry::new(
        "12_hcccon/acqu2s",
        mk_acq(None, "15N", 81.0987, 81.0886, 36.0, 1536, 5, 128, 0),
    ));
    entries.push(VirtualEntry::new(
        "12_hcccon/acqu3s",
        mk_acq(None, "13C", 201.232, 201.223, 79.88, 1, 5, 48, 0),
    ));
    entries.push(VirtualEntry::header_only(
        "12_hcccon/ser",
        Vec::new(),
        65536,
    ));

    // Genuine 3D `c_canco_ia3d` (`13C, 15N, 13C`) must remain Canco3D
    entries.push(VirtualEntry::new(
        "13_canco/acqus",
        mk_acq(
            Some("c_canco_ia3d"),
            "13C",
            201.258,
            201.223,
            11.83,
            1024,
            0,
            0,
            0,
        ),
    ));
    entries.push(VirtualEntry::new(
        "13_canco/acqu2s",
        mk_acq(None, "15N", 81.0987, 81.0886, 36.0, 64, 5, 0, 0),
    ));
    entries.push(VirtualEntry::new(
        "13_canco/acqu3s",
        mk_acq(None, "13C", 201.234, 201.223, 40.0, 64, 5, 0, 0),
    ));
    entries.push(VirtualEntry::header_only("13_canco/ser", Vec::new(), 65536));

    // 4. `BT_HNCACONH` (`1H, 15N, 15N`) -> Hncaconh (NOT Hncaco!)
    entries.push(VirtualEntry::new(
        "14_bt_hnn/acqus",
        mk_acq(
            Some("BT_HNCACONH"),
            "1H",
            800.254,
            800.25,
            12.01,
            1362,
            0,
            0,
            2,
        ),
    ));
    entries.push(VirtualEntry::new(
        "14_bt_hnn/acqu2s",
        mk_acq(None, "15N", 81.0982, 81.0886, 36.0, 4096, 6, 64, 0),
    ));
    entries.push(VirtualEntry::new(
        "14_bt_hnn/acqu3s",
        mk_acq(None, "15N", 81.0982, 81.0886, 36.0, 1, 5, 128, 0),
    ));
    entries.push(VirtualEntry::header_only(
        "14_bt_hnn/ser",
        Vec::new(),
        65536,
    ));

    let inv = classify_nmr_entries(&entries);
    assert_eq!(inv.problem_regime, ProblemRegime::GlobularBackbone);
    assert!(inv
        .secondary_regimes
        .contains(&ProblemRegime::Idp13cDetected));

    let by_path = |rel: &str| {
        inv.experiments
            .iter()
            .find(|e| e.relative_path == rel)
            .unwrap_or_else(|| panic!("missing experiment {rel}"))
    };
    assert_eq!(by_path("01_con").pulse_class, PulseSequenceClass::Con2D);
    assert!(by_path("01_con").cas_residue_filter().is_none());

    let con2d_count = inv
        .experiments
        .iter()
        .filter(|e| e.pulse_class == PulseSequenceClass::Con2D)
        .count();
    assert_eq!(
        con2d_count, 2,
        "only generic CONs should be Con2D, never CAS"
    );

    for (dir, _, expected_cls, expected_scope, expected_tag, expected_aas) in &cas_cases {
        let exp = by_path(dir);
        assert_eq!(exp.pulse_class, *expected_cls, "{dir}");
        assert_eq!(
            exp.axes
                .iter()
                .map(|a| a.nucleus.as_str())
                .collect::<Vec<_>>(),
            vec!["13C", "15N"]
        );
        let f = exp.cas_residue_filter().expect("CAS filter metadata");
        assert_eq!(f.scope, *expected_scope, "{dir}");
        assert_eq!(f.filter_tag, *expected_tag, "{dir}");
        assert_eq!(f.residue_types, *expected_aas, "{dir}");
    }

    for dir in ["10_hncacon", "11_hacancacon"] {
        let exp = by_path(dir);
        assert_eq!(exp.pulse_class, PulseSequenceClass::Hncacon3D, "{dir}");
        assert_eq!(exp.pulse_class.canonical_solver_exp(), None);
        assert_eq!(
            exp.axes
                .iter()
                .map(|a| a.nucleus.as_str())
                .collect::<Vec<_>>(),
            vec!["13C", "15N", "15N"]
        );
    }
    assert_eq!(
        by_path("11b_hncacon_2d").pulse_class,
        PulseSequenceClass::Con2D
    );

    let cccon = by_path("12_hcccon");
    assert_eq!(cccon.pulse_class, PulseSequenceClass::Cccon3D);
    assert_eq!(cccon.pulse_class.canonical_solver_exp(), Some("CCCON"));
    assert!(cccon.pulse_class.is_13c_detected());
    assert_eq!(
        cccon
            .axes
            .iter()
            .map(|a| a.nucleus.as_str())
            .collect::<Vec<_>>(),
        vec!["13C", "15N", "13C"]
    );

    assert_eq!(by_path("13_canco").pulse_class, PulseSequenceClass::Canco3D);

    let hnn = by_path("14_bt_hnn");
    assert_eq!(hnn.pulse_class, PulseSequenceClass::Hncaconh);
    assert_eq!(
        hnn.axes
            .iter()
            .map(|a| a.nucleus.as_str())
            .collect::<Vec<_>>(),
        vec!["1H", "15N", "15N"]
    );

    // Axis-mismatch protection (`RD-3`): `c_canco_ia3d` with wrong `13C, 15N, 15N` axes MUST NOT
    // trust the `canco` substring to claim `Canco3D`!
    let mismatched_inv = classify_nmr_entries(&[
        VirtualEntry::new(
            "bad_canco/acqus",
            mk_acq(
                Some("c_canco_ia3d"),
                "13C",
                201.258,
                201.223,
                11.83,
                1024,
                0,
                0,
                0,
            ),
        ),
        VirtualEntry::new(
            "bad_canco/acqu2s",
            mk_acq(None, "15N", 81.0987, 81.0886, 36.0, 64, 5, 0, 0),
        ),
        VirtualEntry::new(
            "bad_canco/acqu3s",
            mk_acq(None, "15N", 81.0987, 81.0886, 36.0, 64, 5, 0, 0),
        ),
        VirtualEntry::header_only("bad_canco/ser", Vec::new(), 4096),
    ]);
    assert_ne!(
        mismatched_inv.experiments[0].pulse_class,
        PulseSequenceClass::Canco3D
    );
}

#[test]
fn test_rd6_and_rd9_nus_indirect_td_and_processed_axis_range_warnings() {
    // RD-6: `b_hncogp3d.2` with Bruker NUS bookkeeping (`acqu2s TD=4096`, `acqu3s TD=1`)
    // MUST derive indirect points from `NusTD`/`nuslist`/`proc SI` (NOT 4096 or 1), plus
    // test fallback when `NusTD = 0` using `nuslist` col extents (`31, 63` -> `64, 128`).
    let mut nuslist = String::new();
    for i in 0..1024usize {
        let a = (i * 7) % 32;
        let b = (i * 11) % 64;
        nuslist.push_str(&format!("{a} {b}\n"));
    }

    let inv_nustd = classify_nmr_entries(&[
        VirtualEntry::new(
            "hnco_nus/acqus",
            b"##$PULPROG= <b_hncogp3d.2>\n##$FnTYPE= 2\n##$NUC1= <1H>\n##$SFO1= 800.25\n##$BF1= 800.25\n##$SW= 14.2\n##$TD= 698\n##$NusTD= 0\n".to_vec(),
        ),
        VirtualEntry::new(
            "hnco_nus/acqu2s",
            b"##$NUC1= <15N>\n##$SFO1= 81.09\n##$BF1= 81.0886\n##$SW= 35.0\n##$TD= 4096\n##$FnMODE= 6\n##$NusTD= 64\n".to_vec(),
        ),
        VirtualEntry::new(
            "hnco_nus/acqu3s",
            b"##$NUC1= <13C>\n##$SFO1= 201.258\n##$BF1= 201.223\n##$SW= 14.0\n##$TD= 1\n##$FnMODE= 5\n##$NusTD= 128\n".to_vec(),
        ),
        VirtualEntry::new("hnco_nus/nuslist", nuslist.as_bytes().to_vec()),
        VirtualEntry::new("hnco_nus/pdata/1/procs", b"##$SI= 4096\n##$XDIM= 4\n##$SW_p= 11363.6\n##$SF= 800.25\n##$OFFSET= 11.8\n".to_vec()),
        VirtualEntry::new("hnco_nus/pdata/1/proc2s", b"##$SI= 128\n##$XDIM= 8\n##$SW_p= 2838.4\n##$SF= 81.0886\n##$OFFSET= 134.5\n".to_vec()),
        VirtualEntry::new("hnco_nus/pdata/1/proc3s", b"##$SI= 256\n##$XDIM= 8\n##$SW_p= 2817.6\n##$SF= 201.223\n##$OFFSET= 180.5\n".to_vec()),
        VirtualEntry::header_only("hnco_nus/pdata/1/3rrr", Vec::new(), 4096),
    ]);
    let exp = &inv_nustd.experiments[0];
    assert_eq!(exp.pulse_class, PulseSequenceClass::Hnco);
    assert_eq!(exp.axes[0].points, 698);
    assert_eq!(exp.axes[1].points, 64, "must NOT report acqu2s TD=4096");
    assert_eq!(exp.axes[2].points, 128, "must NOT report acqu3s TD=1");
    assert!(
        exp.notes
            .iter()
            .any(|n| n.contains("procSI=128") && n.contains("procSI=256")),
        "expected labeled NUS procSI note, got {:?}",
        exp.notes
    );
    assert!(inv_nustd
        .missing_input_hints
        .iter()
        .all(|h| !h.contains("Axis warning")));

    // When `NusTD = 0`, falls back to `nuslist` col extents (`31, 63` -> `64, 128`)
    let inv_nl = classify_nmr_entries(&[
        VirtualEntry::new(
            "hnco_nl/acqus",
            b"##$PULPROG= <b_hncogp3d.2>\n##$FnTYPE= 2\n##$NUC1= <1H>\n##$SFO1= 800.25\n##$BF1= 800.25\n##$SW= 14.2\n##$TD= 698\n".to_vec(),
        ),
        VirtualEntry::new(
            "hnco_nl/acqu2s",
            b"##$NUC1= <15N>\n##$SFO1= 81.09\n##$SW= 35.0\n##$TD= 4096\n##$FnMODE= 6\n##$NusTD= 0\n".to_vec(),
        ),
        VirtualEntry::new(
            "hnco_nl/acqu3s",
            b"##$NUC1= <13C>\n##$SFO1= 201.258\n##$SW= 14.0\n##$TD= 1\n##$FnMODE= 5\n##$NusTD= 0\n".to_vec(),
        ),
        VirtualEntry::new("hnco_nl/nuslist", nuslist.into_bytes()),
        VirtualEntry::header_only("hnco_nl/ser", Vec::new(), 4096),
    ]);
    assert_eq!(inv_nl.experiments[0].axes[1].points, 64);
    assert_eq!(inv_nl.experiments[0].axes[2].points, 128);

    // RD-9: `c_hcccon_ia3d` where `proc3s` left `SF = BF1` and `OFFSET = 211.95` (carrier 172 ppm)
    let inv_bad_ref = classify_nmr_entries(&[
        VirtualEntry::new(
            "cccon_badref/acqus",
            b"##$PULPROG= <c_hcccon_ia3d>\n##$FnTYPE= 2\n##$NUC1= <13C>\n##$SFO1= 201.2576\n##$BF1= 201.22302255\n##$SW= 8.01\n##$TD= 1024\n".to_vec(),
        ),
        VirtualEntry::new(
            "cccon_badref/acqu2s",
            b"##$NUC1= <15N>\n##$SFO1= 81.09868\n##$BF1= 81.08862828\n##$SW= 35.99\n##$TD= 1536\n##$NusTD= 128\n##$FnMODE= 5\n".to_vec(),
        ),
        VirtualEntry::new(
            "cccon_badref/acqu3s",
            b"##$NUC1= <13C>\n##$SFO1= 201.25763\n##$BF1= 201.22302255\n##$SW= 79.88\n##$TD= 1\n##$NusTD= 48\n##$FnMODE= 5\n".to_vec(),
        ),
        VirtualEntry::new("cccon_badref/pdata/1/procs", b"##$SI= 2048\n##$XDIM= 4\n##$SW_p= 1612.9\n##$SF= 201.22302255\n##$OFFSET= 176.0\n".to_vec()),
        VirtualEntry::new("cccon_badref/pdata/1/proc2s", b"##$SI= 256\n##$XDIM= 8\n##$SW_p= 2918.9\n##$SF= 81.08862828\n##$OFFSET= 142.0\n".to_vec()),
        VirtualEntry::new("cccon_badref/pdata/1/proc3s", b"##$SI= 128\n##$XDIM= 8\n##$SW_p= 16077.17\n##$SF= 201.22302255\n##$OFFSET= 211.9486\n".to_vec()),
        VirtualEntry::header_only("cccon_badref/pdata/1/3rrr", Vec::new(), 4096),
    ]);
    assert_eq!(
        inv_bad_ref.experiments[0].pulse_class,
        PulseSequenceClass::Cccon3D
    );
    assert!(
        inv_bad_ref.experiments[0]
            .notes
            .iter()
            .any(|n| n.contains("Axis warning (F1)") && n.contains("SR/SF")),
        "expected RD-9 axis warning on F1, notes={:?}",
        inv_bad_ref.experiments[0].notes
    );
    assert!(
        inv_bad_ref
            .missing_input_hints
            .iter()
            .any(|h| h.contains("cccon_badref") && h.contains("SF == BF")),
        "hints={:?}",
        inv_bad_ref.missing_input_hints
    );
}

#[test]
fn test_v0173_r2_p1_8_proc_axes_swapped_2d_ucsf_and_unknown_conflict_note() {
    // 1. Swapped 2D UCSF (`CACO` and `CON`): `parse_ucsf_header` puts direct 13C' first;
    //    `append_processed_axis_warnings` must NOT reverse indices and emit spurious RD-9 warnings.
    let build_ucsf_2d = |axes: [(&str, i32, f32, f32, f32); 2]| -> Vec<u8> {
        let mut buf = vec![0u8; 180 + 2 * 128];
        buf[0..10].copy_from_slice(b"UCSF NMR\0\0");
        buf[10] = 2; // 2D
        buf[11] = 1;
        buf[13] = 2;
        for (d, &(nuc, npts, obs, sw_hz, xmtr)) in axes.iter().enumerate() {
            let off = 180 + d * 128;
            let nb = nuc.as_bytes();
            buf[off..off + nb.len().min(6)].copy_from_slice(&nb[..nb.len().min(6)]);
            buf[off + 8..off + 12].copy_from_slice(&npts.to_be_bytes());
            buf[off + 20..off + 24].copy_from_slice(&obs.to_be_bytes());
            buf[off + 24..off + 28].copy_from_slice(&sw_hz.to_be_bytes());
            buf[off + 28..off + 32].copy_from_slice(&xmtr.to_be_bytes());
        }
        buf
    };
    let caco_aliph_first = build_ucsf_2d([
        ("13C", 256, 201.22, 16097.6, 45.0),
        ("13C", 512, 201.22, 3018.3, 174.0),
    ]);
    let con_n_first = build_ucsf_2d([
        ("15N", 128, 81.09, 2919.2, 122.0),
        ("13C", 512, 201.22, 2414.6, 174.0),
    ]);
    let caco_bad_ref = build_ucsf_2d([
        ("13C", 256, 201.22, 16097.6, 172.0),
        ("13C", 512, 201.22, 3018.3, 174.0),
    ]);
    let inv_ucsf = classify_nmr_entries(&[
        VirtualEntry::new("CACO_good.ucsf", caco_aliph_first),
        VirtualEntry::new("CON_good.ucsf", con_n_first),
        VirtualEntry::new("CACO_bad.ucsf", caco_bad_ref),
    ]);
    let caco_good = inv_ucsf
        .experiments
        .iter()
        .find(|e| e.relative_path == "CACO_good.ucsf")
        .expect("CACO_good");
    assert_eq!(caco_good.pulse_class, PulseSequenceClass::Caco2D);
    assert!(
        caco_good.notes.iter().all(|n| !n.contains("Axis warning")),
        "valid swapped 2D UCSF CACO must not emit spurious RD-9 warnings: {:?}",
        caco_good.notes
    );
    let con_good = inv_ucsf
        .experiments
        .iter()
        .find(|e| e.relative_path == "CON_good.ucsf")
        .expect("CON_good");
    assert_eq!(con_good.pulse_class, PulseSequenceClass::Con2D);
    assert!(
        con_good.notes.iter().all(|n| !n.contains("Axis warning")),
        "valid swapped 2D UCSF CON must not emit spurious RD-9 warnings: {:?}",
        con_good.notes
    );
    let caco_bad = inv_ucsf
        .experiments
        .iter()
        .find(|e| e.relative_path == "CACO_bad.ucsf")
        .expect("CACO_bad");
    assert!(
        caco_bad.notes.iter().any(|n| n.contains("Axis warning")),
        "misreferenced 2D UCSF CACO must emit RD-9 warning: {:?}",
        caco_bad.notes
    );

    // 2. `091119/8000` shape: `inspect` reports valid `proc3s` axis (3..83 ppm, carrier ~43 ppm),
    //    not `acqu3s` (132..212 ppm), with zero RD-9 warnings.
    let inv_8000 = classify_nmr_entries(&[
        VirtualEntry::new(
            "8000/acqus",
            b"##$PULPROG= <c_hcccon_ia3d>\n##$FnTYPE= 2\n##$NUC1= <13C>\n##$SFO1= 201.257633\n##$BF1= 201.223023\n##$SW= 8.014\n##$TD= 1024\n".to_vec(),
        ),
        VirtualEntry::new(
            "8000/acqu2s",
            b"##$NUC1= <15N>\n##$SFO1= 81.098683\n##$BF1= 81.088628\n##$SW= 35.991\n##$TD= 1536\n##$NusTD= 128\n##$FnMODE= 5\n".to_vec(),
        ),
        VirtualEntry::new(
            "8000/acqu3s",
            b"##$NUC1= <13C>\n##$SFO1= 201.257633\n##$BF1= 201.223023\n##$SW= 79.884\n##$TD= 1\n##$NusTD= 48\n##$FnMODE= 5\n".to_vec(),
        ),
        VirtualEntry::new("8000/pdata/1/procs", b"##$SI= 1024\n##$XDIM= 4\n##$SW_p= 1612.90\n##$SF= 201.223023\n##$OFFSET= 176.008\n".to_vec()),
        VirtualEntry::new("8000/pdata/1/proc2s", b"##$SI= 128\n##$XDIM= 8\n##$SW_p= 2918.86\n##$SF= 81.088628\n##$OFFSET= 141.998\n".to_vec()),
        VirtualEntry::new("8000/pdata/1/proc3s", b"##$SI= 64\n##$XDIM= 8\n##$SW_p= 16077.17\n##$SF= 201.248968\n##$OFFSET= 83.0\n".to_vec()),
        VirtualEntry::header_only("8000/pdata/1/3rrr", Vec::new(), 4096),
    ]);
    let exp_8000 = &inv_8000.experiments[0];
    assert_eq!(exp_8000.pulse_class, PulseSequenceClass::Cccon3D);
    let f1_car = exp_8000.axes[2].carrier_ppm.expect("F1 carrier");
    assert!(
        (f1_car - 43.06).abs() < 0.5,
        "8000 F1 should display proc3s aliphatic carrier ~43 ppm, got {f1_car:.2}"
    );
    assert!(
        exp_8000.notes.iter().all(|n| !n.contains("Axis warning")),
        "8000 must have zero axis warnings: {:?}",
        exp_8000.notes
    );

    // 3. `IDPR1_MG_112019/10007` shape (`106..241 ppm` vs `-5..90 ppm`) triggers RD-9 warning.
    let inv_10007 = classify_nmr_entries(&[
        VirtualEntry::new(
            "10007/acqus",
            b"##$PULPROG= <c_hcccon_ia3d>\n##$FnTYPE= 2\n##$NUC1= <13C>\n##$SFO1= 201.257935\n##$BF1= 201.223023\n##$SW= 6.901\n##$TD= 1024\n".to_vec(),
        ),
        VirtualEntry::new(
            "10007/acqu2s",
            b"##$NUC1= <15N>\n##$SFO1= 81.098602\n##$BF1= 81.088628\n##$SW= 34.006\n##$TD= 2048\n##$NusTD= 128\n##$FnMODE= 5\n".to_vec(),
        ),
        VirtualEntry::new(
            "10007/acqu3s",
            b"##$NUC1= <13C>\n##$SFO1= 201.257935\n##$BF1= 201.223023\n##$SW= 135.020\n##$TD= 1\n##$NusTD= 64\n##$FnMODE= 5\n".to_vec(),
        ),
        VirtualEntry::new("10007/pdata/1/procs", b"##$SI= 2048\n##$XDIM= 4\n##$SW_p= 1388.89\n##$SF= 201.223023\n##$OFFSET= 176.951\n".to_vec()),
        VirtualEntry::new("10007/pdata/1/proc2s", b"##$SI= 256\n##$XDIM= 8\n##$SW_p= 2757.86\n##$SF= 81.088628\n##$OFFSET= 140.005\n".to_vec()),
        VirtualEntry::new("10007/pdata/1/proc3s", b"##$SI= 256\n##$XDIM= 8\n##$SW_p= 27173.91\n##$SF= 201.223023\n##$OFFSET= 241.022\n".to_vec()),
        VirtualEntry::header_only("10007/pdata/1/3rrr", Vec::new(), 4096),
    ]);
    assert!(
        inv_10007.experiments[0]
            .notes
            .iter()
            .any(|n| n.contains("Axis warning (F1)") && n.contains("SF == BF")),
        "10007 must emit RD-9 warning: {:?}",
        inv_10007.experiments[0].notes
    );

    // 4. `112019/100` shape (`c_hncacon_iare3d` with `13C,15N,1H` axes) explains why it is `Unknown`.
    let inv_100 = classify_nmr_entries(&[
        VirtualEntry::new(
            "100/acqus",
            b"##$PULPROG= <c_hncacon_iare3d>\n##$NUC1= <13C>\n##$SFO1= 201.258\n##$BF1= 201.223\n##$SW= 11.83\n##$TD= 1024\n".to_vec(),
        ),
        VirtualEntry::new(
            "100/acqu2s",
            b"##$NUC1= <15N>\n##$SFO1= 81.099\n##$BF1= 81.089\n##$SW= 35.99\n##$TD= 128\n##$FnMODE= 5\n".to_vec(),
        ),
        VirtualEntry::new(
            "100/acqu3s",
            b"##$NUC1= <1H>\n##$SFO1= 800.254\n##$BF1= 800.250\n##$SW= 10.24\n##$TD= 256\n##$FnMODE= 0\n".to_vec(),
        ),
    ]);
    assert_eq!(
        inv_100.experiments[0].pulse_class,
        PulseSequenceClass::Unknown
    );
    assert!(
        inv_100.experiments[0]
            .notes
            .iter()
            .any(|n| n.contains("conflicts with axis nuclei")),
        "expected PULPROG/axis conflict note: {:?}",
        inv_100.experiments[0].notes
    );
}
