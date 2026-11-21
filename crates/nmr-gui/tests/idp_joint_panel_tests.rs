//! Headless `egui` layout, typography (`>= 11.0 px`), zero-clipping / zero-overlap,
//! canonical `nmr_core::joint_idp_solver` integration, runnable `cli_preview_command`,
//! live CAS axis validation (`P2-7`), `MAX_CAS_SLOTS` cap (`P2-8`), consistent Proline
//! `131.5-144.5 ppm` window (`P2-9`), ingest Joint `HN + CON` + CAS extraction, ribbon &
//! strip-tiler `[Pro]`/`XVal`/`HN`/`CON` badge rendering, and zero-emoji source hygiene tests
//! for `nmr_gui::idp_joint_panel` (`T23.3`).

use std::collections::{BTreeMap, BTreeSet};

use eframe::egui;
use nmr_core::c13_detected::{parse_cas_cli_spec, CasMode, CasPeak};
use nmr_core::classify::{
    ClassifiedAxis, ClassifiedExperiment, DataState, ExperimentDimensionality, ExperimentVendor,
    NusSummary, ProblemRegime, PulseSequenceClass, QuadratureKind, WorkspaceInventory,
};
use nmr_core::joint_idp_solver::{
    trosy_expected_shift_ppm, JointRouteSupport, CANONICAL_TROSY_FALLBACK_MHZ,
    CTERM_COO_DSS_CENTER_PPM, PROLINE_N15_MAX_PPM, PROLINE_N15_MIN_PPM, TROSY_HALF_J_HN_HZ,
};
use nmr_core::priors::Priors;
use nmr_core::solver::{pair_matrix, ScoreConfig};
use nmr_core::spinsystems::{HncacbPolarity, SpinSystem};
use nmr_core::topology::SidechainPriors;
use nmr_gui::assign_setup_drawer::{
    AssignProteinRegime, AssignSetupAction, AssignSetupDrawerState, AssignSetupInput,
};
use nmr_gui::idp_joint_panel::{
    draw_route_support_pill, joint_route_color, joint_route_short_code, CasAxisConsistency,
    CasDirection, CasResidueType, CasSlotEntry, IdpJointAction, IdpJointPanelState,
    IdpJointPanelTheme, IdpJointSubTab, IdpRegimeSelection, ALL_JOINT_ROUTE_SUPPORTS,
    MAX_CAS_SLOTS,
};
use nmr_gui::ingest_panel::IngestPanelState;
use nmr_gui::sidechain_widget::{
    SequenceRibbonState, SidechainInspectorPanel, SidechainWorkspaceInput,
};
use nmr_gui::strip_tiler::{StripSpectra3D, StripTilerContext, StripTilerPanel};
use nmr_gui::workbench_mode::{apply_type_scale, audit_text_layout, AuditedText, MIN_FONT_PX};

fn click_events(pos: egui::Pos2) -> Vec<egui::Event> {
    vec![
        egui::Event::PointerMoved(pos),
        egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: egui::Modifiers::NONE,
        },
        egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        },
    ]
}

fn run_panel_frame(
    ctx: &egui::Context,
    state: &mut IdpJointPanelState,
    width: f32,
    height: f32,
    paper_white: bool,
    events: Vec<egui::Event>,
) -> (egui::FullOutput, Vec<IdpJointAction>) {
    let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(width, height));
    let raw = egui::RawInput {
        screen_rect: Some(rect),
        events,
        ..Default::default()
    };
    let mut actions = Vec::new();
    let mut out = ctx.run_ui(raw, |ui| {
        apply_type_scale(ui.ctx());
        egui::CentralPanel::default().show(ui, |ui| {
            actions = state.show(ui, paper_white);
        });
    });
    out.textures_delta.clear();
    (out, actions)
}

fn find_text_center(texts: &[AuditedText], needle: &str) -> egui::Pos2 {
    texts
        .iter()
        .find(|t| t.text.trim() == needle)
        .or_else(|| texts.iter().find(|t| t.text.contains(needle)))
        .map(|t| t.rect.center())
        .unwrap_or_else(|| {
            let available: Vec<&str> = texts.iter().map(|t| t.text.as_str()).collect();
            panic!("could not find rendered text {needle:?} in {available:?}");
        })
}

#[test]
fn test_canonical_core_types_trosy_parity_and_proline_window_p1_6_p2_9() {
    // Regime CLI mapping: AutoJoint omits `--regime`, explicit regimes match real CLI values
    assert_eq!(IdpRegimeSelection::ALL.len(), 4);
    assert_eq!(IdpRegimeSelection::AutoJoint.cli_regime_arg(), None);
    assert_eq!(
        IdpRegimeSelection::IdpJoint.cli_regime_arg(),
        Some("idp-joint")
    );
    assert_eq!(
        IdpRegimeSelection::Idp13cOnly.cli_regime_arg(),
        Some("idp-13c")
    );
    assert_eq!(
        IdpRegimeSelection::Globular.cli_regime_arg(),
        Some("globular")
    );

    // Exact canonical Proline 15N window from nmr_core::joint_idp_solver (`P2-9`)
    assert!((PROLINE_N15_MIN_PPM - 131.5).abs() < 1e-12);
    assert!((PROLINE_N15_MAX_PPM - 144.5).abs() < 1e-12);
    assert!((TROSY_HALF_J_HN_HZ - 46.0).abs() < 1e-12);
    assert!((CANONICAL_TROSY_FALLBACK_MHZ - 800.0).abs() < 1e-12);

    // Zero duplicate TROSY math: GUI delegates bit-identically to core `trosy_expected_shift_ppm`
    let mut state = IdpJointPanelState::new();
    for mhz in [600.0, 800.0, 950.0, 40.0, 0.0, f64::NAN] {
        state.spectrometer_mhz = mhz;
        let gui_shift = state.trosy_half_j_offsets_ppm();
        let core_shift = trosy_expected_shift_ppm(mhz);
        assert_eq!(gui_shift, core_shift, "TROSY shift mismatch at {mhz:?} MHz");
    }

    // CasDirection <-> nmr_core::c13_detected::CasMode roundtrip
    assert_eq!(CasDirection::Cacon.to_core_mode(), CasMode::Cacon);
    assert_eq!(CasDirection::Canco.to_core_mode(), CasMode::Canco);
    assert_eq!(
        CasDirection::from_core_mode(CasMode::Cacon),
        CasDirection::Cacon
    );
    assert_eq!(
        CasDirection::from_core_mode(CasMode::Canco),
        CasDirection::Canco
    );

    // Canonical JointRouteSupport distinct theme colors & crisp short codes
    assert_eq!(
        joint_route_short_code(JointRouteSupport::CrossValidatedHnAndCon),
        "XVal"
    );
    assert_eq!(joint_route_short_code(JointRouteSupport::HnRoute), "HN");
    assert_eq!(
        joint_route_short_code(JointRouteSupport::ConProBridge),
        "[Pro]"
    );
    assert_eq!(
        joint_route_short_code(JointRouteSupport::ConRouteOnly),
        "CON"
    );

    for paper_white in [false, true] {
        let colors: Vec<egui::Color32> = ALL_JOINT_ROUTE_SUPPORTS
            .iter()
            .map(|&r| joint_route_color(r, paper_white))
            .collect();
        for i in 0..colors.len() {
            for j in (i + 1)..colors.len() {
                assert_ne!(colors[i], colors[j], "route colors must be distinct");
            }
        }
    }
    let dark_theme = IdpJointPanelTheme::for_mode(false);
    let light_theme = IdpJointPanelTheme::for_mode(true);
    assert_ne!(dark_theme.card_bg, light_theme.card_bg);
    assert_ne!(dark_theme.text_primary, light_theme.text_primary);
}

#[test]
fn test_live_axis_validation_gating_p2_7_and_max_cas_slots_cap_p2_8() {
    let mut state = IdpJointPanelState::new();

    // 1. Live axis validation and honest `is_ready_for_solve` gating (`P2-7`)
    let id1 = state
        .add_cas_slot(
            CasResidueType::Ser,
            "",
            CasDirection::Cacon,
            "cacon_ser.tsv",
        )
        .expect("slot 1");
    // Loaded=true with Unchecked axes must NOT be ready!
    assert!(state.set_cas_slot_loaded(id1, true, 12, CasAxisConsistency::Unchecked));
    assert!(
        !state.cas_slots[0].is_ready_for_solve(),
        "Unchecked CAS slot must not be ready for solve"
    );
    assert_eq!(state.ready_cas_slot_count(), 0);

    // Loading real peaks runs live `validate_peaks` -> Consistent
    let valid_peaks = vec![
        CasPeak {
            c_prime_ppm: 174.2,
            n_ppm: 118.5,
            height: 1.0,
            volume: 1.0,
        },
        CasPeak {
            c_prime_ppm: 173.4,
            n_ppm: 136.8,
            height: 0.9,
            volume: 0.9,
        },
    ];
    let ax_ok = state.load_cas_slot_from_peaks(id1, valid_peaks, None);
    assert_eq!(ax_ok, Some(CasAxisConsistency::Consistent));
    assert!(state.cas_slots[0].is_ready_for_solve());
    assert_eq!(state.ready_cas_slot_count(), 1);
    assert!(state.cas_slots[0].to_cas_spectrum().is_some());

    // Transposed peaks automatically trigger SwappedAxes and block readiness
    let swapped_peaks = vec![CasPeak {
        c_prime_ppm: 120.4,
        n_ppm: 174.8,
        height: 1.0,
        volume: 1.0,
    }];
    let ax_swap = state.load_cas_slot_from_peaks(id1, swapped_peaks, None);
    assert_eq!(ax_swap, Some(CasAxisConsistency::SwappedAxes));
    assert!(!state.cas_slots[0].is_ready_for_solve());

    // Axis-header validation via `set_cas_slot_axis_window`
    assert_eq!(
        state.set_cas_slot_axis_window(id1, "1H", 10.0, 6.0, "15N", 132.0, 105.0),
        Some(CasAxisConsistency::WrongNuclei)
    );
    assert!(!state.cas_slots[0].is_ready_for_solve());
    assert_eq!(
        state.set_cas_slot_axis_window(id1, "13C", 182.0, 168.0, "15N", 142.0, 102.0),
        Some(CasAxisConsistency::Consistent)
    );
    assert!(state.cas_slots[0].is_ready_for_solve());

    // Changing path resets loaded/axes back to Unchecked
    assert!(state.set_cas_slot_path(id1, "other_ser.tsv"));
    assert_eq!(
        state.cas_slots[0].axis_consistency,
        CasAxisConsistency::Unchecked
    );
    assert!(!state.cas_slots[0].loaded);
    assert!(!state.cas_slots[0].is_ready_for_solve());

    // 2. Hard MAX_CAS_SLOTS (16) cap (`P2-8`)
    for idx in 1..MAX_CAS_SLOTS {
        assert!(
            state
                .add_cas_slot(
                    CasResidueType::Ala,
                    "",
                    CasDirection::Cacon,
                    format!("slot_{idx}.tsv")
                )
                .is_ok(),
            "expected slot {idx} below cap {MAX_CAS_SLOTS} to succeed"
        );
    }
    assert_eq!(state.cas_slots.len(), MAX_CAS_SLOTS);
    let err = state
        .add_cas_slot(CasResidueType::Gly, "", CasDirection::Canco, "overflow.tsv")
        .expect_err("adding beyond MAX_CAS_SLOTS must fail");
    assert!(err.contains(&MAX_CAS_SLOTS.to_string()));
    assert_eq!(state.cas_slots.len(), MAX_CAS_SLOTS);

    // Removing one slot frees capacity again
    assert!(state.remove_cas_slot(id1));
    assert!(state
        .add_cas_slot(CasResidueType::Gly, "", CasDirection::Canco, "refill.tsv")
        .is_ok());
    assert_eq!(state.cas_slots.len(), MAX_CAS_SLOTS);
}

#[test]
fn test_runnable_cli_preview_command_strict_flag_and_cas_validation_p1_7() {
    let cli_main_src = include_str!("../../nmr-cli/src/main.rs");
    let valid_assign_flags: BTreeSet<&str> = [
        "--regime",
        "--seq",
        "--out",
        "--engine",
        "--peaks",
        "--con",
        "--cccon",
        "--hncacon",
        "--hncaconh",
        "--roots",
        "--hncacb-polarity",
        "--temp-k",
        "--ph",
        "--cas",
    ]
    .into_iter()
    .collect();

    // Cross-check every allowlisted flag literally exists in `crates/nmr-cli/src/main.rs`
    for flag in &valid_assign_flags {
        assert!(
            cli_main_src.contains(&format!("\"{flag}\"")),
            "expected real CLI flag {flag} in crates/nmr-cli/src/main.rs"
        );
    }

    let mut state = IdpJointPanelState::new();
    state.cli_paths.seq_path = "idp.fasta".to_string();
    state.cli_paths.out_prefix = "idp_out".to_string();
    state.cli_paths.peaks_path = "hn_peaks.tsv".to_string();
    state.cli_paths.con_path = "con_2d.ucsf".to_string();
    state.cli_paths.cccon_path = "hcccon_3d.ucsf".to_string();
    state.cli_paths.hncacon_path = "hncacon_3d.ucsf".to_string();
    state.cli_paths.hncaconh_path = "bt_hncaconh.ucsf".to_string();
    state.hncacb_polarity = HncacbPolarity::CbNegative;
    state.filter_nh2_pairs = false; // must NOT invent `--no-nh2-filter`
    state.trosy_half_j_calibration = true; // must NOT invent `--trosy-calibrate`

    // Add 4 CAS slots: 1 ready valid, 1 unloaded, 1 empty `Other`, 1 swapped-axis
    let id_ok = state
        .add_cas_slot(
            CasResidueType::Ser,
            "",
            CasDirection::Cacon,
            "cacon_ser.tsv",
        )
        .expect("ok slot");
    let _ = state.load_cas_slot_from_peaks(
        id_ok,
        vec![CasPeak {
            c_prime_ppm: 174.1,
            n_ppm: 119.2,
            height: 1.0,
            volume: 1.0,
        }],
        None,
    );
    let _id_unloaded = state
        .add_cas_slot(CasResidueType::Ala, "", CasDirection::Cacon, "")
        .expect("unloaded slot");
    let id_bad_other = state
        .add_cas_slot(CasResidueType::Other, "", CasDirection::Canco, "bad.tsv")
        .expect("empty other slot");
    let _ = state.set_cas_slot_loaded(id_bad_other, true, 5, CasAxisConsistency::Consistent);
    let id_swapped = state
        .add_cas_slot(CasResidueType::Gly, "", CasDirection::Cacon, "swapped.tsv")
        .expect("swapped slot");
    let _ = state.set_cas_slot_loaded(id_swapped, true, 5, CasAxisConsistency::SwappedAxes);

    for regime in IdpRegimeSelection::ALL {
        state.regime = regime;
        let cmd = state.cli_preview_command();
        assert!(cmd.starts_with("ferric assign "));
        assert!(cmd.contains("--seq idp.fasta"));
        assert!(cmd.contains("--out idp_out"));
        assert!(cmd.contains("--engine cpu"));

        // Strictly forbid non-existent flags and placeholder/invalid tokens
        for forbidden in [
            "--trosy-calibrate",
            "--no-nh2-filter",
            "--regime auto",
            "<unassigned_path>",
            "custom",
            "bad.tsv",
            "swapped.tsv",
        ] {
            assert!(
                !cmd.contains(forbidden),
                "forbidden token {forbidden:?} emitted in CLI preview for {regime:?}: {cmd}"
            );
        }

        // Verify every emitted `--flag` is in `valid_assign_flags` and every `--cas` parses cleanly
        let tokens: Vec<&str> = cmd.split_whitespace().collect();
        let mut idx = 2usize;
        while idx < tokens.len() {
            let tok = tokens[idx];
            assert!(
                tok.starts_with("--") && valid_assign_flags.contains(tok),
                "unexpected CLI flag {tok:?} in: {cmd}"
            );
            let val = tokens.get(idx + 1).expect("flag value");
            if tok == "--regime" {
                assert!(matches!(*val, "globular" | "idp-13c" | "idp-joint"));
            } else if tok == "--cas" {
                let parsed =
                    parse_cas_cli_spec(val).expect("parse_cas_cli_spec must accept preview");
                assert!(parsed.0.is_some(), "CAS preview must specify explicit mode");
                assert!(!parsed.2.is_empty());
            }
            idx += 2;
        }
    }
}

#[test]
fn test_real_solver_synthetic_demo_ribbon_and_strip_tiler_route_indicators_p1_5_p1_6() {
    let mut state = IdpJointPanelState::new();
    let (seq, seq_ids, _ds, res) = state
        .load_synthetic_idp_demo()
        .expect("real solver synthetic demo");

    // Real assign_joint_idp_cpu ran and populated bridges, C-term anchor, QC, and concordance
    assert_eq!(seq.len(), 22);
    assert_eq!(state.cas_slots.len(), 3);
    assert_eq!(state.ready_cas_slot_count(), 3);
    assert!(state.last_qc_verdict.is_some());
    assert!(
        !state.proline_bridges.is_empty(),
        "expected Proline bridges from real solver"
    );
    assert!(
        state.proline_bridges.iter().any(|b| b.bridged),
        "expected at least one bridged Proline in synthetic Pro/Ser IDP"
    );
    assert!(
        !state.c_term_hints.is_empty(),
        "expected C-terminal COO- anchor summary from real solver"
    );
    assert!((state.c_term_hints[0].expected_coo_ppm - CTERM_COO_DSS_CENTER_PPM).abs() < 1e-9);

    let route_map = state.route_map_by_res_idx();
    assert!(!route_map.is_empty());
    assert!(
        route_map
            .values()
            .any(|&r| r == JointRouteSupport::CrossValidatedHnAndCon),
        "expected XVal residues"
    );
    assert!(
        route_map
            .values()
            .any(|&r| r == JointRouteSupport::ConProBridge),
        "expected [Pro] bridge residues"
    );

    // Verify SidechainInspectorPanel::show_ribbon renders [Pro] and XVal badges cleanly
    let ctx = egui::Context::default();
    let mut ribbon_state = SequenceRibbonState::new();
    ribbon_state.joint_route_by_res = route_map.clone();
    let empty_atoms = BTreeMap::new();
    let empty_locks = BTreeSet::new();
    let mut backbone_confs = BTreeMap::new();
    for ra in &res.residue_assignments {
        if ra.node_idx.is_some() {
            backbone_confs.insert(ra.seq_idx, ra.confidence);
        }
    }
    let priors = Priors::default();
    let sc_priors = SidechainPriors::default();
    let sc_input = SidechainWorkspaceInput {
        sequence: &seq,
        seq_ids: &seq_ids,
        assigned_atoms: empty_atoms,
        locked_residues: empty_locks,
        backbone_confidences: backbone_confs,
        priors: &priors,
        sc_priors: &sc_priors,
    };

    let viewport = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1280.0, 420.0));
    let mut out = ctx.run_ui(
        egui::RawInput {
            screen_rect: Some(viewport),
            ..Default::default()
        },
        |ui| {
            apply_type_scale(ui.ctx());
            egui::CentralPanel::default().show(ui, |ui| {
                let _ = SidechainInspectorPanel::show_ribbon(ui, &mut ribbon_state, &sc_input);
            });
        },
    );
    out.textures_delta.clear();
    let audit = audit_text_layout(&out.shapes, viewport);
    assert!(audit.min_font_px >= MIN_FONT_PX - 1e-3);
    assert!(
        audit.texts.iter().any(|t| t.text.contains("XVal")),
        "sequence ribbon should display XVal route badge"
    );
    assert!(
        audit.texts.iter().any(|t| t.text.contains("[Pro]")),
        "sequence ribbon should display [Pro] Proline-bridge badge"
    );

    // Verify StripTilerPanel renders route support indicators in column headers
    let mut systems = Vec::new();
    for k in 1..4 {
        let mut ss = SpinSystem::new(8.2 + 0.1 * (k as f64), 118.0 + 3.0 * (k as f64));
        ss.cands
            .entry(("CA".to_string(), 0))
            .or_default()
            .push((55.0 + k as f64, 2));
        ss.cands
            .entry(("CA".to_string(), -1))
            .or_default()
            .push((54.0 + k as f64, 2));
        systems.push(ss);
    }
    let p_mat = pair_matrix(&systems, &ScoreConfig::default());
    let assignments = vec![2_i32, 3_i32, 4_i32]; // SER3 (XVal), PRO4 ([Pro]), GLY5 (XVal)
    let mut tiler = StripTilerPanel::new();
    let mut tiler_routes = BTreeMap::new();
    tiler_routes.insert(2usize, JointRouteSupport::CrossValidatedHnAndCon);
    tiler_routes.insert(3usize, JointRouteSupport::ConProBridge);
    tiler_routes.insert(4usize, JointRouteSupport::ConRouteOnly);
    tiler.set_joint_route_support(tiler_routes);

    let tctx = StripTilerContext {
        systems: &systems,
        pair_mat: &p_mat,
        u_mat: None,
        sequence: &seq,
        seq_ids: &seq_ids,
        active_root: 1,
        assignments: Some(&assignments),
        locks: None,
        confirmed_links: None,
        spectra_3d: StripSpectra3D::default(),
    };
    let mut out_tiler = ctx.run_ui(
        egui::RawInput {
            screen_rect: Some(viewport),
            ..Default::default()
        },
        |ui| {
            apply_type_scale(ui.ctx());
            egui::CentralPanel::default().show(ui, |ui| {
                let _ = tiler.show(ui, &tctx, None);
            });
        },
    );
    out_tiler.textures_delta.clear();
    let audit_tiler = audit_text_layout(&out_tiler.shapes, viewport);
    assert!(audit_tiler.min_font_px >= MIN_FONT_PX - 1e-3);
    assert!(
        audit_tiler.texts.iter().any(|t| t.text.contains("[Pro]")),
        "strip tiler header should display [Pro] badge for assigned Proline bridge: {:?}",
        audit_tiler
            .texts
            .iter()
            .map(|t| &t.text)
            .collect::<Vec<_>>()
    );
}

#[test]
fn test_ingest_panel_joint_hn_con_detection_and_cas_slot_extraction_p1_5() {
    let make_axis = |label: &str, nuc: &str, car: f64, sw: f64| ClassifiedAxis {
        label: label.to_string(),
        nucleus: nuc.to_string(),
        points: 512,
        sw_ppm: Some(sw),
        carrier_ppm: Some(car),
        obs_mhz: Some(800.0),
        quadrature: QuadratureKind::Processed,
    };
    let make_exp = |rel: &str,
                    cls: PulseSequenceClass,
                    pp: &str,
                    axes: Vec<ClassifiedAxis>|
     -> ClassifiedExperiment {
        ClassifiedExperiment {
            relative_path: rel.to_string(),
            display_name: rel.to_string(),
            vendor: ExperimentVendor::Bruker,
            dimensionality: ExperimentDimensionality::Dim2D,
            data_state: DataState::Processed,
            spectrometer_mhz: Some(800.13),
            axes,
            quadrature_badge: "Processed".to_string(),
            nus: NusSummary::default(),
            pulse_class: cls,
            pulse_program: Some(pp.to_string()),
            title: Some(pp.to_string()),
            raw_signal_bytes: 0,
            processed_bytes: 4096,
            selected_by_default: true,
            notes: Vec::new(),
        }
    };

    let inv = WorkspaceInventory {
        root_name: "idp_joint_study".to_string(),
        experiments: vec![
            make_exp(
                "10_hsqc",
                PulseSequenceClass::N15Hsqc,
                "hsqcetfpf3gpsi",
                vec![
                    make_axis("F2", "1H", 4.7, 14.0),
                    make_axis("F1", "15N", 118.0, 32.0),
                ],
            ),
            make_exp(
                "20_con",
                PulseSequenceClass::Con2D,
                "c_con_iasq",
                vec![
                    make_axis("F2", "13C", 174.0, 20.0),
                    make_axis("F1", "15N", 120.0, 40.0),
                ],
            ),
            make_exp(
                "21_cacon_ser",
                PulseSequenceClass::CasCacon2D,
                "c_cacon_ser",
                vec![
                    make_axis("F2", "13C", 174.0, 20.0),
                    make_axis("F1", "15N", 120.0, 40.0),
                ],
            ),
            make_exp(
                "22_canco_tavi_swapped",
                PulseSequenceClass::CasCanco2D,
                "c_canco_tavi",
                vec![
                    make_axis("F2", "15N", 120.0, 40.0),
                    make_axis("F1", "13C", 174.0, 20.0),
                ],
            ),
        ],
        auxiliary_files: Vec::new(),
        problem_regime: ProblemRegime::Idp13cDetected,
        secondary_regimes: vec![ProblemRegime::GlobularBackbone],
        missing_input_hints: Vec::new(),
        elapsed_ms: 1.0,
    };

    let ingest = IngestPanelState::from_inventory(inv);
    assert!(ingest.is_joint_hn_con_study());
    let plan = ingest.build_execution_plan();
    assert!(plan.has_joint_hn_con_suite());

    let slots = plan.extract_cas_slot_entries();
    assert_eq!(slots.len(), 2);
    assert_eq!(slots[0].residue_type, CasResidueType::Ser);
    assert_eq!(slots[0].direction, CasDirection::Cacon);
    assert_eq!(slots[0].axis_consistency, CasAxisConsistency::Consistent);

    assert_eq!(slots[1].residue_type, CasResidueType::Tavi);
    assert_eq!(slots[1].direction, CasDirection::Canco);
    assert_eq!(slots[1].axis_consistency, CasAxisConsistency::SwappedAxes);

    // Syncing into IdpJointPanelState preserves slots and respects cap
    let mut panel = IdpJointPanelState::new();
    let synced = panel.sync_cas_slots_from_ingest(slots);
    assert_eq!(synced, 2);
    assert_eq!(panel.cas_slots.len(), 2);

    // Verify AssignSetupDrawer exposes AssignProteinRegime::IdpJoint and emits OpenIdpJointPanel
    let ctx = egui::Context::default();
    let mut drawer = AssignSetupDrawerState::new();
    drawer.regime = AssignProteinRegime::IdpJoint;
    let input = AssignSetupInput::default();
    let viewport = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1280.0, 800.0));
    let mut out = ctx.run_ui(
        egui::RawInput {
            screen_rect: Some(viewport),
            ..Default::default()
        },
        |ui| {
            apply_type_scale(ui.ctx());
            egui::CentralPanel::default().show(ui, |ui| {
                let _ = drawer.show_contents(ui, &input);
            });
        },
    );
    out.textures_delta.clear();
    let audit = audit_text_layout(&out.shapes, viewport);
    let pos_open = find_text_center(&audit.texts, "CAS & Joint IDP...");
    let mut drawer_actions = Vec::new();
    let mut out2 = ctx.run_ui(
        egui::RawInput {
            screen_rect: Some(viewport),
            events: click_events(pos_open),
            ..Default::default()
        },
        |ui| {
            apply_type_scale(ui.ctx());
            egui::CentralPanel::default().show(ui, |ui| {
                drawer_actions = drawer.show_contents(ui, &input);
            });
        },
    );
    out2.textures_delta.clear();
    assert!(drawer_actions.contains(&AssignSetupAction::OpenIdpJointPanel));
}

#[test]
fn test_headless_layout_1280x800_and_1440x900_all_tabs_zero_overlap_clipping_and_font_floor() {
    let ctx = egui::Context::default();
    let mut state = IdpJointPanelState::new();
    let _ = state.load_synthetic_idp_demo();

    // Add an `Other` slot with swapped axes to stress-test extra controls & badges
    let extra_id = state
        .add_cas_slot(
            CasResidueType::Other,
            "ASN+ASP",
            CasDirection::Canco,
            "demo/canco_asx_swapped.tsv",
        )
        .expect("extra slot");
    state.set_cas_slot_loaded(extra_id, true, 7, CasAxisConsistency::SwappedAxes);

    for (width, height) in [(1280.0_f32, 800.0_f32), (1440.0_f32, 900.0_f32)] {
        let viewport = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(width, height));
        for paper_white in [false, true] {
            for tab in IdpJointSubTab::ALL {
                state.active_tab = tab;
                let _ = run_panel_frame(&ctx, &mut state, width, height, paper_white, Vec::new());
                let (out, _) =
                    run_panel_frame(&ctx, &mut state, width, height, paper_white, Vec::new());
                let audit = audit_text_layout(&out.shapes, viewport);

                assert!(
                    !audit.texts.is_empty(),
                    "expected visible text on tab {tab:?} at {width}x{height}"
                );
                assert!(
                    audit.min_font_px >= MIN_FONT_PX - 1e-3,
                    "tab {tab:?} at {width}x{height} (paper_white={paper_white}) rendered font {:.2} < {MIN_FONT_PX}",
                    audit.min_font_px
                );
                assert!(
                    audit.clipped.is_empty(),
                    "tab {tab:?} at {width}x{height} (paper_white={paper_white}) had clipped texts: {:?}",
                    audit
                        .clipped
                        .iter()
                        .map(|t| (&t.text, t.rect, t.clip_rect))
                        .collect::<Vec<_>>()
                );
                assert!(
                    audit.overlapping.is_empty(),
                    "tab {tab:?} at {width}x{height} (paper_white={paper_white}) had overlapping texts: {:?}",
                    audit
                        .overlapping
                        .iter()
                        .map(|(a, b)| (&a.text, &b.text))
                        .collect::<Vec<_>>()
                );
            }
        }
    }
}

#[test]
fn test_headless_window_and_reusable_route_support_pill_layout() {
    let ctx = egui::Context::default();
    let mut state = IdpJointPanelState::new();
    let _ = state.load_synthetic_idp_demo();
    let mut open = true;
    let viewport = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1280.0, 800.0));

    for pass in 0..2 {
        let raw = egui::RawInput {
            screen_rect: Some(viewport),
            ..Default::default()
        };
        let mut out = ctx.run_ui(raw, |ui| {
            apply_type_scale(ui.ctx());
            let _ = state.show_window(ui.ctx(), &mut open, true);
            egui::Panel::bottom("route_support_strip_test").show(ui, |ui| {
                ui.horizontal(|ui| {
                    for route in ALL_JOINT_ROUTE_SUPPORTS {
                        draw_route_support_pill(ui, route, true);
                    }
                });
            });
        });
        out.textures_delta.clear();
        if pass == 1 {
            let audit = audit_text_layout(&out.shapes, viewport);
            assert!(audit.min_font_px >= MIN_FONT_PX - 1e-3);
            assert!(audit.clipped.is_empty(), "clipped: {:?}", audit.clipped);
            assert!(
                audit.overlapping.is_empty(),
                "overlapping: {:?}",
                audit.overlapping
            );
        }
    }
    assert!(open);
}

#[test]
fn test_interactive_regime_toggles_and_cas_slot_mutations_via_egui_clicks() {
    let ctx = egui::Context::default();
    let mut state = IdpJointPanelState::new();
    let w = 1280.0_f32;
    let h = 800.0_f32;
    let viewport = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(w, h));

    let _ = run_panel_frame(&ctx, &mut state, w, h, false, Vec::new());
    let (out, _) = run_panel_frame(&ctx, &mut state, w, h, false, Vec::new());
    let audit = audit_text_layout(&out.shapes, viewport);

    // 1. Click "IDP Joint (HN + 13C)" regime button
    let pos_joint = find_text_center(&audit.texts, "IDP Joint (HN + 13C)");
    let (_, actions) = run_panel_frame(&ctx, &mut state, w, h, false, click_events(pos_joint));
    assert_eq!(state.regime, IdpRegimeSelection::IdpJoint);
    assert!(actions.contains(&IdpJointAction::RegimeChanged(IdpRegimeSelection::IdpJoint)));

    // 2. Click TROSY calibration checkbox to toggle it off
    let (out, _) = run_panel_frame(&ctx, &mut state, w, h, false, Vec::new());
    let audit = audit_text_layout(&out.shapes, viewport);
    let pos_trosy = find_text_center(&audit.texts, "TROSY 1/2 1J_NH auto-calibration");
    let (_, actions) = run_panel_frame(&ctx, &mut state, w, h, false, click_events(pos_trosy));
    assert!(!state.trosy_half_j_calibration);
    assert!(actions.contains(&IdpJointAction::ToggleTrosyCalibration(false)));

    // 3. Click "+ Add CAS Slot" to create a new slot
    let (out, _) = run_panel_frame(&ctx, &mut state, w, h, false, Vec::new());
    let audit = audit_text_layout(&out.shapes, viewport);
    let pos_add = find_text_center(&audit.texts, "+ Add CAS Slot");
    let (_, actions) = run_panel_frame(&ctx, &mut state, w, h, false, click_events(pos_add));
    assert_eq!(state.cas_slots.len(), 1);
    let created_id = state.cas_slots[0].id;
    assert!(actions.iter().any(|a| matches!(
        a,
        IdpJointAction::AddCasSlot(CasSlotEntry { id, .. }) if *id == created_id
    )));

    // 4. Click "Load..." on the newly created slot
    let _ = run_panel_frame(&ctx, &mut state, w, h, false, Vec::new());
    let (out, _) = run_panel_frame(&ctx, &mut state, w, h, false, Vec::new());
    let audit = audit_text_layout(&out.shapes, viewport);
    let pos_load = find_text_center(&audit.texts, "Load...");
    let (_, actions) = run_panel_frame(&ctx, &mut state, w, h, false, click_events(pos_load));
    assert!(actions.contains(&IdpJointAction::RequestLoadCasSlot(created_id)));

    // 5. Click "Remove" on the slot
    let (out, _) = run_panel_frame(&ctx, &mut state, w, h, false, Vec::new());
    let audit = audit_text_layout(&out.shapes, viewport);
    let pos_remove = find_text_center(&audit.texts, "Remove");
    let (_, actions) = run_panel_frame(&ctx, &mut state, w, h, false, click_events(pos_remove));
    assert!(state.cas_slots.is_empty());
    assert!(actions.contains(&IdpJointAction::RemoveCasSlot(created_id)));

    // 6. Click "Run IDP Assignment"
    let (out, _) = run_panel_frame(&ctx, &mut state, w, h, false, Vec::new());
    let audit = audit_text_layout(&out.shapes, viewport);
    let pos_run = find_text_center(&audit.texts, "Run IDP Assignment");
    let (_, actions) = run_panel_frame(&ctx, &mut state, w, h, false, click_events(pos_run));
    assert!(actions.contains(&IdpJointAction::RunJointAssignment));
}

#[test]
fn test_interactive_hnco_nh2_toggles_and_cas_slot_retyping_via_combobox() {
    let ctx = egui::Context::default();
    let mut state = IdpJointPanelState::new();
    let slot_id = state
        .add_cas_slot(
            CasResidueType::Gly,
            "",
            CasDirection::Cacon,
            "cacon_gly.tsv",
        )
        .expect("gly slot");
    let w = 1280.0_f32;
    let h = 800.0_f32;
    let viewport = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(w, h));

    let _ = run_panel_frame(&ctx, &mut state, w, h, false, Vec::new());
    let (out, _) = run_panel_frame(&ctx, &mut state, w, h, false, Vec::new());
    let audit = audit_text_layout(&out.shapes, viewport);

    // 1. Toggle HNCO-supported roots off
    let pos_hnco = find_text_center(&audit.texts, "HNCO-supported roots (--roots hnco)");
    let (_, actions) = run_panel_frame(&ctx, &mut state, w, h, false, click_events(pos_hnco));
    assert!(!state.roots_hnco);
    assert!(actions.contains(&IdpJointAction::ToggleRootsHnco(false)));

    // 2. Toggle NH2 doublet filtering off
    let (out, _) = run_panel_frame(&ctx, &mut state, w, h, false, Vec::new());
    let audit = audit_text_layout(&out.shapes, viewport);
    let pos_nh2 = find_text_center(&audit.texts, "Filter Asn/Gln NH2 doublet pairs");
    let (_, actions) = run_panel_frame(&ctx, &mut state, w, h, false, click_events(pos_nh2));
    assert!(!state.filter_nh2_pairs);
    assert!(actions.contains(&IdpJointAction::ToggleNh2Filtering(false)));

    // 3. Re-type the CAS slot from Gly -> Ala via ComboBox popup click
    let (out, _) = run_panel_frame(&ctx, &mut state, w, h, false, Vec::new());
    let audit = audit_text_layout(&out.shapes, viewport);
    let pos_gly_combo = find_text_center(&audit.texts, "Gly");
    let _ = run_panel_frame(&ctx, &mut state, w, h, false, click_events(pos_gly_combo));
    let (out, _) = run_panel_frame(&ctx, &mut state, w, h, false, Vec::new());
    let audit = audit_text_layout(&out.shapes, viewport);
    let pos_ala_opt = find_text_center(&audit.texts, "Ala");
    let (_, actions) = run_panel_frame(&ctx, &mut state, w, h, false, click_events(pos_ala_opt));
    assert_eq!(state.cas_slots[0].residue_type, CasResidueType::Ala);
    assert!(actions.iter().any(|a| matches!(
        a,
        IdpJointAction::UpdateCasSlot(s)
            if s.id == slot_id && s.residue_type == CasResidueType::Ala
    )));

    // 4. Switch direction of the CAS slot from CACON (i-1) -> CANCO (i) via ComboBox
    let _ = run_panel_frame(&ctx, &mut state, w, h, false, Vec::new());
    let (out, _) = run_panel_frame(&ctx, &mut state, w, h, false, Vec::new());
    let audit = audit_text_layout(&out.shapes, viewport);
    let cacon_boxes: Vec<egui::Pos2> = audit
        .texts
        .iter()
        .filter(|t| t.text.trim() == "CACON (i-1)")
        .map(|t| t.rect.center())
        .collect();
    assert!(cacon_boxes.len() >= 2);
    let row_cacon_pos = cacon_boxes[cacon_boxes.len() - 1];
    let _ = run_panel_frame(&ctx, &mut state, w, h, false, click_events(row_cacon_pos));
    let (out, _) = run_panel_frame(&ctx, &mut state, w, h, false, Vec::new());
    let audit = audit_text_layout(&out.shapes, viewport);
    let pos_canco_opt = find_text_center(&audit.texts, "CANCO (i)");
    let (_, actions) = run_panel_frame(&ctx, &mut state, w, h, false, click_events(pos_canco_opt));
    assert_eq!(state.cas_slots[0].direction, CasDirection::Canco);
    assert!(actions.iter().any(|a| matches!(
        a,
        IdpJointAction::UpdateCasSlot(s)
            if s.id == slot_id && s.direction == CasDirection::Canco
    )));
}

#[test]
fn test_interactive_proline_bridges_c_term_hints_and_concordance_filtering() {
    let ctx = egui::Context::default();
    let mut state = IdpJointPanelState::new();
    let w = 1440.0_f32;
    let h = 900.0_f32;
    let viewport = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(w, h));

    // Load real-solver synthetic demo via button click
    let _ = run_panel_frame(&ctx, &mut state, w, h, false, Vec::new());
    let (out, _) = run_panel_frame(&ctx, &mut state, w, h, false, Vec::new());
    let audit = audit_text_layout(&out.shapes, viewport);
    let pos_demo = find_text_center(&audit.texts, "Load Synthetic Demo");
    let (_, actions) = run_panel_frame(&ctx, &mut state, w, h, false, click_events(pos_demo));
    assert!(actions.contains(&IdpJointAction::LoadSyntheticDemo));
    assert!(state.synthetic_demo_loaded);
    assert!(!state.proline_bridges.is_empty());
    assert!(!state.c_term_hints.is_empty());

    // Switch to Pro Bridges & C-Term tab
    let (out, _) = run_panel_frame(&ctx, &mut state, w, h, false, Vec::new());
    let audit = audit_text_layout(&out.shapes, viewport);
    let pos_bridges_tab = find_text_center(&audit.texts, "Pro Bridges & C-Term");
    let _ = run_panel_frame(&ctx, &mut state, w, h, false, click_events(pos_bridges_tab));
    assert_eq!(state.active_tab, IdpJointSubTab::ProBridgesAndCTerm);

    // Toggle Lock/Locked on first bridge and click Focus C'
    let _ = run_panel_frame(&ctx, &mut state, w, h, false, Vec::new());
    let (out, _) = run_panel_frame(&ctx, &mut state, w, h, false, Vec::new());
    let audit = audit_text_layout(&out.shapes, viewport);
    let lock_btn_label = if state.proline_bridges[0].locked {
        "Locked"
    } else {
        "Lock"
    };
    let expected_after_toggle = !state.proline_bridges[0].locked;
    let first_pro_id = state.proline_bridges[0].pro_seq_id;
    let pos_lock = find_text_center(&audit.texts, lock_btn_label);
    let (_, actions) = run_panel_frame(&ctx, &mut state, w, h, false, click_events(pos_lock));
    assert!(actions.contains(&IdpJointAction::ToggleProBridgeLock {
        pro_seq_id: first_pro_id,
        locked: expected_after_toggle,
    }));

    let (out, _) = run_panel_frame(&ctx, &mut state, w, h, false, Vec::new());
    let audit = audit_text_layout(&out.shapes, viewport);
    let pos_focus_c = find_text_center(&audit.texts, "Focus C'");
    let (_, actions) = run_panel_frame(&ctx, &mut state, w, h, false, click_events(pos_focus_c));
    assert!(actions
        .iter()
        .any(|a| matches!(a, IdpJointAction::FocusCTermAnchor { .. })));

    // Switch to HN <-> CON Concordance tab, filter by [Pro], and click Strip
    let pos_conc_tab = find_text_center(&audit.texts, "HN <-> CON Concordance");
    let _ = run_panel_frame(&ctx, &mut state, w, h, false, click_events(pos_conc_tab));
    assert_eq!(state.active_tab, IdpJointSubTab::RouteConcordance);

    let _ = run_panel_frame(&ctx, &mut state, w, h, false, Vec::new());
    let (out, _) = run_panel_frame(&ctx, &mut state, w, h, false, Vec::new());
    let audit = audit_text_layout(&out.shapes, viewport);
    let stats = state.concordance_stats();
    let pro_filter_lbl = format!("[Pro] ({})", stats.con_pro_bridge);
    let pos_pro_filter = find_text_center(&audit.texts, &pro_filter_lbl);
    let _ = run_panel_frame(&ctx, &mut state, w, h, false, click_events(pos_pro_filter));
    assert_eq!(state.route_filter, Some(JointRouteSupport::ConProBridge));

    let _ = run_panel_frame(&ctx, &mut state, w, h, false, Vec::new());
    let (out, _) = run_panel_frame(&ctx, &mut state, w, h, false, Vec::new());
    let audit = audit_text_layout(&out.shapes, viewport);
    let pos_strip = find_text_center(&audit.texts, "Strip");
    let (_, actions) = run_panel_frame(&ctx, &mut state, w, h, false, click_events(pos_strip));
    assert!(actions.iter().any(|a| matches!(
        a,
        IdpJointAction::InspectResidue {
            route: JointRouteSupport::ConProBridge,
            ..
        }
    )));
}

#[test]
fn test_source_hygiene_zero_emojis_no_duplicates_and_zero_panics_in_production_module() {
    let prod_src = include_str!("../src/idp_joint_panel.rs");
    let test_src = include_str!("idp_joint_panel_tests.rs");

    // Verify no stale 132-143 Proline range or duplicate JointRouteBadge enum remains (`P1-6`, `P2-9`)
    assert!(!prod_src.contains("enum JointRouteBadge"));
    assert!(!prod_src.contains("132-143"));
    assert!(!prod_src.contains("46 Hz / MHz"));

    for (name, src) in [
        ("idp_joint_panel.rs", prod_src),
        ("idp_joint_panel_tests.rs", test_src),
    ] {
        for (line_no, line) in src.lines().enumerate() {
            for ch in line.chars() {
                let cp = ch as u32;
                let is_emoji = (0x1F000..=0x1FFFF).contains(&cp)
                    || (0x2600..=0x27BF).contains(&cp)
                    || (0xFE00..=0xFE0F).contains(&cp);
                assert!(
                    !is_emoji,
                    "forbidden emoji U+{cp:04X} ('{ch}') in {name}:{}",
                    line_no + 1
                );
            }
        }
    }

    for (line_no, line) in prod_src.lines().enumerate() {
        let code_part = line.split("//").next().unwrap_or("");
        for forbidden in [
            ".unwrap()",
            ".expect(",
            "panic!(",
            "todo!(",
            "unimplemented!(",
            "unsafe ",
            "std::fs",
            "std::thread",
        ] {
            assert!(
                !code_part.contains(forbidden),
                "forbidden '{forbidden}' in idp_joint_panel.rs:{}",
                line_no + 1
            );
        }
    }
}
