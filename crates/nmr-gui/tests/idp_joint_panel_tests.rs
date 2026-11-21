//! Headless `egui` layout, typography (`>= 11.0 px`), zero-clipping / zero-overlap,
//! interactive `CAS` slot state-mutation, Proline-bridge / C-terminal `COO-` / `HN <-> CON`
//! concordance, and zero-emoji source hygiene tests for `nmr_gui::idp_joint_panel` (`T23.3`).

use eframe::egui;
use nmr_core::c13_detected::CasMode;
use nmr_gui::idp_joint_panel::{
    draw_route_badge_pill, CasAxisConsistency, CasDirection, CasResidueType, CasSlotEntry,
    IdpJointAction, IdpJointPanelState, IdpJointPanelTheme, IdpJointSubTab, IdpRegimeSelection,
    JointRouteBadge, C_TERM_COO_NOMINAL_PPM, GAMMA_N15_OVER_H1, TROSY_HALF_J_HN_HZ,
};
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
fn test_regime_cas_direction_axis_and_route_badge_domain_logic() {
    // Regime flags & route participation
    assert_eq!(IdpRegimeSelection::ALL.len(), 4);
    assert_eq!(IdpRegimeSelection::AutoJoint.cli_flag(), "auto");
    assert_eq!(IdpRegimeSelection::IdpJoint.cli_flag(), "idp-joint");
    assert_eq!(IdpRegimeSelection::Idp13cOnly.cli_flag(), "idp-13c");
    assert_eq!(IdpRegimeSelection::Globular.cli_flag(), "globular");

    assert!(IdpRegimeSelection::IdpJoint.uses_joint_fusion());
    assert!(IdpRegimeSelection::IdpJoint.uses_hn_route());
    assert!(IdpRegimeSelection::IdpJoint.uses_con_route());
    assert!(!IdpRegimeSelection::Idp13cOnly.uses_hn_route());
    assert!(IdpRegimeSelection::Idp13cOnly.uses_con_route());
    assert!(IdpRegimeSelection::Globular.uses_hn_route());
    assert!(!IdpRegimeSelection::Globular.uses_con_route());

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

    // CasResidueType resolution including TAVI and Other custom selectors
    assert_eq!(
        CasResidueType::Ala.resolve_residues("").expect("ala"),
        vec!["ALA"]
    );
    assert_eq!(
        CasResidueType::Gly.resolve_residues("").expect("gly"),
        vec!["GLY"]
    );
    assert_eq!(
        CasResidueType::Ser.resolve_residues("").expect("ser"),
        vec!["SER"]
    );
    assert_eq!(
        CasResidueType::Thr.resolve_residues("").expect("thr"),
        vec!["THR"]
    );
    assert_eq!(
        CasResidueType::Tavi.resolve_residues("").expect("tavi"),
        vec!["THR", "ALA", "VAL", "ILE"]
    );
    assert_eq!(
        CasResidueType::Other
            .resolve_residues("ASN+ASP")
            .expect("other valid"),
        vec!["ASN", "ASP"]
    );
    assert!(CasResidueType::Other.resolve_residues("").is_err());
    assert!(CasResidueType::Other
        .resolve_residues("NOT_AN_AA_99")
        .is_err());

    // Axis consistency validation for 2D (13C', 15N) CAS spectra
    assert_eq!(
        CasAxisConsistency::validate_axes("13C", 182.0, 168.0, "15N", 142.0, 102.0),
        CasAxisConsistency::Consistent
    );
    assert_eq!(
        CasAxisConsistency::validate_axes("15N", 142.0, 102.0, "13C", 182.0, 168.0),
        CasAxisConsistency::SwappedAxes
    );
    assert_eq!(
        CasAxisConsistency::validate_axes("13C", 70.0, 15.0, "15N", 132.0, 105.0),
        CasAxisConsistency::OutOfRange
    );
    assert_eq!(
        CasAxisConsistency::validate_axes("1H", 10.0, 6.0, "15N", 132.0, 105.0),
        CasAxisConsistency::WrongNuclei
    );
    assert_eq!(
        CasAxisConsistency::validate_axes("13C", f64::NAN, 168.0, "15N", 142.0, 102.0),
        CasAxisConsistency::OutOfRange
    );

    // JointRouteBadge themes and distinct colors
    for paper_white in [false, true] {
        let colors: Vec<egui::Color32> = JointRouteBadge::ALL
            .iter()
            .map(|b| b.color(paper_white))
            .collect();
        for i in 0..colors.len() {
            for j in (i + 1)..colors.len() {
                assert_ne!(colors[i], colors[j], "route badge colors must be distinct");
            }
        }
    }
    let dark_theme = IdpJointPanelTheme::for_mode(false);
    let light_theme = IdpJointPanelTheme::for_mode(true);
    assert_ne!(dark_theme.card_bg, light_theme.card_bg);
    assert_ne!(dark_theme.text_primary, light_theme.text_primary);
}

#[test]
fn test_cas_slot_state_mutations_cli_preview_and_trosy_offsets() {
    let mut state = IdpJointPanelState::new();
    assert_eq!(state.regime, IdpRegimeSelection::AutoJoint);
    assert!(state.roots_hnco);
    assert!(state.filter_nh2_pairs);
    assert!(state.trosy_half_j_calibration);

    // TROSY 1/2 1J_HN (+/-46 Hz / MHz) calculation
    state.spectrometer_mhz = 800.0;
    let (dh_800, dn_800) = state.trosy_half_j_offsets_ppm();
    assert!((dh_800 - TROSY_HALF_J_HN_HZ / 800.0).abs() < 1e-12);
    assert!((dn_800 - TROSY_HALF_J_HN_HZ / (800.0 * GAMMA_N15_OVER_H1)).abs() < 1e-12);

    // Non-finite spectrometer MHz safely falls back to 800 MHz
    state.spectrometer_mhz = f64::NAN;
    let (dh_fallback, _) = state.trosy_half_j_offsets_ppm();
    assert!((dh_fallback - dh_800).abs() < 1e-12);
    state.spectrometer_mhz = 600.0;

    // Add, re-type, redirect, load, and remove CAS slots
    let id1 = state.add_cas_slot(CasResidueType::Ser, "", CasDirection::Cacon, "ser_2d.tsv");
    let id2 = state.add_cas_slot(
        CasResidueType::Other,
        "VAL,ILE",
        CasDirection::Canco,
        "vi_2d.tsv",
    );
    assert_eq!(state.cas_slots.len(), 2);
    assert_eq!(state.ready_cas_slot_count(), 0);

    assert!(state.set_cas_slot_loaded(id1, true, 16, CasAxisConsistency::Consistent));
    assert!(state.set_cas_slot_loaded(id2, true, 12, CasAxisConsistency::SwappedAxes));
    assert_eq!(
        state.ready_cas_slot_count(),
        1,
        "swapped-axis slot must not count as ready"
    );

    // Fix axis consistency and re-type slot 1 from Ser -> Thr, direction Cacon -> Canco
    assert!(state.set_cas_slot_loaded(id2, true, 12, CasAxisConsistency::Consistent));
    assert_eq!(state.ready_cas_slot_count(), 2);
    assert!(state.set_cas_slot_type(id1, CasResidueType::Thr, None));
    assert!(state.set_cas_slot_direction(id1, CasDirection::Canco));
    assert_eq!(state.cas_slots[0].to_cli_arg(), "canco:thr=ser_2d.tsv");
    assert_eq!(state.cas_slots[1].to_cli_arg(), "canco:val,ile=vi_2d.tsv");

    state.regime = IdpRegimeSelection::IdpJoint;
    let cli = state.cli_preview_command();
    assert!(cli.contains("--regime idp-joint"));
    assert!(cli.contains("--roots hnco"));
    assert!(cli.contains("--trosy-calibrate"));
    assert!(cli.contains("--cas canco:thr=ser_2d.tsv"));
    assert!(cli.contains("--cas canco:val,ile=vi_2d.tsv"));

    assert!(state.remove_cas_slot(id1));
    assert!(!state.remove_cas_slot(id1));
    assert_eq!(state.cas_slots.len(), 1);
    assert_eq!(state.cas_slots[0].id, id2);
}

#[test]
fn test_headless_layout_1280x800_and_1440x900_all_tabs_zero_overlap_clipping_and_font_floor() {
    let ctx = egui::Context::default();
    let mut state = IdpJointPanelState::new();
    state.load_synthetic_idp_demo();

    // Also add an `Other` slot with swapped axes to stress-test extra controls & badges
    let extra_id = state.add_cas_slot(
        CasResidueType::Other,
        "ASN+ASP",
        CasDirection::Canco,
        "demo/canco_asx_swapped.tsv",
    );
    state.set_cas_slot_loaded(extra_id, true, 7, CasAxisConsistency::SwappedAxes);

    for (width, height) in [(1280.0_f32, 800.0_f32), (1440.0_f32, 900.0_f32)] {
        let viewport = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(width, height));
        for paper_white in [false, true] {
            for tab in IdpJointSubTab::ALL {
                state.active_tab = tab;
                // Two passes so egui stabilizes grid column widths and scroll areas
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
fn test_headless_window_and_reusable_route_badge_pill_layout() {
    let ctx = egui::Context::default();
    let mut state = IdpJointPanelState::new();
    state.load_synthetic_idp_demo();
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
            egui::Panel::bottom("route_badge_strip_test").show(ui, |ui| {
                ui.horizontal(|ui| {
                    for badge in JointRouteBadge::ALL {
                        draw_route_badge_pill(ui, badge, true);
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

    // Stabilize initial layout
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
    let pos_trosy = find_text_center(&audit.texts, "TROSY 1/2 1J_HN auto-calibration");
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
    let slot_id = state.add_cas_slot(
        CasResidueType::Gly,
        "",
        CasDirection::Cacon,
        "cacon_gly.tsv",
    );
    let w = 1280.0_f32;
    let h = 800.0_f32;
    let viewport = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(w, h));

    // Stabilize layout
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
    // Find the row's CACON (i-1) combo (the second one in the UI, below the draft bar)
    let cacon_boxes: Vec<egui::Pos2> = audit
        .texts
        .iter()
        .filter(|t| t.text.trim() == "CACON (i-1)")
        .map(|t| t.rect.center())
        .collect();
    assert!(
        cacon_boxes.len() >= 2,
        "expected draft and slot CACON combos, got {cacon_boxes:?}"
    );
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

    // Load synthetic demo via button click
    let _ = run_panel_frame(&ctx, &mut state, w, h, false, Vec::new());
    let (out, _) = run_panel_frame(&ctx, &mut state, w, h, false, Vec::new());
    let audit = audit_text_layout(&out.shapes, viewport);
    let pos_demo = find_text_center(&audit.texts, "Load Synthetic Demo");
    let (_, actions) = run_panel_frame(&ctx, &mut state, w, h, false, click_events(pos_demo));
    assert!(actions.contains(&IdpJointAction::LoadSyntheticDemo));
    assert!(state.synthetic_demo_loaded);
    assert_eq!(state.cas_slots.len(), 4);
    assert_eq!(state.proline_bridges.len(), 3);
    assert_eq!(state.c_term_hints.len(), 1);
    assert!((state.c_term_hints[0].expected_coo_ppm - C_TERM_COO_NOMINAL_PPM).abs() < 1e-9);

    let stats = state.concordance_stats();
    assert_eq!(stats.total, 12);
    assert_eq!(stats.cross_validated, 7);
    assert_eq!(stats.con_pro_bridge, 3);
    assert_eq!(stats.hn_only, 1);
    assert_eq!(stats.con_only, 1);
    assert_eq!(stats.high_conf_count, 12);

    // Switch to Pro Bridges & C-Term tab
    let (out, _) = run_panel_frame(&ctx, &mut state, w, h, false, Vec::new());
    let audit = audit_text_layout(&out.shapes, viewport);
    let pos_bridges_tab = find_text_center(&audit.texts, "Pro Bridges & C-Term");
    let _ = run_panel_frame(&ctx, &mut state, w, h, false, click_events(pos_bridges_tab));
    assert_eq!(state.active_tab, IdpJointSubTab::ProBridgesAndCTerm);

    // Toggle Lock on unlocked P29 bridge and Focus C-term anchor
    let _ = run_panel_frame(&ctx, &mut state, w, h, false, Vec::new());
    let (out, _) = run_panel_frame(&ctx, &mut state, w, h, false, Vec::new());
    let audit = audit_text_layout(&out.shapes, viewport);
    let pos_lock = find_text_center(&audit.texts, "Lock");
    let (_, actions) = run_panel_frame(&ctx, &mut state, w, h, false, click_events(pos_lock));
    assert!(actions.contains(&IdpJointAction::ToggleProBridgeLock {
        pro_seq_id: 29,
        locked: true,
    }));
    assert!(state.proline_bridges[2].locked);

    let (out, _) = run_panel_frame(&ctx, &mut state, w, h, false, Vec::new());
    let audit = audit_text_layout(&out.shapes, viewport);
    let pos_focus_c = find_text_center(&audit.texts, "Focus C'");
    let (_, actions) = run_panel_frame(&ctx, &mut state, w, h, false, click_events(pos_focus_c));
    assert!(actions.contains(&IdpJointAction::FocusCTermAnchor {
        seq_id: 64,
        c_prime_ppm: 179.48,
    }));
    assert_eq!(state.selected_residue, Some(64));

    // Switch to HN <-> CON Concordance tab, filter by PRO-CON, and inspect a residue strip
    let pos_conc_tab = find_text_center(&audit.texts, "HN <-> CON Concordance");
    let _ = run_panel_frame(&ctx, &mut state, w, h, false, click_events(pos_conc_tab));
    assert_eq!(state.active_tab, IdpJointSubTab::RouteConcordance);

    let _ = run_panel_frame(&ctx, &mut state, w, h, false, Vec::new());
    let (out, _) = run_panel_frame(&ctx, &mut state, w, h, false, Vec::new());
    let audit = audit_text_layout(&out.shapes, viewport);
    let pos_pro_filter = find_text_center(&audit.texts, "PRO-CON (3)");
    let _ = run_panel_frame(&ctx, &mut state, w, h, false, click_events(pos_pro_filter));
    assert_eq!(state.route_filter, Some(JointRouteBadge::ConProBridge));

    let _ = run_panel_frame(&ctx, &mut state, w, h, false, Vec::new());
    let (out, _) = run_panel_frame(&ctx, &mut state, w, h, false, Vec::new());
    let audit = audit_text_layout(&out.shapes, viewport);
    assert!(audit.texts.iter().any(|t| t.text.contains("PRO14")));
    assert!(!audit.texts.iter().any(|t| t.text.contains("GLU31")));

    let pos_strip = find_text_center(&audit.texts, "Strip");
    let (_, actions) = run_panel_frame(&ctx, &mut state, w, h, false, click_events(pos_strip));
    assert!(actions.contains(&IdpJointAction::InspectResidue {
        seq_id: 14,
        badge: JointRouteBadge::ConProBridge,
    }));
    assert_eq!(state.selected_residue, Some(14));
}

#[test]
fn test_source_hygiene_zero_emojis_and_zero_panics_in_production_module() {
    let prod_src = include_str!("../src/idp_joint_panel.rs");
    let test_src = include_str!("idp_joint_panel_tests.rs");

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
