//! Headless `egui` layout, interaction, typography floor (`>= 11.0 px`), zero-clipping,
//! zero-overlap, preset loading, typed-edge validation, node drag, and zero-emoji tests
//! for `nmr_gui::workflow_dag_panel` (`T22.11`).

use eframe::egui;
use nmr_core::workflow_dag::{DagEdge, DagError, NodeKind, PortType, MAX_DAG_NODES};
use nmr_gui::workbench_mode::{apply_type_scale, audit_text_layout, MIN_FONT_PX};
use nmr_gui::workflow_dag_panel::{
    port_type_color, DagPresetKind, DagTheme, WorkflowDagAction, WorkflowDagPanel,
};

fn run_headless_frame(
    ctx: &egui::Context,
    panel: &mut WorkflowDagPanel,
    width: f32,
    height: f32,
    paper_white: bool,
    events: Vec<egui::Event>,
) -> (egui::FullOutput, Vec<WorkflowDagAction>) {
    let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(width, height));
    let mut actions = Vec::new();
    let mut out = ctx.run_ui(
        egui::RawInput {
            screen_rect: Some(rect),
            events,
            ..Default::default()
        },
        |ui| {
            apply_type_scale(ui.ctx());
            egui::CentralPanel::default().show(ui, |ui| {
                actions = panel.show(ui, paper_white);
            });
        },
    );
    out.textures_delta.clear();
    (out, actions)
}

#[test]
fn test_all_presets_zero_clipped_zero_overlapping_and_min_11px_at_1280x800() {
    let viewport = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1280.0, 800.0));

    for paper_white in [false, true] {
        for preset in DagPresetKind::ALL {
            let ctx = egui::Context::default();
            let mut panel = WorkflowDagPanel::new();
            panel.load_preset(preset);

            // Two passes so egui layout converges cleanly
            let _ = run_headless_frame(&ctx, &mut panel, 1280.0, 800.0, paper_white, Vec::new());
            let (out, _) =
                run_headless_frame(&ctx, &mut panel, 1280.0, 800.0, paper_white, Vec::new());

            let audit = audit_text_layout(&out.shapes, viewport);
            assert!(
                !audit.texts.is_empty(),
                "expected rendered text for preset {preset:?}"
            );
            assert!(
                audit.clipped.is_empty(),
                "clipped labels at 1280x800 (preset={preset:?}, paper_white={paper_white}): {:?}",
                audit.clipped
            );
            assert!(
                audit.overlapping.is_empty(),
                "overlapping labels at 1280x800 (preset={preset:?}, paper_white={paper_white}): {:?}",
                audit.overlapping
            );
            assert!(
                audit.min_font_px >= MIN_FONT_PX - 1e-3,
                "font floor violated (preset={preset:?}, paper_white={paper_white}): min={} < {}",
                audit.min_font_px,
                MIN_FONT_PX
            );

            assert!(panel.last_telemetry.is_valid);
            assert_eq!(panel.last_telemetry.nodes_drawn, panel.dag.nodes.len());
            assert_eq!(panel.last_telemetry.edges_drawn, panel.dag.edges.len());
            assert!(panel.last_telemetry.ports_drawn >= panel.dag.nodes.len() * 2);

            let joined: String = audit
                .texts
                .iter()
                .map(|t| t.text.as_str())
                .collect::<Vec<_>>()
                .join(" | ");
            assert!(
                joined.contains("Valid DAG"),
                "missing Valid DAG banner for {preset:?}: {joined}"
            );
            assert!(
                joined.contains("ferric-dag-v1"),
                "missing ferric-dag-v1 schema tag for {preset:?}: {joined}"
            );
            let sha = panel
                .last_telemetry
                .last_manifest_sha256
                .as_deref()
                .expect("manifest sha256 recorded");
            assert!(
                joined.contains(&sha[..12]),
                "missing short manifest_sha256 {} in banner for {preset:?}",
                &sha[..12]
            );
        }
    }
}

#[test]
fn test_interactive_node_add_connect_disconnect_delete_and_invalid_edge_rejection() {
    let ctx = egui::Context::default();
    let mut panel = WorkflowDagPanel::new();
    panel.load_preset(DagPresetKind::Backbone);
    let initial_sha = panel
        .compiled_result
        .as_ref()
        .expect("valid backbone preset")
        .manifest_sha256
        .clone();

    // 1. Add a SidechainAssign node -> initially invalid (missing required `assignments` input)
    let new_id = panel
        .add_node(NodeKind::SidechainAssign)
        .expect("add SidechainAssign node");
    assert!(matches!(
        panel.compiled_result,
        Err(DagError::MissingRequiredInput { ref node_id, ref port })
            if node_id == &new_id && port == "assignments"
    ));

    // Render invalid banner and verify layout audit still passes with 0 clipped/overlapping
    let viewport = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1280.0, 800.0));
    let _ = run_headless_frame(&ctx, &mut panel, 1280.0, 800.0, false, Vec::new());
    let (out_invalid, _) = run_headless_frame(&ctx, &mut panel, 1280.0, 800.0, false, Vec::new());
    let audit_invalid = audit_text_layout(&out_invalid.shapes, viewport);
    assert!(
        audit_invalid.clipped.is_empty(),
        "{:?}",
        audit_invalid.clipped
    );
    assert!(
        audit_invalid.overlapping.is_empty(),
        "{:?}",
        audit_invalid.overlapping
    );
    assert!(audit_invalid
        .texts
        .iter()
        .any(|t| t.text.contains("Invalid DAG")));

    // 2. Reject invalid edge with PortTypeMismatch (ingest:raw_fid -> sidechain:assignments)
    let bad_edge = DagEdge::new("ingest", "raw_fid", &new_id, "assignments");
    assert!(matches!(
        panel.connect_edge(bad_edge),
        Err(DagError::PortTypeMismatch {
            from_type: PortType::RawFid,
            to_type: PortType::BackboneAssignments,
            ..
        })
    ));
    assert!(panel.last_interaction_error.is_some());

    // Render with rejection alert bar active -> still 0 clipped/overlapping at 1280x800
    let _ = run_headless_frame(&ctx, &mut panel, 1280.0, 800.0, false, Vec::new());
    let (out_rej, _) = run_headless_frame(&ctx, &mut panel, 1280.0, 800.0, false, Vec::new());
    let audit_rej = audit_text_layout(&out_rej.shapes, viewport);
    assert!(audit_rej.clipped.is_empty(), "{:?}", audit_rej.clipped);
    assert!(
        audit_rej.overlapping.is_empty(),
        "{:?}",
        audit_rej.overlapping
    );
    assert!(audit_rej.texts.iter().any(|t| t.text.contains("Rejected:")));

    // 3. Connect valid edge (assign_bb:assignments -> sidechain:assignments) -> DAG becomes valid!
    panel
        .connect_edge(DagEdge::new(
            "assign_bb",
            "assignments",
            &new_id,
            "assignments",
        ))
        .expect("valid typed edge connects");
    let updated_sha = panel
        .compiled_result
        .as_ref()
        .expect("DAG valid after connecting required input")
        .manifest_sha256
        .clone();
    assert_ne!(updated_sha, initial_sha);

    // 4. Attempt self-loop cycle edge on SidechainAssign -> rejected!
    assert!(matches!(
        panel.connect_edge(DagEdge::new(&new_id, "assignments", &new_id, "assignments")),
        Err(DagError::InputPortAlreadyConnected { .. } | DagError::CycleDetected { .. })
    ));

    // 5. Disconnect newly added edge -> DAG returns to invalid missing-input state
    let last_edge_idx = panel.dag.edges.len() - 1;
    let removed = panel.disconnect_edge(last_edge_idx);
    assert!(removed.is_some());
    assert!(panel.compiled_result.is_err());

    // 6. Delete added node -> DAG returns to original valid preset with identical manifest_sha256
    assert!(panel.delete_node(&new_id));
    let restored_sha = panel
        .compiled_result
        .as_ref()
        .expect("restored preset valid")
        .manifest_sha256
        .clone();
    assert_eq!(restored_sha, initial_sha);
}

#[test]
fn test_canvas_pointer_drag_and_port_click_wiring_and_dry_run_cache() {
    let ctx = egui::Context::default();
    let mut panel = WorkflowDagPanel::new();
    panel.load_preset(DagPresetKind::Metabolomics);

    // First dry run executes all 4 stages; second dry run gets 4 cache hits
    let trace1 = panel.run_dry_run().expect("first dry run");
    assert_eq!(trace1.executed_count, 4);
    assert_eq!(trace1.cache_hit_count, 0);
    let trace2 = panel.run_dry_run().expect("second dry run");
    assert_eq!(trace2.executed_count, 0);
    assert_eq!(trace2.cache_hit_count, 4);

    let sha_before_drag = panel
        .compiled_result
        .as_ref()
        .expect("valid manifest")
        .manifest_sha256
        .clone();

    // Initial layout pass to capture exact node/port coordinates
    let _ = run_headless_frame(&ctx, &mut panel, 1280.0, 800.0, false, Vec::new());
    let ingest_rect = *panel
        .last_telemetry
        .node_rects
        .get("ingest")
        .expect("ingest node rect");
    let start_pos = egui::pos2(ingest_rect.min.x + 24.0, ingest_rect.min.y + 14.0);
    let drag_target = egui::pos2(start_pos.x + 30.0, start_pos.y + 24.0);
    let pos_before = panel.dag.nodes[0].position;

    // Simulate pointer press -> drag -> release on `ingest` node card
    let _ = run_headless_frame(
        &ctx,
        &mut panel,
        1280.0,
        800.0,
        false,
        vec![
            egui::Event::PointerMoved(start_pos),
            egui::Event::PointerButton {
                pos: start_pos,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
        ],
    );
    let _ = run_headless_frame(
        &ctx,
        &mut panel,
        1280.0,
        800.0,
        false,
        vec![egui::Event::PointerMoved(drag_target)],
    );
    let _ = run_headless_frame(
        &ctx,
        &mut panel,
        1280.0,
        800.0,
        false,
        vec![egui::Event::PointerButton {
            pos: drag_target,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        }],
    );

    let pos_after = panel.dag.nodes[0].position;
    assert!(
        (pos_after[0] - pos_before[0]).abs() > 10.0 && (pos_after[1] - pos_before[1]).abs() > 10.0,
        "expected node drag to move position: before={pos_before:?} after={pos_after:?}"
    );
    let sha_after_drag = panel
        .compiled_result
        .as_ref()
        .expect("manifest after drag")
        .manifest_sha256
        .clone();
    assert_eq!(
        sha_before_drag, sha_after_drag,
        "dragging visual layout must not change content-addressed manifest_sha256"
    );

    // Interactive port-to-port click rejection test: click `ingest:raw_fid` out -> click `export:peaks` in
    let out_port_pos = *panel
        .last_telemetry
        .output_port_centers
        .get(&("ingest".to_string(), "raw_fid".to_string()))
        .expect("ingest:raw_fid output center");
    let bad_in_port_pos = *panel
        .last_telemetry
        .input_port_centers
        .get(&("export".to_string(), "peaks".to_string()))
        .expect("export:peaks input center");

    let _ = run_headless_frame(
        &ctx,
        &mut panel,
        1280.0,
        800.0,
        false,
        vec![
            egui::Event::PointerMoved(out_port_pos),
            egui::Event::PointerButton {
                pos: out_port_pos,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
            egui::Event::PointerButton {
                pos: out_port_pos,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            },
        ],
    );
    let (_, wire_actions) = run_headless_frame(
        &ctx,
        &mut panel,
        1280.0,
        800.0,
        false,
        vec![
            egui::Event::PointerMoved(bad_in_port_pos),
            egui::Event::PointerButton {
                pos: bad_in_port_pos,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
            egui::Event::PointerButton {
                pos: bad_in_port_pos,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            },
        ],
    );
    assert!(
        wire_actions
            .iter()
            .any(|a| matches!(a, WorkflowDagAction::EdgeRejected(msg) if msg.contains("Port type mismatch"))),
        "expected EdgeRejected action on incompatible port click, got {wire_actions:?}"
    );

    // Interactive valid port-to-port click: `chemometrics:peaks` -> `export:peaks`
    let good_out_pos = *panel
        .last_telemetry
        .output_port_centers
        .get(&("chemometrics".to_string(), "peaks".to_string()))
        .expect("chemometrics:peaks output center");
    let _ = run_headless_frame(
        &ctx,
        &mut panel,
        1280.0,
        800.0,
        false,
        vec![
            egui::Event::PointerMoved(good_out_pos),
            egui::Event::PointerButton {
                pos: good_out_pos,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
            egui::Event::PointerButton {
                pos: good_out_pos,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            },
        ],
    );
    let (_, ok_actions) = run_headless_frame(
        &ctx,
        &mut panel,
        1280.0,
        800.0,
        false,
        vec![
            egui::Event::PointerMoved(bad_in_port_pos),
            egui::Event::PointerButton {
                pos: bad_in_port_pos,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
            egui::Event::PointerButton {
                pos: bad_in_port_pos,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            },
        ],
    );
    assert!(
        ok_actions
            .iter()
            .any(|a| matches!(a, WorkflowDagAction::EdgeConnected(_))),
        "expected EdgeConnected action on compatible port click, got {ok_actions:?}"
    );
}

#[test]
fn test_theme_palette_zero_emojis_and_no_panics_in_production_sources() {
    // Verify distinct dark/light themes and non-transparent port colors for all 13 PortTypes
    let dark = DagTheme::for_mode(false);
    let light = DagTheme::for_mode(true);
    assert_ne!(dark.canvas_bg, light.canvas_bg);
    assert_ne!(dark.text_primary, light.text_primary);
    for pt in PortType::ALL {
        assert_eq!(port_type_color(pt, false).a(), 255);
        assert_eq!(port_type_color(pt, true).a(), 255);
    }

    // Strict zero-emoji audit across all T22.11 sources
    let files: [(&str, &str); 4] = [
        (
            "workflow_dag.rs",
            include_str!("../../nmr-core/src/workflow_dag.rs"),
        ),
        (
            "workflow_dag_tests.rs",
            include_str!("../../nmr-core/tests/workflow_dag_tests.rs"),
        ),
        (
            "workflow_dag_panel.rs",
            include_str!("../src/workflow_dag_panel.rs"),
        ),
        (
            "workflow_dag_panel_tests.rs",
            include_str!("workflow_dag_panel_tests.rs"),
        ),
    ];
    for (name, src) in files {
        for (line_idx, line) in src.lines().enumerate() {
            for ch in line.chars() {
                let cp = ch as u32;
                let is_emoji = (0x1F000..=0x1FFFF).contains(&cp)
                    || (0x2600..=0x27BF).contains(&cp)
                    || (0xFE00..=0xFE0F).contains(&cp)
                    || cp == 0x200D;
                assert!(
                    !is_emoji,
                    "forbidden emoji U+{cp:04X} ('{ch}') in {name}:{}",
                    line_idx + 1
                );
            }
        }
    }

    // Strict production-code audit: zero .unwrap(), .expect(), panic!, unsafe
    for (name, src) in [
        (
            "workflow_dag.rs",
            include_str!("../../nmr-core/src/workflow_dag.rs"),
        ),
        (
            "workflow_dag_panel.rs",
            include_str!("../src/workflow_dag_panel.rs"),
        ),
        ("dag_cmd.rs", include_str!("../../nmr-cli/src/dag_cmd.rs")),
    ] {
        for forbidden in [".unwrap(", ".expect(", "panic!(", "unsafe "] {
            assert!(
                !src.contains(forbidden),
                "forbidden token {forbidden:?} found in production file {name}"
            );
        }
    }
}

#[test]
fn test_add_node_enforces_max_dag_nodes_cap_fail_closed() {
    let mut panel = WorkflowDagPanel::new();
    panel.load_preset(DagPresetKind::Metabolomics);
    while panel.dag.nodes.len() < MAX_DAG_NODES {
        panel
            .add_node(NodeKind::IngestBrukerVarian)
            .expect("adding up to MAX_DAG_NODES succeeds");
    }
    assert_eq!(panel.dag.nodes.len(), MAX_DAG_NODES);

    let err = panel.add_node(NodeKind::IngestBrukerVarian);
    assert_eq!(
        err,
        Err(DagError::TooManyNodes {
            count: MAX_DAG_NODES + 1,
            max: MAX_DAG_NODES,
        })
    );
    assert_eq!(panel.dag.nodes.len(), MAX_DAG_NODES);
    assert!(panel.last_interaction_error.is_some());
}
