//! Headless `egui`, `PAE` `0..31.75 A` green-to-white colormap, spectral domain box & Tolman `GDO`
//! badge rendering, 1-click `mol3d_viewer` domain-pair & putty coupling, Co-Fold pose ranking,
//! Excited-State `BIC` / parity plot interaction, `Paper White` + `Dark Oscilloscope` theme, and
//! strict zero-emoji hygiene tests for `nmr-gui::integrative_af_panel` (`T21.6b`).

use eframe::egui;
use nmr_gui::contour_painter::SyncCrosshair;
use nmr_gui::integrative_af_panel::{
    apply_integrative_af_action, pae_green_to_white_color, CofoldPreviewDialect,
    CofoldRankVerdictView, IntegrativeAfAction, IntegrativeAfPanel, IntegrativeStudioTab,
    IntegrativeStudioTheme, PoseSortColumn, PAE_COLORBAR_MAX_ANGSTROM,
};
use nmr_gui::mol3d_overlay::QuantitativeTrack;
use nmr_gui::mol3d_viewer::Mol3dViewerPanel;
use nmr_gui::sidechain_widget::SequenceRibbonState;
use nmr_gui::ss_ribbon_widget::SsRibbonState;

fn run_headless_frame(
    ctx: &egui::Context,
    panel: &mut IntegrativeAfPanel,
    paper_white: bool,
    events: Vec<egui::Event>,
) -> Vec<IntegrativeAfAction> {
    let raw_input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::pos2(0.0, 0.0),
            egui::vec2(1024.0, 768.0),
        )),
        events,
        ..Default::default()
    };
    let mut actions = Vec::new();
    let mut out = ctx.run_ui(raw_input, |ui| {
        actions = panel.show(ui, paper_white);
    });
    out.textures_delta.clear();
    actions
}

#[test]
fn test_pae_green_to_white_colormap_monotonic_0_to_31_75_angstrom() {
    let c0 = pae_green_to_white_color(0.0);
    let c8 = pae_green_to_white_color(8.0);
    let c16 = pae_green_to_white_color(16.0);
    let c24 = pae_green_to_white_color(24.0);
    let c_max = pae_green_to_white_color(PAE_COLORBAR_MAX_ANGSTROM);
    let c_over = pae_green_to_white_color(64.0);
    let c_nan = pae_green_to_white_color(f32::NAN);

    assert_eq!(c0, egui::Color32::from_rgb(0, 77, 38));
    assert_eq!(c_max, egui::Color32::from_rgb(255, 255, 255));
    assert_eq!(c_over, c_max);
    assert_eq!(c_nan, c_max);

    let lum = |c: egui::Color32| -> f64 {
        0.2126 * (c.r() as f64) + 0.7152 * (c.g() as f64) + 0.0722 * (c.b() as f64)
    };
    assert!(lum(c0) < lum(c8));
    assert!(lum(c8) < lum(c16));
    assert!(lum(c16) < lum(c24));
    assert!(lum(c24) < lum(c_max));

    // Verify Paper White and Dark Oscilloscope themes are distinct and well-formed
    let pw = IntegrativeStudioTheme::for_mode(true);
    let dk = IntegrativeStudioTheme::for_mode(false);
    assert_ne!(pw.panel_bg, dk.panel_bg);
    assert_ne!(pw.text_primary, dk.text_primary);
}

#[test]
fn test_headless_concordance_and_pae_domain_heatmap_click_to_mol3d() {
    let ctx = egui::Context::default();
    let mut panel = IntegrativeAfPanel::new();
    let (demo_ref, status) = panel
        .load_synthetic_two_domain_demo()
        .expect("load synthetic 2-domain AF demo");
    assert!(status.contains("Analyzed 76 residues"), "got: {status}");
    assert!(status.contains("synthetic demo"), "got: {status}");
    assert!(!status.contains("Validated"), "got: {status}");

    // Verify 76-residue Ubiquitin PAE matrix & 2 rigid domains clustered by T21.5 spectral PAE partitioning
    let pae = panel.pae_matrix.as_ref().expect("PAE matrix loaded");
    assert_eq!(
        pae.n_residues, 76,
        "PAE tab must use the consistent 76-residue Ubiquitin model"
    );
    let part = panel
        .domain_partition
        .as_ref()
        .expect("PAE domain partition");
    assert_eq!(part.domains.len(), 2);

    // Verify Tolman GDO motion test flagged decoupled Domain 2 (`0.48x` scaled RDCs) AND Q_free is strictly finite!
    let gdo = panel.tolman_gdo_report.as_ref().expect("Tolman GDO report");
    assert_eq!(gdo.domain_fits.len(), 2);
    assert!(gdo.min_rho_gdo < 0.70);
    for fit in &gdo.domain_fits {
        assert!(
            fit.q_free.is_finite(),
            "Domain {} Q_free must be finite (never NaN), got {}",
            fit.domain_id,
            fit.q_free
        );
    }

    let mut crosshair = SyncCrosshair::default();
    let mut seq_ribbon = SequenceRibbonState::new();
    let mut ss_ribbon = SsRibbonState::new();
    let mut mol3d = Mol3dViewerPanel::new();

    // Sync initial structure & concordance onto SS ribbon and 3D viewer
    panel.sync_to_ribbon_and_mol3d(
        &mut ss_ribbon,
        &mut mol3d,
        &demo_ref.sequence,
        &demo_ref.seq_ids,
    );
    assert!(mol3d.ensemble.is_some());
    assert_eq!(ss_ribbon.ss3d_track.len(), 76);
    assert_eq!(ss_ribbon.concordance_summary.classes.len(), 76);

    // Verify residues 1..=9 N-terminal beta-strand consistency between 3D view, SS ribbon, and AF concordance (`B13`)
    let conc = panel
        .concordance_report
        .as_ref()
        .expect("concordance report");
    assert_eq!(conc.residues.len(), 76);
    let beta_in_1_to_9_af = conc.residues[0..9]
        .iter()
        .filter(|r| r.state4_af.as_str() == "Sheet" || format!("{:?}", r.state4_af) == "Sheet")
        .count();
    let beta_in_1_to_9_ribbon = ss_ribbon.ss3d_track[0..9]
        .iter()
        .filter(|&&s| matches!(s, nmr_gui::ss_ribbon_widget::SsState4::Sheet))
        .count();
    assert!(
        beta_in_1_to_9_af >= 5,
        "AF concordance residues 1..=9 must reflect Ubiquitin beta1 strand (not Coil), got {beta_in_1_to_9_af}/9 Sheet: {:?}",
        conc.residues[0..9]
            .iter()
            .map(|r| (r.seq_id, r.state4_af))
            .collect::<Vec<_>>()
    );
    assert!(
        beta_in_1_to_9_ribbon >= 5,
        "SS ribbon residues 1..=9 must reflect Ubiquitin beta1 strand, got {beta_in_1_to_9_ribbon}/9 Sheet"
    );

    // Render Tab 1 in both Dark Oscilloscope and Paper White modes
    panel.active_tab = IntegrativeStudioTab::ConcordanceSummary;
    let _ = run_headless_frame(&ctx, &mut panel, false, Vec::new());
    let _ = run_headless_frame(&ctx, &mut panel, true, Vec::new());

    // Render Tab 2 (PAE Heatmap) and verify telemetry
    panel.active_tab = IntegrativeStudioTab::PaeDomainHeatmap;
    let _ = run_headless_frame(&ctx, &mut panel, false, Vec::new());
    assert_eq!(panel.last_telemetry.pae_cells_drawn, 76 * 76);
    assert_eq!(panel.last_telemetry.pae_domain_boxes_drawn, 2);
    assert!(panel.last_telemetry.gdo_badges_drawn >= 3);

    // Simulate clicking an inter-domain off-diagonal cell `(i ~ 10 in D1, j ~ 50 in D2)` on the PAE heatmap
    let pae_rect = panel.last_pae_plot_rect.expect("pae_plot_rect recorded");
    let click_pos = egui::pos2(
        pae_rect.min.x + pae_rect.width() * (50.5 / 76.0),
        pae_rect.min.y + pae_rect.height() * (10.5 / 76.0),
    );
    let click_events = vec![
        egui::Event::PointerMoved(click_pos),
        egui::Event::PointerButton {
            pos: click_pos,
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: egui::Modifiers::NONE,
        },
        egui::Event::PointerButton {
            pos: click_pos,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        },
    ];
    let actions = run_headless_frame(&ctx, &mut panel, false, click_events);
    assert!(
        actions.iter().any(|a| matches!(
            a,
            IntegrativeAfAction::HighlightDomainPair {
                domain_a: 1,
                domain_b: 2,
                ..
            }
        )),
        "expected HighlightDomainPair(1, 2), got {actions:?}"
    );
    assert_eq!(panel.selected_domain_pair, Some((1, 2)));

    // Apply action to workbench state and confirm Mol3dViewerPanel highlights D1 <-> D2!
    for act in &actions {
        apply_integrative_af_action(
            act,
            &panel,
            Some(&demo_ref),
            &demo_ref.sequence,
            &demo_ref.seq_ids,
            &mut crosshair,
            &mut seq_ribbon,
            &mut ss_ribbon,
            &mut mol3d,
        );
    }
    assert_eq!(mol3d.scene_cfg.highlighted_seq_idx, Some(10));
    assert_eq!(mol3d.restraints.len(), 1);
    assert!(mol3d.restraints[0].label.contains("D1:11 <-> D2:51"));

    // Verify SetMol3dTrack switches to AlphaFoldPlddt and AfNmrConcordance cleanly
    for trk in [
        QuantitativeTrack::AlphaFoldPlddt,
        QuantitativeTrack::AfNmrConcordance,
    ] {
        apply_integrative_af_action(
            &IntegrativeAfAction::SetMol3dTrack {
                track: trk,
                putty_enabled: true,
            },
            &panel,
            Some(&demo_ref),
            &demo_ref.sequence,
            &demo_ref.seq_ids,
            &mut crosshair,
            &mut seq_ribbon,
            &mut ss_ribbon,
            &mut mol3d,
        );
        assert_eq!(mol3d.track_data.active_track, trk);
        assert!(mol3d.track_data.putty_enabled);
    }
}

#[test]
fn test_headless_cofold_tab_and_excited_state_bic_parity_interaction() {
    let ctx = egui::Context::default();
    let mut panel = IntegrativeAfPanel::new();
    let (demo_ref, _) = panel.load_synthetic_two_domain_demo().unwrap();

    // Render Tab 3 (Co-Fold Pose Ranker) across dialects & sort columns
    panel.active_tab = IntegrativeStudioTab::CofoldPoseRanker;
    for dialect in [
        CofoldPreviewDialect::Boltz2Yaml,
        CofoldPreviewDialect::AlphaFold3Json,
    ] {
        panel.preview_dialect = dialect;
        for sort_col in [
            PoseSortColumn::TotalScoreDesc,
            PoseSortColumn::CspScoreDesc,
            PoseSortColumn::NoePreViolationAsc,
            PoseSortColumn::RdcQFreeAsc,
        ] {
            panel.pose_sort = sort_col;
            let _ = run_headless_frame(&ctx, &mut panel, true, Vec::new());
        }
    }
    assert_eq!(panel.pose_verdict, CofoldRankVerdictView::TopPoseSelected);
    assert!(panel
        .af3_json_preview
        .contains("\"dialect\": \"alphafold3\""));
    assert!(panel.boltz2_yaml_preview.contains("pocket:"));

    // Render Tab 4 (Excited-State Hypothesis Explorer)
    panel.active_tab = IntegrativeStudioTab::ExcitedStateExplorer;
    let _ = run_headless_frame(&ctx, &mut panel, false, Vec::new());
    assert_eq!(panel.last_telemetry.bic_bars_drawn, 5);
    assert_eq!(panel.last_telemetry.parity_points_drawn, 6);

    // Click top bar on the BIC chart to trigger JumpToDynamicsResidue
    let bic_rect = panel.last_bic_chart_rect.expect("bic_chart_rect");
    let bar_click = egui::pos2(bic_rect.min.x + 150.0, bic_rect.min.y + 18.0);
    let click_events = vec![
        egui::Event::PointerMoved(bar_click),
        egui::Event::PointerButton {
            pos: bar_click,
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: egui::Modifiers::NONE,
        },
        egui::Event::PointerButton {
            pos: bar_click,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        },
    ];
    let actions = run_headless_frame(&ctx, &mut panel, false, click_events);
    assert!(
        actions.iter().any(|a| matches!(
            a,
            IntegrativeAfAction::JumpToDynamicsResidue {
                seq_idx: 37,
                seq_id: 38,
                ..
            }
        )),
        "expected JumpToDynamicsResidue(37, 38), got {actions:?}"
    );

    let mut crosshair = SyncCrosshair::default();
    let mut seq_ribbon = SequenceRibbonState::new();
    let mut ss_ribbon = SsRibbonState::new();
    let mut mol3d = Mol3dViewerPanel::new();
    for act in &actions {
        apply_integrative_af_action(
            act,
            &panel,
            Some(&demo_ref),
            &demo_ref.sequence,
            &demo_ref.seq_ids,
            &mut crosshair,
            &mut seq_ribbon,
            &mut ss_ribbon,
            &mut mol3d,
        );
    }
    assert_eq!(ss_ribbon.selected_residue, Some(37));
    assert_eq!(mol3d.scene_cfg.highlighted_seq_idx, Some(37));

    // Also test live `evaluate_excited_state_from_buffers` against `nmr_core::excited_state` on 1UBQ
    panel.structure_input_buf =
        include_str!("../../../data/fixtures/ubiquitin_1ubq_xray.pdb").to_string();
    panel.exchange_tsv_buf = "chain_id\tseq_id\tres_name\tatom\tdw_ppm\tdw_err_ppm\tsign_mode\n\
                              A\t19\tPRO\tCB\t2.40\t0.10\tsigned\n\
                              A\t19\tPRO\tCG\t-2.60\t0.10\tsigned\n\
                              A\t19\tPRO\tCA\t-0.55\t0.10\tsigned\n\
                              A\t18\tGLU\tCA\t1.15\t0.10\tunsigned\n\
                              A\t18\tGLU\tC\t1.35\t0.10\tunsigned\n\
                              A\t20\tSER\tN\t-1.82\t0.12\tsigned\n\
                              A\t21\tASP\tN\t0.48\t0.10\tunsigned\n"
        .to_string();
    let live_rep = panel
        .evaluate_excited_state_from_buffers(None)
        .expect("live excited state classification");
    assert_eq!(live_rep.verdict.as_str(), "Committed(XxxProCisTrans)");
    assert_eq!(panel.excited_parity_points.len(), 7);
    let _ = run_headless_frame(&ctx, &mut panel, true, Vec::new());

    // Also test live `populate_from_cofold_report` against `nmr_core::cofold_bridge`
    let apo_ubq = nmr_core::struct_shifts::StructureModel::parse_str(
        include_str!("../../../data/fixtures/ubiquitin_1ubq_xray.pdb"),
        Some("pdb"),
        false,
    )
    .expect("parse 1ubq");
    let (csp_rows, _) = nmr_core::cofold_bridge::parse_csp_restraint_str(
        "seq_id\tres_name\tcsp_ppm\n\
         8\tLEU\t0.52\n44\tILE\t0.65\n68\tHIS\t0.48\n70\tVAL\t0.56\n3\tILE\t0.45\n\
         10\tGLY\t0.02\n11\tLYS\t0.02\n12\tTHR\t0.03\n14\tTHR\t0.02\n16\tGLU\t0.02\n\
         18\tGLU\t0.01\n20\tSER\t0.02\n22\tTHR\t0.02\n24\tGLU\t0.01\n26\tVAL\t0.02\n\
         28\tALA\t0.02\n31\tGLN\t0.01\n32\tASP\t0.02\n33\tLYS\t0.02\n35\tGLY\t0.01\n",
    )
    .expect("parse csp");
    let bundle = nmr_core::cofold_bridge::extract_nmr_interface_restraints(
        &nmr_core::cofold_bridge::CofoldRestraintInput {
            job_name: "ubq_pep".to_string(),
            receptor_chain_id: Some("A".to_string()),
            binder_chain_id: Some("B".to_string()),
            entities: nmr_core::cofold_bridge::parse_cofold_fasta_str(
                ">A\nMQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG\n>B\nFAL\n",
            )
            .expect("fasta"),
            apo_receptor_model: Some(apo_ubq),
            raw_csp_rows: csp_rows,
            ..Default::default()
        },
        &nmr_core::cofold_bridge::CofoldRestraintExtractConfig::default(),
    )
    .expect("extract cofold restraints");
    let pose_rep = nmr_core::cofold_bridge::score_cofold_poses(
        &[nmr_core::cofold_bridge::StructurePoseInput::from_coord_str(
            "ubq_self_pose",
            include_str!("../../../data/fixtures/ubiquitin_1ubq_xray.pdb"),
            Some("pdb"),
        )],
        &bundle,
        &nmr_core::cofold_bridge::PoseScoreConfig::default(),
    )
    .expect("score pose");
    let ubq_seq_ids: Vec<i32> = (1..=76).collect();
    panel.populate_from_cofold_report(Some(&bundle), &pose_rep, &ubq_seq_ids);
    assert!(panel
        .af3_json_preview
        .contains("\"dialect\": \"alphafold3\""));
    assert!(panel.boltz2_yaml_preview.contains("pocket:"));
    assert!(!panel.surface_pocket_seq_ids.is_empty());
    assert!(!panel.buried_allosteric_seq_ids.is_empty());
    panel.active_tab = IntegrativeStudioTab::CofoldPoseRanker;
    let _ = run_headless_frame(&ctx, &mut panel, false, Vec::new());
}

#[test]
fn test_hostile_buffer_ingest_errors_and_zero_emoji_audit() {
    let mut panel = IntegrativeAfPanel::new();
    assert!(panel.ingest_from_buffers(None).is_err());

    panel.structure_input_buf =
        "ATOM      1  CA  ALA A   1      NaN   0.000   0.000  1.00 90.00           C\n".to_string();
    assert!(panel.ingest_from_buffers(None).is_err());
    assert!(panel.last_error.is_some());

    // Strict zero-emoji audit on `integrative_af_panel.rs` and `integrative_af_panel_tests.rs`
    for src in [
        include_str!("../src/integrative_af_panel.rs"),
        include_str!("integrative_af_panel_tests.rs"),
    ] {
        for ch in src.chars() {
            let u = ch as u32;
            let is_emoji = (0x1F300..=0x1FAFF).contains(&u)
                || (0x2600..=0x27BF).contains(&u)
                || (0xFE00..=0xFE0F).contains(&u);
            assert!(
                !is_emoji,
                "forbidden emoji character U+{u:04X} ('{ch}') detected"
            );
        }
    }
}

#[test]
fn test_pae_heatmap_cached_texture_uploads_once_over_five_frames() {
    let ctx = egui::Context::default();
    let mut panel = IntegrativeAfPanel::new();
    let _ = panel
        .load_synthetic_two_domain_demo()
        .expect("load synthetic 2-domain AF demo");
    panel.active_tab = IntegrativeStudioTab::PaeDomainHeatmap;

    for _ in 0..5 {
        let _ = run_headless_frame(&ctx, &mut panel, false, Vec::new());
    }
    assert_eq!(panel.last_telemetry().pae_texture_uploads, 1);
}

#[test]
fn test_af_demo_local_synthetic_labels_76aa_and_no_nan() {
    use nmr_gui::workbench_mode::audit_text_layout;

    let mut panel = IntegrativeAfPanel::new();
    let (demo_ref, _) = panel
        .load_synthetic_two_domain_demo()
        .expect("load 76-aa Ubiquitin synthetic AF demo");
    assert_eq!(demo_ref.sequence.len(), 76);
    assert_eq!(panel.pae_matrix.as_ref().expect("pae").n_residues, 76);

    let vp = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1280.0, 800.0));
    for tab in IntegrativeStudioTab::ALL {
        let ctx = egui::Context::default();
        panel.active_tab = tab;
        let mut out = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(vp),
                ..Default::default()
            },
            |ui| {
                egui::CentralPanel::default().show(ui, |ui| {
                    let _ = panel.show(ui, false);
                });
            },
        );
        out.textures_delta.clear();
        let audit = audit_text_layout(&out.shapes, vp);
        let joined: String = audit
            .texts
            .iter()
            .map(|t| t.text.as_str())
            .collect::<Vec<_>>()
            .join(" | ");
        assert!(
            joined.contains("Synthetic Demo"),
            "AF demo tab {tab:?} must display explicit local 'Synthetic Demo' label: {joined}"
        );
        assert!(
            !joined.contains("NaN"),
            "AF demo tab {tab:?} must never render NaN (e.g. Q_free=NaN): {joined}"
        );
    }
}
