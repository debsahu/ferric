//! Headless `egui` integration tests for `v0.17.1` VnmrJ color-coded vertical mouse-mode rails,
//! connected folder-style tab widget (`draw_folder_tab`), 1D Suite upgrades (`Predict 1H / 13C`,
//! amber/mirror `Overlay Predicted 1H / 13C`, 1-Click SVG/ACS export, 1D VnmrJ rail), and the
//! searchable `About FERRIC & Scientific References` modal.
//!
//! Provenance: Original clean-room headless `egui::Context` verification suite for FERRIC.

use std::path::Path;

use eframe::egui;
use nmr_core::nmr1d::{Experiment1DKind, Spectrum1D};
use nmr_core::spectrum::AxisPpm;
use nmr_gui::about_references_modal::{
    AboutReferencesModal, CitationDomain, OPEN_DATA_ACKNOWLEDGMENTS, VERIFIED_CITATIONS,
};
use nmr_gui::stack1d_panel::{
    export_stack1d_svg, render_stack1d_panel, InspectorTab1d, Stack1dState, ToolMode1d, Trace1d,
};
use nmr_gui::vnmrj_rail::{
    draw_folder_tab, draw_folder_tab_themed, render_2d_vnmrj_rail, InstantAction2D, MouseMode2D,
    RailAction2D, RailColorGroup, VnmrJRail2DConfig, RAIL_WIDTH_PX,
};
use nmr_gui::workbench_mode::{apply_type_scale, audit_text_layout, MIN_FONT_PX};

fn run_frame<F: FnMut(&mut egui::Ui)>(
    ctx: &egui::Context,
    w: f32,
    h: f32,
    time_s: f64,
    events: Vec<egui::Event>,
    mut f: F,
) -> egui::FullOutput {
    let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(w, h));
    let raw = egui::RawInput {
        screen_rect: Some(rect),
        time: Some(time_s),
        events,
        ..Default::default()
    };
    let mut out = ctx.run_ui(raw, |ui| {
        apply_type_scale(ui.ctx());
        f(ui);
    });
    out.textures_delta.clear();
    out
}

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

fn synthetic_1h_spectrum(label: &str, peaks: &[(f64, f64, f64)]) -> Spectrum1D {
    let n = 512;
    let axis = AxisPpm::from_bounds("1H", n, 10.0, 0.0, 600.0);
    let mut real = vec![0.02_f64; n];
    for (k, slot) in real.iter_mut().enumerate() {
        let ppm = axis.index_to_ppm(k as f64);
        for &(center, height, width) in peaks {
            *slot += height * (-0.5 * ((ppm - center) / width).powi(2)).exp();
        }
    }
    Spectrum1D::new(label, Experiment1DKind::Proton1D, axis, real, None)
}

#[test]
fn test_2d_vnmrj_rail_all_modes_instants_and_color_groups() {
    // Verify canonical VnmrJ 5-family color assignments
    assert_eq!(
        MouseMode2D::ZoomBox.color_group(),
        RailColorGroup::Navigation
    );
    assert_eq!(MouseMode2D::Pan.color_group(), RailColorGroup::Navigation);
    assert_eq!(
        MouseMode2D::CrosshairMeasure.color_group(),
        RailColorGroup::MeasureReference
    );
    assert_eq!(
        MouseMode2D::PickPeak.color_group(),
        RailColorGroup::PickQuantify
    );
    assert_eq!(
        MouseMode2D::EditDeletePeak.color_group(),
        RailColorGroup::Curate
    );
    assert_eq!(
        InstantAction2D::ContourUp.color_group(),
        RailColorGroup::ProcessingScale
    );
    assert_eq!(
        InstantAction2D::ContourDown.color_group(),
        RailColorGroup::ProcessingScale
    );
    assert_eq!(
        InstantAction2D::WaterSuppress.color_group(),
        RailColorGroup::ProcessingScale
    );
    assert_eq!(
        InstantAction2D::FullView.color_group(),
        RailColorGroup::Navigation
    );

    // Every functional color family has a distinct accent in both dark and paper-white themes
    for paper_white in [false, true] {
        let accents: Vec<_> = RailColorGroup::ALL
            .iter()
            .map(|g| g.palette(paper_white).accent)
            .collect();
        for i in 0..accents.len() {
            for j in (i + 1)..accents.len() {
                assert_ne!(
                    accents[i], accents[j],
                    "color groups {i} and {j} must have distinct accents (paper_white={paper_white})"
                );
            }
        }
    }

    // Headless click transitions across all 5 modes + 4 instant actions
    let ctx = egui::Context::default();
    let mut mode = MouseMode2D::ZoomBox;
    let mut last_out = None;
    let mut t = 0.1;

    // Warm-up frame to capture exact slot rects
    let _ = run_frame(&ctx, 400.0, 500.0, t, Vec::new(), |ui| {
        egui::CentralPanel::default().show(ui, |ui| {
            last_out = Some(render_2d_vnmrj_rail(
                ui,
                &mut mode,
                VnmrJRail2DConfig::new(false, false),
            ));
        });
    });
    let rail = last_out.expect("2D rail output");
    assert!(
        (rail.rail_rect.width() - RAIL_WIDTH_PX).abs() < 0.5,
        "expected ~38 px rail width, got {}",
        rail.rail_rect.width()
    );

    // Click every mouse mode slot and confirm mode transition + action return
    for target in MouseMode2D::ALL {
        t += 0.2;
        let center = rail.mode_rect(target).center();
        let mut emitted = None;
        let _ = run_frame(&ctx, 400.0, 500.0, t, click_events(center), |ui| {
            egui::CentralPanel::default().show(ui, |ui| {
                let out = render_2d_vnmrj_rail(ui, &mut mode, VnmrJRail2DConfig::new(false, false));
                emitted = out.action;
            });
        });
        assert_eq!(mode, target, "clicking {target:?} slot must activate it");
        assert_eq!(emitted, Some(RailAction2D::SetMode(target)));
    }

    // Click every instant action slot and confirm action emission without clobbering active mode
    let saved_mode = mode;
    for target in InstantAction2D::ALL {
        t += 0.2;
        let center = rail.instant_rect(target).center();
        let mut emitted = None;
        let _ = run_frame(&ctx, 400.0, 500.0, t, click_events(center), |ui| {
            egui::CentralPanel::default().show(ui, |ui| {
                let out = render_2d_vnmrj_rail(
                    ui,
                    &mut mode,
                    VnmrJRail2DConfig::new(true, target == InstantAction2D::WaterSuppress),
                );
                emitted = out.action;
            });
        });
        assert_eq!(mode, saved_mode, "instant action must preserve sticky mode");
        assert_eq!(emitted, Some(RailAction2D::Instant(target)));
    }

    // Disabled rail must ignore clicks
    t += 0.2;
    let center = rail.mode_rect(MouseMode2D::Pan).center();
    let mut disabled_action = None;
    let _ = run_frame(&ctx, 400.0, 500.0, t, click_events(center), |ui| {
        egui::CentralPanel::default().show(ui, |ui| {
            let out = render_2d_vnmrj_rail(
                ui,
                &mut mode,
                VnmrJRail2DConfig {
                    paper_white: false,
                    water_suppressed: false,
                    enabled: false,
                },
            );
            disabled_action = out.action;
        });
    });
    assert_eq!(disabled_action, None);
    assert_eq!(mode, saved_mode);
}

#[test]
fn test_draw_folder_tab_visual_chrome_selection_and_font_floor() {
    let ctx = egui::Context::default();
    let mut selected_idx = 0_usize;
    let labels = ["Peaks", "Multiplets", "Integrals"];
    let mut tab_rects = [egui::Rect::NOTHING; 3];

    // Frame 1: render dark & light folder tabs and verify shapes + font floor
    let out = run_frame(&ctx, 640.0, 200.0, 0.1, Vec::new(), |ui| {
        egui::CentralPanel::default().show(ui, |ui| {
            ui.horizontal(|ui| {
                for (i, lbl) in labels.iter().enumerate() {
                    let resp = draw_folder_tab(ui, lbl, selected_idx == i);
                    tab_rects[i] = resp.rect;
                    if resp.clicked() {
                        selected_idx = i;
                    }
                }
            });
            ui.add_space(12.0);
            ui.horizontal(|ui| {
                let _ = draw_folder_tab_themed(ui, "Paper Active", true, true);
                let _ = draw_folder_tab_themed(ui, "Paper Inactive", false, true);
            });
        });
    });

    let vp = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(640.0, 200.0));
    let audit = audit_text_layout(&out.shapes, vp);
    assert!(audit.clipped.is_empty(), "clipped: {:?}", audit.clipped);
    assert!(
        audit.overlapping.is_empty(),
        "overlapping: {:?}",
        audit.overlapping
    );
    assert!(
        audit.min_font_px >= MIN_FONT_PX - 1e-3,
        "folder tab font {} < {MIN_FONT_PX}",
        audit.min_font_px
    );

    // Verify inactive tabs have distinct filled + stroked background chrome (never plain static text)
    let mut inactive_rect_shapes = 0_usize;
    let mut active_joined_shapes = 0_usize;
    fn scan_shapes(
        shape: &egui::Shape,
        inactive_r: egui::Rect,
        active_r: egui::Rect,
        inactive_hits: &mut usize,
        active_joined_hits: &mut usize,
    ) {
        match shape {
            egui::Shape::Rect(rs) => {
                if (rs.rect.min - inactive_r.min).length() < 1.5
                    && (rs.rect.max - inactive_r.max).length() < 1.5
                    && rs.fill != egui::Color32::TRANSPARENT
                    && rs.stroke.width >= 0.9
                {
                    *inactive_hits += 1;
                }
                if (rs.rect.left() - active_r.left()).abs() < 1.5
                    && rs.rect.bottom() >= active_r.bottom() + 1.0
                    && rs.fill != egui::Color32::TRANSPARENT
                {
                    *active_joined_hits += 1;
                }
            }
            egui::Shape::Vec(v) => {
                for s in v {
                    scan_shapes(s, inactive_r, active_r, inactive_hits, active_joined_hits);
                }
            }
            _ => {}
        }
    }
    for cs in &out.shapes {
        scan_shapes(
            &cs.shape,
            tab_rects[1],
            tab_rects[0],
            &mut inactive_rect_shapes,
            &mut active_joined_shapes,
        );
    }
    assert!(
        inactive_rect_shapes >= 1,
        "inactive folder tab must draw a visible background fill + 1px outline"
    );
    assert!(
        active_joined_shapes >= 1,
        "active folder tab must seamlessly extend downward into content pane seam"
    );

    // Clicking the second tab switches selection
    let _ = run_frame(
        &ctx,
        640.0,
        200.0,
        0.3,
        click_events(tab_rects[1].center()),
        |ui| {
            egui::CentralPanel::default().show(ui, |ui| {
                ui.horizontal(|ui| {
                    for (i, lbl) in labels.iter().enumerate() {
                        let resp = draw_folder_tab(ui, lbl, selected_idx == i);
                        if resp.clicked() {
                            selected_idx = i;
                        }
                    }
                });
            });
        },
    );
    assert_eq!(selected_idx, 1, "clicking folder tab 1 must select it");
}

#[test]
fn test_stack1d_vnmrj_rail_predicted_overlay_and_one_click_exports() {
    // 1. Verify 1D VnmrJ color groups
    assert_eq!(ToolMode1d::Zoom.color_group(), RailColorGroup::Navigation);
    assert_eq!(ToolMode1d::Pan.color_group(), RailColorGroup::Navigation);
    assert_eq!(
        ToolMode1d::Reference.color_group(),
        RailColorGroup::MeasureReference
    );
    assert_eq!(
        ToolMode1d::PeakPick.color_group(),
        RailColorGroup::PickQuantify
    );
    assert_eq!(
        ToolMode1d::IntegrateRoi.color_group(),
        RailColorGroup::PickQuantify
    );
    assert_eq!(
        ToolMode1d::Multiplet.color_group(),
        RailColorGroup::PickQuantify
    );
    assert_eq!(
        ToolMode1d::Phase.color_group(),
        RailColorGroup::ProcessingScale
    );

    // 2. Build 1D state with experimental spectrum + predicted amber/mirror Spectrum1D overlay
    let exp_spec = synthetic_1h_spectrum(
        "Exp 1H",
        &[(7.26, 8.0, 0.02), (3.85, 5.5, 0.025), (1.25, 11.0, 0.02)],
    );
    let pred_spec = synthetic_1h_spectrum(
        "Pred 1H",
        &[(7.24, 6.0, 0.025), (3.88, 4.8, 0.025), (1.27, 9.5, 0.025)],
    );

    let mut state = Stack1dState::default();
    state.load_user_spectra(vec![exp_spec]);
    assert_eq!(state.traces.len(), 1);
    assert!(state.pending_predict_1d_request.is_none());

    // Queue Predict 1H & Predict 13C requests
    state.request_predict_1d("CC(=O)Oc1ccccc1C(=O)O", false);
    assert_eq!(
        state.pending_predict_1d_request,
        Some(("CC(=O)Oc1ccccc1C(=O)O".to_string(), false))
    );
    state.request_predict_1d("", true);
    assert_eq!(
        state.pending_predict_1d_request,
        Some((String::new(), true))
    );

    // Attach predicted overlay (mirror mode by default) and render headless frames
    state.set_predicted_overlay(pred_spec, "Predicted 1H (Aspirin)");
    assert!(state.show_predicted_overlay);
    assert!(state.predicted_overlay_mirror);

    let ctx = egui::Context::default();
    let mut t = 0.1;
    let out_mirror = run_frame(&ctx, 1280.0, 800.0, t, Vec::new(), |ui| {
        egui::CentralPanel::default().show(ui, |ui| {
            render_stack1d_panel(ui, &mut state);
        });
    });
    let vp = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1280.0, 800.0));
    let audit_mirror = audit_text_layout(&out_mirror.shapes, vp);
    assert!(
        audit_mirror.clipped.is_empty(),
        "1D mirror overlay clipped texts: {:?}",
        audit_mirror.clipped
    );
    assert!(
        audit_mirror.overlapping.is_empty(),
        "1D mirror overlay overlapping texts: {:?}",
        audit_mirror.overlapping
    );
    assert!(audit_mirror.min_font_px >= MIN_FONT_PX - 1e-3);
    assert!(
        audit_mirror
            .texts
            .iter()
            .any(|tx| tx.text.contains("Predicted 1H (Aspirin) (Mirror)")),
        "expected amber mirror overlay legend on 1D canvas"
    );

    // Verify 1D left VnmrJ rail rect & clicking each tool mode slot on the 1D rail
    let rail_r = state
        .last_rail_rect
        .expect("1D VnmrJ rail must record rect");
    assert!((rail_r.width() - RAIL_WIDTH_PX).abs() < 0.5);
    assert_eq!(state.last_tool_rail_rects.len(), ToolMode1d::ALL.len());

    for target in [
        ToolMode1d::Pan,
        ToolMode1d::Phase,
        ToolMode1d::Reference,
        ToolMode1d::IntegrateRoi,
        ToolMode1d::Multiplet,
        ToolMode1d::PeakPick,
        ToolMode1d::Zoom,
    ] {
        // Settle layout for any inline contextual sub-bar before clicking the rendered slot
        t += 0.1;
        let _ = run_frame(&ctx, 1280.0, 800.0, t, Vec::new(), |ui| {
            egui::CentralPanel::default().show(ui, |ui| {
                render_stack1d_panel(ui, &mut state);
            });
        });
        t += 0.2;
        let slot_r = state
            .tool_rail_slot_rect(target)
            .expect("1D rail must record slot rect");
        let _ = run_frame(
            &ctx,
            1280.0,
            800.0,
            t,
            click_events(slot_r.center()),
            |ui| {
                egui::CentralPanel::default().show(ui, |ui| {
                    render_stack1d_panel(ui, &mut state);
                });
            },
        );
        assert_eq!(
            state.tool_mode, target,
            "clicking 1D VnmrJ rail icon must switch to {target:?}"
        );
    }

    // Also verify upright predicted overlay mode & SVG export embedding amber predicted trace
    state.predicted_overlay_mirror = false;
    t += 0.2;
    let out_upright = run_frame(&ctx, 1280.0, 800.0, t, Vec::new(), |ui| {
        egui::CentralPanel::default().show(ui, |ui| {
            render_stack1d_panel(ui, &mut state);
        });
    });
    let audit_up = audit_text_layout(&out_upright.shapes, vp);
    assert!(
        audit_up
            .texts
            .iter()
            .any(|tx| tx.text.contains("Predicted 1H (Aspirin) (Overlay)")),
        "expected upright overlay legend on 1D canvas"
    );

    let svg = export_stack1d_svg(&state, 1200, 640);
    assert!(svg.starts_with("<svg ") && svg.contains("</svg>"));
    assert!(
        svg.contains("stroke=\"#f59e0b\""),
        "exported SVG must include amber predicted overlay path when active"
    );

    // 1-Click Export helpers & toolbar button presence
    assert!(audit_up.texts.iter().any(|tx| tx.text == "Export SVG"));
    assert!(audit_up.texts.iter().any(|tx| tx.text == "Copy ACS"));
    assert!(audit_up.texts.iter().any(|tx| tx.text == "Predict 1H/13C"));
    assert!(audit_up.texts.iter().any(|tx| tx.text == "Overlay Pred"));

    let exported_svg = state.export_current_svg(1200, 640);
    assert_eq!(
        state.last_exported_svg.as_deref(),
        Some(exported_svg.as_str())
    );
    let acs = state.build_acs_journal_string();
    assert!(
        acs.contains("NMR") && acs.contains('δ'),
        "ACS journal string must contain NMR header and chemical shift delta: {acs}"
    );
    assert_eq!(state.last_copied_acs_report.as_deref(), Some(acs.as_str()));

    // Open inspector drawer and confirm connected folder tabs render cleanly without overlap
    state.show_1d_analysis_drawer = true;
    state.selected_inspector_tab = InspectorTab1d::Multiplets;
    t += 0.2;
    let out_drawer = run_frame(&ctx, 1280.0, 800.0, t, Vec::new(), |ui| {
        egui::CentralPanel::default().show(ui, |ui| {
            render_stack1d_panel(ui, &mut state);
        });
    });
    let audit_drawer = audit_text_layout(&out_drawer.shapes, vp);
    assert!(audit_drawer.min_font_px >= MIN_FONT_PX - 1e-3);

    // Clear predicted overlay removes amber SVG path
    state.clear_predicted_overlay();
    let svg_no_pred = export_stack1d_svg(&state, 1200, 640);
    assert!(!svg_no_pred.contains("stroke=\"#f59e0b\""));
    let _ = Trace1d::new("t", "t", 0.0, vec![1.0, 0.0], vec![0.0, 1.0]);
}

#[test]
fn test_about_references_modal_search_render_copy_and_no_hallucinated_dois() {
    // 1. Verify zero fabricated DOIs in citation catalog & all domains populated
    for dom in CitationDomain::ALL {
        let count = VERIFIED_CITATIONS
            .iter()
            .filter(|c| c.domain == dom)
            .count();
        assert!(
            count >= 5,
            "domain {dom:?} should have >= 5 classic NMR citations"
        );
    }
    for cit in VERIFIED_CITATIONS {
        let formatted = cit.format_reference();
        let lower = formatted.to_ascii_lowercase();
        assert!(
            !lower.contains("doi:") && !lower.contains("doi.org") && !lower.contains("10.10"),
            "must not contain fabricated DOIs: {formatted}"
        );
        assert!(cit.year >= 1955 && cit.year <= 2025);
    }
    assert!(OPEN_DATA_ACKNOWLEDGMENTS.len() >= 4);

    // 2. Verify search filter & domain filter behavior
    let mut modal = AboutReferencesModal::new();
    assert!(!modal.open);
    modal.open_modal();
    assert!(modal.open);
    assert_eq!(modal.filtered_citations().len(), VERIFIED_CITATIONS.len());

    modal.search_query = "Lipari".to_string();
    let lipari = modal.filtered_citations();
    assert_eq!(lipari.len(), 1);
    assert_eq!(lipari[0].year, 1982);

    modal.search_query = "icoshift".to_string();
    let ico = modal.filtered_citations();
    assert_eq!(ico.len(), 1);
    assert_eq!(ico[0].year, 2010);

    modal.search_query.clear();
    modal.selected_domain = Some(CitationDomain::SmallMoleculeDosy);
    let dosy_cits = modal.filtered_citations();
    assert!(dosy_cits
        .iter()
        .all(|c| c.domain == CitationDomain::SmallMoleculeDosy));
    assert!(dosy_cits.iter().any(|c| c.authors.contains("Stejskal")));

    // 3. Headless render (2 passes so egui::Window sizes & paints) & click "Copy Citation"
    modal.selected_domain = None;
    modal.search_query = "Lipari".to_string();
    let ctx = egui::Context::default();
    let vp = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1280.0, 800.0));
    let _ = run_frame(&ctx, 1280.0, 800.0, 0.1, Vec::new(), |ui| {
        let _ = modal.show(ui.ctx());
    });
    let out = run_frame(&ctx, 1280.0, 800.0, 0.2, Vec::new(), |ui| {
        let _ = modal.show(ui.ctx());
    });
    let audit = audit_text_layout(&out.shapes, vp);
    assert!(
        audit.min_font_px >= MIN_FONT_PX - 1e-3,
        "modal min font {} < {MIN_FONT_PX}",
        audit.min_font_px
    );
    let expected_ver = format!("FERRIC v{}", nmr_core::VERSION);
    assert!(
        audit.texts.iter().any(|t| t.text.contains(&expected_ver)),
        "modal must display {expected_ver}"
    );
    assert!(
        audit.texts.iter().any(|t| t.text.contains("MIT Licence")),
        "modal must display MIT Licence"
    );
    for required_ack in ["BMRB", "CC0", "PDB", "AlphaFold DB", "CC-BY-4.0", "NP-MRD"] {
        assert!(
            audit.texts.iter().any(|t| t.text.contains(required_ack)),
            "modal must acknowledge {required_ack}"
        );
    }

    // Locate and click the "Copy Citation" button in headless egui
    let copy_btn_rect = audit
        .texts
        .iter()
        .find(|t| t.text == "Copy Citation")
        .expect("expected Copy Citation button in modal")
        .rect;
    let _ = run_frame(
        &ctx,
        1280.0,
        800.0,
        0.3,
        click_events(copy_btn_rect.center()),
        |ui| {
            let _ = modal.show(ui.ctx());
        },
    );
    let copied = modal
        .last_copied_citation
        .as_deref()
        .expect("clicking Copy Citation must populate last_copied_citation");
    assert!(copied.contains("Lipari") && copied.contains("1982"));
}

#[test]
fn test_zero_emojis_unwrap_panic_or_unsafe_in_new_rail_and_modal_sources() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let prod_files = [
        manifest_dir.join("src/vnmrj_rail.rs"),
        manifest_dir.join("src/about_references_modal.rs"),
    ];
    for path in &prod_files {
        let src = std::fs::read_to_string(path).expect("read prod source file");
        assert!(
            !src.contains(".unwrap("),
            "{} must have zero .unwrap() calls",
            path.display()
        );
        assert!(
            !src.contains("panic!("),
            "{} must have zero panic!() calls",
            path.display()
        );
        assert!(
            !src.contains("unsafe "),
            "{} must have zero unsafe blocks",
            path.display()
        );
        for ch in src.chars() {
            let cp = ch as u32;
            let is_emoji = (0x1F300..=0x1FAFF).contains(&cp)
                || (0x2600..=0x26FF).contains(&cp)
                || (0x2700..=0x27BF).contains(&cp);
            assert!(
                !is_emoji,
                "forbidden emoji U+{cp:04X} found in {}",
                path.display()
            );
        }
    }
}
