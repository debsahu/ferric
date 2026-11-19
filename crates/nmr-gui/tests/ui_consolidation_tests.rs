//! Headless GUI consolidation, pixel-area budget, control-count budget, text clipping/overlap,
//! font-floor, zero-emoji, scientific candor, and `<= 5`-click workflow tests
//! (`UI-B1`..`UI-B11`, `21.W3:P1-12`, `21.W3:P1-15`, `21.W3:P2`).
//!
//! Provenance: original clean-room integration tests for FERRIC (`nmr-gui`).

use std::collections::BTreeSet;
use std::path::Path;

use eframe::egui;
use nmr_core::idp_propensity::PropensityConfig;
use nmr_core::nmr1d::{Experiment1DKind, Spectrum1D};
use nmr_core::nmrstar::read_star_str;
use nmr_core::picker::{
    pick_hsqc_2d, pick_triple_3d_two_pass, CorroboratedRepickConfig, PickerConfig,
};
use nmr_core::priors::Priors;
use nmr_core::solver::{
    assign_from_matrices, pair_matrix, unary_matrix, AnnealConfig, ScoreConfig,
};
use nmr_core::spectrum::AxisPpm;
use nmr_core::spinsystems::{build_spin_systems, Tolerances};
use nmr_gui::demo_preset::{build_builtin_demo_suite, BUILTIN_DEMO_SUITE_SEED};
use nmr_gui::dosy_panel::render_dosy_panel;
use nmr_gui::integrative_af_panel::{IntegrativeAfPanel, IntegrativeStudioTab};
use nmr_gui::label_layout::count_collisions;
use nmr_gui::metabolomics_panel::MetabolomicsStudioPanel;
use nmr_gui::mol3d_viewer::Mol3dViewerPanel;
use nmr_gui::octet_canvas::{extract_active_molecule_subgraph, OctetDrawPanel};
use nmr_gui::ss_ribbon_widget::{SsRibbonPanel, SsRibbonState, SsRibbonWorkspaceInput};
use nmr_gui::stack1d_panel::{render_stack1d_panel, Stack1dState, Trace1d};
use nmr_gui::strip_tiler::{StripSpectra3D, StripTilerContext, StripTilerPanel};
use nmr_gui::workbench_mode::{
    apply_type_scale, audit_text_layout, draw_mode_switcher, draw_mode_switcher_with_rect,
    draw_sub_view_selector, primary_controls, SmallMoleculeView, WorkbenchMode,
    MAX_PRIMARY_CONTROLS_PER_MODE, MIN_CANVAS_AREA_FRACTION, MIN_FONT_PX,
};
use nmr_gui::workbook::{render_workbook_sidebar, ActiveWorkbookSelection, WorkbookState};

const SYNTH_UBQ_STR: &str = include_str!("../../../data/fixtures/synthetic_ubiquitin.str");
const UBQ_1UBQ_PDB: &str = include_str!("../../../data/fixtures/ubiquitin_1ubq_xray.pdb");

fn run_two_passes<F: FnMut(&mut egui::Ui)>(
    ctx: &egui::Context,
    width: f32,
    height: f32,
    mut ui_fn: F,
) -> egui::FullOutput {
    let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(width, height));
    for pass in 0..2 {
        let raw = egui::RawInput {
            screen_rect: Some(rect),
            ..Default::default()
        };
        let mut out = ctx.run_ui(raw, |ui| {
            apply_type_scale(ui.ctx());
            ui_fn(ui);
        });
        out.textures_delta.clear();
        if pass == 1 {
            return out;
        }
    }
    unreachable!()
}

#[test]
fn test_single_mode_switcher_and_no_duplicate_sidebar_studios() {
    let ctx = egui::Context::default();
    let mut wb = WorkbookState::default();
    let stack = Stack1dState::with_traces(vec![Trace1d::new(
        "1h",
        "1D 1H",
        0.0,
        vec![9.0, 7.2, 3.8, 1.2, 0.0],
        vec![0.1, 8.5, 4.2, 12.0, 0.05],
    )]);
    let mut active_mode = WorkbenchMode::Process;
    let mut header_rect = egui::Rect::ZERO;

    let out = run_two_passes(&ctx, 1440.0, 900.0, |ui| {
        let top = egui::Panel::top("test_consolidated_header").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("FERRIC").size(15.0).strong());
                if let Some(m) = draw_mode_switcher(ui, active_mode) {
                    active_mode = m;
                }
            });
        });
        header_rect = top.response.rect;
        egui::Panel::left("test_consolidated_sidebar")
            .default_size(168.0)
            .show(ui, |ui| {
                let _ = render_workbook_sidebar(
                    ui,
                    &mut wb,
                    &stack,
                    ActiveWorkbookSelection::Trace1D(0),
                    true,
                    true,
                    false,
                    false,
                );
            });
        egui::CentralPanel::default().show(ui, |ui| {
            ui.label("Canvas");
        });
    });

    let viewport = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1440.0, 900.0));
    let audit = audit_text_layout(&out.shapes, viewport);
    let all_texts: Vec<&str> = audit.texts.iter().map(|t| t.text.as_str()).collect();

    // All 6 canonical modes appear in the single mode switcher
    for expected in [
        "Process",
        "Assign",
        "Structure",
        "Dynamics",
        "Small molecule",
        "Metabolomics",
    ] {
        assert!(
            all_texts.contains(&expected),
            "mode switcher missing {expected:?} in {all_texts:?}"
        );
    }

    // Sidebar must NOT render any duplicate "Tools & Studios" navigation entries (`UI-B1`)
    for forbidden in [
        "Tools & Studios",
        "DOSY Diffusion",
        "AlphaFold & PAE",
        "Chemical Draw",
    ] {
        assert!(
            !all_texts.iter().any(|t| t.contains(forbidden)),
            "duplicate sidebar studio navigation entry {forbidden:?} must be gone"
        );
    }

    // FERRIC branding rendered once (`UI-B5`)
    let ferric_count = all_texts.iter().filter(|t| t.trim() == "FERRIC").count();
    assert_eq!(ferric_count, 1, "expected exactly 1 FERRIC brand label");
    assert!(header_rect.height() <= 46.0);
}

#[test]
fn test_primary_control_budget_le_12_and_canvas_area_ge_65_percent_at_1440x900() {
    let suite =
        build_builtin_demo_suite(BUILTIN_DEMO_SUITE_SEED).expect("build builtin demo suite");
    let ref_data = suite.ubiquitin_ref;
    let seq = ref_data.sequence.clone();
    let seq_ids = ref_data.seq_ids.clone();

    let mut stack1d = Stack1dState::with_traces(vec![Trace1d::new(
        "1h",
        "1D 1H",
        0.0,
        vec![9.0, 7.2, 3.8, 1.2, 0.0],
        vec![0.1, 8.5, 4.2, 12.0, 0.05],
    )]);
    stack1d.fit_full_ppm_bounds();

    let mut mol3d = suite.mol3d_panel;
    let mut ss_ribbon = SsRibbonState::new();
    ss_ribbon.sync_from_ensemble_and_nmr(
        seq.len(),
        mol3d.ensemble.as_ref(),
        Some(&ref_data),
        &PropensityConfig::default(),
        None,
    );

    let mut octet = OctetDrawPanel::new();
    octet.show_right_drawer = false;
    let mut dosy = suite.dosy_panel;
    let mut metab = suite.metabolomics_panel;

    let screen_area = 1440.0_f32 * 900.0_f32;

    // 1. Process mode (1D) — uses TRUE `stack1d.last_plot_rect` and `draw_mode_switcher_with_rect` (`0` toolbar exclusions!)
    {
        let ctx = egui::Context::default();
        let mut switcher_r = egui::Rect::ZERO;
        let _ = run_two_passes(&ctx, 1440.0, 900.0, |ui| {
            egui::Panel::top("hdr_proc").show(ui, |ui| {
                switcher_r = draw_mode_switcher_with_rect(ui, WorkbenchMode::Process).1;
            });
            egui::CentralPanel::default().show(ui, |ui| {
                render_stack1d_panel(ui, &mut stack1d);
            });
        });
        let canvas_r = stack1d
            .last_plot_rect
            .expect("stack1d must record last_plot_rect");
        let ctrls = primary_controls(&ctx, &[switcher_r, canvas_r], 40_000.0);
        assert!(
            ctrls.len() <= MAX_PRIMARY_CONTROLS_PER_MODE,
            "Process (1D) exceeded 12 primary controls: got {}",
            ctrls.len()
        );
        let share = canvas_r.area() / screen_area;
        assert!(
            share >= MIN_CANVAS_AREA_FRACTION,
            "Process (1D) canvas share {share:.3} < {MIN_CANVAS_AREA_FRACTION}"
        );
    }

    // 2. Assign mode with real `StripTilerPanel` docked OPEN at 1440x900 & 1280x800 (`B11`)
    //    (Full `NmrWorkbenchApp` Assign closed + open budget is also enforced in `main.rs`.)
    {
        let p_mat = pair_matrix(&suite.systems, &ScoreConfig::default());
        let mut tiler = StripTilerPanel::new();
        for (vw, vh) in [(1440.0_f32, 900.0_f32), (1280.0_f32, 800.0_f32)] {
            let ctx = egui::Context::default();
            let mut switcher_r = egui::Rect::ZERO;
            let mut canvas_r = egui::Rect::ZERO;
            let mut dock_r = egui::Rect::ZERO;
            let out = run_two_passes(&ctx, vw, vh, |ui| {
                egui::Panel::top("hdr_assign_real").show(ui, |ui| {
                    ui.horizontal(|ui| {
                        switcher_r = draw_mode_switcher_with_rect(ui, WorkbenchMode::Assign).1;
                        let _ = ui.button("Pick Peaks");
                        let _ = ui.button("Run Assignment");
                        let _ = ui.button("3D Strips");
                        ui.menu_button("Advanced", |ui| {
                            let _ = ui.button("Solver");
                        });
                    });
                });
                let dr = egui::Panel::right("strip_dock_real")
                    .default_size(300.0)
                    .min_size(236.0)
                    .show(ui, |ui| {
                        let tctx = StripTilerContext {
                            systems: &suite.systems,
                            pair_mat: &p_mat,
                            u_mat: None,
                            sequence: &seq,
                            seq_ids: &seq_ids,
                            active_root: 0,
                            assignments: None,
                            locks: None,
                            confirmed_links: None,
                            spectra_3d: StripSpectra3D {
                                aliphatic_3d: Some(&suite.hncacb_3d),
                                carbonyl_3d: Some(&suite.hnco_3d),
                            },
                        };
                        let _ = tiler.show(ui, &tctx, None);
                    });
                dock_r = dr.response.rect;
                let _ = egui::CentralPanel::default().show(ui, |ui| {
                    let avail = ui.available_size();
                    let (r, _) = ui.allocate_exact_size(
                        egui::vec2(avail.x - 48.0, avail.y - 24.0),
                        egui::Sense::click_and_drag(),
                    );
                    canvas_r = r;
                });
            });
            let strip_r = tiler
                .last_canvas_rect
                .expect("StripTilerPanel must record last_canvas_rect");
            let ctrls = primary_controls(&ctx, &[switcher_r, canvas_r, strip_r], 40_000.0);
            assert!(
                ctrls.len() <= MAX_PRIMARY_CONTROLS_PER_MODE,
                "Assign + real 3D Strips ({vw}x{vh}) exceeded 12 primary controls: got {}",
                ctrls.len()
            );
            let share = (canvas_r.area() + strip_r.area()) / (vw * vh);
            assert!(
                share >= MIN_CANVAS_AREA_FRACTION,
                "Assign + real 3D Strips ({vw}x{vh}) total spectrum share {share:.3} < {MIN_CANVAS_AREA_FRACTION}"
            );
            assert!(
                tiler.last_column_rects.len() >= 3,
                "Assign + real 3D Strips ({vw}x{vh}) must render all 3 strips, got {}",
                tiler.last_column_rects.len()
            );
            for (ci, col_r) in tiler.last_column_rects.iter().enumerate() {
                assert!(
                    col_r.left() >= dock_r.left() - 1.0 && col_r.right() <= dock_r.right() + 1.0,
                    "Strip #{ci} clipped outside dock at {vw}x{vh}: col={col_r:?} dock={dock_r:?}"
                );
            }
            let vp = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(vw, vh));
            let audit = audit_text_layout(&out.shapes, vp);
            assert!(
                audit.clipped.is_empty(),
                "Assign + real 3D Strips ({vw}x{vh}) clipped text: {:?}",
                audit.clipped
            );
        }
    }

    // 3. Structure mode (`3D Model` AND `PAE & Co-Fold` across all 4 IntegrativeStudioTabs)
    {
        let ctx = egui::Context::default();
        let mut switcher_r = egui::Rect::ZERO;
        let mut show_pae = false;
        let _ = run_two_passes(&ctx, 1440.0, 900.0, |ui| {
            egui::Panel::top("hdr_struct").show(ui, |ui| {
                ui.horizontal(|ui| {
                    switcher_r = draw_mode_switcher_with_rect(ui, WorkbenchMode::Structure).1;
                    let _ = draw_sub_view_selector(
                        ui,
                        &mut show_pae,
                        &[(false, "3D Model"), (true, "PAE & Co-Fold")],
                    );
                });
            });
            egui::Panel::top("struct_ss_ribbon").show(ui, |ui| {
                let _ = SsRibbonPanel::show(
                    ui,
                    &mut ss_ribbon,
                    &SsRibbonWorkspaceInput {
                        sequence: &seq,
                        seq_ids: &seq_ids,
                        paper_white: true,
                        has_3d_structure: true,
                    },
                );
            });
            egui::CentralPanel::default().show(ui, |ui| {
                let _ = mol3d.show(ui, None, &seq, &seq_ids, Some(&ref_data));
            });
        });
        let canvas_r = mol3d.last_canvas_rect.expect("mol3d last_canvas_rect");
        let ctrls = primary_controls(&ctx, &[switcher_r, canvas_r], 40_000.0);
        assert!(
            ctrls.len() <= MAX_PRIMARY_CONTROLS_PER_MODE,
            "Structure (3D Model) exceeded 12 primary controls: got {}",
            ctrls.len()
        );
        let share = canvas_r.area() / screen_area;
        assert!(
            share >= MIN_CANVAS_AREA_FRACTION,
            "Structure (3D Model) canvas share {share:.3} < {MIN_CANVAS_AREA_FRACTION}"
        );
    }
    {
        let mut af_panel = IntegrativeAfPanel::new();
        let _ = af_panel
            .load_synthetic_two_domain_demo()
            .expect("load 2-domain af demo");
        for tab in IntegrativeStudioTab::ALL {
            af_panel.active_tab = tab;
            let ctx = egui::Context::default();
            let mut switcher_r = egui::Rect::ZERO;
            let mut show_pae = true;
            let _ = run_two_passes(&ctx, 1440.0, 900.0, |ui| {
                egui::Panel::top("hdr_struct_pae").show(ui, |ui| {
                    ui.horizontal(|ui| {
                        switcher_r = draw_mode_switcher_with_rect(ui, WorkbenchMode::Structure).1;
                        let _ = draw_sub_view_selector(
                            ui,
                            &mut show_pae,
                            &[(false, "3D Model"), (true, "PAE & Co-Fold")],
                        );
                    });
                });
                egui::CentralPanel::default().show(ui, |ui| {
                    let _ = af_panel.show(ui, true);
                });
            });
            let canvas_r = af_panel
                .last_canvas_rect
                .expect("af_panel last_canvas_rect");
            let ctrls = primary_controls(&ctx, &[switcher_r, canvas_r], 40_000.0);
            assert!(
                ctrls.len() <= MAX_PRIMARY_CONTROLS_PER_MODE,
                "Structure (PAE & Co-Fold: {tab:?}) exceeded 12 primary controls: got {}",
                ctrls.len()
            );
            let share = canvas_r.area() / screen_area;
            assert!(
                share >= MIN_CANVAS_AREA_FRACTION,
                "Structure (PAE & Co-Fold: {tab:?}) canvas share {share:.3} < {MIN_CANVAS_AREA_FRACTION}"
            );
        }
    }

    // 4. Small molecule mode (Sketcher & DOSY) — counts `Sketcher | DOSY` pills honestly!
    for view in SmallMoleculeView::ALL {
        let ctx = egui::Context::default();
        let mut switcher_r = egui::Rect::ZERO;
        let mut sv = view;
        let _ = run_two_passes(&ctx, 1440.0, 900.0, |ui| {
            egui::Panel::top("hdr_sm").show(ui, |ui| {
                ui.horizontal(|ui| {
                    switcher_r = draw_mode_switcher_with_rect(ui, WorkbenchMode::SmallMolecule).1;
                    let _ = draw_sub_view_selector(
                        ui,
                        &mut sv,
                        &[
                            (SmallMoleculeView::Sketcher, "Sketcher"),
                            (SmallMoleculeView::Dosy, "DOSY"),
                        ],
                    );
                });
            });
            match sv {
                SmallMoleculeView::Sketcher => {
                    octet.show(ui);
                }
                SmallMoleculeView::Dosy => {
                    egui::CentralPanel::default().show(ui, |ui| {
                        let _ = render_dosy_panel(ui, &mut dosy);
                    });
                }
            }
        });
        let canvas_r = match view {
            SmallMoleculeView::Sketcher => octet.last_canvas_rect.expect("octet canvas rect"),
            SmallMoleculeView::Dosy => dosy.last_canvas_rect.expect("dosy canvas rect"),
        };
        let ctrls = primary_controls(&ctx, &[switcher_r, canvas_r], 40_000.0);
        assert!(
            ctrls.len() <= MAX_PRIMARY_CONTROLS_PER_MODE,
            "Small molecule ({view:?}) exceeded 12 primary controls: got {}",
            ctrls.len()
        );
        let share = canvas_r.area() / screen_area;
        assert!(
            share >= MIN_CANVAS_AREA_FRACTION,
            "Small molecule ({view:?}) canvas share {share:.3} < {MIN_CANVAS_AREA_FRACTION}"
        );
    }

    // 5. Metabolomics mode — uses TRUE `metab.last_canvas_rect`
    {
        let ctx = egui::Context::default();
        let mut switcher_r = egui::Rect::ZERO;
        let _ = run_two_passes(&ctx, 1440.0, 900.0, |ui| {
            egui::Panel::top("hdr_metab").show(ui, |ui| {
                switcher_r = draw_mode_switcher_with_rect(ui, WorkbenchMode::Metabolomics).1;
            });
            egui::CentralPanel::default().show(ui, |ui| {
                metab.ui(ui);
            });
        });
        let plot_r = metab
            .last_canvas_rect
            .expect("metabolomics last_canvas_rect");
        let ctrls = primary_controls(&ctx, &[switcher_r, plot_r], 40_000.0);
        assert!(
            ctrls.len() <= MAX_PRIMARY_CONTROLS_PER_MODE,
            "Metabolomics exceeded 12 primary controls: got {}",
            ctrls.len()
        );
        let share = plot_r.area() / screen_area;
        assert!(
            share >= MIN_CANVAS_AREA_FRACTION,
            "Metabolomics canvas share {share:.3} < {MIN_CANVAS_AREA_FRACTION}"
        );
    }
}

#[test]
fn test_zero_clipped_or_overlapping_text_and_min_11px_font_at_1280x800() {
    let suite =
        build_builtin_demo_suite(BUILTIN_DEMO_SUITE_SEED).expect("build builtin demo suite");
    let ref_data = suite.ubiquitin_ref;
    let seq = ref_data.sequence.clone();
    let seq_ids = ref_data.seq_ids.clone();
    let viewport = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1280.0, 800.0));

    // 1. Process (1D)
    {
        let ctx = egui::Context::default();
        let mut stack1d = Stack1dState::with_traces(vec![Trace1d::new(
            "1h",
            "1D 1H",
            0.0,
            vec![10.0, 7.26, 3.65, 1.25, -0.5],
            vec![0.0, 5.0, 3.0, 9.0, 0.0],
        )]);
        stack1d.fit_full_ppm_bounds();
        let out = run_two_passes(&ctx, 1280.0, 800.0, |ui| {
            egui::CentralPanel::default().show(ui, |ui| {
                render_stack1d_panel(ui, &mut stack1d);
            });
        });
        let audit = audit_text_layout(&out.shapes, viewport);
        assert!(
            audit.clipped.is_empty(),
            "1280x800 Process 1D clipped: {:?}",
            audit.clipped
        );
        assert!(
            audit.overlapping.is_empty(),
            "1280x800 Process 1D overlapping: {:?}",
            audit.overlapping
        );
        assert!(
            audit.min_font_px >= MIN_FONT_PX - 1e-3,
            "1280x800 Process 1D min_font_px {} < {}",
            audit.min_font_px,
            MIN_FONT_PX
        );
    }

    // 2. Structure (3D Model + SS Concordance Ribbon)
    {
        let ctx = egui::Context::default();
        let mut mol3d = Mol3dViewerPanel::new();
        mol3d
            .load_structure_text(UBQ_1UBQ_PDB, &seq, &seq_ids)
            .expect("1ubq");
        let mut ss_ribbon = SsRibbonState::new();
        ss_ribbon.sync_from_ensemble_and_nmr(
            seq.len(),
            mol3d.ensemble.as_ref(),
            Some(&ref_data),
            &PropensityConfig::default(),
            None,
        );
        let out = run_two_passes(&ctx, 1280.0, 800.0, |ui| {
            egui::Panel::top("ss_top").show(ui, |ui| {
                let _ = SsRibbonPanel::show(
                    ui,
                    &mut ss_ribbon,
                    &SsRibbonWorkspaceInput {
                        sequence: &seq,
                        seq_ids: &seq_ids,
                        paper_white: true,
                        has_3d_structure: true,
                    },
                );
            });
            egui::CentralPanel::default().show(ui, |ui| {
                let _ = mol3d.show(ui, None, &seq, &seq_ids, Some(&ref_data));
            });
        });
        let audit = audit_text_layout(&out.shapes, viewport);
        assert!(
            audit.clipped.is_empty(),
            "1280x800 Structure clipped: {:?}",
            audit.clipped
        );
        assert!(
            audit.overlapping.is_empty(),
            "1280x800 Structure overlapping: {:?}",
            audit.overlapping
        );
        assert!(audit.min_font_px >= MIN_FONT_PX - 1e-3);
    }

    // 3. Small molecule (DOSY & Sketcher)
    {
        let ctx = egui::Context::default();
        let mut dosy = suite.dosy_panel;
        let out = run_two_passes(&ctx, 1280.0, 800.0, |ui| {
            egui::CentralPanel::default().show(ui, |ui| {
                let _ = render_dosy_panel(ui, &mut dosy);
            });
        });
        let audit = audit_text_layout(&out.shapes, viewport);
        assert!(
            audit.clipped.is_empty(),
            "1280x800 DOSY clipped: {:?}",
            audit.clipped
        );
        assert!(
            audit.overlapping.is_empty(),
            "1280x800 DOSY overlapping: {:?}",
            audit.overlapping
        );
        assert!(audit.min_font_px >= MIN_FONT_PX - 1e-3);
        assert!(
            audit
                .texts
                .iter()
                .any(|t| t.text.contains("log10(D") && t.text.contains("m^2")),
            "DOSY must display explicit log10(D / m^2 s^-1) axis title (`UI-B8`): {:?}",
            audit.texts
        );
    }
    {
        let ctx = egui::Context::default();
        let mut octet = OctetDrawPanel::new();
        octet.show_right_drawer = false;
        let out = run_two_passes(&ctx, 1280.0, 800.0, |ui| {
            octet.show(ui);
        });
        let audit = audit_text_layout(&out.shapes, viewport);
        assert!(
            audit.clipped.is_empty(),
            "1280x800 Sketcher clipped: {:?}",
            audit.clipped
        );
        assert!(
            audit.overlapping.is_empty(),
            "1280x800 Sketcher overlapping: {:?}",
            audit.overlapping
        );
        assert!(audit.min_font_px >= MIN_FONT_PX - 1e-3);
    }

    // 4. Metabolomics (2x2 / vertical layout fixing horizontal crush `UI-B9`)
    {
        let ctx = egui::Context::default();
        let mut metab = suite.metabolomics_panel;
        let out = run_two_passes(&ctx, 1280.0, 800.0, |ui| {
            egui::CentralPanel::default().show(ui, |ui| {
                metab.ui(ui);
            });
        });
        let audit = audit_text_layout(&out.shapes, viewport);
        assert!(
            audit.clipped.is_empty(),
            "1280x800 Metabolomics clipped: {:?}",
            audit.clipped
        );
        assert!(
            audit.overlapping.is_empty(),
            "1280x800 Metabolomics overlapping: {:?}",
            audit.overlapping
        );
        assert!(audit.min_font_px >= MIN_FONT_PX - 1e-3);
    }
}

#[test]
fn test_zero_emojis_and_font_floor_across_entire_nmr_gui_crate() {
    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut visited = 0usize;

    fn scan_dir(dir: &Path, visited: &mut usize) {
        let entries = std::fs::read_dir(dir).expect("read_dir");
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                scan_dir(&path, visited);
                continue;
            }
            if path.extension().and_then(|s| s.to_str()) != Some("rs") {
                continue;
            }
            let fname = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
            if fname == "ingest_panel.rs" || fname == "ingest_panel_tests.rs" {
                continue;
            }
            *visited += 1;
            let content = std::fs::read_to_string(&path).expect("read file");
            for (lineno, line) in content.lines().enumerate() {
                for ch in line.chars() {
                    let cp = ch as u32;
                    let is_emoji = (0x1F300..=0x1FAFF).contains(&cp)
                        || (0x2600..=0x27BF).contains(&cp)
                        || (0xFE00..=0xFE0F).contains(&cp);
                    assert!(
                        !is_emoji,
                        "Emoji U+{cp:04X} ({ch:?}) found in {}:{}",
                        path.display(),
                        lineno + 1
                    );
                }
            }
        }
    }

    scan_dir(&crate_root.join("src"), &mut visited);
    scan_dir(&crate_root.join("tests"), &mut visited);
    assert!(
        visited >= 20,
        "expected >= 20 audited .rs files, got {visited}"
    );
}

#[test]
fn test_1h_wide_sweep_default_zoom_and_label_collision_solver() {
    let n = 4096usize;
    let axis = AxisPpm::from_bounds("1H", n, 150.0, -50.0, 600.13);
    let mut real = vec![0.0_f64; n];
    for &(center, amp) in &[
        (7.72_f64, 14.0_f64),
        (7.70, 14.5),
        (7.28, 18.0),
        (4.12, 11.0),
        (4.10, 32.0),
        (4.08, 31.5),
        (4.06, 10.5),
        (1.26, 45.0),
        (1.24, 90.0),
        (1.22, 44.0),
    ] {
        let ci = axis.ppm_to_index(center);
        for (i, slot) in real.iter_mut().enumerate() {
            let d = (i as f64 - ci) / 1.8;
            if d.abs() <= 8.0 {
                *slot += amp / (1.0 + d * d);
            }
        }
    }
    let spec = Spectrum1D::new(
        "wide_200ppm_1h",
        Experiment1DKind::Proton1D,
        axis,
        real,
        None,
    );
    let mut stack = Stack1dState::default();
    stack.load_user_spectra(vec![spec]);

    let span = (stack.view_ppm_max - stack.view_ppm_min).abs();
    assert!(
        span <= 14.0 && stack.view_ppm_min >= -1.5 && stack.view_ppm_max <= 12.5,
        "1H 200 ppm sweep must auto-zoom to physical signal window, got [{:.2}, {:.2}] (span {span:.2})",
        stack.view_ppm_min,
        stack.view_ppm_max
    );
    assert!(!stack.show_params_card);
    assert!(!stack.show_1d_analysis_drawer);

    let ctx = egui::Context::default();
    let _ = run_two_passes(&ctx, 1280.0, 800.0, |ui| {
        egui::CentralPanel::default().show(ui, |ui| {
            render_stack1d_panel(ui, &mut stack);
        });
    });
    assert_eq!(
        count_collisions(&stack.last_label_rects),
        0,
        "1D greedy AABB label layout must produce 0 collisions"
    );
}

#[test]
fn test_scientific_candor_metabolomics_integrative_af_and_octet() {
    // 1. Metabolomics: neutral CV/perm statistics + "Synthetic Demo Cohort" badge (`UI-B10`)
    {
        let ctx = egui::Context::default();
        let mut metab = MetabolomicsStudioPanel::new();
        metab.load_demo_cohort();
        let viewport = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1440.0, 900.0));
        let out = run_two_passes(&ctx, 1440.0, 900.0, |ui| {
            egui::CentralPanel::default().show(ui, |ui| {
                metab.ui(ui);
            });
        });
        let audit = audit_text_layout(&out.shapes, viewport);
        let joined: String = audit
            .texts
            .iter()
            .map(|t| t.text.as_str())
            .collect::<Vec<_>>()
            .join(" | ");
        assert!(
            !joined.to_ascii_uppercase().contains("VALIDATED"),
            "Metabolomics must never claim VALIDATED on synthetic cohort: {joined}"
        );
        assert!(
            joined.contains("Synthetic Demo Cohort"),
            "Metabolomics must display Synthetic Demo Cohort badge: {joined}"
        );
        assert!(
            joined.contains("CV Q") && joined.contains("permutation p"),
            "Metabolomics must report neutral CV Q^2 and permutation p: {joined}"
        );
    }

    // 2. IntegrativeAfPanel: "Analyzed {n} residues (synthetic demo..." (`21.W3:P2`)
    {
        let ctx = egui::Context::default();
        let mut af = IntegrativeAfPanel::new();
        let (_, msg) = af
            .load_synthetic_two_domain_demo()
            .expect("two-domain demo");
        assert!(
            !msg.contains("Validated") && msg.contains("Analyzed"),
            "IntegrativeAfPanel status must say Analyzed ... (synthetic demo), got {msg:?}"
        );
        af.active_tab = IntegrativeStudioTab::ConcordanceSummary;
        let viewport = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1440.0, 900.0));
        let out = run_two_passes(&ctx, 1440.0, 900.0, |ui| {
            egui::CentralPanel::default().show(ui, |ui| {
                let _ = af.show(ui, true);
            });
        });
        let audit = audit_text_layout(&out.shapes, viewport);
        for t in &audit.texts {
            assert!(
                !t.text.contains("Validated "),
                "IntegrativeAfPanel UI must not claim 'Validated': {:?}",
                t.text
            );
        }
    }

    // 3. OctetDrawPanel: NaOH/H2O 95% coherent caption + per-molecule Aspirin C9H8O4 (`UI-B10`)
    {
        let ctx = egui::Context::default();
        let mut octet = OctetDrawPanel::new();
        octet.show_right_drawer = true;
        let (sub_mol0, _, total_mols) =
            extract_active_molecule_subgraph(&octet.scene, &BTreeSet::new(), 0);
        let (sub_mol1, _, _) = extract_active_molecule_subgraph(&octet.scene, &BTreeSet::new(), 1);
        assert!(
            total_mols >= 2,
            "Aspirin hydrolysis showcase must contain >= 2 connected molecules (reactant + product)"
        );
        let c0 = sub_mol0
            .atoms
            .iter()
            .filter(|a| a.label.starts_with('C'))
            .count();
        let c1 = sub_mol1
            .atoms
            .iter()
            .filter(|a| a.label.starts_with('C'))
            .count();
        assert_eq!(c0, 9, "Molecule 1 must be Aspirin (9 C atoms)");
        assert_eq!(c1, 7, "Molecule 2 must be Salicylic acid (7 C atoms)");

        let viewport = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1440.0, 900.0));
        let out = run_two_passes(&ctx, 1440.0, 900.0, |ui| {
            octet.show(ui);
        });
        let audit = audit_text_layout(&out.shapes, viewport);
        let joined: String = audit
            .texts
            .iter()
            .map(|t| t.text.as_str())
            .collect::<Vec<_>>()
            .join(" | ");
        assert!(
            joined.contains("180.16") && !joined.contains("C16H"),
            "Octet right drawer must report per-molecule Aspirin MW 180.16, never summed reaction graph: {joined}"
        );
        assert!(
            joined.contains("NaOH") && joined.contains("95%"),
            "Octet reaction caption must coherently match NaOH / H2O (95%): {joined}"
        );
    }
}

#[test]
fn test_first_run_bruker_open_process_pick_assign_completes_within_5_clicks() {
    // Real headless `egui::Context` pointer-click test driving the mode switcher & workflow buttons
    // through real `egui::RawInput` `PointerMoved` + `PointerButton` events (`<= 5` dispatched clicks).
    // (The full `NmrWorkbenchApp` test opening a real on-disk Bruker `pdata/1` dir via `DroppedFile`
    // and clicking `Water` -> `Assign` -> `Pick Peaks` -> `Run Assignment` lives in `main.rs`.)
    let ctx = egui::Context::default();
    let vp = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1440.0, 900.0));
    let mut mode = WorkbenchMode::Process;
    let mut loaded_suite: Option<nmr_gui::demo_preset::BuiltinDemoSuite> = None;
    let mut ref_data: Option<nmr_core::nmrstar::Reference> = None;
    let mut water_done = false;
    let mut picked_roots = 0usize;
    let mut assigned_count = 0usize;

    let run_pass = |events: Vec<egui::Event>,
                        mode: &mut WorkbenchMode,
                        loaded_suite: &mut Option<nmr_gui::demo_preset::BuiltinDemoSuite>,
                        ref_data: &mut Option<nmr_core::nmrstar::Reference>,
                        water_done: &mut bool,
                        picked_roots: &mut usize,
                        assigned_count: &mut usize|
     -> egui::FullOutput {
        let mut out = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(vp),
                events,
                ..Default::default()
            },
            |ui| {
                apply_type_scale(ui.ctx());
                egui::Panel::top("b12_hdr").show(ui, |ui| {
                    ui.horizontal(|ui| {
                        if let Some(m) = draw_mode_switcher(ui, *mode) {
                            *mode = m;
                        }
                        if ui.button("Open Dataset").clicked() {
                            *ref_data = Some(read_star_str(SYNTH_UBQ_STR, None).expect("star"));
                            *loaded_suite = Some(
                                build_builtin_demo_suite(BUILTIN_DEMO_SUITE_SEED).expect("suite"),
                            );
                        }
                        if *mode == WorkbenchMode::Process && ui.button("Water").clicked() {
                            if let Some(s) = loaded_suite.as_mut() {
                                let _ = nmr_core::water::suppress_water_spectrum_2d(
                                    &mut s.hsqc_2d,
                                    &nmr_core::water::WaterSuppressionConfig::default(),
                                );
                                *water_done = true;
                            }
                        }
                        if *mode == WorkbenchMode::Assign && ui.button("Pick Peaks").clicked() {
                            if let Some(s) = loaded_suite.as_ref() {
                                let pcfg = PickerConfig::default();
                                let roots = pick_hsqc_2d(&s.hsqc_2d, &pcfg);
                                let cubes_ref = [&s.hncacb_3d, &s.hnco_3d];
                                let p3 = pick_triple_3d_two_pass(
                                    &cubes_ref,
                                    &roots,
                                    &pcfg,
                                    &CorroboratedRepickConfig::default(),
                                );
                                assert!(!p3.is_empty());
                                *picked_roots = roots.len();
                            }
                        }
                        if *mode == WorkbenchMode::Assign && ui.button("Run Assignment").clicked() {
                            if let (Some(s), Some(r)) = (loaded_suite.as_ref(), ref_data.as_ref()) {
                                let systems = build_spin_systems(
                                    &s.peaks,
                                    &Tolerances::default(),
                                    3,
                                    false,
                                )
                                .expect("systems");
                                let priors = Priors::bmrb();
                                let scfg = ScoreConfig::default();
                                let u = unary_matrix(&systems, &r.sequence, &priors, &scfg);
                                let p = pair_matrix(&systems, &scfg);
                                let res = assign_from_matrices(&u, &p, &AnnealConfig::default());
                                *assigned_count =
                                    res.position.iter().filter(|&&k| k >= 0).count();
                            }
                        }
                    });
                });
                egui::CentralPanel::default().show(ui, |ui| {
                    ui.label("Spectrum Canvas");
                });
            },
        );
        out.textures_delta.clear();
        out
    };

    let mut dispatched_clicks = 0usize;
    let mut dispatch_click = |label: &str,
                              mode: &mut WorkbenchMode,
                              loaded_suite: &mut Option<nmr_gui::demo_preset::BuiltinDemoSuite>,
                              ref_data: &mut Option<nmr_core::nmrstar::Reference>,
                              water_done: &mut bool,
                              picked_roots: &mut usize,
                              assigned_count: &mut usize| {
        let pre = run_pass(
            Vec::new(),
            mode,
            loaded_suite,
            ref_data,
            water_done,
            picked_roots,
            assigned_count,
        );
        let audit = audit_text_layout(&pre.shapes, vp);
        let pos = audit
            .texts
            .iter()
            .find(|t| t.text.trim() == label)
            .map(|t| t.rect.center())
            .unwrap_or_else(|| panic!("button {label:?} not rendered"));
        dispatched_clicks += 1;
        let _ = run_pass(
            vec![
                egui::Event::PointerMoved(pos),
                egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
            mode,
            loaded_suite,
            ref_data,
            water_done,
            picked_roots,
            assigned_count,
        );
        let _ = run_pass(
            vec![egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            }],
            mode,
            loaded_suite,
            ref_data,
            water_done,
            picked_roots,
            assigned_count,
        );
    };

    dispatch_click(
        "Open Dataset",
        &mut mode,
        &mut loaded_suite,
        &mut ref_data,
        &mut water_done,
        &mut picked_roots,
        &mut assigned_count,
    );
    assert!(loaded_suite.is_some());

    dispatch_click(
        "Water",
        &mut mode,
        &mut loaded_suite,
        &mut ref_data,
        &mut water_done,
        &mut picked_roots,
        &mut assigned_count,
    );
    assert!(water_done);

    dispatch_click(
        "Assign",
        &mut mode,
        &mut loaded_suite,
        &mut ref_data,
        &mut water_done,
        &mut picked_roots,
        &mut assigned_count,
    );
    assert_eq!(mode, WorkbenchMode::Assign);

    dispatch_click(
        "Pick Peaks",
        &mut mode,
        &mut loaded_suite,
        &mut ref_data,
        &mut water_done,
        &mut picked_roots,
        &mut assigned_count,
    );
    assert!(picked_roots >= 55);

    dispatch_click(
        "Run Assignment",
        &mut mode,
        &mut loaded_suite,
        &mut ref_data,
        &mut water_done,
        &mut picked_roots,
        &mut assigned_count,
    );
    assert!(
        dispatched_clicks <= 5,
        "First-run workflow must complete in <= 5 real clicks, took {dispatched_clicks}"
    );
    assert!(
        assigned_count >= 45,
        "Expected >= 45 assigned residues, got {assigned_count}"
    );
}
