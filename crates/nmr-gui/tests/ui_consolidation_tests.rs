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
use nmr_gui::workbench_mode::{
    apply_type_scale, audit_text_layout, draw_mode_switcher, draw_sub_view_selector,
    primary_controls, SmallMoleculeView, WorkbenchMode, MAX_PRIMARY_CONTROLS_PER_MODE,
    MIN_CANVAS_AREA_FRACTION, MIN_FONT_PX,
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

    // 1. Process mode (1D)
    {
        let ctx = egui::Context::default();
        let mut switcher_r = egui::Rect::ZERO;
        let mut canvas_r = egui::Rect::ZERO;
        let _ = run_two_passes(&ctx, 1440.0, 900.0, |ui| {
            let top = egui::Panel::top("hdr_proc").show(ui, |ui| {
                let _ = draw_mode_switcher(ui, WorkbenchMode::Process);
            });
            switcher_r = top.response.rect;
            let c = egui::CentralPanel::default().show(ui, |ui| {
                render_stack1d_panel(ui, &mut stack1d);
            });
            canvas_r = egui::Rect::from_min_max(
                egui::pos2(c.response.rect.left(), c.response.rect.top() + 36.0),
                c.response.rect.max,
            );
        });
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

    // 2. Assign mode (2D HSQC)
    {
        let ctx = egui::Context::default();
        let mut switcher_r = egui::Rect::ZERO;
        let mut canvas_r = egui::Rect::ZERO;
        let _ = run_two_passes(&ctx, 1440.0, 900.0, |ui| {
            let top = egui::Panel::top("hdr_assign").show(ui, |ui| {
                ui.horizontal(|ui| {
                    let _ = draw_mode_switcher(ui, WorkbenchMode::Assign);
                    let _ = ui.button("Pick Peaks");
                    let _ = ui.button("Run Assignment");
                    let _ = ui.button("3D Strips");
                    ui.menu_button("Advanced", |ui| {
                        let _ = ui.button("Solver");
                    });
                });
            });
            switcher_r = egui::Rect::from_min_size(top.response.rect.min, egui::vec2(520.0, 36.0));
            egui::Panel::top("seq_strip_assign").show(ui, |ui| {
                let mut q = String::new();
                ui.horizontal(|ui| {
                    ui.add(egui::TextEdit::singleline(&mut q).desired_width(96.0));
                    let _ = ui.allocate_exact_size(egui::vec2(1200.0, 24.0), egui::Sense::click());
                });
            });
            let _ = egui::CentralPanel::default().show(ui, |ui| {
                let avail = ui.available_size();
                let (r, _) = ui.allocate_exact_size(
                    egui::vec2(avail.x - 72.0, avail.y - 24.0),
                    egui::Sense::click_and_drag(),
                );
                canvas_r = r;
            });
        });
        let ctrls = primary_controls(&ctx, &[switcher_r, canvas_r], 30_000.0);
        assert!(
            ctrls.len() <= MAX_PRIMARY_CONTROLS_PER_MODE,
            "Assign mode exceeded 12 primary controls: got {}",
            ctrls.len()
        );
        let share = canvas_r.area() / screen_area;
        assert!(
            share >= MIN_CANVAS_AREA_FRACTION,
            "Assign mode 2D canvas share {share:.3} < {MIN_CANVAS_AREA_FRACTION}"
        );
    }

    // 3. Structure mode (Concordance Ribbon + 3D Model)
    {
        let ctx = egui::Context::default();
        let mut switcher_r = egui::Rect::ZERO;
        let mut show_pae = false;
        let _ = run_two_passes(&ctx, 1440.0, 900.0, |ui| {
            let top = egui::Panel::top("hdr_struct").show(ui, |ui| {
                ui.horizontal(|ui| {
                    let _ = draw_mode_switcher(ui, WorkbenchMode::Structure);
                    let _ = draw_sub_view_selector(
                        ui,
                        &mut show_pae,
                        &[(false, "3D Model"), (true, "PAE & Co-Fold")],
                    );
                });
            });
            switcher_r = egui::Rect::from_min_size(top.response.rect.min, egui::vec2(520.0, 36.0));
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
            "Structure mode exceeded 12 primary controls: got {}",
            ctrls.len()
        );
        let share = canvas_r.area() / screen_area;
        assert!(
            share >= MIN_CANVAS_AREA_FRACTION,
            "Structure mode 3D canvas share {share:.3} < {MIN_CANVAS_AREA_FRACTION}"
        );
    }

    // 4. Small molecule mode (Sketcher & DOSY)
    for view in SmallMoleculeView::ALL {
        let ctx = egui::Context::default();
        let mut switcher_r = egui::Rect::ZERO;
        let mut sv = view;
        let _ = run_two_passes(&ctx, 1440.0, 900.0, |ui| {
            let top = egui::Panel::top("hdr_sm").show(ui, |ui| {
                ui.horizontal(|ui| {
                    let _ = draw_mode_switcher(ui, WorkbenchMode::SmallMolecule);
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
            switcher_r = top.response.rect;
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

    // 5. Metabolomics mode
    {
        let ctx = egui::Context::default();
        let mut switcher_r = egui::Rect::ZERO;
        let mut plot_r = egui::Rect::ZERO;
        let _ = run_two_passes(&ctx, 1440.0, 900.0, |ui| {
            let top = egui::Panel::top("hdr_metab").show(ui, |ui| {
                let _ = draw_mode_switcher(ui, WorkbenchMode::Metabolomics);
            });
            switcher_r = top.response.rect;
            let c = egui::CentralPanel::default().show(ui, |ui| {
                metab.ui(ui);
            });
            plot_r = egui::Rect::from_min_max(
                egui::pos2(c.response.rect.left(), c.response.rect.top() + 64.0),
                c.response.rect.max,
            );
        });
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
    // Scripted 4-click (`<= 5` clicks!) end-to-end workflow:
    // Click 1: Open dataset -> Click 2: Process / Phase check -> Click 3: Pick Peaks -> Click 4: Run Assignment
    let mut click_count = 0usize;

    // Click 1: Open Ubiquitin 2D HSQC + 3D HNCACB/HNCO workspace
    click_count += 1;
    let ref_data = read_star_str(SYNTH_UBQ_STR, None).expect("parse sequence & shifts");
    let suite =
        build_builtin_demo_suite(BUILTIN_DEMO_SUITE_SEED).expect("build ubiquitin workspace");

    // Click 2: Process (water-safe 2D HSQC ready)
    click_count += 1;
    let mut hsqc = suite.hsqc_2d;
    let cfg_w = nmr_core::water::WaterSuppressionConfig::default();
    let _ = nmr_core::water::suppress_water_spectrum_2d(&mut hsqc, &cfg_w);

    // Click 3: Pick 2D HSQC roots & multi-experiment 3D peaks (from full workspace peak set)
    click_count += 1;
    let pcfg = PickerConfig::default();
    let roots = pick_hsqc_2d(&hsqc, &pcfg);
    assert!(roots.len() >= 55, "expected >= 55 picked 2D HSQC roots");
    let cubes_ref = [&suite.hncacb_3d, &suite.hnco_3d];
    let p3 = pick_triple_3d_two_pass(
        &cubes_ref,
        &roots,
        &pcfg,
        &CorroboratedRepickConfig::default(),
    );
    assert!(!p3.is_empty(), "expected 3D strip peaks picked");
    let systems =
        build_spin_systems(&suite.peaks, &Tolerances::default(), 3, false).expect("build systems");
    assert!(systems.len() >= 50, "expected >= 50 spin systems");

    // Click 4: Run Assignment
    click_count += 1;
    let priors = Priors::bmrb();
    let scfg = ScoreConfig::default();
    let u = unary_matrix(&systems, &ref_data.sequence, &priors, &scfg);
    let p = pair_matrix(&systems, &scfg);
    let res = assign_from_matrices(&u, &p, &AnnealConfig::default());
    let n_assigned = res.position.iter().filter(|&&k| k >= 0).count();

    assert!(
        click_count <= 5,
        "First-run open -> process -> pick -> assign must finish in <= 5 clicks, took {click_count}"
    );
    assert!(
        n_assigned >= 45,
        "Expected >= 45 assigned residues in 4-click workflow, got {n_assigned}"
    );
}
