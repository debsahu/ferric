//! Headless `egui` UX integration tests for:
//! 1. Full `CO` (`HNCO` + `HN(CA)CO`) sequential walk & connected `CA/CB | CO | Both` folder tabs
//!    in `nmr_gui::strip_tiler`.
//! 2. 3-Step `AssignSetupDrawer` (`Sequence`, `Spectra & Peaks`, `Solve & Triage`) in
//!    `nmr_gui::assign_setup_drawer`.
//! 3. GUI Pseudo-3D `T1`/`T2`/`hetNOE` -> `Dynamics` bridge (HSQC peak sync, Bruker pseudo-3D
//!    folder load, Model-Free, and Phase 22 Detectors / IMPACT / TRACT) in `nmr_gui::relax_panel`.

use std::collections::BTreeMap;

use eframe::egui;
use nmr_core::relax_io::{HetNoeOrder, RelaxExperimentKind};
use nmr_core::solver::{pair_matrix, ScoreConfig};
use nmr_core::spectrum::{AxisPpm, Spectrum3D};
use nmr_core::spinsystems::SpinSystem;
use nmr_gui::assign_setup_drawer::{
    parse_inline_sequence, AssignProteinRegime, AssignSetupAction, AssignSetupDrawerState,
    AssignSetupInput, AssignSlotAction, AssignSolverEngine, AssignTriageSummary,
    ExperimentChecklistRow, TriageBucket, CANONICAL_BACKBONE_EXPERIMENTS,
};
use nmr_gui::contour_painter::SyncCrosshair;
use nmr_gui::relax_panel::{Peak2D, RelaxPanelState, RelaxStudioSubTab, TRACT_MISSING_RATES_ERR};
use nmr_gui::strip_tiler::{
    StripBandView, StripSpectra3D, StripSpectra3DFull, StripTilerContext, StripTilerPanel,
};
use nmr_gui::workbench_mode::{apply_type_scale, audit_text_layout, MIN_FONT_PX};

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

fn make_ss(h: f64, n: f64, entries: &[(&str, i32, f64, usize)]) -> SpinSystem {
    let mut ss = SpinSystem::new(h, n);
    for &(atom, off, ppm, sup) in entries {
        ss.cands
            .entry((atom.to_string(), off))
            .or_default()
            .push((ppm, sup));
    }
    ss
}

fn make_carbonyl_cube(name: &str, exp: &str, peaks: &[(f64, f64, f64, f32)]) -> Spectrum3D {
    let nx = 20usize;
    let ny = 16usize;
    let nz = 32usize;
    let axis_x = AxisPpm::from_bounds("1H", nx, 9.2, 7.6, 600.0);
    let axis_y = AxisPpm::from_bounds("15N", ny, 128.0, 112.0, 60.8);
    let axis_z = AxisPpm::from_bounds("13C", nz, 182.5, 167.5, 150.9);
    let mut data = vec![0.0_f32; nx * ny * nz];
    for &(h, n, c, amp) in peaks {
        for iz in 0..nz {
            let cz = axis_z.index_to_ppm(iz as f64);
            let dz = (cz - c) / 0.35;
            if dz.abs() > 3.5 {
                continue;
            }
            for iy in 0..ny {
                let ny_ppm = axis_y.index_to_ppm(iy as f64);
                let dy = (ny_ppm - n) / 0.6;
                if dy.abs() > 3.5 {
                    continue;
                }
                for ix in 0..nx {
                    let hx = axis_x.index_to_ppm(ix as f64);
                    let dx = (hx - h) / 0.06;
                    if dx.abs() > 3.5 {
                        continue;
                    }
                    let val = amp * ((-0.5 * (dx * dx + dy * dy + dz * dz)).exp() as f32);
                    data[(iz * ny + iy) * nx + ix] += val;
                }
            }
        }
    }
    Spectrum3D {
        name: name.to_string(),
        exp: exp.to_string(),
        axis_x,
        axis_y,
        axis_z,
        data,
    }
}

#[test]
fn test_strip_tiler_full_co_walk_folder_tabs_and_hncaco_pairing() {
    // Chain SS0 (C'_i=175.10) -> SS1 (C'_{i-1}=175.08, C'_i=173.80) -> SS2 (C'_{i-1}=173.82, C'_i=171.90)
    let systems = vec![
        make_ss(
            8.30,
            120.0,
            &[
                ("CA", 0, 56.20, 2),
                ("CB", 0, 31.40, 2),
                ("C", 0, 175.10, 2),
            ],
        ),
        make_ss(
            8.10,
            118.4,
            &[
                ("CA", -1, 56.22, 2),
                ("CB", -1, 31.38, 2),
                ("C", -1, 175.08, 2),
                ("CA", 0, 61.10, 2),
                ("CB", 0, 68.50, 2),
                ("C", 0, 173.80, 2),
            ],
        ),
        make_ss(
            8.65,
            123.2,
            &[
                ("CA", -1, 61.08, 2),
                ("CB", -1, 68.52, 2),
                ("C", -1, 173.82, 2),
                ("CA", 0, 45.10, 2),
                ("C", 0, 171.90, 2),
            ],
        ),
    ];
    let seq: Vec<String> = ["MET", "GLN", "ILE", "THR"]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    let seq_ids = vec![1, 2, 3, 4];
    let p_mat = pair_matrix(&systems, &ScoreConfig::default());
    let assignments = vec![1, 2, 3];

    let hnco = make_carbonyl_cube(
        "HNCO",
        "HNCO",
        &[(8.10, 118.4, 175.08, 10.0), (8.65, 123.2, 173.82, 10.0)],
    );
    let hncaco = make_carbonyl_cube(
        "HN(CA)CO",
        "HNCACO",
        &[
            (8.30, 120.0, 175.10, 9.0),
            (8.10, 118.4, 175.08, 6.0),
            (8.10, 118.4, 173.80, 10.0),
            (8.65, 123.2, 173.82, 6.0),
            (8.65, 123.2, 171.90, 10.0),
        ],
    );

    let vp = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1280.0, 800.0));
    let ctx = egui::Context::default();
    let mut tiler = StripTilerPanel::new();
    let mut xh = SyncCrosshair::new(8.10, 118.4, Some(173.80));
    assert_eq!(tiler.band_view, StripBandView::Both);

    let run_frame = |tiler: &mut StripTilerPanel,
                     xh: &mut SyncCrosshair,
                     events: Vec<egui::Event>|
     -> egui::FullOutput {
        let mut out = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(vp),
                events,
                ..Default::default()
            },
            |ui| {
                apply_type_scale(ui.ctx());
                egui::Panel::right("co_walk_dock")
                    .default_size(320.0)
                    .min_size(260.0)
                    .show(ui, |ui| {
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
                            spectra_3d: StripSpectra3D {
                                aliphatic_3d: None,
                                carbonyl_3d: Some(&hnco),
                            },
                        };
                        let full = StripSpectra3DFull {
                            aliphatic_3d: None,
                            carbonyl_3d: Some(&hnco),
                            carbonyl_intra_3d: Some(&hncaco),
                        };
                        let _ = tiler.show_full(ui, &tctx, full, Some(xh));
                    });
                egui::CentralPanel::default().show(ui, |ui| {
                    ui.label("HSQC");
                });
            },
        );
        out.textures_delta.clear();
        out
    };

    // 1. Default `Both` mode: 2 plot rects per strip column (CO upper + CA/CB lower) = 6 rects
    let _ = run_frame(&mut tiler, &mut xh, Vec::new());
    let out_both = run_frame(&mut tiler, &mut xh, Vec::new());
    assert_eq!(tiler.last_column_rects.len(), 3);
    assert_eq!(tiler.last_plot_rects.len(), 6);
    let split_co_height = tiler.last_plot_rects[0].height();
    let audit_both = audit_text_layout(&out_both.shapes, vp);
    assert!(audit_both.min_font_px >= MIN_FONT_PX);
    assert!(
        audit_both.clipped.is_empty(),
        "Both mode clipped: {:?}",
        audit_both.clipped
    );
    assert!(
        audit_both.overlapping.is_empty(),
        "Both mode overlapping: {:?}",
        audit_both.overlapping
    );

    // 2. Click the connected `CO (HNCO/HN(CA)CO)` folder tab -> switches to `StripBandView::Co`
    let co_tab_rect = tiler
        .last_band_tab_rects
        .iter()
        .find(|(v, _)| *v == StripBandView::Co)
        .map(|(_, r)| *r)
        .expect("CO tab rect must be recorded");
    assert!(co_tab_rect.width() > 30.0 && co_tab_rect.height() >= 16.0);
    let _ = run_frame(&mut tiler, &mut xh, click_events(co_tab_rect.center()));
    assert_eq!(tiler.band_view, StripBandView::Co);

    let out_co = run_frame(&mut tiler, &mut xh, Vec::new());
    assert_eq!(
        tiler.last_plot_rects.len(),
        3,
        "CO-only mode must render 1 full-height carbonyl band per strip column"
    );
    let full_co_height = tiler.last_plot_rects[0].height();
    assert!(
        full_co_height > split_co_height * 2.5,
        "CO-only band height ({full_co_height:.1}) must expand to full height vs split ({split_co_height:.1})"
    );
    let audit_co = audit_text_layout(&out_co.shapes, vp);
    assert!(audit_co.min_font_px >= MIN_FONT_PX);
    assert!(
        audit_co.clipped.is_empty(),
        "CO mode clipped: {:?}",
        audit_co.clipped
    );
    assert!(
        audit_co.overlapping.is_empty(),
        "CO mode overlapping: {:?}",
        audit_co.overlapping
    );
    assert!(
        audit_co.texts.iter().any(|t| t.text.contains("dC'")),
        "CO mode must annotate C' sequential walk residual"
    );

    // 3. Click `CA/CB` tab -> switches to `StripBandView::CaCb` (full-height aliphatic band)
    let cacb_tab_rect = tiler
        .last_band_tab_rects
        .iter()
        .find(|(v, _)| *v == StripBandView::CaCb)
        .map(|(_, r)| *r)
        .expect("CA/CB tab rect must be recorded");
    let _ = run_frame(&mut tiler, &mut xh, click_events(cacb_tab_rect.center()));
    assert_eq!(tiler.band_view, StripBandView::CaCb);
    let out_cacb = run_frame(&mut tiler, &mut xh, Vec::new());
    assert_eq!(tiler.last_plot_rects.len(), 3);
    let audit_cacb = audit_text_layout(&out_cacb.shapes, vp);
    assert!(audit_cacb.min_font_px >= MIN_FONT_PX);
    assert!(audit_cacb.clipped.is_empty());
}

#[test]
fn test_assign_setup_drawer_three_steps_and_triage_jump_actions() {
    // Pure sequence parsing checks
    assert_eq!(
        parse_inline_sequence(">ubq\nMQIFV KTLTG\n12345").expect("valid FASTA"),
        "MQIFVKTLTG"
    );
    assert!(parse_inline_sequence("   \n>only_header\n").is_err());
    assert!(parse_inline_sequence("MQIFVZ").is_err());

    let vp = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1280.0, 900.0));
    let ctx = egui::Context::default();
    let mut drawer = AssignSetupDrawerState::new();
    drawer.sequence_paste_buf = ">Ubq76\nMQIFVKTLTG".to_string();

    let mut input = AssignSetupInput {
        sequence_len: 76,
        sequence_preview: "MQIFVKTLTGKTITLEVEPS...".to_string(),
        experiments: AssignSetupInput::canonical_checklist([
            (true, 72),
            (true, 260),
            (true, 134),
            (true, 140),
            (false, 0),
            (true, 71),
            (true, 138),
        ]),
        has_af_structure: true,
        solver_running: false,
        triage: AssignTriageSummary {
            high_conf_residues: vec![1, 2, 3, 4],
            ambiguous_residues: vec![9, 14],
            unassigned_residues: vec![23],
            locked_residues: vec![1],
            residue_labels: (0..76).map(|k| format!("R{}", k + 1)).collect(),
        },
        last_status: "Ready".to_string(),
    };

    let render = |drawer: &mut AssignSetupDrawerState,
                  input: &AssignSetupInput,
                  events: Vec<egui::Event>|
     -> (egui::FullOutput, Vec<AssignSetupAction>) {
        let mut emitted = Vec::new();
        let mut out = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(vp),
                events,
                ..Default::default()
            },
            |ui| {
                apply_type_scale(ui.ctx());
                egui::Panel::left("setup_drawer_host")
                    .default_size(380.0)
                    .show(ui, |ui| {
                        emitted = drawer.show_contents(ui, input);
                    });
                egui::CentralPanel::default().show(ui, |ui| {
                    ui.label("Main");
                });
            },
        );
        out.textures_delta.clear();
        (out, emitted)
    };

    let (_, _) = render(&mut drawer, &input, Vec::new());
    let (out, _) = render(&mut drawer, &input, Vec::new());
    let audit = audit_text_layout(&out.shapes, vp);
    assert!(
        audit.min_font_px >= MIN_FONT_PX,
        "drawer min font {} < {MIN_FONT_PX}",
        audit.min_font_px
    );
    assert!(
        audit.clipped.is_empty(),
        "drawer clipped text: {:?}",
        audit.clipped
    );
    assert!(
        audit.overlapping.is_empty(),
        "drawer overlapping text: {:?}",
        audit.overlapping
    );
    assert!(audit.texts.iter().any(|t| t.text.contains("76 aa loaded")));
    for exp_name in CANONICAL_BACKBONE_EXPERIMENTS {
        assert!(
            audit.texts.iter().any(|t| t.text.contains(exp_name)),
            "Step 2 checklist missing {exp_name}"
        );
    }

    let find_text_rect = |out: &egui::FullOutput, needle: &str| -> egui::Rect {
        audit_text_layout(&out.shapes, vp)
            .texts
            .iter()
            .find(|t| t.text.contains(needle))
            .map(|t| t.rect)
            .unwrap_or_else(|| panic!("missing text '{needle}' in AssignSetupDrawer"))
    };

    // Click `Apply Sequence`
    let apply_r = find_text_rect(&out, "Apply Sequence");
    let (_, acts_apply) = render(&mut drawer, &input, click_events(apply_r.center()));
    assert_eq!(
        acts_apply,
        vec![AssignSetupAction::ApplySequenceText(
            "MQIFVKTLTG".to_string()
        )]
    );

    // Click `Pick All (2D + 3D)`
    let pick_r = find_text_rect(&out, "Pick All (2D + 3D)");
    let (_, acts_pick) = render(&mut drawer, &input, click_events(pick_r.center()));
    assert_eq!(
        acts_pick,
        vec![AssignSetupAction::PickAllSpectra {
            snr_threshold: drawer.snr_threshold
        }]
    );

    // Click `Run Assignment`
    let run_r = find_text_rect(&out, "Run Assignment");
    let (_, acts_run) = render(&mut drawer, &input, click_events(run_r.center()));
    assert_eq!(
        acts_run,
        vec![AssignSetupAction::RunAssignment {
            engine: AssignSolverEngine::EvoGraph,
            regime: AssignProteinRegime::Globular,
            use_af_priors: true,
        }]
    );

    // Click `Ambiguous (2)` pill twice -> cycles through residue 9 then residue 14
    let amb_r = find_text_rect(&out, "Ambiguous (2)");
    let (_, acts_jump1) = render(&mut drawer, &input, click_events(amb_r.center()));
    assert_eq!(acts_jump1, vec![AssignSetupAction::JumpToResidue(9)]);
    assert_eq!(drawer.active_triage_bucket, Some(TriageBucket::Ambiguous));

    let (_, acts_jump2) = render(&mut drawer, &input, click_events(amb_r.center()));
    assert_eq!(acts_jump2, vec![AssignSetupAction::JumpToResidue(14)]);

    // Empty sequence shows warning pill cleanly
    input.sequence_len = 0;
    let (out_empty, _) = render(&mut drawer, &input, Vec::new());
    let audit_empty = audit_text_layout(&out_empty.shapes, vp);
    assert!(audit_empty
        .texts
        .iter()
        .any(|t| t.text.contains("Warning: No sequence")));
}

#[test]
fn test_assign_setup_drawer_7slot_mapper_actions_and_af_priors_guard() {
    let vp = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1280.0, 960.0));
    let ctx = egui::Context::default();
    let mut drawer = AssignSetupDrawerState::new();

    let input = AssignSetupInput {
        sequence_len: 76,
        sequence_preview: "MQIFVKTLTG...".to_string(),
        experiments: vec![
            ExperimentChecklistRow::new("HSQC", true, 72)
                .with_source("exp1_hsqc")
                .with_pulprog("hsqcetf3gpsi"),
            ExperimentChecklistRow::new("HNCACB", true, 260)
                .with_source("exp2_hncacb")
                .with_pulprog("hncacbgpwg3d")
                .with_registration_drift(0.008, -0.06),
            ExperimentChecklistRow::new("CBCA(CO)NH", false, 0),
            ExperimentChecklistRow::new("HNCA", true, 140)
                .with_source("exp4_hnca")
                .with_registration_drift(0.045, 0.32),
            ExperimentChecklistRow::new("HN(CO)CA", false, 0),
            ExperimentChecklistRow::new("HNCO", true, 71)
                .with_source("exp6_hnco")
                .with_pulprog("hncogpwg3d")
                .with_registration_drift(-0.004, 0.03),
            ExperimentChecklistRow::new("HN(CA)CO", false, 0),
        ],
        has_af_structure: false,
        solver_running: false,
        triage: AssignTriageSummary::default(),
        last_status: "Ready".to_string(),
    };

    let render = |drawer: &mut AssignSetupDrawerState,
                  input: &AssignSetupInput,
                  events: Vec<egui::Event>|
     -> (
        egui::FullOutput,
        Vec<AssignSetupAction>,
        Vec<AssignSlotAction>,
    ) {
        let mut emitted = Vec::new();
        let mut out = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(vp),
                events,
                ..Default::default()
            },
            |ui| {
                apply_type_scale(ui.ctx());
                egui::Panel::left("setup_mapper_host")
                    .default_size(420.0)
                    .show(ui, |ui| {
                        emitted = drawer.show_contents(ui, input);
                    });
                egui::CentralPanel::default().show(ui, |ui| {
                    ui.label("Main");
                });
            },
        );
        let slot_acts = drawer.take_slot_actions();
        out.textures_delta.clear();
        (out, emitted, slot_acts)
    };

    let (_, _, _) = render(&mut drawer, &input, Vec::new());
    let (out, _, _) = render(&mut drawer, &input, Vec::new());
    let audit = audit_text_layout(&out.shapes, vp);
    assert!(
        audit.min_font_px >= MIN_FONT_PX,
        "7-slot mapper min font {} < {MIN_FONT_PX}",
        audit.min_font_px
    );
    assert!(
        audit.clipped.is_empty(),
        "7-slot mapper clipped text: {:?}",
        audit.clipped
    );
    assert!(
        audit.overlapping.is_empty(),
        "7-slot mapper overlapping text: {:?}",
        audit.overlapping
    );

    let joined: String = audit
        .texts
        .iter()
        .map(|t| t.text.as_str())
        .collect::<Vec<_>>()
        .join(" | ");
    for expected in [
        "Load Study Folder...",
        "Pick All (2D + 3D)",
        "exp1_hsqc",
        "exp2_hncacb",
        "Not loaded",
        "ΔH +0.008 ΔN -0.06 ppm",
        "ΔH +0.045 ΔN +0.32 ppm",
        "Use AF Structure Priors (no PDB/AF loaded)",
    ] {
        assert!(
            joined.contains(expected),
            "7-slot mapper UI missing {expected:?} in: {joined}"
        );
    }

    // 1. Click `Load Study Folder...` -> AssignSlotAction::LoadStudyFolder
    let study_r = audit
        .texts
        .iter()
        .find(|t| t.text.contains("Load Study Folder..."))
        .map(|t| t.rect)
        .expect("Load Study Folder... rect");
    let (_, _, acts_study) = render(&mut drawer, &input, click_events(study_r.center()));
    assert_eq!(acts_study, vec![AssignSlotAction::LoadStudyFolder]);

    // 2. Click `Load...` on CBCA(CO)NH row (same y-band as CBCA(CO)NH label)
    let cbcaconh_y = audit
        .texts
        .iter()
        .find(|t| t.text == "CBCA(CO)NH")
        .map(|t| t.rect.center().y)
        .expect("CBCA(CO)NH row label");
    let load_cbcaconh_r = audit
        .texts
        .iter()
        .filter(|t| t.text == "Load...")
        .min_by(|a, b| {
            (a.rect.center().y - cbcaconh_y)
                .abs()
                .total_cmp(&(b.rect.center().y - cbcaconh_y).abs())
        })
        .map(|t| t.rect)
        .expect("CBCA(CO)NH Load... button");
    let (_, _, acts_load) = render(&mut drawer, &input, click_events(load_cbcaconh_r.center()));
    assert_eq!(
        acts_load,
        vec![AssignSlotAction::LoadSlotSpectrum("CBCA(CO)NH".to_string())]
    );

    // 3. Click `Clear` on HNCO row
    let hnco_y = audit
        .texts
        .iter()
        .find(|t| t.text == "HNCO")
        .map(|t| t.rect.center().y)
        .expect("HNCO row label");
    let clear_hnco_r = audit
        .texts
        .iter()
        .filter(|t| t.text == "Clear")
        .min_by(|a, b| {
            (a.rect.center().y - hnco_y)
                .abs()
                .total_cmp(&(b.rect.center().y - hnco_y).abs())
        })
        .map(|t| t.rect)
        .expect("HNCO Clear button");
    let (_, _, acts_clear) = render(&mut drawer, &input, click_events(clear_hnco_r.center()));
    assert_eq!(
        acts_clear,
        vec![AssignSlotAction::ClearSlotSpectrum("HNCO".to_string())]
    );

    // 4. Open HNCO experiment-type override ComboBox (right-aligned in HNCO card) and select HN(CA)CO
    let hnco_combo_r = audit
        .texts
        .iter()
        .filter(|t| t.text == "HNCO" && (t.rect.center().y - hnco_y).abs() < 42.0)
        .max_by(|a, b| a.rect.left().total_cmp(&b.rect.left()))
        .map(|t| t.rect)
        .expect("HNCO ComboBox button");
    let (_, _, _) = render(&mut drawer, &input, click_events(hnco_combo_r.center()));
    // Second pass paints the opened ComboBox popup layer
    let (out_open, _, _) = render(&mut drawer, &input, Vec::new());
    let audit_open = audit_text_layout(&out_open.shapes, vp);
    // In the open popup (right side, x > 150), find the selectable option "HN(CA)CO"
    let hncaco_opt_r = audit_open
        .texts
        .iter()
        .filter(|t| t.text == "HN(CA)CO" && t.rect.left() > 150.0)
        .map(|t| t.rect)
        .next()
        .expect("HN(CA)CO option in ComboBox popup");
    let (_, _, acts_reassign) = render(&mut drawer, &input, click_events(hncaco_opt_r.center()));
    assert_eq!(
        acts_reassign,
        vec![AssignSlotAction::ReassignSlotExperiment {
            from_slot: "HNCO".to_string(),
            to_exp: "HN(CA)CO".to_string(),
        }]
    );

    // 5. With has_af_structure == false, Run Assignment must emit use_af_priors: false
    let (out_after, _, _) = render(&mut drawer, &input, Vec::new());
    let run_r = audit_text_layout(&out_after.shapes, vp)
        .texts
        .iter()
        .find(|t| t.text.contains("Run Assignment"))
        .map(|t| t.rect)
        .expect("Run Assignment rect");
    let (_, acts_run, _) = render(&mut drawer, &input, click_events(run_r.center()));
    assert_eq!(
        acts_run,
        vec![AssignSetupAction::RunAssignment {
            engine: AssignSolverEngine::EvoGraph,
            regime: AssignProteinRegime::Globular,
            use_af_priors: false,
        }]
    );
}

#[test]
fn test_relax_panel_hsqc_peak_sync_pseudo3d_delays_and_phase22_detectors() {
    let mut panel = RelaxPanelState::new();
    panel.mc_replicates = 60;

    // 171.assign:P1-3: hetNOE order MUST default to fail-closed Auto (never RefFirst)
    assert_eq!(panel.hetnoe_order, HetNoeOrder::Auto);

    // 171.assign:P1-1: TRACT inputs MUST start empty (no placeholder 5.5/16.5 or hardcoded sigmas)
    assert!(panel.tract_r_alpha_input.is_none());
    assert!(panel.tract_r_beta_input.is_none());
    let tract_err = panel
        .run_phase22_tract_from_inputs()
        .expect_err("TRACT without experimental TROSY/anti-TROSY rates must fail closed");
    assert!(
        tract_err.contains(TRACT_MISSING_RATES_ERR),
        "unexpected TRACT error: {tract_err}"
    );
    assert!(panel.tract_report.is_none());

    // 1. Sync assigned HSQC peaks from Assign mode -> locked TargetPeakSeed reference
    let hsqc_peaks = vec![
        Peak2D::assigned(2, "GLN", 8.42, 121.3),
        Peak2D {
            h_ppm: 8.15,
            n_ppm: 118.7,
            seq_id: None,
            res_name: None,
            label: Some("I3".to_string()),
            intensity: Some(1200.0),
        },
        Peak2D::new(f64::NAN, 120.0), // ignored non-finite
    ];
    let synced = panel.sync_assigned_hsqc_peaks(&hsqc_peaks, ">ubq\nMQIFVKTLTG");
    assert_eq!(synced, 2);
    assert_eq!(panel.synced_peak_count(), 2);
    assert_eq!(panel.synced_peak_seeds[0].seq_id, 2);
    assert_eq!(panel.synced_peak_seeds[0].res_name, "GLN");
    assert_eq!(panel.synced_peak_seeds[1].seq_id, 3);
    assert_eq!(panel.synced_peak_seeds[1].res_name, "ILE");

    // 2. Editable vdlist / vclist delay parser (both seconds/ms suffixes & integer vclist counts)
    assert_eq!(
        RelaxPanelState::parse_delays_override_str("", None).unwrap(),
        None
    );
    let vd = RelaxPanelState::parse_delays_override_str("10ms, 50m, 0.25s, 1.0", None)
        .unwrap()
        .expect("parsed vdlist");
    assert_eq!(vd.len(), 4);
    assert!((vd[0] - 0.01).abs() < 1e-9);
    assert!((vd[1] - 0.05).abs() < 1e-9);
    assert!((vd[2] - 0.25).abs() < 1e-9);
    assert!((vd[3] - 1.00).abs() < 1e-9);

    let vc = RelaxPanelState::parse_delays_override_str("1 2 4 8", Some(16.0))
        .unwrap()
        .expect("parsed vclist");
    assert!((vc[0] - 0.016).abs() < 1e-9);
    assert!((vc[3] - 0.128).abs() < 1e-9);

    // 171.assign:P1-2: plain integer vclist with 0.0 ms / None / negative cycle duration MUST fail closed!
    let err_zero = RelaxPanelState::parse_delays_override_str("2 4 8 16", Some(0.0))
        .expect_err("integer vclist with 0.0 ms cycle duration must fail closed");
    assert!(
        err_zero.contains("cpmg_cycle_ms > 0.0") || err_zero.contains("CPMG cycle"),
        "unexpected vclist 0.0 ms error: {err_zero}"
    );
    assert!(
        RelaxPanelState::parse_delays_override_str("2 4 8 16", None).is_err(),
        "integer vclist with None cycle duration must fail closed"
    );
    assert!(
        RelaxPanelState::parse_delays_override_str("2 4 8 16", Some(-4.0)).is_err(),
        "integer vclist with negative cycle duration must fail closed"
    );
    assert!(RelaxPanelState::parse_delays_override_str("-0.1", None).is_err());

    // Fail-closed desktop folder validation
    #[cfg(not(target_arch = "wasm32"))]
    {
        assert!(panel
            .load_bruker_pseudo3d_series(
                RelaxExperimentKind::R1,
                "/nonexistent/bruker/pseudo3d_folder",
                HetNoeOrder::RefFirst,
                None,
            )
            .is_err());
        assert!(panel
            .load_bruker_pseudo3d_series(
                RelaxExperimentKind::Cpmg,
                "/nonexistent",
                HetNoeOrder::Auto,
                None,
            )
            .is_err());
    }

    // 3. Populate realistic dual-field R1/R2/hetNOE + Model-Free + Phase 22 Detectors / IMPACT
    let seq: Vec<String> = [
        "MET", "GLN", "ILE", "PHE", "VAL", "LYS", "THR", "LEU", "THR", "GLY",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect();
    let seq_ids: Vec<i32> = (1..=10).collect();

    panel
        .load_demo_dynamics(&seq, &seq_ids, None)
        .expect("load demo dynamics");
    let suite_msg = panel
        .run_phase22_suite()
        .expect("Phase 22 Detectors / IMPACT must succeed on fitted rates");
    assert!(suite_msg.contains("Dynamics Detectors"));
    assert!(
        suite_msg.contains(TRACT_MISSING_RATES_ERR),
        "suite must report TRACT skipped without experimental rates: {suite_msg}"
    );
    assert!(
        panel.tract_report.is_none(),
        "run_phase22_suite must NOT fabricate placeholder TRACT report"
    );

    let design = panel
        .detector_design
        .as_ref()
        .expect("detector design populated");
    assert!(design.num_detectors >= 2);
    assert_eq!(panel.detector_fits.len(), 10);
    assert_eq!(panel.impact_fits.len(), 10);
    for fit in panel.impact_fits.values() {
        assert!((fit.sum_amplitudes - 1.0).abs() < 1e-6);
    }

    // Supply explicit experimental TROSY (R_alpha ± sigma_alpha) and anti-TROSY (R_beta ± sigma_beta)
    // and verify TRACT computes eta_xy and propagates the user's actual sigmas (not hardcoded 0.20/0.35).
    panel
        .import_tract_rates_tsv(
            "r_alpha\tsigma_alpha\tr_beta\tsigma_beta\n5.80\t0.12\t18.20\t0.16\n",
        )
        .expect("TRACT with explicit experimental rates must succeed");
    let tract = panel.tract_report.as_ref().expect("TRACT report populated");
    assert!((tract.eta_xy_s_inv - 6.20).abs() < 1e-9);
    // sigma_eta_xy = 0.5 * sqrt(0.12^2 + 0.16^2) = 0.5 * 0.20 = 0.10 s^-1
    assert!(
        (tract.sigma_eta_xy_s_inv - 0.10).abs() < 1e-9,
        "sigma_eta_xy must reflect user sigmas (0.10), got {}",
        tract.sigma_eta_xy_s_inv
    );
    assert!(tract.tau_m_rigid.tau_m_sec > 1.0e-9);
    assert!(tract.kappa0_sec > 1.0);

    // 4. Headless egui render across all 3 sub-tabs with strict >= 11 px font & zero-clip audit
    let vp = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1280.0, 800.0));
    let ctx = egui::Context::default();
    for tab in RelaxStudioSubTab::ALL {
        panel.active_sub_tab = tab;
        let mut out = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(vp),
                ..Default::default()
            },
            |ui| {
                apply_type_scale(ui.ctx());
                egui::CentralPanel::default().show(ui, |ui| {
                    let _ = panel.ui(ui, &seq, &seq_ids, Some(1), None);
                });
            },
        );
        let audit = audit_text_layout(&out.shapes, vp);
        out.textures_delta.clear();
        assert!(
            audit.min_font_px >= MIN_FONT_PX,
            "RelaxPanel tab {tab:?} font below floor: {}",
            audit.min_font_px
        );
        assert!(
            audit.clipped.is_empty(),
            "RelaxPanel tab {tab:?} clipped text: {:?}",
            audit.clipped
        );
        assert!(
            audit.overlapping.is_empty(),
            "RelaxPanel tab {tab:?} overlapping text: {:?}",
            audit.overlapping
        );
    }

    // Verify no unused import warning on BTreeMap
    let _: BTreeMap<i32, f64> = BTreeMap::new();
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_relax_panel_bruker_pseudo3d_folder_end_to_end_t1_fit() {
    use std::f64::consts::TAU;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    // Synthesize a small real Bruker pseudo-3D T1 `ser` directory (`acqus`, `acqu2s`, `vdlist`, `ser`)
    // with one clean 2D peak decaying at R1 = 1.8 s^-1 across 5 delays, then load & fit directly via
    // `RelaxPanelState::load_bruker_pseudo3d_series` using synced HSQC footprints!
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = std::env::temp_dir().join(format!("ferric_gui_p3d_t1_{}_{}", std::process::id(), ts));
    fs::create_dir_all(&dir).expect("create temp bruker dir");

    let n_planes = 5usize;
    let td_x = 64usize; // 32 complex points along direct 1H
    let n_t1 = 16usize; // 16 t1 increments -> 32 real rows per plane (States real/imag)
    let td_y = 2 * n_t1;
    let sw_h = 10.0_f64;
    let o1p = 4.7_f64;
    let sfo1_h = 600.0_f64;
    let sw_hz_h = sw_h * sfo1_h;

    let sw_n = 30.0_f64;
    let o2p = 120.0_f64;
    let sfo1_n = 60.8_f64;
    let sw_hz_n = sw_n * sfo1_n;

    let delays = [0.02_f64, 0.15, 0.35, 0.65, 1.10];
    let true_r1 = 1.80_f64;
    let peak_h = 8.20_f64;
    let peak_n = 118.5_f64;

    let acqus = format!(
        "##TITLE= Synthetic T1\n##$TD= {td_x}\n##$BYTORDA= 0\n##$DTYPA= 0\n##$SFO1= {sfo1_h}\n##$BF1= {sfo1_h}\n##$SW= {sw_h}\n##$SW_h= {sw_hz_h}\n##$O1= {}\n##$NUC1= <1H>\n##$GRPDLY= 0\n##END=\n",
        o1p * sfo1_h
    );
    let acqu2s = format!(
        "##TITLE= Synthetic T1 F1\n##$TD= {td_y}\n##$SFO1= {sfo1_n}\n##$BF1= {sfo1_n}\n##$SW= {sw_n}\n##$SW_h= {sw_hz_n}\n##$O1= {}\n##$NUC1= <15N>\n##$FnMODE= 5\n##END=\n",
        o2p * sfo1_n
    );
    let acqu3s = format!(
        "##TITLE= Synthetic T1 Planes\n##$TD= {n_planes}\n##$SFO1= {sfo1_n}\n##$SW= 1.0\n##END=\n"
    );
    let vdlist = delays
        .iter()
        .map(|d| format!("{d:.4}"))
        .collect::<Vec<_>>()
        .join("\n");

    fs::write(dir.join("acqus"), acqus).expect("write acqus");
    fs::write(dir.join("acqu2s"), acqu2s).expect("write acqu2s");
    fs::write(dir.join("acqu3s"), acqu3s).expect("write acqu3s");
    fs::write(dir.join("vdlist"), vdlist).expect("write vdlist");

    // Generate plane-queued [plane][t1][quad][t2] 32-bit little-endian integer FID `ser`
    let f_h_hz = (peak_h - o1p) * sfo1_h;
    let f_n_hz = (peak_n - o2p) * sfo1_n;
    let dt_x = 1.0 / sw_hz_h;
    let dt_y = 1.0 / sw_hz_n;
    let n_cx = td_x / 2;

    let mut ser_bytes = Vec::with_capacity(n_planes * td_y * td_x * 4);
    for &tau in &delays {
        let amp = 50_000.0 * (-true_r1 * tau).exp();
        for it1 in 0..n_t1 {
            let t1 = (it1 as f64) * dt_y;
            let env1 = (-t1 * 18.0).exp();
            let ph1 = TAU * f_n_hz * t1;
            for quad in 0..2usize {
                let q_mod = if quad == 0 { ph1.cos() } else { ph1.sin() };
                for it2 in 0..n_cx {
                    let t2 = (it2 as f64) * dt_x;
                    let env2 = (-t2 * 35.0).exp();
                    let ph2 = TAU * f_h_hz * t2;
                    let re = (amp * env1 * env2 * q_mod * ph2.cos()).round() as i32;
                    let im = (amp * env1 * env2 * q_mod * ph2.sin()).round() as i32;
                    ser_bytes.extend_from_slice(&re.to_le_bytes());
                    ser_bytes.extend_from_slice(&im.to_le_bytes());
                }
            }
        }
    }
    fs::write(dir.join("ser"), &ser_bytes).expect("write ser");

    let mut panel = RelaxPanelState::new();
    panel.mc_replicates = 60;
    panel.sync_assigned_hsqc_peaks(&[Peak2D::assigned(5, "VAL", peak_h, peak_n)], "MQIFV");

    let msg = panel
        .load_bruker_pseudo3d_series(RelaxExperimentKind::R1, &dir, HetNoeOrder::Auto, None)
        .expect("Bruker pseudo-3D T1 load + fit must succeed");
    assert!(msg.contains("R1"), "unexpected status: {msg}");
    assert_eq!(panel.t1_slot.loaded_planes, n_planes);
    assert_eq!(panel.t1_slot.fitted_residues, 1);
    let fit = panel.r1_fits.get(&5).expect("VAL5 R1 fit");
    assert!(
        (fit.rate - true_r1).abs() < 0.25,
        "fitted R1 ({:.3}) should recover true R1 ({true_r1:.3})",
        fit.rate
    );

    let _ = fs::remove_dir_all(&dir);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_relax_panel_hetnoe_auto_default_fails_closed_without_pulprog() {
    use std::f64::consts::TAU;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = std::env::temp_dir().join(format!(
        "ferric_gui_hetnoe_auto_{}_{}",
        std::process::id(),
        ts
    ));
    fs::create_dir_all(&dir).expect("create temp hetnoe dir");

    let n_planes = 2usize;
    let td_x = 64usize;
    let n_t1 = 16usize;
    let td_y = 2 * n_t1;
    let sw_h = 10.0_f64;
    let o1p = 4.7_f64;
    let sfo1_h = 600.0_f64;
    let sw_hz_h = sw_h * sfo1_h;
    let sw_n = 30.0_f64;
    let o2p = 120.0_f64;
    let sfo1_n = 60.8_f64;
    let sw_hz_n = sw_n * sfo1_n;
    let peak_h = 8.20_f64;
    let peak_n = 118.5_f64;

    // Schedule-less 2-plane hetNOE folder (NO vdlist, NO pulseprogram file, PULPROG = hsqcnoef3gpsi)
    let acqus = format!(
        "##TITLE= Synthetic hetNOE\n##$PULPROG= <hsqcnoef3gpsi>\n##$TD= {td_x}\n##$BYTORDA= 0\n##$DTYPA= 0\n##$SFO1= {sfo1_h}\n##$BF1= {sfo1_h}\n##$SW= {sw_h}\n##$SW_h= {sw_hz_h}\n##$O1= {}\n##$NUC1= <1H>\n##$GRPDLY= 0\n##$D= (0..63)\n0 3.0 0 0 0 0 0 0\n##END=\n",
        o1p * sfo1_h
    );
    let acqu2s = format!(
        "##TITLE= Synthetic hetNOE F1\n##$TD= {td_y}\n##$SFO1= {sfo1_n}\n##$BF1= {sfo1_n}\n##$SW= {sw_n}\n##$SW_h= {sw_hz_n}\n##$O1= {}\n##$NUC1= <15N>\n##$FnMODE= 5\n##END=\n",
        o2p * sfo1_n
    );
    let acqu3s = format!(
        "##TITLE= Synthetic hetNOE Planes\n##$TD= {n_planes}\n##$SFO1= {sfo1_n}\n##$SW= 1.0\n##END=\n"
    );
    fs::write(dir.join("acqus"), acqus).expect("write acqus");
    fs::write(dir.join("acqu2s"), acqu2s).expect("write acqu2s");
    fs::write(dir.join("acqu3s"), acqu3s).expect("write acqu3s");

    // Plane 0 = saturated (0.80), Plane 1 = reference (1.00) -> true hetNOE = 0.80
    let f_h_hz = (peak_h - o1p) * sfo1_h;
    let f_n_hz = (peak_n - o2p) * sfo1_n;
    let dt_x = 1.0 / sw_hz_h;
    let dt_y = 1.0 / sw_hz_n;
    let n_cx = td_x / 2;
    let mut ser_bytes = Vec::with_capacity(n_planes * td_y * td_x * 4);
    for &amp in &[40_000.0_f64, 50_000.0_f64] {
        for it1 in 0..n_t1 {
            let t1 = (it1 as f64) * dt_y;
            let env1 = (-t1 * 18.0).exp();
            let ph1 = TAU * f_n_hz * t1;
            for quad in 0..2usize {
                let q_mod = if quad == 0 { ph1.cos() } else { ph1.sin() };
                for it2 in 0..n_cx {
                    let t2 = (it2 as f64) * dt_x;
                    let env2 = (-t2 * 35.0).exp();
                    let ph2 = TAU * f_h_hz * t2;
                    let re = (amp * env1 * env2 * q_mod * ph2.cos()).round() as i32;
                    let im = (amp * env1 * env2 * q_mod * ph2.sin()).round() as i32;
                    ser_bytes.extend_from_slice(&re.to_le_bytes());
                    ser_bytes.extend_from_slice(&im.to_le_bytes());
                }
            }
        }
    }
    fs::write(dir.join("ser"), &ser_bytes).expect("write ser");

    let mut panel = RelaxPanelState::new();
    assert_eq!(panel.hetnoe_order, HetNoeOrder::Auto);
    panel.sync_assigned_hsqc_peaks(&[Peak2D::assigned(5, "VAL", peak_h, peak_n)], "MQIFV");

    // 1. Default HetNoeOrder::Auto MUST fail closed when pulseprogram is absent
    let auto_res = panel.load_bruker_pseudo3d_series(
        RelaxExperimentKind::HetNoe,
        &dir,
        panel.hetnoe_order,
        None,
    );
    assert!(
        auto_res.is_err(),
        "schedule-less 2-plane hetNOE without pulseprogram must fail closed under default Auto"
    );

    // 2. Explicit SatFirst succeeds and recovers hetNOE ~= 0.80 (not inverted 1.25!)
    let sat_res = panel.load_bruker_pseudo3d_series(
        RelaxExperimentKind::HetNoe,
        &dir,
        HetNoeOrder::SatFirst,
        None,
    );
    assert!(
        sat_res.is_ok(),
        "explicit SatFirst must succeed: {:?}",
        sat_res
    );
    let noe = panel.hetnoe_fits.get(&5).expect("VAL5 hetNOE fit");
    assert!(
        (noe.noe - 0.80).abs() < 0.08,
        "explicit SatFirst should recover NOE ~= 0.80, got {:.3}",
        noe.noe
    );

    let _ = fs::remove_dir_all(&dir);
}
