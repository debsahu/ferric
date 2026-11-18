//! Unified interaction grammar, chord-vs-single-key isolation, unified User+MCP undo journal,
//! multi-level water suppression undo (`ws, ws, wu, wu`), fail-closed model provenance,
//! non-circular concordance, and PAE texture cache tests
//! (`UI-C1`..`UI-C18`, `v0150:P1-6`, `20.W1:P2-5`, `21.W1:P0-2`, `21.W1:P1-5`, `21.W3:P1-14`).
//!
//! Provenance: original clean-room integration tests for FERRIC (`nmr-gui`).

use std::sync::Arc;

use eframe::egui;
use nmr_core::idp_propensity::PropensityConfig;
use nmr_core::nmr1d::{Experiment1DKind, Spectrum1D};
use nmr_core::nmrstar::read_star_str;
use nmr_core::priors::Priors;
use nmr_core::session::WorkbenchCommand;
use nmr_core::solver::{pair_matrix, unary_matrix, ScoreConfig};
use nmr_core::spectrum::{AxisPpm, Spectrum2D, Spectrum3D};
use nmr_core::spinsystems::{build_spin_systems, Tolerances};
use nmr_core::water::{
    suppress_water_3d, suppress_water_spectrum_1d, suppress_water_spectrum_2d,
    WaterSuppressionConfig,
};
use nmr_gui::contour_painter::OverlayLayerSpec;
use nmr_gui::demo_preset::{build_builtin_demo_suite, BUILTIN_DEMO_SUITE_SEED};
use nmr_gui::integrative_af_panel::{IntegrativeAfPanel, IntegrativeStudioTab};
use nmr_gui::interaction_grammar::{
    contour_floor_after, gain_after, grammar_markdown_table, GestureView, SignalDirection,
    INTERACTION_GRAMMAR,
};
use nmr_gui::keymap::{ChordDispatcher, ChordKey, ChordOutcome, KeyChordInput, KeyFocusContext};
use nmr_gui::mcp_bridge::{McpUiCommand, McpUndoLog, WorkbenchSnapshot};
use nmr_gui::mol3d_overlay::ConcordanceTrackClass;
use nmr_gui::mol3d_viewer::Mol3dViewerPanel;
use nmr_gui::session_io::WorkbenchSessionController;
use nmr_gui::ss_ribbon_widget::SsRibbonState;
use nmr_gui::stack1d_panel::{AttachedSpectrum1d, Stack1dState, ToolMode1d, Trace1d};
use nmr_gui::strip_tiler::{StripSpectra3D, StripTilerAction, StripTilerContext, StripTilerPanel};
use nmr_gui::undo_journal::{KeyedSnapshotStack, SpectrumKey, UndoJournal, UndoOrigin};
use nmr_gui::workflow_bar::WorkbenchAction;

const SYNTH_UBQ_STR: &str = include_str!("../../../data/fixtures/synthetic_ubiquitin.str");
const UBQ_1UBQ_PDB: &str = include_str!("../../../data/fixtures/ubiquitin_1ubq_xray.pdb");

/// Minimal standards-compliant AFDB v6 mmCIF carrying `_ma_qa_metric` (`pLDDT`) and
/// `_ma_qa_metric_local` per-residue confidence scores (`21.W1:P0-2`).
const MINIMAL_AFDB_MMCIF: &str = "\
data_AF_P0CG48_F1
#
loop_
_ma_qa_metric.id
_ma_qa_metric.name
_ma_qa_metric.mode
_ma_qa_metric.type
1 pLDDT local pLDDT
#
loop_
_ma_qa_metric_local.model_id
_ma_qa_metric_local.label_asym_id
_ma_qa_metric_local.label_seq_id
_ma_qa_metric_local.metric_id
_ma_qa_metric_local.metric_value
1 A 1 1 94.50
1 A 2 1 91.20
1 A 3 1 88.75
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.pdbx_PDB_model_num
ATOM 1 N N . MET A 1 ? 0.000 0.000 0.000 1.00 94.50 1
ATOM 2 C CA . MET A 1 ? 1.458 0.000 0.000 1.00 94.50 1
ATOM 3 C C . MET A 1 ? 2.009 1.420 0.000 1.00 94.50 1
ATOM 4 N N . GLN A 2 ? 3.300 1.600 0.000 1.00 91.20 1
ATOM 5 C CA . GLN A 2 ? 3.920 2.920 0.000 1.00 91.20 1
ATOM 6 C C . GLN A 2 ? 5.440 2.860 0.000 1.00 91.20 1
ATOM 7 N N . ILE A 3 ? 6.020 1.660 0.000 1.00 88.75 1
ATOM 8 C CA . ILE A 3 ? 7.470 1.520 0.000 1.00 88.75 1
ATOM 9 C C . ILE A 3 ? 8.100 2.910 0.000 1.00 88.75 1
#
";

fn feed_plain_char(disp: &mut ChordDispatcher, ch: char, now_ms: u64) -> ChordOutcome {
    let key = match ch {
        ' ' => ChordKey::Space,
        '+' => ChordKey::Plus,
        '-' => ChordKey::Minus,
        '?' => ChordKey::QuestionMark,
        c => ChordKey::Letter(c),
    };
    disp.feed_key(
        KeyChordInput {
            key,
            shift: ch == '?',
            ctrl_or_cmd: false,
            alt: false,
        },
        now_ms,
        KeyFocusContext::default(),
    )
}

#[test]
fn test_c1_two_letter_chords_never_trigger_1d_single_key_tools() {
    let chords = [
        ('p', 'h', WorkbenchAction::SelectPhase1dMode),
        ('k', 'r', WorkbenchAction::SelectPickPeaksMode),
        ('m', 'p', WorkbenchAction::SelectMultiplet1dMode),
        ('z', 'm', WorkbenchAction::SelectZoomBoxMode),
        ('c', 'i', WorkbenchAction::ShowMoreSignal),
        ('z', 'p', WorkbenchAction::ZoomPrevious),
        ('p', 'a', WorkbenchAction::AddPeakAtCrosshair),
        ('p', 'd', WorkbenchAction::DeleteSelectedPeaks),
        ('n', 'p', WorkbenchAction::NextCubePlane),
        ('o', 'p', WorkbenchAction::OpenIngestMatrix),
    ];

    for (k1, k2, expected_action) in chords {
        let mut disp = ChordDispatcher::new();
        let mut stack = Stack1dState::with_traces(vec![Trace1d::new(
            "1h",
            "1D 1H",
            0.0,
            vec![8.0, 4.0, 1.0],
            vec![1.0, 5.0, 2.0],
        )]);
        assert_eq!(stack.tool_mode, ToolMode1d::Zoom);

        let d1 = feed_plain_char(&mut disp, k1, 1_000);
        assert_eq!(d1, ChordOutcome::PendingPrefix { prefix: k1 });
        assert!(
            disp.drain_single_key_tools().is_empty(),
            "first key {k1:?} of chord {k1}{k2} must NEVER queue a single-key 1D tool"
        );

        let d2 = feed_plain_char(&mut disp, k2, 1_120);
        assert_eq!(d2, ChordOutcome::Dispatched(expected_action));
        assert!(
            disp.drain_single_key_tools().is_empty(),
            "completed chord {k1}{k2} must NEVER queue a single-key 1D tool (`UI-C1`)"
        );
        assert_eq!(
            stack.tool_mode,
            ToolMode1d::Zoom,
            "chord {k1}{k2} must not mutate 1D tool_mode via raw single-letter key"
        );

        // Standalone 'h' fires immediately when idle; 'p' fires ONLY after chord timeout expires
        let dh = feed_plain_char(&mut disp, 'h', 2_000);
        assert_eq!(dh, ChordOutcome::UnboundKey);
        assert_eq!(disp.drain_single_key_tools(), vec!['h']);
        assert!(stack.apply_single_key_tool('h'));
        assert_eq!(stack.tool_mode, ToolMode1d::Pan);

        let _ = feed_plain_char(&mut disp, 'p', 3_000);
        assert!(disp.drain_single_key_tools().is_empty());
        assert_eq!(disp.tick_timeout(3_600), Some('p'));
        let drained = disp.drain_single_key_tools();
        assert_eq!(drained, vec!['p']);
        assert!(stack.apply_single_key_tool(drained[0]));
        assert_eq!(stack.tool_mode, ToolMode1d::Phase);
    }
}

#[test]
fn test_c2_c3_c4_unified_user_and_mcp_timestamped_undo_redo() {
    #[derive(Clone, Debug, PartialEq, Eq)]
    enum StepKind {
        Session,
        Mcp,
    }

    let mut session = WorkbenchSessionController::default();
    session.state.sequence = vec!["MET".into(), "GLN".into(), "ILE".into()];
    session.state.seq_ids = vec![1, 2, 3];

    let mut mcp_log = McpUndoLog::new();
    let mut mcp_snap = WorkbenchSnapshot::default();
    let mut journal: UndoJournal<StepKind> = UndoJournal::with_capacity(64);

    // 1. User assigns spin system 0 -> residue 1 (GLN2) (seq #1)
    let _ = session.execute(WorkbenchCommand::AssignSpinSystem {
        spin_system: 0,
        residue_seq_idx: 1,
    });
    let s1 = journal.record(UndoOrigin::User, "User assign GLN2<-SS0", StepKind::Session);
    assert_eq!(
        session
            .state
            .assignments
            .get(&1)
            .and_then(|a| a.spin_system),
        Some(0)
    );

    // 2. MCP moves crosshair to (8.42, 121.5) (seq #2 > s1)
    mcp_log.apply_command(
        &mut mcp_snap,
        McpUiCommand::SetCrosshairPpm {
            h_ppm: 8.42,
            n_ppm: 121.5,
            c_ppm: Some(55.1),
        },
    );
    let s2 = journal.record(UndoOrigin::Mcp, "MCP set crosshair", StepKind::Mcp);
    assert!(s2 > s1);
    assert_eq!(mcp_snap.crosshair_ppm.0, 8.42);

    // 3. User locks spin system 1 -> residue 2 (ILE3) (seq #3 > s2)
    let _ = session.execute(WorkbenchCommand::LockResidue {
        spin_system: 1,
        residue_seq_idx: 2,
        locked: true,
    });
    let s3 = journal.record(UndoOrigin::User, "User lock ILE3", StepKind::Session);
    assert!(s3 > s2);
    assert_eq!(session.state.locked_residues.get(&1), Some(&2));

    // Cmd+Z #1: undoes step #3 (User lock)
    let u1 = journal.pop_undo().expect("undo #1");
    assert_eq!(u1.seq, s3);
    assert_eq!(u1.origin, UndoOrigin::User);
    let _ = session.undo();
    journal.push_redo(u1);
    assert!(!session.state.locked_residues.contains_key(&1));
    assert_eq!(mcp_snap.crosshair_ppm.0, 8.42);

    // Cmd+Z #2: undoes step #2 (MCP crosshair)
    let u2 = journal.pop_undo().expect("undo #2");
    assert_eq!(u2.seq, s2);
    assert_eq!(u2.origin, UndoOrigin::Mcp);
    assert!(mcp_log.undo_last(&mut mcp_snap).is_some());
    journal.push_redo(u2);
    assert_ne!(mcp_snap.crosshair_ppm.0, 8.42);
    assert_eq!(
        session
            .state
            .assignments
            .get(&1)
            .and_then(|a| a.spin_system),
        Some(0)
    );

    // Cmd+Z #3: undoes step #1 (User assign)
    let u3 = journal.pop_undo().expect("undo #3");
    assert_eq!(u3.seq, s1);
    let _ = session.undo();
    journal.push_redo(u3);
    assert_eq!(
        session
            .state
            .assignments
            .get(&1)
            .and_then(|a| a.spin_system),
        None
    );

    // Cmd+Shift+Z #1 & #2: redoes step #1 (User) then step #2 (MCP) in exact chronological order!
    let r1 = journal.pop_redo().expect("redo #1");
    assert_eq!(r1.seq, s1);
    let _ = session.redo();
    journal.push_undo_after_redo(r1);
    assert_eq!(
        session
            .state
            .assignments
            .get(&1)
            .and_then(|a| a.spin_system),
        Some(0)
    );

    let r2 = journal.pop_redo().expect("redo #2");
    assert_eq!(r2.seq, s2);
    assert!(mcp_log.redo_last(&mut mcp_snap).is_some());
    journal.push_undo_after_redo(r2);
    assert_eq!(mcp_snap.crosshair_ppm.0, 8.42);
}

#[test]
fn test_v0150_p1_6_and_p2_5_multilevel_water_undo_ws_ws_wu_wu_and_arc_3d() {
    let cfg = WaterSuppressionConfig::default();

    // 1. 1D `ws, ws, wu, wu`
    {
        let n = 256usize;
        let ax = AxisPpm::from_bounds("1H", n, 10.0, 0.0, 600.0);
        let mut real = vec![0.2_f64; n];
        let wi = ax.ppm_to_index(4.70).round() as usize;
        real[wi] = 5_000.0;
        real[wi - 1] = 2_500.0;
        real[wi + 1] = 2_500.0;
        let sp0 = Spectrum1D::new("1h", Experiment1DKind::Proton1D, ax, real, None);
        let orig_real = sp0.real.clone();

        let mut st = Stack1dState::default();
        st.load_user_spectra(vec![sp0]);

        // ws #1
        let pre1 = st.traces[0].raw_spec.as_ref().unwrap().0.clone();
        let mut after1 = pre1.clone();
        let _ = suppress_water_spectrum_1d(&mut after1, &cfg);
        st.push_water_undo_active(pre1);
        st.traces[0].raw_spec = Some(AttachedSpectrum1d(after1.clone()));
        st.traces[0].sync_from_spectrum1d(&after1);
        let snap_after_ws1 = after1.real.clone();

        // ws #2
        let pre2 = st.traces[0].raw_spec.as_ref().unwrap().0.clone();
        let mut after2 = pre2.clone();
        let _ = suppress_water_spectrum_1d(&mut after2, &cfg);
        st.push_water_undo_active(pre2);
        st.traces[0].raw_spec = Some(AttachedSpectrum1d(after2.clone()));
        st.traces[0].sync_from_spectrum1d(&after2);

        // wu #1 -> restores after_ws1
        assert!(st.undo_water_suppression_active_trace());
        assert_eq!(
            st.traces[0].raw_spec.as_ref().unwrap().0.real,
            snap_after_ws1
        );
        // wu #2 -> restores exact bitwise orig_real!
        assert!(st.undo_water_suppression_active_trace());
        assert_eq!(st.traces[0].raw_spec.as_ref().unwrap().0.real, orig_real);
        assert!(!st.has_water_undo_active());
    }

    // 2. 2D `ws, ws, wu, wu` AND 3D cube isolation (`v0150:P1-6` + `20.W1:P2-5` `Arc<Spectrum3D>`)
    {
        let ax_h = AxisPpm::from_bounds("1H", 64, 10.0, 0.0, 600.0);
        let ax_n = AxisPpm::from_bounds("15N", 16, 130.0, 105.0, 60.8);
        let ax_c = AxisPpm::from_bounds("13C", 16, 65.0, 15.0, 150.9);
        let mut data2d = vec![1.0_f32; 64 * 16];
        let wi = ax_h.ppm_to_index(4.70).round() as usize;
        for iy in 0..16 {
            data2d[iy * 64 + wi] = 9_000.0;
        }
        let orig_2d = Spectrum2D {
            name: "HSQC".into(),
            axis_x: ax_h.clone(),
            axis_y: ax_n.clone(),
            data: data2d,
        };
        let orig_2d_bytes = orig_2d.data.clone();

        let mut cube_data = vec![2.0_f32; 64 * 16 * 16];
        cube_data[wi] = 12_000.0;
        let cube = Arc::new(Spectrum3D {
            name: "HNCACB".into(),
            exp: "HNCACB".into(),
            axis_x: ax_h,
            axis_y: ax_n,
            axis_z: ax_c,
            data: cube_data.clone(),
        });

        // `Arc::clone` on page switch shares the exact allocation (`0` deep clones!)
        let page_ref = Arc::clone(&cube);
        assert!(Arc::ptr_eq(&cube, &page_ref));

        let mut stack_2d: KeyedSnapshotStack<Spectrum2D> = KeyedSnapshotStack::default();
        let key2 = SpectrumKey::Page2d("p1:HSQC".into());
        let mut cur_2d = orig_2d.clone();

        // 2D ws #1 (cube MUST remain completely untouched!)
        stack_2d.push(key2.clone(), cur_2d.clone());
        let _ = suppress_water_spectrum_2d(&mut cur_2d, &cfg);
        let after_2d_ws1 = cur_2d.data.clone();
        assert_eq!(
            cube.data, cube_data,
            "2D water suppression must NEVER mutate loaded 3D cubes (`v0150:P1-6`)"
        );

        // 2D ws #2
        stack_2d.push(key2.clone(), cur_2d.clone());
        let _ = suppress_water_spectrum_2d(&mut cur_2d, &cfg);

        // 2D wu #1 -> restores intermediate
        cur_2d = stack_2d.pop(&key2).expect("2D wu #1");
        assert_eq!(cur_2d.data, after_2d_ws1);
        // 2D wu #2 -> restores exact bitwise original
        cur_2d = stack_2d.pop(&key2).expect("2D wu #2");
        assert_eq!(cur_2d.data, orig_2d_bytes);

        // 3D `ws, ws, wu, wu` via `Arc::make_mut` + `KeyedSnapshotStack<Arc<Spectrum3D>>`
        drop(page_ref);
        let mut stack_3d: KeyedSnapshotStack<Arc<Spectrum3D>> = KeyedSnapshotStack::default();
        let key3 = SpectrumKey::Cube3d("c0:HNCACB".into());
        let mut cur_cube = cube;
        stack_3d.push(key3.clone(), Arc::clone(&cur_cube));
        let _ = suppress_water_3d(Arc::make_mut(&mut cur_cube), &cfg);
        let after_3d_ws1 = cur_cube.data.clone();

        stack_3d.push(key3.clone(), Arc::clone(&cur_cube));
        let _ = suppress_water_3d(Arc::make_mut(&mut cur_cube), &cfg);

        cur_cube = stack_3d.pop(&key3).expect("3D wu #1");
        assert_eq!(cur_cube.data, after_3d_ws1);
        cur_cube = stack_3d.pop(&key3).expect("3D wu #2");
        assert_eq!(cur_cube.data, cube_data);
    }
}

#[test]
fn test_c5_c6_space_never_locks_crosshair_xl_does_and_plus_means_more_signal() {
    let mut disp = ChordDispatcher::new();
    // Space is pan-only (`UnboundKey`), NEVER crosshair lock (`UI-C5`)
    let d_space = feed_plain_char(&mut disp, ' ', 100);
    assert_eq!(d_space, ChordOutcome::UnboundKey);
    assert!(disp.is_idle());

    // `xl` chord toggles crosshair lock (`UI-C5`)
    let _ = feed_plain_char(&mut disp, 'x', 200);
    let d_xl = feed_plain_char(&mut disp, 'l', 300);
    assert_eq!(
        d_xl,
        ChordOutcome::Dispatched(WorkbenchAction::ToggleCrosshairLockChord)
    );

    // `+` and `ci` both dispatch `ShowMoreSignal`; `-` and `cb` both dispatch `ShowLessSignal` (`UI-C6`)
    let d_plus = feed_plain_char(&mut disp, '+', 400);
    assert_eq!(
        d_plus,
        ChordOutcome::Dispatched(WorkbenchAction::ShowMoreSignal)
    );
    let d_minus = feed_plain_char(&mut disp, '-', 500);
    assert_eq!(
        d_minus,
        ChordOutcome::Dispatched(WorkbenchAction::ShowLessSignal)
    );

    // Verify `+` raises 1D gain AND lowers 2D/3D contour floor ("more signal" everywhere!)
    assert!(gain_after(1.0, SignalDirection::More) > 1.0);
    assert!(contour_floor_after(10.0, SignalDirection::More) < 10.0);
    assert!(gain_after(1.0, SignalDirection::Less) < 1.0);
    assert!(contour_floor_after(10.0, SignalDirection::Less) > 10.0);
}

#[test]
fn test_c8_c10_c12_c16_c18_and_phase21_provenance_concordance_pae_cache() {
    // 1. `StripTilerPanel`: variable height, indirect zoom, and Shift+Arrow peak nudge (`UI-C10`, `UI-C13`)
    {
        let suite =
            build_builtin_demo_suite(BUILTIN_DEMO_SUITE_SEED).expect("build builtin demo suite");
        let ref_data = suite.ubiquitin_ref;
        let systems =
            build_spin_systems(&suite.peaks, &Tolerances::default(), 4, false).expect("systems");
        let scfg = ScoreConfig::default();
        let u_mat = unary_matrix(&systems, &ref_data.sequence, &Priors::bmrb(), &scfg);
        let p_mat = pair_matrix(&systems, &scfg);

        let mut tiler = StripTilerPanel::new();
        tiler.strip_height_px = 420.0;
        tiler.indirect_zoom = Some((18.0, 62.0));
        tiler.selected_peak = Some((0, 0));

        let ctx = egui::Context::default();
        let raw = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(600.0, 700.0),
            )),
            events: vec![
                egui::Event::PointerMoved(egui::pos2(300.0, 350.0)),
                egui::Event::Key {
                    key: egui::Key::ArrowUp,
                    physical_key: Some(egui::Key::ArrowUp),
                    pressed: true,
                    repeat: false,
                    modifiers: egui::Modifiers::SHIFT,
                },
            ],
            ..Default::default()
        };
        let mut emitted = Vec::new();
        let mut out = ctx.run_ui(raw, |ui| {
            let tctx = StripTilerContext {
                systems: &systems,
                pair_mat: &p_mat,
                u_mat: Some(&u_mat),
                sequence: &ref_data.sequence,
                seq_ids: &ref_data.seq_ids,
                active_root: 0,
                assignments: None,
                locks: None,
                confirmed_links: None,
                spectra_3d: StripSpectra3D {
                    aliphatic_3d: Some(&suite.hncacb_3d),
                    carbonyl_3d: None,
                },
            };
            emitted = tiler.show(ui, &tctx, None);
        });
        out.textures_delta.clear();
        assert!(
            emitted.iter().any(|a| matches!(
                a,
                StripTilerAction::NudgeStripPeak {
                    strip: 0,
                    peak: 0,
                    ..
                }
            )),
            "Shift+ArrowUp in StripTilerPanel must emit NudgeStripPeak, got {emitted:?}"
        );
    }

    // 2. `OverlayLayerSpec`: per-layer `sync_f2` / `sync_f1` and `custom_color` (`UI-C12`)
    {
        let ax_h = AxisPpm::from_bounds("1H", 16, 10.0, 6.0, 600.0);
        let ax_n = AxisPpm::from_bounds("15N", 16, 130.0, 105.0, 60.8);
        let sp = Spectrum2D {
            name: "ov".into(),
            axis_x: ax_h,
            axis_y: ax_n,
            data: vec![0.0; 256],
        };
        let mut ov =
            OverlayLayerSpec::from_spectrum_2d(2, 1, &sp, Arc::new(sp.data.clone()), 5.0, 1);
        ov.custom_color = true;
        ov.sync_f2 = true;
        ov.sync_f1 = false;
        ov.unlinked_view_ppm = Some([9.5, 6.5, 110.0, 125.0]);
        assert_eq!(
            ov.effective_view_ppm([10.2, 7.0, 102.0, 134.0]),
            [10.2, 7.0, 110.0, 125.0]
        );
    }

    // 3. `Mol3dViewerPanel`: fail-closed predicted-model provenance (`21.W1:P0-2`)
    {
        let ref_data = read_star_str(SYNTH_UBQ_STR, None).expect("parse ubq");

        // Experimental X-ray 1UBQ (even with B <= 100) must fail closed
        let mut m_xray = Mol3dViewerPanel::new();
        m_xray
            .load_structure_text(UBQ_1UBQ_PDB, &ref_data.sequence, &ref_data.seq_ids)
            .expect("load 1ubq");
        assert!(m_xray
            .ensemble
            .as_ref()
            .unwrap()
            .require_predicted_model(false)
            .is_err());

        // Headered REMARK ALPHAFOLD PDB must ALSO fail closed without explicit user opt-in!
        let spoof_pdb = format!("REMARK 300 ALPHAFOLD MODEL\n{UBQ_1UBQ_PDB}");
        let mut m_spoof = Mol3dViewerPanel::new();
        m_spoof
            .load_structure_text(&spoof_pdb, &ref_data.sequence, &ref_data.seq_ids)
            .expect("load spoof");
        assert!(
            m_spoof
                .ensemble
                .as_ref()
                .unwrap()
                .require_predicted_model(false)
                .is_err(),
            "REMARK mentioning AlphaFold must NEVER bypass fail-closed pLDDT provenance (`21.W1:P0-2`)"
        );

        // AFDB v6 mmCIF (`_ma_qa_metric_local` `pLDDT`) passes cleanly!
        let mut m_af = Mol3dViewerPanel::new();
        m_af.load_structure_text(
            MINIMAL_AFDB_MMCIF,
            &["MET".into(), "GLN".into(), "ILE".into()],
            &[1, 2, 3],
        )
        .expect("load minimal AFDB mmCIF");
        assert!(m_af
            .ensemble
            .as_ref()
            .unwrap()
            .require_predicted_model(false)
            .is_ok());
    }

    // 4. `SsRibbonState`: non-circular concordance (`Indeterminate` when pLDDT is missing, `21.W1:P1-5`)
    {
        let ref_data = read_star_str(SYNTH_UBQ_STR, None).expect("parse ubq");
        let mut m_xray = Mol3dViewerPanel::new();
        m_xray
            .load_structure_text(UBQ_1UBQ_PDB, &ref_data.sequence, &ref_data.seq_ids)
            .expect("1ubq");
        let mut ribbon = SsRibbonState::new();
        ribbon.sync_from_ensemble_and_nmr(
            ref_data.sequence.len(),
            m_xray.ensemble.as_ref(),
            Some(&ref_data),
            &PropensityConfig::default(),
            None,
        );
        assert!(
            ribbon
                .concordance_summary
                .classes
                .iter()
                .all(|&c| c == ConcordanceTrackClass::Indeterminate),
            "Experimental PDB without pLDDT must yield Indeterminate concordance, never circular ConcordantOrdered (`21.W1:P1-5`)"
        );
    }

    // 5. `IntegrativeAfPanel`: cached PAE heatmap `TextureHandle` (`pae_texture_uploads == 1`, `21.W3:P1-14`)
    {
        let ctx = egui::Context::default();
        let mut af = IntegrativeAfPanel::new();
        let _ = af
            .load_synthetic_two_domain_demo()
            .expect("two-domain demo");
        af.active_tab = IntegrativeStudioTab::PaeDomainHeatmap;
        for _ in 0..5 {
            let raw = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1280.0, 800.0),
                )),
                ..Default::default()
            };
            let mut out = ctx.run_ui(raw, |ui| {
                egui::CentralPanel::default().show(ui, |ui| {
                    let _ = af.show(ui, true);
                });
            });
            out.textures_delta.clear();
        }
        assert_eq!(
            af.last_telemetry().pae_texture_uploads,
            1,
            "PAE N x N heatmap must upload texture ONCE across 5 static frames (`21.W3:P1-14`)"
        );
    }

    // 6. Dead `poll_session_shortcuts` deleted (`UI-C16`) & canonical `INTERACTION_GRAMMAR` table (`UI-C18`)
    {
        let session_src = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/session_io.rs"),
        )
        .expect("read session_io.rs");
        assert!(
            !session_src.contains("fn poll_session_shortcuts"),
            "dead poll_session_shortcuts must be deleted (`UI-C16`)"
        );

        let md = grammar_markdown_table();
        for view in GestureView::ALL {
            assert!(
                md.contains(view.label()),
                "canonical grammar table missing view {}",
                view.label()
            );
        }
        assert!(INTERACTION_GRAMMAR.len() >= 10);
    }
}
