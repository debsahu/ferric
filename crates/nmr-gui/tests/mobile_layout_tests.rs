//! Headless responsive mobile/tablet/desktop layout and multi-touch gesture tests (`T17.5`).
//!
//! # Clean-Room Provenance
//! Original clean-room headless `egui::Context` layout and multi-touch verification suite testing
//! dynamic viewport breakpoints (`CompactMobile < 680 px`, `TabletSplit 680..1024 px`,
//! `DesktopFull >= 1024 px`), touch-friendly bottom navigation across `ProteinNmr` (5 views) and
//! `ChemStudio` (4 views), viewport area guarantees on `375x667` / `390x844` / `412x915` phones,
//! single-dock tablet enforcement on `768x1024`, dual-dock desktop preservation on `1440x900`, and
//! deterministic two-finger pinch-to-zoom & pan on both the 2D `HSQC` spectrum canvas and 3D
//! `1UBQ` structure viewport.

use std::sync::Arc;

use eframe::egui;
use nmr_core::octet_render::{compile_scene_primitives, DocumentStyle, JournalStyle, RenderScene};
use nmr_core::priors::Priors;
use nmr_core::solver::{pair_matrix, unary_matrix, Matrix2D, ScoreConfig};
use nmr_gui::contour_painter::{
    apply_pinch_zoom_and_pan_2d, clamp_view_ppm_2d, SyncCrosshair, ViewBounds2dLimits,
};
use nmr_gui::demo_preset::{build_builtin_demo_suite, BuiltinDemoSuite, BUILTIN_DEMO_SUITE_SEED};
use nmr_gui::dosy_panel::render_dosy_panel;
use nmr_gui::dynamics_panel::DynamicsStudioPanel;
use nmr_gui::relax_panel::RelaxPanelState;
use nmr_gui::stack1d_panel::render_stack1d_panel;
use nmr_gui::strip_tiler::{StripSpectra3D, StripTilerContext, StripTilerPanel};
use nmr_gui::workflow_bar::{
    draw_workflow_stepper, render_responsive_workbench, HostWorkspaceMode, MobileChemTab,
    MobileDynamicsSubTab, MobileProteinTab, MobileWorkspaceFamily, ResponsiveBreakpoint,
    ResponsiveFrameMetrics, ResponsiveShellState, ResponsiveWorkbenchHost, TabletActiveDock,
    WorkbenchAction, WorkspaceReadiness, MOBILE_TOUCH_MIN_HEIGHT,
};

struct DemoWorkbenchHost {
    demo: BuiltinDemoSuite,
    mode: HostWorkspaceMode,
    show_mol3d_dock: bool,
    view_ppm: [f32; 4],
    crosshair: SyncCrosshair,
    suppress_2d_pan_until_release: bool,
    strip_tiler: StripTilerPanel,
    u_mat: Matrix2D,
    p_mat: Matrix2D,
    relax_panel: RelaxPanelState,
    dynamics_panel: DynamicsStudioPanel,
    octet_scene: RenderScene,
    octet_style: DocumentStyle,
    octet_show_right_drawer: bool,
    status: String,
    demo_reload_count: usize,
}

impl DemoWorkbenchHost {
    fn from_builtin_demo() -> Self {
        let demo =
            build_builtin_demo_suite(BUILTIN_DEMO_SUITE_SEED).expect("build_builtin_demo_suite");
        let priors = Priors::provisional();
        let scfg = ScoreConfig::default();
        let u_mat = unary_matrix(&demo.systems, &demo.ubiquitin_ref.sequence, &priors, &scfg);
        let p_mat = pair_matrix(&demo.systems, &scfg);

        let view_ppm = [
            demo.hsqc_2d.axis_x.max_ppm() as f32,
            demo.hsqc_2d.axis_x.min_ppm() as f32,
            demo.hsqc_2d.axis_y.min_ppm() as f32,
            demo.hsqc_2d.axis_y.max_ppm() as f32,
        ];
        let mut dynamics_panel = DynamicsStudioPanel::new();
        let _ = dynamics_panel.ensure_demo_exchange_and_csp(
            &demo.ubiquitin_ref.sequence,
            &demo.ubiquitin_ref.seq_ids,
        );
        let status = demo.status_summary();

        let mol_pep =
            nmr_core::octet_mol::build_peptide_2d("APY").expect("build APY demo peptide 2D");
        let octet_scene = RenderScene::from(&mol_pep);

        Self {
            demo,
            mode: HostWorkspaceMode::ProteinNmr,
            show_mol3d_dock: true,
            view_ppm,
            crosshair: SyncCrosshair::default(),
            suppress_2d_pan_until_release: false,
            strip_tiler: StripTilerPanel::new(),
            u_mat,
            p_mat,
            relax_panel: RelaxPanelState::new(),
            dynamics_panel,
            octet_scene,
            octet_style: JournalStyle::Acs1996.to_style(),
            octet_show_right_drawer: true,
            status,
            demo_reload_count: 0,
        }
    }
}

impl ResponsiveWorkbenchHost for DemoWorkbenchHost {
    fn host_workspace_mode(&self) -> HostWorkspaceMode {
        self.mode
    }

    fn set_host_workspace_mode(&mut self, mode: HostWorkspaceMode) {
        self.mode = mode;
    }

    fn workspace_readiness(&self) -> WorkspaceReadiness {
        WorkspaceReadiness {
            sequence_loaded: true,
            sequence_len: self.demo.ubiquitin_ref.sequence.len(),
            spectra_2d_count: 1,
            spectra_3d_count: 2,
            picked_peaks_count: self.demo.peaks.len(),
            spin_systems_count: self.demo.systems.len(),
            assigned_residues_count: 68,
            high_confidence_count: 64,
            has_structure_or_dynamics: true,
            ..WorkspaceReadiness::default()
        }
    }

    fn active_residue_badge(&self) -> Option<String> {
        Some(format!("{} aa", self.demo.ubiquitin_ref.sequence.len()))
    }

    fn status_banner_text(&self) -> &str {
        &self.status
    }

    fn trigger_builtin_demo(&mut self) {
        self.demo_reload_count += 1;
    }

    fn dispatch_chrome_action(&mut self, _ctx: &egui::Context, action: WorkbenchAction) {
        self.status = format!("Dispatched {:?}", action);
    }

    fn show_mol3d_dock(&self) -> bool {
        self.show_mol3d_dock
    }

    fn draw_desktop_mode_header(
        &mut self,
        ui: &mut egui::Ui,
        tablet_dock: Option<&mut TabletActiveDock>,
    ) {
        ui.horizontal_wrapped(|ui| {
            ui.strong("FERRIC");
            ui.selectable_value(&mut self.mode, HostWorkspaceMode::ProteinNmr, "Protein");
            ui.selectable_value(&mut self.mode, HostWorkspaceMode::OctetChem, "Octet");
            ui.selectable_value(&mut self.mode, HostWorkspaceMode::Dosy, "DOSY");
            ui.selectable_value(&mut self.mode, HostWorkspaceMode::Metabolomics, "Metab");
            if let Some(dock) = tablet_dock {
                ui.selectable_value(dock, TabletActiveDock::StripsRight, "Strips");
                ui.selectable_value(dock, TabletActiveDock::Mol3dLeft, "3D");
                ui.selectable_value(dock, TabletActiveDock::Hidden, "Full");
            }
        });
    }

    fn draw_desktop_status_bar(&mut self, ui: &mut egui::Ui, _now_ms: u64) {
        ui.horizontal(|ui| {
            ui.small(&self.status);
        });
    }

    fn draw_protein_top_controls_and_ribbon(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(format!(
                "Ubiquitin ({} aa, {} spin systems)",
                self.demo.ubiquitin_ref.sequence.len(),
                self.demo.systems.len()
            ));
        });
    }

    fn draw_protein_desktop_central(&mut self, ui: &mut egui::Ui) -> egui::Rect {
        ui.horizontal(|ui| {
            ui.label("2D HSQC Contours");
        });
        ui.separator();
        self.draw_hsqc_2d_view(ui)
    }

    fn draw_hsqc_2d_view(&mut self, ui: &mut egui::Ui) -> egui::Rect {
        let is_mobile = ui.ctx().content_rect().width() < 680.0;
        if is_mobile {
            egui::ScrollArea::horizontal()
                .id_salt("test_hsqc_mobile_bar")
                .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysHidden)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().interact_size.y = MOBILE_TOUCH_MIN_HEIGHT;
                        let _ = ui.add(
                            egui::Button::new("Levels")
                                .min_size(egui::vec2(64.0, MOBILE_TOUCH_MIN_HEIGHT)),
                        );
                        let _ = ui.add(
                            egui::Button::new("Fit")
                                .min_size(egui::vec2(44.0, MOBILE_TOUCH_MIN_HEIGHT)),
                        );
                    });
                });
        }

        let avail = ui.available_size();
        let (rect, response) = ui.allocate_exact_size(
            egui::vec2(avail.x.max(160.0), (avail.y - 4.0).max(160.0)),
            egui::Sense::click_and_drag(),
        );
        let limits = ViewBounds2dLimits::from_spectrum_2d(&self.demo.hsqc_2d);
        if !ui.input(|i| i.pointer.any_down()) {
            self.suppress_2d_pan_until_release = false;
        }
        let mut mt_active = false;
        if let Some(mt) = ui.ctx().input(|i| i.multi_touch()) {
            if mt.num_touches >= 2
                && (rect.contains(mt.start_pos)
                    || rect.contains(mt.center_pos)
                    || response.hovered()
                    || response.is_pointer_button_down_on())
            {
                self.suppress_2d_pan_until_release = true;
                self.view_ppm = apply_pinch_zoom_and_pan_2d(
                    self.view_ppm,
                    rect,
                    mt.center_pos,
                    mt.zoom_delta,
                    mt.translation_delta,
                    limits,
                );
                mt_active = true;
            }
        }
        if !mt_active && !self.suppress_2d_pan_until_release && response.dragged() {
            let delta = response.drag_delta();
            let dh = -delta.x / rect.width().max(1.0) * (self.view_ppm[1] - self.view_ppm[0]);
            let dn = -delta.y / rect.height().max(1.0) * (self.view_ppm[3] - self.view_ppm[2]);
            self.view_ppm = clamp_view_ppm_2d(
                [
                    self.view_ppm[0] + dh,
                    self.view_ppm[1] + dh,
                    self.view_ppm[2] + dn,
                    self.view_ppm[3] + dn,
                ],
                limits,
            );
        }
        ui.painter()
            .rect_filled(rect, 2.0, egui::Color32::from_rgb(14, 18, 25));
        rect
    }

    fn draw_strips_inspector_view(&mut self, ui: &mut egui::Ui) {
        if self.demo.systems.is_empty() {
            return;
        }
        let tiler_ctx = StripTilerContext {
            systems: &self.demo.systems,
            pair_mat: &self.p_mat,
            u_mat: Some(&self.u_mat),
            sequence: &self.demo.ubiquitin_ref.sequence,
            seq_ids: &self.demo.ubiquitin_ref.seq_ids,
            active_root: 0,
            assignments: Some(&self.demo.truth),
            locks: None,
            confirmed_links: None,
            spectra_3d: StripSpectra3D {
                aliphatic_3d: Some(&self.demo.hncacb_3d),
                carbonyl_3d: Some(&self.demo.hnco_3d),
            },
        };
        let _ = self
            .strip_tiler
            .show(ui, &tiler_ctx, Some(&mut self.crosshair));
    }

    fn draw_mol3d_structure_view(&mut self, ui: &mut egui::Ui) -> egui::Rect {
        let _ = self.demo.mol3d_panel.ui(
            ui,
            None,
            &self.demo.ubiquitin_ref.sequence,
            &self.demo.ubiquitin_ref.seq_ids,
            Some(&self.demo.ubiquitin_ref),
        );
        self.demo
            .mol3d_panel
            .last_canvas_rect
            .unwrap_or_else(|| ui.min_rect())
    }

    fn draw_relax_s2_view(&mut self, ui: &mut egui::Ui) {
        let _ = self.relax_panel.ui(
            ui,
            &self.demo.ubiquitin_ref.sequence,
            &self.demo.ubiquitin_ref.seq_ids,
            Some(0),
            Some(self.demo.pdb_text),
        );
    }

    fn draw_dynamics_exchange_view(&mut self, ui: &mut egui::Ui) {
        let _ = self.dynamics_panel.show(ui);
    }

    fn draw_stack1d_view(&mut self, ui: &mut egui::Ui) {
        render_stack1d_panel(ui, &mut self.demo.stack1d_panel);
    }

    fn draw_mobile_seq_and_tools_view(
        &mut self,
        ui: &mut egui::Ui,
        readiness: &WorkspaceReadiness,
    ) -> Vec<WorkbenchAction> {
        let actions = draw_workflow_stepper(ui, readiness);
        ui.separator();
        egui::ScrollArea::horizontal()
            .id_salt("test_seq_ribbon")
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    for (idx, aa) in self.demo.ubiquitin_ref.sequence.iter().enumerate() {
                        let _ =
                            ui.button(egui::RichText::new(format!("{aa}{}", idx + 1)).size(11.0));
                    }
                });
            });
        actions
    }

    fn draw_octet_desktop_or_tablet(
        &mut self,
        ui: &mut egui::Ui,
        is_tablet: bool,
    ) -> (egui::Rect, Option<egui::Rect>, Option<egui::Rect>) {
        if is_tablet {
            self.octet_show_right_drawer = false;
        }
        let l = egui::Panel::left("test_octet_left")
            .default_size(138.0)
            .min_size(124.0)
            .show(ui, |ui| {
                ui.label("Tools");
            });
        let r = if self.octet_show_right_drawer {
            Some(
                egui::Panel::right("test_octet_right")
                    .default_size(300.0)
                    .show(ui, |ui| {
                        ui.label("Props");
                    })
                    .response
                    .rect,
            )
        } else {
            None
        };
        let c = egui::CentralPanel::default().show(ui, |ui| {
            let _ = self.draw_mobile_octet_2d_view(ui);
        });
        (c.response.rect, Some(l.response.rect), r)
    }

    fn draw_mobile_octet_2d_view(&mut self, ui: &mut egui::Ui) -> egui::Rect {
        egui::ScrollArea::horizontal()
            .id_salt("test_mobile_octet_bar")
            .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysHidden)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.spacing_mut().interact_size.y = MOBILE_TOUCH_MIN_HEIGHT;
                    let _ = ui.add(
                        egui::Button::new("Clean 2D")
                            .min_size(egui::vec2(78.0, MOBILE_TOUCH_MIN_HEIGHT)),
                    );
                });
            });
        let avail = ui.available_size();
        let (rect, _) = ui.allocate_exact_size(
            egui::vec2(avail.x.max(160.0), (avail.y - 4.0).max(160.0)),
            egui::Sense::click_and_drag(),
        );
        let compiled = compile_scene_primitives(&self.octet_scene, &self.octet_style);
        assert!(!compiled.primitives.is_empty());
        ui.painter()
            .rect_filled(rect, 3.0, egui::Color32::from_rgb(252, 253, 255));
        rect
    }

    fn draw_mobile_chem_1d_qnmr_view(&mut self, ui: &mut egui::Ui) {
        render_stack1d_panel(ui, &mut self.demo.stack1d_panel);
    }

    fn draw_dosy_view(&mut self, ui: &mut egui::Ui) {
        let _ = render_dosy_panel(ui, &mut self.demo.dosy_panel);
    }

    fn draw_metabolomics_view(&mut self, ui: &mut egui::Ui) {
        self.demo.metabolomics_panel.ui(ui);
    }
}

fn run_frame_at_size(
    ctx: &egui::Context,
    width: f32,
    height: f32,
    events: Vec<egui::Event>,
    shell: &mut ResponsiveShellState,
    host: &mut DemoWorkbenchHost,
) -> ResponsiveFrameMetrics {
    let raw = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(width, height),
        )),
        events,
        ..Default::default()
    };
    let mut metrics = ResponsiveFrameMetrics::default();
    let mut out = ctx.run_ui(raw, |ui| {
        metrics = render_responsive_workbench(ui, 1_000, shell, host);
    });
    out.textures_delta.clear();
    metrics
}

#[test]
fn test_responsive_breakpoint_classification_and_dynamic_resize() {
    assert_eq!(
        ResponsiveBreakpoint::from_width(375.0),
        ResponsiveBreakpoint::CompactMobile
    );
    assert_eq!(
        ResponsiveBreakpoint::from_width(679.9),
        ResponsiveBreakpoint::CompactMobile
    );
    assert_eq!(
        ResponsiveBreakpoint::from_width(680.0),
        ResponsiveBreakpoint::TabletSplit
    );
    assert_eq!(
        ResponsiveBreakpoint::from_width(768.0),
        ResponsiveBreakpoint::TabletSplit
    );
    assert_eq!(
        ResponsiveBreakpoint::from_width(1023.9),
        ResponsiveBreakpoint::TabletSplit
    );
    assert_eq!(
        ResponsiveBreakpoint::from_width(1024.0),
        ResponsiveBreakpoint::DesktopFull
    );
    assert_eq!(
        ResponsiveBreakpoint::from_width(1440.0),
        ResponsiveBreakpoint::DesktopFull
    );

    let ctx = egui::Context::default();
    let mut shell = ResponsiveShellState::new();
    let mut host = DemoWorkbenchHost::from_builtin_demo();

    let m_desktop = run_frame_at_size(&ctx, 1440.0, 900.0, Vec::new(), &mut shell, &mut host);
    assert_eq!(m_desktop.breakpoint, ResponsiveBreakpoint::DesktopFull);
    assert_eq!(m_desktop.docked_side_panel_count(), 2);

    let m_tablet = run_frame_at_size(&ctx, 768.0, 1024.0, Vec::new(), &mut shell, &mut host);
    assert_eq!(m_tablet.breakpoint, ResponsiveBreakpoint::TabletSplit);
    assert_eq!(m_tablet.docked_side_panel_count(), 1);

    let m_phone = run_frame_at_size(&ctx, 375.0, 667.0, Vec::new(), &mut shell, &mut host);
    assert_eq!(m_phone.breakpoint, ResponsiveBreakpoint::CompactMobile);
    assert_eq!(m_phone.docked_side_panel_count(), 0);

    let m_back = run_frame_at_size(&ctx, 1440.0, 900.0, Vec::new(), &mut shell, &mut host);
    assert_eq!(m_back.breakpoint, ResponsiveBreakpoint::DesktopFull);
    assert_eq!(m_back.docked_side_panel_count(), 2);
}

#[test]
fn test_phone_protein_and_chemstudio_all_tabs_zero_panics_and_viewport_budget() {
    let ctx = egui::Context::default();
    let mut shell = ResponsiveShellState::new();
    let mut host = DemoWorkbenchHost::from_builtin_demo();

    let phone_viewports = [
        (375.0_f32, 667.0_f32, "iPhone SE"),
        (390.0_f32, 844.0_f32, "iPhone 15"),
        (412.0_f32, 915.0_f32, "Android Pixel"),
    ];

    for (w, h, device_name) in phone_viewports {
        // 1. All 5 ProteinNmr primary tabs (including all 3 Dyn / 1D sub-tabs)
        shell.mobile_family = MobileWorkspaceFamily::ProteinNmr;
        for tab in MobileProteinTab::ALL {
            shell.protein_tab = tab;
            let sub_list: &[MobileDynamicsSubTab] = if tab == MobileProteinTab::Dynamics1d {
                &MobileDynamicsSubTab::ALL
            } else {
                &[MobileDynamicsSubTab::RelaxationS2]
            };
            for &sub in sub_list {
                shell.dynamics_sub_tab = sub;
                // Run 2 frames so egui settles exact panel/scroll metrics
                let _ = run_frame_at_size(&ctx, w, h, Vec::new(), &mut shell, &mut host);
                let m = run_frame_at_size(&ctx, w, h, Vec::new(), &mut shell, &mut host);

                assert_eq!(m.breakpoint, ResponsiveBreakpoint::CompactMobile);
                assert_eq!(
                    m.docked_side_panel_count(),
                    0,
                    "{device_name} ({w}x{h}) {tab:?} must mount 0 side docks"
                );
                assert!(
                    m.top_header_rect.height() <= 44.0,
                    "{device_name} top header height {:.1} must be <= 44.0 px",
                    m.top_header_rect.height()
                );
                let nav_rect = m.bottom_nav_rect.expect("CompactMobile bottom nav bar");
                assert!(
                    nav_rect.height() >= 40.0,
                    "{device_name} bottom nav height {:.1} must be >= 40.0 px",
                    nav_rect.height()
                );
                assert!(
                    m.central_panel_rect.width() >= 340.0 && m.active_viz_rect.width() >= 340.0,
                    "{device_name} {tab:?} {sub:?}: active viz width {:.1} (central {:.1}) < 340.0 px",
                    m.active_viz_rect.width(),
                    m.central_panel_rect.width()
                );
                assert!(
                    m.active_viz_rect.width() >= 0.90 * w,
                    "{device_name} {tab:?}: active viz width {:.1} < 90% of {w} ({:.1})",
                    m.active_viz_rect.width(),
                    0.90 * w
                );
                assert!(
                    m.central_panel_rect.height() >= 430.0 && m.active_viz_rect.height() >= 430.0,
                    "{device_name} {tab:?} {sub:?}: active viz height {:.1} (central {:.1}) < 430.0 px",
                    m.active_viz_rect.height(),
                    m.central_panel_rect.height()
                );
                assert!(
                    m.active_viz_rect.height() >= 0.65 * h,
                    "{device_name} {tab:?} {sub:?}: active viz height {:.1} < 65% of {h} ({:.1})",
                    m.active_viz_rect.height(),
                    0.65 * h
                );
            }
        }

        // Verify [Menu] mobile quick-actions drawer renders cleanly on phone without panics
        shell.mobile_menu_open = true;
        let m_drawer = run_frame_at_size(&ctx, w, h, Vec::new(), &mut shell, &mut host);
        assert_eq!(m_drawer.docked_side_panel_count(), 0);
        shell.mobile_menu_open = false;

        // 2. All 4 ChemStudio mobile tabs (`Octet 2D Draw`, `1D Multiplets/qNMR`, `DOSY`, `Metabolomics`)
        shell.mobile_family = MobileWorkspaceFamily::ChemStudio;
        for ctab in MobileChemTab::ALL {
            shell.chem_tab = ctab;
            let _ = run_frame_at_size(&ctx, w, h, Vec::new(), &mut shell, &mut host);
            let m = run_frame_at_size(&ctx, w, h, Vec::new(), &mut shell, &mut host);

            assert_eq!(m.breakpoint, ResponsiveBreakpoint::CompactMobile);
            assert_eq!(m.docked_side_panel_count(), 0);
            assert!(m.top_header_rect.height() <= 44.0);
            assert!(m.bottom_nav_rect.expect("bottom nav").height() >= 40.0);
            assert!(
                m.active_viz_rect.width() >= 340.0 && m.active_viz_rect.width() >= 0.90 * w,
                "{device_name} ChemStudio {ctab:?}: width {:.1}",
                m.active_viz_rect.width()
            );
            assert!(
                m.active_viz_rect.height() >= 430.0 && m.active_viz_rect.height() >= 0.65 * h,
                "{device_name} ChemStudio {ctab:?}: height {:.1}",
                m.active_viz_rect.height()
            );
        }
    }
}

#[test]
fn test_tablet_768x1024_single_dock_cap_and_central_width() {
    let ctx = egui::Context::default();
    let mut shell = ResponsiveShellState::new();
    let mut host = DemoWorkbenchHost::from_builtin_demo();

    for &(w, h) in &[(768.0_f32, 1024.0_f32), (680.0_f32, 1024.0_f32)] {
        host.mode = HostWorkspaceMode::ProteinNmr;
        for dock in [
            TabletActiveDock::StripsRight,
            TabletActiveDock::Mol3dLeft,
            TabletActiveDock::Hidden,
        ] {
            shell.tablet_dock = dock;
            let _ = run_frame_at_size(&ctx, w, h, Vec::new(), &mut shell, &mut host);
            let m = run_frame_at_size(&ctx, w, h, Vec::new(), &mut shell, &mut host);

            assert_eq!(m.breakpoint, ResponsiveBreakpoint::TabletSplit);
            assert!(
                m.docked_side_panel_count() <= 1,
                "Tablet {w}x{h} {dock:?} must never dock left and right simultaneously"
            );
            match dock {
                TabletActiveDock::StripsRight => {
                    let r = m.right_dock_rect.expect("right strips dock");
                    assert!(m.left_dock_rect.is_none());
                    assert!(
                        r.width() >= 219.0 && r.width() <= 301.0,
                        "got {}",
                        r.width()
                    );
                }
                TabletActiveDock::Mol3dLeft => {
                    let l = m.left_dock_rect.expect("left mol3d dock");
                    assert!(m.right_dock_rect.is_none());
                    assert!(
                        l.width() >= 219.0 && l.width() <= 301.0,
                        "got {}",
                        l.width()
                    );
                }
                TabletActiveDock::Hidden => {
                    assert_eq!(m.docked_side_panel_count(), 0);
                }
            }
            assert!(
                m.central_panel_rect.width() >= 420.0,
                "Tablet {w}x{h} {dock:?} CentralPanel width {:.1} must remain >= 420.0 px",
                m.central_panel_rect.width()
            );
        }

        // Also verify OctetChem, Dosy, and Metabolomics on TabletSplit keep <= 1 dock and >= 420 px
        for mode in [
            HostWorkspaceMode::OctetChem,
            HostWorkspaceMode::Dosy,
            HostWorkspaceMode::Metabolomics,
        ] {
            host.mode = mode;
            let _ = run_frame_at_size(&ctx, w, h, Vec::new(), &mut shell, &mut host);
            let m = run_frame_at_size(&ctx, w, h, Vec::new(), &mut shell, &mut host);
            assert!(m.docked_side_panel_count() <= 1);
            assert!(
                m.central_panel_rect.width() >= 420.0,
                "Tablet {w}x{h} {mode:?} CentralPanel width {:.1} < 420.0 px",
                m.central_panel_rect.width()
            );
        }
    }
}

#[test]
fn test_desktop_1440x900_preserves_dual_docks() {
    let ctx = egui::Context::default();
    let mut shell = ResponsiveShellState::new();
    let mut host = DemoWorkbenchHost::from_builtin_demo();
    host.mode = HostWorkspaceMode::ProteinNmr;
    host.show_mol3d_dock = true;

    let _ = run_frame_at_size(&ctx, 1440.0, 900.0, Vec::new(), &mut shell, &mut host);
    let m = run_frame_at_size(&ctx, 1440.0, 900.0, Vec::new(), &mut shell, &mut host);

    assert_eq!(m.breakpoint, ResponsiveBreakpoint::DesktopFull);
    assert!(m.bottom_nav_rect.is_none());
    assert!(
        m.left_dock_rect.is_some(),
        "Desktop must preserve left 3D dock"
    );
    assert!(
        m.right_dock_rect.is_some(),
        "Desktop must preserve right strips dock"
    );
    assert_eq!(m.docked_side_panel_count(), 2);
    assert!(
        m.central_panel_rect.width() >= 550.0,
        "Desktop CentralPanel width {:.1}",
        m.central_panel_rect.width()
    );
    assert!(
        m.active_viz_rect.height() >= 500.0,
        "Desktop active 2D canvas height {:.1}",
        m.active_viz_rect.height()
    );
}

#[test]
fn test_2d_hsqc_pinch_zoom_helper_and_live_multitouch_events() {
    let mut host = DemoWorkbenchHost::from_builtin_demo();
    let limits = ViewBounds2dLimits::from_spectrum_2d(&host.demo.hsqc_2d);
    let rect = egui::Rect::from_min_size(egui::pos2(10.0, 50.0), egui::vec2(355.0, 510.0));
    let center = egui::pos2(160.0, 280.0);

    let initial = host.view_ppm;
    let init_h_span = initial[0] - initial[1];
    let init_n_span = initial[3] - initial[2];
    assert!(init_h_span > 2.0 && init_n_span > 15.0);

    // Anchor ppm before pinch
    let u = (center.x - rect.left()) / rect.width();
    let v = (center.y - rect.top()) / rect.height();
    let anchor_h_before = initial[0] + u * (initial[1] - initial[0]);
    let anchor_n_before = initial[2] + v * (initial[3] - initial[2]);

    // 1. Pinch-in (`zoom_delta = 2.0`) halves the ppm window while keeping gesture anchor fixed
    let zoomed_in =
        apply_pinch_zoom_and_pan_2d(initial, rect, center, 2.0, egui::Vec2::ZERO, limits);
    assert!(
        zoomed_in[0] > zoomed_in[1],
        "NMR 1H axis must stay left > right"
    );
    assert!(
        zoomed_in[2] < zoomed_in[3],
        "NMR 15N axis must stay top < bottom"
    );
    assert!(((zoomed_in[0] - zoomed_in[1]) - 0.5 * init_h_span).abs() < 1e-3);
    assert!(((zoomed_in[3] - zoomed_in[2]) - 0.5 * init_n_span).abs() < 1e-2);

    let anchor_h_after = zoomed_in[0] + u * (zoomed_in[1] - zoomed_in[0]);
    let anchor_n_after = zoomed_in[2] + v * (zoomed_in[3] - zoomed_in[2]);
    assert!((anchor_h_after - anchor_h_before).abs() < 1e-4);
    assert!((anchor_n_after - anchor_n_before).abs() < 1e-3);

    // 2. Extreme pinch-out & runaway translation clamp safely within physical spectrum limits
    let zoomed_out = apply_pinch_zoom_and_pan_2d(
        zoomed_in,
        rect,
        center,
        0.02,
        egui::vec2(50_000.0, -50_000.0),
        limits,
    );
    assert!(zoomed_out[0] <= limits.x_max_ppm + 1e-4);
    assert!(zoomed_out[1] >= limits.x_min_ppm - 1e-4);
    assert!(zoomed_out[2] >= limits.y_min_ppm - 1e-4);
    assert!(zoomed_out[3] <= limits.y_max_ppm + 1e-4);

    // 3. Drive real `egui::Event::Touch` two-finger pinch through `egui::Context` on `390x844`
    let ctx = egui::Context::default();
    let mut shell = ResponsiveShellState::new();
    shell.mobile_family = MobileWorkspaceFamily::ProteinNmr;
    shell.protein_tab = MobileProteinTab::Contours2d;

    let m0 = run_frame_at_size(&ctx, 390.0, 844.0, Vec::new(), &mut shell, &mut host);
    let c = m0.active_viz_rect.center();
    let span_before = host.view_ppm[0] - host.view_ppm[1];

    let d0 = egui::TouchDeviceId(1);
    let t1 = egui::TouchId(1);
    let t2 = egui::TouchId(2);
    let p1_start = egui::pos2(c.x - 30.0, c.y - 30.0);
    let p2_start = egui::pos2(c.x + 30.0, c.y + 30.0);
    // Finger 1 touches down (establishing `pointer.interact_pos`), followed by Finger 2
    let _ = run_frame_at_size(
        &ctx,
        390.0,
        844.0,
        vec![
            egui::Event::PointerMoved(p1_start),
            egui::Event::PointerButton {
                pos: p1_start,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
            egui::Event::Touch {
                device_id: d0,
                id: t1,
                phase: egui::TouchPhase::Start,
                pos: p1_start,
                force: None,
            },
        ],
        &mut shell,
        &mut host,
    );
    let _ = run_frame_at_size(
        &ctx,
        390.0,
        844.0,
        vec![egui::Event::Touch {
            device_id: d0,
            id: t2,
            phase: egui::TouchPhase::Start,
            pos: p2_start,
            force: None,
        }],
        &mut shell,
        &mut host,
    );

    let p1_move = egui::pos2(c.x - 65.0, c.y - 65.0);
    let p2_move = egui::pos2(c.x + 65.0, c.y + 65.0);
    let spread_events = vec![
        egui::Event::PointerMoved(p1_move),
        egui::Event::Touch {
            device_id: d0,
            id: t1,
            phase: egui::TouchPhase::Move,
            pos: p1_move,
            force: None,
        },
        egui::Event::Touch {
            device_id: d0,
            id: t2,
            phase: egui::TouchPhase::Move,
            pos: p2_move,
            force: None,
        },
    ];
    let _ = run_frame_at_size(&ctx, 390.0, 844.0, spread_events, &mut shell, &mut host);

    let span_after = host.view_ppm[0] - host.view_ppm[1];
    assert!(
        span_after < span_before * 0.70,
        "2-finger spread on 2D HSQC must zoom in deterministically (before {span_before:.3}, after {span_after:.3})"
    );
}

#[test]
fn test_3d_structure_pinch_zoom_pan_and_single_finger_orbit_isolation() {
    let mut host = DemoWorkbenchHost::from_builtin_demo();
    let ctx = egui::Context::default();
    let mut shell = ResponsiveShellState::new();
    shell.mobile_family = MobileWorkspaceFamily::ProteinNmr;
    shell.protein_tab = MobileProteinTab::Structure3d;

    let m0 = run_frame_at_size(&ctx, 390.0, 844.0, Vec::new(), &mut shell, &mut host);
    let c = m0.active_viz_rect.center();

    let dist_initial = host.demo.mol3d_panel.camera.distance;
    let quat_initial = host.demo.mol3d_panel.camera.orientation;
    let pan_initial = host.demo.mol3d_panel.camera.pan_xy;

    // 1. Two-finger pinch-in + translation pan through `egui::Context`
    // (Hover pointer at `p1_start` first so `egui`'s `pointer.interact_pos` is initialized when
    // both fingers touch down simultaneously, ensuring `0` single-finger orbit occurs before or
    // during the 2-finger pinch/pan!)
    let d0 = egui::TouchDeviceId(7);
    let t1 = egui::TouchId(10);
    let t2 = egui::TouchId(11);
    let p1_start = egui::pos2(c.x - 25.0, c.y - 25.0);
    let p2_start = egui::pos2(c.x + 25.0, c.y + 25.0);
    let _ = run_frame_at_size(
        &ctx,
        390.0,
        844.0,
        vec![egui::Event::PointerMoved(p1_start)],
        &mut shell,
        &mut host,
    );
    let _ = run_frame_at_size(
        &ctx,
        390.0,
        844.0,
        vec![
            egui::Event::PointerButton {
                pos: p1_start,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
            egui::Event::Touch {
                device_id: d0,
                id: t1,
                phase: egui::TouchPhase::Start,
                pos: p1_start,
                force: None,
            },
            egui::Event::Touch {
                device_id: d0,
                id: t2,
                phase: egui::TouchPhase::Start,
                pos: p2_start,
                force: None,
            },
        ],
        &mut shell,
        &mut host,
    );

    // Spread fingers 2x AND translate +24 px right / +18 px down simultaneously
    let p1_move = egui::pos2(c.x - 50.0 + 24.0, c.y - 50.0 + 18.0);
    let p2_move = egui::pos2(c.x + 50.0 + 24.0, c.y + 50.0 + 18.0);
    let _ = run_frame_at_size(
        &ctx,
        390.0,
        844.0,
        vec![
            egui::Event::PointerMoved(p1_move),
            egui::Event::Touch {
                device_id: d0,
                id: t1,
                phase: egui::TouchPhase::Move,
                pos: p1_move,
                force: None,
            },
            egui::Event::Touch {
                device_id: d0,
                id: t2,
                phase: egui::TouchPhase::Move,
                pos: p2_move,
                force: None,
            },
        ],
        &mut shell,
        &mut host,
    );

    assert!(
        host.demo.mol3d_panel.camera.distance < dist_initial * 0.70,
        "2-finger pinch spread must reduce 3D camera distance (was {dist_initial:.2}, now {:.2})",
        host.demo.mol3d_panel.camera.distance
    );
    let pan_shift = ((host.demo.mol3d_panel.camera.pan_xy[0] - pan_initial[0]).powi(2)
        + (host.demo.mol3d_panel.camera.pan_xy[1] - pan_initial[1]).powi(2))
    .sqrt();
    assert!(
        pan_shift > 0.1,
        "2-finger translation delta must pan 3D camera view-space pan_xy (shift={pan_shift:.4})"
    );
    assert_eq!(
        host.demo.mol3d_panel.camera.orientation, quat_initial,
        "2-finger pinch/pan must NOT fight single-finger trackball rotation"
    );

    // Release both fingers, then verify single-finger drag orbits the 3D trackball cleanly!
    let _ = run_frame_at_size(
        &ctx,
        390.0,
        844.0,
        vec![
            egui::Event::PointerButton {
                pos: p1_move,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            },
            egui::Event::Touch {
                device_id: d0,
                id: t1,
                phase: egui::TouchPhase::End,
                pos: p1_move,
                force: None,
            },
            egui::Event::Touch {
                device_id: d0,
                id: t2,
                phase: egui::TouchPhase::End,
                pos: p2_move,
                force: None,
            },
            egui::Event::PointerGone,
        ],
        &mut shell,
        &mut host,
    );
    // Idle frame with no pointer down resets `suppress_single_drag_until_release`
    let _ = run_frame_at_size(&ctx, 390.0, 844.0, Vec::new(), &mut shell, &mut host);

    let _ = run_frame_at_size(
        &ctx,
        390.0,
        844.0,
        vec![
            egui::Event::PointerMoved(c),
            egui::Event::PointerButton {
                pos: c,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
        ],
        &mut shell,
        &mut host,
    );
    let _ = run_frame_at_size(
        &ctx,
        390.0,
        844.0,
        vec![egui::Event::PointerMoved(egui::pos2(
            c.x + 48.0,
            c.y - 32.0,
        ))],
        &mut shell,
        &mut host,
    );
    assert_ne!(
        host.demo.mol3d_panel.camera.orientation, quat_initial,
        "Single-finger drag after multi-touch release must orbit 3D trackball cleanly"
    );

    // Unused Arc import check suppressed cleanly
    let _ = Arc::new(());
}
