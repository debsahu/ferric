//! Headless `egui` end-to-end integration, fail-closed diagnostics, non-destructive editing,
//! typography (`>= 11.0 px`), and zero-emoji / zero-panic source hygiene tests for
//! `nmr_gui::p22_tools_panel` (`T22.1`, `T22.13`, `T22.9`, `T22.10`).

use std::f64::consts::PI;
use std::time::{Duration, Instant};

use eframe::egui;
use nmr_core::artsy_rdc::ArtsyQualityFlag;
use nmr_core::deconv_nd::{DeconvDiagnostic, LineshapeKind, PeakSeed2D};
use nmr_core::fid::QuadMode;
use nmr_core::maxent_nus::MAXENT_LINEARITY_DISCLOSURE;
use nmr_core::pcs_transfer::PcsMatchStatus;
use nmr_core::spectrum::{AxisPpm, Spectrum2D};
use nmr_gui::p22_tools_panel::{
    ArtsySortColumn, DeconvDataSource, MaxEntDataSource, MaxEntLambdaUiMode, P22ToolTab,
    P22ToolsPanel, PcsCcrMode, DEFAULT_ARTSY_J_ISO_SIGMA_HZ,
};
use nmr_gui::workbench_mode::{apply_type_scale, audit_text_layout, MIN_FONT_PX};

const WINDOW_VIEWPORT: egui::Rect =
    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1280.0, 840.0));
const CONTENTS_VIEWPORT: egui::Rect =
    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1024.0, 760.0));

fn run_window_frame(
    ctx: &egui::Context,
    panel: &mut P22ToolsPanel,
    active_2d: Option<&Spectrum2D>,
    paper_white: bool,
) -> egui::FullOutput {
    apply_type_scale(ctx);
    let raw_input = egui::RawInput {
        screen_rect: Some(WINDOW_VIEWPORT),
        ..Default::default()
    };
    let mut out = ctx.run_ui(raw_input, |ui| {
        panel.show_window(ui.ctx(), active_2d, paper_white);
    });
    out.textures_delta.clear();
    out
}

fn run_contents_frame(
    ctx: &egui::Context,
    panel: &mut P22ToolsPanel,
    active_2d: Option<&Spectrum2D>,
    paper_white: bool,
) -> egui::FullOutput {
    apply_type_scale(ctx);
    let raw_input = egui::RawInput {
        screen_rect: Some(CONTENTS_VIEWPORT),
        ..Default::default()
    };
    let mut out = ctx.run_ui(raw_input, |ui| {
        panel.show_contents(ui, active_2d, paper_white);
    });
    out.textures_delta.clear();
    out
}

fn synthetic_active_spectrum_2d() -> Spectrum2D {
    let nx = 32usize;
    let ny = 32usize;
    let axis_x = AxisPpm::from_bounds("1H", nx, 8.60, 8.00, 600.0);
    let axis_y = AxisPpm::from_bounds("15N", ny, 123.0, 117.0, 60.8);
    let mut data = vec![0.0_f32; nx * ny];
    for iy in 0..ny {
        let f1 = axis_y.index_to_ppm(iy as f64);
        let u1 = 2.0 * (f1 - 120.1) / 0.55;
        let l1 = 1.0 / (1.0 + u1 * u1);
        for ix in 0..nx {
            let f2 = axis_x.index_to_ppm(ix as f64);
            let u2 = 2.0 * (f2 - 8.30) / 0.06;
            let l2 = 1.0 / (1.0 + u2 * u2);
            data[iy * nx + ix] = (100.0 * l1 * l2) as f32;
        }
    }
    Spectrum2D {
        name: "synthetic_hsqc".to_string(),
        axis_x,
        axis_y,
        data,
    }
}

#[test]
fn test_tab1_deconv_varpro_demo_active2d_and_fail_closed_diagnostics() {
    let ctx = egui::Context::default();
    let mut panel = P22ToolsPanel::new();
    panel.open_tab(P22ToolTab::Deconv2d3d);
    assert!(panel.open);
    assert_eq!(panel.active_tab, P22ToolTab::Deconv2d3d);

    // 1. 1-click overlapped doublet synthetic demo -> Converged with valid CRLB
    panel
        .deconv
        .load_and_run_synthetic_demo()
        .expect("synthetic VarPro doublet demo");
    let res = panel.deconv.result.as_ref().expect("VarPro result");
    assert_eq!(res.peaks.len(), 2);
    assert_eq!(res.diagnostic, DeconvDiagnostic::Converged);
    assert!(res.crlb_valid, "expected valid CRLB on clean doublet");
    assert!(res.reduced_chi2.is_finite() && res.reduced_chi2 > 0.0 && res.reduced_chi2 < 3.0);
    assert!(res.peaks[0].amplitude_se.is_some());
    assert!(res.peaks[1].amplitude_se.is_some());

    let out = run_window_frame(&ctx, &mut panel, None, false);
    let audit = audit_text_layout(&out.shapes, WINDOW_VIEWPORT);
    assert!(
        audit.min_font_px >= MIN_FONT_PX - 1e-3,
        "Deconv tab min font {} < {MIN_FONT_PX}",
        audit.min_font_px
    );

    // 2. Fail-closed ActiveBoundHit: constrain center shift tightly on displaced seed
    panel.deconv.use_center_shift_bound = true;
    panel.deconv.max_center_shift_fwhm = 0.05;
    panel.deconv.seeds[0].center_f1 -= 0.25;
    panel
        .deconv
        .run(None)
        .expect("VarPro runs and reports fail-closed diagnostic");
    let bound_res = panel.deconv.result.as_ref().expect("bound result");
    assert!(
        !bound_res.active_bounds.is_empty(),
        "expected active bound hit with tight center shift cap"
    );
    assert!(
        bound_res.diagnostic == DeconvDiagnostic::ActiveBoundHit
            || bound_res
                .all_diagnostics
                .contains(&DeconvDiagnostic::ActiveBoundHit),
        "expected ActiveBoundHit diagnostic, got {:?} ({:?})",
        bound_res.diagnostic,
        bound_res.all_diagnostics
    );
    assert!(
        !bound_res.crlb_valid,
        "CRLB must be fail-closed suppressed when an active bound is hit"
    );
    let _ = run_contents_frame(&ctx, &mut panel, None, true);

    // 3. Fail-closed ModelMisspecified: fit 1 peak to 2-peak overlapped doublet with known noise_sigma
    panel.deconv.use_center_shift_bound = false;
    panel.deconv.use_known_sigma = true;
    panel.deconv.known_noise_sigma = 0.005;
    panel.deconv.seeds = vec![PeakSeed2D {
        center_f1: 120.0,
        center_f2: 8.30,
        fwhm_f1: 0.65,
        fwhm_f2: 0.065,
        eta_f1: 0.5,
        eta_f2: 0.5,
    }];
    panel
        .deconv
        .run(None)
        .expect("VarPro runs and detects model misspecification");
    let mis_res = panel.deconv.result.as_ref().expect("misspecified result");
    assert!(
        matches!(
            mis_res.diagnostic,
            DeconvDiagnostic::ModelMisspecified { .. }
        ),
        "expected ModelMisspecified, got {:?}",
        mis_res.diagnostic
    );
    assert!(
        !mis_res.crlb_valid,
        "CRLB must be suppressed on ModelMisspecified"
    );
    assert!(mis_res.reduced_chi2 > 5.0);
    let _ = run_contents_frame(&ctx, &mut panel, None, false);

    // 4. Active 2D Spectrum ROI deconvolution
    let spec2d = synthetic_active_spectrum_2d();
    panel.deconv.sync_from_active_2d(&spec2d);
    assert_eq!(panel.deconv.source, DeconvDataSource::ActiveSpectrum2D);
    panel.deconv.lineshape = LineshapeKind::Lorentzian;
    panel.deconv.use_known_sigma = false;
    panel
        .deconv
        .run(Some(&spec2d))
        .expect("active 2D spectrum ROI VarPro fit");
    let spec_res = panel.deconv.result.as_ref().expect("active 2D fit result");
    assert_eq!(spec_res.peaks.len(), 1);
    assert!((spec_res.peaks[0].center_f1 - 120.1).abs() < 0.08);
    assert!((spec_res.peaks[0].center_f2 - 8.30).abs() < 0.01);
    let _ = run_window_frame(&ctx, &mut panel, Some(&spec2d), true);
}

#[test]
fn test_tab2_maxent_nus_states_tppi_fail_closed_echo_antiecho_and_disclosure() {
    let ctx = egui::Context::default();
    let mut panel = P22ToolsPanel::new();
    panel.open_tab(P22ToolTab::MaxEntNus);

    // 1. 1-click States Poisson-gap demo
    panel
        .maxent
        .load_and_run_synthetic_demo()
        .expect("2D MaxEnt States demo");
    let res = panel.maxent.result.as_ref().expect("MaxEnt result");
    assert!(res.kkt_residual_inf.is_finite() && res.kkt_residual_inf < 1e-3);
    assert!(res.duality_gap.is_finite());
    assert_eq!(res.linearity_note, MAXENT_LINEARITY_DISCLOSURE);
    assert!(!panel.maxent.schedule_text.is_empty());

    let out_dark = run_contents_frame(&ctx, &mut panel, None, false);
    let audit = audit_text_layout(&out_dark.shapes, CONTENTS_VIEWPORT);
    assert!(
        audit.min_font_px >= MIN_FONT_PX - 1e-3,
        "MaxEnt tab min font {} < {MIN_FONT_PX}",
        audit.min_font_px
    );

    // 2. Non-destructive switch to States-TPPI + Morozov discrepancy mode
    let saved_schedule = panel.maxent.schedule_text.clone();
    panel.maxent.quad_mode_axis2 = QuadMode::StatesTppi;
    panel.maxent.quad_mode_axis1 = QuadMode::States;
    panel.maxent.lambda_ui_mode = MaxEntLambdaUiMode::DiscrepancyPrinciple;
    panel
        .maxent
        .run()
        .expect("2D MaxEnt States-TPPI discrepancy run");
    assert_eq!(
        panel.maxent.schedule_text, saved_schedule,
        "schedule must remain non-destructively editable"
    );
    let tppi_res = panel.maxent.result.as_ref().expect("States-TPPI result");
    assert!(tppi_res.chi2_reduced.is_some());

    // 3. Fail-closed error surface on EchoAntiecho
    panel.maxent.quad_mode_axis2 = QuadMode::EchoAntiecho;
    let err = panel
        .maxent
        .run()
        .expect_err("EchoAntiecho must fail closed in MaxEnt");
    assert!(
        err.contains("EchoAntiecho") || err.contains("Echo-AntiEcho"),
        "expected EchoAntiecho fail-closed explanation, got: {err}"
    );
    assert!(panel.maxent.error_message.is_some());
    assert!(
        panel.maxent.result.is_none(),
        "stale MaxEnt result must be cleared when run returns Err"
    );
    assert!(panel.maxent.last_run_source.is_none());
    assert_eq!(
        panel.maxent.schedule_text, saved_schedule,
        "fail-closed error must never wipe user schedule"
    );
    let out_err = run_contents_frame(&ctx, &mut panel, None, true);
    assert!(
        audit_text_layout(&out_err.shapes, CONTENTS_VIEWPORT).min_font_px >= MIN_FONT_PX - 1e-3
    );
}

#[test]
fn test_tab3_artsy_rdc_demo_sorting_fail_closed_ood_and_copy_tsv() {
    let ctx = egui::Context::default();
    let mut panel = P22ToolsPanel::new();
    panel.open_tab(P22ToolTab::ArtsyRdc);

    // 1. 1-click Synthetic Demo
    panel
        .artsy
        .load_and_run_synthetic_demo()
        .expect("ARTSY RDC synthetic demo");
    assert_eq!(panel.artsy.results.len(), 12);
    let n_good = panel
        .artsy
        .results
        .iter()
        .filter(|r| r.quality == ArtsyQualityFlag::Good)
        .count();
    let n_low_snr = panel
        .artsy
        .results
        .iter()
        .filter(|r| r.quality == ArtsyQualityFlag::LowSnr)
        .count();
    let n_near_sat = panel
        .artsy
        .results
        .iter()
        .filter(|r| r.quality == ArtsyQualityFlag::NearSaturation)
        .count();
    assert_eq!(n_good, 10);
    assert_eq!(n_low_snr, 1);
    assert_eq!(n_near_sat, 1);

    // Verify residue 2 (true D_NH = -14.2 Hz) recovered accurately
    let r2 = panel
        .artsy
        .results
        .iter()
        .find(|r| r.seq_id == 2)
        .expect("residue 2");
    assert!((r2.d_nh_hz.expect("d_nh") - (-14.2)).abs() < 0.05);
    assert!(r2.sigma_d_nh_hz.expect("sigma") > 0.0);

    // 2. Sort by 1D_NH descending and verify ordering
    panel.artsy.sort_col = ArtsySortColumn::DNh;
    panel.artsy.sort_ascending = false;
    panel.artsy.apply_sort();
    let first_d = panel.artsy.results[0].d_nh_hz.expect("max d_nh");
    let last_d = panel
        .artsy
        .results
        .last()
        .and_then(|r| r.d_nh_hz)
        .expect("min d_nh");
    assert!(first_d > last_d);

    // 3. Export TSV matches rows and columns
    let tsv = panel.artsy.export_tsv();
    assert!(tsv.starts_with("seq_id\tres_name\ti_att_over_i_ref\tabs_1J_plus_1D_hz\t1D_NH_hz"));
    assert_eq!(tsv.lines().count(), 13);

    // 4. Non-destructive edit: append an OutOfDomainRatio row (|I_att/I_ref| > 2) and re-run
    panel
        .artsy
        .tsv_input
        .push_str("99\tGLY\t100.0\t250.0\t5.0\n");
    panel
        .artsy
        .run()
        .expect("ARTSY handles out-of-domain fail-closed");
    let r99 = panel
        .artsy
        .results
        .iter()
        .find(|r| r.seq_id == 99)
        .expect("residue 99");
    assert_eq!(r99.quality, ArtsyQualityFlag::OutOfDomainRatio);
    assert!(r99.d_nh_hz.is_none());

    let out = run_contents_frame(&ctx, &mut panel, None, false);
    assert!(audit_text_layout(&out.shapes, CONTENTS_VIEWPORT).min_font_px >= MIN_FONT_PX - 1e-3);
    let out_light = run_contents_frame(&ctx, &mut panel, None, true);
    assert!(
        audit_text_layout(&out_light.shapes, CONTENTS_VIEWPORT).min_font_px >= MIN_FONT_PX - 1e-3
    );
}

#[test]
fn test_tab4_pcs_transfer_and_curie_ccr_demo_and_diagonal_mode() {
    let ctx = egui::Context::default();
    let mut panel = P22ToolsPanel::new();
    panel.open_tab(P22ToolTab::PcsCurieCcr);

    // 1. 1-click Synthetic Demo: EM PCS cone transfer + joint PCS & Curie-spin Gamma_CCR fit
    panel
        .pcs_ccr
        .load_and_run_synthetic_demo()
        .expect("PCS & Curie CCR synthetic demo");
    let tr = panel
        .pcs_ccr
        .transfer_result
        .as_ref()
        .expect("PCS transfer result");
    assert_eq!(tr.transfers.len(), 14);
    assert_eq!(tr.n_matched, 13, "expected 13 observable residues matched");
    assert_eq!(
        tr.n_unmatched_bleached, 1,
        "expected residue 10 classified UnmatchedBleached"
    );
    assert_eq!(tr.transfers[0].status, PcsMatchStatus::UnmatchedBleached);

    let jf = panel
        .pcs_ccr
        .joint_fit
        .as_ref()
        .expect("joint PCS + Curie CCR fit");
    assert!(
        (jf.delta_chi.dchi_ax - 12.0).abs() < 1.0,
        "expected dchi_ax ~ 12.0, got {}",
        jf.delta_chi.dchi_ax
    );
    assert!(
        (jf.delta_chi.dchi_rh - 3.2).abs() < 0.8,
        "expected dchi_rh ~ 3.2, got {}",
        jf.delta_chi.dchi_rh
    );
    assert!(
        (jf.mu_eff_bohr - 9.5).abs() < 1.0,
        "expected mu_eff ~ 9.5 mu_B, got {}",
        jf.mu_eff_bohr
    );
    assert!(jf.pcs_q_factor < 0.15);
    assert!(jf.ccr_q_factor < 0.20);

    let exported = panel.pcs_ccr.export_matched_tsv();
    assert!(exported.contains("UnmatchedBleached"));
    assert_eq!(exported.lines().count(), 15);

    let out_dark = run_window_frame(&ctx, &mut panel, None, false);
    assert!(audit_text_layout(&out_dark.shapes, WINDOW_VIEWPORT).min_font_px >= MIN_FONT_PX - 1e-3);
    let out_light = run_window_frame(&ctx, &mut panel, None, true);
    assert!(
        audit_text_layout(&out_light.shapes, WINDOW_VIEWPORT).min_font_px >= MIN_FONT_PX - 1e-3
    );

    // 2. Non-destructive Single-Pass Diagonal mode without 3D coordinates
    panel.pcs_ccr.mode = PcsCcrMode::SinglePassTransfer;
    panel.pcs_ccr.use_initial_tensor = false;
    panel.pcs_ccr.dia_tsv =
        "seq_id\tlabel\th_ppm\tx_ppm\n1\tALA1\t8.10\t120.0\n2\tGLY2\t8.60\t110.0\n".to_string();
    panel.pcs_ccr.para_tsv =
        "peak_id\th_ppm\tx_ppm\n10\t8.45\t120.36\n11\t8.25\t109.64\n".to_string();
    panel
        .pcs_ccr
        .run()
        .expect("diagonal-only 2D PCS transfer without 3D coords");
    let diag_tr = panel
        .pcs_ccr
        .transfer_result
        .as_ref()
        .expect("diagonal transfer result");
    assert_eq!(diag_tr.n_matched, 2);
}

#[test]
fn test_all_tabs_guidance_headers_and_source_hygiene_zero_emoji_unwrap_panic_unsafe() {
    // Verify crisp 1-line Input -> Output guidance headers on all 4 tabs
    for tab in P22ToolTab::ALL {
        let header = tab.guidance_header();
        assert!(
            header.starts_with("Input:") && header.contains("-> Output:"),
            "tab {tab:?} missing 'Input: ... -> Output:' header: {header}"
        );
        assert!(
            !header.contains('\n'),
            "tab {tab:?} guidance header must be single-line"
        );
    }

    let prod_src = include_str!("../src/p22_tools_panel.rs");
    let test_src = include_str!("p22_gui_proc_struct_tests.rs");

    // Strict zero-emoji check across production and test sources
    for (name, src) in [
        ("p22_tools_panel.rs", prod_src),
        ("p22_gui_proc_struct_tests.rs", test_src),
    ] {
        for (line_no, line) in src.lines().enumerate() {
            for ch in line.chars() {
                let cp = ch as u32;
                let is_emoji = (0x1F000..=0x1FFFF).contains(&cp)
                    || (0x2600..=0x27BF).contains(&cp)
                    || cp == 0xFE0F;
                assert!(
                    !is_emoji,
                    "forbidden emoji U+{cp:04X} ('{ch}') in {name}:{}",
                    line_no + 1
                );
            }
        }
    }

    // Strict zero .unwrap() / .expect( / panic! / unsafe in production source
    for (line_no, line) in prod_src.lines().enumerate() {
        let code_part = line.split("//").next().unwrap_or("");
        for forbidden in [
            ".unwrap()",
            ".expect(",
            "panic!(",
            "todo!(",
            "unimplemented!(",
            "unsafe ",
        ] {
            assert!(
                !code_part.contains(forbidden),
                "forbidden '{forbidden}' in p22_tools_panel.rs:{}",
                line_no + 1
            );
        }
    }
}

#[test]
fn test_all_four_folder_tabs_render_and_preserve_non_destructive_edits() {
    let ctx = egui::Context::default();
    let mut panel = P22ToolsPanel::new();
    panel.open_tab(P22ToolTab::Deconv2d3d);

    // Edit user inputs across all 4 tabs before running or switching
    panel.deconv.max_iterations = 88;
    panel.maxent.lambda = 7.5;
    panel.artsy.dephasing_delay_ms = 11.2;
    panel.pcs_ccr.tau_m_ns = 9.25;

    // Render every tab in both Dark Oscilloscope and Paper White themes
    for tab in P22ToolTab::ALL {
        panel.open_tab(tab);
        for paper_white in [false, true] {
            let out = run_window_frame(&ctx, &mut panel, None, paper_white);
            let audit = audit_text_layout(&out.shapes, WINDOW_VIEWPORT);
            assert!(
                audit.min_font_px >= MIN_FONT_PX - 1e-3,
                "tab {tab:?} (paper_white={paper_white}) min font {} < {MIN_FONT_PX}",
                audit.min_font_px
            );
        }
    }

    // Verify all user edits survived every tab switch & render pass intact
    assert_eq!(panel.deconv.max_iterations, 88);
    assert!((panel.maxent.lambda - 7.5).abs() < 1e-12);
    assert!((panel.artsy.dephasing_delay_ms - 11.2).abs() < 1e-12);
    assert!((panel.pcs_ccr.tau_m_ns - 9.25).abs() < 1e-12);
}

#[test]
fn test_p1_2_accurate_2d_tab_labels_no_false_3d_claim() {
    assert_eq!(P22ToolTab::Deconv2d3d.label(), "Deconv 2D (VarPro)");
    assert_eq!(P22ToolTab::MaxEntNus.label(), "MaxEnt 2D NUS");
    for tab in P22ToolTab::ALL {
        let lbl = tab.label();
        assert!(
            !lbl.contains("3D") && !lbl.contains("2D/3D"),
            "tab label '{lbl}' must not claim 3D where only 2D is wired"
        );
    }
}

#[test]
fn test_p1_1_maxent_user_data_tsv_and_honest_synthetic_demo_badge() {
    let ctx = egui::Context::default();
    let mut panel = P22ToolsPanel::new();
    panel.open_tab(P22ToolTab::MaxEntNus);

    // 1. Synthetic demo is explicitly badged [Synthetic 3-Peak Demo]
    panel
        .maxent
        .load_and_run_synthetic_demo()
        .expect("synthetic demo run");
    assert_eq!(
        panel.maxent.last_run_source,
        Some(MaxEntDataSource::SyntheticDemo)
    );
    assert!(
        panel
            .maxent
            .status_message
            .contains("[Synthetic 3-Peak Demo]"),
        "synthetic demo status must carry [Synthetic 3-Peak Demo] badge: {}",
        panel.maxent.status_message
    );

    // 2. Real user-supplied 6-column 2D hypercomplex TSV (t1_idx, t2_idx, re_cc, im_cs, im_sc, re_ss)
    panel.maxent.n2 = 8;
    panel.maxent.n1 = 8;
    panel.maxent.zero_fill_factor = 1;
    panel.maxent.quad_mode_axis2 = QuadMode::States;
    panel.maxent.quad_mode_axis1 = QuadMode::States;
    panel.maxent.lambda_ui_mode = MaxEntLambdaUiMode::Fixed;
    panel.maxent.lambda = 15.0;
    panel.maxent.data_source = MaxEntDataSource::UserDataTsv;

    // Synthesize a single-peak user FID at normalized frequencies (f2 = 0.25, f1 = 0.50)
    // corresponding to discrete DFT bin (2, 4) on an 8x8 grid — distinct from the 3-peak demo.
    let mut tsv = String::from("t1_idx\tt2_idx\tre_cc\tim_cs\tim_sc\tre_ss\n");
    for k2 in 0..8usize {
        for k1 in 0..8usize {
            if (k1 + k2) % 3 == 2 && !(k1 == 0 && k2 == 0) {
                continue; // 67% NUS subset
            }
            let t2 = k2 as f64;
            let t1 = k1 as f64;
            let env = (-0.06 * t2 - 0.06 * t1).exp();
            let ph2 = 2.0 * PI * 0.25 * t2;
            let ph1 = 2.0 * PI * 0.50 * t1;
            let re_cc = env * ph2.cos() * ph1.cos();
            let im_cs = env * ph2.cos() * ph1.sin();
            let im_sc = env * ph2.sin() * ph1.cos();
            let re_ss = env * ph2.sin() * ph1.sin();
            tsv.push_str(&format!(
                "{k1}\t{k2}\t{re_cc:.6}\t{im_cs:.6}\t{im_sc:.6}\t{re_ss:.6}\n"
            ));
        }
    }
    panel.maxent.user_data_tsv = tsv;
    panel
        .maxent
        .run_sync()
        .expect("MaxEnt 2D reconstruction on real 6-column user TSV");
    assert_eq!(
        panel.maxent.last_run_source,
        Some(MaxEntDataSource::UserDataTsv)
    );
    assert!(panel.maxent.status_message.contains("[User Data TSV]"));
    let user_res = panel.maxent.result.as_ref().expect("user MaxEnt result");
    assert_eq!(user_res.spectrum_shape, vec![8, 8]);

    // Locate dominant RR peak in user spectrum — verify it sits at the user's frequency bin (2, 4)
    let mut best_bin = (0usize, 0usize);
    let mut best_val = f64::NEG_INFINITY;
    for m2 in 0..8 {
        for m1 in 0..8 {
            let v = user_res.spectrum[m2 * 8 + m1][0].abs();
            if v > best_val {
                best_val = v;
                best_bin = (m2, m1);
            }
        }
    }
    assert_eq!(
        best_bin,
        (2, 4),
        "user 6-col TSV peak must reconstruct at (2, 4), got {best_bin:?}"
    );

    // Render User Data TSV mode in both Dark and Paper White themes
    for pw in [false, true] {
        let out = run_contents_frame(&ctx, &mut panel, None, pw);
        assert!(
            audit_text_layout(&out.shapes, CONTENTS_VIEWPORT).min_font_px >= MIN_FONT_PX - 1e-3
        );
    }

    // 3. Real user-supplied 3-column TSV (t1_idx, re, im) on t2_idx = 0
    let mut tsv3 = String::from("t1_idx\tre\tim\n");
    for k1 in 0..8usize {
        let ph1 = 2.0 * PI * 0.25 * (k1 as f64);
        let env = (-0.08 * (k1 as f64)).exp();
        tsv3.push_str(&format!(
            "{k1}\t{:.6}\t{:.6}\n",
            env * ph1.cos(),
            env * ph1.sin()
        ));
    }
    panel.maxent.user_data_tsv = tsv3;
    panel
        .maxent
        .run_sync()
        .expect("MaxEnt 2D reconstruction on 3-column user TSV");
    assert_eq!(
        panel.maxent.last_run_source,
        Some(MaxEntDataSource::UserDataTsv)
    );
    assert!(panel.maxent.result.is_some());

    // 4. Malformed user TSV fails closed and clears stale result
    panel.maxent.user_data_tsv = "t1_idx\tre\tim\n0\tNaN\t0.0\n".to_string();
    assert!(panel.maxent.run_sync().is_err());
    assert!(
        panel.maxent.result.is_none(),
        "stale MaxEnt result must be cleared on user TSV error"
    );
    assert!(panel.maxent.last_run_source.is_none());
}

#[test]
fn test_p1_3_off_ui_thread_background_compute_deconv_and_maxent() {
    let ctx = egui::Context::default();
    let mut panel = P22ToolsPanel::new();

    // 1. Deconv 2D background thread lifecycle
    panel.open_tab(P22ToolTab::Deconv2d3d);
    panel.deconv.start_synthetic_demo_async();
    let deadline = Instant::now() + Duration::from_secs(15);
    while panel.deconv.is_computing() {
        assert!(
            Instant::now() < deadline,
            "Deconv background worker timed out"
        );
        let out = run_contents_frame(&ctx, &mut panel, None, false);
        assert!(
            audit_text_layout(&out.shapes, CONTENTS_VIEWPORT).min_font_px >= MIN_FONT_PX - 1e-3
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    let dres = panel
        .deconv
        .result
        .as_ref()
        .expect("Deconv background result should arrive via mpsc");
    assert_eq!(dres.peaks.len(), 2);
    assert_eq!(dres.diagnostic, DeconvDiagnostic::Converged);

    // 2. MaxEnt 2D background thread lifecycle
    panel.open_tab(P22ToolTab::MaxEntNus);
    panel.maxent.n2 = 16;
    panel.maxent.n1 = 16;
    panel.maxent.start_synthetic_demo_async();
    let deadline2 = Instant::now() + Duration::from_secs(15);
    while panel.maxent.is_computing() {
        assert!(
            Instant::now() < deadline2,
            "MaxEnt background worker timed out"
        );
        let out = run_contents_frame(&ctx, &mut panel, None, true);
        assert!(
            audit_text_layout(&out.shapes, CONTENTS_VIEWPORT).min_font_px >= MIN_FONT_PX - 1e-3
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    let mres = panel
        .maxent
        .result
        .as_ref()
        .expect("MaxEnt background result should arrive via mpsc");
    assert!(mres.kkt_residual_inf.is_finite());
    assert_eq!(
        panel.maxent.last_run_source,
        Some(MaxEntDataSource::SyntheticDemo)
    );

    // 3. Stale Deconv result cleared on ActiveSpectrum2D error when no spectrum is loaded
    panel.deconv.source = DeconvDataSource::ActiveSpectrum2D;
    assert!(panel.deconv.run_sync(None).is_err());
    assert!(
        panel.deconv.result.is_none(),
        "stale Deconv result must be cleared on error"
    );
}

#[test]
fn test_p1_4_artsy_editable_j_iso_and_sigma_j_quadrature_propagation() {
    let ctx = egui::Context::default();
    let mut panel = P22ToolsPanel::new();
    panel.open_tab(P22ToolTab::ArtsyRdc);

    // Default J_iso = -93.0 Hz and sigma_J = 1.0 Hz (never silently 0.0)
    assert!((panel.artsy.j_iso_default_hz - (-93.0)).abs() < 1e-12);
    assert!(
        (panel.artsy.j_iso_default_sigma_hz - DEFAULT_ARTSY_J_ISO_SIGMA_HZ).abs() < 1e-12
            && panel.artsy.j_iso_default_sigma_hz > 0.0,
        "default sigma_J must be 1.0 Hz, not 0.0"
    );

    panel
        .artsy
        .load_and_run_synthetic_demo()
        .expect("ARTSY demo with default sigma_J = 1.0 Hz");
    let r2_default = panel
        .artsy
        .results
        .iter()
        .find(|r| r.seq_id == 2)
        .expect("residue 2")
        .clone();
    let sig_jd = r2_default
        .aligned
        .sigma_analytical_hz
        .expect("analytical sigma_|J+D|");
    let sig_d_1 = r2_default.sigma_d_nh_hz.expect("sigma(D) with sigma_J=1");
    let expected_1 = sig_jd.hypot(1.0);
    assert!(
        (sig_d_1 - expected_1).abs() < 1e-9,
        "sigma(D)={sig_d_1} must equal sqrt(sigma_JD^2 + 1.0^2)={expected_1}"
    );
    assert!(
        sig_d_1 > sig_jd,
        "quadrature propagation must strictly increase sigma(D) above sigma_JD"
    );

    // Increase sigma_J to 2.5 Hz and shift J_iso to -91.0 Hz
    panel.artsy.j_iso_default_hz = -91.0;
    panel.artsy.j_iso_default_sigma_hz = 2.5;
    panel
        .artsy
        .run()
        .expect("ARTSY run with edited J_iso & sigma_J");
    let r2_edited = panel
        .artsy
        .results
        .iter()
        .find(|r| r.seq_id == 2)
        .expect("residue 2 edited")
        .clone();
    let expected_25 = sig_jd.hypot(2.5);
    assert!(
        (r2_edited.sigma_d_nh_hz.expect("sigma_2.5") - expected_25).abs() < 1e-9,
        "sigma(D) must match sqrt(sigma_JD^2 + 2.5^2)"
    );
    assert!(
        ((r2_edited.d_nh_hz.expect("d_edited") - r2_default.d_nh_hz.expect("d_def")).abs() - 2.0)
            .abs()
            < 1e-6,
        "shifting J_iso by 2 Hz must shift extracted D_NH by 2 Hz"
    );

    // Render with both sigma_J > 0 (disclosure) and sigma_J == 0 (warning disclosure)
    let out_pos = run_contents_frame(&ctx, &mut panel, None, false);
    assert!(
        audit_text_layout(&out_pos.shapes, CONTENTS_VIEWPORT).min_font_px >= MIN_FONT_PX - 1e-3
    );

    panel.artsy.j_iso_default_sigma_hz = 0.0;
    panel
        .artsy
        .run()
        .expect("ARTSY run with explicit sigma_J = 0");
    let r2_zero = panel
        .artsy
        .results
        .iter()
        .find(|r| r.seq_id == 2)
        .expect("residue 2 sigma_J=0");
    assert!((r2_zero.sigma_d_nh_hz.expect("sig0") - sig_jd).abs() < 1e-9);
    let out_zero = run_contents_frame(&ctx, &mut panel, None, true);
    assert!(
        audit_text_layout(&out_zero.shapes, CONTENTS_VIEWPORT).min_font_px >= MIN_FONT_PX - 1e-3
    );

    // Negative sigma_J fails closed and clears stale results
    panel.artsy.j_iso_default_sigma_hz = -0.5;
    assert!(panel.artsy.run().is_err());
    assert!(
        panel.artsy.results.is_empty(),
        "stale ARTSY results must be cleared on error"
    );
}
