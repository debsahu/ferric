//! Headless `egui` integration tests for `T22.6` (`DANTE-CEST` & `cos-CEST` in
//! `Dynamics -> Exchange -> CEST`) and `T22.8` (2D Titration Trajectory Tracking,
//! Exchange-Regime Diagnostics, and Coupled 2D Bloch–McConnell Lineshape Fitting in
//! `Dynamics -> Exchange -> CSP`).
//!
//! Provenance: original clean-room integration tests for `nmr-gui::dynamics_panel`.

use eframe::egui;
use nmr_core::cest::CestNormalization;
use nmr_core::multi_cest::{simulate_multi_cest_profile, MultiCestSimParams};
use nmr_gui::dynamics_panel::{CestSchemeTab, CspSubTab, DynamicsStudioPanel, DynamicsStudioTab};
use nmr_gui::workbench_mode::{
    apply_type_scale, audit_text_layout, draw_mode_switcher_with_rect, primary_controls,
    WorkbenchMode, MAX_PRIMARY_CONTROLS_PER_MODE, MIN_CANVAS_AREA_FRACTION, MIN_FONT_PX,
};

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

fn click_text_button(
    ctx: &egui::Context,
    panel: &mut DynamicsStudioPanel,
    width: f32,
    height: f32,
    label_substr: &str,
) {
    let vp = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(width, height));
    let pre = run_two_passes(ctx, width, height, |ui| {
        egui::CentralPanel::default().show(ui, |ui| {
            let _ = panel.show(ui);
        });
    });
    let audit = audit_text_layout(&pre.shapes, vp);
    let pos = audit
        .texts
        .iter()
        .find(|t| t.text.contains(label_substr))
        .map(|t| t.rect.center())
        .unwrap_or_else(|| {
            panic!(
                "could not find text containing {label_substr:?} in {:?}",
                audit.texts.iter().map(|t| &t.text).collect::<Vec<_>>()
            )
        });

    let events = vec![
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
    ];
    let mut out = ctx.run_ui(
        egui::RawInput {
            screen_rect: Some(vp),
            events,
            ..Default::default()
        },
        |ui| {
            apply_type_scale(ui.ctx());
            egui::CentralPanel::default().show(ui, |ui| {
                let _ = panel.show(ui);
            });
        },
    );
    out.textures_delta.clear();
}

#[test]
fn test_t226_dante_and_cos_cest_folder_tabs_fit_and_crlb_rendering() {
    let ctx = egui::Context::default();
    let mut panel = DynamicsStudioPanel::new();
    panel
        .ensure_demo_exchange_and_csp(&[], &[])
        .expect("load base demo");
    panel.active_tab = DynamicsStudioTab::CestDest;
    panel.cest_scheme_tab = CestSchemeTab::CwCest;

    // 1. All 3 scheme folder tabs render cleanly on CW-CEST
    let vp = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1440.0, 900.0));
    let out_cw = run_two_passes(&ctx, 1440.0, 900.0, |ui| {
        egui::CentralPanel::default().show(ui, |ui| {
            let _ = panel.show(ui);
        });
    });
    let audit_cw = audit_text_layout(&out_cw.shapes, vp);
    for expected in ["CW-CEST", "DANTE-CEST", "cos-CEST", "Input:", "Output:"] {
        assert!(
            audit_cw.texts.iter().any(|t| t.text.contains(expected)),
            "CEST tab missing {expected:?}"
        );
    }
    assert!(
        audit_cw.clipped.is_empty(),
        "CW-CEST clipped: {:?}",
        audit_cw.clipped
    );
    assert!(
        audit_cw.overlapping.is_empty(),
        "CW-CEST overlapping: {:?}",
        audit_cw.overlapping
    );
    assert!(audit_cw.min_font_px >= MIN_FONT_PX - 1e-3);

    // 2. Click DANTE-CEST folder tab, run 2-state fit, verify parameter recovery & valid CRLB
    click_text_button(&ctx, &mut panel, 1440.0, 900.0, "DANTE-CEST");
    assert_eq!(panel.cest_scheme_tab, CestSchemeTab::DanteCest);

    let dante_msg = panel
        .fit_multi_cest_2state(CestSchemeTab::DanteCest)
        .expect("DANTE-CEST fit must succeed");
    assert!(dante_msg.contains("[Synthetic Demo]"));
    let dfit = panel
        .multi_cest
        .dante_fit
        .as_ref()
        .expect("dante_fit populated");
    assert!(dfit.converged, "DANTE-CEST fit must converge");
    assert!(dfit.covariance_valid, "DANTE-CEST fit must have valid CRLB");
    assert!(dfit.std_errors.is_some());
    assert!(
        (dfit.params.p_b - panel.multi_cest.demo_p_b).abs() / panel.multi_cest.demo_p_b < 0.10,
        "DANTE p_B recovery off: fit={} vs true={}",
        dfit.params.p_b,
        panel.multi_cest.demo_p_b
    );
    assert!(
        (dfit.params.k_ex_s_inv - panel.multi_cest.demo_k_ex_s_inv).abs()
            / panel.multi_cest.demo_k_ex_s_inv
            < 0.12,
        "DANTE k_ex recovery off: fit={} vs true={}",
        dfit.params.k_ex_s_inv,
        panel.multi_cest.demo_k_ex_s_inv
    );
    assert!(!panel.multi_cest.dante_sim_curves.is_empty());

    // Open Params & TSV editor and render DANTE-CEST cleanly
    panel.multi_cest.show_params_editor = true;
    let out_dante = run_two_passes(&ctx, 1440.0, 900.0, |ui| {
        egui::CentralPanel::default().show(ui, |ui| {
            let _ = panel.show(ui);
        });
    });
    let audit_dante = audit_text_layout(&out_dante.shapes, vp);
    let joined_dante: String = audit_dante
        .texts
        .iter()
        .map(|t| t.text.as_str())
        .collect::<Vec<_>>()
        .join(" | ");
    for expected in [
        "b1_hz:",
        "t_ex_s:",
        "n_pulses:",
        "CRLB: Valid",
        "[Synthetic Demo]",
    ] {
        assert!(
            joined_dante.contains(expected),
            "DANTE-CEST UI missing {expected:?} in: {joined_dante}"
        );
    }
    let non_scroll_clipped_dante: Vec<_> = audit_dante
        .clipped
        .iter()
        .filter(|t| !t.text.contains('\n'))
        .collect();
    assert!(
        non_scroll_clipped_dante.is_empty(),
        "DANTE-CEST editor clipped non-scroll text: {:?}",
        non_scroll_clipped_dante
    );
    assert!(
        audit_dante.overlapping.is_empty(),
        "DANTE-CEST editor overlapping: {:?}",
        audit_dante.overlapping
    );
    assert!(audit_dante.min_font_px >= MIN_FONT_PX - 1e-3);

    // 3. Switch to cos-CEST via click, verify mod_freq_hz exposed and cos-CEST 2-state fit succeeds
    panel.multi_cest.show_params_editor = false;
    click_text_button(&ctx, &mut panel, 1440.0, 900.0, "cos-CEST");
    assert_eq!(panel.cest_scheme_tab, CestSchemeTab::CosCest);

    panel.multi_cest.mod_freq_hz = 320.0;
    let cos_msg = panel
        .fit_multi_cest_2state(CestSchemeTab::CosCest)
        .expect("cos-CEST fit must succeed");
    assert!(cos_msg.contains("cos-CEST"));
    let cfit = panel
        .multi_cest
        .cos_fit
        .as_ref()
        .expect("cos_fit populated");
    assert!(cfit.converged, "cos-CEST fit must converge");
    assert!(cfit.covariance_valid, "cos-CEST fit must have valid CRLB");
    assert!(
        (cfit.params.dw_ab_ppm - panel.multi_cest.demo_dw_ab_ppm).abs() < 0.25,
        "cos-CEST dw_ab_ppm recovery off: fit={} vs true={}",
        cfit.params.dw_ab_ppm,
        panel.multi_cest.demo_dw_ab_ppm
    );
    assert!(!panel.multi_cest.cos_sim_curves.is_empty());

    panel.multi_cest.show_params_editor = true;
    let out_cos = run_two_passes(&ctx, 1440.0, 900.0, |ui| {
        egui::CentralPanel::default().show(ui, |ui| {
            let _ = panel.show(ui);
        });
    });
    let audit_cos = audit_text_layout(&out_cos.shapes, vp);
    let joined_cos: String = audit_cos
        .texts
        .iter()
        .map(|t| t.text.as_str())
        .collect::<Vec<_>>()
        .join(" | ");
    for expected in ["mod_freq_hz:", "b1_hz:", "t_ex_s:", "CRLB: Valid"] {
        assert!(
            joined_cos.contains(expected),
            "cos-CEST UI missing {expected:?} in: {joined_cos}"
        );
    }
    let non_scroll_clipped_cos: Vec<_> = audit_cos
        .clipped
        .iter()
        .filter(|t| !t.text.contains('\n'))
        .collect();
    assert!(
        non_scroll_clipped_cos.is_empty(),
        "cos-CEST editor clipped non-scroll text: {:?}",
        non_scroll_clipped_cos
    );
    assert!(
        audit_cos.overlapping.is_empty(),
        "cos-CEST overlapping: {:?}",
        audit_cos.overlapping
    );

    // 4. Non-destructive preservation: switching back to DANTE-CEST retains its fit & curves
    panel.cest_scheme_tab = CestSchemeTab::DanteCest;
    assert!(panel.multi_cest.dante_fit.is_some());
    assert!(!panel.multi_cest.dante_sim_curves.is_empty());
}

#[test]
fn test_t226_multi_cest_tsv_import_and_honest_crlb_unavailable_on_flat_profile() {
    let ctx = egui::Context::default();
    let mut panel = DynamicsStudioPanel::new();
    panel.active_tab = DynamicsStudioTab::CestDest;
    panel.cest_scheme_tab = CestSchemeTab::DanteCest;

    // 1. Valid synthetic TSV import sets is_synthetic_demo = false and fits cleanly
    let offsets: Vec<f64> = (-20..=20).map(|k| 118.0 + 0.22 * (k as f64)).collect();
    let scheme = panel
        .build_multi_cest_scheme(CestSchemeTab::DanteCest, 16.0)
        .expect("dante scheme");
    let sim = simulate_multi_cest_profile(
        &MultiCestSimParams {
            p_b: 0.02,
            k_ex_s_inv: 150.0,
            major_ppm: 118.0,
            dw_ab_ppm: 3.5,
            r1a_s_inv: 1.3,
            r2a_s_inv: 14.0,
            r1b_s_inv: 1.3,
            r2b_s_inv: 20.0,
            larmor_mhz: 60.8106,
            b1_inhomogeneity_frac: 0.0,
            n_b1_quad: 1,
            normalization: CestNormalization::ZeroTimePhaseCycled,
            scheme,
        },
        &offsets,
    )
    .expect("sim dante");
    let mut tsv = String::from("offset_ppm\tintensity\tsigma\tb1_hz\n");
    for (&x, &y) in offsets.iter().zip(sim.iter()) {
        tsv.push_str(&format!("{x:.4}\t{y:.6}\t0.004\t16.0\n"));
    }
    let msg = panel
        .import_multi_cest_tsv(CestSchemeTab::DanteCest, &tsv)
        .expect("import valid DANTE TSV");
    assert!(!panel.multi_cest.dante_is_synthetic_demo);
    assert!(!msg.contains("[Synthetic Demo]"));
    assert!(panel.multi_cest.dante_fit.is_some());

    // 2. Degenerate zero-B1 flat profile yields rank-deficient Fisher (`covariance_valid == false`)
    //    and UI must honestly display "CRLB: Unavailable" + "(SE N/A)" without fake +/- 0.00!
    let mut flat_tsv = String::from("offset_ppm\tintensity\tsigma\tb1_hz\n");
    for k in -10..=10 {
        let off = 118.0 + 0.4 * (k as f64);
        flat_tsv.push_str(&format!("{off:.3}\t0.600000\t0.01\t0.0\n"));
    }
    let flat_msg = panel
        .import_multi_cest_tsv(CestSchemeTab::DanteCest, &flat_tsv)
        .expect("flat profile fit returns honest unidentifiable result");
    let flat_fit = panel
        .multi_cest
        .dante_fit
        .as_ref()
        .expect("flat fit result");
    assert!(
        !flat_fit.covariance_valid,
        "zero-B1 flat profile must report covariance_valid = false"
    );
    assert!(flat_fit.std_errors.is_none());
    assert!(flat_msg.contains("CRLB: Unavailable"));
    assert!(flat_msg.contains("SE N/A"));
    assert!(!flat_msg.contains("± 0.00"));

    let vp = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1440.0, 900.0));
    let out = run_two_passes(&ctx, 1440.0, 900.0, |ui| {
        egui::CentralPanel::default().show(ui, |ui| {
            let _ = panel.show(ui);
        });
    });
    let audit = audit_text_layout(&out.shapes, vp);
    let joined: String = audit
        .texts
        .iter()
        .map(|t| t.text.as_str())
        .collect::<Vec<_>>()
        .join(" | ");
    assert!(joined.contains("CRLB: Unavailable"));
    assert!(joined.contains("SE N/A"));

    // 3. Hostile TSV with sigma = 0.0 fails closed cleanly
    let bad = panel.import_multi_cest_tsv(
        CestSchemeTab::DanteCest,
        "offset_ppm\tintensity\tsigma\n118.0\t0.5\t0.0\n",
    );
    assert!(bad.is_err());
    assert!(panel.multi_cest.last_error.is_some());
}

#[test]
fn test_t226_dante_and_cos_cest_refit_preserves_imported_dual_b1() {
    // 171.assign:P1-4 regression: refitting imported multi-B1 ([14, 28] Hz) DANTE-CEST and
    // cos-CEST profiles MUST preserve each profile's own B1 and NEVER overwrite profile 0 with
    // `panel.multi_cest.b1_hz` (the demo slider).
    for scheme_tab in [CestSchemeTab::DanteCest, CestSchemeTab::CosCest] {
        let mut panel = DynamicsStudioPanel::new();
        panel.active_tab = DynamicsStudioTab::CestDest;
        panel.cest_scheme_tab = scheme_tab;
        panel.multi_cest.larmor_mhz = 60.8106;
        panel.multi_cest.major_ppm = 118.0;
        panel.multi_cest.b1_inhomogeneity_frac = 0.0;
        panel.multi_cest.n_b1_quad = 1;

        let offsets: Vec<f64> = (-24..=24).map(|k| 118.0 + 0.25 * (k as f64)).collect();
        let true_dw = 4.20;
        let mut tsv = String::from("offset_ppm\tintensity\tsigma\tb1_hz\n");
        for &b1 in &[14.0_f64, 28.0_f64] {
            let scheme = panel
                .build_multi_cest_scheme(scheme_tab, b1)
                .expect("build scheme");
            let sim = simulate_multi_cest_profile(
                &MultiCestSimParams {
                    p_b: 0.025,
                    k_ex_s_inv: 180.0,
                    major_ppm: 118.0,
                    dw_ab_ppm: true_dw,
                    r1a_s_inv: 1.3,
                    r2a_s_inv: 14.0,
                    r1b_s_inv: 1.3,
                    r2b_s_inv: 20.0,
                    larmor_mhz: 60.8106,
                    b1_inhomogeneity_frac: 0.0,
                    n_b1_quad: 1,
                    normalization: CestNormalization::ZeroTimePhaseCycled,
                    scheme,
                },
                &offsets,
            )
            .expect("simulate profile");
            for (&x, &y) in offsets.iter().zip(sim.iter()) {
                tsv.push_str(&format!("{x:.4}\t{y:.6}\t0.004\t{b1:.1}\n"));
            }
        }

        panel
            .import_multi_cest_tsv(scheme_tab, &tsv)
            .expect("import dual-B1 TSV");

        // Move slider far away (40 Hz) and refit — imported [14, 28] Hz MUST remain intact!
        panel.multi_cest.b1_hz = 40.0;
        panel
            .fit_multi_cest_2state(scheme_tab)
            .expect("refit imported dual-B1 profiles");

        let (profiles, fit_opt) = match scheme_tab {
            CestSchemeTab::DanteCest => (
                &panel.multi_cest.dante_profiles,
                panel.multi_cest.dante_fit.as_ref(),
            ),
            CestSchemeTab::CosCest => (
                &panel.multi_cest.cos_profiles,
                panel.multi_cest.cos_fit.as_ref(),
            ),
            CestSchemeTab::CwCest => unreachable!(),
        };
        assert_eq!(profiles.len(), 2);
        let b1s: Vec<f64> = profiles
            .iter()
            .map(|p| p.scheme.effective_b1_hz())
            .collect();
        assert!(
            (b1s[0] - 14.0).abs() < 1e-6 && (b1s[1] - 28.0).abs() < 1e-6,
            "{scheme_tab:?}: imported B1 [14, 28] Hz corrupted on refit to {:?}",
            b1s
        );
        let fit = fit_opt.expect("refit result present");
        assert!(
            fit.reduced_chi2 < 5.0,
            "{scheme_tab:?}: refit reduced_chi2 exploded to {} (B1 corruption?)",
            fit.reduced_chi2
        );
        assert!(
            (fit.params.dw_ab_ppm - true_dw).abs() < 0.15,
            "{scheme_tab:?}: refit dw_ab_ppm drifted to {} vs true {true_dw}",
            fit.params.dw_ab_ppm
        );
    }
}

#[test]
fn test_t228_titration_2d_folder_tab_demo_tracking_regime_and_lineshape_fit() {
    let ctx = egui::Context::default();
    let mut panel = DynamicsStudioPanel::new();
    panel
        .ensure_demo_exchange_and_csp(&[], &[])
        .expect("load base demo");
    panel.active_tab = DynamicsStudioTab::LigandCsp;
    panel.csp_sub_tab = CspSubTab::FastExchangeCsp;

    let vp = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1440.0, 900.0));
    let out_fast = run_two_passes(&ctx, 1440.0, 900.0, |ui| {
        egui::CentralPanel::default().show(ui, |ui| {
            let _ = panel.show(ui);
        });
    });
    let audit_fast = audit_text_layout(&out_fast.shapes, vp);
    for expected in [
        "Fast-Exchange CSP",
        "2D Titration Lineshape (T22.8)",
        "Input:",
        "Output:",
    ] {
        assert!(
            audit_fast.texts.iter().any(|t| t.text.contains(expected)),
            "CSP tab missing {expected:?}"
        );
    }

    // Click 2D Titration Lineshape (T22.8) folder tab, then click Load [Synthetic Demo]
    click_text_button(
        &ctx,
        &mut panel,
        1440.0,
        900.0,
        "2D Titration Lineshape (T22.8)",
    );
    assert_eq!(panel.csp_sub_tab, CspSubTab::Titration2dLineshape);

    click_text_button(&ctx, &mut panel, 1440.0, 900.0, "Load [Synthetic Demo]");
    assert!(panel.titration_2d.is_synthetic_demo);

    let tr = panel
        .titration_2d
        .tracking_result
        .as_ref()
        .expect("trajectory tracking populated");
    assert_eq!(tr.trajectories.len(), 4);
    assert_eq!(
        tr.trajectories[2].n_dropouts, 1,
        "I44 midpoint dropout must be tracked"
    );

    let fit = panel
        .titration_2d
        .lineshape_fit
        .as_ref()
        .expect("2D Bloch-McConnell lineshape fit populated");
    assert!(fit.converged);
    assert!(
        (fit.kd_um - 25.0).abs() / 25.0 < 0.10,
        "2D lineshape Kd recovery off: {} vs 25.0 uM",
        fit.kd_um
    );
    assert!(
        (fit.k_off_s_inv - 120.0).abs() / 120.0 < 0.12,
        "2D lineshape k_off recovery off: {} vs 120.0 s^-1",
        fit.k_off_s_inv
    );
    assert!(
        !fit.fast_exchange_comparison.fast_exchange_assumption_valid,
        "intermediate-exchange demo must flag fast-exchange CSP assumption as invalid"
    );
    assert_eq!(fit.residues.len(), 2);

    // Verify rendered T22.8 view shows [Synthetic Demo], Kd, k_off, dwH, dwN, and |dw|/k_off
    let out_2d = run_two_passes(&ctx, 1440.0, 900.0, |ui| {
        egui::CentralPanel::default().show(ui, |ui| {
            let _ = panel.show(ui);
        });
    });
    let audit_2d = audit_text_layout(&out_2d.shapes, vp);
    let joined_2d: String = audit_2d
        .texts
        .iter()
        .map(|t| t.text.as_str())
        .collect::<Vec<_>>()
        .join(" | ");
    for expected in [
        "[Synthetic Demo]",
        "2D Fit: Kd =",
        "k_off =",
        "Fast-CSP Valid: No",
        "|dwH|/koff",
        "|dwN|/koff",
        "I44",
        "D61",
    ] {
        assert!(
            joined_2d.contains(expected),
            "T22.8 UI missing {expected:?} in: {joined_2d}"
        );
    }
    assert!(
        audit_2d.clipped.is_empty(),
        "T22.8 clipped: {:?}",
        audit_2d.clipped
    );
    assert!(
        audit_2d.overlapping.is_empty(),
        "T22.8 overlapping: {:?}",
        audit_2d.overlapping
    );
    assert!(audit_2d.min_font_px >= MIN_FONT_PX - 1e-3);

    // Non-destructive edit: tweak demo Kd -> 45 uM, k_off -> 180 s^-1, re-run & verify update
    panel.titration_2d.demo_kd_um = 45.0;
    panel.titration_2d.demo_k_off_s_inv = 180.0;
    panel
        .run_titration_2d_analysis()
        .expect("re-run with tweaked params");
    let fit2 = panel
        .titration_2d
        .lineshape_fit
        .as_ref()
        .expect("updated 2D fit");
    assert!(
        (fit2.kd_um - 45.0).abs() / 45.0 < 0.12,
        "re-run Kd should follow edited 45 uM, got {}",
        fit2.kd_um
    );
    assert!(
        (fit2.k_off_s_inv - 180.0).abs() / 180.0 < 0.15,
        "re-run k_off should follow edited 180 s^-1, got {}",
        fit2.k_off_s_inv
    );

    // Switch to Fast-Exchange CSP and back -> non-destructive preservation
    panel.csp_sub_tab = CspSubTab::FastExchangeCsp;
    panel.csp_sub_tab = CspSubTab::Titration2dLineshape;
    assert!(panel.titration_2d.lineshape_fit.is_some());
    assert!((panel.titration_2d.demo_kd_um - 45.0).abs() < 1e-9);
}

#[test]
fn test_t228_titration_2d_tsv_import_crossing_tracks_and_regime_audit() {
    let ctx = egui::Context::default();
    let mut panel = DynamicsStudioPanel::new();

    // Custom 5-step TSV with a crossing pair (A10 / L11)
    let mut tsv = String::from(
        "# nmrassign_titration_2d_peaks\nstep_index\tstep_label\tligand_conc_um\tprotein_conc_um\tpeak_id\th_ppm\tn_ppm\tlabel\tres_name\n",
    );
    let concs = [0.0, 25.0, 50.0, 100.0, 200.0];
    for (t, &l) in concs.iter().enumerate() {
        let tf = t as f64;
        let lbl0 = if t == 0 { "A10\tALA" } else { "\t" };
        let lbl1 = if t == 0 { "L11\tLEU" } else { "\t" };
        tsv.push_str(&format!(
            "{t}\t{l:.0}uM\t{l:.1}\t50.0\t0\t{:.4}\t{:.3}\t{lbl0}\n",
            7.80 + 0.04 * tf,
            118.0 + 0.35 * tf
        ));
        tsv.push_str(&format!(
            "{t}\t{l:.0}uM\t{l:.1}\t50.0\t1\t{:.4}\t{:.3}\t{lbl1}\n",
            7.80 + 0.04 * tf,
            119.4 - 0.35 * tf
        ));
    }

    assert!(DynamicsStudioPanel::looks_like_exchange_or_csp_tsv(&tsv));
    let summary = panel
        .import_exchange_or_csp_tsv(&tsv)
        .expect("import 2D titration peak TSV");
    assert_eq!(panel.active_tab, DynamicsStudioTab::LigandCsp);
    assert_eq!(panel.csp_sub_tab, CspSubTab::Titration2dLineshape);
    assert!(!panel.titration_2d.is_synthetic_demo);
    assert!(summary.contains("Tracked 2 trajectories"));
    assert_eq!(panel.titration_2d.regime_reports.len(), 2);

    // Render peak-list TSV mode (trajectory map + |dw|/k_off regime bars)
    let vp = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1440.0, 900.0));
    let out = run_two_passes(&ctx, 1440.0, 900.0, |ui| {
        egui::CentralPanel::default().show(ui, |ui| {
            let _ = panel.show(ui);
        });
    });
    let audit = audit_text_layout(&out.shapes, vp);
    assert!(
        audit.clipped.is_empty(),
        "T22.8 TSV clipped: {:?}",
        audit.clipped
    );
    assert!(
        audit.overlapping.is_empty(),
        "T22.8 TSV overlapping: {:?}",
        audit.overlapping
    );
    assert!(audit.min_font_px >= MIN_FONT_PX - 1e-3);
    assert!(audit
        .texts
        .iter()
        .any(|t| t.text.contains("Exchange Regime Diagnostics")));
}

#[test]
fn test_primary_controls_le_12_canvas_ge_65_percent_and_text_audit_at_1440x900_and_1280x800() {
    let mut panel = DynamicsStudioPanel::new();
    panel
        .ensure_demo_exchange_and_csp(&[], &[])
        .expect("load base demo");
    panel
        .load_multi_cest_synthetic_demo(CestSchemeTab::DanteCest)
        .expect("dante demo");
    panel
        .load_multi_cest_synthetic_demo(CestSchemeTab::CosCest)
        .expect("cos demo");
    panel
        .load_titration_2d_synthetic_demo()
        .expect("titration 2d demo");

    let cases: [(&str, DynamicsStudioTab, CestSchemeTab, CspSubTab); 5] = [
        (
            "CEST / CW-CEST",
            DynamicsStudioTab::CestDest,
            CestSchemeTab::CwCest,
            CspSubTab::FastExchangeCsp,
        ),
        (
            "CEST / DANTE-CEST",
            DynamicsStudioTab::CestDest,
            CestSchemeTab::DanteCest,
            CspSubTab::FastExchangeCsp,
        ),
        (
            "CEST / cos-CEST",
            DynamicsStudioTab::CestDest,
            CestSchemeTab::CosCest,
            CspSubTab::FastExchangeCsp,
        ),
        (
            "CSP / Fast-Exchange",
            DynamicsStudioTab::LigandCsp,
            CestSchemeTab::CwCest,
            CspSubTab::FastExchangeCsp,
        ),
        (
            "CSP / 2D Titration (T22.8)",
            DynamicsStudioTab::LigandCsp,
            CestSchemeTab::CwCest,
            CspSubTab::Titration2dLineshape,
        ),
    ];

    for (name, tab, cest_scheme, csp_sub) in cases {
        panel.active_tab = tab;
        panel.cest_scheme_tab = cest_scheme;
        panel.csp_sub_tab = csp_sub;
        panel.multi_cest.show_params_editor = false;
        panel.titration_2d.show_params_editor = false;

        // 1440x900 budget audit (<= 12 primary controls & >= 65% canvas/plot share)
        {
            let ctx = egui::Context::default();
            let mut switcher_r = egui::Rect::ZERO;
            let vp = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1440.0, 900.0));
            let out = run_two_passes(&ctx, 1440.0, 900.0, |ui| {
                egui::Panel::top("hdr_dyn_p22").show(ui, |ui| {
                    switcher_r = draw_mode_switcher_with_rect(ui, WorkbenchMode::Dynamics).1;
                });
                egui::CentralPanel::default().show(ui, |ui| {
                    let _ = panel.show(ui);
                });
            });

            let canvas_r = panel.last_canvas_rect.expect("last_canvas_rect");
            let plot_r = panel.last_plot_rect.expect("last_plot_rect");
            let screen_area = 1440.0_f32 * 900.0_f32;

            // Count ALL primary controls outside switcher & pure data-plot rect!
            let ctrls = primary_controls(&ctx, &[switcher_r, plot_r], 40_000.0);
            assert!(
                ctrls.len() <= MAX_PRIMARY_CONTROLS_PER_MODE,
                "{name}: exceeded {MAX_PRIMARY_CONTROLS_PER_MODE} primary controls, got {} (rects={:?})",
                ctrls.len(),
                ctrls.iter().map(|c| c.rect).collect::<Vec<_>>()
            );

            let canvas_share = canvas_r.area() / screen_area;
            let plot_share = plot_r.area() / screen_area;
            assert!(
                canvas_share >= MIN_CANVAS_AREA_FRACTION,
                "{name}: last_canvas_rect share {canvas_share:.3} < {MIN_CANVAS_AREA_FRACTION}"
            );
            assert!(
                plot_share >= MIN_CANVAS_AREA_FRACTION,
                "{name}: pure data plot share {plot_share:.3} < {MIN_CANVAS_AREA_FRACTION}"
            );

            let audit = audit_text_layout(&out.shapes, vp);
            assert!(
                audit.clipped.is_empty(),
                "{name} at 1440x900 clipped: {:?}",
                audit.clipped
            );
            assert!(
                audit.overlapping.is_empty(),
                "{name} at 1440x900 overlapping: {:?}",
                audit.overlapping
            );
            assert!(
                audit.min_font_px >= MIN_FONT_PX - 1e-3,
                "{name} at 1440x900 min_font_px {} < {MIN_FONT_PX}",
                audit.min_font_px
            );
        }

        // 1280x800 layout & font-floor audit
        {
            let ctx = egui::Context::default();
            let vp = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1280.0, 800.0));
            let out = run_two_passes(&ctx, 1280.0, 800.0, |ui| {
                egui::CentralPanel::default().show(ui, |ui| {
                    let _ = panel.show(ui);
                });
            });
            let audit = audit_text_layout(&out.shapes, vp);
            assert!(
                audit.clipped.is_empty(),
                "{name} at 1280x800 clipped: {:?}",
                audit.clipped
            );
            assert!(
                audit.overlapping.is_empty(),
                "{name} at 1280x800 overlapping: {:?}",
                audit.overlapping
            );
            assert!(
                audit.min_font_px >= MIN_FONT_PX - 1e-3,
                "{name} at 1280x800 min_font_px {} < {MIN_FONT_PX}",
                audit.min_font_px
            );
        }
    }
}
