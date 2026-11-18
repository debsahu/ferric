//! Headless integration & `egui` rendering tests for `nmr_gui::demo_preset` (`T16.4`).
//!
//! Provenance: Original clean-room integration test suite verifying zero-filesystem-I/O assembly
//! of synthetic Ubiquitin 2D `HSQC`, 3D `HNCACB` strips, `1UBQ` 3D coordinates, 1D multiplet
//! reaction stack, pseudo-2D `DOSY` decay series, and 2-group `Metabolomics` cohort.

use eframe::egui;
use nmr_gui::demo_preset::{build_builtin_demo_suite, BUILTIN_DEMO_SUITE_SEED};
use nmr_gui::dosy_panel::render_dosy_panel;
use nmr_gui::stack1d_panel::render_stack1d_panel;

#[test]
fn test_builtin_demo_suite_in_memory_and_headless_panels() {
    let mut suite = build_builtin_demo_suite(BUILTIN_DEMO_SUITE_SEED)
        .expect("built-in demo suite must assemble in memory without filesystem I/O");

    // 1. Ubiquitin 2D HSQC + 3D HNCACB/HNCO strips + spin systems
    assert_eq!(suite.ubiquitin_ref.sequence.len(), 76);
    assert_eq!(suite.ubiquitin_ref.seq_ids.len(), 76);
    assert!(
        suite.peaks.len() >= 200,
        "expected rich 2D/3D synthetic Ubiquitin peak list"
    );
    assert!(
        suite.systems.len() >= 60,
        "expected >= 60 assembled backbone spin systems"
    );
    assert_eq!(suite.hsqc_2d.axis_x.size, 512);
    assert_eq!(suite.hsqc_2d.axis_y.size, 256);
    assert!(suite.hsqc_2d.data.iter().any(|&v| v.abs() > 0.0));
    assert_eq!(suite.hncacb_3d.axis_z.size, 96);
    assert!(suite.hncacb_3d.data.iter().any(|&v| v > 0.0));
    assert!(suite.hncacb_3d.data.iter().any(|&v| v < 0.0));
    assert_eq!(suite.hnco_3d.axis_z.size, 96);

    // 2. 3D Ubiquitin (1UBQ) backbone coordinates
    let ens = suite
        .mol3d_panel
        .ensemble
        .as_ref()
        .expect("1UBQ 3D coordinates must be loaded");
    assert!(!ens.models.is_empty());
    assert_eq!(ens.models[0].chains[0].residues.len(), 76);

    // 3. Sample 1D multiplet stack + ROIs
    assert!(suite.stack1d_panel.traces.len() >= 4);
    assert!(!suite.stack1d_panel.roi_regions.is_empty());

    // 4. Sample DOSY decay series
    assert!(suite.dosy_panel.peaks.len() >= 3);
    assert!(!suite.dosy_panel.b_values_s_m2.is_empty());

    // 5. Sample 2-group Metabolomics cohort (`Control` vs `Case`)
    assert_eq!(suite.metabolomics_panel.samples.len(), 24);
    let has_control = suite
        .metabolomics_panel
        .samples
        .iter()
        .any(|s| s.group.eq_ignore_ascii_case("control"));
    let has_case = suite
        .metabolomics_panel
        .samples
        .iter()
        .any(|s| s.group.eq_ignore_ascii_case("case"));
    assert!(
        has_control && has_case,
        "expected 2-group Control vs Case cohort"
    );
    let study = suite
        .metabolomics_panel
        .study
        .as_ref()
        .expect("metabolomics pipeline must be pre-solved");
    let metab_buckets = study.buckets.len();
    let opls_q2 = study.opls_da.as_ref().expect("OPLS-DA must be solved").q2;
    assert!(metab_buckets >= 10);
    assert!(suite.metabolomics_panel.active_stocsy.is_some());

    // 6. Headless egui rendering smoke test across 1D Stack, DOSY, and Metabolomics panels
    let ctx = egui::Context::default();
    let mut out1 = ctx.run_ui(egui::RawInput::default(), |ui| {
        render_stack1d_panel(ui, &mut suite.stack1d_panel);
    });
    out1.textures_delta.clear();
    let mut out2 = ctx.run_ui(egui::RawInput::default(), |ui| {
        let _ = render_dosy_panel(ui, &mut suite.dosy_panel);
    });
    out2.textures_delta.clear();
    let mut out3 = ctx.run_ui(egui::RawInput::default(), |ui| {
        suite.metabolomics_panel.ui(ui);
    });
    out3.textures_delta.clear();

    // 7. Determinism check on identical seed
    let suite_repeat =
        build_builtin_demo_suite(BUILTIN_DEMO_SUITE_SEED).expect("repeat build must succeed");
    assert_eq!(suite.peaks.len(), suite_repeat.peaks.len());
    assert_eq!(suite.hsqc_2d.data, suite_repeat.hsqc_2d.data);
    assert_eq!(suite.status_summary(), suite_repeat.status_summary());
    println!(
        "P16_METRIC_JSON:{{\"suite\":\"demo_preset_tests\",\"seq_len\":{},\"peaks_count\":{},\"spin_systems_count\":{},\"hsqc_nx\":{},\"hsqc_ny\":{},\"hncacb_nz\":{},\"hnco_nz\":{},\"stack1d_traces\":{},\"dosy_peaks\":{},\"metabolomics_samples\":{},\"metabolomics_buckets\":{},\"opls_q2\":{:.6},\"headless_egui_panels\":3,\"deterministic_repeat\":true}}",
        suite.ubiquitin_ref.sequence.len(),
        suite.peaks.len(),
        suite.systems.len(),
        suite.hsqc_2d.axis_x.size,
        suite.hsqc_2d.axis_y.size,
        suite.hncacb_3d.axis_z.size,
        suite.hnco_3d.axis_z.size,
        suite.stack1d_panel.traces.len(),
        suite.dosy_panel.peaks.len(),
        suite.metabolomics_panel.samples.len(),
        metab_buckets,
        opls_q2
    );
}
