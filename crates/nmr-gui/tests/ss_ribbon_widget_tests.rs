//! Headless `egui`, geometry, bidirectional synchronization, `_ma_qa_metric_local` / `B_iso`
//! provenance, `< 1.0 ms` `SSP` & `< 2.0 ms` 3D putty performance, and zero-emoji hygiene tests
//! for Phase 21 `T21.3` (`docs/specs/21-alphafold-integrative-nmr.md` §3 & §8).

use std::collections::BTreeMap;
use std::time::Instant;

use eframe::egui;
use nmr_core::idp_propensity::PropensityConfig;
use nmr_core::nmrstar::{read_star_str, Reference};
use nmr_gui::contour_painter::SyncCrosshair;
use nmr_gui::mol3d_overlay::{
    alphafold_plddt_rgba, ConcordanceTrackClass, QuantitativeTrack, QuantitativeTrackData,
    SsState4, UNASSIGNED_PUTTY_RGBA,
};
use nmr_gui::mol3d_viewer::{
    build_mol3d_scene, parse_mmcif_ensemble, parse_pdb_ensemble, parse_structure_auto,
    CoordinateProvenance, Mol3dViewerPanel, RepresentationStyle, SceneBuildConfig,
};
use nmr_gui::sidechain_widget::SequenceRibbonState;
use nmr_gui::ss_ribbon_widget::{
    apply_ss_ribbon_click_sync, evaluate_concordance_with_summary, SsRibbonPanel, SsRibbonState,
    SsRibbonTheme, SsRibbonWorkspaceInput,
};

const UBIQUITIN_1UBQ_PDB: &str = include_str!("../../../data/fixtures/ubiquitin_1ubq_xray.pdb");
const SYNTHETIC_UBIQUITIN_STR: &str =
    include_str!("../../../data/fixtures/synthetic_ubiquitin.str");

/// Synthesize a multi-conformer all-atom `n_res`-residue ensemble (`17` atoms/residue -> `51,000`
/// atoms across `20 x 150 aa`) for 3D putty timing benchmarks.
fn synthesize_ensemble_pdb(n_models: usize, n_res: usize) -> String {
    const RES_CYCLE: &[&str] = &["LEU", "ILE", "PHE", "GLU", "LYS", "ASP", "VAL", "ALA"];
    const ATOM_TEMPLATES: &[(&str, &str, [f64; 3])] = &[
        ("N", "N", [-1.458, 0.000, 0.000]),
        ("H", "H", [-1.960, 0.820, 0.280]),
        ("CA", "C", [0.000, 0.000, 0.000]),
        ("HA", "H", [0.360, -0.890, -0.480]),
        ("C", "C", [0.540, 1.420, 0.000]),
        ("O", "O", [0.020, 2.390, -0.540]),
        ("CB", "C", [0.530, -0.760, 1.220]),
        ("HB2", "H", [0.180, -1.780, 1.210]),
        ("HB3", "H", [1.610, -0.770, 1.190]),
        ("CG", "C", [0.080, -0.120, 2.540]),
        ("HG2", "H", [-0.990, -0.100, 2.580]),
        ("HG3", "H", [0.460, 0.890, 2.610]),
        ("CD1", "C", [0.620, -0.920, 3.720]),
        ("HD11", "H", [0.290, -0.470, 4.640]),
        ("HD12", "H", [1.700, -0.940, 3.690]),
        ("CD2", "C", [-1.440, -0.180, 2.660]),
        ("HD21", "H", [-1.760, 0.280, 3.580]),
    ];

    let mut pdb = String::with_capacity(n_models * n_res * ATOM_TEMPLATES.len() * 82 + 256);
    pdb.push_str("HEADER    ALPHAFOLD ENSEMBLE BENCHMARK\n");
    for m in 0..n_models {
        pdb.push_str(&format!("MODEL     {:>4}\n", m + 1));
        let mut serial = 1u32;
        for i in 0..n_res {
            let res_name = RES_CYCLE[i % RES_CYCLE.len()];
            let seq_id = (i + 1) as i32;
            let (bx, by, bz) = if (i / 25) % 2 == 0 {
                let ang = (i as f64) * 1.745;
                (2.3 * ang.cos(), 2.3 * ang.sin(), (i as f64) * 1.50)
            } else {
                let zig = if i % 2 == 0 { 0.45 } else { -0.45 };
                (zig, 6.0 + zig * 0.5, (i as f64) * 3.25)
            };
            let plddt = match i % 4 {
                0 => 94.5,
                1 => 81.0,
                2 => 61.5,
                _ => 38.0,
            };
            let jitter = ((m * 29 + i * 13) as f64).sin() * 0.12;
            for &(atom_name, elem, off) in ATOM_TEMPLATES {
                let atom_field = if atom_name.len() == 4 {
                    atom_name.to_string()
                } else {
                    format!(" {atom_name:<3}")
                };
                pdb.push_str(&format!(
                    "ATOM  {:>5} {} {:>3} A{:>4}    {:>8.3}{:>8.3}{:>8.3}  1.00{:>6.2}          {:>2}\n",
                    serial % 100_000,
                    atom_field,
                    res_name,
                    seq_id,
                    bx + off[0] + jitter,
                    by + off[1],
                    bz + off[2],
                    plddt,
                    elem
                ));
                serial += 1;
            }
        }
        pdb.push_str("ENDMDL\n");
    }
    pdb.push_str("END\n");
    pdb
}

/// Build an all-backbone-atom polypeptide via forward NeRF placement with prescribed `(phi, psi)`
/// angles (`deg`) so Adzhubei et al. (2013) `PPII` (`phi = -75, psi = +145`) is tested directly.
fn build_backbone_pdb_from_torsions(res_names: &[&str], phi_psi_deg: &[(f64, f64)]) -> String {
    fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }
    fn norm(v: [f64; 3]) -> [f64; 3] {
        let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-12);
        [v[0] / l, v[1] / l, v[2] / l]
    }
    fn nerf(
        a: [f64; 3],
        b: [f64; 3],
        c: [f64; 3],
        r: f64,
        theta_deg: f64,
        chi_deg: f64,
    ) -> [f64; 3] {
        let bc = norm([c[0] - b[0], c[1] - b[1], c[2] - b[2]]);
        let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let n = norm(cross(ab, bc));
        let m = cross(n, bc);
        // Supplement angle along extension of b->c
        let sup = (180.0 - theta_deg).to_radians();
        let chi = chi_deg.to_radians();
        let dx = r * sup.cos();
        let dy = r * sup.sin() * chi.cos();
        let dz = -r * sup.sin() * chi.sin();
        [
            c[0] + dx * bc[0] + dy * m[0] + dz * n[0],
            c[1] + dx * bc[1] + dy * m[1] + dz * n[1],
            c[2] + dx * bc[2] + dy * m[2] + dz * n[2],
        ]
    }

    let mut n_prev = [0.0, 1.458, 0.0];
    let mut ca_prev = [0.0, 0.0, 0.0];
    let mut c_prev = [1.525, 0.0, 0.0];
    let mut pdb = String::from("HEADER    SYNTHETIC PPII PEPTIDE\n");
    let mut serial = 1u32;

    for (i, &rn) in res_names.iter().enumerate() {
        let (phi, psi) = phi_psi_deg[i];
        let (n_i, ca_i, c_i) = if i == 0 {
            (n_prev, ca_prev, c_prev)
        } else {
            let psi_prev = phi_psi_deg[i - 1].1;
            let n_cur = nerf(n_prev, ca_prev, c_prev, 1.329, 116.2, psi_prev);
            let ca_cur = nerf(ca_prev, c_prev, n_cur, 1.458, 121.7, 180.0);
            let c_cur = nerf(c_prev, n_cur, ca_cur, 1.525, 111.2, phi);
            n_prev = n_cur;
            ca_prev = ca_cur;
            c_prev = c_cur;
            (n_cur, ca_cur, c_cur)
        };
        let o_i = nerf(n_i, ca_i, c_i, 1.231, 120.8, psi + 180.0);
        for (aname, elem, p) in [
            ("N", "N", n_i),
            ("CA", "C", ca_i),
            ("C", "C", c_i),
            ("O", "O", o_i),
        ] {
            pdb.push_str(&format!(
                "ATOM  {:>5}  {:<3} {:>3} A{:>4}    {:>8.3}{:>8.3}{:>8.3}  1.00 88.00          {:>2}\n",
                serial,
                aname,
                rn,
                i + 1,
                p[0],
                p[1],
                p[2],
                elem
            ));
            serial += 1;
        }
    }
    pdb.push_str("END\n");
    pdb
}

#[test]
fn t21_3_three_track_vector_geometry_partial_ssp_under_1ms_and_themes() {
    let ubiq_ref = read_star_str(SYNTHETIC_UBIQUITIN_STR, None).expect("parse ubiquitin STAR");
    let seq_len = ubiq_ref.sequence.len();
    assert_eq!(seq_len, 76);

    // Build a partial (45% assigned) shift table to verify live Track 2 SSP + hollow unassigned dashes
    let mut partial_ref = Reference::new(ubiq_ref.sequence.clone(), ubiq_ref.seq_ids.clone());
    for (&(k, ref atom), &ppm) in &ubiq_ref.shifts {
        if k < 34 {
            partial_ref.shifts.insert((k, atom.clone()), ppm);
        }
    }

    let mut state = SsRibbonState::new();
    let prop_cfg = PropensityConfig::default();

    // Warm up once, then measure live incremental Track 2 SSP update across partial -> full assignment
    let _ = state.sync_nmr_shifts(Some(&partial_ref), &prop_cfg);
    // Repeated sync with identical shifts must hit fingerprint cache (false, zero recompute)
    assert!(
        !state.sync_nmr_shifts(Some(&partial_ref), &prop_cfg),
        "unchanged shift table must short-circuit via fingerprint cache"
    );

    // Incremental shift mutation: add residue 35..50 shifts and verify `< 1.0 ms` update gate
    for (&(k, ref atom), &ppm) in &ubiq_ref.shifts {
        if (34..50).contains(&k) {
            partial_ref.shifts.insert((k, atom.clone()), ppm);
        }
    }
    let t0 = Instant::now();
    let updated = state.sync_nmr_shifts(Some(&partial_ref), &prop_cfg);
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    assert!(
        updated,
        "mutated shift table must trigger incremental SSP update"
    );
    let ssp_budget_ms = if cfg!(debug_assertions) { 15.0 } else { 1.0 };
    assert!(
        elapsed_ms < ssp_budget_ms && state.last_nmr_update_ms < ssp_budget_ms,
        "Track 2 live SSP update took {elapsed_ms:.3} ms (reported {:.3} ms), exceeding {ssp_budget_ms:.1} ms gate",
        state.last_nmr_update_ms
    );

    // Populate all 4 Track-1 states (Helix, Sheet, Ppii, Coil) and all 5 Track-3 concordance states
    let mut ss3d = vec![SsState4::Coil; seq_len];
    for slot in &mut ss3d[2..9] {
        *slot = SsState4::Sheet;
    }
    for slot in &mut ss3d[12..17] {
        *slot = SsState4::Ppii;
    }
    for slot in &mut ss3d[22..34] {
        *slot = SsState4::Helix;
    }
    for slot in &mut ss3d[40..46] {
        *slot = SsState4::Sheet;
    }
    state.set_ss3d_slice(&ss3d);

    let mut concordance = vec![ConcordanceTrackClass::ConcordantOrdered; seq_len];
    for slot in &mut concordance[9..15] {
        *slot = ConcordanceTrackClass::ConcordantDisordered;
    }
    for slot in &mut concordance[18..24] {
        *slot = ConcordanceTrackClass::ConditionalFoldingOrCrystalArtifact;
    }
    for slot in &mut concordance[35..41] {
        *slot = ConcordanceTrackClass::DynamicHingeOrAltState;
    }
    for slot in &mut concordance[55..seq_len] {
        *slot = ConcordanceTrackClass::UnassignedNmr;
    }
    state.set_concordance_override(&concordance);

    // Render headlessly in BOTH Paper White and Dark Oscilloscope themes and inspect shapes
    let ctx = egui::Context::default();
    for paper_white in [true, false] {
        let theme = SsRibbonTheme::for_mode(paper_white);
        assert_eq!(theme.helix_coral, egui::Color32::from_rgb(0xE0, 0x5A, 0x47));
        assert_eq!(theme.sheet_teal, egui::Color32::from_rgb(0x2A, 0x9D, 0x8F));
        assert_eq!(theme.ppii_violet, egui::Color32::from_rgb(0x7B, 0x61, 0xFF));
        assert_eq!(
            theme.concordant_ordered,
            egui::Color32::from_rgb(0x16, 0xA3, 0x4A)
        );
        assert_eq!(
            theme.concordant_disordered,
            egui::Color32::from_rgb(0x64, 0x74, 0x8B)
        );
        assert_eq!(
            theme.conditional_folding,
            egui::Color32::from_rgb(0xD9, 0x77, 0x06)
        );
        assert_eq!(
            theme.dynamic_hinge,
            egui::Color32::from_rgb(0xDC, 0x26, 0x26)
        );

        let raw = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1280.0, 220.0),
            )),
            ..Default::default()
        };
        let mut full_out = ctx.run_ui(raw, |ui| {
            egui::CentralPanel::default().show(ui, |ui| {
                let _ = SsRibbonPanel::show(
                    ui,
                    &mut state,
                    &SsRibbonWorkspaceInput {
                        sequence: &ubiq_ref.sequence,
                        seq_ids: &ubiq_ref.seq_ids,
                        paper_white,
                        has_3d_structure: true,
                    },
                );
            });
        });
        full_out.textures_delta.clear();

        assert!(
            full_out.shapes.len() > 80,
            "expected rich vector paint shapes in paper_white={paper_white}, got {}",
            full_out.shapes.len()
        );
        assert_eq!(state.last_col_rects.len(), seq_len);
        assert!(state.last_track_rects.iter().all(Option::is_some));

        let counts = state.last_prim_counts;
        assert!(counts.track1_helix_wave_points >= 50);
        assert_eq!(counts.track1_sheet_arrows, 2);
        assert_eq!(counts.track1_ppii_diamonds, 5);
        assert!(counts.track1_coil_segments >= 3);
        assert!(counts.track2_upward_helix_bars >= 8);
        assert!(counts.track2_downward_strand_bars >= 8);
        assert!(counts.track2_unassigned_dashes >= 20);
        assert_eq!(counts.track3_concordance_blocks, seq_len);
        assert!(counts.track3_hatch_strokes >= 6);
        assert_eq!(counts.track3_double_borders, 6);
    }

    // Verify horizontal zoom scaling & reset
    let base_w = state.col_width_px;
    state.zoom_by(1.5);
    assert!((state.col_width_px - base_w * 1.5).abs() < 1e-4);
    state.reset_zoom();
    assert!((state.col_width_px - base_w).abs() < 1e-4);

    // Also verify Adzhubei (2013) PPII coordinate detection on a synthetic non-H-bonded PPII segment
    let ppii_pdb = build_backbone_pdb_from_torsions(
        &["ALA", "PRO", "ALA", "PRO", "GLY", "ALA"],
        &[
            (-75.0, 145.0),
            (-75.0, 145.0),
            (-75.0, 145.0),
            (-75.0, 145.0),
            (-75.0, 145.0),
            (-75.0, 145.0),
        ],
    );
    let ppii_ens = parse_pdb_ensemble(&ppii_pdb).expect("parse synthetic PPII peptide");
    let n_ppii = ppii_ens
        .primary_ss4
        .iter()
        .filter(|&&s| s == SsState4::Ppii)
        .count();
    assert!(
        n_ppii >= 3,
        "expected >= 3 PPII residues in (-75, +145) polyproline-II peptide, got {:?}",
        ppii_ens.primary_ss4
    );
}

#[test]
fn t21_3_live_concordance_regimes_ordered_disordered_morf_and_hinge() {
    let ubiq_ref = read_star_str(SYNTHETIC_UBIQUITIN_STR, None).expect("parse ubiquitin STAR");
    let mut ubiq_ens = parse_pdb_ensemble(UBIQUITIN_1UBQ_PDB).expect("parse 1UBQ");
    ubiq_ens.bind_to_nmr_sequence(&ubiq_ref.sequence, &ubiq_ref.seq_ids);

    // 1. Experimental 1UBQ (`per_residue_plddt: None`) must NOT fabricate 85.0 pLDDT -> ord_cnt == 0;
    //    when explicit per-residue pLDDT is present, folded Ubiquitin is predominantly ConcordantOrdered
    let mut state = SsRibbonState::new();
    state.sync_from_ensemble_and_nmr(
        ubiq_ref.sequence.len(),
        Some(&ubiq_ens),
        Some(&ubiq_ref),
        &PropensityConfig::default(),
        None,
    );
    assert_eq!(
        state.concordance_summary.class_counts[0], 0,
        "experimental 1UBQ without pLDDT must not fabricate ConcordantOrdered"
    );
    assert_eq!(state.rigidity_source_label(), "RCI-S² (shift-derived)");

    ubiq_ens.per_residue_plddt = Some(vec![92.0; 76]);
    state.sync_from_ensemble_and_nmr(
        ubiq_ref.sequence.len(),
        Some(&ubiq_ens),
        Some(&ubiq_ref),
        &PropensityConfig::default(),
        None,
    );
    let ord_cnt = state.concordance_summary.class_counts[0];
    assert!(
        ord_cnt >= 50,
        "expected >= 50 ConcordantOrdered residues on Ubiquitin with pLDDT, got {ord_cnt} ({:?})",
        state.concordance_summary.class_counts
    );

    // 2. Planted regimes across realistic contiguous segments:
    //    - 0..=1: ConcordantOrdered (3D Helix, pLDDT=93, helical NMR shifts, S^2=0.87)
    //    - 2..=3: ConcordantDisordered (3D Coil, pLDDT=41, coil NMR shifts, S^2=0.30)
    //    - 4..=6: ConditionalFoldingOrCrystalArtifact (3D Helix, pLDDT=88, random-coil NMR shifts, S^2=0.34)
    //    - 7..=8: DynamicHingeOrAltState (3D Helix, pLDDT=91, strong beta-strand NMR shifts, S^2=0.86)
    //    - 9    : UnassignedNmr (0 backbone shifts, no S^2)
    let seq: Vec<String> = [
        "ALA", "ALA", "GLY", "SER", "GLU", "ALA", "GLN", "VAL", "ILE", "GLY",
    ]
    .into_iter()
    .map(str::to_string)
    .collect();
    let seq_ids: Vec<i32> = (1..=10).collect();
    let mut custom_ref = Reference::new(seq, seq_ids);
    for k in [0usize, 1] {
        custom_ref.shifts.insert((k, "CA".to_string()), 56.4);
        custom_ref.shifts.insert((k, "CB".to_string()), 17.3);
    }
    custom_ref.shifts.insert((2, "CA".to_string()), 45.1);
    custom_ref.shifts.insert((2, "C".to_string()), 173.9);
    custom_ref.shifts.insert((3, "CA".to_string()), 58.3);
    custom_ref.shifts.insert((3, "CB".to_string()), 63.8);
    custom_ref.shifts.insert((4, "CA".to_string()), 56.6);
    custom_ref.shifts.insert((4, "CB".to_string()), 30.0);
    custom_ref.shifts.insert((5, "CA".to_string()), 52.5);
    custom_ref.shifts.insert((5, "CB".to_string()), 19.1);
    custom_ref.shifts.insert((6, "CA".to_string()), 55.7);
    custom_ref.shifts.insert((6, "CB".to_string()), 29.4);
    custom_ref.shifts.insert((7, "CA".to_string()), 58.5);
    custom_ref.shifts.insert((7, "CB".to_string()), 36.4);
    custom_ref.shifts.insert((8, "CA".to_string()), 57.6);
    custom_ref.shifts.insert((8, "CB".to_string()), 42.1);

    let prop =
        nmr_core::idp_propensity::compute_idp_propensity(&custom_ref, &PropensityConfig::default());
    let ss3d = [
        SsState4::Helix,
        SsState4::Helix,
        SsState4::Coil,
        SsState4::Coil,
        SsState4::Helix,
        SsState4::Helix,
        SsState4::Helix,
        SsState4::Helix,
        SsState4::Helix,
        SsState4::Coil,
    ];
    let plddt = [
        Some(93.0),
        Some(92.0),
        Some(41.0),
        Some(39.0),
        Some(88.0),
        Some(90.0),
        Some(87.0),
        Some(91.0),
        Some(92.0),
        Some(78.0),
    ];
    let mut exp_s2 = BTreeMap::new();
    for k in [0, 1] {
        exp_s2.insert(k, 0.87);
    }
    for k in [2, 3] {
        exp_s2.insert(k, 0.30);
    }
    for k in [4, 5, 6] {
        exp_s2.insert(k, 0.34);
    }
    for k in [7, 8] {
        exp_s2.insert(k, 0.86);
    }

    let summary = evaluate_concordance_with_summary(
        10,
        Some(&ss3d),
        Some(&plddt),
        Some(&prop),
        Some(&exp_s2),
    );
    assert_eq!(summary.classes[0], ConcordanceTrackClass::ConcordantOrdered);
    assert_eq!(
        summary.classes[2],
        ConcordanceTrackClass::ConcordantDisordered
    );
    assert_eq!(
        summary.classes[5],
        ConcordanceTrackClass::ConditionalFoldingOrCrystalArtifact
    );
    assert_eq!(
        summary.classes[7],
        ConcordanceTrackClass::DynamicHingeOrAltState
    );
    assert_eq!(summary.classes[9], ConcordanceTrackClass::UnassignedNmr);
}

#[test]
fn t21_3_bidirectional_click_syncs_crosshair_sidechain_and_3d_highlight() {
    let ubiq_ref = read_star_str(SYNTHETIC_UBIQUITIN_STR, None).expect("parse ubiquitin STAR");
    let mut mol3d = Mol3dViewerPanel::new();
    mol3d
        .load_structure_text(UBIQUITIN_1UBQ_PDB, &ubiq_ref.sequence, &ubiq_ref.seq_ids)
        .expect("load 1UBQ into Mol3dViewerPanel");

    let mut ss_state = SsRibbonState::new();
    ss_state.sync_from_ensemble_and_nmr(
        ubiq_ref.sequence.len(),
        mol3d.ensemble.as_ref(),
        Some(&ubiq_ref),
        &PropensityConfig::default(),
        None,
    );

    let ctx = egui::Context::default();
    let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(2000.0, 200.0));

    // Frame 1: lay out columns
    let mut out1 = ctx.run_ui(
        egui::RawInput {
            screen_rect: Some(screen),
            ..Default::default()
        },
        |ui| {
            egui::CentralPanel::default().show(ui, |ui| {
                let _ = SsRibbonPanel::show(
                    ui,
                    &mut ss_state,
                    &SsRibbonWorkspaceInput {
                        sequence: &ubiq_ref.sequence,
                        seq_ids: &ubiq_ref.seq_ids,
                        paper_white: false,
                        has_3d_structure: true,
                    },
                );
            });
        },
    );
    out1.textures_delta.clear();

    // Click residue index 26 (`LYS 27` in Ubiquitin alpha-helix)
    let target_k = 26usize;
    let click_pos = ss_state.last_col_rects[target_k].center();
    let mut ribbon_out = Default::default();
    let mut out2 = ctx.run_ui(
        egui::RawInput {
            screen_rect: Some(screen),
            events: vec![
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
            ],
            ..Default::default()
        },
        |ui| {
            egui::CentralPanel::default().show(ui, |ui| {
                ribbon_out = SsRibbonPanel::show(
                    ui,
                    &mut ss_state,
                    &SsRibbonWorkspaceInput {
                        sequence: &ubiq_ref.sequence,
                        seq_ids: &ubiq_ref.seq_ids,
                        paper_white: false,
                        has_3d_structure: true,
                    },
                );
            });
        },
    );
    out2.textures_delta.clear();

    assert_eq!(ribbon_out.clicked_residue, Some(target_k));
    assert_eq!(ss_state.selected_residue, Some(target_k));

    let mut crosshair = SyncCrosshair::default();
    let mut sc_state = SequenceRibbonState::new();
    apply_ss_ribbon_click_sync(
        target_k,
        Some(&ubiq_ref),
        &mut crosshair,
        &mut sc_state,
        &mut mol3d,
    );
    mol3d.flush_pending_updates();

    let exp_h = ubiq_ref.shifts[&(target_k, "H".to_string())];
    let exp_n = ubiq_ref.shifts[&(target_k, "N".to_string())];
    let exp_ca = ubiq_ref.shifts[&(target_k, "CA".to_string())];
    assert!((crosshair.h_ppm - exp_h).abs() < 1e-9);
    assert!((crosshair.n_ppm - exp_n).abs() < 1e-9);
    assert_eq!(crosshair.c_ppm, Some(exp_ca));
    assert_eq!(sc_state.selected_residue, target_k);
    assert_eq!(mol3d.scene_cfg.highlighted_seq_idx, Some(target_k));

    let k27_ca = mol3d
        .ensemble
        .as_ref()
        .and_then(|e| e.models[0].chains[0].residues[target_k].atom_pos("CA"))
        .expect("LYS 27 CA");
    assert!((mol3d.camera.target[0] - (k27_ca[0] as f64)).abs() < 1e-5);
    assert!((mol3d.camera.target[1] - (k27_ca[1] as f64)).abs() < 1e-5);
    assert!((mol3d.camera.target[2] - (k27_ca[2] as f64)).abs() < 1e-5);

    // Reverse direction: analytical 3D ray-pick at viewport center resolves LYS 27 and syncs ribbon
    let vp_w = 800.0;
    let vp_h = 600.0;
    let (px, py, _) = mol3d
        .camera
        .project_world_to_screen(
            [k27_ca[0] as f64, k27_ca[1] as f64, k27_ca[2] as f64],
            vp_w,
            vp_h,
        )
        .expect("projected CA inside viewport");
    let picked = mol3d
        .pick_at_viewport_pixel(
            px,
            py,
            vp_w,
            vp_h,
            &ubiq_ref.sequence,
            &ubiq_ref.seq_ids,
            Some(&ubiq_ref),
        )
        .expect("3D ray pick hit");
    ss_state.selected_residue = Some(picked.residue_idx);
    assert_eq!(picked.residue_idx, target_k);
    assert_eq!(ss_state.selected_residue, Some(target_k));
}

#[test]
fn t21_3_plddt_mmcif_qa_local_bfactor_guard_and_2ms_putty_benchmark() {
    // 1. Verify experimental X-ray (`1UBQ`) B-factor guard: thermal B-factors MUST NOT become pLDDT
    let xray_ens = parse_pdb_ensemble(UBIQUITIN_1UBQ_PDB).expect("parse 1UBQ");
    assert_eq!(xray_ens.provenance, CoordinateProvenance::Experimental);
    assert!(
        xray_ens.per_residue_plddt.is_none(),
        "X-RAY DIFFRACTION thermal B-factors must never be misinterpreted as AlphaFold pLDDT"
    );

    // 2. Verify experimental SOLUTION NMR mmCIF guard (`_exptl.method 'SOLUTION NMR'`)
    let nmr_cif = "\
data_nmr_test
_exptl.method 'SOLUTION NMR'
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.B_iso_or_equiv
ATOM 1 N ALA A 1 0.000 0.000 0.000 42.0
ATOM 2 CA ALA A 1 1.458 0.000 0.000 42.0
ATOM 3 C ALA A 1 2.009 1.420 0.000 42.0
ATOM 4 O ALA A 1 1.251 2.390 0.000 42.0
";
    let nmr_ens = parse_mmcif_ensemble(nmr_cif).expect("parse SOLUTION NMR mmCIF");
    assert_eq!(nmr_ens.provenance, CoordinateProvenance::Experimental);
    assert!(nmr_ens.per_residue_plddt.is_none());

    // 3. Verify ModelArchive `_ma_qa_metric_local` mmCIF extraction (exact values, ignoring B_iso)
    let af_cif = "\
data_AF_Q9Y261_F1
loop_
_ma_qa_metric.id
_ma_qa_metric.name
_ma_qa_metric.mode
_ma_qa_metric.type
1 pLDDT local pLDDT
2 pTM global pTM
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.B_iso_or_equiv
ATOM 1 CA MET A 1 0.000 0.000 0.000 12.00
ATOM 2 CA GLN A 2 3.800 0.000 0.000 12.00
ATOM 3 CA ILE A 3 7.600 0.000 0.000 12.00
ATOM 4 CA PHE A 4 11.400 0.000 0.000 12.00
loop_
_ma_qa_metric_local.label_asym_id
_ma_qa_metric_local.label_seq_id
_ma_qa_metric_local.label_comp_id
_ma_qa_metric_local.metric_id
_ma_qa_metric_local.metric_value
A 1 MET 1 95.40
A 2 GLN 1 82.10
A 3 ILE 1 63.50
A 4 PHE 1 41.25
";
    let af_ens = parse_structure_auto(af_cif).expect("parse AFDB mmCIF");
    assert_eq!(
        af_ens.provenance,
        CoordinateProvenance::ModelArchiveQaMetric
    );
    let plddt = af_ens.per_residue_plddt.as_ref().expect("extracted pLDDT");
    assert_eq!(plddt.len(), 4);
    assert!((plddt[0] - 95.40).abs() < 1e-6);
    assert!((plddt[1] - 82.10).abs() < 1e-6);
    assert!((plddt[2] - 63.50).abs() < 1e-6);
    assert!((plddt[3] - 41.25).abs() < 1e-6);

    // 4. Verify `[0, 1] -> [0, 100]` automatic unit-interval rescaling on fractional pLDDT models
    let unit_cif = af_cif
        .replace("95.40", "0.954")
        .replace("82.10", "0.821")
        .replace("63.50", "0.635")
        .replace("41.25", "0.4125");
    let unit_ens = parse_mmcif_ensemble(&unit_cif).expect("parse unit-scale pLDDT mmCIF");
    assert!(unit_ens.plddt_rescaled_from_unit);
    let unit_plddt = unit_ens.per_residue_plddt.as_ref().expect("rescaled pLDDT");
    assert!((unit_plddt[0] - 95.4).abs() < 1e-4);

    // 5. Hostile input checks: NaN in `_ma_qa_metric_local` and truncated `_ma_qa_metric_local` loop
    let nan_cif = af_cif.replace("63.50", "NaN");
    assert!(
        parse_mmcif_ensemble(&nan_cif).is_err(),
        "NaN pLDDT must return Err"
    );
    let trunc_qa_cif = format!("{af_cif}A 5 LEU 1\n");
    assert!(
        parse_mmcif_ensemble(&trunc_qa_cif).is_err(),
        "truncated _ma_qa_metric_local row must return Err"
    );

    // 6. Verify canonical 4-band Varadi (2022) palette & disorder putty formula + AfNmrConcordance putty
    assert_eq!(
        alphafold_plddt_rgba(92.0, 1.0),
        [
            0x00 as f32 / 255.0,
            0x53 as f32 / 255.0,
            0xD6 as f32 / 255.0,
            1.0
        ]
    );
    assert_eq!(
        alphafold_plddt_rgba(78.0, 1.0),
        [
            0x65 as f32 / 255.0,
            0xCB as f32 / 255.0,
            0xF3 as f32 / 255.0,
            1.0
        ]
    );
    assert_eq!(
        alphafold_plddt_rgba(58.0, 1.0),
        [
            0xFF as f32 / 255.0,
            0xDB as f32 / 255.0,
            0x13 as f32 / 255.0,
            1.0
        ]
    );
    assert_eq!(
        alphafold_plddt_rgba(34.0, 1.0),
        [
            0xFF as f32 / 255.0,
            0x7D as f32 / 255.0,
            0x45 as f32 / 255.0,
            1.0
        ]
    );

    let mut td = QuantitativeTrackData {
        active_track: QuantitativeTrack::AlphaFoldPlddt,
        putty_enabled: true,
        putty_min_radius: 0.22,
        putty_max_radius: 1.22,
        ..QuantitativeTrackData::default()
    };
    td.set_plddt_dense(&[95.0, 80.0, 60.0, 25.0]);

    let s95 = td.evaluate_residue_style(Some(0), [0.0; 4], 0.28, None, 1.0);
    let s25 = td.evaluate_residue_style(Some(3), [0.0; 4], 0.28, None, 1.0);
    let s_missing = td.evaluate_residue_style(Some(99), [0.0; 4], 0.28, None, 1.0);
    assert!((s95.radius_angstrom - (0.22 + 0.05 * 1.00)).abs() < 1e-5);
    assert!((s25.radius_angstrom - (0.22 + 0.75 * 1.00)).abs() < 1e-5);
    assert!(s25.radius_angstrom > s95.radius_angstrom);
    assert_eq!(s_missing.rgba, UNASSIGNED_PUTTY_RGBA);

    td.active_track = QuantitativeTrack::AfNmrConcordance;
    td.set_concordance_slice(&[
        ConcordanceTrackClass::ConcordantOrdered,
        ConcordanceTrackClass::ConcordantDisordered,
        ConcordanceTrackClass::ConditionalFoldingOrCrystalArtifact,
        ConcordanceTrackClass::DynamicHingeOrAltState,
        ConcordanceTrackClass::UnassignedNmr,
    ]);
    let c_ord = td.evaluate_residue_style(Some(0), [0.0; 4], 0.28, None, 1.0);
    let c_dis = td.evaluate_residue_style(Some(1), [0.0; 4], 0.28, None, 1.0);
    let c_morf = td.evaluate_residue_style(Some(2), [0.0; 4], 0.28, None, 1.0);
    let c_hinge = td.evaluate_residue_style(Some(3), [0.0; 4], 0.28, None, 1.0);
    let c_unasg = td.evaluate_residue_style(Some(4), [0.0; 4], 0.28, None, 1.0);
    assert!(c_hinge.radius_angstrom > c_morf.radius_angstrom);
    assert!(c_morf.radius_angstrom > c_dis.radius_angstrom);
    assert!(c_dis.radius_angstrom > c_ord.radius_angstrom);
    assert_eq!(c_unasg.rgba, UNASSIGNED_PUTTY_RGBA);

    // 7. `< 2.0 ms` in-place putty update gate across a 150-aa 20-conformer (51,000-atom) ensemble!
    let bench_pdb = synthesize_ensemble_pdb(20, 150);
    let bench_ens = nmr_gui::mol3d_viewer::parse_pdb_ensemble_with_plddt_override(&bench_pdb, true)
        .expect("parse 20x150 ensemble");
    assert_eq!(bench_ens.models.len(), 20);
    assert_eq!(bench_ens.primary_sequence.len(), 150);
    assert_eq!(
        bench_ens.provenance,
        CoordinateProvenance::ComputationalBFactor
    );
    assert_eq!(
        bench_ens.per_residue_plddt.as_ref().map(Vec::len),
        Some(150)
    );

    let cfg = SceneBuildConfig {
        style: RepresentationStyle::CartoonAndSidechains,
        show_all_ensemble_models: true,
        ..SceneBuildConfig::default()
    };
    let mut bench_tracks = QuantitativeTrackData {
        putty_enabled: true,
        ..QuantitativeTrackData::default()
    };
    if let Some(ref pv) = bench_ens.per_residue_plddt {
        bench_tracks.set_plddt_dense(pv);
    }
    let conc_vec: Vec<ConcordanceTrackClass> = (0..150)
        .map(|k| ConcordanceTrackClass::ALL[k % ConcordanceTrackClass::ALL.len()])
        .collect();
    bench_tracks.set_concordance_slice(&conc_vec);

    let mut scene = build_mol3d_scene(&bench_ens, &cfg, &bench_tracks, None, 1);
    assert!(scene.opaque_ribbon_verts.len() + scene.blend_ribbon_verts.len() >= 90_000);

    for track in [
        QuantitativeTrack::AlphaFoldPlddt,
        QuantitativeTrack::AfNmrConcordance,
    ] {
        bench_tracks.active_track = track;
        // Warm up caches
        scene.update_quantitative_putty(&bench_tracks);

        let t0 = Instant::now();
        let iters = 5u32;
        for _ in 0..iters {
            scene.update_quantitative_putty(&bench_tracks);
        }
        let avg_ms = (t0.elapsed().as_secs_f64() * 1000.0) / f64::from(iters);
        let putty_budget_ms = if cfg!(debug_assertions) { 15.0 } else { 2.0 };
        assert!(
            avg_ms < putty_budget_ms,
            "{track:?} putty update across 20x150aa ensemble averaged {avg_ms:.3} ms (gate < {putty_budget_ms:.1} ms)"
        );
    }
}

#[test]
fn t21_3_zero_emojis_across_all_owned_gui_files() {
    fn is_emoji_codepoint(ch: char) -> bool {
        let u = ch as u32;
        // Pictographs, emoticons, transport, supplemental symbols, dingbats, misc symbols, FE0F VS16
        (0x1F000..=0x1FFFF).contains(&u)
            || (0x2600..=0x27BF).contains(&u)
            || u == 0xFE0F
            || u == 0x200D
    }

    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let owned_rel_paths = [
        "src/lib.rs",
        "src/ss_ribbon_widget.rs",
        "src/mol3d_overlay.rs",
        "src/mol3d_viewer.rs",
        "src/main.rs",
        "tests/ss_ribbon_widget_tests.rs",
    ];

    for rel in owned_rel_paths {
        let full_path = manifest_dir.join(rel);
        let content = std::fs::read_to_string(&full_path)
            .unwrap_or_else(|e| panic!("failed reading {}: {e}", full_path.display()));
        for (line_idx, line) in content.lines().enumerate() {
            for ch in line.chars() {
                assert!(
                    !is_emoji_codepoint(ch),
                    "Forbidden emoji character U+{:04X} ('{ch}') at {}:{}",
                    ch as u32,
                    rel,
                    line_idx + 1
                );
            }
        }
    }
}
