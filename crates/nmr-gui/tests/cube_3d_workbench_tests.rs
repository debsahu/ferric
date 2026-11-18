//! Comprehensive headless verification suite for the 3D NMR Cube Workbench, 3-axis orthogonal
//! slicing, skyline MIP slab projections, physical nucleus axis inference, 3D peak picking &
//! mouse curation, HSQC->3D `(1H, 15N)` cursor/strip synchronization, and lossless 3D water
//! suppression undo.
//!
//! Provenance: Original clean-room verification suite built from first-principles 3D Fourier NMR
//! spectroscopy (Ernst, Bodenhausen & Wokaun 1987; Kay, Ikura, Tschudin & Bax 1990).

use std::collections::BTreeMap;

use eframe::egui;
use nmr_core::solver::Matrix2D;
use nmr_core::spectrum::{AxisPpm, Spectrum2D, Spectrum3D};
use nmr_core::spinsystems::SpinSystem;
use nmr_core::water::WaterSuppressionConfig;
use nmr_gui::contour_painter::SyncCrosshair;
use nmr_gui::strip_tiler::{
    strip_vertical_axis_metadata, StripSpectra3D, StripTilerContext, StripTilerPanel,
};
use nmr_gui::workbook::{
    build_showcase_small_molecule_1h_trace, extract_3d_plane, extract_3d_skyline,
    infer_3d_cube_nuclei, CubePlaneOrientation, PlanePeakVisibility, WorkbookPageKind,
    WorkbookState,
};

/// Build a synthetic `3D H(CCO)NH` cube (`F3 = 1H_NH 6.2..10.2 ppm @ 600.13 MHz`,
/// `F2 = 15N 106..130 ppm @ 60.81 MHz`, `F1 = 1H_aliph -0.5..9.5 ppm @ 600.13 MHz`)
/// deliberately setting `axis_z.label = "13C"` to simulate a Bruker `proc3s` `AXNUC=<off>` header
/// fallback and verify physical Larmor-ratio nucleus resolution.
fn make_synthetic_hccconh_cube(with_water_slab: bool) -> Spectrum3D {
    let (nx, ny, nz) = (40usize, 24usize, 48usize);
    let axis_x = AxisPpm::from_bounds("1H", nx, 10.2, 6.2, 600.13);
    let axis_y = AxisPpm::from_bounds("15N", ny, 130.0, 106.0, 60.81);
    // Deliberately pass parser-fallback "13C" label with 600.13 MHz 1H frequency!
    let axis_z = AxisPpm::from_bounds("13C", nz, 9.5, -0.5, 600.13);

    let mut data = vec![0.0_f32; nx * ny * nz];
    for (i, slot) in data.iter_mut().enumerate() {
        *slot = (((i * 73 + 19) % 53) as f32 - 26.0) * 0.04;
    }

    let add_3d_gaussian = |buf: &mut [f32], h_nh: f64, n15: f64, h_ali: f64, amp: f32| {
        let cx = axis_x.ppm_to_index(h_nh);
        let cy = axis_y.ppm_to_index(n15);
        let cz = axis_z.ppm_to_index(h_ali);
        for iz in 0..nz {
            let dz = (iz as f64 - cz) / 1.15;
            if dz.abs() > 4.0 {
                continue;
            }
            for iy in 0..ny {
                let dy = (iy as f64 - cy) / 1.10;
                if dy.abs() > 4.0 {
                    continue;
                }
                for ix in 0..nx {
                    let dx = (ix as f64 - cx) / 1.15;
                    if dx.abs() > 4.0 {
                        continue;
                    }
                    buf[(iz * ny + iy) * nx + ix] +=
                        amp * (-0.5 * (dx * dx + dy * dy + dz * dz) as f32).exp();
                }
            }
        }
    };

    // Spin system A at (8.42, 118.5): aliphatic 1H cross-peaks at 1.42 ppm (HB) and 4.28 ppm (HA),
    // plus a homonuclear diagonal self-peak at (8.42, 118.5, 8.42) that must be excluded when exclude_diagonal=true!
    add_3d_gaussian(&mut data, 8.42, 118.5, 1.42, 620.0);
    add_3d_gaussian(&mut data, 8.42, 118.5, 4.28, 410.0);
    add_3d_gaussian(&mut data, 8.42, 118.5, 8.42, 950.0);

    // Spin system B at (7.76, 123.2): negative cross-peak at 2.95 ppm
    add_3d_gaussian(&mut data, 7.76, 123.2, 2.95, -340.0);

    if with_water_slab {
        let cw = axis_x.ppm_to_index(6.45);
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let dw = (ix as f64 - cw) / 0.9;
                    if dw.abs() <= 3.5 {
                        data[(iz * ny + iy) * nx + ix] += 8_500.0 * (-0.5 * (dw * dw) as f32).exp();
                    }
                }
            }
        }
    }

    Spectrum3D {
        name: "3D_HCCCONH".to_string(),
        exp: "HCCCONH".to_string(),
        axis_x,
        axis_y,
        axis_z,
        data,
    }
}

#[test]
fn test_physical_nucleus_inference_and_all_3_orthogonal_orientations_plus_skyline_slab() {
    let hccconh = make_synthetic_hccconh_cube(false);
    assert_eq!(
        infer_3d_cube_nuclei(&hccconh),
        ("1H", "15N", "1H"),
        "Larmor ratio 600.13/600.13 must resolve indirect 1H axis even when proc3s label defaulted to '13C'"
    );

    // Strip tiler vertical axis metadata must also report 1H and exact -0.5..9.5 ppm bounds (never 13..74 or 0..185!)
    let (_, strip_nuc, (s_lo, s_hi)) = strip_vertical_axis_metadata(&hccconh);
    assert_eq!(strip_nuc, "1H");
    assert!((s_lo - (-0.5)).abs() < 1e-6);
    assert!((s_hi - 9.5).abs() < 1e-6);

    // Headless StripTilerPanel render with 3D H(CCO)NH cube verifies zero panics and crosshair sync
    let mut ss0 = SpinSystem::new(8.42, 118.5);
    ss0.cands.insert(("CA".to_string(), 0), vec![(56.0, 2)]);
    let systems = vec![ss0];
    let p_mat = Matrix2D::filled(1, 1, 0.0);
    let seq = vec!["MET".to_string(), "GLN".to_string()];
    let seq_ids = vec![1, 2];
    let mut tiler = StripTilerPanel::new();
    let mut xh = SyncCrosshair::new(8.42, 118.5, Some(1.42));
    let ctx_egui = egui::Context::default();
    let mut out = ctx_egui.run_ui(egui::RawInput::default(), |ui| {
        let sctx = StripTilerContext {
            systems: &systems,
            pair_mat: &p_mat,
            u_mat: None,
            sequence: &seq,
            seq_ids: &seq_ids,
            active_root: 0,
            assignments: None,
            locks: None,
            confirmed_links: None,
            spectra_3d: StripSpectra3D {
                aliphatic_3d: Some(&hccconh),
                carbonyl_3d: None,
            },
        };
        let _ = tiler.show(ui, &sctx, Some(&mut xh));
    });
    out.textures_delta.clear();

    // 1. F3xF2 plane stepped along F1 (1H_NH x 15N at 1H_aliph = 1.42 ppm)
    let iz_hb = hccconh.axis_z.ppm_to_index(1.42).round() as usize;
    let p_xy = extract_3d_plane(&hccconh, CubePlaneOrientation::F3xF2StepF1, iz_hb);
    assert_eq!(p_xy.nx(), hccconh.nx());
    assert_eq!(p_xy.ny(), hccconh.ny());
    assert_eq!(p_xy.axis_x.label, "1H");
    assert_eq!(p_xy.axis_y.label, "15N");

    // 2. F3xF1 strip plane stepped along F2 (1H_NH x 1H_aliph at 15N = 118.5 ppm)
    let iy_n = hccconh.axis_y.ppm_to_index(118.5).round() as usize;
    let p_xz = extract_3d_plane(&hccconh, CubePlaneOrientation::F3xF1StepF2, iy_n);
    assert_eq!(p_xz.nx(), hccconh.nx());
    assert_eq!(p_xz.ny(), hccconh.nz());
    assert_eq!(p_xz.axis_x.label, "1H");
    assert_eq!(p_xz.axis_y.label, "1H");
    assert!((p_xz.axis_y.min_ppm() - (-0.5)).abs() < 1e-6);
    assert!((p_xz.axis_y.max_ppm() - 9.5).abs() < 1e-6);

    // 3. F2xF1 indirect plane stepped along F3 (15N x 1H_aliph at 1H_NH = 8.42 ppm)
    let ix_h = hccconh.axis_x.ppm_to_index(8.42).round() as usize;
    let p_yz = extract_3d_plane(&hccconh, CubePlaneOrientation::F2xF1StepF3, ix_h);
    assert_eq!(p_yz.nx(), hccconh.ny());
    assert_eq!(p_yz.ny(), hccconh.nz());
    assert_eq!(p_yz.axis_x.label, "15N");
    assert_eq!(p_yz.axis_y.label, "1H");

    // 4. 2D Skyline signed Pos+Neg MIP with & without slab bounds
    let sky_full = extract_3d_skyline(&hccconh, CubePlaneOrientation::F3xF2StepF1, None);
    let ix_b = sky_full.axis_x.ppm_to_index(7.76).round() as usize;
    let iy_b = sky_full.axis_y.ppm_to_index(123.2).round() as usize;
    assert!(
        sky_full.get(ix_b, iy_b) < -200.0,
        "Pos+Neg MIP must preserve negative sign of cross-peak B, got {}",
        sky_full.get(ix_b, iy_b)
    );

    // Restricting F1 slab to 0.5..2.0 ppm includes HB (1.42 ppm) on A, excludes B (2.95 ppm)!
    let sky_slab = extract_3d_skyline(
        &hccconh,
        CubePlaneOrientation::F3xF2StepF1,
        Some((0.5, 2.0)),
    );
    let ix_a = sky_slab.axis_x.ppm_to_index(8.42).round() as usize;
    let iy_a = sky_slab.axis_y.ppm_to_index(118.5).round() as usize;
    assert!(sky_slab.get(ix_a, iy_a) > 400.0);
    assert!(sky_slab.get(ix_b, iy_b).abs() < 5.0);
}

#[test]
fn test_3d_peak_picking_projection_ghosts_curation_and_jump_plane() {
    let hccconh = make_synthetic_hccconh_cube(false);
    let mut wb = WorkbookState::default();
    let p_idx = wb.add_3d_cube_page(0, "3D HCCCONH", &hccconh);
    assert_eq!(
        wb.pages_2d[p_idx].kind,
        WorkbookPageKind::Cube3d { cube_idx: 0 }
    );
    assert_eq!(wb.pages_2d[p_idx].nuc_x, "1H");
    assert_eq!(wb.pages_2d[p_idx].nuc_y, "15N");

    let pg = &mut wb.pages_2d[p_idx];
    let n_picked = pg.pick_3d_cube_peaks(&hccconh);
    // Expect the 3 off-diagonal peaks: (8.42, 118.5, 1.42) [+620], (8.42, 118.5, 4.28) [+410], (7.76, 123.2, 2.95) [-340]
    // Homonuclear diagonal (8.42, 118.5, 8.42) must be rejected!
    assert_eq!(
        n_picked, 3,
        "Expected 3 off-diagonal 3D peaks, got {:?}",
        pg.peaks_3d
    );
    assert!(pg.peaks_3d[0].height.abs() >= pg.peaks_3d[1].height.abs());
    assert!(pg.peaks_3d[1].height.abs() >= pg.peaks_3d[2].height.abs());
    assert!((pg.peaks_3d[0].f3_ppm - 8.42).abs() < 0.03);
    assert!((pg.peaks_3d[0].f2_ppm - 118.5).abs() < 0.25);
    assert!((pg.peaks_3d[0].f1_ppm - 1.42).abs() < 0.05);

    // Jump directly to peak #0's F3xF2 plane (F1 = 1.42 ppm): |Δplane| <= 0.5 -> OnPlane
    pg.cube_slice
        .set_orientation(&hccconh, CubePlaneOrientation::F3xF2StepF1);
    let jumped = pg.jump_to_3d_peak_plane(&hccconh, 0);
    assert!(jumped.is_some());
    let pr0_on = pg.cube_slice.project_peak_3d(&hccconh, 0, &pg.peaks_3d[0]);
    assert_eq!(pr0_on.visibility, PlanePeakVisibility::OnPlane);
    assert!(pr0_on.delta_planes.abs() <= 0.5);

    // Step +1 plane (`np`): 0.5 < |Δplane| <= 1.5 -> AdjacentGhost!
    pg.cube_slice.step_plane(&hccconh, 1);
    let pr0_ghost = pg.cube_slice.project_peak_3d(&hccconh, 0, &pg.peaks_3d[0]);
    assert_eq!(pr0_ghost.visibility, PlanePeakVisibility::AdjacentGhost);

    // Step +2 more planes (+3 total): |Δplane| > 1.5 -> OffPlane!
    pg.cube_slice.step_plane(&hccconh, 2);
    let pr0_off = pg.cube_slice.project_peak_3d(&hccconh, 0, &pg.peaks_3d[0]);
    assert_eq!(pr0_off.visibility, PlanePeakVisibility::OffPlane);

    // Interactive click-to-add (`pa`), drag-to-move, parabolic snap (`pc`), relabel (`pl`), and delete (`pd`)
    let _ = pg.jump_to_3d_peak_plane(&hccconh, 0);
    let added_idx = pg.add_3d_peak_on_plane(&hccconh, 8.38, 118.9, false);
    assert_eq!(pg.peaks_3d.len(), 4);
    assert!(pg.relabel_3d_peak(added_idx, "Q2-HB2"));
    assert_eq!(pg.peaks_3d[added_idx].label, "Q2-HB2");

    // Drag slightly off-center, then snap back to true 3D extremum (8.42, 118.5, 1.42)
    assert!(pg.move_3d_peak_on_plane(&hccconh, added_idx, 8.39, 118.8));
    assert!((pg.peaks_3d[added_idx].f3_ppm - 8.39).abs() < 1e-6);
    assert!(pg.snap_3d_peak(&hccconh, added_idx));
    assert!((pg.peaks_3d[added_idx].f3_ppm - 8.42).abs() < 0.04);
    assert!((pg.peaks_3d[added_idx].f2_ppm - 118.5).abs() < 0.30);

    // Delete added peak
    let removed = pg.delete_3d_peak(added_idx);
    assert_eq!(removed.map(|r| r.label).as_deref(), Some("Q2-HB2"));
    assert_eq!(pg.peaks_3d.len(), 3);
}

#[test]
fn test_multi_page_1d_2d_hsqc_3d_sync_and_lossless_3d_water_undo() {
    let (tr_1h, _) = build_showcase_small_molecule_1h_trace();
    assert!(!tr_1h.is_empty());

    let ax_h = AxisPpm::from_bounds("1H", 64, 10.2, 6.2, 600.13);
    let ax_n = AxisPpm::from_bounds("15N", 32, 130.0, 106.0, 60.81);
    let hsqc = Spectrum2D {
        name: "15N-HSQC".to_string(),
        axis_x: ax_h,
        axis_y: ax_n,
        data: vec![1.0_f32; 64 * 32],
    };

    // Build 3D H(CCO)NH with a strong residual H2O ridge at direct F3 = 4.70 ppm (extend F3 to 4.2..10.2)
    let (nx, ny, nz) = (48usize, 20usize, 32usize);
    let cx_ax = AxisPpm::from_bounds("1H", nx, 10.2, 4.2, 600.13);
    let cy_ax = AxisPpm::from_bounds("15N", ny, 130.0, 106.0, 60.81);
    let cz_ax = AxisPpm::from_bounds("1H", nz, 9.5, -0.5, 600.13);
    let mut data = vec![0.0_f32; nx * ny * nz];
    let ix_water = cx_ax.ppm_to_index(4.70).round() as usize;
    let ix_amide = cx_ax.ppm_to_index(8.42).round() as usize;
    let iy_amide = cy_ax.ppm_to_index(118.5).round() as usize;
    let iz_ali = cz_ax.ppm_to_index(1.42).round() as usize;
    for iz in 0..nz {
        for iy in 0..ny {
            data[(iz * ny + iy) * nx + ix_water] = 12_000.0;
        }
    }
    data[(iz_ali * ny + iy_amide) * nx + ix_amide] = 550.0;
    let mut cube = Spectrum3D {
        name: "HCCCONH".to_string(),
        exp: "HCCCONH".to_string(),
        axis_x: cx_ax,
        axis_y: cy_ax,
        axis_z: cz_ax,
        data,
    };
    let original_bytes = cube.data.clone();

    let mut wb = WorkbookState::default();
    let hsqc_page = wb.add_2d_page("2D 15N-HSQC", hsqc, false, true);
    let cube_page = wb.add_3d_cube_page(0, "3D H(CCO)NH", &cube);
    assert_eq!(hsqc_page, 0);
    assert_eq!(cube_page, 1);
    assert_eq!(wb.pages_2d.len(), 2);

    // Sync (1H = 8.42, 15N = 118.5) from HSQC to open 3D cubes (`gs` chord)
    let cubes_vec = vec![cube.clone()];
    let synced = wb.sync_open_3d_cubes_to_hn(&cubes_vec, 8.42, 118.5);
    assert!(synced.is_some());
    let (hit_page, sx, _) = synced.unwrap_or((0, 0.0, 0.0));
    assert_eq!(hit_page, cube_page);
    assert!((sx - 8.42).abs() < 1e-6);
    assert_eq!(
        wb.pages_2d[cube_page].cube_slice.orientation,
        CubePlaneOrientation::F3xF1StepF2,
        "`gs` must flip 3D H(CCO)NH into F3xF1 strip orientation stepped along 15N (F2)"
    );
    assert!(!wb.pages_2d[cube_page].cube_slice.skyline_enabled);
    assert_eq!(wb.pages_2d[cube_page].cube_slice.plane_f2_idx, iy_amide);

    // Verify shared nucleus crosshair map propagates 1H & 15N across 1D/2D/3D pages
    wb.record_1d_cursor_by_nucleus("13C", 56.4);
    assert_eq!(
        wb.shared_crosshair_for_axes("1H", "15N", 0.0, 0.0),
        (8.42, 118.5)
    );
    assert_eq!(
        wb.shared_crosshair_for_axes("15N", "13C", 0.0, 0.0),
        (118.5, 56.4)
    );

    // 3D Water suppression (`ws`) + lossless undo (`wu`) on Cube3d page
    let wcfg = WaterSuppressionConfig::default();
    let carrier = wb.pages_2d[cube_page].suppress_water_on_3d_cube(&mut cube, &wcfg);
    assert!((carrier - 4.70).abs() < 0.25);
    assert!(
        cube.get(ix_water, iy_amide, iz_ali).abs() < 200.0,
        "4.70 ppm direct 1H water wall must be strongly attenuated"
    );
    assert!(
        (cube.get(ix_amide, iy_amide, iz_ali) - 550.0).abs() < 5.0,
        "8.42 ppm amide strip cross-peak must remain intact"
    );

    assert!(wb.pages_2d[cube_page].undo_water_on_3d_cube(&mut cube));
    assert_eq!(
        cube.data, original_bytes,
        "`wu` undo must restore 3D cube voxels losslessly bit-for-bit"
    );
    assert!(wb.pages_2d[cube_page].water_undo_3d.is_none());
    let _ = BTreeMap::<usize, usize>::new();
}
