//! Multi-page NMR spectrum workbook shell, geometric vector chrome icon renderer, publication
//! Paper White / Dark Oscilloscope canvas themes, small-molecule showcase dataset builders, and
//! pixel-aligned 2D `(F2, F1)` 1D marginal projection & slice engines.
//!
//! Provenance: Clean-room original human-computer interaction, geometric vector iconography, and
//! numerical projection implementation built strictly from peer-reviewed NMR spectroscopy
//! literature and open mathematical principles (zero external source code, macros, or proprietary
//! software conventions inspected):
//! - Ernst, R. R., Bodenhausen, G. & Wokaun, A. (*Principles of Nuclear Magnetic Resonance in One
//!   and Two Dimensions*, Oxford University Press, **1987**, Ch. 6) — two-dimensional `(F2, F1)`
//!   frequency-domain representations, homonuclear diagonal symmetry, skyline maximum-magnitude
//!   projections, and orthogonal trace cross-sections.
//! - Hoye, T. R., Hanson, P. R. & Vyvyan, J. R. (*J. Org. Chem.* **1994**, 59, 4096–4103) —
//!   first-order scalar coupling multiplet patterns (`dd`, `ddd`, `td`, `q`, `t`, `s`).
//! - Fulmer, G. R. et al. (*Organometallics* **2010**, 29, 2176–2179) — residual deuterated
//!   solvent chemical shifts (`CDCl3` `δH 7.26`, `δC 77.16`; `TMS` `0.00 ppm`).

use std::collections::BTreeMap;
use std::f64::consts::PI;
use std::sync::Arc;

use eframe::egui;
use nmr_core::c13_detected::C13PickedDataset;
use nmr_core::nmr1d::{Experiment1DKind, Nucleus1D, Spectrum1D};
use nmr_core::peaks::Peak;
use nmr_core::picker::{
    estimate_noise_mad, pick_cube_3d_plane_ridge_safe, pick_cube_3d_ridge_safe,
    sample_cube_3d_trilinear, snap_cube_3d_extremum, Cube3dPickerConfig, CubePeak3D,
};
use nmr_core::spectrum::{AxisPpm, Spectrum2D, Spectrum3D};
use nmr_core::water::{suppress_water_3d, WaterSuppressionConfig};

use crate::contour_painter::{default_overlay_palette, OverlayLayerSpec, SyncCrosshair};
use crate::stack1d_panel::{default_trace_color, RoiRegion1d, Stack1dState, Trace1d};

/// Default width in logical pixels of the collapsible left `"Spectra"` workbook sidebar on Desktop.
pub const WORKBOOK_SIDEBAR_DEFAULT_WIDTH: f32 = 196.0;

/// Exact width in logical pixels of the slim vertical left tool rail on 1D and 2D spectrum canvases.
pub const CANVAS_TOOL_RAIL_WIDTH: f32 = 38.0;

/// Exact thickness in logical pixels of the Top `F2` (`height = 64 px`) and Left `F1` (`width = 64 px`)
/// 1D marginal projection strips around the 2D contour canvas.
pub const MARGINAL_2D_STRIP_PX: f32 = 64.0;

/// Multiplicative step per scroll-wheel notch or `+`/`−` click on 2D contour base and marginal gains.
pub const CONTOUR_SCROLL_STEP_RATIO: f32 = 1.18;

// ============================================================================
// 1. Clean Geometric Vector Iconography (`14×14` Crisp Painter Strokes, Zero Emojis)
// ============================================================================

/// Geometric vector icons drawn via `egui::Painter` for studio headers, menus, page cards, and tool rails.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WorkbookIcon {
    SidebarToggle,
    Folder,
    FileDoc,
    SunPaper,
    MoonDark,
    Trace1D,
    Contour2D,
    Cube3D,
    EyeVisible,
    PlusOverlay,
    CloseX,
    ZoomBox,
    PanMove,
    CrosshairSlice,
    PeakPick,
    FullSweep,
    UndoBack,
    Plus,
    Minus,
    Projections2D,
    ParamsCard,
    StructureChem,
    Strips3D,
    RelaxDynamics,
    DosyGradient,
    MetabolomicsCluster,
    SlidersTune,
    CommandKey,
    HelpQuestion,
    LockClosed,
    PlayRun,
    DownloadSave,
    WaterSuppress,
    TableGrid,
    ChevronRight,
    ChevronDown,
}

/// Paint a crisp anti-aliased geometric vector icon centered inside `rect`.
pub fn paint_icon(
    painter: &egui::Painter,
    rect: egui::Rect,
    icon: WorkbookIcon,
    color: egui::Color32,
) {
    let s = rect.width().min(rect.height()).clamp(8.0, 24.0);
    let c = rect.center();
    let r = egui::Rect::from_center_size(c, egui::vec2(s, s));
    let sw = (s * 0.105).clamp(1.15, 1.65);
    let stroke = egui::Stroke::new(sw, color);
    let pt = |ux: f32, uy: f32| -> egui::Pos2 {
        egui::pos2(r.left() + ux * r.width(), r.top() + uy * r.height())
    };

    match icon {
        WorkbookIcon::SidebarToggle => {
            let box_r = r.shrink(s * 0.10);
            painter.rect_stroke(box_r, 2.0, stroke, egui::StrokeKind::Inside);
            let div_x = box_r.left() + box_r.width() * 0.36;
            painter.line_segment(
                [
                    egui::pos2(div_x, box_r.top()),
                    egui::pos2(div_x, box_r.bottom()),
                ],
                stroke,
            );
            painter.line_segment([pt(0.18, 0.36), pt(0.32, 0.36)], stroke);
            painter.line_segment([pt(0.18, 0.52), pt(0.32, 0.52)], stroke);
        }
        WorkbookIcon::Folder => {
            let pts = vec![
                pt(0.10, 0.26),
                pt(0.40, 0.26),
                pt(0.49, 0.36),
                pt(0.90, 0.36),
                pt(0.90, 0.82),
                pt(0.10, 0.82),
                pt(0.10, 0.26),
            ];
            painter.add(egui::Shape::line(pts, stroke));
        }
        WorkbookIcon::FileDoc => {
            let pts = vec![
                pt(0.20, 0.12),
                pt(0.62, 0.12),
                pt(0.82, 0.32),
                pt(0.82, 0.88),
                pt(0.20, 0.88),
                pt(0.20, 0.12),
            ];
            painter.add(egui::Shape::line(pts, stroke));
            painter.add(egui::Shape::line(
                vec![pt(0.60, 0.12), pt(0.60, 0.33), pt(0.82, 0.33)],
                stroke,
            ));
        }
        WorkbookIcon::SunPaper => {
            painter.circle_stroke(c, s * 0.21, stroke);
            for k in 0..8 {
                let ang = (k as f32) * (std::f32::consts::PI * 0.25);
                let dir = egui::vec2(ang.cos(), ang.sin());
                painter.line_segment([c + dir * (s * 0.30), c + dir * (s * 0.43)], stroke);
            }
        }
        WorkbookIcon::MoonDark => {
            let rad = s * 0.34;
            let mut arc = Vec::with_capacity(18);
            for k in 0..=16 {
                let a = 0.55 + (k as f32 / 16.0) * 4.45;
                arc.push(c + egui::vec2(a.cos() * rad, a.sin() * rad));
            }
            painter.add(egui::Shape::line(arc, stroke));
            painter.circle_filled(c + egui::vec2(s * 0.14, -s * 0.12), s * 0.08, color);
        }
        WorkbookIcon::Trace1D => {
            let pts = vec![
                pt(0.08, 0.78),
                pt(0.28, 0.78),
                pt(0.36, 0.46),
                pt(0.42, 0.76),
                pt(0.52, 0.16),
                pt(0.60, 0.76),
                pt(0.68, 0.52),
                pt(0.75, 0.78),
                pt(0.92, 0.78),
            ];
            painter.add(egui::Shape::line(pts, stroke));
        }
        WorkbookIcon::Contour2D => {
            let draw_ellipse = |rx: f32, ry: f32| {
                let mut pts = Vec::with_capacity(17);
                for k in 0..=16 {
                    let a = (k as f32 / 16.0) * std::f32::consts::TAU;
                    let x = a.cos() * rx - a.sin() * ry * 0.35;
                    let y = a.cos() * rx * 0.35 + a.sin() * ry;
                    pts.push(c + egui::vec2(x, y));
                }
                painter.add(egui::Shape::line(pts, stroke));
            };
            draw_ellipse(s * 0.37, s * 0.23);
            draw_ellipse(s * 0.19, s * 0.11);
        }
        WorkbookIcon::Cube3D => {
            let top = pt(0.50, 0.12);
            let tr = pt(0.86, 0.31);
            let br = pt(0.86, 0.70);
            let bot = pt(0.50, 0.89);
            let bl = pt(0.14, 0.70);
            let tl = pt(0.14, 0.31);
            let mid = pt(0.50, 0.50);
            painter.add(egui::Shape::line(
                vec![top, tr, br, bot, bl, tl, top],
                stroke,
            ));
            painter.line_segment([tl, mid], stroke);
            painter.line_segment([tr, mid], stroke);
            painter.line_segment([bot, mid], stroke);
        }
        WorkbookIcon::EyeVisible => {
            let mut upper = Vec::with_capacity(13);
            let mut lower = Vec::with_capacity(13);
            for k in 0..=12 {
                let t = k as f32 / 12.0;
                let x = 0.10 + 0.80 * t;
                let arch = (t * std::f32::consts::PI).sin() * 0.25;
                upper.push(pt(x, 0.50 - arch));
                lower.push(pt(x, 0.50 + arch));
            }
            painter.add(egui::Shape::line(upper, stroke));
            painter.add(egui::Shape::line(lower, stroke));
            painter.circle_filled(c, s * 0.11, color);
        }
        WorkbookIcon::PlusOverlay | WorkbookIcon::Plus => {
            painter.line_segment([pt(0.50, 0.20), pt(0.50, 0.80)], stroke);
            painter.line_segment([pt(0.20, 0.50), pt(0.80, 0.50)], stroke);
        }
        WorkbookIcon::Minus => {
            painter.line_segment([pt(0.20, 0.50), pt(0.80, 0.50)], stroke);
        }
        WorkbookIcon::CloseX => {
            painter.line_segment([pt(0.24, 0.24), pt(0.76, 0.76)], stroke);
            painter.line_segment([pt(0.76, 0.24), pt(0.24, 0.76)], stroke);
        }
        WorkbookIcon::ZoomBox => {
            let cc = pt(0.44, 0.44);
            painter.circle_stroke(cc, s * 0.26, stroke);
            painter.line_segment([pt(0.63, 0.63), pt(0.88, 0.88)], stroke);
            painter.line_segment([pt(0.44, 0.31), pt(0.44, 0.57)], stroke);
            painter.line_segment([pt(0.31, 0.44), pt(0.57, 0.44)], stroke);
        }
        WorkbookIcon::PanMove => {
            painter.line_segment([pt(0.50, 0.14), pt(0.50, 0.86)], stroke);
            painter.line_segment([pt(0.14, 0.50), pt(0.86, 0.50)], stroke);
            painter.add(egui::Shape::line(
                vec![pt(0.38, 0.26), pt(0.50, 0.14), pt(0.62, 0.26)],
                stroke,
            ));
            painter.add(egui::Shape::line(
                vec![pt(0.38, 0.74), pt(0.50, 0.86), pt(0.62, 0.74)],
                stroke,
            ));
            painter.add(egui::Shape::line(
                vec![pt(0.26, 0.38), pt(0.14, 0.50), pt(0.26, 0.62)],
                stroke,
            ));
            painter.add(egui::Shape::line(
                vec![pt(0.74, 0.38), pt(0.86, 0.50), pt(0.74, 0.62)],
                stroke,
            ));
        }
        WorkbookIcon::CrosshairSlice => {
            painter.line_segment([pt(0.50, 0.10), pt(0.50, 0.38)], stroke);
            painter.line_segment([pt(0.50, 0.62), pt(0.50, 0.90)], stroke);
            painter.line_segment([pt(0.10, 0.50), pt(0.38, 0.50)], stroke);
            painter.line_segment([pt(0.62, 0.50), pt(0.90, 0.50)], stroke);
            painter.circle_stroke(c, s * 0.16, stroke);
        }
        WorkbookIcon::PeakPick => {
            painter.add(egui::Shape::line(
                vec![
                    pt(0.14, 0.84),
                    pt(0.38, 0.84),
                    pt(0.50, 0.28),
                    pt(0.62, 0.84),
                    pt(0.86, 0.84),
                ],
                stroke,
            ));
            painter.add(egui::Shape::convex_polygon(
                vec![pt(0.50, 0.20), pt(0.41, 0.08), pt(0.59, 0.08)],
                color,
                egui::Stroke::NONE,
            ));
        }
        WorkbookIcon::FullSweep => {
            painter.add(egui::Shape::line(
                vec![pt(0.14, 0.36), pt(0.14, 0.14), pt(0.36, 0.14)],
                stroke,
            ));
            painter.add(egui::Shape::line(
                vec![pt(0.64, 0.14), pt(0.86, 0.14), pt(0.86, 0.36)],
                stroke,
            ));
            painter.add(egui::Shape::line(
                vec![pt(0.86, 0.64), pt(0.86, 0.86), pt(0.64, 0.86)],
                stroke,
            ));
            painter.add(egui::Shape::line(
                vec![pt(0.36, 0.86), pt(0.14, 0.86), pt(0.14, 0.64)],
                stroke,
            ));
        }
        WorkbookIcon::UndoBack => {
            painter.add(egui::Shape::line(
                vec![pt(0.38, 0.22), pt(0.18, 0.42), pt(0.38, 0.62)],
                stroke,
            ));
            painter.add(egui::Shape::line(
                vec![
                    pt(0.20, 0.42),
                    pt(0.66, 0.42),
                    pt(0.82, 0.54),
                    pt(0.82, 0.74),
                    pt(0.66, 0.84),
                    pt(0.36, 0.84),
                ],
                stroke,
            ));
        }
        WorkbookIcon::Projections2D => {
            let main_r = egui::Rect::from_min_max(pt(0.36, 0.36), pt(0.88, 0.88));
            painter.rect_stroke(main_r, 1.5, stroke, egui::StrokeKind::Inside);
            painter.add(egui::Shape::line(
                vec![
                    pt(0.36, 0.26),
                    pt(0.54, 0.26),
                    pt(0.62, 0.12),
                    pt(0.70, 0.26),
                    pt(0.88, 0.26),
                ],
                stroke,
            ));
            painter.add(egui::Shape::line(
                vec![
                    pt(0.26, 0.36),
                    pt(0.26, 0.54),
                    pt(0.12, 0.62),
                    pt(0.26, 0.70),
                    pt(0.26, 0.88),
                ],
                stroke,
            ));
        }
        WorkbookIcon::ParamsCard => {
            let card_r = r.shrink(s * 0.12);
            painter.rect_stroke(card_r, 2.0, stroke, egui::StrokeKind::Inside);
            painter.line_segment([pt(0.26, 0.35), pt(0.74, 0.35)], stroke);
            painter.line_segment([pt(0.26, 0.52), pt(0.62, 0.52)], stroke);
            painter.line_segment([pt(0.26, 0.69), pt(0.70, 0.69)], stroke);
        }
        WorkbookIcon::StructureChem => {
            let mut hex = Vec::with_capacity(7);
            for k in 0..=6 {
                let a = -std::f32::consts::FRAC_PI_2 + (k as f32 / 6.0) * std::f32::consts::TAU;
                hex.push(c + egui::vec2(a.cos() * s * 0.38, a.sin() * s * 0.38));
            }
            painter.add(egui::Shape::line(hex, stroke));
            painter.circle_stroke(c, s * 0.18, stroke);
        }
        WorkbookIcon::Strips3D => {
            for x in [0.22_f32, 0.50, 0.78] {
                painter.line_segment([pt(x, 0.16), pt(x, 0.84)], stroke);
            }
            painter.circle_filled(pt(0.22, 0.36), s * 0.08, color);
            painter.circle_filled(pt(0.50, 0.36), s * 0.08, color);
            painter.circle_filled(pt(0.50, 0.64), s * 0.08, color);
            painter.circle_filled(pt(0.78, 0.64), s * 0.08, color);
            painter.line_segment([pt(0.22, 0.36), pt(0.50, 0.36)], stroke);
            painter.line_segment([pt(0.50, 0.64), pt(0.78, 0.64)], stroke);
        }
        WorkbookIcon::RelaxDynamics => {
            let mut curve = Vec::with_capacity(13);
            for k in 0..=12 {
                let t = k as f32 / 12.0;
                let y = 0.80 - 0.60 * (-2.6 * t).exp();
                curve.push(pt(0.14 + 0.74 * t, y));
            }
            painter.line_segment([pt(0.14, 0.16), pt(0.14, 0.84)], stroke);
            painter.line_segment([pt(0.14, 0.84), pt(0.88, 0.84)], stroke);
            painter.add(egui::Shape::line(curve, stroke));
        }
        WorkbookIcon::DosyGradient => {
            painter.line_segment([pt(0.12, 0.30), pt(0.88, 0.30)], stroke);
            painter.line_segment([pt(0.22, 0.54), pt(0.78, 0.54)], stroke);
            painter.line_segment([pt(0.34, 0.78), pt(0.66, 0.78)], stroke);
        }
        WorkbookIcon::MetabolomicsCluster => {
            painter.line_segment([pt(0.14, 0.84), pt(0.86, 0.84)], stroke);
            painter.line_segment([pt(0.14, 0.16), pt(0.14, 0.84)], stroke);
            for &(ux, uy) in &[(0.34, 0.34), (0.44, 0.28), (0.68, 0.60), (0.78, 0.52)] {
                painter.circle_filled(pt(ux, uy), s * 0.08, color);
            }
        }
        WorkbookIcon::SlidersTune => {
            painter.line_segment([pt(0.14, 0.34), pt(0.86, 0.34)], stroke);
            painter.circle_filled(pt(0.38, 0.34), s * 0.11, color);
            painter.line_segment([pt(0.14, 0.68), pt(0.86, 0.68)], stroke);
            painter.circle_filled(pt(0.66, 0.68), s * 0.11, color);
        }
        WorkbookIcon::CommandKey => {
            let box_r = r.shrink(s * 0.28);
            painter.rect_stroke(box_r, 1.0, stroke, egui::StrokeKind::Inside);
            for &(ux, uy) in &[(0.23, 0.23), (0.77, 0.23), (0.23, 0.77), (0.77, 0.77)] {
                painter.circle_stroke(pt(ux, uy), s * 0.10, stroke);
            }
        }
        WorkbookIcon::HelpQuestion => {
            painter.circle_stroke(c, s * 0.39, stroke);
            painter.add(egui::Shape::line(
                vec![
                    pt(0.38, 0.35),
                    pt(0.44, 0.26),
                    pt(0.57, 0.26),
                    pt(0.63, 0.35),
                    pt(0.50, 0.49),
                    pt(0.50, 0.58),
                ],
                stroke,
            ));
            painter.circle_filled(pt(0.50, 0.73), s * 0.06, color);
        }
        WorkbookIcon::LockClosed => {
            let body = egui::Rect::from_min_max(pt(0.22, 0.46), pt(0.78, 0.86));
            painter.rect_stroke(body, 2.0, stroke, egui::StrokeKind::Inside);
            painter.add(egui::Shape::line(
                vec![
                    pt(0.34, 0.46),
                    pt(0.34, 0.28),
                    pt(0.50, 0.18),
                    pt(0.66, 0.28),
                    pt(0.66, 0.46),
                ],
                stroke,
            ));
        }
        WorkbookIcon::PlayRun => {
            painter.add(egui::Shape::convex_polygon(
                vec![pt(0.28, 0.20), pt(0.80, 0.50), pt(0.28, 0.80)],
                color,
                egui::Stroke::NONE,
            ));
        }
        WorkbookIcon::DownloadSave => {
            painter.line_segment([pt(0.50, 0.16), pt(0.50, 0.66)], stroke);
            painter.add(egui::Shape::line(
                vec![pt(0.32, 0.48), pt(0.50, 0.68), pt(0.68, 0.48)],
                stroke,
            ));
            painter.line_segment([pt(0.20, 0.84), pt(0.80, 0.84)], stroke);
        }
        WorkbookIcon::WaterSuppress => {
            // Crisp droplet contour + horizontal solvent-suppression notch bar
            let mut drop_pts = Vec::with_capacity(15);
            drop_pts.push(pt(0.50, 0.12));
            drop_pts.push(pt(0.76, 0.52));
            for k in 0..=8 {
                let a = (k as f32 / 8.0) * std::f32::consts::PI;
                drop_pts.push(egui::pos2(
                    c.x + a.cos() * (s * 0.26),
                    pt(0.50, 0.58).y + a.sin() * (s * 0.26),
                ));
            }
            drop_pts.push(pt(0.24, 0.52));
            drop_pts.push(pt(0.50, 0.12));
            painter.add(egui::Shape::line(drop_pts, stroke));
            painter.line_segment([pt(0.14, 0.54), pt(0.86, 0.54)], stroke);
        }
        WorkbookIcon::TableGrid => {
            let box_r = r.shrink(s * 0.12);
            painter.rect_stroke(box_r, 1.8, stroke, egui::StrokeKind::Inside);
            painter.line_segment([pt(0.12, 0.38), pt(0.88, 0.38)], stroke);
            painter.line_segment([pt(0.12, 0.63), pt(0.88, 0.63)], stroke);
            painter.line_segment([pt(0.42, 0.38), pt(0.42, 0.88)], stroke);
        }
        WorkbookIcon::ChevronRight => {
            painter.add(egui::Shape::line(
                vec![pt(0.36, 0.24), pt(0.64, 0.50), pt(0.36, 0.76)],
                stroke,
            ));
        }
        WorkbookIcon::ChevronDown => {
            painter.add(egui::Shape::line(
                vec![pt(0.24, 0.36), pt(0.50, 0.64), pt(0.76, 0.36)],
                stroke,
            ));
        }
    }
}

/// Render a refined slate instrument button with a crisp `14×14` vector icon and optional text label.
pub fn chrome_icon_button(
    ui: &mut egui::Ui,
    icon: WorkbookIcon,
    label: Option<&str>,
    selected: bool,
    tooltip: &str,
    height: f32,
) -> egui::Response {
    let icon_sz = 13.0_f32;
    let pad_x = if label.is_some() { 7.5_f32 } else { 6.0_f32 };
    let text_w = if let Some(lbl) = label {
        let galley = ui.painter().layout_no_wrap(
            lbl.to_string(),
            egui::FontId::proportional(11.5),
            egui::Color32::WHITE,
        );
        galley.size().x + 5.5
    } else {
        0.0
    };
    let desired = egui::vec2((icon_sz + text_w + pad_x * 2.0).max(height), height);
    let (rect, resp) = ui.allocate_exact_size(desired, egui::Sense::click());

    if ui.is_rect_visible(rect) {
        let hovered = resp.hovered();
        let fill = if selected {
            egui::Color32::from_rgb(24, 62, 98)
        } else if hovered {
            egui::Color32::from_rgb(32, 41, 56)
        } else {
            egui::Color32::from_rgb(22, 28, 39)
        };
        let border = if selected {
            egui::Color32::from_rgb(92, 188, 250)
        } else if hovered {
            egui::Color32::from_rgb(76, 92, 118)
        } else {
            egui::Color32::from_rgb(44, 55, 74)
        };
        let fg = if selected {
            egui::Color32::from_rgb(230, 246, 255)
        } else if hovered {
            egui::Color32::from_rgb(241, 245, 249)
        } else {
            egui::Color32::from_rgb(198, 210, 228)
        };

        let painter = ui.painter();
        painter.rect_filled(rect, 4.0, fill);
        painter.rect_stroke(
            rect,
            4.0,
            egui::Stroke::new(1.0, border),
            egui::StrokeKind::Inside,
        );

        if let Some(lbl) = label {
            let ir = egui::Rect::from_center_size(
                egui::pos2(rect.left() + pad_x + icon_sz * 0.5, rect.center().y),
                egui::vec2(icon_sz, icon_sz),
            );
            paint_icon(painter, ir, icon, fg);
            painter.text(
                egui::pos2(ir.right() + 5.0, rect.center().y),
                egui::Align2::LEFT_CENTER,
                lbl,
                egui::FontId::proportional(11.5),
                fg,
            );
        } else {
            let ir = egui::Rect::from_center_size(rect.center(), egui::vec2(icon_sz, icon_sz));
            paint_icon(painter, ir, icon, fg);
        }
    }

    if tooltip.is_empty() {
        resp
    } else {
        resp.on_hover_text(tooltip)
    }
}

/// Render a `30×26` tool button inside a `38 px` vertical left tool rail.
pub fn rail_tool_button(
    ui: &mut egui::Ui,
    icon: WorkbookIcon,
    selected: bool,
    tooltip: &str,
) -> egui::Response {
    chrome_icon_button(ui, icon, None, selected, tooltip, 26.0)
}

// ============================================================================
// 2. Publication Paper White vs Dark Oscilloscope Theme Palettes for 2D Canvases
// ============================================================================

/// Color tokens for the 2D contour plot, Top `F2` / Left `F1` marginals, rulers, and floating cards.
#[derive(Clone, Copy, Debug)]
pub struct CanvasTheme2D {
    pub paper_white: bool,
    pub canvas_bg: egui::Color32,
    pub grid_line: egui::Color32,
    pub marginal_bg: egui::Color32,
    pub marginal_border: egui::Color32,
    pub marginal_trace_ext: egui::Color32,
    pub marginal_trace_sky: egui::Color32,
    pub axis_border: egui::Color32,
    pub axis_text: egui::Color32,
    pub axis_muted: egui::Color32,
    pub diagonal_stroke: egui::Color32,
    pub crosshair_unlocked: egui::Color32,
    pub crosshair_locked: egui::Color32,
    pub zoom_box_fill: egui::Color32,
    pub zoom_box_stroke: egui::Color32,
    pub card_bg: egui::Color32,
    pub card_border: egui::Color32,
    pub card_title: egui::Color32,
    pub card_body: egui::Color32,
    pub peak_hi_conf: egui::Color32,
    pub peak_med_conf: egui::Color32,
    pub peak_unassigned: egui::Color32,
    pub peak_locked: egui::Color32,
}

impl CanvasTheme2D {
    pub const fn for_mode(paper_white: bool) -> Self {
        if paper_white {
            Self {
                paper_white: true,
                canvas_bg: egui::Color32::from_rgb(255, 255, 255),
                grid_line: egui::Color32::from_rgba_premultiplied(15, 23, 42, 16),
                marginal_bg: egui::Color32::from_rgb(250, 252, 255),
                marginal_border: egui::Color32::from_rgb(203, 213, 225),
                marginal_trace_ext: egui::Color32::from_rgb(15, 61, 145),
                marginal_trace_sky: egui::Color32::from_rgb(4, 120, 87),
                axis_border: egui::Color32::from_rgb(71, 85, 105),
                axis_text: egui::Color32::from_rgb(17, 24, 39),
                axis_muted: egui::Color32::from_rgb(71, 85, 105),
                diagonal_stroke: egui::Color32::from_rgba_premultiplied(51, 65, 85, 96),
                crosshair_unlocked: egui::Color32::from_rgba_premultiplied(15, 61, 145, 130),
                crosshair_locked: egui::Color32::from_rgba_premultiplied(180, 83, 9, 210),
                zoom_box_fill: egui::Color32::from_rgba_premultiplied(15, 61, 145, 24),
                zoom_box_stroke: egui::Color32::from_rgb(15, 61, 145),
                card_bg: egui::Color32::from_rgba_premultiplied(248, 250, 252, 242),
                card_border: egui::Color32::from_rgb(203, 213, 225),
                card_title: egui::Color32::from_rgb(15, 23, 42),
                card_body: egui::Color32::from_rgb(51, 65, 85),
                peak_hi_conf: egui::Color32::from_rgb(4, 120, 87),
                peak_med_conf: egui::Color32::from_rgb(180, 83, 9),
                peak_unassigned: egui::Color32::from_rgb(30, 58, 138),
                peak_locked: egui::Color32::from_rgb(153, 27, 27),
            }
        } else {
            Self {
                paper_white: false,
                canvas_bg: egui::Color32::from_rgb(11, 15, 23),
                grid_line: egui::Color32::from_rgba_premultiplied(148, 163, 184, 18),
                marginal_bg: egui::Color32::from_rgb(14, 19, 29),
                marginal_border: egui::Color32::from_rgb(45, 58, 79),
                marginal_trace_ext: egui::Color32::from_rgb(56, 189, 248),
                marginal_trace_sky: egui::Color32::from_rgb(52, 211, 153),
                axis_border: egui::Color32::from_rgb(65, 80, 105),
                axis_text: egui::Color32::from_rgb(220, 232, 248),
                axis_muted: egui::Color32::from_rgb(148, 168, 196),
                diagonal_stroke: egui::Color32::from_rgba_premultiplied(195, 215, 245, 95),
                crosshair_unlocked: egui::Color32::from_rgba_premultiplied(110, 215, 255, 110),
                crosshair_locked: egui::Color32::from_rgba_premultiplied(255, 215, 90, 185),
                zoom_box_fill: egui::Color32::from_rgba_premultiplied(56, 189, 248, 34),
                zoom_box_stroke: egui::Color32::from_rgb(110, 230, 255),
                card_bg: egui::Color32::from_rgba_premultiplied(16, 22, 33, 235),
                card_border: egui::Color32::from_rgb(51, 65, 85),
                card_title: egui::Color32::from_rgb(241, 245, 249),
                card_body: egui::Color32::from_rgb(176, 192, 214),
                peak_hi_conf: egui::Color32::from_rgb(95, 245, 165),
                peak_med_conf: egui::Color32::from_rgb(255, 205, 95),
                peak_unassigned: egui::Color32::from_rgb(145, 195, 245),
                peak_locked: egui::Color32::from_rgb(255, 222, 92),
            }
        }
    }
}

/// Publication-grade high-contrast `(positive_rgba, negative_rgba)` contour palette for Paper White 2D backgrounds.
pub const PAPER_OVERLAY_PALETTES: [([f32; 4], [f32; 4]); 6] = [
    // 0: Royal cobalt (+) / Crimson (-)
    ([0.04, 0.20, 0.56, 1.00], [0.72, 0.10, 0.14, 0.96]),
    // 1: Deep emerald (+) / Burnt amber (-)
    ([0.02, 0.44, 0.30, 0.95], [0.76, 0.40, 0.04, 0.92]),
    // 2: Imperial violet (+) / Terracotta (-)
    ([0.40, 0.12, 0.64, 0.94], [0.74, 0.24, 0.06, 0.90]),
    // 3: Deep teal (+) / Burgundy (-)
    ([0.04, 0.42, 0.52, 0.94], [0.66, 0.08, 0.32, 0.90]),
    // 4: Slate navy (+) / Rust (-)
    ([0.12, 0.28, 0.48, 0.92], [0.70, 0.28, 0.12, 0.88]),
    // 5: Indigo (+) / Coral crimson (-)
    ([0.22, 0.20, 0.62, 0.92], [0.76, 0.18, 0.24, 0.88]),
];

#[inline]
pub fn contour_palette_for_theme(layer_idx: usize, paper_white: bool) -> ([f32; 4], [f32; 4]) {
    if paper_white {
        PAPER_OVERLAY_PALETTES[layer_idx % PAPER_OVERLAY_PALETTES.len()]
    } else {
        default_overlay_palette(layer_idx)
    }
}

// ============================================================================
// 3. Textbook Small-Molecule Showcase Builders (`1D ¹H`, `1D ¹³C`, `2D ¹H–¹H COSY`)
// ============================================================================

fn add_complex_lorentzian(
    real: &mut [f64],
    imag: &mut [f64],
    axis: &AxisPpm,
    center_ppm: f64,
    fwhm_hz: f64,
    integral_area: f64,
) {
    let mhz = axis.obs_mhz.max(60.0);
    let gamma_ppm = (0.5 * fwhm_hz / mhz).max(1e-6);
    let g2 = gamma_ppm * gamma_ppm;
    let inv_pi = 1.0 / PI;
    for i in 0..axis.size.min(real.len()).min(imag.len()) {
        let dp = axis.index_to_ppm(i as f64) - center_ppm;
        let denom = dp * dp + g2;
        real[i] += integral_area * inv_pi * gamma_ppm / denom;
        imag[i] += integral_area * inv_pi * (-dp) / denom;
    }
}

fn add_coupled_multiplet(
    real: &mut [f64],
    imag: &mut [f64],
    axis: &AxisPpm,
    center_ppm: f64,
    couplings_hz: &[f64],
    fwhm_hz: f64,
    protons: f64,
) {
    let mhz = axis.obs_mhz.max(60.0);
    let mut lines: Vec<(f64, f64)> = vec![(center_ppm, protons)];
    for &j_hz in couplings_hz {
        let half_j_ppm = 0.5 * j_hz / mhz;
        let mut next = Vec::with_capacity(lines.len() * 2);
        for (pos, amp) in lines {
            next.push((pos + half_j_ppm, 0.5 * amp));
            next.push((pos - half_j_ppm, 0.5 * amp));
        }
        lines = next;
    }
    for (pos_ppm, amp) in lines {
        add_complex_lorentzian(real, imag, axis, pos_ppm, fwhm_hz, amp);
    }
}

/// Build a textbook `600.13 MHz` `CDCl3` `1D ¹H` small-molecule showcase spectrum
/// (*Ethyl 2-acetoxybenzoate*, `C11H12O4`, `12 H`) backed by full complex `(real, imag)` data
/// and pre-normalized multiplet ROI brackets.
pub fn build_showcase_small_molecule_1h_trace() -> (Trace1d, Vec<RoiRegion1d>) {
    let n_pts = 32_768usize;
    let obs_mhz = 600.13_f64;
    let axis = AxisPpm::from_bounds("1H", n_pts, 9.20, -0.40, obs_mhz);
    let mut real = vec![0.0_f64; n_pts];
    let mut imag = vec![0.0_f64; n_pts];

    // Aromatic ortho-substituted benzoate spin system (H6 dd, H4 ddd, H5 td, H3 dd)
    add_coupled_multiplet(&mut real, &mut imag, &axis, 7.98, &[7.9, 1.7], 1.15, 1.0);
    add_coupled_multiplet(
        &mut real,
        &mut imag,
        &axis,
        7.54,
        &[8.2, 7.5, 1.7],
        1.15,
        1.0,
    );
    add_coupled_multiplet(
        &mut real,
        &mut imag,
        &axis,
        7.10,
        &[7.6, 7.6, 1.1],
        1.15,
        1.0,
    );
    add_coupled_multiplet(&mut real, &mut imag, &axis, 6.94, &[8.2, 1.1], 1.15, 1.0);

    // Ethyl ester (-OCH2CH3: quartet 2H at 4.34 ppm + triplet 3H at 1.37 ppm, J = 7.1 Hz)
    add_coupled_multiplet(
        &mut real,
        &mut imag,
        &axis,
        4.34,
        &[7.1, 7.1, 7.1],
        1.10,
        2.0,
    );
    add_coupled_multiplet(&mut real, &mut imag, &axis, 1.37, &[7.1, 7.1], 1.10, 3.0);

    // Acetyl methyl (-OCOCH3: singlet 3H at 2.35 ppm)
    add_coupled_multiplet(&mut real, &mut imag, &axis, 2.35, &[], 1.05, 3.0);

    // Residual CHCl3 in CDCl3 (7.260 ppm) + internal TMS standard (0.000 ppm)
    add_complex_lorentzian(&mut real, &mut imag, &axis, 7.260, 1.05, 0.14);
    add_complex_lorentzian(&mut real, &mut imag, &axis, 0.000, 1.00, 0.18);

    // Deterministic low-amplitude thermal baseline noise (~2800:1 SNR)
    let mut seed_state = 0x5EED_1D01_u64;
    for r in &mut real {
        seed_state = seed_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = ((seed_state >> 33) as f64 / (u32::MAX as f64)) - 0.5;
        *r += u * 0.045;
    }

    let mut spec = Spectrum1D::new(
        "10 · ¹H Ethyl 2-acetoxybenzoate",
        Experiment1DKind::Proton1D,
        axis,
        real,
        Some(imag),
    );
    spec.solvent = Some("CDCl3".to_string());

    let tr = Trace1d::from_spectrum1d("demo_1h_ethyl_acetoxybenzoate", 0.0, spec)
        .with_color(default_trace_color(0))
        .with_demo(true);

    let rois = vec![
        RoiRegion1d::new("Ar-H6 (1H)", 7.92, 8.04),
        RoiRegion1d::new("Ar-H4 (1H)", 7.48, 7.60),
        RoiRegion1d::new("Ar-H5 (1H)", 7.04, 7.16),
        RoiRegion1d::new("Ar-H3 (1H)", 6.88, 7.00),
        RoiRegion1d::new("OCH2 (2H)", 4.27, 4.41),
        RoiRegion1d::new("COCH3 (3H)", 2.30, 2.40),
        RoiRegion1d::new("CH3 (3H)", 1.31, 1.43),
    ];
    (tr, rois)
}

/// Build the companion `150.90 MHz` `CDCl3` proton-decoupled `1D ¹³C{¹H}` spectrum (`11` carbons + `CDCl3` triplet).
pub fn build_showcase_small_molecule_13c_trace() -> Trace1d {
    let n_pts = 16_384usize;
    let obs_mhz = 150.90_f64;
    let axis = AxisPpm::from_bounds("13C", n_pts, 195.0, -5.0, obs_mhz);
    let mut real = vec![0.0_f64; n_pts];
    let mut imag = vec![0.0_f64; n_pts];

    // Ester & acetyl C=O (quaternary, lower NOE intensity) + 6 aromatic C + OCH2 + 2 CH3
    let signals = [
        (169.7, 0.52), // Acetyl C=O
        (164.5, 0.55), // Ethyl ester C=O
        (150.6, 0.50), // C2ipso-OAc
        (133.7, 1.00), // C4H
        (131.6, 0.98), // C6H
        (125.9, 0.96), // C5H
        (123.8, 0.95), // C3H
        (123.3, 0.52), // C1ipso-COOEt
        (61.1, 1.05),  // OCH2
        (21.1, 1.12),  // COCH3
        (14.3, 1.15),  // CH3
    ];
    for &(ppm, rel_i) in &signals {
        add_complex_lorentzian(&mut real, &mut imag, &axis, ppm, 2.2, rel_i);
    }

    // CDCl3 1:1:1 C-D triplet at 77.16 ppm (1J_CD = 32.0 Hz) + TMS at 0.00 ppm
    let j_cd_ppm = 32.0 / obs_mhz;
    for m in [-1.0_f64, 0.0, 1.0] {
        add_complex_lorentzian(&mut real, &mut imag, &axis, 77.16 + m * j_cd_ppm, 2.1, 0.28);
    }
    add_complex_lorentzian(&mut real, &mut imag, &axis, 0.00, 2.0, 0.22);

    let mut seed_state = 0x5EED_13C2_u64;
    for r in &mut real {
        seed_state = seed_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = ((seed_state >> 33) as f64 / (u32::MAX as f64)) - 0.5;
        *r += u * 0.025;
    }

    let mut spec = Spectrum1D::new(
        "11 · ¹³C{¹H} Ethyl 2-acetoxybenzoate",
        Experiment1DKind::Carbon13Decoupled,
        axis,
        real,
        Some(imag),
    );
    spec.solvent = Some("CDCl3".to_string());

    Trace1d::from_spectrum1d("demo_13c_ethyl_acetoxybenzoate", 1.0, spec)
        .with_color(default_trace_color(1))
        .with_demo(true)
}

/// Build the companion `600.13 MHz` `2D ¹H–¹H COSY` spectrum (`512 × 256`) whose diagonal and
/// scalar-coupled cross-peaks align identically with [`build_showcase_small_molecule_1h_trace`].
pub fn build_showcase_small_molecule_cosy_2d() -> Spectrum2D {
    let nx = 512usize;
    let ny = 256usize;
    let axis_x = AxisPpm::from_bounds("1H", nx, 8.60, 0.40, 600.13);
    let axis_y = AxisPpm::from_bounds("1H", ny, 0.40, 8.60, 600.13);
    let mut data = vec![0.0_f32; nx * ny];

    let mut add_peak_2d = |x_ppm: f64, y_ppm: f64, amp: f32, sx_ppm: f64, sy_ppm: f64| {
        let ix_c = axis_x.ppm_to_index(x_ppm);
        let iy_c = axis_y.ppm_to_index(y_ppm);
        let rx = (4.5 * sx_ppm / axis_x.step_ppm.abs().max(1e-6)).ceil() as isize;
        let ry = (4.5 * sy_ppm / axis_y.step_ppm.abs().max(1e-6)).ceil() as isize;
        let ix0 = ((ix_c.floor() as isize) - rx).max(0) as usize;
        let ix1 = ((ix_c.ceil() as isize) + rx).min((nx - 1) as isize) as usize;
        let iy0 = ((iy_c.floor() as isize) - ry).max(0) as usize;
        let iy1 = ((iy_c.ceil() as isize) + ry).min((ny - 1) as isize) as usize;
        let inv_2sx2 = 0.5 / (sx_ppm * sx_ppm).max(1e-9);
        let inv_2sy2 = 0.5 / (sy_ppm * sy_ppm).max(1e-9);
        for iy in iy0..=iy1 {
            let dy = axis_y.index_to_ppm(iy as f64) - y_ppm;
            let ey = (-dy * dy * inv_2sy2).exp() as f32;
            let row = iy * nx;
            for ix in ix0..=ix1 {
                let dx = axis_x.index_to_ppm(ix as f64) - x_ppm;
                let ex = (-dx * dx * inv_2sx2).exp() as f32;
                data[row + ix] += amp * ex * ey;
            }
        }
    };

    // Diagonal peaks
    for &(d_ppm, amp) in &[
        (7.98, 95.0_f32),
        (7.54, 95.0),
        (7.26, 22.0),
        (7.10, 95.0),
        (6.94, 95.0),
        (4.34, 135.0),
        (2.35, 165.0),
        (1.37, 175.0),
    ] {
        add_peak_2d(d_ppm, d_ppm, amp, 0.022, 0.028);
    }

    // Symmetric scalar-coupled COSY cross-peaks:
    // H6(7.98) <-> H5(7.10), H5(7.10) <-> H4(7.54), H4(7.54) <-> H3(6.94), OCH2(4.34) <-> CH3(1.37)
    for &(pa, pb, amp) in &[
        (7.98, 7.10, 62.0_f32),
        (7.98, 7.54, 20.0),
        (7.54, 7.10, 58.0),
        (7.54, 6.94, 64.0),
        (4.34, 1.37, 88.0),
    ] {
        add_peak_2d(pa, pb, amp, 0.020, 0.026);
        add_peak_2d(pb, pa, amp, 0.020, 0.026);
    }

    // Deterministic subtle noise floor so MAD contour estimator locks cleanly
    let mut seed_state = 0xC051_2D03_u64;
    for v in &mut data {
        seed_state = seed_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = ((seed_state >> 33) as f32 / (u32::MAX as f32)) - 0.5;
        *v += u * 1.4;
    }

    Spectrum2D {
        name: "12 · 2D COSY".to_string(),
        axis_x,
        axis_y,
        data,
    }
}

// ============================================================================
// 4. Multi-Page Workbook Model (`Stored2dPage`, `WorkbookState`, Sidebar Renderer)
// ============================================================================

/// Canonical axis nucleus tag (`"1H"`, `"13C"`, `"15N"`, `"19F"`, `"31P"`, ...) inferred from label + bounds.
pub fn canonical_axis_nucleus(raw_label: &str, min_ppm: f64, max_ppm: f64) -> &'static str {
    let up = raw_label
        .trim()
        .trim_matches(|c: char| c == '<' || c == '>' || c == '"' || c == '\'')
        .trim()
        .to_ascii_uppercase();
    match up.as_str() {
        "1H" | "H1" | "H" | "HN" | "HA" | "HB" | "PROTON" => "1H",
        "15N" | "N15" | "N" | "NH" => "15N",
        "13C" | "C13" | "C" | "CA" | "CB" | "CO" | "CACB" => "13C",
        "19F" | "F19" | "F" => "19F",
        "31P" | "P31" | "P" => "31P",
        "2H" | "H2" | "D" => "2H",
        "11B" | "B11" => "11B",
        "29SI" | "SI29" => "29Si",
        _ => {
            if max_ppm <= 19.0 && min_ppm >= -6.0 {
                "1H"
            } else if min_ppm >= 88.0 && max_ppm <= 152.0 && (max_ppm - min_ppm) <= 65.0 {
                "15N"
            } else if max_ppm > 19.0 {
                "13C"
            } else {
                "1H"
            }
        }
    }
}

fn proton_ref_mhz_from_axes(axes: &[&AxisPpm]) -> Option<f64> {
    let mut best = 0.0_f64;
    for ax in axes {
        if !ax.obs_mhz.is_finite() || ax.obs_mhz < 60.0 {
            continue;
        }
        let c_nuc = canonical_axis_nucleus(&ax.label, ax.min_ppm(), ax.max_ppm());
        if c_nuc == "1H" && ax.obs_mhz > best {
            best = ax.obs_mhz;
        }
    }
    (best >= 60.0).then_some(best)
}

/// Resolve physical nucleus tag (`"1H"`, `"13C"`, `"15N"`, `"19F"`, `"31P"`, ...) from an
/// [`AxisPpm`] using Larmor frequency ratio `axis.obs_mhz / ref_proton_mhz` when a `${}^{1}\text{H}$
/// reference axis is present (preventing Bruker `AXNUC=<off>` default `"13C"` labels from
/// mislabeling indirect `1H` axes in `3D H(CCO)NH` or `15N-NOESY`), while preserving `13C`-detected
/// `2D CON` / `CACO` axes where `13C` `obs_mhz` exceeds `200 MHz` at high field.
pub fn axis_physical_nucleus(axis: &AxisPpm, ref_proton_mhz: Option<f64>) -> &'static str {
    let min_ppm = axis.min_ppm();
    let max_ppm = axis.max_ppm();
    let up = axis
        .label
        .trim()
        .trim_matches(|c: char| c == '<' || c == '>' || c == '"' || c == '\'')
        .trim()
        .to_ascii_uppercase();
    match up.as_str() {
        "1H" | "H1" | "H" | "HN" | "HA" | "HB" | "PROTON" | "1H_IND" | "1H_ALIPH" => {
            return "1H";
        }
        "15N" | "N15" | "N" | "NH" => return "15N",
        "19F" | "F19" | "F" => return "19F",
        "31P" | "P31" | "P" => return "31P",
        "2H" | "H2" | "D" => return "2H",
        "13C" | "C13" | "C" | "CA" | "CB" | "CO" | "CACB" if max_ppm > 19.0 => return "13C",
        _ => {}
    }
    if let Some(h_mhz) = ref_proton_mhz.filter(|&m| m.is_finite() && m >= 60.0) {
        if axis.obs_mhz.is_finite() && axis.obs_mhz > 5.0 {
            let ratio = axis.obs_mhz / h_mhz;
            if (0.980..=1.020).contains(&ratio) {
                return "1H";
            } else if (0.925..=0.958).contains(&ratio) {
                return "19F";
            } else if (0.390..=0.420).contains(&ratio) {
                return "31P";
            } else if (0.235..=0.270).contains(&ratio) {
                return "13C";
            } else if (0.145..=0.165).contains(&ratio) {
                return "2H";
            } else if (0.090..=0.115).contains(&ratio) {
                return "15N";
            }
        }
    }
    // Guard against unannotated indirect 1H axes whose parser fallback label defaulted to "13C".
    if up == "13C"
        && max_ppm <= 18.0
        && min_ppm >= -6.0
        && (max_ppm - min_ppm) <= 22.0
        && axis.obs_mhz >= 290.0
    {
        return "1H";
    }
    canonical_axis_nucleus(&axis.label, min_ppm, max_ppm)
}

/// Infer resolved physical `(F2=X, F1=Y)` nucleus tags for a [`Spectrum2D`].
pub fn infer_2d_spectrum_nuclei(spec: &Spectrum2D) -> (&'static str, &'static str) {
    let ref_opt = proton_ref_mhz_from_axes(&[&spec.axis_x, &spec.axis_y]);
    (
        axis_physical_nucleus(&spec.axis_x, ref_opt),
        axis_physical_nucleus(&spec.axis_y, ref_opt),
    )
}

/// Infer resolved physical `(F3=X, F2=Y, F1=Z)` nucleus tags for a [`Spectrum3D`] cube.
pub fn infer_3d_cube_nuclei(s3: &Spectrum3D) -> (&'static str, &'static str, &'static str) {
    let ref_opt = proton_ref_mhz_from_axes(&[&s3.axis_x, &s3.axis_y, &s3.axis_z]);
    (
        axis_physical_nucleus(&s3.axis_x, ref_opt),
        axis_physical_nucleus(&s3.axis_y, ref_opt),
        axis_physical_nucleus(&s3.axis_z, ref_opt),
    )
}

/// Format a canonical nucleus tag with Unicode superscripts (`"1H" -> "¹H"`).
pub const fn pretty_nucleus(nuc: &str) -> &'static str {
    if eq_str(nuc, "1H") {
        "¹H"
    } else if eq_str(nuc, "13C") {
        "¹³C"
    } else if eq_str(nuc, "15N") {
        "¹⁵N"
    } else if eq_str(nuc, "19F") {
        "¹⁹F"
    } else if eq_str(nuc, "31P") {
        "³¹P"
    } else if eq_str(nuc, "2H") {
        "²H"
    } else if eq_str(nuc, "11B") {
        "¹¹B"
    } else if eq_str(nuc, "29Si") {
        "²⁹Si"
    } else {
        "X"
    }
}

const fn eq_str(a: &str, b: &str) -> bool {
    let ab = a.as_bytes();
    let bb = b.as_bytes();
    if ab.len() != bb.len() {
        return false;
    }
    let mut i = 0;
    while i < ab.len() {
        if ab[i] != bb[i] {
            return false;
        }
        i += 1;
    }
    true
}

/// Infer canonical nucleus tag (`"1H"`, `"13C"`, ...) for a [`Trace1d`].
pub fn trace_1d_nucleus(trace: &Trace1d) -> &'static str {
    if let Some(ref att) = trace.raw_spec {
        match att.0.experiment.nucleus() {
            Nucleus1D::H1 => "1H",
            Nucleus1D::C13 => "13C",
            Nucleus1D::N15 => "15N",
            Nucleus1D::F19 => "19F",
            Nucleus1D::P31 => "31P",
            Nucleus1D::Other(_) => {
                let (lo, hi) = trace.ppm_bounds().unwrap_or((0.0, 10.0));
                canonical_axis_nucleus(&att.0.axis.label, lo, hi)
            }
        }
    } else {
        let (lo, hi) = trace.ppm_bounds().unwrap_or((0.0, 10.0));
        canonical_axis_nucleus(&trace.label, lo, hi)
    }
}

/// Format concise `(title, subtitle)` strings for a 1D [`Trace1d`] page card.
pub fn describe_1d_trace_card(trace: &Trace1d, idx: usize) -> (String, String) {
    let nuc = pretty_nucleus(trace_1d_nucleus(trace));
    let title = if trace.label.trim().is_empty() {
        format!("{} · {nuc}", idx + 1)
    } else {
        trace.label.clone()
    };
    let subtitle = if let Some(ref att) = trace.raw_spec {
        let mhz = att.0.axis.obs_mhz.round() as u32;
        let solv = att.0.solvent.as_deref().unwrap_or("neat");
        let pts_k = if trace.len() >= 1024 {
            format!("{}k pts", trace.len() / 1024)
        } else {
            format!("{} pts", trace.len())
        };
        format!("{nuc} · {mhz} MHz · {solv} · {pts_k}")
    } else {
        let (lo, hi) = trace.ppm_bounds().unwrap_or((0.0, 10.0));
        format!("{nuc} · {hi:.1}..{lo:.1} ppm · {} pts", trace.len())
    };
    (title, subtitle)
}

/// Estimate `(contour_base, noise_ref)` robustly even when >50% of 2D spectral bins are zero-padded/masked
/// or when spectrometer `NC_proc` scaling produces `10^8`–`10^9` integer-scaled intensities.
pub fn estimate_contour_base_and_noise(data: &[f32]) -> (f32, f32) {
    let max_abs = data
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(0.0_f32, |m, v| m.max(v.abs()));
    let mut mad = estimate_noise_mad(data);
    // Rescue >50% exact-zero-padded / NUS-masked arrays while preserving extreme-dynamic-range
    // homonuclear 2D spectra (e.g. NOESY/TOCSY where diagonal/water max_abs can be ~6e5 * sigma0).
    if (!mad.is_finite() || mad <= max_abs * 1.0e-9) && max_abs > 1e-12 {
        let stride = ((data.len() / 65_536).max(1)) | 1;
        let nz: Vec<f32> = data
            .iter()
            .step_by(stride)
            .copied()
            .filter(|&v| v.is_finite() && v.abs() > max_abs * 1e-11)
            .collect();
        if nz.len() >= 16 {
            let nz_mad = estimate_noise_mad(&nz);
            if nz_mad.is_finite() && nz_mad > mad {
                mad = nz_mad;
            }
        }
    }
    if mad.is_finite() && mad > 1e-12 {
        let nref = mad.max(max_abs * 1.0e-7).max(1e-6);
        let base = if max_abs > nref * 5.5 {
            (nref * 5.5).min((max_abs * 0.45).max(1e-5)).max(1e-6)
        } else {
            (nref * 4.8).max(1e-6)
        };
        (base, nref)
    } else if max_abs.is_finite() && max_abs > 1e-12 {
        let base = (max_abs * 0.02).max(1e-6);
        (base, (base / 5.5).max(1e-6))
    } else {
        (1.0, 1.0)
    }
}

/// Extract up to `max_dots` normalized `(u, v, is_positive)` peak positions in `[0, 1]²` for fast
/// sidebar thumbnail rendering (`u = 0` at high `F2` ppm left, `u = 1` at low `F2` ppm right;
/// `v = 0` at low `F1` ppm top, `v = 1` at high `F1` ppm bottom).
pub fn build_mini_2d_preview_dots(spec: &Spectrum2D, max_dots: usize) -> Vec<([f32; 2], bool)> {
    let nx = spec.nx();
    let ny = spec.ny();
    if nx < 3 || ny < 3 || spec.data.len() < nx * ny {
        return Vec::new();
    }
    let (base, _) = estimate_contour_base_and_noise(&spec.data);
    let thresh = base * 1.35;
    let x_hi = spec.axis_x.max_ppm();
    let x_lo = spec.axis_x.min_ppm();
    let y_lo = spec.axis_y.min_ppm();
    let y_hi = spec.axis_y.max_ppm();
    let x_span = (x_hi - x_lo).max(1e-6);
    let y_span = (y_hi - y_lo).max(1e-6);

    let step_x = (nx / 64).max(1);
    let step_y = (ny / 64).max(1);
    let mut cands: Vec<(f32, [f32; 2], bool)> = Vec::new();
    let mut iy = step_y;
    while iy + step_y < ny {
        let mut ix = step_x;
        while ix + step_x < nx {
            let v = spec.get(ix, iy);
            let av = v.abs();
            if av >= thresh
                && av >= spec.get(ix - step_x, iy).abs()
                && av >= spec.get(ix + step_x, iy).abs()
                && av >= spec.get(ix, iy - step_y).abs()
                && av >= spec.get(ix, iy + step_y).abs()
            {
                let px = spec.axis_x.index_to_ppm(ix as f64);
                let py = spec.axis_y.index_to_ppm(iy as f64);
                let u = ((x_hi - px) / x_span).clamp(0.04, 0.96) as f32;
                let vn = ((py - y_lo) / y_span).clamp(0.04, 0.96) as f32;
                cands.push((av, [u, vn], v >= 0.0));
            }
            ix += step_x;
        }
        iy += step_y;
    }
    cands.sort_by(|a, b| b.0.total_cmp(&a.0));
    cands
        .into_iter()
        .take(max_dots.max(8))
        .map(|(_, uv, pos)| (uv, pos))
        .collect()
}

/// Backing data kind for a page inside the Multi-Page Workbook (`pages_2d`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum WorkbookPageKind {
    /// Native 2D spectrum page (`HSQC`, `COSY`, `NOESY`, `HMBC`, `CON`, ...).
    #[default]
    Spectrum2d,
    /// Interactive 3D NMR cube page backed by `cubes_3d[cube_idx]`.
    Cube3d { cube_idx: usize },
}

/// Orthogonal 2D plane orientation through a 3D `(F3=X, F2=Y, F1=Z)` NMR cube.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum CubePlaneOrientation {
    /// `F3×F2` planes stepped along `F1` (`axis_x × axis_y` stepped along `axis_z`; chord `xy` / `p1`).
    #[default]
    F3xF2StepF1,
    /// `F3×F1` strip planes stepped along `F2` (`axis_x × axis_z` stepped along `axis_y`; chord `xz` / `p2`).
    F3xF1StepF2,
    /// `F2×F1` indirect planes stepped along `F3` (`axis_y × axis_z` stepped along `axis_x`; chord `yz` / `p3`).
    F2xF1StepF3,
}

impl CubePlaneOrientation {
    pub const ALL: [Self; 3] = [Self::F3xF2StepF1, Self::F3xF1StepF2, Self::F2xF1StepF3];

    #[inline]
    pub const fn short_label(self) -> &'static str {
        match self {
            Self::F3xF2StepF1 => "F3×F2",
            Self::F3xF1StepF2 => "F3×F1",
            Self::F2xF1StepF3 => "F2×F1",
        }
    }

    #[inline]
    pub const fn chord_badge(self) -> &'static str {
        match self {
            Self::F3xF2StepF1 => "xy",
            Self::F3xF1StepF2 => "xz",
            Self::F2xF1StepF3 => "yz",
        }
    }

    #[inline]
    pub const fn ortho_axis_tag(self) -> &'static str {
        match self {
            Self::F3xF2StepF1 => "F1",
            Self::F3xF1StepF2 => "F2",
            Self::F2xF1StepF3 => "F3",
        }
    }

    /// Map to `pick_cube_3d_plane_ridge_safe` `ortho_axis` (`0` = fixed `F3/X`, `1` = fixed `F2/Y`, `2` = fixed `F1/Z`).
    #[inline]
    pub const fn ortho_picker_axis(self) -> usize {
        match self {
            Self::F3xF2StepF1 => 2,
            Self::F3xF1StepF2 => 1,
            Self::F2xF1StepF3 => 0,
        }
    }
}

/// Visibility tier of a 3D `(F3, F2, F1)` peak relative to the active 2D slice or skyline slab.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PlanePeakVisibility {
    /// Crisp crosshair + label (`|Δplane| <= 0.5` slices, or inside active skyline slab).
    OnPlane,
    /// Faded ghost marker (`0.5 < |Δplane| <= 1.5` slices on adjacent planes).
    AdjacentGhost,
    /// Hidden (`|Δplane| > 1.5` slices, or outside active skyline slab bounds).
    OffPlane,
}

/// 2D projection of a [`CubePeak3D`] onto the active [`Cube3dSliceState`] plane.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Projected3dPeak {
    pub peak_idx: usize,
    pub plane_x_ppm: f64,
    pub plane_y_ppm: f64,
    pub ortho_ppm: f64,
    pub delta_planes: f64,
    pub visibility: PlanePeakVisibility,
}

/// Per-page interactive 3D slicing, skyline MIP slab, and 3D peak-curation state for [`WorkbookPageKind::Cube3d`].
#[derive(Clone, Debug, PartialEq)]
pub struct Cube3dSliceState {
    pub orientation: CubePlaneOrientation,
    pub skyline_enabled: bool,
    /// Plane index along `F1` (`axis_z`, `0..nz`).
    pub plane_f1_idx: usize,
    /// Plane index along `F2` (`axis_y`, `0..ny`).
    pub plane_f2_idx: usize,
    /// Plane index along `F3` (`axis_x`, `0..nx`).
    pub plane_f3_idx: usize,
    /// Optional orthogonal chemical-shift slab bounds `(ppm_a, ppm_b)` when `skyline_enabled == true`.
    pub slab_ppm_bounds: Option<(f64, f64)>,
    pub selected_peak_3d_idx: Option<usize>,
    pub dragging_peak_3d_idx: Option<usize>,
    pub exclude_diagonal: bool,
    pub exclude_water_band: bool,
    pub snr_threshold: f32,
}

impl Default for Cube3dSliceState {
    fn default() -> Self {
        Self {
            orientation: CubePlaneOrientation::F3xF2StepF1,
            skyline_enabled: true,
            plane_f1_idx: 0,
            plane_f2_idx: 0,
            plane_f3_idx: 0,
            slab_ppm_bounds: None,
            selected_peak_3d_idx: None,
            dragging_peak_3d_idx: None,
            exclude_diagonal: true,
            exclude_water_band: true,
            snr_threshold: 6.0,
        }
    }
}

impl Cube3dSliceState {
    pub fn from_cube(s3: &Spectrum3D) -> Self {
        Self {
            plane_f1_idx: s3.nz() / 2,
            plane_f2_idx: s3.ny() / 2,
            plane_f3_idx: s3.nx() / 2,
            ..Self::default()
        }
    }

    #[inline]
    pub fn ortho_axis<'a>(&self, s3: &'a Spectrum3D) -> &'a AxisPpm {
        match self.orientation {
            CubePlaneOrientation::F3xF2StepF1 => &s3.axis_z,
            CubePlaneOrientation::F3xF1StepF2 => &s3.axis_y,
            CubePlaneOrientation::F2xF1StepF3 => &s3.axis_x,
        }
    }

    #[inline]
    pub fn plane_axes<'a>(&self, s3: &'a Spectrum3D) -> (&'a AxisPpm, &'a AxisPpm) {
        match self.orientation {
            CubePlaneOrientation::F3xF2StepF1 => (&s3.axis_x, &s3.axis_y),
            CubePlaneOrientation::F3xF1StepF2 => (&s3.axis_x, &s3.axis_z),
            CubePlaneOrientation::F2xF1StepF3 => (&s3.axis_y, &s3.axis_z),
        }
    }

    #[inline]
    pub fn ortho_len(&self, s3: &Spectrum3D) -> usize {
        self.ortho_axis(s3).size.max(1)
    }

    #[inline]
    pub fn active_plane_idx(&self, s3: &Spectrum3D) -> usize {
        match self.orientation {
            CubePlaneOrientation::F3xF2StepF1 => self.plane_f1_idx.min(s3.nz().saturating_sub(1)),
            CubePlaneOrientation::F3xF1StepF2 => self.plane_f2_idx.min(s3.ny().saturating_sub(1)),
            CubePlaneOrientation::F2xF1StepF3 => self.plane_f3_idx.min(s3.nx().saturating_sub(1)),
        }
    }

    #[inline]
    pub fn set_active_plane_idx(&mut self, s3: &Spectrum3D, idx: usize) {
        match self.orientation {
            CubePlaneOrientation::F3xF2StepF1 => {
                self.plane_f1_idx = idx.min(s3.nz().saturating_sub(1));
            }
            CubePlaneOrientation::F3xF1StepF2 => {
                self.plane_f2_idx = idx.min(s3.ny().saturating_sub(1));
            }
            CubePlaneOrientation::F2xF1StepF3 => {
                self.plane_f3_idx = idx.min(s3.nx().saturating_sub(1));
            }
        }
    }

    #[inline]
    pub fn active_ortho_ppm(&self, s3: &Spectrum3D) -> f64 {
        let idx = self.active_plane_idx(s3);
        self.ortho_axis(s3).index_to_ppm(idx as f64)
    }

    /// Step the orthogonal plane by `delta` slices (automatically exiting `skyline_enabled` mode into discrete plane inspection).
    pub fn step_plane(&mut self, s3: &Spectrum3D, delta: isize) -> usize {
        let max_idx = self.ortho_len(s3).saturating_sub(1) as isize;
        let cur = self.active_plane_idx(s3) as isize;
        let next = (cur + delta).clamp(0, max_idx) as usize;
        self.skyline_enabled = false;
        self.set_active_plane_idx(s3, next);
        next
    }

    /// Flip to an explicit orthogonal 2D plane orientation (exiting skyline mode so the plane is inspected directly).
    pub fn set_orientation(&mut self, s3: &Spectrum3D, orientation: CubePlaneOrientation) {
        self.orientation = orientation;
        self.skyline_enabled = false;
        let clamped = self.active_plane_idx(s3);
        self.set_active_plane_idx(s3, clamped);
    }

    /// Toggle 2D Skyline (`Max |I|` signed MIP) across the active orthogonal axis.
    pub fn toggle_skyline(&mut self) -> bool {
        self.skyline_enabled = !self.skyline_enabled;
        self.skyline_enabled
    }

    /// Extract the currently active 2D plane or 2D skyline projection from `s3` with resolved physical nucleus labels.
    pub fn extract_active_2d(&self, s3: &Spectrum3D) -> Spectrum2D {
        if self.skyline_enabled {
            extract_3d_skyline(s3, self.orientation, self.slab_ppm_bounds)
        } else {
            extract_3d_plane(s3, self.orientation, self.active_plane_idx(s3))
        }
    }

    /// Map `(plane_x_ppm, plane_y_ppm)` on the active 2D view back to full 3D `(f3_ppm, f2_ppm, f1_ppm)`.
    pub fn coords_from_plane_xy(
        &self,
        s3: &Spectrum3D,
        plane_x_ppm: f64,
        plane_y_ppm: f64,
        override_ortho_ppm: Option<f64>,
    ) -> (f64, f64, f64) {
        let ortho_ppm = override_ortho_ppm.unwrap_or_else(|| self.active_ortho_ppm(s3));
        match self.orientation {
            CubePlaneOrientation::F3xF2StepF1 => (plane_x_ppm, plane_y_ppm, ortho_ppm),
            CubePlaneOrientation::F3xF1StepF2 => (plane_x_ppm, ortho_ppm, plane_y_ppm),
            CubePlaneOrientation::F2xF1StepF3 => (ortho_ppm, plane_x_ppm, plane_y_ppm),
        }
    }

    /// Project a 3D peak onto the active 2D orientation and classify `OnPlane` (`|Δplane| <= 0.5`),
    /// `AdjacentGhost` (`0.5 < |Δplane| <= 1.5`), or `OffPlane` (`|Δplane| > 1.5`).
    pub fn project_peak_3d(
        &self,
        s3: &Spectrum3D,
        peak_idx: usize,
        pk: &CubePeak3D,
    ) -> Projected3dPeak {
        let (plane_x_ppm, plane_y_ppm, ortho_ppm) = match self.orientation {
            CubePlaneOrientation::F3xF2StepF1 => (pk.f3_ppm, pk.f2_ppm, pk.f1_ppm),
            CubePlaneOrientation::F3xF1StepF2 => (pk.f3_ppm, pk.f1_ppm, pk.f2_ppm),
            CubePlaneOrientation::F2xF1StepF3 => (pk.f2_ppm, pk.f1_ppm, pk.f3_ppm),
        };
        if self.skyline_enabled {
            let in_slab = match self.slab_ppm_bounds {
                Some((a, b)) => (a.min(b)..=a.max(b)).contains(&ortho_ppm),
                None => true,
            };
            return Projected3dPeak {
                peak_idx,
                plane_x_ppm,
                plane_y_ppm,
                ortho_ppm,
                delta_planes: 0.0,
                visibility: if in_slab {
                    PlanePeakVisibility::OnPlane
                } else {
                    PlanePeakVisibility::OffPlane
                },
            };
        }
        let exact_idx = self.ortho_axis(s3).ppm_to_index(ortho_ppm);
        let cur_idx = self.active_plane_idx(s3) as f64;
        let delta_planes = exact_idx - cur_idx;
        let ad = delta_planes.abs();
        let visibility = if ad <= 0.5 + 1e-9 {
            PlanePeakVisibility::OnPlane
        } else if ad <= 1.5 + 1e-9 {
            PlanePeakVisibility::AdjacentGhost
        } else {
            PlanePeakVisibility::OffPlane
        };
        Projected3dPeak {
            peak_idx,
            plane_x_ppm,
            plane_y_ppm,
            ortho_ppm,
            delta_planes,
            visibility,
        }
    }

    /// Return all projected 3D peaks visible (`OnPlane` or `AdjacentGhost`) on the active view.
    pub fn visible_projected_peaks(
        &self,
        s3: &Spectrum3D,
        peaks: &[CubePeak3D],
    ) -> Vec<Projected3dPeak> {
        peaks
            .iter()
            .enumerate()
            .map(|(i, pk)| self.project_peak_3d(s3, i, pk))
            .filter(|pr| pr.visibility != PlanePeakVisibility::OffPlane)
            .collect()
    }

    /// Synchronize this 3D cube slice to an amide `(1H, 15N)` coordinate (for the `gs` "goto strip" chord),
    /// flipping into the strip plane stepped along `15N` (`F3×F1` when `F2` is `15N`, or `F3×F2` when `F1` is `15N`)
    /// and returning `(plane_x_ppm, plane_y_ppm)` for cursor placement.
    pub fn sync_to_hn(&mut self, s3: &Spectrum3D, h_ppm: f64, n_ppm: f64) -> (f64, f64) {
        let (nx, ny, nz) = infer_3d_cube_nuclei(s3);
        let clamp_idx = |ax: &AxisPpm, ppm: f64| -> usize {
            if ax.size == 0 {
                0
            } else {
                ax.ppm_to_index(ppm)
                    .round()
                    .clamp(0.0, (ax.size - 1) as f64) as usize
            }
        };
        self.skyline_enabled = false;
        if ny == "15N" {
            self.orientation = CubePlaneOrientation::F3xF1StepF2;
            self.plane_f2_idx = clamp_idx(&s3.axis_y, n_ppm);
            if nx == "1H" {
                self.plane_f3_idx = clamp_idx(&s3.axis_x, h_ppm);
            }
            let mid_y = 0.5 * (s3.axis_z.min_ppm() + s3.axis_z.max_ppm());
            (h_ppm, mid_y)
        } else if nz == "15N" {
            self.orientation = CubePlaneOrientation::F3xF2StepF1;
            self.plane_f1_idx = clamp_idx(&s3.axis_z, n_ppm);
            if nx == "1H" {
                self.plane_f3_idx = clamp_idx(&s3.axis_x, h_ppm);
            }
            let mid_y = 0.5 * (s3.axis_y.min_ppm() + s3.axis_y.max_ppm());
            (h_ppm, mid_y)
        } else if nx == "15N" {
            self.orientation = CubePlaneOrientation::F2xF1StepF3;
            self.plane_f3_idx = clamp_idx(&s3.axis_x, n_ppm);
            let x_ppm = if ny == "1H" {
                h_ppm
            } else {
                0.5 * (s3.axis_y.min_ppm() + s3.axis_y.max_ppm())
            };
            let mid_y = 0.5 * (s3.axis_z.min_ppm() + s3.axis_z.max_ppm());
            (x_ppm, mid_y)
        } else {
            self.plane_f3_idx = clamp_idx(&s3.axis_x, h_ppm);
            let mid_y = match self.orientation {
                CubePlaneOrientation::F3xF2StepF1 => {
                    0.5 * (s3.axis_y.min_ppm() + s3.axis_y.max_ppm())
                }
                _ => 0.5 * (s3.axis_z.min_ppm() + s3.axis_z.max_ppm()),
            };
            (h_ppm, mid_y)
        }
    }
}

/// Persistent per-page state for a loaded 2D spectrum or 3D cube inside the Multi-Page Workbook.
#[derive(Clone, Debug)]
pub struct Stored2dPage {
    pub id: u64,
    pub kind: WorkbookPageKind,
    pub title: String,
    pub subtitle: String,
    pub nuc_x: &'static str,
    pub nuc_y: &'static str,
    pub spec: Spectrum2D,
    pub data_arc: Arc<Vec<f32>>,
    pub spec_gen: u64,
    pub base_level: f32,
    pub noise_ref: f32,
    pub level_factor: f32,
    pub num_levels: u32,
    pub show_negative: bool,
    pub ppm_nudge: [f32; 2],
    /// `[x_left, x_right, y_top, y_bottom]`
    pub view_ppm: [f32; 4],
    pub overlay_enabled: bool,
    pub has_3d_strips: bool,
    pub is_demo: bool,
    pub peaks: Vec<Peak>,
    pub peak_2d_labels: BTreeMap<(i64, i64), String>,
    pub c13_picked_dataset: Option<C13PickedDataset>,
    pub water_undo_2d: Option<(Spectrum2D, Option<Spectrum3D>, Option<Spectrum3D>)>,
    pub zoom_history: Vec<[f32; 4]>,
    pub pick_diagonal: bool,
    pub mini_dots: Vec<([f32; 2], bool)>,
    pub source_cube_idx: Option<usize>,
    pub cube_slice: Cube3dSliceState,
    pub peaks_3d: Vec<CubePeak3D>,
    pub water_undo_3d: Option<Spectrum3D>,
}

fn relabeled_axis(axis: &AxisPpm, nuc: &'static str) -> AxisPpm {
    let mut out = axis.clone();
    out.label = nuc.to_string();
    out
}

/// Extract a signed max-abs 2D `(F3, F2)` skyline projection from a `Spectrum3D` cube.
pub fn extract_3d_skyline_f3_f2(s3: &Spectrum3D) -> Spectrum2D {
    extract_3d_skyline(s3, CubePlaneOrientation::F3xF2StepF1, None)
}

/// Extract a signed max-abs (`Pos+Neg` MIP) 2D skyline projection across any [`CubePlaneOrientation`],
/// optionally restricted to `slab_ppm_bounds` along the orthogonal axis.
pub fn extract_3d_skyline(
    s3: &Spectrum3D,
    orientation: CubePlaneOrientation,
    slab_ppm_bounds: Option<(f64, f64)>,
) -> Spectrum2D {
    let (nx, ny, nz) = (s3.nx(), s3.ny(), s3.nz());
    let total = nx.saturating_mul(ny).saturating_mul(nz);
    let (nuc_x, nuc_y, nuc_z) = infer_3d_cube_nuclei(s3);

    let in_slab = |ax: &AxisPpm, idx: usize| -> bool {
        match slab_ppm_bounds {
            Some((a, b)) => {
                let p = ax.index_to_ppm(idx as f64);
                (a.min(b)..=a.max(b)).contains(&p)
            }
            None => true,
        }
    };

    match orientation {
        CubePlaneOrientation::F3xF2StepF1 => {
            let mut data = vec![0.0_f32; ny * nx];
            if s3.data.len() >= total {
                for gz in 0..nz {
                    if !in_slab(&s3.axis_z, gz) {
                        continue;
                    }
                    let plane = &s3.data[gz * ny * nx..(gz + 1) * ny * nx];
                    for (dst, &val) in data.iter_mut().zip(plane.iter()) {
                        if val.abs() > dst.abs() {
                            *dst = val;
                        }
                    }
                }
            }
            Spectrum2D {
                name: s3.name.clone(),
                axis_x: relabeled_axis(&s3.axis_x, nuc_x),
                axis_y: relabeled_axis(&s3.axis_y, nuc_y),
                data,
            }
        }
        CubePlaneOrientation::F3xF1StepF2 => {
            let mut data = vec![0.0_f32; nz * nx];
            if s3.data.len() >= total {
                for gz in 0..nz {
                    let dst_row = &mut data[gz * nx..(gz + 1) * nx];
                    for gy in 0..ny {
                        if !in_slab(&s3.axis_y, gy) {
                            continue;
                        }
                        let src_row = &s3.data[(gz * ny + gy) * nx..(gz * ny + gy + 1) * nx];
                        for (dst, &val) in dst_row.iter_mut().zip(src_row.iter()) {
                            if val.abs() > dst.abs() {
                                *dst = val;
                            }
                        }
                    }
                }
            }
            Spectrum2D {
                name: s3.name.clone(),
                axis_x: relabeled_axis(&s3.axis_x, nuc_x),
                axis_y: relabeled_axis(&s3.axis_z, nuc_z),
                data,
            }
        }
        CubePlaneOrientation::F2xF1StepF3 => {
            let mut data = vec![0.0_f32; nz * ny];
            if s3.data.len() >= total {
                for gz in 0..nz {
                    for gy in 0..ny {
                        let dst = &mut data[gz * ny + gy];
                        let src_row = &s3.data[(gz * ny + gy) * nx..(gz * ny + gy + 1) * nx];
                        for (gx, &val) in src_row.iter().enumerate() {
                            if !in_slab(&s3.axis_x, gx) {
                                continue;
                            }
                            if val.abs() > dst.abs() {
                                *dst = val;
                            }
                        }
                    }
                }
            }
            Spectrum2D {
                name: s3.name.clone(),
                axis_x: relabeled_axis(&s3.axis_y, nuc_y),
                axis_y: relabeled_axis(&s3.axis_z, nuc_z),
                data,
            }
        }
    }
}

/// Extract a 2D `(F3, F2)` plane at fixed `F1` (`iz`) from a `Spectrum3D` cube.
pub fn extract_3d_plane_f1(s3: &Spectrum3D, iz: usize) -> Spectrum2D {
    let (nx, ny, nz) = (s3.nx(), s3.ny(), s3.nz());
    let (nuc_x, nuc_y, _) = infer_3d_cube_nuclei(s3);
    let z = iz.min(nz.saturating_sub(1));
    let off = z * ny * nx;
    let data = if off + ny * nx <= s3.data.len() {
        s3.data[off..off + ny * nx].to_vec()
    } else {
        vec![0.0_f32; ny * nx]
    };
    Spectrum2D {
        name: s3.name.clone(),
        axis_x: relabeled_axis(&s3.axis_x, nuc_x),
        axis_y: relabeled_axis(&s3.axis_y, nuc_y),
        data,
    }
}

/// Extract a 2D `(F3, F1)` strip plane at fixed `F2` (`iy`) from a `Spectrum3D` cube.
pub fn extract_3d_plane_f2(s3: &Spectrum3D, iy: usize) -> Spectrum2D {
    let (nx, ny, nz) = (s3.nx(), s3.ny(), s3.nz());
    let (nuc_x, _, nuc_z) = infer_3d_cube_nuclei(s3);
    let y = iy.min(ny.saturating_sub(1));
    let mut data = vec![0.0_f32; nz * nx];
    for gz in 0..nz {
        let src = (gz * ny + y) * nx;
        let dst = gz * nx;
        if src + nx <= s3.data.len() {
            data[dst..dst + nx].copy_from_slice(&s3.data[src..src + nx]);
        }
    }
    Spectrum2D {
        name: s3.name.clone(),
        axis_x: relabeled_axis(&s3.axis_x, nuc_x),
        axis_y: relabeled_axis(&s3.axis_z, nuc_z),
        data,
    }
}

/// Extract a 2D `(F2, F1)` indirect plane at fixed `F3` (`ix`) from a `Spectrum3D` cube.
pub fn extract_3d_plane_f3(s3: &Spectrum3D, ix: usize) -> Spectrum2D {
    let (nx, ny, nz) = (s3.nx(), s3.ny(), s3.nz());
    let (_, nuc_y, nuc_z) = infer_3d_cube_nuclei(s3);
    let x = ix.min(nx.saturating_sub(1));
    let mut data = vec![0.0_f32; nz * ny];
    for gz in 0..nz {
        for gy in 0..ny {
            let src = (gz * ny + gy) * nx + x;
            if src < s3.data.len() {
                data[gz * ny + gy] = s3.data[src];
            }
        }
    }
    Spectrum2D {
        name: s3.name.clone(),
        axis_x: relabeled_axis(&s3.axis_y, nuc_y),
        axis_y: relabeled_axis(&s3.axis_z, nuc_z),
        data,
    }
}

/// Extract an orthogonal 2D plane along any [`CubePlaneOrientation`] at `plane_idx`.
pub fn extract_3d_plane(
    s3: &Spectrum3D,
    orientation: CubePlaneOrientation,
    plane_idx: usize,
) -> Spectrum2D {
    match orientation {
        CubePlaneOrientation::F3xF2StepF1 => extract_3d_plane_f1(s3, plane_idx),
        CubePlaneOrientation::F3xF1StepF2 => extract_3d_plane_f2(s3, plane_idx),
        CubePlaneOrientation::F2xF1StepF3 => extract_3d_plane_f3(s3, plane_idx),
    }
}

impl Stored2dPage {
    pub fn from_spectrum_2d(
        id: u64,
        label: impl Into<String>,
        spec: Spectrum2D,
        has_3d_strips: bool,
    ) -> Self {
        let raw_lbl = label.into();
        let (nuc_x, nuc_y) = infer_2d_spectrum_nuclei(&spec);
        let px = pretty_nucleus(nuc_x);
        let py = pretty_nucleus(nuc_y);
        let title = if raw_lbl.trim().is_empty() {
            format!("2D {px}–{py}")
        } else {
            raw_lbl
        };
        let strip_tag = if has_3d_strips { " · +3D" } else { "" };
        let subtitle = format!("{}×{} · F2 {px} / F1 {py}{strip_tag}", spec.nx(), spec.ny());
        let (base_level, noise_ref) = estimate_contour_base_and_noise(&spec.data);
        let view_ppm = [
            spec.axis_x.max_ppm() as f32,
            spec.axis_x.min_ppm() as f32,
            spec.axis_y.min_ppm() as f32,
            spec.axis_y.max_ppm() as f32,
        ];
        let mini_dots = build_mini_2d_preview_dots(&spec, 42);
        let data_arc = Arc::new(spec.data.clone());
        Self {
            id,
            kind: WorkbookPageKind::Spectrum2d,
            title,
            subtitle,
            nuc_x,
            nuc_y,
            spec,
            data_arc,
            spec_gen: 1,
            base_level,
            noise_ref,
            level_factor: 1.42,
            num_levels: 10,
            show_negative: true,
            ppm_nudge: [0.0, 0.0],
            view_ppm,
            overlay_enabled: false,
            has_3d_strips,
            is_demo: false,
            peaks: Vec::new(),
            peak_2d_labels: BTreeMap::new(),
            c13_picked_dataset: None,
            water_undo_2d: None,
            zoom_history: Vec::new(),
            pick_diagonal: false,
            mini_dots,
            source_cube_idx: None,
            cube_slice: Cube3dSliceState::default(),
            peaks_3d: Vec::new(),
            water_undo_3d: None,
        }
    }

    pub fn from_3d_cube(
        id: u64,
        cube_idx: usize,
        label: impl Into<String>,
        spec3d: &Spectrum3D,
    ) -> Self {
        let slice_state = Cube3dSliceState::from_cube(spec3d);
        let skyline = slice_state.extract_active_2d(spec3d);
        let raw_lbl = label.into();
        let title = if raw_lbl.trim().is_empty() {
            format!("3D {}", spec3d.exp)
        } else if raw_lbl
            .to_ascii_uppercase()
            .contains(&spec3d.exp.to_ascii_uppercase())
        {
            raw_lbl
        } else {
            format!("{raw_lbl} · 3D {}", spec3d.exp)
        };
        let (nx, ny, nz) = infer_3d_cube_nuclei(spec3d);
        let px = pretty_nucleus(nx);
        let py = pretty_nucleus(ny);
        let pz = pretty_nucleus(nz);
        let mut page = Self::from_spectrum_2d(id, title, skyline, true);
        page.kind = WorkbookPageKind::Cube3d { cube_idx };
        page.subtitle = format!("3D {px}×{py}×{pz} · {} planes", spec3d.nz());
        page.source_cube_idx = Some(cube_idx);
        page.cube_slice = slice_state;
        page
    }

    #[inline]
    pub const fn cube_idx(&self) -> Option<usize> {
        match self.kind {
            WorkbookPageKind::Cube3d { cube_idx } => Some(cube_idx),
            WorkbookPageKind::Spectrum2d => self.source_cube_idx,
        }
    }

    #[inline]
    pub const fn is_cube_3d(&self) -> bool {
        self.cube_idx().is_some()
    }

    #[inline]
    pub fn is_homonuclear(&self) -> bool {
        self.nuc_x == self.nuc_y
    }

    #[inline]
    pub fn is_compatible_overlay_with(&self, other: &Self) -> bool {
        self.nuc_x == other.nuc_x && self.nuc_y == other.nuc_y
    }

    /// Re-extract `spec`, `data_arc`, `nuc_x`, `nuc_y`, and `mini_dots` from `s3` after a plane step,
    /// orientation flip, skyline/slab change, or 3D water suppression.
    pub fn refresh_from_3d_cube(&mut self, s3: &Spectrum3D, reset_viewport: bool) {
        let prev_axes = (self.nuc_x, self.nuc_y);
        let spec2d = self.cube_slice.extract_active_2d(s3);
        let ref_opt = proton_ref_mhz_from_axes(&[&s3.axis_x, &s3.axis_y, &s3.axis_z]);
        self.nuc_x = axis_physical_nucleus(&spec2d.axis_x, ref_opt);
        self.nuc_y = axis_physical_nucleus(&spec2d.axis_y, ref_opt);
        let (base_est, noise_est) = estimate_contour_base_and_noise(&spec2d.data);
        if reset_viewport || prev_axes != (self.nuc_x, self.nuc_y) {
            self.view_ppm = [
                spec2d.axis_x.max_ppm() as f32,
                spec2d.axis_x.min_ppm() as f32,
                spec2d.axis_y.min_ppm() as f32,
                spec2d.axis_y.max_ppm() as f32,
            ];
            self.zoom_history.clear();
            self.base_level = base_est;
        } else if self.base_level <= 0.0 || !self.base_level.is_finite() {
            self.base_level = base_est;
        }
        self.noise_ref = noise_est;
        self.mini_dots = build_mini_2d_preview_dots(&spec2d, 42);
        self.data_arc = Arc::new(spec2d.data.clone());
        self.spec = spec2d;
        self.spec_gen = self.spec_gen.wrapping_add(1);
    }

    pub fn cube_picker_config(&self) -> Cube3dPickerConfig {
        Cube3dPickerConfig {
            snr_threshold: self.cube_slice.snr_threshold.max(2.0),
            allow_negative: self.show_negative,
            exclude_diagonal: self.cube_slice.exclude_diagonal && !self.pick_diagonal,
            exclude_water_band: self.cube_slice.exclude_water_band,
            ..Cube3dPickerConfig::default()
        }
    }

    /// Run true 3D `(F3, F2, F1)` ridge-safe peak picking across the full `s3` cube.
    pub fn pick_3d_cube_peaks(&mut self, s3: &Spectrum3D) -> usize {
        let cfg = self.cube_picker_config();
        self.peaks_3d = pick_cube_3d_ridge_safe(s3, &cfg);
        self.cube_slice.selected_peak_3d_idx = (!self.peaks_3d.is_empty()).then_some(0);
        self.peaks_3d.len()
    }

    /// Run ridge-safe 2D plane peak picking on the currently active orthogonal plane of `s3`,
    /// merging new non-duplicate peaks into `peaks_3d` sorted descending by `|height|`.
    pub fn pick_3d_current_plane_peaks(&mut self, s3: &Spectrum3D) -> usize {
        let cfg = self.cube_picker_config();
        let plane_idx = self.cube_slice.active_plane_idx(s3);
        let ortho_ax = self.cube_slice.orientation.ortho_picker_axis();
        let picked = pick_cube_3d_plane_ridge_safe(s3, ortho_ax, plane_idx, &cfg);
        let tol_x = (s3.axis_x.step_ppm.abs() * 1.25).max(0.015);
        let tol_y = (s3.axis_y.step_ppm.abs() * 1.25).max(0.08);
        let tol_z = (s3.axis_z.step_ppm.abs() * 1.25).max(0.05);
        let mut added = 0usize;
        for pk in picked {
            let dup = self.peaks_3d.iter().any(|ex| {
                (ex.f3_ppm - pk.f3_ppm).abs() <= tol_x
                    && (ex.f2_ppm - pk.f2_ppm).abs() <= tol_y
                    && (ex.f1_ppm - pk.f1_ppm).abs() <= tol_z
            });
            if !dup {
                self.peaks_3d.push(pk);
                added += 1;
            }
        }
        self.peaks_3d
            .sort_by(|a, b| b.height.abs().total_cmp(&a.height.abs()));
        if !self.peaks_3d.is_empty() && self.cube_slice.selected_peak_3d_idx.is_none() {
            self.cube_slice.selected_peak_3d_idx = Some(0);
        }
        added
    }

    /// Interactive click-to-add (`pa` / Peak Add tool) on the active 3D plane, snapping within a small local
    /// 3D voxel neighborhood when `snap_to_extremum` is enabled.
    pub fn add_3d_peak_on_plane(
        &mut self,
        s3: &Spectrum3D,
        plane_x_ppm: f64,
        plane_y_ppm: f64,
        snap_to_extremum: bool,
    ) -> usize {
        let (raw_f3, raw_f2, raw_f1) =
            self.cube_slice
                .coords_from_plane_xy(s3, plane_x_ppm, plane_y_ppm, None);
        let search = if snap_to_extremum {
            if self.cube_slice.skyline_enabled {
                [2, 2, 2]
            } else {
                match self.cube_slice.orientation {
                    CubePlaneOrientation::F3xF2StepF1 => [2, 2, 1],
                    CubePlaneOrientation::F3xF1StepF2 => [2, 1, 2],
                    CubePlaneOrientation::F2xF1StepF3 => [1, 2, 2],
                }
            }
        } else {
            [0, 0, 0]
        };
        let (f3_ppm, f2_ppm, f1_ppm, height, volume, sign) = if snap_to_extremum {
            snap_cube_3d_extremum(s3, raw_f3, raw_f2, raw_f1, search, self.show_negative)
        } else {
            let h = sample_cube_3d_trilinear(s3, raw_f3, raw_f2, raw_f1);
            (
                raw_f3,
                raw_f2,
                raw_f1,
                h,
                h.abs(),
                if h < 0.0 { -1 } else { 1 },
            )
        };
        let noise = (self.noise_ref as f64).max(1e-9);
        let idx = self.peaks_3d.len();
        self.peaks_3d.push(CubePeak3D {
            exp: s3.exp.clone(),
            f3_ppm,
            f2_ppm,
            f1_ppm,
            sign,
            height,
            volume,
            snr: height.abs() / noise,
            label: String::new(),
        });
        self.cube_slice.selected_peak_3d_idx = Some(idx);
        idx
    }

    /// Drag/move `peaks_3d[peak_idx]` across the active 2D plane while preserving its orthogonal shift
    /// and updating its interpolated 3D intensity.
    pub fn move_3d_peak_on_plane(
        &mut self,
        s3: &Spectrum3D,
        peak_idx: usize,
        new_plane_x_ppm: f64,
        new_plane_y_ppm: f64,
    ) -> bool {
        let Some(pk) = self.peaks_3d.get_mut(peak_idx) else {
            return false;
        };
        match self.cube_slice.orientation {
            CubePlaneOrientation::F3xF2StepF1 => {
                pk.f3_ppm = new_plane_x_ppm;
                pk.f2_ppm = new_plane_y_ppm;
            }
            CubePlaneOrientation::F3xF1StepF2 => {
                pk.f3_ppm = new_plane_x_ppm;
                pk.f1_ppm = new_plane_y_ppm;
            }
            CubePlaneOrientation::F2xF1StepF3 => {
                pk.f2_ppm = new_plane_x_ppm;
                pk.f1_ppm = new_plane_y_ppm;
            }
        }
        let h = sample_cube_3d_trilinear(s3, pk.f3_ppm, pk.f2_ppm, pk.f1_ppm);
        pk.height = h;
        pk.sign = if h < 0.0 { -1 } else { 1 };
        pk.snr = h.abs() / (self.noise_ref as f64).max(1e-9);
        true
    }

    /// Snap `peaks_3d[peak_idx]` (`pc` chord) to the nearest 3D sub-voxel parabolic extremum.
    pub fn snap_3d_peak(&mut self, s3: &Spectrum3D, peak_idx: usize) -> bool {
        let Some(cur) = self.peaks_3d.get(peak_idx).cloned() else {
            return false;
        };
        let (f3, f2, f1, height, volume, sign) = snap_cube_3d_extremum(
            s3,
            cur.f3_ppm,
            cur.f2_ppm,
            cur.f1_ppm,
            [2, 2, 2],
            self.show_negative,
        );
        if let Some(pk) = self.peaks_3d.get_mut(peak_idx) {
            pk.f3_ppm = f3;
            pk.f2_ppm = f2;
            pk.f1_ppm = f1;
            pk.height = height;
            pk.volume = volume;
            pk.sign = sign;
            pk.snr = height.abs() / (self.noise_ref as f64).max(1e-9);
            return true;
        }
        false
    }

    /// Relabel `peaks_3d[peak_idx]` (`pl` chord / inline table edit).
    pub fn relabel_3d_peak(&mut self, peak_idx: usize, label: impl Into<String>) -> bool {
        if let Some(pk) = self.peaks_3d.get_mut(peak_idx) {
            pk.label = label.into().trim().to_string();
            true
        } else {
            false
        }
    }

    /// Delete `peaks_3d[peak_idx]` (`pd` / `Delete`), keeping `selected_peak_3d_idx` valid.
    pub fn delete_3d_peak(&mut self, peak_idx: usize) -> Option<CubePeak3D> {
        if peak_idx >= self.peaks_3d.len() {
            return None;
        }
        let removed = self.peaks_3d.remove(peak_idx);
        self.cube_slice.dragging_peak_3d_idx = None;
        self.cube_slice.selected_peak_3d_idx = if self.peaks_3d.is_empty() {
            None
        } else {
            Some(peak_idx.min(self.peaks_3d.len() - 1))
        };
        Some(removed)
    }

    /// Jump the active 3D plane directly to `peaks_3d[peak_idx]` (exiting skyline mode and re-extracting
    /// the slice), returning `(plane_x_ppm, plane_y_ppm)` so the 2D crosshair centers on the peak.
    pub fn jump_to_3d_peak_plane(
        &mut self,
        s3: &Spectrum3D,
        peak_idx: usize,
    ) -> Option<(f64, f64)> {
        let pk = self.peaks_3d.get(peak_idx)?.clone();
        self.cube_slice.skyline_enabled = false;
        self.cube_slice.selected_peak_3d_idx = Some(peak_idx);
        let clamp_idx = |ax: &AxisPpm, ppm: f64| -> usize {
            if ax.size == 0 {
                0
            } else {
                ax.ppm_to_index(ppm)
                    .round()
                    .clamp(0.0, (ax.size - 1) as f64) as usize
            }
        };
        self.cube_slice.plane_f3_idx = clamp_idx(&s3.axis_x, pk.f3_ppm);
        self.cube_slice.plane_f2_idx = clamp_idx(&s3.axis_y, pk.f2_ppm);
        self.cube_slice.plane_f1_idx = clamp_idx(&s3.axis_z, pk.f1_ppm);
        self.refresh_from_3d_cube(s3, false);
        let pr = self.cube_slice.project_peak_3d(s3, peak_idx, &pk);
        Some((pr.plane_x_ppm, pr.plane_y_ppm))
    }

    /// Apply lossless-undoable 3D water suppression (`ws`) directly to `s3` and refresh the active plane/skyline.
    pub fn suppress_water_on_3d_cube(
        &mut self,
        s3: &mut Spectrum3D,
        cfg: &WaterSuppressionConfig,
    ) -> f64 {
        self.water_undo_3d = Some(s3.clone());
        let carrier = suppress_water_3d(s3, cfg);
        self.refresh_from_3d_cube(s3, false);
        carrier
    }

    /// Losslessly restore `s3` prior to 3D water suppression (`wu`) and refresh the active plane/skyline.
    pub fn undo_water_on_3d_cube(&mut self, s3: &mut Spectrum3D) -> bool {
        let Some(prev) = self.water_undo_3d.take() else {
            return false;
        };
        *s3 = prev;
        self.refresh_from_3d_cube(s3, false);
        true
    }
}

/// Specialized analytical studios accessible from the collapsible `"Tools & Studios"` footer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WorkbookStudioKind {
    ChemicalStructure,
    Protein3dStrips,
    Biomolecular3dPdb,
    RelaxationDynamics,
    DosyDiffusion,
    MetabolomicsCohort,
}

/// Active central canvas kind highlighted in the left `"Spectra"` sidebar.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActiveWorkbookSelection {
    Trace1D(usize),
    Spectrum2D(usize),
    Studio(WorkbookStudioKind),
}

/// User actions emitted by [`render_workbook_sidebar`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WorkbookSidebarAction {
    Select1dPage(usize),
    Toggle1dOverlay(usize),
    Remove1dPage(usize),
    Select2dPage(usize),
    Toggle2dOverlay(usize),
    Remove2dPage(usize),
    OpenStudio(WorkbookStudioKind),
    TriggerOpenFiles,
}

// ============================================================================
// 5. 2D Workstation Tool Rail & Aligned `(F2, F1)` 1D Marginal Projection Engine
// ============================================================================

/// Primary pointer interaction mode on the 2D NMR contour canvas.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ToolMode2d {
    /// Click-drag or 2-click rubber-band 2D zoom box (`Z` key; right/middle-drag pans).
    #[default]
    ZoomBox,
    /// Primary-drag 2D translation pan (`H` key).
    Pan,
    /// Live crosshair & orthogonal `(F2, F1)` 1D row/column slice inspection (`X` key).
    CrosshairSlice,
    /// Click peak to snap/select or add/pick 2D cross-peaks (`K` key).
    PickPeaks,
}

/// Data source mode for the Top `F2` and Left `F1` 1D marginal strips.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum MarginalSourceMode {
    /// Prefer an external loaded 1D trace from `Stack1dState` when nucleus & ppm bounds overlap;
    /// otherwise fall back to fast 2D skyline maximum-magnitude projection.
    #[default]
    AutoExternalOrSkyline,
    /// Force internal 2D skyline projection across the visible viewport window.
    SkylineOnly,
    /// Live 1D row (`F2` at `crosshair.n_ppm`) and column (`F1` at `crosshair.h_ppm`) cross-sections.
    CrosshairSlice,
}

impl MarginalSourceMode {
    pub const fn short_badge(self) -> &'static str {
        match self {
            Self::AutoExternalOrSkyline => "Auto 1D",
            Self::SkylineOnly => "Skyline",
            Self::CrosshairSlice => "Slice",
        }
    }

    pub const fn cycle_next(self) -> Self {
        match self {
            Self::AutoExternalOrSkyline => Self::SkylineOnly,
            Self::SkylineOnly => Self::CrosshairSlice,
            Self::CrosshairSlice => Self::AutoExternalOrSkyline,
        }
    }
}

/// Evaluated 1D curve `(ppm, amplitude)` for a Top `F2` or Left `F1` marginal strip.
#[derive(Clone, Debug)]
pub struct MarginalCurve1D {
    /// `(ppm, intensity)` samples ordered monotonically across the visible viewport axis.
    pub points: Vec<(f64, f64)>,
    pub max_abs: f64,
    pub source_label: String,
    pub is_external_1d: bool,
}

#[derive(Clone, Debug, PartialEq)]
struct SkylineCacheKey {
    spec_gen: u64,
    view_bits: [u64; 4],
    cols: usize,
    rows: usize,
    slice_mode: bool,
    xh_bits: [u64; 2],
}

/// Interactive state for the upgraded 2D Contour + F2/F1 Marginal Projections Workstation.
#[derive(Clone, Debug)]
pub struct TwoDCanvasState {
    pub tool_mode: ToolMode2d,
    pub show_projections: bool,
    pub show_params_card: bool,
    pub show_peak_table: bool,
    pub pick_diagonal: bool,
    pub selected_peak_idx: Option<usize>,
    pub dragging_peak_idx: Option<usize>,
    pub pick_box_start_pos: Option<egui::Pos2>,
    pub marginal_mode: MarginalSourceMode,
    pub f2_gain: f32,
    pub f1_gain: f32,
    /// Stack of `[h_left, h_right, n_top, n_bottom]` windows for 1-click Undo Zoom.
    pub zoom_history: Vec<[f32; 4]>,
    pub drag_box_start_pos: Option<egui::Pos2>,
    skyline_cache_key: Option<SkylineCacheKey>,
    cached_f2_internal: Option<MarginalCurve1D>,
    cached_f1_internal: Option<MarginalCurve1D>,
}

impl Default for TwoDCanvasState {
    fn default() -> Self {
        Self {
            tool_mode: ToolMode2d::ZoomBox,
            show_projections: true,
            show_params_card: true,
            show_peak_table: false,
            pick_diagonal: false,
            selected_peak_idx: None,
            dragging_peak_idx: None,
            pick_box_start_pos: None,
            marginal_mode: MarginalSourceMode::AutoExternalOrSkyline,
            f2_gain: 1.0,
            f1_gain: 1.0,
            zoom_history: Vec::new(),
            drag_box_start_pos: None,
            skyline_cache_key: None,
            cached_f2_internal: None,
            cached_f1_internal: None,
        }
    }
}

impl TwoDCanvasState {
    pub fn push_zoom(&mut self, current_view: [f32; 4]) {
        if self.zoom_history.last().is_some_and(|prev| {
            prev.iter()
                .zip(current_view)
                .all(|(a, b)| (a - b).abs() < 1e-5)
        }) {
            return;
        }
        if self.zoom_history.len() >= 48 {
            self.zoom_history.remove(0);
        }
        self.zoom_history.push(current_view);
    }

    pub fn pop_zoom(&mut self) -> Option<[f32; 4]> {
        self.zoom_history.pop()
    }

    pub fn effective_marginal_mode(&self) -> MarginalSourceMode {
        if self.tool_mode == ToolMode2d::CrosshairSlice {
            MarginalSourceMode::CrosshairSlice
        } else {
            self.marginal_mode
        }
    }
}

/// Top-level Multi-Page Workbook state owned by `NmrWorkbenchApp`.
#[derive(Clone, Debug)]
pub struct WorkbookState {
    pub show_sidebar: bool,
    pub show_tools_section: bool,
    pub pages_2d: Vec<Stored2dPage>,
    pub active_2d_idx: usize,
    next_page_id: u64,
    pub two_d: TwoDCanvasState,
    /// Shared chemical-shift cursor map keyed by canonical physical nucleus (`"1H"`, `"15N"`, `"13C"`, ...).
    pub shared_nucleus_ppm: BTreeMap<String, f64>,
}

impl Default for WorkbookState {
    fn default() -> Self {
        Self {
            show_sidebar: true,
            show_tools_section: false,
            pages_2d: Vec::new(),
            active_2d_idx: 0,
            next_page_id: 100,
            two_d: TwoDCanvasState::default(),
            shared_nucleus_ppm: BTreeMap::new(),
        }
    }
}

impl WorkbookState {
    #[inline]
    pub fn invalidate_marginal_cache(&mut self) {
        self.two_d.skyline_cache_key = None;
    }

    pub fn record_1d_cursor_by_nucleus(&mut self, nucleus: &str, ppm: f64) {
        if ppm.is_finite() && !nucleus.is_empty() {
            self.shared_nucleus_ppm.insert(nucleus.to_string(), ppm);
        }
    }

    pub fn record_2d_crosshair_by_nucleus(
        &mut self,
        nuc_x: &str,
        x_ppm: f64,
        nuc_y: &str,
        y_ppm: f64,
    ) {
        if x_ppm.is_finite() && !nuc_x.is_empty() {
            self.shared_nucleus_ppm.insert(nuc_x.to_string(), x_ppm);
        }
        if y_ppm.is_finite() && !nuc_y.is_empty() {
            self.shared_nucleus_ppm.insert(nuc_y.to_string(), y_ppm);
        }
    }

    /// Resolve `(x_ppm, y_ppm)` for an active 2D/3D page's `(nuc_x, nuc_y)` from `shared_nucleus_ppm`,
    /// falling back to `(fallback_x, fallback_y)` when a nucleus has not been visited yet.
    pub fn shared_crosshair_for_axes(
        &self,
        nuc_x: &str,
        nuc_y: &str,
        fallback_x: f64,
        fallback_y: f64,
    ) -> (f64, f64) {
        let x = self
            .shared_nucleus_ppm
            .get(nuc_x)
            .copied()
            .filter(|v| v.is_finite())
            .unwrap_or(fallback_x);
        let y = self
            .shared_nucleus_ppm
            .get(nuc_y)
            .copied()
            .filter(|v| v.is_finite())
            .unwrap_or(fallback_y);
        (x, y)
    }

    /// Synchronize all open `WorkbookPageKind::Cube3d` pages to `(h_ppm, n_ppm)` (`gs` chord)
    /// and return the page index of the first synchronized `Cube3d` page (if any).
    pub fn sync_open_3d_cubes_to_hn(
        &mut self,
        cubes_3d: &[Spectrum3D],
        h_ppm: f64,
        n_ppm: f64,
    ) -> Option<(usize, f64, f64)> {
        self.record_2d_crosshair_by_nucleus("1H", h_ppm, "15N", n_ppm);
        let mut first_hit: Option<(usize, f64, f64)> = None;
        for (p_idx, pg) in self.pages_2d.iter_mut().enumerate() {
            let Some(c_idx) = pg.cube_idx() else {
                continue;
            };
            let Some(s3) = cubes_3d.get(c_idx) else {
                continue;
            };
            let (sx, sy) = pg.cube_slice.sync_to_hn(s3, h_ppm, n_ppm);
            pg.refresh_from_3d_cube(s3, false);
            if first_hit.is_none() {
                first_hit = Some((p_idx, sx, sy));
            }
        }
        if first_hit.is_some() {
            self.two_d.skyline_cache_key = None;
        }
        first_hit
    }

    /// Reindex `Cube3d` pages after removing `removed_cube_idx` from `cubes_3d`.
    pub fn reindex_after_cube_removed(&mut self, removed_cube_idx: usize) {
        for pg in &mut self.pages_2d {
            if let Some(ci) = pg.cube_idx() {
                if ci > removed_cube_idx {
                    let next_ci = ci - 1;
                    pg.source_cube_idx = Some(next_ci);
                    pg.kind = WorkbookPageKind::Cube3d { cube_idx: next_ci };
                }
            }
        }
    }
    pub fn alloc_page_id(&mut self) -> u64 {
        let id = self.next_page_id;
        self.next_page_id = self.next_page_id.wrapping_add(1);
        id
    }

    pub fn clear_2d_pages(&mut self) {
        self.pages_2d.clear();
        self.active_2d_idx = 0;
        self.two_d.zoom_history.clear();
    }

    pub fn set_single_2d_page(
        &mut self,
        label: impl Into<String>,
        spec: Spectrum2D,
        has_3d_strips: bool,
    ) {
        self.pages_2d.clear();
        let id = self.alloc_page_id();
        self.pages_2d.push(Stored2dPage::from_spectrum_2d(
            id,
            label,
            spec,
            has_3d_strips,
        ));
        self.active_2d_idx = 0;
        self.two_d.zoom_history.clear();
    }

    pub fn add_3d_cube_page(
        &mut self,
        cube_idx: usize,
        label: impl Into<String>,
        spec3d: &Spectrum3D,
    ) -> usize {
        let id = self.alloc_page_id();
        self.pages_2d
            .push(Stored2dPage::from_3d_cube(id, cube_idx, label, spec3d));
        self.pages_2d.len() - 1
    }

    pub fn add_2d_page(
        &mut self,
        label: impl Into<String>,
        spec: Spectrum2D,
        overlay_enabled: bool,
        has_3d_strips: bool,
    ) -> usize {
        let id = self.alloc_page_id();
        let mut p = Stored2dPage::from_spectrum_2d(id, label, spec, has_3d_strips);
        p.overlay_enabled = overlay_enabled;
        self.pages_2d.push(p);
        self.pages_2d.len() - 1
    }

    pub fn add_demo_2d_page(
        &mut self,
        label: impl Into<String>,
        spec: Spectrum2D,
        overlay_enabled: bool,
        has_3d_strips: bool,
    ) -> usize {
        let idx = self.add_2d_page(label, spec, overlay_enabled, has_3d_strips);
        self.pages_2d[idx].is_demo = true;
        idx
    }

    pub fn evict_demo_2d_pages(&mut self) {
        self.pages_2d.retain(|p| !p.is_demo);
        if self.active_2d_idx >= self.pages_2d.len() {
            self.active_2d_idx = self.pages_2d.len().saturating_sub(1);
        }
    }

    /// Replace `spec` + `data_arc` + `mini_dots` on the currently active 2D page (e.g. after Water Suppression or Undo).
    pub fn update_active_2d_spectrum(&mut self, spec: Spectrum2D) {
        if let Some(pg) = self.pages_2d.get_mut(self.active_2d_idx) {
            let (base_est, noise_est) = estimate_contour_base_and_noise(&spec.data);
            pg.mini_dots = build_mini_2d_preview_dots(&spec, 42);
            pg.data_arc = Arc::new(spec.data.clone());
            pg.spec = spec;
            pg.spec_gen = pg.spec_gen.wrapping_add(1);
            pg.noise_ref = noise_est;
            if pg.base_level <= 0.0 || !pg.base_level.is_finite() {
                pg.base_level = base_est;
            }
            self.two_d.skyline_cache_key = None;
        }
    }

    /// Build `OverlayLayerSpec` entries for all non-active compatible `pages_2d` that have `overlay_enabled == true`.
    pub fn build_compatible_overlay_layers(
        &self,
        paper_white: bool,
    ) -> Vec<(String, OverlayLayerSpec)> {
        let Some(active) = self.pages_2d.get(self.active_2d_idx) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        let mut pal_idx = 1usize;
        for (idx, pg) in self.pages_2d.iter().enumerate() {
            if idx == self.active_2d_idx || !pg.overlay_enabled {
                continue;
            }
            if !pg.is_compatible_overlay_with(active) {
                continue;
            }
            let mut ov = OverlayLayerSpec::from_spectrum_2d(
                pg.id,
                pg.spec_gen,
                &pg.spec,
                Arc::clone(&pg.data_arc),
                pg.base_level,
                pal_idx,
            );
            let (pos, neg) = contour_palette_for_theme(pal_idx, paper_white);
            ov.pos_rgba = pos;
            ov.neg_rgba = neg;
            ov.level_factor = pg.level_factor;
            ov.num_levels = pg.num_levels;
            ov.show_negative = pg.show_negative;
            ov.ppm_nudge = pg.ppm_nudge;
            ov.visible = true;
            out.push((pg.title.clone(), ov));
            pal_idx += 1;
        }
        out
    }
}

// ============================================================================
// 6. Mathematical Skyline, Slice & External 1D Marginal Projection Evaluators
// ============================================================================

/// Find the best matching external [`Trace1d`] in `stack1d` whose nucleus matches `target_nucleus`
/// and whose chemical-shift span overlaps `[ppm_a, ppm_b]`.
pub fn find_matching_external_1d_trace<'a>(
    stack1d: &'a Stack1dState,
    target_nucleus: &str,
    ppm_a: f64,
    ppm_b: f64,
) -> Option<&'a Trace1d> {
    if stack1d.traces.is_empty() {
        return None;
    }
    let lo = ppm_a.min(ppm_b);
    let hi = ppm_a.max(ppm_b);
    let matches_trace = |tr: &Trace1d| -> bool {
        if tr.is_empty() || trace_1d_nucleus(tr) != target_nucleus {
            return false;
        }
        let Some((t_lo, t_hi)) = tr.ppm_bounds() else {
            return false;
        };
        let ov = t_hi.min(hi) - t_lo.max(lo);
        ov > 0.02 * (hi - lo).max(1e-4)
    };

    let ref_idx = stack1d.ref_trace_idx.min(stack1d.traces.len() - 1);
    if matches_trace(&stack1d.traces[ref_idx]) {
        return Some(&stack1d.traces[ref_idx]);
    }
    stack1d.traces.iter().find(|t| matches_trace(t))
}

/// Compute signed maximum-magnitude skyline projection along `F2` (`x` axis, `[ppm_left .. ppm_right]`)
/// from a [`Spectrum2D`] across `F1` rows inside `[f1_top_ppm .. f1_bottom_ppm]`.
pub fn compute_f2_skyline_curve(
    spec: &Spectrum2D,
    ppm_left: f64,
    ppm_right: f64,
    f1_top_ppm: f64,
    f1_bottom_ppm: f64,
    target_bins: usize,
) -> MarginalCurve1D {
    let nx = spec.nx();
    let ny = spec.ny();
    let bins = target_bins.clamp(16, 1024);
    if nx == 0 || ny == 0 || spec.data.len() < nx * ny {
        return MarginalCurve1D {
            points: Vec::new(),
            max_abs: 1.0,
            source_label: "F2 Skyline".to_string(),
            is_external_1d: false,
        };
    }

    let iy_a = spec
        .axis_y
        .ppm_to_index(f1_top_ppm)
        .round()
        .clamp(0.0, (ny - 1) as f64) as usize;
    let iy_b = spec
        .axis_y
        .ppm_to_index(f1_bottom_ppm)
        .round()
        .clamp(0.0, (ny - 1) as f64) as usize;
    let (iy0, iy1) = (iy_a.min(iy_b), iy_a.max(iy_b));

    let ix_a = spec
        .axis_x
        .ppm_to_index(ppm_left)
        .floor()
        .clamp(0.0, (nx - 1) as f64) as usize;
    let ix_b = spec
        .axis_x
        .ppm_to_index(ppm_right)
        .ceil()
        .clamp(0.0, (nx - 1) as f64) as usize;
    let (ix0, ix1) = (ix_a.min(ix_b), ix_a.max(ix_b));
    let span_ix = ix1 - ix0 + 1;

    let mut points = Vec::with_capacity(span_ix.min(bins * 2));
    let mut max_abs = 0.0_f64;

    if span_ix <= bins {
        for ix in ix0..=ix1 {
            let mut best = 0.0_f32;
            for iy in iy0..=iy1 {
                let v = spec.data[iy * nx + ix];
                if v.abs() > best.abs() {
                    best = v;
                }
            }
            let p = spec.axis_x.index_to_ppm(ix as f64);
            let val = best as f64;
            max_abs = max_abs.max(val.abs());
            points.push((p, val));
        }
    } else {
        for b in 0..bins {
            let t0 = b as f64 / bins as f64;
            let t1 = (b + 1) as f64 / bins as f64;
            let p0 = ppm_left + t0 * (ppm_right - ppm_left);
            let p1 = ppm_left + t1 * (ppm_right - ppm_left);
            let j_a = spec
                .axis_x
                .ppm_to_index(p0)
                .round()
                .clamp(0.0, (nx - 1) as f64) as usize;
            let j_b = spec
                .axis_x
                .ppm_to_index(p1)
                .round()
                .clamp(0.0, (nx - 1) as f64) as usize;
            let (j0, j1) = (j_a.min(j_b), j_a.max(j_b));
            let mut best = 0.0_f32;
            let mut best_ix = j0;
            for ix in j0..=j1 {
                for iy in iy0..=iy1 {
                    let v = spec.data[iy * nx + ix];
                    if v.abs() > best.abs() {
                        best = v;
                        best_ix = ix;
                    }
                }
            }
            let p = spec
                .axis_x
                .index_to_ppm(best_ix as f64)
                .clamp(ppm_left.min(ppm_right), ppm_left.max(ppm_right));
            let val = best as f64;
            max_abs = max_abs.max(val.abs());
            points.push((p, val));
        }
    }

    MarginalCurve1D {
        points,
        max_abs: max_abs.max(1e-12),
        source_label: "F2 Skyline".to_string(),
        is_external_1d: false,
    }
}

/// Compute signed maximum-magnitude skyline projection along `F1` (`y` axis, `[ppm_top .. ppm_bottom]`)
/// from a [`Spectrum2D`] across `F2` columns inside `[f2_left_ppm .. f2_right_ppm]`.
pub fn compute_f1_skyline_curve(
    spec: &Spectrum2D,
    ppm_top: f64,
    ppm_bottom: f64,
    f2_left_ppm: f64,
    f2_right_ppm: f64,
    target_bins: usize,
) -> MarginalCurve1D {
    let nx = spec.nx();
    let ny = spec.ny();
    let bins = target_bins.clamp(16, 1024);
    if nx == 0 || ny == 0 || spec.data.len() < nx * ny {
        return MarginalCurve1D {
            points: Vec::new(),
            max_abs: 1.0,
            source_label: "F1 Skyline".to_string(),
            is_external_1d: false,
        };
    }

    let ix_a = spec
        .axis_x
        .ppm_to_index(f2_left_ppm)
        .round()
        .clamp(0.0, (nx - 1) as f64) as usize;
    let ix_b = spec
        .axis_x
        .ppm_to_index(f2_right_ppm)
        .round()
        .clamp(0.0, (nx - 1) as f64) as usize;
    let (ix0, ix1) = (ix_a.min(ix_b), ix_a.max(ix_b));

    let iy_a = spec
        .axis_y
        .ppm_to_index(ppm_top)
        .floor()
        .clamp(0.0, (ny - 1) as f64) as usize;
    let iy_b = spec
        .axis_y
        .ppm_to_index(ppm_bottom)
        .ceil()
        .clamp(0.0, (ny - 1) as f64) as usize;
    let (iy0, iy1) = (iy_a.min(iy_b), iy_a.max(iy_b));
    let span_iy = iy1 - iy0 + 1;

    let mut points = Vec::with_capacity(span_iy.min(bins * 2));
    let mut max_abs = 0.0_f64;

    if span_iy <= bins {
        for iy in iy0..=iy1 {
            let row = &spec.data[iy * nx..(iy + 1) * nx];
            let mut best = 0.0_f32;
            for &v in &row[ix0..=ix1] {
                if v.abs() > best.abs() {
                    best = v;
                }
            }
            let p = spec.axis_y.index_to_ppm(iy as f64);
            let val = best as f64;
            max_abs = max_abs.max(val.abs());
            points.push((p, val));
        }
    } else {
        for b in 0..bins {
            let t0 = b as f64 / bins as f64;
            let t1 = (b + 1) as f64 / bins as f64;
            let p0 = ppm_top + t0 * (ppm_bottom - ppm_top);
            let p1 = ppm_top + t1 * (ppm_bottom - ppm_top);
            let j_a = spec
                .axis_y
                .ppm_to_index(p0)
                .round()
                .clamp(0.0, (ny - 1) as f64) as usize;
            let j_b = spec
                .axis_y
                .ppm_to_index(p1)
                .round()
                .clamp(0.0, (ny - 1) as f64) as usize;
            let (j0, j1) = (j_a.min(j_b), j_a.max(j_b));
            let mut best = 0.0_f32;
            let mut best_iy = j0;
            for iy in j0..=j1 {
                let row = &spec.data[iy * nx..(iy + 1) * nx];
                for &v in &row[ix0..=ix1] {
                    if v.abs() > best.abs() {
                        best = v;
                        best_iy = iy;
                    }
                }
            }
            let p = spec
                .axis_y
                .index_to_ppm(best_iy as f64)
                .clamp(ppm_top.min(ppm_bottom), ppm_top.max(ppm_bottom));
            let val = best as f64;
            max_abs = max_abs.max(val.abs());
            points.push((p, val));
        }
    }

    MarginalCurve1D {
        points,
        max_abs: max_abs.max(1e-12),
        source_label: "F1 Skyline".to_string(),
        is_external_1d: false,
    }
}

/// Extract a live 1D row slice along `F2` (`[ppm_left .. ppm_right]`) at `f1_slice_ppm`.
pub fn compute_f2_slice_curve(
    spec: &Spectrum2D,
    ppm_left: f64,
    ppm_right: f64,
    f1_slice_ppm: f64,
    target_bins: usize,
) -> MarginalCurve1D {
    let nx = spec.nx();
    let ny = spec.ny();
    if nx == 0 || ny == 0 || spec.data.len() < nx * ny {
        return MarginalCurve1D {
            points: Vec::new(),
            max_abs: 1.0,
            source_label: "F2 Slice".to_string(),
            is_external_1d: false,
        };
    }
    let iy_f = spec
        .axis_y
        .ppm_to_index(f1_slice_ppm)
        .clamp(0.0, (ny - 1) as f64);
    let iy0 = (iy_f.floor() as usize).min(ny - 1);
    let iy1 = (iy0 + 1).min(ny - 1);
    let ty = (iy_f - iy0 as f64) as f32;

    let n_samples = target_bins.clamp(32, 1024).min(nx.max(32));
    let mut points = Vec::with_capacity(n_samples);
    let mut max_abs = 0.0_f64;
    for k in 0..n_samples {
        let u = k as f64 / (n_samples.saturating_sub(1).max(1) as f64);
        let p = ppm_left + u * (ppm_right - ppm_left);
        let ix_f = spec.axis_x.ppm_to_index(p);
        let val = if ix_f >= 0.0 && ix_f <= (nx - 1) as f64 {
            let ix0 = (ix_f.floor() as usize).min(nx - 1);
            let ix1 = (ix0 + 1).min(nx - 1);
            let tx = (ix_f - ix0 as f64) as f32;
            let v0 = spec.data[iy0 * nx + ix0] * (1.0 - tx) + spec.data[iy0 * nx + ix1] * tx;
            let v1 = spec.data[iy1 * nx + ix0] * (1.0 - tx) + spec.data[iy1 * nx + ix1] * tx;
            (v0 * (1.0 - ty) + v1 * ty) as f64
        } else {
            0.0
        };
        max_abs = max_abs.max(val.abs());
        points.push((p, val));
    }

    MarginalCurve1D {
        points,
        max_abs: max_abs.max(1e-12),
        source_label: format!("Slice F1={f1_slice_ppm:.2}"),
        is_external_1d: false,
    }
}

/// Extract a live 1D column slice along `F1` (`[ppm_top .. ppm_bottom]`) at `f2_slice_ppm`.
pub fn compute_f1_slice_curve(
    spec: &Spectrum2D,
    ppm_top: f64,
    ppm_bottom: f64,
    f2_slice_ppm: f64,
    target_bins: usize,
) -> MarginalCurve1D {
    let nx = spec.nx();
    let ny = spec.ny();
    if nx == 0 || ny == 0 || spec.data.len() < nx * ny {
        return MarginalCurve1D {
            points: Vec::new(),
            max_abs: 1.0,
            source_label: "F1 Slice".to_string(),
            is_external_1d: false,
        };
    }
    let ix_f = spec
        .axis_x
        .ppm_to_index(f2_slice_ppm)
        .clamp(0.0, (nx - 1) as f64);
    let ix0 = (ix_f.floor() as usize).min(nx - 1);
    let ix1 = (ix0 + 1).min(nx - 1);
    let tx = (ix_f - ix0 as f64) as f32;

    let n_samples = target_bins.clamp(32, 1024).min(ny.max(32));
    let mut points = Vec::with_capacity(n_samples);
    let mut max_abs = 0.0_f64;
    for k in 0..n_samples {
        let v_frac = k as f64 / (n_samples.saturating_sub(1).max(1) as f64);
        let p = ppm_top + v_frac * (ppm_bottom - ppm_top);
        let iy_f = spec.axis_y.ppm_to_index(p);
        let val = if iy_f >= 0.0 && iy_f <= (ny - 1) as f64 {
            let iy0 = (iy_f.floor() as usize).min(ny - 1);
            let iy1 = (iy0 + 1).min(ny - 1);
            let ty = (iy_f - iy0 as f64) as f32;
            let v0 = spec.data[iy0 * nx + ix0] * (1.0 - tx) + spec.data[iy0 * nx + ix1] * tx;
            let v1 = spec.data[iy1 * nx + ix0] * (1.0 - tx) + spec.data[iy1 * nx + ix1] * tx;
            (v0 * (1.0 - ty) + v1 * ty) as f64
        } else {
            0.0
        };
        max_abs = max_abs.max(val.abs());
        points.push((p, val));
    }

    MarginalCurve1D {
        points,
        max_abs: max_abs.max(1e-12),
        source_label: format!("Slice F2={f2_slice_ppm:.2}"),
        is_external_1d: false,
    }
}

fn curve_from_external_trace(
    trace: &Trace1d,
    ppm_a: f64,
    ppm_b: f64,
    target_cols: usize,
) -> MarginalCurve1D {
    let pts = trace.decimate_for_viewport(ppm_a.min(ppm_b), ppm_a.max(ppm_b), target_cols.max(64));
    let max_abs = pts
        .iter()
        .fold(0.0_f64, |acc, &(_, v)| acc.max(v.abs()))
        .max(1e-12);
    MarginalCurve1D {
        points: pts,
        max_abs,
        source_label: format!("1D {}", pretty_nucleus(trace_1d_nucleus(trace))),
        is_external_1d: true,
    }
}

/// Evaluate and cache both `(F2_curve, F1_curve)` aligned with `view_ppm = [h_left, h_right, n_top, n_bottom]`.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_2d_marginals(
    two_d: &mut TwoDCanvasState,
    spec: &Spectrum2D,
    spec_gen: u64,
    nuc_x: &str,
    nuc_y: &str,
    view_ppm: [f32; 4],
    crosshair: &SyncCrosshair,
    stack1d: &Stack1dState,
    plot_width_px: f32,
    plot_height_px: f32,
) -> (MarginalCurve1D, MarginalCurve1D) {
    let mode = two_d.effective_marginal_mode();
    let cols = (plot_width_px.ceil() as usize).clamp(48, 900);
    let rows = (plot_height_px.ceil() as usize).clamp(48, 900);

    let ext_f2 = if mode == MarginalSourceMode::AutoExternalOrSkyline {
        find_matching_external_1d_trace(stack1d, nuc_x, view_ppm[0] as f64, view_ppm[1] as f64)
            .map(|tr| curve_from_external_trace(tr, view_ppm[0] as f64, view_ppm[1] as f64, cols))
    } else {
        None
    };

    let ext_f1 = if mode == MarginalSourceMode::AutoExternalOrSkyline {
        find_matching_external_1d_trace(stack1d, nuc_y, view_ppm[2] as f64, view_ppm[3] as f64)
            .map(|tr| curve_from_external_trace(tr, view_ppm[2] as f64, view_ppm[3] as f64, rows))
    } else {
        None
    };

    if let (Some(c2), Some(c1)) = (ext_f2.clone(), ext_f1.clone()) {
        return (c2, c1);
    }

    let is_slice = mode == MarginalSourceMode::CrosshairSlice;
    let key = SkylineCacheKey {
        spec_gen,
        view_bits: view_ppm.map(|v| (v as f64).to_bits()),
        cols,
        rows,
        slice_mode: is_slice,
        xh_bits: if is_slice {
            [crosshair.h_ppm.to_bits(), crosshair.n_ppm.to_bits()]
        } else {
            [0, 0]
        },
    };

    if two_d.skyline_cache_key.as_ref() != Some(&key)
        || two_d.cached_f2_internal.is_none()
        || two_d.cached_f1_internal.is_none()
    {
        let (c2_int, c1_int) = if is_slice {
            (
                compute_f2_slice_curve(
                    spec,
                    view_ppm[0] as f64,
                    view_ppm[1] as f64,
                    crosshair.n_ppm,
                    cols,
                ),
                compute_f1_slice_curve(
                    spec,
                    view_ppm[2] as f64,
                    view_ppm[3] as f64,
                    crosshair.h_ppm,
                    rows,
                ),
            )
        } else {
            (
                compute_f2_skyline_curve(
                    spec,
                    view_ppm[0] as f64,
                    view_ppm[1] as f64,
                    view_ppm[2] as f64,
                    view_ppm[3] as f64,
                    cols,
                ),
                compute_f1_skyline_curve(
                    spec,
                    view_ppm[2] as f64,
                    view_ppm[3] as f64,
                    view_ppm[0] as f64,
                    view_ppm[1] as f64,
                    rows,
                ),
            )
        };
        two_d.skyline_cache_key = Some(key);
        two_d.cached_f2_internal = Some(c2_int);
        two_d.cached_f1_internal = Some(c1_int);
    }

    let f2_out = ext_f2.unwrap_or_else(|| {
        two_d
            .cached_f2_internal
            .clone()
            .unwrap_or_else(|| MarginalCurve1D {
                points: Vec::new(),
                max_abs: 1.0,
                source_label: "F2".to_string(),
                is_external_1d: false,
            })
    });
    let f1_out = ext_f1.unwrap_or_else(|| {
        two_d
            .cached_f1_internal
            .clone()
            .unwrap_or_else(|| MarginalCurve1D {
                points: Vec::new(),
                max_abs: 1.0,
                source_label: "F1".to_string(),
                is_external_1d: false,
            })
    });
    (f2_out, f1_out)
}

/// Exact horizontal screen `x` coordinate aligned pixel-for-pixel with `SyncCrosshair::ppm_to_screen_2d`.
#[inline]
pub fn f2_ppm_to_screen_x(view_ppm: [f32; 4], plot_or_f2_rect: egui::Rect, x_ppm: f64) -> f32 {
    SyncCrosshair::ppm_to_screen_2d(view_ppm, plot_or_f2_rect, x_ppm, view_ppm[2] as f64).x
}

/// Exact vertical screen `y` coordinate aligned pixel-for-pixel with `SyncCrosshair::ppm_to_screen_2d`.
#[inline]
pub fn f1_ppm_to_screen_y(view_ppm: [f32; 4], plot_or_f1_rect: egui::Rect, y_ppm: f64) -> f32 {
    SyncCrosshair::ppm_to_screen_2d(view_ppm, plot_or_f1_rect, view_ppm[0] as f64, y_ppm).y
}

/// Paint the Top `F2` (`64 px` tall) 1D marginal trace inside `f2_rect` (`f2_rect.left() == plot_rect.left()`,
/// `f2_rect.right() == plot_rect.right()`).
pub fn paint_f2_marginal_strip(
    painter: &egui::Painter,
    f2_rect: egui::Rect,
    view_ppm: [f32; 4],
    curve: &MarginalCurve1D,
    gain: f32,
    crosshair_h_ppm: f64,
    theme: &CanvasTheme2D,
) {
    let p = painter.with_clip_rect(f2_rect);
    p.rect_filled(f2_rect, 2.0, theme.marginal_bg);

    let y_base = f2_rect.bottom() - 8.0;
    let headroom = (f2_rect.height() - 14.0).max(12.0);
    p.line_segment(
        [
            egui::pos2(f2_rect.left(), y_base),
            egui::pos2(f2_rect.right(), y_base),
        ],
        egui::Stroke::new(0.8, theme.grid_line),
    );

    if curve.points.len() >= 2 {
        let trace_col = if curve.is_external_1d {
            theme.marginal_trace_ext
        } else {
            theme.marginal_trace_sky
        };
        let scale = (headroom * gain.clamp(0.05, 64.0)) / (curve.max_abs.max(1e-12) as f32);
        let mut poly: Vec<egui::Pos2> = Vec::with_capacity(curve.points.len());
        for &(ppm, val) in &curve.points {
            let sx = f2_ppm_to_screen_x(view_ppm, f2_rect, ppm);
            let sy =
                (y_base - (val as f32) * scale).clamp(f2_rect.top() + 2.0, f2_rect.bottom() - 2.0);
            poly.push(egui::pos2(sx, sy));
        }
        // Sort by screen x if external decimation arrived in reverse axis direction
        if poly
            .first()
            .zip(poly.last())
            .is_some_and(|(a, b)| a.x > b.x)
        {
            poly.reverse();
        }
        p.add(egui::Shape::line(poly, egui::Stroke::new(1.15, trace_col)));
    }

    let xh_x = f2_ppm_to_screen_x(view_ppm, f2_rect, crosshair_h_ppm);
    if (f2_rect.left()..=f2_rect.right()).contains(&xh_x) {
        p.line_segment(
            [
                egui::pos2(xh_x, f2_rect.top()),
                egui::pos2(xh_x, f2_rect.bottom()),
            ],
            egui::Stroke::new(0.85, theme.crosshair_unlocked),
        );
    }

    p.text(
        egui::pos2(f2_rect.right() - 5.0, f2_rect.top() + 3.0),
        egui::Align2::RIGHT_TOP,
        &curve.source_label,
        egui::FontId::monospace(9.0),
        theme.axis_muted,
    );
    p.rect_stroke(
        f2_rect,
        2.0,
        egui::Stroke::new(1.0, theme.marginal_border),
        egui::StrokeKind::Inside,
    );
}

/// Paint the rotated 90° Left `F1` (`64 px` wide) 1D marginal trace inside `f1_rect`
/// (`f1_rect.top() == plot_rect.top()`, `f1_rect.bottom() == plot_rect.bottom()`).
pub fn paint_f1_marginal_strip(
    painter: &egui::Painter,
    f1_rect: egui::Rect,
    view_ppm: [f32; 4],
    curve: &MarginalCurve1D,
    gain: f32,
    crosshair_n_ppm: f64,
    theme: &CanvasTheme2D,
) {
    let p = painter.with_clip_rect(f1_rect);
    p.rect_filled(f1_rect, 2.0, theme.marginal_bg);

    let x_base = f1_rect.right() - 8.0;
    let headroom = (f1_rect.width() - 14.0).max(12.0);
    p.line_segment(
        [
            egui::pos2(x_base, f1_rect.top()),
            egui::pos2(x_base, f1_rect.bottom()),
        ],
        egui::Stroke::new(0.8, theme.grid_line),
    );

    if curve.points.len() >= 2 {
        let trace_col = if curve.is_external_1d {
            theme.marginal_trace_ext
        } else {
            theme.marginal_trace_sky
        };
        let scale = (headroom * gain.clamp(0.05, 64.0)) / (curve.max_abs.max(1e-12) as f32);
        let mut poly: Vec<egui::Pos2> = Vec::with_capacity(curve.points.len());
        for &(ppm, val) in &curve.points {
            let sy = f1_ppm_to_screen_y(view_ppm, f1_rect, ppm);
            let sx =
                (x_base - (val as f32) * scale).clamp(f1_rect.left() + 2.0, f1_rect.right() - 2.0);
            poly.push(egui::pos2(sx, sy));
        }
        if poly
            .first()
            .zip(poly.last())
            .is_some_and(|(a, b)| a.y > b.y)
        {
            poly.reverse();
        }
        p.add(egui::Shape::line(poly, egui::Stroke::new(1.15, trace_col)));
    }

    let xh_y = f1_ppm_to_screen_y(view_ppm, f1_rect, crosshair_n_ppm);
    if (f1_rect.top()..=f1_rect.bottom()).contains(&xh_y) {
        p.line_segment(
            [
                egui::pos2(f1_rect.left(), xh_y),
                egui::pos2(f1_rect.right(), xh_y),
            ],
            egui::Stroke::new(0.85, theme.crosshair_unlocked),
        );
    }

    p.text(
        egui::pos2(f1_rect.left() + 4.0, f1_rect.bottom() - 3.0),
        egui::Align2::LEFT_BOTTOM,
        &curve.source_label,
        egui::FontId::monospace(8.5),
        theme.axis_muted,
    );
    p.rect_stroke(
        f1_rect,
        2.0,
        egui::Stroke::new(1.0, theme.marginal_border),
        egui::StrokeKind::Inside,
    );
}

// ============================================================================
// 7. Unified Left `"Spectra"` Workbook Sidebar Renderer
// ============================================================================

fn paint_mini_1d_sparkline(
    painter: &egui::Painter,
    rect: egui::Rect,
    trace: &Trace1d,
    active: bool,
) {
    let Some((lo, hi)) = trace.ppm_bounds() else {
        return;
    };
    let span = (hi - lo).max(1e-6);
    let pts = trace.decimate_for_viewport(lo, hi, 56);
    if pts.len() < 2 {
        return;
    }
    let max_abs = pts
        .iter()
        .fold(0.0_f64, |m, &(_, v)| m.max(v.abs()))
        .max(1e-12);
    let y_base = rect.bottom() - 2.0;
    let h_avail = (rect.height() - 4.0).max(4.0) as f64;
    let [r, g, b, _] = trace.color_rgba;
    let col = if active {
        egui::Color32::from_rgb(r.max(90), g.max(175), b.max(235))
    } else {
        egui::Color32::from_rgba_unmultiplied(r, g, b, 195)
    };

    let mut poly: Vec<egui::Pos2> = Vec::with_capacity(pts.len());
    for &(ppm, val) in &pts {
        let u = ((hi - ppm) / span).clamp(0.0, 1.0) as f32;
        let x = rect.left() + u * rect.width();
        let y = (y_base as f64 - (val / max_abs) * h_avail)
            .clamp(rect.top() as f64 + 1.0, rect.bottom() as f64 - 1.0) as f32;
        poly.push(egui::pos2(x, y));
    }
    if poly
        .first()
        .zip(poly.last())
        .is_some_and(|(a, b)| a.x > b.x)
    {
        poly.reverse();
    }
    painter
        .with_clip_rect(rect)
        .add(egui::Shape::line(poly, egui::Stroke::new(1.05, col)));
}

fn paint_mini_2d_preview(
    painter: &egui::Painter,
    rect: egui::Rect,
    page: &Stored2dPage,
    active: bool,
) {
    let p = painter.with_clip_rect(rect);
    p.rect_filled(rect, 2.0, egui::Color32::from_rgb(12, 17, 26));
    if page.is_homonuclear() {
        p.line_segment(
            [rect.left_bottom(), rect.right_top()],
            egui::Stroke::new(
                0.75,
                egui::Color32::from_rgba_unmultiplied(148, 163, 184, 80),
            ),
        );
    }
    let pos_col = if active {
        egui::Color32::from_rgb(80, 210, 255)
    } else {
        egui::Color32::from_rgb(56, 189, 248)
    };
    let neg_col = egui::Color32::from_rgb(248, 113, 113);
    for &([u, v], is_pos) in &page.mini_dots {
        let pt = egui::pos2(
            rect.left() + u * rect.width(),
            rect.top() + (1.0 - v) * rect.height(),
        );
        p.circle_filled(pt, 1.35, if is_pos { pos_col } else { neg_col });
    }
    p.rect_stroke(
        rect,
        2.0,
        egui::Stroke::new(0.8, egui::Color32::from_rgb(45, 58, 78)),
        egui::StrokeKind::Inside,
    );
}

/// Render the unified Multi-Page `"Spectra"` Workbook Sidebar (`196 px` default Desktop width).
#[allow(clippy::too_many_arguments)]
pub fn render_workbook_sidebar(
    ui: &mut egui::Ui,
    workbook: &mut WorkbookState,
    stack1d: &Stack1dState,
    selection: ActiveWorkbookSelection,
    has_3d_cubes: bool,
    has_pdb: bool,
    show_strips_dock: bool,
    show_mol3d_dock: bool,
) -> Vec<WorkbookSidebarAction> {
    let mut actions = Vec::new();
    let total_pages = stack1d.traces.len() + workbook.pages_2d.len();

    // 1. Sidebar Header: "SPECTRA" + page count badge + quick Open icon
    ui.add_space(2.0);
    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new("SPECTRA")
                .size(10.5)
                .strong()
                .color(egui::Color32::from_rgb(148, 163, 184)),
        );
        egui::Frame::new()
            .fill(egui::Color32::from_rgb(30, 41, 59))
            .corner_radius(3.0)
            .inner_margin(egui::Margin::symmetric(5, 1))
            .show(ui, |ui| {
                ui.label(
                    egui::RichText::new(format!("{total_pages}"))
                        .size(10.0)
                        .monospace()
                        .color(egui::Color32::from_rgb(203, 213, 225)),
                );
            });
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if chrome_icon_button(
                ui,
                WorkbookIcon::PlusOverlay,
                None,
                false,
                "Open spectrum folder or file into workbook",
                22.0,
            )
            .clicked()
            {
                actions.push(WorkbookSidebarAction::TriggerOpenFiles);
            }
        });
    });
    ui.add_space(3.0);

    // Reserve bottom space for the collapsible "Tools & Studios" section
    let studio_active = matches!(selection, ActiveWorkbookSelection::Studio(_));
    let tools_expanded = workbook.show_tools_section || studio_active;
    let bottom_reserve = if tools_expanded { 196.0 } else { 34.0 };
    let list_h = (ui.available_height() - bottom_reserve).max(96.0);

    // 2. Scrollable numbered Page Cards (`1..=N`)
    egui::ScrollArea::vertical()
        .id_salt("wb_spectra_pages_scroll")
        .max_height(list_h)
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing.y = 4.0;
            let card_w = ui.available_width().max(140.0);

            if total_pages == 0 {
                egui::Frame::new()
                    .fill(egui::Color32::from_rgb(20, 26, 36))
                    .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(44, 55, 74)))
                    .corner_radius(5.0)
                    .inner_margin(egui::Margin::same(10))
                    .show(ui, |ui| {
                        ui.label(
                            egui::RichText::new("Drop Bruker, JCAMP, UCSF, or NMRPipe files here")
                                .size(11.0)
                                .color(egui::Color32::from_rgb(148, 163, 184)),
                        );
                    });
            }

            let mut page_num = 1usize;

            // --- 1D Trace Page Cards ---
            let active_is_1d = matches!(selection, ActiveWorkbookSelection::Trace1D(_));
            for (idx, tr) in stack1d.traces.iter().enumerate() {
                let is_active = selection == ActiveWorkbookSelection::Trace1D(idx);
                let is_overlaid = active_is_1d && !is_active && tr.visible;
                let (title, subtitle) = describe_1d_trace_card(tr, idx);

                let card_h = 56.0_f32;
                let (rect, resp) =
                    ui.allocate_exact_size(egui::vec2(card_w, card_h), egui::Sense::click());

                // Sub-hit rectangles for Overlay pill & Close `×` button
                let close_rect = egui::Rect::from_center_size(
                    egui::pos2(rect.right() - 12.0, rect.top() + 12.0),
                    egui::vec2(16.0, 16.0),
                );
                let ov_rect = egui::Rect::from_center_size(
                    egui::pos2(rect.right() - 30.0, rect.top() + 12.0),
                    egui::vec2(18.0, 16.0),
                );
                let ptr = ui.input(|i| i.pointer.interact_pos());
                let hov_pos = resp.hover_pos();
                let card_hovered = resp.hovered();
                let show_ov_btn = active_is_1d && !is_active && stack1d.traces.len() > 1;

                if resp.clicked() {
                    if card_hovered && ptr.is_some_and(|p| close_rect.expand(2.0).contains(p)) {
                        actions.push(WorkbookSidebarAction::Remove1dPage(idx));
                    } else if show_ov_btn && ptr.is_some_and(|p| ov_rect.expand(2.0).contains(p)) {
                        actions.push(WorkbookSidebarAction::Toggle1dOverlay(idx));
                    } else {
                        actions.push(WorkbookSidebarAction::Select1dPage(idx));
                    }
                }

                if ui.is_rect_visible(rect) {
                    let painter = ui.painter();
                    let fill = if is_active {
                        egui::Color32::from_rgb(24, 42, 66)
                    } else if is_overlaid {
                        egui::Color32::from_rgb(20, 38, 48)
                    } else if card_hovered {
                        egui::Color32::from_rgb(24, 31, 44)
                    } else {
                        egui::Color32::from_rgb(18, 24, 34)
                    };
                    let border = if is_active {
                        egui::Color32::from_rgb(86, 182, 248)
                    } else if is_overlaid {
                        egui::Color32::from_rgb(52, 211, 153)
                    } else if card_hovered {
                        egui::Color32::from_rgb(68, 84, 110)
                    } else {
                        egui::Color32::from_rgb(36, 46, 62)
                    };
                    painter.rect_filled(rect, 5.0, fill);
                    painter.rect_stroke(
                        rect,
                        5.0,
                        egui::Stroke::new(1.0, border),
                        egui::StrokeKind::Inside,
                    );
                    if is_active {
                        let bar = egui::Rect::from_min_max(
                            egui::pos2(rect.left(), rect.top() + 4.0),
                            egui::pos2(rect.left() + 3.0, rect.bottom() - 4.0),
                        );
                        painter.rect_filled(bar, 1.5, egui::Color32::from_rgb(86, 182, 248));
                    }

                    // Page number badge
                    let badge_r = egui::Rect::from_min_size(
                        egui::pos2(rect.left() + 7.0, rect.top() + 5.0),
                        egui::vec2(17.0, 15.0),
                    );
                    painter.rect_filled(
                        badge_r,
                        3.0,
                        if is_active {
                            egui::Color32::from_rgb(38, 94, 150)
                        } else {
                            egui::Color32::from_rgb(30, 41, 59)
                        },
                    );
                    painter.text(
                        badge_r.center(),
                        egui::Align2::CENTER_CENTER,
                        format!("{page_num}"),
                        egui::FontId::monospace(9.5),
                        egui::Color32::from_rgb(226, 232, 240),
                    );

                    // Title (clipped safely before action icons)
                    let max_title_x = if show_ov_btn {
                        ov_rect.left() - 4.0
                    } else if card_hovered {
                        close_rect.left() - 4.0
                    } else {
                        rect.right() - 6.0
                    };
                    let title_clip = egui::Rect::from_min_max(
                        egui::pos2(badge_r.right() + 5.0, rect.top() + 3.0),
                        egui::pos2(max_title_x.max(badge_r.right() + 20.0), rect.top() + 21.0),
                    );
                    painter.with_clip_rect(title_clip).text(
                        egui::pos2(title_clip.left(), badge_r.center().y),
                        egui::Align2::LEFT_CENTER,
                        &title,
                        egui::FontId::proportional(11.5),
                        egui::Color32::from_rgb(241, 245, 249),
                    );

                    // Subtitle
                    let sub_clip = egui::Rect::from_min_max(
                        egui::pos2(rect.left() + 7.0, rect.top() + 21.0),
                        egui::pos2(rect.right() - 6.0, rect.top() + 35.0),
                    );
                    painter.with_clip_rect(sub_clip).text(
                        egui::pos2(sub_clip.left(), sub_clip.center().y),
                        egui::Align2::LEFT_CENTER,
                        &subtitle,
                        egui::FontId::monospace(9.2),
                        egui::Color32::from_rgb(148, 163, 184),
                    );

                    // Mini 1D sparkline along bottom of card
                    let spark_r = egui::Rect::from_min_max(
                        egui::pos2(rect.left() + 7.0, rect.top() + 36.0),
                        egui::pos2(rect.right() - 7.0, rect.bottom() - 4.0),
                    );
                    paint_mini_1d_sparkline(painter, spark_r, tr, is_active);

                    // Overlay toggle icon pill
                    if show_ov_btn {
                        let ov_hov = hov_pos.is_some_and(|p| ov_rect.contains(p));
                        painter.rect_filled(
                            ov_rect,
                            3.0,
                            if is_overlaid {
                                egui::Color32::from_rgb(16, 88, 68)
                            } else if ov_hov {
                                egui::Color32::from_rgb(38, 52, 72)
                            } else {
                                egui::Color32::from_rgb(26, 35, 49)
                            },
                        );
                        paint_icon(
                            painter,
                            ov_rect.shrink(2.5),
                            if is_overlaid {
                                WorkbookIcon::EyeVisible
                            } else {
                                WorkbookIcon::PlusOverlay
                            },
                            if is_overlaid {
                                egui::Color32::from_rgb(110, 242, 185)
                            } else {
                                egui::Color32::from_rgb(165, 182, 205)
                            },
                        );
                    }

                    // Close `×` icon on hover
                    if card_hovered {
                        let x_hov = hov_pos.is_some_and(|p| close_rect.contains(p));
                        paint_icon(
                            painter,
                            close_rect.shrink(3.5),
                            WorkbookIcon::CloseX,
                            if x_hov {
                                egui::Color32::from_rgb(248, 113, 113)
                            } else {
                                egui::Color32::from_rgb(148, 163, 184)
                            },
                        );
                    }
                }

                resp.on_hover_text(if show_ov_btn {
                    format!(
                        "{title}\nClick card to solo · Click '+'/eye to superimpose with active 1D"
                    )
                } else {
                    format!("{title}\n{subtitle}")
                });
                page_num += 1;
            }

            // --- 2D Spectrum Page Cards ---
            let active_2d_opt = match selection {
                ActiveWorkbookSelection::Spectrum2D(k) => workbook.pages_2d.get(k),
                _ => None,
            };
            for (idx, pg) in workbook.pages_2d.iter().enumerate() {
                let is_active = selection == ActiveWorkbookSelection::Spectrum2D(idx);
                let compat_with_active = active_2d_opt
                    .is_some_and(|act| !is_active && pg.is_compatible_overlay_with(act));
                let is_overlaid = compat_with_active && pg.overlay_enabled;

                let card_h = 56.0_f32;
                let (rect, resp) =
                    ui.allocate_exact_size(egui::vec2(card_w, card_h), egui::Sense::click());

                let close_rect = egui::Rect::from_center_size(
                    egui::pos2(rect.right() - 12.0, rect.top() + 12.0),
                    egui::vec2(16.0, 16.0),
                );
                let ov_rect = egui::Rect::from_center_size(
                    egui::pos2(rect.right() - 30.0, rect.top() + 12.0),
                    egui::vec2(18.0, 16.0),
                );
                let ptr = ui.input(|i| i.pointer.interact_pos());
                let hov_pos = resp.hover_pos();
                let card_hovered = resp.hovered();

                if resp.clicked() {
                    if card_hovered && ptr.is_some_and(|p| close_rect.expand(2.0).contains(p)) {
                        actions.push(WorkbookSidebarAction::Remove2dPage(idx));
                    } else if compat_with_active
                        && ptr.is_some_and(|p| ov_rect.expand(2.0).contains(p))
                    {
                        actions.push(WorkbookSidebarAction::Toggle2dOverlay(idx));
                    } else {
                        actions.push(WorkbookSidebarAction::Select2dPage(idx));
                    }
                }

                if ui.is_rect_visible(rect) {
                    let painter = ui.painter();
                    let fill = if is_active {
                        egui::Color32::from_rgb(24, 42, 66)
                    } else if is_overlaid {
                        egui::Color32::from_rgb(20, 38, 48)
                    } else if card_hovered {
                        egui::Color32::from_rgb(24, 31, 44)
                    } else {
                        egui::Color32::from_rgb(18, 24, 34)
                    };
                    let border = if is_active {
                        egui::Color32::from_rgb(86, 182, 248)
                    } else if is_overlaid {
                        egui::Color32::from_rgb(52, 211, 153)
                    } else if card_hovered {
                        egui::Color32::from_rgb(68, 84, 110)
                    } else {
                        egui::Color32::from_rgb(36, 46, 62)
                    };
                    painter.rect_filled(rect, 5.0, fill);
                    painter.rect_stroke(
                        rect,
                        5.0,
                        egui::Stroke::new(1.0, border),
                        egui::StrokeKind::Inside,
                    );
                    if is_active {
                        let bar = egui::Rect::from_min_max(
                            egui::pos2(rect.left(), rect.top() + 4.0),
                            egui::pos2(rect.left() + 3.0, rect.bottom() - 4.0),
                        );
                        painter.rect_filled(bar, 1.5, egui::Color32::from_rgb(86, 182, 248));
                    }

                    // Mini 2D preview thumbnail on right-bottom
                    let thumb_r = egui::Rect::from_min_max(
                        egui::pos2(rect.right() - 42.0, rect.bottom() - 25.0),
                        egui::pos2(rect.right() - 6.0, rect.bottom() - 5.0),
                    );
                    paint_mini_2d_preview(painter, thumb_r, pg, is_active);

                    // Page number badge (tinted violet for 3D cubes)
                    let is_3d_cube = pg.is_cube_3d();
                    let badge_r = egui::Rect::from_min_size(
                        egui::pos2(rect.left() + 7.0, rect.top() + 5.0),
                        egui::vec2(if is_3d_cube { 23.0 } else { 17.0 }, 15.0),
                    );
                    painter.rect_filled(
                        badge_r,
                        3.0,
                        if is_3d_cube {
                            if is_active {
                                egui::Color32::from_rgb(109, 40, 217)
                            } else {
                                egui::Color32::from_rgb(76, 29, 149)
                            }
                        } else if is_active {
                            egui::Color32::from_rgb(38, 94, 150)
                        } else {
                            egui::Color32::from_rgb(30, 41, 59)
                        },
                    );
                    painter.text(
                        badge_r.center(),
                        egui::Align2::CENTER_CENTER,
                        if is_3d_cube {
                            "3D".to_string()
                        } else {
                            format!("{page_num}")
                        },
                        egui::FontId::monospace(9.5),
                        egui::Color32::from_rgb(226, 232, 240),
                    );

                    let max_title_x = if compat_with_active {
                        ov_rect.left() - 4.0
                    } else if card_hovered {
                        close_rect.left() - 4.0
                    } else {
                        rect.right() - 6.0
                    };
                    let title_clip = egui::Rect::from_min_max(
                        egui::pos2(badge_r.right() + 5.0, rect.top() + 3.0),
                        egui::pos2(max_title_x.max(badge_r.right() + 20.0), rect.top() + 21.0),
                    );
                    painter.with_clip_rect(title_clip).text(
                        egui::pos2(title_clip.left(), badge_r.center().y),
                        egui::Align2::LEFT_CENTER,
                        &pg.title,
                        egui::FontId::proportional(11.5),
                        egui::Color32::from_rgb(241, 245, 249),
                    );

                    let sub_clip = egui::Rect::from_min_max(
                        egui::pos2(rect.left() + 7.0, rect.top() + 24.0),
                        egui::pos2(thumb_r.left() - 4.0, rect.bottom() - 4.0),
                    );
                    painter.with_clip_rect(sub_clip).text(
                        egui::pos2(sub_clip.left(), sub_clip.center().y),
                        egui::Align2::LEFT_CENTER,
                        &pg.subtitle,
                        egui::FontId::monospace(9.0),
                        egui::Color32::from_rgb(148, 163, 184),
                    );

                    if compat_with_active {
                        let ov_hov = hov_pos.is_some_and(|p| ov_rect.contains(p));
                        painter.rect_filled(
                            ov_rect,
                            3.0,
                            if is_overlaid {
                                egui::Color32::from_rgb(16, 88, 68)
                            } else if ov_hov {
                                egui::Color32::from_rgb(38, 52, 72)
                            } else {
                                egui::Color32::from_rgb(26, 35, 49)
                            },
                        );
                        paint_icon(
                            painter,
                            ov_rect.shrink(2.5),
                            if is_overlaid {
                                WorkbookIcon::EyeVisible
                            } else {
                                WorkbookIcon::PlusOverlay
                            },
                            if is_overlaid {
                                egui::Color32::from_rgb(110, 242, 185)
                            } else {
                                egui::Color32::from_rgb(165, 182, 205)
                            },
                        );
                    }

                    if card_hovered {
                        let x_hov = hov_pos.is_some_and(|p| close_rect.contains(p));
                        paint_icon(
                            painter,
                            close_rect.shrink(3.5),
                            WorkbookIcon::CloseX,
                            if x_hov {
                                egui::Color32::from_rgb(248, 113, 113)
                            } else {
                                egui::Color32::from_rgb(148, 163, 184)
                            },
                        );
                    }
                }

                resp.on_hover_text(if compat_with_active {
                    format!(
                        "{}\nClick card to inspect · Click '+'/eye to overlay GPU contours",
                        pg.title
                    )
                } else {
                    format!("{}\n{}", pg.title, pg.subtitle)
                });
                page_num += 1;
            }
        });

    // 3. Bottom Collapsible "Tools & Studios" Section
    ui.add_space(4.0);
    ui.separator();
    let chev = if tools_expanded {
        WorkbookIcon::ChevronDown
    } else {
        WorkbookIcon::ChevronRight
    };
    if chrome_icon_button(
        ui,
        chev,
        Some("Tools & Studios"),
        studio_active,
        "Expand specialized Chemical Structure, 3D Strips, Dynamics, DOSY, and Metabolomics studios",
        24.0,
    )
    .clicked()
    {
        workbook.show_tools_section = !tools_expanded;
    }

    if workbook.show_tools_section || studio_active {
        ui.add_space(2.0);
        ui.spacing_mut().item_spacing.y = 3.0;
        let items = [
            (
                WorkbookStudioKind::ChemicalStructure,
                WorkbookIcon::StructureChem,
                "Chemical Structure",
                selection == ActiveWorkbookSelection::Studio(WorkbookStudioKind::ChemicalStructure),
            ),
            (
                WorkbookStudioKind::Protein3dStrips,
                WorkbookIcon::Strips3D,
                if has_3d_cubes {
                    "3D Protein Strips (Loaded)"
                } else {
                    "3D Protein Strips"
                },
                show_strips_dock,
            ),
            (
                WorkbookStudioKind::Biomolecular3dPdb,
                WorkbookIcon::Cube3D,
                if has_pdb {
                    "3D Structure PDB (Loaded)"
                } else {
                    "3D Structure PDB"
                },
                show_mol3d_dock,
            ),
            (
                WorkbookStudioKind::RelaxationDynamics,
                WorkbookIcon::RelaxDynamics,
                "Relaxation & CEST",
                selection
                    == ActiveWorkbookSelection::Studio(WorkbookStudioKind::RelaxationDynamics),
            ),
            (
                WorkbookStudioKind::DosyDiffusion,
                WorkbookIcon::DosyGradient,
                "DOSY Diffusion",
                selection == ActiveWorkbookSelection::Studio(WorkbookStudioKind::DosyDiffusion),
            ),
            (
                WorkbookStudioKind::MetabolomicsCohort,
                WorkbookIcon::MetabolomicsCluster,
                "Metabolomics",
                selection
                    == ActiveWorkbookSelection::Studio(WorkbookStudioKind::MetabolomicsCohort),
            ),
        ];
        for (kind, icon, label, active) in items {
            if chrome_icon_button(ui, icon, Some(label), active, label, 24.0).clicked() {
                actions.push(WorkbookSidebarAction::OpenStudio(kind));
            }
        }
    }

    actions
}

// ============================================================================
// 8. Unit Tests (Multi-Page Workbook, Showcase Physics & F2/F1 Pixel Alignment)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_showcase_small_molecule_1h_13c_and_cosy_consistency() {
        let (tr_1h, rois) = build_showcase_small_molecule_1h_trace();
        assert_eq!(trace_1d_nucleus(&tr_1h), "1H");
        assert_eq!(rois.len(), 7);
        assert!(tr_1h.is_demo);
        assert!(tr_1h.raw_spec.is_some());

        let tr_13c = build_showcase_small_molecule_13c_trace();
        assert_eq!(trace_1d_nucleus(&tr_13c), "13C");
        assert!(tr_13c.is_demo);

        let cosy = build_showcase_small_molecule_cosy_2d();
        assert_eq!(cosy.nx(), 512);
        assert_eq!(cosy.ny(), 256);

        // Verify ethyl quartet-triplet off-diagonal cross-peak at (4.34, 1.37) ppm is prominent
        let ix = cosy.axis_x.ppm_to_index(4.34).round() as usize;
        let iy = cosy.axis_y.ppm_to_index(1.37).round() as usize;
        let cross_val = cosy.get(ix, iy);
        let blank_val = cosy.get(
            cosy.axis_x.ppm_to_index(5.80).round() as usize,
            cosy.axis_y.ppm_to_index(3.10).round() as usize,
        );
        assert!(
            cross_val > 50.0 && cross_val > blank_val * 15.0,
            "expected strong (4.34, 1.37) COSY cross-peak, got {cross_val} vs {blank_val}"
        );
    }

    #[test]
    fn test_multi_page_workbook_creation_switching_and_compatible_overlays() {
        let mut wb = WorkbookState::default();
        let cosy = build_showcase_small_molecule_cosy_2d();

        let ax_h = AxisPpm::from_bounds("1H", 64, 10.2, 6.0, 600.0);
        let ax_n = AxisPpm::from_bounds("15N", 32, 105.0, 134.0, 60.8);
        let hsqc = Spectrum2D {
            name: "15N-HSQC".to_string(),
            axis_x: ax_h,
            axis_y: ax_n,
            data: vec![1.0_f32; 64 * 32],
        };

        let mut noesy = cosy.clone();
        noesy.name = "2D NOESY".to_string();

        wb.set_single_2d_page("2D COSY", cosy, false);
        let noesy_idx = wb.add_2d_page("2D NOESY", noesy, true, false);
        let hsqc_idx = wb.add_2d_page("Ubiquitin HSQC", hsqc, true, true);
        assert_eq!(wb.pages_2d.len(), 3);
        assert_eq!(noesy_idx, 1);
        assert_eq!(hsqc_idx, 2);

        // Active = Page 0 (1H-1H COSY): compatible overlay includes NOESY (1H-1H), excludes HSQC (1H-15N)
        let ov_homonuc = wb.build_compatible_overlay_layers(true);
        assert_eq!(ov_homonuc.len(), 1);
        assert_eq!(ov_homonuc[0].0, "2D NOESY");
        assert_eq!(ov_homonuc[0].1.pos_rgba, PAPER_OVERLAY_PALETTES[1].0);

        // Switch active 2D page to Ubiquitin HSQC (1H-15N): incompatible homonuclear overlays excluded
        wb.active_2d_idx = hsqc_idx;
        assert!(wb.pages_2d[wb.active_2d_idx].has_3d_strips);
        let ov_hetero = wb.build_compatible_overlay_layers(false);
        assert!(ov_hetero.is_empty());

        // Verify 2D zoom history push/pop
        wb.two_d.push_zoom([10.2, 6.0, 105.0, 134.0]);
        wb.two_d.push_zoom([9.1, 7.4, 112.0, 126.0]);
        assert_eq!(wb.two_d.pop_zoom(), Some([9.1, 7.4, 112.0, 126.0]));
        assert_eq!(wb.two_d.pop_zoom(), Some([10.2, 6.0, 105.0, 134.0]));
        assert_eq!(wb.two_d.pop_zoom(), None);
    }

    #[test]
    fn test_f2_and_f1_skyline_and_external_1d_pixel_alignment() {
        let ax_x = AxisPpm::from_bounds("1H", 256, 10.0, 6.0, 600.0);
        let ax_y = AxisPpm::from_bounds("15N", 128, 104.0, 132.0, 60.8);
        let mut data = vec![0.0_f32; 256 * 128];
        let target_h = 8.35_f64;
        let target_n = 119.4_f64;
        let ix = ax_x.ppm_to_index(target_h).round() as usize;
        let iy = ax_y.ppm_to_index(target_n).round() as usize;
        let exact_h = ax_x.index_to_ppm(ix as f64);
        let exact_n = ax_y.index_to_ppm(iy as f64);
        data[iy * 256 + ix] = 420.0;

        let spec = Spectrum2D {
            name: "HSQC".to_string(),
            axis_x: ax_x,
            axis_y: ax_y,
            data,
        };

        let view_ppm = [9.6_f32, 6.8, 108.0, 128.0];
        let plot_rect =
            egui::Rect::from_min_size(egui::pos2(110.0, 90.0), egui::vec2(720.0, 480.0));
        let f2_rect = egui::Rect::from_min_max(
            egui::pos2(plot_rect.left(), plot_rect.top() - 68.0),
            egui::pos2(plot_rect.right(), plot_rect.top() - 4.0),
        );
        let f1_rect = egui::Rect::from_min_max(
            egui::pos2(plot_rect.left() - 68.0, plot_rect.top()),
            egui::pos2(plot_rect.left() - 4.0, plot_rect.bottom()),
        );

        let f2_sky = compute_f2_skyline_curve(
            &spec,
            view_ppm[0] as f64,
            view_ppm[1] as f64,
            view_ppm[2] as f64,
            view_ppm[3] as f64,
            512,
        );
        let f1_sky = compute_f1_skyline_curve(
            &spec,
            view_ppm[2] as f64,
            view_ppm[3] as f64,
            view_ppm[0] as f64,
            view_ppm[1] as f64,
            512,
        );

        let best_f2 = f2_sky
            .points
            .iter()
            .copied()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .expect("F2 skyline peak");
        let best_f1 = f1_sky
            .points
            .iter()
            .copied()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .expect("F1 skyline peak");

        assert!((best_f2.0 - exact_h).abs() < 1e-6);
        assert!((best_f1.0 - exact_n).abs() < 1e-6);

        let pt_2d = SyncCrosshair::ppm_to_screen_2d(view_ppm, plot_rect, exact_h, exact_n);
        let sx_f2 = f2_ppm_to_screen_x(view_ppm, f2_rect, best_f2.0);
        let sy_f1 = f1_ppm_to_screen_y(view_ppm, f1_rect, best_f1.0);
        assert!(
            (sx_f2 - pt_2d.x).abs() < 1e-4,
            "Top F2 marginal x ({sx_f2}) must align pixel-for-pixel with 2D peak x ({})",
            pt_2d.x
        );
        assert!(
            (sy_f1 - pt_2d.y).abs() < 1e-4,
            "Left F1 marginal y ({sy_f1}) must align pixel-for-pixel with 2D peak y ({})",
            pt_2d.y
        );

        // External 1D trace auto-matching: 1H trace matches F2 ("1H"), while F1 ("15N") uses Skyline
        let (tr_1h, _) = build_showcase_small_molecule_1h_trace();
        let tr_13c = build_showcase_small_molecule_13c_trace();
        let stack = Stack1dState::with_traces(vec![tr_1h, tr_13c]);
        let mut two_d = TwoDCanvasState::default();
        let xh = SyncCrosshair::new(exact_h, exact_n, None);
        let (c_f2, c_f1) = evaluate_2d_marginals(
            &mut two_d,
            &spec,
            1,
            "1H",
            "15N",
            view_ppm,
            &xh,
            &stack,
            plot_rect.width(),
            plot_rect.height(),
        );
        assert!(c_f2.is_external_1d, "F2 (1H) must use external 1D 1H trace");
        assert!(
            !c_f1.is_external_1d,
            "F1 (15N) must fall back to 2D F1 Skyline when no 15N 1D trace is loaded"
        );
    }

    #[test]
    fn test_headless_sidebar_and_vector_icons_render_zero_panics() {
        let (tr_1h, rois) = build_showcase_small_molecule_1h_trace();
        let tr_13c = build_showcase_small_molecule_13c_trace();
        let mut stack = Stack1dState::with_traces(vec![tr_1h, tr_13c]);
        stack.roi_regions = rois;

        let mut wb = WorkbookState {
            show_tools_section: true,
            ..Default::default()
        };
        wb.set_single_2d_page("2D COSY", build_showcase_small_molecule_cosy_2d(), false);

        let ctx = egui::Context::default();
        let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
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
            for icon in [
                WorkbookIcon::SidebarToggle,
                WorkbookIcon::Folder,
                WorkbookIcon::FileDoc,
                WorkbookIcon::SunPaper,
                WorkbookIcon::MoonDark,
                WorkbookIcon::Trace1D,
                WorkbookIcon::Contour2D,
                WorkbookIcon::Cube3D,
                WorkbookIcon::ZoomBox,
                WorkbookIcon::PanMove,
                WorkbookIcon::CrosshairSlice,
                WorkbookIcon::PeakPick,
                WorkbookIcon::FullSweep,
                WorkbookIcon::UndoBack,
                WorkbookIcon::Projections2D,
                WorkbookIcon::ParamsCard,
                WorkbookIcon::StructureChem,
                WorkbookIcon::Strips3D,
                WorkbookIcon::RelaxDynamics,
                WorkbookIcon::DosyGradient,
                WorkbookIcon::MetabolomicsCluster,
                WorkbookIcon::CommandKey,
                WorkbookIcon::HelpQuestion,
            ] {
                let _ = rail_tool_button(ui, icon, false, "test");
            }
        });
    }
}
