//! Color-Coded VnmrJ-Style Left Vertical Mouse-Mode Vector Icon Rail & Connected Folder Tab Widget
//! (`v0.17.1`, native desktop & `wasm32-unknown-unknown`).
//!
//! Provenance: Original clean-room human-computer interaction and pure-vector `egui::Painter`
//! iconography for the FERRIC NMR workbench. Functional color-groupings (Cyan = Navigation,
//! Amber = Measure & Reference, Emerald = Pick & Quantify, Violet = Processing & Scale,
//! Coral = Curate) and physical file-folder tab geometry are implemented from scratch with zero
//! emojis, zero external assets, and zero third-party source code.

use eframe::egui;

use crate::workbench_mode::{TYPE_BODY_PX, TYPE_CAPTION_PX};

/// Standard slim vertical rail width in logical pixels (`~38 px`).
pub const RAIL_WIDTH_PX: f32 = 38.0;

/// Standard square-ish icon slot size inside the `38 px` vertical rail.
pub const RAIL_SLOT_SIZE: egui::Vec2 = egui::vec2(32.0, 26.0);

/// Vertical spacing between adjacent icon slots inside the rail.
pub const RAIL_SLOT_GAP_PX: f32 = 3.0;

/// Functional color-coding families for NMR vertical mouse-mode rails.
///
/// Every tool and instant action on both the 2D/3D and 1D vertical rails belongs to one of these
/// five semantic families so spectroscopists can locate navigation, measurement, picking,
/// processing, and curation controls peripherally without reading labels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RailColorGroup {
    /// Cyan / Blue — viewport navigation (`Zoom Box`, `Pan`, `Full View`, `Undo Zoom`, `Fit Y`).
    Navigation,
    /// Amber / Gold — measurement and chemical-shift referencing (`Crosshair / Δδ-Δν`, `Reference`).
    MeasureReference,
    /// Emerald / Green — peak picking, integration, and multiplet analysis (`Pick Peak`, `Integrate`, `Multiplet`).
    PickQuantify,
    /// Violet / Purple — spectral processing and vertical/contour scaling (`Phase`, `Contour +/-`, `Water Suppress`, `Gain`).
    ProcessingScale,
    /// Coral / Red — peak curation and destructive edits (`Edit / Delete Peak`).
    Curate,
}

/// Resolved RGBA colors for one [`RailColorGroup`] under either Dark Oscilloscope or Paper White theme.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RailColorPalette {
    /// Saturated functional accent (left indicator bar, active border, hover highlight).
    pub accent: egui::Color32,
    /// Background fill when the tool/toggle is active.
    pub active_bg: egui::Color32,
    /// Background fill on pointer hover.
    pub hover_bg: egui::Color32,
    /// Subtle resting background fill.
    pub idle_bg: egui::Color32,
    /// Resting 1px border outline.
    pub idle_border: egui::Color32,
    /// Vector stroke color when idle (tinted to the functional family).
    pub idle_fg: egui::Color32,
    /// Vector stroke color when active (high contrast).
    pub active_fg: egui::Color32,
    /// Vector stroke color when disabled.
    pub disabled_fg: egui::Color32,
}

impl RailColorGroup {
    /// All five canonical functional color families.
    pub const ALL: [Self; 5] = [
        Self::Navigation,
        Self::MeasureReference,
        Self::PickQuantify,
        Self::ProcessingScale,
        Self::Curate,
    ];

    /// Short human-readable family title.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Navigation => "Navigation",
            Self::MeasureReference => "Measure & Reference",
            Self::PickQuantify => "Pick & Quantify",
            Self::ProcessingScale => "Processing & Scale",
            Self::Curate => "Curate",
        }
    }

    /// Resolve theme-aware functional colors for dark oscilloscope or publication paper-white mode.
    pub const fn palette(self, paper_white: bool) -> RailColorPalette {
        if paper_white {
            match self {
                Self::Navigation => RailColorPalette {
                    accent: egui::Color32::from_rgb(2, 132, 199), // Sky-600
                    active_bg: egui::Color32::from_rgb(2, 132, 199),
                    hover_bg: egui::Color32::from_rgb(224, 242, 254),
                    idle_bg: egui::Color32::from_rgb(241, 247, 252),
                    idle_border: egui::Color32::from_rgb(186, 214, 232),
                    idle_fg: egui::Color32::from_rgb(3, 105, 161),
                    active_fg: egui::Color32::WHITE,
                    disabled_fg: egui::Color32::from_rgb(148, 163, 184),
                },
                Self::MeasureReference => RailColorPalette {
                    accent: egui::Color32::from_rgb(217, 119, 6), // Amber-600
                    active_bg: egui::Color32::from_rgb(217, 119, 6),
                    hover_bg: egui::Color32::from_rgb(254, 243, 199),
                    idle_bg: egui::Color32::from_rgb(253, 249, 237),
                    idle_border: egui::Color32::from_rgb(236, 206, 146),
                    idle_fg: egui::Color32::from_rgb(180, 83, 9),
                    active_fg: egui::Color32::WHITE,
                    disabled_fg: egui::Color32::from_rgb(148, 163, 184),
                },
                Self::PickQuantify => RailColorPalette {
                    accent: egui::Color32::from_rgb(5, 150, 105), // Emerald-600
                    active_bg: egui::Color32::from_rgb(5, 150, 105),
                    hover_bg: egui::Color32::from_rgb(209, 250, 229),
                    idle_bg: egui::Color32::from_rgb(239, 250, 245),
                    idle_border: egui::Color32::from_rgb(156, 219, 193),
                    idle_fg: egui::Color32::from_rgb(4, 120, 87),
                    active_fg: egui::Color32::WHITE,
                    disabled_fg: egui::Color32::from_rgb(148, 163, 184),
                },
                Self::ProcessingScale => RailColorPalette {
                    accent: egui::Color32::from_rgb(124, 58, 237), // Violet-600
                    active_bg: egui::Color32::from_rgb(124, 58, 237),
                    hover_bg: egui::Color32::from_rgb(237, 233, 254),
                    idle_bg: egui::Color32::from_rgb(246, 244, 254),
                    idle_border: egui::Color32::from_rgb(199, 184, 242),
                    idle_fg: egui::Color32::from_rgb(109, 40, 217),
                    active_fg: egui::Color32::WHITE,
                    disabled_fg: egui::Color32::from_rgb(148, 163, 184),
                },
                Self::Curate => RailColorPalette {
                    accent: egui::Color32::from_rgb(225, 29, 72), // Rose-600
                    active_bg: egui::Color32::from_rgb(225, 29, 72),
                    hover_bg: egui::Color32::from_rgb(255, 228, 230),
                    idle_bg: egui::Color32::from_rgb(254, 243, 244),
                    idle_border: egui::Color32::from_rgb(242, 179, 188),
                    idle_fg: egui::Color32::from_rgb(190, 18, 60),
                    active_fg: egui::Color32::WHITE,
                    disabled_fg: egui::Color32::from_rgb(148, 163, 184),
                },
            }
        } else {
            match self {
                Self::Navigation => RailColorPalette {
                    accent: egui::Color32::from_rgb(56, 189, 248), // Cyan/Sky-400
                    active_bg: egui::Color32::from_rgb(12, 74, 110),
                    hover_bg: egui::Color32::from_rgb(20, 46, 68),
                    idle_bg: egui::Color32::from_rgb(14, 24, 38),
                    idle_border: egui::Color32::from_rgb(36, 70, 96),
                    idle_fg: egui::Color32::from_rgb(125, 211, 252),
                    active_fg: egui::Color32::WHITE,
                    disabled_fg: egui::Color32::from_rgb(71, 85, 105),
                },
                Self::MeasureReference => RailColorPalette {
                    accent: egui::Color32::from_rgb(251, 191, 36), // Amber-400
                    active_bg: egui::Color32::from_rgb(120, 53, 15),
                    hover_bg: egui::Color32::from_rgb(54, 38, 18),
                    idle_bg: egui::Color32::from_rgb(26, 24, 18),
                    idle_border: egui::Color32::from_rgb(94, 68, 28),
                    idle_fg: egui::Color32::from_rgb(252, 211, 77),
                    active_fg: egui::Color32::WHITE,
                    disabled_fg: egui::Color32::from_rgb(71, 85, 105),
                },
                Self::PickQuantify => RailColorPalette {
                    accent: egui::Color32::from_rgb(52, 211, 153), // Emerald-400
                    active_bg: egui::Color32::from_rgb(6, 95, 70),
                    hover_bg: egui::Color32::from_rgb(14, 48, 42),
                    idle_bg: egui::Color32::from_rgb(14, 28, 26),
                    idle_border: egui::Color32::from_rgb(28, 84, 68),
                    idle_fg: egui::Color32::from_rgb(110, 231, 183),
                    active_fg: egui::Color32::WHITE,
                    disabled_fg: egui::Color32::from_rgb(71, 85, 105),
                },
                Self::ProcessingScale => RailColorPalette {
                    accent: egui::Color32::from_rgb(192, 132, 252), // Violet-400
                    active_bg: egui::Color32::from_rgb(88, 28, 135),
                    hover_bg: egui::Color32::from_rgb(44, 26, 68),
                    idle_bg: egui::Color32::from_rgb(24, 19, 38),
                    idle_border: egui::Color32::from_rgb(82, 52, 118),
                    idle_fg: egui::Color32::from_rgb(216, 180, 254),
                    active_fg: egui::Color32::WHITE,
                    disabled_fg: egui::Color32::from_rgb(71, 85, 105),
                },
                Self::Curate => RailColorPalette {
                    accent: egui::Color32::from_rgb(251, 113, 133), // Coral/Rose-400
                    active_bg: egui::Color32::from_rgb(136, 19, 55),
                    hover_bg: egui::Color32::from_rgb(62, 24, 38),
                    idle_bg: egui::Color32::from_rgb(32, 18, 26),
                    idle_border: egui::Color32::from_rgb(110, 44, 62),
                    idle_fg: egui::Color32::from_rgb(253, 164, 175),
                    active_fg: egui::Color32::WHITE,
                    disabled_fg: egui::Color32::from_rgb(71, 85, 105),
                },
            }
        }
    }
}

/// Interactive left-drag mouse mode on 2D HSQC / COSY / NOESY and 3D strip spectra.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum MouseMode2D {
    /// Drag a 2D chemical-shift box to zoom (`zm`).
    #[default]
    ZoomBox,
    /// Drag to pan across F2 (`1H`) and F1 (`15N`/`13C`) axes (`mv`).
    Pan,
    /// Interactive crosshair and `Δδ` (ppm) / `Δν` (Hz) caliper ruler (`ms`).
    CrosshairMeasure,
    /// Click on a contour cross-peak to pick a new 2D resonance (`pk`).
    PickPeak,
    /// Click or drag to reposition or delete an existing 2D peak (`ed`).
    EditDeletePeak,
}

impl MouseMode2D {
    /// Ordered modes rendered in the top section of the 2D/3D vertical rail.
    pub const ALL: [Self; 5] = [
        Self::ZoomBox,
        Self::Pan,
        Self::CrosshairMeasure,
        Self::PickPeak,
        Self::EditDeletePeak,
    ];

    /// Functional color group for this 2D mouse mode.
    pub const fn color_group(self) -> RailColorGroup {
        match self {
            Self::ZoomBox | Self::Pan => RailColorGroup::Navigation,
            Self::CrosshairMeasure => RailColorGroup::MeasureReference,
            Self::PickPeak => RailColorGroup::PickQuantify,
            Self::EditDeletePeak => RailColorGroup::Curate,
        }
    }

    /// Short human-readable label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::ZoomBox => "Zoom Box",
            Self::Pan => "Pan",
            Self::CrosshairMeasure => "Crosshair / Δδ-Δν",
            Self::PickPeak => "Pick Peak",
            Self::EditDeletePeak => "Edit / Delete Peak",
        }
    }

    /// Two-letter mnemonic chord code.
    pub const fn short_code(self) -> &'static str {
        match self {
            Self::ZoomBox => "zm",
            Self::Pan => "mv",
            Self::CrosshairMeasure => "ms",
            Self::PickPeak => "pk",
            Self::EditDeletePeak => "ed",
        }
    }

    /// Hover tooltip describing mouse behavior.
    pub const fn tooltip(self) -> &'static str {
        match self {
            Self::ZoomBox => {
                "Zoom Box (zm) [Navigation]: Drag a rectangular region to zoom in 2D/3D chemical shift"
            }
            Self::Pan => {
                "Pan (mv) [Navigation]: Drag freely across F2 and F1 chemical-shift axes"
            }
            Self::CrosshairMeasure => {
                "Crosshair & Measure (ms) [Measure]: Inspect exact δ1/δ2 and drag to measure Δδ (ppm) / Δν (Hz)"
            }
            Self::PickPeak => {
                "Pick Peak (pk) [Pick & Quantify]: Click any cross-peak contour to pick and snap to local extremum"
            }
            Self::EditDeletePeak => {
                "Edit / Delete Peak (ed) [Curate]: Drag a picked peak to refine position or click to remove"
            }
        }
    }
}

/// Instant one-click actions rendered in the lower section of the 2D/3D vertical rail.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InstantAction2D {
    /// Raise contour floor (fewer/cleaner contour levels; `+` / scroll down).
    ContourUp,
    /// Lower contour floor (deeper contour levels; `-` / scroll up).
    ContourDown,
    /// Toggle residual `H2O` solvent band suppression (`ws` / `wu`).
    WaterSuppress,
    /// Reset full 2D sweep bounds (`zf`).
    FullView,
}

impl InstantAction2D {
    /// Ordered instant actions rendered below the divider in the 2D/3D vertical rail.
    pub const ALL: [Self; 4] = [
        Self::ContourUp,
        Self::ContourDown,
        Self::WaterSuppress,
        Self::FullView,
    ];

    /// Functional color group for this instant action.
    pub const fn color_group(self) -> RailColorGroup {
        match self {
            Self::ContourUp | Self::ContourDown | Self::WaterSuppress => {
                RailColorGroup::ProcessingScale
            }
            Self::FullView => RailColorGroup::Navigation,
        }
    }

    /// Short human-readable label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::ContourUp => "Contour +",
            Self::ContourDown => "Contour -",
            Self::WaterSuppress => "Water Suppress",
            Self::FullView => "Full View",
        }
    }

    /// Hover tooltip describing the instant action.
    pub const fn tooltip(self) -> &'static str {
        match self {
            Self::ContourUp => {
                "Raise Contour Level [Processing & Scale]: Step contour threshold up to hide baseline noise"
            }
            Self::ContourDown => {
                "Lower Contour Level [Processing & Scale]: Step contour threshold down to reveal weaker cross-peaks"
            }
            Self::WaterSuppress => {
                "Water Suppress (ws/wu) [Processing & Scale]: Attenuate residual H2O solvent stripe (click again to undo)"
            }
            Self::FullView => {
                "Full View (zf) [Navigation]: Reset F2 and F1 chemical-shift bounds to full sweep width"
            }
        }
    }
}

/// Result event emitted when the user clicks any button on the 2D/3D VnmrJ rail.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RailAction2D {
    /// Switch active 2D/3D mouse interaction mode.
    SetMode(MouseMode2D),
    /// Execute a one-shot 2D/3D viewport or processing action.
    Instant(InstantAction2D),
}

/// Visual configuration passed into [`render_2d_vnmrj_rail`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VnmrJRail2DConfig {
    /// `true` for Publication Paper White canvas styling; `false` for Dark Oscilloscope styling.
    pub paper_white: bool,
    /// `true` when water suppression is currently active on the focused spectrum (highlights toggle).
    pub water_suppressed: bool,
    /// `false` disables click interactions (e.g., when no 2D spectrum is loaded).
    pub enabled: bool,
}

impl VnmrJRail2DConfig {
    /// Convenience constructor for an enabled rail.
    pub const fn new(paper_white: bool, water_suppressed: bool) -> Self {
        Self {
            paper_white,
            water_suppressed,
            enabled: true,
        }
    }
}

/// Layout geometry and clicked action returned by [`render_2d_vnmrj_rail`].
#[derive(Clone, Copy, Debug)]
pub struct VnmrJRail2DOutput {
    /// Bounding rectangle of the entire vertical rail container.
    pub rail_rect: egui::Rect,
    /// Action triggered during this frame, if any.
    pub action: Option<RailAction2D>,
    /// Exact screen rects of the 5 mouse-mode icon slots.
    pub mode_rects: [(MouseMode2D, egui::Rect); 5],
    /// Exact screen rects of the 4 instant-action icon slots.
    pub instant_rects: [(InstantAction2D, egui::Rect); 4],
}

impl VnmrJRail2DOutput {
    /// Look up the rendered screen rect of a given [`MouseMode2D`] slot.
    pub fn mode_rect(&self, mode: MouseMode2D) -> egui::Rect {
        for (m, r) in self.mode_rects {
            if m == mode {
                return r;
            }
        }
        self.rail_rect
    }

    /// Look up the rendered screen rect of a given [`InstantAction2D`] slot.
    pub fn instant_rect(&self, action: InstantAction2D) -> egui::Rect {
        for (a, r) in self.instant_rects {
            if a == action {
                return r;
            }
        }
        self.rail_rect
    }
}

/// Paint one color-coded VnmrJ icon slot background, functional left accent stripe, and border.
///
/// Returns the resolved icon foreground color so callers (both 2D/3D and 1D rails) can draw crisp
/// pure-vector `egui::Painter` geometry centered inside `slot_rect`.
pub fn paint_rail_slot(
    painter: &egui::Painter,
    slot_rect: egui::Rect,
    group: RailColorGroup,
    paper_white: bool,
    active: bool,
    hovered: bool,
    enabled: bool,
) -> egui::Color32 {
    let pal = group.palette(paper_white);
    let bg = if !enabled {
        if paper_white {
            egui::Color32::from_rgb(241, 245, 249)
        } else {
            egui::Color32::from_rgb(15, 20, 30)
        }
    } else if active {
        pal.active_bg
    } else if hovered {
        pal.hover_bg
    } else {
        pal.idle_bg
    };

    let border_col = if !enabled {
        if paper_white {
            egui::Color32::from_rgb(203, 213, 225)
        } else {
            egui::Color32::from_rgb(38, 48, 64)
        }
    } else if active || hovered {
        pal.accent
    } else {
        pal.idle_border
    };

    painter.rect(
        slot_rect,
        4.0,
        bg,
        egui::Stroke::new(if active { 1.5 } else { 1.0 }, border_col),
        egui::StrokeKind::Inside,
    );

    // Crisp 2.5 px left color-family stripe so functional groups read immediately at a glance
    let bar_col = if enabled { pal.accent } else { pal.disabled_fg };
    let bar_rect = egui::Rect::from_min_max(
        egui::pos2(slot_rect.left() + 1.5, slot_rect.top() + 4.0),
        egui::pos2(slot_rect.left() + 4.0, slot_rect.bottom() - 4.0),
    );
    painter.rect_filled(bar_rect, 1.2, bar_col);

    if !enabled {
        pal.disabled_fg
    } else if active {
        pal.active_fg
    } else {
        pal.idle_fg
    }
}

/// Paint the slim outer container backdrop behind a vertical VnmrJ rail.
pub fn paint_rail_backdrop(painter: &egui::Painter, rail_rect: egui::Rect, paper_white: bool) {
    let (bg, border) = if paper_white {
        (
            egui::Color32::from_rgb(248, 250, 252),
            egui::Color32::from_rgb(203, 213, 225),
        )
    } else {
        (
            egui::Color32::from_rgb(12, 17, 27),
            egui::Color32::from_rgb(38, 49, 66),
        )
    };
    painter.rect(
        rail_rect,
        5.0,
        bg,
        egui::Stroke::new(1.0, border),
        egui::StrokeKind::Inside,
    );
}

/// Paint a crisp pure-vector 2D mouse-mode icon inside `rect` (zero emojis, zero tiny fonts).
pub fn paint_2d_mode_icon(
    painter: &egui::Painter,
    rect: egui::Rect,
    mode: MouseMode2D,
    col: egui::Color32,
) {
    let c = rect.center();
    let r = (rect.width().min(rect.height()) * 0.42).max(4.5);
    let sw = egui::Stroke::new(1.4, col);
    let sw_thin = egui::Stroke::new(1.15, col);

    match mode {
        MouseMode2D::ZoomBox => {
            // Dashed-corner zoom selection box + small magnifier circle
            let b = r * 0.85;
            let k = r * 0.38;
            for (sx, sy) in [(-1.0_f32, -1.0_f32), (1.0, -1.0), (-1.0, 1.0)] {
                let corner = egui::pos2(c.x + sx * b, c.y + sy * b);
                painter.line_segment([corner, egui::pos2(corner.x - sx * k, corner.y)], sw_thin);
                painter.line_segment([corner, egui::pos2(corner.x, corner.y - sy * k)], sw_thin);
            }
            let gc = egui::pos2(c.x + r * 0.08, c.y + r * 0.08);
            let gr = r * 0.44;
            painter.circle_stroke(gc, gr, sw);
            painter.line_segment(
                [
                    egui::pos2(gc.x + gr * 0.72, gc.y + gr * 0.72),
                    egui::pos2(c.x + r * 0.90, c.y + r * 0.90),
                ],
                egui::Stroke::new(1.7, col),
            );
        }
        MouseMode2D::Pan => {
            painter.line_segment([egui::pos2(c.x - r, c.y), egui::pos2(c.x + r, c.y)], sw);
            painter.line_segment([egui::pos2(c.x, c.y - r), egui::pos2(c.x, c.y + r)], sw);
            for (dx, dy, px, py) in [
                (-r, 0.0, -r + 2.5, 2.2),
                (r, 0.0, r - 2.5, 2.2),
                (0.0, -r, 2.2, -r + 2.5),
                (0.0, r, 2.2, r - 2.5),
            ] {
                painter.line_segment(
                    [
                        egui::pos2(c.x + dx, c.y + dy),
                        egui::pos2(c.x + px, c.y + py),
                    ],
                    sw_thin,
                );
                let mx = if dx == 0.0 { -px } else { px };
                let my = if dy == 0.0 { -py } else { py };
                painter.line_segment(
                    [
                        egui::pos2(c.x + dx, c.y + dy),
                        egui::pos2(c.x + mx, c.y + my),
                    ],
                    sw_thin,
                );
            }
        }
        MouseMode2D::CrosshairMeasure => {
            // Crosshair reticle + diagonal measurement caliper tick
            painter.circle_stroke(c, r * 0.56, sw);
            painter.circle_filled(c, 1.3, col);
            painter.line_segment(
                [egui::pos2(c.x - r, c.y), egui::pos2(c.x - r * 0.34, c.y)],
                sw_thin,
            );
            painter.line_segment(
                [egui::pos2(c.x + r * 0.34, c.y), egui::pos2(c.x + r, c.y)],
                sw_thin,
            );
            painter.line_segment(
                [egui::pos2(c.x, c.y - r), egui::pos2(c.x, c.y - r * 0.34)],
                sw_thin,
            );
            painter.line_segment(
                [egui::pos2(c.x, c.y + r * 0.34), egui::pos2(c.x, c.y + r)],
                sw_thin,
            );
        }
        MouseMode2D::PickPeak => {
            // Concentric 2D contour rings + peak cross marker
            painter.circle_stroke(c, r * 0.84, sw_thin);
            painter.circle_stroke(c, r * 0.44, sw_thin);
            let k = r * 0.34;
            painter.line_segment([egui::pos2(c.x - k, c.y), egui::pos2(c.x + k, c.y)], sw);
            painter.line_segment([egui::pos2(c.x, c.y - k), egui::pos2(c.x, c.y + k)], sw);
        }
        MouseMode2D::EditDeletePeak => {
            // 2D contour peak with crisp diagonal X curate/delete mark
            painter.circle_stroke(c, r * 0.78, sw_thin);
            let k = r * 0.52;
            painter.line_segment(
                [egui::pos2(c.x - k, c.y - k), egui::pos2(c.x + k, c.y + k)],
                egui::Stroke::new(1.6, col),
            );
            painter.line_segment(
                [egui::pos2(c.x + k, c.y - k), egui::pos2(c.x - k, c.y + k)],
                egui::Stroke::new(1.6, col),
            );
        }
    }
}

/// Paint a crisp pure-vector 2D instant-action icon inside `rect` (zero emojis, zero tiny fonts).
pub fn paint_2d_instant_icon(
    painter: &egui::Painter,
    rect: egui::Rect,
    action: InstantAction2D,
    col: egui::Color32,
) {
    let c = rect.center();
    let r = (rect.width().min(rect.height()) * 0.42).max(4.5);
    let sw = egui::Stroke::new(1.4, col);
    let sw_thin = egui::Stroke::new(1.15, col);

    match action {
        InstantAction2D::ContourUp => {
            // Contour ring on left + pure-vector '+' sign
            let ring_c = egui::pos2(c.x - r * 0.25, c.y + r * 0.12);
            painter.circle_stroke(ring_c, r * 0.56, sw_thin);
            painter.circle_stroke(ring_c, r * 0.25, sw_thin);
            let pc = egui::pos2(c.x + r * 0.52, c.y - r * 0.48);
            let arm = r * 0.36;
            painter.line_segment(
                [egui::pos2(pc.x - arm, pc.y), egui::pos2(pc.x + arm, pc.y)],
                sw,
            );
            painter.line_segment(
                [egui::pos2(pc.x, pc.y - arm), egui::pos2(pc.x, pc.y + arm)],
                sw,
            );
        }
        InstantAction2D::ContourDown => {
            // Contour ring on left + pure-vector '-' sign
            let ring_c = egui::pos2(c.x - r * 0.25, c.y + r * 0.12);
            painter.circle_stroke(ring_c, r * 0.56, sw_thin);
            painter.circle_stroke(ring_c, r * 0.25, sw_thin);
            let mc = egui::pos2(c.x + r * 0.52, c.y - r * 0.48);
            let arm = r * 0.38;
            painter.line_segment(
                [egui::pos2(mc.x - arm, mc.y), egui::pos2(mc.x + arm, mc.y)],
                sw,
            );
        }
        InstantAction2D::WaterSuppress => {
            // Pure-vector water droplet with suppression bar
            let mut drop_pts = Vec::with_capacity(13);
            drop_pts.push(egui::pos2(c.x, c.y - r * 0.88));
            drop_pts.push(egui::pos2(c.x + r * 0.60, c.y + r * 0.08));
            for k in 0..=8 {
                let a = (k as f32 / 8.0) * std::f32::consts::PI;
                drop_pts.push(egui::pos2(
                    c.x + a.cos() * (r * 0.60),
                    c.y + r * 0.22 + a.sin() * (r * 0.60),
                ));
            }
            drop_pts.push(egui::pos2(c.x - r * 0.60, c.y + r * 0.08));
            drop_pts.push(egui::pos2(c.x, c.y - r * 0.88));
            painter.add(egui::Shape::line(drop_pts, sw_thin));
            painter.line_segment(
                [
                    egui::pos2(c.x - r * 0.86, c.y + r * 0.14),
                    egui::pos2(c.x + r * 0.86, c.y + r * 0.14),
                ],
                sw,
            );
        }
        InstantAction2D::FullView => {
            let b = r * 0.82;
            let k = r * 0.42;
            for (sx, sy) in [(-1.0_f32, -1.0_f32), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
                let corner = egui::pos2(c.x + sx * b, c.y + sy * b);
                painter.line_segment([corner, egui::pos2(corner.x - sx * k, corner.y)], sw);
                painter.line_segment([corner, egui::pos2(corner.x, corner.y - sy * k)], sw);
            }
        }
    }
}

/// Render the reusable slim `~38 px` color-coded VnmrJ left vertical mouse-mode rail for 2D/3D
/// spectra (`ZoomBox`, `Pan`, `CrosshairMeasure`, `PickPeak`, `EditDeletePeak` plus instant
/// `ContourUp`, `ContourDown`, `WaterSuppress`, `FullView`).
///
/// Allocates as a single cohesive palette widget (`Sense::click()`) with per-slot hit-testing,
/// hover glow, pointing-hand cursor, and rich per-slot tooltips.
pub fn render_2d_vnmrj_rail(
    ui: &mut egui::Ui,
    active_mode: &mut MouseMode2D,
    cfg: VnmrJRail2DConfig,
) -> VnmrJRail2DOutput {
    let pad_y = 4.0_f32;
    let divider_h = 9.0_f32;
    let n_modes = MouseMode2D::ALL.len();
    let n_instants = InstantAction2D::ALL.len();
    let total_h = pad_y * 2.0
        + (n_modes as f32) * RAIL_SLOT_SIZE.y
        + ((n_modes.saturating_sub(1)) as f32) * RAIL_SLOT_GAP_PX
        + divider_h
        + (n_instants as f32) * RAIL_SLOT_SIZE.y
        + ((n_instants.saturating_sub(1)) as f32) * RAIL_SLOT_GAP_PX;

    let sense = if cfg.enabled {
        egui::Sense::click()
    } else {
        egui::Sense::hover()
    };
    let (rail_rect, resp) = ui.allocate_exact_size(egui::vec2(RAIL_WIDTH_PX, total_h), sense);
    let painter = ui.painter();
    paint_rail_backdrop(painter, rail_rect, cfg.paper_white);

    let slot_x = rail_rect.center().x - RAIL_SLOT_SIZE.x * 0.5;
    let mut y = rail_rect.top() + pad_y;
    let mut mode_rects = [(MouseMode2D::ZoomBox, egui::Rect::NOTHING); 5];
    for (idx, mode) in MouseMode2D::ALL.into_iter().enumerate() {
        let r = egui::Rect::from_min_size(egui::pos2(slot_x, y), RAIL_SLOT_SIZE);
        mode_rects[idx] = (mode, r);
        y += RAIL_SLOT_SIZE.y + RAIL_SLOT_GAP_PX;
    }

    // Subtle divider between sticky mouse modes and one-shot instant actions
    let div_y = y - RAIL_SLOT_GAP_PX + divider_h * 0.5;
    let div_col = if cfg.paper_white {
        egui::Color32::from_rgb(203, 213, 225)
    } else {
        egui::Color32::from_rgb(51, 65, 85)
    };
    painter.line_segment(
        [
            egui::pos2(rail_rect.left() + 6.0, div_y),
            egui::pos2(rail_rect.right() - 6.0, div_y),
        ],
        egui::Stroke::new(1.0, div_col),
    );
    y = div_y + divider_h * 0.5;

    let mut instant_rects = [(InstantAction2D::FullView, egui::Rect::NOTHING); 4];
    for (idx, act) in InstantAction2D::ALL.into_iter().enumerate() {
        let r = egui::Rect::from_min_size(egui::pos2(slot_x, y), RAIL_SLOT_SIZE);
        instant_rects[idx] = (act, r);
        y += RAIL_SLOT_SIZE.y + RAIL_SLOT_GAP_PX;
    }

    let hover_pos = resp.hover_pos();
    let click_pos = if cfg.enabled && resp.clicked() {
        resp.interact_pointer_pos()
            .or_else(|| ui.ctx().pointer_latest_pos())
    } else {
        None
    };

    let mut hovered_tooltip: Option<&'static str> = None;
    let mut triggered_action: Option<RailAction2D> = None;

    for (mode, r) in mode_rects {
        let hovered = cfg.enabled && hover_pos.is_some_and(|p| r.contains(p));
        if hovered {
            hovered_tooltip = Some(mode.tooltip());
        }
        if click_pos.is_some_and(|p| r.contains(p)) {
            *active_mode = mode;
            triggered_action = Some(RailAction2D::SetMode(mode));
        }
        let fg = paint_rail_slot(
            painter,
            r,
            mode.color_group(),
            cfg.paper_white,
            *active_mode == mode,
            hovered,
            cfg.enabled,
        );
        let icon_rect = egui::Rect::from_center_size(
            egui::pos2(r.center().x + 1.5, r.center().y),
            egui::vec2(15.0, 15.0),
        );
        paint_2d_mode_icon(painter, icon_rect, mode, fg);
    }

    for (act, r) in instant_rects {
        let hovered = cfg.enabled && hover_pos.is_some_and(|p| r.contains(p));
        if hovered {
            hovered_tooltip = Some(act.tooltip());
        }
        if click_pos.is_some_and(|p| r.contains(p)) {
            triggered_action = Some(RailAction2D::Instant(act));
        }
        let is_active = act == InstantAction2D::WaterSuppress && cfg.water_suppressed;
        let fg = paint_rail_slot(
            painter,
            r,
            act.color_group(),
            cfg.paper_white,
            is_active,
            hovered,
            cfg.enabled,
        );
        let icon_rect = egui::Rect::from_center_size(
            egui::pos2(r.center().x + 1.5, r.center().y),
            egui::vec2(15.0, 15.0),
        );
        paint_2d_instant_icon(painter, icon_rect, act, fg);
    }

    let resp = if let Some(tip) = hovered_tooltip {
        resp.on_hover_cursor(egui::CursorIcon::PointingHand)
            .on_hover_text(tip)
    } else {
        resp
    };
    let _ = resp;

    VnmrJRail2DOutput {
        rail_rect,
        action: triggered_action,
        mode_rects,
        instant_rects,
    }
}

/// Draw a reusable connected physical folder-style tab (`draw_folder_tab`).
///
/// Visual guarantees:
/// - **Inactive tab**: subtle distinct background fill + crisp `1 px` outline (never looks like
///   plain unstyled static text).
/// - **Hovered tab**: noticeable fill brighten + border accent + pointing-hand cursor.
/// - **Selected / active tab**: matches the content pane below, draws a crisp top accent bar and
///   left/top/right border with an open bottom seam extended `1.5 px` downward so the tab joins
///   seamlessly into the panel frame underneath.
/// - **Typography**: always renders at [`TYPE_BODY_PX`] (`12.5 px >= 11.0 px` floor).
pub fn draw_folder_tab(ui: &mut egui::Ui, label: &str, selected: bool) -> egui::Response {
    let dark_mode = ui.visuals().dark_mode;
    draw_folder_tab_themed(ui, label, selected, !dark_mode)
}

/// Theme-explicit variant of [`draw_folder_tab`] (`paper_white = true` for light/paper surfaces,
/// `false` for dark oscilloscope surfaces).
pub fn draw_folder_tab_themed(
    ui: &mut egui::Ui,
    label: &str,
    selected: bool,
    paper_white: bool,
) -> egui::Response {
    let font_id = egui::FontId::proportional(TYPE_BODY_PX.max(TYPE_CAPTION_PX));
    let text_color = if paper_white {
        if selected {
            egui::Color32::from_rgb(15, 23, 42)
        } else {
            egui::Color32::from_rgb(51, 65, 85)
        }
    } else if selected {
        egui::Color32::from_rgb(248, 250, 252)
    } else {
        egui::Color32::from_rgb(191, 204, 222)
    };

    let galley = ui
        .painter()
        .layout_no_wrap(label.to_string(), font_id, text_color);
    let desired = egui::vec2((galley.size().x + 22.0).max(52.0), 28.0);
    let (rect, resp) = ui.allocate_exact_size(desired, egui::Sense::click());
    let hovered = resp.hovered();

    let (fill, border_col, accent_col) = if paper_white {
        if selected {
            (
                egui::Color32::from_rgb(255, 255, 255),
                egui::Color32::from_rgb(148, 163, 184),
                egui::Color32::from_rgb(2, 132, 199),
            )
        } else if hovered {
            (
                egui::Color32::from_rgb(241, 245, 249),
                egui::Color32::from_rgb(100, 116, 139),
                egui::Color32::from_rgb(2, 132, 199),
            )
        } else {
            (
                egui::Color32::from_rgb(226, 232, 240),
                egui::Color32::from_rgb(168, 181, 198),
                egui::Color32::TRANSPARENT,
            )
        }
    } else if selected {
        (
            egui::Color32::from_rgb(15, 21, 32),
            egui::Color32::from_rgb(68, 86, 114),
            egui::Color32::from_rgb(56, 189, 248),
        )
    } else if hovered {
        (
            egui::Color32::from_rgb(31, 43, 62),
            egui::Color32::from_rgb(94, 114, 144),
            egui::Color32::from_rgb(56, 189, 248),
        )
    } else {
        (
            egui::Color32::from_rgb(22, 31, 46),
            egui::Color32::from_rgb(54, 69, 92),
            egui::Color32::TRANSPARENT,
        )
    };

    let tab_radius = egui::CornerRadius {
        nw: 6,
        ne: 6,
        sw: 0,
        se: 0,
    };
    let painter = ui.painter();

    if selected {
        // Extend 1.5 px downward over the pane seam so active tab seamlessly joins content below
        let joined_rect =
            egui::Rect::from_min_max(rect.min, egui::pos2(rect.max.x, rect.max.y + 1.5));
        painter.rect_filled(joined_rect, tab_radius, fill);

        // Top functional accent bar inside the active tab cap
        let accent_rect = egui::Rect::from_min_max(
            egui::pos2(rect.left() + 4.0, rect.top() + 1.0),
            egui::pos2(rect.right() - 4.0, rect.top() + 3.0),
        );
        painter.rect_filled(accent_rect, 1.0, accent_col);

        // Left, top, and right 1 px outline segments (open bottom edge!)
        let stroke = egui::Stroke::new(1.0, border_col);
        let r_px = 5.0_f32;
        painter.line_segment(
            [
                egui::pos2(rect.left() + 0.5, rect.bottom() + 1.0),
                egui::pos2(rect.left() + 0.5, rect.top() + r_px),
            ],
            stroke,
        );
        painter.line_segment(
            [
                egui::pos2(rect.left() + r_px, rect.top() + 0.5),
                egui::pos2(rect.right() - r_px, rect.top() + 0.5),
            ],
            stroke,
        );
        painter.line_segment(
            [
                egui::pos2(rect.right() - 0.5, rect.top() + r_px),
                egui::pos2(rect.right() - 0.5, rect.bottom() + 1.0),
            ],
            stroke,
        );
    } else {
        // Inactive tab: subtle filled background + full 1 px outline (distinct tab affordance)
        painter.rect(
            rect,
            tab_radius,
            fill,
            egui::Stroke::new(1.0, border_col),
            egui::StrokeKind::Inside,
        );
    }

    let text_pos = egui::pos2(
        rect.center().x - galley.size().x * 0.5,
        rect.center().y - galley.size().y * 0.5 + if selected { 1.0 } else { 0.5 },
    );
    painter.galley(text_pos, galley, text_color);

    resp.on_hover_cursor(egui::CursorIcon::PointingHand)
}
