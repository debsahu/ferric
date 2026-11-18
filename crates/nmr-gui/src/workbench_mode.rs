//! Single top-level workbench mode switcher, 3-step type scale, and headless layout audit
//! (`UI-B1`, `UI-B2`, `UI-B6`, `UI-B7`, `UI-B8`, `UI-B11`, `21.W3:P1-15`).
//!
//! Provenance: original clean-room interaction design for this workbench. The six modes replace
//! the four overlapping navigation systems (context-toolbar studio pills, right-panel studio
//! chips, sidebar "Tools & Studios" list and mode-hijacking menu items) with ONE segmented
//! switcher; everything else lives in the command palette (`Cmd/Ctrl+K`) and right-click
//! context menus. The layout audit measures the rendered `egui` frame (text shapes, clip rects
//! and widget rects) so the per-mode budgets are enforced by tests against real pixels rather
//! than by inspection.

use eframe::egui;

use crate::workflow_bar::HostWorkspaceMode;

/// Maximum number of visible primary interactive controls per mode (excluding the six mode
/// segments themselves and the main canvas).
pub const MAX_PRIMARY_CONTROLS_PER_MODE: usize = 12;

/// Minimum share of the window area given to the spectrum / plot / structure canvas at
/// `1440 x 900`.
pub const MIN_CANVAS_AREA_FRACTION: f32 = 0.65;

/// 3-step type scale (`UI-B8`): caption / axis labels.
pub const TYPE_CAPTION_PX: f32 = 11.0;
/// 3-step type scale: body text and controls.
pub const TYPE_BODY_PX: f32 = 12.5;
/// 3-step type scale: section headers.
pub const TYPE_HEADER_PX: f32 = 15.0;
/// Hard minimum font size anywhere in the workbench.
pub const MIN_FONT_PX: f32 = TYPE_CAPTION_PX;

/// The six canonical top-level workbench modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum WorkbenchMode {
    /// 1D / 2D / 3D spectrum processing (phase, baseline, reference, water, pick).
    #[default]
    Process,
    /// 2D HSQC / 3D cube canvas, sequential strips and sequence strip.
    Assign,
    /// 3D model viewer, secondary-structure concordance ribbon and PAE / AF studio.
    Structure,
    /// Relaxation (R1, R2, hetNOE, model-free) and exchange (CEST / CPMG / CSP).
    Dynamics,
    /// Small-molecule 2D sketcher and DOSY.
    SmallMolecule,
    /// Metabolomics cohort studio.
    Metabolomics,
}

impl WorkbenchMode {
    /// All six modes in switcher order.
    pub const ALL: [Self; 6] = [
        Self::Process,
        Self::Assign,
        Self::Structure,
        Self::Dynamics,
        Self::SmallMolecule,
        Self::Metabolomics,
    ];

    /// Switcher label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Process => "Process",
            Self::Assign => "Assign",
            Self::Structure => "Structure",
            Self::Dynamics => "Dynamics",
            Self::SmallMolecule => "Small molecule",
            Self::Metabolomics => "Metabolomics",
        }
    }

    /// One-line hover description.
    pub const fn description(self) -> &'static str {
        match self {
            Self::Process => {
                "Process 1D / 2D / 3D spectra: phase, baseline, reference, water, pick"
            }
            Self::Assign => "Assign backbone resonances on 2D HSQC / 3D strips",
            Self::Structure => "3D model, secondary-structure concordance and PAE domains",
            Self::Dynamics => "Relaxation, model-free S2, CEST / CPMG / CSP",
            Self::SmallMolecule => "2D chemical sketcher and DOSY diffusion",
            Self::Metabolomics => "Cohort PCA / OPLS-DA / STOCSY",
        }
    }

    /// Legacy host workspace this mode renders through.
    pub const fn host_mode(self, small_molecule_view: SmallMoleculeView) -> HostWorkspaceMode {
        match self {
            Self::Process | Self::Assign | Self::Structure | Self::Dynamics => {
                HostWorkspaceMode::ProteinNmr
            }
            Self::SmallMolecule => match small_molecule_view {
                SmallMoleculeView::Sketcher => HostWorkspaceMode::OctetChem,
                SmallMoleculeView::Dosy => HostWorkspaceMode::Dosy,
            },
            Self::Metabolomics => HostWorkspaceMode::Metabolomics,
        }
    }

    /// Best-effort mapping from a legacy host workspace to a mode.
    pub const fn from_host_mode(mode: HostWorkspaceMode) -> Self {
        match mode {
            HostWorkspaceMode::ProteinNmr => Self::Assign,
            HostWorkspaceMode::OctetChem | HostWorkspaceMode::Dosy => Self::SmallMolecule,
            HostWorkspaceMode::Metabolomics => Self::Metabolomics,
        }
    }

    /// Status-bar cursor readout template for the mode (`UI-B8`, `UI-C11`).
    pub const fn cursor_readout_fields(self) -> &'static str {
        match self {
            Self::Process | Self::Assign => "d1, d2, d3 (ppm), Hz, intensity, S/N",
            Self::Structure => "residue, pLDDT, concordance class",
            Self::Dynamics => "residue, R1, R2, hetNOE, S2",
            Self::SmallMolecule => "atom / bond, formula, d (ppm), log10 D",
            Self::Metabolomics => "bucket d (ppm), VIP, p (BH)",
        }
    }
}

/// Sub-view inside the `Small molecule` mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SmallMoleculeView {
    #[default]
    Sketcher,
    Dosy,
}

impl SmallMoleculeView {
    pub const ALL: [Self; 2] = [Self::Sketcher, Self::Dosy];

    pub const fn label(self) -> &'static str {
        match self {
            Self::Sketcher => "Sketcher",
            Self::Dosy => "DOSY",
        }
    }
}

/// Sub-view inside the `Dynamics` mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DynamicsView {
    #[default]
    Relaxation,
    Exchange,
}

impl DynamicsView {
    pub const ALL: [Self; 2] = [Self::Relaxation, Self::Exchange];

    pub const fn label(self) -> &'static str {
        match self {
            Self::Relaxation => "Relaxation",
            Self::Exchange => "Exchange",
        }
    }
}

/// `egui::Id` salt used by every mode-switcher segment so tests can locate them.
pub const MODE_SWITCHER_ID_SALT: &str = "ferric_mode_switcher";

/// Draw the single segmented top-level mode switcher and also return its exact bounding rectangle
/// (`mode_switcher_rect`) so control-count audits exclude ONLY the 6-mode switcher itself (`UI-B2`).
pub fn draw_mode_switcher_with_rect(
    ui: &mut egui::Ui,
    current: WorkbenchMode,
) -> (Option<WorkbenchMode>, egui::Rect) {
    let mut picked = None;
    let inner = ui.push_id(MODE_SWITCHER_ID_SALT, |ui| {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 2.0;
            for mode in WorkbenchMode::ALL {
                let selected = mode == current;
                let text = egui::RichText::new(mode.label()).size(TYPE_BODY_PX);
                let text = if selected { text.strong() } else { text };
                let resp = ui
                    .add(
                        egui::Button::selectable(selected, text)
                            .min_size(egui::vec2(0.0, 24.0))
                            .corner_radius(4.0),
                    )
                    .on_hover_text(mode.description());
                if resp.clicked() && !selected {
                    picked = Some(mode);
                }
            }
        })
        .response
        .rect
    });
    (picked, inner.inner)
}

/// Draw the single segmented top-level mode switcher. Returns the newly selected mode when the
/// user clicked a different segment this frame.
pub fn draw_mode_switcher(ui: &mut egui::Ui, current: WorkbenchMode) -> Option<WorkbenchMode> {
    draw_mode_switcher_with_rect(ui, current).0
}

/// Draw a compact segmented sub-view selector (e.g. `Sketcher | DOSY`). Returns true when the
/// value changed.
pub fn draw_sub_view_selector<T: Copy + PartialEq>(
    ui: &mut egui::Ui,
    value: &mut T,
    options: &[(T, &str)],
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 2.0;
        for (opt, label) in options {
            let selected = *value == *opt;
            if ui
                .add(egui::Button::selectable(
                    selected,
                    egui::RichText::new(*label).size(TYPE_BODY_PX),
                ))
                .clicked()
                && !selected
            {
                *value = *opt;
                changed = true;
            }
        }
    });
    changed
}

/// Install the 3-step type scale on `ctx` so `.small()` / default body / headings never render
/// below [`MIN_FONT_PX`] (`UI-B8`).
pub fn apply_type_scale(ctx: &egui::Context) {
    ctx.all_styles_mut(|style| {
        use egui::{FontFamily, FontId, TextStyle};
        style.text_styles.insert(
            TextStyle::Small,
            FontId::new(TYPE_CAPTION_PX, FontFamily::Proportional),
        );
        style.text_styles.insert(
            TextStyle::Body,
            FontId::new(TYPE_BODY_PX, FontFamily::Proportional),
        );
        style.text_styles.insert(
            TextStyle::Button,
            FontId::new(TYPE_BODY_PX, FontFamily::Proportional),
        );
        style.text_styles.insert(
            TextStyle::Monospace,
            FontId::new(12.0, FontFamily::Monospace),
        );
        style.text_styles.insert(
            TextStyle::Heading,
            FontId::new(TYPE_HEADER_PX, FontFamily::Proportional),
        );
    });
}

/// Clamp an arbitrary font size to the workbench minimum.
#[inline]
pub fn font_px(size: f32) -> f32 {
    size.max(MIN_FONT_PX)
}

/// One text run measured from a rendered frame.
#[derive(Debug, Clone)]
pub struct AuditedText {
    pub text: String,
    pub rect: egui::Rect,
    pub clip_rect: egui::Rect,
    pub min_font_px: f32,
}

/// Result of auditing a rendered frame's text shapes.
#[derive(Debug, Clone, Default)]
pub struct TextLayoutAudit {
    pub texts: Vec<AuditedText>,
    /// Texts whose visible part is cut by their clip rect (partially visible).
    pub clipped: Vec<AuditedText>,
    /// Pairs of visible texts whose rects overlap by more than the tolerance.
    pub overlapping: Vec<(AuditedText, AuditedText)>,
    /// Smallest font size seen across all visible text runs.
    pub min_font_px: f32,
}

/// Overlap tolerance (`px^2`) below which two glyph boxes are considered touching, not
/// overlapping (glyph boxes include line-gap padding).
pub const TEXT_OVERLAP_TOLERANCE_PX2: f32 = 6.0;
/// Clip tolerance (`px`) — glyph boxes include ascender / descender padding.
pub const TEXT_CLIP_TOLERANCE_PX: f32 = 2.5;

fn collect_text_shapes(shape: &egui::Shape, clip_rect: egui::Rect, out: &mut Vec<AuditedText>) {
    match shape {
        egui::Shape::Text(ts) => {
            let text = ts.galley.job.text.clone();
            if text.trim().is_empty() {
                return;
            }
            let local_rect = if ts.galley.mesh_bounds.is_positive() {
                ts.galley.mesh_bounds
            } else {
                ts.galley.rect
            };
            let rect = local_rect.translate(ts.pos.to_vec2());
            let min_font_px = ts
                .galley
                .job
                .sections
                .iter()
                .filter(|s| {
                    let r = usize::from(s.byte_range.start)..usize::from(s.byte_range.end);
                    ts.galley
                        .job
                        .text
                        .get(r)
                        .is_some_and(|t| !t.trim().is_empty())
                })
                .map(|s| s.format.font_id.size)
                .fold(f32::INFINITY, f32::min);
            out.push(AuditedText {
                text,
                rect,
                clip_rect,
                min_font_px,
            });
        }
        egui::Shape::Vec(v) => {
            for s in v {
                collect_text_shapes(s, clip_rect, out);
            }
        }
        _ => {}
    }
}

/// Audit all text shapes in a frame's output: clipping, pairwise overlap and minimum font size.
///
/// Only texts that are at least partially visible inside `viewport` are considered.
pub fn audit_text_layout(
    shapes: &[egui::epaint::ClippedShape],
    viewport: egui::Rect,
) -> TextLayoutAudit {
    let mut all = Vec::new();
    for cs in shapes {
        collect_text_shapes(&cs.shape, cs.clip_rect, &mut all);
    }
    let visible: Vec<AuditedText> = all
        .into_iter()
        .filter(|t| {
            let vis = t.rect.intersect(t.clip_rect).intersect(viewport);
            vis.is_positive() && vis.width() > 1.0 && vis.height() > 1.0
        })
        .collect();

    let mut audit = TextLayoutAudit {
        min_font_px: f32::INFINITY,
        ..Default::default()
    };
    for t in &visible {
        audit.min_font_px = audit.min_font_px.min(t.min_font_px);
        let clip = t
            .clip_rect
            .intersect(viewport)
            .expand(TEXT_CLIP_TOLERANCE_PX);
        if !clip.contains_rect(t.rect) {
            audit.clipped.push(t.clone());
        }
    }
    for i in 0..visible.len() {
        let a = &visible[i];
        let av = a.rect.intersect(a.clip_rect);
        for b in visible.iter().skip(i + 1) {
            let bv = b.rect.intersect(b.clip_rect);
            let inter = av.intersect(bv);
            if inter.is_positive() && inter.area() > TEXT_OVERLAP_TOLERANCE_PX2 {
                audit.overlapping.push((a.clone(), b.clone()));
            }
        }
    }
    audit.texts = visible;
    audit
}

/// One interactive widget measured from the previous pass.
#[derive(Debug, Clone, Copy)]
pub struct AuditedControl {
    pub id: egui::Id,
    pub rect: egui::Rect,
}

/// Collect visible, enabled, click-sensing widgets from the most recently finished pass,
/// excluding widgets whose centre lies inside any of `exclude` (e.g. the mode switcher and
/// the main canvas) and widgets larger than `max_area_px2` (canvases and list cards).
pub fn primary_controls(
    ctx: &egui::Context,
    exclude: &[egui::Rect],
    max_area_px2: f32,
) -> Vec<AuditedControl> {
    let screen = ctx.content_rect();
    ctx.viewport(|vp| {
        let mut out = Vec::new();
        for (layer, rects) in vp.prev_pass.widgets.layers() {
            if layer.order == egui::Order::Tooltip || layer.order == egui::Order::Debug {
                continue;
            }
            for w in rects {
                if !w.visible || !w.enabled || !w.sense.senses_click() {
                    continue;
                }
                let r = w.interact_rect;
                if !r.is_positive() || r.area() < 16.0 || r.area() > max_area_px2 {
                    continue;
                }
                if !screen.contains(r.center()) {
                    continue;
                }
                if exclude.iter().any(|ex| ex.contains(r.center())) {
                    continue;
                }
                out.push(AuditedControl { id: w.id, rect: r });
            }
        }
        out
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_roundtrip_and_labels() {
        assert_eq!(WorkbenchMode::ALL.len(), 6);
        let labels: Vec<_> = WorkbenchMode::ALL.iter().map(|m| m.label()).collect();
        assert_eq!(
            labels,
            [
                "Process",
                "Assign",
                "Structure",
                "Dynamics",
                "Small molecule",
                "Metabolomics"
            ]
        );
        assert_eq!(
            WorkbenchMode::SmallMolecule.host_mode(SmallMoleculeView::Dosy),
            HostWorkspaceMode::Dosy
        );
        assert_eq!(
            WorkbenchMode::from_host_mode(HostWorkspaceMode::Metabolomics),
            WorkbenchMode::Metabolomics
        );
    }

    #[test]
    fn font_px_clamps_to_minimum() {
        assert_eq!(font_px(8.0), MIN_FONT_PX);
        assert_eq!(font_px(14.0), 14.0);
    }
}
