//! Deterministic 64x64 RGBA8 window icon loader for `ferric-gui` (Concept A: `FID-F` squircle).
//!
//! Provenance: Original clean-room brand asset (MIT/Apache-2.0) embedded directly from
//! `crates/nmr-gui/assets/ferric-icon-64.rgba` (byte-identical to
//! `docs/assets/brand/ferric-icon-64.rgba`), which is generated deterministically from
//! `scripts/package_release.py::render_icon_scanlines(64)` via `scripts/render_ferric_brand.py`
//! so the desktop window icon, macOS `.icns`, Windows `.ico`, and Linux hicolor PNGs share a
//! single pixel-identical source of truth.

use eframe::egui;

/// Edge length (in pixels) of the square window icon.
pub const ICON_WIDTH: u32 = 64;
/// Edge height (in pixels) of the square window icon.
pub const ICON_HEIGHT: u32 = 64;

/// Single-source-of-truth 64x64 RGBA8 pixels emitted by `scripts/render_ferric_brand.py`.
const ICON_64_RGBA: &[u8] = include_bytes!("../assets/ferric-icon-64.rgba");

/// Builds the deterministic 64x64 RGBA8 `FERRIC` window icon (`Concept A: FID-F`).
pub fn build_ferric_window_icon() -> egui::IconData {
    debug_assert_eq!(ICON_64_RGBA.len(), (ICON_WIDTH * ICON_HEIGHT * 4) as usize);
    egui::IconData {
        rgba: ICON_64_RGBA.to_vec(),
        width: ICON_WIDTH,
        height: ICON_HEIGHT,
    }
}

/// Lazily upload `ferric-icon-64.rgba` (`Concept A: FID-F`) into `egui` texture memory and paint
/// a crisp `size_px × size_px` rounded-square brand mark in the top-left workbook header.
pub fn paint_ferric_brand_logo(ui: &mut egui::Ui, size_px: f32) -> egui::Response {
    let tex_id = egui::Id::unique("ferric_brand_logo_tex_v1");
    let tex_opt: Option<egui::TextureHandle> =
        ui.ctx().data(|d| d.get_temp::<egui::TextureHandle>(tex_id));
    let tex: egui::TextureHandle = if let Some(existing) = tex_opt {
        existing
    } else {
        let img = egui::ColorImage::from_rgba_unmultiplied(
            [ICON_WIDTH as usize, ICON_HEIGHT as usize],
            ICON_64_RGBA,
        );
        let handle =
            ui.ctx()
                .load_texture("ferric_brand_logo_64", img, egui::TextureOptions::LINEAR);
        ui.ctx().data_mut(|d| d.insert_temp(tex_id, handle.clone()));
        handle
    };
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(size_px, size_px), egui::Sense::hover());
    if ui.is_rect_visible(rect) {
        let mut mesh = egui::Mesh::with_texture(tex.id());
        mesh.add_rect_with_uv(
            rect,
            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            egui::Color32::WHITE,
        );
        ui.painter().add(egui::Shape::mesh(mesh));
        ui.painter().rect_stroke(
            rect,
            4.0,
            egui::Stroke::new(0.8, egui::Color32::from_rgb(71, 85, 105)),
            egui::StrokeKind::Inside,
        );
    }
    resp
}

#[cfg(test)]
mod tests {
    use super::*;

    const SLATE_BG: [u8; 4] = [0x16, 0x20, 0x2A, 0xFF];
    const RUST_DARK: [u8; 4] = [0xB7, 0x41, 0x0E, 0xFF];
    const FID_ORANGE: [u8; 4] = [0xFB, 0x92, 0x3C, 0xFF];

    #[test]
    fn ferric_window_icon_is_deterministic_64x64_rgba8() {
        let icon_a = build_ferric_window_icon();
        let icon_b = build_ferric_window_icon();

        assert_eq!(icon_a.width, 64);
        assert_eq!(icon_a.height, 64);
        assert_eq!(icon_a.rgba.len(), 64 * 64 * 4);
        assert_eq!(icon_a.rgba, icon_b.rgba);
    }

    #[test]
    fn ferric_window_icon_contains_slate_bg_and_rust_orange_foreground() {
        let icon = build_ferric_window_icon();
        let pixels: Vec<[u8; 4]> = icon
            .rgba
            .chunks_exact(4)
            .map(|c| [c[0], c[1], c[2], c[3]])
            .collect();

        assert!(
            pixels.contains(&SLATE_BG),
            "expected #16202A deep iron slate background pixels"
        );
        assert!(
            pixels.contains(&RUST_DARK),
            "expected #B7410E rust-orange pixels"
        );
        assert!(
            pixels.contains(&FID_ORANGE),
            "expected #FB923C FID sinusoid orange pixels"
        );
        assert!(
            pixels.contains(&[0, 0, 0, 0]),
            "expected transparent squircle corner pixels"
        );
    }

    #[test]
    fn ferric_in_crate_icon_matches_docs_brand_asset_byte_for_byte() {
        let docs_bytes = include_bytes!("../../../docs/assets/brand/ferric-icon-64.rgba");
        assert_eq!(ICON_64_RGBA.len(), 16_384);
        assert_eq!(
            ICON_64_RGBA, docs_bytes,
            "crates/nmr-gui/assets/ferric-icon-64.rgba must be byte-identical to docs/assets/brand/ferric-icon-64.rgba"
        );
    }
}
