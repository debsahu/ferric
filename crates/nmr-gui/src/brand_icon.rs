//! Deterministic 64x64 RGBA8 window icon builder for `ferric-gui` (Concept A: `FID-F` squircle).
//!
//! Provenance: Original clean-room procedural rasterizer (MIT/Apache-2.0) rendering the
//! `FERRIC` Concept A (`FID-F`) brand mark (`docs/assets/brand/concepts/concept-a-fid-f.svg`)
//! using signed-distance-field primitives without external image decoding dependencies.

use eframe::egui;

/// Edge length (in pixels) of the square window icon.
pub const ICON_WIDTH: u32 = 64;
/// Edge height (in pixels) of the square window icon.
pub const ICON_HEIGHT: u32 = 64;

const SLATE_BG: [u8; 4] = [0x16, 0x20, 0x2A, 0xFF];
const RUST_DARK: [u8; 4] = [0xB7, 0x41, 0x0E, 0xFF];
const RUST_MID: [u8; 4] = [0xD9, 0x5B, 0x26, 0xFF];
const GUIDE_SLATE: [u8; 4] = [0x47, 0x55, 0x69, 0xFF];
const FID_ORANGE: [u8; 4] = [0xF2, 0x8C, 0x38, 0xFF];
const ORIGIN_DOT: [u8; 4] = [0xFE, 0xD7, 0xAA, 0xFF];

/// Signed distance to an axis-aligned rounded rectangle centered at `(cx, cy)`
/// with half-extents `(hx, hy)` and corner radius `r`.
fn sd_rounded_box(px: f32, py: f32, cx: f32, cy: f32, hx: f32, hy: f32, r: f32) -> f32 {
    let qx = (px - cx).abs() - (hx - r);
    let qy = (py - cy).abs() - (hy - r);
    let out_x = qx.max(0.0);
    let out_y = qy.max(0.0);
    (out_x * out_x + out_y * out_y).sqrt() + qx.max(qy).min(0.0) - r
}

/// Signed distance to a line segment `ab` with radius `r`.
fn sd_capsule(px: f32, py: f32, ax: f32, ay: f32, bx: f32, by: f32, r: f32) -> f32 {
    let pax = px - ax;
    let pay = py - ay;
    let bax = bx - ax;
    let bay = by - ay;
    let denom = bax * bax + bay * bay;
    let h = if denom > 1e-6 {
        ((pax * bax + pay * bay) / denom).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let dx = pax - bax * h;
    let dy = pay - bay * h;
    (dx * dx + dy * dy).sqrt() - r
}

/// Alpha-blend `src` over `dst` in sRGB-encoded 8-bit channels with coverage `cov` in `[0, 1]`.
fn blend_over(dst: [u8; 4], src: [u8; 4], cov: f32) -> [u8; 4] {
    let a = (cov.clamp(0.0, 1.0)) * (src[3] as f32 / 255.0);
    if a <= 0.0 {
        return dst;
    }
    if a >= 1.0 && dst[3] == 255 {
        return src;
    }
    let inv = 1.0 - a;
    let dst_a = dst[3] as f32 / 255.0;
    let out_a = a + dst_a * inv;
    if out_a <= 1e-6 {
        return [0, 0, 0, 0];
    }
    let blend_ch = |s: u8, d: u8| -> u8 {
        let v = ((s as f32 * a + d as f32 * dst_a * inv) / out_a).round();
        v.clamp(0.0, 255.0) as u8
    };
    [
        blend_ch(src[0], dst[0]),
        blend_ch(src[1], dst[1]),
        blend_ch(src[2], dst[2]),
        (out_a * 255.0).round().clamp(0.0, 255.0) as u8,
    ]
}

/// Coverage from signed distance using a 1-pixel anti-aliasing ramp in 512-space (`aa = 8.0`).
fn cov_from_sdf(d: f32, aa: f32) -> f32 {
    (0.5 - d / aa).clamp(0.0, 1.0)
}

/// Evaluates the exponentially damped FID sinusoid `y(x)` in 512x512 icon coordinates.
fn fid_curve_y(x: f32) -> f32 {
    let t = x - 168.0;
    let decay = (-t / 91.2).exp();
    let phase = 2.0 * std::f32::consts::PI * t / 98.4;
    248.0 - 120.0 * decay * phase.cos()
}

/// Builds the deterministic 64x64 RGBA8 `FERRIC` window icon (`Concept A: FID-F`).
pub fn build_ferric_window_icon() -> egui::IconData {
    const W: usize = ICON_WIDTH as usize;
    const H: usize = ICON_HEIGHT as usize;
    const SCALE: f32 = 512.0 / W as f32; // 8.0 units in SVG space per pixel
    const AA: f32 = SCALE;

    // Precompute polyline samples for the damped FID sinusoid (x in [168, 414]).
    const FID_SAMPLES: usize = 64;
    let mut fid_pts = [(0.0_f32, 0.0_f32); FID_SAMPLES + 1];
    for (i, pt) in fid_pts.iter_mut().enumerate() {
        let x = 168.0 + (414.0 - 168.0) * (i as f32 / FID_SAMPLES as f32);
        *pt = (x, fid_curve_y(x));
    }

    let mut rgba = vec![0_u8; W * H * 4];

    for py in 0..H {
        let y = (py as f32 + 0.5) * SCALE;
        for px in 0..W {
            let x = (px as f32 + 0.5) * SCALE;

            // 1. Rounded squircle background (#16202A) + rust-orange border (#B7410E / #D95B26).
            let d_outer = sd_rounded_box(x, y, 256.0, 256.0, 224.0, 224.0, 104.0);
            let outer_cov = cov_from_sdf(d_outer, AA);
            if outer_cov <= 0.0 {
                continue;
            }

            let d_inner = sd_rounded_box(x, y, 256.0, 256.0, 204.0, 204.0, 88.0);
            let inner_cov = cov_from_sdf(d_inner, AA);

            // Border band transitions from #B7410E (outer) to #D95B26 (inner highlight).
            let border_color = if y < 256.0 { RUST_MID } else { RUST_DARK };
            let mut pixel = blend_over([0, 0, 0, 0], border_color, outer_cov);
            pixel = blend_over(pixel, SLATE_BG, inner_cov);

            // 2. Faint baseline guide across mid-bar.
            let d_guide = sd_capsule(x, y, 174.0, 248.0, 416.0, 248.0, 5.0);
            pixel = blend_over(pixel, GUIDE_SLATE, cov_from_sdf(d_guide, AA) * outer_cov);

            // 3. Outer F stem, top bar, and mid bar in #B7410E.
            let d_f_outer = sd_capsule(x, y, 152.0, 124.0, 152.0, 388.0, 28.0)
                .min(sd_capsule(x, y, 152.0, 124.0, 376.0, 124.0, 28.0))
                .min(sd_capsule(x, y, 152.0, 248.0, 316.0, 248.0, 24.0));
            pixel = blend_over(pixel, RUST_DARK, cov_from_sdf(d_f_outer, AA));

            // 4. Inner F core in #D95B26.
            let d_f_inner = sd_capsule(x, y, 152.0, 124.0, 152.0, 388.0, 17.0)
                .min(sd_capsule(x, y, 152.0, 124.0, 376.0, 124.0, 17.0))
                .min(sd_capsule(x, y, 152.0, 248.0, 316.0, 248.0, 14.0));
            pixel = blend_over(pixel, RUST_MID, cov_from_sdf(d_f_inner, AA));

            // 5. Exponentially damped FID sinusoid cross-stroke in #F28C38.
            let mut d_fid = f32::INFINITY;
            for seg in fid_pts.windows(2) {
                let d_seg = sd_capsule(x, y, seg[0].0, seg[0].1, seg[1].0, seg[1].1, 14.0);
                if d_seg < d_fid {
                    d_fid = d_seg;
                }
            }
            pixel = blend_over(pixel, FID_ORANGE, cov_from_sdf(d_fid, AA));

            // 6. Origin pulse highlight at (152, 124).
            let dx = x - 152.0;
            let dy = y - 124.0;
            let d_dot = (dx * dx + dy * dy).sqrt() - 16.0;
            pixel = blend_over(pixel, ORIGIN_DOT, cov_from_sdf(d_dot, AA));

            let idx = (py * W + px) * 4;
            rgba[idx..idx + 4].copy_from_slice(&pixel);
        }
    }

    egui::IconData {
        rgba,
        width: ICON_WIDTH,
        height: ICON_HEIGHT,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            "expected #F28C38 FID sinusoid orange pixels"
        );
        assert!(
            pixels.contains(&[0, 0, 0, 0]),
            "expected transparent squircle corner pixels"
        );
    }
}
