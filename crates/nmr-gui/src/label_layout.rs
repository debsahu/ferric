//! Greedy axis-aligned bounding-box (AABB) label decluttering for 1D peak / integral labels and
//! 2D cross-peak labels (`UI-B4`).
//!
//! Provenance: original clean-room implementation of the classic greedy point-feature label
//! placement heuristic (candidate positions around each anchor, highest priority first, first
//! non-colliding candidate wins, otherwise the label is dropped), as described in
//! Christensen, J., Marks, J. & Shieber, S. (1995) "An empirical study of algorithms for
//! point-feature label placement", *ACM Trans. Graph.* 14(3), 203-232, section 2 (greedy
//! baseline). No third-party code.

use eframe::egui;

/// One label to place.
#[derive(Debug, Clone)]
pub struct LabelCandidate {
    /// Anchor point (e.g. peak apex) in screen coordinates.
    pub anchor: egui::Pos2,
    /// Label box size in pixels.
    pub size: egui::Vec2,
    /// Higher priority labels are placed first (e.g. peak intensity).
    pub priority: f32,
    /// Candidate offsets of the label box's *bottom-centre* relative to the anchor, tried in
    /// order. If empty, [`default_offsets_above`] is used.
    pub offsets: Vec<egui::Vec2>,
}

/// Default candidate offsets: directly above, then staggered higher, then up-left / up-right.
pub fn default_offsets_above(gap: f32, label_h: f32) -> Vec<egui::Vec2> {
    let step = label_h + 2.0;
    vec![
        egui::vec2(0.0, -gap),
        egui::vec2(0.0, -gap - step),
        egui::vec2(0.0, -gap - 2.0 * step),
        egui::vec2(-0.6 * step - 8.0, -gap - step),
        egui::vec2(0.6 * step + 8.0, -gap - step),
        egui::vec2(0.0, -gap - 3.0 * step),
    ]
}

/// Candidate offsets for a dedicated band below a baseline at `baseline_y`: the label's
/// bottom-centre is placed in `[baseline_y + band_top, baseline_y + band_top + n*row]`.
pub fn band_offsets_below(
    anchor: egui::Pos2,
    baseline_y: f32,
    band_top: f32,
    label_h: f32,
    rows: usize,
) -> Vec<egui::Vec2> {
    (0..rows.max(1))
        .map(|r| {
            let bottom = baseline_y + band_top + label_h + r as f32 * (label_h + 2.0);
            egui::vec2(0.0, bottom - anchor.y)
        })
        .collect()
}

/// Box for a label whose bottom-centre sits at `anchor + offset`.
#[inline]
pub fn label_rect(anchor: egui::Pos2, offset: egui::Vec2, size: egui::Vec2) -> egui::Rect {
    let bc = anchor + offset;
    egui::Rect::from_min_max(
        egui::pos2(bc.x - size.x * 0.5, bc.y - size.y),
        egui::pos2(bc.x + size.x * 0.5, bc.y),
    )
}

/// Greedy placement. Returns, for each input label (in input order), the placed rect or `None`
/// if every candidate collided with an already placed label, an obstacle, or left `bounds`.
pub fn place_labels(
    labels: &[LabelCandidate],
    obstacles: &[egui::Rect],
    bounds: egui::Rect,
    padding: f32,
) -> Vec<Option<egui::Rect>> {
    let mut order: Vec<usize> = (0..labels.len()).collect();
    order.sort_by(|&a, &b| {
        labels[b]
            .priority
            .total_cmp(&labels[a].priority)
            .then(a.cmp(&b))
    });
    let mut placed: Vec<egui::Rect> = Vec::with_capacity(labels.len());
    let mut out = vec![None; labels.len()];
    for idx in order {
        let lab = &labels[idx];
        let defaults;
        let offsets: &[egui::Vec2] = if lab.offsets.is_empty() {
            defaults = default_offsets_above(4.0, lab.size.y);
            &defaults
        } else {
            &lab.offsets
        };
        for off in offsets {
            let r = label_rect(lab.anchor, *off, lab.size);
            if !bounds.contains_rect(r) {
                continue;
            }
            let padded = r.expand(padding);
            let hits_label = placed.iter().any(|p| p.intersects(padded));
            let hits_obstacle = obstacles.iter().any(|o| o.intersects(padded));
            if !hits_label && !hits_obstacle {
                placed.push(r);
                out[idx] = Some(r);
                break;
            }
        }
    }
    out
}

/// Count pairwise intersections among placed rects (should be `0` after [`place_labels`]).
pub fn count_collisions(rects: &[egui::Rect]) -> usize {
    let mut n = 0;
    for i in 0..rects.len() {
        for j in (i + 1)..rects.len() {
            if rects[i].intersects(rects[j]) {
                n += 1;
            }
        }
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_labels_never_collide() {
        let bounds = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(400.0, 300.0));
        let labels: Vec<LabelCandidate> = (0..40)
            .map(|i| LabelCandidate {
                anchor: egui::pos2(20.0 + i as f32 * 9.0, 200.0),
                size: egui::vec2(36.0, 13.0),
                priority: (i % 7) as f32,
                offsets: Vec::new(),
            })
            .collect();
        let placed = place_labels(&labels, &[], bounds, 1.0);
        let rects: Vec<_> = placed.iter().flatten().copied().collect();
        assert!(!rects.is_empty());
        assert_eq!(count_collisions(&rects), 0);
    }

    #[test]
    fn band_below_baseline_stays_below() {
        let anchor = egui::pos2(100.0, 50.0);
        let offs = band_offsets_below(anchor, 200.0, 6.0, 12.0, 2);
        for o in offs {
            let r = label_rect(anchor, o, egui::vec2(30.0, 12.0));
            assert!(r.top() >= 200.0 + 6.0 - 0.01);
        }
    }
}
