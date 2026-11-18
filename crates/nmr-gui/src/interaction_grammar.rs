//! Canonical gesture + chord + modifier interaction grammar table and the shared signal-level
//! semantics used by every canvas (`UI-C5`, `UI-C6`, `UI-C7`, `UI-C9`, `UI-C13`, `UI-C14`,
//! `UI-C18`).
//!
//! Provenance: original clean-room interaction design. One table is the single source of truth
//! for the `?` shortcuts overlay and for the hand-off documentation; canvases implement the
//! gestures listed here and the tests in `tests/interaction_grammar_tests.rs` check that the
//! table and the behaviour agree.

/// Views a gesture applies to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GestureView {
    Spectrum1d,
    Spectrum2d,
    CubePlane3d,
    Strips,
    Structure3d,
    Sketcher,
    PageCards,
}

impl GestureView {
    pub const ALL: [Self; 7] = [
        Self::Spectrum1d,
        Self::Spectrum2d,
        Self::CubePlane3d,
        Self::Strips,
        Self::Structure3d,
        Self::Sketcher,
        Self::PageCards,
    ];

    pub const fn label(self) -> &'static str {
        match self {
            Self::Spectrum1d => "1D",
            Self::Spectrum2d => "2D",
            Self::CubePlane3d => "3D plane",
            Self::Strips => "Strips",
            Self::Structure3d => "3D model",
            Self::Sketcher => "Sketcher",
            Self::PageCards => "Page cards",
        }
    }
}

/// Input class of a grammar row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GestureKind {
    Mouse,
    Key,
    Chord,
    Modifier,
}

/// One row of the canonical grammar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GestureBinding {
    pub input: &'static str,
    pub kind: GestureKind,
    pub meaning: &'static str,
    pub views: &'static [GestureView],
}

use GestureView as V;

const SPECTRA: &[GestureView] = &[V::Spectrum1d, V::Spectrum2d, V::CubePlane3d, V::Strips];
const ALL_CANVASES: &[GestureView] = &[
    V::Spectrum1d,
    V::Spectrum2d,
    V::CubePlane3d,
    V::Strips,
    V::Structure3d,
    V::Sketcher,
];
const CONTEXT_MENU_VIEWS: &[GestureView] = &[
    V::Spectrum1d,
    V::Spectrum2d,
    V::CubePlane3d,
    V::Strips,
    V::Structure3d,
    V::Sketcher,
    V::PageCards,
];

/// The canonical interaction grammar (`UI-C18`). The `?` overlay is generated from this table
/// plus [`crate::keymap::SPECTRUM_ACCELERATORS`] and
/// [`crate::keymap::GLOBAL_MODIFIER_SHORTCUTS`].
pub const INTERACTION_GRAMMAR: &[GestureBinding] = &[
    GestureBinding {
        input: "Left-click",
        kind: GestureKind::Mouse,
        meaning: "Select peak / residue / atom under the pointer (pick tool: add peak)",
        views: ALL_CANVASES,
    },
    GestureBinding {
        input: "Cmd/Ctrl+click",
        kind: GestureKind::Mouse,
        meaning: "Add to / remove from the selection",
        views: SPECTRA,
    },
    GestureBinding {
        input: "Left-drag",
        kind: GestureKind::Mouse,
        meaning: "Active tool (pan by default; move selected peak; phase / integrate in 1D tools)",
        views: ALL_CANVASES,
    },
    GestureBinding {
        input: "Shift+drag",
        kind: GestureKind::Mouse,
        meaning: "Zoom box (pushes zoom history)",
        views: SPECTRA,
    },
    GestureBinding {
        input: "Space+drag",
        kind: GestureKind::Mouse,
        meaning: "Pan while held (never toggles anything)",
        views: ALL_CANVASES,
    },
    GestureBinding {
        input: "Right/Middle-drag",
        kind: GestureKind::Mouse,
        meaning: "Pan (1D phase tool: PH1)",
        views: SPECTRA,
    },
    GestureBinding {
        input: "Right-click",
        kind: GestureKind::Mouse,
        meaning: "Context menu",
        views: CONTEXT_MENU_VIEWS,
    },
    GestureBinding {
        input: "Double-click",
        kind: GestureKind::Mouse,
        meaning: "Reset view (pushes zoom history; never adds a peak)",
        views: ALL_CANVASES,
    },
    GestureBinding {
        input: "Scroll",
        kind: GestureKind::Mouse,
        meaning: "Zoom about the pointer (horizontal trackpad scroll pans F2)",
        views: ALL_CANVASES,
    },
    GestureBinding {
        input: "Cmd/Ctrl+Scroll",
        kind: GestureKind::Mouse,
        meaning: "Zoom the indirect axis only (Strips: 13C range)",
        views: &[V::Spectrum2d, V::CubePlane3d, V::Strips],
    },
    GestureBinding {
        input: "Shift+Scroll",
        kind: GestureKind::Mouse,
        meaning: "Step 3D plane / vertical gain",
        views: &[V::Spectrum1d, V::CubePlane3d],
    },
    GestureBinding {
        input: "+ / -",
        kind: GestureKind::Key,
        meaning: "More / less signal (1D gain x1.25; 2D/3D contour floor / 1.25)",
        views: SPECTRA,
    },
    GestureBinding {
        input: "Arrow keys",
        kind: GestureKind::Key,
        meaning: "Pan the view by 10% of the visible span",
        views: SPECTRA,
    },
    GestureBinding {
        input: "Shift+Arrow",
        kind: GestureKind::Key,
        meaning: "Nudge the selected peak (undoable)",
        views: SPECTRA,
    },
    GestureBinding {
        input: "[ ] / PgUp PgDn",
        kind: GestureKind::Key,
        meaning: "Previous / next 3D plane",
        views: &[V::CubePlane3d],
    },
    GestureBinding {
        input: "H P R I M K",
        kind: GestureKind::Key,
        meaning: "1D tools: Pan, Phase, Reference, Integrate, Multiplet, Peak pick (fire only when no chord is pending)",
        views: &[V::Spectrum1d],
    },
    GestureBinding {
        input: "Esc",
        kind: GestureKind::Key,
        meaning: "Cancel pending chord / clear selection / exit tool",
        views: ALL_CANVASES,
    },
    GestureBinding {
        input: "xl",
        kind: GestureKind::Chord,
        meaning: "Toggle crosshair lock",
        views: SPECTRA,
    },
    GestureBinding {
        input: "Cmd/Ctrl+Z / Shift+Cmd/Ctrl+Z",
        kind: GestureKind::Modifier,
        meaning: "Undo / redo the most recent change (user or MCP, one stack)",
        views: ALL_CANVASES,
    },
    GestureBinding {
        input: "Cmd/Ctrl+K",
        kind: GestureKind::Modifier,
        meaning: "Command palette (all commands, modes and developer toggles)",
        views: ALL_CANVASES,
    },
];

/// Geometric step for every signal-level change.
pub const SIGNAL_STEP_RATIO: f64 = 1.25;

/// Direction of a signal-level change (`+` = more, `-` = less).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalDirection {
    More,
    Less,
}

/// New 1D vertical gain after a `+` / `-` press (`+` increases gain).
pub fn gain_after(gain: f64, dir: SignalDirection) -> f64 {
    let g = match dir {
        SignalDirection::More => gain * SIGNAL_STEP_RATIO,
        SignalDirection::Less => gain / SIGNAL_STEP_RATIO,
    };
    g.clamp(1e-3, 1e5)
}

/// New 2D / 3D contour floor after a `+` / `-` press (`+` LOWERS the floor so more signal shows).
pub fn contour_floor_after(floor: f32, dir: SignalDirection) -> f32 {
    let f = match dir {
        SignalDirection::More => floor / SIGNAL_STEP_RATIO as f32,
        SignalDirection::Less => floor * SIGNAL_STEP_RATIO as f32,
    };
    f.max(1e-6)
}

/// Rows of the grammar that apply to `view`.
pub fn bindings_for(view: GestureView) -> impl Iterator<Item = &'static GestureBinding> {
    INTERACTION_GRAMMAR
        .iter()
        .filter(move |b| b.views.contains(&view))
}

/// Render the grammar as a GitHub-flavoured markdown table (used for the hand-off summary and
/// checked by tests so docs and behaviour cannot drift).
pub fn grammar_markdown_table() -> String {
    let mut s = String::from("| Input | Kind | Meaning | Views |\n|---|---|---|---|\n");
    for b in INTERACTION_GRAMMAR {
        let views: Vec<&str> = b.views.iter().map(|v| v.label()).collect();
        s.push_str(&format!(
            "| `{}` | {:?} | {} | {} |\n",
            b.input,
            b.kind,
            b.meaning,
            views.join(", ")
        ));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plus_means_more_signal_everywhere() {
        assert!(gain_after(1.0, SignalDirection::More) > 1.0);
        assert!(contour_floor_after(1.0, SignalDirection::More) < 1.0);
        assert!(gain_after(1.0, SignalDirection::Less) < 1.0);
        assert!(contour_floor_after(1.0, SignalDirection::Less) > 1.0);
    }

    #[test]
    fn right_click_context_menu_covers_every_view() {
        let row = INTERACTION_GRAMMAR
            .iter()
            .find(|b| b.input == "Right-click")
            .expect("row");
        for v in GestureView::ALL {
            assert!(row.views.contains(&v), "{v:?}");
        }
    }
}
