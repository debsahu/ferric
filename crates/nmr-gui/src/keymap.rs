//! Clean-room finite-state two-key NMR spectrum keyboard accelerator engine and searchable
//! keyboard cheat-sheet modal (`T09.4`).
//!
//! Provenance: Original clean-room finite-state chord input architecture implementing interactive
//! two-key and single-key NMR spectrum accelerators (`docs/specs/12-workbench-ux.md` §3). Standard
//! two-letter NMR spectroscopy mnemonics (`zp`, `zf`, `zi`, `zo`, `zm`, `ct`, `ci`, `cb`, `kr`,
//! `pa`, `pd`, `pc`, `pl`, `lt`, `ph`, `bc`, `pi`, `mp`, `rf`, `ws`, `nt`, `pt`, etc.) follow
//! published peer-reviewed NMR interactive assignment conventions:
//! - Lee, W., Tonelli, M., & Markley, J. L. (*Bioinformatics* **2015**, 31, 1325–1327).
//! - Lee, W., Rahimi, M., Lee, Y., & Chiu, A. (*Bioinformatics* **2021**, 37, 3041–3042).
//!
//! Zero external source code or keytable files consulted. Fully deterministic across native
//! desktop and `wasm32-unknown-unknown` (driven by explicit monotonic millisecond timestamps).

use eframe::egui;

use crate::workflow_bar::WorkbenchAction;

/// Default maximum inter-key interval (`450 ms`) for completing a two-letter NMR spectrum chord.
pub const DEFAULT_CHORD_TIMEOUT_MS: u64 = 450;

/// Geometric multiplier (`1.25x`) applied by `ct` (raise contour floor) and `cb` (lower floor).
pub const CONTOUR_FLOOR_STEP_RATIO: f32 = 1.25;

/// Abstract platform-independent keystroke event fed into [`ChordDispatcher`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyChordInput {
    pub key: ChordKey,
    pub ctrl_or_cmd: bool,
    pub alt: bool,
    pub shift: bool,
}

impl KeyChordInput {
    /// Unmodified lowercase ASCII letter press (e.g., `'p'`, `'a'`, `'v'`, `'t'`).
    #[inline]
    pub const fn plain_char(ch: char) -> Self {
        Self {
            key: ChordKey::Letter(ch.to_ascii_lowercase()),
            ctrl_or_cmd: false,
            alt: false,
            shift: false,
        }
    }

    /// Spacebar press without `Cmd`/`Ctrl`/`Alt`.
    #[inline]
    pub const fn space() -> Self {
        Self {
            key: ChordKey::Space,
            ctrl_or_cmd: false,
            alt: false,
            shift: false,
        }
    }

    /// `?` press (either direct `Questionmark` key or `Shift + /`).
    #[inline]
    pub const fn question_mark() -> Self {
        Self {
            key: ChordKey::QuestionMark,
            ctrl_or_cmd: false,
            alt: false,
            shift: true,
        }
    }

    /// `Esc` press.
    #[inline]
    pub const fn escape() -> Self {
        Self {
            key: ChordKey::Escape,
            ctrl_or_cmd: false,
            alt: false,
            shift: false,
        }
    }

    /// Key press accompanied by `Cmd` or `Ctrl` (e.g., `Cmd/Ctrl+Z`, `Cmd/Ctrl+K`, `Cmd/Ctrl+S`).
    #[inline]
    pub const fn with_cmd_or_ctrl(ch: char, shift: bool) -> Self {
        Self {
            key: ChordKey::Letter(ch.to_ascii_lowercase()),
            ctrl_or_cmd: true,
            alt: false,
            shift,
        }
    }
}

/// Normalized key token recognized by the spectrum chord finite-state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChordKey {
    Letter(char),
    Space,
    QuestionMark,
    Plus,
    Minus,
    OpenBracket,
    CloseBracket,
    PageUp,
    PageDown,
    Escape,
    Other,
}

/// Finite-state machine state of the two-key NMR chord dispatcher.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChordState {
    /// Ready for a single-key accelerator (`Space`, `?`) or the first letter of a two-key chord.
    #[default]
    Idle,
    /// Waiting up to `timeout_ms` for the second letter completing `prefix`.
    Pending { prefix: char, started_at_ms: u64 },
}

/// Context flags indicating whether spectrum single/two-key accelerators must stand down.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct KeyFocusContext {
    /// True when any text input (`TextEdit`, search field, numeric drag editor) holds focus.
    pub text_focus_active: bool,
    /// True when the Octet 2D Chemical Studio canvas/editor is the active workspace.
    pub octet_editor_active: bool,
}

impl KeyFocusContext {
    #[inline]
    pub const fn is_suppressed(self) -> bool {
        self.text_focus_active || self.octet_editor_active
    }
}

/// Deterministic outcome of feeding a single [`KeyChordInput`] into [`ChordDispatcher`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChordOutcome {
    /// First letter of a valid two-key chord accepted; awaiting second letter before deadline.
    PendingPrefix { prefix: char },
    /// Two-key chord or single-key accelerator matched and emitted a [`WorkbenchAction`].
    Dispatched(WorkbenchAction),
    /// Pending half-chord was explicitly cancelled with `Esc`.
    CancelledPending { prefix: char },
    /// Cheat-sheet modal was closed with `Esc` while chord state was `Idle`.
    ClosedCheatSheet,
    /// `Esc` pressed while `Idle` and cheat-sheet already closed.
    IdleEscape,
    /// Pending chord expired before the new key arrived, and the new key is not a valid start/action.
    ExpiredToIdle { expired_prefix: char },
    /// Second key arrived in time, but `(prefix, second)` is not a registered chord; reset to `Idle`.
    InvalidChord { prefix: char, second: char },
    /// Key pressed while `Idle` is not bound to any chord prefix or single-key accelerator.
    UnboundKey,
    /// Modifier shortcut (`Cmd`, `Ctrl`, or `Alt`) passed straight through without consumption.
    PassthroughModifier,
    /// Suppressed because a text input widget or the Octet chemical editor holds focus.
    SuppressedFocus,
}

impl ChordOutcome {
    /// Extract the emitted [`WorkbenchAction`], if any.
    #[inline]
    pub const fn action(self) -> Option<WorkbenchAction> {
        match self {
            Self::Dispatched(a) => Some(a),
            _ => None,
        }
    }
}

/// Static descriptor for a registered spectrum keyboard accelerator displayed in the cheat-sheet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AcceleratorBinding {
    pub keys_label: &'static str,
    pub chord_pair: Option<(char, char)>,
    pub category: &'static str,
    pub title: &'static str,
    pub description: &'static str,
    pub action: WorkbenchAction,
}

/// Canonical table of all two-key and single-key NMR spectrum accelerators (`T09.4`).
pub const SPECTRUM_ACCELERATORS: &[AcceleratorBinding] = &[
    // Zoom, Navigation & Tool Modes
    AcceleratorBinding {
        keys_label: "zp",
        chord_pair: Some(('z', 'p')),
        category: "Zoom & Navigation",
        title: "Zoom Previous",
        description: "Pop previous 2D or 1D zoom window from zoom history",
        action: WorkbenchAction::ZoomPrevious,
    },
    AcceleratorBinding {
        keys_label: "zf",
        chord_pair: Some(('z', 'f')),
        category: "Zoom & Navigation",
        title: "Zoom Full Sweep",
        description: "Reset active 2D or 1D viewport bounds to full spectral sweep width",
        action: WorkbenchAction::ZoomFullSweep,
    },
    AcceleratorBinding {
        keys_label: "ze",
        chord_pair: Some(('z', 'e')),
        category: "Zoom & Navigation",
        title: "Zoom Entire Spectrum",
        description: "Alias for Zoom Full Sweep across active 2D or 1D spectrum",
        action: WorkbenchAction::ZoomFullSweep,
    },
    AcceleratorBinding {
        keys_label: "oz",
        chord_pair: Some(('o', 'z')),
        category: "Zoom & Navigation",
        title: "Zoom Out to Full Sweep",
        description: "Reset 2D or 1D viewport bounds to the full spectral sweep width",
        action: WorkbenchAction::ZoomFullSweep,
    },
    AcceleratorBinding {
        keys_label: "zi",
        chord_pair: Some(('z', 'i')),
        category: "Zoom & Navigation",
        title: "Zoom In Viewport",
        description: "Zoom into the active 2D/1D spectral region by 1.25x",
        action: WorkbenchAction::ZoomIn,
    },
    AcceleratorBinding {
        keys_label: "zo",
        chord_pair: Some(('z', 'o')),
        category: "Zoom & Navigation",
        title: "Zoom Out Viewport",
        description: "Zoom out of the active 2D/1D spectral region by 1.25x",
        action: WorkbenchAction::ZoomOut,
    },
    AcceleratorBinding {
        keys_label: "zm",
        chord_pair: Some(('z', 'm')),
        category: "Zoom & Navigation",
        title: "Zoom Box Tool Mode",
        description: "Switch pointer to rubber-band Zoom Box mode on active 2D or 1D canvas",
        action: WorkbenchAction::SelectZoomBoxMode,
    },
    AcceleratorBinding {
        keys_label: "mv",
        chord_pair: Some(('m', 'v')),
        category: "Zoom & Navigation",
        title: "Move / Pan Tool Mode",
        description: "Switch pointer to drag-pan mode on active 2D or 1D canvas",
        action: WorkbenchAction::SelectPanMoveMode,
    },
    AcceleratorBinding {
        keys_label: "xh",
        chord_pair: Some(('x', 'h')),
        category: "Zoom & Navigation",
        title: "Crosshair & 1D Slice Mode",
        description: "Toggle live orthogonal F2/F1 slice inspection (2D) or caliper crosshair (1D)",
        action: WorkbenchAction::SelectCrosshairSliceMode,
    },
    AcceleratorBinding {
        keys_label: "Space",
        chord_pair: None,
        category: "Zoom & Navigation",
        title: "Toggle Crosshair Lock",
        description:
            "Pin or unpin the synchronized (1H, 15N, 13C) frequency crosshair across views",
        action: WorkbenchAction::ToggleCrosshairLock,
    },
    // Contour Levels, Views & Pages
    AcceleratorBinding {
        keys_label: "ct",
        chord_pair: Some(('c', 't')),
        category: "Spectrum & Contours",
        title: "Contour Levels & Threshold Dialog",
        description: "Open contour base threshold, geometric step, and level count dialog",
        action: WorkbenchAction::OpenContourThresholdPopup,
    },
    AcceleratorBinding {
        keys_label: "vt",
        chord_pair: Some(('v', 't')),
        category: "Spectrum & Contours",
        title: "View / Contour Threshold Dialog",
        description:
            "Open interactive contour base threshold, geometric step, and level count dialog",
        action: WorkbenchAction::OpenContourThresholdPopup,
    },
    AcceleratorBinding {
        keys_label: "ci",
        chord_pair: Some(('c', 'i')),
        category: "Spectrum & Contours",
        title: "Raise Contour Floor / 1D Gain (×1.25)",
        description: "Multiply contour threshold or 1D vertical gain by 1.25x (or press +)",
        action: WorkbenchAction::RaiseContourFloor,
    },
    AcceleratorBinding {
        keys_label: "cu",
        chord_pair: Some(('c', 'u')),
        category: "Spectrum & Contours",
        title: "Contour Up / Raise Floor (×1.25)",
        description: "Raise contour threshold or 1D vertical gain by 1.25x",
        action: WorkbenchAction::RaiseContourFloor,
    },
    AcceleratorBinding {
        keys_label: "cb",
        chord_pair: Some(('c', 'b')),
        category: "Spectrum & Contours",
        title: "Lower Contour Floor / 1D Gain (÷1.25)",
        description: "Divide contour threshold or 1D vertical gain by 1.25x (or press -)",
        action: WorkbenchAction::LowerContourFloor,
    },
    AcceleratorBinding {
        keys_label: "cd",
        chord_pair: Some(('c', 'd')),
        category: "Spectrum & Contours",
        title: "Contour Down / Lower Floor (÷1.25)",
        description: "Lower contour threshold or 1D vertical gain by 1.25x",
        action: WorkbenchAction::LowerContourFloor,
    },
    AcceleratorBinding {
        keys_label: "v1",
        chord_pair: Some(('v', '1')),
        category: "Spectrum & Contours",
        title: "Toggle 1D Marginal Projections",
        description: "Show or hide Top F2 and Left F1 1D marginal projection strips on 2D canvas",
        action: WorkbenchAction::ToggleMarginalProjections,
    },
    AcceleratorBinding {
        keys_label: "v2",
        chord_pair: Some(('v', '2')),
        category: "Spectrum & Contours",
        title: "Cycle 2D Projection / Slice Source",
        description: "Toggle 2D marginal projection strips / slice mode",
        action: WorkbenchAction::ToggleMarginalProjections,
    },
    AcceleratorBinding {
        keys_label: "ol",
        chord_pair: Some(('o', 'l')),
        category: "Spectrum & Contours",
        title: "Overlay Views Dialog",
        description: "Open multi-spectrum 2D contour overlay stack & ppm nudge manager",
        action: WorkbenchAction::OpenOverlayStackManager,
    },
    AcceleratorBinding {
        keys_label: "nt",
        chord_pair: Some(('n', 't')),
        category: "Workbook Navigation",
        title: "Next Spectrum Page",
        description: "Switch forward to the next open 1D or 2D spectrum page",
        action: WorkbenchAction::NextWorkbookPage,
    },
    AcceleratorBinding {
        keys_label: "pt",
        chord_pair: Some(('p', 't')),
        category: "Workbook Navigation",
        title: "Previous Spectrum Page",
        description: "Switch backward to the previous open 1D, 2D, or 3D spectrum page",
        action: WorkbenchAction::PrevWorkbookPage,
    },
    AcceleratorBinding {
        keys_label: "np",
        chord_pair: Some(('n', 'p')),
        category: "3D Cube & Planes",
        title: "Next 3D Cube Plane",
        description: "Step forward one orthogonal plane in active 3D cube (also ] / PageUp / Shift+Scroll)",
        action: WorkbenchAction::NextCubePlane,
    },
    AcceleratorBinding {
        keys_label: "pp",
        chord_pair: Some(('p', 'p')),
        category: "3D Cube & Planes",
        title: "Previous 3D Cube Plane",
        description: "Step backward one orthogonal plane in active 3D cube (also [ / PageDown / Shift+Scroll)",
        action: WorkbenchAction::PrevCubePlane,
    },
    AcceleratorBinding {
        keys_label: "xy",
        chord_pair: Some(('x', 'y')),
        category: "3D Cube & Planes",
        title: "3D Orientation F3×F2 (Step F1)",
        description: "Flip 3D cube to F3×F2 planes stepped along F1 (e.g. 1H×15N at 13C/1H slice)",
        action: WorkbenchAction::CubePlaneF3xF2,
    },
    AcceleratorBinding {
        keys_label: "p1",
        chord_pair: Some(('p', '1')),
        category: "3D Cube & Planes",
        title: "3D Plane Axis F1 (F3×F2)",
        description: "Alias for F3×F2 planes stepped along F1",
        action: WorkbenchAction::CubePlaneF3xF2,
    },
    AcceleratorBinding {
        keys_label: "xz",
        chord_pair: Some(('x', 'z')),
        category: "3D Cube & Planes",
        title: "3D Strip Orientation F3×F1 (Step F2)",
        description: "Flip 3D cube to F3×F1 strip planes stepped along F2 (e.g. 1H×13C/1H at 15N plane)",
        action: WorkbenchAction::CubePlaneF3xF1,
    },
    AcceleratorBinding {
        keys_label: "p2",
        chord_pair: Some(('p', '2')),
        category: "3D Cube & Planes",
        title: "3D Plane Axis F2 (F3×F1 Strips)",
        description: "Alias for F3×F1 strip planes stepped along F2",
        action: WorkbenchAction::CubePlaneF3xF1,
    },
    AcceleratorBinding {
        keys_label: "yz",
        chord_pair: Some(('y', 'z')),
        category: "3D Cube & Planes",
        title: "3D Indirect Orientation F2×F1 (Step F3)",
        description: "Flip 3D cube to F2×F1 indirect planes stepped along F3",
        action: WorkbenchAction::CubePlaneF2xF1,
    },
    AcceleratorBinding {
        keys_label: "p3",
        chord_pair: Some(('p', '3')),
        category: "3D Cube & Planes",
        title: "3D Plane Axis F3 (F2×F1 Indirect)",
        description: "Alias for F2×F1 indirect planes stepped along F3",
        action: WorkbenchAction::CubePlaneF2xF1,
    },
    AcceleratorBinding {
        keys_label: "sk",
        chord_pair: Some(('s', 'k')),
        category: "3D Cube & Planes",
        title: "Toggle 3D Cube 2D Skyline MIP",
        description: "Toggle signed Max-|I| 2D skyline projection across the orthogonal 3D axis",
        action: WorkbenchAction::ToggleCubeSkyline,
    },
    AcceleratorBinding {
        keys_label: "gs",
        chord_pair: Some(('g', 's')),
        category: "3D Cube & Planes",
        title: "Go to 3D Strip at (1H, 15N) Cursor",
        description: "Jump from 2D HSQC cursor or picked HSQC peak directly to matching (1H, 15N) plane/strip in 3D cube",
        action: WorkbenchAction::GotoCubeStripFromHsqc,
    },
    AcceleratorBinding {
        keys_label: "wt",
        chord_pair: Some(('w', 't')),
        category: "Spectrum & Contours",
        title: "Toggle Paper White / Dark Theme",
        description: "Switch between publication Paper White and Dark Oscilloscope canvas",
        action: WorkbenchAction::ToggleCanvasTheme,
    },
    // Peak Picking, Editing & Lists
    AcceleratorBinding {
        keys_label: "kr",
        chord_pair: Some(('k', 'r')),
        category: "Peak Editing",
        title: "Restricted Peak-Pick Tool Mode",
        description: "Switch to interactive Peak-Pick mode (click to add, drag box to pick)",
        action: WorkbenchAction::SelectPickPeaksMode,
    },
    AcceleratorBinding {
        keys_label: "pa",
        chord_pair: Some(('p', 'a')),
        category: "Peak Editing",
        title: "Add Peak at Crosshair",
        description:
            "Insert a new peak at the active cursor or locked crosshair position (2D & 1D)",
        action: WorkbenchAction::AddPeakAtCrosshair,
    },
    AcceleratorBinding {
        keys_label: "pd",
        chord_pair: Some(('p', 'd')),
        category: "Peak Editing",
        title: "Delete Selected Peak(s)",
        description: "Delete currently selected peak in active 2D or 1D view (also Del/Backspace)",
        action: WorkbenchAction::DeleteSelectedPeaks,
    },
    AcceleratorBinding {
        keys_label: "pc",
        chord_pair: Some(('p', 'c')),
        category: "Peak Editing",
        title: "Center / Parabolic Snap Selected Peak",
        description: "Snap selected peak to local parabolic intensity extremum (2D & 1D)",
        action: WorkbenchAction::SnapPeakParabolicCenter,
    },
    AcceleratorBinding {
        keys_label: "pl",
        chord_pair: Some(('p', 'l')),
        category: "Peak Editing",
        title: "Toggle Peak Labels",
        description:
            "Show or hide peak numbers, notes, and residue annotations on canvas (2D & 1D)",
        action: WorkbenchAction::TogglePeakLabels,
    },
    AcceleratorBinding {
        keys_label: "lt",
        chord_pair: Some(('l', 't')),
        category: "Peak Editing",
        title: "Peak List Table",
        description: "Toggle interactive 2D Peak Table or 1D Peaks inspector tab",
        action: WorkbenchAction::TogglePeakListTable,
    },
    AcceleratorBinding {
        keys_label: "se",
        chord_pair: Some(('s', 'e')),
        category: "Peak Editing",
        title: "Pointer / Select Tool",
        description: "Return pointer to selection / zoom mode",
        action: WorkbenchAction::SelectPointerMode,
    },
    AcceleratorBinding {
        keys_label: "st",
        chord_pair: Some(('s', 't')),
        category: "Strips & Assignment",
        title: "Tile Sequential i-1 / i / i+1 Strips",
        description:
            "Tile predecessor (i-1), active root (i), and successor (i+1) 3D match-finder strips",
        action: WorkbenchAction::TileSequentialStrips,
    },
    AcceleratorBinding {
        keys_label: "at",
        chord_pair: Some(('a', 't')),
        category: "Strips & Assignment",
        title: "Assignment / Peak Label Editor",
        description: "Open residue assignment popover (protein) or peak label editor (2D/1D)",
        action: WorkbenchAction::OpenResidueAssignmentPopover,
    },
    // 1D & 2D Processing & Analysis
    AcceleratorBinding {
        keys_label: "ph",
        chord_pair: Some(('p', 'h')),
        category: "1D & 2D Processing",
        title: "Interactive Live Phase Mode",
        description: "Enter live mouse-drag PHC0 / PHC1 phasing mode in 1D Studio",
        action: WorkbenchAction::SelectPhase1dMode,
    },
    AcceleratorBinding {
        keys_label: "ap",
        chord_pair: Some(('a', 'p')),
        category: "1D & 2D Processing",
        title: "Auto-Phase Active 1D Spectrum",
        description: "Run automatic minimum-entropy PH0/PH1 phasing on active 1D trace",
        action: WorkbenchAction::AutoPhaseDirect1D,
    },
    AcceleratorBinding {
        keys_label: "bc",
        chord_pair: Some(('b', 'c')),
        category: "1D & 2D Processing",
        title: "Baseline Correction (Whittaker)",
        description: "Subtract Whittaker / median baseline from active 1D or 2D spectrum",
        action: WorkbenchAction::ApplyMedianBaseline,
    },
    AcceleratorBinding {
        keys_label: "pi",
        chord_pair: Some(('p', 'i')),
        category: "1D & 2D Processing",
        title: "Peak Integration Mode",
        description: "Enter ROI drag-to-integrate mode in 1D Studio",
        action: WorkbenchAction::SelectIntegrate1dMode,
    },
    AcceleratorBinding {
        keys_label: "it",
        chord_pair: Some(('i', 't')),
        category: "1D & 2D Processing",
        title: "Integrate Regions Mode",
        description: "Alias for 1D ROI Peak Integration mode",
        action: WorkbenchAction::SelectIntegrate1dMode,
    },
    AcceleratorBinding {
        keys_label: "mp",
        chord_pair: Some(('m', 'p')),
        category: "1D & 2D Processing",
        title: "Multiplet Analysis Mode",
        description: "Enter drag-to-multiplet & J-tree analysis mode in 1D Studio",
        action: WorkbenchAction::SelectMultiplet1dMode,
    },
    AcceleratorBinding {
        keys_label: "rf",
        chord_pair: Some(('r', 'f')),
        category: "1D & 2D Processing",
        title: "Chemical Shift Reference / Align",
        description: "Enter 1D solvent referencing mode or open 2D ppm alignment",
        action: WorkbenchAction::SelectReferenceOrAlignMode,
    },
    AcceleratorBinding {
        keys_label: "ws",
        chord_pair: Some(('w', 's')),
        category: "1D & 2D Processing",
        title: "Suppress Water Peak",
        description: "Attenuate residual H2O/HDO peak & t1 ridge on active 1D/2D/3D spectrum",
        action: WorkbenchAction::SuppressWaterPeak,
    },
    AcceleratorBinding {
        keys_label: "wu",
        chord_pair: Some(('w', 'u')),
        category: "1D & 2D Processing",
        title: "Undo Water Suppression",
        description: "Restore pre-water-suppression 1D, 2D, or 3D spectrum snapshot",
        action: WorkbenchAction::UndoWaterSuppression,
    },
    AcceleratorBinding {
        keys_label: "op",
        chord_pair: Some(('o', 'p')),
        category: "File & Ingest",
        title: "Open Spectrometer Folder / Files",
        description: "Open spectrometer dataset ingest panel",
        action: WorkbenchAction::OpenIngestMatrix,
    },
    // Help
    AcceleratorBinding {
        keys_label: "?",
        chord_pair: None,
        category: "Help & Reference",
        title: "Keyboard Cheat-Sheet Modal",
        description:
            "Open or filter this searchable reference of two-key chords and modifier shortcuts",
        action: WorkbenchAction::OpenKeyboardCheatSheet,
    },
];

/// Global modifier shortcuts listed alongside two-key chords in the `?` cheat-sheet modal.
pub const GLOBAL_MODIFIER_SHORTCUTS: &[(&str, &str, &str, WorkbenchAction)] = &[
    (
        "Cmd/Ctrl+K",
        "Navigation",
        "Open fuzzy command palette across all workbench actions",
        WorkbenchAction::OpenCommandPalette,
    ),
    (
        "Cmd/Ctrl+Z",
        "Session & Undo",
        "Undo most recent peak, assignment, lock, or contour edit",
        WorkbenchAction::UndoLastCommand,
    ),
    (
        "Cmd/Ctrl+Shift+Z",
        "Session & Undo",
        "Redo most recently undone workbench command",
        WorkbenchAction::RedoLastCommand,
    ),
    (
        "Cmd/Ctrl+S",
        "File & Session",
        "Save lossless NEF v1.1 workbench session",
        WorkbenchAction::SaveSessionNef,
    ),
    (
        "Cmd/Ctrl+Shift+S",
        "File & Session",
        "Save lossless NEF v1.1 session to new path",
        WorkbenchAction::SaveSessionNefAs,
    ),
    (
        "Cmd/Ctrl+O",
        "File & Session",
        "Open spectrometer folder / archive ingest matrix",
        WorkbenchAction::OpenIngestMatrix,
    ),
    (
        "Cmd/Ctrl+E",
        "File & Session",
        "Export assigned chemical shifts (.str NMR-STAR v3)",
        WorkbenchAction::ExportNmrStar,
    ),
    (
        "Cmd/Ctrl+Enter",
        "Assignment",
        "Run automated backbone resonance assignment",
        WorkbenchAction::RunBackboneAssignment,
    ),
];

#[inline]
fn is_valid_chord_prefix(ch: char) -> bool {
    matches!(
        ch,
        'a' | 'b'
            | 'c'
            | 'g'
            | 'i'
            | 'k'
            | 'l'
            | 'm'
            | 'n'
            | 'o'
            | 'p'
            | 'r'
            | 's'
            | 'v'
            | 'w'
            | 'x'
            | 'y'
            | 'z'
    )
}

#[inline]
fn lookup_two_key_chord(prefix: char, second: char) -> Option<WorkbenchAction> {
    match (prefix, second) {
        // Zoom & Viewport
        ('z', 'p') => Some(WorkbenchAction::ZoomPrevious),
        ('z', 'f') | ('z', 'e') | ('o', 'z') => Some(WorkbenchAction::ZoomFullSweep),
        ('z', 'i') => Some(WorkbenchAction::ZoomIn),
        ('z', 'o') => Some(WorkbenchAction::ZoomOut),
        ('z', 'm') => Some(WorkbenchAction::SelectZoomBoxMode),
        ('m', 'v') => Some(WorkbenchAction::SelectPanMoveMode),
        ('x', 'h') => Some(WorkbenchAction::SelectCrosshairSliceMode),
        // Contour Levels, Views, Projections, 3D Planes & Pages
        ('c', 't') | ('v', 't') => Some(WorkbenchAction::OpenContourThresholdPopup),
        ('c', 'i') | ('c', 'u') => Some(WorkbenchAction::RaiseContourFloor),
        ('c', 'b') | ('c', 'd') => Some(WorkbenchAction::LowerContourFloor),
        ('v', '1') | ('v', '2') => Some(WorkbenchAction::ToggleMarginalProjections),
        ('o', 'l') => Some(WorkbenchAction::OpenOverlayStackManager),
        ('n', 't') => Some(WorkbenchAction::NextWorkbookPage),
        ('p', 't') => Some(WorkbenchAction::PrevWorkbookPage),
        ('n', 'p') => Some(WorkbenchAction::NextCubePlane),
        ('p', 'p') => Some(WorkbenchAction::PrevCubePlane),
        ('x', 'y') | ('p', '1') => Some(WorkbenchAction::CubePlaneF3xF2),
        ('x', 'z') | ('p', '2') => Some(WorkbenchAction::CubePlaneF3xF1),
        ('y', 'z') | ('p', '3') => Some(WorkbenchAction::CubePlaneF2xF1),
        ('s', 'k') => Some(WorkbenchAction::ToggleCubeSkyline),
        ('g', 's') => Some(WorkbenchAction::GotoCubeStripFromHsqc),
        ('w', 't') => Some(WorkbenchAction::ToggleCanvasTheme),
        // Peak Picking, Editing, Lists & Assignment
        ('k', 'r') => Some(WorkbenchAction::SelectPickPeaksMode),
        ('p', 'a') => Some(WorkbenchAction::AddPeakAtCrosshair),
        ('p', 'd') => Some(WorkbenchAction::DeleteSelectedPeaks),
        ('p', 'c') => Some(WorkbenchAction::SnapPeakParabolicCenter),
        ('p', 'l') => Some(WorkbenchAction::TogglePeakLabels),
        ('l', 't') => Some(WorkbenchAction::TogglePeakListTable),
        ('s', 'e') => Some(WorkbenchAction::SelectPointerMode),
        ('a', 't') => Some(WorkbenchAction::OpenResidueAssignmentPopover),
        ('s', 't') => Some(WorkbenchAction::TileSequentialStrips),
        // 1D & 2D Processing & Analysis
        ('p', 'h') => Some(WorkbenchAction::SelectPhase1dMode),
        ('a', 'p') => Some(WorkbenchAction::AutoPhaseDirect1D),
        ('b', 'c') => Some(WorkbenchAction::ApplyMedianBaseline),
        ('p', 'i') | ('i', 't') => Some(WorkbenchAction::SelectIntegrate1dMode),
        ('m', 'p') => Some(WorkbenchAction::SelectMultiplet1dMode),
        ('r', 'f') => Some(WorkbenchAction::SelectReferenceOrAlignMode),
        ('w', 's') => Some(WorkbenchAction::SuppressWaterPeak),
        ('w', 'u') => Some(WorkbenchAction::UndoWaterSuppression),
        ('o', 'p') => Some(WorkbenchAction::OpenIngestMatrix),
        _ => None,
    }
}

/// Clean-room finite-state two-key NMR chord dispatcher and searchable `?` cheat-sheet state.
#[derive(Debug, Clone)]
pub struct ChordDispatcher {
    state: ChordState,
    timeout_ms: u64,
    pub cheat_sheet_open: bool,
    pub cheat_sheet_query: String,
    cheat_sheet_filter_id: Option<egui::Id>,
}

impl Default for ChordDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl ChordDispatcher {
    /// Construct a dispatcher with the standard `450 ms` chord timeout.
    pub fn new() -> Self {
        Self::with_timeout_ms(DEFAULT_CHORD_TIMEOUT_MS)
    }

    /// Construct a dispatcher with a custom chord timeout in milliseconds (`>= 50 ms`).
    pub fn with_timeout_ms(timeout_ms: u64) -> Self {
        Self {
            state: ChordState::Idle,
            timeout_ms: timeout_ms.max(50),
            cheat_sheet_open: false,
            cheat_sheet_query: String::new(),
            cheat_sheet_filter_id: None,
        }
    }

    /// Configured chord timeout in milliseconds.
    #[inline]
    pub const fn timeout_ms(&self) -> u64 {
        self.timeout_ms
    }

    /// Update the chord timeout window in milliseconds (`>= 50 ms`).
    pub fn set_timeout_ms(&mut self, timeout_ms: u64) {
        self.timeout_ms = timeout_ms.max(50);
    }

    /// Current raw state of the chord finite-state machine.
    #[inline]
    pub const fn state(&self) -> ChordState {
        self.state
    }

    /// Reset any pending half-chord back to [`ChordState::Idle`].
    pub fn reset(&mut self) {
        self.state = ChordState::Idle;
    }

    /// Expire any pending half-chord whose elapsed age exceeds `self.timeout_ms`.
    ///
    /// Returns `Some(prefix)` if a stale pending chord was expired on this tick.
    pub fn tick_timeout(&mut self, now_ms: u64) -> Option<char> {
        if let ChordState::Pending {
            prefix,
            started_at_ms,
        } = self.state
        {
            if now_ms.saturating_sub(started_at_ms) > self.timeout_ms {
                self.state = ChordState::Idle;
                return Some(prefix);
            }
        }
        None
    }

    /// Status-bar pending-chord indicator (e.g. `Some("Chord: p_")`) when a prefix is active
    /// within the `timeout_ms` window, or `None` when idle or expired.
    pub fn status_indicator(&mut self, now_ms: u64) -> Option<String> {
        self.tick_timeout(now_ms);
        match self.state {
            ChordState::Idle => None,
            ChordState::Pending { prefix, .. } => Some(format!("Chord: {prefix}_")),
        }
    }

    /// Non-mutating read of the status-bar pending-chord indicator at `now_ms`.
    pub fn peek_status_indicator(&self, now_ms: u64) -> Option<String> {
        match self.state {
            ChordState::Idle => None,
            ChordState::Pending {
                prefix,
                started_at_ms,
            } => {
                if now_ms.saturating_sub(started_at_ms) <= self.timeout_ms {
                    Some(format!("Chord: {prefix}_"))
                } else {
                    None
                }
            }
        }
    }

    /// Open the searchable `?` keyboard cheat-sheet modal and clear any pending chord.
    pub fn open_cheat_sheet(&mut self) {
        self.state = ChordState::Idle;
        self.cheat_sheet_open = true;
    }

    /// Close the searchable `?` keyboard cheat-sheet modal.
    pub fn close_cheat_sheet(&mut self) {
        self.cheat_sheet_open = false;
    }

    /// Feed a single normalized [`KeyChordInput`] at monotonic timestamp `now_ms`.
    pub fn feed_key(
        &mut self,
        input: KeyChordInput,
        now_ms: u64,
        focus: KeyFocusContext,
    ) -> ChordOutcome {
        // 1. Automatic focus suppression whenever text inputs or Octet editor hold focus
        if focus.is_suppressed() {
            self.state = ChordState::Idle;
            return ChordOutcome::SuppressedFocus;
        }

        // 2. Modifier passthrough: Cmd/Ctrl/Alt shortcuts pass straight through untouched
        if input.ctrl_or_cmd || input.alt {
            self.state = ChordState::Idle;
            return ChordOutcome::PassthroughModifier;
        }

        // 3. Check whether an earlier pending prefix has already timed out (`> timeout_ms`)
        let expired_prefix = self.tick_timeout(now_ms);

        // 4. Handle `Esc` cancellation immediately
        if input.key == ChordKey::Escape {
            if let ChordState::Pending { prefix, .. } = self.state {
                self.state = ChordState::Idle;
                return ChordOutcome::CancelledPending { prefix };
            }
            if self.cheat_sheet_open {
                self.cheat_sheet_open = false;
                return ChordOutcome::ClosedCheatSheet;
            }
            return ChordOutcome::IdleEscape;
        }

        // 5. State transition
        match self.state {
            ChordState::Pending { prefix, .. } => match input.key {
                ChordKey::Letter(ch) if !input.shift => {
                    let second = ch.to_ascii_lowercase();
                    self.state = ChordState::Idle;
                    if let Some(action) = lookup_two_key_chord(prefix, second) {
                        ChordOutcome::Dispatched(action)
                    } else {
                        ChordOutcome::InvalidChord { prefix, second }
                    }
                }
                ChordKey::Letter(ch) => {
                    self.state = ChordState::Idle;
                    ChordOutcome::InvalidChord { prefix, second: ch }
                }
                ChordKey::Space => {
                    self.state = ChordState::Idle;
                    ChordOutcome::InvalidChord {
                        prefix,
                        second: ' ',
                    }
                }
                ChordKey::QuestionMark => {
                    self.state = ChordState::Idle;
                    ChordOutcome::InvalidChord {
                        prefix,
                        second: '?',
                    }
                }
                ChordKey::Plus | ChordKey::Minus => {
                    self.state = ChordState::Idle;
                    ChordOutcome::InvalidChord {
                        prefix,
                        second: if input.key == ChordKey::Plus {
                            '+'
                        } else {
                            '-'
                        },
                    }
                }
                ChordKey::Other
                | ChordKey::OpenBracket
                | ChordKey::CloseBracket
                | ChordKey::PageUp
                | ChordKey::PageDown
                | ChordKey::Escape => {
                    self.state = ChordState::Idle;
                    ChordOutcome::InvalidChord {
                        prefix,
                        second: '\0',
                    }
                }
            },
            ChordState::Idle => match input.key {
                ChordKey::Letter(ch) if !input.shift => {
                    let lower = ch.to_ascii_lowercase();
                    if is_valid_chord_prefix(lower) {
                        self.state = ChordState::Pending {
                            prefix: lower,
                            started_at_ms: now_ms,
                        };
                        ChordOutcome::PendingPrefix { prefix: lower }
                    } else if let Some(exp) = expired_prefix {
                        ChordOutcome::ExpiredToIdle {
                            expired_prefix: exp,
                        }
                    } else {
                        ChordOutcome::UnboundKey
                    }
                }
                ChordKey::Space if !input.shift => {
                    ChordOutcome::Dispatched(WorkbenchAction::ToggleCrosshairLock)
                }
                ChordKey::Plus => ChordOutcome::Dispatched(WorkbenchAction::RaiseContourFloor),
                ChordKey::Minus if !input.shift => {
                    ChordOutcome::Dispatched(WorkbenchAction::LowerContourFloor)
                }
                ChordKey::CloseBracket | ChordKey::PageUp if !input.shift => {
                    ChordOutcome::Dispatched(WorkbenchAction::NextCubePlane)
                }
                ChordKey::OpenBracket | ChordKey::PageDown if !input.shift => {
                    ChordOutcome::Dispatched(WorkbenchAction::PrevCubePlane)
                }
                ChordKey::QuestionMark => {
                    self.cheat_sheet_open = !self.cheat_sheet_open;
                    ChordOutcome::Dispatched(WorkbenchAction::OpenKeyboardCheatSheet)
                }
                _ => {
                    if let Some(exp) = expired_prefix {
                        ChordOutcome::ExpiredToIdle {
                            expired_prefix: exp,
                        }
                    } else {
                        ChordOutcome::UnboundKey
                    }
                }
            },
        }
    }

    /// Filter [`SPECTRUM_ACCELERATORS`] against `query` (case-insensitive substring over key,
    /// category, title, and description).
    pub fn filter_accelerators(query: &str) -> Vec<&'static AcceleratorBinding> {
        let q = query.trim().to_ascii_lowercase();
        if q.is_empty() {
            return SPECTRUM_ACCELERATORS.iter().collect();
        }
        SPECTRUM_ACCELERATORS
            .iter()
            .filter(|b| {
                b.keys_label.to_ascii_lowercase().contains(&q)
                    || b.category.to_ascii_lowercase().contains(&q)
                    || b.title.to_ascii_lowercase().contains(&q)
                    || b.description.to_ascii_lowercase().contains(&q)
            })
            .collect()
    }

    /// Filter [`GLOBAL_MODIFIER_SHORTCUTS`] against `query`.
    pub fn filter_modifier_shortcuts(
        query: &str,
    ) -> Vec<&'static (&'static str, &'static str, &'static str, WorkbenchAction)> {
        let q = query.trim().to_ascii_lowercase();
        if q.is_empty() {
            return GLOBAL_MODIFIER_SHORTCUTS.iter().collect();
        }
        GLOBAL_MODIFIER_SHORTCUTS
            .iter()
            .filter(|(keys, cat, desc, _)| {
                keys.to_ascii_lowercase().contains(&q)
                    || cat.to_ascii_lowercase().contains(&q)
                    || desc.to_ascii_lowercase().contains(&q)
            })
            .collect()
    }

    /// Poll `egui` frame events, advance chord timeout, and collect emitted [`WorkbenchAction`]s.
    ///
    /// Automatically suppresses chords when `ctx.text_edit_focused()` or
    /// `octet_editor_active` is true, and requests a scheduled repaint when a chord prefix is
    /// pending so the status-bar `"Chord: p_"` indicator expires crisply at `timeout_ms`.
    pub fn handle_egui_input(
        &mut self,
        ctx: &egui::Context,
        octet_editor_active: bool,
    ) -> Vec<WorkbenchAction> {
        let text_focus_active = ctx.text_edit_focused();
        let focus = KeyFocusContext {
            text_focus_active,
            octet_editor_active,
        };

        let (now_ms, inputs) = ctx.input(|i| {
            let ms = (i.time.max(0.0) * 1000.0).round() as u64;
            (ms, extract_egui_chord_inputs(&i.events))
        });

        self.tick_timeout(now_ms);
        if focus.is_suppressed() {
            self.state = ChordState::Idle;
            return Vec::new();
        }

        let mut actions = Vec::new();
        for inp in inputs {
            if let ChordOutcome::Dispatched(act) = self.feed_key(inp, now_ms, focus) {
                actions.push(act);
            }
        }

        if matches!(self.state, ChordState::Pending { .. }) {
            ctx.request_repaint_after_secs((self.timeout_ms as f32) / 1000.0 + 0.02);
        }

        actions
    }

    /// Render the subtle status-bar pending-chord badge (e.g. `Chord: p_`) when active.
    pub fn draw_status_bar_indicator(&mut self, ui: &mut egui::Ui, now_ms: u64) {
        if let Some(ind) = self.status_indicator(now_ms) {
            egui::Frame::NONE
                .fill(egui::Color32::from_rgb(32, 74, 96))
                .stroke(egui::Stroke::new(
                    1.0,
                    egui::Color32::from_rgb(105, 220, 255),
                ))
                .corner_radius(3.0)
                .inner_margin(egui::Margin::symmetric(6, 2))
                .show(ui, |ui| {
                    ui.label(
                        egui::RichText::new(ind)
                            .monospace()
                            .size(11.0)
                            .strong()
                            .color(egui::Color32::from_rgb(205, 245, 255)),
                    );
                });
        }
    }

    /// Render the searchable `?` Keyboard Accelerators Cheat-Sheet `egui::Window` modal.
    ///
    /// Clicking `"Run"` next to any accelerator emits its [`WorkbenchAction`] directly.
    pub fn show_cheat_sheet_window(&mut self, ctx: &egui::Context) -> Option<WorkbenchAction> {
        if !self.cheat_sheet_open {
            return None;
        }

        if !ctx.text_edit_focused()
            && ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Escape))
        {
            self.cheat_sheet_open = false;
            return None;
        }

        let mut window_open = self.cheat_sheet_open;
        let mut triggered: Option<WorkbenchAction> = None;

        egui::Window::new("⌨ NMR Spectrum Keyboard Accelerators")
            .id(egui::Id::unique("nmr_keyboard_cheat_sheet_modal"))
            .open(&mut window_open)
            .collapsible(false)
            .resizable(true)
            .default_width(620.0)
            .default_height(460.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Filter:").strong());
                    let filter_resp = ui.add(
                        egui::TextEdit::singleline(&mut self.cheat_sheet_query)
                            .hint_text("Search key (vt, pa, ct, st, Space), category, or description…")
                            .desired_width(360.0),
                    );
                    self.cheat_sheet_filter_id = Some(filter_resp.id);
                    if !self.cheat_sheet_query.is_empty() && ui.small_button("Clear").clicked() {
                        self.cheat_sheet_query.clear();
                    }
                });

                ui.label(
                    egui::RichText::new(format!(
                        "Two-key chords complete within {} ms (Esc cancels pending prefix; suppressed automatically in text fields and Octet editor).",
                        self.timeout_ms
                    ))
                    .small()
                    .weak(),
                );
                ui.separator();

                let chord_hits = Self::filter_accelerators(&self.cheat_sheet_query);
                let mod_hits = Self::filter_modifier_shortcuts(&self.cheat_sheet_query);

                egui::ScrollArea::vertical()
                    .id_salt("keyboard_cheat_sheet_scroll")
                    .show(ui, |ui| {
                        ui.label(
                            egui::RichText::new("Two-Key & Single-Key Spectrum Accelerators")
                                .strong()
                                .color(egui::Color32::from_rgb(120, 220, 255)),
                        );
                        ui.add_space(3.0);

                        egui::Grid::new("cheat_sheet_chords_grid")
                            .striped(true)
                            .min_col_width(64.0)
                            .show(ui, |ui| {
                                ui.strong("Key(s)");
                                ui.strong("Category");
                                ui.strong("Action & Description");
                                ui.strong("");
                                ui.end_row();

                                for b in chord_hits {
                                    ui.monospace(
                                        egui::RichText::new(format!("[{}]", b.keys_label))
                                            .strong()
                                            .color(egui::Color32::from_rgb(135, 248, 205)),
                                    );
                                    ui.label(
                                        egui::RichText::new(b.category)
                                            .small()
                                            .color(egui::Color32::from_rgb(175, 195, 222)),
                                    );
                                    ui.vertical(|ui| {
                                        ui.label(egui::RichText::new(b.title).strong().size(12.0));
                                        ui.label(
                                            egui::RichText::new(b.description)
                                                .size(11.0)
                                                .weak(),
                                        );
                                    });
                                    if ui.small_button("Run").clicked() {
                                        triggered = Some(b.action);
                                    }
                                    ui.end_row();
                                }
                            });

                        if !mod_hits.is_empty() {
                            ui.add_space(10.0);
                            ui.separator();
                            ui.label(
                                egui::RichText::new("Global Modifier Shortcuts (Passthrough)")
                                    .strong()
                                    .color(egui::Color32::from_rgb(195, 220, 255)),
                            );
                            ui.add_space(3.0);

                            egui::Grid::new("cheat_sheet_modifiers_grid")
                                .striped(true)
                                .min_col_width(96.0)
                                .show(ui, |ui| {
                                    ui.strong("Shortcut");
                                    ui.strong("Category");
                                    ui.strong("Description");
                                    ui.strong("");
                                    ui.end_row();

                                    for &&(sc, cat, desc, act) in &mod_hits {
                                        ui.monospace(
                                            egui::RichText::new(sc)
                                                .strong()
                                                .color(egui::Color32::from_rgb(200, 225, 255)),
                                        );
                                        ui.label(egui::RichText::new(cat).small());
                                        ui.label(egui::RichText::new(desc).size(11.5));
                                        if ui.small_button("Run").clicked() {
                                            triggered = Some(act);
                                        }
                                        ui.end_row();
                                    }
                                });
                        }
                    });
            });

        self.cheat_sheet_open = window_open;
        if triggered.is_some() && triggered != Some(WorkbenchAction::OpenKeyboardCheatSheet) {
            self.cheat_sheet_open = false;
        }
        if !self.cheat_sheet_open {
            if let Some(fid) = self.cheat_sheet_filter_id.take() {
                ctx.memory_mut(|m| m.surrender_focus(fid));
            }
        }
        triggered
    }
}

fn egui_key_to_ascii_letter(key: egui::Key) -> Option<char> {
    match key {
        egui::Key::A => Some('a'),
        egui::Key::B => Some('b'),
        egui::Key::C => Some('c'),
        egui::Key::D => Some('d'),
        egui::Key::E => Some('e'),
        egui::Key::F => Some('f'),
        egui::Key::G => Some('g'),
        egui::Key::H => Some('h'),
        egui::Key::I => Some('i'),
        egui::Key::J => Some('j'),
        egui::Key::K => Some('k'),
        egui::Key::L => Some('l'),
        egui::Key::M => Some('m'),
        egui::Key::N => Some('n'),
        egui::Key::O => Some('o'),
        egui::Key::P => Some('p'),
        egui::Key::Q => Some('q'),
        egui::Key::R => Some('r'),
        egui::Key::S => Some('s'),
        egui::Key::T => Some('t'),
        egui::Key::U => Some('u'),
        egui::Key::V => Some('v'),
        egui::Key::W => Some('w'),
        egui::Key::X => Some('x'),
        egui::Key::Y => Some('y'),
        egui::Key::Z => Some('z'),
        egui::Key::Num1 => Some('1'),
        egui::Key::Num2 => Some('2'),
        egui::Key::Num3 => Some('3'),
        _ => None,
    }
}

/// Convert raw `egui::Event` slice for a single frame into deduplicated [`KeyChordInput`] tokens.
pub fn extract_egui_chord_inputs(events: &[egui::Event]) -> Vec<KeyChordInput> {
    let mut out = Vec::new();
    let mut saw_question_in_frame = false;

    for ev in events {
        match ev {
            egui::Event::Key {
                key,
                pressed: true,
                repeat: false,
                modifiers,
                ..
            } => {
                let ctrl_or_cmd = modifiers.command || modifiers.ctrl;
                let alt = modifiers.alt;
                let shift = modifiers.shift;

                if (*key == egui::Key::Questionmark || (*key == egui::Key::Slash && shift))
                    && !ctrl_or_cmd
                    && !alt
                {
                    if !saw_question_in_frame {
                        saw_question_in_frame = true;
                        out.push(KeyChordInput::question_mark());
                    }
                    continue;
                }

                let mapped = if let Some(ch) = egui_key_to_ascii_letter(*key) {
                    ChordKey::Letter(ch)
                } else if *key == egui::Key::Plus || (*key == egui::Key::Equals && !ctrl_or_cmd) {
                    ChordKey::Plus
                } else if *key == egui::Key::Minus && !ctrl_or_cmd {
                    ChordKey::Minus
                } else if *key == egui::Key::OpenBracket && !ctrl_or_cmd {
                    ChordKey::OpenBracket
                } else if *key == egui::Key::CloseBracket && !ctrl_or_cmd {
                    ChordKey::CloseBracket
                } else if *key == egui::Key::PageUp && !ctrl_or_cmd {
                    ChordKey::PageUp
                } else if *key == egui::Key::PageDown && !ctrl_or_cmd {
                    ChordKey::PageDown
                } else if *key == egui::Key::Space {
                    ChordKey::Space
                } else if *key == egui::Key::Escape {
                    ChordKey::Escape
                } else {
                    ChordKey::Other
                };

                out.push(KeyChordInput {
                    key: mapped,
                    ctrl_or_cmd,
                    alt,
                    shift,
                });
            }
            egui::Event::Text(txt) if txt == "?" && !saw_question_in_frame => {
                saw_question_in_frame = true;
                out.push(KeyChordInput::question_mark());
            }
            _ => {}
        }
    }

    out
}

/// Optional helper polling global modifier shortcuts (`Cmd/Ctrl+Z`, `Cmd/Ctrl+Shift+Z`,
/// `Cmd/Ctrl+S`, `Cmd/Ctrl+Shift+S`, `Cmd/Ctrl+O`, `Cmd/Ctrl+E`, `Cmd/Ctrl+Enter`) when the
/// Octet editor does not own focus.
pub fn poll_global_modifier_shortcuts(
    ctx: &egui::Context,
    octet_editor_active: bool,
) -> Vec<WorkbenchAction> {
    if octet_editor_active {
        return Vec::new();
    }
    let text_focused = ctx.text_edit_focused();

    ctx.input_mut(|i| {
        let cmd = i.modifiers.command || i.modifiers.ctrl;
        if !cmd || i.modifiers.alt {
            return Vec::new();
        }
        let mut out = Vec::new();

        // Allow Save / Open / Export / Run Assignment even if a path box has focus,
        // while leaving Cmd/Ctrl+Z to text widgets when a TextEdit holds focus.
        if !text_focused && i.key_pressed(egui::Key::Z) {
            if i.modifiers.shift {
                out.push(WorkbenchAction::RedoLastCommand);
            } else {
                out.push(WorkbenchAction::UndoLastCommand);
            }
        }
        if !text_focused && !i.modifiers.shift && i.key_pressed(egui::Key::Y) {
            out.push(WorkbenchAction::RedoLastCommand);
        }
        if i.key_pressed(egui::Key::S) {
            if i.modifiers.shift {
                out.push(WorkbenchAction::SaveSessionNefAs);
            } else {
                out.push(WorkbenchAction::SaveSessionNef);
            }
        }
        if !i.modifiers.shift && i.key_pressed(egui::Key::O) {
            out.push(WorkbenchAction::OpenIngestMatrix);
        }
        if !i.modifiers.shift && i.key_pressed(egui::Key::E) {
            out.push(WorkbenchAction::ExportNmrStar);
        }
        if i.key_pressed(egui::Key::Enter) {
            out.push(WorkbenchAction::RunBackboneAssignment);
        }

        out
    })
}
