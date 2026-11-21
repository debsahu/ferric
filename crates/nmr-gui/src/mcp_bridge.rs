//! Live bidirectional `nmr-gui` desktop workbench Model Context Protocol (`MCP`) loopback bridge,
//! bounded `256`-step reversible AI curation command log (`Cmd/Ctrl+Z`), lossless pure-Rust
//! `image/png` viewport encoder, and interactive `egui` control panel (`T15.2`).
//!
//! Provenance: Clean-room original implementation built exclusively on the Rust standard library,
//! `serde_json`, and `eframe::egui` (`docs/roadmap/MASTER_ROADMAP.md` §T15.2):
//! - Public Model Context Protocol (`MCP`, `2024-11-05`) JSON-RPC 2.0 specification (`initialize`,
//!   `tools/list`, `tools/call`) over authenticated loopback HTTP (`127.0.0.1`).
//! - Boutell, T. et al., *PNG (Portable Network Graphics) Specification Version 1.0*, IETF RFC 2083
//!   (March 1997) — `\x89PNG\r\n\x1a\n` signature, `IHDR`, `IDAT`, `IEND`, scanline filter `0`.
//! - Deutsch, P. & Gailly, J.-L., *ZLIB Compressed Data Format Specification version 3.3*,
//!   IETF RFC 1950 (May 1996) — 2-byte `CMF`/`FLG` header and big-endian Adler-32 checksum.
//! - Deutsch, P., *DEFLATE Compressed Data Format Specification version 1.3*, IETF RFC 1951
//!   (May 1996) §3.2.4 — non-compressed (`BTYPE = 00`) stored blocks (`<= 65,535` bytes).
//! - IEEE Std 802.3 — reflected polynomial `0xEDB88320` CRC-32 over PNG chunk `[type || data]`.
//! - Josefsson, S., *The Base16, Base32, and Base64 Data Encodings*, IETF RFC 4648 (October 2006) §4.
//! - Original bounded `256`-step reversible `McpUndoLog` preserving bijective
//!   `(cluster_id <-> seq_id)` assignment invariants in 1 `Cmd/Ctrl+Z` keystroke.

#[cfg(not(any(unix, windows)))]
use std::collections::hash_map::RandomState;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
#[cfg(not(any(unix, windows)))]
use std::hash::{BuildHasher, Hasher};
#[cfg(not(target_arch = "wasm32"))]
use std::io::{ErrorKind, Read, Write};
#[cfg(not(target_arch = "wasm32"))]
use std::net::{Ipv4Addr, Shutdown, SocketAddr, SocketAddrV4, TcpListener, TcpStream};
#[cfg(not(target_arch = "wasm32"))]
use std::path::{Component, Path, PathBuf};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::atomic::AtomicBool;
#[cfg(not(any(unix, windows)))]
use std::sync::atomic::AtomicU64;
#[cfg(any(not(target_arch = "wasm32"), not(any(unix, windows))))]
use std::sync::atomic::Ordering;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Condvar;
use std::sync::{Arc, Mutex};
#[cfg(not(target_arch = "wasm32"))]
use std::thread::{self, JoinHandle};
#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};

use eframe::egui;
#[cfg(not(target_arch = "wasm32"))]
use nmr_core::mcp::{McpAccessMode, McpSecurityError, McpSecurityErrorKind, McpSecurityPolicy};
use serde_json::{json, Value};

use crate::contour_painter::SyncCrosshair;
use crate::strip_tiler::{StripMatchCandidate, StripMatchDirection};
use crate::workflow_bar::WorkbenchAction;

/// Negotiated Model Context Protocol specification revision.
pub const MCP_PROTOCOL_VERSION: &str = "2024-11-05";

/// Bounded maximum depth (`256` reversible steps) for [`McpUndoLog`].
pub const MAX_MCP_UNDO_DEPTH: usize = 256;

/// Default loopback bind address (`127.0.0.1:0`, OS-assigned ephemeral port).
pub const DEFAULT_LOOPBACK_BIND: &str = "127.0.0.1:0";

/// Preferred stable loopback TCP port (`17420`) tried first on desktop start when
/// `bind_input == DEFAULT_LOOPBACK_BIND`, falling back to `127.0.0.1:0` if already bound.
pub const PREFERRED_LOOPBACK_PORT: u16 = 17420;

/// PNG 8-byte file signature (`\x89PNG\r\n\x1a\n`, RFC 2083 §3.1).
pub const PNG_SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];

#[cfg(not(target_arch = "wasm32"))]
const MAX_MCP_HTTP_BODY_BYTES: usize = 4 * 1024 * 1024; // 4 MiB

/// Default total HTTP request read deadline (`2 s`) enforcing slow-loris protection.
#[cfg(not(target_arch = "wasm32"))]
pub const DEFAULT_MCP_HTTP_REQUEST_DEADLINE: Duration = Duration::from_secs(2);

#[cfg(not(target_arch = "wasm32"))]
const MCP_HTTP_SOCKET_READ_SLICE: Duration = Duration::from_millis(200);

#[cfg(not(target_arch = "wasm32"))]
const MCP_HTTP_WRITE_TIMEOUT: Duration = Duration::from_secs(5);

#[cfg(not(target_arch = "wasm32"))]
const MAX_MCP_CONCURRENT_CONNECTIONS: usize = 16;

#[cfg(not(target_arch = "wasm32"))]
const MAX_MCP_SEQUENCE_FILE_BYTES: u64 = 16 * 1024 * 1024;

#[cfg(not(any(unix, windows)))]
static TOKEN_COUNTER: AtomicU64 = AtomicU64::new(1);

// ============================================================================
// 1. Live Workbench Snapshot, Reversible McpUiCommand & 256-Step McpUndoLog
// ============================================================================

/// Access-control tier governing incoming MCP tool invocations.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum McpBridgePermission {
    /// Allows state inspection (`get_workbench_state`) and viewport capture (`capture_viewport_png`)
    /// while deterministically blocking all UI navigation and curation mutations.
    #[default]
    ReadOnly,
    /// Allows both inspection/capture and Undo-logged UI navigation & curation edits.
    ReadWrite,
}

impl McpBridgePermission {
    /// Human-readable label for UI badges and JSON status serialization.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ReadOnly => "read_only",
            Self::ReadWrite => "read_write",
        }
    }
}

/// Ranked `i - 1 / i / i + 1` strip match summary exposed in [`WorkbenchSnapshot`].
#[derive(Clone, Debug, PartialEq)]
pub struct StripCandidateSummary {
    /// Candidate spin-system cluster index.
    pub cluster_id: usize,
    /// Sequential direction relative to the active strip (`"i-1"` or `"i+1"`).
    pub direction: String,
    /// Pairwise sequential match log-likelihood ratio (`P` score in nats).
    pub pair_score: f64,
    /// Normalized posterior relative probability in `[0.0, 1.0]`.
    pub rel_prob: f64,
    /// Currently assigned 1-based residue `Seq_ID`, if placed.
    pub assigned_seq_id: Option<i32>,
}

impl StripCandidateSummary {
    /// Convert a [`StripMatchCandidate`] into a transport-ready [`StripCandidateSummary`].
    pub fn from_strip_match_candidate(cand: &StripMatchCandidate, first_seq_id: i32) -> Self {
        let direction = match cand.direction {
            StripMatchDirection::Predecessor => "i-1".to_string(),
            StripMatchDirection::Successor => "i+1".to_string(),
        };
        let assigned_seq_id = cand
            .assigned_residue
            .map(|idx| first_seq_id.saturating_add(idx as i32));
        Self {
            cluster_id: cand.spin_system_idx,
            direction,
            pair_score: cand.pair_score,
            rel_prob: cand.rel_prob,
            assigned_seq_id,
        }
    }

    /// Serialize this candidate summary to a JSON object.
    pub fn to_json(&self) -> Value {
        json!({
            "cluster_id": self.cluster_id,
            "direction": self.direction,
            "pair_score": self.pair_score,
            "rel_prob": self.rel_prob,
            "assigned_seq_id": self.assigned_seq_id,
        })
    }
}

/// Curated spin-system placement record attached to a polymer residue in [`WorkbenchSnapshot`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssignedClusterRecord {
    /// Spin-system cluster identifier bijectively linked to this residue.
    pub cluster_id: usize,
    /// Human- and AI-auditable rationale recorded when this placement was curated.
    pub reason: String,
}

/// Live inspectable snapshot of the interactive `nmr-gui` desktop workbench state.
#[derive(Clone, Debug, PartialEq)]
pub struct WorkbenchSnapshot {
    /// Active 1-based polymer residue `Seq_ID` focused in the strip viewer / sequence ribbon.
    pub active_residue_seq: Option<i32>,
    /// Synchronized 2D/3D viewport crosshair coordinates `(h_ppm, n_ppm, c_ppm)`.
    pub crosshair_ppm: (f64, f64, Option<f64>),
    /// Active primary spectrum / experiment identifier (e.g. `"HSQC"`, `"HNCACB"`).
    pub active_experiment: String,
    /// Ranked `i - 1` and `i + 1` candidate spin-system strips around the active root.
    pub strip_candidates: Vec<StripCandidateSummary>,
    /// Selected `(chain_id, res_seq_id)` in the 3D biomolecular structure viewport, if any.
    pub selected_pdb_chain_res: Option<(String, i32)>,
    /// Current reversible Undo stack depth (`0..=256`).
    pub undo_depth: usize,
    /// Current Redo stack depth (`0..=256`).
    pub redo_depth: usize,
    /// Dataset quality-control verdict badge (`"PASS"`, `"WARN"`, `"FAIL"`, `"UNRUN"`).
    pub qc_status: String,
    /// Bijective residue-to-cluster assignments (`seq_id -> AssignedClusterRecord`).
    pub assigned_residues: BTreeMap<i32, AssignedClusterRecord>,
    /// Set of picked peak indices currently marked as excluded artifacts/noise.
    pub excluded_peaks: BTreeSet<usize>,
    /// Ordered 1-based residue `Seq_ID`s highlighted in the 3D structure viewer.
    pub highlighted_residues_3d: Vec<i32>,
    /// Total residue count in the loaded target polymer sequence.
    pub sequence_len: usize,
    /// Labels of spectra currently loaded in the workbench.
    pub loaded_spectra: Vec<String>,
    /// Total picked peak count currently loaded across workbench spectra.
    pub peak_count: usize,
    /// Total assembled spin-system count currently in the workbench.
    pub spin_system_count: usize,
    /// Number of assigned residues produced by automated or curated assignment.
    pub assigned_count: usize,
    /// Number of high-confidence assigned residues.
    pub high_conf_count: usize,
    /// Current workbench status bar message.
    pub status_msg: String,
    /// Whether an asynchronous study-folder / spectrum ingest task is running.
    pub ingest_in_progress: bool,
    /// Active peak-picking backend mode (`"classical"` or `"neural"`).
    pub picker_mode: String,
    /// Active peak-picking minimum SNR threshold.
    pub picker_snr: f64,
    /// Active top-level workspace mode (`"ProteinNmr"`, `"OctetChem"`, `"Dosy"`, `"Metabolomics"`).
    pub workspace_mode: String,
    /// Active 2D/3D workbook page index.
    pub active_page_index: usize,
    /// Active backbone solver backend (`"evo-graph"`, `"standard"`, `"bayes-sa"`).
    pub solver_kind: String,
    /// Active 2D/3D contour floor threshold.
    pub contour_base: f64,
    /// Active 2D viewport chemical shift bounds `[h_left, h_right, n_top, n_bottom]`.
    pub view_bounds: [f64; 4],
    /// Live background ingest progress status text, if any.
    pub ingest_progress_text: String,
    /// Per-experiment picked peak counts keyed by experiment label.
    pub peaks_by_experiment: BTreeMap<String, usize>,
}

impl Default for WorkbenchSnapshot {
    fn default() -> Self {
        let xh = SyncCrosshair::default();
        Self {
            active_residue_seq: None,
            crosshair_ppm: (xh.h_ppm, xh.n_ppm, xh.c_ppm),
            active_experiment: "HSQC".to_string(),
            strip_candidates: Vec::new(),
            selected_pdb_chain_res: None,
            undo_depth: 0,
            redo_depth: 0,
            qc_status: "UNRUN".to_string(),
            assigned_residues: BTreeMap::new(),
            excluded_peaks: BTreeSet::new(),
            highlighted_residues_3d: Vec::new(),
            sequence_len: 0,
            loaded_spectra: Vec::new(),
            peak_count: 0,
            spin_system_count: 0,
            assigned_count: 0,
            high_conf_count: 0,
            status_msg: String::new(),
            ingest_in_progress: false,
            picker_mode: "classical".to_string(),
            picker_snr: 5.0,
            workspace_mode: "ProteinNmr".to_string(),
            active_page_index: 0,
            solver_kind: "evo-graph".to_string(),
            contour_base: 0.0,
            view_bounds: [10.5, 6.0, 102.0, 134.0],
            ingest_progress_text: String::new(),
            peaks_by_experiment: BTreeMap::new(),
        }
    }
}

impl WorkbenchSnapshot {
    /// Synchronize crosshair coordinates from a [`SyncCrosshair`].
    pub fn set_sync_crosshair(&mut self, xh: SyncCrosshair) {
        self.crosshair_ppm = (xh.h_ppm, xh.n_ppm, xh.c_ppm);
    }

    /// Export crosshair coordinates as a [`SyncCrosshair`].
    pub fn to_sync_crosshair(&self) -> SyncCrosshair {
        SyncCrosshair::new(
            self.crosshair_ppm.0,
            self.crosshair_ppm.1,
            self.crosshair_ppm.2,
        )
    }

    /// Serialize the full workbench snapshot into a structured JSON value for `get_workbench_state`.
    pub fn to_json(&self) -> Value {
        let mut assigned_obj = serde_json::Map::new();
        for (seq_id, rec) in &self.assigned_residues {
            assigned_obj.insert(
                seq_id.to_string(),
                json!({
                    "cluster_id": rec.cluster_id,
                    "reason": rec.reason,
                }),
            );
        }
        let excluded: Vec<usize> = self.excluded_peaks.iter().copied().collect();
        let candidates: Vec<Value> = self
            .strip_candidates
            .iter()
            .map(StripCandidateSummary::to_json)
            .collect();
        let selected_pdb = self
            .selected_pdb_chain_res
            .as_ref()
            .map(|(chain, res)| json!({ "chain": chain, "res_seq": res }));

        json!({
            "active_residue_seq": self.active_residue_seq,
            "crosshair_ppm": {
                "h_ppm": self.crosshair_ppm.0,
                "n_ppm": self.crosshair_ppm.1,
                "c_ppm": self.crosshair_ppm.2,
            },
            "active_experiment": self.active_experiment,
            "strip_candidates": candidates,
            "selected_pdb_chain_res": selected_pdb,
            "undo_depth": self.undo_depth,
            "redo_depth": self.redo_depth,
            "qc_status": self.qc_status,
            "assigned_residues": Value::Object(assigned_obj),
            "excluded_peaks": excluded,
            "highlighted_residues_3d": self.highlighted_residues_3d,
            "sequence_len": self.sequence_len,
            "loaded_spectra": self.loaded_spectra,
            "peak_count": self.peak_count,
            "spin_system_count": self.spin_system_count,
            "assigned_count": self.assigned_count,
            "high_conf_count": self.high_conf_count,
            "status_msg": self.status_msg,
            "ingest_in_progress": self.ingest_in_progress,
            "picker_mode": self.picker_mode,
            "picker_snr": self.picker_snr,
            "workspace_mode": self.workspace_mode,
            "active_page_index": self.active_page_index,
            "solver_kind": self.solver_kind,
            "contour_base": self.contour_base,
            "view_bounds": self.view_bounds,
            "ingest_progress_text": self.ingest_progress_text,
            "peaks_by_experiment": self.peaks_by_experiment,
        })
    }
}

/// Structured tabular workbench data published read-only for `inspect_workbench_data`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct WorkbenchDataTables {
    /// Per-residue assignment rows (`seq_id`, `res_name`, `spin_system_idx`, `confidence`, etc.).
    pub assignments: Vec<Value>,
    /// Assembled spin-system cluster rows.
    pub spin_systems: Vec<Value>,
    /// Picked 2D/3D cross-peak rows (`experiment`, `h_ppm`, `n_ppm`, `c_ppm`, `intensity`, etc.).
    pub peaks: Vec<Value>,
    /// Loaded 2D/3D workbook page descriptors.
    pub workbook_pages: Vec<Value>,
    /// Latest dataset quality-control report JSON, if evaluated.
    pub qc_report: Option<Value>,
}

fn normalize_action_token(s: &str) -> String {
    s.chars()
        .filter(|c| !matches!(c, '_' | '-' | ' ' | '\t'))
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

/// Return the canonical PascalCase name of a [`WorkbenchAction`] (matching its Rust variant name).
pub fn workbench_action_canonical_name(action: WorkbenchAction) -> String {
    format!("{action:?}")
}

/// Return the snake_case identifier for a [`WorkbenchAction`].
fn workbench_action_snake_name(action: WorkbenchAction) -> String {
    let raw = workbench_action_canonical_name(action);
    let mut out = String::with_capacity(raw.len() + 6);
    let mut prev: Option<char> = None;
    for ch in raw.chars() {
        if let Some(p) = prev {
            if (ch.is_ascii_uppercase() || ch.is_ascii_digit()) && p.is_ascii_lowercase() {
                out.push('_');
            }
        }
        out.push(ch.to_ascii_lowercase());
        prev = Some(ch);
    }
    out
}

/// Parse a [`WorkbenchAction`] name case-insensitively across PascalCase (`"SwitchModeAssign"`),
/// snake_case (`"switch_mode_assign"`), and kebab-case (`"switch-mode-assign"`).
pub fn parse_workbench_action(name: &str) -> Option<WorkbenchAction> {
    let target = normalize_action_token(name.trim());
    if target.is_empty() {
        return None;
    }
    WorkbenchAction::ALL
        .iter()
        .copied()
        .find(|&act| normalize_action_token(&format!("{act:?}")) == target)
}

/// Return `true` when dispatching `action` via MCP replaces or wipes workspace state without
/// an MCP undo entry and must therefore carry `destructive: true` metadata and be gated behind
/// GUI `allow_destructive_actions` plus explicit `confirm: true` (or remain blocked via MCP for
/// interactive file dialogs such as [`WorkbenchAction::LoadPathOrSpectrum`]).
pub const fn workbench_action_is_destructive(action: WorkbenchAction) -> bool {
    matches!(
        action,
        WorkbenchAction::ResetWorkspace
            | WorkbenchAction::LoadUbiquitinDemo
            | WorkbenchAction::LoadSessionNef
            | WorkbenchAction::LoadPathOrSpectrum
            | WorkbenchAction::ReSimulateSyntheticNoise
    )
}

/// Destructive-action classification for parameterized `run_workbench_workflow` actions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkflowDestructiveMode {
    /// Action does not replace or wipe existing workspace/assignment state.
    Never,
    /// Action always replaces workspace state without an MCP undo entry.
    Always,
    /// Action replaces or clears existing sequence/assignment state when non-empty.
    WhenReplacingState,
}

impl WorkflowDestructiveMode {
    /// Wire-stable string representation for `inspect_workbench_data(section="actions")`.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Never => "never",
            Self::Always => "always",
            Self::WhenReplacingState => "when_replacing_state",
        }
    }

    /// Return `true` when this workflow action can destructively replace workspace state.
    pub const fn is_destructive(self) -> bool {
        !matches!(self, Self::Never)
    }
}

/// Canonical list of all `run_workbench_workflow` action names.
const ALL_WORKFLOW_ACTIONS: &[&str] = &[
    "load_study_folder",
    "set_sequence",
    "pick_peaks",
    "run_assignment",
    "select_workbook_page",
    "set_contour_level",
    "set_viewport_bounds",
    "export_results",
    "set_lock_residue",
];

/// Return the [`WorkflowDestructiveMode`] for a `run_workbench_workflow` action name.
pub fn workflow_action_destructive_mode(action: &str) -> WorkflowDestructiveMode {
    match action.trim() {
        "load_study_folder" => WorkflowDestructiveMode::Always,
        "set_sequence" => WorkflowDestructiveMode::WhenReplacingState,
        _ => WorkflowDestructiveMode::Never,
    }
}

/// Undo-tagged UI navigation and resonance curation commands executable via the MCP bridge.
#[derive(Clone, Debug, PartialEq)]
pub enum McpUiCommand {
    /// Focus sequence strip and 3D residue selection on 1-based `seq_id`.
    JumpToResidue { seq_id: i32 },
    /// Move synchronized 2D/3D viewport crosshair to `(h_ppm, n_ppm, c_ppm)`.
    SetCrosshairPpm {
        h_ppm: f64,
        n_ppm: f64,
        c_ppm: Option<f64>,
    },
    /// Assign spin-system `cluster_id` to polymer residue `seq_id` (preserving bijection).
    AssignSpinSystemToResidue {
        cluster_id: usize,
        seq_id: i32,
        reason: String,
    },
    /// Clear curated assignment on polymer residue `seq_id`.
    UnassignResidue { seq_id: i32, reason: String },
    /// Mark or unmark picked peak `peak_id` as an excluded artifact.
    TogglePeakExclusion { peak_id: usize, excluded: bool },
    /// Highlight 1-based residue `seq_ids` in the 3D biomolecular viewport.
    HighlightResidues3d { seq_ids: Vec<i32> },
    /// Queue loading a study folder of NMR spectra (optionally running the full pipeline).
    LoadStudyFolder {
        path: String,
        run_full_pipeline: bool,
        confirm: bool,
    },
    /// Queue setting the target protein amino-acid sequence in the workbench (inline or via path).
    SetSequence {
        sequence: String,
        sequence_path: Option<String>,
        confirm: bool,
    },
    /// Queue peak picking across loaded spectra with optional manual/auto SNR and picker backend.
    PickAllSpectra {
        snr_threshold: Option<f64>,
        use_auto_threshold: bool,
        picker_mode: Option<String>,
    },
    /// Queue running backbone resonance assignment (`"evo-graph"`, `"standard"`, `"bayes-sa"`)
    /// with optional compute engine (`"cpu"` or `"gpu"`).
    RunAssignment {
        solver: String,
        engine: Option<String>,
    },
    /// Queue selecting an active 2D/3D workbook page by index and/or name.
    SelectWorkbookPage {
        page_index: Option<usize>,
        page_name: Option<String>,
    },
    /// Queue updating 2D/3D contour floor, level multiplier, and/or negative contours toggle.
    SetContourLevel {
        contour_base: Option<f64>,
        multiplier: Option<f64>,
        show_negative: Option<bool>,
    },
    /// Queue setting 2D viewport chemical-shift bounds `[h_left, h_right, n_top, n_bottom]`.
    SetViewportBounds {
        h_left: f64,
        h_right: f64,
        n_top: f64,
        n_bottom: f64,
    },
    /// Queue exporting workbench outputs (`"star"`, `"peaks"`, `"nef"`, or `"html"`) to `prefix`.
    ExportResults {
        prefix: String,
        format: String,
        overwrite: bool,
    },
    /// Queue locking or unlocking a 1-based residue assignment anchor.
    SetLockResidue { seq_id: i32, locked: bool },
    /// Queue dispatching any registered [`WorkbenchAction`] from the workbench action catalog.
    DispatchWorkbenchAction { action: WorkbenchAction },
    /// Queue dispatching a [`WorkbenchAction`] carrying explicit overwrite/confirmation flags.
    DispatchGuardedWorkbenchAction {
        action: WorkbenchAction,
        overwrite: bool,
        confirm: bool,
    },
}

impl McpUiCommand {
    /// Short human-readable summary recorded in the live AI audit log.
    pub fn summary(&self) -> String {
        match self {
            Self::JumpToResidue { seq_id } => format!("JumpToResidue(seq={seq_id})"),
            Self::SetCrosshairPpm {
                h_ppm,
                n_ppm,
                c_ppm: Some(c),
            } => format!("SetCrosshairPpm(H={h_ppm:.3}, N={n_ppm:.2}, C={c:.2})"),
            Self::SetCrosshairPpm {
                h_ppm,
                n_ppm,
                c_ppm: None,
            } => format!("SetCrosshairPpm(H={h_ppm:.3}, N={n_ppm:.2})"),
            Self::AssignSpinSystemToResidue {
                cluster_id,
                seq_id,
                reason,
            } => format!("AssignSpinSystem(cluster={cluster_id} -> seq={seq_id}: {reason})"),
            Self::UnassignResidue { seq_id, reason } => {
                format!("UnassignResidue(seq={seq_id}: {reason})")
            }
            Self::TogglePeakExclusion { peak_id, excluded } => {
                format!("TogglePeakExclusion(peak={peak_id}, excluded={excluded})")
            }
            Self::HighlightResidues3d { seq_ids } => {
                format!(
                    "HighlightResidues3d(count={}, {:?})",
                    seq_ids.len(),
                    seq_ids
                )
            }
            Self::LoadStudyFolder {
                path,
                run_full_pipeline,
                confirm,
            } => {
                format!(
                    "LoadStudyFolder(path={path}, run_full_pipeline={run_full_pipeline}, confirm={confirm})"
                )
            }
            Self::SetSequence {
                sequence,
                sequence_path,
                confirm,
            } => match sequence_path {
                Some(p) if sequence.is_empty() => {
                    format!("SetSequence(path={p}, confirm={confirm})")
                }
                Some(p) => format!(
                    "SetSequence(len={}, path={p}, confirm={confirm})",
                    sequence.len()
                ),
                None => format!("SetSequence(len={}, confirm={confirm})", sequence.len()),
            },
            Self::PickAllSpectra {
                snr_threshold,
                use_auto_threshold,
                picker_mode,
            } => {
                let snr_str = match snr_threshold {
                    Some(snr) => format!("{snr:.2}"),
                    None => "current".to_string(),
                };
                match picker_mode {
                    Some(mode) => format!(
                        "PickAllSpectra(snr={snr_str}, auto={use_auto_threshold}, mode={mode})"
                    ),
                    None => format!("PickAllSpectra(snr={snr_str}, auto={use_auto_threshold})"),
                }
            }
            Self::RunAssignment { solver, engine } => match engine {
                Some(eng) => format!("RunAssignment(solver={solver}, engine={eng})"),
                None => format!("RunAssignment(solver={solver})"),
            },
            Self::SelectWorkbookPage {
                page_index,
                page_name,
            } => {
                format!("SelectWorkbookPage(index={page_index:?}, name={page_name:?})")
            }
            Self::SetContourLevel {
                contour_base,
                multiplier,
                show_negative,
            } => format!(
                "SetContourLevel(base={contour_base:?}, mult={multiplier:?}, neg={show_negative:?})"
            ),
            Self::SetViewportBounds {
                h_left,
                h_right,
                n_top,
                n_bottom,
            } => format!(
                "SetViewportBounds(H=[{h_left:.2}, {h_right:.2}], N=[{n_top:.2}, {n_bottom:.2}])"
            ),
            Self::ExportResults {
                prefix,
                format,
                overwrite,
            } => {
                format!("ExportResults(prefix={prefix}, format={format}, overwrite={overwrite})")
            }
            Self::SetLockResidue { seq_id, locked } => {
                format!("SetLockResidue(seq={seq_id}, locked={locked})")
            }
            Self::DispatchWorkbenchAction { action } => {
                format!("DispatchWorkbenchAction({action:?})")
            }
            Self::DispatchGuardedWorkbenchAction {
                action,
                overwrite,
                confirm,
            } => {
                format!(
                    "DispatchGuardedWorkbenchAction({action:?}, overwrite={overwrite}, confirm={confirm})"
                )
            }
        }
    }
}

/// Exact O(1) inverse state delta paired with each applied [`McpUiCommand`].
#[derive(Clone, Debug)]
enum UndoDelta {
    JumpToResidue {
        prev_active_residue_seq: Option<i32>,
        prev_selected_pdb_chain_res: Option<(String, i32)>,
    },
    SetCrosshairPpm {
        prev_crosshair_ppm: (f64, f64, Option<f64>),
    },
    AssignSpinSystemToResidue {
        seq_id: i32,
        prev_at_target: Option<AssignedClusterRecord>,
        displaced_source: Option<(i32, AssignedClusterRecord)>,
    },
    UnassignResidue {
        seq_id: i32,
        prev_record: Option<AssignedClusterRecord>,
    },
    TogglePeakExclusion {
        peak_id: usize,
        prev_excluded: bool,
    },
    HighlightResidues3d {
        prev_highlighted: Vec<i32>,
    },
    WorkflowAction {
        prev_status_msg: String,
        prev_picker_snr: f64,
        prev_picker_mode: String,
        prev_solver_kind: String,
        prev_active_page_index: usize,
        prev_contour_base: f64,
        prev_view_bounds: [f64; 4],
    },
}

#[derive(Clone, Debug)]
struct McpUndoEntry {
    action_id: u64,
    command: McpUiCommand,
    delta: UndoDelta,
}

/// Status kind recorded in the live UI AI action audit trail.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum McpAuditOutcome {
    Applied,
    Undone,
    Redone,
    BlockedReadOnly,
    BlockedSecurity,
}

impl McpAuditOutcome {
    /// Short badge label for the `egui` audit table.
    pub fn badge(self) -> &'static str {
        match self {
            Self::Applied => "APPLIED",
            Self::Undone => "UNDONE",
            Self::Redone => "REDONE",
            Self::BlockedReadOnly => "BLOCKED",
            Self::BlockedSecurity => "DENIED",
        }
    }
}

/// Audit log row displayed in [`render_mcp_bridge_panel`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct McpAuditRecord {
    /// Monotonic audit sequence number (`1, 2, ...`).
    pub seq: u64,
    /// Outcome of the requested AI action.
    pub outcome: McpAuditOutcome,
    /// Concise human-readable description of the command or refusal.
    pub detail: String,
}

/// Bounded `256`-step reversible Undo/Redo log for MCP workbench curation commands.
///
/// Every AI navigation or curation command applied through [`McpUndoLog::apply_command`] records an
/// exact pre-image [`UndoDelta`] so a single `Cmd/Ctrl+Z` ([`McpUndoLog::undo_last`]) restores the
/// pre-command [`WorkbenchSnapshot`] in one atomic step, including bijective cluster displacements.
#[derive(Clone, Debug)]
pub struct McpUndoLog {
    max_depth: usize,
    next_action_id: u64,
    next_audit_seq: u64,
    undo_stack: VecDeque<McpUndoEntry>,
    redo_stack: Vec<McpUndoEntry>,
    audit_log: VecDeque<McpAuditRecord>,
    pending_ui_commands: VecDeque<McpUiCommand>,
}

impl Default for McpUndoLog {
    fn default() -> Self {
        Self::new()
    }
}

impl McpUndoLog {
    /// Construct a new Undo/Redo log bounded to [`MAX_MCP_UNDO_DEPTH`] (`256` steps).
    pub fn new() -> Self {
        Self::with_max_depth(MAX_MCP_UNDO_DEPTH)
    }

    /// Construct a new Undo/Redo log with an explicit maximum step capacity (`>= 1`).
    pub fn with_max_depth(max_depth: usize) -> Self {
        let cap = max_depth.max(1);
        Self {
            max_depth: cap,
            next_action_id: 1,
            next_audit_seq: 1,
            undo_stack: VecDeque::with_capacity(cap.min(64)),
            redo_stack: Vec::new(),
            audit_log: VecDeque::with_capacity(64),
            pending_ui_commands: VecDeque::new(),
        }
    }

    /// Maximum Undo stack capacity (`256` by default).
    pub fn max_depth(&self) -> usize {
        self.max_depth
    }

    /// Number of commands currently available to Undo (`0..=max_depth`).
    pub fn undo_depth(&self) -> usize {
        self.undo_stack.len()
    }

    /// Number of undone commands currently available to Redo.
    pub fn redo_depth(&self) -> usize {
        self.redo_stack.len()
    }

    /// Return `true` when at least one AI command can be undone.
    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    /// Return `true` when at least one undone AI command can be redone.
    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    /// Return `true` when a `SetSequence` command is already pending in the GUI command queue.
    fn has_pending_set_sequence(&self) -> bool {
        self.pending_ui_commands
            .iter()
            .any(|cmd| matches!(cmd, McpUiCommand::SetSequence { .. }))
    }

    /// Immutable view of recent AI audit records (oldest to newest, bounded to `256` entries).
    pub fn audit_records(&self) -> &VecDeque<McpAuditRecord> {
        &self.audit_log
    }

    /// Drain queued [`McpUiCommand`] notifications for downstream GUI synchronization.
    pub fn drain_pending_ui_commands(&mut self) -> Vec<McpUiCommand> {
        self.pending_ui_commands.drain(..).collect()
    }

    /// Record a `ReadOnly`-blocked mutation attempt in the live audit trail without mutating state.
    pub fn record_blocked_readonly(&mut self, summary: impl Into<String>) {
        self.push_audit(McpAuditOutcome::BlockedReadOnly, summary.into());
    }

    /// Record a security-sandbox, overwrite, or destructive-action refusal in the live audit trail.
    pub fn record_blocked_security(&mut self, summary: impl Into<String>) {
        self.push_audit(McpAuditOutcome::BlockedSecurity, summary.into());
    }

    /// Apply `command` to `snapshot`, push an exact inverse record onto the bounded Undo stack,
    /// clear the Redo branch, and return the assigned `action_id`.
    pub fn apply_command(
        &mut self,
        snapshot: &mut WorkbenchSnapshot,
        command: McpUiCommand,
    ) -> u64 {
        let summary = command.summary();
        let delta = Self::execute_forward(snapshot, &command);
        let action_id = self.next_action_id;
        self.next_action_id = self.next_action_id.saturating_add(1);

        self.redo_stack.clear();
        if self.undo_stack.len() >= self.max_depth {
            self.undo_stack.pop_front();
        }
        self.undo_stack.push_back(McpUndoEntry {
            action_id,
            command: command.clone(),
            delta,
        });
        self.enqueue_ui_command(command);
        self.sync_depths(snapshot);
        self.push_audit(McpAuditOutcome::Applied, format!("#{action_id} {summary}"));
        action_id
    }

    /// Revert the most recently applied command in 1 atomic step (`Cmd/Ctrl+Z`), pushing it onto
    /// the Redo stack and returning the reverted [`McpUiCommand`].
    pub fn undo_last(&mut self, snapshot: &mut WorkbenchSnapshot) -> Option<McpUiCommand> {
        let entry = self.undo_stack.pop_back()?;
        Self::apply_delta_inverse(snapshot, &entry.delta);
        let action_id = entry.action_id;
        let cmd = entry.command.clone();
        let summary = cmd.summary();
        if self.redo_stack.len() < self.max_depth {
            self.redo_stack.push(entry);
        }
        self.sync_depths(snapshot);
        self.push_audit(McpAuditOutcome::Undone, format!("#{action_id} {summary}"));
        Some(cmd)
    }

    /// Re-apply the most recently undone command (`Cmd/Ctrl+Shift+Z`), pushing a fresh inverse
    /// delta back onto the Undo stack and returning the redone [`McpUiCommand`].
    pub fn redo_last(&mut self, snapshot: &mut WorkbenchSnapshot) -> Option<McpUiCommand> {
        let prev_entry = self.redo_stack.pop()?;
        let cmd = prev_entry.command;
        let action_id = prev_entry.action_id;
        let summary = cmd.summary();
        let delta = Self::execute_forward(snapshot, &cmd);
        if self.undo_stack.len() >= self.max_depth {
            self.undo_stack.pop_front();
        }
        self.undo_stack.push_back(McpUndoEntry {
            action_id,
            command: cmd.clone(),
            delta,
        });
        self.enqueue_ui_command(cmd.clone());
        self.sync_depths(snapshot);
        self.push_audit(McpAuditOutcome::Redone, format!("#{action_id} {summary}"));
        Some(cmd)
    }

    fn execute_forward(snapshot: &mut WorkbenchSnapshot, command: &McpUiCommand) -> UndoDelta {
        match command {
            McpUiCommand::JumpToResidue { seq_id } => {
                let prev_active_residue_seq = snapshot.active_residue_seq;
                let prev_selected_pdb_chain_res = snapshot.selected_pdb_chain_res.clone();
                snapshot.active_residue_seq = Some(*seq_id);
                if let Some((ref chain, _)) = prev_selected_pdb_chain_res {
                    snapshot.selected_pdb_chain_res = Some((chain.clone(), *seq_id));
                }
                UndoDelta::JumpToResidue {
                    prev_active_residue_seq,
                    prev_selected_pdb_chain_res,
                }
            }
            McpUiCommand::SetCrosshairPpm {
                h_ppm,
                n_ppm,
                c_ppm,
            } => {
                let prev_crosshair_ppm = snapshot.crosshair_ppm;
                snapshot.crosshair_ppm = (*h_ppm, *n_ppm, *c_ppm);
                UndoDelta::SetCrosshairPpm { prev_crosshair_ppm }
            }
            McpUiCommand::AssignSpinSystemToResidue {
                cluster_id,
                seq_id,
                reason,
            } => {
                // Enforce strict cluster_id <-> seq_id bijection across assigned_residues.
                let existing_other_seq = snapshot
                    .assigned_residues
                    .iter()
                    .find(|(&k, v)| k != *seq_id && v.cluster_id == *cluster_id)
                    .map(|(&k, _)| k);
                let displaced_source = existing_other_seq.and_then(|other_seq| {
                    snapshot
                        .assigned_residues
                        .remove(&other_seq)
                        .map(|rec| (other_seq, rec))
                });
                let prev_at_target = snapshot.assigned_residues.insert(
                    *seq_id,
                    AssignedClusterRecord {
                        cluster_id: *cluster_id,
                        reason: reason.clone(),
                    },
                );
                UndoDelta::AssignSpinSystemToResidue {
                    seq_id: *seq_id,
                    prev_at_target,
                    displaced_source,
                }
            }
            McpUiCommand::UnassignResidue { seq_id, reason: _ } => {
                let prev_record = snapshot.assigned_residues.remove(seq_id);
                UndoDelta::UnassignResidue {
                    seq_id: *seq_id,
                    prev_record,
                }
            }
            McpUiCommand::TogglePeakExclusion { peak_id, excluded } => {
                let prev_excluded = snapshot.excluded_peaks.contains(peak_id);
                if *excluded {
                    snapshot.excluded_peaks.insert(*peak_id);
                } else {
                    snapshot.excluded_peaks.remove(peak_id);
                }
                UndoDelta::TogglePeakExclusion {
                    peak_id: *peak_id,
                    prev_excluded,
                }
            }
            McpUiCommand::HighlightResidues3d { seq_ids } => {
                let prev_highlighted =
                    std::mem::replace(&mut snapshot.highlighted_residues_3d, seq_ids.clone());
                UndoDelta::HighlightResidues3d { prev_highlighted }
            }
            McpUiCommand::LoadStudyFolder {
                path,
                run_full_pipeline,
                ..
            } => {
                let delta = Self::capture_workflow_undo(snapshot);
                snapshot.status_msg =
                    format!("Queued study folder load: {path} (full_pipeline={run_full_pipeline})");
                delta
            }
            McpUiCommand::SetSequence {
                sequence,
                sequence_path,
                ..
            } => {
                let delta = Self::capture_workflow_undo(snapshot);
                snapshot.status_msg = match sequence_path {
                    Some(p) if sequence.is_empty() => format!("Queued sequence file: {p}"),
                    Some(p) => {
                        format!(
                            "Queued sequence update ({} chars, path={p})",
                            sequence.len()
                        )
                    }
                    None => format!("Queued sequence update ({} chars)", sequence.len()),
                };
                delta
            }
            McpUiCommand::PickAllSpectra {
                snr_threshold,
                use_auto_threshold,
                picker_mode,
            } => {
                let delta = Self::capture_workflow_undo(snapshot);
                if let Some(snr) = snr_threshold {
                    if !*use_auto_threshold && snr.is_finite() && *snr > 0.0 {
                        snapshot.picker_snr = *snr;
                    }
                }
                if let Some(mode) = picker_mode {
                    snapshot.picker_mode = mode.clone();
                }
                snapshot.status_msg = match snr_threshold {
                    Some(snr) => {
                        format!("Queued peak picking (snr={snr:.2}, auto={use_auto_threshold})")
                    }
                    None => format!("Queued peak picking (auto={use_auto_threshold})"),
                };
                delta
            }
            McpUiCommand::RunAssignment { solver, engine } => {
                let delta = Self::capture_workflow_undo(snapshot);
                snapshot.solver_kind = solver.clone();
                snapshot.status_msg = match engine {
                    Some(eng) => format!("Queued assignment run (solver={solver}, engine={eng})"),
                    None => format!("Queued assignment run (solver={solver})"),
                };
                delta
            }
            McpUiCommand::SelectWorkbookPage {
                page_index,
                page_name,
            } => {
                let delta = Self::capture_workflow_undo(snapshot);
                if let Some(idx) = page_index {
                    snapshot.active_page_index = *idx;
                }
                snapshot.status_msg =
                    format!("Queued workbook page selection ({page_index:?}, {page_name:?})");
                delta
            }
            McpUiCommand::SetContourLevel {
                contour_base,
                multiplier,
                show_negative,
            } => {
                let delta = Self::capture_workflow_undo(snapshot);
                if let Some(cb) = contour_base {
                    if cb.is_finite() && *cb >= 0.0 {
                        snapshot.contour_base = *cb;
                    }
                }
                snapshot.status_msg = format!(
                    "Queued contour update (base={contour_base:?}, mult={multiplier:?}, neg={show_negative:?})"
                );
                delta
            }
            McpUiCommand::SetViewportBounds {
                h_left,
                h_right,
                n_top,
                n_bottom,
            } => {
                let delta = Self::capture_workflow_undo(snapshot);
                snapshot.view_bounds = [*h_left, *h_right, *n_top, *n_bottom];
                snapshot.status_msg = format!(
                    "Queued viewport bounds H=[{h_left:.2}, {h_right:.2}], N=[{n_top:.2}, {n_bottom:.2}]"
                );
                delta
            }
            McpUiCommand::ExportResults {
                prefix,
                format,
                overwrite: _,
            } => {
                let delta = Self::capture_workflow_undo(snapshot);
                snapshot.status_msg = format!("Queued export ({format} -> {prefix})");
                delta
            }
            McpUiCommand::SetLockResidue { seq_id, locked } => {
                let delta = Self::capture_workflow_undo(snapshot);
                snapshot.status_msg = format!(
                    "Queued residue {} ({seq_id})",
                    if *locked { "lock" } else { "unlock" }
                );
                delta
            }
            McpUiCommand::DispatchWorkbenchAction { action }
            | McpUiCommand::DispatchGuardedWorkbenchAction { action, .. } => {
                let delta = Self::capture_workflow_undo(snapshot);
                snapshot.status_msg =
                    format!("Queued workbench action: {}", action.metadata().title);
                delta
            }
        }
    }

    fn capture_workflow_undo(snapshot: &WorkbenchSnapshot) -> UndoDelta {
        UndoDelta::WorkflowAction {
            prev_status_msg: snapshot.status_msg.clone(),
            prev_picker_snr: snapshot.picker_snr,
            prev_picker_mode: snapshot.picker_mode.clone(),
            prev_solver_kind: snapshot.solver_kind.clone(),
            prev_active_page_index: snapshot.active_page_index,
            prev_contour_base: snapshot.contour_base,
            prev_view_bounds: snapshot.view_bounds,
        }
    }

    fn apply_delta_inverse(snapshot: &mut WorkbenchSnapshot, delta: &UndoDelta) {
        match delta {
            UndoDelta::JumpToResidue {
                prev_active_residue_seq,
                prev_selected_pdb_chain_res,
            } => {
                snapshot.active_residue_seq = *prev_active_residue_seq;
                snapshot.selected_pdb_chain_res = prev_selected_pdb_chain_res.clone();
            }
            UndoDelta::SetCrosshairPpm { prev_crosshair_ppm } => {
                snapshot.crosshair_ppm = *prev_crosshair_ppm;
            }
            UndoDelta::AssignSpinSystemToResidue {
                seq_id,
                prev_at_target,
                displaced_source,
            } => {
                match prev_at_target {
                    Some(prev_rec) => {
                        snapshot.assigned_residues.insert(*seq_id, prev_rec.clone());
                    }
                    None => {
                        snapshot.assigned_residues.remove(seq_id);
                    }
                }
                if let Some((other_seq, other_rec)) = displaced_source {
                    snapshot
                        .assigned_residues
                        .insert(*other_seq, other_rec.clone());
                }
            }
            UndoDelta::UnassignResidue {
                seq_id,
                prev_record,
            } => {
                if let Some(rec) = prev_record {
                    snapshot.assigned_residues.insert(*seq_id, rec.clone());
                } else {
                    snapshot.assigned_residues.remove(seq_id);
                }
            }
            UndoDelta::TogglePeakExclusion {
                peak_id,
                prev_excluded,
            } => {
                if *prev_excluded {
                    snapshot.excluded_peaks.insert(*peak_id);
                } else {
                    snapshot.excluded_peaks.remove(peak_id);
                }
            }
            UndoDelta::HighlightResidues3d { prev_highlighted } => {
                snapshot.highlighted_residues_3d = prev_highlighted.clone();
            }
            UndoDelta::WorkflowAction {
                prev_status_msg,
                prev_picker_snr,
                prev_picker_mode,
                prev_solver_kind,
                prev_active_page_index,
                prev_contour_base,
                prev_view_bounds,
            } => {
                snapshot.status_msg = prev_status_msg.clone();
                snapshot.picker_snr = *prev_picker_snr;
                snapshot.picker_mode = prev_picker_mode.clone();
                snapshot.solver_kind = prev_solver_kind.clone();
                snapshot.active_page_index = *prev_active_page_index;
                snapshot.contour_base = *prev_contour_base;
                snapshot.view_bounds = *prev_view_bounds;
            }
        }
    }

    fn sync_depths(&self, snapshot: &mut WorkbenchSnapshot) {
        snapshot.undo_depth = self.undo_stack.len();
        snapshot.redo_depth = self.redo_stack.len();
    }

    fn enqueue_ui_command(&mut self, cmd: McpUiCommand) {
        if self.pending_ui_commands.len() >= MAX_MCP_UNDO_DEPTH {
            self.pending_ui_commands.pop_front();
        }
        self.pending_ui_commands.push_back(cmd);
    }

    fn push_audit(&mut self, outcome: McpAuditOutcome, detail: String) {
        let seq = self.next_audit_seq;
        self.next_audit_seq = self.next_audit_seq.saturating_add(1);
        if self.audit_log.len() >= MAX_MCP_UNDO_DEPTH {
            self.audit_log.pop_front();
        }
        self.audit_log.push_back(McpAuditRecord {
            seq,
            outcome,
            detail,
        });
    }
}

// ============================================================================
// 2. Pure-Rust Lossless RFC 2083 PNG + Adler-32 + IEEE CRC-32 + RFC 4648 Base64
// ============================================================================

const CRC32_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut c = i as u32;
        let mut bit = 0;
        while bit < 8 {
            c = if (c & 1) != 0 {
                0xEDB8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
            bit += 1;
        }
        table[i] = c;
        i += 1;
    }
    table
};

/// Compute IEEE 802.3 / RFC 2083 CRC-32 (`0xEDB88320` reflected polynomial).
pub fn crc32_ieee(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in bytes {
        let idx = ((crc ^ u32::from(b)) & 0xFF) as usize;
        crc = CRC32_TABLE[idx] ^ (crc >> 8);
    }
    !crc
}

/// Compute RFC 1950 §8.2 Adler-32 checksum (`MOD_ADLER = 65521`).
pub fn adler32_rfc1950(bytes: &[u8]) -> u32 {
    const MOD_ADLER: u32 = 65_521;
    let mut s1 = 1u32;
    let mut s2 = 0u32;
    for chunk in bytes.chunks(5552) {
        for &b in chunk {
            s1 += u32::from(b);
            s2 += s1;
        }
        s1 %= MOD_ADLER;
        s2 %= MOD_ADLER;
    }
    (s2 << 16) | s1
}

fn write_png_chunk(out: &mut Vec<u8>, chunk_type: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let crc_start = out.len();
    out.extend_from_slice(chunk_type);
    out.extend_from_slice(data);
    let crc = crc32_ieee(&out[crc_start..]);
    out.extend_from_slice(&crc.to_be_bytes());
}

/// Encode an 8-bit RGBA framebuffer (`width * height * 4` bytes) into a lossless RFC 2083 `PNG`
/// byte vector (`\x89PNG\r\n\x1a\n` signature, `IHDR`, chunked stored-DEFLATE `IDAT` with
/// RFC 1950 Adler-32 and IEEE CRC-32, and `IEND`), compiling with zero external crates on both
/// native desktop and `wasm32-unknown-unknown`.
pub fn encode_rgba_to_png(width: u32, height: u32, rgba_bytes: &[u8]) -> Vec<u8> {
    let w = width.max(1);
    let h = height.max(1);
    let row_bytes = (w as usize).saturating_mul(4);
    let expected_rgba = row_bytes.saturating_mul(h as usize);

    // Build filter-0 (`None`) scanlines: [0x00 || row_rgba] * height
    let raw_len = (h as usize).saturating_mul(1 + row_bytes);
    let mut raw = Vec::with_capacity(raw_len);
    for y in 0..(h as usize) {
        raw.push(0x00);
        let src_start = y.saturating_mul(row_bytes);
        if src_start < rgba_bytes.len() {
            let src_end = (src_start + row_bytes).min(rgba_bytes.len());
            let slice = &rgba_bytes[src_start..src_end];
            raw.extend_from_slice(slice);
            if slice.len() < row_bytes {
                raw.resize(raw.len() + (row_bytes - slice.len()), 0);
            }
        } else {
            let _ = expected_rgba;
            raw.resize(raw.len() + row_bytes, 0);
        }
    }

    // Wrap raw filtered scanlines in RFC 1950 ZLIB + RFC 1951 stored DEFLATE (`<= 65,535` B blocks)
    let num_blocks = raw.len().div_ceil(65_535).max(1);
    let mut zlib = Vec::with_capacity(2 + num_blocks * 5 + raw.len() + 4);
    zlib.extend_from_slice(&[0x78, 0x01]); // CMF=0x78 (deflate, 32K), FLG=0x01 (30721 % 31 == 0)
    let chunks: Vec<&[u8]> = raw.chunks(65_535).collect();
    for (i, block) in chunks.iter().enumerate() {
        let is_final = i + 1 == chunks.len();
        zlib.push(u8::from(is_final)); // BFINAL = 0/1, BTYPE = 00 (stored)
        let len = block.len() as u16;
        zlib.extend_from_slice(&len.to_le_bytes());
        zlib.extend_from_slice(&(!len).to_le_bytes());
        zlib.extend_from_slice(block);
    }
    zlib.extend_from_slice(&adler32_rfc1950(&raw).to_be_bytes());

    // Assemble RFC 2083 PNG stream: signature + IHDR + IDAT + IEND
    let mut png = Vec::with_capacity(8 + 25 + (12 + zlib.len()) + 12);
    png.extend_from_slice(&PNG_SIGNATURE);

    let mut ihdr = [0u8; 13];
    ihdr[0..4].copy_from_slice(&w.to_be_bytes());
    ihdr[4..8].copy_from_slice(&h.to_be_bytes());
    ihdr[8] = 8; // 8-bit sample depth
    ihdr[9] = 6; // color type 6 = Truecolour with alpha (RGBA)
    ihdr[10] = 0; // compression 0 = deflate/inflate
    ihdr[11] = 0; // filter 0 = adaptive 5-filter
    ihdr[12] = 0; // interlace 0 = non-interlaced
    write_png_chunk(&mut png, b"IHDR", &ihdr);
    write_png_chunk(&mut png, b"IDAT", &zlib);
    write_png_chunk(&mut png, b"IEND", &[]);
    png
}

const BASE64_ALPHABET: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Encode arbitrary binary bytes using standard RFC 4648 §4 Base64 (`+`, `/`, `=` padding).
pub fn encode_base64(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for triple in bytes.chunks(3) {
        let b0 = u32::from(triple[0]);
        let b1 = if triple.len() > 1 {
            u32::from(triple[1])
        } else {
            0
        };
        let b2 = if triple.len() > 2 {
            u32::from(triple[2])
        } else {
            0
        };
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(BASE64_ALPHABET[((n >> 18) & 0x3F) as usize] as char);
        out.push(BASE64_ALPHABET[((n >> 12) & 0x3F) as usize] as char);
        if triple.len() > 1 {
            out.push(BASE64_ALPHABET[((n >> 6) & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
        if triple.len() > 2 {
            out.push(BASE64_ALPHABET[(n & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

/// Decode standard RFC 4648 §4 Base64 into raw bytes.
pub fn decode_base64(encoded: &str) -> Result<Vec<u8>, String> {
    let clean: Vec<u8> = encoded
        .bytes()
        .filter(|b| !b.is_ascii_whitespace())
        .collect();
    if clean.is_empty() {
        return Ok(Vec::new());
    }
    if !clean.len().is_multiple_of(4) {
        return Err("Base64 input length must be a multiple of 4".to_string());
    }
    let val_of = |b: u8| -> Result<u32, String> {
        match b {
            b'A'..=b'Z' => Ok(u32::from(b - b'A')),
            b'a'..=b'z' => Ok(u32::from(b - b'a') + 26),
            b'0'..=b'9' => Ok(u32::from(b - b'0') + 52),
            b'+' => Ok(62),
            b'/' => Ok(63),
            _ => Err(format!("invalid Base64 byte 0x{b:02x}")),
        }
    };
    let mut out = Vec::with_capacity((clean.len() / 4) * 3);
    for (chunk_idx, quad) in clean.chunks_exact(4).enumerate() {
        let is_last = (chunk_idx + 1) * 4 == clean.len();
        let pad = match (quad[2] == b'=', quad[3] == b'=') {
            (true, true) if is_last => 2,
            (false, true) if is_last => 1,
            (false, false) => 0,
            _ => return Err("malformed Base64 '=' padding".to_string()),
        };
        if quad[0] == b'=' || quad[1] == b'=' {
            return Err("malformed Base64 leading '=' padding".to_string());
        }
        let v0 = val_of(quad[0])?;
        let v1 = val_of(quad[1])?;
        let v2 = if pad >= 2 { 0 } else { val_of(quad[2])? };
        let v3 = if pad >= 1 { 0 } else { val_of(quad[3])? };
        let n = (v0 << 18) | (v1 << 12) | (v2 << 6) | v3;
        out.push(((n >> 16) & 0xFF) as u8);
        if pad < 2 {
            out.push(((n >> 8) & 0xFF) as u8);
        }
        if pad < 1 {
            out.push((n & 0xFF) as u8);
        }
    }
    Ok(out)
}

/// Build a spec-compliant MCP multimodal `image/png` content block (`{ "type": "image",
/// "mimeType": "image/png", "data": "<base64>" }`) from an 8-bit RGBA pixel buffer.
pub fn build_mcp_png_image_block(width: u32, height: u32, rgba_bytes: &[u8]) -> Value {
    let png = encode_rgba_to_png(width, height, rgba_bytes);
    json!({
        "type": "image",
        "mimeType": "image/png",
        "data": encode_base64(&png),
    })
}

/// Retained viewport RGBA framebuffer ready for instant `< 1 ms` MCP `capture_viewport_png` calls.
#[derive(Clone, Debug)]
pub struct ViewportFrameBuffer {
    /// Framebuffer width in pixels (`>= 1`).
    pub width: u32,
    /// Framebuffer height in pixels (`>= 1`).
    pub height: u32,
    /// Tightly packed row-major 8-bit RGBA pixels (`width * height * 4` bytes).
    pub rgba: Vec<u8>,
    /// Logical viewport source tag (`"contour_2d"`, `"strip_3d"`, `"mol3d"`, `"diagnostic"`).
    pub viewport_label: String,
}

impl ViewportFrameBuffer {
    /// Synthesize a crisp dark-theme 2D HSQC contour + crosshair diagnostic RGBA buffer when no
    /// live `wgpu` readback frame has been uploaded yet.
    pub fn diagnostic_card(width: u32, height: u32, crosshair: (f64, f64, Option<f64>)) -> Self {
        let w = width.max(1);
        let h = height.max(1);
        let mut rgba = vec![0u8; (w as usize) * (h as usize) * 4];
        // Map amide H in [6.0, 10.5] ppm and N in [105.0, 135.0] ppm onto viewport pixels
        let xh_px = (((10.5 - crosshair.0) / 4.5).clamp(0.05, 0.95) * f64::from(w - 1)) as u32;
        let xh_py = (((crosshair.1 - 105.0) / 30.0).clamp(0.05, 0.95) * f64::from(h - 1)) as u32;

        for y in 0..h {
            for x in 0..w {
                let idx = ((y * w + x) as usize) * 4;
                let on_border = x == 0 || y == 0 || x + 1 == w || y + 1 == h;
                let on_crosshair = x == xh_px || y == xh_py;
                let dx = (x as i32) - (xh_px as i32);
                let dy = (y as i32) - (xh_py as i32);
                let r2 = dx * dx + dy * dy;
                let pixel = if on_border {
                    [58, 72, 96, 255]
                } else if on_crosshair {
                    [245, 194, 66, 255]
                } else if (12..=28).contains(&r2) || (48..=68).contains(&r2) {
                    [46, 196, 246, 255]
                } else {
                    [16, 20, 28, 255]
                };
                rgba[idx..idx + 4].copy_from_slice(&pixel);
            }
        }
        Self {
            width: w,
            height: h,
            rgba,
            viewport_label: "diagnostic_contour_2d".to_string(),
        }
    }
}

// ============================================================================
// 3. Constant-Time Bearer Auth, DNS-Rebinding Guards & MCP JSON-RPC Engine
// ============================================================================

/// Constant-time byte slice equality check (`O(max(expected.len(), provided.len()))` without
/// early-exit branches on length or first mismatching byte).
pub fn constant_time_eq(expected: &[u8], provided: &[u8]) -> bool {
    let len_diff = expected.len() ^ provided.len();
    let max_len = expected.len().max(provided.len());
    let mut acc = 0u8;
    for i in 0..max_len {
        let a = expected.get(i).copied().unwrap_or(0);
        let b = provided.get(i).copied().unwrap_or(0);
        acc |= a ^ b;
    }
    let combined = len_diff | usize::from(acc);
    std::hint::black_box(combined) == 0
}

#[cfg(unix)]
fn fill_os_entropy(raw: &mut [u8; 32]) -> bool {
    if let Ok(mut f) = std::fs::File::open("/dev/urandom") {
        use std::io::Read as _;
        if f.read_exact(raw).is_ok() && raw.iter().any(|&b| b != 0) {
            return true;
        }
    }
    false
}

#[cfg(windows)]
fn fill_os_entropy(raw: &mut [u8; 32]) -> bool {
    const BCRYPT_USE_SYSTEM_PREFERRED_RNG: u32 = 0x0000_0002;
    #[link(name = "bcrypt")]
    #[allow(non_snake_case)]
    extern "system" {
        fn BCryptGenRandom(
            hAlgorithm: *mut core::ffi::c_void,
            pbBuffer: *mut u8,
            cbBuffer: u32,
            dwFlags: u32,
        ) -> i32;
    }
    // SAFETY: `pbBuffer` points to a valid, mutable 32-byte stack array (`cbBuffer = 32`),
    // `hAlgorithm` is NULL as required when `BCRYPT_USE_SYSTEM_PREFERRED_RNG` (`0x00000002`)
    // is passed per the Windows CNG `BCryptGenRandom` contract, and `status == 0`
    // (`STATUS_SUCCESS`) plus non-zero output bytes are verified before returning `true`.
    let status = unsafe {
        BCryptGenRandom(
            core::ptr::null_mut(),
            raw.as_mut_ptr(),
            raw.len() as u32,
            BCRYPT_USE_SYSTEM_PREFERRED_RNG,
        )
    };
    status == 0 && raw.iter().any(|&b| b != 0)
}

#[cfg(not(any(unix, windows)))]
fn fill_os_entropy(_raw: &mut [u8; 32]) -> bool {
    false
}

fn format_bearer_token_hex(raw: &[u8; 32]) -> String {
    let mut out = String::with_capacity(8 + 64);
    out.push_str("nmr_mcp_");
    for &b in raw {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0F) as usize] as char);
    }
    out
}

fn try_generate_bearer_token_with(fill_fn: fn(&mut [u8; 32]) -> bool) -> Result<String, String> {
    let mut raw = [0u8; 32];
    if !fill_fn(&mut raw) || !raw.iter().any(|&b| b != 0) {
        return Err(
            "OS CSPRNG (/dev/urandom or BCryptGenRandom) failed to supply 256 bits of non-zero entropy; refusing to start or authenticate TCP MCP bridge"
                .to_string(),
        );
    }
    Ok(format_bearer_token_hex(&raw))
}

#[cfg(not(any(unix, windows)))]
fn wasm_preview_display_token() -> String {
    let mut raw = [0u8; 32];
    let seq = TOKEN_COUNTER.fetch_add(1, Ordering::Relaxed);
    for (i, chunk) in raw.chunks_exact_mut(8).enumerate() {
        let rs = RandomState::new();
        let mut h = rs.build_hasher();
        h.write_u64(
            seq.wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(i as u64),
        );
        chunk.copy_from_slice(&h.finish().to_le_bytes());
    }
    format_bearer_token_hex(&raw)
}

/// Generate a fresh `nmr_mcp_<64-hex>` loopback bearer token (`72` ASCII chars total), failing
/// closed (`Err(...)`) whenever OS CSPRNG entropy is unavailable:
///
/// - **Native Unix (`#[cfg(unix)]`)**: draws 256 bits (`32` non-zero-verified bytes) from the
///   kernel CSPRNG via `/dev/urandom`, returning `Err(...)` if `/dev/urandom` cannot be read.
/// - **Native Windows (`#[cfg(windows)]`)**: draws 256 bits (`32` non-zero-verified bytes) from
///   the Windows CNG system CSPRNG via `BCryptGenRandom(..., BCRYPT_USE_SYSTEM_PREFERRED_RNG)`
///   (`bcrypt.dll`, checking `NTSTATUS == 0` and non-zero output bytes), returning `Err(...)` on failure.
/// - **Non-Unix / non-Windows targets (`#[cfg(not(any(unix, windows)))]`, e.g. `wasm32`)**:
///   returns `Err(...)` (a `RandomState` preview token is used exclusively for offline UI display on
///   `wasm32`, where `start_bridge` unconditionally disables TCP socket binding).
pub fn generate_bearer_token() -> Result<String, String> {
    try_generate_bearer_token_with(fill_os_entropy)
}

/// Mask a bearer token for safe on-screen display (`nmr_mcp_abcd••••••••••••ef01`).
pub fn mask_bearer_token(token: &str) -> String {
    if token.len() <= 16 {
        return "••••••••••••".to_string();
    }
    let prefix = &token[..12];
    let suffix = &token[token.len() - 4..];
    format!("{prefix}••••••••••••{suffix}")
}

/// Target desktop AI assistant client for 1-click *"Copy MCP Config"* JSON generation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum McpClientTarget {
    #[default]
    Antigravity,
    ClaudeDesktop,
    Cursor,
}

impl McpClientTarget {
    /// Ordered list of supported client configuration presets.
    pub const ALL: &'static [Self] = &[Self::Antigravity, Self::ClaudeDesktop, Self::Cursor];

    /// Human-readable UI selector label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Antigravity => "Antigravity",
            Self::ClaudeDesktop => "Claude Desktop",
            Self::Cursor => "Cursor",
        }
    }
    /// Map to the shared `nmr_core::mcp::McpClientTarget` enum for unified schema generation.
    pub const fn to_core_target(self) -> nmr_core::mcp::McpClientTarget {
        match self {
            Self::Antigravity => nmr_core::mcp::McpClientTarget::Antigravity,
            Self::ClaudeDesktop => nmr_core::mcp::McpClientTarget::ClaudeDesktop,
            Self::Cursor => nmr_core::mcp::McpClientTarget::Cursor,
        }
    }
}

/// Build ready-to-paste client MCP configuration JSON for Antigravity, Claude Desktop, or Cursor
/// using the canonical `nmr_core::mcp::generate_mcp_client_config_pretty` schema (`"ferric-workbench"`,
/// `"transport": "streamable-http"` for Claude Desktop, `"serverUrl"` for Antigravity, `"url"` for Cursor).
pub fn build_mcp_client_config_json(
    client: McpClientTarget,
    endpoint_url: &str,
    bearer_token: &str,
) -> String {
    build_mcp_client_config_json_with_permission(
        client,
        endpoint_url,
        bearer_token,
        McpBridgePermission::ReadOnly,
    )
}

/// Build ready-to-paste client MCP configuration JSON with explicit [`McpBridgePermission`] mode,
/// delegating directly to [`nmr_core::mcp::generate_mcp_client_config_pretty`] for 100% parity
/// with `ferric mcp-config --bridge-port`.
pub fn build_mcp_client_config_json_with_permission(
    client: McpClientTarget,
    endpoint_url: &str,
    bearer_token: &str,
    permission: McpBridgePermission,
) -> String {
    let access_mode = match permission {
        McpBridgePermission::ReadOnly => nmr_core::mcp::McpAccessMode::ReadOnly,
        McpBridgePermission::ReadWrite => nmr_core::mcp::McpAccessMode::ReadWrite,
    };
    let mode = nmr_core::mcp::McpClientMode::HttpBridge {
        endpoint_url: endpoint_url.trim().to_string(),
        bearer_token: bearer_token.trim().to_string(),
        access_mode,
    };
    nmr_core::mcp::generate_mcp_client_config_pretty(client.to_core_target(), &mode).unwrap_or_else(
        |err| {
            serde_json::to_string_pretty(&json!({
                "mcpServers": {},
                "error": format!("failed to build MCP client config: {err}")
            }))
            .unwrap_or_else(|_| {
                "{\"mcpServers\":{},\"error\":\"failed to build MCP client config\"}".to_string()
            })
        },
    )
}

/// Validate `Host` and optional browser `Origin` headers to prevent DNS-rebinding and browser CSRF
/// attacks against the loopback desktop bridge:
/// - Rejects missing or duplicate `Host` headers.
/// - Requires `Host` hostname to be strictly `127.0.0.1` or `localhost` (with `:port` matching
///   `bound_port` whenever specified).
/// - Rejects any cross-origin browser `Origin` header (including `Origin: null` and foreign ports).
pub fn validate_loopback_host_and_origin(
    host: &str,
    duplicate_host: bool,
    origin: Option<&str>,
    bound_port: u16,
) -> Result<(), &'static str> {
    let trimmed_host = host.trim().to_ascii_lowercase();
    if duplicate_host || trimmed_host.is_empty() {
        return Err("missing or duplicate Host header");
    }

    let (hostname, port_opt) = match trimmed_host.split_once(':') {
        Some((h, p_str)) => {
            if p_str.is_empty() || !p_str.bytes().all(|b| b.is_ascii_digit()) {
                return Err("malformed Host port");
            }
            let port = p_str.parse::<u16>().map_err(|_| "invalid Host port")?;
            (h, Some(port))
        }
        None => (trimmed_host.as_str(), None),
    };

    if !matches!(hostname, "127.0.0.1" | "localhost") {
        return Err("non-loopback Host header rejected (DNS-rebinding protection)");
    }
    if bound_port != 0 {
        if let Some(p) = port_opt {
            if p != bound_port {
                return Err("Host port mismatch against bound loopback port");
            }
        }
    }

    if let Some(raw_origin) = origin {
        let norm = raw_origin.trim().trim_end_matches('/').to_ascii_lowercase();
        if norm.is_empty() || norm == "null" {
            return Err("untrusted null/empty Origin header rejected");
        }
        let allowed_v4 = if bound_port == 0 {
            "http://127.0.0.1".to_string()
        } else {
            format!("http://127.0.0.1:{bound_port}")
        };
        let allowed_local = if bound_port == 0 {
            "http://localhost".to_string()
        } else {
            format!("http://localhost:{bound_port}")
        };
        if norm != allowed_v4 && norm != allowed_local {
            return Err("cross-origin browser Origin header rejected (CSRF protection)");
        }
    }
    Ok(())
}

/// Extract and constant-time verify `Authorization: Bearer <token>`.
pub fn verify_bearer_authorization(auth_header: Option<&str>, expected_token: &str) -> bool {
    let Some(raw) = auth_header else {
        return false;
    };
    let trimmed = raw.trim();
    let Some((scheme, candidate)) = trimmed.split_once(char::is_whitespace) else {
        return false;
    };
    if !scheme.eq_ignore_ascii_case("bearer") {
        return false;
    }
    let candidate = candidate.trim();
    if expected_token.is_empty() {
        return false;
    }
    constant_time_eq(expected_token.as_bytes(), candidate.as_bytes())
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Debug)]
enum SandboxedWriteCheckError {
    Security(McpSecurityError),
    OverwriteRefused { display_name: String },
}

#[cfg(not(target_arch = "wasm32"))]
impl SandboxedWriteCheckError {
    fn error_kind(&self) -> &'static str {
        match self {
            Self::Security(sec) => sec.kind.as_str(),
            Self::OverwriteRefused { .. } => "overwrite_refused",
        }
    }

    fn rpc_code(&self) -> i32 {
        match self {
            Self::Security(sec) => sec.kind.rpc_code(),
            Self::OverwriteRefused { .. } => nmr_core::mcp::ERR_SECURITY_VIOLATION,
        }
    }

    fn message(&self) -> String {
        match self {
            Self::Security(sec) => sec.message.clone(),
            Self::OverwriteRefused { display_name } => format!(
                "refusing silent overwrite of existing file '{display_name}'; pass overwrite: true to replace it"
            ),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
const FORBIDDEN_CREDENTIAL_SEGMENTS: &[&str] = &[
    ".ssh",
    ".aws",
    ".gnupg",
    ".env",
    ".netrc",
    ".kube",
    ".git-credentials",
    ".docker",
    ".npmrc",
    ".pypirc",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
    "id_dsa",
];

#[cfg(not(target_arch = "wasm32"))]
fn is_forbidden_credential_seg(seg: &str) -> bool {
    let lower = seg.to_ascii_lowercase();
    FORBIDDEN_CREDENTIAL_SEGMENTS.contains(&lower.as_str()) || lower.starts_with(".env.")
}

#[cfg(not(target_arch = "wasm32"))]
fn has_forbidden_uri_scheme(raw: &str) -> bool {
    let trimmed = raw.trim();
    if trimmed.contains("://") {
        return true;
    }
    if let Some((scheme, _rest)) = trimmed.split_once(':') {
        if (2..=24).contains(&scheme.len()) {
            let bytes = scheme.as_bytes();
            if bytes[0].is_ascii_alphabetic()
                && bytes
                    .iter()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'+' | b'-' | b'.'))
            {
                return true;
            }
        }
    }
    false
}

#[cfg(not(target_arch = "wasm32"))]
fn validate_raw_prefix(raw_prefix: &str) -> Result<PathBuf, McpSecurityError> {
    if raw_prefix.contains('\0') {
        return Err(McpSecurityError::new(
            McpSecurityErrorKind::NullByteOrInvalidPath,
            "export prefix contains a forbidden null byte",
        ));
    }
    let trimmed = raw_prefix.trim();
    if trimmed.is_empty() {
        return Err(McpSecurityError::new(
            McpSecurityErrorKind::NullByteOrInvalidPath,
            "export prefix must not be empty",
        ));
    }
    if has_forbidden_uri_scheme(trimmed) {
        return Err(McpSecurityError::new(
            McpSecurityErrorKind::NullByteOrInvalidPath,
            "URI schemes are forbidden in MCP export paths",
        ));
    }
    if trimmed.ends_with('/') || trimmed.ends_with('\\') {
        return Err(McpSecurityError::new(
            McpSecurityErrorKind::NullByteOrInvalidPath,
            "export prefix must name a file prefix, not a directory",
        ));
    }
    let path = Path::new(trimmed);
    for comp in path.components() {
        match comp {
            Component::ParentDir => {
                return Err(McpSecurityError::new(
                    McpSecurityErrorKind::ParentTraversal,
                    "relative parent traversal ('..') is forbidden in MCP export paths",
                ));
            }
            Component::Normal(os) => {
                let s = os.to_str().ok_or_else(|| {
                    McpSecurityError::new(
                        McpSecurityErrorKind::NullByteOrInvalidPath,
                        "non-UTF-8 path segment is forbidden",
                    )
                })?;
                if is_forbidden_credential_seg(s) {
                    return Err(McpSecurityError::new(
                        McpSecurityErrorKind::CredentialPathBlocked,
                        format!("access to sensitive credential path segment '{s}' is forbidden"),
                    ));
                }
            }
            Component::CurDir | Component::RootDir | Component::Prefix(_) => {}
        }
    }
    let Some(fname) = path.file_name().and_then(|s| s.to_str()) else {
        return Err(McpSecurityError::new(
            McpSecurityErrorKind::NullByteOrInvalidPath,
            "export prefix must include a valid file stem",
        ));
    };
    if fname.is_empty() || fname == "." || fname == ".." {
        return Err(McpSecurityError::new(
            McpSecurityErrorKind::NullByteOrInvalidPath,
            "export prefix must include a valid file stem",
        ));
    }
    Ok(path.to_path_buf())
}

#[cfg(not(target_arch = "wasm32"))]
fn detected_home_dirs() -> Vec<PathBuf> {
    let mut homes = Vec::new();
    let mut push_candidate = |raw: PathBuf| {
        if raw.as_os_str().is_empty() {
            return;
        }
        if let Ok(canon) = std::fs::canonicalize(&raw) {
            if !homes.contains(&canon) {
                homes.push(canon);
            }
        } else if !homes.contains(&raw) {
            homes.push(raw);
        }
    };
    for key in ["HOME", "USERPROFILE"] {
        if let Some(val) = std::env::var_os(key) {
            push_candidate(PathBuf::from(val));
        }
    }
    if let (Some(drive), Some(path)) = (std::env::var_os("HOMEDRIVE"), std::env::var_os("HOMEPATH"))
    {
        let mut combined = drive;
        combined.push(path);
        push_candidate(PathBuf::from(combined));
    }
    homes
}

#[cfg(not(target_arch = "wasm32"))]
fn is_home_or_ancestor_of(canon: &Path, homes: &[PathBuf]) -> bool {
    homes.iter().any(|home| home.starts_with(canon))
}

#[cfg(not(target_arch = "wasm32"))]
fn is_home_or_ancestor_of_home(canon: &Path) -> bool {
    let homes = detected_home_dirs();
    is_home_or_ancestor_of(canon, &homes)
}

#[cfg(not(target_arch = "wasm32"))]
fn canonicalize_approved_root(raw: &Path) -> Result<PathBuf, String> {
    let canon = std::fs::canonicalize(raw).map_err(|e| {
        format!(
            "failed to canonicalize root directory '{}': {e}",
            raw.display()
        )
    })?;
    if !canon.is_dir() {
        return Err(format!(
            "approved MCP root '{}' must be an existing directory",
            canon.display()
        ));
    }
    if canon.parent().is_none() {
        return Err(
            "refusing filesystem root ('/' or drive root) as an MCP sandbox root".to_string(),
        );
    }
    Ok(canon)
}

#[cfg(not(target_arch = "wasm32"))]
fn canonicalize_automatic_root_with_homes(
    raw: &Path,
    homes: &[PathBuf],
) -> Result<PathBuf, String> {
    let canon = canonicalize_approved_root(raw)?;
    if is_home_or_ancestor_of(&canon, homes) {
        return Err(format!(
            "refusing home directory or ancestor ('{}') as an automatic MCP sandbox root; add an explicit root in GUI MCP Bridge settings if intended",
            canon.display()
        ));
    }
    Ok(canon)
}

#[cfg(not(target_arch = "wasm32"))]
fn canonicalize_automatic_root(raw: &Path) -> Result<PathBuf, String> {
    let homes = detected_home_dirs();
    canonicalize_automatic_root_with_homes(raw, &homes)
}

#[cfg(not(target_arch = "wasm32"))]
fn default_cwd_roots() -> Vec<PathBuf> {
    let Ok(cwd) = std::env::current_dir() else {
        return Vec::new();
    };
    match canonicalize_automatic_root(&cwd) {
        Ok(canon) => vec![canon],
        Err(_) => Vec::new(),
    }
}

#[cfg(not(target_arch = "wasm32"))]
const MAX_STUDY_TREE_SCAN_DEPTH: usize = 24;
#[cfg(not(target_arch = "wasm32"))]
const MAX_STUDY_TREE_SCAN_ENTRIES: usize = 50_000;

/// Recursively inspect every child entry inside `canon_study_root` without following directory
/// symlinks into cycles, refusing any symlink that is dangling or resolves outside
/// `canon_study_root` / `approved_roots`, and blocking forbidden credential segments.
#[cfg(not(target_arch = "wasm32"))]
fn validate_study_folder_tree_containment(
    canon_study_root: &Path,
    approved_roots: &[PathBuf],
) -> Result<(), McpSecurityError> {
    let mut stack: Vec<(PathBuf, usize)> = vec![(canon_study_root.to_path_buf(), 0)];
    let mut visited = 0usize;

    while let Some((dir, depth)) = stack.pop() {
        if depth > MAX_STUDY_TREE_SCAN_DEPTH {
            return Err(McpSecurityError::new(
                McpSecurityErrorKind::NullByteOrInvalidPath,
                format!(
                    "study folder exceeds maximum directory nesting depth ({MAX_STUDY_TREE_SCAN_DEPTH})"
                ),
            ));
        }
        let entries = std::fs::read_dir(&dir).map_err(|e| {
            McpSecurityError::new(
                McpSecurityErrorKind::IoError,
                format!("failed to scan study folder contents: {e}"),
            )
        })?;
        for entry_res in entries {
            visited = visited.saturating_add(1);
            if visited > MAX_STUDY_TREE_SCAN_ENTRIES {
                return Err(McpSecurityError::new(
                    McpSecurityErrorKind::NullByteOrInvalidPath,
                    format!(
                        "study folder exceeds maximum scanned entry limit ({MAX_STUDY_TREE_SCAN_ENTRIES})"
                    ),
                ));
            }
            let entry = entry_res.map_err(|e| {
                McpSecurityError::new(
                    McpSecurityErrorKind::IoError,
                    format!("failed to read study folder entry: {e}"),
                )
            })?;
            let entry_path = entry.path();
            if let Some(name) = entry.file_name().to_str() {
                if is_forbidden_credential_seg(name) {
                    return Err(McpSecurityError::new(
                        McpSecurityErrorKind::CredentialPathBlocked,
                        format!(
                            "study folder contains forbidden sensitive credential segment '{name}'"
                        ),
                    ));
                }
            }
            let meta = std::fs::symlink_metadata(&entry_path).map_err(|e| {
                McpSecurityError::new(
                    McpSecurityErrorKind::IoError,
                    format!("failed to inspect study folder entry metadata: {e}"),
                )
            })?;
            if meta.file_type().is_symlink() {
                let canon_target = std::fs::canonicalize(&entry_path).map_err(|_| {
                    McpSecurityError::new(
                        McpSecurityErrorKind::SymlinkEscape,
                        "dangling or unresolvable nested symlink inside study folder is forbidden",
                    )
                })?;
                if !canon_target.starts_with(canon_study_root)
                    || !approved_roots.iter().any(|r| canon_target.starts_with(r))
                {
                    return Err(McpSecurityError::new(
                        McpSecurityErrorKind::SymlinkEscape,
                        "nested symlink inside study folder escapes the study root",
                    ));
                }
                for comp in canon_target.components() {
                    if let Component::Normal(os) = comp {
                        if let Some(seg) = os.to_str() {
                            if is_forbidden_credential_seg(seg) {
                                return Err(McpSecurityError::new(
                                    McpSecurityErrorKind::CredentialPathBlocked,
                                    format!(
                                        "nested symlink targets forbidden credential segment '{seg}'"
                                    ),
                                ));
                            }
                        }
                    }
                }
            } else if meta.is_dir() {
                stack.push((entry_path, depth + 1));
            }
        }
    }
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn absolutize_against_cwd(p: &Path) -> Result<PathBuf, McpSecurityError> {
    if p.is_absolute() {
        Ok(p.to_path_buf())
    } else {
        let cwd = std::env::current_dir().map_err(|e| {
            McpSecurityError::new(
                McpSecurityErrorKind::IoError,
                format!("failed to determine current working directory: {e}"),
            )
        })?;
        Ok(cwd.join(p))
    }
}

struct SharedBridgeInner {
    snapshot: WorkbenchSnapshot,
    data_tables: WorkbenchDataTables,
    undo_log: McpUndoLog,
    viewport: ViewportFrameBuffer,
    permission: McpBridgePermission,
    allow_destructive_actions: bool,
    #[cfg(not(target_arch = "wasm32"))]
    allow_cwd_fallback: bool,
    #[cfg(not(target_arch = "wasm32"))]
    active_study_root: Option<PathBuf>,
    #[cfg(not(target_arch = "wasm32"))]
    approved_roots: Vec<PathBuf>,
    #[cfg(not(target_arch = "wasm32"))]
    roots_customized: bool,
    #[cfg(not(target_arch = "wasm32"))]
    last_mcp_loaded_path: Option<PathBuf>,
    #[cfg(not(target_arch = "wasm32"))]
    active_ingest_root: Option<PathBuf>,
    #[cfg(not(target_arch = "wasm32"))]
    current_export_prefix: String,
    bearer_token: String,
    bearer_token_csprng: bool,
    handled_requests: u64,
    repaint_ctx: Option<egui::Context>,
}

impl SharedBridgeInner {
    fn new(snapshot: WorkbenchSnapshot) -> Self {
        let xh = snapshot.crosshair_ppm;
        #[cfg(any(unix, windows))]
        let (bearer_token, bearer_token_csprng) = match generate_bearer_token() {
            Ok(tok) => (tok, true),
            Err(_) => (String::new(), false),
        };
        #[cfg(not(any(unix, windows)))]
        let (bearer_token, bearer_token_csprng) = (wasm_preview_display_token(), false);

        Self {
            snapshot,
            data_tables: WorkbenchDataTables::default(),
            undo_log: McpUndoLog::new(),
            viewport: ViewportFrameBuffer::diagnostic_card(64, 48, xh),
            permission: McpBridgePermission::ReadOnly,
            allow_destructive_actions: false,
            #[cfg(not(target_arch = "wasm32"))]
            allow_cwd_fallback: false,
            #[cfg(not(target_arch = "wasm32"))]
            active_study_root: None,
            #[cfg(not(target_arch = "wasm32"))]
            approved_roots: Vec::new(),
            #[cfg(not(target_arch = "wasm32"))]
            roots_customized: false,
            #[cfg(not(target_arch = "wasm32"))]
            last_mcp_loaded_path: None,
            #[cfg(not(target_arch = "wasm32"))]
            active_ingest_root: None,
            #[cfg(not(target_arch = "wasm32"))]
            current_export_prefix: "result_gui".to_string(),
            bearer_token,
            bearer_token_csprng,
            handled_requests: 0,
            repaint_ctx: None,
        }
    }

    fn request_ui_repaint(&self) {
        if let Some(ctx) = &self.repaint_ctx {
            ctx.request_repaint();
        }
    }

    /// Return `true` when setting a sequence would replace or clear existing sequence or
    /// assignment state (including a prior `SetSequence` already queued in this batch/frame).
    fn has_existing_sequence_or_assignments(&self) -> bool {
        self.snapshot.sequence_len > 0
            || self.snapshot.assigned_count > 0
            || !self.snapshot.assigned_residues.is_empty()
            || self.undo_log.has_pending_set_sequence()
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn effective_roots(&self) -> Vec<PathBuf> {
        if self.roots_customized {
            self.approved_roots.clone()
        } else if let Some(ref study) = self.active_study_root {
            if is_home_or_ancestor_of_home(study) {
                Vec::new()
            } else {
                vec![study.clone()]
            }
        } else if self.allow_cwd_fallback {
            default_cwd_roots()
        } else {
            Vec::new()
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn security_policy(&self, effective_permission: McpBridgePermission) -> McpSecurityPolicy {
        let mode = match effective_permission {
            McpBridgePermission::ReadOnly => McpAccessMode::ReadOnly,
            McpBridgePermission::ReadWrite => McpAccessMode::ReadWrite,
        };
        McpSecurityPolicy::new(self.effective_roots(), mode)
            .unwrap_or_else(|_| McpSecurityPolicy::empty_read_only())
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn resolve_sandboxed_read_path(
        &self,
        raw_path: &str,
        allow_dir: bool,
    ) -> Result<PathBuf, McpSecurityError> {
        let policy = self.security_policy(McpBridgePermission::ReadWrite);
        let resolved = policy.resolve_sandboxed_path(raw_path)?;
        let meta = std::fs::symlink_metadata(&resolved).map_err(|e| {
            McpSecurityError::new(
                McpSecurityErrorKind::IoError,
                format!("failed to inspect sandboxed path metadata: {e}"),
            )
        })?;
        if meta.file_type().is_symlink() {
            return Err(McpSecurityError::new(
                McpSecurityErrorKind::SymlinkEscape,
                "symlink target is forbidden in sandboxed MCP paths",
            ));
        }
        if allow_dir {
            if !meta.is_file() && !meta.is_dir() {
                return Err(McpSecurityError::new(
                    McpSecurityErrorKind::NullByteOrInvalidPath,
                    "sandboxed path must be a regular file or directory",
                ));
            }
        } else {
            if !meta.is_file() {
                return Err(McpSecurityError::new(
                    McpSecurityErrorKind::NullByteOrInvalidPath,
                    "sandboxed path must be a regular file",
                ));
            }
            if meta.len() > MAX_MCP_SEQUENCE_FILE_BYTES {
                return Err(McpSecurityError::new(
                    McpSecurityErrorKind::NullByteOrInvalidPath,
                    format!(
                        "file exceeds {} MiB MCP read limit",
                        MAX_MCP_SEQUENCE_FILE_BYTES / (1024 * 1024)
                    ),
                ));
            }
        }
        Ok(resolved)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn resolve_sandboxed_study_path(&self, raw_path: &str) -> Result<PathBuf, McpSecurityError> {
        let resolved = self.resolve_sandboxed_read_path(raw_path, true)?;
        if resolved.is_dir() {
            let roots = self.effective_roots();
            validate_study_folder_tree_containment(&resolved, &roots)?;
        }
        Ok(resolved)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn resolve_sandboxed_export_targets(
        &self,
        raw_prefix: &str,
        exts: &[&str],
        overwrite: bool,
        resolve_relative_against_cwd: bool,
        tool_name: &str,
    ) -> Result<Vec<PathBuf>, SandboxedWriteCheckError> {
        let base_p = validate_raw_prefix(raw_prefix).map_err(SandboxedWriteCheckError::Security)?;
        let policy = self.security_policy(McpBridgePermission::ReadWrite);
        let mut resolved_targets = Vec::with_capacity(exts.len());
        for ext in exts {
            let mut target = base_p.with_extension(ext);
            if resolve_relative_against_cwd && !target.is_absolute() {
                target =
                    absolutize_against_cwd(&target).map_err(SandboxedWriteCheckError::Security)?;
            }
            let target_str = target.to_str().ok_or_else(|| {
                SandboxedWriteCheckError::Security(McpSecurityError::new(
                    McpSecurityErrorKind::NullByteOrInvalidPath,
                    "export path must be valid UTF-8",
                ))
            })?;
            let resolved = policy
                .resolve_sandboxed_write_path(target_str, tool_name)
                .map_err(SandboxedWriteCheckError::Security)?;
            match std::fs::symlink_metadata(&resolved) {
                Ok(meta) => {
                    if meta.file_type().is_symlink() {
                        return Err(SandboxedWriteCheckError::Security(McpSecurityError::new(
                            McpSecurityErrorKind::SymlinkEscape,
                            "refusing to write through a symlink",
                        )));
                    }
                    if !meta.is_file() {
                        return Err(SandboxedWriteCheckError::Security(McpSecurityError::new(
                            McpSecurityErrorKind::NullByteOrInvalidPath,
                            "export destination exists and is not a regular file",
                        )));
                    }
                    if !overwrite {
                        return Err(SandboxedWriteCheckError::OverwriteRefused {
                            display_name: policy.relativize_display_path(&resolved),
                        });
                    }
                }
                Err(e) if e.kind() == ErrorKind::NotFound => {}
                Err(e) => {
                    return Err(SandboxedWriteCheckError::Security(McpSecurityError::new(
                        McpSecurityErrorKind::IoError,
                        format!("failed to inspect destination metadata: {e}"),
                    )));
                }
            }
            resolved_targets.push(resolved);
        }
        Ok(resolved_targets)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn resolve_sandboxed_session_load_path(
        &self,
        raw_prefix: &str,
    ) -> Result<PathBuf, McpSecurityError> {
        let base_p = validate_raw_prefix(raw_prefix)?;
        let mut target = base_p.with_extension("nef");
        if !target.is_absolute() {
            target = absolutize_against_cwd(&target)?;
        }
        let target_str = target.to_str().ok_or_else(|| {
            McpSecurityError::new(
                McpSecurityErrorKind::NullByteOrInvalidPath,
                "session path must be valid UTF-8",
            )
        })?;
        let policy = self.security_policy(McpBridgePermission::ReadWrite);
        // Validate containment even if the `.nef` file does not exist yet (prevents out-of-root probing).
        let resolved = policy.resolve_sandboxed_write_path(target_str, "LoadSessionNef")?;
        if let Ok(meta) = std::fs::symlink_metadata(&resolved) {
            if meta.file_type().is_symlink() {
                return Err(McpSecurityError::new(
                    McpSecurityErrorKind::SymlinkEscape,
                    "refusing to load session through a symlink",
                ));
            }
            if !meta.is_file() {
                return Err(McpSecurityError::new(
                    McpSecurityErrorKind::NullByteOrInvalidPath,
                    "session path exists and is not a regular file",
                ));
            }
        }
        Ok(resolved)
    }
}

fn mcp_tool_schemas_json() -> Value {
    json!([
        {
            "name": "get_workbench_state",
            "description": "Inspect live nmr-gui workbench state (active_residue_seq, crosshair_ppm, active_experiment, strip_candidates, selected_pdb_chain_res, undo_depth, redo_depth, qc_status, assigned_residues, excluded_peaks, highlighted_residues_3d, sequence_len, loaded_spectra, peak_count, spin_system_count, assigned_count, high_conf_count, status_msg, ingest_in_progress, picker_mode, picker_snr, workspace_mode, active_page_index, solver_kind, contour_base, view_bounds, ingest_progress_text, peaks_by_experiment).",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }
        },
        {
            "name": "navigate_workbench",
            "description": "Execute Undo-backed UI navigation (`jump_to_residue`, `set_crosshair_ppm`, or `highlight_residues_3d`). Requires ReadWrite permission.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["jump_to_residue", "set_crosshair_ppm", "highlight_residues_3d"]
                    },
                    "seq_id": { "type": "integer" },
                    "h_ppm": { "type": "number" },
                    "n_ppm": { "type": "number" },
                    "c_ppm": { "type": ["number", "null"] },
                    "seq_ids": {
                        "type": "array",
                        "items": { "type": "integer" }
                    }
                },
                "required": ["action"]
            }
        },
        {
            "name": "propose_curation_edit",
            "description": "Apply an Undo-tagged resonance assignment or peak curation action (`assign_spin_system`, `unassign_residue`, `toggle_peak_exclusion`) onto the 256-step McpUndoLog so Cmd/Ctrl+Z reverts in 1 step. Requires ReadWrite permission.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["assign_spin_system", "unassign_residue", "toggle_peak_exclusion"]
                    },
                    "cluster_id": { "type": "integer", "minimum": 0 },
                    "seq_id": { "type": "integer" },
                    "reason": { "type": "string" },
                    "peak_id": { "type": "integer", "minimum": 0 },
                    "excluded": { "type": "boolean" }
                },
                "required": ["action"]
            }
        },
        {
            "name": "undo_last_ai_edit",
            "description": "Undo (or redo when direction='redo') the most recent AI navigation/curation command on the 256-step McpUndoLog. Requires ReadWrite permission.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["undo", "redo"]
                    }
                }
            }
        },
        {
            "name": "capture_viewport_png",
            "description": "Capture the live workbench viewport RGBA framebuffer as a lossless RFC 2083 image/png MCP content block.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "viewport": {
                        "type": "string",
                        "enum": ["active", "contour_2d", "strip_3d", "mol3d"]
                    }
                }
            }
        },
        {
            "name": "run_workbench_workflow",
            "description": "Queue parameterized workbench workflows (`load_study_folder`, `set_sequence`, `pick_peaks`, `run_assignment`, `select_workbook_page`, `set_contour_level`, `set_viewport_bounds`, `export_results`, `set_lock_residue`). Requires ReadWrite permission; filesystem paths are strictly sandboxed to user-approved roots (fail-closed when none are configured), existing files are not overwritten unless `overwrite: true` is passed, and destructive workflows (`load_study_folder`, plus `set_sequence` when replacing existing sequence/assignment state) require enabling destructive actions in GUI MCP settings and passing `confirm: true`.",
            "annotations": {
                "readOnlyHint": false,
                "destructiveHint": true
            },
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "load_study_folder",
                            "set_sequence",
                            "pick_peaks",
                            "run_assignment",
                            "select_workbook_page",
                            "set_contour_level",
                            "set_viewport_bounds",
                            "export_results",
                            "set_lock_residue"
                        ]
                    },
                    "path": { "type": "string" },
                    "run_full_pipeline": { "type": "boolean" },
                    "sequence": { "type": "string" },
                    "sequence_path": { "type": "string" },
                    "snr_threshold": { "type": "number" },
                    "auto_threshold": { "type": "boolean" },
                    "picker_mode": {
                        "type": "string",
                        "enum": ["classical", "neural"]
                    },
                    "solver": {
                        "type": "string",
                        "enum": ["evo-graph", "standard", "bayes-sa"]
                    },
                    "engine": {
                        "type": "string",
                        "enum": ["cpu", "gpu"]
                    },
                    "page_index": { "type": "integer", "minimum": 0 },
                    "page_name": { "type": "string" },
                    "contour_base": { "type": "number" },
                    "multiplier": { "type": "number" },
                    "show_negative": { "type": "boolean" },
                    "h_left": { "type": "number" },
                    "h_right": { "type": "number" },
                    "n_top": { "type": "number" },
                    "n_bottom": { "type": "number" },
                    "prefix": { "type": "string" },
                    "format": {
                        "type": "string",
                        "enum": ["star", "peaks", "nef", "html"]
                    },
                    "overwrite": {
                        "type": "boolean",
                        "description": "Explicitly allow overwriting an existing output file during export_results (default: false)."
                    },
                    "confirm": {
                        "type": "boolean",
                        "description": "Explicit confirmation flag required for destructive workflows (`load_study_folder`, and `set_sequence` when replacing existing sequence/assignment state) alongside enabling destructive actions in GUI MCP settings (default: false)."
                    },
                    "seq_id": { "type": "integer" },
                    "locked": { "type": "boolean" }
                },
                "required": ["action"]
            }
        },
        {
            "name": "dispatch_workbench_action",
            "description": "Dispatch any registered WorkbenchAction (menu, workflow, view, picker, assignment, structure, dynamics, or mode switch) by PascalCase or snake_case name. Requires ReadWrite permission; file-backed actions obey approved-root sandboxing & overwrite refusal, and destructive actions (`ResetWorkspace`, `LoadUbiquitinDemo`, `LoadSessionNef`, `ReSimulateSyntheticNoise`) are denied unless explicitly enabled in MCP settings and confirmed (`confirm: true`); `LoadPathOrSpectrum` is blocked via MCP.",
            "annotations": {
                "readOnlyHint": false,
                "destructiveHint": true
            },
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "WorkbenchAction variant name in PascalCase (e.g. 'SwitchModeAssign') or snake_case (e.g. 'switch_mode_assign'). Call inspect_workbench_data with section='actions' to list all valid names."
                    },
                    "overwrite": {
                        "type": "boolean",
                        "description": "Explicitly allow overwriting existing files for session/STAR/peaks/HTML export actions (default: false)."
                    },
                    "confirm": {
                        "type": "boolean",
                        "description": "Explicit confirmation flag required for destructive actions when enabled in GUI MCP settings (default: false)."
                    }
                },
                "required": ["action"]
            }
        },
        {
            "name": "inspect_workbench_data",
            "description": "Inspect structured workbench tables (`assignments`, `spin_systems`, `peaks`, `unassigned_residues`, `low_confidence`, `qc_report`, `workbook_pages`, `actions`) with optional filtering and pagination. ReadOnly-safe.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "section": {
                        "type": "string",
                        "enum": [
                            "assignments",
                            "spin_systems",
                            "peaks",
                            "unassigned_residues",
                            "low_confidence",
                            "qc_report",
                            "workbook_pages",
                            "actions"
                        ]
                    },
                    "limit": { "type": "integer", "minimum": 1, "maximum": 500 },
                    "offset": { "type": "integer", "minimum": 0 },
                    "experiment": { "type": "string" },
                    "max_confidence": { "type": "number" }
                },
                "required": ["section"]
            }
        }
    ])
}

fn jsonrpc_ok(id: Value, result: Value) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": result,
    })
}

fn jsonrpc_err(id: Value, code: i64, message: impl Into<String>) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": {
            "code": code,
            "message": message.into(),
        }
    })
}

fn mcp_text_result(payload: Value, is_error: bool) -> Value {
    let text = serde_json::to_string(&payload).unwrap_or_else(|_| payload.to_string());
    json!({
        "content": [
            {
                "type": "text",
                "text": text,
            }
        ],
        "isError": is_error,
    })
}

fn parse_navigation_command(args: &Value) -> Result<McpUiCommand, String> {
    let action = args
        .get("action")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim();
    match action {
        "jump_to_residue" => {
            let seq_id = args
                .get("seq_id")
                .and_then(Value::as_i64)
                .and_then(|v| i32::try_from(v).ok())
                .ok_or_else(|| "jump_to_residue requires integer 'seq_id'".to_string())?;
            Ok(McpUiCommand::JumpToResidue { seq_id })
        }
        "set_crosshair_ppm" => {
            let h_ppm = args
                .get("h_ppm")
                .and_then(Value::as_f64)
                .filter(|v| v.is_finite())
                .ok_or_else(|| "set_crosshair_ppm requires finite 'h_ppm'".to_string())?;
            let n_ppm = args
                .get("n_ppm")
                .and_then(Value::as_f64)
                .filter(|v| v.is_finite())
                .ok_or_else(|| "set_crosshair_ppm requires finite 'n_ppm'".to_string())?;
            let c_ppm = match args.get("c_ppm") {
                None | Some(Value::Null) => None,
                Some(v) => Some(
                    v.as_f64()
                        .filter(|c| c.is_finite())
                        .ok_or_else(|| "'c_ppm' must be a finite number or null".to_string())?,
                ),
            };
            Ok(McpUiCommand::SetCrosshairPpm {
                h_ppm,
                n_ppm,
                c_ppm,
            })
        }
        "highlight_residues_3d" => {
            let arr = args
                .get("seq_ids")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    "highlight_residues_3d requires integer array 'seq_ids'".to_string()
                })?;
            let mut seq_ids = Vec::with_capacity(arr.len());
            for item in arr {
                let s = item
                    .as_i64()
                    .and_then(|v| i32::try_from(v).ok())
                    .ok_or_else(|| "'seq_ids' elements must be 32-bit integers".to_string())?;
                seq_ids.push(s);
            }
            Ok(McpUiCommand::HighlightResidues3d { seq_ids })
        }
        other => Err(format!("unknown navigate_workbench action '{other}'")),
    }
}

fn parse_curation_command(args: &Value) -> Result<McpUiCommand, String> {
    let action = args
        .get("action")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim();
    match action {
        "assign_spin_system" => {
            let cluster_id = args
                .get("cluster_id")
                .and_then(Value::as_u64)
                .and_then(|v| usize::try_from(v).ok())
                .ok_or_else(|| {
                    "assign_spin_system requires non-negative 'cluster_id'".to_string()
                })?;
            let seq_id = args
                .get("seq_id")
                .and_then(Value::as_i64)
                .and_then(|v| i32::try_from(v).ok())
                .ok_or_else(|| "assign_spin_system requires integer 'seq_id'".to_string())?;
            let reason = args
                .get("reason")
                .and_then(Value::as_str)
                .unwrap_or("AI MCP curation proposal")
                .trim()
                .to_string();
            Ok(McpUiCommand::AssignSpinSystemToResidue {
                cluster_id,
                seq_id,
                reason,
            })
        }
        "unassign_residue" => {
            let seq_id = args
                .get("seq_id")
                .and_then(Value::as_i64)
                .and_then(|v| i32::try_from(v).ok())
                .ok_or_else(|| "unassign_residue requires integer 'seq_id'".to_string())?;
            let reason = args
                .get("reason")
                .and_then(Value::as_str)
                .unwrap_or("AI MCP unassign proposal")
                .trim()
                .to_string();
            Ok(McpUiCommand::UnassignResidue { seq_id, reason })
        }
        "toggle_peak_exclusion" => {
            let peak_id = args
                .get("peak_id")
                .and_then(Value::as_u64)
                .and_then(|v| usize::try_from(v).ok())
                .ok_or_else(|| {
                    "toggle_peak_exclusion requires non-negative 'peak_id'".to_string()
                })?;
            let excluded = args
                .get("excluded")
                .and_then(Value::as_bool)
                .ok_or_else(|| "toggle_peak_exclusion requires boolean 'excluded'".to_string())?;
            Ok(McpUiCommand::TogglePeakExclusion { peak_id, excluded })
        }
        other => Err(format!("unknown propose_curation_edit action '{other}'")),
    }
}

fn parse_optional_bool(args: &Value, key: &str, default: bool) -> Result<bool, String> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(v) => v
            .as_bool()
            .ok_or_else(|| format!("'{key}' must be a boolean")),
    }
}

fn parse_optional_tri_bool(args: &Value, key: &str) -> Result<Option<bool>, String> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v
            .as_bool()
            .map(Some)
            .ok_or_else(|| format!("'{key}' must be a boolean or null")),
    }
}

fn parse_optional_nonempty_str(args: &Value, key: &str) -> Result<Option<String>, String> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => {
            let s = v
                .as_str()
                .map(str::trim)
                .ok_or_else(|| format!("'{key}' must be a string"))?;
            if s.is_empty() || s.contains('\0') {
                return Err(format!("'{key}' must be a non-empty string"));
            }
            Ok(Some(s.to_string()))
        }
    }
}

fn parse_required_finite_f64(args: &Value, key: &str) -> Result<f64, String> {
    args.get(key)
        .and_then(Value::as_f64)
        .filter(|v| v.is_finite())
        .ok_or_else(|| format!("'{key}' must be a finite number"))
}

fn parse_workflow_command(args: &Value) -> Result<(&'static str, McpUiCommand), String> {
    let action = args
        .get("action")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim();
    match action {
        "load_study_folder" => {
            let path = parse_optional_nonempty_str(args, "path")?
                .ok_or_else(|| "load_study_folder requires non-empty string 'path'".to_string())?;
            let run_full_pipeline = parse_optional_bool(args, "run_full_pipeline", false)?;
            let confirm = parse_optional_bool(args, "confirm", false)?;
            Ok((
                "load_study_folder",
                McpUiCommand::LoadStudyFolder {
                    path,
                    run_full_pipeline,
                    confirm,
                },
            ))
        }
        "set_sequence" => {
            let seq_opt = parse_optional_nonempty_str(args, "sequence")?;
            let path_opt = parse_optional_nonempty_str(args, "sequence_path")?;
            if seq_opt.is_none() && path_opt.is_none() {
                return Err(
                    "set_sequence requires non-empty string 'sequence' or 'sequence_path'"
                        .to_string(),
                );
            }
            let confirm = parse_optional_bool(args, "confirm", false)?;
            Ok((
                "set_sequence",
                McpUiCommand::SetSequence {
                    sequence: seq_opt.unwrap_or_default(),
                    sequence_path: path_opt,
                    confirm,
                },
            ))
        }
        "pick_peaks" => {
            let snr_threshold = match args.get("snr_threshold") {
                None | Some(Value::Null) => None,
                Some(v) => Some(
                    v.as_f64()
                        .filter(|s| s.is_finite() && *s > 0.0)
                        .ok_or_else(|| {
                            "'snr_threshold' must be a finite positive number or null".to_string()
                        })?,
                ),
            };
            let use_auto_threshold = parse_optional_bool(args, "auto_threshold", false)?;
            let picker_mode = match args.get("picker_mode") {
                None | Some(Value::Null) => None,
                Some(v) => {
                    let s = v.as_str().map(str::trim).ok_or_else(|| {
                        "'picker_mode' must be 'classical' or 'neural'".to_string()
                    })?;
                    if !matches!(s, "classical" | "neural") {
                        return Err(format!(
                            "unknown picker_mode '{s}'; expected 'classical' or 'neural'"
                        ));
                    }
                    Some(s.to_string())
                }
            };
            Ok((
                "pick_peaks",
                McpUiCommand::PickAllSpectra {
                    snr_threshold,
                    use_auto_threshold,
                    picker_mode,
                },
            ))
        }
        "run_assignment" => {
            let solver = match args.get("solver") {
                None | Some(Value::Null) => "evo-graph".to_string(),
                Some(v) => {
                    let s = v.as_str().map(str::trim).ok_or_else(|| {
                        "'solver' must be one of 'evo-graph', 'standard', 'bayes-sa'".to_string()
                    })?;
                    if !matches!(s, "evo-graph" | "standard" | "bayes-sa") {
                        return Err(format!(
                            "unknown solver '{s}'; expected one of 'evo-graph', 'standard', 'bayes-sa'"
                        ));
                    }
                    s.to_string()
                }
            };
            let engine = match args.get("engine") {
                None | Some(Value::Null) => None,
                Some(v) => {
                    let s = v
                        .as_str()
                        .map(str::trim)
                        .ok_or_else(|| "'engine' must be 'cpu' or 'gpu'".to_string())?;
                    if !matches!(s, "cpu" | "gpu") {
                        return Err(format!("unknown engine '{s}'; expected 'cpu' or 'gpu'"));
                    }
                    Some(s.to_string())
                }
            };
            Ok((
                "run_assignment",
                McpUiCommand::RunAssignment { solver, engine },
            ))
        }
        "select_workbook_page" => {
            let page_index = match args.get("page_index") {
                None | Some(Value::Null) => None,
                Some(v) => Some(
                    v.as_u64()
                        .and_then(|u| usize::try_from(u).ok())
                        .ok_or_else(|| "'page_index' must be a non-negative integer".to_string())?,
                ),
            };
            let page_name = parse_optional_nonempty_str(args, "page_name")?;
            if page_index.is_none() && page_name.is_none() {
                return Err(
                    "select_workbook_page requires 'page_index' and/or 'page_name'".to_string(),
                );
            }
            Ok((
                "select_workbook_page",
                McpUiCommand::SelectWorkbookPage {
                    page_index,
                    page_name,
                },
            ))
        }
        "set_contour_level" => {
            let contour_base = match args.get("contour_base") {
                None | Some(Value::Null) => None,
                Some(v) => Some(
                    v.as_f64()
                        .filter(|c| c.is_finite() && *c >= 0.0)
                        .ok_or_else(|| {
                            "'contour_base' must be a finite non-negative number or null"
                                .to_string()
                        })?,
                ),
            };
            let multiplier = match args.get("multiplier") {
                None | Some(Value::Null) => None,
                Some(v) => Some(
                    v.as_f64()
                        .filter(|m| m.is_finite() && *m > 0.0)
                        .ok_or_else(|| {
                            "'multiplier' must be a finite positive number or null".to_string()
                        })?,
                ),
            };
            let show_negative = parse_optional_tri_bool(args, "show_negative")?;
            if contour_base.is_none() && multiplier.is_none() && show_negative.is_none() {
                return Err(
                    "set_contour_level requires at least one of 'contour_base', 'multiplier', or 'show_negative'"
                        .to_string(),
                );
            }
            Ok((
                "set_contour_level",
                McpUiCommand::SetContourLevel {
                    contour_base,
                    multiplier,
                    show_negative,
                },
            ))
        }
        "set_viewport_bounds" => {
            let h_left = parse_required_finite_f64(args, "h_left")?;
            let h_right = parse_required_finite_f64(args, "h_right")?;
            let n_top = parse_required_finite_f64(args, "n_top")?;
            let n_bottom = parse_required_finite_f64(args, "n_bottom")?;
            if (h_left - h_right).abs() <= 1e-6 || (n_top - n_bottom).abs() <= 1e-6 {
                return Err(
                    "set_viewport_bounds requires non-degenerate H and N ranges".to_string()
                );
            }
            Ok((
                "set_viewport_bounds",
                McpUiCommand::SetViewportBounds {
                    h_left,
                    h_right,
                    n_top,
                    n_bottom,
                },
            ))
        }
        "export_results" => {
            let prefix = parse_optional_nonempty_str(args, "prefix")?
                .ok_or_else(|| "export_results requires non-empty string 'prefix'".to_string())?;
            let format = args
                .get("format")
                .and_then(Value::as_str)
                .map(str::trim)
                .ok_or_else(|| {
                    "export_results requires 'format' ('star', 'peaks', 'nef', or 'html')"
                        .to_string()
                })?;
            if !matches!(format, "star" | "peaks" | "nef" | "html") {
                return Err(format!(
                    "unknown export format '{format}'; expected 'star', 'peaks', 'nef', or 'html'"
                ));
            }
            let overwrite = parse_optional_bool(args, "overwrite", false)?;
            Ok((
                "export_results",
                McpUiCommand::ExportResults {
                    prefix,
                    format: format.to_string(),
                    overwrite,
                },
            ))
        }
        "set_lock_residue" => {
            let seq_id = args
                .get("seq_id")
                .and_then(Value::as_i64)
                .and_then(|v| i32::try_from(v).ok())
                .ok_or_else(|| "set_lock_residue requires integer 'seq_id'".to_string())?;
            let locked = args
                .get("locked")
                .and_then(Value::as_bool)
                .ok_or_else(|| "set_lock_residue requires boolean 'locked'".to_string())?;
            Ok((
                "set_lock_residue",
                McpUiCommand::SetLockResidue { seq_id, locked },
            ))
        }
        other => Err(format!("unknown run_workbench_workflow action '{other}'")),
    }
}

fn parse_dispatch_action_command(args: &Value) -> Result<(String, McpUiCommand), String> {
    let raw = args
        .get("action")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            "dispatch_workbench_action requires non-empty string 'action'".to_string()
        })?;
    let action =
        parse_workbench_action(raw).ok_or_else(|| format!("unknown WorkbenchAction '{raw}'"))?;
    let overwrite = parse_optional_bool(args, "overwrite", false)?;
    let confirm = parse_optional_bool(args, "confirm", false)?;
    let canonical = workbench_action_canonical_name(action);
    let cmd = if overwrite || confirm {
        McpUiCommand::DispatchGuardedWorkbenchAction {
            action,
            overwrite,
            confirm,
        }
    } else {
        McpUiCommand::DispatchWorkbenchAction { action }
    };
    Ok((canonical, cmd))
}

fn slice_paginated_items(items: Vec<Value>, offset: usize, limit: usize) -> Value {
    let total_matching = items.len();
    let sliced: Vec<Value> = items.into_iter().skip(offset).take(limit).collect();
    json!({
        "total_matching": total_matching,
        "offset": offset,
        "limit": limit,
        "items": sliced,
    })
}

fn execute_inspect_workbench_data(
    inner: &SharedBridgeInner,
    args: &Value,
) -> Result<Value, String> {
    let section = args
        .get("section")
        .and_then(Value::as_str)
        .map(str::trim)
        .ok_or_else(|| "inspect_workbench_data requires string 'section'".to_string())?;
    let limit = match args.get("limit") {
        None | Some(Value::Null) => 100usize,
        Some(v) => v
            .as_i64()
            .map(|n| n.clamp(1, 500) as usize)
            .ok_or_else(|| "'limit' must be an integer".to_string())?,
    };
    let offset = match args.get("offset") {
        None | Some(Value::Null) => 0usize,
        Some(v) => v
            .as_u64()
            .and_then(|u| usize::try_from(u).ok())
            .ok_or_else(|| "'offset' must be a non-negative integer".to_string())?,
    };
    let experiment_filter = match args.get("experiment") {
        None | Some(Value::Null) => None,
        Some(v) => {
            let s = v
                .as_str()
                .map(str::trim)
                .ok_or_else(|| "'experiment' must be a string".to_string())?;
            if s.is_empty() {
                None
            } else {
                Some(s.to_string())
            }
        }
    };
    let max_confidence = match args.get("max_confidence") {
        None | Some(Value::Null) => None,
        Some(v) => Some(
            v.as_f64()
                .filter(|c| c.is_finite())
                .ok_or_else(|| "'max_confidence' must be a finite number or null".to_string())?,
        ),
    };

    let payload = match section {
        "assignments" => {
            let mut body =
                slice_paginated_items(inner.data_tables.assignments.clone(), offset, limit);
            if let Some(m) = body.as_object_mut() {
                m.insert("section".to_string(), json!("assignments"));
            }
            body
        }
        "spin_systems" => {
            let mut body =
                slice_paginated_items(inner.data_tables.spin_systems.clone(), offset, limit);
            if let Some(m) = body.as_object_mut() {
                m.insert("section".to_string(), json!("spin_systems"));
            }
            body
        }
        "peaks" => {
            let filtered: Vec<Value> = match experiment_filter.as_deref() {
                Some(exp_q) => inner
                    .data_tables
                    .peaks
                    .iter()
                    .filter(|row| {
                        row.get("experiment")
                            .or_else(|| row.get("exp"))
                            .and_then(Value::as_str)
                            .is_some_and(|e| e.trim().eq_ignore_ascii_case(exp_q))
                    })
                    .cloned()
                    .collect(),
                None => inner.data_tables.peaks.clone(),
            };
            let mut body = slice_paginated_items(filtered, offset, limit);
            if let Some(m) = body.as_object_mut() {
                m.insert("section".to_string(), json!("peaks"));
                m.insert("experiment".to_string(), json!(experiment_filter));
            }
            body
        }
        "unassigned_residues" => {
            let filtered: Vec<Value> = inner
                .data_tables
                .assignments
                .iter()
                .filter(|row| match row.get("spin_system_idx") {
                    None | Some(Value::Null) => true,
                    Some(v) => v.as_i64().is_none_or(|idx| idx < 0),
                })
                .cloned()
                .collect();
            let mut body = slice_paginated_items(filtered, offset, limit);
            if let Some(m) = body.as_object_mut() {
                m.insert("section".to_string(), json!("unassigned_residues"));
            }
            body
        }
        "low_confidence" => {
            let thresh = max_confidence.unwrap_or(0.8);
            let filtered: Vec<Value> = inner
                .data_tables
                .assignments
                .iter()
                .filter(|row| match row.get("confidence") {
                    None | Some(Value::Null) => true,
                    Some(v) => v.as_f64().is_none_or(|c| !c.is_finite() || c < thresh),
                })
                .cloned()
                .collect();
            let mut body = slice_paginated_items(filtered, offset, limit);
            if let Some(m) = body.as_object_mut() {
                m.insert("section".to_string(), json!("low_confidence"));
                m.insert("max_confidence".to_string(), json!(thresh));
            }
            body
        }
        "qc_report" => {
            json!({
                "section": "qc_report",
                "available": inner.data_tables.qc_report.is_some(),
                "qc_report": inner.data_tables.qc_report,
            })
        }
        "workbook_pages" => {
            let mut body =
                slice_paginated_items(inner.data_tables.workbook_pages.clone(), offset, limit);
            if let Some(m) = body.as_object_mut() {
                m.insert("section".to_string(), json!("workbook_pages"));
            }
            body
        }
        "actions" => {
            let items: Vec<Value> = WorkbenchAction::ALL
                .iter()
                .map(|&act| {
                    let meta = act.metadata();
                    json!({
                        "name": workbench_action_canonical_name(act),
                        "snake_name": workbench_action_snake_name(act),
                        "category": meta.category,
                        "title": meta.title,
                        "description": meta.description,
                        "chord_badge": meta.chord_badge,
                        "destructive": workbench_action_is_destructive(act),
                        "blocked_via_mcp": act == WorkbenchAction::LoadPathOrSpectrum,
                    })
                })
                .collect();
            let workflow_actions: Vec<Value> = ALL_WORKFLOW_ACTIONS
                .iter()
                .map(|&name| {
                    let mode = workflow_action_destructive_mode(name);
                    json!({
                        "name": name,
                        "destructive": mode.is_destructive(),
                        "destructive_mode": mode.as_str(),
                    })
                })
                .collect();
            let mut body = slice_paginated_items(items, offset, limit);
            if let Some(m) = body.as_object_mut() {
                m.insert("section".to_string(), json!("actions"));
                m.insert(
                    "workflow_actions".to_string(),
                    Value::Array(workflow_actions),
                );
            }
            body
        }
        other => {
            return Err(format!("unknown inspect_workbench_data section '{other}'"));
        }
    };
    Ok(payload)
}

fn mcp_security_denied_response(
    id: Value,
    action_name: &str,
    permission: McpBridgePermission,
    error_kind: &str,
    rpc_code: i32,
    reason: String,
) -> Value {
    jsonrpc_ok(
        id,
        mcp_text_result(
            json!({
                "status": "denied",
                "queued": false,
                "action": action_name,
                "permission": permission.as_str(),
                "error_kind": error_kind,
                "rpc_code": rpc_code,
                "reason": reason,
            }),
            true,
        ),
    )
}

fn check_destructive_action_gate(
    inner: &mut SharedBridgeInner,
    id: Value,
    tool_label: &str,
    action_name: &str,
    effective_permission: McpBridgePermission,
    confirm: bool,
) -> Option<Value> {
    if !inner.allow_destructive_actions {
        let reason = format!(
            "destructive action '{action_name}' is denied via MCP (enable 'Allow destructive MCP actions' in GUI MCP Bridge settings and pass confirm: true)"
        );
        inner
            .undo_log
            .record_blocked_security(format!("{tool_label}({action_name}): {reason}"));
        return Some(mcp_security_denied_response(
            id,
            action_name,
            effective_permission,
            "destructive_action_denied",
            nmr_core::mcp::ERR_SECURITY_VIOLATION,
            reason,
        ));
    }
    if !confirm {
        let reason = format!(
            "destructive action '{action_name}' requires explicit confirm: true in the MCP payload"
        );
        inner
            .undo_log
            .record_blocked_security(format!("{tool_label}({action_name}): {reason}"));
        return Some(mcp_security_denied_response(
            id,
            action_name,
            effective_permission,
            "confirmation_required",
            nmr_core::mcp::ERR_SECURITY_VIOLATION,
            reason,
        ));
    }
    None
}

/// Execute a JSON-RPC 2.0 MCP payload (single request, notification, or batch array) against
/// `inner`, returning `None` for one-way notifications (or all-notification batches) and
/// `Some(response_json)` for requests/batches.
fn dispatch_mcp_jsonrpc(inner: &mut SharedBridgeInner, raw_body: &[u8]) -> Option<Value> {
    dispatch_mcp_jsonrpc_with_permission(inner, raw_body, None)
}

fn dispatch_mcp_jsonrpc_with_permission(
    inner: &mut SharedBridgeInner,
    raw_body: &[u8],
    permission_override: Option<McpBridgePermission>,
) -> Option<Value> {
    let effective_permission = match (inner.permission, permission_override) {
        (_, Some(McpBridgePermission::ReadOnly)) | (McpBridgePermission::ReadOnly, _) => {
            McpBridgePermission::ReadOnly
        }
        (McpBridgePermission::ReadWrite, _) => McpBridgePermission::ReadWrite,
    };
    inner.handled_requests = inner.handled_requests.saturating_add(1);
    let parsed: Value = match serde_json::from_slice(raw_body) {
        Ok(v) => v,
        Err(e) => {
            return Some(jsonrpc_err(
                Value::Null,
                -32700,
                format!("Parse error: {e}"),
            ))
        }
    };
    match parsed {
        Value::Object(obj) => dispatch_mcp_jsonrpc_object(inner, &obj, effective_permission),
        Value::Array(items) => {
            if items.is_empty() {
                return Some(jsonrpc_err(
                    Value::Null,
                    -32600,
                    "Invalid Request: JSON-RPC batch array must not be empty",
                ));
            }
            let mut responses = Vec::with_capacity(items.len());
            for item in items {
                match item {
                    Value::Object(obj) => {
                        if let Some(resp) =
                            dispatch_mcp_jsonrpc_object(inner, &obj, effective_permission)
                        {
                            responses.push(resp);
                        }
                    }
                    _ => responses.push(jsonrpc_err(
                        Value::Null,
                        -32600,
                        "Invalid Request: batch item must be a JSON-RPC request object",
                    )),
                }
            }
            if responses.is_empty() {
                None
            } else {
                Some(Value::Array(responses))
            }
        }
        _ => Some(jsonrpc_err(
            Value::Null,
            -32600,
            "Invalid Request: JSON-RPC payload must be an object or non-empty batch array",
        )),
    }
}

fn dispatch_mcp_jsonrpc_object(
    inner: &mut SharedBridgeInner,
    obj: &serde_json::Map<String, Value>,
    effective_permission: McpBridgePermission,
) -> Option<Value> {
    let id_opt = obj.get("id").cloned();
    if let Some(ref id_val) = id_opt {
        if !matches!(id_val, Value::Null | Value::Number(_) | Value::String(_)) {
            return Some(jsonrpc_err(
                Value::Null,
                -32600,
                "Invalid Request: JSON-RPC 'id' must be a string, number, or null",
            ));
        }
    }
    if obj.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
        return Some(jsonrpc_err(
            id_opt.unwrap_or(Value::Null),
            -32600,
            "Invalid Request: expected jsonrpc == '2.0'",
        ));
    }
    let Some(method) = obj.get("method").and_then(Value::as_str) else {
        return Some(jsonrpc_err(
            id_opt.unwrap_or(Value::Null),
            -32600,
            "Invalid Request: missing string 'method'",
        ));
    };

    // JSON-RPC notifications carry no `id`
    if id_opt.is_none() || method.starts_with("notifications/") {
        return None;
    }
    let id = id_opt.unwrap_or(Value::Null);
    let params = obj.get("params").cloned().unwrap_or_else(|| json!({}));

    match method {
        "initialize" => {
            let negotiated_version = params
                .get("protocolVersion")
                .and_then(Value::as_str)
                .filter(|v| nmr_core::mcp::SUPPORTED_MCP_PROTOCOL_VERSIONS.contains(v))
                .unwrap_or(MCP_PROTOCOL_VERSION);
            Some(jsonrpc_ok(
                id,
                json!({
                    "protocolVersion": negotiated_version,
                    "capabilities": {
                        "tools": { "listChanged": false }
                    },
                    "serverInfo": {
                        "name": "ferric-gui-mcp-bridge",
                        "version": nmr_core::VERSION,
                    }
                }),
            ))
        }
        "ping" => Some(jsonrpc_ok(id, json!({}))),
        "tools/list" => Some(jsonrpc_ok(id, json!({ "tools": mcp_tool_schemas_json() }))),
        "tools/call" => {
            let Some(tool_name) = params.get("name").and_then(Value::as_str) else {
                return Some(jsonrpc_err(
                    id,
                    -32602,
                    "Invalid params: tools/call requires string 'name'",
                ));
            };
            let args = params
                .get("arguments")
                .cloned()
                .unwrap_or_else(|| json!({}));
            match tool_name {
                "get_workbench_state" => {
                    let mut snap = inner.snapshot.to_json();
                    if let Some(m) = snap.as_object_mut() {
                        m.insert(
                            "permission".to_string(),
                            Value::String(effective_permission.as_str().to_string()),
                        );
                        m.insert(
                            "allow_destructive_actions".to_string(),
                            Value::Bool(inner.allow_destructive_actions),
                        );
                        #[cfg(not(target_arch = "wasm32"))]
                        {
                            let roots: Vec<Value> = inner
                                .effective_roots()
                                .into_iter()
                                .map(|p| Value::String(p.display().to_string()))
                                .collect();
                            m.insert("approved_roots".to_string(), Value::Array(roots));
                            m.insert(
                                "allow_cwd_fallback".to_string(),
                                Value::Bool(inner.allow_cwd_fallback),
                            );
                        }
                    }
                    Some(jsonrpc_ok(id, mcp_text_result(snap, false)))
                }
                "navigate_workbench" => {
                    let cmd = match parse_navigation_command(&args) {
                        Ok(c) => c,
                        Err(msg) => return Some(jsonrpc_err(id, -32602, msg)),
                    };
                    if effective_permission == McpBridgePermission::ReadOnly {
                        inner.undo_log.record_blocked_readonly(format!(
                            "navigate_workbench: {}",
                            cmd.summary()
                        ));
                        return Some(jsonrpc_ok(
                            id,
                            mcp_text_result(
                                json!({
                                    "status": "denied",
                                    "reason": "MCP bridge is in ReadOnly mode; switch UI panel to ReadWrite to permit navigation mutations.",
                                }),
                                true,
                            ),
                        ));
                    }
                    let summary = cmd.summary();
                    let action_id = inner.undo_log.apply_command(&mut inner.snapshot, cmd);
                    if inner.viewport.viewport_label == "diagnostic_contour_2d" {
                        inner.viewport = ViewportFrameBuffer::diagnostic_card(
                            inner.viewport.width,
                            inner.viewport.height,
                            inner.snapshot.crosshair_ppm,
                        );
                    }
                    inner.request_ui_repaint();
                    Some(jsonrpc_ok(
                        id,
                        mcp_text_result(
                            json!({
                                "status": "applied",
                                "action_id": action_id,
                                "summary": summary,
                                "undo_depth": inner.snapshot.undo_depth,
                                "redo_depth": inner.snapshot.redo_depth,
                            }),
                            false,
                        ),
                    ))
                }
                "propose_curation_edit" => {
                    let cmd = match parse_curation_command(&args) {
                        Ok(c) => c,
                        Err(msg) => return Some(jsonrpc_err(id, -32602, msg)),
                    };
                    if effective_permission == McpBridgePermission::ReadOnly {
                        inner.undo_log.record_blocked_readonly(format!(
                            "propose_curation_edit: {}",
                            cmd.summary()
                        ));
                        return Some(jsonrpc_ok(
                            id,
                            mcp_text_result(
                                json!({
                                    "status": "denied",
                                    "reason": "MCP bridge is in ReadOnly mode; switch UI panel to ReadWrite to permit curation edits.",
                                }),
                                true,
                            ),
                        ));
                    }
                    let summary = cmd.summary();
                    let action_id = inner.undo_log.apply_command(&mut inner.snapshot, cmd);
                    inner.request_ui_repaint();
                    Some(jsonrpc_ok(
                        id,
                        mcp_text_result(
                            json!({
                                "status": "applied",
                                "action_id": action_id,
                                "summary": summary,
                                "undo_depth": inner.snapshot.undo_depth,
                                "redo_depth": inner.snapshot.redo_depth,
                            }),
                            false,
                        ),
                    ))
                }
                "undo_last_ai_edit" => {
                    if effective_permission == McpBridgePermission::ReadOnly {
                        inner
                            .undo_log
                            .record_blocked_readonly("undo_last_ai_edit blocked in ReadOnly mode");
                        return Some(jsonrpc_ok(
                            id,
                            mcp_text_result(
                                json!({
                                    "status": "denied",
                                    "reason": "MCP bridge is in ReadOnly mode; switch UI panel to ReadWrite to undo/redo edits.",
                                }),
                                true,
                            ),
                        ));
                    }
                    let direction = args
                        .get("direction")
                        .and_then(Value::as_str)
                        .unwrap_or("undo");
                    let outcome = match direction {
                        "undo" => inner.undo_log.undo_last(&mut inner.snapshot),
                        "redo" => inner.undo_log.redo_last(&mut inner.snapshot),
                        other => {
                            return Some(jsonrpc_err(
                                id,
                                -32602,
                                format!("Invalid direction '{other}'; expected 'undo' or 'redo'"),
                            ))
                        }
                    };
                    match outcome {
                        Some(reverted_cmd) => {
                            inner.request_ui_repaint();
                            Some(jsonrpc_ok(
                                id,
                                mcp_text_result(
                                    json!({
                                        "status": direction,
                                        "command": reverted_cmd.summary(),
                                        "undo_depth": inner.snapshot.undo_depth,
                                        "redo_depth": inner.snapshot.redo_depth,
                                    }),
                                    false,
                                ),
                            ))
                        }
                        None => Some(jsonrpc_ok(
                            id,
                            mcp_text_result(
                                json!({
                                    "status": "empty",
                                    "reason": format!("No command available for '{direction}'"),
                                    "undo_depth": inner.snapshot.undo_depth,
                                    "redo_depth": inner.snapshot.redo_depth,
                                }),
                                true,
                            ),
                        )),
                    }
                }
                "capture_viewport_png" => {
                    let requested_vp = args
                        .get("viewport")
                        .and_then(Value::as_str)
                        .unwrap_or("active");
                    let img_block = build_mcp_png_image_block(
                        inner.viewport.width,
                        inner.viewport.height,
                        &inner.viewport.rgba,
                    );
                    let meta = json!({
                        "viewport": requested_vp,
                        "source": inner.viewport.viewport_label,
                        "width": inner.viewport.width,
                        "height": inner.viewport.height,
                    });
                    Some(jsonrpc_ok(
                        id,
                        json!({
                            "content": [
                                img_block,
                                {
                                    "type": "text",
                                    "text": meta.to_string(),
                                }
                            ],
                            "isError": false,
                        }),
                    ))
                }
                "run_workbench_workflow" => {
                    let (action_name, cmd) = match parse_workflow_command(&args) {
                        Ok(parsed) => parsed,
                        Err(msg) => return Some(jsonrpc_err(id, -32602, msg)),
                    };
                    if effective_permission == McpBridgePermission::ReadOnly {
                        inner.undo_log.record_blocked_readonly(format!(
                            "run_workbench_workflow: {}",
                            cmd.summary()
                        ));
                        return Some(jsonrpc_ok(
                            id,
                            mcp_text_result(
                                json!({
                                    "status": "denied",
                                    "queued": false,
                                    "action": action_name,
                                    "permission": effective_permission.as_str(),
                                    "reason": "MCP bridge is in ReadOnly mode; switch UI panel to ReadWrite to permit workflow execution.",
                                }),
                                true,
                            ),
                        ));
                    }
                    #[cfg(not(target_arch = "wasm32"))]
                    match &cmd {
                        McpUiCommand::LoadStudyFolder { path, .. } => {
                            if let Err(sec) = inner.resolve_sandboxed_study_path(path) {
                                inner.undo_log.record_blocked_security(format!(
                                    "run_workbench_workflow(load_study_folder): {}",
                                    sec.message
                                ));
                                return Some(mcp_security_denied_response(
                                    id,
                                    action_name,
                                    effective_permission,
                                    sec.kind.as_str(),
                                    sec.kind.rpc_code(),
                                    sec.message,
                                ));
                            }
                        }
                        McpUiCommand::SetSequence {
                            sequence_path: Some(seq_p),
                            ..
                        } => {
                            if let Err(sec) = inner.resolve_sandboxed_read_path(seq_p, false) {
                                inner.undo_log.record_blocked_security(format!(
                                    "run_workbench_workflow(set_sequence): {}",
                                    sec.message
                                ));
                                return Some(mcp_security_denied_response(
                                    id,
                                    action_name,
                                    effective_permission,
                                    sec.kind.as_str(),
                                    sec.kind.rpc_code(),
                                    sec.message,
                                ));
                            }
                        }
                        McpUiCommand::ExportResults {
                            prefix,
                            format,
                            overwrite,
                        } => {
                            let ext = match format.trim().to_ascii_lowercase().as_str() {
                                "peaks" => "tsv",
                                "nef" => "nef",
                                "html" => "html",
                                _ => "str",
                            };
                            if let Err(err) = inner.resolve_sandboxed_export_targets(
                                prefix,
                                &[ext],
                                *overwrite,
                                false,
                                "export_results",
                            ) {
                                let msg = err.message();
                                inner.undo_log.record_blocked_security(format!(
                                    "run_workbench_workflow(export_results): {msg}"
                                ));
                                return Some(mcp_security_denied_response(
                                    id,
                                    action_name,
                                    effective_permission,
                                    err.error_kind(),
                                    err.rpc_code(),
                                    msg,
                                ));
                            }
                        }
                        _ => {}
                    }
                    match &cmd {
                        McpUiCommand::LoadStudyFolder { confirm, .. } => {
                            if let Some(denied) = check_destructive_action_gate(
                                inner,
                                id.clone(),
                                "run_workbench_workflow",
                                action_name,
                                effective_permission,
                                *confirm,
                            ) {
                                return Some(denied);
                            }
                        }
                        McpUiCommand::SetSequence { confirm, .. }
                            if inner.has_existing_sequence_or_assignments() =>
                        {
                            if let Some(denied) = check_destructive_action_gate(
                                inner,
                                id.clone(),
                                "run_workbench_workflow",
                                action_name,
                                effective_permission,
                                *confirm,
                            ) {
                                return Some(denied);
                            }
                        }
                        _ => {}
                    }
                    let summary = cmd.summary();
                    let action_id = inner.undo_log.apply_command(&mut inner.snapshot, cmd);
                    inner.request_ui_repaint();
                    Some(jsonrpc_ok(
                        id,
                        mcp_text_result(
                            json!({
                                "queued": true,
                                "action": action_name,
                                "permission": effective_permission.as_str(),
                                "action_id": action_id,
                                "summary": summary,
                                "undo_depth": inner.snapshot.undo_depth,
                                "redo_depth": inner.snapshot.redo_depth,
                            }),
                            false,
                        ),
                    ))
                }
                "dispatch_workbench_action" => {
                    let (canonical_action, cmd) = match parse_dispatch_action_command(&args) {
                        Ok(parsed) => parsed,
                        Err(msg) => return Some(jsonrpc_err(id, -32602, msg)),
                    };
                    if effective_permission == McpBridgePermission::ReadOnly {
                        inner.undo_log.record_blocked_readonly(format!(
                            "dispatch_workbench_action: {}",
                            cmd.summary()
                        ));
                        return Some(jsonrpc_ok(
                            id,
                            mcp_text_result(
                                json!({
                                    "status": "denied",
                                    "queued": false,
                                    "action": canonical_action,
                                    "permission": effective_permission.as_str(),
                                    "reason": "MCP bridge is in ReadOnly mode; switch UI panel to ReadWrite to dispatch workbench actions.",
                                }),
                                true,
                            ),
                        ));
                    }
                    let (wb_action, overwrite, confirm) = match &cmd {
                        McpUiCommand::DispatchWorkbenchAction { action } => (*action, false, false),
                        McpUiCommand::DispatchGuardedWorkbenchAction {
                            action,
                            overwrite,
                            confirm,
                        } => (*action, *overwrite, *confirm),
                        _ => unreachable!(),
                    };
                    if wb_action == WorkbenchAction::LoadPathOrSpectrum {
                        let reason = "Use run_workbench_workflow(action='load_study_folder', path=..., confirm=true) to load sandboxed datasets via MCP".to_string();
                        inner.undo_log.record_blocked_security(format!(
                            "dispatch_workbench_action: {reason}"
                        ));
                        return Some(mcp_security_denied_response(
                            id,
                            &canonical_action,
                            effective_permission,
                            "sandbox_required",
                            nmr_core::mcp::ERR_SECURITY_VIOLATION,
                            reason,
                        ));
                    }
                    #[cfg(target_arch = "wasm32")]
                    let _ = overwrite;
                    #[cfg(not(target_arch = "wasm32"))]
                    match wb_action {
                        WorkbenchAction::SaveSessionNef | WorkbenchAction::SaveSessionNefAs => {
                            if let Err(err) = inner.resolve_sandboxed_export_targets(
                                &inner.current_export_prefix,
                                &["nef"],
                                overwrite,
                                true,
                                "dispatch_workbench_action",
                            ) {
                                let msg = err.message();
                                inner.undo_log.record_blocked_security(format!(
                                    "dispatch_workbench_action({canonical_action}): {msg}"
                                ));
                                return Some(mcp_security_denied_response(
                                    id,
                                    &canonical_action,
                                    effective_permission,
                                    err.error_kind(),
                                    err.rpc_code(),
                                    msg,
                                ));
                            }
                        }
                        WorkbenchAction::ExportNmrStar | WorkbenchAction::ExportPeaksTsv => {
                            if let Err(err) = inner.resolve_sandboxed_export_targets(
                                &inner.current_export_prefix,
                                &["str", "tsv"],
                                overwrite,
                                true,
                                "dispatch_workbench_action",
                            ) {
                                let msg = err.message();
                                inner.undo_log.record_blocked_security(format!(
                                    "dispatch_workbench_action({canonical_action}): {msg}"
                                ));
                                return Some(mcp_security_denied_response(
                                    id,
                                    &canonical_action,
                                    effective_permission,
                                    err.error_kind(),
                                    err.rpc_code(),
                                    msg,
                                ));
                            }
                        }
                        WorkbenchAction::ExportQualityReportHtml => {
                            if let Err(err) = inner.resolve_sandboxed_export_targets(
                                &inner.current_export_prefix,
                                &["html"],
                                overwrite,
                                true,
                                "dispatch_workbench_action",
                            ) {
                                let msg = err.message();
                                inner.undo_log.record_blocked_security(format!(
                                    "dispatch_workbench_action({canonical_action}): {msg}"
                                ));
                                return Some(mcp_security_denied_response(
                                    id,
                                    &canonical_action,
                                    effective_permission,
                                    err.error_kind(),
                                    err.rpc_code(),
                                    msg,
                                ));
                            }
                        }
                        WorkbenchAction::LoadSessionNef => {
                            if let Err(sec) = inner
                                .resolve_sandboxed_session_load_path(&inner.current_export_prefix)
                            {
                                inner.undo_log.record_blocked_security(format!(
                                    "dispatch_workbench_action(LoadSessionNef): {}",
                                    sec.message
                                ));
                                return Some(mcp_security_denied_response(
                                    id,
                                    &canonical_action,
                                    effective_permission,
                                    sec.kind.as_str(),
                                    sec.kind.rpc_code(),
                                    sec.message,
                                ));
                            }
                        }
                        WorkbenchAction::ProcessAllRawFids
                        | WorkbenchAction::ReconstructNusIstFista => {
                            if let Some(ref ingest_root) = inner.active_ingest_root {
                                if let Err(sec) = inner
                                    .resolve_sandboxed_study_path(&ingest_root.to_string_lossy())
                                {
                                    inner.undo_log.record_blocked_security(format!(
                                        "dispatch_workbench_action({canonical_action}): {}",
                                        sec.message
                                    ));
                                    return Some(mcp_security_denied_response(
                                        id,
                                        &canonical_action,
                                        effective_permission,
                                        sec.kind.as_str(),
                                        sec.kind.rpc_code(),
                                        sec.message,
                                    ));
                                }
                            }
                        }
                        _ => {}
                    }
                    if workbench_action_is_destructive(wb_action) {
                        if let Some(denied) = check_destructive_action_gate(
                            inner,
                            id.clone(),
                            "dispatch_workbench_action",
                            &canonical_action,
                            effective_permission,
                            confirm,
                        ) {
                            return Some(denied);
                        }
                    }
                    let title = wb_action.metadata().title;
                    let summary = cmd.summary();
                    let action_id = inner.undo_log.apply_command(&mut inner.snapshot, cmd);
                    inner.request_ui_repaint();
                    Some(jsonrpc_ok(
                        id,
                        mcp_text_result(
                            json!({
                                "queued": true,
                                "action": canonical_action,
                                "title": title,
                                "permission": effective_permission.as_str(),
                                "action_id": action_id,
                                "summary": summary,
                                "undo_depth": inner.snapshot.undo_depth,
                                "redo_depth": inner.snapshot.redo_depth,
                            }),
                            false,
                        ),
                    ))
                }
                "inspect_workbench_data" => match execute_inspect_workbench_data(inner, &args) {
                    Ok(mut payload) => {
                        if let Some(m) = payload.as_object_mut() {
                            m.insert(
                                "permission".to_string(),
                                Value::String(effective_permission.as_str().to_string()),
                            );
                        }
                        Some(jsonrpc_ok(id, mcp_text_result(payload, false)))
                    }
                    Err(msg) => Some(jsonrpc_err(id, -32602, msg)),
                },
                unknown => Some(jsonrpc_err(
                    id,
                    -32602,
                    format!("Unknown MCP tool '{unknown}'"),
                )),
            }
        }
        unknown => Some(jsonrpc_err(
            id,
            -32601,
            format!("Method not found: '{unknown}'"),
        )),
    }
}

// ============================================================================
// 4. Native Loopback HTTP Server (`#[cfg(not(target_arch = "wasm32"))]`)
// ============================================================================

#[cfg(not(target_arch = "wasm32"))]
struct LoopbackServerHandle {
    bound_addr: SocketAddr,
    shutdown: Arc<AtomicBool>,
    active_workers: Arc<(Mutex<usize>, Condvar)>,
    join_handle: Option<JoinHandle<()>>,
}

#[cfg(not(target_arch = "wasm32"))]
impl LoopbackServerHandle {
    fn stop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        let _ = TcpStream::connect_timeout(&self.bound_addr, Duration::from_millis(100));
        if let Some(handle) = self.join_handle.take() {
            let _ = handle.join();
        }
        let (lock, cvar) = &*self.active_workers;
        if let Ok(guard) = lock.lock() {
            let _ = cvar.wait_timeout_while(guard, Duration::from_secs(2), |cnt| *cnt > 0);
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for LoopbackServerHandle {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Strictly parse and validate that `bind_addr` targets IPv4 loopback `127.0.0.1:<port>`.
#[cfg(not(target_arch = "wasm32"))]
pub fn parse_strict_loopback_bind(bind_addr: &str) -> Result<SocketAddrV4, String> {
    let addr: SocketAddr = bind_addr
        .trim()
        .parse()
        .map_err(|e| format!("invalid socket bind address '{bind_addr}': {e}"))?;
    match addr {
        SocketAddr::V4(v4) if *v4.ip() == Ipv4Addr::LOCALHOST => Ok(v4),
        other => Err(format!(
            "refusing non-127.0.0.1 MCP bridge bind address '{other}'; strictly 127.0.0.1:<port> is permitted"
        )),
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum McpHttpResponseFormat {
    Json,
    EventStream,
}

#[cfg(not(target_arch = "wasm32"))]
fn negotiate_mcp_accept(accept_header: Option<&str>) -> Result<McpHttpResponseFormat, ()> {
    let Some(raw) = accept_header.map(str::trim).filter(|s| !s.is_empty()) else {
        return Ok(McpHttpResponseFormat::Json);
    };

    // Track (specificity, q) per RFC 9110 §12.5.1 so exact ranges override wildcards.
    let mut best_json: Option<(u8, f64)> = None;
    let mut best_sse: Option<(u8, f64)> = None;

    let update_best = |slot: &mut Option<(u8, f64)>, spec: u8, q: f64| match *slot {
        None => *slot = Some((spec, q)),
        Some((prev_spec, prev_q)) => {
            if spec > prev_spec || (spec == prev_spec && q > prev_q) {
                *slot = Some((spec, q));
            }
        }
    };

    for range in raw.split(',') {
        let mut parts = range.split(';');
        let media = parts.next().unwrap_or("").trim().to_ascii_lowercase();
        if media.is_empty() {
            continue;
        }
        let mut q = 1.0_f64;
        let mut valid_q = true;
        for param in parts {
            if let Some((k, v)) = param.trim().split_once('=') {
                if k.trim().eq_ignore_ascii_case("q") {
                    match v.trim().parse::<f64>() {
                        Ok(parsed) if parsed.is_finite() && (0.0..=1.0).contains(&parsed) => {
                            q = parsed;
                        }
                        _ => {
                            valid_q = false;
                        }
                    }
                }
            }
        }
        if !valid_q {
            continue;
        }
        match media.as_str() {
            "application/json" => update_best(&mut best_json, 3, q),
            "application/*" => update_best(&mut best_json, 2, q),
            "text/event-stream" => update_best(&mut best_sse, 3, q),
            "text/*" => update_best(&mut best_sse, 2, q),
            "*/*" => {
                update_best(&mut best_json, 1, q);
                update_best(&mut best_sse, 1, q);
            }
            _ => {}
        }
    }

    let q_json = best_json.map(|(_, q)| q).unwrap_or(0.0);
    let q_sse = best_sse.map(|(_, q)| q).unwrap_or(0.0);
    if q_json <= 0.0 && q_sse <= 0.0 {
        return Err(());
    }
    if q_json >= q_sse {
        Ok(McpHttpResponseFormat::Json)
    } else {
        Ok(McpHttpResponseFormat::EventStream)
    }
}

#[cfg(not(target_arch = "wasm32"))]
enum HttpReadError {
    Timeout,
    Shutdown,
    InvalidUtf8,
    UnexpectedEof,
    Io(String),
}

#[cfg(not(target_arch = "wasm32"))]
struct DeadlineReader<'a> {
    stream: &'a TcpStream,
    deadline: Instant,
    shutdown: &'a AtomicBool,
    buf: [u8; 4096],
    pos: usize,
    len: usize,
}

#[cfg(not(target_arch = "wasm32"))]
impl<'a> DeadlineReader<'a> {
    fn new(stream: &'a TcpStream, deadline: Instant, shutdown: &'a AtomicBool) -> Self {
        Self {
            stream,
            deadline,
            shutdown,
            buf: [0u8; 4096],
            pos: 0,
            len: 0,
        }
    }

    fn fill(&mut self) -> Result<usize, HttpReadError> {
        loop {
            let now = Instant::now();
            if now >= self.deadline {
                return Err(HttpReadError::Timeout);
            }
            if self.shutdown.load(Ordering::SeqCst) {
                return Err(HttpReadError::Shutdown);
            }
            let slice = (self.deadline - now)
                .min(MCP_HTTP_SOCKET_READ_SLICE)
                .max(Duration::from_millis(1));
            let _ = self.stream.set_read_timeout(Some(slice));
            match (&*self.stream).read(&mut self.buf) {
                Ok(n) => {
                    self.pos = 0;
                    self.len = n;
                    return Ok(n);
                }
                Err(e) if e.kind() == ErrorKind::Interrupted => continue,
                Err(e) if matches!(e.kind(), ErrorKind::TimedOut | ErrorKind::WouldBlock) => {
                    if Instant::now() >= self.deadline {
                        return Err(HttpReadError::Timeout);
                    }
                    continue;
                }
                Err(e) => return Err(HttpReadError::Io(e.to_string())),
            }
        }
    }

    fn read_line_bounded(
        &mut self,
        max_bytes: usize,
    ) -> Result<Option<(String, bool)>, HttpReadError> {
        let mut raw = Vec::new();
        let mut exceeded = false;
        loop {
            if Instant::now() >= self.deadline {
                return Err(HttpReadError::Timeout);
            }
            if self.pos >= self.len {
                let n = self.fill()?;
                if n == 0 {
                    if raw.is_empty() {
                        return Ok(None);
                    }
                    break;
                }
            }
            let slice = &self.buf[self.pos..self.len];
            if let Some(nl_rel) = slice.iter().position(|&b| b == b'\n') {
                let take = nl_rel + 1;
                if !exceeded {
                    if raw.len().saturating_add(take) > max_bytes {
                        exceeded = true;
                    } else {
                        raw.extend_from_slice(&slice[..take]);
                    }
                }
                self.pos += take;
                break;
            } else {
                if !exceeded {
                    if raw.len().saturating_add(slice.len()) > max_bytes {
                        exceeded = true;
                    } else {
                        raw.extend_from_slice(slice);
                    }
                }
                self.pos = self.len;
            }
        }
        if exceeded {
            return Ok(Some((String::new(), true)));
        }
        let s = String::from_utf8(raw).map_err(|_| HttpReadError::InvalidUtf8)?;
        Ok(Some((s, false)))
    }

    fn read_exact_bounded(&mut self, exact_len: usize) -> Result<Vec<u8>, HttpReadError> {
        let mut out = vec![0u8; exact_len];
        let mut written = 0usize;
        while written < exact_len {
            if Instant::now() >= self.deadline {
                return Err(HttpReadError::Timeout);
            }
            if self.pos < self.len {
                let avail = (self.len - self.pos).min(exact_len - written);
                out[written..written + avail]
                    .copy_from_slice(&self.buf[self.pos..self.pos + avail]);
                self.pos += avail;
                written += avail;
            } else {
                let n = self.fill()?;
                if n == 0 {
                    return Err(HttpReadError::UnexpectedEof);
                }
            }
        }
        Ok(out)
    }
}

#[cfg(not(target_arch = "wasm32"))]
struct ParsedHttpHeaders {
    method: String,
    path: String,
    host: String,
    duplicate_host: bool,
    origin: Option<String>,
    duplicate_origin: bool,
    authorization: Option<String>,
    accept: Option<String>,
    access_mode_readonly: bool,
    content_length: usize,
    content_length_err: Option<(u16, String)>,
    unsupported_transfer_encoding: bool,
}

#[cfg(not(target_arch = "wasm32"))]
fn read_loopback_http_headers(
    reader: &mut DeadlineReader<'_>,
) -> Result<Option<ParsedHttpHeaders>, (u16, String)> {
    use nmr_core::mcp::{
        parse_bounded_content_length, ContentLengthError, MAX_MCP_HEADER_COUNT,
        MAX_MCP_HEADER_LINE_BYTES,
    };

    let map_read_err = |err: HttpReadError| -> (u16, String) {
        match err {
            HttpReadError::Timeout => (408, "HTTP request read deadline exceeded".to_string()),
            HttpReadError::Shutdown => (503, "MCP bridge shutting down".to_string()),
            HttpReadError::InvalidUtf8 => (400, "invalid UTF-8 in HTTP headers".to_string()),
            HttpReadError::UnexpectedEof => (400, "unexpected EOF in HTTP headers".to_string()),
            HttpReadError::Io(e) => (400, format!("failed reading HTTP headers: {e}")),
        }
    };

    let Some((req_line, req_exceeded)) = reader
        .read_line_bounded(MAX_MCP_HEADER_LINE_BYTES)
        .map_err(map_read_err)?
    else {
        return Ok(None);
    };
    if req_line.trim().is_empty() && !req_exceeded {
        return Ok(None);
    }
    if req_exceeded {
        return Err((413, "HTTP request line exceeds maximum length".to_string()));
    }

    let mut parts = req_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_ascii_uppercase();
    let raw_target = parts.next().unwrap_or("");
    if method.is_empty() || raw_target.is_empty() {
        return Err((400, "malformed HTTP request line".to_string()));
    }
    let path = raw_target.split('?').next().unwrap_or("/").to_string();

    let mut content_length = 0usize;
    let mut seen_content_length = false;
    let mut content_length_err: Option<(u16, String)> = None;
    let mut host = String::new();
    let mut seen_host = false;
    let mut duplicate_host = false;
    let mut origin: Option<String> = None;
    let mut duplicate_origin = false;
    let mut authorization = None;
    let mut accept_parts: Vec<String> = Vec::new();
    let mut access_mode_readonly = false;
    let mut unsupported_transfer_encoding = false;
    let mut header_count = 0usize;

    loop {
        header_count += 1;
        if header_count > MAX_MCP_HEADER_COUNT {
            return Err((400, "too many HTTP header lines".to_string()));
        }
        let Some((line, hdr_exceeded)) = reader
            .read_line_bounded(MAX_MCP_HEADER_LINE_BYTES)
            .map_err(map_read_err)?
        else {
            break;
        };
        if hdr_exceeded {
            return Err((413, "HTTP header line exceeds maximum length".to_string()));
        }
        if line == "\r\n" || line == "\n" || line.is_empty() {
            break;
        }
        if let Some((k, v)) = line.split_once(':') {
            let key = k.trim();
            if key.eq_ignore_ascii_case("content-length") {
                if seen_content_length {
                    content_length_err = Some((400, "duplicate Content-Length header".to_string()));
                } else {
                    seen_content_length = true;
                    match parse_bounded_content_length(v, MAX_MCP_HTTP_BODY_BYTES) {
                        Ok(len) => content_length = len,
                        Err(ContentLengthError::Malformed) => {
                            content_length_err = Some((400, "invalid Content-Length".to_string()));
                        }
                        Err(ContentLengthError::ExceedsLimit { .. }) => {
                            content_length_err =
                                Some((413, "MCP HTTP payload exceeds 4 MiB limit".to_string()));
                        }
                    }
                }
            } else if key.eq_ignore_ascii_case("transfer-encoding") {
                let te = v.trim();
                if !te.is_empty() && !te.eq_ignore_ascii_case("identity") {
                    unsupported_transfer_encoding = true;
                }
            } else if key.eq_ignore_ascii_case("host") {
                if seen_host {
                    duplicate_host = true;
                }
                seen_host = true;
                host = v.trim().to_string();
            } else if key.eq_ignore_ascii_case("origin") {
                if origin.is_some() {
                    duplicate_origin = true;
                }
                let trimmed = v.trim();
                origin = Some(trimmed.to_string());
            } else if key.eq_ignore_ascii_case("authorization") {
                authorization = Some(v.trim().to_string());
            } else if key.eq_ignore_ascii_case("accept") {
                let trimmed = v.trim();
                if !trimmed.is_empty() {
                    accept_parts.push(trimmed.to_string());
                }
            } else if key.eq_ignore_ascii_case("x-nmr-access-mode") {
                let mode = v.trim();
                if mode.eq_ignore_ascii_case("read-only") || mode.eq_ignore_ascii_case("readonly") {
                    access_mode_readonly = true;
                }
            }
        }
    }

    let accept = if accept_parts.is_empty() {
        None
    } else {
        Some(accept_parts.join(", "))
    };

    Ok(Some(ParsedHttpHeaders {
        method,
        path,
        host,
        duplicate_host,
        origin,
        duplicate_origin,
        authorization,
        accept,
        access_mode_readonly,
        content_length,
        content_length_err,
        unsupported_transfer_encoding,
    }))
}

#[cfg(not(target_arch = "wasm32"))]
fn write_http_response_with_type(
    stream: &mut TcpStream,
    status_code: u16,
    reason_phrase: &str,
    content_type: &str,
    extra_headers: &[(&str, &str)],
    body_bytes: &[u8],
) {
    let mut header = format!(
        "HTTP/1.1 {status_code} {reason_phrase}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n",
        body_bytes.len()
    );
    for (k, v) in extra_headers {
        header.push_str(k);
        header.push_str(": ");
        header.push_str(v);
        header.push_str("\r\n");
    }
    header.push_str("\r\n");
    let _ = stream.write_all(header.as_bytes());
    if !body_bytes.is_empty() {
        let _ = stream.write_all(body_bytes);
    }
    let _ = stream.flush();
}

#[cfg(not(target_arch = "wasm32"))]
fn write_http_json_response(
    stream: &mut TcpStream,
    status_code: u16,
    reason_phrase: &str,
    extra_headers: &[(&str, &str)],
    body_bytes: &[u8],
) {
    write_http_response_with_type(
        stream,
        status_code,
        reason_phrase,
        "application/json",
        extra_headers,
        body_bytes,
    );
}

#[cfg(not(target_arch = "wasm32"))]
fn finish_rejected_http_response(
    stream: &mut TcpStream,
    status_code: u16,
    reason_phrase: &str,
    extra_headers: &[(&str, &str)],
    msg: &str,
    drain_already_buffered: bool,
) {
    let payload = json!({ "error": msg }).to_string();
    write_http_json_response(
        stream,
        status_code,
        reason_phrase,
        extra_headers,
        payload.as_bytes(),
    );
    let _ = stream.shutdown(Shutdown::Write);
    if drain_already_buffered && stream.set_nonblocking(true).is_ok() {
        let mut scratch = [0u8; 4096];
        let mut drained = 0usize;
        while drained < 64 * 1024 {
            match (&*stream).read(&mut scratch) {
                Ok(0) | Err(_) => break,
                Ok(n) => drained += n,
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn status_reason_phrase(code: u16) -> &'static str {
    match code {
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        406 => "Not Acceptable",
        408 => "Request Timeout",
        413 => "Payload Too Large",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "Bad Request",
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn handle_loopback_connection(
    mut stream: TcpStream,
    shared: &Arc<Mutex<SharedBridgeInner>>,
    bound_port: u16,
    shutdown: &AtomicBool,
    request_deadline: Duration,
) {
    let _ = stream.set_write_timeout(Some(MCP_HTTP_WRITE_TIMEOUT));
    let deadline = Instant::now() + request_deadline;
    let mut reader = DeadlineReader::new(&stream, deadline, shutdown);

    // Step 0: Read request line and headers ONLY (do NOT read body before Host/Origin/Auth checks!)
    let headers = match read_loopback_http_headers(&mut reader) {
        Ok(Some(h)) => h,
        Ok(None) => return,
        Err((code, msg)) => {
            finish_rejected_http_response(
                &mut stream,
                code,
                status_reason_phrase(code),
                &[],
                &msg,
                code != 408,
            );
            return;
        }
    };

    // Step 1: Enforce strict Host + cross-origin browser Origin guards BEFORE reading body (403)
    if headers.duplicate_origin {
        finish_rejected_http_response(
            &mut stream,
            403,
            "Forbidden",
            &[],
            "duplicate Origin header rejected",
            true,
        );
        return;
    }
    if let Err(reason) = validate_loopback_host_and_origin(
        &headers.host,
        headers.duplicate_host,
        headers.origin.as_deref(),
        bound_port,
    ) {
        finish_rejected_http_response(&mut stream, 403, "Forbidden", &[], reason, true);
        return;
    }

    // Step 2: Enforce constant-time Bearer token authentication BEFORE reading body (401)
    let (expected_token, token_csprng) = match shared.lock() {
        Ok(guard) => (guard.bearer_token.clone(), guard.bearer_token_csprng),
        Err(_) => {
            finish_rejected_http_response(
                &mut stream,
                500,
                "Internal Server Error",
                &[],
                "internal lock poisoned",
                false,
            );
            return;
        }
    };
    if !token_csprng
        || expected_token.len() < 32
        || !verify_bearer_authorization(headers.authorization.as_deref(), &expected_token)
    {
        finish_rejected_http_response(
            &mut stream,
            401,
            "Unauthorized",
            &[("WWW-Authenticate", "Bearer realm=\"nmr-gui-mcp\"")],
            "missing or invalid Bearer token",
            true,
        );
        return;
    }

    // Step 3: Route `/mcp` (and `/`), POST method, and `Accept` header BEFORE reading body
    if !matches!(headers.path.as_str(), "/" | "/mcp") {
        finish_rejected_http_response(
            &mut stream,
            404,
            "Not Found",
            &[],
            "not found; POST JSON-RPC 2.0 to /mcp",
            true,
        );
        return;
    }
    if headers.method != "POST" {
        finish_rejected_http_response(
            &mut stream,
            405,
            "Method Not Allowed",
            &[("Allow", "POST")],
            "method not allowed; expected POST",
            true,
        );
        return;
    }
    let Ok(resp_format) = negotiate_mcp_accept(headers.accept.as_deref()) else {
        finish_rejected_http_response(
            &mut stream,
            406,
            "Not Acceptable",
            &[],
            "Not Acceptable: Accept header must permit application/json or text/event-stream",
            true,
        );
        return;
    };

    // Step 4: Validate bounded Content-Length / Transfer-Encoding BEFORE allocating or reading body
    if headers.unsupported_transfer_encoding {
        finish_rejected_http_response(
            &mut stream,
            400,
            "Bad Request",
            &[],
            "unsupported Transfer-Encoding; send explicit Content-Length",
            true,
        );
        return;
    }
    if let Some((code, ref msg)) = headers.content_length_err {
        finish_rejected_http_response(
            &mut stream,
            code,
            status_reason_phrase(code),
            &[],
            msg,
            true,
        );
        return;
    }

    // Step 5: Read HTTP body strictly within total request deadline
    let body = if headers.content_length > 0 {
        match reader.read_exact_bounded(headers.content_length) {
            Ok(b) => b,
            Err(HttpReadError::Timeout) => {
                finish_rejected_http_response(
                    &mut stream,
                    408,
                    "Request Timeout",
                    &[],
                    "HTTP request body read deadline exceeded",
                    false,
                );
                return;
            }
            Err(HttpReadError::Shutdown) => {
                finish_rejected_http_response(
                    &mut stream,
                    503,
                    "Service Unavailable",
                    &[],
                    "MCP bridge shutting down",
                    false,
                );
                return;
            }
            Err(HttpReadError::UnexpectedEof) => {
                finish_rejected_http_response(
                    &mut stream,
                    400,
                    "Bad Request",
                    &[],
                    "incomplete HTTP body",
                    false,
                );
                return;
            }
            Err(HttpReadError::Io(e)) => {
                finish_rejected_http_response(
                    &mut stream,
                    400,
                    "Bad Request",
                    &[],
                    &format!("failed reading HTTP body: {e}"),
                    false,
                );
                return;
            }
            Err(HttpReadError::InvalidUtf8) => unreachable!(),
        }
    } else {
        Vec::new()
    };

    let perm_override = if headers.access_mode_readonly {
        Some(McpBridgePermission::ReadOnly)
    } else {
        None
    };
    let response_opt = match shared.lock() {
        Ok(mut guard) => dispatch_mcp_jsonrpc_with_permission(&mut guard, &body, perm_override),
        Err(_) => {
            finish_rejected_http_response(
                &mut stream,
                500,
                "Internal Server Error",
                &[],
                "internal lock poisoned",
                false,
            );
            return;
        }
    };

    match response_opt {
        Some(resp_json) => {
            let json_bytes = serde_json::to_vec(&resp_json).unwrap_or_else(|_| b"{}".to_vec());
            match resp_format {
                McpHttpResponseFormat::Json => {
                    write_http_json_response(&mut stream, 200, "OK", &[], &json_bytes);
                }
                McpHttpResponseFormat::EventStream => {
                    let json_str = String::from_utf8_lossy(&json_bytes);
                    let sse_payload = format!("event: message\ndata: {json_str}\n\n");
                    write_http_response_with_type(
                        &mut stream,
                        200,
                        "OK",
                        "text/event-stream",
                        &[],
                        sse_payload.as_bytes(),
                    );
                }
            }
        }
        None => {
            write_http_json_response(&mut stream, 202, "Accepted", &[], b"");
        }
    }
    let _ = stream.shutdown(Shutdown::Write);
}

// ============================================================================
// 5. Cross-Platform Controller & `egui` Desktop MCP Panel (`render_mcp_bridge_panel`)
// ============================================================================

/// Cross-platform (`native` + `wasm32`) controller owning the live [`WorkbenchSnapshot`],
/// `256`-step [`McpUndoLog`], bearer credentials, and optional loopback HTTP worker.
pub struct McpBridgeController {
    shared: Arc<Mutex<SharedBridgeInner>>,
    #[cfg(not(target_arch = "wasm32"))]
    server: Option<LoopbackServerHandle>,
    /// Bind address configured in the UI (`"127.0.0.1:0"` by default).
    pub bind_input: String,
    /// Pending directory path typed into the GUI MCP settings "Add Root" input.
    pub new_root_input: String,
    /// Selected target AI client preset for 1-click JSON config copy.
    pub selected_client: McpClientTarget,
    /// Whether the full bearer token is temporarily revealed in the UI panel.
    pub reveal_token: bool,
    /// Whether the raw client MCP JSON config preview drawer is shown in the UI panel.
    pub show_config_json: bool,
    /// Last informational or error banner message displayed in the panel.
    pub status_banner: String,
}

/// Alias matching the workbench state naming in `NmrWorkbenchApp`.
pub type McpBridgeState = McpBridgeController;

impl Default for McpBridgeController {
    fn default() -> Self {
        Self::new(WorkbenchSnapshot::default())
    }
}

impl McpBridgeController {
    /// Create a new controller seeded with `initial_snapshot` (defaulting to `ReadOnly` mode and
    /// stopped server state until explicitly started).
    pub fn new(initial_snapshot: WorkbenchSnapshot) -> Self {
        Self {
            shared: Arc::new(Mutex::new(SharedBridgeInner::new(initial_snapshot))),
            #[cfg(not(target_arch = "wasm32"))]
            server: None,
            bind_input: DEFAULT_LOOPBACK_BIND.to_string(),
            new_root_input: String::new(),
            selected_client: McpClientTarget::Antigravity,
            reveal_token: false,
            show_config_json: false,
            status_banner: String::new(),
        }
    }

    /// Return `true` when the native loopback HTTP listener is actively bound and serving.
    pub fn is_running(&self) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.server.is_some()
        }
        #[cfg(target_arch = "wasm32")]
        {
            false
        }
    }

    /// Bound `SocketAddr` string (`"127.0.0.1:<port>"`) when running on native desktop.
    pub fn bound_addr_string(&self) -> Option<String> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.server.as_ref().map(|s| s.bound_addr.to_string())
        }
        #[cfg(target_arch = "wasm32")]
        {
            None
        }
    }

    /// Full HTTP MCP endpoint URL (`"http://127.0.0.1:<port>/mcp"` when running, or placeholder).
    pub fn endpoint_url(&self) -> String {
        match self.bound_addr_string() {
            Some(addr) => format!("http://{addr}/mcp"),
            None => "http://127.0.0.1:0/mcp".to_string(),
        }
    }

    /// Start the authenticated loopback HTTP bridge on `bind_addr` (e.g. `"127.0.0.1:0"`).
    /// Rejects non-`127.0.0.1` addresses on native targets, fails closed (`Err`) if OS CSPRNG
    /// entropy is unavailable, and returns a descriptive error on `wasm32`.
    pub fn start_bridge(&mut self, bind_addr: &str) -> Result<String, String> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.start_bridge_with_options(
                bind_addr,
                fill_os_entropy,
                DEFAULT_MCP_HTTP_REQUEST_DEADLINE,
            )
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = bind_addr;
            let msg =
                "Local TCP loopback MCP bridge is desktop-only (unavailable on wasm32)".to_string();
            self.status_banner = msg.clone();
            Err(msg)
        }
    }

    /// Start the authenticated loopback HTTP bridge, trying [`PREFERRED_LOOPBACK_PORT`] (`17420`)
    /// first when [`Self::bind_input`] equals [`DEFAULT_LOOPBACK_BIND`] (`"127.0.0.1:0"`), and
    /// silently falling back to `"127.0.0.1:0"` if port `17420` is already bound.
    ///
    /// Preserves custom user-edited `bind_input` values verbatim without fallback. On `wasm32`,
    /// delegates cleanly to [`Self::start_bridge`].
    pub fn start_bridge_preferred(&mut self) -> Result<String, String> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.start_bridge_with_preferred_port(PREFERRED_LOOPBACK_PORT)
        }
        #[cfg(target_arch = "wasm32")]
        {
            let target = self.bind_input.clone();
            self.start_bridge(&target)
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn start_bridge_with_preferred_port(&mut self, preferred_port: u16) -> Result<String, String> {
        let target = self.bind_input.trim().to_string();
        if target == DEFAULT_LOOPBACK_BIND {
            let preferred = format!("127.0.0.1:{preferred_port}");
            match self.start_bridge(&preferred) {
                Ok(addr) => Ok(addr),
                Err(preferred_err) => {
                    if preferred_err.contains("OS CSPRNG") {
                        Err(preferred_err)
                    } else {
                        self.start_bridge(DEFAULT_LOOPBACK_BIND)
                    }
                }
            }
        } else {
            self.start_bridge(&target)
        }
    }

    #[cfg(all(test, not(target_arch = "wasm32")))]
    fn start_bridge_with_entropy(
        &mut self,
        bind_addr: &str,
        fill_fn: fn(&mut [u8; 32]) -> bool,
    ) -> Result<String, String> {
        self.start_bridge_with_options(bind_addr, fill_fn, DEFAULT_MCP_HTTP_REQUEST_DEADLINE)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn start_bridge_with_options(
        &mut self,
        bind_addr: &str,
        fill_fn: fn(&mut [u8; 32]) -> bool,
        request_deadline: Duration,
    ) -> Result<String, String> {
        self.stop_bridge();
        let needs_fresh_csprng = self
            .shared
            .lock()
            .map(|g| !g.bearer_token_csprng || g.bearer_token.len() != 72)
            .map_err(|_| "MCP bridge state lock poisoned".to_string())?;
        if needs_fresh_csprng {
            let fresh = match try_generate_bearer_token_with(fill_fn) {
                Ok(t) => t,
                Err(err) => {
                    self.status_banner = err.clone();
                    return Err(err);
                }
            };
            let mut g = self
                .shared
                .lock()
                .map_err(|_| "MCP bridge state lock poisoned".to_string())?;
            g.bearer_token = fresh;
            g.bearer_token_csprng = true;
        }
        let v4 = parse_strict_loopback_bind(bind_addr)?;
        let listener = TcpListener::bind(SocketAddr::V4(v4))
            .map_err(|e| format!("failed to bind loopback MCP listener on {v4}: {e}"))?;
        let bound_addr = listener
            .local_addr()
            .map_err(|e| format!("failed to query bound socket address: {e}"))?;
        let bound_port = bound_addr.port();
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_bg = Arc::clone(&shutdown);
        let active_workers = Arc::new((Mutex::new(0usize), Condvar::new()));
        let active_bg = Arc::clone(&active_workers);
        let shared_bg = Arc::clone(&self.shared);

        struct WorkerSlotGuard(Arc<(Mutex<usize>, Condvar)>);
        impl Drop for WorkerSlotGuard {
            fn drop(&mut self) {
                let (lock, cvar) = &*self.0;
                if let Ok(mut cnt) = lock.lock() {
                    *cnt = cnt.saturating_sub(1);
                    cvar.notify_all();
                }
            }
        }

        let join_handle = thread::Builder::new()
            .name("nmr-gui-mcp-loopback".to_string())
            .spawn(move || {
                for incoming in listener.incoming() {
                    if shutdown_bg.load(Ordering::SeqCst) {
                        break;
                    }
                    let Ok(mut stream) = incoming else {
                        continue;
                    };
                    if shutdown_bg.load(Ordering::SeqCst) {
                        break;
                    }
                    let acquired = {
                        let (lock, _) = &*active_bg;
                        if let Ok(mut cnt) = lock.lock() {
                            if *cnt < MAX_MCP_CONCURRENT_CONNECTIONS {
                                *cnt += 1;
                                true
                            } else {
                                false
                            }
                        } else {
                            false
                        }
                    };
                    if !acquired {
                        finish_rejected_http_response(
                            &mut stream,
                            503,
                            "Service Unavailable",
                            &[],
                            "too many concurrent MCP connections",
                            false,
                        );
                        continue;
                    }
                    let worker_active = Arc::clone(&active_bg);
                    let worker_shared = Arc::clone(&shared_bg);
                    let worker_shutdown = Arc::clone(&shutdown_bg);
                    let spawn_res = thread::Builder::new()
                        .name("nmr-gui-mcp-worker".to_string())
                        .spawn(move || {
                            let _slot_guard = WorkerSlotGuard(worker_active);
                            handle_loopback_connection(
                                stream,
                                &worker_shared,
                                bound_port,
                                &worker_shutdown,
                                request_deadline,
                            );
                        });
                    if spawn_res.is_err() {
                        let (lock, cvar) = &*active_bg;
                        if let Ok(mut cnt) = lock.lock() {
                            *cnt = cnt.saturating_sub(1);
                            cvar.notify_all();
                        }
                    }
                }
            })
            .map_err(|e| format!("failed to spawn MCP loopback accept thread: {e}"))?;

        let addr_str = bound_addr.to_string();
        self.server = Some(LoopbackServerHandle {
            bound_addr,
            shutdown,
            active_workers,
            join_handle: Some(join_handle),
        });
        self.status_banner = format!("Listening on http://{addr_str}/mcp");
        Ok(addr_str)
    }

    /// Stop the running loopback HTTP listener cleanly, if active.
    pub fn stop_bridge(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(mut srv) = self.server.take() {
            srv.stop();
            self.status_banner = "MCP loopback bridge stopped".to_string();
        }
    }

    /// Current permission mode (`ReadOnly` or `ReadWrite`).
    pub fn permission(&self) -> McpBridgePermission {
        self.shared
            .lock()
            .map(|g| g.permission)
            .unwrap_or(McpBridgePermission::ReadOnly)
    }

    /// Update permission mode (`ReadOnly` vs `ReadWrite`), taking effect immediately for live calls.
    pub fn set_permission(&self, perm: McpBridgePermission) {
        if let Ok(mut g) = self.shared.lock() {
            g.permission = perm;
        }
    }

    /// Whether destructive MCP actions (`ResetWorkspace`, `LoadUbiquitinDemo`, `LoadSessionNef`,
    /// `ReSimulateSyntheticNoise`, `load_study_folder`, and `set_sequence` when replacing existing
    /// sequence/assignment state) are enabled (false by default).
    pub fn destructive_actions_allowed(&self) -> bool {
        self.shared
            .lock()
            .map(|g| g.allow_destructive_actions)
            .unwrap_or(false)
    }

    /// Enable or disable destructive MCP actions (still requires explicit `confirm: true`).
    pub fn set_destructive_actions_allowed(&self, allowed: bool) {
        if let Ok(mut g) = self.shared.lock() {
            g.allow_destructive_actions = allowed;
        }
    }

    /// Whether explicit current-working-directory fallback is enabled when no roots or active
    /// study folder are configured (false by default; never permits `$HOME` or its ancestors).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn cwd_fallback_allowed(&self) -> bool {
        self.shared
            .lock()
            .map(|g| g.allow_cwd_fallback)
            .unwrap_or(false)
    }

    /// Explicitly opt in or out of using a non-`$HOME` current working directory as a fallback
    /// sandbox root when no roots are customized and no study is open (false by default).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn set_cwd_fallback_allowed(&self, allowed: bool) {
        if let Ok(mut g) = self.shared.lock() {
            g.allow_cwd_fallback = allowed;
        }
    }

    /// Current effective canonicalized sandbox root directories (native desktop only).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn approved_roots(&self) -> Vec<PathBuf> {
        self.shared
            .lock()
            .map(|g| g.effective_roots())
            .unwrap_or_default()
    }

    /// Return `true` when the user has explicitly customized approved MCP roots in GUI settings.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn roots_customized(&self) -> bool {
        self.shared
            .lock()
            .map(|g| g.roots_customized)
            .unwrap_or(false)
    }

    /// Replace approved sandbox root directories with canonicalized existing non-root directories.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn set_approved_roots<I, P>(&self, roots: I) -> Result<(), String>
    where
        I: IntoIterator<Item = P>,
        P: AsRef<Path>,
    {
        let mut canon_roots = Vec::new();
        for r in roots {
            let canon = canonicalize_approved_root(r.as_ref())?;
            if !canon_roots.contains(&canon) {
                canon_roots.push(canon);
            }
        }
        let mut g = self
            .shared
            .lock()
            .map_err(|_| "MCP bridge state lock poisoned".to_string())?;
        g.approved_roots = canon_roots;
        g.roots_customized = true;
        Ok(())
    }

    /// Canonicalize and append a user-approved sandbox root directory.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn add_approved_root(&self, raw_dir: &str) -> Result<PathBuf, String> {
        let trimmed = raw_dir.trim();
        if trimmed.is_empty() {
            return Err("root directory path must not be empty".to_string());
        }
        let canon = canonicalize_approved_root(Path::new(trimmed))?;
        let mut g = self
            .shared
            .lock()
            .map_err(|_| "MCP bridge state lock poisoned".to_string())?;
        if !g.roots_customized {
            g.approved_roots = g.effective_roots();
            g.roots_customized = true;
        }
        if !g.approved_roots.contains(&canon) {
            g.approved_roots.push(canon.clone());
        }
        Ok(canon)
    }

    /// Remove an approved sandbox root directory by index (customizing the root list).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn remove_approved_root(&self, idx: usize) -> bool {
        let Ok(mut g) = self.shared.lock() else {
            return false;
        };
        if !g.roots_customized {
            g.approved_roots = g.effective_roots();
            g.roots_customized = true;
        }
        if idx < g.approved_roots.len() {
            g.approved_roots.remove(idx);
            true
        } else {
            false
        }
    }

    /// Restore approved roots to the automatic fail-closed default (active non-`$HOME` study only;
    /// empty when no study is active, and disables CWD fallback).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn reset_approved_roots_to_default(&self) {
        if let Ok(mut g) = self.shared.lock() {
            g.roots_customized = false;
            g.allow_cwd_fallback = false;
            g.approved_roots.clear();
        }
    }

    /// Synchronize the active user-opened study directory (ignoring paths loaded via MCP itself,
    /// and refusing `$HOME` or any ancestor of `$HOME` as an automatic sandbox root).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn sync_user_study_root(&self, root_path_opt: Option<&Path>) {
        let Ok(mut g) = self.shared.lock() else {
            return;
        };
        g.active_ingest_root = root_path_opt.map(Path::to_path_buf);
        let Some(raw) = root_path_opt else {
            g.active_study_root = None;
            g.last_mcp_loaded_path = None;
            return;
        };
        let dir_candidate = if raw.is_dir() {
            raw.to_path_buf()
        } else if let Some(parent) = raw.parent() {
            parent.to_path_buf()
        } else {
            g.active_study_root = None;
            return;
        };
        if let Some(ref mcp_loaded) = g.last_mcp_loaded_path {
            let same = std::fs::canonicalize(raw)
                .ok()
                .is_some_and(|c| &c == mcp_loaded)
                || raw == mcp_loaded.as_path();
            if same {
                return;
            }
        }
        g.last_mcp_loaded_path = None;
        match canonicalize_automatic_root(&dir_candidate) {
            Ok(canon) => g.active_study_root = Some(canon),
            Err(_) => g.active_study_root = None,
        }
    }

    /// Record the canonical path loaded by an MCP `load_study_folder` call so GUI sync does not
    /// reassign `active_study_root`.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn set_last_mcp_loaded_path(&self, path_opt: Option<&Path>) {
        if let Ok(mut g) = self.shared.lock() {
            g.last_mcp_loaded_path = path_opt.map(Path::to_path_buf);
        }
    }

    /// Synchronize the workbench `export_prefix` used by `dispatch_workbench_action` export/save/load checks.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn set_current_export_prefix(&self, prefix: &str) {
        if let Ok(mut g) = self.shared.lock() {
            g.current_export_prefix = prefix.trim().to_string();
        }
    }

    /// Validate and canonicalize an existing sandboxed file or directory inside approved roots.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn resolve_sandboxed_read_path(
        &self,
        raw_path: &str,
        allow_dir: bool,
    ) -> Result<PathBuf, String> {
        let g = self
            .shared
            .lock()
            .map_err(|_| "MCP bridge state lock poisoned".to_string())?;
        g.resolve_sandboxed_read_path(raw_path, allow_dir)
            .map_err(|e| e.message)
    }

    /// Validate and canonicalize a study folder (or single spectrum file) inside approved roots,
    /// recursively ensuring no nested symlink inside a study directory escapes the study root.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn resolve_sandboxed_study_path(&self, raw_path: &str) -> Result<PathBuf, String> {
        let g = self
            .shared
            .lock()
            .map_err(|_| "MCP bridge state lock poisoned".to_string())?;
        g.resolve_sandboxed_study_path(raw_path)
            .map_err(|e| e.message)
    }

    /// Validate, canonicalize, and check non-overwrite on export destination paths inside approved roots.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn resolve_sandboxed_export_targets(
        &self,
        raw_prefix: &str,
        exts: &[&str],
        overwrite: bool,
        resolve_relative_against_cwd: bool,
    ) -> Result<Vec<PathBuf>, String> {
        let g = self
            .shared
            .lock()
            .map_err(|_| "MCP bridge state lock poisoned".to_string())?;
        g.resolve_sandboxed_export_targets(
            raw_prefix,
            exts,
            overwrite,
            resolve_relative_against_cwd,
            "export_results",
        )
        .map_err(|e| e.message())
    }

    /// Validate containment of a `.nef` session path before `LoadSessionNef`.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn resolve_sandboxed_session_load_path(&self, raw_prefix: &str) -> Result<PathBuf, String> {
        let g = self
            .shared
            .lock()
            .map_err(|_| "MCP bridge state lock poisoned".to_string())?;
        g.resolve_sandboxed_session_load_path(raw_prefix)
            .map_err(|e| e.message)
    }

    /// Current secret bearer token required on `Authorization: Bearer <token>`.
    pub fn bearer_token(&self) -> String {
        self.shared
            .lock()
            .map(|g| g.bearer_token.clone())
            .unwrap_or_default()
    }

    /// Rotate the secret bearer token in-place using OS CSPRNG entropy (`#[cfg(any(unix, windows))]`)
    /// and return the new token, failing closed (`Err`) if OS CSPRNG fails.
    pub fn rotate_bearer_token(&mut self) -> Result<String, String> {
        #[cfg(any(unix, windows))]
        {
            match generate_bearer_token() {
                Ok(fresh) => {
                    if let Ok(mut g) = self.shared.lock() {
                        g.bearer_token = fresh.clone();
                        g.bearer_token_csprng = true;
                    }
                    Ok(fresh)
                }
                Err(err) => {
                    self.stop_bridge();
                    if let Ok(mut g) = self.shared.lock() {
                        g.bearer_token.clear();
                        g.bearer_token_csprng = false;
                    }
                    Err(err)
                }
            }
        }
        #[cfg(not(any(unix, windows)))]
        {
            let fresh = wasm_preview_display_token();
            if let Ok(mut g) = self.shared.lock() {
                g.bearer_token = fresh.clone();
                g.bearer_token_csprng = false;
            }
            Ok(fresh)
        }
    }

    /// Clone the current [`WorkbenchSnapshot`].
    pub fn snapshot(&self) -> WorkbenchSnapshot {
        self.shared
            .lock()
            .map(|g| g.snapshot.clone())
            .unwrap_or_default()
    }

    /// Mutate the host [`WorkbenchSnapshot`] from the GUI loop while preserving Undo/Redo depths.
    pub fn update_snapshot<F: FnOnce(&mut WorkbenchSnapshot)>(&self, f: F) {
        if let Ok(mut g) = self.shared.lock() {
            f(&mut g.snapshot);
            g.snapshot.undo_depth = g.undo_log.undo_depth();
            g.snapshot.redo_depth = g.undo_log.redo_depth();
        }
    }

    /// Store the live `egui::Context` so background MCP commands trigger an immediate UI repaint.
    pub fn set_repaint_context(&self, ctx: egui::Context) {
        if let Ok(mut g) = self.shared.lock() {
            g.repaint_ctx = Some(ctx);
        }
    }

    /// Update structured tabular inspection data exposed read-only via `inspect_workbench_data`.
    pub fn update_data_tables(&self, f: impl FnOnce(&mut WorkbenchDataTables)) {
        if let Ok(mut g) = self.shared.lock() {
            f(&mut g.data_tables);
        }
    }

    /// Upload a rendered 8-bit RGBA viewport framebuffer for `capture_viewport_png`.
    pub fn update_viewport_rgba(
        &self,
        width: u32,
        height: u32,
        rgba: Vec<u8>,
        viewport_label: impl Into<String>,
    ) {
        if let Ok(mut g) = self.shared.lock() {
            g.viewport = ViewportFrameBuffer {
                width: width.max(1),
                height: height.max(1),
                rgba,
                viewport_label: viewport_label.into(),
            };
        }
    }

    /// Apply an [`McpUiCommand`] directly onto the shared Undo-backed state.
    pub fn apply_ui_command(&self, command: McpUiCommand) -> Option<u64> {
        let mut g = self.shared.lock().ok()?;
        let mut snap = g.snapshot.clone();
        let id = g.undo_log.apply_command(&mut snap, command);
        g.snapshot = snap;
        Some(id)
    }

    /// Undo the most recent AI command (`Cmd/Ctrl+Z`).
    pub fn undo_last(&self) -> Option<McpUiCommand> {
        let mut g = self.shared.lock().ok()?;
        let mut snap = g.snapshot.clone();
        let out = g.undo_log.undo_last(&mut snap);
        g.snapshot = snap;
        out
    }

    /// Redo the most recently undone AI command (`Cmd/Ctrl+Shift+Z`).
    pub fn redo_last(&self) -> Option<McpUiCommand> {
        let mut g = self.shared.lock().ok()?;
        let mut snap = g.snapshot.clone();
        let out = g.undo_log.redo_last(&mut snap);
        g.snapshot = snap;
        out
    }

    /// Drain pending [`McpUiCommand`] items queued since the previous GUI frame.
    pub fn drain_pending_ui_commands(&self) -> Vec<McpUiCommand> {
        self.shared
            .lock()
            .map(|mut g| g.undo_log.drain_pending_ui_commands())
            .unwrap_or_default()
    }

    /// Build ready-to-paste client MCP configuration JSON for the currently selected client.
    pub fn build_client_config_json(&self) -> String {
        build_mcp_client_config_json_with_permission(
            self.selected_client,
            &self.endpoint_url(),
            &self.bearer_token(),
            self.permission(),
        )
    }

    /// Build on-screen client MCP configuration JSON preview, keeping the bearer token masked
    /// whenever [`Self::reveal_token`] is `false`.
    pub fn build_client_config_json_preview(&self) -> String {
        let raw_tok = self.bearer_token();
        let preview_tok = if self.reveal_token {
            raw_tok
        } else {
            mask_bearer_token(&raw_tok)
        };
        build_mcp_client_config_json_with_permission(
            self.selected_client,
            &self.endpoint_url(),
            &preview_tok,
            self.permission(),
        )
    }

    /// Dispatch a raw JSON-RPC 2.0 MCP request byte payload in-process (available on both native
    /// desktop and `wasm32` targets). Returns `None` for one-way JSON-RPC notifications.
    pub fn handle_jsonrpc_bytes(&self, raw_body: &[u8]) -> Option<Value> {
        let mut guard = self.shared.lock().ok()?;
        dispatch_mcp_jsonrpc(&mut guard, raw_body)
    }
}

/// Summary of user interactions emitted during a frame of [`render_mcp_bridge_panel`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct McpPanelResponse {
    /// Set when the user clicked *"Undo AI Action"* in the panel.
    pub undone_command: Option<McpUiCommand>,
    /// Set when the user clicked *"Redo"* in the panel.
    pub redone_command: Option<McpUiCommand>,
    /// `true` when *"Copy MCP Config"* copied JSON to the system clipboard on this frame.
    pub copied_config: bool,
}

/// Render the interactive desktop `MCP` Bridge Control Panel inside any `egui::Ui` container.
pub fn render_mcp_bridge_panel(
    ui: &mut egui::Ui,
    controller: &mut McpBridgeController,
) -> McpPanelResponse {
    let mut response = McpPanelResponse::default();

    ui.heading("MCP Live Workbench Bridge");
    ui.add_space(4.0);

    // 1. Bridge Lifecycle Row (native Start/Stop vs wasm32 informational desktop-only badge)
    #[cfg(target_arch = "wasm32")]
    {
        ui.horizontal_wrapped(|ui| {
            ui.colored_label(
                egui::Color32::from_rgb(120, 195, 255),
                "Desktop-Only Feature: Local 127.0.0.1 TCP loopback MCP bridge requires the native desktop app (disabled in browser wasm32).",
            );
        });
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let running = controller.is_running();
        ui.horizontal_wrapped(|ui| {
            if running {
                if ui.button("Stop Bridge").clicked() {
                    controller.stop_bridge();
                }
                if let Some(addr) = controller.bound_addr_string() {
                    ui.colored_label(
                        egui::Color32::from_rgb(90, 220, 140),
                        format!("Active: http://{addr}/mcp"),
                    );
                }
            } else {
                ui.label("Bind:");
                ui.add(egui::TextEdit::singleline(&mut controller.bind_input).desired_width(130.0));
                if ui.button("Start Local Bridge").clicked() {
                    if let Err(err) = controller.start_bridge_preferred() {
                        controller.status_banner = format!("Start failed: {err}");
                    }
                }
                ui.colored_label(egui::Color32::GRAY, "Stopped");
            }
        });
    }

    ui.separator();

    // Snapshot shared state once for consistent panel rendering
    let (mut perm, mut allow_destructive, token, undo_depth, redo_depth, handled_reqs, audit_rows) =
        match controller.shared.lock() {
            Ok(g) => (
                g.permission,
                g.allow_destructive_actions,
                g.bearer_token.clone(),
                g.undo_log.undo_depth(),
                g.undo_log.redo_depth(),
                g.handled_requests,
                g.undo_log
                    .audit_records()
                    .iter()
                    .cloned()
                    .collect::<Vec<_>>(),
            ),
            Err(_) => (
                McpBridgePermission::ReadOnly,
                false,
                String::new(),
                0,
                0,
                0,
                Vec::new(),
            ),
        };

    // 2. Permission Tier Switch (`ReadOnly` vs `ReadWrite`) & Destructive Guard
    ui.horizontal_wrapped(|ui| {
        ui.label("Permission:");
        let prev_perm = perm;
        ui.selectable_value(
            &mut perm,
            McpBridgePermission::ReadOnly,
            "ReadOnly (Inspect & PNG Only)",
        );
        ui.selectable_value(
            &mut perm,
            McpBridgePermission::ReadWrite,
            "ReadWrite (Undo-Backed Curation)",
        );
        if perm != prev_perm {
            controller.set_permission(perm);
        }
        ui.label(format!("| Requests: {handled_reqs}"));
    });
    let prev_destructive = allow_destructive;
    ui.checkbox(
        &mut allow_destructive,
        "Allow destructive MCP actions (ResetWorkspace, LoadUbiquitinDemo, LoadSessionNef, load_study_folder, replacing set_sequence; still requires confirm: true)",
    );
    if allow_destructive != prev_destructive {
        controller.set_destructive_actions_allowed(allow_destructive);
    }

    // 2b. User-Approved Filesystem Sandbox Roots (native desktop)
    #[cfg(not(target_arch = "wasm32"))]
    {
        let roots = controller.approved_roots();
        let customized = controller.roots_customized();
        let mut allow_cwd_fb = controller.cwd_fallback_allowed();
        ui.horizontal_wrapped(|ui| {
            ui.strong(if customized {
                "Approved Filesystem Roots (Custom):"
            } else {
                "Approved Filesystem Roots (Active Study Only — Fail-Closed by Default):"
            });
            if (customized || allow_cwd_fb) && ui.small_button("Reset Default").clicked() {
                controller.reset_approved_roots_to_default();
                controller.status_banner =
                    "Reset MCP sandbox roots to fail-closed active-study default".to_string();
                allow_cwd_fb = false;
            }
        });
        if !customized {
            let prev_cwd_fb = allow_cwd_fb;
            ui.checkbox(
                &mut allow_cwd_fb,
                "Allow current working directory fallback when no study is open (excludes $HOME and ancestors)",
            );
            if allow_cwd_fb != prev_cwd_fb {
                controller.set_cwd_fallback_allowed(allow_cwd_fb);
            }
        }
        if roots.is_empty() {
            ui.colored_label(
                egui::Color32::from_rgb(245, 180, 90),
                "No approved roots configured — file read/write tools are blocked (open a study folder or add a root below).",
            );
        } else {
            let mut remove_idx = None;
            for (idx, root) in roots.iter().enumerate() {
                ui.horizontal(|ui| {
                    ui.monospace(root.display().to_string());
                    if ui.small_button("Remove").clicked() {
                        remove_idx = Some(idx);
                    }
                });
            }
            if let Some(idx) = remove_idx {
                if controller.remove_approved_root(idx) {
                    controller.status_banner = "Removed MCP sandbox root".to_string();
                }
            }
        }
        ui.horizontal_wrapped(|ui| {
            ui.label("Add Root:");
            ui.add(
                egui::TextEdit::singleline(&mut controller.new_root_input)
                    .hint_text("/path/to/approved_study_dir")
                    .desired_width(240.0),
            );
            if ui.small_button("Add Root").clicked() {
                let candidate = controller.new_root_input.clone();
                match controller.add_approved_root(&candidate) {
                    Ok(canon) => {
                        controller.new_root_input.clear();
                        controller.status_banner =
                            format!("Added approved MCP root: {}", canon.display());
                    }
                    Err(err) => {
                        controller.status_banner = format!("Invalid MCP root: {err}");
                    }
                }
            }
        });
    }

    // 3. Masked Bearer Token & 1-Click Client Config Copy
    ui.horizontal_wrapped(|ui| {
        ui.label("Bearer Token:");
        let shown = if controller.reveal_token {
            token.clone()
        } else {
            mask_bearer_token(&token)
        };
        ui.monospace(shown);
        let toggle_lbl = if controller.reveal_token {
            "Hide"
        } else {
            "Show"
        };
        if ui.small_button(toggle_lbl).clicked() {
            controller.reveal_token = !controller.reveal_token;
        }
        if ui.small_button("Rotate").clicked() {
            match controller.rotate_bearer_token() {
                Ok(_) => {
                    controller.status_banner = "Rotated bearer token".to_string();
                }
                Err(err) => {
                    controller.status_banner = format!("Token rotation failed: {err}");
                }
            }
        }
    });

    ui.horizontal_wrapped(|ui| {
        ui.label("Client Preset:");
        egui::ComboBox::from_id_salt("mcp_bridge_client_target")
            .selected_text(controller.selected_client.label())
            .show_ui(ui, |ui| {
                for &target in McpClientTarget::ALL {
                    ui.selectable_value(&mut controller.selected_client, target, target.label());
                }
            });
        if ui.button("Copy MCP Config").clicked() {
            let cfg_json = controller.build_client_config_json();
            ui.ctx().copy_text(cfg_json);
            controller.status_banner = format!(
                "Copied {} MCP JSON config to clipboard (includes live bearer token)",
                controller.selected_client.label()
            );
            response.copied_config = true;
        }
        let toggle_label = if controller.show_config_json {
            "Hide JSON"
        } else {
            "Show JSON"
        };
        if ui.small_button(toggle_label).clicked() {
            controller.show_config_json = !controller.show_config_json;
        }
    });

    if controller.show_config_json {
        if !controller.reveal_token {
            ui.weak("Config preview (bearer token masked; click Show above to reveal)");
        }
        let cfg_json = controller.build_client_config_json_preview();
        let mut preview = cfg_json.as_str();
        egui::ScrollArea::vertical()
            .id_salt("mcp_bridge_config_json_scroll")
            .max_height(140.0)
            .show(ui, |ui| {
                ui.add(
                    egui::TextEdit::multiline(&mut preview)
                        .font(egui::TextStyle::Monospace)
                        .desired_width(f32::INFINITY),
                );
            });
    }

    if !controller.status_banner.is_empty() {
        ui.small(&controller.status_banner);
    }

    ui.separator();

    // 4. Live AI Action Audit Log & 1-Click Undo/Redo Controls
    ui.horizontal_wrapped(|ui| {
        ui.strong(format!(
            "AI Curation Audit Log (Undo: {undo_depth}/{MAX_MCP_UNDO_DEPTH}, Redo: {redo_depth})"
        ));
        if ui
            .add_enabled(undo_depth > 0, egui::Button::new("Undo AI Action"))
            .clicked()
        {
            if let Some(cmd) = controller.undo_last() {
                controller.status_banner = format!("Undid {}", cmd.summary());
                response.undone_command = Some(cmd);
            }
        }
        if ui
            .add_enabled(redo_depth > 0, egui::Button::new("Redo"))
            .clicked()
        {
            if let Some(cmd) = controller.redo_last() {
                controller.status_banner = format!("Redid {}", cmd.summary());
                response.redone_command = Some(cmd);
            }
        }
    });

    egui::ScrollArea::vertical()
        .id_salt("mcp_bridge_audit_scroll")
        .max_height(160.0)
        .show(ui, |ui| {
            if audit_rows.is_empty() {
                ui.weak("No AI navigation or curation actions recorded yet.");
            } else {
                for row in audit_rows.iter().rev() {
                    let color = match row.outcome {
                        McpAuditOutcome::Applied | McpAuditOutcome::Redone => {
                            egui::Color32::from_rgb(120, 220, 150)
                        }
                        McpAuditOutcome::Undone => egui::Color32::from_rgb(240, 200, 95),
                        McpAuditOutcome::BlockedReadOnly | McpAuditOutcome::BlockedSecurity => {
                            egui::Color32::from_rgb(245, 110, 110)
                        }
                    };
                    ui.horizontal(|ui| {
                        ui.colored_label(color, format!("[{}]", row.outcome.badge()));
                        ui.monospace(format!("({}) {}", row.seq, row.detail));
                    });
                }
            }
        });

    response
}

// ============================================================================
// 6. Unit & Loopback E2E Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(target_arch = "wasm32"))]
    use std::io::{BufRead, BufReader};

    #[cfg(not(target_arch = "wasm32"))]
    struct TempTestDir {
        path: PathBuf,
    }

    #[cfg(not(target_arch = "wasm32"))]
    impl TempTestDir {
        fn new(tag: &str) -> Self {
            static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let raw = std::env::temp_dir().join(format!(
                "ferric_mcp_test_{tag}_{}_{}",
                std::process::id(),
                n
            ));
            std::fs::create_dir_all(&raw).expect("create temp test dir");
            let canon = std::fs::canonicalize(&raw).expect("canonicalize temp test dir");
            Self { path: canon }
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    impl Drop for TempTestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    /// Parse an RFC 2083 PNG stream and verify its signature, `IHDR`, `IDAT` ZLIB/DEFLATE stored
    /// blocks + Adler-32, `IEND`, every chunk's IEEE CRC-32, and lossless RGBA pixel equality.
    fn verify_and_decode_png(png: &[u8]) -> Result<(u32, u32, Vec<u8>), String> {
        if png.len() < 8 || png[..8] != PNG_SIGNATURE {
            return Err("invalid PNG 8-byte signature".to_string());
        }
        let mut pos = 8usize;
        let mut width = 0u32;
        let mut height = 0u32;
        let mut seen_ihdr = false;
        let mut seen_iend = false;
        let mut idat_payload = Vec::new();

        while pos < png.len() {
            if pos + 12 > png.len() {
                return Err("truncated PNG chunk header".to_string());
            }
            let len = u32::from_be_bytes(png[pos..pos + 4].try_into().unwrap()) as usize;
            let chunk_type = &png[pos + 4..pos + 8];
            let data_start = pos + 8;
            let data_end = data_start + len;
            if data_end + 4 > png.len() {
                return Err("truncated PNG chunk data/CRC".to_string());
            }
            let expected_crc = crc32_ieee(&png[pos + 4..data_end]);
            let stored_crc = u32::from_be_bytes(png[data_end..data_end + 4].try_into().unwrap());
            if expected_crc != stored_crc {
                return Err(format!(
                    "CRC-32 mismatch on chunk {:?}: expected 0x{expected_crc:08x}, got 0x{stored_crc:08x}",
                    std::str::from_utf8(chunk_type).unwrap_or("????")
                ));
            }
            let data = &png[data_start..data_end];
            match chunk_type {
                b"IHDR" => {
                    if len != 13 {
                        return Err(format!("invalid IHDR length {len}"));
                    }
                    width = u32::from_be_bytes(data[0..4].try_into().unwrap());
                    height = u32::from_be_bytes(data[4..8].try_into().unwrap());
                    if data[8] != 8
                        || data[9] != 6
                        || data[10] != 0
                        || data[11] != 0
                        || data[12] != 0
                    {
                        return Err("unexpected IHDR bit depth / RGBA mode".to_string());
                    }
                    seen_ihdr = true;
                }
                b"IDAT" => idat_payload.extend_from_slice(data),
                b"IEND" => {
                    if len != 0 {
                        return Err("non-empty IEND chunk".to_string());
                    }
                    seen_iend = true;
                }
                _ => {}
            }
            pos = data_end + 4;
        }
        if !seen_ihdr || !seen_iend {
            return Err("missing IHDR or IEND chunk".to_string());
        }
        if idat_payload.len() < 6 {
            return Err("IDAT ZLIB stream too short".to_string());
        }
        let cmf = u16::from(idat_payload[0]);
        let flg = u16::from(idat_payload[1]);
        if (cmf & 0x0F) != 8 || !(cmf * 256 + flg).is_multiple_of(31) {
            return Err("invalid RFC 1950 ZLIB header check bits".to_string());
        }

        // Inflate stored DEFLATE (`BTYPE = 00`) blocks
        let deflate_end = idat_payload.len() - 4;
        let mut zpos = 2usize;
        let mut raw_scanlines = Vec::new();
        loop {
            if zpos >= deflate_end {
                return Err("unterminated DEFLATE stream before BFINAL".to_string());
            }
            let hdr = idat_payload[zpos];
            let bfinal = (hdr & 1) != 0;
            let btype = (hdr >> 1) & 0b11;
            if btype != 0 {
                return Err(format!(
                    "expected stored DEFLATE block (BTYPE=0), got {btype}"
                ));
            }
            if zpos + 5 > deflate_end {
                return Err("truncated stored DEFLATE block header".to_string());
            }
            let blen = u16::from_le_bytes(idat_payload[zpos + 1..zpos + 3].try_into().unwrap());
            let nlen = u16::from_le_bytes(idat_payload[zpos + 3..zpos + 5].try_into().unwrap());
            if (blen ^ nlen) != 0xFFFF {
                return Err("stored DEFLATE LEN/NLEN complement mismatch".to_string());
            }
            let bstart = zpos + 5;
            let bend = bstart + usize::from(blen);
            if bend > deflate_end {
                return Err("stored DEFLATE block overruns ZLIB payload".to_string());
            }
            raw_scanlines.extend_from_slice(&idat_payload[bstart..bend]);
            zpos = bend;
            if bfinal {
                break;
            }
        }
        if zpos != deflate_end {
            return Err("trailing bytes after BFINAL DEFLATE block".to_string());
        }
        let stored_adler = u32::from_be_bytes(idat_payload[deflate_end..].try_into().unwrap());
        let computed_adler = adler32_rfc1950(&raw_scanlines);
        if stored_adler != computed_adler {
            return Err(format!(
                "Adler-32 mismatch: expected 0x{computed_adler:08x}, got 0x{stored_adler:08x}"
            ));
        }

        let stride = 1 + (width as usize) * 4;
        if raw_scanlines.len() != (height as usize) * stride {
            return Err("decoded scanline byte count mismatch".to_string());
        }
        let mut rgba = Vec::with_capacity((width as usize) * (height as usize) * 4);
        for row in raw_scanlines.chunks_exact(stride) {
            if row[0] != 0 {
                return Err(format!("expected filter 0, got {}", row[0]));
            }
            rgba.extend_from_slice(&row[1..]);
        }
        Ok((width, height, rgba))
    }

    #[test]
    fn test_crc32_adler32_and_base64_rfc_vectors() {
        assert_eq!(crc32_ieee(b""), 0x0000_0000);
        assert_eq!(crc32_ieee(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32_ieee(b"IEND"), 0xAE42_6082);

        assert_eq!(adler32_rfc1950(b""), 1);
        assert_eq!(adler32_rfc1950(b"Wikipedia"), 0x11E6_0398);

        // Multi-chunk (> 5552 bytes) Adler-32 check against naive reference
        let large: Vec<u8> = (0..12_345usize)
            .map(|i| ((i * 37 + 11) & 0xFF) as u8)
            .collect();
        let mut ref_s1 = 1u32;
        let mut ref_s2 = 0u32;
        for &b in &large {
            ref_s1 = (ref_s1 + u32::from(b)) % 65_521;
            ref_s2 = (ref_s2 + ref_s1) % 65_521;
        }
        assert_eq!(adler32_rfc1950(&large), (ref_s2 << 16) | ref_s1);

        // RFC 4648 §10 test vectors
        let vectors = [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foob", "Zm9vYg=="),
            ("fooba", "Zm9vYmE="),
            ("foobar", "Zm9vYmFy"),
        ];
        for (plain, b64) in vectors {
            assert_eq!(encode_base64(plain.as_bytes()), b64);
            assert_eq!(decode_base64(b64).unwrap(), plain.as_bytes());
        }
    }

    #[test]
    fn test_encode_rgba_to_png_lossless_single_and_multiblock() {
        // Small 4x3 RGBA image
        let small_rgba: Vec<u8> = (0..(4 * 3 * 4)).map(|i| (i * 5 + 3) as u8).collect();
        let png_small = encode_rgba_to_png(4, 3, &small_rgba);
        let (w, h, dec_small) = verify_and_decode_png(&png_small).expect("valid 4x3 PNG");
        assert_eq!((w, h), (4, 3));
        assert_eq!(dec_small, small_rgba);

        // Larger 180x100 RGBA image (72,100 filtered bytes > 65,535 -> multi-block stored DEFLATE)
        let big_rgba: Vec<u8> = (0..(180 * 100 * 4))
            .map(|i| ((i * 131 + 17) & 0xFF) as u8)
            .collect();
        let png_big = encode_rgba_to_png(180, 100, &big_rgba);
        let (bw, bh, dec_big) = verify_and_decode_png(&png_big).expect("valid 180x100 PNG");
        assert_eq!((bw, bh), (180, 100));
        assert_eq!(dec_big, big_rgba);

        // MCP multimodal content block schema check
        let block = build_mcp_png_image_block(4, 3, &small_rgba);
        assert_eq!(block["type"], "image");
        assert_eq!(block["mimeType"], "image/png");
        let b64_bytes = decode_base64(block["data"].as_str().unwrap()).unwrap();
        assert_eq!(b64_bytes, png_small);
    }

    #[test]
    fn test_undo_log_all_commands_bijection_and_256_ring_cap() {
        let mut snap = WorkbenchSnapshot {
            active_residue_seq: Some(5),
            crosshair_ppm: (8.31, 121.4, Some(54.2)),
            active_experiment: "HNCACB".to_string(),
            strip_candidates: vec![StripCandidateSummary {
                cluster_id: 2,
                direction: "i-1".to_string(),
                pair_score: 6.4,
                rel_prob: 0.91,
                assigned_seq_id: Some(4),
            }],
            selected_pdb_chain_res: Some(("A".to_string(), 5)),
            undo_depth: 0,
            redo_depth: 0,
            qc_status: "PASS".to_string(),
            assigned_residues: BTreeMap::new(),
            excluded_peaks: BTreeSet::new(),
            highlighted_residues_3d: vec![5],
            ..WorkbenchSnapshot::default()
        };
        let mut log = McpUndoLog::new();
        let s0 = snap.clone();

        log.apply_command(&mut snap, McpUiCommand::JumpToResidue { seq_id: 12 });
        let s1 = snap.clone();
        assert_eq!(s1.active_residue_seq, Some(12));
        assert_eq!(s1.selected_pdb_chain_res, Some(("A".to_string(), 12)));

        log.apply_command(
            &mut snap,
            McpUiCommand::SetCrosshairPpm {
                h_ppm: 7.88,
                n_ppm: 116.2,
                c_ppm: None,
            },
        );
        let s2 = snap.clone();
        assert_eq!(s2.crosshair_ppm, (7.88, 116.2, None));

        log.apply_command(
            &mut snap,
            McpUiCommand::AssignSpinSystemToResidue {
                cluster_id: 3,
                seq_id: 10,
                reason: "Initial placement".to_string(),
            },
        );
        let s3 = snap.clone();

        log.apply_command(
            &mut snap,
            McpUiCommand::AssignSpinSystemToResidue {
                cluster_id: 7,
                seq_id: 20,
                reason: "Target occupant".to_string(),
            },
        );
        let s4 = snap.clone();

        // Bijective displacement: move cluster 3 from seq 10 onto seq 20 (displacing cluster 7)
        log.apply_command(
            &mut snap,
            McpUiCommand::AssignSpinSystemToResidue {
                cluster_id: 3,
                seq_id: 20,
                reason: "Reassign cluster 3 -> seq 20".to_string(),
            },
        );
        assert!(!snap.assigned_residues.contains_key(&10));
        assert_eq!(snap.assigned_residues[&20].cluster_id, 3);
        let s5 = snap.clone();

        log.apply_command(
            &mut snap,
            McpUiCommand::UnassignResidue {
                seq_id: 20,
                reason: "Clear ambiguous".to_string(),
            },
        );
        let s6 = snap.clone();
        assert!(s6.assigned_residues.is_empty());

        log.apply_command(
            &mut snap,
            McpUiCommand::TogglePeakExclusion {
                peak_id: 42,
                excluded: true,
            },
        );
        let s7 = snap.clone();
        assert!(s7.excluded_peaks.contains(&42));

        log.apply_command(
            &mut snap,
            McpUiCommand::HighlightResidues3d {
                seq_ids: vec![18, 19, 20],
            },
        );
        let s8 = snap.clone();
        assert_eq!(s8.highlighted_residues_3d, vec![18, 19, 20]);
        assert_eq!(log.undo_depth(), 8);

        // Single-step Cmd/Ctrl+Z backward verification across all 8 mutations
        let mut expected_back = [&s7, &s6, &s5, &s4, &s3, &s2, &s1, &s0];
        for target in &mut expected_back {
            assert!(log.undo_last(&mut snap).is_some());
            assert_eq!(snap.active_residue_seq, target.active_residue_seq);
            assert_eq!(snap.crosshair_ppm, target.crosshair_ppm);
            assert_eq!(snap.selected_pdb_chain_res, target.selected_pdb_chain_res);
            assert_eq!(snap.assigned_residues, target.assigned_residues);
            assert_eq!(snap.excluded_peaks, target.excluded_peaks);
            assert_eq!(snap.highlighted_residues_3d, target.highlighted_residues_3d);
        }
        assert_eq!(log.undo_depth(), 0);
        assert_eq!(log.redo_depth(), 8);

        // Forward Redo replay verification
        let expected_fwd = [&s1, &s2, &s3, &s4, &s5, &s6, &s7, &s8];
        for target in expected_fwd {
            assert!(log.redo_last(&mut snap).is_some());
            assert_eq!(snap.assigned_residues, target.assigned_residues);
            assert_eq!(snap.excluded_peaks, target.excluded_peaks);
            assert_eq!(snap.highlighted_residues_3d, target.highlighted_residues_3d);
        }

        // Verify strict 256-step ring-buffer bound
        for seq_id in 1..=300 {
            log.apply_command(&mut snap, McpUiCommand::JumpToResidue { seq_id });
        }
        assert_eq!(log.undo_depth(), MAX_MCP_UNDO_DEPTH);
        assert_eq!(snap.undo_depth, MAX_MCP_UNDO_DEPTH);
    }

    #[test]
    fn test_constant_time_eq_and_client_config_builders() {
        assert!(constant_time_eq(
            b"nmr_mcp_secret_123",
            b"nmr_mcp_secret_123"
        ));
        assert!(!constant_time_eq(
            b"nmr_mcp_secret_123",
            b"nmr_mcp_secret_124"
        ));
        assert!(!constant_time_eq(b"nmr_mcp_secret_123", b"nmr_mcp_secret"));
        assert!(!constant_time_eq(b"nmr_mcp_secret_123", b""));
        assert!(constant_time_eq(b"", b""));

        let tok = generate_bearer_token().expect("OS CSPRNG token 1");
        let tok2 = generate_bearer_token().expect("OS CSPRNG token 2");
        assert!(tok.starts_with("nmr_mcp_"));
        assert_eq!(tok.len(), 72);
        assert_ne!(tok, tok2);
        let hex_tail = &tok["nmr_mcp_".len()..];
        assert_eq!(hex_tail.len(), 64);
        assert!(hex_tail.chars().all(|c| matches!(c, '0'..='9' | 'a'..='f')));
        assert_ne!(hex_tail, "0".repeat(64));
        #[cfg(any(unix, windows))]
        {
            let mut probe = [0u8; 32];
            assert!(fill_os_entropy(&mut probe));
            assert!(probe.iter().any(|&b| b != 0));
        }
        // Fail-closed CSPRNG contract (`17.3` hardening): simulated OS RNG failure or all-zero
        // output MUST return `Err(...)` and refuse to start a live TCP listener rather than
        // falling back to `RandomState`.
        assert!(try_generate_bearer_token_with(|_| false).is_err());
        assert!(try_generate_bearer_token_with(|buf| {
            buf.fill(0);
            true
        })
        .is_err());
        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut fail_ctrl = McpBridgeController::default();
            if let Ok(mut g) = fail_ctrl.shared.lock() {
                g.bearer_token.clear();
                g.bearer_token_csprng = false;
            }
            let err = fail_ctrl
                .start_bridge_with_entropy("127.0.0.1:0", |_| false)
                .expect_err("start_bridge must fail closed when OS CSPRNG fails");
            assert!(err.contains("OS CSPRNG"), "got: {err}");
            assert!(!fail_ctrl.is_running());
            assert!(fail_ctrl.bound_addr_string().is_none());
        }
        let masked = mask_bearer_token(&tok);
        assert!(masked.contains("••••••••••••"));
        assert!(!masked.contains(&tok[16..48]));

        for &client in McpClientTarget::ALL {
            let cfg_str = build_mcp_client_config_json(
                client,
                "http://127.0.0.1:54321/mcp",
                "nmr_mcp_testtoken_0123456789abcdef",
            );
            let core_expected = nmr_core::mcp::generate_mcp_client_config_pretty(
                client.to_core_target(),
                &nmr_core::mcp::McpClientMode::HttpBridge {
                    endpoint_url: "http://127.0.0.1:54321/mcp".to_string(),
                    bearer_token: "nmr_mcp_testtoken_0123456789abcdef".to_string(),
                    access_mode: nmr_core::mcp::McpAccessMode::ReadOnly,
                },
            )
            .expect("core config pretty");
            assert_eq!(
                cfg_str, core_expected,
                "GUI bridge config for {client:?} must match nmr_core::mcp::generate_mcp_client_config_pretty byte-for-byte"
            );

            let parsed: Value = serde_json::from_str(&cfg_str).expect("valid client config JSON");
            let entry = &parsed["mcpServers"]["ferric-workbench"];
            assert_eq!(
                entry["headers"]["Authorization"],
                "Bearer nmr_mcp_testtoken_0123456789abcdef"
            );
            let url_val = entry
                .get("serverUrl")
                .or_else(|| entry.get("url"))
                .and_then(Value::as_str);
            assert_eq!(url_val, Some("http://127.0.0.1:54321/mcp"));
            if client == McpClientTarget::ClaudeDesktop {
                assert_eq!(entry["transport"], "streamable-http");
            }
        }
    }

    #[test]
    fn test_headless_egui_mcp_bridge_panel_renders() {
        let ctrl = McpBridgeController::default();
        ctrl.set_permission(McpBridgePermission::ReadWrite);
        let _ = ctrl.apply_ui_command(McpUiCommand::JumpToResidue { seq_id: 9 });
        let mut ctrl = ctrl;

        let ctx = egui::Context::default();
        let mut out = ctx.run_ui(egui::RawInput::default(), |ui| {
            let resp = render_mcp_bridge_panel(ui, &mut ctrl);
            assert!(!resp.copied_config);
        });
        out.textures_delta.clear();
    }

    #[test]
    fn test_in_memory_jsonrpc_dispatch_and_error_codes() {
        let ctrl = McpBridgeController::default();

        // Malformed JSON -> -32700
        let err_parse = ctrl.handle_jsonrpc_bytes(b"{not-json").unwrap();
        assert_eq!(err_parse["error"]["code"], -32700);

        // Invalid JSON-RPC envelope -> -32600
        let err_req = ctrl
            .handle_jsonrpc_bytes(br#"{"jsonrpc":"1.0","id":1,"method":"ping"}"#)
            .unwrap();
        assert_eq!(err_req["error"]["code"], -32600);

        // Unknown method -> -32601
        let err_method = ctrl
            .handle_jsonrpc_bytes(br#"{"jsonrpc":"2.0","id":2,"method":"unknown/op"}"#)
            .unwrap();
        assert_eq!(err_method["error"]["code"], -32601);

        // Unknown tool -> -32602
        let err_tool = ctrl
            .handle_jsonrpc_bytes(
                br#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"no_such_tool"}}"#,
            )
            .unwrap();
        assert_eq!(err_tool["error"]["code"], -32602);

        // Notification returns None
        assert!(ctrl
            .handle_jsonrpc_bytes(br#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#)
            .is_none());

        // ReadWrite navigation + redo round-trip
        ctrl.set_permission(McpBridgePermission::ReadWrite);
        let nav = ctrl
            .handle_jsonrpc_bytes(
                serde_json::to_vec(&json!({
                    "jsonrpc": "2.0",
                    "id": 4,
                    "method": "tools/call",
                    "params": {
                        "name": "navigate_workbench",
                        "arguments": {
                            "action": "set_crosshair_ppm",
                            "h_ppm": 8.12,
                            "n_ppm": 117.5,
                            "c_ppm": 58.3
                        }
                    }
                }))
                .unwrap()
                .as_slice(),
            )
            .unwrap();
        assert_eq!(nav["result"]["isError"], false);
        assert_eq!(ctrl.snapshot().crosshair_ppm, (8.12, 117.5, Some(58.3)));

        let _ = ctrl.undo_last();
        let redo_resp = ctrl
            .handle_jsonrpc_bytes(
                serde_json::to_vec(&json!({
                    "jsonrpc": "2.0",
                    "id": 5,
                    "method": "tools/call",
                    "params": {
                        "name": "undo_last_ai_edit",
                        "arguments": { "direction": "redo" }
                    }
                }))
                .unwrap()
                .as_slice(),
            )
            .unwrap();
        assert_eq!(redo_resp["result"]["isError"], false);
        assert_eq!(ctrl.snapshot().crosshair_ppm, (8.12, 117.5, Some(58.3)));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_non_loopback_bind_rejected() {
        assert!(parse_strict_loopback_bind("0.0.0.0:0").is_err());
        assert!(parse_strict_loopback_bind("192.168.1.20:8080").is_err());
        assert!(parse_strict_loopback_bind("[::]:0").is_err());
        assert!(parse_strict_loopback_bind("127.0.0.1:0").is_ok());
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn send_raw_http_post(
        addr: &str,
        host_header: &str,
        auth_header: Option<&str>,
        origin_header: Option<&str>,
        body: &Value,
    ) -> (u16, Value) {
        let mut stream = TcpStream::connect(addr).expect("connect to loopback MCP bridge");
        let body_bytes = serde_json::to_vec(body).unwrap();
        let mut req = format!(
            "POST /mcp HTTP/1.1\r\nHost: {host_header}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n",
            body_bytes.len()
        );
        if let Some(auth) = auth_header {
            req.push_str(&format!("Authorization: {auth}\r\n"));
        }
        if let Some(orig) = origin_header {
            req.push_str(&format!("Origin: {orig}\r\n"));
        }
        req.push_str("Connection: close\r\n\r\n");
        stream.write_all(req.as_bytes()).unwrap();
        stream.write_all(&body_bytes).unwrap();
        stream.flush().unwrap();

        let mut resp_bytes = Vec::new();
        stream.read_to_end(&mut resp_bytes).unwrap();
        let resp_str = String::from_utf8_lossy(&resp_bytes);
        let (head, payload) = resp_str
            .split_once("\r\n\r\n")
            .expect("HTTP response header/body split");
        let status_line = head.lines().next().unwrap_or("");
        let status_code: u16 = status_line
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let json_val = if payload.trim().is_empty() {
            Value::Null
        } else {
            serde_json::from_str(payload).unwrap_or(Value::Null)
        };
        (status_code, json_val)
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_loopback_e2e_auth_origin_readonly_curation_undo_and_png() {
        let initial = WorkbenchSnapshot {
            active_residue_seq: Some(3),
            crosshair_ppm: (8.42, 119.8, Some(56.1)),
            active_experiment: "HNCACB".to_string(),
            strip_candidates: vec![StripCandidateSummary {
                cluster_id: 4,
                direction: "i+1".to_string(),
                pair_score: 7.25,
                rel_prob: 0.94,
                assigned_seq_id: None,
            }],
            selected_pdb_chain_res: Some(("A".to_string(), 3)),
            undo_depth: 0,
            redo_depth: 0,
            qc_status: "PASS".to_string(),
            assigned_residues: BTreeMap::new(),
            excluded_peaks: BTreeSet::new(),
            highlighted_residues_3d: vec![3],
            ..WorkbenchSnapshot::default()
        };

        let mut ctrl = McpBridgeController::new(initial.clone());
        let addr = ctrl
            .start_bridge("127.0.0.1:0")
            .expect("bind ephemeral loopback MCP port");
        let bearer = format!("Bearer {}", ctrl.bearer_token());

        // 1. Unauthenticated request -> 401 Unauthorized
        let init_req = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": { "protocolVersion": MCP_PROTOCOL_VERSION }
        });
        let (code_no_auth, _) = send_raw_http_post(&addr, &addr, None, None, &init_req);
        assert_eq!(code_no_auth, 401);

        let (code_bad_auth, _) = send_raw_http_post(
            &addr,
            &addr,
            Some("Bearer nmr_mcp_wrong_token"),
            None,
            &init_req,
        );
        assert_eq!(code_bad_auth, 401);

        // 2. Cross-origin Origin header & spoofed DNS-rebinding Host -> 403 Forbidden
        let (code_evil_origin, _) = send_raw_http_post(
            &addr,
            &addr,
            Some(&bearer),
            Some("https://evil.example"),
            &init_req,
        );
        assert_eq!(code_evil_origin, 403);

        let (code_evil_host, _) =
            send_raw_http_post(&addr, "evil.example", Some(&bearer), None, &init_req);
        assert_eq!(code_evil_host, 403);

        // 3. Authenticated initialize + tools/list
        let (code_init, resp_init) =
            send_raw_http_post(&addr, &addr, Some(&bearer), None, &init_req);
        assert_eq!(code_init, 200);
        assert_eq!(resp_init["result"]["protocolVersion"], MCP_PROTOCOL_VERSION);

        let (code_list, resp_list) = send_raw_http_post(
            &addr,
            &addr,
            Some(&bearer),
            None,
            &json!({ "jsonrpc": "2.0", "id": 2, "method": "tools/list" }),
        );
        assert_eq!(code_list, 200);
        let tool_names: Vec<&str> = resp_list["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|t| t.get("name").and_then(Value::as_str))
            .collect();
        assert_eq!(
            tool_names,
            vec![
                "get_workbench_state",
                "navigate_workbench",
                "propose_curation_edit",
                "undo_last_ai_edit",
                "capture_viewport_png",
                "run_workbench_workflow",
                "dispatch_workbench_action",
                "inspect_workbench_data"
            ]
        );

        // 4. ReadOnly mode blocks curation mutation & preserves state identically
        assert_eq!(ctrl.permission(), McpBridgePermission::ReadOnly);
        let edit_req = json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "propose_curation_edit",
                "arguments": {
                    "action": "assign_spin_system",
                    "cluster_id": 4,
                    "seq_id": 27,
                    "reason": "Corroborated HNCACB/CBCACONH match"
                }
            }
        });
        let (code_ro, resp_ro) = send_raw_http_post(&addr, &addr, Some(&bearer), None, &edit_req);
        assert_eq!(code_ro, 200);
        assert_eq!(resp_ro["result"]["isError"], true);
        assert_eq!(ctrl.snapshot(), initial);

        // 5. Switch to ReadWrite -> X-Nmr-Access-Mode: read-only header still enforces ReadOnly ->
        //    without override, curation edit succeeds -> undo_last_ai_edit -> exact restoration
        ctrl.set_permission(McpBridgePermission::ReadWrite);
        {
            let mut stream = TcpStream::connect(&addr).expect("connect for X-Nmr-Access-Mode test");
            let body_bytes = serde_json::to_vec(&edit_req).unwrap();
            let req = format!(
                "POST /mcp HTTP/1.1\r\nHost: {addr}\r\nAuthorization: {bearer}\r\nX-Nmr-Access-Mode: read-only\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body_bytes.len()
            );
            stream.write_all(req.as_bytes()).unwrap();
            stream.write_all(&body_bytes).unwrap();
            stream.flush().unwrap();
            let mut resp_bytes = Vec::new();
            stream.read_to_end(&mut resp_bytes).unwrap();
            let resp_str = String::from_utf8_lossy(&resp_bytes);
            let (_, payload) = resp_str.split_once("\r\n\r\n").expect("HTTP split");
            let resp_hdr_ro: Value = serde_json::from_str(payload).unwrap();
            assert_eq!(resp_hdr_ro["result"]["isError"], true);
            assert_eq!(ctrl.snapshot(), initial);
            assert_eq!(ctrl.permission(), McpBridgePermission::ReadWrite);
        }
        let (code_rw, resp_rw) = send_raw_http_post(&addr, &addr, Some(&bearer), None, &edit_req);
        assert_eq!(code_rw, 200);
        assert_eq!(resp_rw["result"]["isError"], false);

        let (code_state, resp_state) = send_raw_http_post(
            &addr,
            &addr,
            Some(&bearer),
            None,
            &json!({
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": { "name": "get_workbench_state", "arguments": {} }
            }),
        );
        assert_eq!(code_state, 200);
        let state_json: Value =
            serde_json::from_str(resp_state["result"]["content"][0]["text"].as_str().unwrap())
                .unwrap();
        assert_eq!(state_json["assigned_residues"]["27"]["cluster_id"], 4);
        assert_eq!(state_json["undo_depth"], 1);

        let (code_undo, resp_undo) = send_raw_http_post(
            &addr,
            &addr,
            Some(&bearer),
            None,
            &json!({
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": { "name": "undo_last_ai_edit", "arguments": { "direction": "undo" } }
            }),
        );
        assert_eq!(code_undo, 200);
        assert_eq!(resp_undo["result"]["isError"], false);

        let restored = ctrl.snapshot();
        assert_eq!(restored.active_residue_seq, initial.active_residue_seq);
        assert_eq!(restored.crosshair_ppm, initial.crosshair_ppm);
        assert_eq!(restored.assigned_residues, initial.assigned_residues);
        assert_eq!(restored.excluded_peaks, initial.excluded_peaks);
        assert_eq!(restored.undo_depth, 0);
        assert_eq!(restored.redo_depth, 1);

        // 6. capture_viewport_png returns valid multimodal PNG block (magic + CRC-32 + Adler-32)
        let custom_rgba: Vec<u8> = (0..(16 * 12 * 4))
            .map(|i| ((i * 29 + 7) & 0xFF) as u8)
            .collect();
        ctrl.update_viewport_rgba(16, 12, custom_rgba.clone(), "contour_2d");

        let (code_png, resp_png) = send_raw_http_post(
            &addr,
            &addr,
            Some(&bearer),
            None,
            &json!({
                "jsonrpc": "2.0",
                "id": 6,
                "method": "tools/call",
                "params": {
                    "name": "capture_viewport_png",
                    "arguments": { "viewport": "contour_2d" }
                }
            }),
        );
        assert_eq!(code_png, 200);
        assert_eq!(resp_png["result"]["isError"], false);
        let img_block = &resp_png["result"]["content"][0];
        assert_eq!(img_block["type"], "image");
        assert_eq!(img_block["mimeType"], "image/png");
        let png_bytes = decode_base64(img_block["data"].as_str().unwrap()).expect("valid Base64");
        let (pw, ph, decoded_rgba) =
            verify_and_decode_png(&png_bytes).expect("valid PNG magic/CRC-32/Adler-32");
        assert_eq!((pw, ph), (16, 12));
        assert_eq!(decoded_rgba, custom_rgba);

        // 7. Adversarial pre-allocation Content-Length cap (oversized 64 MiB and usize overflow -> 413)
        for adv_len in ["67108864", "999999999999999999999999999999"] {
            let mut sock = TcpStream::connect(&addr).expect("connect loopback for 413 test");
            let req_413 = format!(
                "POST /mcp HTTP/1.1\r\nHost: {addr}\r\nAuthorization: {bearer}\r\nContent-Length: {adv_len}\r\nConnection: close\r\n\r\n{{}}"
            );
            sock.write_all(req_413.as_bytes()).expect("send 413 header");
            let _ = sock.flush();
            let mut resp_buf = String::new();
            let mut rdr = BufReader::new(sock);
            let _ = rdr.read_line(&mut resp_buf);
            assert!(
                resp_buf.contains(" 413 "),
                "expected 413 Payload Too Large before allocation for Content-Length={adv_len}, got {resp_buf:?}"
            );
        }

        // Negative / non-numeric Content-Length -> 400 Bad Request
        {
            let mut sock = TcpStream::connect(&addr).expect("connect loopback for 400 test");
            let req_400 = format!(
                "POST /mcp HTTP/1.1\r\nHost: {addr}\r\nAuthorization: {bearer}\r\nContent-Length: -42\r\nConnection: close\r\n\r\n"
            );
            sock.write_all(req_400.as_bytes()).expect("send 400 header");
            let _ = sock.flush();
            let mut resp_buf = String::new();
            let mut rdr = BufReader::new(sock);
            let _ = rdr.read_line(&mut resp_buf);
            assert!(
                resp_buf.contains(" 400 "),
                "expected 400 Bad Request for negative Content-Length, got {resp_buf:?}"
            );
        }

        println!(
            "MCP_METRIC_JSON:{}",
            json!({
                "suite": "nmr_gui_mcp_bridge",
                "undo_cap_depth": MAX_MCP_UNDO_DEPTH,
                "max_http_body_bytes": MAX_MCP_HTTP_BODY_BYTES,
                "png_width": pw,
                "png_height": ph,
                "png_bytes": png_bytes.len(),
                "http_auth_401_verified": true,
                "http_rebind_403_verified": true,
                "http_prealloc_413_verified": true,
                "readonly_write_blocked_verified": true,
                "undo_roundtrip_verified": true
            })
        );

        ctrl.stop_bridge();
        assert!(!ctrl.is_running());
    }

    #[test]
    fn test_protocol_version_negotiation_and_show_config_json_preview() {
        let mut ctrl = McpBridgeController::default();

        // Requested supported version 2025-03-26 is echoed verbatim
        let init_2025 = ctrl
            .handle_jsonrpc_bytes(
                br#"{"jsonrpc":"2.0","id":10,"method":"initialize","params":{"protocolVersion":"2025-03-26"}}"#,
            )
            .expect("initialize response");
        assert_eq!(init_2025["result"]["protocolVersion"], "2025-03-26");

        // Legacy supported version 2024-11-05 is echoed verbatim
        let init_2024 = ctrl
            .handle_jsonrpc_bytes(
                br#"{"jsonrpc":"2.0","id":11,"method":"initialize","params":{"protocolVersion":"2024-11-05"}}"#,
            )
            .expect("initialize response");
        assert_eq!(init_2024["result"]["protocolVersion"], "2024-11-05");

        // Unsupported version falls back safely to default MCP_PROTOCOL_VERSION
        let init_unknown = ctrl
            .handle_jsonrpc_bytes(
                br#"{"jsonrpc":"2.0","id":12,"method":"initialize","params":{"protocolVersion":"1999-01-01"}}"#,
            )
            .expect("initialize response");
        assert_eq!(
            init_unknown["result"]["protocolVersion"],
            MCP_PROTOCOL_VERSION
        );

        // Headless egui renders cleanly with Show JSON preview toggled open
        ctrl.show_config_json = true;
        let ctx = egui::Context::default();
        let mut out = ctx.run_ui(egui::RawInput::default(), |ui| {
            let resp = render_mcp_bridge_panel(ui, &mut ctrl);
            assert!(!resp.copied_config);
        });
        out.textures_delta.clear();
    }

    #[test]
    fn test_enriched_workbench_snapshot_defaults_and_get_workbench_state() {
        let def = WorkbenchSnapshot::default();
        assert_eq!(def.sequence_len, 0);
        assert!(def.loaded_spectra.is_empty());
        assert_eq!(def.peak_count, 0);
        assert_eq!(def.spin_system_count, 0);
        assert_eq!(def.assigned_count, 0);
        assert_eq!(def.high_conf_count, 0);
        assert!(def.status_msg.is_empty());
        assert!(!def.ingest_in_progress);
        assert_eq!(def.picker_mode, "classical");
        assert!((def.picker_snr - 5.0).abs() < 1e-12);
        assert_eq!(def.workspace_mode, "ProteinNmr");
        assert_eq!(def.active_page_index, 0);
        assert_eq!(def.solver_kind, "evo-graph");
        assert!((def.contour_base - 0.0).abs() < 1e-12);
        assert_eq!(def.view_bounds, [10.5, 6.0, 102.0, 134.0]);
        assert!(def.ingest_progress_text.is_empty());
        assert!(def.peaks_by_experiment.is_empty());

        let mut peaks_by_exp = BTreeMap::new();
        peaks_by_exp.insert("HSQC".to_string(), 74usize);
        peaks_by_exp.insert("HNCACB".to_string(), 68usize);

        let populated = WorkbenchSnapshot {
            sequence_len: 76,
            loaded_spectra: vec!["15N-HSQC".to_string(), "HNCACB".to_string()],
            peak_count: 142,
            spin_system_count: 68,
            assigned_count: 61,
            high_conf_count: 54,
            status_msg: "Assigned 61/76 residues".to_string(),
            ingest_in_progress: true,
            picker_mode: "neural".to_string(),
            picker_snr: 8.5,
            workspace_mode: "Dosy".to_string(),
            active_page_index: 2,
            solver_kind: "bayes-sa".to_string(),
            contour_base: 12500.0,
            view_bounds: [9.8, 6.4, 105.0, 131.0],
            ingest_progress_text: "Loading 2/4 HNCACB".to_string(),
            peaks_by_experiment: peaks_by_exp,
            ..WorkbenchSnapshot::default()
        };
        let ctrl = McpBridgeController::new(populated);
        let resp = ctrl
            .handle_jsonrpc_bytes(
                br#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"get_workbench_state","arguments":{}}}"#,
            )
            .expect("get_workbench_state response");
        assert_eq!(resp["result"]["isError"], false);
        let state: Value =
            serde_json::from_str(resp["result"]["content"][0]["text"].as_str().unwrap())
                .expect("valid JSON text");
        assert_eq!(state["sequence_len"], 76);
        assert_eq!(state["loaded_spectra"], json!(["15N-HSQC", "HNCACB"]));
        assert_eq!(state["peak_count"], 142);
        assert_eq!(state["spin_system_count"], 68);
        assert_eq!(state["assigned_count"], 61);
        assert_eq!(state["high_conf_count"], 54);
        assert_eq!(state["status_msg"], "Assigned 61/76 residues");
        assert_eq!(state["ingest_in_progress"], true);
        assert_eq!(state["picker_mode"], "neural");
        assert_eq!(state["picker_snr"], 8.5);
        assert_eq!(state["workspace_mode"], "Dosy");
        assert_eq!(state["active_page_index"], 2);
        assert_eq!(state["solver_kind"], "bayes-sa");
        assert_eq!(state["contour_base"], 12500.0);
        assert_eq!(state["view_bounds"], json!([9.8, 6.4, 105.0, 131.0]));
        assert_eq!(state["ingest_progress_text"], "Loading 2/4 HNCACB");
        assert_eq!(
            state["peaks_by_experiment"],
            json!({ "HSQC": 74, "HNCACB": 68 })
        );
        assert_eq!(state["permission"], "read_only");
    }

    #[test]
    fn test_run_workbench_workflow_readonly_refusal_validation_and_readwrite_queueing() {
        let ctrl = McpBridgeController::default();

        // 1. tools/list exposes run_workbench_workflow with all 9 actions
        let list_resp = ctrl
            .handle_jsonrpc_bytes(br#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#)
            .expect("tools/list response");
        let tools = list_resp["result"]["tools"].as_array().unwrap();
        let wf_tool = tools
            .iter()
            .find(|t| t.get("name").and_then(Value::as_str) == Some("run_workbench_workflow"))
            .expect("run_workbench_workflow in tools/list");
        assert_eq!(wf_tool["inputSchema"]["required"], json!(["action"]));
        assert_eq!(
            wf_tool["inputSchema"]["properties"]["action"]["enum"],
            json!([
                "load_study_folder",
                "set_sequence",
                "pick_peaks",
                "run_assignment",
                "select_workbook_page",
                "set_contour_level",
                "set_viewport_bounds",
                "export_results",
                "set_lock_residue"
            ])
        );
        assert_eq!(
            wf_tool["inputSchema"]["properties"]["solver"]["enum"],
            json!(["evo-graph", "standard", "bayes-sa"])
        );

        // 2. ReadOnly mode refuses workflow commands and records BlockedReadOnly audit
        assert_eq!(ctrl.permission(), McpBridgePermission::ReadOnly);
        for args in [
            json!({ "action": "load_study_folder", "path": "/tmp/study" }),
            json!({ "action": "set_sequence", "sequence": "MQIFVK" }),
            json!({ "action": "pick_peaks", "snr_threshold": 6.5 }),
            json!({ "action": "run_assignment", "solver": "evo-graph" }),
            json!({ "action": "select_workbook_page", "page_index": 1 }),
            json!({ "action": "set_contour_level", "contour_base": 5000.0 }),
            json!({ "action": "set_viewport_bounds", "h_left": 10.0, "h_right": 6.5, "n_top": 105.0, "n_bottom": 132.0 }),
            json!({ "action": "export_results", "prefix": "/tmp/out", "format": "star" }),
            json!({ "action": "set_lock_residue", "seq_id": 12, "locked": true }),
        ] {
            let req = serde_json::to_vec(&json!({
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "run_workbench_workflow",
                    "arguments": args,
                }
            }))
            .unwrap();
            let ro_resp = ctrl.handle_jsonrpc_bytes(&req).unwrap();
            assert_eq!(ro_resp["result"]["isError"], true);
            let ro_payload: Value =
                serde_json::from_str(ro_resp["result"]["content"][0]["text"].as_str().unwrap())
                    .unwrap();
            assert_eq!(ro_payload["queued"], false);
            assert_eq!(ro_payload["status"], "denied");
            assert_eq!(ro_payload["permission"], "read_only");
        }
        assert!(ctrl.drain_pending_ui_commands().is_empty());
        {
            let guard = ctrl.shared.lock().unwrap();
            assert_eq!(guard.undo_log.audit_records().len(), 9);
            assert!(guard
                .undo_log
                .audit_records()
                .iter()
                .all(|r| r.outcome == McpAuditOutcome::BlockedReadOnly));
        }

        // 3. Invalid arguments return JSON-RPC -32602 even in ReadWrite mode
        ctrl.set_permission(McpBridgePermission::ReadWrite);
        for bad_args in [
            json!({ "action": "unknown_action" }),
            json!({ "action": "load_study_folder", "path": "   " }),
            json!({ "action": "load_study_folder", "path": "/tmp/s", "run_full_pipeline": "yes" }),
            json!({ "action": "set_sequence", "sequence": "" }),
            json!({ "action": "set_sequence" }),
            json!({ "action": "pick_peaks", "snr_threshold": -1.0 }),
            json!({ "action": "pick_peaks", "auto_threshold": "true" }),
            json!({ "action": "pick_peaks", "picker_mode": "invalid" }),
            json!({ "action": "run_assignment", "solver": "invalid-solver" }),
            json!({ "action": "run_assignment", "engine": "tpu" }),
            json!({ "action": "select_workbook_page" }),
            json!({ "action": "select_workbook_page", "page_index": -1 }),
            json!({ "action": "set_contour_level" }),
            json!({ "action": "set_contour_level", "multiplier": 0.0 }),
            json!({ "action": "set_viewport_bounds", "h_left": 8.0, "h_right": 8.0, "n_top": 105.0, "n_bottom": 130.0 }),
            json!({ "action": "export_results", "prefix": "", "format": "star" }),
            json!({ "action": "export_results", "prefix": "/tmp/out", "format": "pdf" }),
            json!({ "action": "set_lock_residue", "seq_id": 12 }),
        ] {
            let req = serde_json::to_vec(&json!({
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "run_workbench_workflow",
                    "arguments": bad_args,
                }
            }))
            .unwrap();
            let err_resp = ctrl.handle_jsonrpc_bytes(&req).unwrap();
            assert_eq!(err_resp["error"]["code"], -32602, "bad_args: {bad_args}");
        }

        // 4. X-Nmr-Access-Mode read-only override still blocks workflow execution
        {
            let req = serde_json::to_vec(&json!({
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "run_workbench_workflow",
                    "arguments": { "action": "run_assignment", "solver": "standard" }
                }
            }))
            .unwrap();
            let mut guard = ctrl.shared.lock().unwrap();
            let ov_resp = dispatch_mcp_jsonrpc_with_permission(
                &mut guard,
                &req,
                Some(McpBridgePermission::ReadOnly),
            )
            .unwrap();
            assert_eq!(ov_resp["result"]["isError"], true);
            assert_eq!(
                guard.undo_log.audit_records().back().unwrap().outcome,
                McpAuditOutcome::BlockedReadOnly
            );
        }
        assert!(ctrl.drain_pending_ui_commands().is_empty());

        // 5. ReadWrite mode queues all 9 workflow actions + curation undo compatibility
        let sandbox_dir = TempTestDir::new("workflow");
        let study_dir = sandbox_dir.path().join("ubq_study");
        std::fs::create_dir_all(&study_dir).expect("create study dir");
        let fasta_path = sandbox_dir.path().join("ubq.fasta");
        std::fs::write(&fasta_path, ">UBQ\nMQIFVKTLTG\n").expect("create fasta");
        let export_prefix = sandbox_dir.path().join("ubq_export");
        ctrl.set_approved_roots([sandbox_dir.path()])
            .expect("register approved root");

        let study_str = study_dir.to_string_lossy().into_owned();
        let fasta_str = fasta_path.to_string_lossy().into_owned();
        let export_str = export_prefix.to_string_lossy().into_owned();

        let _ = ctrl.apply_ui_command(McpUiCommand::AssignSpinSystemToResidue {
            cluster_id: 2,
            seq_id: 14,
            reason: "Anchor check".to_string(),
        });
        let _ = ctrl.drain_pending_ui_commands();
        ctrl.set_destructive_actions_allowed(true);

        let valid_calls = [
            (
                "load_study_folder",
                json!({
                    "action": "load_study_folder",
                    "path": study_str,
                    "run_full_pipeline": true,
                    "confirm": true
                }),
            ),
            (
                "set_sequence",
                json!({
                    "action": "set_sequence",
                    "sequence": "MQIFVKTLTG",
                    "confirm": true
                }),
            ),
            (
                "set_sequence",
                json!({
                    "action": "set_sequence",
                    "sequence_path": fasta_str,
                    "confirm": true
                }),
            ),
            (
                "pick_peaks",
                json!({
                    "action": "pick_peaks",
                    "snr_threshold": 7.5,
                    "auto_threshold": false,
                    "picker_mode": "neural"
                }),
            ),
            (
                "pick_peaks",
                json!({
                    "action": "pick_peaks",
                    "auto_threshold": true
                }),
            ),
            (
                "run_assignment",
                json!({
                    "action": "run_assignment",
                    "solver": "bayes-sa",
                    "engine": "gpu"
                }),
            ),
            (
                "run_assignment",
                json!({
                    "action": "run_assignment"
                }),
            ),
            (
                "select_workbook_page",
                json!({
                    "action": "select_workbook_page",
                    "page_index": 2,
                    "page_name": "HNCACB"
                }),
            ),
            (
                "set_contour_level",
                json!({
                    "action": "set_contour_level",
                    "contour_base": 4200.0,
                    "multiplier": 1.3,
                    "show_negative": true
                }),
            ),
            (
                "set_viewport_bounds",
                json!({
                    "action": "set_viewport_bounds",
                    "h_left": 9.9,
                    "h_right": 6.2,
                    "n_top": 104.0,
                    "n_bottom": 133.0
                }),
            ),
            (
                "export_results",
                json!({
                    "action": "export_results",
                    "prefix": export_str,
                    "format": "nef"
                }),
            ),
            (
                "set_lock_residue",
                json!({
                    "action": "set_lock_residue",
                    "seq_id": 14,
                    "locked": true
                }),
            ),
        ];

        for (idx, (expected_action, args)) in valid_calls.iter().enumerate() {
            let req = serde_json::to_vec(&json!({
                "jsonrpc": "2.0",
                "id": 100 + idx,
                "method": "tools/call",
                "params": {
                    "name": "run_workbench_workflow",
                    "arguments": args
                }
            }))
            .unwrap();
            let resp = ctrl.handle_jsonrpc_bytes(&req).unwrap();
            assert_eq!(resp["result"]["isError"], false, "unexpected error: {resp}");
            let payload: Value =
                serde_json::from_str(resp["result"]["content"][0]["text"].as_str().unwrap())
                    .unwrap();
            assert_eq!(payload["queued"], true);
            assert_eq!(payload["action"], *expected_action);
            assert_eq!(payload["permission"], "read_write");
        }

        let queued = ctrl.drain_pending_ui_commands();
        assert_eq!(
            queued,
            vec![
                McpUiCommand::LoadStudyFolder {
                    path: study_str,
                    run_full_pipeline: true,
                    confirm: true,
                },
                McpUiCommand::SetSequence {
                    sequence: "MQIFVKTLTG".to_string(),
                    sequence_path: None,
                    confirm: true,
                },
                McpUiCommand::SetSequence {
                    sequence: String::new(),
                    sequence_path: Some(fasta_str),
                    confirm: true,
                },
                McpUiCommand::PickAllSpectra {
                    snr_threshold: Some(7.5),
                    use_auto_threshold: false,
                    picker_mode: Some("neural".to_string()),
                },
                McpUiCommand::PickAllSpectra {
                    snr_threshold: None,
                    use_auto_threshold: true,
                    picker_mode: None,
                },
                McpUiCommand::RunAssignment {
                    solver: "bayes-sa".to_string(),
                    engine: Some("gpu".to_string()),
                },
                McpUiCommand::RunAssignment {
                    solver: "evo-graph".to_string(),
                    engine: None,
                },
                McpUiCommand::SelectWorkbookPage {
                    page_index: Some(2),
                    page_name: Some("HNCACB".to_string()),
                },
                McpUiCommand::SetContourLevel {
                    contour_base: Some(4200.0),
                    multiplier: Some(1.3),
                    show_negative: Some(true),
                },
                McpUiCommand::SetViewportBounds {
                    h_left: 9.9,
                    h_right: 6.2,
                    n_top: 104.0,
                    n_bottom: 133.0,
                },
                McpUiCommand::ExportResults {
                    prefix: export_str,
                    format: "nef".to_string(),
                    overwrite: false,
                },
                McpUiCommand::SetLockResidue {
                    seq_id: 14,
                    locked: true,
                },
            ]
        );
        let snap_after = ctrl.snapshot();
        assert!((snap_after.picker_snr - 7.5).abs() < 1e-12);
        assert_eq!(snap_after.picker_mode, "neural");
        assert_eq!(snap_after.active_page_index, 2);
        assert!((snap_after.contour_base - 4200.0).abs() < 1e-12);
        assert_eq!(snap_after.view_bounds, [9.9, 6.2, 104.0, 133.0]);

        // Undo across all 12 workflow commands + earlier AssignSpinSystemToResidue cleanly
        for _ in 0..12 {
            assert!(ctrl.undo_last().is_some());
        }
        let snap_reverted = ctrl.snapshot();
        assert!((snap_reverted.picker_snr - 5.0).abs() < 1e-12);
        assert_eq!(snap_reverted.picker_mode, "classical");
        assert_eq!(snap_reverted.solver_kind, "evo-graph");
        assert_eq!(snap_reverted.active_page_index, 0);
        assert!((snap_reverted.contour_base - 0.0).abs() < 1e-12);
        assert_eq!(snap_reverted.view_bounds, [10.5, 6.0, 102.0, 134.0]);
        assert_eq!(snap_reverted.assigned_residues[&14].cluster_id, 2);
        let reverted_assign = ctrl.undo_last().expect("undo prior assign");
        assert!(matches!(
            reverted_assign,
            McpUiCommand::AssignSpinSystemToResidue { seq_id: 14, .. }
        ));
        assert!(ctrl.snapshot().assigned_residues.is_empty());
        assert!(ctrl.redo_last().is_some());
        assert_eq!(ctrl.snapshot().assigned_residues[&14].cluster_id, 2);
    }

    #[test]
    fn test_dispatch_workbench_action_all_variants_and_repaint_wakeup() {
        // 1. Every WorkbenchAction::ALL variant round-trips in PascalCase, snake_case, and lowercase
        assert!(!WorkbenchAction::ALL.is_empty());
        for &action in WorkbenchAction::ALL {
            let pascal = workbench_action_canonical_name(action);
            let snake = workbench_action_snake_name(action);
            assert_eq!(
                parse_workbench_action(&pascal),
                Some(action),
                "PascalCase round-trip failed for {pascal}"
            );
            assert_eq!(
                parse_workbench_action(&snake),
                Some(action),
                "snake_case round-trip failed for {snake} ({pascal})"
            );
            assert_eq!(
                parse_workbench_action(&pascal.to_ascii_lowercase()),
                Some(action),
                "lowercase round-trip failed for {pascal}"
            );
        }
        assert_eq!(
            parse_workbench_action("switch_mode_assign"),
            Some(WorkbenchAction::SwitchModeAssign)
        );
        assert_eq!(
            parse_workbench_action("SuppressWaterPeak"),
            Some(WorkbenchAction::SuppressWaterPeak)
        );
        assert_eq!(
            parse_workbench_action("fold-structure-ensemble"),
            Some(WorkbenchAction::FoldStructureEnsemble)
        );
        assert_eq!(parse_workbench_action("not_a_real_action"), None);
        assert_eq!(parse_workbench_action("   "), None);

        // 2. ReadOnly blocks dispatch_workbench_action and records BlockedReadOnly audit
        let ctrl = McpBridgeController::default();
        let ctx = egui::Context::default();
        ctrl.set_repaint_context(ctx);

        let ro_req = serde_json::to_vec(&json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "dispatch_workbench_action",
                "arguments": { "action": "switch_mode_assign" }
            }
        }))
        .unwrap();
        let ro_resp = ctrl.handle_jsonrpc_bytes(&ro_req).unwrap();
        assert_eq!(ro_resp["result"]["isError"], true);
        let ro_payload: Value =
            serde_json::from_str(ro_resp["result"]["content"][0]["text"].as_str().unwrap())
                .unwrap();
        assert_eq!(ro_payload["queued"], false);
        assert_eq!(ro_payload["status"], "denied");
        assert!(ctrl.drain_pending_ui_commands().is_empty());
        {
            let guard = ctrl.shared.lock().unwrap();
            assert_eq!(
                guard.undo_log.audit_records().back().unwrap().outcome,
                McpAuditOutcome::BlockedReadOnly
            );
        }

        // 3. Invalid action returns -32602; ReadWrite queues DispatchWorkbenchAction
        ctrl.set_permission(McpBridgePermission::ReadWrite);
        let bad_resp = ctrl
            .handle_jsonrpc_bytes(
                br#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"dispatch_workbench_action","arguments":{"action":"nonexistent_xyz"}}}"#,
            )
            .unwrap();
        assert_eq!(bad_resp["error"]["code"], -32602);

        let rw_resp = ctrl.handle_jsonrpc_bytes(&ro_req).unwrap();
        assert_eq!(rw_resp["result"]["isError"], false);
        let rw_payload: Value =
            serde_json::from_str(rw_resp["result"]["content"][0]["text"].as_str().unwrap())
                .unwrap();
        assert_eq!(rw_payload["queued"], true);
        assert_eq!(rw_payload["action"], "SwitchModeAssign");
        assert_eq!(rw_payload["permission"], "read_write");
        assert_eq!(
            ctrl.drain_pending_ui_commands(),
            vec![McpUiCommand::DispatchWorkbenchAction {
                action: WorkbenchAction::SwitchModeAssign
            }]
        );
    }

    #[test]
    fn test_inspect_workbench_data_all_sections_filters_and_pagination() {
        let ctrl = McpBridgeController::default();
        assert_eq!(ctrl.permission(), McpBridgePermission::ReadOnly);

        // Seed realistic tabular workbench state
        ctrl.update_data_tables(|t| {
            t.assignments = vec![
                json!({ "seq_id": 1, "res_name": "MET", "spin_system_idx": 0, "confidence": 0.96 }),
                json!({ "seq_id": 2, "res_name": "GLN", "spin_system_idx": 1, "confidence": 0.62 }),
                json!({ "seq_id": 3, "res_name": "ILE", "spin_system_idx": null, "confidence": null }),
                json!({ "seq_id": 4, "res_name": "PHE", "spin_system_idx": -1, "confidence": 0.45 }),
            ];
            t.spin_systems = vec![
                json!({ "cluster_id": 0, "h_ppm": 8.21, "n_ppm": 120.4 }),
                json!({ "cluster_id": 1, "h_ppm": 8.45, "n_ppm": 118.9 }),
            ];
            t.peaks = vec![
                json!({ "experiment": "HSQC", "h_ppm": 8.21, "n_ppm": 120.4 }),
                json!({ "experiment": "HNCACB", "h_ppm": 8.21, "n_ppm": 120.4, "c_ppm": 55.2 }),
                json!({ "experiment": "hsqc", "h_ppm": 8.45, "n_ppm": 118.9 }),
                json!({ "experiment": "HNCO", "h_ppm": 8.45, "n_ppm": 118.9, "c_ppm": 175.1 }),
            ];
            t.workbook_pages = vec![
                json!({ "index": 0, "label": "HSQC" }),
                json!({ "index": 1, "label": "HNCACB" }),
            ];
        });

        let call_inspect = |args: Value| -> Value {
            let req = serde_json::to_vec(&json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "inspect_workbench_data",
                    "arguments": args
                }
            }))
            .unwrap();
            let resp = ctrl.handle_jsonrpc_bytes(&req).unwrap();
            assert_eq!(resp["result"]["isError"], false);
            serde_json::from_str(resp["result"]["content"][0]["text"].as_str().unwrap()).unwrap()
        };

        // 1. Actions catalog lists all WorkbenchAction::ALL with snake_name + metadata + pagination
        let actions_p1 = call_inspect(json!({ "section": "actions", "limit": 50, "offset": 0 }));
        assert_eq!(actions_p1["total_matching"], WorkbenchAction::ALL.len());
        assert_eq!(actions_p1["items"].as_array().unwrap().len(), 50);
        let first_act = &actions_p1["items"][0];
        assert!(first_act["name"].is_string());
        assert!(first_act["snake_name"].is_string());
        assert!(first_act["category"].is_string());
        assert!(first_act["title"].is_string());
        assert!(first_act["description"].is_string());
        assert!(first_act.get("chord_badge").is_some());

        // Limit clamps to 1..=500
        let actions_clamped =
            call_inspect(json!({ "section": "actions", "limit": 9999, "offset": 0 }));
        assert_eq!(actions_clamped["limit"], 500);
        assert_eq!(
            actions_clamped["items"].as_array().unwrap().len(),
            WorkbenchAction::ALL.len()
        );

        // 2. Peaks filter (case-insensitive) + pagination
        let hsqc_peaks = call_inspect(json!({
            "section": "peaks",
            "experiment": "HsQc",
            "limit": 1,
            "offset": 1
        }));
        assert_eq!(hsqc_peaks["total_matching"], 2);
        assert_eq!(hsqc_peaks["items"].as_array().unwrap().len(), 1);
        assert_eq!(hsqc_peaks["items"][0]["h_ppm"], 8.45);

        // 3. low_confidence (default 0.8 + explicit threshold)
        let low_default = call_inspect(json!({ "section": "low_confidence" }));
        assert_eq!(low_default["total_matching"], 3);
        let low_strict =
            call_inspect(json!({ "section": "low_confidence", "max_confidence": 0.5 }));
        assert_eq!(low_strict["total_matching"], 2); // seq 3 (null) + seq 4 (0.45)

        // 4. unassigned_residues (null or < 0)
        let unassigned = call_inspect(json!({ "section": "unassigned_residues" }));
        assert_eq!(unassigned["total_matching"], 2); // seq 3 (null) + seq 4 (-1)

        // 5. assignments, spin_systems, workbook_pages, qc_report
        assert_eq!(
            call_inspect(json!({ "section": "assignments" }))["total_matching"],
            4
        );
        assert_eq!(
            call_inspect(json!({ "section": "spin_systems" }))["total_matching"],
            2
        );
        assert_eq!(
            call_inspect(json!({ "section": "workbook_pages" }))["total_matching"],
            2
        );
        let qc_empty = call_inspect(json!({ "section": "qc_report" }));
        assert_eq!(qc_empty["available"], false);
        assert!(qc_empty["qc_report"].is_null());

        ctrl.update_data_tables(|t| {
            t.qc_report = Some(json!({ "verdict": "PASS", "completeness": 0.94 }));
        });
        let qc_pop = call_inspect(json!({ "section": "qc_report" }));
        assert_eq!(qc_pop["available"], true);
        assert_eq!(qc_pop["qc_report"]["verdict"], "PASS");

        // 6. Unknown section returns -32602
        let bad = ctrl
            .handle_jsonrpc_bytes(
                br#"{"jsonrpc":"2.0","id":9,"method":"tools/call","params":{"name":"inspect_workbench_data","arguments":{"section":"bogus"}}}"#,
            )
            .unwrap();
        assert_eq!(bad["error"]["code"], -32602);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_start_bridge_preferred_port_and_ephemeral_fallback() {
        // 1. Custom bind_input values are preserved verbatim without silent :0 fallback
        let busy_ephemeral = TcpListener::bind("127.0.0.1:0").expect("bind dummy ephemeral port");
        let busy_addr_str = busy_ephemeral.local_addr().unwrap().to_string();
        let mut custom_ctrl = McpBridgeController {
            bind_input: busy_addr_str.clone(),
            ..Default::default()
        };
        assert!(custom_ctrl.start_bridge_preferred().is_err());
        assert_eq!(custom_ctrl.bind_input, busy_addr_str);
        assert!(!custom_ctrl.is_running());

        // Use an unbound host-invalid or occupied ephemeral bind address instead of hardcoded 17420
        custom_ctrl.bind_input = busy_addr_str.clone();
        assert!(custom_ctrl.start_bridge_preferred().is_err());
        assert_eq!(custom_ctrl.bind_input, busy_addr_str);
        drop(busy_ephemeral);

        // 2. When the preferred ephemeral port is held, start_bridge_with_preferred_port falls back
        //    silently to another ephemeral port without ever touching hardcoded port 17420.
        let holder = TcpListener::bind("127.0.0.1:0").expect("bind held ephemeral preferred port");
        let held_port = holder.local_addr().unwrap().port();
        let mut fallback_ctrl = McpBridgeController::default();
        let fb_addr = fallback_ctrl
            .start_bridge_with_preferred_port(held_port)
            .expect("fallback to ephemeral port while held_port is occupied");
        assert!(fallback_ctrl.is_running());
        assert_ne!(
            fb_addr,
            format!("127.0.0.1:{held_port}"),
            "must use a distinct ephemeral port when preferred port is held"
        );
        assert_eq!(fallback_ctrl.bind_input, DEFAULT_LOOPBACK_BIND);
        fallback_ctrl.stop_bridge();
        drop(holder);

        // 3. Probe an open ephemeral preferred port: start_bridge_with_preferred_port binds it (or
        //    falls back cleanly if concurrently claimed), and a second controller concurrently binds
        //    a distinct ephemeral port.
        let probe = TcpListener::bind("127.0.0.1:0").expect("probe free ephemeral port");
        let free_port = probe.local_addr().unwrap().port();
        drop(probe);

        let mut primary_ctrl = McpBridgeController::default();
        let prim_addr = primary_ctrl
            .start_bridge_with_preferred_port(free_port)
            .expect("bind probed preferred port or ephemeral fallback");
        assert!(primary_ctrl.is_running());
        assert_eq!(
            primary_ctrl.endpoint_url(),
            format!("http://{prim_addr}/mcp")
        );
        assert_eq!(primary_ctrl.bind_input, DEFAULT_LOOPBACK_BIND);

        let prim_port: u16 = prim_addr
            .rsplit(':')
            .next()
            .and_then(|p| p.parse().ok())
            .expect("valid bound port");
        let mut second_ctrl = McpBridgeController::default();
        let sec_addr = second_ctrl
            .start_bridge_with_preferred_port(prim_port)
            .expect("second bridge falls back to distinct ephemeral port");
        assert_ne!(sec_addr, prim_addr);
        assert!(second_ctrl.is_running());

        second_ctrl.stop_bridge();
        primary_ctrl.stop_bridge();
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_path_sandboxing_overwrite_refusal_and_destructive_guards() {
        let sandbox = TempTestDir::new("sandbox_guards");
        let outside = TempTestDir::new("outside_guards");
        let study_dir = sandbox.path().join("study_ubq");
        std::fs::create_dir_all(&study_dir).expect("create study dir");
        let in_fasta = sandbox.path().join("seq.fasta");
        std::fs::write(&in_fasta, ">UBQ\nMQIFVKTLTG\n").expect("write in-root fasta");
        let out_fasta = outside.path().join("evil.fasta");
        std::fs::write(&out_fasta, ">EVIL\nAAAA\n").expect("write out-of-root fasta");

        let ctrl = McpBridgeController::default();
        // Root "/" must never be accepted as an approved root
        assert!(ctrl.add_approved_root("/").is_err());
        assert!(ctrl
            .add_approved_root("/nonexistent_ferric_dir_xyz")
            .is_err());
        ctrl.set_approved_roots([sandbox.path()])
            .expect("approve sandbox dir");
        assert_eq!(ctrl.approved_roots(), vec![sandbox.path().to_path_buf()]);
        ctrl.set_permission(McpBridgePermission::ReadWrite);

        let call_tool = |name: &str, args: Value| -> Value {
            let req = serde_json::to_vec(&json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": { "name": name, "arguments": args }
            }))
            .unwrap();
            ctrl.handle_jsonrpc_bytes(&req).unwrap()
        };
        let assert_denied = |resp: &Value, expected_kind: &str| {
            assert_eq!(resp["result"]["isError"], true, "expected denial: {resp}");
            let payload: Value =
                serde_json::from_str(resp["result"]["content"][0]["text"].as_str().unwrap())
                    .unwrap();
            assert_eq!(payload["status"], "denied", "payload={payload}");
            assert_eq!(payload["queued"], false, "payload={payload}");
            assert_eq!(payload["error_kind"], expected_kind, "payload={payload}");
        };

        // 1. Out-of-root / traversal / file:// / sensitive file paths are refused pre-queue
        let outside_study_resp = call_tool(
            "run_workbench_workflow",
            json!({ "action": "load_study_folder", "path": outside.path().to_string_lossy() }),
        );
        assert_denied(&outside_study_resp, "outside_allowed_roots");

        let trav_study = format!(
            "{}/../{}",
            sandbox.path().display(),
            outside.path().file_name().unwrap().to_string_lossy()
        );
        let trav_study_resp = call_tool(
            "run_workbench_workflow",
            json!({ "action": "load_study_folder", "path": trav_study }),
        );
        assert_denied(&trav_study_resp, "parent_traversal");

        let out_fasta_resp = call_tool(
            "run_workbench_workflow",
            json!({ "action": "set_sequence", "sequence_path": out_fasta.to_string_lossy() }),
        );
        assert_denied(&out_fasta_resp, "outside_allowed_roots");

        let file_url_resp = call_tool(
            "run_workbench_workflow",
            json!({ "action": "set_sequence", "sequence_path": format!("file://{}", in_fasta.display()) }),
        );
        assert_denied(&file_url_resp, "null_byte_or_invalid_path");

        let dir_as_fasta_resp = call_tool(
            "run_workbench_workflow",
            json!({ "action": "set_sequence", "sequence_path": study_dir.to_string_lossy() }),
        );
        assert_denied(&dir_as_fasta_resp, "null_byte_or_invalid_path");

        // Prefix equal to root itself becomes <root>.nef outside root -> must be refused!
        let root_escape_prefix_resp = call_tool(
            "run_workbench_workflow",
            json!({
                "action": "export_results",
                "prefix": sandbox.path().to_string_lossy(),
                "format": "nef"
            }),
        );
        assert_denied(&root_escape_prefix_resp, "outside_allowed_roots");

        // Symlink inside approved root pointing outside must be refused
        #[cfg(unix)]
        {
            let link_path = sandbox.path().join("escape_link.fasta");
            std::os::unix::fs::symlink(&out_fasta, &link_path).expect("create symlink");
            let sym_resp = call_tool(
                "run_workbench_workflow",
                json!({
                    "action": "set_sequence",
                    "sequence_path": link_path.to_string_lossy()
                }),
            );
            assert_eq!(sym_resp["result"]["isError"], true);
            let link_dir = sandbox.path().join("escape_dir");
            std::os::unix::fs::symlink(outside.path(), &link_dir).expect("create dir symlink");
            let sym_dir_resp = call_tool(
                "run_workbench_workflow",
                json!({
                    "action": "load_study_folder",
                    "path": link_dir.to_string_lossy()
                }),
            );
            assert_eq!(sym_dir_resp["result"]["isError"], true);
        }

        // In-root reads: set_sequence on an empty workspace succeeds without destructive confirm,
        // while load_study_folder is always destructive and requires allow_destructive_actions + confirm: true.
        let study_no_perm = call_tool(
            "run_workbench_workflow",
            json!({ "action": "load_study_folder", "path": study_dir.to_string_lossy(), "confirm": true }),
        );
        assert_denied(&study_no_perm, "destructive_action_denied");
        let ok_seq = call_tool(
            "run_workbench_workflow",
            json!({ "action": "set_sequence", "sequence_path": in_fasta.to_string_lossy() }),
        );
        assert_eq!(ok_seq["result"]["isError"], false);
        let _ = ctrl.drain_pending_ui_commands();

        // 2. Overwrite refusal vs overwrite: true
        let existing_prefix = sandbox.path().join("existing_export");
        let existing_nef = existing_prefix.with_extension("nef");
        std::fs::write(&existing_nef, "existing-nef").expect("seed existing nef");

        let ow_omitted = call_tool(
            "run_workbench_workflow",
            json!({
                "action": "export_results",
                "prefix": existing_prefix.to_string_lossy(),
                "format": "nef"
            }),
        );
        assert_denied(&ow_omitted, "overwrite_refused");

        let ow_false = call_tool(
            "run_workbench_workflow",
            json!({
                "action": "export_results",
                "prefix": existing_prefix.to_string_lossy(),
                "format": "nef",
                "overwrite": false
            }),
        );
        assert_denied(&ow_false, "overwrite_refused");

        let ow_true = call_tool(
            "run_workbench_workflow",
            json!({
                "action": "export_results",
                "prefix": existing_prefix.to_string_lossy(),
                "format": "nef",
                "overwrite": true
            }),
        );
        assert_eq!(ow_true["result"]["isError"], false);
        assert_eq!(
            ctrl.drain_pending_ui_commands(),
            vec![McpUiCommand::ExportResults {
                prefix: existing_prefix.to_string_lossy().into_owned(),
                format: "nef".to_string(),
                overwrite: true,
            }]
        );

        // Guarded dispatch actions: SaveSessionNef, ExportNmrStar, LoadSessionNef
        ctrl.set_current_export_prefix(&outside.path().join("bad_prefix").to_string_lossy());
        let bad_save = call_tool(
            "dispatch_workbench_action",
            json!({ "action": "save_session_nef", "overwrite": true }),
        );
        assert_denied(&bad_save, "outside_allowed_roots");

        ctrl.set_current_export_prefix(&existing_prefix.to_string_lossy());
        let ow_save_denied = call_tool(
            "dispatch_workbench_action",
            json!({ "action": "save_session_nef" }),
        );
        assert_denied(&ow_save_denied, "overwrite_refused");

        let ow_save_ok = call_tool(
            "dispatch_workbench_action",
            json!({ "action": "save_session_nef", "overwrite": true }),
        );
        assert_eq!(ow_save_ok["result"]["isError"], false);
        assert_eq!(
            ctrl.drain_pending_ui_commands(),
            vec![McpUiCommand::DispatchGuardedWorkbenchAction {
                action: WorkbenchAction::SaveSessionNef,
                overwrite: true,
                confirm: false,
            }]
        );

        // 3. Destructive & interactive file-picker action guards
        let picker_denied = call_tool(
            "dispatch_workbench_action",
            json!({ "action": "load_path_or_spectrum" }),
        );
        assert_denied(&picker_denied, "sandbox_required");

        assert!(!ctrl.destructive_actions_allowed());
        let reset_disabled = call_tool(
            "dispatch_workbench_action",
            json!({ "action": "reset_workspace", "confirm": true }),
        );
        assert_denied(&reset_disabled, "destructive_action_denied");

        ctrl.set_destructive_actions_allowed(true);
        let reset_unconfirmed = call_tool(
            "dispatch_workbench_action",
            json!({ "action": "reset_workspace" }),
        );
        assert_denied(&reset_unconfirmed, "confirmation_required");

        let reset_ok = call_tool(
            "dispatch_workbench_action",
            json!({ "action": "reset_workspace", "confirm": true }),
        );
        assert_eq!(reset_ok["result"]["isError"], false);
        assert_eq!(
            ctrl.drain_pending_ui_commands(),
            vec![McpUiCommand::DispatchGuardedWorkbenchAction {
                action: WorkbenchAction::ResetWorkspace,
                overwrite: false,
                confirm: true,
            }]
        );
        assert!(ctrl
            .shared
            .lock()
            .unwrap()
            .undo_log
            .audit_records()
            .iter()
            .any(|r| r.outcome == McpAuditOutcome::BlockedSecurity));
    }

    #[test]
    fn test_masked_json_preview_never_exposes_raw_bearer_token() {
        let mut ctrl = McpBridgeController::default();
        assert!(!ctrl.reveal_token);
        let raw_token = ctrl.bearer_token();
        assert!(!raw_token.is_empty());

        let masked_preview = ctrl.build_client_config_json_preview();
        assert!(
            !masked_preview.contains(&raw_token),
            "masked JSON preview must never expose the raw bearer token"
        );
        assert!(
            masked_preview.contains(&mask_bearer_token(&raw_token)),
            "masked JSON preview must contain masked token: {masked_preview}"
        );
        assert!(
            masked_preview.contains("••••••••••••"),
            "masked JSON preview must contain masked placeholder: {masked_preview}"
        );

        ctrl.reveal_token = true;
        let revealed_preview = ctrl.build_client_config_json_preview();
        assert!(
            revealed_preview.contains(&raw_token),
            "revealed JSON preview should expose raw bearer token when reveal_token=true"
        );

        ctrl.reveal_token = false;
        assert!(!ctrl.build_client_config_json_preview().contains(&raw_token));

        // Headless egui render check: panel with Show JSON open renders cleanly when masked
        ctrl.show_config_json = true;
        let ctx = egui::Context::default();
        let raw_in = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(640.0, 900.0),
            )),
            ..Default::default()
        };
        let mut out = ctx.run_ui(raw_in, |ui| {
            let _ = render_mcp_bridge_panel(ui, &mut ctrl);
        });
        out.textures_delta.clear();
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_http_prebody_host_origin_accept_batch_and_slow_loris_deadline() {
        fn deterministic_csprng(buf: &mut [u8; 32]) -> bool {
            for (i, b) in buf.iter_mut().enumerate() {
                *b = (i as u8).wrapping_mul(17).wrapping_add(5);
            }
            true
        }

        let mut ctrl = McpBridgeController::default();
        let addr = ctrl
            .start_bridge_with_options(
                "127.0.0.1:0",
                deterministic_csprng,
                Duration::from_millis(250),
            )
            .expect("start loopback bridge with 250ms deadline");
        let bearer = ctrl.bearer_token();

        let send_http_raw = |raw_req: &str| -> String {
            let mut sock = TcpStream::connect(&addr).expect("connect loopback");
            sock.set_read_timeout(Some(Duration::from_secs(3))).unwrap();
            sock.write_all(raw_req.as_bytes()).expect("write request");
            let _ = sock.flush();
            let _ = sock.shutdown(std::net::Shutdown::Write);
            let mut resp = String::new();
            std::io::Read::read_to_string(&mut sock, &mut resp).expect("read response");
            resp
        };

        // 1. Pre-body Host/Origin checks: write ONLY headers with large Content-Length and NO body,
        //    keep write side OPEN! Server must reject immediately with 403 without waiting for body/deadline.
        for bad_hdr in [
            format!(
                "POST /mcp HTTP/1.1\r\nHost: evil.example\r\nAuthorization: Bearer {bearer}\r\nContent-Length: 4000000\r\n\r\n"
            ),
            format!(
                "POST /mcp HTTP/1.1\r\nHost: {addr}\r\nOrigin: https://evil.example\r\nAuthorization: Bearer {bearer}\r\nContent-Length: 4000000\r\n\r\n"
            ),
        ] {
            let t0 = Instant::now();
            let mut sock = TcpStream::connect(&addr).expect("connect pre-body check");
            sock.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
            sock.write_all(bad_hdr.as_bytes()).expect("send headers only");
            let _ = sock.flush();
            let mut status_line = String::new();
            let mut rdr = BufReader::new(sock);
            let _ = rdr.read_line(&mut status_line);
            assert!(
                status_line.contains(" 403 "),
                "expected immediate 403 before reading body, got {status_line:?}"
            );
            assert!(
                t0.elapsed() < Duration::from_millis(200),
                "403 rejection should not wait for body or slow-loris deadline"
            );
        }

        // 2. Accept header handling (checked before body, plus JSON vs SSE framing)
        let r_406 = send_http_raw(&format!(
            "POST /mcp HTTP/1.1\r\nHost: {addr}\r\nAuthorization: Bearer {bearer}\r\nAccept: text/plain\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{{}}"
        ));
        assert!(r_406.contains("406 Not Acceptable"), "got {r_406:?}");

        let r_406_q0 = send_http_raw(&format!(
            "POST /mcp HTTP/1.1\r\nHost: {addr}\r\nAuthorization: Bearer {bearer}\r\nAccept: application/json;q=0\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{{}}"
        ));
        assert!(r_406_q0.contains("406 Not Acceptable"), "got {r_406_q0:?}");

        let ping_body = r#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#;
        let r_sse = send_http_raw(&format!(
            "POST /mcp HTTP/1.1\r\nHost: {addr}\r\nAuthorization: Bearer {bearer}\r\nAccept: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{ping_body}",
            ping_body.len()
        ));
        assert!(r_sse.contains("200 OK"), "got {r_sse:?}");
        assert!(
            r_sse.contains("Content-Type: text/event-stream"),
            "got {r_sse:?}"
        );
        assert!(r_sse.contains("event: message\ndata: "), "got {r_sse:?}");

        // 3. JSON-RPC batch arrays (mixed, empty [], notification-only -> 202)
        let mixed_batch = r#"[{"jsonrpc":"2.0","id":10,"method":"ping"},{"jsonrpc":"2.0","method":"notifications/initialized"},42]"#;
        let r_batch = send_http_raw(&format!(
            "POST /mcp HTTP/1.1\r\nHost: {addr}\r\nAuthorization: Bearer {bearer}\r\nAccept: application/json, text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{mixed_batch}",
            mixed_batch.len()
        ));
        assert!(r_batch.contains("200 OK"), "got {r_batch:?}");
        assert!(
            r_batch.contains("Content-Type: application/json"),
            "got {r_batch:?}"
        );
        let body_part = r_batch.split("\r\n\r\n").nth(1).unwrap();
        let batch_json: Value = serde_json::from_str(body_part).expect("batch json array");
        let arr = batch_json.as_array().expect("batch array response");
        assert_eq!(
            arr.len(),
            2,
            "expected pong + invalid-item error: {batch_json}"
        );
        assert_eq!(arr[0]["id"], 10);
        assert_eq!(arr[1]["error"]["code"], -32600);

        let notif_batch = r#"[{"jsonrpc":"2.0","method":"notifications/initialized"}]"#;
        let r_notif = send_http_raw(&format!(
            "POST /mcp HTTP/1.1\r\nHost: {addr}\r\nAuthorization: Bearer {bearer}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{notif_batch}",
            notif_batch.len()
        ));
        assert!(r_notif.contains("202 Accepted"), "got {r_notif:?}");

        let empty_batch = r#"[]"#;
        let r_empty = send_http_raw(&format!(
            "POST /mcp HTTP/1.1\r\nHost: {addr}\r\nAuthorization: Bearer {bearer}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{empty_batch}",
            empty_batch.len()
        ));
        let empty_json: Value =
            serde_json::from_str(r_empty.split("\r\n\r\n").nth(1).unwrap()).unwrap();
        assert_eq!(empty_json["error"]["code"], -32600);

        // 4. Slow-loris deadline + concurrent unblocked healthy request
        let mut slow_sock = TcpStream::connect(&addr).expect("connect slow-loris header socket");
        slow_sock
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        slow_sock
            .write_all(b"POST /mcp HTTP/1.1\r\nHo")
            .expect("send partial header");
        let _ = slow_sock.flush();

        // While slow_sock stalls, a legitimate ping must succeed promptly via another worker thread
        let t_fast = Instant::now();
        let fast_resp = send_http_raw(&format!(
            "POST /mcp HTTP/1.1\r\nHost: {addr}\r\nAuthorization: Bearer {bearer}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{ping_body}",
            ping_body.len()
        ));
        assert!(
            fast_resp.contains("200 OK"),
            "fast resp failed: {fast_resp:?}"
        );
        assert!(
            t_fast.elapsed() < Duration::from_secs(1),
            "slow-loris client must not head-of-line block healthy client"
        );

        let mut slow_resp = String::new();
        std::io::Read::read_to_string(&mut slow_sock, &mut slow_resp).expect("read 408");
        assert!(
            slow_resp.contains("408 Request Timeout"),
            "expected 408 Request Timeout for stalled header, got {slow_resp:?}"
        );

        // Slow body drip even with periodic 1-byte writes exceeds total request deadline -> 408
        let mut slow_body_sock = TcpStream::connect(&addr).expect("connect slow body socket");
        slow_body_sock
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        let hdr = format!(
            "POST /mcp HTTP/1.1\r\nHost: {addr}\r\nAuthorization: Bearer {bearer}\r\nContent-Length: 100\r\n\r\n{{"
        );
        slow_body_sock.write_all(hdr.as_bytes()).unwrap();
        let _ = slow_body_sock.flush();
        std::thread::sleep(Duration::from_millis(120));
        let _ = slow_body_sock.write_all(b"\"a\":");
        let _ = slow_body_sock.flush();
        let mut slow_body_resp = String::new();
        std::io::Read::read_to_string(&mut slow_body_sock, &mut slow_body_resp)
            .expect("read body 408");
        assert!(
            slow_body_resp.contains("408 Request Timeout"),
            "expected 408 Request Timeout for slow-loris body, got {slow_body_resp:?}"
        );

        ctrl.stop_bridge();
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_p1_9_fail_closed_default_roots_and_home_ancestor_rejection() {
        // 1. Fresh controller with no configured roots and no active study allows NOTHING (fail-closed)
        let ctrl = McpBridgeController::default();
        assert!(
            ctrl.approved_roots().is_empty(),
            "default controller must have empty approved_roots (fail-closed)"
        );
        assert!(!ctrl.cwd_fallback_allowed());
        ctrl.set_permission(McpBridgePermission::ReadWrite);
        ctrl.set_destructive_actions_allowed(true);

        // Simulate sensitive file like ~/.config/gh/hosts.yml inside a temp directory
        let fake_home = TempTestDir::new("p1_9_home");
        let gh_dir = fake_home.path().join(".config").join("gh");
        std::fs::create_dir_all(&gh_dir).expect("create fake .config/gh");
        let hosts_yml = gh_dir.join("hosts.yml");
        std::fs::write(&hosts_yml, "github.com:\n  oauth_token: ghp_MQIFVKTLTG\n")
            .expect("write fake hosts.yml");
        let study_sub = fake_home.path().join("studies").join("ubq");
        std::fs::create_dir_all(&study_sub).expect("create study subdir");

        let call_tool = |name: &str, args: Value| -> Value {
            let req = serde_json::to_vec(&json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": { "name": name, "arguments": args }
            }))
            .unwrap();
            ctrl.handle_jsonrpc_bytes(&req).unwrap()
        };
        let assert_no_roots = |resp: &Value| {
            assert_eq!(resp["result"]["isError"], true, "expected denial: {resp}");
            let payload: Value =
                serde_json::from_str(resp["result"]["content"][0]["text"].as_str().unwrap())
                    .unwrap();
            assert_eq!(payload["status"], "denied", "payload={payload}");
            assert_eq!(payload["queued"], false, "payload={payload}");
            assert_eq!(
                payload["error_kind"], "no_allowed_roots",
                "payload={payload}"
            );
        };

        // All file-touching MCP tools must be denied fail-closed with no_allowed_roots
        assert_no_roots(&call_tool(
            "run_workbench_workflow",
            json!({
                "action": "set_sequence",
                "sequence_path": hosts_yml.to_string_lossy(),
                "confirm": true
            }),
        ));
        assert_no_roots(&call_tool(
            "run_workbench_workflow",
            json!({
                "action": "export_results",
                "prefix": fake_home.path().join("leak").to_string_lossy(),
                "format": "star",
                "overwrite": true
            }),
        ));
        assert_no_roots(&call_tool(
            "run_workbench_workflow",
            json!({
                "action": "load_study_folder",
                "path": study_sub.to_string_lossy(),
                "confirm": true
            }),
        ));
        ctrl.set_current_export_prefix(&fake_home.path().join("sess").to_string_lossy());
        assert_no_roots(&call_tool(
            "dispatch_workbench_action",
            json!({ "action": "save_session_nef", "overwrite": true }),
        ));
        assert_no_roots(&call_tool(
            "dispatch_workbench_action",
            json!({ "action": "load_session_nef", "confirm": true }),
        ));
        assert!(ctrl.drain_pending_ui_commands().is_empty());

        // 2. $HOME and any ancestor of $HOME are rejected as automatic/default roots,
        //    while a dedicated subdirectory under $HOME is accepted.
        let homes = vec![fake_home.path().to_path_buf()];
        assert!(
            canonicalize_automatic_root_with_homes(fake_home.path(), &homes).is_err(),
            "$HOME must be rejected as an automatic root"
        );
        if let Some(parent_of_home) = fake_home.path().parent() {
            assert!(
                canonicalize_automatic_root_with_homes(parent_of_home, &homes).is_err(),
                "ancestor of $HOME ({}) must be rejected as an automatic root",
                parent_of_home.display()
            );
        }
        assert!(canonicalize_automatic_root_with_homes(Path::new("/"), &homes).is_err());
        assert_eq!(
            canonicalize_automatic_root_with_homes(&study_sub, &homes).unwrap(),
            study_sub
        );

        // Real process HOME / USERPROFILE must never become an automatic study root via sync_user_study_root
        for home_dir in detected_home_dirs() {
            if home_dir.is_dir() {
                ctrl.sync_user_study_root(Some(&home_dir));
                assert!(
                    ctrl.approved_roots().is_empty(),
                    "sync_user_study_root($HOME) must NOT expose {} as an automatic root",
                    home_dir.display()
                );
                if let Some(parent) = home_dir.parent() {
                    ctrl.sync_user_study_root(Some(parent));
                    assert!(
                        ctrl.approved_roots().is_empty(),
                        "sync_user_study_root(ancestor of $HOME) must NOT expose {}",
                        parent.display()
                    );
                }
            }
        }

        // Syncing a non-$HOME study directory anchors only that study; clearing it returns to fail-closed empty
        ctrl.sync_user_study_root(Some(&study_sub));
        assert_eq!(ctrl.approved_roots(), vec![study_sub.clone()]);
        ctrl.sync_user_study_root(None);
        assert!(ctrl.approved_roots().is_empty());

        // Explicit opt-in CWD fallback can be toggled (still excludes $HOME and ancestors)
        ctrl.set_cwd_fallback_allowed(true);
        assert!(ctrl.cwd_fallback_allowed());
        if let Ok(cwd) = std::env::current_dir().and_then(std::fs::canonicalize) {
            if is_home_or_ancestor_of_home(&cwd) {
                assert!(ctrl.approved_roots().is_empty());
            } else {
                assert_eq!(ctrl.approved_roots(), vec![cwd]);
            }
        }
        ctrl.reset_approved_roots_to_default();
        assert!(!ctrl.cwd_fallback_allowed());
        assert!(ctrl.approved_roots().is_empty());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_p1_10_destructive_metadata_and_complete_gating() {
        let sandbox = TempTestDir::new("p1_10_destructive");
        let study_dir = sandbox.path().join("study");
        std::fs::create_dir_all(&study_dir).expect("create study dir");
        let sess_prefix = sandbox.path().join("session_save");
        std::fs::write(sess_prefix.with_extension("nef"), "data_nef_demo\n")
            .expect("write dummy nef");

        let ctrl = McpBridgeController::default();
        ctrl.set_approved_roots([sandbox.path()])
            .expect("approve sandbox");
        ctrl.set_current_export_prefix(&sess_prefix.to_string_lossy());
        ctrl.set_permission(McpBridgePermission::ReadWrite);

        let call_tool = |name: &str, args: Value| -> Value {
            let req = serde_json::to_vec(&json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": { "name": name, "arguments": args }
            }))
            .unwrap();
            ctrl.handle_jsonrpc_bytes(&req).unwrap()
        };
        let assert_denied = |resp: &Value, expected_kind: &str| {
            assert_eq!(resp["result"]["isError"], true, "expected denial: {resp}");
            let payload: Value =
                serde_json::from_str(resp["result"]["content"][0]["text"].as_str().unwrap())
                    .unwrap();
            assert_eq!(payload["status"], "denied", "payload={payload}");
            assert_eq!(payload["queued"], false, "payload={payload}");
            assert_eq!(payload["error_kind"], expected_kind, "payload={payload}");
        };

        // 1. inspect_workbench_data(section="actions") carries explicit destructive metadata
        let insp = call_tool(
            "inspect_workbench_data",
            json!({ "section": "actions", "limit": 500 }),
        );
        assert_eq!(insp["result"]["isError"], false);
        let insp_json: Value =
            serde_json::from_str(insp["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
        let items = insp_json["items"].as_array().unwrap();
        let find_act = |name: &str| -> &Value {
            items
                .iter()
                .find(|it| it["name"].as_str() == Some(name))
                .unwrap_or_else(|| panic!("missing action {name}"))
        };
        for d in [
            "ResetWorkspace",
            "LoadUbiquitinDemo",
            "LoadSessionNef",
            "LoadPathOrSpectrum",
            "ReSimulateSyntheticNoise",
        ] {
            assert_eq!(find_act(d)["destructive"], true, "{d} must be destructive");
        }
        assert_eq!(find_act("LoadPathOrSpectrum")["blocked_via_mcp"], true);
        for safe in ["SwitchModeAssign", "ZoomIn", "TogglePeakLabels"] {
            assert_eq!(
                find_act(safe)["destructive"],
                false,
                "{safe} must not be destructive"
            );
        }
        let wf_acts = insp_json["workflow_actions"].as_array().unwrap();
        let find_wf = |name: &str| -> &Value {
            wf_acts
                .iter()
                .find(|it| it["name"].as_str() == Some(name))
                .unwrap_or_else(|| panic!("missing workflow {name}"))
        };
        assert_eq!(find_wf("load_study_folder")["destructive"], true);
        assert_eq!(find_wf("load_study_folder")["destructive_mode"], "always");
        assert_eq!(find_wf("set_sequence")["destructive"], true);
        assert_eq!(
            find_wf("set_sequence")["destructive_mode"],
            "when_replacing_state"
        );
        assert_eq!(find_wf("pick_peaks")["destructive"], false);

        // 2. Every destructive WorkbenchAction requires allow_destructive_actions AND confirm: true
        assert_denied(
            &call_tool(
                "dispatch_workbench_action",
                json!({ "action": "load_path_or_spectrum", "confirm": true }),
            ),
            "sandbox_required",
        );
        for act_name in [
            "reset_workspace",
            "load_ubiquitin_demo",
            "load_session_nef",
            "re_simulate_synthetic_noise",
        ] {
            ctrl.set_destructive_actions_allowed(false);
            assert_denied(
                &call_tool(
                    "dispatch_workbench_action",
                    json!({ "action": act_name, "confirm": true }),
                ),
                "destructive_action_denied",
            );
            ctrl.set_destructive_actions_allowed(true);
            assert_denied(
                &call_tool("dispatch_workbench_action", json!({ "action": act_name })),
                "confirmation_required",
            );
            assert_denied(
                &call_tool(
                    "dispatch_workbench_action",
                    json!({ "action": act_name, "confirm": false }),
                ),
                "confirmation_required",
            );
            let ok = call_tool(
                "dispatch_workbench_action",
                json!({ "action": act_name, "confirm": true }),
            );
            assert_eq!(ok["result"]["isError"], false, "{act_name} failed: {ok}");
        }
        let _ = ctrl.drain_pending_ui_commands();

        // 3. load_study_folder requires allow_destructive_actions AND confirm: true
        ctrl.set_destructive_actions_allowed(false);
        assert_denied(
            &call_tool(
                "run_workbench_workflow",
                json!({ "action": "load_study_folder", "path": study_dir.to_string_lossy(), "confirm": true }),
            ),
            "destructive_action_denied",
        );
        ctrl.set_destructive_actions_allowed(true);
        assert_denied(
            &call_tool(
                "run_workbench_workflow",
                json!({ "action": "load_study_folder", "path": study_dir.to_string_lossy() }),
            ),
            "confirmation_required",
        );
        let ok_study = call_tool(
            "run_workbench_workflow",
            json!({ "action": "load_study_folder", "path": study_dir.to_string_lossy(), "confirm": true }),
        );
        assert_eq!(ok_study["result"]["isError"], false);
        let _ = ctrl.drain_pending_ui_commands();

        // 4. set_sequence is allowed on empty workspace, but gated when replacing existing state
        ctrl.set_destructive_actions_allowed(false);
        let ok_empty_seq = call_tool(
            "run_workbench_workflow",
            json!({ "action": "set_sequence", "sequence": "MQIFVK" }),
        );
        assert_eq!(ok_empty_seq["result"]["isError"], false);
        // Second set_sequence while previous is still pending (or after snapshot sync) must be gated!
        assert_denied(
            &call_tool(
                "run_workbench_workflow",
                json!({ "action": "set_sequence", "sequence": "AAAA", "confirm": true }),
            ),
            "destructive_action_denied",
        );
        let _ = ctrl.drain_pending_ui_commands();
        ctrl.update_snapshot(|s| s.sequence_len = 6);
        assert_denied(
            &call_tool(
                "run_workbench_workflow",
                json!({ "action": "set_sequence", "sequence": "AAAA", "confirm": true }),
            ),
            "destructive_action_denied",
        );
        ctrl.set_destructive_actions_allowed(true);
        assert_denied(
            &call_tool(
                "run_workbench_workflow",
                json!({ "action": "set_sequence", "sequence": "AAAA" }),
            ),
            "confirmation_required",
        );
        let ok_replace_seq = call_tool(
            "run_workbench_workflow",
            json!({ "action": "set_sequence", "sequence": "AAAA", "confirm": true }),
        );
        assert_eq!(ok_replace_seq["result"]["isError"], false);
    }

    #[cfg(unix)]
    #[test]
    fn test_load_study_folder_rejects_nested_and_dangling_symlinks() {
        let sandbox = TempTestDir::new("study_nested_symlink");
        let outside = TempTestDir::new("study_outside_secret");
        let secret = outside.path().join("secret.fasta");
        std::fs::write(&secret, ">SECRET\nMQIFVK\n").expect("write outside secret");

        let study = sandbox.path().join("my_study");
        let sub = study.join("exp1");
        std::fs::create_dir_all(&sub).expect("create study subdir");
        let real_seq = study.join("real.fasta");
        std::fs::write(&real_seq, ">REAL\nMQIFVK\n").expect("write real fasta");

        let ctrl = McpBridgeController::default();
        ctrl.set_approved_roots([sandbox.path()])
            .expect("approve sandbox");
        ctrl.set_permission(McpBridgePermission::ReadWrite);
        ctrl.set_destructive_actions_allowed(true);

        let call_load = |p: &Path| -> Value {
            let req = serde_json::to_vec(&json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "run_workbench_workflow",
                    "arguments": {
                        "action": "load_study_folder",
                        "path": p.to_string_lossy(),
                        "confirm": true
                    }
                }
            }))
            .unwrap();
            ctrl.handle_jsonrpc_bytes(&req).unwrap()
        };

        // Internal in-root symlink is allowed
        let internal_link = sub.join("seq_alias.fasta");
        std::os::unix::fs::symlink(&real_seq, &internal_link).expect("create internal symlink");
        let ok_resp = call_load(&study);
        assert_eq!(
            ok_resp["result"]["isError"], false,
            "in-root symlink should be allowed: {ok_resp}"
        );

        // Escaping file symlink inside nested subdir is refused
        let escape_file_link = sub.join("leak.fasta");
        std::os::unix::fs::symlink(&secret, &escape_file_link).expect("create escaping file link");
        let esc_resp = call_load(&study);
        assert_eq!(esc_resp["result"]["isError"], true);
        let esc_payload: Value =
            serde_json::from_str(esc_resp["result"]["content"][0]["text"].as_str().unwrap())
                .unwrap();
        assert_eq!(esc_payload["error_kind"], "symlink_escape");
        std::fs::remove_file(&escape_file_link).unwrap();

        // Escaping directory symlink inside nested subdir is refused
        let escape_dir_link = sub.join("ext_dir");
        std::os::unix::fs::symlink(outside.path(), &escape_dir_link)
            .expect("create escaping dir link");
        let esc_dir_resp = call_load(&study);
        assert_eq!(esc_dir_resp["result"]["isError"], true);
        std::fs::remove_file(&escape_dir_link).unwrap();

        // Dangling symlink inside study folder is refused
        let dangling_link = sub.join("broken.ucsf");
        std::os::unix::fs::symlink(
            outside.path().join("nonexistent_target_xyz"),
            &dangling_link,
        )
        .expect("create dangling symlink");
        let dang_resp = call_load(&study);
        assert_eq!(dang_resp["result"]["isError"], true);
        let dang_payload: Value =
            serde_json::from_str(dang_resp["result"]["content"][0]["text"].as_str().unwrap())
                .unwrap();
        assert_eq!(dang_payload["error_kind"], "symlink_escape");
    }

    #[test]
    fn test_validate_loopback_host_rejects_ipv6_loopback_matching_manual() {
        assert!(validate_loopback_host_and_origin("127.0.0.1:17420", false, None, 17420).is_ok());
        assert!(validate_loopback_host_and_origin("localhost:17420", false, None, 17420).is_ok());
        assert!(validate_loopback_host_and_origin("[::1]:17420", false, None, 17420).is_err());
        assert!(validate_loopback_host_and_origin("[::1]", false, None, 17420).is_err());
    }
}
