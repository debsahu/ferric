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

use std::collections::hash_map::RandomState;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::hash::{BuildHasher, Hasher};
#[cfg(not(target_arch = "wasm32"))]
use std::io::{BufReader, Read, Write};
#[cfg(not(target_arch = "wasm32"))]
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4, TcpListener, TcpStream};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::atomic::AtomicBool;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
#[cfg(not(target_arch = "wasm32"))]
use std::thread::{self, JoinHandle};
#[cfg(not(target_arch = "wasm32"))]
use std::time::Duration;

use eframe::egui;
use serde_json::{json, Value};

use crate::contour_painter::SyncCrosshair;
use crate::strip_tiler::{StripMatchCandidate, StripMatchDirection};

/// Negotiated Model Context Protocol specification revision.
pub const MCP_PROTOCOL_VERSION: &str = "2024-11-05";

/// Bounded maximum depth (`256` reversible steps) for [`McpUndoLog`].
pub const MAX_MCP_UNDO_DEPTH: usize = 256;

/// Default loopback bind address (`127.0.0.1:0`, OS-assigned ephemeral port).
pub const DEFAULT_LOOPBACK_BIND: &str = "127.0.0.1:0";

/// PNG 8-byte file signature (`\x89PNG\r\n\x1a\n`, RFC 2083 §3.1).
pub const PNG_SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];

#[cfg(not(target_arch = "wasm32"))]
const MAX_MCP_HTTP_BODY_BYTES: usize = 4 * 1024 * 1024; // 4 MiB

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
        })
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
}

impl McpAuditOutcome {
    /// Short badge label for the `egui` audit table.
    pub fn badge(self) -> &'static str {
        match self {
            Self::Applied => "APPLIED",
            Self::Undone => "UNDONE",
            Self::Redone => "REDONE",
            Self::BlockedReadOnly => "BLOCKED",
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
        if f.read_exact(raw).is_ok() {
            return true;
        }
    }
    false
}

#[cfg(not(unix))]
fn fill_os_entropy(_raw: &mut [u8; 32]) -> bool {
    false
}

/// Generate a fresh 256-bit (`nmr_mcp_<64-hex>`) loopback bearer token from OS entropy
/// (`/dev/urandom` on Unix, with multi-round `RandomState` SipHash mixer fallback on `wasm32`).
pub fn generate_bearer_token() -> String {
    let mut raw = [0u8; 32];
    if !fill_os_entropy(&mut raw) {
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
    }
    let mut out = String::with_capacity(8 + 64);
    out.push_str("nmr_mcp_");
    for b in raw {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0F) as usize] as char);
    }
    out
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
}

/// Build ready-to-paste client MCP configuration JSON for Antigravity, Claude Desktop, or Cursor.
pub fn build_mcp_client_config_json(
    client: McpClientTarget,
    endpoint_url: &str,
    bearer_token: &str,
) -> String {
    let auth_value = format!("Bearer {}", bearer_token.trim());
    let server_entry = match client {
        McpClientTarget::Antigravity => json!({
            "serverUrl": endpoint_url,
            "headers": {
                "Authorization": auth_value,
            }
        }),
        McpClientTarget::ClaudeDesktop => json!({
            "transport": "http",
            "url": endpoint_url,
            "headers": {
                "Authorization": auth_value,
            }
        }),
        McpClientTarget::Cursor => json!({
            "url": endpoint_url,
            "headers": {
                "Authorization": auth_value,
            }
        }),
    };
    let doc = json!({
        "mcpServers": {
            "nmrassign-workbench": server_entry,
        }
    });
    serde_json::to_string_pretty(&doc).unwrap_or_else(|_| "{}".to_string())
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

#[derive(Debug)]
struct SharedBridgeInner {
    snapshot: WorkbenchSnapshot,
    undo_log: McpUndoLog,
    viewport: ViewportFrameBuffer,
    permission: McpBridgePermission,
    bearer_token: String,
    handled_requests: u64,
}

impl SharedBridgeInner {
    fn new(snapshot: WorkbenchSnapshot) -> Self {
        let xh = snapshot.crosshair_ppm;
        Self {
            snapshot,
            undo_log: McpUndoLog::new(),
            viewport: ViewportFrameBuffer::diagnostic_card(64, 48, xh),
            permission: McpBridgePermission::ReadOnly,
            bearer_token: generate_bearer_token(),
            handled_requests: 0,
        }
    }
}

fn mcp_tool_schemas_json() -> Value {
    json!([
        {
            "name": "get_workbench_state",
            "description": "Inspect live nmr-gui workbench state (active_residue_seq, crosshair_ppm, active_experiment, strip_candidates, selected_pdb_chain_res, undo_depth, redo_depth, qc_status, assigned_residues, excluded_peaks, highlighted_residues_3d).",
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

/// Execute a single JSON-RPC 2.0 MCP message against `inner`, returning `None` for one-way
/// JSON-RPC notifications (`notifications/initialized`) and `Some(response_json)` for requests.
fn dispatch_mcp_jsonrpc(inner: &mut SharedBridgeInner, raw_body: &[u8]) -> Option<Value> {
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
    let Some(obj) = parsed.as_object() else {
        return Some(jsonrpc_err(
            Value::Null,
            -32600,
            "Invalid Request: JSON-RPC payload must be an object",
        ));
    };
    let id_opt = obj.get("id").cloned();
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
        "initialize" => Some(jsonrpc_ok(
            id,
            json!({
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {
                    "tools": { "listChanged": false }
                },
                "serverInfo": {
                    "name": "nmrassign-gui-mcp-bridge",
                    "version": nmr_core::VERSION,
                }
            }),
        )),
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
                            Value::String(inner.permission.as_str().to_string()),
                        );
                    }
                    Some(jsonrpc_ok(id, mcp_text_result(snap, false)))
                }
                "navigate_workbench" => {
                    let cmd = match parse_navigation_command(&args) {
                        Ok(c) => c,
                        Err(msg) => return Some(jsonrpc_err(id, -32602, msg)),
                    };
                    if inner.permission == McpBridgePermission::ReadOnly {
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
                    if inner.permission == McpBridgePermission::ReadOnly {
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
                    if inner.permission == McpBridgePermission::ReadOnly {
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
                        Some(reverted_cmd) => Some(jsonrpc_ok(
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
                        )),
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
struct ParsedHttpRequest {
    method: String,
    path: String,
    host: String,
    duplicate_host: bool,
    origin: Option<String>,
    authorization: Option<String>,
    body: Vec<u8>,
}

#[cfg(not(target_arch = "wasm32"))]
fn read_loopback_http_request(
    stream: &mut TcpStream,
) -> Result<Option<ParsedHttpRequest>, (u16, String)> {
    use nmr_core::mcp::{
        parse_bounded_content_length, read_bounded_line, ContentLengthError, MAX_MCP_HEADER_COUNT,
        MAX_MCP_HEADER_LINE_BYTES,
    };

    let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(5)));

    let mut reader = BufReader::new(stream);
    let mut req_line = String::new();
    let (n, req_exceeded) =
        read_bounded_line(&mut reader, &mut req_line, MAX_MCP_HEADER_LINE_BYTES)
            .map_err(|e| (400, format!("failed reading HTTP request line: {e}")))?;
    if n == 0 || req_line.trim().is_empty() {
        return Ok(None);
    }
    if req_exceeded {
        return Err((413, "HTTP request line exceeds maximum length".to_string()));
    }

    let mut parts = req_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_ascii_uppercase();
    let raw_target = parts.next().unwrap_or("/");
    let path = raw_target.split('?').next().unwrap_or("/").to_string();

    let mut content_length = 0usize;
    let mut host = String::new();
    let mut seen_host = false;
    let mut duplicate_host = false;
    let mut origin = None;
    let mut authorization = None;
    let mut header_count = 0usize;

    loop {
        header_count += 1;
        if header_count > MAX_MCP_HEADER_COUNT {
            return Err((400, "too many HTTP header lines".to_string()));
        }
        let mut line = String::new();
        let (bytes, hdr_exceeded) =
            read_bounded_line(&mut reader, &mut line, MAX_MCP_HEADER_LINE_BYTES)
                .map_err(|e| (400, format!("failed reading HTTP header: {e}")))?;
        if bytes == 0 || line == "\r\n" || line == "\n" {
            break;
        }
        if hdr_exceeded {
            return Err((413, "HTTP header line exceeds maximum length".to_string()));
        }
        if let Some((k, v)) = line.split_once(':') {
            let key = k.trim();
            if key.eq_ignore_ascii_case("content-length") {
                content_length = match parse_bounded_content_length(v, MAX_MCP_HTTP_BODY_BYTES) {
                    Ok(len) => len,
                    Err(ContentLengthError::Malformed) => {
                        return Err((400, "invalid Content-Length".to_string()));
                    }
                    Err(ContentLengthError::ExceedsLimit { .. }) => {
                        return Err((413, "MCP HTTP payload exceeds 4 MiB limit".to_string()));
                    }
                };
            } else if key.eq_ignore_ascii_case("host") {
                if seen_host {
                    duplicate_host = true;
                }
                seen_host = true;
                host = v.trim().to_string();
            } else if key.eq_ignore_ascii_case("origin") {
                let trimmed = v.trim();
                if !trimmed.is_empty() {
                    origin = Some(trimmed.to_string());
                }
            } else if key.eq_ignore_ascii_case("authorization") {
                authorization = Some(v.trim().to_string());
            }
        }
    }

    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        reader
            .read_exact(&mut body)
            .map_err(|e| (400, format!("incomplete HTTP body: {e}")))?;
    }

    Ok(Some(ParsedHttpRequest {
        method,
        path,
        host,
        duplicate_host,
        origin,
        authorization,
        body,
    }))
}

#[cfg(not(target_arch = "wasm32"))]
fn write_http_json_response(
    stream: &mut TcpStream,
    status_code: u16,
    reason_phrase: &str,
    extra_headers: &[(&str, &str)],
    body_bytes: &[u8],
) {
    let mut header = format!(
        "HTTP/1.1 {status_code} {reason_phrase}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n",
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
fn handle_loopback_connection(
    mut stream: TcpStream,
    shared: &Arc<Mutex<SharedBridgeInner>>,
    bound_port: u16,
) {
    let req = match read_loopback_http_request(&mut stream) {
        Ok(Some(r)) => r,
        Ok(None) => return,
        Err((code, msg)) => {
            let reason = if code == 413 {
                "Payload Too Large"
            } else {
                "Bad Request"
            };
            let payload = json!({ "error": msg }).to_string();
            write_http_json_response(&mut stream, code, reason, &[], payload.as_bytes());
            return;
        }
    };

    // Step 1: Enforce strict Host + cross-origin browser Origin guards FIRST (403 Forbidden)
    if let Err(reason) = validate_loopback_host_and_origin(
        &req.host,
        req.duplicate_host,
        req.origin.as_deref(),
        bound_port,
    ) {
        let payload = json!({ "error": reason }).to_string();
        write_http_json_response(&mut stream, 403, "Forbidden", &[], payload.as_bytes());
        return;
    }

    // Step 2: Enforce constant-time Bearer token authentication (401 Unauthorized)
    let expected_token = match shared.lock() {
        Ok(guard) => guard.bearer_token.clone(),
        Err(_) => {
            let payload = json!({ "error": "internal lock poisoned" }).to_string();
            write_http_json_response(
                &mut stream,
                500,
                "Internal Server Error",
                &[],
                payload.as_bytes(),
            );
            return;
        }
    };
    if !verify_bearer_authorization(req.authorization.as_deref(), &expected_token) {
        let payload = json!({ "error": "missing or invalid Bearer token" }).to_string();
        write_http_json_response(
            &mut stream,
            401,
            "Unauthorized",
            &[("WWW-Authenticate", "Bearer realm=\"nmr-gui-mcp\"")],
            payload.as_bytes(),
        );
        return;
    }

    // Step 3: Route `/mcp` (and `/`) POST JSON-RPC 2.0 requests
    if !matches!(req.path.as_str(), "/" | "/mcp") {
        let payload = json!({ "error": "not found; POST JSON-RPC 2.0 to /mcp" }).to_string();
        write_http_json_response(&mut stream, 404, "Not Found", &[], payload.as_bytes());
        return;
    }
    if req.method != "POST" {
        let payload = json!({ "error": "method not allowed; expected POST" }).to_string();
        write_http_json_response(
            &mut stream,
            405,
            "Method Not Allowed",
            &[("Allow", "POST")],
            payload.as_bytes(),
        );
        return;
    }

    let response_opt = match shared.lock() {
        Ok(mut guard) => dispatch_mcp_jsonrpc(&mut guard, &req.body),
        Err(_) => {
            let payload = json!({ "error": "internal lock poisoned" }).to_string();
            write_http_json_response(
                &mut stream,
                500,
                "Internal Server Error",
                &[],
                payload.as_bytes(),
            );
            return;
        }
    };

    match response_opt {
        Some(resp_json) => {
            let bytes = serde_json::to_vec(&resp_json).unwrap_or_else(|_| b"{}".to_vec());
            write_http_json_response(&mut stream, 200, "OK", &[], &bytes);
        }
        None => {
            write_http_json_response(&mut stream, 202, "Accepted", &[], b"");
        }
    }
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
    /// Selected target AI client preset for 1-click JSON config copy.
    pub selected_client: McpClientTarget,
    /// Whether the full bearer token is temporarily revealed in the UI panel.
    pub reveal_token: bool,
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
            selected_client: McpClientTarget::Antigravity,
            reveal_token: false,
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
    /// Rejects non-`127.0.0.1` addresses on native targets and returns a descriptive error on `wasm32`.
    pub fn start_bridge(&mut self, bind_addr: &str) -> Result<String, String> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.stop_bridge();
            let v4 = parse_strict_loopback_bind(bind_addr)?;
            let listener = TcpListener::bind(SocketAddr::V4(v4))
                .map_err(|e| format!("failed to bind loopback MCP listener on {v4}: {e}"))?;
            let bound_addr = listener
                .local_addr()
                .map_err(|e| format!("failed to query bound socket address: {e}"))?;
            let bound_port = bound_addr.port();
            let shutdown = Arc::new(AtomicBool::new(false));
            let shutdown_bg = Arc::clone(&shutdown);
            let shared_bg = Arc::clone(&self.shared);

            let join_handle = thread::Builder::new()
                .name("nmr-gui-mcp-loopback".to_string())
                .spawn(move || {
                    for incoming in listener.incoming() {
                        if shutdown_bg.load(Ordering::SeqCst) {
                            break;
                        }
                        let Ok(stream) = incoming else {
                            continue;
                        };
                        if shutdown_bg.load(Ordering::SeqCst) {
                            break;
                        }
                        handle_loopback_connection(stream, &shared_bg, bound_port);
                    }
                })
                .map_err(|e| format!("failed to spawn MCP loopback accept thread: {e}"))?;

            let addr_str = bound_addr.to_string();
            self.server = Some(LoopbackServerHandle {
                bound_addr,
                shutdown,
                join_handle: Some(join_handle),
            });
            self.status_banner = format!("Listening on http://{addr_str}/mcp");
            Ok(addr_str)
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

    /// Current secret bearer token required on `Authorization: Bearer <token>`.
    pub fn bearer_token(&self) -> String {
        self.shared
            .lock()
            .map(|g| g.bearer_token.clone())
            .unwrap_or_default()
    }

    /// Rotate the secret bearer token in-place and return the new token.
    pub fn rotate_bearer_token(&self) -> String {
        let fresh = generate_bearer_token();
        if let Ok(mut g) = self.shared.lock() {
            g.bearer_token = fresh.clone();
        }
        fresh
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
        build_mcp_client_config_json(
            self.selected_client,
            &self.endpoint_url(),
            &self.bearer_token(),
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
                if ui.button("⏹ Stop Bridge").clicked() {
                    controller.stop_bridge();
                }
                if let Some(addr) = controller.bound_addr_string() {
                    ui.colored_label(
                        egui::Color32::from_rgb(90, 220, 140),
                        format!("● Active: http://{addr}/mcp"),
                    );
                }
            } else {
                ui.label("Bind:");
                ui.add(egui::TextEdit::singleline(&mut controller.bind_input).desired_width(130.0));
                if ui.button("▶ Start Local Bridge").clicked() {
                    let target = controller.bind_input.clone();
                    if let Err(err) = controller.start_bridge(&target) {
                        controller.status_banner = format!("Start failed: {err}");
                    }
                }
                ui.colored_label(egui::Color32::GRAY, "○ Stopped");
            }
        });
    }

    ui.separator();

    // Snapshot shared state once for consistent panel rendering
    let (mut perm, token, undo_depth, redo_depth, handled_reqs, audit_rows) =
        match controller.shared.lock() {
            Ok(g) => (
                g.permission,
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
                String::new(),
                0,
                0,
                0,
                Vec::new(),
            ),
        };

    // 2. Permission Tier Switch (`ReadOnly` vs `ReadWrite`)
    ui.horizontal_wrapped(|ui| {
        ui.label("Permission:");
        let prev_perm = perm;
        ui.selectable_value(
            &mut perm,
            McpBridgePermission::ReadOnly,
            "🔒 ReadOnly (Inspect & PNG Only)",
        );
        ui.selectable_value(
            &mut perm,
            McpBridgePermission::ReadWrite,
            "✏ ReadWrite (Undo-Backed Curation)",
        );
        if perm != prev_perm {
            controller.set_permission(perm);
        }
        ui.label(format!("| Requests: {handled_reqs}"));
    });

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
            let _ = controller.rotate_bearer_token();
            controller.status_banner = "Rotated bearer token".to_string();
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
        if ui.button("📋 Copy MCP Config").clicked() {
            let cfg_json = controller.build_client_config_json();
            ui.ctx().copy_text(cfg_json);
            controller.status_banner = format!(
                "Copied {} MCP JSON config to clipboard",
                controller.selected_client.label()
            );
            response.copied_config = true;
        }
    });

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
            .add_enabled(undo_depth > 0, egui::Button::new("↩ Undo AI Action"))
            .clicked()
        {
            if let Some(cmd) = controller.undo_last() {
                controller.status_banner = format!("Undid {}", cmd.summary());
                response.undone_command = Some(cmd);
            }
        }
        if ui
            .add_enabled(redo_depth > 0, egui::Button::new("↪ Redo"))
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
                        McpAuditOutcome::BlockedReadOnly => egui::Color32::from_rgb(245, 110, 110),
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
    use std::io::BufRead;

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

        let tok = generate_bearer_token();
        assert!(tok.starts_with("nmr_mcp_"));
        assert_eq!(tok.len(), 72);
        let masked = mask_bearer_token(&tok);
        assert!(masked.contains("••••••••••••"));
        assert!(!masked.contains(&tok[16..48]));

        for &client in McpClientTarget::ALL {
            let cfg_str = build_mcp_client_config_json(
                client,
                "http://127.0.0.1:54321/mcp",
                "nmr_mcp_testtoken",
            );
            let parsed: Value = serde_json::from_str(&cfg_str).expect("valid client config JSON");
            let entry = &parsed["mcpServers"]["nmrassign-workbench"];
            assert_eq!(
                entry["headers"]["Authorization"],
                "Bearer nmr_mcp_testtoken"
            );
            let url_val = entry
                .get("serverUrl")
                .or_else(|| entry.get("url"))
                .and_then(Value::as_str);
            assert_eq!(url_val, Some("http://127.0.0.1:54321/mcp"));
        }
    }

    #[test]
    fn test_headless_egui_mcp_bridge_panel_renders() {
        let ctrl = McpBridgeController::default();
        ctrl.set_permission(McpBridgePermission::ReadWrite);
        let _ = ctrl.apply_ui_command(McpUiCommand::JumpToResidue { seq_id: 9 });
        let mut ctrl = ctrl;

        let ctx = egui::Context::default();
        let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
            let resp = render_mcp_bridge_panel(ui, &mut ctrl);
            assert!(!resp.copied_config);
        });
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
                "capture_viewport_png"
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

        // 5. Switch to ReadWrite -> apply curation edit -> verify state -> undo_last_ai_edit -> exact restoration
        ctrl.set_permission(McpBridgePermission::ReadWrite);
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
                "POST /mcp HTTP/1.1\r\nHost: {addr}\r\nAuthorization: Bearer {bearer}\r\nContent-Length: {adv_len}\r\nConnection: close\r\n\r\n{{}}"
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
                "POST /mcp HTTP/1.1\r\nHost: {addr}\r\nAuthorization: Bearer {bearer}\r\nContent-Length: -42\r\nConnection: close\r\n\r\n"
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
}
