//! Clean-room Model Context Protocol (`MCP`) JSON-RPC 2.0 server engine, workspace path
//! sandboxing, `ReadOnly` / `ReadWrite` permission guardrails, multimodal `image/png` helpers,
//! guided spectroscopist prompt workflows, and client configuration generator (`T15.1` & `T15.3`).
//!
//! Provenance: Clean-room original implementation of `docs/specs/18-mcp-server-and-bridge.md`
//! from public open specifications and peer-reviewed NMR literature:
//! - Model Context Protocol (`MCP`) Open Specification (`2024-11-05` / `2025-03-26`) and
//!   JSON-RPC 2.0 Specification (`initialize`, `notifications/initialized`, `ping`, `tools/list`,
//!   `tools/call`, `resources/list`, `resources/read`, `prompts/list`, `prompts/get`).
//! - Bray, T. (RFC 8259, Dec 2017) — JSON Data Interchange Format; Fielding, R. et al.
//!   (RFC 9110, Jun 2022) — HTTP Semantics (`Authorization`, `Host`, and `Origin` validation).
//! - Boutell, T. et al. (RFC 2083, Mar 1997) — PNG Specification v1.0; Deutsch, P. & Gailly, J.-L.
//!   (RFC 1950 & RFC 1951, May 1996) — ZLIB stored blocks & `Adler-32`; IEEE 802.3 `CRC-32`;
//!   Josefsson, S. (RFC 4648, Oct 2006) — Base64 encoding.
//! - Shen, Y., Vernon, R., Baker, D. & Bax, A. (*Proc. Natl. Acad. Sci. U.S.A.* 2008, 105,
//!   4685–4690); Shen, Y. & Bax, A. (*J. Biomol. NMR* 2013, 56, 227–241); Berjanskii, M. V. &
//!   Wishart, D. S. (*J. Am. Chem. Soc.* 2005, 127, 14970–14971) — `HomologFirewall` 3/9-mer
//!   chemical-shift torsion prediction.
//! - Guo, C. et al. (*ICML* 2017, 70, 1321–1330) & Chow, C. K. (*IEEE Trans. Inf. Theory* 1970,
//!   16, 41–46) — `DatasetQcReport` calibration and selective abstention.
//! - Kay, L. E., Torchia, D. A. & Bax, A. (*Biochemistry* 1989, 28, 8972–8979); Vallurupalli, P.,
//!   Hansen, D. F. & Kay, L. E. (*Proc. Natl. Acad. Sci. U.S.A.* 2012, 109, 11177–11182); Baldwin,
//!   A. J. (*J. Magn. Reson.* 2014, 244, 114–124); Williamson, M. P. (*Prog. Nucl. Magn. Reson.
//!   Spectrosc.* 2013, 73, 1–16) — Phase 13 relaxation, `CEST`/`CPMG` exchange, and `CSP` tracking.
//! - Fulmer, G. R. et al. (*Organometallics* 2010, 29, 2176–2179); Hoye, T. R., Hanson, P. R. &
//!   Vyvyan, J. R. (*J. Org. Chem.* 1994, 59, 4096–4103); Stejskal, E. O. & Tanner, J. E. (*J.
//!   Chem. Phys.* 1965, 42, 288–292); Wu, D. H., Chen, A. D. & Johnson, C. S. Jr. (*J. Magn.
//!   Reson. A* 1995, 115, 260–264) — Phase 14 1D deconvolution, multiplet trees, and `DOSY`.
//! - De Meyer, T. et al. (*Anal. Chem.* 2008, 80, 3783–3790); Dieterle, F. et al. (*Anal. Chem.*
//!   2006, 78, 4281–4290); Trygg, J. & Wold, S. (*J. Chemom.* 2002, 16, 119–128); Westerhuis,
//!   J. A. et al. (*Metabolomics* 2008, 4, 81–89); Cloarec, O. et al. (*Anal. Chem.* 2005, 77,
//!   1282–1289) — Phase 14 `AI-Binning`, `PQN`, `PCA`, `OPLS-DA`, and `STOCSY` chemometrics.

use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::io::{self, BufRead, Read, Write};
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::cest::{fit_cest_global_cpu, read_cest_profiles_tsv, CestFitConfig};
use crate::classify::classify_nmr_workspace;
use crate::cpmg::{fit_cpmg_global_cpu, read_cpmg_series_tsv, CpmgFitConfig, CpmgModel};
use crate::csp::{
    fit_csp_global_kd, track_methyl_hmqc_trajectories, track_titration_trajectories,
    AssignedAmidePeak, CspAnalysisConfig, UnassignedHsqcPeak, UnassignedTitrationStep,
};
use crate::dosy::{
    compute_b_values_s_m2, estimate_hydrodynamics, fit_stejskal_tanner_multi, read_bruker_dosy_dir,
    read_varian_dosy_dir, solve_tikhonov_ilt, BrukerDosyOverrides, DosyFitConfig, DosyNucleus,
    DosySequenceType, DosySolvent, GradientShape, IltConfig, ModelSelectionCriterion,
    SoluteCompactnessClass, VarianDosyOverrides, ILT_ARTIFACT_CAVEAT,
};
use crate::metabolomics::{
    compute_stocsy_at_ppm, run_metabolomics_pipeline, BiofluidMaskPreset, BucketingMethod,
    CohortSample1D, MetabolomicsPipelineConfig, NormalizationMethod, ScalingMethod,
};
use crate::multiplet::{
    analyze_multiplet_region, format_journal_nmr_report, JournalNmrReport, MultipletConfig,
    PublicationStyle, ReportNucleus, ReportedSignal, TextEncoding,
};
use crate::nmr1d::{
    auto_calibrate_solvent_1d, deconvolve_spectrum_1d, estimate_noise_sigma_1d,
    load_processed_spectrum_1d, match_trace_impurities, pick_peaks_1d, DeconvolveConfig1D,
    PeakPickConfig1D,
};
use crate::nmrstar::{read_star_all_atoms, write_star_with_header, Reference};
use crate::peaks::{
    experiment_rules, read_peaks, read_peaks_str, write_peaks, Peak, ALL_EXPERIMENT_NAMES,
    EXPERIMENT_NAMES,
};
use crate::priors::{
    estimate_reference_offsets_detailed, one_to_three, sync_detailed_reference_offsets, Priors,
    ATOMS,
};
use crate::pulprog::interpret_experiment_dir;
use crate::qc::{apply_dataset_qc_to_assignments, evaluate_dataset_qc};
use crate::relax_fit::{decay_inputs_from_series, fit_decay_batch_cpu, ExpFitConfig};
use crate::relax_io::{parse_relax_series_nef_str, parse_relax_series_tsv_str};
use crate::solver::{assign, prepare_scoring_priors, AnnealConfig, ScoreConfig};
use crate::spinsystems::{build_spin_systems, Tolerances};
use crate::torsion_db::{
    sha256_hex, HomologFirewall, HomologFirewallQuery, TorsionPredictionClass,
    DEFAULT_FIREWALL_MAX_SEQ_IDENTITY,
};

/// Required JSON-RPC protocol version string.
pub const JSONRPC_VERSION: &str = "2.0";

/// Default negotiated Model Context Protocol revision.
pub const MCP_PROTOCOL_VERSION: &str = "2024-11-05";

/// Supported MCP protocol revisions accepted during `initialize` negotiation.
pub const SUPPORTED_MCP_PROTOCOL_VERSIONS: &[&str] = &["2024-11-05", "2025-03-26", "2025-06-18"];

/// Maximum single JSON-RPC frame size (`16 MiB`) accepted over `stdio` or HTTP.
pub const MAX_MCP_FRAME_BYTES: usize = 16 * 1024 * 1024;

/// Maximum resource file size (`8 MiB`) returned by `resources/read`.
pub const MAX_RESOURCE_READ_BYTES: u64 = 8 * 1024 * 1024;

/// JSON-RPC 2.0 standard parse error (`Invalid JSON was received by the server`).
pub const ERR_PARSE_ERROR: i32 = -32700;
/// JSON-RPC 2.0 standard invalid request (`The JSON sent is not a valid Request object`).
pub const ERR_INVALID_REQUEST: i32 = -32600;
/// JSON-RPC 2.0 standard method not found (`The method does not exist / is not available`).
pub const ERR_METHOD_NOT_FOUND: i32 = -32601;
/// JSON-RPC 2.0 standard invalid method parameter(s).
pub const ERR_INVALID_PARAMS: i32 = -32602;
/// JSON-RPC 2.0 standard internal error.
pub const ERR_INTERNAL_ERROR: i32 = -32603;

/// MCP server-defined error code for path sandboxing or credential-guard violations.
pub const ERR_SECURITY_VIOLATION: i32 = -32001;
/// MCP server-defined error code when a mutating action is blocked in `ReadOnly` mode.
pub const ERR_READ_ONLY_DENIED: i32 = -32002;
/// MCP server-defined error code when a requested `nmr://` resource URI does not exist.
pub const ERR_RESOURCE_NOT_FOUND: i32 = -32003;

const FORBIDDEN_CREDENTIAL_NAMES: &[&str] = &[
    ".ssh",
    ".aws",
    ".gnupg",
    ".env",
    ".netrc",
    ".npmrc",
    ".pypirc",
    ".kube",
    ".docker",
    ".git-credentials",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
];

// ============================================================================
// Pure-Rust RFC 4648 Base64 & RFC 2083 RGBA PNG Helpers
// ============================================================================

const BASE64_ALPHABET: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Encode arbitrary bytes as standard padded RFC 4648 Base64.
pub fn encode_base64(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;

        out.push(BASE64_ALPHABET[((triple >> 18) & 0x3F) as usize] as char);
        out.push(BASE64_ALPHABET[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            out.push(BASE64_ALPHABET[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(BASE64_ALPHABET[(triple & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

/// Decode standard RFC 4648 Base64 (ignoring ASCII whitespace) into raw bytes.
pub fn decode_base64(input: &str) -> Result<Vec<u8>, String> {
    let filtered: Vec<u8> = input.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    if filtered.is_empty() {
        return Ok(Vec::new());
    }
    if !filtered.len().is_multiple_of(4) {
        return Err("invalid Base64 length (must be a multiple of 4)".to_string());
    }
    let decode_sextet = |b: u8| -> Result<u8, String> {
        match b {
            b'A'..=b'Z' => Ok(b - b'A'),
            b'a'..=b'z' => Ok(b - b'a' + 26),
            b'0'..=b'9' => Ok(b - b'0' + 52),
            b'+' => Ok(62),
            b'/' => Ok(63),
            _ => Err(format!("invalid Base64 byte 0x{b:02x}")),
        }
    };

    let mut out = Vec::with_capacity((filtered.len() / 4) * 3);
    for (ci, chunk) in filtered.chunks_exact(4).enumerate() {
        let is_last = ci + 1 == filtered.len() / 4;
        let pad = if chunk[3] == b'=' {
            if chunk[2] == b'=' {
                2
            } else {
                1
            }
        } else {
            0
        };
        if pad > 0 && !is_last {
            return Err("unexpected Base64 '=' padding outside final quantum".to_string());
        }
        if chunk[2] == b'=' && chunk[3] != b'=' {
            return Err("invalid Base64 '=X' padding".to_string());
        }

        let s0 = decode_sextet(chunk[0])? as u32;
        let s1 = decode_sextet(chunk[1])? as u32;
        let s2 = if pad >= 2 {
            0
        } else {
            decode_sextet(chunk[2])? as u32
        };
        let s3 = if pad >= 1 {
            0
        } else {
            decode_sextet(chunk[3])? as u32
        };
        let triple = (s0 << 18) | (s1 << 12) | (s2 << 6) | s3;
        out.push(((triple >> 16) & 0xFF) as u8);
        if pad < 2 {
            out.push(((triple >> 8) & 0xFF) as u8);
        }
        if pad < 1 {
            out.push((triple & 0xFF) as u8);
        }
    }
    Ok(out)
}

fn png_crc32(chunk_type: &[u8; 4], data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &b in chunk_type.iter().chain(data.iter()) {
        crc ^= b as u32;
        for _ in 0..8 {
            if (crc & 1) != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

fn adler32_checksum(data: &[u8]) -> u32 {
    const MOD_ADLER: u32 = 65_521;
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    for chunk in data.chunks(5552) {
        for &byte in chunk {
            a += byte as u32;
            b += a;
        }
        a %= MOD_ADLER;
        b %= MOD_ADLER;
    }
    (b << 16) | a
}

fn append_png_chunk(out: &mut Vec<u8>, chunk_type: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(chunk_type);
    out.extend_from_slice(data);
    out.extend_from_slice(&png_crc32(chunk_type, data).to_be_bytes());
}

/// Encode an 8-bit RGBA framebuffer (`width * height * 4` bytes) as a valid RFC 2083 PNG byte stream.
pub fn encode_rgba_png(width: u32, height: u32, rgba: &[u8]) -> Result<Vec<u8>, String> {
    if width == 0 || height == 0 {
        return Err("PNG width and height must both be >= 1".to_string());
    }
    let row_bytes = (width as usize)
        .checked_mul(4)
        .ok_or_else(|| "PNG row byte length overflow".to_string())?;
    let expected_len = row_bytes
        .checked_mul(height as usize)
        .ok_or_else(|| "PNG buffer byte length overflow".to_string())?;
    if rgba.len() != expected_len {
        return Err(format!(
            "RGBA buffer length mismatch: got {} bytes, expected {} ({width}x{height}x4)",
            rgba.len(),
            expected_len
        ));
    }

    let raw_scanline_len = (row_bytes + 1)
        .checked_mul(height as usize)
        .ok_or_else(|| "PNG scanline length overflow".to_string())?;
    let mut raw_scanlines = Vec::with_capacity(raw_scanline_len);
    for row in rgba.chunks_exact(row_bytes) {
        raw_scanlines.push(0u8); // Filter type 0 (None)
        raw_scanlines.extend_from_slice(row);
    }

    // RFC 1950 ZLIB stream wrapping uncompressed stored DEFLATE blocks (max 65,535 bytes each)
    let mut zlib =
        Vec::with_capacity(raw_scanlines.len() + (raw_scanlines.len() / 65535 + 1) * 5 + 6);
    zlib.extend_from_slice(&[0x78, 0x01]); // CMF = deflate 32K window, FLG = no dict, check bits valid
    let chunks: Vec<&[u8]> = raw_scanlines.chunks(65_535).collect();
    for (idx, block) in chunks.iter().enumerate() {
        let is_final = idx + 1 == chunks.len();
        zlib.push(u8::from(is_final)); // BFINAL | BTYPE=00 (stored)
        let len16 = block.len() as u16;
        let nlen16 = !len16;
        zlib.extend_from_slice(&len16.to_le_bytes());
        zlib.extend_from_slice(&nlen16.to_le_bytes());
        zlib.extend_from_slice(block);
    }
    zlib.extend_from_slice(&adler32_checksum(&raw_scanlines).to_be_bytes());

    let mut png = Vec::with_capacity(zlib.len() + 64);
    png.extend_from_slice(b"\x89PNG\r\n\x1a\n");

    let mut ihdr = [0u8; 13];
    ihdr[0..4].copy_from_slice(&width.to_be_bytes());
    ihdr[4..8].copy_from_slice(&height.to_be_bytes());
    ihdr[8] = 8; // 8 bits per channel
    ihdr[9] = 6; // Color type 6 = RGBA
    ihdr[10] = 0; // Compression = deflate
    ihdr[11] = 0; // Filter = adaptive
    ihdr[12] = 0; // Interlace = none
    append_png_chunk(&mut png, b"IHDR", &ihdr);
    append_png_chunk(&mut png, b"IDAT", &zlib);
    append_png_chunk(&mut png, b"IEND", &[]);
    Ok(png)
}

// ============================================================================
// Security Policy, Access Modes & Canonicalized Path Sandbox (`T15.1` & `T15.3`)
// ============================================================================

/// Capability tier governing whether MCP tools may mutate files inside `allowed_roots`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum McpAccessMode {
    /// Default-safe read-only inspection, QC triage, and torsion prediction mode.
    #[default]
    ReadOnly,
    /// Explicit opt-in read-write mode enabling curated peak TSV and NMR-STAR shift exports
    /// strictly inside canonicalized `allowed_roots`.
    ReadWrite,
}

impl McpAccessMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ReadOnly => "read-only",
            Self::ReadWrite => "read-write",
        }
    }

    pub fn cli_flag(self) -> &'static str {
        match self {
            Self::ReadOnly => "--read-only",
            Self::ReadWrite => "--read-write",
        }
    }
}

impl fmt::Display for McpAccessMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Classification of security or sandbox rejections produced by [`McpSecurityPolicy`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum McpSecurityErrorKind {
    NoAllowedRoots,
    NullByteOrInvalidPath,
    ParentTraversal,
    CredentialPathBlocked,
    SymlinkEscape,
    OutsideAllowedRoots,
    ReadOnlyModeDenied,
    PathNotFound,
    IoError,
}

impl McpSecurityErrorKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NoAllowedRoots => "no_allowed_roots",
            Self::NullByteOrInvalidPath => "null_byte_or_invalid_path",
            Self::ParentTraversal => "parent_traversal",
            Self::CredentialPathBlocked => "credential_path_blocked",
            Self::SymlinkEscape => "symlink_escape",
            Self::OutsideAllowedRoots => "outside_allowed_roots",
            Self::ReadOnlyModeDenied => "read_only_mode_denied",
            Self::PathNotFound => "path_not_found",
            Self::IoError => "io_error",
        }
    }

    pub fn rpc_code(self) -> i32 {
        match self {
            Self::ReadOnlyModeDenied => ERR_READ_ONLY_DENIED,
            Self::PathNotFound => ERR_RESOURCE_NOT_FOUND,
            Self::NoAllowedRoots
            | Self::NullByteOrInvalidPath
            | Self::ParentTraversal
            | Self::CredentialPathBlocked
            | Self::SymlinkEscape
            | Self::OutsideAllowedRoots => ERR_SECURITY_VIOLATION,
            Self::IoError => ERR_INTERNAL_ERROR,
        }
    }
}

/// Structured security/sandbox error convertible into both JSON-RPC errors and MCP tool blocks.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpSecurityError {
    pub kind: McpSecurityErrorKind,
    pub message: String,
}

impl McpSecurityError {
    pub fn new(kind: McpSecurityErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    pub fn to_json_rpc_error(&self) -> JsonRpcError {
        JsonRpcError {
            code: self.kind.rpc_code(),
            message: self.message.clone(),
            data: Some(json!({
                "errorKind": self.kind.as_str(),
            })),
        }
    }

    pub fn to_tool_error_result(&self) -> McpCallToolResult {
        let code = self.kind.rpc_code();
        let kind = self.kind.as_str();
        McpCallToolResult {
            content: vec![McpContentBlock::text(format!(
                "[MCP security/{kind} ({code})]: {}",
                self.message
            ))],
            is_error: Some(true),
            structured_content: Some(json!({
                "error": {
                    "code": code,
                    "kind": kind,
                    "message": self.message,
                }
            })),
        }
    }
}

impl fmt::Display for McpSecurityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.kind.as_str(), self.message)
    }
}

/// Workspace sandbox and read/write permission policy enforced across all MCP tools and resources.
#[derive(Clone, Debug)]
pub struct McpSecurityPolicy {
    allowed_roots: Vec<PathBuf>,
    access_mode: McpAccessMode,
    hide_write_tools_in_read_only: bool,
}

impl Default for McpSecurityPolicy {
    fn default() -> Self {
        Self::empty_read_only()
    }
}

impl McpSecurityPolicy {
    /// Construct an empty `ReadOnly` policy with zero filesystem roots (blocks all file access).
    pub fn empty_read_only() -> Self {
        Self {
            allowed_roots: Vec::new(),
            access_mode: McpAccessMode::ReadOnly,
            hide_write_tools_in_read_only: false,
        }
    }

    /// Construct a policy canonicalizing every root directory in `allowed_roots`.
    pub fn new<I, P>(allowed_roots: I, access_mode: McpAccessMode) -> Result<Self, McpSecurityError>
    where
        I: IntoIterator<Item = P>,
        P: AsRef<Path>,
    {
        let mut canonical_roots = Vec::new();
        for root_ref in allowed_roots {
            let root = root_ref.as_ref();
            let canon = fs::canonicalize(root).map_err(|e| {
                McpSecurityError::new(
                    McpSecurityErrorKind::IoError,
                    format!("failed to canonicalize allowed workspace root: {e}"),
                )
            })?;
            if !canon.is_dir() {
                return Err(McpSecurityError::new(
                    McpSecurityErrorKind::NullByteOrInvalidPath,
                    "allowed workspace root must be an existing directory",
                ));
            }
            if !canonical_roots.contains(&canon) {
                canonical_roots.push(canon);
            }
        }
        Ok(Self {
            allowed_roots: canonical_roots,
            access_mode,
            hide_write_tools_in_read_only: false,
        })
    }

    /// Convenience constructor for the default `ReadOnly` sandbox over `allowed_roots`.
    pub fn read_only<I, P>(allowed_roots: I) -> Result<Self, McpSecurityError>
    where
        I: IntoIterator<Item = P>,
        P: AsRef<Path>,
    {
        Self::new(allowed_roots, McpAccessMode::ReadOnly)
    }

    /// Convenience constructor for `ReadWrite` sandbox mode over `allowed_roots`.
    pub fn read_write<I, P>(allowed_roots: I) -> Result<Self, McpSecurityError>
    where
        I: IntoIterator<Item = P>,
        P: AsRef<Path>,
    {
        Self::new(allowed_roots, McpAccessMode::ReadWrite)
    }

    /// Configure whether `tools/list` omits mutating tools altogether when `ReadOnly` is active.
    pub fn with_hide_write_tools_in_read_only(mut self, hide: bool) -> Self {
        self.hide_write_tools_in_read_only = hide;
        self
    }

    pub fn allowed_roots(&self) -> &[PathBuf] {
        &self.allowed_roots
    }

    pub fn access_mode(&self) -> McpAccessMode {
        self.access_mode
    }

    pub fn hide_write_tools_in_read_only(&self) -> bool {
        self.hide_write_tools_in_read_only
    }

    /// Verify that `McpAccessMode::ReadWrite` is active before executing a mutating tool.
    pub fn require_write_access(&self, tool_name: &str) -> Result<(), McpSecurityError> {
        if self.access_mode != McpAccessMode::ReadWrite {
            return Err(McpSecurityError::new(
                McpSecurityErrorKind::ReadOnlyModeDenied,
                format!(
                    "mutating tool '{tool_name}' is blocked because MCP server is in ReadOnly mode"
                ),
            ));
        }
        Ok(())
    }

    /// Resolve and validate an existing path inside `allowed_roots`, rejecting null bytes, `..`
    /// traversal, hidden credential paths, and symlinks resolving outside `allowed_roots`.
    pub fn resolve_sandboxed_path(&self, raw_path: &str) -> Result<PathBuf, McpSecurityError> {
        self.resolve_internal(raw_path, false)
    }

    /// Resolve and validate an output write path (which may or may not exist yet) inside
    /// `allowed_roots`, enforcing `ReadWrite` mode, symlink containment, and credential guards.
    pub fn resolve_sandboxed_write_path(
        &self,
        raw_path: &str,
        tool_name: &str,
    ) -> Result<PathBuf, McpSecurityError> {
        self.require_write_access(tool_name)?;
        self.resolve_internal(raw_path, true)
    }

    /// Format a resolved sandboxed path as a clean workspace-relative string without host prefixes.
    pub fn relativize_display_path(&self, resolved: &Path) -> String {
        for (idx, root) in self.allowed_roots.iter().enumerate() {
            if let Ok(rel) = resolved.strip_prefix(root) {
                let rel_str = path_to_forward_slashes(rel);
                let label = if rel_str.is_empty() { "." } else { &rel_str };
                return if self.allowed_roots.len() == 1 {
                    label.to_string()
                } else {
                    format!("root[{idx}]/{label}")
                };
            }
        }
        resolved
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(".")
            .to_string()
    }

    fn resolve_internal(
        &self,
        raw_path: &str,
        allow_create_leaf: bool,
    ) -> Result<PathBuf, McpSecurityError> {
        if raw_path.contains('\0') {
            return Err(McpSecurityError::new(
                McpSecurityErrorKind::NullByteOrInvalidPath,
                "path contains a forbidden null byte",
            ));
        }
        let trimmed = raw_path.trim();
        if trimmed.is_empty() {
            return Err(McpSecurityError::new(
                McpSecurityErrorKind::NullByteOrInvalidPath,
                "path must not be empty",
            ));
        }
        if self.allowed_roots.is_empty() {
            return Err(McpSecurityError::new(
                McpSecurityErrorKind::NoAllowedRoots,
                "no allowed workspace roots are configured on this MCP server",
            ));
        }

        let input_path = Path::new(trimmed);
        validate_lexical_components(input_path)?;

        let candidate_paths: Vec<PathBuf> = if input_path.is_absolute() {
            vec![input_path.to_path_buf()]
        } else {
            self.allowed_roots
                .iter()
                .map(|r| r.join(input_path))
                .collect()
        };

        let mut saw_symlink = false;
        let mut saw_outside_root = false;
        let mut first_credential_err: Option<McpSecurityError> = None;

        for cand in &candidate_paths {
            if path_chain_contains_symlink(&self.allowed_roots, cand) {
                saw_symlink = true;
            }

            if let Ok(canon) = fs::canonicalize(cand) {
                if let Some(matched_root) = self
                    .allowed_roots
                    .iter()
                    .find(|root| canon.starts_with(root.as_path()))
                {
                    if let Ok(rel) = canon.strip_prefix(matched_root) {
                        if let Err(cred_err) = validate_relative_credential_components(rel) {
                            first_credential_err = Some(cred_err);
                            continue;
                        }
                    }
                    return Ok(canon);
                }
                saw_outside_root = true;
            } else if allow_create_leaf {
                // Reject dangling symlinks at or above the target path immediately.
                if fs::symlink_metadata(cand)
                    .map(|m| m.file_type().is_symlink())
                    .unwrap_or(false)
                {
                    return Err(McpSecurityError::new(
                        McpSecurityErrorKind::SymlinkEscape,
                        "refusing to write through a dangling symlink",
                    ));
                }

                // Walk upward to the deepest existing ancestor, canonicalize it, and verify containment.
                let mut cur: Option<&Path> = cand.parent();
                let mut tail_segments: Vec<String> = Vec::new();
                if let Some(leaf) = cand.file_name().and_then(|s| s.to_str()) {
                    tail_segments.push(leaf.to_string());
                } else {
                    continue;
                }

                while let Some(anc) = cur {
                    if let Ok(meta) = fs::symlink_metadata(anc) {
                        if meta.file_type().is_symlink() {
                            saw_symlink = true;
                        }
                        if let Ok(canon_anc) = fs::canonicalize(anc) {
                            if let Some(matched_root) = self
                                .allowed_roots
                                .iter()
                                .find(|root| canon_anc.starts_with(root.as_path()))
                            {
                                let mut full = canon_anc;
                                for seg in tail_segments.iter().rev() {
                                    full.push(seg);
                                }
                                if let Ok(rel) = full.strip_prefix(matched_root) {
                                    if let Err(cred_err) =
                                        validate_relative_credential_components(rel)
                                    {
                                        first_credential_err = Some(cred_err);
                                        break;
                                    }
                                }
                                return Ok(full);
                            }
                            saw_outside_root = true;
                        }
                        break;
                    }
                    if let Some(seg) = anc.file_name().and_then(|s| s.to_str()) {
                        tail_segments.push(seg.to_string());
                        cur = anc.parent();
                    } else {
                        break;
                    }
                }
            }
        }

        if let Some(cred_err) = first_credential_err {
            return Err(cred_err);
        }
        if saw_symlink && saw_outside_root {
            return Err(McpSecurityError::new(
                McpSecurityErrorKind::SymlinkEscape,
                "symlink target resolves outside allowed workspace roots",
            ));
        }
        if saw_symlink {
            return Err(McpSecurityError::new(
                McpSecurityErrorKind::SymlinkEscape,
                "symlink target is dangling or resolves outside allowed workspace roots",
            ));
        }
        if saw_outside_root || input_path.is_absolute() {
            return Err(McpSecurityError::new(
                McpSecurityErrorKind::OutsideAllowedRoots,
                "requested path resolves outside allowed workspace roots",
            ));
        }
        Err(McpSecurityError::new(
            McpSecurityErrorKind::PathNotFound,
            "requested workspace path does not exist",
        ))
    }
}

fn is_forbidden_credential_segment(seg: &str) -> bool {
    let lower = seg.to_ascii_lowercase();
    FORBIDDEN_CREDENTIAL_NAMES.contains(&lower.as_str()) || lower.starts_with(".env.")
}

fn validate_lexical_components(path: &Path) -> Result<(), McpSecurityError> {
    for comp in path.components() {
        match comp {
            Component::ParentDir => {
                return Err(McpSecurityError::new(
                    McpSecurityErrorKind::ParentTraversal,
                    "relative parent traversal ('..') is forbidden in MCP paths",
                ));
            }
            Component::Normal(os) => {
                let s = os.to_str().ok_or_else(|| {
                    McpSecurityError::new(
                        McpSecurityErrorKind::NullByteOrInvalidPath,
                        "non-UTF-8 path segment is forbidden",
                    )
                })?;
                if is_forbidden_credential_segment(s) {
                    return Err(McpSecurityError::new(
                        McpSecurityErrorKind::CredentialPathBlocked,
                        format!("access to sensitive credential path segment '{s}' is forbidden"),
                    ));
                }
            }
            Component::CurDir | Component::RootDir | Component::Prefix(_) => {}
        }
    }
    Ok(())
}

fn validate_relative_credential_components(rel: &Path) -> Result<(), McpSecurityError> {
    for comp in rel.components() {
        if let Component::Normal(os) = comp {
            if let Some(s) = os.to_str() {
                if is_forbidden_credential_segment(s) {
                    return Err(McpSecurityError::new(
                        McpSecurityErrorKind::CredentialPathBlocked,
                        format!(
                            "resolved path enters forbidden credential file or directory '{s}'"
                        ),
                    ));
                }
            }
        }
    }
    Ok(())
}

fn path_chain_contains_symlink(roots: &[PathBuf], candidate: &Path) -> bool {
    let mut cur = PathBuf::new();
    let mut inside_root = false;
    for comp in candidate.components() {
        cur.push(comp.as_os_str());
        if inside_root {
            if fs::symlink_metadata(&cur)
                .map(|m| m.file_type().is_symlink())
                .unwrap_or(false)
            {
                return true;
            }
        } else if roots.iter().any(|r| r == &cur)
            || fs::canonicalize(&cur)
                .ok()
                .is_some_and(|canon| roots.iter().any(|r| r == &canon))
        {
            inside_root = true;
        }
    }
    false
}

fn path_to_forward_slashes(path: &Path) -> String {
    let mut parts = Vec::new();
    for c in path.components() {
        if let Component::Normal(s) = c {
            if let Some(txt) = s.to_str() {
                parts.push(txt);
            }
        }
    }
    parts.join("/")
}

// ============================================================================
// JSON-RPC 2.0 & MCP Wire Types
// ============================================================================

/// Structured JSON-RPC 2.0 error payload.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

/// Outbound JSON-RPC 2.0 response envelope.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

impl JsonRpcResponse {
    pub fn success(id: Value, result: Value) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_string(),
            id,
            result: Some(result),
            error: None,
        }
    }

    pub fn error(id: Value, code: i32, message: impl Into<String>, data: Option<Value>) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_string(),
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message: message.into(),
                data,
            }),
        }
    }

    pub fn from_rpc_error(id: Value, err: JsonRpcError) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_string(),
            id,
            result: None,
            error: Some(err),
        }
    }
}

/// Embedded resource payload inside MCP `resources/read` or multimodal tool results.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpResourceContents {
    pub uri: String,
    #[serde(rename = "mimeType", skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blob: Option<String>,
}

/// Multimodal MCP content block (`text`, `image`, or `resource`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum McpContentBlock {
    Text {
        text: String,
    },
    Image {
        data: String,
        #[serde(rename = "mimeType")]
        mime_type: String,
    },
    Resource {
        resource: McpResourceContents,
    },
}

impl McpContentBlock {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    pub fn image_png_base64(base64_data: impl Into<String>) -> Self {
        Self::Image {
            data: base64_data.into(),
            mime_type: "image/png".to_string(),
        }
    }

    /// Encode an 8-bit RGBA framebuffer directly into an MCP `image/png` content block.
    pub fn image_png_from_rgba(width: u32, height: u32, rgba: &[u8]) -> Result<Self, String> {
        let png = encode_rgba_png(width, height, rgba)?;
        Ok(Self::image_png_base64(encode_base64(&png)))
    }
}

/// MCP tool safety annotations (`2025-03-26` / `2024-11-05` compatible).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpToolAnnotations {
    pub title: String,
    #[serde(rename = "readOnlyHint")]
    pub read_only_hint: bool,
    #[serde(rename = "destructiveHint")]
    pub destructive_hint: bool,
    #[serde(rename = "idempotentHint")]
    pub idempotent_hint: bool,
    #[serde(rename = "openWorldHint")]
    pub open_world_hint: bool,
}

/// MCP `tools/list` entry with JSON Schema input validation metadata and safety hints.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct McpToolDescriptor {
    pub name: String,
    pub description: String,
    #[serde(rename = "inputSchema")]
    pub input_schema: Value,
    pub annotations: McpToolAnnotations,
}

/// MCP `tools/call` result envelope.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct McpCallToolResult {
    pub content: Vec<McpContentBlock>,
    #[serde(rename = "isError", skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
    #[serde(rename = "structuredContent", skip_serializing_if = "Option::is_none")]
    pub structured_content: Option<Value>,
}

impl McpCallToolResult {
    pub fn ok_json(summary_payload: &Value) -> Self {
        let pretty =
            serde_json::to_string_pretty(summary_payload).unwrap_or_else(|_| "{}".to_string());
        Self {
            content: vec![McpContentBlock::text(pretty)],
            is_error: Some(false),
            structured_content: Some(summary_payload.clone()),
        }
    }

    pub fn domain_error(kind: &str, message: impl Into<String>) -> Self {
        let msg = message.into();
        Self {
            content: vec![McpContentBlock::text(format!("[MCP {kind}]: {msg}"))],
            is_error: Some(true),
            structured_content: Some(json!({
                "error": {
                    "kind": kind,
                    "message": msg,
                }
            })),
        }
    }
}

/// MCP `resources/list` descriptor.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpResourceDescriptor {
    pub uri: String,
    pub name: String,
    pub description: String,
    #[serde(rename = "mimeType")]
    pub mime_type: String,
}

/// Argument metadata for a guided spectroscopist MCP prompt.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpPromptArgument {
    pub name: String,
    pub description: String,
    pub required: bool,
}

/// MCP `prompts/list` descriptor.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpPromptDescriptor {
    pub name: String,
    pub description: String,
    pub arguments: Vec<McpPromptArgument>,
}

/// Individual message returned by `prompts/get`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpPromptMessage {
    pub role: String,
    pub content: McpContentBlock,
}

/// MCP `prompts/get` result payload.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpGetPromptResult {
    pub description: String,
    pub messages: Vec<McpPromptMessage>,
}

// ============================================================================
// Core MCP Server State & Synchronous Request Dispatcher (`T15.1`)
// ============================================================================

/// Telemetry counters returned when `run_mcp_stdio_loop` reaches EOF.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpLoopStats {
    pub frames_read: usize,
    pub responses_written: usize,
    pub notifications_handled: usize,
}

/// Clean-room Model Context Protocol JSON-RPC 2.0 server instance.
#[derive(Clone, Debug)]
pub struct McpServer {
    policy: McpSecurityPolicy,
    initialized: bool,
    negotiated_protocol_version: String,
    client_name: Option<String>,
}

impl McpServer {
    pub fn new(policy: McpSecurityPolicy) -> Self {
        Self {
            policy,
            initialized: false,
            negotiated_protocol_version: MCP_PROTOCOL_VERSION.to_string(),
            client_name: None,
        }
    }

    pub fn policy(&self) -> &McpSecurityPolicy {
        &self.policy
    }

    pub fn policy_mut(&mut self) -> &mut McpSecurityPolicy {
        &mut self.policy
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    pub fn negotiated_protocol_version(&self) -> &str {
        &self.negotiated_protocol_version
    }

    /// Dispatch a single raw JSON-RPC 2.0 string (single request, notification, or batch).
    /// Returns `None` for notifications (no `"id"` field) per JSON-RPC 2.0.
    pub fn handle_json_request(&mut self, raw_json: &str) -> Option<String> {
        if raw_json.len() > MAX_MCP_FRAME_BYTES {
            let resp = JsonRpcResponse::error(
                Value::Null,
                ERR_INVALID_REQUEST,
                format!("JSON-RPC frame exceeds maximum size ({MAX_MCP_FRAME_BYTES} bytes)"),
                None,
            );
            return serde_json::to_string(&resp).ok();
        }

        let parsed: Value = match serde_json::from_str(raw_json) {
            Ok(v) => v,
            Err(e) => {
                let resp = JsonRpcResponse::error(
                    Value::Null,
                    ERR_PARSE_ERROR,
                    format!("malformed JSON: {e}"),
                    None,
                );
                return serde_json::to_string(&resp).ok();
            }
        };

        match parsed {
            Value::Object(obj) => self
                .dispatch_request_object(obj)
                .and_then(|resp| serde_json::to_string(&resp).ok()),
            Value::Array(items) => {
                if items.is_empty() {
                    let resp = JsonRpcResponse::error(
                        Value::Null,
                        ERR_INVALID_REQUEST,
                        "JSON-RPC batch array must not be empty",
                        None,
                    );
                    return serde_json::to_string(&resp).ok();
                }
                let mut responses = Vec::new();
                for item in items {
                    match item {
                        Value::Object(obj) => {
                            if let Some(r) = self.dispatch_request_object(obj) {
                                responses.push(r);
                            }
                        }
                        _ => responses.push(JsonRpcResponse::error(
                            Value::Null,
                            ERR_INVALID_REQUEST,
                            "batch item must be a JSON-RPC request object",
                            None,
                        )),
                    }
                }
                if responses.is_empty() {
                    None
                } else {
                    serde_json::to_string(&responses).ok()
                }
            }
            _ => {
                let resp = JsonRpcResponse::error(
                    Value::Null,
                    ERR_INVALID_REQUEST,
                    "JSON-RPC payload must be a JSON object or non-empty batch array",
                    None,
                );
                serde_json::to_string(&resp).ok()
            }
        }
    }

    fn dispatch_request_object(&mut self, mut obj: Map<String, Value>) -> Option<JsonRpcResponse> {
        let has_id = obj.contains_key("id");
        let raw_id = obj.remove("id").unwrap_or(Value::Null);
        if has_id && !matches!(raw_id, Value::Null | Value::Number(_) | Value::String(_)) {
            return Some(JsonRpcResponse::error(
                Value::Null,
                ERR_INVALID_REQUEST,
                "JSON-RPC 'id' must be a string, number, or null",
                None,
            ));
        }

        let Some(ver) = obj.get("jsonrpc").and_then(Value::as_str) else {
            return Some(JsonRpcResponse::error(
                if has_id { raw_id } else { Value::Null },
                ERR_INVALID_REQUEST,
                "missing or non-string 'jsonrpc' field (expected '2.0')",
                None,
            ));
        };
        if ver != JSONRPC_VERSION {
            return Some(JsonRpcResponse::error(
                if has_id { raw_id } else { Value::Null },
                ERR_INVALID_REQUEST,
                format!("unsupported jsonrpc version '{ver}' (expected '2.0')"),
                None,
            ));
        }

        let Some(method) = obj
            .get("method")
            .and_then(Value::as_str)
            .map(str::to_string)
        else {
            return Some(JsonRpcResponse::error(
                if has_id { raw_id } else { Value::Null },
                ERR_INVALID_REQUEST,
                "missing or non-string 'method' field",
                None,
            ));
        };

        let params = obj.remove("params");
        if let Some(ref p) = params {
            if !matches!(p, Value::Object(_) | Value::Array(_) | Value::Null) {
                return Some(JsonRpcResponse::error(
                    if has_id { raw_id } else { Value::Null },
                    ERR_INVALID_REQUEST,
                    "JSON-RPC 'params' must be a structured object or array when present",
                    None,
                ));
            }
        }

        // Notifications (no `id` member) never produce a JSON-RPC response.
        if !has_id {
            match method.as_str() {
                "notifications/initialized" | "initialized" => {
                    self.initialized = true;
                }
                "notifications/cancelled" => {}
                _ => {}
            }
            return None;
        }

        let outcome = match method.as_str() {
            "initialize" => self.rpc_initialize(params),
            "ping" => Ok(json!({})),
            "tools/list" => self.rpc_tools_list(),
            "tools/call" => self.rpc_tools_call(params),
            "resources/list" => self.rpc_resources_list(),
            "resources/templates/list" => Ok(json!({
                "resourceTemplates": [
                    {
                        "uriTemplate": "nmr://file/{root_index}/{relative_path}",
                        "name": "Sandboxed Workspace NMR File",
                        "description": "Read a sandboxed NMR-STAR (.str), NEF (.nef), peak TSV (.tsv), or FASTA (.fasta) file inside allowed_roots.",
                        "mimeType": "text/plain"
                    }
                ]
            })),
            "resources/read" => self.rpc_resources_read(params),
            "prompts/list" => self.rpc_prompts_list(),
            "prompts/get" => self.rpc_prompts_get(params),
            _ => Err(JsonRpcError {
                code: ERR_METHOD_NOT_FOUND,
                message: format!("method not found: '{method}'"),
                data: None,
            }),
        };

        Some(match outcome {
            Ok(val) => JsonRpcResponse::success(raw_id, val),
            Err(err) => JsonRpcResponse::from_rpc_error(raw_id, err),
        })
    }

    fn rpc_initialize(&mut self, params: Option<Value>) -> Result<Value, JsonRpcError> {
        let obj = require_params_object_or_empty(params, "initialize")?;
        if let Some(req_ver) = obj.get("protocolVersion").and_then(Value::as_str) {
            if SUPPORTED_MCP_PROTOCOL_VERSIONS.contains(&req_ver) {
                self.negotiated_protocol_version = req_ver.to_string();
            } else {
                self.negotiated_protocol_version = MCP_PROTOCOL_VERSION.to_string();
            }
        }
        if let Some(cname) = obj
            .get("clientInfo")
            .and_then(Value::as_object)
            .and_then(|ci| ci.get("name"))
            .and_then(Value::as_str)
        {
            self.client_name = Some(cname.to_string());
        }
        self.initialized = true;

        Ok(json!({
            "protocolVersion": self.negotiated_protocol_version,
            "capabilities": {
                "tools": { "listChanged": false },
                "resources": { "subscribe": false, "listChanged": false },
                "prompts": { "listChanged": false }
            },
            "serverInfo": {
                "name": "ferric-mcp",
                "version": crate::VERSION
            },
            "instructions": format!(
                "Clean-room biomolecular & analytical NMR MCP server (mode: {}). All file paths are strictly confined to canonicalized workspace roots; backbone solvers remain sequence-only and torsion predictions enforce the <=25% HomologFirewall.",
                self.policy.access_mode().as_str()
            )
        }))
    }

    fn rpc_tools_list(&self) -> Result<Value, JsonRpcError> {
        let mut tools = builtin_tool_descriptors();
        if self.policy.access_mode() == McpAccessMode::ReadOnly
            && self.policy.hide_write_tools_in_read_only()
        {
            tools.retain(|t| t.annotations.read_only_hint);
        }
        serde_json::to_value(json!({ "tools": tools })).map_err(|e| JsonRpcError {
            code: ERR_INTERNAL_ERROR,
            message: format!("failed to serialize tool list: {e}"),
            data: None,
        })
    }

    fn rpc_tools_call(&mut self, params: Option<Value>) -> Result<Value, JsonRpcError> {
        let obj = require_params_object(params, "tools/call")?;
        let Some(name) = obj.get("name").and_then(Value::as_str) else {
            return Err(JsonRpcError {
                code: ERR_INVALID_PARAMS,
                message: "tools/call requires string parameter 'name'".to_string(),
                data: None,
            });
        };

        let args_map: Map<String, Value> = match obj.get("arguments") {
            None | Some(Value::Null) => Map::new(),
            Some(Value::Object(m)) => m.clone(),
            Some(_) => {
                return Err(JsonRpcError {
                    code: ERR_INVALID_PARAMS,
                    message: "tools/call 'arguments' must be a JSON object".to_string(),
                    data: None,
                });
            }
        };

        let result = match name {
            "inspect_nmr_directory" => self.tool_inspect_nmr_directory(&args_map),
            "read_nmrstar_summary" => self.tool_read_nmrstar_summary(&args_map),
            "read_peak_list_summary" => self.tool_read_peak_list_summary(&args_map),
            "evaluate_dataset_qc" => self.tool_evaluate_dataset_qc(&args_map),
            "predict_homolog_free_torsions" => self.tool_predict_homolog_free_torsions(&args_map),
            "fit_relaxation_or_exchange" => self.tool_fit_relaxation_or_exchange(&args_map),
            "track_csp_titration" => self.tool_track_csp_titration(&args_map),
            "analyze_nmr1d_spectrum" => self.tool_analyze_nmr1d_spectrum(&args_map),
            "fit_dosy_diffusion" => self.tool_fit_dosy_diffusion(&args_map),
            "run_metabolomics_chemometrics" => self.tool_run_metabolomics_chemometrics(&args_map),
            "write_curated_peak_tsv" => self.tool_write_curated_peak_tsv(&args_map),
            "export_nmrstar_shifts" => self.tool_export_nmrstar_shifts(&args_map),
            _ => {
                return Err(JsonRpcError {
                    code: ERR_INVALID_PARAMS,
                    message: format!("unknown MCP tool '{name}'"),
                    data: None,
                });
            }
        };

        serde_json::to_value(result).map_err(|e| JsonRpcError {
            code: ERR_INTERNAL_ERROR,
            message: format!("failed to serialize tool result: {e}"),
            data: None,
        })
    }

    fn rpc_resources_list(&self) -> Result<Value, JsonRpcError> {
        let mut resources = vec![
            McpResourceDescriptor {
                uri: "nmr://workspace/policy".to_string(),
                name: "Workspace Security Policy".to_string(),
                description: "Active MCP access mode (read-only vs read-write) and canonicalized sandbox root count.".to_string(),
                mime_type: "application/json".to_string(),
            },
            McpResourceDescriptor {
                uri: "nmr://catalog/experiments".to_string(),
                name: "Supported NMR Experiments & Regimes".to_string(),
                description: "Canonical multidimensional NMR correlation rules and clean-room solver regimes.".to_string(),
                mime_type: "application/json".to_string(),
            },
        ];

        if !self.policy.allowed_roots().is_empty() {
            resources.push(McpResourceDescriptor {
                uri: "nmr://workspace/inventory".to_string(),
                name: "Primary Workspace NMR Inventory".to_string(),
                description:
                    "Auto-classified spectrometer folders, pulse sequences, FASTA, and STAR/NEF files."
                        .to_string(),
                mime_type: "application/json".to_string(),
            });

            for (root_idx, root) in self.policy.allowed_roots().iter().enumerate() {
                let mut discovered = Vec::new();
                collect_sandboxed_resource_files(root, root, &mut discovered, 0, 64);
                discovered.sort();
                for rel in discovered {
                    let mime = if rel.ends_with(".tsv") {
                        "text/tab-separated-values"
                    } else {
                        "text/plain"
                    };
                    resources.push(McpResourceDescriptor {
                        uri: format!("nmr://file/{root_idx}/{rel}"),
                        name: rel.clone(),
                        description: format!(
                            "Sandboxed NMR interchange file '{rel}' in root[{root_idx}]"
                        ),
                        mime_type: mime.to_string(),
                    });
                }
            }
        }

        Ok(json!({ "resources": resources }))
    }

    fn rpc_resources_read(&self, params: Option<Value>) -> Result<Value, JsonRpcError> {
        let obj = require_params_object(params, "resources/read")?;
        let Some(uri) = obj.get("uri").and_then(Value::as_str) else {
            return Err(JsonRpcError {
                code: ERR_INVALID_PARAMS,
                message: "resources/read requires string parameter 'uri'".to_string(),
                data: None,
            });
        };

        if uri == "nmr://workspace/policy" {
            let root_labels: Vec<String> = self
                .policy
                .allowed_roots()
                .iter()
                .enumerate()
                .map(|(idx, p)| {
                    let base = p
                        .file_name()
                        .and_then(|s| s.to_str())
                        .unwrap_or("workspace");
                    format!("root[{idx}]:{base}")
                })
                .collect();
            let text = serde_json::to_string_pretty(&json!({
                "access_mode": self.policy.access_mode().as_str(),
                "allowed_roots_count": self.policy.allowed_roots().len(),
                "allowed_root_labels": root_labels,
            }))
            .unwrap_or_default();
            return Ok(json!({
                "contents": [{
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": text,
                }]
            }));
        }

        if uri == "nmr://catalog/experiments" {
            let mut rules_map = Map::new();
            for &exp in &EXPERIMENT_NAMES {
                if let Some(rules) = experiment_rules(exp) {
                    let entries: Vec<Value> = rules
                        .iter()
                        .map(|&(atom, offset, sign)| {
                            json!({ "atom": atom, "residue_offset": offset, "sign": sign })
                        })
                        .collect();
                    rules_map.insert(exp.to_string(), Value::Array(entries));
                }
            }
            let text = serde_json::to_string_pretty(&json!({
                "backbone_experiments": EXPERIMENT_NAMES,
                "all_supported_experiments": ALL_EXPERIMENT_NAMES,
                "triple_resonance_rules": rules_map,
                "solvers": ["standard", "evo-graph", "bayes-sa"],
                "problem_regimes": ["globular-triple-resonance", "idp-13c", "paramagnetic", "methyl-trosy"],
            }))
            .unwrap_or_default();
            return Ok(json!({
                "contents": [{
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": text,
                }]
            }));
        }

        if uri == "nmr://workspace/inventory" {
            let Some(primary_root) = self.policy.allowed_roots().first() else {
                return Err(McpSecurityError::new(
                    McpSecurityErrorKind::NoAllowedRoots,
                    "cannot read nmr://workspace/inventory without an allowed workspace root",
                )
                .to_json_rpc_error());
            };
            let inv = classify_nmr_workspace(primary_root).map_err(|e| JsonRpcError {
                code: ERR_INTERNAL_ERROR,
                message: e,
                data: None,
            })?;
            let text = serde_json::to_string_pretty(&inv).unwrap_or_default();
            return Ok(json!({
                "contents": [{
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": text,
                }]
            }));
        }

        if let Some(rest) = uri.strip_prefix("nmr://file/") {
            let (root_idx, rel_path) = match rest.split_once('/') {
                Some((idx_str, rel)) => {
                    let idx = idx_str.parse::<usize>().map_err(|_| JsonRpcError {
                        code: ERR_INVALID_PARAMS,
                        message: format!("invalid root index in resource URI '{uri}'"),
                        data: None,
                    })?;
                    (idx, rel)
                }
                None => (0usize, rest),
            };
            let Some(root) = self.policy.allowed_roots().get(root_idx) else {
                return Err(JsonRpcError {
                    code: ERR_RESOURCE_NOT_FOUND,
                    message: format!("workspace root index {root_idx} does not exist"),
                    data: None,
                });
            };
            // Validate lexical & credential rules on `rel_path` first, then resolve inside `root`.
            validate_lexical_components(Path::new(rel_path)).map_err(|e| e.to_json_rpc_error())?;
            let single_root_policy = McpSecurityPolicy {
                allowed_roots: vec![root.clone()],
                access_mode: McpAccessMode::ReadOnly,
                hide_write_tools_in_read_only: false,
            };
            let resolved = single_root_policy
                .resolve_sandboxed_path(rel_path)
                .map_err(|e| e.to_json_rpc_error())?;
            let meta = fs::metadata(&resolved).map_err(|e| JsonRpcError {
                code: ERR_RESOURCE_NOT_FOUND,
                message: format!("failed to stat resource: {e}"),
                data: None,
            })?;
            if !meta.is_file() {
                return Err(JsonRpcError {
                    code: ERR_INVALID_PARAMS,
                    message: "requested resource URI points to a directory, not a file".to_string(),
                    data: None,
                });
            }
            if meta.len() > MAX_RESOURCE_READ_BYTES {
                return Err(JsonRpcError {
                    code: ERR_INVALID_PARAMS,
                    message: format!(
                        "resource exceeds maximum read size ({} > {MAX_RESOURCE_READ_BYTES} bytes)",
                        meta.len()
                    ),
                    data: None,
                });
            }
            let content = fs::read_to_string(&resolved).map_err(|e| JsonRpcError {
                code: ERR_INTERNAL_ERROR,
                message: format!("failed to read UTF-8 resource text: {e}"),
                data: None,
            })?;
            let mime = if rel_path.ends_with(".tsv") {
                "text/tab-separated-values"
            } else {
                "text/plain"
            };
            return Ok(json!({
                "contents": [{
                    "uri": uri,
                    "mimeType": mime,
                    "text": content,
                }]
            }));
        }

        Err(JsonRpcError {
            code: ERR_RESOURCE_NOT_FOUND,
            message: format!("unknown resource URI '{uri}'"),
            data: None,
        })
    }

    fn rpc_prompts_list(&self) -> Result<Value, JsonRpcError> {
        Ok(json!({ "prompts": builtin_prompt_descriptors() }))
    }

    fn rpc_prompts_get(&self, params: Option<Value>) -> Result<Value, JsonRpcError> {
        let obj = require_params_object(params, "prompts/get")?;
        let Some(name) = obj.get("name").and_then(Value::as_str) else {
            return Err(JsonRpcError {
                code: ERR_INVALID_PARAMS,
                message: "prompts/get requires string parameter 'name'".to_string(),
                data: None,
            });
        };
        let args: Map<String, Value> = match obj.get("arguments") {
            None | Some(Value::Null) => Map::new(),
            Some(Value::Object(m)) => m.clone(),
            Some(_) => {
                return Err(JsonRpcError {
                    code: ERR_INVALID_PARAMS,
                    message: "prompts/get 'arguments' must be a JSON object".to_string(),
                    data: None,
                });
            }
        };

        let prompt_result = build_spectroscopist_prompt(name, &args)?;
        serde_json::to_value(prompt_result).map_err(|e| JsonRpcError {
            code: ERR_INTERNAL_ERROR,
            message: format!("failed to serialize prompt result: {e}"),
            data: None,
        })
    }

    // ========================================================================
    // Built-In NMR Tool Implementations (`T15.1` & `T15.3`)
    // ========================================================================

    fn tool_inspect_nmr_directory(&self, args: &Map<String, Value>) -> McpCallToolResult {
        let raw_path = args.get("path").and_then(Value::as_str).unwrap_or(".");
        let decode_pulprog = args
            .get("decode_pulprog")
            .and_then(Value::as_bool)
            .unwrap_or(true);

        let resolved = match self.policy.resolve_sandboxed_path(raw_path) {
            Ok(p) => p,
            Err(sec) => return sec.to_tool_error_result(),
        };

        let inventory = match classify_nmr_workspace(&resolved) {
            Ok(inv) => inv,
            Err(e) => return McpCallToolResult::domain_error("classify_error", e),
        };

        let mut pulprog_reports = Vec::new();
        if decode_pulprog && resolved.is_dir() {
            if let Ok(interp) = interpret_experiment_dir(&resolved) {
                pulprog_reports.push(json!({
                    "relative_path": ".",
                    "vendor": interp.vendor,
                    "pulse_program": interp.pulse_program_name,
                    "dimensionality": interp.dimensionality,
                    "pseudo_3d_order": interp.pseudo_3d_order,
                    "pseudo_3d_list_kind": interp.pseudo_3d_list_kind,
                    "transfer_graph": interp.transfer_graph,
                    "warnings": interp.warnings,
                }));
            } else {
                for exp in inventory.experiments.iter().take(16) {
                    let sub = resolved.join(&exp.relative_path);
                    if sub.is_dir() {
                        if let Ok(interp) = interpret_experiment_dir(&sub) {
                            pulprog_reports.push(json!({
                                "relative_path": exp.relative_path,
                                "vendor": interp.vendor,
                                "pulse_program": interp.pulse_program_name,
                                "dimensionality": interp.dimensionality,
                                "pseudo_3d_order": interp.pseudo_3d_order,
                                "pseudo_3d_list_kind": interp.pseudo_3d_list_kind,
                                "deduced_family": interp.transfer_graph.deduced_family,
                                "warnings": interp.warnings,
                            }));
                        }
                    }
                }
            }
        }

        McpCallToolResult::ok_json(&json!({
            "workspace_path": self.policy.relativize_display_path(&resolved),
            "root_name": inventory.root_name,
            "problem_regime": inventory.problem_regime,
            "secondary_regimes": inventory.secondary_regimes,
            "experiment_count": inventory.experiments.len(),
            "experiments": inventory.experiments,
            "auxiliary_files": inventory.auxiliary_files,
            "missing_input_hints": inventory.missing_input_hints,
            "pulprog_interpretations": pulprog_reports,
        }))
    }

    fn tool_read_nmrstar_summary(&self, args: &Map<String, Value>) -> McpCallToolResult {
        let Some(raw_path) = args.get("path").and_then(Value::as_str) else {
            return McpCallToolResult::domain_error(
                "invalid_arguments",
                "missing required string argument 'path'",
            );
        };
        let entity = args.get("entity").and_then(Value::as_str);

        let resolved = match self.policy.resolve_sandboxed_path(raw_path) {
            Ok(p) => p,
            Err(sec) => return sec.to_tool_error_result(),
        };

        let ref_data = match read_star_all_atoms(&resolved, entity) {
            Ok(r) => r,
            Err(e) => return McpCallToolResult::domain_error("nmrstar_parse_error", e),
        };

        let n_res = ref_data.sequence.len();
        let n_prolines = ref_data.sequence.iter().filter(|r| *r == "PRO").count();
        let n_assignable_amides = (1..n_res)
            .filter(|&i| ref_data.sequence[i] != "PRO")
            .count();

        let mut per_atom_counts: BTreeMap<String, usize> = BTreeMap::new();
        let mut backbone_shifts = 0usize;
        let mut sidechain_shifts = 0usize;
        let mut residues_with_bb = vec![false; n_res];

        for &(res_idx, ref atom) in ref_data.shifts.keys() {
            *per_atom_counts.entry(atom.clone()).or_insert(0) += 1;
            if ATOMS.contains(&atom.as_str()) {
                backbone_shifts += 1;
                if res_idx < n_res {
                    residues_with_bb[res_idx] = true;
                }
            } else {
                sidechain_shifts += 1;
            }
        }

        let mut ambiguity_counts: BTreeMap<String, usize> = BTreeMap::new();
        for &code in ref_data.ambiguity_codes.values() {
            *ambiguity_counts.entry(code.to_string()).or_insert(0) += 1;
        }

        let n_res_with_bb = residues_with_bb.iter().filter(|&&b| b).count();
        let bb_completeness = if n_res > 0 {
            (n_res_with_bb as f64) / (n_res as f64)
        } else {
            0.0
        };
        let seq_id_range = match (ref_data.seq_ids.first(), ref_data.seq_ids.last()) {
            (Some(&lo), Some(&hi)) => json!([lo, hi]),
            _ => Value::Null,
        };

        McpCallToolResult::ok_json(&json!({
            "path": self.policy.relativize_display_path(&resolved),
            "sequence_length": n_res,
            "one_letter_sequence": ref_data.one_letter(),
            "seq_id_range": seq_id_range,
            "n_prolines": n_prolines,
            "n_assignable_amides": n_assignable_amides,
            "total_shifts": ref_data.shifts.len(),
            "backbone_shifts": backbone_shifts,
            "sidechain_shifts": sidechain_shifts,
            "residues_with_backbone_shifts": n_res_with_bb,
            "backbone_completeness_fraction": bb_completeness,
            "per_atom_counts": per_atom_counts,
            "ambiguity_code_counts": ambiguity_counts,
        }))
    }

    fn tool_read_peak_list_summary(&self, args: &Map<String, Value>) -> McpCallToolResult {
        let Some(raw_path) = args.get("path").and_then(Value::as_str) else {
            return McpCallToolResult::domain_error(
                "invalid_arguments",
                "missing required string argument 'path'",
            );
        };

        let resolved = match self.policy.resolve_sandboxed_path(raw_path) {
            Ok(p) => p,
            Err(sec) => return sec.to_tool_error_result(),
        };

        let peaks = match read_peaks(&resolved) {
            Ok(p) => p,
            Err(e) => return McpCallToolResult::domain_error("peak_parse_error", e),
        };

        #[derive(Default)]
        struct ExpAcc {
            count: usize,
            pos: usize,
            neg: usize,
            with_c: usize,
            with_h2: usize,
            with_height: usize,
            h_min: Option<f64>,
            h_max: Option<f64>,
            n_min: Option<f64>,
            n_max: Option<f64>,
            c_min: Option<f64>,
            c_max: Option<f64>,
        }

        let mut by_exp: BTreeMap<String, ExpAcc> = BTreeMap::new();
        for p in &peaks {
            let acc = by_exp.entry(p.exp.clone()).or_default();
            acc.count += 1;
            if p.sign >= 0 {
                acc.pos += 1;
            } else {
                acc.neg += 1;
            }
            acc.h_min = Some(acc.h_min.map_or(p.h, |m| m.min(p.h)));
            acc.h_max = Some(acc.h_max.map_or(p.h, |m| m.max(p.h)));
            acc.n_min = Some(acc.n_min.map_or(p.n, |m| m.min(p.n)));
            acc.n_max = Some(acc.n_max.map_or(p.n, |m| m.max(p.n)));
            if let Some(c) = p.c {
                acc.with_c += 1;
                acc.c_min = Some(acc.c_min.map_or(c, |m| m.min(c)));
                acc.c_max = Some(acc.c_max.map_or(c, |m| m.max(c)));
            }
            if p.h2.is_some() {
                acc.with_h2 += 1;
            }
            if p.height.is_some() || p.volume.is_some() {
                acc.with_height += 1;
            }
        }

        let mut exp_summaries = Map::new();
        for (exp, acc) in &by_exp {
            let dims = if acc.with_c > 0 && acc.with_h2 > 0 {
                3
            } else if acc.with_c > 0 || acc.with_h2 > 0 {
                if exp == "C13HSQC" || exp == "AROM_C13HSQC" {
                    2
                } else {
                    3
                }
            } else {
                2
            };
            exp_summaries.insert(
                exp.clone(),
                json!({
                    "count": acc.count,
                    "dimensions": dims,
                    "positive_peaks": acc.pos,
                    "negative_peaks": acc.neg,
                    "has_intensities": acc.with_height > 0,
                    "has_h2": acc.with_h2 > 0,
                    "h_ppm_range": [acc.h_min, acc.h_max],
                    "n_ppm_range": [acc.n_min, acc.n_max],
                    "c_ppm_range": [acc.c_min, acc.c_max],
                }),
            );
        }

        let has_hsqc = by_exp.contains_key("HSQC");
        let has_3d_triple = EXPERIMENT_NAMES
            .iter()
            .skip(1)
            .any(|e| by_exp.contains_key(*e));
        let spin_system_count = if has_hsqc {
            build_spin_systems(&peaks, &Tolerances::default(), 4, false)
                .ok()
                .map(|s| s.len())
        } else {
            None
        };

        McpCallToolResult::ok_json(&json!({
            "path": self.policy.relativize_display_path(&resolved),
            "total_peaks": peaks.len(),
            "experiment_count": by_exp.len(),
            "experiments": exp_summaries,
            "is_backbone_ready": has_hsqc && has_3d_triple,
            "assembled_spin_systems": spin_system_count,
        }))
    }

    fn tool_evaluate_dataset_qc(&self, args: &Map<String, Value>) -> McpCallToolResult {
        let Some(raw_peaks_path) = args.get("peaks_path").and_then(Value::as_str) else {
            return McpCallToolResult::domain_error(
                "invalid_arguments",
                "missing required string argument 'peaks_path'",
            );
        };
        let resolved_peaks = match self.policy.resolve_sandboxed_path(raw_peaks_path) {
            Ok(p) => p,
            Err(sec) => return sec.to_tool_error_result(),
        };
        let peaks = match read_peaks(&resolved_peaks) {
            Ok(p) => p,
            Err(e) => return McpCallToolResult::domain_error("peak_parse_error", e),
        };

        let seq3: Vec<String> = if let Some(seq_text) = args.get("sequence").and_then(Value::as_str)
        {
            match parse_sequence_to_three_letter(seq_text) {
                Ok(s) => s,
                Err(e) => return McpCallToolResult::domain_error("sequence_parse_error", e),
            }
        } else if let Some(star_raw) = args.get("star_path").and_then(Value::as_str) {
            let resolved_star = match self.policy.resolve_sandboxed_path(star_raw) {
                Ok(p) => p,
                Err(sec) => return sec.to_tool_error_result(),
            };
            match read_star_all_atoms(&resolved_star, None) {
                Ok(r) => r.sequence,
                Err(e) => return McpCallToolResult::domain_error("nmrstar_parse_error", e),
            }
        } else {
            return McpCallToolResult::domain_error(
                "invalid_arguments",
                "evaluate_dataset_qc requires either 'sequence' or 'star_path'",
            );
        };

        let systems = match build_spin_systems(&peaks, &Tolerances::default(), 3, false) {
            Ok(s) => s,
            Err(e) => return McpCallToolResult::domain_error("spin_system_error", e),
        };

        let seed = args.get("seed").and_then(Value::as_u64).unwrap_or(0);
        let priors = Priors::bmrb();
        let score_cfg = ScoreConfig::default();
        let base_detailed = estimate_reference_offsets_detailed(&systems, &seq3, &priors);
        let (_, offsets) = prepare_scoring_priors(&systems, &seq3, &priors, &score_cfg);
        let detailed_offsets = sync_detailed_reference_offsets(&offsets, &base_detailed);
        let anneal_cfg = AnnealConfig {
            runs: 16,
            steps_per_system: 1500,
            seed,
            ..AnnealConfig::default()
        };

        let mut assign_res = assign(&systems, &seq3, &priors, &score_cfg, &anneal_cfg);
        let qc = evaluate_dataset_qc(
            Some(&peaks),
            &systems,
            &seq3,
            &priors,
            &score_cfg,
            &assign_res,
            Some(&detailed_offsets),
        );
        apply_dataset_qc_to_assignments(&mut assign_res, &qc);

        // Summarize intensity / SNR proxy and root-to-3D linewidth/registration drift.
        let mut heights: Vec<f64> = peaks
            .iter()
            .filter_map(|p| p.height.map(f64::abs))
            .filter(|&h| h > 0.0 && h.is_finite())
            .collect();
        heights.sort_by(|a, b| a.total_cmp(b));
        let snr_proxy = if heights.len() >= 4 {
            let p10 = heights[heights.len() / 10];
            let p50 = heights[heights.len() / 2];
            let p90 = heights[(heights.len() * 9) / 10];
            Some(json!({
                "peaks_with_intensity": heights.len(),
                "p10_height": p10,
                "median_height": p50,
                "p90_height": p90,
                "dynamic_range_p90_over_p10": if p10 > 0.0 { p90 / p10 } else { 0.0 },
            }))
        } else {
            None
        };

        let n_assigned = assign_res.position.iter().filter(|&&k| k >= 0).count();
        let n_high_conf = assign_res
            .confidence
            .iter()
            .zip(assign_res.position.iter())
            .filter(|&(&c, &k)| k >= 0 && c >= 0.80)
            .count();

        McpCallToolResult::ok_json(&json!({
            "peaks_path": self.policy.relativize_display_path(&resolved_peaks),
            "verdict": qc.verdict.as_str(),
            "confidence_scale": qc.confidence_scale,
            "confidence_cap": qc.confidence_cap,
            "protein_hc_gate_triggered": qc.protein_hc_gate_triggered,
            "warn_sub_hc_uncalibrated": qc.warn_sub_hc_uncalibrated,
            "completeness": {
                "n_sequence_residues": seq3.len(),
                "n_assignable_amides": qc.n_assignable_amides,
                "n_amide_hsqc_roots": qc.n_amide_hsqc_roots,
                "hsqc_3d_support_frac": qc.hsqc_3d_support_frac,
                "supported_roots_per_assignable": qc.supported_roots_per_assignable,
                "aliphatic_support_frac": qc.aliphatic_support_frac,
                "resolved_bidir_aliphatic_frac": qc.resolved_bidir_aliphatic_frac,
                "connectivity_frac": qc.connectivity_frac,
                "corroborated_connectivity_frac": qc.corroborated_connectivity_frac,
                "n_assigned": n_assigned,
                "n_high_confidence_ge_0_80": n_high_conf,
            },
            "linewidth_and_registration": {
                "hsqc_3d_rms_dh_ppm": qc.hsqc_3d_rms_dh_ppm,
                "hsqc_3d_rms_dn_ppm": qc.hsqc_3d_rms_dn_ppm,
                "hsqc_multiplicity_ratio": qc.hsqc_multiplicity_ratio,
                "offset_consistency_ok": qc.offset_consistency_ok,
            },
            "intensity_snr_summary": snr_proxy,
            "gate_warnings": qc.reasons,
            "qc_report": qc,
        }))
    }

    fn tool_predict_homolog_free_torsions(&self, args: &Map<String, Value>) -> McpCallToolResult {
        let Some(raw_star_path) = args.get("star_path").and_then(Value::as_str) else {
            return McpCallToolResult::domain_error(
                "invalid_arguments",
                "missing required string argument 'star_path'",
            );
        };
        let resolved_star = match self.policy.resolve_sandboxed_path(raw_star_path) {
            Ok(p) => p,
            Err(sec) => return sec.to_tool_error_result(),
        };
        let shifts = match read_star_all_atoms(&resolved_star, None) {
            Ok(r) => r,
            Err(e) => return McpCallToolResult::domain_error("nmrstar_parse_error", e),
        };

        let parse_str_list = |key: &str| -> Vec<String> {
            args.get(key)
                .and_then(Value::as_array)
                .map(|arr| {
                    arr.iter()
                        .filter_map(Value::as_str)
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default()
        };

        let query = HomologFirewallQuery {
            query_pdb_ids: parse_str_list("exclude_pdb_ids"),
            query_bmrb_ids: parse_str_list("exclude_bmrb_ids"),
            query_uniprot_id: args
                .get("exclude_uniprot_id")
                .and_then(Value::as_str)
                .map(str::to_string),
            query_family_id: args
                .get("exclude_family_id")
                .and_then(Value::as_str)
                .map(str::to_string),
            query_topology_id: args
                .get("exclude_topology_id")
                .and_then(Value::as_str)
                .map(str::to_string),
            query_ca_coords: None,
        };

        // Never permit relaxing the <= 25% HomologFirewall cutoff via MCP arguments.
        let requested_cutoff = args
            .get("max_seq_identity")
            .and_then(Value::as_f64)
            .unwrap_or(DEFAULT_FIREWALL_MAX_SEQ_IDENTITY);
        if requested_cutoff > DEFAULT_FIREWALL_MAX_SEQ_IDENTITY + 1e-9 {
            return McpCallToolResult::domain_error(
                "homolog_firewall_violation",
                format!(
                    "requested max_seq_identity ({requested_cutoff:.3}) exceeds mandatory HomologFirewall ceiling ({DEFAULT_FIREWALL_MAX_SEQ_IDENTITY:.2})"
                ),
            );
        }
        let max_seq_id = requested_cutoff.clamp(0.05, DEFAULT_FIREWALL_MAX_SEQ_IDENTITY);
        let include_chi1 = args
            .get("include_chi1")
            .and_then(Value::as_bool)
            .unwrap_or(false);

        let firewall = HomologFirewall::new(query).with_max_seq_identity(max_seq_id);
        let mut report = firewall.predict(&shifts);
        if !include_chi1 {
            for r in &mut report.residues {
                r.chi1_deg = None;
                r.chi1_std_deg = None;
            }
        }

        let n_good = report
            .residues
            .iter()
            .filter(|r| r.class == TorsionPredictionClass::Good)
            .count();
        let n_dynamic = report
            .residues
            .iter()
            .filter(|r| r.class == TorsionPredictionClass::Dynamic)
            .count();
        let n_ambiguous = report
            .residues
            .iter()
            .filter(|r| r.class == TorsionPredictionClass::Ambiguous)
            .count();
        let n_unassigned = report
            .residues
            .iter()
            .filter(|r| r.class == TorsionPredictionClass::Unassigned)
            .count();

        McpCallToolResult::ok_json(&json!({
            "star_path": self.policy.relativize_display_path(&resolved_star),
            "n_residues": report.residues.len(),
            "n_good": n_good,
            "n_dynamic": n_dynamic,
            "n_ambiguous": n_ambiguous,
            "n_unassigned": n_unassigned,
            "include_chi1": include_chi1,
            "enforced_max_seq_identity": max_seq_id,
            "max_retained_seq_identity": report.max_retained_seq_identity,
            "excluded_templates_count": report.excluded_template_ids.len(),
            "exclusion_list_sha256": report.exclusion_list_sha256,
            "shift_offsets_ppm": report.shift_offsets,
            "residues": report.residues,
        }))
    }

    fn tool_write_curated_peak_tsv(&self, args: &Map<String, Value>) -> McpCallToolResult {
        if let Err(sec) = self.policy.require_write_access("write_curated_peak_tsv") {
            return sec.to_tool_error_result();
        }
        let Some(raw_out) = args.get("output_path").and_then(Value::as_str) else {
            return McpCallToolResult::domain_error(
                "invalid_arguments",
                "missing required string argument 'output_path'",
            );
        };
        let resolved_out = match self
            .policy
            .resolve_sandboxed_write_path(raw_out, "write_curated_peak_tsv")
        {
            Ok(p) => p,
            Err(sec) => return sec.to_tool_error_result(),
        };

        let source_peaks: Vec<Peak> =
            if let Some(tsv_text) = args.get("peaks_tsv").and_then(Value::as_str) {
                match read_peaks_str(tsv_text) {
                    Ok(p) => p,
                    Err(e) => return McpCallToolResult::domain_error("peak_parse_error", e),
                }
            } else if let Some(src_path) = args.get("source_peaks_path").and_then(Value::as_str) {
                let resolved_src = match self.policy.resolve_sandboxed_path(src_path) {
                    Ok(p) => p,
                    Err(sec) => return sec.to_tool_error_result(),
                };
                match read_peaks(&resolved_src) {
                    Ok(p) => p,
                    Err(e) => return McpCallToolResult::domain_error("peak_parse_error", e),
                }
            } else {
                return McpCallToolResult::domain_error(
                    "invalid_arguments",
                    "write_curated_peak_tsv requires 'peaks_tsv' or 'source_peaks_path'",
                );
            };

        let excluded_indices: Vec<usize> = args
            .get("exclude_indices")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(Value::as_u64)
                    .map(|v| v as usize)
                    .collect()
            })
            .unwrap_or_default();

        let allowed_exps: Option<Vec<String>> = args
            .get("filter_experiments")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(Value::as_str)
                    .map(|s| s.trim().to_ascii_uppercase())
                    .collect()
            });

        let total_before = source_peaks.len();
        let curated: Vec<Peak> = source_peaks
            .into_iter()
            .enumerate()
            .filter(|(idx, p)| {
                !excluded_indices.contains(idx)
                    && allowed_exps
                        .as_ref()
                        .is_none_or(|exps| exps.contains(&p.exp))
            })
            .map(|(_, p)| p)
            .collect();

        if let Some(parent) = resolved_out.parent() {
            if let Err(e) = fs::create_dir_all(parent) {
                return McpCallToolResult::domain_error(
                    "io_error",
                    format!("failed to create output directory: {e}"),
                );
            }
        }
        if let Err(e) = write_peaks(&resolved_out, &curated) {
            return McpCallToolResult::domain_error("io_error", e);
        }
        let written_bytes = fs::read(&resolved_out).unwrap_or_default();

        McpCallToolResult::ok_json(&json!({
            "output_path": self.policy.relativize_display_path(&resolved_out),
            "source_peaks": total_before,
            "written_peaks": curated.len(),
            "excluded_peaks": total_before.saturating_sub(curated.len()),
            "sha256": sha256_hex(&written_bytes),
        }))
    }

    fn tool_export_nmrstar_shifts(&self, args: &Map<String, Value>) -> McpCallToolResult {
        if let Err(sec) = self.policy.require_write_access("export_nmrstar_shifts") {
            return sec.to_tool_error_result();
        }
        let Some(raw_out) = args.get("output_path").and_then(Value::as_str) else {
            return McpCallToolResult::domain_error(
                "invalid_arguments",
                "missing required string argument 'output_path'",
            );
        };
        let resolved_out = match self
            .policy
            .resolve_sandboxed_write_path(raw_out, "export_nmrstar_shifts")
        {
            Ok(p) => p,
            Err(sec) => return sec.to_tool_error_result(),
        };

        let mut confidence_map: BTreeMap<usize, f64> = BTreeMap::new();
        let mut ref_data: Reference =
            if let Some(src_star) = args.get("source_star_path").and_then(Value::as_str) {
                let resolved_src = match self.policy.resolve_sandboxed_path(src_star) {
                    Ok(p) => p,
                    Err(sec) => return sec.to_tool_error_result(),
                };
                match read_star_all_atoms(&resolved_src, None) {
                    Ok(r) => r,
                    Err(e) => return McpCallToolResult::domain_error("nmrstar_parse_error", e),
                }
            } else if let Some(seq_str) = args.get("sequence").and_then(Value::as_str) {
                let seq3 = match parse_sequence_to_three_letter(seq_str) {
                    Ok(s) => s,
                    Err(e) => return McpCallToolResult::domain_error("sequence_parse_error", e),
                };
                let first_seq_id = args
                    .get("first_seq_id")
                    .and_then(Value::as_i64)
                    .unwrap_or(1) as i32;
                let seq_ids: Vec<i32> = (0..seq3.len()).map(|i| first_seq_id + i as i32).collect();
                Reference::new(seq3, seq_ids)
            } else {
                return McpCallToolResult::domain_error(
                    "invalid_arguments",
                    "export_nmrstar_shifts requires 'source_star_path' or 'sequence'",
                );
            };

        if let Some(shift_entries) = args.get("shifts").and_then(Value::as_array) {
            for entry in shift_entries {
                let Some(obj) = entry.as_object() else {
                    return McpCallToolResult::domain_error(
                        "invalid_arguments",
                        "each item in 'shifts' must be an object {seq_id, atom, val}",
                    );
                };
                let Some(seq_id_i64) = obj.get("seq_id").and_then(Value::as_i64) else {
                    return McpCallToolResult::domain_error(
                        "invalid_arguments",
                        "shift entry missing integer 'seq_id'",
                    );
                };
                let seq_id = seq_id_i64 as i32;
                let Some(atom) = obj.get("atom").and_then(Value::as_str) else {
                    return McpCallToolResult::domain_error(
                        "invalid_arguments",
                        "shift entry missing string 'atom'",
                    );
                };
                let Some(val) = obj
                    .get("val")
                    .and_then(Value::as_f64)
                    .filter(|v| v.is_finite())
                else {
                    return McpCallToolResult::domain_error(
                        "invalid_arguments",
                        "shift entry missing finite numeric 'val'",
                    );
                };
                let Some(res_idx) = ref_data.seq_ids.iter().position(|&s| s == seq_id) else {
                    return McpCallToolResult::domain_error(
                        "invalid_arguments",
                        format!("shift entry seq_id={seq_id} is outside sequence numbering"),
                    );
                };
                ref_data
                    .shifts
                    .insert((res_idx, atom.trim().to_ascii_uppercase()), val);
                if let Some(conf) = obj
                    .get("confidence")
                    .and_then(Value::as_f64)
                    .filter(|c| c.is_finite())
                {
                    confidence_map.insert(res_idx, conf.clamp(0.0, 1.0));
                }
            }
        }

        if let Some(parent) = resolved_out.parent() {
            if let Err(e) = fs::create_dir_all(parent) {
                return McpCallToolResult::domain_error(
                    "io_error",
                    format!("failed to create output directory: {e}"),
                );
            }
        }

        let header = args.get("header_comment").and_then(Value::as_str);
        let conf_opt = (!confidence_map.is_empty()).then_some(&confidence_map);
        if let Err(e) = write_star_with_header(&resolved_out, &ref_data, conf_opt, header) {
            return McpCallToolResult::domain_error("io_error", e);
        }
        let written_bytes = fs::read(&resolved_out).unwrap_or_default();

        McpCallToolResult::ok_json(&json!({
            "output_path": self.policy.relativize_display_path(&resolved_out),
            "n_residues": ref_data.sequence.len(),
            "n_shifts_written": ref_data.shifts.len(),
            "sha256": sha256_hex(&written_bytes),
        }))
    }

    fn collect_sandboxed_input_paths(
        &self,
        args: &Map<String, Value>,
    ) -> Result<Vec<PathBuf>, McpCallToolResult> {
        let mut raw_paths: Vec<String> = Vec::new();
        if let Some(p) = args.get("path").and_then(Value::as_str) {
            if !p.trim().is_empty() {
                raw_paths.push(p.to_string());
            }
        }
        if let Some(arr) = args.get("paths").and_then(Value::as_array) {
            for item in arr {
                let Some(s) = item.as_str() else {
                    return Err(McpCallToolResult::domain_error(
                        "invalid_arguments",
                        "every entry in 'paths' must be a string",
                    ));
                };
                if !s.trim().is_empty() {
                    raw_paths.push(s.to_string());
                }
            }
        }
        if raw_paths.is_empty() {
            return Err(McpCallToolResult::domain_error(
                "invalid_arguments",
                "requires non-empty string 'path' or array 'paths'",
            ));
        }
        let mut resolved = Vec::with_capacity(raw_paths.len());
        for raw in &raw_paths {
            match self.policy.resolve_sandboxed_path(raw) {
                Ok(p) => resolved.push(p),
                Err(sec) => return Err(sec.to_tool_error_result()),
            }
        }
        Ok(resolved)
    }

    fn tool_fit_relaxation_or_exchange(&self, args: &Map<String, Value>) -> McpCallToolResult {
        let Some(mode_raw) = args.get("mode").and_then(Value::as_str) else {
            return McpCallToolResult::domain_error(
                "invalid_arguments",
                "missing required string argument 'mode' ('relaxation', 'cest', or 'cpmg')",
            );
        };
        let resolved_paths = match self.collect_sandboxed_input_paths(args) {
            Ok(p) => p,
            Err(err_res) => return err_res,
        };
        let display_paths: Vec<String> = resolved_paths
            .iter()
            .map(|p| self.policy.relativize_display_path(p))
            .collect();

        let mode = mode_raw.trim().to_ascii_lowercase();
        match mode.as_str() {
            "relaxation" | "r1" | "r2" => {
                let first = &resolved_paths[0];
                let text = match fs::read_to_string(first) {
                    Ok(t) => t,
                    Err(e) => {
                        return McpCallToolResult::domain_error(
                            "io_error",
                            format!("failed to read relaxation series: {e}"),
                        );
                    }
                };
                let is_nef = first
                    .extension()
                    .and_then(|s| s.to_str())
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("nef"));
                let ds = if is_nef {
                    match parse_relax_series_nef_str(&text) {
                        Ok(d) => d,
                        Err(e) => {
                            return McpCallToolResult::domain_error("relax_parse_error", e);
                        }
                    }
                } else {
                    match parse_relax_series_tsv_str(&text) {
                        Ok(d) => d,
                        Err(e) => {
                            return McpCallToolResult::domain_error("relax_parse_error", e);
                        }
                    }
                };
                let mc_replicates = args
                    .get("mc_replicates")
                    .and_then(Value::as_u64)
                    .unwrap_or(200)
                    .min(2000) as usize;
                let seed = args
                    .get("seed")
                    .and_then(Value::as_u64)
                    .unwrap_or(20_261_302);
                let cfg = ExpFitConfig {
                    mc_replicates,
                    seed,
                    ..ExpFitConfig::default()
                };
                let inputs = decay_inputs_from_series(&ds);
                let fits = match fit_decay_batch_cpu(&inputs, &cfg) {
                    Ok(f) => f,
                    Err(e) => return McpCallToolResult::domain_error("relax_fit_error", e),
                };
                let mean_rate = if fits.is_empty() {
                    0.0
                } else {
                    fits.iter().map(|f| f.rate).sum::<f64>() / (fits.len() as f64)
                };
                let residue_rows: Vec<Value> = fits
                    .iter()
                    .map(|f| {
                        json!({
                            "seq_id": f.seq_id,
                            "res_name": f.res_name,
                            "rate_s_inv": f.rate,
                            "sigma_rate_fisher": f.sigma_rate_fisher,
                            "sigma_rate_mc": f.sigma_rate_mc,
                            "rate_ci95": f.rate_ci95,
                            "i0": f.i0,
                            "chi2": f.chi2,
                            "dof": f.dof,
                            "used_three_param": f.used_three_param,
                        })
                    })
                    .collect();

                McpCallToolResult::ok_json(&json!({
                    "mode": "relaxation",
                    "paths": display_paths,
                    "experiment_kind": format!("{:?}", ds.experiment_kind),
                    "field_mhz": ds.field_mhz,
                    "n_residues": fits.len(),
                    "mean_rate_s_inv": mean_rate,
                    "mc_replicates": mc_replicates,
                    "residues": residue_rows,
                }))
            }
            "cest" | "dest" => {
                let mut profiles = Vec::new();
                for p in &resolved_paths {
                    let text = match fs::read_to_string(p) {
                        Ok(t) => t,
                        Err(e) => {
                            return McpCallToolResult::domain_error(
                                "io_error",
                                format!("failed to read CEST profile TSV: {e}"),
                            );
                        }
                    };
                    match read_cest_profiles_tsv(&text) {
                        Ok(mut parsed) => profiles.append(&mut parsed),
                        Err(e) => return McpCallToolResult::domain_error("cest_parse_error", e),
                    }
                }
                let compute_chi2_surface = args
                    .get("compute_chi2_surface")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                let cfg = CestFitConfig {
                    compute_chi2_surface,
                    ..CestFitConfig::default()
                };
                let fit = match fit_cest_global_cpu(&profiles, &cfg) {
                    Ok(r) => r,
                    Err(e) => return McpCallToolResult::domain_error("cest_fit_error", e),
                };
                McpCallToolResult::ok_json(&json!({
                    "mode": "cest",
                    "paths": display_paths,
                    "n_profiles": profiles.len(),
                    "n_residues": fit.residues.len(),
                    "p_b": fit.p_b,
                    "p_b_err": fit.p_b_err,
                    "k_ex_s_inv": fit.k_ex_s_inv,
                    "k_ex_err": fit.k_ex_err,
                    "k_ab_s_inv": fit.k_ab_s_inv,
                    "k_ba_s_inv": fit.k_ba_s_inv,
                    "total_chi2": fit.total_chi2,
                    "reduced_chi2": fit.reduced_chi2,
                    "dof": fit.dof,
                    "converged": fit.converged,
                    "residues": fit.residues,
                }))
            }
            "cpmg" => {
                let mut series = Vec::new();
                for p in &resolved_paths {
                    let text = match fs::read_to_string(p) {
                        Ok(t) => t,
                        Err(e) => {
                            return McpCallToolResult::domain_error(
                                "io_error",
                                format!("failed to read CPMG series TSV: {e}"),
                            );
                        }
                    };
                    match read_cpmg_series_tsv(&text) {
                        Ok(mut parsed) => series.append(&mut parsed),
                        Err(e) => return McpCallToolResult::domain_error("cpmg_parse_error", e),
                    }
                }
                let model = match args.get("cpmg_model").and_then(Value::as_str) {
                    Some(lbl) => match CpmgModel::from_str_label(lbl) {
                        Ok(m) => m,
                        Err(e) => return McpCallToolResult::domain_error("invalid_arguments", e),
                    },
                    None => CpmgModel::Baldwin2014,
                };
                let cfg = CpmgFitConfig {
                    model,
                    ..CpmgFitConfig::default()
                };
                let fit = match fit_cpmg_global_cpu(&series, &cfg) {
                    Ok(r) => r,
                    Err(e) => return McpCallToolResult::domain_error("cpmg_fit_error", e),
                };
                McpCallToolResult::ok_json(&json!({
                    "mode": "cpmg",
                    "model": fit.model.as_str(),
                    "paths": display_paths,
                    "n_series": series.len(),
                    "n_residues": fit.residues.len(),
                    "p_b": fit.p_b,
                    "p_b_err": fit.p_b_err,
                    "k_ex_s_inv": fit.k_ex_s_inv,
                    "k_ex_err": fit.k_ex_err,
                    "k_ab_s_inv": fit.k_ab_s_inv,
                    "k_ba_s_inv": fit.k_ba_s_inv,
                    "total_chi2": fit.total_chi2,
                    "reduced_chi2": fit.reduced_chi2,
                    "dof": fit.dof,
                    "converged": fit.converged,
                    "residues": fit.residues,
                }))
            }
            other => McpCallToolResult::domain_error(
                "invalid_arguments",
                format!("unsupported mode '{other}' (expected 'relaxation', 'cest', or 'cpmg')"),
            ),
        }
    }

    fn tool_track_csp_titration(&self, args: &Map<String, Value>) -> McpCallToolResult {
        let nucleus_mode = args
            .get("nucleus_mode")
            .and_then(Value::as_str)
            .unwrap_or("amide")
            .trim()
            .to_ascii_lowercase();
        if nucleus_mode != "amide" && nucleus_mode != "methyl" {
            return McpCallToolResult::domain_error(
                "invalid_arguments",
                format!("unsupported nucleus_mode '{nucleus_mode}' (expected 'amide' or 'methyl')"),
            );
        }

        let apo_peaks: Vec<AssignedAmidePeak> =
            if let Some(raw_star) = args.get("apo_star_path").and_then(Value::as_str) {
                let resolved_star = match self.policy.resolve_sandboxed_path(raw_star) {
                    Ok(p) => p,
                    Err(sec) => return sec.to_tool_error_result(),
                };
                let ref_data = match read_star_all_atoms(&resolved_star, None) {
                    Ok(r) => r,
                    Err(e) => return McpCallToolResult::domain_error("nmrstar_parse_error", e),
                };
                let mut out = Vec::new();
                for (res_idx, res_name) in ref_data.sequence.iter().enumerate() {
                    if nucleus_mode == "amide" && res_name == "PRO" {
                        continue;
                    }
                    let seq_id = ref_data
                        .seq_ids
                        .get(res_idx)
                        .copied()
                        .unwrap_or((res_idx + 1) as i32);
                    let h_opt = ref_data
                        .shifts
                        .get(&(res_idx, "H".to_string()))
                        .or_else(|| ref_data.shifts.get(&(res_idx, "HN".to_string())))
                        .copied();
                    let het_opt = if nucleus_mode == "methyl" {
                        ref_data
                            .shifts
                            .get(&(res_idx, "CD1".to_string()))
                            .or_else(|| ref_data.shifts.get(&(res_idx, "CG1".to_string())))
                            .or_else(|| ref_data.shifts.get(&(res_idx, "CB".to_string())))
                            .copied()
                    } else {
                        ref_data.shifts.get(&(res_idx, "N".to_string())).copied()
                    };
                    if let (Some(h_ppm), Some(n_ppm)) = (h_opt, het_opt) {
                        out.push(AssignedAmidePeak {
                            seq_index: res_idx,
                            seq_id,
                            res_name: res_name.clone(),
                            h_ppm,
                            n_ppm,
                        });
                    }
                }
                out
            } else if let Some(raw_apo_tsv) = args.get("apo_peaks_path").and_then(Value::as_str) {
                let resolved_apo = match self.policy.resolve_sandboxed_path(raw_apo_tsv) {
                    Ok(p) => p,
                    Err(sec) => return sec.to_tool_error_result(),
                };
                let text = match fs::read_to_string(&resolved_apo) {
                    Ok(t) => t,
                    Err(e) => {
                        return McpCallToolResult::domain_error(
                            "io_error",
                            format!("failed to read apo peaks TSV: {e}"),
                        );
                    }
                };
                match parse_assigned_apo_peaks_tsv(&text) {
                    Ok(p) => p,
                    Err(e) => return McpCallToolResult::domain_error("apo_parse_error", e),
                }
            } else {
                return McpCallToolResult::domain_error(
                    "invalid_arguments",
                    "track_csp_titration requires either 'apo_star_path' or 'apo_peaks_path'",
                );
            };

        if apo_peaks.is_empty() {
            return McpCallToolResult::domain_error(
                "invalid_arguments",
                "zero assigned apo peaks extracted from reference file",
            );
        }

        let Some(steps_arr) = args.get("steps").and_then(Value::as_array) else {
            return McpCallToolResult::domain_error(
                "invalid_arguments",
                "track_csp_titration requires non-empty array 'steps'",
            );
        };
        if steps_arr.is_empty() {
            return McpCallToolResult::domain_error(
                "invalid_arguments",
                "track_csp_titration 'steps' array must contain at least one titration point",
            );
        }

        let mut steps = Vec::with_capacity(steps_arr.len());
        for (idx, item) in steps_arr.iter().enumerate() {
            let Some(obj) = item.as_object() else {
                return McpCallToolResult::domain_error(
                    "invalid_arguments",
                    format!("steps[{idx}] must be a JSON object"),
                );
            };
            let Some(raw_step_path) = obj.get("peaks_path").and_then(Value::as_str) else {
                return McpCallToolResult::domain_error(
                    "invalid_arguments",
                    format!("steps[{idx}] requires string 'peaks_path'"),
                );
            };
            let resolved_step = match self.policy.resolve_sandboxed_path(raw_step_path) {
                Ok(p) => p,
                Err(sec) => return sec.to_tool_error_result(),
            };
            let Some(protein_conc_um) = obj
                .get("protein_conc_um")
                .and_then(Value::as_f64)
                .filter(|v| v.is_finite() && *v > 0.0)
            else {
                return McpCallToolResult::domain_error(
                    "invalid_arguments",
                    format!("steps[{idx}] requires positive finite 'protein_conc_um'"),
                );
            };
            let Some(ligand_conc_um) = obj
                .get("ligand_conc_um")
                .and_then(Value::as_f64)
                .filter(|v| v.is_finite() && *v >= 0.0)
            else {
                return McpCallToolResult::domain_error(
                    "invalid_arguments",
                    format!("steps[{idx}] requires non-negative finite 'ligand_conc_um'"),
                );
            };
            let label = obj
                .get("label")
                .and_then(Value::as_str)
                .map(str::to_string)
                .unwrap_or_else(|| format!("step_{}", idx + 1));

            let step_peaks_raw = match read_peaks(&resolved_step) {
                Ok(pk) => pk,
                Err(e) => return McpCallToolResult::domain_error("peak_parse_error", e),
            };
            let step_peaks: Vec<UnassignedHsqcPeak> = step_peaks_raw
                .iter()
                .map(|p| UnassignedHsqcPeak {
                    h_ppm: p.h,
                    n_ppm: p.c.unwrap_or(p.n),
                })
                .collect();
            steps.push(UnassignedTitrationStep {
                label,
                protein_conc_um,
                ligand_conc_um,
                peaks: step_peaks,
            });
        }

        let max_step_csp_ppm = args
            .get("max_step_csp_ppm")
            .and_then(Value::as_f64)
            .filter(|v| v.is_finite() && *v > 0.0)
            .unwrap_or(0.45);
        let n_mc = args
            .get("n_mc")
            .and_then(Value::as_u64)
            .unwrap_or(200)
            .min(2000) as usize;
        let mc_seed = args
            .get("seed")
            .and_then(Value::as_u64)
            .unwrap_or(20_261_306);
        let cfg = CspAnalysisConfig {
            max_step_csp_ppm,
            n_mc,
            mc_seed,
            ..CspAnalysisConfig::default()
        };

        let trajectories = if nucleus_mode == "methyl" {
            match track_methyl_hmqc_trajectories(&apo_peaks, &steps, &cfg) {
                Ok(t) => t,
                Err(e) => return McpCallToolResult::domain_error("csp_track_error", e),
            }
        } else {
            match track_titration_trajectories(&apo_peaks, &steps, &cfg) {
                Ok(t) => t,
                Err(e) => return McpCallToolResult::domain_error("csp_track_error", e),
            }
        };

        let fit = match fit_csp_global_kd(&trajectories, &cfg) {
            Ok(f) => f,
            Err(e) => return McpCallToolResult::domain_error("csp_fit_error", e),
        };

        McpCallToolResult::ok_json(&json!({
            "nucleus_mode": nucleus_mode,
            "n_apo_peaks": apo_peaks.len(),
            "n_steps": steps.len(),
            "global_kd_um": fit.global_kd_um,
            "global_kd_sd_um": fit.global_kd_sd_um,
            "global_kd_ci95_um": [fit.global_kd_ci95_low_um, fit.global_kd_ci95_high_um],
            "global_naive_mm_kd_um": fit.global_naive_mm_kd_um,
            "global_reduced_chi2": fit.global_reduced_chi2,
            "thresholds": fit.thresholds,
            "residue_fits": fit.residue_fits,
        }))
    }

    fn tool_analyze_nmr1d_spectrum(&self, args: &Map<String, Value>) -> McpCallToolResult {
        let Some(raw_path) = args.get("path").and_then(Value::as_str) else {
            return McpCallToolResult::domain_error(
                "invalid_arguments",
                "missing required string argument 'path'",
            );
        };
        let resolved = match self.policy.resolve_sandboxed_path(raw_path) {
            Ok(p) => p,
            Err(sec) => return sec.to_tool_error_result(),
        };
        let mhz_override = args
            .get("mhz")
            .and_then(Value::as_f64)
            .filter(|v| v.is_finite() && *v > 0.0);
        let mut spec = match load_processed_spectrum_1d(&resolved, mhz_override) {
            Ok(s) => s,
            Err(e) => return McpCallToolResult::domain_error("nmr1d_load_error", e),
        };

        let solvent_opt = args
            .get("solvent")
            .and_then(Value::as_str)
            .map(str::to_string)
            .or_else(|| spec.solvent.clone());
        let mut calibration_offset_ppm: Option<f64> = None;
        if args
            .get("calibrate_solvent")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            if let Some(ref solv) = solvent_opt {
                match auto_calibrate_solvent_1d(&mut spec, solv, 0.25) {
                    Ok(off) => calibration_offset_ppm = Some(off),
                    Err(e) => {
                        return McpCallToolResult::domain_error("solvent_calibration_error", e);
                    }
                }
            }
        }

        let noise_sigma = estimate_noise_sigma_1d(&spec);
        let min_snr = args
            .get("min_snr")
            .and_then(Value::as_f64)
            .filter(|v| v.is_finite() && *v > 0.0)
            .unwrap_or(6.0);
        let pick_cfg = PeakPickConfig1D {
            min_snr,
            ..PeakPickConfig1D::default()
        };
        let picked = pick_peaks_1d(&spec, &pick_cfg);

        let deconvolve = args
            .get("deconvolve")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        let deconv_opt = if deconvolve {
            let dcfg = DeconvolveConfig1D {
                pick_config: pick_cfg,
                ..DeconvolveConfig1D::default()
            };
            Some(deconvolve_spectrum_1d(&spec, &dcfg))
        } else {
            None
        };

        let match_imp = args
            .get("match_impurities")
            .and_then(Value::as_bool)
            .unwrap_or(solvent_opt.is_some());
        let mut impurity_hits = Vec::new();
        if match_imp {
            if let Some(ref solv) = solvent_opt {
                let nuc = spec.experiment.nucleus();
                for pk in picked.iter().take(64) {
                    for m in match_trace_impurities(solv, &nuc, pk.ppm, 0.03) {
                        impurity_hits.push(json!({
                            "observed_ppm": pk.ppm,
                            "compound": m.entry.compound,
                            "site": m.entry.site,
                            "reference_ppm": m.entry.ppm,
                            "delta_ppm": m.delta_ppm,
                        }));
                    }
                }
            }
        }

        let mut multiplet_reports = Vec::new();
        let mut reported_signals = Vec::new();
        if let Some(reg_arr) = args.get("multiplet_regions").and_then(Value::as_array) {
            let ppm_vec: Vec<f64> = (0..spec.len()).map(|i| spec.ppm_at(i)).collect();
            let mcfg = MultipletConfig::default();
            for (idx, item) in reg_arr.iter().enumerate() {
                let Some(obj) = item.as_object() else {
                    return McpCallToolResult::domain_error(
                        "invalid_arguments",
                        format!("multiplet_regions[{idx}] must be a JSON object"),
                    );
                };
                let (Some(lo), Some(hi)) = (
                    obj.get("low_ppm")
                        .and_then(Value::as_f64)
                        .filter(|v| v.is_finite()),
                    obj.get("high_ppm")
                        .and_then(Value::as_f64)
                        .filter(|v| v.is_finite()),
                ) else {
                    return McpCallToolResult::domain_error(
                        "invalid_arguments",
                        format!(
                            "multiplet_regions[{idx}] requires finite 'low_ppm' and 'high_ppm'"
                        ),
                    );
                };
                let proton_count = obj
                    .get("proton_count")
                    .and_then(Value::as_u64)
                    .map(|n| n as usize);
                let label = obj.get("label").and_then(Value::as_str).map(str::to_string);
                match analyze_multiplet_region(
                    &ppm_vec,
                    &spec.real,
                    lo,
                    hi,
                    spec.axis.obs_mhz,
                    &mcfg,
                ) {
                    Ok(ana) => {
                        reported_signals.push(ReportedSignal::from_analysis(
                            &ana,
                            proton_count,
                            label.clone(),
                        ));
                        multiplet_reports.push(json!({
                            "label": label,
                            "center_ppm": ana.center_ppm,
                            "range_ppm": [ana.range_ppm.0, ana.range_ppm.1],
                            "class": format!("{:?}", ana.class),
                            "couplings_hz": ana.couplings.iter().map(|c| c.j_hz).collect::<Vec<_>>(),
                            "proton_count": proton_count,
                        }));
                    }
                    Err(e) => {
                        return McpCallToolResult::domain_error(
                            "multiplet_analysis_error",
                            format!("{e}"),
                        );
                    }
                }
            }
        }

        let journal_report = if !reported_signals.is_empty() {
            let style = match args
                .get("journal_style")
                .and_then(Value::as_str)
                .unwrap_or("acs")
                .trim()
                .to_ascii_lowercase()
                .as_str()
            {
                "joc" => PublicationStyle::Joc,
                "rsc" => PublicationStyle::Rsc,
                "nature" => PublicationStyle::Nature,
                _ => PublicationStyle::Acs,
            };
            let report_nuc = ReportNucleus::from_ascii_token(spec.experiment.nucleus().as_str())
                .unwrap_or(ReportNucleus::H1);
            let jrep = JournalNmrReport {
                nucleus: report_nuc,
                proton_decoupled: matches!(
                    spec.experiment,
                    crate::nmr1d::Experiment1DKind::Carbon13Decoupled
                ),
                mhz: spec.axis.obs_mhz,
                solvent: solvent_opt.clone().unwrap_or_else(|| "CDCl3".to_string()),
                signals: reported_signals,
            };
            Some(format_journal_nmr_report(&jrep, style, TextEncoding::Ascii))
        } else {
            None
        };

        McpCallToolResult::ok_json(&json!({
            "path": self.policy.relativize_display_path(&resolved),
            "name": spec.name,
            "nucleus": spec.experiment.nucleus().as_str(),
            "obs_mhz": spec.axis.obs_mhz,
            "n_points": spec.len(),
            "solvent": solvent_opt,
            "calibration_offset_ppm": calibration_offset_ppm,
            "noise_sigma": noise_sigma,
            "picked_peak_count": picked.len(),
            "deconvolved_peak_count": deconv_opt.as_ref().map(|d| d.peaks.len()),
            "deconvolved_peaks": deconv_opt.as_ref().map(|d| &d.peaks),
            "residual_rmsd": deconv_opt.as_ref().map(|d| d.residual_rmsd),
            "impurity_matches": impurity_hits,
            "multiplets": multiplet_reports,
            "journal_report_ascii": journal_report,
        }))
    }

    fn tool_fit_dosy_diffusion(&self, args: &Map<String, Value>) -> McpCallToolResult {
        let Some(raw_path) = args.get("path").and_then(Value::as_str) else {
            return McpCallToolResult::domain_error(
                "invalid_arguments",
                "missing required string argument 'path'",
            );
        };
        let resolved = match self.policy.resolve_sandboxed_path(raw_path) {
            Ok(p) => p,
            Err(sec) => return sec.to_tool_error_result(),
        };

        let solvent = match parse_dosy_solvent_arg(
            args.get("solvent").and_then(Value::as_str).unwrap_or("D2O"),
        ) {
            Ok(s) => s,
            Err(e) => return McpCallToolResult::domain_error("invalid_arguments", e),
        };
        let temperature_k = args
            .get("temperature_k")
            .and_then(Value::as_f64)
            .filter(|v| v.is_finite() && *v > 150.0 && *v < 450.0)
            .unwrap_or(298.15);
        let compactness = match parse_dosy_compactness_arg(
            args.get("compactness_class")
                .and_then(Value::as_str)
                .unwrap_or("small-molecule"),
        ) {
            Ok(c) => c,
            Err(e) => return McpCallToolResult::domain_error("invalid_arguments", e),
        };

        if resolved.is_dir() {
            let exp_params = match read_bruker_dosy_dir(&resolved, &BrukerDosyOverrides::default())
                .or_else(|_| read_varian_dosy_dir(&resolved, &VarianDosyOverrides::default()))
            {
                Ok(p) => p,
                Err(e) => {
                    return McpCallToolResult::domain_error("dosy_parse_error", format!("{e}"));
                }
            };
            return McpCallToolResult::ok_json(&json!({
                "path": self.policy.relativize_display_path(&resolved),
                "mode": "directory_schedule",
                "n_gradient_steps": exp_params.b_values_s_m2.len(),
                "b_values_s_m2": exp_params.b_values_s_m2,
                "experiment_params": exp_params,
            }));
        }

        let text = match fs::read_to_string(&resolved) {
            Ok(t) => t,
            Err(e) => {
                return McpCallToolResult::domain_error(
                    "io_error",
                    format!("failed to read DOSY dataset JSON: {e}"),
                );
            }
        };
        let parsed: McpDosyInputJson = match serde_json::from_str(&text) {
            Ok(d) => d,
            Err(e) => {
                return McpCallToolResult::domain_error(
                    "dosy_parse_error",
                    format!("invalid DOSY JSON payload: {e}"),
                );
            }
        };
        let b_values = match parsed.resolve_b_values() {
            Ok(b) => b,
            Err(e) => return McpCallToolResult::domain_error("dosy_physics_error", e),
        };
        if parsed.peaks.is_empty() {
            return McpCallToolResult::domain_error(
                "invalid_arguments",
                "DOSY JSON dataset contains zero peak decay traces",
            );
        }

        let max_components = args
            .get("max_components")
            .and_then(Value::as_u64)
            .unwrap_or(2)
            .clamp(1, 3) as usize;
        let criterion = match args
            .get("criterion")
            .and_then(Value::as_str)
            .unwrap_or("bic")
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "aic" => ModelSelectionCriterion::Aic,
            "bic" => ModelSelectionCriterion::Bic,
            other => {
                return McpCallToolResult::domain_error(
                    "invalid_arguments",
                    format!("unsupported DOSY criterion '{other}' (expected 'bic' or 'aic')"),
                );
            }
        };
        let bootstrap_samples = args
            .get("bootstrap_samples")
            .and_then(Value::as_u64)
            .unwrap_or(64)
            .min(500) as usize;
        let rng_seed = args
            .get("seed")
            .and_then(Value::as_u64)
            .unwrap_or(20_261_404);
        let include_ilt = args
            .get("include_ilt")
            .and_then(Value::as_bool)
            .unwrap_or(false);

        let fit_cfg = DosyFitConfig {
            max_components,
            criterion,
            bootstrap_samples,
            rng_seed,
            ..DosyFitConfig::default()
        };
        let ilt_cfg = IltConfig::default();

        let mut peak_reports = Vec::with_capacity(parsed.peaks.len());
        for pk in &parsed.peaks {
            let sel = match fit_stejskal_tanner_multi(&b_values, &pk.intensities, &fit_cfg) {
                Ok(s) => s,
                Err(e) => {
                    return McpCallToolResult::domain_error(
                        "dosy_fit_error",
                        format!("peak '{}': {e}", pk.label),
                    );
                }
            };
            let mut comps = Vec::with_capacity(sel.best_fit.components.len());
            for (c_idx, comp) in sel.best_fit.components.iter().enumerate() {
                let hydro =
                    estimate_hydrodynamics(comp.d_m2_s, solvent, temperature_k, compactness).ok();
                comps.push(json!({
                    "component_index": c_idx,
                    "amplitude": comp.amplitude,
                    "fraction": comp.fraction,
                    "d_m2_s": comp.d_m2_s,
                    "log10_d": comp.log10_d,
                    "ci95_d_m2_s": comp.ci95_d_m2_s,
                    "r_h_angstrom": hydro.as_ref().map(|h| h.r_h_angstrom),
                    "estimated_mw_da": hydro.as_ref().map(|h| h.estimated_mw_da),
                }));
            }
            let ilt_summary = if include_ilt {
                match solve_tikhonov_ilt(&b_values, &pk.intensities, &ilt_cfg) {
                    Ok(ilt) => Some(json!({
                        "selected_lambda": ilt.selected_lambda,
                        "modes": ilt.modes,
                        "caveat": ilt.caveat,
                    })),
                    Err(e) => {
                        return McpCallToolResult::domain_error(
                            "dosy_ilt_error",
                            format!("peak '{}': {e}", pk.label),
                        );
                    }
                }
            } else {
                None
            };
            peak_reports.push(json!({
                "label": pk.label,
                "center_ppm": pk.center_ppm,
                "selected_k": sel.selected_k,
                "bic": sel.best_fit.bic,
                "aic": sel.best_fit.aic,
                "rmse": sel.best_fit.rmse,
                "components": comps,
                "ilt": ilt_summary,
            }));
        }

        McpCallToolResult::ok_json(&json!({
            "path": self.policy.relativize_display_path(&resolved),
            "solvent": solvent.label(),
            "temperature_k": temperature_k,
            "compactness_class": compactness.label(),
            "n_gradient_steps": b_values.len(),
            "n_peaks": peak_reports.len(),
            "ilt_caveat": include_ilt.then_some(ILT_ARTIFACT_CAVEAT),
            "peaks": peak_reports,
        }))
    }

    fn tool_run_metabolomics_chemometrics(&self, args: &Map<String, Value>) -> McpCallToolResult {
        let Some(raw_path) = args.get("path").and_then(Value::as_str) else {
            return McpCallToolResult::domain_error(
                "invalid_arguments",
                "missing required string argument 'path'",
            );
        };
        let resolved = match self.policy.resolve_sandboxed_path(raw_path) {
            Ok(p) => p,
            Err(sec) => return sec.to_tool_error_result(),
        };

        let samples = match self.load_sandboxed_metabolomics_cohort(&resolved) {
            Ok(s) => s,
            Err(err_res) => return err_res,
        };
        if samples.len() < 4 {
            return McpCallToolResult::domain_error(
                "invalid_arguments",
                format!(
                    "metabolomics cohort requires at least 4 samples for PCA/OPLS-DA, got {}",
                    samples.len()
                ),
            );
        }

        let mask_preset = match args
            .get("mask_preset")
            .and_then(Value::as_str)
            .unwrap_or("urine")
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "none" => BiofluidMaskPreset::None,
            "aqueous-minimal" | "aqueous" => BiofluidMaskPreset::AqueousMinimal,
            "serum-plasma" | "serum" | "plasma" => BiofluidMaskPreset::SerumPlasma,
            "urine" => BiofluidMaskPreset::Urine,
            other => {
                return McpCallToolResult::domain_error(
                    "invalid_arguments",
                    format!(
                        "unsupported mask_preset '{other}' (expected 'urine', 'serum-plasma', 'aqueous-minimal', or 'none')"
                    ),
                );
            }
        };

        let bucketing = match args
            .get("bucketing")
            .and_then(Value::as_str)
            .unwrap_or("adaptive-intelligent")
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "adaptive-intelligent" | "aib" | "ai-binning" => BucketingMethod::default(),
            "uniform" => {
                let w = args
                    .get("bin_width_ppm")
                    .and_then(Value::as_f64)
                    .filter(|v| v.is_finite() && *v >= 0.002 && *v <= 0.5)
                    .unwrap_or(0.04);
                BucketingMethod::Uniform {
                    bin_width_ppm: w,
                    min_bucket_snr: 0.0,
                }
            }
            other => {
                return McpCallToolResult::domain_error(
                    "invalid_arguments",
                    format!(
                        "unsupported bucketing '{other}' (expected 'adaptive-intelligent' or 'uniform')"
                    ),
                );
            }
        };

        let normalization = match args
            .get("normalization")
            .and_then(Value::as_str)
            .unwrap_or("pqn")
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "pqn" | "probabilistic-quotient" => NormalizationMethod::ProbabilisticQuotient,
            "tsa" | "total-area" => NormalizationMethod::TotalSpectralArea,
            "internal-standard" | "tsp" | "dss" => NormalizationMethod::InternalStandard {
                target_ppm: 0.0,
                half_window_ppm: 0.08,
            },
            "none" => NormalizationMethod::None,
            other => {
                return McpCallToolResult::domain_error(
                    "invalid_arguments",
                    format!(
                        "unsupported normalization '{other}' (expected 'pqn', 'tsa', 'internal-standard', or 'none')"
                    ),
                );
            }
        };

        let scaling = match args
            .get("scaling")
            .and_then(Value::as_str)
            .unwrap_or("pareto")
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "pareto" => ScalingMethod::Pareto,
            "uv" | "unit-variance" => ScalingMethod::UnitVariance,
            "mean" | "mean-center" => ScalingMethod::MeanCenter,
            "none" => ScalingMethod::None,
            other => {
                return McpCallToolResult::domain_error(
                    "invalid_arguments",
                    format!(
                        "unsupported scaling '{other}' (expected 'pareto', 'uv', 'mean', or 'none')"
                    ),
                );
            }
        };

        let pca_components = args
            .get("pca_components")
            .and_then(Value::as_u64)
            .unwrap_or(3)
            .clamp(1, 10) as usize;
        let opls_orthogonal_components = args
            .get("opls_orthogonal_components")
            .and_then(Value::as_u64)
            .unwrap_or(1)
            .clamp(1, 5) as usize;
        let cv_folds = args
            .get("cv_folds")
            .and_then(Value::as_u64)
            .unwrap_or(5)
            .clamp(2, 10) as usize;
        let n_permutations = args
            .get("permutations")
            .and_then(Value::as_u64)
            .unwrap_or(40)
            .clamp(5, 500) as usize;
        let rng_seed = args
            .get("seed")
            .and_then(Value::as_u64)
            .unwrap_or(20_261_405);
        let top_k = args
            .get("top_k_biomarkers")
            .and_then(Value::as_u64)
            .unwrap_or(10)
            .clamp(1, 50) as usize;

        let cfg = MetabolomicsPipelineConfig {
            mask_preset,
            bucketing,
            normalization,
            scaling,
            pca_components,
            opls_orthogonal_components,
            cv_folds,
            n_permutations,
            rng_seed,
            ..MetabolomicsPipelineConfig::default()
        };

        let study = run_metabolomics_pipeline(&samples, &cfg);
        let driver_ppm_opt = args
            .get("stocsy_driver_ppm")
            .and_then(Value::as_f64)
            .filter(|v| v.is_finite())
            .or_else(|| study.biomarkers.first().map(|b| b.center_ppm))
            .or_else(|| study.buckets.first().map(|b| b.center_ppm));

        let stocsy_summary = driver_ppm_opt
            .and_then(|drv| compute_stocsy_at_ppm(drv, &study))
            .map(|st| {
                let mut hits: Vec<Value> = study
                    .buckets
                    .iter()
                    .enumerate()
                    .filter_map(|(i, b)| {
                        let r2 = st.bucket_r2.get(i).copied().unwrap_or(0.0);
                        (r2 >= 0.50).then(|| {
                            json!({
                                "bucket_index": i,
                                "center_ppm": b.center_ppm,
                                "correlation": st.bucket_correlation.get(i).copied().unwrap_or(0.0),
                                "r2": r2,
                                "covariance": st.bucket_covariance.get(i).copied().unwrap_or(0.0),
                            })
                        })
                    })
                    .collect();
                hits.truncate(20);
                json!({
                    "driver_bucket_index": st.driver_bucket_index,
                    "driver_ppm": st.driver_ppm,
                    "covarying_buckets_r2_ge_0_50": hits,
                    "bmrb_metabolite_matches": st.metabolite_matches.iter().take(5).collect::<Vec<_>>(),
                })
            });

        let top_biomarkers: Vec<_> = study.biomarkers.iter().take(top_k).cloned().collect();

        McpCallToolResult::ok_json(&json!({
            "path": self.policy.relativize_display_path(&resolved),
            "n_samples": study.sample_ids.len(),
            "n_buckets": study.buckets.len(),
            "pca": {
                "explained_variance_ratio": study.pca.explained_variance_ratio,
                "hotelling_t2_limit_95": study.pca.hotelling_t2_limit_95,
                "q_residual_limit_95": study.pca.q_residual_limit_95,
            },
            "opls_da": study.opls_da.as_ref().map(|o| json!({
                "control_group": o.control_group,
                "case_group": o.case_group,
                "r2_x_predictive": o.r2_x_predictive,
                "r2_x_orthogonal": o.r2_x_orthogonal,
                "r2_y": o.r2_y,
                "q2": o.q2,
                "permutation_p_value_q2": o.permutation.p_value_q2,
                "is_validated": o.permutation.is_validated,
            })),
            "top_biomarkers": top_biomarkers,
            "stocsy": stocsy_summary,
        }))
    }

    fn load_sandboxed_metabolomics_cohort(
        &self,
        resolved: &Path,
    ) -> Result<Vec<CohortSample1D>, McpCallToolResult> {
        let text = fs::read_to_string(resolved).map_err(|e| {
            McpCallToolResult::domain_error(
                "io_error",
                format!("failed to read metabolomics cohort file: {e}"),
            )
        })?;
        let is_json = resolved
            .extension()
            .and_then(|s| s.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("json"))
            || text.trim_start().starts_with('[')
            || text.trim_start().starts_with('{');
        if is_json {
            if let Ok(samples) = serde_json::from_str::<Vec<CohortSample1D>>(&text) {
                return Ok(samples);
            }
            let val: Value = serde_json::from_str(&text).map_err(|e| {
                McpCallToolResult::domain_error(
                    "metabolomics_parse_error",
                    format!("invalid metabolomics cohort JSON: {e}"),
                )
            })?;
            if let Some(arr) = val.get("samples") {
                let samples: Vec<CohortSample1D> =
                    serde_json::from_value(arr.clone()).map_err(|e| {
                        McpCallToolResult::domain_error(
                            "metabolomics_parse_error",
                            format!("invalid 'samples' array in cohort JSON: {e}"),
                        )
                    })?;
                return Ok(samples);
            }
            return Err(McpCallToolResult::domain_error(
                "metabolomics_parse_error",
                "JSON cohort file must be an array of CohortSample1D or contain a 'samples' array",
            ));
        }

        // Manifest TSV: `sample_id\tgroup\tpath` — every referenced path MUST pass sandbox resolution!
        let parent_dir = resolved.parent();
        let mut samples = Vec::new();
        for (line_idx, raw_line) in text.lines().enumerate() {
            let trimmed = raw_line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let cols: Vec<&str> = trimmed.split('\t').map(str::trim).collect();
            if line_idx == 0
                && cols.len() >= 3
                && (cols[0].eq_ignore_ascii_case("sample_id") || cols[0].eq_ignore_ascii_case("id"))
            {
                continue;
            }
            if cols.len() < 3 {
                return Err(McpCallToolResult::domain_error(
                    "metabolomics_parse_error",
                    format!(
                        "manifest line {} requires 3 tab-separated columns (sample_id, group, path)",
                        line_idx + 1
                    ),
                ));
            }
            let sample_id = cols[0];
            let group = cols[1];
            let raw_spec_path = cols[2];

            // Validate lexical components directly on the manifest entry first so `..` or credential
            // segments inside a manifest are always caught even when joined with `parent_dir`.
            if let Err(sec) = validate_lexical_components(Path::new(raw_spec_path)) {
                return Err(sec.to_tool_error_result());
            }
            let candidate_str = if Path::new(raw_spec_path).is_absolute() {
                raw_spec_path.to_string()
            } else if let Some(pdir) = parent_dir {
                pdir.join(raw_spec_path).to_string_lossy().into_owned()
            } else {
                raw_spec_path.to_string()
            };
            let resolved_spec = match self.policy.resolve_sandboxed_path(&candidate_str) {
                Ok(p) => p,
                Err(sec) => return Err(sec.to_tool_error_result()),
            };
            let spec = load_processed_spectrum_1d(&resolved_spec, None).map_err(|e| {
                McpCallToolResult::domain_error(
                    "nmr1d_load_error",
                    format!("sample '{sample_id}': {e}"),
                )
            })?;
            samples.push(CohortSample1D::new(sample_id, group, spec));
        }
        Ok(samples)
    }
}

#[derive(Clone, Debug, Deserialize)]
struct McpDosyPeakJson {
    label: String,
    center_ppm: f64,
    intensities: Vec<f64>,
}

#[derive(Clone, Debug, Deserialize)]
struct McpDosyInputJson {
    #[serde(default)]
    b_values_s_m2: Vec<f64>,
    #[serde(default)]
    gradients_g_per_cm: Vec<f64>,
    #[serde(default)]
    big_delta_s: Option<f64>,
    #[serde(default)]
    little_delta_s: Option<f64>,
    #[serde(default)]
    tau_s: Option<f64>,
    peaks: Vec<McpDosyPeakJson>,
}

impl McpDosyInputJson {
    fn resolve_b_values(&self) -> Result<Vec<f64>, String> {
        if !self.b_values_s_m2.is_empty() {
            return Ok(self.b_values_s_m2.clone());
        }
        if self.gradients_g_per_cm.is_empty() {
            return Err(
                "DOSY JSON requires non-empty 'b_values_s_m2' or 'gradients_g_per_cm'".to_string(),
            );
        }
        let big_delta_s = self
            .big_delta_s
            .ok_or_else(|| "missing 'big_delta_s' when computing b-values".to_string())?;
        let little_delta_s = self
            .little_delta_s
            .ok_or_else(|| "missing 'little_delta_s' when computing b-values".to_string())?;
        compute_b_values_s_m2(
            &self.gradients_g_per_cm,
            DosyNucleus::H1,
            DosySequenceType::Ste,
            GradientShape::Rectangular,
            big_delta_s,
            little_delta_s,
            self.tau_s.unwrap_or(0.0),
        )
        .map_err(|e| format!("{e}"))
    }
}

fn parse_dosy_solvent_arg(raw: &str) -> Result<DosySolvent, String> {
    match raw
        .trim()
        .to_ascii_lowercase()
        .replace(['-', '_'], "")
        .as_str()
    {
        "d2o" => Ok(DosySolvent::D2O),
        "h2o" => Ok(DosySolvent::H2O),
        "cdcl3" => Ok(DosySolvent::CDCl3),
        "dmsod6" | "dmso" => Ok(DosySolvent::DmsoD6),
        "cd3od" | "meod" => Ok(DosySolvent::CD3OD),
        "cd3cn" => Ok(DosySolvent::CD3CN),
        _ => Err(format!(
            "unsupported DOSY solvent '{raw}' (expected D2O, H2O, CDCl3, DMSO-d6, CD3OD, or CD3CN)"
        )),
    }
}

fn parse_dosy_compactness_arg(raw: &str) -> Result<SoluteCompactnessClass, String> {
    match raw
        .trim()
        .to_ascii_lowercase()
        .replace(['-', '_', ' '], "")
        .as_str()
    {
        "smallmolecule" | "small" => Ok(SoluteCompactnessClass::SmallMolecule),
        "globularprotein" | "globular" | "protein" => Ok(SoluteCompactnessClass::GlobularProtein),
        "expandedpolymer" | "polymer" | "idp" => Ok(SoluteCompactnessClass::ExpandedPolymer),
        _ => Err(format!(
            "unsupported compactness_class '{raw}' (expected 'small-molecule', 'globular-protein', or 'expanded-polymer')"
        )),
    }
}

fn parse_assigned_apo_peaks_tsv(text: &str) -> Result<Vec<AssignedAmidePeak>, String> {
    let mut out = Vec::new();
    for (line_idx, raw_line) in text.lines().enumerate() {
        let trimmed = raw_line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = trimmed.split('\t').map(str::trim).collect();
        if line_idx == 0
            && cols.first().is_some_and(|c| {
                c.eq_ignore_ascii_case("seq_id") || c.eq_ignore_ascii_case("residue")
            })
        {
            continue;
        }
        if cols.len() < 4 {
            return Err(format!(
                "apo_peaks_path line {} requires 4 tab columns (seq_id, res_name, h_ppm, n_ppm)",
                line_idx + 1
            ));
        }
        let seq_id: i32 = cols[0]
            .parse()
            .map_err(|_| format!("invalid seq_id '{}' on line {}", cols[0], line_idx + 1))?;
        let res_name = cols[1].to_ascii_uppercase();
        let h_ppm: f64 = cols[2]
            .parse::<f64>()
            .ok()
            .filter(|v| v.is_finite())
            .ok_or_else(|| format!("invalid h_ppm '{}' on line {}", cols[2], line_idx + 1))?;
        let n_ppm: f64 = cols[3]
            .parse::<f64>()
            .ok()
            .filter(|v| v.is_finite())
            .ok_or_else(|| format!("invalid n_ppm '{}' on line {}", cols[3], line_idx + 1))?;
        out.push(AssignedAmidePeak {
            seq_index: out.len(),
            seq_id,
            res_name,
            h_ppm,
            n_ppm,
        });
    }
    Ok(out)
}

/// Top-level synchronous JSON-RPC 2.0 MCP request dispatcher (`T15.1`).
/// Returns `None` for JSON-RPC notifications (`notifications/initialized`) and `Some(json_string)`
/// for all requests or protocol errors.
pub fn handle_mcp_json_request(server: &mut McpServer, raw_json: &str) -> Option<String> {
    server.handle_json_request(raw_json)
}

/// Maximum single HTTP header or request line length (`16 KiB`) read prior to rejection.
pub const MAX_MCP_HEADER_LINE_BYTES: usize = 16 * 1024;
/// Maximum number of HTTP or MIME header lines accepted per request (`128`).
pub const MAX_MCP_HEADER_COUNT: usize = 128;

/// Classification of an HTTP or `stdio` `Content-Length` header value evaluated strictly
/// before any payload buffer allocation (`0` panics, immune to `usize` overflow).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContentLengthError {
    Malformed,
    ExceedsLimit { max_bytes: usize },
}

/// Parse a decimal `Content-Length` string digit-by-digit and enforce `<= max_bytes`
/// strictly before allocating any payload buffer, distinguishing malformed/negative values
/// (`Malformed`) from oversized or `usize`-overflowing values (`ExceedsLimit`).
pub fn parse_bounded_content_length(
    raw: &str,
    max_bytes: usize,
) -> Result<usize, ContentLengthError> {
    let trimmed = raw.trim();
    if trimmed.is_empty() || !trimmed.bytes().all(|b| b.is_ascii_digit()) {
        return Err(ContentLengthError::Malformed);
    }
    let mut acc: usize = 0;
    for b in trimmed.bytes() {
        let digit = (b - b'0') as usize;
        acc = acc
            .checked_mul(10)
            .and_then(|v| v.checked_add(digit))
            .ok_or(ContentLengthError::ExceedsLimit { max_bytes })?;
        if acc > max_bytes {
            return Err(ContentLengthError::ExceedsLimit { max_bytes });
        }
    }
    Ok(acc)
}

/// Read at most `max_bytes + 1` bytes up to `\n` so an unterminated or oversized line
/// cannot trigger unbounded memory growth before size validation.
pub fn read_bounded_line<R: BufRead>(
    reader: &mut R,
    buf: &mut String,
    max_bytes: usize,
) -> io::Result<(usize, bool)> {
    buf.clear();
    let cap = (max_bytes as u64).saturating_add(1);
    let n = reader.take(cap).read_line(buf)?;
    let exceeded = buf.len() > max_bytes;
    if exceeded && !buf.ends_with('\n') {
        loop {
            let avail = reader.fill_buf()?;
            if avail.is_empty() {
                break;
            }
            if let Some(nl_pos) = avail.iter().position(|&b| b == b'\n') {
                reader.consume(nl_pos + 1);
                break;
            }
            let take_len = avail.len();
            reader.consume(take_len);
        }
    }
    Ok((n, exceeded))
}

/// Run the MCP `stdio` message loop over any `BufRead` + `Write` pair, transparently supporting
/// both newline-delimited JSON-RPC (`{...}\n`) and `Content-Length: <N>\r\n\r\n{...}` framing
/// with strict pre-allocation frame size enforcement (`MAX_MCP_FRAME_BYTES`).
pub fn run_mcp_stdio_loop<R: BufRead, W: Write>(
    server: &mut McpServer,
    mut reader: R,
    mut writer: W,
) -> io::Result<McpLoopStats> {
    let mut stats = McpLoopStats::default();
    let mut line_buf = String::new();

    loop {
        let (n, exceeded) = read_bounded_line(&mut reader, &mut line_buf, MAX_MCP_FRAME_BYTES)?;
        if n == 0 {
            break; // Clean EOF
        }
        if exceeded {
            let err_resp = JsonRpcResponse::error(
                Value::Null,
                ERR_INVALID_REQUEST,
                format!("stdio line exceeds MAX_MCP_FRAME_BYTES ({MAX_MCP_FRAME_BYTES})"),
                None,
            );
            let out = serde_json::to_string(&err_resp).unwrap_or_default();
            writeln!(writer, "{out}")?;
            writer.flush()?;
            stats.responses_written += 1;
            continue;
        }

        let trimmed = line_buf.trim();
        if trimmed.is_empty() {
            continue;
        }

        if let Some(len_str) = strip_ascii_prefix_ignore_case(trimmed, "content-length:") {
            let content_len =
                parse_bounded_content_length(len_str, MAX_MCP_FRAME_BYTES).map_err(|err| {
                    let detail = match err {
                        ContentLengthError::Malformed => {
                            format!("invalid Content-Length header '{trimmed}'")
                        }
                        ContentLengthError::ExceedsLimit { max_bytes } => format!(
                            "Content-Length header '{trimmed}' exceeds MAX_MCP_FRAME_BYTES ({max_bytes})"
                        ),
                    };
                    io::Error::new(io::ErrorKind::InvalidData, detail)
                })?;

            // Consume remaining MIME headers up to the blank `\r\n` / `\n` separator line.
            let mut hdr_count = 0usize;
            loop {
                hdr_count += 1;
                if hdr_count > MAX_MCP_HEADER_COUNT {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "too many MIME headers in Content-Length stdio frame",
                    ));
                }
                let mut hdr_line = String::new();
                let (hn, hdr_exceeded) =
                    read_bounded_line(&mut reader, &mut hdr_line, MAX_MCP_HEADER_LINE_BYTES)?;
                if hn == 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "unexpected EOF before end of Content-Length headers",
                    ));
                }
                if hdr_exceeded {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Content-Length MIME header line exceeds maximum length",
                    ));
                }
                if hdr_line.trim().is_empty() {
                    break;
                }
            }

            let mut payload_bytes = vec![0u8; content_len];
            reader.read_exact(&mut payload_bytes)?;
            stats.frames_read += 1;

            let payload_str = match std::str::from_utf8(&payload_bytes) {
                Ok(s) => s,
                Err(e) => {
                    let err_resp = JsonRpcResponse::error(
                        Value::Null,
                        ERR_PARSE_ERROR,
                        format!("invalid UTF-8 in Content-Length frame: {e}"),
                        None,
                    );
                    let body = serde_json::to_string(&err_resp).unwrap_or_default();
                    write!(writer, "Content-Length: {}\r\n\r\n{body}", body.len())?;
                    writer.flush()?;
                    stats.responses_written += 1;
                    continue;
                }
            };

            if let Some(resp_json) = handle_mcp_json_request(server, payload_str) {
                write!(
                    writer,
                    "Content-Length: {}\r\n\r\n{resp_json}",
                    resp_json.len()
                )?;
                writer.flush()?;
                stats.responses_written += 1;
            } else {
                stats.notifications_handled += 1;
            }
        } else {
            stats.frames_read += 1;
            if let Some(resp_json) = handle_mcp_json_request(server, trimmed) {
                writeln!(writer, "{resp_json}")?;
                writer.flush()?;
                stats.responses_written += 1;
            } else {
                stats.notifications_handled += 1;
            }
        }
    }

    Ok(stats)
}

// ============================================================================
// Constant-Time Bearer Auth & Loopback HTTP Header Guard (`T15.1` / `T15.2`)
// ============================================================================

/// Constant-time byte slice equality check for bearer tokens (`std::hint::black_box` prevents
/// compiler short-circuit optimization).
pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (&x, &y) in a.iter().zip(b.iter()) {
        diff |= std::hint::black_box(x ^ y);
    }
    std::hint::black_box(diff) == 0
}

fn extract_host_without_port(raw_host: &str) -> &str {
    let trimmed = raw_host.trim();
    if let Some(rest) = trimmed.strip_prefix('[') {
        if let Some((ipv6, _tail)) = rest.split_once(']') {
            return ipv6;
        }
    }
    trimmed.split(':').next().unwrap_or("")
}

/// Validate a `--bind <addr:port>` specification and require a strict loopback IP
/// (`127.0.0.0/8`, `[::1]`, or `localhost:<port>` -> `127.0.0.1:<port>`).
/// Rejects `0.0.0.0`, `[::]`, LAN/WAN IPs, and external hostnames with `PermissionDenied`.
#[cfg(not(target_arch = "wasm32"))]
pub fn parse_loopback_bind_addr(bind: &str) -> io::Result<std::net::SocketAddr> {
    let trimmed = bind.trim();
    if let Some(port_str) = strip_ascii_prefix_ignore_case(trimmed, "localhost:") {
        let port = port_str.trim().parse::<u16>().map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("invalid port in loopback bind '{trimmed}': {e}"),
            )
        })?;
        return Ok(std::net::SocketAddr::from(([127, 0, 0, 1], port)));
    }
    let addr: std::net::SocketAddr = trimmed.parse().map_err(|e| {
        io::Error::new(
            io::ErrorKind::PermissionDenied,
            format!(
                "invalid or non-IP bind '{trimmed}' (only loopback IPs such as 127.0.0.1:<port> or [::1]:<port> are permitted): {e}"
            ),
        )
    })?;
    if !addr.ip().is_loopback() {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            format!(
                "refusing non-loopback HTTP bind '{addr}': MCP HTTP server must bind exclusively to loopback (127.0.0.1 or [::1])"
            ),
        ));
    }
    Ok(addr)
}

/// Validate HTTP headers for a local loopback MCP endpoint (`127.0.0.1` / `localhost` / `[::1]`
/// `Host`, no cross-origin `Origin`, and constant-time `Authorization: Bearer <token>`).
/// Returns `Ok(())` or `Err((http_status_code, reason))`.
pub fn validate_loopback_http_headers(
    headers: &BTreeMap<String, String>,
    expected_bearer_token: &str,
) -> Result<(), (u16, String)> {
    // 1. Validate `Host` header against DNS-rebinding attacks.
    let Some(host_val) = headers.get("host") else {
        return Err((403, "missing required Host header".to_string()));
    };
    let host_only = extract_host_without_port(host_val).to_ascii_lowercase();
    if host_only != "127.0.0.1" && host_only != "localhost" && host_only != "::1" {
        return Err((
            403,
            format!("non-loopback Host header '{host_val}' rejected (DNS-rebinding protection)"),
        ));
    }

    // 2. Reject cross-origin browser `Origin` headers (CSRF protection).
    if let Some(origin) = headers.get("origin") {
        let orig_trim = origin.trim().to_ascii_lowercase();
        let is_loopback_origin = orig_trim.starts_with("http://127.0.0.1:")
            || orig_trim == "http://127.0.0.1"
            || orig_trim.starts_with("http://localhost:")
            || orig_trim == "http://localhost"
            || orig_trim.starts_with("http://[::1]:")
            || orig_trim == "http://[::1]";
        if !is_loopback_origin {
            return Err((
                403,
                format!("cross-origin Origin header '{origin}' rejected"),
            ));
        }
    }

    // 3. Constant-time `Authorization: Bearer <token>` verification.
    if expected_bearer_token.trim().is_empty() {
        return Err((500, "empty server bearer token is forbidden".to_string()));
    }
    let Some(auth) = headers.get("authorization") else {
        return Err((401, "missing Authorization: Bearer header".to_string()));
    };
    let Some(provided_token) = strip_ascii_prefix_ignore_case(auth.trim(), "bearer ") else {
        return Err((
            401,
            "Authorization header must use Bearer scheme".to_string(),
        ));
    };
    if !constant_time_eq(
        provided_token.trim().as_bytes(),
        expected_bearer_token.as_bytes(),
    ) {
        return Err((401, "invalid bearer token".to_string()));
    }

    Ok(())
}

/// Handle a single synchronous HTTP connection on a loopback `TcpStream` (`T15.1`), capping header
/// and `Content-Length` sizes strictly before any payload buffer allocation (`0` panics).
#[cfg(not(target_arch = "wasm32"))]
pub fn handle_mcp_http_stream(
    server: &mut McpServer,
    mut stream: std::net::TcpStream,
    expected_bearer_token: &str,
) -> io::Result<()> {
    if let Ok(peer) = stream.peer_addr() {
        if !peer.ip().is_loopback() {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "rejected non-loopback peer connection",
            ));
        }
    }

    let mut reader = io::BufReader::new(stream.try_clone()?);
    let mut request_line = String::new();
    let (req_bytes, req_exceeded) =
        read_bounded_line(&mut reader, &mut request_line, MAX_MCP_HEADER_LINE_BYTES)?;
    if req_bytes == 0 {
        return Ok(());
    }
    if req_exceeded {
        write_http_response(
            &mut stream,
            413,
            "Payload Too Large",
            r#"{"error":"request line exceeds maximum length"}"#,
        )?;
        return Ok(());
    }
    let parts: Vec<&str> = request_line.split_whitespace().collect();
    if parts.len() < 2 {
        write_http_response(
            &mut stream,
            400,
            "Bad Request",
            r#"{"error":"bad request"}"#,
        )?;
        return Ok(());
    }
    let method = parts[0];

    let mut headers = BTreeMap::new();
    let mut duplicate_host = false;
    let mut header_count = 0usize;
    loop {
        header_count += 1;
        if header_count > MAX_MCP_HEADER_COUNT {
            write_http_response(
                &mut stream,
                400,
                "Bad Request",
                r#"{"error":"too many HTTP headers"}"#,
            )?;
            return Ok(());
        }
        let mut line = String::new();
        let (hb, h_exceeded) =
            read_bounded_line(&mut reader, &mut line, MAX_MCP_HEADER_LINE_BYTES)?;
        if hb == 0 || line.trim().is_empty() {
            break;
        }
        if h_exceeded {
            write_http_response(
                &mut stream,
                413,
                "Payload Too Large",
                r#"{"error":"HTTP header line exceeds maximum length"}"#,
            )?;
            return Ok(());
        }
        if let Some((k, v)) = line.split_once(':') {
            let key_lower = k.trim().to_ascii_lowercase();
            if key_lower == "host" && headers.contains_key("host") {
                duplicate_host = true;
            }
            headers.insert(key_lower, v.trim().to_string());
        }
    }

    if duplicate_host {
        write_http_response(
            &mut stream,
            403,
            "Forbidden",
            r#"{"error":"duplicate Host header rejected"}"#,
        )?;
        return Ok(());
    }

    if let Err((status, reason)) = validate_loopback_http_headers(&headers, expected_bearer_token) {
        let status_text = if status == 401 {
            "Unauthorized"
        } else {
            "Forbidden"
        };
        let body = serde_json::to_string(&json!({ "error": reason })).unwrap_or_default();
        write_http_response(&mut stream, status, status_text, &body)?;
        return Ok(());
    }

    if method != "POST" {
        write_http_response(
            &mut stream,
            405,
            "Method Not Allowed",
            r#"{"error":"MCP Streamable HTTP requires POST"}"#,
        )?;
        return Ok(());
    }

    let content_len = match headers.get("content-length") {
        None => 0usize,
        Some(raw_len) => match parse_bounded_content_length(raw_len, MAX_MCP_FRAME_BYTES) {
            Ok(n) => n,
            Err(ContentLengthError::Malformed) => {
                write_http_response(
                    &mut stream,
                    400,
                    "Bad Request",
                    r#"{"error":"invalid Content-Length header"}"#,
                )?;
                return Ok(());
            }
            Err(ContentLengthError::ExceedsLimit { .. }) => {
                write_http_response(
                    &mut stream,
                    413,
                    "Payload Too Large",
                    r#"{"error":"payload exceeds MAX_MCP_FRAME_BYTES"}"#,
                )?;
                return Ok(());
            }
        },
    };

    let mut body = vec![0u8; content_len];
    reader.read_exact(&mut body)?;
    let body_str = match std::str::from_utf8(&body) {
        Ok(s) => s,
        Err(_) => {
            write_http_response(
                &mut stream,
                400,
                "Bad Request",
                r#"{"error":"invalid utf8"}"#,
            )?;
            return Ok(());
        }
    };

    match handle_mcp_json_request(server, body_str) {
        Some(resp_json) => write_http_response(&mut stream, 200, "OK", &resp_json)?,
        None => write_http_response(&mut stream, 202, "Accepted", "")?,
    }
    Ok(())
}

/// Bind a synchronous loopback MCP Streamable HTTP server on `bind_spec` (`127.0.0.1:<port>` or
/// `[::1]:<port>`), enforcing non-empty bearer-token authentication and optionally stopping after
/// `max_requests` connections (for deterministic CLI/integration tests).
#[cfg(not(target_arch = "wasm32"))]
pub fn run_mcp_http_server(
    server: &mut McpServer,
    bind_spec: &str,
    expected_bearer_token: &str,
    max_requests: Option<usize>,
) -> io::Result<std::net::SocketAddr> {
    if expected_bearer_token.trim().is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "MCP HTTP transport requires a non-empty --token <bearer>",
        ));
    }
    let bind_addr = parse_loopback_bind_addr(bind_spec)?;
    let listener = std::net::TcpListener::bind(bind_addr)?;
    let local_addr = listener.local_addr()?;

    let mut served = 0usize;
    if max_requests == Some(0) {
        return Ok(local_addr);
    }
    for stream in listener.incoming().flatten() {
        let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(10)));
        let _ = stream.set_write_timeout(Some(std::time::Duration::from_secs(10)));
        let _ = handle_mcp_http_stream(server, stream, expected_bearer_token);
        served += 1;
        if let Some(limit) = max_requests {
            if served >= limit {
                break;
            }
        }
    }
    Ok(local_addr)
}

#[cfg(not(target_arch = "wasm32"))]
fn write_http_response(
    stream: &mut std::net::TcpStream,
    status_code: u16,
    status_text: &str,
    body: &str,
) -> io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 {status_code} {status_text}\r\nContent-Type: application/json\r\nCache-Control: no-store\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )?;
    stream.flush()
}

// ============================================================================
// Guided Spectroscopist Prompts (`T15.3`)
// ============================================================================

fn builtin_prompt_descriptors() -> Vec<McpPromptDescriptor> {
    vec![
        McpPromptDescriptor {
            name: "triage-qc".to_string(),
            description: "Multi-step dataset quality-control triage over unlabelled referencing, 3D aliphatic support, sequential connectivity, and calibrated confidence caps.".to_string(),
            arguments: vec![
                McpPromptArgument {
                    name: "workspace_path".to_string(),
                    description: "Workspace directory or relative dataset folder.".to_string(),
                    required: true,
                },
                McpPromptArgument {
                    name: "peaks_path".to_string(),
                    description: "Relative path to the picked or curated peak list TSV.".to_string(),
                    required: false,
                },
                McpPromptArgument {
                    name: "sequence_or_star_path".to_string(),
                    description: "Relative path to the sequence FASTA or NMR-STAR file.".to_string(),
                    required: false,
                },
                McpPromptArgument {
                    name: "solver".to_string(),
                    description: "Backbone solver engine ('evo-graph', 'standard', or 'bayes-sa').".to_string(),
                    required: false,
                },
            ],
        },
        McpPromptDescriptor {
            name: "resolve-ambiguous-strips".to_string(),
            description: "Guided sequential (i, i-1) 3D strip disambiguation for crowded amide roots, pre-Proline shifts, and degenerate Gly/Ser/Thr/Ala spin systems.".to_string(),
            arguments: vec![
                McpPromptArgument {
                    name: "workspace_path".to_string(),
                    description: "Workspace directory containing spectra or peak lists.".to_string(),
                    required: true,
                },
                McpPromptArgument {
                    name: "ambiguous_residues".to_string(),
                    description: "Comma-separated residue numbers or sequence stretch requiring strip disambiguation.".to_string(),
                    required: false,
                },
                McpPromptArgument {
                    name: "experiments".to_string(),
                    description: "Available 2D/3D experiments (e.g. 'HSQC,HNCACB,CBCACONH,HNCO,HNCACO').".to_string(),
                    required: false,
                },
            ],
        },
        McpPromptDescriptor {
            name: "deconvolve-1d-mixture-dosy".to_string(),
            description: "Analytical 1D NMR multiplet deconvolution, Fulmer-2010 solvent referencing, qNMR molar purity, and BIC-selected discrete Stejskal-Tanner DOSY mixture analysis.".to_string(),
            arguments: vec![
                McpPromptArgument {
                    name: "workspace_path".to_string(),
                    description: "Workspace directory containing 1D and/or pseudo-2D diffusion data.".to_string(),
                    required: true,
                },
                McpPromptArgument {
                    name: "solvent".to_string(),
                    description: "Deuterated solvent ('D2O', 'CDCl3', 'DMSO-d6', or 'CD3OD').".to_string(),
                    required: false,
                },
                McpPromptArgument {
                    name: "fit_mode".to_string(),
                    description: "Diffusion fitting mode ('discrete-bic' default vs opt-in 'contin-ilt').".to_string(),
                    required: false,
                },
            ],
        },
        McpPromptDescriptor {
            name: "fit-cest-cpmg-exchange".to_string(),
            description: "Guided invisible-state chemical exchange workflow for 2-/3-state Bloch-McConnell CEST and multi-field Baldwin-2014 CPMG relaxation dispersion.".to_string(),
            arguments: vec![
                McpPromptArgument {
                    name: "workspace_path".to_string(),
                    description: "Workspace directory containing pseudo-3D CEST or CPMG plane series.".to_string(),
                    required: true,
                },
                McpPromptArgument {
                    name: "experiment_type".to_string(),
                    description: "Exchange experiment family ('CEST', 'DEST', 'CPMG', or 'R1rho').".to_string(),
                    required: false,
                },
                McpPromptArgument {
                    name: "field_mhz".to_string(),
                    description: "Static spectrometer 1H field strengths in MHz (e.g. '600,800').".to_string(),
                    required: false,
                },
                McpPromptArgument {
                    name: "states".to_string(),
                    description: "Number of kinetic exchange states ('2' or '3').".to_string(),
                    required: false,
                },
            ],
        },
        McpPromptDescriptor {
            name: "analyze-metabolomics-cohort".to_string(),
            description: "Guided untargeted 1D NMR biofluid metabolomics workflow covering solvent/urea masking, AI-Binning, PQN normalization, PCA outlier screening, leak-free OPLS-DA permutation validation, and STOCSY CC0 BMRB matching.".to_string(),
            arguments: vec![
                McpPromptArgument {
                    name: "workspace_path".to_string(),
                    description: "Workspace directory containing cohort JSON/TSV manifest and 1D spectra.".to_string(),
                    required: true,
                },
                McpPromptArgument {
                    name: "cohort_path".to_string(),
                    description: "Relative path to the cohort JSON or manifest TSV (default: 'cohort.json').".to_string(),
                    required: false,
                },
                McpPromptArgument {
                    name: "biofluid".to_string(),
                    description: "Biofluid exclusion mask preset ('urine', 'serum-plasma', or 'aqueous-minimal').".to_string(),
                    required: false,
                },
                McpPromptArgument {
                    name: "driver_ppm".to_string(),
                    description: "Optional 1H chemical shift in ppm for STOCSY spin-system structural correlation.".to_string(),
                    required: false,
                },
            ],
        },
    ]
}

fn build_spectroscopist_prompt(
    name: &str,
    args: &Map<String, Value>,
) -> Result<McpGetPromptResult, JsonRpcError> {
    let get_opt = |k: &str, default: &str| -> String {
        args.get(k)
            .and_then(Value::as_str)
            .filter(|s| !s.trim().is_empty())
            .unwrap_or(default)
            .to_string()
    };
    let require_workspace = || -> Result<String, JsonRpcError> {
        args.get("workspace_path")
            .and_then(Value::as_str)
            .filter(|s| !s.trim().is_empty())
            .map(str::to_string)
            .ok_or_else(|| JsonRpcError {
                code: ERR_INVALID_PARAMS,
                message: format!("prompt '{name}' requires non-empty argument 'workspace_path'"),
                data: None,
            })
    };

    match name {
        "triage-qc" => {
            let ws = require_workspace()?;
            let peaks = get_opt("peaks_path", "peaks.tsv");
            let seq = get_opt("sequence_or_star_path", "sequence.fasta");
            let solver = get_opt("solver", "evo-graph");
            Ok(McpGetPromptResult {
                description: format!("Dataset QC triage workflow for '{ws}' (solver: {solver})"),
                messages: vec![
                    McpPromptMessage {
                        role: "user".to_string(),
                        content: McpContentBlock::text(format!(
                            "Perform a rigorous spectroscopist dataset quality-control triage on workspace '{ws}' using peak list '{peaks}', sequence/reference '{seq}', and solver '{solver}'.\n\
                            1. Call `inspect_nmr_directory` on '{ws}' to verify experiment dimensionality, quadrature polarity, NUS schedules, and missing triple-resonance pathways.\n\
                            2. Call `read_peak_list_summary` on '{peaks}' to inspect 2D HSQC amide root stoichiometry, 3D peak counts, signs (`CA` vs `-CB` in HNCACB), and intensity availability.\n\
                            3. Call `evaluate_dataset_qc` on '{peaks}' + '{seq}' and examine `verdict` (`PASS`, `WARN`, `ABSTAIN`), `hsqc_3d_support_frac`, `aliphatic_support_frac`, `resolved_bidir_aliphatic_frac` (canonical gate >= 0.58), `corroborated_connectivity_frac` (P >= 3.5 nats), `hsqc_3d_rms_dh_ppm` / `hsqc_3d_rms_dn_ppm`, and `offset_consistency_ok`.\n\
                            4. Interpret per-residue confidences conservatively: when `verdict == WARN`, sub-0.80 calls carry `warn_sub_hc_uncalibrated = true`; when `protein_hc_gate_triggered == true` or `ABSTAIN`, confidences are capped below 0.80 (`0` high-confidence calls). Report observed cold-cohort precision caveats alongside actionable acquisition/referencing remedies."
                        )),
                    },
                ],
            })
        }
        "resolve-ambiguous-strips" => {
            let ws = require_workspace()?;
            let residues = get_opt("ambiguous_residues", "all low-confidence (<0.80) segments");
            let exps = get_opt("experiments", "HSQC,HNCACB,CBCACONH,HNCO,HNCACO");
            Ok(McpGetPromptResult {
                description: format!(
                    "Sequential (i, i-1) strip disambiguation for '{ws}' ({residues})"
                ),
                messages: vec![McpPromptMessage {
                    role: "user".to_string(),
                    content: McpContentBlock::text(format!(
                        "Resolve ambiguous backbone strip assignments in workspace '{ws}' focusing on [{residues}] across [{exps}].\n\
                        1. Inspect available peaks and shifts with `read_peak_list_summary` and `read_nmrstar_summary` (never use external 3D coordinates during sequence-only backbone assignment).\n\
                        2. Compare intraresidue `(i)` vs sequential `(i-1)` carbon ladders (`HNCACB` sign-inverted `CB` vs `CBCACONH` `i-1`-only `CA/CB`, and `HNCACO` vs `HNCO` `C'(i-1)`).\n\
                        3. Check deterministic primary-sequence covalent shift offsets (`Wishart 1995` pre-Proline `CA`/`CB`/`C'` upfield shifts) and unique landmark spin systems (`Gly` lacking `CB`, `Ala` upfield `CB ~ 19 ppm`, `Ser/Thr` downfield `CB ~ 63-70 ppm`).\n\
                        4. If connected to the live desktop workbench (`T15.2`), propose strip edits exclusively through Undo-logged curation commands so every change can be reverted in a single `Cmd/Ctrl+Z` step."
                    )),
                }],
            })
        }
        "deconvolve-1d-mixture-dosy" => {
            let ws = require_workspace()?;
            let solvent = get_opt("solvent", "D2O");
            let fit_mode = get_opt("fit_mode", "discrete-bic");
            Ok(McpGetPromptResult {
                description: format!(
                    "1D NMR multiplet deconvolution, qNMR, and DOSY workflow for '{ws}' ({solvent}, {fit_mode})"
                ),
                messages: vec![McpPromptMessage {
                    role: "user".to_string(),
                    content: McpContentBlock::text(format!(
                        "Analyze the 1D NMR and pseudo-2D diffusion (`DOSY`) dataset in workspace '{ws}' (solvent: '{solvent}', diffusion model: '{fit_mode}').\n\
                        1. Inspect the workspace via `inspect_nmr_directory` to identify 1D (`1H`/`13C`/`DEPT`/`19F`/`31P`) and pulsed-field-gradient diffusion (`difflist`/`gzlvl1`, `STE`/`LED`/`BPP-LED`) experiments.\n\
                        2. Verify zero/first-order phase (`PHC0`/`PHC1`), Whittaker/Bernstein baseline flatness, HSVD residual water suppression, and chemical-shift referencing against the Fulmer-2010 residual '{solvent}' peak.\n\
                        3. Deconvolve overlapping 1D multiplets with global pseudo-Voigt line shapes, extract recursive Hoye (1994) `J`-coupling trees with AB roofing correction, and propagate internal-standard `qNMR` uncertainties.\n\
                        4. For `DOSY`, run the reproducible `{fit_mode}` discrete 1-3 component Stejskal-Tanner / Wu-Johnson `BPP-LED` nonlinear fit with `BIC` order selection first to avoid inverse-Laplace spike splitting; only compare against opt-in second-derivative regularized `contin-ilt` when polydispersity is suspected."
                    )),
                }],
            })
        }
        "fit-cest-cpmg-exchange" => {
            let ws = require_workspace()?;
            let exp_type = get_opt("experiment_type", "CEST+CPMG");
            let fields = get_opt("field_mhz", "600,800");
            let states = get_opt("states", "2");
            Ok(McpGetPromptResult {
                description: format!(
                    "Invisible-state {exp_type} exchange analysis for '{ws}' ({states}-state, {fields} MHz)"
                ),
                messages: vec![McpPromptMessage {
                    role: "user".to_string(),
                    content: McpContentBlock::text(format!(
                        "Guide invisible excited-state conformational exchange analysis (`{exp_type}`, {states}-state model, fields: {fields} MHz) in workspace '{ws}'.\n\
                        1. Inspect acquisition headers and `vdlist`/`vclist`/`fqlist` arrays via `inspect_nmr_directory` (`decode_pulprog: true`) to confirm interleaved vs plane-queued pseudo-3D loop order and reference ($T_{{\\text{{relax}}}} = 0$ / unsaturated) planes.\n\
                        2. Verify locked-shape plane integration uncertainties from per-plane MAD noise and duplicate-delay/offset replicates.\n\
                        3. For `CPMG`, screen each residue against the flat null model (`AIC`/`BIC` and $\\Delta R_{{2,\\text{{eff}}}} \\ge 2.0\\text{{ s}}^{{-1}}$) before global Baldwin-2014 fitting across static fields ({fields} MHz) to resolve slow-vs-fast exchange $\\alpha$-scaling.\n\
                        4. For `CEST`/`DEST`, integrate the homogeneous Bloch-McConnell matrix exponential across dual $B_1$ nutation fields, inspect `(p_B, k_ex)` $\\chi^2$ surface identifiability, and compare excited-state shifts $\\Delta\\varpi_{{AB}}$ against homolog-firewalled torsion/random-coil hypotheses (`predict_homolog_free_torsions`)."
                    )),
                }],
            })
        }
        "analyze-metabolomics-cohort" => {
            let ws = require_workspace()?;
            let cohort = get_opt("cohort_path", "cohort.json");
            let biofluid = get_opt("biofluid", "urine");
            let driver = get_opt("driver_ppm", "top VIP biomarker apex");
            Ok(McpGetPromptResult {
                description: format!(
                    "Untargeted 1D NMR metabolomics & STOCSY workflow for '{ws}' ({biofluid}, '{cohort}')"
                ),
                messages: vec![McpPromptMessage {
                    role: "user".to_string(),
                    content: McpContentBlock::text(format!(
                        "Conduct a reproducible untargeted 1D NMR metabolomics and chemometric analysis in workspace '{ws}' on cohort '{cohort}' (`mask_preset: '{biofluid}'`, `STOCSY` driver: {driver}).\n\
                        1. Inspect '{ws}' with `inspect_nmr_directory` and run `run_metabolomics_chemometrics` on '{cohort}' with `mask_preset: '{biofluid}'`, `bucketing: 'adaptive-intelligent'` (De Meyer 2008 valley-seeking `AI-Binning`), `normalization: 'pqn'` (Dieterle 2006 Probabilistic Quotient Normalization), and `scaling: 'pareto'`.\n\
                        2. Screen unsupervised `PCA` scores against the 95% Hotelling $T^2$ ellipse and Jackson-Mudholkar $Q$-residual threshold before interpreting supervised models.\n\
                        3. Verify leak-free stratified $k$-fold `OPLS-DA` cross-validated $Q^2$ and Westerhuis (2008) label-permutation $p$-value (`is_validated == true`) so small-$N$/large-$P$ class separation is never overclaimed.\n\
                        4. Cross-check top `VIP` / Wiklund `S-plot` / Benjamini-Hochberg `FDR` biomarkers with Cloarec (2005) `STOCSY` covarying intra-metabolite multiplets (`stocsy_driver_ppm`) matched against the CC0 `BMRB` biofluid reference library, noting matrix/pH shift caveats."
                    )),
                }],
            })
        }
        _ => Err(JsonRpcError {
            code: ERR_INVALID_PARAMS,
            message: format!("unknown spectroscopist prompt '{name}'"),
            data: None,
        }),
    }
}

// ============================================================================
// Client Configuration Generator (`T15.3`)
// ============================================================================

/// Target AI coding/scientific assistant client for MCP configuration export.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum McpClientTarget {
    Antigravity,
    ClaudeDesktop,
    Cursor,
}

impl McpClientTarget {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Antigravity => "antigravity",
            Self::ClaudeDesktop => "claude-desktop",
            Self::Cursor => "cursor",
        }
    }
}

/// Transport profile for generated MCP client configurations.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "kebab-case")]
pub enum McpClientMode {
    /// Launch `ferric mcp` over `stdio` confined to `workspace_roots`.
    Stdio {
        command: String,
        workspace_roots: Vec<String>,
        access_mode: McpAccessMode,
    },
    /// Connect to a live authenticated loopback `nmr-gui` desktop HTTP bridge (`127.0.0.1`).
    HttpBridge {
        endpoint_url: String,
        bearer_token: String,
        access_mode: McpAccessMode,
    },
}

/// Generate a validated client-specific `mcpServers` JSON configuration object for `Antigravity`,
/// `ClaudeDesktop`, or `Cursor` in either `Stdio` CLI mode or `HttpBridge` desktop token mode.
pub fn generate_mcp_client_config(
    target: McpClientTarget,
    mode: &McpClientMode,
) -> Result<Value, String> {
    match mode {
        McpClientMode::Stdio {
            command,
            workspace_roots,
            access_mode,
        } => {
            if command.trim().is_empty() {
                return Err("MCP stdio binary command must not be empty".to_string());
            }
            if workspace_roots.is_empty() || workspace_roots.iter().any(|r| r.trim().is_empty()) {
                return Err(
                    "at least one non-empty workspace root is required for stdio MCP config"
                        .to_string(),
                );
            }

            let mut args = vec![
                "mcp".to_string(),
                "--transport".to_string(),
                "stdio".to_string(),
            ];
            for root in workspace_roots {
                args.push("--workspace".to_string());
                args.push(root.clone());
            }
            args.push(access_mode.cli_flag().to_string());

            let server_entry = match target {
                McpClientTarget::Antigravity => json!({
                    "command": command,
                    "args": args,
                    "env": {
                        "NMR_MCP_ACCESS_MODE": access_mode.as_str()
                    }
                }),
                McpClientTarget::ClaudeDesktop | McpClientTarget::Cursor => json!({
                    "command": command,
                    "args": args,
                }),
            };

            Ok(json!({
                "mcpServers": {
                    "ferric": server_entry
                }
            }))
        }
        McpClientMode::HttpBridge {
            endpoint_url,
            bearer_token,
            access_mode,
        } => {
            let url = endpoint_url.trim();
            let lower = url.to_ascii_lowercase();
            let is_loopback = lower.starts_with("http://127.0.0.1:")
                || lower == "http://127.0.0.1"
                || lower.starts_with("http://127.0.0.1/")
                || lower.starts_with("http://localhost:")
                || lower == "http://localhost"
                || lower.starts_with("http://localhost/")
                || lower.starts_with("http://[::1]:")
                || lower == "http://[::1]"
                || lower.starts_with("http://[::1]/");
            if !is_loopback {
                return Err(format!(
                    "HTTP bridge URL '{url}' must target loopback (http://127.0.0.1 or http://localhost)"
                ));
            }
            if bearer_token.trim().is_empty() {
                return Err("HTTP bridge bearer_token must not be empty".to_string());
            }

            let auth_header = format!("Bearer {}", bearer_token.trim());
            let server_entry = match target {
                McpClientTarget::Antigravity => json!({
                    "serverUrl": url,
                    "headers": {
                        "Authorization": auth_header,
                        "X-Nmr-Access-Mode": access_mode.as_str()
                    }
                }),
                McpClientTarget::ClaudeDesktop => json!({
                    "transport": "streamable-http",
                    "url": url,
                    "headers": {
                        "Authorization": auth_header
                    }
                }),
                McpClientTarget::Cursor => json!({
                    "url": url,
                    "headers": {
                        "Authorization": auth_header
                    }
                }),
            };

            Ok(json!({
                "mcpServers": {
                    "ferric-workbench": server_entry
                }
            }))
        }
    }
}

/// Serialize `generate_mcp_client_config` as formatted pretty-printed JSON.
pub fn generate_mcp_client_config_pretty(
    target: McpClientTarget,
    mode: &McpClientMode,
) -> Result<String, String> {
    let val = generate_mcp_client_config(target, mode)?;
    serde_json::to_string_pretty(&val).map_err(|e| format!("failed to format config JSON: {e}"))
}

// ============================================================================
// Internal Schema & Parsing Helpers
// ============================================================================

fn builtin_tool_descriptors() -> Vec<McpToolDescriptor> {
    vec![
        McpToolDescriptor {
            name: "inspect_nmr_directory".to_string(),
            description: "Classify a sandboxed NMR workspace folder or archive (Bruker pdata/ser/pulseprogram, Varian procpar, NMRPipe, Sparky .ucsf, FASTA, NMR-STAR, NEF) and decode acquisition physics.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path inside the allowed workspace root (defaults to '.')."
                    },
                    "decode_pulprog": {
                        "type": "boolean",
                        "description": "Whether to run executed pulseprogram / procpar physics interpretation (default: true)."
                    }
                },
                "additionalProperties": false
            }),
            annotations: McpToolAnnotations {
                title: "Inspect NMR Workspace & Pulse Programs".to_string(),
                read_only_hint: true,
                destructive_hint: false,
                idempotent_hint: true,
                open_world_hint: false,
            },
        },
        McpToolDescriptor {
            name: "read_nmrstar_summary".to_string(),
            description: "Parse a sandboxed NMR-STAR v2.1/v3 file and summarize primary sequence, Proline count, backbone/sidechain chemical shift completeness, and ambiguity codes.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative or sandboxed path to the .str file."
                    },
                    "entity": {
                        "type": "string",
                        "description": "Optional Entity_ID selector."
                    }
                },
                "required": ["path"],
                "additionalProperties": false
            }),
            annotations: McpToolAnnotations {
                title: "Read NMR-STAR Chemical Shift Summary".to_string(),
                read_only_hint: true,
                destructive_hint: false,
                idempotent_hint: true,
                open_world_hint: false,
            },
        },
        McpToolDescriptor {
            name: "read_peak_list_summary".to_string(),
            description: "Parse a sandboxed peak list TSV and summarize per-experiment peak counts, dimensions, signs, chemical-shift spans, and assembled HSQC spin systems.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative or sandboxed path to the peak list .tsv file."
                    }
                },
                "required": ["path"],
                "additionalProperties": false
            }),
            annotations: McpToolAnnotations {
                title: "Read Peak List TSV Summary".to_string(),
                read_only_hint: true,
                destructive_hint: false,
                idempotent_hint: true,
                open_world_hint: false,
            },
        },
        McpToolDescriptor {
            name: "evaluate_dataset_qc".to_string(),
            description: "Compute DatasetQcReport completeness, unlabelled chemical-shift offset consistency, root-to-3D linewidth/registration RMS, SNR dynamic range, and protein-level HC gate warnings.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "peaks_path": {
                        "type": "string",
                        "description": "Sandboxed path to the peak list TSV."
                    },
                    "sequence": {
                        "type": "string",
                        "description": "One-letter protein sequence or FASTA text."
                    },
                    "star_path": {
                        "type": "string",
                        "description": "Optional sandboxed NMR-STAR file supplying the sequence when 'sequence' is omitted."
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Deterministic PRNG seed (default: 0)."
                    }
                },
                "required": ["peaks_path"],
                "additionalProperties": false
            }),
            annotations: McpToolAnnotations {
                title: "Evaluate Dataset Quality Control".to_string(),
                read_only_hint: true,
                destructive_hint: false,
                idempotent_hint: true,
                open_world_hint: false,
            },
        },
        McpToolDescriptor {
            name: "predict_homolog_free_torsions".to_string(),
            description: "Predict post-assignment backbone (phi, psi) and optional chi1 torsion restraints from NMR-STAR shifts using 3-mer/9-mer matching under the mandatory <=25% HomologFirewall.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "star_path": {
                        "type": "string",
                        "description": "Sandboxed path to the assigned NMR-STAR (.str) file."
                    },
                    "exclude_pdb_ids": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Query PDB IDs to exclude in Layer 1."
                    },
                    "exclude_bmrb_ids": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Query BMRB IDs to exclude in Layer 1."
                    },
                    "exclude_uniprot_id": {
                        "type": "string",
                        "description": "Optional UniProt accession to exclude in Layer 1."
                    },
                    "exclude_family_id": {
                        "type": "string",
                        "description": "Optional protein family identifier to exclude in Layer 1."
                    },
                    "exclude_topology_id": {
                        "type": "string",
                        "description": "Optional CATH/ECOD topology identifier to exclude in Layer 2."
                    },
                    "max_seq_identity": {
                        "type": "number",
                        "description": "Maximum sequence identity cutoff in [0.05, 0.25] (default: 0.25; values > 0.25 are rejected)."
                    },
                    "include_chi1": {
                        "type": "boolean",
                        "description": "Include sidechain chi1 predictions (default: false)."
                    }
                },
                "required": ["star_path"],
                "additionalProperties": false
            }),
            annotations: McpToolAnnotations {
                title: "Predict Homolog-Firewalled Torsion Restraints".to_string(),
                read_only_hint: true,
                destructive_hint: false,
                idempotent_hint: true,
                open_world_hint: false,
            },
        },
        McpToolDescriptor {
            name: "fit_relaxation_or_exchange".to_string(),
            description: "Fit sandboxed Phase 13 15N/13C exponential R1/R2 relaxation series, 2-state Bloch-McConnell CEST/DEST profiles, or multi-field Baldwin-2014 CPMG dispersion curves.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["relaxation", "cest", "cpmg"],
                        "description": "Kinetic or relaxation fitting family ('relaxation', 'cest', or 'cpmg')."
                    },
                    "path": {
                        "type": "string",
                        "description": "Sandboxed path to a relaxation TSV/NEF, CEST profile TSV, or CPMG series TSV."
                    },
                    "paths": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Optional list of sandboxed TSV paths for multi-field CPMG or multi-B1 CEST global fits."
                    },
                    "mc_replicates": {
                        "type": "integer",
                        "description": "Monte Carlo replicates for R1/R2 confidence intervals (default: 200)."
                    },
                    "cpmg_model": {
                        "type": "string",
                        "description": "CPMG dispersion model ('baldwin2014' default, 'cr72', 'lm63', 'numerical', or 'flat')."
                    },
                    "compute_chi2_surface": {
                        "type": "boolean",
                        "description": "Compute 2D (p_B, k_ex) chi2 profile grid for CEST (default: false)."
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Deterministic PRNG seed."
                    }
                },
                "required": ["mode"],
                "additionalProperties": false
            }),
            annotations: McpToolAnnotations {
                title: "Fit R1/R2 Relaxation or CEST/CPMG Exchange".to_string(),
                read_only_hint: true,
                destructive_hint: false,
                idempotent_hint: true,
                open_world_hint: false,
            },
        },
        McpToolDescriptor {
            name: "track_csp_titration".to_string(),
            description: "Track fast-exchange 2D HSQC/HMQC chemical-shift perturbation trajectories via Kuhn-Munkres assignment and fit global 1:1 tight-binding quadratic K_d.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "apo_star_path": {
                        "type": "string",
                        "description": "Sandboxed NMR-STAR (.str) file providing assigned apo amide/methyl shifts."
                    },
                    "apo_peaks_path": {
                        "type": "string",
                        "description": "Alternative sandboxed 4-column assigned apo TSV (seq_id, res_name, h_ppm, n_ppm)."
                    },
                    "nucleus_mode": {
                        "type": "string",
                        "enum": ["amide", "methyl"],
                        "description": "Heteronuclear scaling mode ('amide' 1H-15N default or 'methyl' 1H-13C)."
                    },
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": { "type": "string" },
                                "protein_conc_um": { "type": "number" },
                                "ligand_conc_um": { "type": "number" },
                                "peaks_path": { "type": "string" }
                            },
                            "required": ["protein_conc_um", "ligand_conc_um", "peaks_path"],
                            "additionalProperties": false
                        },
                        "description": "Ordered ligand titration steps referencing sandboxed 2D peak TSV files."
                    },
                    "max_step_csp_ppm": {
                        "type": "number",
                        "description": "Maximum single-step combined shift jump in ppm (default: 0.45)."
                    },
                    "n_mc": {
                        "type": "integer",
                        "description": "Monte Carlo replicates for 95% K_d confidence intervals (default: 200)."
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Deterministic PRNG seed."
                    }
                },
                "required": ["steps"],
                "additionalProperties": false
            }),
            annotations: McpToolAnnotations {
                title: "Track CSP Titration & Fit Global K_d".to_string(),
                read_only_hint: true,
                destructive_hint: false,
                idempotent_hint: true,
                open_world_hint: false,
            },
        },
        McpToolDescriptor {
            name: "analyze_nmr1d_spectrum".to_string(),
            description: "Analyze a sandboxed 1D NMR spectrum or FID (Bruker pdata/fid, JCAMP-DX, TSV, JSON): Fulmer-2010 solvent/impurity matching, pseudo-Voigt deconvolution, and Hoye J-coupling multiplet trees.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Sandboxed path to the 1D spectrum or FID directory/file."
                    },
                    "mhz": {
                        "type": "number",
                        "description": "Optional spectrometer observe frequency override in MHz."
                    },
                    "solvent": {
                        "type": "string",
                        "description": "Optional deuterated solvent ('CDCl3', 'D2O', 'DMSO-d6', 'CD3OD', etc.)."
                    },
                    "calibrate_solvent": {
                        "type": "boolean",
                        "description": "Auto-lock chemical-shift axis to residual solvent peak (default: false)."
                    },
                    "match_impurities": {
                        "type": "boolean",
                        "description": "Screen picked peaks against Fulmer-2010 trace impurities."
                    },
                    "deconvolve": {
                        "type": "boolean",
                        "description": "Run pseudo-Voigt lineshape deconvolution (default: true)."
                    },
                    "min_snr": {
                        "type": "number",
                        "description": "Minimum peak-picking SNR threshold (default: 6.0)."
                    },
                    "multiplet_regions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "low_ppm": { "type": "number" },
                                "high_ppm": { "type": "number" },
                                "proton_count": { "type": "integer" },
                                "label": { "type": "string" }
                            },
                            "required": ["low_ppm", "high_ppm"],
                            "additionalProperties": false
                        },
                        "description": "Optional 1H chemical-shift windows for recursive J-coupling tree extraction."
                    },
                    "journal_style": {
                        "type": "string",
                        "enum": ["acs", "joc", "rsc", "nature"],
                        "description": "Publication report style when multiplet_regions is provided (default: 'acs')."
                    }
                },
                "required": ["path"],
                "additionalProperties": false
            }),
            annotations: McpToolAnnotations {
                title: "Analyze 1D NMR Spectrum & Multiplets".to_string(),
                read_only_hint: true,
                destructive_hint: false,
                idempotent_hint: true,
                open_world_hint: false,
            },
        },
        McpToolDescriptor {
            name: "fit_dosy_diffusion".to_string(),
            description: "Fit sandboxed pulsed-field-gradient DOSY diffusion decays via BIC/AIC discrete Stejskal-Tanner multi-exponential models, Stokes-Einstein R_h / MW, and opt-in Tikhonov ILT.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Sandboxed path to a Bruker/Varian DOSY directory or DOSY decay JSON file."
                    },
                    "solvent": {
                        "type": "string",
                        "description": "Solvent viscosity preset ('D2O' default, 'H2O', 'CDCl3', 'DMSO-d6', 'CD3OD', 'CD3CN')."
                    },
                    "temperature_k": {
                        "type": "number",
                        "description": "Sample temperature in Kelvin (default: 298.15)."
                    },
                    "compactness_class": {
                        "type": "string",
                        "enum": ["small-molecule", "globular-protein", "expanded-polymer"],
                        "description": "Empirical power-law MW compactness regime (default: 'small-molecule')."
                    },
                    "max_components": {
                        "type": "integer",
                        "description": "Maximum discrete exponential components K_max in 1..=3 (default: 2)."
                    },
                    "criterion": {
                        "type": "string",
                        "enum": ["bic", "aic"],
                        "description": "Model selection criterion (default: 'bic')."
                    },
                    "bootstrap_samples": {
                        "type": "integer",
                        "description": "Bootstrap replicates for 95% D confidence intervals (default: 64)."
                    },
                    "include_ilt": {
                        "type": "boolean",
                        "description": "Include opt-in second-derivative Tikhonov NNLS ILT distribution (default: false)."
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Deterministic PRNG seed."
                    }
                },
                "required": ["path"],
                "additionalProperties": false
            }),
            annotations: McpToolAnnotations {
                title: "Fit DOSY Diffusion & Hydrodynamic Radius".to_string(),
                read_only_hint: true,
                destructive_hint: false,
                idempotent_hint: true,
                open_world_hint: false,
            },
        },
        McpToolDescriptor {
            name: "run_metabolomics_chemometrics".to_string(),
            description: "Run untargeted 1D NMR metabolomics chemometrics on a sandboxed cohort JSON or manifest TSV: AI-Binning, PQN normalization, Pareto PCA, leak-free OPLS-DA permutation validation, and STOCSY CC0 BMRB matching.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Sandboxed path to a cohort JSON file or 3-column manifest TSV (sample_id, group, path)."
                    },
                    "mask_preset": {
                        "type": "string",
                        "enum": ["urine", "serum-plasma", "aqueous-minimal", "none"],
                        "description": "Biofluid solvent/water/urea exclusion mask (default: 'urine')."
                    },
                    "bucketing": {
                        "type": "string",
                        "enum": ["adaptive-intelligent", "uniform"],
                        "description": "Spectral bucketing method ('adaptive-intelligent' AI-Binning default or 'uniform')."
                    },
                    "bin_width_ppm": {
                        "type": "number",
                        "description": "Uniform bucket width in ppm when bucketing='uniform' (default: 0.04)."
                    },
                    "normalization": {
                        "type": "string",
                        "enum": ["pqn", "tsa", "internal-standard", "none"],
                        "description": "Sample dilution normalization ('pqn' default)."
                    },
                    "scaling": {
                        "type": "string",
                        "enum": ["pareto", "uv", "mean", "none"],
                        "description": "Feature scaling ('pareto' default)."
                    },
                    "pca_components": {
                        "type": "integer",
                        "description": "Number of PCA components (default: 3)."
                    },
                    "opls_orthogonal_components": {
                        "type": "integer",
                        "description": "Number of orthogonal OPLS-DA components (default: 1)."
                    },
                    "cv_folds": {
                        "type": "integer",
                        "description": "Stratified cross-validation folds (default: 5)."
                    },
                    "permutations": {
                        "type": "integer",
                        "description": "Label permutation count for OPLS-DA validation (default: 40)."
                    },
                    "stocsy_driver_ppm": {
                        "type": "number",
                        "description": "Optional driver chemical shift in ppm for STOCSY covariance/correlation."
                    },
                    "top_k_biomarkers": {
                        "type": "integer",
                        "description": "Number of ranked VIP/S-plot/BH-FDR biomarkers to return (default: 10)."
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Deterministic PRNG seed."
                    }
                },
                "required": ["path"],
                "additionalProperties": false
            }),
            annotations: McpToolAnnotations {
                title: "Run 1D Metabolomics Chemometrics & STOCSY".to_string(),
                read_only_hint: true,
                destructive_hint: false,
                idempotent_hint: true,
                open_world_hint: false,
            },
        },
        McpToolDescriptor {
            name: "write_curated_peak_tsv".to_string(),
            description: "Write a validated, curated peak list TSV inside a sandboxed workspace directory (requires ReadWrite access mode).".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "output_path": {
                        "type": "string",
                        "description": "Sandboxed relative destination path for the curated .tsv file."
                    },
                    "source_peaks_path": {
                        "type": "string",
                        "description": "Optional existing sandboxed .tsv file to filter."
                    },
                    "peaks_tsv": {
                        "type": "string",
                        "description": "Optional inline peak TSV text when 'source_peaks_path' is not used."
                    },
                    "exclude_indices": {
                        "type": "array",
                        "items": { "type": "integer" },
                        "description": "Zero-based peak indices to exclude from the output."
                    },
                    "filter_experiments": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Optional allowlist of experiment names to retain."
                    }
                },
                "required": ["output_path"],
                "additionalProperties": false
            }),
            annotations: McpToolAnnotations {
                title: "Write Curated Peak List TSV".to_string(),
                read_only_hint: false,
                destructive_hint: true,
                idempotent_hint: true,
                open_world_hint: false,
            },
        },
        McpToolDescriptor {
            name: "export_nmrstar_shifts".to_string(),
            description: "Export assigned chemical shifts to a sandboxed NMR-STAR v3 (.str) file inside the workspace (requires ReadWrite access mode).".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "output_path": {
                        "type": "string",
                        "description": "Sandboxed relative destination path for the exported .str file."
                    },
                    "source_star_path": {
                        "type": "string",
                        "description": "Optional existing sandboxed .str file used as base sequence/shifts."
                    },
                    "sequence": {
                        "type": "string",
                        "description": "One-letter or FASTA protein sequence when creating a new NMR-STAR table."
                    },
                    "first_seq_id": {
                        "type": "integer",
                        "description": "Starting Seq_ID for 'sequence' (default: 1)."
                    },
                    "shifts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "seq_id": { "type": "integer" },
                                "atom": { "type": "string" },
                                "val": { "type": "number" },
                                "confidence": { "type": "number" }
                            },
                            "required": ["seq_id", "atom", "val"]
                        },
                        "description": "Chemical shift rows to insert or update."
                    },
                    "header_comment": {
                        "type": "string",
                        "description": "Optional provenance header comment."
                    }
                },
                "required": ["output_path"],
                "additionalProperties": false
            }),
            annotations: McpToolAnnotations {
                title: "Export NMR-STAR v3 Chemical Shifts".to_string(),
                read_only_hint: false,
                destructive_hint: true,
                idempotent_hint: true,
                open_world_hint: false,
            },
        },
    ]
}

fn require_params_object(
    params: Option<Value>,
    method: &str,
) -> Result<Map<String, Value>, JsonRpcError> {
    match params {
        Some(Value::Object(m)) => Ok(m),
        _ => Err(JsonRpcError {
            code: ERR_INVALID_PARAMS,
            message: format!("{method} requires a JSON object in 'params'"),
            data: None,
        }),
    }
}

fn require_params_object_or_empty(
    params: Option<Value>,
    method: &str,
) -> Result<Map<String, Value>, JsonRpcError> {
    match params {
        None | Some(Value::Null) => Ok(Map::new()),
        Some(Value::Object(m)) => Ok(m),
        Some(_) => Err(JsonRpcError {
            code: ERR_INVALID_PARAMS,
            message: format!("{method} 'params' must be a JSON object when provided"),
            data: None,
        }),
    }
}

fn strip_ascii_prefix_ignore_case<'a>(s: &'a str, prefix: &str) -> Option<&'a str> {
    if s.len() >= prefix.len() && s.is_char_boundary(prefix.len()) {
        let (head, tail) = s.split_at(prefix.len());
        if head.eq_ignore_ascii_case(prefix) {
            return Some(tail);
        }
    }
    None
}

fn parse_sequence_to_three_letter(raw: &str) -> Result<Vec<String>, String> {
    let mut one_letter = String::new();
    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('>') || trimmed.starts_with('#') {
            continue;
        }
        for ch in trimmed.chars() {
            if ch.is_ascii_alphabetic() {
                one_letter.push(ch.to_ascii_uppercase());
            }
        }
    }
    if one_letter.is_empty() {
        return Err("sequence text contains zero amino-acid residues".to_string());
    }
    let mut seq3 = Vec::with_capacity(one_letter.len());
    for ch in one_letter.chars() {
        let three = one_to_three(ch);
        if three == "UNK" && ch != 'X' {
            return Err(format!("unknown one-letter amino-acid code '{ch}'"));
        }
        seq3.push(three.to_string());
    }
    Ok(seq3)
}

fn collect_sandboxed_resource_files(
    canon_root: &Path,
    dir: &Path,
    out: &mut Vec<String>,
    depth: usize,
    max_files: usize,
) {
    if depth > 4 || out.len() >= max_files {
        return;
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let mut paths: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|d| d.path())).collect();
    paths.sort();

    for p in paths {
        if out.len() >= max_files {
            break;
        }
        let Some(name) = p.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if name.starts_with('.') || is_forbidden_credential_segment(name) {
            continue;
        }
        let Ok(meta) = fs::symlink_metadata(&p) else {
            continue;
        };
        if meta.file_type().is_symlink() {
            continue;
        }
        if meta.is_dir() {
            collect_sandboxed_resource_files(canon_root, &p, out, depth + 1, max_files);
        } else if meta.is_file() {
            let low = name.to_ascii_lowercase();
            if low.ends_with(".str")
                || low.ends_with(".nef")
                || low.ends_with(".tsv")
                || low.ends_with(".fasta")
                || low.ends_with(".fa")
            {
                if let Ok(canon_file) = fs::canonicalize(&p) {
                    if canon_file.starts_with(canon_root) {
                        if let Ok(rel) = canon_file.strip_prefix(canon_root) {
                            out.push(path_to_forward_slashes(rel));
                        }
                    }
                }
            }
        }
    }
}
