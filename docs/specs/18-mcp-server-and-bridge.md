# 18. Clean-Room Model Context Protocol (`MCP`) Server & Live Desktop Workbench Bridge

Status: implemented in `crates/nmr-core/src/mcp.rs` (`T15.1` & `T15.3`), `crates/nmr-core/tests/mcp_tests.rs`, and `crates/nmr-gui/src/mcp_bridge.rs` (`T15.2`). Built strictly on the Rust standard library (`std::io`, `std::fs`, `std::path`, `std::net`) plus `serde` and `serde_json` with zero external SDK, HTTP, or image-encoding crates, maintaining full `wasm32-unknown-unknown` compilation parity.

## 1. Provenance, Standards, and Clean-Room Scope

Every protocol handler, transport framing parser, security sandbox validator, image encoder, and spectroscopist workflow in this module is implemented clean-room from public open specifications and peer-reviewed NMR literature:

1. **Model Context Protocol (`MCP`) Specification** (Open Specification, Protocol Revisions `2024-11-05` & `2025-03-26`) and **JSON-RPC 2.0 Specification** — lifecycle negotiation (`initialize`, `notifications/initialized`, `ping`), capability discovery and invocation (`tools/list`, `tools/call`, `resources/list`, `resources/read`, `prompts/list`, `prompts/get`), tool safety annotations (`readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`), and multimodal `{ "type": "image", "mimeType": "image/png" }` content blocks.
2. **Bray, T. (Ed.)** (RFC 8259, December 2017) — The JavaScript Object Notation (JSON) Data Interchange Format; **Fielding, R., Nottingham, M. & Reschke, J.** (RFC 9110, June 2022) — HTTP Semantics (`Authorization: Bearer`, `Host`, and `Origin` header validation).
3. **Boutell, T. et al.** (RFC 2083, March 1997) — PNG (Portable Network Graphics) Specification v1.0; **Deutsch, P. & Gailly, J.-L.** (RFC 1950 & RFC 1951, May 1996) — ZLIB Compressed Data Format (`Adler-32` checksum & stored uncompressed `DEFLATE` blocks) and IEEE 802.3 `CRC-32`; **Josefsson, S.** (RFC 4648, October 2006) — Base64 Data Encoding.
4. **Scientific NMR Engines Exposed via MCP**:
   - Spectrometer & pulse-program physics (`nmr_core::classify`, `nmr_core::pulprog`): Kay, Ikura, Tschudin & Bax (*J. Magn. Reson.* 1990, **89**, 496–514); Grzesiek & Bax (*J. Am. Chem. Soc.* 1992, **114**, 6291–6293); Cavanagh et al. (*Protein NMR Spectroscopy*, 2nd ed., 2007).
   - Unlabelled dataset quality control & confidence calibration (`nmr_core::qc`): Guo, Pleiss, Sun & Weinberger (*ICML* 2017, **70**, 1321–1330); Chow (*IEEE Trans. Inf. Theory* 1970, **16**, 41–46); Zhang, Neal & Wishart (*J. Biomol. NMR* 2003, **25**, 173–195).
   - Post-assignment homolog-firewalled torsion prediction (`nmr_core::torsion_db`): Shen, Vernon, Baker & Bax (*Proc. Natl. Acad. Sci. U.S.A.* 2008, **105**, 4685–4690); Shen & Bax (*J. Biomol. NMR* 2013, **56**, 227–241); Berjanskii & Wishart (*J. Am. Chem. Soc.* 2005, **127**, 14970–14971).
   - Chemical exchange & diffusion workflows (`T13.4`–`T13.5`, `T14.1`–`T14.4`): Vallurupalli, Bouvignies & Kay (*J. Am. Chem. Soc.* 2012, **134**, 8148–8161); Baldwin (*J. Magn. Reson.* 2014, **244**, 114–124); Hoye, Hanson & Vyvyan (*J. Org. Chem.* 1994, **59**, 4096–4103); Stejskal & Tanner (*J. Chem. Phys.* 1965, **42**, 288–292); Provencher (*Comput. Phys. Commun.* 1982, **27**, 213–227).

## 2. Core JSON-RPC 2.0 MCP Server Engine (`T15.1`, `nmr_core::mcp`)

### 2.1 Framing & Transport Layer

`nmr_core::mcp` decouples synchronous JSON-RPC 2.0 request dispatch (`handle_mcp_json_request`) from byte-stream I/O (`run_mcp_stdio_loop<R: BufRead, W: Write>` and native loopback HTTP):

1. **Line-Delimited `stdio` JSON-RPC (`McpFramingMode::NewlineDelimited`)**: Each JSON-RPC 2.0 request or notification is read as a UTF-8 line terminated by `\n` (or `\r\n`), bounded by `MAX_MCP_FRAME_BYTES` (`16 MiB`). Responses are serialized onto a single compact UTF-8 JSON line terminated by `\n` and flushed immediately. JSON-RPC notifications (messages omitting the `"id"` member, such as `notifications/initialized`) never emit a response frame.
2. **`Content-Length` Header Framing (`McpFramingMode::ContentLength`)**: `run_mcp_stdio_loop` transparently auto-detects `Content-Length: <N>\r\n\r\n<payload>` headers on incoming streams, enforces `N <= MAX_MCP_FRAME_BYTES`, reads exact byte payloads, and replies using matching `Content-Length` headers.
3. **Loopback HTTP Transport (`#[cfg(not(target_arch = "wasm32"))]`)**: Binds strictly to `127.0.0.1` (`rejects` `0.0.0.0` or external interfaces), enforces constant-time `Authorization: Bearer <token>` comparison, validates `Host` (`127.0.0.1` or `localhost`), and rejects cross-origin `Origin` headers (`403 Forbidden`) to prevent browser CSRF and DNS-rebinding attacks.

### 2.2 JSON-RPC 2.0 Error Taxonomy

| Code | Constant | Condition |
| :--- | :--- | :--- |
| `-32700` | `ERR_PARSE_ERROR` | Malformed UTF-8 or syntactically invalid JSON payload |
| `-32600` | `ERR_INVALID_REQUEST` | Missing `"jsonrpc": "2.0"`, non-string `"method"`, invalid `"id"` type, or empty batch `[]` |
| `-32601` | `ERR_METHOD_NOT_FOUND` | Unsupported JSON-RPC method name |
| `-32602` | `ERR_INVALID_PARAMS` | Missing/mistyped method parameters, unknown tool/prompt name, or schema violation |
| `-32603` | `ERR_INTERNAL_ERROR` | Unhandled internal serialization or I/O failure |
| `-32001` | `ERR_SECURITY_VIOLATION` | Sandbox escape (`..`, escaping symlink, outside `allowed_roots`, null byte, credential path) |
| `-32002` | `ERR_READ_ONLY_DENIED` | Mutating tool invoked while `McpAccessMode::ReadOnly` is active |
| `-32003` | `ERR_RESOURCE_NOT_FOUND` | Unknown or missing `nmr://` resource URI |

In accordance with the MCP `tools/call` specification, protocol-level parameter mistakes (missing `"name"`, unknown tool name, non-object `"arguments"`) return JSON-RPC `-32602`, whereas tool-level security/permission denials and domain validation errors return a structured `McpCallToolResult` with `isError: true` and machine-readable `structuredContent` carrying the error code and kind so an AI assistant can observe and respect the boundary without dropping the session. Resource read violations (`resources/read`) return JSON-RPC error objects (`-32001` / `-32003`).

### 2.3 Built-In NMR Tools & Resources

| Tool Name | Mode | `readOnlyHint` | `destructiveHint` | Underlying `nmr-core` Engine |
| :--- | :--- | :---: | :---: | :--- |
| `inspect_nmr_directory` | Read | `true` | `false` | `classify::classify_nmr_workspace` + `pulprog::interpret_experiment_dir` |
| `read_nmrstar_summary` | Read | `true` | `false` | `nmrstar::read_star_all_atoms` (sequence, backbone/sidechain shift & ambiguity counts) |
| `read_peak_list_summary` | Read | `true` | `false` | `peaks::read_peaks` + `spinsystems::build_spin_systems` |
| `evaluate_dataset_qc` | Read | `true` | `false` | `solver::assign` + `qc::evaluate_dataset_qc` (`DatasetQcReport`, SNR, root-3D drift, gates) |
| `predict_homolog_free_torsions` | Read | `true` | `false` | `torsion_db::HomologFirewall` (`<= 25%` seq-id, `L1`–`L4` audit SHA-256) |
| `write_curated_peak_tsv` | Write | `false` | `true` | `peaks::write_peaks` (requires `McpAccessMode::ReadWrite` + sandboxed output path) |
| `export_nmrstar_shifts` | Write | `false` | `true` | `nmrstar::write_star_with_header` (requires `McpAccessMode::ReadWrite` + sandboxed path) |

Built-in `resources/list` & `resources/read` URIs:
- `nmr://workspace/policy`: Active `McpSecurityPolicy` access mode and sandbox root count/labels.
- `nmr://workspace/inventory`: Classified `WorkspaceInventory` JSON for the primary workspace root.
- `nmr://catalog/experiments`: Canonical experiment correlation rules and supported solver regimes.
- `nmr://file/<root_index>/<relative_path>`: Bounded read-only access to discovered `.str`, `.nef`, `.tsv`, and `.fasta` files strictly validated through `resolve_sandboxed_path`.

## 3. Live Bidirectional `nmr-gui` Desktop Bridge (`T15.2`, `nmr_gui::mcp_bridge`)

### 3.1 `WorkbenchSnapshot` & Reversible `McpUndoLog`

The desktop workbench exposes a thread-safe `WorkbenchSnapshot` (`active_residue_seq`, `crosshair_ppm`, `active_experiment`, `strip_candidates`, `selected_pdb_chain_res`, `undo_depth`, `redo_depth`, `qc_status`) and processes AI actions exclusively through reversible `McpUiCommand` entries (`JumpToResidue`, `SetCrosshairPpm`, `AssignSpinSystemToResidue`, `UnassignResidue`, `TogglePeakExclusion`, `HighlightResidues3d`) recorded in a bounded `256`-step `McpUndoLog` (`apply_command`, `undo_last`, `redo_last`). Every AI-initiated curation edit can be reverted cleanly in a single `Cmd/Ctrl+Z` step or via the `undo_last_ai_edit` MCP tool.

### 3.2 Authenticated Loopback Security & Multimodal `image/png` Capture

1. **Loopback-Only Binding**: The desktop HTTP bridge binds strictly to `127.0.0.1:0` (ephemeral port by default) and refuses non-loopback addresses.
2. **Bearer Token & DNS-Rebinding Defense**: Every HTTP request must supply `Authorization: Bearer <token>` verified in constant time alongside a loopback `Host` header (`127.0.0.1` or `localhost`) and no cross-origin `Origin` header (`401 Unauthorized` / `403 Forbidden`).
3. **Zero-Dependency `image/png` Encoder**: `encode_rgba_to_png` / `encode_rgba_png` serializes 8-bit RGBA framebuffers into RFC 2083 PNG streams (`\x89PNG\r\n\x1a\n` magic, `IHDR`, scanline filter `0`, RFC 1950 ZLIB stored blocks with `Adler-32`, IEEE `CRC-32`, and `IEND`) encoded via RFC 4648 Base64 into MCP `{ "type": "image", "mimeType": "image/png", "data": "<base64>" }` blocks.

## 4. Security Policy, Domain Prompts & Client Configs (`T15.3`)

### 4.1 Path Sandboxing & `ReadOnly` / `ReadWrite` Guardrails (`McpSecurityPolicy`)

1. **Default-Safe Access Mode**: `McpAccessMode` defaults to `ReadOnly`. Mutating tools (`write_curated_peak_tsv`, `export_nmrstar_shifts`) carry `annotations: { readOnlyHint: false, destructiveHint: true }`, can optionally be hidden from `tools/list`, and are unconditionally blocked at execution time whenever `access_mode == McpAccessMode::ReadOnly`.
2. **Canonicalized Sandbox Verification (`resolve_sandboxed_path` / `resolve_sandboxed_write_path`)**:
   - Rejects empty strings and embedded null bytes (`\0`).
   - Rejects any lexical `Component::ParentDir` (`..`) segment prior to filesystem access.
   - Rejects hidden credential and secret paths (`.ssh`, `.aws`, `.gnupg`, `.env`, `.env.*`, `.netrc`, `.kube`, `id_rsa`, `id_ed25519`) across both input components and canonicalized workspace-relative components.
   - Canonicalizes every configured `allowed_root` via `std::fs::canonicalize` and verifies that the canonicalized target path (for existing paths) or canonicalized deepest existing ancestor plus symlink-free normal tail (for new output files) resides strictly within at least one canonicalized `allowed_root`, blocking symlink directory escapes and dangling symlinks.
   - Sanitizes workspace-relative paths in responses so host user home paths are never leaked.

### 4.2 Guided Spectroscopist Prompts

`prompts/list` and `prompts/get` provide four parameterized multi-step spectroscopist workflows:
- `triage-qc`: Unlabelled referencing check, `DatasetQcReport` completeness/aliphatic/connectivity gate audit, and `PASS` / `WARN` / `ABSTAIN` confidence interpretation (preserving cold high-confidence precision caveats).
- `resolve-ambiguous-strips`: Multi-pathway `(i, i-1)` strip ladder disambiguation (`HNCACB`/`CBCACONH`, `HNCO`/`HNCACO`), pre-Proline covalent shift checks, and Undo-safe curation.
- `deconvolve-1d-mixture-dosy`: 1D phase/baseline/Fulmer-2010 referencing, global pseudo-Voigt multiplet deconvolution, `qNMR` purity stoichiometry, and `BIC`-selected discrete Stejskal–Tanner / `BPP-LED` `DOSY` vs opt-in `CONTIN` `ILT`.
- `fit-cest-cpmg-exchange`: Locked-shape pseudo-3D plane integration, flat-null vs exchange model selection (`AIC`/`BIC`, $\Delta R_{2,\text{eff}} \ge 2.0\text{ s}^{-1}$), multi-field $\alpha$-scaling, and 2-/3-state Bloch–McConnell `CEST` or Baldwin-2014 `CPMG` fitting.

### 4.3 Client Configuration Generator (`generate_mcp_client_config`)

Generates validated client configuration JSON objects for `McpClientTarget::{Antigravity, ClaudeDesktop, Cursor}` across two transport profiles:
- **`StdioCli`**: Launches `nmrassign mcp --transport stdio --workspace <root>` with explicit `--read-only` (default) or `--read-write` flags.
- **`HttpBridge`**: Connects to a loopback `http://127.0.0.1:<port>/mcp` desktop workbench endpoint (rejecting non-loopback URLs or empty tokens) with `Authorization: Bearer <token>` headers formatted for each client's `mcpServers` schema.
