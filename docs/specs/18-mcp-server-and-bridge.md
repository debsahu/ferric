# 18. Clean-Room Model Context Protocol (`MCP`) Server & Live Desktop Workbench Bridge

Status: implemented in `crates/nmr-core/src/mcp.rs` (`T15.1` & `T15.3`), `crates/nmr-cli/src/mcp_cmd.rs` (`ferric mcp` & `ferric mcp-config`), `crates/nmr-gui/src/mcp_bridge.rs` + `crates/nmr-gui/src/main.rs` (`T15.2`), `crates/nmr-core/tests/mcp_tests.rs`, and `crates/nmr-cli/tests/mcp_cli_tests.rs`. Built strictly on the Rust standard library (`std::io`, `std::fs`, `std::path`, `std::net`) plus `serde` and `serde_json` with zero external SDK, HTTP, or image-encoding crates, maintaining full `wasm32-unknown-unknown` compilation parity.

> [!IMPORTANT]
> **Validation Scope (In-Process / Subprocess Protocol Self-Consistency)**: All MCP JSON-RPC 2.0 framing (`NewlineDelimited` & `Content-Length`), pre-allocation payload caps (`413 Payload Too Large`), canonicalized workspace path sandboxing, loopback HTTP `Bearer` / `Host` / `Origin` enforcement, `256`-step reversible UI Undo/Redo curation, client configuration generation (`Antigravity`, `Claude Desktop`, `Cursor`), and RFC 2083 `image/png` viewport encoding are verified via deterministic in-process Rust unit tests and `ferric mcp` / `ferric mcp-config` subprocess E2E tests (`0` live interactive sessions against external `Antigravity`, `Claude Desktop`, or `Cursor` desktop binaries are executed in automated CI; tracked as open follow-up `F18`).

## 1. Provenance, Standards, and Clean-Room Scope

Every protocol handler, transport framing parser, security sandbox validator, image encoder, and spectroscopist workflow in this module is implemented clean-room from public open specifications and peer-reviewed NMR literature:

1. **Model Context Protocol (`MCP`) Specification** (Open Specification, Protocol Revisions `2024-11-05` & `2025-03-26`) and **JSON-RPC 2.0 Specification** — lifecycle negotiation (`initialize`, `notifications/initialized`, `ping`), capability discovery and invocation (`tools/list`, `tools/call`, `resources/list`, `resources/read`, `prompts/list`, `prompts/get`), tool safety annotations (`readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`), and multimodal `{ "type": "image", "mimeType": "image/png" }` content blocks.
2. **Bray, T. (Ed.)** (RFC 8259, December 2017) — The JavaScript Object Notation (JSON) Data Interchange Format; **Fielding, R., Nottingham, M. & Reschke, J.** (RFC 9110, June 2022) — HTTP Semantics (`Authorization: Bearer`, `Host`, `Origin`, and `413` payload limits).
3. **Boutell, T. et al.** (RFC 2083, March 1997) — PNG (Portable Network Graphics) Specification v1.0; **Deutsch, P. & Gailly, J.-L.** (RFC 1950 & RFC 1951, May 1996) — ZLIB Compressed Data Format (`Adler-32` checksum & stored uncompressed `DEFLATE` blocks) and IEEE 802.3 `CRC-32`; **Josefsson, S.** (RFC 4648, October 2006) — Base64 Data Encoding.
4. **Scientific NMR Engines Exposed via MCP**:
   - Spectrometer & pulse-program physics (`nmr_core::classify`, `nmr_core::pulprog`): Kay, Ikura, Tschudin & Bax (*J. Magn. Reson.* 1990, **89**, 496–514); Grzesiek & Bax (*J. Am. Chem. Soc.* 1992, **114**, 6291–6293); Cavanagh et al. (*Protein NMR Spectroscopy*, 2nd ed., 2007).
   - Unlabelled dataset quality control & confidence calibration (`nmr_core::qc`): Guo, Pleiss, Sun & Weinberger (*ICML* 2017, **70**, 1321–1330); Chow (*IEEE Trans. Inf. Theory* 1970, **16**, 41–46); Zhang, Neal & Wishart (*J. Biomol. NMR* 2003, **25**, 173–195).
   - Post-assignment homolog-firewalled torsion prediction (`nmr_core::torsion_db`): Shen, Vernon, Baker & Bax (*Proc. Natl. Acad. Sci. U.S.A.* 2008, **105**, 4685–4690); Shen & Bax (*J. Biomol. NMR* 2013, **56**, 227–241); Berjanskii & Wishart (*J. Am. Chem. Soc.* 2005, **127**, 14970–14971).
   - Spin relaxation, chemical exchange & ligand titration (`nmr_core::relax_io`, `nmr_core::relax_fit`, `nmr_core::cest`, `nmr_core::cpmg`, `nmr_core::csp`): Farrow et al. (*Biochemistry* 1994, **33**, 5984–6003); Vallurupalli, Bouvignies & Kay (*J. Am. Chem. Soc.* 2012, **134**, 8148–8161); Baldwin (*J. Magn. Reson.* 2014, **244**, 114–124); Williamson (*Prog. Nucl. Magn. Reson. Spectrosc.* 2013, **73**, 1–16).
   - Analytical 1D NMR, `DOSY` & 1D metabolomics chemometrics (`nmr_core::nmr1d`, `nmr_core::multiplet`, `nmr_core::dosy`, `nmr_core::metabolomics`): Fulmer et al. (*Organometallics* 2010, **29**, 2176–2179); Hoye, Hanson & Vyvyan (*J. Org. Chem.* 1994, **59**, 4096–4103); Stejskal & Tanner (*J. Chem. Phys.* 1965, **42**, 288–292); Provencher (*Comput. Phys. Commun.* 1982, **27**, 213–227); De Meyer et al. (*Anal. Chem.* 2008, **80**, 3783–3790); Dieterle et al. (*Anal. Chem.* 2006, **78**, 4281–4290); Trygg & Wold (*J. Chemom.* 2002, **16**, 119–128); Westerhuis et al. (*Metabolomics* 2008, **4**, 81–89); Cloarec et al. (*Anal. Chem.* 2005, **77**, 1282–1289).

## 2. Core JSON-RPC 2.0 MCP Server Engine (`T15.1`, `nmr_core::mcp` & `ferric mcp`)

### 2.1 Framing, Pre-Allocation Bounds & Loopback HTTP Transport

`nmr_core::mcp` decouples synchronous JSON-RPC 2.0 request dispatch (`handle_mcp_json_request`) from byte-stream I/O (`run_mcp_stdio_loop<R: BufRead, W: Write>` and `run_mcp_http_server`):

1. **Line-Delimited `stdio` JSON-RPC (`McpFramingMode::NewlineDelimited`)**: Each JSON-RPC 2.0 request or notification is read via bounded `reader.take(...)` chunks capped at `MAX_MCP_FRAME_BYTES` (`16 MiB`) prior to heap growth. JSON-RPC notifications (messages omitting `"id"`, such as `notifications/initialized`) never emit a response frame.
2. **`Content-Length` Header Framing (`McpFramingMode::ContentLength`)**: `parse_bounded_content_length` parses decimal digits one byte at a time via checked arithmetic (`checked_mul` / `checked_add`) and enforces `<= MAX_MCP_FRAME_BYTES` **strictly before allocating any payload buffer** (`Err`, zero panics, immune to `usize` overflow).
3. **Loopback HTTP Transport (`#[cfg(not(target_arch = "wasm32"))]`)**: `parse_loopback_bind_addr` restricts `--bind` exclusively to `127.0.0.1:<port>`, `[::1]:<port>`, or `localhost:<port>`, unconditionally rejecting `0.0.0.0`, `[::]`, and LAN/WAN interfaces (`Err`). In `ferric mcp --transport http`, user-supplied bearer tokens (strongly preferred via `NMR_MCP_BEARER_TOKEN` environment injection rather than `ps`-visible `--token <bearer>` CLI `argv`) must contain at least `MIN_HTTP_BEARER_TOKEN_LEN = 32` non-whitespace characters (`Err` on `< 32`-character inputs); when neither `NMR_MCP_BEARER_TOKEN` nor `--token` is supplied, `ferric mcp` auto-generates a 256-bit (`64`-hex-char) OS CSPRNG bearer token (`/dev/urandom` on Unix, Windows CNG `BCryptGenRandom` on Windows) and logs it exclusively to `stderr` (`MCP_HTTP_BEARER_TOKEN <hex64>`) so `stdout` stays clean. Incoming HTTP requests enforce pre-allocation `Content-Length` bounds (`413 Payload Too Large` on `ExceedsLimit`, `400 Bad Request` on negative/malformed values), constant-time `Authorization: Bearer <token>` verification (`401 Unauthorized`), loopback `Host` header validation (`127.0.0.1`, `[::1]`, `localhost`, rejecting duplicate or external `Host` headers with `403 Forbidden`), and cross-origin `Origin` rejection (`403 Forbidden`). Before writing any `400`/`401`/`403`/`405`/`413` HTTP rejection and closing `TcpStream`, `handle_mcp_http_stream` drains bounded unread request-body bytes into an `8 KiB` stack buffer (`<= 1 MiB`, `200 ms` read timeout, `0 B` heap allocation) and performs a clean TCP write-half shutdown (`Shutdown::Write`), eliminating BSD/Darwin loopback `TCP RST` (`ECONNRESET`) on split-`send()` HTTP clients (`F25`).

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

### 2.3 Built-In Core NMR Tools (`12` Total: `10` Read-Only + `2` Mutating) & Resources

| Tool Name | Mode | `readOnlyHint` | `destructiveHint` | Underlying `nmr-core` Engine |
| :--- | :--- | :---: | :---: | :--- |
| `inspect_nmr_directory` | Read | `true` | `false` | `classify::classify_nmr_workspace` + `pulprog::interpret_experiment_dir` |
| `read_nmrstar_summary` | Read | `true` | `false` | `nmrstar::read_star_all_atoms` (sequence, backbone/sidechain shift & ambiguity counts) |
| `read_peak_list_summary` | Read | `true` | `false` | `peaks::read_peaks` + `spinsystems::build_spin_systems` |
| `evaluate_dataset_qc` | Read | `true` | `false` | `solver::assign` + `qc::evaluate_dataset_qc` (`DatasetQcReport`, SNR, root-3D drift, gates) |
| `predict_homolog_free_torsions` | Read | `true` | `false` | `torsion_db::HomologFirewall` (`<= 25%` seq-id ceiling, `L1`–`L4` audit SHA-256) |
| `fit_relaxation_or_exchange` | Read | `true` | `false` | `relax_fit::fit_decay_batch_cpu`, `cest::fit_cest_global_cpu`, `cpmg::fit_cpmg_global_cpu` |
| `track_csp_titration` | Read | `true` | `false` | `csp::track_titration_trajectories` / `track_methyl_hmqc_trajectories` + `fit_csp_global_kd` |
| `analyze_nmr1d_spectrum` | Read | `true` | `false` | `nmr1d` (`pick_peaks_1d`, `deconvolve_spectrum_1d`, Fulmer impurities) + `multiplet` (`ACS`/`JOC`/`RSC`/`Nature`) |
| `fit_dosy_diffusion` | Read | `true` | `false` | `dosy::fit_stejskal_tanner_multi` (`BIC`/`AIC`), `estimate_hydrodynamics`, opt-in `solve_tikhonov_ilt` |
| `run_metabolomics_chemometrics` | Read | `true` | `false` | `metabolomics::run_metabolomics_pipeline` (`AI-Binning`, `PQN`, `PCA`, `OPLS-DA`) + `compute_stocsy_at_ppm` |
| `write_curated_peak_tsv` | Write | `false` | `true` | `peaks::write_peaks` (requires `McpAccessMode::ReadWrite` + sandboxed output path) |
| `export_nmrstar_shifts` | Write | `false` | `true` | `nmrstar::write_star_with_header` (requires `McpAccessMode::ReadWrite` + sandboxed path) |

Built-in `resources/list` & `resources/read` URIs:
- `nmr://workspace/policy`: Active `McpSecurityPolicy` access mode and sandbox root count/labels.
- `nmr://workspace/inventory`: Classified `WorkspaceInventory` JSON for the primary workspace root.
- `nmr://catalog/experiments`: Canonical experiment correlation rules and supported solver regimes.
- `nmr://file/<root_index>/<relative_path>`: Bounded read-only access to discovered `.str`, `.nef`, `.tsv`, and `.fasta` files strictly validated through `resolve_sandboxed_path`.

## 3. Live Bidirectional `nmr-gui` Desktop Bridge (`T15.2`, `nmr_gui::mcp_bridge`)

### 3.1 `WorkbenchSnapshot`, `McpBridgeState` & Reversible `McpUndoLog`

The desktop workbench wires `McpBridgeState` (`McpBridgeController`) into the top `nmr-gui` toolbar (*"🔌 MCP Bridge"* control window + 1-click *"📋 Copy MCP Config"* button), publishing a thread-safe `WorkbenchSnapshot` (`active_residue_seq`, `crosshair_ppm`, `active_experiment`, `strip_candidates`, `selected_pdb_chain_res`, `undo_depth`, `redo_depth`, `qc_status`) and processing AI actions exclusively through reversible `McpUiCommand` entries recorded in a bounded `256`-step `McpUndoLog` (`apply_command`, `undo_last`, `redo_last`):

| Bridge Tool Name | Mutates UI State | Description |
| :--- | :---: | :--- |
| `get_workbench_state` | `false` | Read live `WorkbenchSnapshot` (`active_residue_seq`, `crosshair_ppm`, `strip_candidates`, `qc_status`, Undo/Redo depths) |
| `jump_to_residue` | `true` (Undoable) | Focus sequence ribbon, strip walk, and 3D structure selection on `seq_id` |
| `set_crosshair_ppm` | `true` (Undoable) | Move synchronized 2D/3D cursor crosshair (`h_ppm`, `n_ppm`, optional `c_ppm`) |
| `assign_spin_system_to_residue` | `true` (Undoable) | Bijectively assign `cluster_id` to `seq_id`, preserving any displaced occupant for 1-step Undo |
| `unassign_residue` | `true` (Undoable) | Clear residue assignment while recording prior spin-system mapping on Undo stack |
| `toggle_peak_exclusion` | `true` (Undoable) | Mark or unmark an artifact peak index as excluded |
| `highlight_residues_3d` | `true` (Undoable) | Highlight target residues (`seq_ids`, `label`) in the `wgpu` 3D structure viewer |
| `undo_last_ai_edit` | `true` | Revert the most recent AI or interactive `McpUiCommand` (`Cmd/Ctrl+Z` equivalent) |
| `capture_viewport_png` | `false` | Return active workbench contour/strip/ribbon viewport as RFC 2083 `{ "type": "image", "mimeType": "image/png" }` |

### 3.2 Authenticated Loopback Security, `wasm32` Guarding & Multimodal `image/png` Capture

1. **Loopback-Only Binding & `wasm32` Parity**: Desktop socket code is gated behind `#[cfg(not(target_arch = "wasm32"))]`, binding strictly to `127.0.0.1:0` (ephemeral port by default) while compiling cleanly with `0` warnings on `wasm32-unknown-unknown` (where `start_loopback_http` returns a descriptive unsupported-platform `Err`).
2. **Pre-Allocation Body Cap, OS CSPRNG Bearer Token & DNS-Rebinding Defense**: `generate_bearer_token` mints `nmr_mcp_<64-hex>` tokens from 256-bit OS CSPRNG entropy on native Unix (`/dev/urandom`) and native Windows (Windows CNG `BCryptGenRandom` via `bcrypt` with `BCRYPT_USE_SYSTEM_PREFERRED_RNG = 0x00000002`, verifying `NTSTATUS == 0` and non-zero output bytes); on non-Unix/non-Windows targets (`#[cfg(not(any(unix, windows)))]`, such as `wasm32` where loopback TCP sockets are inactive), it falls back to a multi-round `RandomState` SipHash mixer bounded by `std`'s `<= 128`-bit per-thread seed ceiling (`F19`). `read_loopback_http_request` caps header lines at `16 KiB` and checks `Content-Length <= MAX_MCP_HTTP_BODY_BYTES` (`4 MiB`) before buffer allocation (`413 Payload Too Large`), verifies `Authorization: Bearer <token>` in constant time (`401 Unauthorized`), enforces loopback `Host` (`127.0.0.1` or `localhost`), and rejects cross-origin `Origin` headers (`403 Forbidden`).
3. **Zero-Dependency `image/png` Encoder**: `encode_rgba_to_png` / `encode_rgba_png` serializes 8-bit RGBA framebuffers into RFC 2083 PNG streams (`\x89PNG\r\n\x1a\n` magic, `IHDR`, scanline filter `0`, RFC 1950 ZLIB stored blocks with `Adler-32`, IEEE `CRC-32`, and `IEND`) encoded via RFC 4648 Base64.

## 4. Security Policy, Domain Prompts & CLI Subcommands (`T15.3`)

### 4.1 Path Sandboxing & `ReadOnly` / `ReadWrite` Guardrails (`McpSecurityPolicy`)

1. **Default-Safe Access Mode**: `McpAccessMode` defaults to `ReadOnly`. Mutating tools (`write_curated_peak_tsv`, `export_nmrstar_shifts`) carry `annotations: { readOnlyHint: false, destructiveHint: true }`, are hidden from `tools/list` by default in `ferric mcp --mode read-only` (unless `--expose-blocked-write-tools` is passed), and are unconditionally denied (`ERR_READ_ONLY_DENIED`) whenever `access_mode == McpAccessMode::ReadOnly`.
2. **Canonicalized Sandbox Verification (`resolve_sandboxed_path` / `resolve_sandboxed_write_path`)**:
   - Rejects empty strings and embedded null bytes (`\0`).
   - Rejects any lexical `Component::ParentDir` (`..`) segment prior to filesystem access (including nested per-sample paths inside metabolomics cohort manifests and multi-step CSP titration manifests).
   - Rejects hidden credential and secret paths (`.ssh`, `.aws`, `.gnupg`, `.env`, `.env.*`, `.netrc`, `.kube`, `id_rsa`, `id_ed25519`) across both input components and canonicalized workspace-relative components.
   - Canonicalizes every configured `allowed_root` via `std::fs::canonicalize` and verifies that the canonicalized target path (for existing paths) or canonicalized deepest existing ancestor plus symlink-free normal tail (for new output files) resides strictly within at least one canonicalized `allowed_root`, blocking sibling-prefix bypasses (`workspace_evil` vs `workspace`), symlink directory escapes, and dangling symlinks.
   - Sanitizes workspace-relative paths in responses so host user home paths are never leaked.

### 4.2 Guided Spectroscopist Prompts (`5` Built-In Workflows)

`prompts/list` and `prompts/get` provide five parameterized multi-step spectroscopist workflows:
- `triage-qc`: Unlabelled referencing check, `DatasetQcReport` completeness/aliphatic/connectivity gate audit, and `PASS` / `WARN` / `ABSTAIN` confidence interpretation (preserving cold high-confidence precision caveats).
- `resolve-ambiguous-strips`: Multi-pathway `(i, i-1)` strip ladder disambiguation (`HNCACB`/`CBCACONH`, `HNCO`/`HNCACO`), pre-Proline covalent shift checks, and Undo-safe curation.
- `deconvolve-1d-mixture-dosy`: 1D phase/baseline/Fulmer-2010 referencing, global pseudo-Voigt multiplet deconvolution, `qNMR` purity stoichiometry, and `BIC`-selected discrete Stejskal–Tanner / `BPP-LED` `DOSY` vs opt-in Tikhonov `ILT`.
- `fit-cest-cpmg-exchange`: Locked-shape pseudo-3D plane integration, flat-null vs exchange model selection (`AIC`/`BIC`, $\Delta R_{2,\text{eff}} \ge 2.0\text{ s}^{-1}$), multi-field $\alpha$-scaling, and 2-/3-state Bloch–McConnell `CEST` or Baldwin-2014 `CPMG` fitting.
- `analyze-metabolomics-cohort`: Biofluid residual water/urea/TSP region masking, De Meyer (2008) `AI-Binning`, Dieterle (2006) `PQN` dilution normalization, Pareto-scaled `PCA` Hotelling $T^2$ / $Q$-residual outlier screening, leak-free stratified $k$-fold `OPLS-DA` with Westerhuis (2008) permutation $Q^2$ anti-overfitting checks, and Cloarec (2005) `STOCSY` spin-system covariance verification.

### 4.3 CLI Interface (`ferric mcp` & `ferric mcp-config`)

- **`ferric mcp`**:
  `[NMR_MCP_BEARER_TOKEN=<32+-char-secret>] ferric mcp [--transport stdio|http] [--root <dir> ...] [--mode read-only|read-write] [--bind 127.0.0.1:<port>] [--token <bearer>] [--max-requests <N>]`
  - On `--transport http`, prefer injecting `NMR_MCP_BEARER_TOKEN` (`>= 32` characters, `MIN_HTTP_BEARER_TOKEN_LEN = 32`) via environment variables rather than passing `--token <bearer>` in `ps`-visible process `argv`. Any supplied `NMR_MCP_BEARER_TOKEN` or `--token` shorter than `32` trimmed characters is rejected with `Err`; omitting both auto-generates a 256-bit (`64`-hex-char) OS CSPRNG bearer token logged to `stderr` (`F20`).
- **`ferric mcp-config`**:
  `[NMR_MCP_BEARER_TOKEN=<32+-char-secret>] ferric mcp-config --client <antigravity|claude-desktop|cursor> [--root <dir> ...] [--mode read-only|read-write] [--bridge-port <port> [--token <bearer>]] [-o <mcp_config.json>]`
  - When `--bridge-port <port>` is selected and a token is provided via `NMR_MCP_BEARER_TOKEN` (preferred) or `--token <bearer>`, `MIN_HTTP_BEARER_TOKEN_LEN = 32` (`>= 32` trimmed characters) is enforced.
  - Both `ferric mcp-config --bridge-port` and `nmr-gui`'s 1-click *"Copy MCP Config"* (`nmr_gui::mcp_bridge::build_mcp_client_config_json` / `build_mcp_client_config_json_with_permission`) delegate to `nmr_core::mcp::generate_mcp_client_config_pretty`, emitting the exact same canonical server entry key `"ferric-workbench"` and per-client Streamable HTTP schema (`F26`):
    - `Antigravity`: `{ "serverUrl": "http://127.0.0.1:<port>/mcp", "headers": { "Authorization": "Bearer <token>", "X-Nmr-Access-Mode": "read-only"|"read-write" } }`
    - `Claude Desktop`: `{ "transport": "streamable-http", "url": "http://127.0.0.1:<port>/mcp", "headers": { "Authorization": "Bearer <token>" } }`
    - `Cursor` (`--client cursor` / `generic`): `{ "url": "http://127.0.0.1:<port>/mcp", "headers": { "Authorization": "Bearer <token>" } }`

## 5. Validation Scope (In-Process / Subprocess Protocol Self-Consistency)

All Phase 15 verification gates (`crates/nmr-core/tests/mcp_tests.rs`, `crates/nmr-gui/src/mcp_bridge.rs` unit tests, and `crates/nmr-cli/tests/mcp_cli_tests.rs`) evaluate deterministic in-process and OS-subprocess JSON-RPC 2.0 / loopback HTTP client harnesses against synthetic and public fixture workspaces:
- **Verified Automatically**: `NewlineDelimited` & `Content-Length` `stdio` framing, pre-allocation `Content-Length` overflow/cap rejection (`413 Payload Too Large` on `stdio` and HTTP without buffer allocation), non-loopback bind rejection (`0.0.0.0`, `[::]`, LAN/WAN IPs), HTTP `401` Bearer / `403` DNS-rebinding `Host` & `Origin` defenses, `MIN_HTTP_BEARER_TOKEN_LEN = 32` rejection on `< 32`-char `--token` and `NMR_MCP_BEARER_TOKEN` inputs, `NMR_MCP_BEARER_TOKEN` environment authentication and 256-bit (`64`-hex-char) OS CSPRNG token auto-generation (`F19` & `F20`), `ReadOnly` vs `ReadWrite` enforcement, `9` sandbox escape classes (`..`, nested `..`, sibling-prefix dirs, `\0`, file/dir/dangling symlinks, `.ssh`/`.aws`/`.gnupg`/`.env` paths, and manifest-embedded path escapes), `256`-step bijective Undo/Redo state restoration, RFC 2083 `image/png` encoding, and `mcp-config` JSON schema export for `Antigravity`, `Claude Desktop`, and `Cursor`.
- **Not Yet Executed in Automated CI (`F18`)**: `0` live interactive sessions against external `Antigravity`, `Claude Desktop`, or `Cursor` host binaries have been recorded in automated CI; manual/scripted external-client smoke verification is tracked as open follow-up `F18`.
