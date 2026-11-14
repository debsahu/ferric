//! Comprehensive end-to-end protocol, tool/resource/prompt, client-config, and adversarial
//! security tests for `nmr_core::mcp` (`T15.1` & `T15.3`).

use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{json, Value};

use nmr_core::mcp::{
    decode_base64, encode_rgba_png, generate_mcp_client_config, generate_mcp_client_config_pretty,
    handle_mcp_json_request, run_mcp_stdio_loop, McpAccessMode, McpClientMode, McpClientTarget,
    McpContentBlock, McpSecurityErrorKind, McpSecurityPolicy, McpServer, ERR_INVALID_PARAMS,
    ERR_INVALID_REQUEST, ERR_METHOD_NOT_FOUND, ERR_PARSE_ERROR, ERR_READ_ONLY_DENIED,
    ERR_RESOURCE_NOT_FOUND, ERR_SECURITY_VIOLATION, MCP_PROTOCOL_VERSION,
};
use nmr_core::nmrstar::read_star_all_atoms;
use nmr_core::peaks::write_peaks;
use nmr_core::simulate::{simulate, SimConfig};

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(1);

struct TempTestSandbox {
    base: PathBuf,
    workspace: PathBuf,
    outside: PathBuf,
    sibling_prefix: PathBuf,
}

impl TempTestSandbox {
    fn new(tag: &str) -> Self {
        let seq = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let base = std::env::temp_dir().join(format!("nmr_mcp_test_{tag}_{pid}_{seq}"));
        let _ = fs::remove_dir_all(&base);
        let workspace = base.join("workspace");
        let outside = base.join("outside");
        let sibling_prefix = base.join("workspace_evil");
        fs::create_dir_all(&workspace).expect("create workspace dir");
        fs::create_dir_all(&outside).expect("create outside dir");
        fs::create_dir_all(&sibling_prefix).expect("create sibling prefix dir");
        Self {
            base,
            workspace,
            outside,
            sibling_prefix,
        }
    }
}

impl Drop for TempTestSandbox {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.base);
    }
}

fn fixture_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../data/fixtures")
        .join(rel)
}

fn populate_sample_nmr_workspace(ws: &Path) {
    // 1. Copy Ubiquitin NMR-STAR fixtures (synthetic + public BMRB 17769 all-atom shifts)
    let ubq_src = fixture_path("synthetic_ubiquitin.str");
    let ubq_dst = ws.join("synthetic_ubiquitin.str");
    fs::copy(&ubq_src, &ubq_dst).expect("copy synthetic_ubiquitin.str");
    fs::copy(
        fixture_path("bmr17769_ubiquitin_allatom.str"),
        ws.join("bmr17769_ubiquitin_allatom.str"),
    )
    .expect("copy bmr17769_ubiquitin_allatom.str");

    // 2. Simulate realistic Bruker 5-spectrum peak list and write `peaks.tsv`
    let ref_data = read_star_all_atoms(&ubq_dst, None).expect("parse ubiquitin star");
    let mut sim_cfg = SimConfig::realistic_bruker();
    sim_cfg.seed = 42;
    let (peaks, _truth) = simulate(&ref_data, &sim_cfg);
    write_peaks(ws.join("peaks.tsv"), &peaks).expect("write peaks.tsv");

    // 3. Write minimal Bruker 2D HSQC experiment directory (`10_hsqc/`) with executed `pulseprogram`
    let hsqc_dir = ws.join("10_hsqc");
    fs::create_dir_all(&hsqc_dir).expect("create 10_hsqc");
    fs::write(
        hsqc_dir.join("acqus"),
        "##TITLE= Parameter file, TopSpin\n\
         ##JCAMPDX= 5.0\n\
         ##$PULPROG= <hsqcetf3gpsi>\n\
         ##$PARMODE= 1\n\
         ##$TD= 2048\n\
         ##$SW= 14.0\n\
         ##$SFO1= 600.13\n\
         ##$O1= 2820.6\n\
         ##$NUC1= <1H>\n\
         ##END=\n",
    )
    .expect("write acqus");
    fs::write(
        hsqc_dir.join("acqu2s"),
        "##TITLE= Parameter file, F1\n\
         ##JCAMPDX= 5.0\n\
         ##$TD= 256\n\
         ##$FnMODE= 6\n\
         ##$SW= 32.0\n\
         ##$SFO1= 60.81\n\
         ##$O1= 7175.6\n\
         ##$NUC1= <15N>\n\
         ##END=\n",
    )
    .expect("write acqu2s");
    fs::write(
        hsqc_dir.join("pulseprogram"),
        ";hsqcetf3gpsi\n\
         #include <Avance.incl>\n\
         1 ze\n\
         2 d1\n\
           (p1 ph1):f1\n\
           d0\n\
           (p3 ph3):f2\n\
           go=2 ph31\n\
           mc #0 to 2 F1EA(calgrad(EA), caldel(d0, +in0))\n\
         exit\n",
    )
    .expect("write pulseprogram");
    fs::write(hsqc_dir.join("ser"), vec![0u8; 4096]).expect("write dummy ser");
}

#[test]
fn test_e2e_stdio_newline_lifecycle_and_all_read_tools() {
    let sb = TempTestSandbox::new("e2e_stdio");
    populate_sample_nmr_workspace(&sb.workspace);

    let policy = McpSecurityPolicy::read_only([&sb.workspace]).expect("read_only policy");
    assert_eq!(policy.access_mode(), McpAccessMode::ReadOnly);
    let mut server = McpServer::new(policy);

    let requests = [
        json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": { "name": "test-harness", "version": "1.0.0" }
            }
        }),
        // Notification (no `id`) — must NOT produce a response line!
        json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }),
        json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "ping"
        }),
        json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/list"
        }),
        json!({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "resources/list"
        }),
        json!({
            "jsonrpc": "2.0",
            "id": 5,
            "method": "resources/read",
            "params": { "uri": "nmr://file/0/synthetic_ubiquitin.str" }
        }),
        json!({
            "jsonrpc": "2.0",
            "id": 6,
            "method": "prompts/list"
        }),
        json!({
            "jsonrpc": "2.0",
            "id": 7,
            "method": "prompts/get",
            "params": {
                "name": "triage-qc",
                "arguments": {
                    "workspace_path": ".",
                    "peaks_path": "peaks.tsv",
                    "sequence_or_star_path": "synthetic_ubiquitin.str",
                    "solver": "evo-graph"
                }
            }
        }),
        json!({
            "jsonrpc": "2.0",
            "id": 8,
            "method": "tools/call",
            "params": {
                "name": "inspect_nmr_directory",
                "arguments": { "path": ".", "decode_pulprog": true }
            }
        }),
        json!({
            "jsonrpc": "2.0",
            "id": 9,
            "method": "tools/call",
            "params": {
                "name": "read_nmrstar_summary",
                "arguments": { "path": "synthetic_ubiquitin.str" }
            }
        }),
        json!({
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tools/call",
            "params": {
                "name": "read_peak_list_summary",
                "arguments": { "path": "peaks.tsv" }
            }
        }),
        json!({
            "jsonrpc": "2.0",
            "id": 11,
            "method": "tools/call",
            "params": {
                "name": "evaluate_dataset_qc",
                "arguments": {
                    "peaks_path": "peaks.tsv",
                    "star_path": "synthetic_ubiquitin.str",
                    "seed": 0
                }
            }
        }),
        json!({
            "jsonrpc": "2.0",
            "id": 12,
            "method": "tools/call",
            "params": {
                "name": "predict_homolog_free_torsions",
                "arguments": {
                    "star_path": "bmr17769_ubiquitin_allatom.str",
                    "exclude_pdb_ids": ["1D3Z", "1UBQ"],
                    "exclude_bmrb_ids": ["17769", "6457", "5387"]
                }
            }
        }),
    ];

    let mut wire_in = String::new();
    for req in &requests {
        wire_in.push_str(&serde_json::to_string(req).unwrap());
        wire_in.push('\n');
    }

    let mut wire_out = Vec::new();
    let stats = run_mcp_stdio_loop(
        &mut server,
        Cursor::new(wire_in.into_bytes()),
        &mut wire_out,
    )
    .expect("stdio loop succeeds");

    assert!(server.is_initialized());
    assert_eq!(stats.frames_read, 13);
    assert_eq!(stats.notifications_handled, 1);
    assert_eq!(stats.responses_written, 12);

    let out_str = String::from_utf8(wire_out).expect("valid utf8 responses");
    let lines: Vec<Value> = out_str
        .lines()
        .map(|l| serde_json::from_str(l).expect("valid JSON-RPC response line"))
        .collect();
    assert_eq!(lines.len(), 12);

    // Response 0 (id=1): initialize
    assert_eq!(lines[0]["id"], 1);
    assert_eq!(lines[0]["result"]["protocolVersion"], MCP_PROTOCOL_VERSION);
    assert_eq!(lines[0]["result"]["serverInfo"]["name"], "nmr-assign-mcp");

    // Response 1 (id=2): ping
    assert_eq!(lines[1]["id"], 2);
    assert_eq!(lines[1]["result"], json!({}));

    // Response 2 (id=3): tools/list
    let tools = lines[2]["result"]["tools"].as_array().expect("tools array");
    assert_eq!(tools.len(), 7);
    for t in tools {
        let tname = t["name"].as_str().unwrap();
        let ro = t["annotations"]["readOnlyHint"].as_bool().unwrap();
        let destr = t["annotations"]["destructiveHint"].as_bool().unwrap();
        if tname == "write_curated_peak_tsv" || tname == "export_nmrstar_shifts" {
            assert!(!ro, "{tname} must mark readOnlyHint=false");
            assert!(destr, "{tname} must mark destructiveHint=true");
        } else {
            assert!(ro, "{tname} must mark readOnlyHint=true");
            assert!(!destr, "{tname} must mark destructiveHint=false");
        }
    }

    // Response 3 & 4 (id=4, 5): resources/list & resources/read
    let res_list = lines[3]["result"]["resources"]
        .as_array()
        .expect("resources array");
    assert!(res_list
        .iter()
        .any(|r| r["uri"] == "nmr://file/0/synthetic_ubiquitin.str"));
    assert!(lines[4]["result"]["contents"][0]["text"]
        .as_str()
        .unwrap()
        .contains("loop_"));

    // Response 5 & 6 (id=6, 7): prompts/list & prompts/get
    let prompts = lines[5]["result"]["prompts"]
        .as_array()
        .expect("prompts array");
    let prompt_names: Vec<&str> = prompts
        .iter()
        .map(|p| p["name"].as_str().unwrap())
        .collect();
    assert_eq!(
        prompt_names,
        vec![
            "triage-qc",
            "resolve-ambiguous-strips",
            "deconvolve-1d-mixture-dosy",
            "fit-cest-cpmg-exchange"
        ]
    );
    let triage_msg = lines[6]["result"]["messages"][0]["content"]["text"]
        .as_str()
        .unwrap();
    assert!(triage_msg.contains("evaluate_dataset_qc"));
    assert!(triage_msg.contains("evo-graph"));

    // Response 7 (id=8): inspect_nmr_directory
    let inspect_sc = &lines[7]["result"]["structuredContent"];
    assert_eq!(lines[7]["result"]["isError"], false);
    assert!(inspect_sc["experiment_count"].as_u64().unwrap() >= 1);
    assert!(!inspect_sc["pulprog_interpretations"]
        .as_array()
        .unwrap()
        .is_empty());

    // Response 8 (id=9): read_nmrstar_summary
    let star_sc = &lines[8]["result"]["structuredContent"];
    assert_eq!(star_sc["sequence_length"], 76);
    assert!(star_sc["one_letter_sequence"]
        .as_str()
        .unwrap()
        .starts_with(
            "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
        ));
    assert!(star_sc["backbone_shifts"].as_u64().unwrap() > 300);

    // Response 9 (id=10): read_peak_list_summary
    let peak_sc = &lines[9]["result"]["structuredContent"];
    assert_eq!(peak_sc["is_backbone_ready"], true);
    assert_eq!(peak_sc["experiment_count"], 5);
    assert!(peak_sc["assembled_spin_systems"].as_u64().unwrap() >= 60);

    // Response 10 (id=11): evaluate_dataset_qc
    let qc_sc = &lines[10]["result"]["structuredContent"];
    assert!(matches!(
        qc_sc["verdict"].as_str().unwrap(),
        "PASS" | "WARN"
    ));
    let n_assigned = qc_sc["completeness"]["n_assigned"].as_u64().unwrap();
    assert!(
        n_assigned >= 45,
        "expected >= 45 assigned residues on realistic Bruker Ubiquitin, got {n_assigned} (qc_sc={qc_sc})"
    );
    assert!(qc_sc["intensity_snr_summary"]["median_height"]
        .as_f64()
        .is_some());

    // Response 11 (id=12): predict_homolog_free_torsions
    let tor_sc = &lines[11]["result"]["structuredContent"];
    assert_eq!(tor_sc["n_residues"], 76);
    assert!(tor_sc["n_good"].as_u64().unwrap() >= 35);
    assert!(tor_sc["excluded_templates_count"].as_u64().unwrap() >= 1);
    assert!(tor_sc["max_retained_seq_identity"].as_f64().unwrap() <= 0.25 + 1e-9);
    assert_eq!(tor_sc["exclusion_list_sha256"].as_str().unwrap().len(), 64);
    // Verify default `include_chi1 == false` strips chi1 values
    assert!(tor_sc["residues"]
        .as_array()
        .unwrap()
        .iter()
        .all(|r| r["chi1_deg"].is_null()));

    // Verify all other guided prompts also render cleanly
    for (pname, extra_args) in [
        (
            "resolve-ambiguous-strips",
            json!({"workspace_path": "ubq_ws", "ambiguous_residues": "24,25,52"}),
        ),
        (
            "deconvolve-1d-mixture-dosy",
            json!({"workspace_path": "dosy_ws", "solvent": "CDCl3", "fit_mode": "discrete-bic"}),
        ),
        (
            "fit-cest-cpmg-exchange",
            json!({"workspace_path": "cest_ws", "experiment_type": "CEST", "field_mhz": "600,800", "states": "3"}),
        ),
    ] {
        let raw = handle_mcp_json_request(
            &mut server,
            &serde_json::to_string(&json!({
                "jsonrpc": "2.0",
                "id": 99,
                "method": "prompts/get",
                "params": { "name": pname, "arguments": extra_args }
            }))
            .unwrap(),
        )
        .expect("prompt response");
        let parsed: Value = serde_json::from_str(&raw).unwrap();
        assert!(parsed["error"].is_null(), "prompt {pname} failed: {parsed}");
        assert!(!parsed["result"]["messages"].as_array().unwrap().is_empty());
    }

    // Verify HomologFirewall rejects attempts to raise max_seq_identity > 0.25
    let fw_bypass = handle_mcp_json_request(
        &mut server,
        &serde_json::to_string(&json!({
            "jsonrpc": "2.0",
            "id": 100,
            "method": "tools/call",
            "params": {
                "name": "predict_homolog_free_torsions",
                "arguments": {
                    "star_path": "synthetic_ubiquitin.str",
                    "max_seq_identity": 0.60
                }
            }
        }))
        .unwrap(),
    )
    .unwrap();
    let fw_val: Value = serde_json::from_str(&fw_bypass).unwrap();
    assert_eq!(fw_val["result"]["isError"], true);
    assert_eq!(
        fw_val["result"]["structuredContent"]["error"]["kind"],
        "homolog_firewall_violation"
    );
}

#[test]
fn test_stdio_content_length_framing_roundtrip() {
    let sb = TempTestSandbox::new("content_length");
    let policy = McpSecurityPolicy::read_only([&sb.workspace]).unwrap();
    let mut server = McpServer::new(policy);

    let frame = |v: Value| -> String {
        let s = serde_json::to_string(&v).unwrap();
        format!("Content-Length: {}\r\n\r\n{s}", s.len())
    };

    let mut input = String::new();
    input.push_str(&frame(json!({
        "jsonrpc": "2.0",
        "id": "init-1",
        "method": "initialize",
        "params": { "protocolVersion": "2025-03-26" }
    })));
    input.push_str(&frame(json!({
        "jsonrpc": "2.0",
        "method": "notifications/initialized"
    })));
    input.push_str(&frame(json!({
        "jsonrpc": "2.0",
        "id": "ping-2",
        "method": "ping"
    })));

    let mut output = Vec::new();
    let stats = run_mcp_stdio_loop(&mut server, Cursor::new(input.into_bytes()), &mut output)
        .expect("Content-Length stdio loop");
    assert_eq!(stats.frames_read, 3);
    assert_eq!(stats.notifications_handled, 1);
    assert_eq!(stats.responses_written, 2);
    assert_eq!(server.negotiated_protocol_version(), "2025-03-26");

    let out_text = String::from_utf8(output).unwrap();
    assert!(out_text.starts_with("Content-Length: "));
    assert!(out_text.contains(r#""id":"init-1""#));
    assert!(out_text.contains(r#""id":"ping-2""#));
}

#[test]
fn test_adversarial_parent_traversal_sibling_and_null_byte_rejection() {
    let sb = TempTestSandbox::new("traversal");
    let secret_outside = sb.outside.join("secret.str");
    let sibling_secret = sb.sibling_prefix.join("secret.str");
    fs::write(&secret_outside, "secret").unwrap();
    fs::write(&sibling_secret, "sibling_secret").unwrap();

    let policy = McpSecurityPolicy::read_only([&sb.workspace]).unwrap();

    // 1. Direct relative `..` escape
    let err = policy
        .resolve_sandboxed_path("../outside/secret.str")
        .unwrap_err();
    assert_eq!(err.kind, McpSecurityErrorKind::ParentTraversal);
    assert_eq!(err.kind.rpc_code(), ERR_SECURITY_VIOLATION);

    // 2. Nested `sub/../../outside/secret.str` escape
    let err2 = policy
        .resolve_sandboxed_path("sub/../../outside/secret.str")
        .unwrap_err();
    assert_eq!(err2.kind, McpSecurityErrorKind::ParentTraversal);

    // 3. Sibling prefix directory (`workspace_evil` vs `workspace`)
    let err3 = policy
        .resolve_sandboxed_path(sibling_secret.to_str().unwrap())
        .unwrap_err();
    assert_eq!(err3.kind, McpSecurityErrorKind::OutsideAllowedRoots);

    // 4. Null byte injection
    let err4 = policy
        .resolve_sandboxed_path("peaks.tsv\0.str")
        .unwrap_err();
    assert_eq!(err4.kind, McpSecurityErrorKind::NullByteOrInvalidPath);

    // 5. Over MCP `tools/call` and `resources/read`
    let mut server = McpServer::new(policy);
    let tool_raw = handle_mcp_json_request(
        &mut server,
        &serde_json::to_string(&json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "read_nmrstar_summary",
                "arguments": { "path": "../outside/secret.str" }
            }
        }))
        .unwrap(),
    )
    .unwrap();
    let tool_resp: Value = serde_json::from_str(&tool_raw).unwrap();
    assert_eq!(tool_resp["result"]["isError"], true);
    assert_eq!(
        tool_resp["result"]["structuredContent"]["error"]["code"],
        ERR_SECURITY_VIOLATION
    );

    let res_raw = handle_mcp_json_request(
        &mut server,
        &serde_json::to_string(&json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "resources/read",
            "params": { "uri": "nmr://file/0/../outside/secret.str" }
        }))
        .unwrap(),
    )
    .unwrap();
    let res_resp: Value = serde_json::from_str(&res_raw).unwrap();
    assert_eq!(res_resp["error"]["code"], ERR_SECURITY_VIOLATION);
}

#[cfg(unix)]
#[test]
fn test_adversarial_symlink_escape_and_dangling_symlink_rejection() {
    use std::os::unix::fs::symlink;

    let sb = TempTestSandbox::new("symlink");
    let outside_file = sb.outside.join("outside_shifts.str");
    fs::write(&outside_file, "outside shifts").unwrap();
    let outside_sub = sb.outside.join("dropzone");
    fs::create_dir_all(&outside_sub).unwrap();

    // Symlinks planted inside `workspace` pointing outside `allowed_roots`
    symlink(&outside_file, sb.workspace.join("escaped_file.str")).unwrap();
    symlink(&outside_sub, sb.workspace.join("escaped_dir")).unwrap();
    symlink(
        sb.outside.join("nonexistent_target.tsv"),
        sb.workspace.join("dangling_escape.tsv"),
    )
    .unwrap();

    // Symlink inside `workspace` pointing to hidden `.env` inside `workspace`
    fs::write(sb.workspace.join(".env"), "API_KEY=secret").unwrap();
    symlink(sb.workspace.join(".env"), sb.workspace.join("harmless.str")).unwrap();

    let rw_policy = McpSecurityPolicy::read_write([&sb.workspace]).unwrap();

    // File symlink escape rejected
    let e1 = rw_policy
        .resolve_sandboxed_path("escaped_file.str")
        .unwrap_err();
    assert_eq!(e1.kind, McpSecurityErrorKind::SymlinkEscape);

    // Directory symlink write escape rejected (even in ReadWrite mode!)
    let e2 = rw_policy
        .resolve_sandboxed_write_path("escaped_dir/pwned.tsv", "write_curated_peak_tsv")
        .unwrap_err();
    assert_eq!(e2.kind, McpSecurityErrorKind::SymlinkEscape);
    assert!(!outside_sub.join("pwned.tsv").exists());

    // Dangling symlink write rejected
    let e3 = rw_policy
        .resolve_sandboxed_write_path("dangling_escape.tsv", "write_curated_peak_tsv")
        .unwrap_err();
    assert_eq!(e3.kind, McpSecurityErrorKind::SymlinkEscape);

    // Intra-workspace symlink into `.env` rejected as CredentialPathBlocked!
    let e4 = rw_policy
        .resolve_sandboxed_path("harmless.str")
        .unwrap_err();
    assert_eq!(e4.kind, McpSecurityErrorKind::CredentialPathBlocked);
}

#[test]
fn test_adversarial_hidden_credential_directories_blocked() {
    let sb = TempTestSandbox::new("creds");
    for rel in [
        ".ssh/id_rsa",
        ".aws/credentials",
        ".gnupg/secring.gpg",
        ".env",
        ".env.production",
    ] {
        let full = sb.workspace.join(rel);
        if let Some(parent) = full.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(&full, "TOP_SECRET").unwrap();
    }

    let policy = McpSecurityPolicy::read_write([&sb.workspace]).unwrap();
    for rel in [
        ".ssh/id_rsa",
        ".aws/credentials",
        ".gnupg/secring.gpg",
        ".env",
        ".env.production",
    ] {
        let err = policy.resolve_sandboxed_path(rel).unwrap_err();
        assert_eq!(
            err.kind,
            McpSecurityErrorKind::CredentialPathBlocked,
            "expected credential block for {rel}"
        );
    }

    // Also verify `resources/list` never exposes dot/credential files
    let mut server = McpServer::new(policy);
    let raw = handle_mcp_json_request(
        &mut server,
        r#"{"jsonrpc":"2.0","id":1,"method":"resources/list"}"#,
    )
    .unwrap();
    assert!(!raw.contains(".ssh"));
    assert!(!raw.contains(".aws"));
    assert!(!raw.contains(".gnupg"));
    assert!(!raw.contains(".env"));
}

#[test]
fn test_read_only_denial_vs_read_write_sandboxed_mutation_success() {
    let sb = TempTestSandbox::new("rw_guard");
    populate_sample_nmr_workspace(&sb.workspace);

    // 1. Default ReadOnly mode blocks both mutating tools and leaves filesystem untouched
    assert_eq!(McpAccessMode::default(), McpAccessMode::ReadOnly);
    let ro_policy = McpSecurityPolicy::read_only([&sb.workspace]).unwrap();
    let mut ro_server = McpServer::new(ro_policy);

    for (tool_name, args) in [
        (
            "write_curated_peak_tsv",
            json!({
                "output_path": "curated_peaks.tsv",
                "source_peaks_path": "peaks.tsv",
                "exclude_indices": [0]
            }),
        ),
        (
            "export_nmrstar_shifts",
            json!({
                "output_path": "exported.str",
                "sequence": "MQIFVK",
                "shifts": [{ "seq_id": 2, "atom": "H", "val": 8.25, "confidence": 0.95 }]
            }),
        ),
    ] {
        let raw = handle_mcp_json_request(
            &mut ro_server,
            &serde_json::to_string(&json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": { "name": tool_name, "arguments": args }
            }))
            .unwrap(),
        )
        .unwrap();
        let resp: Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(resp["result"]["isError"], true);
        assert_eq!(
            resp["result"]["structuredContent"]["error"]["code"],
            ERR_READ_ONLY_DENIED
        );
    }
    assert!(!sb.workspace.join("curated_peaks.tsv").exists());
    assert!(!sb.workspace.join("exported.str").exists());

    // Optional hiding of write tools in ReadOnly mode
    let hidden_policy = McpSecurityPolicy::read_only([&sb.workspace])
        .unwrap()
        .with_hide_write_tools_in_read_only(true);
    let mut hidden_server = McpServer::new(hidden_policy);
    let list_raw = handle_mcp_json_request(
        &mut hidden_server,
        r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#,
    )
    .unwrap();
    let list_val: Value = serde_json::from_str(&list_raw).unwrap();
    let visible_tools = list_val["result"]["tools"].as_array().unwrap();
    assert_eq!(visible_tools.len(), 5);
    assert!(visible_tools
        .iter()
        .all(|t| t["annotations"]["readOnlyHint"] == true));

    // 2. ReadWrite mode permits sandboxed writes and round-trips cleanly
    let rw_policy = McpSecurityPolicy::read_write([&sb.workspace]).unwrap();
    let mut rw_server = McpServer::new(rw_policy);

    let write_tsv_raw = handle_mcp_json_request(
        &mut rw_server,
        &serde_json::to_string(&json!({
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tools/call",
            "params": {
                "name": "write_curated_peak_tsv",
                "arguments": {
                    "output_path": "curated/filtered_hsqc.tsv",
                    "source_peaks_path": "peaks.tsv",
                    "filter_experiments": ["HSQC"],
                    "exclude_indices": [0]
                }
            }
        }))
        .unwrap(),
    )
    .unwrap();
    let write_tsv_val: Value = serde_json::from_str(&write_tsv_raw).unwrap();
    assert_eq!(write_tsv_val["result"]["isError"], false);
    let written_peaks = write_tsv_val["result"]["structuredContent"]["written_peaks"]
        .as_u64()
        .unwrap();
    assert!(written_peaks >= 60);
    assert!(sb.workspace.join("curated/filtered_hsqc.tsv").exists());

    let export_star_raw = handle_mcp_json_request(
        &mut rw_server,
        &serde_json::to_string(&json!({
            "jsonrpc": "2.0",
            "id": 11,
            "method": "tools/call",
            "params": {
                "name": "export_nmrstar_shifts",
                "arguments": {
                    "output_path": "curated/mini_shifts.str",
                    "sequence": ">mini\nMQIFVK",
                    "shifts": [
                        { "seq_id": 2, "atom": "H", "val": 8.412, "confidence": 0.94 },
                        { "seq_id": 2, "atom": "N", "val": 121.35, "confidence": 0.94 },
                        { "seq_id": 2, "atom": "CA", "val": 55.21, "confidence": 0.94 }
                    ],
                    "header_comment": "Curated via clean-room MCP server"
                }
            }
        }))
        .unwrap(),
    )
    .unwrap();
    let export_star_val: Value = serde_json::from_str(&export_star_raw).unwrap();
    assert_eq!(export_star_val["result"]["isError"], false);
    assert_eq!(
        export_star_val["result"]["structuredContent"]["n_residues"],
        6
    );
    assert_eq!(
        export_star_val["result"]["structuredContent"]["n_shifts_written"],
        3
    );

    // Verify round-trip read via `read_nmrstar_summary`
    let verify_raw = handle_mcp_json_request(
        &mut rw_server,
        &serde_json::to_string(&json!({
            "jsonrpc": "2.0",
            "id": 12,
            "method": "tools/call",
            "params": {
                "name": "read_nmrstar_summary",
                "arguments": { "path": "curated/mini_shifts.str" }
            }
        }))
        .unwrap(),
    )
    .unwrap();
    let verify_val: Value = serde_json::from_str(&verify_raw).unwrap();
    assert_eq!(
        verify_val["result"]["structuredContent"]["one_letter_sequence"],
        "MQIFVK"
    );
    assert_eq!(verify_val["result"]["structuredContent"]["total_shifts"], 3);
}

#[test]
fn test_jsonrpc_error_codes_and_batch_semantics() {
    let mut server = McpServer::new(McpSecurityPolicy::empty_read_only());

    // -32700 Parse error
    let r_parse: Value =
        serde_json::from_str(&handle_mcp_json_request(&mut server, "{not valid json").unwrap())
            .unwrap();
    assert_eq!(r_parse["error"]["code"], ERR_PARSE_ERROR);

    // -32600 Invalid Request cases
    for bad_req in [
        "42",
        "[]",
        r#"{"id":1,"method":"ping"}"#,
        r#"{"jsonrpc":"1.0","id":1,"method":"ping"}"#,
        r#"{"jsonrpc":"2.0","id":[1,2],"method":"ping"}"#,
        r#"{"jsonrpc":"2.0","id":1,"method":999}"#,
        r#"{"jsonrpc":"2.0","id":1,"method":"ping","params":"scalar"}"#,
    ] {
        let resp: Value =
            serde_json::from_str(&handle_mcp_json_request(&mut server, bad_req).unwrap()).unwrap();
        assert_eq!(
            resp["error"]["code"], ERR_INVALID_REQUEST,
            "expected -32600 for {bad_req}"
        );
    }

    // -32601 Method not found
    let r_method: Value = serde_json::from_str(
        &handle_mcp_json_request(
            &mut server,
            r#"{"jsonrpc":"2.0","id":5,"method":"unknown/method"}"#,
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(r_method["error"]["code"], ERR_METHOD_NOT_FOUND);

    // -32602 Invalid params
    for bad_params in [
        json!({"jsonrpc":"2.0","id":6,"method":"tools/call"}),
        json!({"jsonrpc":"2.0","id":7,"method":"tools/call","params":{}}),
        json!({"jsonrpc":"2.0","id":8,"method":"tools/call","params":{"name":"nonexistent_tool"}}),
        json!({"jsonrpc":"2.0","id":9,"method":"tools/call","params":{"name":"read_peak_list_summary","arguments":[1,2]}}),
        json!({"jsonrpc":"2.0","id":10,"method":"resources/read","params":{}}),
        json!({"jsonrpc":"2.0","id":11,"method":"prompts/get","params":{"name":"unknown-prompt"}}),
        json!({"jsonrpc":"2.0","id":12,"method":"prompts/get","params":{"name":"triage-qc","arguments":{}}}),
    ] {
        let resp: Value = serde_json::from_str(
            &handle_mcp_json_request(&mut server, &serde_json::to_string(&bad_params).unwrap())
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            resp["error"]["code"], ERR_INVALID_PARAMS,
            "expected -32602 for {bad_params}"
        );
    }

    // -32003 Resource not found
    let r_res: Value = serde_json::from_str(
        &handle_mcp_json_request(
            &mut server,
            r#"{"jsonrpc":"2.0","id":13,"method":"resources/read","params":{"uri":"nmr://does-not-exist"}}"#,
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(r_res["error"]["code"], ERR_RESOURCE_NOT_FOUND);

    // Mixed batch: 2 requests + 1 notification -> 2 responses
    let batch = json!([
        {"jsonrpc": "2.0", "id": "b1", "method": "ping"},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": "b2", "method": "nope"}
    ]);
    let batch_resp: Value = serde_json::from_str(
        &handle_mcp_json_request(&mut server, &serde_json::to_string(&batch).unwrap()).unwrap(),
    )
    .unwrap();
    let arr = batch_resp.as_array().unwrap();
    assert_eq!(arr.len(), 2);
    assert_eq!(arr[0]["id"], "b1");
    assert_eq!(arr[0]["result"], json!({}));
    assert_eq!(arr[1]["id"], "b2");
    assert_eq!(arr[1]["error"]["code"], ERR_METHOD_NOT_FOUND);
}

#[test]
fn test_client_config_generator_schemas_for_antigravity_claude_and_cursor() {
    let stdio_ro = McpClientMode::Stdio {
        command: "nmrassign".to_string(),
        workspace_roots: vec!["/data/nmr_project".to_string()],
        access_mode: McpAccessMode::ReadOnly,
    };
    let stdio_rw = McpClientMode::Stdio {
        command: "nmrassign".to_string(),
        workspace_roots: vec!["/data/nmr_project".to_string()],
        access_mode: McpAccessMode::ReadWrite,
    };
    let bridge_mode = McpClientMode::HttpBridge {
        endpoint_url: "http://127.0.0.1:51439/mcp".to_string(),
        bearer_token: "tok_secret_987654321".to_string(),
        access_mode: McpAccessMode::ReadOnly,
    };

    for target in [
        McpClientTarget::Antigravity,
        McpClientTarget::ClaudeDesktop,
        McpClientTarget::Cursor,
    ] {
        let cfg_ro = generate_mcp_client_config(target, &stdio_ro).expect("stdio ro config");
        let entry = &cfg_ro["mcpServers"]["nmr-assign"];
        assert_eq!(entry["command"], "nmrassign");
        let args: Vec<&str> = entry["args"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert_eq!(
            args,
            vec![
                "mcp",
                "--transport",
                "stdio",
                "--workspace",
                "/data/nmr_project",
                "--read-only"
            ]
        );

        let cfg_rw = generate_mcp_client_config(target, &stdio_rw).expect("stdio rw config");
        assert!(cfg_rw["mcpServers"]["nmr-assign"]["args"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v == "--read-write"));

        let pretty_http =
            generate_mcp_client_config_pretty(target, &bridge_mode).expect("http bridge config");
        let cfg_http: Value = serde_json::from_str(&pretty_http).unwrap();
        let wb = &cfg_http["mcpServers"]["nmr-assign-workbench"];
        assert_eq!(
            wb["headers"]["Authorization"],
            "Bearer tok_secret_987654321"
        );
        match target {
            McpClientTarget::Antigravity => {
                assert_eq!(wb["serverUrl"], "http://127.0.0.1:51439/mcp");
            }
            McpClientTarget::ClaudeDesktop => {
                assert_eq!(wb["transport"], "streamable-http");
                assert_eq!(wb["url"], "http://127.0.0.1:51439/mcp");
            }
            McpClientTarget::Cursor => {
                assert_eq!(wb["url"], "http://127.0.0.1:51439/mcp");
            }
        }
    }

    // Reject non-loopback HTTP bridge URLs & empty bearer tokens
    assert!(generate_mcp_client_config(
        McpClientTarget::Antigravity,
        &McpClientMode::HttpBridge {
            endpoint_url: "http://0.0.0.0:8090/mcp".to_string(),
            bearer_token: "tok".to_string(),
            access_mode: McpAccessMode::ReadOnly,
        }
    )
    .is_err());
    assert!(generate_mcp_client_config(
        McpClientTarget::Cursor,
        &McpClientMode::HttpBridge {
            endpoint_url: "https://evil.example.com/mcp".to_string(),
            bearer_token: "tok".to_string(),
            access_mode: McpAccessMode::ReadOnly,
        }
    )
    .is_err());
    assert!(generate_mcp_client_config(
        McpClientTarget::ClaudeDesktop,
        &McpClientMode::HttpBridge {
            endpoint_url: "http://127.0.0.1:8090/mcp".to_string(),
            bearer_token: "   ".to_string(),
            access_mode: McpAccessMode::ReadOnly,
        }
    )
    .is_err());
}

#[test]
fn test_multimodal_png_encoder_and_loopback_http_security() {
    let rgba = vec![
        255, 0, 0, 255, 0, 255, 0, 255, // row 0 (2 px)
        0, 0, 255, 255, 255, 255, 255, 255, // row 1 (2 px)
    ];
    let png = encode_rgba_png(2, 2, &rgba).expect("encode 2x2 RGBA PNG");
    assert_eq!(&png[0..8], b"\x89PNG\r\n\x1a\n");
    assert_eq!(&png[12..16], b"IHDR");
    assert!(png.windows(4).any(|w| w == b"IDAT"));
    assert_eq!(&png[png.len() - 8..png.len() - 4], b"IEND");

    let block = McpContentBlock::image_png_from_rgba(2, 2, &rgba).expect("MCP image block");
    let block_json = serde_json::to_value(&block).unwrap();
    assert_eq!(block_json["type"], "image");
    assert_eq!(block_json["mimeType"], "image/png");
    let decoded = decode_base64(block_json["data"].as_str().unwrap()).unwrap();
    assert_eq!(decoded, png);

    #[cfg(not(target_arch = "wasm32"))]
    {
        use nmr_core::mcp::handle_mcp_http_stream;
        use std::io::{Read, Write};
        use std::net::{TcpListener, TcpStream};

        let send_http = |raw_req: &str| -> String {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap();
            let req_owned = raw_req.to_string();
            let client = std::thread::spawn(move || {
                let mut s = TcpStream::connect(addr).unwrap();
                s.write_all(req_owned.as_bytes()).unwrap();
                s.flush().unwrap();
                let mut resp = String::new();
                s.read_to_string(&mut resp).unwrap();
                resp
            });
            let (stream, _) = listener.accept().unwrap();
            let mut srv = McpServer::new(McpSecurityPolicy::empty_read_only());
            handle_mcp_http_stream(&mut srv, stream, "secret-loopback-token").unwrap();
            client.join().unwrap()
        };

        // 1. Missing Authorization -> 401
        let r401 = send_http(
            "POST /mcp HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 36\r\n\r\n{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}",
        );
        assert!(r401.starts_with("HTTP/1.1 401"), "got: {r401}");

        // 2. Cross-origin browser Origin -> 403
        let r403_origin = send_http(
            "POST /mcp HTTP/1.1\r\nHost: 127.0.0.1\r\nOrigin: https://evil.example\r\nAuthorization: Bearer secret-loopback-token\r\nContent-Length: 40\r\n\r\n{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}",
        );
        assert!(
            r403_origin.starts_with("HTTP/1.1 403"),
            "got: {r403_origin}"
        );

        // 3. DNS-rebinding Host -> 403
        let r403_host = send_http(
            "POST /mcp HTTP/1.1\r\nHost: rebind.attacker.example\r\nAuthorization: Bearer secret-loopback-token\r\nContent-Length: 40\r\n\r\n{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}",
        );
        assert!(r403_host.starts_with("HTTP/1.1 403"), "got: {r403_host}");

        // 4. Authenticated loopback request -> 200 OK
        let body = r#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#;
        let r200 = send_http(&format!(
            "POST /mcp HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer secret-loopback-token\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        ));
        assert!(r200.starts_with("HTTP/1.1 200 OK"), "got: {r200}");
        assert!(r200.contains(r#""result":{}"#), "got: {r200}");
    }
}

#[test]
fn test_no_host_path_leakage_in_tool_and_resource_outputs() {
    let sb = TempTestSandbox::new("no_host_leak");
    populate_sample_nmr_workspace(&sb.workspace);
    let base_str = sb.base.to_str().unwrap();

    let rw_policy = McpSecurityPolicy::read_write([&sb.workspace]).unwrap();
    let mut server = McpServer::new(rw_policy);

    for req in [
        json!({"jsonrpc":"2.0","id":1,"method":"resources/read","params":{"uri":"nmr://workspace/policy"}}),
        json!({"jsonrpc":"2.0","id":2,"method":"resources/read","params":{"uri":"nmr://workspace/inventory"}}),
        json!({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"inspect_nmr_directory","arguments":{"path":"."}}}),
        json!({"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"read_nmrstar_summary","arguments":{"path":"synthetic_ubiquitin.str"}}}),
        json!({"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"read_peak_list_summary","arguments":{"path":"peaks.tsv"}}}),
        json!({"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"write_curated_peak_tsv","arguments":{"output_path":"sub/clean.tsv","source_peaks_path":"peaks.tsv"}}}),
    ] {
        let resp =
            handle_mcp_json_request(&mut server, &serde_json::to_string(&req).unwrap()).unwrap();
        assert!(
            !resp.contains(base_str),
            "response leaked host sandbox parent path '{base_str}': {resp}"
        );
    }
}

#[test]
fn test_include_chi1_opt_in_and_strict_20pct_firewall_cutoff() {
    let sb = TempTestSandbox::new("chi1_fw20");
    fs::copy(
        fixture_path("bmr17769_ubiquitin_allatom.str"),
        sb.workspace.join("ubq.str"),
    )
    .unwrap();

    let policy = McpSecurityPolicy::read_only([&sb.workspace]).unwrap();
    let mut server = McpServer::new(policy);

    let raw = handle_mcp_json_request(
        &mut server,
        &serde_json::to_string(&json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "predict_homolog_free_torsions",
                "arguments": {
                    "star_path": "ubq.str",
                    "exclude_pdb_ids": ["1D3Z", "1UBQ"],
                    "exclude_bmrb_ids": ["17769"],
                    "max_seq_identity": 0.20,
                    "include_chi1": true
                }
            }
        }))
        .unwrap(),
    )
    .unwrap();
    let val: Value = serde_json::from_str(&raw).unwrap();
    let sc = &val["result"]["structuredContent"];
    assert_eq!(val["result"]["isError"], false);
    assert_eq!(sc["include_chi1"], true);
    assert!(sc["max_retained_seq_identity"].as_f64().unwrap() <= 0.20 + 1e-9);
    assert!(sc["residues"]
        .as_array()
        .unwrap()
        .iter()
        .any(|r| r["chi1_deg"].is_number()));
}

#[test]
fn test_multi_root_sandbox_and_multiblock_png_encoding() {
    let sb = TempTestSandbox::new("multi_root");
    let root_a = sb.workspace.join("root_a");
    let root_b = sb.workspace.join("root_b");
    fs::create_dir_all(&root_a).unwrap();
    fs::create_dir_all(&root_b).unwrap();
    fs::write(root_a.join("a.fasta"), ">a\nMQIFVK\n").unwrap();
    fs::write(root_b.join("b.fasta"), ">b\nACDEFG\n").unwrap();

    let policy = McpSecurityPolicy::read_only([&root_a, &root_b]).unwrap();
    assert_eq!(policy.allowed_roots().len(), 2);
    let mut server = McpServer::new(policy);

    let r_b = handle_mcp_json_request(
        &mut server,
        r#"{"jsonrpc":"2.0","id":1,"method":"resources/read","params":{"uri":"nmr://file/1/b.fasta"}}"#,
    )
    .unwrap();
    let v_b: Value = serde_json::from_str(&r_b).unwrap();
    assert!(v_b["result"]["contents"][0]["text"]
        .as_str()
        .unwrap()
        .contains("ACDEFG"));

    let r_bad = handle_mcp_json_request(
        &mut server,
        r#"{"jsonrpc":"2.0","id":2,"method":"resources/read","params":{"uri":"nmr://file/5/b.fasta"}}"#,
    )
    .unwrap();
    let v_bad: Value = serde_json::from_str(&r_bad).unwrap();
    assert_eq!(v_bad["error"]["code"], ERR_RESOURCE_NOT_FOUND);

    // Multi-block (>65,535 byte) stored ZLIB PNG test: 160 * 120 * 4 = 76,800 bytes
    let big_rgba = vec![128u8; 160 * 120 * 4];
    let png = encode_rgba_png(160, 120, &big_rgba).expect("multi-block PNG");
    assert_eq!(&png[0..8], b"\x89PNG\r\n\x1a\n");
    assert_eq!(&png[png.len() - 8..png.len() - 4], b"IEND");
    assert!(png.len() > 76_800);
}
