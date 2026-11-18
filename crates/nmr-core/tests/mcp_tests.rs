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
    assert_eq!(lines[0]["result"]["serverInfo"]["name"], "ferric-mcp");

    // Response 1 (id=2): ping
    assert_eq!(lines[1]["id"], 2);
    assert_eq!(lines[1]["result"], json!({}));

    // Response 2 (id=3): tools/list
    let tools = lines[2]["result"]["tools"].as_array().expect("tools array");
    assert_eq!(tools.len(), 12);
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
            "fit-cest-cpmg-exchange",
            "analyze-metabolomics-cohort"
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
        (
            "analyze-metabolomics-cohort",
            json!({"workspace_path": "metab_ws", "cohort_path": "manifest.tsv", "biofluid": "urine"}),
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
    assert_eq!(visible_tools.len(), 10);
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
        command: "ferric".to_string(),
        workspace_roots: vec!["/data/nmr_project".to_string()],
        access_mode: McpAccessMode::ReadOnly,
    };
    let stdio_rw = McpClientMode::Stdio {
        command: "ferric".to_string(),
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
        let entry = &cfg_ro["mcpServers"]["ferric"];
        assert_eq!(entry["command"], "ferric");
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
        assert!(cfg_rw["mcpServers"]["ferric"]["args"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v == "--read-write"));

        let pretty_http =
            generate_mcp_client_config_pretty(target, &bridge_mode).expect("http bridge config");
        let cfg_http: Value = serde_json::from_str(&pretty_http).unwrap();
        let wb = &cfg_http["mcpServers"]["ferric-workbench"];
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

#[test]
fn test_phase13_dynamics_and_csp_mcp_tools_and_sandbox_guards() {
    use nmr_core::cest::{
        simulate_cest_2state_profile, write_cest_profiles_tsv, Cest2StateParams, CestNormalization,
        CestProfileDataset,
    };
    use nmr_core::cpmg::{r2eff_baldwin2014, write_cpmg_series_tsv, CpmgSeriesDataset};
    use nmr_core::peaks::Peak;

    let sb = TempTestSandbox::new("phase13_mcp");

    // 1. Write synthetic R1 relaxation series TSV (2 residues, true R1 = 1.65 & 2.15 s^-1)
    let delays = [0.02, 0.06, 0.12, 0.25, 0.45, 0.70, 1.00, 1.40];
    let mut r1_tsv = String::from(
        "# name: r1_600\n\
         # experiment_kind: R1\n\
         # field_mhz: 600.130000\n\
         # schedule_kind: vdlist\n\
         # duplicate_variance_scale: 1.000000\n\
         # duplicate_dof: 0\n\
         seq_id\tres_name\tatom_h\tatom_x\tdelta_h_ppm\tdelta_x_ppm\tsigma_h_ppm\tsigma_x_ppm\teta_h\teta_x\toverlap_cluster_id\tplane_idx\tx_value\tx_extra\tintensity\tsigma_intensity\tvolume\tplane_mad\tis_duplicate\n",
    );
    for (seq_id, res_name, h_ppm, n_ppm, true_r1) in [
        (10, "VAL", 8.22, 120.5, 1.65_f64),
        (24, "GLU", 8.58, 123.1, 2.15_f64),
    ] {
        for (p_idx, &t) in delays.iter().enumerate() {
            let inten = 1000.0 * (-true_r1 * t).exp();
            r1_tsv.push_str(&format!(
                "{seq_id}\t{res_name}\tH\tN\t{h_ppm:.5}\t{n_ppm:.5}\t0.01800\t0.14000\t0.30000\t0.30000\t0\t{p_idx}\t{t:.9}\t.\t{inten:.6e}\t5.000000e0\t{inten:.6e}\t3.000000e0\tfalse\n"
            ));
        }
    }
    fs::write(sb.workspace.join("r1_600.tsv"), &r1_tsv).unwrap();

    // 2. Write synthetic dual-B1 15N CEST profiles TSV (true p_B = 0.030, k_ex = 180 s^-1, dw = +3.5 ppm)
    let offsets_ppm: Vec<f64> = (-20..=25).map(|i| 118.0 + 0.25 * (i as f64)).collect();
    let mut cest_profiles = Vec::new();
    for b1_hz in [15.0, 30.0] {
        let params = Cest2StateParams {
            p_b: 0.030,
            k_ex_s_inv: 180.0,
            major_ppm: 118.0,
            dw_ab_ppm: 3.5,
            r1a_s_inv: 1.5,
            r2a_s_inv: 12.0,
            r1b_s_inv: 1.5,
            r2b_s_inv: 18.0,
            larmor_mhz: 60.8106,
            t_ex_s: 0.40,
            b1_hz,
            b1_inhomogeneity_frac: 0.0,
            n_b1_quad: 1,
            normalization: CestNormalization::ZeroTimePhaseCycled,
        };
        let intensities = simulate_cest_2state_profile(&params, &offsets_ppm).unwrap();
        cest_profiles.push(CestProfileDataset {
            seq_id: 42,
            res_name: "LEU".to_string(),
            larmor_mhz: 60.8106,
            t_ex_s: 0.40,
            b1_hz,
            b1_inhomogeneity_frac: 0.0,
            normalization: CestNormalization::ZeroTimePhaseCycled,
            major_ppm_hint: Some(118.0),
            offsets_ppm: offsets_ppm.clone(),
            uncertainties: vec![0.002; offsets_ppm.len()],
            intensities,
        });
    }
    fs::write(
        sb.workspace.join("cest_dual_b1.tsv"),
        write_cest_profiles_tsv(&cest_profiles),
    )
    .unwrap();

    // 3. Write synthetic dual-field 15N CPMG dispersion TSV (true p_B = 0.040, k_ex = 900 s^-1, dw = 3.2 ppm)
    let nu_cpmg: Vec<f64> = vec![
        50.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0, 650.0, 800.0, 1000.0,
    ];
    let mut cpmg_series = Vec::new();
    for (larmor_mhz, r20) in [(60.8106_f64, 11.5_f64), (81.0808_f64, 13.5_f64)] {
        let dw_rad_s = 2.0 * std::f64::consts::PI * larmor_mhz * 3.2;
        let r2_eff: Vec<f64> = nu_cpmg
            .iter()
            .map(|&nu| r2eff_baldwin2014(r20, r20, 0.040, 900.0, dw_rad_s, 0.040, nu))
            .collect();
        cpmg_series.push(CpmgSeriesDataset {
            seq_id: 55,
            res_name: "ALA".to_string(),
            larmor_mhz,
            t_relax_s: 0.040,
            nu_cpmg_hz: nu_cpmg.clone(),
            r2_eff_err: vec![0.12; nu_cpmg.len()],
            r2_eff_s_inv: r2_eff,
        });
    }
    fs::write(
        sb.workspace.join("cpmg_dual_field.tsv"),
        write_cpmg_series_tsv(&cpmg_series),
    )
    .unwrap();

    // 4. Write synthetic 2D HSQC CSP titration series (true global K_d = 30.0 uM, [P]_T = 50.0 uM)
    let apo_rows = [
        (12, "ILE", 8.10_f64, 119.0_f64, 0.24_f64, 1.40_f64),
        (18, "PHE", 8.65_f64, 122.8_f64, -0.18_f64, -1.10_f64),
        (31, "GLY", 7.82_f64, 109.4_f64, 0.01_f64, 0.04_f64),
        (44, "TYR", 9.05_f64, 126.2_f64, 0.21_f64, -0.95_f64),
    ];
    let mut apo_tsv = String::from("seq_id\tres_name\th_ppm\tn_ppm\n");
    for (sid, rname, h0, n0, _, _) in apo_rows {
        apo_tsv.push_str(&format!("{sid}\t{rname}\t{h0:.5}\t{n0:.5}\n"));
    }
    fs::write(sb.workspace.join("apo_hsqc.tsv"), &apo_tsv).unwrap();

    let p_tot = 50.0_f64;
    let true_kd = 30.0_f64;
    let lig_concs = [0.0_f64, 15.0, 35.0, 70.0, 140.0, 280.0];
    let mut step_specs = Vec::new();
    for (idx, &l_tot) in lig_concs.iter().enumerate() {
        let sum = p_tot + l_tot + true_kd;
        let fb = if l_tot <= 0.0 {
            0.0
        } else {
            (sum - (sum * sum - 4.0 * p_tot * l_tot).max(0.0).sqrt()) / (2.0 * p_tot)
        };
        let step_peaks: Vec<Peak> = apo_rows
            .iter()
            .map(|&(_, _, h0, n0, dh, dn)| Peak::hsqc(h0 + fb * dh, n0 + fb * dn))
            .collect();
        let fname = format!("titration_step_{idx}.tsv");
        write_peaks(sb.workspace.join(&fname), &step_peaks).unwrap();
        step_specs.push(json!({
            "label": format!("L_{l_tot:.0}uM"),
            "peaks_path": fname,
            "protein_conc_um": p_tot,
            "ligand_conc_um": l_tot
        }));
    }

    let policy = McpSecurityPolicy::read_only([&sb.workspace]).unwrap();
    let mut server = McpServer::new(policy);

    let call_tool = |srv: &mut McpServer, name: &str, args: Value| -> Value {
        let raw = handle_mcp_json_request(
            srv,
            &serde_json::to_string(&json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": { "name": name, "arguments": args }
            }))
            .unwrap(),
        )
        .unwrap();
        serde_json::from_str(&raw).unwrap()
    };

    // Verify relaxation fit
    let r1_resp = call_tool(
        &mut server,
        "fit_relaxation_or_exchange",
        json!({ "mode": "relaxation", "path": "r1_600.tsv", "mc_replicates": 100 }),
    );
    assert_eq!(r1_resp["result"]["isError"], false, "got: {r1_resp}");
    let r1_rows = r1_resp["result"]["structuredContent"]["residues"]
        .as_array()
        .unwrap();
    assert_eq!(r1_rows.len(), 2);
    let r1_first = r1_rows[0]["rate_s_inv"].as_f64().unwrap();
    assert!((r1_first - 1.65).abs() < 0.02, "r1_first={r1_first}");

    // Verify CEST fit
    let cest_resp = call_tool(
        &mut server,
        "fit_relaxation_or_exchange",
        json!({ "mode": "cest", "path": "cest_dual_b1.tsv" }),
    );
    assert_eq!(cest_resp["result"]["isError"], false, "got: {cest_resp}");
    let cest_sc = &cest_resp["result"]["structuredContent"];
    assert!(
        cest_sc["reduced_chi2"].as_f64().unwrap() < 1.0,
        "cest reduced_chi2={}",
        cest_sc["reduced_chi2"]
    );
    let cest_pb = cest_sc["p_b"].as_f64().unwrap();
    let cest_kex = cest_sc["k_ex_s_inv"].as_f64().unwrap();
    assert!((cest_pb - 0.030).abs() < 0.005, "cest_pb={cest_pb}");
    assert!((cest_kex - 180.0).abs() < 25.0, "cest_kex={cest_kex}");

    // Verify CPMG fit
    let cpmg_resp = call_tool(
        &mut server,
        "fit_relaxation_or_exchange",
        json!({ "mode": "cpmg", "path": "cpmg_dual_field.tsv", "cpmg_model": "baldwin2014" }),
    );
    assert_eq!(cpmg_resp["result"]["isError"], false, "got: {cpmg_resp}");
    let cpmg_sc = &cpmg_resp["result"]["structuredContent"];
    assert!(
        cpmg_sc["reduced_chi2"].as_f64().unwrap() < 1.0,
        "cpmg reduced_chi2={}",
        cpmg_sc["reduced_chi2"]
    );
    let cpmg_pb = cpmg_sc["p_b"].as_f64().unwrap();
    let cpmg_kex = cpmg_sc["k_ex_s_inv"].as_f64().unwrap();
    assert!((cpmg_pb - 0.040).abs() < 0.008, "cpmg_pb={cpmg_pb}");
    assert!((cpmg_kex - 900.0).abs() < 90.0, "cpmg_kex={cpmg_kex}");

    // Verify CSP global K_d fit
    let csp_resp = call_tool(
        &mut server,
        "track_csp_titration",
        json!({
            "apo_peaks_path": "apo_hsqc.tsv",
            "steps": step_specs,
            "n_mc": 100
        }),
    );
    assert_eq!(csp_resp["result"]["isError"], false, "got: {csp_resp}");
    let csp_kd = csp_resp["result"]["structuredContent"]["global_kd_um"]
        .as_f64()
        .unwrap();
    assert!((csp_kd - 30.0).abs() < 2.0, "csp_kd={csp_kd}");

    // Adversarial sandbox checks on Phase 13 tools
    let bad_relax = call_tool(
        &mut server,
        "fit_relaxation_or_exchange",
        json!({ "mode": "cest", "paths": ["cest_dual_b1.tsv", "../outside/evil.tsv"] }),
    );
    assert_eq!(bad_relax["result"]["isError"], true);
    assert_eq!(
        bad_relax["result"]["structuredContent"]["error"]["code"],
        ERR_SECURITY_VIOLATION
    );

    let bad_csp = call_tool(
        &mut server,
        "track_csp_titration",
        json!({
            "apo_peaks_path": "apo_hsqc.tsv",
            "steps": [{ "peaks_path": "../outside/evil_step.tsv", "protein_conc_um": 50.0, "ligand_conc_um": 20.0 }]
        }),
    );
    assert_eq!(bad_csp["result"]["isError"], true);
    assert_eq!(
        bad_csp["result"]["structuredContent"]["error"]["code"],
        ERR_SECURITY_VIOLATION
    );

    println!(
        "MCP_METRIC_JSON:{}",
        json!({
            "suite": "nmr_core_mcp_phase13_tools",
            "r1_rate_s_inv": r1_first,
            "cest_pb": cest_pb,
            "cest_kex_s_inv": cest_kex,
            "cpmg_pb": cpmg_pb,
            "cpmg_kex_s_inv": cpmg_kex,
            "csp_global_kd_um": csp_kd
        })
    );
}

#[test]
fn test_phase14_nmr1d_dosy_and_metabolomics_mcp_tools_and_manifest_sandbox() {
    let sb = TempTestSandbox::new("phase14_mcp");

    // 1. Synthesize 500 MHz 1H 1D spectrum in CDCl3: CHCl3 singlet (7.26 ppm) + quartet (4.12 ppm, J=7.1 Hz) + triplet (1.26 ppm, J=7.1 Hz)
    let n_pts = 4096usize;
    let mhz = 500.0_f64;
    let j_ppm = 7.1 / mhz;
    let hwhm_ppm = 0.9 / mhz;
    let lorentz = |x: f64, x0: f64, amp: f64| -> f64 {
        let dx = (x - x0) / hwhm_ppm;
        amp / (1.0 + dx * dx)
    };
    let mut spec1d_tsv = String::from("# mhz=500.0, solvent=CDCl3\nppm\treal\n");
    for i in 0..n_pts {
        let ppm = 8.5 - 8.0 * (i as f64) / ((n_pts - 1) as f64);
        let mut y = 0.002 * ((i as f64 * 1.37).sin());
        // Residual CHCl3 trace impurity at 7.26 ppm
        y += lorentz(ppm, 7.26, 12.0);
        // Quartet at 4.12 ppm (1:3:3:1)
        for (k, w) in [(-1.5, 25.0), (-0.5, 75.0), (0.5, 75.0), (1.5, 25.0)] {
            y += lorentz(ppm, 4.12 + k * j_ppm, w);
        }
        // Triplet at 1.26 ppm (1:2:1)
        for (k, w) in [(-1.0, 50.0), (0.0, 100.0), (1.0, 50.0)] {
            y += lorentz(ppm, 1.26 + k * j_ppm, w);
        }
        spec1d_tsv.push_str(&format!("{ppm:.6}\t{y:.6}\n"));
    }
    fs::write(sb.workspace.join("ethyl_cdcl3.tsv"), &spec1d_tsv).unwrap();

    // 2. Synthesize 16-point DOSY dataset JSON (bi-exponential 50:50 mixture + pure monomer)
    let b_vals: Vec<f64> = (0..16)
        .map(|i| 1.0e7 + (3.2e9 - 1.0e7) * ((i as f64) / 15.0).powi(2))
        .collect();
    let mix_intensities: Vec<f64> = b_vals
        .iter()
        .map(|&b| 50.0 * (-b * 2.0e-10).exp() + 50.0 * (-b * 1.3e-9).exp())
        .collect();
    let pure_intensities: Vec<f64> = b_vals
        .iter()
        .map(|&b| 100.0 * (-b * 6.0e-10).exp())
        .collect();
    let dosy_json = json!({
        "b_values_s_m2": b_vals,
        "peaks": [
            { "label": "overlapped_3_65ppm", "center_ppm": 3.65, "intensities": mix_intensities },
            { "label": "pure_2_10ppm", "center_ppm": 2.10, "intensities": pure_intensities }
        ]
    });
    fs::write(
        sb.workspace.join("dosy_mixture.json"),
        serde_json::to_string_pretty(&dosy_json).unwrap(),
    )
    .unwrap();

    // 3. Synthesize 8-sample 1D metabolomics cohort (4 Control + 4 Case) + manifest TSV inside `metab/`
    let metab_dir = sb.workspace.join("metab");
    fs::create_dir_all(&metab_dir).unwrap();
    let mut manifest = String::from("sample_id\tgroup\tpath\n");
    let n_metab_pts = 1024usize;
    let w_ppm = 0.004_f64;
    let peak = |x: f64, x0: f64, amp: f64| -> f64 {
        let d = (x - x0) / w_ppm;
        amp / (1.0 + d * d)
    };
    for s_idx in 0..8usize {
        let is_case = s_idx >= 4;
        let group = if is_case { "Case" } else { "Control" };
        let sample_id = format!("S{:02}", s_idx + 1);
        let dilution = if s_idx % 2 == 0 { 1.0 } else { 1.75 };
        let bio_level = if is_case {
            85.0 + 6.0 * ((s_idx - 4) as f64)
        } else {
            15.0 + 3.0 * (s_idx as f64)
        };
        let mut tsv = String::from("# mhz=600.0, solvent=D2O\nppm\treal\n");
        for i in 0..n_metab_pts {
            let ppm = 9.0 - 8.5 * (i as f64) / ((n_metab_pts - 1) as f64);
            let mut y = 0.02 * (((i + s_idx * 17) as f64 * 0.71).sin());
            // Housekeeping endogenous metabolites (scaled by sample dilution -> normalized by PQN)
            for (c_ppm, c_amp) in [
                (3.04, 120.0),
                (4.05, 80.0),
                (2.55, 65.0),
                (2.69, 65.0),
                (7.55, 50.0),
                (7.84, 50.0),
                (3.26, 70.0),
                (3.92, 55.0),
            ] {
                y += dilution * peak(ppm, c_ppm, c_amp);
            }
            // Case-elevated covarying spin-system pair at 1.33 ppm & 4.11 ppm (lactate-like)
            y += dilution * peak(ppm, 1.33, bio_level);
            y += dilution * peak(ppm, 4.11, 0.34 * bio_level);
            tsv.push_str(&format!("{ppm:.5}\t{y:.5}\n"));
        }
        let rel_file = format!("{sample_id}.tsv");
        fs::write(metab_dir.join(&rel_file), &tsv).unwrap();
        manifest.push_str(&format!("{sample_id}\t{group}\t{rel_file}\n"));
    }
    fs::write(metab_dir.join("manifest.tsv"), &manifest).unwrap();

    // Adversarial manifests attempting path traversal and `.env` exfiltration
    fs::write(sb.outside.join("secret_spec.tsv"), "ppm\treal\n1.0\t1.0\n").unwrap();
    fs::write(sb.workspace.join(".env"), "SECRET_TOKEN=123").unwrap();
    fs::write(
        metab_dir.join("evil_traversal_manifest.tsv"),
        "sample_id\tgroup\tpath\nS01\tControl\t../../outside/secret_spec.tsv\n",
    )
    .unwrap();
    fs::write(
        metab_dir.join("evil_dotenv_manifest.tsv"),
        "sample_id\tgroup\tpath\nS01\tControl\t../.env\n",
    )
    .unwrap();

    let policy = McpSecurityPolicy::read_only([&sb.workspace]).unwrap();
    let mut server = McpServer::new(policy);

    let call_tool = |srv: &mut McpServer, name: &str, args: Value| -> Value {
        let raw = handle_mcp_json_request(
            srv,
            &serde_json::to_string(&json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": { "name": name, "arguments": args }
            }))
            .unwrap(),
        )
        .unwrap();
        serde_json::from_str(&raw).unwrap()
    };

    // Verify 1D NMR analysis + multiplet + ACS report
    let nmr1d_resp = call_tool(
        &mut server,
        "analyze_nmr1d_spectrum",
        json!({
            "path": "ethyl_cdcl3.tsv",
            "solvent": "CDCl3",
            "deconvolve": true,
            "match_impurities": true,
            "journal_style": "acs",
            "multiplet_regions": [
                { "low_ppm": 4.05, "high_ppm": 4.19, "proton_count": 2, "label": "OCH2" },
                { "low_ppm": 1.20, "high_ppm": 1.32, "proton_count": 3, "label": "CH3" }
            ]
        }),
    );
    assert_eq!(nmr1d_resp["result"]["isError"], false, "got: {nmr1d_resp}");
    let sc1d = &nmr1d_resp["result"]["structuredContent"];
    assert!(sc1d["picked_peak_count"].as_u64().unwrap() >= 8);
    assert!(sc1d["deconvolved_peak_count"].as_u64().unwrap() >= 8);
    assert!(!sc1d["impurity_matches"].as_array().unwrap().is_empty());
    let mults = sc1d["multiplets"].as_array().unwrap();
    assert_eq!(mults.len(), 2);
    assert_eq!(mults[0]["class"], "Quartet");
    assert_eq!(mults[1]["class"], "Triplet");
    let j_q = mults[0]["couplings_hz"][0].as_f64().unwrap();
    assert!((j_q - 7.1).abs() < 0.25, "quartet J={j_q}");
    let acs_str = sc1d["journal_report_ascii"].as_str().unwrap();
    assert!(
        acs_str.contains("1H NMR (500 MHz, CDCl3)"),
        "got: {acs_str}"
    );

    // Verify DOSY BIC model selection + hydrodynamics + opt-in ILT caveat
    let dosy_resp = call_tool(
        &mut server,
        "fit_dosy_diffusion",
        json!({
            "path": "dosy_mixture.json",
            "solvent": "D2O",
            "max_components": 3,
            "bootstrap_samples": 24,
            "include_ilt": true
        }),
    );
    assert_eq!(dosy_resp["result"]["isError"], false, "got: {dosy_resp}");
    let sc_dosy = &dosy_resp["result"]["structuredContent"];
    let d_peaks = sc_dosy["peaks"].as_array().unwrap();
    assert_eq!(d_peaks[0]["selected_k"], 2);
    assert_eq!(d_peaks[1]["selected_k"], 1);
    assert!(
        d_peaks[0]["components"][0]["r_h_angstrom"]
            .as_f64()
            .unwrap()
            > 0.5
    );
    assert!(sc_dosy["ilt_caveat"]
        .as_str()
        .unwrap()
        .contains("ill-posed"));

    // Verify Metabolomics Chemometrics (AI-Binning + PQN + PCA + OPLS-DA + STOCSY)
    let metab_resp = call_tool(
        &mut server,
        "run_metabolomics_chemometrics",
        json!({
            "path": "metab/manifest.tsv",
            "mask_preset": "urine",
            "bucketing": "adaptive-intelligent",
            "normalization": "pqn",
            "scaling": "pareto",
            "cv_folds": 4,
            "permutations": 40,
            "stocsy_driver_ppm": 1.33
        }),
    );
    assert_eq!(metab_resp["result"]["isError"], false, "got: {metab_resp}");
    let sc_metab = &metab_resp["result"]["structuredContent"];
    assert_eq!(sc_metab["n_samples"], 8);
    let q2 = sc_metab["opls_da"]["q2"].as_f64().unwrap();
    assert!(q2 > 0.75, "expected high OPLS-DA Q2, got {q2}");
    assert_eq!(sc_metab["opls_da"]["is_validated"], true);
    let top_ppm = sc_metab["top_biomarkers"][0]["center_ppm"]
        .as_f64()
        .unwrap();
    assert!(
        (top_ppm - 1.33).abs() < 0.05 || (top_ppm - 4.11).abs() < 0.05,
        "expected top biomarker near 1.33 or 4.11 ppm, got {top_ppm}"
    );
    let stocsy_hits = sc_metab["stocsy"]["covarying_buckets_r2_ge_0_50"]
        .as_array()
        .unwrap();
    assert!(
        stocsy_hits
            .iter()
            .any(|h| (h["center_ppm"].as_f64().unwrap() - 4.11).abs() < 0.05
                && h["r2"].as_f64().unwrap() > 0.90),
        "expected STOCSY correlation between 1.33 and 4.11 ppm, got {stocsy_hits:?}"
    );

    // Verify adversarial manifest traversal & `.env` exfiltration are blocked by sandbox!
    for evil_manifest in [
        "metab/evil_traversal_manifest.tsv",
        "metab/evil_dotenv_manifest.tsv",
    ] {
        let evil_resp = call_tool(
            &mut server,
            "run_metabolomics_chemometrics",
            json!({ "path": evil_manifest }),
        );
        assert_eq!(evil_resp["result"]["isError"], true, "got: {evil_resp}");
        assert_eq!(
            evil_resp["result"]["structuredContent"]["error"]["code"],
            ERR_SECURITY_VIOLATION
        );
    }

    let stocsy_411_r2 = stocsy_hits
        .iter()
        .find(|h| (h["center_ppm"].as_f64().unwrap() - 4.11).abs() < 0.05)
        .and_then(|h| h["r2"].as_f64())
        .unwrap_or(0.0);
    println!(
        "MCP_METRIC_JSON:{}",
        json!({
            "suite": "nmr_core_mcp_phase14_tools",
            "nmr1d_quartet_j_hz": j_q,
            "dosy_mix_selected_k": d_peaks[0]["selected_k"].as_u64().unwrap(),
            "dosy_pure_selected_k": d_peaks[1]["selected_k"].as_u64().unwrap(),
            "metab_n_samples": sc_metab["n_samples"].as_u64().unwrap(),
            "metab_opls_q2": q2,
            "metab_top_biomarker_ppm": top_ppm,
            "metab_stocsy_r2": stocsy_411_r2
        })
    );
}

#[test]
fn test_adversarial_prealloc_content_length_413_and_non_loopback_bind_rejection() {
    use nmr_core::mcp::{
        parse_bounded_content_length, parse_loopback_bind_addr, ContentLengthError,
        MAX_MCP_FRAME_BYTES,
    };

    // 1. Pre-allocation `Content-Length` parser rejects negative/malformed and `usize` overflow
    assert_eq!(
        parse_bounded_content_length("-42", MAX_MCP_FRAME_BYTES),
        Err(ContentLengthError::Malformed)
    );
    assert_eq!(
        parse_bounded_content_length("12abc", MAX_MCP_FRAME_BYTES),
        Err(ContentLengthError::Malformed)
    );
    assert_eq!(
        parse_bounded_content_length("999999999999999999999999999999999999", MAX_MCP_FRAME_BYTES),
        Err(ContentLengthError::ExceedsLimit {
            max_bytes: MAX_MCP_FRAME_BYTES
        })
    );
    assert_eq!(
        parse_bounded_content_length(&(MAX_MCP_FRAME_BYTES + 1).to_string(), MAX_MCP_FRAME_BYTES),
        Err(ContentLengthError::ExceedsLimit {
            max_bytes: MAX_MCP_FRAME_BYTES
        })
    );

    // 2. `run_mcp_stdio_loop` rejects oversized/overflowing `Content-Length` strictly before allocating
    let mut server = McpServer::new(McpSecurityPolicy::empty_read_only());
    let mut out = Vec::new();
    let stdio_err = run_mcp_stdio_loop(
        &mut server,
        Cursor::new(b"Content-Length: 999999999999999999999999999999\r\n\r\n{}".to_vec()),
        &mut out,
    )
    .expect_err("oversized stdio Content-Length must fail before allocation");
    assert_eq!(stdio_err.kind(), std::io::ErrorKind::InvalidData);

    // 3. `parse_loopback_bind_addr` rejects non-loopback binds (`0.0.0.0`, `[::]`, LAN/WAN IPs)
    for forbidden_bind in [
        "0.0.0.0:8765",
        "[::]:8765",
        "192.168.1.25:8765",
        "10.0.0.12:8765",
        "8.8.8.8:8765",
        "evil.example.com:8765",
    ] {
        assert!(
            parse_loopback_bind_addr(forbidden_bind).is_err(),
            "expected non-loopback bind '{forbidden_bind}' to be rejected"
        );
    }
    assert!(parse_loopback_bind_addr("127.0.0.1:0").is_ok());
    assert!(parse_loopback_bind_addr("[::1]:0").is_ok());
    assert!(parse_loopback_bind_addr("localhost:8765").is_ok());

    // 4. Loopback HTTP stream returns 413 on oversized/overflowing Content-Length, 400 on negative, 403 on duplicate Host
    #[cfg(not(target_arch = "wasm32"))]
    {
        use nmr_core::mcp::handle_mcp_http_stream;
        use std::io::{Read, Write};
        use std::net::{TcpListener, TcpStream};

        let send_raw = |raw_req: &str| -> String {
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
            handle_mcp_http_stream(&mut srv, stream, "tok-413").unwrap();
            client.join().unwrap()
        };

        let r413 = send_raw(
            "POST /mcp HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer tok-413\r\nContent-Length: 999999999999999999999999999999\r\n\r\n{}",
        );
        assert!(
            r413.starts_with("HTTP/1.1 413"),
            "expected 413, got: {r413}"
        );

        let r400 = send_raw(
            "POST /mcp HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer tok-413\r\nContent-Length: -5\r\n\r\n{}",
        );
        assert!(
            r400.starts_with("HTTP/1.1 400"),
            "expected 400, got: {r400}"
        );

        let r403_dup = send_raw(
            "POST /mcp HTTP/1.1\r\nHost: 127.0.0.1\r\nHost: attacker.example\r\nAuthorization: Bearer tok-413\r\nContent-Length: 2\r\n\r\n{}",
        );
        assert!(
            r403_dup.starts_with("HTTP/1.1 403"),
            "expected 403 on duplicate Host, got: {r403_dup}"
        );
    }

    println!(
        "MCP_METRIC_JSON:{}",
        json!({
            "suite": "nmr_core_mcp_tests",
            "protocol_version": MCP_PROTOCOL_VERSION,
            "core_tools_total": 12,
            "core_tools_read_only": 10,
            "core_tools_mutating": 2,
            "guided_prompts_total": 5,
            "max_stdio_frame_bytes": MAX_MCP_FRAME_BYTES,
            "max_http_body_bytes": MAX_MCP_FRAME_BYTES,
            "http_413_prealloc_verified": true,
            "non_loopback_bind_rejected": true,
            "sandbox_escape_vectors_blocked": 9
        })
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_f25_split_send_loopback_http_rejected_401_403_413_zero_econnreset() {
    use nmr_core::mcp::handle_mcp_http_stream;
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::time::Duration;

    let run_split_send = |headers: String, body: Vec<u8>, inter_send_pause_ms: u64| -> String {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral loopback");
        let addr = listener.local_addr().expect("local_addr");
        let srv_handle = std::thread::spawn(move || {
            let (stream, _) = listener.accept().expect("accept");
            let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
            let _ = stream.set_write_timeout(Some(Duration::from_secs(5)));
            let mut srv = McpServer::new(McpSecurityPolicy::empty_read_only());
            handle_mcp_http_stream(&mut srv, stream, "valid-32-char-secret-bearer-token-01")
                .expect("handle_mcp_http_stream");
        });

        let mut client = TcpStream::connect(addr).expect("connect");
        client.set_nodelay(true).expect("TCP_NODELAY");
        client
            .set_read_timeout(Some(Duration::from_secs(5)))
            .expect("client read timeout");
        client
            .set_write_timeout(Some(Duration::from_secs(5)))
            .expect("client write timeout");

        // Split syscall #1: HTTP request line + headers only
        client
            .write_all(headers.as_bytes())
            .expect("split send #1 (headers)");
        client.flush().expect("flush headers");

        if inter_send_pause_ms > 0 {
            std::thread::sleep(Duration::from_millis(inter_send_pause_ms));
        }

        // Split syscall #2: JSON-RPC request body (mirrors Python http.client / requests split send)
        if !body.is_empty() {
            client
                .write_all(&body)
                .expect("split send #2 (body must not fail with BrokenPipe)");
            client.flush().expect("flush body");
        }

        let mut resp = String::new();
        client
            .read_to_string(&mut resp)
            .expect("reading HTTP rejection must never fail with ECONNRESET / ConnectionReset");
        srv_handle.join().expect("server thread join");
        resp
    };

    let big_pad = "x".repeat(32 * 1024);
    let json_body = serde_json::to_vec(&json!({
        "jsonrpc": "2.0",
        "id": 99,
        "method": "initialize",
        "params": { "padding": big_pad }
    }))
    .unwrap();

    for pause_ms in [0_u64, 8_u64] {
        // 1. Split-send 401 Unauthorized (missing Bearer token & invalid Bearer token)
        for auth_hdr in ["", "Authorization: Bearer wrong-bearer-token\r\n"] {
            let hdr_401 = format!(
                "POST /mcp HTTP/1.1\r\nHost: 127.0.0.1\r\n{auth_hdr}Content-Type: application/json\r\nContent-Length: {}\r\n\r\n",
                json_body.len()
            );
            let resp_401 = run_split_send(hdr_401, json_body.clone(), pause_ms);
            assert!(
                resp_401.starts_with("HTTP/1.1 401 Unauthorized\r\n"),
                "expected clean 401 on pause={pause_ms}ms, got: {resp_401}"
            );
            assert!(resp_401.contains("\"error\""));
        }

        // 2. Split-send 403 Forbidden (DNS-rebinding Host, duplicate Host, cross-origin Origin)
        for bad_hdrs in [
            format!(
                "POST /mcp HTTP/1.1\r\nHost: attacker.example\r\nAuthorization: Bearer valid-32-char-secret-bearer-token-01\r\nContent-Length: {}\r\n\r\n",
                json_body.len()
            ),
            format!(
                "POST /mcp HTTP/1.1\r\nHost: 127.0.0.1\r\nHost: evil.example\r\nAuthorization: Bearer valid-32-char-secret-bearer-token-01\r\nContent-Length: {}\r\n\r\n",
                json_body.len()
            ),
            format!(
                "POST /mcp HTTP/1.1\r\nHost: 127.0.0.1\r\nOrigin: https://evil.example\r\nAuthorization: Bearer valid-32-char-secret-bearer-token-01\r\nContent-Length: {}\r\n\r\n",
                json_body.len()
            ),
        ] {
            let resp_403 = run_split_send(bad_hdrs, json_body.clone(), pause_ms);
            assert!(
                resp_403.starts_with("HTTP/1.1 403 Forbidden\r\n"),
                "expected clean 403 on pause={pause_ms}ms, got: {resp_403}"
            );
            assert!(resp_403.contains("\"error\""));
        }

        // 3. Split-send 413 Payload Too Large (Content-Length > MAX_MCP_FRAME_BYTES + body chunk)
        let hdr_413 = format!(
            "POST /mcp HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer valid-32-char-secret-bearer-token-01\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            nmr_core::mcp::MAX_MCP_FRAME_BYTES + 1024 * 1024
        );
        let resp_413 = run_split_send(hdr_413, vec![b'A'; 32 * 1024], pause_ms);
        assert!(
            resp_413.starts_with("HTTP/1.1 413 Payload Too Large\r\n"),
            "expected clean 413 on pause={pause_ms}ms, got: {resp_413}"
        );
        assert!(resp_413.contains("MAX_MCP_FRAME_BYTES"));
    }
}
