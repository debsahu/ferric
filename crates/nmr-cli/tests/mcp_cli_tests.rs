//! Subprocess E2E & adversarial CLI integration tests for `nmrassign mcp` and
//! `nmrassign mcp-config` (`T15.1` & `T15.3`).
//!
//! Provenance: Clean-room verification suite against the Model Context Protocol (`MCP`)
//! JSON-RPC 2.0 specification (`2024-11-05` / `2025-03-26`) and RFC 9110 HTTP loopback rules.

use std::env;
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use serde_json::{json, Value};

use nmr_core::mcp::{ERR_READ_ONLY_DENIED, ERR_SECURITY_VIOLATION, MCP_PROTOCOL_VERSION};
use nmr_core::peaks::{write_peaks, Peak};

static CLI_TMP_SEQ: AtomicU64 = AtomicU64::new(1);

fn nmrassign_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_nmrassign"))
}

struct CliSandbox {
    base: PathBuf,
    workspace: PathBuf,
    outside: PathBuf,
}

impl CliSandbox {
    fn new(tag: &str) -> Self {
        let seq = CLI_TMP_SEQ.fetch_add(1, Ordering::Relaxed);
        let base = env::temp_dir().join(format!("nmr_mcp_cli_{tag}_{}_{seq}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        let workspace = base.join("ws");
        let outside = base.join("outside");
        fs::create_dir_all(&workspace).expect("create ws");
        fs::create_dir_all(&outside).expect("create outside");
        Self {
            base,
            workspace,
            outside,
        }
    }
}

impl Drop for CliSandbox {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.base);
    }
}

#[test]
fn test_cli_mcp_stdio_read_only_and_read_write_subprocess_roundtrip() {
    let sb = CliSandbox::new("stdio_e2e");
    let peaks = vec![
        Peak::hsqc(8.22, 120.4),
        Peak::hsqc(8.51, 122.9),
        Peak::hsqc(7.94, 118.1),
    ];
    write_peaks(sb.workspace.join("peaks.tsv"), &peaks).expect("write peaks.tsv");
    fs::write(sb.outside.join("secret.str"), "TOP_SECRET").expect("write outside secret");

    // 1. Subprocess in default `--mode read-only`: hides mutating tools in `tools/list`, denies mutating calls & traversal
    let mut child = Command::new(nmrassign_bin())
        .args([
            "mcp",
            "--transport",
            "stdio",
            "--root",
            sb.workspace.to_str().expect("utf8 path"),
            "--mode",
            "read-only",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn nmrassign mcp stdio");

    {
        let stdin = child.stdin.as_mut().expect("child stdin");
        for req in [
            json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": { "protocolVersion": MCP_PROTOCOL_VERSION }
            }),
            json!({
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }),
            json!({
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list"
            }),
            json!({
                "jsonrpc": "2.0",
                "id": 3,
                "method": "prompts/list"
            }),
            json!({
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "read_peak_list_summary",
                    "arguments": { "path": "peaks.tsv" }
                }
            }),
            json!({
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {
                    "name": "write_curated_peak_tsv",
                    "arguments": {
                        "output_path": "blocked.tsv",
                        "source_peaks_path": "peaks.tsv"
                    }
                }
            }),
            json!({
                "jsonrpc": "2.0",
                "id": 6,
                "method": "tools/call",
                "params": {
                    "name": "read_nmrstar_summary",
                    "arguments": { "path": "../outside/secret.str" }
                }
            }),
        ] {
            writeln!(stdin, "{}", serde_json::to_string(&req).expect("json")).expect("write line");
        }
    }

    let out = child.wait_with_output().expect("wait child");
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let lines: Vec<Value> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(|l| serde_json::from_str(l).expect("parse jsonrpc line"))
        .collect();
    assert_eq!(lines.len(), 6);
    assert_eq!(lines[0]["result"]["protocolVersion"], MCP_PROTOCOL_VERSION);
    let ro_tools = lines[1]["result"]["tools"].as_array().expect("tools");
    assert_eq!(ro_tools.len(), 10);
    let prompts = lines[2]["result"]["prompts"].as_array().expect("prompts");
    assert_eq!(prompts.len(), 5);
    assert_eq!(lines[3]["result"]["isError"], false);
    assert_eq!(lines[3]["result"]["structuredContent"]["total_peaks"], 3);
    assert_eq!(lines[4]["result"]["isError"], true);
    assert_eq!(
        lines[4]["result"]["structuredContent"]["error"]["code"],
        ERR_READ_ONLY_DENIED
    );
    assert!(!sb.workspace.join("blocked.tsv").exists());
    assert_eq!(lines[5]["result"]["isError"], true);
    assert_eq!(
        lines[5]["result"]["structuredContent"]["error"]["code"],
        ERR_SECURITY_VIOLATION
    );

    // 2. Subprocess in `--mode read-write` with `Content-Length` framing
    let mut rw_child = Command::new(nmrassign_bin())
        .args([
            "mcp",
            "--transport",
            "stdio",
            "--root",
            sb.workspace.to_str().expect("utf8 path"),
            "--mode",
            "read-write",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn nmrassign mcp rw");

    {
        let stdin = rw_child.stdin.as_mut().expect("rw stdin");
        let write_frame = |s: &mut dyn Write, payload: Value| {
            let body = serde_json::to_string(&payload).expect("json");
            write!(s, "Content-Length: {}\r\n\r\n{body}", body.len()).expect("write frame");
        };
        write_frame(
            stdin,
            json!({
                "jsonrpc": "2.0",
                "id": "rw-1",
                "method": "tools/list"
            }),
        );
        write_frame(
            stdin,
            json!({
                "jsonrpc": "2.0",
                "id": "rw-2",
                "method": "tools/call",
                "params": {
                    "name": "write_curated_peak_tsv",
                    "arguments": {
                        "output_path": "curated/sub_peaks.tsv",
                        "source_peaks_path": "peaks.tsv",
                        "exclude_indices": [1]
                    }
                }
            }),
        );
    }

    let rw_out = rw_child.wait_with_output().expect("wait rw child");
    assert!(
        rw_out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&rw_out.stderr)
    );
    let rw_stdout = String::from_utf8_lossy(&rw_out.stdout);
    assert!(rw_stdout.starts_with("Content-Length: "));
    assert!(rw_stdout.contains(r#""written_peaks":2"#));
    assert!(sb.workspace.join("curated/sub_peaks.tsv").exists());
}

#[test]
fn test_cli_mcp_http_loopback_bearer_401_403_413_and_non_loopback_bind_rejection() {
    let sb = CliSandbox::new("http_e2e");

    // 1. Adversarial CLI bind rejections (`0.0.0.0`, LAN IP, missing `--token`, invalid `--transport` / `--mode`)
    for bad_args in [
        vec![
            "mcp",
            "--transport",
            "http",
            "--bind",
            "0.0.0.0:8765",
            "--token",
            "tok123",
        ],
        vec![
            "mcp",
            "--transport",
            "http",
            "--bind",
            "[::]:8765",
            "--token",
            "tok123",
        ],
        vec![
            "mcp",
            "--transport",
            "http",
            "--bind",
            "192.168.1.42:8765",
            "--token",
            "tok123",
        ],
        vec!["mcp", "--transport", "http", "--bind", "127.0.0.1:0"],
        vec!["mcp", "--transport", "websocket"],
        vec!["mcp", "--mode", "root-admin"],
    ] {
        let out = Command::new(nmrassign_bin())
            .args(&bad_args)
            .output()
            .expect("run bad mcp invocation");
        assert!(
            !out.status.success(),
            "expected failure for args {bad_args:?}, got success"
        );
    }

    // 2. Live loopback HTTP server on ephemeral port (`127.0.0.1:0`) serving 4 requests (`200`, `401`, `403`, `413`)
    let mut child = Command::new(nmrassign_bin())
        .args([
            "mcp",
            "--transport",
            "http",
            "--root",
            sb.workspace.to_str().expect("utf8 path"),
            "--mode",
            "read-only",
            "--bind",
            "127.0.0.1:0",
            "--token",
            "cli-loopback-tok-999",
            "--max-requests",
            "4",
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn nmrassign mcp http");

    let stderr = child.stderr.take().expect("piped stderr");
    let mut stderr_reader = BufReader::new(stderr);
    let mut listen_line = String::new();
    stderr_reader
        .read_line(&mut listen_line)
        .expect("read MCP_HTTP_LISTENING line");
    let addr_str = listen_line
        .trim()
        .strip_prefix("MCP_HTTP_LISTENING http://")
        .expect("expected MCP_HTTP_LISTENING prefix")
        .to_string();

    let send_http = |raw_req: &str| -> String {
        let mut stream = TcpStream::connect(&addr_str).expect("connect loopback mcp http");
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .expect("set read timeout");
        stream
            .write_all(raw_req.as_bytes())
            .expect("write raw http");
        stream.flush().expect("flush raw http");
        let mut resp = String::new();
        stream.read_to_string(&mut resp).expect("read http resp");
        resp
    };

    // (a) Valid Bearer token -> 200 OK
    let ping_body = r#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#;
    let r200 = send_http(&format!(
        "POST /mcp HTTP/1.1\r\nHost: {addr_str}\r\nAuthorization: Bearer cli-loopback-tok-999\r\nContent-Length: {}\r\n\r\n{ping_body}",
        ping_body.len()
    ));
    assert!(r200.starts_with("HTTP/1.1 200 OK"), "got: {r200}");
    assert!(r200.contains(r#""result":{}"#), "got: {r200}");

    // (b) Invalid Bearer token -> 401 Unauthorized
    let r401 = send_http(&format!(
        "POST /mcp HTTP/1.1\r\nHost: {addr_str}\r\nAuthorization: Bearer wrong-token\r\nContent-Length: {}\r\n\r\n{ping_body}",
        ping_body.len()
    ));
    assert!(r401.starts_with("HTTP/1.1 401"), "got: {r401}");

    // (c) DNS-rebinding Host -> 403 Forbidden
    let r403 = send_http(&format!(
        "POST /mcp HTTP/1.1\r\nHost: attacker.dns-rebind.invalid\r\nAuthorization: Bearer cli-loopback-tok-999\r\nContent-Length: {}\r\n\r\n{ping_body}",
        ping_body.len()
    ));
    assert!(r403.starts_with("HTTP/1.1 403"), "got: {r403}");

    // (d) Oversized `usize`-overflowing `Content-Length` -> 413 Payload Too Large (zero OOM / zero panic!)
    let r413 = send_http(&format!(
        "POST /mcp HTTP/1.1\r\nHost: {addr_str}\r\nAuthorization: Bearer cli-loopback-tok-999\r\nContent-Length: 99999999999999999999999999999999\r\n\r\n{{}}"
    ));
    assert!(r413.starts_with("HTTP/1.1 413"), "got: {r413}");

    let status = child.wait().expect("wait http server exit");
    assert!(status.success(), "http server exited with {status}");
}

#[test]
fn test_cli_mcp_config_all_clients_and_adversarial_flag_errors() {
    for client in ["antigravity", "claude-desktop", "cursor", "generic"] {
        // Stdio config
        let out_stdio = Command::new(nmrassign_bin())
            .args([
                "mcp-config",
                "--client",
                client,
                "--root",
                "/data/nmr_project",
                "--mode",
                "read-only",
            ])
            .output()
            .expect("run mcp-config stdio");
        assert!(out_stdio.status.success());
        let parsed_stdio: Value =
            serde_json::from_slice(&out_stdio.stdout).expect("valid mcp-config JSON");
        assert_eq!(
            parsed_stdio["mcpServers"]["nmr-assign"]["command"],
            "nmrassign"
        );

        // Bridge config
        let out_bridge = Command::new(nmrassign_bin())
            .args([
                "mcp-config",
                "--client",
                client,
                "--bridge-port",
                "51439",
                "--token",
                "bridge_tok_abc",
            ])
            .output()
            .expect("run mcp-config bridge");
        assert!(out_bridge.status.success());
        let parsed_bridge: Value =
            serde_json::from_slice(&out_bridge.stdout).expect("valid bridge mcp-config JSON");
        assert_eq!(
            parsed_bridge["mcpServers"]["nmr-assign-workbench"]["headers"]["Authorization"],
            "Bearer bridge_tok_abc"
        );
    }

    // Adversarial `mcp-config` invocations must exit non-zero
    for bad_cfg in [
        vec!["mcp-config", "--client", "unknown-ide", "--root", "/tmp"],
        vec!["mcp-config", "--client", "antigravity"],
        vec!["mcp-config", "--client", "cursor", "--bridge-port", "0"],
        vec!["mcp-config", "--root", "/tmp"],
    ] {
        let out = Command::new(nmrassign_bin())
            .args(&bad_cfg)
            .output()
            .expect("run bad mcp-config");
        assert!(
            !out.status.success(),
            "expected failure for mcp-config {bad_cfg:?}"
        );
    }

    let help_out = Command::new(nmrassign_bin())
        .arg("--help")
        .output()
        .expect("nmrassign --help");
    let help_str = String::from_utf8_lossy(&help_out.stdout);
    assert!(help_str.contains("nmrassign mcp"));
    assert!(help_str.contains("nmrassign mcp-config"));

    println!(
        "MCP_METRIC_JSON:{}",
        json!({
            "suite": "nmr_cli_mcp_cli_tests",
            "cli_stdio_ro_rw_verified": true,
            "cli_http_loopback_200_401_403_413_verified": true,
            "cli_non_loopback_binds_rejected": 3,
            "cli_client_configs_verified": 3
        })
    );
}
