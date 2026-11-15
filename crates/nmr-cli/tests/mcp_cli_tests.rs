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
    let valid_env_token = "cli-loopback-env-bearer-token-32ch-999";
    assert!(valid_env_token.len() >= 32);

    // 1. Adversarial CLI bind & flag rejections (`0.0.0.0`, `[::]`, LAN IP with valid env token; invalid `--transport` / `--mode`)
    for bad_args in [
        vec!["mcp", "--transport", "http", "--bind", "0.0.0.0:8765"],
        vec!["mcp", "--transport", "http", "--bind", "[::]:8765"],
        vec!["mcp", "--transport", "http", "--bind", "192.168.1.42:8765"],
        vec!["mcp", "--transport", "websocket"],
        vec!["mcp", "--mode", "root-admin"],
    ] {
        let out = Command::new(nmrassign_bin())
            .env("NMR_MCP_BEARER_TOKEN", valid_env_token)
            .args(&bad_args)
            .output()
            .expect("run bad mcp invocation");
        assert!(
            !out.status.success(),
            "expected failure for args {bad_args:?}, got success"
        );
    }

    // 2. Live loopback HTTP server on ephemeral port (`127.0.0.1:0`) using preferred `NMR_MCP_BEARER_TOKEN`
    //    (>= 32 chars, no `ps`-visible `--token` argv) serving 4 requests (`200`, `401`, `403`, `413`)
    let mut child = Command::new(nmrassign_bin())
        .env("NMR_MCP_BEARER_TOKEN", valid_env_token)
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
        "POST /mcp HTTP/1.1\r\nHost: {addr_str}\r\nAuthorization: Bearer {valid_env_token}\r\nContent-Length: {}\r\n\r\n{ping_body}",
        ping_body.len()
    ));
    assert!(r200.starts_with("HTTP/1.1 200 OK"), "got: {r200}");
    assert!(r200.contains(r#""result":{}"#), "got: {r200}");

    // (b) Invalid Bearer token -> 401 Unauthorized
    let r401 = send_http(&format!(
        "POST /mcp HTTP/1.1\r\nHost: {addr_str}\r\nAuthorization: Bearer wrong-token-0123456789abcdef01234567\r\nContent-Length: {}\r\n\r\n{ping_body}",
        ping_body.len()
    ));
    assert!(r401.starts_with("HTTP/1.1 401"), "got: {r401}");

    // (c) DNS-rebinding Host -> 403 Forbidden
    let r403 = send_http(&format!(
        "POST /mcp HTTP/1.1\r\nHost: attacker.dns-rebind.invalid\r\nAuthorization: Bearer {valid_env_token}\r\nContent-Length: {}\r\n\r\n{ping_body}",
        ping_body.len()
    ));
    assert!(r403.starts_with("HTTP/1.1 403"), "got: {r403}");

    // (d) Oversized `usize`-overflowing `Content-Length` -> 413 Payload Too Large (zero OOM / zero panic!)
    let r413 = send_http(&format!(
        "POST /mcp HTTP/1.1\r\nHost: {addr_str}\r\nAuthorization: Bearer {valid_env_token}\r\nContent-Length: 99999999999999999999999999999999\r\n\r\n{{}}"
    ));
    assert!(r413.starts_with("HTTP/1.1 413"), "got: {r413}");

    let status = child.wait().expect("wait http server exit");
    assert!(status.success(), "http server exited with {status}");
}

#[test]
fn test_cli_mcp_http_token_floor_env_auth_and_csprng_autogen() {
    let sb = CliSandbox::new("http_f20");

    // (a) `< 32`-char `--token` and `< 32`-char `NMR_MCP_BEARER_TOKEN` must be rejected with `Err`
    for short_cli_tok in [
        "",
        "   ",
        "tok123",
        "0123456789abcdef0123456789abcde", // 31 chars (boundary below 32)
    ] {
        let out = Command::new(nmrassign_bin())
            .env_remove("NMR_MCP_BEARER_TOKEN")
            .args([
                "mcp",
                "--transport",
                "http",
                "--bind",
                "127.0.0.1:0",
                "--token",
                short_cli_tok,
                "--max-requests",
                "0",
            ])
            .output()
            .expect("run short --token invocation");
        assert!(
            !out.status.success(),
            "expected Err for <32-char --token {short_cli_tok:?}"
        );
        let stderr_str = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr_str.contains("32"),
            "expected descriptive >=32-char error message on stderr, got: {stderr_str}"
        );
    }

    for short_env_tok in [
        "",
        "   ",
        "env-short-13c",
        "0123456789abcdef0123456789abcde", // 31 chars (boundary below 32)
    ] {
        let out = Command::new(nmrassign_bin())
            .env("NMR_MCP_BEARER_TOKEN", short_env_tok)
            .args([
                "mcp",
                "--transport",
                "http",
                "--bind",
                "127.0.0.1:0",
                "--max-requests",
                "0",
            ])
            .output()
            .expect("run short NMR_MCP_BEARER_TOKEN invocation");
        assert!(
            !out.status.success(),
            "expected Err for <32-char NMR_MCP_BEARER_TOKEN {short_env_tok:?}"
        );
        let stderr_str = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr_str.contains("NMR_MCP_BEARER_TOKEN") && stderr_str.contains("32"),
            "expected descriptive NMR_MCP_BEARER_TOKEN >=32-char error on stderr, got: {stderr_str}"
        );
    }

    // (b) `NMR_MCP_BEARER_TOKEN` (exact 32-char floor boundary) authenticates `initialize` cleanly without `--token`
    let exact_32_env_token = "exact_32_char_env_token_01234567";
    assert_eq!(exact_32_env_token.len(), 32);
    let mut env_child = Command::new(nmrassign_bin())
        .env("NMR_MCP_BEARER_TOKEN", exact_32_env_token)
        .args([
            "mcp",
            "--transport",
            "http",
            "--root",
            sb.workspace.to_str().expect("utf8 path"),
            "--bind",
            "127.0.0.1:0",
            "--max-requests",
            "2",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn nmrassign mcp http with NMR_MCP_BEARER_TOKEN");

    let env_stderr = env_child.stderr.take().expect("piped stderr");
    let mut env_stderr_reader = BufReader::new(env_stderr);
    let mut env_listen_line = String::new();
    env_stderr_reader
        .read_line(&mut env_listen_line)
        .expect("read MCP_HTTP_LISTENING");
    assert!(
        !env_listen_line.contains(exact_32_env_token),
        "user-supplied NMR_MCP_BEARER_TOKEN must not be echoed to stderr"
    );
    let env_addr = env_listen_line
        .trim()
        .strip_prefix("MCP_HTTP_LISTENING http://")
        .expect("expected MCP_HTTP_LISTENING")
        .to_string();

    let init_body = serde_json::to_string(&json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": { "protocolVersion": MCP_PROTOCOL_VERSION }
    }))
    .expect("serialize initialize");

    let send_to = |addr: &str, bearer: Option<&str>, body: &str| -> String {
        let mut stream = TcpStream::connect(addr).expect("connect loopback mcp http");
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .expect("set read timeout");
        let auth_hdr = match bearer {
            Some(tok) => format!("Authorization: Bearer {tok}\r\n"),
            None => String::new(),
        };
        let req = format!(
            "POST /mcp HTTP/1.1\r\nHost: {addr}\r\n{auth_hdr}Content-Length: {}\r\n\r\n{body}",
            body.len()
        );
        stream.write_all(req.as_bytes()).expect("write http req");
        stream.flush().expect("flush http req");
        let mut resp = String::new();
        stream.read_to_string(&mut resp).expect("read http resp");
        resp
    };

    let env_unauth = send_to(&env_addr, None, &init_body);
    assert!(env_unauth.starts_with("HTTP/1.1 401"), "got: {env_unauth}");

    let env_ok = send_to(&env_addr, Some(exact_32_env_token), &init_body);
    assert!(env_ok.starts_with("HTTP/1.1 200 OK"), "got: {env_ok}");
    let (_, env_payload) = env_ok.split_once("\r\n\r\n").expect("split http response");
    let env_json: Value = serde_json::from_str(env_payload).expect("parse initialize response");
    assert_eq!(env_json["result"]["protocolVersion"], MCP_PROTOCOL_VERSION);

    let env_out = env_child.wait_with_output().expect("wait env_child");
    assert!(env_out.status.success());
    assert!(
        env_out.stdout.is_empty(),
        "stdout must remain completely clean on HTTP transport"
    );

    // (c) Omitting BOTH `NMR_MCP_BEARER_TOKEN` and `--token` on `--transport http` auto-generates
    //     a valid 64-hex-char OS CSPRNG token emitted on `stderr` that authenticates `initialize`!
    let mut auto_child = Command::new(nmrassign_bin())
        .env_remove("NMR_MCP_BEARER_TOKEN")
        .args([
            "mcp",
            "--transport",
            "http",
            "--root",
            sb.workspace.to_str().expect("utf8 path"),
            "--bind",
            "127.0.0.1:0",
            "--max-requests",
            "2",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn nmrassign mcp http with auto-generated CSPRNG token");

    let auto_stderr = auto_child.stderr.take().expect("piped stderr");
    let mut auto_stderr_reader = BufReader::new(auto_stderr);
    let mut auto_token: Option<String> = None;
    let mut auto_addr: Option<String> = None;
    for _ in 0..8 {
        let mut line = String::new();
        let n = auto_stderr_reader
            .read_line(&mut line)
            .expect("read stderr line");
        if n == 0 {
            break;
        }
        let trimmed = line.trim();
        if let Some(tok) = trimmed.strip_prefix("MCP_HTTP_BEARER_TOKEN ") {
            auto_token = Some(tok.trim().to_string());
        } else if let Some(addr) = trimmed.strip_prefix("MCP_HTTP_LISTENING http://") {
            auto_addr = Some(addr.trim().to_string());
            break;
        }
    }

    let generated_tok = auto_token.expect("expected MCP_HTTP_BEARER_TOKEN line on stderr");
    let generated_addr = auto_addr.expect("expected MCP_HTTP_LISTENING line on stderr");
    assert_eq!(
        generated_tok.len(),
        64,
        "expected 64-hex-char (256-bit) CSPRNG token, got {generated_tok:?}"
    );
    assert!(
        generated_tok
            .chars()
            .all(|c| matches!(c, '0'..='9' | 'a'..='f')),
        "expected lowercase hex CSPRNG token, got {generated_tok:?}"
    );
    assert_ne!(generated_tok, "0".repeat(64));

    let auto_401 = send_to(
        &generated_addr,
        Some("wrong_token_0123456789abcdef0123456789abcdef"),
        &init_body,
    );
    assert!(auto_401.starts_with("HTTP/1.1 401"), "got: {auto_401}");

    let auto_200 = send_to(&generated_addr, Some(&generated_tok), &init_body);
    assert!(auto_200.starts_with("HTTP/1.1 200 OK"), "got: {auto_200}");
    let (_, auto_payload) = auto_200
        .split_once("\r\n\r\n")
        .expect("split auto http response");
    let auto_json: Value =
        serde_json::from_str(auto_payload).expect("parse auto initialize response");
    assert_eq!(auto_json["result"]["protocolVersion"], MCP_PROTOCOL_VERSION);

    let auto_out = auto_child.wait_with_output().expect("wait auto_child");
    assert!(auto_out.status.success());
    assert!(
        auto_out.stdout.is_empty(),
        "stdout must remain completely empty when CSPRNG token is emitted on stderr"
    );
}

#[test]
fn test_cli_mcp_config_all_clients_and_adversarial_flag_errors() {
    let bridge_env_token = "bridge_session_tok_0123456789abcdef";
    assert!(bridge_env_token.len() >= 32);
    let bridge_cli_token = "bridge_cli_override_tok_0123456789ab";
    assert!(bridge_cli_token.len() >= 32);

    for client in ["antigravity", "claude-desktop", "cursor", "generic"] {
        // Stdio config
        let out_stdio = Command::new(nmrassign_bin())
            .env_remove("NMR_MCP_BEARER_TOKEN")
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

        // Bridge config via preferred `NMR_MCP_BEARER_TOKEN` environment injection (>= 32 chars)
        let out_bridge_env = Command::new(nmrassign_bin())
            .env("NMR_MCP_BEARER_TOKEN", bridge_env_token)
            .args(["mcp-config", "--client", client, "--bridge-port", "51439"])
            .output()
            .expect("run mcp-config bridge env");
        assert!(out_bridge_env.status.success());
        let parsed_bridge_env: Value = serde_json::from_slice(&out_bridge_env.stdout)
            .expect("valid bridge env mcp-config JSON");
        assert_eq!(
            parsed_bridge_env["mcpServers"]["nmr-assign-workbench"]["headers"]["Authorization"],
            format!("Bearer {bridge_env_token}")
        );

        // Bridge config with explicit `--token` (>= 32 chars)
        let out_bridge = Command::new(nmrassign_bin())
            .env_remove("NMR_MCP_BEARER_TOKEN")
            .args([
                "mcp-config",
                "--client",
                client,
                "--bridge-port",
                "51439",
                "--token",
                bridge_cli_token,
            ])
            .output()
            .expect("run mcp-config bridge");
        assert!(out_bridge.status.success());
        let parsed_bridge: Value =
            serde_json::from_slice(&out_bridge.stdout).expect("valid bridge mcp-config JSON");
        assert_eq!(
            parsed_bridge["mcpServers"]["nmr-assign-workbench"]["headers"]["Authorization"],
            format!("Bearer {bridge_cli_token}")
        );
    }

    // Adversarial `mcp-config` invocations (including < 32-char `--token` on `--bridge-port`) must exit non-zero
    for bad_cfg in [
        vec!["mcp-config", "--client", "unknown-ide", "--root", "/tmp"],
        vec!["mcp-config", "--client", "antigravity"],
        vec!["mcp-config", "--client", "cursor", "--bridge-port", "0"],
        vec![
            "mcp-config",
            "--client",
            "antigravity",
            "--bridge-port",
            "51439",
            "--token",
            "bridge_tok_short",
        ],
        vec![
            "mcp-config",
            "--client",
            "cursor",
            "--bridge-port",
            "51439",
            "--token",
            "0123456789abcdef0123456789abcde",
        ],
        vec!["mcp-config", "--root", "/tmp"],
    ] {
        let out = Command::new(nmrassign_bin())
            .env_remove("NMR_MCP_BEARER_TOKEN")
            .args(&bad_cfg)
            .output()
            .expect("run bad mcp-config");
        assert!(
            !out.status.success(),
            "expected failure for mcp-config {bad_cfg:?}"
        );
    }

    let help_mcp = Command::new(nmrassign_bin())
        .args(["mcp", "--help"])
        .output()
        .expect("nmrassign mcp --help");
    let help_mcp_str = String::from_utf8_lossy(&help_mcp.stdout);
    assert!(help_mcp_str.contains("NMR_MCP_BEARER_TOKEN"));
    assert!(help_mcp_str.contains("32"));

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
            "cli_http_min_token_len_32_enforced": true,
            "cli_http_csprng_autogen_verified": true,
            "cli_non_loopback_binds_rejected": 3,
            "cli_client_configs_verified": 3
        })
    );
}
