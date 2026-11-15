//! CLI subcommands `nmrassign mcp` and `nmrassign mcp-config` (`T15.1` & `T15.3`):
//! clean-room Model Context Protocol (`MCP`) JSON-RPC 2.0 server over `stdio` or authenticated
//! loopback `127.0.0.1` / `[::1]` Streamable HTTP, plus 1-command client configuration export
//! for `Antigravity`, `Claude Desktop`, and `Cursor`.
//!
//! # Clean-Room Provenance
//! Original implementation of `docs/specs/18-mcp-server-and-bridge.md` derived strictly from open
//! public specifications:
//! - Model Context Protocol (`MCP`) Open Specification (`2024-11-05` / `2025-03-26`) & JSON-RPC 2.0.
//! - Fielding, R. et al. (RFC 9110, Jun 2022) — HTTP Semantics (`Host`, `Origin`, `Bearer` auth).

use std::env;
use std::fs;
use std::io;
use std::net::TcpListener;
use std::path::PathBuf;
use std::time::Duration;

use nmr_core::mcp::{
    generate_mcp_client_config_pretty, handle_mcp_http_stream, parse_loopback_bind_addr,
    run_mcp_stdio_loop, McpAccessMode, McpClientMode, McpClientTarget, McpSecurityPolicy,
    McpServer,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum McpTransportKind {
    Stdio,
    Http,
}

impl McpTransportKind {
    fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "stdio" => Ok(Self::Stdio),
            "http" | "streamable-http" => Ok(Self::Http),
            other => Err(format!(
                "invalid --transport {other:?} (expected 'stdio' or 'http')"
            )),
        }
    }
}

fn parse_access_mode(raw: &str) -> Result<McpAccessMode, String> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "read-only" | "readonly" | "ro" => Ok(McpAccessMode::ReadOnly),
        "read-write" | "readwrite" | "rw" => Ok(McpAccessMode::ReadWrite),
        other => Err(format!(
            "invalid --mode {other:?} (expected 'read-only' or 'read-write')"
        )),
    }
}

fn parse_client_target(raw: &str) -> Result<McpClientTarget, String> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "antigravity" | "agy" => Ok(McpClientTarget::Antigravity),
        "claude-desktop" | "claude" => Ok(McpClientTarget::ClaudeDesktop),
        "cursor" | "generic" | "generic-json" => Ok(McpClientTarget::Cursor),
        other => Err(format!(
            "invalid --client {other:?} (expected 'antigravity', 'claude-desktop', 'cursor', or 'generic')"
        )),
    }
}

/// Usage text for `nmrassign mcp`.
pub fn mcp_usage() -> &'static str {
    "\
Usage:
  nmrassign mcp [--transport stdio|http] [--root <dir> ...] [--mode read-only|read-write]
                [--bind 127.0.0.1:<port>] [--token <bearer>] [--max-requests <N>]

Options:
  --transport <stdio|http>       MCP transport framing (default: stdio).
  --root, --workspace <dir>      Allowed canonicalized sandbox workspace root (repeatable; defaults to '.').
  --mode <read-only|read-write>  Tool permission policy (default: read-only).
  --read-only / --read-write     Shorthand flags for --mode.
  --expose-blocked-write-tools   Include write tools in tools/list while still denying execution in read-only mode.
  --bind <127.0.0.1:port>        Loopback bind address for --transport http (non-loopback IPs are rejected).
  --token <bearer>               Required secret Bearer token when --transport http is selected.
  --max-requests <N>             Optional request cap before clean exit on --transport http.
"
}

/// Usage text for `nmrassign mcp-config`.
pub fn mcp_config_usage() -> &'static str {
    "\
Usage:
  nmrassign mcp-config --client <antigravity|claude-desktop|cursor|generic>
                       [--root <dir> ...] [--mode read-only|read-write]
                       [--bridge-port <port> [--token <bearer>]]
                       [--command <nmrassign_bin>] [-o <mcp_config.json>]
"
}

/// Execute `nmrassign mcp`.
pub fn run_mcp_cli(args: &[String]) -> Result<(), String> {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print!("{}", mcp_usage());
        return Ok(());
    }

    let mut transport = McpTransportKind::Stdio;
    let mut roots: Vec<PathBuf> = Vec::new();
    let mut mode = McpAccessMode::ReadOnly;
    let mut hide_write_tools_in_read_only = true;
    let mut bind_spec = "127.0.0.1:8765".to_string();
    let mut bearer_token: Option<String> = env::var("NMR_MCP_BEARER_TOKEN")
        .ok()
        .filter(|s| !s.trim().is_empty());
    let mut max_requests: Option<usize> = None;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--transport" => {
                i += 1;
                let val = args.get(i).ok_or_else(|| {
                    "--transport requires a value ('stdio' or 'http')".to_string()
                })?;
                transport = McpTransportKind::parse(val)?;
            }
            "--root" | "--workspace" => {
                i += 1;
                let val = args
                    .get(i)
                    .ok_or_else(|| format!("{} requires a directory path", args[i - 1]))?;
                roots.push(PathBuf::from(val));
            }
            "--mode" => {
                i += 1;
                let val = args.get(i).ok_or_else(|| {
                    "--mode requires a value ('read-only' or 'read-write')".to_string()
                })?;
                mode = parse_access_mode(val)?;
            }
            "--read-only" => {
                mode = McpAccessMode::ReadOnly;
            }
            "--read-write" => {
                mode = McpAccessMode::ReadWrite;
            }
            "--expose-blocked-write-tools" => {
                hide_write_tools_in_read_only = false;
            }
            "--bind" => {
                i += 1;
                let val = args
                    .get(i)
                    .ok_or_else(|| "--bind requires <127.0.0.1:port>".to_string())?;
                bind_spec = val.clone();
            }
            "--token" => {
                i += 1;
                let val = args
                    .get(i)
                    .ok_or_else(|| "--token requires a non-empty bearer string".to_string())?;
                bearer_token = Some(val.clone());
            }
            "--max-requests" => {
                i += 1;
                let val = args
                    .get(i)
                    .ok_or_else(|| "--max-requests requires an integer".to_string())?;
                let parsed = val
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --max-requests {val:?}: {e}"))?;
                max_requests = Some(parsed);
            }
            other => {
                return Err(format!("unknown `mcp` option {other:?}\n\n{}", mcp_usage()));
            }
        }
        i += 1;
    }

    if roots.is_empty() {
        let cwd = env::current_dir()
            .map_err(|e| format!("failed to resolve current directory as workspace root: {e}"))?;
        roots.push(cwd);
    }

    let policy = McpSecurityPolicy::new(&roots, mode)
        .map_err(|e| format!("failed to initialize MCP sandbox policy: {e}"))?
        .with_hide_write_tools_in_read_only(hide_write_tools_in_read_only);
    let mut server = McpServer::new(policy);

    match transport {
        McpTransportKind::Stdio => {
            let stdin = io::stdin();
            let stdout = io::stdout();
            let reader = stdin.lock();
            let writer = io::BufWriter::new(stdout.lock());
            run_mcp_stdio_loop(&mut server, reader, writer)
                .map_err(|e| format!("MCP stdio loop terminated with I/O error: {e}"))?;
            Ok(())
        }
        McpTransportKind::Http => {
            let bind_addr = parse_loopback_bind_addr(&bind_spec).map_err(|e| format!("{e}"))?;
            let token = bearer_token
                .as_deref()
                .map(str::trim)
                .filter(|t| !t.is_empty())
                .ok_or_else(|| {
                    "MCP HTTP transport requires a non-empty --token <bearer>".to_string()
                })?;

            let listener = TcpListener::bind(bind_addr)
                .map_err(|e| format!("failed to bind loopback MCP HTTP socket {bind_addr}: {e}"))?;
            let local_addr = listener
                .local_addr()
                .map_err(|e| format!("failed to query local MCP HTTP socket address: {e}"))?;
            eprintln!("MCP_HTTP_LISTENING http://{local_addr}");

            if max_requests == Some(0) {
                return Ok(());
            }

            let mut served = 0usize;
            for incoming in listener.incoming() {
                match incoming {
                    Ok(stream) => {
                        let _ = stream.set_read_timeout(Some(Duration::from_secs(10)));
                        let _ = stream.set_write_timeout(Some(Duration::from_secs(10)));
                        let _ = handle_mcp_http_stream(&mut server, stream, token);
                        served += 1;
                        if let Some(limit) = max_requests {
                            if served >= limit {
                                break;
                            }
                        }
                    }
                    Err(e) => {
                        return Err(format!("MCP HTTP accept error: {e}"));
                    }
                }
            }
            Ok(())
        }
    }
}

/// Execute `nmrassign mcp-config`.
pub fn run_mcp_config_cli(args: &[String]) -> Result<(), String> {
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        print!("{}", mcp_config_usage());
        return Ok(());
    }

    let mut client: Option<McpClientTarget> = None;
    let mut roots: Vec<String> = Vec::new();
    let mut mode = McpAccessMode::ReadOnly;
    let mut bridge_port: Option<u16> = None;
    let mut bearer_token: Option<String> = None;
    let mut command = "nmrassign".to_string();
    let mut out_path: Option<PathBuf> = None;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--client" => {
                i += 1;
                let val = args.get(i).ok_or_else(|| {
                    "--client requires 'antigravity', 'claude-desktop', or 'cursor'".to_string()
                })?;
                client = Some(parse_client_target(val)?);
            }
            "--root" | "--workspace" => {
                i += 1;
                let val = args
                    .get(i)
                    .ok_or_else(|| format!("{} requires a directory path", args[i - 1]))?;
                if val.trim().is_empty() {
                    return Err("workspace --root must not be empty".to_string());
                }
                roots.push(val.clone());
            }
            "--mode" => {
                i += 1;
                let val = args
                    .get(i)
                    .ok_or_else(|| "--mode requires 'read-only' or 'read-write'".to_string())?;
                mode = parse_access_mode(val)?;
            }
            "--read-only" => {
                mode = McpAccessMode::ReadOnly;
            }
            "--read-write" => {
                mode = McpAccessMode::ReadWrite;
            }
            "--bridge-port" => {
                i += 1;
                let val = args
                    .get(i)
                    .ok_or_else(|| "--bridge-port requires a TCP port (1..=65535)".to_string())?;
                let port = val
                    .parse::<u16>()
                    .map_err(|e| format!("invalid --bridge-port {val:?}: {e}"))?;
                if port == 0 {
                    return Err("--bridge-port must be in 1..=65535 (got 0)".to_string());
                }
                bridge_port = Some(port);
            }
            "--token" => {
                i += 1;
                let val = args
                    .get(i)
                    .ok_or_else(|| "--token requires a non-empty bearer token".to_string())?;
                bearer_token = Some(val.clone());
            }
            "--command" => {
                i += 1;
                let val = args
                    .get(i)
                    .ok_or_else(|| "--command requires a binary name or path".to_string())?;
                command = val.clone();
            }
            "-o" | "--out" => {
                i += 1;
                let val = args
                    .get(i)
                    .ok_or_else(|| "-o/--out requires a destination file path".to_string())?;
                out_path = Some(PathBuf::from(val));
            }
            other => {
                return Err(format!(
                    "unknown `mcp-config` option {other:?}\n\n{}",
                    mcp_config_usage()
                ));
            }
        }
        i += 1;
    }

    let target = client.ok_or_else(|| {
        format!(
            "missing required --client <antigravity|claude-desktop|cursor>\n\n{}",
            mcp_config_usage()
        )
    })?;

    let client_mode = if let Some(port) = bridge_port {
        let token = bearer_token
            .filter(|t| !t.trim().is_empty())
            .unwrap_or_else(|| "REPLACE_WITH_WORKBENCH_SESSION_TOKEN".to_string());
        McpClientMode::HttpBridge {
            endpoint_url: format!("http://127.0.0.1:{port}/mcp"),
            bearer_token: token,
            access_mode: mode,
        }
    } else {
        if roots.is_empty() {
            return Err(
                "mcp-config in stdio mode requires at least one --root <dir> (or pass --bridge-port <port>)"
                    .to_string(),
            );
        }
        McpClientMode::Stdio {
            command,
            workspace_roots: roots,
            access_mode: mode,
        }
    };

    let rendered = generate_mcp_client_config_pretty(target, &client_mode)?;
    if let Some(dest) = out_path {
        if let Some(parent) = dest.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)
                    .map_err(|e| format!("failed to create {}: {e}", parent.display()))?;
            }
        }
        fs::write(&dest, format!("{rendered}\n"))
            .map_err(|e| format!("failed to write {}: {e}", dest.display()))?;
    } else {
        println!("{rendered}");
    }
    Ok(())
}
