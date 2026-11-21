//! Embeds the short git commit SHA (`FERRIC_GIT_SHA`) into `ferric` at compile time
//! without external build dependencies. Honors `FERRIC_GIT_HASH` (or `FERRIC_GIT_SHA`)
//! environment overrides so source/container builds without `.git` report the commit.

use std::path::Path;
use std::process::Command;

fn resolve_git_sha() -> String {
    for key in ["FERRIC_GIT_HASH", "FERRIC_GIT_SHA"] {
        if let Ok(raw) = std::env::var(key) {
            let trimmed = raw.trim();
            if !trimmed.is_empty() && trimmed != "unknown" && !trimmed.contains(['\n', '\r']) {
                return trimmed.to_string();
            }
        }
    }

    Command::new("git")
        .args(["rev-parse", "--short=7", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

fn main() {
    println!("cargo:rerun-if-env-changed=FERRIC_GIT_HASH");
    println!("cargo:rerun-if-env-changed=FERRIC_GIT_SHA");

    for rel in [
        "../../.git/HEAD",
        "../../.git/refs/heads",
        "../../.git/packed-refs",
    ] {
        if Path::new(rel).exists() {
            println!("cargo:rerun-if-changed={rel}");
        }
    }

    let git_sha = resolve_git_sha();
    println!("cargo:rustc-env=FERRIC_GIT_SHA={git_sha}");
}
