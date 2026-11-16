#!/usr/bin/env bash
# Deterministic builder for the zero-install wasm32 WebGPU browser demo (T16.4 / P17).
#
# Compiles `crates/nmr-gui` for `wasm32-unknown-unknown`, wraps the binary with
# `wasm-bindgen --target web`, validates mobile/tablet WebGPU HTML shell invariants,
# and stages `docs/demo/index.html` into the target output directory ready for local
# static serving or GitHub Pages deployment.
#
# Provenance: Original clean-room build script (MIT License).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$REPO_ROOT/target/web-demo"
INSTALL_CLI=0
CHECK_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)
      if [[ $# -lt 2 ]]; then
        echo "error: missing directory path after --out" >&2
        exit 1
      fi
      OUT_DIR="$2"
      shift 2
      ;;
    --install-cli)
      INSTALL_CLI=1
      shift
      ;;
    --check-only)
      CHECK_ONLY=1
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Usage: bash scripts/build_web_demo.sh [--out <dir>] [--install-cli] [--check-only]

Options:
  --out <dir>      Destination folder for index.html + ferric_gui.{js,wasm} (default: target/web-demo)
  --install-cli    Install matching wasm-bindgen-cli version from Cargo.lock if missing/mismatched
  --check-only     Run cargo check for wasm32-unknown-unknown and validate docs/demo/index.html
EOF
      exit 0
      ;;
    *)
      echo "error: unknown argument '$1'" >&2
      exit 1
      ;;
  esac
done

DEMO_HTML="$REPO_ROOT/docs/demo/index.html"
TRUNK_HTML="$REPO_ROOT/crates/nmr-gui/index.html"

if [[ ! -f "$DEMO_HTML" ]]; then
  echo "error: missing $DEMO_HTML" >&2
  exit 1
fi
if [[ ! -f "$TRUNK_HTML" ]]; then
  echo "error: missing $TRUNK_HTML" >&2
  exit 1
fi

render_demo_shell_html() {
  cat <<'DEMO_EOF'
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover" />
  <meta name="theme-color" content="#0e131b" />
  <meta name="mobile-web-app-capable" content="yes" />
  <meta name="apple-mobile-web-app-capable" content="yes" />
  <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent" />
  <meta name="apple-mobile-web-app-title" content="FERRIC" />
  <title>FERRIC — Live WebGPU Browser Workbench Demo</title>
  <style>
    :root {
      --bg: #0e131b;
      --bar-bg: #141c27;
      --bar-border: #253347;
      --text: #dce7f5;
      --muted: #93a8c4;
      --accent: #5fd4ff;
      --ok-bg: rgba(46, 204, 113, 0.16);
      --ok-fg: #6ee7a8;
      --warn-bg: rgba(245, 158, 11, 0.16);
      --warn-fg: #fbbf24;
      --err-bg: rgba(239, 68, 68, 0.16);
      --err-fg: #fca5a5;
    }
    * {
      box-sizing: border-box;
    }
    html, body {
      margin: 0;
      padding: 0;
      width: 100%;
      height: 100vh;
      height: 100dvh;
      overflow: hidden;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    body {
      display: flex;
      flex-direction: column;
    }
    html, body, #canvas-wrap, canvas#nmr_canvas {
      touch-action: none;
      -webkit-user-select: none;
      user-select: none;
      -webkit-touch-callout: none;
      overscroll-behavior: none;
    }
    header.demo-bar {
      min-height: calc(38px + env(safe-area-inset-top, 0px));
      flex-shrink: 0;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding-top: env(safe-area-inset-top, 0px);
      padding-right: calc(14px + env(safe-area-inset-right, 0px));
      padding-bottom: 0;
      padding-left: calc(14px + env(safe-area-inset-left, 0px));
      background: var(--bar-bg);
      border-bottom: 1px solid var(--bar-border);
      font-size: 13px;
      z-index: 20;
      position: relative;
      touch-action: manipulation;
    }
    .demo-brand {
      display: flex;
      align-items: center;
      gap: 10px;
      min-width: 0;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .demo-brand strong {
      color: #ffffff;
      letter-spacing: 0.01em;
    }
    .preset-note {
      color: var(--muted);
      font-size: 12px;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .demo-actions {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-shrink: 0;
    }
    .cap-badge {
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 600;
      white-space: nowrap;
      background: rgba(95, 212, 255, 0.14);
      color: var(--accent);
      border: 1px solid rgba(95, 212, 255, 0.28);
    }
    .cap-badge.ok {
      background: var(--ok-bg);
      color: var(--ok-fg);
      border-color: rgba(110, 231, 168, 0.32);
    }
    .cap-badge.warn {
      background: var(--warn-bg);
      color: var(--warn-fg);
      border-color: rgba(251, 191, 36, 0.34);
    }
    .cap-badge.err {
      background: var(--err-bg);
      color: var(--err-fg);
      border-color: rgba(252, 165, 165, 0.34);
    }
    a.manual-link {
      color: var(--accent);
      text-decoration: none;
      font-weight: 600;
      padding: 4px 10px;
      border-radius: 6px;
      border: 1px solid rgba(95, 212, 255, 0.32);
      background: rgba(95, 212, 255, 0.08);
      white-space: nowrap;
      touch-action: manipulation;
    }
    a.manual-link:hover {
      background: rgba(95, 212, 255, 0.18);
    }
    a.manual-link.github-link {
      color: var(--text);
      border-color: var(--bar-border);
      background: rgba(255, 255, 255, 0.05);
    }
    a.manual-link.github-link:hover {
      border-color: rgba(95, 212, 255, 0.36);
      background: rgba(95, 212, 255, 0.12);
    }
    .label-mobile {
      display: none;
    }
    #canvas-wrap {
      position: relative;
      flex: 1 1 auto;
      min-height: 0;
      width: 100%;
      height: calc(100vh - 38px);
      height: calc(100dvh - 38px - env(safe-area-inset-top, 0px));
      padding-right: env(safe-area-inset-right, 0px);
      padding-bottom: env(safe-area-inset-bottom, 0px);
      padding-left: env(safe-area-inset-left, 0px);
      overflow: hidden;
    }
    canvas#nmr_canvas {
      display: block;
      width: 100%;
      height: 100%;
      outline: none;
    }
    #nmr_loading {
      position: absolute;
      inset: 0;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 14px;
      padding: 24px;
      overflow-y: auto;
      -webkit-overflow-scrolling: touch;
      background: var(--bg);
      z-index: 10;
      text-align: center;
    }
    #fallback {
      position: absolute;
      inset: 0;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: flex-start;
      overflow-y: auto;
      -webkit-overflow-scrolling: touch;
      padding: 16px;
      padding-bottom: calc(16px + env(safe-area-inset-bottom, 0px));
      background: var(--bg);
      z-index: 15;
      touch-action: pan-y;
      -webkit-user-select: text;
      user-select: text;
    }
    #fallback[hidden] {
      display: none;
    }
    .spinner {
      width: 36px;
      height: 36px;
      border: 3px solid rgba(95, 212, 255, 0.22);
      border-top-color: var(--accent);
      border-radius: 50%;
      animation: spin 0.85s linear infinite;
    }
    @keyframes spin {
      to { transform: rotate(360deg); }
    }
    .status-card {
      margin: auto 0;
      width: 100%;
      max-width: 600px;
      background: #151e2b;
      border: 1px solid var(--bar-border);
      border-radius: 10px;
      padding: 18px 20px;
      line-height: 1.55;
      font-size: 13.5px;
      color: var(--text);
      text-align: left;
    }
    .status-card h2 {
      margin: 0 0 8px 0;
      font-size: 16px;
      color: #ffffff;
    }
    .status-card code {
      background: rgba(255, 255, 255, 0.08);
      padding: 2px 6px;
      border-radius: 4px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      overflow-wrap: anywhere;
    }
    .status-card p {
      margin: 8px 0;
      color: var(--muted);
    }
    .status-card .hint-line {
      font-size: 12.5px;
      color: var(--text);
    }
    @media (max-width: 960px) {
      .preset-note {
        display: none;
      }
    }
    @media (max-width: 680px) {
      header.demo-bar {
        min-height: calc(36px + env(safe-area-inset-top, 0px));
        gap: 6px;
        padding-right: calc(8px + env(safe-area-inset-right, 0px));
        padding-left: calc(8px + env(safe-area-inset-left, 0px));
        font-size: 12px;
      }
      #canvas-wrap {
        height: calc(100vh - 36px);
        height: calc(100dvh - 36px - env(safe-area-inset-top, 0px));
      }
      .preset-note,
      .brand-sub,
      .label-desktop {
        display: none;
      }
      .label-mobile {
        display: inline;
      }
      .demo-brand {
        gap: 6px;
      }
      .demo-actions {
        gap: 5px;
      }
      .cap-badge {
        padding: 2px 6px;
        font-size: 10.5px;
      }
      a.manual-link {
        padding: 3px 7px;
        font-size: 11.5px;
      }
      .status-card {
        padding: 14px 16px;
        font-size: 13px;
      }
    }
  </style>
</head>
<body>
  <header class="demo-bar">
    <div class="demo-brand">
      <strong>FERRIC<span class="brand-sub"> &middot; Live WebGPU Demo</span></strong>
      <span class="preset-note">
        Preloaded in-memory: Ubiquitin 2D HSQC + 3D HNCACB + 1UBQ 3D + 1D Stack + DOSY + Metabolomics
      </span>
    </div>
    <div class="demo-actions">
      <span id="gpu_cap_badge" class="cap-badge">Checking GPU&hellip;</span>
      <a class="manual-link" href="../index.html"><span class="label-desktop">&larr; Documentation Manual</span><span class="label-mobile">Manual</span></a>
      <a class="manual-link github-link" href="https://github.com/debsahu/nmr-assign" target="_blank" rel="noopener noreferrer">GitHub</a>
    </div>
  </header>

  <main id="canvas-wrap" class="canvas-stage">
    <div id="nmr_loading" role="status">
      <div id="nmr_spinner" class="spinner"></div>
      <div id="nmr_status_box" style="font-size: 14px; letter-spacing: 0.01em;">
        Initializing <strong>FERRIC</strong> WebAssembly + WebGPU Workbench&hellip;
      </div>
    </div>
    <div id="fallback" role="alert" hidden></div>
    <canvas id="nmr_canvas"></canvas>
  </main>

  <script type="module">
    const badgeEl = document.getElementById("gpu_cap_badge");
    const loadingEl = document.getElementById("nmr_loading");
    const fallbackEl = document.getElementById("fallback");

    function hasWebGL2() {
      try {
        const probe = document.createElement("canvas");
        return Boolean(probe.getContext("webgl2"));
      } catch (_) {
        return false;
      }
    }

    async function detectCapabilities() {
      const webgl2 = hasWebGL2();
      if (typeof navigator !== "undefined" && "gpu" in navigator && navigator.gpu) {
        try {
          const adapter = await navigator.gpu.requestAdapter();
          if (adapter) {
            return { webgpu: true, webgl2, detail: "WebGPU Ready" };
          }
          return { webgpu: false, webgl2, detail: "No WebGPU" };
        } catch (_) {
          return { webgpu: false, webgl2, detail: "WebGPU Blocked" };
        }
      }
      return { webgpu: false, webgl2, detail: webgl2 ? "WebGL2 Only" : "No WebGPU" };
    }

    function showBanner(title, paragraphsHtml, badgeText, badgeClass) {
      if (badgeEl) {
        badgeEl.textContent = badgeText;
        badgeEl.className = "cap-badge " + badgeClass;
      }
      if (loadingEl) {
        loadingEl.style.display = "none";
      }
      if (fallbackEl) {
        fallbackEl.innerHTML =
          '<div class="status-card">' +
          "<h2>" + title + "</h2>" +
          paragraphsHtml +
          '<p style="margin-top:14px"><a class="manual-link" href="../index.html">&larr; Return to Documentation Manual</a></p>' +
          "</div>";
        fallbackEl.hidden = false;
      }
    }

    async function bootDemo() {
      const caps = await detectCapabilities();
      if (!caps.webgpu) {
        const glNote = caps.webgl2
          ? "Your browser supports <code>WebGL2</code>, whereas <code>FERRIC</code>'s zero-install GPU contour and 3D ribbon shaders use <code>WebGPU</code> (<code>navigator.gpu</code>)."
          : "Neither <code>WebGPU</code> (<code>navigator.gpu</code>) nor <code>WebGL2</code> could be initialized in this browser context.";
        showBanner(
          "WebGPU Adapter Required for Live Browser Workbench",
          "<p>" + glNote + "</p>" +
          '<p class="hint-line"><strong>iOS / iPadOS 18+ Safari:</strong> Settings &rarr; Apps &rarr; Safari &rarr; Advanced &rarr; Feature Flags &rarr; enable <code>WebGPU</code>.</p>' +
          '<p class="hint-line"><strong>Android 12+ Chrome 121+:</strong> Enabled by default on supported GPUs (or toggle <code>chrome://flags/#enable-unsafe-webgpu</code>).</p>' +
          '<p class="hint-line"><strong>Desktop:</strong> Chrome / Edge 113+, Safari 18+, or Firefox 130+ with hardware acceleration, or run <code>cargo run --release -p nmr-gui --bin ferric-gui</code>.</p>',
          caps.detail,
          caps.webgl2 ? "warn" : "err"
        );
        return;
      }

      if (badgeEl) {
        badgeEl.textContent = "WebGPU Ready";
        badgeEl.className = "cap-badge ok";
      }

      const bundlePath = "./ferric" + "_gui.js";
      try {
        const wasmModule = await import(bundlePath);
        if (typeof wasmModule.default === "function") {
          await wasmModule.default();
        }
      } catch (err) {
        const msg = err && err.message ? String(err.message) : String(err);
        showBanner(
          "WebAssembly Demo Bundle Not Found or Failed to Load",
          "<p>WebGPU is available in your browser, but <code>ferric_gui.js</code> / <code>ferric_gui_bg.wasm</code> has not been built into this directory yet.</p>" +
          "<p>Build the release <code>wasm32-unknown-unknown</code> bundle locally with:</p>" +
          "<p><code>bash scripts/build_web_demo.sh --out target/manual/demo</code></p>" +
          "<p style=\"font-size:12px\">Detail: <code>" + msg.replace(/[<>&]/g, "") + "</code></p>",
          "Bundle Missing",
          "warn"
        );
      }
    }

    if (loadingEl) {
      bootDemo();
    }
  </script>
</body>
</html>
DEMO_EOF
}

validate_demo_shells() {
  local tmp_expected
  tmp_expected="$(mktemp)"
  render_demo_shell_html > "$tmp_expected"
  if ! diff -u "$tmp_expected" "$DEMO_HTML" >/dev/null; then
    echo "error: docs/demo/index.html is out of sync with render_demo_shell_html in scripts/build_web_demo.sh" >&2
    diff -u "$tmp_expected" "$DEMO_HTML" >&2 || true
    rm -f "$tmp_expected"
    exit 1
  fi
  rm -f "$tmp_expected"

  python3 - "$DEMO_HTML" "$TRUNK_HTML" <<'PY'
import sys
from pathlib import Path

demo_text = Path(sys.argv[1]).read_text(encoding="utf-8")
trunk_text = Path(sys.argv[2]).read_text(encoding="utf-8")

shared_tokens = [
    'name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover"',
    "viewport-fit=cover",
    'name="theme-color" content="#0e131b"',
    'name="mobile-web-app-capable" content="yes"',
    'name="apple-mobile-web-app-capable" content="yes"',
    "100vh",
    "100dvh",
    "env(safe-area-inset-top",
    "env(safe-area-inset-right",
    "env(safe-area-inset-bottom",
    "env(safe-area-inset-left",
    "touch-action: none;",
    "-webkit-user-select: none;",
    "user-select: none;",
    "-webkit-touch-callout: none;",
    "overscroll-behavior: none;",
    "#canvas-wrap",
    'id="canvas-wrap"',
    'id="nmr_canvas"',
    'id="nmr_loading"',
]

forbidden_wcag_tokens = [
    "user-scalable=no",
    "maximum-scale=1.0",
]

errors = []
for label, doc in (("docs/demo/index.html", demo_text), ("crates/nmr-gui/index.html", trunk_text)):
    for tok in shared_tokens:
        if tok not in doc:
            errors.append(f"{label}: missing required mobile WebGPU token {tok!r}")
    for bad in forbidden_wcag_tokens:
        if bad in doc:
            errors.append(f"{label}: forbidden WCAG 1.4.4 zoom-blocking viewport token {bad!r}")
    if doc.find("100vh") > doc.find("100dvh"):
        errors.append(f"{label}: 100vh fallback must precede 100dvh")

demo_only_tokens = [
    "header.demo-bar",
    "@media (max-width: 680px)",
    "#fallback",
    'id="fallback"',
    "overflow-y: auto;",
    "-webkit-overflow-scrolling: touch;",
    "Feature Flags",
    "Chrome 121+",
    ">Manual<",
    ">GitHub<",
]
for tok in demo_only_tokens:
    if tok not in demo_text:
        errors.append(f"docs/demo/index.html: missing required demo-shell token {tok!r}")

if errors:
    for err in errors:
        print(f"error: {err}", file=sys.stderr)
    sys.exit(1)
PY
}

validate_demo_shells

WB_LOCK_VERSION="$(python3 - "$REPO_ROOT/Cargo.lock" <<'PY'
import re, sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8")
m = re.search(r'\[\[package\]\]\s+name\s*=\s*"wasm-bindgen"\s+version\s*=\s*"([^"]+)"', text)
if not m:
    sys.exit(1)
print(m.group(1))
PY
)"

if [[ "$CHECK_ONLY" -eq 1 ]]; then
  echo "[web-demo] Checking crates/nmr-gui for wasm32-unknown-unknown (wasm-bindgen $WB_LOCK_VERSION)..."
  cargo check --manifest-path "$REPO_ROOT/Cargo.toml" --release -p nmr-gui --target wasm32-unknown-unknown
  mkdir -p "$OUT_DIR"
  render_demo_shell_html > "$OUT_DIR/index.html"
  echo "[web-demo] Check-only passed -> $(basename "$OUT_DIR")/index.html"
  exit 0
fi

ensure_wasm_bindgen_cli() {
  local have_ver=""
  if command -v wasm-bindgen >/dev/null 2>&1; then
    have_ver="$(wasm-bindgen --version | awk '{print $2}')"
  fi
  if [[ "$have_ver" == "$WB_LOCK_VERSION" ]]; then
    return 0
  fi
  if [[ "$INSTALL_CLI" -eq 0 ]]; then
    echo "error: wasm-bindgen-cli $WB_LOCK_VERSION is required (found '${have_ver:-none}')." >&2
    echo "hint: re-run with --install-cli or run: cargo install --locked wasm-bindgen-cli --version $WB_LOCK_VERSION" >&2
    exit 1
  fi
  echo "[web-demo] Installing wasm-bindgen-cli $WB_LOCK_VERSION..."
  cargo install --locked wasm-bindgen-cli --version "$WB_LOCK_VERSION"
}

ensure_wasm_bindgen_cli

echo "[web-demo] Building nmr-gui (release, wasm32-unknown-unknown)..."
cargo build --manifest-path "$REPO_ROOT/Cargo.toml" --release -p nmr-gui --target wasm32-unknown-unknown

WASM_BIN="$REPO_ROOT/target/wasm32-unknown-unknown/release/ferric-gui.wasm"
if [[ ! -f "$WASM_BIN" ]]; then
  WASM_BIN="$REPO_ROOT/target/wasm32-unknown-unknown/release/nmrassign-gui.wasm"
fi
if [[ ! -f "$WASM_BIN" ]]; then
  echo "error: expected compiled WASM binary at ferric-gui.wasm (or fallback nmrassign-gui.wasm)" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
echo "[web-demo] Generating JS/WASM web bindings via wasm-bindgen $WB_LOCK_VERSION..."
wasm-bindgen \
  --target web \
  --no-typescript \
  --out-dir "$OUT_DIR" \
  --out-name ferric_gui \
  "$WASM_BIN"

render_demo_shell_html > "$OUT_DIR/index.html"
touch "$OUT_DIR/.nojekyll"

for expected in "index.html" "ferric_gui.js" "ferric_gui_bg.wasm"; do
  if [[ ! -s "$OUT_DIR/$expected" ]]; then
    echo "error: missing or empty output artifact $expected" >&2
    exit 1
  fi
done

echo "[web-demo] Built zero-install WebGPU demo bundle in $(basename "$OUT_DIR")/"
