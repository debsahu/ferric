#!/usr/bin/env bash
# Deterministic builder for the zero-install wasm32 WebGPU browser demo (T16.4).
#
# Compiles `crates/nmr-gui` for `wasm32-unknown-unknown`, wraps the binary with
# `wasm-bindgen --target web`, and stages `docs/demo/index.html` into the target
# output directory ready for local static serving or GitHub Pages deployment.
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
  --out <dir>      Destination folder for index.html + nmrassign_gui.{js,wasm} (default: target/web-demo)
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
if [[ ! -f "$DEMO_HTML" ]]; then
  echo "error: missing $DEMO_HTML" >&2
  exit 1
fi

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
  cp "$DEMO_HTML" "$OUT_DIR/index.html"
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

WASM_BIN="$REPO_ROOT/target/wasm32-unknown-unknown/release/nmrassign-gui.wasm"
if [[ ! -f "$WASM_BIN" ]]; then
  echo "error: expected compiled WASM binary at nmrassign-gui.wasm" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
echo "[web-demo] Generating JS/WASM web bindings via wasm-bindgen $WB_LOCK_VERSION..."
wasm-bindgen \
  --target web \
  --no-typescript \
  --out-dir "$OUT_DIR" \
  --out-name nmrassign_gui \
  "$WASM_BIN"

cp "$DEMO_HTML" "$OUT_DIR/index.html"
touch "$OUT_DIR/.nojekyll"

for expected in "index.html" "nmrassign_gui.js" "nmrassign_gui_bg.wasm"; do
  if [[ ! -s "$OUT_DIR/$expected" ]]; then
    echo "error: missing or empty output artifact $expected" >&2
    exit 1
  fi
done

echo "[web-demo] Built zero-install WebGPU demo bundle in $(basename "$OUT_DIR")/"
