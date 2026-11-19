#!/usr/bin/env python3
"""Deterministic, zero-dependency static HTML5 Interactive Tour & Cinema builder.

Builds a self-contained, offline-first GitHub Pages Interactive Showcase Cinema
from ``ferric export-tour`` JSON bundles (schema ``"ferric-tour-v1"``):
- Responsive gallery index (``index.html``) with 2x4 desktop/tablet and 1x8 mobile
  card grid, validation-scope legend, live animated mini-canvas previews, and
  links back to the User Manual (``../index.html``) and Live WebGPU Workbench
  (``../demo/index.html``).
- Per-module dual-mode player pages (``<slug>.html``) supporting ``?embed=1``
  for seamless iframe embedding inside ``docs/manual/01..08.html``:
  1. Interactive Live Player (60fps HiDPI HTML5 <canvas>, hover/tap tooltips,
     scrubbable timeline with Chapter 1-4 markers, 0.5x-2x speed, loop,
     per-module interactive scientific controls, and MediaRecorder .webm export)
  2. Pre-Rendered HD Video (<video controls playsinline> with .webm + .mp4
     sources, chapter jump buttons, and direct download pills)
- Shared dark/light theme toggle synchronized with the User Manual via
  ``localStorage.getItem("nmr_manual_theme")`` and ``[data-theme="dark"]``.
- Strict ``--check`` and ``--self-test`` modes verifying determinism, zero
  external CDN URLs, WCAG 1.4.4 zoom-preserving viewport meta, explicit
  SYNTHETIC showcase disclosures on modules 04-08, clean-room token hygiene,
  and valid ES2020 JS syntax.

Provenance: Original clean-room Python 3 standard-library build script (MIT License).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _read_workspace_version() -> str:
    cargo_toml = Path(__file__).resolve().parent.parent / "Cargo.toml"
    if cargo_toml.is_file():
        in_ws_pkg = False
        for raw_line in cargo_toml.read_text(encoding="utf-8").splitlines():
            stripped = raw_line.strip()
            if stripped.startswith("["):
                in_ws_pkg = stripped == "[workspace.package]"
                continue
            if in_ws_pkg:
                m = re.match(r'^version\s*=\s*"([^"]+)"', stripped)
                if m:
                    return m.group(1)
    return "0.14.0"


SITE_VERSION = _read_workspace_version()
EXPECTED_SCHEMA_VERSION = "ferric-tour-v1"

RESTRICTED_PRODUCT_TOKENS: Tuple[str, ...] = (
    "am" + "ix",
    "che" + "nomx",
    "metabo" + "analyst",
    "nmrproc" + "flow",
    "sim" + "ca",
    "ico" + "shift",
    "bar" + "asa",
    "cy" + "ana",
    "fl" + "ya",
    "art" + "ina",
    "nmr" + "tist",
    "xp" + "lor",
    "cc" + "pn",
    "mn" + "ova",
    "mestre" + "nova",
    "che" + "mex",
    "ca" + "tia",
    "gl" + "ove",
    "gua" + "rdd",
    "ti" + "tan",
    "ta" + "los",
    "deep " + "picker",
    "ar" + "ia",
    "com" + "pass",
    "model" + "free4",
    "fast-model" + "free",
    "ten" + "sor2",
    "nes" + "sie",
    "ring" + "nmr",
)

BADGE_KIND_CLASSES: Dict[str, str] = {
    "empirical_calibrated": "badge-empirical",
    "fid_ingest_synthetic_nus": "badge-empirical",
    "cold_gate_failed_withdrawn": "badge-holdout",
    "cold_holdout_disclosed": "badge-holdout",
    "synthetic_showcase": "badge-synthetic",
    "synthetic_and_protocol_showcase": "badge-protocol",
    "protocol_showcase": "badge-protocol",
}


@dataclass(frozen=True)
class NormalizedBundle:
    module_index: int
    slug: str
    title: str
    subtitle: str
    manual_chapter: str
    badge_kind: str
    badge_label: str
    badge_detail: str
    duration_sec: float
    fps: int
    cli_command: str
    chapters: List[Dict[str, Any]]
    interactive_control: Dict[str, Any]
    raw_json: Dict[str, Any]


def safe_json_for_script_tag(obj: Any) -> str:
    """Serialize JSON deterministically and escape HTML script-closing sequences."""
    raw = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return raw.replace("</", "<\\/").replace("<!--", "\\u003c!--")


def normalize_bundle(data: Dict[str, Any], source_name: str) -> NormalizedBundle:
    """Validate and normalize a ``ferric-tour-v1`` JSON payload."""
    schema = data.get("schema_version")
    if schema != EXPECTED_SCHEMA_VERSION:
        raise ValueError(
            f"{source_name}: expected schema_version={EXPECTED_SCHEMA_VERSION!r}, got {schema!r}"
        )

    module_index = int(data.get("module_index", 0))
    if module_index < 1 or module_index > 8:
        raise ValueError(f"{source_name}: module_index must be in 1..8, got {module_index}")

    slug = str(data.get("slug", "")).strip()
    if not slug or not re.match(r"^[a-z0-9_]+$", slug):
        raise ValueError(f"{source_name}: invalid slug {slug!r}")

    title = str(data.get("title", f"Module {module_index:02d}")).strip()
    subtitle = str(data.get("subtitle", "")).strip()
    manual_chapter = str(data.get("manual_chapter", "index.html")).strip() or "index.html"
    duration_sec = float(data.get("duration_sec", 16.0))
    if duration_sec <= 0.0:
        duration_sec = 16.0
    fps = int(data.get("fps", 30))
    if fps <= 0:
        fps = 30
    cli_command = str(data.get("cli_command", f"ferric export-tour --module {module_index:02d}")).strip()

    raw_badge = data.get("validation_badge")
    if not isinstance(raw_badge, dict):
        raw_badge = {}
    badge_kind = str(raw_badge.get("kind", "synthetic_showcase" if module_index >= 4 else "empirical_calibrated")).strip()
    badge_label = str(raw_badge.get("label", "VALIDATION SCOPE")).strip()
    badge_detail = str(raw_badge.get("detail", "")).strip()

    # Enforce candid synthetic showcase disclosure for modules 04-08
    if module_index >= 4 and "SYNTHETIC" not in (badge_label + " " + badge_detail).upper():
        prefix = "SYNTHETIC FORWARD-MODEL SHOWCASE"
        if module_index == 8:
            prefix = "SYNTHETIC & PROTOCOL SHOWCASE"
        badge_label = f"{prefix} — {badge_label}" if badge_label else prefix
        if not badge_detail:
            badge_detail = (
                "SYNTHETIC forward-model showcase; Phase 17 blind real-data gates "
                "F10-F18 remain open in TODO.md."
            )
        else:
            badge_detail = f"SYNTHETIC forward-model showcase: {badge_detail}"

    raw_chapters = data.get("chapters")
    if not isinstance(raw_chapters, list) or len(raw_chapters) == 0:
        raise ValueError(f"{source_name}: chapters must be a non-empty list")

    chapters: List[Dict[str, Any]] = []
    seg = duration_sec / float(len(raw_chapters))
    for i, ch in enumerate(raw_chapters):
        if not isinstance(ch, dict):
            raise ValueError(f"{source_name}: chapter {i} must be an object")
        start_sec = float(ch.get("start_sec", ch.get("start", i * seg)))
        end_sec = float(ch.get("end_sec", ch.get("end", (i + 1) * seg)))
        ch_title = str(ch.get("title", ch.get("name", f"Chapter {i + 1}"))).strip()
        ch_caption = str(
            ch.get("caption", ch.get("narrative", ch.get("description", "")))
        ).strip()
        ch_metric = str(
            ch.get("metric", ch.get("metric_pill", ch.get("equation", ch.get("highlight", ""))))
        ).strip()
        chapters.append(
            {
                "index": i + 1,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "title": ch_title,
                "caption": ch_caption,
                "metric": ch_metric,
            }
        )

    raw_ctrl = data.get("interactive_control")
    if not isinstance(raw_ctrl, dict):
        raw_ctrl = {}
    ctrl_min = float(raw_ctrl.get("min", 0.0))
    ctrl_max = float(raw_ctrl.get("max", 10.0))
    ctrl_step = float(raw_ctrl.get("step", 1.0))
    if ctrl_step <= 0.0:
        ctrl_step = 1.0
    ctrl_def = float(raw_ctrl.get("default_val", raw_ctrl.get("default", ctrl_min)))
    ctrl = {
        "id": str(raw_ctrl.get("id", f"param_{module_index}")).strip(),
        "label": str(raw_ctrl.get("label", "Parameter")).strip(),
        "min": ctrl_min,
        "max": ctrl_max,
        "step": ctrl_step,
        "default_val": ctrl_def,
        "unit": str(raw_ctrl.get("unit", "")).strip(),
    }

    normalized_json = dict(data)
    normalized_json["schema_version"] = EXPECTED_SCHEMA_VERSION
    normalized_json["module_index"] = module_index
    normalized_json["slug"] = slug
    normalized_json["title"] = title
    normalized_json["subtitle"] = subtitle
    normalized_json["manual_chapter"] = manual_chapter
    normalized_json["duration_sec"] = duration_sec
    normalized_json["fps"] = fps
    normalized_json["cli_command"] = cli_command
    normalized_json["validation_badge"] = {
        "kind": badge_kind,
        "label": badge_label,
        "detail": badge_detail,
    }
    normalized_json["chapters"] = chapters
    normalized_json["interactive_control"] = ctrl

    return NormalizedBundle(
        module_index=module_index,
        slug=slug,
        title=title,
        subtitle=subtitle,
        manual_chapter=manual_chapter,
        badge_kind=badge_kind,
        badge_label=badge_label,
        badge_detail=badge_detail,
        duration_sec=duration_sec,
        fps=fps,
        cli_command=cli_command,
        chapters=chapters,
        interactive_control=ctrl,
        raw_json=normalized_json,
    )


def load_tour_bundles(data_dir: Path) -> List[NormalizedBundle]:
    """Load and deterministically sort ``*.tour.json`` bundles from ``data_dir``."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data directory does not exist: {data_dir}")
    paths = sorted(data_dir.glob("*.tour.json"))
    if not paths:
        raise FileNotFoundError(f"no *.tour.json files found in {data_dir}")
    bundles: List[NormalizedBundle] = []
    for p in paths:
        parsed = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError(f"{p.name}: root JSON must be an object")
        bundles.append(normalize_bundle(parsed, p.name))
    bundles.sort(key=lambda b: (b.module_index, b.slug))
    return bundles


def format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    mins, secs = divmod(total, 60)
    return f"{mins}:{secs:02d}"


def badge_css_class(kind: str) -> str:
    return BADGE_KIND_CLASSES.get(kind, "badge-synthetic")


def render_poster_svg(bundle: NormalizedBundle) -> str:
    """Build a crisp deterministic fallback poster SVG for gallery cards."""
    idx = bundle.module_index
    accent_map = {
        1: "#38bdf8",
        2: "#22d3ee",
        3: "#60a5fa",
        4: "#f59e0b",
        5: "#fb923c",
        6: "#a78bfa",
        7: "#34d399",
        8: "#f472b6",
    }
    col = accent_map.get(idx, "#38bdf8")
    waves: List[str] = []
    for k in range(5):
        y = 32 + k * 24
        pts = []
        for step in range(17):
            x = 18 + step * 20
            amp = 14.0 if k == 2 else 8.0
            dy = amp * (((idx * 7 + step * 3 + k * 5) % 11) - 5) / 5.0
            pts.append(f"{x},{y + dy:.1f}")
        opacity = "0.85" if k == 2 else "0.32"
        waves.append(
            f'<polyline fill="none" stroke="{col}" stroke-opacity="{opacity}" '
            f'stroke-width="2" points="{" ".join(pts)}" />'
        )
    return (
        f'<svg class="mini-poster" viewBox="0 0 360 180" width="360" height="180" '
        f'aria-hidden="true" xmlns="http://www.w3.org/2000/svg">'
        f'<rect width="360" height="180" rx="8" fill="#0b1120" />'
        f'{"".join(waves)}'
        f'<text x="16" y="24" fill="#94a3b8" font-family="ui-monospace,monospace" '
        f'font-size="11">MOD {idx:02d} · {html.escape(bundle.slug)}</text>'
        f'</svg>'
    )


def render_gallery_html(bundles: Sequence[NormalizedBundle]) -> str:
    """Render the responsive Gallery Index (``index.html``)."""
    cards_html: List[str] = []
    gallery_manifest: List[Dict[str, Any]] = []

    for b in bundles:
        b_cls = badge_css_class(b.badge_kind)
        dur = format_duration(b.duration_sec)
        chapter_pills = "".join(
            f'<a class="ch-pill" href="{html.escape(b.slug)}.html?ch={ch["index"]}" '
            f'title="{html.escape(ch["title"])}">Ch {ch["index"]}: {html.escape(ch["title"])}</a>'
            for ch in b.chapters
        )
        synth_note = ""
        if b.module_index >= 4:
            synth_note = (
                '<p class="card-disclosure">'
                '<strong>SYNTHETIC SHOWCASE:</strong> Deterministic forward-model physics; '
                'Phase 17 real-data gates open.'
                "</p>"
            )
        elif b.module_index == 3:
            synth_note = (
                '<p class="card-disclosure card-disclosure-holdout">'
                '<strong>F8 HOLDOUT DISCLOSED:</strong> Unseen 1KHM &rarr; 1ZZK '
                '1.86&ndash;4.14 &Aring; (0/5 &lt; 1.80 &Aring;).'
                "</p>"
            )

        cards_html.append(
            f'<article class="tour-card" data-module="{b.module_index}">'
            f'  <a class="card-stage-link" href="{html.escape(b.slug)}.html" '
            f'     aria-label="Open {html.escape(b.title)} interactive tour">'
            f'    <div class="card-stage">'
            f'      {render_poster_svg(b)}'
            f'      <canvas class="mini-canvas" data-module-index="{b.module_index}" '
            f'              width="360" height="180" aria-hidden="true"></canvas>'
            f'      <span class="card-mod-pill">Module {b.module_index:02d}</span>'
            f'      <span class="card-dur-pill">{html.escape(dur)}</span>'
            f"    </div>"
            f"  </a>"
            f'  <div class="card-body">'
            f'    <div class="card-badge-row">'
            f'      <span class="val-badge {b_cls}">{html.escape(b.badge_label)}</span>'
            f"    </div>"
            f'    <h2 class="card-title"><a href="{html.escape(b.slug)}.html">{html.escape(b.title)}</a></h2>'
            f'    <p class="card-subtitle">{html.escape(b.subtitle)}</p>'
            f"    {synth_note}"
            f'    <div class="ch-pills">{chapter_pills}</div>'
            f'    <div class="card-actions">'
            f'      <a class="btn-primary" href="{html.escape(b.slug)}.html">&#127918; Launch Interactive Tour</a>'
            f'      <a class="btn-secondary" href="{html.escape(b.slug)}.html?mode=video">&#127916; Watch MP4 / WebM</a>'
            f"    </div>"
            f"  </div>"
            f"</article>"
        )

        gallery_manifest.append(
            {
                "module_index": b.module_index,
                "slug": b.slug,
                "title": b.title,
                "badge_kind": b.badge_kind,
                "payload": b.raw_json.get("payload", {}),
                "interactive_control": b.interactive_control,
            }
        )

    manifest_json = safe_json_for_script_tag(gallery_manifest)

    m02_legend_text = "Module 02 live solver fixture + historical benchmark disclosure"
    for b in bundles:
        if b.module_index == 2:
            bm = b.raw_json.get("payload", {}).get("benchmarks", {})
            if isinstance(bm, dict) and all(
                k in bm
                for k in (
                    "ubiquitin_realistic_pct",
                    "cisd3_acc_pct",
                    "cisd3_hc_prec_pct",
                    "cold_bmrb_hc_min_pct",
                    "cold_bmrb_hc_max_pct",
                    "cold_bmrb_gate_pct",
                    "cold_bmrb_withdrawal_ref",
                    "dev_target",
                )
            ):
                # Every scalar below is read from payload.benchmarks (B15); a missing key falls back to
                # the scalar-free generic legend above and is reported by --check as a missing payload key.
                m02_legend_text = (
                    f"Module 02 [historical disclosure] synthetic Ubiquitin ({float(bm['ubiquitin_realistic_pct']):.1f}%), "
                    f"dev {html.escape(str(bm['dev_target']))} ({float(bm['cisd3_acc_pct']):.2f}% / {float(bm['cisd3_hc_prec_pct']):.0f}% HC); "
                    f"default evo-graph cold HCP {float(bm['cold_bmrb_hc_min_pct']):.2f}&ndash;{float(bm['cold_bmrb_hc_max_pct']):.2f}% "
                    f"(latest {float(bm['cold_bmrb_hc_min_pct']):.2f}% FAILS &ge; {float(bm['cold_bmrb_gate_pct']):.0f}%; "
                    f"withdrawn {html.escape(str(bm['cold_bmrb_withdrawal_ref']))})"
                )
            break

    return f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover" />
  <title>FERRIC — Interactive Showcase Cinema &amp; Video Tours</title>
  <link rel="icon" type="image/svg+xml" href="../favicon.svg" />
  <link rel="icon" type="image/png" sizes="32x32" href="../favicon-32.png" />
  <link rel="stylesheet" href="tours.css" />
</head>
<body class="gallery-body">
  <header class="topbar">
    <div class="brand-wrap">
      <a class="brand-link" href="../index.html">
        <svg class="brand-icon" viewBox="0 0 512 512" width="22" height="22" aria-hidden="true">
          <rect x="32" y="32" width="448" height="448" rx="104" ry="104" fill="#16202A" stroke="#B7410E" stroke-width="24"/>
          <line x1="152" y1="124" x2="152" y2="388" stroke="#E05A2B" stroke-width="44" stroke-linecap="round"/>
          <line x1="152" y1="124" x2="376" y2="124" stroke="#E05A2B" stroke-width="44" stroke-linecap="round"/>
          <line x1="152" y1="248" x2="316" y2="248" stroke="#E05A2B" stroke-width="36" stroke-linecap="round"/>
          <path d="M 168 128 Q 205 318 242 248 T 316 268 T 414 256" fill="none" stroke="#FB923C" stroke-width="30" stroke-linecap="round"/>
          <circle cx="152" cy="124" r="20" fill="#FED7AA"/>
        </svg>
        <span>FERRIC</span>
      </a>
      <span class="version-pill">v{html.escape(SITE_VERSION)} &middot; Interactive Tours</span>
    </div>
    <nav class="topbar-actions" aria-label="Primary navigation">
      <a class="nav-pill" href="../index.html">User Manual</a>
      <a class="nav-pill nav-pill-accent" href="../demo/index.html">Live WebGPU Workbench</a>
      <button id="theme-toggle" class="theme-btn" type="button" aria-label="Toggle dark or light theme">Light</button>
    </nav>
  </header>

  <main class="gallery-main">
    <section class="hero-card">
      <div class="hero-copy">
        <p class="hero-eyebrow">Deterministic Clean-Room NMR Showcase &middot; Seed 20261801</p>
        <h1>Interactive Module Showcase Cinema</h1>
        <p class="hero-lead">
          Every tour below is generated directly from deterministic <code>nmr-core</code> outputs.
          Launch the <strong>60 fps HTML5 Interactive Live Player</strong> to scrub timelines, drag 3D
          backbones, inspect coordinates, and tweak live parameters &mdash; or switch instantly to
          pre-rendered <strong>1280&times;720 MP4 / WebM</strong> video.
        </p>
      </div>
      <div class="scope-legend" aria-label="Validation Scope Legend">
        <div class="legend-item">
          <span class="val-badge badge-empirical">Raw Bruker/Varian FID Ingest &middot; Synthetic NUS Demo</span>
          <span>Module 01 real Bruker/Varian raw-FID &amp; pulseprogram AST ingest; 25% Poisson-gap NUS IST/FISTA demo is synthetic</span>
        </div>
        <div class="legend-item">
          <span class="val-badge badge-holdout">Synthetic / Dev &middot; Cold HCP Gate Failed (Withdrawn)</span>
          <span>{m02_legend_text}</span>
        </div>
        <div class="legend-item">
          <span class="val-badge badge-holdout">Cold Holdout Disclosed (F8)</span>
          <span>Module 03 unseen 1KHM &rarr; 1ZZK disclosed (1.86&ndash;4.14 &Aring;, 0/5 &lt; 1.80 &Aring;)</span>
        </div>
        <div class="legend-item">
          <span class="val-badge badge-synthetic">Synthetic Forward-Model Showcase &mdash; F10..F17 open</span>
          <span>Modules 04&ndash;08 synthetic forward-model demonstrations; empirical gates open</span>
        </div>
        <div class="legend-item">
          <span class="val-badge badge-protocol">Protocol Showcase &mdash; F18 open</span>
          <span>Module 08 sandboxed MCP JSON-RPC 2.0 &amp; Undo bridge showcase</span>
        </div>
      </div>
    </section>

    <section class="tour-grid" aria-label="Interactive module tours">
      {"".join(cards_html)}
    </section>
  </main>

  <footer class="tours-footer">
    <span><strong>FERRIC</strong> v{html.escape(SITE_VERSION)} &middot; Clean-Room Rust + WebGPU NMR Workbench</span>
    <span class="tours-footer-links">
      <a href="../index.html">User Manual</a>
      <a href="../demo/index.html">Live WebGPU Workbench</a>
      <a href="https://github.com/debsahu/ferric" target="_blank" rel="noopener">GitHub</a>
    </span>
  </footer>

  <script type="application/json" id="tour-gallery-data">{manifest_json}</script>
  <script src="tours_player.js" defer></script>
</body>
</html>
"""


def render_module_html(
    bundle: NormalizedBundle,
    bundles: Sequence[NormalizedBundle],
    videos_rel: str,
) -> str:
    """Render a per-module dual-mode Interactive Cinema Player page."""
    b_cls = badge_css_class(bundle.badge_kind)
    ctrl = bundle.interactive_control
    embedded_json = safe_json_for_script_tag(bundle.raw_json)
    v_rel = videos_rel.rstrip("/")
    webm_src = f"{v_rel}/{bundle.slug}.webm"
    mp4_src = f"{v_rel}/{bundle.slug}.mp4"

    # Explicit scientific candor disclosure block
    disclosure_extra = ""
    if bundle.module_index >= 4:
        disclosure_extra = (
            " <strong>SYNTHETIC FORWARD-MODEL SHOWCASE:</strong> Outputs reflect deterministic "
            "forward-model physics in <code>nmr-core</code>; Phase 17 blind real-data empirical "
            "gates (<code>F10</code>&ndash;<code>F17</code>) remain open in <code>TODO.md</code>."
        )
        if bundle.module_index == 8:
            disclosure_extra += " Includes <strong>PROTOCOL SHOWCASE (F18 open)</strong> for MCP JSON-RPC 2.0."
    elif bundle.module_index == 3:
        disclosure_extra = (
            " <strong>F8 COLD HOLDOUT DISCLOSED:</strong> Dev 1D3Z + unseen 1KHM &rarr; 1ZZK "
            "improves NOE-only on 5/5 seeds (1.86&ndash;4.14 &Aring;, 0/5 &lt; 1.80 &Aring;)."
        )

    # Chapter timeline markers & video jump buttons
    chapter_markers_html: List[str] = []
    video_jumps_html: List[str] = []
    for ch in bundle.chapters:
        idx = ch["index"]
        start_s = ch["start_sec"]
        chapter_markers_html.append(
            f'<button class="ch-marker" type="button" data-ch="{idx}" data-start="{start_s:.3f}" '
            f'title="Key {idx}: {html.escape(ch["title"])}">'
            f'<span class="ch-num">{idx}</span>'
            f'<span class="ch-lbl">{html.escape(ch["title"])}</span>'
            f"</button>"
        )
        video_jumps_html.append(
            f'<button class="vid-jump-btn" type="button" data-ch="{idx}" data-start="{start_s:.3f}">'
            f'Ch {idx} ({format_duration(start_s)}): {html.escape(ch["title"])}'
            f"</button>"
        )

    # Module-specific quick preset chips
    preset_chips: List[Tuple[str, float]] = []
    if bundle.module_index == 1:
        preset_chips = [("Zero-Fill (0)", 0), ("IST 5", 5), ("IST 12", 12), ("Converged (20)", 20)]
    elif bundle.module_index == 2:
        preset_chips = [("Phe4", 4), ("Lys6", 6), ("Thr7 (Inv)", 7), ("Gly10 (Null)", 10)]
    elif bundle.module_index == 3:
        preset_chips = [("0° Front", 0), ("90° Side", 90), ("180° Back", 180), ("270°", 270)]
    elif bundle.module_index == 4:
        # R_ex chips come from the fitted payload.rex_sites (B15), never a hard-coded residue.
        p4 = bundle.raw_json.get("payload", {}) or {}
        rex_sites = p4.get("rex_sites") if isinstance(p4, dict) else None
        if isinstance(rex_sites, list):
            for site in rex_sites[:2]:
                if isinstance(site, dict) and isinstance(site.get("seq_id"), (int, float)):
                    preset_chips.append(
                        (f"{site.get('label', site['seq_id'])} ({site.get('model', 'R_ex')} Rex)", site["seq_id"])
                    )
        preset_chips += [("Core Res 45", 45), ("Tail Gly75", 75)]
    elif bundle.module_index == 5:
        preset_chips = [("0 uM", 0), ("50 uM", 50), ("100 uM (Equiv)", 100), ("250 uM", 250)]
    elif bundle.module_index == 6:
        preset_chips = [("COOH (0)", 0), ("Chiral CH (3)", 3), ("Aryl (6)", 6), ("iBu (11)", 11)]
    elif bundle.module_index == 7:
        preset_chips = [("g1 (Low)", 0), ("g6 (Mid)", 5), ("g12", 11), ("g16 (High)", 15)]
    elif bundle.module_index == 8:
        preset_chips = [("1.33 ppm Lactate", 4), ("3.04 ppm Creatine", 10), ("3.56 ppm Glycine", 14)]

    chips_html = "".join(
        f'<button class="preset-chip" type="button" data-val="{val}">{html.escape(label)}</button>'
        for label, val in preset_chips
    )

    mcp_tab_html = ""
    if bundle.module_index == 8:
        mcp_tab_html = (
            '<div class="mcp-inspector-wrap">'
            '  <button id="btn-mcp-toggle" class="btn-outline" type="button" aria-expanded="false">'
            "    &#128268; MCP JSON-RPC Packet"
            "  </button>"
            '  <pre id="mcp-packet-drawer" class="mcp-packet-drawer" hidden></pre>'
            "</div>"
        )

    # Prev / Next navigation
    pos = next((i for i, item in enumerate(bundles) if item.slug == bundle.slug), 0)
    prev_link = (
        f'<a class="nav-adj" href="{html.escape(bundles[pos - 1].slug)}.html">&larr; {bundles[pos - 1].module_index:02d}</a>'
        if pos > 0
        else '<span class="nav-adj disabled">&larr; Start</span>'
    )
    next_link = (
        f'<a class="nav-adj" href="{html.escape(bundles[pos + 1].slug)}.html">{bundles[pos + 1].module_index:02d} &rarr;</a>'
        if pos + 1 < len(bundles)
        else '<span class="nav-adj disabled">End &rarr;</span>'
    )

    ch0 = bundle.chapters[0]
    stage_hint = (
        "Drag to 3D Orbit · Hover/Tap Atoms"
        if bundle.module_index == 3
        else "Hover or Tap Canvas Features to Inspect"
    )
    canvas_cls = "tour-canvas is-orbit-canvas" if bundle.module_index == 3 else "tour-canvas"

    return f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover" />
  <title>Module {bundle.module_index:02d}: {html.escape(bundle.title)} — FERRIC Tour</title>
  <link rel="icon" type="image/svg+xml" href="../favicon.svg" />
  <link rel="icon" type="image/png" sizes="32x32" href="../favicon-32.png" />
  <link rel="stylesheet" href="tours.css" />
</head>
<body class="player-body" data-module="{bundle.module_index}">
  <header class="topbar player-topbar">
    <div class="brand-wrap">
      <a class="brand-link" href="../index.html" title="FERRIC User Manual">
        <svg class="brand-icon" viewBox="0 0 512 512" width="20" height="20" aria-hidden="true">
          <rect x="32" y="32" width="448" height="448" rx="104" ry="104" fill="#16202A" stroke="#B7410E" stroke-width="24"/>
          <line x1="152" y1="124" x2="152" y2="388" stroke="#E05A2B" stroke-width="44" stroke-linecap="round"/>
          <line x1="152" y1="124" x2="376" y2="124" stroke="#E05A2B" stroke-width="44" stroke-linecap="round"/>
          <line x1="152" y1="248" x2="316" y2="248" stroke="#E05A2B" stroke-width="36" stroke-linecap="round"/>
          <path d="M 168 128 Q 205 318 242 248 T 316 268 T 414 256" fill="none" stroke="#FB923C" stroke-width="30" stroke-linecap="round"/>
          <circle cx="152" cy="124" r="20" fill="#FED7AA"/>
        </svg>
      </a>
      <a class="nav-pill" href="index.html">&larr; All Tours</a>
      <a class="nav-pill hide-compact" href="../index.html">Manual</a>
      <a class="nav-pill nav-pill-accent hide-compact" href="../demo/index.html">WebGPU Demo</a>
      <span class="player-title-inline">
        <strong>{bundle.module_index:02d}</strong> {html.escape(bundle.title)}
      </span>
    </div>

    <div class="mode-switcher" role="tablist" aria-label="Tour playback mode">
      <button id="mode-interactive" class="mode-tab is-active" type="button" role="tab" aria-selected="true">
        &#127918; Interactive Live Player
      </button>
      <button id="mode-video" class="mode-tab" type="button" role="tab" aria-selected="false">
        &#127916; Pre-Rendered HD Video
      </button>
    </div>

    <div class="topbar-actions">
      <button id="theme-toggle" class="theme-btn" type="button" aria-label="Toggle dark or light theme">Light</button>
    </div>
  </header>

  <div class="validation-banner {b_cls}" role="region" aria-label="Validation scope disclosure">
    <span class="val-badge {b_cls}">{html.escape(bundle.badge_label)}</span>
    <span class="val-detail">{html.escape(bundle.badge_detail)}{disclosure_extra}</span>
  </div>

  <main class="cinema-shell">
    <section id="panel-interactive" class="player-panel" role="tabpanel" aria-labelledby="mode-interactive">
      <div id="canvas-wrap" class="stage-wrap">
        <canvas id="tour-canvas" class="{canvas_cls}" aria-label="Interactive simulation canvas for {html.escape(bundle.title)}"></canvas>
        <div id="stage-overlay-pill" class="stage-overlay-pill">
          <span id="overlay-ch-label">Ch 1/4</span>
          <span class="overlay-sep">&middot;</span>
          <span>{html.escape(stage_hint)}</span>
        </div>
        <div id="fallback-banner" class="fallback-banner" role="alert" hidden></div>
        <div id="hud-tooltip" class="hud-tooltip" hidden>
          <strong id="hud-tooltip-title"></strong>
          <div id="hud-tooltip-body" class="hud-tooltip-body"></div>
        </div>
      </div>

      <div class="control-strip">
        <div class="ctrl-primary">
          <label class="ctrl-label" for="ctrl-slider">
            <span id="ctrl-name">{html.escape(ctrl["label"])}</span>:
            <strong id="ctrl-readout">{ctrl["default_val"]:g}</strong>
            <span class="ctrl-unit">{html.escape(ctrl["unit"])}</span>
          </label>
          <div class="ctrl-slider-row">
            <button id="ctrl-dec" class="step-btn" type="button" aria-label="Decrease parameter">&minus;</button>
            <input id="ctrl-slider" type="range"
                   min="{ctrl["min"]:g}" max="{ctrl["max"]:g}"
                   step="{ctrl["step"]:g}" value="{ctrl["default_val"]:g}" />
            <button id="ctrl-inc" class="step-btn" type="button" aria-label="Increase parameter">+</button>
            <button id="ctrl-resume" class="btn-outline" type="button" hidden>Resume Auto</button>
          </div>
        </div>
        <div class="ctrl-presets">
          {chips_html}
          {mcp_tab_html}
        </div>
      </div>

      <div class="transport-bar" aria-label="Cinema playback transport">
        <div class="transport-left">
          <button id="btn-play" class="play-btn" type="button" aria-label="Play or pause">
            &#9654; Play
          </button>
          <span id="time-readout" class="time-readout">0:00.0 / {format_duration(bundle.duration_sec)}.0</span>
        </div>

        <div class="timeline-wrap">
          <input id="timeline-range" class="timeline-range" type="range"
                 min="0" max="{bundle.duration_sec:.3f}" step="0.033" value="0"
                 aria-label="Timeline scrubber" />
          <div class="ch-markers">
            {"".join(chapter_markers_html)}
          </div>
        </div>

        <div class="transport-right">
          <div class="speed-group" role="group" aria-label="Playback speed">
            <button class="speed-btn" type="button" data-speed="0.5">0.5&times;</button>
            <button class="speed-btn is-active" type="button" data-speed="1">1&times;</button>
            <button class="speed-btn" type="button" data-speed="1.5">1.5&times;</button>
            <button class="speed-btn" type="button" data-speed="2">2&times;</button>
          </div>
          <button id="btn-loop" class="btn-outline is-active" type="button" aria-pressed="true">Loop</button>
          <button id="btn-export-webm" class="btn-outline" type="button">&#127909; Export .webm</button>
          <span id="export-status" class="export-status" aria-live="polite"></span>
        </div>
      </div>
    </section>

    <section id="panel-video" class="player-panel" role="tabpanel" aria-labelledby="mode-video" hidden>
      <div class="video-stage-wrap">
        <video id="tour-video" class="tour-video" controls playsinline preload="metadata">
          <source src="{html.escape(webm_src)}" type="video/webm" />
          <source src="{html.escape(mp4_src)}" type="video/mp4" />
          Your browser does not support HTML5 video playback.
        </video>
      </div>
      <div class="video-controls-bar">
        <div class="video-jumps" aria-label="Jump to video chapter">
          {"".join(video_jumps_html)}
        </div>
        <div class="video-downloads">
          <a class="btn-primary" href="{html.escape(mp4_src)}" download="{html.escape(bundle.slug)}.mp4">&#11015; Download MP4</a>
          <a class="btn-secondary" href="{html.escape(webm_src)}" download="{html.escape(bundle.slug)}.webm">&#11015; Download WebM</a>
        </div>
      </div>
    </section>

    <footer class="bottom-drawer">
      <div class="drawer-narrative">
        <div class="drawer-title-row">
          <span id="drawer-ch-badge" class="drawer-ch-badge">Chapter 1 / {len(bundle.chapters)}</span>
          <h1 id="drawer-ch-title" class="drawer-ch-title">{html.escape(ch0["title"])}</h1>
          <span id="drawer-metric" class="drawer-metric">{html.escape(ch0["metric"])}</span>
        </div>
        <p id="drawer-caption" class="drawer-caption">{html.escape(ch0["caption"])}</p>
      </div>

      <div class="drawer-actions">
        <div class="cli-box">
          <code id="cli-cmd">{html.escape(bundle.cli_command)}</code>
          <button id="btn-copy-cli" class="copy-btn" type="button">Copy CLI</button>
        </div>
        <div class="drawer-links">
          <a class="nav-pill" href="../{html.escape(bundle.manual_chapter)}">Open Manual Chapter &rarr;</a>
          <a class="nav-pill" href="https://github.com/debsahu/ferric" target="_blank" rel="noopener">GitHub</a>
          <div class="adj-wrap">{prev_link}{next_link}</div>
        </div>
      </div>
    </footer>
  </main>

  <script type="application/json" id="tour-bundle-data">{embedded_json}</script>
  <script src="tours_player.js" defer></script>
</body>
</html>
"""


TOURS_CSS = r""":root {
  --bg: #f8fafc;
  --bg-elev: #ffffff;
  --bg-subtle: #f1f5f9;
  --bg-stage: #09111f;
  --fg: #0f172a;
  --fg-muted: #475569;
  --fg-soft: #64748b;
  --border: #cbd5e1;
  --accent: #B7410E;
  --accent-soft: rgba(183, 65, 14, 0.12);
  --accent-strong: #9A3412;
  --ok-bg: rgba(16, 185, 129, 0.14);
  --ok-fg: #047857;
  --ok-bd: rgba(16, 185, 129, 0.38);
  --warn-bg: rgba(217, 119, 6, 0.14);
  --warn-fg: #b45309;
  --warn-bd: rgba(217, 119, 6, 0.42);
  --synth-bg: rgba(234, 88, 12, 0.14);
  --synth-fg: #c2410c;
  --synth-bd: rgba(234, 88, 12, 0.42);
  --proto-bg: rgba(124, 58, 237, 0.14);
  --proto-fg: #6d28d9;
  --proto-bd: rgba(124, 58, 237, 0.42);
  --shadow: 0 6px 24px rgba(15, 23, 42, 0.08);
}

[data-theme="dark"] {
  --bg: #0b1120;
  --bg-elev: #111827;
  --bg-subtle: #1e293b;
  --bg-stage: #070d19;
  --fg: #e2e8f0;
  --fg-muted: #94a3b8;
  --fg-soft: #64748b;
  --border: #253347;
  --accent: #FB923C;
  --accent-soft: rgba(251, 146, 60, 0.14);
  --accent-strong: #FDBA74;
  --ok-bg: rgba(16, 185, 129, 0.16);
  --ok-fg: #6ee7b7;
  --ok-bd: rgba(110, 231, 183, 0.34);
  --warn-bg: rgba(245, 158, 11, 0.16);
  --warn-fg: #fbbf24;
  --warn-bd: rgba(251, 191, 36, 0.36);
  --synth-bg: rgba(251, 146, 60, 0.16);
  --synth-fg: #fdba74;
  --synth-bd: rgba(251, 146, 60, 0.36);
  --proto-bg: rgba(167, 139, 250, 0.16);
  --proto-fg: #c4b5fd;
  --proto-bd: rgba(167, 139, 250, 0.36);
  --shadow: 0 10px 30px rgba(0, 0, 0, 0.45);
}

*, *::before, *::after { box-sizing: border-box; }
html {
  font-size: 16px;
  -webkit-text-size-adjust: 100%;
  scroll-behavior: smooth;
}
body {
  margin: 0;
  background: var(--bg);
  color: var(--fg);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  line-height: 1.5;
  overflow-x: hidden;
  min-height: 100vh;
  min-height: 100dvh;
}

a { color: var(--accent); text-decoration: none; }
a:hover { color: var(--accent-strong); text-decoration: underline; }
button, a, input { touch-action: manipulation; }

.topbar {
  position: sticky;
  top: 0;
  z-index: 40;
  min-height: calc(3.4rem + env(safe-area-inset-top, 0px));
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 0.65rem;
  padding: env(safe-area-inset-top, 0px) calc(1rem + env(safe-area-inset-right, 0px)) 0.35rem calc(1rem + env(safe-area-inset-left, 0px));
  background: var(--bg-elev);
  border-bottom: 1px solid var(--border);
}
.brand-wrap {
  display: flex;
  align-items: center;
  gap: 0.55rem;
  flex-wrap: wrap;
  min-width: 0;
}
.brand-link {
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  font-weight: 800;
  font-size: 1.05rem;
  color: var(--fg);
  white-space: nowrap;
}
.brand-icon { flex-shrink: 0; border-radius: 5px; }
.brand-link:hover { text-decoration: none; color: var(--accent); }
.tours-footer {
  max-width: 1240px;
  margin: 0 auto 1.5rem;
  padding: 0.9rem 1.25rem;
  border-top: 1px solid var(--border);
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  gap: 0.75rem;
  font-size: 0.85rem;
  color: var(--fg-muted);
}
.tours-footer-links { display: inline-flex; gap: 1rem; flex-wrap: wrap; }
.version-pill {
  font-size: 0.74rem;
  font-weight: 600;
  padding: 0.16rem 0.55rem;
  border-radius: 999px;
  background: var(--accent-soft);
  color: var(--accent);
  white-space: nowrap;
}
.topbar-actions {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-wrap: wrap;
}
.nav-pill, .theme-btn, .btn-primary, .btn-secondary, .btn-outline, .copy-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0.35rem;
  min-height: 38px;
  padding: 0.38rem 0.75rem;
  border-radius: 8px;
  font-size: 0.84rem;
  font-weight: 600;
  cursor: pointer;
  border: 1px solid var(--border);
  background: var(--bg-subtle);
  color: var(--fg);
  white-space: nowrap;
  text-decoration: none;
}
.nav-pill:hover, .theme-btn:hover, .btn-outline:hover, .copy-btn:hover {
  border-color: var(--accent);
  text-decoration: none;
}
.nav-pill-accent, .btn-primary {
  background: var(--accent);
  color: #06101f;
  border-color: var(--accent);
}
.nav-pill-accent:hover, .btn-primary:hover {
  filter: brightness(1.08);
  color: #06101f;
  text-decoration: none;
}
.btn-secondary {
  background: var(--accent-soft);
  color: var(--accent);
  border-color: var(--accent);
}
.btn-secondary:hover {
  filter: brightness(1.08);
  text-decoration: none;
}
.btn-outline.is-active, .speed-btn.is-active {
  background: var(--accent-soft);
  color: var(--accent);
  border-color: var(--accent);
}

/* Validation Badges */
.val-badge {
  display: inline-flex;
  align-items: center;
  padding: 0.18rem 0.58rem;
  border-radius: 999px;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.02em;
  border: 1px solid transparent;
  line-height: 1.3;
}
.badge-empirical.val-badge { background: var(--ok-bg); color: var(--ok-fg); border-color: var(--ok-bd); }
.badge-holdout.val-badge { background: var(--warn-bg); color: var(--warn-fg); border-color: var(--warn-bd); }
.badge-synthetic.val-badge { background: var(--synth-bg); color: var(--synth-fg); border-color: var(--synth-bd); }
.badge-protocol.val-badge { background: var(--proto-bg); color: var(--proto-fg); border-color: var(--proto-bd); }

/* Gallery Layout */
.gallery-main {
  max-width: 92rem;
  margin: 0 auto;
  padding: 1.25rem calc(1.1rem + env(safe-area-inset-right, 0px)) calc(2.5rem + env(safe-area-inset-bottom, 0px)) calc(1.1rem + env(safe-area-inset-left, 0px));
}
.hero-card {
  background: var(--bg-elev);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 1.35rem 1.45rem;
  box-shadow: var(--shadow);
  margin-bottom: 1.35rem;
  display: grid;
  grid-template-columns: 1.25fr 1fr;
  gap: 1.25rem;
  align-items: center;
}
.hero-eyebrow {
  margin: 0 0 0.25rem;
  font-size: 0.76rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--accent);
}
.hero-card h1 {
  margin: 0 0 0.5rem;
  font-size: clamp(1.35rem, 2.5vw, 1.95rem);
  line-height: 1.2;
}
.hero-lead {
  margin: 0;
  color: var(--fg-muted);
  font-size: 0.94rem;
}
.scope-legend {
  display: flex;
  flex-direction: column;
  gap: 0.52rem;
  background: var(--bg-subtle);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 0.85rem 0.95rem;
}
.legend-item {
  display: flex;
  flex-direction: column;
  gap: 0.16rem;
  font-size: 0.79rem;
  color: var(--fg-muted);
}

/* Responsive 2x4 (desktop), 2x4 (tablet), 1x8 (mobile) Card Grid */
.tour-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 1rem;
}
@media (max-width: 1023px) and (min-width: 680px) {
  .hero-card { grid-template-columns: 1fr; }
  .tour-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 679px) {
  .hero-card { grid-template-columns: 1fr; padding: 1rem; }
  .tour-grid { grid-template-columns: 1fr; }
}

.tour-card {
  background: var(--bg-elev);
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  box-shadow: var(--shadow);
  transition: transform 0.15s ease, border-color 0.15s ease;
}
.tour-card:hover {
  transform: translateY(-2px);
  border-color: var(--accent);
}
.card-stage {
  position: relative;
  aspect-ratio: 2 / 1;
  background: var(--bg-stage);
  overflow: hidden;
}
.mini-poster, .mini-canvas {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  display: block;
}
.card-mod-pill, .card-dur-pill {
  position: absolute;
  bottom: 8px;
  padding: 0.14rem 0.48rem;
  border-radius: 6px;
  font-size: 0.72rem;
  font-weight: 700;
  background: rgba(7, 13, 25, 0.82);
  color: #e2e8f0;
  border: 1px solid rgba(148, 163, 184, 0.28);
}
.card-mod-pill { left: 8px; }
.card-dur-pill { right: 8px; font-variant-numeric: tabular-nums; }
.card-body {
  padding: 0.9rem;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  flex: 1 1 auto;
}
.card-title {
  margin: 0;
  font-size: 1.02rem;
  line-height: 1.28;
}
.card-title a { color: var(--fg); }
.card-title a:hover { color: var(--accent); }
.card-subtitle {
  margin: 0;
  font-size: 0.82rem;
  color: var(--fg-muted);
}
.card-disclosure {
  margin: 0;
  font-size: 0.74rem;
  padding: 0.35rem 0.5rem;
  border-radius: 6px;
  background: var(--synth-bg);
  color: var(--synth-fg);
  border: 1px solid var(--synth-bd);
}
.card-disclosure-holdout {
  background: var(--warn-bg);
  color: var(--warn-fg);
  border-color: var(--warn-bd);
}
.ch-pills {
  display: flex;
  flex-wrap: wrap;
  gap: 0.32rem;
  margin-top: auto;
}
.ch-pill {
  font-size: 0.72rem;
  padding: 0.16rem 0.44rem;
  border-radius: 6px;
  background: var(--bg-subtle);
  border: 1px solid var(--border);
  color: var(--fg-muted);
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.ch-pill:hover {
  border-color: var(--accent);
  color: var(--accent);
  text-decoration: none;
}
.card-actions {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.45rem;
  padding-top: 0.25rem;
}
.card-actions .btn-primary, .card-actions .btn-secondary {
  font-size: 0.78rem;
  padding: 0.42rem 0.45rem;
}

/* Per-Module Dual-Mode Cinema Player */
.player-body {
  display: flex;
  flex-direction: column;
  height: 100vh;
  height: 100dvh;
}
body.is-embed .hide-compact { display: none; }
body.is-embed .player-topbar { min-height: 2.9rem; padding-top: 0.25rem; }

.player-title-inline {
  font-size: 0.92rem;
  color: var(--fg);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 24rem;
}
.mode-switcher {
  display: inline-flex;
  background: var(--bg-subtle);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 3px;
  gap: 3px;
}
.mode-tab {
  min-height: 34px;
  padding: 0.3rem 0.75rem;
  border-radius: 7px;
  border: 0;
  background: transparent;
  color: var(--fg-muted);
  font-size: 0.82rem;
  font-weight: 700;
  cursor: pointer;
}
.mode-tab.is-active {
  background: var(--accent);
  color: #06101f;
}

.validation-banner {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.55rem;
  padding: 0.45rem calc(1rem + env(safe-area-inset-right, 0px)) 0.45rem calc(1rem + env(safe-area-inset-left, 0px));
  border-bottom: 1px solid var(--border);
  background: var(--bg-elev);
  font-size: 0.8rem;
  flex-shrink: 0;
}
.validation-banner.badge-synthetic, .validation-banner.badge-protocol {
  background: var(--synth-bg);
  border-bottom-color: var(--synth-bd);
}
.validation-banner.badge-holdout {
  background: var(--warn-bg);
  border-bottom-color: var(--warn-bd);
}
.validation-banner.badge-empirical {
  background: var(--ok-bg);
  border-bottom-color: var(--ok-bd);
}
.val-detail {
  color: var(--fg);
  font-size: 0.79rem;
}

.cinema-shell {
  flex: 1 1 auto;
  min-height: 0;
  display: flex;
  flex-direction: column;
  max-width: 96rem;
  width: 100%;
  margin: 0 auto;
  padding: 0.65rem calc(0.85rem + env(safe-area-inset-right, 0px)) calc(0.65rem + env(safe-area-inset-bottom, 0px)) calc(0.85rem + env(safe-area-inset-left, 0px));
  gap: 0.6rem;
}
.player-panel {
  flex: 1 1 auto;
  min-height: 0;
  display: flex;
  flex-direction: column;
  gap: 0.55rem;
}
.player-panel[hidden] { display: none; }

.stage-wrap, .video-stage-wrap {
  position: relative;
  flex: 1 1 auto;
  min-height: 240px;
  background: var(--bg-stage);
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
}
.tour-canvas {
  display: block;
  width: 100%;
  height: 100%;
  cursor: crosshair;
}
.tour-canvas.is-orbit-canvas {
  touch-action: none;
  cursor: grab;
}
.tour-canvas.is-orbit-canvas:active { cursor: grabbing; }
.tour-video {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: contain;
  background: #000000;
}

#canvas-wrap { position: relative; overflow: hidden; }
.stage-overlay-pill {
  position: absolute;
  top: 10px;
  right: 16px;
  left: auto;
  max-width: calc(100% - 32px);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  z-index: 3;
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 4px 10px;
  border-radius: 999px;
  background: rgba(7, 13, 25, 0.78);
  color: #cbd5e1;
  border: 1px solid rgba(148, 163, 184, 0.28);
  font-size: 11px;
  font-weight: 600;
  pointer-events: none;
}
.fallback-banner {
  position: absolute;
  top: 10px;
  left: 12px;
  right: 12px;
  z-index: 25;
  padding: 0.42rem 0.72rem;
  border-radius: 7px;
  background: rgba(185, 28, 28, 0.94);
  color: #fef2f2;
  border: 1px solid #f87171;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.76rem;
  font-weight: 700;
}
.hud-tooltip {
  position: absolute;
  z-index: 20;
  pointer-events: none;
  min-width: 165px;
  max-width: 270px;
  padding: 0.48rem 0.65rem;
  border-radius: 8px;
  background: rgba(9, 16, 30, 0.94);
  color: #f1f5f9;
  border: 1px solid #38bdf8;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.45);
  font-size: 0.76rem;
  line-height: 1.38;
  white-space: pre-line;
}
.hud-tooltip strong {
  display: block;
  color: #38bdf8;
  margin-bottom: 0.15rem;
}
.hud-tooltip-body {
  white-space: pre-line;
}

/* Interactive Control Strip */
.control-strip {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 0.65rem;
  background: var(--bg-elev);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 0.5rem 0.8rem;
  flex-shrink: 0;
}
.ctrl-primary {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.65rem;
  flex: 1 1 280px;
}
.ctrl-label {
  font-size: 0.82rem;
  white-space: nowrap;
}
.ctrl-slider-row {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  flex: 1 1 180px;
}
.ctrl-slider-row input[type="range"] {
  flex: 1 1 auto;
  min-width: 110px;
  accent-color: var(--accent);
}
.step-btn {
  min-width: 36px;
  min-height: 36px;
  border-radius: 7px;
  border: 1px solid var(--border);
  background: var(--bg-subtle);
  color: var(--fg);
  font-size: 1rem;
  font-weight: 700;
  cursor: pointer;
}
.ctrl-presets {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.35rem;
}
.preset-chip {
  min-height: 32px;
  padding: 0.22rem 0.55rem;
  border-radius: 6px;
  border: 1px solid var(--border);
  background: var(--bg-subtle);
  color: var(--fg);
  font-size: 0.76rem;
  font-weight: 600;
  cursor: pointer;
}
.preset-chip:hover { border-color: var(--accent); color: var(--accent); }
.mcp-inspector-wrap { position: relative; }
.mcp-packet-drawer {
  margin: 0.45rem 0 0;
  padding: 0.6rem;
  max-height: 150px;
  overflow: auto;
  border-radius: 8px;
  background: #070d19;
  color: #7dd3fc;
  border: 1px solid var(--accent);
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.72rem;
}

/* Transport Bar */
.transport-bar, .video-controls-bar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 0.65rem;
  background: var(--bg-elev);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 0.55rem 0.8rem;
  flex-shrink: 0;
}
.transport-left, .transport-right, .video-jumps, .video-downloads {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.45rem;
}
.play-btn {
  min-height: 44px;
  min-width: 90px;
  padding: 0.4rem 0.9rem;
  border-radius: 8px;
  border: 0;
  background: var(--accent);
  color: #06101f;
  font-weight: 800;
  font-size: 0.86rem;
  cursor: pointer;
}
.time-readout {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.8rem;
  color: var(--fg-muted);
  font-variant-numeric: tabular-nums;
}
.timeline-wrap {
  flex: 1 1 260px;
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
}
.timeline-range {
  width: 100%;
  accent-color: var(--accent);
}
.ch-markers {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0.3rem;
}
.ch-marker, .vid-jump-btn {
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  min-height: 32px;
  padding: 0.2rem 0.45rem;
  border-radius: 6px;
  border: 1px solid var(--border);
  background: var(--bg-subtle);
  color: var(--fg-muted);
  font-size: 0.74rem;
  font-weight: 600;
  cursor: pointer;
  overflow: hidden;
}
.ch-marker.is-active, .vid-jump-btn.is-active {
  background: var(--accent-soft);
  color: var(--accent);
  border-color: var(--accent);
}
.ch-num {
  font-weight: 800;
  flex-shrink: 0;
}
.ch-lbl {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.speed-group {
  display: inline-flex;
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
}
.speed-btn {
  min-height: 36px;
  padding: 0.25rem 0.5rem;
  border: 0;
  border-right: 1px solid var(--border);
  background: var(--bg-subtle);
  color: var(--fg);
  font-size: 0.76rem;
  font-weight: 700;
  cursor: pointer;
}
.speed-btn:last-child { border-right: 0; }
.export-status {
  font-size: 0.75rem;
  color: var(--accent);
}

/* Bottom Drawer */
.bottom-drawer {
  background: var(--bg-elev);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 0.65rem 0.85rem;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  flex-shrink: 0;
}
.drawer-narrative { flex: 1 1 340px; min-width: 0; }
.drawer-title-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.45rem;
  margin-bottom: 0.2rem;
}
.drawer-ch-badge {
  font-size: 0.72rem;
  font-weight: 700;
  padding: 0.12rem 0.48rem;
  border-radius: 999px;
  background: var(--accent-soft);
  color: var(--accent);
}
.drawer-ch-title {
  margin: 0;
  font-size: 0.98rem;
}
.drawer-metric {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.74rem;
  padding: 0.12rem 0.48rem;
  border-radius: 6px;
  background: var(--bg-subtle);
  border: 1px solid var(--border);
  color: var(--fg-muted);
}
.drawer-caption {
  margin: 0;
  font-size: 0.84rem;
  color: var(--fg-muted);
}
.drawer-actions {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 0.45rem;
  flex: 0 1 auto;
}
.cli-box {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  background: var(--bg-subtle);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.22rem 0.35rem 0.22rem 0.6rem;
  max-width: 100%;
}
.cli-box code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.76rem;
  overflow-x: auto;
  white-space: nowrap;
  max-width: 28rem;
  scrollbar-width: thin;
  scrollbar-color: rgba(148, 163, 184, 0.3) transparent;
}
.cli-box code::-webkit-scrollbar { height: 4px; }
.cli-box code::-webkit-scrollbar-track { background: transparent; }
.cli-box code::-webkit-scrollbar-thumb { background: rgba(148, 163, 184, 0.3); border-radius: 2px; }
.drawer-links {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.45rem;
}
.adj-wrap { display: inline-flex; gap: 0.3rem; }
.nav-adj {
  display: inline-flex;
  align-items: center;
  min-height: 36px;
  padding: 0.25rem 0.58rem;
  border-radius: 7px;
  border: 1px solid var(--border);
  background: var(--bg-subtle);
  color: var(--fg);
  font-size: 0.78rem;
  font-weight: 700;
}
.nav-adj.disabled { opacity: 0.45; pointer-events: none; }

@media (max-width: 1023px) {
  .player-body { height: auto; min-height: 100vh; min-height: 100dvh; }
  .stage-wrap, .video-stage-wrap { min-height: 300px; aspect-ratio: 16 / 9; }
  .drawer-actions { align-items: stretch; width: 100%; }
}

@media (max-width: 679px) {
  .player-title-inline { display: none; }
  .mode-tab { font-size: 0.76rem; padding: 0.28rem 0.55rem; }
  .ch-markers { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .cli-box code { max-width: 12rem; }
  .stage-wrap, .video-stage-wrap { min-height: 220px; }
}
"""


TOURS_PLAYER_JS = r"""(function () {
  "use strict";

  var root = document.documentElement;
  var savedTheme = null;
  try { savedTheme = localStorage.getItem("nmr_manual_theme"); } catch (_) {}
  if (savedTheme === "dark" || savedTheme === "light") {
    root.setAttribute("data-theme", savedTheme);
  } else if (window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches) {
    root.setAttribute("data-theme", "light");
  } else {
    root.setAttribute("data-theme", "dark");
  }

  function syncThemeBtn() {
    var btn = document.getElementById("theme-toggle");
    if (!btn) return;
    var cur = root.getAttribute("data-theme") === "dark" ? "dark" : "light";
    btn.textContent = cur === "dark" ? "Light" : "Dark";
  }

  function isDark() {
    return root.getAttribute("data-theme") !== "light";
  }

  function palette() {
    if (isDark()) {
      return {
        bg: "#070d19",
        panel: "#0f172a",
        grid: "rgba(148, 163, 184, 0.14)",
        axis: "#64748b",
        text: "#e2e8f0",
        muted: "#94a3b8",
        cyan: "#38bdf8",
        teal: "#2dd4bf",
        green: "#34d399",
        amber: "#fbbf24",
        coral: "#fb7185",
        violet: "#a78bfa",
        border: "rgba(148, 163, 184, 0.28)"
      };
    }
    return {
      bg: "#0b1324",
      panel: "#131e33",
      grid: "rgba(148, 163, 184, 0.16)",
      axis: "#94a3b8",
      text: "#f1f5f9",
      muted: "#cbd5e1",
      cyan: "#38bdf8",
      teal: "#2dd4bf",
      green: "#34d399",
      amber: "#fbbf24",
      coral: "#fb7185",
      violet: "#c4b5fd",
      border: "rgba(148, 163, 184, 0.28)"
    };
  }

  function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
  }

  function formatTime(sec) {
    var s = Math.max(0, sec);
    var m = Math.floor(s / 60);
    var rem = (s - m * 60).toFixed(1);
    if (rem.length < 4) rem = "0" + rem;
    return m + ":" + rem;
  }

  function activeChapterIndex(bundle, tSec) {
    var chs = bundle.chapters || [];
    for (var i = 0; i < chs.length; i++) {
      if (tSec >= chs[i].start_sec && tSec < chs[i].end_sec) return i;
    }
    return Math.max(0, chs.length - 1);
  }

  function drawAxes(ctx, x0, y0, pw, ph, xLbl, yLbl, pal) {
    ctx.save();
    ctx.strokeStyle = pal.grid;
    ctx.lineWidth = 1;
    for (var gx = 1; gx < 5; gx++) {
      var xx = x0 + (pw * gx) / 5;
      ctx.beginPath();
      ctx.moveTo(xx, y0);
      ctx.lineTo(xx, y0 + ph);
      ctx.stroke();
    }
    for (var gy = 1; gy < 4; gy++) {
      var yy = y0 + (ph * gy) / 4;
      ctx.beginPath();
      ctx.moveTo(x0, yy);
      ctx.lineTo(x0 + pw, yy);
      ctx.stroke();
    }
    ctx.strokeStyle = pal.axis;
    ctx.lineWidth = 1.4;
    ctx.strokeRect(x0, y0, pw, ph);
    ctx.fillStyle = pal.muted;
    ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
    if (xLbl) {
      ctx.textAlign = "right";
      ctx.fillText(xLbl, x0 + pw - 6, y0 + ph + 16);
    }
    if (yLbl) {
      ctx.textAlign = "left";
      ctx.fillText(yLbl, x0 + 6, y0 + 14);
    }
    ctx.restore();
  }

  function drawRoundRect(ctx, x, y, w, h, r) {
    var rr = Math.min(r, w * 0.5, h * 0.5);
    ctx.beginPath();
    ctx.moveTo(x + rr, y);
    ctx.arcTo(x + w, y, x + w, y + h, rr);
    ctx.arcTo(x + w, y + h, x, y + h, rr);
    ctx.arcTo(x, y + h, x, y, rr);
    ctx.arcTo(x, y, x + w, y, rr);
    ctx.closePath();
  }

  function drawCardTitle(ctx, x, y, title, sub, pal) {
    ctx.save();
    ctx.fillStyle = pal.text;
    ctx.font = "600 13px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(title, x, y);
    if (sub) {
      ctx.fillStyle = pal.muted;
      ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
      ctx.fillText(sub, x, y + 16);
    }
    ctx.restore();
  }

  function getByPath(obj, path) {
    if (!obj || typeof obj !== "object") return undefined;
    var parts = path.split(".");
    var cur = obj;
    for (var i = 0; i < parts.length; i++) {
      if (cur === null || cur === undefined || typeof cur !== "object") return undefined;
      cur = cur[parts[i]];
    }
    return cur;
  }

  function renderModuleScene(ctx, w, h, modIdx, payload, tSec, ctrlVal, state, hotspots) {
    var pal = palette();
    ctx.fillStyle = pal.bg;
    ctx.fillRect(0, 0, w, h);

    var fallbackReasons = [];
    var plottedKeys = [];
    function markMissing(path) {
      if (fallbackReasons.indexOf(path) === -1) fallbackReasons.push(path);
    }
    function markPlotted(path) {
      if (plottedKeys.indexOf(path) === -1) plottedKeys.push(path);
    }
    function reqArr(path, minLen) {
      var v = getByPath(payload, path);
      var need = minLen === undefined ? 1 : minLen;
      if (Array.isArray(v) && v.length >= need) {
        markPlotted(path);
        return v;
      }
      markMissing(path);
      return null;
    }
    function reqNum(path) {
      var v = getByPath(payload, path);
      if (typeof v === "number" && isFinite(v)) {
        markPlotted(path);
        return v;
      }
      markMissing(path);
      return null;
    }
    function reqStr(path) {
      var v = getByPath(payload, path);
      if (typeof v === "string" && v.length > 0) {
        markPlotted(path);
        return v;
      }
      markMissing(path);
      return null;
    }
    function reqBool(path) {
      var v = getByPath(payload, path);
      if (typeof v === "boolean") {
        markPlotted(path);
        return v;
      }
      markMissing(path);
      return null;
    }

    var chIdx = Math.min(3, Math.max(0, Math.floor(tSec / 4.0)));
    var localT = (tSec - chIdx * 4.0) / 4.0;
    var padX = Math.max(24, w * 0.06);
    var padY = Math.max(46, h * 0.14);
    var pw = Math.max(120, w - padX * 2);
    var ph = Math.max(100, h - padY - 34);

    function addSpot(x, y, r, title, lines, val) {
      if (hotspots) {
        hotspots.push({ x: x, y: y, r: r, title: title, lines: lines, val: val });
      }
    }

    function drawFallbackPlaceholder(titleText) {
      drawCardTitle(
        ctx,
        padX,
        padY - 12,
        titleText,
        "ILLUSTRATIVE FALLBACK (not engine output) — missing payload keys: " + fallbackReasons.join(", "),
        pal
      );
      drawAxes(ctx, padX, padY + 8, pw, ph - 12, "Missing payload series", "Warning", pal);
      ctx.save();
      ctx.fillStyle = "rgba(159, 18, 57, 0.24)";
      ctx.strokeStyle = pal.coral;
      ctx.lineWidth = 1.6;
      drawRoundRect(ctx, padX + 16, padY + ph * 0.30, pw - 32, ph * 0.36, 8);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = pal.text;
      ctx.font = "700 12px ui-monospace, SFMono-Regular, Menlo, monospace";
      ctx.textAlign = "center";
      ctx.fillText(
        "ILLUSTRATIVE FALLBACK (not engine output): missing payload." + fallbackReasons.join(", "),
        padX + pw * 0.5,
        padY + ph * 0.50
      );
      ctx.restore();
    }

    function renderSceneBody() {
      if (modIdx === 1) {
        if (chIdx === 0) {
          var ppName = reqStr("pulprog_name");
          var aqseq = reqStr("aqseq");
          var f1Quad = reqStr("f1_quad");
          var f2Quad = reqStr("f2_quad");
          var trSum = reqStr("transfer_summary");
          if (!ppName || !aqseq || !f1Quad || !f2Quad || !trSum) {
            drawFallbackPlaceholder("Module 01 Ch 1: Pulseprogram AST");
            return;
          }
          drawCardTitle(
            ctx,
            padX,
            padY - 12,
            "Sandboxed Bruker Pulseprogram AST · " + ppName,
            "AQSEQ=" + aqseq + " · F1=" + f1Quad + " · F2=" + f2Quad + " · " + trSum,
            pal
          );
          var rows = [
            ["PULSEPROG  " + ppName, "Parsed AST header & transfer graph: " + trSum],
            ["AQSEQ      " + aqseq, "Indirect loop nesting order from acqus + mc macro"],
            ["F1 QUAD    " + f1Quad, "Rance-Kay igrad EA / FnMODE quadrature deduction"],
            ["F2 QUAD    " + f2Quad, "Second indirect dimension phase-sensitive quadrature"]
          ];
          for (var i = 0; i < rows.length; i++) {
            var ry = padY + 14 + i * (ph * 0.22);
            var active = i === Math.floor(localT * 4) % 4;
            ctx.fillStyle = active ? "rgba(56, 189, 248, 0.16)" : pal.panel;
            ctx.strokeStyle = active ? pal.cyan : pal.grid;
            ctx.lineWidth = active ? 1.8 : 1;
            drawRoundRect(ctx, padX, ry, pw, ph * 0.18, 8);
            ctx.fill();
            ctx.stroke();
            ctx.fillStyle = pal.cyan;
            ctx.font = "600 12px ui-monospace, monospace";
            ctx.fillText(rows[i][0], padX + 12, ry + ph * 0.11);
            ctx.fillStyle = pal.muted;
            ctx.font = "12px -apple-system, sans-serif";
            ctx.fillText(rows[i][1], padX + Math.min(240, pw * 0.44), ry + ph * 0.11);
            addSpot(padX + pw * 0.5, ry + ph * 0.09, 24, rows[i][0], [rows[i][1], "Engine AST payload (" + ppName + ")"]);
          }
        } else if (chIdx === 1) {
          var fidSrc = reqArr("fid_real", 8);
          var sineWin = reqArr("sine_window", 8);
          var fftPhased = reqArr("fft_phased", 8);
          var fftUnphased = reqArr("fft_unphased", 8);
          var p0Deg = reqNum("p0_deg");
          var p1Deg = reqNum("p1_deg");
          if (!fidSrc || !sineWin || !fftPhased || !fftUnphased || p0Deg === null || p1Deg === null) {
            drawFallbackPlaceholder("Module 01 Ch 2: FID Apodization & FFT");
            return;
          }
          drawCardTitle(
            ctx,
            padX,
            padY - 12,
            "1D Raw FID Decay -> Shifted Sine-Bell -> Auto-Phased FFT",
            "Chen (2002) PH0=" + p0Deg.toFixed(1) + " deg · PH1=" + p1Deg.toFixed(1) + " deg · N_TD=" + fidSrc.length + " -> N_FFT=" + fftPhased.length,
            pal
          );
          drawAxes(ctx, padX, padY + 8, pw, ph - 12, "Time / Normalized Frequency", "Engine Signal", pal);
          var maxFid = 1e-9;
          for (var i = 0; i < fidSrc.length; i++) maxFid = Math.max(maxFid, Math.abs(fidSrc[i]));
          ctx.strokeStyle = pal.amber;
          ctx.lineWidth = 1.2;
          ctx.setLineDash([3, 3]);
          ctx.beginPath();
          for (var i = 0; i < sineWin.length; i++) {
            var wx = padX + (i / (sineWin.length - 1)) * pw;
            var wy = padY + ph * 0.34 - sineWin[i] * ph * 0.22;
            if (i === 0) ctx.moveTo(wx, wy);
            else ctx.lineTo(wx, wy);
          }
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.strokeStyle = pal.teal;
          ctx.lineWidth = 2;
          ctx.beginPath();
          for (var i = 0; i < fidSrc.length; i++) {
            var x = padX + (i / (fidSrc.length - 1)) * pw;
            var y = padY + ph * 0.34 - (fidSrc[i] / maxFid) * ph * 0.22;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          }
          ctx.stroke();

          var maxSpec = 1e-9;
          for (var i = 0; i < fftPhased.length; i++) maxSpec = Math.max(maxSpec, Math.abs(fftPhased[i]));
          for (var i = 0; i < fftUnphased.length; i++) maxSpec = Math.max(maxSpec, Math.abs(fftUnphased[i]));
          ctx.strokeStyle = pal.coral;
          ctx.lineWidth = 1.4;
          ctx.setLineDash([4, 3]);
          ctx.beginPath();
          for (var i = 0; i < fftUnphased.length; i++) {
            var ux = padX + (i / (fftUnphased.length - 1)) * pw;
            var uy = padY + ph * 0.88 - (fftUnphased[i] / maxSpec) * ph * 0.30;
            if (i === 0) ctx.moveTo(ux, uy);
            else ctx.lineTo(ux, uy);
          }
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.strokeStyle = pal.cyan;
          ctx.lineWidth = 2.2;
          ctx.beginPath();
          for (var i = 0; i < fftPhased.length; i++) {
            var sx = padX + (i / (fftPhased.length - 1)) * pw;
            var sy = padY + ph * 0.88 - (fftPhased[i] / maxSpec) * ph * 0.30;
            if (i === 0) ctx.moveTo(sx, sy);
            else ctx.lineTo(sx, sy);
          }
          ctx.stroke();
          for (var i = 1; i < fftPhased.length - 1; i++) {
            if (fftPhased[i] > fftPhased[i - 1] && fftPhased[i] > fftPhased[i + 1] && fftPhased[i] > maxSpec * 0.20) {
              var px = padX + (i / (fftPhased.length - 1)) * pw;
              var py = padY + ph * 0.88 - (fftPhased[i] / maxSpec) * ph * 0.30;
              var fNorm = i / (fftPhased.length - 1);
              ctx.fillStyle = pal.amber;
              ctx.beginPath();
              ctx.arc(px, py, 4, 0, Math.PI * 2);
              ctx.fill();
              addSpot(px, py, 14, "FFT Bin #" + i + " (f=" + fNorm.toFixed(2) + ")", ["Phased amp: " + fftPhased[i].toFixed(2), "PH0=" + p0Deg.toFixed(1) + " deg, PH1=" + p1Deg.toFixed(1) + " deg"]);
            }
          }
        } else if (chIdx === 2) {
          var mask = reqArr("nus_mask", 8);
          var dirty = reqArr("nus_dirty_spec", 8);
          if (!mask || !dirty) {
            drawFallbackPlaceholder("Module 01 Ch 3: NUS Schedule & Zero-Fill Spectrum");
            return;
          }
          var nAcq = 0;
          for (var m = 0; m < mask.length; m++) { if (mask[m]) nAcq++; }
          var pctAcq = ((100.0 * nAcq) / mask.length).toFixed(0);
          drawCardTitle(
            ctx,
            padX,
            padY - 12,
            pctAcq + "% Poisson-Gap NUS Schedule & Zero-Fill PSF Artifacts",
            nAcq + " / " + mask.length + " indirect increments acquired · zero-filled DFT spectrum below",
            pal
          );
          drawAxes(ctx, padX, padY + 8, pw, ph - 12, "Indirect Increment / Frequency Bin", "NUS Mask & Dirty DFT", pal);
          for (var i = 0; i < mask.length; i++) {
            var bx = padX + (i / mask.length) * pw;
            if (mask[i]) {
              ctx.fillStyle = pal.cyan;
              ctx.fillRect(bx, padY + 18, Math.max(2, pw / (mask.length * 1.25)), ph * 0.25);
              addSpot(bx + 2, padY + 18 + ph * 0.12, 10, "NUS Increment #" + i, ["Acquired (" + nAcq + "/" + mask.length + " Poisson-gap)"]);
            } else {
              ctx.fillStyle = "rgba(148,163,184,0.16)";
              ctx.fillRect(bx, padY + 18, Math.max(2, pw / (mask.length * 1.25)), ph * 0.08);
            }
          }
          var maxDirty = 1e-9;
          for (var i = 0; i < dirty.length; i++) maxDirty = Math.max(maxDirty, Math.abs(dirty[i]));
          ctx.strokeStyle = pal.coral;
          ctx.lineWidth = 2;
          ctx.beginPath();
          for (var i = 0; i < dirty.length; i++) {
            var sx = padX + (i / (dirty.length - 1)) * pw;
            var sy = padY + ph * 0.90 - (dirty[i] / maxDirty) * ph * 0.44;
            if (i === 0) ctx.moveTo(sx, sy);
            else ctx.lineTo(sx, sy);
          }
          ctx.stroke();
        } else {
          var fistaIters = reqArr("nus_fista_iters", 2);
          var resArr = reqArr("nus_residual_l1", 2);
          if (!fistaIters || !resArr) {
            drawFallbackPlaceholder("Module 01 Ch 4: IST/FISTA Reconstruction");
            return;
          }
          var iter = Math.round(clamp(ctrlVal, 0, fistaIters.length - 1));
          var specIter = fistaIters[iter];
          if (!Array.isArray(specIter) || specIter.length < 4) {
            markMissing("nus_fista_iters[" + iter + "]");
            drawFallbackPlaceholder("Module 01 Ch 4: IST/FISTA Iteration Slice");
            return;
          }
          var l1Cur = Number(resArr[Math.min(iter, resArr.length - 1)]);
          var l1Init = Number(resArr[0]);
          var resText = "Normalized L1 error: " + l1Cur.toFixed(3) + " (iter 0: " + l1Init.toFixed(3) + ")";
          drawCardTitle(
            ctx,
            padX,
            padY - 12,
            "Hypercomplex IST/FISTA Threshold Shrinkage (iter = " + iter + " / " + (fistaIters.length - 1) + ")",
            resText,
            pal
          );
          drawAxes(ctx, padX, padY + 8, pw, ph - 12, "Frequency Bin (0.." + (specIter.length - 1) + ")", "Reconstructed Amplitude", pal);
          var maxFista = 1e-9;
          for (var i = 0; i < specIter.length; i++) maxFista = Math.max(maxFista, Math.abs(specIter[i]));
          ctx.strokeStyle = pal.green;
          ctx.lineWidth = 2.4;
          ctx.beginPath();
          for (var i = 0; i < specIter.length; i++) {
            var sx = padX + (i / (specIter.length - 1)) * pw;
            var sy = padY + ph * 0.88 - (specIter[i] / maxFista) * ph * 0.72;
            if (i === 0) ctx.moveTo(sx, sy);
            else ctx.lineTo(sx, sy);
          }
          ctx.stroke();
          for (var i = 1; i < specIter.length - 1; i++) {
            if (specIter[i] > specIter[i - 1] && specIter[i] > specIter[i + 1] && specIter[i] > maxFista * 0.18) {
              var px = padX + (i / (specIter.length - 1)) * pw;
              var py = padY + ph * 0.88 - (specIter[i] / maxFista) * ph * 0.72;
              ctx.fillStyle = pal.amber;
              ctx.beginPath();
              ctx.arc(px, py, 5, 0, Math.PI * 2);
              ctx.fill();
              addSpot(px, py, 16, "FISTA Peak bin #" + i, ["Amplitude: " + specIter[i].toFixed(2), resText]);
            }
          }
        }
        return;
      }

      if (modIdx === 2) {
        var focusRes = Math.round(clamp(ctrlVal, 1, 10));
        if (chIdx === 0) {
          var posSegs = reqArr("hsqc_pos_contours", 1);
          var negSegs = reqArr("hsqc_neg_contours", 1);
          var roots = reqArr("hsqc_roots", 5);
          if (!posSegs || !negSegs || !roots) {
            drawFallbackPlaceholder("Module 02 Ch 1: 2D HSQC Contours & Roots");
            return;
          }
          drawCardTitle(
            ctx,
            padX,
            padY - 12,
            "2D 1H-15N HSQC Contours & Picked Backbone Roots (" + roots.length + " roots)",
            "Cyan: +1 backbone N-H contours · Coral: -1 folded Arg Ne-He contours",
            pal
          );
          drawAxes(ctx, padX, padY + 8, pw, ph - 12, "1H (ppm, 10.2 -> 6.2)", "15N (ppm, 104 -> 132)", pal);
          ctx.strokeStyle = "rgba(56, 189, 248, 0.34)";
          ctx.lineWidth = 1.0;
          for (var ps = 0; ps < posSegs.length; ps++) {
            var pseg = posSegs[ps];
            if (!Array.isArray(pseg) || pseg.length < 2) continue;
            ctx.beginPath();
            for (var k = 0; k < pseg.length; k++) {
              var px0 = padX + clamp((10.2 - pseg[k][0]) / 4.0, 0.02, 0.98) * pw;
              var py0 = padY + 12 + clamp((pseg[k][1] - 104.0) / 28.0, 0.02, 0.98) * (ph - 24);
              if (k === 0) ctx.moveTo(px0, py0);
              else ctx.lineTo(px0, py0);
            }
            ctx.stroke();
          }
          ctx.strokeStyle = pal.coral;
          ctx.lineWidth = 1.4;
          for (var ns = 0; ns < negSegs.length; ns++) {
            var seg = negSegs[ns];
            if (!Array.isArray(seg) || seg.length < 2) continue;
            ctx.beginPath();
            for (var k = 0; k < seg.length; k++) {
              var nx = padX + clamp((10.2 - seg[k][0]) / 4.0, 0.02, 0.98) * pw;
              var ny = padY + 12 + clamp((seg[k][1] - 104.0) / 28.0, 0.02, 0.98) * (ph - 24);
              if (k === 0) ctx.moveTo(nx, ny);
              else ctx.lineTo(nx, ny);
            }
            ctx.stroke();
          }
          if (negSegs[0] && negSegs[0][0]) {
            var s0 = negSegs[0][0];
            addSpot(
              padX + clamp((10.2 - s0[0]) / 4.0, 0.02, 0.98) * pw,
              padY + 12 + clamp((s0[1] - 104.0) / 28.0, 0.02, 0.98) * (ph - 24),
              14,
              "Aliased Arg Ne-He (-1)",
              ["1H: " + s0[0].toFixed(2) + " ppm", "15N: " + s0[1].toFixed(1) + " ppm (folded negative)"]
            );
          }
          for (var i = 0; i < roots.length; i++) {
            var hPpm = Number(roots[i].h);
            var nPpm = Number(roots[i].n);
            var lbl = String(roots[i].label || ("Res" + (i + 1)));
            var seqNum = parseInt(lbl.replace(/\D+/g, ""), 10) || (i + 1);
            var cx = padX + clamp((10.2 - hPpm) / 4.0, 0.03, 0.97) * pw;
            var cy = padY + 12 + clamp((nPpm - 104.0) / 28.0, 0.04, 0.96) * (ph - 24);
            var isSel = seqNum === focusRes;
            ctx.strokeStyle = isSel ? pal.amber : pal.cyan;
            ctx.lineWidth = isSel ? 2.2 : 1.3;
            ctx.beginPath();
            ctx.ellipse(cx, cy, 7.5, 4.6, 0, 0, Math.PI * 2);
            ctx.stroke();
            addSpot(cx, cy, 12, lbl + " HSQC Root", ["1H: " + hPpm.toFixed(2) + " ppm", "15N: " + nPpm.toFixed(1) + " ppm"], Math.min(10, Math.max(1, seqNum)));
          }
        } else if (chIdx === 1) {
          var allStrips = reqArr("strips", 10);
          var pulseRegime = reqStr("hncacb_pulse_regime");
          var walkStart = reqStr("strip_walk.start_label");
          var walkEnd = reqStr("strip_walk.end_label");
          if (!allStrips || !pulseRegime || !walkStart || !walkEnd) {
            drawFallbackPlaceholder("Module 02 Ch 2: 3D HNCACB Sequential Strip Walk");
            return;
          }
          var rawStrips = allStrips.slice(3, 10);
          var focLabel = allStrips[focusRes - 1] ? allStrips[focusRes - 1].label : ("Res" + focusRes);
          drawCardTitle(
            ctx,
            padX,
            padY - 14,
            "3D HNCACB Sequential Strip Walk (" + walkStart + " -> " + walkEnd + " · Focus " + focLabel + ")",
            "Non-CT HNCACB (2*d28=7.2ms): Cyan (+1) CA [incl. Gly10 CA +1, CB null; CT 2T=28ms inverts Gly CH2] · Coral (-1) CB",
            pal
          );
          var stripLeft = padX + 38;
          var stripRight = padX + pw - 4;
          var stripTop = padY + 26;
          var stripBot = padY + ph - 6;
          var cols = rawStrips.length;
          var gap = Math.max(6, pw * 0.014);
          var stripW = (stripRight - stripLeft - gap * (cols - 1)) / cols;
          function mapCy(cPpm) {
            return stripTop + ((cPpm - 14.0) / (76.0 - 14.0)) * (stripBot - stripTop);
          }

          ctx.save();
          ctx.fillStyle = pal.cyan;
          ctx.font = "600 10.5px ui-monospace, SFMono-Regular, Menlo, monospace";
          ctx.textAlign = "left";
          ctx.fillText("13C ppm", padX, stripTop - 8);
          var cTicks = [20, 30, 40, 50, 60, 70];
          for (var ti = 0; ti < cTicks.length; ti++) {
            var ty = mapCy(cTicks[ti]);
            ctx.strokeStyle = pal.grid;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(stripLeft - 4, ty);
            ctx.lineTo(stripRight, ty);
            ctx.stroke();
            ctx.fillStyle = pal.muted;
            ctx.textAlign = "right";
            ctx.fillText(String(cTicks[ti]), stripLeft - 7, ty + 3.5);
          }
          ctx.restore();

          var centersX = [];
          for (var s = 0; s < cols; s++) {
            var sx0 = stripLeft + s * (stripW + gap);
            var scx = sx0 + stripW * 0.5;
            centersX.push(scx);
            var st = rawStrips[s];
            var isFoc = st.seq_id === focusRes;

            ctx.fillStyle = isFoc ? "rgba(56, 189, 248, 0.10)" : pal.panel;
            ctx.strokeStyle = isFoc ? pal.amber : pal.border;
            ctx.lineWidth = isFoc ? 1.8 : 1.1;
            drawRoundRect(ctx, sx0, stripTop, stripW, stripBot - stripTop, 7);
            ctx.fill();
            ctx.stroke();

            ctx.save();
            ctx.textAlign = "center";
            ctx.fillStyle = isFoc ? pal.amber : pal.text;
            ctx.font = "700 11.5px ui-monospace, SFMono-Regular, Menlo, monospace";
            ctx.fillText(String(st.label), scx, stripTop - 6);
            if (typeof st.h_ppm === "number" && typeof st.n_ppm === "number") {
              ctx.fillStyle = pal.muted;
              ctx.font = "9.5px ui-monospace, SFMono-Regular, Menlo, monospace";
              ctx.fillText(st.h_ppm.toFixed(2) + "/" + st.n_ppm.toFixed(0), scx, stripTop + 11);
            }
            ctx.restore();

            if (Array.isArray(st.pos_pts) && st.pos_pts.length > 4) {
              ctx.strokeStyle = isFoc ? "rgba(56, 189, 248, 0.55)" : "rgba(56, 189, 248, 0.30)";
              ctx.lineWidth = 1.2;
              ctx.beginPath();
              for (var p = 0; p < st.pos_pts.length; p++) {
                var px = scx + st.pos_pts[p][1] * stripW * 0.36;
                var py = mapCy(st.pos_pts[p][0]);
                if (p === 0) ctx.moveTo(px, py);
                else ctx.lineTo(px, py);
              }
              ctx.stroke();
            }
            if (Array.isArray(st.neg_pts) && st.neg_pts.length > 4) {
              ctx.strokeStyle = isFoc ? "rgba(251, 113, 133, 0.55)" : "rgba(251, 113, 133, 0.30)";
              ctx.lineWidth = 1.2;
              ctx.beginPath();
              for (var p = 0; p < st.neg_pts.length; p++) {
                var nx = scx - st.neg_pts[p][1] * stripW * 0.36;
                var ny = mapCy(st.neg_pts[p][0]);
                if (p === 0) ctx.moveTo(nx, ny);
                else ctx.lineTo(nx, ny);
              }
              ctx.stroke();
            }
          }

          ctx.save();
          ctx.setLineDash([5, 3]);
          ctx.lineWidth = 1.4;
          for (var s = 1; s < cols; s++) {
            var prevSt = rawStrips[s - 1];
            var curSt = rawStrips[s];
            if (typeof prevSt.ca_i === "number" && typeof curSt.ca_im1 === "number") {
              ctx.strokeStyle = "rgba(56, 189, 248, 0.72)";
              ctx.beginPath();
              ctx.moveTo(centersX[s - 1], mapCy(prevSt.ca_i));
              ctx.lineTo(centersX[s], mapCy(curSt.ca_im1));
              ctx.stroke();
            }
            if (typeof prevSt.cb_i === "number" && typeof curSt.cb_im1 === "number") {
              ctx.strokeStyle = "rgba(251, 113, 133, 0.72)";
              ctx.beginPath();
              ctx.moveTo(centersX[s - 1], mapCy(prevSt.cb_i));
              ctx.lineTo(centersX[s], mapCy(curSt.cb_im1));
              ctx.stroke();
            }
          }
          ctx.restore();

          var dispLabels = ["Phe4", "Val5", "Lys6", "Thr7", "Leu8", "Thr9", "Gly10"];
          var prevLabels = ["Ile3", "Phe4", "Val5", "Lys6", "Thr7", "Leu8", "Thr9"];
          for (var s = 0; s < cols; s++) {
            var scx = centersX[s];
            var st = rawStrips[s];
            var dLbl = dispLabels[s] || String(st.label);
            var pLbl = prevLabels[s] || ("Res" + (st.seq_id - 1));
            var signColor = function (sgn, defSign) {
              var eff = (typeof sgn === "number") ? sgn : defSign;
              return eff >= 0 ? pal.cyan : pal.coral;
            };
            var signLabel = function (sgn, defSign) {
              var eff = (typeof sgn === "number") ? sgn : defSign;
              return eff >= 0 ? "+1" : "-1";
            };

            if (typeof st.ca_im1 === "number") {
              var yCaPrev = mapCy(st.ca_im1);
              var colCaPrev = signColor(st.ca_im1_sign, 1);
              ctx.save();
              ctx.fillStyle = pal.panel;
              ctx.beginPath();
              ctx.ellipse(scx, yCaPrev, 8.5, 4.5, 0, 0, Math.PI * 2);
              ctx.fill();
              ctx.strokeStyle = colCaPrev;
              ctx.lineWidth = 1.6;
              ctx.setLineDash([3, 2]);
              ctx.stroke();
              ctx.restore();
              addSpot(scx, yCaPrev, 11, dLbl + " CA(i-1) [" + pLbl + "]: " + st.ca_im1.toFixed(1) + " ppm (" + signLabel(st.ca_im1_sign, 1) + ", seq)", ["Phase: " + signLabel(st.ca_im1_sign, 1) + " (hollow ring)", "Sequential link from " + pLbl + " CA(i)"], st.seq_id);
            }

            if (typeof st.cb_im1 === "number") {
              var yCbPrev = mapCy(st.cb_im1);
              var colCbPrev = signColor(st.cb_im1_sign, -1);
              ctx.save();
              ctx.fillStyle = pal.panel;
              ctx.beginPath();
              ctx.ellipse(scx, yCbPrev, 8.5, 4.5, 0, 0, Math.PI * 2);
              ctx.fill();
              ctx.strokeStyle = colCbPrev;
              ctx.lineWidth = 1.6;
              ctx.setLineDash([3, 2]);
              ctx.stroke();
              ctx.restore();
              addSpot(scx, yCbPrev, 11, dLbl + " CB(i-1) [" + pLbl + "]: " + st.cb_im1.toFixed(1) + " ppm (" + signLabel(st.cb_im1_sign, -1) + ", seq)", ["Phase: " + signLabel(st.cb_im1_sign, -1) + " (hollow ring)", "Sequential link from " + pLbl + " CB(i)"], st.seq_id);
            }

            if (typeof st.ca_i === "number") {
              var yCa = mapCy(st.ca_i);
              var colCa = signColor(st.ca_i_sign, 1);
              var sgnCa = signLabel(st.ca_i_sign, 1);
              ctx.strokeStyle = colCa;
              ctx.lineWidth = 1.8;
              ctx.beginPath();
              ctx.ellipse(scx, yCa, 11, 5.5, 0, 0, Math.PI * 2);
              ctx.stroke();
              ctx.beginPath();
              ctx.ellipse(scx, yCa, 6, 3.0, 0, 0, Math.PI * 2);
              ctx.stroke();
              ctx.fillStyle = colCa;
              ctx.beginPath();
              ctx.arc(scx, yCa, 2.8, 0, Math.PI * 2);
              ctx.fill();
              var glyNote = dLbl === "Gly10"
                ? "Non-CT HNCACB (2*d28=7.2ms): Gly10 CA is +1 and CB is absent (CT 2T=28ms inverts Gly CH2 to -1)"
                : "Intra-residue CA (" + pulseRegime.split(":")[0] + ")";
              addSpot(scx, yCa, 12, dLbl + " CA(i): " + st.ca_i.toFixed(1) + " ppm (" + sgnCa + ", intra)", ["Phase: " + sgnCa + " sampled from 3D HNCACB trace", glyNote], st.seq_id);
            }

            if (typeof st.cb_i === "number") {
              var yCb = mapCy(st.cb_i);
              var colCb = signColor(st.cb_i_sign, -1);
              var sgnCb = signLabel(st.cb_i_sign, -1);
              var invNote = (typeof st.ca_i === "number" && st.cb_i > st.ca_i) ? ", CB>CA" : "";
              ctx.strokeStyle = colCb;
              ctx.lineWidth = 1.8;
              ctx.beginPath();
              ctx.ellipse(scx, yCb, 11, 5.5, 0, 0, Math.PI * 2);
              ctx.stroke();
              ctx.beginPath();
              ctx.ellipse(scx, yCb, 6, 3.0, 0, 0, Math.PI * 2);
              ctx.stroke();
              ctx.fillStyle = colCb;
              ctx.beginPath();
              ctx.arc(scx, yCb, 2.8, 0, Math.PI * 2);
              ctx.fill();
              addSpot(scx, yCb, 12, dLbl + " CB(i): " + st.cb_i.toFixed(1) + " ppm (" + sgnCb + ", intra" + invNote + ")", ["Phase: " + sgnCb + " sampled from 3D HNCACB trace", invNote ? "Thr/Ser hydroxyl downfield CB > CA" : "Intra-residue CB"], st.seq_id);
            }
          }
        } else if (chIdx === 2) {
          var cvVals = reqArr("cv_vals", 4);
          var cvTemps = reqArr("cv_temps", 4);
          var meanE = reqArr("mean_energies", 4);
          var tCrit = reqNum("t_crit");
          var maxCv = reqNum("max_cv");
          var evoAcc = reqNum("evo_acc_pct");
          var bayesAcc = reqNum("bayes_acc_pct");
          var stdAcc = reqNum("std_acc_pct");
          var ubqReal = reqNum("benchmarks.ubiquitin_realistic_pct");
          var cisd3Acc = reqNum("benchmarks.cisd3_acc_pct");
          var cisd3Hc = reqNum("benchmarks.cisd3_hc_prec_pct");
          var coldMin = reqNum("benchmarks.cold_bmrb_hc_min_pct");
          var coldMax = reqNum("benchmarks.cold_bmrb_hc_max_pct");
          var coldGate = reqNum("benchmarks.cold_bmrb_gate_pct");
          var devTarget = reqStr("benchmarks.dev_target");
          if (coldGate === null || !devTarget) {
            drawFallbackPlaceholder("Module 02 Ch 3: Solver Cv(T) & Benchmark Trajectory");
            return;
          }
          if (!cvVals || !cvTemps || !meanE || tCrit === null || maxCv === null || evoAcc === null || bayesAcc === null || stdAcc === null || ubqReal === null || cisd3Acc === null || cisd3Hc === null || coldMin === null || coldMax === null) {
            drawFallbackPlaceholder("Module 02 Ch 3: Solver Cv(T) & Benchmark Trajectory");
            return;
          }
          drawCardTitle(
            ctx,
            padX,
            padY - 12,
            "Replica-Exchange bayes-sa Cv(T), Mean Energy & Solver Coverage",
            "Live T_crit=" + tCrit.toFixed(2) + " (max Cv=" + maxCv.toFixed(1) + ") · [Historical] UBQ " + ubqReal.toFixed(1) + "% · " + devTarget + " " + cisd3Acc.toFixed(2) + "%/" + cisd3Hc.toFixed(0) + "% HC · Cold " + coldMin.toFixed(2) + "-" + coldMax.toFixed(2) + "% (FAILS >=" + coldGate.toFixed(0) + "%; withdrawn)",
            pal
          );
          drawAxes(ctx, padX, padY + 8, pw, ph - 12, "Cooling Stage (High T -> Low T)", "Normalized Cv(T) / Energy / Assigned %", pal);
          var cvPeak = 1e-9;
          for (var ci = 0; ci < cvVals.length; ci++) cvPeak = Math.max(cvPeak, cvVals[ci]);
          ctx.strokeStyle = pal.violet;
          ctx.lineWidth = 2.2;
          ctx.beginPath();
          for (var ci = 0; ci < cvVals.length; ci++) {
            var u = ci / (cvVals.length - 1);
            var sx = padX + u * pw;
            var sy = padY + ph - 10 - (cvVals[ci] / cvPeak) * 0.82 * (ph - 28);
            if (ci === 0) ctx.moveTo(sx, sy);
            else ctx.lineTo(sx, sy);
            if (cvVals[ci] === cvPeak) {
              addSpot(sx, sy, 14, "Heat-Capacity Peak Cv(T_crit)", ["T = " + cvTemps[ci].toFixed(3), "Cv = " + cvVals[ci].toFixed(2)]);
            }
          }
          ctx.stroke();

          var eMin = Infinity, eMax = -Infinity;
          for (var ei = 0; ei < meanE.length; ei++) {
            eMin = Math.min(eMin, meanE[ei]);
            eMax = Math.max(eMax, meanE[ei]);
          }
          var eSpan = Math.max(1e-6, eMax - eMin);
          ctx.strokeStyle = pal.cyan;
          ctx.lineWidth = 2.0;
          ctx.setLineDash([4, 3]);
          ctx.beginPath();
          for (var ei = 0; ei < meanE.length; ei++) {
            var eu = ei / (meanE.length - 1);
            var ex = padX + eu * pw;
            var ey = padY + ph - 10 - ((meanE[ei] - eMin) / eSpan) * 0.78 * (ph - 28);
            if (ei === 0) ctx.moveTo(ex, ey);
            else ctx.lineTo(ex, ey);
          }
          ctx.stroke();
          ctx.setLineDash([]);

          var solverBars = [
            { lbl: "evo-graph", pct: evoAcc, col: pal.green },
            { lbl: "bayes-sa", pct: bayesAcc, col: pal.teal },
            { lbl: "standard SA", pct: stdAcc, col: pal.amber }
          ];
          for (var sb = 0; sb < solverBars.length; sb++) {
            var barY = padY + 22 + sb * 22;
            var barW = clamp(solverBars[sb].pct / 100.0, 0.05, 1.0) * (pw * 0.34);
            var barX = padX + pw * 0.62;
            ctx.fillStyle = solverBars[sb].col;
            ctx.fillRect(barX, barY, barW, 13);
            ctx.fillStyle = pal.text;
            ctx.font = "600 10.5px ui-monospace, monospace";
            ctx.fillText(solverBars[sb].lbl + ": " + solverBars[sb].pct.toFixed(1) + "%", barX - 6, barY + 10);
            addSpot(barX + barW * 0.5, barY + 6, 16, solverBars[sb].lbl + " (this fixture)", ["Assigned non-Pro coverage: " + solverBars[sb].pct.toFixed(1) + "%"]);
          }
        } else {
          var confArr = reqArr("res_confidences", 76);
          var resLabels = reqArr("res_labels", 76);
          var qcVerdict = reqStr("qc_verdict");
          var nHc = reqNum("n_high_conf");
          var nResTot = reqNum("n_res_total");
          var hcThr = reqNum("hc_conf_threshold");
          if (!confArr || !resLabels || !qcVerdict || nHc === null || nResTot === null || hcThr === null) {
            drawFallbackPlaceholder("Module 02 Ch 4: 76-Residue Confidence Ribbon");
            return;
          }
          drawCardTitle(
            ctx,
            padX,
            padY - 12,
            nResTot + "-Residue Ubiquitin Confidence Ribbon (QC=" + qcVerdict + " · " + nHc + "/" + nResTot + " >= " + hcThr.toFixed(2) + ")",
            "Focus residue: " + (resLabels[focusRes - 1] || ("#" + focusRes)),
            pal
          );
          drawAxes(ctx, padX, padY + 8, pw, ph - 12, "Residue 1.." + nResTot, "Posterior Confidence", pal);
          var threshY = padY + 12 + (1 - hcThr) * (ph - 24);
          ctx.strokeStyle = pal.amber;
          ctx.setLineDash([5, 4]);
          ctx.beginPath();
          ctx.moveTo(padX, threshY);
          ctx.lineTo(padX + pw, threshY);
          ctx.stroke();
          ctx.setLineDash([]);
          for (var r = 1; r <= confArr.length; r++) {
            var conf = Number(confArr[r - 1]) || 0;
            var bx = padX + ((r - 1) / confArr.length) * pw;
            var bw = Math.max(2, pw / 86);
            var bh = Math.max(2, conf * (ph - 24));
            var by = padY + ph - 12 - bh;
            ctx.fillStyle = (r === focusRes) ? pal.amber : (conf >= hcThr ? pal.green : pal.coral);
            ctx.fillRect(bx, by, bw, bh);
            if (r % 5 === 0 || r === focusRes) {
              addSpot(bx + bw / 2, by, 10, resLabels[r - 1] || ("Residue " + r), ["Confidence: " + (conf * 100).toFixed(1) + "%", "QC verdict: " + qcVerdict], r <= 10 ? r : focusRes);
            }
          }
        }
        return;
      }

      if (modIdx === 3) {
        var orbitDeg = clamp(ctrlVal, 0, 360);
        var yaw = (orbitDeg * Math.PI) / 180.0;
        var pitch = (state && typeof state.pitchRad === "number") ? state.pitchRad : 0.35;
        if (chIdx === 2) {
          var rdcPts = reqArr("rdc_points", 8);
          var qWork = reqNum("q_work");
          var qFree = reqNum("q_free");
          var f8Note = reqStr("f8_holdout_note");
          if (!rdcPts || qWork === null || qFree === null || !f8Note) {
            drawFallbackPlaceholder("Module 03 Ch 3: Saupe Alignment Tensor & RDC Q_free");
            return;
          }
          drawCardTitle(
            ctx,
            padX,
            padY - 12,
            "Saupe Alignment Tensor & N-H RDC D_obs vs D_calc",
            "Q_work=" + qWork.toFixed(3) + " · Q_free=" + qFree.toFixed(3) + " · " + f8Note,
            pal
          );
          drawAxes(ctx, padX, padY + 8, pw, ph - 12, "D_calc (Hz)", "D_obs (Hz)", pal);
          ctx.strokeStyle = pal.amber;
          ctx.setLineDash([5, 4]);
          ctx.beginPath();
          ctx.moveTo(padX + 12, padY + ph - 14);
          ctx.lineTo(padX + pw - 12, padY + 18);
          ctx.stroke();
          ctx.setLineDash([]);
          var maxD = 16.0;
          for (var i = 0; i < rdcPts.length; i++) {
            maxD = Math.max(maxD, Math.abs(rdcPts[i].d_calc || 0), Math.abs(rdcPts[i].d_obs || 0));
          }
          var stepRdc = Math.max(1, Math.floor(rdcPts.length / 72));
          for (var i = 0; i < rdcPts.length; i += stepRdc) {
            var rp = rdcPts[i];
            var sx = padX + clamp((rp.d_calc + maxD) / (2 * maxD), 0.03, 0.97) * pw;
            var sy = padY + ph - 12 - clamp((rp.d_obs + maxD) / (2 * maxD), 0.03, 0.97) * (ph - 24);
            ctx.fillStyle = rp.is_free ? pal.coral : pal.cyan;
            ctx.beginPath();
            ctx.arc(sx, sy, rp.is_free ? 4.8 : 3.8, 0, Math.PI * 2);
            ctx.fill();
            addSpot(sx, sy, 10, (rp.is_free ? "Q_free" : "Q_work") + " Res #" + rp.seq_id, ["D_calc: " + rp.d_calc.toFixed(2) + " Hz", "D_obs: " + rp.d_obs.toFixed(2) + " Hz"]);
          }
          return;
        }

        var rawCa = reqArr("ca_coords", 10);
        var rawSs = reqArr("ss_classes", 10);
        var rawS2 = reqArr("s2_vals", 10);
        var f8Sub = reqStr("f8_holdout_note");
        var noePairs = chIdx === 1 ? reqArr("noe_pairs", 4) : null;
        if (!rawCa || !rawSs || !rawS2 || !f8Sub || (chIdx === 1 && !noePairs)) {
          drawFallbackPlaceholder("Module 03: 1UBQ 3D Coordinates & NOE/S^2 Projection");
          return;
        }
        var cx0 = 0, cy0 = 0, cz0 = 0;
        for (var i = 0; i < rawCa.length; i++) {
          cx0 += rawCa[i][0]; cy0 += rawCa[i][1]; cz0 += rawCa[i][2];
        }
        cx0 /= rawCa.length; cy0 /= rawCa.length; cz0 /= rawCa.length;
        var maxR = 1e-6;
        for (var i = 0; i < rawCa.length; i++) {
          var dx = rawCa[i][0] - cx0, dy = rawCa[i][1] - cy0, dz = rawCa[i][2] - cz0;
          maxR = Math.max(maxR, Math.sqrt(dx * dx + dy * dy + dz * dz));
        }
        var coords = [];
        for (var i = 0; i < rawCa.length; i++) {
          var ssCode = rawSs[i] !== undefined ? rawSs[i] : 0;
          var ssName = ssCode === 1 ? "helix" : (ssCode === 2 ? "sheet" : "coil");
          coords.push({
            x: (rawCa[i][0] - cx0) / maxR,
            y: (rawCa[i][1] - cy0) / maxR,
            z: (rawCa[i][2] - cz0) / maxR,
            ss: ssName,
            s2: typeof rawS2[i] === "number" ? rawS2[i] : 0.84
          });
        }
        function proj(pt) {
          var cx = Math.cos(yaw), sx = Math.sin(yaw);
          var cp = Math.cos(pitch), sp = Math.sin(pitch);
          var x1 = pt.x * cx - pt.z * sx;
          var z1 = pt.x * sx + pt.z * cx;
          var y2 = pt.y * cp - z1 * sp;
          var z2 = pt.y * sp + z1 * cp;
          var scale = Math.min(pw, ph) * 0.98 / (2.8 - z2 * 0.35);
          return { x: w * 0.5 + x1 * scale, y: padY + ph * 0.50 + y2 * scale, z: z2 };
        }
        var titleStr = chIdx === 0
          ? "1UBQ 3D Backbone CA Trace (Drag Canvas to Orbit · " + Math.round(orbitDeg) + " deg)"
          : (chIdx === 1 ? "Long-Range 1H-1H NOE Cylinders (" + noePairs.length + " pairs) + <=25% Firewall" : "1UBQ S^2 Putty Projection · 360 deg Orbit (" + Math.round(orbitDeg) + " deg)");
        drawCardTitle(ctx, padX, padY - 12, titleStr, f8Sub, pal);
        var projected = coords.map(function (c, idx) {
          var p = proj(c);
          return { x: p.x, y: p.y, z: p.z, ss: c.ss, s2: c.s2, idx: idx + 1 };
        });
        if (chIdx === 1 && noePairs) {
          ctx.strokeStyle = "rgba(251, 191, 36, 0.55)";
          ctx.setLineDash([4, 3]);
          ctx.lineWidth = 1.5;
          for (var k = 0; k < noePairs.length; k++) {
            var np = noePairs[k];
            var pa = projected[clamp(np.a || 0, 0, projected.length - 1)];
            var pb = projected[clamp(np.b || 0, 0, projected.length - 1)];
            ctx.beginPath();
            ctx.moveTo(pa.x, pa.y);
            ctx.lineTo(pb.x, pb.y);
            ctx.stroke();
          }
          ctx.setLineDash([]);
        }
        for (var i = 1; i < projected.length; i++) {
          var p0 = projected[i - 1], p1 = projected[i];
          ctx.strokeStyle = p1.ss === "helix" ? pal.coral : (p1.ss === "sheet" ? pal.cyan : pal.teal);
          ctx.lineWidth = chIdx === 3 ? (2 + (1 - p1.s2) * 14) : 3.2;
          ctx.beginPath();
          ctx.moveTo(p0.x, p0.y);
          ctx.lineTo(p1.x, p1.y);
          ctx.stroke();
        }
        var sorted = projected.slice().sort(function (a, b) { return a.z - b.z; });
        for (var i = 0; i < sorted.length; i++) {
          var pt = sorted[i];
          ctx.fillStyle = pt.ss === "helix" ? pal.coral : (pt.ss === "sheet" ? pal.cyan : pal.green);
          ctx.beginPath();
          ctx.arc(pt.x, pt.y, 4.0, 0, Math.PI * 2);
          ctx.fill();
          if (pt.idx % 2 === 1) {
            addSpot(pt.x, pt.y, 11, "1UBQ CA #" + pt.idx, ["Secondary: " + pt.ss, "S^2: " + pt.s2.toFixed(2)]);
          }
        }
        return;
      }

      if (modIdx === 4) {
        var lastSeq = reqNum("last_seq_id");
        var fieldA = reqNum("field_a_mhz");
        var fieldB = reqNum("field_b_mhz");
        if (lastSeq === null || fieldA === null || fieldB === null) {
          drawFallbackPlaceholder("Module 04: missing payload last_seq_id / field_a_mhz / field_b_mhz");
          return;
        }
        var fieldALbl = fieldA.toFixed(0) + " MHz";
        var fieldBLbl = fieldB.toFixed(0) + " MHz";
        var resSeq = Math.round(clamp(ctrlVal, 1, lastSeq));
        if (chIdx === 0) {
          var snrLevels = reqArr("snr_levels", 3);
          var lsBias = reqArr("locked_ls_bias_pct", 3);
          var maxBias = reqArr("local_max_bias_pct", 3);
          if (!snrLevels || !lsBias || !maxBias) {
            drawFallbackPlaceholder("Module 04 Ch 1: Locked-Footprint Integration Bias");
            return;
          }
          var nTiers = Math.min(snrLevels.length, lsBias.length, maxBias.length);
          var topBias = 5.0;
          for (var i = 0; i < nTiers; i++) {
            topBias = Math.max(topBias, Math.abs(lsBias[i]), Math.abs(maxBias[i]));
          }
          var lastIdx = nTiers - 1;
          drawCardTitle(
            ctx,
            padX,
            padY - 12,
            "SYNTHETIC: Pseudo-3D Locked-Footprint Integration (SNR " + snrLevels[0] + " -> " + snrLevels[lastIdx] + ")",
            "SNR=" + snrLevels[lastIdx] + " bias: Locked LS " + lsBias[lastIdx].toFixed(2) + "% vs Local-Max +" + maxBias[lastIdx].toFixed(1) + "%",
            pal
          );
          drawAxes(ctx, padX, padY + 8, pw, ph - 12, "SNR Tier (" + snrLevels.join(" -> ") + ")", "|Relative Bias| (%)", pal);
          for (var i = 0; i < nTiers; i++) {
            var bx = padX + (i + 0.5) * (pw / nTiers);
            var hMax = (Math.abs(maxBias[i]) / (topBias * 1.12)) * (ph - 24);
            var hLs = Math.max(2, (Math.abs(lsBias[i]) / (topBias * 1.12)) * (ph - 24));
            ctx.fillStyle = pal.coral;
            ctx.fillRect(bx - 14, padY + ph - 12 - hMax, 12, hMax);
            ctx.fillStyle = pal.green;
            ctx.fillRect(bx + 2, padY + ph - 12 - hLs, 12, hLs);
            addSpot(bx, padY + ph * 0.5, 20, "SNR = " + snrLevels[i], ["Local-Max bias: +" + maxBias[i].toFixed(2) + "%", "Locked-Footprint LS: " + lsBias[i].toFixed(2) + "%"]);
          }
        } else if (chIdx === 1) {
          var timesR1 = reqArr("times_r1", 4);
          var c600 = reqArr("curve_600_r1", 4);
          var c800 = reqArr("curve_800_r1", 4);
          var r600 = reqNum("fit_600_rate");
          var se600 = reqNum("fit_600_se");
          var r800 = reqNum("fit_800_rate");
          var se800 = reqNum("fit_800_se");
          var decayLbl = reqStr("decay_demo_label");
          var mcReps = reqNum("fit_mc_replicates");
          if (!timesR1 || !c600 || !c800 || r600 === null || se600 === null || r800 === null || se800 === null || !decayLbl || mcReps === null) {
            drawFallbackPlaceholder("Module 04 Ch 2: Dual-Field Exponential Decay Fit");
            return;
          }
          drawCardTitle(
            ctx,
            padX,
            padY - 12,
            "SYNTHETIC: Dual-Field " + fieldA.toFixed(0) + " & " + fieldBLbl + " R1 Decay Fit (" + decayLbl + ")",
            fieldALbl + " R1=" + r600.toFixed(2) + "+-" + se600.toFixed(2) + " s^-1 · " + fieldBLbl + " R1=" + r800.toFixed(2) + "+-" + se800.toFixed(2) + " s^-1 (" + mcReps.toFixed(0) + "-trial MC SE)",
            pal
          );
          drawAxes(ctx, padX, padY + 8, pw, ph - 12, "Relaxation Delay T (s)", "Intensity I(T)", pal);
          var tMax = Math.max(0.1, timesR1[timesR1.length - 1]);
          var iMax = 1e-6;
          for (var k = 0; k < c600.length; k++) iMax = Math.max(iMax, c600[k], c800[k] || 0);
          var fitSeries = [
            { col: pal.cyan, rate: r600, se: se600, obs: c600, i0: c600[0] * Math.exp(r600 * timesR1[0]), lbl: fieldALbl },
            { col: pal.amber, rate: r800, se: se800, obs: c800, i0: c800[0] * Math.exp(r800 * timesR1[0]), lbl: fieldBLbl }
          ];
          for (var fi = 0; fi < fitSeries.length; fi++) {
            var fs = fitSeries[fi];
            ctx.strokeStyle = fs.col;
            ctx.lineWidth = 2.2;
            ctx.beginPath();
            for (var j = 0; j <= 48; j++) {
              var tSecVal = (j / 48) * tMax;
              var yFit = fs.i0 * Math.exp(-fs.rate * tSecVal);
              var sx = padX + (tSecVal / tMax) * pw;
              var sy = padY + ph - 12 - clamp(yFit / (iMax * 1.05), 0, 1) * (ph - 26);
              if (j === 0) ctx.moveTo(sx, sy);
              else ctx.lineTo(sx, sy);
            }
            ctx.stroke();
            for (var p = 0; p < timesR1.length && p < fs.obs.length; p++) {
              var px = padX + (timesR1[p] / tMax) * pw;
              var py = padY + ph - 12 - clamp(fs.obs[p] / (iMax * 1.05), 0, 1) * (ph - 26);
              ctx.fillStyle = fs.col;
              ctx.fillRect(px - 3.5, py - 3.5, 7, 7);
              addSpot(px, py, 12, fs.lbl + " @ T=" + timesR1[p].toFixed(2) + " s", ["Observed I(T): " + fs.obs[p].toFixed(2), "Fitted R1: " + fs.rate.toFixed(3) + " +- " + fs.se.toFixed(3) + " s^-1"]);
            }
          }
        } else if (chIdx === 2) {
          var seqIds = reqArr("seq_ids", 10);
          var ratios = reqArr("r2_over_r1", 10);
          var tauM = reqNum("tau_m_ns");
          var dParPerp = reqNum("d_par_perp");
          var tailStart = reqNum("sim_tail_start_seq");
          if (!seqIds || !ratios || tauM === null || dParPerp === null || tailStart === null) {
            drawFallbackPlaceholder("Module 04 Ch 3: Kay-Torchia-Bax R2/R1 & tau_m");
            return;
          }
          drawCardTitle(
            ctx,
            padX,
            padY - 12,
            "SYNTHETIC: R2/R1 Global Tumbling tau_m = " + tauM.toFixed(2) + " ns (Inspect Res #" + resSeq + ")",
            "Woessner axial D_par/D_perp = " + dParPerp.toFixed(2) + " · 10% outlier-trimmed rigid core",
            pal
          );
          drawAxes(ctx, padX, padY + 8, pw, ph - 12, "Residue Sequence (1.." + lastSeq + ")", fieldALbl + " R2 / R1", pal);
          var maxRatio = 1.0;
          var sortedR = ratios.slice().sort(function (a, b) { return a - b; });
          var medRatio = sortedR[Math.floor(sortedR.length * 0.5)] || 1.0;
          for (var i = 0; i < ratios.length; i++) maxRatio = Math.max(maxRatio, ratios[i]);
          var coreY = padY + ph - 12 - clamp(medRatio / (maxRatio * 1.12), 0, 1) * (ph - 26);
          ctx.strokeStyle = pal.teal;
          ctx.setLineDash([5, 4]);
          ctx.beginPath();
          ctx.moveTo(padX, coreY);
          ctx.lineTo(padX + pw, coreY);
          ctx.stroke();
          ctx.setLineDash([]);
          for (var i = 0; i < seqIds.length && i < ratios.length; i++) {
            var rId = seqIds[i];
            var ratio = ratios[i];
            var sx = padX + ((rId - 0.5) / lastSeq) * pw;
            var sy = padY + ph - 12 - clamp(ratio / (maxRatio * 1.12), 0, 1) * (ph - 26);
            var isSel = rId === resSeq;
            var isOutlier = ratio > medRatio * 1.22;
            ctx.fillStyle = isSel ? pal.amber : (isOutlier ? pal.coral : (rId >= tailStart ? pal.violet : pal.teal));
            ctx.beginPath();
            ctx.arc(sx, sy, isSel ? 6 : 3.5, 0, Math.PI * 2);
            ctx.fill();
            addSpot(sx, sy, 10, "Res #" + rId, ["R2/R1: " + ratio.toFixed(2), "tau_m = " + tauM.toFixed(2) + " ns"], rId);
          }
        } else {
          var seqIds = reqArr("seq_ids", 10);
          var s2Vals = reqArr("s2_vals", 10);
          var rexVals = reqArr("rex_vals", 10);
          var modelIds = reqArr("model_ids", 10);
          var rexThr = reqNum("rex_report_threshold_s_inv");
          var rexClause = reqStr("rex_clause");
          if (!seqIds || !s2Vals || !rexVals || !modelIds || rexThr === null || !rexClause) {
            drawFallbackPlaceholder("Module 04 Ch 4: Lipari-Szabo M1-M5 S^2 Profile");
            return;
          }
          var nRex = 0;
          for (var k = 0; k < rexVals.length; k++) { if (rexVals[k] > rexThr) nRex++; }
          drawCardTitle(
            ctx,
            padX,
            padY - 12,
            "SYNTHETIC: Lipari-Szabo M1-M5 S^2 Profile (Inspect Res #" + resSeq + ")",
            seqIds.length + " fitted non-Pro residues · " + nRex + " chemical-exchange (R_ex > " + rexThr.toFixed(1) + " s^-1) sites · fit: " + rexClause,
            pal
          );
          drawAxes(ctx, padX, padY + 8, pw, ph - 12, "Residue Sequence (1.." + lastSeq + ")", "Order Parameter S^2", pal);
          for (var i = 0; i < seqIds.length && i < s2Vals.length; i++) {
            var rId = seqIds[i];
            var s2 = s2Vals[i];
            var rex = rexVals[i] || 0;
            var mdl = modelIds[i] || "M1";
            var sx = padX + ((rId - 0.5) / lastSeq) * pw;
            var sy = padY + 14 + (1 - clamp(s2, 0, 1)) * (ph - 28);
            var isSel = rId === resSeq;
            ctx.fillStyle = isSel ? pal.amber : (rex > rexThr ? pal.coral : pal.cyan);
            ctx.beginPath();
            ctx.arc(sx, sy, isSel ? 6 : 3.5, 0, Math.PI * 2);
            ctx.fill();
            addSpot(sx, sy, 10, "Res #" + rId + " (" + mdl + ")", ["S^2: " + s2.toFixed(3), "R_ex: " + rex.toFixed(2) + " s^-1"], rId);
          }
        }
        return;
      }

      if (modIdx === 5) {
        var ligUm = clamp(ctrlVal, 0, 300);
        if (chIdx === 0) {
          var offsets = reqArr("cest.offsets_ppm", 8);
          var obs15 = reqArr("cest.b1_15hz_obs", 8);
          var fit15 = reqArr("cest.b1_15hz_fit", 8);
          var obs30 = reqArr("cest.b1_30hz_obs", 8);
          var fit30 = reqArr("cest.b1_30hz_fit", 8);
          var pB = reqNum("cest.p_b");
          var kEx = reqNum("cest.k_ex_s_inv");
          var majPpm = reqNum("cest.major_ppm");
          var dwPpm = reqNum("cest.dw_ab_ppm");
          if (!offsets || !obs15 || !fit15 || !obs30 || !fit30 || pB === null || kEx === null || majPpm === null || dwPpm === null) {
            drawFallbackPlaceholder("Module 05 Ch 1: CEST Saturation Sweep");
            return;
          }
          var minOff = offsets[0], maxOff = offsets[offsets.length - 1];
          var spanOff = Math.max(1e-6, maxOff - minOff);
          drawCardTitle(
            ctx,
            padX,
            padY - 12,
            "SYNTHETIC: Higham [6/6] Pade 15N CEST Sweep (B1 = 15 & 30 Hz)",
            "Fitted pB=" + (pB * 100).toFixed(2) + "% · k_ex=" + kEx.toFixed(0) + " s^-1 · major=" + majPpm.toFixed(2) + " ppm · dw=+" + dwPpm.toFixed(2) + " ppm",
            pal
          );
          drawAxes(ctx, padX, padY + 8, pw, ph - 12, "15N B1 Offset (" + minOff.toFixed(1) + ".." + maxOff.toFixed(1) + " ppm)", "I / I0", pal);
          var cestCurves = [
            { col: pal.cyan, fit: fit15, obs: obs15, lbl: "15 Hz B1" },
            { col: pal.amber, fit: fit30, obs: obs30, lbl: "30 Hz B1" }
          ];
          for (var b = 0; b < cestCurves.length; b++) {
            var cc = cestCurves[b];
            ctx.strokeStyle = cc.col;
            ctx.lineWidth = 2.1;
            ctx.beginPath();
            for (var i = 0; i < offsets.length && i < cc.fit.length; i++) {
              var sx = padX + ((offsets[i] - minOff) / spanOff) * pw;
              var sy = padY + 14 + (1 - clamp(cc.fit[i], 0, 1.05)) * (ph - 28);
              if (i === 0) ctx.moveTo(sx, sy);
              else ctx.lineTo(sx, sy);
            }
            ctx.stroke();
            for (var i = 0; i < offsets.length && i < cc.obs.length; i += 2) {
              var ox = padX + ((offsets[i] - minOff) / spanOff) * pw;
              var oy = padY + 14 + (1 - clamp(cc.obs[i], 0, 1.05)) * (ph - 28);
              ctx.fillStyle = cc.col;
              ctx.beginPath();
              ctx.arc(ox, oy, 2.6, 0, Math.PI * 2);
              ctx.fill();
              if (Math.abs(offsets[i] - (majPpm + dwPpm)) < 0.25) {
                addSpot(ox, oy, 15, "Minor State Dip (" + cc.lbl + ")", ["Offset: " + offsets[i].toFixed(2) + " ppm", "I/I0: " + cc.obs[i].toFixed(3)]);
              }
            }
          }
        } else if (chIdx === 1) {
          var nuHz = reqArr("cpmg.nu_hz", 4);
          var r600 = reqArr("cpmg.r2_600_obs", 4);
          var r800 = reqArr("cpmg.r2_800_obs", 4);
          var cPb = reqNum("cpmg.p_b");
          var cKex = reqNum("cpmg.k_ex_s_inv");
          var dAicc = reqNum("cpmg.delta_aicc");
          if (!nuHz || !r600 || !r800 || cPb === null || cKex === null || dAicc === null) {
            drawFallbackPlaceholder("Module 05 Ch 2: Baldwin CPMG Dispersion");
            return;
          }
          drawCardTitle(
            ctx,
            padX,
            padY - 12,
            "SYNTHETIC: Dual-Field Baldwin (2014) CPMG Dispersion & AICc Gate",
            "Fitted pB=" + (cPb * 100).toFixed(2) + "% · k_ex=" + cKex.toFixed(0) + " s^-1 · delta_AICc=" + dAicc.toFixed(1),
            pal
          );
          drawAxes(ctx, padX, padY + 8, pw, ph - 12, "nu_CPMG (Hz)", "R2,eff (s^-1)", pal);
          var nuMax = Math.max(100, nuHz[nuHz.length - 1]);
          var rMin = Infinity, rMax = -Infinity;
          for (var i = 0; i < r600.length; i++) { rMin = Math.min(rMin, r600[i]); rMax = Math.max(rMax, r600[i]); }
          for (var i = 0; i < r800.length; i++) { rMin = Math.min(rMin, r800[i]); rMax = Math.max(rMax, r800[i]); }
          rMin = Math.max(0, rMin - 1.5);
          rMax = rMax + 1.5;
          var rSpan = Math.max(1.0, rMax - rMin);
          var dispSeries = [
            { col: pal.cyan, arr: r600, lbl: "600 MHz" },
            { col: pal.coral, arr: r800, lbl: "800 MHz" }
          ];
          for (var f = 0; f < dispSeries.length; f++) {
            var ds = dispSeries[f];
            ctx.strokeStyle = ds.col;
            ctx.lineWidth = 2.3;
            ctx.beginPath();
            for (var i = 0; i < nuHz.length && i < ds.arr.length; i++) {
              var sx = padX + (nuHz[i] / nuMax) * pw;
              var sy = padY + ph - 12 - ((ds.arr[i] - rMin) / rSpan) * (ph - 26);
              if (i === 0) ctx.moveTo(sx, sy);
              else ctx.lineTo(sx, sy);
            }
            ctx.stroke();
            for (var i = 0; i < nuHz.length && i < ds.arr.length; i++) {
              var px = padX + (nuHz[i] / nuMax) * pw;
              var py = padY + ph - 12 - ((ds.arr[i] - rMin) / rSpan) * (ph - 26);
              ctx.fillStyle = ds.col;
              ctx.beginPath();
              ctx.arc(px, py, 4.2, 0, Math.PI * 2);
              ctx.fill();
              addSpot(px, py, 12, ds.lbl + " @ " + nuHz[i].toFixed(0) + " Hz", ["R2,eff: " + ds.arr[i].toFixed(2) + " s^-1", "delta_AICc: " + dAicc.toFixed(1)]);
            }
          }
        } else if (chIdx === 2) {
          var trajs = reqArr("hsqc_trajectories", 2);
          if (!trajs) {
            drawFallbackPlaceholder("Module 05 Ch 3: Hungarian HSQC Titration Trajectories");
            return;
          }
          var hMin = Infinity, hMax = -Infinity, nMin = Infinity, nMax = -Infinity;
          for (var t = 0; t < trajs.length; t++) {
            var pts = trajs[t].points || [];
            for (var p = 0; p < pts.length; p++) {
              if (typeof pts[p].h_ppm === "number" && typeof pts[p].n_ppm === "number") {
                hMin = Math.min(hMin, pts[p].h_ppm);
                hMax = Math.max(hMax, pts[p].h_ppm);
                nMin = Math.min(nMin, pts[p].n_ppm);
                nMax = Math.max(nMax, pts[p].n_ppm);
              }
            }
          }
          hMin -= 0.12; hMax += 0.12; nMin -= 0.8; nMax += 0.8;
          var hSpan = Math.max(0.2, hMax - hMin);
          var nSpan = Math.max(1.0, nMax - nMin);
          drawCardTitle(
            ctx,
            padX,
            padY - 12,
            "SYNTHETIC: Hungarian 2D HSQC Titration Trajectories ([L]t = " + ligUm.toFixed(0) + " uM)",
            trajs.length + " tracked residues · max observed CSP = " + Number(trajs[0].max_csp || 0).toFixed(3) + " ppm",
            pal
          );
          drawAxes(ctx, padX, padY + 8, pw, ph - 12, "1H (ppm, " + hMax.toFixed(2) + " -> " + hMin.toFixed(2) + ")", "15N (ppm, " + nMin.toFixed(1) + " -> " + nMax.toFixed(1) + ")", pal);
          var cols = [pal.cyan, pal.green, pal.amber, pal.violet, pal.coral];
          for (var t = 0; t < trajs.length; t++) {
            var tr = trajs[t];
            var pts = (tr.points || []).filter(function (pt) { return typeof pt.h_ppm === "number" && typeof pt.n_ppm === "number"; });
            if (!pts.length) continue;
            var col = cols[t % cols.length];
            ctx.strokeStyle = col;
            ctx.lineWidth = 1.8;
            ctx.setLineDash([4, 3]);
            ctx.beginPath();
            var bestPt = pts[0];
            var bestDiff = Infinity;
            for (var p = 0; p < pts.length; p++) {
              var sx = padX + ((hMax - pts[p].h_ppm) / hSpan) * pw;
              var sy = padY + 14 + ((pts[p].n_ppm - nMin) / nSpan) * (ph - 28);
              if (p === 0) ctx.moveTo(sx, sy);
              else ctx.lineTo(sx, sy);
              var dL = Math.abs((pts[p].ligand_um || 0) - ligUm);
              if (dL < bestDiff) { bestDiff = dL; bestPt = pts[p]; }
            }
            ctx.stroke();
            ctx.setLineDash([]);
            var bx = padX + ((hMax - bestPt.h_ppm) / hSpan) * pw;
            var by = padY + 14 + ((bestPt.n_ppm - nMin) / nSpan) * (ph - 28);
            ctx.fillStyle = col;
            ctx.beginPath();
            ctx.arc(bx, by, 5.8, 0, Math.PI * 2);
            ctx.fill();
            var rLbl = String(tr.res_name || "Res") + String(tr.seq_id || "");
            ctx.fillStyle = pal.text;
            ctx.font = "600 11px ui-monospace, monospace";
            ctx.fillText(rLbl, bx + 8, by + 4);
            addSpot(bx, by, 14, rLbl + " @ [L]=" + Number(bestPt.ligand_um || 0).toFixed(0) + " uM", ["1H: " + bestPt.h_ppm.toFixed(3) + " ppm, 15N: " + bestPt.n_ppm.toFixed(2) + " ppm", "CSP: " + Number(bestPt.csp_ppm || 0).toFixed(3) + " ppm"], Number(bestPt.ligand_um || 0));
          }
        } else {
          var P0 = reqNum("csp_isotherm.protein_conc_um");
          var kdQuad = reqNum("csp_isotherm.fitted_kd_um");
          var kdMm = reqNum("csp_isotherm.naive_mm_kd_um");
          var dMax = reqNum("csp_isotherm.ddelta_max_ppm");
          var obsPts = reqArr("csp_isotherm.observed_points", 3);
          if (P0 === null || kdQuad === null || kdMm === null || dMax === null || !obsPts) {
            drawFallbackPlaceholder("Module 05 Ch 4: Morrison Quadratic vs Naive MM Isotherm");
            return;
          }
          drawCardTitle(
            ctx,
            padX,
            padY - 12,
            "SYNTHETIC: Morrison Quadratic (Kd=" + kdQuad.toFixed(2) + " uM, [P]0=" + P0.toFixed(0) + " uM) vs Naive MM (" + kdMm.toFixed(1) + " uM)",
            "Cursor [L]t = " + ligUm.toFixed(0) + " uM · ddelta_max = " + dMax.toFixed(3) + " ppm",
            pal
          );
          drawAxes(ctx, padX, padY + 8, pw, ph - 12, "[L] total (0..300 uM)", "CSP (ppm)", pal);
          var yCeil = Math.max(1e-4, dMax * 1.12);
          function quadCsp(L) {
            var b = P0 + L + kdQuad;
            var frac = (b - Math.sqrt(Math.max(0, b * b - 4 * P0 * L))) / (2 * P0);
            return dMax * frac;
          }
          function mmCsp(L) {
            return dMax * (L / (kdMm + L));
          }
          ctx.strokeStyle = pal.green;
          ctx.lineWidth = 2.4;
          ctx.beginPath();
          for (var i = 0; i <= 60; i++) {
            var L = i * 5;
            var sx = padX + (L / 300) * pw;
            var sy = padY + ph - 12 - clamp(quadCsp(L) / yCeil, 0, 1) * (ph - 24);
            if (i === 0) ctx.moveTo(sx, sy);
            else ctx.lineTo(sx, sy);
          }
          ctx.stroke();
          ctx.strokeStyle = pal.coral;
          ctx.setLineDash([5, 4]);
          ctx.beginPath();
          for (var i = 0; i <= 60; i++) {
            var L = i * 5;
            var sx = padX + (L / 300) * pw;
            var sy = padY + ph - 12 - clamp(mmCsp(L) / yCeil, 0, 1) * (ph - 24);
            if (i === 0) ctx.moveTo(sx, sy);
            else ctx.lineTo(sx, sy);
          }
          ctx.stroke();
          ctx.setLineDash([]);
          for (var p = 0; p < obsPts.length; p++) {
            var ox = padX + clamp((obsPts[p].ligand_um || 0) / 300, 0, 1) * pw;
            var oy = padY + ph - 12 - clamp((obsPts[p].csp_ppm || 0) / yCeil, 0, 1) * (ph - 24);
            ctx.fillStyle = pal.cyan;
            ctx.beginPath();
            ctx.arc(ox, oy, 4.5, 0, Math.PI * 2);
            ctx.fill();
            addSpot(ox, oy, 12, "Titration Step [L]=" + Number(obsPts[p].ligand_um).toFixed(0) + " uM", ["Observed CSP: " + Number(obsPts[p].csp_ppm).toFixed(3) + " ppm"], Number(obsPts[p].ligand_um));
          }
          var cx = padX + (ligUm / 300) * pw;
          var cyQ = padY + ph - 12 - clamp(quadCsp(ligUm) / yCeil, 0, 1) * (ph - 24);
          ctx.strokeStyle = pal.amber;
          ctx.beginPath();
          ctx.moveTo(cx, padY + 8);
          ctx.lineTo(cx, padY + ph - 4);
          ctx.stroke();
          ctx.fillStyle = pal.amber;
          ctx.beginPath();
          ctx.arc(cx, cyQ, 6, 0, Math.PI * 2);
          ctx.fill();
          addSpot(cx, cyQ, 18, "Morrison Fit @ " + ligUm.toFixed(0) + " uM", ["Morrison CSP: " + quadCsp(ligUm).toFixed(3) + " ppm", "Naive MM CSP: " + mmCsp(ligUm).toFixed(3) + " ppm"]);
        }
        return;
      }

      if (modIdx === 6) {
        var atomIdx = Math.round(clamp(ctrlVal, 0, 14));
        if (chIdx === 0) {
          var fPpm = reqArr("fid_processing.ppm", 10);
          var withW = reqArr("fid_processing.with_water", 10);
          var flat = reqArr("fid_processing.hsvd_flattened", 10);
          var roll = reqArr("fid_processing.baseline_roll", 10);
          var vPeaks = reqArr("fid_processing.voigt_peaks", 1);
          if (!fPpm || !withW || !flat || !roll || !vPeaks) {
            drawFallbackPlaceholder("Module 06 Ch 1: 1D HSVD & Whittaker Baseline");
            return;
          }
          var pHi = Math.max(fPpm[0], fPpm[fPpm.length - 1]);
          var pLo = Math.min(fPpm[0], fPpm[fPpm.length - 1]);
          var pSpan = Math.max(0.1, pHi - pLo);
          var maxFlat = 1e-6;
          for (var i = 0; i < flat.length; i++) maxFlat = Math.max(maxFlat, Math.abs(flat[i]), Math.abs(roll[i] || 0));
          var yCap = maxFlat * 1.45;
          drawCardTitle(
            ctx,
            padX,
            padY - 12,
            "SYNTHETIC: 1D HSVD Solvent Suppression (4.78 ppm H2O) + Whittaker Baseline",
            vPeaks.length + " deconvolved Voigt solute components · Coral=pre-HSVD (clipped) · Cyan=HSVD+Whittaker",
            pal
          );
          drawAxes(ctx, padX, padY + 8, pw, ph - 12, "1H (ppm, " + pHi.toFixed(2) + " -> " + pLo.toFixed(2) + ")", "Intensity", pal);
          ctx.strokeStyle = pal.coral;
          ctx.lineWidth = 1.4;
          ctx.setLineDash([4, 3]);
          ctx.beginPath();
          for (var i = 0; i < fPpm.length && i < withW.length; i++) {
            var sx = padX + ((pHi - fPpm[i]) / pSpan) * pw;
            var sy = padY + ph - 14 - clamp(withW[i] / yCap, -0.05, 1.0) * (ph - 28);
            if (i === 0) ctx.moveTo(sx, sy);
            else ctx.lineTo(sx, sy);
          }
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.strokeStyle = pal.cyan;
          ctx.lineWidth = 2.2;
          ctx.beginPath();
          for (var i = 0; i < fPpm.length && i < flat.length; i++) {
            var sx = padX + ((pHi - fPpm[i]) / pSpan) * pw;
            var sy = padY + ph - 14 - clamp(flat[i] / yCap, -0.05, 1.0) * (ph - 28);
            if (i === 0) ctx.moveTo(sx, sy);
            else ctx.lineTo(sx, sy);
          }
          ctx.stroke();
          for (var v = 0; v < vPeaks.length; v++) {
            var vp = vPeaks[v];
            var vx = padX + clamp((pHi - vp.center_ppm) / pSpan, 0.02, 0.98) * pw;
            var vy = padY + ph - 14 - clamp((vp.height || 0) / yCap, 0, 1.0) * (ph - 28);
            ctx.fillStyle = pal.amber;
            ctx.beginPath();
            ctx.arc(vx, vy, 4.2, 0, Math.PI * 2);
            ctx.fill();
            addSpot(vx, vy, 14, "Voigt Peak " + vp.center_ppm.toFixed(3) + " ppm", ["FWHM: " + Number(vp.fwhm_hz || 0).toFixed(2) + " Hz", "Lorentz frac: " + Number(vp.lorentz_fraction || 0).toFixed(2)]);
          }
        } else if (chIdx === 1) {
          var mPpm = reqArr("multiplet.ppm", 20);
          var mInt = reqArr("multiplet.intensity", 20);
          var mClass = reqStr("multiplet.class");
          var jHz = reqArr("multiplet.j_hz", 1);
          var acsRep = reqStr("multiplet.acs_report");
          if (!mPpm || !mInt || !mClass || !jHz || !acsRep) {
            drawFallbackPlaceholder("Module 06 Ch 2: Recursive Hoye Multiplet Tree");
            return;
          }
          var jStr = jHz.map(function (v) { return Number(v).toFixed(1); }).join(", ");
          drawCardTitle(
            ctx,
            padX,
            padY - 12,
            "SYNTHETIC: Hoye Multiplet Peeling (" + mClass + ", J = " + jStr + " Hz)",
            acsRep,
            pal
          );
          var mHi = Math.max(mPpm[0], mPpm[mPpm.length - 1]);
          var mLo = Math.min(mPpm[0], mPpm[mPpm.length - 1]);
          var mSpan = Math.max(1e-4, mHi - mLo);
          var maxM = 1e-6;
          for (var i = 0; i < mInt.length; i++) maxM = Math.max(maxM, mInt[i]);
          drawAxes(ctx, padX, padY + 8, pw, ph - 12, "1H (ppm, " + mHi.toFixed(3) + " -> " + mLo.toFixed(3) + ")", "Intensity", pal);
          ctx.strokeStyle = pal.cyan;
          ctx.lineWidth = 2.1;
          ctx.beginPath();
          for (var i = 0; i < mPpm.length && i < mInt.length; i++) {
            var sx = padX + ((mHi - mPpm[i]) / mSpan) * pw;
            var sy = padY + ph - 14 - clamp(mInt[i] / (maxM * 1.12), -0.05, 1.0) * (ph - 28);
            if (i === 0) ctx.moveTo(sx, sy);
            else ctx.lineTo(sx, sy);
          }
          ctx.stroke();
          for (var i = 2; i < mInt.length - 2; i++) {
            if (mInt[i] > mInt[i - 1] && mInt[i] > mInt[i + 1] && mInt[i] > maxM * 0.45) {
              var px = padX + ((mHi - mPpm[i]) / mSpan) * pw;
              var py = padY + ph - 14 - clamp(mInt[i] / (maxM * 1.12), 0, 1.0) * (ph - 28);
              ctx.fillStyle = pal.amber;
              ctx.beginPath();
              ctx.arc(px, py, 3.6, 0, Math.PI * 2);
              ctx.fill();
              addSpot(px, py, 12, mClass + " Subpeak " + mPpm[i].toFixed(4) + " ppm", ["J = " + jStr + " Hz", acsRep]);
            }
          }
        } else if (chIdx === 2) {
          var pur = reqNum("qnmr.purity_percent");
          var uK2 = reqNum("qnmr.u_k2_percent");
          var guardOk = reqBool("qnmr.relaxation_guard_passed");
          var anaInt = reqNum("qnmr.analyte_integral");
          var stdInt = reqNum("qnmr.std_integral");
          if (pur === null || uK2 === null || guardOk === null || anaInt === null || stdInt === null) {
            drawFallbackPlaceholder("Module 06 Ch 3: GUM Internal-Standard qNMR");
            return;
          }
          drawCardTitle(
            ctx,
            padX,
            padY - 12,
            "SYNTHETIC: GUM Internal-Standard qNMR Purity (" + pur.toFixed(2) + "% +- " + uK2.toFixed(2) + "%, k=2)",
            "5*T1 relaxation guard passed=" + String(guardOk) + " · Analyte norm=" + anaInt.toFixed(3) + "H · DMS Std=" + stdInt.toFixed(3) + "H",
            pal
          );
          drawAxes(ctx, padX, padY + 8, pw, ph - 12, "Audited qNMR Observable", "Normalized Scale (%)", pal);
          var bars = [
            { lbl: "Molar Purity", val: pur, disp: pur.toFixed(2) + "% +- " + uK2.toFixed(2) + "%", frac: clamp(pur / 100.0, 0, 1), col: pal.green },
            { lbl: "Analyte Integral (2H)", val: anaInt, disp: anaInt.toFixed(3) + " H", frac: clamp(anaInt / 6.5, 0, 1), col: pal.cyan },
            { lbl: "NIST DMS Std (6H)", val: stdInt, disp: stdInt.toFixed(3) + " H", frac: clamp(stdInt / 6.5, 0, 1), col: pal.teal }
          ];
          for (var b = 0; b < bars.length; b++) {
            var bx = padX + (b + 0.5) * (pw / 3) - 28;
            var bh = Math.max(8, bars[b].frac * (ph - 36));
            ctx.fillStyle = bars[b].col;
            ctx.fillRect(bx, padY + ph - 12 - bh, 56, bh);
            ctx.fillStyle = pal.text;
            ctx.font = "600 11.5px ui-monospace, monospace";
            ctx.fillText(bars[b].disp, bx - 8, padY + ph - 18 - bh);
            addSpot(bx + 28, padY + ph - 12 - bh / 2, 24, bars[b].lbl, ["Value: " + bars[b].disp, "5*T1 guard: " + (guardOk ? "PASS" : "WARN")]);
          }
        } else {
          var atoms = reqArr("octet_ibuprofen.atoms", 6);
          var bonds = reqArr("octet_ibuprofen.bonds", 6);
          var h1Links = reqArr("octet_ibuprofen.h1_links", 3);
          var smiles = reqStr("octet_ibuprofen.smiles");
          if (!atoms || !bonds || !h1Links || !smiles) {
            drawFallbackPlaceholder("Module 06 Ch 4: Octet 2D Structure <-> 1H Peak Linking");
            return;
          }
          atomIdx = Math.round(clamp(ctrlVal, 0, atoms.length - 1));
          drawCardTitle(
            ctx,
            padX,
            padY - 12,
            "SYNTHETIC: Octet (S)-Ibuprofen 2D Structure <-> Predicted 1H Shifts (Atom #" + atomIdx + ")",
            "SMILES: " + smiles + " · Click atom or 1H peak to cross-highlight",
            pal
          );
          var halfW = pw * 0.46;
          var axMin = Infinity, axMax = -Infinity, ayMin = Infinity, ayMax = -Infinity;
          for (var a = 0; a < atoms.length; a++) {
            axMin = Math.min(axMin, atoms[a].x); axMax = Math.max(axMax, atoms[a].x);
            ayMin = Math.min(ayMin, atoms[a].y); ayMax = Math.max(ayMax, atoms[a].y);
          }
          var axSpan = Math.max(1e-4, axMax - axMin);
          var aySpan = Math.max(1e-4, ayMax - ayMin);
          function mapAtomPt(at) {
            return {
              x: padX + 14 + ((at.x - axMin) / axSpan) * (halfW - 28),
              y: padY + 20 + ((at.y - ayMin) / aySpan) * (ph - 36)
            };
          }
          var h1ByAtom = {};
          for (var h1 = 0; h1 < h1Links.length; h1++) {
            h1ByAtom[h1Links[h1].atom_idx] = h1Links[h1];
          }
          ctx.strokeStyle = pal.muted;
          for (var b = 0; b < bonds.length; b++) {
            var a0 = atoms[bonds[b].a], a1 = atoms[bonds[b].b];
            if (!a0 || !a1) continue;
            var p0 = mapAtomPt(a0), p1 = mapAtomPt(a1);
            ctx.lineWidth = (bonds[b].order || 1) >= 2 ? 3.0 : 1.8;
            ctx.beginPath();
            ctx.moveTo(p0.x, p0.y);
            ctx.lineTo(p1.x, p1.y);
            ctx.stroke();
          }
          for (var a = 0; a < atoms.length; a++) {
            var ap = mapAtomPt(atoms[a]);
            var isSel = atoms[a].idx === atomIdx;
            ctx.fillStyle = isSel ? pal.amber : pal.cyan;
            ctx.beginPath();
            ctx.arc(ap.x, ap.y, isSel ? 8.5 : 5.0, 0, Math.PI * 2);
            ctx.fill();
            var elTxt = atoms[a].element || "C";
            ctx.fillStyle = pal.text;
            ctx.font = "600 10px ui-monospace, monospace";
            ctx.fillText(elTxt + atoms[a].idx, ap.x + 6, ap.y - 5);
            var lk = h1ByAtom[atoms[a].idx];
            addSpot(ap.x, ap.y, 14, "(S)-Ibuprofen Atom #" + atoms[a].idx + " (" + elTxt + ")", [lk ? ("Linked 1H: " + lk.shift_ppm.toFixed(2) + " ppm (" + lk.multiplicity + ")") : "Quaternary / non-protonated heavy atom"], atoms[a].idx);
          }

          var rx0 = padX + pw * 0.52;
          var rw = pw * 0.48;
          drawAxes(ctx, rx0, padY + 8, rw, ph - 12, "Predicted 1H (ppm, 12 -> 0)", "nH", pal);
          for (var k = 0; k < h1Links.length; k++) {
            var hl = h1Links[k];
            var px = rx0 + clamp((12.0 - hl.shift_ppm) / 12.0, 0.04, 0.96) * rw;
            var matched = hl.atom_idx === atomIdx;
            var hgt = clamp((hl.n_protons || 1) / 3.5, 0.22, 0.92) * (ph - 34);
            var py = padY + ph - 14 - hgt;
            ctx.strokeStyle = matched ? pal.amber : pal.teal;
            ctx.lineWidth = matched ? 3.4 : 2.0;
            ctx.beginPath();
            ctx.moveTo(px, padY + ph - 14);
            ctx.lineTo(px, py);
            ctx.stroke();
            addSpot(px, py, 13, "1H " + hl.shift_ppm.toFixed(2) + " ppm (" + hl.multiplicity + ")", ["Atom #" + hl.atom_idx + " · " + hl.n_protons + "H"], hl.atom_idx);
          }
        }
        return;
      }

      if (modIdx === 7) {
        var gradStep = Math.round(clamp(ctrlVal, 0, 15));
        var grads = reqArr("gradients_g_cm", 8);
        var bVals = reqArr("b_values_s_m2", 8);
        var peaks = reqArr("peaks", 3);
        if (!grads || !bVals || !peaks) {
          drawFallbackPlaceholder("Module 07: DOSY Gradient Stack & Stejskal-Tanner Payload");
          return;
        }
        gradStep = Math.round(clamp(ctrlVal, 0, grads.length - 1));
        if (chIdx === 0) {
          var ppmAxis = reqArr("ppm_axis", 20);
          if (!ppmAxis) {
            drawFallbackPlaceholder("Module 07 Ch 1: BPP-LED Gradient Stack");
            return;
          }
          var pHi = Math.max(ppmAxis[0], ppmAxis[ppmAxis.length - 1]);
          var pLo = Math.min(ppmAxis[0], ppmAxis[ppmAxis.length - 1]);
          var pSpan = Math.max(0.2, pHi - pLo);
          drawCardTitle(
            ctx,
            padX,
            padY - 12,
            "SYNTHETIC: BPP-LED Gradient Stack (g" + (gradStep + 1) + "/" + grads.length + " = " + grads[gradStep].toFixed(1) + " G/cm)",
            "b = " + (bVals[gradStep] * 1e-9).toFixed(3) + "e9 s/m^2 · SINE.100 2/pi shape factor",
            pal
          );
          drawAxes(ctx, padX, padY + 8, pw, ph - 12, "1H (ppm, " + pHi.toFixed(1) + " -> " + pLo.toFixed(1) + ")", "Gradient Slices g1..g" + grads.length, pal);
          for (var g = 0; g < grads.length; g += 2) {
            var baseLine = padY + ph - 18 - (g / grads.length) * (ph * 0.72);
            var isSel = (g === gradStep || g + 1 === gradStep);
            ctx.strokeStyle = isSel ? pal.amber : pal.cyan;
            ctx.lineWidth = isSel ? 2.3 : 1.2;
            ctx.beginPath();
            for (var i = 0; i < ppmAxis.length; i++) {
              var xPpm = ppmAxis[i];
              var ySum = 0;
              for (var pkIdx = 0; pkIdx < peaks.length; pkIdx++) {
                var pk = peaks[pkIdx];
                var amp = (pk.intensities && pk.intensities[g] !== undefined) ? pk.intensities[g] : 0;
                var dx = (xPpm - pk.center_ppm) / 0.05;
                ySum += amp / (1 + dx * dx);
              }
              var sx = padX + ((pHi - xPpm) / pSpan) * pw;
              var sy = baseLine - ySum * 28;
              if (i === 0) ctx.moveTo(sx, sy);
              else ctx.lineTo(sx, sy);
            }
            ctx.stroke();
            addSpot(padX + pw * 0.3, baseLine - 16, 14, "Step g" + (g + 1) + " (" + grads[g].toFixed(1) + " G/cm)", ["b = " + (bVals[g] * 1e-9).toFixed(3) + "e9 s/m^2"], g);
          }
        } else if (chIdx === 1) {
          drawCardTitle(
            ctx,
            padX,
            padY - 12,
            "SYNTHETIC: Stejskal-Tanner Decay & BIC K=1 vs K=2 Selection (g" + (gradStep + 1) + ")",
            peaks.length + " fitted peaks · Overlapped 3.66 ppm selects BIC K=" + (peaks[3] ? peaks[3].selected_k : peaks[0].selected_k),
            pal
          );
          var bMax = Math.max(1e6, bVals[bVals.length - 1]);
          drawAxes(ctx, padX, padY + 8, pw, ph - 12, "b-value (0.." + (bMax * 1e-9).toFixed(2) + "e9 s/m^2)", "Attenuation I(b)", pal);
          var pCols = [pal.amber, pal.cyan, pal.green, pal.coral];
          var showIdxs = [0, 1, 2, Math.min(3, peaks.length - 1)];
          for (var si = 0; si < showIdxs.length; si++) {
            var pk = peaks[showIdxs[si]];
            var col = pCols[si % pCols.length];
            ctx.strokeStyle = col;
            ctx.lineWidth = pk.selected_k > 1 ? 2.5 : 1.7;
            ctx.beginPath();
            for (var g = 0; g < bVals.length && g < pk.intensities.length; g++) {
              var sx = padX + (bVals[g] / bMax) * pw;
              var sy = padY + 14 + (1 - clamp(pk.intensities[g], 0, 1.05)) * (ph - 28);
              if (g === 0) ctx.moveTo(sx, sy);
              else ctx.lineTo(sx, sy);
            }
            ctx.stroke();
            var gx = padX + (bVals[gradStep] / bMax) * pw;
            var gy = padY + 14 + (1 - clamp(pk.intensities[gradStep] || 0, 0, 1.05)) * (ph - 28);
            ctx.fillStyle = col;
            ctx.beginPath();
            ctx.arc(gx, gy, 5, 0, Math.PI * 2);
            ctx.fill();
            addSpot(gx, gy, 13, pk.label + " (BIC K=" + pk.selected_k + ")", ["b = " + (bVals[gradStep] * 1e-9).toFixed(2) + "e9 s/m^2", "I(b) = " + Number(pk.intensities[gradStep] || 0).toFixed(3)], gradStep);
          }
        } else if (chIdx === 2) {
          var iltGrid = reqArr("ilt_overlap.log10_d_grid", 10);
          var iltSpec = reqArr("ilt_overlap.spectrum", 10);
          var iltLam = reqNum("ilt_overlap.selected_lambda");
          var iltModes = reqArr("ilt_overlap.modes", 1);
          if (!iltGrid || !iltSpec || iltLam === null || !iltModes) {
            drawFallbackPlaceholder("Module 07 Ch 3: 2D DOSY & Tikhonov ILT");
            return;
          }
          drawCardTitle(
            ctx,
            padX,
            padY - 12,
            "SYNTHETIC: 2D DOSY (ppm x log10 D) & GCV Tikhonov ILT",
            "3.66 ppm ILT lambda_GCV=" + iltLam.toExponential(2) + " · " + iltModes.length + " continuous modes",
            pal
          );
          var leftW = pw * 0.58;
          drawAxes(ctx, padX, padY + 8, leftW, ph - 12, "1H (ppm, 7.8 -> 0.4)", "log10 D (-8.8 -> -10.3)", pal);
          for (var p = 0; p < peaks.length; p++) {
            var pk = peaks[p];
            var comps = pk.components || [];
            for (var c = 0; c < comps.length; c++) {
              var sx = padX + clamp((7.8 - pk.center_ppm) / 7.4, 0.05, 0.95) * leftW;
              var sy = padY + 14 + clamp((-8.75 - comps[c].log10_d) / 1.6, 0.05, 0.95) * (ph - 28);
              ctx.strokeStyle = pk.selected_k > 1 ? pal.coral : pal.cyan;
              ctx.lineWidth = 2;
              ctx.beginPath();
              ctx.ellipse(sx, sy, 14, 8, 0, 0, Math.PI * 2);
              ctx.stroke();
              addSpot(sx, sy, 16, pk.label + " (" + pk.center_ppm.toFixed(2) + " ppm)", ["log10 D: " + comps[c].log10_d.toFixed(2), "D: " + comps[c].d_m2_s.toExponential(2) + " m^2/s"]);
            }
          }
          var rx0 = padX + pw * 0.63;
          var rw = pw * 0.37;
          drawAxes(ctx, rx0, padY + 8, rw, ph - 12, "log10 D", "ILT f(D)", pal);
          var maxIlt = 1e-9;
          for (var i = 0; i < iltSpec.length; i++) maxIlt = Math.max(maxIlt, iltSpec[i]);
          var gLo = iltGrid[0], gHi = iltGrid[iltGrid.length - 1];
          var gSpan = Math.max(1e-4, gHi - gLo);
          ctx.strokeStyle = pal.green;
          ctx.lineWidth = 2.2;
          ctx.beginPath();
          for (var i = 0; i < iltGrid.length && i < iltSpec.length; i++) {
            var ix = rx0 + ((iltGrid[i] - gLo) / gSpan) * rw;
            var iy = padY + ph - 14 - clamp(iltSpec[i] / (maxIlt * 1.1), 0, 1) * (ph - 28);
            if (i === 0) ctx.moveTo(ix, iy);
            else ctx.lineTo(ix, iy);
          }
          ctx.stroke();
        } else {
          var hydro = reqArr("hydrodynamics", 3);
          if (!hydro) {
            drawFallbackPlaceholder("Module 07 Ch 4: Stokes-Einstein Hydrodynamic Sizing");
            return;
          }
          drawCardTitle(
            ctx,
            padX,
            padY - 12,
            "SYNTHETIC: Evans Stokes-Einstein Hydrodynamic Radius Rh & MW (D2O, 298 K)",
            hydro.map(function (h) { return h.name + " Rh=" + Number(h.r_h_angstrom).toFixed(1) + " A"; }).join(" · "),
            pal
          );
          drawAxes(ctx, padX, padY + 8, pw, ph - 12, "log10 MW (Da, 1.8 -> 4.4)", "Rh (Angstrom)", pal);
          var maxRh = 1.0;
          for (var s = 0; s < hydro.length; s++) maxRh = Math.max(maxRh, hydro[s].r_h_angstrom || 1);
          var hCols = [pal.cyan, pal.green, pal.amber];
          for (var s = 0; s < hydro.length; s++) {
            var hd = hydro[s];
            var logMw = Math.log(Math.max(10, hd.mw_da)) / Math.LN10;
            var sx = padX + clamp((logMw - 1.8) / 2.6, 0.08, 0.90) * pw;
            var sy = padY + ph - 16 - clamp(hd.r_h_angstrom / (maxRh * 1.18), 0.06, 0.94) * (ph - 30);
            ctx.fillStyle = hCols[s % hCols.length];
            ctx.beginPath();
            ctx.arc(sx, sy, 6.5, 0, Math.PI * 2);
            ctx.fill();
            ctx.fillStyle = pal.text;
            ctx.font = "600 12px ui-monospace, monospace";
            ctx.fillText(hd.name + " (Rh=" + hd.r_h_angstrom.toFixed(1) + " A, ~" + Math.round(hd.mw_da) + " Da)", sx + 10, sy - 6);
            addSpot(sx, sy, 18, hd.name + " Stokes-Einstein", ["Rh: " + hd.r_h_angstrom.toFixed(2) + " A", "Estimated MW: " + Math.round(hd.mw_da) + " Da"]);
          }
        }
        return;
      }

      // Module 08: Metabolomics (AI-Binning/PQN, PCA/OPLS-DA, STOCSY, MCP)
      var driverIdx = Math.round(clamp(ctrlVal, 0, 20));
      if (chIdx === 0) {
        var nSamples = reqNum("n_samples");
        var nBuckets = reqNum("n_buckets");
        var sPpm = reqArr("stocsy.ppm", 10);
        var mCtrl = reqArr("stocsy.mean_ctrl", 10);
        var mCase = reqArr("stocsy.mean_case", 10);
        if (nSamples === null || nBuckets === null || !sPpm || !mCtrl || !mCase) {
          drawFallbackPlaceholder("Module 08 Ch 1: Cohort PQN Spectra & AI-Binning");
          return;
        }
        drawCardTitle(
          ctx,
          padX,
          padY - 12,
          "SYNTHETIC: " + nSamples + "-Sample Biofluid Cohort (" + nBuckets + " De Meyer AI Bins + PQN)",
          "Cyan = Mean Control · Coral = Mean Case aligned PQN spectra",
          pal
        );
        var pHi = Math.max(sPpm[0], sPpm[sPpm.length - 1]);
        var pLo = Math.min(sPpm[0], sPpm[sPpm.length - 1]);
        var pSpan = Math.max(0.2, pHi - pLo);
        var maxCoh = 1e-9;
        for (var i = 0; i < mCtrl.length; i++) maxCoh = Math.max(maxCoh, mCtrl[i], mCase[i] || 0);
        drawAxes(ctx, padX, padY + 8, pw, ph - 12, "1H (ppm, " + pHi.toFixed(2) + " -> " + pLo.toFixed(2) + ")", "PQN Mean Intensity", pal);
        var cohortTraces = [
          { col: pal.cyan, arr: mCtrl, lbl: "Mean Control" },
          { col: pal.coral, arr: mCase, lbl: "Mean Case" }
        ];
        for (var c = 0; c < cohortTraces.length; c++) {
          var tr = cohortTraces[c];
          ctx.strokeStyle = tr.col;
          ctx.lineWidth = 1.9;
          ctx.beginPath();
          for (var i = 0; i < sPpm.length && i < tr.arr.length; i++) {
            var sx = padX + ((pHi - sPpm[i]) / pSpan) * pw;
            var sy = padY + ph - 12 - clamp(tr.arr[i] / (maxCoh * 1.1), 0, 1) * (ph - 26);
            if (i === 0) ctx.moveTo(sx, sy);
            else ctx.lineTo(sx, sy);
            if (c === 1 && i % Math.max(1, Math.floor(sPpm.length / 10)) === 0) {
              addSpot(sx, sy, 12, "Cohort Grid " + sPpm[i].toFixed(2) + " ppm", ["Case mean: " + mCase[i].toFixed(3), "Control mean: " + mCtrl[i].toFixed(3)]);
            }
          }
          ctx.stroke();
        }
      } else if (chIdx === 1) {
        var scores = reqArr("pca.scores", 6);
        var ell95 = reqArr("pca.ellipse_95", 8);
        var q2 = reqNum("oplsda.q2");
        var r2y = reqNum("oplsda.r2y");
        var permP = reqNum("oplsda.perm_p_q2");
        if (!scores || !ell95 || q2 === null || r2y === null || permP === null) {
          drawFallbackPlaceholder("Module 08 Ch 2: PCA Hotelling T^2 & OPLS-DA");
          return;
        }
        drawCardTitle(
          ctx,
          padX,
          padY - 12,
          "SYNTHETIC: PCA 95% Hotelling T^2 & OPLS-DA (Q^2=" + q2.toFixed(3) + ", R^2Y=" + r2y.toFixed(3) + ")",
          "100-permutation p=" + permP.toFixed(4) + " · Leak-free stratified k-fold CV",
          pal
        );
        var xBound = 1.0, yBound = 1.0;
        for (var i = 0; i < scores.length; i++) {
          xBound = Math.max(xBound, Math.abs(scores[i].t1 || 0));
          yBound = Math.max(yBound, Math.abs(scores[i].t2 || 0));
        }
        for (var i = 0; i < ell95.length; i++) {
          xBound = Math.max(xBound, Math.abs(ell95[i][0] || 0));
          yBound = Math.max(yBound, Math.abs(ell95[i][1] || 0));
        }
        xBound *= 1.15; yBound *= 1.15;
        drawAxes(ctx, padX, padY + 8, pw, ph - 12, "PC1 t[1]", "PC2 t[2]", pal);
        ctx.strokeStyle = pal.amber;
        ctx.setLineDash([5, 4]);
        ctx.beginPath();
        for (var i = 0; i < ell95.length; i++) {
          var ex = padX + ((ell95[i][0] + xBound) / (2 * xBound)) * pw;
          var ey = padY + ph - 12 - ((ell95[i][1] + yBound) / (2 * yBound)) * (ph - 24);
          if (i === 0) ctx.moveTo(ex, ey);
          else ctx.lineTo(ex, ey);
        }
        ctx.closePath();
        ctx.stroke();
        ctx.setLineDash([]);
        for (var i = 0; i < scores.length; i++) {
          var sc = scores[i];
          var isCase = String(sc.group || "").toLowerCase().indexOf("case") !== -1;
          var sx = padX + ((sc.t1 + xBound) / (2 * xBound)) * pw;
          var sy = padY + ph - 12 - ((sc.t2 + yBound) / (2 * yBound)) * (ph - 24);
          ctx.fillStyle = isCase ? pal.coral : pal.cyan;
          ctx.beginPath();
          ctx.arc(sx, sy, 5, 0, Math.PI * 2);
          ctx.fill();
          addSpot(sx, sy, 12, String(sc.group || "Sample") + " #" + (i + 1), ["t1=" + sc.t1.toFixed(2) + ", t2=" + sc.t2.toFixed(2), "OPLS-DA Q^2=" + q2.toFixed(3)]);
        }
      } else if (chIdx === 2) {
        var sPpm = reqArr("stocsy.ppm", 10);
        var mCase = reqArr("stocsy.mean_case", 10);
        var r2Arr = reqArr("stocsy.r2", 10);
        var drvPpm = reqNum("stocsy.driver_ppm");
        var topMatch = reqStr("stocsy.top_match");
        var sPlot = reqArr("oplsda.s_plot", 3);
        if (!sPpm || !mCase || !r2Arr || drvPpm === null || !topMatch || !sPlot) {
          drawFallbackPlaceholder("Module 08 Ch 3: STOCSY Correlation Trace");
          return;
        }
        var selBin = sPlot[Math.min(driverIdx, sPlot.length - 1)];
        drawCardTitle(
          ctx,
          padX,
          padY - 12,
          "SYNTHETIC: STOCSY Driver " + drvPpm.toFixed(2) + " ppm -> " + topMatch,
          "Inspect S-plot bin #" + driverIdx + " (" + selBin.ppm.toFixed(2) + " ppm, VIP=" + selBin.vip.toFixed(2) + ", |r|=" + Math.abs(selBin.corr).toFixed(2) + ")",
          pal
        );
        var pHi = Math.max(sPpm[0], sPpm[sPpm.length - 1]);
        var pLo = Math.min(sPpm[0], sPpm[sPpm.length - 1]);
        var pSpan = Math.max(0.2, pHi - pLo);
        var maxSpec = 1e-9;
        for (var i = 0; i < mCase.length; i++) maxSpec = Math.max(maxSpec, mCase[i]);
        drawAxes(ctx, padX, padY + 8, pw, ph - 12, "1H (ppm, " + pHi.toFixed(2) + " -> " + pLo.toFixed(2) + ")", "Intensity Colored by STOCSY r^2", pal);
        for (var i = 1; i < sPpm.length && i < mCase.length && i < r2Arr.length; i++) {
          var r2v = r2Arr[i] || 0;
          ctx.strokeStyle = r2v > 0.75 ? pal.coral : (r2v > 0.35 ? pal.amber : pal.cyan);
          ctx.lineWidth = r2v > 0.75 ? 2.6 : 1.6;
          var x0 = padX + ((pHi - sPpm[i - 1]) / pSpan) * pw;
          var y0 = padY + ph - 12 - clamp(mCase[i - 1] / (maxSpec * 1.1), 0, 1) * (ph - 26);
          var x1 = padX + ((pHi - sPpm[i]) / pSpan) * pw;
          var y1 = padY + ph - 12 - clamp(mCase[i] / (maxSpec * 1.1), 0, 1) * (ph - 26);
          ctx.beginPath();
          ctx.moveTo(x0, y0);
          ctx.lineTo(x1, y1);
          ctx.stroke();
          if (r2v > 0.75 && i % 2 === 0) {
            addSpot(x1, y1, 13, "STOCSY Peak " + sPpm[i].toFixed(2) + " ppm", ["r^2 to " + drvPpm.toFixed(2) + " ppm: " + r2v.toFixed(3), topMatch]);
          }
        }
      } else {
        var accMode = reqStr("mcp.access_mode");
        var fwMax = reqNum("mcp.homolog_firewall_max_seq_id");
        var undoCap = reqNum("mcp.undo_capacity");
        var undoEq = reqBool("mcp.undo_bitwise_eq");
        var initResp = getByPath(payload, "mcp.initialize_response");
        var torResp = getByPath(payload, "mcp.torsion_call_response");
        var denResp = getByPath(payload, "mcp.readonly_denied_response");
        if (!accMode || fwMax === null || undoCap === null || undoEq === null || !initResp || !torResp || !denResp) {
          markMissing("mcp.responses");
          drawFallbackPlaceholder("Module 08 Ch 4: MCP JSON-RPC Protocol Showcase");
          return;
        }
        markPlotted("mcp.initialize_response");
        markPlotted("mcp.torsion_call_response");
        markPlotted("mcp.readonly_denied_response");
        drawCardTitle(
          ctx,
          padX,
          padY - 12,
          "ILLUSTRATIVE PROTOCOL DIAGRAM (live engine JSON-RPC in payload): MCP " + accMode + " + Undo(" + undoCap + ")",
          "HomologFirewall <=" + Math.round(fwMax * 100) + "% · Undo bitwise_eq=" + String(undoEq) + " · Click 'Inspect MCP JSON-RPC' below",
          pal
        );
        var steps = [
          ["1. initialize", "protocolVersion 2024-11-05"],
          ["2. tools/call torsions", "Firewall <=" + Math.round(fwMax * 100) + "% OK"],
          ["3. write_curated_peak_tsv", accMode + " guard DENIED"],
          ["4. CommandHistory(" + undoCap + ")", "bitwise_eq=" + String(undoEq)]
        ];
        for (var s = 0; s < steps.length; s++) {
          var bx = padX + (s / 4) * pw + 6;
          var bw = pw / 4 - 14;
          var active = s === Math.floor(localT * 4) % 4;
          ctx.fillStyle = active ? "rgba(167, 139, 250, 0.22)" : pal.panel;
          ctx.strokeStyle = active ? pal.violet : pal.border;
          ctx.lineWidth = active ? 2 : 1.2;
          drawRoundRect(ctx, bx, padY + ph * 0.28, bw, ph * 0.40, 8);
          ctx.fill();
          ctx.stroke();
          ctx.fillStyle = pal.text;
          ctx.font = "600 11px ui-monospace, monospace";
          ctx.fillText(steps[s][0], bx + 8, padY + ph * 0.44, bw - 16);
          ctx.fillStyle = pal.muted;
          ctx.font = "10.5px -apple-system, sans-serif";
          ctx.fillText(steps[s][1], bx + 8, padY + ph * 0.56, bw - 16);
          addSpot(bx + bw / 2, padY + ph * 0.47, 24, steps[s][0], [steps[s][1], "ILLUSTRATIVE protocol box backed by payload.mcp"]);
        }
      }
    }

    renderSceneBody();

    if (state) {
      state.fallbackReasons = fallbackReasons;
      state.plottedPayloadKeys = plottedKeys;
    }
    if (fallbackReasons.length > 0) {
      ctx.save();
      ctx.fillStyle = "rgba(136, 19, 55, 0.94)";
      ctx.fillRect(0, 0, w, 26);
      ctx.fillStyle = "#ffe4e6";
      ctx.font = "700 11px ui-monospace, SFMono-Regular, Menlo, monospace";
      ctx.textAlign = "left";
      ctx.fillText(
        "ILLUSTRATIVE FALLBACK (not engine output): missing payload." + fallbackReasons.join(", "),
        10,
        17
      );
      ctx.restore();
    }
  }

  function initGallery() {
    var dataEl = document.getElementById("tour-gallery-data");
    if (!dataEl) return;
    var items = [];
    try { items = JSON.parse(dataEl.textContent || "[]"); } catch (_) {}
    var canvases = document.querySelectorAll("canvas.mini-canvas");
    if (!canvases.length) return;

    var startTs = performance.now();
    function tick(now) {
      var elapsed = ((now - startTs) * 0.001) % 16.0;
      for (var i = 0; i < canvases.length; i++) {
        var cvs = canvases[i];
        var modIdx = parseInt(cvs.getAttribute("data-module-index") || "1", 10);
        var item = items.find(function (x) { return x.module_index === modIdx; }) || {};
        var defVal = item.interactive_control ? item.interactive_control.default_val : 0;
        var ctx = cvs.getContext("2d");
        if (ctx) {
          var gState = {};
          renderModuleScene(ctx, cvs.width, cvs.height, modIdx, item.payload || {}, (elapsed + modIdx * 2.0) % 16.0, defVal, gState, null);
          cvs._fallbackReasons = gState.fallbackReasons || [];
          cvs._plottedPayloadKeys = gState.plottedPayloadKeys || [];
        }
      }
      requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }

  function initPlayer() {
    var bundleEl = document.getElementById("tour-bundle-data");
    if (!bundleEl) return;
    var bundle = JSON.parse(bundleEl.textContent || "{}");
    var params = new URLSearchParams(window.location.search);

    if (params.get("embed") === "1") {
      document.body.classList.add("is-embed");
    }

    var modeIntBtn = document.getElementById("mode-interactive");
    var modeVidBtn = document.getElementById("mode-video");
    var panelInt = document.getElementById("panel-interactive");
    var panelVid = document.getElementById("panel-video");
    var canvas = document.getElementById("tour-canvas");
    var fallbackBanner = document.getElementById("fallback-banner");
    var tooltip = document.getElementById("hud-tooltip");
    var tooltipTitle = document.getElementById("hud-tooltip-title");
    var tooltipBody = document.getElementById("hud-tooltip-body");
    var btnPlay = document.getElementById("btn-play");
    var timeReadout = document.getElementById("time-readout");
    var timeline = document.getElementById("timeline-range");
    var chMarkers = document.querySelectorAll(".ch-marker");
    var speedBtns = document.querySelectorAll(".speed-btn");
    var btnLoop = document.getElementById("btn-loop");
    var btnExport = document.getElementById("btn-export-webm");
    var exportStatus = document.getElementById("export-status");
    var ctrlSlider = document.getElementById("ctrl-slider");
    var ctrlReadout = document.getElementById("ctrl-readout");
    var ctrlDec = document.getElementById("ctrl-dec");
    var ctrlInc = document.getElementById("ctrl-inc");
    var ctrlResume = document.getElementById("ctrl-resume");
    var presetChips = document.querySelectorAll(".preset-chip");
    var drawerChBadge = document.getElementById("drawer-ch-badge");
    var drawerChTitle = document.getElementById("drawer-ch-title");
    var drawerMetric = document.getElementById("drawer-metric");
    var drawerCaption = document.getElementById("drawer-caption");
    var overlayChLabel = document.getElementById("overlay-ch-label");
    var tourVideo = document.getElementById("tour-video");
    var vidJumpBtns = document.querySelectorAll(".vid-jump-btn");
    var btnCopyCli = document.getElementById("btn-copy-cli");
    var cliCmd = document.getElementById("cli-cmd");

    var duration = bundle.duration_sec || 16.0;
    var state = {
      tSec: 0,
      playing: params.get("autoplay") === "1",
      speed: 1.0,
      loop: true,
      ctrlVal: bundle.interactive_control ? bundle.interactive_control.default_val : 0,
      manualOverride: false,
      pitchRad: 0.35,
      hotspots: [],
      fallbackReasons: [],
      plottedPayloadKeys: [],
      dragging: false,
      lastX: 0,
      lastY: 0,
      lastFrameTs: performance.now()
    };

    var initCh = parseInt(params.get("ch") || "0", 10);
    if (initCh >= 1 && initCh <= (bundle.chapters || []).length) {
      state.tSec = bundle.chapters[initCh - 1].start_sec;
    } else if (params.has("t")) {
      state.tSec = clamp(parseFloat(params.get("t") || "0") || 0, 0, duration);
    }

    function setMode(mode) {
      var isVid = mode === "video";
      panelInt.hidden = isVid;
      panelVid.hidden = !isVid;
      modeIntBtn.classList.toggle("is-active", !isVid);
      modeVidBtn.classList.toggle("is-active", isVid);
      modeIntBtn.setAttribute("aria-selected", String(!isVid));
      modeVidBtn.setAttribute("aria-selected", String(isVid));
      if (isVid) {
        state.playing = false;
        syncTransportUI();
      } else if (tourVideo && !tourVideo.paused) {
        tourVideo.pause();
      }
      resizeCanvas();
    }

    if (params.get("mode") === "video") {
      setMode("video");
    }
    modeIntBtn.addEventListener("click", function () { setMode("interactive"); });
    modeVidBtn.addEventListener("click", function () { setMode("video"); });

    function syncDrawerForTime(t) {
      var idx = activeChapterIndex(bundle, t);
      var ch = (bundle.chapters || [])[idx];
      if (!ch) return;
      drawerChBadge.textContent = "Chapter " + (idx + 1) + " / " + bundle.chapters.length;
      drawerChTitle.textContent = ch.title;
      var metricText = ch.metric_pill || ch.metric || "";
      drawerMetric.textContent = metricText;
      drawerMetric.hidden = !metricText;
      drawerCaption.textContent = ch.caption;
      if (overlayChLabel) overlayChLabel.textContent = "Ch " + (idx + 1) + "/" + bundle.chapters.length;
      for (var i = 0; i < chMarkers.length; i++) {
        chMarkers[i].classList.toggle("is-active", i === idx);
      }
    }

    function syncTransportUI() {
      btnPlay.textContent = state.playing ? "\u275A\u275A Pause" : "\u25B6 Play";
      timeReadout.textContent = formatTime(state.tSec) + " / " + formatTime(duration);
      timeline.value = String(state.tSec);
      syncDrawerForTime(state.tSec);
    }

    function applyControl(val, pausePlayback) {
      var min = parseFloat(ctrlSlider.min);
      var max = parseFloat(ctrlSlider.max);
      var clamped = clamp(val, min, max);
      state.ctrlVal = clamped;
      ctrlSlider.value = String(clamped);
      ctrlReadout.textContent = Number.isInteger(clamped) ? String(clamped) : clamped.toFixed(1);
      if (pausePlayback) {
        state.manualOverride = true;
        state.playing = false;
        if (ctrlResume) ctrlResume.hidden = false;
        syncTransportUI();
      }
    }

    ctrlSlider.addEventListener("input", function () {
      applyControl(parseFloat(ctrlSlider.value), true);
    });
    if (ctrlDec) {
      ctrlDec.addEventListener("click", function () {
        applyControl(state.ctrlVal - parseFloat(ctrlSlider.step || "1"), true);
      });
    }
    if (ctrlInc) {
      ctrlInc.addEventListener("click", function () {
        applyControl(state.ctrlVal + parseFloat(ctrlSlider.step || "1"), true);
      });
    }
    if (ctrlResume) {
      ctrlResume.addEventListener("click", function () {
        state.manualOverride = false;
        state.playing = true;
        ctrlResume.hidden = true;
        syncTransportUI();
      });
    }
    for (var i = 0; i < presetChips.length; i++) {
      presetChips[i].addEventListener("click", function (ev) {
        var v = parseFloat(ev.currentTarget.getAttribute("data-val") || "0");
        applyControl(v, true);
      });
    }

    btnPlay.addEventListener("click", function () {
      state.playing = !state.playing;
      if (state.playing) state.manualOverride = false;
      if (ctrlResume) ctrlResume.hidden = true;
      syncTransportUI();
    });

    timeline.addEventListener("input", function () {
      state.tSec = clamp(parseFloat(timeline.value) || 0, 0, duration);
      syncTransportUI();
    });

    for (var i = 0; i < chMarkers.length; i++) {
      chMarkers[i].addEventListener("click", function (ev) {
        var s = parseFloat(ev.currentTarget.getAttribute("data-start") || "0");
        state.tSec = clamp(s, 0, duration);
        syncTransportUI();
      });
    }

    for (var i = 0; i < speedBtns.length; i++) {
      speedBtns[i].addEventListener("click", function (ev) {
        state.speed = parseFloat(ev.currentTarget.getAttribute("data-speed") || "1");
        for (var j = 0; j < speedBtns.length; j++) {
          speedBtns[j].classList.toggle("is-active", speedBtns[j] === ev.currentTarget);
        }
      });
    }

    btnLoop.addEventListener("click", function () {
      state.loop = !state.loop;
      btnLoop.classList.toggle("is-active", state.loop);
      btnLoop.setAttribute("aria-pressed", String(state.loop));
    });

    document.addEventListener("keydown", function (ev) {
      if (ev.target && (ev.target.tagName === "INPUT" || ev.target.tagName === "TEXTAREA")) return;
      if (ev.code === "Space") {
        ev.preventDefault();
        btnPlay.click();
      } else if (ev.key >= "1" && ev.key <= "4") {
        var idx = parseInt(ev.key, 10) - 1;
        if (bundle.chapters && bundle.chapters[idx]) {
          state.tSec = bundle.chapters[idx].start_sec;
          syncTransportUI();
        }
      }
    });

    for (var i = 0; i < vidJumpBtns.length; i++) {
      vidJumpBtns[i].addEventListener("click", function (ev) {
        var s = parseFloat(ev.currentTarget.getAttribute("data-start") || "0");
        if (tourVideo) {
          tourVideo.currentTime = s;
          tourVideo.play().catch(function () {});
        }
      });
    }
    if (tourVideo) {
      tourVideo.addEventListener("timeupdate", function () {
        var idx = activeChapterIndex(bundle, tourVideo.currentTime || 0);
        for (var i = 0; i < vidJumpBtns.length; i++) {
          vidJumpBtns[i].classList.toggle("is-active", i === idx);
        }
        syncDrawerForTime(tourVideo.currentTime || 0);
      });
    }

    if (btnCopyCli && cliCmd) {
      btnCopyCli.addEventListener("click", function () {
        var text = cliCmd.textContent || "";
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(text).then(function () {
            btnCopyCli.textContent = "Copied!";
            setTimeout(function () { btnCopyCli.textContent = "Copy CLI"; }, 1400);
          }).catch(function () {});
        }
      });
    }

    var mcpBtn = document.getElementById("btn-mcp-toggle");
    var mcpDrawer = document.getElementById("mcp-packet-drawer");
    if (mcpBtn && mcpDrawer) {
      var pkt = (bundle.payload && bundle.payload.mcp) ? bundle.payload.mcp : {
        warning: "ILLUSTRATIVE FALLBACK (not engine output): missing payload.mcp"
      };
      mcpDrawer.textContent = JSON.stringify(pkt, null, 2);
      mcpBtn.addEventListener("click", function () {
        var open = mcpDrawer.hidden;
        mcpDrawer.hidden = !open;
        mcpBtn.setAttribute("aria-expanded", String(open));
      });
    }

    if (btnExport) {
      btnExport.addEventListener("click", function () {
        if (!canvas.captureStream || typeof MediaRecorder === "undefined") {
          exportStatus.textContent = "MediaRecorder unsupported in this browser.";
          return;
        }
        try {
          var mime = MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
            ? "video/webm;codecs=vp9"
            : "video/webm";
          var stream = canvas.captureStream(30);
          var recorder = new MediaRecorder(stream, { mimeType: mime });
          var chunks = [];
          recorder.ondataavailable = function (e) { if (e.data && e.data.size > 0) chunks.push(e.data); };
          recorder.onstop = function () {
            var blob = new Blob(chunks, { type: "video/webm" });
            var url = URL.createObjectURL(blob);
            var a = document.createElement("a");
            a.href = url;
            a.download = "ferric-tour-" + (bundle.slug || "module") + "_clip.webm";
            a.click();
            setTimeout(function () { URL.revokeObjectURL(url); }, 2000);
            exportStatus.textContent = "Clip exported.";
          };
          exportStatus.textContent = "Recording 4s clip...";
          state.playing = true;
          recorder.start();
          setTimeout(function () {
            if (recorder.state === "recording") recorder.stop();
          }, 4000);
        } catch (err) {
          exportStatus.textContent = "Export unavailable: " + err.message;
        }
      });
    }

    function showTooltip(cx, cy, spot) {
      if (!tooltip) return;
      var titleText = String(spot.title || "");
      var bodyText = Array.isArray(spot.lines) ? spot.lines.join("\n") : "";
      if (tooltipTitle && tooltipBody) {
        tooltipTitle.textContent = titleText;
        tooltipBody.textContent = bodyText;
      } else {
        tooltip.textContent = bodyText ? (titleText + "\n" + bodyText) : titleText;
      }
      tooltip.hidden = false;
      var rect = canvas.getBoundingClientRect();
      var tx = clamp(cx + 14, 8, Math.max(8, rect.width - 240));
      var ty = clamp(cy - 18, 8, Math.max(8, rect.height - 80));
      tooltip.style.left = tx + "px";
      tooltip.style.top = ty + "px";
    }

    function hideTooltip() {
      if (tooltip) tooltip.hidden = true;
    }

    function inspectPointer(clientX, clientY, commitSelection) {
      var rect = canvas.getBoundingClientRect();
      var px = clientX - rect.left;
      var py = clientY - rect.top;
      var best = null;
      var bestDist = Infinity;
      for (var i = 0; i < state.hotspots.length; i++) {
        var h = state.hotspots[i];
        var d = Math.hypot(px - h.x, py - h.y);
        var hitR = Math.max(h.r + 8, 20);
        if (d <= hitR && d < bestDist) {
          bestDist = d;
          best = h;
        }
      }
      if (best) {
        showTooltip(px, py, best);
        if (commitSelection && best.val !== undefined) {
          applyControl(best.val, true);
        }
      } else {
        hideTooltip();
      }
    }

    canvas.addEventListener("pointerdown", function (ev) {
      state.dragging = true;
      state.lastX = ev.clientX;
      state.lastY = ev.clientY;
      inspectPointer(ev.clientX, ev.clientY, true);
    });
    canvas.addEventListener("pointermove", function (ev) {
      if (state.dragging && bundle.module_index === 3) {
        var dx = ev.clientX - state.lastX;
        var dy = ev.clientY - state.lastY;
        state.lastX = ev.clientX;
        state.lastY = ev.clientY;
        var nextYaw = (state.ctrlVal + dx * 0.7 + 360) % 360;
        state.pitchRad = clamp(state.pitchRad + dy * 0.01, -1.2, 1.2);
        applyControl(nextYaw, true);
      } else {
        inspectPointer(ev.clientX, ev.clientY, false);
      }
    });
    window.addEventListener("pointerup", function () { state.dragging = false; });
    canvas.addEventListener("pointerleave", hideTooltip);

    var cssW = 800, cssH = 450;
    function resizeCanvas() {
      var wrap = document.getElementById("canvas-wrap");
      if (!wrap) return;
      var rect = wrap.getBoundingClientRect();
      cssW = Math.max(240, rect.width);
      cssH = Math.max(180, rect.height);
      var dpr = window.devicePixelRatio || 1;
      canvas.width = Math.round(cssW * dpr);
      canvas.height = Math.round(cssH * dpr);
    }
    if (typeof ResizeObserver !== "undefined") {
      new ResizeObserver(resizeCanvas).observe(document.getElementById("canvas-wrap"));
    }
    window.addEventListener("resize", resizeCanvas);
    resizeCanvas();
    syncTransportUI();

    function syncFallbackBanner() {
      if (!fallbackBanner) return;
      if (state.fallbackReasons && state.fallbackReasons.length > 0) {
        fallbackBanner.hidden = false;
        fallbackBanner.textContent = "ILLUSTRATIVE FALLBACK (not engine output): missing payload." + state.fallbackReasons.join(", ");
      } else {
        fallbackBanner.hidden = true;
        fallbackBanner.textContent = "";
      }
    }

    function frameLoop(now) {
      var dt = Math.min(0.1, (now - state.lastFrameTs) * 0.001);
      state.lastFrameTs = now;
      if (state.playing && !panelInt.hidden) {
        state.tSec += dt * state.speed;
        if (state.tSec >= duration) {
          if (state.loop) state.tSec = state.tSec % duration;
          else {
            state.tSec = duration;
            state.playing = false;
          }
        }
        if (bundle.module_index === 3 && !state.manualOverride && state.tSec >= 12.0) {
          applyControl(((state.tSec - 12.0) / 4.0) * 360.0, false);
        }
        syncTransportUI();
      }
      var ctx = canvas.getContext("2d");
      if (ctx && !panelInt.hidden) {
        var dpr = window.devicePixelRatio || 1;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        state.hotspots = [];
        renderModuleScene(ctx, cssW, cssH, bundle.module_index, bundle.payload || {}, state.tSec, state.ctrlVal, state, state.hotspots);
        canvas._lastHotspots = state.hotspots;
        canvas._fallbackReasons = state.fallbackReasons;
        canvas._plottedPayloadKeys = state.plottedPayloadKeys;
        syncFallbackBanner();
      }
      requestAnimationFrame(frameLoop);
    }
    requestAnimationFrame(frameLoop);
  }

  document.addEventListener("DOMContentLoaded", function () {
    syncThemeBtn();
    var themeBtn = document.getElementById("theme-toggle");
    if (themeBtn) {
      themeBtn.addEventListener("click", function () {
        var next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
        root.setAttribute("data-theme", next);
        try { localStorage.setItem("nmr_manual_theme", next); } catch (_) {}
        syncThemeBtn();
      });
    }
    initGallery();
    initPlayer();
  });
})();
"""


def build_tour_site_files(
    bundles: Sequence[NormalizedBundle],
    videos_rel: str = "videos",
) -> Dict[str, bytes]:
    """Generate all deterministic static files for the tour showcase site."""
    files: Dict[str, bytes] = {
        "tours.css": TOURS_CSS.encode("utf-8"),
        "tours_player.js": TOURS_PLAYER_JS.encode("utf-8"),
        "index.html": render_gallery_html(bundles).encode("utf-8"),
    }
    for b in bundles:
        files[f"{b.slug}.html"] = render_module_html(b, bundles, videos_rel).encode("utf-8")
    return files


def write_site_to_disk(out_dir: Path, site_files: Dict[str, bytes]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for rel_path, content in sorted(site_files.items()):
        dest = out_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)


def site_digest(site_files: Dict[str, bytes]) -> str:
    h = hashlib.sha256()
    for rel_path in sorted(site_files):
        h.update(rel_path.encode("utf-8"))
        h.update(b"\0")
        h.update(site_files[rel_path])
        h.update(b"\0")
    return h.hexdigest()


def validate_js_syntax(js_text: str) -> List[str]:
    """Check balanced delimiters and run ``node --check`` if ``node`` is available."""
    errors: List[str] = []
    # Python structural balance check (ignoring strings & comments)
    stack: List[str] = []
    pairs = {"(": ")", "[": "]", "{": "}"}
    closing = {")": "(", "]": "[", "}": "{"}
    i = 0
    n = len(js_text)
    in_str: Optional[str] = None
    in_line_comment = False
    in_block_comment = False
    while i < n:
        ch = js_text[i]
        nxt = js_text[i + 1] if i + 1 < n else ""
        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
        elif in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 1
        elif in_str is not None:
            if ch == "\\":
                i += 1
            elif ch == in_str:
                in_str = None
        else:
            if ch == "/" and nxt == "/":
                in_line_comment = True
                i += 1
            elif ch == "/" and nxt == "*":
                in_block_comment = True
                i += 1
            elif ch in ('"', "'", "`"):
                in_str = ch
            elif ch in pairs:
                stack.append(ch)
            elif ch in closing:
                if not stack or stack[-1] != closing[ch]:
                    errors.append(f"unbalanced JS delimiter {ch!r} at offset {i}")
                    break
                stack.pop()
        i += 1
    if stack:
        errors.append(f"unclosed JS delimiters at EOF: {stack}")

    node_bin = shutil.which("node")
    if node_bin:
        with tempfile.NamedTemporaryFile("w", suffix=".js", encoding="utf-8", delete=True) as tf:
            tf.write(js_text)
            tf.flush()
            proc = subprocess.run(
                [node_bin, "--check", tf.name],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                errors.append(f"node --check failed: {proc.stderr.strip()}")
    return errors


REQUIRED_MODULE_PAYLOAD_KEYS: Dict[int, Tuple[str, ...]] = {
    1: (
        "pulprog_name",
        "aqseq",
        "f1_quad",
        "f2_quad",
        "transfer_summary",
        "fid_real",
        "sine_window",
        "fft_unphased",
        "fft_phased",
        "p0_deg",
        "p1_deg",
        "nus_mask",
        "nus_dirty_spec",
        "nus_fista_iters",
        "nus_residual_l1",
    ),
    2: (
        "hsqc_pos_contours",
        "hsqc_neg_contours",
        "hsqc_roots",
        "hncacb_pulse_regime",
        "strips",
        "cv_temps",
        "cv_vals",
        "mean_energies",
        "t_crit",
        "max_cv",
        "evo_acc_pct",
        "std_acc_pct",
        "bayes_acc_pct",
        "res_confidences",
        "res_labels",
        "qc_verdict",
        "n_high_conf",
        "benchmarks.ubiquitin_realistic_pct",
        "benchmarks.cisd3_acc_pct",
        "benchmarks.cisd3_hc_prec_pct",
        "benchmarks.cold_bmrb_hc_min_pct",
        "benchmarks.cold_bmrb_hc_max_pct",
        "benchmarks.cold_bmrb_gate_pct",
        "benchmarks.cold_bmrb_withdrawal_ref",
        "benchmarks.dev_target",
        "benchmarks.ubiquitin_hc_correct",
        "benchmarks.ubiquitin_hc_total",
        "benchmarks.cisd3_hc_correct",
        "benchmarks.cisd3_hc_total",
        "n_res_total",
        "hc_conf_threshold",
        "strip_walk.start_label",
        "strip_walk.end_label",
    ),
    3: (
        "ca_coords",
        "ss_classes",
        "s2_vals",
        "noe_pairs",
        "rdc_points",
        "q_work",
        "q_free",
        "da_nh_hz",
        "rhombicity",
        "f8_holdout_note",
    ),
    4: (
        "snr_levels",
        "locked_ls_bias_pct",
        "local_max_bias_pct",
        "times_r1",
        "curve_600_r1",
        "curve_800_r1",
        "fit_600_rate",
        "fit_600_se",
        "fit_800_rate",
        "fit_800_se",
        "tau_m_ns",
        "d_par_perp",
        "seq_ids",
        "r2_over_r1",
        "s2_vals",
        "rex_vals",
        "model_ids",
        "last_seq_id",
        "field_a_mhz",
        "field_b_mhz",
        "fit_mc_replicates",
        "decay_demo_label",
        "sim_tail_start_seq",
        "rex_report_threshold_s_inv",
        "rex_clause",
        "validation_status.blind_gates_passed",
        "validation_status.blind_gates_total",
    ),
    5: (
        "cest.offsets_ppm",
        "cest.b1_15hz_obs",
        "cest.b1_15hz_fit",
        "cest.b1_30hz_obs",
        "cest.b1_30hz_fit",
        "cest.p_b",
        "cest.k_ex_s_inv",
        "cest.major_ppm",
        "cest.dw_ab_ppm",
        "cpmg.nu_hz",
        "cpmg.r2_600_obs",
        "cpmg.r2_800_obs",
        "cpmg.p_b",
        "cpmg.k_ex_s_inv",
        "cpmg.delta_aicc",
        "hsqc_trajectories",
        "csp_isotherm.protein_conc_um",
        "csp_isotherm.fitted_kd_um",
        "csp_isotherm.naive_mm_kd_um",
        "csp_isotherm.ddelta_max_ppm",
        "csp_isotherm.observed_points",
    ),
    6: (
        "fid_processing.ppm",
        "fid_processing.with_water",
        "fid_processing.hsvd_flattened",
        "fid_processing.baseline_roll",
        "fid_processing.voigt_peaks",
        "multiplet.ppm",
        "multiplet.intensity",
        "multiplet.class",
        "multiplet.j_hz",
        "multiplet.acs_report",
        "qnmr.purity_percent",
        "qnmr.u_k2_percent",
        "qnmr.relaxation_guard_passed",
        "qnmr.analyte_integral",
        "qnmr.std_integral",
        "octet_ibuprofen.smiles",
        "octet_ibuprofen.atoms",
        "octet_ibuprofen.bonds",
        "octet_ibuprofen.h1_links",
    ),
    7: (
        "gradients_g_cm",
        "b_values_s_m2",
        "ppm_axis",
        "peaks",
        "ilt_overlap.log10_d_grid",
        "ilt_overlap.spectrum",
        "ilt_overlap.selected_lambda",
        "ilt_overlap.modes",
        "hydrodynamics",
    ),
    8: (
        "n_samples",
        "n_buckets",
        "pca.scores",
        "pca.ellipse_95",
        "oplsda.q2",
        "oplsda.r2y",
        "oplsda.perm_p_q2",
        "oplsda.s_plot",
        "stocsy.driver_ppm",
        "stocsy.top_match",
        "stocsy.ppm",
        "stocsy.mean_ctrl",
        "stocsy.mean_case",
        "stocsy.r2",
        "mcp.access_mode",
        "mcp.homolog_firewall_max_seq_id",
        "mcp.undo_capacity",
        "mcp.undo_bitwise_eq",
        "mcp.initialize_response",
        "mcp.torsion_call_response",
        "mcp.readonly_denied_response",
    ),
}


def _lookup_dot_path(obj: Any, dot_path: str) -> Tuple[bool, Any]:
    cur = obj
    for part in dot_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False, None
        cur = cur[part]
    return True, cur


NODE_VM_CHECK_HARNESS = r"""
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const outDir = process.argv[1];
const expectFallback = process.argv[2] === "expect-fallback";
// Optional "dump-text=<path>": write every canvas fillText string per "<file>#<ch>" as JSON (B15 self-test).
const dumpArg = process.argv.slice(2).find(a => a.startsWith("dump-text="));
const dumpPath = dumpArg ? dumpArg.slice("dump-text=".length) : null;
const textDump = {};
const jsCode = fs.readFileSync(path.join(outDir, "tours_player.js"), "utf8");
const htmlFiles = fs.readdirSync(outDir).filter(f => /^\d{2}_.*\.html$/.test(f)).sort();
if (htmlFiles.length !== 8) {
  throw new Error("Expected 8 module HTML files, got " + htmlFiles.length);
}

let pairs = 0;
let fallbackTriggers = 0;
for (const file of htmlFiles) {
  const html = fs.readFileSync(path.join(outDir, file), "utf8");
  const m = html.match(/<script[^>]*id="tour-bundle-data"[^>]*>([\s\S]*?)<\/script>/);
  if (!m) throw new Error("Missing tour-bundle-data in " + file);
  const jsonText = m[1].replace(/\\u003c/gi, "<");
  const parsed = JSON.parse(jsonText);

  for (const ch of [1, 2, 3, 4]) {
    let drawCalls = 0;
    const texts = [];
    textDump[file + "#" + ch] = texts;
    const ctx = {
      _fillStyle: "#000",
      _strokeStyle: "#000",
      get fillStyle() { return this._fillStyle; },
      set fillStyle(v) {
        if (v === undefined || v === null) throw new Error("undefined fillStyle in " + file + " ch=" + ch);
        this._fillStyle = v;
      },
      get strokeStyle() { return this._strokeStyle; },
      set strokeStyle(v) {
        if (v === undefined || v === null) throw new Error("undefined strokeStyle in " + file + " ch=" + ch);
        this._strokeStyle = v;
      },
      lineWidth: 1, font: "12px sans-serif", textAlign: "left", textBaseline: "alphabetic", globalAlpha: 1,
      setTransform() {}, save() {}, restore() {}, translate() {}, rotate() {}, scale() {},
      beginPath() {}, closePath() {}, moveTo() {}, lineTo() {}, arc() {}, arcTo() {}, ellipse() {},
      rect() {}, roundRect() {}, clip() {}, setLineDash() {}, clearRect() {},
      quadraticCurveTo() {}, bezierCurveTo() {},
      fillRect() { drawCalls++; },
      strokeRect() { drawCalls++; },
      fill() { drawCalls++; },
      stroke() { drawCalls++; },
      fillText(t) { drawCalls++; texts.push(String(t)); }
    };

    const makeEl = (id) => ({
      id,
      textContent: "",
      value: "0",
      min: "0",
      max: "20",
      step: "1",
      hidden: id === "fallback-banner",
      paused: true,
      width: 960,
      height: 540,
      style: {},
      classList: { toggle() {}, add() {}, remove() {} },
      setAttribute() {},
      getAttribute() { return ""; },
      addEventListener() {},
      getBoundingClientRect() { return { width: 960, height: 540, left: 0, top: 0 }; },
      getContext(kind) { return kind === "2d" ? ctx : null; }
    });

    const elements = {};
    const domListeners = [];
    const doc = {
      documentElement: {
        _attrs: {},
        setAttribute(k, v) { this._attrs[k] = String(v); },
        getAttribute(k) { return this._attrs[k] || "dark"; }
      },
      body: { classList: { add() {}, remove() {}, toggle() {} } },
      getElementById(id) {
        if (!elements[id]) elements[id] = makeEl(id);
        if (id === "tour-bundle-data") elements[id].textContent = jsonText;
        return elements[id];
      },
      querySelectorAll() { return []; },
      addEventListener(ev, fn) { if (ev === "DOMContentLoaded") domListeners.push(fn); }
    };

    let rafCount = 0;
    const sandbox = {
      console,
      Math,
      Number,
      String,
      Boolean,
      Array,
      Object,
      JSON,
      parseInt,
      parseFloat,
      isNaN,
      isFinite,
      URLSearchParams,
      document: doc,
      localStorage: { getItem() { return null; }, setItem() {} },
      navigator: { clipboard: { writeText() { return Promise.resolve(); } } },
      performance: { now() { return 1000; } },
      requestAnimationFrame(fn) {
        rafCount++;
        if (rafCount <= 1) fn(1016);
      },
      window: null
    };
    sandbox.window = {
      location: { search: "?ch=" + ch },
      devicePixelRatio: 1,
      matchMedia() { return { matches: true, addEventListener() {} }; },
      addEventListener() {}
    };

    vm.runInNewContext(jsCode, sandbox, { filename: "tours_player.js" });
    for (const fn of domListeners) fn();

    const cvs = elements["tour-canvas"];
    const fbReasons = (cvs && cvs._fallbackReasons) || [];
    const plottedKeys = (cvs && cvs._plottedPayloadKeys) || [];
    const fbEl = elements["fallback-banner"];

    if (fbReasons.length > 0) {
      fallbackTriggers++;
      if (!fbEl || fbEl.hidden || !fbEl.textContent.includes("ILLUSTRATIVE FALLBACK")) {
        throw new Error(file + " ?ch=" + ch + " fallback fired (" + fbReasons.join(",") + ") without visible #fallback-banner");
      }
      if (!expectFallback) {
        throw new Error(file + " ?ch=" + ch + " triggered payload fallback: " + fbReasons.join(", "));
      }
      continue;
    }

    if (drawCalls < 15) {
      throw new Error(file + " ?ch=" + ch + " drew only " + drawCalls + " canvas calls (< 15)");
    }
    if (plottedKeys.length < 1) {
      throw new Error(file + " ?ch=" + ch + " did not record any plotted payload keys");
    }
    if (fbEl && !fbEl.hidden) {
      throw new Error(file + " ?ch=" + ch + " left #fallback-banner visible despite 0 fallbacks");
    }
    const chObj = (parsed.chapters || [])[ch - 1];
    if (chObj && (chObj.metric_pill || chObj.metric)) {
      const mEl = elements["drawer-metric"];
      if (!mEl || mEl.hidden || !mEl.textContent) {
        throw new Error(file + " ?ch=" + ch + " failed to populate drawer-metric");
      }
    }
    if (file.startsWith("02_") && ch === 2) {
      const spots = (cvs && cvs._lastHotspots) || [];
      for (const rName of ["Phe4", "Val5", "Lys6", "Thr7", "Leu8", "Thr9"]) {
        const nPeaks = spots.filter(s => s.title.startsWith(rName + " C")).length;
        if (nPeaks !== 4) {
          throw new Error("Expected 4 HNCACB peak hotspots on " + rName + ", got " + nPeaks);
        }
      }
      const glySpots = spots.filter(s => s.title.startsWith("Gly10 C"));
      if (glySpots.length !== 3) {
        throw new Error("Expected 3 HNCACB peak hotspots on Gly10, got " + glySpots.length);
      }
      const glyCa = glySpots.find(s => s.title.startsWith("Gly10 CA(i):"));
      if (!glyCa || !glyCa.title.includes("+1") || !(glyCa.lines || []).join(" ").includes("Non-CT HNCACB")) {
        throw new Error("Gly10 CA(i) hotspot missing +1 Non-CT HNCACB sign disclosure");
      }
    }
    if (file.startsWith("02_") && ch === 3) {
      const reqBmKeys = [
        "evo_acc_pct",
        "bayes_acc_pct",
        "std_acc_pct",
        "benchmarks.ubiquitin_realistic_pct",
        "benchmarks.cisd3_acc_pct",
        "benchmarks.cisd3_hc_prec_pct",
        "benchmarks.cold_bmrb_hc_min_pct",
        "benchmarks.cold_bmrb_hc_max_pct"
      ];
      for (const rk of reqBmKeys) {
        if (!plottedKeys.includes(rk)) {
          throw new Error("Module 02 Ch 3 player did not read payload." + rk + " (plotted=" + plottedKeys.join(",") + ")");
        }
      }
    }
    if (file.startsWith("04_") && ch === 3) {
      if (!plottedKeys.includes("tau_m_ns") || !plottedKeys.includes("d_par_perp")) {
        throw new Error("Module 04 Ch 3 player did not read payload.tau_m_ns / d_par_perp");
      }
    }
    pairs++;
  }
}
if (expectFallback && fallbackTriggers === 0) {
  throw new Error("Expected at least one fallback trigger in adversarial mode, got 0");
}
if (dumpPath) {
  fs.writeFileSync(dumpPath, JSON.stringify(textDump));
}
console.log("NODE_VM_OK pairs=" + pairs + " fallbacks=" + fallbackTriggers);
"""


def validate_tour_site(
    site_files: Dict[str, bytes],
    bundles: Sequence[NormalizedBundle],
    videos_rel: str = "videos",
) -> List[str]:
    """Validate offline CSP, WCAG 1.4.4 viewport, payload completeness, zero innerHTML, clean-room hygiene, and Node vm execution."""
    errors: List[str] = []
    expected_names = {"index.html", "tours.css", "tours_player.js"} | {
        f"{b.slug}.html" for b in bundles
    }
    missing = sorted(expected_names - set(site_files.keys()))
    if missing:
        errors.append(f"missing expected output files: {missing}")

    cdn_re = re.compile(r'<(?:script|link|img|source)\b[^>]+(?:src|href)\s*=\s*["\']https?://', re.I)
    forbidden_vp_re = re.compile(r"user-scalable\s*=\s*no|maximum-scale\s*=\s*1(\.0)?", re.I)
    v_rel = videos_rel.rstrip("/")

    for name, raw in sorted(site_files.items()):
        text = raw.decode("utf-8")
        # Strip standard WAI-ARIA HTML/JS attribute names (aria-label, aria-hidden, etc.)
        # before clean-room product token checks so ARIA accessibility attributes do not false-trigger.
        sanitized_low = re.sub(r"\baria-[a-z]+\b", " ", text.lower())
        for tok in RESTRICTED_PRODUCT_TOKENS:
            if re.search(rf"(?<![a-z0-9]){re.escape(tok)}(?![a-z0-9])", sanitized_low):
                errors.append(f"{name}: contains forbidden NMR product token {tok!r}")

        if name.endswith(".html"):
            if cdn_re.search(text):
                errors.append(f"{name}: external http:// or https:// CDN resource reference detected")
            if 'content="width=device-width, initial-scale=1.0, viewport-fit=cover"' not in text:
                errors.append(f"{name}: missing WCAG 1.4.4 viewport-fit=cover meta")
            if forbidden_vp_re.search(text):
                errors.append(f"{name}: forbidden zoom-disabling viewport attribute detected")

    for b in bundles:
        page_name = f"{b.slug}.html"
        if page_name in site_files:
            page_text = site_files[page_name].decode("utf-8")
            if b.module_index >= 4 and "SYNTHETIC" not in page_text:
                errors.append(f"{page_name}: missing explicit SYNTHETIC showcase disclosure text")
            if 'id="fallback-banner"' not in page_text:
                errors.append(f"{page_name}: missing #fallback-banner warning element")
            if "<video" not in page_text:
                errors.append(f"{page_name}: missing <video> element")
            if f'src="{v_rel}/{b.slug}.webm"' not in page_text:
                errors.append(f"{page_name}: missing .webm <source> tag")
            if f'src="{v_rel}/{b.slug}.mp4"' not in page_text:
                errors.append(f"{page_name}: missing .mp4 <source> tag")

        payload = b.raw_json.get("payload")
        req_keys = REQUIRED_MODULE_PAYLOAD_KEYS.get(b.module_index, ())
        if not isinstance(payload, dict):
            errors.append(f"{b.slug}.tour.json: missing object payload")
        else:
            for dot_key in req_keys:
                found, val = _lookup_dot_path(payload, dot_key)
                if not found or val is None:
                    errors.append(f"{b.slug}.tour.json: missing required payload.{dot_key}")
                elif isinstance(val, (list, str, dict)) and len(val) == 0:
                    errors.append(f"{b.slug}.tour.json: empty required payload.{dot_key}")

            # B15: caption-vs-payload scalar consistency checks for Module 02 & Module 04
            if b.module_index == 2:
                bm = payload.get("benchmarks")
                if isinstance(bm, dict):
                    try:
                        ubq_s = f"{float(bm['ubiquitin_realistic_pct']):.1f}%"
                        cisd3_s = f"{float(bm['cisd3_acc_pct']):.2f}%"
                        cold_rng = f"{float(bm['cold_bmrb_hc_min_pct']):.2f}-{float(bm['cold_bmrb_hc_max_pct']):.2f}%"
                    except (KeyError, TypeError, ValueError):
                        ubq_s = cisd3_s = cold_rng = ""
                    if cold_rng:
                        ch3_cap = str(b.chapters[2].get("caption", "")) if len(b.chapters) >= 3 else ""
                        for field_lbl, text in (
                            ("subtitle", b.subtitle),
                            ("validation_badge.detail", b.badge_detail),
                            ("chapters[2].caption", ch3_cap),
                        ):
                            if cold_rng not in text:
                                errors.append(
                                    f"{b.slug}.tour.json: {field_lbl} does not match payload.benchmarks cold range {cold_rng}"
                                )
                            if ubq_s and ubq_s not in text:
                                errors.append(
                                    f"{b.slug}.tour.json: {field_lbl} does not match payload.benchmarks ubiquitin_realistic_pct {ubq_s}"
                                )
                            if cisd3_s and cisd3_s not in text:
                                errors.append(
                                    f"{b.slug}.tour.json: {field_lbl} does not match payload.benchmarks cisd3_acc_pct {cisd3_s}"
                                )
                    try:
                        u_cnt = f"({int(bm['ubiquitin_hc_correct'])}/{int(bm['ubiquitin_hc_total'])})"
                        c_cnt = f"({int(bm['cisd3_hc_correct'])}/{int(bm['cisd3_hc_total'])})"
                        gate_s = f">={float(bm['cold_bmrb_gate_pct']):.0f}%"
                        gate_s2 = f">= {float(bm['cold_bmrb_gate_pct']):.0f}%"
                    except (KeyError, TypeError, ValueError):
                        u_cnt = c_cnt = gate_s = gate_s2 = ""
                    if u_cnt:
                        for cnt_s in (u_cnt, c_cnt):
                            if cnt_s not in b.badge_detail:
                                errors.append(
                                    f"{b.slug}.tour.json: validation_badge.detail does not quote payload.benchmarks HC counts {cnt_s}"
                                )
                        ch3_cap = str(b.chapters[2].get("caption", "")) if len(b.chapters) >= 3 else ""
                        for field_lbl, text in (("subtitle", b.subtitle), ("chapters[2].caption", ch3_cap)):
                            if gate_s not in text:
                                errors.append(
                                    f"{b.slug}.tour.json: {field_lbl} does not match payload.benchmarks cold_bmrb_gate_pct {gate_s}"
                                )
                        if gate_s2 not in b.badge_detail:
                            errors.append(
                                f"{b.slug}.tour.json: validation_badge.detail does not match payload.benchmarks cold_bmrb_gate_pct {gate_s2}"
                            )
                strips = payload.get("strips")
                if isinstance(strips, list) and len(strips) >= 7 and len(b.chapters) >= 2:
                    thr7 = strips[6]
                    if isinstance(thr7, dict) and isinstance(thr7.get("cb_i"), (int, float)):
                        thr7_cb_str = f"Thr7 CB={float(thr7['cb_i']):.1f}"
                        ch2_cap = str(b.chapters[1].get("caption", ""))
                        if "Thr7 CB=" in ch2_cap and thr7_cb_str not in ch2_cap:
                            errors.append(
                                f"{b.slug}.tour.json: chapters[1].caption Thr7 CB does not match payload.strips[6].cb_i ({thr7_cb_str})"
                            )
            elif b.module_index == 4:
                tau_m = payload.get("tau_m_ns")
                if isinstance(tau_m, (int, float)) and len(b.chapters) >= 3:
                    expected_tau = f"tau_m={float(tau_m):.2f} ns"
                    ch3_cap = str(b.chapters[2].get("caption", ""))
                    ch3_pill = str(b.chapters[2].get("metric_pill", "") or b.chapters[2].get("metric", ""))
                    if expected_tau not in ch3_cap:
                        errors.append(
                            f"{b.slug}.tour.json: chapters[2].caption missing payload-matched '{expected_tau}' (got: {ch3_cap!r})"
                        )
                    if expected_tau not in ch3_pill:
                        errors.append(
                            f"{b.slug}.tour.json: chapters[2].metric_pill missing payload-matched '{expected_tau}' (got: {ch3_pill!r})"
                        )
                # Ch4 R_ex claim must name exactly the fitted payload.rex_sites (label + selected model).
                rex_sites = payload.get("rex_sites")
                rex_clause = payload.get("rex_clause")
                if len(b.chapters) >= 4 and isinstance(rex_sites, list):
                    ch4_cap = str(b.chapters[3].get("caption", ""))
                    if isinstance(rex_clause, str) and rex_clause not in ch4_cap:
                        errors.append(
                            f"{b.slug}.tour.json: chapters[3].caption missing payload.rex_clause {rex_clause!r} (got: {ch4_cap!r})"
                        )
                    for site in rex_sites:
                        if isinstance(site, dict):
                            want = f"{site.get('label')} ({site.get('model')})"
                            if want not in ch4_cap:
                                errors.append(
                                    f"{b.slug}.tour.json: chapters[3].caption missing fitted R_ex site {want!r}"
                                )
                    for mdl in re.findall(r"\bM[1-5]\b", re.sub(r"M1-M5", " ", ch4_cap)):
                        if not any(isinstance(x, dict) and x.get("model") == mdl for x in rex_sites):
                            errors.append(
                                f"{b.slug}.tour.json: chapters[3].caption quotes model {mdl} not selected in payload.rex_sites"
                            )
                vs = payload.get("validation_status")
                if isinstance(vs, dict) and "blind_gates_passed" in vs and "blind_gates_total" in vs:
                    want_g = f"{int(vs['blind_gates_passed'])}/{int(vs['blind_gates_total'])}"
                    if want_g not in b.badge_detail:
                        errors.append(
                            f"{b.slug}.tour.json: validation_badge.detail does not quote payload.validation_status gates {want_g}"
                        )

    if "tours_player.js" in site_files:
        js_text = site_files["tours_player.js"].decode("utf-8")
        if ".innerHTML" in js_text:
            errors.append("tours_player.js: forbidden .innerHTML assignment detected (use .textContent)")
        errors.extend(validate_js_syntax(js_text))

    node_bin = shutil.which("node")
    if node_bin is not None and not errors:
        with tempfile.TemporaryDirectory() as tmp_vm:
            vm_dir = Path(tmp_vm)
            write_site_to_disk(vm_dir, site_files)
            proc = subprocess.run(
                [node_bin, "-e", NODE_VM_CHECK_HARNESS, str(vm_dir)],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if proc.returncode != 0:
                errors.append(f"Node vm runtime payload check failed: {(proc.stderr or proc.stdout).strip()}")

    # Verify build determinism
    second = build_tour_site_files(bundles, videos_rel=videos_rel)
    if site_digest(site_files) != site_digest(second):
        errors.append("non-deterministic tour build output detected across consecutive renders")

    return errors


def _make_mock_payload(module_index: int) -> Dict[str, Any]:
    """Construct complete schema-accurate mock payloads for Modules 01..08 self-tests."""
    if module_index == 1:
        fid = [round((0.85 ** k) * (1.0 if k % 3 != 2 else -0.6), 4) for k in range(32)]
        sine = [round(0.2 + 0.8 * (k / 31.0), 4) for k in range(32)]
        fft_p = [0.05] * 64
        for idx, amp in ((12, 1.0), (24, 0.72), (38, 0.28), (50, 0.85)):
            fft_p[idx - 1] = amp * 0.45
            fft_p[idx] = amp
            fft_p[idx + 1] = amp * 0.45
        fft_u = [round(v * 0.7 - 0.08, 4) for v in fft_p]
        return {
            "pulprog_name": "hncacbgpwg3d",
            "aqseq": "321",
            "f1_quad": "EchoAntiecho",
            "f2_quad": "StatesTppi",
            "transfer_summary": "Hncacb (3D)",
            "fid_real": fid,
            "sine_window": sine,
            "fft_unphased": fft_u,
            "fft_phased": fft_p,
            "p0_deg": 35.5,
            "p1_deg": -20.1,
            "nus_mask": [k % 4 == 0 for k in range(64)],
            "nus_dirty_spec": [round(v + (0.12 if i % 5 == 0 else 0.02), 4) for i, v in enumerate(fft_p)],
            "nus_fista_iters": [
                [round(v + (0.14 * (0.85 ** it) if i % 5 == 0 else 0.0), 4) for i, v in enumerate(fft_p)]
                for it in range(21)
            ],
            "nus_residual_l1": [round(0.42 * (0.90 ** it), 3) for it in range(21)],
        }
    if module_index == 2:
        strip_defs = [
            (1, "MET1", 8.25, 120.0, 54.5, 33.1, None, None),
            (2, "GLN2", 8.90, 122.8, 55.1, 30.8, 54.5, 33.1),
            (3, "ILE3", 8.32, 115.2, 59.6, 42.1, 55.1, 30.8),
            (4, "PHE4", 8.61, 118.4, 56.8, 41.2, 59.6, 42.1),
            (5, "VAL5", 9.30, 121.2, 61.8, 32.7, 56.8, 41.2),
            (6, "LYS6", 8.94, 127.8, 54.9, 32.9, 61.8, 32.7),
            (7, "THR7", 8.76, 115.3, 62.8, 73.3, 54.9, 32.9),
            (8, "LEU8", 9.12, 121.1, 57.0, 41.6, 62.8, 73.3),
            (9, "THR9", 7.65, 105.9, 61.5, 68.4, 57.0, 41.6),
            (10, "GLY10", 7.84, 109.2, 45.3, None, 61.5, 68.4),
        ]
        strips = []
        for sid, lbl, h, n, cai, cbi, caim1, cbim1 in strip_defs:
            strips.append({
                "seq_id": sid,
                "label": lbl,
                "h_ppm": h,
                "n_ppm": n,
                "ca_i": cai,
                "cb_i": cbi,
                "ca_im1": caim1,
                "cb_im1": cbim1,
                "ca_i_sign": 1 if cai is not None else None,
                "cb_i_sign": -1 if cbi is not None else None,
                "ca_im1_sign": 1 if caim1 is not None else None,
                "cb_im1_sign": -1 if cbim1 is not None else None,
                "pos_pts": [[20.0 + k * 5.0, 0.8 if abs((20.0 + k * 5.0) - cai) < 4.0 else 0.05] for k in range(11)],
                "neg_pts": [[20.0 + k * 5.0, 0.7 if (cbi and abs((20.0 + k * 5.0) - cbi) < 4.0) else 0.04] for k in range(11)],
            })
        return {
            "hsqc_pos_contours": [[[8.6, 118.2], [8.55, 118.6], [8.62, 118.9]]],
            "hsqc_neg_contours": [[[7.36, 115.4], [7.32, 115.8], [7.38, 116.0]]],
            "hsqc_roots": [{"h": 8.0 + (i % 15) * 0.1, "n": 108.0 + (i % 20), "label": f"RES{i + 1}"} for i in range(72)],
            "hncacb_pulse_regime": "non-CT Wittekind-Mueller HNCACB (2*d28=7.2ms): CA +1 (incl. Gly10 CA +1, CB absent; CT 2T=28ms inverts Gly CH2), CB -1",
            "strips": strips,
            "cv_temps": [round(5.0 * (0.85 ** k), 4) for k in range(12)],
            "cv_vals": [0.8, 1.1, 1.9, 3.4, 6.8, 11.2, 8.5, 4.2, 2.1, 1.3, 0.9, 0.6],
            "mean_energies": [-20.0 - k * 14.5 for k in range(12)],
            "t_crit": 2.21,
            "max_cv": 11.2,
            "evo_acc_pct": 98.6,
            "std_acc_pct": 88.9,
            "bayes_acc_pct": 94.4,
            "res_confidences": [0.92 if r not in (23, 52) else 0.68 for r in range(76)],
            "res_labels": [f"RES{r + 1}" for r in range(76)],
            "qc_verdict": "PASS",
            "n_high_conf": 74,
            "benchmarks": {
                "scope": "historical_disclosure",
                "ubiquitin_realistic_pct": 98.4,
                "cisd3_acc_pct": 64.38,
                "cisd3_hc_prec_pct": 100.0,
                "cold_bmrb_hc_min_pct": 88.25,
                "cold_bmrb_hc_max_pct": 95.30,
                "cold_bmrb_gate_pct": 94.0,
                "cold_bmrb_withdrawal_ref": "13.1:P1-2",
                "cold_bmrb_status": "FAILS >=94%; withdrawn under 13.1:P1-2",
                "historical_seed_count": 5,
                "ubiquitin_hc_correct": 310,
                "ubiquitin_hc_total": 315,
                "dev_target": "CISD3",
                "dev_target_bmrb_id": 51439,
                "cisd3_hc_correct": 59,
                "cisd3_hc_total": 59,
            },
            "n_res_total": 76,
            "n_hsqc_roots": 72,
            "hsqc_sf_h_mhz": 600.13,
            "hc_conf_threshold": 0.80,
            "strip_walk": {
                "start_label": "Phe4",
                "end_label": "Gly10",
                "inversion_label": "Thr7",
                "inversion_cb_ppm": 73.3,
                "inversion_ca_ppm": 62.8,
                "end_cb_null": True,
            },
        }
    if module_index == 3:
        return {
            "ca_coords": [[float(i) * 1.2, float((i % 5) - 2) * 1.8, float((i % 7) - 3) * 1.4] for i in range(24)],
            "ss_classes": [2 if i < 8 else (1 if i < 16 else 0) for i in range(24)],
            "s2_vals": [0.86 if i < 20 else 0.45 for i in range(24)],
            "noe_pairs": [{"a": 1, "b": 14, "upper_a": 4.2}, {"a": 3, "b": 12, "upper_a": 3.8}, {"a": 5, "b": 18, "upper_a": 4.5}, {"a": 7, "b": 20, "upper_a": 4.8}],
            "rdc_points": [{"d_obs": -10.0 + i * 2.1, "d_calc": -9.8 + i * 2.08, "is_free": (i % 4 == 0), "seq_id": i + 2} for i in range(12)],
            "q_work": 0.142,
            "q_free": 0.188,
            "da_nh_hz": -11.4,
            "rhombicity": 0.24,
            "f8_holdout_note": "F8: unseen 1KHM -> 1ZZK improves NOE-only on 5/5 seeds (1.86-4.14 A), 0/5 < 1.80 A",
        }
    if module_index == 4:
        seq_ids = [r for r in range(2, 32) if r != 19]
        return {
            "snr_levels": [80.0, 40.0, 20.0, 10.0, 6.0, 4.0],
            "locked_ls_bias_pct": [0.05, -0.08, 0.12, -0.19, 0.31, 0.42],
            "local_max_bias_pct": [1.4, 2.9, 5.8, 11.7, 19.4, 29.2],
            "times_r1": [0.02, 0.08, 0.16, 0.30, 0.50, 0.75, 1.05, 1.40],
            "curve_600_r1": [95.8, 84.1, 70.9, 52.4, 34.1, 19.9, 10.5, 4.9],
            "curve_800_r1": [96.8, 87.8, 77.1, 61.5, 44.5, 29.7, 18.2, 10.3],
            "fit_600_rate": 2.15,
            "fit_600_se": 0.04,
            "fit_800_rate": 1.62,
            "fit_800_se": 0.03,
            "tau_m_ns": 4.0837,
            "d_par_perp": 0.9987,
            "seq_ids": seq_ids,
            "r2_over_r1": [6.8 if r == 23 else 4.35 for r in seq_ids],
            "s2_vals": [0.84 if r == 23 else 0.87 for r in seq_ids],
            "rex_vals": [3.6 if r == 23 else 0.0 for r in seq_ids],
            "model_ids": ["M3" if r == 23 else "M2" for r in seq_ids],
            "fields_mhz": [600.0, 800.0],
            "field_a_mhz": 600.0,
            "field_b_mhz": 800.0,
            "fit_mc_replicates": 100,
            "decay_demo_seq_id": 23,
            "decay_demo_res_name": "ILE",
            "decay_demo_label": "Ile23",
            "snr_report_level": 4.0,
            "snr_report_ls_bias_pct": 0.42,
            "snr_report_max_bias_pct": 29.2,
            "pdb_id": "1UBQ",
            "first_seq_id": 1,
            "last_seq_id": 76,
            "sim_tail_start_seq": 72,
            "core_s2_mean": 0.87,
            "rex_report_threshold_s_inv": 0.5,
            "rex_sites": [{"seq_id": 23, "res_name": "ILE", "label": "Ile23", "model": "M3", "rex_s_inv": 3.6}],
            "rex_clause": "R_ex at Ile23 (M3)",
            "validation_status": {
                "scope": "todo_md_disclosure",
                "phase": 17,
                "blind_gates_passed": 0,
                "blind_gates_total": 8,
                "open_gates": ["F10", "F11"],
                "open_gate_scopes": ["real-data relaxation & model-free", "raw pseudo-3D relax_io"],
            },
        }
    if module_index == 5:
        offs = [round(115.0 + 0.5 * i, 2) for i in range(16)]
        fit15 = [0.18 if abs(x - 117.5) < 0.3 else (0.72 if abs(x - 121.0) < 0.3 else 0.98) for x in offs]
        fit30 = [0.08 if abs(x - 117.5) < 0.5 else (0.55 if abs(x - 121.0) < 0.4 else 0.96) for x in offs]
        return {
            "cest": {
                "offsets_ppm": offs,
                "b1_15hz_obs": fit15,
                "b1_15hz_fit": fit15,
                "b1_30hz_obs": fit30,
                "b1_30hz_fit": fit30,
                "p_b": 0.015,
                "k_ex_s_inv": 150.0,
                "major_ppm": 117.5,
                "dw_ab_ppm": 3.5,
            },
            "cpmg": {
                "nu_hz": [25.0, 50.0, 100.0, 200.0, 400.0, 800.0],
                "r2_600_obs": [18.4, 17.6, 15.8, 13.2, 11.1, 10.5],
                "r2_800_obs": [24.2, 22.9, 19.8, 15.6, 13.1, 12.6],
                "p_b": 0.019,
                "k_ex_s_inv": 850.0,
                "delta_aicc": 48.2,
                "significant_exchange": True,
                "f_pvalue": 1.2e-6,
            },
            "hsqc_trajectories": [
                {
                    "seq_id": 44,
                    "res_name": "ILE",
                    "max_csp": 0.412,
                    "points": [
                        {"ligand_um": l, "h_ppm": 8.42 + 0.0012 * l, "n_ppm": 121.2 - 0.006 * l, "csp_ppm": 0.41 * (l / (l + 30.0))}
                        for l in (0.0, 25.0, 75.0, 150.0, 250.0)
                    ],
                },
                {
                    "seq_id": 68,
                    "res_name": "HIS",
                    "max_csp": 0.275,
                    "points": [
                        {"ligand_um": l, "h_ppm": 7.92 + 0.0008 * l, "n_ppm": 118.1 + 0.004 * l, "csp_ppm": 0.28 * (l / (l + 30.0))}
                        for l in (0.0, 25.0, 75.0, 150.0, 250.0)
                    ],
                },
            ],
            "csp_isotherm": {
                "protein_conc_um": 100.0,
                "true_kd_um": 5.0,
                "fitted_kd_um": 5.08,
                "naive_mm_kd_um": 41.2,
                "ddelta_max_ppm": 0.412,
                "observed_points": [
                    {"ligand_um": 20.0, "csp_ppm": 0.078},
                    {"ligand_um": 75.0, "csp_ppm": 0.274},
                    {"ligand_um": 130.0, "csp_ppm": 0.382},
                    {"ligand_um": 250.0, "csp_ppm": 0.402},
                ],
            },
        }
    if module_index == 6:
        fppm = [round(5.2 - i * 0.12, 3) for i in range(28)]
        flat = [4.8 if abs(p - 3.84) < 0.08 else (5.5 if abs(p - 2.35) < 0.08 else 0.08) for p in fppm]
        with_w = [85.0 if abs(p - 4.78) < 0.12 else v + 0.6 for p, v in zip(fppm, flat)]
        mppm = [round(5.88 - i * 0.004, 4) for i in range(31)]
        mint = [0.85 if i in (4, 8, 12, 14, 16, 18, 22, 26) else 0.06 for i in range(31)]
        return {
            "fid_processing": {
                "ppm": fppm,
                "with_water": with_w,
                "hsvd_flattened": flat,
                "baseline_roll": [0.55 for _ in fppm],
                "voigt_peaks": [{"center_ppm": 3.85, "fwhm_hz": 2.2, "height": 4.8, "lorentz_fraction": 0.88}],
            },
            "multiplet": {
                "center_ppm": 5.82,
                "class": "ddd",
                "j_hz": [14.2, 7.1, 2.4],
                "acs_report": "1H NMR (500 MHz, CDCl3) delta 5.82 (ddd, J = 14.2, 7.1, 2.4 Hz, 1H).",
                "ppm": mppm,
                "intensity": mint,
            },
            "qnmr": {
                "purity_percent": 98.40,
                "u_k2_percent": 0.14,
                "relaxation_guard_passed": True,
                "analyte_integral": 1.968,
                "std_integral": 6.000,
            },
            "octet_ibuprofen": {
                "smiles": "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",
                "n_atoms": 8,
                "atoms": [{"idx": i, "element": "O" if i == 7 else "C", "x": float(i), "y": float(i % 2)} for i in range(8)],
                "bonds": [{"a": i, "b": i + 1, "order": 2 if i == 6 else 1} for i in range(7)],
                "h1_links": [
                    {"atom_idx": 0, "shift_ppm": 0.91, "n_protons": 6.0, "multiplicity": "d"},
                    {"atom_idx": 3, "shift_ppm": 7.18, "n_protons": 2.0, "multiplicity": "d"},
                    {"atom_idx": 5, "shift_ppm": 3.71, "n_protons": 1.0, "multiplicity": "q"},
                ],
                "c13_links": [{"atom_idx": 0, "shift_ppm": 22.4}],
            },
        }
    if module_index == 7:
        grads = [2.0 + i * (50.0 / 15.0) for i in range(16)]
        bvals = [(i / 15.0) ** 2 * 3.8e9 for i in range(16)]
        return {
            "gradients_g_cm": grads,
            "b_values_s_m2": bvals,
            "ppm_axis": [round(7.6 - i * 0.3, 2) for i in range(24)],
            "peaks": [
                {"label": "Lysozyme Ar-H", "center_ppm": 7.18, "selected_k": 1, "intensities": [round(0.95 ** i, 4) for i in range(16)], "components": [{"amplitude": 0.85, "d_m2_s": 1.12e-10, "log10_d": -9.95}]},
                {"label": "Maleate =CH", "center_ppm": 6.02, "selected_k": 1, "intensities": [round(0.76 ** i, 4) for i in range(16)], "components": [{"amplitude": 1.0, "d_m2_s": 8.40e-10, "log10_d": -9.08}]},
                {"label": "Raffinose H1", "center_ppm": 5.24, "selected_k": 1, "intensities": [round(0.86 ** i, 4) for i in range(16)], "components": [{"amplitude": 0.92, "d_m2_s": 3.10e-10, "log10_d": -9.51}]},
                {"label": "Overlap (Oligo+Ref)", "center_ppm": 3.66, "selected_k": 2, "intensities": [round(0.55 * (0.86 ** i) + 0.45 * (0.76 ** i), 4) for i in range(16)], "components": [{"amplitude": 0.55, "d_m2_s": 3.10e-10, "log10_d": -9.51}, {"amplitude": 0.45, "d_m2_s": 8.40e-10, "log10_d": -9.08}]},
            ],
            "ilt_overlap": {
                "log10_d_grid": [round(-10.3 + i * 0.1, 2) for i in range(16)],
                "spectrum": [0.85 if i in (8, 12) else 0.04 for i in range(16)],
                "selected_lambda": 0.012,
                "modes": [{"log10_d": -9.51, "d_m2_s": 3.10e-10, "area_fraction": 0.55}, {"log10_d": -9.08, "d_m2_s": 8.40e-10, "area_fraction": 0.45}],
            },
            "hydrodynamics": [
                {"name": "Maleate", "r_h_angstrom": 2.9, "mw_da": 116.0},
                {"name": "Raffinose", "r_h_angstrom": 5.8, "mw_da": 504.0},
                {"name": "Lysozyme", "r_h_angstrom": 19.2, "mw_da": 14300.0},
            ],
        }
    return {
        "n_samples": 24,
        "n_buckets": 42,
        "pca": {
            "explained_var": [0.62, 0.18],
            "scores": [{"t1": (-2.1 if i < 6 else 2.2) + 0.1 * i, "t2": (-0.6 if i % 2 == 0 else 0.5), "group": "Control" if i < 6 else "Case"} for i in range(12)],
            "ellipse_95": [[3.2 * (1.0 if k in (0, 1, 7, 8) else -1.0), 1.8 * (1.0 if k < 4 else -1.0)] for k in range(12)],
        },
        "oplsda": {
            "r2y": 0.984,
            "q2": 0.969,
            "perm_p_q2": 0.0100,
            "perm_null_q2": [-0.12, 0.04, -0.08],
            "s_plot": [
                {"ppm": 1.33, "cov": 0.62, "corr": 0.98, "vip": 2.41},
                {"ppm": 4.11, "cov": 0.41, "corr": 0.96, "vip": 2.18},
                {"ppm": 3.04, "cov": -0.18, "corr": -0.45, "vip": 0.88},
            ],
        },
        "stocsy": {
            "driver_ppm": 1.33,
            "top_match": "L-Lactate (bmse000288, score=0.99)",
            "ppm": [round(4.5 - i * 0.2, 2) for i in range(20)],
            "mean_ctrl": [0.35 if i not in (2, 16) else 0.65 for i in range(20)],
            "mean_case": [0.38 if i not in (2, 16) else 1.45 for i in range(20)],
            "r2": [0.99 if i in (2, 16) else 0.08 for i in range(20)],
        },
        "mcp": {
            "access_mode": "ReadOnly",
            "homolog_firewall_max_seq_id": 0.25,
            "undo_capacity": 256,
            "undo_bitwise_eq": True,
            "initialize_response": {"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2024-11-05"}},
            "torsion_call_response": {"jsonrpc": "2.0", "id": 2, "result": {"max_seq_identity": 0.25}},
            "readonly_denied_response": {"jsonrpc": "2.0", "id": 3, "error": {"code": -32001, "message": "ReadOnly policy denied write"}},
        },
    }


def _make_mock_bundles(data_dir: Path) -> None:
    """Create schema-accurate synthetic 01..08 *.tour.json fixtures for --self-test."""
    specs = [
        (1, "01_fid_pulprog_nus", "Raw-FID DSP & Pulprog Engine", "02-custom-pulprog-and-dsp.html", "fid_ingest_synthetic_nus", "RAW BRUKER/VARIAN FID INGEST · SYNTHETIC NUS DEMO", "25% Poisson-gap NUS & IST (synthetic demo)", "nus_iter", 0, 20, 1, 20, "iter"),
        (2, "02_hsqc_strips_solvers", "HSQC Strips & Backbone Solvers", "03-solvers-and-calibration.html", "cold_gate_failed_withdrawn", "SYNTHETIC / DEV · COLD HCP GATE FAILED (WITHDRAWN)", "Synthetic UBQ 98.4%, dev CISD3 64.38%; cold evo-graph 88.25-95.30% (FAILS >=94%; withdrawn)", "focus_res", 1, 10, 1, 7, "res"),
        (3, "03_structure_noe_rdc", "3D Structure, NOE & RDC", "01-quickstart-and-workflows.html", "cold_holdout_disclosed", "COLD HOLDOUT DISCLOSED (F8)", "Unseen 1KHM->1ZZK 1.86-4.14 A (0/5 < 1.80 A)", "orbit_deg", 0, 360, 5, 45, "deg"),
        (4, "04_relax_modelfree", "Relaxation & Lipari-Szabo", "04-dynamics-and-exchange.html", "synthetic_showcase", "SYNTHETIC FORWARD-MODEL SHOWCASE (F10/F11 OPEN)", "Dual-field R1/R2/hetNOE & M1-M5 S^2", "residue_seq", 1, 76, 1, 23, "res"),
        (5, "05_exchange_cest_cpmg_csp", "CEST, CPMG & Tight-Binding CSP", "04-dynamics-and-exchange.html", "synthetic_showcase", "SYNTHETIC FORWARD-MODEL SHOWCASE (F13 OPEN)", "Pade CEST, Baldwin CPMG & Morrison Kd", "ligand_um", 0, 300, 5, 100, "uM"),
        (6, "06_nmr1d_multiplet_qnmr_octet", "1D Multiplets, qNMR & Octet", "05-nmr1d-multiplets-and-dosy.html", "synthetic_showcase", "SYNTHETIC FORWARD-MODEL SHOWCASE (F14 OPEN)", "(S)-Ibuprofen Hoye ddd, GUM qNMR & Octet", "atom_idx", 0, 14, 1, 3, "atom"),
        (7, "07_dosy_diffusion", "DOSY Diffusion & Rh Sizing", "05-nmr1d-multiplets-and-dosy.html", "synthetic_showcase", "SYNTHETIC FORWARD-MODEL SHOWCASE (F16 OPEN)", "BPP-LED Stejskal-Tanner, 2D DOSY & Rh", "grad_step", 0, 15, 1, 4, "step"),
        (8, "08_metabolomics_mcp", "Metabolomics, STOCSY & MCP", "08-metabolomics-and-chemometrics.html", "synthetic_and_protocol_showcase", "SYNTHETIC & PROTOCOL SHOWCASE (F17/F18 OPEN)", "AI-Binning, PQN, OPLS-DA, STOCSY & MCP JSON-RPC", "driver_bucket", 0, 20, 1, 4, "bucket"),
    ]
    data_dir.mkdir(parents=True, exist_ok=True)
    for idx, slug, title, man_ch, b_kind, b_lbl, b_det, cid, cmin, cmax, cstep, cdef, cunit in specs:
        payload = _make_mock_payload(idx)
        subtitle = f"Deterministic nmr-core showcase for {title}"
        chapters = [
            {"index": 1, "start_sec": 0.0, "end_sec": 4.0, "title": "Stage One", "caption": "Chapter 1 physics", "metric": "Ch1"},
            {"index": 2, "start_sec": 4.0, "end_sec": 8.0, "title": "Stage Two", "caption": "Chapter 2 physics", "metric": "Ch2"},
            {"index": 3, "start_sec": 8.0, "end_sec": 12.0, "title": "Stage Three", "caption": "Chapter 3 physics", "metric": "Ch3"},
            {"index": 4, "start_sec": 12.0, "end_sec": 16.0, "title": "Stage Four", "caption": "Chapter 4 physics", "metric": "Ch4"},
        ]
        if idx == 2:
            bm = payload["benchmarks"]
            thr7 = payload["strips"][6]
            subtitle = (
                f"Live fixture: evo-graph {payload['evo_acc_pct']:.1f}% · [Historical disclosure] "
                f"UBQ {bm['ubiquitin_realistic_pct']:.1f}%, dev {bm['dev_target']} {bm['cisd3_acc_pct']:.2f}%/"
                f"{bm['cisd3_hc_prec_pct']:.0f}% HC, cold BMRB {bm['cold_bmrb_hc_min_pct']:.2f}-"
                f"{bm['cold_bmrb_hc_max_pct']:.2f}% (FAILS >={bm['cold_bmrb_gate_pct']:.0f}%; withdrawn)"
            )
            b_det = (
                f"Synthetic UBQ {bm['ubiquitin_realistic_pct']:.1f}% "
                f"({bm['ubiquitin_hc_correct']}/{bm['ubiquitin_hc_total']}), dev {bm['dev_target']} "
                f"{bm['cisd3_acc_pct']:.2f}% ({bm['cisd3_hc_correct']}/{bm['cisd3_hc_total']}); cold evo-graph "
                f"{bm['cold_bmrb_hc_min_pct']:.2f}-{bm['cold_bmrb_hc_max_pct']:.2f}% "
                f"(FAILS >= {bm['cold_bmrb_gate_pct']:.0f}% gate; withdrawn under {bm['cold_bmrb_withdrawal_ref']})"
            )
            chapters[1]["caption"] = (
                f"Walks Phe4 -> Gly10: Thr7 CB={thr7['cb_i']:.1f} > CA={thr7['ca_i']:.1f} inversion."
            )
            chapters[2]["caption"] = (
                f"Live bayes-sa Cv(T) alongside [historical disclosure]: UBQ {bm['ubiquitin_realistic_pct']:.1f}%, "
                f"dev CISD3 {bm['cisd3_acc_pct']:.2f}%/{bm['cisd3_hc_prec_pct']:.0f}% HC, "
                f"cold BMRB {bm['cold_bmrb_hc_min_pct']:.2f}-{bm['cold_bmrb_hc_max_pct']:.2f}% "
                f"(FAILS >={bm['cold_bmrb_gate_pct']:.0f}%)."
            )
        elif idx == 4:
            tau_m = float(payload["tau_m_ns"])
            d_pp = float(payload["d_par_perp"])
            chapters[2]["caption"] = (
                f"Outlier-trimmed R2/R1 global tau_m={tau_m:.2f} ns & Woessner axial D_par/D_perp={d_pp:.2f}."
            )
            chapters[2]["metric"] = f"tau_m={tau_m:.2f} ns  D_par/D_perp={d_pp:.2f}"
            chapters[3]["caption"] = f"S^2 profile: {payload['rex_clause']}."
            vs = payload["validation_status"]
            b_det = (
                f"Synthetic forward-model showcase: {vs['blind_gates_passed']}/{vs['blind_gates_total']} "
                f"Phase {vs['phase']} blind real-data gates passed."
            )
        doc = {
            "schema_version": EXPECTED_SCHEMA_VERSION,
            "module_index": idx,
            "slug": slug,
            "title": title,
            "subtitle": subtitle,
            "manual_chapter": man_ch,
            "validation_badge": {"kind": b_kind, "label": b_lbl, "detail": b_det},
            "duration_sec": 16.0,
            "fps": 30,
            "cli_command": f"ferric export-tour --module {idx:02d} --out-dir docs/tours",
            "chapters": chapters,
            "interactive_control": {
                "id": cid,
                "label": cid,
                "min": cmin,
                "max": cmax,
                "step": cstep,
                "default_val": cdef,
                "unit": cunit,
            },
            "payload": payload,
        }
        (data_dir / f"{slug}.tour.json").write_text(
            json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8"
        )


def run_self_tests() -> int:
    """Run deterministic, offline-CSP, WCAG, payload-completeness, fallback-banner, zero-innerHTML, and Node vm self-tests."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data_dir = root / "data"
        out_a = root / "out_a"
        out_b = root / "out_b"
        _make_mock_bundles(data_dir)

        bundles = load_tour_bundles(data_dir)
        assert len(bundles) == 8, f"expected 8 bundles, got {len(bundles)}"

        files_a = build_tour_site_files(bundles, videos_rel="videos")
        files_b = build_tour_site_files(bundles, videos_rel="videos")
        write_site_to_disk(out_a, files_a)
        write_site_to_disk(out_b, files_b)

        # 1. Determinism & 11 expected artifacts (9 HTML + CSS + JS)
        assert len(files_a) == 11, f"expected 11 output files, got {len(files_a)}"
        assert site_digest(files_a) == site_digest(files_b), "build is non-deterministic"

        # 2. Full validator pass & zero .innerHTML / raw_json in generated JS
        errs = validate_tour_site(files_a, bundles, videos_rel="videos")
        assert not errs, f"unexpected validation errors: {errs}"
        assert b"raw_json" not in files_a["tours_player.js"], (
            "generated tours_player.js must not reference Python raw_json wrapper"
        )
        assert b".innerHTML" not in files_a["tours_player.js"], (
            "generated tours_player.js must not contain .innerHTML"
        )

        # 3. Adversarial checks: external CDN must fail
        mut_cdn = dict(files_a)
        mut_cdn["index.html"] = files_a["index.html"].replace(
            b"</head>", b'<script src="https://cdn.example.com/x.js"></script></head>'
        )
        assert any("CDN" in e for e in validate_tour_site(mut_cdn, bundles))

        # 4. Adversarial check: user-scalable=no must fail WCAG 1.4.4
        mut_wcag = dict(files_a)
        mut_wcag["index.html"] = files_a["index.html"].replace(
            b"viewport-fit=cover", b"viewport-fit=cover, user-scalable=no"
        )
        assert any("zoom-disabling" in e for e in validate_tour_site(mut_wcag, bundles))

        # 5. Adversarial check: missing SYNTHETIC disclosure on module 04 must fail
        mut_synth = dict(files_a)
        mut_synth["04_relax_modelfree.html"] = files_a["04_relax_modelfree.html"].replace(
            b"SYNTHETIC", b"HIDDEN"
        )
        assert any("SYNTHETIC" in e for e in validate_tour_site(mut_synth, bundles))

        # 6. Adversarial check: restricted product token must fail
        mut_tok = dict(files_a)
        mut_tok["index.html"] = files_a["index.html"] + f"<!-- {RESTRICTED_PRODUCT_TOKENS[0]} -->".encode("utf-8")
        assert any("forbidden NMR product token" in e for e in validate_tour_site(mut_tok, bundles))

        # 7. Adversarial check: .innerHTML in tours_player.js must fail
        mut_xss = dict(files_a)
        mut_xss["tours_player.js"] = files_a["tours_player.js"] + b"\n// tooltip.innerHTML = x;\n"
        assert any("innerHTML" in e for e in validate_tour_site(mut_xss, bundles))

        # 8. Adversarial check: missing payload series unhides #fallback-banner AND fails --check
        bad_data_dir = root / "bad_data"
        _make_mock_bundles(bad_data_dir)
        mod04_path = bad_data_dir / "04_relax_modelfree.tour.json"
        mod04_doc = json.loads(mod04_path.read_text(encoding="utf-8"))
        del mod04_doc["payload"]["tau_m_ns"]
        del mod04_doc["payload"]["curve_600_r1"]
        mod04_path.write_text(json.dumps(mod04_doc, indent=2, sort_keys=True), encoding="utf-8")
        bad_out = root / "bad_out"
        rc_bad = main(["--data-dir", str(bad_data_dir), "--out", str(bad_out), "--videos-rel", "videos", "--check"])
        assert rc_bad == 1, f"expected --check rc=1 on missing payload series, got {rc_bad}"
        node_bin = shutil.which("node")
        if node_bin is not None:
            proc_fb = subprocess.run(
                [node_bin, "-e", NODE_VM_CHECK_HARNESS, str(bad_out), "expect-fallback"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            assert proc_fb.returncode == 0, f"Fallback banner runtime check failed:\n{proc_fb.stdout}\n{proc_fb.stderr}"

        # 9. Adversarial check (B15): caption/payload mismatch on Mod02 cold range or Mod04 tau_m must fail
        b15_data_dir = root / "b15_mismatch_data"
        _make_mock_bundles(b15_data_dir)
        m02_p = b15_data_dir / "02_hsqc_strips_solvers.tour.json"
        m02_d = json.loads(m02_p.read_text(encoding="utf-8"))
        # Replicate f8408b9 bug: payload has 86.18-96.31 while captions quote 88.25-95.30
        m02_d["payload"]["benchmarks"]["cold_bmrb_hc_min_pct"] = 86.18
        m02_d["payload"]["benchmarks"]["cold_bmrb_hc_max_pct"] = 96.31
        m02_p.write_text(json.dumps(m02_d, indent=2, sort_keys=True), encoding="utf-8")
        m04_p = b15_data_dir / "04_relax_modelfree.tour.json"
        m04_d = json.loads(m04_p.read_text(encoding="utf-8"))
        # Replicate f8408b9 bug: caption quotes 4.10 ns while payload is 4.0837 ns
        m04_d["chapters"][2]["caption"] = "Outlier-trimmed R2/R1 global tau_m=4.10 ns & Woessner axial."
        m04_p.write_text(json.dumps(m04_d, indent=2, sort_keys=True), encoding="utf-8")
        b15_bundles = load_tour_bundles(b15_data_dir)
        b15_files = build_tour_site_files(b15_bundles, videos_rel="videos")
        b15_errs = validate_tour_site(b15_files, b15_bundles, videos_rel="videos")
        assert any("02_hsqc_strips_solvers.tour.json" in e and "cold range" in e for e in b15_errs), (
            f"expected B15 Mod02 cold-range mismatch error, got: {b15_errs}"
        )
        assert any("04_relax_modelfree.tour.json" in e and "tau_m=4.08 ns" in e for e in b15_errs), (
            f"expected B15 Mod04 tau_m mismatch error, got: {b15_errs}"
        )

        # 11. B15: legend, preset chips and canvas JS must render payload scalars (not literals).
        #     Plant sentinel values in the mock payloads and require them to surface verbatim.
        sent_dir = root / "b15_sentinel_data"
        _make_mock_bundles(sent_dir)
        s02 = sent_dir / "02_hsqc_strips_solvers.tour.json"
        d02 = json.loads(s02.read_text(encoding="utf-8"))
        d02["payload"]["benchmarks"]["cold_bmrb_gate_pct"] = 91.0
        d02["payload"]["benchmarks"]["cold_bmrb_withdrawal_ref"] = "99.9:PX-9"
        d02["payload"]["benchmarks"]["dev_target"] = "PROTX"
        d02["payload"]["hc_conf_threshold"] = 0.85
        d02["payload"]["n_res_total"] = 77
        d02["payload"]["strip_walk"]["start_label"] = "Abc4"
        s02.write_text(json.dumps(d02, indent=2, sort_keys=True), encoding="utf-8")
        s04 = sent_dir / "04_relax_modelfree.tour.json"
        d04 = json.loads(s04.read_text(encoding="utf-8"))
        d04["payload"].update({
            "field_a_mhz": 611.0,
            "field_b_mhz": 822.0,
            "decay_demo_label": "Xaa99",
            "fit_mc_replicates": 37,
            "rex_report_threshold_s_inv": 0.7,
            "rex_clause": "R_ex at Xaa99 (M4)",
            "rex_sites": [{"seq_id": 23, "res_name": "XAA", "label": "Xaa99", "model": "M4", "rex_s_inv": 2.0}],
            "last_seq_id": 81,
        })
        s04.write_text(json.dumps(d04, indent=2, sort_keys=True), encoding="utf-8")
        sent_bundles = load_tour_bundles(sent_dir)
        sent_files = build_tour_site_files(sent_bundles, videos_rel="videos")
        index_html = sent_files["index.html"].decode("utf-8")
        leg_at = index_html.find("<span>Module 02 [historical disclosure]")
        assert leg_at >= 0, "gallery legend for Module 02 historical disclosure not rendered"
        legend = index_html[leg_at:index_html.find("</span>", leg_at)]
        assert "FAILS &ge; 91%" in legend and "99.9:PX-9" in legend and "dev PROTX" in legend, (
            "gallery legend must read benchmarks.cold_bmrb_gate_pct / cold_bmrb_withdrawal_ref / dev_target"
        )
        assert "&ge; 94%" not in legend and "13.1:P1-2" not in legend, "legend still hard-codes 94% / 13.1:P1-2"
        page04 = sent_files["04_relax_modelfree.html"].decode("utf-8")
        assert "Xaa99 (M4 Rex)" in page04 and "Ile23 (Rex)" not in page04, "Mod04 R_ex preset chip must come from payload.rex_sites"
        if node_bin is not None:
            sent_out = root / "b15_sentinel_out"
            write_site_to_disk(sent_out, sent_files)
            dump_file = root / "b15_text_dump.json"
            proc_txt = subprocess.run(
                [node_bin, "-e", NODE_VM_CHECK_HARNESS, str(sent_out), f"dump-text={dump_file}"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            assert proc_txt.returncode == 0, f"B15 sentinel Node run failed:\n{proc_txt.stdout}\n{proc_txt.stderr}"
            dump = json.loads(dump_file.read_text(encoding="utf-8"))
            joined = {k: " | ".join(v) for k, v in dump.items()}
            want = {
                "02_hsqc_strips_solvers.html#2": ["Abc4 ->"],
                "02_hsqc_strips_solvers.html#3": ["FAILS >=91%", "PROTX "],
                "02_hsqc_strips_solvers.html#4": ["77-Residue", "/77 >= 0.85", "Residue 1..77"],
                "04_relax_modelfree.html#2": ["611 & 822 MHz", "(Xaa99)", "37-trial", "611 MHz"],
                "04_relax_modelfree.html#3": ["611 MHz R2 / R1", "(1..81)"],
                "04_relax_modelfree.html#4": ["R_ex > 0.7 s^-1", "R_ex at Xaa99 (M4)", "(1..81)"],
            }
            for key, needles in want.items():
                txt = joined.get(key, "")
                for needle in needles:
                    assert needle in txt, f"B15: {key} canvas text missing payload-derived {needle!r}: {txt[:400]}"
            forbid = {
                "02_hsqc_strips_solvers.html#2": ["Phe4 -> Gly10"],
                "02_hsqc_strips_solvers.html#4": ["76-Residue", "/76", ">= 0.80"],
                "04_relax_modelfree.html#2": ["600 & 800", "(Ile23)", "600 MHz", "800 MHz"],
                "04_relax_modelfree.html#4": ["R_ex > 0.5"],
            }
            for key, needles in forbid.items():
                txt = joined.get(key, "")
                for needle in needles:
                    assert needle not in txt, f"B15: {key} canvas text still hard-codes {needle!r}"
        # Missing B15 keys must fail --check (required payload reads fail loudly).
        miss_dir = root / "b15_missing_data"
        _make_mock_bundles(miss_dir)
        m04m = miss_dir / "04_relax_modelfree.tour.json"
        dm = json.loads(m04m.read_text(encoding="utf-8"))
        del dm["payload"]["rex_report_threshold_s_inv"]
        m04m.write_text(json.dumps(dm, indent=2, sort_keys=True), encoding="utf-8")
        m02m = miss_dir / "02_hsqc_strips_solvers.tour.json"
        dm2 = json.loads(m02m.read_text(encoding="utf-8"))
        del dm2["payload"]["benchmarks"]["cold_bmrb_gate_pct"]
        m02m.write_text(json.dumps(dm2, indent=2, sort_keys=True), encoding="utf-8")
        miss_b = load_tour_bundles(miss_dir)
        miss_errs = validate_tour_site(build_tour_site_files(miss_b, videos_rel="videos"), miss_b, videos_rel="videos")
        assert any("payload.rex_report_threshold_s_inv" in e for e in miss_errs), miss_errs
        assert any("payload.benchmarks.cold_bmrb_gate_pct" in e for e in miss_errs), miss_errs
        # Caption naming a model the fit did not select must fail.
        wrong_dir = root / "b15_wrong_model"
        _make_mock_bundles(wrong_dir)
        m04w = wrong_dir / "04_relax_modelfree.tour.json"
        dw = json.loads(m04w.read_text(encoding="utf-8"))
        dw["chapters"][3]["caption"] = "S^2 profile: Ile23 R_ex (M4)."
        m04w.write_text(json.dumps(dw, indent=2, sort_keys=True), encoding="utf-8")
        wb = load_tour_bundles(wrong_dir)
        w_errs = validate_tour_site(build_tour_site_files(wb, videos_rel="videos"), wb, videos_rel="videos")
        assert any("rex_clause" in e for e in w_errs) and any("model M4" in e for e in w_errs), w_errs

        # 10. CLI --check end-to-end verification (including live Node vm 32/32 check inside validate_tour_site)
        check_out = root / "out_cli_check"
        rc = main(["--data-dir", str(data_dir), "--out", str(check_out), "--videos-rel", "videos", "--check"])
        assert rc == 0, f"expected --check rc=0, got {rc}"
        assert (check_out / "index.html").is_file()
        assert (check_out / "tours.css").is_file()
        assert (check_out / "tours_player.js").is_file()

        digest = site_digest(files_a)[:16]
        print(
            f"build_interactive_tours.py self-tests passed (11/11 checks: 11 files, "
            f"zero-CDN, WCAG 1.4.4, 01..08 payload-backed, fallback-banner guard, zero-innerHTML, clean-room, B15 caption-payload parity, "
            f"B15 payload-sentinel legend/chips/canvas text, --check CLI, Node vm 32/32; sha256={digest})."
        )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Build offline static HTML5 Interactive Tour Gallery & Player from *.tour.json"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repo_root / "docs" / "tours" / "data",
        help="Directory containing 01..08_*.tour.json files",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=repo_root / "target" / "tours",
        help="Output directory for generated static HTML/CSS/JS tour site",
    )
    parser.add_argument(
        "--videos-rel",
        type=str,
        default="videos",
        help="Relative directory path for .mp4 and .webm pre-rendered videos (default: videos)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate determinism, zero CDN, WCAG 1.4.4, synthetic disclosures, and JS syntax",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run built-in mock fixture & adversarial validation self-tests and exit",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_tests()

    bundles = load_tour_bundles(args.data_dir)
    site_files = build_tour_site_files(bundles, videos_rel=args.videos_rel)
    write_site_to_disk(args.out, site_files)
    brand_dir = repo_root / "docs" / "assets" / "brand"
    for fav_name in ("favicon.svg", "favicon-32.png"):
        fav_src = brand_dir / fav_name
        if fav_src.is_file():
            (args.out / fav_name).write_bytes(fav_src.read_bytes())

    if args.check:
        errors = validate_tour_site(site_files, bundles, videos_rel=args.videos_rel)
        if errors:
            for err in errors:
                print(f"error: {err}", file=sys.stderr)
            print(f"tour site check FAILED with {len(errors)} error(s).", file=sys.stderr)
            return 1

    digest = site_digest(site_files)[:16]
    print(
        f"Built {len(bundles)} interactive tour pages + gallery ({len(site_files)} files, "
        f"sha256={digest}) -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
