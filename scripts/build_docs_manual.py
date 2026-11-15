#!/usr/bin/env python3
"""Deterministic, zero-dependency static HTML manual builder and link checker.

Builds the multi-chapter GitHub-Pages documentation manual under ``docs/manual/``
into a self-contained static site with:
- Responsive dark/light scientific theme (persisted in localStorage, zero CDN calls)
- Left chapter navigation, sticky right-hand "On This Page" outline, prev/next cards
- Pure-stdlib Markdown parser supporting pipe tables (preserving ``|`` in code/math),
  GitHub-style callout alerts (``[!NOTE]``, ``[!TIP]``, ``[!IMPORTANT]``,
  ``[!WARNING]``, ``[!CAUTION]``), nested lists, fenced code blocks with syntax
  highlighting & copy buttons, and offline typographic LaTeX math rendering
- Client-side tokenized instant search index (``search-index.json``, ``/`` or ``Cmd/Ctrl+K``)
- Strict ``--check`` validator verifying deterministic output and 0 broken internal
  page, asset, or ``#anchor`` fragment links.

Provenance: Original clean-room Python 3 standard-library build script (MIT License).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import html
import json
from pathlib import Path
import posixpath
import re
import sys
import tempfile
from typing import Dict, List, Optional, Sequence, Set, Tuple
from urllib.parse import unquote, urlsplit


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
    return "0.11.0"


SITE_TITLE = "nmr-assign Manual"
SITE_VERSION_BADGE = f"v{_read_workspace_version()} · Rust + WebGPU"
SITE_TAGLINE = "Clean-Room Biomolecular & Analytical NMR Platform"

EXPECTED_MANUAL_CHAPTERS: Tuple[str, ...] = (
    "index.md",
    "01-quickstart-and-workflows.md",
    "02-custom-pulprog-and-dsp.md",
    "03-solvers-and-calibration.md",
    "04-dynamics-and-exchange.md",
    "05-nmr1d-multiplets-and-dosy.md",
    "06-mcp-and-cli-reference.md",
    "07-clean-room-provenance.md",
    "08-metabolomics-and-chemometrics.md",
)

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


@dataclass(frozen=True)
class HeadingItem:
    level: int
    title_html: str
    title_text: str
    slug: str
    line_no: int


@dataclass(frozen=True)
class LinkRef:
    source_md: str
    line_no: int
    raw_target: str
    label_text: str


@dataclass
class SearchRecord:
    page_slug: str
    page_title: str
    section_title: str
    href: str
    body_text: str


@dataclass
class ParsedPage:
    md_name: str
    html_name: str
    chapter_badge: str
    nav_title: str
    page_title: str
    headings: List[HeadingItem] = field(default_factory=list)
    anchor_ids: Set[str] = field(default_factory=set)
    links: List[LinkRef] = field(default_factory=list)
    body_html: str = ""
    search_records: List[SearchRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Slug & plain-text helpers
# ---------------------------------------------------------------------------

def strip_markdown_inline(raw: str) -> str:
    """Convert Markdown inline markup into clean human-readable plain text."""
    text = raw
    # Replace images ![alt](url) -> alt and links [label](url) -> label
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Strip inline code backticks
    text = re.sub(r"`+([^`]+)`+", r"\1", text)
    # Simplify common inline LaTeX tokens for readable headings/search
    text = text.replace(r"\Delta\chi", "Delta-chi")
    text = text.replace(r"\Delta", "Delta")
    text = text.replace(r"\delta", "delta")
    text = text.replace(r"\tau_m", "tau_m")
    text = text.replace(r"\tau_e", "tau_e")
    text = text.replace(r"\phi", "phi")
    text = text.replace(r"\psi", "psi")
    text = text.replace(r"\chi_1", "chi1")
    text = re.sub(r"\\(?:text|mathrm|mathbf|operatorname)\{([^{}]*)\}", r"\1", text)
    text = text.replace("$", "")
    text = text.replace("{", "").replace("}", "")
    text = text.replace("\\", "")
    # Strip bold/italic markers
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def slugify_heading(raw_heading: str, used_slugs: Set[str]) -> str:
    """Generate deterministic GitHub-compatible heading anchor IDs."""
    plain = strip_markdown_inline(raw_heading).lower()
    plain = plain.replace("&", " and ")
    plain = re.sub(r"[^a-z0-9\s-]", "", plain)
    base = re.sub(r"[\s-]+", "-", plain).strip("-")
    if not base:
        base = "section"
    candidate = base
    suffix = 1
    while candidate in used_slugs:
        candidate = f"{base}-{suffix}"
        suffix += 1
    used_slugs.add(candidate)
    return candidate


# ---------------------------------------------------------------------------
# Offline typographic LaTeX math converter
# ---------------------------------------------------------------------------

GREEK_AND_SYMBOLS: Dict[str, str] = {
    r"\alpha": "α",
    r"\beta": "β",
    r"\gamma": "γ",
    r"\delta": "δ",
    r"\epsilon": "ϵ",
    r"\varepsilon": "ε",
    r"\zeta": "ζ",
    r"\eta": "η",
    r"\theta": "θ",
    r"\vartheta": "ϑ",
    r"\iota": "ι",
    r"\kappa": "κ",
    r"\lambda": "λ",
    r"\mu": "μ",
    r"\nu": "ν",
    r"\xi": "ξ",
    r"\pi": "π",
    r"\varpi": "ϖ",
    r"\rho": "ρ",
    r"\sigma": "σ",
    r"\tau": "τ",
    r"\upsilon": "υ",
    r"\phi": "ϕ",
    r"\varphi": "φ",
    r"\chi": "χ",
    r"\psi": "ψ",
    r"\omega": "ω",
    r"\Gamma": "Γ",
    r"\Delta": "Δ",
    r"\Theta": "Θ",
    r"\Lambda": "Λ",
    r"\Xi": "Ξ",
    r"\Pi": "Π",
    r"\Sigma": "Σ",
    r"\Upsilon": "Υ",
    r"\Phi": "Φ",
    r"\Psi": "Ψ",
    r"\Omega": "Ω",
    r"\nabla": "∇",
    r"\partial": "∂",
    r"\infty": "∞",
    r"\hbar": "ℏ",
    r"\ell": "ℓ",
    r"\AA": "Å",
    r"\pm": "±",
    r"\mp": "∓",
    r"\times": "×",
    r"\cdot": "·",
    r"\cdots": "⋯",
    r"\ldots": "…",
    r"\dots": "…",
    r"\le": "≤",
    r"\leq": "≤",
    r"\ge": "≥",
    r"\geq": "≥",
    r"\ne": "≠",
    r"\neq": "≠",
    r"\approx": "≈",
    r"\sim": "∼",
    r"\simeq": "≃",
    r"\equiv": "≡",
    r"\propto": "∝",
    r"\ll": "≪",
    r"\gg": "≫",
    r"\in": "∈",
    r"\notin": "∉",
    r"\subset": "⊂",
    r"\subseteq": "⊆",
    r"\cup": "∪",
    r"\cap": "∩",
    r"\forall": "∀",
    r"\exists": "∃",
    r"\to": "→",
    r"\rightarrow": "→",
    r"\leftarrow": "←",
    r"\leftrightarrow": "↔",
    r"\Rightarrow": "⇒",
    r"\Leftrightarrow": "⇔",
    r"\mapsto": "↦",
    r"\langle": "⟨",
    r"\rangle": "⟩",
    r"\lceil": "⌈",
    r"\rceil": "⌉",
    r"\lfloor": "⌊",
    r"\rfloor": "⌋",
    r"\lvert": "|",
    r"\rvert": "|",
    r"\lVert": "‖",
    r"\rVert": "‖",
    r"\sum": "∑",
    r"\prod": "∏",
    r"\int": "∫",
    r"\top": "⊤",
    r"\perp": "⊥",
    r"\parallel": "∥",
    r"\circ": "∘",
    r"\^\circ": "°",
}

ROMAN_OPERATORS: Tuple[str, ...] = (
    "argmin",
    "argmax",
    "atan2",
    "median",
    "corr",
    "cov",
    "cosh",
    "sinh",
    "tanh",
    "cos",
    "sin",
    "tan",
    "exp",
    "ln",
    "log",
    "max",
    "min",
    "det",
    "diag",
    "tr",
    "Var",
    "Cov",
    "MAD",
    "RSS",
    "SNR",
    "Re",
    "Im",
)


def _extract_braced(tex: str, pos: int) -> Tuple[str, int]:
    """Read a balanced ``{...}`` group starting at ``pos`` (where ``tex[pos] == '{'``)."""
    assert tex[pos] == "{"
    depth = 1
    i = pos + 1
    n = len(tex)
    while i < n and depth > 0:
        ch = tex[i]
        if ch == "\\" and i + 1 < n:
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        i += 1
    return tex[pos + 1 : i - 1], i


def _extract_script_atom(tex: str, pos: int) -> Tuple[str, int]:
    """Extract a single superscript/subscript argument starting at ``pos``."""
    n = len(tex)
    while pos < n and tex[pos] == " ":
        pos += 1
    if pos >= n:
        return "", pos
    if tex[pos] == "{":
        return _extract_braced(tex, pos)
    if tex[pos] == "\\":
        j = pos + 1
        while j < n and tex[j].isalpha():
            j += 1
        if j == pos + 1 and j < n:
            j += 1
        cmd = tex[pos:j]
        if j < n and tex[j] == "{":
            inner, end_j = _extract_braced(tex, j)
            return f"{cmd}{{{inner}}}", end_j
        return cmd, j
    return tex[pos], pos + 1


def _render_tex_inner(tex: str) -> str:
    """Recursively convert a LaTeX math fragment into semantic HTML."""
    # Handle bmatrix environments first
    while r"\begin{bmatrix}" in tex and r"\end{bmatrix}" in tex:
        start = tex.index(r"\begin{bmatrix}")
        end = tex.index(r"\end{bmatrix}", start)
        body = tex[start + len(r"\begin{bmatrix}") : end].strip()
        rows = [r.strip() for r in body.split(r"\\") if r.strip()]
        row_htmls = []
        for row in rows:
            cells = [f"<td>{_render_tex_inner(c.strip())}</td>" for c in row.split("&")]
            row_htmls.append("<tr>" + "".join(cells) + "</tr>")
        mat_html = '<span class="mmatrix">[<table>' + "".join(row_htmls) + "</table>]</span>"
        tex = tex[:start] + f"\x00MAT{len(row_htmls)}\x00" + tex[end + len(r"\end{bmatrix}") :]
        tex = tex.replace(f"\x00MAT{len(row_htmls)}\x00", mat_html, 1)

    out: List[str] = []
    i = 0
    n = len(tex)
    while i < n:
        ch = tex[i]
        # Pass through already-rendered HTML spans from matrix substitution
        if ch == "<" and tex.startswith('<span class="mmatrix">', i):
            close_tag = "</table>]</span>"
            end_idx = tex.find(close_tag, i)
            if end_idx != -1:
                out.append(tex[i : end_idx + len(close_tag)])
                i = end_idx + len(close_tag)
                continue

        if ch == "{":
            inner, i = _extract_braced(tex, i)
            out.append(_render_tex_inner(inner))
            continue

        if ch in ("^", "_"):
            tag = "sup" if ch == "^" else "sub"
            arg, i = _extract_script_atom(tex, i + 1)
            out.append(f"<{tag}>{_render_tex_inner(arg)}</{tag}>")
            continue

        if ch == "\\":
            # Escaped delimiters and spacing commands
            if tex.startswith(r"\|", i):
                out.append("‖")
                i += 2
                continue
            if tex.startswith(r"\{", i):
                out.append("{")
                i += 2
                continue
            if tex.startswith(r"\}", i):
                out.append("}")
                i += 2
                continue
            if tex.startswith(r"\%", i):
                out.append("%")
                i += 2
                continue
            if tex.startswith(r"\_", i):
                out.append("_")
                i += 2
                continue
            if tex.startswith((r"\,", r"\;", r"\:", r"\!"), i):
                if not tex.startswith(r"\!", i):
                    out.append("&#8239;")
                i += 2
                continue
            if tex.startswith(r"\quad", i):
                out.append("&emsp;")
                i += 5
                continue
            if tex.startswith(r"\qquad", i):
                out.append("&emsp;&emsp;")
                i += 6
                continue
            if tex.startswith(r"\\", i):
                out.append("<br>")
                i += 2
                continue

            j = i + 1
            while j < n and tex[j].isalpha():
                j += 1
            if j == i + 1:
                if j < n:
                    out.append(html.escape(tex[j]))
                    i = j + 1
                else:
                    i += 1
                continue

            cmd = tex[i:j]
            cmd_name = cmd[1:]

            # Sizing modifiers (\left, \right, \bigl, \bigr, \Big, etc.)
            if cmd_name in (
                "left",
                "right",
                "bigl",
                "bigr",
                "Bigl",
                "Bigr",
                "big",
                "Big",
                "bigg",
                "Bigg",
                "middle",
            ):
                i = j
                while i < n and tex[i] == " ":
                    i += 1
                if i < n and tex[i] == ".":
                    i += 1
                continue

            if cmd_name in ("frac", "dfrac", "tfrac"):
                k = j
                while k < n and tex[k] == " ":
                    k += 1
                if k < n and tex[k] == "{":
                    num, k = _extract_braced(tex, k)
                    while k < n and tex[k] == " ":
                        k += 1
                    if k < n and tex[k] == "{":
                        den, k = _extract_braced(tex, k)
                        out.append(
                            f'<span class="mfrac"><span class="mnum">{_render_tex_inner(num)}</span>'
                            f'<span class="mden">{_render_tex_inner(den)}</span></span>'
                        )
                        i = k
                        continue

            if cmd_name == "sqrt":
                k = j
                while k < n and tex[k] == " ":
                    k += 1
                if k < n and tex[k] == "[":
                    end_b = tex.find("]", k)
                    if end_b != -1:
                        root_deg = tex[k + 1 : end_b]
                        k = end_b + 1
                        while k < n and tex[k] == " ":
                            k += 1
                        if k < n and tex[k] == "{":
                            rad, k = _extract_braced(tex, k)
                            out.append(
                                f'<sup class="mroot">{_render_tex_inner(root_deg)}</sup>√'
                                f'<span class="mrad">{_render_tex_inner(rad)}</span>'
                            )
                            i = k
                            continue
                if k < n and tex[k] == "{":
                    rad, k = _extract_braced(tex, k)
                    out.append(f'√<span class="mrad">{_render_tex_inner(rad)}</span>')
                    i = k
                    continue

            if cmd_name in ("text", "textrm", "mathrm", "operatorname"):
                k = j
                while k < n and tex[k] == " ":
                    k += 1
                if k < n and tex[k] == "{":
                    txt, k = _extract_braced(tex, k)
                    out.append(f'<span class="mrm">{_render_tex_inner(txt)}</span>')
                    i = k
                    continue

            if cmd_name in ("mathbf", "boldsymbol"):
                k = j
                while k < n and tex[k] == " ":
                    k += 1
                if k < n and tex[k] == "{":
                    txt, k = _extract_braced(tex, k)
                    out.append(f'<strong>{_render_tex_inner(txt)}</strong>')
                    i = k
                    continue

            if cmd_name in ("mathcal", "mathbb"):
                k = j
                while k < n and tex[k] == " ":
                    k += 1
                if k < n and tex[k] == "{":
                    txt, k = _extract_braced(tex, k)
                    cls = "mcal" if cmd_name == "mathcal" else "mbb"
                    out.append(f'<span class="{cls}">{_render_tex_inner(txt)}</span>')
                    i = k
                    continue

            if cmd_name in ("hat", "widehat", "bar", "overline", "tilde", "widetilde", "vec"):
                accent_map = {
                    "hat": "\u0302",
                    "widehat": "\u0302",
                    "bar": "\u0304",
                    "overline": "\u0304",
                    "tilde": "\u0303",
                    "widetilde": "\u0303",
                    "vec": "\u20d7",
                }
                k = j
                while k < n and tex[k] == " ":
                    k += 1
                if k < n and tex[k] == "{":
                    arg, k = _extract_braced(tex, k)
                    rendered = _render_tex_inner(arg)
                    if cmd_name in ("overline", "widehat", "widetilde") and len(arg) > 2:
                        out.append(f'<span class="mover">{rendered}</span>')
                    else:
                        out.append(f"{rendered}{accent_map[cmd_name]}")
                    i = k
                    continue

            if cmd in GREEK_AND_SYMBOLS:
                out.append(GREEK_AND_SYMBOLS[cmd])
                i = j
                continue

            if cmd_name in ROMAN_OPERATORS:
                out.append(f'<span class="mop">{html.escape(cmd_name)}</span>')
                i = j
                continue

            out.append(html.escape(cmd_name))
            i = j
            continue

        if ch == "-":
            out.append("−")
        else:
            out.append(html.escape(ch))
        i += 1

    return "".join(out)


def render_math_html(tex: str, display: bool) -> str:
    clean_tex = tex.strip()
    rendered = _render_tex_inner(clean_tex)
    escaped_attr = html.escape(clean_tex, quote=True)
    if display:
        return f'<div class="math-block" data-tex="{escaped_attr}">{rendered}</div>'
    return f'<span class="math-inline" data-tex="{escaped_attr}">{rendered}</span>'


# ---------------------------------------------------------------------------
# Zero-dependency syntax highlighter for fenced code blocks
# ---------------------------------------------------------------------------

RUST_KW = {
    "fn", "pub", "let", "mut", "const", "struct", "enum", "impl", "trait", "for",
    "in", "if", "else", "match", "return", "use", "mod", "crate", "self", "Self",
    "where", "type", "true", "false", "Some", "None", "Ok", "Err", "async", "await",
}

SH_KW = {
    "cargo", "nmrassign", "python3", "git", "curl", "export", "cd", "mkdir",
}


def highlight_code(code: str, lang: str) -> str:
    """Apply lightweight, deterministic token highlighting without external deps."""
    norm = lang.lower().strip()
    escaped_lines: List[str] = []
    for line in code.split("\n"):
        if norm in ("bash", "sh", "zsh", "shell", "console"):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                escaped_lines.append(f'<span class="tok-c">{html.escape(line)}</span>')
                continue
            # Highlight CLI flags and commands safely
            parts = re.split(r'(\s+|"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\')', line)
            line_out: List[str] = []
            for idx, part in enumerate(parts):
                if not part:
                    continue
                if part.startswith(('"', "'")):
                    line_out.append(f'<span class="tok-s">{html.escape(part)}</span>')
                elif part.startswith("--") or (part.startswith("-") and len(part) > 1 and part[1].isalpha()):
                    line_out.append(f'<span class="tok-f">{html.escape(part)}</span>')
                elif part in SH_KW or (idx == 0 and part == "$"):
                    line_out.append(f'<span class="tok-k">{html.escape(part)}</span>')
                else:
                    line_out.append(html.escape(part))
            escaped_lines.append("".join(line_out))
        elif norm == "json":
            def _json_repl(m: re.Match[str]) -> str:
                tok = m.group(0)
                if tok.startswith('"'):
                    if tok.endswith(":"):
                        key_part = html.escape(tok[:-1])
                        return f'<span class="tok-f">{key_part}</span>:'
                    return f'<span class="tok-s">{html.escape(tok)}</span>'
                return f'<span class="tok-k">{html.escape(tok)}</span>'

            escaped = html.escape(line)
            # Re-tokenize raw line then escape non-matched segments
            pos = 0
            seg_out: List[str] = []
            for m in re.finditer(r'"(?:[^"\\]|\\.)*"\s*:|"(?:[^"\\]|\\.)*"|\b(?:true|false|null|-?\d+(?:\.\d+)?)\b', line):
                seg_out.append(html.escape(line[pos : m.start()]))
                raw_tok = m.group(0)
                if raw_tok.rstrip().endswith(":") and raw_tok.startswith('"'):
                    colon_idx = raw_tok.rfind(":")
                    seg_out.append(
                        f'<span class="tok-f">{html.escape(raw_tok[:colon_idx])}</span>'
                        f'{html.escape(raw_tok[colon_idx:])}'
                    )
                elif raw_tok.startswith('"'):
                    seg_out.append(f'<span class="tok-s">{html.escape(raw_tok)}</span>')
                else:
                    seg_out.append(f'<span class="tok-k">{html.escape(raw_tok)}</span>')
                pos = m.end()
            seg_out.append(html.escape(line[pos:]))
            escaped_lines.append("".join(seg_out))
        elif norm in ("rust", "rs"):
            comment_idx = line.find("//")
            code_part = line if comment_idx == -1 else line[:comment_idx]
            comment_part = "" if comment_idx == -1 else line[comment_idx:]
            pos = 0
            seg_out = []
            for m in re.finditer(r'"(?:[^"\\]|\\.)*"|\b[A-Za-z_][A-Za-z0-9_]*\b', code_part):
                seg_out.append(html.escape(code_part[pos : m.start()]))
                tok = m.group(0)
                if tok.startswith('"'):
                    seg_out.append(f'<span class="tok-s">{html.escape(tok)}</span>')
                elif tok in RUST_KW:
                    seg_out.append(f'<span class="tok-k">{html.escape(tok)}</span>')
                elif tok[0].isupper():
                    seg_out.append(f'<span class="tok-t">{html.escape(tok)}</span>')
                else:
                    seg_out.append(html.escape(tok))
                pos = m.end()
            seg_out.append(html.escape(code_part[pos:]))
            if comment_part:
                seg_out.append(f'<span class="tok-c">{html.escape(comment_part)}</span>')
            escaped_lines.append("".join(seg_out))
        else:
            escaped_lines.append(html.escape(line))
    return "\n".join(escaped_lines)


# ---------------------------------------------------------------------------
# Markdown parser (inline & block)
# ---------------------------------------------------------------------------

def rewrite_internal_href(raw_href: str) -> str:
    """Rewrite internal ``*.md`` and ``../assets/*`` links for static manual output."""
    parsed = urlsplit(raw_href)
    if parsed.scheme or parsed.netloc:
        return raw_href
    path_part = parsed.path
    if path_part.startswith("../assets/"):
        rel_asset = path_part[len("../assets/") :]
        if rel_asset and ".." not in Path(rel_asset).parts and not rel_asset.startswith("/"):
            path_part = f"assets/{rel_asset}"
    elif path_part in ("../demo", "../demo/", "../demo/index.html"):
        path_part = "demo/index.html"
    elif path_part.endswith(".md"):
        path_part = path_part[:-3] + ".html"
    if parsed.fragment:
        return f"{path_part}#{parsed.fragment}"
    return path_part


def render_inline_markdown(
    text: str,
    md_name: str,
    line_no: int,
    links_out: Optional[List[LinkRef]] = None,
) -> str:
    """Render inline code, math, links, bold, and italics safely."""
    stash: List[str] = []

    def put_token(rendered_html: str) -> str:
        idx = len(stash)
        stash.append(rendered_html)
        return f"\x00TOK{idx}END\x00"

    # 1. Extract multi-backtick or single-backtick inline code spans first
    def repl_code(m: re.Match[str]) -> str:
        inner = m.group(2)
        if inner.startswith(" ") and inner.endswith(" ") and len(inner) > 2:
            inner = inner[1:-1]
        return put_token(f"<code>{html.escape(inner)}</code>")

    work = re.sub(r"(`+)(.+?)\1", repl_code, text)

    # 2. Extract inline display/inline math ($...$) where $ is not followed/preceded by space-digit currency
    def repl_math(m: re.Match[str]) -> str:
        expr = m.group(1)
        return put_token(render_math_html(expr, display=False))

    work = re.sub(r"(?<!\\)\$([^\$\n]+?)(?<!\\)\$", repl_math, work)

    # 2b. Extract Markdown images ![alt](src)
    def repl_image(m: re.Match[str]) -> str:
        raw_alt = m.group(1)
        raw_src = m.group(2).strip()
        plain_alt = strip_markdown_inline(re.sub(r"\x00TOK\d+END\x00", "code", raw_alt))
        if links_out is not None:
            links_out.append(
                LinkRef(
                    source_md=md_name,
                    line_no=line_no,
                    raw_target=raw_src,
                    label_text=plain_alt,
                )
            )
        final_src = html.escape(rewrite_internal_href(raw_src), quote=True)
        alt_escaped = html.escape(plain_alt, quote=True)
        return put_token(
            f'<img class="manual-img" src="{final_src}" alt="{alt_escaped}" loading="lazy" decoding="async">'
        )

    work = re.sub(r"!\[([^\]]*)\]\(([^)\s]+)\)", repl_image, work)

    # 3. Extract Markdown links [label](href)
    def repl_link(m: re.Match[str]) -> str:
        raw_label = m.group(1)
        raw_href = m.group(2).strip()
        if links_out is not None:
            # Recover plain label text (expanding any stashed code/math minimally)
            plain_label = re.sub(r"\x00TOK\d+END\x00", "code", raw_label)
            links_out.append(
                LinkRef(
                    source_md=md_name,
                    line_no=line_no,
                    raw_target=raw_href,
                    label_text=strip_markdown_inline(plain_label),
                )
            )
        final_href = html.escape(rewrite_internal_href(raw_href), quote=True)
        # Format label bold/italics/escaped text while preserving stashed tokens
        label_escaped = _format_emphasis(html.escape(raw_label))
        label_restored = _restore_tokens(label_escaped, stash)
        ext_attrs = ""
        if raw_href.startswith(("http://", "https://")):
            ext_attrs = ' target="_blank" rel="noopener noreferrer"'
        return put_token(f'<a href="{final_href}"{ext_attrs}>{label_restored}</a>')

    work = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", repl_link, work)

    # 4. Escape remaining HTML characters and apply bold / italic emphasis
    work = html.escape(work)
    work = _format_emphasis(work)

    # 5. Restore all stashed tokens
    return _restore_tokens(work, stash)


def _format_emphasis(escaped_text: str) -> str:
    out = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped_text)
    out = re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", r"<em>\1</em>", out)
    return out


def _restore_tokens(text: str, stash: Sequence[str]) -> str:
    def _sub(m: re.Match[str]) -> str:
        return stash[int(m.group(1))]

    # Apply up to twice in case a link restored nested code/math tokens
    for _ in range(2):
        if "\x00TOK" not in text:
            break
        text = re.sub(r"\x00TOK(\d+)END\x00", _sub, text)
    return text


def split_table_row(line: str) -> List[str]:
    """Split a Markdown pipe-table row while ignoring ``|`` inside code or math."""
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith(r"\|"):
        s = s[:-1]

    cells: List[str] = []
    cur: List[str] = []
    in_code = False
    code_ticks = 0
    in_math = False
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch == "\\" and i + 1 < n:
            if s[i + 1] == "|" and not in_code and not in_math:
                cur.append("|")
            else:
                cur.append(s[i : i + 2])
            i += 2
            continue
        if ch == "`" and not in_math:
            j = i
            while j < n and s[j] == "`":
                j += 1
            run = j - i
            if not in_code:
                in_code = True
                code_ticks = run
            elif run == code_ticks:
                in_code = False
                code_ticks = 0
            cur.append(s[i:j])
            i = j
            continue
        if ch == "$" and not in_code:
            in_math = not in_math
            cur.append(ch)
            i += 1
            continue
        if ch == "|" and not in_code and not in_math:
            cells.append("".join(cur).strip())
            cur = []
            i += 1
            continue
        cur.append(ch)
        i += 1
    cells.append("".join(cur).strip())
    return cells


def is_table_separator(line: str) -> bool:
    s = line.strip()
    if "|" not in s:
        return False
    cells = split_table_row(s)
    if not cells:
        return False
    return all(bool(re.match(r"^:?-{3,}:?$", c.strip())) for c in cells)


CALLOUT_TITLES: Dict[str, Tuple[str, str]] = {
    "NOTE": ("Note", "ℹ️"),
    "TIP": ("Tip", "💡"),
    "IMPORTANT": ("Important", "⚡"),
    "WARNING": ("Warning", "⚠️"),
    "CAUTION": ("Caution", "🛑"),
}


def parse_markdown_blocks(
    lines: Sequence[str],
    md_name: str,
    base_line: int,
    used_slugs: Set[str],
    headings_out: List[HeadingItem],
    links_out: List[LinkRef],
    search_sections: Optional[List[Tuple[str, str, List[str]]]] = None,
) -> str:
    """Parse Markdown lines into semantic HTML and collect headings, links & search chunks."""
    out_html: List[str] = []
    i = 0
    n = len(lines)

    def append_search_text(plain: str) -> None:
        if search_sections is not None and plain:
            if not search_sections:
                search_sections.append(("", "", []))
            search_sections[-1][2].append(plain)

    while i < n:
        line = lines[i]
        stripped = line.strip()
        abs_line = base_line + i

        if not stripped:
            i += 1
            continue

        # 1. Fenced code block (3+ backticks)
        fence_m = re.match(r"^(\s*)(`{3,})([A-Za-z0-9_+-]*)\s*$", line)
        if fence_m:
            ticks = fence_m.group(2)
            lang = fence_m.group(3).strip()
            code_lines: List[str] = []
            i += 1
            while i < n and not re.match(rf"^\s*{re.escape(ticks)}\s*$", lines[i]):
                code_lines.append(lines[i])
                i += 1
            if i < n:
                i += 1
            raw_code = "\n".join(code_lines)
            lang_badge = html.escape(lang or "text")
            highlighted = highlight_code(raw_code, lang)
            out_html.append(
                f'<div class="code-card">'
                f'<div class="code-header"><span class="code-lang">{lang_badge}</span>'
                f'<button type="button" class="copy-btn" data-copy="1">Copy</button></div>'
                f'<pre><code class="language-{lang_badge}">{highlighted}</code></pre>'
                f"</div>"
            )
            append_search_text(raw_code)
            continue

        # 2. Display math block $$ ... $$
        if stripped.startswith("$$"):
            if stripped.endswith("$$") and len(stripped) > 4:
                expr = stripped[2:-2]
                out_html.append(render_math_html(expr, display=True))
                append_search_text(strip_markdown_inline(expr))
                i += 1
                continue
            math_lines: List[str] = []
            first_rem = stripped[2:].strip()
            if first_rem:
                math_lines.append(first_rem)
            i += 1
            while i < n:
                cur_s = lines[i].strip()
                if cur_s.endswith("$$"):
                    before = cur_s[:-2].strip()
                    if before:
                        math_lines.append(before)
                    i += 1
                    break
                math_lines.append(lines[i])
                i += 1
            expr = "\n".join(math_lines)
            out_html.append(render_math_html(expr, display=True))
            append_search_text(strip_markdown_inline(expr))
            continue

        # 3. ATX Heading (# .. ######)
        head_m = re.match(r"^(#{1,6})\s+(.+?)\s*$", stripped)
        if head_m:
            level = len(head_m.group(1))
            raw_title = head_m.group(2)
            slug = slugify_heading(raw_title, used_slugs)
            title_html = render_inline_markdown(raw_title, md_name, abs_line, links_out)
            title_text = strip_markdown_inline(raw_title)
            headings_out.append(
                HeadingItem(
                    level=level,
                    title_html=title_html,
                    title_text=title_text,
                    slug=slug,
                    line_no=abs_line,
                )
            )
            out_html.append(
                f'<h{level} id="{slug}">{title_html}'
                f'<a class="heading-anchor" href="#{slug}" title="Link to section">#</a>'
                f"</h{level}>"
            )
            if search_sections is not None and level <= 3:
                search_sections.append((slug, title_text, []))
            else:
                append_search_text(title_text)
            i += 1
            continue

        # 4. Horizontal rule
        if re.match(r"^(?:-{3,}|\*{3,}|_{3,})$", stripped):
            out_html.append("<hr>")
            i += 1
            continue

        # 5. Blockquote or GitHub-style Callout (> [!NOTE] ...)
        if stripped.startswith(">"):
            bq_start = abs_line
            bq_lines: List[str] = []
            while i < n and lines[i].lstrip().startswith(">"):
                raw_bq = lines[i].lstrip()[1:]
                if raw_bq.startswith(" "):
                    raw_bq = raw_bq[1:]
                bq_lines.append(raw_bq)
                i += 1
            alert_m = re.match(
                r"^\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*(.*)$",
                bq_lines[0].strip(),
            )
            if alert_m:
                kind = alert_m.group(1)
                inline_head = alert_m.group(2).strip()
                inner_lines = ([inline_head] if inline_head else []) + bq_lines[1:]
                label, icon = CALLOUT_TITLES[kind]
                inner_html = parse_markdown_blocks(
                    inner_lines,
                    md_name,
                    bq_start,
                    used_slugs,
                    headings_out,
                    links_out,
                    search_sections,
                )
                out_html.append(
                    f'<aside class="callout callout-{kind.lower()}">'
                    f'<div class="callout-title"><span class="callout-icon">{icon}</span>{label}</div>'
                    f'<div class="callout-body">{inner_html}</div>'
                    f"</aside>"
                )
            else:
                inner_html = parse_markdown_blocks(
                    bq_lines,
                    md_name,
                    bq_start,
                    used_slugs,
                    headings_out,
                    links_out,
                    search_sections,
                )
                out_html.append(f"<blockquote>{inner_html}</blockquote>")
            continue

        # 6. Pipe table
        if "|" in stripped and i + 1 < n and is_table_separator(lines[i + 1]):
            headers = split_table_row(lines[i])
            sep_cells = split_table_row(lines[i + 1])
            aligns: List[str] = []
            for sc in sep_cells:
                sc_t = sc.strip()
                if sc_t.startswith(":") and sc_t.endswith(":"):
                    aligns.append("center")
                elif sc_t.endswith(":"):
                    aligns.append("right")
                else:
                    aligns.append("left")
            i += 2
            body_rows: List[Tuple[int, List[str]]] = []
            while i < n and lines[i].strip() and "|" in lines[i]:
                body_rows.append((base_line + i, split_table_row(lines[i])))
                i += 1

            th_cells = []
            for col_idx, h_cell in enumerate(headers):
                align = aligns[col_idx] if col_idx < len(aligns) else "left"
                th_html = render_inline_markdown(h_cell, md_name, abs_line, links_out)
                th_cells.append(f'<th style="text-align:{align}">{th_html}</th>')
                append_search_text(strip_markdown_inline(h_cell))

            tr_rows = []
            for r_line, row in body_rows:
                tds = []
                for col_idx in range(len(headers)):
                    val = row[col_idx] if col_idx < len(row) else ""
                    align = aligns[col_idx] if col_idx < len(aligns) else "left"
                    td_html = render_inline_markdown(val, md_name, r_line, links_out)
                    tds.append(f'<td style="text-align:{align}">{td_html}</td>')
                    append_search_text(strip_markdown_inline(val))
                tr_rows.append("<tr>" + "".join(tds) + "</tr>")

            out_html.append(
                '<div class="table-wrap"><table>'
                f"<thead><tr>{''.join(th_cells)}</tr></thead>"
                f"<tbody>{''.join(tr_rows)}</tbody>"
                "</table></div>"
            )
            continue

        # 7. Unordered or Ordered Lists (supporting indented continuation & nested sub-lists)
        list_m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", line)
        if list_m:
            list_html, i = _parse_list(
                lines,
                i,
                base_line,
                md_name,
                used_slugs,
                headings_out,
                links_out,
                search_sections,
            )
            out_html.append(list_html)
            continue

        # 7b. Standalone Markdown figure ![alt](src) (with optional italic caption on next line)
        fig_m = re.match(r"^!\[([^\]]*)\]\(([^)\s]+)\)$", stripped)
        if fig_m:
            raw_alt = fig_m.group(1)
            raw_src = fig_m.group(2).strip()
            plain_alt = strip_markdown_inline(raw_alt)
            links_out.append(
                LinkRef(
                    source_md=md_name,
                    line_no=abs_line,
                    raw_target=raw_src,
                    label_text=plain_alt,
                )
            )
            final_src = html.escape(rewrite_internal_href(raw_src), quote=True)
            alt_escaped = html.escape(plain_alt, quote=True)
            append_search_text(plain_alt)
            i += 1
            caption_html = ""
            if i < n:
                next_s = lines[i].strip()
                if len(next_s) >= 3 and next_s.startswith("*") and next_s.endswith("*") and not next_s.startswith("**"):
                    cap_inner = next_s[1:-1].strip()
                    cap_rendered = render_inline_markdown(cap_inner, md_name, base_line + i, links_out)
                    caption_html = f"<figcaption>{cap_rendered}</figcaption>"
                    append_search_text(strip_markdown_inline(cap_inner))
                    i += 1
            out_html.append(
                f'<figure class="manual-figure">'
                f'<img class="manual-img" src="{final_src}" alt="{alt_escaped}" loading="lazy" decoding="async">'
                f"{caption_html}"
                f"</figure>"
            )
            continue

        # 8. Paragraph
        para_lines: List[Tuple[int, str]] = []
        while i < n:
            cur = lines[i]
            cur_s = cur.strip()
            if not cur_s:
                break
            if (
                re.match(r"^(\s*)(`{3,})", cur)
                or cur_s.startswith("$$")
                or re.match(r"^#{1,6}\s+", cur_s)
                or re.match(r"^(?:-{3,}|\*{3,}|_{3,})$", cur_s)
                or cur_s.startswith(">")
                or ("|" in cur_s and i + 1 < n and is_table_separator(lines[i + 1]))
                or re.match(r"^(\s*)([-*]|\d+\.)\s+", cur)
                or re.match(r"^!\[([^\]]*)\]\(([^)\s]+)\)$", cur_s)
            ):
                break
            para_lines.append((base_line + i, cur_s))
            i += 1

        rendered_parts = [
            render_inline_markdown(txt, md_name, ln, links_out) for ln, txt in para_lines
        ]
        for _, txt in para_lines:
            append_search_text(strip_markdown_inline(txt))
        out_html.append(f"<p>{' '.join(rendered_parts)}</p>")

    return "\n".join(out_html)


def _parse_list(
    lines: Sequence[str],
    start_idx: int,
    base_line: int,
    md_name: str,
    used_slugs: Set[str],
    headings_out: List[HeadingItem],
    links_out: List[LinkRef],
    search_sections: Optional[List[Tuple[str, str, List[str]]]],
) -> Tuple[str, int]:
    first_m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", lines[start_idx])
    assert first_m is not None
    base_indent = len(first_m.group(1).expandtabs(4))
    is_ordered = first_m.group(2).endswith(".")
    tag = "ol" if is_ordered else "ul"

    items_html: List[str] = []
    i = start_idx
    n = len(lines)

    while i < n:
        cur = lines[i]
        if not cur.strip():
            # Look ahead to see if another sibling item follows a single blank line
            k = i + 1
            while k < n and not lines[k].strip():
                k += 1
            if k < n:
                next_m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", lines[k])
                if next_m:
                    next_indent = len(next_m.group(1).expandtabs(4))
                    next_ordered = next_m.group(2).endswith(".")
                    if next_indent == base_indent and next_ordered == is_ordered:
                        i = k
                        continue
            break

        item_m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", cur)
        if not item_m:
            break
        indent = len(item_m.group(1).expandtabs(4))
        item_ordered = item_m.group(2).endswith(".")
        if indent < base_indent or (indent == base_indent and item_ordered != is_ordered):
            break
        if indent > base_indent:
            # Should have been consumed by preceding item, break defensively
            break

        head_line_no = base_line + i
        lead_text_lines: List[Tuple[int, str]] = [(head_line_no, item_m.group(3).strip())]
        sub_blocks: List[str] = []
        i += 1

        while i < n:
            sub_line = lines[i]
            sub_s = sub_line.strip()
            if not sub_s:
                # Peek past blank line for indented continuation block
                k = i + 1
                while k < n and not lines[k].strip():
                    k += 1
                if k < n and len(lines[k].expandtabs(4)) - len(lines[k].lstrip()) >= base_indent + 2:
                    i = k
                    continue
                break

            sub_indent = len(sub_line.expandtabs(4)) - len(sub_line.lstrip())
            if sub_indent < base_indent + 2:
                break

            nested_list_m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", sub_line)
            if nested_list_m:
                nested_html, i = _parse_list(
                    lines,
                    i,
                    base_line,
                    md_name,
                    used_slugs,
                    headings_out,
                    links_out,
                    search_sections,
                )
                sub_blocks.append(nested_html)
                continue

            if re.match(r"^(\s*)(`{3,})", sub_line) or sub_s.startswith("$$"):
                # Delegate indented code/math block to block parser
                block_slice: List[str] = []
                block_start = base_line + i
                if sub_s.startswith("`"):
                    ticks_m = re.match(r"^\s*(`{3,})", sub_line)
                    ticks = ticks_m.group(1) if ticks_m else "```"
                    block_slice.append(sub_line[min(len(sub_line) - len(sub_line.lstrip()), base_indent + 2) :])
                    i += 1
                    while i < n:
                        ln_raw = lines[i]
                        dedented = ln_raw[min(len(ln_raw) - len(ln_raw.lstrip()), base_indent + 2) :]
                        block_slice.append(dedented)
                        i += 1
                        if re.match(rf"^\s*{re.escape(ticks)}\s*$", ln_raw):
                            break
                else:
                    block_slice.append(sub_s)
                    i += 1
                    if not (sub_s.endswith("$$") and len(sub_s) > 4):
                        while i < n:
                            block_slice.append(lines[i].strip())
                            end_hit = lines[i].strip().endswith("$$")
                            i += 1
                            if end_hit:
                                break
                sub_blocks.append(
                    parse_markdown_blocks(
                        block_slice,
                        md_name,
                        block_start,
                        used_slugs,
                        headings_out,
                        links_out,
                        search_sections,
                    )
                )
                continue

            lead_text_lines.append((base_line + i, sub_s))
            i += 1

        lead_html = " ".join(
            render_inline_markdown(txt, md_name, ln, links_out)
            for ln, txt in lead_text_lines
        )
        if search_sections is not None:
            if not search_sections:
                search_sections.append(("", "", []))
            for _, txt in lead_text_lines:
                search_sections[-1][2].append(strip_markdown_inline(txt))
        items_html.append(
            "<li>" + lead_html + (("\n" + "\n".join(sub_blocks)) if sub_blocks else "") + "</li>"
        )

    return f"<{tag}>\n" + "\n".join(items_html) + f"\n</{tag}>", i


# ---------------------------------------------------------------------------
# Static assets (CSS & JS)
# ---------------------------------------------------------------------------

MANUAL_CSS = r""":root {
  --bg: #f8fafc;
  --bg-elev: #ffffff;
  --bg-subtle: #f1f5f9;
  --bg-code: #0f172a;
  --fg: #0f172a;
  --fg-muted: #475569;
  --fg-soft: #64748b;
  --border: #e2e8f0;
  --accent: #0284c7;
  --accent-soft: rgba(2, 132, 199, 0.10);
  --accent-strong: #0369a1;
  --inline-code-bg: #e2e8f0;
  --inline-code-fg: #0f172a;
  --shadow: 0 4px 18px rgba(15, 23, 42, 0.06);
  --note-c: #0284c7;
  --tip-c: #059669;
  --imp-c: #7c3aed;
  --warn-c: #d97706;
  --caut-c: #dc2626;
}

[data-theme="dark"] {
  --bg: #0b1120;
  --bg-elev: #111827;
  --bg-subtle: #1e293b;
  --bg-code: #090d16;
  --fg: #e2e8f0;
  --fg-muted: #94a3b8;
  --fg-soft: #64748b;
  --border: #1e293b;
  --accent: #38bdf8;
  --accent-soft: rgba(56, 189, 248, 0.13);
  --accent-strong: #7dd3fc;
  --inline-code-bg: #1e293b;
  --inline-code-fg: #e2e8f0;
  --shadow: 0 8px 28px rgba(0, 0, 0, 0.42);
  --note-c: #38bdf8;
  --tip-c: #34d399;
  --imp-c: #a78bfa;
  --warn-c: #fbbf24;
  --caut-c: #f87171;
}

*, *::before, *::after { box-sizing: border-box; }
html { scroll-behavior: smooth; font-size: 16px; -webkit-text-size-adjust: 100%; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--fg);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  line-height: 1.65;
  overflow-x: hidden;
}

a { color: var(--accent); text-decoration: none; }
a:hover { color: var(--accent-strong); text-decoration: underline; }

button, .menu-btn, .theme-btn, .search-trigger, .demo-btn, .copy-btn, .nav-link, .chapter-link, .toc-link, .pager-card, .search-hit {
  touch-action: manipulation;
}

.topbar {
  position: sticky;
  top: 0;
  z-index: 40;
  min-height: calc(3.6rem + env(safe-area-inset-top, 0px));
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  padding: env(safe-area-inset-top, 0px) calc(1.25rem + env(safe-area-inset-right, 0px)) 0 calc(1.25rem + env(safe-area-inset-left, 0px));
  background: var(--bg-elev);
  border-bottom: 1px solid var(--border);
}

.brand-wrap { display: flex; align-items: center; gap: 0.75rem; min-width: 0; }
.menu-btn {
  display: none;
  min-width: 44px;
  min-height: 44px;
  background: transparent;
  border: 1px solid var(--border);
  color: var(--fg);
  border-radius: 8px;
  padding: 0.3rem 0.55rem;
  cursor: pointer;
  font-size: 1.1rem;
  line-height: 1;
}
.brand-link {
  font-weight: 700;
  font-size: 1.05rem;
  color: var(--fg);
  white-space: nowrap;
}
.brand-link:hover { text-decoration: none; color: var(--accent); }
.version-pill {
  font-size: 0.75rem;
  font-weight: 600;
  padding: 0.15rem 0.55rem;
  border-radius: 999px;
  background: var(--accent-soft);
  color: var(--accent);
  white-space: nowrap;
}

.topbar-actions { display: flex; align-items: center; gap: 0.6rem; flex-shrink: 0; }
.demo-label-short, .search-label-short { display: none; }
.search-trigger {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  background: var(--bg-subtle);
  color: var(--fg-muted);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.38rem 0.75rem;
  font-size: 0.875rem;
  cursor: pointer;
  min-width: 13rem;
  justify-content: space-between;
}
.search-trigger:hover { border-color: var(--accent); color: var(--fg); }
.kbd-hint {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.72rem;
  padding: 0.08rem 0.38rem;
  border-radius: 4px;
  border: 1px solid var(--border);
  background: var(--bg-elev);
}
.theme-btn {
  background: var(--bg-subtle);
  color: var(--fg);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.38rem 0.68rem;
  cursor: pointer;
  font-size: 0.85rem;
  font-weight: 600;
}
.theme-btn:hover { border-color: var(--accent); }
.demo-btn {
  background: var(--accent-soft);
  color: var(--accent);
  border: 1px solid var(--accent);
  border-radius: 8px;
  padding: 0.36rem 0.72rem;
  font-size: 0.84rem;
  font-weight: 600;
  white-space: nowrap;
}
.demo-btn:hover { text-decoration: none; filter: brightness(1.06); }

.layout {
  max-width: 92rem;
  margin: 0 auto;
  display: grid;
  grid-template-columns: 17.5rem minmax(0, 1fr) 15.5rem;
  gap: 2rem;
  padding: 0 calc(1.25rem + env(safe-area-inset-right, 0px)) 0 calc(1.25rem + env(safe-area-inset-left, 0px));
}

.sidebar {
  position: sticky;
  top: 3.6rem;
  height: calc(100vh - 3.6rem);
  height: calc(100dvh - 3.6rem);
  overflow-y: auto;
  padding: 1.5rem 0.75rem 2rem 0;
  border-right: 1px solid var(--border);
}
.sidebar-backdrop {
  position: fixed;
  inset: 0;
  top: calc(3.6rem + env(safe-area-inset-top, 0px));
  z-index: 55;
  background: rgba(15, 23, 42, 0.52);
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.2s ease;
}
.sidebar-backdrop.open {
  opacity: 1;
  pointer-events: auto;
}
@media (min-width: 861px) {
  .sidebar-backdrop { display: none; }
}
.sidebar-caption {
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--fg-soft);
  font-weight: 700;
  margin: 0 0 0.65rem 0.6rem;
}
.chapter-list { list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 0.22rem; }
.chapter-link, .nav-link {
  display: flex;
  align-items: baseline;
  gap: 0.5rem;
  padding: 0.48rem 0.65rem;
  border-radius: 8px;
  color: var(--fg-muted);
  font-size: 0.9rem;
  line-height: 1.35;
}
.chapter-link:hover, .nav-link:hover { background: var(--bg-subtle); color: var(--fg); text-decoration: none; }
.chapter-link.active, .nav-link.active {
  background: var(--accent-soft);
  color: var(--accent);
  font-weight: 600;
}
.chap-num {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.75rem;
  opacity: 0.8;
  flex-shrink: 0;
}

.content {
  min-width: 0;
  width: 100%;
  max-width: 54rem;
  padding: 2rem 0 calc(4rem + env(safe-area-inset-bottom, 0px)) 0;
}
.content h1 {
  font-size: 2rem;
  line-height: 1.22;
  margin: 0 0 1.1rem 0;
  letter-spacing: -0.02em;
}
.content h2 {
  font-size: 1.42rem;
  line-height: 1.3;
  margin: 2.4rem 0 0.8rem 0;
  padding-bottom: 0.35rem;
  border-bottom: 1px solid var(--border);
  scroll-margin-top: 4.6rem;
}
.content h3 {
  font-size: 1.14rem;
  margin: 1.75rem 0 0.55rem 0;
  scroll-margin-top: 4.6rem;
}
.content h4 { font-size: 1rem; margin: 1.3rem 0 0.4rem 0; scroll-margin-top: 4.6rem; }
.heading-anchor {
  margin-left: 0.45rem;
  opacity: 0;
  font-weight: 400;
  color: var(--fg-soft);
}
.content h1:hover .heading-anchor,
.content h2:hover .heading-anchor,
.content h3:hover .heading-anchor,
.content h4:hover .heading-anchor { opacity: 1; text-decoration: none; }

.content p, .content li, blockquote, .callout { overflow-wrap: break-word; }
.content p { margin: 0.75rem 0; }
.content ul, .content ol { padding-left: 1.45rem; margin: 0.65rem 0; }
.content li { margin: 0.32rem 0; }
.content hr { border: 0; border-top: 1px solid var(--border); margin: 2rem 0; }

code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 0.86em;
  background: var(--inline-code-bg);
  color: var(--inline-code-fg);
  padding: 0.14rem 0.36rem;
  border-radius: 5px;
}
:not(pre) > code {
  overflow-wrap: break-word;
  word-break: break-word;
}

.code-card {
  margin: 1rem 0;
  max-width: 100%;
  border-radius: 10px;
  overflow: hidden;
  background: var(--bg-code);
  border: 1px solid #1e293b;
  box-shadow: var(--shadow);
}
.code-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.38rem 0.85rem;
  background: rgba(255, 255, 255, 0.04);
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  font-size: 0.75rem;
  color: #94a3b8;
}
.code-lang { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; text-transform: uppercase; letter-spacing: 0.05em; }
.copy-btn {
  background: rgba(255, 255, 255, 0.08);
  color: #e2e8f0;
  border: 1px solid rgba(255, 255, 255, 0.14);
  border-radius: 5px;
  padding: 0.22rem 0.6rem;
  font-size: 0.74rem;
  cursor: pointer;
}
.copy-btn:hover { background: rgba(56, 189, 248, 0.25); }
.code-card pre {
  margin: 0;
  padding: 0.9rem 1rem;
  max-width: 100%;
  overflow-x: auto;
  -webkit-overflow-scrolling: touch;
  color: #e2e8f0;
  font-size: 0.85rem;
  line-height: 1.55;
}
.code-card pre code {
  background: transparent;
  color: inherit;
  padding: 0;
  font-size: inherit;
  overflow-wrap: normal;
  word-break: normal;
}
.tok-c { color: #64748b; font-style: italic; }
.tok-k { color: #38bdf8; font-weight: 600; }
.tok-s { color: #a7f3d0; }
.tok-f { color: #fcd34d; }
.tok-t { color: #c4b5fd; }

.table-wrap {
  margin: 1.1rem 0;
  width: 100%;
  max-width: 100%;
  overflow-x: auto;
  -webkit-overflow-scrolling: touch;
  border: 1px solid var(--border);
  border-radius: 10px;
  background: var(--bg-elev);
}
table { width: 100%; border-collapse: collapse; font-size: 0.89rem; }
th, td { padding: 0.6rem 0.8rem; border-bottom: 1px solid var(--border); vertical-align: top; }
thead th { background: var(--bg-subtle); font-weight: 600; color: var(--fg); }
tbody tr:last-child td { border-bottom: none; }

img, figure svg, .manual-figure svg {
  max-width: 100%;
  height: auto;
}
.manual-figure {
  margin: 1.35rem 0;
  max-width: 100%;
  padding: 0.85rem;
  border: 1px solid var(--border);
  border-radius: 10px;
  background: var(--bg-elev);
  box-shadow: var(--shadow);
  text-align: center;
}
.manual-img {
  display: block;
  max-width: 100%;
  height: auto;
  margin: 0 auto;
  border-radius: 6px;
}
td .manual-img {
  margin-bottom: 0.45rem;
}
.manual-figure figcaption {
  margin-top: 0.65rem;
  font-size: 0.84rem;
  line-height: 1.45;
  color: var(--fg-muted);
  text-align: left;
}

blockquote {
  margin: 1rem 0;
  padding: 0.55rem 1rem;
  border-left: 4px solid var(--accent);
  background: var(--bg-subtle);
  border-radius: 0 8px 8px 0;
  color: var(--fg-muted);
}

.callout {
  margin: 1.15rem 0;
  max-width: 100%;
  padding: 0.85rem 1.05rem;
  border-left: 4px solid var(--note-c);
  background: var(--bg-elev);
  border-top: 1px solid var(--border);
  border-right: 1px solid var(--border);
  border-bottom: 1px solid var(--border);
  border-radius: 0 10px 10px 0;
  box-shadow: var(--shadow);
}
.callout-title {
  display: flex;
  align-items: center;
  gap: 0.45rem;
  font-weight: 700;
  font-size: 0.9rem;
  margin-bottom: 0.35rem;
  color: var(--note-c);
}
.callout-body > p:first-child { margin-top: 0; }
.callout-body > p:last-child { margin-bottom: 0; }
.callout-tip { border-left-color: var(--tip-c); }
.callout-tip .callout-title { color: var(--tip-c); }
.callout-important { border-left-color: var(--imp-c); }
.callout-important .callout-title { color: var(--imp-c); }
.callout-warning { border-left-color: var(--warn-c); }
.callout-warning .callout-title { color: var(--warn-c); }
.callout-caution { border-left-color: var(--caut-c); }
.callout-caution .callout-title { color: var(--caut-c); }

.math-inline {
  font-family: "STIX Two Math", "Cambria Math", "Latin Modern Math", Georgia, serif;
  font-size: 1.02em;
  white-space: nowrap;
}
.math-block {
  margin: 1.1rem 0;
  max-width: 100%;
  padding: 0.85rem 1.1rem;
  overflow-x: auto;
  -webkit-overflow-scrolling: touch;
  text-align: center;
  background: var(--bg-elev);
  border: 1px solid var(--border);
  border-radius: 10px;
  font-family: "STIX Two Math", "Cambria Math", "Latin Modern Math", Georgia, serif;
  font-size: 1.08rem;
}
.mfrac {
  display: inline-flex;
  flex-direction: column;
  vertical-align: middle;
  text-align: center;
  padding: 0 0.2em;
  font-size: 0.92em;
}
.mnum { border-bottom: 1px solid currentColor; padding: 0 0.18em 0.06em; }
.mden { padding: 0.06em 0.18em 0; }
.mrad { border-top: 1px solid currentColor; padding-top: 0.04em; }
.mrm, .mop { font-style: normal; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; font-size: 0.92em; }
.mover { border-top: 1px solid currentColor; }
.mmatrix { display: inline-flex; align-items: center; vertical-align: middle; }
.mmatrix table { border: none; font-size: 0.9em; width: auto; }
.mmatrix td { border: none; padding: 0.1rem 0.35rem; text-align: center; }

.toc-aside {
  position: sticky;
  top: 3.6rem;
  height: calc(100vh - 3.6rem);
  height: calc(100dvh - 3.6rem);
  overflow-y: auto;
  padding: 1.5rem 0 2rem 0.5rem;
  font-size: 0.82rem;
}
.toc-list { list-style: none; padding: 0; margin: 0; border-left: 1px solid var(--border); }
.toc-list li { margin: 0; }
.toc-link {
  display: block;
  padding: 0.28rem 0 0.28rem 0.8rem;
  color: var(--fg-muted);
  line-height: 1.35;
}
.toc-link.toc-h3 { padding-left: 1.45rem; font-size: 0.78rem; }
.toc-link:hover { color: var(--accent); text-decoration: none; }

.pager {
  margin-top: 3rem;
  padding-top: 1.5rem;
  border-top: 1px solid var(--border);
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
}
.pager-card {
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
  padding: 0.85rem 1rem;
  border: 1px solid var(--border);
  border-radius: 10px;
  background: var(--bg-elev);
}
.pager-card:hover { border-color: var(--accent); text-decoration: none; }
.pager-card.next { text-align: right; }
.pager-dir { font-size: 0.74rem; color: var(--fg-soft); text-transform: uppercase; letter-spacing: 0.06em; }
.pager-title { font-weight: 600; color: var(--accent); }

.footer-note {
  margin-top: 2rem;
  font-size: 0.78rem;
  color: var(--fg-soft);
}

/* Search modal */
.search-backdrop {
  position: fixed;
  inset: 0;
  z-index: 90;
  background: rgba(15, 23, 42, 0.58);
  display: none;
  align-items: flex-start;
  justify-content: center;
  padding: calc(4.5rem + env(safe-area-inset-top, 0px)) 1rem calc(1rem + env(safe-area-inset-bottom, 0px));
}
.search-backdrop.open { display: flex; }
.search-modal {
  width: 100%;
  max-width: 40rem;
  background: var(--bg-elev);
  border: 1px solid var(--border);
  border-radius: 12px;
  box-shadow: var(--shadow);
  overflow: hidden;
}
.search-input-row {
  display: flex;
  align-items: center;
  padding: 0.75rem 1rem;
  border-bottom: 1px solid var(--border);
  gap: 0.6rem;
}
.search-input {
  flex: 1;
  background: transparent;
  border: none;
  outline: none;
  color: var(--fg);
  font-size: 1rem;
}
.search-results {
  max-height: 24rem;
  overflow-y: auto;
  -webkit-overflow-scrolling: touch;
  list-style: none;
  margin: 0;
  padding: 0.4rem;
}
.search-hit {
  display: block;
  padding: 0.65rem 0.8rem;
  border-radius: 8px;
  color: var(--fg);
}
.search-hit:hover, .search-hit.selected {
  background: var(--accent-soft);
  text-decoration: none;
}
.hit-meta { font-size: 0.74rem; color: var(--accent); font-weight: 600; }
.hit-title { font-weight: 600; font-size: 0.93rem; margin: 0.1rem 0; }
.hit-snippet { font-size: 0.8rem; color: var(--fg-muted); }
.search-empty { padding: 1.25rem; text-align: center; color: var(--fg-soft); font-size: 0.88rem; }

@media (max-width: 1120px) {
  .layout { grid-template-columns: 16.5rem minmax(0, 1fr); }
  .toc-aside { display: none; }
}
@media (max-width: 860px) {
  .menu-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 44px;
    height: 44px;
    min-width: 44px;
    min-height: 44px;
    padding: 0;
    flex-shrink: 0;
  }
  .version-pill,
  .kbd-hint {
    display: none;
  }
  .search-trigger {
    min-width: auto;
    min-height: 44px;
    padding: 0.4rem 0.72rem;
  }
  .theme-btn,
  .demo-btn {
    min-height: 44px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
  }
  .layout {
    grid-template-columns: minmax(0, 1fr);
    padding-left: calc(1rem + env(safe-area-inset-left, 0px));
    padding-right: calc(1rem + env(safe-area-inset-right, 0px));
  }
  .sidebar {
    position: fixed;
    left: 0;
    top: calc(3.6rem + env(safe-area-inset-top, 0px));
    width: 320px;
    max-width: min(86vw, 320px);
    height: calc(100vh - 3.6rem);
    height: calc(100dvh - 3.6rem - env(safe-area-inset-top, 0px));
    z-index: 60;
    background: var(--bg-elev);
    box-shadow: var(--shadow);
    padding: 1.1rem 0.9rem calc(1.5rem + env(safe-area-inset-bottom, 0px)) calc(0.9rem + env(safe-area-inset-left, 0px));
    overflow-y: auto;
    -webkit-overflow-scrolling: touch;
    overscroll-behavior: contain;
    transform: translateX(-105%);
    transition: transform 0.22s cubic-bezier(0.22, 1, 0.36, 1);
  }
  .sidebar.open { transform: translateX(0); }
  .chapter-link, .nav-link {
    min-height: 44px;
    align-items: center;
    padding: 0.55rem 0.75rem;
  }
  .copy-btn {
    min-height: 36px;
    min-width: 52px;
    padding: 0.32rem 0.7rem;
    font-size: 0.78rem;
  }
  .content table {
    display: block;
    width: 100%;
    max-width: 100%;
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
  }
  .mmatrix table {
    display: table;
    width: auto;
    overflow: visible;
  }
  .pager { grid-template-columns: 1fr; }
  .pager-card.next { text-align: left; }
}
@media (max-width: 520px) {
  .topbar {
    gap: 0.45rem;
    padding-left: calc(0.75rem + env(safe-area-inset-left, 0px));
    padding-right: calc(0.75rem + env(safe-area-inset-right, 0px));
  }
  .brand-wrap {
    gap: 0.45rem;
    flex: 1 1 auto;
  }
  .brand-link {
    font-size: 0.94rem;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .topbar-actions {
    gap: 0.35rem;
  }
  .demo-label-full,
  .search-label-full {
    display: none;
  }
  .demo-label-short,
  .search-label-short {
    display: inline;
  }
  .demo-btn,
  .search-trigger,
  .theme-btn {
    padding: 0.35rem 0.55rem;
    font-size: 0.8rem;
  }
  .layout {
    padding-left: calc(0.85rem + env(safe-area-inset-left, 0px));
    padding-right: calc(0.85rem + env(safe-area-inset-right, 0px));
  }
  .content {
    padding: 1.35rem 0 calc(3rem + env(safe-area-inset-bottom, 0px)) 0;
  }
  .content h1 { font-size: 1.55rem; }
  .content h2 { font-size: 1.24rem; margin-top: 1.85rem; }
  .content h3 { font-size: 1.06rem; }
  th, td { min-width: 6.5rem; padding: 0.52rem 0.65rem; }
  .mmatrix td { min-width: 0; padding: 0.1rem 0.35rem; }
  .math-inline { white-space: normal; overflow-wrap: anywhere; }
  .manual-figure { padding: 0.6rem; }
  .search-backdrop {
    padding: calc(1rem + env(safe-area-inset-top, 0px)) 0.65rem calc(0.65rem + env(safe-area-inset-bottom, 0px));
  }
  .search-input { font-size: 16px; }
  .search-results {
    max-height: 65vh;
    max-height: 65dvh;
  }
}
"""


MANUAL_JS = r"""(function () {
  var root = document.documentElement;
  var saved = null;
  try { saved = localStorage.getItem("nmr_manual_theme"); } catch (_) {}
  if (saved === "dark" || saved === "light") {
    root.setAttribute("data-theme", saved);
  } else if (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) {
    root.setAttribute("data-theme", "dark");
  } else {
    root.setAttribute("data-theme", "light");
  }

  function syncThemeBtn() {
    var btn = document.getElementById("theme-toggle");
    if (!btn) return;
    var cur = root.getAttribute("data-theme") === "dark" ? "dark" : "light";
    btn.textContent = cur === "dark" ? "Light" : "Dark";
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

    var menuBtn = document.getElementById("menu-toggle");
    var sidebar = document.getElementById("chapter-sidebar");
    var sidebarBackdrop = document.getElementById("sidebar-backdrop");

    function setDrawerOpen(isOpen) {
      if (!sidebar) return;
      if (isOpen) {
        sidebar.classList.add("open");
        if (sidebarBackdrop) sidebarBackdrop.classList.add("open");
        if (menuBtn) menuBtn.setAttribute("aria-expanded", "true");
      } else {
        sidebar.classList.remove("open");
        if (sidebarBackdrop) sidebarBackdrop.classList.remove("open");
        if (menuBtn) menuBtn.setAttribute("aria-expanded", "false");
      }
    }

    function closeDrawer() {
      setDrawerOpen(false);
    }

    if (menuBtn && sidebar) {
      menuBtn.addEventListener("click", function () {
        setDrawerOpen(!sidebar.classList.contains("open"));
      });
    }
    if (sidebarBackdrop) {
      sidebarBackdrop.addEventListener("click", closeDrawer);
    }
    if (sidebar) {
      sidebar.querySelectorAll("a").forEach(function (a) {
        a.addEventListener("click", closeDrawer);
      });
    }

    // Copy buttons for fenced code blocks
    var copyBtns = document.querySelectorAll(".copy-btn");
    copyBtns.forEach(function (btn) {
      btn.addEventListener("click", function () {
        var card = btn.closest(".code-card");
        var code = card ? card.querySelector("pre code") : null;
        if (!code || !navigator.clipboard) return;
        navigator.clipboard.writeText(code.textContent || "").then(function () {
          var prev = btn.textContent;
          btn.textContent = "Copied";
          setTimeout(function () { btn.textContent = prev; }, 1200);
        });
      });
    });

    // Instant client-side search
    var backdrop = document.getElementById("search-backdrop");
    var openBtn = document.getElementById("search-open");
    var closeBtn = document.getElementById("search-close");
    var input = document.getElementById("search-input");
    var resultsEl = document.getElementById("search-results");
    var indexCache = null;
    var activeIdx = 0;

    function loadIndex() {
      if (indexCache !== null) return Promise.resolve(indexCache);
      if (Array.isArray(window.NMR_MANUAL_SEARCH_INDEX)) {
        indexCache = window.NMR_MANUAL_SEARCH_INDEX;
        return Promise.resolve(indexCache);
      }
      return fetch("search-index.json")
        .then(function (r) { return r.json(); })
        .then(function (data) { indexCache = data; return data; })
        .catch(function () { indexCache = []; return []; });
    }

    function escapeHtml(s) {
      return s.replace(/[&<>"']/g, function (c) {
        return ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c];
      });
    }

    function openSearch() {
      if (!backdrop || !input) return;
      closeDrawer();
      backdrop.classList.add("open");
      input.focus();
      input.select();
      loadIndex().then(function () { runQuery(input.value); });
    }

    function closeSearch() {
      if (!backdrop) return;
      backdrop.classList.remove("open");
    }

    function runQuery(raw) {
      if (!resultsEl || !indexCache) return;
      var q = raw.trim().toLowerCase();
      var tokens = q ? q.split(/\s+/).filter(Boolean) : [];
      var scored = [];
      for (var i = 0; i < indexCache.length; i++) {
        var rec = indexCache[i];
        if (tokens.length === 0) {
          if (rec.section_title === rec.page_title) {
            scored.push({ score: 1, rec: rec });
          }
          continue;
        }
        var hayPage = rec.page_title.toLowerCase();
        var haySec = rec.section_title.toLowerCase();
        var hayBody = rec.body_text.toLowerCase();
        var ok = true;
        var total = 0;
        for (var t = 0; t < tokens.length; t++) {
          var tok = tokens[t];
          var s = 0;
          if (haySec.indexOf(tok) !== -1) s += 12;
          if (hayPage.indexOf(tok) !== -1) s += 6;
          if (hayBody.indexOf(tok) !== -1) s += 2;
          if (s === 0) { ok = false; break; }
          total += s;
        }
        if (ok) scored.push({ score: total, rec: rec });
      }
      scored.sort(function (a, b) { return b.score - a.score; });
      var top = scored.slice(0, 14);
      activeIdx = 0;
      if (top.length === 0) {
        resultsEl.innerHTML = '<li class="search-empty">No matching manual sections found.</li>';
        return;
      }
      var items = top.map(function (entry, idx) {
        var r = entry.rec;
        var snippet = r.body_text;
        if (tokens.length > 0) {
          var pos = snippet.toLowerCase().indexOf(tokens[0]);
          if (pos > 45) snippet = "…" + snippet.slice(pos - 35, pos + 135) + "…";
          else if (snippet.length > 160) snippet = snippet.slice(0, 160) + "…";
        } else if (snippet.length > 140) {
          snippet = snippet.slice(0, 140) + "…";
        }
        var cls = idx === 0 ? "search-hit selected" : "search-hit";
        return (
          '<li><a class="' + cls + '" href="' + escapeHtml(r.href) + '">' +
          '<div class="hit-meta">' + escapeHtml(r.page_title) + '</div>' +
          '<div class="hit-title">' + escapeHtml(r.section_title) + '</div>' +
          '<div class="hit-snippet">' + escapeHtml(snippet) + '</div>' +
          '</a></li>'
        );
      });
      resultsEl.innerHTML = items.join("");
      resultsEl.querySelectorAll(".search-hit").forEach(function (a) {
        a.addEventListener("click", function () {
          closeSearch();
          closeDrawer();
        });
      });
    }

    if (openBtn) openBtn.addEventListener("click", openSearch);
    if (closeBtn) closeBtn.addEventListener("click", closeSearch);
    if (backdrop) {
      backdrop.addEventListener("click", function (ev) {
        if (ev.target === backdrop) closeSearch();
      });
    }
    if (input) {
      input.addEventListener("input", function () { runQuery(input.value); });
      input.addEventListener("keydown", function (ev) {
        var hits = resultsEl ? resultsEl.querySelectorAll(".search-hit") : [];
        if (ev.key === "Escape") {
          closeSearch();
        } else if (ev.key === "ArrowDown" && hits.length > 0) {
          ev.preventDefault();
          hits[activeIdx].classList.remove("selected");
          activeIdx = (activeIdx + 1) % hits.length;
          hits[activeIdx].classList.add("selected");
          hits[activeIdx].scrollIntoView({ block: "nearest" });
        } else if (ev.key === "ArrowUp" && hits.length > 0) {
          ev.preventDefault();
          hits[activeIdx].classList.remove("selected");
          activeIdx = (activeIdx - 1 + hits.length) % hits.length;
          hits[activeIdx].classList.add("selected");
          hits[activeIdx].scrollIntoView({ block: "nearest" });
        } else if (ev.key === "Enter" && hits.length > 0) {
          ev.preventDefault();
          var href = hits[activeIdx].getAttribute("href");
          closeSearch();
          closeDrawer();
          if (href) window.location.href = href;
        }
      });
    }

    document.addEventListener("keydown", function (ev) {
      var tag = ev.target && ev.target.tagName ? ev.target.tagName.toLowerCase() : "";
      var editing = tag === "input" || tag === "textarea" || (ev.target && ev.target.isContentEditable);
      if ((ev.metaKey || ev.ctrlKey) && ev.key.toLowerCase() === "k") {
        ev.preventDefault();
        openSearch();
      } else if (!editing && ev.key === "/") {
        ev.preventDefault();
        openSearch();
      } else if (ev.key === "Escape") {
        if (backdrop && backdrop.classList.contains("open")) {
          closeSearch();
        }
        if (sidebar && sidebar.classList.contains("open")) {
          closeDrawer();
        }
      }
    });
  });
})();
"""


# ---------------------------------------------------------------------------
# Page discovery, HTML assembly & validation
# ---------------------------------------------------------------------------

def discover_manual_pages(src_dir: Path) -> List[Path]:
    if not src_dir.is_dir():
        raise FileNotFoundError(f"manual source directory not found: {src_dir}")
    md_files = {p.name: p for p in sorted(src_dir.glob("*.md")) if p.is_file()}
    if not md_files:
        raise ValueError(f"no markdown chapters found in {src_dir}")
    ordered: List[Path] = []
    for expected in EXPECTED_MANUAL_CHAPTERS:
        if expected in md_files:
            ordered.append(md_files.pop(expected))
    for remaining_name in sorted(md_files):
        ordered.append(md_files[remaining_name])
    return ordered


def chapter_badge_and_short_title(md_name: str, h1_text: str) -> Tuple[str, str]:
    if md_name == "index.md":
        return "00", "Manual Overview"
    m = re.match(r"^(\d+)-", md_name)
    badge = m.group(1) if m else "•"
    cleaned = re.sub(r"^Chapter\s+\d+\s*:\s*", "", h1_text, flags=re.IGNORECASE).strip()
    return badge, cleaned or h1_text


def parse_manual_page(md_path: Path) -> ParsedPage:
    raw_md = md_path.read_text(encoding="utf-8")
    lines = raw_md.splitlines()
    used_slugs: Set[str] = set()
    headings: List[HeadingItem] = []
    links: List[LinkRef] = []
    search_sections: List[Tuple[str, str, List[str]]] = []

    body_html = parse_markdown_blocks(
        lines=lines,
        md_name=md_path.name,
        base_line=1,
        used_slugs=used_slugs,
        headings_out=headings,
        links_out=links,
        search_sections=search_sections,
    )

    h1_text = headings[0].title_text if headings else md_path.stem
    badge, nav_title = chapter_badge_and_short_title(md_path.name, h1_text)
    html_name = md_path.stem + ".html"

    records: List[SearchRecord] = []
    for sec_slug, sec_title, chunks in search_sections:
        joined = re.sub(r"\s+", " ", " ".join(chunks)).strip()
        if not sec_title and not joined:
            continue
        effective_title = sec_title or h1_text
        href = f"{html_name}#{sec_slug}" if sec_slug else html_name
        records.append(
            SearchRecord(
                page_slug=md_path.stem,
                page_title=h1_text,
                section_title=effective_title,
                href=href,
                body_text=joined[:900],
            )
        )

    return ParsedPage(
        md_name=md_path.name,
        html_name=html_name,
        chapter_badge=badge,
        nav_title=nav_title,
        page_title=h1_text,
        headings=headings,
        anchor_ids=set(used_slugs),
        links=links,
        body_html=body_html,
        search_records=records,
    )


def render_page_html(
    page: ParsedPage,
    all_pages: Sequence[ParsedPage],
    page_idx: int,
    has_demo_page: bool = False,
) -> str:
    nav_items: List[str] = []
    for p in all_pages:
        active_cls = " active" if p.html_name == page.html_name else ""
        nav_items.append(
            f'<li><a class="chapter-link nav-link{active_cls}" href="{html.escape(p.html_name)}">'
            f'<span class="chap-num">{html.escape(p.chapter_badge)}</span>'
            f'<span>{html.escape(p.nav_title)}</span></a></li>'
        )

    toc_items: List[str] = []
    for h in page.headings:
        if h.level in (2, 3):
            lvl_cls = " toc-h3" if h.level == 3 else ""
            toc_items.append(
                f'<li><a class="toc-link{lvl_cls}" href="#{html.escape(h.slug)}">'
                f"{html.escape(h.title_text)}</a></li>"
            )
    toc_html = (
        f'<div class="sidebar-caption">On This Page</div><ul class="toc-list">{"".join(toc_items)}</ul>'
        if toc_items
        else ""
    )

    prev_html = "<div></div>"
    if page_idx > 0:
        prev_p = all_pages[page_idx - 1]
        prev_html = (
            f'<a class="pager-card prev" href="{html.escape(prev_p.html_name)}">'
            f'<span class="pager-dir">← Previous</span>'
            f'<span class="pager-title">{html.escape(prev_p.chapter_badge)} · {html.escape(prev_p.nav_title)}</span>'
            f"</a>"
        )

    next_html = "<div></div>"
    if page_idx + 1 < len(all_pages):
        next_p = all_pages[page_idx + 1]
        next_html = (
            f'<a class="pager-card next" href="{html.escape(next_p.html_name)}">'
            f'<span class="pager-dir">Next →</span>'
            f'<span class="pager-title">{html.escape(next_p.chapter_badge)} · {html.escape(next_p.nav_title)}</span>'
            f"</a>"
        )

    full_title = (
        f"{SITE_TITLE} — {SITE_TAGLINE}"
        if page.md_name == "index.md"
        else f"{page.page_title} | {SITE_TITLE}"
    )

    demo_action_html = (
        '<a class="demo-btn" href="demo/index.html">'
        '<span class="demo-label-full">Live WebGPU Demo</span>'
        '<span class="demo-label-short">Demo</span></a>'
        if has_demo_page
        else ""
    )

    return f"""<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
  <meta name="theme-color" content="#0b1120" media="(prefers-color-scheme: dark)">
  <meta name="theme-color" content="#f8fafc" media="(prefers-color-scheme: light)">
  <title>{html.escape(full_title)}</title>
  <link rel="stylesheet" href="assets/style.css">
  <script src="assets/search-data.js" defer></script>
  <script src="assets/manual.js" defer></script>
</head>
<body>
  <header class="topbar">
    <div class="brand-wrap">
      <button type="button" id="menu-toggle" class="menu-btn" title="Toggle chapter menu" aria-label="Toggle chapter menu" aria-expanded="false" aria-controls="chapter-sidebar">☰</button>
      <a class="brand-link" href="index.html">{html.escape(SITE_TITLE)}</a>
      <span class="version-pill">{html.escape(SITE_VERSION_BADGE)}</span>
    </div>
    <div class="topbar-actions">
      {demo_action_html}
      <button type="button" id="search-open" class="search-trigger" title="Search manual">
        <span class="search-label-full">Search manual…</span>
        <span class="search-label-short">Search</span>
        <span class="kbd-hint">/ · ⌘K</span>
      </button>
      <button type="button" id="theme-toggle" class="theme-btn" title="Toggle color theme">Dark</button>
    </div>
  </header>

  <div class="layout">
    <nav id="chapter-sidebar" class="sidebar" title="Manual chapters">
      <div class="sidebar-caption">Chapters</div>
      <ul class="chapter-list">
        {"".join(nav_items)}
      </ul>
    </nav>
    <div id="sidebar-backdrop" class="sidebar-backdrop" aria-hidden="true"></div>

    <main class="content">
      <article>
{page.body_html}
      </article>
      <nav class="pager" title="Chapter pagination">
        {prev_html}
        {next_html}
      </nav>
      <footer class="footer-note">
        {html.escape(SITE_TITLE)} ({html.escape(SITE_VERSION_BADGE)}) · MIT License · 100% Clean-Room Open-Source NMR Platform
      </footer>
    </main>

    <aside class="toc-aside" title="Page outline">
      {toc_html}
    </aside>
  </div>

  <div id="search-backdrop" class="search-backdrop" role="dialog" title="Search documentation">
    <div class="search-modal">
      <div class="search-input-row">
        <input id="search-input" class="search-input" type="search" placeholder="Search chapters, CLI flags, equations, workflows…" autocomplete="off">
        <button type="button" id="search-close" class="theme-btn">Esc</button>
      </div>
      <ul id="search-results" class="search-results"></ul>
    </div>
  </div>
</body>
</html>
"""


def build_site_files(
    pages: Sequence[ParsedPage],
    demo_html_path: Optional[Path] = None,
    assets_dir: Optional[Path] = None,
) -> Dict[str, bytes]:
    """Assemble the complete deterministic static site in memory as relative_path -> UTF-8 bytes."""
    files: Dict[str, bytes] = {}
    files[".nojekyll"] = b""
    files["assets/style.css"] = MANUAL_CSS.encode("utf-8")
    files["assets/manual.js"] = MANUAL_JS.encode("utf-8")

    if assets_dir is not None and assets_dir.is_dir():
        for svg_path in sorted(assets_dir.glob("*.svg")):
            files[f"assets/{svg_path.name}"] = svg_path.read_bytes()

    has_demo = demo_html_path is not None and demo_html_path.is_file()
    if has_demo and demo_html_path is not None:
        files["demo/index.html"] = demo_html_path.read_bytes()

    search_payload = [
        {
            "body_text": rec.body_text,
            "href": rec.href,
            "page_slug": rec.page_slug,
            "page_title": rec.page_title,
            "section_title": rec.section_title,
        }
        for p in pages
        for rec in p.search_records
    ]
    search_json_bytes = (
        json.dumps(search_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    files["search-index.json"] = search_json_bytes
    files["assets/search-data.js"] = (
        b"window.NMR_MANUAL_SEARCH_INDEX = " + search_json_bytes.rstrip(b"\n") + b";\n"
    )

    for idx, p in enumerate(pages):
        html_str = render_page_html(p, pages, idx, has_demo_page=has_demo)
        files[p.html_name] = html_str.encode("utf-8")

    return files


def validate_markdown_hygiene(
    md_paths: Sequence[Path],
    enforce_expected_roster: bool = True,
) -> List[str]:
    """Verify chapter roster, clean-room token hygiene, host-path hygiene, and GFM table safety."""
    errors: List[str] = []
    present_names = {p.name for p in md_paths}
    if enforce_expected_roster:
        for req in EXPECTED_MANUAL_CHAPTERS:
            if req not in present_names:
                errors.append(f"missing required manual chapter '{req}'")

    token_patterns = [
        (tok, re.compile(rf"(?<![a-z0-9]){re.escape(tok)}(?![a-z0-9])", re.IGNORECASE))
        for tok in RESTRICTED_PRODUCT_TOKENS
    ]
    host_prefixes = ("/" + "Users/", "/" + "home/")

    for md_path in md_paths:
        raw_text = md_path.read_text(encoding="utf-8")
        lines = raw_text.splitlines()
        in_fence = False
        prev_was_table = False
        had_non_blank_since_table = True
        expected_cols: Optional[int] = None
        seen_sep_in_table = False

        for line_no, ln in enumerate(lines, 1):
            for hp in host_prefixes:
                if hp in ln:
                    errors.append(f"{md_path.name}:{line_no}: forbidden host path prefix '{hp}'")
            for tok, pat in token_patterns:
                if pat.search(ln):
                    errors.append(
                        f"{md_path.name}:{line_no}: restricted product name '{tok}' is not permitted"
                    )

            stripped = ln.strip()
            if re.match(r"^`{3,}", stripped):
                in_fence = not in_fence
                prev_was_table = False
                had_non_blank_since_table = True
                expected_cols = None
                seen_sep_in_table = False
                continue
            if in_fence:
                continue

            is_tbl_row = stripped.startswith("|") and stripped.endswith("|")
            if is_tbl_row:
                if not prev_was_table and not had_non_blank_since_table:
                    errors.append(
                        f"{md_path.name}:{line_no}: adjacent Markdown tables must be separated by a heading or caption"
                    )
                # GFM pipe check: even inside `...` or $...$, unescaped | breaks GitHub table cells
                for span_m in re.finditer(r"`[^`]+`|\$[^$]+\$", stripped):
                    if re.search(r"(?<!\\)\|", span_m.group(0)):
                        errors.append(
                            f"{md_path.name}:{line_no}: unescaped '|' inside table cell code/math {span_m.group(0)}"
                        )
                cells = split_table_row(stripped)
                if expected_cols is None:
                    expected_cols = len(cells)
                    seen_sep_in_table = False
                else:
                    if len(cells) != expected_cols:
                        errors.append(
                            f"{md_path.name}:{line_no}: table column mismatch (got {len(cells)}, expected {expected_cols})"
                        )
                    if is_table_separator(stripped):
                        if seen_sep_in_table:
                            errors.append(
                                f"{md_path.name}:{line_no}: fused table detected (duplicate separator row)"
                            )
                        seen_sep_in_table = True
                prev_was_table = True
                had_non_blank_since_table = False
            elif stripped:
                prev_was_table = False
                had_non_blank_since_table = True
                expected_cols = None
                seen_sep_in_table = False
            else:
                prev_was_table = False
                expected_cols = None
                seen_sep_in_table = False

    return errors


def validate_internal_links(
    pages: Sequence[ParsedPage],
    site_files: Dict[str, bytes],
) -> List[str]:
    """Verify every Markdown link and generated HTML href/src resolves cleanly."""
    errors: List[str] = []
    by_html: Dict[str, ParsedPage] = {p.html_name: p for p in pages}

    # 1. Validate Markdown-level links (including exact source line numbers)
    for page in pages:
        for ref in page.links:
            parsed = urlsplit(ref.raw_target)
            if parsed.scheme in ("http", "https", "mailto"):
                continue
            if parsed.scheme:
                errors.append(
                    f"{ref.source_md}:{ref.line_no}: unsupported URI scheme in link '{ref.raw_target}'"
                )
                continue

            rewritten = rewrite_internal_href(ref.raw_target)
            r_split = urlsplit(rewritten)
            rel_path = unquote(r_split.path)
            frag = unquote(r_split.fragment)

            if ".." in Path(rel_path).parts or rel_path.startswith("/"):
                errors.append(
                    f"{ref.source_md}:{ref.line_no}: link '{ref.raw_target}' escapes manual root"
                )
                continue

            target_html = page.html_name if not rel_path else rel_path
            if target_html not in site_files:
                errors.append(
                    f"{ref.source_md}:{ref.line_no}: broken internal link target '{ref.raw_target}' "
                    f"(resolved '{target_html}' does not exist)"
                )
                continue

            if frag:
                target_page = by_html.get(target_html)
                if target_page is None:
                    errors.append(
                        f"{ref.source_md}:{ref.line_no}: fragment '#{frag}' points to non-HTML asset '{target_html}'"
                    )
                elif frag not in target_page.anchor_ids:
                    errors.append(
                        f"{ref.source_md}:{ref.line_no}: broken anchor '#{frag}' on '{target_html}'"
                    )

    # 2. Validate every href / src attribute in rendered HTML files
    attr_re = re.compile(r'\b(?:href|src)="([^"]+)"')
    for html_name, raw_bytes in sorted(site_files.items()):
        if not html_name.endswith(".html"):
            continue
        doc = raw_bytes.decode("utf-8")
        # Collect all id="..." attributes actually emitted in this HTML document
        emitted_ids = set(re.findall(r'\bid="([^"]+)"', doc))
        # Strip <script>, <style>, <pre>, <code>, and HTML tag attributes to verify no unrendered Markdown/backticks
        prose_only = re.sub(r"<(?:script|style|pre|code)\b[^>]*>.*?</(?:script|style|pre|code)>", " ", doc, flags=re.DOTALL)
        prose_only = re.sub(r"<[^>]+>", " ", prose_only)
        if "`" in prose_only:
            errors.append(f"{html_name}: unrendered backtick literal detected in prose output")
        if re.search(r"\$[^$\n]+\$", prose_only):
            errors.append(f"{html_name}: unrendered inline LaTeX '$...$' detected in prose output")
        for m in attr_re.finditer(doc):
            href = html.unescape(m.group(1))
            parsed = urlsplit(href)
            if parsed.scheme in ("http", "https", "mailto", "data"):
                continue
            rel_path = unquote(parsed.path)
            frag = unquote(parsed.fragment)
            if not rel_path:
                target_file = html_name
            else:
                base_dir = posixpath.dirname(html_name)
                joined = posixpath.join(base_dir, rel_path) if base_dir else rel_path
                target_file = posixpath.normpath(joined)
            if target_file.startswith("../") or target_file == ".." or target_file not in site_files:
                errors.append(f"{html_name}: broken rendered asset/page reference '{href}'")
                continue
            if frag and target_file.endswith(".html"):
                target_doc = site_files[target_file].decode("utf-8")
                target_ids = (
                    emitted_ids
                    if target_file == html_name
                    else set(re.findall(r'\bid="([^"]+)"', target_doc))
                )
                if frag not in target_ids:
                    errors.append(f"{html_name}: broken rendered fragment '#{frag}' in '{href}'")

    return errors


def validate_mobile_ux(
    site_files: Dict[str, bytes],
    repo_root: Optional[Path] = None,
) -> List[str]:
    """Verify mobile/tablet responsiveness, safe-area insets, drawer UX, and scroll containment."""
    errors: List[str] = []

    manual_html_names = [
        k for k in sorted(site_files) if k.endswith(".html") and not k.startswith("demo/")
    ]
    if not manual_html_names:
        errors.append("mobile-ux: no manual HTML pages emitted")
    for html_name in manual_html_names:
        doc = site_files[html_name].decode("utf-8")
        for tok in (
            "viewport-fit=cover",
            'id="menu-toggle"',
            'id="chapter-sidebar"',
            'id="sidebar-backdrop"',
            "nav-link",
        ):
            if tok not in doc:
                errors.append(f"{html_name}: missing mobile-UX HTML token {tok!r}")

    css_bytes = site_files.get("assets/style.css")
    if css_bytes is None:
        errors.append("assets/style.css: missing stylesheet")
    else:
        css = css_bytes.decode("utf-8")
        for tok in (
            "100vh",
            "100dvh",
            "env(safe-area-inset-top",
            "env(safe-area-inset-bottom",
            "touch-action: manipulation",
            "overflow-x: hidden",
            "overflow-x: auto",
            "-webkit-overflow-scrolling: touch",
            "max-width: min(86vw, 320px)",
            "min-width: 44px",
            "min-height: 44px",
            "@media (max-width: 860px)",
            "@media (max-width: 520px)",
            "figure svg",
        ):
            if tok not in css:
                errors.append(f"assets/style.css: missing mobile-UX CSS rule {tok!r}")
        if "100vh" in css and "100dvh" in css and css.find("100vh") > css.find("100dvh"):
            errors.append("assets/style.css: 100vh fallback must precede 100dvh")

    js_bytes = site_files.get("assets/manual.js")
    if js_bytes is None:
        errors.append("assets/manual.js: missing manual script")
    else:
        js = js_bytes.decode("utf-8")
        for tok in (
            "menu-toggle",
            "chapter-sidebar",
            "sidebar-backdrop",
            "closeDrawer",
            "Escape",
        ):
            if tok not in js:
                errors.append(f"assets/manual.js: missing mobile drawer JS token {tok!r}")

    shell_targets: List[Tuple[str, str, bool]] = []
    if "demo/index.html" in site_files:
        shell_targets.append(("demo/index.html", site_files["demo/index.html"].decode("utf-8"), True))
    if repo_root is not None:
        demo_disk = repo_root / "docs" / "demo" / "index.html"
        trunk_disk = repo_root / "crates" / "nmr-gui" / "index.html"
        if demo_disk.is_file() and "demo/index.html" not in site_files:
            shell_targets.append(("docs/demo/index.html", demo_disk.read_text(encoding="utf-8"), True))
        if trunk_disk.is_file():
            shell_targets.append(("crates/nmr-gui/index.html", trunk_disk.read_text(encoding="utf-8"), False))

    for label, shell_doc, is_demo in shell_targets:
        for tok in (
            "viewport-fit=cover",
            "100vh",
            "100dvh",
            "touch-action: none",
            "overscroll-behavior: none",
            "env(safe-area-inset-top",
            "env(safe-area-inset-bottom",
            "#canvas-wrap",
        ):
            if tok not in shell_doc:
                errors.append(f"{label}: missing mobile WebGPU shell token {tok!r}")
        if "100vh" in shell_doc and "100dvh" in shell_doc and shell_doc.find("100vh") > shell_doc.find("100dvh"):
            errors.append(f"{label}: 100vh fallback must precede 100dvh")
        if is_demo:
            for tok in ("@media (max-width: 680px)", "#fallback", "overflow-y: auto"):
                if tok not in shell_doc:
                    errors.append(f"{label}: missing demo mobile token {tok!r}")

    return errors


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


def run_self_tests() -> int:
    """Run adversarial hygiene, link-validator, and mobile-UX unit tests in an ephemeral directory."""
    repo_root = Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        valid_md = root / "index.md"
        valid_md.write_text(
            "# Overview\n\n## Section Alpha\n\nSee [Alpha](#section-alpha).\n\n"
            "| A | B |\n| :--- | :--- |\n| `ok` | $\\lvert x \\rvert$ |\n",
            encoding="utf-8",
        )
        pages = [parse_manual_page(valid_md)]
        site = build_site_files(pages)
        assert not validate_markdown_hygiene([valid_md], enforce_expected_roster=False)
        assert not validate_internal_links(pages, site)
        mob_ok = validate_mobile_ux(site, repo_root=repo_root)
        assert not mob_ok, f"expected clean mobile-UX check, got {mob_ok}"

        # 1. Broken anchor must fail
        bad_link = root / "01-bad-link.md"
        bad_link.write_text("# Ch1\n\nBroken [ref](index.md#missing-anchor).\n", encoding="utf-8")
        p_bad = [parse_manual_page(valid_md), parse_manual_page(bad_link)]
        link_errs = validate_internal_links(p_bad, build_site_files(p_bad))
        assert any("broken anchor" in e for e in link_errs), f"expected broken anchor error, got {link_errs}"

        # 2. Adjacent tables without heading/caption must fail
        adj_tbl = root / "02-adj.md"
        adj_tbl.write_text(
            "# Ch2\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n\n| C | D |\n| --- | --- |\n| 3 | 4 |\n",
            encoding="utf-8",
        )
        adj_errs = validate_markdown_hygiene([adj_tbl], enforce_expected_roster=False)
        assert any("adjacent Markdown tables" in e for e in adj_errs), f"expected adjacent table error, got {adj_errs}"

        # 3. Unescaped pipe inside table code/math must fail
        pipe_tbl = root / "03-pipe.md"
        pipe_tbl.write_text(
            "# Ch3\n\n| Flag | Math |\n| --- | --- |\n| `a|b` | $|x|$ |\n",
            encoding="utf-8",
        )
        pipe_errs = validate_markdown_hygiene([pipe_tbl], enforce_expected_roster=False)
        assert any("unescaped '|'" in e for e in pipe_errs), f"expected unescaped pipe error, got {pipe_errs}"

        # 4. Restricted token & host path must fail
        tok_md = root / "04-tok.md"
        bad_tok = RESTRICTED_PRODUCT_TOKENS[0]
        bad_path = "/" + "Users/example"
        tok_md.write_text(f"# Ch4\n\nMentions {bad_tok} at {bad_path}.\n", encoding="utf-8")
        tok_errs = validate_markdown_hygiene([tok_md], enforce_expected_roster=False)
        assert any("restricted product name" in e for e in tok_errs), f"expected token error, got {tok_errs}"
        assert any("forbidden host path" in e for e in tok_errs), f"expected host path error, got {tok_errs}"

        # 5. Embedded SVG figure & missing-asset image check
        assets_tmp = root / "assets_src"
        assets_tmp.mkdir()
        (assets_tmp / "01_test.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"></svg>\n', encoding="utf-8")
        fig_md = root / "05-fig.md"
        fig_md.write_text(
            "# Ch5\n\n![Test showcase SVG](../assets/01_test.svg)\n*Figure 1: Deterministic showcase.*\n",
            encoding="utf-8",
        )
        p_fig = [parse_manual_page(valid_md), parse_manual_page(fig_md)]
        site_fig = build_site_files(p_fig, assets_dir=assets_tmp)
        assert not validate_internal_links(p_fig, site_fig), "bundled ../assets/*.svg should validate cleanly"
        assert '<figure class="manual-figure">' in site_fig["05-fig.html"].decode("utf-8")
        missing_fig_errs = validate_internal_links(p_fig, build_site_files(p_fig))
        assert any("broken internal link target" in e for e in missing_fig_errs), (
            f"expected missing SVG asset error, got {missing_fig_errs}"
        )

        # 6. Adversarial mobile-UX checks (viewport-fit=cover, 100dvh, touch-action, drawer toggle, table overflow-x: auto)
        mut_vp = dict(site)
        mut_vp["index.html"] = site["index.html"].replace(b"viewport-fit=cover", b"")
        assert any("viewport-fit=cover" in e for e in validate_mobile_ux(mut_vp))

        for css_tok in (b"100dvh", b"touch-action: manipulation", b"overflow-x: auto"):
            mut_css = dict(site)
            mut_css["assets/style.css"] = site["assets/style.css"].replace(css_tok, b"")
            css_errs = validate_mobile_ux(mut_css)
            assert any(css_tok.decode("utf-8") in e for e in css_errs), (
                f"expected mobile-UX error when stripping {css_tok!r}, got {css_errs}"
            )

        mut_js = dict(site)
        mut_js["assets/manual.js"] = site["assets/manual.js"].replace(b"closeDrawer", b"noop")
        assert any("closeDrawer" in e for e in validate_mobile_ux(mut_js))

    print("build_docs_manual.py self-tests passed (7/7 adversarial, mobile-UX & determinism checks).")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    repo_root = Path(__file__).resolve().parent.parent
    default_src = repo_root / "docs" / "manual"
    default_demo_html = repo_root / "docs" / "demo" / "index.html"
    default_assets_dir = repo_root / "docs" / "assets"
    default_out = repo_root / "target" / "manual"

    parser = argparse.ArgumentParser(
        description="Build deterministic static HTML documentation manual from docs/manual/*.md"
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=default_src,
        help="Source directory containing Markdown manual chapters (default: docs/manual)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=default_out,
        help="Output directory for generated static HTML manual",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate internal page/anchor links, hygiene invariants, and build determinism",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run built-in unit and adversarial validation self-tests and exit",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_tests()

    md_paths = discover_manual_pages(args.src)
    is_default_src = args.src.resolve() == default_src.resolve()
    demo_html = default_demo_html if is_default_src else None
    assets_dir = default_assets_dir if is_default_src else None
    pages = [parse_manual_page(p) for p in md_paths]
    site_files = build_site_files(pages, demo_html_path=demo_html, assets_dir=assets_dir)
    write_site_to_disk(args.out, site_files)

    if args.check:
        errors = validate_markdown_hygiene(
            md_paths,
            enforce_expected_roster=is_default_src,
        )
        errors.extend(validate_internal_links(pages, site_files))
        errors.extend(validate_mobile_ux(site_files, repo_root=repo_root if is_default_src else None))
        # Re-parse and re-render to guarantee 100% deterministic byte output
        pages_second = [parse_manual_page(p) for p in discover_manual_pages(args.src)]
        second_files = build_site_files(pages_second, demo_html_path=demo_html, assets_dir=assets_dir)
        if site_digest(site_files) != site_digest(second_files):
            errors.append("non-deterministic build output detected across consecutive renders")

        if errors:
            for err in errors:
                print(f"error: {err}", file=sys.stderr)
            print(
                f"manual check FAILED with {len(errors)} error(s).",
                file=sys.stderr,
            )
            return 1

    total_sections = sum(len(p.search_records) for p in pages)
    total_links = sum(len(p.links) for p in pages)
    digest = site_digest(site_files)[:16]
    print(
        f"Built {len(pages)} manual pages ({total_sections} indexed sections, "
        f"{total_links} links verified, sha256={digest}) -> {args.out.name}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
