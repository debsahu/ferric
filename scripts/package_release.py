#!/usr/bin/env python3
"""Zero-dependency cross-platform desktop release packager and icon generator.

Builds reproducible release bundles and procedural vector/raster desktop icons
for ``FERRIC`` across:
- ``macos-arm64``: ``FERRIC.app`` bundle (``ferric-gui`` + ``ferric`` CLI/MCP,
  ``Info.plist``, ``PkgInfo``, ``FERRIC.icns``, ad-hoc Apple Silicon ``codesign``),
  compressed drag-and-drop ``.dmg`` (with ``/Applications`` symlink and quickstart
  guide), and ``.tar.gz`` archive.
- ``macos-x86_64``: ``FERRIC.app`` bundle cross-compiled for Intel Macs
  (``x86_64-apple-darwin``), ad-hoc ``codesign``, compressed ``.dmg``, and ``.tar.gz``
  archive.
- ``windows-x86_64``: deterministic ``.zip`` archive containing
  ``ferric-gui.exe``, ``ferric.exe``, multi-resolution ``ferric.ico``,
  ``Launch-NMR-Workbench.bat``, ``README.md``, and ``THIRD_PARTY.md``.
- ``linux-x86_64``: deterministic ``.tar.gz`` archive (``bin/ferric-gui``,
  ``bin/ferric``, FreeDesktop ``ferric.desktop``, scalable SVG + ``256x256`` PNG
  hicolor icons, user-level ``install-desktop.sh``, ``README.md``, ``LICENSE``,
  ``THIRD_PARTY.md``) plus native Debian/Ubuntu ``.deb`` (``amd64``, with
  ``/usr/share/doc/ferric/copyright`` and GUI runtime dependencies) and RHEL 9 /
  AlmaLinux 9 / Rocky 9 / Fedora ``.rpm`` (``x86_64``, ``glibc >= 2.34``).
- ``linux-i686-cli``: static ``musl`` 32-bit CLI-only ``ferric`` packaged as
  deterministic ``.tar.gz``, ``.deb`` (``i386``), and ``.rpm`` (``i686``) with
  ``LICENSE`` and ``THIRD_PARTY.md``.
- ``--checksums <dir>``: deterministic ``SHA256SUMS.txt`` generator and verifier.

Provenance: Original clean-room Python 3 standard-library release packaging and
RFC 2083 PNG / Apple ICNS / Windows ICO / SVG icon script (MIT License).
"""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import math
import os
from pathlib import Path
import plistlib
import re
import shutil
import stat
import struct
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Dict, List, Optional, Sequence, Tuple
import unittest
import xml.etree.ElementTree as ET
import zipfile
import zlib

APP_NAME = "FERRIC"
APP_SLUG = "ferric"
CLI_BIN = "ferric"
CLI_ALIAS_BIN = "nmrassign"
GUI_BIN = "ferric-gui"
BUNDLE_ID = "io.github.debsahu.ferric"
PROJECT_URL = "https://github.com/debsahu/ferric"
DEB_MAINTAINER = "Debashish Sahu <debashish.sahu@gmail.com>"
MACOS_MIN_SYSTEM_VERSION = "13.0"
DETERMINISTIC_MTIME = 1700000000  # 2023-11-14T22:13:20Z
DETERMINISTIC_ZIP_DT = (2026, 1, 1, 0, 0, 0)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
PNG_SIZES: Tuple[int, ...] = (16, 32, 48, 64, 128, 256, 512)
ICO_SIZES: Tuple[int, ...] = (16, 32, 48, 256)
ICNS_SLOTS: Tuple[Tuple[bytes, int], ...] = (
    (b"icp4", 16),
    (b"icp5", 32),
    (b"ic11", 32),
    (b"ic12", 64),
    (b"ic07", 128),
    (b"ic13", 256),
    (b"ic08", 256),
    (b"ic14", 512),
    (b"ic09", 512),
)

SUPPORTED_PLATFORMS: Tuple[str, ...] = (
    "macos-arm64",
    "macos-x86_64",
    "windows-x86_64",
    "linux-x86_64",
    "linux-i686-cli",
)

RELEASE_ARCHIVE_SUFFIXES: Tuple[str, ...] = (
    ".dmg",
    ".tar.gz",
    ".zip",
    ".deb",
    ".rpm",
)

# Runtime libraries loaded via dlopen() by winit/xkbcommon-dl/wayland-sys/ash in ferric-gui.
GUI_DEB_DEPENDS: Tuple[str, ...] = (
    "libc6 (>= 2.34)",
    "libgcc-s1 (>= 3.0)",
    "libvulkan1",
    "libxkbcommon0",
    "libxkbcommon-x11-0",
    "libx11-6",
    "libx11-xcb1",
    "libxcb1",
    "libxcursor1",
    "libxi6",
    "libwayland-client0",
)
GUI_DEB_RECOMMENDS: Tuple[str, ...] = ("mesa-vulkan-drivers",)
GUI_RPM_REQUIRES: Tuple[str, ...] = (
    "vulkan-loader",
    "libxkbcommon",
    "libxkbcommon-x11",
    "libX11",
    "libX11-xcb",
    "libxcb",
    "libXcursor",
    "libXi",
    "libwayland-client",
)
GUI_RPM_RECOMMENDS: Tuple[str, ...] = ("mesa-vulkan-drivers",)


def _write_text_lf(path: Path, text: str) -> None:
    """Write UTF-8 text with LF line endings (compatible with Python 3.9+)."""
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def read_workspace_version(repo_root: Path) -> str:
    """Read ``[workspace.package] version`` from ``Cargo.toml``."""
    cargo_toml = repo_root / "Cargo.toml"
    if not cargo_toml.is_file():
        raise FileNotFoundError(f"Cargo.toml not found at {cargo_toml}")
    in_ws_pkg = False
    for raw_line in cargo_toml.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("["):
            in_ws_pkg = stripped == "[workspace.package]"
            continue
        if in_ws_pkg:
            match = re.match(r'^version\s*=\s*"([^"]+)"', stripped)
            if match:
                return normalize_version(match.group(1))
    raise ValueError(f"Failed to locate [workspace.package] version in {cargo_toml}")


def normalize_version(raw_version: str) -> str:
    """Normalize a SemVer version string by stripping any leading ``v``."""
    cleaned = raw_version.strip()
    if cleaned.startswith(("v", "V")):
        cleaned = cleaned[1:]
    if not re.match(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?$", cleaned):
        raise ValueError(f"Invalid semantic version string: {raw_version!r}")
    return cleaned


def _clamp_u8(val: float) -> int:
    if val <= 0.0:
        return 0
    if val >= 255.0:
        return 255
    return int(val + 0.5)


def _blend_pixel(
    buf: bytearray,
    size: int,
    px: int,
    py: int,
    r: int,
    g: int,
    b: int,
    alpha: float,
) -> None:
    """Alpha-blend ``(r, g, b, alpha)`` over pixel ``(px, py)`` in PNG scanline buffer."""
    if alpha <= 0.0 or px < 0 or py < 0 or px >= size or py >= size:
        return
    if alpha > 1.0:
        alpha = 1.0
    row_stride = 1 + size * 4
    idx = py * row_stride + 1 + px * 4
    dst_a = buf[idx + 3] / 255.0
    if dst_a <= 0.0:
        return  # Clip strictly to inside of rounded-rect card
    inv = 1.0 - alpha
    buf[idx] = _clamp_u8(r * alpha + buf[idx] * inv)
    buf[idx + 1] = _clamp_u8(g * alpha + buf[idx + 1] * inv)
    buf[idx + 2] = _clamp_u8(b * alpha + buf[idx + 2] * inv)


def _draw_segment_aa(
    buf: bytearray,
    size: int,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    half_width: float,
    color: Tuple[int, int, int],
    opacity: float,
) -> None:
    """Draw an anti-aliased rounded-cap line segment in normalized ``[0, 1]`` coords."""
    ax, ay = x0 * size, y0 * size
    bx, by = x1 * size, y1 * size
    hw = max(0.55, half_width * size)
    feather = 0.85
    pad = int(math.ceil(hw + feather + 1.0))
    xmin = max(0, int(min(ax, bx)) - pad)
    xmax = min(size - 1, int(max(ax, bx)) + pad)
    ymin = max(0, int(min(ay, by)) - pad)
    ymax = min(size - 1, int(max(ay, by)) + pad)
    vx, vy = bx - ax, by - ay
    seg_len_sq = vx * vx + vy * vy
    r, g, b = color
    for py in range(ymin, ymax + 1):
        cy = py + 0.5
        for px in range(xmin, xmax + 1):
            cx = px + 0.5
            if seg_len_sq > 1e-9:
                t = ((cx - ax) * vx + (cy - ay) * vy) / seg_len_sq
                t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
            else:
                t = 0.0
            dx = cx - (ax + t * vx)
            dy = cy - (ay + t * vy)
            dist = math.hypot(dx, dy)
            if dist <= hw + feather:
                cov = 1.0 if dist <= hw - 0.35 else max(0.0, (hw + feather - dist) / (feather + 0.35))
                _blend_pixel(buf, size, px, py, r, g, b, opacity * cov)


def _draw_contour_family(
    buf: bytearray,
    size: int,
    cx_n: float,
    cy_n: float,
    rx_n: float,
    ry_n: float,
    angle_deg: float,
    rings: Sequence[Tuple[float, Tuple[int, int, int], float]],
    core_color: Tuple[int, int, int],
) -> None:
    """Draw concentric tilted 2D NMR elliptical iso-contours and resonance core."""
    cx, cy = cx_n * size, cy_n * size
    rx_base = max(1.2, rx_n * size)
    ry_base = max(1.2, ry_n * size)
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    max_scale = max(scale for scale, _, _ in rings)
    stroke_px = max(0.75, size * 0.0135)
    bound = int(math.ceil(max(rx_base, ry_base) * max_scale + stroke_px + 2.5))
    xmin = max(0, int(cx) - bound)
    xmax = min(size - 1, int(cx) + bound)
    ymin = max(0, int(cy) - bound)
    ymax = min(size - 1, int(cy) + bound)

    for py in range(ymin, ymax + 1):
        dy = (py + 0.5) - cy
        for px in range(xmin, xmax + 1):
            dx = (px + 0.5) - cx
            ux = (dx * cos_t + dy * sin_t) / rx_base
            uy = (-dx * sin_t + dy * cos_t) / ry_base
            rho = math.hypot(ux, uy)
            if rho > max_scale + 0.45:
                continue

            # Soft spectral glow around peak center
            if rho < max_scale:
                glow = math.exp(-2.6 * rho * rho) * 0.28
                _, outer_col, _ = rings[0]
                _blend_pixel(buf, size, px, py, outer_col[0], outer_col[1], outer_col[2], glow)

            # Concentric iso-contour rings
            approx_px_per_rho = 0.5 * (rx_base + ry_base)
            for scale, col, op in rings:
                dist_px = abs(rho - scale) * approx_px_per_rho
                if dist_px <= stroke_px + 0.75:
                    cov = 1.0 if dist_px <= stroke_px * 0.45 else max(
                        0.0, (stroke_px + 0.75 - dist_px) / (stroke_px * 0.55 + 0.75)
                    )
                    _blend_pixel(buf, size, px, py, col[0], col[1], col[2], op * cov)

            # Bright spin-resonance peak maximum core
            core_r_px = max(0.95, size * 0.026)
            dist_center_px = math.hypot(dx, dy)
            if dist_center_px <= core_r_px + 0.85:
                cov = 1.0 if dist_center_px <= core_r_px - 0.25 else max(
                    0.0, (core_r_px + 0.85 - dist_center_px) / 1.1
                )
                _blend_pixel(buf, size, px, py, core_color[0], core_color[1], core_color[2], 0.96 * cov)


def _sdf_rounded_rect(px: float, py: float, x: float, y: float, w: float, h: float, rx: float) -> float:
    cx = x + 0.5 * w
    cy = y + 0.5 * h
    hx = max(0.0, 0.5 * w - rx)
    hy = max(0.0, 0.5 * h - rx)
    qx = abs(px - cx) - hx
    qy = abs(py - cy) - hy
    return math.hypot(max(qx, 0.0), max(qy, 0.0)) + min(max(qx, qy), 0.0) - rx


def _sdf_segment(px: float, py: float, x1: float, y1: float, x2: float, y2: float, width: float) -> float:
    bax = x2 - x1
    bay = y2 - y1
    denom = bax * bax + bay * bay
    if denom <= 1e-12:
        return math.hypot(px - x1, py - y1) - 0.5 * width
    h = ((px - x1) * bax + (py - y1) * bay) / denom
    h = 0.0 if h < 0.0 else (1.0 if h > 1.0 else h)
    return math.hypot((px - x1) - bax * h, (py - y1) - bay * h) - 0.5 * width


def _sample_concept_a_fid_pts() -> Tuple[Tuple[float, float], ...]:
    pts: List[Tuple[float, float]] = []
    samples = 120
    for i in range(samples + 1):
        u = i / samples
        x = 168.0 + (414.0 - 168.0) * u
        y = 248.0 - 120.0 * math.exp(-2.7 * u) * math.cos(2.0 * math.pi * 2.5 * u)
        pts.append((x, y))
    return tuple(pts)


def render_icon_scanlines(size: int) -> bytes:
    """Deterministically rasterize the canonical Concept A ``FERRIC`` icon at ``size x size`` RGBA."""
    if size <= 0 or size > 2048:
        raise ValueError(f"Unsupported icon dimension: {size}")

    fid_pts = _sample_concept_a_fid_pts()
    fid_segs = [
        (
            fid_pts[i][0],
            fid_pts[i][1],
            fid_pts[i + 1][0],
            fid_pts[i + 1][1],
            min(fid_pts[i][0], fid_pts[i + 1][0]) - 14.0,
            max(fid_pts[i][0], fid_pts[i + 1][0]) + 14.0,
            min(fid_pts[i][1], fid_pts[i + 1][1]) - 14.0,
            max(fid_pts[i][1], fid_pts[i + 1][1]) + 14.0,
        )
        for i in range(len(fid_pts) - 1)
    ]

    # Colors matching docs/assets/brand/ferric-icon.svg 1-to-1
    col_bg = (22, 32, 42)          # #16202A
    col_frame_outer = (183, 65, 14)  # #B7410E
    col_frame_inner = (217, 91, 38)  # #D95B26
    col_axis = (71, 85, 105)       # #475569
    col_f_outer = (183, 65, 14)    # #B7410E
    col_f_inner = (224, 90, 43)    # #E05A2B
    col_fid = (251, 146, 60)       # #FB923C
    col_cap = (254, 215, 170)      # #FED7AA

    def _eval_sample(vx: float, vy: float) -> Optional[Tuple[int, int, int]]:
        # Top-down layer evaluation (first hit wins)
        if (vx - 152.0) ** 2 + (vy - 124.0) ** 2 <= 256.0:
            return col_cap

        if 154.0 <= vx <= 428.0 and 114.0 <= vy <= 333.0:
            for ax, ay, bx, by, x0, x1, y0, y1 in fid_segs:
                if x0 <= vx <= x1 and y0 <= vy <= y1:
                    if _sdf_segment(vx, vy, ax, ay, bx, by, 28.0) <= 0.0:
                        return col_fid

        if (
            _sdf_segment(vx, vy, 152.0, 124.0, 152.0, 388.0, 34.0) <= 0.0
            or _sdf_segment(vx, vy, 152.0, 124.0, 376.0, 124.0, 34.0) <= 0.0
            or _sdf_segment(vx, vy, 152.0, 248.0, 316.0, 248.0, 28.0) <= 0.0
        ):
            return col_f_inner

        if (
            _sdf_segment(vx, vy, 152.0, 124.0, 152.0, 388.0, 56.0) <= 0.0
            or _sdf_segment(vx, vy, 152.0, 124.0, 376.0, 124.0, 56.0) <= 0.0
            or _sdf_segment(vx, vy, 152.0, 248.0, 316.0, 248.0, 48.0) <= 0.0
        ):
            return col_f_outer

        if _sdf_segment(vx, vy, 174.0, 248.0, 416.0, 248.0, 10.0) <= 0.0:
            return col_axis

        d_inner = _sdf_rounded_rect(vx, vy, 44.0, 44.0, 424.0, 424.0, 92.0)
        if abs(d_inner) <= 4.0:
            return col_frame_inner

        d_outer = _sdf_rounded_rect(vx, vy, 32.0, 32.0, 448.0, 448.0, 104.0)
        if abs(d_outer) <= 12.0:
            return col_frame_outer
        if d_outer <= 0.0:
            return col_bg
        return None

    scale = 512.0 / size
    edge_pad = 14.0 * scale
    sub_offsets = (0.125, 0.375, 0.625, 0.875)
    row_stride = 1 + size * 4
    buf = bytearray(size * row_stride)

    for py in range(size):
        vy_c = (py + 0.5) * scale
        row_off = py * row_stride
        buf[row_off] = 0
        for px in range(size):
            vx_c = (px + 0.5) * scale
            p_off = row_off + 1 + px * 4
            d_outer_c = _sdf_rounded_rect(vx_c, vy_c, 32.0, 32.0, 448.0, 448.0, 104.0)
            if d_outer_c > 12.0 + edge_pad:
                continue

            acc_r = acc_g = acc_b = acc_a = 0.0
            for oy in sub_offsets:
                vy = (py + oy) * scale
                for ox in sub_offsets:
                    vx = (px + ox) * scale
                    hit = _eval_sample(vx, vy)
                    if hit is not None:
                        acc_r += hit[0]
                        acc_g += hit[1]
                        acc_b += hit[2]
                        acc_a += 1.0
            if acc_a > 0.0:
                buf[p_off] = _clamp_u8(acc_r / acc_a)
                buf[p_off + 1] = _clamp_u8(acc_g / acc_a)
                buf[p_off + 2] = _clamp_u8(acc_b / acc_a)
                buf[p_off + 3] = _clamp_u8((acc_a / 16.0) * 255.0)

    return bytes(buf)


def _png_chunk(tag: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(tag + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", crc)


def encode_png_rgba(size: int, raw_scanlines: bytes) -> bytes:
    """Encode RGBA scanlines (with filter byte ``0`` per row) into an RFC 2083 PNG."""
    expected_len = size * (1 + size * 4)
    if len(raw_scanlines) != expected_len:
        raise ValueError(f"Expected {expected_len} scanline bytes for {size}x{size}, got {len(raw_scanlines)}")
    ihdr = struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)
    idat = zlib.compress(raw_scanlines, 9)
    return PNG_MAGIC + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", idat) + _png_chunk(b"IEND", b"")


def generate_icon_pngs(sizes: Sequence[int] = PNG_SIZES) -> Dict[int, bytes]:
    """Generate deterministic multi-size PNG bytes keyed by pixel dimension."""
    return {sz: encode_png_rgba(sz, render_icon_scanlines(sz)) for sz in sizes}


def build_icns(png_by_size: Dict[int, bytes]) -> bytes:
    """Pack PNG representations into an Apple ``.icns`` container (`ic07`/`ic08`/`ic09` + Retina)."""
    chunks: List[bytes] = []
    for ostype, sz in ICNS_SLOTS:
        if sz not in png_by_size:
            raise KeyError(f"Missing {sz}x{sz} PNG required for ICNS slot {ostype!r}")
        png_data = png_by_size[sz]
        chunks.append(ostype + struct.pack(">I", 8 + len(png_data)) + png_data)
    body = b"".join(chunks)
    return b"icns" + struct.pack(">I", 8 + len(body)) + body


def build_ico(png_by_size: Dict[int, bytes], sizes: Sequence[int] = ICO_SIZES) -> bytes:
    """Pack PNG representations into a Windows multi-resolution ``.ico`` container."""
    count = len(sizes)
    header = struct.pack("<HHH", 0, 1, count)
    offset = 6 + 16 * count
    entries: List[bytes] = []
    payloads: List[bytes] = []
    for sz in sizes:
        if sz not in png_by_size:
            raise KeyError(f"Missing {sz}x{sz} PNG required for ICO container")
        png_data = png_by_size[sz]
        dim_byte = 0 if sz >= 256 else sz
        entries.append(
            struct.pack(
                "<BBBBHHII",
                dim_byte,
                dim_byte,
                0,
                0,
                1,
                32,
                len(png_data),
                offset,
            )
        )
        payloads.append(png_data)
        offset += len(png_data)
    return header + b"".join(entries) + b"".join(payloads)


def render_icon_svg() -> str:
    """Render the matching scalable vector ``ferric.svg`` FreeDesktop launcher icon (Concept A)."""
    pts = _sample_concept_a_fid_pts()
    cmd_parts = [f"M {pts[0][0]:.2f}".rstrip("0").rstrip(".") + f" {pts[0][1]:.2f}".rstrip("0").rstrip(".")]
    for px, py in pts[1:]:
        sx = f"{px:.2f}".rstrip("0").rstrip(".")
        sy = f"{py:.2f}".rstrip("0").rstrip(".")
        cmd_parts.append(f"L {sx} {sy}")
    d_str = " ".join(cmd_parts)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512" role="img">\n'
        "  <title>FERRIC Icon</title>\n"
        '  <rect x="32" y="32" width="448" height="448" rx="104" ry="104" fill="#16202A" stroke="#B7410E" stroke-width="24"/>\n'
        '  <rect x="44" y="44" width="424" height="424" rx="92" ry="92" fill="none" stroke="#D95B26" stroke-width="8"/>\n'
        '  <line x1="174" y1="248" x2="416" y2="248" stroke="#475569" stroke-width="10" stroke-linecap="round"/>\n'
        '  <line x1="152" y1="124" x2="152" y2="388" stroke="#B7410E" stroke-width="56" stroke-linecap="round"/>\n'
        '  <line x1="152" y1="124" x2="376" y2="124" stroke="#B7410E" stroke-width="56" stroke-linecap="round"/>\n'
        '  <line x1="152" y1="248" x2="316" y2="248" stroke="#B7410E" stroke-width="48" stroke-linecap="round"/>\n'
        '  <line x1="152" y1="124" x2="152" y2="388" stroke="#E05A2B" stroke-width="34" stroke-linecap="round"/>\n'
        '  <line x1="152" y1="124" x2="376" y2="124" stroke="#E05A2B" stroke-width="34" stroke-linecap="round"/>\n'
        '  <line x1="152" y1="248" x2="316" y2="248" stroke="#E05A2B" stroke-width="28" stroke-linecap="round"/>\n'
        f'  <path d="{d_str}" fill="none" stroke="#FB923C" stroke-width="28" stroke-linecap="round" stroke-linejoin="round"/>\n'
        '  <circle cx="152" cy="124" r="16" fill="#FED7AA"/>\n'
        "</svg>\n"
    )


def validate_png_bytes(data: bytes, expected_size: int) -> None:
    """Validate RFC 2083 PNG header, chunk CRCs, dimensions, and RGBA payload length."""
    if not data.startswith(PNG_MAGIC):
        raise ValueError("Invalid PNG signature")
    pos = len(PNG_MAGIC)
    saw_ihdr = False
    saw_iend = False
    idat_parts: List[bytes] = []
    while pos < len(data):
        if pos + 8 > len(data):
            raise ValueError("Truncated PNG chunk header")
        length, tag = struct.unpack(">I4s", data[pos : pos + 8])
        pos += 8
        if pos + length + 4 > len(data):
            raise ValueError(f"Truncated PNG chunk {tag!r}")
        payload = data[pos : pos + length]
        pos += length
        (expected_crc,) = struct.unpack(">I", data[pos : pos + 4])
        pos += 4
        actual_crc = zlib.crc32(tag + payload) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise ValueError(f"CRC mismatch in PNG chunk {tag!r}")
        if tag == b"IHDR":
            w, h, bit_depth, color_type, comp, filt, interlace = struct.unpack(">IIBBBBB", payload)
            if (w, h) != (expected_size, expected_size):
                raise ValueError(f"PNG IHDR size {(w, h)} != expected {expected_size}")
            if (bit_depth, color_type, comp, filt, interlace) != (8, 6, 0, 0, 0):
                raise ValueError("Unexpected PNG IHDR pixel format")
            saw_ihdr = True
        elif tag == b"IDAT":
            idat_parts.append(payload)
        elif tag == b"IEND":
            saw_iend = True
            break
    if not (saw_ihdr and saw_iend and pos == len(data)):
        raise ValueError("Incomplete or trailing bytes in PNG stream")
    raw = zlib.decompress(b"".join(idat_parts))
    if len(raw) != expected_size * (1 + expected_size * 4):
        raise ValueError("Decompressed PNG scanline buffer size mismatch")


def validate_icns_bytes(data: bytes, verify_macos_tools: bool = True) -> None:
    """Validate Apple ``.icns`` header, `ic07`/`ic08`/`ic09` PNG slots, and macOS tools."""
    if len(data) < 16 or data[:4] != b"icns":
        raise ValueError("Invalid ICNS magic header")
    (total_len,) = struct.unpack(">I", data[4:8])
    if total_len != len(data):
        raise ValueError(f"ICNS header length {total_len} != actual size {len(data)}")
    pos = 8
    seen: Dict[bytes, bytes] = {}
    expected_slot_sizes = dict(ICNS_SLOTS)
    while pos < len(data):
        if pos + 8 > len(data):
            raise ValueError("Truncated ICNS element header")
        ostype = data[pos : pos + 4]
        (elem_len,) = struct.unpack(">I", data[pos + 4 : pos + 8])
        if elem_len < 8 or pos + elem_len > len(data):
            raise ValueError(f"Invalid ICNS element length {elem_len} for {ostype!r}")
        payload = data[pos + 8 : pos + elem_len]
        if ostype in expected_slot_sizes:
            validate_png_bytes(payload, expected_slot_sizes[ostype])
        seen[ostype] = payload
        pos += elem_len
    for req in (b"ic07", b"ic08", b"ic09"):
        if req not in seen:
            raise ValueError(f"Missing required ICNS slot {req!r}")

    if verify_macos_tools and sys.platform == "darwin":
        with tempfile.TemporaryDirectory() as tmp:
            icns_file = Path(tmp) / f"{APP_NAME}.icns"
            icns_file.write_bytes(data)
            if shutil.which("sips"):
                res = subprocess.run(
                    ["sips", "-g", "format", "-g", "pixelWidth", "-g", "pixelHeight", str(icns_file)],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                if "format: icns" not in res.stdout:
                    raise ValueError(f"sips failed to recognize ICNS format: {res.stdout}")
            if shutil.which("iconutil"):
                out_iconset = Path(tmp) / "verify.iconset"
                subprocess.run(
                    ["iconutil", "--convert", "iconset", "--output", str(out_iconset), str(icns_file)],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                extracted = {p.name for p in out_iconset.glob("*.png")}
                for req_png in ("icon_16x16.png", "icon_32x32.png", "icon_128x128.png", "icon_256x256.png", "icon_512x512.png"):
                    if req_png not in extracted:
                        raise ValueError(f"iconutil missing expected {req_png} in {sorted(extracted)}")


def validate_ico_bytes(data: bytes, expected_sizes: Sequence[int] = ICO_SIZES) -> None:
    """Validate Windows ``.ico`` directory entries and embedded multi-resolution PNGs."""
    if len(data) < 6:
        raise ValueError("Truncated ICO header")
    reserved, ico_type, count = struct.unpack("<HHH", data[:6])
    if (reserved, ico_type, count) != (0, 1, len(expected_sizes)):
        raise ValueError(f"Unexpected ICONDIR header {(reserved, ico_type, count)}")
    dir_end = 6 + 16 * count
    if len(data) < dir_end:
        raise ValueError("Truncated ICONDIRENTRY table")
    cursor = dir_end
    for idx, exp_sz in enumerate(expected_sizes):
        entry_off = 6 + idx * 16
        w_b, h_b, pal, res_b, planes, bpp, size_bytes, img_offset = struct.unpack(
            "<BBBBHHII", data[entry_off : entry_off + 16]
        )
        exp_dim = 0 if exp_sz >= 256 else exp_sz
        if (w_b, h_b, pal, res_b, planes, bpp) != (exp_dim, exp_dim, 0, 0, 1, 32):
            raise ValueError(f"Invalid ICONDIRENTRY metadata at index {idx}")
        if img_offset != cursor or img_offset + size_bytes > len(data):
            raise ValueError(f"Invalid ICONDIRENTRY offset/size at index {idx}")
        validate_png_bytes(data[img_offset : img_offset + size_bytes], exp_sz)
        cursor += size_bytes
    if cursor != len(data):
        raise ValueError("Trailing unreferenced bytes in ICO file")


def validate_svg_text(svg_text: str) -> None:
    """Validate standalone SVG XML structure and root attributes."""
    if not svg_text.startswith('<?xml version="1.0" encoding="UTF-8"?>'):
        raise ValueError("SVG missing UTF-8 XML declaration")
    root = ET.fromstring(svg_text)
    if not root.tag.endswith("svg"):
        raise ValueError(f"Unexpected SVG root element: {root.tag}")
    if root.attrib.get("viewBox") != "0 0 512 512":
        raise ValueError(f"Unexpected SVG viewBox: {root.attrib.get('viewBox')!r}")


def emit_icon_suite(out_dir: Path) -> Dict[str, Path]:
    """Generate and validate ``.png``, ``.icns``, ``.ico``, and ``.svg`` icons in ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pngs = generate_icon_pngs(PNG_SIZES)
    written: Dict[str, Path] = {}
    for sz in (16, 32, 48, 64, 128, 256, 512):
        validate_png_bytes(pngs[sz], sz)
        p = out_dir / f"{APP_SLUG}-{sz}.png"
        p.write_bytes(pngs[sz])
        written[f"png_{sz}"] = p

    icns_bytes = build_icns(pngs)
    validate_icns_bytes(icns_bytes)
    icns_path = out_dir / f"{APP_NAME}.icns"
    icns_path.write_bytes(icns_bytes)
    written["icns"] = icns_path

    ico_bytes = build_ico(pngs, ICO_SIZES)
    validate_ico_bytes(ico_bytes, ICO_SIZES)
    ico_path = out_dir / f"{APP_SLUG}.ico"
    ico_path.write_bytes(ico_bytes)
    written["ico"] = ico_path

    svg_text = render_icon_svg()
    validate_svg_text(svg_text)
    svg_path = out_dir / f"{APP_SLUG}.svg"
    _write_text_lf(svg_path, svg_text)
    written["svg"] = svg_path
    return written


def build_macos_info_plist_bytes(version: str) -> bytes:
    """Serialize macOS ``Contents/Info.plist`` XML dictionary."""
    info = {
        "CFBundleDevelopmentRegion": "en",
        "CFBundleDisplayName": APP_NAME,
        "CFBundleExecutable": GUI_BIN,
        "CFBundleIconFile": f"{APP_NAME}.icns",
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleInfoDictionaryVersion": "6.0",
        "CFBundleName": APP_NAME,
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": version,
        "CFBundleVersion": version,
        "LSMinimumSystemVersion": MACOS_MIN_SYSTEM_VERSION,
        "NSHighResolutionCapable": True,
        "NSSupportsAutomaticGraphicsSwitching": True,
    }
    return plistlib.dumps(info, fmt=plistlib.FMT_XML, sort_keys=True)


def validate_info_plist(plist_path: Path, expected_version: str) -> None:
    """Validate ``Info.plist`` required keys and run ``plutil -lint`` on macOS."""
    data = plistlib.loads(plist_path.read_bytes())
    expected_pairs = {
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleExecutable": GUI_BIN,
        "CFBundleIconFile": f"{APP_NAME}.icns",
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": expected_version,
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": MACOS_MIN_SYSTEM_VERSION,
    }
    for key, exp_val in expected_pairs.items():
        if data.get(key) != exp_val:
            raise ValueError(f"Info.plist key {key}={data.get(key)!r} != expected {exp_val!r}")
    if sys.platform == "darwin" and shutil.which("plutil"):
        subprocess.run(
            ["plutil", "-lint", str(plist_path)],
            capture_output=True,
            text=True,
            check=True,
        )


def render_cli_and_mcp_quickstart(version: str, arch_title: str = "Apple Silicon") -> str:
    """Render plain-text macOS quickstart bundled inside the ``.dmg`` volume."""
    header = f"FERRIC v{version} — {arch_title} Desktop, CLI & MCP Quickstart"
    rule = "=" * len(header)
    return f"""{header}
{rule}

1. Install the Desktop Workbench (`FERRIC.app`)
-----------------------------------------------
Drag `FERRIC.app` onto the `Applications` folder shortcut in this disk image.
Double-click `FERRIC` in `/Applications` to launch the interactive WebGPU
2D/3D NMR spectrum workbench (`Metal` GPU backend with multi-threaded CPU `f64`
fallback).

2. Command-Line Interface (`ferric`)
------------------------------------
Both `ferric-gui` and the `ferric` CLI / MCP server ship directly inside
`FERRIC.app/Contents/MacOS/`.
To add `ferric` to your `zsh` PATH:

  export PATH="/Applications/FERRIC.app/Contents/MacOS:$PATH"
  ferric --version

Example end-to-end raw-FID backbone assignment pipeline:

  ferric pipeline \\
    --seq ubiquitin.fasta \\
    --fid-hsqc ./10_hsqc \\
    --fid-triple HNCACB=./20_hncacb \\
    --fid-triple CBCACONH=./21_cbcaconh \\
    --out-dir ./ubq_out --engine auto

3. Sandboxed AI Research Copilot (`MCP` JSON-RPC 2.0 Server)
------------------------------------------------------------
Generate ready-to-paste client configuration JSON (read-only sandbox by default):

  /Applications/FERRIC.app/Contents/MacOS/ferric mcp-config \\
    --client claude-desktop --root ~/nmr_projects --mode read-only

Documentation & Browser Demo: https://debsahu.github.io/ferric/
Repository & Source Code:     https://github.com/debsahu/ferric
"""


def render_windows_launcher_bat() -> bytes:
    """Render ``Launch-NMR-Workbench.bat`` with Windows CRLF line endings."""
    lines = [
        "@echo off",
        "setlocal",
        'if "%~1"=="--cli-help" (',
        f'  "%~dp0{CLI_BIN}.exe" --help',
        "  exit /b %ERRORLEVEL%",
        ")",
        f'start "" "%~dp0{GUI_BIN}.exe" %*',
        "",
    ]
    return "\r\n".join(lines).encode("utf-8")


def render_linux_desktop_entry() -> str:
    """Render FreeDesktop ``ferric.desktop`` specification entry."""
    return f"""[Desktop Entry]
Type=Application
Version=1.5
Name={APP_NAME}
GenericName=NMR Resonance Assignment & Spectrum Workbench
Comment=Open-source GPU-accelerated protein and analytical NMR spectrum workbench
Exec={GUI_BIN} %F
TryExec={GUI_BIN}
Icon={APP_SLUG}
Terminal=false
Categories=Science;Chemistry;Biology;DataVisualization;
Keywords=NMR;Spectroscopy;Protein;ChemicalShift;Metabolomics;DOSY;WebGPU;
StartupWMClass={GUI_BIN}
StartupNotify=true
"""


def validate_desktop_entry(desktop_text: str) -> None:
    """Validate required FreeDesktop ``.desktop`` keys and category formatting."""
    lines = [ln.strip() for ln in desktop_text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if not lines or lines[0] != "[Desktop Entry]":
        raise ValueError("Desktop entry must start with [Desktop Entry]")
    kv: Dict[str, str] = {}
    for ln in lines[1:]:
        if "=" not in ln:
            raise ValueError(f"Malformed .desktop line: {ln!r}")
        k, v = ln.split("=", 1)
        kv[k] = v
    required = {
        "Type": "Application",
        "Name": APP_NAME,
        "Exec": f"{GUI_BIN} %F",
        "TryExec": GUI_BIN,
        "Icon": APP_SLUG,
        "Terminal": "false",
    }
    for k, exp in required.items():
        if kv.get(k) != exp:
            raise ValueError(f".desktop key {k}={kv.get(k)!r} != expected {exp!r}")
    cats = kv.get("Categories", "")
    if not cats.endswith(";") or "Science;" not in cats:
        raise ValueError(f"Invalid FreeDesktop Categories value: {cats!r}")


def render_linux_install_desktop_sh() -> str:
    """Render ``install-desktop.sh`` user-level XDG desktop & binary installer."""
    return """#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX_DIR="${HOME}/.local"
UNINSTALL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix)
      PREFIX_DIR="$2"
      shift 2
      ;;
    --uninstall)
      UNINSTALL=1
      shift
      ;;
    -h|--help)
      echo "Usage: ./install-desktop.sh [--prefix ~/.local] [--uninstall]"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

BIN_DIR="${PREFIX_DIR}/bin"
DATA_DIR="${XDG_DATA_HOME:-${PREFIX_DIR}/share}"
if [[ "${PREFIX_DIR}" != "${HOME}/.local" ]]; then
  DATA_DIR="${PREFIX_DIR}/share"
fi

APP_DIR="${DATA_DIR}/applications"
ICON_SVG_DIR="${DATA_DIR}/icons/hicolor/scalable/apps"
ICON_PNG_DIR="${DATA_DIR}/icons/hicolor/256x256/apps"

if [[ "${UNINSTALL}" -eq 1 ]]; then
  rm -f "${BIN_DIR}/ferric-gui" "${BIN_DIR}/ferric" "${BIN_DIR}/nmrassign" "${BIN_DIR}/nmrassign-gui"
  rm -f "${APP_DIR}/ferric.desktop" "${APP_DIR}/nmr-assign.desktop"
  rm -f "${ICON_SVG_DIR}/ferric.svg" "${ICON_SVG_DIR}/nmr-assign.svg"
  rm -f "${ICON_PNG_DIR}/ferric.png" "${DATA_DIR}"/icons/hicolor/*/apps/nmr-assign.png
  echo "Uninstalled FERRIC from ${PREFIX_DIR}"
  exit 0
fi

# Clean up legacy v0.13.0/v0.14.x launcher/icons/deprecated alias binaries if upgrading in place
rm -f "${BIN_DIR}/nmrassign" "${BIN_DIR}/nmrassign-gui"
rm -f "${APP_DIR}/nmr-assign.desktop"
rm -f "${ICON_SVG_DIR}/nmr-assign.svg"
rm -f "${DATA_DIR}"/icons/hicolor/*/apps/nmr-assign.png

mkdir -p "${BIN_DIR}" "${APP_DIR}" "${ICON_SVG_DIR}" "${ICON_PNG_DIR}"
install -m 0755 "${SCRIPT_DIR}/bin/ferric-gui" "${BIN_DIR}/ferric-gui"
install -m 0755 "${SCRIPT_DIR}/bin/ferric" "${BIN_DIR}/ferric"
install -m 0644 "${SCRIPT_DIR}/share/icons/hicolor/scalable/apps/ferric.svg" "${ICON_SVG_DIR}/ferric.svg"
install -m 0644 "${SCRIPT_DIR}/share/icons/hicolor/256x256/apps/ferric.png" "${ICON_PNG_DIR}/ferric.png"

sed \\
  -e "s|^Exec=ferric-gui|Exec=${BIN_DIR}/ferric-gui|" \\
  -e "s|^TryExec=ferric-gui|TryExec=${BIN_DIR}/ferric-gui|" \\
  "${SCRIPT_DIR}/share/applications/ferric.desktop" > "${APP_DIR}/ferric.desktop"
chmod 0644 "${APP_DIR}/ferric.desktop"

if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database "${APP_DIR}" >/dev/null 2>&1 || true
fi
if command -v gtk-update-icon-cache >/dev/null 2>&1; then
  gtk-update-icon-cache -f -t "${DATA_DIR}/icons/hicolor" >/dev/null 2>&1 || true
fi

echo "Installed FERRIC desktop launcher and CLI to ${PREFIX_DIR}"
"""


def _source_mtime() -> int:
    raw = os.environ.get("SOURCE_DATE_EPOCH")
    return int(raw) if raw is not None else DETERMINISTIC_MTIME


def _write_deterministic_tar_gz(
    staging_root: Path,
    archive_prefix: str,
    out_tar_gz: Path,
    executable_relpaths: Sequence[str],
) -> None:
    """Create a reproducible PAX ``.tar.gz`` with normalized uid/gid/mtime/modes."""
    exec_set = set(executable_relpaths)
    mtime = _source_mtime()
    out_tar_gz.parent.mkdir(parents=True, exist_ok=True)

    all_paths = sorted(staging_root.rglob("*"), key=lambda p: p.relative_to(staging_root).as_posix())
    with out_tar_gz.open("wb") as raw_fh:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_fh, compresslevel=9, mtime=0) as gz_fh:
            with tarfile.open(mode="w", fileobj=gz_fh, format=tarfile.PAX_FORMAT) as tf:
                root_ti = tarfile.TarInfo(name=archive_prefix)
                root_ti.type = tarfile.DIRTYPE
                root_ti.mode = 0o755
                root_ti.uid = 0
                root_ti.gid = 0
                root_ti.uname = ""
                root_ti.gname = ""
                root_ti.mtime = mtime
                tf.addfile(root_ti)

                for item in all_paths:
                    rel_posix = item.relative_to(staging_root).as_posix()
                    arcname = f"{archive_prefix}/{rel_posix}"
                    ti = tarfile.TarInfo(name=arcname)
                    ti.uid = 0
                    ti.gid = 0
                    ti.uname = ""
                    ti.gname = ""
                    ti.mtime = mtime

                    if item.is_symlink():
                        ti.type = tarfile.SYMTYPE
                        ti.linkname = os.readlink(item)
                        ti.mode = 0o777
                        tf.addfile(ti)
                    elif item.is_dir():
                        ti.type = tarfile.DIRTYPE
                        ti.mode = 0o755
                        tf.addfile(ti)
                    else:
                        data = item.read_bytes()
                        ti.type = tarfile.REGTYPE
                        ti.size = len(data)
                        is_exec = rel_posix in exec_set or bool(item.stat().st_mode & stat.S_IXUSR)
                        ti.mode = 0o755 if is_exec else 0o644
                        from io import BytesIO

                        tf.addfile(ti, BytesIO(data))


def _write_deterministic_zip(
    staging_root: Path,
    archive_prefix: str,
    out_zip: Path,
    executable_relpaths: Sequence[str],
) -> None:
    """Create a reproducible ``.zip`` archive with fixed timestamps and sorted entries."""
    exec_set = set(executable_relpaths)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(
        (p for p in staging_root.rglob("*") if p.is_file()),
        key=lambda p: p.relative_to(staging_root).as_posix(),
    )
    with zipfile.ZipFile(out_zip, mode="w") as zf:
        for item in files:
            rel_posix = item.relative_to(staging_root).as_posix()
            arcname = f"{archive_prefix}/{rel_posix}"
            zi = zipfile.ZipInfo(filename=arcname, date_time=DETERMINISTIC_ZIP_DT)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.create_system = 3
            mode = 0o100755 if rel_posix in exec_set else 0o100644
            zi.external_attr = mode << 16
            zf.writestr(zi, item.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def _require_binary(bin_dir: Path, name: str) -> Path:
    candidate = bin_dir / name
    if not candidate.is_file():
        raise FileNotFoundError(
            f"Required binary {name!r} not found in {bin_dir}. "
            "Build release binaries first (e.g. `cargo build --release --workspace --bins`)."
        )
    return candidate


def _codesign_macos_app(app_dir: Path) -> None:
    """Apply ad-hoc macOS code signature (arm64 or x86_64) and strict verification."""
    if sys.platform != "darwin" or not shutil.which("codesign"):
        return
    cli_bin = app_dir / "Contents" / "MacOS" / CLI_BIN
    gui_bin = app_dir / "Contents" / "MacOS" / GUI_BIN
    subprocess.run(["codesign", "--force", "--sign", "-", str(cli_bin)], check=True, capture_output=True, text=True)
    subprocess.run(["codesign", "--force", "--sign", "-", str(gui_bin)], check=True, capture_output=True, text=True)
    subprocess.run(
        ["codesign", "--force", "--deep", "--sign", "-", str(app_dir)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["codesign", "--verify", "--deep", "--strict", str(app_dir)],
        check=True,
        capture_output=True,
        text=True,
    )


def _codesign_macos_binary(bin_path: Path) -> None:
    if sys.platform != "darwin" or not shutil.which("codesign"):
        return
    subprocess.run(["codesign", "--force", "--sign", "-", str(bin_path)], check=True, capture_output=True, text=True)


def _create_macos_dmg(volname: str, src_folder: Path, out_dmg: Path) -> None:
    """Create compressed UDZO ``.dmg`` via ``hdiutil``, retrying transient Spotlight locks."""
    if not shutil.which("hdiutil"):
        raise RuntimeError("hdiutil is required to build macOS .dmg images")
    out_dmg.parent.mkdir(parents=True, exist_ok=True)
    if out_dmg.exists():
        out_dmg.unlink()
    cmd = [
        "hdiutil",
        "create",
        "-volname",
        volname,
        "-srcfolder",
        str(src_folder),
        "-ov",
        "-format",
        "UDZO",
        str(out_dmg),
    ]
    last_err: Optional[subprocess.CalledProcessError] = None
    for attempt in range(3):
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return
        except subprocess.CalledProcessError as exc:
            last_err = exc
            time.sleep(1.0 * (attempt + 1))
    assert last_err is not None
    raise RuntimeError(f"hdiutil create failed after 3 attempts: {last_err.stderr}") from last_err


def _package_macos_bundle(
    repo_root: Path,
    bin_dir: Path,
    out_dir: Path,
    version: str,
    platform_slug: str,
    arch_title: str,
    skip_dmg: bool = False,
) -> List[Path]:
    """Build ``FERRIC.app``, ``.dmg``, and ``.tar.gz`` for ``macos-arm64`` or ``macos-x86_64``."""
    gui_src = _require_binary(bin_dir, GUI_BIN)
    cli_src = _require_binary(bin_dir, CLI_BIN)
    readme_src = repo_root / "README.md"
    third_party_src = repo_root / "THIRD_PARTY.md"

    out_dir.mkdir(parents=True, exist_ok=True)
    pngs = generate_icon_pngs(PNG_SIZES)
    icns_bytes = build_icns(pngs)
    validate_icns_bytes(icns_bytes)

    prefix = f"{APP_SLUG}-v{version}-{platform_slug}"
    outputs: List[Path] = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        tar_stage = tmp_path / "tar_stage"
        app_dir = tar_stage / f"{APP_NAME}.app"
        macos_dir = app_dir / "Contents" / "MacOS"
        res_dir = app_dir / "Contents" / "Resources"
        macos_dir.mkdir(parents=True, exist_ok=True)
        res_dir.mkdir(parents=True, exist_ok=True)

        info_plist_path = app_dir / "Contents" / "Info.plist"
        info_plist_path.write_bytes(build_macos_info_plist_bytes(version))
        validate_info_plist(info_plist_path, version)

        (app_dir / "Contents" / "PkgInfo").write_bytes(b"APPL????")
        (res_dir / f"{APP_NAME}.icns").write_bytes(icns_bytes)

        gui_dst = macos_dir / GUI_BIN
        cli_dst = macos_dir / CLI_BIN
        shutil.copy2(gui_src, gui_dst)
        shutil.copy2(cli_src, cli_dst)
        os.chmod(gui_dst, 0o755)
        os.chmod(cli_dst, 0o755)

        _codesign_macos_app(app_dir)

        # Copy signed .app bundle to out_dir for direct inspection / CI verification
        out_app_dir = out_dir / f"{APP_NAME}.app"
        if out_app_dir.exists():
            shutil.rmtree(out_app_dir)
        shutil.copytree(app_dir, out_app_dir, symlinks=True)

        # Top-level bin/ferric + docs inside .tar.gz staging root
        top_bin = tar_stage / "bin"
        top_bin.mkdir(parents=True, exist_ok=True)
        top_cli = top_bin / CLI_BIN
        shutil.copy2(cli_src, top_cli)
        os.chmod(top_cli, 0o755)
        _codesign_macos_binary(top_cli)

        shutil.copyfile(readme_src, tar_stage / "README.md")
        shutil.copyfile(third_party_src, tar_stage / "THIRD_PARTY.md")

        tar_gz_path = out_dir / f"{prefix}.tar.gz"
        _write_deterministic_tar_gz(
            staging_root=tar_stage,
            archive_prefix=prefix,
            out_tar_gz=tar_gz_path,
            executable_relpaths=(
                f"{APP_NAME}.app/Contents/MacOS/{GUI_BIN}",
                f"{APP_NAME}.app/Contents/MacOS/{CLI_BIN}",
                f"bin/{CLI_BIN}",
            ),
        )
        outputs.append(tar_gz_path)

        if not skip_dmg:
            if sys.platform != "darwin" or not shutil.which("hdiutil"):
                raise RuntimeError(
                    f"Building {platform_slug} .dmg requires macOS `hdiutil` (pass --skip-dmg to build .tar.gz only)."
                )
            dmg_stage = tmp_path / "dmg_stage"
            dmg_stage.mkdir(parents=True, exist_ok=True)
            shutil.copytree(app_dir, dmg_stage / f"{APP_NAME}.app", symlinks=True)
            os.symlink("/Applications", dmg_stage / "Applications")
            shutil.copyfile(readme_src, dmg_stage / "README.md")
            _write_text_lf(
                dmg_stage / "CLI_AND_MCP_QUICKSTART.txt",
                render_cli_and_mcp_quickstart(version, arch_title=arch_title),
            )
            dmg_path = out_dir / f"{prefix}.dmg"
            _create_macos_dmg(f"{APP_NAME} v{version}", dmg_stage, dmg_path)
            outputs.insert(0, dmg_path)

    return outputs


def package_macos_arm64(
    repo_root: Path,
    bin_dir: Path,
    out_dir: Path,
    version: str,
    skip_dmg: bool = False,
) -> List[Path]:
    """Build ``FERRIC.app``, ``.dmg``, and ``.tar.gz`` for ``macos-arm64``."""
    return _package_macos_bundle(
        repo_root=repo_root,
        bin_dir=bin_dir,
        out_dir=out_dir,
        version=version,
        platform_slug="macos-arm64",
        arch_title="Apple Silicon",
        skip_dmg=skip_dmg,
    )


def package_macos_x86_64(
    repo_root: Path,
    bin_dir: Path,
    out_dir: Path,
    version: str,
    skip_dmg: bool = False,
) -> List[Path]:
    """Build ``FERRIC.app``, ``.dmg``, and ``.tar.gz`` for Intel ``macos-x86_64``."""
    return _package_macos_bundle(
        repo_root=repo_root,
        bin_dir=bin_dir,
        out_dir=out_dir,
        version=version,
        platform_slug="macos-x86_64",
        arch_title="Intel (x86_64)",
        skip_dmg=skip_dmg,
    )


def package_windows_x86_64(
    repo_root: Path,
    bin_dir: Path,
    out_dir: Path,
    version: str,
) -> List[Path]:
    """Build deterministic ``.zip`` release bundle for ``windows-x86_64``."""
    gui_src = _require_binary(bin_dir, f"{GUI_BIN}.exe")
    cli_src = _require_binary(bin_dir, f"{CLI_BIN}.exe")
    readme_src = repo_root / "README.md"
    third_party_src = repo_root / "THIRD_PARTY.md"

    pngs = generate_icon_pngs(ICO_SIZES)
    ico_bytes = build_ico(pngs, ICO_SIZES)
    validate_ico_bytes(ico_bytes, ICO_SIZES)

    prefix = f"{APP_SLUG}-v{version}-windows-x86_64"
    out_zip = out_dir / f"{prefix}.zip"

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / prefix
        stage.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(gui_src, stage / f"{GUI_BIN}.exe")
        shutil.copyfile(cli_src, stage / f"{CLI_BIN}.exe")
        (stage / f"{APP_SLUG}.ico").write_bytes(ico_bytes)
        (stage / "Launch-NMR-Workbench.bat").write_bytes(render_windows_launcher_bat())
        shutil.copyfile(readme_src, stage / "README.md")
        shutil.copyfile(third_party_src, stage / "THIRD_PARTY.md")

        _write_deterministic_zip(
            staging_root=stage,
            archive_prefix=prefix,
            out_zip=out_zip,
            executable_relpaths=(
                f"{GUI_BIN}.exe",
                f"{CLI_BIN}.exe",
                "Launch-NMR-Workbench.bat",
            ),
        )

    return [out_zip]


def _deb_safe_version(version: str) -> str:
    return f"{version.replace('-', '~')}-1"


def _rpm_safe_version(version: str) -> str:
    return version.replace("-", "~")


def _md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def render_deb_control(
    version: str,
    deb_arch: str,
    installed_kib: int,
    include_gui: bool,
) -> str:
    """Render Debian ``DEBIAN/control`` metadata for ``amd64`` desktop+CLI or ``i386`` static CLI."""
    lines = [
        f"Package: {APP_SLUG}",
        f"Version: {_deb_safe_version(version)}",
        "Section: science",
        "Priority: optional",
        f"Architecture: {deb_arch}",
        f"Installed-Size: {max(1, installed_kib)}",
    ]
    if include_gui:
        lines.append(f"Depends: {', '.join(GUI_DEB_DEPENDS)}")
        lines.append(f"Recommends: {', '.join(GUI_DEB_RECOMMENDS)}")
        synopsis = "Open-source GPU-accelerated protein NMR assignment and spectrum workbench"
        ext_desc = (
            " FERRIC provides automated protein NMR backbone/sidechain resonance assignment,\n"
            " raw Bruker/Varian FID and NUS processing, spin relaxation/CEST/CPMG/CSP\n"
            " dynamics, 1D analytical NMR/DOSY/metabolomics, an interactive WebGPU desktop\n"
            " workbench (ferric-gui), and a CLI / MCP server (ferric)."
        )
    else:
        synopsis = "Open-source protein NMR assignment and analytical spectroscopy CLI (32-bit static)"
        ext_desc = (
            " Statically linked 32-bit x86 (i686 musl) command-line interface and MCP server\n"
            " for FERRIC automated protein NMR resonance assignment, raw FID processing,\n"
            " spin relaxation/exchange, and 1D analytical NMR/DOSY/metabolomics."
        )
    lines.extend(
        [
            f"Maintainer: {DEB_MAINTAINER}",
            f"Homepage: {PROJECT_URL}",
            f"Description: {synopsis}",
            ext_desc,
        ]
    )
    return "\n".join(lines) + "\n"


def validate_deb_control(
    control_text: str,
    expected_arch: str,
    expect_glibc_dep: bool,
) -> None:
    """Validate Debian ``DEBIAN/control`` fields and architecture/dependency invariants."""
    if not control_text.endswith("\n"):
        raise ValueError("DEBIAN/control must end with a trailing newline")
    fields: Dict[str, str] = {}
    saw_continuation = False
    last_key: Optional[str] = None
    for raw_line in control_text.splitlines():
        if not raw_line.strip():
            raise ValueError("Unexpected blank stanza separator in single-package DEBIAN/control")
        if raw_line.startswith(" "):
            if last_key != "Description":
                raise ValueError(f"Unexpected continuation line under {last_key!r}: {raw_line!r}")
            saw_continuation = True
            continue
        if ":" not in raw_line:
            raise ValueError(f"Malformed DEBIAN/control header line: {raw_line!r}")
        k, v = raw_line.split(":", 1)
        fields[k.strip()] = v.strip()
        last_key = k.strip()

    expected_exact = {
        "Package": APP_SLUG,
        "Section": "science",
        "Priority": "optional",
        "Architecture": expected_arch,
        "Maintainer": DEB_MAINTAINER,
        "Homepage": PROJECT_URL,
    }
    for key, exp_val in expected_exact.items():
        if fields.get(key) != exp_val:
            raise ValueError(f"DEBIAN/control {key}={fields.get(key)!r} != expected {exp_val!r}")
    if int(fields.get("Installed-Size", "0")) <= 0:
        raise ValueError(f"Invalid DEBIAN/control Installed-Size: {fields.get('Installed-Size')!r}")
    if not fields.get("Description") or not saw_continuation:
        raise ValueError("DEBIAN/control Description must include synopsis and extended lines")
    if expect_glibc_dep:
        deps = [tok.strip() for tok in fields.get("Depends", "").split(",") if tok.strip()]
        for req_dep in GUI_DEB_DEPENDS:
            if req_dep not in deps:
                raise ValueError(f"Expected {req_dep!r} in DEBIAN/control Depends, got {deps!r}")
        recs = [tok.strip() for tok in fields.get("Recommends", "").split(",") if tok.strip()]
        for rec_dep in GUI_DEB_RECOMMENDS:
            if rec_dep not in recs:
                raise ValueError(f"Expected {rec_dep!r} in DEBIAN/control Recommends, got {recs!r}")
    else:
        if "Depends" in fields:
            raise ValueError(f"Static i386 musl CLI .deb must not declare shared Depends, got {fields['Depends']!r}")
        if "Recommends" in fields:
            raise ValueError(f"Static i386 musl CLI .deb must not declare GUI Recommends, got {fields['Recommends']!r}")


def render_deb_copyright(license_text: str) -> str:
    """Render Debian policy DEP-5 ``/usr/share/doc/ferric/copyright`` metadata."""
    cleaned = license_text.strip()
    if not cleaned:
        raise ValueError("LICENSE file must not be empty")
    body_lines = [f" {ln}" if ln.strip() else " ." for ln in cleaned.splitlines()]
    return "\n".join(
        [
            "Format: https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/",
            f"Upstream-Name: {APP_SLUG}",
            f"Upstream-Contact: {DEB_MAINTAINER}",
            f"Source: {PROJECT_URL}",
            "",
            "Files: *",
            "Copyright: 2026 Debashish Sahu",
            "License: MIT",
            *body_lines,
            "Comment: Third-party crate licenses and open scientific data attributions are",
            f" documented in /usr/share/doc/{APP_SLUG}/THIRD_PARTY.md.",
        ]
    ) + "\n"


def validate_deb_copyright(copyright_text: str) -> None:
    """Validate Debian DEP-5 ``copyright`` structure and MIT license text."""
    if not copyright_text.endswith("\n"):
        raise ValueError("Debian copyright file must end with a trailing newline")
    required_lines = (
        "Format: https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/",
        f"Upstream-Name: {APP_SLUG}",
        f"Upstream-Contact: {DEB_MAINTAINER}",
        f"Source: {PROJECT_URL}",
        "Files: *",
        "Copyright: 2026 Debashish Sahu",
        "License: MIT",
    )
    lines = copyright_text.splitlines()
    for req in required_lines:
        if req not in lines:
            raise ValueError(f"Debian copyright missing required DEP-5 line: {req!r}")
    if "Permission is hereby granted, free of charge" not in copyright_text:
        raise ValueError("Debian copyright missing MIT permission clause")
    # Ensure multi-line License block has no unindented blank lines between License: MIT and Comment:
    lic_idx = lines.index("License: MIT")
    for ln in lines[lic_idx + 1 :]:
        if ln.startswith("Comment:"):
            break
        if not ln.startswith(" "):
            raise ValueError(f"Malformed DEP-5 License continuation line: {ln!r}")


def render_rpm_spec(
    version: str,
    stage_payload: Path,
    include_gui: bool,
) -> str:
    """Render RPM ``.spec`` file for ``x86_64`` desktop+CLI (RHEL 9+) or ``i686`` static CLI."""
    rpm_ver = _rpm_safe_version(version)
    if include_gui:
        summary = "Open-source GPU-accelerated protein NMR assignment and spectrum workbench"
        extra_header = "\n".join(
            [
                f"Requires:       {', '.join(GUI_RPM_REQUIRES)}",
                f"Recommends:     {', '.join(GUI_RPM_RECOMMENDS)}",
            ]
        )
        description = (
            "FERRIC provides automated protein NMR backbone and sidechain resonance\n"
            "assignment, raw Bruker/Varian FID and NUS processing, spin relaxation,\n"
            "CEST/CPMG/CSP dynamics, 1D analytical NMR/DOSY/metabolomics, an interactive\n"
            "WebGPU desktop workbench (ferric-gui), and a CLI / MCP server (ferric)."
        )
        files_entries = "\n".join(
            [
                f"%attr(0755,root,root) /usr/bin/{CLI_BIN}",
                f"%attr(0755,root,root) /usr/bin/{GUI_BIN}",
                f"%attr(0644,root,root) /usr/share/applications/{APP_SLUG}.desktop",
                f"%attr(0644,root,root) /usr/share/icons/hicolor/256x256/apps/{APP_SLUG}.png",
                f"%attr(0644,root,root) /usr/share/icons/hicolor/scalable/apps/{APP_SLUG}.svg",
                f"%dir /usr/share/doc/{APP_SLUG}",
                f"%doc /usr/share/doc/{APP_SLUG}/README.md",
                f"%license /usr/share/doc/{APP_SLUG}/LICENSE",
                f"%license /usr/share/doc/{APP_SLUG}/THIRD_PARTY.md",
            ]
        )
    else:
        summary = "Open-source protein NMR assignment and analytical spectroscopy CLI (32-bit static)"
        extra_header = "AutoReqProv:    no"
        description = (
            "Statically linked 32-bit x86 (i686 musl) command-line interface and MCP\n"
            "server for FERRIC automated protein NMR resonance assignment, raw FID\n"
            "processing, spin relaxation/exchange, and 1D analytical NMR/DOSY/metabolomics."
        )
        files_entries = "\n".join(
            [
                f"%attr(0755,root,root) /usr/bin/{CLI_BIN}",
                f"%dir /usr/share/doc/{APP_SLUG}",
                f"%doc /usr/share/doc/{APP_SLUG}/README.md",
                f"%license /usr/share/doc/{APP_SLUG}/LICENSE",
                f"%license /usr/share/doc/{APP_SLUG}/THIRD_PARTY.md",
            ]
        )

    return f"""%global debug_package %{{nil}}
%define _build_id_links none
%define _binary_payload w9.xzdio
%define __os_install_post %{{nil}}
%define use_source_date_epoch_as_buildtime 1
%define clamp_mtime_to_source_date_epoch 1

Name:           {APP_SLUG}
Version:        {rpm_ver}
Release:        1
Summary:        {summary}
License:        MIT
URL:            {PROJECT_URL}
{extra_header}

%description
{description}

%prep

%build

%install
rm -rf %{{buildroot}}
mkdir -p %{{buildroot}}
cp -a "{stage_payload.as_posix()}/." %{{buildroot}}/

%files
%defattr(-,root,root,-)
{files_entries}
"""


def _parse_rpm_header_list(spec_text: str, prefix: str) -> List[str]:
    values: List[str] = []
    for raw_line in spec_text.splitlines():
        if raw_line.startswith(prefix):
            rhs = raw_line[len(prefix) :].strip()
            values.extend(tok.strip() for tok in rhs.split(",") if tok.strip())
    return values


def validate_rpm_spec(spec_text: str, include_gui: bool) -> None:
    """Validate RPM ``.spec`` directives, payload compression, and file ownership rules."""
    for req_directive in (
        "%global debug_package %{nil}",
        "%define _build_id_links none",
        "%define _binary_payload w9.xzdio",
        f"Name:           {APP_SLUG}",
        "License:        MIT",
        f"%attr(0755,root,root) /usr/bin/{CLI_BIN}",
        f"%dir /usr/share/doc/{APP_SLUG}",
        f"%doc /usr/share/doc/{APP_SLUG}/README.md",
        f"%license /usr/share/doc/{APP_SLUG}/LICENSE",
        f"%license /usr/share/doc/{APP_SLUG}/THIRD_PARTY.md",
    ):
        if req_directive not in spec_text:
            raise ValueError(f"RPM spec missing required directive: {req_directive!r}")
    for forbidden_dir in ("%dir /usr\n", "%dir /usr/bin\n", "%dir /usr/share/applications\n"):
        if forbidden_dir in spec_text:
            raise ValueError(f"RPM spec must not claim system directory ownership: {forbidden_dir.strip()!r}")
    reqs = _parse_rpm_header_list(spec_text, "Requires:")
    recs = _parse_rpm_header_list(spec_text, "Recommends:")
    if include_gui:
        for req_pkg in GUI_RPM_REQUIRES:
            if req_pkg not in reqs:
                raise ValueError(f"Desktop RPM spec missing Requires {req_pkg!r} in {reqs!r}")
        for rec_pkg in GUI_RPM_RECOMMENDS:
            if rec_pkg not in recs:
                raise ValueError(f"Desktop RPM spec missing Recommends {rec_pkg!r} in {recs!r}")
        for gui_entry in (
            f"%attr(0755,root,root) /usr/bin/{GUI_BIN}",
            f"%attr(0644,root,root) /usr/share/applications/{APP_SLUG}.desktop",
            f"%attr(0644,root,root) /usr/share/icons/hicolor/256x256/apps/{APP_SLUG}.png",
        ):
            if gui_entry not in spec_text:
                raise ValueError(f"Desktop RPM spec missing GUI entry: {gui_entry!r}")
    else:
        if "AutoReqProv:    no" not in spec_text:
            raise ValueError("Static i686 CLI RPM spec must set AutoReqProv: no")
        if reqs or recs:
            raise ValueError(f"CLI-only RPM spec must not declare GUI Requires/Recommends: {reqs!r} / {recs!r}")
        if GUI_BIN in spec_text or f"{APP_SLUG}.desktop" in spec_text:
            raise ValueError("CLI-only RPM spec must not reference ferric-gui or ferric.desktop")


def _stage_linux_sysroot(
    sysroot_dir: Path,
    cli_src: Path,
    readme_src: Path,
    third_party_src: Path,
    license_src: Path,
    gui_src: Optional[Path] = None,
    desktop_text: Optional[str] = None,
    png_256: Optional[bytes] = None,
    svg_text: Optional[str] = None,
) -> None:
    """Populate FHS ``/usr`` tree for ``.deb`` and ``.rpm`` builders with deterministic modes/mtimes."""
    mtime = _source_mtime()
    bin_dir = sysroot_dir / "usr" / "bin"
    doc_dir = sysroot_dir / "usr" / "share" / "doc" / APP_SLUG
    bin_dir.mkdir(parents=True, exist_ok=True)
    doc_dir.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(cli_src, bin_dir / CLI_BIN)
    os.chmod(bin_dir / CLI_BIN, 0o755)

    if gui_src is not None:
        if desktop_text is None or png_256 is None or svg_text is None:
            raise ValueError("Desktop assets (desktop_text, png_256, svg_text) are required when gui_src is set")
        app_dir = sysroot_dir / "usr" / "share" / "applications"
        png_dir = sysroot_dir / "usr" / "share" / "icons" / "hicolor" / "256x256" / "apps"
        svg_dir = sysroot_dir / "usr" / "share" / "icons" / "hicolor" / "scalable" / "apps"
        for d in (app_dir, png_dir, svg_dir):
            d.mkdir(parents=True, exist_ok=True)

        shutil.copyfile(gui_src, bin_dir / GUI_BIN)
        os.chmod(bin_dir / GUI_BIN, 0o755)
        _write_text_lf(app_dir / f"{APP_SLUG}.desktop", desktop_text)
        (png_dir / f"{APP_SLUG}.png").write_bytes(png_256)
        _write_text_lf(svg_dir / f"{APP_SLUG}.svg", svg_text)
        os.chmod(app_dir / f"{APP_SLUG}.desktop", 0o644)
        os.chmod(png_dir / f"{APP_SLUG}.png", 0o644)
        os.chmod(svg_dir / f"{APP_SLUG}.svg", 0o644)

    shutil.copyfile(readme_src, doc_dir / "README.md")
    shutil.copyfile(license_src, doc_dir / "LICENSE")
    shutil.copyfile(third_party_src, doc_dir / "THIRD_PARTY.md")
    os.chmod(doc_dir / "README.md", 0o644)
    os.chmod(doc_dir / "LICENSE", 0o644)
    os.chmod(doc_dir / "THIRD_PARTY.md", 0o644)

    os.chmod(sysroot_dir, 0o755)
    for item in sorted(sysroot_dir.rglob("*"), key=lambda p: p.as_posix(), reverse=True):
        if item.is_dir():
            os.chmod(item, 0o755)
        os.utime(item, (mtime, mtime))
    os.utime(sysroot_dir, (mtime, mtime))


def _prepare_deb_tree(
    deb_stage: Path,
    sysroot_dir: Path,
    version: str,
    deb_arch: str,
    include_gui: bool,
) -> str:
    """Copy staged ``/usr`` tree into ``deb_stage`` and emit validated ``copyright`` and ``DEBIAN/{control,md5sums}``."""
    mtime = _source_mtime()
    if deb_stage.exists():
        shutil.rmtree(deb_stage)
    shutil.copytree(sysroot_dir, deb_stage)

    doc_dir = deb_stage / "usr" / "share" / "doc" / APP_SLUG
    license_path = doc_dir / "LICENSE"
    if not license_path.is_file():
        raise FileNotFoundError(f"Expected staged LICENSE at {license_path}")
    copyright_text = render_deb_copyright(license_path.read_text(encoding="utf-8"))
    validate_deb_copyright(copyright_text)
    copyright_path = doc_dir / "copyright"
    _write_text_lf(copyright_path, copyright_text)
    os.chmod(copyright_path, 0o644)
    os.utime(copyright_path, (mtime, mtime))
    os.utime(doc_dir, (mtime, mtime))

    payload_files = sorted(
        (p for p in deb_stage.rglob("*") if p.is_file()),
        key=lambda p: p.relative_to(deb_stage).as_posix(),
    )
    total_bytes = sum(p.stat().st_size for p in payload_files)
    installed_kib = max(1, (total_bytes + 1023) // 1024)

    debian_dir = deb_stage / "DEBIAN"
    debian_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(debian_dir, 0o755)

    control_text = render_deb_control(
        version=version,
        deb_arch=deb_arch,
        installed_kib=installed_kib,
        include_gui=include_gui,
    )
    validate_deb_control(control_text, expected_arch=deb_arch, expect_glibc_dep=include_gui)
    control_path = debian_dir / "control"
    _write_text_lf(control_path, control_text)
    os.chmod(control_path, 0o644)

    md5_lines = [f"{_md5_file(p)}  {p.relative_to(deb_stage).as_posix()}" for p in payload_files]
    md5_path = debian_dir / "md5sums"
    _write_text_lf(md5_path, "\n".join(md5_lines) + "\n")
    os.chmod(md5_path, 0o644)

    os.chmod(deb_stage, 0o755)
    for p in (control_path, md5_path, debian_dir, deb_stage):
        os.utime(p, (mtime, mtime))
    return control_text


def _build_deb_package(
    deb_stage: Path,
    out_deb: Path,
) -> Path:
    """Build XZ-compressed ``.deb`` archive via ``dpkg-deb --build --root-owner-group``."""
    if not shutil.which("dpkg-deb"):
        raise RuntimeError(
            f"dpkg-deb is required to build {out_deb.name} (install dpkg or pass --skip-deb)."
        )
    out_deb.parent.mkdir(parents=True, exist_ok=True)
    if out_deb.exists():
        out_deb.unlink()
    env = dict(os.environ)
    env["SOURCE_DATE_EPOCH"] = str(_source_mtime())
    subprocess.run(
        [
            "dpkg-deb",
            "-Zxz",
            "--build",
            "--root-owner-group",
            str(deb_stage),
            str(out_deb),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return out_deb


def _build_rpm_package(
    rpm_top: Path,
    sysroot_dir: Path,
    out_rpm: Path,
    version: str,
    rpm_arch: str,
    include_gui: bool,
) -> Path:
    """Build XZ-compressed ``.rpm`` package via ``rpmbuild -bb --target <arch>-linux``."""
    for sub in ("BUILD", "RPMS", "SOURCES", "SPECS", "SRPMS", "BUILDROOT"):
        (rpm_top / sub).mkdir(parents=True, exist_ok=True)
    spec_text = render_rpm_spec(version=version, stage_payload=sysroot_dir, include_gui=include_gui)
    validate_rpm_spec(spec_text, include_gui=include_gui)
    spec_path = rpm_top / "SPECS" / f"{APP_SLUG}.spec"
    _write_text_lf(spec_path, spec_text)

    if not shutil.which("rpmbuild"):
        raise RuntimeError(
            f"rpmbuild is required to build {out_rpm.name} (install rpm or pass --skip-rpm)."
        )
    out_rpm.parent.mkdir(parents=True, exist_ok=True)
    if out_rpm.exists():
        out_rpm.unlink()
    env = dict(os.environ)
    env["SOURCE_DATE_EPOCH"] = str(_source_mtime())
    subprocess.run(
        [
            "rpmbuild",
            "-bb",
            "--target",
            f"{rpm_arch}-linux",
            "--define",
            f"_topdir {rpm_top.as_posix()}",
            str(spec_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    built_rpms = sorted((rpm_top / "RPMS").rglob("*.rpm"))
    if len(built_rpms) != 1:
        raise RuntimeError(f"Expected exactly 1 built .rpm in {rpm_top / 'RPMS'}, found {built_rpms}")
    shutil.copyfile(built_rpms[0], out_rpm)
    return out_rpm


def package_linux_x86_64(
    repo_root: Path,
    bin_dir: Path,
    out_dir: Path,
    version: str,
    skip_deb: bool = False,
    skip_rpm: bool = False,
) -> List[Path]:
    """Build deterministic ``.tar.gz``, ``.deb`` (``amd64``), and ``.rpm`` (``x86_64``) for ``linux-x86_64``."""
    gui_src = _require_binary(bin_dir, GUI_BIN)
    cli_src = _require_binary(bin_dir, CLI_BIN)
    readme_src = repo_root / "README.md"
    license_src = repo_root / "LICENSE"
    third_party_src = repo_root / "THIRD_PARTY.md"

    png_256 = generate_icon_pngs((256,))[256]
    validate_png_bytes(png_256, 256)
    svg_text = render_icon_svg()
    validate_svg_text(svg_text)
    desktop_text = render_linux_desktop_entry()
    validate_desktop_entry(desktop_text)

    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{APP_SLUG}-v{version}-linux-x86_64"
    out_tar_gz = out_dir / f"{prefix}.tar.gz"
    outputs: List[Path] = [out_tar_gz]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        stage = tmp_path / prefix
        bin_stage = stage / "bin"
        app_stage = stage / "share" / "applications"
        svg_stage = stage / "share" / "icons" / "hicolor" / "scalable" / "apps"
        png_stage = stage / "share" / "icons" / "hicolor" / "256x256" / "apps"
        for d in (bin_stage, app_stage, svg_stage, png_stage):
            d.mkdir(parents=True, exist_ok=True)

        shutil.copyfile(gui_src, bin_stage / GUI_BIN)
        shutil.copyfile(cli_src, bin_stage / CLI_BIN)
        os.chmod(bin_stage / GUI_BIN, 0o755)
        os.chmod(bin_stage / CLI_BIN, 0o755)

        _write_text_lf(app_stage / f"{APP_SLUG}.desktop", desktop_text)
        _write_text_lf(svg_stage / f"{APP_SLUG}.svg", svg_text)
        (png_stage / f"{APP_SLUG}.png").write_bytes(png_256)

        installer_path = stage / "install-desktop.sh"
        _write_text_lf(installer_path, render_linux_install_desktop_sh())
        os.chmod(installer_path, 0o755)

        shutil.copyfile(readme_src, stage / "README.md")
        shutil.copyfile(license_src, stage / "LICENSE")
        shutil.copyfile(third_party_src, stage / "THIRD_PARTY.md")

        _write_deterministic_tar_gz(
            staging_root=stage,
            archive_prefix=prefix,
            out_tar_gz=out_tar_gz,
            executable_relpaths=(
                f"bin/{GUI_BIN}",
                f"bin/{CLI_BIN}",
                "install-desktop.sh",
            ),
        )

        # Always stage and validate .deb / .rpm metadata even when host tools are skipped
        sysroot_dir = tmp_path / "sysroot_x86_64"
        _stage_linux_sysroot(
            sysroot_dir=sysroot_dir,
            cli_src=cli_src,
            readme_src=readme_src,
            third_party_src=third_party_src,
            license_src=license_src,
            gui_src=gui_src,
            desktop_text=desktop_text,
            png_256=png_256,
            svg_text=svg_text,
        )
        deb_stage = tmp_path / "deb_x86_64"
        _prepare_deb_tree(
            deb_stage=deb_stage,
            sysroot_dir=sysroot_dir,
            version=version,
            deb_arch="amd64",
            include_gui=True,
        )
        validate_rpm_spec(
            render_rpm_spec(version=version, stage_payload=sysroot_dir, include_gui=True),
            include_gui=True,
        )

        if not skip_deb:
            out_deb = out_dir / f"{prefix}.deb"
            _build_deb_package(deb_stage=deb_stage, out_deb=out_deb)
            outputs.append(out_deb)

        if not skip_rpm:
            out_rpm = out_dir / f"{prefix}.rpm"
            _build_rpm_package(
                rpm_top=tmp_path / "rpm_x86_64",
                sysroot_dir=sysroot_dir,
                out_rpm=out_rpm,
                version=version,
                rpm_arch="x86_64",
                include_gui=True,
            )
            outputs.append(out_rpm)

    return outputs


def package_linux_i686_cli(
    repo_root: Path,
    bin_dir: Path,
    out_dir: Path,
    version: str,
    skip_deb: bool = False,
    skip_rpm: bool = False,
) -> List[Path]:
    """Build deterministic ``.tar.gz``, ``.deb`` (``i386``), and ``.rpm`` (``i686``) for ``linux-i686-cli``."""
    cli_src = _require_binary(bin_dir, CLI_BIN)
    readme_src = repo_root / "README.md"
    license_src = repo_root / "LICENSE"
    third_party_src = repo_root / "THIRD_PARTY.md"

    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{APP_SLUG}-v{version}-linux-i686-cli"
    out_tar_gz = out_dir / f"{prefix}.tar.gz"
    outputs: List[Path] = [out_tar_gz]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        stage = tmp_path / prefix
        bin_stage = stage / "bin"
        bin_stage.mkdir(parents=True, exist_ok=True)

        shutil.copyfile(cli_src, bin_stage / CLI_BIN)
        os.chmod(bin_stage / CLI_BIN, 0o755)
        shutil.copyfile(readme_src, stage / "README.md")
        shutil.copyfile(license_src, stage / "LICENSE")
        shutil.copyfile(third_party_src, stage / "THIRD_PARTY.md")

        _write_deterministic_tar_gz(
            staging_root=stage,
            archive_prefix=prefix,
            out_tar_gz=out_tar_gz,
            executable_relpaths=(f"bin/{CLI_BIN}",),
        )

        # Always stage and validate .deb / .rpm metadata even when host tools are skipped
        sysroot_dir = tmp_path / "sysroot_i686"
        _stage_linux_sysroot(
            sysroot_dir=sysroot_dir,
            cli_src=cli_src,
            readme_src=readme_src,
            third_party_src=third_party_src,
            license_src=license_src,
            gui_src=None,
        )
        deb_stage = tmp_path / "deb_i686"
        _prepare_deb_tree(
            deb_stage=deb_stage,
            sysroot_dir=sysroot_dir,
            version=version,
            deb_arch="i386",
            include_gui=False,
        )
        validate_rpm_spec(
            render_rpm_spec(version=version, stage_payload=sysroot_dir, include_gui=False),
            include_gui=False,
        )

        if not skip_deb:
            out_deb = out_dir / f"{prefix}.deb"
            _build_deb_package(deb_stage=deb_stage, out_deb=out_deb)
            outputs.append(out_deb)

        if not skip_rpm:
            out_rpm = out_dir / f"{prefix}.rpm"
            _build_rpm_package(
                rpm_top=tmp_path / "rpm_i686",
                sysroot_dir=sysroot_dir,
                out_rpm=out_rpm,
                version=version,
                rpm_arch="i686",
                include_gui=False,
            )
            outputs.append(out_rpm)

    return outputs


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _iter_release_archives(target_dir: Path) -> List[Path]:
    if not target_dir.is_dir():
        raise FileNotFoundError(f"Checksum directory does not exist: {target_dir}")
    archives: List[Path] = []
    for entry in sorted(target_dir.iterdir(), key=lambda p: p.name):
        if not entry.is_file():
            continue
        if entry.name == "SHA256SUMS.txt" or entry.name.startswith("."):
            continue
        if entry.name.endswith(RELEASE_ARCHIVE_SUFFIXES):
            archives.append(entry)
    return archives


def write_checksums(target_dir: Path) -> Path:
    """Write deterministic ``SHA256SUMS.txt`` over all release archives in ``target_dir``."""
    archives = _iter_release_archives(target_dir)
    if not archives:
        raise ValueError(
            f"No release archives ({', '.join(RELEASE_ARCHIVE_SUFFIXES)}) found in {target_dir}"
        )
    lines: List[str] = []
    for arc in archives:
        size = arc.stat().st_size
        if size <= 0:
            raise ValueError(f"Release archive is empty (0 bytes): {arc}")
        lines.append(f"{_sha256_file(arc)}  {arc.name}")
    out_file = target_dir / "SHA256SUMS.txt"
    _write_text_lf(out_file, "\n".join(lines) + "\n")
    return out_file


def verify_checksums(target_dir: Path) -> List[Tuple[str, str]]:
    """Verify ``SHA256SUMS.txt`` entries against release archives in ``target_dir``."""
    sums_path = target_dir / "SHA256SUMS.txt"
    if not sums_path.is_file():
        raise FileNotFoundError(f"SHA256SUMS.txt not found in {target_dir}")
    verified: List[Tuple[str, str]] = []
    raw_lines = [ln for ln in sums_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not raw_lines:
        raise ValueError("SHA256SUMS.txt is empty")
    for line in raw_lines:
        match = re.match(r"^([0-9a-f]{64})  ([^\s/\\]+)$", line)
        if not match:
            raise ValueError(f"Malformed SHA256SUMS.txt line: {line!r}")
        expected_hex, filename = match.group(1), match.group(2)
        file_path = target_dir / filename
        if not file_path.is_file() or file_path.stat().st_size <= 0:
            raise ValueError(f"Missing or empty archive referenced by SHA256SUMS.txt: {filename}")
        actual_hex = _sha256_file(file_path)
        if actual_hex != expected_hex:
            raise ValueError(f"SHA-256 mismatch for {filename}: expected {expected_hex}, got {actual_hex}")
        verified.append((filename, actual_hex))
    return verified


def _build_mock_binaries(mock_bin_dir: Path, macos_arch: Optional[str] = None) -> bool:
    """Create tiny valid mock executables for ``--self-test`` across all 5 platform targets."""
    mock_bin_dir.mkdir(parents=True, exist_ok=True)
    unix_stub: Optional[bytes] = None
    built_macho = False
    if sys.platform == "darwin" and shutil.which("cc"):
        c_src = mock_bin_dir / "_stub.c"
        out_macho = mock_bin_dir / "_stub_bin"
        _write_text_lf(c_src, "int main(void) { return 0; }\n")
        cc_cmd = ["cc", "-O2"]
        if macos_arch is not None:
            cc_cmd.extend(["-arch", macos_arch])
        cc_cmd.extend([str(c_src), "-o", str(out_macho)])
        res = subprocess.run(cc_cmd, capture_output=True, text=True)
        if res.returncode == 0 and out_macho.is_file():
            unix_stub = out_macho.read_bytes()
            built_macho = True
        c_src.unlink(missing_ok=True)
        out_macho.unlink(missing_ok=True)

    if unix_stub is None:
        unix_stub = b"#!/bin/sh\necho 'ferric mock 0.0.0'\n"

    for unix_name in (GUI_BIN, CLI_BIN, CLI_ALIAS_BIN):
        p = mock_bin_dir / unix_name
        p.write_bytes(unix_stub)
        os.chmod(p, 0o755)

    # Minimal PE/MZ stub header + deterministic payload for Windows .exe packaging tests
    pe_stub = b"MZ" + b"\x00" * 58 + struct.pack("<I", 64) + b"PE\x00\x00ferric-mock-win64"
    for win_name in (f"{GUI_BIN}.exe", f"{CLI_BIN}.exe", f"{CLI_ALIAS_BIN}.exe"):
        (mock_bin_dir / win_name).write_bytes(pe_stub)
    return built_macho


def _verify_macos_self_test_bundle(
    dist_dir: Path,
    version: str,
    platform_slug: str,
    expected_lipo_arch: Optional[str],
    is_macos_dmg_capable: bool,
) -> None:
    app_bundle = dist_dir / f"{APP_NAME}.app"
    validate_info_plist(app_bundle / "Contents" / "Info.plist", version)
    if (app_bundle / "Contents" / "PkgInfo").read_bytes() != b"APPL????":
        raise AssertionError("Invalid Contents/PkgInfo bytes")
    validate_icns_bytes((app_bundle / "Contents" / "Resources" / f"{APP_NAME}.icns").read_bytes())

    if expected_lipo_arch is not None and sys.platform == "darwin" and shutil.which("lipo"):
        for bin_name in (GUI_BIN, CLI_BIN):
            archs = subprocess.check_output(
                ["lipo", "-archs", str(app_bundle / "Contents" / "MacOS" / bin_name)],
                text=True,
            ).strip()
            if archs != expected_lipo_arch:
                raise AssertionError(
                    f"Expected lipo -archs {expected_lipo_arch!r} for {platform_slug}/{bin_name}, got {archs!r}"
                )

    mac_prefix = f"{APP_SLUG}-v{version}-{platform_slug}"
    mac_tar = dist_dir / f"{mac_prefix}.tar.gz"
    with tarfile.open(mac_tar, "r:gz") as tf:
        members = {m.name: m for m in tf.getmembers()}
        for req_exec in (
            f"{mac_prefix}/{APP_NAME}.app/Contents/MacOS/{GUI_BIN}",
            f"{mac_prefix}/{APP_NAME}.app/Contents/MacOS/{CLI_BIN}",
            f"{mac_prefix}/bin/{CLI_BIN}",
        ):
            if req_exec not in members:
                raise AssertionError(f"Missing {req_exec} in {platform_slug} .tar.gz")
            if (members[req_exec].mode & 0o777) != 0o755:
                raise AssertionError(f"Expected 0755 on {req_exec}, got {oct(members[req_exec].mode)}")
        for absent_alias in (
            f"{mac_prefix}/{APP_NAME}.app/Contents/MacOS/{CLI_ALIAS_BIN}",
            f"{mac_prefix}/bin/{CLI_ALIAS_BIN}",
        ):
            if absent_alias in members:
                raise AssertionError(f"Deprecated alias {absent_alias} must not ship")
        for req_doc in (
            f"{mac_prefix}/{APP_NAME}.app/Contents/Info.plist",
            f"{mac_prefix}/{APP_NAME}.app/Contents/PkgInfo",
            f"{mac_prefix}/{APP_NAME}.app/Contents/Resources/{APP_NAME}.icns",
            f"{mac_prefix}/README.md",
            f"{mac_prefix}/THIRD_PARTY.md",
        ):
            if req_doc not in members:
                raise AssertionError(f"Missing {req_doc} in {platform_slug} .tar.gz")

    if is_macos_dmg_capable:
        mac_dmg = dist_dir / f"{mac_prefix}.dmg"
        if not mac_dmg.is_file() or mac_dmg.stat().st_size <= 0:
            raise AssertionError(f"{platform_slug} .dmg was not created")
        subprocess.run(["hdiutil", "imageinfo", str(mac_dmg)], check=True, capture_output=True, text=True)


class PackageReleaseUnitTests(unittest.TestCase):
    """Unit tests for Python 3.9 compatibility and Linux RPM/DEB metadata generation."""

    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parent.parent
        self.version = read_workspace_version(self.repo_root)
        self.license_text = (self.repo_root / "LICENSE").read_text(encoding="utf-8")

    def test_python39_compatibility_ast(self) -> None:
        src = Path(__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        bad_nodes = [
            type(node).__name__
            for node in ast.walk(tree)
            if type(node).__name__ in ("Match", "MatchValue", "MatchAs", "TryStar")
        ]
        self.assertEqual(bad_nodes, [], f"Found Python 3.10+ AST syntax: {bad_nodes}")
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and node.func.attr == "write_text":
                    self.fail(f"Found forbidden Path.write_text() call at line {node.lineno}")
                for kw in node.keywords:
                    if kw.arg == "strict":
                        self.fail(f"Found Python 3.10+ zip(..., strict=) at line {node.lineno}")

    def test_write_text_lf_helper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "sample.txt"
            _write_text_lf(p, "alpha\nbeta\n")
            self.assertEqual(p.read_bytes(), b"alpha\nbeta\n")

    def test_rpm_spec_gui_requires_and_recommends(self) -> None:
        spec = render_rpm_spec(
            version=self.version,
            stage_payload=Path("/tmp/ferric-stage"),
            include_gui=True,
        )
        validate_rpm_spec(spec, include_gui=True)
        reqs = _parse_rpm_header_list(spec, "Requires:")
        recs = _parse_rpm_header_list(spec, "Recommends:")
        for pkg in GUI_RPM_REQUIRES:
            self.assertIn(pkg, reqs)
        self.assertIn("mesa-vulkan-drivers", recs)
        self.assertIn(f"%license /usr/share/doc/{APP_SLUG}/LICENSE", spec)
        self.assertIn(f"%license /usr/share/doc/{APP_SLUG}/THIRD_PARTY.md", spec)
        with self.assertRaises(ValueError):
            validate_rpm_spec(spec.replace("vulkan-loader, ", ""), include_gui=True)
        with self.assertRaises(ValueError):
            validate_rpm_spec(spec.replace("mesa-vulkan-drivers", "other-pkg"), include_gui=True)
        with self.assertRaises(ValueError):
            validate_rpm_spec(spec.replace(f"%license /usr/share/doc/{APP_SLUG}/LICENSE\n", ""), include_gui=True)

    def test_rpm_spec_cli_only(self) -> None:
        spec = render_rpm_spec(
            version=self.version,
            stage_payload=Path("/tmp/ferric-cli-stage"),
            include_gui=False,
        )
        validate_rpm_spec(spec, include_gui=False)
        self.assertIn("AutoReqProv:    no", spec)
        self.assertEqual(_parse_rpm_header_list(spec, "Requires:"), [])
        self.assertEqual(_parse_rpm_header_list(spec, "Recommends:"), [])
        self.assertIn(f"%license /usr/share/doc/{APP_SLUG}/LICENSE", spec)

    def test_deb_control_gui_depends_and_recommends(self) -> None:
        ctrl = render_deb_control(
            version=self.version,
            deb_arch="amd64",
            installed_kib=4096,
            include_gui=True,
        )
        validate_deb_control(ctrl, expected_arch="amd64", expect_glibc_dep=True)
        for pkg in GUI_DEB_DEPENDS:
            self.assertIn(pkg, ctrl)
        self.assertIn("Recommends: mesa-vulkan-drivers", ctrl)
        with self.assertRaises(ValueError):
            validate_deb_control(ctrl.replace(", libxkbcommon0", ""), expected_arch="amd64", expect_glibc_dep=True)
        with self.assertRaises(ValueError):
            validate_deb_control(ctrl.replace("mesa-vulkan-drivers", "other-pkg"), expected_arch="amd64", expect_glibc_dep=True)

    def test_deb_control_cli_only(self) -> None:
        ctrl = render_deb_control(
            version=self.version,
            deb_arch="i386",
            installed_kib=2048,
            include_gui=False,
        )
        validate_deb_control(ctrl, expected_arch="i386", expect_glibc_dep=False)
        self.assertNotIn("Depends:", ctrl)
        self.assertNotIn("Recommends:", ctrl)

    def test_deb_copyright_generation_and_staging(self) -> None:
        copyright_text = render_deb_copyright(self.license_text)
        validate_deb_copyright(copyright_text)
        with self.assertRaises(ValueError):
            render_deb_copyright("   \n")
        with self.assertRaises(ValueError):
            validate_deb_copyright("License: MIT\n")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cli_stub = tmp_path / CLI_BIN
            cli_stub.write_bytes(b"#!/bin/sh\necho ferric\n")
            os.chmod(cli_stub, 0o755)
            sysroot = tmp_path / "sysroot"
            deb_stage = tmp_path / "deb_stage"
            _stage_linux_sysroot(
                sysroot_dir=sysroot,
                cli_src=cli_stub,
                readme_src=self.repo_root / "README.md",
                third_party_src=self.repo_root / "THIRD_PARTY.md",
                license_src=self.repo_root / "LICENSE",
                gui_src=None,
            )
            _prepare_deb_tree(
                deb_stage=deb_stage,
                sysroot_dir=sysroot,
                version=self.version,
                deb_arch="i386",
                include_gui=False,
            )
            staged_license = deb_stage / "usr" / "share" / "doc" / APP_SLUG / "LICENSE"
            staged_copyright = deb_stage / "usr" / "share" / "doc" / APP_SLUG / "copyright"
            self.assertTrue(staged_license.is_file())
            self.assertTrue(staged_copyright.is_file())
            self.assertEqual(staged_license.read_bytes(), (self.repo_root / "LICENSE").read_bytes())
            validate_deb_copyright(staged_copyright.read_text(encoding="utf-8"))
            md5sums = (deb_stage / "DEBIAN" / "md5sums").read_text(encoding="utf-8")
            self.assertIn(f"usr/share/doc/{APP_SLUG}/LICENSE", md5sums)
            self.assertIn(f"usr/share/doc/{APP_SLUG}/copyright", md5sums)


def run_self_test(repo_root: Path) -> int:
    """Exercise icon encoders, all 5 platform packagers, .deb/.rpm metadata, and SHA256SUMS."""
    version = read_workspace_version(repo_root)
    is_macos_dmg_capable = sys.platform == "darwin" and bool(shutil.which("hdiutil"))
    has_dpkg_deb = bool(shutil.which("dpkg-deb"))
    has_rpmbuild = bool(shutil.which("rpmbuild"))
    expected_license_bytes = (repo_root / "LICENSE").read_bytes()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        icons_dir = tmp_path / "icons"
        emitted = emit_icon_suite(icons_dir)
        # Verify deterministic byte-identical icon generation & explicit 512px scanline determinism
        if render_icon_scanlines(512) != render_icon_scanlines(512):
            raise AssertionError("Non-deterministic 512px scanline raster")
        icons_dir_2 = tmp_path / "icons_repeat"
        emitted_2 = emit_icon_suite(icons_dir_2)
        for key, p1 in emitted.items():
            if p1.read_bytes() != emitted_2[key].read_bytes():
                raise AssertionError(f"Non-deterministic icon output for {key}")
        print(
            f"[self-test 1/8] Icon suite OK: PNGs (16,32,48,64,128,256,512), "
            f"ICNS ({emitted['icns'].stat().st_size} B), "
            f"ICO ({emitted['ico'].stat().st_size} B), "
            f"SVG ({emitted['svg'].stat().st_size} B)"
        )

        mock_bins_arm64 = tmp_path / "mock_bins_arm64"
        built_arm64_macho = _build_mock_binaries(mock_bins_arm64, macos_arch="arm64")
        mock_bins_x86_64 = tmp_path / "mock_bins_x86_64"
        built_x86_64_macho = _build_mock_binaries(mock_bins_x86_64, macos_arch="x86_64")
        dist_dir = tmp_path / "dist"

        # 2. macOS arm64 bundle + .dmg (on macOS) + .tar.gz
        mac_arm_outputs = package_macos_arm64(
            repo_root=repo_root,
            bin_dir=mock_bins_arm64,
            out_dir=dist_dir,
            version=version,
            skip_dmg=not is_macos_dmg_capable,
        )
        _verify_macos_self_test_bundle(
            dist_dir=dist_dir,
            version=version,
            platform_slug="macos-arm64",
            expected_lipo_arch="arm64" if built_arm64_macho else None,
            is_macos_dmg_capable=is_macos_dmg_capable,
        )
        print(
            f"[self-test 2/8] macos-arm64 OK: {[p.name for p in mac_arm_outputs]} "
            f"(Info.plist + PkgInfo + 0755 binaries"
            f"{' + arm64 lipo + codesign + UDZO .dmg' if is_macos_dmg_capable else ''})"
        )

        # 3. macOS Intel x86_64 bundle + .dmg (on macOS) + .tar.gz
        mac_x64_outputs = package_macos_x86_64(
            repo_root=repo_root,
            bin_dir=mock_bins_x86_64,
            out_dir=dist_dir,
            version=version,
            skip_dmg=not is_macos_dmg_capable,
        )
        _verify_macos_self_test_bundle(
            dist_dir=dist_dir,
            version=version,
            platform_slug="macos-x86_64",
            expected_lipo_arch="x86_64" if built_x86_64_macho else None,
            is_macos_dmg_capable=is_macos_dmg_capable,
        )
        print(
            f"[self-test 3/8] macos-x86_64 OK: {[p.name for p in mac_x64_outputs]} "
            f"(Info.plist + PkgInfo + 0755 binaries"
            f"{' + x86_64 lipo + codesign + UDZO .dmg' if is_macos_dmg_capable else ''})"
        )

        # 4. Windows x86_64 .zip
        win_outputs = package_windows_x86_64(repo_root, mock_bins_arm64, dist_dir, version)
        win_prefix = f"{APP_SLUG}-v{version}-windows-x86_64"
        win_zip = win_outputs[0]
        with zipfile.ZipFile(win_zip, "r") as zf:
            names = set(zf.namelist())
            for req_win in (
                f"{win_prefix}/{GUI_BIN}.exe",
                f"{win_prefix}/{CLI_BIN}.exe",
                f"{win_prefix}/{APP_SLUG}.ico",
                f"{win_prefix}/Launch-NMR-Workbench.bat",
                f"{win_prefix}/README.md",
                f"{win_prefix}/THIRD_PARTY.md",
            ):
                if req_win not in names:
                    raise AssertionError(f"Missing {req_win} in Windows .zip")
            if f"{win_prefix}/{CLI_ALIAS_BIN}.exe" in names:
                raise AssertionError("Deprecated nmrassign.exe alias must not ship")
            validate_ico_bytes(zf.read(f"{win_prefix}/{APP_SLUG}.ico"), ICO_SIZES)
        win_repeat_dir = tmp_path / "win_repeat"
        win_repeat = package_windows_x86_64(repo_root, mock_bins_arm64, win_repeat_dir, version)[0]
        if win_zip.read_bytes() != win_repeat.read_bytes():
            raise AssertionError("Windows .zip output is not byte-for-byte deterministic")
        print(f"[self-test 4/8] windows-x86_64 OK: {win_zip.name} ({win_zip.stat().st_size} B, deterministic)")

        # 5. Linux x86_64 .tar.gz + .deb/.rpm metadata (and live .deb/.rpm when tools exist) + XDG installer
        lin_outputs = package_linux_x86_64(
            repo_root,
            mock_bins_arm64,
            dist_dir,
            version,
            skip_deb=not has_dpkg_deb,
            skip_rpm=not has_rpmbuild,
        )
        lin_prefix = f"{APP_SLUG}-v{version}-linux-x86_64"
        lin_tar = dist_dir / f"{lin_prefix}.tar.gz"
        with tarfile.open(lin_tar, "r:gz") as tf:
            members = {m.name: m for m in tf.getmembers()}
            for req_exec in (
                f"{lin_prefix}/bin/{GUI_BIN}",
                f"{lin_prefix}/bin/{CLI_BIN}",
                f"{lin_prefix}/install-desktop.sh",
            ):
                if req_exec not in members:
                    raise AssertionError(f"Missing {req_exec} in Linux .tar.gz")
                if (members[req_exec].mode & 0o777) != 0o755:
                    raise AssertionError(f"Expected 0755 on {req_exec}, got {oct(members[req_exec].mode)}")
            if f"{lin_prefix}/bin/{CLI_ALIAS_BIN}" in members:
                raise AssertionError("Deprecated nmrassign alias must not ship in Linux archive")
            for req_file in (
                f"{lin_prefix}/share/applications/{APP_SLUG}.desktop",
                f"{lin_prefix}/share/icons/hicolor/scalable/apps/{APP_SLUG}.svg",
                f"{lin_prefix}/share/icons/hicolor/256x256/apps/{APP_SLUG}.png",
                f"{lin_prefix}/README.md",
                f"{lin_prefix}/LICENSE",
                f"{lin_prefix}/THIRD_PARTY.md",
            ):
                if req_file not in members:
                    raise AssertionError(f"Missing {req_file} in Linux .tar.gz")
            lic_member = tf.extractfile(members[f"{lin_prefix}/LICENSE"])
            assert lic_member is not None
            if lic_member.read() != expected_license_bytes:
                raise AssertionError("LICENSE content mismatch in Linux .tar.gz")
            desktop_member = tf.extractfile(members[f"{lin_prefix}/share/applications/{APP_SLUG}.desktop"])
            assert desktop_member is not None
            validate_desktop_entry(desktop_member.read().decode("utf-8"))

            if shutil.which("bash") and os.name != "nt":
                extract_dir = tmp_path / "lin_extract"
                if sys.version_info >= (3, 12):
                    tf.extractall(extract_dir, filter="data")
                else:
                    tf.extractall(extract_dir)
                xdg_prefix = tmp_path / "xdg_local"
                legacy_alias = xdg_prefix / "bin" / CLI_ALIAS_BIN
                legacy_bin = xdg_prefix / "bin" / "nmrassign-gui"
                legacy_desktop = xdg_prefix / "share" / "applications" / "nmr-assign.desktop"
                legacy_svg = xdg_prefix / "share" / "icons" / "hicolor" / "scalable" / "apps" / "nmr-assign.svg"
                legacy_png_256 = xdg_prefix / "share" / "icons" / "hicolor" / "256x256" / "apps" / "nmr-assign.png"
                legacy_png_128 = xdg_prefix / "share" / "icons" / "hicolor" / "128x128" / "apps" / "nmr-assign.png"
                for lp in (legacy_alias, legacy_bin, legacy_desktop, legacy_svg, legacy_png_256, legacy_png_128):
                    lp.parent.mkdir(parents=True, exist_ok=True)
                    _write_text_lf(lp, "legacy-v0.13-v0.14\n")

                subprocess.run(
                    ["bash", str(extract_dir / lin_prefix / "install-desktop.sh"), "--prefix", str(xdg_prefix)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                for lp in (legacy_alias, legacy_bin, legacy_desktop, legacy_svg, legacy_png_256, legacy_png_128):
                    if lp.exists():
                        raise AssertionError(f"Legacy file not removed on upgrade: {lp}")
                installed_desktop = xdg_prefix / "share" / "applications" / f"{APP_SLUG}.desktop"
                if (
                    not (xdg_prefix / "bin" / GUI_BIN).is_file()
                    or not (xdg_prefix / "bin" / CLI_BIN).is_file()
                    or not installed_desktop.is_file()
                ):
                    raise AssertionError("install-desktop.sh failed to populate XDG prefix")

        if has_dpkg_deb:
            lin_deb = dist_dir / f"{lin_prefix}.deb"
            info_out = subprocess.check_output(["dpkg-deb", "--info", str(lin_deb)], text=True)
            contents_out = subprocess.check_output(["dpkg-deb", "--contents", str(lin_deb)], text=True)
            if "Architecture: amd64" not in info_out:
                raise AssertionError(f"Unexpected linux-x86_64 .deb metadata: {info_out}")
            for req_dep in GUI_DEB_DEPENDS + GUI_DEB_RECOMMENDS:
                if req_dep not in info_out:
                    raise AssertionError(f"Missing {req_dep!r} in linux-x86_64 .deb metadata: {info_out}")
            for req_deb_path in (
                "./usr/bin/ferric",
                "./usr/bin/ferric-gui",
                "./usr/share/applications/ferric.desktop",
                "./usr/share/icons/hicolor/256x256/apps/ferric.png",
                f"./usr/share/doc/{APP_SLUG}/LICENSE",
                f"./usr/share/doc/{APP_SLUG}/copyright",
            ):
                if req_deb_path not in contents_out:
                    raise AssertionError(f"Missing {req_deb_path} in linux-x86_64 .deb")

        if has_rpmbuild and shutil.which("rpm"):
            lin_rpm = dist_dir / f"{lin_prefix}.rpm"
            rpm_out = subprocess.check_output(["rpm", "-qp", "--info", "--list", str(lin_rpm)], text=True)
            rpm_reqs = subprocess.check_output(["rpm", "-qp", "--requires", str(lin_rpm)], text=True)
            rpm_recs = subprocess.check_output(["rpm", "-qp", "--recommends", str(lin_rpm)], text=True)
            for req_pkg in GUI_RPM_REQUIRES:
                if req_pkg not in rpm_reqs:
                    raise AssertionError(f"Missing {req_pkg!r} in linux-x86_64 .rpm Requires: {rpm_reqs}")
            for rec_pkg in GUI_RPM_RECOMMENDS:
                if rec_pkg not in rpm_recs:
                    raise AssertionError(f"Missing {rec_pkg!r} in linux-x86_64 .rpm Recommends: {rpm_recs}")
            for req_rpm_path in (
                "/usr/bin/ferric",
                "/usr/bin/ferric-gui",
                "/usr/share/applications/ferric.desktop",
                "/usr/share/icons/hicolor/256x256/apps/ferric.png",
                f"/usr/share/doc/{APP_SLUG}/LICENSE",
            ):
                if req_rpm_path not in rpm_out:
                    raise AssertionError(f"Missing {req_rpm_path} in linux-x86_64 .rpm")

        lin_repeat_dir = tmp_path / "lin_repeat"
        lin_repeat = package_linux_x86_64(
            repo_root, mock_bins_arm64, lin_repeat_dir, version, skip_deb=True, skip_rpm=True
        )[0]
        if lin_tar.read_bytes() != lin_repeat.read_bytes():
            raise AssertionError("Linux x86_64 .tar.gz output is not byte-for-byte deterministic")
        print(
            f"[self-test 5/8] linux-x86_64 OK: {[p.name for p in lin_outputs]} "
            f"(deterministic .tar.gz + LICENSE/copyright + validated amd64/x86_64 .deb/.rpm specs)"
        )

        # 6. Linux i686 static CLI-only .tar.gz + .deb/.rpm metadata (without ferric-gui!)
        cli_only_bins = tmp_path / "mock_bins_i686_cli"
        cli_only_bins.mkdir(parents=True, exist_ok=True)
        shutil.copy2(mock_bins_arm64 / CLI_BIN, cli_only_bins / CLI_BIN)
        i686_outputs = package_linux_i686_cli(
            repo_root,
            cli_only_bins,
            dist_dir,
            version,
            skip_deb=not has_dpkg_deb,
            skip_rpm=not has_rpmbuild,
        )
        i686_prefix = f"{APP_SLUG}-v{version}-linux-i686-cli"
        i686_tar = dist_dir / f"{i686_prefix}.tar.gz"
        with tarfile.open(i686_tar, "r:gz") as tf:
            members = {m.name: m for m in tf.getmembers()}
            if f"{i686_prefix}/bin/{CLI_BIN}" not in members:
                raise AssertionError("Missing bin/ferric in linux-i686-cli .tar.gz")
            if (members[f"{i686_prefix}/bin/{CLI_BIN}"].mode & 0o777) != 0o755:
                raise AssertionError("Expected 0755 on bin/ferric in linux-i686-cli .tar.gz")
            for forbidden in (
                f"{i686_prefix}/bin/{GUI_BIN}",
                f"{i686_prefix}/share/applications/{APP_SLUG}.desktop",
            ):
                if forbidden in members:
                    raise AssertionError(f"CLI-only i686 archive must not include {forbidden}")
            for req_doc in (f"{i686_prefix}/README.md", f"{i686_prefix}/LICENSE", f"{i686_prefix}/THIRD_PARTY.md"):
                if req_doc not in members:
                    raise AssertionError(f"Missing {req_doc} in linux-i686-cli .tar.gz")

        if has_dpkg_deb:
            i686_deb = dist_dir / f"{i686_prefix}.deb"
            info_out = subprocess.check_output(["dpkg-deb", "--info", str(i686_deb)], text=True)
            contents_out = subprocess.check_output(["dpkg-deb", "--contents", str(i686_deb)], text=True)
            if "Architecture: i386" not in info_out:
                raise AssertionError(f"Expected Architecture: i386 in i686 .deb: {info_out}")
            for req_deb_doc in (f"./usr/share/doc/{APP_SLUG}/LICENSE", f"./usr/share/doc/{APP_SLUG}/copyright"):
                if req_deb_doc not in contents_out:
                    raise AssertionError(f"Missing {req_deb_doc} in i686 CLI .deb: {contents_out}")
            if "./usr/bin/ferric" not in contents_out or "ferric-gui" in contents_out:
                raise AssertionError(f"Unexpected contents in i686 CLI .deb: {contents_out}")

        if has_rpmbuild and shutil.which("rpm"):
            i686_rpm = dist_dir / f"{i686_prefix}.rpm"
            rpm_out = subprocess.check_output(["rpm", "-qp", "--info", "--list", str(i686_rpm)], text=True)
            if "/usr/bin/ferric" not in rpm_out or f"/usr/share/doc/{APP_SLUG}/LICENSE" not in rpm_out or "ferric-gui" in rpm_out:
                raise AssertionError(f"Unexpected contents in i686 CLI .rpm: {rpm_out}")

        i686_repeat_dir = tmp_path / "i686_repeat"
        i686_repeat = package_linux_i686_cli(
            repo_root, cli_only_bins, i686_repeat_dir, version, skip_deb=True, skip_rpm=True
        )[0]
        if i686_tar.read_bytes() != i686_repeat.read_bytes():
            raise AssertionError("Linux i686 CLI .tar.gz output is not byte-for-byte deterministic")

        # Exercise _build_deb_package / _build_rpm_package subprocess plumbing on macOS hosts too
        if os.name != "nt" and shutil.which("bash") and (not has_dpkg_deb or not has_rpmbuild):
            shim_dir = tmp_path / "pkg_shims"
            shim_dir.mkdir(parents=True, exist_ok=True)
            dpkg_shim = shim_dir / "dpkg-deb"
            _write_text_lf(
                dpkg_shim,
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                'test "$1" = "-Zxz"\n'
                'test "$2" = "--build"\n'
                'test "$3" = "--root-owner-group"\n'
                'test -f "$4/DEBIAN/control"\n'
                'test -f "$4/DEBIAN/md5sums"\n'
                f'test -f "$4/usr/share/doc/{APP_SLUG}/LICENSE"\n'
                f'test -f "$4/usr/share/doc/{APP_SLUG}/copyright"\n'
                f'grep -q "usr/share/doc/{APP_SLUG}/copyright" "$4/DEBIAN/md5sums"\n'
                'printf "!<arch>\\ndebian-binary   mock\\n" > "$5"\n',
            )
            os.chmod(dpkg_shim, 0o755)
            rpm_shim = shim_dir / "rpmbuild"
            _write_text_lf(
                rpm_shim,
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                'test "$1" = "-bb"\n'
                'test "$2" = "--target"\n'
                'ARCH="${3%-linux}"\n'
                'test "$4" = "--define"\n'
                'TOPDIR="${5#_topdir }"\n'
                'test -f "$6"\n'
                f'grep -q "/usr/share/doc/{APP_SLUG}/LICENSE" "$6"\n'
                'mkdir -p "${TOPDIR}/RPMS/${ARCH}"\n'
                'printf "\\xed\\xab\\xee\\xdbmock-rpm" > "${TOPDIR}/RPMS/${ARCH}/ferric-mock.${ARCH}.rpm"\n',
            )
            os.chmod(rpm_shim, 0o755)
            old_path = os.environ.get("PATH", "")
            try:
                os.environ["PATH"] = f"{shim_dir}{os.pathsep}{old_path}"
                shim_out = tmp_path / "shim_out"
                x64_all = package_linux_x86_64(
                    repo_root, mock_bins_arm64, shim_out, version, skip_deb=False, skip_rpm=False
                )
                i686_all = package_linux_i686_cli(
                    repo_root, cli_only_bins, shim_out, version, skip_deb=False, skip_rpm=False
                )
                if [p.name for p in x64_all] != [
                    f"{lin_prefix}.tar.gz",
                    f"{lin_prefix}.deb",
                    f"{lin_prefix}.rpm",
                ]:
                    raise AssertionError(f"Unexpected x86_64 artifact list: {[p.name for p in x64_all]}")
                if [p.name for p in i686_all] != [
                    f"{i686_prefix}.tar.gz",
                    f"{i686_prefix}.deb",
                    f"{i686_prefix}.rpm",
                ]:
                    raise AssertionError(f"Unexpected i686 artifact list: {[p.name for p in i686_all]}")
            finally:
                os.environ["PATH"] = old_path

        print(
            f"[self-test 6/8] linux-i686-cli OK: {[p.name for p in i686_outputs]} "
            f"(CLI-only deterministic .tar.gz + LICENSE/copyright + validated i386/i686 .deb/.rpm specs)"
        )

        # 7. Checksum generation & tamper detection (including .deb and .rpm suffixes)
        sums_file = write_checksums(dist_dir)
        verified = verify_checksums(dist_dir)
        expected_count = (
            (4 if is_macos_dmg_capable else 2)  # macos-arm64 + macos-x86_64
            + 1  # windows-x86_64 .zip
            + 2  # linux-x86_64 + linux-i686-cli .tar.gz
            + (2 if has_dpkg_deb else 0)
            + (2 if has_rpmbuild else 0)
        )
        if len(verified) != expected_count:
            raise AssertionError(f"Expected {expected_count} checksummed archives, got {len(verified)}")

        # Explicitly verify .deb and .rpm discovery in write_checksums even on macOS/Windows hosts
        suffix_probe_dir = tmp_path / "suffix_probe"
        suffix_probe_dir.mkdir(parents=True, exist_ok=True)
        for ext in RELEASE_ARCHIVE_SUFFIXES:
            (suffix_probe_dir / f"probe{ext}").write_bytes(f"payload-{ext}".encode("utf-8"))
        write_checksums(suffix_probe_dir)
        if len(verify_checksums(suffix_probe_dir)) != len(RELEASE_ARCHIVE_SUFFIXES):
            raise AssertionError("SHA256SUMS failed to index all 5 release archive extensions")

        # Flip one byte in Windows zip and ensure verify_checksums rejects it
        orig_bytes = win_zip.read_bytes()
        win_zip.write_bytes(orig_bytes[:-1] + bytes([orig_bytes[-1] ^ 0xFF]))
        tamper_caught = False
        try:
            verify_checksums(dist_dir)
        except ValueError:
            tamper_caught = True
        finally:
            win_zip.write_bytes(orig_bytes)
        if not tamper_caught:
            raise AssertionError("verify_checksums failed to detect archive byte corruption")
        verify_checksums(dist_dir)
        print(
            f"[self-test 7/8] SHA256SUMS.txt OK: {len(verified)} archives verified "
            f"({sums_file.stat().st_size} B, all 5 suffixes indexed) + tamper rejection confirmed"
        )

        # 8. Python 3.9 compatibility & Linux RPM/DEB metadata unit tests
        from io import StringIO

        unit_stream = StringIO()
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(PackageReleaseUnitTests)
        unit_result = unittest.TextTestRunner(stream=unit_stream, verbosity=1).run(suite)
        if not unit_result.wasSuccessful():
            raise AssertionError(f"Unit tests failed:\n{unit_stream.getvalue()}")
        print(
            f"[self-test 8/8] Python 3.9 & RPM/DEB metadata unit tests OK: "
            f"{unit_result.testsRun} tests passed"
        )

    print(f"ALL SELF-TESTS PASSED ({APP_NAME} v{version})")
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package cross-platform FERRIC desktop/CLI release archives and icons."
    )
    parser.add_argument(
        "--platform",
        choices=(*SUPPORTED_PLATFORMS, "all"),
        help="Target platform bundle to build.",
    )
    parser.add_argument(
        "--bin-dir",
        type=Path,
        default=None,
        help="Directory containing compiled release binaries (default: <repo>/target/release).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for generated release bundles (default: <repo>/dist).",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Explicit release version (defaults to [workspace.package] version in Cargo.toml).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Path to the FERRIC repository root.",
    )
    parser.add_argument(
        "--skip-dmg",
        action="store_true",
        help="Skip macOS hdiutil .dmg image creation when packaging macos-arm64 or macos-x86_64.",
    )
    parser.add_argument(
        "--skip-deb",
        action="store_true",
        help="Skip Debian/Ubuntu .deb creation when packaging linux-x86_64 or linux-i686-cli.",
    )
    parser.add_argument(
        "--skip-rpm",
        action="store_true",
        help="Skip RHEL/Fedora .rpm creation when packaging linux-x86_64 or linux-i686-cli.",
    )
    parser.add_argument(
        "--emit-icons",
        type=Path,
        default=None,
        help="Write standalone multi-size PNG, ICNS, ICO, and SVG icons into the given directory.",
    )
    parser.add_argument(
        "--checksums",
        type=Path,
        default=None,
        help="Generate deterministic SHA256SUMS.txt over all release archives in the given directory.",
    )
    parser.add_argument(
        "--verify-checksums",
        type=Path,
        default=None,
        help="Verify SHA256SUMS.txt in the given directory against all listed release archives.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run end-to-end packaging, icon header, permission, and SHA256SUMS self-tests.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()

    if args.self_test:
        return run_self_test(repo_root)

    did_work = False

    if args.emit_icons is not None:
        written = emit_icon_suite(args.emit_icons.resolve())
        for name, path in sorted(written.items()):
            print(f"icon[{name}]: {path} ({path.stat().st_size} B)")
        did_work = True

    if args.platform is not None:
        version = normalize_version(args.version) if args.version else read_workspace_version(repo_root)
        bin_dir = (args.bin_dir if args.bin_dir is not None else repo_root / "target" / "release").resolve()
        out_dir = (args.out_dir if args.out_dir is not None else repo_root / "dist").resolve()
        targets = SUPPORTED_PLATFORMS if args.platform == "all" else (args.platform,)
        for target in targets:
            if target == "macos-arm64":
                built = package_macos_arm64(repo_root, bin_dir, out_dir, version, skip_dmg=args.skip_dmg)
            elif target == "macos-x86_64":
                built = package_macos_x86_64(repo_root, bin_dir, out_dir, version, skip_dmg=args.skip_dmg)
            elif target == "windows-x86_64":
                built = package_windows_x86_64(repo_root, bin_dir, out_dir, version)
            elif target == "linux-x86_64":
                built = package_linux_x86_64(
                    repo_root,
                    bin_dir,
                    out_dir,
                    version,
                    skip_deb=args.skip_deb,
                    skip_rpm=args.skip_rpm,
                )
            elif target == "linux-i686-cli":
                built = package_linux_i686_cli(
                    repo_root,
                    bin_dir,
                    out_dir,
                    version,
                    skip_deb=args.skip_deb,
                    skip_rpm=args.skip_rpm,
                )
            else:
                raise ValueError(f"Unhandled platform: {target}")
            for artifact in built:
                print(f"packaged[{target}]: {artifact} ({artifact.stat().st_size} B, sha256={_sha256_file(artifact)})")
        did_work = True

    if args.checksums is not None:
        sums_file = write_checksums(args.checksums.resolve())
        print(f"checksums: {sums_file}")
        print(sums_file.read_text(encoding="utf-8"), end="")
        did_work = True

    if args.verify_checksums is not None:
        verified = verify_checksums(args.verify_checksums.resolve())
        for filename, digest in verified:
            print(f"verified: {digest}  {filename}")
        did_work = True

    if not did_work:
        raise SystemExit("No action requested. Pass --platform, --checksums, --emit-icons, or --self-test.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
