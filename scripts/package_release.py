#!/usr/bin/env python3
"""Zero-dependency cross-platform desktop release packager and icon generator.

Builds reproducible release bundles and procedural vector/raster desktop icons
for ``FERRIC`` across:
- ``macos-arm64``: ``FERRIC.app`` bundle (``ferric-gui`` + ``ferric`` CLI/MCP
  + deprecated ``nmrassign`` alias for ``v0.14.x``, ``Info.plist``, ``PkgInfo``,
  ``FERRIC.icns``, ad-hoc Apple Silicon ``codesign``), compressed drag-and-drop
  ``.dmg`` (with ``/Applications`` symlink and quickstart guide), and ``.tar.gz``
  archive.
- ``windows-x86_64``: deterministic ``.zip`` archive containing
  ``ferric-gui.exe``, ``ferric.exe``, ``nmrassign.exe`` (``v0.14.x`` alias),
  multi-resolution ``ferric.ico``, ``Launch-NMR-Workbench.bat``, ``README.md``,
  and ``THIRD_PARTY.md``.
- ``linux-x86_64``: deterministic ``.tar.gz`` archive containing ``bin/``
  executables (``ferric-gui``, ``ferric``, ``nmrassign``, ``0755``), FreeDesktop
  ``ferric.desktop``, scalable SVG + ``256x256`` PNG hicolor icons, user-level
  ``install-desktop.sh`` XDG installer, ``README.md``, and ``THIRD_PARTY.md``.
- ``--checksums <dir>``: deterministic ``SHA256SUMS.txt`` generator and verifier.

Provenance: Original clean-room Python 3 standard-library release packaging and
RFC 2083 PNG / Apple ICNS / Windows ICO / SVG icon script (MIT License).
"""

from __future__ import annotations

import argparse
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
import xml.etree.ElementTree as ET
import zipfile
import zlib

APP_NAME = "FERRIC"
APP_SLUG = "ferric"
CLI_BIN = "ferric"
CLI_ALIAS_BIN = "nmrassign"
GUI_BIN = "ferric-gui"
BUNDLE_ID = "org.ferric-nmr.desktop"
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
    "windows-x86_64",
    "linux-x86_64",
)


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


def render_icon_scanlines(size: int) -> bytes:
    """Deterministically rasterize the ``FERRIC`` icon at ``size x size`` RGBA."""
    if size <= 0 or size > 2048:
        raise ValueError(f"Unsupported icon dimension: {size}")

    row_stride = 1 + size * 4
    buf = bytearray(size * row_stride)

    margin = 0.0625 * size
    radius = 0.225 * size
    inner_lo = margin + radius
    inner_hi = size - margin - radius
    border_w = max(0.85, size * 0.018)

    # Precompute 1D corner offsets for fast rounded-rectangle distance field
    x_corner_dx = [0.0] * size
    x_norm = [0.0] * size
    for px in range(size):
        xc = px + 0.5
        x_norm[px] = xc / size
        if xc < inner_lo:
            x_corner_dx[px] = inner_lo - xc
        elif xc > inner_hi:
            x_corner_dx[px] = xc - inner_hi

    for py in range(size):
        yc = py + 0.5
        yn = yc / size
        if yc < inner_lo:
            dy_c = inner_lo - yc
        elif yc > inner_hi:
            dy_c = yc - inner_hi
        else:
            dy_c = 0.0

        row_off = py * row_stride
        buf[row_off] = 0  # PNG filter type 0 (None)
        p_off = row_off + 1

        for px in range(size):
            dx_c = x_corner_dx[px]
            if dx_c > 0.0 and dy_c > 0.0:
                sdf = math.hypot(dx_c, dy_c) - radius
            else:
                xc = px + 0.5
                sdf = max(margin - xc, xc - (size - margin), margin - yc, yc - (size - margin))

            if sdf >= 0.65:
                p_off += 4
                continue

            alpha = 1.0 if sdf <= -0.55 else (0.65 - sdf) / 1.20
            xn = x_norm[px]
            diag = 0.55 * xn + 0.45 * yn
            # Deep midnight-slate to marine-indigo scientific background
            r = 10.0 + 14.0 * diag
            g = 17.0 + 24.0 * diag
            b = 34.0 + 48.0 * diag

            # Subtle inner rim highlight
            rim_dist = -sdf
            if 0.0 <= rim_dist <= border_w + 0.8:
                rim_cov = 1.0 - min(1.0, abs(rim_dist - 0.5 * border_w) / (0.5 * border_w + 0.8))
                rim_strength = 0.42 * rim_cov
                r = r * (1.0 - rim_strength) + 56.0 * rim_strength
                g = g * (1.0 - rim_strength) + 189.0 * rim_strength
                b = b * (1.0 - rim_strength) + 248.0 * rim_strength

            buf[p_off] = _clamp_u8(r)
            buf[p_off + 1] = _clamp_u8(g)
            buf[p_off + 2] = _clamp_u8(b)
            buf[p_off + 3] = _clamp_u8(alpha * 255.0)
            p_off += 4

    # Spectral chemical-shift reference grid (F2 1H / F1 15N-13C) for >= 32 px
    if size >= 32:
        grid_hw = max(0.004, 0.55 / size)
        for pos in (0.34, 0.505, 0.67):
            _draw_segment_aa(buf, size, 0.17, pos, 0.83, pos, grid_hw, (51, 92, 138), 0.30)
            _draw_segment_aa(buf, size, pos, 0.17, pos, 0.83, grid_hw, (51, 92, 138), 0.30)

    # Sequential backbone spin-walk connectivity pathway: (0.34, 0.66) -> (0.34, 0.35) -> (0.67, 0.35) -> (0.67, 0.66)
    walk_hw = max(0.016, 0.85 / size)
    _draw_segment_aa(buf, size, 0.34, 0.66, 0.34, 0.35, walk_hw, (52, 211, 153), 0.88)
    _draw_segment_aa(buf, size, 0.34, 0.35, 0.67, 0.35, walk_hw, (52, 211, 153), 0.88)
    _draw_segment_aa(buf, size, 0.67, 0.35, 0.67, 0.66, walk_hw * 0.78, (45, 212, 191), 0.58)

    pos_rings: Tuple[Tuple[float, Tuple[int, int, int], float], ...] = (
        (1.00, (14, 165, 233), 0.82),
        (0.66, (56, 189, 248), 0.92),
        (0.36, (186, 230, 253), 0.96),
    )
    neg_rings: Tuple[Tuple[float, Tuple[int, int, int], float], ...] = (
        (1.00, (234, 88, 12), 0.86),
        (0.60, (251, 146, 60), 0.94),
    )
    sat_rings: Tuple[Tuple[float, Tuple[int, int, int], float], ...] = (
        (1.00, (6, 182, 212), 0.80),
        (0.58, (103, 232, 249), 0.92),
    )

    # Four 2D NMR spin-system cross-peaks (positive cyan + phase-inverted coral)
    _draw_contour_family(buf, size, 0.34, 0.66, 0.145, 0.105, -22.0, pos_rings, (240, 249, 255))
    _draw_contour_family(buf, size, 0.67, 0.35, 0.145, 0.105, -22.0, pos_rings, (240, 249, 255))
    _draw_contour_family(buf, size, 0.34, 0.35, 0.120, 0.090, 20.0, neg_rings, (255, 237, 213))
    _draw_contour_family(buf, size, 0.67, 0.66, 0.112, 0.086, -18.0, sat_rings, (236, 254, 255))

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
    """Render the matching scalable vector ``ferric.svg`` FreeDesktop launcher icon."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512">
  <defs>
    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0a1122"/>
      <stop offset="55%" stop-color="#111d38"/>
      <stop offset="100%" stop-color="#182952"/>
    </linearGradient>
    <radialGradient id="cyanGlow" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#38bdf8" stop-opacity="0.34"/>
      <stop offset="100%" stop-color="#38bdf8" stop-opacity="0"/>
    </radialGradient>
    <radialGradient id="coralGlow" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#fb923c" stop-opacity="0.30"/>
      <stop offset="100%" stop-color="#fb923c" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <rect x="32" y="32" width="448" height="448" rx="115" ry="115"
        fill="url(#bgGrad)" stroke="#38bdf8" stroke-opacity="0.42" stroke-width="8"/>
  <g stroke="#335c8a" stroke-opacity="0.34" stroke-width="3">
    <line x1="87" y1="174" x2="425" y2="174"/>
    <line x1="87" y1="258" x2="425" y2="258"/>
    <line x1="87" y1="343" x2="425" y2="343"/>
    <line x1="174" y1="87" x2="174" y2="425"/>
    <line x1="258" y1="87" x2="258" y2="425"/>
    <line x1="343" y1="87" x2="343" y2="425"/>
  </g>
  <polyline points="174,338 174,179 343,179"
            fill="none" stroke="#34d399" stroke-width="15"
            stroke-linecap="round" stroke-linejoin="round" stroke-opacity="0.90"/>
  <line x1="343" y1="179" x2="343" y2="338"
        stroke="#2dd4bf" stroke-width="11" stroke-dasharray="16 12"
        stroke-linecap="round" stroke-opacity="0.65"/>
  <g transform="translate(174 338) rotate(-22)">
    <circle cx="0" cy="0" r="82" fill="url(#cyanGlow)"/>
    <ellipse cx="0" cy="0" rx="74" ry="54" fill="none" stroke="#0ea5e9" stroke-width="7"/>
    <ellipse cx="0" cy="0" rx="49" ry="36" fill="none" stroke="#38bdf8" stroke-width="7"/>
    <ellipse cx="0" cy="0" rx="27" ry="19" fill="none" stroke="#bae6fd" stroke-width="7"/>
    <circle cx="0" cy="0" r="11" fill="#f0f9ff"/>
  </g>
  <g transform="translate(343 179) rotate(-22)">
    <circle cx="0" cy="0" r="82" fill="url(#cyanGlow)"/>
    <ellipse cx="0" cy="0" rx="74" ry="54" fill="none" stroke="#0ea5e9" stroke-width="7"/>
    <ellipse cx="0" cy="0" rx="49" ry="36" fill="none" stroke="#38bdf8" stroke-width="7"/>
    <ellipse cx="0" cy="0" rx="27" ry="19" fill="none" stroke="#bae6fd" stroke-width="7"/>
    <circle cx="0" cy="0" r="11" fill="#f0f9ff"/>
  </g>
  <g transform="translate(174 179) rotate(20)">
    <circle cx="0" cy="0" r="68" fill="url(#coralGlow)"/>
    <ellipse cx="0" cy="0" rx="61" ry="46" fill="none" stroke="#ea580c" stroke-width="7"/>
    <ellipse cx="0" cy="0" rx="37" ry="28" fill="none" stroke="#fb923c" stroke-width="7"/>
    <circle cx="0" cy="0" r="10" fill="#ffedd5"/>
  </g>
  <g transform="translate(343 338) rotate(-18)">
    <ellipse cx="0" cy="0" rx="57" ry="44" fill="none" stroke="#06b6d4" stroke-width="6.5"/>
    <ellipse cx="0" cy="0" rx="33" ry="25" fill="none" stroke="#67e8f9" stroke-width="6.5"/>
    <circle cx="0" cy="0" r="9.5" fill="#ecfeff"/>
  </g>
</svg>
"""


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
    svg_path.write_text(svg_text, encoding="utf-8", newline="\n")
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


def render_cli_and_mcp_quickstart(version: str) -> str:
    """Render plain-text macOS quickstart bundled inside the ``.dmg`` volume."""
    return f"""FERRIC v{version} — Apple Silicon Desktop, CLI & MCP Quickstart
================================================================

1. Install the Desktop Workbench (`FERRIC.app`)
-----------------------------------------------
Drag `FERRIC.app` onto the `Applications` folder shortcut in this disk image.
Double-click `FERRIC` in `/Applications` to launch the interactive WebGPU
2D/3D NMR spectrum workbench (`Metal` GPU backend with multi-threaded CPU `f64`
fallback).

2. Command-Line Interface (`ferric`)
------------------------------------
`ferric-gui`, the `ferric` CLI / MCP server, and the `v0.14.x` deprecated
`nmrassign` compatibility alias ship directly inside `FERRIC.app/Contents/MacOS/`.
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

Documentation & Browser Demo: https://debsahu.github.io/nmr-assign/
Repository & Source Code:     https://github.com/debsahu/nmr-assign
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
  rm -f "${BIN_DIR}/ferric-gui" "${BIN_DIR}/ferric" "${BIN_DIR}/nmrassign"
  rm -f "${APP_DIR}/ferric.desktop"
  rm -f "${ICON_SVG_DIR}/ferric.svg" "${ICON_PNG_DIR}/ferric.png"
  echo "Uninstalled FERRIC from ${PREFIX_DIR}"
  exit 0
fi

mkdir -p "${BIN_DIR}" "${APP_DIR}" "${ICON_SVG_DIR}" "${ICON_PNG_DIR}"
install -m 0755 "${SCRIPT_DIR}/bin/ferric-gui" "${BIN_DIR}/ferric-gui"
install -m 0755 "${SCRIPT_DIR}/bin/ferric" "${BIN_DIR}/ferric"
install -m 0755 "${SCRIPT_DIR}/bin/nmrassign" "${BIN_DIR}/nmrassign"
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
    """Apply ad-hoc Apple Silicon code signature and strict verification on macOS."""
    if sys.platform != "darwin" or not shutil.which("codesign"):
        return
    cli_bin = app_dir / "Contents" / "MacOS" / CLI_BIN
    alias_bin = app_dir / "Contents" / "MacOS" / CLI_ALIAS_BIN
    gui_bin = app_dir / "Contents" / "MacOS" / GUI_BIN
    subprocess.run(["codesign", "--force", "--sign", "-", str(cli_bin)], check=True, capture_output=True, text=True)
    subprocess.run(["codesign", "--force", "--sign", "-", str(alias_bin)], check=True, capture_output=True, text=True)
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


def package_macos_arm64(
    repo_root: Path,
    bin_dir: Path,
    out_dir: Path,
    version: str,
    skip_dmg: bool = False,
) -> List[Path]:
    """Build ``FERRIC.app``, ``.dmg``, and ``.tar.gz`` for ``macos-arm64``."""
    gui_src = _require_binary(bin_dir, GUI_BIN)
    cli_src = _require_binary(bin_dir, CLI_BIN)
    alias_src = _require_binary(bin_dir, CLI_ALIAS_BIN)
    readme_src = repo_root / "README.md"
    third_party_src = repo_root / "THIRD_PARTY.md"

    out_dir.mkdir(parents=True, exist_ok=True)
    pngs = generate_icon_pngs(PNG_SIZES)
    icns_bytes = build_icns(pngs)
    validate_icns_bytes(icns_bytes)

    prefix = f"{APP_SLUG}-v{version}-macos-arm64"
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
        alias_dst = macos_dir / CLI_ALIAS_BIN
        shutil.copy2(gui_src, gui_dst)
        shutil.copy2(cli_src, cli_dst)
        shutil.copy2(alias_src, alias_dst)
        os.chmod(gui_dst, 0o755)
        os.chmod(cli_dst, 0o755)
        os.chmod(alias_dst, 0o755)

        _codesign_macos_app(app_dir)

        # Copy signed .app bundle to out_dir for direct inspection / CI verification
        out_app_dir = out_dir / f"{APP_NAME}.app"
        if out_app_dir.exists():
            shutil.rmtree(out_app_dir)
        shutil.copytree(app_dir, out_app_dir, symlinks=True)

        # Top-level bin/{ferric,nmrassign} + docs inside .tar.gz staging root
        top_bin = tar_stage / "bin"
        top_bin.mkdir(parents=True, exist_ok=True)
        top_cli = top_bin / CLI_BIN
        top_alias = top_bin / CLI_ALIAS_BIN
        shutil.copy2(cli_src, top_cli)
        shutil.copy2(alias_src, top_alias)
        os.chmod(top_cli, 0o755)
        os.chmod(top_alias, 0o755)
        _codesign_macos_binary(top_cli)
        _codesign_macos_binary(top_alias)

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
                f"{APP_NAME}.app/Contents/MacOS/{CLI_ALIAS_BIN}",
                f"bin/{CLI_BIN}",
                f"bin/{CLI_ALIAS_BIN}",
            ),
        )
        outputs.append(tar_gz_path)

        if not skip_dmg:
            if sys.platform != "darwin" or not shutil.which("hdiutil"):
                raise RuntimeError(
                    "Building macos-arm64 .dmg requires macOS `hdiutil` (pass --skip-dmg to build .tar.gz only)."
                )
            dmg_stage = tmp_path / "dmg_stage"
            dmg_stage.mkdir(parents=True, exist_ok=True)
            shutil.copytree(app_dir, dmg_stage / f"{APP_NAME}.app", symlinks=True)
            os.symlink("/Applications", dmg_stage / "Applications")
            shutil.copyfile(readme_src, dmg_stage / "README.md")
            (dmg_stage / "CLI_AND_MCP_QUICKSTART.txt").write_text(
                render_cli_and_mcp_quickstart(version),
                encoding="utf-8",
                newline="\n",
            )
            dmg_path = out_dir / f"{prefix}.dmg"
            _create_macos_dmg(f"{APP_NAME} v{version}", dmg_stage, dmg_path)
            outputs.insert(0, dmg_path)

    return outputs


def package_windows_x86_64(
    repo_root: Path,
    bin_dir: Path,
    out_dir: Path,
    version: str,
) -> List[Path]:
    """Build deterministic ``.zip`` release bundle for ``windows-x86_64``."""
    gui_src = _require_binary(bin_dir, f"{GUI_BIN}.exe")
    cli_src = _require_binary(bin_dir, f"{CLI_BIN}.exe")
    alias_src = _require_binary(bin_dir, f"{CLI_ALIAS_BIN}.exe")
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
        shutil.copyfile(alias_src, stage / f"{CLI_ALIAS_BIN}.exe")
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
                f"{CLI_ALIAS_BIN}.exe",
                "Launch-NMR-Workbench.bat",
            ),
        )

    return [out_zip]


def package_linux_x86_64(
    repo_root: Path,
    bin_dir: Path,
    out_dir: Path,
    version: str,
) -> List[Path]:
    """Build deterministic ``.tar.gz`` FreeDesktop bundle for ``linux-x86_64``."""
    gui_src = _require_binary(bin_dir, GUI_BIN)
    cli_src = _require_binary(bin_dir, CLI_BIN)
    alias_src = _require_binary(bin_dir, CLI_ALIAS_BIN)
    readme_src = repo_root / "README.md"
    third_party_src = repo_root / "THIRD_PARTY.md"

    png_256 = generate_icon_pngs((256,))[256]
    validate_png_bytes(png_256, 256)
    svg_text = render_icon_svg()
    validate_svg_text(svg_text)
    desktop_text = render_linux_desktop_entry()
    validate_desktop_entry(desktop_text)

    prefix = f"{APP_SLUG}-v{version}-linux-x86_64"
    out_tar_gz = out_dir / f"{prefix}.tar.gz"

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / prefix
        bin_stage = stage / "bin"
        app_stage = stage / "share" / "applications"
        svg_stage = stage / "share" / "icons" / "hicolor" / "scalable" / "apps"
        png_stage = stage / "share" / "icons" / "hicolor" / "256x256" / "apps"
        for d in (bin_stage, app_stage, svg_stage, png_stage):
            d.mkdir(parents=True, exist_ok=True)

        shutil.copyfile(gui_src, bin_stage / GUI_BIN)
        shutil.copyfile(cli_src, bin_stage / CLI_BIN)
        shutil.copyfile(alias_src, bin_stage / CLI_ALIAS_BIN)
        os.chmod(bin_stage / GUI_BIN, 0o755)
        os.chmod(bin_stage / CLI_BIN, 0o755)
        os.chmod(bin_stage / CLI_ALIAS_BIN, 0o755)

        (app_stage / f"{APP_SLUG}.desktop").write_text(desktop_text, encoding="utf-8", newline="\n")
        (svg_stage / f"{APP_SLUG}.svg").write_text(svg_text, encoding="utf-8", newline="\n")
        (png_stage / f"{APP_SLUG}.png").write_bytes(png_256)

        installer_path = stage / "install-desktop.sh"
        installer_path.write_text(render_linux_install_desktop_sh(), encoding="utf-8", newline="\n")
        os.chmod(installer_path, 0o755)

        shutil.copyfile(readme_src, stage / "README.md")
        shutil.copyfile(third_party_src, stage / "THIRD_PARTY.md")

        _write_deterministic_tar_gz(
            staging_root=stage,
            archive_prefix=prefix,
            out_tar_gz=out_tar_gz,
            executable_relpaths=(
                f"bin/{GUI_BIN}",
                f"bin/{CLI_BIN}",
                f"bin/{CLI_ALIAS_BIN}",
                "install-desktop.sh",
            ),
        )

    return [out_tar_gz]


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
        if entry.name.endswith((".dmg", ".tar.gz", ".zip")):
            archives.append(entry)
    return archives


def write_checksums(target_dir: Path) -> Path:
    """Write deterministic ``SHA256SUMS.txt`` over all release archives in ``target_dir``."""
    archives = _iter_release_archives(target_dir)
    if not archives:
        raise ValueError(f"No release archives (.dmg, .tar.gz, .zip) found in {target_dir}")
    lines: List[str] = []
    for arc in archives:
        size = arc.stat().st_size
        if size <= 0:
            raise ValueError(f"Release archive is empty (0 bytes): {arc}")
        lines.append(f"{_sha256_file(arc)}  {arc.name}")
    out_file = target_dir / "SHA256SUMS.txt"
    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
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


def _build_mock_binaries(mock_bin_dir: Path) -> None:
    """Create tiny valid mock executables for ``--self-test`` across all 3 platforms."""
    mock_bin_dir.mkdir(parents=True, exist_ok=True)
    unix_stub: Optional[bytes] = None
    if sys.platform == "darwin" and shutil.which("cc"):
        c_src = mock_bin_dir / "_stub.c"
        out_macho = mock_bin_dir / "_stub_bin"
        c_src.write_text("int main(void) { return 0; }\n", encoding="utf-8")
        res = subprocess.run(["cc", "-O2", str(c_src), "-o", str(out_macho)], capture_output=True, text=True)
        if res.returncode == 0 and out_macho.is_file():
            unix_stub = out_macho.read_bytes()
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


def run_self_test(repo_root: Path) -> int:
    """Exercise icon encoders, all 3 platform packagers, permissions, and SHA256SUMS."""
    version = read_workspace_version(repo_root)
    is_macos_dmg_capable = sys.platform == "darwin" and bool(shutil.which("hdiutil"))

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        icons_dir = tmp_path / "icons"
        emitted = emit_icon_suite(icons_dir)
        # Verify deterministic byte-identical icon generation
        icons_dir_2 = tmp_path / "icons_repeat"
        emitted_2 = emit_icon_suite(icons_dir_2)
        for key, p1 in emitted.items():
            if p1.read_bytes() != emitted_2[key].read_bytes():
                raise AssertionError(f"Non-deterministic icon output for {key}")
        print(
            f"[self-test 1/5] Icon suite OK: PNGs (16,32,48,64,128,256,512), "
            f"ICNS ({emitted['icns'].stat().st_size} B), "
            f"ICO ({emitted['ico'].stat().st_size} B), "
            f"SVG ({emitted['svg'].stat().st_size} B)"
        )

        mock_bins = tmp_path / "mock_bins"
        _build_mock_binaries(mock_bins)
        dist_dir = tmp_path / "dist"

        # 1. macOS arm64 bundle + .dmg (on macOS) + .tar.gz
        mac_outputs = package_macos_arm64(
            repo_root=repo_root,
            bin_dir=mock_bins,
            out_dir=dist_dir,
            version=version,
            skip_dmg=not is_macos_dmg_capable,
        )
        app_bundle = dist_dir / f"{APP_NAME}.app"
        validate_info_plist(app_bundle / "Contents" / "Info.plist", version)
        if (app_bundle / "Contents" / "PkgInfo").read_bytes() != b"APPL????":
            raise AssertionError("Invalid Contents/PkgInfo bytes")
        validate_icns_bytes((app_bundle / "Contents" / "Resources" / f"{APP_NAME}.icns").read_bytes())

        mac_prefix = f"{APP_SLUG}-v{version}-macos-arm64"
        mac_tar = dist_dir / f"{mac_prefix}.tar.gz"
        with tarfile.open(mac_tar, "r:gz") as tf:
            members = {m.name: m for m in tf.getmembers()}
            for req_exec in (
                f"{mac_prefix}/{APP_NAME}.app/Contents/MacOS/{GUI_BIN}",
                f"{mac_prefix}/{APP_NAME}.app/Contents/MacOS/{CLI_BIN}",
                f"{mac_prefix}/{APP_NAME}.app/Contents/MacOS/{CLI_ALIAS_BIN}",
                f"{mac_prefix}/bin/{CLI_BIN}",
                f"{mac_prefix}/bin/{CLI_ALIAS_BIN}",
            ):
                if req_exec not in members:
                    raise AssertionError(f"Missing {req_exec} in macOS .tar.gz")
                if (members[req_exec].mode & 0o777) != 0o755:
                    raise AssertionError(f"Expected 0755 on {req_exec}, got {oct(members[req_exec].mode)}")
            for req_doc in (
                f"{mac_prefix}/{APP_NAME}.app/Contents/Info.plist",
                f"{mac_prefix}/{APP_NAME}.app/Contents/PkgInfo",
                f"{mac_prefix}/{APP_NAME}.app/Contents/Resources/{APP_NAME}.icns",
                f"{mac_prefix}/README.md",
                f"{mac_prefix}/THIRD_PARTY.md",
            ):
                if req_doc not in members:
                    raise AssertionError(f"Missing {req_doc} in macOS .tar.gz")

        if is_macos_dmg_capable:
            mac_dmg = dist_dir / f"{mac_prefix}.dmg"
            if not mac_dmg.is_file() or mac_dmg.stat().st_size <= 0:
                raise AssertionError("macOS .dmg was not created")
            subprocess.run(["hdiutil", "imageinfo", str(mac_dmg)], check=True, capture_output=True, text=True)
        print(
            f"[self-test 2/5] macos-arm64 OK: {[p.name for p in mac_outputs]} "
            f"(Info.plist + PkgInfo + 0755 binaries"
            f"{' + codesign + UDZO .dmg' if is_macos_dmg_capable else ''})"
        )

        # 2. Windows x86_64 .zip
        win_outputs = package_windows_x86_64(repo_root, mock_bins, dist_dir, version)
        win_prefix = f"{APP_SLUG}-v{version}-windows-x86_64"
        win_zip = win_outputs[0]
        with zipfile.ZipFile(win_zip, "r") as zf:
            names = set(zf.namelist())
            for req_win in (
                f"{win_prefix}/{GUI_BIN}.exe",
                f"{win_prefix}/{CLI_BIN}.exe",
                f"{win_prefix}/{CLI_ALIAS_BIN}.exe",
                f"{win_prefix}/{APP_SLUG}.ico",
                f"{win_prefix}/Launch-NMR-Workbench.bat",
                f"{win_prefix}/README.md",
                f"{win_prefix}/THIRD_PARTY.md",
            ):
                if req_win not in names:
                    raise AssertionError(f"Missing {req_win} in Windows .zip")
            validate_ico_bytes(zf.read(f"{win_prefix}/{APP_SLUG}.ico"), ICO_SIZES)
        win_repeat_dir = tmp_path / "win_repeat"
        win_repeat = package_windows_x86_64(repo_root, mock_bins, win_repeat_dir, version)[0]
        if win_zip.read_bytes() != win_repeat.read_bytes():
            raise AssertionError("Windows .zip output is not byte-for-byte deterministic")
        print(f"[self-test 3/5] windows-x86_64 OK: {win_zip.name} ({win_zip.stat().st_size} B, deterministic)")

        # 3. Linux x86_64 .tar.gz + XDG desktop installer test
        lin_outputs = package_linux_x86_64(repo_root, mock_bins, dist_dir, version)
        lin_prefix = f"{APP_SLUG}-v{version}-linux-x86_64"
        lin_tar = lin_outputs[0]
        with tarfile.open(lin_tar, "r:gz") as tf:
            members = {m.name: m for m in tf.getmembers()}
            for req_exec in (
                f"{lin_prefix}/bin/{GUI_BIN}",
                f"{lin_prefix}/bin/{CLI_BIN}",
                f"{lin_prefix}/bin/{CLI_ALIAS_BIN}",
                f"{lin_prefix}/install-desktop.sh",
            ):
                if req_exec not in members:
                    raise AssertionError(f"Missing {req_exec} in Linux .tar.gz")
                if (members[req_exec].mode & 0o777) != 0o755:
                    raise AssertionError(f"Expected 0755 on {req_exec}, got {oct(members[req_exec].mode)}")
            for req_file in (
                f"{lin_prefix}/share/applications/{APP_SLUG}.desktop",
                f"{lin_prefix}/share/icons/hicolor/scalable/apps/{APP_SLUG}.svg",
                f"{lin_prefix}/share/icons/hicolor/256x256/apps/{APP_SLUG}.png",
                f"{lin_prefix}/README.md",
                f"{lin_prefix}/THIRD_PARTY.md",
            ):
                if req_file not in members:
                    raise AssertionError(f"Missing {req_file} in Linux .tar.gz")
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
                subprocess.run(
                    ["bash", str(extract_dir / lin_prefix / "install-desktop.sh"), "--prefix", str(xdg_prefix)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                installed_desktop = xdg_prefix / "share" / "applications" / f"{APP_SLUG}.desktop"
                if (
                    not (xdg_prefix / "bin" / GUI_BIN).is_file()
                    or not (xdg_prefix / "bin" / CLI_BIN).is_file()
                    or not (xdg_prefix / "bin" / CLI_ALIAS_BIN).is_file()
                    or not installed_desktop.is_file()
                ):
                    raise AssertionError("install-desktop.sh failed to populate XDG prefix")

        lin_repeat_dir = tmp_path / "lin_repeat"
        lin_repeat = package_linux_x86_64(repo_root, mock_bins, lin_repeat_dir, version)[0]
        if lin_tar.read_bytes() != lin_repeat.read_bytes():
            raise AssertionError("Linux .tar.gz output is not byte-for-byte deterministic")
        print(f"[self-test 4/5] linux-x86_64 OK: {lin_tar.name} ({lin_tar.stat().st_size} B, deterministic)")

        # 4. Checksum generation & tamper detection
        sums_file = write_checksums(dist_dir)
        verified = verify_checksums(dist_dir)
        expected_count = 4 if is_macos_dmg_capable else 3
        if len(verified) != expected_count:
            raise AssertionError(f"Expected {expected_count} checksummed archives, got {len(verified)}")

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
            f"[self-test 5/5] SHA256SUMS.txt OK: {len(verified)} archives verified "
            f"({sums_file.stat().st_size} B) + tamper rejection confirmed"
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
        help="Skip macOS hdiutil .dmg image creation when packaging macos-arm64.",
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
            elif target == "windows-x86_64":
                built = package_windows_x86_64(repo_root, bin_dir, out_dir, version)
            elif target == "linux-x86_64":
                built = package_linux_x86_64(repo_root, bin_dir, out_dir, version)
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
