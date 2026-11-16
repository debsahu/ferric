#!/usr/bin/env python3
"""Deterministic pure-Python generator for the canonical FERRIC brand suite.

Zero third-party dependencies (Python 3 stdlib only). Generates:
  - docs/assets/brand/ferric-icon.svg
  - docs/assets/brand/ferric-icon-mono.svg
  - docs/assets/brand/ferric-wordmark.svg
  - docs/assets/brand/ferric-logo-light.svg
  - docs/assets/brand/ferric-logo-dark.svg
  - docs/assets/brand/favicon.svg
  - docs/assets/brand/favicon-32.png (32x32 RGBA PNG)
  - docs/assets/brand/ferric-social-preview.png (1280x640 RGBA PNG)
  - docs/assets/brand/previews/packager_icon_{16,32,512}px.png
  - docs/assets/brand/previews/packager_vs_concept_a_diff.png

Usage:
    python3 scripts/render_ferric_brand.py --self-test
    python3 scripts/render_ferric_brand.py --dump-icon-previews docs/assets/brand/previews
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
import struct
import sys
import tempfile
from typing import Dict, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET
import zlib

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

# Canonical FERRIC Palette
PALETTE = {
    "iron_slate": "#16202A",
    "midnight_bg": "#0B1120",
    "rust_primary": "#B7410E",
    "rust_frame": "#D95B26",
    "ember_mid": "#E05A2B",
    "fid_bright": "#FB923C",
    "cap_highlight": "#FED7AA",
    "axis_slate": "#475569",
    "ink_dark": "#0F172A",
    "ink_light": "#F8FAFC",
}

LIGHT_BG_RGB = (248, 250, 252)
DARK_BG_RGB = (11, 17, 32)
MONO_LIGHT_RGB = (15, 23, 42)


def _hex_to_rgb(color: str, current_color: Tuple[int, int, int] = MONO_LIGHT_RGB) -> Tuple[int, int, int]:
    if color == "currentColor":
        return current_color
    c = color.lstrip("#")
    if len(c) != 6:
        raise ValueError(f"Expected #RRGGBB color, got {color!r}")
    return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16))


def _clamp_u8(val: float) -> int:
    if val <= 0.0:
        return 0
    if val >= 255.0:
        return 255
    return int(round(val))


def _fmt(val: float) -> str:
    if abs(val - round(val)) < 1e-6:
        return str(int(round(val)))
    return f"{val:.2f}".rstrip("0").rstrip(".")


# ---------------------------------------------------------------------------
# Vector primitives + 4x4 supersampled SDF rasterizer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoundedRect:
    x: float
    y: float
    w: float
    h: float
    rx: float
    fill: Optional[str] = None
    stroke: Optional[str] = None
    stroke_width: float = 0.0

    def bbox(self) -> Tuple[float, float, float, float]:
        pad = 0.5 * self.stroke_width if self.stroke else 0.0
        return (self.x - pad, self.y - pad, self.x + self.w + pad, self.y + self.h + pad)

    def sdf_fill(self, px: float, py: float) -> float:
        cx = self.x + 0.5 * self.w
        cy = self.y + 0.5 * self.h
        hx = max(0.0, 0.5 * self.w - self.rx)
        hy = max(0.0, 0.5 * self.h - self.rx)
        qx = abs(px - cx) - hx
        qy = abs(py - cy) - hy
        return math.hypot(max(qx, 0.0), max(qy, 0.0)) + min(max(qx, qy), 0.0) - self.rx


@dataclass(frozen=True)
class Circle:
    cx: float
    cy: float
    r: float
    fill: Optional[str] = None
    stroke: Optional[str] = None
    stroke_width: float = 0.0

    def bbox(self) -> Tuple[float, float, float, float]:
        pad = 0.5 * self.stroke_width if self.stroke else 0.0
        rad = self.r + pad
        return (self.cx - rad, self.cy - rad, self.cx + rad, self.cy + rad)

    def sdf_fill(self, px: float, py: float) -> float:
        return math.hypot(px - self.cx, py - self.cy) - self.r


@dataclass(frozen=True)
class LineSegment:
    x1: float
    y1: float
    x2: float
    y2: float
    stroke: str
    stroke_width: float

    def bbox(self) -> Tuple[float, float, float, float]:
        pad = 0.5 * self.stroke_width
        return (
            min(self.x1, self.x2) - pad,
            min(self.y1, self.y2) - pad,
            max(self.x1, self.x2) + pad,
            max(self.y1, self.y2) + pad,
        )

    def sdf_stroke(self, px: float, py: float) -> float:
        bax = self.x2 - self.x1
        bay = self.y2 - self.y1
        denom = bax * bax + bay * bay
        if denom <= 1e-12:
            return math.hypot(px - self.x1, py - self.y1) - 0.5 * self.stroke_width
        h = ((px - self.x1) * bax + (py - self.y1) * bay) / denom
        h = 0.0 if h < 0.0 else (1.0 if h > 1.0 else h)
        return math.hypot((px - self.x1) - bax * h, (py - self.y1) - bay * h) - 0.5 * self.stroke_width


@dataclass(frozen=True)
class Polyline:
    points: Tuple[Tuple[float, float], ...]
    stroke: str
    stroke_width: float
    closed: bool = False

    def bbox(self) -> Tuple[float, float, float, float]:
        pad = 0.5 * self.stroke_width
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)

    def sdf_stroke(self, px: float, py: float) -> float:
        best = 1e9
        pts = self.points
        n = len(pts)
        limit = n if self.closed else n - 1
        for i in range(limit):
            ax, ay = pts[i]
            bx, by = pts[(i + 1) % n]
            bax = bx - ax
            bay = by - ay
            denom = bax * bax + bay * bay
            if denom <= 1e-12:
                d = math.hypot(px - ax, py - ay)
            else:
                h = ((px - ax) * bax + (py - ay) * bay) / denom
                h = 0.0 if h < 0.0 else (1.0 if h > 1.0 else h)
                d = math.hypot((px - ax) - bax * h, (py - ay) - bay * h)
            if d < best:
                best = d
        return best - 0.5 * self.stroke_width


@dataclass(frozen=True)
class Polygon:
    points: Tuple[Tuple[float, float], ...]
    fill: str

    def bbox(self) -> Tuple[float, float, float, float]:
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return (min(xs), min(ys), max(xs), max(ys))

    def sdf_fill(self, px: float, py: float) -> float:
        pts = self.points
        n = len(pts)
        inside = False
        best_sq = 1e18
        j = n - 1
        for i in range(n):
            xi, yi = pts[i]
            xj, yj = pts[j]
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / ((yj - yi) or 1e-18) + xi):
                inside = not inside
            ex, ey = xj - xi, yj - yi
            denom = ex * ex + ey * ey
            if denom > 1e-12:
                t = ((px - xi) * ex + (py - yi) * ey) / denom
                t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
                dx = px - (xi + t * ex)
                dy = py - (yi + t * ey)
                d2 = dx * dx + dy * dy
            else:
                d2 = (px - xi) ** 2 + (py - yi) ** 2
            if d2 < best_sq:
                best_sq = d2
            j = i
        d = math.sqrt(best_sq)
        return -d if inside else d


@dataclass(frozen=True)
class PolyCompound:
    """Multiple closed polygon rings with even-odd fill (for glyphs with holes like 'R')."""
    rings: Tuple[Tuple[Tuple[float, float], ...], ...]
    fill: str

    def bbox(self) -> Tuple[float, float, float, float]:
        xs = [p[0] for r in self.rings for p in r]
        ys = [p[1] for r in self.rings for p in r]
        return (min(xs), min(ys), max(xs), max(ys))

    def sdf_fill(self, px: float, py: float) -> float:
        inside = False
        best_sq = 1e18
        for pts in self.rings:
            n = len(pts)
            j = n - 1
            for i in range(n):
                xi, yi = pts[i]
                xj, yj = pts[j]
                if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / ((yj - yi) or 1e-18) + xi):
                    inside = not inside
                ex, ey = xj - xi, yj - yi
                denom = ex * ex + ey * ey
                if denom > 1e-12:
                    t = ((px - xi) * ex + (py - yi) * ey) / denom
                    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
                    dx = px - (xi + t * ex)
                    dy = py - (yi + t * ey)
                    d2 = dx * dx + dy * dy
                else:
                    d2 = (px - xi) ** 2 + (py - yi) ** 2
                if d2 < best_sq:
                    best_sq = d2
                j = i
        d = math.sqrt(best_sq)
        return -d if inside else d


Shape = object


def sample_fid_wave(
    x_start: float,
    x_end: float,
    y_base: float,
    amp: float,
    decay: float,
    cycles: float,
    phase: float = 0.0,
    samples: int = 120,
) -> Tuple[Tuple[float, float], ...]:
    pts: List[Tuple[float, float]] = []
    for i in range(samples + 1):
        u = i / samples
        x = x_start + (x_end - x_start) * u
        y = y_base - amp * math.exp(-decay * u) * math.cos(2.0 * math.pi * cycles * u + phase)
        pts.append((x, y))
    return tuple(pts)


def _sample_ellipse(cx: float, cy: float, rx: float, ry: float, samples: int = 96) -> Tuple[Tuple[float, float], ...]:
    pts: List[Tuple[float, float]] = []
    for i in range(samples):
        ang = 2.0 * math.pi * i / samples
        pts.append((cx + rx * math.cos(ang), cy + ry * math.sin(ang)))
    return tuple(pts)


def _arrow_triangle(
    tip: Tuple[float, float],
    tail: Tuple[float, float],
    length: float,
    half_width: float,
) -> Tuple[Tuple[float, float], ...]:
    dx = tip[0] - tail[0]
    dy = tip[1] - tail[1]
    mag = math.hypot(dx, dy) or 1.0
    ux, uy = dx / mag, dy / mag
    nx, ny = -uy, ux
    bx = tip[0] - ux * length
    by = tip[1] - uy * length
    return (
        tip,
        (bx + nx * half_width, by + ny * half_width),
        (bx - nx * half_width, by - ny * half_width),
    )


# ---------------------------------------------------------------------------
# Canonical Icon (Concept A: FID-F with rust-framed #16202A squircle)
# ---------------------------------------------------------------------------


def build_canonical_icon_shapes(mono: bool = False) -> Tuple[Shape, ...]:
    """Canonical Concept A (FID-F) on 512x512 viewBox."""
    fid_pts = sample_fid_wave(
        x_start=168.0,
        x_end=414.0,
        y_base=248.0,
        amp=120.0,
        decay=2.7,
        cycles=2.5,
        phase=0.0,
        samples=120,
    )
    if mono:
        ink = "currentColor"
        return (
            RoundedRect(32.0, 32.0, 448.0, 448.0, 104.0, fill=None, stroke=ink, stroke_width=24.0),
            LineSegment(152.0, 124.0, 152.0, 388.0, stroke=ink, stroke_width=52.0),
            LineSegment(152.0, 124.0, 376.0, 124.0, stroke=ink, stroke_width=52.0),
            LineSegment(152.0, 248.0, 304.0, 248.0, stroke=ink, stroke_width=44.0),
            Polyline(fid_pts, stroke=ink, stroke_width=24.0, closed=False),
        )
    return (
        RoundedRect(32.0, 32.0, 448.0, 448.0, 104.0, fill="#16202A", stroke="#B7410E", stroke_width=24.0),
        RoundedRect(44.0, 44.0, 424.0, 424.0, 92.0, fill=None, stroke="#D95B26", stroke_width=8.0),
        LineSegment(174.0, 248.0, 416.0, 248.0, stroke="#475569", stroke_width=10.0),
        LineSegment(152.0, 124.0, 152.0, 388.0, stroke="#B7410E", stroke_width=56.0),
        LineSegment(152.0, 124.0, 376.0, 124.0, stroke="#B7410E", stroke_width=56.0),
        LineSegment(152.0, 248.0, 316.0, 248.0, stroke="#B7410E", stroke_width=48.0),
        LineSegment(152.0, 124.0, 152.0, 388.0, stroke="#E05A2B", stroke_width=34.0),
        LineSegment(152.0, 124.0, 376.0, 124.0, stroke="#E05A2B", stroke_width=34.0),
        LineSegment(152.0, 248.0, 316.0, 248.0, stroke="#E05A2B", stroke_width=28.0),
        Polyline(fid_pts, stroke="#FB923C", stroke_width=28.0, closed=False),
        Circle(152.0, 124.0, 16.0, fill="#FED7AA"),
    )


def build_larmor_emblem_shapes() -> Tuple[Shape, ...]:
    """Concept C Larmor precession-cone + spin-vector + damped FID emblem on 512x512 viewBox."""
    orbit_pts = _sample_ellipse(256.0, 158.0, 112.0, 38.0, samples=96)
    fid_pts = sample_fid_wave(
        x_start=104.0,
        x_end=408.0,
        y_base=358.0,
        amp=40.0,
        decay=2.0,
        cycles=2.5,
        phase=0.0,
        samples=120,
    )
    arrow_head = _arrow_triangle((344.0, 174.0), (256.0, 300.0), length=58.0, half_width=30.0)
    return (
        Circle(256.0, 256.0, 224.0, fill="#16202A", stroke="#B7410E", stroke_width=24.0),
        Circle(256.0, 256.0, 200.0, fill=None, stroke="#D95B26", stroke_width=8.0),
        LineSegment(256.0, 300.0, 256.0, 100.0, stroke="#64748B", stroke_width=12.0),
        LineSegment(256.0, 300.0, 144.0, 158.0, stroke="#64748B", stroke_width=14.0),
        LineSegment(256.0, 300.0, 368.0, 158.0, stroke="#64748B", stroke_width=14.0),
        Polyline(orbit_pts, stroke="#FB923C", stroke_width=24.0, closed=True),
        LineSegment(104.0, 358.0, 408.0, 358.0, stroke="#475569", stroke_width=8.0),
        Polyline(fid_pts, stroke="#FB923C", stroke_width=26.0, closed=False),
        LineSegment(256.0, 300.0, 318.0, 211.0, stroke="#F8FAFC", stroke_width=28.0),
        Polygon(arrow_head, fill="#F8FAFC"),
        Circle(256.0, 300.0, 14.0, fill="#94A3B8"),
        Circle(346.0, 171.0, 20.0, fill="#FB923C"),
        Circle(346.0, 171.0, 9.0, fill="#FED7AA"),
    )


# ---------------------------------------------------------------------------
# Original MIT-Licensed Geometric <path> Wordmark ("FERRIC")
# ---------------------------------------------------------------------------


def _sample_cubic(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    steps: int = 12,
) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    for i in range(1, steps + 1):
        t = i / steps
        mt = 1.0 - t
        x = mt**3 * p0[0] + 3 * mt**2 * t * p1[0] + 3 * mt * t**2 * p2[0] + t**3 * p3[0]
        y = mt**3 * p0[1] + 3 * mt**2 * t * p1[1] + 3 * mt * t**2 * p2[1] + t**3 * p3[1]
        pts.append((x, y))
    return pts


def _wordmark_glyphs() -> List[Tuple[str, float, List[List[Tuple[float, float]]]]]:
    """Return original MIT geometric glyph outlines (cap height 80) as (svg_d, advance, rings)."""
    glyphs: List[Tuple[str, float, List[List[Tuple[float, float]]]]] = []

    # F (width 54)
    f_pts = [(0.0, 0.0), (54.0, 0.0), (54.0, 15.0), (16.0, 15.0), (16.0, 33.0), (46.0, 33.0), (46.0, 48.0), (16.0, 48.0), (16.0, 80.0), (0.0, 80.0)]
    f_d = "M 0 0 H 54 V 15 H 16 V 33 H 46 V 48 H 16 V 80 H 0 Z"
    glyphs.append((f_d, 66.0, [f_pts]))

    # E (width 56)
    e_pts = [(0.0, 0.0), (56.0, 0.0), (56.0, 15.0), (16.0, 15.0), (16.0, 32.0), (48.0, 32.0), (48.0, 47.0), (16.0, 47.0), (16.0, 65.0), (56.0, 65.0), (56.0, 80.0), (0.0, 80.0)]
    e_d = "M 0 0 H 56 V 15 H 16 V 32 H 48 V 47 H 16 V 65 H 56 V 80 H 0 Z"
    glyphs.append((e_d, 68.0, [e_pts]))

    # R (width 64, outer + inner counter)
    r_outer: List[Tuple[float, float]] = [(0.0, 0.0), (40.0, 0.0)]
    r_outer += _sample_cubic((40.0, 0.0), (55.0, 0.0), (62.0, 9.0), (62.0, 23.0))
    r_outer += _sample_cubic((62.0, 23.0), (62.0, 35.0), (55.0, 43.0), (44.0, 45.0))
    r_outer += [(64.0, 80.0), (46.0, 80.0), (28.0, 47.0), (16.0, 47.0), (16.0, 80.0), (0.0, 80.0)]

    r_inner: List[Tuple[float, float]] = [(16.0, 14.0), (16.0, 34.0), (37.0, 34.0)]
    r_inner += _sample_cubic((37.0, 34.0), (44.0, 34.0), (46.0, 30.0), (46.0, 24.0))
    r_inner += _sample_cubic((46.0, 24.0), (46.0, 18.0), (44.0, 14.0), (37.0, 14.0))

    r_d = (
        "M 0 0 H 40 C 55 0 62 9 62 23 C 62 35 55 43 44 45 "
        "L 64 80 H 46 L 28 47 H 16 V 80 H 0 Z "
        "M 16 14 V 34 H 37 C 44 34 46 30 46 24 C 46 18 44 14 37 14 Z"
    )
    glyphs.append((r_d, 74.0, [r_outer, r_inner]))
    glyphs.append((r_d, 74.0, [r_outer, r_inner]))

    # I (width 18)
    i_pts = [(0.0, 0.0), (18.0, 0.0), (18.0, 80.0), (0.0, 80.0)]
    i_d = "M 0 0 H 18 V 80 H 0 Z"
    glyphs.append((i_d, 30.0, [i_pts]))

    # C (width 58)
    c_pts: List[Tuple[float, float]] = [(56.0, 16.0)]
    c_pts += _sample_cubic((56.0, 16.0), (48.0, 5.0), (38.0, 0.0), (26.0, 0.0))
    c_pts += _sample_cubic((26.0, 0.0), (9.0, 0.0), (0.0, 14.0), (0.0, 40.0))
    c_pts += _sample_cubic((0.0, 40.0), (0.0, 66.0), (9.0, 80.0), (26.0, 80.0))
    c_pts += _sample_cubic((26.0, 80.0), (38.0, 80.0), (48.0, 75.0), (56.0, 64.0))
    c_pts.append((44.0, 53.0))
    c_pts += _sample_cubic((44.0, 53.0), (39.0, 61.0), (33.0, 65.0), (26.0, 65.0))
    c_pts += _sample_cubic((26.0, 65.0), (18.0, 65.0), (16.0, 56.0), (16.0, 40.0))
    c_pts += _sample_cubic((16.0, 40.0), (16.0, 24.0), (18.0, 15.0), (26.0, 15.0))
    c_pts += _sample_cubic((26.0, 15.0), (33.0, 15.0), (39.0, 19.0), (44.0, 27.0))

    c_d = (
        "M 56 16 C 48 5 38 0 26 0 C 9 0 0 14 0 40 C 0 66 9 80 26 80 "
        "C 38 80 48 75 56 64 L 44 53 C 39 61 33 65 26 65 "
        "C 18 65 16 56 16 40 C 16 24 18 15 26 15 C 33 15 39 19 44 27 Z"
    )
    glyphs.append((c_d, 58.0, [c_pts]))
    return glyphs


def build_wordmark_shapes(x0: float, y0: float, scale: float, fill: str) -> Tuple[Shape, ...]:
    shapes: List[Shape] = []
    cx = x0
    for _, adv, rings in _wordmark_glyphs():
        scaled_rings = tuple(tuple((cx + px * scale, y0 + py * scale) for px, py in ring) for ring in rings)
        shapes.append(PolyCompound(scaled_rings, fill=fill))
        cx += adv * scale
    return tuple(shapes)


def _shapes_to_svg_elements(shapes: Sequence[Shape], indent: str = "  ") -> List[str]:
    lines: List[str] = []
    for shape in shapes:
        if isinstance(shape, RoundedRect):
            fill_attr = shape.fill if shape.fill is not None else "none"
            attrs = [
                f'x="{_fmt(shape.x)}"',
                f'y="{_fmt(shape.y)}"',
                f'width="{_fmt(shape.w)}"',
                f'height="{_fmt(shape.h)}"',
                f'rx="{_fmt(shape.rx)}"',
                f'ry="{_fmt(shape.rx)}"',
                f'fill="{fill_attr}"',
            ]
            if shape.stroke is not None:
                attrs.append(f'stroke="{shape.stroke}"')
                attrs.append(f'stroke-width="{_fmt(shape.stroke_width)}"')
            lines.append(f"{indent}<rect {' '.join(attrs)}/>")
        elif isinstance(shape, Circle):
            fill_attr = shape.fill if shape.fill is not None else "none"
            attrs = [
                f'cx="{_fmt(shape.cx)}"',
                f'cy="{_fmt(shape.cy)}"',
                f'r="{_fmt(shape.r)}"',
                f'fill="{fill_attr}"',
            ]
            if shape.stroke is not None:
                attrs.append(f'stroke="{shape.stroke}"')
                attrs.append(f'stroke-width="{_fmt(shape.stroke_width)}"')
            lines.append(f"{indent}<circle {' '.join(attrs)}/>")
        elif isinstance(shape, LineSegment):
            lines.append(
                f'{indent}<line x1="{_fmt(shape.x1)}" y1="{_fmt(shape.y1)}" '
                f'x2="{_fmt(shape.x2)}" y2="{_fmt(shape.y2)}" '
                f'stroke="{shape.stroke}" stroke-width="{_fmt(shape.stroke_width)}" stroke-linecap="round"/>'
            )
        elif isinstance(shape, Polyline):
            cmd_parts = [f"M {_fmt(shape.points[0][0])} {_fmt(shape.points[0][1])}"]
            for px, py in shape.points[1:]:
                cmd_parts.append(f"L {_fmt(px)} {_fmt(py)}")
            if shape.closed:
                cmd_parts.append("Z")
            lines.append(
                f'{indent}<path d="{" ".join(cmd_parts)}" fill="none" stroke="{shape.stroke}" '
                f'stroke-width="{_fmt(shape.stroke_width)}" stroke-linecap="round" stroke-linejoin="round"/>'
            )
        elif isinstance(shape, Polygon):
            pts_str = " ".join(f"{_fmt(px)},{_fmt(py)}" for px, py in shape.points)
            lines.append(f'{indent}<polygon points="{pts_str}" fill="{shape.fill}"/>')
        else:
            raise TypeError(f"Unsupported shape: {type(shape)!r}")
    return lines


def render_icon_svg(mono: bool = False) -> str:
    shapes = build_canonical_icon_shapes(mono=mono)
    title = "FERRIC Icon (Monochrome)" if mono else "FERRIC Icon"
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512" role="img">',
        f"  <title>{title}</title>",
    ]
    lines.extend(_shapes_to_svg_elements(shapes))
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def render_wordmark_svg() -> str:
    """Render standalone MIT-licensed geometric <path> wordmark 'FERRIC' (zero <text>)."""
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 100" width="400" height="100" role="img">',
        "  <title>FERRIC Wordmark</title>",
        '  <g transform="translate(15, 10)" fill="currentColor" fill-rule="evenodd">',
    ]
    cx = 0.0
    for d_str, adv, _ in _wordmark_glyphs():
        lines.append(f'    <path transform="translate({_fmt(cx)}, 0)" d="{d_str}"/>')
        cx += adv
    lines.append("  </g>")
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def render_lockup_svg(dark: bool) -> str:
    """Render README header lockup: Concept A badge + FERRIC <path> wordmark + Concept C Larmor emblem."""
    ink = "#F8FAFC" if dark else "#0F172A"
    sub_rule = "#D95B26" if dark else "#B7410E"
    title = "FERRIC Logo (Dark Mode)" if dark else "FERRIC Logo (Light Mode)"

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 160" width="760" height="160" role="img">',
        f"  <title>{title}</title>",
        '  <g transform="translate(16, 16) scale(0.25)">',
    ]
    lines.extend(_shapes_to_svg_elements(build_canonical_icon_shapes(mono=False), indent="    "))
    lines.append("  </g>")

    lines.append(f'  <g transform="translate(174, 34)" fill="{ink}" fill-rule="evenodd">')
    cx = 0.0
    for d_str, adv, _ in _wordmark_glyphs():
        lines.append(f'    <path transform="translate({_fmt(cx)}, 0)" d="{d_str}"/>')
        cx += adv
    lines.append("  </g>")

    # Rust-orange accent rule below wordmark
    lines.append(f'  <rect x="174" y="128" width="370" height="6" rx="3" fill="{sub_rule}"/>')

    # Concept C Larmor precession-cone + FID emblem on the right
    lines.append('  <g transform="translate(616, 16) scale(0.25)">')
    lines.extend(_shapes_to_svg_elements(build_larmor_emblem_shapes(), indent="    "))
    lines.append("  </g>")
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Rasterization & PNG Encoding
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CompiledLayer:
    kind: str
    shape: Shape
    rgb: Tuple[int, int, int]
    bbox: Tuple[float, float, float, float]


def _compile_layers(
    shapes: Sequence[Shape],
    current_color: Tuple[int, int, int] = MONO_LIGHT_RGB,
) -> List[_CompiledLayer]:
    layers: List[_CompiledLayer] = []
    for sh in shapes:
        bb = sh.bbox()
        if isinstance(sh, RoundedRect):
            if sh.fill is not None:
                layers.append(_CompiledLayer("rect_fill", sh, _hex_to_rgb(sh.fill, current_color), bb))
            if sh.stroke is not None and sh.stroke_width > 0.0:
                layers.append(_CompiledLayer("rect_stroke", sh, _hex_to_rgb(sh.stroke, current_color), bb))
        elif isinstance(sh, Circle):
            if sh.fill is not None:
                layers.append(_CompiledLayer("circle_fill", sh, _hex_to_rgb(sh.fill, current_color), bb))
            if sh.stroke is not None and sh.stroke_width > 0.0:
                layers.append(_CompiledLayer("circle_stroke", sh, _hex_to_rgb(sh.stroke, current_color), bb))
        elif isinstance(sh, LineSegment):
            layers.append(_CompiledLayer("line", sh, _hex_to_rgb(sh.stroke, current_color), bb))
        elif isinstance(sh, Polyline):
            layers.append(_CompiledLayer("poly", sh, _hex_to_rgb(sh.stroke, current_color), bb))
        elif isinstance(sh, Polygon):
            layers.append(_CompiledLayer("polygon", sh, _hex_to_rgb(sh.fill, current_color), bb))
        elif isinstance(sh, PolyCompound):
            layers.append(_CompiledLayer("polycompound", sh, _hex_to_rgb(sh.fill, current_color), bb))
    return layers


def _layer_sdf(layer: _CompiledLayer, vx: float, vy: float) -> float:
    kind = layer.kind
    sh = layer.shape
    if kind == "rect_fill":
        assert isinstance(sh, RoundedRect)
        return sh.sdf_fill(vx, vy)
    if kind == "rect_stroke":
        assert isinstance(sh, RoundedRect)
        return abs(sh.sdf_fill(vx, vy)) - 0.5 * sh.stroke_width
    if kind == "circle_fill":
        assert isinstance(sh, Circle)
        return sh.sdf_fill(vx, vy)
    if kind == "circle_stroke":
        assert isinstance(sh, Circle)
        return abs(sh.sdf_fill(vx, vy)) - 0.5 * sh.stroke_width
    if kind == "line":
        assert isinstance(sh, LineSegment)
        return sh.sdf_stroke(vx, vy)
    if kind == "poly":
        assert isinstance(sh, Polyline)
        return sh.sdf_stroke(vx, vy)
    if kind == "polygon":
        assert isinstance(sh, Polygon)
        return sh.sdf_fill(vx, vy)
    if kind == "polycompound":
        assert isinstance(sh, PolyCompound)
        return sh.sdf_fill(vx, vy)
    raise ValueError(kind)


def rasterize_rgba_4x4(
    shapes: Sequence[Shape],
    width: int,
    height: int,
    viewbox_w: float,
    viewbox_h: float,
    bg_rgb: Optional[Tuple[int, int, int]] = None,
) -> bytearray:
    """Rasterize shapes with 4x4 stratified supersampling into PNG scanlines (filter 0)."""
    layers = _compile_layers(shapes)
    sx = viewbox_w / width
    sy = viewbox_h / height
    scale = 0.5 * (sx + sy)
    edge_thresh = 0.78 * scale
    sub_offsets = (0.125, 0.375, 0.625, 0.875)
    top_down = list(reversed(layers))

    row_stride = 1 + width * 4
    buf = bytearray(height * row_stride)
    for py in range(height):
        vy_c = (py + 0.5) * sy
        py_lo = py * sy - sy
        py_hi = (py + 1) * sy + sy
        row_off = py * row_stride
        buf[row_off] = 0
        for px in range(width):
            vx_c = (px + 0.5) * sx
            px_lo = px * sx - sx
            px_hi = (px + 1) * sx + sx

            cand = [
                ly
                for ly in top_down
                if ly.bbox[2] >= px_lo and ly.bbox[0] <= px_hi and ly.bbox[3] >= py_lo and ly.bbox[1] <= py_hi
            ]
            p_off = row_off + 1 + px * 4
            if not cand:
                if bg_rgb is not None:
                    buf[p_off] = bg_rgb[0]
                    buf[p_off + 1] = bg_rgb[1]
                    buf[p_off + 2] = bg_rgb[2]
                    buf[p_off + 3] = 255
                continue

            center_hit: Optional[Tuple[int, int, int]] = None
            near_edge = False
            for ly in cand:
                d = _layer_sdf(ly, vx_c, vy_c)
                if abs(d) <= edge_thresh:
                    near_edge = True
                    break
                if d < -edge_thresh:
                    center_hit = ly.rgb
                    break

            if not near_edge:
                if center_hit is not None:
                    buf[p_off] = center_hit[0]
                    buf[p_off + 1] = center_hit[1]
                    buf[p_off + 2] = center_hit[2]
                    buf[p_off + 3] = 255
                elif bg_rgb is not None:
                    buf[p_off] = bg_rgb[0]
                    buf[p_off + 1] = bg_rgb[1]
                    buf[p_off + 2] = bg_rgb[2]
                    buf[p_off + 3] = 255
                continue

            acc_r = acc_g = acc_b = acc_a = 0.0
            for oy in sub_offsets:
                vy = (py + oy) * sy
                for ox in sub_offsets:
                    vx = (px + ox) * sx
                    hit = None
                    for ly in cand:
                        if _layer_sdf(ly, vx, vy) <= 0.0:
                            hit = ly.rgb
                            break
                    if hit is not None:
                        acc_r += hit[0]
                        acc_g += hit[1]
                        acc_b += hit[2]
                        acc_a += 1.0
                    elif bg_rgb is not None:
                        acc_r += bg_rgb[0]
                        acc_g += bg_rgb[1]
                        acc_b += bg_rgb[2]
                        acc_a += 1.0
            if acc_a > 0.0:
                if bg_rgb is not None:
                    buf[p_off] = _clamp_u8(acc_r / 16.0)
                    buf[p_off + 1] = _clamp_u8(acc_g / 16.0)
                    buf[p_off + 2] = _clamp_u8(acc_b / 16.0)
                    buf[p_off + 3] = 255
                else:
                    buf[p_off] = _clamp_u8(acc_r / acc_a)
                    buf[p_off + 1] = _clamp_u8(acc_g / acc_a)
                    buf[p_off + 2] = _clamp_u8(acc_b / acc_a)
                    buf[p_off + 3] = _clamp_u8((acc_a / 16.0) * 255.0)
    return buf


def _png_chunk(tag: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(tag + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", crc)


def encode_png_rgba(width: int, height: int, raw_scanlines: bytes) -> bytes:
    expected = height * (1 + width * 4)
    if len(raw_scanlines) != expected:
        raise ValueError(f"Expected {expected} scanline bytes for {width}x{height}, got {len(raw_scanlines)}")
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    idat = zlib.compress(raw_scanlines, 9)
    return PNG_MAGIC + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", idat) + _png_chunk(b"IEND", b"")


def validate_png_bytes(data: bytes, expected_w: int, expected_h: int) -> bytes:
    if not data.startswith(PNG_MAGIC):
        raise AssertionError("Missing PNG magic")
    pos = len(PNG_MAGIC)
    idat_parts: List[bytes] = []
    seen_ihdr = seen_iend = False
    while pos < len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        tag = data[pos + 4 : pos + 8]
        payload = data[pos + 8 : pos + 8 + length]
        crc_got = struct.unpack(">I", data[pos + 8 + length : pos + 12 + length])[0]
        crc_exp = zlib.crc32(tag + payload) & 0xFFFFFFFF
        if crc_got != crc_exp:
            raise AssertionError(f"Bad CRC on {tag!r}")
        if tag == b"IHDR":
            w, h, bd, ct, comp, flt, inter = struct.unpack(">IIBBBBB", payload)
            if (w, h, bd, ct, comp, flt, inter) != (expected_w, expected_h, 8, 6, 0, 0, 0):
                raise AssertionError(f"Unexpected IHDR {(w, h, bd, ct)}")
            seen_ihdr = True
        elif tag == b"IDAT":
            idat_parts.append(payload)
        elif tag == b"IEND":
            seen_iend = True
        pos += 12 + length
    if not (seen_ihdr and seen_iend):
        raise AssertionError("Missing IHDR or IEND")
    raw = zlib.decompress(b"".join(idat_parts))
    if len(raw) != expected_h * (1 + expected_w * 4):
        raise AssertionError("Decompressed scanline length mismatch")
    return raw


def _transform_shapes(shapes: Sequence[Shape], dx: float, dy: float, scale: float) -> List[Shape]:
    out: List[Shape] = []
    for sh in shapes:
        if isinstance(sh, RoundedRect):
            out.append(
                RoundedRect(
                    dx + sh.x * scale,
                    dy + sh.y * scale,
                    sh.w * scale,
                    sh.h * scale,
                    sh.rx * scale,
                    fill=sh.fill,
                    stroke=sh.stroke,
                    stroke_width=sh.stroke_width * scale,
                )
            )
        elif isinstance(sh, Circle):
            out.append(
                Circle(
                    dx + sh.cx * scale,
                    dy + sh.cy * scale,
                    sh.r * scale,
                    fill=sh.fill,
                    stroke=sh.stroke,
                    stroke_width=sh.stroke_width * scale,
                )
            )
        elif isinstance(sh, LineSegment):
            out.append(
                LineSegment(
                    dx + sh.x1 * scale,
                    dy + sh.y1 * scale,
                    dx + sh.x2 * scale,
                    dy + sh.y2 * scale,
                    stroke=sh.stroke,
                    stroke_width=sh.stroke_width * scale,
                )
            )
        elif isinstance(sh, Polyline):
            out.append(
                Polyline(
                    tuple((dx + px * scale, dy + py * scale) for px, py in sh.points),
                    stroke=sh.stroke,
                    stroke_width=sh.stroke_width * scale,
                    closed=sh.closed,
                )
            )
        elif isinstance(sh, Polygon):
            out.append(
                Polygon(
                    tuple((dx + px * scale, dy + py * scale) for px, py in sh.points),
                    fill=sh.fill,
                )
            )
    return out


def build_social_preview_png() -> bytes:
    """Render deterministic 1280x640 GitHub social preview PNG."""
    w, h = 1280, 640
    shapes: List[Shape] = [
        RoundedRect(24.0, 24.0, 1232.0, 592.0, 36.0, fill="#16202A", stroke="#B7410E", stroke_width=6.0),
        RoundedRect(34.0, 34.0, 1212.0, 572.0, 28.0, fill=None, stroke="#D95B26", stroke_width=2.0),
    ]
    # Left: Concept A FID-F badge (224x224)
    shapes.extend(_transform_shapes(build_canonical_icon_shapes(mono=False), 76.0, 118.0, 224.0 / 512.0))
    # Center: FERRIC geometric <path> wordmark
    shapes.extend(build_wordmark_shapes(344.0, 166.0, scale=1.55, fill="#F8FAFC"))
    shapes.append(RoundedRect(344.0, 310.0, 574.0, 8.0, 4.0, fill="#D95B26"))
    # Right: Concept C Larmor precession-cone + FID emblem (248x248)
    shapes.extend(_transform_shapes(build_larmor_emblem_shapes(), 954.0, 106.0, 248.0 / 512.0))
    # Lower panoramic damped FID trace
    banner_fid = sample_fid_wave(
        x_start=88.0,
        x_end=1192.0,
        y_base=492.0,
        amp=84.0,
        decay=2.8,
        cycles=5.5,
        phase=0.0,
        samples=240,
    )
    shapes.append(LineSegment(88.0, 492.0, 1192.0, 492.0, stroke="#475569", stroke_width=4.0))
    shapes.append(Polyline(banner_fid, stroke="#B7410E", stroke_width=14.0, closed=False))
    shapes.append(Polyline(banner_fid, stroke="#FB923C", stroke_width=7.0, closed=False))

    scans = rasterize_rgba_4x4(shapes, w, h, float(w), float(h), bg_rgb=DARK_BG_RGB)
    return encode_png_rgba(w, h, bytes(scans))


# ---------------------------------------------------------------------------
# Packager preview & diff helper (18.3:P2-2)
# ---------------------------------------------------------------------------


def dump_packager_icon_previews(out_dir: Path) -> List[Path]:
    """Import package_release.render_icon_scanlines and write 16/32/512px PNGs + diff."""
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import package_release  # type: ignore

    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    pkg_scans_512: Optional[bytes] = None
    for sz in (16, 32, 512):
        scans = package_release.render_icon_scanlines(sz)
        if sz == 512:
            pkg_scans_512 = scans
        png_bytes = package_release.encode_png_rgba(sz, scans)
        dest = out_dir / f"packager_icon_{sz}px.png"
        dest.write_bytes(png_bytes)
        written.append(dest)

    assert pkg_scans_512 is not None
    ref_scans_512 = bytes(
        rasterize_rgba_4x4(build_canonical_icon_shapes(mono=False), 512, 512, 512.0, 512.0, bg_rgb=None)
    )
    # Build 512x512 RGB diff (amplified 4x in red/cyan for easy visual inspection)
    stride = 1 + 512 * 4
    diff_buf = bytearray(512 * stride)
    for py in range(512):
        row_off = py * stride
        diff_buf[row_off] = 0
        for px in range(512):
            i = row_off + 1 + px * 4
            dr = abs(pkg_scans_512[i] - ref_scans_512[i])
            dg = abs(pkg_scans_512[i + 1] - ref_scans_512[i + 1])
            db = abs(pkg_scans_512[i + 2] - ref_scans_512[i + 2])
            da = abs(pkg_scans_512[i + 3] - ref_scans_512[i + 3])
            mag = min(255, (dr + dg + db + da))
            diff_buf[i] = min(255, mag * 2)
            diff_buf[i + 1] = mag // 2
            diff_buf[i + 2] = 24 if mag == 0 else min(255, mag)
            diff_buf[i + 3] = 255
    diff_path = out_dir / "packager_vs_concept_a_diff.png"
    diff_path.write_bytes(encode_png_rgba(512, 512, bytes(diff_buf)))
    written.append(diff_path)
    return written


# ---------------------------------------------------------------------------
# Artifact generation + self-test
# ---------------------------------------------------------------------------


def generate_all_brand_files() -> Tuple[Dict[str, bytes], Dict[str, Tuple[int, int, bytes]]]:
    svgs: Dict[str, bytes] = {
        "ferric-icon.svg": render_icon_svg(mono=False).encode("utf-8"),
        "ferric-icon-mono.svg": render_icon_svg(mono=True).encode("utf-8"),
        "ferric-wordmark.svg": render_wordmark_svg().encode("utf-8"),
        "ferric-logo-light.svg": render_lockup_svg(dark=False).encode("utf-8"),
        "ferric-logo-dark.svg": render_lockup_svg(dark=True).encode("utf-8"),
        "favicon.svg": render_icon_svg(mono=False).encode("utf-8"),
    }
    fav32_scans = rasterize_rgba_4x4(build_canonical_icon_shapes(mono=False), 32, 32, 512.0, 512.0, bg_rgb=None)
    pngs: Dict[str, Tuple[int, int, bytes]] = {
        "favicon-32.png": (32, 32, encode_png_rgba(32, 32, bytes(fav32_scans))),
        "ferric-social-preview.png": (1280, 640, build_social_preview_png()),
    }
    return svgs, pngs


def build_packager_icon_64_rgba() -> bytes:
    """Return raw 64x64 RGBA8 bytes directly from package_release.render_icon_scanlines(64)."""
    import sys

    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import package_release  # type: ignore

    scans = package_release.render_icon_scanlines(64)
    stride = 1 + 64 * 4
    if len(scans) != 64 * stride:
        raise AssertionError(f"Unexpected scanline buffer length: {len(scans)}")
    raw = b"".join(scans[y * stride + 1 : (y + 1) * stride] for y in range(64))
    if len(raw) != 64 * 64 * 4:
        raise AssertionError(f"Unexpected RGBA64 byte length: {len(raw)}")
    return raw


def run_self_test(brand_dir: Path) -> None:
    svgs_1, pngs_1 = generate_all_brand_files()
    svgs_2, pngs_2 = generate_all_brand_files()
    if svgs_1 != svgs_2 or pngs_1 != pngs_2:
        raise AssertionError("Non-deterministic brand output across two runs")

    svg_ns = "http://www.w3.org/2000/svg"
    url_re = re.compile(r"https?://|xlink:href|url\(#?(?!\s*$)", re.IGNORECASE)
    for fname, raw in svgs_1.items():
        text = raw.decode("utf-8")
        root = ET.fromstring(text)
        if root.tag != f"{{{svg_ns}}}svg":
            raise AssertionError(f"{fname}: unexpected root tag {root.tag}")
        for elem in root.iter():
            local = elem.tag.split("}")[-1].lower()
            if local in ("text", "tspan", "image", "script", "foreignobject"):
                raise AssertionError(f"{fname}: forbidden SVG tag <{local}>")
        stripped = text.replace('xmlns="http://www.w3.org/2000/svg"', "")
        if url_re.search(stripped):
            raise AssertionError(f"{fname}: unexpected external URL reference")
        if fname.endswith("-mono.svg") and ("currentColor" not in text or "#0F172A" in text):
            raise AssertionError(f"{fname}: mono SVG must use currentColor")

    for fname, (w, h, data) in pngs_1.items():
        validate_png_bytes(data, w, h)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for fname, raw in svgs_1.items():
            (tmp_path / fname).write_bytes(raw)
            assert (tmp_path / fname).read_bytes() == svgs_2[fname]

    for fname, raw in svgs_1.items():
        disk = brand_dir / fname
        if disk.exists() and disk.read_bytes() != raw:
            raise AssertionError(f"Stale SVG on disk: {disk}")
    for fname, (_, _, data) in pngs_1.items():
        disk = brand_dir / fname
        if disk.exists() and disk.read_bytes() != data:
            raise AssertionError(f"Stale PNG on disk: {disk}")

    rgba64 = build_packager_icon_64_rgba()
    rgba64_disk = brand_dir / "ferric-icon-64.rgba"
    if not rgba64_disk.exists():
        raise AssertionError(f"Missing 64x64 RGBA asset: {rgba64_disk}")
    if rgba64_disk.read_bytes() != rgba64:
        raise AssertionError(f"Stale 64x64 RGBA asset on disk: {rgba64_disk}")

    print(
        f"SELF-TEST PASS: {len(svgs_1)} SVGs + {len(pngs_1)} PNGs + ferric-icon-64.rgba verified "
        f"(byte-identical across 2 runs, zero <text>/external refs, valid PNG CRCs, 32x32 & 1280x640)"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate canonical FERRIC brand assets.")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/assets/brand"))
    parser.add_argument("--dump-icon-previews", type=Path, default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    svgs, pngs = generate_all_brand_files()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for fname, raw in svgs.items():
        p = args.out_dir / fname
        p.write_bytes(raw)
        written.append(p)
    for fname, (_, _, data) in pngs.items():
        p = args.out_dir / fname
        p.write_bytes(data)
        written.append(p)
    rgba64_path = args.out_dir / "ferric-icon-64.rgba"
    rgba64_path.write_bytes(build_packager_icon_64_rgba())
    written.append(rgba64_path)

    if args.dump_icon_previews is not None:
        written.extend(dump_packager_icon_previews(args.dump_icon_previews))

    for p in sorted(written):
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        print(f"{digest}  {p.as_posix()}")

    if args.self_test:
        run_self_test(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
