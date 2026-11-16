#!/usr/bin/env python3
"""Deterministic SVG generator and 4x4-supersampled PNG previewer for FERRIC brand concepts.

This script has zero third-party dependencies (Python 3 stdlib only). It defines
the exact vector geometry for the three FERRIC brand concepts (plus monochrome
variants) in ``docs/assets/brand/concepts/`` and deterministically rasterizes
4x4-supersampled anti-aliased RGBA PNG previews at 512x512, 64x64, and 16x16 px
on both light (#F8FAFC) and dark (#0B1120) backgrounds, plus a 16x16 8x pixel
loupe sheet and a composite comparison sheet.

Usage:
    python3 scripts/render_ferric_brand_concepts.py --out-dir docs/assets/brand/concepts/previews --self-test
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import re
import struct
import tempfile
from typing import Dict, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET
import zlib

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
VIEWBOX_SIZE = 512.0

LIGHT_BG_HEX = "#F8FAFC"
DARK_BG_HEX = "#0B1120"

LIGHT_BG_RGB = (248, 250, 252)
DARK_BG_RGB = (11, 17, 32)
MONO_INK_HEX = "currentColor"
MONO_LIGHT_RGB = (15, 23, 42)
MONO_DARK_RGB = (248, 250, 252)

PREVIEW_SIZES: Tuple[int, ...] = (512, 64, 16)


def _hex_to_rgb(color: str, current_color_rgb: Tuple[int, int, int] = MONO_LIGHT_RGB) -> Tuple[int, int, int]:
    if color == "currentColor":
        return current_color_rgb
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


# ---------------------------------------------------------------------------
# Shared vector primitives (single source of truth for SVG + 4x4 rasterizer)
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
        if h < 0.0:
            h = 0.0
        elif h > 1.0:
            h = 1.0
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
        pts = self.points
        n = len(pts)
        min_d2 = float("inf")
        limit = n if self.closed else n - 1
        for i in range(limit):
            ax, ay = pts[i]
            bx, by = pts[(i + 1) % n]
            bax = bx - ax
            bay = by - ay
            denom = bax * bax + bay * bay
            if denom <= 1e-12:
                dx = px - ax
                dy = py - ay
            else:
                h = ((px - ax) * bax + (py - ay) * bay) / denom
                if h < 0.0:
                    h = 0.0
                elif h > 1.0:
                    h = 1.0
                dx = (px - ax) - bax * h
                dy = (py - ay) - bay * h
            d2 = dx * dx + dy * dy
            if d2 < min_d2:
                min_d2 = d2
        return math.sqrt(min_d2) - 0.5 * self.stroke_width


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
        min_d2 = float("inf")
        j = n - 1
        for i in range(n):
            xi, yi = pts[i]
            xj, yj = pts[j]
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-30) + xi):
                inside = not inside
            bax = xj - xi
            bay = yj - yi
            denom = bax * bax + bay * bay
            if denom <= 1e-12:
                dx, dy = px - xi, py - yi
            else:
                h = ((px - xi) * bax + (py - yi) * bay) / denom
                h = 0.0 if h < 0.0 else (1.0 if h > 1.0 else h)
                dx = (px - xi) - bax * h
                dy = (py - yi) - bay * h
            d2 = dx * dx + dy * dy
            if d2 < min_d2:
                min_d2 = d2
            j = i
        d = math.sqrt(min_d2)
        return -d if inside else d


Shape = RoundedRect | Circle | LineSegment | Polyline | Polygon


def _sample_fid_wave(
    x_start: float,
    x_end: float,
    y_base: float,
    amp: float,
    decay: float,
    cycles: float,
    phase: float = 0.0,
    samples: int = 128,
) -> Tuple[Tuple[float, float], ...]:
    pts: List[Tuple[float, float]] = []
    for i in range(samples + 1):
        t = i / samples
        x = x_start + (x_end - x_start) * t
        y = y_base - amp * math.exp(-decay * t) * math.cos(2.0 * math.pi * cycles * t + phase)
        pts.append((round(x, 2), round(y, 2)))
    return tuple(pts)


def _sample_ellipse(cx: float, cy: float, rx: float, ry: float, samples: int = 96) -> Tuple[Tuple[float, float], ...]:
    pts: List[Tuple[float, float]] = []
    for i in range(samples):
        ang = 2.0 * math.pi * i / samples
        pts.append((round(cx + rx * math.cos(ang), 2), round(cy + ry * math.sin(ang), 2)))
    return tuple(pts)


def _arrow_triangle(
    tip: Tuple[float, float],
    from_pt: Tuple[float, float],
    length: float,
    half_width: float,
) -> Tuple[Tuple[float, float], ...]:
    dx = tip[0] - from_pt[0]
    dy = tip[1] - from_pt[1]
    norm = math.hypot(dx, dy)
    ux, uy = dx / norm, dy / norm
    nx, ny = -uy, ux
    bx = tip[0] - length * ux
    by = tip[1] - length * uy
    return (
        (round(tip[0], 2), round(tip[1], 2)),
        (round(bx + half_width * nx, 2), round(by + half_width * ny, 2)),
        (round(bx - half_width * nx, 2), round(by - half_width * ny, 2)),
    )


# ---------------------------------------------------------------------------
# Three concept definitions (+ monochrome/outline variants)
# ---------------------------------------------------------------------------


def build_concept_a(mono: bool = False) -> Tuple[Shape, ...]:
    """Concept A: Stylized 'F' sculpted by a decaying FID envelope on a squircle badge."""
    fid_pts = _sample_fid_wave(
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
        return (
            RoundedRect(32.0, 32.0, 448.0, 448.0, 104.0, fill=None, stroke=MONO_INK_HEX, stroke_width=24.0),
            LineSegment(152.0, 124.0, 152.0, 388.0, stroke=MONO_INK_HEX, stroke_width=52.0),
            LineSegment(152.0, 124.0, 376.0, 124.0, stroke=MONO_INK_HEX, stroke_width=52.0),
            LineSegment(152.0, 248.0, 304.0, 248.0, stroke=MONO_INK_HEX, stroke_width=44.0),
            Polyline(fid_pts, stroke=MONO_INK_HEX, stroke_width=24.0, closed=False),
        )
    return (
        RoundedRect(32.0, 32.0, 448.0, 448.0, 104.0, fill="#1C2430", stroke="#64748B", stroke_width=16.0),
        # Subtle FID equilibrium axis across right field
        LineSegment(174.0, 248.0, 416.0, 248.0, stroke="#475569", stroke_width=10.0),
        # Iron-oxide outer "F" silhouette bars (t=0 excitation spine + coherence bars)
        LineSegment(152.0, 124.0, 152.0, 388.0, stroke="#B7410E", stroke_width=56.0),
        LineSegment(152.0, 124.0, 376.0, 124.0, stroke="#B7410E", stroke_width=56.0),
        LineSegment(152.0, 248.0, 316.0, 248.0, stroke="#B7410E", stroke_width=48.0),
        # Warm ember inner "F" spine & bars
        LineSegment(152.0, 124.0, 152.0, 388.0, stroke="#E05A2B", stroke_width=34.0),
        LineSegment(152.0, 124.0, 376.0, 124.0, stroke="#E05A2B", stroke_width=34.0),
        LineSegment(152.0, 248.0, 316.0, 248.0, stroke="#E05A2B", stroke_width=28.0),
        # Bright exponentially damped FID wave sculpting the upper/middle "F" envelope
        Polyline(fid_pts, stroke="#FB923C", stroke_width=28.0, closed=False),
        # Crisp t=0 excitation cap dot
        Circle(152.0, 124.0, 16.0, fill="#FED7AA"),
    )


def build_concept_b(mono: bool = False) -> Tuple[Shape, ...]:
    """Concept B: Periodic-table square tile with 45-deg spin vector M feeding a 4-lobe damped FID."""
    fid_pts = _sample_fid_wave(
        x_start=144.0,
        x_end=424.0,
        y_base=344.0,
        amp=78.0,
        decay=1.9,
        cycles=2.0,
        phase=0.0,
        samples=140,
    )
    precession_ring = _sample_ellipse(226.0, 184.0, 56.0, 20.0, samples=80)
    arrow_head = _arrow_triangle((306.0, 104.0), (144.0, 266.0), length=64.0, half_width=38.0)

    if mono:
        return (
            RoundedRect(32.0, 32.0, 448.0, 448.0, 40.0, fill=None, stroke=MONO_INK_HEX, stroke_width=28.0),
            LineSegment(84.0, 86.0, 146.0, 86.0, stroke=MONO_INK_HEX, stroke_width=20.0),
            Circle(178.0, 86.0, 10.0, fill=MONO_INK_HEX),
            Polyline(precession_ring, stroke=MONO_INK_HEX, stroke_width=10.0, closed=True),
            LineSegment(144.0, 266.0, 266.0, 144.0, stroke=MONO_INK_HEX, stroke_width=32.0),
            Polygon(arrow_head, fill=MONO_INK_HEX),
            Polyline(fid_pts, stroke=MONO_INK_HEX, stroke_width=28.0, closed=False),
            Circle(144.0, 266.0, 22.0, fill=MONO_INK_HEX),
        )
    return (
        RoundedRect(32.0, 32.0, 448.0, 448.0, 40.0, fill="#1E293B", stroke="#B7410E", stroke_width=30.0),
        # Periodic-tile geometric index motif (top-left, zero text)
        LineSegment(84.0, 86.0, 146.0, 86.0, stroke="#E05A2B", stroke_width=20.0),
        Circle(178.0, 86.0, 10.0, fill="#94A3B8"),
        # FID zero baseline across lower half + equilibrium reference tick
        LineSegment(84.0, 344.0, 428.0, 344.0, stroke="#475569", stroke_width=10.0),
        LineSegment(88.0, 266.0, 232.0, 266.0, stroke="#334155", stroke_width=10.0),
        # Subtle precession orbit ellipse around spin vector shaft
        Polyline(precession_ring, stroke="#64748B", stroke_width=12.0, closed=True),
        # 4-lobe exponentially damped FID fed directly from spin origin (144, 266)
        Polyline(fid_pts, stroke="#FB923C", stroke_width=34.0, closed=False),
        # Bold 45-degree magnetic moment spin vector M
        LineSegment(144.0, 266.0, 266.0, 144.0, stroke="#F8FAFC", stroke_width=34.0),
        Polygon(arrow_head, fill="#F8FAFC"),
        # Coupling pivot at t=0
        Circle(144.0, 266.0, 24.0, fill="#E05A2B"),
        Circle(144.0, 266.0, 10.0, fill="#F8FAFC"),
    )


def build_concept_c(mono: bool = False) -> Tuple[Shape, ...]:
    """Concept C: Iron-oxide disc with Larmor precession cone, M tip, and equatorial damped FID."""
    orbit_pts = _sample_ellipse(256.0, 158.0, 112.0, 38.0, samples=96)
    fid_pts = _sample_fid_wave(
        x_start=104.0,
        x_end=408.0,
        y_base=358.0,
        amp=48.0,
        decay=2.2,
        cycles=2.5,
        phase=-0.5 * math.pi,
        samples=120,
    )
    arrow_head = _arrow_triangle((344.0, 174.0), (256.0, 300.0), length=52.0, half_width=30.0)

    if mono:
        return (
            Circle(256.0, 256.0, 214.0, fill=None, stroke=MONO_INK_HEX, stroke_width=28.0),
            LineSegment(256.0, 300.0, 256.0, 102.0, stroke=MONO_INK_HEX, stroke_width=10.0),
            LineSegment(256.0, 300.0, 144.0, 158.0, stroke=MONO_INK_HEX, stroke_width=12.0),
            LineSegment(256.0, 300.0, 368.0, 158.0, stroke=MONO_INK_HEX, stroke_width=12.0),
            Polyline(orbit_pts, stroke=MONO_INK_HEX, stroke_width=18.0, closed=True),
            LineSegment(256.0, 300.0, 318.0, 211.0, stroke=MONO_INK_HEX, stroke_width=24.0),
            Polygon(arrow_head, fill=MONO_INK_HEX),
            Circle(346.0, 171.0, 18.0, fill=MONO_INK_HEX),
            Polyline(fid_pts, stroke=MONO_INK_HEX, stroke_width=24.0, closed=False),
        )
    return (
        # Deep steel disc with smooth dual-tone iron-oxide ring (zero gear teeth)
        Circle(256.0, 256.0, 224.0, fill="#1B2432"),
        Circle(256.0, 256.0, 212.0, fill=None, stroke="#B7410E", stroke_width=28.0),
        Circle(256.0, 256.0, 198.0, fill=None, stroke="#E05A2B", stroke_width=8.0),
        # B0 static field axis & Larmor precession cone boundaries
        LineSegment(256.0, 300.0, 256.0, 100.0, stroke="#64748B", stroke_width=12.0),
        LineSegment(256.0, 300.0, 144.0, 158.0, stroke="#64748B", stroke_width=14.0),
        LineSegment(256.0, 300.0, 368.0, 158.0, stroke="#64748B", stroke_width=14.0),
        # Larmor precession ellipse
        Polyline(orbit_pts, stroke="#F97316", stroke_width=24.0, closed=True),
        # Equatorial damped FID baseline & trace
        LineSegment(104.0, 358.0, 408.0, 358.0, stroke="#475569", stroke_width=8.0),
        Polyline(fid_pts, stroke="#FB923C", stroke_width=28.0, closed=False),
        # Precessing magnetization vector M + bright tip
        LineSegment(256.0, 300.0, 318.0, 211.0, stroke="#F8FAFC", stroke_width=28.0),
        Polygon(arrow_head, fill="#F8FAFC"),
        Circle(256.0, 300.0, 14.0, fill="#94A3B8"),
        Circle(346.0, 171.0, 20.0, fill="#FB923C"),
        Circle(346.0, 171.0, 9.0, fill="#FFF7ED"),
    )


CONCEPT_SPECS: Tuple[Tuple[str, str, str, bool], ...] = (
    ("a", "concept-a-fid-f", "Concept A: FID-F Squircle", False),
    ("a-mono", "concept-a-fid-f-mono", "Concept A (Mono)", True),
    ("b", "concept-b-spin-tile", "Concept B: Spin Tile", False),
    ("b-mono", "concept-b-spin-tile-mono", "Concept B (Mono)", True),
    ("c", "concept-c-larmor-disc", "Concept C: Larmor Disc", False),
    ("c-mono", "concept-c-larmor-disc-mono", "Concept C (Mono)", True),
)


def get_concept_shapes(key: str) -> Tuple[Shape, ...]:
    if key == "a":
        return build_concept_a(mono=False)
    if key == "a-mono":
        return build_concept_a(mono=True)
    if key == "b":
        return build_concept_b(mono=False)
    if key == "b-mono":
        return build_concept_b(mono=True)
    if key == "c":
        return build_concept_c(mono=False)
    if key == "c-mono":
        return build_concept_c(mono=True)
    raise KeyError(f"Unknown concept key: {key!r}")


# ---------------------------------------------------------------------------
# Deterministic SVG rendering
# ---------------------------------------------------------------------------


def _fmt(val: float) -> str:
    if abs(val - round(val)) < 1e-6:
        return str(int(round(val)))
    return f"{val:.2f}".rstrip("0").rstrip(".")


def render_concept_svg(shapes: Sequence[Shape], title: str) -> str:
    """Render pure-geometric vector SVG (zero <text>/<tspan>, zero external URLs)."""
    lines: List[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512" role="img">',
        f"  <title>{title}</title>",
    ]
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
            lines.append(f'  <rect {" ".join(attrs)}/>')
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
            lines.append(f'  <circle {" ".join(attrs)}/>')
        elif isinstance(shape, LineSegment):
            lines.append(
                f'  <line x1="{_fmt(shape.x1)}" y1="{_fmt(shape.y1)}" '
                f'x2="{_fmt(shape.x2)}" y2="{_fmt(shape.y2)}" '
                f'stroke="{shape.stroke}" stroke-width="{_fmt(shape.stroke_width)}" '
                f'stroke-linecap="round"/>'
            )
        elif isinstance(shape, Polyline):
            cmd_parts = [f"M {_fmt(shape.points[0][0])} {_fmt(shape.points[0][1])}"]
            for px, py in shape.points[1:]:
                cmd_parts.append(f"L {_fmt(px)} {_fmt(py)}")
            if shape.closed:
                cmd_parts.append("Z")
            d_str = " ".join(cmd_parts)
            lines.append(
                f'  <path d="{d_str}" fill="none" stroke="{shape.stroke}" '
                f'stroke-width="{_fmt(shape.stroke_width)}" stroke-linecap="round" stroke-linejoin="round"/>'
            )
        elif isinstance(shape, Polygon):
            pts_str = " ".join(f"{_fmt(px)},{_fmt(py)}" for px, py in shape.points)
            lines.append(f'  <polygon points="{pts_str}" fill="{shape.fill}"/>')
        else:
            raise TypeError(f"Unsupported shape: {type(shape)!r}")
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 4x4 supersampled anti-aliased scanline rasterizer + RFC 2083 PNG encoder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CompiledLayer:
    kind: str  # "rect_fill", "rect_stroke", "circle_fill", "circle_stroke", "line", "poly", "polygon"
    shape: Shape
    rgb: Tuple[int, int, int]
    bbox: Tuple[float, float, float, float]


def _compile_layers(
    shapes: Sequence[Shape],
    current_color_rgb: Tuple[int, int, int] = MONO_LIGHT_RGB,
) -> List[_CompiledLayer]:
    layers: List[_CompiledLayer] = []
    for sh in shapes:
        bb = sh.bbox()
        if isinstance(sh, RoundedRect):
            if sh.fill is not None:
                layers.append(_CompiledLayer("rect_fill", sh, _hex_to_rgb(sh.fill, current_color_rgb), bb))
            if sh.stroke is not None and sh.stroke_width > 0.0:
                layers.append(_CompiledLayer("rect_stroke", sh, _hex_to_rgb(sh.stroke, current_color_rgb), bb))
        elif isinstance(sh, Circle):
            if sh.fill is not None:
                layers.append(_CompiledLayer("circle_fill", sh, _hex_to_rgb(sh.fill, current_color_rgb), bb))
            if sh.stroke is not None and sh.stroke_width > 0.0:
                layers.append(_CompiledLayer("circle_stroke", sh, _hex_to_rgb(sh.stroke, current_color_rgb), bb))
        elif isinstance(sh, LineSegment):
            layers.append(_CompiledLayer("line", sh, _hex_to_rgb(sh.stroke, current_color_rgb), bb))
        elif isinstance(sh, Polyline):
            layers.append(_CompiledLayer("poly", sh, _hex_to_rgb(sh.stroke, current_color_rgb), bb))
        elif isinstance(sh, Polygon):
            layers.append(_CompiledLayer("polygon", sh, _hex_to_rgb(sh.fill, current_color_rgb), bb))
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
    raise ValueError(kind)


def rasterize_rgba_4x4(
    shapes: Sequence[Shape],
    size: int,
    current_color_rgb: Tuple[int, int, int] = MONO_LIGHT_RGB,
) -> bytearray:
    """Rasterize shapes on transparent background with exact 4x4 stratified supersampling."""
    if size <= 0 or size > 2048:
        raise ValueError(f"Unsupported dimension: {size}")
    layers = _compile_layers(shapes, current_color_rgb=current_color_rgb)
    scale = VIEWBOX_SIZE / size
    edge_thresh = 0.78 * scale
    sub_offsets = (0.125, 0.375, 0.625, 0.875)
    top_down_layers = list(reversed(layers))

    buf = bytearray(size * size * 4)
    for py in range(size):
        vy_c = (py + 0.5) * scale
        for px in range(size):
            vx_c = (px + 0.5) * scale
            # Collect only layers whose bbox intersects this pixel square (with 1px margin)
            px_lo = px * scale - scale
            px_hi = (px + 1) * scale + scale
            py_lo = py * scale - scale
            py_hi = (py + 1) * scale + scale

            active: List[_CompiledLayer] = []
            need_ss = False
            first_solid_rgb: Optional[Tuple[int, int, int]] = None

            for lyr in top_down_layers:
                bx0, by0, bx1, by1 = lyr.bbox
                if px_hi < bx0 or px_lo > bx1 or py_hi < by0 or py_lo > by1:
                    continue
                d_c = _layer_sdf(lyr, vx_c, vy_c)
                if abs(d_c) <= edge_thresh:
                    need_ss = True
                    active.append(lyr)
                elif d_c < -edge_thresh:
                    active.append(lyr)
                    if not need_ss:
                        first_solid_rgb = lyr.rgb
                    break

            off = (py * size + px) * 4
            if not active:
                continue

            if not need_ss and first_solid_rgb is not None:
                buf[off] = first_solid_rgb[0]
                buf[off + 1] = first_solid_rgb[1]
                buf[off + 2] = first_solid_rgb[2]
                buf[off + 3] = 255
                continue

            # Exact 4x4 stratified supersampling over active layers
            sum_r = 0
            sum_g = 0
            sum_b = 0
            hits = 0
            for sy in sub_offsets:
                vy = (py + sy) * scale
                for sx in sub_offsets:
                    vx = (px + sx) * scale
                    for lyr in active:
                        if _layer_sdf(lyr, vx, vy) <= 0.0:
                            r, g, b = lyr.rgb
                            sum_r += r
                            sum_g += g
                            sum_b += b
                            hits += 1
                            break

            if hits > 0:
                buf[off] = _clamp_u8(sum_r / hits)
                buf[off + 1] = _clamp_u8(sum_g / hits)
                buf[off + 2] = _clamp_u8(sum_b / hits)
                buf[off + 3] = _clamp_u8(hits * 255.0 / 16.0)

    return buf


def composite_over_bg(rgba: bytes, size: int, bg_rgb: Tuple[int, int, int]) -> bytes:
    """Composite RGBA buffer onto an opaque solid background and return PNG scanlines."""
    row_stride = 1 + size * 4
    out = bytearray(size * row_stride)
    br, bg, bb = bg_rgb
    for py in range(size):
        row_off = py * row_stride
        out[row_off] = 0
        src_row = py * size * 4
        dst = row_off + 1
        for px in range(size):
            si = src_row + px * 4
            a = rgba[si + 3]
            if a == 255:
                out[dst] = rgba[si]
                out[dst + 1] = rgba[si + 1]
                out[dst + 2] = rgba[si + 2]
                out[dst + 3] = 255
            elif a == 0:
                out[dst] = br
                out[dst + 1] = bg
                out[dst + 2] = bb
                out[dst + 3] = 255
            else:
                fa = a / 255.0
                inv = 1.0 - fa
                out[dst] = _clamp_u8(rgba[si] * fa + br * inv)
                out[dst + 1] = _clamp_u8(rgba[si + 1] * fa + bg * inv)
                out[dst + 2] = _clamp_u8(rgba[si + 2] * fa + bb * inv)
                out[dst + 3] = 255
            dst += 4
    return bytes(out)


def _png_chunk(tag: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(tag + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", crc)


def encode_png_rgba(width: int, height: int, raw_scanlines: bytes) -> bytes:
    """Encode RGBA scanlines (filter byte 0 per row) into an RFC 2083 PNG."""
    expected = height * (1 + width * 4)
    if len(raw_scanlines) != expected:
        raise ValueError(f"Expected {expected} scanline bytes for {width}x{height}, got {len(raw_scanlines)}")
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    idat = zlib.compress(raw_scanlines, 9)
    return PNG_MAGIC + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", idat) + _png_chunk(b"IEND", b"")


def validate_png_bytes(data: bytes, expected_w: int, expected_h: int) -> bytes:
    """Validate RFC 2083 PNG header, chunk CRCs, dimensions, and return raw scanlines."""
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
            if (w, h) != (expected_w, expected_h):
                raise ValueError(f"PNG size {(w, h)} != expected {(expected_w, expected_h)}")
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
    if len(raw) != expected_h * (1 + expected_w * 4):
        raise ValueError("Decompressed PNG scanline buffer size mismatch")
    return raw


# ---------------------------------------------------------------------------
# Minimal 5x7 bitmap font for comparison sheet labels ONLY (never in SVGs)
# ---------------------------------------------------------------------------

_FONT_5X7: Dict[str, Tuple[str, ...]] = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01110", "10001", "10000", "10000", "10000", "10001", "01110"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01110", "10001", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("01110", "00100", "00100", "00100", "00100", "00100", "01110"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "0": ("01110", "10011", "10101", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00110", "01000", "10000", "11111"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "11110", "00001", "00001", "10001", "01110"),
    "6": ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    ":": ("00000", "00100", "00100", "00000", "00100", "00100", "00000"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    "+": ("00000", "00100", "00100", "11111", "00100", "00100", "00000"),
    "/": ("00001", "00010", "00010", "00100", "01000", "01000", "10000"),
    "(": ("00010", "00100", "01000", "01000", "01000", "00100", "00010"),
    ")": ("01000", "00100", "00010", "00010", "00010", "00100", "01000"),
    "=": ("00000", "00000", "11111", "00000", "11111", "00000", "00000"),
    "#": ("01010", "01010", "11111", "01010", "11111", "01010", "01010"),
    " ": ("00000", "00000", "00000", "00000", "00000", "00000", "00000"),
}


class _Canvas:
    def __init__(self, width: int, height: int, fill_rgb: Tuple[int, int, int]) -> None:
        self.w = width
        self.h = height
        self.stride = 1 + width * 4
        self.buf = bytearray(height * self.stride)
        r, g, b = fill_rgb
        for y in range(height):
            row_off = y * self.stride
            self.buf[row_off] = 0
            p = row_off + 1
            for _ in range(width):
                self.buf[p] = r
                self.buf[p + 1] = g
                self.buf[p + 2] = b
                self.buf[p + 3] = 255
                p += 4

    def fill_rect(self, x: int, y: int, w: int, h: int, rgb: Tuple[int, int, int]) -> None:
        r, g, b = rgb
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(self.w, x + w)
        y1 = min(self.h, y + h)
        for py in range(y0, y1):
            p = py * self.stride + 1 + x0 * 4
            for _ in range(x0, x1):
                self.buf[p] = r
                self.buf[p + 1] = g
                self.buf[p + 2] = b
                self.buf[p + 3] = 255
                p += 4

    def blit_scanlines(self, x: int, y: int, src_w: int, src_h: int, scanlines: bytes) -> None:
        src_stride = 1 + src_w * 4
        for sy in range(src_h):
            dy = y + sy
            if dy < 0 or dy >= self.h:
                continue
            s_off = sy * src_stride + 1
            d_off = dy * self.stride + 1 + x * 4
            self.buf[d_off : d_off + src_w * 4] = scanlines[s_off : s_off + src_w * 4]

    def blit_scaled_nn(
        self,
        x: int,
        y: int,
        src_w: int,
        src_h: int,
        scanlines: bytes,
        scale: int,
        grid_rgb: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        src_stride = 1 + src_w * 4
        for sy in range(src_h):
            for sx in range(src_w):
                si = sy * src_stride + 1 + sx * 4
                r, g, b = scanlines[si], scanlines[si + 1], scanlines[si + 2]
                for oy in range(scale):
                    dy = y + sy * scale + oy
                    if dy < 0 or dy >= self.h:
                        continue
                    for ox in range(scale):
                        dx = x + sx * scale + ox
                        if dx < 0 or dx >= self.w:
                            continue
                        is_grid = grid_rgb is not None and (ox == 0 or oy == 0)
                        pr, pg, pb = grid_rgb if is_grid else (r, g, b)
                        di = dy * self.stride + 1 + dx * 4
                        self.buf[di] = pr
                        self.buf[di + 1] = pg
                        self.buf[di + 2] = pb
                        self.buf[di + 3] = 255

    def draw_text(self, x: int, y: int, text: str, rgb: Tuple[int, int, int], scale: int = 2) -> None:
        cx = x
        for ch in text.upper():
            glyph = _FONT_5X7.get(ch, _FONT_5X7[" "])
            for gy, row in enumerate(glyph):
                for gx, bit in enumerate(row):
                    if bit == "1":
                        self.fill_rect(cx + gx * scale, y + gy * scale, scale, scale, rgb)
            cx += 6 * scale


def build_loupe_sheet(scanlines_16: Dict[Tuple[str, str], bytes]) -> Tuple[int, int, bytes]:
    """Build 8x pixel-grid loupe (128x128 tiles) for Concept A/B/C at 16x16 light & dark."""
    tile = 128
    pad = 24
    header_h = 54
    row_lbl_w = 110
    cols = ("a", "b", "c")
    rows = (("light", "LIGHT"), ("dark", "DARK"))
    width = row_lbl_w + len(cols) * tile + (len(cols) + 1) * pad
    height = header_h + len(rows) * (tile + 18) + (len(rows) + 1) * pad

    canvas = _Canvas(width, height, (226, 232, 240))
    canvas.fill_rect(0, 0, width, header_h, (15, 23, 42))
    canvas.draw_text(20, 18, "FERRIC 16X16 PIXEL LOUPE (8X = 128X128)", (248, 250, 252), scale=2)

    titles = {"a": "CONCEPT A", "b": "CONCEPT B", "c": "CONCEPT C"}
    for ci, ck in enumerate(cols):
        tx = row_lbl_w + pad + ci * (tile + pad)
        canvas.draw_text(tx + 8, header_h + 6, titles[ck], (15, 23, 42), scale=2)

    for ri, (rk, rlabel) in enumerate(rows):
        ry = header_h + pad + 18 + ri * (tile + pad + 18)
        canvas.draw_text(18, ry + 54, rlabel, (15, 23, 42), scale=2)
        bg = LIGHT_BG_RGB if rk == "light" else DARK_BG_RGB
        grid = (226, 232, 240) if rk == "light" else (30, 41, 59)
        for ci, ck in enumerate(cols):
            tx = row_lbl_w + pad + ci * (tile + pad)
            canvas.fill_rect(tx - 2, ry - 2, tile + 4, tile + 4, (100, 116, 139))
            canvas.fill_rect(tx, ry, tile, tile, bg)
            canvas.blit_scaled_nn(tx, ry, 16, 16, scanlines_16[(ck, rk)], scale=8, grid_rgb=grid)

    return width, height, bytes(canvas.buf)


def build_comparison_sheet(
    scanlines: Dict[Tuple[str, int, str], bytes],
) -> Tuple[int, int, bytes]:
    """Build side-by-side composite sheet showing A/B/C at 512, 64, 16, and 16x8 loupe on light & dark."""
    col_w = 290
    row_lbl_w = 160
    pad = 24
    header_h = 64
    sub_h = 34
    box_h = 264  # Fits 256px downscaled 512 + padding, or 64 + 16 + 128 loupe

    width = row_lbl_w + 3 * col_w + 4 * pad
    height = header_h + sub_h + 6 * (box_h + pad) + pad
    canvas = _Canvas(width, height, (241, 245, 249))
    canvas.fill_rect(0, 0, width, header_h, (15, 23, 42))
    canvas.draw_text(24, 22, "FERRIC BRAND CONCEPTS - 512 / 64 / 16 PX + 8X LOUPE", (248, 250, 252), scale=2)

    col_names = (("a", "A: FID-F"), ("b", "B: SPIN TILE"), ("c", "C: LARMOR DISC"))
    for ci, (_, cname) in enumerate(col_names):
        cx = row_lbl_w + pad + ci * (col_w + pad)
        canvas.draw_text(cx + 32, header_h + 10, cname, (15, 23, 42), scale=2)

    def _downsample_half(src_w: int, src: bytes) -> bytes:
        dst_w = src_w // 2
        s_stride = 1 + src_w * 4
        d_stride = 1 + dst_w * 4
        out = bytearray(dst_w * d_stride)
        for y in range(dst_w):
            out[y * d_stride] = 0
            for x in range(dst_w):
                r = g = b = 0
                for dy in (0, 1):
                    for dx in (0, 1):
                        si = (y * 2 + dy) * s_stride + 1 + (x * 2 + dx) * 4
                        r += src[si]
                        g += src[si + 1]
                        b += src[si + 2]
                di = y * d_stride + 1 + x * 4
                out[di] = r // 4
                out[di + 1] = g // 4
                out[di + 2] = b // 4
                out[di + 3] = 255
        return bytes(out)

    row_specs = (
        ("512PX LIGHT", "light", "hero"),
        ("512PX DARK", "dark", "hero"),
        ("64/16/LOUPE LIGHT", "light", "small"),
        ("64/16/LOUPE DARK", "dark", "small"),
        ("MONO LIGHT", "light", "mono"),
        ("MONO DARK", "dark", "mono"),
    )

    for ri, (rtitle, bg_key, mode) in enumerate(row_specs):
        ry = header_h + sub_h + pad + ri * (box_h + pad)
        parts = rtitle.split(" ")
        canvas.draw_text(16, ry + 110, parts[0], (15, 23, 42), scale=2)
        if len(parts) > 1:
            canvas.draw_text(16, ry + 132, parts[1], (71, 85, 105), scale=2)
        bg = LIGHT_BG_RGB if bg_key == "light" else DARK_BG_RGB
        grid = (226, 232, 240) if bg_key == "light" else (30, 41, 59)

        for ci, (ck, _) in enumerate(col_names):
            cx = row_lbl_w + pad + ci * (col_w + pad)
            canvas.fill_rect(cx - 1, ry - 1, col_w + 2, box_h + 2, (148, 163, 184))
            canvas.fill_rect(cx, ry, col_w, box_h, bg)
            if mode == "hero":
                # 512px preview scaled cleanly 2x2 -> 256x256 inside card
                half = _downsample_half(512, scanlines[(ck, 512, bg_key)])
                canvas.blit_scanlines(cx + (col_w - 256) // 2, ry + 4, 256, 256, half)
            else:
                # 64px native + 16px native + 128x128 loupe side-by-side
                scan_key = f"{ck}-mono" if mode == "mono" else ck
                s64 = scanlines[(scan_key, 64, bg_key)]
                s16 = scanlines[(scan_key, 16, bg_key)]
                canvas.blit_scanlines(cx + 20, ry + 64, 64, 64, s64)
                canvas.blit_scanlines(cx + 44, ry + 152, 16, 16, s16)
                canvas.blit_scaled_nn(cx + 128, ry + 68, 16, 16, s16, scale=8, grid_rgb=grid)
                lbl_rgb = (71, 85, 105) if bg_key == "light" else (203, 213, 225)
                canvas.draw_text(cx + 28, ry + 34, "64PX", lbl_rgb, scale=2)
                canvas.draw_text(cx + 28, ry + 184, "16PX", lbl_rgb, scale=2)
                canvas.draw_text(cx + 144, ry + 34, "16PX 8X", lbl_rgb, scale=2)

    return width, height, bytes(canvas.buf)


# ---------------------------------------------------------------------------
# Artifact generation + self-test
# ---------------------------------------------------------------------------


def generate_all_bytes() -> Tuple[Dict[str, bytes], Dict[str, Tuple[int, int, bytes]]]:
    """Generate all SVG and PNG byte payloads deterministically in memory."""
    svgs: Dict[str, bytes] = {}
    for key, stem, title, _ in CONCEPT_SPECS:
        shapes = get_concept_shapes(key)
        svgs[f"{stem}.svg"] = render_concept_svg(shapes, title).encode("utf-8")

    pngs: Dict[str, Tuple[int, int, bytes]] = {}
    all_scanlines: Dict[Tuple[str, int, str], bytes] = {}
    loupe_16: Dict[Tuple[str, str], bytes] = {}

    for key, stem, _, is_mono in CONCEPT_SPECS:
        shapes = get_concept_shapes(key)
        if is_mono:
            for sz in (64, 16):
                for bg_key, bg_rgb, ink_rgb in (
                    ("light", LIGHT_BG_RGB, MONO_LIGHT_RGB),
                    ("dark", DARK_BG_RGB, MONO_DARK_RGB),
                ):
                    rgba = rasterize_rgba_4x4(shapes, sz, current_color_rgb=ink_rgb)
                    all_scanlines[(key, sz, bg_key)] = composite_over_bg(rgba, sz, bg_rgb)
            continue
        for sz in PREVIEW_SIZES:
            rgba = rasterize_rgba_4x4(shapes, sz)
            for bg_key, bg_rgb in (("light", LIGHT_BG_RGB), ("dark", DARK_BG_RGB)):
                scans = composite_over_bg(rgba, sz, bg_rgb)
                all_scanlines[(key, sz, bg_key)] = scans
                if sz == 16:
                    loupe_16[(key, bg_key)] = scans
                fname = f"{stem}_{sz}_{bg_key}.png"
                pngs[fname] = (sz, sz, encode_png_rgba(sz, sz, scans))

    lw, lh, l_scans = build_loupe_sheet(loupe_16)
    pngs["concepts_16px_loupe.png"] = (lw, lh, encode_png_rgba(lw, lh, l_scans))

    cw, ch, c_scans = build_comparison_sheet(all_scanlines)
    pngs["concepts_comparison_sheet.png"] = (cw, ch, encode_png_rgba(cw, ch, c_scans))

    return svgs, pngs


def _rel_luminance(r: int, g: int, b: int) -> float:
    def _lin(c: int) -> float:
        x = c / 255.0
        return x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4

    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def _contrast_ratio(rgb1: Tuple[int, int, int], rgb2: Tuple[int, int, int]) -> float:
    l1 = _rel_luminance(*rgb1)
    l2 = _rel_luminance(*rgb2)
    hi, lo = (l1, l2) if l1 >= l2 else (l2, l1)
    return (hi + 0.05) / (lo + 0.05)


def run_self_test(svg_dir: Path, out_dir: Path) -> None:
    """Verify two-run byte determinism, PNG validity/CRCs, SVG hygiene, and 16px contrast."""
    svgs_1, pngs_1 = generate_all_bytes()
    svgs_2, pngs_2 = generate_all_bytes()

    if svgs_1 != svgs_2:
        raise AssertionError("Non-deterministic SVG output across two runs")
    if pngs_1 != pngs_2:
        raise AssertionError("Non-deterministic PNG output across two runs")

    # Verify SVG hygiene: valid XML, zero <text>/<tspan>, zero external URLs
    svg_ns = "http://www.w3.org/2000/svg"
    url_re = re.compile(r"https?://|xlink:href|url\(#?(?!\s*$)", re.IGNORECASE)
    for fname, raw in svgs_1.items():
        text = raw.decode("utf-8")
        root = ET.fromstring(text)
        if root.tag != f"{{{svg_ns}}}svg":
            raise AssertionError(f"{fname}: unexpected root tag {root.tag}")
        if root.attrib.get("viewBox") != "0 0 512 512":
            raise AssertionError(f"{fname}: unexpected viewBox {root.attrib.get('viewBox')!r}")
        for elem in root.iter():
            local = elem.tag.split("}")[-1].lower()
            if local in ("text", "tspan", "image", "script", "foreignobject"):
                raise AssertionError(f"{fname}: forbidden SVG tag <{local}>")
        stripped = text.replace('xmlns="http://www.w3.org/2000/svg"', "")
        if url_re.search(stripped):
            raise AssertionError(f"{fname}: unexpected external URL or url() reference")
        if fname.endswith("-mono.svg"):
            if "currentColor" not in text:
                raise AssertionError(f"{fname}: mono SVG must use currentColor")
            if "#0F172A" in text:
                raise AssertionError(f"{fname}: mono SVG must not hardcode #0F172A")

    # Verify all PNGs and 16x16 contrast & legibility
    raw_16_by_key: Dict[str, bytes] = {}
    for fname, (w, h, data) in pngs_1.items():
        raw_scans = validate_png_bytes(data, w, h)
        if fname.endswith("_16_light.png") or fname.endswith("_16_dark.png"):
            is_light = fname.endswith("_16_light.png")
            bg = LIGHT_BG_RGB if is_light else DARK_BG_RGB
            changed = 0
            max_bg_cr = 1.0
            for py in range(16):
                row_off = py * (1 + 16 * 4) + 1
                for px in range(16):
                    i = row_off + px * 4
                    rgb = (raw_scans[i], raw_scans[i + 1], raw_scans[i + 2])
                    if rgb != bg:
                        changed += 1
                    cr = _contrast_ratio(rgb, bg)
                    if cr > max_bg_cr:
                        max_bg_cr = cr
            cov = changed / 256.0
            if cov < 0.35:
                raise AssertionError(f"{fname}: 16x16 foreground coverage too low ({cov:.2f})")
            if max_bg_cr < 3.0:
                raise AssertionError(f"{fname}: 16x16 peak contrast vs bg too low ({max_bg_cr:.2f}:1)")
            if is_light:
                raw_16_by_key[fname] = raw_scans

    # Ensure all three concepts are pairwise distinct at 16x16
    keys_16 = sorted(raw_16_by_key.keys())
    for i in range(len(keys_16)):
        for j in range(i + 1, len(keys_16)):
            a = raw_16_by_key[keys_16[i]]
            b = raw_16_by_key[keys_16[j]]
            diff_px = sum(1 for k in range(len(a)) if abs(a[k] - b[k]) > 12)
            if diff_px < 40:
                raise AssertionError(f"16x16 concepts {keys_16[i]} and {keys_16[j]} are too similar")

    # Verify on-disk files match generated bytes if already written, and test temp-dir write determinism
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for fname, raw in svgs_1.items():
            (tmp_path / fname).write_bytes(raw)
            assert (tmp_path / fname).read_bytes() == svgs_2[fname]

    for fname, raw in svgs_1.items():
        disk = svg_dir / fname
        if disk.exists() and disk.read_bytes() != raw:
            raise AssertionError(f"Stale SVG on disk: {disk}")
    for fname, (_, _, data) in pngs_1.items():
        disk = out_dir / fname
        if disk.exists() and disk.read_bytes() != data:
            raise AssertionError(f"Stale PNG on disk: {disk}")

    print(
        f"SELF-TEST PASS: {len(svgs_1)} SVGs + {len(pngs_1)} PNGs verified "
        f"(byte-identical across 2 runs, valid RFC 2083 CRCs, zero text/URLs, 16px contrast >= 3.0:1)"
    )


def write_outputs(svg_dir: Path, out_dir: Path) -> List[Path]:
    svgs, pngs = generate_all_bytes()
    svg_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    for fname, raw in sorted(svgs.items()):
        p = svg_dir / fname
        p.write_bytes(raw)
        written.append(p)
    for fname, (_, _, data) in sorted(pngs.items()):
        p = out_dir / fname
        p.write_bytes(data)
        written.append(p)
    return written


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Render FERRIC brand concept SVGs and PNG previews.")
    parser.add_argument(
        "--svg-dir",
        type=Path,
        default=Path("docs/assets/brand/concepts"),
        help="Output directory for concept SVG files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/assets/brand/concepts/previews"),
        help="Output directory for preview PNG files.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run deterministic two-pass, PNG CRC, SVG hygiene, and 16px contrast verification.",
    )
    args = parser.parse_args(argv)

    written = write_outputs(args.svg_dir, args.out_dir)
    if args.self_test:
        run_self_test(args.svg_dir, args.out_dir)

    for p in written:
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        print(f"{digest}  {p.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
