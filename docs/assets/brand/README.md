# FERRIC Brand Guidelines

The canonical **FERRIC** brand suite is generated deterministically in pure Python (`scripts/render_ferric_brand.py`) with zero third-party dependencies, zero `<text>` elements, and zero external font or URL references.

## Assets

| File | Purpose | Notes |
| :--- | :--- | :--- |
| `ferric-icon.svg` | Canonical app icon (Concept A: `FID-F`) | Rust-framed `#16202A` squircle (`#B7410E` + `#D95B26`) |
| `ferric-icon-mono.svg` | Monochrome outline/glyph icon | Uses `currentColor` for light & dark themes |
| `ferric-wordmark.svg` | Standalone `"FERRIC"` wordmark | Original **MIT-licensed** geometric `<path>` outlines |
| `ferric-logo-light.svg` | README & light-surface header lockup | Concept A badge + wordmark + Concept C Larmor emblem |
| `ferric-logo-dark.svg` | README & dark-surface header lockup | High-contrast `#F8FAFC` wordmark + `#D95B26` accent |
| `favicon.svg` / `favicon-32.png` | Web manual & WebGPU demo favicons | Vector SVG + `32x32` 4x4-supersampled RGBA PNG |
| `ferric-social-preview.png` | GitHub social preview (`1280x640`) | Concept A + wordmark + Concept C Larmor cone + FID |

## Usage Rules, Clear Space & Minimum Sizes

- **Clear space:** Maintain at least `0.25x` icon width (`128` units on the `512` viewBox) around the icon badge, and `0.5x` cap height around the lockup.
- **Minimum sizes:**
  - Icon (`ferric-icon.svg` / `favicon.svg`): **`16x16` px** minimum.
  - Horizontal lockup (`ferric-logo-light.svg` / `ferric-logo-dark.svg`): **`120 px`** width minimum.
- **Do not** stretch, skew, recolor the rust/iron-slate icon palette outside the approved hex values, or add text inside the squircle icon.

## Colour Palette

| Role | Hex | RGB |
| :--- | :--- | :--- |
| Deep Iron Slate (Squircle Fill) | `#16202A` | `(22, 32, 42)` |
| Midnight Canvas (Dark Background) | `#0B1120` | `(11, 17, 32)` |
| Primary Iron Oxide / Rust | `#B7410E` | `(183, 65, 14)` |
| Rust Frame Highlight | `#D95B26` | `(217, 91, 38)` |
| Warm Ember | `#E05A2B` | `(224, 90, 43)` |
| Bright Damped FID Trace | `#FB923C` | `(251, 146, 60)` |
| `t = 0` Excitation Cap | `#FED7AA` | `(254, 215, 170)` |
| Equilibrium Axis Slate | `#475569` | `(71, 85, 105)` |
| Light-Mode Wordmark Ink | `#0F172A` | `(15, 23, 42)` |
| Dark-Mode Wordmark Ink | `#F8FAFC` | `(248, 250, 252)` |

## Wordmark & Icon Licence

All brand vectors and rasters in `docs/assets/brand/`—including the `"FERRIC"` geometric `<path>` outlines in `ferric-wordmark.svg`, `ferric-logo-light.svg`, and `ferric-logo-dark.svg`—are original works released under the **MIT License** (`LICENSE`).
