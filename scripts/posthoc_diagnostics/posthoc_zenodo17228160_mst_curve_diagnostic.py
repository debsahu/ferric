#!/usr/bin/env python3
"""Post-hoc, non-gate audit diagnostic for ZENODO:17228160 MST curve back-solve (17.5e:P1-1 / P1-2).

Provenance & Scientific Disclosure:
- Original clean-room Python 3 standard-library diagnostic script (MIT License).
- Lives OUTSIDE `scripts/p17_adapters/` so pre-download `bbc770a` adapters remain 0-byte diff.
- Documents how the withdrawn Wave 3 summary scalars (`226.8254413 uM` from
  `MST.tar.xz :: MST/B1_labeled_WT_tau_titrated.dat` lines 29..128 and `221.05369 uM` from
  `Source_Data.xlsx :: SuppFig3BC`) were computed post-download, and proves why NEITHER is a
  valid pre-registered `F13c-C2` reference for the NMR `tau1-243 + DnaJB1` titration:
  1. Unmodified `bbc770a` `scripts/p17_adapters/w3_adapters.py :: extract_f13c_reference_kd_um`
     raises `ValueError: could not locate independent ITC/MST K_d value in biophysical metadata`
     on every file and worksheet in `ZENODO:17228160` (neither scalar is deposited literally).
  2. Construct mismatch (`17.5e:P1-2`): the MST `WT` series (`B1_labeled_WT_tau_titrated.dat`
     and `Source_Data.xlsx :: SuppFig3BC`) accompanies sibling mutant columns `A152T`, `S320F`,
     and `R406W` (residues 320 and 406 lie outside `tau1-243`), matching `Fig3BC`
     `DnaJB1_WT2N4Rtau` (full-length 441-aa `2N4R` tau), whereas `NMR.tar.xz` titrates truncated
     `tau1-243`.
  3. Extrapolation (`17.5e:P1-2`): the highest experimental `[tau]` titration point in
     `MST/B1_labeled_WT_tau_titrated.dat` (lines 13..27) and `SuppFig3BC` (`A4:A19`) is
     `100,000 nM` (`100.0 uM`), well below the extrapolated `~221-227 uM` 1:1 quadratic curve.

Quadratic 1:1 binding isotherm (Morrison 1969, Biochim. Biophys. Acta 185, 269-286):
  f_bound(L_t; C_t, K_d) = ((C_t + L_t + K_d) - sqrt((C_t + L_t + K_d)^2 - 4 * C_t * L_t)) / (2 * C_t)
  F_fit(L_t) = F_unbound + (F_bound - F_unbound) * f_bound(L_t; C_t, K_d)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile


def morrison_f_bound(l_nm: float, c_target_nm: float, kd_nm: float) -> float:
    s = c_target_nm + l_nm + kd_nm
    disc = max(0.0, s * s - 4.0 * c_target_nm * l_nm)
    return (s - math.sqrt(disc)) / (2.0 * c_target_nm)


def solve_linear_2x2_least_squares(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Fit y = a + b * x in closed form; return (a, b, sse)."""
    n = float(len(xs))
    mx = math.fsum(xs) / n
    my = math.fsum(ys) / n
    sxx = math.fsum((x - mx) * (x - mx) for x in xs)
    sxy = math.fsum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx if sxx > 0.0 else 0.0
    a = my - b * mx
    sse = math.fsum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    return a, b, sse


def golden_section_kd_nm(
    l_nm_list: list[float],
    y_list: list[float],
    c_target_nm: float,
    lo_nm: float = 1.0,
    hi_nm: float = 1.0e8,
    iters: int = 200,
) -> dict[str, float]:
    """Deterministic golden-section + VarPro 1D minimization of SSE(K_d) in nanomolar."""
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    a = math.log(lo_nm)
    b = math.log(hi_nm)

    def eval_log_kd(lk: float) -> tuple[float, float, float]:
        kd = math.exp(lk)
        fb = [morrison_f_bound(x, c_target_nm, kd) for x in l_nm_list]
        f0, df, sse = solve_linear_2x2_least_squares(fb, y_list)
        return sse, f0, df

    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc = eval_log_kd(c)[0]
    fd = eval_log_kd(d)[0]
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = eval_log_kd(c)[0]
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = eval_log_kd(d)[0]

    best_lk = 0.5 * (a + b)
    kd_nm = math.exp(best_lk)
    sse, f0, df = eval_log_kd(best_lk)
    preds = [f0 + df * morrison_f_bound(x, c_target_nm, kd_nm) for x in l_nm_list]
    max_abs_err = max(abs(p - y) for p, y in zip(preds, y_list))
    return {
        "kd_nm": kd_nm,
        "kd_um": kd_nm / 1000.0,
        "c_target_nm": c_target_nm,
        "f_unbound": f0,
        "delta_f_sat": df,
        "sse": sse,
        "rmse": math.sqrt(sse / float(len(y_list))),
        "max_abs_err": max_abs_err,
    }


def parse_mo_affinity_dat(dat_path: Path) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Parse first experimental block and first `-666.666` fitted-curve block in NanoTemper `.dat`."""
    raw_pts: list[tuple[float, float]] = []
    curve_pts: list[tuple[float, float]] = []
    state = 0
    for raw in dat_path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 2:
            continue
        try:
            x = float(parts[0])
            y = float(parts[1])
        except ValueError:
            continue
        if x < 0.0 and x == y:
            state += 1
            if state >= 2:
                break
            continue
        if state == 0:
            raw_pts.append((x, y))
        elif state == 1:
            curve_pts.append((x, y))
    return raw_pts, curve_pts


def parse_suppfig3bc_blocks(
    xlsx_path: Path,
) -> tuple[list[str], list[tuple[float, float]], list[tuple[float, float]]]:
    """Extract SuppFig3BC series headers, WT replicate markers, and WT fitted curve."""
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(xlsx_path) as zf:
        shared = [
            "".join(t.text or "" for t in si.findall(".//m:t", ns))
            for si in ET.fromstring(zf.read("xl/sharedStrings.xml")).findall("m:si", ns)
        ]
        wb = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rmap = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
        target = None
        for sh in wb.findall(".//m:sheet", ns):
            if sh.attrib["name"] == "SuppFig3BC":
                rid = sh.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
                target = "xl/" + rmap[rid].lstrip("/")
                break
        if target is None:
            raise ValueError("SuppFig3BC sheet not found")
        sxml = ET.fromstring(zf.read(target))
        rows: list[list[str]] = []
        for row_el in sxml.findall(".//m:sheetData/m:row", ns):
            vals: list[str] = []
            for c_el in row_el.findall("m:c", ns):
                v = c_el.find("m:v", ns)
                if c_el.attrib.get("t") == "s" and v is not None and v.text is not None:
                    vals.append(shared[int(v.text)])
                elif v is not None and v.text is not None:
                    vals.append(v.text)
                else:
                    vals.append("")
            rows.append(vals)

    series_headers = [x.strip() for x in rows[1] if x.strip()]
    markers: list[tuple[float, float]] = []
    fits_curve: list[tuple[float, float]] = []
    section = "markers"
    for r in rows[3:]:
        if not r:
            continue
        first = r[0].strip()
        if first == "Fits":
            section = "fits"
            continue
        if not first:
            continue
        try:
            c_nm = float(first)
        except ValueError:
            continue
        if section == "markers":
            for col_idx in (1, 2, 3):
                if col_idx < len(r) and r[col_idx].strip():
                    markers.append((c_nm, float(r[col_idx])))
        elif section == "fits":
            if len(r) > 1 and r[1].strip():
                fits_curve.append((c_nm, float(r[1])))
    return series_headers, markers, fits_curve


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mst-dat",
        type=Path,
        required=True,
        help="Path to MST/B1_labeled_WT_tau_titrated.dat",
    )
    ap.add_argument(
        "--xlsx",
        type=Path,
        required=True,
        help="Path to Source_Data.xlsx",
    )
    ap.add_argument(
        "--c-target-nm",
        type=float,
        default=100.0,
        help="Assumed labeled DnaJB1 target concentration in nM (default: 100.0 nM)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output JSON path",
    )
    args = ap.parse_args()

    raw_pts, curve_pts = parse_mo_affinity_dat(args.mst_dat)
    curve_fit = golden_section_kd_nm(
        [p[0] for p in curve_pts],
        [p[1] for p in curve_pts],
        args.c_target_nm,
    )
    series_headers, wt_markers, wt_fits_curve = parse_suppfig3bc_blocks(args.xlsx)
    xlsx_markers_fit = golden_section_kd_nm(
        [p[0] for p in wt_markers],
        [p[1] for p in wt_markers],
        args.c_target_nm,
    )
    xlsx_curve_fit = golden_section_kd_nm(
        [p[0] for p in wt_fits_curve],
        [p[1] for p in wt_fits_curve],
        args.c_target_nm,
    )

    out_obj = {
        "diagnostic_classification": "POST_HOC_CONSTRUCT_MISMATCHED_EXTRAPOLATED_CURVE_BACKSOLVE_NOT_A_GATE_REFERENCE",
        "excluded_from_gate_verdict": True,
        "disclosures_17_5e_p1_1_and_p1_2": {
            "literal_kd_scalar_deposited_in_archive": False,
            "frozen_bbc770a_extractor_result": "ValueError: could not locate independent ITC/MST K_d value in biophysical metadata",
            "nmr_observed_construct": "tau1-243 (25 uM 15N-tau1-243 + 0..250 uM DnaJB1)",
            "mst_titrated_construct": "Full-length 2N4R WT tau (sibling SuppFig3BC/MST series carry A152T, S320F, R406W outside residues 1..243; Fig3BC labels matching ITC DnaJB1_WT2N4Rtau)",
            "suppfig3bc_series_headers": series_headers,
            "max_experimental_mst_tau_um": max(p[0] for p in raw_pts) / 1000.0,
            "is_extrapolated_above_max_titration_point": True,
        },
        "mo_affinity_100pt_curve_backsolve": {
            "n_experimental_points_in_block_0": len(raw_pts),
            "n_curve_points_in_block_1": len(curve_pts),
            **curve_fit,
        },
        "suppfig3bc_wt_fits_curve_backsolve": {
            "n_curve_points": len(wt_fits_curve),
            **xlsx_curve_fit,
        },
        "suppfig3bc_wt_replicate_markers_fit": {
            "n_replicate_points": len(wt_markers),
            **xlsx_markers_fit,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_obj, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out_obj, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
