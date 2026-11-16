#!/usr/bin/env python3
"""Post-hoc, non-gate diagnostic for Gate F16 DOSY b-factors, MAT5/VnmrJ decays & GPU parity.

Provenance & Scientific Disclosure:
- Original clean-room Python 3 standard-library diagnostic script (MIT License).
- Lives OUTSIDE `scripts/p17_adapters/` so pre-download `bbc770a` adapters remain 0-byte diff.
- Zero external dependencies and zero network calls.
- **Alters NO primary gate verdict (`F16 = FAIL`)**:
  1. Unmodified `bbc770a` `scripts/p17_adapters/w4_adapters.py :: evaluate_f16_dosy` contains
     zero automated `.mat`, `dosy_in`, literature, or Stokes-Einstein `D_ref` / `ordering_pairs`
     extractor (`17.5e:P1-1` standard), and uncurated all-peaks `min_principal_decay_fit_r2` on
     Varian `MENDELEY:d7bdxz9hsk` is `0.9080 < 0.95` (`39/41 >= 0.95`; `2` weak Quinine multiplet
     shoulders at `2.858 / 2.810 ppm` have `R^2 = 0.9080 / 0.9240`).
  2. Neither `TyrosinaseHCCA_pfg_data.mat` (`ZENODO:12170718`) nor VnmrJ `dosy_in`
     (`MENDELEY:d7bdxz9hsk`) contains deposited scalar `D_ref` values from an independent
     experiment (`17.5e:P1-2` standard): both store vendor/author-processed spectral amplitudes
     of the *same* raw pseudo-2D FIDs (`SPECTRA` in NMRPipe/DOSY-Toolbox and `line_amps` in
     VnmrJ). Comparing against them is strictly a DSP + `b`-factor processing cross-check, NOT an
     independent biophysical diffusivity benchmark.
  3. On `MENDELEY:d7bdxz9hsk`, `gates.F16.primary_datasets[1].files` pre-registered ONLY `fid`
     and `procpar`. Auxiliary file `dosy_in` (Mendeley file UUID
     `ae9285f6-7ab3-4f26-b13b-e409c995f801`) is an **unregistered auxiliary file** inspected
     strictly here in post-hoc diagnostics (`--varian-dosy-in`, optional) to verify Wu-Johnson
     Oneshot `b`-factor calibration (`b_nmrassign / b_vnmrj = 1.001641`, `+0.000712 dex`).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import struct
import zlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

GAMMA_1H_RAD_S_T = 2.6752218744e8


def ols_slope_through_log_amps(b_s_m2: Sequence[float], amps: Sequence[float]) -> float:
    """Closed-form unweighted OLS slope -d(ln I)/db for positive amplitudes."""
    pts = [(float(b), math.log(float(a))) for b, a in zip(b_s_m2, amps) if float(a) > 0.0]
    if len(pts) < 3:
        return math.nan
    n = float(len(pts))
    mx = math.fsum(x for x, _ in pts) / n
    my = math.fsum(y for _, y in pts) / n
    sxx = math.fsum((x - mx) * (x - mx) for x, _ in pts)
    sxy = math.fsum((x - mx) * (y - my) for x, y in pts)
    return (-sxy / sxx) if sxx > 0.0 else math.nan


def compute_cpu_gpu_parity(staging_f16_dir: Path, slugs: Sequence[str]) -> Dict[str, Any]:
    """Compare CPU f64 vs WebGPU Metal f32 BIC model selection K and dominant log10(D)."""
    per_run: Dict[str, Any] = {}
    total_peaks = 0
    total_k_agree = 0
    global_max_k1 = 0.0
    global_max_all = 0.0

    for slug in slugs:
        cpu_path = staging_f16_dir / f"f16_{slug}_dosy_fit_cpu.json"
        gpu_path = staging_f16_dir / f"f16_{slug}_dosy_fit_gpu.json"
        if not (cpu_path.is_file() and gpu_path.is_file()):
            continue
        cpu_js = json.loads(cpu_path.read_text(encoding="utf-8"))
        gpu_js = json.loads(gpu_path.read_text(encoding="utf-8"))

        d_all: List[float] = []
        d_k1: List[float] = []
        k_same = 0
        for pc, pg in zip(cpu_js.get("peaks", []), gpu_js.get("peaks", [])):
            dc = float(max(pc["components"], key=lambda c: float(c["fraction"]))["log10_d"])
            dg = float(max(pg["components"], key=lambda c: float(c["fraction"]))["log10_d"])
            diff = abs(dc - dg)
            d_all.append(diff)
            if int(pc["selected_k"]) == int(pg["selected_k"]):
                k_same += 1
            if int(pc["selected_k"]) == 1 and int(pg["selected_k"]) == 1:
                d_k1.append(diff)

        n_p = len(d_all)
        max_all = max(d_all) if d_all else 0.0
        max_k1 = max(d_k1) if d_k1 else 0.0
        total_peaks += n_p
        total_k_agree += k_same
        global_max_all = max(global_max_all, max_all)
        global_max_k1 = max(global_max_k1, max_k1)

        per_run[slug] = {
            "n_peaks": n_p,
            "bic_k_agreement": f"{k_same}/{n_p}",
            "bic_k_concordance_fraction": (float(k_same) / float(n_p)) if n_p else 1.0,
            "max_abs_delta_log10_d_k1_dex": max_k1,
            "max_abs_delta_log10_d_all_dex": max_all,
            "gpu_adapter": gpu_js.get("gpu_adapter"),
        }

    return {
        "total_peaks_evaluated": total_peaks,
        "total_bic_k_agreement": f"{total_k_agree}/{total_peaks}",
        "global_max_abs_delta_log10_d_k1_dex": global_max_k1,
        "global_max_abs_delta_log10_d_all_dex": global_max_all,
        "per_dataset": per_run,
    }


def parse_mat5_pfgnmrdata_structs(mat_path: Path) -> Dict[str, Dict[str, Any]]:
    """Minimal stdlib MAT-level-5 zlib reader for DOSY-Toolbox `pfgnmrdata_*` structs."""
    raw = mat_path.read_bytes()
    pos = 128
    structs: Dict[str, Dict[str, Any]] = {}
    while pos + 8 <= len(raw):
        dt, nb = struct.unpack("<II", raw[pos : pos + 8])
        pos += 8
        if dt == 15:
            d = zlib.decompress(raw[pos : pos + nb])
            ndt, nnb = struct.unpack("<II", d[40:48])
            if (ndt >> 16) != 0:
                vname = d[44 : 44 + (ndt >> 16)].split(b"\x00", 1)[0].decode("ascii", errors="replace")
            else:
                vname = d[48 : 48 + nnb].split(b"\x00", 1)[0].decode("ascii", errors="replace")

            fn_idx = d.find(b"acqus\x00")
            pul_idx = d.find(b"pulprog\x00", fn_idx)
            if fn_idx >= 0 and pul_idx > fn_idx:
                stride = pul_idx - fn_idx
                _, ftag_nb = struct.unpack("<II", d[fn_idx - 8 : fn_idx])
                n_fields = ftag_nb // stride
                fnames = [
                    d[fn_idx + k * stride : fn_idx + (k + 1) * stride]
                    .split(b"\x00", 1)[0]
                    .decode("ascii", errors="replace")
                    for k in range(n_fields)
                ]
                cur = fn_idx + ((ftag_nb + 7) & ~7)
                fields: Dict[str, Any] = {}
                for fn in fnames:
                    if cur + 8 > len(d):
                        break
                    _, enb = struct.unpack("<II", d[cur : cur + 8])
                    elem = d[cur + 8 : cur + 8 + enb]
                    cur += 8 + ((enb + 7) & ~7)
                    if fn in ("dosyconstant", "DELTA", "delta", "tau", "sfrq", "ngrad") and len(elem) >= 8:
                        fields[fn] = struct.unpack("<d", elem[-8:])[0]
                    elif fn in ("Gzlvl", "Ppmscale") and len(elem) >= 48:
                        _, dnb = struct.unpack("<II", elem[40:48])
                        cnt = dnb // 8
                        fields[fn] = struct.unpack(f"<{cnt}d", elem[48 : 48 + dnb])
                    elif fn == "SPECTRA" and len(elem) >= 32:
                        rows, cols = struct.unpack("<II", elem[24:32])
                        target_tag = struct.pack("<II", 9, rows * cols * 8)
                        t_off = elem.find(target_tag, 32)
                        if t_off >= 0:
                            vals = struct.unpack(
                                f"<{rows * cols}d", elem[t_off + 8 : t_off + 8 + rows * cols * 8]
                            )
                            fields["SPECTRA_shape"] = (rows, cols)
                            fields["SPECTRA"] = vals
                structs[vname] = fields
        pos += nb if dt == 15 else ((nb + 7) & ~7)
    return structs


def analyze_bruker_zenodo12170718_mat_vs_nmrassign(
    mat_path: Path,
    staging_f16_dir: Path,
) -> Dict[str, Any]:
    """Compare nmrassign SMSQ10.100 fits vs author NMRPipe/DOSY-Toolbox `.mat` SPECTRA."""
    if not mat_path.is_file():
        return {"available": False, "reason": f"missing {mat_path}"}

    structs = parse_mat5_pfgnmrdata_structs(mat_path)
    smsq_shape_factor_sq = 0.81  # Bruker SMSQ10.100 integral factor = 0.90 -> 0.90^2 = 0.81

    comparisons: Dict[str, Any] = {}
    mapping: List[Tuple[str, str, Tuple[float, ...]]] = [
        ("pfgnmrdata_Pb", "bruker_lth", (3.658, 1.972)),
        ("pfgnmrdata_Pa", "bruker_lt", (3.658, 1.972)),
        ("pfgnmrdata_L", "bruker_ligands_only", (7.926, 7.244, 6.876, 5.408, 3.659, 1.972)),
    ]

    for struct_name, slug, target_ppms in mapping:
        fit_path = staging_f16_dir / f"f16_{slug}_dosy_fit_cpu.json"
        if struct_name not in structs or not fit_path.is_file():
            continue
        st = structs[struct_name]
        fit_js = json.loads(fit_path.read_text(encoding="utf-8"))
        b_nmrassign = [float(b) for b in fit_js["b_values_s_m2"]]
        dosyconst = float(st["dosyconstant"])
        gz = [float(g) for g in st["Gzlvl"]]
        b_toolbox_rect = [g * g * dosyconst for g in gz]
        b_toolbox_smsq = [b * smsq_shape_factor_sq for b in b_toolbox_rect]

        rows, cols = st["SPECTRA_shape"]
        spec = st["SPECTRA"]
        ppm = st["Ppmscale"]

        peak_rows: List[Dict[str, Any]] = []
        for t_ppm in target_ppms:
            pk_fit = min(fit_js["peaks"], key=lambda p: abs(float(p["center_ppm"]) - t_ppm))
            d_fit = float(max(pk_fit["components"], key=lambda c: float(c["fraction"]))["d_m2_s"])

            cand_r = [r for r, p in enumerate(ppm) if abs(float(p) - t_ppm) <= 0.035]
            if not cand_r:
                continue
            best_r = max(cand_r, key=lambda r: float(spec[0 * rows + r]))
            amps_all = [float(spec[c * rows + best_r]) for c in range(cols)]
            if amps_all[0] <= 0.0:
                continue
            keep_idx = [c for c in range(cols) if amps_all[c] > 0.15 * amps_all[0]]
            d_mat_smsq = ols_slope_through_log_amps(
                [b_toolbox_smsq[c] for c in keep_idx],
                [amps_all[c] for c in keep_idx],
            )
            d_mat_rect = ols_slope_through_log_amps(
                [b_toolbox_rect[c] for c in keep_idx],
                [amps_all[c] for c in keep_idx],
            )
            peak_rows.append(
                {
                    "target_ppm": t_ppm,
                    "nmrassign_peak_label": pk_fit["label"],
                    "nmrassign_center_ppm": float(pk_fit["center_ppm"]),
                    "mat_ppm": float(ppm[best_r]),
                    "d_nmrassign_m2_s": d_fit,
                    "d_mat_smsq081_ols_m2_s": d_mat_smsq,
                    "d_mat_uncalibrated_rect_ols_m2_s": d_mat_rect,
                    "abs_log10_diff_vs_mat_smsq_dex": abs(math.log10(d_fit / d_mat_smsq)),
                    "abs_log10_diff_vs_mat_rect_dex": abs(math.log10(d_fit / d_mat_rect)),
                }
            )

        m_cmp = min(len(b_nmrassign), len(b_toolbox_smsq)) - 1
        comparisons[slug] = {
            "mat_struct": struct_name,
            "n_planes_nmrassign": len(b_nmrassign),
            "n_planes_mat_gradrange": len(b_toolbox_smsq),
            "compared_plane_index": m_cmp,
            "b_at_compared_plane_nmrassign_s_m2": b_nmrassign[m_cmp],
            "b_at_compared_plane_mat_smsq081_s_m2": b_toolbox_smsq[m_cmp],
            "b_at_compared_plane_mat_uncalibrated_rect_s_m2": b_toolbox_rect[m_cmp],
            "b_max_256_nmrassign_s_m2": b_nmrassign[-1],
            "ratio_b_nmrassign_over_mat_smsq081_same_plane": b_nmrassign[m_cmp] / b_toolbox_smsq[m_cmp],
            "max_abs_log10_diff_vs_mat_smsq_dex": max(
                r["abs_log10_diff_vs_mat_smsq_dex"] for r in peak_rows
            )
            if peak_rows
            else None,
            "peaks": peak_rows,
        }

    return {
        "available": True,
        "independence_caveat": (
            "NON-INDEPENDENT PROCESSING CROSS-CHECK ONLY: TyrosinaseHCCA_pfg_data.mat stores "
            "NMRPipe-processed F2 planes (`SPECTRA`) of the SAME raw Bruker `ser` dataset rather "
            "than deposited scalar diffusivities or an independent biophysical reference."
        ),
        "comparisons": comparisons,
    }


def analyze_bruker_zenodo16423019_oneshot_pulprog(
    exp_23_dir: Path,
    staging_f16_dir: Path,
) -> Dict[str, Any]:
    """Diagnose custom Bruker Oneshot `PULPROG=<gdp_dosyoneshot_tlg>` P30 vs 2*P30 b-factor scaling."""
    fit_path = staging_f16_dir / "f16_bruker_z16423019_dosy_fit_cpu.json"
    acqus_path = exp_23_dir / "acqus"
    difflist_path = exp_23_dir / "difflist"
    if not (fit_path.is_file() and acqus_path.is_file() and difflist_path.is_file()):
        return {"available": False}

    fit_js = json.loads(fit_path.read_text(encoding="utf-8"))
    b_nmrassign = [float(b) for b in fit_js["b_values_s_m2"]]
    grads_g_cm = [
        float(ln.strip().split()[0])
        for ln in difflist_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]

    acqus_lines = acqus_path.read_text(encoding="utf-8", errors="replace").splitlines()
    cnst_vals: List[float] = []
    for i, ln in enumerate(acqus_lines):
        if ln.strip().startswith("##$CNST="):
            j = i + 1
            while j < len(acqus_lines) and not acqus_lines[j].strip().startswith("##"):
                cnst_vals.extend(float(t) for t in acqus_lines[j].split())
                j += 1
            break

    cnst14_alpha = cnst_vals[14] if len(cnst_vals) > 14 else 0.2
    cnst18_dosytimecubed_s3 = cnst_vals[18] if len(cnst_vals) > 18 else math.nan
    sine100_shape_sq = (2.0 / math.pi) ** 2
    g_max_t_m = grads_g_cm[-1] * 1.0e-2
    b_max_oneshot_sine100 = (
        (GAMMA_1H_RAD_S_T ** 2) * cnst18_dosytimecubed_s3 * sine100_shape_sq * (g_max_t_m ** 2)
    )
    ratio = b_max_oneshot_sine100 / b_nmrassign[-1]

    return {
        "available": True,
        "pulprog": "gdp_dosyoneshot_tlg",
        "alpha_cnst14": cnst14_alpha,
        "cnst18_dosytimecubed_s3": cnst18_dosytimecubed_s3,
        "gpnam_shape": "SINE.100",
        "b_max_nmrassign_s_m2": b_nmrassign[-1],
        "b_max_oneshot_cnst18_sine100_s_m2": b_max_oneshot_sine100,
        "ratio_b_oneshot_over_nmrassign": ratio,
        "log10_ratio_dex": math.log10(ratio),
        "root_cause": (
            "parse_bruker_dosy in crates/nmr-core/src/dosy.rs checks pulprog_low.contains('bp') "
            "for bipolar P30=delta/2 doubling (recognizing 'oneshot' in Varian seqfil but not in "
            "Bruker PULPROG=<gdp_dosyoneshot_tlg>), causing little_delta = P30 (1.1 ms) instead of "
            "2*P30 (2.2 ms) and a 4.013x b-factor under-scaling."
        ),
    }


def analyze_varian_mendeley_dosy_in(
    dosy_in_path: Optional[Path],
    staging_f16_dir: Path,
) -> Dict[str, Any]:
    """Post-hoc comparison against unregistered Mendeley VnmrJ `dosy_in` auxiliary file."""
    fit_path = staging_f16_dir / "f16_varian_mendeley_dosy_fit_cpu.json"
    if dosy_in_path is None or not dosy_in_path.is_file() or not fit_path.is_file():
        return {
            "available": False,
            "unregistered_auxiliary_file_disclosure": (
                "dosy_in is NOT pre-registered in gates.F16.primary_datasets[1].files "
                "(only `fid` and `procpar` are registered); omitted or file not present."
            ),
        }

    lines = [
        ln.strip()
        for ln in dosy_in_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if ln.strip()
    ]
    dosyconst_cgs = 0.0
    grad_idx = 0
    for idx, ln in enumerate(lines):
        if ln.startswith("dosyconstant"):
            dosyconst_cgs = float(ln.split("=", 1)[1].strip())
        elif ln.startswith("gradient calibration flag"):
            grad_idx = idx + 1
            break

    grads_gcm = [float(lines[grad_idx + k]) for k in range(12)]
    b_vnmrj = [(g * g) * (dosyconst_cgs * 1.0e-4) for g in grads_gcm]

    fit_js = json.loads(fit_path.read_text(encoding="utf-8"))
    b_nmrassign = [float(b) for b in fit_js["b_values_s_m2"]]
    b_ratio = b_nmrassign[-1] / b_vnmrj[-1]

    freq_hdr_idx = next(i for i, ln in enumerate(lines) if ln.startswith("line") and "frequency" in ln)
    amp_hdr_idx = next(i for i, ln in enumerate(lines) if ln.startswith("line") and "amplitude" in ln)
    line_hz: Dict[int, float] = {}
    for ln in lines[freq_hdr_idx + 1 : amp_hdr_idx]:
        parts = ln.split()
        if len(parts) == 2:
            line_hz[int(parts[0])] = float(parts[1])

    line_amps: Dict[int, List[float]] = {k: [] for k in line_hz}
    for ln in lines[amp_hdr_idx + 1 :]:
        parts = ln.split()
        if len(parts) == 3 and int(parts[0]) in line_amps:
            line_amps[int(parts[0])].append(float(parts[2]))

    species_targets = [
        (1, 8.657, "Quinine H2' (MW 324.42 Da)"),
        (24, 3.877, "Quinine OCH3 (MW 324.42 Da)"),
        (25, 3.399, "Residual HDO (MW 19.02 Da)"),
        (37, 2.483, "Residual DMSO-d5 (MW 84.17 Da)"),
        (50, -0.091, "TSP (MW 172.27 Da)"),
    ]
    species_rows: List[Dict[str, Any]] = []
    for lid, target_ppm, desc in species_targets:
        pk_fit = min(fit_js["peaks"], key=lambda p: abs(float(p["center_ppm"]) - target_ppm))
        d_fit = float(max(pk_fit["components"], key=lambda c: float(c["fraction"]))["d_m2_s"])
        d_vnmrj_ols = ols_slope_through_log_amps(b_vnmrj, line_amps[lid])
        species_rows.append(
            {
                "vnmrj_line_id": lid,
                "species": desc,
                "nmrassign_peak_label": pk_fit["label"],
                "nmrassign_center_ppm": float(pk_fit["center_ppm"]),
                "d_nmrassign_m2_s": d_fit,
                "d_vnmrj_dosy_in_ols_m2_s": d_vnmrj_ols,
                "abs_log10_diff_dex": abs(math.log10(d_fit / d_vnmrj_ols)),
            }
        )

    return {
        "available": True,
        "unregistered_auxiliary_file_disclosure": (
            "POST-HOC DIAGNOSTIC ONLY: `dosy_in` (Mendeley file UUID ae9285f6-7ab3-4f26-b13b-e409c995f801) "
            "is NOT pre-registered in `gates.F16.primary_datasets[1].files` (which registers ONLY `fid` and "
            "`procpar`). Furthermore, `dosy_in` line amplitudes come from VnmrJ processing of the SAME 12-plane "
            "Varian FID rather than an independent PFG-STE/ITC/DLS measurement. Excluded from `f16_evaluation.json`."
        ),
        "b_min_nmrassign_s_m2": b_nmrassign[0],
        "b_min_vnmrj_s_m2": b_vnmrj[0],
        "b_max_nmrassign_s_m2": b_nmrassign[-1],
        "b_max_vnmrj_s_m2": b_vnmrj[-1],
        "ratio_b_nmrassign_over_vnmrj": b_ratio,
        "log10_b_ratio_dex": math.log10(b_ratio),
        "max_species_abs_log10_diff_dex": max(r["abs_log10_diff_dex"] for r in species_rows),
        "species_comparisons": species_rows,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Post-hoc, non-blind F16 DOSY diagnostic (alters NO primary gate verdict; F16 = FAIL)."
    )
    ap.add_argument(
        "--staging-f16-dir",
        default="/tmp/p17_w4_staging_r2/f16",
        help="Directory holding strict F16 nmrassign dosy fit CPU/GPU JSON outputs",
    )
    ap.add_argument(
        "--bruker-mat",
        default="/tmp/p17_empirical_downloads/w4/f16_r2/ZENODO_12170718/extracted/CompetitiveCooperative_CoLD_CoP_for_archiving/TyrosinaseHCCA/TyrosinaseHCCA_pfg_data.mat",
        help="Path to TyrosinaseHCCA_pfg_data.mat inside ZENODO:12170718",
    )
    ap.add_argument(
        "--zenodo16423019-exp-dir",
        default="/tmp/p17_empirical_downloads/w4/f16_r2/ZENODO_16423019/extracted/23",
        help="Path to extracted `23/` Bruker directory inside ZENODO:16423019 (`backup_1`)",
    )
    ap.add_argument(
        "--varian-dosy-in",
        default="/tmp/p17_empirical_downloads/w4/f16/MENDELEY_d7bdxz9hsk_aux/dosy_in",
        help="Optional path to unregistered Mendeley VnmrJ `dosy_in` file (strictly post-hoc diagnostic)",
    )
    ap.add_argument(
        "--json-out",
        default=None,
        help="Optional output path for deterministic JSON diagnostic report",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    stg_dir = Path(args.staging_f16_dir)
    slugs = ("bruker_lth", "bruker_lt", "bruker_ligands_only", "varian_mendeley", "bruker_z16423019")

    report = {
        "gate_id": "F16",
        "diagnostic_only_non_blind": True,
        "alters_primary_gate_verdict": False,
        "primary_strict_bbc770a_verdict": "FAIL",
        "cpu_f64_vs_webgpu_metal_f32_parity": compute_cpu_gpu_parity(stg_dir, slugs),
        "bruker_zenodo12170718_smsq10_vs_mat_decay_diagnostic": analyze_bruker_zenodo12170718_mat_vs_nmrassign(
            Path(args.bruker_mat), stg_dir
        ),
        "bruker_zenodo16423019_oneshot_pulprog_diagnostic": analyze_bruker_zenodo16423019_oneshot_pulprog(
            Path(args.zenodo16423019_exp_dir), stg_dir
        ),
        "varian_mendeley_unregistered_dosy_in_posthoc_diagnostic": analyze_varian_mendeley_dosy_in(
            Path(args.varian_dosy_in) if args.varian_dosy_in else None,
            stg_dir,
        ),
    }

    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.json_out:
        out_p = Path(args.json_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
