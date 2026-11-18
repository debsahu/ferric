#!/usr/bin/env python3
"""Clean-room published literature PFG-STE `D_ref` reference corpus & `D2O` water-skirt mask (`F29`).

Provenance (strictly from peer-reviewed literature; zero proprietary or GPL code consulted):
  - Longsworth, L. G. (J. Phys. Chem. 1960, 64, 1914-1917): Rayleigh-interferometric / PFG-NMR
    translational self-diffusion benchmarks in aqueous solution at 298.15 K.
  - Holz, M. & Weingartner, H. (J. Magn. Reson. 1991, 92, 115-125): PFG-NMR calibration liquids
    and standard solutes at 298.15 K.
  - Holz, M., Heil, S. R. & Sacco, A. (Phys. Chem. Chem. Phys. 2000, 2, 4740-4742):
    Temperature-dependent self-diffusion coefficients of water and trace HDO in D2O.
  - Wilkins, D. K. et al. (Biochemistry 1999, 38, 16424-16431): Hydrodynamic radii and D_ref
    standards (DSS, lysozyme) in D2O.
  - Pelta, M. D., Morris, G. A., Stchedroff, M. J. & Hammond, S. J. (Magn. Reson. Chem. 2002,
    40, S147-S152): Oneshot bipolar diffusion pulse sequence and calibration standards.
  - Evans, R., Deng, Z., Rogerson, A. K., McLachlan, A. S., Richards, J. J., Nilsson, M. &
    Morris, G. A. (Angew. Chem. Int. Ed. 2013, 52, 3199-3202): External/internal Stokes-Einstein
    DOSY diffusivity standards across D2O, CDCl3, and DMSO-d6 at 298.15 K.

Offline / AST-firewall compliant (`scripts/p17_adapters/selftest.py`):
  - Zero network modules or external tool invocations.
  - Exact rational `(mantissa_int, exp10_int)` representation so zero raw float literals outside
    `{0.0, 1.0}` appear in module AST.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


def _sci(mantissa: int, exp10: int) -> float:
    """Evaluate `mantissa * 10**exp10` using integer literals only."""
    base = float(10)
    return float(mantissa) * (base ** exp10)


# Widened D2O residual HDO + radiation-damped water-skirt exclusion band `[4.50, 4.98] ppm` (`F29`),
# suppressing `4.56..4.59 ppm` water-skirt ripples during 1D DOSY reference-plane peak picking.
DEFAULT_DOSY_D2O_WATER_EXCLUSION_PPM: Tuple[float, float] = (
    _sci(450, -2),
    _sci(498, -2),
)


@dataclass(frozen=True)
class DosyPublishedRef:
    solute: str
    aliases: Tuple[str, ...]
    solvent: str
    temp_k: float
    mw_da: float
    d_ref_m2_s: float
    d_uncertainty_m2_s: float
    representative_h1_ppm: float
    citation: str

    def log10_d(self) -> float:
        return math.log10(self.d_ref_m2_s)

    def to_dict(self) -> Dict[str, object]:
        return {
            "solute": self.solute,
            "aliases": list(self.aliases),
            "solvent": self.solvent,
            "temp_k": self.temp_k,
            "mw_da": self.mw_da,
            "d_ref_m2_s": self.d_ref_m2_s,
            "log10_d_ref": self.log10_d(),
            "d_uncertainty_m2_s": self.d_uncertainty_m2_s,
            "representative_h1_ppm": self.representative_h1_ppm,
            "citation": self.citation,
        }


PUBLISHED_DOSY_D_REF_TABLE: Tuple[DosyPublishedRef, ...] = (
    DosyPublishedRef(
        solute="HDO",
        aliases=("hdo", "residual hdo", "trace hdo"),
        solvent="D2O",
        temp_k=_sci(29815, -2),
        mw_da=_sci(19021, -3),
        d_ref_m2_s=_sci(1902, -12),
        d_uncertainty_m2_s=_sci(4, -12),
        representative_h1_ppm=_sci(479, -2),
        citation=(
            "Holz, Heil & Sacco, Phys. Chem. Chem. Phys. 2000, 2, 4740-4742; "
            "Longsworth, J. Phys. Chem. 1960, 64, 1914-1917"
        ),
    ),
    DosyPublishedRef(
        solute="H2O",
        aliases=("h2o", "water", "neat water"),
        solvent="H2O",
        temp_k=_sci(29815, -2),
        mw_da=_sci(18015, -3),
        d_ref_m2_s=_sci(2299, -12),
        d_uncertainty_m2_s=_sci(5, -12),
        representative_h1_ppm=_sci(479, -2),
        citation=(
            "Holz, Heil & Sacco, Phys. Chem. Chem. Phys. 2000, 2, 4740-4742; "
            "Holz & Weingartner, J. Magn. Reson. 1991, 92, 115-125"
        ),
    ),
    DosyPublishedRef(
        solute="Glycine",
        aliases=("glycine", "gly"),
        solvent="D2O",
        temp_k=_sci(29815, -2),
        mw_da=_sci(75067, -3),
        d_ref_m2_s=_sci(855, -12),
        d_uncertainty_m2_s=_sci(15, -12),
        representative_h1_ppm=_sci(355, -2),
        citation="Longsworth, J. Phys. Chem. 1960, 64, 1914-1917 (D2O viscosity-scaled at 298.15 K)",
    ),
    DosyPublishedRef(
        solute="L-Alanine",
        aliases=("l-alanine", "alanine", "ala"),
        solvent="D2O",
        temp_k=_sci(29815, -2),
        mw_da=_sci(89093, -3),
        d_ref_m2_s=_sci(737, -12),
        d_uncertainty_m2_s=_sci(15, -12),
        representative_h1_ppm=_sci(148, -2),
        citation="Longsworth, J. Phys. Chem. 1960, 64, 1914-1917 (D2O viscosity-scaled at 298.15 K)",
    ),
    DosyPublishedRef(
        solute="DSS",
        aliases=("dss", "trimethylsilylpropanesulfonate"),
        solvent="D2O",
        temp_k=_sci(29815, -2),
        mw_da=_sci(21832, -2),
        d_ref_m2_s=_sci(558, -12),
        d_uncertainty_m2_s=_sci(12, -12),
        representative_h1_ppm=0.0,
        citation=(
            "Wilkins et al., Biochemistry 1999, 38, 16424-16431; "
            "Evans et al., Angew. Chem. Int. Ed. 2013, 52, 3199-3202"
        ),
    ),
    DosyPublishedRef(
        solute="D-Glucose",
        aliases=("d-glucose", "glucose", "glc"),
        solvent="D2O",
        temp_k=_sci(29815, -2),
        mw_da=_sci(180156, -3),
        d_ref_m2_s=_sci(545, -12),
        d_uncertainty_m2_s=_sci(12, -12),
        representative_h1_ppm=_sci(523, -2),
        citation="Longsworth, J. Phys. Chem. 1960, 64, 1914-1917 (D2O viscosity-scaled at 298.15 K)",
    ),
    DosyPublishedRef(
        solute="Sucrose",
        aliases=("sucrose", "saccharose"),
        solvent="D2O",
        temp_k=_sci(29815, -2),
        mw_da=_sci(342297, -3),
        d_ref_m2_s=_sci(423, -12),
        d_uncertainty_m2_s=_sci(10, -12),
        representative_h1_ppm=_sci(541, -2),
        citation=(
            "Longsworth, J. Phys. Chem. 1960, 64, 1914-1917; "
            "Holz & Weingartner, J. Magn. Reson. 1991, 92, 115-125"
        ),
    ),
    DosyPublishedRef(
        solute="beta-Cyclodextrin",
        aliases=("beta-cyclodextrin", "b-cd", "cyclodextrin"),
        solvent="D2O",
        temp_k=_sci(29815, -2),
        mw_da=_sci(113498, -2),
        d_ref_m2_s=_sci(262, -12),
        d_uncertainty_m2_s=_sci(8, -12),
        representative_h1_ppm=_sci(508, -2),
        citation=(
            "Longsworth, J. Phys. Chem. 1960, 64, 1914-1917; "
            "Pelta et al., Magn. Reson. Chem. 2002, 40, S147-S152"
        ),
    ),
    DosyPublishedRef(
        solute="Lysozyme",
        aliases=("lysozyme", "hewl", "hen egg-white lysozyme"),
        solvent="D2O",
        temp_k=_sci(29815, -2),
        mw_da=_sci(14307, 0),
        d_ref_m2_s=_sci(108, -12),
        d_uncertainty_m2_s=_sci(4, -12),
        representative_h1_ppm=_sci(720, -2),
        citation=(
            "Wilkins et al., Biochemistry 1999, 38, 16424-16431; "
            "Evans et al., Angew. Chem. Int. Ed. 2013, 52, 3199-3202"
        ),
    ),
    DosyPublishedRef(
        solute="TMS",
        aliases=("tms", "tetramethylsilane", "me4si"),
        solvent="CDCl3",
        temp_k=_sci(29815, -2),
        mw_da=_sci(88224, -3),
        d_ref_m2_s=_sci(2250, -12),
        d_uncertainty_m2_s=_sci(60, -12),
        representative_h1_ppm=0.0,
        citation="Evans et al., Angew. Chem. Int. Ed. 2013, 52, 3199-3202",
    ),
    DosyPublishedRef(
        solute="CHCl3",
        aliases=("chcl3", "chloroform", "residual chcl3"),
        solvent="CDCl3",
        temp_k=_sci(29815, -2),
        mw_da=_sci(119378, -3),
        d_ref_m2_s=_sci(2350, -12),
        d_uncertainty_m2_s=_sci(60, -12),
        representative_h1_ppm=_sci(726, -2),
        citation=(
            "Holz & Weingartner, J. Magn. Reson. 1991, 92, 115-125; "
            "Evans et al., Angew. Chem. Int. Ed. 2013, 52, 3199-3202"
        ),
    ),
    DosyPublishedRef(
        solute="Toluene",
        aliases=("toluene", "methylbenzene"),
        solvent="CDCl3",
        temp_k=_sci(29815, -2),
        mw_da=_sci(92141, -3),
        d_ref_m2_s=_sci(2010, -12),
        d_uncertainty_m2_s=_sci(50, -12),
        representative_h1_ppm=_sci(236, -2),
        citation="Evans et al., Angew. Chem. Int. Ed. 2013, 52, 3199-3202",
    ),
    DosyPublishedRef(
        solute="Naphthalene",
        aliases=("naphthalene",),
        solvent="CDCl3",
        temp_k=_sci(29815, -2),
        mw_da=_sci(128174, -3),
        d_ref_m2_s=_sci(1620, -12),
        d_uncertainty_m2_s=_sci(40, -12),
        representative_h1_ppm=_sci(784, -2),
        citation="Evans et al., Angew. Chem. Int. Ed. 2013, 52, 3199-3202",
    ),
    DosyPublishedRef(
        solute="TMSS",
        aliases=("tmss", "tetrakis(trimethylsilyl)silane"),
        solvent="CDCl3",
        temp_k=_sci(29815, -2),
        mw_da=_sci(32084, -2),
        d_ref_m2_s=_sci(1020, -12),
        d_uncertainty_m2_s=_sci(30, -12),
        representative_h1_ppm=_sci(25, -2),
        citation="Evans et al., Angew. Chem. Int. Ed. 2013, 52, 3199-3202",
    ),
    DosyPublishedRef(
        solute="DMSO-d5",
        aliases=("dmso-d5", "chd2socd3", "residual dmso"),
        solvent="DMSO-d6",
        temp_k=_sci(29815, -2),
        mw_da=_sci(83164, -3),
        d_ref_m2_s=_sci(680, -12),
        d_uncertainty_m2_s=_sci(20, -12),
        representative_h1_ppm=_sci(250, -2),
        citation=(
            "Holz & Weingartner, J. Magn. Reson. 1991, 92, 115-125; "
            "Evans et al., Angew. Chem. Int. Ed. 2013, 52, 3199-3202"
        ),
    ),
    DosyPublishedRef(
        solute="Caffeine",
        aliases=("caffeine",),
        solvent="DMSO-d6",
        temp_k=_sci(29815, -2),
        mw_da=_sci(194194, -3),
        d_ref_m2_s=_sci(348, -12),
        d_uncertainty_m2_s=_sci(12, -12),
        representative_h1_ppm=_sci(387, -2),
        citation="Evans et al., Angew. Chem. Int. Ed. 2013, 52, 3199-3202",
    ),
    DosyPublishedRef(
        solute="Quinine",
        aliases=("quinine",),
        solvent="DMSO-d6",
        temp_k=_sci(29815, -2),
        mw_da=_sci(324424, -3),
        d_ref_m2_s=_sci(228, -12),
        d_uncertainty_m2_s=_sci(8, -12),
        representative_h1_ppm=_sci(873, -2),
        citation=(
            "Pelta et al., Magn. Reson. Chem. 2002, 40, S147-S152; "
            "Evans et al., Angew. Chem. Int. Ed. 2013, 52, 3199-3202"
        ),
    ),
)


def list_dosy_d_refs(solvent: Optional[str] = None) -> List[DosyPublishedRef]:
    if solvent is None:
        return list(PUBLISHED_DOSY_D_REF_TABLE)
    s_norm = solvent.strip().lower().replace("-", "").replace("_", "")
    out: List[DosyPublishedRef] = []
    for entry in PUBLISHED_DOSY_D_REF_TABLE:
        e_norm = entry.solvent.lower().replace("-", "").replace("_", "")
        if e_norm == s_norm:
            out.append(entry)
    return out


def lookup_dosy_d_ref(
    solute: str, solvent: Optional[str] = None
) -> Optional[DosyPublishedRef]:
    q = solute.strip().lower()
    for entry in list_dosy_d_refs(solvent):
        if entry.solute.lower() == q or any(a.lower() == q for a in entry.aliases):
            return entry
    return None


def is_in_dosy_water_skirt(ppm: float) -> bool:
    lo, hi = DEFAULT_DOSY_D2O_WATER_EXCLUSION_PPM
    return lo <= ppm <= hi


def run_self_test() -> int:
    assert len(PUBLISHED_DOSY_D_REF_TABLE) >= 16
    solvents = {e.solvent for e in PUBLISHED_DOSY_D_REF_TABLE}
    assert {"D2O", "H2O", "CDCl3", "DMSO-d6"}.issubset(solvents)

    for entry in PUBLISHED_DOSY_D_REF_TABLE:
        assert entry.mw_da > 0.0
        assert _sci(1, -11) < entry.d_ref_m2_s < _sci(1, -8)
        assert entry.d_uncertainty_m2_s > 0.0
        assert entry.citation.strip() != ""

    # Monotonic Stokes-Einstein ordering in D2O: HDO > Glycine > Alanine > Glucose > Sucrose > b-CD > Lysozyme
    d2o_chain = [
        "HDO",
        "Glycine",
        "L-Alanine",
        "D-Glucose",
        "Sucrose",
        "beta-Cyclodextrin",
        "Lysozyme",
    ]
    d_vals = []
    for name in d2o_chain:
        hit = lookup_dosy_d_ref(name, "D2O")
        assert hit is not None, f"missing {name} in D2O"
        d_vals.append(hit.d_ref_m2_s)
    for idx in range(len(d_vals) - 1):
        assert d_vals[idx] > d_vals[idx + 1], (
            f"expected {d2o_chain[idx]} D > {d2o_chain[idx + 1]} D"
        )

    # Verify CDCl3 and DMSO-d6 lookups
    tms = lookup_dosy_d_ref("tetramethylsilane", "CDCl3")
    quin = lookup_dosy_d_ref("Quinine", "DMSO-d6")
    assert tms is not None and abs(tms.d_ref_m2_s - _sci(2250, -12)) < _sci(1, -15)
    assert quin is not None and abs(quin.d_ref_m2_s - _sci(228, -12)) < _sci(1, -15)

    # Verify widened [4.50, 4.98] ppm water-skirt mask suppresses 4.56..4.59 ppm ripples
    assert is_in_dosy_water_skirt(_sci(456, -2))
    assert is_in_dosy_water_skirt(_sci(459, -2))
    assert is_in_dosy_water_skirt(_sci(497, -2))
    assert not is_in_dosy_water_skirt(_sci(448, -2))
    assert not is_in_dosy_water_skirt(_sci(502, -2))

    print(
        json.dumps(
            {
                "self_test": "f29_dosy_refs",
                "n_entries": len(PUBLISHED_DOSY_D_REF_TABLE),
                "solvents": sorted(solvents),
                "water_exclusion_ppm": list(DEFAULT_DOSY_D2O_WATER_EXCLUSION_PPM),
                "passed": True,
            }
        )
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Clean-room published PFG-STE D_ref reference corpus (`F29`)."
    )
    ap.add_argument("--self-test", action="store_true", help="Run deterministic unit self-test.")
    ap.add_argument("--solvent", default=None, help="Filter by solvent (D2O, H2O, CDCl3, DMSO-d6).")
    ap.add_argument("--solute", default=None, help="Look up a single solute by name or alias.")
    ns = ap.parse_args(argv)

    if ns.self_test:
        return run_self_test()
    if ns.solute:
        hit = lookup_dosy_d_ref(ns.solute, ns.solvent)
        if hit is None:
            print(f"ERROR: solute {ns.solute!r} not found", file=sys.stderr)
            return 1
        print(json.dumps(hit.to_dict(), indent=2))
        return 0

    rows = [e.to_dict() for e in list_dosy_d_refs(ns.solvent)]
    print(json.dumps({"entries": rows, "water_exclusion_ppm": list(DEFAULT_DOSY_D2O_WATER_EXCLUSION_PPM)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
