"""`gauged_ad_atlas` — `BPSAtlas` builders for the **U(1)-gauged AD** theories
(u1a1d4/6/8, u1e7), the design record Stage-4 catalogue.

**The key:** a gauged theory has the **same BPS quiver as the
ungauged one, with a larger Γ** — "as usual."  Gauging a flavour U(1) does not
change the BPS *quiver* (the Dirac pairing among the BPS states); it promotes that
U(1) from a flavour direction to a **gauge** direction, enlarging the charge
lattice Γ.  So each U(1)-gauged AD theory **shares the ungauged AD theory's BPS
chart**, and its `BPSAtlas` chart graph is the ungauged one's:

  * **u1a1d4 / u1a1d6 / u1a1d8** = u(1)-gauged [A₁,D₄/D₆/D₈] → the **a1d4/6/8**
    quiver (catalogue I), with the U(1) flavour gauged (larger Γ);
  * **u1e7** = u(1)-gauged E₇ → the **E₇ = [A₁,E₇]** quiver (`E7_BPS_PAIRING`).

`BPSAtlas(chart).mutation_complete()` therefore closes the same finite folded
cluster graph as the ungauged AD theory — the gauge enlargement of Γ is a
coefficient/grading refinement, not a new chart graph:

    u1a1d4 → 10   u1a1d6 → 80   u1a1d8 → 810   u1e7 → 416   (folded charts)

This resolves "build BPSAtlas for the gauged-AD theories": there is no separate
gauged-theory quiver to complete (the earlier "no native gauged quiver" read was
the wrong frame) — the gauged theory *is* the ungauged AD quiver with the gauge Γ,
so its atlas is built from that quiver directly.  The flow-composition cone test
(`atlas_catalogue_gauged_ad.py`) remains the *matter-content* cross-check; this
module is the *chart-graph* (atlas) side.

Public API (mirrors `a1dn_atlas` / `su2_gauged_a1dn_atlas`):
  * `gauged_ad_names() -> list[str]`            (`"u1a1d4","u1a1d6","u1a1d8","u1e7"`)
  * `gauged_ad_entry(name) -> GaugedADEntry`
  * `gauged_ad_charts() -> dict[str, BPSKAlgebra]`
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import lru_cache

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bps_kalgebra import BPSKAlgebra


__all__ = [
    "GaugedADEntry",
    "gauged_ad_names",
    "gauged_ad_entry",
    "gauged_ad_charts",
]

# name -> (the ungauged AD quiver source, the gauged-Γ description, expected
# folded-chart count from mutation_complete).
#   u1a1dn = u(1)-gauged [A1,Dn] -> the [A1,Dn] (a1dn) quiver, larger Γ.
#   u1e7   = u(1)-gauged E7      -> the [A1,E7] (e7) quiver, larger Γ.
_FAMILY = {
    "u1a1d4": ("a1d4", "u(1)-gauged [A₁,D₄]", 10),
    "u1a1d6": ("a1d6", "u(1)-gauged [A₁,D₆]", 80),
    "u1a1d8": ("a1d8", "u(1)-gauged [A₁,D₈]", 810),
    "u1e7":   ("e7",   "u(1)-gauged E₇ (= [A₁,E₇])", 416),
}


def _ad_chart(source: str) -> BPSKAlgebra:
    """The ungauged AD BPS chart for `source` ('a1dₙ' or 'e7'), built from its
    embedded quiver literal — the quiver the gauged theory shares."""
    if source.startswith("a1d"):
        mod = __import__(f"finite_{source}_kalg")
        P = getattr(mod, f"A1D{source[len('a1d'):]}_BPS_PAIRING")
        N = getattr(mod, f"A1D{source[len('a1d'):]}_BPS_NODE_CHARGES")
    else:                                   # e6/e7/e8
        mod = __import__(f"finite_{source}_kalg")
        P = getattr(mod, f"{source.upper()}_BPS_PAIRING")
        N = getattr(mod, f"{source.upper()}_BPS_NODE_CHARGES")
    return BPSKAlgebra(pairing=[list(r) for r in P],
                       node_charges=[tuple(g) for g in N], verify="off")


@dataclass
class GaugedADEntry:
    """One U(1)-gauged AD theory: its BPS chart (= the ungauged AD quiver, larger
    Γ) + atlas metadata."""
    name: str
    source: str                 # the ungauged AD quiver it shares ('a1dₙ'/'e7')
    chart: BPSKAlgebra
    description: str
    folded_charts: int          # mutation_complete folded-chart count
    note: str = ""

    @property
    def rank(self) -> int:
        return len(self.chart.lattice.pairing)


@lru_cache(maxsize=None)
def _entry(name: str) -> GaugedADEntry:
    if name not in _FAMILY:
        raise KeyError(f"unknown gauged-AD theory {name!r}; "
                       f"have {list(_FAMILY)}")
    source, desc, folded = _FAMILY[name]
    chart = _ad_chart(source)
    return GaugedADEntry(
        name=name, source=source, chart=chart, description=desc,
        folded_charts=folded,
        note=f"{desc}: same BPS quiver as the ungauged {source} (rank {len(chart.lattice.pairing)}), "
             f"larger Γ (the gauged U(1)); mutation_complete folds to {folded} charts")


def gauged_ad_names() -> list[str]:
    """The gauged-AD family in canonical order."""
    return list(_FAMILY)


def gauged_ad_entry(name) -> GaugedADEntry:
    """The `GaugedADEntry` for `name` (e.g. `"u1e7"`)."""
    return _entry(name)


def gauged_ad_charts() -> dict:
    """All gauged-AD BPS charts keyed by name (= the shared ungauged AD quivers)."""
    return {nm: _entry(nm).chart for nm in _FAMILY}
