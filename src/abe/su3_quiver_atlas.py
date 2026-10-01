"""`su3_quiver_atlas` — `BPSAtlas` builders for the SU(3) **linear quivers**.

Two-node gauge quivers with one bifundamental hypermultiplet, built from the same
`pure_ade.SUN_bifund` machinery the SU(2) family used for SU(2)-SU(2)
(`su2_family_atlas.py`), now with an SU(3) node:

    SU(3)-SU(2)     pure_ade.SUN_bifund(3, 2)
    SU(3)-SU(3)     pure_ade.SUN_bifund(3, 3)

`SUN_bifund` composes two `SUN_Nf` pure-gauge factors and prepends a single
**bifundamental** matter node (the bifund block contributes `(2N₁−1)(2N₂−1)`
ordered spec factors).  Both quivers are asymptotically free (the builder enforces
`N₂ ≤ 2N₁` and `N₁ ≤ 2N₂`).  Verified structure:

| quiver | source | rank | \|spec\| | coeff ring |
|---|---|--:|--:|---|
| SU(3)-SU(2) | `SUN_bifund(3, 2)` | 7 | 23 | `R(U(1))` (bifund baryon) |
| SU(3)-SU(3) | `SUN_bifund(3, 3)` | 9 | 37 | `R(U(1))` (bifund baryon) |

The single flavour direction (`ker B`) is the bifundamental **baryonic U(1)**, as
for SU(2)-SU(2).  The spec is recipe-valid (`SUN_bifund.verify()` =
`verify_spectrum_generator`), so charts are built `verify="off"`.

**Feasibility (the build wall, cf. `su3_family_atlas.py`).**  SU(3)-SU(2) (rank 7)
builds in seconds; **SU(3)-SU(3) (rank 9) is at the wall — its BPS-chart
construction is many minutes** (the σ half-monodromy derivation at high rank), and
the trace/dynamics-walk certification is correspondingly heavy.  Both charts are
constructible; the SU(3)-SU(3) certification is necessarily lighter (the RG
bifund-drop iso is structurally guaranteed; the dynamics walk is left bounded).

The matter node is the single bifundamental dyon at index `n_pure1 + n_pure2`;
dropping it is the **node-drop RG flow** to the IR `pure SU(N₁) × pure SU(N₂)`
(`rg_flow.SubquiverRG`), with the identity-on-labels `KAlgebraIso` to the BPS
chart guaranteed once the auxiliaries match.

Public API:
  * `su3_quiver_names()        -> list[str]`
  * `su3_quiver_entry(name)    -> SU3QuiverEntry`
  * `su3_quiver_charts()       -> dict[str, BPSKAlgebra]`
  * `su3_quiver_rg_nodedrop(name) -> SubquiverRG`
  * `su3_quiver_rg_iso(name)   -> KAlgebraIso`
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

import pure_ade as _pa
from bps_kalgebra import BPSKAlgebra


__all__ = [
    "SU3QuiverEntry",
    "su3_quiver_names",
    "su3_quiver_entry",
    "su3_quiver_charts",
    "su3_quiver_rg_nodedrop",
    "su3_quiver_rg_iso",
]

# name -> (N1, N2) gauge ranks
_QUIVERS = {
    "SU3_SU2": (3, 2),
    "SU3_SU3": (3, 3),
}


def _bps_from_pure_ade(builder) -> BPSKAlgebra:
    if not builder.verify():
        raise RuntimeError(
            f"{builder!r}: assembled spec is not a valid negating sequence "
            f"(pure_ade recipe bug — should not happen)")
    return BPSKAlgebra(
        pairing=[list(r) for r in builder.B],
        node_charges=[tuple(g) for g in builder.nodes],
        spec=[tuple(g) for g in builder.spec],
        verify="off",
    )


@dataclass
class SU3QuiverEntry:
    """One SU(3) linear quiver: its canonical BPS chart + atlas metadata."""
    name: str
    gauge_ranks: tuple[int, int]
    chart: BPSKAlgebra
    matter_indices: tuple[int, ...]      # the single bifundamental dyon node
    pure_ir: str
    note: str = ""

    @property
    def rank(self) -> int:
        return len(self.chart.lattice.pairing)


@lru_cache(maxsize=None)
def _entry(name: str) -> SU3QuiverEntry:
    if name not in _QUIVERS:
        raise KeyError(f"unknown SU(3) quiver {name!r}; have {list(_QUIVERS)}")
    N1, N2 = _QUIVERS[name]
    s = _pa.SUN_bifund(N1, N2)
    n_pure = len(s.pure1.nodes) + len(s.pure2.nodes)
    matter = (n_pure,)                               # the single bifundamental node
    return SU3QuiverEntry(
        name=name, gauge_ranks=(N1, N2), chart=_bps_from_pure_ade(s),
        matter_indices=matter,
        pure_ir=f"pure SU({N1}) x pure SU({N2})  (drop the bifundamental)",
        note=f"pure_ade.SUN_bifund({N1},{N2}); spec = bifund block · two pure-gauge specs")


def su3_quiver_names() -> list[str]:
    """The SU(3) quivers in canonical order."""
    return ["SU3_SU2", "SU3_SU3"]


def su3_quiver_entry(name: str) -> SU3QuiverEntry:
    """The `SU3QuiverEntry` for `name` (`"SU3_SU2"` or `"SU3_SU3"`)."""
    return _entry(name)


def su3_quiver_charts() -> dict:
    """Both quiver charts keyed by name (built lazily, cached)."""
    return {nm: su3_quiver_entry(nm).chart for nm in su3_quiver_names()}


def su3_quiver_rg_nodedrop(name: str):
    """The node-drop RG flow `SU(N₁)-SU(N₂) → pure SU(N₁) × pure SU(N₂)` (drop the
    bifundamental dyon)."""
    from rg_flow import SubquiverRG
    e = su3_quiver_entry(name)
    return SubquiverRG(e.chart, list(e.matter_indices))


def su3_quiver_rg_iso(name: str):
    """The identity-on-labels `KAlgebraIso` from the node-drop RG presentation to
    the BPS chart — guaranteed once the auxiliaries match (light battery)."""
    from kalgebra_iso import KAlgebraIso
    rg = su3_quiver_rg_nodedrop(name)
    bps = su3_quiver_entry(name).chart
    return KAlgebraIso.identity_on_labels(rg, bps)
