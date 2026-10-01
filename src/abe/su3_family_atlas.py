"""`su3_family_atlas` — the SU(3)+N_f `BPSAtlas` builders (the design record catalogue).

The SU(3) gauge-theory family, one theory per `N_f`, each presented as a
canonical-surface `BPSKAlgebra` from the **known BPS quiver + (matter)(pure
gauge) spec recipe** — exactly as the SU(2) family
(`su2_family_atlas.py`), with the pure-gauge core now **pure SU(3)**
(`pure_ade.PureADE([("A", 2)])`, see `pure_su3_atlas.py`):

    SU(3) + N_f = 1 .. 6     pure_ade.SUN_Nf(3, N_f)
                             (N_f matter dyon blocks · pure-SU(3) spec)

SU(3) is asymptotically free for `N_f = 1..5` and **conformal at N_f=6**
(`β ∝ 2·3 − N_f`).  Structure (verified): `rank = 4 + N_f`,
`|spec| = 6 + 5·N_f`, `n_matter = N_f`, coefficient ring
`AbelianZPlusRing(rank = N_f)` (the Cartan torus of the flavour symmetry,
`ker B`).  The spec is recipe-valid by construction
(`SUN_Nf.verify()` = `verify_spectrum_generator`), so charts are built
`verify="off"`.

**Feasibility (the build wall).**  `SUN_Nf(3, N_f)` and its `.verify()` are
instant at every `N_f`, but the canonical-surface `BPSKAlgebra` *construction*
(the σ half-monodromy derivation from the length-`6+5·N_f` spec) grows steeply
with rank: `N_f=3` (rank 7) ≈ 4 s, `N_f=4` (rank 8) ≈ 40 s, `N_f≥5` (rank ≥ 9)
runs to many minutes.  So `MAX_NF = 4` is the **default family** (`su3_family_names`);
`su3_family_entry("SU3_Nf5"/"SU3_Nf6")` still *constructs* (the recipe is valid —
SU(3) exists through the conformal `N_f=6`), but expect a multi-minute build.  The
wall is the BPS-chart construction, not the physics.

Each entry records `matter_indices` — the N_f matter dyon node positions; dropping
them is the **node-drop RG flow** to the pure-SU(3) IR
(`rg_flow.SubquiverRG`), whose identity-on-labels `KAlgebraIso` to the BPS chart
is *guaranteed* once the auxiliaries match, so only the light structural battery is needed.

**Wild chambers** (cf. `pure_su3_atlas.py`): like pure SU(3), the flavoured
charts have wild BPS chambers off the necklace-rotation orbit; the spec-based
chart graph declines those moves (`max_local_moves=0`), keeping the atlas in a
finite chamber.

Public API:
  * `su3_family_names()        -> list[str]`
  * `su3_family_entry(name)    -> SU3FamilyEntry`
  * `su3_family_charts()       -> dict[str, BPSKAlgebra]`
  * `su3_family_rg_nodedrop(name) -> SubquiverRG`
  * `su3_family_rg_iso(name)   -> KAlgebraIso`
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pure_ade as _pa
from bps_kalgebra import BPSKAlgebra


__all__ = [
    "SU3FamilyEntry",
    "MAX_NF",
    "su3_family_names",
    "su3_family_entry",
    "su3_family_charts",
    "su3_family_rg_nodedrop",
    "su3_family_rg_iso",
]

# Default family range.  SU(3) is AF for N_f=1..5, conformal at N_f=6, but the
# BPS-chart build is only feasible (seconds–minute) through N_f=4 (rank 8);
# N_f=5,6 (rank 9,10) are constructible on demand but take many minutes.
MAX_NF = 4


def _bps_from_pure_ade(builder) -> BPSKAlgebra:
    """Transcribe a `pure_ade.SUN_Nf` BPS-data object into a canonical
    `BPSKAlgebra` (same `.B/.nodes/.spec`; spec recipe-valid ⇒ `verify="off"`)."""
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
class SU3FamilyEntry:
    """One SU(3)+N_f theory: its canonical BPS chart + atlas metadata."""
    name: str
    nf: int
    chart: BPSKAlgebra
    matter_indices: tuple[int, ...]      # the N_f matter dyon node positions
    pure_ir: str
    conformal: bool
    note: str = ""

    @property
    def rank(self) -> int:
        return len(self.chart.lattice.pairing)


@lru_cache(maxsize=None)
def _entry(nf: int) -> SU3FamilyEntry:
    s = _pa.SUN_Nf(3, nf)
    n_pure = len(s.pure.nodes)                       # 4 for pure SU(3)
    matter = tuple(range(n_pure, len(s.nodes)))      # the N_f matter dyon nodes
    conformal = (nf == 2 * 3)                         # β ∝ 2N − N_f
    return SU3FamilyEntry(
        name=f"SU3_Nf{nf}", nf=nf, chart=_bps_from_pure_ade(s),
        matter_indices=matter,
        pure_ir="pure SU(3)  (drop the N_f matter dyons)",
        conformal=conformal,
        note=f"pure_ade.SUN_Nf(3,{nf}); spec = {nf} matter blocks · pure-SU(3) spec"
             + ("  [conformal: β=0]" if conformal else ""))


def su3_family_names() -> list[str]:
    """The family in canonical order, `SU3_Nf1 .. SU3_Nf{MAX_NF}`."""
    return [f"SU3_Nf{nf}" for nf in range(1, MAX_NF + 1)]


def _nf_of(name: str) -> int:
    if not name.startswith("SU3_Nf"):
        raise KeyError(f"unknown SU(3)-family theory {name!r}")
    return int(name[len("SU3_Nf"):])


def su3_family_entry(name: str) -> SU3FamilyEntry:
    """The `SU3FamilyEntry` for `name` (e.g. `"SU3_Nf2"`)."""
    return _entry(_nf_of(name))


def su3_family_charts() -> dict:
    """All family charts keyed by name (built lazily, cached)."""
    return {nm: su3_family_entry(nm).chart for nm in su3_family_names()}


def su3_family_rg_nodedrop(name: str):
    """The node-drop RG flow `SU(3)+N_f → pure SU(3)` (drop the matter dyons)."""
    from rg_flow import SubquiverRG
    e = su3_family_entry(name)
    return SubquiverRG(e.chart, list(e.matter_indices))


def su3_family_rg_iso(name: str):
    """The identity-on-labels `KAlgebraIso` from the node-drop RG presentation to
    the BPS chart — guaranteed once the auxiliaries match (light battery)."""
    from kalgebra_iso import KAlgebraIso
    rg = su3_family_rg_nodedrop(name)
    bps = su3_family_entry(name).chart
    return KAlgebraIso.identity_on_labels(rg, bps)
