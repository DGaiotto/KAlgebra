"""`su2_gauged_a1dn_atlas` — `BPSAtlas` builders for the **SU(2)-gauged [A₁,Dₙ]**
family (the design record catalogue).

Gauging the SU(2) flavour of the [A₁, Dₙ] Argyres–Douglas theory: the two short-leg (fork) nodes of Dₙ are replaced by a single matter
node coupled to a pure-SU(2) **Kronecker** quiver (like an N_f=1 fundamental),
with the Dₙ stem (an A_{n−2} chain) attached by one edge and a U(1) **tail** node
gauging the baryonic U(1).  The constructors live in `su2a1d3_gauged.py`
(`su2_gauged_a1dn(n)`, `su2_gauged_a1dn_drop(n, j)`); this module wraps them as a
uniform atlas family.

Verified structure: `rank = n + 1`, `|spec| = n + 2` (a **short** spec — these are
finite-chamber theories, so the atlas certifies *fully*, unlike the heavy SU(3)
quivers).  The flavour ring alternates with the Dₙ leaf parity:

  * **odd n** → `TrivialZPlusRing` (flavourless — the SU(2) and the baryonic U(1)
    are both gauged; an interacting flavourless SCFT);
  * **even n** → `AbelianZPlusRing(rank=1)` (one surviving U(1)).

Two single-node-drop RG flows out of each theory (`su2_gauged_a1dn_drop`):
  * **matter drop** (node 2) → pure SU(2) × (Dₙ-stem theory);
  * **tail drop** (last node) → the U(1)-ungauged theory (for n=3, U(2) N_f=1).

Both are `SingleNodeRG` flows whose identity-on-labels `KAlgebraIso` to the BPS
chart is guaranteed once the auxiliaries match.

Public API:
  * `su2_gauged_a1dn_names(max_n=MAX_N) -> list[str]`   (`"SU2gA1D3" ..`)
  * `su2_gauged_a1dn_entry(name|n)     -> SU2gA1DnEntry`
  * `su2_gauged_a1dn_charts(max_n)     -> dict[str, BPSKAlgebra]`
  * `su2_gauged_a1dn_rg_iso(name|n, which) -> KAlgebraIso`  (which ∈ {"matter","tail"})
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
from su2a1d3_gauged import su2_gauged_a1dn, su2_gauged_a1dn_drop


__all__ = [
    "SU2gA1DnEntry",
    "MAX_N",
    "su2_gauged_a1dn_names",
    "su2_gauged_a1dn_entry",
    "su2_gauged_a1dn_charts",
    "su2_gauged_a1dn_rg_iso",
]

# Default family ceiling.  Builds + full certification stay cheap to well beyond
# this (short specs, rank n+1); 8 is a comfortable, well-tested span.
MAX_N = 8
MIN_N = 3


def _n_of(name) -> int:
    if isinstance(name, int):
        return name
    if not name.startswith("SU2gA1D"):
        raise KeyError(f"unknown SU(2)-gauged-A1Dn theory {name!r}")
    return int(name[len("SU2gA1D"):])


@dataclass
class SU2gA1DnEntry:
    """One SU(2)-gauged [A₁,Dₙ] theory: its BPS chart + atlas metadata."""
    name: str
    n: int
    chart: BPSKAlgebra
    matter_drop_index: int      # the matter node (2)
    tail_drop_index: int        # the U(1) tail node (last)
    flavourless: bool
    note: str = ""

    @property
    def rank(self) -> int:
        return len(self.chart.lattice.pairing)


@lru_cache(maxsize=None)
def _entry(n: int) -> SU2gA1DnEntry:
    if n < MIN_N:
        raise ValueError(f"SU(2)-gauged [A1,Dn] needs n >= {MIN_N}, got {n}")
    A = su2_gauged_a1dn(n)
    rank = len(A.lattice.pairing)
    from zplus_ring import TrivialZPlusRing
    flavourless = isinstance(A.coefficient_ring(), TrivialZPlusRing)
    return SU2gA1DnEntry(
        name=f"SU2gA1D{n}", n=n, chart=A,
        matter_drop_index=2, tail_drop_index=rank - 1,
        flavourless=flavourless,
        note=f"SU(2)-gauged [A1,D{n}] (+tail); rank {rank}, |spec| {len(A.spec)}; "
             + ("flavourless (Z)" if flavourless else "one U(1) (even n)"))


def su2_gauged_a1dn_names(max_n: int = MAX_N) -> list[str]:
    """The family `SU2gA1D3 .. SU2gA1D{max_n}` in canonical order."""
    return [f"SU2gA1D{n}" for n in range(MIN_N, max_n + 1)]


def su2_gauged_a1dn_entry(name) -> SU2gA1DnEntry:
    """The `SU2gA1DnEntry` for `name` (e.g. `"SU2gA1D5"`) or an int `n`."""
    return _entry(_n_of(name))


def su2_gauged_a1dn_charts(max_n: int = MAX_N) -> dict:
    """All family charts keyed by name (built lazily, cached)."""
    return {nm: su2_gauged_a1dn_entry(nm).chart
            for nm in su2_gauged_a1dn_names(max_n)}


def su2_gauged_a1dn_rg_iso(name, which: str = "matter"):
    """Identity-on-labels `KAlgebraIso` from a single-node-drop RG flow to the BPS
    chart.  `which` ∈ {"matter", "tail"} selects the dropped node (matter node 2
    → pure SU(2) × stem; tail node → the U(1)-ungauged theory)."""
    from kalgebra_iso import KAlgebraIso
    e = su2_gauged_a1dn_entry(name)
    j = e.matter_drop_index if which == "matter" else e.tail_drop_index
    flow = su2_gauged_a1dn_drop(e.n, j, e.chart)
    return KAlgebraIso.identity_on_labels(flow, e.chart)
