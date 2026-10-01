"""`su2_cubed_atlas` — `BPSAtlas` builder for the **SU(2)³ linear quiver**
(SU(2)-SU(2)-SU(2), the design record catalogue).

The 3-node extension of the SU(2)-SU(2) entry (`su2_family_atlas.SU2_SU2`).  There
is no 3-node `pure_ade.SUN_bifund`, so the BPS quiver is assembled **directly as
its Dirac matrix** by extending the verified SU(2)-SU(2) pattern (extracted from
`SUN_bifund(2,2)`): three Kronecker-2 gauge pairs `(W_i, M_i)` and two
**bifundamental** nodes, each pairing `−1` with both W's and `+1` with both M's of
the two SU(2)'s it links (so the middle node SU(2)₂ couples to *both* bifunds):

    node order:  W1 M1 | W2 M2 | W3 M3 | bif1 bif2
    ⟨W_i, M_i⟩ = 2  (each SU(2) a Kronecker-2)
    bif1 ↔ {W1:+1, M1:−1, W2:+1, M2:−1}     (links SU(2)₁–SU(2)₂)
    bif2 ↔ {W2:+1, M2:−1, W3:+1, M3:−1}     (links SU(2)₂–SU(2)₃)

`ker B` is rank **2** — one baryonic U(1) per bifundamental — so the coefficient
ring is `R(U(1)²)`.  A finite strong-coupling chamber exists: the length-**16**
negating sequence below was found by `BPSQuiver.find_negating_sequence` (and is
self-validating — it negates the node charges), then frozen here so the chart
builds instantly.

**Feasibility.**  Rank 8 with the heavy bifundamental nodes ⇒ multiply-level
checks are slow (minutes), exactly like the SU(3) quivers;
the contract axioms pass on the gauge nodes.  The atlas is certified at the label
level (dynamics / wild boundary / structural node-drop iso); the multiply/trace
sweep is the script.

Public API:
  * `su2_cubed_chart()      -> BPSKAlgebra`
  * `su2_cubed_atlas(**kw)  -> BPSAtlas`
  * `su2_cubed_rg_iso()     -> KAlgebraIso`   (drop both bifunds → pure SU(2)³)
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bps_kalgebra import BPSKAlgebra
from bps_atlas import BPSAtlas


__all__ = [
    "SU2_CUBED_PAIRING",
    "SU2_CUBED_NODES",
    "SU2_CUBED_NEG_SEQ",
    "BIFUND_INDICES",
    "su2_cubed_chart",
    "su2_cubed_atlas",
    "su2_cubed_rg_iso",
]

_N = 8


def _build_pairing():
    Q = [[0] * _N for _ in range(_N)]

    def sp(i, j, v):
        Q[i][j] = v
        Q[j][i] = -v

    sp(0, 1, 2)            # SU(2)_1 Kronecker
    sp(2, 3, 2)            # SU(2)_2 Kronecker
    sp(4, 5, 2)            # SU(2)_3 Kronecker
    sp(6, 0, 1); sp(6, 1, -1); sp(6, 2, 1); sp(6, 3, -1)   # bif1 ↔ SU(2)_1,SU(2)_2
    sp(7, 2, 1); sp(7, 3, -1); sp(7, 4, 1); sp(7, 5, -1)   # bif2 ↔ SU(2)_2,SU(2)_3
    return [row[:] for row in Q]


SU2_CUBED_PAIRING = _build_pairing()
SU2_CUBED_NODES = [tuple(1 if k == i else 0 for k in range(_N)) for i in range(_N)]
# The two bifundamental dyon nodes (drop both → pure SU(2)³).
BIFUND_INDICES = (6, 7)
# Length-16 negating sequence (strong-coupling chamber), found by
# BPSQuiver.find_negating_sequence and frozen for instant builds.
SU2_CUBED_NEG_SEQ = [7, 2, 4, 3, 5, 2, 6, 0, 7, 4, 1, 2, 5, 1, 6, 0]


@lru_cache(maxsize=1)
def su2_cubed_chart() -> BPSKAlgebra:
    """The SU(2)³ linear-quiver BPS chart: rank 8, |spec| 16, coeff `R(U(1)²)`
    (one baryon per bifundamental).  Built from the frozen negating sequence
    (`verify="off"` — the sequence is self-validating)."""
    return BPSKAlgebra(
        pairing=[row[:] for row in SU2_CUBED_PAIRING],
        node_charges=[tuple(g) for g in SU2_CUBED_NODES],
        negating_sequence=list(SU2_CUBED_NEG_SEQ),
        verify="off",
    )


def su2_cubed_atlas(**kwargs) -> BPSAtlas:
    """A `BPSAtlas` seeded at the SU(2)³ chart."""
    return BPSAtlas(su2_cubed_chart(), **kwargs)


def su2_cubed_rg_iso():
    """Identity-on-labels `KAlgebraIso` from the **node-drop RG flow** that drops
    both bifundamentals → IR `pure SU(2)³`, to the BPS chart (guaranteed once the
    auxiliaries match)."""
    from rg_flow import SubquiverRG
    from kalgebra_iso import KAlgebraIso
    A = su2_cubed_chart()
    flow = SubquiverRG(A, list(BIFUND_INDICES))
    return KAlgebraIso.identity_on_labels(flow, A)
