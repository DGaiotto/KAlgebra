"""bps_su2_nf1 — canonical `BPSKAlgebra` for SU(2) + N_f = 1.

**Auxiliary** reference oracle for `su2_nf1_kalgebra.SU2Nf1KAlgebra`.
Uses the canonical-surface `BPSKAlgebra` (Plan 07, repo root); the
coefficient ring is `AbelianZPlusRing(rank=1)` because `ker(B)` is
the third lattice direction (γ_3 is the U(1)_F flavour direction).

Pairing / nodes / spec (canonical 5-step strong-coupling chamber of
SU(2) + 1 fundamental hyper — 3 matter dyons then the pure-SU(2)
monopole / dyon):

    pairing B =  [[0,  1, 0],
                  [-1, 0, 0],
                  [0,  0, 0]]

    nodes  = [(1, 0, 0), (-1, 2, 0), (0, -1, 1)]

    spec   = [(0, -1, 1), (-1, 1, 1), (0, 1, 1), (1, 0, 0), (-1, 2, 0)]
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from bps_kalgebra import BPSKAlgebra


SU2_NF1_PAIRING = [
    [0, 1, 0],
    [-1, 0, 0],
    [0, 0, 0],
]

SU2_NF1_NODE_CHARGES = [
    (1, 0, 0),
    (-1, 2, 0),
    (0, -1, 1),
]

SU2_NF1_SPEC = [
    (0, -1, 1),
    (-1, 1, 1),
    (0, 1, 1),
    (1, 0, 0),
    (-1, 2, 0),
]


def build_bps_su2_nf1() -> BPSKAlgebra:
    """Construct the canonical auxiliary `BPSKAlgebra` for
    `A_𝖖[SU(2) + N_f = 1]`.

    `coefficient_ring()` is `AbelianZPlusRing(rank=1)` (U(1)_F).
    """
    return BPSKAlgebra(
        pairing=SU2_NF1_PAIRING,
        node_charges=SU2_NF1_NODE_CHARGES,
        spec=SU2_NF1_SPEC,
    )
