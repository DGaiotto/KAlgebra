"""bps_su2_nf2 — canonical `BPSKAlgebra` for SU(2) + N_f = 2.

**Spin(4)-manifest chart** (from ClusterApplet mutation γ_3 fwd → γ_4 fwd
→ γ_2 fwd on the strong-coupling 8-state chamber, then flavour basis
relabelled so that SU(2)_L × SU(2)_R Weyl reflections act independently
on the third (m_L) and fourth (m_R) lattice coordinates).

Pairing / node charges / spec:

    pairing B_lat =  [[0,  1, 0, 0],
                       [-1, 0, 0, 0],
                       [0,  0, 0, 0],
                       [0,  0, 0, 0]]
    node_charges = [γ_1, γ_2, γ_3, γ_4] where
        γ_1 = (1, 0, +1, 0)   ─┐
        γ_2 = (1, 0, -1, 0)   ─┴── SU(2)_L doublet (Weyl: m_L → -m_L)
        γ_3 = (-1, 1, 0, +1)  ─┐
        γ_4 = (-1, 1, 0, -1)  ─┴── SU(2)_R doublet (Weyl: m_R → -m_R)

    spec = [γ_1, γ_2, γ_3, γ_4]   (4 BPS states, two doublets)

`coefficient_ring()` = `AbelianZPlusRing(rank=2)` (Cartan torus of
Spin(4) = SU(2)_L × SU(2)_R, with generators μ_L, μ_R).

Z₂ × Z₂ Weyl symmetry:
  * SU(2)_L Weyl:   γ_1 ↔ γ_2,  γ_3, γ_4 fixed  (m_L → -m_L).
  * SU(2)_R Weyl:   γ_3 ↔ γ_4,  γ_1, γ_2 fixed  (m_R → -m_R).

Multiplication examples (verified):
  γ_1 · γ_2 = F_(2, 0, 0, 0)    [L-Weyl-invariant: m_L sum = 0]
  γ_3 · γ_4 = F_(-2, 2, 0, 0)   [R-Weyl-invariant: m_R sum = 0]
  γ_1 · γ_3 = q · F_(0, 1, 1, 1) [mixed (m_L=+1, m_R=+1)]
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from bps_kalgebra import BPSKAlgebra


SU2_NF2_PAIRING = [
    [0,  1, 0, 0],
    [-1, 0, 0, 0],
    [0,  0, 0, 0],
    [0,  0, 0, 0],
]

# Spin(4)-manifest node charges (mutated chart, two doublets).
SU2_NF2_NODE_CHARGES = [
    (1, 0,  1, 0),    # γ_1: SU(2)_L doublet, m_L = +1.
    (1, 0, -1, 0),    # γ_2: SU(2)_L doublet, m_L = -1.
    (-1, 1, 0,  1),   # γ_3: SU(2)_R doublet, m_R = +1.
    (-1, 1, 0, -1),   # γ_4: SU(2)_R doublet, m_R = -1.
]

# Spec is the 4 nodes in this order (two doublets back-to-back).
SU2_NF2_SPEC = list(SU2_NF2_NODE_CHARGES)


def build_bps_su2_nf2() -> BPSKAlgebra:
    """Construct the canonical Spin(4)-manifest `BPSKAlgebra` for
    `A_𝖖[SU(2) + N_f = 2]`.

    `coefficient_ring()` is `AbelianZPlusRing(rank=2)` — the Cartan
    torus of Spin(4) = SU(2)_L × SU(2)_R with generators (μ_L, μ_R)
    sitting on lattice coordinates 3 and 4 respectively.
    """
    return BPSKAlgebra(
        pairing=SU2_NF2_PAIRING,
        node_charges=SU2_NF2_NODE_CHARGES,
        spec=SU2_NF2_SPEC,
    )


def weyl_L(v: tuple) -> tuple:
    """SU(2)_L Weyl reflection: flip m_L (lattice coord index 2)."""
    return (v[0], v[1], -v[2], v[3])


def weyl_R(v: tuple) -> tuple:
    """SU(2)_R Weyl reflection: flip m_R (lattice coord index 3)."""
    return (v[0], v[1], v[2], -v[3])
