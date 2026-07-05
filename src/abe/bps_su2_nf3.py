"""bps_su2_nf3 — canonical `BPSKAlgebra` for SU(2) + N_f = 3 with
**manifest SU(4) flavour symmetry on a rank-3 flavour lattice**.

User-specified SU(4)-manifest BPS quiver (5 nodes in 5-dim lattice,
2 gauge + 3 flavour = rank-3 SU(4) Cartan):

    γ_1 = (1, 0, 0, 0, 0)                  [SU(2) gauge node]
    γ_2 = (0, 1, w_1) where w_1 = (1, 0, 0)        ┐
    γ_3 = (0, 1, w_2) where w_2 = (-1, 1, 0)       ├── 4 fund-weight
    γ_4 = (0, 1, w_3) where w_3 = (0, -1, 1)       │   matter dyons,
    γ_5 = (0, 1, w_4) where w_4 = (0, 0, -1)       ┘   permuted by S_4

The 4 fund weights sum to zero (Σ w_k = 0) — true rank-3 SU(4) Cartan
in Dynkin / fundamental-weight basis.

Lattice pairing B_lat: only B[0][1] = -B[1][0] = 1 nonzero (3 flavour
directions in the kernel).

Node-basis pairings:
    ⟨γ_1, γ_i⟩ = +1  for i = 2, 3, 4, 5  (S_4-symmetric)
    ⟨γ_i, γ_j⟩ = 0   for i, j ∈ {2, 3, 4, 5}

`coefficient_ring()` = `AbelianZPlusRing(rank=3)` (rank-3 SU(4) Cartan
torus with generators μ_1, μ_2, μ_3 on flavour lattice slots 2, 3, 4).
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from bps_kalgebra import BPSKAlgebra


SU2_NF3_PAIRING = [[0]*5 for _ in range(5)]
SU2_NF3_PAIRING[0][1] = 1
SU2_NF3_PAIRING[1][0] = -1

# 4 SU(4) fund weights in rank-3 Dynkin basis (summing to zero).
SU4_FUND_WEIGHTS = [
    ( 1,  0,  0),    # w_1
    (-1,  1,  0),    # w_2
    ( 0, -1,  1),    # w_3
    ( 0,  0, -1),    # w_4
]

SU2_NF3_NODE_CHARGES = [
    (1, 0, 0, 0, 0),
] + [
    (0, 1, *w) for w in SU4_FUND_WEIGHTS
]

SU2_NF3_SPEC = list(SU2_NF3_NODE_CHARGES)


def build_bps_su2_nf3() -> BPSKAlgebra:
    """Canonical SU(4)-manifest `BPSKAlgebra` for `A_𝖖[SU(2) + N_f = 3]`
    on rank-3 SU(4) Cartan.

    `coefficient_ring()` = `AbelianZPlusRing(rank=3)` — Cartan torus
    of SU(4) flavour with 3 generators μ_1, μ_2, μ_3 on lattice slots
    2, 3, 4 respectively.  S_4 = Weyl(SU(4)) permutes the 4 fund-weight
    vectors w_1..w_4.
    """
    return BPSKAlgebra(
        pairing=SU2_NF3_PAIRING,
        node_charges=SU2_NF3_NODE_CHARGES,
        spec=SU2_NF3_SPEC,
    )


def s4_permute(v: tuple, perm: tuple) -> tuple:
    """Apply an S_4 permutation `perm` (of (0,1,2,3) indexing the 4 fund
    weights w_1..w_4) to a 5-tuple lattice vector.

    Acts as the linear transformation on the rank-3 flavour Cartan that
    permutes w_1..w_4 according to `perm`.
    """
    if len(perm) != 4 or sorted(perm) != [0, 1, 2, 3]:
        raise ValueError(f"perm must be a permutation of (0,1,2,3); got {perm}")
    # Reconstruct the flavour 3-vector as a linear combination of w_k's,
    # then permute the coefficients, then re-express in Dynkin basis.
    flav = v[2:5]
    # Decompose: flav = c_1·w_1 + c_2·w_2 + c_3·w_3 + c_4·w_4 with Σ c_k = 0
    # (since Σ w_k = 0).  Use c_k = "amount of weight w_k in flav" — this
    # has a 1-parameter family (Σ c_k free), so we fix by setting min c_k = 0.
    import numpy as _np
    W = _np.array(SU4_FUND_WEIGHTS).T               # 3x4 matrix (each col = w_k).
    # Solve W·c = flav with min-c-shift normalisation.
    # Just compute c via pinv (rank 3 matrix).
    c_solve, *_ = _np.linalg.lstsq(W, _np.array(flav, dtype=float), rcond=None)
    # Normalise: c_k → c_k - min(c_k).  But for permutation we just need
    # to permute the c's; the min-shift is invariant.
    permuted_c = [c_solve[perm[i]] for i in range(4)]
    new_flav = W @ _np.array(permuted_c)
    # Round to integers (should be exact).
    new_flav = tuple(int(round(x)) for x in new_flav)
    return (v[0], v[1], *new_flav)
