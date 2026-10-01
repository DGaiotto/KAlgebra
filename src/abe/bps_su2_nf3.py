"""bps_su2_nf3 — canonical `BPSKAlgebra` for SU(2) + N_f = 3 with
**manifest SU(4) flavour symmetry on a rank-3 flavour lattice**.

Author-specified SU(4)-manifest BPS quiver (5 nodes in 5-dim lattice,
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

    **Exact over `Z`** — no solver and no floating point.  Decompose
    `flav = Σ_k c_k w_k`.  The `w_k` span the rank-3 Cartan with the single
    relation `Σ_k w_k = 0`, so `c` is determined only up to a common shift
    `c_k → c_k + t`; that shift is invisible in the answer, because permuting
    `c` and re-contracting moves the result by `t·Σ_k w_k = 0`.  Fixing `c_4 = 0`
    leaves a `3×3` system that is **unit upper-triangular** in this basis
    (`w_1, w_2, w_3` as columns), so back-substitution is exact in integers:

        c_3 = f_3,   c_2 = f_2 + c_3,   c_1 = f_1 + c_2,   c_4 = 0.
    """
    if len(perm) != 4 or sorted(perm) != [0, 1, 2, 3]:
        raise ValueError(f"perm must be a permutation of (0,1,2,3); got {perm}")
    f1, f2, f3 = v[2:5]
    c = (f1 + f2 + f3, f2 + f3, f3, 0)
    new_flav = [0, 0, 0]
    for i, w in enumerate(SU4_FUND_WEIGHTS):
        ci = c[perm[i]]
        for j in range(3):
            new_flav[j] += ci * w[j]
    return (v[0], v[1], *new_flav)
