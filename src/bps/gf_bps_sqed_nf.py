"""gf_bps_sqed_nf — SQED_{N_f} as a faithful SU(N_f)-flavoured BPS realisation
via the canonical `GfBPSKAlgebra`, over the **general** flavour ring
`SUNZPlusRing(N_f)` (all N_f).

The BPS quiver: **just the N_f hyper nodes in
`Γ = Z²(gauge m, e) × Λ_weight(SU(N_f))`** — node `j` at `(0, 1, w_j)`, `w_j`
the j-th fundamental weight of SU(N_f) in the rank-(N_f-1) torus, the standard
projection `proj : Z^{N_f} → Z^{N_f-1}, w ↦ (w_i - w_{N_f})` of `e_j`
(`proj(e_j)=δ_j` for j<N_f, `proj(e_{N_f})=(-1,…,-1)`).  The N_f weights are
permuted by the flavour Weyl group `S_{N_f}` (the simple reflections below).
The pairing is the gauge symplectic `[[0,1],[-1,0]]` on `(m, e)` —
`⟨(1,0,0),(0,1,0)⟩ = +1`, the draft's convention (§"The central quotient of
`U_𝖖(sl_2)` and SQED_2": `u₊ = D_{1,0}` the positive monopole, `v = D_{0,1}` the
Wilson line, the two hypers at `(0, 1, ±1)`) — with the weight directions in
`ker B` (rank `N_f-1` = the SU(N_f) torus rank; the diagonal U(1) is
gauge-centre — D5).  The magnetic direction carries no node: it is the
draft's dashed, frozen node.  `GfBPSKAlgebra` then presents the Weyl-invariant
subalgebra over `R(SU(N_f))`.

**Coordinate order `(m, e, w)` — magnetic first (since 2026-09-21).**  Until then this
module built the lattice as `(e, m, w)` with `⟨e, m⟩ = +1`, which is the
OPPOSITE orientation to the draft's on `(m, e)`; labels read `(e, m, w)`.
The conversion makes the SQED_2 instance literally the draft's quiver, aligns
the labels with `SQEDNfSampleKAlgebra`'s `(m, n, w)` (the positive monopole
`(1, 0, …)` is `u₊` on both), and is what lets
`src/bps/uq_su2_bps_iso.py` read the draft's own generator dictionary
`E = u₊ = D_{1,0}`, `F = u₋ = D_{−1,−1}`, `K = v = D_{0,1}` off this chart.
The identity traces against the standalone sample and the axiom battery
(the suite in the source repository) are unchanged by the conversion.

Because `GfBPSKAlgebra` now drives its weight machinery through
`SUNZPlusRing.to_abelian` / `from_abelian` (the general-N un-brancher), this is
**uncapped in N_f** and uses the *same* `SUNZPlusRing(N_f)` coefficient ring as
the native / flow-enhancement / standalone presentations — so it joins them in
one `KAlgebraObject`.  Verified to reproduce
`SQEDNfSampleKAlgebra(N_f)` exactly (trace + axioms) for N_f=2..5.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from gf_bps_kalgebra import GfBPSKAlgebra
from zplus_ring import SUNZPlusRing


def sqed_nf_quiver(Nf: int):
    """`(pairing, node_charges, reflections, mu_basis)` for the SQED_{N_f} BPS
    quiver `Γ = Z²(m, e) × Λ_weight(SU(N_f))` (rank N_f+1), coordinates
    `(m, e, w)`: magnetic charge first, `⟨(1,0,…),(0,1,…)⟩ = +1`."""
    if Nf < 2:
        raise ValueError("sqed_nf_quiver: N_f >= 2 (SU(1) flavour is trivial)")
    n = Nf + 1
    d = Nf - 1                                   # SU(N_f) torus rank
    B = [[0] * n for _ in range(n)]
    B[0][1], B[1][0] = 1, -1                     # ⟨m, e⟩ = +1 on (m, e)

    def w(j):                                    # projected j-th fundamental weight
        if j < Nf - 1:
            v = [0] * d
            v[j] = 1
            return tuple(v)
        return tuple([-1] * d)                   # proj(e_{N_f})

    nodes = [tuple([0, 1] + list(w(j))) for j in range(Nf)]   # hypers at (m, e) = (0, 1)
    mu_basis = [tuple([0, 0] + [1 if i == k else 0 for i in range(d)])
                for k in range(d)]

    def _bidx(i):
        return 2 + i                             # weight block lives at coords 2..n-1

    reflections = []
    for k in range(1, Nf):                        # simple reflections s_1 … s_{N_f-1}
        M = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
        if k < Nf - 1:                            # swap weight coords (k-1, k)
            a, b = _bidx(k - 1), _bidx(k)
            M[a][a], M[a][b] = 0, 1
            M[b][b], M[b][a] = 0, 1
        else:                                     # the special reflection s_{N_f-1}
            for i in range(d):
                for j in range(d):
                    M[_bidx(i)][_bidx(j)] = 0
            for i in range(d - 1):
                M[_bidx(i)][_bidx(i)] = 1
                M[_bidx(i)][_bidx(d - 1)] = -1
            M[_bidx(d - 1)][_bidx(d - 1)] = -1
        reflections.append(M)
    return B, nodes, reflections, mu_basis


def gf_bps_sqed_nf(Nf: int) -> GfBPSKAlgebra:
    """SQED_{N_f} as a faithful BPS realisation (`GfBPSKAlgebra` over
    `SUNZPlusRing(N_f)`), for any N_f >= 2."""
    B, nodes, reflections, mu_basis = sqed_nf_quiver(Nf)
    return GfBPSKAlgebra(B, nodes, reflections, SUNZPlusRing(Nf),
                         mu_basis=mu_basis)
