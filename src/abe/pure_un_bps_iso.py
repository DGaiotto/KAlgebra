"""`pure_un_bps_iso` — the N-general `KAlgebraIso` between the pure-U(N)
**AbeKAlgebra keystone** (`PureUNKAlgebra`) and the **BPS** chart
(`BPSKAlgebra` on `pure_ade.UN_Nf(N, 0)` — rank `2N` with the free photon),
on the anchored + rigidified `UNTropicalMap` (`un_bps_chamber`; user
direction 2026-07-02, validated at U(2) and — via the same architecture —
at SU(2)/SU(3)).

Certified at U(3) in `tests/test_pure_un_bps_iso_u3.py` (anchors, crossed
products, σ-transport, round-trips).  The BPS trace on the free-photon chart
stays impractical (the documented U(2) caveat), so trace-equivariance is not
claimed on this edge — the trace legs are certified through the SU(N)
projections instead."""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (_HERE, os.path.dirname(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from kalgebra import Element
from kalgebra_iso import KAlgebraIso
from laurent_poly import LaurentPoly
from bps_kalgebra import BPSKAlgebra
import pure_ade as pa
from pure_un_kalgebra import PureUNKAlgebra, default_rays
from un_bps_chamber import UNTropicalMap


__all__ = ["pure_un_bps_chart", "pure_un_bps_iso"]

_ONE = LaurentPoly.one()


def pure_un_bps_chart(N: int) -> BPSKAlgebra:
    """The pure-U(N) BPS chart (`UN_Nf(N, 0)`, rank 2N, free photon)."""
    t = pa.UN_Nf(N, 0)
    return BPSKAlgebra(pairing=[list(r) for r in t.B],
                       node_charges=[tuple(g) for g in t.nodes],
                       spec=[tuple(g) for g in t.spec],
                       verify="off").shorten_spec()


def pure_un_bps_iso(N: int, abe: PureUNKAlgebra | None = None,
                    max_len: int = 2, K: int = 8) -> KAlgebraIso:
    A = abe if abe is not None else PureUNKAlgebra(
        N, default_rays(N), max_len=max_len, K=K)
    B = pure_un_bps_chart(N)
    T = UNTropicalMap(A, B, N=N)

    def forward(label):
        return Element({T.gamma(label): _ONE})

    def inverse(g):
        return Element({T.label(g): _ONE})

    iso = KAlgebraIso(A, B, forward, inverse,
                      name=f"PureUNKAlgebra({N}) ≅ BPS pure U({N}) [anchored]")
    iso.tropical_map = T
    return iso
