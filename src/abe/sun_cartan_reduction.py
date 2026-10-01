"""`sun_cartan_reduction` — the flavour reduction `R(SU(N_f)) → R(Cartan)`.

`sun_to_cartan_hom(Nf)`: the `RingHom` `SUNZPlusRing(N_f) →
AbelianZPlusRing(N_f−1)` sending an irrep to the sum of its weights in
Dynkin (consecutive-difference) Cartan coordinates — `χ_b ↦
Σ_w mult(w)·μ^{(w_1−w_2, …, w_{N_f−1}−w_{N_f})}` off the ring's own Kostka
`character`.  Composing a U(N)+N_f algebra over `R(SU(N_f))` with it — the general
tier's preset, e.g. `roster("u2-nf2").base_change(un_to_sun_hom(2))` (the retired
`UNNfKAlgebra(N, N_f)` until 2026-09-19) —
yields the **Cartan-reduced** abe presentation whose coefficients speak the
abelian μ-language of the native flavoured BPS chart — the presentation the
N_f ≥ 2 anchored iso is built on (the design record notes, pickup 1).

`cartan_lower(algebra)`: the **label-splitting** counterpart —
`lower_flavour` along the same hom, so the canonical basis re-indexes as
`(section, Cartan weight)` pairs (an SU multiplet `χ_w·L_s` splits into its
weights).  This is the presentation the iso witness actually runs on: its
labels biject with the BPS chart's `(section, collapsed weight)` labels.

`central_collapse_hom(Nf)`: the matching BPS-side reduction
`R(U(1)^{N_f}) → R(U(1)^{N_f−1})`, `μ^w ↦ μ^{(w_1−w_2, …)}` — kill the
central flavour U(1) (it sits inside the gauge centre on the abe
side, so the native flavoured BPS chart carries one extra ker-B direction
that the abe presentation stores in the gauge label; collapsing it aligns
the two coefficient rings in the same consecutive-difference coordinates)."""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (_HERE, os.path.dirname(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from zplus_ring import (SUNZPlusRing, AbelianZPlusRing, RingHom, RElement,
                        restriction_hom)


__all__ = ["sun_to_cartan_hom", "cartan_reduce", "cartan_lower",
           "central_collapse_hom"]


def sun_to_cartan_hom(Nf: int) -> RingHom:
    src = SUNZPlusRing(Nf)
    tgt = AbelianZPlusRing(rank=Nf - 1)

    def on_basis(b):
        out: dict = {}
        for w, mult in src.character(tuple(b)).items():
            key = tuple(w[i] - w[i + 1] for i in range(Nf - 1))
            out[key] = out.get(key, 0) + mult
        return RElement(tgt, out)

    return RingHom(src, tgt, on_basis)


def cartan_reduce(algebra):
    """`base_change` an SU(N_f)-flavoured `KAlgebra` down to its Cartan."""
    R = algebra.coefficient_ring()
    if not isinstance(R, SUNZPlusRing):
        raise TypeError(f"cartan_reduce: coefficient ring {R} is not SU(N_f)")
    return algebra.base_change(sun_to_cartan_hom(R.N))


def cartan_lower(algebra):
    """`lower_flavour` an SU(N_f)-flavoured `KAlgebra` down to its Cartan:
    coefficient-ring reduction *plus* the label split onto
    `(section, Cartan weight)` pairs (`χ_j·M_s ↦ Σ_k μ^k·M_s`)."""
    R = algebra.coefficient_ring()
    if not isinstance(R, SUNZPlusRing):
        raise TypeError(f"cartan_lower: coefficient ring {R} is not SU(N_f)")
    return algebra.lower_flavour(sun_to_cartan_hom(R.N))


def central_collapse_hom(Nf: int) -> RingHom:
    """`R(U(1)^{N_f}) → R(U(1)^{N_f−1})` in consecutive-difference
    coordinates, `μ^w ↦ μ^{(w_1−w_2, …, w_{N_f−1}−w_{N_f})}` — the
    central-flavour collapse matching `sun_to_cartan_hom`'s coordinates."""
    M = [[1 if j == i else (-1 if j == i + 1 else 0) for j in range(Nf)]
         for i in range(Nf - 1)]
    return restriction_hom(AbelianZPlusRing(rank=Nf),
                           AbelianZPlusRing(rank=Nf - 1), M)
