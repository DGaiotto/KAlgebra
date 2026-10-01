"""
a1dn_a1dodd_iso.py
==================

`KAlgebraIso`: `A1DnKAlg(n)` (the curves `(x, ℓ)` of the n-gon with one
interior puncture) ≅ `A1DoddConeKAlg((n − 3)/2)` (the generators `(a, p, i)`
of `a1dodd_cone_data`), n odd ≥ 3 — two presentations of `[A_1, D_n]` over
`R(SU(2))`.

The label maps are the curve dictionary of `a1dn_kalg`, applied letter by
letter with the powers and the SU(2) weight κ unchanged:

    (a, p, i)  ↦  (x, ℓ):   ℓ = 2a + 1 (p = 0),  ℓ = 2(k + 2 − a) (p = 1),
                            x = (_ap_base(a, p, k) + i) mod n.

It is a bijection of canonical labels (a cone monomial of one side is a cone
monomial of the other, since 𝖖-commuting ⟺ not crossing), so each map sends a
canonical-basis element to a single canonical-basis element with coefficient 1.
Certified by `KAlgebraIso.verify_all` (unit, round trip,
multiplicativity, ρ-equivariance and trace-equivariance, both directions) at
n = 3, 5, 7 — `tests/test_cones.py` runs it on a sample — with the mirror
dictionary `x ↦ −x − ℓ` as the negative control in the source repository
(it keeps every crossing and fails multiplicativity).

Since `A1DnKAlg` computes through `A1DoddConeKAlg`, the checks are of the
dictionary and of the class's translation code, not an independent
construction of either algebra.  The independent oracles for the algebra
(in the source repository's tests) are, for products, the BPS chart at n = 3
and the zoo tables at n = 5, 7; for traces, the BPS chart at n = 3 (the single
curves through `𝖖¹²`), Pan–Yang's closed form for `Tr(1)` and orthonormality.
A single-curve seed check is not one of them: the delegate's trace of one
generator is `seed_trace_ap` itself.
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kalgebra import Element
from kalgebra_iso import KAlgebraIso
from laurent_poly import LaurentPoly

_ONE = LaurentPoly.one()


def a1dn_a1dodd_iso(n: int) -> KAlgebraIso:
    """`A1DnKAlg(n) ≅ A1DoddConeKAlg((n − 3)/2)` as a `KAlgebraIso`, n odd ≥ 3."""
    from a1dn_kalg import A1DnKAlg, _ap_to_arc, _arc_to_ap
    from a1dodd_kalg import A1DoddConeKAlg

    source = A1DnKAlg(n)                 # refuses even n / n < 3
    k = source.k
    target = A1DoddConeKAlg(k)

    def forward(label):
        curves, kappa = source.canonicalise(label)
        word = tuple(sorted((_arc_to_ap(c, k), m) for c, m in curves))
        return Element({target.canonicalise((word, kappa)): _ONE})

    def inverse(label):
        word, kappa = target.canonicalise(label)
        curves = tuple(sorted((_ap_to_arc(g, k), m) for g, m in word))
        return Element({source.canonicalise((curves, kappa)): _ONE})

    return KAlgebraIso(
        source, target, forward, inverse,
        name=f"A1DnKAlg({n}) ≅ A1DoddConeKAlg({k})  (curves ↔ (a,p,i))",
    )
