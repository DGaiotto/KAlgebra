"""
qtorus_1d.py
============

`QuantumTorus1DKAlg`: 1-dimensional commutative quantum torus
Z[v, v⁻¹] as a `KAlgebra` over `TrivialZPlusRing`.

Labels are integers (powers of v); multiply is addition; bar is
v ↔ v⁻¹.  Coefficient ring is explicitly `TrivialZPlusRing` (NOT
`AbelianZPlusRing` as the generic `QuantumTorusKAlg` would assign
to a 1-dim torus with zero pairing).

Used as a factor of `TensorKAlgebra` when the K-algebra needs an
additional Laurent direction kept as part of the canonical basis
rather than absorbed into a flavour ring.  In particular, an IR
realization such as `TensorKAlgebra(PentagonKAlg(),
QuantumTorus1DKAlg())` keeps the dropped-flavour-node degree of
freedom as a Z-labelled Laurent factor of the IR canonical
basis.
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kalgebra import Element, KAlgebra
from laurent_poly import LaurentPoly
from zplus_ring import TrivialZPlusRing, ZPlusRing, RPowerSeries


class QuantumTorus1DKAlg(KAlgebra):
    """1D commutative quantum torus Z[v, v⁻¹] over Z[q±]."""

    def __init__(self) -> None:
        self._R = TrivialZPlusRing()

    def coefficient_ring(self) -> ZPlusRing:
        return self._R

    def identity(self):
        return 0

    def multiply(self, a, b) -> Element:
        return Element({a + b: LaurentPoly.one()})

    def rho(self, a):
        """ρ(v) = v⁻¹: the natural Z₂ on the Laurent variable."""
        return -a

    def rho_inverse(self, a):
        return -a

    def rho_squared_is_identity(self) -> bool:
        return True

    def trace(self, a, K: int = 20) -> RPowerSeries:
        """Trace functional: Tr(v^n) = δ_{n, 0} on the gauge cell."""
        if a == 0:
            return RPowerSeries(self._R, {0: 1}, K)
        return RPowerSeries(self._R, {}, K)

    # Unflavoured (Trivial coefficient ring): the flavour-lift coordinate and
    # _label_section_decompose (trivial: each label is its own section rep) are
    # inherited from the KAlgebra universal trivial-flavour treatment.

    def __repr__(self) -> str:
        return "QuantumTorus1DKAlg()"


if __name__ == "__main__":
    Q = QuantumTorus1DKAlg()
    print(Q)
    print(f"  coefficient_ring = {Q.coefficient_ring()}")
    print(f"  identity = {Q.identity()}")
    print(f"  v^2 * v^3 = {dict(Q.multiply(2, 3).terms)}")
    print(f"  rho(v^5) = {Q.rho(5)}")
    print(f"  Tr(v^0) = {dict(Q.trace(0, K=2).coeffs)}")
    print(f"  Tr(v^3) = {dict(Q.trace(3, K=2).coeffs)}")
