"""`KAlgebraIso` between the **abelianized** pure-SU(2) realisation
(`AbelianizedSU2KAlg`, a light wrapper of `PureUNKAlgebra(2)`, labels
`L_{m,e}`) and the **BPS** pure-SU(2) realisation
(`pure_ade_kalgebra([("A", 1)])`, labels = lower tropical charges
`γ ∈ ℤ²`).

The non-trivial content is the label map — the 't Hooft–Wilson line
`L_{m,e}` does **not** sit at a linear function of `(m, e)`; the tropical
charge is the **Weyl-folded lowest weight**:

    γ(m, e) = (0, -e)        m = 0          (Wilson χ_e, lowest weight)
            = (m, e)         m ≥ 1, e ≤ 0
            = (m - e, e)     m ≥ 1, 0 ≤ e ≤ 2m
            = (-m, 4m - e)   m ≥ 1, e ≥ 2m

with inverse `me(γ)` returning the SU(2)-Weyl-chamber label.  Landmarks:
Wilson `χ_e = (0, -e)` (negative-electric lowest weight); bare monopole
`L_{m,0} = (m, 0)`; the BPS dyon node `γ₂ = (-1, 2)` is `L_{1,2}` (magnetic
**+1**).

Certified (see `tests/test_abelianized_su2_iso.py`) on the photon-independent
structure: `verify_unit`, `verify_round_trip`, `verify_multiplicative`
(the w₁-ladder + principal-cone Plückers), `verify_rho_equivariant`.  The
inherited `trace` differs by the decoupled U(1) photon factor (U(2) vs pure
SU(2)), so inner-product/trace equivariance is **not** claimed.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kalgebra import Element
from laurent_poly import LaurentPoly
from kalgebra_iso import KAlgebraIso
from pure_ade_lattice import pure_ade_kalgebra
from abelianized_su2_kalgebra import AbelianizedSU2KAlg, su2_fold


__all__ = ["gamma_of", "me_of", "abelianized_su2_bps_iso"]


def gamma_of(m: int, e: int) -> tuple[int, int]:
    """SU(2) line label `(m, e)` ↦ BPS lower tropical charge `γ`."""
    m, e = su2_fold(m, e)
    if m == 0:
        return (0, -e)
    if e <= 0:
        return (m, e)
    if e <= 2 * m:
        return (m - e, e)
    return (-m, 4 * m - e)


def me_of(x: int, y: int) -> tuple[int, int]:
    """BPS lower tropical charge `γ = (x, y)` ↦ SU(2) line label `(m, e)`.

    Inverse of `gamma_of`.  For `y > 0` the principal-cone wall is the sign
    of `2x + y` (not of `x`); for `y ≤ 0` the `x = 0` axis is the Wilson
    tower (folded to `e ≥ 0`)."""
    if y > 0:
        return (x + y, y) if 2 * x + y >= 0 else (-x, -4 * x - y)
    if x > 0:
        return (x, y)
    if x == 0:
        return (0, -y)
    return (-x, -4 * x - y)


def abelianized_su2_bps_iso(K: int = 12) -> KAlgebraIso:
    """The `KAlgebraIso  AbelianizedSU2KAlg  ≅  BPS pure SU(2)`."""
    A = AbelianizedSU2KAlg(K=K)
    B = pure_ade_kalgebra([("A", 1)])
    one = LaurentPoly.one()

    def forward(label):
        m, e = label
        return Element({gamma_of(m, e): one})

    def inverse(gamma):
        return Element({me_of(gamma[0], gamma[1]): one})

    return KAlgebraIso(
        source=A,
        target=B,
        forward_label_map=forward,
        inverse_label_map=inverse,
        name="AbelianizedSU2KAlg ≅ BPS pure SU(2) (A_1)",
    )
