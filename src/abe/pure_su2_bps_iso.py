"""`pure_su2_bps_iso` — the **full-battery** `KAlgebraIso` between the modern
native `PureSU2KAlgebra` (the AbeKAlgebra realisation on `SU2RQTorus` —
honest SU(2) trace) and the **BPS** pure SU(2) (`pure_ade_kalgebra([("A",1)])`,
the Kronecker chart).

Upgrades `abelianized_su2_bps_iso` (same `gamma_of`/`me_of` chamber map — the
Weyl-folded lowest weight): that witness wrapped `PureUNKAlgebra(2)` whose
inherited trace carried the decoupled U(1) photon, so trace/inner-product
equivariance was **not claimed** there.  `PureSU2KAlgebra` carries the honest
rank-1 SU(2) Schur measure (validated == the BPS trace per label in its own
suite), so THIS witness claims the **whole** battery: unit, round-trip,
multiplicativity, ρ-equivariance, **and trace-equivariance** —
`tests/test_pure_su2_bps_iso.py`.
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

from kalgebra import Element
from kalgebra_iso import KAlgebraIso
from laurent_poly import LaurentPoly
from pure_ade_lattice import pure_ade_kalgebra
from abelianized_su2_bps_iso import gamma_of, me_of
from pure_su2_kalgebra import PureSU2KAlgebra


__all__ = ["pure_su2_bps_iso"]


def pure_su2_bps_iso(A: PureSU2KAlgebra | None = None) -> KAlgebraIso:
    """`PureSU2KAlgebra ≅ BPS pure SU(2)` — the modern abe↔BPS witness with
    the honest trace on both sides (full battery incl. trace)."""
    A = A if A is not None else PureSU2KAlgebra()
    B = pure_ade_kalgebra([("A", 1)])
    one = LaurentPoly.one()

    def forward(label):
        return Element({gamma_of(*label): one})

    def inverse(gamma):
        return Element({me_of(gamma[0], gamma[1]): one})

    return KAlgebraIso(
        source=A, target=B,
        forward_label_map=forward, inverse_label_map=inverse,
        name="PureSU2KAlgebra ≅ BPS pure SU(2) (A_1) [full battery]",
    )
