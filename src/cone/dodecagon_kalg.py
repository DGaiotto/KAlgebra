"""
dodecagon_kalg.py
=================

`DodecagonKAlg` — ungauged dodecagon K-algebra (k=4), μ-flavoured wrapper
around the standalone `U1DodecagonKAlg` (u(1)-gauged [A_1, A_9]).  The k=4
sibling of `HexagonKAlg`.

The shared construction lives in `ungauged_polygon_kalg.UngaugedPolygonKAlg`
(centralizer of the gauge generator E=μ + measure-restored, **BPS-free**,
ungauged trace):

    Tr_ung(a)(z)  =  [ Σ_n z^n · Tr_gauged(a·μ^n) ] / (fq²;fq²)_∞²
                  =  ungauge_kalgebra.ungauge_u1polygon(4).trace .

Physical chord families (mag-zero / flavour-neutral): types 2 and 4
(12 + 12 chords); the type-1 / type-3 chords are magnetic and the type-5
diameter is magnetic too (k=4 is even, so the diameter is not physical).
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from ungauged_polygon_kalg import UngaugedPolygonKAlg
from u1_dodecagon_kalg import U1DodecagonKAlg
import u1a1aodd_k4_chord_charges as _charges


class DodecagonKAlg(UngaugedPolygonKAlg):
    """Ungauged dodecagon K-algebra (k=4), μ-flavoured."""

    k = 4
    _GAUGED_CLASS = U1DodecagonKAlg
    _CHARGES = _charges


if __name__ == "__main__":
    A = DodecagonKAlg()
    print(f"DodecagonKAlg (k={A.k}): coefficient_ring rank = {A.coefficient_ring().rank}")
    print(f"  identity            = {A.identity()}")
    print(f"  physical chord types= {A.physical_chord_types()}")
    print(f"  #mult-generators    = {len(A.mult_generators())}")
    print(f"  Tr_ung(1, K=6)      = {A.trace(((), 0), 6)}")
