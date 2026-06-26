"""
octagon_kalg.py
===============

`OctagonKAlg` — ungauged octagon K-algebra (k=2), μ-flavoured wrapper around
the standalone `U1OctagonKAlg` (u(1)-gauged [A_1, A_5]).  The k=2 sibling of
`HexagonKAlg`.

The shared construction lives in `ungauged_polygon_kalg.UngaugedPolygonKAlg`
(centralizer of the gauge generator E=μ + measure-restored, **BPS-free**,
ungauged trace):

    Tr_ung(a)(z)  =  [ Σ_n z^n · Tr_gauged(a·μ^n) ] / (fq²;fq²)_∞²
                  =  ungauge_kalgebra.ungauge_u1polygon(2).trace .

Physical chord families (mag-zero / flavour-neutral): type 2 only
(8 long chords); the type-1 short chords are magnetic and the type-3
diameter is composite.
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from ungauged_polygon_kalg import UngaugedPolygonKAlg
from u1_octagon_kalg import U1OctagonKAlg
import u1a1aodd_k2_chord_charges as _charges


class OctagonKAlg(UngaugedPolygonKAlg):
    """Ungauged octagon K-algebra (k=2), μ-flavoured."""

    k = 2
    _GAUGED_CLASS = U1OctagonKAlg
    _CHARGES = _charges


if __name__ == "__main__":
    A = OctagonKAlg()
    print(f"OctagonKAlg (k={A.k}): coefficient_ring rank = {A.coefficient_ring().rank}")
    print(f"  identity            = {A.identity()}")
    print(f"  physical chord types= {A.physical_chord_types()}")
    print(f"  #mult-generators    = {len(A.mult_generators())}")
    print(f"  Tr_ung(1, K=6)      = {A.trace(((), 0), 6)}")
    print(f"  Tr_ung(L_long(0), K=6) = {A.trace(A.L_long(0), 6)}")
