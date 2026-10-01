"""
dodecagon_kalg.py
=================

`DodecagonKAlg` — the ungauged μ-flavoured `[A_1, A_9]` on the dodecagon (12-gon) (k = 4), the
k = 4 member of `ungauged_polygon_kalg.UngaugedPolygonKAlg`: the centralizer
of the gauge generator `E` in the closed-form `u1a1aodd_kalg.U1A1AoddKAlg(4)`,
with `E` the flavour fugacity μ.  No frozen data, no BPS and no bootstrap:
products are the gauged closed-form peel, traces the measure-restored sum of
the gauged closed-form traces (see `ungauged_polygon_kalg`).

Labels: `U1A1AoddKAlg(4)`'s balanced multisets of non-crossing diagonals of
the dodecagon (12-gon) with the μ-charge; physical single
chords (magnetic charge 0) are the even types {2, 4} (12 + 12 chords; the type-5 diameter is magnetic at even k).  They replaced, on
2026-09-23, the letters of the retired stand-alone `U1DodecagonKAlg`;
the dictionary is in `ungauged_polygon_kalg`'s docstring.
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.append(_HERE)

from ungauged_polygon_kalg import UngaugedPolygonKAlg


class DodecagonKAlg(UngaugedPolygonKAlg):
    """Ungauged dodecagon K-algebra (k=4), μ-flavoured: `UngaugedPolygonKAlg(4)`."""

    k = 4

    def __init__(self) -> None:
        super().__init__(4)


if __name__ == "__main__":
    A = DodecagonKAlg()
    print(f"DodecagonKAlg (k={A.k}): coefficient_ring rank = {A.coefficient_ring().rank}")
    print(f"  identity            = {A.identity()}")
    print(f"  physical chord types= {A.physical_chord_types()}")
    print(f"  #mult-generators    = {len(A.mult_generators())}")
    print(f"  Tr_ung(1, K=6)      = {A.trace(((), 0), 6)}")
    print(f"  Tr_ung(L_long(0), K=6) = {A.trace(A.L_long(0), 6)}")
