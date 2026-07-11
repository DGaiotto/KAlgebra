"""`bordered_a1dn` — [A1, D_n] as the DISK with n boundary marks + a single
regular interior puncture: the once-punctured n-gon.

Family map:
  * **n-disk + 1 regular puncture = A1Dn**            (this module)
  * (n,1) annulus                   = SU(2)A1Dn        (gauge the A1Dn SU(2))
  * n-disk + 2 regular punctures    = SU(2) + 2 doublets + A1Dn

Construction: fan the single interior puncture P=0 to the n boundary marks
O1..On by n radial arcs; triangle i = (P, O_{i+1}, O_{(i+1)%n+1}).  The
puncture's n incident arcs form the FST oriented n-cycle mutable quiver;
the n boundary edges (the n-gon) are frozen.  No self-gluing (each
triangle-pair shares one radial edge).

Flavour parity (a real D_n feature): odd n → rank-1 flavour (SU(2)
Cartan), even n → rank-2.

Certified: n=3 = A1D3 — the flavour-
neutral (μ→1) vacuum Schur index `{0:1, 2:3, 4:9, 6:19}` equals
`a1d3_bps`'s SU(2) index (`χ_k → k+1`), and the once-punctured triangle
is the certified A1D3 chart (χ₁ = the peripheral loop).  The
chart carries the abelian (Cartan) flavour in the minuscule normalization
(chart charge ±1 = the reference's χ₂ at ±2), consistent with the whole
bordered family.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import bordered_triangulation as _bt
from bordered_triangulation import BorderedTriangulation
from bordered_skein_kalg import BorderedSkeinKAlg

__all__ = ["a1dn_triangulation", "bordered_a1dn"]

_A, _B, _C = _bt.ROLE_A, _bt.ROLE_B, _bt.ROLE_C


def a1dn_triangulation(n: int) -> BorderedTriangulation:
    """The once-punctured n-gon (n>=2): 1 interior regular puncture P=0 +
    n boundary marks O1..On, fanned by n radial arcs."""
    if n < 2:
        raise ValueError(f"a1dn needs n >= 2 (got {n})")
    triangle_verts = tuple((0, i + 1, (i + 1) % n + 1) for i in range(n))
    gluings = tuple(((i, _A), ((i - 1) % n, _C)) for i in range(n))  # radial
    boundary = tuple((i, _B) for i in range(n))                      # n-gon
    return BorderedTriangulation(triangle_verts, gluings, boundary)


def bordered_a1dn(n: int, *, spec=None, verify: str = "off") -> BorderedSkeinKAlg:
    """[A1, D_n] as the bordered once-punctured n-gon skein.  n-node FST
    oriented-cycle quiver; flavour rank 1 (odd n) / 2 (even n)."""
    return BorderedSkeinKAlg(a1dn_triangulation(n), spec=spec, verify=verify)


if __name__ == "__main__":
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for n in (2, 3, 4, 5):
            T = a1dn_triangulation(n)
            S = bordered_a1dn(n)
            print(f"A1D{n}: {T.n_triangles} tri, {len(T.mutable_block())}-node "
                  f"quiver, flavour {S.coefficient_ring()}")
