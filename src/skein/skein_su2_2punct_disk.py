"""`bordered_su2_2punct` — SU(2) + 2 hyper doublets + A1Dn as the DISK with
2 regular interior punctures + n boundary marks: the 2-puncture
generalization of the A1Dn once-punctured n-gon.

Family map:
  * n-disk + 1 reg puncture  = A1Dn                       (`skein_a1dn_disk`)
  * (n,1) annulus            = SU(2)A1Dn                  (`skein_su2a1dn_annulus`)
  * **n-disk + 2 reg punctures = SU(2) + 2 doublets + A1Dn**  (this module)

Construction: the A1Dn fan (puncture P, n boundary marks) with a SECOND
regular puncture Q inserted into the triangle `(P,O1,O2)` (connected to
P, O1, O2 — splitting one triangle into three).  Result: **n+2 triangles,
an (n+3)-node quiver**, 2 interior punctures + n boundary marks.

Physics: the 2 regular punctures = 2 SU(2)-doublet hypers; the n boundary
marks build the A1Dn irregular matter.  For SMALL n the A1Dn matter is
free and this is ordinary SU(2)+N_f:
  * n=1 → A1D1 (trivial) → **SU(2) N_f=2**  (`skein_su2_nf2_disk`)
  * n=2 → A1D2 (free)    → **SU(2) N_f=3**  (`skein_su2_nf3_disk`; verified
    identical, flavour-neutral index `{0:1, 2:15, 4:100}`)
For n>=3 the A1Dn matter is interacting (Argyres-Douglas), so this is a
genuinely AD-matter-coupled SU(2) theory — NOT SU(2) N_f=(n+1) (e.g. n=3
has flavour rank 2, not N_f=4's rank 4).

Certified by the test suite.
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

__all__ = ["su2_2punct_triangulation", "bordered_su2_2punct"]

_A, _B, _C = _bt.ROLE_A, _bt.ROLE_B, _bt.ROLE_C


def su2_2punct_triangulation(n: int) -> BorderedTriangulation:
    """The 2-puncture n-disk (n>=2): punctures P=0, Q=n+1 + n boundary
    marks O1..On.  The A1Dn fan with Q inserted into triangle (P,O1,O2)."""
    if n < 2:
        raise ValueError(f"su2_2punct needs n >= 2 (got {n}); n=1 = SU(2) N_f=2")
    Q = n + 1
    # triangles: S0,S1,S2 (the split of the fan triangle (P,O1,O2)) then T1..T_{n-1}
    tv = [(0, 1, Q), (1, 2, Q), (2, 0, Q)] \
        + [(0, i + 1, (i + 1) % n + 1) for i in range(1, n)]
    # index map: S0=0, S1=1, S2=2 ; original fan T_i (i=1..n-1) -> 2+i
    gluings = [((0, _B), (1, _C)), ((1, _B), (2, _C)), ((0, _C), (2, _B))]  # Q arcs
    gluings += [((0, _A), (n + 1, _C))]        # S0.A = P-O1 ~ T_{n-1}.C
    gluings += [((3, _A), (2, _A))]            # T1.A ~ S2.A = O2-P
    gluings += [((2 + i, _A), (1 + i, _C)) for i in range(2, n)]   # fan i=2..n-1
    boundary = [(1, _A)] + [(2 + i, _B) for i in range(1, n)]      # O1-O2 + Ti bdys
    return BorderedTriangulation(tuple(tv), tuple(gluings), tuple(boundary))


def bordered_su2_2punct(n: int, *, spec=None, verify: str = "off") -> BorderedSkeinKAlg:
    """SU(2) + 2 doublets + A1Dn as the bordered 2-puncture n-disk skein.
    (n+3)-node quiver.  n=1,2 = SU(2) N_f=2,3; n>=3 AD-matter-coupled."""
    return BorderedSkeinKAlg(su2_2punct_triangulation(n), spec=spec, verify=verify)


if __name__ == "__main__":
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for n in (2, 3, 4):
            T = su2_2punct_triangulation(n)
            S = bordered_su2_2punct(n)
            print(f"2-puncture n={n}-disk (SU(2)+2 doublets+A1D{n}): "
                  f"{T.n_triangles} tri, {len(T.mutable_block())}-node quiver, "
                  f"flavour {S.coefficient_ring()}")
