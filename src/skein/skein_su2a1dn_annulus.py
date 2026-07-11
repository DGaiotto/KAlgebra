"""`bordered_su2a1dn` — SU(2)A1Dn as the **(n,1) annulus**: the
SU(2)-GAUGING of A1Dn, gluing the A1Dn regular puncture
to a 1-mark boundary.

Family map:
  * n-disk + 1 reg puncture  = A1Dn            (`skein_a1dn_disk`)
  * **(n,1) annulus          = SU(2)A1Dn**     (this module)
  * n-disk + 2 reg punctures = SU(2) + 2 doublets + A1Dn

Construction: take the A1Dn once-punctured n-gon fan and **open the
puncture P into a 1-mark boundary O** (the SU(2)-gauging).  P's radial
`P–I1` arc is doubled and a folded-monogon triangle `(O,I1,O)` carrying
the outer boundary is inserted; the other n−1 radials and the inner n-gon
are unchanged.  Result: **n+1 triangles, an (n+1)-node quiver**, the inner
n marks + the outer 1 mark.

Flavour: gauging the A1Dn SU(2) removes one rank, so
  SU(2)A1Dn flavour rank = (A1Dn rank) − 1
    = 0 for odd n (A1Dn had SU(2)=rank1, fully gauged → TrivialZPlusRing)
    = 1 for even n (A1Dn had rank2 → rank1 left).

Low cases: **(1,1) = pure SU(2)** (`SkeinAnnulusKAlg`); **(2,1) =
SU(2)A1D2 = the SU(2) N_f=1 annulus** (`skein_su2_nf1_annulus`) — verified
identical.
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

__all__ = ["su2a1dn_triangulation", "bordered_su2a1dn"]

_A, _B, _C = _bt.ROLE_A, _bt.ROLE_B, _bt.ROLE_C


def su2a1dn_triangulation(n: int) -> BorderedTriangulation:
    """The (n,1) annulus (n>=2): outer 1-mark boundary O=0 (folded monogon,
    triangle n) + inner n-mark boundary I1..In, the A1Dn fan with the
    puncture opened."""
    if n < 2:
        raise ValueError(f"su2a1dn needs n >= 2 (got {n}); (1,1)=pure SU(2)")
    tv = [(0, i + 1, (i + 1) % n + 1) for i in range(n)] + [(0, 1, 0)]
    gluings = [((i, _A), ((i - 1) % n, _C)) for i in range(1, n)]  # radials but P-I1
    gluings += [((0, _A), (n, _B)), ((n, _A), (n - 1, _C))]        # insert monogon
    boundary = [(i, _B) for i in range(n)] + [(n, _C)]             # n inner + outer
    return BorderedTriangulation(tuple(tv), tuple(gluings), tuple(boundary))


def bordered_su2a1dn(n: int, *, spec=None, verify: str = "off") -> BorderedSkeinKAlg:
    """SU(2)A1Dn as the bordered (n,1) annulus skein.  (n+1)-node quiver;
    flavour rank 0 (odd n) / 1 (even n) — the A1Dn SU(2) gauged away."""
    return BorderedSkeinKAlg(su2a1dn_triangulation(n), spec=spec, verify=verify)


if __name__ == "__main__":
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for n in (2, 3, 4):
            T = su2a1dn_triangulation(n)
            S = bordered_su2a1dn(n)
            print(f"SU(2)A1D{n} = ({n},1) annulus: {T.n_triangles} tri, "
                  f"{len(T.mutable_block())}-node quiver, flavour {S.coefficient_ring()}")
