"""`bordered_disk_su2_nf3` — SU(2)+N_f=3 as the DISK with 2 regular
interior punctures + a **2-marked boundary** (rank-1 irregular puncture)
(user frame, 2026-07-10): the next rung of the cut = RG family, one
boundary mark above the N_f=2 disk.

Surface: outer boundary circle with **2 marks** (O1,O2 → a bigon) + 2
regular interior punctures (I1,I2).  Compared to N_f=2
(`skein_su2_nf2_disk`, 1-mark boundary monogon) this adds one boundary
mark — the extra fundamental hyper — turning the outer monogon into a
bigon and the quiver from 4 to 5 nodes.

Triangulation (4 triangles, vertices O1=0,O2=1,I1=2,I2=3):
    T0=(O1,O2,I1)  T1=(O2,O1,I2)  T2=(O1,I1,I2)  T3=(O2,I2,I1)
    interior arcs  O1-I1, O2-I1, O1-I2, O2-I2, I1-I2  (5 nodes)
    boundary       the outer bigon O1-O2 (two edges)
Every triangle-pair shares a single edge; no self-gluing.

Result (certified in `tests/test_su2_nf3_disk_skein.py`): 5-node BPS
quiver, `AbelianZPlusRing(rank=3)` flavour (SO(6)=SU(4) Cartan), vacuum
Schur index = `build_bps_su2_nf3` (same q-backbone: 12 flavour states +3
at q², 54 +10 at q⁴; the flavour-neutral μ→1 index matches verbatim).
The chart uses the minuscule flavour basis; bps the SU(4) Dynkin/fund
basis.
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

__all__ = ["disk_2b_2p_triangulation", "bordered_disk_su2_nf3"]

_A, _B, _C = _bt.ROLE_A, _bt.ROLE_B, _bt.ROLE_C


def disk_2b_2p_triangulation() -> BorderedTriangulation:
    """The SU(2) N_f=3 disk: outer 2-mark boundary (O1,O2) + 2 regular
    interior punctures (I1,I2).  4 triangles, 5 interior arcs, 2 boundary
    edges."""
    triangle_verts = ((0, 1, 2), (1, 0, 3), (0, 2, 3), (1, 3, 2))
    gluings = (
        ((0, _C), (2, _A)),   # O1-I1
        ((0, _B), (3, _C)),   # O2-I1
        ((1, _B), (2, _C)),   # O1-I2
        ((1, _C), (3, _A)),   # O2-I2
        ((2, _B), (3, _B)),   # I1-I2
    )
    boundary = ((0, _A), (1, _A))   # the outer bigon (2 boundary marks)
    return BorderedTriangulation(triangle_verts, gluings, boundary)


def bordered_disk_su2_nf3(*, spec=None, verify: str = "off") -> BorderedSkeinKAlg:
    """SU(2)+N_f=3 as the bordered disk (2-mark boundary + 2 regular
    punctures), minuscule flavour normalization.  5-node mutable quiver;
    rank-3 flavour (SO(6)=SU(4) Cartan)."""
    return BorderedSkeinKAlg(disk_2b_2p_triangulation(), spec=spec,
                             verify=verify)


if __name__ == "__main__":
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        S = bordered_disk_su2_nf3()
        T = disk_2b_2p_triangulation()
        print("bordered_disk_su2_nf3 (SU(2) N_f=3, disk 2-mark bdry + 2 reg punctures)")
        print("  mutable quiver (5 nodes):")
        for r in T.mutable_block():
            print("    ", r)
        print("  coeff ring:", S.coefficient_ring())
        vac = S.trace(S.identity(), 2)
        print("  vacuum index (K=2):",
              {e: str(c) for e, c in sorted(vac.coeffs.items())})
