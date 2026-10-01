"""`bordered_disk_su2_nf2` — SU(2)+N_f=2 as the DISK with a 1-marked
boundary + 2 regular interior punctures, and its
**cut = RG flow** to the SU(2)+N_f=1 annulus.

The author's construction: SU(2) N_f=1 is the IR of the RG flow from SU(2)
N_f=2 by CUTTING the edge between the two regular punctures.  Concretely,
the N_f=2 disk chart is the fan (1,2)-annulus of `skein_su2_nf1_annulus`
with its inner `I1–I2` edge (`e4~e5`) **glued back** (un-cut): the two
regular punctures `I1,I2` are interior; the outer 1-mark boundary stays.
Cutting that glued edge un-glues `e4,e5` into the inner 2-boundary — the
N_f=1 annulus verbatim.

Result (certified in the suite in the source repository):
* the disk chart builds directly as a `BorderedSkeinKAlg` — **4-node**
  BPS quiver `[[0,-1,-1,1],[1,0,-1,0],[1,1,0,-1],[-1,0,1,0]]`,
  `AbelianZPlusRing(rank=2)` flavour (the 2 regular punctures);
* its vacuum Schur index = `build_bps_su2_nf2`'s with the identical q
  backbone, in the **minuscule** flavour basis `(±1,±1),(0,±1)` vs bps's
  orthogonal Cartan `(±2,0),(0,±2)` (the finer / U(1)-gaugeable lattice,
  same as N_f=1's ±1);
* CUT = RG: `cut_to_nf1()` un-glues the `I1–I2` edge → the exact
  `skein_su2_nf1_annulus` triangulation.
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
from skein_su2_nf1_annulus import annulus_1_2_triangulation

__all__ = ["disk_1b_2p_triangulation", "bordered_disk_su2_nf2", "cut_to_nf1"]

_A, _B, _C = _bt.ROLE_A, _bt.ROLE_B, _bt.ROLE_C


def disk_1b_2p_triangulation() -> BorderedTriangulation:
    """The SU(2) N_f=2 disk: outer 1-mark boundary (vertex O) + 2 REGULAR
    interior punctures (I1,I2).  = the (1,2)-annulus with the inner `I1–I2`
    edge glued back (un-cut `e4~e5`)."""
    triangle_verts = ((0, 1, 2), (0, 2, 1), (0, 1, 0))
    gluings = (
        ((0, _A), (2, _B)),   # e0  O–I1
        ((0, _C), (1, _A)),   # e1  O–I2
        ((1, _C), (2, _A)),   # e2  O–I1
        ((0, _B), (1, _B)),   # e4~e5 GLUED: the I1–I2 edge (UN-CUT)
    )
    boundary = ((2, _C),)     # e3 outer monogon only
    return BorderedTriangulation(triangle_verts, gluings, boundary)


def bordered_disk_su2_nf2(*, spec=None, verify: str = "off") -> BorderedSkeinKAlg:
    """SU(2)+N_f=2 as the bordered disk (1-boundary + 2 regular punctures),
    minuscule flavour normalization.  4-node mutable quiver; rank-2 flavour
    (the 2 regular punctures)."""
    return BorderedSkeinKAlg(disk_1b_2p_triangulation(), spec=spec,
                             verify=verify)


def cut_to_nf1() -> BorderedTriangulation:
    """The RG flow SU(2) N_f=2 → N_f=1: CUT the `I1–I2` edge (un-glue
    `e4~e5`) → the SU(2) N_f=1 annulus triangulation verbatim."""
    return annulus_1_2_triangulation()


if __name__ == "__main__":
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        S = bordered_disk_su2_nf2()
        T = disk_1b_2p_triangulation()
        print("bordered_disk_su2_nf2 (SU(2) N_f=2, disk 1-bdry + 2 reg punctures)")
        print("  mutable quiver (4 nodes):", T.mutable_block())
        print("  coeff ring:", S.coefficient_ring())
        vac = S.trace(S.identity(), 4)
        print("  vacuum index:", {e: str(c) for e, c in sorted(vac.coeffs.items())})
        print("  CUT the I1-I2 edge -> N_f=1 annulus:",
              cut_to_nf1().mutable_block())
