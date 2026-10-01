"""`cut_internal_edge` + the certified RG-flow catalog for the SU(2)+matter
bordered-skein charts.

Cutting an FST internal edge un-glues it (its two slots become boundary
marks) and DELETES that node from the mutable BPS quiver — the node-drop
RG IR theory (`BorderedSkeinKAlg`: "cutting an FST edge deletes that node
from the quiver, so the canonical algebra is the node-drop RG IR theory").

Two physically distinct flows out of SU(2) N_f=k (the 2-regular-puncture
disk; see `skein_su2_nf{2,3}_disk`):

  * **cut the inter-puncture `I1–I2` edge → SU(2) N_f=(k−1)** — integrate
    out the mutual doublet (N_f lowering);
  * **cut a radial `O–I` edge → [A1, D_{k+1}]** — flow to the maximal
    Argyres-Douglas point of SU(2) N_f=k.

Certified (flavour-neutral vacuum Schur index, K=4) in
the suite in the source repository:
    N_f=2 --I1I2-->  N_f=1 {1,1};      N_f=2 --radial-->  A1D3 {1,3,9}
    N_f=3 --I1I2-->  N_f=2 {1,6,17};   N_f=3 --radial-->  A1D4 {1,8,36}
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bordered_triangulation import BorderedTriangulation
from bordered_skein_kalg import BorderedSkeinKAlg

__all__ = ["cut_internal_edge", "cut_bordered_skein"]


def cut_internal_edge(bt: BorderedTriangulation, gluing_index: int
                      ) -> BorderedTriangulation:
    """Cut internal edge `gluing_index` (an index into `bt.gluings`):
    un-glue it so its two triangle slots become boundary marks.  Deletes
    that node from the mutable quiver (node-drop RG)."""
    gluings = list(bt.gluings)
    slot_a, slot_b = gluings.pop(gluing_index)
    boundary = tuple(bt.boundary_sides) + (slot_a, slot_b)
    return BorderedTriangulation(bt.triangle_verts, tuple(gluings), boundary)


def cut_bordered_skein(bt: BorderedTriangulation, gluing_index: int,
                       *, spec=None, verify: str = "off") -> BorderedSkeinKAlg:
    """The IR `BorderedSkeinKAlg` of cutting edge `gluing_index` of `bt`."""
    return BorderedSkeinKAlg(cut_internal_edge(bt, gluing_index),
                             spec=spec, verify=verify)


if __name__ == "__main__":
    import warnings
    from skein_su2_nf2_disk import disk_1b_2p_triangulation
    from skein_su2_nf3_disk import disk_2b_2p_triangulation
    from zplus_ring import augmentation_hom

    def nidx(S, K=4):
        Sa = S.base_change(augmentation_hom(S.coefficient_ring()))
        return {e: str(c) for e, c in sorted(Sa.trace(Sa.identity(), K).coeffs.items())}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, T in [("N_f=2", disk_1b_2p_triangulation()),
                        ("N_f=3", disk_2b_2p_triangulation())]:
            print(f"--- cuts of SU(2) {name} ---")
            for gi in range(len(T.gluings)):
                S = cut_bordered_skein(T, gi)
                print(f"  cut g{gi} {T.gluings[gi]}: "
                      f"{len(S.node_charges)}-node  index={nidx(S)}")
