"""`bordered_annulus_su2_nf1` — the fan (1,2)-annulus as a
`BorderedSkeinKAlg`, the SU(2)+N_f=1 skein chart (minuscule / U(1)-gaugeable
normalization).  The base chart for the U(1)-gauged theory: pinning the
inner 2-marked boundary is the U(1)-gauging.

Geometry (fan (1,2)-annulus; two boundary circles):
    outer boundary = 1 marked point  (vertex O = 0; folded monogon)
    inner boundary = 2 marked points (vertices I1 = 1, I2 = 2)
    interior arcs  = e0,e2 (O–I1), e1 (O–I2); boundary = e3 (outer),
                     e4,e5 (inner).
The three triangles pairwise share a SINGLE edge (no self-GLUING; the outer
monogon merely has two identified vertices), so `BorderedTriangulation`
takes it directly — build it with the constructor, NOT `from_closed`
(whose Triangle-based path can't host the self-loop chart).

Result (certified): the mutable
BPS quiver is the oriented 3-cycle `[[0,-1,-1],[1,0,-1],[1,1,0]]`, and the
vacuum Schur index equals `build_bps_su2_nf1`'s **up to the flavour
normalization** — the annulus fugacity is charged ±1 (minuscule
endpoints), `build_bps_su2_nf1`'s ±2 (the non-minuscule SU(2)
normalization).  Same q-tower, finer (U(1)-gaugeable) flavour lattice.

The U(1)-GAUGING (→ `u1gauged_su2_nf1`, the `SU2Nf1PureSU2RGKAlgebra`
anchor) is the PINNING of the inner 2-boundary; `μ` = the pin / formal
monodromy.  That step is the follow-on.
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


__all__ = ["annulus_1_2_triangulation", "bordered_annulus_su2_nf1"]

_A, _B, _C = _bt.ROLE_A, _bt.ROLE_B, _bt.ROLE_C


def annulus_1_2_triangulation() -> BorderedTriangulation:
    """The fan (1,2)-annulus bordered triangulation (outer 1-mark monogon
    + inner 2-mark bigon; each triangle-pair shares a single edge)."""
    triangle_verts = ((0, 1, 2), (0, 2, 1), (0, 1, 0))
    gluings = (
        ((0, _A), (2, _B)),   # e0: tri0.a ~ tri2.b  (O–I1)
        ((0, _C), (1, _A)),   # e1: tri0.c ~ tri1.a  (O–I2)
        ((1, _C), (2, _A)),   # e2: tri1.c ~ tri2.a  (O–I1)
    )
    boundary = ((0, _B), (1, _B), (2, _C))   # e4, e5 (inner) ; e3 (outer)
    return BorderedTriangulation(triangle_verts, gluings, boundary)


def bordered_annulus_su2_nf1(*, spec=None, verify: str = "off") -> BorderedSkeinKAlg:
    """SU(2)+N_f=1 as the bordered (1,2)-annulus skein (minuscule /
    U(1)-gaugeable normalization).  Full `BPSKAlgebra` contract on the
    3-node mutable quiver; the inner 2-boundary carries `μ` (pin →
    U(1)-gauging)."""
    return BorderedSkeinKAlg(annulus_1_2_triangulation(), spec=spec,
                             verify=verify)


if __name__ == "__main__":
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        S = bordered_annulus_su2_nf1()
        print("bordered_annulus_su2_nf1 (SU(2) N_f=1, minuscule flavour)")
        print("  coeff ring    :", S.coefficient_ring())
        print("  mutable quiver:", annulus_1_2_triangulation().mutable_block())
        print("  node charges  :", S.node_charges)
        print("  boundary pair :", S.boundary_pairing())
        vac = S.trace(S.identity(), 8)
        print("  vacuum index  :", {e: str(c) for e, c in sorted(vac.coeffs.items())})
