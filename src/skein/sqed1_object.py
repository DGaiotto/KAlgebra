"""`KAlgebraObject` for **SQED1** = `A_𝖖[U(1) + N_f=1]` — the
U(1)-gauged square.

The simplest gauge-theory
KAlgebra and the UV end of the RG flow that realizes the pentagon
(`pentagon_square_rgkalg.PentagonSquareKAlg` flows over this algebra).

Four certified presentations (all DOp-free — no `abelianized_torus`
code):

* ``'intrinsic'`` — `Sqed1KAlg`: URQTorus-backed (`u_± = U_{±1}` on
  `MatterURQTorus(1,1)`; multiply = the rational 2-cocycle at the μ→1
  face, trace = the magnetic-0 matter-measure residue), satisfying
  `u_+u_- = 1 + 𝖖v`, `u_-u_+ = 1 + 𝖖⁻¹v`, `u_±v = 𝖖^{±2} v u_±`.
* ``'bps'`` — `BPSKAlgebra` on the O→F quiver (1 dynamical node + 1
  frozen flavour), the `k = 0` corner of the `U1A1AoddKAlg` family.
* ``'cone'`` — `U1SquareKAlg`: the cone-monomial (`QTCone`) presentation
  — "the U(1)-gauged square" — in the same `ConeKAlgebra` tier as the
  pentagon.  Same `(m, n)` labels and ρ as the intrinsic; it *is* the
  torus/abelianized presentation (the URQTorus matter realization would
  be essentially identical, so it is not separately wired).
* ``'skein-pinned'`` — `SkeinSquareKAlg`: the stated-SQUARE pinned
  skein realization, on the quantum torus over
  Γ~ = Z ⊕ Z⁴ of `PinnedPolygon(4)`.  The dressed diagonals carry the
  residual pins ±μ — the square cannot unpin (the χ = (1,-1,1,-1)
  obstruction; 't Hooft half-quantization) — yet satisfy the SQED1
  exchange pair exactly, and multiply is genuinely pinned-side (peel
  against the intrinsic-hinted label set, normalization δ ≡ 0
  asserted verbatim on every product).  Same `(m, n)` labels; ρ/trace
  transported from the intrinsic.

Witnesses (each verified by the full `KAlgebraIso` battery — unit,
round-trip, multiplicativity, ρ-equivariance, trace-equivariance):

* ``intrinsic ↔ bps`` — the label bijection `(m, n) ↔ (n, −m)`
  (`sqed1_bps_iso`).
* ``intrinsic ↔ cone`` — the identity on labels (`U1SquareKAlg` shares
  `Sqed1KAlg`'s `(m, n)` labels, ρ, multiply, and trace verbatim).
* ``bps ↔ cone`` — the composite through the intrinsic hub, stored
  explicitly so the groupoid is a full triangle and the coherence
  certificate is non-vacuous.
* ``skein-pinned ↔ intrinsic`` — the identity on labels
  (`SkeinSquareKAlg.build_iso()`; verifying it tests the PINNED
  structure constants against the intrinsic's — non-circular).
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from kalgebra import Element
from kalgebra_iso import KAlgebraIso
from kalgebra_object import KAlgebraObject
from laurent_poly import LaurentPoly
from kalgebra_samples import Sqed1KAlg
from bps_kalgebra import BPSKAlgebra
from u1_square_kalg import U1SquareKAlg


__all__ = ["sqed1_object"]

_ONE = LaurentPoly.one()
_B_SQED1 = [[0, 1], [-1, 0]]
_NODE_CHARGES = [(1, 0)]  # 1 dynamical; (0, 1) is the frozen flavour


def sqed1_object() -> KAlgebraObject:
    """The abstract `A_𝖖[U(1)+N_f=1]` as a `KAlgebraObject`."""
    obj = KAlgebraObject("A_q[U(1)+Nf=1]")

    intrinsic = Sqed1KAlg()
    obj.add_realization("intrinsic", intrinsic,
                        {"multiply-fast", "trace-exact"})

    bps = BPSKAlgebra(pairing=_B_SQED1, node_charges=_NODE_CHARGES,
                      verify="off")
    obj.add_realization("bps", bps, {"chart", "trace-exact", "rg"})
    obj.add_iso(
        "intrinsic", "bps",
        KAlgebraIso(
            intrinsic, bps,
            forward_label_map=lambda l: Element({(l[1], -l[0]): _ONE}),
            inverse_label_map=lambda c: Element({(-c[1], c[0]): _ONE}),
            name="sqed1[intrinsic→bps]  (k=0 of U1A1AoddKAlg)"))

    cone = U1SquareKAlg()
    obj.add_realization("cone", cone, {"multiply-fast", "trace-exact"})
    obj.add_iso(
        "intrinsic", "cone",
        KAlgebraIso(
            intrinsic, cone,
            forward_label_map=lambda l: Element({l: _ONE}),
            inverse_label_map=lambda l: Element({l: _ONE}),
            name="sqed1[intrinsic→cone]  (identity on labels)"))

    # Close the triangle: a direct bps ↔ cone edge (composite through the
    # intrinsic hub) so the coherence certificate is non-vacuous.
    obj.add_iso("bps", "cone", obj.iso("bps", "cone"))

    # The stated-SQUARE pinned skein realization: the dressed
    # diagonals of `PinnedPolygon(4)` — pin-0 dressing is impossible
    # (the χ obstruction), so they keep the residual pins ±μ, but the
    # SQED1 exchange pair holds exactly and multiply is genuinely
    # pinned-side (δ ≡ 0 asserted against the intrinsic on every
    # product).  Built ON the registered intrinsic instance, so the
    # witness endpoints match.
    from skein_square_kalg import SkeinSquareKAlg
    sk = SkeinSquareKAlg(intrinsic=intrinsic)
    obj.add_realization("skein-pinned", sk, {"geometric"})
    obj.add_iso("skein-pinned", "intrinsic", sk.build_iso())

    # The unified SkeinKAlgebra instance: derived cone multiply over the pinned
    # engine's cross-products, cone combinatorics from the registered
    # 'cone' (U1SquareKAlg) instance — endpoints shared, so the witness
    # is the identity on the (m, n) labels.
    from skein_kalgebra import U1SquareSkeinKAlgebra
    skc = U1SquareSkeinKAlgebra(cone=cone, engine=sk)
    obj.add_realization("skein-cone", skc, {"geometric", "cone"})
    obj.add_iso("skein-cone", "cone", skc.build_iso(cone))

    return obj
