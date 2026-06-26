"""
pentagon_sample_cone_iso.py
===========================

A **certified, engine-free** ``KAlgebraIso`` identifying two presentations
of the pentagon algebra ``K_𝖖([A_1, A_2])`` (= A_2 Argyres-Douglas /
Yang-Lee / M(2,5)):

  * the **cone** presentation  ``FinitePentagonKAlgebra`` (a
    ``ConeKAlgebra``; this Step-2 package), as the iso **source**, and
  * the **direct sample** presentation ``PentagonSampleKAlgebra``
    (the Step-1 ``export/KAlgebra`` package), as the iso **target**.

Both presentations are *closed-form* — neither this module nor the two
algebras it bridges import any realisation-spine engine
(no ``rgkalgebra`` / ``bps_kalgebra`` / quantum-torus backend).  The iso
is built purely from a **mult-gen correspondence** and certified by
``KAlgebraIso.verify_all`` (see ``test_sample_cone_iso.py``).

Cross-package import contract
-----------------------------
This module imports ``kalgebra_iso`` and ``samples`` from the **Step-1**
package (``export/KAlgebra``) and the cone algebra from the **Step-2**
package (``export/ConeKAlgebra``).  Run everything with BOTH packages on
the path::

    PYTHONPATH=/abs/.../export/KAlgebra:/abs/.../export/ConeKAlgebra

(The two packages ship byte-identical copies of the shared substrate —
``kalgebra.py`` / ``laurent_poly.py`` / ``zplus_ring.py`` — so with both
on ``sys.path`` a single ``kalgebra`` module is imported and ``Element``
is the *same* class object on both sides; there is no cross-package
class-identity hazard.)  As a convenience this module also appends the
two sibling ``export/*`` directories to ``sys.path`` when it can locate
them, so ``import`` works from a checkout without setting ``PYTHONPATH``
explicitly — but the documented ``PYTHONPATH`` above is the contract.

The correspondence
------------------
``FinitePentagonKAlgebra`` has 5 multiplicative generators indexed
``0..4`` with lattice vectors
``((-1,-1), (-1,0), (0,-1), (0,1), (1,0))`` and the ρ-permutation
``{0:3, 1:4, 2:1, 3:2, 4:0}`` (cone ρ cycles the rays
``0→3→2→1→4→0``).  ``PentagonSampleKAlgebra`` has generators
``L_i = (i,1,0)``, ``i ∈ Z/5``, with ``ρ: L_i ↦ L_{i+2}`` (sample ρ
cycles the indices ``0→2→4→1→3→0``).

A ``KAlgebraIso`` must intertwine ρ, so the mult-gen map
``σ: cone-gen ↦ sample index`` is a *cyclic* correspondence solving

    σ(ρ_cone(g)) == σ(g) + 2  (mod 5).

There are exactly **five** bijections satisfying this (the ρ-orbit of a
single iso — ρ generates the pentagon's Z/5 automorphism group); all
five certify under ``verify_all``.  We pin the natural representative
(cone-gen ``0 ↦ L_0``, cone-gen ``1 ↦ L_1``):

    σ = {0: 0, 1: 1, 2: 4, 3: 2, 4: 3}

i.e. the mult-gen images

    g=0 (-1,-1) ↦ L_0     g=1 (-1,0) ↦ L_1     g=2 (0,-1) ↦ L_4
    g=3 (0,1)   ↦ L_2     g=4 (1,0)  ↦ L_3 .

Forward / inverse label maps
----------------------------
The **forward** map (cone → sample) is the cone-data-driven lift
``KAlgebraIso.from_cone_mult_gen_map``: a source cone label
``L_{(gens, powers)}`` is the ordered mult-gen product in
``cone_data().canonical_cone_order`` with the convention phase absorbed,
each mult-gen replaced by its sample image.

The **inverse** map (sample → cone) is hand-supplied (the cone-data
factory derives only the forward direction).  A sample label ``(i,a,b)``
denotes ``𝖖^{ab} L_i^a L_{i+1}^b``; since an iso is an algebra
homomorphism,

    inverse((i,a,b)) = 𝖖^{ab} · cone(L_i)^a · cone(L_{i+1})^b

computed with the **cone**'s own ``multiply`` (which absorbs the cone
cocycle phases), where ``cone(L_j)`` is the cone mult-gen ``σ⁻¹(j)`` as a
native cone ``Element``.
"""
from __future__ import annotations

import os
import sys

# --- Make both sibling export packages importable (convenience; the
#     documented contract is to set PYTHONPATH to both dirs). -----------
_HERE = os.path.dirname(os.path.abspath(__file__))                 # .../export/ConeKAlgebra
_EXPORT_ROOT = os.path.dirname(_HERE)                              # .../export
for _sib in ("KAlgebra", "ConeKAlgebra"):
    _p = os.path.join(_EXPORT_ROOT, _sib)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from laurent_poly import LaurentPoly                               # shared substrate
from kalgebra import Element                                       # shared substrate
from kalgebra_iso import KAlgebraIso, _mul                         # Step-1 package
from samples import PentagonSampleKAlgebra                         # Step-1 package
from finite_pentagon_kalg import FinitePentagonKAlgebra            # Step-2 package

_ONE = LaurentPoly.one()

# The pinned, ρ-intertwining mult-gen correspondence  cone-gen -> sample index.
# (One of the five ρ-rotations that certify; this is the natural anchoring
#  0 -> L_0, 1 -> L_1.)
PENTAGON_SIGMA: dict[int, int] = {0: 0, 1: 1, 2: 4, 3: 2, 4: 3}


def build_pentagon_sample_cone_iso(name: str | None = None) -> KAlgebraIso:
    """Return the certified, engine-free ``KAlgebraIso``

        FinitePentagonKAlgebra (cone, source)  ↔  PentagonSampleKAlgebra (target)

    built from the pinned mult-gen correspondence ``PENTAGON_SIGMA``.

    The returned iso is **not** verified inside this function (build is
    cheap and side-effect-free); run ``verify_all`` against it — see
    ``test_sample_cone_iso.py`` — to obtain the certificate.  The cone
    side is ``FinitePentagonKAlgebra``; it certifies, so the
    ``A1A2kKAlg(1)`` fall-back is not needed.
    """
    sample = PentagonSampleKAlgebra()
    cone = FinitePentagonKAlgebra()
    cd = cone.cone_data()
    mult_gens = list(cd.mult_gens())                  # [0, 1, 2, 3, 4]

    sigma = PENTAGON_SIGMA
    inv_sigma = {v: k for k, v in sigma.items()}      # sample index -> cone gen

    # Forward images: cone mult-gen g -> sample generator L_{sigma[g]}.
    sampleL = {i: Element({(i, 1, 0): _ONE}) for i in range(5)}
    mult_gen_forward = {g: sampleL[sigma[g]] for g in mult_gens}

    # --- Inverse direction (sample label -> cone Element). -------------
    cone_ident = Element({cone.identity(): _ONE})

    def _cone_gen_elem(j: int) -> Element:
        """The cone mult-gen σ⁻¹(j mod 5) as a native cone ``Element``."""
        g = inv_sigma[j % 5]
        native = cd.from_cone_label(frozenset({g}), {g: 1})
        return Element({native: _ONE})

    def target_label_to_source(label) -> Element:
        # Sample label (i,a,b) = 𝖖^{ab} L_i^a L_{i+1}^b ; the iso is an
        # algebra hom, so its inverse is the same product of preimages,
        # evaluated with the CONE's multiply (absorbing cone cocycles).
        i, a, b = label
        res = cone_ident
        gi = _cone_gen_elem(i)
        for _ in range(a):
            res = _mul(cone, res, gi)
        gi1 = _cone_gen_elem(i + 1)
        for _ in range(b):
            res = _mul(cone, res, gi1)
        if a * b != 0:
            res = res * LaurentPoly({a * b: 1})
        return res

    return KAlgebraIso.from_cone_mult_gen_map(
        source=cone,
        target=sample,
        mult_gen_forward=mult_gen_forward,
        target_label_to_source=target_label_to_source,
        name=name or "FinitePentagonKAlgebra (cone) ↔ PentagonSampleKAlgebra",
    )


__all__ = ["build_pentagon_sample_cone_iso", "PENTAGON_SIGMA"]


if __name__ == "__main__":
    iso = build_pentagon_sample_cone_iso()
    print(repr(iso))
    print("mult-gen map (cone g -> sample index):", PENTAGON_SIGMA)
