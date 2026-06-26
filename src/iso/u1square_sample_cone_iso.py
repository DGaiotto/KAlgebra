"""
u1square_sample_cone_iso.py
===========================

A **certified, engine-free** ``KAlgebraIso`` identifying two presentations
of SQED N_f=1 = the U(1)-gauged ``[A_1, A_1]`` Argyres-Douglas theory:

  * the **cone** presentation  ``U1SquareKAlg`` (a ``ConeKAlgebra``; this
    Step-2 package), as the iso **source**, and
  * the **direct sample** presentation ``SQED1SampleKAlgebra``
    (the Step-1 ``export/KAlgebra`` package), as the iso **target**.

This is the SQED1 ↔ U1Square companion to the pentagon
``KAlgebraIso`` (``pentagon_sample_cone_iso.py``).  Unlike the pentagon
case — where the cone and sample use *different* label conventions and the
correspondence is a non-trivial cyclic mult-gen map — here the two
presentations are **the same algebra on the same labels**: both use the
``(m, n)`` charge labels (``m`` the magnetic / charged-hyper charge, ``n``
the gauge-monopole power), both have ``ρ(m,n) = (−m, −n − max(m,0))``, and
both are unflavoured (``TrivialZPlusRing``).  The two ``multiply`` engines
(``U1SquareKAlg``'s generic cone-monomial reducer vs the sample's direct
``u_±/v`` straightener ``_sqed1_ureduce``) and the two traces (the cone's
ρ²-cyclicity Layer-1 reduction vs the sample's Nahm-sum index) **agree term
by term**, so the correspondence is the **identity on labels** and the
mathematical content is the certificate that the two engines coincide.

Both presentations are *closed-form* and neither imports any realisation
spine (no ``rgkalgebra`` / ``bps_kalgebra`` / quantum-torus backend).  In
particular ``U1SquareKAlg``'s ``m=0`` Schur-index seed is delegated to the
Step-1 ``SQED1SampleKAlgebra`` (the spine-free Nahm sum), *not* the
source-repo ``Sqed1KAlg`` (which drags in the quantum-torus engine).

Cross-package import contract
-----------------------------
This module imports ``kalgebra_iso`` and ``samples`` from the **Step-1**
package (``export/KAlgebra``) and ``u1_square_kalg`` from the **Step-2**
package (``export/ConeKAlgebra``).  Run everything with BOTH packages on
the path::

    PYTHONPATH=/abs/.../export/KAlgebra:/abs/.../export/ConeKAlgebra

(As a convenience this module also appends the two sibling ``export/*``
directories to ``sys.path`` when it can locate them, so ``import`` works
from a checkout without setting ``PYTHONPATH`` explicitly — but the
documented ``PYTHONPATH`` above is the contract.)
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

from kalgebra_iso import KAlgebraIso                               # Step-1 package
from samples import SQED1SampleKAlgebra                            # Step-1 package
from u1_square_kalg import U1SquareKAlg                            # Step-2 package


def build_u1square_sqed1_sample_cone_iso(name: str | None = None) -> KAlgebraIso:
    """Return the certified, engine-free ``KAlgebraIso``

        U1SquareKAlg (cone, source)  ↔  SQED1SampleKAlgebra (target)

    as the **identity on labels** (both presentations share the ``(m, n)``
    charge labels, the ``ρ(m,n) = (−m, −n − max(m,0))`` action, and the
    ``TrivialZPlusRing`` coefficient ring).

    The returned iso is **not** verified inside this function (build is
    cheap and side-effect-free); run ``verify_all`` against it — see
    ``test_sample_cone_iso.py`` — to obtain the certificate (that the two
    presentations' ``multiply`` / ``ρ`` / ``trace`` coincide on the shared
    labels, in both directions).
    """
    cone = U1SquareKAlg()
    sample = SQED1SampleKAlgebra()
    return KAlgebraIso.identity_on_labels(
        cone, sample,
        name=name or "U1SquareKAlg (cone) ↔ SQED1SampleKAlgebra",
    )


__all__ = ["build_u1square_sqed1_sample_cone_iso"]


if __name__ == "__main__":
    iso = build_u1square_sqed1_sample_cone_iso()
    print(repr(iso))
    print("source (cone)  :", type(iso.source).__name__)
    print("target (sample):", type(iso.target).__name__)
    print("correspondence : identity on (m, n) labels")
