"""
u1square_sample_cone_iso.py
===========================

A **certified, engine-free** ``KAlgebraIso`` identifying two presentations
of SQED N_f=1 = the U(1)-gauged ``[A_1, A_1]`` Argyres-Douglas theory:

  * the **cone** presentation  ``U1SquareKAlg`` (a ``ConeKAlgebra``; this
    Step-2 package), as the iso **source**, and
  * the **direct sample** presentation ``SQED1SampleKAlgebra``
    (Step-1, ``src/samples``), as the iso **target**.

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

Imports
-------
This module imports ``kalgebra_iso`` and ``samples`` (Step-1: ``src/core`` /
``src/samples``) and ``u1_square_kalg`` (Step-2: ``src/cone``) by flat name.
``run_tests.py`` / ``conftest.py`` put every ``src/<layer>/`` directory on the
path, and the module bootstraps the path itself so it also imports standalone.
"""
from __future__ import annotations

import os
import sys

# Put every src/<layer>/ directory on sys.path (the project's bare-name import
# convention) so this module also imports standalone from a checkout;
# run_tests.py / conftest.py do the same for the full gate.
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _root, _dirs, _ in os.walk(_SRC):
    _dirs[:] = [_d for _d in _dirs if _d != "__pycache__"]
    if _root not in sys.path:
        sys.path.insert(0, _root)

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
