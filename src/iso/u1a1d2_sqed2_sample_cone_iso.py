"""
u1a1d2_sqed2_sample_cone_iso.py
===============================

A **certified, engine-free** ``KAlgebraIso`` identifying two presentations
of SQED₂ = [A₁, D₂] = U(1) gauge + two charged hypers = ``U_𝖖(𝔰𝔩₂)``:

  * the **cone** presentation  ``U1A1D2ConeKAlgebra`` (a ``ConeKAlgebra``;
    this Step-2 package), as the iso **source**, and
  * the **direct sample** presentation ``SQED2SampleKAlgebra``
    (Step-1, ``src/samples``), as the iso **target**.

This is the SQED₂ companion to the pentagon / A1A1 sample↔cone isos.  The
two presentations are the **same algebra** under a simple **relabeling
bijection** of the canonical basis — the cone's gauge ``(m, n)`` + spin
``k`` 3-tuple ``(m, n, k)`` versus the sample's ``(gauge, k)`` with
``gauge ∈ {('K',n), ('E',a,b), ('F',a,b)}``:

    m > 0 :  (m, n, k)  ↔  (('E', m, n),  k)        # E_{m,n} · χ_k
    m < 0 :  (m, n, k)  ↔  (('F', −m, n), k)        # F_{−m,n} · χ_k
    m = 0 :  (m, n, k)  ↔  (('K', n),     k)        # Kⁿ · χ_k

with the SU(2) spin ``k`` identical on both sides and the coefficient
``LaurentPoly`` carried through unchanged (each map is a single-term,
coefficient-1 relabeling).  Both presentations share the same ``ρ``
(Lusztig's braid) and the same ``SU2ZPlusRing`` coefficient ring, and
their two ``multiply`` engines (the cone's generic cone-monomial reducer
— de-risked to reproduce ``U_𝖖(𝔰𝔩₂)`` exactly — versus the sample's PBW
straightener ``_uqsl2_multiply``) and two ``trace`` routes (the cone's
spine-free delegation versus the sample's direct ``[x^n] G(x,μ)``) agree
term by term, which is exactly what ``KAlgebraIso.verify_all`` certifies.

Both presentations are *closed-form* and neither imports any realisation
spine (no ``rgkalgebra`` / ``bps_kalgebra`` / quantum-torus backend).  In
particular ``U1A1D2ConeKAlgebra``'s ``m=0`` Cartan-sector Schur-index
trace is delegated to the Step-1 ``SQED2SampleKAlgebra`` (the spine-free
``[x^n] G(x,μ)``).

Imports
-------
This module imports ``kalgebra_iso`` and ``samples`` (Step-1: ``src/core`` /
``src/samples``) and ``u1a1d2_cone_kalg`` (Step-2: ``src/cone``) by flat name.
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

from laurent_poly import LaurentPoly                               # shared substrate
from kalgebra import Element                                       # shared substrate
from kalgebra_iso import KAlgebraIso                               # Step-1 package
from samples import SQED2SampleKAlgebra                            # Step-1 package
from u1a1d2_cone_kalg import U1A1D2ConeKAlgebra                    # Step-2 package

_ONE = LaurentPoly.one()


def _cone_to_sample_label(label):
    """Cone ``(m, n, k)`` → sample ``(gauge, k)``."""
    m, n, k = label
    if m > 0:
        return (("E", m, n), k)
    if m < 0:
        return (("F", -m, n), k)
    return (("K", n), k)


def _sample_to_cone_label(label):
    """Sample ``(gauge, k)`` → cone ``(m, n, k)``."""
    gauge, k = label
    kind = gauge[0]
    if kind == "E":
        _, a, b = gauge
        return (a, b, k)
    if kind == "F":
        _, a, b = gauge
        return (-a, b, k)
    if kind == "K":
        _, n = gauge
        return (0, n, k)
    raise ValueError(f"_sample_to_cone_label: unknown gauge {gauge!r}")


def build_u1a1d2_sqed2_sample_cone_iso(name: str | None = None) -> KAlgebraIso:
    """Return the certified, engine-free ``KAlgebraIso``

        U1A1D2ConeKAlgebra (cone, source)  ↔  SQED2SampleKAlgebra (target)

    as the relabeling bijection ``(m, n, k) ↔ (gauge, k)`` (see module
    docstring).

    The returned iso is **not** verified inside this function (build is
    cheap and side-effect-free); run ``verify_all`` against it — see
    ``test_sample_cone_iso.py`` — to obtain the certificate (that the two
    presentations' ``multiply`` / ``ρ`` / ``trace`` coincide on the
    corresponding labels, in both directions)."""
    cone = U1A1D2ConeKAlgebra()
    sample = SQED2SampleKAlgebra()

    def forward(cone_label):                  # cone (m,n,k) -> sample Element
        return Element({_cone_to_sample_label(cone_label): _ONE})

    def inverse(sample_label):                # sample (gauge,k) -> cone Element
        return Element({_sample_to_cone_label(sample_label): _ONE})

    return KAlgebraIso(
        source=cone,
        target=sample,
        forward_label_map=forward,
        inverse_label_map=inverse,
        name=name or "U1A1D2ConeKAlgebra (cone) ↔ SQED2SampleKAlgebra",
    )


__all__ = [
    "build_u1a1d2_sqed2_sample_cone_iso",
    "_cone_to_sample_label",
    "_sample_to_cone_label",
]


if __name__ == "__main__":
    iso = build_u1a1d2_sqed2_sample_cone_iso()
    print(repr(iso))
    print("source (cone)  :", type(iso.source).__name__)
    print("target (sample):", type(iso.target).__name__)
    print("correspondence : (m, n, k) ↔ (gauge, k)  [E_{m,n}/F_{-m,n}/Kⁿ · χ_k]")
