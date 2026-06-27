"""`U1SquareKAlg` — SQED N_f=1 = U(1)-gauged [A_1, A_1] AD theory,
encoded with BPS-quiver-square geometry (the **A1A1 cone standalone**).

Algebraically identical to `SQED1SampleKAlgebra` (the Step-1 direct
sample); this class exposes the **cone-monomial presentation** with
`QTCone` structure, fitting U1SquareKAlg into the `ConeKAlgebra` tier
alongside Pentagon, A1A2k, U1A1Aodd.  The certified Step-1↔Step-2
correspondence is `u1square_sqed1_sample_cone_iso.py`
(`U1SquareKAlg (cone) ↔ SQED1SampleKAlgebra (sample)`), an
identity-on-labels `KAlgebraIso` (both presentations use the same
`(m, n)` labels, the same `ρ`, and the same `TrivialZPlusRing`).

Native labels
-------------
`(m, n) ∈ Z²`:
  * m is the magnetic / charged-hyper direction (signed integer),
  * n is the gauge-monopole / v direction (signed integer).
ρ:  ρ((m, n)) = (-m, -n - max(m, 0)).
ρ² :  ρ²((m, n)) = (m, n + m).

trace handling (spine-free)
---------------------------
The Schur index vanishes for m ≠ 0 (a hypermultiplet in a U(1)-gauged
theory; only the central / Coulomb-branch labels contribute).  The
standard `ConeKAlgebra.trace` walks ρ²-orbits for canonicalisation, but
for `U1SquareKAlg` those orbits are *infinite* when m ≠ 0 (ρ² shifts n
by m per application), so we supply the closed-form drift-quotient
`_canonical_rho2_orbit_rep` and a `_trace_residual` that dispatches
directly on the native seed:
  * m ≠ 0: trace = 0 (no canonicalisation needed; the seed family
    collapses to zero trivially).
  * m = 0: Tr(v^n) = [x^n] (𝖖²;𝖖²)_∞² E_𝖖(x) E_𝖖(x⁻¹), the SQED_1
    Schur index — **delegated to the Step-1 `SQED1SampleKAlgebra`**
    (the same algebra, same labels), keeping a single spine-free source
    of truth for the Nahm-sum closed form.  (The source-repo class
    delegated this to `kalgebra_samples.Sqed1KAlg`, which drags in the
    quantum-torus realisation engine; the Step-1 sample computes the
    identical series with no engine — see `_sqed1_helper`.)
"""

from __future__ import annotations

import sys
import os

# Put every src/<layer>/ directory on sys.path (the project's bare-name import
# convention) so this module also imports standalone from a checkout;
# run_tests.py / conftest.py do the same for the full gate.
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _root, _dirs, _ in os.walk(_SRC):
    _dirs[:] = [_d for _d in _dirs if _d != "__pycache__"]
    if _root not in sys.path:
        sys.path.insert(0, _root)

from kalgebra import KAlgebra, Element
from cone_kalgebra import ConeKAlgebra
from zplus_ring import ZPlusRing, RPowerSeries, TrivialZPlusRing
from u1_square_cone_data import U1SQUARE_CONE_DATA


__all__ = ["U1SquareKAlg"]


class U1SquareKAlg(ConeKAlgebra):
    """SQED N_f=1 K-algebra, cone-monomial-presented over `QTCone`
    (the A1A1 cone standalone).

    See module docstring for the structure and the spine-free `trace`
    (m ≠ 0 ⇒ 0; m = 0 ⇒ the SQED_1 index via the Step-1 sample).
    """

    _R = TrivialZPlusRing()

    def coefficient_ring(self) -> ZPlusRing:
        return self._R

    def identity(self):
        return (0, 0)

    def cone_data(self):
        return U1SQUARE_CONE_DATA

    # ρ and ρ_inverse identical to SQED1SampleKAlgebra's.
    def rho(self, a):
        m, n = a
        return (-m, -n - max(m, 0))

    def rho_inverse(self, a):
        m, n = a
        m_orig = -m
        n_orig = -n - max(m_orig, 0)
        return (m_orig, n_orig)

    # ----- ρ²-orbit canonicalisation (closed-form drift quotient) ----

    def _canonical_rho2_orbit_rep(self, label):
        """Closed-form canonical orbit representative.

        ρ²((m, n)) = (m, n + m), so the orbit of (m, n) under ρ² is
        `{(m, n + km) : k ∈ Z}`.

        For m ≠ 0: unique canonical representative with n in [0, |m|),
                   computed as (m, n mod |m|).
        For m == 0: orbit is the singleton {(0, n)} (ρ² fixes it).
        """
        m, n = label
        if m == 0:
            return (0, n)
        return (m, n % abs(m))

    # ----- Layer-2 trace residual ---------------------------------------
    #
    # Canonical ρ²-orbit seeds emitted by simplify_trace_via_cone_data
    # after _canonical_rho2_orbit_rep folding:
    #
    #   * (0, n) for n ∈ Z    → Tr(v^n), Nahm-sum closed form
    #     (delegated to the Step-1 SQED1SampleKAlgebra).
    #   * (m, n_canon) for m ≠ 0, n_canon ∈ [0, |m|)
    #                          → Tr = 0 (the non-central labels vanish;
    #                                    equivalently, follows from
    #                                    ρ²-cyclicity since Tr(v · O)
    #                                    = 0 unless O q-commutes with v).
    #
    # Trace and multiply are otherwise inherited universally from
    # ConeKAlgebra.

    def _trace_residual(self, seed_label, K):
        m, n = seed_label
        if m != 0:
            # Trace vanishes for non-central seeds (m ≠ 0).  Whatever
            # n_canon happens to be (= n % abs(m), in [0, |m|)), the
            # value is 0.
            return RPowerSeries(self._R, {}, K)
        # m = 0: Tr(v^n).  Delegate to the Step-1 SQED1SampleKAlgebra's
        # closed-form (Nahm-sum) trace; identical algebra, same labels,
        # spine-free.
        return self._sqed1_helper().trace((0, n), K=K)

    def _sqed1_helper(self):
        """Cached SQED1SampleKAlgebra (Step 1) for trace delegation on
        the m=0 seeds — the spine-free source of the SQED_1 index."""
        cached = getattr(self, "_sqed1_cache", None)
        if cached is None:
            from samples import SQED1SampleKAlgebra
            cached = SQED1SampleKAlgebra()
            self._sqed1_cache = cached
        return cached
