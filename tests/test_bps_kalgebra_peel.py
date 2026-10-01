"""Sanity tests for the peeling F-solver and QNumberPoly-typed F-cache.

Focuses on the small examples called out by the user: pentagon and a
few small-charge cases on hexagon / SU(3).  Verifies:

* F-coefficients are QNumberPoly in the cache.
* Public F() still returns LaurentPoly (backward compat).
* The two representations expand to the same Laurent polynomial.
* Spec-mode and recipe-mode solvers agree on small examples.
"""

import sys
sys.path.insert(0, "legacy")
sys.path.insert(0, ".")

from bps_kalgebra import BPSKAlgebra
from q_number_poly import QNumberPoly
from laurent_poly import LaurentPoly


PENTAGON_B = [[0, 1], [-1, 0]]
PENTAGON_NODES = [(1, 0), (0, 1)]


def test_F_cache_stores_qnumberpoly():
    A = BPSKAlgebra(pairing=PENTAGON_B, node_charges=PENTAGON_NODES)
    F_qn = A.F_qn((0, 1))
    for delta, qn in F_qn.items():
        assert isinstance(qn, QNumberPoly), (delta, type(qn))


def test_F_public_returns_laurent():
    A = BPSKAlgebra(pairing=PENTAGON_B, node_charges=PENTAGON_NODES)
    F = A.F((0, 1))
    for delta, lp in F.items():
        assert isinstance(lp, LaurentPoly), (delta, type(lp))


def test_qn_and_laurent_agree_pentagon():
    A = BPSKAlgebra(pairing=PENTAGON_B, node_charges=PENTAGON_NODES)
    for gamma in [(0, 1), (1, 0), (1, 1), (-1, 0), (0, -1), (-1, -1)]:
        F = A.F(gamma)
        F_qn = A.F_qn(gamma)
        assert set(F) == set(F_qn)
        for delta in F:
            assert F[delta] == F_qn[delta].to_laurent(), (gamma, delta)


def test_F_coefficients_are_palindromic_pentagon():
    A = BPSKAlgebra(pairing=PENTAGON_B, node_charges=PENTAGON_NODES)
    for gamma in [(0, 1), (1, 0), (1, 1), (-1, 0), (-1, -1), (-2, -2)]:
        F = A.F(gamma)
        for delta, lp in F.items():
            for e, c in lp._coeffs.items():
                assert lp._coeffs.get(-e, 0) == c, (gamma, delta, e)


def test_spec_recipe_agree_pentagon():
    A_spec = BPSKAlgebra(pairing=PENTAGON_B, node_charges=PENTAGON_NODES)

    # Recipe mode: feed the spec-derived s_coefficient back in.
    from spec_sigma import sigma_forward as _sfwd, sigma_inverse as _sinv
    from nahm_local import s_gamma_habiro
    spec_t = A_spec.spec
    kmat = A_spec._kmat

    def s_fn(g):
        return s_gamma_habiro(tuple(g), spec_t, kmat)

    def sfwd_fn(g):
        return tuple(_sfwd(A_spec.lattice, spec_t, tuple(g)))

    def sinv_fn(g):
        return tuple(_sinv(A_spec.lattice, spec_t, tuple(g)))

    A_rec = BPSKAlgebra(
        pairing=PENTAGON_B, node_charges=PENTAGON_NODES,
        s_coefficient=s_fn, sigma=sfwd_fn, sigma_inverse=sinv_fn,
        cone_gens=A_spec.cone_gens,
    )
    for gamma in [(0, 1), (1, 0), (1, 1), (-1, 0)]:
        assert A_rec.F(gamma) == A_spec.F(gamma), gamma


# ---------- SU(2) / hexagon / SU(3) smoke tests ----------

def test_su2_F_small_charges():
    # SU(2) BPS quiver: pairing matrix [[0,2],[-2,0]], nodes (1,0), (0,1).
    A = BPSKAlgebra(pairing=[[0, 2], [-2, 0]], node_charges=[(1, 0), (0, 1)])
    # Just check a few small charges solve and have palindromic coefficients.
    for gamma in [(1, 0), (0, 1), (1, 1)]:
        F = A.F(gamma)
        assert gamma in F  # the diagonal term
        for delta, lp in F.items():
            for e, c in lp._coeffs.items():
                assert lp._coeffs.get(-e, 0) == c, (gamma, delta, e)


def test_su3_F_small_charges():
    # SU(3) BPS quiver: pairing [[0,1,1],[-1,0,1],[-1,-1,0]] (A2 Cartan-like).
    A = BPSKAlgebra(
        pairing=[[0, 1, 1], [-1, 0, 1], [-1, -1, 0]],
        node_charges=[(1, 0, 0), (0, 1, 0), (0, 0, 1)],
    )
    for gamma in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
        F = A.F(gamma)
        assert gamma in F


# ---------- Pentagon known F values ----------

def test_pentagon_F_known_diagonal_coefficient():
    """F_γ always has coefficient 1 at δ = γ (definition: leading X_γ)."""
    A = BPSKAlgebra(pairing=PENTAGON_B, node_charges=PENTAGON_NODES)
    for gamma in [(1, 0), (0, 1), (1, 1), (-1, 0), (0, -1)]:
        F_qn = A.F_qn(gamma)
        assert F_qn[gamma] == QNumberPoly.one(), gamma


if __name__ == "__main__":
    passed = 0
    for name, func in list(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
            print(f"  PASS: {name}")
            passed += 1
    print(f"\nAll {passed} tests passed!")
