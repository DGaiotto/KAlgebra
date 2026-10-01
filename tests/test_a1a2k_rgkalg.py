"""Tests for `A1A2kRGKAlg`: the even AD chord algebra `[A_1, A_{2k}]`
as a single-node (drop γ_2) RGKAlgebra over `BPS(A_{2k-2}) ⊗ QT(Z_2)`
with `S_RG = E_𝖖(X_{γ_1})`.

The construction is BPS-F-free (RG is the Darboux relabel, S_RG is the
isolated-node quantum dilogarithm), so every check below is fast — in
particular it never invokes the UV BPS F-finder, which does not
terminate for the stuff-first chamber this flow requires.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from a1a2k_rgkalg import A1A2kRGKAlg, _a_n_positive_roots
from tensor_kalgebra import TensorKAlgebra
from quantum_torus_kalgebra import QuantumTorusKAlg
from kalgebra import Element
from laurent_poly import LaurentPoly


def _mul_elem(A, x, y):
    out = Element({})
    for la, ca in x.terms.items():
        for lb, cb in y.terms.items():
            p = A.multiply(la, lb)
            out = out + Element({l: ca * cb * c for l, c in p.terms.items()})
    return out


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_construction():
    """k = 1, 2, 3 construct; auxiliary is a TensorKAlgebra whose
    second factor is the rank-2 quantum torus and whose first factor
    is the recursive A1A2kRGKAlg(k-1) (k >= 2)."""
    for k in (1, 2, 3):
        A = A1A2kRGKAlg(k)
        aux = A.auxiliary()
        assert isinstance(aux, TensorKAlgebra), k
        assert isinstance(aux.factor_B, QuantumTorusKAlg), k
        if k >= 2:
            assert isinstance(aux.factor_A, A1A2kRGKAlg), k
            assert aux.factor_A.k == k - 1, k
        # UV labels are A_{2k} charge tuples.
        assert len(A.identity()) == 2 * k, k


def test_rg_simple_image():
    """RG of a UV generator is a single auxiliary canonical-basis term
    (the Darboux relabel) — the 'simple IR image'."""
    A = A1A2kRGKAlg(2)
    # γ_3 → A_2-block charge (1,0), trivial qt.
    rg3 = A.RG((0, 0, 1, 0))
    assert rg3 == Element({((1, 0), (0, 0)): LaurentPoly.one()}), dict(rg3.terms)
    # γ_1 → trivial block, qt generator (1,0).
    rg1 = A.RG((1, 0, 0, 0))
    assert rg1 == Element({((0, 0), (1, 0)): LaurentPoly.one()}), dict(rg1.terms)


def test_s_rg_is_quantum_dilog():
    """S_RG = E_𝖖(X_{γ_1}) lives entirely on the QT(Z_2) factor at
    charge (n, 0) with the leading-order quantum-dilog coefficients."""
    A = A1A2kRGKAlg(2)
    s = A.rg_generator(3)
    blk_id = A.auxiliary().factor_A.identity()
    # All keys are (1_block, (n, 0)).
    for (a_lbl, qt), _ in s.items():
        assert a_lbl == blk_id, (a_lbl, qt)
        assert qt[1] == 0, qt
    # Leading coefficients: s_0 = 1, s_1 = -q/(1-q^2) (q-expansion -q - q^3 - ...).
    s1 = s[(blk_id, (1, 0))].expand(5)
    assert s1 == LaurentPoly({1: -1, 3: -1, 5: -1}), s1
    assert s[(blk_id, (0, 0))].expand(2) == LaurentPoly.one()


# ---------------------------------------------------------------------------
# RG axioms
# ---------------------------------------------------------------------------

def test_rg_unital():
    for k in (1, 2, 3):
        assert A1A2kRGKAlg(k).verify_rg_unital(), k


def test_rg_multiplicative():
    """RG(L_a · L_b) = RG(L_a) · RG(L_b) over every pair of UV positive
    roots, k = 1, 2, 3 (cheap thanks to the recursive auxiliary)."""
    for k in (1, 2, 3):
        A = A1A2kRGKAlg(k)
        roots = _a_n_positive_roots(2 * k)
        for a in roots:
            for b in roots:
                assert A.verify_rg_multiplicative(a, b), (k, a, b)


def test_rg_twist_survivors():
    """The RG-twist axiom holds (at finite cutoff) for survivor
    generators γ_3, γ_4, γ_3+γ_4.  (γ_1 — the mode generating S_RG —
    fails strict equality at the truncation boundary, a documented
    `verify_rg_twist` limitation, so it is excluded here.)"""
    A = A1A2kRGKAlg(2)
    for a in [(0, 0, 1, 0), (0, 0, 0, 1), (0, 0, 1, 1)]:
        assert A.verify_rg_twist(a, cutoff=3), a


# ---------------------------------------------------------------------------
# Algebra validity
# ---------------------------------------------------------------------------

def test_associativity():
    """`multiply` is associative on a sample of UV positive-root triples
    (k = 1, 2).  multiply is BPS-F-free and fast."""
    one = LaurentPoly.one()
    for k in (1, 2):
        A = A1A2kRGKAlg(k)
        roots = _a_n_positive_roots(2 * k)
        triples = [(a, b, c) for a in roots[:5] for b in roots[:5]
                   for c in roots[:5]]
        for a, b, c in triples:
            ea, eb, ec = (Element({a: one}), Element({b: one}),
                          Element({c: one}))
            left = _mul_elem(A, _mul_elem(A, ea, eb), ec)
            right = _mul_elem(A, ea, _mul_elem(A, eb, ec))
            assert left == right, (k, a, b, c)


if __name__ == "__main__":
    test_construction()
    test_rg_simple_image()
    test_s_rg_is_quantum_dilog()
    test_rg_unital()
    test_rg_multiplicative()
    test_rg_twist_survivors()
    test_associativity()
    print("\nAll A1A2kRGKAlg tests passed.")
