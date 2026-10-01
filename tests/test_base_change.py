"""Tests for `RingHom` and `KAlgebra.base_change`.

Exercises:

  * `RingHom` arithmetic (apply to RElement / RLaurent / RPowerSeries).
  * `identity_hom`, `augmentation_hom`, `restriction_hom` factories.
  * `KAlgebra.base_change(phi)` lifts a source KAlgebra over R_1 to one
    over R_2 = phi.target.
  * Special case: `augmentation_hom` collapses a flavoured BPSKAlgebra
    to its gauge quotient (μ → 1).  The result over `Z` matches what
    one would compute by simply summing μ-monomial coefficients.
  * Identity base change is functorially the identity.

Run:  PYTHONPATH=. python restructuring/tests/test_base_change.py
"""

from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
_RESTRUCT = os.path.dirname(_HERE)
_REPO = _RESTRUCT  # post-flatten: _RESTRUCT is the repo root
sys.path.insert(0, _REPO)
sys.path.insert(0, _RESTRUCT)

from zplus_ring import (
    ZPlusRing, RElement, RLaurent, RPowerSeries,
    TrivialZPlusRing, AbelianZPlusRing,
    RingHom, identity_hom, augmentation_hom, restriction_hom,
)
from kalgebra import KAlgebra, Element
from kalgebra_samples import QuantumTorusZ2KAlg
from bps_kalgebra import BPSKAlgebra


# ===========================================================================
# RingHom: identity
# ===========================================================================


def test_identity_hom_on_RElement():
    R = AbelianZPlusRing(rank=2)
    phi = identity_hom(R)
    mu1 = R.basis_element((1, 0))
    mu2 = R.basis_element((0, 1))
    elem = mu1 + mu2 - R.one()
    assert phi.apply_RElement(elem) == elem


def test_identity_hom_on_RLaurent():
    R = AbelianZPlusRing(rank=1)
    phi = identity_hom(R)
    mu = R.basis_element((1,))
    L = RLaurent(R, {0: R.one(), 1: mu, -1: -R.one()})
    assert phi.apply_RLaurent(L) == L


# ===========================================================================
# RingHom: augmentation (μ → 1)
# ===========================================================================


def test_augmentation_collapses_mu_to_1():
    R = AbelianZPlusRing(rank=1)
    phi = augmentation_hom(R)
    mu = R.basis_element((1,))
    mu_inv = R.basis_element((-1,))
    # μ + μ^{-1} - 1  ↦  1 + 1 - 1 = 1
    elem = mu + mu_inv - R.one()
    out = phi.apply_RElement(elem)
    Rt = TrivialZPlusRing()
    assert out == Rt.one(), f"augmentation gives {out}, expected 1"


def test_augmentation_on_RLaurent():
    R = AbelianZPlusRing(rank=2)
    phi = augmentation_hom(R)
    mu1 = R.basis_element((1, 0))
    mu2 = R.basis_element((0, 1))
    L = RLaurent(R, {0: mu1 + mu2, 1: mu1 - R.one()})
    # μ_1 + μ_2 ↦ 2; q · (μ_1 - 1) ↦ q · 0 = 0
    out = phi.apply_RLaurent(L)
    Rt = TrivialZPlusRing()
    expected = RLaurent(Rt, {0: 2})
    assert out == expected, f"got {out}, expected {expected}"


def test_augmentation_target_is_trivial_ring():
    R = AbelianZPlusRing(rank=3)
    phi = augmentation_hom(R)
    assert isinstance(phi.target, TrivialZPlusRing)


# ===========================================================================
# RingHom: restriction (group hom α : U(1)^k → U(1)^n)
# ===========================================================================


def test_restriction_diagonal_inclusion():
    """α : U(1) → U(1)² as the diagonal embedding sends (μ_1, μ_2) ↦ μ.
    Pullback on characters: (a, b) ∈ Z² ↦ a + b ∈ Z.
    Matrix M (target_rank × source_rank = 1 × 2): [[1, 1]]."""
    src = AbelianZPlusRing(rank=2)
    tgt = AbelianZPlusRing(rank=1)
    phi = restriction_hom(src, tgt, [[1, 1]])
    # μ_1^{2} · μ_2^{-3} ↦ μ^{2 + (-3)} = μ^{-1}
    elem = src.basis_element((2, -3))
    out = phi.apply_RElement(elem)
    expected = tgt.basis_element((-1,))
    assert out == expected


def test_restriction_to_trivial_via_zero_matrix():
    """Restriction with the zero matrix is the augmentation: α = constant."""
    src = AbelianZPlusRing(rank=2)
    tgt = AbelianZPlusRing(rank=0)
    # Zero matrix: target_rank=0, source_rank=2 => empty list of rows
    phi = restriction_hom(src, tgt, [])
    elem = src.basis_element((3, -7))
    out = phi.apply_RElement(elem)
    assert out == tgt.basis_element(())   # = tgt.one()


# ===========================================================================
# KAlgebra.base_change on samples
# ===========================================================================


def test_identity_base_change_preserves_qtorus():
    """`identity_hom`-based change wraps the algebra; trace and multiply
    should give the same RPowerSeries / Element values."""
    A = QuantumTorusZ2KAlg()
    R = A.coefficient_ring()
    phi = identity_hom(R)
    A_id = A.base_change(phi)
    # Trace of vacuum agrees
    K = 4
    Tr_orig = A.trace((0, 0), K)
    Tr_id = A_id.trace((0, 0), K)
    assert Tr_id == Tr_orig
    # Multiply agrees
    p_orig = A.multiply((1, 0), (0, 1))
    p_id = A_id.multiply((1, 0), (0, 1))
    assert p_id == p_orig


def test_base_change_rejects_wrong_source_ring():
    A = QuantumTorusZ2KAlg()   # over TrivialZPlusRing
    abel = AbelianZPlusRing(rank=1)
    phi_bad = identity_hom(abel)
    try:
        A.base_change(phi_bad)
    except ValueError as e:
        assert "source" in str(e)
    else:
        raise AssertionError("expected ValueError for wrong source")


def test_augmentation_base_change_on_flavoured_bps():
    """Apply augmentation to the flavoured hexagon BPSKAlgebra.  The
    resulting KAlgebra is over Z; its trace gives the gauge-quotient
    Schur index (= μ → 1 specialization)."""
    HEX_B = [[0, 1, -1], [-1, 0, 1], [1, -1, 0]]
    HEX_N = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    A_flav = BPSKAlgebra(pairing=HEX_B, node_charges=HEX_N)
    R = A_flav.coefficient_ring()
    aug = augmentation_hom(R)
    A_quot = A_flav.base_change(aug)
    # Coefficient ring is now trivial (R = Z)
    assert isinstance(A_quot.coefficient_ring(), TrivialZPlusRing)
    # Vacuum trace is non-trivial; the q⁰ coefficient is 1 (orthonormality).
    Tr = A_quot.trace((0, 0, 0), K=2)
    assert isinstance(Tr, RPowerSeries)
    assert Tr.ring == TrivialZPlusRing()
    # Identity orbit's q⁰ coefficient remains 1 after μ → 1.
    assert Tr[0] == 1


def test_augmentation_then_identity_is_augmentation():
    """Composition: aug ∘ identity = aug.  Verify functoriality at the
    base-change level.  (We don't compose RingHoms directly yet — instead
    we verify that successively base-changing through identity-then-aug
    matches a direct base-change through aug.)"""
    HEX_B = [[0, 1, -1], [-1, 0, 1], [1, -1, 0]]
    HEX_N = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    A = BPSKAlgebra(pairing=HEX_B, node_charges=HEX_N)
    R = A.coefficient_ring()
    aug = augmentation_hom(R)
    id_ = identity_hom(R)
    A_id_then_aug = A.base_change(id_).base_change(aug)
    A_direct = A.base_change(aug)
    Tr_chain = A_id_then_aug.trace((0, 0, 0), K=3)
    Tr_direct = A_direct.trace((0, 0, 0), K=3)
    assert Tr_chain == Tr_direct, (
        f"composition broke: chain={Tr_chain}, direct={Tr_direct}"
    )


def test_augmentation_multiply_drops_mu():
    """multiply on the augmentation-base-changed algebra has Z-coefficients
    (no μ); after the design record, multiply returns Z-form Element with
    LaurentPoly coefficients in q only — μ never appears."""
    from kalgebra import LaurentPoly
    HEX_B = [[0, 1, -1], [-1, 0, 1], [1, -1, 0]]
    HEX_N = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    A_flav = BPSKAlgebra(pairing=HEX_B, node_charges=HEX_N)
    R = A_flav.coefficient_ring()
    aug = augmentation_hom(R)
    A_quot = A_flav.base_change(aug)
    p = A_quot.multiply((1, 0, 0), (0, 1, 0))
    for c in p.terms.values():
        assert isinstance(c, LaurentPoly)


# ===========================================================================
# RLaurent-coefficient sources (the widening): base_change / forget
# must push coefficients through phi (regression for the 2026-07-02
# Plan-37 finding: derived trace/verifier paths crashed with
# "RLaurent ring mismatch" on forgotten flavour-in-labels entries)
# ===========================================================================


class _RLaurentTorusKAlg(KAlgebra):
    """Minimal flavoured source whose multiply emits RLaurent
    coefficients: labels n in Z with L_n = mu^n * L_0 (one section),
    rho(n) = -n, Tr(L_n) = delta_{n,0}."""

    def __init__(self):
        self._R = AbelianZPlusRing(rank=1)

    def coefficient_ring(self):
        return self._R

    def identity(self):
        return 0

    def multiply(self, a, b):
        c = RLaurent(self._R, {0: self._R.basis_element((0,))})
        return Element({a + b: c})

    def rho(self, a):
        return -a

    def rho_inverse(self, a):
        return -a

    def trace(self, a, K=20):
        return RPowerSeries(self._R, {0: 1} if a == 0 else {}, K)

    def r_label_decompose(self, label):
        return 0, (label,)                     # L_n = mu^n * L_0


def test_base_change_pushes_rlaurent_coefficients():
    A = _RLaurentTorusKAlg()
    B = A.base_change(augmentation_hom(A.coefficient_ring()))
    p = B.multiply(1, 2)
    (lab, c), = p.terms.items()
    assert lab == 3
    assert isinstance(c, RLaurent) and c.ring == B.coefficient_ring(), (
        "base-changed multiply must emit coefficients over phi.target")
    # the previously-crashing derived paths
    assert B.verify_rho_twisted_trace(1, -1, 4)
    assert B.verify_orthonormality(0, 0, 4)


def test_forget_pushes_rlaurent_coefficients():
    from kalgebra import LaurentPoly
    A = _RLaurentTorusKAlg()
    F = A.forget()
    p = F.multiply(1, 2)                       # all labels section to 0
    (lab, c), = p.terms.items()
    assert lab == 0
    assert isinstance(c, LaurentPoly), (
        "forgotten multiply must emit plain LaurentPoly coefficients")
    assert F.verify_rho_twisted_trace(0, 0, 4)
    assert F.verify_orthonormality(0, 0, 4)


# ===========================================================================
# Test runner
# ===========================================================================


def main():
    tests = [
        test_identity_hom_on_RElement,
        test_identity_hom_on_RLaurent,
        test_augmentation_collapses_mu_to_1,
        test_augmentation_on_RLaurent,
        test_augmentation_target_is_trivial_ring,
        test_restriction_diagonal_inclusion,
        test_restriction_to_trivial_via_zero_matrix,
        test_identity_base_change_preserves_qtorus,
        test_base_change_rejects_wrong_source_ring,
        test_augmentation_base_change_on_flavoured_bps,
        test_augmentation_then_identity_is_augmentation,
        test_augmentation_multiply_drops_mu,
        test_base_change_pushes_rlaurent_coefficients,
        test_forget_pushes_rlaurent_coefficients,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  [ok]   {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERR ] {t.__name__}: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            failed += 1
    print(f"\n{'=' * 50}")
    print(f"  Results: {passed} passed, {failed} failed of {passed + failed}")
    print(f"{'=' * 50}")
    return failed == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
