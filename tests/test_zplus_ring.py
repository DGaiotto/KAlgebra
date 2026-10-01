"""Tests for `restructuring/zplus_ring.py`.

Run:  PYTHONPATH=. python restructuring/tests/test_zplus_ring.py
"""

from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
_RESTRUCT = os.path.dirname(_HERE)
_REPO = _RESTRUCT  # post-flatten: _RESTRUCT is the repo root
sys.path.insert(0, _REPO)
sys.path.insert(0, _RESTRUCT)

from zplus_ring import (
    ZPlusRing,
    RElement,
    RLaurent,
    TrivialZPlusRing,
    AbelianZPlusRing,
    SU2ZPlusRing,
    SU2xU1ZPlusRing,
    SO3ZPlusRing,
    SU3ZPlusRing,
    SU4ZPlusRing,
    SUNZPlusRing,
    RingHom,
    identity_hom,
    augmentation_hom,
    so3_to_u1_hom,
    so3_to_su2_hom,
    u1_weyl_to_so3,
)


# ===========================================================================
# TrivialZPlusRing tests
# ===========================================================================


def test_trivial_basics():
    R = TrivialZPlusRing()
    one = R.one()
    zero = R.zero()
    assert zero.is_zero()
    assert one.is_one()
    assert (one + one).terms == {(): 2}
    assert (one - one).is_zero()
    assert (one * one) == one
    assert one.star() == one             # ⋆ is identity on Z
    assert (-one).terms == {(): -1}


def test_trivial_arithmetic():
    R = TrivialZPlusRing()
    one = R.one()
    a = one + one + one          # = 3
    b = -one - one               # = -2
    assert (a + b).terms == {(): 1}
    assert (a * b).terms == {(): -6}
    assert (a * 5).terms == {(): 15}
    assert (3 * a).terms == {(): 9}


def test_trivial_eq_int():
    R = TrivialZPlusRing()
    assert R.zero() == 0
    assert R.one() == 1
    assert (R.one() + R.one()) == 2
    assert (-R.one()) == -1


def test_trivial_repr():
    R = TrivialZPlusRing()
    assert repr(R.zero()) == "0"
    assert repr(R.one()) == "1"
    assert repr(-R.one()) == "-1"


# ===========================================================================
# AbelianZPlusRing tests
# ===========================================================================


def test_abelian_basics_rank0():
    """Rank-0 abelian = trivial up to repr."""
    R = AbelianZPlusRing(rank=0)
    assert R.one_basis() == ()
    assert R.one().is_one()
    assert R.zero().is_zero()


def test_abelian_basics_rank1():
    R = AbelianZPlusRing(rank=1)
    assert R.one_basis() == (0,)
    mu = R.basis_element((1,))
    mu_inv = R.basis_element((-1,))
    one = R.one()
    # μ · μ^{-1} = 1
    assert (mu * mu_inv) == one
    # ⋆(μ) = μ^{-1}: rep-ring duality, structurally = ρ-action on central.
    assert mu.star() == mu_inv
    assert mu_inv.star() == mu
    # Group multiplication on basis
    assert (mu * mu).terms == {(2,): 1}


def test_abelian_star_palindrome_rank1():
    """⋆ fixes μ-palindromic elements."""
    R = AbelianZPlusRing(rank=1)
    one = R.one()
    mu = R.basis_element((1,))
    mu_inv = R.basis_element((-1,))
    pal = mu + one + mu_inv
    assert pal.star() == pal
    # Asymmetric elements are *not* fixed
    asym = one + mu
    assert asym.star() == one + mu_inv
    assert asym.star() != asym


def test_abelian_distributivity_rank1():
    R = AbelianZPlusRing(rank=1)
    mu = R.basis_element((1,))
    mu_inv = R.basis_element((-1,))
    one = R.one()
    a = one + mu                  # 1 + μ
    b = one + mu_inv              # 1 + μ^{-1}
    # (1 + μ)(1 + μ^{-1}) = 2 + μ + μ^{-1}
    expected_terms = {(0,): 2, (1,): 1, (-1,): 1}
    assert (a * b).terms == expected_terms
    # ⋆(1 + μ) = 1 + μ^{-1}
    assert a.star() == b


def test_abelian_basics_rank2():
    R = AbelianZPlusRing(rank=2)
    assert R.one_basis() == (0, 0)
    mu1 = R.basis_element((1, 0))
    mu2 = R.basis_element((0, 1))
    # μ_1 · μ_2 = μ_1 μ_2 (commutative; single term)
    assert (mu1 * mu2).terms == {(1, 1): 1}
    # ⋆ flips charges
    assert mu1.star() == R.basis_element((-1, 0))
    elem = mu1 + mu2 - R.one()
    assert elem.star().terms == {(-1, 0): 1, (0, -1): 1, (0, 0): -1}


def test_abelian_zero_rank_lemma():
    """For rank=0, AbelianZPlusRing should behave like TrivialZPlusRing."""
    R0 = AbelianZPlusRing(rank=0)
    Rt = TrivialZPlusRing()
    # Both have a single basis element (() in both cases).  Arithmetic should
    # give analogous results term-for-term.
    assert R0.one().terms == Rt.one().terms
    assert (R0.one() + R0.one()).terms == (Rt.one() + Rt.one()).terms


def test_abelian_repr():
    R = AbelianZPlusRing(rank=1)
    mu = R.basis_element((1,))
    one = R.one()
    s = repr(mu + one)
    # Just sanity: contains both the [(1,)] form and a 1 token
    assert "1" in s
    assert "(1,)" in s


# ===========================================================================
# RElement: cross-cutting / hashability
# ===========================================================================


def test_relement_unhashable():
    R = AbelianZPlusRing(rank=1)
    try:
        hash(R.one())
    except TypeError:
        pass
    else:
        raise AssertionError("RElement should be unhashable")


def test_relement_ring_mismatch_raises():
    R1 = AbelianZPlusRing(rank=1)
    R2 = AbelianZPlusRing(rank=2)
    try:
        R1.one() + R2.one()
    except ValueError:
        pass
    else:
        raise AssertionError("ring mismatch should raise ValueError")


def test_relement_int_eq():
    """Equality with raw integers, treating them as multiples of identity."""
    R = AbelianZPlusRing(rank=1)
    assert R.zero() == 0
    assert R.one() == 1
    assert (R.one() * 3) == 3
    assert (-R.one() * 5) == -5
    # Non-trivial element shouldn't equal an int
    assert R.basis_element((1,)) != 1


# ===========================================================================
# RLaurent — over TrivialZPlusRing
# ===========================================================================


def test_rlaurent_trivial_basics():
    R = TrivialZPlusRing()
    one = RLaurent.one(R)
    q = RLaurent.q(R, 1)
    qi = RLaurent.q(R, -1)
    assert (q * qi) == one
    assert q.bar() == qi                      # bar = q-flip
    # palindrome (in q)
    pal = q + qi
    assert pal.bar() == pal
    # 0
    assert RLaurent.zero(R).is_zero()


def test_rlaurent_trivial_distributivity():
    R = TrivialZPlusRing()
    q = RLaurent.q(R, 1)
    qi = RLaurent.q(R, -1)
    one = RLaurent.one(R)
    p = (one + q) * (one + qi)
    # = 1 + q + q^{-1} + 1 = 2 + q + q^{-1}
    expected = RLaurent(R, {-1: 1, 0: 2, 1: 1})
    assert p == expected


def test_rlaurent_trivial_signed():
    R = TrivialZPlusRing()
    q = RLaurent.q(R, 1)
    one = RLaurent.one(R)
    poly = (one - q) * (one - q)        # 1 - 2q + q^2
    expected = RLaurent(R, {0: 1, 1: -2, 2: 1})
    assert poly == expected
    # bar: 1 - 2q^{-1} + q^{-2}
    expected_bar = RLaurent(R, {0: 1, -1: -2, -2: 1})
    assert poly.bar() == expected_bar


# ===========================================================================
# RLaurent — over AbelianZPlusRing
# ===========================================================================


def test_rlaurent_abelian_bar_acts_only_on_q():
    """Bar = q-flip; μ stays."""
    R = AbelianZPlusRing(rank=1)
    mu = R.basis_element((1,))

    qmu = RLaurent(R, {1: mu})       # q · μ
    qmu_bar = qmu.bar()              # q^{-1} · μ  (μ unchanged)
    expected = RLaurent(R, {-1: mu})
    assert qmu_bar == expected


def test_rlaurent_abelian_palindrome_in_q_only():
    """Palindromic in q with arbitrary μ-coefficients is bar-invariant."""
    R = AbelianZPlusRing(rank=1)
    mu = R.basis_element((1,))
    # μ·(q + q^{-1}) is bar-invariant: bar fixes the q±1 pair, μ unchanged.
    pal = RLaurent(R, {1: mu, -1: mu})
    assert pal.bar() == pal


def test_rlaurent_abelian_non_palindrome():
    """A μ-asymmetric q-palindrome is *not* a palindrome under the
    old (q+μ)-flipping convention; under the KAlgebra bar (q only) it
    IS preserved when q-symmetric."""
    R = AbelianZPlusRing(rank=1)
    mu = R.basis_element((1,))
    mu_i = R.basis_element((-1,))
    # q μ + q^{-1} μ^{-1}: was palindromic under the old (q + μ)-flip;
    # under q-only bar it becomes q^{-1} μ + q μ^{-1} ≠ original.
    elem = RLaurent(R, {1: mu, -1: mu_i})
    barred = elem.bar()
    expected = RLaurent(R, {-1: mu, 1: mu_i})
    assert barred == expected
    assert barred != elem


def test_rlaurent_abelian_distributivity():
    R = AbelianZPlusRing(rank=2)
    mu1 = R.basis_element((1, 0))
    mu2 = R.basis_element((0, 1))
    one_R = R.one()

    a = RLaurent(R, {0: mu1, 1: mu2})        # μ_1 + q · μ_2
    b = RLaurent(R, {0: one_R, -1: mu1})     # 1 + q^{-1} · μ_1
    prod = a * b
    # (μ_1 + q μ_2)(1 + q^{-1} μ_1)
    #   = μ_1 + q^{-1} μ_1^2 + q μ_2 + μ_1 μ_2
    expected = RLaurent(R, {
        0: mu1 + R.basis_element((1, 1)),
        -1: R.basis_element((2, 0)),
        1: mu2,
    })
    assert prod == expected


def test_rlaurent_abelian_bar_intertwines_mul():
    """`bar` is a ring (anti)homomorphism on the commutative R[q^±].
    bar(a · b) = bar(a) · bar(b) — and with bar acting only on q, this
    still holds since q-flip is a ring hom."""
    R = AbelianZPlusRing(rank=1)
    mu = R.basis_element((1,))
    a = RLaurent(R, {0: R.one(), 1: mu})              # 1 + q μ
    b = RLaurent(R, {0: mu - R.one(), -1: R.one()})   # (μ - 1) + q^{-1}
    assert (a * b).bar() == a.bar() * b.bar()


def test_rlaurent_abelian_bar_involutive():
    """bar² = id."""
    R = AbelianZPlusRing(rank=1)
    mu = R.basis_element((1,))
    elem = RLaurent(R, {2: mu, -1: R.one() + mu, 0: -R.one()})
    assert elem.bar().bar() == elem


def test_rlaurent_star_acts_only_on_R():
    """`RLaurent.star()` applies ⋆ to R-coefficients, leaves q alone."""
    R = AbelianZPlusRing(rank=1)
    mu = R.basis_element((1,))
    mu_i = R.basis_element((-1,))

    qmu = RLaurent(R, {1: mu})       # q · μ
    starred = qmu.star()              # q · μ^{-1}  (q untouched, μ flipped)
    expected = RLaurent(R, {1: mu_i})
    assert starred == expected


def test_rlaurent_star_trivial_on_trivial_ring():
    """⋆ is identity on TrivialZPlusRing, so RLaurent.star() is identity."""
    R = TrivialZPlusRing()
    elem = RLaurent.q(R, 2) + RLaurent.q(R, -1) + (-RLaurent.one(R))
    assert elem.star() == elem


def test_rlaurent_bar_and_star_commute():
    """bar (q-flip) and star (R-flip) act on independent factors,
    so they commute on R[q^±]."""
    R = AbelianZPlusRing(rank=2)
    mu1 = R.basis_element((1, 0))
    mu2 = R.basis_element((0, 1))
    elem = RLaurent(R, {0: mu1 + mu2, 1: mu1, -1: -R.one()})
    assert elem.bar().star() == elem.star().bar()


def test_rlaurent_star_involutive():
    """star² = id."""
    R = AbelianZPlusRing(rank=1)
    mu = R.basis_element((1,))
    elem = RLaurent(R, {2: mu, -1: R.one() + mu, 0: -R.one()})
    assert elem.star().star() == elem


def test_rlaurent_star_distributes_over_mult():
    """star is a ring homomorphism on R[q^±]: star(a·b) = star(a)·star(b)."""
    R = AbelianZPlusRing(rank=1)
    mu = R.basis_element((1,))
    a = RLaurent(R, {0: R.one(), 1: mu})              # 1 + q μ
    b = RLaurent(R, {0: mu - R.one(), -1: R.one()})   # (μ - 1) + q^{-1}
    assert (a * b).star() == a.star() * b.star()


def test_rlaurent_zero_and_one_compare_int():
    R = AbelianZPlusRing(rank=1)
    assert RLaurent.zero(R) == 0
    assert RLaurent.one(R) == 1
    # 1 + something is not == 1
    mu = R.basis_element((1,))
    assert RLaurent(R, {0: R.one() + mu}) != 1


def test_rlaurent_q_times_q_inverse_is_one():
    R = AbelianZPlusRing(rank=2)
    q = RLaurent.q(R, 1)
    qi = RLaurent.q(R, -1)
    assert (q * qi) == RLaurent.one(R)


def test_rlaurent_int_and_relement_scalar_mul():
    R = AbelianZPlusRing(rank=1)
    mu = R.basis_element((1,))
    a = RLaurent(R, {0: R.one(), 1: mu})              # 1 + q μ

    # Scalar multiplication by int
    assert (a * 3).coeffs == {0: R.one() * 3, 1: mu * 3}
    assert (3 * a).coeffs == (a * 3).coeffs

    # Scalar multiplication by RElement
    twist = R.basis_element((2,))
    twisted = a * twist
    # (1 + q μ) · μ^2 = μ^2 + q μ^3
    assert twisted.coeffs == {
        0: R.basis_element((2,)),
        1: R.basis_element((3,)),
    }


# ===========================================================================
# Z₊-ring axioms
# ===========================================================================


def _axiom_one_in_basis(ring: ZPlusRing) -> bool:
    """1 ∈ B and is its own ⋆."""
    ob = ring.one_basis()
    return ring.star_basis(ob) == ob


def _axiom_star_involutive(ring: ZPlusRing, basis_samples) -> bool:
    return all(
        ring.star_basis(ring.star_basis(b)) == b for b in basis_samples
    )


def _axiom_one_is_unit(ring: ZPlusRing, basis_samples) -> bool:
    """multiply_basis(1, b) == {b: 1} == multiply_basis(b, 1)."""
    ob = ring.one_basis()
    for b in basis_samples:
        if ring.multiply_basis(ob, b) != {b: 1}:
            return False
        if ring.multiply_basis(b, ob) != {b: 1}:
            return False
    return True


def _axiom_star_antihomomorphism(ring: ZPlusRing, basis_samples) -> bool:
    """(b1 · b2)⋆ = b2⋆ · b1⋆ — for commutative rings, also = b1⋆ · b2⋆."""
    for b1 in basis_samples:
        for b2 in basis_samples:
            lhs_terms = ring.multiply_basis(b1, b2)
            lhs = {ring.star_basis(c): n for c, n in lhs_terms.items()}
            rhs = ring.multiply_basis(ring.star_basis(b2), ring.star_basis(b1))
            if lhs != rhs:
                return False
    return True


def _axiom_nonneg_structure_constants(ring: ZPlusRing, basis_samples) -> bool:
    for b1 in basis_samples:
        for b2 in basis_samples:
            for n in ring.multiply_basis(b1, b2).values():
                if n < 0:
                    return False
    return True


def test_axioms_trivial_ring():
    R = TrivialZPlusRing()
    samples = [()]
    assert _axiom_one_in_basis(R)
    assert _axiom_star_involutive(R, samples)
    assert _axiom_one_is_unit(R, samples)
    assert _axiom_star_antihomomorphism(R, samples)
    assert _axiom_nonneg_structure_constants(R, samples)


def test_axioms_abelian_rank2():
    R = AbelianZPlusRing(rank=2)
    samples = [(0, 0), (1, 0), (0, 1), (1, 1), (-1, 0), (-2, 3)]
    assert _axiom_one_in_basis(R)
    assert _axiom_star_involutive(R, samples)
    assert _axiom_one_is_unit(R, samples)
    assert _axiom_star_antihomomorphism(R, samples)
    assert _axiom_nonneg_structure_constants(R, samples)


# ===========================================================================
# SU2ZPlusRing tests
# ===========================================================================


def test_su2_basics():
    R = SU2ZPlusRing()
    assert R.one_basis() == 0
    one = R.one()
    chi1 = R.basis_element(1)
    chi2 = R.basis_element(2)
    chi3 = R.basis_element(3)
    # χ_0 · χ_n = χ_n.
    assert (one * chi1) == chi1
    assert (chi1 * one) == chi1
    # χ_1 · χ_1 = χ_0 + χ_2  (spin-1/2 × spin-1/2 = trivial ⊕ vector).
    assert (chi1 * chi1).terms == {0: 1, 2: 1}
    # χ_1 · χ_2 = χ_1 + χ_3.
    assert (chi1 * chi2).terms == {1: 1, 3: 1}
    # χ_2 · χ_2 = χ_0 + χ_2 + χ_4.
    assert (chi2 * chi2).terms == {0: 1, 2: 1, 4: 1}
    # χ_2 · χ_3 = χ_1 + χ_3 + χ_5.
    assert (chi2 * chi3).terms == {1: 1, 3: 1, 5: 1}
    # χ_3 · χ_3 = χ_0 + χ_2 + χ_4 + χ_6.
    assert (chi3 * chi3).terms == {0: 1, 2: 1, 4: 1, 6: 1}


def test_su2_commutative():
    R = SU2ZPlusRing()
    for n in range(6):
        for m in range(6):
            a = R.basis_element(n)
            b = R.basis_element(m)
            assert (a * b).terms == (b * a).terms


def test_su2_star_identity():
    R = SU2ZPlusRing()
    for n in range(8):
        assert R.star_basis(n) == n
    e = R.basis_element(2) * 3 - R.basis_element(5)
    assert e.star().terms == e.terms


def test_su2_to_abelian_chi1():
    R = SU2ZPlusRing()
    A = AbelianZPlusRing(rank=1)
    chi1_abelian = R.to_abelian(R.basis_element(1))
    # χ_1 ↦ μ + μ^{-1}.
    assert chi1_abelian.terms == {(1,): 1, (-1,): 1}
    # χ_2 ↦ μ^2 + 1 + μ^{-2}.
    chi2_abelian = R.to_abelian(R.basis_element(2))
    assert chi2_abelian.terms == {(2,): 1, (0,): 1, (-2,): 1}
    # χ_0 ↦ 1.
    assert R.to_abelian(R.one()).terms == {(0,): 1}


def test_su2_to_abelian_intertwines_multiplication():
    """χ_n · χ_m via Clebsch-Gordan must agree with U(1) multiplication
    of the Weyl-symmetrised images."""
    R = SU2ZPlusRing()
    for n in range(5):
        for m in range(5):
            lhs = R.to_abelian(R.basis_element(n) * R.basis_element(m))
            rhs = R.to_abelian(R.basis_element(n)) * R.to_abelian(R.basis_element(m))
            assert lhs.terms == rhs.terms, (
                f"SU2 χ_{n} · χ_{m}: Clebsch-Gordan mismatches U(1) "
                f"multiplication of Weyl images.\n  CG -> abelian = {lhs.terms}\n"
                f"  U(1) product = {rhs.terms}"
            )


def test_su2_negative_basis_raises():
    R = SU2ZPlusRing()
    try:
        R.multiply_basis(-1, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for negative basis")
    try:
        R.star_basis(-2)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for negative basis")


def test_su2_repr_eq_hash():
    R1 = SU2ZPlusRing()
    R2 = SU2ZPlusRing()
    assert R1 == R2
    assert hash(R1) == hash(R2)
    assert repr(R1) == "SU2ZPlusRing()"
    assert R1 != SO3ZPlusRing()
    assert R1 != AbelianZPlusRing(rank=1)
    assert R1 != TrivialZPlusRing()


def test_axioms_su2():
    R = SU2ZPlusRing()
    samples = [0, 1, 2, 3, 4]
    assert _axiom_one_in_basis(R)
    assert _axiom_star_involutive(R, samples)
    assert _axiom_one_is_unit(R, samples)
    assert _axiom_star_antihomomorphism(R, samples)
    assert _axiom_nonneg_structure_constants(R, samples)


def test_rlaurent_su2_bar_acts_only_on_q():
    R = SU2ZPlusRing()
    chi1 = R.basis_element(1)
    chi2 = R.basis_element(2)
    elt = RLaurent(R, {0: chi1, 2: chi2, -1: R.one()})
    bar = elt.bar()
    # Bar conjugates q exponents, fixes R-coefficients.
    assert bar.coeffs == {0: chi1, -2: chi2, 1: R.one()}


def test_rlaurent_su2_palindrome_in_q():
    R = SU2ZPlusRing()
    chi1 = R.basis_element(1)
    chi2 = R.basis_element(2)
    # 1 + (χ_1) (q + q^{-1}) + (χ_2) (q^2 + q^{-2})  is bar-palindromic.
    p = RLaurent(R, {
        0: R.one(), 1: chi1, -1: chi1, 2: chi2, -2: chi2,
    })
    assert p.bar().coeffs == p.coeffs


# ===========================================================================
# SO3ZPlusRing tests
# ===========================================================================


def test_so3_basics():
    R = SO3ZPlusRing()
    assert R.one_basis() == 0
    one = R.one()
    v = R.basis_element(1)  # vector rep (dim 3)
    j2 = R.basis_element(2)  # spin-2 (dim 5)
    j3 = R.basis_element(3)
    # χ_0 · anything = anything.
    assert (one * v) == v
    # vector × vector = trivial + vector + spin-2  (= 1 + 3 + 5 = 9 ✓).
    assert (v * v).terms == {0: 1, 1: 1, 2: 1}
    # spin-1 × spin-2 = spin-1 + spin-2 + spin-3.
    assert (v * j2).terms == {1: 1, 2: 1, 3: 1}
    # spin-2 × spin-2 = spin-0 + spin-1 + spin-2 + spin-3 + spin-4  (= 25).
    assert (j2 * j2).terms == {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}
    # spin-2 × spin-3 = spin-1 + spin-2 + spin-3 + spin-4 + spin-5.
    assert (j2 * j3).terms == {1: 1, 2: 1, 3: 1, 4: 1, 5: 1}


def test_so3_dimensions_match():
    """The product of two SO(3) reps has total dimension dim_a · dim_b."""
    R = SO3ZPlusRing()
    for a in range(5):
        for b in range(5):
            prod = R.basis_element(a) * R.basis_element(b)
            total_dim = sum((2 * j + 1) * c for j, c in prod.terms.items())
            assert total_dim == (2 * a + 1) * (2 * b + 1), (
                f"χ_{a} · χ_{b}: total dim {total_dim} ≠ {(2*a+1)*(2*b+1)}"
            )


def test_so3_commutative():
    R = SO3ZPlusRing()
    for a in range(5):
        for b in range(5):
            assert (R.basis_element(a) * R.basis_element(b)).terms == (
                R.basis_element(b) * R.basis_element(a)
            ).terms


def test_so3_star_identity():
    R = SO3ZPlusRing()
    for j in range(8):
        assert R.star_basis(j) == j


def test_so3_to_abelian_vector():
    R = SO3ZPlusRing()
    v_abelian = R.to_abelian(R.basis_element(1))
    # vector χ_1 ↦ μ + 1 + μ^{-1}  (weights -1, 0, +1).
    assert v_abelian.terms == {(1,): 1, (0,): 1, (-1,): 1}
    # spin-2 χ_2 ↦ μ^2 + μ + 1 + μ^{-1} + μ^{-2}.
    j2_abelian = R.to_abelian(R.basis_element(2))
    assert j2_abelian.terms == {(2,): 1, (1,): 1, (0,): 1, (-1,): 1, (-2,): 1}


def test_so3_to_abelian_intertwines_multiplication():
    R = SO3ZPlusRing()
    for a in range(4):
        for b in range(4):
            lhs = R.to_abelian(R.basis_element(a) * R.basis_element(b))
            rhs = R.to_abelian(R.basis_element(a)) * R.to_abelian(R.basis_element(b))
            assert lhs.terms == rhs.terms, (
                f"SO3 χ_{a} · χ_{b}: Clebsch-Gordan mismatches U(1) product "
                f"of integer-weight Weyl images."
            )


def test_so3_to_su2_intertwines_multiplication():
    """χ_j^{SO(3)} ↦ χ_{2j}^{SU(2)} as a Z₊-ring map."""
    R = SO3ZPlusRing()
    S = SU2ZPlusRing()
    for a in range(4):
        for b in range(4):
            lhs = R.to_su2(R.basis_element(a) * R.basis_element(b))
            rhs = R.to_su2(R.basis_element(a)) * R.to_su2(R.basis_element(b))
            assert lhs.terms == rhs.terms, (
                f"SO3 ↪ SU2 not a ring map at χ_{a} · χ_{b}: "
                f"lhs = {lhs.terms}, rhs = {rhs.terms}"
            )


def test_so3_to_su2_image_is_even():
    """The image of SO(3) ↪ SU(2) consists of even-n SU(2) basis elements."""
    R = SO3ZPlusRing()
    for j in range(6):
        img = R.to_su2(R.basis_element(j))
        for n in img.terms.keys():
            assert n % 2 == 0, f"χ_{j}^SO(3) ↦ χ_{n}^SU(2) but n is odd"


def test_so3_negative_basis_raises():
    R = SO3ZPlusRing()
    try:
        R.multiply_basis(-1, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for negative basis")


def test_so3_repr_eq_hash():
    R1 = SO3ZPlusRing()
    R2 = SO3ZPlusRing()
    assert R1 == R2
    assert hash(R1) == hash(R2)
    assert repr(R1) == "SO3ZPlusRing()"
    assert R1 != SU2ZPlusRing()


def test_axioms_so3():
    R = SO3ZPlusRing()
    samples = [0, 1, 2, 3, 4]
    assert _axiom_one_in_basis(R)
    assert _axiom_star_involutive(R, samples)
    assert _axiom_one_is_unit(R, samples)
    assert _axiom_star_antihomomorphism(R, samples)
    assert _axiom_nonneg_structure_constants(R, samples)


def test_rlaurent_so3_distributivity():
    """(1 + q χ_1)(1 + q^{-1} χ_1) = 1 + q χ_1 + q^{-1} χ_1 + χ_1^2
    = 1 + q χ_1 + q^{-1} χ_1 + χ_0 + χ_1 + χ_2."""
    R = SO3ZPlusRing()
    chi1 = R.basis_element(1)
    a = RLaurent(R, {0: R.one(), 1: chi1})
    b = RLaurent(R, {0: R.one(), -1: chi1})
    prod = a * b
    expected = {
        0: R.one() + R.one() + chi1 + R.basis_element(2),  # 1 + χ_1·χ_1 = 1 + (1+χ_1+χ_2)
        1: chi1,
        -1: chi1,
    }
    # Hmm — 1·1 contributes to q^0 with coeff 1; χ_1·χ_1 contributes to q^0
    # with coeff (1 + χ_1 + χ_2).  So total q^0 coeff = 1 + 1 + χ_1 + χ_2.
    assert prod.coeffs[0].terms == expected[0].terms
    assert prod.coeffs[1].terms == chi1.terms
    assert prod.coeffs[-1].terms == chi1.terms


# ===========================================================================
# so3_to_u1_hom / so3_to_su2_hom (symmetry-reduction ring homs)
# ===========================================================================


def test_so3_to_u1_hom_identity_image():
    """χ_0 ↦ μ^0 = 1 in the abelian target."""
    phi = so3_to_u1_hom()
    src = SO3ZPlusRing()
    tgt = AbelianZPlusRing(rank=1)
    img = phi.apply_basis(0)
    assert img.ring == tgt
    assert img.terms == {(0,): 1}


def test_so3_to_u1_hom_vector_image():
    """χ_1 ↦ μ + 1 + μ^{-1}."""
    phi = so3_to_u1_hom()
    img = phi.apply_basis(1)
    assert img.terms == {(-1,): 1, (0,): 1, (1,): 1}


def test_so3_to_u1_hom_higher():
    phi = so3_to_u1_hom()
    img = phi.apply_basis(3)
    expected = {(k,): 1 for k in range(-3, 4)}
    assert img.terms == expected


def test_so3_to_u1_hom_is_multiplicative():
    """For all j, k in a window: phi(χ_j · χ_k) = phi(χ_j) · phi(χ_k)."""
    phi = so3_to_u1_hom()
    src = SO3ZPlusRing()
    for j in range(5):
        for k in range(5):
            chi_j_chi_k = src.basis_element(j) * src.basis_element(k)
            lhs = phi.apply_RElement(chi_j_chi_k)
            rhs = phi.apply_basis(j) * phi.apply_basis(k)
            assert lhs.terms == rhs.terms, (
                f"phi(χ_{j} · χ_{k}) ≠ phi(χ_{j}) · phi(χ_{k}): "
                f"lhs={lhs.terms}, rhs={rhs.terms}"
            )


def test_so3_to_u1_hom_preserves_identity():
    phi = so3_to_u1_hom()
    src = SO3ZPlusRing()
    tgt_one = AbelianZPlusRing(rank=1).one()
    assert phi.apply_RElement(src.one()).terms == tgt_one.terms


def test_so3_to_u1_hom_intertwines_star():
    """SO(3) reps are self-dual (⋆ = id); U(1) ⋆ flips sign; the hom
    should intertwine: phi(b⋆) = phi(b)⋆ on all source basis elements."""
    phi = so3_to_u1_hom()
    src = SO3ZPlusRing()
    for j in range(5):
        b_star = src.basis_element(src.star_basis(j))
        img_star = phi.apply_RElement(b_star)
        img_then_star = phi.apply_basis(j).star()
        assert img_star.terms == img_then_star.terms


def test_so3_to_u1_hom_image_is_weyl_symmetric():
    """The image lands in the Weyl-symmetric subring of Z[μ^±] (= μ ↔ μ^{-1})."""
    phi = so3_to_u1_hom()
    for j in range(6):
        img = phi.apply_basis(j)
        # Check symmetry: μ^k coefficient = μ^{-k} coefficient.
        for (k,), c in img.terms.items():
            assert img.terms.get((-k,), 0) == c, (
                f"χ_{j} image not Weyl-symmetric at μ^{k}"
            )


def test_so3_to_u1_hom_argument_validation():
    try:
        so3_to_u1_hom(source=AbelianZPlusRing(rank=1))  # wrong source
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError for wrong source")
    try:
        so3_to_u1_hom(target=AbelianZPlusRing(rank=2))  # wrong target rank
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError for wrong target")


def test_so3_to_su2_hom_image_is_even_n():
    """The image lands in the even-n subring of SU2."""
    phi = so3_to_su2_hom()
    for j in range(5):
        img = phi.apply_basis(j)
        assert img.terms == {2 * j: 1}


def test_so3_to_su2_hom_is_multiplicative():
    phi = so3_to_su2_hom()
    src = SO3ZPlusRing()
    for j in range(4):
        for k in range(4):
            chi_jk = src.basis_element(j) * src.basis_element(k)
            lhs = phi.apply_RElement(chi_jk)
            rhs = phi.apply_basis(j) * phi.apply_basis(k)
            assert lhs.terms == rhs.terms


def test_u1_weyl_to_so3_round_trip():
    """so3_to_u1_hom and u1_weyl_to_so3 are inverses on R(SO(3))."""
    R = SO3ZPlusRing()
    phi = so3_to_u1_hom()
    for j in range(5):
        chi_j = R.basis_element(j)
        u1_img = phi.apply_RElement(chi_j)
        back = u1_weyl_to_so3(u1_img)
        assert back.terms == chi_j.terms, (
            f"round-trip on χ_{j}: forward {u1_img.terms}, "
            f"back {back.terms}, expected {chi_j.terms}"
        )


def test_u1_weyl_to_so3_virtual_character():
    """A Weyl-symmetric μ-Laurent that's NOT a sum of χ_j ↦ μ^j + ... + μ^{-j}
    images can still be expressed as a *virtual* SO(3) character (with
    signed coefficients).

    Concretely μ + μ^{-1} = χ_1 - χ_0 (since χ_1 = μ + 1 + μ^{-1}).
    """
    u1 = AbelianZPlusRing(rank=1)
    elt = RElement(u1, {(1,): 1, (-1,): 1})
    result = u1_weyl_to_so3(elt)
    assert result.terms == {1: 1, 0: -1}


def test_u1_weyl_to_so3_rejects_non_symmetric():
    """μ alone (= μ¹, no μ⁻¹) isn't Weyl-symmetric: must raise."""
    u1 = AbelianZPlusRing(rank=1)
    elt = RElement(u1, {(1,): 1})
    try:
        u1_weyl_to_so3(elt)
    except ValueError as e:
        assert "Weyl-symmetric" in str(e) or "symmetric" in str(e)
    else:
        raise AssertionError("expected ValueError for non-symmetric input")


def test_u1_weyl_to_so3_rejects_wrong_source_ring():
    """Source must be AbelianZPlusRing(rank=1)."""
    R = SO3ZPlusRing()
    chi_1 = R.basis_element(1)
    try:
        u1_weyl_to_so3(chi_1)
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError for SO3 source")


def test_so3_chain_diagram_commutes_up_to_mu_squared():
    """The two compositions
        (a) direct      R(SO(3)) --so3_to_u1-->                R(U(1)),
        (b) via SU(2)   R(SO(3)) --so3_to_su2--> R(SU(2)) --su2_to_u1--> R(U(1)),
    are related by the rescaling μ_{SO(3)} = μ_{SU(2)}²: paths agree
    when we substitute μ ↦ μ^2 in the direct path.  This encodes the
    convention difference documented in the SU2ZPlusRing and SO3ZPlusRing
    docstrings (μ carries integer weights in SO(3); doubled / spin-1/2
    weights in SU(2)).
    """
    phi_so3_u1 = so3_to_u1_hom()
    phi_so3_su2 = so3_to_su2_hom()
    u1 = AbelianZPlusRing(rank=1)

    def su2_to_u1_apply(n: int) -> RElement:
        # SU(2) χ_n ↦ μ^n + μ^{n-2} + ⋯ + μ^{-n} (step 2, "spin-1/2 weight" μ).
        return RElement(u1, {(k,): 1 for k in range(-n, n + 1, 2)})

    for j in range(5):
        # Direct path, rescaled μ ↦ μ^2: every μ^k becomes μ^{2k}.
        direct = phi_so3_u1.apply_basis(j)
        direct_rescaled = RElement(
            u1, {(2 * k,): c for (k,), c in direct.terms.items()},
        )
        # Via SU(2) path.
        via_su2 = phi_so3_su2.apply_basis(j)
        via_path = u1.zero()
        for n, c in via_su2.terms.items():
            via_path = via_path + su2_to_u1_apply(n) * c
        assert direct_rescaled.terms == via_path.terms, (
            f"χ_{j}: direct (μ↦μ²) = {direct_rescaled.terms}, "
            f"via SU(2) = {via_path.terms}"
        )


# ===========================================================================
# Hash / eq contract (2026-06-13 flavour-ring audit regression)
# ===========================================================================


def test_hash_eq_contract():
    """Every shipped Z₊-ring obeys the hash/eq invariant — equal rings
    hash equal and work as dict/set keys — *including* tensor-product rings
    built from equal-but-distinct factor objects.  Regression for the
    `TensorZPlusRing.__hash__` `id()` bug (equal rings hashed differently →
    dict miss) and the stray-token `SU3ZPlusRing.__hash__` merge artifact.
    """
    from tensor_zplus_ring import TensorZPlusRing

    equal_pairs = [
        (TrivialZPlusRing(), TrivialZPlusRing()),
        (AbelianZPlusRing(2), AbelianZPlusRing(2)),
        (SU2ZPlusRing(), SU2ZPlusRing()),
        (SU2xU1ZPlusRing(), SU2xU1ZPlusRing()),
        (SO3ZPlusRing(), SO3ZPlusRing()),
        (SU3ZPlusRing(), SU3ZPlusRing()),
        (SU4ZPlusRing(), SU4ZPlusRing()),
        (SUNZPlusRing(5), SUNZPlusRing(5)),
        # built from equal-but-distinct factor objects — the id()-hash trap:
        (TensorZPlusRing(SU2ZPlusRing(), AbelianZPlusRing(1)),
         TensorZPlusRing(SU2ZPlusRing(), AbelianZPlusRing(1))),
        (TensorZPlusRing(SU2ZPlusRing(), SU2ZPlusRing()),
         TensorZPlusRing(SU2ZPlusRing(), SU2ZPlusRing())),
    ]
    for a, b in equal_pairs:
        assert a == b, f"{a!r} should equal {b!r}"
        assert hash(a) == hash(b), f"equal rings hash differently: {a!r}"
        assert {a: 1}.get(b) == 1, f"dict lookup missed for equal key {b!r}"

    # Distinct rings stay unequal (SU3 must not collide with SUN by eq).
    assert SU3ZPlusRing() != SUNZPlusRing(3)
    assert SUNZPlusRing(3) != SUNZPlusRing(4)
    assert (TensorZPlusRing(SU2ZPlusRing(), AbelianZPlusRing(1))
            != TensorZPlusRing(SU2ZPlusRing(), AbelianZPlusRing(2)))


def test_augmentation_dim_is_hom_all_rings():
    """the design record: every ZPlusRing carries a `dim` augmentation ε: R → Z that is a
    unital, ⋆-invariant, multiplicative ring hom; `augmentation()` realises it."""
    from zplus_ring import TensorZPlusRing
    cases = [
        (TrivialZPlusRing(), [()]),
        (AbelianZPlusRing(2), [(0, 0), (1, 0), (0, 1), (2, -1), (-3, 2)]),
        (SU2ZPlusRing(), list(range(0, 6))),
        (TensorZPlusRing([SU2ZPlusRing(), SU2ZPlusRing()]),
         [(0, 0), (1, 0), (1, 1), (2, 1)]),
        (SU2xU1ZPlusRing(), [(0, 0), (1, 2), (2, -1), (3, 0)]),
        (SO3ZPlusRing(), list(range(0, 6))),
        (SU3ZPlusRing(), [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0)]),
        (SU4ZPlusRing(), [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1)]),
        (SUNZPlusRing(4), [(), (1,), (2,), (1, 1), (2, 1)]),
    ]
    for R, samples in cases:
        assert R.verify_augmentation_is_hom(samples), R
        eps = R.augmentation()
        assert isinstance(eps, RingHom)
        assert isinstance(eps.target, TrivialZPlusRing)
        for b in samples:
            assert R.dim(b) >= 1
            assert eps.apply_basis(b).terms.get((), 0) == R.dim(b)


def test_augmentation_known_dimensions():
    """Spot-check `dim` against textbook representation dimensions."""
    assert SU2ZPlusRing().dim(1) == 2            # doublet
    assert SU2ZPlusRing().dim(2) == 3            # triplet
    assert SO3ZPlusRing().dim(1) == 3            # vector
    assert SU3ZPlusRing().dim((1, 0)) == 3       # fundamental
    assert SU3ZPlusRing().dim((1, 1)) == 8       # adjoint
    assert SU4ZPlusRing().dim((1, 0, 0)) == 4    # fundamental
    assert SU4ZPlusRing().dim((0, 1, 0)) == 6    # ∧² fund = Spin(6) vector
    assert SU4ZPlusRing().dim((1, 0, 1)) == 15   # adjoint
    assert SUNZPlusRing(4).dim((1, 1)) == 6      # ∧² of the fundamental


def test_augmentation_generalises_abelian_hom():
    """The generalised augmentation_hom reproduces the classic abelian μ → 1."""
    A = AbelianZPlusRing(3)
    eps = augmentation_hom(A)
    assert isinstance(eps, RingHom)
    for b in [(0, 0, 0), (1, -2, 3), (5, 0, -1)]:
        assert eps.apply_basis(b).terms.get((), 0) == 1


def test_one_dim_reps_structure_all_rings():
    """the design record: every ZPlusRing exposes its 1-dim-rep group Λ ≅ Z^c (the lift
    torsor) via one_dim_rep_rank / embed_one_dim_rep; verify_one_dim_reps checks
    ι is a group-like ring-hom inclusion.  c = 0 for semisimple G_f."""
    from zplus_ring import TensorZPlusRing
    from sun_characters import SUNFlavourRing
    from so2nf_characters import SO2NfZPlusRing
    import itertools

    def Zc(c):
        return list(itertools.product(range(-2, 3), repeat=c))

    cases = [
        (TrivialZPlusRing(),                                      0, [()]),
        (AbelianZPlusRing(2),                                     2, Zc(2)),
        (SU2ZPlusRing(),                                          0, [()]),
        (SO3ZPlusRing(),                                          0, [()]),
        (SU3ZPlusRing(),                                          0, [()]),
        (SU4ZPlusRing(),                                          0, [()]),
        (SUNZPlusRing(5),                                         0, [()]),
        (SU2xU1ZPlusRing(),                                       1, [(m,) for m in range(-2, 3)]),
        (TensorZPlusRing([SU2ZPlusRing(), SU2ZPlusRing()]),       0, [()]),
        (TensorZPlusRing([AbelianZPlusRing(1), SU2ZPlusRing()]),  1, [(m,) for m in range(-2, 3)]),
        (SUNFlavourRing([3], 2),                                  2, Zc(2)),
        (SO2NfZPlusRing(3),                                       0, [()]),
    ]
    for R, exp_c, samples in cases:
        assert R.one_dim_rep_rank() == exp_c, R
        assert R.one_dim_reps() == AbelianZPlusRing(exp_c), R
        assert R.verify_one_dim_reps(samples), R
        for f in samples:                       # ι lands on group-like (dim-1) elements
            assert R.dim(R.embed_one_dim_rep(f)) == 1, (R, f)


def test_one_dim_rep_inclusion_values():
    """ι: AbelianZPlusRing(c) → R is the group-like inclusion with the expected
    images, and ρ inverts the torsor (ι(−f) = ι(f)⋆)."""
    A = AbelianZPlusRing(2)
    incA = A.one_dim_rep_inclusion()
    assert incA.source == AbelianZPlusRing(2) and incA.target == A
    assert incA.apply_basis((3, -1)).terms == {(3, -1): 1}      # ι = identity
    S = SU2xU1ZPlusRing()
    incS = S.one_dim_rep_inclusion()
    assert incS.apply_basis((2,)).terms == {(0, 2): 1}          # trivial SU(2) ⊗ μ^2
    assert S.embed_one_dim_rep((-2,)) == S.star_basis(S.embed_one_dim_rep((2,)))


def test_tensor_zplus_ring_unified():
    """the design record streamline: the binary tensor_zplus_ring.TensorZPlusRing is now
    the n-ary zplus_ring.TensorZPlusRing; both constructor forms agree and the
    binary back-compat accessors work."""
    from zplus_ring import TensorZPlusRing as TN
    import tensor_zplus_ring as tz
    assert tz.TensorZPlusRing is TN                             # re-export, same class
    a, b = SU2ZPlusRing(), AbelianZPlusRing(1)
    binary = TN(a, b)                                           # binary positional form
    listed = TN([a, b])                                        # n-ary list form
    assert binary == listed
    assert binary.factor_a == a and binary.factor_b == b       # back-compat accessors
    assert binary.factors == (a, b)
    tern = TN(SU2ZPlusRing(), SU2ZPlusRing(), AbelianZPlusRing(1))   # >2 factors
    assert tern.one_dim_rep_rank() == 1
    assert tern.embed_one_dim_rep((4,)) == (0, 0, (4,))


def test_ringhom_preserves_augmentation():
    """the design record: the shipped flavour homs preserve dimension (ε_target∘φ = ε_source)."""
    from zplus_ring import restriction_hom
    assert identity_hom(AbelianZPlusRing(2)).verify_preserves_augmentation(
        [(0, 0), (1, -1), (2, 0)])
    assert SU2ZPlusRing().augmentation().verify_preserves_augmentation([0, 1, 2, 3])
    assert restriction_hom(AbelianZPlusRing(2), AbelianZPlusRing(1), [[1, 1]]
                           ).verify_preserves_augmentation([(0, 0), (2, -1), (3, 5)])
    assert so3_to_u1_hom().verify_preserves_augmentation([0, 1, 2, 3])


def test_ringhom_preserves_one_dim_reps():
    """the design record: shipped flavour homs carry 1-dim reps to 1-dim reps,
    inducing Λ(source) → Λ(target)."""
    from zplus_ring import restriction_hom
    assert identity_hom(AbelianZPlusRing(2)).verify_preserves_one_dim_reps()
    assert SU2ZPlusRing().augmentation().verify_preserves_one_dim_reps()    # Λ(SU2)=0
    assert restriction_hom(AbelianZPlusRing(2), AbelianZPlusRing(1), [[1, 1]]
                           ).verify_preserves_one_dim_reps()
    assert so3_to_u1_hom().verify_preserves_one_dim_reps()                  # Λ(SO3)=0


def _un_dominant(M, lo=-1, hi=3):
    """The `U(M)`-dominant weights in a small box — weakly decreasing tuples."""
    import itertools
    return [w for w in itertools.product(range(lo, hi), repeat=M)
            if list(w) == sorted(w, reverse=True)]


def test_un_to_sun_hom_basis_values():
    """The central specialization `R(U(M)) → R(SU(M))` sets `det` to 1: a
    weight is shifted down by its last entry and trimmed.  `M = 1` is entirely
    central, so the target is `TrivialZPlusRing` — which is what
    `UNNfKAlgebra(N, 1).coefficient_ring()` is."""
    from zplus_ring import un_to_sun_hom, UNZPlusRing, SUNZPlusRing
    h1 = un_to_sun_hom(1)
    assert isinstance(h1.target, TrivialZPlusRing)
    for lam in [(0,), (1,), (-2,), (5,)]:
        assert h1.apply_basis(lam) == h1.target.one(), lam

    h = un_to_sun_hom(2)
    assert h.source == UNZPlusRing(2) and h.target == SUNZPlusRing(2)
    S = h.target
    assert h.apply_basis((0, 0)) == S.one()
    assert h.apply_basis((1, 1)) == S.one()          # det ↦ 1
    assert h.apply_basis((-1, -1)) == S.one()        # det⁻¹ ↦ 1
    # fundamental and antifundamental both land on the SU(2) doublet
    assert h.apply_basis((1, 0)) == S.basis_element((1,))
    assert h.apply_basis((0, -1)) == S.basis_element((1,))
    assert h.apply_basis((2, -1)) == S.basis_element((3,))


def test_un_to_sun_hom_is_a_ring_hom():
    """Unit, multiplicativity and ⋆-compatibility — the three `RingHom`
    contract properties — on a box of weights at `M = 2, 3, 4`.

    Multiplicativity is the load-bearing one: it is what makes the type-A seam
    the `GNAbeKAlgebra` ↔ `UNNfKAlgebra` seam legitimate
    (D8b: a central specialization is an algebra homomorphism, so associativity,
    bar and orthonormality survive)."""
    from zplus_ring import un_to_sun_hom
    for M in (2, 3, 4):
        h = un_to_sun_hom(M)
        U, S = h.source, h.target
        assert h.apply_basis(U.one_basis()) == S.one(), M
        ws = _un_dominant(M)
        for a in ws:
            assert h.apply_RElement(U.basis_element(a).star()) == \
                h.apply_basis(a).star(), (M, a)
            for b in ws:
                lhs = h.apply_RElement(U.basis_element(a) * U.basis_element(b))
                rhs = h.apply_basis(a) * h.apply_basis(b)
                assert lhs == rhs, (M, a, b, lhs, rhs)


def test_un_to_sun_hom_is_a_restriction():
    """It is `α*` for `α : SU(M) ↪ U(M)`, so it preserves dimension and carries
    1-dim reps to 1-dim reps (`Λ(U(M))` = det powers → `Λ(SU(M)) = 0`)."""
    from zplus_ring import un_to_sun_hom
    for M in (1, 2, 3):
        h = un_to_sun_hom(M)
        assert h.verify_preserves_augmentation(_un_dominant(M)), M
        assert h.verify_preserves_one_dim_reps(), M


def test_un_to_sun_hom_argument_validation():
    from zplus_ring import un_to_sun_hom, UNZPlusRing, SUNZPlusRing
    try:
        un_to_sun_hom(0)
    except ValueError:
        pass
    else:
        raise AssertionError("un_to_sun_hom(0) should raise")
    for bad in (lambda: un_to_sun_hom(2, source=UNZPlusRing(3)),
                lambda: un_to_sun_hom(2, target=SUNZPlusRing(3)),
                lambda: un_to_sun_hom(1, target=SUNZPlusRing(1)),
                lambda: un_to_sun_hom(2, target=TrivialZPlusRing())):
        try:
            bad()
        except TypeError:
            pass
        else:
            raise AssertionError("un_to_sun_hom should reject mismatched rings")


def test_un_to_cartan_hom_basis_values():
    """The Cartan restriction `R(U(n)) → R(U(1)^n)` is the weight diagram — the
    `U(n)` analogue of `su2_to_u1_hom`.  It keeps every fugacity (contrast
    `un_to_sun_hom`, which quotients the centre)."""
    from zplus_ring import un_to_cartan_hom, UNZPlusRing
    h = un_to_cartan_hom(2)
    A = h.target
    assert h.source == UNZPlusRing(2) and A == AbelianZPlusRing(rank=2)
    assert h.apply_basis((0, 0)) == A.one()
    # the U(2) doublet restricts to the two Cartan characters
    assert h.apply_basis((1, 0)) == \
        A.basis_element((1, 0)) + A.basis_element((0, 1))
    # det stays a genuine character here (it is NOT sent to 1)
    assert h.apply_basis((1, 1)) == A.basis_element((1, 1))
    assert h.apply_basis((1, 1)) != A.one()
    # the adjoint-ish (1,-1): three weights incl. the trivial one
    assert h.apply_basis((1, -1)) == (A.basis_element((1, -1)) + A.one()
                                      + A.basis_element((-1, 1)))


def test_un_to_cartan_hom_is_a_restriction():
    """Unit, multiplicativity, ⋆-compatibility, and the two restriction
    properties, at `n = 1, 2, 3`."""
    from zplus_ring import un_to_cartan_hom
    for n in (1, 2, 3):
        h = un_to_cartan_hom(n)
        U, A = h.source, h.target
        assert h.apply_basis(U.one_basis()) == A.one(), n
        ws = _un_dominant(n)
        for a in ws:
            assert h.apply_RElement(U.basis_element(a).star()) == \
                h.apply_basis(a).star(), (n, a)
            for b in ws:
                assert h.apply_RElement(
                    U.basis_element(a) * U.basis_element(b)) == \
                    h.apply_basis(a) * h.apply_basis(b), (n, a, b)
        assert h.verify_preserves_augmentation(ws), n
        assert h.verify_preserves_one_dim_reps(), n


def test_un_homs_are_different_quotients():
    """The two `(G, N)` flavour seams are genuinely different maps: the central
    specialization kills `det`, the Cartan restriction keeps it."""
    from zplus_ring import un_to_sun_hom, un_to_cartan_hom
    c = un_to_cartan_hom(2)
    s = un_to_sun_hom(2)
    assert s.apply_basis((1, 1)) == s.target.one()          # det ↦ 1
    assert c.apply_basis((1, 1)) != c.target.one()          # det survives
    # and the fundamental: one irrep vs two Cartan characters
    assert len(s.apply_basis((1, 0)).terms) == 1
    assert len(c.apply_basis((1, 0)).terms) == 2


def test_ringhom_compat_catches_violation():
    """A hom that breaks dim-preservation fails the verifier (it is emergent,
    not vacuous): send the SU(2) doublet (dim 2) to the trivial rep (dim 1)."""
    S, Z = SU2ZPlusRing(), TrivialZPlusRing()
    bad = RingHom(S, Z, lambda b: Z.one())      # everything ↦ 1  (not dim-preserving)
    assert not bad.verify_preserves_augmentation([0, 1, 2])


# ===========================================================================
# Test runner
# ===========================================================================


def test_tensor_product_torus_embedding_round_trips():
    """`TensorZPlusRing.to_abelian` / `from_abelian` — the product weight layer.

    The class had fusion, duality and dimension but no torus embedding at all,
    so a product flavour group had no WEIGHTS and nothing needing a weight
    diagram (a Weyl-character lift, a Schur trace un-branched into multiplets)
    could use one.  Peeled factor by factor through each factor's own
    `from_abelian`.
    """
    from zplus_ring import (SUNZPlusRing, SU2ZPlusRing, SU3ZPlusRing,
                            TensorZPlusRing)
    import itertools

    # POSITIVE CONTROL: a ONE-factor product must equal the bare factor.
    for N in (2, 3, 4):
        F = SUNZPlusRing(N)
        T = TensorZPlusRing([F])
        assert T.torus_rank() == N - 1
        for lab in [F.one_basis(), (1,), (2,)]:
            assert (dict(F.to_abelian(F.basis_element(lab)).terms)
                    == dict(T.to_abelian(T.basis_element((lab,))).terms)), (N, lab)

    def labels(f):
        n = type(f).__name__
        if n == "SU2ZPlusRing":
            return [f.one_basis(), 1, 2]
        if n == "SU3ZPlusRing":
            return [f.one_basis(), (1, 0), (0, 1)]
        return [f.one_basis()] + ([] if f.N <= 1 else [(1,), (2,)])

    for facs in ([SUNZPlusRing(2), SUNZPlusRing(2)],
                 [SUNZPlusRing(2), SUNZPlusRing(3)],
                 [SUNZPlusRing(3), SUNZPlusRing(2), SUNZPlusRing(2)],
                 [SUNZPlusRing(1), SUNZPlusRing(2)],
                 [SU2ZPlusRing(), SU3ZPlusRing()]):
        T = TensorZPlusRing(facs)
        assert T.torus_rank() == sum(T.factor_torus_ranks())
        for lab in itertools.product(*[labels(f) for f in facs]):
            lab = tuple(lab)
            ab = T.to_abelian(T.basis_element(lab))
            # the diagram carries dim(r) weights, counted with multiplicity
            assert sum(ab.terms.values()) == T.dim(lab), (facs, lab)
            back = T.from_abelian(ab, allow_virtual=True)
            assert dict(back.terms) == {lab: 1}, (facs, lab, dict(back.terms))


def test_tensor_product_torus_embedding_is_linear():
    """A SUM of product characters round-trips, not only a single basis one."""
    from zplus_ring import SUNZPlusRing, TensorZPlusRing

    T = TensorZPlusRing([SUNZPlusRing(2), SUNZPlusRing(3)])
    e = (T.basis_element(((1,), (1,)))
         + T.basis_element(((2,), ())) * 2
         + T.basis_element(T.one_basis()) * 3)
    assert dict(T.from_abelian(T.to_abelian(e)).terms) == dict(e.terms)


def main():
    tests = [
        # TrivialZPlusRing
        test_trivial_basics,
        test_trivial_arithmetic,
        test_trivial_eq_int,
        test_trivial_repr,
        # AbelianZPlusRing
        test_abelian_basics_rank0,
        test_abelian_basics_rank1,
        test_abelian_star_palindrome_rank1,
        test_abelian_distributivity_rank1,
        test_abelian_basics_rank2,
        test_abelian_zero_rank_lemma,
        test_abelian_repr,
        # RElement cross-cutting
        test_relement_unhashable,
        test_relement_ring_mismatch_raises,
        test_relement_int_eq,
        # RLaurent over Trivial
        test_rlaurent_trivial_basics,
        test_rlaurent_trivial_distributivity,
        test_rlaurent_trivial_signed,
        # RLaurent over Abelian — bar acts only on q, ⋆ acts only on R
        test_rlaurent_abelian_bar_acts_only_on_q,
        test_rlaurent_abelian_palindrome_in_q_only,
        test_rlaurent_abelian_non_palindrome,
        test_rlaurent_abelian_distributivity,
        test_rlaurent_abelian_bar_intertwines_mul,
        test_rlaurent_abelian_bar_involutive,
        test_rlaurent_star_acts_only_on_R,
        test_rlaurent_star_trivial_on_trivial_ring,
        test_rlaurent_bar_and_star_commute,
        test_rlaurent_star_involutive,
        test_rlaurent_star_distributes_over_mult,
        test_rlaurent_zero_and_one_compare_int,
        test_rlaurent_q_times_q_inverse_is_one,
        test_rlaurent_int_and_relement_scalar_mul,
        # Z₊-ring axioms
        test_axioms_trivial_ring,
        test_axioms_abelian_rank2,
        # SU2ZPlusRing
        test_su2_basics,
        test_su2_commutative,
        test_su2_star_identity,
        test_su2_to_abelian_chi1,
        test_su2_to_abelian_intertwines_multiplication,
        test_su2_negative_basis_raises,
        test_su2_repr_eq_hash,
        test_axioms_su2,
        test_rlaurent_su2_bar_acts_only_on_q,
        test_rlaurent_su2_palindrome_in_q,
        # SO3ZPlusRing
        test_so3_basics,
        test_so3_dimensions_match,
        test_so3_commutative,
        test_so3_star_identity,
        test_so3_to_abelian_vector,
        test_so3_to_abelian_intertwines_multiplication,
        test_so3_to_su2_intertwines_multiplication,
        test_so3_to_su2_image_is_even,
        test_so3_negative_basis_raises,
        test_so3_repr_eq_hash,
        test_axioms_so3,
        test_rlaurent_so3_distributivity,
        # Symmetry-reduction ring homs
        test_so3_to_u1_hom_identity_image,
        test_so3_to_u1_hom_vector_image,
        test_so3_to_u1_hom_higher,
        test_so3_to_u1_hom_is_multiplicative,
        test_so3_to_u1_hom_preserves_identity,
        test_so3_to_u1_hom_intertwines_star,
        test_so3_to_u1_hom_image_is_weyl_symmetric,
        test_so3_to_u1_hom_argument_validation,
        test_so3_to_su2_hom_image_is_even_n,
        test_so3_to_su2_hom_is_multiplicative,
        test_u1_weyl_to_so3_round_trip,
        test_u1_weyl_to_so3_virtual_character,
        test_u1_weyl_to_so3_rejects_non_symmetric,
        test_u1_weyl_to_so3_rejects_wrong_source_ring,
        test_so3_chain_diagram_commutes_up_to_mu_squared,
        # hash/eq contract (audit regression)
        test_hash_eq_contract,
        # the design record — augmentation (dim) primitive
        test_augmentation_dim_is_hom_all_rings,
        test_augmentation_known_dimensions,
        test_augmentation_generalises_abelian_hom,
        # the design record — 1-dim-rep subring Λ (lift torsor) + TensorZPlusRing streamline
        test_one_dim_reps_structure_all_rings,
        test_one_dim_rep_inclusion_values,
        test_tensor_zplus_ring_unified,
        # the design record — RingHom compatibility with augmentation + 1-dim reps
        test_ringhom_preserves_augmentation,
        test_ringhom_preserves_one_dim_reps,
        test_ringhom_compat_catches_violation,
        # the central specialization R(U(M)) → R(SU(M)) (the
        # (G,N) ↔ U(N)+N_f flavour seam)
        test_un_to_sun_hom_basis_values,
        test_un_to_sun_hom_is_a_ring_hom,
        test_un_to_sun_hom_is_a_restriction,
        test_un_to_sun_hom_argument_validation,
        test_un_to_cartan_hom_basis_values,
        test_un_to_cartan_hom_is_a_restriction,
        test_un_homs_are_different_quotients,
        test_tensor_product_torus_embedding_round_trips,
        test_tensor_product_torus_embedding_is_linear,
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
            failed += 1
    print(f"\n{'=' * 50}")
    print(f"  Results: {passed} passed, {failed} failed of {passed + failed}")
    print(f"{'=' * 50}")
    return failed == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
