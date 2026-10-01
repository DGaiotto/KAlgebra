"""Tests for `ConeKAlgebra` + the `Cone` class hierarchy + ρ²-orbit
canonicalisation at Layer 1.

Covers (against `PentagonKAlg`):

  * Cone-class surface: `MonomialCone` instances produced by
    `ConeData.iter_cones()`, `Cone.mult_gens`, `Cone.pbw_to_canonical`,
    `Cone.canonical_to_pbw`.

  * `CharacterCone` is implemented (SU(2) Chebyshev change-of-basis).

  * Layer-1 ρ²-orbit canonicalisation: every seed produced by
    `simplify_trace_via_cone_data` is the minimum-by-Python-sort label
    in its ρ²-orbit.  Pentagon's five mult-gen seeds collapse to the
    single canonical (0, 1, 0).

  * The "miracle": for `Tr(L_0^a)` reduced to `c_1(q)·Tr(1) + c_L(q)·Tr(L)`,
    the normalised polynomials stabilise to
        c_1(q) / q^{emin} → -Tr(L)
        c_L(q) / q^{emin} →  Tr(1)
    to depth ~2a in q.  Pinned for a ∈ {4, 6, 8, 10, 12} at K=60.
"""

from __future__ import annotations

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from cone_data import (
    Cone, ConeData, FiniteConeData,
)
from cone_kalgebra import ConeKAlgebra
from kalgebra_samples import PentagonKAlg
from pentagon_cone_data import PENTAGON_CONE_DATA
from laurent_poly import LaurentPoly


def test_pentagon_is_a_cone_kalgebra():
    assert issubclass(PentagonKAlg, ConeKAlgebra)
    assert isinstance(PentagonKAlg(), ConeKAlgebra)


def test_iter_cones_yields_monomial_cones():
    cones = list(PENTAGON_CONE_DATA.iter_cones())
    assert len(cones) == 5
    for c in cones:
        assert c.is_monomial()
        assert isinstance(c, Cone)
        assert len(c.mult_gens()) == 2  # pentagon: each cone is {i, i+1}
        assert c.parent_data() is PENTAGON_CONE_DATA


def test_cone_of_label_locates_the_right_cone():
    # Identity → any cone (we don't pin which).
    c0 = PENTAGON_CONE_DATA.cone_of_label((0, 0, 0))
    assert c0.is_monomial()
    # Single mult-gen on tile 2 → its cone contains 2.
    c2 = PENTAGON_CONE_DATA.cone_of_label((2, 1, 0))
    assert 2 in c2.mult_gens()
    # Two-mult-gen label (cone {2,3}).
    c23 = PENTAGON_CONE_DATA.cone_of_label((2, 1, 1))
    assert c23.mult_gens() == frozenset({2, 3})
    # Wrap-around cone {0, 4}.
    c04 = PENTAGON_CONE_DATA.cone_of_label((4, 2, 1))
    assert c04.mult_gens() == frozenset({0, 4})


def test_monomial_cone_pbw_canonical_roundtrip():
    """For MonomialCone, canonical_to_pbw ∘ pbw_to_canonical = identity
    on PBW monomials whose mult-gens lie in the cone."""
    cone_01 = next(
        c for c in PENTAGON_CONE_DATA.iter_cones()
        if c.mult_gens() == frozenset({0, 1})
    )
    # PBW monomial L_0^a L_1^b → canonical-basis label (0, a, b).
    for a, b in [(1, 0), (0, 1), (1, 1), (2, 3), (3, 2)]:
        powers = {}
        if a > 0:
            powers[0] = a
        if b > 0:
            powers[1] = b
        elem = cone_01.pbw_to_canonical(powers)
        # Single-term element for MonomialCone.
        assert len(elem.terms) == 1
        # Recover via canonical_to_pbw.
        (native_label, _coeff), = list(elem.terms.items())
        cp = cone_01.canonical_to_pbw(native_label)
        assert len(cp) == 1
        (word, _wcoeff), = list(cp.items())
        # The word should reproduce the input powers (in canonical order).
        recovered_powers = {}
        for g in word:
            recovered_powers[g] = recovered_powers.get(g, 0) + 1
        assert recovered_powers == powers


def test_character_cone_implemented():
    """`CharacterCone` is implemented (SU(2) Chebyshev change-of-basis)."""
    from cone_data import su2_monomial_to_chars, su2_char_to_monomials
    # SU(2) Chebyshev: χ_1^2 = χ_0+χ_2 ; χ_2 = χ_1^2 − 1 ; and they round-trip
    assert su2_monomial_to_chars(2) == {0: 1, 2: 1}
    assert su2_char_to_monomials(2) == {0: -1, 2: 1}
    for k in range(6):
        acc = {}
        for l, c in su2_char_to_monomials(k).items():
            for j, cc in su2_monomial_to_chars(l).items():
                acc[j] = acc.get(j, 0) + c * cc
        assert {j: v for j, v in acc.items() if v} == {k: 1}


def test_layer1_seeds_are_rho2_canonical():
    """Layer 1 (= tagged-cycle + ρ²-orbit canonicalisation) outputs
    seed labels that are each the minimum-by-Python-sort element of
    their ρ²-orbit.  For Pentagon: only (0, 0, 0) and (0, 1, 0) ever
    appear, never (i, 1, 0) for i ∈ {1, 2, 3, 4}."""
    A = PentagonKAlg()
    cd = A.cone_data()
    # Sample inputs that exercise the reducer's full path.
    inputs = [
        (0, 0, 0),
        (0, 1, 0), (1, 1, 0), (2, 1, 0), (3, 1, 0), (4, 1, 0),
        (0, 2, 0), (1, 2, 0), (0, 3, 0),
        (0, 1, 1), (1, 1, 1), (2, 2, 1),
        (0, 0, 1),  # canonicalises through `from_cone_label`
        (3, 1, 2),
    ]
    allowed = {(0, 0, 0), (0, 1, 0)}
    for label in inputs:
        simplified = cd.simplify_trace_via_cone_data(A, label)
        for seed in simplified.terms:
            assert seed in allowed, (
                f"Layer-1 seed {seed} is not a canonical ρ²-orbit rep "
                f"for input {label}; allowed = {allowed}"
            )


def test_rho2_invariance_of_trace_on_single_mult_gens():
    """`Tr(L_i)` is independent of `i` (all 5 ρ²-related).  This was
    previously enforced implicitly by `_trace_residual` returning the
    same value on all (i, 1, 0); now it's enforced at Layer 1 by the
    ρ²-orbit fold."""
    A = PentagonKAlg()
    K = 16
    tr0 = A.trace((0, 1, 0), K=K)
    for i in range(1, 5):
        tri = A.trace((i, 1, 0), K=K)
        assert tri.coeffs == tr0.coeffs, (
            f"Tr(L_{i}) differs from Tr(L_0) at K={K}"
        )


def test_rho2_invariance_of_trace_on_squares():
    """Same test for `L_i^2`."""
    A = PentagonKAlg()
    K = 16
    tr0 = A.trace((0, 2, 0), K=K)
    for i in range(1, 5):
        tri = A.trace((i, 2, 0), K=K)
        assert tri.coeffs == tr0.coeffs, (
            f"Tr(L_{i}^2) differs from Tr(L_0^2) at K={K}"
        )


def _to_int_dict(rps):
    """Extract integer q-coefficients from an RPowerSeries over
    TrivialZPlusRing."""
    out = {}
    for e, rc in rps.coeffs.items():
        for _, v in rc.terms.items():
            out[e] = v
    return out


def test_miracle_c1_stabilises_to_minus_TrL():
    """For Tr(L_0^a) = c_1·Tr(1) + c_L·Tr(L), the normalised
    coefficient `c_1(q) / q^{emin}` stabilises (as a → ∞) to -Tr(L).

    Pin: for each a ∈ {4, 6, 8, 10, 12}, c_1/q^{emin} agrees with
    -Tr(L) up to q^{2a−1} inclusive.
    """
    A = PentagonKAlg()
    cd = A.cone_data()
    K_HIGH = 60
    tr_L = _to_int_dict(A._trace_residual((0, 1, 0), K=K_HIGH))
    neg_trL = {e: -v for e, v in tr_L.items()}

    for a, depth in [(4, 7), (6, 11), (8, 15), (10, 19), (12, 23)]:
        simplified = cd.simplify_trace_via_cone_data(A, (0, a, 0))
        c1 = simplified.terms.get((0, 0, 0), LaurentPoly.zero())
        cL = simplified.terms.get((0, 1, 0), LaurentPoly.zero())
        emin = min(list(c1._coeffs.keys()) + list(cL._coeffs.keys()))
        c1_norm = {e - emin: c for e, c in c1._coeffs.items()}
        for e in range(0, depth + 1):
            assert c1_norm.get(e, 0) == neg_trL.get(e, 0), (
                f"miracle (c_1 vs -Tr(L)) fails at a={a}, e={e}: "
                f"got {c1_norm.get(e, 0)}, expected {neg_trL.get(e, 0)}"
            )


def test_miracle_cL_stabilises_to_Tr1():
    """For Tr(L_0^a) = c_1·Tr(1) + c_L·Tr(L), the normalised
    coefficient `c_L(q) / q^{emin}` stabilises (as a → ∞) to Tr(1).

    Pin: for each a ∈ {4, 6, 8, 10, 12}, c_L/q^{emin} agrees with
    Tr(1) up to q^{2a−2} inclusive.
    """
    A = PentagonKAlg()
    cd = A.cone_data()
    K_HIGH = 60
    tr_1 = _to_int_dict(A._trace_residual((0, 0, 0), K=K_HIGH))

    for a, depth in [(4, 6), (6, 10), (8, 14), (10, 18), (12, 22)]:
        simplified = cd.simplify_trace_via_cone_data(A, (0, a, 0))
        c1 = simplified.terms.get((0, 0, 0), LaurentPoly.zero())
        cL = simplified.terms.get((0, 1, 0), LaurentPoly.zero())
        emin = min(list(c1._coeffs.keys()) + list(cL._coeffs.keys()))
        cL_norm = {e - emin: c for e, c in cL._coeffs.items()}
        for e in range(0, depth + 1):
            assert cL_norm.get(e, 0) == tr_1.get(e, 0), (
                f"miracle (c_L vs Tr(1)) fails at a={a}, e={e}: "
                f"got {cL_norm.get(e, 0)}, expected {tr_1.get(e, 0)}"
            )


def test_pentagon_trace_of_powers_is_O_of_q_to_the_a():
    """Canonical-orthonormality axiom (`I_{a,b} = δ_{a,b} + O(q)`)
    made visible: `Tr(L_0^a) = ⟨L_0^a, 1⟩` must start at q^a for
    `a ≥ 1`, because `L_0^a ≠ 1`.

    If this fails, the Yang-Lee Rogers-Ramanujan Layer-2 wiring is
    wrong: the cancellation `c_1·Tr(1) + c_L·Tr(L) = O(q^a)` requires
    the Layer-1 coefficients and Layer-2 Nahm sums to be in exact
    mutual agreement.
    """
    A = PentagonKAlg()
    K = 12
    for a in range(1, 8):
        # L_0^a = (0, a, 0) in Pentagon's canonical labels.
        label = (0, a, 0)
        tr = _to_int_dict(A.trace(label, K=K))
        if not tr:
            continue
        emin = min(tr)
        assert emin >= a, (
            f"orthonormality violation: Tr(L_0^{a}) starts at q^{emin}, "
            f"expected q^{a} or higher; got coeffs {sorted(tr.items())[:5]}"
        )


if __name__ == "__main__":
    tests = [
        test_pentagon_is_a_cone_kalgebra,
        test_iter_cones_yields_monomial_cones,
        test_cone_of_label_locates_the_right_cone,
        test_monomial_cone_pbw_canonical_roundtrip,
        # test_character_cone_and_wild_cone_are_stubs,  # uses pytest_compat
        test_layer1_seeds_are_rho2_canonical,
        test_rho2_invariance_of_trace_on_single_mult_gens,
        test_rho2_invariance_of_trace_on_squares,
        test_miracle_c1_stabilises_to_minus_TrL,
        test_miracle_cL_stabilises_to_Tr1,
        test_pentagon_trace_of_powers_is_O_of_q_to_the_a,
    ]
    fails = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
        except AssertionError as e:
            print(f"FAIL: {t.__name__}: {e}")
            fails += 1
        except Exception as e:
            print(f"ERROR: {t.__name__}: {type(e).__name__}: {e}")
            fails += 1
    print()
    if fails == 0:
        print(f"All {len(tests)} cone-kalgebra tests passed.")
    else:
        print(f"{fails} failure(s).")
        sys.exit(1)

    # CharacterCone is implemented (Chebyshev).
    from cone_data import su2_monomial_to_chars
    if su2_monomial_to_chars(2) == {0: 1, 2: 1}:
        print("PASS: CharacterCone (SU(2) Chebyshev) implemented")
    else:
        print("FAIL: CharacterCone Chebyshev wrong"); sys.exit(1)
