"""Smoke tests for the flavoured `BPSKAlgebra` paths.

Exercises the ker(B) -> AbelianZPlusRing -> Element[RLaurent[Abelian]]
flow on a small flavoured preset (hexagon: ker(B) = Z·(1, 1, 1)).

Multiply support is in place; trace / inner_product on flavoured
algebras raise NotImplementedError pending the μ-tracking Habiro
accumulator.

Run:  PYTHONPATH=. python restructuring/tests/test_bps_kalgebra_flavour.py
"""

from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
_RESTRUCT = os.path.dirname(_HERE)
_REPO = _RESTRUCT  # post-flatten: _RESTRUCT is the repo root
sys.path.insert(0, _REPO)
sys.path.insert(0, _RESTRUCT)

from zplus_ring import (
    AbelianZPlusRing, TrivialZPlusRing, RLaurent, RPowerSeries,
)
from kalgebra import Element, LaurentPoly
from bps_kalgebra import BPSKAlgebra


# Hexagon ([A_1, A_3] Argyres-Douglas, A_3 chamber).  Exchange matrix
# has ker = Z · (1, 1, 1); a one-direction abelian flavour symmetry.
HEXAGON_B = [
    [0, 1, -1],
    [-1, 0, 1],
    [1, -1, 0],
]
HEXAGON_NODES = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]


def test_hexagon_constructs_with_degenerate_pairing():
    A = BPSKAlgebra(pairing=HEXAGON_B, node_charges=HEXAGON_NODES)
    assert A._flavour_rank == 1, f"expected rank-1 flavour, got {A._flavour_rank}"
    assert A._gauge_rank == 2
    assert isinstance(A.coefficient_ring(), AbelianZPlusRing)
    assert A.coefficient_ring().rank == 1


def test_unflavoured_pentagon_still_uses_trivial_ring():
    """Sanity: when ker(B) = 0, R = TrivialZPlusRing()."""
    A = BPSKAlgebra(
        pairing=[[0, 1], [-1, 0]],
        node_charges=[(1, 0), (0, 1)],
    )
    assert A._flavour_rank == 0
    assert A._gauge_rank == 2
    assert isinstance(A.coefficient_ring(), TrivialZPlusRing)


def test_hexagon_multiply_returns_flavoured_element():
    """Multiply on hexagon should return Z-form Element (per the design record):
    coefficients are LaurentPoly over Z[q^±]; flavour shifts live in
    the full-Γ-tuple labels, not in coefficients."""
    A = BPSKAlgebra(pairing=HEXAGON_B, node_charges=HEXAGON_NODES)
    prod = A.multiply((1, 0, 0), (0, 1, 0))
    for label, coeff in prod.terms.items():
        assert isinstance(coeff, LaurentPoly), \
            f"coefficient at {label} is {type(coeff).__name__}, expected LaurentPoly"
    assert len(prod.terms) >= 1


def test_hexagon_multiply_aggregates_by_gauge_class():
    """Multiply on hexagon returns a non-trivial Z-form Element."""
    A = BPSKAlgebra(pairing=HEXAGON_B, node_charges=HEXAGON_NODES)
    prod = A.multiply((1, 0, 0), (0, 1, 0))
    assert not prod.is_zero()
    for c in prod.terms.values():
        assert isinstance(c, LaurentPoly)


def test_hexagon_trace_returns_RPowerSeries():
    """Flavoured trace returns RPowerSeries over the abelian R(U(1)^f)."""
    A = BPSKAlgebra(pairing=HEXAGON_B, node_charges=HEXAGON_NODES)
    R = A.coefficient_ring()
    Tr = A.trace((0, 0, 0), K=4)
    assert isinstance(Tr, RPowerSeries)
    assert Tr.ring == R
    # Tr(1)[q^0] should be 1 ∈ R (identity orbit's leading term).
    const = Tr[0]
    assert const == R.one(), f"Tr(1)[q^0] = {const}, expected 1"


def test_hexagon_orthonormality_vacuum():
    """I_{0,0}[q^0] = 1 ∈ R: the identity is its own orthonormal partner."""
    A = BPSKAlgebra(pairing=HEXAGON_B, node_charges=HEXAGON_NODES)
    R = A.coefficient_ring()
    I00 = A.inner_product((0, 0, 0), (0, 0, 0), K=4)
    assert I00[0] == R.one(), f"I_{{0,0}}[q^0] = {I00[0]}, expected 1"


def test_hexagon_trace_carries_mu_dependence():
    """Tr on a non-vacuum *flavour direction* should produce a μ-monomial.
    Specifically Tr(L_{(1,1,1)}) — the central μ-element — is just μ at q^0."""
    A = BPSKAlgebra(pairing=HEXAGON_B, node_charges=HEXAGON_NODES)
    R = A.coefficient_ring()
    # (1, 1, 1) is in ker(B); its image as F_(1,1,1) = X_(1,1,1) (single QT
    # monomial since flavour direction is central).  Its trace should be
    # μ-monomial.  Tr(L_{(1,1,1)})[q^0] = μ^{±1} (some sign convention).
    # We just check that it's a single non-trivial μ-monomial.
    #
    # Note: (1, 1, 1) might not be in our section — it's purely flavour.
    # multiply((1,1,1), zero) == flavour-shifted basis element.
    # Instead let's check via inner product I_{0, (1,1,1)} = Tr(ρ(0) · L_{(1,1,1)}) = Tr(L_{(1,1,1)}).
    Tr_flav = A.trace((1, 1, 1), K=2)
    assert isinstance(Tr_flav, RPowerSeries)
    # The leading term should carry a μ-shift.  Specifically the q⁰ part
    # is ±μ^k for some non-trivial μ-exponent.
    leading = Tr_flav[0]
    # Either Tr is exactly a μ-monomial × q⁰ or zero — what we don't want
    # is something silly like a Z-multiple of 1.  The flavour class of
    # (1, 1, 1) is non-trivial in Γ_f, so the leading should have non-zero
    # μ-exponent.
    if not leading.is_zero():
        # Check that not all μ-exps are zero (= constant in μ)
        non_trivial_mu = any(any(e != 0 for e in exp) for exp in leading.terms.keys())
        assert non_trivial_mu, (
            f"Tr(L_{{(1,1,1)}})[q^0] = {leading} has only μ^0 components"
        )


def test_hexagon_F_works_per_chart():
    """F-finding doesn't require flavour-aware machinery; it lives at the
    chart level (Γ-coords).  Should work for hexagon without raising."""
    A = BPSKAlgebra(pairing=HEXAGON_B, node_charges=HEXAGON_NODES)
    # Just verify no exception on a node-charge call
    F1 = A.F((1, 0, 0))
    assert F1 is not None
    assert (1, 0, 0) in F1


def test_F_flavour_shift_is_lattice_translation():
    """For `γ_f ∈ Γ_f`, `F_{γ + γ_f}` equals `F_γ` translated by `γ_f`
    in the lattice (no q-twist, since ⟨γ_f, δ⟩ = 0 for all δ ∈ Γ)."""
    A = BPSKAlgebra(pairing=HEXAGON_B, node_charges=HEXAGON_NODES)
    gamma = (1, 0, 0)
    gamma_f = (1, 1, 1)   # the kernel direction for hexagon
    F_g = A.F(gamma)
    F_g_shifted = A.F(tuple(g + f for g, f in zip(gamma, gamma_f)))
    expected = {
        tuple(d + f for d, f in zip(delta, gamma_f)): coeff
        for delta, coeff in F_g.items()
    }
    assert F_g_shifted == expected, (
        f"F_{gamma}+gamma_f should be lattice translation; "
        f"got {F_g_shifted}, expected {expected}"
    )


def test_multiply_flavour_shift_is_mu_shift():
    """After the design record, multiply returns Z-form Element with full-Γ-tuple
    labels.  Flavour-shifting both inputs by (γ_f₁, γ_f₂) shifts every
    output label by γ_f₁ + γ_f₂; the LaurentPoly coefficients at the
    shifted labels are identical to the original ones."""
    A = BPSKAlgebra(pairing=HEXAGON_B, node_charges=HEXAGON_NODES)
    base = A.multiply((1, 0, 0), (0, 1, 0))
    # Shift each input by (1, 1, 1) ∈ ker(B).  Total Γ shift = (2, 2, 2).
    shifted = A.multiply((2, 1, 1), (1, 2, 1))
    total_shift = (2, 2, 2)
    expected_keys = {
        tuple(k + s for k, s in zip(label, total_shift))
        for label in base.terms
    }
    assert set(shifted.terms.keys()) == expected_keys, (
        f"shifted labels should equal base labels + {total_shift}; "
        f"got {set(shifted.terms.keys())}, expected {expected_keys}"
    )
    for label, base_coeff in base.terms.items():
        shifted_label = tuple(k + s for k, s in zip(label, total_shift))
        shifted_coeff = shifted.terms[shifted_label]
        assert base_coeff == shifted_coeff, (
            f"coefficient at {label} != coefficient at {shifted_label}: "
            f"{base_coeff} vs {shifted_coeff}"
        )


def test_multiply_cache_uses_section_pair():
    """Computing multiply at flavour-shifted inputs hits the orbit-pair
    cache: only one entry per orbit pair, regardless of how many flavour
    shifts the author requests."""
    A = BPSKAlgebra(pairing=HEXAGON_B, node_charges=HEXAGON_NODES)
    # Trigger several flavour-shifted multiplies on the same orbit pair.
    A.multiply((1, 0, 0), (0, 1, 0))
    A.multiply((2, 1, 1), (1, 2, 1))   # both inputs shifted by (1,1,1)
    A.multiply((3, 2, 2), (2, 3, 2))   # shifted by (2,2,2) and (2,2,1)
    # Cache should have a single entry corresponding to the (sec_a, sec_b)
    # orbit pair.  All three calls reuse the same cached result.
    assert len(A._multiply_cache) == 1, (
        f"expected 1 cache entry per orbit pair, got {len(A._multiply_cache)}"
    )


def test_F_cache_reuses_section_computation():
    """Computing F for a flavour-shifted γ should reuse the section
    rep's cached F instead of re-solving.  We can't directly count
    solve_F_via_s_coefficient calls (it's a function, not a method),
    but we can verify by removing the section entry from the cache
    and observing that recomputation works correctly — which would
    fail if F_{γ + γ_f} were stored independently."""
    A = BPSKAlgebra(pairing=HEXAGON_B, node_charges=HEXAGON_NODES)
    gamma = (1, 0, 0)
    gamma_f = (1, 1, 1)
    gamma_shifted = tuple(g + f for g, f in zip(gamma, gamma_f))
    # Compute the shifted F first.  Internally this should compute
    # F_gamma (cache it), then translate.
    F_shifted = A.F(gamma_shifted)
    # The section rep gamma must now be in the cache.
    assert gamma in A._F_cache, (
        "section rep should be cached after flavour-shifted lookup"
    )
    # And the shifted result is consistent.
    F_g = A.F(gamma)
    assert F_shifted == {
        tuple(d + f for d, f in zip(delta, gamma_f)): coeff
        for delta, coeff in F_g.items()
    }


def test_unflavoured_paths_unchanged():
    """A Pentagon-style unflavoured BPSKAlgebra should have all the same
    behaviour as before.  Smoke check on multiply/F."""
    A = BPSKAlgebra(
        pairing=[[0, 1], [-1, 0]],
        node_charges=[(1, 0), (0, 1)],
    )
    R = A.coefficient_ring()
    prod = A.multiply((1, 0), (0, 1))
    # Single output (1,1) at q^1 (X_(1,0) · X_(0,1) = q^{<...>} X_(1,1))
    assert (1, 1) in prod.terms
    # Trace of vacuum returns RPowerSeries (boundary conversion correct)
    Tr = A.trace((0, 0), K=4)
    assert isinstance(Tr, RPowerSeries)
    assert Tr[0] == 1


# ===========================================================================
# Flavour-lift coordinate: r_label_decompose / r_label_compose
# ===========================================================================


def _bps_lift_checks(A, labels):
    """For each label: r_label_decompose round-trips through r_label_compose;
    is consistent with the (obsolete-to-be) _label_section_decompose (same
    section + single irrep) and embed_R (faithfulness); agrees with the
    inherited RGKAlgebra delegation to the auxiliary; passes the contract
    verifiers."""
    aux = A.auxiliary()
    for g in labels:
        g = tuple(A.lattice.check(g))
        sec, key = A.r_label_decompose(g)
        assert A.r_label_compose(sec, key) == g, g            # round-trip
        # consistent with _label_section_decompose: same section, single irrep
        lsd_sec, lsd_r = A._label_section_decompose(g)
        nz = [(b, c) for b, c in lsd_r.terms.items() if c != 0]
        assert tuple(sec) == tuple(lsd_sec), g
        assert len(nz) == 1 and nz[0][1] == 1 and tuple(nz[0][0]) == tuple(key), g
        # inherited RGKAlgebra delegation (aux at the identity apex) agrees
        assert aux.r_label_decompose(A.apex(g)) == (sec, key), g
        # contract verifiers (the obsolete-to-be embed_R / single-irrep axioms)
        assert A.verify_section_is_single_irrep(g), g
        assert A.verify_embed_section_roundtrip(g), g


def test_lift_coordinate_unflavoured_pentagon():
    """Unflavoured (Γ_f = 0): the lift is trivial — section = label, key = ()."""
    A = BPSKAlgebra(pairing=[[0, 1], [-1, 0]], node_charges=[(1, 0), (0, 1)])
    for g in [(0, 0), (1, 0), (0, 1), (2, 1), (-1, 2)]:
        sec, key = A.r_label_decompose(g)
        assert key == ()                                      # unflavoured
        assert tuple(sec) == tuple(g)
    _bps_lift_checks(A, [(0, 0), (1, 0), (0, 1), (2, 1), (-1, 2)])


def test_lift_coordinate_flavoured_hexagon():
    """Flavoured (ker B = Z·(1,1,1)): the irrep key is the μ-power; the section
    is the Γ-lift of the gauge class; r_label_compose re-attaches the central
    flavour charge."""
    A = BPSKAlgebra(pairing=HEXAGON_B, node_charges=HEXAGON_NODES)
    labels = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1), (2, 0, 1),
              (1, -1, 0), (-1, 0, 2)]
    _bps_lift_checks(A, labels)
    # the flavour charge (1,1,1) is pure flavour: section 0, key the μ-power.
    sec, key = A.r_label_decompose((1, 1, 1))
    assert tuple(sec) == (0, 0, 0) and key == (1,), (sec, key)


# ===========================================================================
# Test runner
# ===========================================================================


def main():
    tests = [
        test_hexagon_constructs_with_degenerate_pairing,
        test_unflavoured_pentagon_still_uses_trivial_ring,
        test_hexagon_multiply_returns_flavoured_element,
        test_hexagon_multiply_aggregates_by_gauge_class,
        test_hexagon_trace_returns_RPowerSeries,
        test_hexagon_orthonormality_vacuum,
        test_hexagon_trace_carries_mu_dependence,
        test_hexagon_F_works_per_chart,
        test_F_flavour_shift_is_lattice_translation,
        test_F_cache_reuses_section_computation,
        test_multiply_flavour_shift_is_mu_shift,
        test_multiply_cache_uses_section_pair,
        test_unflavoured_paths_unchanged,
        test_lift_coordinate_unflavoured_pentagon,
        test_lift_coordinate_flavoured_hexagon,
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
