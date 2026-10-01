"""Tests for `QuantumTorusKAlg`.

Run:  PYTHONPATH=. python restructuring/tests/test_quantum_torus_kalgebra.py
"""

from __future__ import annotations

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESTRUCT = os.path.dirname(_HERE)
_REPO = _RESTRUCT  # post-flatten: _RESTRUCT is the repo root
sys.path.insert(0, _REPO)
sys.path.insert(0, _RESTRUCT)

from kalgebra import Element
from laurent_poly import LaurentPoly
from quantum_torus_kalgebra import QuantumTorusKAlg
from zplus_ring import (
    AbelianZPlusRing, TrivialZPlusRing, RLaurent,
)


# ---------------------------------------------------------------------------
# Constructor / shape
# ---------------------------------------------------------------------------


def test_construct_z2_symplectic():
    A = QuantumTorusKAlg([[0, 1], [-1, 0]])
    assert A.rank == 2
    assert A.gauge_rank == 2
    assert A.flavour_rank == 0
    assert isinstance(A.coefficient_ring(), TrivialZPlusRing)
    assert A.identity() == (0, 0)


def test_construct_hexagon_pairing_flavoured():
    A = QuantumTorusKAlg([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])
    assert A.rank == 3
    assert A.gauge_rank == 2
    assert A.flavour_rank == 1
    assert isinstance(A.coefficient_ring(), AbelianZPlusRing)
    assert A.coefficient_ring().rank == 1
    assert A.identity() == (0, 0, 0)
    assert A.kernel_basis == [(1, 1, 1)]


def test_construct_zero_pairing_pure_flavour():
    A = QuantumTorusKAlg([[0, 0], [0, 0]])
    assert A.gauge_rank == 0
    assert A.flavour_rank == 2
    assert A.identity() == (0, 0)


def test_construct_rejects_non_square():
    try:
        QuantumTorusKAlg([[0, 1, -1], [-1, 0, 1]])
    except ValueError:
        return
    raise AssertionError("expected ValueError on non-square pairing")


def test_construct_rejects_non_antisymmetric():
    try:
        QuantumTorusKAlg([[0, 1], [1, 0]])
    except ValueError:
        return
    raise AssertionError("expected ValueError on non-antisymmetric")


# ---------------------------------------------------------------------------
# Multiplication
# ---------------------------------------------------------------------------


def test_multiply_unflavoured_z2_matches_hand_coded():
    """Should agree with `QuantumTorusZ2KAlg`: `<(a,b),(c,d)> = ad - bc`."""
    A = QuantumTorusKAlg([[0, 1], [-1, 0]])
    R = A.coefficient_ring()
    e = A.multiply((1, 0), (0, 1))  # <(1,0),(0,1)> = 1
    assert e == Element({(1, 1): LaurentPoly({1: 1})}), f"got {e}"
    e2 = A.multiply((0, 1), (1, 0))  # <(0,1),(1,0)> = -1
    assert e2 == Element({(1, 1): LaurentPoly({-1: 1})}), f"got {e2}"


def test_multiply_associativity_unflavoured():
    A = QuantumTorusKAlg([[0, 1], [-1, 0]])
    labels = [(0, 0), (1, 0), (0, 1), (1, 1), (-1, 2)]
    R = A.coefficient_ring()
    for a in labels:
        for b in labels:
            for c in labels:
                ab = A.multiply(a, b)
                lhs = A.multiply_elements(ab, Element.basis(c))
                bc = A.multiply(b, c)
                rhs = A.multiply_elements(Element.basis(a), bc)
                assert lhs == rhs, f"associativity fails on a={a}, b={b}, c={c}"


def test_multiply_flavoured_hexagon_pairing_uses_full_gamma():
    """Multiplication uses the full ambient pairing on Γ-tuples."""
    A = QuantumTorusKAlg([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])
    R = A.coefficient_ring()
    e = A.multiply((1, 0, 0), (0, 1, 0))  # <γ_0, γ_1> = 1
    assert e == Element({(1, 1, 0): LaurentPoly({1: 1})}), f"got {e}"
    e2 = A.multiply((1, 0, 0), (0, 0, 1))  # <γ_0, γ_2> = -1
    assert e2 == Element({(1, 0, 1): LaurentPoly({-1: 1})}), f"got {e2}"


def test_multiply_central_flavour_commutes():
    """`X_{γ_f} · X_γ = X_γ · X_{γ_f} = X_{γ + γ_f}` for `γ_f ∈ Γ_f`."""
    A = QuantumTorusKAlg([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])
    flavour = (1, 1, 1)  # in ker
    for gamma in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1)]:
        e_left = A.multiply(flavour, gamma)
        e_right = A.multiply(gamma, flavour)
        assert e_left == e_right, (
            f"central flavour fails to commute: {flavour}·{gamma} != {gamma}·{flavour}"
        )


# ---------------------------------------------------------------------------
# ρ
# ---------------------------------------------------------------------------


def test_rho_negation_and_inverse():
    A = QuantumTorusKAlg([[0, 1], [-1, 0]])
    assert A.rho((1, 2)) == (-1, -2)
    assert A.rho_inverse((1, 2)) == (-1, -2)
    assert A.verify_rho_inverse((3, -4))
    assert A.verify_rho_fixes_identity()


def test_rho_is_twisted_automorphism_unflavoured():
    A = QuantumTorusKAlg([[0, 1], [-1, 0]])
    labels = [(0, 0), (1, 0), (0, 1), (1, 1), (-1, 2)]
    for a in labels:
        for b in labels:
            assert A.verify_rho_is_twisted_automorphism(a, b), (
                f"ρ-twisted automorphism fails on a={a}, b={b}"
            )


def test_rho_is_twisted_automorphism_flavoured():
    A = QuantumTorusKAlg([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])
    labels = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)]
    for a in labels:
        for b in labels:
            assert A.verify_rho_is_twisted_automorphism(a, b), (
                f"ρ-twisted automorphism fails on a={a}, b={b}"
            )


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------


def test_trace_zero_label_is_qpoch_prefactor():
    """Tr(L_0) = (q²; q²)_∞^{gauge_rank}."""
    A = QuantumTorusKAlg([[0, 1], [-1, 0]])
    R = A.coefficient_ring()
    tr = A.trace((0, 0), K=6)
    assert tr[0] == R.one()
    assert tr[1].is_zero()


def test_trace_nonzero_gauge_label_is_zero():
    """Labels with non-trivial gauge class trace to zero."""
    A = QuantumTorusKAlg([[0, 1], [-1, 0]])
    tr = A.trace((1, 0), K=4)
    assert tr.is_zero()


def test_trace_central_flavour_carries_mu_hexagon():
    """`Tr(L_{(1,1,1)}) = μ · (q²;q²)_∞^{2}`: the central flavour direction."""
    A = QuantumTorusKAlg([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])
    R = A.coefficient_ring()
    tr = A.trace((1, 1, 1), K=4)
    # q^0 coefficient should be μ^(1,) · 1 = μ.
    assert tr[0] == R.basis_element((1,))


def test_trace_non_central_label_in_kernel_orbit_zero_gauge():
    """`Tr(L_{(0,0,0)}) = (q²;q²)_∞^{2}` with no μ shift."""
    A = QuantumTorusKAlg([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])
    R = A.coefficient_ring()
    tr = A.trace((0, 0, 0), K=4)
    assert tr[0] == R.one()


# ---------------------------------------------------------------------------
# Orthonormality (only between distinct gauge classes)
# ---------------------------------------------------------------------------


def test_orthonormality_unflavoured():
    A = QuantumTorusKAlg([[0, 1], [-1, 0]])
    labels = [(0, 0), (1, 0), (0, 1), (1, 1), (-1, 2)]
    for a in labels:
        for b in labels:
            assert A.verify_orthonormality(a, b, K=4), (
                f"orthonormality fails on a={a}, b={b}"
            )


def test_orthonormality_flavoured_distinct_gauge_classes():
    """Hexagon: orthonormality holds between labels in distinct gauge
    classes.  (Two labels in the same Γ_f-coset are different basis
    elements but trace-paired by a μ-monomial -- not orthonormal in the
    strict δ_{a,b} sense, by design.)"""
    A = QuantumTorusKAlg([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])
    labels = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    # All these have distinct gauge classes (verified below).
    seen_classes = {A.gauge_class(l) for l in labels}
    assert len(seen_classes) == len(labels)
    for a in labels:
        for b in labels:
            assert A.verify_orthonormality(a, b, K=4), (
                f"orthonormality fails on a={a}, b={b}"
            )


# ---------------------------------------------------------------------------
# Bar involution
# ---------------------------------------------------------------------------


def test_bar_involution_unflavoured():
    A = QuantumTorusKAlg([[0, 1], [-1, 0]])
    labels = [(0, 0), (1, 0), (0, 1), (1, 1), (-1, 2)]
    for a in labels:
        for b in labels:
            assert A.verify_bar_involution(a, b), (
                f"bar fails on a={a}, b={b}"
            )


def test_bar_involution_flavoured_hexagon():
    A = QuantumTorusKAlg([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])
    labels = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    for a in labels:
        for b in labels:
            assert A.verify_bar_involution(a, b), (
                f"bar fails on a={a}, b={b}"
            )


# ---------------------------------------------------------------------------
# Flavour accessors
# ---------------------------------------------------------------------------


def test_flavour_generator_label_hexagon():
    A = QuantumTorusKAlg([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])
    # γ_f at flavour exponent (1,) is the first kernel basis vector.
    assert A.flavour_generator_label((1,)) == (1, 1, 1)
    assert A.flavour_generator_label((2,)) == (2, 2, 2)
    assert A.flavour_generator_label((0,)) == (0, 0, 0)


def test_gauge_class_and_flavour_part_decompose():
    """`γ` reconstructs from `gauge_class(γ)` and `flavour_part(γ)` via the SNF basis."""
    A = QuantumTorusKAlg([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])
    sec = A.section_basis
    ker = A.kernel_basis
    for gamma in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 1), (2, 1, 0), (-3, 0, 5)]:
        gc = A.gauge_class(gamma)
        fp = A.flavour_part(gamma)
        recon = [0] * 3
        for i, c in enumerate(gc):
            for k in range(3):
                recon[k] += c * sec[i][k]
        for i, c in enumerate(fp):
            for k in range(3):
                recon[k] += c * ker[i][k]
        assert tuple(recon) == gamma, (
            f"reconstruction failed for {gamma}: got {tuple(recon)}"
        )


def test_flavour_generator_zero_for_unflavoured():
    A = QuantumTorusKAlg([[0, 1], [-1, 0]])
    # No flavour -- only the empty-tuple input is allowed and it gives 0 vector.
    assert A.flavour_generator_label(()) == (0, 0)


# ---------------------------------------------------------------------------
# Flavour-lift coordinate (r_label_decompose / r_label_compose), embed_R,
# and forget = QuantumTorusKAlg(B_g)   (Γ_g as section-of-projection)
# ---------------------------------------------------------------------------

import itertools  # noqa: E402

_LIFT_PAIRINGS = [
    [[0, 1, 0], [-1, 0, 0], [0, 0, 0]],     # gauge 2 + flavour 1
    [[0, 1, -1], [-1, 0, 1], [1, -1, 0]],   # hexagon pairing: gauge 2 + flav 1
    [[0, 0], [0, 0]],                       # pure flavour (gauge 0)
    [[0, 1], [-1, 0]],                      # non-degenerate (flavour 0)
]


def _cube(A, lo=-1, hi=2):
    return [t for t in itertools.product(range(lo, hi), repeat=A.rank)]


def test_r_label_decompose_compose_roundtrip():
    for B in _LIFT_PAIRINGS:
        A = QuantumTorusKAlg(B)
        for g in _cube(A):
            sec, flav = A.r_label_decompose(g)
            assert A.r_label_compose(sec, flav) == tuple(g), (B, g)


def test_r_label_section_is_a_section_of_the_projection():
    """The section label lies in span(sec_basis) — a copy of Γ_g inside Γ —
    so its own flavour part is zero, and it projects to the same gauge class
    as the original label (π ∘ s = id)."""
    for B in _LIFT_PAIRINGS:
        A = QuantumTorusKAlg(B)
        zero_flav = tuple([0] * A.flavour_rank)
        for g in _cube(A):
            sec, _ = A.r_label_decompose(g)
            assert A.flavour_part(sec) == zero_flav, (B, g, sec)
            assert A.gauge_class(sec) == A.gauge_class(g), (B, g, sec)


def test_embed_R_faithfulness_and_rho_intertwine():
    """embed_R was missing (default only embeds 1_R); with the override the
    two faithfulness axioms hold for every label / character."""
    for B in _LIFT_PAIRINGS:
        A = QuantumTorusKAlg(B)
        R = A.coefficient_ring()
        for g in _cube(A):
            assert A.verify_embed_section_roundtrip(g), (B, g)
            assert A.verify_section_is_single_irrep(g), (B, g)
        # rho-intertwine on a spread of characters
        for f in itertools.product(range(-1, 2), repeat=A.flavour_rank):
            assert A.verify_embed_intertwines_rho(R.basis_element(f)), (B, f)


def test_from_R_form_roundtrip():
    """from_R_form ∘ to_R_form == id — works now that embed_R is implemented."""
    for B in _LIFT_PAIRINGS:
        A = QuantumTorusKAlg(B)
        for g in _cube(A):
            x = Element.basis(g)
            assert A.from_R_form(A.to_R_form(x)) == x, (B, g)


def test_label_section_decompose_bridges_to_r_label_decompose():
    """The retired _label_section_decompose is reconstructed by the base
    forward bridge: (section, μ^flav) with the SAME Γ-lift section."""
    for B in _LIFT_PAIRINGS:
        A = QuantumTorusKAlg(B)
        R = A.coefficient_ring()
        for g in _cube(A):
            sec_r, flav = A.r_label_decompose(g)
            sec_l, r_coef = A._label_section_decompose(g)
            assert sec_l == sec_r
            assert r_coef == R.basis_element(flav)


def test_forget_is_quantum_torus_on_gamma_g():
    for B in _LIFT_PAIRINGS:
        A = QuantumTorusKAlg(B)
        F = A.forget()
        assert isinstance(F, QuantumTorusKAlg)
        if A.flavour_rank == 0:
            assert F is A                     # already unflavoured
            continue
        assert F.flavour_rank == 0            # Γ_g is now abstract, non-degenerate
        assert F.rank == A.gauge_rank
        assert F.gauge_rank == A.gauge_rank
        # B_g is the induced pairing on the section basis
        sb = A.section_basis
        for i in range(A.gauge_rank):
            for j in range(A.gauge_rank):
                assert F.pairing[i][j] == A._bracket(sb[i], sb[j])


def test_forget_is_trace_preserving_homomorphism():
    """forget: L_γ ↦ M_{gauge_class(γ)} is an algebra homomorphism onto
    QuantumTorusKAlg(B_g) that intertwines (ε ∘ trace) with the Γ_g trace."""
    for B in _LIFT_PAIRINGS:
        A = QuantumTorusKAlg(B)
        if A.flavour_rank == 0:
            continue
        F = A.forget()
        eps = A.coefficient_ring().augmentation()
        labels = _cube(A)
        # multiply-homomorphism: φ(L_a · L_b) == M_{φa} · M_{φb}
        for a, b in itertools.product(labels[:9], repeat=2):
            prod = A.multiply(a, b)
            phi_prod = {A.forget_label(g): lp for g, lp in prod.terms.items()}
            rhs = F.multiply(A.forget_label(a), A.forget_label(b))
            assert phi_prod == dict(rhs.terms), (B, a, b)
        # trace: ε(Tr_A(L_a)) == Tr_F(M_{φa})
        for a in labels:
            tA = eps.apply_RPowerSeries(A.trace(a, 6))
            tF = F.trace(A.forget_label(a), 6)
            za = {q: c.terms.get((), 0) for q, c in tA.coeffs.items()
                  if c.terms.get((), 0)}
            zf = {q: c.terms.get((), 0) for q, c in tF.coeffs.items()
                  if c.terms.get((), 0)}
            assert za == zf, (B, a, za, zf)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback
    failures = 0
    for name in sorted(globals()):
        fn = globals()[name]
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  PASS: {name}")
        except Exception:
            failures += 1
            print(f"  FAIL: {name}")
            traceback.print_exc()
    if failures:
        print(f"\n{failures} failure(s).")
        sys.exit(1)
    print(f"\nAll QuantumTorusKAlg tests passed.")
