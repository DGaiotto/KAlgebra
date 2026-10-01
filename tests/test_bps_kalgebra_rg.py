"""Tests for `BPSKAlgebra` as `RGKAlgebra`.

The BPS realisation `BPSKAlgebra` inherits from `RGKAlgebra` and
exposes the canonical-basis morphism into the quantum torus as a
first-class RG flow:

    auxiliary()    = QuantumTorusKAlg(pairing)
    RG(L_a)        = F_a   (lifted into the auxiliary's Element type)
    rg_generator(K) = S    (Habiro form, leading q-order ≤ K)

These tests cover Tasks 1, 2, 3 of the design record.  Task 1 covers
inheritance + `auxiliary()` + `RG(a)` and the three "non-twist"
verifiers (unital, multiplicative, bar-invariant).  The
truncation-sensitive verifiers (`verify_rg_twist`,
`verify_rg_inner_product`) need `rg_generator` and will land with
Task 2.

Run:  PYTHONPATH=. python restructuring/tests/test_bps_kalgebra_rg.py
"""

from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
_RESTRUCT = os.path.dirname(_HERE)
_REPO = _RESTRUCT  # post-flatten: _RESTRUCT is the repo root
sys.path.insert(0, _REPO)
sys.path.insert(0, _RESTRUCT)

from bps_kalgebra import BPSKAlgebra
from rgkalgebra import RGKAlgebra
from quantum_torus_kalgebra import QuantumTorusKAlg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pentagon():
    return BPSKAlgebra(
        pairing=[[0, 1], [-1, 0]],
        node_charges=[(1, 0), (0, 1)],
    )


# ---------------------------------------------------------------------------
# Task 1 — inheritance + auxiliary + RG(a)
# ---------------------------------------------------------------------------


def test_bpskalgebra_is_rgkalgebra():
    """Inheritance flip: BPSKAlgebra IS-A RGKAlgebra."""
    A = _pentagon()
    assert isinstance(A, RGKAlgebra), (
        f"expected BPSKAlgebra to inherit from RGKAlgebra; "
        f"got MRO {type(A).__mro__}"
    )


def test_auxiliary_is_quantum_torus():
    """auxiliary() returns a QuantumTorusKAlg over the same pairing."""
    A = _pentagon()
    aux = A.auxiliary()
    assert isinstance(aux, QuantumTorusKAlg), (
        f"auxiliary() should return QuantumTorusKAlg; got {type(aux)}"
    )
    assert aux.pairing == [list(row) for row in A.lattice.pairing], (
        f"auxiliary pairing mismatch: aux={aux.pairing} "
        f"vs A.lattice.pairing={A.lattice.pairing}"
    )


def test_auxiliary_is_cached():
    """D4: auxiliary() returns the SAME instance every call (Python `is`)."""
    A = _pentagon()
    aux1 = A.auxiliary()
    aux2 = A.auxiliary()
    assert aux1 is aux2, (
        "auxiliary() must return the cached instance every call "
        "(D4 in the design record)"
    )


def test_auxiliary_coefficient_ring_matches():
    """The auxiliary's R must equal self's R (asserted at construction)."""
    A = _pentagon()
    assert A.auxiliary().coefficient_ring() == A.coefficient_ring()


def test_starting_algebra_default_is_self():
    """starting_algebra() defaults to self; RGKAlgebra ABC default."""
    A = _pentagon()
    assert A.starting_algebra() is A


# ---------------------------------------------------------------------------
# RG(a) — F lifted into the auxiliary's Element type
# ---------------------------------------------------------------------------


def test_RG_identity_is_X_zero():
    """RG(identity) = X_0 = the basis element of the auxiliary at 0."""
    from kalgebra import Element
    A = _pentagon()
    aux = A.auxiliary()
    R_aux = aux.coefficient_ring()
    rg_e = A.RG(A.identity())
    expected = Element.basis(A.identity())
    assert rg_e == expected, (
        f"RG(identity()) should be the basis element at zero; "
        f"got {rg_e}, expected {expected}"
    )


def test_RG_matches_F_on_a_window():
    """RG(γ).terms[label] q-coefficients match A.F(γ)[label]
    coefficients. After the design record, both Element and F use the same
    LaurentPoly type (Z[q^±]), so the comparison is direct."""
    A = _pentagon()
    labels = [(0, 0), (1, 0), (0, 1), (1, 1), (-1, 0), (0, -1)]
    for gamma in labels:
        rg_e = A.RG(gamma)
        F_gamma = A.F(gamma)
        # Same set of labels.
        assert set(rg_e.terms.keys()) == set(F_gamma.keys()), (
            f"RG({gamma}) labels {set(rg_e.terms.keys())} != "
            f"F({gamma}) labels {set(F_gamma.keys())}"
        )
        # Same q-coefficients at each label (both LaurentPoly).
        for label, lp in F_gamma.items():
            rg_lp = rg_e.terms[label]
            assert dict(rg_lp._coeffs) == dict(lp._coeffs), (
                f"RG({gamma})[{label}]={dict(rg_lp._coeffs)} != "
                f"F({gamma})[{label}]={dict(lp._coeffs)}"
            )


# ---------------------------------------------------------------------------
# Three "non-twist" RG verifiers (don't need rg_generator)
# ---------------------------------------------------------------------------


def test_verify_rg_unital_pentagon():
    """RG(1_self) == 1_aux."""
    A = _pentagon()
    assert A.verify_rg_unital()


def test_verify_rg_multiplicative_pentagon():
    """RG(L_a · L_b) == RG(a) · RG(b) on a small label window.

    For the pentagon, F_a · F_b decomposes back into F-basis with
    integer coefficients, and the same product in the auxiliary
    quantum torus must match (multiplicativity of the canonical-basis
    embedding).
    """
    A = _pentagon()
    labels = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1)]
    for a in labels:
        for b in labels:
            assert A.verify_rg_multiplicative(a, b), (
                f"RG multiplicativity fails on {a} · {b}"
            )


def test_verify_rg_bar_invariant_pentagon():
    """bar(RG(L_a)) == RG(L_a) — F_a's coefficients are q-palindromic."""
    A = _pentagon()
    labels = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, -1)]
    for a in labels:
        assert A.verify_rg_bar_invariant(a), (
            f"RG bar-invariance fails on {a}: RG(a)={A.RG(a)}, "
            f"bar={A.RG(a).bar()}"
        )


# ---------------------------------------------------------------------------
# Task 2 — rg_generator(K) Habiro form
# ---------------------------------------------------------------------------


def test_rg_generator_identity_is_one():
    """The identity term s_0 = 1 is always in the result, regardless of K."""
    from habiro import HabiroElement
    A = _pentagon()
    rg = A.rg_generator(0)
    assert A.identity() in rg, (
        f"identity {A.identity()} missing from rg_generator(0); "
        f"got keys {list(rg.keys())}"
    )
    # The identity Habiro element should be 1.
    assert (rg[A.identity()] - HabiroElement.one()).is_zero(), (
        f"rg_generator's identity term should be HabiroElement.one(); "
        f"got {rg[A.identity()]}"
    )


def test_rg_generator_pentagon_closed_form():
    """For the pentagon, s_γ = (-1)^{a+b} q^{a+b+ab} / ((q²;q²)_a (q²;q²)_b)
    for γ = (a, b), a, b ≥ 0 (single Nahm tuple, since N = rank).
    rg_generator(K) should include exactly those γ's with a+b+ab ≤ K
    (modulo over-inclusion the contract permits)."""
    A = _pentagon()
    K = 4
    rg = A.rg_generator(K)
    # Every γ = (a, b) with a, b ≥ 0 and a+b+ab ≤ K must be in rg.
    expected_required: set = set()
    for a in range(K + 1):
        for b in range(K + 1):
            if a + b + a * b <= K:
                expected_required.add((a, b))
    for gamma in expected_required:
        assert gamma in rg, (
            f"rg_generator({K}) missing γ={gamma} (a+b+ab="
            f"{gamma[0] + gamma[1] + gamma[0] * gamma[1]} ≤ {K})"
        )
    # Every γ in rg must have a, b ≥ 0 (the cone) — no negative
    # charges in the spectrum generator.
    for gamma in rg:
        a, b = gamma
        assert a >= 0 and b >= 0, (
            f"rg_generator returned negative-charge γ={gamma}"
        )


def test_rg_generator_su2_canonical_spec():
    """Canonical pure SU(2) (= A_1 in pure_ade_lattice's SC convention):
    pairing is the standard symplectic [[0,1],[-1,0]], and the spec is
    the bipartite Coxeter sequence — composite charges are normal.
    Per the design notes, the SC SU(2) gives spec = [(1, 0), (-1, 2)].

    rg_generator should include the identity, both spec charges
    (each with shift 1), and (0, 2) which is the leading non-trivial
    γ in the cone with shift 4."""
    from pure_ade_lattice import pure_ade_kalgebra
    A = pure_ade_kalgebra([("A", 1)])
    assert A is not None, "pure SU(2) should produce a non-None BPSKAlgebra"
    K = 4
    rg = A.rg_generator(K)
    assert A.identity() in rg, "identity must be in rg_generator output"
    # The canonical SC SU(2) spec is [(1, 0), (-1, 2)].
    for spec_g in A.spec:
        assert spec_g in rg, (
            f"spec charge {spec_g} should appear in rg_generator({K}) "
            f"since its single-tuple shift is 1 ≤ {K}"
        )


def test_rg_generator_agrees_with_spectrum_generator_pentagon():
    """rg_generator(K) and spectrum_generator(K) describe the same S
    truncated to q^K. After expanding each rg_generator HabiroElement
    to q^K, the two dicts should agree on every shared γ."""
    from laurent_poly import LaurentPoly as QTLP
    A = _pentagon()
    K = 4
    rg = A.rg_generator(K)
    sg = A.spectrum_generator(K)
    # Every γ in spectrum_generator(K) (the truncated S) must appear in
    # rg_generator(K) — rg_generator may over-include but never miss.
    for gamma, sg_lp in sg.items():
        assert gamma in rg, (
            f"rg_generator({K}) missing γ={gamma} that appears in "
            f"spectrum_generator({K})"
        )
        rg_expanded = rg[gamma].expand(K)
        rg_lp = QTLP({e: c for e, c in rg_expanded._coeffs.items() if e <= K})
        assert rg_lp == sg_lp, (
            f"rg_generator({K})[{gamma}].expand({K}) = {rg_lp} != "
            f"spectrum_generator({K})[{gamma}] = {sg_lp}"
        )


# ---------------------------------------------------------------------------
# verify_rg_twist and verify_rg_inner_product are *not* tested here
# with strict equality.
#
# Both verifiers fail at any finite truncation due to a boundary
# residual: `S_RG` is shift-truncated, but the multiplication
# `X_γ · S_RG` (and `S_RG · X_{-γ}`) q-shifts terms by `<γ, ·>`
# (which can be negative), so contributions at high q-order from
# γ's just outside the truncation window land at *low* q-orders
# of the product. For verify_rg_twist this produces a label-
# `(1, cutoff+1)` disagreement at q^0; for verify_rg_inner_product
# the trace at the end accumulates all labels' (a_S)·(b_S)
# contributions, so the residual at the spurious label adds a
# constant `+1` to the trace's q^0.
#
# Increasing cutoff just moves the spurious label outwards
# without removing the residual.
#
# The existing `restructuring/the suite in the source repository (lines
# 490-497) documents this same phenomenon and explicitly defers
# to "a follow-up boundary-aware verifier matching coefficients
# on `n_j ≤ cutoff − safety` and `q ≤ K_expand − safety`". Plan
# 01 follows that lead and does not add strict-equality tests
# here. The `BPSKAlgebra.inner_product(a, b, K)` direct path
# (covered by `test_bps_kalgebra.py` orthonormality tests) gives
# the BPS-side answer with no truncation residual.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Task 3 + 5 — verify_canonical_basis(K) on every small theory shipped
#
# `verify_canonical_basis` bundles the four currently-meaningful
# canonical-basis checks: verify_rg_unital, verify_rg_multiplicative,
# verify_rg_bar_invariant, verify_orthonormality.  We run it across
# the standard small theories of the canonical surface and assert
# all four are green.
# ---------------------------------------------------------------------------


def _assert_canonical_basis_passes(A, name, K=4):
    """Helper: run verify_canonical_basis(K) on `A` and assert all keys
    are True.  On failure, raise with the failing-key list."""
    results = A.verify_canonical_basis(K=K)
    failing = [k for k, v in results.items() if v is not True]
    assert not failing, (
        f"{name}: verify_canonical_basis(K={K}) failed on "
        f"{failing}; full result = {results}"
    )


def test_verify_canonical_basis_pentagon():
    A = _pentagon()
    _assert_canonical_basis_passes(A, "pentagon", K=4)


def test_verify_canonical_basis_pure_su2():
    """Canonical pure SU(2) (= A_1) via pure_ade_kalgebra."""
    from pure_ade_lattice import pure_ade_kalgebra
    A = pure_ade_kalgebra([("A", 1)])
    assert A is not None
    _assert_canonical_basis_passes(A, "pure SU(2)", K=4)


def test_verify_canonical_basis_pure_su3():
    """Canonical pure SU(3) (= A_2) via pure_ade_kalgebra.

    Heavier than pentagon / SU(2) (rank 4, |spec|=6); restrict to
    K=2 for runtime sanity.  The rg side checks (unital, mult,
    bar) are q-independent / constant cost; only orthonormality
    scales with K.
    """
    from pure_ade_lattice import pure_ade_kalgebra
    A = pure_ade_kalgebra([("A", 2)])
    assert A is not None
    _assert_canonical_basis_passes(A, "pure SU(3)", K=2)


def test_verify_canonical_basis_hexagon():
    """Hexagon (= [A_1, A_3] in the Â_3 chamber, flavoured rank-3,
    `det(B) = 0`).

    Pre-Plan-10: `verify_rg_multiplicative` and `verify_orthonormality`
    used to fail because `BPSKAlgebra.multiply` produced section-rep
    + μ-coefficients while `QuantumTorusKAlg.multiply` produced
    full-Γ-tuple labels. After the design record's Z-form contract refactor
    both algebras emit `Element` over Z[q^±] with full-Γ-tuple
    labels, and the verifier suite passes cleanly on flavoured
    theories.
    """
    A = BPSKAlgebra(
        pairing=[[0, 1, -1], [-1, 0, 1], [1, -1, 0]],
        node_charges=[(1, 0, 0), (0, 1, 0), (0, 0, 1)],
    )
    _assert_canonical_basis_passes(A, "hexagon", K=2)


def test_verify_canonical_basis_results_dict_shape():
    """The shorthand returns a dict with exactly the four expected
    keys, each a bool."""
    A = _pentagon()
    results = A.verify_canonical_basis(K=2)
    assert set(results.keys()) == {
        "unital", "multiplicative", "bar_invariant", "orthonormality",
    }, (
        f"verify_canonical_basis returned keys {set(results.keys())}; "
        f"expected {{unital, multiplicative, bar_invariant, "
        f"orthonormality}}"
    )
    for k, v in results.items():
        assert isinstance(v, bool), (
            f"verify_canonical_basis['{k}'] should be bool, got {type(v)}"
        )


def test_verify_canonical_basis_default_window_includes_identity_and_nodes():
    """Sanity: the default label window covers identity + node charges."""
    A = _pentagon()
    window = A._default_verifier_window()
    assert A.identity() in window, "default window must include identity"
    for g in A.node_charges:
        assert tuple(g) in window, (
            f"default window must include node charge {g}"
        )


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
    print(f"\nAll BPSKAlgebra-RG tests passed.")
