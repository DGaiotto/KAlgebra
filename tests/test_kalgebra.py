"""Tests for the `KAlgebra` ABC and the four hard-coded samples.

After parameterizing KAlgebra over a Z₊-ring R (via `RLaurent` / `RPowerSeries`),
all four existing samples sit at `R = TrivialZPlusRing()` (the unflavoured
case).  The ρ-twist axiom collapses to the strict-automorphism axiom there
(since `⋆ = id` on Z).

Run:  PYTHONPATH=. python restructuring/tests/test_kalgebra.py
"""

from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
_RESTRUCT = os.path.dirname(_HERE)             # restructuring/
_REPO = _RESTRUCT  # post-flatten: _RESTRUCT is the repo root             # repo root
sys.path.insert(0, _REPO)
sys.path.insert(0, _RESTRUCT)

from zplus_ring import (
    ZPlusRing, RElement, RLaurent, RPowerSeries,
    TrivialZPlusRing, AbelianZPlusRing,
)
from kalgebra import KAlgebra, Element
from laurent_poly import LaurentPoly
from kalgebra_samples import (
    TrivialKAlg, QuantumTorusZ2KAlg, Sqed1KAlg, PentagonKAlg,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _verify_rho_inverse_all(A, labels):
    return all(A.verify_rho_inverse(a) for a in labels)


def _verify_axioms_pairwise(A, labels, K=5):
    """Helper: run bar / orthonormality / ρ²-twisted-trace on all pairs."""
    for a in labels:
        for b in labels:
            assert A.verify_bar_involution(a, b), f"bar fails on {a}, {b}"
            assert A.verify_orthonormality(a, b, K=K), f"ortho fails on {a}, {b}"
            assert A.verify_rho_twisted_trace(a, b, K=K), \
                f"ρ²-twisted trace fails on {a}, {b}"


def _qpoch_squared(R: ZPlusRing, K: int) -> RPowerSeries:
    """`(q²;q²)_∞²` truncated to q^K, as RPowerSeries[R].  Used to spot-check
    the QTZ2 trace closed form without a `qpoch_infty` import."""
    pref = RPowerSeries.one(R, K)
    k = 1
    while 2 * k <= K:
        factor = RPowerSeries(R, {0: 1, 2 * k: -1}, K)
        pref = pref * factor
        k += 1
    return pref * pref


# ---------------------------------------------------------------------------
# Trivial sample
# ---------------------------------------------------------------------------


def test_trivial_axioms():
    A = TrivialKAlg()
    assert A.verify_identity_in_basis()
    assert A.verify_rho_fixes_identity()
    labels = [()]
    assert _verify_rho_inverse_all(A, labels)
    _verify_axioms_pairwise(A, labels)
    # Tr(1) = 1 exactly
    assert A.trace((), K=5)[0] == 1
    for l in range(1, 6):
        assert A.trace((), K=5)[l] == 0


def test_trivial_coefficient_ring():
    A = TrivialKAlg()
    assert isinstance(A.coefficient_ring(), TrivialZPlusRing)


# ---------------------------------------------------------------------------
# Quantum torus Z² sample
# ---------------------------------------------------------------------------


def test_qtorus_axioms():
    A = QuantumTorusZ2KAlg()
    labels = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1)]
    assert A.verify_identity_in_basis()
    assert A.verify_rho_fixes_identity()
    assert _verify_rho_inverse_all(A, labels)
    _verify_axioms_pairwise(A, labels)


def test_qtorus_trace_closed_form():
    A = QuantumTorusZ2KAlg()
    R = A.coefficient_ring()
    K = 8
    # Tr(X_0) = (q²;q²)_∞²
    expected = _qpoch_squared(R, K)
    got = A.trace((0, 0), K)
    assert (got - expected).is_zero(), f"Tr(X_0) mismatch: {got} vs {expected}"
    # Tr(X_γ) = 0 for γ ≠ 0
    for g in [(1, 0), (0, 1), (1, 1), (-1, 0), (-2, 3)]:
        assert A.trace(g, K).is_zero(), f"Tr(X_{g}) should be 0"


def test_qtorus_multiply_value():
    """Spot check the actual product."""
    A = QuantumTorusZ2KAlg()
    prod = A.multiply((1, 0), (0, 1))
    expected = Element({(1, 1): LaurentPoly({1: 1})})
    assert prod == expected


# ---------------------------------------------------------------------------
# SQED_1 sample
# ---------------------------------------------------------------------------


def test_sqed1_relations():
    """The four SQED_1 relations must be reproduced by `multiply`."""
    A = Sqed1KAlg()
    # u_+ u_- = 1 + q v
    assert A.multiply((1, 0), (-1, 0)) == Element({
        (0, 0): LaurentPoly({0: 1}),
        (0, 1): LaurentPoly({1: 1}),
    })
    # u_- u_+ = 1 + q^{-1} v
    assert A.multiply((-1, 0), (1, 0)) == Element({
        (0, 0): LaurentPoly({0: 1}),
        (0, 1): LaurentPoly({-1: 1}),
    })
    # u_+ v = q · L_{1, 1}     (since L_{1,1} = q^{-1} u_+ v)
    assert A.multiply((1, 0), (0, 1)) == Element({
        (1, 1): LaurentPoly({1: 1}),
    })
    # v u_+ = q^{-1} · L_{1, 1}
    assert A.multiply((0, 1), (1, 0)) == Element({
        (1, 1): LaurentPoly({-1: 1}),
    })


def test_sqed1_axioms():
    A = Sqed1KAlg()
    labels = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
              (1, 1), (-1, -1), (1, -1), (-1, 1), (2, 0), (-2, 0)]
    assert A.verify_identity_in_basis()
    assert A.verify_rho_fixes_identity()
    assert _verify_rho_inverse_all(A, labels)
    for a in labels:
        for b in labels:
            assert A.verify_bar_involution(a, b), \
                f"SQED_1 bar fails on {a}, {b}"
            assert A.verify_orthonormality(a, b, K=5), \
                f"SQED_1 ortho fails on {a}, {b}: I = {A.inner_product(a, b, 5)}"


def test_sqed1_rho_squared():
    """ρ²(m, n) = (m, n + m), per the derived formula."""
    A = Sqed1KAlg()
    for m in range(-2, 3):
        for n in range(-2, 3):
            r2 = A.rho(A.rho((m, n)))
            assert r2 == (m, n + m), f"ρ²({m, n}) = {r2}, expected {(m, n + m)}"


def test_sqed1_trace_vacuum_value():
    """Tr(1) for SQED_1 has [q⁰] = 1 (orthonormality) and is non-trivial."""
    A = Sqed1KAlg()
    K = 8
    Tr1 = A.trace((0, 0), K)
    assert Tr1[0] == 1, f"Tr(1)[q^0] = {Tr1[0]}, expected 1"


# ---------------------------------------------------------------------------
# Pentagon sample
# ---------------------------------------------------------------------------


def test_pentagon_basic_axioms():
    A = PentagonKAlg()
    assert A.verify_identity_in_basis()
    assert A.verify_rho_fixes_identity()
    gens = [(i, 1, 0) for i in range(5)] + [(0, 0, 0)]
    assert _verify_rho_inverse_all(A, gens)
    for a in gens:
        for b in gens:
            assert A.verify_bar_involution(a, b), \
                f"pentagon bar fails on {a}, {b}"


def test_pentagon_trace_seeds_and_rho_invariance():
    """Tr(1)[q^0] = 1 and Tr(L_i) is i-independent (ρ-invariance)."""
    A = PentagonKAlg()
    K = 6
    Tr1 = A.trace((0, 0, 0), K)
    Tr_L0 = A.trace((0, 1, 0), K)
    assert Tr1[0] == 1, f"Tr(1)[q^0] = {Tr1[0]}, expected 1"
    # ρ-invariance: Tr(L_i) = Tr(L_0) for all i mod 5
    for i in range(5):
        Tr_Li = A.trace((i, 1, 0), K)
        assert (Tr_Li - Tr_L0).is_zero(), \
            f"Tr(L_{i}) != Tr(L_0): {Tr_Li} vs {Tr_L0}"


def test_pentagon_orthonormality_window():
    """Orthonormality I_{a,b}[q^0] = δ on a basis-degree-2 window."""
    A = PentagonKAlg()
    K = 5
    labels = [(0, 0, 0)] + [(i, 1, 0) for i in range(5)] + [(0, 1, 1), (0, 2, 0)]
    for a in labels:
        for b in labels:
            assert A.verify_orthonormality(a, b, K=K), \
                f"pentagon ortho fails: a={a}, b={b}, I={A.inner_product(a, b, K)}"


def test_pentagon_trace_satisfies_cyclicity():
    """The hard-coded Nahm-sum trace must satisfy cyclicity Tr(uv) = Tr(ρ²(v) u)."""
    A = PentagonKAlg()
    K = 5
    labels = [
        (0, 0, 0),
        (0, 1, 0), (1, 1, 0), (2, 1, 0), (3, 1, 0), (4, 1, 0),
        (0, 1, 1), (0, 2, 0),
    ]
    for u in labels:
        for v in labels:
            assert A.verify_rho_twisted_trace(u, v, K=K), (
                f"cyclicity fails for u={u}, v={v}"
            )


def test_pentagon_trace_at_q0_normalization():
    """Tr(1)[q^0] = 1 (orthonormality of the identity)."""
    A = PentagonKAlg()
    Tr1 = A.trace((0, 0, 0), K=5)
    assert Tr1[0] == 1, f"PentagonKAlg Tr(1)[q^0] = {Tr1[0]}, expected 1"


# ---------------------------------------------------------------------------
# Non-trivial-R sample (was in test_fkalgebra.py): MuKAlg.
#
# Single-label KAlgebra over R = R(U(1)) = Z[μ^±].  Tests that ρ's
# twisted-linear extension actually applies ⋆ on R-coefficients.
# ---------------------------------------------------------------------------


class MuKAlg(KAlgebra):
    """Single-label KAlgebra over R = Z[μ^±].  Single basis element ()
    (the identity).  After the design record, Element is over Z[q^±], so the
    R-module structure surfaces only at the trace / to_R_form layer.
    """

    _R = AbelianZPlusRing(rank=1)

    def coefficient_ring(self) -> ZPlusRing:
        return self._R

    def identity(self):
        return ()

    def multiply(self, a, b):
        return Element({(): LaurentPoly({0: 1})})

    def rho(self, a):
        return ()

    def rho_inverse(self, a):
        return ()

    def trace(self, a, K=20):
        return RPowerSeries(self._R, {0: 1}, K)

    def _label_section_decompose(self, label):
        return label, self.coefficient_ring().one()


def test_mu_kalg_axioms():
    A = MuKAlg()
    assert A.verify_identity_in_basis()
    assert A.verify_rho_fixes_identity()
    assert A.verify_rho_inverse(())
    assert A.verify_orthonormality((), ())
    assert A.verify_bar_involution((), ())
    assert A.verify_rho_is_automorphism((), ())


def test_element_bar_acts_only_on_q():
    """Element.bar() acts only on q (Z[q^±] coefficients).  After
    the design record, Element is over Z[q^±], so there's no R-side to act on
    from inside Element."""
    elem = Element({("x",): LaurentPoly({1: 1, 3: 2})})
    barred = elem.bar()
    # q-flip: q^1 → q^{-1}, q^3 → q^{-3}, coefficients unchanged
    assert barred.terms[("x",)] == LaurentPoly({-1: 1, -3: 2})


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_associativity_and_unit_law_samples():
    """`verify_associativity` / `verify_unit_law` on the four samples — the
    associative-algebra and unit axioms, previously asserted in the class
    docstring but unchecked (the audit finding A7)."""
    cases = [
        (TrivialKAlg(), [()]),
        (QuantumTorusZ2KAlg(), [(0, 0), (1, 0), (0, 1), (1, 1), (-1, 0)]),
        (Sqed1KAlg(), [(0, 0), (1, 0), (-1, 0), (0, 1), (1, 1)]),
        (PentagonKAlg(),
         [(0, 0, 0), (0, 1, 0), (1, 1, 0), (0, 1, 1), (0, 2, 0)]),
    ]
    for A, labels in cases:
        name = type(A).__name__
        for a in labels:
            assert A.verify_unit_law(a), f"{name} unit law fails on {a}"
        for a in labels:
            for b in labels:
                for c in labels:
                    assert A.verify_associativity(a, b, c), \
                        f"{name} associativity fails on {a}, {b}, {c}"


# ---------------------------------------------------------------------------
# trace_element widening + bilinear inner_product_element
# (2026-07-01 A1D3 adjudication: the unwidened assembly silently corrupted
#  every order above q^{K - |valuation|} whenever an Element coefficient had
#  negative q-valuation — the source of the spurious mixed-tile
#  orthonormality "violations" in the external audit's §1.1.)
# ---------------------------------------------------------------------------


def test_trace_element_widens_negative_valuation():
    """`trace_element` on `q^{-n}·L_a` must equal the q^{-n}-shift of
    `trace(a, K+n)` truncated to K — i.e. the per-label trace request is
    widened by the coefficient's negative valuation.  Pentagon's
    `Tr L = -q - q^7 - q^9 - …` makes the test sensitive: at K=4 the
    `q^{-3}` coefficient must pick up the `-q^7` tail as `-q^4`."""
    A = PentagonKAlg()
    R = A.coefficient_ring()
    lab = (0, 1, 0)
    K, shift = 4, -3
    x = Element({lab: LaurentPoly({shift: 1})})
    got = A.trace_element(x, K)
    tr_wide = A.trace(lab, K - shift)
    expected = RPowerSeries(
        R,
        {e + shift: c for e, c in tr_wide.coeffs.items() if e + shift <= K},
        K,
    )
    assert got == expected, f"widened trace_element mismatch: {got} != {expected}"
    # Sensitivity guard: the widened window must actually contribute,
    # otherwise this test is vacuous.
    assert any(K < e <= K - shift for e in tr_wide.coeffs), \
        "test lost sensitivity: no trace tail in the widened window"
    # Mixed element: q^{-3}·L_lab + 1·L_e adds the plain trace on top.
    e_id = A.identity()
    x2 = Element({lab: LaurentPoly({shift: 1}), e_id: LaurentPoly({0: 1})})
    got2 = A.trace_element(x2, K)
    expected2 = expected + A.trace(e_id, K)
    assert got2 == expected2


def test_inner_product_element_bilinear():
    """`inner_product_element` = the bilinear extension of `I_{a,b}` over
    formal sums.  Checks (i) agreement with
    `inner_product` on basis Elements, (ii) genuine bilinearity with a
    negative-valuation scalar, widened per pair."""
    A = PentagonKAlg()
    R = A.coefficient_ring()
    K = 4
    a, b, c = (0, 1, 0), (0, 0, 0), (1, 1, 0)
    # (i) basis Elements reduce to the per-pair pairing.
    assert A.inner_product_element(
        Element.basis(a), Element.basis(c), K) == A.inner_product(a, c, K)
    # (ii) x = q^{-2}·L_a + L_b against y = L_c.
    x = Element({a: LaurentPoly({-2: 1}), b: LaurentPoly({0: 1})})
    got = A.inner_product_element(x, Element.basis(c), K)
    iac_wide = A.inner_product(a, c, K + 2)
    expected = RPowerSeries(
        R,
        {e - 2: v for e, v in iac_wide.coeffs.items() if e - 2 <= K},
        K,
    ) + A.inner_product(b, c, K)
    assert got == expected, f"bilinear pairing mismatch: {got} != {expected}"


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
    print(f"\nAll K-algebra ABC tests passed.")
