"""Test the explicit formulae in the LaTeX note for PentagonKAlg.

Verifies:
* `Tr(1) = χ₀(q²)` and `Tr(L_i) = q⁻¹ (χ₀(q²) - χ₁(q²))`
  match their leading expansions
      `Tr(1) = 1 + q^4 + q^6 + q^8 + q^{10} + ...`
      `Tr(L) = -q - q^7 - q^9 - q^{11} + ...`
* The Schur-like Layer 1 recursion gives the exact integer
  combinations from the note for `n = 2..6`:
      q²  Tr(L²) = Tr(1) + q⁻¹ Tr(L)              (i.e. Tr(L²) = q⁻²Tr(1) + q⁻³ Tr(L))
      q⁸  Tr(L³) = q Tr(1) + (1 + q⁴) Tr(L)
      q¹⁵ Tr(L⁴) = (q + q⁷) Tr(1) + (1 + q⁴ + q⁶) Tr(L)
      q²⁴ Tr(L⁵) = (q + q⁷ + q⁹) Tr(1) + (1 + q⁴ + q⁶ + q⁸ + q¹²) Tr(L)
      q³⁵ Tr(L⁶) = (q + q⁷ + q⁹ + q¹¹ + q¹⁷) Tr(1)
                  + (1 + q⁴ + q⁶ + q⁸ + q¹⁰ + q¹² + q¹⁴ + q¹⁶) Tr(L)
* The convergence claim: `Tr(L_i)/Tr(1) = -q + O(q⁴)` and more
  generally `q^{n²-1} Tr(L_i^n)` matches `-Tr(L) Tr(1) + Tr(1) Tr(L)`...
  i.e. coefficients of `q^{n²-1} Tr(L^n)` converge to `(-Tr(L_1), Tr(1))`
  up to order `O(q^{n²})`.
"""

from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from laurent_poly import LaurentPoly
from kalgebra_samples import PentagonKAlg, _pent_tr_power_coeffs


def _lp(coeffs: dict[int, int]) -> LaurentPoly:
    return LaurentPoly(coeffs)


def _z(re) -> int:
    """Extract the Z-coefficient of a TrivialZPlusRing RElement (basis = `()`)."""
    return int(re.terms.get((), 0))


def test_tr_1_leading_coefficients():
    A = PentagonKAlg()
    Tr1 = A.trace((0, 0, 0), K=20)
    # χ₀(q²) = 1 + q^4 + q^6 + q^8 + q^{10} + 2q^{12} + 2q^{14} + ...
    expected = {0: 1, 4: 1, 6: 1, 8: 1, 10: 1, 12: 2, 14: 2, 16: 3, 18: 3, 20: 4}
    for e, c in expected.items():
        assert _z(Tr1[e]) == c, (
            f"Tr(1)[q^{e}] = {Tr1[e]}, expected {c}"
        )
    # No other nonzero coefficients ≤ 20 outside `expected`.
    for e, rc in Tr1.coeffs.items():
        if e <= 20 and e not in expected:
            assert rc.is_zero(), f"unexpected Tr(1)[q^{e}] = {rc}"


def test_tr_L_leading_coefficients():
    A = PentagonKAlg()
    TrL = A.trace((0, 1, 0), K=20)
    # Tr(L) = -q - q^7 - q^9 - q^{11} - q^{13} - q^{15} - 2q^{17} - 2q^{19} - ...
    expected = {1: -1, 7: -1, 9: -1, 11: -1, 13: -1, 15: -1, 17: -2, 19: -2}
    for e, c in expected.items():
        assert _z(TrL[e]) == c, (
            f"Tr(L)[q^{e}] = {TrL[e]}, expected {c}"
        )
    for e, rc in TrL.coeffs.items():
        if e <= 19 and e not in expected:
            assert rc.is_zero(), f"unexpected Tr(L)[q^{e}] = {rc}"


def _expected_layer1(n: int) -> tuple[LaurentPoly, LaurentPoly]:
    """Hand-coded (c1, cL) from the LaTeX note for n=2..6."""
    if n == 2:
        # Tr(L²) = q⁻² Tr(1) + q⁻³ Tr(L)
        return _lp({-2: 1}), _lp({-3: 1})
    if n == 3:
        # q⁸ Tr(L³) = q Tr(1) + (1 + q⁴) Tr(L)
        return _lp({-7: 1}), _lp({-8: 1, -4: 1})
    if n == 4:
        # q¹⁵ Tr(L⁴) = (q + q⁷) Tr(1) + (1 + q⁴ + q⁶) Tr(L)
        return _lp({-14: 1, -8: 1}), _lp({-15: 1, -11: 1, -9: 1})
    if n == 5:
        # q²⁴ Tr(L⁵) = (q + q⁷ + q⁹) Tr(1) + (1 + q⁴ + q⁶ + q⁸ + q¹²) Tr(L)
        return (
            _lp({-23: 1, -17: 1, -15: 1}),
            _lp({-24: 1, -20: 1, -18: 1, -16: 1, -12: 1}),
        )
    if n == 6:
        # q³⁵ Tr(L⁶) = (q + q⁷ + q⁹ + q¹¹ + q¹⁷) Tr(1)
        #            + (1 + q⁴ + q⁶ + q⁸ + q¹⁰ + q¹² + q¹⁴ + q¹⁶) Tr(L)
        return (
            _lp({-34: 1, -28: 1, -26: 1, -24: 1, -18: 1}),
            _lp({-35: 1, -31: 1, -29: 1, -27: 1, -25: 1, -23: 1, -21: 1, -19: 1}),
        )
    raise ValueError(n)


def test_schur_recursion_against_note_n2_through_n6():
    for n in range(2, 7):
        c1, cL = _pent_tr_power_coeffs(n)
        e1, eL = _expected_layer1(n)
        assert (c1 - e1).is_zero(), (
            f"n={n}: c1 mismatch.\n  computed: {c1._coeffs}\n  expected: {e1._coeffs}"
        )
        assert (cL - eL).is_zero(), (
            f"n={n}: cL mismatch.\n  computed: {cL._coeffs}\n  expected: {eL._coeffs}"
        )


def test_tr_L_squared_starts_at_q2():
    """Tr(L_i²) = q⁻² Tr(1) + q⁻³ Tr(L_i) = O(q): the q⁻² terms cancel."""
    A = PentagonKAlg()
    Tr_L2 = A.trace((0, 2, 0), K=12)
    # Should start at q^2 (no negative q powers, no q^0 or q^1).
    for e in range(-3, 2):
        assert Tr_L2[e].is_zero(), f"Tr(L²)[q^{e}] should be 0, got {Tr_L2[e]}"
    # q² coefficient is +1.
    assert _z(Tr_L2[2]) == 1, (
        f"Tr(L²)[q^2] = {Tr_L2[2]}, expected 1"
    )


def test_tr_L_over_tr_1_leading_term():
    """The note's note: Tr(L_i)/Tr(1) = -q + O(q⁴).

    Since Tr(1) = 1 + O(q⁴), the leading term of Tr(L)/Tr(1) equals
    Tr(L) up to O(q⁴): -q + O(q⁴).  Verify directly from Tr(L)."""
    A = PentagonKAlg()
    TrL = A.trace((0, 1, 0), K=5)
    assert _z(TrL[1]) == -1
    for e in (0, 2, 3):
        assert TrL[e].is_zero(), f"Tr(L)[q^{e}] should be 0, got {TrL[e]}"


def test_layer1_coefficients_converge_to_traces():
    """The note's main convergence observation: in
    `q^{n²-1} Tr(L^n) = A_n(q) Tr(1) + B_n(q) Tr(L)`,
    the coefficients of A_n and B_n converge (as n grows) to those of
    -Tr(L) and Tr(1) respectively.

    Empirically: A_n[q^k] = -Tr(L)[q^k] and B_n[q^k] = Tr(1)[q^k]
    for k ≤ 2n-1.  Each recursion step extends agreement by two
    q-orders."""
    A = PentagonKAlg()
    K = 40
    Tr1 = A.trace((0, 0, 0), K=K)
    TrL = A.trace((0, 1, 0), K=K)
    for n in range(2, 9):
        c1, cL = _pent_tr_power_coeffs(n)
        shift = n * n - 1
        A_n = {e + shift: c for e, c in c1._coeffs.items()}
        B_n = {e + shift: c for e, c in cL._coeffs.items()}
        for k in range(0, 2 * n):  # k = 0..2n-1
            assert A_n.get(k, 0) == -_z(TrL[k]), (
                f"n={n}, k={k}: A_n[q^{k}] = {A_n.get(k, 0)}, "
                f"-Tr(L)[q^{k}] = {-_z(TrL[k])}"
            )
            assert B_n.get(k, 0) == _z(Tr1[k]), (
                f"n={n}, k={k}: B_n[q^{k}] = {B_n.get(k, 0)}, "
                f"Tr(1)[q^{k}] = {_z(Tr1[k])}"
            )


def test_tr_powers_have_no_negative_q_powers():
    """The note's observation: each Tr(L^n) is a power series in q, with
    no negative q-powers, even though Layer 1's (c1, cL) carry q^{1-2n},
    q^{2-2n}.  The negative-power cancellation is exactly what forces
    Tr(L)/Tr(1) to leading order -q."""
    A = PentagonKAlg()
    for n in range(0, 8):
        Tr_Ln = A.trace((0, n, 0), K=20)
        for e, rc in Tr_Ln.coeffs.items():
            if e < 0:
                assert rc.is_zero(), (
                    f"Tr(L^{n})[q^{e}] = {rc} but expected 0 (no neg q-powers)"
                )


def test_tr_L_over_tr_1_ratio_against_ansatz():
    """Note's claim: `q^{n²-1} Tr(L^n) = A_n(q) Tr(1) + B_n(q) Tr(L)`,
    and rearranging gives `Tr(L)/Tr(1) = -A_n/B_n + O(q^{n²})`.

    Verify this by computing both Tr(L)/Tr(1) and -A_n/B_n as truncated
    power series up to order n²-1; they must agree."""
    A = PentagonKAlg()
    K = 60
    Tr1 = A.trace((0, 0, 0), K=K)
    TrL = A.trace((0, 1, 0), K=K)
    # Tr(L)/Tr(1) = TrL * Tr1^{-1}.
    inv_tr1 = _ps_inverse({e: _z(Tr1[e]) for e in Tr1.coeffs}, K)
    trL_dict = {e: _z(TrL[e]) for e in TrL.coeffs}
    true_ratio = _ps_mul(trL_dict, inv_tr1, K)
    for n in range(2, 7):
        c1, cL = _pent_tr_power_coeffs(n)
        shift = n * n - 1
        A_n_dict = {e + shift: c for e, c in c1._coeffs.items()}
        B_n_dict = {e + shift: c for e, c in cL._coeffs.items()}
        # -A_n / B_n up to q^{n²-1}.
        approx = _ps_mul(
            {e: -c for e, c in A_n_dict.items()},
            _ps_inverse(B_n_dict, n * n - 1),
            n * n - 1,
        )
        for k in range(0, n * n):
            assert approx.get(k, 0) == true_ratio.get(k, 0), (
                f"n={n}, k={k}: -A_n/B_n[q^{k}] = {approx.get(k, 0)}, "
                f"Tr(L)/Tr(1)[q^{k}] = {true_ratio.get(k, 0)}"
            )


def _ps_mul(a: dict[int, int], b: dict[int, int], K: int) -> dict[int, int]:
    out: dict[int, int] = {}
    for e1, c1 in a.items():
        for e2, c2 in b.items():
            e = e1 + e2
            if e > K:
                continue
            out[e] = out.get(e, 0) + c1 * c2
    return {e: c for e, c in out.items() if c != 0}


def _ps_inverse(p: dict[int, int], K: int) -> dict[int, int]:
    """Inverse of a power series with leading coef 1 at exponent 0."""
    assert p.get(0, 0) == 1, f"_ps_inverse: leading coef must be 1, got {p.get(0)}"
    out = {0: 1}
    for k in range(1, K + 1):
        # out[k] + sum_{j=1..k} p[j] * out[k-j] = 0  ⇒  out[k] = -sum_{j=1..k} p[j] out[k-j]
        s = 0
        for j in range(1, k + 1):
            s += p.get(j, 0) * out.get(k - j, 0)
        if s != 0:
            out[k] = -s
    return out




def test_axioms_low_degree():
    """Pentagon axioms on a degree-≤2 basis window."""
    A = PentagonKAlg()
    assert A.verify_identity_in_basis()
    assert A.verify_rho_fixes_identity()
    labels = [(0, 0, 0)] + [(i, 1, 0) for i in range(5)] + [(0, 1, 1), (0, 2, 0)]
    K = 8
    for a in labels:
        assert A.verify_rho_inverse(a)
        for b in labels:
            assert A.verify_bar_involution(a, b), (a, b)
            assert A.verify_rho_twisted_trace(a, b, K=K), (a, b)
            assert A.verify_orthonormality(a, b, K=K), (a, b)


if __name__ == "__main__":
    tests = [
        test_tr_1_leading_coefficients,
        test_tr_L_leading_coefficients,
        test_schur_recursion_against_note_n2_through_n6,
        test_tr_L_squared_starts_at_q2,
        test_tr_L_over_tr_1_leading_term,
        test_layer1_coefficients_converge_to_traces,
        test_tr_powers_have_no_negative_q_powers,
        test_tr_L_over_tr_1_ratio_against_ansatz,
        test_axioms_low_degree,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS: {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL: {t.__name__}: {e}")
            raise
    print("\nAll Pentagon-note formula tests passed.")
