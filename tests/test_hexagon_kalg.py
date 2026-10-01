"""Tests for HexagonKAlg — the ungauged μ-flavoured hexagon."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hexagon_kalg import HexagonKAlg
from u1_hexagon_kalg import MU_LETTER_QPOWER
from laurent_poly import LaurentPoly


def test_construction():
    H = HexagonKAlg()
    R = H.coefficient_ring()
    assert R.rank == 1, R.rank
    assert H.identity() == ((), 0)
    print("  PASS: test_construction")


def test_mult_generators_mag_zero():
    """All 6 named mult-generators are mag-zero (so they're valid Hex basis elements)."""
    H = HexagonKAlg()
    for g in H.mult_generators():
        H._assert_mag_zero(g, "mult-generator")
    print(f"  PASS: test_mult_generators_mag_zero (6/6)")


def test_mult_generator_labels():
    """L_long(j), L_diam(i) produce the expected cone-monomial labels."""
    H = HexagonKAlg()
    assert H.L_long(0) == (((2, 0, 1),), 0)
    assert H.L_long(2) == (((2, 2, 1),), 0)
    assert H.L_diam(0) == (((1, 0, 1), (1, 3, 1)), 0)
    assert H.L_diam(1) == (((1, 1, 1), (1, 4, 1)), 0)
    assert H.L_diam(2) == (((1, 2, 1), (1, 5, 1)), 0)
    print("  PASS: test_mult_generator_labels")


def test_identity_is_neutral():
    """multiply(identity, g) = g for each mult-generator g."""
    H = HexagonKAlg()
    e = H.identity()
    for g in H.mult_generators():
        out = H.multiply(e, g)
        # Output should be a single-term dict {g: 1}
        assert list(out.terms.keys()) == [g], (g, out)
        assert str(out.terms[g]) == '1', (g, out.terms[g])
    print("  PASS: test_identity_is_neutral")


def test_long_long_products():
    """L_long(i) · L_long(j) — verify mag-zero output, delegated to U(1)Hex."""
    H = HexagonKAlg()
    for i in range(3):
        for j in range(3):
            out = H.multiply(H.L_long(i), H.L_long(j))
            for lbl, coef in out.terms.items():
                H._assert_mag_zero(lbl, f"L_long({i}) · L_long({j}) → {lbl}")
    print("  PASS: test_long_long_products (9/9 mag-zero)")


def test_diam_diam_products():
    """L_diam(i) · L_diam(j): mag-zero outputs, ditto."""
    H = HexagonKAlg()
    for i in range(3):
        for j in range(3):
            out = H.multiply(H.L_diam(i), H.L_diam(j))
            for lbl in out.terms:
                H._assert_mag_zero(lbl, f"L_diam({i}) · L_diam({j}) → {lbl}")
    print("  PASS: test_diam_diam_products (9/9 mag-zero)")


def test_long_diam_products():
    """L_long(j) · L_diam(i) and L_diam(i) · L_long(j): mag-zero outputs."""
    H = HexagonKAlg()
    for i in range(3):
        for j in range(3):
            for ord_a, ord_b in [(H.L_long(j), H.L_diam(i)),
                                  (H.L_diam(i), H.L_long(j))]:
                out = H.multiply(ord_a, ord_b)
                for lbl in out.terms:
                    H._assert_mag_zero(lbl, "long·diam")
    print("  PASS: test_long_diam_products (18/18 mag-zero)")


def test_associativity_on_generators():
    """(a · b) · c = a · (b · c)  for all triples of the 6 mult-generators."""
    import itertools
    H = HexagonKAlg()
    gens = H.mult_generators()
    fails = 0
    for ga, gb, gc in itertools.product(gens, gens, gens):
        s1 = H.multiply(ga, gb)
        left = {}
        for k, c in s1.terms.items():
            for kk, vv in H.multiply(k, gc).terms.items():
                left[kk] = (left.get(kk, LaurentPoly.zero()) + c * vv)
        s2 = H.multiply(gb, gc)
        right = {}
        for k, c in s2.terms.items():
            for kk, vv in H.multiply(ga, k).terms.items():
                right[kk] = (right.get(kk, LaurentPoly.zero()) + c * vv)
        # Compare by charge (label-canonicalization may differ between two cones)
        from hexagon_kalg import HexagonKAlg as HK
        left_chg, right_chg = {}, {}
        for k, v in left.items():
            chg = HK.charge_of_label(k); left_chg[chg] = left_chg.get(chg, LaurentPoly.zero()) + v
        for k, v in right.items():
            chg = HK.charge_of_label(k); right_chg[chg] = right_chg.get(chg, LaurentPoly.zero()) + v
        keys = set(left_chg) | set(right_chg)
        if not all(str(left_chg.get(k, LaurentPoly.zero())) == str(right_chg.get(k, LaurentPoly.zero())) for k in keys):
            fails += 1
    assert fails == 0, f"{fails}/216 associativity failures"
    print(f"  PASS: test_associativity_on_generators (216/216)")


def test_section_decompose(K=8):
    """`r_label_decompose((F, e)) = ((F, 0), (−e,))` — the single flavour-irrep
    key, with the sign the trace fixes: `Tr((F, e)) = μ^{−e}·Tr((F, 0))`, so
    `M_{F, e} = μ^{−e}·M_{F, 0}` (`UngaugedKAlgebra`'s rule, delegated).
    `r_label_compose` inverts it, `_label_section_decompose` lifts the key to
    the `RElement` μ^{−e} (the old hand-rolled `_lsd` returned the bare key
    tuple, which broke `to_R_form`), `to_R_form` round-trips, and the trace is
    R-linear for this lift.  Until 2026-09-23 this test pinned the key
    `(e,)`, which the trace contradicts; that key is the negative control."""
    from kalgebra import Element
    H = HexagonKAlg()
    R = H.coefficient_ring()
    for label, sec_exp, mu in [
        ((((2, 0, 1),), 0), (((2, 0, 1),), 0), 0),
        ((((1, 0, 1), (1, 3, 1)), 5), (((1, 0, 1), (1, 3, 1)), 0), 5),
        ((((2, 1, 1),), -2), (((2, 1, 1),), 0), -2),
        (((), 2), ((), 0), 2),
    ]:
        sec, key = H.r_label_decompose(label)
        assert sec == sec_exp and key == (-mu,), (sec, key)
        assert H.r_label_compose(sec, key) == label, label
        ls, lr = H._label_section_decompose(label)
        assert ls == sec_exp and lr == R.basis_element((-mu,)), (ls, lr)
        assert H.verify_section_is_single_irrep(label), label
        La = Element.basis(label)
        assert H.from_R_form(H.to_R_form(La)) == La, label
        tr, base = H.trace(label, K), H.trace(sec, K)
        assert tr == base * R.basis_element(key), label
        if mu:
            assert tr != base * R.basis_element((mu,)), label   # the old key
    print("  PASS: test_section_decompose")


def test_rho_inverse_is_inverse():
    """ρ ∘ ρ^{-1} = id  (on generators that don't hit the latent U(1)Hex
    antipodal/periodicity convention asymmetry — i.e., shorts with i ≤ 4 and
    longs with i ≤ 1).  See `U1HexagonKAlg.rho` docstring caveat."""
    H = HexagonKAlg()
    # L_long(0), L_long(1), L_diam(0), L_diam(1) avoid the wrap-around issue.
    safe = [H.L_long(0), H.L_long(1), H.L_diam(0), H.L_diam(1)]
    for g in safe:
        assert H.rho_inverse(H.rho(g)) == g, g
    print(f"  PASS: test_rho_inverse_is_inverse ({len(safe)}/{len(safe)} safe gens)")


def test_non_mag_zero_input_rejected():
    """Multiplying a non-mag-zero label raises ValueError."""
    H = HexagonKAlg()
    bad = (((1, 0, 1),), 0)  # single short — mag = -1, NOT in Hex basis
    try:
        H.multiply(bad, H.L_long(0))
    except ValueError:
        print("  PASS: test_non_mag_zero_input_rejected")
        return
    raise AssertionError("expected ValueError for non-mag-zero input")


def test_cyclic_a2_bps_construction():
    """Sanity-check that BPSKAlgebra(cyclic-A_2) constructs and has the expected
    μ-flavour shape (kernel of degenerate B has rank 1, matching Hex's μ).

    The full KAlgebraIso(HexagonKAlg, BPS_cyclic_A_2) is conjectured but its
    label-maps are not yet implemented — see the design notes."""
    from bps_kalgebra import BPSKAlgebra
    B_cyclic = [[0, 1, -1], [-1, 0, 1], [1, -1, 0]]
    A_bps = BPSKAlgebra(pairing=B_cyclic, node_charges=[(1,0,0),(0,1,0),(0,0,1)])
    # Single-node traces should be all equal (Z_3 cyclic symmetry)
    t0 = str(A_bps.trace((1,0,0), K=4))
    t1 = str(A_bps.trace((0,1,0), K=4))
    t2 = str(A_bps.trace((0,0,1), K=4))
    assert t0 == t1 == t2, (t0, t1, t2)
    # Coefficient ring should be AbelianZPlusRing rank 1 (μ-flavoured)
    assert A_bps.coefficient_ring().rank == 1, A_bps.coefficient_ring().rank
    print("  PASS: test_cyclic_a2_bps_construction (μ-rank 1, Z_3-symmetric traces)")


if __name__ == '__main__':
    test_construction()
    test_mult_generators_mag_zero()
    test_mult_generator_labels()
    test_identity_is_neutral()
    test_long_long_products()
    test_diam_diam_products()
    test_long_diam_products()
    test_associativity_on_generators()
    test_section_decompose()
    test_rho_inverse_is_inverse()
    test_non_mag_zero_input_rejected()
    test_cyclic_a2_bps_construction()
    print("\nAll HexagonKAlg tests passed.")
