"""Tests for `E7RGKAlgebra` — the exceptional Argyres–Douglas theory
`A_𝖖([A_1, E_7])` as a new-contract `RGKAlgebra` over `A1A2kKAlg(3).add_flavour(1)`
(= `[A_1, A_6]` ⊕ a spectator U(1) flavour), via the single-node-drop flow
`[A_1, E_7] → [A_1, A_6] + U(1)` with `S_RG = E_𝖖(μL)`, `L` the **central** chord.

E₇ is the A₆ chain `1-2-3-4-5-6` with one node at the **centre** (arms 2-3-1);
the warm-up's A₇ attaches at the **end**.  Which chord `L` is dressed selects E₇
vs A₇.  Odd rank 7 ⇒ the substrate is `[A_1, A_6]` + U(1) flavour (the gauged-odd
standalones `U1A1AoddKAlg(k)` are even rank `2k+2`), "as in many other examples".

Covered:
(1) construction — aux `A1A2kKAlg(3).add_flavour(1)`, `L` the central chord
    `(3,0)`, `L^m` a single cone monomial;
(2) intrinsic K-algebra axioms (unital, RG-multiplicative, ρ-automorphism,
    bar-involution, associativity);
(3) **structural identification** — the UV BPS quiver is the **E₇ Dynkin**:
    `uv_cartan_determinant() == 2` (A₆ chain + central chord), vs `8` (A₇) for an
    end chord; among trees on 7 nodes only E₇ has Cartan det 2.  This matches the
    finite BPS(E₇) quiver.  It is necessary, not sufficient: the chord `(2, 4)`
    has the same crossing graph and determinant but a different vacuum, so (4)
    is the discriminating check;
(4) the μ-refined vacuum Schur index at low order: `1 + q²` (the q² is the
    adjoined U(1) flavour current; high order is reducer-limited).  NEGATIVE
    CONTROL: with `L = (2, 4)` the `q²` coefficient is `μ⁻¹ + 1 + μ`.

Run from the repo root: `python3 run_tests.py`.
"""
import os
import sys
import itertools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "implementations"))

from e7_rgkalgebra import E7RGKAlgebra
from kalgebra import Element
from laurent_poly import LaurentPoly

_ONE = LaurentPoly.one()


def _sample_gens(T, per_type=2):
    cd = T._A.cone_data()
    gens = [((), (0,))]                                  # identity
    for a in sorted({x for x, _i in cd.mult_gens()}):
        for i in sorted(i for x, i in cd.mult_gens() if x == a)[:per_type]:
            gens.append((((a, i, 1),), (0,)))
    return gens


def test_construction():
    """aux = `A1A2kKAlg(3).add_flavour(1)`; the dressing chord `L` is the central
    (longest-type) chord `(3, 0)`; `L^m` is a single cone monomial."""
    T = E7RGKAlgebra()
    assert type(T.auxiliary()).__name__ == "AddFlavourKAlgebra"
    assert (T._La, T._i0) == (3, 0)
    e = Element({(T._central_chord_power(1), (1,)): _ONE})
    # L^m single cone monomial (check in the survivor A1A2k(3))
    p = Element({T._central_chord_power(1): _ONE})
    for m in range(2, 6):
        p = T._A.multiply_elements(p, Element({T._central_chord_power(1): _ONE}))
        assert list(p.terms) == [T._central_chord_power(m)], m


def test_kalgebra_axioms():
    """unital, RG-multiplicative, ρ-automorphism, bar-involution, associativity."""
    T = E7RGKAlgebra()
    gens = _sample_gens(T)
    assert T.verify_rg_unital()
    assert T.verify_identity_in_basis()
    assert T.verify_rho_fixes_identity()
    for a, b in itertools.product(gens, gens):
        assert T.verify_rg_multiplicative(a, b), ("rg_mult", a, b)
        assert T.verify_rho_is_automorphism(a, b), ("rho_auto", a, b)
        assert T.verify_bar_involution(a, b), ("bar", a, b)
    for a, b, c in itertools.product(gens[:5], repeat=3):
        lhs = T.multiply_elements(T.multiply(a, b), Element({c: _ONE}))
        rhs = T.multiply_elements(Element({a: _ONE}), T.multiply(b, c))
        assert lhs == rhs, ("assoc", a, b, c)


def test_uv_quiver_is_E7():
    """The UV BPS quiver is the E₇ Dynkin: A₆ chain + central chord has Cartan
    determinant 2 (E₇), vs 8 (A₇) for an end chord.  (Among trees on 7 nodes only
    E₇ has Cartan det 2: A₇=8, D₇=4, E₇=2.)  The crossing graph does not
    determine the flow, though: the chord (2, 4) also gives det 2, and its
    vacuum differs (`test_vacuum_schur_low_order`)."""
    T = E7RGKAlgebra()
    assert T.uv_cartan_determinant() == 2        # E7
    assert T.verify_is_E7()
    chain = T._a6_chain()
    # contrast: an end chord (A7) attaches at the chain end → 8
    assert T._cartan_det(chain + [(1, 6)]) == 8   # A7 (end attachment)
    # the same crossing graph from another chord: necessary, not sufficient
    assert T._cartan_det(chain + [(2, 4)]) == 2


def test_matches_bps_e7_quiver():
    """Direct comparison with the finite **BPS(E₇)** algebra: the E₇RG UV BPS
    quiver (A₆ chain + central chord) is the same quiver as the standard E₇ Dynkin
    that `BPSKAlgebra` is built from — same Cartan determinant 2.  This matches
    the shape of the crossing graph only; it does not identify the flow's
    algebra (the `(2, 4)` dressing has the same crossing graph and a different
    vacuum — `test_vacuum_schur_low_order` is the discriminating check)."""
    from bps_kalgebra import BPSKAlgebra
    from fractions import Fraction

    # standard E7 Dynkin exchange matrix: A6 chain 0-1-2-3-4-5 + node 6 at position 2
    B_E7 = [[0, 1, 0, 0, 0, 0, 0], [-1, 0, 1, 0, 0, 0, 0], [0, -1, 0, 1, 0, 0, 1],
            [0, 0, -1, 0, 1, 0, 0], [0, 0, 0, -1, 0, 1, 0], [0, 0, 0, 0, -1, 0, 0],
            [0, 0, -1, 0, 0, 0, 0]]

    def cartan_det(B):
        n = len(B)
        M = [[Fraction(2 if i == j else -(1 if B[i][j] else 0)) for j in range(n)]
             for i in range(n)]
        d = Fraction(1)
        for c in range(n):
            p = next((i for i in range(c, n) if M[i][c] != 0), None)
            if p is None:
                return 0
            if p != c:
                M[c], M[p] = M[p], M[c]
                d = -d
            d *= M[c][c]
            for i in range(c + 1, n):
                f = M[i][c] / M[c][c]
                M[i] = [a - f * b for a, b in zip(M[i], M[c])]
        return int(d)

    nc = [tuple(1 if j == i else 0 for j in range(7)) for i in range(7)]
    BPSKAlgebra(pairing=B_E7, node_charges=nc)            # builds (finite E7 chamber)
    assert cartan_det(B_E7) == 2                          # the BPS(E7) quiver is E7
    assert E7RGKAlgebra().uv_cartan_determinant() == 2    # E7RG's UV quiver matches


def test_vacuum_schur_low_order():
    """The μ-refined vacuum Schur index at low order: `1 + q²` (q⁰ vacuum, q² the
    adjoined U(1) flavour current).  High order is limited by the A1A2k(3) cone
    trace reducer on the central-chord powers, so only low order is asserted."""
    T = E7RGKAlgebra()
    tv = T.trace(((), (0,)), 2)
    assert str(tv.coeffs.get(0)) == "1"          # vacuum
    assert str(tv.coeffs.get(2)) == "1"          # U(1) flavour current at q²

    # NEGATIVE CONTROL: the chord (2, 4) has the same crossing graph with the A6
    # chain (det 2, `test_uv_quiver_is_E7`) but three currents at q²
    class _Dress24(E7RGKAlgebra):
        def __init__(self):
            super().__init__()
            self._La, self._i0 = 2, 4

    B = _Dress24()
    assert B.uv_cartan_determinant() == 2
    q2 = {k: int(c) for k, c in B.trace(((), (0,)), 2).coeffs[2].terms.items() if int(c)}
    assert q2 == {(-1,): 1, (0,): 1, (1,): 1}, q2


if __name__ == "__main__":
    test_construction()
    print("construction (aux A1A2kKAlg(3).add_flavour(1), central chord L=(3,0)): OK")
    test_kalgebra_axioms()
    print("K-algebra axioms (unital, RG-mult, ρ-auto, bar, assoc): OK")
    test_uv_quiver_is_E7()
    print("UV BPS quiver = E7 Dynkin (Cartan det 2; vs A7=8 end chord): OK")
    test_matches_bps_e7_quiver()
    print("matches finite BPS(E7) quiver (Cartan det 2 both): OK")
    test_vacuum_schur_low_order()
    print("μ-refined vacuum Schur index low order = 1 + q² (U(1) current): OK")
    print("\nAll E7RGKAlgebra tests passed.")
