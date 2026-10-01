"""`A1A2kKAlg.curve`: the canonical-basis labels of `[A_1, A_{2k}]` are
multisets of pairwise non-crossing diagonals of the `(2k+3)`-gon.  The accessor reuses the `curve(x, ell)` convention of
`A1DnKAlg` — the diagonal from marked point `x` to
`x + ell` — with the identification `curve(x, ell) == curve(x + ell,
H - ell)` of the unpunctured polygon.

Every geometric statement below is checked against a crossing predicate
written in this file, not against the class's own tables.  At k = 1..5
(H = 2k + 3):

  G1  `curve` is a bijection from the k(2k+3) diagonals onto the letters;
      the entry `(a, j, e)` is `curve(j, a + 1)` taken `e` times; a
      boundary edge (and any ell outside [2, H - 2]) raises.
  G2  two letters q-commute iff their diagonals do not cross; a
      non-crossing product is the single label holding both diagonals;
      the crossing ordered pairs number 2·C(H, 4) = 10/70/252/660/1430.
  G3  the product of two crossing diagonals is exactly two terms, the two
      pairs of opposite sides of the quadrilateral they span (a boundary
      edge is the identity), each with coefficient a single power of q;
      `L_y L_x = bar(L_x L_y)` on every crossing pair, and a product with
      one exponent shifted fails that check (negative control).
  G4  ρ(curve(x, ell)) = curve(x + 1, ell), ρ is an automorphism on every
      ordered pair of letters, and the order of ρ is H = 5/7/9/11/13.
  G5  the cones of `cone_data()` are the triangulations of the H-gon,
      Catalan(2k + 1) = 5/42/429/4862/58786 of them.
  G6  the q-commutation exponent of a non-crossing pair is 2⟨γ, γ'⟩ on the
      A_{2k} lattice (charges from `A1A2k_naming_audit.natural_orbit_seeds`
      and the lattice ρ `a2k_rho`, pairing
      `A1A2k_plucker_closed_form.A2k_pairing`; neither is read by the
      class's product table).
  G7  the stated-skein engine `SkeinOddPolygonKAlg`, which asserts every
      coefficient it peels, reproduces every crossing product up to
      rotation at k = 5, 6 (110 / 182 ordered pairs).
  N   a dictionary with the orbit-1 letters shifted by one vertex fails
      G2 at k = 2..5 (at k = 1 every letter lies in one ρ-orbit, so a
      uniform shift is the automorphism ρ and cannot fail).

Run: `python3 run_tests.py`.
"""
from __future__ import annotations

import itertools
import math
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, os.path.join(_REPO, "implementations")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from kalgebra import Element
from laurent_poly import LaurentPoly
from a1a2k_kalg import A1A2kKAlg


KS = (1, 2, 3, 4, 5)
CROSSING_ORDERED = {1: 10, 2: 70, 3: 252, 4: 660, 5: 1430}
CATALAN = {1: 5, 2: 42, 3: 429, 4: 4862, 5: 58786}
SKEIN_PAIRS = {5: 110, 6: 182}

_ALG: dict = {}


def _alg(k: int) -> A1A2kKAlg:
    if k not in _ALG:
        _ALG[k] = A1A2kKAlg(k)
    return _ALG[k]


# ---------------------------------------------------------------------------
# Geometry of the H-gon, written independently of the class
# ---------------------------------------------------------------------------


def _ends(H: int, x: int, ell: int) -> frozenset:
    return frozenset({x % H, (x + ell) % H})


def _is_edge(H: int, u: int, v: int) -> bool:
    return (u - v) % H in (1, H - 1)


def _diagonals(H: int) -> list:
    """Every diagonal once, as `(x, ell)` with `x < x + ell < H`."""
    return [(u, v - u) for u in range(H) for v in range(u + 2, H)
            if not _is_edge(H, u, v)]


def _crosses(d1: frozenset, d2: frozenset) -> bool:
    if d1 & d2:
        return False
    p, q = sorted(d1)
    r, s = tuple(d2)
    return (p < r < q) != (p < s < q)


def _as_multiset(H: int, label) -> tuple:
    """The multiset of diagonals of a label, reading the entry `(a, j, e)`
    as the diagonal {j, j + a + 1} taken e times."""
    out = []
    for (a, j, e) in label:
        out += [tuple(sorted(_ends(H, j, a + 1)))] * e
    return tuple(sorted(out))


def _qcommute_exponent(A, x, y):
    """`c` with `L_x L_y = q^c L_y L_x` when both products are one label
    with coefficient a single power of q (with coefficient 1); else None."""
    P, Q = A.multiply(x, y), A.multiply(y, x)
    if len(P.terms) != 1 or len(Q.terms) != 1:
        return None
    (lp, cp), = P.terms.items()
    (lq, cq), = Q.terms.items()
    if lp != lq or not _unit_monomial(cp) or not _unit_monomial(cq):
        return None
    return next(iter(cp._coeffs)) - next(iter(cq._coeffs))


def _unit_monomial(c: LaurentPoly) -> bool:
    return len(c._coeffs) == 1 and next(iter(c._coeffs.values())) == 1


def _g2_mismatches(A, dictionary) -> tuple[int, int]:
    """(mismatches of q-commute ⟺ non-crossing, crossing ordered pairs)
    over all ordered pairs of distinct diagonals, with the letters read
    through `dictionary(x, ell)`."""
    H = A.H
    diags = _diagonals(H)
    bad = n_cross = 0
    for (x1, l1), (x2, l2) in itertools.permutations(diags, 2):
        cr = _crosses(_ends(H, x1, l1), _ends(H, x2, l2))
        n_cross += cr
        qc = _qcommute_exponent(A, dictionary(x1, l1), dictionary(x2, l2))
        if (qc is None) != cr:
            bad += 1
    return bad, n_cross


# ---------------------------------------------------------------------------
# G1
# ---------------------------------------------------------------------------


def test_curve_is_a_bijection_onto_the_letters():
    for k in KS:
        A = _alg(k)
        H = A.H
        image: dict = {}
        for x in range(H):
            for ell in range(2, H - 1):
                lab = A.curve(x, ell)
                assert lab == A.curve(x + ell, H - ell), (k, x, ell)
                assert lab == A.curve(x + H, ell) == A.curve(x - H, ell)
                assert len(lab) == 1 and lab[0][2] == 1, lab
                a, j, _e = lab[0]
                assert 1 <= a <= k and 0 <= j < H, lab
                assert _ends(H, j, a + 1) == _ends(H, x, ell), (x, ell, lab)
                assert A.L((a, j)) == lab
                image.setdefault(_ends(H, x, ell), set()).add(lab)
        assert len(image) == k * (2 * k + 3) == H * (H - 3) // 2
        assert all(len(v) == 1 for v in image.values())
        letters = {A.L((a, j)) for a in range(1, k + 1) for j in range(H)}
        assert {next(iter(v)) for v in image.values()} == letters
        assert len(letters) == len(image)
    print(f"  PASS: curve is a bijection onto the k(2k+3) letters, k={KS}")


def test_entry_is_a_power_of_its_curve():
    for k in (1, 2, 3):
        A = _alg(k)
        H = A.H
        for (x, ell) in _diagonals(H):
            c = A.curve(x, ell)
            (a, j, _e), = c
            cur = c
            for e in (2, 3):
                P = A.multiply(cur, c)
                assert dict(P.terms) == {((a, j, e),): LaurentPoly.one()}, (
                    c, e, dict(P.terms))
                cur = ((a, j, e),)
    print("  PASS: (a, j, e) is curve(j, a+1) taken e times")


def test_boundary_edges_and_out_of_range_raise():
    for k in KS:
        A = _alg(k)
        H = A.H
        for x in range(H):
            for ell in (1, H - 1, 0, H, -1, H + 1, -H):
                try:
                    A.curve(x, ell)
                except ValueError:
                    continue
                raise AssertionError(f"curve({x}, {ell}) accepted at k={k}")
    print("  PASS: boundary edges and out-of-range ell raise")


# ---------------------------------------------------------------------------
# G2, G3
# ---------------------------------------------------------------------------


def test_qcommute_iff_noncrossing():
    for k in KS:
        A = _alg(k)
        bad, n_cross = _g2_mismatches(A, A.curve)
        assert bad == 0, (k, bad)
        assert n_cross == CROSSING_ORDERED[k] == 2 * math.comb(A.H, 4), (
            k, n_cross)
        # a non-crossing product is the single label holding both diagonals
        H = A.H
        for (x1, l1), (x2, l2) in itertools.permutations(_diagonals(H), 2):
            d1, d2 = _ends(H, x1, l1), _ends(H, x2, l2)
            if _crosses(d1, d2):
                continue
            P = A.multiply(A.curve(x1, l1), A.curve(x2, l2))
            (lab, _c), = P.terms.items()
            assert _as_multiset(H, lab) == tuple(sorted(
                [tuple(sorted(d1)), tuple(sorted(d2))])), (d1, d2, lab)
    print(f"  PASS: q-commute <=> non-crossing; crossing ordered pairs "
          f"{[CROSSING_ORDERED[k] for k in KS]}")


def test_crossing_products_are_the_two_pairs_of_opposite_sides():
    for k in KS:
        A = _alg(k)
        H = A.H
        n = 0
        for (x1, l1), (x2, l2) in itertools.permutations(_diagonals(H), 2):
            d1, d2 = _ends(H, x1, l1), _ends(H, x2, l2)
            if not _crosses(d1, d2):
                continue
            a, b, c, d = sorted(d1 | d2)

            def side_pair(u, v, w, z):
                sides = [tuple(sorted((u, v))), tuple(sorted((w, z)))]
                return tuple(sorted(s for s in sides
                                    if not _is_edge(H, s[0], s[1])))

            want = sorted([side_pair(a, b, c, d), side_pair(a, d, b, c)])
            P = A.multiply(A.curve(x1, l1), A.curve(x2, l2))
            got = sorted(_as_multiset(H, lab) for lab in P.terms)
            assert got == want, (k, d1, d2, got, want)
            assert all(_unit_monomial(cf) for cf in P.terms.values()), (
                k, d1, d2, dict(P.terms))
            n += 1
        assert n == CROSSING_ORDERED[k]
    print("  PASS: crossing products = the two pairs of opposite sides")


def test_bar_compatibility_on_crossing_pairs_with_negative_control():
    for k in KS:
        A = _alg(k)
        H = A.H
        n = caught = 0
        for (x1, l1), (x2, l2) in itertools.permutations(_diagonals(H), 2):
            if not _crosses(_ends(H, x1, l1), _ends(H, x2, l2)):
                continue
            x, y = A.curve(x1, l1), A.curve(x2, l2)
            assert A.verify_bar_involution(x, y), (k, x, y)
            # negative control: one exponent of L_x L_y shifted by one
            P = A.multiply(x, y)
            first = min(P.terms)
            bumped = Element({lab: (cf * LaurentPoly.q(1) if lab == first
                                    else cf) for lab, cf in P.terms.items()})
            caught += bumped.bar() != A.multiply(y, x)
            n += 1
        assert n == CROSSING_ORDERED[k] and caught == n, (k, n, caught)
    print("  PASS: L_y L_x = bar(L_x L_y) on every crossing pair; the "
          "shifted-exponent control is caught on every pair")


# ---------------------------------------------------------------------------
# G4, G5, G6
# ---------------------------------------------------------------------------


def test_rho_is_rotation_of_order_H():
    for k in KS:
        A = _alg(k)
        H = A.H
        for x in range(H):
            for ell in range(2, H - 1):
                c = A.curve(x, ell)
                assert A.rho(c) == A.curve(x + 1, ell), (k, x, ell)
                assert A.rho_inverse(A.curve(x + 1, ell)) == c
                cur, order = A.rho(c), 1
                while cur != c:
                    cur, order = A.rho(cur), order + 1
                assert order == H, (k, c, order)
        letters = [A.L((a, j)) for a in range(1, k + 1) for j in range(H)]
        for x, y in itertools.permutations(letters, 2):
            lhs = A.rho_element(A.multiply(x, y))
            rhs = A.multiply(A.rho(x), A.rho(y))
            assert dict(lhs.terms) == dict(rhs.terms), (k, x, y)
    print(f"  PASS: rho = rotation x -> x+1, an automorphism, order "
          f"{[_alg(k).H for k in KS]}")


def test_cones_are_the_triangulations():
    for k in KS:
        A = _alg(k)
        H = A.H
        t0 = time.time()
        cones = list(A.cone_data().cones())
        assert len(cones) == CATALAN[k] == math.comb(2 * (2 * k + 1),
                                                     2 * k + 1) // (2 * k + 2)
        assert len(set(cones)) == len(cones)
        for cone in cones:
            assert len(cone) == H - 3, cone
            ds = [_ends(H, j, a + 1) for (a, j) in cone]
            assert len(set(ds)) == len(ds)
            assert not any(_crosses(u, v)
                           for u, v in itertools.combinations(ds, 2)), cone
        print(f"    k={k}: {len(cones)} cones, each H-3 = {H - 3} pairwise "
              f"non-crossing diagonals ({time.time() - t0:.1f}s)")
    print(f"  PASS: cones = triangulations {[CATALAN[k] for k in KS]}")


def test_qcommute_exponent_is_twice_the_lattice_pairing():
    from A1A2k_naming_audit import natural_orbit_seeds, a2k_rho
    from A1A2k_plucker_closed_form import A2k_pairing
    for k in KS:
        A = _alg(k)
        H = A.H
        B = A2k_pairing(k)
        seeds = natural_orbit_seeds(k)
        gamma = {}
        for a in range(1, k + 1):
            g = tuple(seeds[a])
            for j in range(H):
                gamma[(a, j)] = g
                g = tuple(a2k_rho(k, g))
            assert g == tuple(seeds[a]), (k, a)        # the ρ-orbit closes
        assert len(set(gamma.values())) == len(gamma)

        def pair(u, v):
            return sum(u[i] * B[i][j] * v[j]
                       for i in range(2 * k) for j in range(2 * k))

        n = 0
        for (a1, j1), (a2, j2) in itertools.permutations(gamma, 2):
            qc = _qcommute_exponent(A, A.L((a1, j1)), A.L((a2, j2)))
            if qc is None:
                continue
            assert qc == 2 * pair(gamma[(a1, j1)], gamma[(a2, j2)]), (
                k, (a1, j1), (a2, j2), qc)
            n += 1
        assert n == len(gamma) * (len(gamma) - 1) - CROSSING_ORDERED[k]
    print("  PASS: q-commute exponent = 2<gamma, gamma'> on the A_2k lattice")


# ---------------------------------------------------------------------------
# G7, N
# ---------------------------------------------------------------------------


def test_skein_agreement_on_crossing_pairs_k5_k6():
    from skein_oddgon_kalg import SkeinOddPolygonKAlg
    for k in (5, 6):
        A = _alg(k) if k in KS else A1A2kKAlg(k)
        H = A.H
        t0 = time.time()
        S = SkeinOddPolygonKAlg(intrinsic=A)
        t_build = time.time() - t0
        n = 0
        for ell in range(2, k + 2):
            d1 = _ends(H, 0, ell)
            for (x2, l2) in _diagonals(H):
                if not _crosses(d1, _ends(H, x2, l2)):
                    continue
                x, y = A.curve(0, ell), A.curve(x2, l2)
                # raises on any coefficient or label-set mismatch
                P = S.multiply(x, y)
                assert dict(P.terms) == dict(A.multiply(x, y).terms)
                n += 1
        assert n == SKEIN_PAIRS[k] == 2 * math.comb(H, 4) // H, (k, n)
        print(f"    k={k}: skein build {t_build:.1f}s, {n} crossing pairs "
              f"agree ({time.time() - t0:.1f}s total)")
    print("  PASS: skein engine agrees on every crossing pair up to rotation")


def test_off_by_one_dictionary_fails_the_noncrossing_test():
    for k in (2, 3, 4, 5):
        A = _alg(k)

        def shifted(x, ell, A=A):
            (a, j, _e), = A.curve(x, ell)
            return A.L((a, j + 1)) if a == 1 else A.L((a, j))

        bad, _ = _g2_mismatches(A, shifted)
        assert bad > 0, k
        print(f"    k={k}: orbit-1 letters shifted by one vertex -> "
              f"{bad} mismatches")
    print("  PASS: an off-by-one dictionary fails the non-crossing test")


if __name__ == "__main__":
    tests = [
        test_curve_is_a_bijection_onto_the_letters,
        test_entry_is_a_power_of_its_curve,
        test_boundary_edges_and_out_of_range_raise,
        test_qcommute_iff_noncrossing,
        test_crossing_products_are_the_two_pairs_of_opposite_sides,
        test_bar_compatibility_on_crossing_pairs_with_negative_control,
        test_rho_is_rotation_of_order_H,
        test_cones_are_the_triangulations,
        test_qcommute_exponent_is_twice_the_lattice_pairing,
        test_skein_agreement_on_crossing_pairs_k5_k6,
        test_off_by_one_dictionary_fails_the_noncrossing_test,
    ]
    t_all = time.time()
    for t in tests:
        t0 = time.time()
        t()
        print(f"      ({t.__name__}: {time.time() - t0:.1f}s)")
    print(f"\nAll A1A2kKAlg curve checks passed ({time.time() - t_all:.1f}s).")
