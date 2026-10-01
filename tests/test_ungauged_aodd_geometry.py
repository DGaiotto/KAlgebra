"""The geometry of the ungauged `[A_1, A_{2k+1}]`, `ungauge_u1a1aodd(k)`.

A label is a gauged `U1A1AoddKAlg(k)` label `(F, e)` in the centralizer of the
gauge letter `E`: `F` a multiset of pairwise non-crossing diagonals of the
`(2k+4)`-gon.  A letter's magnetic charge is set by the parities of its
endpoints (−2 both even, +2 both odd, 0 mixed), so `(F, e)` is in the
centralizer iff `F` is balanced; the multiplicative generators are the
mixed-parity diagonals and the non-crossing (even–even, odd–odd) pairs.

Checked here, positive controls first (a failing control aborts the run):
  * controls: the parity rule reproduces the documented `k = 1` facts of
    `tests/test_ungauge_kalgebra.py`; at `k = 1` the generators are exactly
    `HexagonKAlg`'s six hand-named ones; and the map to the finite zoo's
    generators (`FINITE_KALGEBRAS` a3/a5/a7, `<PRE>_MULT_GENS_LATTICE`) is a
    bijection, 6/6, 24/24, 65/65, compatible with q-commutation;
  * generator counts 6/24/65/144/280 at `k = 1..5`, each in the centralizer,
    the set ρ-stable up to the E-power, and every balanced label of a window
    a single-term product of generators;
  * the parity rule equals `mag` (the E-commutator) on every letter at
    `k = 1..5`, and `2·charge_formula[n−1]` at `k = 1..10`;
  * the parity-based `in_centralizer` equals the multiply-based test on a
    window of canonical labels (the fast path verified to fire);
  * orthonormality on generator pairs, the identity first (40/40 at `k = 4`,
    20/20 at `k = 5`, `K = 4`);
  * negative controls: unbalanced labels are outside the centralizer and
    refused by `geometric_label`; two off-by-one parity rules fail the charge
    check;
  * the non-simplicial case: `(A1B1)·(A2B2)` and `(A1B2)·(A2B1)` are single
    terms on one label.
"""
import importlib
import itertools
import os
import random
import sys
from math import comb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ungauge_kalgebra import UngaugedKAlgebra, ungauge_u1a1aodd
from u1a1aodd_kalg import U1A1AoddKAlg


def _crossing(c1, c2):
    a, b = sorted(c1)
    c, d = sorted(c2)
    if len({a, b, c, d}) < 4:
        return False
    return (a < c < b) != (a < d < b)


def _letter_label(g):
    return (((g[0], g[1], 1),), 0)


def _by_multiply(U, x):
    """The multiply-based centralizer test (the pre-2026-09-23 body of
    `UngaugedKAlgebra.in_centralizer`), computed independently here."""
    return U._G.multiply(U._E, x).terms == U._G.multiply(x, U._E).terms


def _canonical_window(cd, max_letters, exps=(1, 2), e_powers=(-1, 0, 2)):
    """Every canonical label with at most `max_letters` distinct pairwise
    non-crossing letters, exponents in `exps`, E-power in `e_powers`."""
    out = []
    for r in range(max_letters + 1):
        for S in itertools.combinations(cd._chords, r):
            if any(_crossing(cd.chord(a), cd.chord(b))
                   for a, b in itertools.combinations(S, 2)):
                continue
            for ms in itertools.product(exps, repeat=r):
                F = tuple(sorted((g[0], g[1], m) for g, m in zip(S, ms)))
                for e in e_powers:
                    out.append((F, e))
    return out


# ---------------------------------------------------------------------------
# positive controls
# ---------------------------------------------------------------------------

def test_control_documented_k1_facts():
    """The rule reproduces what `tests/test_ungauge_kalgebra.py` documents at
    `k = 1`: the six type-1 letters (short diagonals, both endpoints of one
    parity) are magnetically charged, the three type-2 letters (diameters,
    mixed) are neutral — and `mag` agrees."""
    U = ungauge_u1a1aodd(1)
    cd = U._G.cone_data()
    assert all(cd._letter_mag((1, i)) != 0 for i in range(6))
    assert all(cd._letter_mag((2, i)) == 0 for i in range(3))
    assert all(U.mag(_letter_label(g)) == cd._letter_mag(g) for g in cd._chords)


def test_control_k1_generators_are_hexagon_named_ones():
    """Known answer: at `k = 1` the enumeration returns exactly
    `HexagonKAlg`'s six hand-named generators (3 long diagonals + 3
    short-diagonal pairs `L_{1,i}·L_{1,i+3}`), as sets of letters."""
    from hexagon_kalg import HexagonKAlg
    named = {lbl[0] for lbl in HexagonKAlg().mult_generators()}
    ours = {lbl[0] for lbl in ungauge_u1a1aodd(1).mult_generators()}
    assert len(named) == 6 and ours == named, (ours, named)


_ZOO = {1: "a3", 2: "a5", 3: "a7"}


def _zoo_map(k):
    """The map of the scout's probes (`inventory/aodd/labelmap.py`): a
    generator's gauged charge, magnetic coordinate dropped, minus `f` times
    the `E` charge (so the last entry is 0), matched against the zoo's
    `<PRE>_MULT_GENS_LATTICE`.  Returns `(generators, lattice, match)` with
    `match[j]` the list of generator indices hitting lattice vector `j`."""
    import finite_kalgebras as fk
    U = ungauge_u1a1aodd(k)
    cd = U._G.cone_data()
    n, MU = cd._n, cd._MU
    gens = U.mult_generators()
    zid = _ZOO[k]
    mod = importlib.import_module(fk.FINITE_KALGEBRAS[zid].__module__)
    lat = [tuple(v) for v in getattr(mod, zid.upper() + "_MULT_GENS_LATTICE")]
    sec = {}
    for gi, (F, e) in enumerate(gens):
        assert e == 0
        v = [0] * n
        for (t, i, m) in F:
            c = cd._chg[(t, i)]
            for j in range(n):
                v[j] += m * c[j]
        assert v[n - 1] == 0, (k, F, v)          # balanced: no magnetic charge
        w = v[:n - 1]
        f = w[-1]
        sec.setdefault(tuple(w[j] - f * MU[j] for j in range(n - 1)), []).append(gi)
    match = [sec.get(v, []) for v in lat]
    return gens, lat, match, cd, fk.FINITE_KALGEBRAS[zid]()


def test_control_zoo_generator_map():
    """Positive control against an independent construction: the zoo's
    a3/a5/a7 generators are hit one-to-one (6/6, 24/24, 65/65), and zoo
    q-commutation of two generators is exactly "the union of their diagonals
    is pairwise non-crossing"."""
    for k, expect in ((1, 6), (2, 24), (3, 65)):
        gens, lat, match, cd, Z = _zoo_map(k)
        hits = sum(1 for m in match if len(m) == 1)
        used = sorted(gi for m in match for gi in m)
        assert len(lat) == expect and len(gens) == expect, (k, len(lat), len(gens))
        assert hits == expect and used == list(range(expect)), (k, hits)
        zcd = Z.cone_data()
        idx = {m[0]: j for j, m in enumerate(match)}
        diags = [[cd.chord((t, i)) for (t, i, _m) in gens[gi][0]]
                 for gi in range(expect)]
        for ga in range(expect):
            for gb in range(expect):
                if ga == gb:
                    continue
                nc = not any(_crossing(x, y) for x in diags[ga] for y in diags[gb])
                assert zcd.q_commute(idx[ga], idx[gb]) == nc, (k, ga, gb)


# ---------------------------------------------------------------------------
# the enumeration and the parity rule
# ---------------------------------------------------------------------------

def test_generator_counts():
    """6/24/65/144/280 at k = 1..5: `k(k+2)` mixed diagonals plus the
    non-crossing (even–even, odd–odd) pairs; all distinct, all in the
    centralizer (multiply-based test), and each a mixed diagonal or one
    even–even plus one odd–odd diagonal; and the count
    `k(k+2) + (k+2)·C(k+2, 3)` at k = 1..8."""
    for k, expect in zip(range(1, 6), (6, 24, 65, 144, 280)):
        U = ungauge_u1a1aodd(k)
        cd = U._G.cone_data()
        gens = U.mult_generators()
        assert len(gens) == expect and len(set(gens)) == expect, (k, len(gens))
        singles = [F for F, _e in gens if len(F) == 1]
        assert len(singles) == k * (k + 2), (k, len(singles))
        for F, e in gens:
            assert e == 0 and _by_multiply(U, (F, e)), (k, F)
            charges = sorted(cd._letter_mag((t, i)) for (t, i, _m) in F)
            assert charges in ([0], [-2, 2]), (k, F, charges)
    # the closed form k(k+2) + (k+2)·C(k+2, 3) (derived in the docstring of
    # U1A1AoddKAlg._centralizer_generators), k = 1..8
    for k in range(1, 9):
        n_gens = len(ungauge_u1a1aodd(k).mult_generators())
        assert n_gens == k * (k + 2) + (k + 2) * comb(k + 2, 3), (k, n_gens)


def test_generators_generate_balanced_window():
    """Every balanced canonical label of a window (k = 2, 3: up to four
    distinct diagonals with exponent 1; k = 2: up to two, exponents up to 3)
    is a single-term product of generators — its mixed diagonals, and its
    even–even diagonals paired off with its odd–odd ones (the diagonals of a
    label are pairwise non-crossing, so every such pairing gives
    generators)."""
    for k, r, exps, expect in ((2, 4, (1,), 181), (3, 4, (1,), 1591),
                               (2, 2, (1, 2, 3), 181)):
        U = ungauge_u1a1aodd(k)
        cd = U._G.cone_data()
        gens = set(U.mult_generators())
        n = 0
        for x in _canonical_window(cd, r, exps=exps, e_powers=(0,)):
            if not U.in_centralizer(x):
                continue
            n += 1
            letters = [g for (t, i, m) in x[0] for g in [(t, i)] * m]
            ee = sorted(g for g in letters if cd._letter_mag(g) < 0)
            oo = sorted(g for g in letters if cd._letter_mag(g) > 0)
            mx = sorted(g for g in letters if cd._letter_mag(g) == 0)
            assert len(ee) == len(oo), x
            parts = [_letter_label(g) for g in mx] + [
                (tuple(sorted((a + (1,), b + (1,)))), 0) for a, b in zip(ee, oo)]
            assert all(p in gens for p in parts), x
            lbl = U.identity()
            for p in parts:
                prod = U.multiply(lbl, p)
                assert len(prod.terms) == 1, (x, lbl, p)
                (lbl,) = prod.terms
            assert lbl == x, (x, lbl)
        assert n == expect, (k, r, n)          # the window is not vacuous


def test_generators_are_rho_stable():
    """ρ (rotation by one vertex: it swaps even and odd vertices) permutes
    the generators up to the E-power: mixed diagonals among themselves,
    (even–even, odd–odd) pairs among themselves."""
    for k in range(1, 6):
        U = ungauge_u1a1aodd(k)
        Fs = {F for F, _e in U.mult_generators()}
        assert {U.rho((F, 0))[0] for F in Fs} == Fs, k
        assert {U.rho_inverse((F, 0))[0] for F in Fs} == Fs, k


def test_parity_rule_equals_mag_on_every_letter():
    """The endpoint-parity rule equals the intrinsic `mag` (read off the
    E-commutator) on every letter at k = 1..5, and twice the last coordinate
    of `charge_formula` at k = 1..10."""
    for k in range(1, 6):
        U = ungauge_u1a1aodd(k)
        cd = U._G.cone_data()
        bad = [g for g in cd._chords if U.mag(_letter_label(g)) != cd._letter_mag(g)]
        assert not bad, (k, bad)
    for k in range(1, 11):
        cd = U1A1AoddKAlg(k).cone_data()
        n = cd._n
        assert all(2 * cd.charge_formula(*g)[n - 1] == cd._letter_mag(g)
                   for g in cd._chords), k


def test_parity_in_centralizer_equals_multiply_on_window():
    """`in_centralizer` (parity fast path) equals the multiply-based test on
    every canonical label of a window — k = 1, 2 with up to three letters,
    k = 3 with up to two, exponents 1, 2 and E-powers −1, 0, 2 — and the fast
    path fires on every one of them; off its domain it declines (None) and
    the multiply-based test is used."""
    total = neutral = 0
    for k, r in ((1, 3), (2, 3), (3, 2)):
        U = ungauge_u1a1aodd(k)
        cd = U._G.cone_data()
        for x in _canonical_window(cd, r):
            assert U._G._label_mag(x, U._E) is not None, x
            got = U.in_centralizer(x)
            assert got == _by_multiply(U, x), (k, x)
            total += 1
            neutral += got
    assert total == 645 + 8763 + 4833 and 0 < neutral < total, (total, neutral)
    U = ungauge_u1a1aodd(1)
    crossing = (((1, 0, 1), (1, 1, 1)), 0)          # {0,2} and {1,3} cross
    for x in (crossing, (((1, 0, 0),), 0), (((9, 0, 1),), 0)):
        assert U._G._label_mag(x, U._E) is None, x
    assert U.in_centralizer(crossing) == _by_multiply(U, crossing)
    assert U._G._label_mag(((), 0), (((2, 0, 1),), 0)) is None   # E not a gauge power
    assert U._G._label_mag((((1, 0, 1),), 0), ((), -3)) == -3 * -2


def test_geometric_label():
    """`geometric_label` gives the balanced multiset of diagonals: a mixed
    diagonal, or one even–even plus one odd–odd diagonal, for the
    generators; injective on them; E-powers pass through."""
    for k in range(1, 5):
        U = ungauge_u1a1aodd(k)
        cd = U._G.cone_data()
        seen = set()
        for lbl in U.mult_generators():
            curves, e = U.geometric_label(lbl)
            assert e == 0
            assert [c for c, _m in curves] == sorted(
                tuple(sorted(cd.chord((t, i)))) for (t, i, _m) in lbl[0])
            par = sorted((u % 2, v % 2) for (u, v), _m in curves)
            assert par in ([(0, 1)], [(1, 0)], [(0, 0), (1, 1)]), (k, curves)
            seen.add(curves)
        assert len(seen) == len(U.mult_generators()), k
    U = ungauge_u1a1aodd(2)
    assert U.geometric_label(U.identity()) == ((), 0)
    assert U.geometric_label(((), 3)) == ((), 3)
    assert U.geometric_label((((2, 0, 2),), -1)) == ((((0, 3), 2),), -1)


# ---------------------------------------------------------------------------
# orthonormality
# ---------------------------------------------------------------------------

def _strictly_orthonormal(U, a, b, K):
    """`I(a,b) = δ_{a,b} + O(𝖖)` on the WHOLE 𝖖⁰ coefficient (every flavour
    component, not only the χ₀ one that `verify_orthonormality` inspects),
    and no negative 𝖖-power."""
    I = U.inner_product(a, b, K)
    if any(e < 0 and not c.is_zero() for e, c in I.coeffs.items()):
        return False
    c0 = I.coeffs.get(0)
    lead = {key: v for key, v in (c0.terms.items() if c0 is not None else []) if v}
    return lead == ({(0,): 1} if a == b else {})


def test_orthonormality_identity_first():
    """The identity pairs to 1 + O(𝖖) (the control), then generator pairs:
    40/40 at k = 4 and 20/20 at k = 5, K = 4, a fixed seed, diagonal pairs
    included."""
    K = 4
    for k, n_pairs, n_diag in ((4, 40, 8), (5, 20, 4)):
        U = ungauge_u1a1aodd(k)
        one = U.identity()
        assert U.verify_orthonormality(one, one, K) and \
            _strictly_orthonormal(U, one, one, K), k
        gens = U.mult_generators()
        rng = random.Random(20260923 + k)
        pairs = [(g, g) for g in rng.sample(gens, n_diag)]
        while len(pairs) < n_pairs:
            a, b = rng.choice(gens), rng.choice(gens)
            if a != b:
                pairs.append((a, b))
        ok = sum(U.verify_orthonormality(a, b, K) and _strictly_orthonormal(U, a, b, K)
                 for a, b in pairs)
        assert ok == n_pairs, (k, ok, n_pairs)


# ---------------------------------------------------------------------------
# negative controls
# ---------------------------------------------------------------------------

def test_negative_unbalanced_labels():
    """Unbalanced labels — one even–even diagonal, one odd–odd, two even–even
    with one odd–odd — are outside the centralizer by both tests, absent from
    the generators, and refused by `geometric_label`."""
    U = ungauge_u1a1aodd(3)
    cd = U._G.cone_data()
    A1, A2, B1 = (1, 0), (1, 2), (1, 5)          # {0,2}, {2,4}; {5,7}
    assert cd._letter_mag(A1) == cd._letter_mag(A2) == -2 and cd._letter_mag(B1) == 2
    gens = set(U.mult_generators())
    for F in ((A1 + (1,),), (B1 + (1,),), (A1 + (1,), A2 + (1,), B1 + (1,)),
              (A1 + (2,), B1 + (1,))):
        x = (tuple(sorted(F)), 0)
        assert not U.in_centralizer(x) and not _by_multiply(U, x), x
        assert x not in gens
        try:
            U.geometric_label(x)
        except ValueError:
            pass
        else:
            raise AssertionError(f"geometric_label accepted unbalanced {x}")


def test_negative_off_by_one_parity_rules_fail():
    """Two off-by-one versions of the rule — the diagonal taken one vertex
    short, `{i, i+t}`, and the vertices shifted by one (even and odd swapped)
    — each disagree with `mag` on letters at every k = 1..5, so the charge
    check discriminates."""
    def rule(u, v):
        if (u - v) % 2:
            return 0
        return -2 if u % 2 == 0 else 2
    for k in range(1, 6):
        U = ungauge_u1a1aodd(k)
        cd = U._G.cone_data()
        mags = {g: U.mag(_letter_label(g)) for g in cd._chords}
        short = [g for g in cd._chords if rule(g[1], g[1] + g[0]) != mags[g]]
        shifted = [g for g in cd._chords
                   if rule(*(x + 1 for x in cd.chord(g))) != mags[g]]
        assert short and shifted, (k, len(short), len(shifted))


class _PrimitivesOnly:
    """`U1A1AoddKAlg(k)` seen through the KAlgebra primitives only: none of
    the hooks (`_label_mag`, `_centralizer_generators`, `geometric_label`)."""

    def __init__(self, G):
        self._inner = G

    def coefficient_ring(self):
        return self._inner.coefficient_ring()

    def identity(self):
        return self._inner.identity()

    def multiply(self, a, b):
        return self._inner.multiply(a, b)

    def rho(self, a):
        return self._inner.rho(a)

    def rho_inverse(self, a):
        return self._inner.rho_inverse(a)

    def trace(self, a, K=20):
        return self._inner.trace(a, K)


def test_gauged_class_without_hooks_honest_fails():
    """Over a gauged class that supplies neither the generators nor
    geometric labels, both accessors raise `NotImplementedError` rather than
    return a partial answer, and `in_centralizer` keeps the multiply-based
    test — which agrees with the hooked (parity) test on the k = 1 window."""
    U = ungauge_u1a1aodd(1)
    P = UngaugedKAlgebra(_PrimitivesOnly(U._G), ((), 1), epow=lambda lbl: lbl[1])
    for call in (P.mult_generators, lambda: P.geometric_label(P.identity())):
        try:
            call()
        except NotImplementedError:
            pass
        else:
            raise AssertionError("expected NotImplementedError")
    window = _canonical_window(U._G.cone_data(), 3)
    assert all(P.in_centralizer(x) == U.in_centralizer(x) for x in window)


# ---------------------------------------------------------------------------
# the non-simplicial case
# ---------------------------------------------------------------------------

def test_two_words_one_label():
    """Two even–even diagonals A1 = {0,2}, A2 = {2,4} and two odd–odd ones
    B1 = {5,7}, B2 = {7,9} of the decagon (k = 3), pairwise non-crossing (no
    such four exist at k = 2): the pair generators A1B1, A2B2, A1B2, A2B1 are
    all generators, and `(A1B1)·(A2B2)` and `(A1B2)·(A2B1)` are single terms
    on ONE label — the multiset {A1, A2, B1, B2}."""
    U = ungauge_u1a1aodd(3)
    cd = U._G.cone_data()
    A1, A2, B1, B2 = (1, 0), (1, 2), (1, 5), (1, 7)
    assert [tuple(sorted(cd.chord(g))) for g in (A1, A2, B1, B2)] == \
        [(0, 2), (2, 4), (5, 7), (7, 9)]
    assert not any(_crossing(cd.chord(x), cd.chord(y))
                   for x, y in itertools.combinations((A1, A2, B1, B2), 2))

    def pair(a, b):
        return (tuple(sorted((a + (1,), b + (1,)))), 0)
    gens = set(U.mult_generators())
    for p in (pair(A1, B1), pair(A2, B2), pair(A1, B2), pair(A2, B1)):
        assert p in gens, p
    w1 = U.multiply(pair(A1, B1), pair(A2, B2))
    w2 = U.multiply(pair(A1, B2), pair(A2, B1))
    assert len(w1.terms) == 1 and len(w2.terms) == 1, (w1, w2)
    (l1,), (l2,) = w1.terms, w2.terms
    target = (tuple(sorted(g + (1,) for g in (A1, A2, B1, B2))), 0)
    assert l1 == l2 == target, (l1, l2)
    assert U.geometric_label(l1) == ((((0, 2), 1), ((2, 4), 1), ((5, 7), 1),
                                      ((7, 9), 1)), 0)
    # the configuration first appears at k = 3
    cd2 = U1A1AoddKAlg(2).cone_data()
    EE = [g for g in cd2._chords if cd2._letter_mag(g) < 0]
    OO = [g for g in cd2._chords if cd2._letter_mag(g) > 0]
    assert not any(
        not any(_crossing(cd2.chord(x), cd2.chord(y))
                for x, y in itertools.combinations(a + b, 2))
        for a in itertools.combinations(EE, 2) for b in itertools.combinations(OO, 2))


if __name__ == "__main__":
    # positive controls first: an AssertionError here aborts the run
    test_control_documented_k1_facts()
    print("control: k=1 documented facts (type-1 charged, type-2 neutral; mag agrees) — OK")
    test_control_k1_generators_are_hexagon_named_ones()
    print("control: k=1 generators == HexagonKAlg's six named generators — OK")
    test_control_zoo_generator_map()
    print("control: zoo a3/a5/a7 generator map 6/6, 24/24, 65/65, q-commutation == non-crossing — OK")
    test_generator_counts()
    print("generator counts 6/24/65/144/280 at k=1..5, all in the centralizer — OK")
    test_generators_generate_balanced_window()
    print("balanced window labels = single-term products of generators (181, 1591, 181) — OK")
    test_generators_are_rho_stable()
    print("generator set rho-stable up to the E-power, k=1..5 — OK")
    test_parity_rule_equals_mag_on_every_letter()
    print("parity rule == mag on every letter (k=1..5), == 2*charge[n-1] (k=1..10) — OK")
    test_parity_in_centralizer_equals_multiply_on_window()
    print("parity in_centralizer == multiply-based on 14,241 window labels — OK")
    test_geometric_label()
    print("geometric_label: balanced multisets, injective on generators — OK")
    test_orthonormality_identity_first()
    print("orthonormality: identity first, then 40/40 (k=4) and 20/20 (k=5) at K=4 — OK")
    test_negative_unbalanced_labels()
    print("negative control: unbalanced labels rejected — OK")
    test_negative_off_by_one_parity_rules_fail()
    print("negative control: off-by-one parity rules fail the charge check — OK")
    test_gauged_class_without_hooks_honest_fails()
    print("gauged class without the hooks: NotImplementedError; in_centralizer by multiply agrees — OK")
    test_two_words_one_label()
    print("(A1B1)(A2B2) and (A1B2)(A2B1): one label (non-simplicial cone) — OK")
    print("\nAll ungauged A-odd geometry tests passed.")
