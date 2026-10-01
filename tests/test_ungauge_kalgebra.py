"""Tests for the general U(1) ungauger `UngaugedKAlgebra` (centralizer of an
electric generator E; magnetic charge = the E-commutator fq-power)."""
import sys
import os
import itertools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ungauge_kalgebra import ungauge_u1a1aodd
from u1a1aodd_kalg import E_GEN, E_INV


def _gen_label(g):
    if g == E_GEN:
        return ((), 1)
    if g == E_INV:
        return ((), -1)
    return (((g[0], g[1], 1),), 0)


def test_centralizer_is_long_chords_plus_E():
    """For k=1 the centralizer of E = the 3 long chords (mag=0) + E^{±1};
    the 6 short chords (mag=±2 fq-power) are excluded."""
    U = ungauge_u1a1aodd(1)
    cd = U._G.cone_data()
    mags = {}
    cent = []
    for g in cd.mult_gens():
        lbl = _gen_label(g)
        mags[g] = U.mag(lbl)
        if U.in_centralizer(lbl):
            cent.append(g)
    # shorts (type 1) excluded, longs (type 2) + E included
    assert all(mags[(1, i)] != 0 for i in range(6)), mags
    assert all(mags[(2, i)] == 0 for i in range(3)), mags
    assert set(cent) == {(2, 0), (2, 1), (2, 2), E_GEN, E_INV}, cent


def test_multiply_closed_in_centralizer():
    """Z(E) is a subalgebra: products of centralizer labels stay in Z(E)."""
    U = ungauge_u1a1aodd(1)
    cent = [(((2, 0, 1),), 0), (((2, 1, 1),), 0), (((2, 2, 1),), 0),
            ((), 1), ((), -1)]
    for a, b in itertools.product(cent, repeat=2):
        for L in U.multiply(a, b).terms:
            assert U.in_centralizer(L), (a, b, L)


def test_long_chord_plucker():
    """L_{2,0}·L_{2,1} = 1 + fq^{-1}·(neutral short product) — a Plücker
    that lands in the centralizer (the short pair is magnetically neutral)."""
    U = ungauge_u1a1aodd(1)
    prod = U.multiply((((2, 0, 1),), 0), (((2, 1, 1),), 0))
    assert U.identity() in prod.terms
    for L in prod.terms:
        assert U.in_centralizer(L)


def test_kalgebra_axioms():
    """Valid KAlgebra on the E-free **matter** basis: bar-involution,
    ρ-automorphism, identity, and orthonormality.  (E^m = z^{−m}·𝟙 is the
    fugacity, not an independent basis element — see test below — so it is
    excluded from the orthonormality check.)"""
    U = ungauge_u1a1aodd(1)
    matter = [U.identity(), (((2, 0, 1),), 0), (((2, 1, 1),), 0),
              (((2, 2, 1),), 0)]
    full = matter + [((), 1), ((), -1)]
    assert U.verify_identity_in_basis()
    assert U.verify_rho_fixes_identity()
    assert all(U.verify_rho_inverse(a) for a in full)
    assert all(U.verify_bar_involution(a, b)
               for a, b in itertools.product(full, repeat=2))
    assert all(U.verify_rho_is_automorphism(a, b)
               for a, b in itertools.product(full, repeat=2))
    # orthonormality on the genuine (E-free) matter basis
    assert all(U.verify_orthonormality(a, b, 5)
               for a, b in itertools.product(matter, repeat=2))


def test_E_is_the_fugacity_and_measure_factor():
    """E is central → it is the U(1) flavour fugacity z^{-1}·𝟙 (so `Tr(E) =
    z^{-1}·Tr(𝟙)`), and the ungauged flavoured trace equals the gauge-graded
    Wilson-line sum divided by the vector-multiplet measure (fq²;fq²)²_∞ —
    matching the independently-built `A1AoddToEvenRGKAlgebra(1).Tr(1)`."""
    from zplus_ring import AbelianZPlusRing
    from a1aodd_to_even_rgkalgebra import A1AoddToEvenRGKAlgebra
    U = ungauge_u1a1aodd(1)
    assert isinstance(U.coefficient_ring(), AbelianZPlusRing)

    def zdict(tr):
        out = {}
        src = tr.coeffs if hasattr(tr, "coeffs") else tr._coeffs
        for e, rc in src.items():
            d = {}
            if hasattr(rc, "terms"):
                for zb, c in rc.terms.items():
                    zp = zb[0] if isinstance(zb, tuple) else 0
                    v = sum(c.terms.values()) if hasattr(c, "terms") else c
                    d[zp] = d.get(zp, 0) + v
            d = {k: v for k, v in d.items() if v}
            if d:
                out[e] = d
        return out

    K = 8
    mine = zdict(U.trace(U.identity(), K))
    A = A1AoddToEvenRGKAlgebra(1)
    ref = zdict(A.trace(A.identity(), K))
    assert all(mine.get(e, {}) == ref.get(e, {}) for e in range(0, K + 1)), \
        (mine, ref)
    # Tr(E) = z^{-1} · Tr(identity)
    trI = zdict(U.trace(U.identity(), K))
    trE = zdict(U.trace(((), 1), K))
    assert trE.get(0) == {-1: 1} and trI.get(0) == {0: 1}


def test_k2_trace_matches_a1aodd_to_even():
    """General applicability: at k=2 the ungauged trace reproduces
    `A1AoddToEvenRGKAlgebra(2).Tr(1)` (the μ-flavoured [A1,A5] index)."""
    from a1aodd_to_even_rgkalgebra import A1AoddToEvenRGKAlgebra

    def zdict(tr):
        out = {}
        src = tr.coeffs if hasattr(tr, "coeffs") else tr._coeffs
        for e, rc in src.items():
            d = {}
            if hasattr(rc, "terms"):
                for zb, c in rc.terms.items():
                    zp = zb[0] if isinstance(zb, tuple) else 0
                    v = sum(c.terms.values()) if hasattr(c, "terms") else c
                    d[zp] = d.get(zp, 0) + v
            d = {k: v for k, v in d.items() if v}
            if d:
                out[e] = d
        return out

    K = 8
    U = ungauge_u1a1aodd(2)
    mine = zdict(U.trace(U.identity(), K))
    A = A1AoddToEvenRGKAlgebra(2)
    ref = zdict(A.trace(A.identity(), K))
    assert all(mine.get(e, {}) == ref.get(e, {}) for e in range(0, K + 1)), \
        (mine, ref)


def test_trace_window_follows_the_label_E_power():
    """The ungauging sum over n is windowed around the label's OWN E-power:
    Tr(E^e L) = z^{-e} Tr(L) for every e, however large against K.  Until
    2026-09-23 the window sat at n = 0, so Tr(E^2) at K = 0 came back 0 (the
    true value is z^{-2}) and every label with |E-power| near K + 1 lost terms."""
    from ungauge_kalgebra import ungauge_u1a1aodd
    U = ungauge_u1a1aodd(2)

    def zd(t):
        out = {}
        for q, r in t.coeffs.items():
            for key, c in r.terms.items():
                if c:
                    out[(q, key[0])] = int(c)
        return out
    assert zd(U.trace(((), 2), 0)) == {(0, -2): 1}
    assert zd(U.trace(((), -3), 1)) == {(0, 3): 1}
    L = (((2, 0, 1),), 0)
    K = 6
    base = zd(U.trace(L, K))
    for e in (-9, -4, -1, 3, 8):
        shifted = zd(U.trace((((2, 0, 1),), e), K))
        assert shifted == {(q, z - e): c for (q, z), c in base.items()}, (e, shifted)


def test_lift_follows_the_trace(K=8, N=20):
    """The flavour-lift coordinate `r_label_decompose((F, e)) = ((F, 0), (−e,))`:
    the section is the E-free label and the U(1) weight is minus the E-power,
    the sign the trace fixes (`Tr((F, e)) = z^{−e}·Tr((F, 0))`, so `E` is
    `z^{−1}`).  At k = 1, 2, 3 on N labels — the generators and terms of
    generator products, each with a random E-power in [−3, 3]:
    `r_label_compose` inverts it, the key is a single irrep, `to_R_form`
    round-trips, `E^e·L_{(F,0)} = L_{(F,e)}` (the lift is an algebra
    statement, not only a trace one), and the trace is R-linear for it,
    `Tr(L_a) = χ_key·Tr(L_section)`, with `Tr(E²) = z^{−2}·Tr(1)`.  Negative
    control: the opposite sign `(e,)` fails on every such label whose section
    has a nonzero trace.  Until 2026-09-23 the section was the label itself,
    so every E-power was its own section."""
    import random
    from kalgebra import Element
    for k in (1, 2, 3):
        U = ungauge_u1a1aodd(k)
        R = U.coefficient_ring()
        one = U.trace(U.identity(), K)
        assert U.trace(((), 2), K) == one * R.basis_element((-2,)), k
        assert U.trace(((), 2), K) != one * R.basis_element((2,)), k
        rng = random.Random(1000 + k)
        gens = U.mult_generators()
        Fs = [()] + [g[0] for g in gens]
        for _ in range(N):
            terms = sorted(U.multiply(rng.choice(gens), rng.choice(gens)).terms, key=repr)
            if terms:
                Fs.append(rng.choice(terms)[0])
        tested = failed = 0
        for _ in range(N):
            F, e = rng.choice(Fs), rng.randint(-3, 3)
            a = (F, e)
            sec, key = U.r_label_decompose(a)
            assert sec == (F, 0) and key == (-e,), (k, a, sec, key)
            assert U.r_label_compose(sec, key) == a, (k, a)
            assert U.verify_section_is_single_irrep(a), (k, a)
            La = Element.basis(a)
            assert U.from_R_form(U.to_R_form(La)) == La, (k, a)
            assert U.multiply(((), e), sec) == La, (k, a)
            tr, base = U.trace(a, K), U.trace(sec, K)
            assert tr == base * R.basis_element(key), (k, a)
            if e and not base.is_zero():
                tested += 1
                failed += tr != base * R.basis_element((e,))
        assert tested and failed == tested, (k, failed, tested)


class _HiddenGaugeCharge:
    """The gauged class with its `_label_gauge_charge` hook hidden (everything
    else forwarded): the ungauger then windows by the E-power alone."""

    def __init__(self, G):
        self._inner = G

    def __getattr__(self, name):
        if name == "_label_gauge_charge":
            raise AttributeError(name)
        return getattr(self._inner, name)


def test_trace_window_covers_the_gauge_charge():
    """The gauge-charge sum concentrates where the TOTAL gauge charge g + n
    vanishes (g the label's gauge charge, `U1A1AoddKAlg._label_gauge_charge`),
    so the window covers |g + n| <= K + 1 as well as the E-power window.
    Known answer (2026-09-26): at k = 3 the cube of the pair generator
    {(1, 8), (3, 7)} (gauge charge 6) has the term -fq^3 z^-6 at K = 4 — in
    the sum over a wide window (|n| <= 24), and in the finite zoo's a7, whose
    trace is its own Layer-1 reduction over its exported cone table (through
    `aodd_seeds.kalgebra_iso`).  On the powers 2..4 of the two pair
    generators that lost terms, the class equals the wide sum at K = 1..7.
    Negative control: with the hook hidden the E-power window alone drops the
    term (the behaviour before this date)."""
    import aodd_seeds as S
    from kalgebra import Element
    from laurent_poly import LaurentPoly
    from ungauge_kalgebra import UngaugedKAlgebra

    def zd(t, K):
        return {(q, key[0]): int(v) for q, r in t.coeffs.items()
                for key, v in r.terms.items() if v and q <= K}

    U = ungauge_u1a1aodd(3)
    G = U._G
    inv = U._inv_measure

    def wide(F, K, W=24):
        acc = {}
        meas = inv(K)
        for n in range(-W, W + 1):
            for q, rc in G.trace((F, n), K).coeffs.items():
                c = sum(rc.terms.values())
                for fe, fc in meas.items():
                    if c and q + fe <= K:
                        acc[(q + fe, n)] = acc.get((q + fe, n), 0) + c * fc
        return {k: v for k, v in acc.items() if v}

    cube = ((1, 8, 3), (3, 7, 3))
    assert G._label_gauge_charge((cube, 0)) == 6
    assert G._label_gauge_charge((cube, -2)) == 4
    got = zd(U.trace((cube, 0), 4), 4)
    assert got.get((3, -6)) == -1 and got == wide(cube, 4), got
    iso = S.kalgebra_iso("a7")
    (zl, _c), = iso.inverse(Element({(cube, 0): LaurentPoly.one()})).terms.items()
    assert zd(iso.source.trace(zl, 4), 4) == got
    for pair in (((1, 8), (3, 7)), ((1, 9), (3, 8))):
        for a in (2, 3, 4):
            F = tuple(sorted((t, i, a) for (t, i) in pair))
            for K in range(1, 8):
                assert zd(U.trace((F, 0), K), K) == wide(F, K), (F, K)
    P = UngaugedKAlgebra(_HiddenGaugeCharge(G), ((), 1), epow=lambda lbl: lbl[1])
    old = zd(P.trace((cube, 0), 4), 4)
    assert (3, -6) not in old and old != got, old


if __name__ == "__main__":
    test_centralizer_is_long_chords_plus_E()
    print("centralizer(E) = long chords + E (shorts excluded) — OK")
    test_multiply_closed_in_centralizer()
    print("multiply closed in centralizer — OK")
    test_long_chord_plucker()
    print("long-chord Plücker lands in centralizer — OK")
    test_kalgebra_axioms()
    print("KAlgebra axioms (bar, ρ, orthonormality on matter) — OK")
    test_E_is_the_fugacity_and_measure_factor()
    print("E = fugacity; trace == A1AoddToEven via (fq²;fq²)²_∞ measure — OK")
    test_k2_trace_matches_a1aodd_to_even()
    print("k=2: ungauged trace == A1AoddToEven(2) (μ-flavoured [A1,A5]) — OK")
    test_trace_window_follows_the_label_E_power()
    print("trace window follows the label's E-power: Tr(E^e L) = z^-e Tr(L) — OK")
    test_lift_follows_the_trace()
    print("lift ((F,e) -> ((F,0), (-e,))) follows the trace; opposite sign fails — OK")
    test_trace_window_covers_the_gauge_charge()
    print("trace window covers the gauge charge: k=3 pair cube keeps -q^3 z^-6 at K=4 (zoo a7 agrees); hidden hook drops it — OK")
    print("\nAll ungauge_kalgebra tests passed.")
