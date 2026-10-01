"""Tests for the curve frame of the U(1)-gauged `[A_1, D_{2k+2}]`: the public
class `U1A1DevenConeKAlgebra` (`src/cone/u1a1deven_cone_kalgebra.py`;
labels `(curves, e, κ)`) and its implementation module
`src/cone/u1a1deven_geometric_frame.py`.

Every comparison here is against the flow `U1A1DevenViaDoddRG(k)` the frame
was derived from (it stays on the spine), through the closed-form bijection of
labels `(curves, e, κ) ↔ ((word, κ), (c0, e))`.  The comparisons against the
retired ray-keyed tables (the archived tree),
including the ρ-equivariant isomorphism of their rays with the curves, are the
legacy regression the suite in the source repository.

(A) the curve combinatorics, for any `n`: `n(n−1)` curves and `C(2n−2, n−1)`
    maximal cones of `n − 1` curves (`C(4k+2, 2k+1)` at `n = 2k+2`); the
    crossing count equals `a1dn_kalg._arc_crossings` and, at odd `n`,
    `a1dodd_cone_data`'s `arcs_cross` / `arc_puncture_crossing`; the magnetic
    charge.
(B) `Φ` and ρ in closed form, exact against `U1A1DevenViaDoddRG(k)` on every
    curve at k = 1..5, `S` read from `_ap_base` (1, 0, 5, 7, 8); a wrong `S`,
    swapped `E`-powers at the merged vertex and the drift at marked point 3
    each fail.  The full turn: POSITIVE CONTROL `ρ∘ρ⁻¹ = id` first, then
    `ρ^{2k+2}(curve) = E^{2c0}·curve` at k = 1..4, the same for every
    re-normalisation of `E`, and no rule treating every position alike
    reproduces it at k ≥ 2 (the class docstring's argument for naming the edge
    {1, 2}).
(C) the class: `Φ` of labels (`κ` included) is `RG` of the closed-form flow
    label; the section bridges round-trip; its trace calls no `RG`; the gauge
    tower `Tr(E^e)` through the transport equals Creutzig's closed form (k = 1,
    2 through 𝖖⁴⁰, k = 3 through 𝖖³², k = 4 through 𝖖²⁴, k = 5 through 𝖖²⁰;
    the neighbouring `e` differs); traces equal the transport on `RG` of the
    flow label (the flow route) and, on a few labels, the flow's own windowed
    trace; orthonormality on the curves, their squares and the terms of 40
    random products (k = 1, 2; the pairing without ρ and ρ with the wrong
    drift fail); ρ²-twisted cyclicity; the pairing equals the default
    multiply-then-trace pairing.
(D) multiplication.  POSITIVE CONTROL FIRST: the derived peel of `Φ(a)·Φ(b)`
    (skein resolutions as the candidates) reproduces the flow's product on one
    product of each kind at k = 1 and k = 2.  Then: the loop-and-curve χ₁
    daughter is the odd model's doubled-diameter one at odd n; the closed-form
    cocycle equals the peel on every non-crossing pair and on `E` (k = 1, 2,
    3); the analytic rule applies to, and equals the derived peel on, every
    crossing pair at k = 1..4 (72 / 450 / 1568 / 4050; k = 5, 8712, in
    `deep`); `multiply` equals the flow on every generator pair (curves and
    `E^{±1}`; 196 / 1024 / 3364 at k = 1, 2, 3; the derived route too at k =
    1, 2) and on 100 random composite products per k = 1, 2, 3 (with `SU(2)`
    weights), half of them also against the Φ-peel with no reducer; the bar
    involution, ρ as an automorphism, the unit law and associativity on a
    window; the held-out check at k = 4: 200 sampled products equal
    `U1A1DevenViaDoddRG(4).multiply`; `multiply` calls no `RG`; negative
    controls (the χ₁ fork dropped, `E` on the wrong edge, the two-crossing
    anchor with `#B`'s sign flipped, a missing candidate) are caught.

`deep` adds orthonormality at k = 3, the analytic rule at k = 5 and every
generator product at k = 4 against the flow:
`python3 run_tests.py` deep`.
"""
import itertools
import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                "implementations"))

from kalgebra import Element
from laurent_poly import LaurentPoly
import u1a1deven_geometric_frame as G
from u1a1deven_cone_kalgebra import U1A1DevenConeKAlgebra

_FLOWS, _ALGS = {}, {}


def _flow(k):
    if k not in _FLOWS:
        from u1a1deven_via_dodd_rg import U1A1DevenViaDoddRG
        _FLOWS[k] = U1A1DevenViaDoddRG(k)
    return _FLOWS[k]


def _alg(k):
    if k not in _ALGS:
        _ALGS[k] = U1A1DevenConeKAlgebra(k)
    return _ALGS[k]


def _lp(c):
    return c if isinstance(c, LaurentPoly) else LaurentPoly(dict(c._coeffs))


def _rg(k, flow_label):
    img = _flow(k).RG(flow_label)
    return {l: _lp(c) for l, c in img.terms.items() if not c.is_zero()}


def _one(c, e=0, kappa=0):
    return (((c, 1),), e, kappa)


def _terms(el):
    """An `Element` as `{label: {q: int}}` (zero coefficients dropped)."""
    return {l: {q: v for q, v in _lp(c)._coeffs.items() if v}
            for l, c in el.terms.items() if not c.is_zero()}


def _on_flow_labels(A, el):
    """A Z-form element of the class on the flow's labels."""
    return {A._flow_label(l): d for l, d in _terms(el).items()}


def _flow_product(k, a, b):
    """`U1A1DevenViaDoddRG(k).multiply` on the flow labels of `a`, `b`."""
    A = _alg(k)
    return _terms(_flow(k).multiply(A._flow_label(a), A._flow_label(b)))


# ===========================================================================
# (A)
# ===========================================================================

def test_curve_and_cone_counts():
    """`n(n−1)` curves; `C(2n−2, n−1)` maximal cones, each of `n − 1` curves
    (so `C(4k+2, 2k+1)` at n = 2k+2), n = 3..10.  Positive control of the
    enumerator: at odd n it gives `A1DoddConeData`'s cone count (k = 0, 1, 2)."""
    from a1dodd_cone_data import a1dodd_cone_data
    for n in range(3, 11):
        assert len(G._curves(n)) == n * (n - 1)
        cones = G._maximal_cones(n)
        assert len(cones) == math.comb(2 * n - 2, n - 1), (n, len(cones))
        assert {len(c) for c in cones} == {n - 1}
        if n % 2 == 0:
            k = (n - 2) // 2
            assert len(cones) == math.comb(4 * k + 2, 2 * k + 1)
    for k in (0, 1, 2):
        assert len(G._maximal_cones(2 * k + 3)) == len(a1dodd_cone_data(k).cones())


def test_crossings_generalise_a1dodd():
    """The crossing count on the `2n`-gon equals `_arc_crossings` (universal
    cover) on every pair, n = 3..10; at odd n = 2k+3 it reproduces
    `arcs_cross` and `arc_puncture_crossing` (count 2) through `A1DnKAlg`'s
    dictionary, k = 0..3."""
    from a1dn_kalg import _arc_crossings, _ap_to_arc
    from a1dodd_cone_data import arcs_cross, arc_puncture_crossing
    for n in range(3, 11):
        cs = G._curves(n)
        for c, d in itertools.product(cs, cs):
            assert G._curve_crossings(c, d, n) == _arc_crossings(c, d, n), (n, c, d)
    for k in range(4):
        n = 2 * k + 3
        gens = [(a, p, i) for a in range(1, k + 2) for p in (0, 1) for i in range(n)]
        for g, h in itertools.permutations(gens, 2):
            cg, ch = _ap_to_arc(g, k), _ap_to_arc(h, k)
            assert arcs_cross(g, h, k) == G._curves_cross(cg, ch, n)
            if arcs_cross(g, h, k):
                assert arc_puncture_crossing(g, h, k) == (G._curve_crossings(cg, ch, n) == 2)


def test_magnetic_charge():
    """Even n: charge 0 on odd ℓ, else −1 when the endpoints have the parity of
    the merged vertex and +1 otherwise; the rotation flips it on even ℓ;
    undefined (raises) at odd n.  The class's `_label_mag` (the charge
    `UngaugedKAlgebra` reads) equals the `E`-commutator exponent `−2·c0` on
    every curve, k = 1, 2, 3."""
    for n in (4, 6, 8):
        for c in G._curves(n):
            x, l = c
            q = G._magnetic_charge(c, n, 2)
            if l % 2:
                assert q == 0
            else:
                assert q == (-1 if x % 2 == 0 else 1)
                assert G._magnetic_charge(G._rotate_curve(c, n), n, 2) == -q
    try:
        G._magnetic_charge((0, 2), 5, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("odd n accepted")
    for k in (1, 2, 3):
        A = _alg(k)
        E = ((), 1, 0)
        for c in G._curves(2 * k + 2):
            lab = _one(c)
            (l1, c1), = A.multiply(E, lab).terms.items()
            (l2, c2), = A.multiply(lab, E).terms.items()
            assert l1 == l2
            (q1,), (q2,) = c1._coeffs, c2._coeffs
            assert q1 - q2 == A._label_mag(lab, E) == -2 * G._charge(c, k), (k, c)


# ===========================================================================
# (B)
# ===========================================================================

def test_dressing_start_read_from_ap_base():
    """S = 1, 0, 5, 7, 8 at k = 1..5 — the start point of the flow's letter
    L = (k, 1, 0) as a curve, L = (S, 2) — and L = Φ((0, 3))."""
    from a1dn_kalg import _ap_to_arc
    assert [G._dressing_start(k) for k in range(1, 6)] == [1, 0, 5, 7, 8]
    for k in range(1, 6):
        S = G._dressing_start(k)
        assert _ap_to_arc((k, 1, 0), k - 1) == (S, 2)
        assert G._phi_curve_terms((0, 3), k) == [((S, 2), 0, (0, 0), G._ONE)]


def _terms_dict(terms, k):
    out = {}
    for d, kap, qt, co in terms:
        lab = ((G._dodd_word(d, k), kap), qt)
        out[lab] = out.get(lab, LaurentPoly({})) + co
    return {l: c for l, c in out.items() if not c.is_zero()}


def _lowest_label(terms, k):
    (d, kap, (c0, c1), co), = [t for t in terms if t[2][1] == min(u[2][1] for u in terms)]
    return ((G._dodd_word(d, k), kap), (c0, c1))


def test_phi_equals_flow_rg_on_every_curve():
    """Φ(c) == RG(flow label of c) exactly (words, κ, charges, coefficients) on
    every curve at k = 1..5; one term off the merged vertex, two with one
    endpoint on it, four for the loop at it."""
    for k in range(1, 6):
        n = 2 * k + 2
        shapes = {}
        for c in G._curves(n):
            terms = G._phi_curve_terms(c, k)
            fl = _alg(k)._flow_label(_one(c))
            assert fl == _lowest_label(terms, k)
            assert _rg(k, fl) == G._phi_curve(c, k), (k, c)
            shapes[len(terms)] = shapes.get(len(terms), 0) + 1
        assert shapes == {1: n * (n - 1) - 4 * k - 1, 2: 4 * k, 4: 1}, (k, shapes)


def test_phi_negative_controls():
    """Each fails on some curve at every k = 1..5: a wrong S (S + 1, i.e. every
    curve of the (2k+1)-gon rotated by one), and the two terms of a curve
    with an endpoint at the merged vertex at swapped E-powers."""
    for k in range(1, 6):
        nd = 2 * k + 1
        bad_S = bad_swap = 0
        for c in G._curves(2 * k + 2):
            terms = G._phi_curve_terms(c, k)
            shifted = [(None if d is None else ((d[0] + 1) % nd, d[1]), kap, qt, co)
                       for d, kap, qt, co in terms]
            bad_S += _rg(k, _lowest_label(shifted, k)) != _terms_dict(shifted, k)
            if len(terms) == 2:
                swapped = [(d, kap, (qt[0], 1 - qt[1]), co) for d, kap, qt, co in terms]
                bad_swap += _rg(k, _lowest_label(swapped, k)) != _terms_dict(swapped, k)
        assert bad_S > 0 and bad_swap > 0, (k, bad_S, bad_swap)


def test_rho_closed_form():
    """ρ((x, ℓ)) = E^d·(x + 1, ℓ), d = −#(endpoints at marked point 1), κ
    fixed, equals the flow's ρ on every curve at k = 1..5 (at κ = 0 and 1), and
    ρ(E) = E⁻¹; the drift placed at marked point 3 fails."""
    for k in range(1, 6):
        A, Fl, n = _alg(k), _flow(k), 2 * k + 2
        bad_neg = 0
        for c in G._curves(n):
            for kappa in (0, 1):
                fl = A._flow_label(_one(c, 0, kappa))
                assert Fl.rho(fl) == A._flow_label(A.rho(_one(c, 0, kappa))), (k, c)
            fl = A._flow_label(_one(c))
            x, l = c
            d3 = -sum(1 for e in (x % n, (x + l) % n) if e == 3)
            wrong = A._flow_label(_one(((x + 1) % n, l), d3))
            bad_neg += Fl.rho(fl) != wrong
        assert bad_neg > 0, k
        assert Fl.rho((((), 0), (0, 1))) == (((), 0), (0, -1))
        assert A.rho(((), 1, 0)) == ((), -1, 0)


def _full_turn(rho, lab, n):
    for _ in range(n):
        lab = rho(lab)
    return lab


def test_full_turn_shift():
    """The class docstring's argument for naming the edge {1, 2}.

    POSITIVE CONTROL first: `ρ⁻¹∘ρ = ρ∘ρ⁻¹ = id` on every curve at `E`-powers
    −1..1, k = 1..4 (so iterating ρ is meaningful).  Then, k = 1..4:
      * the full turn `ρ^{2k+2}` returns every curve to itself times
        `E^{2c0}` (`E^{±2}` on even ℓ, `E^0` on odd ℓ);
      * the shift does not depend on where `e = 0` is put: for three random
        re-normalisations `L_c ↦ E^{f(c)}·L_c` (ρ conjugated accordingly), the
        full turn is the same;
      * a rule treating every position alike, `ρ((x, ℓ)) = E^{g(c0)}·(x+1, ℓ)`,
        reproduces it for some integers `g(−1), g(0), g(+1)` in −6..6 at k = 1
        and for none at k = 2, 3, 4 (it needs `g(+1) − g(−1) = −2/(k+1)`)."""
    for k in (1, 2, 3, 4):
        A, n = _alg(k), 2 * k + 2
        cs = G._curves(n)
        for c in cs:
            for e in (-1, 0, 1):
                lab = _one(c, e)
                assert A.rho_inverse(A.rho(lab)) == lab == A.rho(A.rho_inverse(lab))
        for c in cs:
            assert _full_turn(A.rho, _one(c), n) == _one(c, 2 * G._charge(c, k)), (k, c)
        rng = random.Random(70 + k)
        for _ in range(3):
            f = {c: rng.randint(-3, 3) for c in cs}

            def rho_f(lab):                  # ρ in the re-normalised labels
                (((c, _m),), e, kap) = lab
                (((c2, _),), e2, _k) = A.rho((((c, 1),), e + f[c], kap))
                return (((c2, 1),), e2 - f[c2], kap)
            for c in cs:
                assert _full_turn(rho_f, _one(c), n) == _one(c, 2 * G._charge(c, k))
        solvable = False
        for gm, g0, gp in itertools.product(range(-6, 7), repeat=3):
            g = {-1: gm, 0: g0, 1: gp}

            def rho_g(lab):
                (((c, _m),), e, kap) = lab
                return (((G._rotate_curve(c, n), 1),), -e + g[G._charge(c, k)], kap)
            if all(_full_turn(rho_g, _one(c), n) == _one(c, 2 * G._charge(c, k))
                   for c in cs):
                solvable = True
                break
        assert solvable == (k == 1), k


# ===========================================================================
# (C)
# ===========================================================================

def _random_monomial(k, rng, maxdeg=3, kappas=(0,)):
    n = 2 * k + 2
    comp = G._compatibility(n)
    cs = sorted(comp)
    pick = [rng.choice(cs)]
    for _ in range(rng.randint(0, maxdeg - 1)):
        pick.append(rng.choice([c for c in cs if all(c == p or c in comp[p] for p in pick)]))
    agg = {}
    for c in pick:
        agg[c] = agg.get(c, 0) + 1
    return (tuple(sorted(agg.items())), rng.randint(-2, 2), rng.choice(kappas))


def test_phi_of_monomials_and_label_maps():
    """k = 1, 2, 3, on every curve and 40 random monomials (`κ` ∈ {0, 1}):
    Φ(label) == RG(flow label), ρ agrees with the flow's on flow labels,
    ρ⁻¹ρ = 1, the inverse label map inverts the flow label, and the section
    bridges `oracle_section_of` / `native_of_oracle_section` round-trip."""
    for k in (1, 2, 3):
        rng = random.Random(k)
        A, Fl = _alg(k), _flow(k)
        labs = [_one(c) for c in G._curves(2 * k + 2)]
        labs += [_random_monomial(k, rng, kappas=(0, 1)) for _ in range(40)]
        for lab in labs:
            lab = A.canonicalise(lab)
            fl = A._flow_label(lab)
            assert {l: _lp(c) for l, c in A._phi(lab).terms.items()} == _rg(k, fl), (k, lab)
            assert Fl.rho(fl) == A._flow_label(A.rho(lab)), (k, lab)
            assert A.rho_inverse(A.rho(lab)) == lab == A.rho(A.rho_inverse(lab))
            assert A._label_of_flow_label(fl) == lab
            sec = A.oracle_section_of(lab)
            assert sec == ((fl[0][0], fl[1]), fl[0][1])
            assert A.native_of_oracle_section(sec) == lab


def test_trace_calls_no_rg():
    """The class's trace (a seed's closed form, and a label that is not a
    seed by Layer 1 onto the seeds), the transport route (the witness,
    `_transport_trace`), pairing and product (both routes) run with the
    flow's `RG` disabled."""
    from u1a1deven_via_dodd_rg import U1A1DevenViaDoddRG

    def refuse(self, *a, **kw):
        raise AssertionError("RG called")

    import u1a1deven_cone_kalgebra as M
    M._TRACE_MEMO.clear()                  # the memo is per process: start cold
    own = U1A1DevenViaDoddRG.__dict__.get("RG")
    U1A1DevenViaDoddRG.RG = refuse
    try:
        A = U1A1DevenConeKAlgebra(2)
        c = (1, 3)                                  # odd ℓ: charge 0
        assert A.trace(_one(c), 6).coeffs           # a seed: closed form
        assert A.trace(((((1, 3), 2),), 0, 0), 4).coeffs  # Layer 1
        assert A.inner_product(_one(c), _one(c), 2)[0].terms
        assert A._transport_trace(((), 1), 6).coeffs
        assert A._transport_trace(((((1, 3), 2),), 0), 4).coeffs   # the witness
        for route in ("analytic", "derived"):
            Ar = U1A1DevenConeKAlgebra(2, route=route)
            assert len(Ar.multiply(_one((0, 3)), _one((2, 6))).terms) == 3
            assert len(Ar._multiply_via_phi(_one((0, 3)), _one((2, 6))).terms) == 3
    finally:
        if own is None:
            del U1A1DevenViaDoddRG.RG
        else:
            U1A1DevenViaDoddRG.RG = own


def test_creutzig_tower():
    """POSITIVE CONTROL: Tr(E^e), e = 0..3, through the transport on Φ(E^e)
    equals Creutzig's closed form (the class's route for `E^e`): k = 1, 2
    through 𝖖⁴⁰, k = 3 through 𝖖³², k = 4 through 𝖖²⁴, k = 5 through 𝖖²⁰.
    NEGATIVE: the closed form at e + 1 differs."""
    from exact_characters import deven_gauged_xn_qn
    for k, K in ((1, 40), (2, 40), (3, 32), (4, 24), (5, 20)):
        A = _alg(k)
        for e in range(4):
            got = dict(A._transport_trace(((), e), K).coeffs)
            assert got == dict(A._to_rps(deven_gauged_xn_qn(k, e, K), K).coeffs), (k, e)
            assert got == dict(A.trace(((), e, 0), K).coeffs), (k, e)
            assert got != dict(A._to_rps(deven_gauged_xn_qn(k, e + 1, K), K).coeffs)


def _neutral_labels(k, rng, count):
    """Charge-0 labels: odd-ℓ curves and balanced pairs, at E-powers −1..1."""
    n = 2 * k + 2
    cs = G._curves(n)
    ch = {c: G._charge(c, k) for c in cs}
    singles = [c for c in cs if ch[c] == 0]
    pairs = [(a, b) for a in cs for b in cs
             if ch[a] == 1 and ch[b] == -1 and not G._curves_cross(a, b, n)]
    out = []
    for _ in range(count):
        e = rng.choice((-1, 0, 1))
        if rng.random() < 0.5:
            out.append((((rng.choice(singles), rng.choice((1, 2))),), e, 0))
        else:
            a, b = rng.choice(pairs)
            out.append((tuple(sorted(((a, 1), (b, 1)))), e, 0))
    return out


def test_traces_equal_flow_route():
    """The class's trace (transport on Φ(label), no `RG`) equals the transport
    on `RG` of the flow label (the flow route, `DevenTraceTransport.trace`):
    20 / 16 / 10 charge-0 labels at k = 1 / 2 / 3, through 𝖖⁸ / 𝖖⁸ / 𝖖⁶.  Some
    traces vanish through that order (e.g. the loop at 1 times (2, 2) at E¹,
    k = 1); at least half of each sample does not.  A label at `κ = 2` is `χ₂`
    times its `κ = 0` trace."""
    from u1a1deven_trace_transport import DevenTraceTransport
    for k, count, K in ((1, 20, 8), (2, 16, 8), (3, 10, 6)):
        rng = random.Random(10 + k)
        A = _alg(k)
        T = DevenTraceTransport(k)                  # a fresh transport: own memos
        nonzero = 0
        for lab in _neutral_labels(k, rng, count):
            a = A.trace(lab, K)
            b = A._to_rps(T.trace(A._flow_label(lab), K), K)
            assert dict(a.coeffs) == dict(b.coeffs), (k, lab)
            nonzero += bool(a.coeffs)
        assert 2 * nonzero >= count, (k, nonzero)
        R = A.coefficient_ring()
        lab = (lab[0], lab[1], 2)
        chi2 = A._times_chi(A.trace((lab[0], lab[1], 0), K), 2, K)
        assert dict(A.trace(lab, K).coeffs) == dict(chi2.coeffs)
        assert R.basis_element(2) != R.one()


def test_traces_equal_flow_windowed_trace():
    """On the vacuum, a charge-0 curve and a balanced pair of curves at k = 1
    and on one label at k = 2, the trace equals the flow's own windowed trace
    (`U1A1DevenViaDoddRG.trace`, the generic RG route) through 𝖖³; the pair
    `((0, 4), (1, 2))` is the loop at 0 (charge −1) and the curve from 1 to 3
    (charge +1), which do not cross."""
    for k, labs in ((1, [((), 0, 0), _one((1, 3)), ((((0, 4), 1), ((1, 2), 1)), 0, 0)]),
                    (2, [_one((0, 3), 1)])):
        A, Fl = _alg(k), _flow(k)
        for lab in labs:
            assert A._magnetic_charge(A.canonicalise(lab)) == 0, lab
            assert A.trace(lab, 3) == Fl.trace(A._flow_label(lab), 3), (k, lab)


def _is_orthonormal(A, a, b, K=1, rho=None):
    if rho is None:
        I = A.inner_product(a, b, K)
    else:
        I = A._trace_of_product(rho(a), b, K)
    if any(e < 0 and not c.is_zero() for e, c in I.coeffs.items()):
        return False
    one = I[0].terms.get(A.coefficient_ring().one_basis(), 0)
    return one == (1 if A.canonicalise(a) == A.canonicalise(b) else 0)


def _orthonormality_sample(k, seed):
    """Identity, E^{±1}, the curves, their squares, and the terms of 40 random
    products of two crossing curves at random E-powers (the class's own
    products; the products themselves are checked against the flow in (D))."""
    rng = random.Random(seed)
    A = _alg(k)
    n = 2 * k + 2
    cs = G._curves(n)
    comp = G._compatibility(n)
    base = [A.identity(), ((), 1, 0), ((), -1, 0)] + [_one(c) for c in cs] \
        + [(((c, 2),), 0, 0) for c in cs]
    crossing = [(c, d) for c in cs for d in cs if c != d and d not in comp[c]]
    products = []
    for _ in range(40):
        c, d = rng.choice(crossing)
        el = A.multiply(_one(c, rng.choice((-1, 0, 1))), _one(d, rng.choice((-1, 0, 1))))
        terms = [l for l, co in el.terms.items() if not co.is_zero()]
        assert len(terms) >= 2
        products.append(terms)
    return base, products


def _orthonormality(k, all_pairs):
    A = _alg(k)
    base, products = _orthonormality_sample(k, 100 + k)
    terms = sorted({t for ts in products for t in ts})
    for x in base + terms:
        assert _is_orthonormal(A, x, x), (k, x)
    pairs = set()
    for ts in products:
        pairs.update((a, b) for a in ts for b in ts if a != b)
    if all_pairs:
        pairs.update((a, b) for a in base for b in base if a != b)
    else:
        rng = random.Random(k)
        pairs.update((rng.choice(base), rng.choice(base)) for _ in range(300))
        pairs = {(a, b) for a, b in pairs if a != b}
    rng = random.Random(7 * k)
    for _ in range(200):
        a, b = rng.choice(terms), rng.choice(base)
        pairs.update(((a, b), (b, a)))
    for a, b in sorted(pairs):
        assert _is_orthonormal(A, a, b), (k, a, b)
    return len(base) + len(terms), len(pairs)


def test_orthonormality():
    """I(a, b) = δ + O(𝖖), no negative 𝖖-power, the pairing taken in the
    auxiliary algebra: on the identity, E^{±1}, the curves, their squares and
    the terms of 40 random products (every diagonal pair; every ordered pair
    of distinct terms of one product; 400 pairs between product terms and the
    rest; at k = 1 every ordered pair of the rest, at k = 2 a sample of 300)."""
    _orthonormality(1, all_pairs=True)
    _orthonormality(2, all_pairs=False)


def test_orthonormality_negative_controls():
    """k = 1, 2: the pairing without ρ, `Tr(L_a·L_a)`, fails the diagonal on
    every curve, the charge-0 ones included (4 / 12 of them); with ρ's drift
    at marked point 3 instead of 1 it fails exactly on the curves whose ρ-image
    that changes (6 / 14 of the 12 / 30)."""
    for k, n_fail in ((1, 6), (2, 14)):
        _orthonormality_negative_controls(k, n_fail)


def _orthonormality_negative_controls(k, n_fail):
    A = _alg(k)
    n = A.n

    def rho_wrong(label):
        curves, e, kap = A.canonicalise(label)
        new, drift = [], 0
        for (x, l), m in curves:
            new.append(((((x + 1) % n), l), m))
            drift += m * -sum(1 for v in (x % n, (x + l) % n) if v == 3)
        return (tuple(sorted(new)), -e + drift, kap)

    labs = [_one(c) for c in G._curves(n)]
    assert all(_is_orthonormal(A, a, a) for a in labs)
    assert not any(_is_orthonormal(A, a, a, rho=lambda x: x) for a in labs)
    changed = [a for a in labs if A.rho(a) != rho_wrong(a)]
    assert len(changed) == n_fail, (k, len(changed))
    assert not any(_is_orthonormal(A, a, a, rho=rho_wrong) for a in changed)


def test_rho2_twisted_cyclicity():
    """Tr(L_a·L_b) == Tr(ρ²(L_b)·L_a) through 𝖖⁶ (products in the auxiliary
    algebra) on 30 pairs of total magnetic charge 0, k = 1, 2."""
    for k in (1, 2):
        A = _alg(k)
        rng = random.Random(30 + k)
        base, products = _orthonormality_sample(k, 100 + k)
        pool = base + sorted({t for ts in products for t in ts})
        done = 0
        while done < 30:
            a, b = rng.choice(pool), rng.choice(pool)
            if A._magnetic_charge(a) + A._magnetic_charge(b):
                continue
            lhs = A._trace_of_product(a, b, 6)
            rhs = A._trace_of_product(A.rho(A.rho(b)), a, 6)
            assert dict(lhs.coeffs) == dict(rhs.coeffs), (k, a, b)
            done += 1


def test_pairing_equals_default_route():
    """`inner_product` (the product taken in the auxiliary algebra) equals the
    contract's default `Tr(multiply(ρ(a), b))` (`verify_inner_product_consistent`)
    on 12 pairs of equal charge per k = 1, 2, through 𝖖³, `κ` included (the
    default route traces every term of the product; about 3x slower)."""
    for k in (1, 2):
        A = _alg(k)
        rng = random.Random(40 + k)
        done = 0
        while done < 12:
            a = _random_monomial(k, rng, kappas=(0, 1))
            b = _random_monomial(k, rng, kappas=(0, 1))
            if A._magnetic_charge(a) != A._magnetic_charge(b):
                continue
            assert A.verify_inner_product_consistent(a, b, 3), (k, a, b)
            done += 1


# ===========================================================================
# (D) multiplication
# ===========================================================================

def _generators(k):
    return [_one(c) for c in G._curves(2 * k + 2)] + [((), 1, 0), ((), -1, 0)]


# one crossing; two crossings (four terms); a loop and a curve; a curve and a
# loop; two loops (with E²) — at k = 1 (square) and k = 2 (hexagon)
_CONTROL_PAIRS = {
    1: [((0, 2), (1, 2)), ((0, 3), (2, 3)), ((1, 4), (0, 2)), ((0, 2), (1, 4)),
        ((1, 4), (2, 4))],
    2: [((0, 3), (1, 4)), ((0, 4), (3, 4)), ((2, 6), (0, 3)), ((0, 3), (2, 6)),
        ((1, 6), (2, 6))],
}


def test_derived_peel_positive_control():
    """POSITIVE CONTROL, before anything below is trusted: the derived peel of
    `Φ(a)·Φ(b)` (candidates the skein resolutions; `E`-power, χ and
    coefficient read off; exact zero residual) reproduces the flow's product
    `U1A1DevenViaDoddRG(k).multiply` on the flow labels, on one product of
    each kind, at `E`-powers 0 and (1, −2), k = 1, 2; the kinds are the ones
    named."""
    kinds = ["one crossing", "two crossings", "loop and curve", "curve and loop",
             "two loops"]
    for k, pairs in _CONTROL_PAIRS.items():
        A = _alg(k)
        for (g, h), kind in zip(pairs, kinds):
            assert G._resolutions(g, h, k)[0] == kind, (k, g, h)
            for ea, eb in ((0, 0), (1, -2)):
                a, b = _one(g, ea), _one(h, eb)
                got = A._multiply_via_phi(a, b)
                assert _on_flow_labels(A, got) == _flow_product(k, a, b), (k, g, h, ea, eb)
                assert len(got.terms) >= 2


def test_loop_fork_is_the_odd_model():
    """The χ₁ daughter of a loop crossing a curve, written as the two arcs
    from the curve's endpoints to the loop's marked point (`_loop_fork_arcs`,
    any n), is — at odd n = 2k + 3 — the χ₁ word of `a1dodd_skein`'s
    doubled-diameter puncture loop, through `A1DnKAlg`'s dictionary, on every
    diameter × non-diameter crossing of `a1dodd_cone_data(k)`: 6 / 60 / 210 /
    504 at k = 0..3."""
    from a1dodd_skein import _doubled_diam_chi_words
    from a1dodd_cone_data import (arcs_cross, arc_puncture_crossing, _is_diameter,
                                  _arcs_to_multgen)
    from a1dn_kalg import _ap_to_arc
    for k, want in ((0, 6), (1, 60), (2, 210), (3, 504)):
        n = 2 * k + 3
        inv = _arcs_to_multgen(k)
        gens = [(a, p, i) for a in range(1, k + 2) for p in (0, 1) for i in range(n)]
        seen = 0
        for g, h in itertools.product(gens, gens):
            if (g == h or not arcs_cross(g, h, k) or not arc_puncture_crossing(g, h, k)
                    or _is_diameter(g, k) == _is_diameter(h, k)):
                continue
            diam, other = (g, h) if _is_diameter(g, k) else (h, g)
            model = {tuple(sorted(_ap_to_arc(f, k) for f in w))
                     for w in _doubled_diam_chi_words(diam, other, k, inv)}
            curves, _e = G._fold_arcs(G._loop_fork_arcs(_ap_to_arc(diam, k),
                                                         _ap_to_arc(other, k), n), n)
            assert model == {tuple(sorted(c for c, m in curves for _ in range(m)))}, (k, g, h)
            seen += 1
        assert seen == want, (k, seen)


def test_cocycle_closed_form():
    """The closed-form cocycle (the `A1Dodd` arc cocycle of the lowest terms;
    `c(curve, E^{±1}) = ±c0`) equals the peel's single term on every ordered
    non-crossing pair of curves (72 / 450 / 1568 with the diagonal) and on
    every curve against `E` and `E⁻¹` on either side, k = 1, 2, 3."""
    for k, want in ((1, 72), (2, 450), (3, 1568)):
        A = _alg(k)
        cs = G._curves(2 * k + 2)
        letters = {c: _one(c) for c in cs}
        letters.update({G._E_POS: ((), 1, 0), G._E_NEG: ((), -1, 0)})
        seen = 0
        for g, h in itertools.product(letters, letters):
            if g not in G._E_LETTERS and h not in G._E_LETTERS and \
                    G._curves_cross(g, h, 2 * k + 2):
                continue
            el = A._multiply_via_phi(letters[g], letters[h])
            (lab, co), = el.terms.items()
            (q, v), = _lp(co)._coeffs.items()
            assert v == 1 and lab[2] == 0, (k, g, h)
            assert q == G._frame_cocycle(g, h, k), (k, g, h)
            seen += g not in G._E_LETTERS and h not in G._E_LETTERS
        assert seen == want, (k, seen)


_KIND_COUNTS = {
    1: {"one crossing": 32, "two crossings": 4, "loop and curve": 12,
        "curve and loop": 12, "two loops": 12},
    2: {"one crossing": 240, "two crossings": 60, "loop and curve": 60,
        "curve and loop": 60, "two loops": 30},
    3: {"one crossing": 896, "two crossings": 280, "loop and curve": 168,
        "curve and loop": 168, "two loops": 56},
    4: {"one crossing": 2400, "two crossings": 840, "loop and curve": 360,
        "curve and loop": 360, "two loops": 90},
    5: {"one crossing": 5280, "two crossings": 1980, "loop and curve": 660,
        "curve and loop": 660, "two loops": 132},
}


def _analytic_vs_derived(k):
    A = _alg(k)
    n = 2 * k + 2
    kinds = {}
    for g, h in itertools.product(G._curves(n), G._curves(n)):
        if not G._curves_cross(g, h, n):
            continue
        kind = G._resolutions(g, h, k)[0]
        kinds[kind] = kinds.get(kind, 0) + 1
        an = G._analytic_product(g, h, k)
        assert an is not None, (k, g, h, kind)            # the rule applies
        assert an == A._cross_derived(g, h), (k, g, h, kind)
    assert kinds == _KIND_COUNTS[k], (k, kinds)


def test_analytic_rule_equals_derived_peel():
    """The analytic rule applies to every ordered crossing pair and equals the
    derived peel there: 72 / 450 / 1568 / 4050 pairs at k = 1..4, by kind as
    in `_KIND_COUNTS` (every coefficient a single power of `𝖖`, `χ` at most
    `χ₁`)."""
    for k in (1, 2, 3, 4):
        _analytic_vs_derived(k)


def test_products_equal_flow_every_generator_pair():
    """`multiply` equals the flow's product on the flow labels on every
    ordered pair of generators (the curves and `E^{±1}`): 196 / 1024 / 3364
    pairs at k = 1, 2, 3 (the flow takes 0.1 / 0.5 / 1.5 s); the derived route
    (`route="derived"`) equals the analytic one at k = 1, 2; the analytic rule
    served every crossing pair; the coefficients are integral `LaurentPoly`
    (Z-form)."""
    for k, want in ((1, 196), (2, 1024), (3, 3364)):
        A = _alg(k)
        gens = _generators(k)
        assert len(gens) ** 2 == want
        for a, b in itertools.product(gens, gens):
            P = A.multiply(a, b)
            assert all(isinstance(c, LaurentPoly) for c in P.terms.values())
            assert _on_flow_labels(A, P) == _flow_product(k, a, b), (k, a, b)
        assert not A._analytic_not_applied
        if k < 3:
            Ad = U1A1DevenConeKAlgebra(k, route="derived")
            for a, b in itertools.product(gens, gens):
                assert Ad.multiply(a, b) == A.multiply(a, b), (k, a, b)


def test_random_composite_products():
    """100 random composite products per k = 1, 2, 3 (one to three curves per
    factor, `E`-powers −2..2, `SU(2)` weights 0, 1) equal the flow's product
    on the flow labels, and the first half equal the Φ-peel with no reducer
    (the tables took about 190 s on the 100 at k = 3; the flow takes 2–4 s per
    100, the class 0.1–0.2 s)."""
    for k in (1, 2, 3):
        A = _alg(k)
        rng = random.Random(1000 + k)
        for i in range(100):
            a = _random_monomial(k, rng, kappas=(0, 0, 1))
            b = _random_monomial(k, rng, kappas=(0, 0, 1))
            got = A.multiply(a, b)
            assert _on_flow_labels(A, got) == _flow_product(k, a, b), (k, a, b)
            if i < 50:
                assert got == A._multiply_via_phi(a, b), (k, a, b)


def test_product_axioms():
    """`e·e = e` and the unit law on the generators; the bar involution and ρ
    as an automorphism on every ordered generator pair (k = 1, 2) and on 60
    random composite pairs (k = 1, 2, 3, `κ` ∈ {0, 1}); associativity on every
    triple of generators at k = 1 (2744) and on the triples of a window at
    k = 2 (the curves at marked points 0..2 with `E^{±1}`: 4913)."""
    for k in (1, 2, 3):
        A = _alg(k)
        gens = _generators(k)
        assert A.verify_identity_in_basis()
        assert all(A.verify_unit_law(a) for a in gens)
        pairs = list(itertools.product(gens, gens)) if k < 3 else []
        rng = random.Random(2000 + k)
        pairs += [(_random_monomial(k, rng, kappas=(0, 1)),
                   _random_monomial(k, rng, kappas=(0, 1))) for _ in range(60)]
        for a, b in pairs:
            assert A.verify_bar_involution(a, b), (k, a, b)
            assert A.verify_rho_is_automorphism(a, b), (k, a, b)
    A = _alg(1)
    for a, b, c in itertools.product(_generators(1), repeat=3):
        assert A.verify_associativity(a, b, c), (a, b, c)
    A = _alg(2)
    window = [_one(c) for c in G._curves(6) if c[0] <= 2] + [((), 1, 0), ((), -1, 0)]
    assert len(window) ** 3 == 4913
    for a, b, c in itertools.product(window, repeat=3):
        assert A.verify_associativity(a, b, c), (a, b, c)


def _heldout_pairs(k, rng, singles, composites):
    n = 2 * k + 2
    cs = G._curves(n)
    pairs = []
    while len(pairs) < singles:
        g, h = rng.choice(cs), rng.choice(cs)
        if not G._curves_cross(g, h, n) and rng.random() < 0.7:
            continue                                # most of them crossing
        pairs.append((_one(g, rng.randint(-2, 2)), _one(h, rng.randint(-2, 2))))
    pairs += [(_random_monomial(k, rng), _random_monomial(k, rng))
              for _ in range(composites)]
    return pairs


def test_heldout_k4_against_flow():
    """HELD OUT (the analytic rule was found at k ≤ 3): 200 sampled products at
    k = 4 — 150 of two single curves at `E`-powers −2..2 (mostly crossing), 50
    of two random composites — equal `U1A1DevenViaDoddRG(4).multiply` on the
    flow labels.  Measured cost: the class's 200 products 0.2 s, the flow's
    2.0 s (worst pair 0.4 s)."""
    k = 4
    A = _alg(k)
    for a, b in _heldout_pairs(k, random.Random(4000 + k), 150, 50):
        assert _on_flow_labels(A, A.multiply(a, b)) == _flow_product(k, a, b), (a, b)


def test_multiplication_negative_controls():
    """Each deliberately wrong rule is caught, k = 1, 2.
    (a) The χ₁ fork dropped from the resolutions: the analytic rule refuses
        every two-crossing pair (no anchor; 4 / 60) and differs from the
        derived peel on every pair with a loop (36 / 150); built into
        `multiply`, every generator product with a loop is wrong, the
        two-crossing ones fall back to the derived peel, which (its candidates
        lacking the χ₁ state's curves) raises at k = 2 and, those curves being
        the other middle state's too, is right at k = 1; the one-crossing and
        the `𝖖`-commuting products are unchanged.
    (b) `E` on the edge {2, 3} (or {0, 1}) instead of {1, 2}: the analytic
        rule differs from the derived peel on some pairs.
    (c) The two-crossing anchor with `#B`'s sign flipped (`A − 1 + #B`):
        differs on every two-crossing pair (4 / 60).
    (d) A missing candidate (either pure daughter of a loop-and-curve pair
        removed): the derived peel raises on every such pair (12 / 60)."""
    own_res, own_edge = G._resolutions, G._E_EDGE
    for k in (1, 2):
        A = _alg(k)
        n = 2 * k + 2
        crossing = [(g, h) for g in G._curves(n) for h in G._curves(n)
                    if G._curves_cross(g, h, n)]
        derived = {p: A._cross_derived(*p) for p in crossing}
        kinds = {p: G._resolutions(*p, k)[0] for p in crossing}
        right_products = {(a, b): A.multiply(a, b)
                          for a, b in itertools.product(_generators(k), repeat=2)}
        # (a)
        outcome = {}
        try:
            G._resolutions = lambda g, h, kk: (
                own_res(g, h, kk)[0],
                [s for s in own_res(g, h, kk)[1] if not s[2]])
            for p in crossing:
                an = G._analytic_product(*p, k)
                outcome[p] = "refuses" if an is None else (
                    "same" if an == derived[p] else "differs")
            bad = U1A1DevenConeKAlgebra(k)
            products = {}
            for (a, b), right in right_products.items():
                kind = "commuting"
                if a[0] and b[0] and G._curves_cross(a[0][0][0], b[0][0][0], n):
                    kind = kinds[(a[0][0][0], b[0][0][0])]
                try:
                    ok = bad.multiply(a, b) == right
                    products.setdefault(kind, set()).add("right" if ok else "wrong")
                except ValueError:
                    products.setdefault(kind, set()).add("raises")
        finally:
            G._resolutions = own_res
        for p in crossing:
            want = {"one crossing": "same", "two crossings": "refuses"}.get(kinds[p], "differs")
            assert outcome[p] == want, (k, p, outcome[p])
        assert sum(1 for p in crossing if "loop" in kinds[p]) == {1: 36, 2: 150}[k]
        assert products["commuting"] == products["one crossing"] == {"right"}, k
        for kind in ("loop and curve", "curve and loop", "two loops"):
            assert products[kind] == {"wrong"}, (k, kind)
        assert products["two crossings"] == {1: {"right"}, 2: {"raises"}}[k], k
        # (b)
        for edge in (0, 2):
            try:
                G._E_EDGE = edge
                diff = sum(G._analytic_product(*p, k) != derived[p] for p in crossing)
            finally:
                G._E_EDGE = own_edge
            assert diff > 0, (k, edge)
        # (c)
        two = [p for p in crossing if kinds[p] == "two crossings"]
        for g, h in two:
            _kind, states = G._resolutions(g, h, k)
            (chi_state,) = [s for s in states if s[2]]
            Ab = G._bulk(g, chi_state[0], chi_state[1], k)
            flipped = {}
            for curves, e, chi, nB in states:
                byk = flipped.setdefault((curves, e), {})
                byk[chi] = byk.get(chi, LaurentPoly({})) + LaurentPoly({Ab - 1 + nB: 1})
            assert flipped != derived[(g, h)], (k, g, h)
        assert len(two) == {1: 4, 2: 60}[k]
        # (d)
        lc = [p for p in crossing if kinds[p] == "loop and curve"]
        for g, h in lc:
            _kind, states = G._resolutions(g, h, k)
            pure = [s[0] for s in states if not s[2]]
            X = A._aux().multiply_elements(A._phi(_one(g)), A._phi(_one(h)))
            try:
                A._peel(X, {s[0] for s in states} - {pure[0]})
            except ValueError:
                pass
            else:
                raise AssertionError(f"a missing candidate was not caught: {k, g, h}")
        assert len(lc) == {1: 12, 2: 60}[k]


def deep_analytic_rule_k5():
    """DEEP: the analytic rule applies to, and equals the derived peel on, all
    8712 ordered crossing pairs at k = 5."""
    _analytic_vs_derived(5)


def deep_every_generator_product_k4():
    """DEEP: every ordered pair of generators at k = 4 (92² = 8464) against
    `U1A1DevenViaDoddRG(4).multiply` on flow labels."""
    k = 4
    A = _alg(k)
    for a, b in itertools.product(_generators(k), repeat=2):
        assert _on_flow_labels(A, A.multiply(a, b)) == _flow_product(k, a, b), (a, b)


def deep_orthonormality_k3():
    """DEEP: the orthonormality sample at k = 3 (every ordered pair of the
    curves and squares)."""
    _orthonormality(3, all_pairs=True)


_TESTS = [test_curve_and_cone_counts, test_crossings_generalise_a1dodd,
          test_magnetic_charge,
          test_dressing_start_read_from_ap_base, test_phi_equals_flow_rg_on_every_curve,
          test_phi_negative_controls, test_rho_closed_form, test_full_turn_shift,
          test_phi_of_monomials_and_label_maps, test_trace_calls_no_rg,
          test_creutzig_tower, test_traces_equal_flow_route,
          test_traces_equal_flow_windowed_trace, test_orthonormality,
          test_orthonormality_negative_controls, test_rho2_twisted_cyclicity,
          test_pairing_equals_default_route,
          # (D): the positive control first
          test_derived_peel_positive_control, test_loop_fork_is_the_odd_model,
          test_cocycle_closed_form,
          test_analytic_rule_equals_derived_peel,
          test_products_equal_flow_every_generator_pair,
          test_random_composite_products, test_product_axioms,
          test_heldout_k4_against_flow, test_multiplication_negative_controls]

_DEEP = [deep_orthonormality_k3, deep_analytic_rule_k5,
         deep_every_generator_product_k4]

if __name__ == "__main__":
    import time
    run = _TESTS + (_DEEP if "deep" in sys.argv[1:] else [])
    for t in run:
        t0 = time.time()
        t()
        print(f"  PASS: {t.__name__} [{time.time() - t0:.1f}s]", flush=True)
    print(f"\nAll {len(run)} geometric-frame tests passed.")
