"""checks_coulomb.py — the design record adapters for the draft's Section 3, "K-theoretic
Coulomb branch algebras as K_q algebras": the abelianized presentation
(eq:fgprod, eq:ccprod, eq:ccclosed, eq:cocydef), rho (eq:rho-witten,
eq:rhotorus), the trace and pairing (eq:measure, eq:Iexplicit), the cocycle
trivialisation (eq:cocha), the pure SU(2)/SO(3) examples (eq:su2thooft,
eq:su2dform, eq:su2res, eq:su2dyonic), the adjoint examples (eq:adjsquare,
eq:adjbubble), the global forms at rank one (eq:su2so3lat), and the main
conjecture's rows (conj:abeKalgebra) on fast windows.

Exact arithmetic throughout: rational functions in (q, v_i, μ) are compared by
cross-multiplication of numerators against the products of the denominator
factors (1 - q^k v^α), never numerically; the trace and pairing formulas are
evaluated as q-series with Laurent-monomial coefficients (the constant term in
v is the contour integral), with the repository's rational functions expanded
independently from their (numerator, denominator factors) data.
"""
from __future__ import annotations

import itertools
from fractions import Fraction
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # the release root
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from kalgebra import Element  # noqa: E402
from laurent_poly import LaurentPoly  # noqa: E402
from checks_kq import check, elem  # noqa: E402
from checks_uq_a1d3 import S2, inv_one_minus  # noqa: E402
from root_datum import su_2, so_n, su_n, sp_n, g_2  # noqa: E402
from wrq_torus import CC, dressing_psi, rho_label  # noqa: E402
from pure_g_abe_kalgebra import PureGAbeKAlgebra  # noqa: E402
from gn_abe_kalgebra import GNAbeKAlgebra  # noqa: E402
from matter_wrq_torus import slot_weights, rep_weights  # noqa: E402
from star_bubbling import criterion, conventional, neumann_row, tropical_support  # noqa: E402
from wrq_torus import WRQTorus  # noqa: E402

BIG = 10 ** 6


# --------------------------------------------------------------------------
# exact rational functions: (numerator S2 polynomial, list of (alpha, k) denominator factors (1 - q^k v^alpha))
# --------------------------------------------------------------------------
def poly(nv, terms):
    """terms: {(q_exp, mono): coeff} -> S2 polynomial (no truncation)."""
    c: dict = {}
    for (e, m), v in terms.items():
        c.setdefault(e, {})[tuple(m)] = c.get(e, {}).get(tuple(m), 0) + v
    return S2(c, BIG, nv)


def one_minus(nv, k, alpha):
    """(1 - q^k v^alpha) as a polynomial; at k = 0 both terms sit at q^0 and are merged."""
    z = (0,) * nv
    if k == 0:
        return S2({0: {z: 1, tuple(alpha): -1}}, BIG, nv)
    return S2({0: {z: 1}, k: {tuple(alpha): -1}}, BIG, nv)


def den_product(nv, factors):
    out = S2.one(BIG, nv)
    for alpha, k in factors:
        out = out * one_minus(nv, k, alpha)
    return out


def rat_eq(numL, denL, numR, denR, nv):
    """numL/ΠdenL == numR/ΠdenR as rational functions (exact cross-multiplication)."""
    return (numL * den_product(nv, denR)).eq_to(numR * den_product(nv, denL), BIG)


def tl_to_poly(tl, nv):
    """TorusLaurent -> S2 polynomial."""
    c: dict = {}
    for mono, lp in tl.terms.items():
        for e, v in lp._coeffs.items():
            if v:
                c.setdefault(e, {})[tuple(mono)] = c.get(e, {}).get(tuple(mono), 0) + v
    return S2(c, BIG, nv)


def tr_to_rat(tr, nv):
    """TorusRational -> (numerator polynomial, denominator factor list)."""
    factors = []
    for (alpha, k), mult in tr.den.items():
        factors += [(tuple(alpha), k)] * mult
    return tl_to_poly(tr.num, nv), factors


def shift_poly(p: S2, c, datum):
    """v^λ -> q^{<c,λ>} v^λ on a polynomial (the paper's f(q^c v))."""
    out: dict = {}
    for e, mono_c in p.c.items():
        for m, v in mono_c.items():
            s = datum.shift_pairing(tuple(c), tuple(m))
            out.setdefault(e + s, {})[m] = out.get(e + s, {}).get(m, 0) + v
    return S2(out, p.K, p.nv)


def shift_factors(factors, c, datum):
    return [(alpha, k + datum.shift_pairing(tuple(c), tuple(alpha))) for alpha, k in factors]


def rat_mul(a, b):
    return a[0] * b[0], a[1] + b[1]


def rat_sum(terms):
    """Σ num_i/Πden_i over a common denominator (the product of all denominators)."""
    all_f = []
    for _n, f in terms:
        all_f += f
    num = S2({}, BIG, terms[0][0].nv)
    for n, f in terms:
        others = list(all_f)
        for x in f:
            others.remove(x)
        num = num + n * den_product(n.nv, others)
    return num, all_f


def rat_to_series(num, factors, nv, K):
    """Expand num/Π(1 - q^k v^α) as a q-series to q^K (k != 0 required: 1/(1-q^k v^α) is geometric in q^|k|).

    Every expanded factor has only non-negative q-powers, so num·Π(factors) is exact through q^K once the factors
    are carried to q^(K+d), d the depth of num's most negative q-power — that exact margin, not a fixed one (a fixed
    +200 was used until 2026-09-23: identical results on eight rank-one/-two, pure and matter lines, 300x-100000x
    slower in two or more Cartan variables, and wrong past d = 200)."""
    KK = K + max(0, -min(num.c)) if num.c else K
    out = num.trunc(KK)
    for alpha, k in factors:
        if k > 0:
            out = out * inv_one_minus(k, alpha, KK, nv)
        elif k < 0:
            # 1/(1 - q^k v^α) = -q^{-k} v^{-α} / (1 - q^{-k} v^{-α})
            neg = tuple(-x for x in alpha)
            out = out * (inv_one_minus(-k, neg, KK, nv) * S2.term(-k, neg, KK, nv, -1))
        else:
            raise ValueError("k = 0 factor cannot be expanded as a q-series")
    return out.trunc(K)


def invert_vars(p: S2, which):
    """v_i -> v_i^{-1} for the variable indices in `which`."""
    return S2({e: {tuple(-x if i in which else x for i, x in enumerate(m)): v for m, v in mc.items()} for e, mc in p.c.items()}, p.K, p.nv)


def bar_rat(num, factors):
    """q -> q^{-1} with v fixed."""
    return S2({-e: mc for e, mc in num.c.items()}, num.K, num.nv), [(alpha, -k) for alpha, k in factors]


# --------------------------------------------------------------------------
# the printed cocycle (eq:ccclosed) and its matter partner, independently
# --------------------------------------------------------------------------
def printed_vec_cocycle(A, B, alpha, nv):
    """cocy^vec(A,B; z = v^α): q^{|AB|} z^p / ((q^r z;q^2)_p (q^{r+2} z;q^2)_p) for A>0>B, bar image for A<0<B, 1 otherwise."""
    if A * B >= 0:
        return S2.one(BIG, nv), []
    p, r = min(abs(A), abs(B)), abs(abs(A) - abs(B))
    num = S2.term(abs(A * B), tuple(p * x for x in alpha), BIG, nv)
    factors = [(tuple(alpha), r + 2 * j) for j in range(p)] + [(tuple(alpha), r + 2 + 2 * j) for j in range(p)]
    if A > 0 > B:
        return num, factors
    return bar_rat(num, factors)


def printed_CC(datum, a, b, nv):
    num, factors = S2.one(BIG, nv), []
    for alpha in datum.positive_roots():
        A, B = datum.shift_pairing(tuple(a), tuple(alpha)), datum.shift_pairing(tuple(b), tuple(alpha))
        n2, f2 = printed_vec_cocycle(A, B, alpha, nv)
        num, factors = num * n2, factors + f2
    return num, factors


def printed_hyp_cocycle(A, B, x_mono, nv):
    """cocy^hyp(A,B; x) = (-q^{r+1} x; q^2)_p for A>0>B, bar image for A<0<B, 1 otherwise — a polynomial."""
    if A * B >= 0:
        return S2.one(BIG, nv)
    p, r = min(abs(A), abs(B)), abs(abs(A) - abs(B))
    out = S2.one(BIG, nv)
    for j in range(p):
        e = r + 1 + 2 * j
        out = out * (S2.one(BIG, nv) + S2.term(e if A > 0 else -e, x_mono, BIG, nv))
    return out



def _padded_tr(Pp, tr, level=None):
    """A TorusRational (in v) as (num, dens) in Printed's padded variables, at mu-exponent `level`."""
    n_, f_ = tr_to_rat(tr, Pp.D.dim)
    mu = tuple(level) if level is not None else (0,) * Pp.M
    return (S2({e: {tuple(m) + mu: v for m, v in mc.items()} for e, mc in n_.c.items()}, BIG, Pp.nv),
            [(tuple(al) + (0,) * Pp.M, k) for al, k in f_])


def _padded_levels(Pp, levels):
    parts = [_padded_tr(Pp, t, lev) for lev, t in levels.items()]
    return rat_sum(parts) if parts else (S2({}, BIG, Pp.nv), [])


def _padded_residuals(Pp, x):
    return {tuple(m): (_padded_levels(Pp, r) if isinstance(r, dict) else _padded_tr(Pp, r)) for m, r in x.residuals().items()}



def printed_hyp_levels(D, a, b, slots, dual=False):
    """{mu-exponent vector: TorusRational}: prod_s prod_{w in slot s} cocy^hyp(<a,w>, <b,w>; mu_s v^w) of eq:ccclosed,
    (-q^{r+1} x; q^2)_p for A > 0 > B and its bar image for A < 0 < B, expanded in powers of mu (dual=True: the weights
    of N^vee, the negative control)."""
    from weyl_torus_ring import TorusRational, TorusLaurent
    out = {(0,) * len(slots): TorusRational.one(D)}
    for si, wts in enumerate(slots):
        for w in wts:
            w = tuple(-x for x in w) if dual else tuple(w)
            A, B = _int(D.shift_pairing(tuple(a), w)), _int(D.shift_pairing(tuple(b), w))
            if A * B >= 0:
                continue
            p, r = min(abs(A), abs(B)), abs(abs(A) - abs(B))
            for j in range(p):
                e = r + 1 + 2 * j
                mono = TorusRational(D, TorusLaurent(D, {w: LaurentPoly({e if A > 0 else -e: 1})}), {})
                new = {}
                for key, t in out.items():
                    new[key] = (new[key] + t) if key in new else t
                    k2 = tuple(x + (1 if i == si else 0) for i, x in enumerate(key))
                    tt = t * mono
                    new[k2] = (new[k2] + tt) if k2 in new else tt
                out = new
    return out


def matter_cocycle_matches(D, a, b, slots, dual=False):
    """CC_N(a, b) == CC(a, b) * printed hyp factor, mu-level by mu-level (exact, in v alone)."""
    from matter_wrq_torus import CC_N
    got = CC_N(D, a, b, slots)
    pure = CC(D, a, b)
    exp = {k: (pure * h).simplify() for k, h in printed_hyp_levels(D, a, b, slots, dual).items()}
    exp = {k: v for k, v in exp.items() if not v.is_zero()}
    got = {tuple(k): v.simplify() for k, v in got.items() if not v.simplify().is_zero()}
    return set(got) == set(exp) and all((got[k] + (-exp[k])).simplify().is_zero() for k in got)


def presentation_widened(environment):
    """The presentation row widened (2026-09-23): the printed vector cocycle and its 2-cocycle identity at the
    non-simply-laced Sp(4), Spin(5), G2 and at SU(2)xSU(2); the matter cocycle at rank two with complex
    representations; the product law eq:fgprod at rank two and with matter; and the bar property of the cocycle,
    bar(cocy_ab) = cocy_ba, which makes the cellwise bar an anti-automorphism.  Rank-two matter comparisons are
    exact cross-multiplications in three variables and cost minutes on full boxes, so the web tier uses the boxes
    below and the local tier the full ones."""
    from root_datum import b_n_simply_connected, product_datum
    from matter_wrq_torus import slot_weights, CC_N
    checks = []
    t0 = time.time()
    full = environment == "local"
    r2 = list(itertools.product(range(-1, 2), repeat=2))
    b3 = r2 if full else [(1, 0), (-1, 1), (0, -1)]
    b2, g2, su2 = b_n_simply_connected(2), g_2(), su_2()
    P2 = product_datum([su2, su2])
    pop = {}
    # (a) the vector cocycle
    na, bad = 0, ""
    for name, D in (("Sp(4)", sp_n(2)), ("Spin(5)", b2), ("G2", g2), ("SU(2)xSU(2)", P2)):
        Pp = Printed(D, 0)
        for a in r2:
            for b in r2:
                na += 1
                if not rat_eq(*_padded_tr(Pp, CC(D, a, b)), *Pp.cocy(a, b, ()), Pp.nv):
                    bad = bad or f"{name} a={a} b={b}"
    checks.append(check(f"eq:ccprod / eq:ccclosed at the non-simply-laced Sp(4), Spin(5), G2 and at SU(2)xSU(2): CC equals the printed cocycle on all {na} pairs of the box [-1,1]^2", not bad, bad))
    pop["vector cocycle pairs"] = na
    # (b) the 2-cocycle identity of the printed cocycle there (associativity of U_a U_b = cocy_ab(q^(a+b) v) U_(a+b))
    nb, bad = 0, ""
    for name, D in (("Sp(4)", sp_n(2)), ("Spin(5)", b2), ("G2", g2)):
        Pp = Printed(D, 0)
        for a in r2:
            for b in r2:
                for c in r2:
                    ab = tuple(x + y for x, y in zip(a, b)); bc = tuple(x + y for x, y in zip(b, c)); abc = tuple(x + y for x, y in zip(ab, c))
                    two_a_bc = tuple(2 * x + y for x, y in zip(a, bc))
                    L = rat_mul(Pp.shift(Pp.cocy(a, b, ()), ab), Pp.shift(Pp.cocy(ab, c, ()), abc))
                    R = rat_mul(Pp.shift(Pp.cocy(b, c, ()), two_a_bc), Pp.shift(Pp.cocy(a, bc, ()), abc))
                    nb += 1
                    if not rat_eq(*L, *R, Pp.nv):
                        bad = bad or f"{name} {a},{b},{c}"
    checks.append(check(f"eq:cocydef  the printed cocycle satisfies the 2-cocycle identity at Sp(4), Spin(5), G2 ({nb} triples)", not bad, bad))
    pop["cocycle triples (non-simply-laced)"] = nb
    # (c) the matter cocycle at rank two, complex representations included; control: the weights of N^vee
    nc, bad, dual_caught = 0, "", {}
    for name, D, matter, bs in (("SU(3)+1", su_n(3), [(1, 0)], r2), ("SU(3)+symmetric", su_n(3), [(2, 0)], r2),
                                ("SU(3)+symmetric+1", su_n(3), [(2, 0), (1, 0)], r2), ("Spin(5)+vector", b2, [(1, 0)], r2),
                                ("G2+7", g2, [(1, 0)], r2), ("SU(2)xSU(2)+bifundamental", P2, [(1, 1)], r2)):
        slots = slot_weights(D, matter)
        Pp = Printed(D, len(slots))
        dual = [tuple(tuple(-x for x in w) for w in wts) for wts in slots]
        differ = 0
        for a in r2:
            for b in bs:
                nc += 1
                if not matter_cocycle_matches(D, a, b, slots):
                    bad = bad or f"{name} a={a} b={b}"
                if name.startswith("SU(3)"):
                    differ += not matter_cocycle_matches(D, a, b, slots, dual=True)
        if name.startswith("SU(3)"):
            dual_caught[name] = differ
    checks.append(check(f"eq:ccprod / eq:ccclosed with matter at rank two: CC_N equals the printed cocycle, cocy^hyp(<a,w>,<b,w>; mu^(w_f) v^w) over the weights of N, level by level in mu, on {nc} pairs of SU(3)+1, SU(3)+symmetric, SU(3)+symmetric+1, Spin(5)+vector, G2+7, SU(2)xSU(2)+bifundamental", not bad, bad))
    checks.append(check("negative control: the weights of N^vee in place of N give a different cocycle on the complex representations (" + ", ".join(f"{k}: {v} pairs" for k, v in dual_caught.items()) + ")", all(v > 0 for v in dual_caught.values())))
    pop["matter cocycle pairs"] = nc
    # (d) the product law with the printed cocycle, against the class's products
    def printed_product(Pp, slots, F, G):
        out = {}
        for m1, f in F.items():
            for m2, g in G.items():
                m = tuple(x + y for x, y in zip(m1, m2))
                t = rat_mul(rat_mul(Pp.shift(f, tuple(x - y for x, y in zip(m1, m))), Pp.shift(g, m1)), Pp.cocy(m1, m2, slots))
                out.setdefault(m, []).append(t)
        return {m: rat_sum(ts) for m, ts in out.items()}

    def class_product(A, Pp, a, b):
        out = {}
        for lab, cf in A.multiply(a, b).terms.items():
            cfp = S2({e: {(0,) * Pp.nv: v} for e, v in cf._coeffs.items() if v}, BIG, Pp.nv)
            for m, (n_, f_) in _padded_residuals(Pp, A.chart(lab)).items():
                out.setdefault(m, []).append((n_ * cfp, f_))
        return {m: rat_sum(ts) for m, ts in out.items()}
    prod_cases = [("SU(3)", PureGAbeKAlgebra(su_n(3)), [], [((1, 1), (0, 0)), ((0, 0), (1, 0))]),
                  ("Spin(5)", PureGAbeKAlgebra(b2), [], [((1, 1), (0, 0)), ((0, 0), (1, 0))]),
                  ("SU(2)+1", GNAbeKAlgebra(su2, (1,), nf=1), [(1,)], [((1,), (0,)), ((1,), (1,)), ((0,), (1,))])]
    if full:
        prod_cases.append(("G2", PureGAbeKAlgebra(g2), [], [((1, 2), (0, 0)), ((0, 0), (1, 0))]))
    nd, bad = 0, ""
    for name, A, matter, labs in prod_cases:
        D = A.datum
        slots = slot_weights(D, matter) if matter else ()
        Pp = Printed(D, len(slots))
        L = [A.fold(m, e) for m, e in labs]
        R = {l: _padded_residuals(Pp, A.chart(l)) for l in L}
        for a in L:
            for b in L:
                Pm, Cm = printed_product(Pp, slots, R[a], R[b]), class_product(A, Pp, a, b)
                Pn = {m for m, (nn, ff) in Pm.items() if nn.c}
                Cn = {m for m, (nn, ff) in Cm.items() if nn.c}
                nd += 1
                if Pn != Cn or not all(rat_eq(*Pm[m], *Cm[m], Pp.nv) for m in Pn):
                    bad = bad or f"{name} {a}·{b}"
    checks.append(check(f"eq:fgprod  the product law with the printed cocycle reproduces the class's products at rank two and with matter ({nd} products: " + ", ".join(f"{c[0]}" for c in prod_cases) + ")", not bad, bad))
    pop["products"] = nd
    # (e) the bar property: bar(cocy_ab) = cocy_ba on the printed cocycle, and bar(xy) = bar(y) bar(x) on the class's
    # torus products (the multiplicative version must fail somewhere: the product is not commutative)
    # (the vector factor is CC, equal to the printed one on these boxes by (a) and the SU(3) check above; the matter
    # factor is printed; mu is bar-invariant, so the comparison runs level by level in mu)
    ne, bad = 0, ""
    for name, D, matter in (("G2", g2, []), ("SU(3)+symmetric+1", su_n(3), [(2, 0), (1, 0)]), ("Spin(5)+vector", b2, [(1, 0)])):
        slots = slot_weights(D, matter) if matter else ()
        for a in r2:
            for b in r2:
                ab = {k: (CC(D, a, b) * h).bar().simplify() for k, h in printed_hyp_levels(D, a, b, slots).items()}
                ba = {k: (CC(D, b, a) * h).simplify() for k, h in printed_hyp_levels(D, b, a, slots).items()}
                ne += 1
                if set(ab) != set(ba) or not all((ab[k] + (-ba[k])).simplify().is_zero() for k in ab):
                    bad = bad or f"{name} {a},{b}"
    anti = mult = nx = 0
    for A, labs in ((PureGAbeKAlgebra(su_n(3)), [((1, 1), (0, 0)), ((1, 1), (1, 0)), ((0, 0), (1, 0))]),
                    (GNAbeKAlgebra(su_n(3), (1, 0), nf=1), [((1, 1), (0, 0)), ((0, 0), (1, 0))])):
        xs = [A.chart(A.fold(m, e)) for m, e in labs]
        for x in xs:
            for y in xs:
                nx += 1
                anti += _same_residuals((x * y).bar().residuals(), (y.bar() * x.bar()).residuals())
                mult += _same_residuals((x * y).bar().residuals(), (x.bar() * y.bar()).residuals())
    checks.append(check(f"the bar property of the printed cocycle, bar(cocy_ab) = cocy_ba, on {ne} pairs (G2; SU(3)+symmetric+1 with two matter slots; Spin(5)+vector), and bar(xy) = bar(y)bar(x) on all {nx} torus products of SU(3) and SU(3)+1 lines", not bad and anti == nx, f"{bad}; anti {anti}/{nx}"))
    checks.append(check(f"negative control: the multiplicative version bar(xy) = bar(x)bar(y) fails on {nx - mult} of the {nx} products", mult < nx))
    pop["bar pairs"] = ne
    pop["seconds"] = round(time.time() - t0, 1)
    return {"checks": checks, "population": pop}


def check_presentation(environment):
    checks = []
    # 1. the vector cocycle closed form, rank 1 (su_2: alpha = (2,), coroot (1,): A = 2a) and rank 2 (su_3)
    ok1 = True
    n1 = 0
    for a in range(-3, 4):
        for b in range(-3, 4):
            got = tr_to_rat(CC(su_2(), (a,), (b,)), 1)
            exp = printed_CC(su_2(), (a,), (b,), 1)
            ok1 &= rat_eq(*got, *exp, 1)
            n1 += 1
    checks.append(check(f"eq:ccclosed  the vector cocycle of SU(2): CC(a,b) equals the printed closed form (A>0>B), its bar image (A<0<B) or 1, on {n1} pairs", ok1))
    ok2 = True
    n2 = 0
    for a in itertools.product(range(-1, 2), repeat=2):
        for b in itertools.product(range(-1, 2), repeat=2):
            got = tr_to_rat(CC(su_n(3), a, b), 2)
            exp = printed_CC(su_n(3), a, b, 2)
            ok2 &= rat_eq(*got, *exp, 2)
            n2 += 1
    checks.append(check(f"eq:ccprod  the pure-gauge cocycle of SU(3) is the product over positive roots of the printed factor, on {n2} pairs", ok2))
    # 2. the 2-cocycle identity of the PRINTED closed form (associativity of U_a U_b = cocy_{a,b}(q^{a+b} v) U_{a+b})
    def cocycle_identity(datum, nv, rng):
        ok = True
        n = 0
        for a in rng:
            for b in rng:
                for c in rng:
                    ab = tuple(x + y for x, y in zip(a, b)); bc = tuple(x + y for x, y in zip(b, c)); abc = tuple(x + y for x, y in zip(ab, c))
                    two_a_bc = tuple(2 * x + y for x, y in zip(a, bc))
                    L1 = printed_CC(datum, a, b, nv); L1 = (shift_poly(L1[0], ab, datum), shift_factors(L1[1], ab, datum))
                    L2 = printed_CC(datum, ab, c, nv); L2 = (shift_poly(L2[0], abc, datum), shift_factors(L2[1], abc, datum))
                    R1 = printed_CC(datum, b, c, nv); R1 = (shift_poly(R1[0], two_a_bc, datum), shift_factors(R1[1], two_a_bc, datum))
                    R2 = printed_CC(datum, a, bc, nv); R2 = (shift_poly(R2[0], abc, datum), shift_factors(R2[1], abc, datum))
                    L, R = rat_mul(L1, L2), rat_mul(R1, R2)
                    ok &= rat_eq(*L, *R, nv)
                    n += 1
        return ok, n
    ok3, n3 = cocycle_identity(su_2(), 1, [(a,) for a in range(-2, 3)])
    ok4, n4 = cocycle_identity(su_n(3), 2, list(itertools.product(range(-1, 2), repeat=2)))
    checks.append(check(f"eq:cocydef  the printed cocycle satisfies the 2-cocycle identity cocy_ab(q^(a+b)v) cocy_(a+b)c(q^(a+b+c)v) = cocy_bc(q^(2a+b+c)v) cocy_a(b+c)(q^(a+b+c)v): SU(2) {n3} triples, SU(3) {n4} triples", ok3 and ok4))
    # 3. the matter cocycle: SO(3) with adjoint matter (one slot, weights -1, 0, 1) and SU(2) with a fundamental
    try:
        from matter_wrq_torus import CC_N
        ok5 = True
        n5 = 0
        detail = ""
        for datum, matter, nvv in ((so_n(3), (1,), 2), (su_2(), (1,), 2)):
            slots = slot_weights(datum, (tuple(matter),) if isinstance(matter[0], int) else matter)
            for a in range(-2, 3):
                for b in range(-2, 3):
                    got = CC_N(datum, (a,), (b,), slots)
                    # expected: the pure cocycle times, for every weight w of the (single) slot, cocy^hyp(<a,w>, <b,w>; μ v^w)
                    num, factors = printed_CC(datum, (a,), (b,), 1)
                    num2 = S2({e: {m + (0,): v for m, v in mc.items()} for e, mc in num.c.items()}, BIG, 2)
                    factors2 = [(tuple(alpha) + (0,), k) for alpha, k in factors]
                    hyp = S2.one(BIG, 2)
                    for w in rep_weights(datum, matter):
                        A, B = datum.shift_pairing((a,), tuple(w)), datum.shift_pairing((b,), tuple(w))
                        hyp = hyp * printed_hyp_cocycle(A, B, tuple(w) + (1,), 2)
                    exp = (num2 * hyp, factors2)
                    if isinstance(got, dict):   # {flavour level: TorusRational}
                        gnum = S2({}, BIG, 2); gfac = None
                        parts = []
                        for lev, trr in got.items():
                            n_, f_ = tr_to_rat(trr, 1)
                            n_ = S2({e: {m + (lev[0] if isinstance(lev, tuple) else lev,): v for m, v in mc.items()} for e, mc in n_.c.items()}, BIG, 2)
                            parts.append((n_, [(tuple(al) + (0,), k) for al, k in f_]))
                        gnum, gfac = rat_sum(parts) if parts else (S2({}, BIG, 2), [])
                        ok = rat_eq(gnum, gfac, *exp, 2)
                    else:
                        n_, f_ = tr_to_rat(got, 2)
                        ok = rat_eq(n_, f_, *exp, 2)
                    if not ok and not detail:
                        detail = f"first mismatch at {datum.name} a={a}, b={b}: got {got}"
                    ok5 &= ok
                    n5 += 1
        checks.append(check(f"eq:ccprod / eq:ccclosed  the matter cocycle CC[N] is the pure cocycle times Π_w cocy^hyp(<a,w>,<b,w>; μ v^w) = (-q^(r+1) x; q^2)_p, on {n5} pairs (SO(3)+adjoint, SU(2)+fundamental)", ok5, detail))
    except Exception as ex:  # noqa: BLE001
        checks.append(check("the matter cocycle against the printed cocy^hyp", False, f"{type(ex).__name__}: {ex}"))
    # 4. eq:fgprod — the product of two SU(2) lines from the printed formula with the printed cocycle, against the class
    A = PureGAbeKAlgebra(su_2())
    def residuals_rat(label):
        return {m: tr_to_rat(r, 1) for m, r in A.chart(label).residuals().items()}
    def class_product_rat(a, b):
        out: dict = {}
        for lab, cf in A.multiply(a, b).terms.items():
            for m, (n_, f_) in residuals_rat(lab).items():
                scaled = S2({e + de: dict(mc) for e, mc in n_.c.items() for de in [0]}, BIG, 1)
                # multiply the numerator by the Laurent coefficient cf(q)
                cfp = S2({e: {(0,): v} for e, v in cf._coeffs.items() if v}, BIG, 1)
                out.setdefault(m, []).append((n_ * cfp, f_))
        return {m: rat_sum(ts) for m, ts in out.items()}
    def printed_product_rat(f, g):
        out: dict = {}
        for m1, (n1_, f1_) in f.items():
            for m2, (n2_, f2_) in g.items():
                m = (m1[0] + m2[0],)
                # f_{m'}(q^{m'-m} v) g_{m-m'}(q^{m'} v) cocy_{m', m-m'}(v)
                t1 = (shift_poly(n1_, (m1[0] - m[0],), su_2()), shift_factors(f1_, (m1[0] - m[0],), su_2()))
                t2 = (shift_poly(n2_, (m1[0],), su_2()), shift_factors(f2_, (m1[0],), su_2()))
                cc = printed_CC(su_2(), m1, m2, 1)
                out.setdefault(m, []).append(rat_mul(rat_mul(t1, t2), cc))
        return {m: rat_sum(ts) for m, ts in out.items()}
    L20, L21, L01, L02 = ((1,), (0,)), ((1,), (1,)), ((0,), (1,)), ((0,), (2,))
    ok6 = True
    detail6 = ""
    for a, b in ((L20, L20), (L20, L21), (L01, L20), (L21, L21), (L02, L21)):
        P = printed_product_rat(residuals_rat(a), residuals_rat(b))
        C = class_product_rat(a, b)
        Pn = {m for m, (n_, f_) in P.items() if n_.c}
        Cn = {m for m, (n_, f_) in C.items() if n_.c}
        if Pn != Cn:
            ok6 = False
            detail6 = detail6 or f"{a}·{b}: supports differ, printed {sorted(Pn)} vs class {sorted(Cn)}"
            continue
        for m in Pn:
            if not rat_eq(*P[m], *C[m], 1):
                ok6 = False
                detail6 = detail6 or f"{a}·{b} at m={m}: printed {P[m][0]} / {P[m][1]} vs class {C[m][0]} / {C[m][1]}"
    checks.append(check("eq:fgprod  (f·g)_m = Σ_m' f_m'(q^(m'-m) v) g_(m-m')(q^m' v) cocy_(m',m-m')(v) with the printed cocycle reproduces the class's products of the SU(2) lines L_20, L_21, L_01, L_02 (5 products, every magnetic sector)", ok6, detail6))
    checks.append(check("negative control: the SU(2) cocycle with the shifts (q^r z;q^2)_p (q^r z;q^2)_p (the second factor unshifted) is detected", not rat_eq(*tr_to_rat(CC(su_2(), (1,), (-1,)), 1), S2.term(4, (4,), BIG, 1), [((2,), 0), ((2,), 2), ((2,), 0), ((2,), 2)], 1)))
    wide = presentation_widened(environment)
    checks += wide["checks"]
    return {"checks": checks, "population": {"SU(2) pairs": n1, "SU(3) pairs": n2, "cocycle triples": n3 + n4, "matter pairs": n5, "products": 5, "widened (2026-09-23)": wide["population"]},
            "controls": {"positive": "the class's products reproduced from the printed formula", "negative": "an altered cocycle detected; the weights of N^vee give a different matter cocycle on the complex representations; the multiplicative bar(xy) = bar(x)bar(y) fails"},
            "notes": "All comparisons are exact rational identities (cross-multiplied numerators against the denominator factors), never truncated.", "inputs": []}


# --------------------------------------------------------------------------
# rho: eq:rho-witten and eq:rhotorus
# --------------------------------------------------------------------------
def kappa(datum, m, matter_weights=()):
    """κ(m) = Σ_{α: <α,m>>0} <α,m> α − Σ_{w∈N: <w,m>>0} <w,m> w (weight coordinates); and D(m) = Σ_{w: <w,m>>0} <w,m>."""
    rank = len(datum.simple_roots[0])
    k = [0] * rank
    for alpha in datum.roots():
        p = datum.shift_pairing(tuple(m), tuple(alpha))
        if p > 0:
            k = [x + p * y for x, y in zip(k, alpha)]
    D = 0
    for w in matter_weights:
        p = datum.shift_pairing(tuple(m), tuple(w))
        if p > 0:
            k = [x - p * y for x, y in zip(k, w)]
            D += p
    return tuple(k), D


def pure_windows():
    return {
        "SU(2)": (su_2(), [(m,) for m in range(0, 4)], [(e,) for e in range(-3, 4)]),
        "SO(3)": (so_n(3), [(m,) for m in range(0, 4)], [(e,) for e in range(-3, 4)]),
        "SU(3)": (su_n(3), list(itertools.product(range(0, 2), repeat=2)), list(itertools.product(range(-1, 2), repeat=2))),
        "Sp(4)": (sp_n(2), [(0, 0), (1, 0), (0, 1), (1, 1)], list(itertools.product(range(-1, 2), repeat=2))),
        "G2": (g_2(), [(0, 0), (1, 0), (0, 1)], list(itertools.product(range(-1, 2), repeat=2))),
    }


def check_rho_witten(environment):
    checks = []
    n_tot = 0
    ok_all = True
    detail = ""
    for name, (datum, ms, es) in pure_windows().items():
        A = PureGAbeKAlgebra(datum)
        for m in ms:
            for e in es:
                lab = A.fold(m, e)
                k, _D = kappa(datum, m)
                exp = A.fold(tuple(-x for x in m), tuple(kx - ex for kx, ex in zip(k, e)))
                got = A.rho(lab)
                n_tot += 1
                if got != exp:
                    ok_all = False
                    if not detail:
                        detail = f"{name} label {lab}: rho -> {got}, expected {exp}"
    checks.append(check(f"eq:rho-witten  rho(m,e) = (-m, κ(m) - e) with κ(m) = Σ_(α:<α,m>>0) <α,m> α (pure gauge; Weyl-canonical labels) on {n_tot} labels of SU(2), SO(3), SU(3), Sp(4), G2", ok_all, detail))
    # with matter: the flavour shift κ_f(m) = -Σ_(w:<w,m>>0) <w,m> w_f, i.e. f -> f^* ⊗ det^{-D}
    ok_m = True
    n_m = 0
    detail = ""
    from g_matter_roster import roster_spec
    specs = [("SO(3) + adjoint", so_n(3), (1,), 1)] + [(nm, *roster_spec(nm)) for nm in ("su2-nf1", "su2-nf2", "su3-nf1")]
    for preset, datum, matter, nf in specs:
        G = GNAbeKAlgebra(datum, matter, nf=nf)
        weights = rep_weights(datum, matter)
        rank = len(datum.simple_roots[0])
        ms = [(m,) for m in range(0, 3)] if rank == 1 else [(0, 0), (1, 0), (0, 1)]
        es = [(e,) for e in range(-2, 3)] if rank == 1 else list(itertools.product(range(-1, 2), repeat=2))
        fs = [(f,) for f in range(-1, 2)] if nf == 1 else [(a, b) for a in range(-1, 2) for b in range(-1, 2) if a >= b]
        for m in ms:
            for e in es:
                for f in fs:
                    lab = G.fold(m, e, f)
                    k, _ = kappa(datum, m, weights * nf)      # κ sums over every weight of N = n_f copies
                    _, D = kappa(datum, m, weights)           # the det^{-D} twist of the U(n_f) factor is per copy
                    if nf == 1:
                        f_exp = (-f[0] - D,)
                    else:
                        f_exp = (-f[1] - D, -f[0] - D)      # dual of the U(2) weight (a, b) is (-b, -a), then ⊗ det^{-D}
                    exp = G.fold(tuple(-x for x in m), tuple(kx - ex for kx, ex in zip(k, e)), f_exp)
                    got = G.rho(lab)
                    n_m += 1
                    if got != exp:
                        ok_m = False
                        if not detail:
                            detail = f"{preset} label {lab}: rho -> {got}, expected {exp} (κ = {k}, D = {D})"
    checks.append(check(f"eq:rho-witten with matter  κ(m) carries -Σ <w,m> w and the flavour shifts by κ_f(m) = -D(m) w_f (f -> f^* ⊗ det^-D, in the diagonal U(1) of U(n_f)) on {n_m} labels of four (G, N)", ok_m, detail))
    # eq:rhotorus on the SU(2) lines: (rho f)_{-m}(v) = v^{κ(m)} f_m(v^{-1})
    A = PureGAbeKAlgebra(su_2())
    ok_t = True
    for lab in (((1,), (0,)), ((1,), (1,)), ((0,), (2,)), ((2,), (0,)), ((2,), (1,))):
        x = A.chart(lab)
        rx = x.rho()
        res, rres = x.residuals(), rx.residuals()
        for m, r in res.items():
            k, _ = kappa(su_2(), m)
            n_, f_ = tr_to_rat(r, 1)
            n_inv, f_inv = invert_vars(n_, {0}), [(tuple(-a for a in alpha), k_) for alpha, k_ in f_]   # f_m(v^{-1})
            n_exp = n_inv * S2.term(0, tuple(k), BIG, 1)
            got = rres.get((-m[0],))
            ok_t &= got is not None and rat_eq(*tr_to_rat(got, 1), n_exp, f_inv, 1)
    checks.append(check("eq:rhotorus  (rho f)_-m(v) = v^κ(m) f_m(v^-1) on the torus images of five SU(2) lines, every magnetic sector", ok_t))
    # the μ part: (rho f)_{-m}(v, μ) = v^κ(m) μ^κ_f(m) f_m(v^-1, μ^-1), i.e. the μ-exponent k goes to κ_f(m) - k = -D(m) - k
    M1 = GNAbeKAlgebra(su_2(), (1,), nf=1)
    wts1 = rep_weights(su_2(), (1,))
    lines_m = [M1.fold(m_, e_) for m_, e_ in (((1,), (0,)), ((1,), (1,)), ((0,), (1,)), ((2,), (0,)))]
    ok_tm = True
    for lab in lines_m:
        x = M1.chart(lab)
        rx = x.rho().residuals()
        for m, row in x.residuals().items():
            k, D = kappa(su_2(), m, wts1)
            got_row = rx.get((-m[0],), {})
            ok_tm &= set(got_row) == {(-D - key[0],) for key in row}
            for key, r in row.items():
                n_, f_ = tr_to_rat(r, 1)
                n_exp = invert_vars(n_, {0}) * S2.term(0, tuple(k), BIG, 1)
                f_inv = [(tuple(-a for a in alpha), k_) for alpha, k_ in f_]
                got = got_row.get((-D - key[0],))
                ok_tm &= got is not None and rat_eq(*tr_to_rat(got, 1), n_exp, f_inv, 1)
    checks.append(check("eq:rhotorus with μ  (rho f)_-m(v, μ) = v^κ(m) μ^κ_f(m) f_m(v^-1, μ^-1) (κ with its matter term, κ_f(m) = -D(m)) on the torus images of four SU(2)+1 lines, every magnetic sector and μ-level", ok_tm))
    # the statement proper: the torus map INDUCES eq:rho-witten (the chart of the label-level image is the torus image of the
    # chart) and is an algebra AUTOMORPHISM (multiplicative on torus products)
    lines_p = [((1,), (0,)), ((1,), (1,)), ((0,), (2,)), ((2,), (0,)), ((2,), (1,))]
    ind = all(_same_residuals(A.chart(A.rho(l)).residuals(), A.chart(l).rho().residuals()) for l in lines_p) and \
        all(_same_residuals(M1.chart(M1.rho(l)).residuals(), M1.chart(l).rho().residuals()) for l in lines_m)
    checks.append(check("eq:rhotorus induces eq:rho-witten  chart(rho(a)) == rho(chart(a)) on the five SU(2) and four SU(2)+1 lines", ind))
    n_mul, ok_mul = 0, True
    for Alg, ls in ((A, lines_p), (M1, lines_m)):
        for a_ in ls:
            for b_ in ls:
                x, y = Alg.chart(a_), Alg.chart(b_)
                n_mul += 1
                ok_mul &= _same_residuals((x * y).rho().residuals(), (x.rho() * y.rho()).residuals())
    checks.append(check(f"eq:rhotorus is an algebra automorphism  rho(x·y) == rho(x)·rho(y) on all {n_mul} torus products of those lines", ok_mul))
    # negative control: the formula without the inversion v -> v^-1 is detected
    x = A.chart(((2,), (1,)))
    rx = x.rho().residuals()
    caught = False
    for m, r in x.residuals().items():
        k, _ = kappa(su_2(), m)
        n_, f_ = tr_to_rat(r, 1)
        got = rx.get((-m[0],))
        caught |= got is None or not rat_eq(*tr_to_rat(got, 1), n_ * S2.term(0, tuple(k), BIG, 1), f_, 1)
    checks.append(check("negative control: the torus formula without v -> v^-1 is detected on L_(2,1)", caught))
    checks.append(check("negative control: dropping the matter term from κ is detected on SU(2)+1 fundamental", any(GNAbeKAlgebra(su_2(), (1,), nf=1).rho(GNAbeKAlgebra(su_2(), (1,), nf=1).fold((m,), (e,), (0,))) != GNAbeKAlgebra(su_2(), (1,), nf=1).fold((-m,), (kappa(su_2(), (m,))[0][0] - e,), (0,)) for m in (1, 2) for e in (0, 1))))
    return {"checks": checks, "population": {"pure labels": n_tot, "matter labels": n_m, "torus lines": 5, "torus lines with μ": len(lines_m), "torus products": n_mul},
            "controls": {"positive": "the label-level rho agrees with the printed κ everywhere", "negative": "the matter term of κ is needed; the torus formula without v -> v^-1 is detected"},
            "notes": "κ recomputed from the root datum's roots and the representation's weights with the datum's pairing; labels compared after Weyl canonicalisation (fold).", "inputs": []}


# --------------------------------------------------------------------------
# eq:measure and eq:Iexplicit at rank one, independently
# --------------------------------------------------------------------------
def qpoch_series(nv, e0, mono, K, sign=1, step=2):
    """(sign·q^{e0} m; q^step)_∞ = Π_{j>=0} (1 - sign q^{e0+step j} m) as a q-series (e0 >= 0; a factor with e0 = 0 is the polynomial (1 - m))."""
    out = S2.one(K, nv)
    j = 0
    while e0 + step * j <= K:
        out = out * (S2.one(K, nv) - S2.term(e0 + step * j, mono, K, nv, sign))
        j += 1
    return out


def inv_qpoch_series(nv, e0, mono, K, sign=1, step=2):
    """1/(sign q^{e0} m; q^step)_∞, e0 >= 1."""
    out = S2.one(K, nv)
    j = 0
    while e0 + step * j <= K:
        out = out * inv_one_minus(e0 + step * j, mono, K, nv, sign=sign)
        j += 1
    return out


def rps_int_S(t, K):
    """An RPowerSeries over the trivial ring -> S2 with no variables (integer coefficients)."""
    c = {}
    for e, v in t.coeffs.items():
        n = v.terms.get(v.ring.one_basis(), 0) if hasattr(v, "terms") else v
        if n:
            c[e] = {(): n}
    return S2(c, K, 0)


def su2_series(rat, K):
    """A rank-1 rational residual (num, factors) -> q-series in (v,)."""
    return rat_to_series(rat[0].trunc(K + 200), rat[1], 1, K)


def constant_term(s: S2, var=0):
    return s.coeff_x(0, var=var)



_PM_CACHE: dict = {}


def _pref_measure(D, slots, m, K, shifted=True):
    """(q^2;q^2)_inf^{2 rk} times the printed vector / matter measure of sector m (eq:Iexplicit), as a q-series.
    shifted=False drops the |<m,alpha>|, |<m,w>| shifts (every sector gets the m = 0 measure): a negative control."""
    key = (D.name, tuple(tuple(map(tuple, w)) for w in slots), tuple(m), K, shifted)
    if key in _PM_CACHE:
        return _PM_CACHE[key]
    d, M = D.dim, len(slots)
    nv = d + M
    out = S2.one(K, nv)
    for _ in range(2 * d):
        out = out * qpoch_series(nv, 2, (0,) * nv, K)
    for alpha in D.roots():
        s_ = abs(_int(D.shift_pairing(tuple(m), tuple(alpha)))) if shifted else 0
        x = tuple(alpha) + (0,) * M
        out = out * qpoch_series(nv, s_, x, K) * qpoch_series(nv, s_ + 2, x, K)
    for si, wts in enumerate(slots):
        for w in wts:
            s_ = abs(_int(D.shift_pairing(tuple(m), tuple(w)))) if shifted else 0
            mu = tuple(1 if i == si else 0 for i in range(M))
            out = out * inv_qpoch_series(nv, 1 + s_, tuple(w) + mu, K, sign=-1) * inv_qpoch_series(nv, 1 + s_, tuple(-t for t in w) + tuple(-t for t in mu), K, sign=-1)
    _PM_CACHE[key] = out
    return out


def _ct_v_product(X, Y, K):
    """The constant term of X*Y in all variables, without forming the product (pure gauge)."""
    out = {}
    for e1, c1 in X.c.items():
        for e2, c2 in Y.c.items():
            if e1 + e2 > K:
                continue
            acc = sum(v * c2.get(tuple(-t for t in mono), 0) for mono, v in c1.items())
            if acc:
                out[e1 + e2] = out.get(e1 + e2, 0) + acc
    return S2({e: {(): v} for e, v in out.items() if v}, K, 0)


def _printed_pairing(D, slots, FA, GB, K, trace=False, shifted=True):
    """|W| times the printed pairing: sum_m CT_v[f_m(1/v,1/mu) g_m(v,mu) (q^2;q^2)^{2 rk} measure(m)]; FA / GB are
    {m: q-series} with FA already inverted; trace=True gives |W| Tr g (m = 0, f = 1)."""
    d, M = D.dim, len(slots)
    nv = d + M
    total = S2({}, K, 0 if M == 0 else M)
    cells = [(0,) * d] if trace else sorted(set(FA) & set(GB))
    for m in cells:
        PM = _pref_measure(D, slots, m, K, shifted)
        g = GB.get(m, S2({}, K, nv))
        f = S2.one(K, nv) if trace else FA[m]
        if M == 0:
            total = total + _ct_v_product(f * g, PM, K)
        else:
            prod = f * g * PM
            for _ in range(d):
                prod = constant_term(prod)
            total = total + prod
    return total


def _rps_to_S(t, K, M):
    c = {}
    for e, v in t.coeffs.items():
        if e > K:
            continue
        if hasattr(v, "terms"):
            for key, n in v.terms.items():
                if n:
                    mono = tuple(key) if isinstance(key, tuple) else (key,)
                    mono = (mono + (0,) * M)[:M]
                    c.setdefault(e, {})[mono] = c.get(e, {}).get(mono, 0) + n
        elif v:
            c.setdefault(e, {})[(0,) * M] = v
    return S2(c, K, M)


def measure_widened(environment):
    """eq:measure / eq:Iexplicit at rank two (2026-09-23): the printed pairing and Tr rho(f)g at SU(3); the printed
    trace at Spin(5) and G2 (prefactor (q^2;q^2)^4/|W| with |W| = 8, 12); with matter at rank two, SU(3)+1 with its
    mu (a complex representation), pairing, trace and Tr rho(f)g."""
    from root_datum import b_n_simply_connected
    from matter_wrq_torus import slot_weights
    from kalgebra import _coeff_min_exp
    checks, pop = [], {}
    t0 = time.time()

    def trace_face(Alg, la, lb, KK):
        x = Alg.multiply(Alg.rho(la), lb)
        neg = min([0] + [_coeff_min_exp(c) for c in x.terms.values() if not c.is_zero()])
        return Alg.trace_element(x, KK - neg)

    def series(Pp, A, lab, K):
        return {m: rat_to_series(nd[0].trunc(K + 200), nd[1], Pp.nv, K) for m, nd in _padded_residuals(Pp, A.chart(lab)).items()}
    # pure SU(3): pairing, Tr rho(f)g and trace
    A = PureGAbeKAlgebra(su_n(3))
    D = A.datum
    Pp = Printed(D, 0)
    K = 4
    L = [A.fold(m, e) for m, e in [((0, 0), (0, 0)), ((0, 0), (1, 0)), ((1, 1), (0, 0)), ((1, 1), (1, 0))]]
    G_ = {l: series(Pp, A, l, K) for l in L}
    F_ = {l: {m: invert_vars(g, set(range(D.dim))) for m, g in G_[l].items()} for l in L}
    ok_p = ok_f = n = 0
    for la in L:
        for lb in L:
            got = rps_int_S(A.inner_product(la, lb, K), K)
            ok_p += _printed_pairing(D, (), F_[la], G_[lb], K).eq_to(got.scale(6), K)
            ok_f += got.eq_to(rps_int_S(trace_face(A, la, lb, K), K), K)
            n += 1
    checks.append(check(f"eq:Iexplicit, the conjectural half, at rank two  SU(3): the printed sum over m of contour integrals (prefactor (q^2;q^2)^4/6, each sector at |v| = 1 against its shifted measure) equals Tr(rho(f)·g) as the class computes it (the Schur residue of the magnetic-0 residual of rho(f)·g, assembled sector by sector) on all {n} pairs of {len(L)} lines through q^{K}", ok_p == n, f"{ok_p}/{n}"))
    checks.append(check(f"eq:Iexplicit at rank two  SU(3), consistency of two computations of Tr(rho(f)·g): the class's equals multiply-then-trace (rho(f)·g decomposed into canonical lines, each traced) on all {n} pairs", ok_f == n, f"{ok_f}/{n}"))
    pop["SU(3) pairs"] = n
    # traces at the non-simply-laced groups
    tr_ok, tr_n = 0, 0
    # G2's measure (twelve roots) costs ~11 min through q^4 (measured 2026-09-23, 4/4 lines equal): local tier only
    nsl = [("Spin(5)", PureGAbeKAlgebra(b_n_simply_connected(2)), [((0, 0), (0, 0)), ((0, 0), (1, 0)), ((1, 1), (0, 0)), ((2, 1), (0, 0))], 4, 8)]
    if environment == "local":
        nsl.append(("G2", PureGAbeKAlgebra(g_2()), [((0, 0), (0, 0)), ((0, 0), (1, 0)), ((1, 2), (0, 0)), ((2, 3), (0, 0))], 4, 12))
    for name, A2, labs, Kt, W in nsl:
        P2_ = Printed(A2.datum, 0)
        for m, e in labs:
            lab = A2.fold(m, e)
            tr_n += 1
            tr_ok += _printed_pairing(A2.datum, (), None, series(P2_, A2, lab, Kt), Kt, trace=True).eq_to(rps_int_S(A2.trace(lab, Kt), Kt).scale(W), Kt)
    checks.append(check(f"eq:measure at the non-simply-laced " + " and ".join(x[0] for x in nsl) + f": the printed trace, prefactor (q^2;q^2)^4/|W| (|W| = " + ", ".join(str(x[4]) for x in nsl) + f"), equals the class's on {tr_n} lines", tr_ok == tr_n, f"{tr_ok}/{tr_n}"))
    pop["non-simply-laced traces"] = tr_n
    # with matter at rank two: SU(3)+1 (a complex representation, one mu)
    A3 = GNAbeKAlgebra(su_n(3), (1, 0), nf=1)
    D3 = A3.datum
    slots = slot_weights(D3, [(1, 0)])
    P3 = Printed(D3, len(slots))
    K3 = 3
    L3 = [A3.fold(m, e) for m, e in [((0, 0), (0, 0)), ((0, 0), (1, 0)), ((1, 1), (0, 0))]]
    G3 = {l: series(P3, A3, l, K3) for l in L3}
    F3 = {l: {m: invert_vars(g, set(range(P3.nv))) for m, g in G3[l].items()} for l in L3}
    ok_p3 = ok_f3 = n3 = 0
    for la in L3:
        for lb in L3:
            got = _rps_to_S(A3.inner_product(la, lb, K3), K3, P3.M)
            ok_p3 += _printed_pairing(D3, slots, F3[la], G3[lb], K3).eq_to(got.scale(6), K3)
            ok_f3 += got.eq_to(_rps_to_S(trace_face(A3, la, lb, K3), K3, P3.M), K3)
            n3 += 1
    ok_t3 = sum(_printed_pairing(D3, slots, None, G3[l], K3, trace=True).eq_to(_rps_to_S(A3.trace(l, K3), K3, P3.M).scale(6), K3) for l in L3)
    checks.append(check(f"eq:measure / eq:Iexplicit with matter at rank two  SU(3)+1: the printed trace ({ok_t3}/{len(L3)} lines) and pairing ({ok_p3}/{n3} pairs), the matter factors (-q^(1+|<m,w>|) mu^(±1) v^(±w); q^2)^-1 over the weights of the fundamental, equal the class's through q^{K3} (the pairing comparison is the conjectural half: the class's pairing is Tr(rho(f)·g) assembled sector by sector); and the class's pairing equals multiply-then-trace on {ok_f3}/{n3} pairs (consistency)", ok_t3 == len(L3) and ok_p3 == n3 and ok_f3 == n3))
    pop["SU(3)+1 pairs"] = n3
    pop["seconds"] = round(time.time() - t0, 1)
    return {"checks": checks, "population": pop}


def check_measure_and_pairing(environment):
    K = 10
    checks = []
    A = PureGAbeKAlgebra(su_2())
    d = su_2()
    labels = {"1": ((0,), (0,)), "L_01": ((0,), (1,)), "L_02": ((0,), (2,)), "L_20": ((1,), (0,)), "L_21": ((1,), (1,)), "L_40": ((2,), (0,))}
    pref = qpoch_series(1, 2, (0,), K) * qpoch_series(1, 2, (0,), K)     # (q^2;q^2)_∞^{2 rk}, rk = 1
    def measure_factor(m):
        out = S2.one(K, 1)
        for alpha in d.roots():
            s = abs(d.shift_pairing(tuple(m), tuple(alpha)))
            out = out * qpoch_series(1, s, tuple(alpha), K) * qpoch_series(1, s + 2, tuple(alpha), K)
        return out
    res = {name: {m: tr_to_rat(r, 1) for m, r in A.chart(lab).residuals().items()} for name, lab in labels.items()}
    # eq:measure: Tr f = (q^2;q^2)^2/|W| CT_v[ f_0(v) Π_α (v^α;q^2)(q^2 v^α;q^2) ]
    ok_tr = True
    detail = ""
    for name, lab in labels.items():
        f0 = res[name].get((0,))
        if f0 is None:
            exp = S2({}, K, 0)
        else:
            integrand = su2_series(f0, K) * measure_factor((0,))
            exp = constant_term(pref * integrand)
        gotS = rps_int_S(A.trace(lab, K), K)
        # |W| = 2: the constant term is twice the trace
        ok = (exp.eq_to(gotS.scale(2), K))
        ok_tr &= ok
        if not ok and not detail:
            detail = f"{name}: class {gotS}, formula/2 {exp}"
    checks.append(check(f"eq:measure  Tr f = (q^2;q^2)^2_∞/|W| ∮ f_0 Π_α (v^α;q^2)(q^2 v^α;q^2) on six SU(2) lines through q^{K} (independent series; contour integral = constant term)", ok_tr, detail))
    # eq:Iexplicit: I_fg = (q^2;q^2)^2/|W| Σ_m CT[ f_m(v^-1) g_m(v) Π_α (q^|<m,α>| v^α;q^2)(q^(2+|<m,α>|) v^α;q^2) ]
    ok_ip = True
    detail = ""
    n_pairs = 0
    for na, la in labels.items():
        for nb, lb in labels.items():
            total = S2({}, K, 0)
            for m in set(res[na]) | set(res[nb]):
                if m not in res[na] or m not in res[nb]:
                    continue
                fa = invert_vars(su2_series(res[na][m], K), {0})
                gb = su2_series(res[nb][m], K)
                total = total + constant_term(pref * fa * gb * measure_factor(m))
            gotS = rps_int_S(A.inner_product(la, lb, K), K)
            ok = total.eq_to(gotS.scale(2), K)
            ok_ip &= ok
            n_pairs += 1
            if not ok and not detail:
                detail = f"({na},{nb}): class {gotS}, formula/2 {total}"
    checks.append(check(f"eq:Iexplicit, the conjectural half  I_fg as the sum over m of contour integrals against the |<m,α>|-shifted measure (each sector at |v| = 1) equals Tr(rho(f)·g) as the class computes it (the Schur residue of the magnetic-0 residual of rho(f)·g, assembled sector by sector), on all {n_pairs} pairs of six SU(2) lines through q^{K}", ok_ip, detail))
    checks.append(check("orthonormality read off the formula: I_ab[q^0] = δ_ab on the six lines (the leading Weyl orbit contributes δ, the bubbling O(q))", all((rps_int_S(A.inner_product(la, lb, 4), 4).c.get(0, {}).get((), 0) == (1 if la == lb else 0)) for la in labels.values() for lb in labels.values())))
    # the adjoint SO(3) case with μ: eq:measure and eq:Iexplicit with the matter denominators
    G = GNAbeKAlgebra(so_n(3), (1,), nf=1)
    d3 = so_n(3)
    K3 = 8
    weights = rep_weights(d3, (1,))
    pref3 = qpoch_series(2, 2, (0, 0), K3) * qpoch_series(2, 2, (0, 0), K3)
    def measure3(m):
        out = S2.one(K3, 2)
        for alpha in d3.roots():
            s = abs(d3.shift_pairing(tuple(m), tuple(alpha)))
            out = out * qpoch_series(2, s, tuple(alpha) + (0,), K3) * qpoch_series(2, s + 2, tuple(alpha) + (0,), K3)
        for w in weights:
            s = abs(d3.shift_pairing(tuple(m), tuple(w)))
            out = out * inv_qpoch_series(2, 1 + s, tuple(w) + (1,), K3, sign=-1) * inv_qpoch_series(2, 1 + s, tuple(-x for x in w) + (-1,), K3, sign=-1)
        return out
    def res3(lab):
        out = {}
        for m, levels in G.chart(lab).residuals().items():
            parts = []
            for lev, trr in levels.items():
                n_, f_ = tr_to_rat(trr, 1)
                n_ = S2({e: {mo + (lev[0],): v for mo, v in mc.items()} for e, mc in n_.c.items()}, BIG, 2)
                parts.append((n_, [(tuple(al) + (0,), k) for al, k in f_]))
            out[m] = rat_sum(parts)
        return out
    labs3 = {"1": (((0,), (0,)), (0,)), "L_01": (((0,), (1,)), (0,)), "L_10": (((1,), (0,)), (0,)), "L_20": (((2,), (0,)), (0,)), "μ": (((0,), (0,)), (1,))}
    r3 = {n: res3(l) for n, l in labs3.items()}
    def to_series2(rat):
        return rat_to_series(rat[0].trunc(K3 + 200), rat[1], 2, K3)
    def rps_to_S(t):
        c = {}
        for e, v in t.coeffs.items():
            terms = {k: n for k, n in v.terms.items() if n} if hasattr(v, "terms") else ({(0,): v} if v else {})
            if terms:
                c[e] = {(k[0] if isinstance(k, tuple) and k else 0,): n for k, n in terms.items()}
        return S2(c, K3, 1)
    ok3t = True
    detail = ""
    for n, l in labs3.items():
        f0 = r3[n].get((0,))
        exp = constant_term(pref3 * to_series2(f0) * measure3((0,))) if f0 is not None else S2({}, K3, 1)
        gotS = rps_to_S(G.trace(l, K3))
        ok = exp.eq_to(gotS.scale(2), K3)
        ok3t &= ok
        if not ok and not detail:
            detail = f"{n}: class {gotS}, formula/2 {exp}"
    checks.append(check(f"eq:measure with matter  SO(3) + adjoint: the μ-dependent measure Π_w (-q μ^(w_f) v^w;q^2)^-1 (-q μ^-w_f v^-w;q^2)^-1 reproduces the class's traces of five lines through q^{K3}", ok3t, detail))
    ok3i = True
    detail = ""
    n3 = 0
    for na, la in labs3.items():
        for nb, lb in labs3.items():
            total = S2({}, K3, 1)
            for m in set(r3[na]) & set(r3[nb]):
                fa = invert_vars(to_series2(r3[na][m]), {0, 1})
                gb = to_series2(r3[nb][m])
                total = total + constant_term(pref3 * fa * gb * measure3(m))
            gotS = rps_to_S(G.inner_product(la, lb, K3))
            ok = total.eq_to(gotS.scale(2), K3)
            ok3i &= ok
            n3 += 1
            if not ok and not detail:
                detail = f"({na},{nb}): class {gotS}, formula/2 {total}"
    checks.append(check(f"eq:Iexplicit with matter, the conjectural half  SO(3) + adjoint on all {n3} pairs of five lines through q^{K3}: the printed sum over m, with the shifted vector and matter factors, equals Tr(rho(f)·g) as the class computes it", ok3i, detail))
    # eq:Iexplicit: I_fg equals Tr(ρ(f)·g) only up to contour shifts, the pole compatibility being conjectural.  The
    # class's pairing is Tr(ρ(f)·g) itself — the Schur residue of the magnetic-0 residual of ρ(f)·g, assembled sector by
    # sector (wrq_torus.inner_by_sector = trace_residual(pairing_residual)) — so the printed-vs-class comparisons above
    # are the conjectural half.  Here a second computation of Tr(ρ(f)·g), KAlgebra's root route multiply-then-trace
    # (ρ(f)·g decomposed into canonical lines, each traced), is compared with the class's: a consistency check of two
    # computations, not the conjecture.  The window is widened for negative q-powers in the structure constants, as in
    # verify_trace_pairing_faces.
    from kalgebra import _coeff_min_exp

    def trace_face(Alg, la, lb, KK):
        x = Alg.multiply(Alg.rho(la), lb)
        neg = min([0] + [_coeff_min_exp(c) for c in x.terms.values() if not c.is_zero()])
        return Alg.trace_element(x, KK - neg)
    n_f, ok_f, detail = 0, True, ""
    for na, la in labels.items():
        for nb, lb in labels.items():
            n_f += 1
            if not rps_int_S(A.inner_product(la, lb, K), K).eq_to(rps_int_S(trace_face(A, la, lb, K), K), K):
                ok_f = False
                detail = detail or f"SU(2) ({na},{nb})"
    n_f3, ok_f3 = 0, True
    for na, la in labs3.items():
        for nb, lb in labs3.items():
            n_f3 += 1
            if not rps_to_S(G.inner_product(la, lb, K3)).eq_to(rps_to_S(trace_face(G, la, lb, K3)), K3):
                ok_f3 = False
                detail = detail or f"SO(3)+adjoint ({na},{nb})"
    checks.append(check(f"eq:Iexplicit, consistency of two computations of Tr(ρ(f)·g)  the class's pairing (the magnetic-0 residual assembled sector by sector) equals multiply-then-trace (ρ(f)·g by the cocycle convolution, decomposed into canonical lines, each traced) on all {n_f} SU(2) pairs through q^{K} and all {n_f3} SO(3)+adjoint pairs through q^{K3}", ok_f and ok_f3, detail))
    # rank two: the prefactor (q^2;q^2)^{2 rk}/|W| at rk = 2, |W| = 6 — at rank one 2 rk = |W| = 2, so a wrong rank
    # dependence is invisible there
    d_su3 = su_n(3)
    A_su3 = PureGAbeKAlgebra(d_su3)
    Ks = 6
    pref_su3 = S2.one(Ks, 2)
    for _ in range(4):
        pref_su3 = pref_su3 * qpoch_series(2, 2, (0, 0), Ks)
    pref_rank1 = qpoch_series(2, 2, (0, 0), Ks) * qpoch_series(2, 2, (0, 0), Ks)
    meas_su3 = S2.one(Ks, 2)
    for alpha in d_su3.roots():
        meas_su3 = meas_su3 * qpoch_series(2, 0, tuple(alpha), Ks) * qpoch_series(2, 2, tuple(alpha), Ks)
    labs_su3 = {"1": ((0, 0), (0, 0)), "L_0,(1,0)": ((0, 0), (1, 0)), "L_0,(1,1)": ((0, 0), (1, 1)), "L_(1,1),0": ((1, 1), (0, 0)), "L_(1,1),(1,0)": ((1, 1), (1, 0))}
    ok_su3, wrong_caught, detail = True, False, ""
    for n, l in labs_su3.items():
        lab = A_su3.fold(*l)
        f0 = A_su3.chart(lab).residuals().get((0, 0))
        if f0 is None or f0.simplify().is_zero():
            ser = S2({}, Ks, 2)
        else:
            num, facs = tr_to_rat(f0, 2)
            ser = rat_to_series(num.trunc(Ks + 200), facs, 2, Ks)
        exp = constant_term(constant_term(pref_su3 * ser * meas_su3))
        wrong = constant_term(constant_term(pref_rank1 * ser * meas_su3))
        got = rps_int_S(A_su3.trace(lab, Ks), Ks)
        ok = exp.eq_to(got.scale(6), Ks)
        ok_su3 &= ok
        wrong_caught |= not wrong.eq_to(got.scale(2), Ks)
        if not ok and not detail:
            detail = f"{n}: class {got}, formula/6 {exp}"
    checks.append(check(f"eq:measure at rank two  SU(3): Tr f = (q^2;q^2)^4_∞/6 ∮ f_0 Π_α (v^α;q^2)(q^2 v^α;q^2) on {len(labs_su3)} lines through q^{Ks} (rk = 2, |W| = 6)", ok_su3, detail))
    checks.append(check("negative control: the rank-one prefactor (q^2;q^2)^2_∞/2 at SU(3) is detected", wrong_caught))
    checks.append(check("negative control: the pairing formula with the unshifted measure (|<m,α>| dropped) fails at m != 0", not sum((constant_term(pref * invert_vars(su2_series(res['L_20'][m], K), {0}) * su2_series(res['L_20'][m], K) * measure_factor((0,))) for m in res['L_20']), S2({}, K, 0)).eq_to(rps_int_S(A.inner_product(labels['L_20'], labels['L_20'], K), K).scale(2), K)))
    wide = measure_widened(environment)
    checks += wide["checks"]
    return {"checks": checks, "population": {"SU(2) lines": 6, "pairs": n_pairs, "SO(3)+adjoint lines": 5, "K": K, "SU(3) lines (trace)": len(labs_su3), "trace-face pairs": n_f + n_f3, "widened (2026-09-23)": wide["population"]},
            "controls": {"positive": "orthonormality read off the formula", "negative": "the unshifted measure fails; the rank-one prefactor fails at SU(3)"},
            "notes": "The repository computes the trace and pairing through the rational sector measure B_m (kalgebra.md, proved equal to the printed infinite products); here the printed formulas are evaluated directly, so this is an independent check of that equality on the lines listed.  The class's pairing is Tr(ρ(f)·g) itself — the Schur residue of the magnetic-0 residual of ρ(f)·g, assembled sector by sector (wrq_torus.inner_by_sector = trace_residual(pairing_residual)) — so the printed-vs-class pairing comparisons are the conjectural half of eq:Iexplicit (each printed sector sits at |v| = 1 against its own shifted measure; moving it onto the common contour is the shift the paper conjectures pole-compatible).  The comparison of the class's pairing with multiply-then-trace compares two computations of Tr(ρ(f)·g) and is a consistency check (relabelled 2026-09-23 after review: it had been named the conjectural half).", "inputs": []}


# --------------------------------------------------------------------------
# eq:cocha, the SU(2) examples, the adjoint examples
# --------------------------------------------------------------------------
def printed_cocha(datum, m, nv=1, matter_weights=(), mu_index=None):
    """cocha_m(v) = (-q)^{½ Σ_{<α,m>>0} <α,m>} Π_{α:<α,m>>0} Π_{k even, -2<α,m> < k <= 0} (1 - q^k v^{-α})^{-1} × Π_w Π_{k odd, 2<m,w> < k < 0} (1 + q^k μ v^w).
    Returns (num, factors); requires the prefactor exponent to be an integer."""
    half = sum(datum.shift_pairing(tuple(m), tuple(alpha)) for alpha in datum.roots() if datum.shift_pairing(tuple(m), tuple(alpha)) > 0)
    assert half % 2 == 0, "half-integral prefactor (the repository ships the floored psi there)"
    s = half // 2
    num = S2.term(s, (0,) * nv, BIG, nv, (-1) ** s)
    factors = []
    for alpha in datum.roots():
        p = datum.shift_pairing(tuple(m), tuple(alpha))
        if p > 0:
            for k in range(-2 * p + 2, 1, 2):
                factors.append((tuple(-x for x in alpha) + ((0,) if nv == 2 else ()), k))
    for w in matter_weights:
        p = datum.shift_pairing(tuple(m), tuple(w))
        for k in range(2 * p + 1, 0, 2):
            num = num * (S2.one(BIG, nv) + S2.term(k, tuple(w) + (1,), BIG, nv))
    return num, factors



# --------------------------------------------------------------------------
# the printed cocycle and trivialization with matter, in padded variables (v_1..v_d, mu_1..mu_M)
# --------------------------------------------------------------------------
def _int(x):
    """An integral pairing as an int.  At a non-simply-connected form the cells carry Fraction coordinates, so
    <a, alpha> and <a, w> arrive as integral Fractions; a genuinely fractional one would mean the form admitted a
    representation it does not have, and is refused."""
    x = Fraction(x)
    if x.denominator != 1:
        raise ValueError(f"non-integral pairing {x}: the matter is not a representation of this form")
    return int(x)


class Printed:
    """The draft's eq:ccclosed / eq:ccprod / eq:cocha as exact rational functions in (q, v, mu):
    (numerator S2 polynomial, list of denominator factors (monomial, k) meaning (1 - q^k x^monomial))."""

    def __init__(self, D, M):
        self.D, self.M, self.nv = D, M, D.dim + M

    def pad(self, lam, s=None):
        mu = [0] * self.M
        if s is not None:
            mu[s] = 1
        return tuple(lam) + tuple(mu)

    def one(self):
        return S2.one(BIG, self.nv)

    def term(self, e, mono, c=1):
        return S2.term(e, mono, BIG, self.nv, c)

    def bar(self, s):
        return S2({-e: dict(c) for e, c in s.c.items()}, BIG, self.nv)

    def vec(self, A, B, alpha):
        A, B = _int(A), _int(B)
        if A * B >= 0:
            return self.one(), []
        p, r = min(abs(A), abs(B)), abs(abs(A) - abs(B))
        x = self.pad(alpha)
        num = self.term(abs(A * B), tuple(p * t for t in x))
        dens = [(x, r + 2 * j) for j in range(p)] + [(x, r + 2 + 2 * j) for j in range(p)]
        if A > 0 > B:
            return num, dens
        return self.bar(num), [(mo, -k) for mo, k in dens]

    def hyp(self, A, B, w, s):
        A, B = _int(A), _int(B)
        if A * B >= 0:
            return self.one()
        p, r = min(abs(A), abs(B)), abs(abs(A) - abs(B))
        out = self.one()
        for j in range(p):
            e = r + 1 + 2 * j
            out = out * (self.one() + self.term(e if A > 0 else -e, self.pad(w, s)))
        return out

    def cocy(self, a, b, slots):
        num, dens = self.one(), []
        for alpha in self.D.positive_roots():
            n2, d2 = self.vec(self.D.shift_pairing(a, tuple(alpha)), self.D.shift_pairing(b, tuple(alpha)), alpha)
            num, dens = num * n2, dens + d2
        for s, wts in enumerate(slots):
            for w in wts:
                num = num * self.hyp(self.D.shift_pairing(a, tuple(w)), self.D.shift_pairing(b, tuple(w)), w, s)
        return num, dens

    def cocha(self, m, slots, old=False):
        """(num, dens, s): eq:cocha with its (-q)^s prefactor exponent s kept apart (a Fraction).  old=True is the
        all-positive-roots form commented out in the draft source."""
        roots = self.D.positive_roots() if old else self.D.roots()
        s = Fraction(0)
        dens = []
        for alpha in roots:
            p = _int(self.D.shift_pairing(m, tuple(alpha)))
            if old or p > 0:
                s += Fraction(p, 2)
            for k in range(-2 * p + 2, 1, 2):          # even k, -2p < k <= 0 (empty for p <= 0)
                dens.append((self.pad(tuple(-t for t in alpha)), k))
        num = self.one()
        for si, wts in enumerate(slots):
            for w in wts:
                c = _int(self.D.shift_pairing(m, tuple(w)))
                for k in range(2 * c + 1, 0, 2):       # odd k, 2c < k < 0
                    num = num * (self.one() + self.term(k, self.pad(w, si)))
        return num, dens, s

    def shift(self, nd, c):
        """f(q^c v): v^lambda -> q^{<c,lambda>} v^lambda, mu untouched."""
        num, dens = nd
        out = {}
        for e, mc in num.c.items():
            for mono, v in mc.items():
                sh = self.D.shift_pairing(c, mono[:self.D.dim])
                out.setdefault(e + sh, {})[mono] = out.get(e + sh, {}).get(mono, 0) + v
        return S2(out, BIG, self.nv), [(mo, k + self.D.shift_pairing(c, mo[:self.D.dim])) for mo, k in dens]

    def trivializes(self, a, b, slots, old=False):
        """cocy_{a,b}(q^{a+b} v) cocha_{a+b}(v) == (-q)^{s(a)+s(b)-s(a+b)} cocha_a(v) cocha_b(q^{2a} v); None if the
        net (-q) exponent is not an integer."""
        ab = tuple(x + y for x, y in zip(a, b))
        cn, cd = self.shift(self.cocy(a, b, slots), ab)
        h_ab, h_a, h_b = self.cocha(ab, slots, old), self.cocha(a, slots, old), self.cocha(b, slots, old)
        net = h_a[2] + h_b[2] - h_ab[2]
        if net.denominator != 1:
            return None
        net = int(net)
        hb = self.shift((h_b[0], h_b[1]), tuple(2 * x for x in a))
        rhs = self.term(net, (0,) * self.nv, (-1) ** (net % 2)) * h_a[0] * hb[0]
        return rat_eq(cn * h_ab[0], cd + h_ab[1], rhs, h_a[1] + hb[1], self.nv)


def printed_matter_factor_levels(D, a, slots, dual=False):
    """{mu-exponent vector: TorusRational}: the matter factor of eq:cocha,
    prod_s prod_{w in slot s} prod_{k odd, 2<a,w> < k < 0} (1 + q^k mu_s v^w); dual=True takes the weights of N^vee
    (the negative control)."""
    from weyl_torus_ring import TorusRational, TorusLaurent
    out = {(0,) * len(slots): TorusRational.one(D)}
    for si, wts in enumerate(slots):
        for w in wts:
            w = tuple(-x for x in w) if dual else tuple(w)
            c = _int(D.shift_pairing(tuple(a), w))
            for k in range(2 * c + 1, 0, 2):
                mono = TorusRational(D, TorusLaurent(D, {w: LaurentPoly({k: 1})}), {})
                new = {}
                for key, t in out.items():
                    new[key] = (new[key] + t) if key in new else t
                    k2 = tuple(x + (1 if i == si else 0) for i, x in enumerate(key))
                    tt = t * mono
                    new[k2] = (new[k2] + tt) if k2 in new else tt
                out = new
    return out


def paper_d_levels(x, D, slots, matter_factor="printed"):
    """{mu-power: {cell a: d_a}} with d_a = f_a(q^a v) cocha_a(v, mu) — f the stored residuals (the paper's f: the
    class's product law uses the printed cocycle), cocha's pure part the code's dressing_psi (floored; the common
    (-q)-power the draft's note after the conjecture discards), its matter part printed.  matter_factor: 'printed',
    'dual' (N^vee, negative control) or 'none' (negative control)."""
    from weyl_torus_ring import TorusRational
    out = {}
    for a, row in x.residuals().items():
        psi = dressing_psi(D, a)
        if matter_factor == "none":
            Ma = {(0,) * len(slots): TorusRational.one(D)}
        else:
            Ma = printed_matter_factor_levels(D, a, slots, dual=(matter_factor == "dual"))
        rows = row.items() if isinstance(row, dict) else [((), row)]
        for k, fr in rows:
            base = fr.q_shift(tuple(a)) * psi
            for i, mt in Ma.items():
                j = tuple(p + r for p, r in zip(k, i)) if k else i
                cell = out.setdefault(j, {})
                term = base * mt
                cell[a] = (cell[a] + term) if a in cell else term
    return {j: {a: t.simplify() for a, t in cells.items()} for j, cells in out.items()}


def paper_A3(D, Dd):
    """The draft's A3 on one d-dict: (simple poles only, pair residues cancel)."""
    import star_bubbling as SB
    from weyl_torus_ring import TorusRational
    simple = all(m <= 1 for d in Dd.values() for m in d.simplify().den.values())
    walls = set()
    for d in Dd.values():
        walls.update(d.simplify().den)
    ok = True
    for (alpha, k) in sorted(walls):
        cor = SB.coroot_of(D, alpha)
        if cor is None or k % 2:
            return simple, False
        seen = set()
        for a in list(Dd):
            if a in seen:
                continue
            p = SB.wall_partner(D, a, alpha, k, cor)
            seen.add(a)
            seen.add(p)
            s = (Dd.get(a, TorusRational.zero(D)) + Dd.get(p, TorusRational.zero(D))).simplify()
            if s.den.get((tuple(alpha), k), 0) > 0 and not s.is_zero():
                ok = False
    return simple, ok


def check_cocha(environment):
    from matter_wrq_torus import slot_weights
    from root_datum import b_n_simply_connected, product_datum
    from g_matter_roster import highest_root
    checks = []
    t0 = time.time()
    ok = True
    detail = ""
    for m in (1, 2, 3):
        got = tr_to_rat(dressing_psi(su_2(), (m,)), 1)
        exp = printed_cocha(su_2(), (m,))
        r = rat_eq(*got, *exp, 1)
        ok &= r
        if not r and not detail:
            detail = f"SU(2) m={m}: psi {dressing_psi(su_2(), (m,))}"
    checks.append(check("eq:cocha  the SU(2) dressing psi_m equals the printed cocha_m (prefactor (-q)^m, the k-even denominators) for m = 1, 2, 3", ok, detail))
    ok2 = True
    for m in (2, 4):
        ok2 &= rat_eq(*tr_to_rat(dressing_psi(so_n(3), (m,)), 1), *printed_cocha(so_n(3), (m,)), 1)
    checks.append(check("eq:cocha  SO(3) at even m (integral prefactor): psi_m equals cocha_m", ok2))
    half = sum(so_n(3).shift_pairing((1,), tuple(a)) for a in so_n(3).roots() if so_n(3).shift_pairing((1,), tuple(a)) > 0)
    checks.append(check("at odd m of SO(3) the printed prefactor (-q)^(½Σ<α,m>) is a half-integral power (½·1): the repository ships the floored psi and restores the phase through the cocycle — recorded, not compared", half % 2 == 1))
    checks.append(check("negative control: cocha with the prefactor (+q)^m is detected at m = 1", not rat_eq(*tr_to_rat(dressing_psi(su_2(), (1,)), 1), S2.term(1, (0,), BIG, 1), printed_cocha(su_2(), (1,))[1], 1)))
    # rank two, non-simply-laced included, every m of a box (dominant or not: the roots positive on m make the formula
    # Weyl-covariant, and dressing_psi is Weyl-transported from the dominant representative)
    r2 = list(itertools.product(range(-2, 3), repeat=2))
    n_r2, ok_r2, bad = 0, True, ""
    for name, D in (("SU(3)", su_n(3)), ("Sp(4)", sp_n(2)), ("Spin(5)", b_n_simply_connected(2)), ("G2", g_2())):
        Pp = Printed(D, 0)
        for m in r2:
            num, dens, s_ = Pp.cocha(m, ())
            if s_.denominator != 1:
                continue            # not on a simply connected datum (every m in Q^vee has <Σ⁺, m> even)
            exp = (Pp.term(int(s_), (0,) * D.dim, (-1) ** (int(s_) % 2)) * num, dens)
            got = tr_to_rat(dressing_psi(D, m), D.dim)
            n_r2 += 1
            if not rat_eq(*got, *exp, D.dim):
                ok_r2 = False
                bad = bad or f"{name} m={m}"
    checks.append(check(f"eq:cocha at rank two  dressing_psi equals the printed pure cocha at every m of the box [-2,2]^2 at SU(3), Sp(4), Spin(5), G2 ({n_r2} cocharacters, dominant or not)", ok_r2, bad))
    # the draft's formulas among themselves: eq:cocha trivializes eq:ccclosed / eq:ccprod, matter included:
    #   cocy_{a,b}(q^{a+b} v) cocha_{a+b}(v) = (-q)^{s(a)+s(b)-s(a+b)} cocha_a(v) cocha_b(q^{2a} v)
    # (from eq:cocydef with U_m = cocha_m(v) u^m, u^m g(v) = g(q^{2m} v) u^m, and d_m = f_m(q^m v) cocha_m(v))
    su2, su3, b2, g2 = su_2(), su_n(3), b_n_simply_connected(2), g_2()
    r1 = [(x,) for x in range(-2, 3)]
    rr = list(itertools.product(range(-1, 2), repeat=2))
    theories = [
        ("pure SU(2)", su2, [], r1), ("pure SO(3)", so_n(3), [], r1), ("pure SU(3)", su3, [], rr), ("pure Sp(4)", sp_n(2), [], rr),
        ("pure Spin(5)", b2, [], rr), ("pure G2", g2, [], rr), ("pure SU(2)xSU(2)", product_datum([su2, su2]), [], rr),
        ("SU(2)+1", su2, [(1,)], r1), ("SU(2)+2", su2, [(1,), (1,)], r1), ("SO(3)+adjoint", so_n(3), [(1,)], r1),
        ("SU(3)+1", su3, [(1, 0)], rr), ("SU(3)+symmetric", su3, [(2, 0)], rr), ("SU(3)+symmetric+1", su3, [(2, 0), (1, 0)], rr),
        ("SU(3)+adjoint", su3, [tuple(highest_root(su3))], rr), ("Spin(5)+vector", b2, [(1, 0)], rr), ("G2+7", g2, [(1, 0)], rr),
        ("SU(2)xSU(2)+bifundamental", product_datum([su2, su2]), [(1, 1)], rr),
        ("[2]-SU(2)-SU(2)-[2]", product_datum([su2, su2]), [(1, 1), (1, 0), (1, 0), (0, 1), (0, 1)], rr),
    ]
    triv = {}
    for name, D, matter, box in theories:
        slots = slot_weights(D, matter) if matter else ()
        Pp = Printed(D, len(slots))
        res = [Pp.trivializes(a, b, slots) for a in box for b in box]
        triv[name] = (sum(1 for x in res if x), len(res))
    n_t = sum(v[1] for v in triv.values())
    ok_t = all(v[0] == v[1] for v in triv.values())
    checks.append(check(f"the draft's eq:cocha trivializes its eq:ccclosed/eq:ccprod, matter factors included: cocy_ab(q^(a+b)v) cocha_(a+b)(v) = (-q)^(s(a)+s(b)-s(a+b)) cocha_a(v) cocha_b(q^(2a)v), the net (-q) exponent integral, on all {n_t} pairs of {len(theories)} theories (seven pure groups incl. Sp(4), Spin(5), G2; complex matter: SU(3) fundamental, symmetric, both; both SU(2)-SU(2) quivers)", ok_t, "; ".join(f"{k} {v[0]}/{v[1]}" for k, v in triv.items() if v[0] != v[1])))
    old = {}
    for name, D, matter, box in theories[:6]:
        Pp = Printed(D, 0)
        res = [Pp.trivializes(a, b, (), old=True) for a in box for b in box]
        old[name] = sum(1 for x in res if not x)
    checks.append(check("negative control: the older all-positive-roots cocha (commented out in the draft source) fails the identity at every pure group tested (" + ", ".join(f"{k} {v}" for k, v in old.items()) + " failing pairs)", all(v > 0 for v in old.values())))
    return {"checks": checks, "population": {"SU(2) m": "1..3", "SO(3) m": "2, 4", "rank-two cocharacters": n_r2, "trivialization pairs": n_t, "theories": {k: f"{v[0]}/{v[1]}" for k, v in triv.items()}},
            "controls": {"positive": "exact rational identities", "negative": "wrong sign detected; the older all-positive-roots cocha fails the trivialization identity"},
            "notes": f"The trivialization identity follows from eq:cocydef, f = sum f_m(q^m v) U_m = sum d_m u^m and d_m = f_m(q^m v) cocha_m(v): U_m = cocha_m(v) u^m.  It tests the draft's printed formulas among themselves; the code's cocycle is compared with the printed one in sec:coulomb/presentation, and the code's lines are tested against A3 with the printed matter factors in conj:abeKalgebra/existence-uniqueness.  Total {round(time.time() - t0, 1)} s.", "inputs": []}


def check_su2_examples(environment):
    checks = []
    A = PureGAbeKAlgebra(su_2())
    d = su_2()
    # eq:su2thooft: L_02 = v^-2 + 1 + v^2 ; L_20 = U_2 + (q+q^-1) v^2/((1-q^2 v^2)(1-q^-2 v^2)) + U_-2
    r02 = A.chart(((0,), (2,))).residuals()
    r20 = A.chart(((1,), (0,))).residuals()
    ok = set(r02) == {(0,)} and rat_eq(*tr_to_rat(r02[(0,)], 1), poly(1, {(0, (-2,)): 1, (0, (0,)): 1, (0, (2,)): 1}), [], 1)
    f0_exp = (poly(1, {(1, (2,)): 1, (-1, (2,)): 1}), [((2,), 2), ((2,), -2)])
    ok &= set(r20) == {(-1,), (0,), (1,)} and rat_eq(*tr_to_rat(r20[(1,)], 1), S2.one(BIG, 1), [], 1) and rat_eq(*tr_to_rat(r20[(-1,)], 1), S2.one(BIG, 1), [], 1) and rat_eq(*tr_to_rat(r20[(0,)], 1), *f0_exp, 1)
    checks.append(check("eq:su2thooft  L_02 = v^-2 + 1 + v^2 and L_20 = U_2 + (q+q^-1) v^2/((1-q^2v^2)(1-q^-2v^2)) + U_-2, residual by residual", ok))
    checks.append(check("L_20 = L_10^2 in the SO(3) form where the minuscule 't Hooft line exists: PureGAbeKAlgebra(so_n(3)).multiply(L_10, L_10) = L_20", PureGAbeKAlgebra(so_n(3)).multiply(((1,), (0,)), ((1,), (0,))) == elem({((2,), (0,)): {0: 1}})))
    # eq:su2dform: d_a = f_a(q^a v) cocha_a(v) from the PRINTED f and cocha, against the printed d's; and the class's conventional form
    f = {(-1,): (S2.one(BIG, 1), []), (0,): f0_exp, (1,): (S2.one(BIG, 1), [])}
    d_printed = {(-1,): (poly(1, {(1, (0,)): -1}), [((2,), 0), ((2,), -2)]),
                 (0,): (poly(1, {(1, (2,)): 1, (-1, (2,)): 1}), [((2,), -2), ((2,), 2)]),
                 (1,): (poly(1, {(3, (4,)): -1}), [((2,), 0), ((2,), 2)])}
    d_mine = {}
    for a, (n_, f_) in f.items():
        sh = (shift_poly(n_, a, d), shift_factors(f_, a, d))
        ch = printed_cocha(d, a) if a[0] > 0 else ((S2.one(BIG, 1), []) if a[0] == 0 else printed_cocha_neg(d, a))
        d_mine[a] = rat_mul(sh, ch)
    ok_d = all(rat_eq(*d_mine[a], *d_printed[a], 1) for a in d_printed)
    checks.append(check("eq:su2dform  d_a = f_a(q^a v) cocha_a(v) from the printed f's and cocha reproduces the printed d_-2, d_0, d_2", ok_d, "" if ok_d else str({a: (str(d_mine[a][0]), d_mine[a][1]) for a in d_mine})))
    conv = conventional(A.chart(((1,), (0,))))
    ok_c = all(rat_eq(*tr_to_rat(conv[a], 1), *d_printed[a], 1) for a in d_printed if a in conv) and set(conv) == set(d_printed)
    checks.append(check("the class's d-form (star_bubbling.conventional) equals the printed d's", ok_c, "" if ok_c else str({a: str(v) for a, v in conv.items()})))
    # eq:su2res: the three printed sums, exactly
    s1 = rat_sum([d_printed[(-1,)], d_printed[(1,)]])
    s2 = rat_sum([d_printed[(-1,)], d_printed[(0,)]])
    s3 = rat_sum([d_printed[(0,)], d_printed[(1,)]])
    e1 = (poly(1, {(1, (0,)): -1, (3, (2,)): 1, (1, (2,)): -1, (1, (4,)): -1}), [((2,), -2), ((2,), 2)])
    e2 = (poly(1, {(1, (0,)): -1, (1, (2,)): 1, (3, (2,)): 1}), [((2,), 0), ((2,), 2)])
    e3 = (poly(1, {(1, (2,)): 1, (-1, (2,)): 1, (-1, (4,)): -1}), [((2,), -2), ((2,), 0)])
    ok_r = rat_eq(*s1, *e1, 1) and rat_eq(*s2, *e2, 1) and rat_eq(*s3, *e3, 1)
    checks.append(check("eq:su2res  the three residue cancellations: d_-2 + d_2, d_-2 + d_0, d_0 + d_2 equal the printed sums, in which the offending factor has disappeared", ok_r))
    okc, rep = criterion(A.chart(((1,), (0,))))
    checks.append(check("the residue rule (★) holds for L_20 by exact division (star_bubbling.criterion)", okc, str(rep)))
    bare = WRQTorus.from_f(d, {(1,): A.chart(((1,), (0,))).residuals()[(1,)], (-1,): A.chart(((1,), (0,))).residuals()[(-1,)]})
    okb, repb = criterion(bare)
    checks.append(check("without the bubbling correction the residue cancellation fails (U_2 + U_-2 alone)", not okb, str(repb)[:200]))
    f0 = A.chart(((1,), (0,))).residuals()[(0,)]
    ser = su2_series(tr_to_rat(f0, 1), 6)
    checks.append(check("f_0 is bar-invariant, O(q) as a series, and not a Laurent polynomial (so rigid: no Laurent polynomial is both bar-invariant and O(q))", f0.bar().simplify() == f0.simplify() and (ser.val() or 0) >= 1 and bool(f0.den), f"f_0 = {ser.trunc(3)}"))
    checks.append(check("rigidity: the axiom route built L_20 as the unique solution of the (★)-guarded solve (route 'star', certify_canonical passes)", A.route(((1,), (0,))) == "star" and bool(_cert_ok(A, ((1,), (0,))))))
    r21 = A.chart(((1,), (1,))).residuals()
    ok21 = rat_eq(*tr_to_rat(r21[(1,)], 1), poly(1, {(0, (1,)): 1}), [], 1) and rat_eq(*tr_to_rat(r21[(-1,)], 1), poly(1, {(0, (-1,)): 1}), [], 1) and rat_eq(*tr_to_rat(r21[(0,)], 1), poly(1, {(0, (1,)): 1, (0, (3,)): 1}), [((2,), 2), ((2,), -2)], 1)
    checks.append(check("eq:su2dyonic  L_21 = q v U_2 + v(1+v^2)/((1-q^2v^2)(1-q^-2v^2)) + q v^-1 U_-2 (residuals f_±1 = v^±1, whose U-coefficients f(q^±1 v) carry the printed q)", ok21))
    return {"checks": checks, "population": {"lines": "L_02, L_20, L_21 (SU(2)); L_10 (SO(3))"}, "controls": {"positive": "the printed residuals reproduced exactly", "negative": "the bare 't Hooft line fails the residue rule"},
            "notes": "Draft units: m = 2 is the repository's coroot coordinate m = 1 on su_2(); the printed U_±2 are the repository's atoms at m = ±1.", "inputs": []}


def printed_cocha_neg(datum, a):
    """cocha for a with <α,a> < 0 for the positive root: the roots with <α,a> > 0 are the negative ones — same formula."""
    return printed_cocha(datum, a)


def _memory_exhausted(ex):
    """True when `ex` is the per-theory memory budget running out, not the code under test failing: a MemoryError, or
    the SystemError CPython raises when an allocation fails where no exception can be built ("error return without
    exception set") — the latter only when this process's peak address space reached its RLIMIT_AS (the cap
    `_set_memory_limit` sets).  Any other exception, and a SystemError below the cap or with no cap set, is the code's
    and stays a failure.  Measured 2026-09-24: [2]-SU(2)-SU(2)-[2]'s pairing of the identity with itself at K = 3 needs
    over 5 GB (its five matter slots), and in the battery's worker that surfaced as the SystemError (that theory has
    since left the lineup, whose SU(2)-SU(2) quiver has a bifundamental only)."""
    if isinstance(ex, MemoryError):
        return True
    if not isinstance(ex, SystemError):
        return False
    try:
        import resource
        soft, _hard = resource.getrlimit(resource.RLIMIT_AS)
        if soft == resource.RLIM_INFINITY:
            return False
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmPeak:"):
                    return int(line.split()[1]) * 1024 >= soft - 2 ** 26
    except Exception:  # noqa: BLE001
        return False
    return False


def _cert_ok(A, label):
    try:
        r = A.certify_canonical(label)
    except MemoryError:
        raise                          # a resource limit, not a failed certification (the caller reports it unreached)
    except Exception as ex:  # noqa: BLE001
        if _memory_exhausted(ex):
            raise MemoryError(f"{type(ex).__name__} at the memory budget") from None
        return False
    if isinstance(r, tuple):
        return bool(r[0])
    return True if r is None else bool(r)


def check_adjoint(environment):
    checks = []
    G = GNAbeKAlgebra(so_n(3), (1,), nf=1)
    d = so_n(3)
    L10 = (((1,), (0,)), (0,))
    L20 = (((2,), (0,)), (0,))
    mu = (((0,), (0,)), (1,))
    checks.append(check("eq:adjsquare  L_10 · L_10 = L_20 + μ", G.multiply(L10, L10) == elem({L20: {0: 1}, mu: {0: 1}})))
    # eq:adjbubble: f_0(L_20) = μ[(q+q^-1)(μ+μ^-1) v^2 + (1+v^2)^2] / ((1-q^2 v^2)(1-q^-2 v^2)); SO(3) coordinates: v^2 -> v
    r = G.chart(L20).residuals()
    lev = r.get((0,), {})
    exp = {0: (poly(1, {(1, (1,)): 1, (-1, (1,)): 1}), [((1,), 2), ((1,), -2)]),
           1: (poly(1, {(0, (0,)): 1, (0, (1,)): 2, (0, (2,)): 1}), [((1,), 2), ((1,), -2)]),
           2: (poly(1, {(1, (1,)): 1, (-1, (1,)): 1}), [((1,), 2), ((1,), -2)])}
    ok = set(k[0] for k in lev) == set(exp) and all(rat_eq(*tr_to_rat(lev[(k,)], 1), *exp[k], 1) for k in exp)
    checks.append(check("eq:adjbubble  f_0(L_20) = μ[(q+q^-1)(μ+μ^-1)v^2 + (1+v^2)^2]/((1-q^2v^2)(1-q^-2v^2)), level by level in μ (SO(3) coordinates v^2 -> v)", ok, str({k: str(v)[:80] for k, v in lev.items()}) if not ok else ""))
    checks.append(check("the U_±2 coefficients of L_20 are 1 and the U_±1 coefficients of L_10 are 1", all(rat_eq(*tr_to_rat(r[(s,)][(0,)], 1), S2.one(BIG, 1), [], 1) for s in (2, -2)) and all(rat_eq(*tr_to_rat(G.chart(L10).residuals()[(s,)][(0,)], 1), S2.one(BIG, 1), [], 1) for s in (1, -1))))
    weights = rep_weights(d, (1,))
    ok_k = all(kappa(d, (m,), weights)[0] == (0,) and kappa(d, (m,), weights)[1] == m for m in range(0, 5))
    checks.append(check("κ ≡ 0 for adjoint matter (gauge and matter contributions cancel) and κ_f(m) = -m, m = 0..4", ok_k))
    ok_r = all(G.rho(G.fold((m,), (e,), (f,))) == G.fold((-m,), (-e,), (-f - m,)) for m in range(0, 4) for e in range(-3, 4) for f in range(-2, 3))
    checks.append(check("rho is the antipode (m,e) -> (-m,-e) with the flavour shifted by -m, on 175 labels", ok_r))
    # the matter factors of cocha: the class's d-form of L_20 is regular under the residue rule
    checks.append(check("L_20 with adjoint matter passes the class's certification (the matter (★)-guarded solve: unique, bar-invariant, W2) and was built by the axiom route", _cert_ok(G, L20) and G.route(L20) in ("star", "solve"), f"route {G.route(L20)!r} (the matter tier names its guarded solve 'solve')"))
    # The SU(2) form (the draft's subsection is "N=2* SU(2) or SO(3)").  L_10 exists only in SO(3), so eq:adjsquare is
    # SO(3)'s; eq:adjbubble concerns L_20, which both forms carry.  In su_2 coordinates the draft's L_20 is the coroot
    # cell (1,) and its v^2 is v^α = v^(2,), so the printed formula applies verbatim; the draft's κ_f(m) = -m, in its
    # units where the coroot is m = 2, is a flavour shift of -2m in su_2 units.
    from g_matter_roster import highest_root
    Gs = GNAbeKAlgebra(su_2(), highest_root(su_2()), nf=1)
    L20s = (Gs.fold((1,), (0,))[0], (0,))
    rs = Gs.chart(L20s).residuals()
    levs = rs.get((0,), {})
    den_s = [((2,), 2), ((2,), -2)]
    exps = {0: (poly(1, {(1, (2,)): 1, (-1, (2,)): 1}), den_s),
            1: (poly(1, {(0, (0,)): 1, (0, (2,)): 2, (0, (4,)): 1}), den_s),
            2: (poly(1, {(1, (2,)): 1, (-1, (2,)): 1}), den_s)}
    ok_s = (set(k[0] for k in levs) == set(exps) and all(rat_eq(*tr_to_rat(levs[(k,)], 1), *exps[k], 1) for k in exps)
            and all(rat_eq(*tr_to_rat(rs[(s,)][(0,)], 1), S2.one(BIG, 1), [], 1) for s in (1, -1)))
    checks.append(check("eq:adjbubble in the SU(2) form of N=2*: f_0(L_20) as printed, verbatim in SU(2) coordinates (the draft's L_20 is the coroot cell, its v^2 = v^α), level by level in μ, and the U-coefficients of L_20 are 1", ok_s, str({k: str(v)[:80] for k, v in levs.items()}) if not ok_s else ""))
    ok_rs = all(Gs.rho(Gs.fold((m,), (e,), (f,))) == Gs.fold((-m,), (-e,), (-f - 2 * m,)) for m in range(0, 4) for e in range(-3, 4) for f in range(-2, 3))
    checks.append(check("SU(2) form: rho is the antipode with the flavour shifted by -2m in SU(2) units (the draft's κ_f(m) = -m, whose units give the coroot m = 2), on 140 labels", ok_rs))
    return {"checks": checks, "population": {"lines": "SO(3) form: L_10, L_20, μ, 175 labels for rho; SU(2) form: L_20, 140 labels for rho"}, "controls": {"positive": "the printed product and residual reproduced", "negative": "n.a."},
            "notes": "GNAbeKAlgebra(so_n(3), (1,), nf=1) is the N=2* SO(3) theory (the SO(3) fundamental is the adjoint); GNAbeKAlgebra(su_2(), highest_root, nf=1) the SU(2) form; μ is the flavour label (((0,),(0,)),(1,)).", "inputs": []}


# --------------------------------------------------------------------------
# global forms at rank one: eq:su2so3lat
# --------------------------------------------------------------------------
def _factor(n):
    """{p: a} with n = prod p^a (trial division; the centres here are tiny)."""
    out, p = {}, 2
    while n > 1:
        while n % p == 0:
            out[p] = out.get(p, 0) + 1
            n //= p
        p += 1
    return out


def _lagrangian_count_formula(divisors):
    """The number of Lagrangian subgroups of Z x Z^ for the linking pairing,
    computed independently of the repository, prime by prime: a cyclic p-part
    Z_{p^a} contributes 1 + p + ... + p^a (sigma(p^a)), an elementary one
    (Z_p)^n contributes prod_{i=1..n} (p^i + 1).  Other p-parts are outside the
    population and raise."""
    parts = {}
    for d in divisors:
        for p, a in _factor(d).items():
            parts.setdefault(p, []).append(a)
    total = 1
    for p, exps in parts.items():
        if len(exps) == 1:
            total *= sum(p ** i for i in range(exps[0] + 1))
        elif set(exps) == {1}:
            for i in range(1, len(exps) + 1):
                total *= p ** i + 1
        else:
            raise ValueError(f"p-part {p}^{exps} outside the formula's population")
    return total


def _lagrangian_census(datum):
    """Every subgroup of order |Z| of the centre-class pairs Z x Z^, in
    LineLattice's own class labelling, and which of them LineLattice reports as
    Dirac-integral (isotropic; with order |Z| that is Lagrangian).  Returns
    (divisors, #order-|Z| subgroups, list of Lagrangian LineLattices)."""
    from global_form import LineLattice
    from math import gcd
    divs = tuple(LineLattice(datum).divisors)
    Z = list(itertools.product(*[range(d) for d in divs]))
    nZ = len(Z)
    zero = tuple(0 for _ in divs)

    def add1(a, b):
        return tuple((x + y) % d for x, y, d in zip(a, b, divs))

    def add(p_, q_):
        return (add1(p_[0], q_[0]), add1(p_[1], q_[1]))

    def order(g):
        o = 1
        for part in (g[0], g[1]):
            for x, d in zip(part, divs):
                k = d // gcd(x, d) if x else 1
                o = o * k // gcd(o, k)
        return o

    def join(S, g):
        mult = [(zero, zero)]
        for _ in range(order(g) - 1):
            mult.append(add(mult[-1], g))
        return frozenset(add(s, m) for s in S for m in mult)

    elems = [(k, l) for k in Z for l in Z]
    start = frozenset({(zero, zero)})
    seen, queue, full = {start}, [start], []
    while queue:
        S = queue.pop()
        if len(S) == nZ:
            full.append(S)
            continue
        for g in elems:
            if g not in S:
                T = join(S, g)
                if len(T) <= nZ and T not in seen:
                    seen.add(T)
                    queue.append(T)
    lag = []
    for S in full:
        L = LineLattice(datum, classes=sorted(S))
        if L.verify_dirac_integral():
            lag.append(L)
    return divs, len(full), lag


def _centre_subgroup_count(divs):
    """Number of subgroups of Z (brute force over generator sets; Z is tiny)."""
    from math import gcd
    Z = list(itertools.product(*[range(d) for d in divs]))
    zero = tuple(0 for _ in divs)
    subs = {frozenset({zero})}
    changed = True
    while changed:
        changed = False
        for S in list(subs):
            for g in Z:
                if g in S:
                    continue
                T = set(S)
                frontier = [g]
                while frontier:
                    x = frontier.pop()
                    for s in list(T):
                        y = tuple((a + b) % d for a, b, d in zip(s, x, divs))
                        if y not in T:
                            T.add(y)
                            frontier.append(y)
                T = frozenset(T)
                if T not in subs:
                    subs.add(T)
                    changed = True
    return len(subs)


def check_global_forms(environment):
    from global_form import LineLattice
    checks = []
    su2 = LineLattice(su_2())
    so3 = LineLattice(so_n(3))
    # draft units: m in (co)weight integers, alpha = 2: SU(2) has m even (coroot lattice), any e; SO(3) any m, e even (root lattice)
    ok_su2 = all(su2.admits((m,), (e,)) for m in range(-2, 3) for e in range(-3, 4))     # repo su_2: m in coroot units = every integer m_repo (m_draft = 2 m_repo), e any weight
    ok_so3 = all(so3.admits((m,), (e,)) for m in range(-3, 4) for e in range(-2, 3))     # repo so_n(3): m any coweight, e in the root lattice basis
    checks.append(check("eq:su2so3lat  K_q[SU(2)]: every (m ∈ Q^∨, e ∈ P) is a line; K_q[SO(3)]: every (m ∈ P^∨, e ∈ Q) is a line (the classes' own coordinates)", ok_su2 and ok_so3))
    checks.append(check("both lattices are Dirac-integral (mutually local) and maximal", su2.verify_dirac_integral() and so3.verify_dirac_integral() and su2.is_maximal() and so3.is_maximal()))
    forms = {}
    for gens in ([(1, 0)], [(0, 1)], [(1, 1)], [(1, 0), (0, 1)]):
        try:
            Lt = LineLattice(su_2(), classes=gens)
            forms[tuple(gens)] = (Lt, Lt.verify_dirac_integral(), Lt.is_maximal())
        except Exception as ex:  # noqa: BLE001
            forms[tuple(gens)] = (None, f"{type(ex).__name__}: {ex}", None)
    good = [g for g, (Lt, i, mx) in forms.items() if Lt is not None and i and mx]
    checks.append(check("the three order-two subgroups of the centre-class pairs (Z/2)^2 — ⟨(1,0)⟩ (SO(3)), ⟨(0,1)⟩ (SU(2)), ⟨(1,1)⟩ (the third form) — give Dirac-integral maximal lattices, and the full group does not",
                        set(good) == {((1, 0),), ((0, 1),), ((1, 1),)} and forms[((1, 0), (0, 1))][1] is False, str({g: (i, mx) for g, (Lt, i, mx) in forms.items()})))
    su2_via_classes = forms[((0, 1),)][0]
    checks.append(check("LineLattice(su_2()) (H trivial) is the SU(2) lattice ⟨(0,1)⟩: same membership on integer labels", su2_via_classes is not None and all(su2.admits((m,), (e,)) == su2_via_classes.admits((m,), (e,)) for m in range(-2, 3) for e in range(-2, 3))))
    third = forms[((1, 1),)][0]
    pattern = {}
    if third is not None:
        for md in range(-2, 3):      # draft units: m_draft = 2 m_repo (coroot basis), e_draft = e_repo
            for e in range(-2, 3):
                try:
                    pattern[(md, e)] = third.admits((md / 2,), (e,)) if md % 2 else third.admits((md // 2,), (e,))
                except Exception as ex:  # noqa: BLE001
                    pattern[(md, e)] = f"{type(ex).__name__}"
        ok_third = all(v == ((md - e) % 2 == 0) for (md, e), v in pattern.items() if isinstance(v, bool)) and all(isinstance(v, bool) for v in pattern.values())
    else:
        ok_third = False
    checks.append(check("the third form admits exactly the labels with e - m even (draft units m ∈ Z coweights, e ∈ Z weights), half-integral coroot coordinates presented on the coweight torus", ok_third, str(pattern)))
    checks.append(check("the Dirac pairing is integral on the third form's lines (verify_mutually_local on a window)", third is not None and third.verify_mutually_local([((md // 2 if md % 2 == 0 else md / 2,), (e,)) for (md, e), v in pattern.items() if v is True])))
    # the census: every simply connected root datum of rank <= 4 in the repository (D4 through so_n(8),
    # whose root system fixes the centre) and four products; every order-|Z| subgroup of Z x Z^
    from root_datum import b_n_simply_connected, product_datum
    census_data = [("SU(2)", su_2), ("SU(3)", lambda: su_n(3)), ("SU(4)", lambda: su_n(4)), ("SU(5)", lambda: su_n(5)),
                   ("Sp(4)", lambda: sp_n(2)), ("Sp(6)", lambda: sp_n(3)), ("Sp(8)", lambda: sp_n(4)),
                   ("Spin(5)", lambda: b_n_simply_connected(2)), ("Spin(7)", lambda: b_n_simply_connected(3)),
                   ("Spin(9)", lambda: b_n_simply_connected(4)), ("G2", g_2), ("D4", lambda: so_n(8)),
                   ("SU(2)xSU(2)", lambda: product_datum([su_2(), su_2()])), ("SU(2)xSU(3)", lambda: product_datum([su_2(), su_n(3)])),
                   ("SU(3)xSU(3)", lambda: product_datum([su_n(3), su_n(3)])), ("SU(2)^3", lambda: product_datum([su_2(), su_2(), su_2()]))]
    census, ok_count, ok_products = {}, True, True
    for name, make in census_data:
        divs, n_full, lag = _lagrangian_census(make())
        want = _lagrangian_count_formula(divs) if divs else 1
        n_prod = sum(1 for L in lag if L.is_product())
        n_sub = _centre_subgroup_count(divs) if divs else 1
        census[name] = {"centre": list(divs), "order-|Z| subgroups": n_full, "Lagrangian": len(lag), "formula": want,
                        "product forms": n_prod, "subgroups of Z": n_sub}
        ok_count = ok_count and len(lag) == want and all(L.is_maximal() for L in lag)
        ok_products = ok_products and n_prod == n_sub
    checks.append(check("census: for every simply connected root datum of rank <= 4 and four products, the order-|Z| subgroups of the centre-class pairs that LineLattice reports Dirac-integral (the Lagrangian ones) number exactly the independent count (a cyclic p-part Z_{p^a} gives 1 + p + ... + p^a, (Z_p)^n gives prod (p^i + 1)); the rest are rejected",
                        ok_count, str({k: (v["Lagrangian"], v["formula"], v["order-|Z| subgroups"]) for k, v in census.items()})))
    checks.append(check("census: the conventional (product) forms among the Lagrangian ones are exactly one per subgroup of the centre",
                        ok_products, str({k: (v["product forms"], v["subgroups of Z"]) for k, v in census.items()})))
    return {"checks": checks, "population": {"forms": "SU(2), SO(3), the third form", "window": "|m|, |e| <= 3", "census": census},
            "controls": {"positive": "SU(2) and SO(3) are the conventional product lattices; SU(2)'s three forms in the census",
                         "negative": "the full lattice is not Dirac-integral; the non-isotropic order-|Z| subgroups are rejected"},
            "notes": "LineLattice(datum, classes=...) generates the lattice from centre-class pairs; the Lagrangian subgroups of the classes are the global forms.  The census uses LineLattice's own class labelling (invariant factors: Z6 at SU(2)xSU(3), where centre_invariants lists (2, 3)).", "inputs": []}


# --------------------------------------------------------------------------
# the conjecture on windows: existence/uniqueness, consistency, support, labels
# --------------------------------------------------------------------------
def build_window(A, ms, es, fs=None):
    labs = set()
    lines = getattr(A, "lines", None)
    for m in ms:
        for e in es:
            if lines is not None and not lines.admits(tuple(m), tuple(e)):
                continue                    # not a line of this 4d gauge group: the form's lattice excludes it
            if fs is None:
                labs.add(A.fold(m, e))
            else:
                for f in fs:
                    labs.add(A.fold(m, e) if f is None else A.fold(m, e, f))
    return sorted(labs)


#: the conjecture rows that grow their windows at the extensive depth (recorded beside the fast run)
EXTENSIVE = {"sec:coulomb/labels", "conj:abeKalgebra/existence-uniqueness", "conj:abeKalgebra/consistency",
             "conj:abeKalgebra/support", "eq:Iexplicit", "sec:coulomb/flavour-enhancement"}
#: seconds per label at the extensive depth; a label not built within it is reported as unreached, never as a pass
LABEL_BUDGET = 300


def battery_depth():
    return os.environ.get("BATTERY_DEPTH", "fast")


class _Budget(BaseException):
    """Raised by the per-label alarm.  A BaseException, so that no `except Exception` fallback inside the solvers
    (the twist route's, for one) can swallow it and carry on."""


class budget:
    """At the extensive depth, raise _Budget after `seconds`; at the fast depth, no limit (the fast windows must build)."""

    def __init__(self, seconds):
        self.seconds = seconds if battery_depth() == "extensive" else None

    def __enter__(self):
        if self.seconds:
            import signal

            def _raise(signum, frame):
                raise _Budget()
            self._old = signal.signal(signal.SIGALRM, _raise)
            signal.alarm(int(self.seconds))
        return self

    def __exit__(self, et, ev, tb):
        if self.seconds:
            import signal
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self._old)
        return False


_BUILT: dict = {}


def built_window(A, ms, es, fs):
    """(labels built, labels not reached within the per-label budget, {label: error} for builds that raised).  Each
    chart is built once per process, so the four conjecture rows run in one runner process share it; every row fails
    its build claim on any error, and reports the unreached labels without counting them as passes."""
    key = (id(A), repr(ms), repr(es), repr(fs))
    if key in _BUILT:
        return _BUILT[key]
    W, unreached, errors = [], [], {}
    t0 = time.time()
    for lab in build_window(A, ms, es, fs):
        try:
            with budget(LABEL_BUDGET):
                A.chart(lab)
            W.append(lab)
        except (_Budget, MemoryError):
            unreached.append(lab)      # the time or the memory budget ran out: reported, never counted as a pass or a failure
        except Exception as ex:  # noqa: BLE001
            if _memory_exhausted(ex):
                unreached.append(lab)
            else:
                errors[lab] = f"{type(ex).__name__}: {str(ex)[:120]}"
    _BUILT[key] = (W, unreached, errors)
    if battery_depth() == "extensive":        # progress for a run that records only at the end of each row
        print(f"[extensive] {getattr(A, 'theory', type(A).__name__)} {A.datum.name}: {len(W)} labels built, "
              f"{len(unreached)} unreached, {len(errors)} errors ({time.time() - t0:.0f} s)", file=sys.stderr, flush=True)
    return W, unreached, errors


#: gigabytes of address space a theory's battery may add at the extensive depth (runners share one container: a
#: consistency battery at G2 once grew to 11 GB and the system killed the whole row, 2026-09-23)
THEORY_MEMORY_GB = 5


def _set_memory_limit(extra_gb):
    """In a forked child: cap the address space at its current size plus `extra_gb`, so an outgrown battery raises
    MemoryError in the child instead of drawing the system's out-of-memory killer onto some process; and ask to be
    killed when the runner dies (Linux PR_SET_PDEATHSIG), so a killed runner leaves no orphaned worker behind.

    Both are best effort: where /proc or prctl are absent (macOS, the local machine) or the limit cannot be set (a
    lower hard `ulimit -v`), the child runs without them rather than dying at start — a worker that died there would
    be reported 'not reached' for every theory and the row would compare nothing.  The cap bounds new allocations
    only: pages inherited from the runner count at fork, and copy-on-write can grow them uncounted — so run the
    per-theory rows alone (`--id`): a runner that has first run an in-process extensive row (labels,
    existence-uniqueness, support build every chart in the runner) hands all of those pages to every fork."""
    try:
        import ctypes
        import signal
        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGKILL)       # PR_SET_PDEATHSIG = 1
    except Exception:  # noqa: BLE001
        pass
    try:
        import resource
        vm = 0
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmSize:"):
                    vm = int(line.split()[1]) * 1024
        if vm:
            lim = vm + int(extra_gb * 2 ** 30)
            resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
    except Exception as ex:  # noqa: BLE001
        print(f"[extensive] memory budget not set in this worker ({type(ex).__name__}: {ex})", file=sys.stderr, flush=True)


#: modules the solvers import lazily; the runner imports them before its first fork, so every worker runs the code
#: of the launch commit rather than whatever the working tree holds when the worker reaches its first matter chart
_LAZY_MODULES = ("matter_multislot", "matter_star_bubbling", "sun_characters", "habiro", "star_bubbling",
                 "matter_wrq_torus", "wrq_torus", "weyl_torus_ring", "global_form", "zplus_ring", "kalgebra",
                 "root_datum", "laurent_poly", "g_matter_roster")
_PRELOADED = []


def _preload_modules():
    if _PRELOADED:
        return
    import importlib
    for mod in _LAZY_MODULES:
        try:
            importlib.import_module(mod)
        except Exception:  # noqa: BLE001
            pass
    _PRELOADED.append(True)


def _checkpoint_file(fn, environment, name):
    """Where a theory's extensive result is kept between runs, or None: only when BATTERY_CHECKPOINT_DIR is set and
    the run was launched from a clean commit (BATTERY_LAUNCH_COMMIT, set by the runner) — a checkpoint is valid for
    exactly the code that produced it, so changed code starts afresh."""
    d, commit = os.environ.get("BATTERY_CHECKPOINT_DIR"), os.environ.get("BATTERY_LAUNCH_COMMIT", "")
    if not d or not commit or commit.endswith("+dirty") or commit == "unknown":
        return None
    import hashlib
    key = hashlib.sha1(f"{fn.__name__}|{environment}|{name}".encode()).hexdigest()[:16]
    return os.path.join(d, commit, fn.__name__, f"{key}.json")


def _in_child(fn, environment, name):
    """Run `fn` in the per-theory worker; the memory budget running out is reported as a MemoryError whatever form it
    took (the classification needs the worker's own address-space cap, so it happens here, before pickling)."""
    try:
        return fn(environment, name)
    except MemoryError:
        raise
    except Exception as ex:  # noqa: BLE001
        if _memory_exhausted(ex):
            raise MemoryError(f"{type(ex).__name__} at the memory budget") from None
        raise


def _per_theory(fn, environment, name):
    """fn(environment, name) -> (checks, population entry).  At the extensive depth it runs in a forked child with a
    memory budget: the child's memory is released when it ends, and a child that exceeds the budget or dies yields no
    checks and a population entry saying so (never a pass).  With a checkpoint directory each finished theory is kept,
    so an interrupted row resumes where it stopped (two rows of hours were lost whole on 2026-09-23: a memory kill and a
    session interruption); a theory whose worker died is not kept.  At the fast depth it runs in-process."""
    if battery_depth() != "extensive":
        return fn(environment, name)
    import json
    ck = _checkpoint_file(fn, environment, name)
    if ck and os.path.exists(ck):
        with open(ck, encoding="utf-8") as fh:
            saved = json.load(fh)
        print(f"[extensive] {name}: resumed from checkpoint", file=sys.stderr, flush=True)
        return saved["checks"], saved["entry"]
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool
    _preload_modules()
    t1 = time.time()
    why, keep, out = None, False, None
    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("fork"),
                             initializer=_set_memory_limit, initargs=(THEORY_MEMORY_GB,)) as ex:
        try:
            out, keep = ex.submit(_in_child, fn, environment, name).result(), True
        except MemoryError:
            why, keep = f"not reached: memory budget ({THEORY_MEMORY_GB} GB) exceeded", True
        except BrokenProcessPool:
            why = "not reached: the worker process died"
        except Exception as err:  # noqa: BLE001
            # an error the battery did not catch: a failure of this theory's run, reported (not a crash of the row)
            out, keep = ([check(f"{name}: the battery ran", False, f"{type(err).__name__}: {str(err)[:200]}")],
                         dict(status=f"error: {type(err).__name__}", seconds=round(time.time() - t1, 1))), True
        except BaseException:
            # an interrupt of the runner: end the in-flight theory now rather than let shutdown wait for it
            for proc in list(getattr(ex, "_processes", {}).values()):
                proc.kill()
            raise
    if out is None:
        print(f"[extensive] {name}: {why} ({time.time() - t1:.0f} s)", file=sys.stderr, flush=True)
        out = ([], dict(status=why, seconds=round(time.time() - t1, 1)))
    if ck and keep:
        os.makedirs(os.path.dirname(ck), exist_ok=True)
        with open(ck + ".tmp", "w", encoding="utf-8") as fh:
            json.dump({"name": name, "checks": out[0], "entry": out[1]}, fh, ensure_ascii=False, default=str)
        os.replace(ck + ".tmp", ck)
    return out


def run_k_verifiers(A, labels, K):
    from checks_kq import VERIFIERS
    out = {}
    for _ax, name, arity, takesK in VERIFIERS:
        fn = getattr(A, name)
        if arity == 0:
            it = [()]
        elif arity == 1:
            it = [(a,) for a in labels]
        elif arity == 2:
            it = itertools.product(labels, labels)
        else:
            it = itertools.product(labels[:4], labels[:4], labels[:4])
        f = 0
        first = None
        for args in it:
            ok = fn(*args, K=K) if takesK else fn(*args)
            if not ok:
                f += 1
                first = first or args
        out[name] = (f, first)
    return out


def chart_support(x):
    res = x.residuals()
    out = set()
    for m, r in res.items():
        if isinstance(r, dict):
            if any(not v.is_zero() for v in r.values()):
                out.add(m)
        elif not r.is_zero():
            out.add(m)
    return out


_WINDOWS: dict = {}


def conjecture_windows(environment):
    """The window algebras shared by the conjecture rows (labels, existence/uniqueness, consistency,
    support): name -> (algebra, magnetic charges, electric charges, flavour labels or None in pure gauge).
    Memoized per (environment, depth), so the rows run in one process share the algebras and their charts."""
    key = (environment, battery_depth())
    if key not in _WINDOWS:
        _WINDOWS[key] = _extensive_windows() if key[1] == "extensive" else _fast_windows(environment)
    return _WINDOWS[key]


def _extensive_windows():
    """The extensive depth: the theories (the Section 3 examples) on windows grown past the fast ones.
    Matter theories carry the neutral flavour label (None)."""
    from fractions import Fraction as Fr
    from root_datum import b_n_simply_connected, product_datum, u_n
    from global_form import adjoint_lines, LineLattice
    from g_matter_roster import highest_root
    H = Fr(1, 2)
    su2, su3, b2, g2 = su_2(), su_n(3), b_n_simply_connected(2), g_2()
    P2 = product_datum([su2, su2])
    r1 = lambda M, E: ([(m,) for m in range(0, M + 1)], [(e,) for e in range(-E, E + 1)])
    dom2 = [(0, 0), (1, 1), (2, 1), (1, 2), (2, 2)]
    e2 = [(0, 0), (1, 0), (0, 1), (-1, 1), (1, -1), (2, 0)]
    N = [None]
    third_form = LineLattice(su2, classes=[(1, 1)])
    return {
        "pure SU(2)": (PureGAbeKAlgebra(su2), *r1(4, 3), None),
        "pure SO(3)": (PureGAbeKAlgebra(so_n(3)), *r1(4, 3), None),
        "pure rank 1, the third global form": (PureGAbeKAlgebra(su2, lines=third_form), [(0,), (H,), (1,), (3 * H,), (2,)], [(-2,), (-1,), (0,), (1,), (2,)], None),
        "pure SU(3)": (PureGAbeKAlgebra(su3), [(0, 0), (1, 1), (2, 2)], e2, None),
        "pure PSU(3)": (PureGAbeKAlgebra(su3, lines=adjoint_lines(su3)), [(0, 0), (Fr(2, 3), Fr(1, 3)), (Fr(1, 3), Fr(2, 3)), (1, 1)], [(0, 0), (1, 1), (3, 0), (0, 3)], None),
        "pure Sp(4)": (PureGAbeKAlgebra(sp_n(2)), dom2, e2, None),
        "pure Spin(5)": (PureGAbeKAlgebra(b2), dom2, e2, None),
        "pure SO(5)": (PureGAbeKAlgebra(b2, lines=adjoint_lines(b2)), [(0, 0), (1, H), (1, 1), (2, 1)], [(0, 0), (0, 2), (1, 0), (2, 0)], None),
        "pure G2": (PureGAbeKAlgebra(g2), [(0, 0), (1, 2), (2, 3), (2, 4)], [(0, 0), (1, 0), (0, 1)], None),
        "SU(2)+1": (GNAbeKAlgebra(su2, (1,), nf=1), *r1(3, 2), N),
        "SU(2)+2": (GNAbeKAlgebra(su2, (1,), nf=2), *r1(2, 2), N),
        "SU(2)+3": (GNAbeKAlgebra(su2, (1,), nf=3), *r1(2, 1), N),
        "SU(2)+4": (GNAbeKAlgebra(su2, (1,), nf=4), *r1(2, 1), N),
        "SU(2)+adjoint (N=2*)": (GNAbeKAlgebra(su2, highest_root(su2), nf=1), *r1(2, 2), N),
        "SO(3)+adjoint (N=2*)": (GNAbeKAlgebra(su2, highest_root(su2), nf=1, lines=adjoint_lines(su2)), [(0,), (H,), (1,), (3 * H,)], [(-2,), (0,), (2,)], N),
        "rank 1 + adjoint (N=2*), the third global form": (GNAbeKAlgebra(su2, highest_root(su2), nf=1, lines=third_form), [(0,), (H,), (1,)], [(-1,), (0,), (1,), (2,)], N),
        "SU(3)+adjoint (N=2*)": (GNAbeKAlgebra(su3, highest_root(su3), nf=1), [(0, 0), (1, 1)], [(0, 0), (1, 0), (1, 1)], N),
        "SU(3)+symmetric": (GNAbeKAlgebra(su3, (2, 0), nf=1), [(0, 0), (1, 1)], e2, N),
        "SU(3)+symmetric+1": (GNAbeKAlgebra(su3, [(2, 0), (1, 0)]), [(0, 0), (1, 1)], [(0, 0), (1, 0), (0, 1)], N),
        "Spin(5)+vector": (GNAbeKAlgebra(b2, (1, 0), nf=1), [(0, 0), (1, 1), (2, 1)], [(0, 0), (1, 0), (0, 1)], N),
        "SO(5)+vector": (GNAbeKAlgebra(b2, (1, 0), nf=1, lines=adjoint_lines(b2)), [(0, 0), (1, H), (1, 1), (2, 1)], [(0, 0), (0, 2), (1, 0)], N),
        "G2+7": (GNAbeKAlgebra(g2, (1, 0), nf=1), [(0, 0), (1, 2), (2, 3)], [(0, 0), (1, 0)], N),
        "SU(2)xSU(2)+bifundamental": (GNAbeKAlgebra(P2, (1, 1), nf=1), [(0, 0), (1, 0), (0, 1), (1, 1)], [(0, 0), (1, 0), (0, 1), (1, 1)], N),
        "U(1)-U(2)+3": (GNAbeKAlgebra(product_datum([u_n(1), u_n(2)]), [(1, 0, -1), ((0, 1, 0), 3)]),
                        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 1, 1), (1, 1, 0), (-1, 0, 0)], [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 1, 1)], N),
    }


def _fast_windows(environment):
    from g_matter_roster import roster
    # matter windows: flux m <= 1 on the web tier (the K1-K5 pair battery at K = 4 costs ~40 s per 25 pairs and scales with the pairs;
    # measured 2026-09-22), m <= 2 on the local environment; the pure rank-1 windows keep m <= 2 everywhere
    mm = 2 if environment == "local" else 1
    return {
        "pure SU(2)": (PureGAbeKAlgebra(su_2()), [(m,) for m in range(0, 3)], [(e,) for e in range(-2, 3)], None),
        "pure SO(3)": (PureGAbeKAlgebra(so_n(3)), [(m,) for m in range(0, 3)], [(e,) for e in range(-2, 3)], None),
        "pure SU(3)": (PureGAbeKAlgebra(su_n(3)), [(0, 0), (1, 0), (0, 1)], [(0, 0), (1, 0), (0, 1)], None),
        "pure Sp(4)": (PureGAbeKAlgebra(sp_n(2)), [(0, 0), (1, 0)], [(0, 0), (1, 0), (0, 1)], None),
        "pure G2": (PureGAbeKAlgebra(g_2()), [(0, 0), (1, 0)], [(0, 0), (1, 0)], None),
        "SU(2) + 1 fund. (roster su2-nf1)": (roster("su2-nf1"), [(m,) for m in range(0, mm + 1)], [(e,) for e in range(-1, 2)], [(0,)]),
        "SU(2) + 2 fund. (roster su2-nf2, U(2) flavour)": (roster("su2-nf2"), [(m,) for m in range(0, mm + 1)], [(e,) for e in range(-1, 2)], [(0, 0)]),
        "SO(3) + adjoint (N=2*)": (GNAbeKAlgebra(so_n(3), (1,), nf=1), [(m,) for m in range(0, mm + 1)], [(e,) for e in range(-1, 2)], [(0,)]),
        "SU(3) + 1 fund. (roster su3-nf1)": (roster("su3-nf1"), [(0, 0), (1, 0)], [(0, 0), (1, 0)], [(0,)]),
    }


def _consistency_one(environment, name):
    """One theory of conj:abeKalgebra/consistency: (its checks, its population entry).  At the extensive depth it
    runs in a forked child (`_per_theory`), so a battery that outgrows its memory budget costs that theory only."""
    A, ms, es, fs = conjecture_windows(environment)[name]
    checks = []
    deep = battery_depth() == "extensive"
    t1 = time.time()
    W, unreached, errors = built_window(A, ms, es, fs)
    unreached = list(unreached)
    built, routes, certs, stars = 0, {}, 0, 0
    fail_detail = ""
    short = []                          # labels whose chart, certification or (★) check outgrew the memory budget
    for lab in W:
        try:
            x = A.chart(lab)
            r = A.route(lab)
            c_ok = bool(_cert_ok(A, lab))
            if fs is None:
                ok_star, rep = criterion(x)          # the pure residue rule; with matter the class's certification carries (★)
            else:
                ok_star, rep = True, "certified (matter)"
        except MemoryError:
            short.append(lab)          # a resource limit: reported not reached, never a failure (the others are still judged)
            continue
        except Exception as ex:  # noqa: BLE001
            if _memory_exhausted(ex):
                short.append(lab)
                continue
            if not fail_detail:
                fail_detail = f"{lab}: {type(ex).__name__}: {ex}"
            continue                   # not counted as built: the build check fails
        built += 1
        routes[r] = routes.get(r, 0) + 1
        certs += c_ok
        stars += bool(ok_star)
        if not ok_star and not fail_detail:
            fail_detail = f"{lab}: {rep}"
    if short:
        W = [lab for lab in W if lab not in short]
        unreached += short
    rank1 = bool(W) and len(W[0][0][0] if fs is not None else W[0][0]) == 1
    legit = all(r == "star" or r == "solve" or r.startswith("twist") or r.startswith("monoid") for r in routes)   # the axiom route or a declared optimization (asserted equal to it); never a retired constructive route
    unr = f" ({len(unreached)} more not reached within the {LABEL_BUDGET} s or the memory budget)" if unreached else ""
    checks.append(check(f"{name}: all {len(W)} labels of the window{unr} build by the axiom route or a declared optimization asserted equal to it, pass certify_canonical and the residue rule (★)", built == len(W) and certs == len(W) and stars == len(W) and legit and not errors, f"routes {routes}; {fail_detail}; errors {errors}" if errors else f"routes {routes}; {fail_detail}"))
    heavy_pairs = {"SU(2) + 2 fund. (roster su2-nf2, U(2) flavour)"}   # its K1-K5 pair battery at K = 4 took 2364 s (rank-2 flavour ring): local environment only
    if deep:
        # the extensive depth: the K1-K5 pair battery and the closure on the window's first eight labels (the
        # smallest charges), at every rank, under a budget; a battery not finished in it is reported, not passed
        Wk = W[:8]
        KV = 4
        try:
            with budget(8 * LABEL_BUDGET):
                ver = run_k_verifiers(A, Wk, KV)
                n_fail = sum(f for f, _x in ver.values())
                closure, n_prod, negc = True, 0, 0
                for a in Wk:
                    for b in Wk:
                        el = A.multiply(a, b)
                        closure &= all(isinstance(cf, LaurentPoly) for cf in el.terms.values())
                        negc += any(v < 0 for cf in el.terms.values() for v in getattr(cf, "_coeffs", {}).values())
                        n_prod += 1
            checks.append(check(f"{name}: K1-K5 verifiers on all pairs of the window's first {len(Wk)} labels (K = {KV}) and closure of {n_prod} products with Z[q^±1] coefficients ({negc} products with a negative coefficient)", n_fail == 0 and closure, "; ".join(f"{k} fails {f} (first {x})" for k, (f, x) in ver.items() if f)))
        except _Budget:
            # not finished within the budget: reported in the population, never as a check (a pass it is not)
            n_fail, closure = f"not reached within {8 * LABEL_BUDGET} s", f"not reached within {8 * LABEL_BUDGET} s"
        except MemoryError:
            # outgrew the per-theory memory budget (_per_theory): the build checks above stand, the pair battery does not
            n_fail, closure = f"not reached within {THEORY_MEMORY_GB} GB", f"not reached within {THEORY_MEMORY_GB} GB"
        except Exception as ex:  # noqa: BLE001
            if _memory_exhausted(ex):
                n_fail, closure = f"not reached within {THEORY_MEMORY_GB} GB", f"not reached within {THEORY_MEMORY_GB} GB"
            else:
                checks.append(check(f"{name}: the K1-K5 pair battery ran", False, f"{type(ex).__name__}: {str(ex)[:200]}"))
                n_fail, closure = f"error: {type(ex).__name__}", f"error: {type(ex).__name__}"
    elif rank1 and (environment == "local" or name not in heavy_pairs):
        # products of window labels reach labels outside the window (m up to 2m); at rank 2 those charts are the
        # extensive tier's cost, so the K1-K5 pair checks and the closure run here at rank 1 only (web tier)
        KV = 4
        ver = run_k_verifiers(A, W, KV)
        n_fail = sum(f for f, _x in ver.values())
        closure = True
        n_prod = 0
        for a in W:
            for b in W:
                ma = a[0][0][0] if fs is not None else a[0][0]
                mb = b[0][0][0] if fs is not None else b[0][0]
                if fs is not None and ma + mb > 2:
                    continue          # with matter the deep product charts (m up to 4) are the extensive tier's cost
                try:
                    el = A.multiply(a, b)
                    closure &= all(isinstance(cf, LaurentPoly) for cf in el.terms.values())
                    n_prod += 1
                except Exception:  # noqa: BLE001
                    closure = False
        checks.append(check(f"{name}: K1-K5 verifiers on all pairs of the window (K = {KV}) and closure of {n_prod} products with Z[q^±1] coefficients", n_fail == 0 and closure, "; ".join(f"{k} fails {f} (first {x})" for k, (f, x) in ver.items() if f)))
    else:
        n_fail, closure = None, None
    return checks, dict(labels=len(W), unreached=[str(l) for l in unreached], errors={str(k): v for k, v in errors.items()}, built=built, routes=routes, certified=certs, star=stars, verifier_failures=n_fail, closure=closure, seconds=round(time.time() - t1, 1))


def check_conjecture_windows(environment):
    checks = []
    t0 = time.time()
    # the fast (web) windows are deliberately small; the extensive rows grow them (README section 3)
    summary = {}
    for name in conjecture_windows(environment):
        c, s_ = _per_theory(_consistency_one, environment, name)
        checks += c
        summary[name] = s_
    deep = battery_depth() == "extensive"
    return {"checks": checks, "population": summary, "controls": {"positive": "every label of every window builds", "negative": "n.a. here: the negative controls of the conjecture are the rows conj:abeKalgebra/existence-uniqueness (perturbed lines) and conj:abeKalgebra/support (grafts)"},
            "notes": (f"Extensive population: each theory in its own process with a {THEORY_MEMORY_GB} GB memory budget" if deep else "Fast-window version of the extensive rows") + f"; total {round(time.time() - t0, 1)} s.  Per algebra: " + "; ".join(f"{k}: {v}" for k, v in summary.items()), "inputs": []}


# --------------------------------------------------------------------------
# the labelling, existence/uniqueness and the support bound (own adapters since 2026-09-23;
# before that the three rows shared the consistency run and two declared controls were not run)
# --------------------------------------------------------------------------
def _ev_mono(pt, mono):
    """v^mono at the rational point pt (integral exponents only)."""
    out = Fraction(1)
    for x, k in zip(pt, mono):
        k = Fraction(k)
        if k.denominator != 1:
            raise ValueError(f"non-integral exponent in {mono}")
        out *= Fraction(x) ** int(k)
    return out


def _ev_tr(f, q0, pt):
    """A TorusRational num/Π(1 - q^k v^α)^mult at q = q0, v = pt, exactly."""
    num = Fraction(0)
    for mono, lp in f.num.terms.items():
        num += sum((Fraction(c) * q0 ** e for e, c in lp._coeffs.items()), Fraction(0)) * _ev_mono(pt, mono)
    den = Fraction(1)
    for (alpha, k), mult in f.den.items():
        den *= (1 - q0 ** k * _ev_mono(pt, alpha)) ** mult
    if den == 0:
        raise ZeroDivisionError("evaluation point on a wall")
    return num / den


def _ev_cell(r, q0, pt, mu):
    if isinstance(r, dict):          # with matter: {μ-exponent vector: TorusRational}
        return sum((_ev_mono(mu, k) * _ev_tr(f, q0, pt) for k, f in r.items()), Fraction(0))
    return _ev_tr(r, q0, pt)


def _exact_rank(rows):
    rows = [list(r) for r in rows]
    rank = 0
    ncol = len(rows[0]) if rows else 0
    for c in range(ncol):
        piv = next((i for i in range(rank, len(rows)) if rows[i][c] != 0), None)
        if piv is None:
            continue
        rows[rank], rows[piv] = rows[piv], rows[rank]
        for i in range(len(rows)):
            if i != rank and rows[i][c] != 0:
                f = rows[i][c] / rows[rank][c]
                rows[i] = [a - f * b for a, b in zip(rows[i], rows[rank])]
        rank += 1
    return rank


def _evaluation_rows(elements, dim):
    """Each element evaluated at s rational points v^(j) (q = 3/11, generic rational flavour fugacities),
    one column per (cell, j).  A nonzero maximal minor of this matrix is a nonzero rational function of
    (q, μ) specialised, so full row rank here certifies linear independence over Q(q, μ), hence over the
    coefficient ring (the entries are the specialisations of coefficient-ring-linear functionals)."""
    q0 = Fraction(3, 11)
    cells = sorted(set().union(*(set(x.residuals()) for x in elements)), key=repr)
    nslots = 0
    for x in elements:
        for r in x.residuals().values():
            if isinstance(r, dict) and r:
                nslots = len(next(iter(r)))
                break
        if nslots:
            break
    mu = tuple(Fraction(7 + t, 17 + 3 * t) for t in range(nslots))
    # s points per cell, at least as many as there are elements: the lines sharing one cell (e.g. every Wilson line
    # lives on the cell 0 alone) must be separable by that cell's columns, else the rank is capped by the harness
    # (measured 2026-09-23: s from the average of lines per cell gave rank 15 of 16 at pure SU(3), four Wilson lines
    # on three points)
    s = max(2, len(elements))
    for shift in range(5):           # move off a wall if a point happens to lie on one
        pts = [tuple(Fraction(2 + 3 * j + i + shift, 5 + 2 * j + 3 * i + 2 * shift) for i in range(dim)) for j in range(s)]
        try:
            return [[_ev_cell(x.residuals()[c], q0, pt, mu) if c in x.residuals() else Fraction(0)
                     for c in cells for pt in pts] for x in elements]
        except ZeroDivisionError:
            continue
    raise ZeroDivisionError("no evaluation point off the walls found")


def _label_parts(lab, matter):
    return lab[0] if matter else lab


def check_labels(environment):
    """sec:coulomb/labels: every window label lies in the fundamental domain (m dominant, e dominant for the
    Levi of m); every Weyl image of every raw (m, e) folds to the same label; distinct labels give linearly
    independent elements (an exact rank certificate at a rational point)."""
    import star_bubbling as SB
    checks, summary = [], {}
    t0 = time.time()
    for name, (A, ms, es, fs) in conjecture_windows(environment).items():
        t1 = time.time()
        D = A.datum
        W, unreached, errors = built_window(A, ms, es, fs)
        # (a) fundamental domain
        dom_bad = []
        for lab in W:
            m, e = _label_parts(lab, fs is not None)
            levi = [SB.coroot_of(D, tuple(a)) for a in D.positive_roots() if D.shift_pairing(tuple(m), tuple(a)) == 0]
            if not D.is_dominant_cochar(tuple(m)) or any(c is None or D.shift_pairing(tuple(c), tuple(e)) < 0 for c in levi):
                dom_bad.append(lab)
        # (b) Weyl invariance of the labelling, with its control: some raw pair of the window has a nontrivial
        # Weyl orbit, so the identity map (no quotient) would fail this test
        n_images, w_bad, nontrivial = 0, [], 0
        for m in ms:
            for e in es:
                for f in (fs or [None]):
                    ref = A.fold(m, e) if f is None else A.fold(m, e, f)
                    orbit = set()
                    for w in D.weyl_elements():
                        mw, ew = tuple(D.act_cochar(w, tuple(m))), tuple(D.act(w, tuple(e)))
                        orbit.add((mw, ew))
                        got = A.fold(mw, ew) if f is None else A.fold(mw, ew, f)
                        n_images += 1
                        if got != ref:
                            w_bad.append(((m, e), got, ref))
                    nontrivial += len(orbit) > 1
        # (c) linear independence: exact rank of the evaluated elements, with an end-to-end control (a genuine
        # relation x0 + 2 x1 among the torus elements must not raise the rank)
        xs = [A.chart(lab) for lab in W]
        rows = _evaluation_rows(xs, D.dim) if xs else []
        rank = _exact_rank(rows) if rows else 0
        ctrl = None
        if len(xs) >= 2:
            z = xs[0] + xs[1] + xs[1]
            rows_z = _evaluation_rows(xs + [z], D.dim)
            ctrl = _exact_rank(rows_z) == rank
        checks.append(check(f"{name}: all {len(W)} window labels lie in the fundamental domain (m dominant, e dominant for the Levi of m)", not dom_bad, f"outside: {dom_bad[:3]}"))
        checks.append(check(f"{name}: all {n_images} Weyl images of the window's raw (m, e) fold to the same label ({nontrivial} raw pairs have a nontrivial orbit, so the test has teeth)", not w_bad and nontrivial > 0, f"{w_bad[:2]}"))
        unr = f" ({len(unreached)} more not reached within the {LABEL_BUDGET} s budget)" if unreached else ""
        checks.append(check(f"{name}: the {len(W)} built labels{unr} give linearly independent elements (exact rank {rank} at a rational point), and the control relation x0 + 2 x1 is detected", rank == len(W) and ctrl is not False and not errors, f"rank {rank} of {len(W)}; control {ctrl}" + (f"; build errors {errors}" if errors else "")))
        summary[name] = dict(labels=len(W), unreached=[str(l) for l in unreached], errors={str(k): v for k, v in errors.items()}, weyl_images=n_images, nontrivial_orbits=nontrivial, rank=rank, seconds=round(time.time() - t1, 1))
    return {"checks": checks, "population": summary,
            "controls": {"positive": "the window labels themselves", "negative": "raw pairs with nontrivial Weyl orbits (the identity labelling would fail); the relation x0 + 2 x1 appended to the rank test does not raise the rank"},
            "notes": f"Rank certificate: residuals evaluated exactly at q = 3/11, v at s rational points, flavour fugacities at generic rationals; a nonzero maximal minor at a point is a nonzero rational function, so full rank there is full rank over Q(q, mu).  Total {round(time.time() - t0, 1)} s.", "inputs": []}


def _zero_cell(x):
    return all(f.simplify().is_zero() for f in (x.values() if isinstance(x, dict) else [x]))


def _same_residuals(a, b):
    """Cellwise equality of two residual dicts (pure: TorusRational; matter: {μ-exponents: TorusRational})."""
    for n in set(a) | set(b):
        fa, fb = a.get(n), b.get(n)
        if fa is None or fb is None:
            if not _zero_cell(fa if fb is None else fb):
                return False
            continue
        if isinstance(fa, dict):
            for k in set(fa) | set(fb):
                ga, gb = fa.get(k), fb.get(k)
                if ga is None or gb is None:
                    if not (ga if gb is None else gb).simplify().is_zero():
                        return False
                elif not (ga + (-gb)).simplify().is_zero():
                    return False
        elif not (fa + (-fb)).simplify().is_zero():
            return False
    return True


def check_existence_uniqueness(environment):
    """conj:abeKalgebra/existence-uniqueness: the guarded solvers called directly on every window label (pure:
    joint_solve 'unique' + solve_canonical with W1, (★) and the box certificate; matter:
    solve_canonical_matter_vec, every μ-sector unique + strict divisibility), each result equal to the
    production chart; positive control the Wilson lines (no unknowns); negative controls: perturbed lines
    that keep W1, W2, O(q) and the support bound fail (★)."""
    import star_bubbling as SB
    from matter_multislot import solve_canonical_matter_vec, divide_by_Z_vec
    from wrq_torus import WRQTorus as _W
    checks, summary = [], {}
    t0 = time.time()
    for name, (A, ms, es, fs) in conjecture_windows(environment).items():
        t1 = time.time()
        D = A.datum
        W, unreached, errors = built_window(A, ms, es, fs)
        solver_unreached = []
        n_ok, n_eq, fails, unknowns = 0, 0, [], {}
        wilson_ok, n_wilson = True, 0
        zero = (0,) * D.dim
        for lab in W:
            m, e = (tuple(t) for t in _label_parts(lab, fs is not None))
            try:
              with budget(2 * LABEL_BUDGET):
                if fs is None:
                    st, _F, _sup, interior, _K = SB.joint_solve(D, m, e, pad=1)
                    x = SB.solve_canonical(D, m, e, pad=1, certify_box=True)   # raises StarGuardFailure unless unique, W1, (★) and the box certificate hold
                    unknowns[lab] = len(interior)
                    same = _same_residuals(x.residuals(), A.chart(lab).residuals())
                    ok = st == "unique"
                else:
                    F = solve_canonical_matter_vec(D, A.matter, m, e, pad=1, lines=A.lines)   # raises unless every μ-sector is unique and divisible
                    percell = {}
                    for k, lvl in F.items():
                        for c, v in lvl.items():
                            percell.setdefault(tuple(c), {})[tuple(k)] = v
                    Q = {}
                    for c, coeffs in percell.items():
                        q, okd = divide_by_Z_vec(D, A.matter, c, coeffs)
                        if not okd:
                            raise ValueError(f"cell {c} not divisible by Z")
                        Q[c] = {tuple(k): v.simplify() for k, v in q.items() if not v.simplify().is_zero()}
                    same = _same_residuals(Q, A.chart(lab).residuals())
                    ok = True
                n_ok += ok
                n_eq += same
                if not same and len(fails) < 3:
                    fails.append(f"{lab}: solver output differs from the chart")
            except _Budget:
                solver_unreached.append(lab)
            except Exception as ex:  # noqa: BLE001
                if len(fails) < 3:
                    fails.append(f"{lab}: {type(ex).__name__}: {str(ex)[:120]}")
            if m == zero:
                n_wilson += 1
                supp = chart_support(A.chart(lab))
                wilson_ok &= supp == {zero} and (fs is not None or unknowns.get(lab) == 0)
        n_try = len(W) - len(solver_unreached)
        unr = (f" ({len(unreached)} labels not built and {len(solver_unreached)} solves not finished within the budget)" if (unreached or solver_unreached) else "")
        checks.append(check(f"{name}: the guarded solver, called directly, pins all {n_try} window labels uniquely and returns the production chart on each{unr}", n_ok == n_try and n_eq == n_try and not errors, "; ".join(fails) + (f"; build errors {errors}" if errors else "")))
        checks.append(check(f"{name}: positive control — the {n_wilson} Wilson lines have a single support cell" + (" and no unknowns" if fs is None else ""), wilson_ok and n_wilson > 0))
        neg = None
        if fs is None:
            # negative controls: remove or double the bubbling of every line that has some
            n_pert, n_rej, n_clean, bad = 0, 0, 0, []
            for lab in W:
                m, e = lab
                x = A.chart(lab)
                orbit = {tuple(D.act_cochar(w, tuple(m))) for w in D.weyl_elements()}
                bub = {n: f for n, f in x.residuals().items() if tuple(n) not in orbit and not f.simplify().is_zero()}
                if not bub:
                    continue
                for factor in (0, 2):
                    res = {}
                    for n, f in x.residuals().items():
                        if n in bub:
                            if factor:
                                res[n] = f + f
                        else:
                            res[n] = f
                    z = _W(D, res)
                    ax = SB.verify_axioms(D, tuple(m), tuple(e), z)
                    others = all(ax[k][0] for k in ("bar (W1)", "seed (W2)", "support", "O(𝖖)"))
                    n_pert += 1
                    n_clean += others
                    n_rej += not ax["(★)"][0]
                    if (not others or ax["(★)"][0]) and len(bad) < 3:
                        bad.append(f"{lab} x{factor}: " + ", ".join(f"{k} {v[0]}" for k, v in ax.items()))
            neg = dict(perturbed=n_pert, other_conditions_hold=n_clean, star_rejects=n_rej)
            checks.append(check(f"{name}: negative control — {n_pert} perturbed lines (bubbling removed or doubled) keep bar, the seed, O(q) and the support bound, and the residue rule (★) rejects every one", n_pert > 0 and n_clean == n_pert and n_rej == n_pert, "; ".join(bad)))
        # the draft's A3 as printed, on every window line: d_a = f_a(q^a v) cocha_a(v, mu), the printed matter factors
        # of eq:cocha included, has simple poles only and its pair residues cancel, power by power in mu
        from matter_wrq_torus import slot_weights as _sw
        slots_w = _sw(D, A.matter) if fs is not None else ()
        a3_simple = a3_res = 0
        for lab in W:
            rs = [paper_A3(D, Dd) for Dd in paper_d_levels(A.chart(lab), D, slots_w).values()]
            a3_simple += all(r[0] for r in rs)
            a3_res += all(r[1] for r in rs)
        checks.append(check(f"{name}: the draft's A3 as printed holds on all {len(W)} window lines: every d_a = f_a(q^a v) cocha_a(v{', mu' if fs is not None else ''}) has simple poles only (by construction on solver-built lines: the solver admits only simple poles) and its pair residues cancel with the printed partner" + (" (the printed matter factors of eq:cocha, power by power in mu)" if fs is not None else ""), a3_simple == len(W) and a3_res == len(W), f"simple poles {a3_simple}, residues {a3_res} of {len(W)}"))
        summary[name] = dict(labels=len(W), unreached=[str(l) for l in unreached], solves_unreached=[str(l) for l in solver_unreached], errors={str(k): v for k, v in errors.items()}, pinned=n_ok, equal_to_chart=n_eq, a3_as_printed=min(a3_simple, a3_res), unknown_cells=dict(sorted(((str(k), v) for k, v in unknowns.items()))) if unknowns else None, wilson=n_wilson, negative=neg, seconds=round(time.time() - t1, 1))
    # A3 with the printed matter factors on further complex / non-simply-laced matter, and its negative controls: the
    # matter factor over the weights of N^vee, and no matter factor (for self-conjugate matter N and N^vee coincide,
    # so only complex representations can tell the draft's convention w in N from its dual)
    from root_datum import b_n_simply_connected
    from matter_wrq_torus import slot_weights as _sw
    extra = [("SU(3)+1", GNAbeKAlgebra(su_n(3), (1, 0), nf=1), [((1, 1), (0, 0)), ((1, 1), (1, 0))]),
             ("SU(3)+symmetric", GNAbeKAlgebra(su_n(3), (2, 0), nf=1), [((1, 1), (0, 0)), ((1, 1), (1, 0))]),
             ("Spin(5)+vector", GNAbeKAlgebra(b_n_simply_connected(2), (1, 0), nf=1), [((1, 1), (0, 0)), ((2, 1), (0, 0))])]
    ctrl = {}
    for name, A, labs in extra:
        D = A.datum
        slots_w = _sw(D, A.matter)
        out = {}
        for mode in ("printed", "dual", "none"):
            good = 0
            for m, e in labs:
                rs = [paper_A3(D, Dd) for Dd in paper_d_levels(A.chart(A.fold(m, e)), D, slots_w, matter_factor=mode).values()]
                good += all(r[0] and r[1] for r in rs)
            out[mode] = good
        ctrl[name] = out
    checks.append(check("the draft's A3 as printed on " + ", ".join(f"{k} ({len(l)} lines)" for k, _a, l in extra) + ": every line passes with the printed matter factors", all(ctrl[k]["printed"] == len(l) for k, _a, l in extra), str(ctrl)))
    checks.append(check("negative controls: the matter factor over the weights of N^vee fails on these complex representations, and dropping the matter factor fails", all(ctrl[k]["dual"] < len(l) and ctrl[k]["none"] < len(l) for k, _a, l in extra if k.startswith("SU(3)")) and all(ctrl[k]["none"] < len(l) for k, _a, l in extra), str(ctrl)))
    summary["A3 with printed matter factors"] = ctrl
    return {"checks": checks, "population": summary,
            "controls": {"positive": "the Wilson lines: a single support cell and no unknowns (the seed is the answer)", "negative": "in pure gauge, every window line with bubbling perturbed two ways (bubbling removed; bubbling doubled): bar, the seed, O(q) and the support bound still hold, and (★) rejects"},
            "notes": f"With matter the solver returns the dressed image F; the chart is F/Z (gn_abe_kalgebra), so the comparison divides first.  The perturbations are Weyl-covariant by construction (the whole bubbling is scaled).  Total {round(time.time() - t0, 1)} s.", "inputs": []}


GRAFT_CASES = [
    # (datum, line (m, e), donor (m, e)) — the cases of a probe in the source repository
    ("SU(2)", su_2, ((2,), (0,)), ((4,), (0,))),
    ("SU(2)", su_2, ((2,), (1,)), ((4,), (0,))),
    ("SU(2)", su_2, ((1,), (0,)), ((3,), (0,))),
    ("SU(3)", lambda: su_n(3), ((1, 1), (0, 0)), ((2, 2), (0, 0))),
    ("Spin(5)", None, ((1, 1), (0, 0)), ((2, 2), (0, 0))),
]


def _weyl_covariant(D, res):
    """f_{w·p} == w·f_p for every cell p and every w (the measured transport, TorusRational.weyl_act)."""
    for p, f in res.items():
        for w in D.weyl_elements():
            g = res.get(tuple(D.act_cochar(w, tuple(p))))
            if g is None:
                if not f.weyl_act(w).simplify().is_zero():
                    return False
            elif not (f.weyl_act(w) + (-g)).simplify().is_zero():
                return False
    return True


def _smallest_dominant_coroot(D):
    """The dominant coroot of smallest height ⟨Σ⁺, ·⟩ — the thinnest step that grows conv(W·m) by a layer (at a
    non-simply-laced group the highest coroot is a thicker step: at G2 it made the solve outgrow 12 GB)."""
    import star_bubbling as SB
    cor = {tuple(SB.coroot_of(D, a)) for a in D.positive_roots()}
    dom = [c for c in cor if tuple(D.dominant_cochar_rep(c)) == c]
    return min(dom, key=lambda c: sum(D.shift_pairing(c, a) for a in D.positive_roots()))


def _enlargement(D, m):
    """(k, β, larger hull): β the smallest dominant coroot, k the fewer of 1 and 2 steps whose added cells include one
    OFF the Weyl orbit of m + kβ.  Cells on that orbit are the extremal orbit of the larger hull: a
    residual there has no poles of its own (every wall is a pole of the dressing ψ), so bar + O(q) alone force it to
    zero — a layer made only of them tests nothing about (★).  At rank one the first step always is such a layer."""
    import star_bubbling as SB
    beta = _smallest_dominant_coroot(D)
    hull = {tuple(p) for p in SB.tropical_support(D, m)}
    for k in (1, 2):
        mp = tuple(a + k * b for a, b in zip(m, beta))
        big = [tuple(p) for p in SB.tropical_support(D, mp)]
        outer = {tuple(D.act_cochar(w, mp)) for w in D.weyl_elements()}
        if any(p not in hull and p not in outer for p in big):
            break
    return k, beta, big


def enlarged_support_one(A, lab):
    """The support condition MEASURED rather than assumed, at one pure-gauge line L_{m,e}.

    (★) + bar + O(q), with Weyl covariance (the paper's A1-A3), are solved on the cells of conv(W·(m + k·β)) — the
    hull grown by k steps of the smallest dominant coroot β (`_enlargement`) — and the unique solution is read off
    outside conv(W·m).  The outermost added layer is the extremal orbit of the larger hull, where a residual has no
    poles of its own (every wall there is already a pole of the dressing ψ), so bar + O(q) alone force it to zero: a
    𝖖-palindromic O(q) polynomial vanishes.  So k is 2 whenever one step adds only that orbit (always at rank one),
    and 1 otherwise.  The CONTROL re-solves with the
    residue rows removed (`admits_e` admitting nothing): the O(q) rows are then cell-local, so an added orbit
    representative with a non-pivot column carries a nonzero bar-invariant O(q) filling, and on those cells it is
    (★), not bar + O(q), that forces the zero (`star_bubbling.joint_solve`, `support=` and `stats=`)."""
    import star_bubbling as SB
    import wrq_torus as WT
    D = A.datum
    m, e = (tuple(t) for t in _label_parts(lab, False))
    k, beta, big = _enlargement(D, m)
    hull = {tuple(p) for p in SB.tropical_support(D, m)}
    added = [p for p in big if p not in hull]
    st, Ff, _sup, _interior, _kf = SB.joint_solve(D, m, e, pad=1, support=big)
    zero = equal = False
    if st == "unique":
        zero = all(Ff.get(p) is None or Ff[p].simplify().is_zero() for p in added)
        seed = {tuple(a): f for a, f in WT.leading_orbit(D, m, e).residuals().items()}
        x = SB.build_element(D, seed, {p: f for p, f in Ff.items() if p in hull})
        equal = _same_residuals(x.residuals(), A.chart(lab).residuals())
    stats = {}
    SB.joint_solve(D, m, e, pad=1, support=big, admits_e=lambda _e: False, stats=stats)
    reps = sorted({stats["rep_of"][p] for p in added if p in stats["rep_of"]})
    free = [r for r in reps if any(c not in stats["pivots"] for c in stats["cols"][r])]
    return dict(status=st, zero=zero, equal=equal, layers=k, step=beta, added=len(added), reps=len(reps), free=len(free))


def enlarged_support_one_matter(A, lab):
    """The support condition MEASURED at one (G, N) line L_{m,e}, every μ-sector.

    The matter solver (`matter_multislot.solve_canonical_matter_vec`, `support=`) solves each μ-sector by (★) + bar
    + O(q) on the cells of conv(W·(m + k·β)) (`_enlargement`), the O(q) condition on the quotient by the dressing Z as
    always.  EVERY sector is solved: the degree law, which forces some sectors without a solve, is a statement about
    the hull's cells and is not let decide the cells it is tested against — the sectors it would force on the hull
    are counted.  The tower is read off outside conv(W·m) at every sector; on the hull its quotient by Z is compared
    with the production chart.  The CONTROL re-solves each sector with its own offset and the residue rows removed,
    as in pure gauge (`enlarged_support_one`), and counts the (sector, added orbit representative) pairs left free."""
    import star_bubbling as SB
    import matter_star_bubbling as MSB
    from matter_multislot import solve_canonical_matter_vec, divide_by_Z_vec
    D = A.datum
    m, e = (tuple(t) for t in _label_parts(lab, True))
    k, beta, big = _enlargement(D, m)
    hull = {tuple(p) for p in SB.tropical_support(D, m)}
    added = [p for p in big if p not in hull]
    st = {}
    F = solve_canonical_matter_vec(D, A.matter, m, e, pad=1, lines=A.lines, support=big, stats=st)   # raises unless every sector is unique
    zero = all(F[s].get(p) is None or F[s][p].simplify().is_zero() for s in F for p in added)
    percell = {}
    for s, lvl in F.items():
        for c, v in lvl.items():
            if tuple(c) in hull:
                percell.setdefault(tuple(c), {})[tuple(s)] = v
    Q = {}
    for c, coeffs in percell.items():
        q, okd = divide_by_Z_vec(D, A.matter, c, coeffs)
        if not okd:
            raise ValueError(f"cell {c} not divisible by Z")
        Q[c] = {tuple(s): v.simplify() for s, v in q.items() if not v.simplify().is_zero()}
    equal = _same_residuals(Q, A.chart(lab).residuals())
    sectors = st.get("sectors", {})
    reps_n = free_n = free_sectors = 0
    for s, info in sectors.items():
        cst = {}
        MSB.solve_level(D, A.matter, m, e, s, pad=1, lines=A.lines, oq_offset=info["offset"] or None,
                        support=big, admits_e=lambda _e: False, stats=cst, max_pad_retry=0)
        reps = {cst["rep_of"][p] for p in added if p in cst.get("rep_of", {})}
        free = [r for r in reps if any(c not in cst.get("pivots", ()) for c in cst["cols"][r])]
        reps_n += len(reps)
        free_n += len(free)
        free_sectors += bool(free)
    return dict(status="unique", zero=zero, equal=equal, layers=k, step=beta, added=len(added), sectors=len(F),
                forced=sum(1 for v in sectors.values() if v["forced_on_hull"]), reps=reps_n, free=free_n,
                free_sectors=free_sectors)


#: the theories whose enlarged-support solves fit the fast depth, which has no per-theory memory cap.  G2 does not:
#: one step of its smallest dominant coroot adds only the extremal orbit of the larger hull (tests nothing about (★)),
#: and two steps at L_((1, 0), (0, 0)) outgrew a 3.5 GB address-space cap (measured 2026-09-24); with the highest
#: coroot as the step the fast run grew to 12.8 GB and was killed.  G2 runs at the extensive depth, capped.  The four
#: matter theories of the fast windows take 14 s together, every μ-sector solved (measured 2026-09-26, 90 MB).
FAST_ENLARGED = ("pure SU(2)", "pure SO(3)", "pure SU(3)", "pure Sp(4)",
                 "SU(2) + 1 fund. (roster su2-nf1)", "SU(2) + 2 fund. (roster su2-nf2, U(2) flavour)",
                 "SO(3) + adjoint (N=2*)", "SU(3) + 1 fund. (roster su3-nf1)")


def _enlarged_support_theory(environment, name):
    """One theory of the enlarged-support measurement of conj:abeKalgebra/support: (checks, population entry) —
    pure gauge by `enlarged_support_one`, (G, N) by `enlarged_support_one_matter` (every μ-sector).  At the extensive
    depth it runs in a forked child with a memory budget (`_per_theory`)."""
    A, ms, es, fs = conjecture_windows(environment)[name]
    matter = fs is not None
    W, unreached, errors = built_window(A, ms, es, fs)
    t1 = time.time()
    agg = dict(unique=0, zero=0, equal=0, added=0, reps=0, free=0, free_lines=0, sectors=0, forced=0)
    short, errs, bad, layers = [], {}, [], None
    for lab in W:
        try:
            with budget(2 * LABEL_BUDGET):
                r = (enlarged_support_one_matter if matter else enlarged_support_one)(A, lab)
        except (_Budget, MemoryError):
            short.append(str(lab))      # a resource limit: reported not finished, never a failure
            continue
        except Exception as ex:  # noqa: BLE001
            if _memory_exhausted(ex):
                short.append(str(lab))
                continue
            errs[str(lab)] = f"{type(ex).__name__}: {str(ex)[:160]}"
            continue
        layers = (layers or set()) | {(r["layers"], r["step"])}
        agg["unique"] += r["status"] == "unique"
        agg["zero"] += r["zero"]
        agg["equal"] += r["equal"]
        for key in ("added", "reps", "free") + (("sectors", "forced") if matter else ()):
            agg[key] += r[key]
        agg["free_lines"] += r["free"] > 0
        if not (r["status"] == "unique" and r["zero"] and r["equal"]) and len(bad) < 3:
            bad.append(f"{lab}: {r}")
    n = len(W) - len(short) - len(errs)
    unr = f" ({len(short)} more not finished within the budget)" if short else ""
    lay = " or ".join(f"{k} step{'s' if k > 1 else ''} of {b}" for k, b in sorted(layers or ()))
    ok = n > 0 and agg["unique"] == n and agg["zero"] == n and agg["equal"] == n and not errs and not errors
    detail = "; ".join(bad) + (f"; errors {errs}" if errs else "") + (f"; build errors {errors}" if errors else "")
    if matter:
        checks = [
            check(f"{name}: enlarged support — on all {n} window lines{unr}, every μ-sector ({agg['sectors']} in all, "
                  f"{agg['forced']} of them sectors the degree law forces on the hull, solved here too) of (★) + bar + "
                  f"O(q) on the cells of conv(W·(m + k·β)), the hull grown by {lay} (β the smallest dominant coroot), "
                  f"has a unique solution; the tower vanishes on all {agg['added']} added cells at every sector, and on "
                  f"the hull its quotient by the dressing Z equals the production line", ok, detail),
            check(f"{name}: control — without the residue rule, {agg['free']} of the {agg['reps']} (μ-sector, added "
                  f"Weyl-orbit representative) pairs (on {agg['free_lines']} of the {n} lines) admit a nonzero "
                  f"bar-invariant O(q) filling, which (★) forces to zero",
                  agg["free"] > 0),
        ]
    else:
        checks = [
            check(f"{name}: enlarged support — on all {n} window lines{unr}, (★) + bar + O(q) solved on the cells of "
                  f"conv(W·(m + k·β)), the hull grown by {lay} (β the smallest dominant coroot), have a unique solution; it "
                  f"vanishes on all {agg['added']} added cells and equals the production line", ok, detail),
            check(f"{name}: control — without the residue rule, {agg['free']} of the {agg['reps']} added Weyl-orbit "
                  f"representatives (on {agg['free_lines']} of the {n} lines) admit a nonzero bar-invariant O(q) filling, "
                  f"which (★) forces to zero; the rest, the outermost layer, are emptied by bar + O(q) alone",
                  agg["free"] > 0),
        ]
        for key in ("sectors", "forced"):
            agg.pop(key)
    return checks, dict(labels=len(W), unreached=[str(l) for l in unreached] + short, errors=errs,
                        steps=sorted(f"{k}×{b}" for k, b in (layers or ())),
                        **agg, seconds=round(time.time() - t1, 1))


def _flavour_windows(nf):
    """The SU(2) + nf doublets window: (magnetic, electric) charges of the sections (neutral flavour label), and the
    largest m_a + m_b whose products are decomposed.  Small at nf >= 3: a trace at nf = 3, m = 2 costs a minute."""
    deep = battery_depth() == "extensive"
    if nf <= 2:
        return [(m, e) for m in (0, 1, 2) for e in (0, 1)], 3
    if nf == 3:
        return [(m, e) for m in ((0, 1, 2) if deep else (0, 1)) for e in ((0, 1) if deep else (0,))], 2
    return [(m, e) for m in (0, 1) for e in ((0, 1) if deep else (0,))], 2


def _det_twisted(R, elt, shift):
    """An R(U(n)) element as its weight diagram with every weight moved by shift·(1, …, 1): its det^shift twist."""
    ab = R.to_abelian(elt)
    return {tuple(Fraction(x) + shift for x in w): c for w, c in ab.terms.items() if c}


def _flavour_one(environment, name):
    """sec:coulomb/flavour-enhancement at SU(2) + nf doublets (name 'SU(2) + nf doublets'): (checks, entry).

    The algebra is built with its manifest R(U(nf)) flavour (`GNAbeKAlgebra(su_2(), (1,), nf)`).  The enhancement
    to SO(2nf) is read off by re-expansion: a U(nf) character lifts to Spin(2nf) iff its weight diagram is
    W(D_nf)-invariant (`so2nf_characters`).  The twist that makes the data covariant is fixed by ρ, not fitted: ρ
    shifts the flavour charge by κ_f(m) = det^{-m} (the draft's formula at N = 2 ⊗ C^nf), so the section at magnetic
    charge m is det^{m/2} times a Spin(2nf) section — twist a trace by det^{-m/2}, a pairing I(a, b) by
    det^{(m_a - m_b)/2}, a structure constant c_{ab}^c by det^{(m_c - m_a - m_b)/2}.  At odd m the twist is
    half-integral and the lifted weights are spinor weights."""
    import so2nf_characters as SO
    nf = int(name.split("+")[1].split()[0])
    A = GNAbeKAlgebra(su_2(), (1,), nf=nf)
    R = A.coefficient_ring()
    K = 4
    w0 = (0,) * nf
    t1 = time.time()
    window, m_prod = _flavour_windows(nf)
    secs = [(A.fold((m,), (e,))[0], w0) for m, e in window]

    def mag(lab):
        return abs(lab[0][0][0])

    tally = {k: dict(n=0, orders=0, fail=[], roundtrip=[], b_flip=0) for k in ("traces", "pairings", "products")}

    def scan(kind, what, coeffs, shift, count=True):
        t = tally[kind]
        t["n"] += count
        for k, c in coeffs.items():
            if not c.terms:
                continue
            poly = _det_twisted(R, c, shift)
            t["orders"] += 1
            enh, content, _gen = SO.verify_flavour_enhancement(nf, poly)
            if not enh:
                if len(t["fail"]) < 3:
                    t["fail"].append(f"{what} q^{k}")
                continue
            if SO.reconstruct(nf, content) != poly:
                t["roundtrip"].append(f"{what} q^{k}")
            flip = {(-w[0],) + tuple(w[1:]): v for w, v in poly.items()}
            t["b_flip"] += flip == poly          # the extra reflection of O(2nf): recorded, not claimed

    checks, unreached = [], []

    def compute(what, fn):
        """fn() under the per-object budget: None (recorded as not reached) past the time or memory budget."""
        try:
            with budget(2 * LABEL_BUDGET):
                return fn()
        except (_Budget, MemoryError):
            unreached.append(what)
            return None
        except Exception as ex:  # noqa: BLE001
            if _memory_exhausted(ex):
                unreached.append(what)
                return None
            raise

    tr1 = A.trace(A.identity(), K=K)
    enh, content, _gen = SO.verify_flavour_enhancement(nf, _det_twisted(R, tr1.coeffs[2], 0)) if 2 in tr1.coeffs else (False, None, False)
    adj = {(1, 1): 1, (1, -1): 1} if nf == 2 else {(1, 1) + (0,) * (nf - 2): 1}
    checks.append(check(f"{name}: positive control — Tr(1) at q^2 (the flavour currents) is the SO({2 * nf}) adjoint "
                        f"{sorted(adj)} after re-expansion", enh and content == adj, f"content {content}"))
    unshifted_fail, odd_orders = 0, 0
    for s in secs:
        tr = compute(f"Tr L_{s[0]}", lambda: A.trace(s, K=K))
        if tr is None:
            continue
        scan("traces", f"Tr L_{s[0]}", tr.coeffs, Fraction(-mag(s), 2))
        if mag(s) % 2:
            for k, c in tr.coeffs.items():
                if c.terms:
                    odd_orders += 1
                    unshifted_fail += not SO.is_weyl_invariant(nf, _det_twisted(R, c, 0))
    for a in secs:
        for b in secs:
            I = compute(f"I({a[0]}, {b[0]})", lambda: A.inner_product(a, b, K=K))
            if I is not None:
                scan("pairings", f"I({a[0]}, {b[0]})", I.coeffs, Fraction(mag(a) - mag(b), 2))
    for i, a in enumerate(secs):
        for b in secs[i:]:
            if mag(a) + mag(b) > m_prod:
                continue
            rf = compute(f"L_{a[0]}·L_{b[0]}", lambda: A.to_R_form(A.multiply(a, b)))
            if rf is None:
                continue
            tally["products"]["n"] += 1
            for u, rl in rf.terms.items():
                scan("products", f"L_{a[0]}·L_{b[0]} → L_{u[0]}", rl.coeffs, Fraction(mag(u) - mag(a) - mag(b), 2),
                     count=False)
    for kind, what, tw in (("traces", "traces of", "det^(-m/2)"), ("pairings", "pairings I(a, b) of", "det^((m_a - m_b)/2)"),
                           ("products", "structure constants of", "det^((m_c - m_a - m_b)/2)")):
        t = tally[kind]
        nobj = {"traces": f"{t['n']} window sections", "pairings": f"{t['n']} pairs of window sections",
                "products": f"{t['n']} products with m_a + m_b <= {m_prod}"}[kind]
        miss = sum(1 for u in unreached if (u.startswith("Tr") if kind == "traces" else u.startswith("I(") if kind == "pairings" else "·" in u))
        nobj += f" ({miss} more not reached within the budget)" if miss else ""
        if not t["n"]:
            continue                    # nothing of this kind computed: reported in the population, never a check
        checks.append(check(f"{name}: the {what} {nobj}, twisted by {tw}, are W(D_{nf})-invariant at every one of their "
                            f"{t['orders']} nonzero orders (through q^{K} for traces and pairings) and lift exactly to "
                            f"Spin({2 * nf}) characters", t["orders"] > 0 and not t["fail"] and not t["roundtrip"],
                            f"not invariant: {t['fail']}; round trip: {t['roundtrip'][:3]}"))
    if odd_orders:
        checks.append(check(f"{name}: negative control — without the det^(-m/2) twist the odd-m traces are not "
                            f"W(D_{nf})-invariant ({unshifted_fail} of {odd_orders} orders fail)", unshifted_fail > 0))
    entry = dict(sections=[str(s[0]) for s in secs], K=K, max_product_m=m_prod, not_reached=unreached,
                 **{k: dict(objects=t["n"], orders=t["orders"], not_invariant=t["fail"], round_trip_failures=t["roundtrip"],
                            also_single_flip_invariant=t["b_flip"]) for k, t in tally.items()},
                 unshifted_odd_trace_orders_failing=unshifted_fail, seconds=round(time.time() - t1, 1))
    return checks, entry


def _usp_windows(form):
    """The N=2* windows of the USp(2) test: (magnetic, electric) charges of the sections in su_2 units (neutral
    flavour label; e even at the SO(3) form, whose lattice admits half-integral m), and the largest m_a + m_b whose
    products are decomposed."""
    deep = battery_depth() == "extensive"
    if form == "SU(2)":
        return [(m, e) for m in ((0, 1, 2, 3) if deep else (0, 1, 2)) for e in (0, 1)], (3 if deep else 2)
    H = Fraction(1, 2)
    return [(m, e) for m in ((0, H, 1, 3 * H) if deep else (0, H, 1)) for e in (0, 2)], (2 if deep else 1)


def _usp_one(environment, name):
    """sec:coulomb/flavour-enhancement, the real case: N=2* — one adjoint hypermultiplet — at the SU(2) and SO(3)
    forms (name 'SU(2) + adjoint (N=2*), USp(2)' or 'SO(3) + ...'), manifest U(1) → USp(2) = SU(2)_F: (checks, entry).

    A U(1) character lifts to USp(2) iff its weight diagram is integral (the fundamental of USp(2) has the U(1)
    weights ±1) and invariant under w ↦ −w, the Weyl group of USp(2).  The twist is fixed by ρ, as in the pseudo-real
    case: ρ shifts the flavour charge by κ_f(m) = μ^{-2m} (su_2 units; the adjoint row's measurement), so the section
    at magnetic charge m is μ^m times a USp(2) section — twist a trace by μ^{-m}, a pairing I(a, b) by μ^{m_a - m_b},
    a structure constant c_{ab}^c by μ^{m_c - m_a - m_b}.  Every twist on nonzero data is integral, the SO(3) form's
    half-integral m included: a trace is read at the zero cell, which conv(W·m) holds only at integral m, and a
    pairing or a structure constant is nonzero only when the magnetic charges agree modulo the coroot lattice."""
    from g_matter_roster import highest_root
    from global_form import adjoint_lines
    form = name.split()[0]
    A = GNAbeKAlgebra(su_2(), highest_root(su_2()), nf=1,
                      **({"lines": adjoint_lines(su_2())} if form == "SO(3)" else {}))
    R = A.coefficient_ring()
    K = 4
    t1 = time.time()
    window, m_prod = _usp_windows(form)
    secs = [(A.fold((m,), (e,))[0], (0,)) for m, e in window]

    def mag(lab):
        return Fraction(abs(lab[0][0][0]))

    def twisted(c, shift):
        return {(Fraction(w[0]) + shift,): v for w, v in R.to_abelian(c).terms.items() if v}

    def usp2(poly):
        return (all(w[0].denominator == 1 for w in poly)
                and {(-w[0],): v for w, v in poly.items()} == poly)

    tally = {k: dict(n=0, orders=0, fail=[]) for k in ("traces", "pairings", "products")}

    def scan(kind, what, coeffs, shift, count=True):
        t = tally[kind]
        t["n"] += count
        for k, c in coeffs.items():
            if not c.terms:
                continue
            t["orders"] += 1
            if not usp2(twisted(c, shift)) and len(t["fail"]) < 3:
                t["fail"].append(f"{what} q^{k}")

    checks, unreached = [], []

    def compute(what, fn):
        """fn() under the per-object budget: None (recorded as not reached) past the time or memory budget."""
        try:
            with budget(2 * LABEL_BUDGET):
                return fn()
        except (_Budget, MemoryError):
            unreached.append(what)
            return None
        except Exception as ex:  # noqa: BLE001
            if _memory_exhausted(ex):
                unreached.append(what)
                return None
            raise

    tr1 = A.trace(A.identity(), K=K)
    cur = twisted(tr1.coeffs[2], 0) if 2 in tr1.coeffs else {}
    adj = {(Fraction(-2),): 1, (Fraction(0),): 1, (Fraction(2),): 1}
    checks.append(check(f"{name}: positive control — Tr(1) at q^2 (the flavour currents) is the USp(2) adjoint "
                        f"μ^-2 + 1 + μ^2", cur == adj, f"q^2 coefficient {cur}"))
    bare_fail, bare_orders = 0, 0
    for s in secs:
        tr = compute(f"Tr L_{s[0]}", lambda: A.trace(s, K=K))
        if tr is None:
            continue
        scan("traces", f"Tr L_{s[0]}", tr.coeffs, -mag(s))
        if mag(s):
            for k, c in tr.coeffs.items():
                if c.terms:
                    bare_orders += 1
                    bare_fail += not usp2(twisted(c, 0))
    for a in secs:
        for b in secs:
            I = compute(f"I({a[0]}, {b[0]})", lambda: A.inner_product(a, b, K=K))
            if I is not None:
                scan("pairings", f"I({a[0]}, {b[0]})", I.coeffs, mag(a) - mag(b))
    for i, a in enumerate(secs):
        for b in secs[i:]:
            if mag(a) + mag(b) > m_prod:
                continue
            rf = compute(f"L_{a[0]}·L_{b[0]}", lambda: A.to_R_form(A.multiply(a, b)))
            if rf is None:
                continue
            tally["products"]["n"] += 1
            for u, rl in rf.terms.items():
                scan("products", f"L_{a[0]}·L_{b[0]} → L_{u[0]}", rl.coeffs, mag(u) - mag(a) - mag(b), count=False)
    for kind, what, tw in (("traces", "traces of", "μ^(-m)"), ("pairings", "pairings I(a, b) of", "μ^(m_a - m_b)"),
                           ("products", "structure constants of", "μ^(m_c - m_a - m_b)")):
        t = tally[kind]
        nobj = {"traces": f"{t['n']} window sections", "pairings": f"{t['n']} pairs of window sections",
                "products": f"{t['n']} products with m_a + m_b <= {m_prod}"}[kind]
        miss = sum(1 for u in unreached if (u.startswith("Tr") if kind == "traces" else u.startswith("I(") if kind == "pairings" else "·" in u))
        nobj += f" ({miss} more not reached within the budget)" if miss else ""
        if not t["n"]:
            continue                    # nothing of this kind computed: reported in the population, never a check
        checks.append(check(f"{name}: the {what} {nobj}, twisted by {tw}, are integral and invariant under μ ↦ μ^-1 "
                            f"— USp(2) characters — at every one of their {t['orders']} nonzero orders (through q^{K} for "
                            f"traces and pairings)", t["orders"] > 0 and not t["fail"], f"not invariant: {t['fail']}"))
    if bare_orders:
        checks.append(check(f"{name}: negative control — without the μ^(-m) twist the traces of the sections with m ≠ 0 "
                            f"are not USp(2) characters ({bare_fail} of {bare_orders} orders fail)", bare_fail > 0))
    entry = dict(sections=[str(s[0]) for s in secs], K=K, max_product_m=str(m_prod), not_reached=unreached,
                 **{k: dict(objects=t["n"], orders=t["orders"], not_invariant=t["fail"]) for k, t in tally.items()},
                 untwisted_trace_orders_failing=f"{bare_fail} of {bare_orders}", seconds=round(time.time() - t1, 1))
    return checks, entry


def check_flavour_enhancement(environment):
    """sec:coulomb/flavour-enhancement: the pseudo-real case, SU(2) with N_f doublets, U(N_f) → SO(2N_f) (N_f = 1 has
    nothing to test: SO(2) = U(1), no Weyl reflection beyond the manifest ones); and the real case, N=2* at the SU(2)
    and SO(3) forms, U(1) → USp(2) (`_usp_one`)."""
    checks, summary = [], {}
    t0 = time.time()
    nfs = (2, 3, 4) if battery_depth() == "extensive" else (2, 3)
    for nf in nfs:
        name = f"SU(2) + {nf} doublets"
        cks, entry = _per_theory(_flavour_one, environment, name)
        checks.extend(cks)
        summary[name] = entry
    summary["SU(2) + 1 doublet"] = "SO(2) = U(1): no Weyl reflection beyond the manifest U(1), so nothing to test"
    for form in ("SU(2)", "SO(3)"):
        name = f"{form} + adjoint (N=2*), USp(2)"
        cks, entry = _per_theory(_usp_one, environment, name)
        checks.extend(cks)
        summary[name] = entry
    # negative control on a theory that must NOT enhance: U(1) with three charge-1 hypers carries the same R(U(3)) but
    # a complex representation, so its U(3) stays U(3) — the test has to reject its Schur index
    import so2nf_characters as SO
    from root_datum import u_n
    B = GNAbeKAlgebra(u_n(1), (1,), nf=3)
    trb = B.trace(B.identity(), K=4)
    bad = [k for k, c in sorted(trb.coeffs.items()) if c.terms
           and not SO.is_weyl_invariant(3, _det_twisted(B.coefficient_ring(), c, 0))]
    checks.append(check("negative control — U(1) with three charge-1 hypers (a complex representation, flavour U(3) "
                        "with no enhancement) fails the same W(D_3) test: its Schur index is not invariant at "
                        f"q^{bad}", bool(bad), f"Tr(1) = {dict((k, str(v)) for k, v in trb.coeffs.items() if v.terms)}"))
    summary["U(1) + 3 (negative control)"] = dict(orders_not_invariant=bad)
    return {"checks": checks, "population": summary,
            "controls": {"positive": "Tr(1) at q^2, the flavour currents, re-expands as the SO(2N_f) adjoint, and at "
                                     "N=2* as the USp(2) adjoint μ^-2 + 1 + μ^2",
                         "negative": "the odd-m traces without the det^(-m/2) twist are not W(D_N_f)-invariant; U(1) with "
                                     "three charge-1 hypers (complex, no enhancement) fails the W(D_3) test; at N=2* the "
                                     "traces with m ≠ 0 without the μ^(-m) twist are not μ ↦ μ^-1 invariant"},
            "notes": f"Total {round(time.time() - t0, 1)} s.", "inputs": []}


def check_support(environment):
    """conj:abeKalgebra/support: the bubbling of every window line lies in conv(W·m) (structural: the solver's
    unknowns live there); (★) + bar + O(q) solved on a LARGER hull have a unique solution that vanishes outside
    conv(W·m) — in pure gauge (`enlarged_support_one`) and, every μ-sector, with matter
    (`enlarged_support_one_matter`) — with the control that the residue rule is what empties the added cells not
    already emptied by bar + O(q); and grafting a bubbling residual onto a genuine line outside
    conv(W·m) is rejected by (★) alone — both as the repository's single-cell graft
    (a probe in the source repository) and Weyl-symmetrised over the orbit of the grafted cell, which keeps
    the paper's A1 (Weyl-covariant, bar-invariant bubbling) intact."""
    import star_bubbling as SB
    from root_datum import b_n_simply_connected
    from wrq_torus import WRQTorus as _W
    checks, summary = [], {}
    t0 = time.time()
    for name, (A, ms, es, fs) in conjecture_windows(environment).items():
        D = A.datum
        W, unreached, errors = built_window(A, ms, es, fs)
        ok, stray = not errors, []
        for lab in W:
            m = _label_parts(lab, fs is not None)[0]
            trop = set(tuple(p) for p in tropical_support(D, m))
            out = chart_support(A.chart(lab)) - trop
            if out:
                ok = False
                stray.append((lab, sorted(out)))
        unr = f" ({len(unreached)} more not reached within the {LABEL_BUDGET} s budget)" if unreached else ""
        checks.append(check(f"{name}: the bubbling of all {len(W)} built window lines{unr} lies in conv(W·m) (tropical_support)", ok, f"{stray[:2]}" + (f"; build errors {errors}" if errors else "")))
        summary[name] = dict(labels=len(W), unreached=[str(l) for l in unreached], contained=ok)
    # The containment above is STRUCTURAL: the solver's unknowns live in conv(W·m) and the declared optimizations
    # preserve support.  The measurement of the draft's remark is the enlarged-support solve (pure gauge, and every
    # μ-sector with matter), after a positive control on a known answer — a scan is not run if its control fails.
    pc = enlarged_support_one(PureGAbeKAlgebra(su_2()), ((2,), (0,)))
    pc_ok = pc["status"] == "unique" and pc["zero"] and pc["equal"] and pc["free"] > 0
    checks.append(check("positive control: at SU(2) L_((2,), (0,)) the enlarged-support solve returns the production line, "
                        "zero outside conv(W·m), and its control finds an added cell that bar + O(q) leave free", pc_ok, str(pc)))
    from g_matter_roster import highest_root
    Am = GNAbeKAlgebra(su_2(), highest_root(su_2()), nf=1)
    pcm = enlarged_support_one_matter(Am, Am.fold((2,), (0,)))
    pcm_ok = pcm["status"] == "unique" and pcm["zero"] and pcm["equal"] and pcm["free"] > 0
    checks.append(check("positive control with matter: at SU(2) + adjoint L_((2,), (0,)) the enlarged-support solve of "
                        "every μ-sector returns the production line, zero outside conv(W·m) at every sector, and its "
                        "control finds a (μ-sector, added cell) pair that bar + O(q) leave free", pcm_ok, str(pcm)))
    if pc_ok:
        for name, (A, ms, es, fs) in conjecture_windows(environment).items():
            if (fs is not None and not pcm_ok) or (battery_depth() != "extensive" and name not in FAST_ENLARGED):
                continue
            cks, entry = _per_theory(_enlarged_support_theory, environment, name)
            checks.extend(cks)
            summary.setdefault(name, {})["enlarged_support"] = entry
    grafts = {}
    for tag, mk, lab0, donor0 in GRAFT_CASES:
        D = mk() if mk is not None else b_n_simply_connected(2)
        P = PureGAbeKAlgebra(D)
        lab, donor = P.fold(*lab0), P.fold(*donor0)
        x = P.chart(lab)
        m, e = lab
        pos = SB.verify_axioms(D, m, e, x)
        pos_ok = all(v[0] for v in pos.values()) and _weyl_covariant(D, x.residuals())
        sup = set(tuple(p) for p in SB.tropical_support(D, m))
        y = P.chart(donor)
        orb_d = {tuple(D.act_cochar(w, donor[0])) for w in D.weyl_elements()}
        donors = [(n, f) for n, f in y.residuals().items() if tuple(n) not in orb_d and not f.simplify().is_zero() and SB._oq_holds(D, f)]
        outside = [tuple(p) for p in SB.tropical_support(D, donor[0]) if tuple(p) not in sup]
        stats = {"single-cell": [0, 0, 0], "Weyl-symmetrised": [0, 0, 0]}     # tried, other conditions hold, (★) rejects
        bad = []
        tried = 0
        for p in outside:
            for (_n, f) in donors:
                for kind in ("single-cell", "Weyl-symmetrised"):
                    res = dict(x.residuals())
                    if kind == "single-cell":
                        res[p] = res[p] + f if p in res else f
                    else:
                        add = {}
                        for w in D.weyl_elements():
                            q = tuple(D.act_cochar(w, p))
                            add[q] = add[q] + f.weyl_act(w) if q in add else f.weyl_act(w)
                        if all(g.simplify().is_zero() for g in add.values()):
                            continue
                        for q, g in add.items():
                            res[q] = res[q] + g if q in res else g
                    z = _W(D, res)
                    ax = SB.verify_axioms(D, m, e, z)
                    others = all(ax[k][0] for k in ("bar (W1)", "seed (W2)", "O(𝖖)"))
                    if kind == "Weyl-symmetrised":
                        others = others and _weyl_covariant(D, res)
                    st = stats[kind]
                    st[0] += 1
                    st[1] += others
                    st[2] += not ax["(★)"][0]
                    if (not others or ax["(★)"][0]) and len(bad) < 3:
                        bad.append(f"{kind} graft at {p}: " + ", ".join(f"{k} {v[0]}" for k, v in ax.items()))
                tried += 1
                if tried >= 6:
                    break
            if tried >= 6:
                break
        grafts[f"{tag} L_{lab} (donor L_{donor})"] = dict(positive_control=pos_ok, **{k: dict(tried=v[0], other_conditions_hold=v[1], star_rejects=v[2]) for k, v in stats.items()})
        for kind, (n_t, n_o, n_r) in stats.items():
            checks.append(check(f"{tag} L_{lab}: {n_t} {kind} grafts outside conv(W·m) keep bar, the seed and O(q){' and Weyl covariance' if kind != 'single-cell' else ''}; (★) rejects every one (true line passes all five conditions: {pos_ok})", pos_ok and n_t > 0 and n_o == n_t and n_r == n_t, "; ".join(bad)))
    summary["grafts"] = grafts
    return {"checks": checks, "population": summary,
            "controls": {"positive": "each genuine line passes all five conditions of star_bubbling.verify_axioms and is Weyl-covariant; the enlarged-support solve returns the production line at SU(2) L_(2,0), and at SU(2) + adjoint L_(2,0) every μ-sector, before any theory is scanned",
                         "negative": "the grafts themselves (the support condition fails for them by construction; the finding is that (★) alone also rejects them); for the enlarged support, the same system without the residue rows, which leaves the added cells strictly inside the larger hull free"},
            "notes": f"The repository's measurement (kalgebra.md, 30/30) grafts onto a single cell, which also breaks Weyl covariance, part of the paper's A1; the Weyl-symmetrised grafts keep A1 and isolate (★).  The containment of the built lines in conv(W·m) is structural (the solver's unknowns live there); the enlarged-support solve is the measurement.  Total {round(time.time() - t0, 1)} s.", "inputs": []}


# --------------------------------------------------------------------------
# eq:Iexplicit at the extensive depth: the printed pairing and its conjectural half on the lineup
# --------------------------------------------------------------------------
#: seconds per theory for the pairing battery; pairs not compared within it are reported, never counted as passes
IEXPLICIT_BUDGET = 1800


def _label_series(Pp, A, lab, K):
    """{m: q-series to q^K} of a line's residuals in Printed's padded variables (v, then one mu per matter slot)."""
    return {m: rat_to_series(nd[0].trunc(K + 200), nd[1], Pp.nv, K) for m, nd in _padded_residuals(Pp, A.chart(lab)).items()}


def _trace_face(Alg, la, lb, K):
    """Tr(rho(f)·g) by multiply-then-trace: rho(f)·g by the cocycle convolution, then its magnetic-0 residue."""
    from kalgebra import _coeff_min_exp
    x = Alg.multiply(Alg.rho(la), lb)
    neg = min([0] + [_coeff_min_exp(c) for c in x.terms.values() if not c.is_zero()])
    return Alg.trace_element(x, K - neg)


def _torus_trace_face(A, la, lb, K, M):
    """Tr(rho(f)·g) on the rational quantum torus itself: the torus rho of f's chart (eq:rhotorus), the cocycle
    product with g's chart (eq:fgprod), and the Schur-measure residue of the magnetic-0 residual (eq:measure).  No
    canonical decomposition is formed, so this is the paper's Tr applied to the torus element rho(f)g — the same
    number multiply-then-trace computes (the trace is linear), without building the lines rho(f)g decomposes into
    (at Sp(4), rho(L_(1,1),0)·L_(1,1),0 needs lines up to m = (2,2) at large electric charge: over 6 GB).
    Pure: the raw exact series |W|·Tr (w_cutoff=False, linear).  Matter: {mu-level: q-series} at Nahm window K + 2,
    the window the class's trace and pairing use (at K alone one SU(2)+2 pair lost a term the class keeps), which the
    class raises to K + pad for a residual whose q-expansion starts at q^-pad (the audit)."""
    x = A.chart(la).rho() * A.chart(lb)
    if M == 0:
        t = x.trace(K, w_cutoff=False)
        return S2({e: {(): c} for e, c in t._coeffs.items() if e <= K}, K, 0)
    c: dict = {}
    for lv, lp in x.trace(K, K + 2).items():
        for e, v in lp._coeffs.items():
            if e <= K and v:
                c.setdefault(e, {})[tuple(lv)] = c.get(e, {}).get(tuple(lv), 0) + v
    return S2(c, K, M)


def _flavour_to_S(A, t, K):
    """An RPowerSeries over GNAbeKAlgebra's flavour ring ⊗_i R(U(n_i)) as a q-series in the M slot fugacities of the
    printed formula: each basis character χ_λ is expanded by its weight diagram onto the slots of its group
    (`A.groups` keeps the slot indices).  At n_i = 1 this is the identity (k,) -> μ^k; for n_i copies of one irrep the
    class carries a U(n_i) character, which one μ per slot sees as its Schur polynomial."""
    M = A.M
    R = A.coefficient_ring()
    tensor = hasattr(R, "factors")
    facs = R.factors if tensor else (R,)
    cache: dict = {}

    def monos(key):
        if key not in cache:
            out = {(0,) * M: 1}
            for fac, (_lam, idxs), b in zip(facs, A.groups, tuple(key) if tensor else (key,)):
                new: dict = {}
                for mono, c in out.items():
                    for wt, mult in fac.character(b).items():
                        mm = list(mono)
                        for i, x in zip(idxs, wt):
                            mm[i] += x
                        new[tuple(mm)] = new.get(tuple(mm), 0) + c * mult
                out = new
            cache[key] = out
        return cache[key]
    c: dict = {}
    for e, v in t.coeffs.items():
        if e > K:
            continue
        for key, n in v.terms.items():
            if n:
                for mono, mult in monos(key).items():
                    c.setdefault(e, {})[mono] = c.get(e, {}).get(mono, 0) + n * mult
    return S2(c, K, M)


def _lineup_pairing_lines(A, ms, es):
    """Five lines of a lineup window for the pairing battery, built one at a time: the identity, the smallest Wilson
    line, the two smallest magnetic charges (each at the smallest electric charge the form admits with it) and one
    dyonic line at the first of them — admitted labels only, distinct after folding."""
    lines = getattr(A, "lines", None)
    adm = lambda m, e: lines is None or lines.admits(tuple(m), tuple(e))
    size = lambda t: (sum(abs(Fraction(x)) for x in t), tuple(-Fraction(x) for x in t))
    zm, ze = tuple(0 for _ in ms[0]), tuple(0 for _ in es[0])
    es_s = sorted(es, key=size)
    picks = [(zm, ze)]
    w = next((e for e in es_s if any(e) and adm(zm, e)), None)
    if w is not None:
        picks.append((zm, w))
    seen = {A.fold(*p) for p in picks}
    mags = []
    for m in sorted((m for m in ms if any(m)), key=size):
        e = next((e for e in es_s if adm(m, e) and A.fold(m, e) not in seen), None)
        if e is not None:
            mags.append((m, e))
            seen.add(A.fold(m, e))
        if len(mags) == 2:
            break
    picks += mags
    if mags:
        m0, e0 = mags[0]
        d = next((e for e in es_s if e != e0 and any(e) and adm(m0, e) and A.fold(m0, e) not in seen), None)
        if d is not None:
            picks.append((m0, d))
    out = []
    for p in picks:
        lab = A.fold(*p)
        if lab not in out:
            out.append(lab)
    return out, [p for p in mags]


def _iexplicit_one(environment, name):
    """One theory of the eq:Iexplicit lineup battery: (its checks, its population entry)."""
    A, ms, es, fs = conjecture_windows(environment)[name]
    checks = []
    t1 = time.time()
    D = A.datum
    matter = fs is not None
    slots = slot_weights(D, A.matter) if matter else ()
    M = len(slots)
    Pp = Printed(D, M)
    K = 3 if (matter and D.dim > 1) else 4
    nW = len(list(D.weyl_elements()))
    conv = (lambda t: _flavour_to_S(A, t, K)) if matter else (lambda t: rps_int_S(t, K))
    L, mags = _lineup_pairing_lines(A, ms, es)
    if len(mags) == 2 and len(L) == 5:
        L = [L[0], L[1], L[2], L[4], L[3]]        # the second (largest) magnetic line last: its pairs are the costly ones
    total = len(L) ** 2
    done = eq_p = eq_f = ortho = 0
    neg, bad, err, stop = None, [], None, None
    G_, F_ = {}, {}

    def ser(l):
        # a line's residual series, expanded only when a pair first needs it (a heavy line costs only its own pairs)
        if l not in G_:
            G_[l] = _label_series(Pp, A, l, K)
            # f·g·measure is exact through q^K only if no factor starts below q^0 (the measure never does); a line
            # that did would leave the top exponents short, so stop rather than compare an inexact series
            low = min((g.val() for g in G_[l].values() if g.val() is not None), default=0)
            if low < 0:
                raise ValueError(f"{l}: a residual series starts at q^{low}; the printed pairing is exact only through q^{K + low}")
            F_[l] = {m: invert_vars(g, set(range(Pp.nv))) for m, g in G_[l].items()}
        return F_[l], G_[l]
    # pairs among the first k lines complete before any pair involving line k+1
    order = [(L[i], L[k]) for k in range(len(L)) for i in range(k + 1)] + [(L[k], L[i]) for k in range(len(L)) for i in range(k)]
    order.sort(key=lambda ab: max(L.index(ab[0]), L.index(ab[1])))
    try:
        with budget(IEXPLICIT_BUDGET):
            for la, lb in order:
                fa, _ga = ser(la)
                _fb, gb = ser(lb)
                got = conv(A.inner_product(la, lb, K))
                # the class's pairing IS Tr(rho(f)·g): the Schur residue of the magnetic-0 residual of rho(f)·g,
                # assembled sector by sector (wrq_torus.inner_by_sector = trace_residual(pairing_residual)); the
                # printed sum takes each sector's contour integral at |v| = 1 against its own shifted measure, so
                # printed == class is the paper's contour shift v -> q^(+-m) v: the conjectural half
                pp = _printed_pairing(D, slots, fa, gb, K)
                p_ok = pp.eq_to(got.scale(nW), K)
                # a second computation of Tr(rho(f)·g), from the torus product itself: equal to the class's by the
                # total-charge-0 part of the convolution (a theorem), so a consistency check, not the conjecture
                tf = _torus_trace_face(A, la, lb, K, M)
                f_ok = tf.eq_to(got.scale(nW) if M == 0 else got, K)
                # positive control on the independently computed printed series (the class's series is truncated to
                # q^0..q^K, so only here can a negative q-power show): |W|·(delta + O(q))
                lead = {k: v for k, v in pp.c.get(0, {}).items() if v}
                o_ok = min((e for e, mc in pp.c.items() if any(mc.values())), default=0) >= 0 and lead == ({(0,) * M: nW} if la == lb else {})
                done += 1
                eq_p += p_ok
                eq_f += f_ok
                ortho += o_ok
                if not (p_ok and f_ok and o_ok) and len(bad) < 3:
                    bad.append(f"({la}, {lb}): printed == Tr {p_ok}, class == torus {f_ok}, printed delta+O(q) {o_ok}")
                if mags and neg is None and la == lb == A.fold(*mags[0]):
                    fm, gm = ser(la)
                    neg = not _printed_pairing(D, slots, fm, gm, K, shifted=False).eq_to(got.scale(nW), K)
    except _Budget:
        stop = f"{IEXPLICIT_BUDGET} s"
    except MemoryError:
        stop = f"{THEORY_MEMORY_GB} GB"
        G_.clear()
        F_.clear()
    except Exception as ex:  # noqa: BLE001
        if _memory_exhausted(ex):
            stop = f"{THEORY_MEMORY_GB} GB"
            G_.clear()
            F_.clear()
        else:
            err = f"{type(ex).__name__}: {str(ex)[:160]}"
    left = f" ({total - done} of {total} pairs not reached within {stop})" if done < total and err is None else ""
    if err is not None:
        checks.append(check(f"{name}: the pairing battery ran", False, err))
    elif done:
        checks.append(check(f"{name}: the conjectural half — the printed sum over m of contour integrals (eq:Iexplicit, |W| = {nW}; each sector at |v| = 1 against its own shifted measure) equals Tr(rho(f)·g) as the class computes it (the Schur residue of the magnetic-0 residual of rho(f)·g, assembled sector by sector) on {done} pairs of {len(L)} lines through q^{K}{left}", eq_p == done, "; ".join(bad)))
        checks.append(check(f"{name}: consistency of two computations of Tr(rho(f)·g) — the class's sector assembly and the torus product rho(f)·g give the same Schur residue on the same {done} pairs (equal by a theorem; not the conjecture)", eq_f == done, "; ".join(bad)))
        checks.append(check(f"{name}: positive control — the printed pairing is |W|·(delta + O(q)) on the same {done} pairs (no negative q-power, delta at q^0)", ortho == done, "; ".join(bad)))
    if neg is not None:
        checks.append(check(f"{name}: negative control — the unshifted measure changes the printed pairing of {A.fold(*mags[0])} with itself", neg))
    entry = dict(lines=[str(l) for l in L], K=K, weyl=nW, slots=M, pairs=total, compared=done, printed_eq_trace=eq_p,
                 class_eq_torus=eq_f, printed_delta_plus_O_q=ortho, unshifted_detected=neg, stopped_by=stop, error=err, seconds=round(time.time() - t1, 1))
    print(f"[extensive] eq:Iexplicit {name}: {done}/{total} pairs, printed == Tr {eq_p}, class == torus {eq_f}, printed delta {ortho}, "
          f"negative {neg}{', ' + err if err else ''} ({time.time() - t1:.0f} s)", file=sys.stderr, flush=True)
    return checks, entry


def check_iexplicit_lineup(environment):
    """eq:Iexplicit at the extensive depth, on the 24 lineup theories, for five lines per theory (four where the window
    has a single nonzero magnetic charge) and all their ordered pairs.  The conjectural half: the printed sum over m
    (each sector's contour integral at |v| = 1 against its shifted vector and matter measure, prefactor
    (q^2;q^2)^{2 rk}/|W|, computed here from the charts) equals Tr(rho(f)·g) as the class computes it (the Schur residue
    of the magnetic-0 residual of rho(f)·g, assembled sector by sector) — the two routes share nothing but the chart,
    and their equality is the paper's contour shift v -> q^(+-m) v.  A consistency check: the torus product
    rho(f)·g gives the same residue (a theorem).  Positive control: the printed pairing is |W|·(delta + O(q)) on every
    pair; negative control: the unshifted measure changes the printed pairing of the first magnetic line with itself."""
    checks, pop = [], {}
    t0 = time.time()
    for name in conjecture_windows(environment):
        c, e_ = _per_theory(_iexplicit_one, environment, name)
        checks += c
        pop[name] = e_
    return {"checks": checks, "population": pop,
            "controls": {"positive": "the printed pairing is |W|·(delta + O(q)) on every compared pair (no negative q-power, delta at q^0); pure SU(2) and SU(3), whose pairs the fast run also compares, are among the theories",
                         "negative": "the unshifted measure (every sector at the m = 0 measure) changes the printed pairing of the first magnetic line with itself"},
            "notes": f"Extensive population of eq:Iexplicit: five lines per lineup theory (identity, the smallest Wilson line, the two smallest magnetic charges at the smallest admitted electric charge, one dyonic line; four where the window has a single nonzero magnetic charge), all ordered pairs, q^4 (q^3 with matter above rank one), {IEXPLICIT_BUDGET} s and {THEORY_MEMORY_GB} GB per theory, each theory in its own process.  Total {round(time.time() - t0, 1)} s.", "inputs": []}


def check_iexplicit(environment):
    """eq:Iexplicit: at the fast depth the run of eq:measure (same adapter); at the extensive depth the lineup battery."""
    if battery_depth() == "extensive":
        return check_iexplicit_lineup(environment)
    return check_measure_and_pairing(environment)


def check_axioms_subsection(environment):
    checks = []
    A = PureGAbeKAlgebra(su_2())
    d = su_2()
    W = build_window(A, [(m,) for m in range(0, 3)], [(e,) for e in range(-2, 3)])
    # (★) <=> the difference operator preserves the characters: neumann_row(x) is a Laurent polynomial (no denominator) iff criterion holds
    ok_eq = True
    detail = ""
    for lab in W:
        x = A.chart(lab)
        c_ok, _rep = criterion(x)
        row = neumann_row(x)
        poly_ok = not row.den
        if c_ok != poly_ok:
            ok_eq = False
            detail = f"{lab}: criterion {c_ok}, polynomial row {poly_ok}"
    bare = WRQTorus.from_f(d, {(1,): A.chart(((1,), (0,))).residuals()[(1,)], (-1,): A.chart(((1,), (0,))).residuals()[(-1,)]})
    c_bare, _ = criterion(bare)
    row_bare = neumann_row(bare)
    checks.append(check(f"the residue rule is equivalent to a well-defined action on the characters: criterion(x) <=> neumann_row(x) has no denominator, on {len(W)} built lines (both true) and on the bare 't Hooft line U_2 + U_-2 (both false)", ok_eq and not c_bare and bool(row_bare.den), detail))
    # O(q) bubbling + I_fg formula => δ + O(q): the leading orbits contribute δ, the bubbling O(q)
    ok_oq = True
    for lab in W:
        x = A.chart(lab)
        m = lab[0]
        for mm, r in x.residuals().items():
            if mm != m and mm != tuple(-t for t in m):   # a bubbling sector
                ser = su2_series(tr_to_rat(r, 1), 6)
                ok_oq &= (ser.val() is None) or ser.val() >= 1
    checks.append(check("the bubbling corrections of every window line are O(q) as series (A2), and the pairing is δ + O(q) on all pairs", ok_oq and all(A.verify_orthonormality(a, b, K=4) for a in W for b in W)))
    checks.append(check("the two faces I_fg = Tr rho(f) g = Tr g rho^-1(f) agree on all pairs (the contour-shift cancellation the paper calls trickier)", all(A.verify_trace_pairing_faces(a, b, K=4) for a in W for b in W)))
    # the draft's statement proper: the residue rule <=> a well-defined action on the space of G characters,
    # u^m chi_e(v) = chi_e(q^{2m} v).  neumann_row above tests chi_0 = 1 only, which the repository itself calls
    # strictly weaker than (★); here every character of a window, at rank one and two and with matter (d-form with the
    # printed matter factors of eq:cocha, power by power in mu), with perturbed lines as controls
    from root_datum import b_n_simply_connected
    from wrq_torus import levi_character
    from weyl_torus_ring import TorusRational, TorusLaurent
    from matter_wrq_torus import slot_weights

    def char_tr(D, e):
        chi = levi_character(D, (0,) * D.dim, tuple(e))
        return TorusRational(D, chi, {}) if isinstance(chi, TorusLaurent) else chi

    def acts(D, levels, es):
        for dd in levels.values():
            for e in es:
                chi = char_tr(D, e)
                tot = TorusRational.zero(D)
                for a, d in dd.items():
                    tot = tot + d * chi.q_shift(tuple(2 * x for x in a))
                tot = tot.simplify()
                if not (tot.is_zero() or not tot.den):
                    return False
        return True

    def perturbed(A, x, lab, factor):
        D = A.datum
        m = (lab[0] if isinstance(A, GNAbeKAlgebra) else lab)[0]
        orbit = {tuple(D.act_cochar(w, tuple(m))) for w in D.weyl_elements()}
        res = {}
        for n, f in x.residuals().items():
            if tuple(n) in orbit:
                res[n] = f
            elif factor:
                res[n] = {k: v + v for k, v in f.items()} if isinstance(f, dict) else f + f
        return type(x)(D, x.Nf, res) if isinstance(A, GNAbeKAlgebra) else type(x)(D, res)
    ca = {}
    for name, A, labs, es in (
        ("pure SU(2)", PureGAbeKAlgebra(su_2()), [((1,), (0,)), ((1,), (1,)), ((2,), (0,)), ((2,), (1,)), ((3,), (0,))], [(0,), (1,), (2,), (3,)]),
        ("pure SU(3)", PureGAbeKAlgebra(su_n(3)), [((1, 1), (0, 0)), ((1, 1), (1, 0)), ((2, 2), (0, 0))], [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0)]),
        ("pure Spin(5)", PureGAbeKAlgebra(b_n_simply_connected(2)), [((1, 1), (0, 0)), ((2, 1), (0, 0))], [(0, 0), (1, 0), (0, 1)]),
        ("SU(2)+1", GNAbeKAlgebra(su_2(), (1,), nf=1), [((1,), (0,)), ((2,), (0,)), ((2,), (1,))], [(0,), (1,), (2,)]),
        ("SU(3)+1", GNAbeKAlgebra(su_n(3), (1, 0), nf=1), [((1, 1), (0, 0)), ((1, 1), (1, 0))], [(0, 0), (1, 0), (0, 1)]),
    ):
        D = A.datum
        slots = slot_weights(D, A.matter) if isinstance(A, GNAbeKAlgebra) else ()
        L = [A.fold(m, e) for m, e in labs]
        g = sum(acts(D, paper_d_levels(A.chart(l), D, slots), es) for l in L)
        bad = sum(acts(D, paper_d_levels(perturbed(A, A.chart(l), l, f), D, slots), es) for l in L for f in (0, 1))
        ca[name] = (g, len(L), bad, len(es))
    checks.append(check("the residue rule as a well-defined action on the G characters, u^m chi_e(v) = chi_e(q^(2m) v): every line acts on every character of its window, sum_a d_a(v) chi_e(q^(2a) v) a Laurent polynomial (" + "; ".join(f"{k}: {v[0]}/{v[1]} lines, {v[3]} characters" for k, v in ca.items()) + ")", all(v[0] == v[1] for v in ca.values())))
    checks.append(check("negative control: with the bubbling removed or doubled no line acts on the characters (" + ", ".join(f"{k}: {v[2]} of {2 * v[1]}" for k, v in ca.items()) + " perturbed lines act)", all(v[2] == 0 for v in ca.values())))
    return {"checks": checks, "population": {"window": f"{len(W)} SU(2) lines", "action on characters": {k: f"{v[0]}/{v[1]} lines x {v[3]} characters" for k, v in ca.items()}}, "controls": {"positive": "built lines satisfy both sides", "negative": "the bare line fails both sides; lines with their bubbling removed or doubled act on no character"},
            "notes": "The equivalence (★) <=> preserving Λ = R(G) is the repository's own reading of the residue rule (residue_cancellation_recognition.md); here it is checked as a biconditional on both truth values.", "inputs": []}


ADAPTERS = {
    "sec:coulomb/presentation": check_presentation,
    "eq:rho-witten": check_rho_witten,
    "eq:rhotorus": check_rho_witten,             # the same run; its record repeats under this id
    "eq:measure": check_measure_and_pairing,
    "eq:Iexplicit": check_iexplicit,             # fast: the run of eq:measure; extensive: the lineup battery
    "eq:cocha": check_cocha,
    "sec:coulomb/su2-examples": check_su2_examples,
    "sec:coulomb/adjoint": check_adjoint,
    "sec:coulomb/global-forms": check_global_forms,
    "conj:abeKalgebra/consistency": check_conjecture_windows,
    "sec:coulomb/labels": check_labels,
    "conj:abeKalgebra/existence-uniqueness": check_existence_uniqueness,
    "conj:abeKalgebra/support": check_support,
    "sec:coulomb/flavour-enhancement": check_flavour_enhancement,
    "sec:coulomb/axioms-subsection": check_axioms_subsection,
}
