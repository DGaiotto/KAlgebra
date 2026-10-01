"""checks_rg.py — the design record adapters for the draft's Section 4, "RG flows of K_q
algebras": the quantum-dilogarithm group (sec:rg/E-group), matter removal as an
RG flow (def:rg/matter-removal), the Seiberg–Witten conjectures on a sample of
the shipped enumerated dictionary (def:sw / conj:cluster / conj:S-unique /
conj:pbw / def:rg/no-exotic), the upper-cluster-algebra conjecture on sample
charts (conj:upper), factored flows (conj:factored), the monodromy = rho^2
statement (def:rg/central-charge) and the flavoured quiver constraint
(sec:rg/flavoured-quivers).

Web tier: the dictionary claims range over a deterministic SAMPLE of the
shipped tier (the first entries in cell order); the full census is the
extensive tier's / the local machine's.  Every record names its sample.
"""
from __future__ import annotations

import itertools
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
from checks_kq import S, check, inv_one_minus as inv_one_minus_1, qpoch_n  # noqa: E402
from checks_uq_a1d3 import S2, chi_mu  # noqa: E402


# --------------------------------------------------------------------------
# sec:rg/E-group — the quantum dilogarithm, its inverse and bar image, E_{q^2} outside the group
# --------------------------------------------------------------------------
def E_coeff(n: int, K: int) -> S:
    """[x^n] E_q(x) = (-q)^n / (q^2;q^2)_n as a q-series to q^K (Laurent in q for the bar image below)."""
    out = S.q(n, K + 40, (-1) ** n)
    for j in range(1, n + 1):
        out = out * inv_one_minus_1(2 * j, K + 40)
    return out.trunc(K)


def E_inv_coeff(n: int, K: int) -> S:
    """[x^n] E_q(x)^{-1} = [x^n] (-q x; q^2)_∞ = q^{n^2} / (q^2;q^2)_n."""
    out = S.q(n * n, K + 40)
    for j in range(1, n + 1):
        out = out * inv_one_minus_1(2 * j, K + 40)
    return out.trunc(K)


def bar_S(s: S) -> S:
    return S({-e: v for e, v in s.c.items()}, 10 ** 6)


def check_E_group(environment):
    K = 24
    checks = []
    # printed expansion E_q(x) = 1 - q/(1-q^2) x + ...
    c1 = E_coeff(1, K)
    checks.append(check("printed: E_q(x) = 1 - q/(1-q^2) x + ... ([x^1] = -q - q^3 - q^5 - ...)", E_coeff(0, K).eq_to(S.one(K), K) and c1.eq_to(S({2 * j + 1: -1 for j in range(0, K // 2 + 1)}, K), K)))
    # g^{-1}(q) = g(q^{-1}) for E_q: the bar image of [x^n]E_q is a rational function; compare its q-expansion with [x^n]E_q^{-1}
    ok = True
    detail = ""
    for n in range(0, 7):
        # bar of (-q)^n/(q^2;q^2)_n = (-q)^{-n}/(q^{-2};q^{-2})_n = (-q)^{-n} * (-1)^n q^{n(n+1)} / (q^2;q^2)_n = q^{n^2}/(q^2;q^2)_n  (exact identity, checked as series)
        lhs = E_inv_coeff(n, K)
        rhs = S.q(-n, K + 40, (-1) ** n)
        for j in range(1, n + 1):
            # 1/(1 - q^{-2j}) = -q^{2j}/(1 - q^{2j})
            rhs = rhs * S.q(2 * j, K + 40, -1) * inv_one_minus_1(2 * j, K + 40)
        rhs = rhs.trunc(K)
        if not lhs.eq_to(rhs, K):
            ok = False
            detail = detail or f"n={n}: E^-1 {lhs.trunc(8)} vs bar {rhs.trunc(8)}"
    checks.append(check(f"g^-1(q) = g(q^-1) for g = E_q(x): [x^n] of the inverse (-qx;q^2)_∞ equals the bar image of [x^n] E_q, n <= 6, through q^{K}", ok, detail))
    # the generalisation E^{(s)} (eq:mult): E^{(s)}_q(x) = Π_{j=-s}^{s} E_q((-1)^{2s} q^{2j} x)^{(-1)^{2s}}.  Each x^n coefficient is a rational
    # function of q -- a Laurent-polynomial numerator over denominators (1-q^{2j}) -- and the bar image is taken EXACTLY on that form
    # (1/(1-q^{-2j}) = -q^{2j}/(1-q^{2j})), never on a truncated series (whose bar is a polynomial in q^{-1}, not the re-expansion).
    # Both sides are compared as Laurent polynomials after clearing the common denominator (q^2;q^2)_n^f (f factors), so the check is exact.
    class Rat:
        """Σ_D num_D(q) / Π_{j∈D}(1-q^{2j}); D a sorted tuple of j's with multiplicity, num_D a Laurent polynomial {exp: int}."""
        __slots__ = ("t",)

        def __init__(self, terms=None):
            t = {}
            for D, num in (terms or {}).items():
                num = {e: v for e, v in num.items() if v}
                if num:
                    t[tuple(sorted(D))] = num
            self.t = t

        def __add__(self, o):
            out = {D: dict(num) for D, num in self.t.items()}
            for D, num in o.t.items():
                acc = out.setdefault(D, {})
                for e, v in num.items():
                    acc[e] = acc.get(e, 0) + v
            return Rat(out)

        def __mul__(self, o):
            out = {}
            for D1, n1 in self.t.items():
                for D2, n2 in o.t.items():
                    acc = out.setdefault(tuple(sorted(D1 + D2)), {})
                    for e1, v1 in n1.items():
                        for e2, v2 in n2.items():
                            acc[e1 + e2] = acc.get(e1 + e2, 0) + v1 * v2
            return Rat(out)

        def bar(self):
            out = {}
            for D, num in self.t.items():
                shift, sgn = sum(2 * j for j in D), (-1) ** len(D)
                out[D] = {shift - e: sgn * v for e, v in num.items()}
            return Rat(out)

        def cleared(self, n, f):
            """Multiply by (q^2;q^2)_n^f (a common multiple of every denominator here) -> a Laurent polynomial {exp: int}."""
            out = {}
            for D, num in self.t.items():
                poly = dict(num)
                for j in range(1, n + 1):
                    for _ in range(f - D.count(j)):
                        nxt = {}
                        for e, v in poly.items():
                            nxt[e] = nxt.get(e, 0) + v
                            nxt[e + 2 * j] = nxt.get(e + 2 * j, 0) - v
                        poly = nxt
                for e, v in poly.items():
                    out[e] = out.get(e, 0) + v
            return {e: v for e, v in out.items() if v}

    def E_factor_coeff(n, shift, sign, inverse):
        """[x^n] E_q(sign q^shift x)^{∓1}: (sign q^shift)^n (-q)^n/(q^2;q^2)_n, or (sign q^shift)^n q^{n^2}/(q^2;q^2)_n for the inverse."""
        D = tuple(range(1, n + 1))
        if inverse:
            return Rat({D: {shift * n + n * n: sign ** n}})
        return Rat({D: {shift * n + n: (sign ** n) * (-1) ** n}})

    def E_factor_x(xmax, shift, sign, inverse):
        return {n: E_factor_coeff(n, shift, sign, inverse) for n in range(xmax + 1)}

    def mul_x(a, b, xmax):
        return {i: sum((a[j] * b[i - j] for j in range(i + 1)), Rat()) for i in range(xmax + 1)}

    def E_s(two_s, xmax, flip_exponent_of_last=False, shifts=None):
        """E^{(s)} as an x-series of Rat coefficients: factors j = -s..s, argument (-1)^{2s} q^{2j} x, exponent (-1)^{2s}.
        Returns (g, g_inverse) with g_inverse built from the inverse exponents factor by factor."""
        sign = (-1) ** two_s
        inverse = bool(two_s % 2)
        shifts = [-two_s + 2 * i for i in range(two_s + 1)] if shifts is None else list(shifts)
        g = {0: Rat({(): {0: 1}})}
        for n in range(1, xmax + 1):
            g[n] = Rat()
        g_inv = dict(g)
        for k, sh in enumerate(shifts):
            inv_here = inverse if not (flip_exponent_of_last and k == len(shifts) - 1) else (not inverse)
            g = mul_x(g, E_factor_x(xmax, sh, sign, inv_here), xmax)
            g_inv = mul_x(g_inv, E_factor_x(xmax, sh, sign, not inv_here), xmax)
        return g, g_inv

    xmax = 6
    ok_s, detail_s = True, ""
    for two_s in (0, 1, 2, 3):
        g, g_inv = E_s(two_s, xmax)
        f = two_s + 1
        for n in range(xmax + 1):
            if g_inv[n].cleared(n, f) != g[n].bar().cleared(n, f):
                ok_s = False
                detail_s = detail_s or f"2s={two_s}, n={n}"
    checks.append(check("eq:mult  E^(s)_q(x) = Π_{j=-s}^{s} E_q((-1)^{2s} q^{2j} x)^{(-1)^{2s}} satisfies g^-1(q) = g(q^-1) for s = 0, 1/2, 1, 3/2: [x^n] compared EXACTLY as Laurent polynomials over (q^2;q^2)_n^{2s+1}, n <= 6", ok_s, detail_s))
    # negative controls: an asymmetric set of shifts (E_q(-x)^-1 E_q(-q x)^-1) and a mixed exponent (E_q(-q^-1 x)^-1 E_q(-q x)) must FAIL
    g_a, g_a_inv = E_s(1, xmax, shifts=[0, 1])
    fail_a = any(g_a_inv[n].cleared(n, 2) != g_a[n].bar().cleared(n, 2) for n in range(1, xmax + 1))
    g_m, g_m_inv = E_s(1, xmax, flip_exponent_of_last=True)
    fail_m = any(g_m_inv[n].cleared(n, 2) != g_m[n].bar().cleared(n, 2) for n in range(1, xmax + 1))
    bar_inv_m = all(g_m[n].cleared(n, 2) == g_m[n].bar().cleared(n, 2) for n in range(xmax + 1))
    checks.append(check("negative controls: the asymmetric shifts E_q(-x)^-1 E_q(-q x)^-1 fail g^-1(q) = g(q^-1), and the mixed exponents E_q(-q^-1 x)^-1 E_q(-q x) are bar-INVARIANT instead (bar g = g, not g^-1)", fail_a and fail_m and bar_inv_m))
    # E_{q^2}(x) is not in the group: at degree x^1 the target -q^2/(1-q^4) has the denominator (1+q^2), while every generator E^{(s)} contributes
    # a Laurent polynomial over (1-q^2); multiplying by (1-q^2) the target is the infinite series -q^2 + q^4 - q^6 + ..., a finite sum of Laurent polynomials cannot match it
    target = S.q(2, K, -1) * inv_one_minus_1(4, K)
    target_times = (target * S({0: 1, 2: -1}, K)).trunc(K)
    # generators' x^1 coefficients times (1-q^2): -(q^{1-2s} + ... + q^{1+2s}) (for integer s; half-integer s carry the sign and inverse) — all Laurent polynomials
    gen_polys = {}
    for two_s in range(0, 9):
        s_ = two_s / 2
        js = [-s_ + i for i in range(two_s + 1)]
        sign = (-1) ** two_s
        coeff = S({}, K)
        for j in js:
            # [x^1] E_q(sign q^{2j} x)^{sign} times (1-q^2): sign * (-q)(sign q^{2j}) / (1-q^2) * (1-q^2) for the +1 power; for the inverse power [x^1]E^{-1} = q/(1-q^2)... use the exact coefficients
            base = E_coeff(1, K + 20) if sign == 1 else E_inv_coeff(1, K + 20)
            coeff = coeff + base * S.q(int(2 * j), K + 40, sign)
        gen_polys[two_s] = (coeff * S({0: 1, 2: -1}, K + 40)).trunc(K)
    finite = all(len(p.c) <= 2 * two_s + 2 and max(p.c) - min(p.c) <= 4 * (two_s + 1) for two_s, p in gen_polys.items() if p.c)
    checks.append(check("E_{q^2}(x) is not in the group: (1-q^2)·[x^1]E_{q^2} = -q^2 + q^4 - q^6 + ... is an infinite series while every generator's (1-q^2)·[x^1]E^(s) is a Laurent polynomial (s <= 4 listed), so no finite integral combination matches degree one",
                        finite and len([e for e in target_times.c if e <= K]) >= K // 2 - 1, f"target·(1-q^2) through q^10: {target_times.trunc(10)}; generator polynomials: " + "; ".join(f"2s={t}: {p.trunc(12)}" for t, p in list(gen_polys.items())[:3])))
    return {"checks": checks, "population": {"K": K, "n": "<= 6", "s": "0, 1/2"}, "controls": {"positive": "the printed leading terms; E_q and E^(s) for s <= 3/2 exactly", "negative": "asymmetric shifts fail; mixed exponents are bar-invariant instead"},
            "notes": "Exact rational arithmetic in one variable (each generator is a function of a single X_gamma).  The statement for every g in the group follows from the generators because the bar involution is ANTImultiplicative on the quantum torus: bar(gh) = bar(h)bar(g) = h^-1 g^-1 = (gh)^-1.  Non-membership of E_{q^2} is the degree-one argument for an indecomposable gamma: only the factors E^(s)(X_gamma) contribute at degree gamma, so (1-q^2)[X_gamma]g is a Laurent polynomial while (1-q^2)[x]E_{q^2} = -q^2/(1+q^2) is not.", "inputs": []}


# --------------------------------------------------------------------------
# def:rg/matter-removal — S = Π_w E_q(μ v^w) on characters; the R-axioms on the flow
# --------------------------------------------------------------------------
def _mr_su2_nf1_hand(environment):
    """The 2026-09-22 checks at SU(2) + 1 fundamental, with the dilogarithm product hand-coded as a two-variable series:
    kept as the positive control of the lineup-wide checks below."""
    from g_matter_over_pure import GMatterOverPure
    from root_datum import su_2
    checks = []
    K = 8
    G = GMatterOverPure(su_2(), (1,), nf=1)
    t0 = time.time()
    S_rg = G.rg_generator(K)
    # independent: Π_{w=±1} E_q(μ v^w) as a series in q with (v, μ) monomials, then organised by SU(2) characters of v: the coefficient of v^e must assemble into chi_e(v) (Weyl symmetric)
    def E_two_var(mono, K):
        out = S2.one(K, 2)
        k = 0
        while 2 * k + 1 <= K:
            out = out * S2({e: p for e, p in inv_one_minus_2(2 * k + 1, mono, K, -1).c.items()}, K, 2)
            k += 1
        return out
    def inv_one_minus_2(e, mono, K, sign):
        c = {}
        j = 0
        while e * j <= K:
            c[e * j] = {tuple(x * j for x in mono): sign ** j}
            j += 1
        return S2(c, K, 2)
    prod = E_two_var((1, 1), K) * E_two_var((-1, 1), K)       # (v, μ) monomials: E(μ v) E(μ v^-1)
    # Weyl symmetry in v
    sym = all(prod.c[e].get((a, b), 0) == prod.c[e].get((-a, b), 0) for e in prod.c for (a, b) in prod.c[e])
    checks.append(check(f"Π_w E_q(μ v^w) for SU(2)+1 fundamental is Weyl-symmetric in v through q^{K} (so it expands on characters χ_e(v))", sym))
    # compare with the class's S_RG: keys are auxiliary labels (Wilson lines dressed by μ powers); expand each Habiro coefficient
    got = {}
    for lab, h in S_rg.items():
        lp = h.expand(K) if hasattr(h, "expand") else h
        got[lab] = lp
    detail = f"S_RG keys (first 6): {list(S_rg)[:6]}"
    # readout: the coefficient of q^e μ^b v^a in the class's S_RG, reconstructed from labels (wilson e, flavour b)
    ok_cmp = None
    try:
        recon = S2({}, K, 2)
        for lab, lp in got.items():
            # a label of the flavoured pure algebra: (((0,),(e,)),(b,)) — Wilson e (character χ_e(v)) times μ^b
            (m, e), f = lab
            if any(m):
                continue
            coeffs = lp._coeffs if hasattr(lp, "_coeffs") else {k: v for k, v in lp.coeffs.items()}
            chi = S2({0: {(w, f[0]): 1 for w in range(-e[0], e[0] + 1, 2)}}, K, 2)
            for qe, v in coeffs.items():
                if v:
                    recon = recon + chi.scale(int(v)).shift(qe)
        ok_cmp = recon.trunc(K).eq_to(prod.trunc(K), K)
        detail = f"first difference at q^{recon.trunc(K).first_diff(prod.trunc(K), K)}" if not ok_cmp else "identical through q^%d" % K
    except Exception as ex:  # noqa: BLE001
        detail = f"{type(ex).__name__}: {ex}; " + detail
    checks.append(check(f"S_RG of GMatterOverPure(SU(2), 1 fundamental) equals Π_w E_q(μ v^w) expanded on Wilson characters, through q^{K}", bool(ok_cmp), detail))
    # the R-axioms as instruments on this flow (emergent here: the UV multiply is derived through the flow).  UV labels of K_q[SU(2), 1]:
    # ((m, e), f) with m the coroot-coordinate flux, e the Levi weight, f the flavour charge — the identity, two Wilson lines, the monopole and a dyon.
    labels = [G.identity(), (((0,), (1,)), (0,)), (((0,), (2,)), (0,)), (((1,), (0,)), (0,)), (((1,), (1,)), (0,))]
    res = {}
    def run(name, fn):
        try:
            res[name] = bool(fn())
        except Exception as ex:  # noqa: BLE001
            res[name] = f"{type(ex).__name__}: {str(ex)[:80]}"
    run("R2 unital", G.verify_rg_unital)
    run("R2 multiplicative (all pairs)", lambda: all(G.verify_rg_multiplicative(a, b) for a in labels for b in labels))
    run("R2 bar-invariant images", lambda: all(G.verify_rg_bar_invariant(a) for a in labels))
    run("R3 discovery RG_a S = L_l(a) + O(q), K=2", lambda: all(G.verify_rg_discovery(a, K=2) for a in labels))
    run("R3 twist, exact form RG(rho_UV a) = rho_IR(tRG a)", lambda: all(G.verify_rg_trg_intertwine(a) for a in labels))
    checks.append(check(f"the R-axiom verifiers as instruments on the flow, {len(labels)} UV labels (identity, Wilson e=1,2, monopole, dyon): R2 unital / multiplicative / bar-invariant, R3 discovery and the exact twist", all(v is True for v in res.values()), str(res)))
    # R4 is BY CONSTRUCTION on this class (the UV trace is derived through the flow), so the meaningful test is against an INDEPENDENT presentation
    # of K_q[SU(2), 1]: the (G,N) abelianized tier (g_matter_roster 'su2-nf1'), whose pairing is the sector-measure constant term.  Compared on the
    # canonical coefficient data (the two series live over different ring instances, so == is not available).
    def canon(ps):
        return {n: dict(c.terms) for n, c in ps.coeffs.items() if c.terms}
    try:
        from g_matter_roster import roster
        T = roster("su2-nf1")
        Kc = 4
        diffs = []
        for a in labels:
            for b in labels:
                i_flow, i_tier = G.inner_product(a, b, Kc), T.inner_product(a, b, Kc)
                if canon(i_flow) != canon(i_tier):
                    diffs.append((a, b, repr(i_flow)[:60], repr(i_tier)[:60]))
        ok_r4 = not diffs
        note_r4 = f"{len(labels) ** 2} pairs, K={Kc}; identity row: {G.inner_product(labels[0], labels[0], Kc)!r}" if ok_r4 else str(diffs[:2])
    except Exception as ex:  # noqa: BLE001
        ok_r4, note_r4 = False, f"{type(ex).__name__}: {str(ex)[:120]}"
    checks.append(check(f"R4 as a CROSS-PRESENTATION check: the flow's I_(a,b) (trace transported through S_RG) equals the (G,N) abelianized tier's on all pairs of the {len(labels)} labels, K=4", ok_r4, note_r4))
    return {"checks": checks, "population": {"flow": "GMatterOverPure(su_2(), fundamental, nf=1)", "K": K, "labels": len(labels), "seconds": round(time.time() - t0, 1)},
            "controls": {"positive": "the product of dilogarithms is Weyl-symmetric", "negative": "n.a."},
            "notes": "The remark's S = Π_w E_q(μ v^w) re-interpreted on characters is the flow's S_RG; the R-axioms are emergent on this construction (the UV product is derived through the flow).", "inputs": []}


# The matter theories of the Section 3 lineup that GMatterOverPure reaches, and SU(3)+1 (not in the
# lineup; in the fast support windows).  GMatterOverPure builds pure G at the
# root datum's default form and takes no line lattice, so the non-simply-connected forms of the lineup are out of reach.
_MR_LINEUP = (
    # name, root datum, GNAbeKAlgebra matter (weight, nf) or (slot list, None), minimal monopole, a Wilson weight,
    # RG verifiers and R4 in the fast tier (measured 2026-09-26: SU(2)+1 4 s, +2 88 s, adjoint 60 s; N_f = 3, 4 and rank two
    # take minutes per step and run at the extensive depth)
    ("SU(2)+1", "su2", ((1,), 1), (1,), (1,), True),
    ("SU(2)+2", "su2", ((1,), 2), (1,), (1,), True),
    ("SU(2)+3", "su2", ((1,), 3), (1,), (1,), False),
    ("SU(2)+4", "su2", ((1,), 4), (1,), (1,), False),
    ("SU(2)+adjoint (N=2*)", "su2", ("adjoint", 1), (1,), (1,), True),
    ("SU(3)+adjoint (N=2*)", "su3", ("adjoint", 1), (1, 1), (1, 0), False),
    ("SU(3)+1", "su3", ((1, 0), 1), (1, 1), (1, 0), False),
    ("SU(3)+symmetric", "su3", ((2, 0), 1), (1, 1), (1, 0), False),
    ("SU(3)+symmetric+1", "su3", ([(2, 0), (1, 0)], None), (1, 1), (1, 0), False),
    ("Spin(5)+vector", "b2", ((1, 0), 1), (1, 1), (1, 0), False),
    ("G2+7", "g2", ((1, 0), 1), (1, 2), (1, 0), False),
    ("SU(2)xSU(2)+bifundamental", "P2", ((1, 1), 1), (1, 0), (1, 0), False),
    ("U(1)-U(2)+3", "u1u2", ([(1, 0, -1), ((0, 1, 0), 3)], None), (0, 1, 0), (0, 1, 0), False),
)
_MR_UNREACHED = ("SO(3)+adjoint (N=2*)", "rank 1 + adjoint (N=2*), the third global form", "SO(5)+vector")
#: seconds per theory for the RG verifiers and R4 (a step past it is reported as not reached, never as a pass)
MR_BUDGET = {"fast": 300, "extensive": 900}


def _mr_datum(key):
    from root_datum import b_n_simply_connected, g_2, product_datum, su_2, su_n, u_n
    return {"su2": su_2, "su3": lambda: su_n(3), "b2": lambda: b_n_simply_connected(2), "g2": g_2,
            "P2": lambda: product_datum([su_2(), su_2()]), "u1u2": lambda: product_datum([u_n(1), u_n(2)])}[key]()


def _mr_build(name):
    """(the (G,N) tier, the matter-removal flow with the same slots, the root datum, monopole, Wilson weight)."""
    from gn_abe_kalgebra import GNAbeKAlgebra
    from g_matter_roster import highest_root
    _, key, (lam, nf), mono, wil, _fast = next(r for r in _MR_LINEUP if r[0] == name)
    D = _mr_datum(key)
    if lam == "adjoint":
        lam = tuple(highest_root(D))
    NAT = GNAbeKAlgebra(D, lam, nf=nf) if nf is not None else GNAbeKAlgebra(D, list(lam))
    return NAT, NAT.flow_twin(), D, tuple(mono), tuple(wil)


def _mr_S_check(NAT, G, D, K=6):
    """The flow's S_RG, its characters expanded into monomials (levi_character), against Π_i Π_{w in wt(N_i)} E_q(μ_i v^w)
    multiplied out directly over the weights: the class builds S through elementary symmetric functions and character
    fusion, so the two routes share only the weights."""
    from g_matter_over_pure import matter_weights
    from wrq_torus import levi_character
    slots = tuple(NAT.matter)
    M = len(slots)

    def mul(a, b):
        c = {}
        for (qa, va, ma), x in a.items():
            for (qb, vb, mb), y in b.items():
                if qa + qb <= K:
                    k = (qa + qb, tuple(p + r for p, r in zip(va, vb)), tuple(p + r for p, r in zip(ma, mb)))
                    c[k] = c.get(k, 0) + x * y
        return {k: v for k, v in c.items() if v}

    def inv_poch(n):                                   # 1/(q^2;q^2)_n through q^K
        s = {0: 1}
        for j in range(1, n + 1):
            t = {}
            for e, c in s.items():
                i = 0
                while e + 2 * j * i <= K:
                    t[e + 2 * j * i] = t.get(e + 2 * j * i, 0) + c
                    i += 1
            s = t
        return s
    dim = len(matter_weights(D, slots[0])[0])
    prod = {(0, (0,) * dim, (0,) * M): 1}
    for i, lam in enumerate(slots):
        for w in matter_weights(D, lam):               # E_q(x) = Σ_n (-q)^n/(q^2;q^2)_n x^n, x = μ_i v^w
            E = {}
            for n in range(K + 1):
                for qe, c in inv_poch(n).items():
                    if qe + n <= K:
                        E[(qe + n, tuple(x * n for x in w), tuple(n if j == i else 0 for j in range(M)))] = c * (-1) ** n
            prod = mul(prod, E)
    recon = {}
    for ((m, e), k), h in G.rg_generator(K).items():
        if any(m):
            continue
        lp = h.expand(K)
        for wt, mlp in levi_character(D, (0,) * len(m), tuple(e)).terms.items():
            mult = mlp._coeffs.get(0, 0)
            for qe, c in lp._coeffs.items():
                if qe <= K and c:
                    key = (qe, tuple(wt), tuple(k))
                    recon[key] = recon.get(key, 0) + c * mult
    recon = {k: v for k, v in recon.items() if v}
    diff = sorted(set(recon) ^ set(prod) | {k for k in set(recon) & set(prod) if recon[k] != prod[k]})
    return recon == prod, len(prod), ("" if not diff else str([(k, recon.get(k), prod.get(k)) for k in diff[:3]]))


def _mr_R_checks(NAT, G, mono, wil, budget):
    """R2/R3 verifiers on the identity, a Wilson line and the minimal monopole, and R4 against the (G,N) tier on their
    9 pairs at K = 4, within `budget` seconds.  R4's flavour dictionary: the same label tuples when the matter irreps are
    distinct (GNAbeKAlgebra.flow_iso), the Cartan expansion U(n) -> U(1)^n when they are n copies of one irrep; a mixed
    theory has no shipped dictionary, so R4 is not compared there.  Returns ({step: True/False/'not reached'}, detail)."""
    import signal
    from zplus_ring import un_to_cartan_hom
    P = G.pure() if callable(G.pure) else G.pure
    z = (0,) * len(mono)
    zk = (0,) * G.M
    labs = [G.identity(), (P.fold(z, wil), zk), (P.fold(mono, z), zk)]
    triv = NAT.coefficient_ring().one_basis()
    nat = [NAT.identity(), NAT.fold(z, wil, triv), NAT.fold(mono, z, triv)]
    groups = NAT.groups
    if all(len(ix) == 1 for _l, ix in groups):
        expand = None
    elif len(groups) == 1:
        expand = un_to_cartan_hom(len(groups[0][1]))
    else:
        expand = "none"

    class _Out(Exception):
        pass

    def _alarm(*_):
        raise _Out()
    res, detail, t0 = {}, "", time.time()
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(budget)
    try:
        steps = (("unital", lambda: G.verify_rg_unital()),
                 ("bar-invariant", lambda: all(G.verify_rg_bar_invariant(a) for a in labs)),
                 ("multiplicative", lambda: all(G.verify_rg_multiplicative(a, b) for a in labs for b in labs)),
                 ("RG_a S = L + O(q)", lambda: all(G.verify_rg_discovery(a, K=2) for a in labs)),
                 ("twist", lambda: all(G.verify_rg_trg_intertwine(a) for a in labs)))
        for nm, fn in steps:
            res[nm] = bool(fn())
        if expand == "none":
            res["R4"] = "not compared (mixed flavour: no shipped dictionary)"
        else:
            def canon(ps):
                if expand is not None and ps.ring is not None and type(ps.ring).__name__ == "UNZPlusRing":
                    ps = expand.apply_RPowerSeries(ps)
                return {n: dict(c.terms) for n, c in ps.coeffs.items() if c.terms}
            bad = [(i, j) for i in range(3) for j in range(3)
                   if canon(G.inner_product(labs[i], labs[j], 4)) != canon(NAT.inner_product(nat[i], nat[j], 4))]
            res["R4"] = not bad
            detail = f"R4 differs on pairs {bad}" if bad else ""
    except _Out:
        for nm in ("unital", "bar-invariant", "multiplicative", "RG_a S = L + O(q)", "twist", "R4"):
            res.setdefault(nm, f"not reached in {budget} s")
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    return res, detail, round(time.time() - t0, 1)


def _mr_theory(environment, name):
    """One lineup theory: the S check always; the RG verifiers and R4 at rank one in the fast tier and on every theory
    at the extensive depth.  (checks, population entry) — the shape checks_coulomb._per_theory expects."""
    from checks_coulomb import battery_depth
    t0 = time.time()
    NAT, G, D, mono, wil = _mr_build(name)
    fast_R = next(r[5] for r in _MR_LINEUP if r[0] == name)
    depth = battery_depth()
    checks = []
    ok, nmono, det = _mr_S_check(NAT, G, D, K=6)
    checks.append(check(f"{name}: S_RG equals the product of E_q(μ_i v^w) over the matter weights, through q^6 ({nmono} monomials)", ok, det))
    entry = {"S monomials": nmono}
    if depth == "extensive" or fast_R:
        res, detail, secs = _mr_R_checks(NAT, G, mono, wil, MR_BUDGET[depth if depth in MR_BUDGET else "fast"])
        done = {k: v for k, v in res.items() if isinstance(v, bool)}
        if done:
            checks.append(check(f"{name}: RG verifiers ({', '.join(k for k in done if k != 'R4')}) on the identity, a Wilson line and the minimal monopole"
                                + ("; R4 against the (G,N) tier on their 9 pairs at K = 4" if "R4" in done else ""), all(done.values()), detail or str(res)))
        entry["RG checks"] = {k: v for k, v in res.items()}
        entry["RG checks seconds"] = secs
    entry["seconds"] = round(time.time() - t0, 1)
    return checks, entry


def _mr_tower(n, misplace=False):
    """The removal tower (SU(2), n doublets) -> ... -> pure, one hypermultiplet per rung (matter_removal_tower), composed by
    hand and compared with removing all n at once.  Rung i's IR presents its identical slots natively with U(r) flavour;
    rung i+1's UV splits them as U(r-1) x U(1), which for r <= 2 is the Cartan expansion used here -- hence n <= 3.
    `misplace` files each rung's removed level under the wrong slot (the negative control)."""
    from g_matter_over_matter import GMatterOverMatter, matter_removal_tower
    from g_matter_over_pure import _normalize_matter
    from root_datum import su_2
    from zplus_ring import un_to_cartan_hom
    assert n <= 3
    D0 = su_2()
    slots = _normalize_matter(D0, (((1,), n),))
    tower = matter_removal_tower(D0, (1,), nf=n)          # rung i drops slot n-1-i
    direct = GMatterOverMatter(D0, keep=(), drop=slots)
    P = direct.ir()

    def lp(c):
        return LaurentPoly({0: int(c)})

    def run(i, uv_label, levels, coeff, out):
        dropped = n - 1 - i
        for (ir_lab, kd), c in tower[i].RG(uv_label).terms.items():
            lv = dict(levels)
            lv[(dropped + 1) % n if misplace else dropped] = kd[0]
            cc = coeff * c
            if i == n - 1:
                key = (ir_lab, tuple(lv.get(s, 0) for s in range(n)))
                out[key] = out.get(key, LaurentPoly.zero()) + cc
                continue
            g, w = ir_lab
            nxt, kept = tower[i + 1], dropped - 1
            for wt, mult in un_to_cartan_hom(dropped).apply_basis(w).terms.items():
                nlab = ((nxt.ir().fold(g[0], g[1], tuple(wt[:kept])) if kept else nxt.ir().fold(*g)), (wt[kept],))
                run(i + 1, nlab, lv, cc * lp(mult), out)
    cases = [((0,), (0,)), ((0,), (1,)), ((1,), (0,)), ((1,), (1,)), ((1,), (-1,)), ((2,), (0,))]
    irreps = {1: [()], 2: [(0,), (1,)], 3: [(0, 0), (1, 0), (1, 1)]}[n]
    agree = disagree = 0
    t0 = time.time()
    ir0 = tower[0].ir()
    for (m, e) in cases:
        for w in irreps:
            for kl in (0, 1):
                got = {}
                run(0, ((ir0.fold(m, e, w) if w else ir0.fold(m, e)), (kl,)), {}, LaurentPoly.one(), got)
                got = {k: v for k, v in got.items() if not v.is_zero()}
                want = {}
                for wt, mult in (un_to_cartan_hom(n - 1).apply_basis(w).terms.items() if w else [((), 1)]):
                    for k, v in direct.RG((P.fold(m, e), tuple(wt) + (kl,))).terms.items():
                        want[k] = want.get(k, LaurentPoly.zero()) + v * lp(mult)
                want = {k: v for k, v in want.items() if not v.is_zero()}
                same = set(want) == set(got) and all((want[x] - got[x]).is_zero() for x in want)
                agree += same
                disagree += not same
    return agree, disagree, len(cases) * len(irreps) * 2, round(time.time() - t0, 1)


def check_matter_removal(environment):
    from checks_coulomb import _per_theory, battery_depth
    t0 = time.time()
    base = _mr_su2_nf1_hand(environment)
    checks = list(base["checks"])
    per = {}
    for row in _MR_LINEUP:
        cks, entry = _per_theory(_mr_theory, environment, row[0])
        checks.extend(cks)
        per[row[0]] = entry
    towers = {}
    for n in (2, 3):
        a, d, c, s = _mr_tower(n)
        towers[f"SU(2)+{n}"] = f"{a}/{c} agree, {s} s"
        checks.append(check(f"the draft's 'this RG flow factors': the tower SU(2)+{n} -> ... -> pure, one hypermultiplet per rung, composed by hand "
                            f"through the flavour dictionary, equals removing all {n} at once on {c} UV lines (6 lines x U({n - 1}) irreps x 2 levels): {a}/{c}", d == 0 and a == c))
    a, d, c, s = _mr_tower(2, misplace=True)
    checks.append(check(f"negative control: with each rung's removed level filed under the wrong slot, the SU(2)+2 tower disagrees with the direct removal on {d} of {c} lines", d > 0))
    return {"checks": checks,
            "population": {"SU(2)+1, the 2026-09-22 checks": base["population"], "lineup theories": per,
                           "not reachable (GMatterOverPure takes no line lattice)": list(_MR_UNREACHED), "towers": towers,
                           "depth": battery_depth(), "seconds": round(time.time() - t0, 1)},
            "controls": {"positive": "the hand-coded SU(2)+1 series; the Weyl symmetry of the product",
                         "negative": "the tower with the removed levels misfiled disagrees"},
            "notes": ("S is checked on every reachable lineup theory; the RG verifiers and R4 at rank one (fast) or on every theory within "
                      f"{MR_BUDGET} s (extensive).  R4 is compared through GNAbeKAlgebra.flow_iso's dictionary: identical label tuples for distinct "
                      "irreps, the Cartan expansion for n copies of one irrep; the mixed U(1)-U(2)+3 has no shipped dictionary."), "inputs": []}


EXTENSIVE = {"def:rg/matter-removal", "conj:upper", "conj:factored", "sec:rg/flavoured-quivers"}


# --------------------------------------------------------------------------
# the Seiberg-Witten conjectures on a sample of the shipped enumerated tier
# --------------------------------------------------------------------------
def sample_entries(n_max=4, limit=120):
    # The shipped tier is shard format 3: it STORES the strongly
    # connected quivers only (every other quiver is answered by composing them,
    # dictionary_loader.lookup_enumerated), so this sample is of strongly connected
    # entries — the irreducible ones.  Under format 2 it was the first cyclic entries.
    from dictionary_loader import iter_enumerated_entries
    out = []
    for e in iter_enumerated_entries():
        if len(e["exchange"]) <= n_max:
            out.append(e)
        if len(out) >= limit:
            break
    return out


def _manifest():
    """The shipped enumerated tier's manifest and a one-line description of it (read, never hard-coded: README section 3)."""
    import json as _json
    m = _json.load(open(os.path.join(ROOT, "dictionaries", "enumerated", "manifest.json"), encoding="utf-8"))
    line = (f"dictionaries/enumerated (shard format {m.get('format')}, E = {m.get('E')}: {m.get('n_entries')} strongly connected quivers, "
            f"{m.get('n_green')} with a green spec, {m.get('n_crystalline_only')} crystalline-only, {m.get('n_uncertified')} uncertified)")
    return m, line


def _node_coords(nodes, g):
    """`g` as an integer combination of the node charges (exact elimination over Q); None when it is not one."""
    from fractions import Fraction
    n, dim = len(nodes), len(g)
    M = [[Fraction(nodes[j][i]) for j in range(n)] + [Fraction(g[i])] for i in range(dim)]
    piv, r = [], 0
    for c in range(n):
        p = next((i for i in range(r, dim) if M[i][c] != 0), None)
        if p is None:
            continue
        M[r], M[p] = M[p], M[r]
        pv = M[r][c]
        M[r] = [x / pv for x in M[r]]
        for i in range(dim):
            if i != r and M[i][c] != 0:
                f = M[i][c]
                M[i] = [x - f * y for x, y in zip(M[i], M[r])]
        piv.append(c)
        r += 1
    if any(M[i][n] != 0 for i in range(r, dim)):
        return None
    x = [Fraction(0)] * n
    for i, c in enumerate(piv):
        x[c] = M[i][n]
    return tuple(int(v) for v in x) if all(v.denominator == 1 for v in x) else None


def _lineup_quivers():
    """The Section 4 lineup as {name: (node-basis exchange matrix, [its known specs in node coordinates])}:
    the charts of _wc_charts (the pentagon's two specs merged) and the Markov quiver, which has no maximal green sequence."""
    out = {}
    for name, A in _wc_charts():
        nodes = [tuple(v) for v in A.node_charges]
        B = [[A.lattice.bracket(u, v) for v in nodes] for u in nodes]
        spec = [_node_coords(nodes, tuple(g)) for g in A.spec]
        assert all(c is not None for c in spec), name
        key = "pentagon" if name.startswith("pentagon") else name
        if key in out:
            out[key][1].append(spec)
        else:
            out[key] = (B, [spec])
    out["Markov"] = ([[0, 2, -2], [-2, 0, 2], [2, -2, 0]], [])
    return out


def _run_controls(module, fns):
    """Run a control module's functions quietly; (verdict per function, passed checks, all checks) from its `_results`."""
    import contextlib
    import io
    module._results.clear()
    with contextlib.redirect_stdout(io.StringIO()):
        verdict = {}
        for fn in fns:
            try:
                verdict[fn.__name__] = bool(fn())
            except Exception as ex:  # noqa: BLE001
                verdict[fn.__name__] = False
                module._results.append((fn.__name__, False, f"{type(ex).__name__}: {ex}"))
    return verdict, sum(1 for _n, o, _d in module._results if o), len(module._results)


N_SAMPLE = 120   # first entries of the shipped tier with n <= 4, in cell order


def check_sw_sample(environment):
    """conj:cluster (and def:sw's acceptance census): the green specs satisfy the quiver constraint and equal S.  Two
    arithmetics: the repo's acceptance check (spec_acceptance.accept_spec) on the shipped tier's sample, and the independent
    one of a probe in the source repository on the same sample and on the lineup's specs, including different green sequences of
    one quiver (the pentagon's two chambers).  S-unique and PBW have their own rows since 2026-09-27."""
    import random
    import experiments.sw_flow_conjectures as swf
    from spec_acceptance import accept_spec, is_green_sequence
    checks = []
    t0 = time.time()
    depth = 4
    m, mline = _manifest()
    verdict, n_ok, n_all = _run_controls(swf, (swf.control_pentagon_S, swf.control_negative_wrong_spec))
    checks.append(check(f"controls: the pentagon's S from the independent arithmetic, both pentagon chambers satisfy the constraint, and wrong specs (reversed, repeated, missing, spurious, a doubled node factor) are caught: {n_ok}/{n_all}",
                        all(verdict.values()) and n_ok == n_all, str({k: v for k, v in verdict.items() if not v})))
    entries = sample_entries(4, N_SAMPLE)
    census = {"entries": len(entries), "with_spec": 0, "no_spec": 0, "green": 0, "accepted": 0, "leading_ok": 0, "matches_crystalline": 0,
              "indep_leading_ok": 0, "indep_equal_S": 0}
    bad = []
    rng = random.Random(20260927)
    for e in entries:
        B, spec = e["exchange"], e["spec"]
        if spec is None:
            census["no_spec"] += 1
            continue
        census["with_spec"] += 1
        acc = accept_spec(B, spec, depth)
        census["accepted"] += bool(acc.ok)
        census["leading_ok"] += bool(acc.leading_ok)
        census["matches_crystalline"] += bool(acc.matches_crystalline)
        census["green"] += bool(is_green_sequence(B, spec))
        o = swf.sweep_entry(str(B), B, [spec], rng)
        census["indep_leading_ok"] += o["specs_leading_ok"]
        census["indep_equal_S"] += o["specs_equal_S_ok"]
        if not acc.ok or not o["specs_equal_S_ok"]:
            bad.append((B, spec, acc.violations[:1], o["note"]))
    Ns = census["with_spec"]
    checks.append(check(f"def:sw / eq:quiver  on the first {len(entries)} entries with n <= 4 of {mline}: every stored spec is accepted at depth {depth} "
                        f"(green replay, the finite-depth constraint, agreement with the crystalline S): {census['accepted']}/{Ns}; stored without a spec: {census['no_spec']}",
                        census["accepted"] == Ns and Ns > 0, str(bad[:2])))
    checks.append(check(f"conj:cluster  the same specs by the independent arithmetic: S_cluster satisfies the constraint {census['indep_leading_ok']}/{Ns} and equals S {census['indep_equal_S']}/{Ns}; "
                        f"green replay {census['green']}/{Ns}", census["indep_leading_ok"] == Ns and census["indep_equal_S"] == Ns and census["green"] == Ns and Ns > 0))
    lineup = {}
    for name, (B, specs) in _lineup_quivers().items():
        if not specs:
            continue
        o = swf.sweep_entry(name, B, specs, rng)
        lineup[name] = (o["specs_tested"], o["specs_leading_ok"], o["specs_equal_S_ok"], o["specs_nontrivial"])
    ok_l = all(t == lo == eq and t > 0 for (t, lo, eq, _nt) in lineup.values())
    checks.append(check("conj:cluster  the lineup's specs (the pentagon's two chambers give one S): " + "; ".join(f"{n} {eq}/{t} equal to S" + (f" ({nt} with a non-node charge)" if nt else "") for n, (t, lo, eq, nt) in lineup.items()),
                        ok_l, str(lineup)))
    return {"checks": checks, "population": {"sample": f"first {len(entries)} entries with n <= 4 of {mline}, cell order", "lineup": list(lineup), "depth": depth,
                                             "seconds": round(time.time() - t0, 1)},
            "controls": {"positive": "the pentagon's two chambers", "negative": "five wrong pentagon specs and a doubled node factor are caught"},
            "notes": f"Web-tier sample; the manifest certifies every entry ({m.get('n_green')} green, {m.get('n_crystalline_only')} crystalline-only, axiom mismatches {len(m.get('axiom_mismatches') or [])}).  The rich build's spec families (several green sequences per quiver) are the local machine's.",
            "inputs": [{"path": "dictionaries/enumerated/manifest.json", "tracked": True}]}


def check_s_unique(environment):
    """conj:S-unique through its computable half, order independence, on the independent arithmetic: S is built by the
    forced recursion (a probe in the source repository) in five total orders -- degree-lex, anti-degree (every top
    degree factor before any node factor: no central charge produces it), lex-reversed, spin-first, a shuffled order of the
    truncated cone times spins -- and must be one element.  The recursion imposes the constraint by construction; that the
    same element comes out of every order is the measurement (pronilpotent_group_conjecture.md section 6c)."""
    import random
    import experiments.sw_flow_conjectures as swf
    checks = []
    t0 = time.time()
    m, mline = _manifest()
    verdict, n_ok, n_all = _run_controls(swf, (swf.control_expansion, swf.control_forced_omega, swf.control_pentagon_S, swf.control_cross_check_repo))
    checks.append(check(f"controls of the independent arithmetic: the E_q expansion, the forced Omega, the pentagon's S, agreement with the repo engine on four quivers: {n_ok}/{n_all}",
                        all(verdict.values()) and n_ok == n_all, str({k: v for k, v in verdict.items() if not v})))
    rng = random.Random(20260927)
    lineup = {name: swf.sweep_entry(name, B, [], rng) for name, (B, _specs) in _lineup_quivers().items()}
    checks.append(check("the Section 4 lineup: S from the five orders is one element on every quiver, the Markov quiver (no maximal green sequence) included: "
                        + ", ".join(f"{n} {o['orders_agree']}/{o['orders']} at cone depth {o['D']}" for n, o in lineup.items()),
                        all(o["unique_ok"] for o in lineup.values())))
    checks.append(check("vacuity guard: the number of factors moves with the order on every lineup quiver (" + "; ".join(f"{n} {o['n_factors']}" for n, o in lineup.items()) + ")",
                        all(len(set(o["n_factors"])) > 1 for o in lineup.values())))
    entries = sample_entries(4, N_SAMPLE)
    agree, bad = 0, []
    for e in entries:
        o = swf.sweep_entry(str(e["exchange"]), e["exchange"], [], rng)
        agree += bool(o["unique_ok"])
        if not o["unique_ok"]:
            bad.append((e["exchange"], o["note"]))
    checks.append(check(f"the first {len(entries)} entries with n <= 4 of {mline}: S from the five orders is one element on {agree}/{len(entries)}",
                        agree == len(entries), str(bad[:2])))
    return {"checks": checks, "population": {"lineup": list(lineup), "sample": f"first {len(entries)} entries with n <= 4 of {mline}", "seconds": round(time.time() - t0, 1)},
            "controls": {"positive": "the pentagon's S; agreement with the repo engine", "negative": "the order axis moves the factor count (vacuity guard)"},
            "notes": "Order independence is the computable half of uniqueness: two orders giving different elements would refute it; agreement does not prove it.  Prior sweep: 1108/1108 quivers of the seed-closure dictionary n_001..n_008 (2026-09-04, pronilpotent_group_conjecture.md 6c).",
            "inputs": [{"path": "dictionaries/enumerated/manifest.json", "tracked": True}]}


def check_pbw(environment):
    """conj:pbw on the independent arithmetic (a probe in the source repository): elements of E other than S -- random words in
    the generators E^(s)(X_gamma)^(+-1) -- are refactorised by the read-off (pbw_factorize) in five total orders and must give
    integral exponents and rebuild the element exactly; S itself, built in one order, is refactorised in the other four.
    The read-off imposes nothing: a (1 - q^(2n)) denominator or a non-palindromic residual raises.  The repo engine's
    integral Omega is manufactured by construction (the audit) and is not evidence here."""
    import random
    import experiments.pbw_controls as pc
    import experiments.pbw_e_group as pe
    from experiments.sw_flow_conjectures import DEGREE_FOR_RANK, nodes_of
    checks = []
    t0 = time.time()
    verdict, n_ok, n_all = _run_controls(pc, (pc.control_linear_coefficient, pc.control_single_generator, pc.control_bar_unitary, pc.control_round_trip,
                                               pc.control_negative_Eq2, pc.control_negative_perturbation, pc.control_order_independence, pc.control_cross_check_repo))
    checks.append(check(f"controls (a probe in the source repository): {n_ok}/{n_all}: generators refactorise to themselves, bar(g) = g^-1, round trips exact; "
                        "E_(q^2)(X), bar-unitary but outside E, must not factorise and does not; a q^0 bump fails; the engine's own S reproduced",
                        all(verdict.values()) and n_ok == n_all, str({k: v for k, v in verdict.items() if not v})))
    rng = random.Random(20260927)
    words_per = 3
    per, moved = {}, {}
    for name, (B, _specs) in _lineup_quivers().items():
        rank = len(B)
        D = DEGREE_FOR_RANK.get(rank, 4)
        keys = [pe.key_degree_lex, pe.key_anti_degree, pe.key_lex_reversed, pe.key_spin_first, pe.make_key_shuffled(rank, D, rng, 3)]
        ok_w = 0
        for _ in range(words_per):
            g, _word = pc.random_word(B, D, rng)
            counts, ok = [], True
            for k in keys:
                try:
                    fac, _om = pe.pbw_factorize(g, B, D, k)
                    ok &= pe.elt_eq(pe.ordered_product(fac, B, D, rank), g, D)
                    counts.append(len(fac))
                except pe.PBWFailure:
                    ok = False
            ok_w += ok
            moved[name] = moved.get(name, 0) + (len(set(counts)) > 1)
        S0, _f = pe.build_S_forced(B, nodes_of(rank), D, pe.key_degree_lex)
        ok_S = True
        for k in keys[1:]:
            try:
                fac, _om = pe.pbw_factorize(S0, B, D, k)
                ok_S &= pe.elt_eq(pe.ordered_product(fac, B, D, rank), S0, D)
            except pe.PBWFailure:
                ok_S = False
        per[name] = (ok_w, words_per, ok_S, D)
    checks.append(check(f"random elements of E ({words_per} words per lineup quiver) refactorise with integral exponents and rebuild exactly in all five orders: "
                        + ", ".join(f"{n} {w}/{t} (depth {D})" for n, (w, t, _s, D) in per.items()),
                        all(w == t for (w, t, _s, _D) in per.values())))
    checks.append(check("S itself, built in degree-lex, refactorises integrally and exactly in the other four orders on every lineup quiver: "
                        + ", ".join(f"{n} {'yes' if s_ else 'NO'}" for n, (_w, _t, s_, _D) in per.items()), all(s_ for (_w, _t, s_, _D) in per.values())))
    checks.append(check("vacuity guard: on every lineup quiver the number of factors moves with the order on some word (a word whose generators commute "
                        "factorises the same way in every order): " + ", ".join(f"{n} {moved.get(n, 0)}/{words_per}" for n in per),
                        all(moved.get(n, 0) > 0 for n in per)))
    return {"checks": checks, "population": {"lineup": list(per), "words per quiver": words_per, "orders": 5, "seconds": round(time.time() - t0, 1)},
            "controls": {"positive": "single generators, round trips, the engine's S", "negative": "E_(q^2) does not factorise; a q^0 bump fails"},
            "notes": "Prior campaign on the same arithmetic: ranks 1-8, brackets -8..8, orders no central charge produces, spin content to 45.5, no counterexample (pronilpotent_group_conjecture.md section 6b).",
            "inputs": []}


# --------------------------------------------------------------------------
# batch 4: sec:rg/F-unique, conj:upper, def:rg/no-exotic
# --------------------------------------------------------------------------
_F_DEPTH = {2: 6, 3: 5, 4: 4, 5: 4}


def _F_independent(B, S, gamma, D):
    """F(gamma) by the triangular recursion, on the independent arithmetic of a probe in the source repository: F_gamma = 1, and at
    cone degree d, F_(gamma+delta) = minus the palindromic completion of the non-positive part of (F_(<d) S)_(gamma+delta).
    Forced at every step -- which is why existence and uniqueness of the formal solution are immediate -- and finite only if
    the recursion stops.  Returns (F, the last cone degree with a nonzero term)."""
    import experiments.pbw_e_group as pe
    n = len(B)
    F = {tuple(gamma): pe.CF_ONE}
    last = 0
    for d in range(1, D + 1):
        new = {}
        for delta in pe.cone_points(n, d):
            acc = pe.CF_ZERO
            for g1, v1 in F.items():
                rest = tuple(x - (a - b) for x, a, b in zip(delta, g1, gamma))
                if min(rest) < 0 or rest not in S:
                    continue
                acc = pe.cf_add(acc, pe.cf_mul(pe.cf_mul(v1, S[rest]), pe.cf_from_lp({pe.bracket(B, g1, rest): 1})))
            low = pe.cf_expand(acc, 0)
            pal = {}
            for e, v in low.items():
                if v:
                    pal[e] = pal.get(e, 0) - v
                    if e < 0:
                        pal[-e] = pal.get(-e, 0) - v
            pal = {e: v for e, v in pal.items() if v}
            if pal:
                new[tuple(g + x for g, x in zip(gamma, delta))] = pe.cf_from_lp(pal)
                last = d
        F.update(new)
    return F, last


def _lp_of_cf(cf):
    import experiments.pbw_e_group as pe
    return {e: v for e, v in (pe.cf_to_lp(cf) or {}).items() if v}


def _node_window(n):
    e = [tuple(1 if j == i else 0 for j in range(n)) for i in range(n)]
    return e + [tuple(a + b for a, b in zip(e[i], e[j])) for i in range(n) for j in range(i + 1, n)] + [tuple(-x for x in e[0])]


def _chart_of(name):
    charts = dict(_wc_charts())
    return charts.get(name) or next(A for nm, A in _wc_charts() if nm.startswith(name))


def _repo_F_node(A, g, D=None):
    """The repo's F(g) for a label g in node coordinates, keyed by node coordinates; with D, only cone degree <= D over g."""
    nodes = [tuple(v) for v in A.node_charges]
    n = len(nodes)
    lat = tuple(sum(g[i] * nodes[i][k] for i in range(n)) for k in range(len(nodes[0])))
    out = {}
    for k, v in A.F(lat).items():
        if v.is_zero():
            continue
        kc = _node_coords(nodes, tuple(k))
        if kc is None:
            out[("off the node lattice", tuple(k))] = {}
        elif D is None or sum(kc) - sum(g) <= D:
            out[kc] = {e: c for e, c in v._coeffs.items() if c}
    return out, lat


def check_f_unique(environment):
    """sec:rg/F-unique.  Given S, RG_gamma S = X_gamma + O(q) has a unique bar-invariant formal solution supported on
    gamma + Gamma_+: the recursion above is forced degree by degree (at the lowest degree where two solutions differ, their
    difference D satisfies (D S)_(gamma+delta) = D_(gamma+delta), bar-invariant and O(q), hence zero).  So the statement
    follows from the recursion; the check guards the repo's F-solver: it must equal the recursion, which shares no code with
    it, through the recursion's depth.  That the solution is FINITE is conj:upper's."""
    import experiments.pbw_e_group as pe
    checks = []
    t0 = time.time()
    per, bad = {}, []
    n_terms = n_multi = 0
    for name, (B, specs) in _lineup_quivers().items():
        if not specs:
            continue
        A = _chart_of(name)
        n = len(B)
        D = _F_DEPTH[n]
        S = pe.spec_product(list(specs[0]), B, D)
        agree = lead = 0
        win = _node_window(n)
        for g in win:
            Fi, _last = _F_independent(B, S, g, D)
            Fi = {k: _lp_of_cf(v) for k, v in Fi.items()}
            Fr, lat = _repo_F_node(A, g, D)
            if Fr == Fi:
                agree += 1
                n_terms += len(Fr)
                n_multi += len(Fr) > 1
            else:
                bad.append((name, g))
            lead += bool(A.verify_F_S_leading(lat, K=3))
        per[name] = (agree, lead, len(win), D)
    # the dictionary sample of conj:upper: eleven charts with a stored spec, n <= 3, node charges = the standard basis
    from dictionary_loader import entry_to_bpskalgebra
    samp_agree = samp_n = 0
    for e in [e for e in sample_entries(3, 60) if len(e["exchange"]) <= 3 and e.get("spec") is not None][:11]:
        A = entry_to_bpskalgebra(e)
        B = [list(r) for r in e["exchange"]]
        D = _F_DEPTH[len(B)]
        S = pe.spec_product([tuple(g) for g in e["spec"]], B, D)
        for g in _node_window(len(B)):
            Fi = {k: _lp_of_cf(v) for k, v in _F_independent(B, S, g, D)[0].items()}
            Fr = _repo_F_node(A, g, D)[0]
            samp_n += 1
            if Fr == Fi:
                samp_agree += 1
                n_terms += len(Fr)
            else:
                bad.append((str(e["exchange"]), g))
    per["dictionary sample (11 charts)"] = (samp_agree, samp_n, samp_n, "5-6")
    checks.append(check("the repo's F-solver equals the independent recursion through its depth on the lineup charts and the dictionary sample: "
                        + ", ".join(f"{n} {a}/{w} (depth {D})" for n, (a, _l, w, D) in per.items())
                        + f"; {n_terms} terms compared, {n_multi} lineup labels with more than one term", not bad and n_multi > 0, str(bad[:4])))
    lineup_per = {k: v for k, v in per.items() if not k.startswith("dictionary")}
    checks.append(check("regression guard (enforced, the audit): F S = X + O(q) on the lineup's labels: "
                        + ", ".join(f"{n} {l}/{w}" for n, (_a, l, w, _D) in lineup_per.items()), all(l == w for (_a, l, w, _D) in lineup_per.values())))
    # negative control: the recursion with the pentagon's S in the reversed (wrong) order gives a different F somewhere
    Bp = [[0, 1], [-1, 0]]
    Sw = pe.spec_product([(0, 1), (1, 0)], Bp, 6)
    Ap = _chart_of("pentagon")
    differ = [g for g in _node_window(2) if {k: _lp_of_cf(v) for k, v in _F_independent(Bp, Sw, g, 6)[0].items()} != _repo_F_node(Ap, g, 6)[0]]
    checks.append(check(f"negative control: with the pentagon's S taken in the reversed order the recursion's F differs from the solver's on {len(differ)} of {len(_node_window(2))} labels {differ}", len(differ) > 0))
    return {"checks": checks, "population": {"charts": {n: f"{w} labels, depth {D}" for n, (_a, _l, w, D) in per.items()}, "terms compared": n_terms, "seconds": round(time.time() - t0, 1)},
            "controls": {"positive": "the pentagon", "negative": "the reversed-order S changes the recursion's F"},
            "notes": "Derived: the recursion is forced degree by degree, so the formal solution exists and is unique; the check guards the F-solver.  Finiteness is conj:upper's.", "inputs": []}


def _laurent_on_paths(lattice, nodes, f, L):
    """f ({lattice charge: LaurentPoly}) followed along every mutation sequence of length <= L from the chart `nodes`: at
    each step conjugated by E_q(X_gamma_k) at the current chart's node gamma_k (lattice_mutation.solve, which raises exactly
    when the result is not a Laurent polynomial), the chart mutated as gamma_k -> -gamma_k, gamma_j -> gamma_j +
    max(<gamma_j, gamma_k>, 0) gamma_k; sequences that repeat a node twice in a row are skipped.  Returns (conjugations that
    stayed Laurent, conjugations tried, first failure, distinct sets of node charges reached).  A set of node charges is not a
    chart: mutating the pentagon at 1, 2, 1, 2 returns its node charges, but the accumulated conjugation is by the monodromy
    S rho(S), not a monomial change -- so the count bounds the charts from below, and each sequence's own transport is what
    is tested."""
    from lattice_mutation import _build_from_dict, solve
    frontier = [(tuple(tuple(v) for v in nodes), f, None)]
    ok = tried = 0
    fail = None
    charts = {frozenset(frontier[0][0])}
    for _ in range(L):
        nxt = []
        for ch, elt, last in frontier:
            for k, gk in enumerate(ch):
                if k == last:            # mutating twice at one node conjugates by a function of X_gk alone: a monomial change
                    continue
                tried += 1
                try:
                    out = solve(_build_from_dict(lattice, elt), gk)
                except ValueError:
                    fail = fail or (ch, gk)
                    continue
                ok += 1
                new = tuple(tuple(-x for x in gk) if j == k else tuple(a + max(lattice.bracket(ch[j], gk), 0) * b for a, b in zip(ch[j], gk))
                            for j in range(len(ch)))
                charts.add(frozenset(new))
                nxt.append((new, {tuple(c): v for c, v in out._terms.items() if not v.is_zero()}, k))
        frontier = nxt
    return ok, tried, fail, len(charts)


_PATH_LEN = {2: 5, 3: 4, 4: 3, 5: 2}


_KAX_BUDGET = {"fast": 30, "extensive": 600}     # seconds per chart for the K_q-algebra axiom verifiers
_MARKOV = [[0, 2, -2], [-2, 0, 2], [2, -2, 0]]


def _chart_markov(depth=6):
    """The Markov quiver's chart, built as the dictionary builds a quiver with no accepted spec: spec-free, S from the
    crystalline factor engine at `depth`, rho through the tRG fallback (dictionary_loader.entry_to_bpskalgebra)."""
    from bps_kalgebra import BPSKAlgebra
    return BPSKAlgebra(pairing=_MARKOV, node_charges=[(1, 0, 0), (0, 1, 0), (0, 0, 1)], build_S=True, build_S_cutoff=depth,
                       extract_spec=False, spec_free_sigma="trg", verify="off")


def _upper_charts():
    """conj:upper's lineup: the charts of _wc_charts (the pentagon once) and the Markov quiver."""
    return [(nm, A) for nm, A in _wc_charts() if nm != "pentagon, 3-factor spec"] + [("Markov", _chart_markov())]


def _finite_confirmed(A, B, S, D, window):
    """For each label of `window` (node coordinates): the repo's F has top cone degree t over the label; where t < D the
    independent recursion must stop at t, i.e. have no term in the degrees t+1..D (the solver searches a window, so its
    F being finite is not evidence -- the audit -- while the recursion searches every degree through D).
    Returns ({label: number of empty degrees checked}, [labels with t >= D, not confirmed], [labels where they disagree])."""
    conf, beyond, bad = {}, [], []
    for g in window:
        Fr, _lat = _repo_F_node(A, g)
        top = max(sum(k) - sum(g) for k in Fr if isinstance(k[0], int)) if Fr else 0
        if top >= D:
            beyond.append(g)
            continue
        _Fi, last = _F_independent(B, S, g, D)
        if last == top:
            conf[g] = D - top
        else:
            bad.append((g, top, last))
    return conf, beyond, bad


def _kax_on_chart(A, budget):
    """The draft's K_q-algebra axioms, and RG multiplicativity, on the labels 1 and the first three nodes of chart A, the
    cheapest first, within `budget` seconds (SIGALRM): {family: True / False / 'not reached in B s'}.  Once the budget is
    spent nothing further runs on the chart, so no verifier sees state an interrupted one left behind."""
    import signal
    nodes = [tuple(v) for v in A.node_charges]
    labs = [tuple(0 for _ in nodes[0])] + nodes[:3]
    pairs = [(a, b) for a in labs for b in labs]
    pairs3 = [(a, b) for a in labs[:3] for b in labs[:3]]
    fams = (("ax:bar", lambda: all(A.verify_bar_involution(a, b) for a, b in pairs)),
            ("ax:rho", lambda: all(A.verify_rho_is_automorphism(a, b) for a, b in pairs)),
            ("RG multiplicative", lambda: all(A.verify_rg_multiplicative(a, b) for a, b in pairs)),
            ("ax:rhotr", lambda: all(A.verify_trace_intertwines_rho(a, K=4) for a in labs)),
            ("ax:trace orthonormality", lambda: all(A.verify_orthonormality(a, b, K=4) for a, b in pairs)),
            ("ax:trace rho^2-twisted cyclicity", lambda: all(A.verify_rho_twisted_trace(a, b, K=3) for a, b in pairs3)))

    class _Out(Exception):
        pass

    def _alarm(*_):
        raise _Out()
    res = {}
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(budget)
    try:
        for nm, fn in fams:
            res[nm] = bool(fn())
    except _Out:
        for nm, _fn in fams:
            res.setdefault(nm, f"not reached in {budget} s")
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    return res


def check_upper_sample(environment):
    """conj:upper.  (i) Finiteness: the repo's F, confirmed by the independent recursion (no term between its top degree and
    the recursion's depth), on the lineup (the Markov quiver included) and on the dictionary sample.  (ii) Decisions D7,
    first half: every RG_gamma of a label window stays a Laurent polynomial in every chart reached by a mutation sequence
    of bounded length (lattice_mutation.solve is exact and raises on a non-Laurent result); the negative control is the
    node monomials and the truncated F's.  (iii) The Z[q^+-1]-subalgebra and the K_q-algebra axioms (ax:bar, ax:basis
    as closure, ax:rho, ax:trace, ax:rhotr) on the lineup, the expensive verifiers within a per-chart budget, and on the
    dictionary sample.  (v) Decisions D7, second half: on a box of charges, the elements Laurent in the root chart and
    its immediate neighbours are exactly the span of the RG_gamma supported in the box
    (a probe in the source repository) -- so the upper cluster algebra meets the box inside that span, and
    equals it there given (ii)."""
    import experiments.pbw_e_group as pe
    from checks_coulomb import battery_depth
    from dictionary_loader import entry_to_bpskalgebra
    depth = battery_depth()
    checks = []
    t0 = time.time()
    lineup = _lineup_quivers()
    charts = _upper_charts()
    # (i) finiteness, confirmed by the recursion
    fin, fin_beyond, fin_bad = {}, {}, []
    for name, A in charts:
        B, specs = lineup[name]
        n = len(B)
        D = _F_DEPTH[n] + (1 if name == "Markov" else 0)
        S = pe.spec_product(list(specs[0]), B, D) if specs else pe.build_S_forced(B, [tuple(1 if j == i else 0 for j in range(n)) for i in range(n)], D, pe.key_degree_lex)[0]
        window = _node_window(n) if name != "Markov" else _node_window(n)[:n]
        conf, beyond, bad = _finite_confirmed(A, B, S, D, window)
        fin[name] = (conf, D, len(window))
        if beyond:
            fin_beyond[name] = beyond
        fin_bad += [(name,) + b for b in bad]
    checks.append(check("finiteness on the lineup: where the repo's F ends below the recursion's depth the independent recursion ends there too -- "
                        + "; ".join(f"{nm} {len(c)}/{w} labels (depth {D}, at least {min(c.values()) if c else 0} empty degrees)" for nm, (c, D, w) in fin.items())
                        + (f"; F reaching the depth, not confirmed: {fin_beyond}" if fin_beyond else ""), not fin_bad and all(fin[nm][0] for nm in fin), str(fin_bad[:4])))
    # (ii) D7, first half: Laurent in every chart along mutation sequences of bounded length
    lau, neg_mono, neg_trunc, lau_bad = {}, {}, {}, []
    for name, A in charts:
        nodes = [tuple(v) for v in A.node_charges]
        n = len(nodes)
        L = _PATH_LEN[n] + (1 if depth == "extensive" else 0)
        window = _node_window(n) if name != "Markov" else _node_window(n)[:n]
        tot_ok = tot = 0
        reached = 0
        mono_fail = trunc_fail = trunc_n = 0
        for g in window:
            f, lat = {}, tuple(sum(g[i] * nodes[i][k] for i in range(n)) for k in range(len(nodes[0])))
            f = {tuple(c): v for c, v in A.F(lat).items() if not v.is_zero()}
            ok, tried, fail, nch = _laurent_on_paths(A.lattice, nodes, f, L)
            tot_ok += ok
            tot += tried
            reached = max(reached, nch)
            if fail:
                lau_bad.append((name, g, fail))
            if len(f) > 1:                      # negative control: F with its top-degree terms dropped
                degs = {c: sum(_node_coords(nodes, c)) for c in f}
                top = max(degs.values())
                trunc = {c: v for c, v in f.items() if degs[c] < top}
                trunc_n += 1
                trunc_fail += _laurent_on_paths(A.lattice, nodes, trunc, L)[2] is not None
        for gj in nodes:                        # negative control: the bare node monomials
            mono_fail += _laurent_on_paths(A.lattice, nodes, {gj: LaurentPoly({0: 1})}, 1)[2] is not None
        lau[name] = (tot_ok, tot, L, reached)
        neg_mono[name] = f"{mono_fail}/{n}"
        neg_trunc[name] = f"{trunc_fail}/{trunc_n}"
    checks.append(check("the design record, first half: RG_gamma stays a Laurent polynomial after every mutation along every sequence of bounded length "
                        "(no node twice in a row), on node charges, pairwise sums and minus the first node -- "
                        + "; ".join(f"{nm} {o}/{t} conjugations, length <= {L}, {r} distinct sets of node charges" for nm, (o, t, L, r) in lau.items()),
                        not lau_bad and all(o == t > 0 for (o, t, _L, _r) in lau.values()), str(lau_bad[:3])))
    checks.append(check("negative controls: a bare node monomial X_gamma_j fails after one mutation (" + ", ".join(f"{k} {v}" for k, v in neg_mono.items())
                        + "), and F with its top-degree terms dropped fails on some sequence (" + ", ".join(f"{k} {v}" for k, v in neg_trunc.items()) + ")",
                        all(int(v.split("/")[0]) > 0 for v in neg_mono.values()) and all(int(v.split("/")[0]) == int(v.split("/")[1]) for v in neg_trunc.values())))
    # (iii) closure and the K_q-algebra axioms on the lineup (not Markov: no finite rho on the spec-free route)
    kax = {name: _kax_on_chart(A, _KAX_BUDGET[depth]) for name, A in charts if name != "Markov"}
    fam_names = list(next(iter(kax.values())))
    for fam in fam_names:
        reached = {nm: r[fam] for nm, r in kax.items() if r[fam] is True or r[fam] is False}
        unreached = [nm for nm, r in kax.items() if nm not in reached]
        guard = " (regression guard: inherited on RGKAlgebra, the audit)" if fam == "ax:rhotr" else ""
        checks.append(check(f"{fam}{guard} on the labels 1 and three nodes: holds on " + ", ".join(nm for nm, v in reached.items() if v)
                            + (f"; not reached within {_KAX_BUDGET[depth]} s on " + ", ".join(unreached) if unreached else ""),
                            all(reached.values()) and bool(reached)))
    # (iv) the dictionary sample: finiteness confirmed, closure, ax:bar and ax:rho on node products
    pool = [e for e in sample_entries(3, 60) if len(e["exchange"]) <= 3][:12]
    skipped = [e["exchange"] for e in pool if e.get("spec") is None]
    entries = [e for e in pool if e.get("spec") is not None]
    samp_conf = samp_beyond = 0
    samp_bad, detail = [], ""
    ok_closure = ok_k13 = True
    for e in entries:
        A = entry_to_bpskalgebra(e)
        B = [list(r) for r in e["exchange"]]
        n = len(B)
        D = _F_DEPTH[n]
        S = pe.spec_product([tuple(g) for g in e["spec"]], B, D)
        conf, beyond, bad = _finite_confirmed(A, B, S, D, _node_window(n))
        samp_conf += len(conf)
        samp_beyond += len(beyond)
        samp_bad += bad
        nodes = [tuple(1 if i == j else 0 for j in range(n)) for i in range(n)]
        for a in nodes:
            for b in nodes:
                try:
                    el = A.multiply(a, b)
                    ok_closure &= all(isinstance(cf, LaurentPoly) for cf in el.terms.values()) and A.verify_rg_multiplicative(a, b)
                    ok_k13 &= A.verify_bar_involution(a, b) and A.verify_rho_is_automorphism(a, b)
                except Exception as ex:  # noqa: BLE001
                    ok_closure = False
                    detail = detail or f"{e['exchange']} product {a}·{b}: {type(ex).__name__}: {str(ex)[:80]}"
    checks.append(check(f"the dictionary sample ({len(entries)} charts, n <= 3): finiteness confirmed by the recursion on {samp_conf} labels "
                        f"({samp_beyond} with F reaching the depth); node products close over Z[q^+-1], are RG-multiplicative, and satisfy ax:bar and ax:rho",
                        not samp_bad and ok_closure and ok_k13, detail or str(samp_bad[:3])))
    # (v) D7, second half: the box certificate
    import experiments.upper_cluster_box_certificate as ub
    box_res, box_bad = [], []
    for name, A in charts:
        nodes = [tuple(v) for v in A.node_charges]
        n = len(nodes)
        for R in ((1, 2) if depth == "extensive" and n <= 3 else (1,)):
            if name == "Markov":
                sdepth = 10 if R == 2 else 6
                Ab, Adeep = _chart_markov(sdepth), _chart_markov(sdepth + 2)
            else:
                Ab, Adeep = A, None
            box = ub._box(nodes, R)
            F = ub._F_of(Ab)
            inside, other, rk, bad = ub.box_certificate(F, Ab.lattice, nodes, box)
            unstable = [g for g in inside if F(g) != ub._F_of(Adeep)(g)] if Adeep is not None else []
            box_res.append((name, R, len(box), len(inside), len(other), rk))
            if rk != len(other) or bad or unstable:
                box_bad.append((name, R, rk, len(other), bad[:2], unstable[:2]))
    checks.append(check("the design record, second half: on a box of charges the elements Laurent in the root chart and its immediate neighbours are exactly "
                        "the span of the RG_gamma supported in the box (each such RG_gamma survives one mutation at every node; the conditions on the "
                        "other box points' monomials have full rank at q = 2) -- "
                        + "; ".join(f"{nm} [-{R},{R}]^n: {nb} monomials, {ni} RG inside, {no} others at rank {rk}" for nm, R, nb, ni, no, rk in box_res),
                        not box_bad and bool(box_res), str(box_bad[:2])))
    Ap = next(A for nm, A in charts if nm == "pentagon")
    pn = [tuple(v) for v in Ap.node_charges]
    pbox = ub._box(pn, 1)
    Fp = ub._F_of(Ap)
    p_in, p_other, p_rk, _pb = ub.box_certificate(Fp, Ap.lattice, pn, pbox)
    rk0 = ub.neighbour_rank(Ap.lattice, pn, p_other + [(0, 0)])
    from lattice_mutation import _build_from_dict as _bfd, solve as _solve
    cut = fail = 0
    for g in p_in:
        f = Fp(g)
        if len(f) < 2:
            continue
        top = max(sum(k) for k in f)
        trunc = {k: v for k, v in f.items() if sum(k) < top}
        cut += 1
        for gk in pn:
            try:
                _solve(_bfd(Ap.lattice, trunc), gk)
            except ValueError:
                fail += 1
                break
    checks.append(check(f"controls for the box test (pentagon, [-1,1]^2): X_0 = 1, Laurent in every chart, added to the other points drops the rank "
                        f"({rk0} < {len(p_other) + 1}); RG_gamma inside the box with the top-degree terms dropped fail one mutation on {fail} of {cut}",
                        p_rk == len(p_other) and rk0 < len(p_other) + 1 and fail > 0))
    pop = {"lineup": [nm for nm, _A in charts], "labels": "node charges, pairwise sums and minus the first node (Markov: the nodes)",
           "boxes": [f"{nm} [-{R},{R}]^n" for nm, R, *_ in box_res],
           "mutation sequences": {nm: f"length <= {L}, {r} distinct sets of node charges reached" for nm, (_o, _t, L, r) in lau.items()},
           "K_q axioms": {nm: {k: v for k, v in r.items()} for nm, r in kax.items()},
           "dictionary sample": f"{len(entries)} charts; skipped (no stored spec, so no finite rho on the BPS route): {skipped}",
           "seconds": round(time.time() - t0, 1)}
    return {"checks": checks, "population": pop,
            "controls": {"positive": "F·S = X + O(q) on every label (sec:rg/F-unique)", "negative": "bare node monomials and truncated F fail the mutation test"},
            "notes": "The box test uses the root chart's immediate neighbours only, so it bounds the upper cluster algebra on the box without enumerating charts; the Markov quiver's chart has no finite spec and is built deeper for the larger box, every F inside the box unchanged two degrees deeper still.",
            "inputs": [{"path": "dictionaries/enumerated/manifest.json", "tracked": True}]}


def _qn_signs(qn):
    """(has a negative [n]_q coefficient, has a q-number beyond [1]_q) for a QNumberPoly."""
    items = list(qn.items())
    return any(c < 0 for _n, c in items), any(n > 1 for n, c in items if c)


def check_no_exotic(environment):
    """def:rg/no-exotic (the draft's footnote to ax:map; not imposed): the coefficients of RG_a are non-negative
    combinations of q-numbers [n]_q.  Census over the lineup's BPS charts (the F of a label window, in the solver's native
    q-number form), the dictionary sample, and the matter-removal flows (RG images of a Wilson line, the minimal monopole
    and every label in the expansions of their products, each coefficient decomposed into q-numbers).  The census function is first shown
    to flag a planted negative, and the census records how many coefficients carry a q-number beyond [1]_q (a census of
    1's alone would say nothing)."""
    from q_number_poly import QNumberPoly
    from dictionary_loader import entry_to_bpskalgebra
    checks = []
    t0 = time.time()
    planted = QNumberPoly.from_palindromic_laurent(LaurentPoly({-2: 1, 2: 1}))       # q^-2 + q^2 = [3]_q - [1]_q
    ok_ctrl = _qn_signs(planted)[0] and not _qn_signs(QNumberPoly.from_palindromic_laurent(LaurentPoly({-1: 1, 1: 1})))[0]
    checks.append(check(f"positive control: the census flags q^-2 + q^2 = [3]_q - [1]_q ({dict(planted.items())}) and passes q^-1 + q = [2]_q", ok_ctrl))
    census = {}

    def add(key, qn):
        neg, big = _qn_signs(qn)
        c = census.setdefault(key, [0, 0, 0])
        c[0] += 1
        c[1] += neg
        c[2] += big
    for name, A in _upper_charts():
        nodes = [tuple(v) for v in A.node_charges]
        n = len(nodes)
        window = _node_window(n) if name != "Markov" else _node_window(n)[:n]
        for g in window:
            lat = tuple(sum(g[i] * nodes[i][k] for i in range(n)) for k in range(len(nodes[0])))
            for qn in A.F_qn(lat).values():
                if not qn.is_zero():
                    add(name, qn)
    for e in [e for e in sample_entries(3, 60) if len(e["exchange"]) <= 3 and e.get("spec") is not None][:11]:
        A = entry_to_bpskalgebra(e)
        n = len(e["exchange"])
        for g in _node_window(n):
            for qn in A.F_qn(g).values():
                if not qn.is_zero():
                    add("dictionary sample", qn)
    mr_detail = {}
    for name in ("SU(2)+1", "SU(2)+2", "SU(2)+adjoint (N=2*)"):
        _NAT, G, _D, mono, wil = _mr_build(name)
        P = G.pure() if callable(G.pure) else G.pure
        z = (0,) * len(mono)
        zk = (0,) * G.M
        labs = [(P.fold(z, wil), zk), (P.fold(mono, z), zk)]
        # the statement is about RG_a of canonical basis elements (a product's structure constants are not bar-invariant),
        # so the labels are the Wilson line, the minimal monopole and every label in the expansions of their products
        seen = list(labs)
        for a in labs:
            for b in labs:
                seen += [c for c in G.multiply(a, b).terms if c not in seen]
        for a in seen:
            for cf in G.RG(a).terms.values():
                if not cf.is_zero():
                    add(f"matter removal {name}", QNumberPoly.from_palindromic_laurent(cf))
        mr_detail[name] = len(seen)
    tot = sum(c[0] for c in census.values())
    neg = sum(c[1] for c in census.values())
    big = sum(c[2] for c in census.values())
    checks.append(check(f"census: {neg} coefficients with a negative q-number coefficient among {tot} ({big} carry a q-number beyond [1]_q): "
                        + "; ".join(f"{k} {c[1]}/{c[0]} ({c[2]} beyond [1])" for k, c in census.items()), neg == 0 and big > 0, str({k: c for k, c in census.items() if c[1]})))
    return {"checks": checks, "population": {"coefficients": {k: c[0] for k, c in census.items()}, "matter-removal labels": mr_detail, "seconds": round(time.time() - t0, 1)},
            "controls": {"positive": "q^-2 + q^2 = [3]_q - [1]_q is flagged", "negative": "n.a."},
            "notes": "A negative coefficient would be a recorded counterexample to a statement the draft does not impose.", "inputs": [{"path": "dictionaries/enumerated/manifest.json", "tracked": True}]}


_FACT_DEPTH = {2: 6, 3: 5, 4: 4, 5: 4}
_FACT_BUDGET = {"fast": 30, "extensive": 600}     # seconds per directional flow for the class-level certificate


def _factored_entry(B, sub, D):
    """One (Q, Q') on the independent arithmetic (a probe in the source repository): (F) S_Q built with the charges inside Q'
    last factors as R S_Q'; (I) R = S_Q S_Q'^-1 decomposes in the RG^Q' basis; (C) every RG^Q at a node decomposes there over
    Z[q^+-1].  Added here: the ring of the draft's ax:S on S_Q^Q' = that decomposition -- every coefficient in
    Z[q, (1-q^2n)^-1], O(q) off the identity -- and the number of RG^Q' elements the peel used that carry a negative power
    of q, since only there does the ring condition have terms to cancel.  Returns (the experiment's dict, ring violations,
    RG elements with negative powers, RG elements used)."""
    import experiments.pbw_e_group as pe
    import experiments.sw_factored_flows as sf
    n = len(B)
    r = sf.factored_flow_entry(B, sub, D)
    SQ, _f = pe.build_S_forced(B, sf.nodes_of(n), D, sf.key_sub_last(sub))
    Ssub = sf.sub_S_embedded(B, sub, D)
    R = pe.elt_mul(SQ, pe.elt_inv(Ssub, B, D, n), B, D)
    cache = {}
    dec = pe.decompose_in_RG(R, Ssub, B, D, cache)
    if dec is None:
        return r, None, 0, len(cache)
    bad = [(e, min(v[0]) if v[0] else None) for e, v in dec.items() if not v[0] or min(v[0]) < (1 if any(e) else 0)]
    neg = 0
    for rg in cache.values():
        lps = [pe.cf_to_lp(v) for v in rg.values() if not pe.cf_is_zero(v)]
        neg += any(lp and min(k for k, x in lp.items() if x) < 0 for lp in lps)
    return r, bad, neg, len(cache)


def _directional_keys(trace_K):
    """The families _directional_checks reports, cheapest first; the traced ones only when trace_K is set."""
    keys = ["RG factorises", "S_RG = the independent decomposition", "flow axioms", "UV = the Q-chart"]
    if trace_K is not None:
        keys += [f"flow axioms with traces to q^{trace_K}", f"UV = the Q-chart with traces to q^{trace_K}"]
    return keys


def _directional_checks(A, drop, budget, trace_K):
    """The class-level checks of one node deletion on a lineup chart: DirectionalSubquiverRG built (spec arranged so that the
    factors carrying the dropped node come first); RG^Q = RG^Q' o RG^(Q,Q') exactly on the node charges and pairwise sums;
    certify_directional_vs_bps (the flow's UV, derived from the IR chart and S_RG alone, against the independently built
    Q-chart); verify_axioms on the flow; the flow's S_RG (a peel of vacuum kets) against the operator-level decomposition of
    S_Q S_Q'^-1 on the independent arithmetic, as exact q-series through q^8.  Within `budget` seconds (SIGALRM)."""
    import signal
    import experiments.pbw_e_group as pe
    import experiments.sw_factored_flows as sf
    from directional_subquiver_rg import DirectionalSubquiverRG, arrange_spec_stuff_first, certify_directional_vs_bps, verify_axioms
    nodes = [tuple(v) for v in A.node_charges]
    n = len(nodes)
    pairing = [list(r) for r in A.lattice.pairing]
    # the class's own arranger caps the spec at len + 6 factors; [A_1,D_4] dropping its centre needs 11, so allow len + 10
    spec = arrange_spec_stuff_first(pairing, nodes, [tuple(g) for g in A.spec], drop, max_len=len(A.spec) + 10)
    if spec is None:
        return None, "no spec with the dropped node's factors first is reachable by local moves"
    dr = DirectionalSubquiverRG(pairing, nodes, spec, drop)

    class _Out(Exception):
        pass

    def _alarm(*_):
        raise _Out()
    res = {}
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(budget)
    try:
        ir = dr.auxiliary()
        win = nodes + [tuple(a + b for a, b in zip(nodes[i], nodes[j])) for i in range(n) for j in range(i + 1, n)]
        ok = 0
        for a in win:
            comp = {}
            for b, cb in dr.RG(a).terms.items():
                for k, v in ir.F(b).items():
                    comp[tuple(k)] = comp.get(tuple(k), LaurentPoly.zero()) + cb * v
            ok += {k: v for k, v in comp.items() if not v.is_zero()} == {tuple(k): v for k, v in A.F(a).items() if not v.is_zero()}
        res["RG factorises"] = (ok, len(win))
        B = [[A.lattice.bracket(u, v) for v in nodes] for u in nodes]
        D = _FACT_DEPTH[n]
        sub = set(range(n)) - set(drop)
        SQ = pe.spec_product([_node_coords(nodes, tuple(g)) for g in dr.spec], B, D)
        Ssub = sf.sub_S_embedded(B, sub, D)
        dec = pe.decompose_in_RG(pe.elt_mul(SQ, pe.elt_inv(Ssub, B, D, n), B, D), Ssub, B, D, {})
        ind = {e: {k: x for k, x in pe.cf_expand(v, 8).items() if x} for e, v in dec.items()}
        cls = {}
        for lab, h in dr.rg_generator(D).items():
            e = _node_coords(nodes, tuple(lab))
            if e is not None and pe.deg(e) <= D:
                cls[e] = {k: x for k, x in h.expand(8)._coeffs.items() if x}
        ind = {e: v for e, v in ind.items() if v}
        cls = {e: v for e, v in cls.items() if v}
        res["S_RG = the independent decomposition"] = (sum(ind.get(e) == cls.get(e) for e in set(ind) | set(cls)), len(set(ind) | set(cls)))
        labs = [dr.identity()] + nodes[:2]
        pairs = [(a, b) for a in nodes[:2] for b in nodes[:2]]
        # the untraced checks first (the fast run's), so that traced additions exhausting the budget never hide them
        ax = verify_axioms(dr, labs, pairs, trace_K=None, deep_rho=False)
        res["flow axioms"] = all(ax.values())
        cert = certify_directional_vs_bps(dr, A, trace_K=None)
        res["UV = the Q-chart"] = all(v for k, v in cert.items() if k != "iso")
        if trace_K is not None:
            ax = verify_axioms(dr, labs, pairs, trace_K=trace_K, deep_rho=False)
            res[f"flow axioms with traces to q^{trace_K}"] = all(ax.values())
            cert = certify_directional_vs_bps(dr, A, trace_K=trace_K)
            res[f"UV = the Q-chart with traces to q^{trace_K}"] = all(v for k, v in cert.items() if k != "iso")
    except _Out:
        for nm in _directional_keys(trace_K):
            res.setdefault(nm, f"not reached in {budget} s")
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    return dr, res


def check_factored(environment):
    """conj:factored: S_Q = RG^(Q',Gamma)(S_Q^Q') S_Q' for every full subquiver Q', and RG^(Q,Gamma) = RG^(Q',Gamma) o
    RG^(Q,Q',Gamma), a flow of K_q-algebras.  (A) On the independent arithmetic, every full subquiver of every lineup quiver
    (the Markov quiver included) and of the dictionary sample: the factorisation, the decomposition of R in the RG^Q' basis
    with the ring of ax:S, and RG^Q factoring over Z[q^+-1].  (B) With DirectionalSubquiverRG, every node deletion of every
    lineup chart that the class can build: RG factorising exactly, S_RG against (A), the flow's axioms, and its derived UV
    algebra against the independently built Q-chart."""
    import experiments.pbw_e_group as pe
    import experiments.sw_factored_flows as sf
    from checks_coulomb import battery_depth
    depth = battery_depth()
    checks = []
    t0 = time.time()
    verdict, passed, total = _run_controls(sf, [sf.control_degenerate_subquivers, sf.control_pentagon, sf.control_negative_wrong_sub_S])
    checks.append(check(f"the experiment's controls: Q' = Q and Q' = empty are forced, the pentagon by hand, a perturbed S_Q' breaks the factorisation, "
                        f"the wrong sub-quiver's RG basis leaves non-Laurent coefficients ({passed}/{total})", all(verdict.values()) and passed == total, str(verdict)))
    # (A) the independent arithmetic
    per, bad, n_pairs, n_neg_pairs = {}, [], 0, 0
    quivers = [(nm, B) for nm, (B, _specs) in _lineup_quivers().items()]
    quivers += [(f"sample {e['exchange']}", [list(r) for r in e["exchange"]]) for e in sample_entries(3, 60) if len(e["exchange"]) <= 3][:11]
    for name, B in quivers:
        n = len(B)
        D = _FACT_DEPTH[n]
        cnt = [0, 0, 0, 0]
        for k in range(1, n):
            for sub in itertools.combinations(range(n), k):
                r, ring_bad, neg, _used = _factored_entry(B, set(sub), D)
                n_pairs += 1
                cnt[0] += 1
                cnt[1] += bool(r["F_ok"])
                cnt[2] += ring_bad == [] and r["image_n"] > 0
                cnt[3] += r["closure_laurent"] == r["closure_n"]
                n_neg_pairs += neg > 0
                if not (r["F_ok"] and ring_bad == [] and r["closure_laurent"] == r["closure_n"]):
                    bad.append((name, sub, r, (ring_bad or [])[:2]))
        per[name] = cnt
    lineup_names = [nm for nm, _B in quivers if not nm.startswith("sample")]
    samp = [v for nm, v in per.items() if nm.startswith("sample")]
    checks.append(check("(A) on every full subquiver Q': S_Q with the charges of Q' last is R S_Q' with R the product of the other factors; R is "
                        "RG^Q'(S_Q^Q') with S_Q^Q' in 1 + q Z[q, (1-q^2n)^-1] (the ring of ax:S); RG^Q at every node decomposes in the RG^Q' basis over "
                        "Z[q^+-1] -- " + "; ".join(f"{nm} {per[nm][1]}/{per[nm][0]}" for nm in lineup_names)
                        + f"; the dictionary sample {sum(v[1] for v in samp)}/{sum(v[0] for v in samp)} pairs over {len(samp)} charts; "
                        f"on {n_neg_pairs} of {n_pairs} pairs the RG^Q' elements used carry negative powers of q that the ring condition needs cancelled",
                        not bad and n_neg_pairs > 0, str(bad[:2])))
    # negative control for the ring: the pentagon's S in the wrong order
    Bp = [[0, 1], [-1, 0]]
    Sw = pe.spec_product([(0, 1), (1, 0)], Bp, 6)
    Ssub = sf.sub_S_embedded(Bp, {1}, 6)
    dw = pe.decompose_in_RG(pe.elt_mul(Sw, pe.elt_inv(Ssub, Bp, 6, 2), Bp, 6), Ssub, Bp, 6, {})
    wrong = [(e, min(v[0])) for e, v in dw.items() if v[0] and min(v[0]) < (1 if any(e) else 0)]
    checks.append(check(f"negative control: with the pentagon's S in the wrong order, S_Q^Q' leaves the ring of ax:S ({len(wrong)} coefficients, e.g. {wrong[:2]})", len(wrong) > 0))
    # (B) the class
    cls, unbuilt, cls_bad = {}, [], []
    for name, A in _wc_charts():
        if name == "pentagon, 3-factor spec":
            continue
        for d in range(len(A.node_charges)):
            dr, res = _directional_checks(A, [d], _FACT_BUDGET[depth], 3 if depth == "extensive" else None)
            if dr is None:
                unbuilt.append(f"{name} dropping node {d}")
                continue
            cls[f"{name} dropping node {d}"] = res
            for k, v in res.items():
                if v is False or (isinstance(v, tuple) and v[0] != v[1]):
                    cls_bad.append((name, d, k, v))
    reached = lambda k: {nm: r[k] for nm, r in cls.items() if not isinstance(r[k], str)}   # noqa: E731
    for key in _directional_keys(3 if depth == "extensive" else None):
        got = reached(key)
        miss = [nm for nm in cls if nm not in got]
        checks.append(check(f"(B) {key}: holds on {len(got)} node deletions" + (f"; not reached within {_FACT_BUDGET[depth]} s on {miss}" if miss else ""),
                            bool(got) and all((v is True) or (isinstance(v, tuple) and v[0] == v[1]) for v in got.values()),
                            str([b for b in cls_bad if b[2] == key][:2])))
    # negative control for (B): pentagon images mapped through the IR chart of the OTHER node deletion
    Ap = dict(_wc_charts())["pentagon"]
    from directional_subquiver_rg import DirectionalSubquiverRG
    d0 = DirectionalSubquiverRG([list(r) for r in Ap.lattice.pairing], [tuple(v) for v in Ap.node_charges], [tuple(g) for g in Ap.spec], [0], arrange=True)
    d1 = DirectionalSubquiverRG([list(r) for r in Ap.lattice.pairing], [tuple(v) for v in Ap.node_charges], [tuple(g) for g in Ap.spec], [1], arrange=True)
    mism = 0
    for a in [tuple(v) for v in Ap.node_charges] + [(1, 1)]:
        comp = {}
        for b, cb in d0.RG(a).terms.items():
            for k, v in d1.auxiliary().F(b).items():
                comp[tuple(k)] = comp.get(tuple(k), LaurentPoly.zero()) + cb * v
        mism += {k: v for k, v in comp.items() if not v.is_zero()} != {tuple(k): v for k, v in Ap.F(a).items() if not v.is_zero()}
    checks.append(check(f"negative control: the pentagon's flow dropping node 0, mapped through the IR chart of the flow dropping node 1, fails to reproduce RG^Q on {mism} of 3 labels", mism > 0))
    return {"checks": checks,
            "population": {"(A) pairs": n_pairs, "(A) quivers": [nm for nm, _B in quivers], "(B) node deletions": cls,
                           "(B) not buildable": unbuilt, "seconds": round(time.time() - t0, 1)},
            "controls": {"positive": "Q' = Q and Q' = empty; the pentagon by hand", "negative": "a perturbed S_Q'; the wrong sub-quiver's basis; the wrong-order S; the wrong IR chart"},
            "notes": "Where no spec with the dropped node's factors first is reachable, the class does not build and only (A) covers the pair.",
            "inputs": [{"path": "dictionaries/enumerated/manifest.json", "tracked": True}]}


# --------------------------------------------------------------------------
# def:wallcrossing — a single mutation of a BPS chart is a wall-crossing relation
# --------------------------------------------------------------------------
def _wc_charts():
    """The Section 4 lineup's BPS charts that carry a spec: (name, BPSKAlgebra)."""
    from bps_kalgebra import BPSKAlgebra
    from bps_su2_nf1 import build_bps_su2_nf1
    from bps_su2_nf2 import build_bps_su2_nf2
    from bps_su2_nf3 import build_bps_su2_nf3
    from pure_ade_lattice import pure_ade_kalgebra
    e = lambda n, i: tuple(1 if j == i else 0 for j in range(n))   # noqa: E731
    d4 = [[0, -1, -1, -1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]   # leaves 1,2,3 -> centre 0: the D_4 quiver of [A_1,D_4]
    return [
        ("pentagon", BPSKAlgebra(pairing=[[0, 1], [-1, 0]], node_charges=[(1, 0), (0, 1)], spec=[(1, 0), (0, 1)], verify="off")),
        ("pentagon, 3-factor spec", BPSKAlgebra(pairing=[[0, 1], [-1, 0]], node_charges=[(1, 0), (0, 1)], spec=[(0, 1), (1, 1), (1, 0)], verify="off")),
        ("A_3", BPSKAlgebra(pairing=[[0, 1, 0], [-1, 0, 1], [0, -1, 0]], node_charges=[e(3, i) for i in range(3)], spec=[e(3, i) for i in range(3)], verify="off")),
        ("[A_1,D_4]", BPSKAlgebra(pairing=d4, node_charges=[e(4, i) for i in range(4)], spec=[e(4, 1), e(4, 2), e(4, 3), e(4, 0)], verify="off")),
        ("Kronecker (pure SU(2))", pure_ade_kalgebra([("A", 1)])),
        ("SU(2)+1", build_bps_su2_nf1()),
        ("SU(2)+2", build_bps_su2_nf2()),
        ("SU(2)+3", build_bps_su2_nf3()),
    ]


def _wc_window(A):
    nodes = [tuple(v) for v in A.node_charges]
    n = A.lattice.rank
    add = lambda u, v, s=1: tuple(x + s * y for x, y in zip(u, v))   # noqa: E731
    win = [tuple([0] * n)] + nodes + [add(u, u) for u in nodes]
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i < j:
                win.append(add(nodes[i], nodes[j]))
            if i != j:
                win.append(add(nodes[i], nodes[j], -1))
    return list(dict.fromkeys(win))


def check_wallcrossing(environment):
    """def:wallcrossing on BPS charts.  A single mutation at the head γ_k of a chart's spec factors S^(11̄) = S^(12) S^(2̄1)
    with S^(12) = E_q(X_γk); the mutated chart's S must be S^(2̄1) ρ_IR^-1(S^(12)) = S^(2̄1) E_q(X_-γk) (the spec rotated),
    and RG^(1)_a S^(12) = S^(12) RG^(2)_a, i.e. conjugating chart 1's F by E_q(X_γk) (lattice_mutation.solve, which never
    expands E_q) gives the mutated chart's F, the latter rebuilt from scratch so its F-solver is independent of the
    atlas's transport.  The other relation and R3 of the new flow follow from these and R3 of the old one."""
    from bps_atlas import BPSAtlas
    from bps_kalgebra import BPSKAlgebra
    from lattice_mutation import _build_from_dict, solve
    checks = []
    t0 = time.time()
    summary = {}
    neg = {}
    for name, A1 in _wc_charts():
        atlas = BPSAtlas(A1)
        k2, _iso = atlas.mutate_head(())
        A2a = atlas.chart(k2)
        gk = tuple(A1.spec[0])
        rot = [tuple(g) for g in A1.spec[1:]] + [tuple(-x for x in gk)]
        spec_ok = [tuple(g) for g in A2a.spec] == rot
        A2 = BPSKAlgebra(pairing=[list(r) for r in A1.lattice.pairing], node_charges=[tuple(v) for v in A2a.node_charges],
                         spec=[tuple(g) for g in A2a.spec], verify="off")
        window = _wc_window(A1)
        agree = 0
        bad = []
        wrong_node = next(tuple(v) for v in A1.node_charges if tuple(v) != gk and A1.lattice.bracket(tuple(v), gk) != 0)
        wrong_detected = 0
        for a in window:
            (b,) = list(atlas.transport(a, (), k2).terms)
            F2 = {tuple(k): v for k, v in A2.F(b).items() if not v.is_zero()}
            O1 = _build_from_dict(A1.lattice, A1.F(a))
            got = {tuple(k): v for k, v in solve(O1, gk)._terms.items() if not v.is_zero()}
            if got == F2:
                agree += 1
            else:
                bad.append((a, b))
            try:                                   # negative control: conjugation by another node's E_q
                w = {tuple(k): v for k, v in solve(O1, wrong_node)._terms.items() if not v.is_zero()}
                wrong_detected += (w != F2)
            except ValueError:
                wrong_detected += 1
        summary[name] = f"head {gk}; spec rotated {spec_ok}; {agree}/{len(window)} labels"
        neg[name] = f"{wrong_detected}/{len(window)} labels differ when conjugating by {wrong_node}"
        checks.append(check(f"{name}: mutation at the spec head {gk} is a wall-crossing relation — the mutated chart's spec is the old one with its head moved to the end and negated, "
                            f"and conjugating F by E_q(X_head) gives the mutated chart's F (rebuilt independently) on {agree}/{len(window)} labels",
                            spec_ok and not bad, str(bad[:3])))
    checks.append(check("negative control: conjugating by E_q of another node (paired with the head) instead of the head's changes the image on some label of every chart",
                        all(int(v.split("/")[0]) > 0 for v in neg.values()), str(neg)))
    # the draft's note: partial factors need not be 1 + O(q).  Pure SU(2) at weak coupling places the W boson, Ω = χ_1/2, i.e. the
    # factor E^(1/2)_q(X_W) = E_q(-q^-1 X)^-1 E_q(-q X)^-1 of eq:mult, whose X-coefficient is -(1+q^2)/(1-q^2) = -1 - 2q^2 - ...
    import cmath
    from bps_factor_spectrum import BPSFactorSpectrum
    eng = BPSFactorSpectrum([[0, 2], [-2, 0]], [(1, 0), (0, 1)], 4, phases=[cmath.rect(1, 2.5), cmath.rect(1, 0.3)])
    eng.run()
    om = {tuple(g): dict(d) for g, d in eng.multiplicities().items()}
    w_ok = om.get((1, 1)) == {-1: 1, 1: 1}
    # X-coefficient of E^(1/2)(x) = Σ_(j=±1/2) [x]E_q(-q^(2j) x)^-1, with [y]E_q(y)^-1 = q/(1-q^2): as a q-series through q^8
    K = 8
    coeff = (S.q(0, K + 4, -1) + S.q(2, K + 4, -1)) * inv_one_minus_1(2, K + 4)
    lead = min(e for e, v in coeff.trunc(K).c.items() if v)
    checks.append(check(f"the draft's note, witnessed: pure SU(2) at a weak-coupling central charge carries the W boson (Ω(γ1+γ2) = χ_1/2: {om.get((1, 1))}), "
                        f"and its factor E^(1/2)_q(X_(γ1+γ2)), a partial factor of S, has X-coefficient -(1+q^2)/(1-q^2) = {coeff.trunc(6)}, of valuation {lead}: not 1 + O(q)",
                        w_ok and lead == 0, str(om)))
    return {"checks": checks, "population": {"charts": summary, "seconds": round(time.time() - t0, 1)},
            "controls": {"positive": "the pentagon (both specs)", "negative": "conjugation by another node's factor"},
            "notes": "RG^(2)_a S^(2̄1) = S^(2̄1) ρ^-1(RG^(1)_ρ(a)) and R3 of the new flow follow from R3 of the old one and the two relations checked.", "inputs": []}


def check_central_charge(environment):
    from bps_kalgebra import BPSKAlgebra
    from bps_atlas import BPSAtlas
    checks = []
    t0 = time.time()
    results = {}
    for name, B, nodes in (("pentagon (A_2)", [[0, 1], [-1, 0]], [(1, 0), (0, 1)]), ("A_3", [[0, 1, 0], [-1, 0, 1], [0, -1, 0]], [(1, 0, 0), (0, 1, 0), (0, 0, 1)])):
        try:
            A = BPSKAlgebra(B, nodes)
            atlas = BPSAtlas(A)
            mono = atlas.monodromy()
            ok = True
            for a in nodes:
                img = mono.map(Element.basis(a))
                rr = A.rho(A.rho(a))
                ok &= (img == Element.basis(rr))
            results[name] = "monodromy = rho^2 on the node labels" if ok else f"differs: {[(a, mono.map(Element.basis(a)).terms, A.rho(A.rho(a))) for a in nodes]}"
        except Exception as ex:  # noqa: BLE001
            results[name] = f"{type(ex).__name__}: {str(ex)[:100]}"
    checks.append(check("def:rg/central-charge  the full rotation of the central charge (the chart-graph monodromy) is rho^2 on the node labels", all(v.startswith("monodromy") for v in results.values()), str(results)))
    # [A_1,D_4] (finite type, so the rotation loop closes) — added 2026-09-26 with the Section 4 lineup
    d4 = dict(_wc_charts())["[A_1,D_4]"]
    try:
        mono = BPSAtlas(d4).monodromy()
        nd = [tuple(v) for v in d4.node_charges]
        ok_d4 = all(mono.map(Element.basis(a)) == Element.basis(d4.rho(d4.rho(a))) for a in nd)
        results["[A_1,D_4]"] = "monodromy = rho^2 on the node labels" if ok_d4 else "differs"
    except Exception as ex:  # noqa: BLE001
        ok_d4 = False
        results["[A_1,D_4]"] = f"{type(ex).__name__}: {str(ex)[:100]}"
    checks.append(check("[A_1,D_4]: the chart-graph monodromy is rho^2 on the node labels", ok_d4, results["[A_1,D_4]"]))
    # the locally constant family: a generic linear central charge orders the factors by phase; in each chamber the factor engine with
    # that order must reproduce the chart's S.  Two chambers per chart (the node phases in one order and the reverse).
    import cmath
    from bps_factor_spectrum import BPSFactorSpectrum
    depth = 4
    chambers = {}
    ok_ch = True
    for name, A in _wc_charts():
        if name not in ("pentagon", "A_3", "Kronecker (pure SU(2))", "SU(2)+1"):
            continue
        nodes = [tuple(v) for v in A.node_charges]
        P = [list(r) for r in A.lattice.pairing]
        n = len(nodes)
        args = [0.2 + 2.6 * i / max(n - 1, 1) for i in range(n)]
        per = []
        for arg_list in (args, list(reversed(args))):
            eng = BPSFactorSpectrum(P, nodes, depth, phases=[cmath.rect(1.0 + 0.1 * i, a) for i, a in enumerate(arg_list)])
            eng.run()
            Sg = eng.spectrum_generator()
            same = all((A._s_coefficient(g) - c).is_zero() for g, c in Sg.items())
            states = sum(1 for d in eng.multiplicities().values() if any(d.values()))
            per.append(f"{states} factor charges, S equal {same}")
            ok_ch &= same and len(Sg) > 0
        chambers[name] = per
    checks.append(check(f"def:rg/central-charge  in two chambers of a generic linear central charge per chart, the factor engine ordered by phase reproduces the chart's S through cone depth {depth}",
                        ok_ch, str(chambers)))
    return {"checks": checks, "population": {"atlases (monodromy)": list(results), "chambers": chambers, "depth": depth, "seconds": round(time.time() - t0, 1)},
            "controls": {"positive": "the pentagon", "negative": "n.a."},
            "notes": "The rotation by pi (the ρ-family) is def:rg/rho-family's; the Kronecker quiver's rotation loop does not close (infinite type), so its monodromy is not computed by the chart graph.", "inputs": []}


_FLAV_TIERS = ((2, "SU(2)"), (3, "SU(3)"), (4, "SU(4)"), (5, "SU(5)"), (6, "SU(6)"), ((2, 2), "SU(2)xSU(2)"), ((3, 2), "SU(3)xSU(2)"),
               ((3, 3), "SU(3)xSU(3)"), ((4, 2), "SU(4)xSU(2)"), ((2, 2, 2), "SU(2)^3"))


def _flav_sample(tier, k, scan=3000):
    """Up to `k` entries of a flavoured tier: its stored entries at the smallest reduced rank among its first `scan`.

    The stored entries are the STRONGLY CONNECTED flavoured quivers, and several thin tiers store
    none at their weight: every quiver of theirs composes.  Such a tier's sample is topped up with COMPOSED answers.
    The quiver is a 3-cycle of arrow multiplicities (1,1,1), (2,1,1) or (1,2,1), with the tier's orbits attached as
    identical sinks to distinct cycle nodes: cyclic, not strongly connected, the family unique.
    `lookup_flavoured` composes it, and the checks below run on it as on a stored entry."""
    from dictionary_loader import iter_flavoured_entries, lookup_flavoured
    by_rank = {}
    for i_, e in enumerate(iter_flavoured_entries(orbit_size=tier)):
        by_rank.setdefault(len(e["reduced"]), []).append(e)
        if i_ >= scan:
            break
    out = []
    for r in sorted(by_rank):
        out += by_rank[r][:k - len(out)]
        if len(out) >= k:
            break
    sizes = [tier] if isinstance(tier, int) else list(tier)
    for mult in ((1, 1, 1), (2, 1, 1), (1, 2, 1)):
        if len(out) >= k:
            break
        n = 3 + sum(sizes)
        B = [[0] * n for _ in range(n)]
        for (i, j), m in zip(((0, 1), (1, 2), (2, 0)), mult):
            B[i][j], B[j][i] = m, -m
        nxt = 3
        for c, N in enumerate(sizes):
            for _ in range(N):
                B[c][nxt], B[nxt][c] = 1, -1
                nxt += 1
        got = lookup_flavoured(B, orbit_size=tier)
        if got is not None and got.get("spec") is not None:
            out.append(got)
    return out


def _flav_random_key(seed):
    import random
    rng = random.Random(seed)
    memo = {}

    def key(*args):
        if args not in memo:
            memo[args] = rng.random()
        return memo[args]
    return key


def check_flavoured_quivers(environment):
    """sec:rg/flavoured-quivers.  On samples of the ten shipped flavoured tiers: Omega(gamma, r) comes out a G_f-character
    at every charge (forced weight by weight, never imposed); the covariant spec replays, or where the tier stores only
    Omega a fresh run reproduces it; S equals the unfolded quiver's S from the unflavoured engine (which knows nothing of
    flavour); S does not depend on the order on (charge, spin, irrep) -- the PBW factorisation in the group generated by
    the flavoured dilogarithms -- while the order does move the factor content, and a weight-level order leaves the group
    (Omega stops being a character).  The flavoured K_q-algebra: GBPSKAlgebra against GfBPSKAlgebra, which share no code,
    at [A_1,D_3] with SU(2) and [A_1,D_4] with SU(3)."""
    import itertools as it
    from checks_coulomb import battery_depth
    from flavoured_factor_spectrum import (FlavouredQuiver, FlavouredFactorSpectrum, build_flavoured_spectrum_generator,
                                           unfolded_spectrum_generator)
    from flavoured_spec import is_flavoured_spec
    from habiro import HabiroElement
    depth = battery_depth()
    k = 12 if depth == "extensive" else 3
    H0 = HabiroElement.zero()
    same = lambda a, b: all(a.get(x, H0) == b.get(x, H0) for x in set(a) | set(b))   # noqa: E731
    checks = []
    t0 = time.time()
    rows, bad, moved, n_tot = {}, [], 0, 0
    for tier, tname in _FLAV_TIERS:
        cnt = {"entries": 0, "character": 0, "spec or stored Omega": 0, "unfolded": 0, "orders": 0, "constraint": 0}
        for e in _flav_sample(tier, k):
            Q = FlavouredQuiver(e["reduced"], e["factors"])
            D = 3
            eng = FlavouredFactorSpectrum(Q, D)
            eng.run()
            S = eng.spectrum_generator()
            cnt["entries"] += 1
            n_tot += 1
            ch = not eng.verify_omega_is_a_character()
            cnt["character"] += ch
            cnt["constraint"] += not eng.verify_leading_data()
            if "spec" in e:
                rep = is_flavoured_spec(Q, e["spec"], D)[0]
            else:
                stored = {tuple(kk): {tuple(tuple(p) for p in r): dict((int(x), int(y)) for x, y in poly) for r, poly in per} for kk, per in e["omega"]}
                fresh = {tuple(kk): {tuple(tuple(p) for p in r): {int(x): int(y) for x, y in poly.items() if y} for r, poly in per.items()}
                         for kk, per in eng.multiplicities().items()}
                rep = stored == fresh
            cnt["spec or stored Omega"] += rep
            unf = same(S, unfolded_spectrum_generator(Q, D)) if Q.unfoldable() else False
            cnt["unfolded"] += unf
            contents = {repr(sorted((kk, sorted(map(str, v.items()))) for kk, v in eng.multiplicities_by_spin().items()))}
            ordok = True
            for seed in range(2):
                e2 = FlavouredFactorSpectrum(Q, D, piece_key=_flav_random_key(seed))
                e2.run()
                ordok &= same(e2.spectrum_generator(), S) and not e2.verify_omega_is_a_character()
                contents.add(repr(sorted((kk, sorted(map(str, v.items()))) for kk, v in e2.multiplicities_by_spin().items())))
            cnt["orders"] += ordok
            moved += len(contents) > 1
            if not (ch and rep and unf and ordok):
                bad.append((tname, e["reduced"], e["factors"], dict(character=ch, replay=rep, unfolded=unf, orders=ordok)))
        rows[tname] = cnt
    checks.append(check("Omega(gamma, r) is a G_f-character at every charge (never imposed), and the stored covariant spec replays -- or, where the tier "
                        "stores only Omega, a fresh run reproduces it -- on " + "; ".join(f"{t} {c['character']}/{c['entries']}, {c['spec or stored Omega']}/{c['entries']}" for t, c in rows.items()),
                        all(c["character"] == c["spec or stored Omega"] == c["entries"] > 0 for c in rows.values()), str(bad[:2])))
    checks.append(check("S equals the S of the unfolded quiver, built by the unflavoured engine, on " + ", ".join(f"{t} {c['unfolded']}/{c['entries']}" for t, c in rows.items()),
                        all(c["unfolded"] == c["entries"] for c in rows.values())))
    checks.append(check(f"the PBW factorisation in the group generated by the flavoured dilogarithms: S is unchanged under two random orders on (charge, spin, irrep), "
                        f"with Omega a character in each, on {sum(c['orders'] for c in rows.values())}/{n_tot} entries; the order moves the factor content on {moved} of them",
                        all(c["orders"] == c["entries"] for c in rows.values()) and moved > 0))
    checks.append(check("regression guard (the engine imposes it): eq:flavouredquiver, S = 1 - q sum_i chi_fund_i X_gamma_i + O(q^2), on "
                        + ", ".join(f"{t} {c['constraint']}/{c['entries']}" for t, c in rows.items()), all(c["constraint"] == c["entries"] for c in rows.values())))
    # negative control: a weight-level order leaves the group
    Qk = FlavouredQuiver([[0, 2], [-2, 0]], [2, 1])
    base = build_flavoured_spectrum_generator(Qk, 4)
    broke = same_all = 0
    for seed in range(6):
        e3 = FlavouredFactorSpectrum(Qk, 4, piece_key=_flav_random_key(seed), split_weights=True)
        e3.run()
        same_all += same(e3.spectrum_generator(), base)
        broke += bool(e3.verify_omega_is_a_character())
    checks.append(check(f"negative control: ordering the individual weights (Kronecker-2 with SU(2) on one node) leaves S unchanged in {same_all}/6 orders "
                        f"but breaks the character property in {broke} of them", broke > 0 and same_all == 6))
    # the flavoured K_q-algebra: two independent presentations
    from gbps_kalgebra import GBPSKAlgebra
    from gf_bps_kalgebra import GfBPSKAlgebra
    agree = {}
    for factors, nm in (([1, 2], "[A_1,D_3] with SU(2)"), ([1, 3], "[A_1,D_4] with SU(3)")):
        fq = FlavouredQuiver([[0, 1], [-1, 0]], factors)
        P, C, Rf, ring, mu = fq.gf_chart_data()
        A = GfBPSKAlgebra(P, C, Rf, ring, mu_basis=mu)
        G = GBPSKAlgebra(fq, max_irrep=4)
        F, g = fq.flavour, fq.rank
        to_gf = lambda lab, A=A, F=F: A.r_label_compose(tuple(lab[0]) + tuple([0] * F.dim), F.irrep_to_ring_key(lab[1]))   # noqa: E731

        def from_gf(lab, A=A, F=F, g=g):
            sec, key = A.r_label_decompose(lab)
            return (tuple(sec[:g]), F.ring_key_to_irrep(key))
        labels = [(c, r) for c in it.product(range(3), repeat=g) for r in G.admissible_irreps(c)][:10]
        ok = tot = 0
        for a in labels:
            for b in labels:
                want = {from_gf(kk): str(v) for kk, v in A.multiply(to_gf(a), to_gf(b)).terms.items()}
                got = {kk: str(v) for kk, v in G.multiply(a, b).terms.items()}
                ok += got == want
                tot += 1
        axioms = all(G.verify_bar_involution(a, b) and G.verify_rho_is_automorphism(a, b) for a in labels[:4] for b in labels[:4]) and all(G.verify_rho_inverse(a) for a in labels[:4])
        agree[nm] = (ok, tot, axioms)
    checks.append(check("the flavoured K_q-algebra, presented twice with no shared code -- GBPSKAlgebra (products on the unfolded chart, irreps recovered by "
                        "character decomposition) and GfBPSKAlgebra (Weyl-character diagrams in the F-basis) -- agrees product for product: "
                        + "; ".join(f"{nm} {o}/{t}" for nm, (o, t, _a) in agree.items()) + "; the bar involution, rho an automorphism and rho o rho^-1 hold on GBPSKAlgebra",
                        all(o == t > 0 and a for (o, t, a) in agree.values())))
    return {"checks": checks, "population": {"tiers": {t: c["entries"] for t, c in rows.items()}, "cutoff": 3, "per tier": f"{k} stored entries at the tier's smallest stored reduced rank",
                                             "presentations": list(agree), "seconds": round(time.time() - t0, 1)},
            "controls": {"positive": "the covariant spec replays; S = the unfolded S", "negative": "a weight-level order breaks the character property"},
            "notes": "The ten shipped tiers are sampled; their full census is the manifests' (no crystalline_failed entry).",
            "inputs": [{"path": "dictionaries/flavoured/manifest.json", "tracked": True}]}


# --------------------------------------------------------------------------
# def:rg/reconstruction — the UV algebra rebuilt from (IR, S) against independent presentations
# --------------------------------------------------------------------------
def _pentagon_object_samples(O):
    """Identity + the multiplicative generators of every realisation of the pentagon object (as in
    the suite in the source repository): cone realisations by their own labels, the others transported."""
    def gen_labels(alg):
        cd = alg.cone_data()
        return [alg.identity()] + [cd.from_cone_label(frozenset({g}), {g: 1}) for g in cd.mult_gens()]
    out = {}
    cone = O.realization("cone-frozen")
    for key in O.keys():
        alg = O.realization(key)
        if getattr(alg, "cone_data", lambda: None)() is not None:
            out[key] = gen_labels(alg)
        else:
            out[key] = [alg.identity()] + [next(iter(O.transport(l, "cone-frozen", key).terms)) for l in gen_labels(cone)[1:]]
    return out


def check_reconstruction(environment):
    import itertools
    import signal
    from bps_kalgebra import BPSKAlgebra
    from finite_kalgebra_objects import kalgebra_object
    from pure_su2_bps_iso import pure_su2_bps_iso
    from uq_su2_bps_iso import uq_su2_bps_iso, iso_samples
    one = LaurentPoly.one()
    checks = []
    t0 = time.time()

    # 1. the pentagon: its BPS chart (UV rebuilt from Q_q(Γ) and S) against the closed-form, cone, a1a2k, sqed1-flow and skein
    #    presentations — every stored witness edge, products, ρ and traces through q^8
    O = kalgebra_object("pentagon")
    res = O.verify_pairwise(_pentagon_object_samples(O), pairs=True, trace_K=8)
    bad = {edge: {k: v for k, v in r.items() if not v} for edge, r in res.items() if not all(r.values())}
    bps_edges = sorted(e for e in res if "bps" in e)
    checks.append(check(f"pentagon: the BPS chart (rebuilt from the quantum torus and S) is isomorphic to the independent presentations {sorted(O.keys())}: "
                        f"{len(res)} witness edges ({len(bps_edges)} touch the chart), unit/round-trip/products/ρ/traces through q^8", not bad and bool(bps_edges), str(bad)[:200]))

    # 2. pure SU(2): the Kronecker chart against the abelianized presentation PureSU2KAlgebra (the suite in the source repository)
    labels = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, -1), (2, 0), (2, 1)]
    iso = pure_su2_bps_iso()
    src = [Element({a: one}) for a in labels]
    tgt = [iso.map(x) for x in src]
    pairs = [(Element({a: one}), Element({b: one})) for a, b in itertools.product(labels[:6], repeat=2)]
    tpairs = [(iso.map(x), iso.map(y)) for x, y in pairs]
    r2 = {"unit": iso.verify_unit(), "round_trip": iso.verify_round_trip(src, tgt), "multiplicative (36 pairs)": iso.verify_multiplicative(pairs, tpairs),
          "rho": iso.verify_rho_equivariant(src, tgt), "trace through q^10": iso.verify_trace_equivariant(src, tgt, K=10)}
    checks.append(check("pure SU(2): the Kronecker-quiver chart (rebuilt from the quantum torus and S) is isomorphic to the abelianized presentation: " + ", ".join(r2), all(r2.values()), str(r2)))

    # 3. U_q(sl_2) (implemented from the draft's definition) against the SU(2)-flavoured SQED_2 chart (the suite in the source repository)
    iso3 = uq_su2_bps_iso()
    A = iso3.source
    labs = iso_samples(A)
    src3 = [Element({l: one}) for l in labs]
    tgt3 = [iso3.map(s) for s in src3]
    gens = [A.K(1), A.K(-1), A.K(0, 1), A.E(), A.F(), A.E(1, 1), A.F(1, -1), A.E(2, 0), A.F(2, 0)]
    pairs3 = [(Element({a: one}), Element({b: one})) for a in gens for b in gens]
    tpairs3 = [(iso3.map(a), iso3.map(b)) for a, b in pairs3]
    r3 = iso3.verify_all(src3, tgt3, pairs3, tpairs3, trace_K=8)
    checks.append(check(f"U_q(sl_2) (built from the draft's definition) is isomorphic to the SQED_2 chart (rebuilt from the quantum torus and S): {len(labs)} labels, {len(pairs3)} generator pairs, traces through q^8",
                        all(r3.values()), str({k: v for k, v in r3.items() if not v})))

    # 4. negative control: the pentagon rebuilt from the WRONG factor order E_q(X_γ2)E_q(X_γ1).  Measured 2026-09-26: the node lines and
    #    their products agree with the correct S, so a product check cannot see it; the S itself violates eq:quiver at X_{γ1+γ2}
    #    (valuation 1, not >= 2), and the pairing does not return within the budget while the correct S's returns at once.
    pent = [[0, 1], [-1, 0]]
    right = BPSKAlgebra(pairing=pent, node_charges=[(1, 0), (0, 1)], spec=[(1, 0), (0, 1)], verify="off")
    wrong = BPSKAlgebra(pairing=pent, node_charges=[(1, 0), (0, 1)], spec=[(0, 1), (1, 0)], verify="off")
    def val(h):
        c = {e: v for e, v in h.expand(8)._coeffs.items() if v}   # LaurentPoly has no public valuation accessor
        return min(c) if c else None
    v_right, v_wrong = val(right._s_coefficient((1, 1))), val(wrong._s_coefficient((1, 1)))
    same_products = all(dict(right.multiply(a, b).terms) == dict(wrong.multiply(a, b).terms)
                        for a in [(1, 0), (0, 1), (1, 1)] for b in [(1, 0), (0, 1), (1, 1)])
    budget = 10

    class _Budget(Exception):
        pass

    def _alarm(*_):
        raise _Budget()
    old = signal.signal(signal.SIGALRM, _alarm)
    outcome = {}
    try:
        for name, alg in (("right", right), ("wrong", wrong)):
            t1 = time.time()
            signal.alarm(budget)
            try:
                outcome[name] = f"returned {alg.inner_product((0, 0), (0, 0), K=4)!r} in {time.time() - t1:.1f} s"
            except _Budget:
                outcome[name] = f"no return within {budget} s"
            finally:
                signal.alarm(0)
    finally:
        signal.signal(signal.SIGALRM, old)
    neg_ok = (v_right is not None and v_right >= 2 and v_wrong == 1 and same_products
              and outcome["right"].startswith("returned") and outcome["wrong"].startswith("no return"))
    checks.append(check("negative control: the pentagon rebuilt from the reversed order E_q(X_γ2)E_q(X_γ1) has the SAME node products as the correct S (a product check cannot see it); "
                        f"its S violates eq:quiver at X_(γ1+γ2) (valuation {v_wrong}, correct S: {v_right}); the pairing of the identity with itself: {outcome.get('wrong')}; correct S: {outcome.get('right')}",
                        neg_ok, str(outcome)))
    return {"checks": checks,
            "population": {"presentations": ["pentagon: BPS chart vs closed form, cone, a1a2k, sqed1 flow, skein (traces q^8)", "pure SU(2): Kronecker chart vs abelianized (traces q^10)",
                                             "U_q(sl_2) vs the SQED_2 chart (traces q^8)"], "seconds": round(time.time() - t0, 1)},
            "controls": {"positive": "the three isomorphisms (each a full KAlgebraIso battery)",
                         "negative": "the pentagon from the reversed factor order: eq:quiver fails, the pairing does not return; its node products are unchanged, so products alone cannot detect a wrong S"},
            "notes": "The UV algebra rebuilt from (IR, S) -- BPSKAlgebra's F-solver F(a)S = X_a + O(q) -- is compared with presentations that do not use S.  Uniqueness of the solution ('fully characterizes') is sec:rg/F-unique, not this row.", "inputs": []}


# --------------------------------------------------------------------------
# def:rg/composition — RG^{1->3} = RG^{2->3} o RG^{1->2}, S^{1->3} = RG^{2->3}(S^{1->2}) S^{2->3}
# --------------------------------------------------------------------------
def check_composition(environment):
    from bps_kalgebra import BPSKAlgebra
    from directional_subquiver_rg import DirectionalSubquiverRG
    from rgkalgebra import _apply_rg_to_habiro_dict, _multiply_habiro_dicts
    checks = []
    t0 = time.time()
    A4P = [[0, 1, 0, 0], [-1, 0, 1, 0], [0, -1, 0, 1], [0, 0, -1, 0]]
    A4N = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]
    D1 = DirectionalSubquiverRG(A4P, A4N, list(A4N), drop=[0], rg_window=6)
    D2 = DirectionalSubquiverRG.from_uv(D1.auxiliary(), [0], attach_f_oracle=False, rg_window=6)
    C = D1.then(D2)
    D12 = DirectionalSubquiverRG(A4P, A4N, list(A4N), drop=[0, 1], rg_window=6)
    labels = [(0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1), (1, 1, 0, 0), (0, 0, 1, 1), (1, 0, 0, 0), (0, 1, 0, 0)]
    rg_ok = all(C.RG(a) == D12.RG(a) for a in labels)
    checks.append(check(f"A_4, delete γ1 then γ2 vs both at once: RG^(1->3) = RG^(2->3) o RG^(1->2) on {len(labels)} UV labels", rg_ok and D2.auxiliary().node_charges == D12.auxiliary().node_charges))
    cut = 3
    sa, sb = C.rg_generator(cut), D12.rg_generator(cut)
    s_ok = set(sa) == set(sb) and all((sa[k] - sb[k]).is_zero() for k in sa)
    checks.append(check(f"A_4: S^(1->3) = RG^(2->3)(S^(1->2)) S^(2->3) equals the direct flow's S through grading height {cut} ({len(sa)} terms)", s_ok))
    UV = BPSKAlgebra(pairing=A4P, node_charges=A4N, spec=list(A4N), verify="off")
    prod_pairs = [((1, 0, 0, 0), (0, 1, 0, 0)), ((0, 1, 0, 0), (1, 0, 0, 0)), ((0, 0, 1, 0), (0, 0, 0, 1)), ((0, 1, 0, 0), (0, 0, 1, 0))]
    p_ok = all(C.multiply(a, b) == UV.multiply(a, b) for a, b in prod_pairs) and C.verify_rg_unital() \
        and all(C.verify_rg_multiplicative(a, b) for a, b in [((0, 0, 1, 0), (0, 0, 0, 1)), ((1, 0, 0, 0), (0, 1, 0, 0))])
    checks.append(check(f"A_4: the composite is unital and multiplicative, and its products equal the UV chart's on {len(prod_pairs)} pairs", p_ok))
    # negative control: the product in the other order, S^(2->3) · RG^(2->3)(S^(1->2)), must differ from the direct flow's S
    aux = D2.auxiliary()
    s_outer = _apply_rg_to_habiro_dict(D2, D1.rg_generator(cut))
    s_inner = D2.rg_generator(cut)
    g = C.grading()
    wrong = {k: v for k, v in _multiply_habiro_dicts(s_inner, s_outer, aux).items() if g.height_of(k) <= cut}
    right = {k: v for k, v in _multiply_habiro_dicts(s_outer, s_inner, aux).items() if g.height_of(k) <= cut}
    differs = set(wrong) != set(sb) or any(not (wrong[k] - sb[k]).is_zero() for k in wrong)
    same_as_class = set(right) == set(sa) and all((right[k] - sa[k]).is_zero() for k in right)
    checks.append(check("negative control: S composed in the other order, S^(2->3) RG^(2->3)(S^(1->2)), differs from the direct flow's S (and the draft's order reproduces the class's)", differs and same_as_class))
    # A_3: two single-node deletions (BPS rg_flow_morphism) vs the two-node SubquiverRG
    A3P = [[0, 1, 0], [-1, 0, 1], [0, -1, 0]]
    A3 = BPSKAlgebra(pairing=A3P, node_charges=[(1, 0, 0), (0, 1, 0), (0, 0, 1)], spec=[(1, 0, 0), (0, 1, 0), (0, 0, 1)], verify="off")
    rg1 = A3.rg_flow_morphism(0)
    rg2 = rg1.auxiliary().rg_flow_morphism(0)
    comp = rg1.then(rg2)
    sq = A3.subquiver_rg_morphism([0, 1])
    lab3 = [(0, 0, 0), (0, 0, 1), (0, 0, 2), (1, 1, 1), (0, 1, 1), (1, 1, 0)]
    a3_ok = sq.auxiliary().root_data() == comp.auxiliary().root_data() and all(sq.RG(a) == comp.RG(a) for a in lab3)
    checks.append(check(f"A_3: deleting γ1 then γ2 (two single-node flows) equals the two-node deletion on {len(lab3)} labels, same IR", a3_ok))
    return {"checks": checks,
            "population": {"flows": ["A_4: delete γ1 then γ2 vs both (directional)", "A_3: delete γ1 then γ2 vs both (BPS node-deletion flows)"], "S window": f"grading height <= {cut}", "seconds": round(time.time() - t0, 1)},
            "controls": {"positive": "the direct two-node deletion", "negative": "S composed in the other order differs"},
            "notes": "The composition law follows from R3 and R4 applied term by term, so this row tests the code (RGKAlgebra.then / ComposedRG), not the statement.  The matter-removal tower does not compose through `then` (each rung's IR carries the removed flavour as a spectator the next rung's UV lacks) and is not covered.", "inputs": []}


# --------------------------------------------------------------------------
# def:rg/rho-family — (RG, S) -> (rho_IR^-1 RG rho_UV, rho_IR^-1 S) is again a flow (derived; a regression check on BPS charts)
# --------------------------------------------------------------------------
def check_rho_family(environment):
    """The statement follows from the axioms (R2: a composite of algebra maps; R3: apply rho_IR^-1 to both equalities at the
    label rho_UV(a); R4: the identity I_{a,b} = I_{rho^-1 b, rho^-1 a}, i.e. K4's two forms of I, used twice).  On a BPS chart
    rho_IR(X_γ) = X_{-γ} (ex:qt), so the new flow is the flow of the same quiver with node charges -γ_i, relabelled by
    a -> -rho_UV(a).  What can be checked is the realisation: the solver's symmetry under negating every charge, the identity
    itself, and R4 of the transformed flow computed on the negated chart."""
    from bps_kalgebra import BPSKAlgebra
    from pure_ade_lattice import pure_ade_kalgebra
    checks = []
    t0 = time.time()
    K = 6
    kron = pure_ade_kalgebra([("A", 1)])
    charts = {
        "pentagon": (BPSKAlgebra(pairing=[[0, 1], [-1, 0]], node_charges=[(1, 0), (0, 1)], spec=[(1, 0), (0, 1)], verify="off"),
                     [[0, 1], [-1, 0]], [(1, 0), (0, 1)], [(1, 0), (0, 1)]),
        "Kronecker (pure SU(2))": (kron, [list(r) for r in kron.lattice.pairing], [tuple(g) for g in kron.node_charges], [tuple(g) for g in kron.spec]),
    }
    window = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (1, 2), (-1, 1), (1, -1)]

    def neg(v):
        return tuple(-x for x in v)
    summary = {}
    for name, (A, P, nodes, spec) in charts.items():
        Aneg = BPSKAlgebra(pairing=P, node_charges=[neg(g) for g in nodes], spec=[neg(g) for g in spec], verify="off")
        # 1. negation symmetry of the F-solver: F^-(−c) = N(F(c)), N: X_γ -> X_{−γ}
        f_ok = all(Aneg.F(neg(c)) == {neg(g): v for g, v in A.F(c).items()} for c in window)
        # 2. the identity the remark uses: I_{a,b} = I_{rho^-1 b, rho^-1 a}
        pairs = [(a, b) for a in window[:5] for b in window[:5]]
        I = {}
        def pair(alg, a, b):
            key = (id(alg), a, b)
            if key not in I:
                I[key] = alg.inner_product(a, b, K=K)
            return I[key]
        id_ok = all(pair(A, a, b) == pair(A, A.rho_inverse(b), A.rho_inverse(a)) for a, b in pairs)
        # 3. R4 of the transformed flow: (L_a|L_b)_UV = (RG'_a S'|RG'_b S')_IR, the latter the negated chart's pairing at −rho(a), −rho(b)
        r4_ok = all(pair(A, a, b) == pair(Aneg, neg(A.rho(a)), neg(A.rho(b))) for a, b in pairs)
        summary[name] = {"F negation symmetry": f_ok, "I_ab = I_(rho^-1 b, rho^-1 a)": id_ok, "R4 of the transformed flow": r4_ok}
        checks.append(check(f"{name}: the negated chart's F is the negation of F on {len(window)} labels; I_(a,b) = I_(rho^-1(b), rho^-1(a)) and R4 of the "
                            f"transformed flow on {len(pairs)} pairs, through q^{K}", f_ok and id_ok and r4_ok, str(summary[name])))
    # negative control: a label permutation that is NOT rho breaks the identity (the check can fail)
    A = charts["pentagon"][0]
    swap = {(1, 0): (0, 1), (0, 1): (1, 0)}
    def fake_rho_inv(x):
        return swap.get(x, A.rho_inverse(x))
    pairs = [(a, b) for a in window[:5] for b in window[:5]]
    broken = [(a, b) for a, b in pairs if A.inner_product(a, b, K=K) != A.inner_product(fake_rho_inv(b), fake_rho_inv(a), K=K)]
    checks.append(check(f"negative control: with rho^-1 replaced by a permutation that swaps (1,0) and (0,1), the identity fails on {len(broken)} of {len(pairs)} pentagon pairs",
                        len(broken) > 0, str(broken[:3])))
    return {"checks": checks, "population": {"charts": list(charts), "labels": window, "K": K, "seconds": round(time.time() - t0, 1)},
            "controls": {"positive": "the pentagon chart", "negative": "a permutation other than rho breaks I_ab = I_(rho^-1 b, rho^-1 a)"},
            "notes": "A regression check of the realisation: the statement is derived from the axioms.", "inputs": []}


# --------------------------------------------------------------------------
# conj:mut -- the mutation relation between spectrum generators (added with the 2026-09-29 draft)
# --------------------------------------------------------------------------
_MUT_DEPTH = {2: 6, 3: 5, 4: 4, 5: 4}


def _mut_independent(B, j, D):
    """conj:mut on the independent arithmetic (a probe in the source repository): S and S' by the forced recursion, each in its
    own node frame, Q' = mu_j(Q) paired by M^T B M with M the charge map of eq:mut-charges.  True when
    E(X_{g_j})^{-1} S and S' E(X_{-g_j})^{-1}, carried to one lattice, are supported in both cones and agree on every
    charge exact in both windows (and at least one charge is compared)."""
    import experiments.pbw_e_group as pe
    r = len(B)
    nodes = [tuple(int(k == i) for k in range(r)) for i in range(r)]
    M = [[0] * r for _ in range(r)]                                  # M[c][a]: component c of the charge of node a of Q'
    for a in range(r):
        if a == j:
            M[j][a] = -1
        else:
            M[a][a], M[j][a] = 1, max(B[a][j], 0)
    Bp = [[sum(M[c][a] * B[c][d] * M[d][b] for c in range(r) for d in range(r)) for b in range(r)] for a in range(r)]
    S, _ = pe.build_S_forced(B, nodes, D, pe.key_degree_lex)
    Sp, _ = pe.build_S_forced(Bp, nodes, D, pe.key_degree_lex)
    L = {g: c for g, c in pe.elt_mul(pe.E_spin_element(nodes[j], 0, -1, D), S, B, D).items() if not pe.cf_is_zero(c)}
    R = {tuple(sum(M[c][a] * g[a] for a in range(r)) for c in range(r)): v
         for g, v in pe.elt_mul(Sp, pe.E_spin_element(nodes[j], 0, -1, D), Bp, D).items() if not pe.cf_is_zero(v)}

    def primed(g):                                                    # Q coordinates -> Q' coordinates
        return tuple(g[i] if i != j else sum(g[a] * max(B[a][j], 0) for a in range(r) if a != j) - g[j] for i in range(r))
    if any(min(primed(g)) < 0 for g in L) or any(min(g) < 0 for g in R):
        return False
    common = [g for g in set(L) | set(R) if pe.deg(g) <= D and pe.deg(primed(g)) <= D]
    zero = pe.cf_from_lp({})
    return bool(common) and all(pe.cf_eq(L.get(g, zero), R.get(g, zero)) for g in common)


def check_conj_mut(environment):
    """conj:mut.  'Q' = mu_j(Q) (node charges eq:mut-charges): S = E(X_{g_j}) S'' and S' = S'' E(X_{-g_j}) for an S'' in
    E and E'.'  Eliminating S'' gives S' = E(X_{g_j})^{-1} S E(X_{-g_j}), which the design record's mutation_s_relation.compare_edge
    compares exactly on every charge the two cones determine, recording a support violation wherever E(X_{g_j})^{-1} S
    leaves the cone of Q' (the S'' in E' half).  Each S is built from its own quiver's constraint in its own node frame;
    that the two are related by the mutation is not imposed."""
    import mutation_s_relation as msr
    t0 = time.time()
    checks, rows = [], {}
    lineup = _lineup_quivers()
    green_ok = mixed_fail = n_edges = 0
    for name, (B, _specs) in lineup.items():
        D = _MUT_DEPTH.get(len(B), 3)
        cs = [msr.compare_edge(B, k, D) for k in range(len(B))]
        mixed = [(msr.compare_edge(B, k, D, variant="mixed_green").ok, msr.compare_edge(B, k, D, variant="mixed_reverse").ok)
                 for k in range(len(B))]
        n_edges += len(B)
        green_ok += sum(1 for c in cs if c.ok and c.n_content > 0)
        mixed_fail += sum(1 for a, b in mixed if not a and not b)
        rows[name] = {"depth": D, "edges": f"{sum(1 for c in cs if c.ok and c.n_content > 0)}/{len(B)}",
                      "charges compared beyond the leading data": [c.n_content for c in cs],
                      "support violations": sum(len(c.support_violations) for c in cs)}
    checks.append(check(f"every edge Q -> mu_j(Q) of the Section 4 lineup, every node j: S' = E(X_g_j)^-1 S E(X_-g_j) on every "
                        f"compared charge and E(X_g_j)^-1 S inside the cone of Q': {green_ok}/{n_edges} edges",
                        green_ok == n_edges))
    checks.append(check(f"negative controls: both mixed variants (one side with the wrong lattice identification) fail on "
                        f"{mixed_fail}/{n_edges} lineup edges", mixed_fail == n_edges))
    indep = [(name, j) for name, (B, _s) in lineup.items() for j in range(len(B))
             if not _mut_independent(B, j, _MUT_DEPTH.get(len(B), 3))]
    checks.append(check(f"an independent construction of S (the forced recursion of a probe in the source repository) gives the same "
                        f"verdict on every lineup edge: {n_edges - len(indep)}/{n_edges}", not indep, str(indep[:3])))
    m, mline = _manifest()
    entries = sample_entries(4, N_SAMPLE)
    s_ok = s_tot = 0
    bad = []
    for e in entries:
        B = e["exchange"]
        for k in range(len(B)):
            c = msr.compare_edge(B, k, _MUT_DEPTH.get(len(B), 3))
            s_tot += 1
            if c.ok and c.n_content > 0:
                s_ok += 1
            else:
                bad.append((B, k, c.as_dict()))
    checks.append(check(f"every edge of the first {len(entries)} entries with n <= 4 of {mline}: {s_ok}/{s_tot}",
                        s_ok == s_tot and s_tot > 0, str(bad[:1])))
    return {"checks": checks,
            "population": {"lineup": rows, "sample": f"first {len(entries)} entries with n <= 4 of {mline}, every node",
                           "seconds": round(time.time() - t0, 1)},
            "controls": {"positive": "the pentagon, where the relation is the pentagon identity",
                         "negative": "the two mixed lattice identifications"},
            "notes": "the design record (mutation_s_relation.py) is the implementation; its local sweep (orbits, crystalline tier "
                     "entries of ranks 3-12, witness paths across the weight cutoff) is recorded in its README.",
            "inputs": [{"path": "dictionaries/enumerated/manifest.json", "tracked": True}]}


ADAPTERS = {
    "def:wallcrossing": check_wallcrossing,
    "def:rg/rho-family": check_rho_family,
    "def:rg/reconstruction": check_reconstruction,
    "def:rg/composition": check_composition,
    "sec:rg/E-group": check_E_group,
    "def:rg/matter-removal": check_matter_removal,
    "conj:cluster": check_sw_sample,
    "conj:S-unique": check_s_unique,
    "conj:pbw": check_pbw,
    "conj:mut": check_conj_mut,
    "sec:rg/F-unique": check_f_unique,
    "conj:upper": check_upper_sample,
    "def:rg/no-exotic": check_no_exotic,
    "conj:factored": check_factored,
    "def:rg/central-charge": check_central_charge,
    "sec:rg/flavoured-quivers": check_flavoured_quivers,
}
