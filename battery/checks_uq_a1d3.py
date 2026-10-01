"""checks_uq_a1d3.py — the design record adapters for the two flavoured examples of the
draft's Section 2: the U_q(sl_2) K_q-algebra (sec:uqsl2) and the punctured
triangle K_q([A_1,D_3]) (sec:a1d3).

Every printed closed form is recomputed independently with a small exact
two-variable series class (q-truncated, Laurent monomials in μ and x with
integer coefficients) and compared coefficient by coefficient with what the
classes return; SU(2) characters χ_j are expanded as μ^j + μ^{j-2} + ... + μ^{-j}.

Rows: sec:uqsl2/algebra (UqSL2PBW implements the definition), sec:uqsl2/trace
(the generating function G(x,μ) against the Wilson traces of SQED_2),
sec:uqsl2/recursion (eq:uqsl2rec / eq:uqsl2con and the iterative uniqueness),
sec:a1d3/algebra (A1D3KAlg implements the definition), sec:a1d3/traces (the
three elementary traces over the Weyl–Kac denominator), rem:a1d3-miracle (the
2x3 minors of the two constraint families against the elementary traces —
measured here for the first time).
"""
from __future__ import annotations

import itertools
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # the release root
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from kalgebra import Element  # noqa: E402
from laurent_poly import LaurentPoly  # noqa: E402
from checks_kq import check, elem  # noqa: E402
from checks_flavoured import F_VERIFIERS  # noqa: E402


# --------------------------------------------------------------------------
# exact q-truncated series with Laurent-monomial coefficients in `nv` variables
# --------------------------------------------------------------------------
class S2:
    __slots__ = ("c", "K", "nv")

    def __init__(self, c: dict, K: int, nv: int):
        self.K = K
        self.nv = nv
        self.c = {}
        for e, poly in c.items():
            if e > K:
                continue
            p = {m: int(v) for m, v in poly.items() if v}
            if p:
                self.c[e] = p

    @classmethod
    def one(cls, K, nv):
        return cls({0: {(0,) * nv: 1}}, K, nv)

    @classmethod
    def term(cls, e, mono, K, nv, coeff=1):
        return cls({e: {tuple(mono): coeff}}, K, nv)

    def __add__(self, o):
        K = min(self.K, o.K)
        c = {e: dict(p) for e, p in self.c.items()}
        for e, p in o.c.items():
            d = c.setdefault(e, {})
            for m, v in p.items():
                d[m] = d.get(m, 0) + v
        return S2(c, K, self.nv)

    def __neg__(self):
        return S2({e: {m: -v for m, v in p.items()} for e, p in self.c.items()}, self.K, self.nv)

    def __sub__(self, o):
        return self + (-o)

    def scale(self, n):
        return S2({e: {m: v * n for m, v in p.items()} for e, p in self.c.items()}, self.K, self.nv)

    def __mul__(self, o):
        if isinstance(o, int):
            return self.scale(o)
        K = min(self.K, o.K)
        c: dict = {}
        for e1, p1 in self.c.items():
            for e2, p2 in o.c.items():
                if e1 + e2 > K:
                    continue
                d = c.setdefault(e1 + e2, {})
                for m1, v1 in p1.items():
                    for m2, v2 in p2.items():
                        m = tuple(a + b for a, b in zip(m1, m2))
                        d[m] = d.get(m, 0) + v1 * v2
        return S2(c, K, self.nv)

    def shift(self, k):
        return S2({e + k: p for e, p in self.c.items()}, self.K + k, self.nv)

    def trunc(self, K):
        return S2({e: p for e, p in self.c.items() if e <= K}, K, self.nv)

    def val(self):
        return min(self.c) if self.c else None

    def eq_to(self, o, K=None):
        K = min(self.K, o.K) if K is None else K
        a = {e: p for e, p in self.c.items() if e <= K}
        b = {e: p for e, p in o.c.items() if e <= K}
        return a == b

    def first_diff(self, o, K=None):
        K = min(self.K, o.K) if K is None else K
        d = (self - o).trunc(K)
        return d.val()

    def coeff_x(self, n, var=0):
        """[var^n]: drop the variable `var` (a series in the remaining ones)."""
        c: dict = {}
        for e, p in self.c.items():
            for m, v in p.items():
                if m[var] == n:
                    mm = m[:var] + m[var + 1:]
                    d = c.setdefault(e, {})
                    d[mm] = d.get(mm, 0) + v
        return S2(c, self.K, self.nv - 1)

    def __repr__(self):
        rows = []
        for e in sorted(self.c):
            rows.append(f"q^{e}:[" + " ".join(f"{v:+d}m{m}" for m, v in sorted(self.c[e].items())) + "]")
        return " ".join(rows) + f" +O(q^{self.K + 1})"


def inv_one_minus(e, mono, K, nv, sign=1):
    """1/(1 - sign*q^e m) = Σ_j sign^j q^{e j} m^j  (e > 0)."""
    c = {}
    j = 0
    while e * j <= K:
        c[e * j] = {tuple(x * j for x in mono): sign ** j}
        j += 1
    return S2(c, K, nv)


def qpoch_inf2(step, mono, K, nv):
    """(q^step m; q^step)_∞ = Π_{j>=1} (1 - q^{step j} m)."""
    out = S2.one(K, nv)
    j = 1
    while step * j <= K:
        out = out * (S2.one(K, nv) - S2.term(step * j, mono, K, nv))
        j += 1
    return out


def inv_qpoch_inf2(step, mono, K, nv):
    out = S2.one(K, nv)
    j = 1
    while step * j <= K:
        out = out * inv_one_minus(step * j, mono, K, nv)
        j += 1
    return out


def series_div2(num: S2, den: S2, K: int) -> S2:
    """num/den where den has q-valuation 0 and constant coefficient the monomial 0 with value ±1."""
    d0 = den.c.get(0, {})
    z = (0,) * den.nv
    assert set(d0) == {z} and d0[z] in (1, -1), f"series_div2: constant term must be ±1, got {d0}"
    lead = d0[z]
    out = S2({}, K, num.nv)
    rem = num.trunc(K)
    for e in range(min(rem.c) if rem.c else 0, K + 1):
        p = rem.c.get(e)
        if not p:
            continue
        piece = S2({e: {m: v * lead for m, v in p.items()}}, K, num.nv)
        out = out + piece
        rem = rem - piece * den
    return out


def chi_mu(j, K=10 ** 6):
    """The SU(2) character χ_j(μ) = μ^j + μ^{j-2} + ... + μ^{-j} as a one-variable S2 at q^0."""
    return S2({0: {(j - 2 * i,): 1 for i in range(j + 1)}}, K, 1)


def key_to_j(key):
    if isinstance(key, int):
        return key
    return key[0] if key else 0


def relement_to_mu(r, K=10 ** 6):
    out = S2({}, K, 1)
    if hasattr(r, "terms"):
        for key, n in r.terms.items():
            if n:
                out = out + chi_mu(key_to_j(key), K).scale(n)
    elif r:
        out = out + S2({0: {(0,): int(r)}}, K, 1)
    return out


def rps_to_mu(t, K=None):
    K = t.K if K is None else K
    out = S2({}, K, 1)
    for e, v in t.coeffs.items():
        if e <= K:
            out = out + relement_to_mu(v, K).shift(e).trunc(K)
    return out


def qdict_to_mu(qd, K=10 ** 6):
    """{q_exp: RElement} (a layer-1 row) -> S2 over μ (a Laurent polynomial in q)."""
    out = S2({}, K, 1)
    for e, r in qd.items():
        out = out + relement_to_mu(r, K).shift(e)
    return out


# --------------------------------------------------------------------------
# U_q(sl_2)
# --------------------------------------------------------------------------
def check_uqsl2_algebra(environment):
    from uq_sl2_pbw import UqSL2PBW, E_LAB, F_LAB, K_LAB, C_LAB
    A = UqSL2PBW()
    E, F, K1, Km = E_LAB, F_LAB, K_LAB(1), K_LAB(-1)
    C = C_LAB(1)
    one = A.identity()
    checks = []
    q = lambda e, lab: elem({lab: {e: 1}})  # noqa: E731
    checks.append(check("KE = q^-2 EK", A.multiply(K1, E) == elem({('E', 1, 1, 0): {-1: 1}}) and A.multiply(E, K1) == elem({('E', 1, 1, 0): {1: 1}})))
    checks.append(check("KF = q^2 FK", A.multiply(K1, F) == elem({('F', 1, 1, 0): {1: 1}}) and A.multiply(F, K1) == elem({('F', 1, 1, 0): {-1: 1}})))
    checks.append(check("EF = χ_1 + q K + q^-1 K^-1 (χ_1 the central Casimir label)", A.multiply(E, F) == elem({C: {0: 1}, K1: {1: 1}, Km: {-1: 1}})))
    checks.append(check("FE = χ_1 + q^-1 K + q K^-1", A.multiply(F, E) == elem({C: {0: 1}, K1: {-1: 1}, Km: {1: 1}})))
    comm = A.multiply(E, F) + (A.multiply(F, E) * -1)
    checks.append(check("[E,F] = (q - q^-1)(K - K^-1)", comm == elem({K1: {1: 1, -1: -1}, Km: {-1: 1, 1: -1}})))
    checks.append(check("the Casimir is central: χ_1 commutes with E, F, K", all(A.multiply(C, g) == A.multiply(g, C) for g in (E, F, K1, Km))))
    ok_norm = all(A.multiply(('E', a, 0, 0), K_LAB(b)) == elem({('E', a, b, 0): {a * b: 1}}) and A.multiply(('F', a, 0, 0), K_LAB(b)) == elem({('F', a, b, 0): {-a * b: 1}})
                  for a in range(1, 4) for b in range(-3, 4) if b != 0)
    checks.append(check("E_{a,b} = q^-ab E^a K^b and F_{a,b} = q^ab F^a K^b (E^a K^b = q^ab E_{a,b}), a <= 3, |b| <= 3", ok_norm))
    checks.append(check("pure powers: E^a = E_{a,0}, F^a = F_{a,0}, K^n K^m = K^{n+m}", all(A.multiply(('E', a, 0, 0), E) == elem({('E', a + 1, 0, 0): {0: 1}}) and A.multiply(('F', a, 0, 0), F) == elem({('F', a + 1, 0, 0): {0: 1}}) for a in range(1, 4)) and all(A.multiply(K_LAB(n), K_LAB(m)) == elem({K_LAB(n + m) if n + m else one: {0: 1}}) for n in range(-2, 3) for m in range(-2, 3))))
    checks.append(check("rho(K) = K^-1, rho(E) = q^-1 F K^-1 = F_{1,-1}, rho(F) = q K E = E_{1,1}, rho fixes χ_1", A.rho(K1) == Km and A.rho(E) == ('F', 1, -1, 0) and A.rho(F) == ('E', 1, 1, 0) and A.rho(C) == C
                        and A.multiply(F, Km) == elem({('F', 1, -1, 0): {1: 1}}) and A.multiply(K1, E) == elem({('E', 1, 1, 0): {-1: 1}})))
    orb = [E]
    for _ in range(12):
        orb.append(A.rho(orb[-1]))
    checks.append(check("rho has infinite order (Lusztig's braid): rho^k(E) distinct for k <= 12, alternating F_{1,-(2m+1)} and E_{1,2m}", len(set(orb)) == len(orb) and orb[1] == ('F', 1, -1, 0) and orb[2] == ('E', 1, 2, 0) and orb[3] == ('F', 1, -3, 0),
                        f"orbit: {orb[:6]}"))
    checks.append(check("rho covers the Weyl reflection: rho^2(K) = K, rho^2(E) = E_{1,2} = q^-2 E K^2 (not E)", A.rho(A.rho(K1)) == K1 and A.rho(A.rho(E)) == ('E', 1, 2, 0)))
    W = [one, E, F, K1, Km, ('E', 1, 1, 0), ('F', 1, -1, 0), ('E', 2, 0, 0), ('F', 2, 1, 0), K_LAB(2), C, ('K', 1, 1)]
    ok_bar = all(A.verify_bar_involution(a, b) for a in W for b in W)
    ok_unit = A.verify_identity_in_basis() and all(A.verify_unit_law(a) for a in W)
    ok_assoc = all(A.verify_associativity(a, b, c) for a in W[:7] for b in W[:7] for c in W[:7])
    ok_rho = A.verify_rho_fixes_identity() and all(A.verify_rho_inverse(a) for a in W) and all(A.verify_rho_is_automorphism(a, b) for a in W for b in W)
    checks.append(check(f"K1 bar-invariance on {len(W)**2} pairs; K2 unit and associativity on {7**3} triples; K3 rho automorphism and inverse", ok_bar and ok_unit and ok_assoc and ok_rho))
    checks.append(check("negative control: the altered relation EF = χ_1 + q^2 K + q^-2 K^-1 is detected", A.multiply(E, F) != elem({C: {0: 1}, K1: {2: 1}, Km: {-2: 1}})))
    return {"checks": checks, "population": {"generators": "E, F, K^{±1}, χ_1", "window": f"{len(W)} labels", "orbit": "rho^k(E), k <= 12"},
            "controls": {"positive": "the printed relations reproduced as elements", "negative": "altered relation detected"},
            "notes": "The class carries the Casimir as a label coordinate (coefficient ring Z), so the flavoured trace lives on the SQED_2 realisation (next rows); K4/K5 are checked there.  The bar involution here is antimultiplicative and fixes K (the design notes False friends).", "inputs": []}


def G_series(K):
    """G(x,μ) = (q^2;q^2)^2_∞ E_q(μx)E_q(μ^-1 x)E_q(μ x^-1)E_q(μ^-1 x^-1), E_q(y) = (-q y; q^2)^-1_∞ = Π_{k>=0} 1/(1 + q^{2k+1} y).
    Variables (x, μ)."""
    nv = 2
    out = qpoch_inf2(2, (0, 0), K, nv) * qpoch_inf2(2, (0, 0), K, nv)
    for mono in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        k = 0
        while 2 * k + 1 <= K:
            out = out * inv_one_minus(2 * k + 1, mono, K, nv, sign=-1)
            k += 1
    return out


def sqed2_traces(K, nmax):
    from gn_abe_kalgebra import GNAbeKAlgebra
    from root_datum import u_n
    from zplus_ring import un_to_sun_hom
    B = GNAbeKAlgebra(u_n(1), (1,), nf=2).base_change(un_to_sun_hom(2))
    return B, {n: rps_to_mu(B.trace((((0,), (n,)), (0, 0)), K), K) for n in range(-nmax, nmax + 1)}


def check_uqsl2_trace(environment):
    K = 12
    nmax = 6
    checks = []
    G = G_series(K)
    Gx = {n: G.coeff_x(n, var=0) for n in range(-nmax, nmax + 1)}
    # positive control: the printed leading terms (kalgebra.md cross-reference): Tr K^0 = 1 + q^2(χ_2 - 1) + ...
    chi2 = chi_mu(2, K)
    lead = S2.one(K, 1) + (chi2 - S2.one(K, 1)).shift(2)
    checks.append(check("positive control: [x^0]G = 1 + q^2(χ_2 - 1) + O(q^4) (the U(1) vector multiplet's -1)", Gx[0].trunc(3).eq_to(lead.trunc(3), 3), f"[x^0]G through q^4: {Gx[0].trunc(4)}"))
    checks.append(check("G is symmetric under x -> x^-1 and under μ -> μ^-1 (Weyl / charge conjugation)", all(Gx[n].eq_to(Gx[-n], K) for n in range(1, nmax + 1)) and all(Gx[n].eq_to(S2({e: {(-m[0],): v for m, v in p.items()} for e, p in Gx[n].c.items()}, K, 1), K) for n in range(nmax + 1))))
    B, T = sqed2_traces(K, nmax)
    ok = {n: T[n].eq_to(Gx[n], K) for n in T}
    checks.append(check(f"Tr K^n = [x^n] G(x,μ) on the SQED_2 Wilson lines (GNAbeKAlgebra(u_n(1),(1,),nf=2), R(U(2)) -> R(SU(2))), |n| <= {nmax}, through q^{K}", all(ok.values()),
                        "; ".join(f"n={n}: {'ok' if v else 'FAIL at q^' + str(T[n].first_diff(Gx[n], K))}" for n, v in sorted(ok.items()))))
    vals = {n: T[n].val() for n in T if n}
    checks.append(check("measured: Tr K^n = O(q^|n|) exactly, 1 <= |n| <= 6", all(v == abs(n) for n, v in vals.items()), str(vals)))
    charged = [(((1,), (e,)), (0, 0)) for e in range(-2, 3)] + [(((2,), (0,)), (0, 0)), (((-1,), (1,)), (0, 0))]
    checks.append(check("the trace vanishes on the charged sector (E_{a,b}, F_{a,b}: magnetic charge != 0 in SQED_2), 7 labels", all(not B.trace(l, 8).coeffs or all(not (v.terms if hasattr(v, 'terms') else v) for v in B.trace(l, 8).coeffs.values()) for l in charged)))
    Gneg = qpoch_inf2(2, (0, 0), K, 2) * qpoch_inf2(2, (0, 0), K, 2)
    for mono in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        k = 0
        while 2 * k + 1 <= K:
            Gneg = Gneg * inv_one_minus(2 * k + 1, mono, K, 2, sign=1)
            k += 1
    checks.append(check("negative control: E_q with the opposite sign (q y; q^2)^-1 is detected at n = 1", not T[1].eq_to(Gneg.coeff_x(1, var=0), K)))
    return {"checks": checks, "population": {"n": f"|n| <= {nmax}", "K": K, "realisation": "SQED_2 as GNAbeKAlgebra(u_n(1),(1,),nf=2) after base_change(un_to_sun_hom(2))"},
            "controls": {"positive": "the printed leading terms of Tr K^0", "negative": "opposite sign in E_q detected"},
            "notes": "G(x,μ) is expanded independently as a two-variable series; the class's SU(2)-character coefficients are expanded in μ for the comparison.  The SQED_2 realisation computes the trace from eq:measure at (U(1), 2 fundamentals), the draft's own derivation of eq:uqsl2trace.", "inputs": []}


def check_uqsl2_recursion(environment):
    K = 10
    nmax = 6
    checks = []
    B, T = sqed2_traces(K + 2, nmax)
    chi1 = chi_mu(1, K + 2)
    # eq:uqsl2rec for n >= 0 (all q-shifts non-negative): q^{2n}(T_n + q χ_1 T_{n+1} + q^2 T_{n+2}) = T_n + q χ_1 T_{n-1} + q^2 T_{n-2}
    ok_rec = True
    for n in range(0, nmax - 1):
        lhs = (T[n] + (chi1 * T[n + 1]).shift(1) + T[n + 2].shift(2)).shift(2 * n)
        rhs = T[n] + (chi1 * T[n - 1]).shift(1) + T[n - 2].shift(2)
        ok_rec &= lhs.eq_to(rhs, K)
    checks.append(check(f"eq:uqsl2rec  q^2n (Tr K^n + q χ_1 Tr K^n+1 + q^2 Tr K^n+2) = Tr K^n + q χ_1 Tr K^n-1 + q^2 Tr K^n-2 on the SQED_2 traces, 0 <= n <= 4, through q^{K}", ok_rec))
    ok_con = True
    for n in range(-(nmax - 2), 0):  # eq:uqsl2con, second line (n < 0)
        m = -n
        lhs = T[n] - T[n].shift(2 * m)
        rhs = -(chi1 * T[n + 1]).shift(1) - T[n + 2].shift(2) + (chi1 * T[n - 1]).shift(1 + 2 * m) + T[n - 2].shift(2 + 2 * m)
        ok_con &= lhs.eq_to(rhs, K)
    checks.append(check(f"eq:uqsl2con (n < 0 form)  (1 - q^-2n) Tr K^n = -q χ_1 Tr K^n+1 - q^2 Tr K^n+2 + q^1-2n χ_1 Tr K^n-1 + q^2-2n Tr K^n-2, -4 <= n <= -1", ok_con))
    # iterative uniqueness: from Tr 1 given and Tr K^n = O(q) for n != 0, the recursion determines everything order by order
    N = K + 1
    G = G_series(K)
    T0 = G.coeff_x(0, var=0)
    Tn = {n: (T0.trunc(K) if n == 0 else S2({}, K, 1)) for n in range(-N, N + 1)}

    def get(n):
        return Tn[n] if -N <= n <= N else S2({}, K, 1)
    c1 = chi_mu(1, K)
    for k in range(1, K + 1):
        new = {}
        for n in range(-N, N + 1):
            if n == 0:
                continue
            if n > 0:
                rhs = (c1 * get(n + 1)).shift(2 * n + 1) + get(n + 2).shift(2 * n + 2) - (c1 * get(n - 1)).shift(1) - get(n - 2).shift(2) + get(n).shift(2 * n)
            else:
                m = -n
                rhs = -(c1 * get(n + 1)).shift(1) - get(n + 2).shift(2) + (c1 * get(n - 1)).shift(1 + 2 * m) + get(n - 2).shift(2 + 2 * m) + get(n).shift(2 * m)
            new[n] = rhs.c.get(k, {})
        for n, p in new.items():
            if p:
                Tn[n] = Tn[n] + S2({k: p}, K, 1)
    ok_iter = all(Tn[n].eq_to(G.coeff_x(n, var=0), K) for n in range(-4, 5) if n)
    checks.append(check(f"iterating eq:uqsl2con from Tr 1 = [x^0]G and Tr K^n = O(q) reproduces [x^n]G for 1 <= |n| <= 4 through q^{K} (the unique solution)", ok_iter,
                        "; ".join(f"n={n}: first difference q^{Tn[n].first_diff(G.coeff_x(n, var=0), K)}" for n in range(-4, 5) if n and not Tn[n].eq_to(G.coeff_x(n, var=0), K)) or "all agree"))
    checks.append(check("the iteration is well-posed: every right-hand-side term carries a positive power of q (orders < k determine order k)", True, "by inspection of eq:uqsl2con: the coefficients q^{2n+1}, q^{2n+2}, q, q^2 (n>0) and q, q^2, q^{1-2n}, q^{2-2n} (n<0) are all positive powers"))
    return {"checks": checks, "population": {"n": "|n| <= 4 (traces to |n| <= 6)", "K": K, "iteration": f"|n| <= {N}"},
            "controls": {"positive": "the class's traces satisfy the recursion", "negative": "n.a."},
            "notes": "Truncating the infinite system at |n| <= K+1 is safe because Tr K^n = O(q^|n|) (measured on the trace row), so the dropped terms enter beyond q^K.", "inputs": []}


# --------------------------------------------------------------------------
# [A_1, D_3]
# --------------------------------------------------------------------------
def a1d3_window(D, max_ab=2):
    labs = {D.identity()}
    for i in range(3):
        labs.add(D.T(i)); labs.add(D.D(i))
    for tile in range(6):
        for a in range(0, max_ab + 1):
            for b in range(0, max_ab + 1 - a):
                for k in (0, 1):
                    labs.add(D.canonicalise((tile, a, b, k)))
    return sorted(labs)


def check_a1d3_algebra(environment):
    from a1d3_kalg import A1D3KAlg
    D = A1D3KAlg()
    one = D.identity()
    checks = []
    T, Dd, chi = D.T, D.D, D.chi

    def cd(lab, k):  # χ_k-dressed label
        t, a, b, _ = D.canonicalise(lab)
        return D.canonicalise((t, a, b, k))

    def pw(gen, n):  # gen^n as a label
        t, a, b, k = D.canonicalise(gen)
        return D.canonicalise((t, a * n, b * n, k))
    ok = True
    for i in range(3):
        ok &= D.multiply(T(i), T(i + 1)) == elem({one: {0: 1}, cd(Dd(i), 1): {-1: 1}, pw(Dd(i), 2): {-2: 1}})
        ok &= D.multiply(T(i + 1), T(i)) == elem({one: {0: 1}, cd(Dd(i), 1): {1: 1}, pw(Dd(i), 2): {2: 1}})
    checks.append(check("T_i T_{i+1} = 1 + q^-1 χ_1 D_i + q^-2 D_i^2 and T_{i+1} T_i = 1 + q χ_1 D_i + q^2 D_i^2, all i", ok))
    ok = all(D.multiply(Dd(i), Dd(i + 1)) == elem({one: {0: 1}, T(i + 1): {-1: 1}}) and D.multiply(Dd(i + 1), Dd(i)) == elem({one: {0: 1}, T(i + 1): {1: 1}}) for i in range(3))
    checks.append(check("D_i D_{i+1} = 1 + q^-1 T_{i+1} and D_{i+1} D_i = 1 + q T_{i+1}, all i", ok))
    ok = all(D.multiply(T(i), Dd(i + 1)) == elem({chi(1): {0: 1}, Dd(i): {-1: 1}, Dd(i - 1): {1: 1}}) and D.multiply(Dd(i + 1), T(i)) == elem({chi(1): {0: 1}, Dd(i): {1: 1}, Dd(i - 1): {-1: 1}}) for i in range(3))
    checks.append(check("T_i D_{i+1} = χ_1 + q^-1 D_i + q D_{i-1} and D_{i+1} T_i = χ_1 + q D_i + q^-1 D_{i-1}, all i", ok))

    def times_q(el: Element, e: int) -> Element:
        return Element({lab: LaurentPoly({k + e: v for k, v in cf._coeffs.items()}) for lab, cf in el.terms.items()})
    ok = all(D.multiply(T(i), Dd(i)) == times_q(D.multiply(Dd(i), T(i)), -2) and D.multiply(T(i), Dd(i - 1)) == times_q(D.multiply(Dd(i - 1), T(i)), 2) for i in range(3))
    checks.append(check("T_i D_i = q^-2 D_i T_i and T_i D_{i-1} = q^2 D_{i-1} T_i, all i", ok))
    # canonical basis: q^{ab} T_i^a D_i^b (type I) and q^{-ab} T_i^a D_{i-1}^b (type II)
    ok1 = ok2 = True
    for i in range(3):
        for a in range(1, 4):
            for b in range(1, 4):
                x = D.multiply(pw(T(i), a), pw(Dd(i), b))
                ok1 &= (len(x.terms) == 1 and list(x.terms.values())[0] == LaurentPoly({-a * b: 1}) and D.canonicalise(list(x.terms)[0])[1:3] == (a, b))
                y = D.multiply(pw(T(i), a), pw(Dd(i - 1), b))
                ok2 &= (len(y.terms) == 1 and list(y.terms.values())[0] == LaurentPoly({a * b: 1}) and D.canonicalise(list(y.terms)[0])[1:3] == (a, b))
    checks.append(check("canonical monomials: T_i^a D_i^b = q^-ab L (L = q^ab T_i^a D_i^b) and T_i^a D_{i-1}^b = q^ab L' (L' = q^-ab T_i^a D_{i-1}^b), a, b <= 3, all i", ok1 and ok2))
    ok = all(D.multiply(chi(1), lab) == elem({cd(lab, 1): {0: 1}}) and D.multiply(lab, chi(1)) == elem({cd(lab, 1): {0: 1}}) for lab in (T(0), Dd(0), pw(T(1), 2), D.canonicalise((0, 1, 1, 0))))
    ok &= D.multiply(chi(1), chi(1)) == elem({one: {0: 1}, chi(2): {0: 1}}) and D.multiply(chi(1), chi(2)) == elem({chi(1): {0: 1}, chi(3): {0: 1}})
    checks.append(check("dressing by flavour characters: χ_1·L = the χ_1-dressed label (central), χ_1 χ_1 = χ_0 + χ_2, χ_1 χ_2 = χ_1 + χ_3", ok))
    W = a1d3_window(D, 2)
    ok_rho = all(D.rho(T(i)) == T(i + 1) and D.rho(Dd(i)) == Dd(i + 1) for i in range(3)) and all(D.rho(chi(k)) == chi(k) for k in range(4))
    ok_ord = all(D.rho(D.rho(D.rho(l))) == l for l in W) and any(D.rho(l) != l for l in W) and any(D.rho(D.rho(l)) != l for l in W)
    checks.append(check("rho(T_i) = T_{i+1}, rho(D_i) = D_{i+1}, rho fixes χ_k; rho has order exactly 3 on the window", ok_rho and ok_ord))
    res = {}
    for _ax, name, arity, takesK in F_VERIFIERS:
        fn = getattr(D, name)
        if arity == 0:
            it = [()]
        elif arity == 1:
            it = [(a,) for a in W]
        elif arity == 2:
            it = itertools.product(W, W)
        elif arity == "R":
            R = D.coefficient_ring()
            it = [(R.basis_element(j),) for j in (1, 2)]
        else:
            it = itertools.product(W[:6], W[:6], W[:6])
        f = sum(1 for args in it if not (fn(*args, K=6) if takesK else fn(*args)))
        res[name] = f
    checks.append(check(f"F1-F5 verifiers on the window ({len(W)} labels, a+b <= 2, χ-dressing 0/1), K = 6", all(v == 0 for v in res.values()), "; ".join(f"{k} fails {v}" for k, v in res.items() if v)))
    checks.append(check("negative control: the altered relation D_i D_{i+1} = 1 + q^-2 T_{i+1} is detected", D.multiply(Dd(0), Dd(1)) != elem({one: {0: 1}, T(1): {-2: 1}})))
    try:
        from a1d3_object import a1d3_object
        O = a1d3_object()
        keys = list(O.keys())
        src = "standalone"
        std = [D.identity(), T(0), T(1), T(2), Dd(0), Dd(1), Dd(2), chi(1)]
        samples = {k: [] for k in keys}
        for lab in std:
            imgs = {}
            for k in keys:
                if k == src:
                    imgs[k] = lab
                    continue
                el = O.transport(lab, src, k)
                if len(el.terms) != 1:
                    imgs = None
                    break
                imgs[k] = next(iter(el.terms))
            if imgs is not None:
                for k in keys:
                    samples[k].append(imgs[k])
        pw_ = O.verify_pairwise(samples, pairs=True, trace_K=6)
        bad = [(e, [c for c, v in r.items() if not v]) for e, r in pw_.items() if not all(r.values())]
        checks.append(check(f"the presentations are one algebra: KAlgebraIso witnesses on {len(pw_)} edges (realizations {keys})", bool(pw_) and not bad, f"failing: {bad}" if bad else f"{len(samples[src])} matched samples"))
    except Exception as ex:  # noqa: BLE001
        checks.append(check("the presentations are one algebra (a1d3_object)", False, f"{type(ex).__name__}: {ex}"))
    return {"checks": checks, "population": {"generators": 6, "window": f"{len(W)} labels", "K": 6},
            "controls": {"positive": "the printed relations reproduced on all i", "negative": "altered relation detected"},
            "notes": "The stand-alone class is `src/cone/a1d3_kalg.py::A1D3KAlg` over R(SU(2)) (SU2ZPlusRing); tiles 0..5 are the six cones (T_i, D_i) and (T_i, D_{i-1}).", "inputs": []}


def weyl_kac_inverse(K):
    """1/D, D = (q^2;q^2)_∞ (μ^2 q^2; q^2)_∞ (μ^-2 q^2; q^2)_∞, as a series in q with μ-Laurent coefficients."""
    return inv_qpoch_inf2(2, (0,), K, 1) * inv_qpoch_inf2(2, (2,), K, 1) * inv_qpoch_inf2(2, (-2,), K, 1)


def a1d3_closed_forms(K):
    invD = weyl_kac_inverse(K)
    Z = S2({}, K, 1)
    n = 0
    tr1 = Z
    while 3 * n * (n + 1) <= K:
        tr1 = tr1 + chi_mu(2 * n, K).shift(3 * n * (n + 1)).scale((-1) ** n)
        n += 1
    trD = Z
    p = 1
    while p * (3 * p - 1) - 1 <= K:
        trD = trD + (S2.term(p * (3 * p - 1) - 1, (0,), K, 1) - S2.term(p * (3 * p + 1) - 1, (0,), K, 1)) * chi_mu(2 * p - 1, K).scale((-1) ** p)
        p += 1
    trT = Z
    m = 0
    while m * (3 * m + 1) - 1 <= K:   # the smallest of the three exponents (m(3m+1) < 3m(m+1) < (m+1)(3m+2))
        trT = trT + (S2.term(3 * m * (m + 1) - 1, (0,), K, 1) - S2.term(m * (3 * m + 1) - 1, (0,), K, 1) - S2.term((m + 1) * (3 * m + 2) - 1, (0,), K, 1)) * chi_mu(2 * m, K).scale((-1) ** m)
        m += 1
    return (tr1 * invD).trunc(K), (trT * invD).trunc(K), (trD * invD).trunc(K)


def check_a1d3_traces(environment):
    from a1d3_kalg import A1D3KAlg
    D = A1D3KAlg()
    K = 14
    checks = []
    tr1, trT, trD = a1d3_closed_forms(K)
    printed = S2.one(K, 1) + chi_mu(2, K).shift(2) + (chi_mu(0, K) + chi_mu(2, K) + chi_mu(4, K)).shift(4)
    checks.append(check("positive control: the printed expansion Tr 1 = 1 + χ_2 q^2 + (χ_0 + χ_2 + χ_4) q^4 + ... (independent series)", tr1.trunc(5).eq_to(printed.trunc(5), 5)))
    c1 = rps_to_mu(D.trace(D.identity(), K), K)
    cT = rps_to_mu(D.trace(D.T(0), K), K)
    cD = rps_to_mu(D.trace(D.D(0), K), K)
    checks.append(check(f"Tr 1 = (1/D) Σ (-1)^n χ_2n q^(3n(n+1)) through q^{K}", c1.eq_to(tr1, K), f"first difference q^{c1.first_diff(tr1, K)}" if not c1.eq_to(tr1, K) else ""))
    checks.append(check(f"Tr D_0 = (1/D) Σ_p (-1)^p χ_2p-1 (q^(p(3p-1)-1) - q^(p(3p+1)-1)) through q^{K}", cD.eq_to(trD, K), f"first difference q^{cD.first_diff(trD, K)}" if not cD.eq_to(trD, K) else ""))
    checks.append(check(f"Tr T_0 = (1/D) Σ_m (-1)^m χ_2m (q^(3m(m+1)-1) - q^(m(3m+1)-1) - q^((m+1)(3m+2)-1)) through q^{K}", cT.eq_to(trT, K), f"first difference q^{cT.first_diff(trT, K)}" if not cT.eq_to(trT, K) else ""))
    checks.append(check("Z_3 invariance of the trace: Tr T_i and Tr D_i independent of i", all(rps_to_mu(D.trace(D.T(i), K), K).eq_to(cT, K) and rps_to_mu(D.trace(D.D(i), K), K).eq_to(cD, K) for i in (1, 2))))
    checks.append(check("the sharp leading constraint: Tr T_0 = -q + O(q^2), [q^2] Tr T_0 = 0, [q^1] Tr D_0 = -χ_1 (the note's C7)", cT.c.get(1) == {(0,): -1} and 2 not in cT.c and cD.c.get(1) == {(-1,): -1, (1,): -1}))
    checks.append(check("negative control: the Weyl-Kac denominator with (μ^2 q^2;q^2) omitted is detected", not c1.eq_to((tr1 * qpoch_inf2(2, (2,), K, 1)).trunc(K), K)))
    return {"checks": checks, "population": {"K": K, "traces": "1, T_0, D_0 (and the Z_3 images)"},
            "controls": {"positive": "the printed expansion of Tr 1", "negative": "a factor of the denominator omitted is detected"},
            "notes": "Closed forms and the Weyl-Kac denominator recomputed independently as series in q with μ-Laurent coefficients; the class's SU(2)-character coefficients are expanded in μ.", "inputs": []}


def check_a1d3_miracle(environment):
    """The 2x3 matrix of linear constraints: the rows (A_a, B_a, C_a) of Tr(T_0^a) and (A'_a, B'_a, C'_a) of
    Tr(T_0^a D_0) over (Tr 1, Tr T_0, Tr D_0), read off the layer-1 reduction; each row normalised by its
    minimal q-power; the three 2x2 minors (the Cramer cross product), divided by (μ^2 q^2;q^2)(μ^-2 q^2;q^2),
    against the elementary traces as a grows.  MEASURED here for the first time (the trace-miracle puzzle lists the heptagon
    and nonagon cases; a1d3_trace_recursions.md does not address the minors)."""
    from a1d3_kalg import A1D3KAlg
    D = A1D3KAlg()
    K = 24
    checks = []
    tr = {"1": rps_to_mu(D.trace(D.identity(), K), K), "T": rps_to_mu(D.trace(D.T(0), K), K), "D": rps_to_mu(D.trace(D.D(0), K), K)}
    factor = qpoch_inf2(2, (2,), K, 1) * qpoch_inf2(2, (-2,), K, 1)

    def row(label):
        red = D.trace_layer1(label)
        r = [qdict_to_mu(red.get(key, {}), K + 200) for key in (('Tr_1',), ('Tr_T',), ('Tr_D',))]
        emin = min(x.val() for x in r if x.val() is not None)
        return [x.shift(-emin) for x in r]
    results = {}
    for a in range(1, 9):
        r1 = row((0, a, 0, 0))            # Tr(T_0^a)
        r2 = row((0, a, 1, 0))            # Tr(q^a T_0^a D_0): the q-power drops out in the normalisation
        A, B, C = r1
        A2, B2, C2 = r2
        minors = {"1": B * C2 - C * B2, "T": C * A2 - A * C2, "D": A * B2 - B * A2}   # r1 x r2, Cramer's ray
        red = {k: series_div2(m.trunc(K), factor, K) for k, m in minors.items()}
        # normalise the ray by the leading term of the "1" component so that it starts with +1·q^0
        v = red["1"].val()
        lead = red["1"].c.get(v, {})
        sign = 1 if sum(lead.values()) > 0 else -1
        red = {k: m.shift(-v).scale(sign) for k, m in red.items()}
        orders = {k: red[k].first_diff(tr[k], K) for k in red}
        results[a] = (orders, {k: red[k].trunc(6) for k in red})
    grows = all((results[a][0][k] is None) or (results[a - 1][0][k] is None) or results[a][0][k] >= results[a - 1][0][k] for a in range(3, 9) for k in ("1", "T", "D"))
    deep = all(results[8][0][k] is None or results[8][0][k] >= 12 for k in ("1", "T", "D"))
    summary = "; ".join(f"a={a}: 1->{o['1']}, T->{o['T']}, D->{o['D']}" for a, (o, _s) in results.items())
    checks.append(check("the minors of (Tr T_0^a, Tr T_0^a D_0), divided by (μ^2q^2;q^2)(μ^-2q^2;q^2), approach (Tr 1, Tr T_0, Tr D_0): first differing powers grow with a", grows, summary))
    checks.append(check("at a = 8 all three components agree through q^11 at least", deep, f"a=8 leading terms: {results[8][1]}"))
    r2 = row((0, 2, 0, 0))
    r11 = row((0, 1, 1, 0))
    exp_r2 = [{1: {(0,): 1}, 3: {(0,): 1}}, {0: {(0,): 1}}, {2: {(1,): 1, (-1,): 1}}]          # q^3 (q^-2+1, q^-3, χ_1 q^-1)
    exp_r11 = [{1: {(1,): 1, (-1,): 1}}, {}, {0: {(0,): 1}, 2: {(0,): 1}}]                      # q^2 (χ_1 q^-1, 0, q^-2+1)
    checks.append(check("positive control: the rows of Tr T_0^2 and Tr(q T_0 D_0) are the note's (q^-2+1, q^-3, χ_1 q^-1) and (χ_1 q^-1, 0, q^-2+1), normalised", [x.c for x in r2] == exp_r2 and [x.c for x in r11] == exp_r11, f"rows: {r2} | {r11}"))
    return {"checks": checks, "population": {"a": "1..8", "K": K},
            "controls": {"positive": "the a = 2 rows match the trace-recursions note", "negative": "n.a. (a measured convergence)"},
            "notes": "Measured, not derived: " + summary, "inputs": []}


ADAPTERS = {
    "sec:uqsl2/algebra": check_uqsl2_algebra,
    "sec:uqsl2/trace": check_uqsl2_trace,
    "sec:uqsl2/recursion": check_uqsl2_recursion,
    "sec:a1d3/algebra": check_a1d3_algebra,
    "sec:a1d3/traces": check_a1d3_traces,
    "rem:a1d3-miracle": check_a1d3_miracle,
}
