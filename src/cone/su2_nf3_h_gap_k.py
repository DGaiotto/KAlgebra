"""Closed-form `H_a · H_b` for SU(2)+N_f=3, the SU(4) analogue of
`su2_nf2_h_gap_k` (Spin(4)→SU(4); the matter doublets become the SU(4)
spinors 4 / 4̄).

Structure (validated against a BPS-quiver oracle, a derivation not
included in this repository):

H-tower piecewise map (clean index n → BPS charge), ρ-closed shift 4−N_f=1
(`ρ(H_n)=H_{n-1}`):
    H_n = (-n,   1-2n, 0,0,0)   for n >= 0
    H_n = (n+1,  2n+1, 0,0,0)   for n <= -1
Wilson fundamental W_1 = (-1,-2,0,0,0); Wilson tower W_a = SU(2) Chebyshev.
Cone monomials M_e = (-e, 2-2e, 0,0,0).

Clebsch:  w_1 · H_n = q·H_{n-1} + q^{-1}·H_{n+1} + ε_n
    ε_n = χ_4   = (1,0,0)   (n even)
    ε_n = χ_4̄   = (0,0,1)   (n odd)
(the two SU(4) matter spinors alternate by parity; N_f=2's L/R doublets are
the SO(4) shadow, N_f=1's ε_n=1/μ the rank-1 shadow.)

ρ acts on the SU(4) flavour character by **complex conjugation** (star,
4↔4̄), consistent with ε_{n-1}=star(ε_n).  All gap base cases for general a
follow from the a=0 forms by ρ-covariance (`H_aH_b = ρ^{-a}(H_0H_{b-a})`):
ρ^{-a} shifts the H index by +a, the M index by +2a, fixes W_e and the
self-dual χ_6/χ_15/χ_1, and applies star^a to the spinor characters.

Clean seeds are gauge cone labels `(m, e)` (m = magnetic multiplicity, e =
electric index) with `RLaurent`-over-SU(4) coefficients; the flavour
character rides in the coefficient (R-form), expanded to Cartan weights only
at the tropical boundary.
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from zplus_ring import SU4ZPlusRing, RLaurent, RElement


# SU(4) flavour character ring.
R_SU4 = SU4ZPlusRing()


# ---- RLaurent[SU4] helpers ------------------------------------------

def rl(qpow: int = 0, key=(0, 0, 0)) -> RLaurent:
    """q^qpow * χ_key as RLaurent over SU(4)."""
    return RLaurent(R_SU4, {qpow: R_SU4.basis_element(key)})


def rl_zero() -> RLaurent:
    return RLaurent(R_SU4, {})


# Named SU(4) characters appearing in the base cases.
CHI_4 = R_SU4.basis_element((1, 0, 0))     # fundamental 4
CHI_4B = R_SU4.basis_element((0, 0, 1))    # antifundamental 4̄
CHI_6 = R_SU4.basis_element((0, 1, 0))     # Λ²4 = SO(6) vector (self-dual)
CHI_15 = R_SU4.basis_element((1, 0, 1))    # adjoint (self-dual)
CHI_1 = R_SU4.basis_element((0, 0, 0))     # trivial


def _star(relt: RElement) -> RElement:
    """Complex conjugation on an SU(4) character: χ_(p,q,r) ↦ χ_(r,q,p)."""
    out: dict = {}
    for (p, q, r), c in relt.terms.items():
        k = (r, q, p)
        out[k] = out.get(k, 0) + c
    return RElement(R_SU4, {k: v for k, v in out.items() if v})


def _star_pow(relt: RElement, a: int) -> RElement:
    """star^a applied to a character (star is an involution)."""
    return relt if a % 2 == 0 else _star(relt)


def eps_char(n: int) -> RElement:
    """Matter character inserted at H_n: 4 (n even), 4̄ (n odd)."""
    return CHI_4 if n % 2 == 0 else CHI_4B


# ---- Clebsch w_1 * H_n ----------------------------------------------

def w1_H(n: int) -> dict:
    """w_1 * H_n = q*H_{n-1} + q^{-1}*H_{n+1} + ε_n*(identity).
    Returns {clean_seed (m,e): RLaurent[SU4]}.
    """
    out: dict = {}
    out[(1, n - 1)] = rl(1)
    out[(1, n + 1)] = out.get((1, n + 1), rl_zero()) + rl(-1)
    out[(0, 0)] = out.get((0, 0), rl_zero()) + RLaurent(R_SU4, {0: eps_char(n)})
    return out


# ---- H*H base cases (gap 0..3), oracle-validated (ρ-covariant in a) --

def _gap0(a: int) -> dict:
    """H_a² = M_{2a} (coeff 1)."""
    return {(2, 2 * a): rl(0)}


def _gap1(a: int) -> dict:
    """q M_{2a+1} + 1  (the identity term is the N_f=3 matter onset)."""
    return {(2, 2 * a + 1): rl(1), (0, 0): rl(0)}


def _gap2(a: int) -> dict:
    """q² M_{2a+2} + q·star^a(4)·H_{a+1} + q^{-1} W_1 + χ_6·1."""
    return {
        (2, 2 * a + 2): rl(2),
        (1, a + 1): RLaurent(R_SU4, {1: _star_pow(CHI_4, a)}),
        (0, 1): rl(-1),
        (0, 0): RLaurent(R_SU4, {0: CHI_6}),
    }


def _gap3(a: int) -> dict:
    """q³ M_{2a+3} + q^{-2} W_2 + q·star^a(4)·H_{a+2} + q^{-1} χ_6·W_1
    + q·star^a(4̄)·H_{a+1} + (χ_15 + 2χ_1 + q²χ_1)·1."""
    return {
        (2, 2 * a + 3): rl(3),
        (0, 2): rl(-2),
        (1, a + 2): RLaurent(R_SU4, {1: _star_pow(CHI_4, a)}),
        (0, 1): RLaurent(R_SU4, {-1: CHI_6}),
        (1, a + 1): RLaurent(R_SU4, {1: _star_pow(CHI_4B, a)}),
        (0, 0): RLaurent(R_SU4, {0: CHI_15 + CHI_1 * 2, 2: CHI_1}),
    }


# ---- H_a * H_b via downward W_1 walk (a <= b) -----------------------

_cache: dict = {}


def h_mul_h(a: int, b: int) -> dict:
    """`H_a * H_b` (a <= b) in clean (m,e) seeds with RLaurent[SU4] coefs.
    Base cases gap 0..3; gap > 3 via the W_1-walk recursion."""
    if a > b:
        raise ValueError("h_mul_h requires a <= b")
    if (a, b) in _cache:
        return _cache[(a, b)]
    gap = b - a
    if gap == 0:
        r = _gap0(a)
    elif gap == 1:
        r = _gap1(a)
    elif gap == 2:
        r = _gap2(a)
    elif gap == 3:
        r = _gap3(a)
    else:
        # H_a H_b = q^{-1} w_1*(H_{a+1} H_b) - q^{-2} H_{a+2} H_b
        #           - q^{-1} eps_{a+1} H_b
        Hab1 = h_mul_h(a + 1, b)
        Hab2 = h_mul_h(a + 2, b)
        wterm: dict = {}
        for seed, c in Hab1.items():
            for s2, c2 in w1_mul_seed(seed).items():
                wterm[s2] = wterm.get(s2, rl_zero()) + c * c2
        r = {}
        for s, c in wterm.items():
            r[s] = r.get(s, rl_zero()) + rl(-1) * c
        for s, c in Hab2.items():
            r[s] = r.get(s, rl_zero()) - rl(-2) * c
        eps = RLaurent(R_SU4, {0: eps_char(a + 1)})
        r[(1, b)] = r.get((1, b), rl_zero()) - rl(-1) * eps
        r = {k: v for k, v in r.items() if not v.is_zero()}
    _cache[(a, b)] = r
    return r


# ---- w_1 * seed  and  seed * H_beta  (coupled induction) ------------

def seed_to_pair(e: int) -> tuple:
    """m=2 seed electric index e -> max-diagonal H-pair (alpha,beta)."""
    if e % 2 == 0:
        return (e // 2, e // 2)
    n = (e - 1) // 2
    return (n, n + 1)


def w1_mul_seed(seed: tuple) -> dict:
    """w_1 * L_seed for m in {0,1,2}."""
    m, e = seed
    if m == 0:
        # Wilson chi_e: w_1*chi_e = chi_{e-1}+chi_{e+1}; identity -> chi_1.
        out: dict = {}
        if e >= 1:
            out[(0, e - 1)] = rl(0)
        out[(0, e + 1)] = out.get((0, e + 1), rl_zero()) + rl(0)
        return out
    if m == 1:
        return w1_H(e)
    if m == 2:
        # w_1 · L_{(2,e)} with the CANONICAL M_e (peeled).  The literal
        # monomial factors as  H_alpha·H_beta = lead·M_e + tail  (tail = the
        # lower canonicals from h_mul_h, present for N_f≥3), so
        #   w_1·M_e = (w_1·(H_alpha H_beta) − w_1·tail) / lead,
        # with 1/lead = q^{phase}.  Omitting the −w_1·tail term (as the N_f=2
        # port did, where tail=0) over-counts by the matter-onset identity.
        alpha, beta = seed_to_pair(e)
        phase = -(beta - alpha) if alpha != beta else 0
        wHa = w1_mul_seed((1, alpha))
        lit: dict = {}
        for s, c in wHa.items():
            for s2, c2 in mul_by_H(s, beta).items():
                lit[s2] = lit.get(s2, rl_zero()) + c * c2
        full = h_mul_h(alpha, beta)
        for s, c in full.items():
            if s == (2, e):
                continue
            for s2, c2 in w1_mul_seed(s).items():
                lit[s2] = lit.get(s2, rl_zero()) - c * c2     # − w_1·tail
        return {k: rl(phase) * v for k, v in lit.items() if not v.is_zero()}
    raise NotImplementedError(f"w1_mul_seed m={m}")


_mc: dict = {}


def mul_by_H(seed: tuple, beta: int) -> dict:
    """L_seed * H_beta for m in {0,1}."""
    key = (seed, beta)
    if key in _mc:
        return _mc[key]
    m, e = seed
    if m == 1:
        n = e
        if n <= beta:
            res = h_mul_h(n, beta)
        else:
            # reverse: H_n·H_beta = bar(H_beta·H_n) = (q→q⁻¹) of the forward
            # product (bar is antimultiplicative and fixes the canonical H_n).
            # Holds for ANY gap — NOT q^{2c}·forward, which only matches when
            # the forward product has a single q-power (N_f=2 gap1).
            fwd = h_mul_h(beta, n)
            res = {k: RLaurent(R_SU4, {-q: r for q, r in v.coeffs.items()})
                   for k, v in fwd.items() if not v.is_zero()}
    elif m == 0:
        if e == 0:
            res = {(1, beta): rl(0)}
        elif e == 1:
            # W_1·H_beta = w_1·H_beta = w1_H(beta) (the left Clebsch; q at
            # H_{beta-1}).  (The N_f≤2 code never reached this branch, so its
            # swapped-q form went untested; the Wilson tower below is built by
            # LEFT w_1-multiplication, so the base must be the left Clebsch.)
            res = dict(w1_H(beta))
        else:
            r1 = mul_by_H((0, e - 1), beta)
            r2 = mul_by_H((0, e - 2), beta) if e >= 2 else {}
            wr: dict = {}
            for s, c in r1.items():
                for s2, c2 in w1_mul_seed(s).items():
                    wr[s2] = wr.get(s2, rl_zero()) + c * c2
            res = dict(wr)
            for s, c in r2.items():
                res[s] = res.get(s, rl_zero()) - c
            res = {k: v for k, v in res.items() if not v.is_zero()}
    else:
        raise NotImplementedError(f"mul_by_H m={m}")
    _mc[key] = res
    return res


def clear_cache() -> None:
    _cache.clear()
    _mc.clear()


# ---- clean (m,e) seed -> BPS tropical gauge charge ------------------

def Hbps(n: int) -> tuple:
    """H-tower clean index n -> BPS gauge charge (n1,n2) (piecewise, ρ-wall
    at -1/0)."""
    return (-n, 1 - 2 * n) if n >= 0 else (n + 1, 2 * n + 1)


def M_trop(e: int) -> tuple:
    """m=2 cone monomial M_e -> BPS gauge charge = sum of the H-pair's Hbps
    (= (-e, 2-2e) for e≥0; piecewise across the ρ-wall for e<0)."""
    base, rem = divmod(e, 2)
    ha, hb = Hbps(base), Hbps(base + 1)
    return ((2 - rem) * ha[0] + rem * hb[0], (2 - rem) * ha[1] + rem * hb[1])


def gauge_trop(m: int, e: int) -> tuple:
    """clean (m,e) gauge seed -> BPS gauge charge (n1,n2)."""
    if m == 0:
        return (0, 0) if e == 0 else (-e, -2 * e)   # Wilson chi_e
    if m == 1:
        return Hbps(e)
    if m == 2:
        return M_trop(e)
    return gauge_trop_ext(m, e)


def gauge_trop_ext(m: int, e: int) -> tuple:
    """clean (m,e) gauge seed -> tropical charge, ALL magnetic m≥0: the
    max-diagonal cone monomial H_b^{m-r}·H_{b+1}^r ((b,r)=divmod(e,m)) maps to
    the sum of its H-charges.  (m≤2 agrees with gauge_trop.)"""
    if m == 0:
        return (0, 0) if e == 0 else (-e, -2 * e)
    b, r = divmod(e, m)
    ha, hb = Hbps(b), Hbps(b + 1)
    return ((m - r) * ha[0] + r * hb[0], (m - r) * ha[1] + r * hb[1])


def trop_to_seed(n1: int, n2: int):
    """tropical charge -> clean (m,e), inverting gauge_trop_ext on the physical
    cone.  Upper chamber (H_n n≥0 + its cone monomials): (m,e)=(n2−2n1, −n1).
    Lower H-tower (H_n n≤−1, the ρ-image ray n2=2n1−1): (1, n1−1).  Returns None
    off both (e.g. the other-chamber g-vectors with n1>0)."""
    m, e = n2 - 2 * n1, -n1
    if m >= 0 and gauge_trop_ext(m, e) == (n1, n2):
        return (m, e)
    if n2 == 2 * n1 - 1:
        return (1, n1 - 1)
    return None


__all__ = [
    "R_SU4", "rl", "rl_zero", "eps_char", "CHI_4", "CHI_4B", "CHI_6",
    "CHI_15", "CHI_1", "_star", "_star_pow",
    "w1_H", "h_mul_h", "w1_mul_seed", "mul_by_H", "seed_to_pair",
    "Hbps", "M_trop", "gauge_trop", "gauge_trop_ext", "trop_to_seed",
    "clear_cache",
]
