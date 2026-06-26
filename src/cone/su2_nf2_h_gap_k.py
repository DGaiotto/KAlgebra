"""Closed-form `H_a · H_b` for SU(2)+N_f=2, the spine-free analogue of
`su2_nf1_h_gap_k` lifted from U(1)_F to **Spin(4) = SU(2)_L × SU(2)_R**.

Structure (validated against the BPS oracle `su2_nf2_kalgebra`):

H-tower piecewise map (clean index n -> BPS charge), rho-closed shift 4-N_f=2:
    H_n = (1,  n,   0, 0)   for n <= 0
    H_n = (-1, 2-n, 0, 0)   for n >= 1
Wilson fundamental W_1 = (0,-1,0,0); Wilson tower W_a = SU(2) Chebyshev.

Clebsch:  w_1 * H_n = q*H_{n-1} + q^{-1}*H_{n+1} + eps_n
    eps_n = chi_{R-fund}   (n even)
    eps_n = chi_{L-fund}   (n odd)
(the two SU(2) matter doublets alternate by parity; N_f=1's eps_n=1/mu is
the rank-1 shadow).

The whole `su2_nf1_h_gap_k` recursion ports verbatim under the substitution
mu -> {chi_L, chi_R}: each matter insertion contributes the SU(2)_L or
SU(2)_R fundamental character by parity, and the Spin(4) ring's Clebsch
handles tensor products automatically.

Clean seeds are gauge cone labels `(m, e)` (m = magnetic multiplicity, e =
electric index) with `RLaurent`-over-Spin(4) coefficients; the flavour
character rides in the coefficient (R-form), expanded to Cartan weights only
at the tropical boundary.
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from zplus_ring import SU2ZPlusRing, RLaurent, RElement
from tensor_zplus_ring import TensorZPlusRing


# Spin(4) = SU(2)_L x SU(2)_R character ring.
R_SP4 = TensorZPlusRing(SU2ZPlusRing(), SU2ZPlusRing())


# ---- RLaurent[Spin4] helpers ----------------------------------------

def rl(qpow: int = 0, key=(0, 0)) -> RLaurent:
    """q^qpow * chi_key as RLaurent over Spin(4)."""
    return RLaurent(R_SP4, {qpow: R_SP4.basis_element(key)})


def rl_zero() -> RLaurent:
    return RLaurent(R_SP4, {})


def chi_L():
    return R_SP4.basis_element((1, 0))


def chi_R():
    return R_SP4.basis_element((0, 1))


def eps_char(n: int):
    """Matter character inserted at H_n: R-fund (n even), L-fund (n odd)."""
    return R_SP4.basis_element((0, 1)) if n % 2 == 0 else R_SP4.basis_element((1, 0))


# ---- Clebsch w_1 * H_n ----------------------------------------------

def w1_H(n: int) -> dict:
    """w_1 * H_n = q*H_{n-1} + q^{-1}*H_{n+1} + eps_n*(identity).
    Returns {clean_seed (m,e): RLaurent[Spin4]}.
    """
    out: dict = {}
    out[(1, n - 1)] = rl(1)
    out[(1, n + 1)] = out.get((1, n + 1), rl_zero()) + rl(-1)
    out[(0, 0)] = out.get((0, 0), rl_zero()) + RLaurent(R_SP4, {0: eps_char(n)})
    return out


# ---- H*H base cases (gap 0..3), oracle-validated ---------------------

def _gap0(a: int) -> dict:
    return {(2, 2 * a): rl(0)}


def _gap1(a: int) -> dict:
    return {(2, 2 * a + 1): rl(1)}


def _gap2(a: int) -> dict:
    """q^2 M_{2(a+1)} + q*chi_{doublet(a)}*H_{a+1} + 1."""
    return {
        (2, 2 * (a + 1)): rl(2),
        (1, a + 1): RLaurent(R_SP4, {1: eps_char(a)}),
        (0, 0): rl(0),
    }


def _gap3(a: int) -> dict:
    """q^3 M_{2a+3} + q^{-1} W_1 + q*(two bridge doublets) + (L*R vector)."""
    bridge = a + 1 if (a + 1) % 2 == 1 else a + 2
    nonbridge = a + 2 if (a + 1) % 2 == 1 else a + 1
    return {
        (2, 2 * a + 3): rl(3),
        (0, 1): rl(-1),                                   # W_1 = seed (0,1)
        (1, bridge): RLaurent(R_SP4, {1: eps_char(bridge)}),
        (1, nonbridge): RLaurent(R_SP4, {1: eps_char(nonbridge)}),
        (0, 0): RLaurent(R_SP4, {0: chi_L() * chi_R()}),  # SO(4) vector
    }


# ---- H_a * H_b via downward W_1 walk (a <= b) -----------------------

_cache: dict = {}


def h_mul_h(a: int, b: int) -> dict:
    """`H_a * H_b` (a <= b) in clean (m,e) seeds with RLaurent[Spin4] coefs.
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
        # H_a H_b = q^{-1} w_1*(H_{a+1} H_b) - q^{-2} H_{a+2} H_b - q^{-1} eps_{a+1} H_b
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
        eps = RLaurent(R_SP4, {0: eps_char(a + 1)})
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
        alpha, beta = seed_to_pair(e)
        phase = -(beta - alpha) if alpha != beta else 0
        wHa = w1_mul_seed((1, alpha))
        lit: dict = {}
        for s, c in wHa.items():
            for s2, c2 in mul_by_H(s, beta).items():
                lit[s2] = lit.get(s2, rl_zero()) + c * c2
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
        elif abs(n - beta) <= 1:
            fwd = h_mul_h(beta, n)
            res = {k: rl(2 * (beta - n)) * v for k, v in fwd.items()
                   if not v.is_zero()}
        else:
            raise NotImplementedError(f"cross reverse H_{n}*H_{beta}")
    elif m == 0:
        if e == 0:
            res = {(1, beta): rl(0)}
        elif e == 1:
            res = {(1, beta - 1): rl(-1), (1, beta + 1): rl(1),
                   (0, 0): RLaurent(R_SP4, {0: eps_char(beta)})}
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
    return (1, n, 0, 0) if n <= 0 else (-1, 2 - n, 0, 0)


def M_trop(e: int) -> tuple:
    """m=2 cone monomial M_e -> tropical (validated piecewise)."""
    if e <= 0:
        return (2, e, 0, 0)
    if e == 1:
        return (0, 1, 0, 0)
    return (-2, 4 - e, 0, 0)


def gauge_trop(m: int, e: int) -> tuple:
    if m == 0:
        return (0, 0, 0, 0) if e == 0 else (0, -e, 0, 0)   # Wilson chi_e
    if m == 1:
        return Hbps(e)
    if m == 2:
        return M_trop(e)
    raise NotImplementedError(f"gauge_trop m={m} (higher cone monomials "
                              f"reached via the literal reducer)")


__all__ = [
    "R_SP4", "rl", "rl_zero", "chi_L", "chi_R", "eps_char",
    "w1_H", "h_mul_h", "w1_mul_seed", "mul_by_H", "seed_to_pair",
    "Hbps", "M_trop", "gauge_trop", "clear_cache",
]
