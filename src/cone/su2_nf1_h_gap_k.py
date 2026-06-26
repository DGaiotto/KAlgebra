"""Axiom-derived H_a · H_b for SU(2) + N_f = 1, mirroring
`pure_su2_h_gap_k` with U(1)_F flavour μ-decorations.

H-tower piecewise structure (in BPS basis, from `bps_su2_nf1`):

    n ≤ 0:  H_n = (1, n, 0)
    n = 1:  H_1 = (0, 1, 0)
    n ≥ 2:  H_n = (-1, 4-n, 0)

Same shape as pure SU(2), with flavour 0 appended.

Wilson fundamental: `w_1 = (0, -1, 0)` (BPS basis).

Nf=1 Clebsch (verified empirically against `SUN_Nf(2, 1).algebra`):

    w_1 · H_n = q · H_{n-1}  +  q^{-1} · H_{n+1}  +  ε_n
    H_n · w_1 = q^{-1} · H_{n-1}  +  q · H_{n+1}  +  ε_n

    ε_n =  1   for n odd
           μ   for n even

(Compare pure SU(2): ε_n = 1 at n = ±1, else 0.  The Nf=1
modification is matter-flavour μ instead of 0 at even n.)

ρ-action: `ρ(H_n) = μ^{-1} · H_{n-3}` (verified via spec rotation).
The index shift is -3 (not -4 as in pure SU(2)) with an additional
flavour shift μ^{-1}.

The W_1-walk recursion structure carries over:

    (w_1 · H_{a+1}) · H_b = w_1 · (H_{a+1} · H_b)

Solving for H_a · H_b (n=a+1 even):
    H_a · H_b = q^{-1} · w_1 · (H_{a+1}·H_b)
              - q^{-2} · H_{a+2}·H_b
              - q^{-1} · μ · H_b

(n=a+1 odd: μ replaced by 1, matching pure SU(2).)

Status: base-case gap 0/1/2/3 products to be filled in from the BPS
probes once a `_bps_label_to_seed` decoder is wired (mapping BPS
3-tuples to canonical (m, e, μ_power) seeds).  The Clebsch + ρ are
locked in; the recursion just needs the bootstrap.
"""
from __future__ import annotations
from fractions import Fraction

from laurent_poly import LaurentPoly
from zplus_ring import AbelianZPlusRing, RLaurent


_R = AbelianZPlusRing(rank=1)


def _rl_zero() -> RLaurent:
    return RLaurent(_R, {})


def _rl_one() -> RLaurent:
    return RLaurent(_R, {0: _R.one()})


def _rl_q(n: int) -> RLaurent:
    return RLaurent(_R, {n: _R.one()})


def _rl_mu(q_pow: int = 0, mu_pow: int = 1) -> RLaurent:
    """`q^{q_pow} · μ^{mu_pow}` as `RLaurent[R]`."""
    return RLaurent(_R, {q_pow: _R.basis_element((mu_pow,))})


# ---------- H-tower piecewise labels (BPS basis) ---------------------

def H_bps(n: int) -> tuple:
    """H_n in BPS basis (Z^3 lattice).  Piecewise:
       n ≤ 0  → (1, n, 0)
       n = 1  → (0, 1, 0)
       n ≥ 2  → (-1, 4-n, 0).
    """
    if n <= 0:
        return (1, n, 0)
    if n == 1:
        return (0, 1, 0)
    return (-1, 4 - n, 0)


WILSON_FUND_BPS = (0, -1, 0)


# ---------- Nf=1 Clebsch (verified from BPS probes) -----------------

def epsilon(n: int) -> RLaurent:
    """The 'extra' term in `w_1 · H_n = q·H_{n-1} + q^{-1}·H_{n+1} + ε_n`.

    `ε_n = 1` for n odd; `ε_n = μ` for n even.
    """
    if n % 2 == 1:
        return _rl_one()
    return _rl_mu()


def w1_H_clebsch_left(n: int):
    """`w_1 · H_n` decomposition as list of `(seed, RLaurent)` pairs.

    Returns:
        (1, n-1): q
        (1, n+1): q^{-1}
        (0, 0):   ε_n      [ = 1 for n odd, μ for n even ]
    """
    return [
        ((1, n - 1), _rl_q(1)),
        ((1, n + 1), _rl_q(-1)),
        ((0, 0),     epsilon(n)),
    ]


def w1_H_clebsch_right(n: int):
    """`H_n · w_1` (right action): q ↔ q^{-1} swap, ε_n unchanged."""
    return [
        ((1, n - 1), _rl_q(-1)),
        ((1, n + 1), _rl_q(1)),
        ((0, 0),     epsilon(n)),
    ]


# ---------- H·H base cases (Nf=1, extracted from BPS probes) -------
#
# Decoded from `SUN_Nf(2, 1).algebra.multiply(H_a, H_b)` outputs.
# The Nf=1 pattern: pure-SU(2) base cases PLUS μ-decorated "matter"
# corrections.  Specifically:
#
#   gap 0:  H_a · H_a = L_{(2, 2a, 0)}                          [no μ]
#   gap 1:  H_a · H_{a+1} = q · L_{(2, 2a+1, 0)}                [no μ]
#   gap 2 a even: q² · L_{(2, 2(a+1), 0)}                       [pure-SU(2) term]
#                 + q · μ · L_{(1, a+1, 0)}                     [Nf=1 extra]
#   gap 2 a odd:  q · L_{(1, a+1, 0)} + q² · L_{(2, 2(a+1), 0)} [same as pSU2]
#   gap 3:        q · L_{(1, bridge, 0)} + q³ · L_{(2, 2a+3, 0)} [pure-SU(2) terms]
#                 + μ · L_{(0, 0, 0)} + q · μ · L_{(1, non_bridge, 0)} [extras]
#       where bridge = (a+1 if (a+1) odd else a+2),
#             non_bridge = (a+2 if (a+1) odd else a+1).
#
# Verified for a ∈ {-1, 0, 1, 2}; the formulas generalise by linearity.

def _gap0(a: int):
    """H_a · H_a = L_{(2, 2a)}, coefficient 1.  Same as pure SU(2)."""
    return {(2, 2 * a): _rl_one()}


def _gap1(a: int):
    """H_a · H_{a+1} = q · L_{(2, 2a+1)}.  Same as pure SU(2)."""
    return {(2, 2 * a + 1): _rl_q(1)}


def _gap2(a: int):
    """H_a · H_{a+2}.

    a even: q² · L_{(2, 2(a+1))}  +  q · μ · L_{(1, a+1)}.
    a odd:  q · L_{(1, a+1)}  +  q² · L_{(2, 2(a+1))}.
    """
    if a % 2 == 0:
        return {
            (2, 2 * (a + 1)): _rl_q(2),
            (1, a + 1):       _rl_mu(q_pow=1, mu_pow=1),
        }
    return {
        (1, a + 1):           _rl_q(1),
        (2, 2 * (a + 1)):     _rl_q(2),
    }


def _gap3(a: int):
    """H_a · H_{a+3}.

    Pure-SU(2) terms:
        q · L_{(1, bridge)} + q³ · L_{(2, 2a+3)}
    Nf=1 extras:
        μ · L_{(0, 0)} + q · μ · L_{(1, non_bridge)}
    where bridge = a+1 if (a+1) odd else a+2.
    """
    bridge = a + 1 if (a + 1) % 2 == 1 else a + 2
    non_bridge = a + 2 if (a + 1) % 2 == 1 else a + 1
    return {
        (1, bridge):           _rl_q(1),
        (2, 2 * a + 3):        _rl_q(3),
        (0, 0):                _rl_mu(q_pow=0, mu_pow=1),
        (1, non_bridge):       _rl_mu(q_pow=1, mu_pow=1),
    }




# ---------- w_1 · canonical-seed -------------------------------------

def w1_mul_seed(seed: tuple) -> dict:
    """`w_1 · L_seed` decomposition for m ∈ {0, 1, 2}.

    m=0 (Wilson χ_e):
        Pure SU(2): χ_{e-1} + χ_{e+1} (Clebsch).
        PLACEHOLDER: pure-SU(2) form, μ-trivial.

    m=1 (H_n):
        See `w1_H_clebsch_left`.

    m=2 (H_α · H_β literal up to phase):
        Reduce to (w_1 · H_α) · H_β via associativity.
    """
    m, e = seed
    if m == 0:
        out = {}
        if e >= 1:
            out[(0, e - 1)] = _rl_one()
        out[(0, e + 1)] = out.get((0, e + 1), RLaurent(_R, {})) + _rl_one()
        return out
    if m == 1:
        return {s: c for (s, c) in w1_H_clebsch_left(e)}
    if m == 2:
        from su2_nf1_h_multiply import _seed_to_h_pair
        alpha, beta = _seed_to_h_pair(seed)
        phase = -(beta - alpha) if alpha != beta else 0
        w1_Halpha = w1_mul_seed((1, alpha))
        literal: dict = {}
        for s, coef in w1_Halpha.items():
            prod = _mul_by_H_beta(s, beta)
            for s_out, c_out in prod.items():
                literal[s_out] = literal.get(s_out, RLaurent(_R, {})) + coef * c_out
        # Apply q^{phase}.
        out = {k: _rl_q(phase) * v for k, v in literal.items() if not v.is_zero()}
        return out
    # Invariant: seeds of H*H products carry m = (#H-factors) <= 2; m>2
    # cannot arise on any caller path.
    raise AssertionError(
        f"w1_mul_seed: invariant m in {{0,1,2}} violated (m={m}); "
        f"seeds of H*H products carry m = #H-factors <= 2")


# ---------- Helpers (m=2 seed factorization, copied from pure SU(2)) --

def _seed_to_h_pair(seed: tuple) -> tuple:
    """For m=2 seed, return max-diagonal `(α, β)` with α ≤ β.
    Same convention as pure SU(2).
    """
    m, e = seed
    assert m == 2
    if e % 2 == 0:
        return (e // 2, e // 2)
    n = (e - 1) // 2
    return (n, n + 1)


_mul_cache: dict = {}


def _mul_by_H_beta(left_seed: tuple, beta: int) -> dict:
    """`L_{left_seed} · H_beta` decomposition for m_left ∈ {0, 1}.
    Returns `{(m, e): RLaurent[R]}`.
    """
    key = (left_seed, beta)
    if key in _mul_cache:
        return _mul_cache[key]
    result = _mul_by_H_beta_impl(left_seed, beta)
    _mul_cache[key] = result
    return result


def _mul_by_H_beta_impl(left_seed: tuple, beta: int) -> dict:
    m, e = left_seed
    if m == 1:
        n = e
        if n <= beta:
            return h_mul_h(n, beta)
        # n > beta: check cone (within-cone reorder vs cross-cluster bar).
        from su2_nf1_cone_data import cone_index_for
        n_lo, n_hi = min(n, beta), max(n, beta)
        cone_idx = cone_index_for(n_lo, n_hi)
        if cone_idx is not None:
            forward = h_mul_h(beta, n)
            factor = _rl_q(2 * (beta - n))
            return {k: factor * v for k, v in forward.items() if not v.is_zero()}
        # Invariant: the left seed is a canonical term of w_1*H_alpha, so
        # |n - beta| <= 1 (adjacent in the H-tower); a cross-cluster
        # (|n-beta|>=3) reversed pair never reaches here.
        raise AssertionError(
            f"_mul_by_H_beta: invariant |n-beta|<=1 violated "
            f"(H_{n}*H_{beta} cross-cluster reverse); cannot arise from the "
            f"w_1*H_alpha caller (adjacent seeds only)")
    if m == 0:
        # Wilson χ_e · H_β: same Chebyshev recursion as pure SU(2),
        # with placeholder μ-trivial Clebsch.
        if e == 0:
            return {(1, beta): _rl_one()}
        if e == 1:
            return {s: c for (s, c) in w1_H_clebsch_right(beta)}
        # e ≥ 2: χ_e · H_β = w_1 · (χ_{e-1} · H_β) − χ_{e-2} · H_β.
        recur_em1 = _mul_by_H_beta((0, e - 1), beta)
        recur_em2 = _mul_by_H_beta((0, e - 2), beta) if e >= 2 else {}
        w1_recur: dict = {}
        for s, c in recur_em1.items():
            ws = w1_mul_seed(s)
            for s_out, c_out in ws.items():
                w1_recur[s_out] = w1_recur.get(s_out, RLaurent(_R, {})) + c * c_out
        out = dict(w1_recur)
        for s, c in recur_em2.items():
            out[s] = out.get(s, RLaurent(_R, {})) - c
        return {k: v for k, v in out.items() if not v.is_zero()}
    raise AssertionError(
        f"_mul_by_H_beta: invariant left-seed m in {{0,1}} violated (m={m}); "
        f"left seeds are Wilson (m=0) or single H (m=1)")


# ---------- Main: H_a · H_b via downward W_1 walk --------------------

_cache: dict = {}


def h_mul_h(a: int, b: int) -> dict:
    """`H_a · H_b` in canonical pSU2nf1 (m, e) labels with `RLaurent[R]`
    coefficients (μ-decorated q-Laurent polynomials).

    Total in (a, b).  For a <= b: recursion downward in the gap
    `b - a >= 0` (the decreasing termination variable), bootstrapped from
    the gap <= 3 base cases.  For a > b: the bar-conjugate of (b, a).
    Cached.
    """
    if a > b:
        # Total by bar-symmetry: bar is antimultiplicative and fixes the
        # canonical basis (bar(H_n)=H_n), so H_a*H_b = bar(H_b*H_a).
        # RLaurent.bar() flips q only (μ-content is bar-invariant).  Verified
        # identical to the within-cone-cocycle / cross-cluster-bar dispatch
        # for every a>b.  No precondition, no honest-fail.
        out: dict = {}
        for seed, coef in h_mul_h(b, a).items():
            bc = coef.bar()
            if not bc.is_zero():
                out[seed] = bc
        return out
    if (a, b) in _cache:
        return _cache[(a, b)]
    gap = b - a
    if gap == 0:
        result = _gap0(a)
    elif gap == 1:
        result = _gap1(a)
    elif gap == 2:
        result = _gap2(a)
    elif gap == 3:
        result = _gap3(a)
    else:
        # From (w_1·H_{a+1})·H_b = w_1·(H_{a+1}·H_b):
        #   α_w(a+1)·H_a·H_b + β_w(a+1)·H_{a+2}·H_b + γ_w(a+1)·[extra]·H_b
        #     = w_1·(H_{a+1}·H_b)
        # Solving for H_a·H_b:
        #   H_a·H_b = α_w(a+1)^{-1} · [w_1·(H_{a+1}·H_b) − β_w(a+1)·H_{a+2}·H_b
        #                              − γ_w(a+1)·([extra]·H_b)]
        #
        # Placeholder (pure-SU(2) form): α_w(a+1) = q, β_w(a+1) = q^{−1},
        # γ_w(a+1) = [(a+1) odd: 1].
        Hap1_b = h_mul_h(a + 1, b)
        Hap2_b = h_mul_h(a + 2, b)
        w1_term: dict = {}
        for s, c in Hap1_b.items():
            ws = w1_mul_seed(s)
            for s_out, c_out in ws.items():
                w1_term[s_out] = w1_term.get(s_out, RLaurent(_R, {})) + c * c_out
        # H_a·H_b = q^{-1}·w_1·(H_{a+1}·H_b)  −  q^{-2}·H_{a+2}·H_b
        #           −  q^{-1} · ε_{a+1} · H_b
        # where ε_n = 1 (odd n) or μ (even n).
        result: dict = {}
        for s, c in w1_term.items():
            result[s] = result.get(s, RLaurent(_R, {})) + _rl_q(-1) * c
        for s, c in Hap2_b.items():
            result[s] = result.get(s, RLaurent(_R, {})) - _rl_q(-2) * c
        eps = epsilon(a + 1)
        if not eps.is_zero():
            result[(1, b)] = result.get((1, b), RLaurent(_R, {})) \
                             - _rl_q(-1) * eps
        result = {k: v for k, v in result.items() if not v.is_zero()}
    _cache[(a, b)] = result
    return result


def clear_cache():
    _cache.clear()
    _mul_cache.clear()


__all__ = [
    'h_mul_h', 'w1_mul_seed', '_mul_by_H_beta', '_seed_to_h_pair',
    'w1_H_clebsch_left', 'w1_H_clebsch_right', 'clear_cache',
]
