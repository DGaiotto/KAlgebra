"""Spine-free multiplication for the SU(2)+N_f=2 standalone `ConeKAlgebra`,
the Spin(4) lift of `su2_nf1_h_multiply`.

Native label: `(h_factors, (wL, wR))` where `h_factors` is a sorted tuple of
`(n, exp)` (H-monomial) or `((('W', e), 1),)` (Wilson chi_e), and `(wL, wR)`
is the Spin(4) Cartan flavour weight (SU(2)_L x SU(2)_R).

Factorisation (validated vs the BPS oracle): gauge generators (H-tower,
Wilson) are flavour-neutral, so

    (gauge_a, w_a) * (gauge_b, w_b)
        = sum over gauge-product terms (gauge_c, chi_c) of
          (gauge_c,  w_a + w_b + weight(chi_c))

i.e. multiply the gauge cone monomials with the closed-form
`su2_nf2_h_gap_k` recursion (whose matter insertions carry Spin(4)
characters chi_c), then expand chi_c to Cartan weights and shift by the
sum of the input flavour weights.  The gauge multiply itself is the
literal-word reducer (cross/swap/bubble) over the rank-2 H-pair cones
`{H_a, H_{a+1}}` plus the Wilson cone, identical in shape to pure SU(2) /
N_f=1.
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from laurent_poly import LaurentPoly
from kalgebra import Element
from zplus_ring import RLaurent
from su2_nf2_h_gap_k import (
    R_SP4, rl, rl_zero, h_mul_h, w1_H, seed_to_pair,
)
from pure_su2_h_wilson import (
    chi_to_w1_powers, w1_power_to_chi, WILSON_FUND, is_wilson_label,
)

_R = R_SP4


def _rl_q(n: int) -> RLaurent:
    return RLaurent(_R, {n: _R.one()})


def _rl_int(c: int) -> RLaurent:
    return RLaurent(_R, {0: _R.one() * c}) if c else rl_zero()


# ---- letter predicates / gauge cone structure (rank-2 H-pair cones) --

def _is_H(l: tuple) -> bool:
    return l[0] == 'H'


def _is_W(l: tuple) -> bool:
    return l[0] == 'W'


def cone_index_for(n_lo: int, n_hi: int):
    """Rank-2 pair cone C_a = {H_a, H_{a+1}}: q-commute iff |b-a| <= 1."""
    return n_lo if (n_hi - n_lo) <= 1 else None


def _q_commute(g: tuple, h: tuple) -> bool:
    if g == h:
        return True
    if _is_W(g) and _is_W(h):
        return True
    if _is_W(g) or _is_W(h):
        return False
    a, b = g[1], h[1]
    return abs(a - b) <= 1


def _cocycle(g: tuple, h: tuple) -> int:
    if g == h:
        return 0
    if _is_W(g) and _is_W(h):
        return 0
    if _is_W(g) or _is_W(h):
        raise ValueError(f"_cocycle: ({g}, {h}) non-q-commuting")
    return h[1] - g[1]


# ---- ray products for non-q-commuting pairs -> literal words ---------

_RAY: dict = {}


def _ray_product(g: tuple, h: tuple) -> list:
    key = (g, h)
    if key in _RAY:
        return _RAY[key]
    if _is_H(g) and _is_H(h):
        res = _hh_product(g[1], h[1])
    elif _is_H(g) and _is_W(h):
        res = _Hw_product(g[1])
    elif _is_W(g) and _is_H(h):
        res = _wH_product(h[1])
    else:
        raise ValueError(f"_ray_product: unexpected ({g}, {h})")
    _RAY[key] = res
    return res


def _hh_product(a: int, b: int) -> list:
    if a <= b:
        canonical = h_mul_h(a, b)
    elif abs(a - b) <= 1:
        fwd = h_mul_h(b, a)
        phase = _rl_q(2 * (b - a))
        canonical = {s: phase * c for s, c in fwd.items() if not c.is_zero()}
    else:
        # cross-cluster reverse: bar-conjugate q -> q^{-1} (flavour bar-invariant)
        fwd = h_mul_h(b, a)
        canonical = {}
        for s, c in fwd.items():
            nc = RLaurent(_R, {-e: r for e, r in c.coeffs.items()})
            if not nc.is_zero():
                canonical[s] = nc
    return _canonical_to_literal(canonical)


def _canonical_to_literal(canonical: dict) -> list:
    """{clean (m,e): RLaurent} -> [(RLaurent, literal-word)]."""
    out: list = []
    for (m, e), coef in canonical.items():
        if coef.is_zero():
            continue
        if m == 0:
            if e == 0:
                out.append((coef, ()))
            else:
                for j, scalar in chi_to_w1_powers(e).items():
                    if scalar:
                        term = coef * _rl_int(scalar)
                        if not term.is_zero():
                            out.append((term, (WILSON_FUND,) * j))
        elif m == 1:
            out.append((coef, (('H', e),)))
        elif m == 2:
            alpha, beta = seed_to_pair(e)
            phase = -(beta - alpha) if alpha != beta else 0
            term = coef * _rl_q(phase)
            if not term.is_zero():
                out.append((term, (('H', alpha), ('H', beta))))
        else:
            raise NotImplementedError(f"_canonical_to_literal m={m}")
    return out


def _Hw_product(n: int) -> list:
    # H_n * w_1 = q^{-1} H_{n-1} + q H_{n+1} + eps_n  (right Clebsch)
    canonical: dict = {}
    base = w1_H(n)   # gives q H_{n-1}+q^-1 H_{n+1}+eps  (left); swap q<->q^-1
    for (m, e), c in base.items():
        if m == 1:
            # swap q exponents sign for right action
            c = RLaurent(_R, {-q: r for q, r in c.coeffs.items()})
        canonical[(m, e)] = canonical.get((m, e), rl_zero()) + c
    return _canonical_to_literal(canonical)


def _wH_product(n: int) -> list:
    # w_1 * H_n (left Clebsch)
    return _canonical_to_literal(w1_H(n))


# ---- native <-> literal ---------------------------------------------

def _psu2_to_native_gauge(m: int, e: int) -> tuple:
    """clean (m,e) gauge seed -> h_factors (max-diagonal), no flavour."""
    if m == 0:
        return () if e == 0 else ((('W', e), 1),)
    if m == 1:
        return ((e, 1),)
    base, rem = divmod(e, m)
    entries = []
    if m - rem > 0:
        entries.append((base, m - rem))
    if rem > 0:
        entries.append((base + 1, rem))
    return tuple(entries)


def _expand_gauge_to_literal(h_factors: tuple) -> list:
    """gauge h_factors -> [(RLaurent, literal word)] (no flavour weight)."""
    if not h_factors:
        return [(rl(0), ())]
    first = h_factors[0]
    if isinstance(first[0], tuple) and first[0][0] == 'W':
        e = first[0][1]
        if e == 0:
            return [(rl(0), ())]
        return [(_rl_int(s), (WILSON_FUND,) * j)
                for j, s in chi_to_w1_powers(e).items() if s]
    chain: list = []
    for (n, exp) in h_factors:
        chain += [n] * exp
    chain.sort()
    phase = sum(chain[j] - chain[i]
                for i in range(len(chain)) for j in range(i + 1, len(chain)))
    word = tuple(('H', n) for n in chain)
    return [(_rl_q(-phase), word)]


def _word_is_canonical(word: tuple) -> bool:
    if not word:
        return True
    if all(_is_W(l) for l in word):
        return True
    if all(_is_H(l) for l in word):
        chain = [l[1] for l in word]
        if any(chain[i] > chain[i + 1] for i in range(len(chain) - 1)):
            return False
        return cone_index_for(min(chain), max(chain)) is not None
    return False


def _letter_gt(g: tuple, h: tuple) -> bool:
    if _is_H(g) and _is_H(h):
        return g[1] > h[1]
    if _is_W(g) and _is_H(h):
        return True
    return False


def _find_action(word: tuple):
    if _word_is_canonical(word):
        return None
    for i in range(len(word) - 1):
        if not _q_commute(word[i], word[i + 1]):
            return (i, 'cross')
    for i in range(len(word) - 1):
        g, h = word[i], word[i + 1]
        if g != h and _q_commute(g, h) and _letter_gt(g, h):
            return (i, 'swap')
    for i in range(len(word)):
        for j in range(i + 1, len(word)):
            if not _q_commute(word[i], word[j]):
                return (i, 'bubble', j)
    return None


def _reduce(initial: list) -> list:
    completed: list = []
    work = list(initial)
    steps = 0
    while work:
        steps += 1
        if steps > 500_000:
            raise RuntimeError("_reduce: step budget exceeded")
        c, w = work.pop()
        if c.is_zero():
            continue
        action = _find_action(w)
        if action is None:
            completed.append((c, w))
            continue
        kind = action[1]
        if kind == 'swap':
            i = action[0]
            g, h = w[i], w[i + 1]
            work.append((c * _rl_q(2 * _cocycle(g, h)), w[:i] + (h, g) + w[i + 2:]))
        elif kind == 'cross':
            i = action[0]
            g, h = w[i], w[i + 1]
            for sc, sw in _ray_product(g, h):
                work.append((c * sc, w[:i] + sw + w[i + 2:]))
        elif kind == 'bubble':
            i, j = action[0], action[2]
            new_w = list(w)
            phase_exp = 0
            target = new_w[j]
            fired = False
            for k in range(j, i + 1, -1):
                partner = new_w[k - 1]
                if not _q_commute(partner, target):
                    cur_w = tuple(new_w)
                    cur_c = c * _rl_q(phase_exp)
                    for sc, sw in _ray_product(partner, target):
                        work.append((cur_c * sc, cur_w[:k - 1] + sw + cur_w[k + 1:]))
                    fired = True
                    break
                phase_exp += 2 * _cocycle(partner, target)
                new_w[k - 1], new_w[k] = new_w[k], new_w[k - 1]
            if not fired:
                work.append((c * _rl_q(phase_exp), tuple(new_w)))
    return completed


def _word_to_native_gauge(word: tuple) -> dict:
    """canonical literal word -> {gauge h_factors: RLaurent}."""
    out: dict = {}
    if not word:
        out[()] = rl(0)
        return out
    if all(_is_W(l) for l in word):
        for e, mult in w1_power_to_chi(len(word)).items():
            if mult:
                hf = () if e == 0 else ((('W', e), 1),)
                out[hf] = out.get(hf, rl_zero()) + _rl_int(mult)
        return {k: v for k, v in out.items() if not v.is_zero()}
    if all(_is_H(l) for l in word):
        chain = [l[1] for l in word]
        phase = sum(chain[j] - chain[i]
                    for i in range(len(chain)) for j in range(i + 1, len(chain)))
        counts: dict = {}
        for n in chain:
            counts[n] = counts.get(n, 0) + 1
        hf = tuple(sorted(counts.items()))
        out[hf] = _rl_q(phase)
        return out
    raise ValueError(f"_word_to_native_gauge: mixed word {word}")


# ---- flavour: expand Spin(4) char coeff into Cartan weights ----------

def _weights_of(relt) -> dict:
    out: dict = {}
    for (nL, nR), c in relt.terms.items():
        for wL in range(-nL, nL + 1, 2):
            for wR in range(-nR, nR + 1, 2):
                out[(wL, wR)] = out.get((wL, wR), 0) + c
    return out


# ---- top-level multiply ---------------------------------------------

def multiply_native(a: tuple, b: tuple) -> Element:
    """(h_factors_a, (wLa,wRa)) * (h_factors_b, (wLb,wRb)) -> Element over
    native labels (h_factors, (wL,wR)) with integer LaurentPoly coeffs."""
    hfa, wa = a[0], tuple(a[1])
    hfb, wb = b[0], tuple(b[1])
    wsum = (wa[0] + wb[0], wa[1] + wb[1])
    # gauge product
    terms_a = _expand_gauge_to_literal(hfa)
    terms_b = _expand_gauge_to_literal(hfb)
    initial = [(ca * cb, wda + wdb)
               for (ca, wda) in terms_a for (cb, wdb) in terms_b]
    completed = _reduce(initial)
    gauge_out: dict = {}      # {gauge_hf: RLaurent[Spin4]}
    for c, w in completed:
        for hf, factor in _word_to_native_gauge(w).items():
            t = c * factor
            if not t.is_zero():
                gauge_out[hf] = gauge_out.get(hf, rl_zero()) + t
    # expand Spin(4) chars -> weights, shift by wsum, assemble Element
    out: dict = {}
    for hf, rlc in gauge_out.items():
        for qexp, relt in rlc.coeffs.items():
            for (wL, wR), z in _weights_of(relt).items():
                if z == 0:
                    continue
                label = (hf, (wsum[0] + wL, wsum[1] + wR))
                lp = out.setdefault(label, LaurentPoly({}))
                out[label] = lp + LaurentPoly({qexp: z})
    return Element({k: v for k, v in out.items() if not v.is_zero()})


__all__ = ['multiply_native', 'cone_index_for', 'WILSON_FUND',
           '_psu2_to_native_gauge']
