"""Axiom-derived multiplication for `SU2Nf1KAlgebra`, mirroring
`pure_su2_h_multiply.py` with U(1)_F-flavored `RLaurent[R]` coefficients
(R = `AbelianZPlusRing(rank=1)`).

Literal-word reducer:
  * State is a polynomial in `(RLaurent coef, literal-ray-word)` pairs.
  * Ray-ray substitutions come from `su2_nf1_h_gap_k.h_mul_h` (W_1-walk
    cyclicity for H × H, any gap) and `w1_H_clebsch_*` (w_1 ↔ H Clebsch),
    all carrying μ-decorated R-coefficients.
  * Wilson-canonical χ_e ↔ literal w_1^k via Chebyshev (μ-trivial).

Native label format: `(h_factors, μ_power)` where `h_factors` is a
sorted tuple of `(n, exp)` (H-monomial) or `((('W', e), 1),)` (Wilson
χ_e).  μ_power is a separate integer (peeled to R at the section-
decompose boundary).
"""
from __future__ import annotations
from fractions import Fraction

from laurent_poly import LaurentPoly
from kalgebra import Element
from zplus_ring import AbelianZPlusRing, RLaurent
from su2_nf1_h_gap_k import (
    h_mul_h, _mul_by_H_beta, _seed_to_h_pair, w1_mul_seed,
    w1_H_clebsch_left, w1_H_clebsch_right, epsilon,
)
from pure_su2_h_wilson import (
    chi_to_w1_powers, w1_power_to_chi, WILSON_FUND, is_wilson_label,
)
from su2_nf1_cone_data import cone_index_for as _cone_idx_for


_R = AbelianZPlusRing(rank=1)


def _rl_zero() -> RLaurent:
    return RLaurent(_R, {})


def _rl_one() -> RLaurent:
    return RLaurent(_R, {0: _R.one()})


def _rl_q(n: int) -> RLaurent:
    return RLaurent(_R, {n: _R.one()})


def _rl_mu(q_pow: int = 0, mu_pow: int = 1) -> RLaurent:
    return RLaurent(_R, {q_pow: _R.basis_element((mu_pow,))})


def _rl_int(c: int) -> RLaurent:
    return RLaurent(_R, {0: _R.one() * c}) if c else _rl_zero()


# -------------------------------------------------------------------
# Letter / q-commute predicates (Nf=1 rank-2 cones)
# -------------------------------------------------------------------

def _is_H(l: tuple) -> bool:
    return l[0] == 'H'


def _is_W(l: tuple) -> bool:
    return l[0] == 'W'


def _q_commute(g: tuple, h: tuple) -> bool:
    """Nf=1 q-commutation: rank-2 pair cones for H's, single-letter
    Wilson, no H ↔ W commute."""
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


# -------------------------------------------------------------------
# Ray-product table — RLaurent coefs with μ-decoration
# -------------------------------------------------------------------

_RAY_CACHE: dict = {}


def _ray_product(g: tuple, h: tuple) -> list:
    """`g · h` for non-q-commuting pair → list of `(RLaurent, literal-word)`."""
    key = (g, h)
    if key in _RAY_CACHE:
        return _RAY_CACHE[key]
    result = _ray_product_impl(g, h)
    _RAY_CACHE[key] = result
    return result


def _ray_product_impl(g: tuple, h: tuple) -> list:
    if _is_H(g) and _is_H(h):
        return _hh_product(g[1], h[1])
    if _is_H(g) and _is_W(h):
        return _Hw_product(g[1])
    if _is_W(g) and _is_H(h):
        return _wH_product(h[1])
    raise ValueError(f"_ray_product: unexpected ({g}, {h})")


def _hh_product(a: int, b: int) -> list:
    """Literal `H_a · H_b` substitution from `h_mul_h`, expanded into
    literal-word replacements.

    `h_mul_h` is total in (a, b): a <= b runs the gap recursion, a > b is
    the bar-conjugate of (b, a) -- handled inside `h_mul_h` (verified equal
    to the within-cone-cocycle / cross-cluster-bar dispatch), so this
    wrapper needs no ordering case."""
    return _canonical_to_literal(h_mul_h(a, b))


def _canonical_to_literal(canonical: dict) -> list:
    """Convert `{(m, e): RLaurent}` to list of `(RLaurent, literal-word)`."""
    out: list = []
    for seed, coef in canonical.items():
        if coef.is_zero():
            continue
        m, e = seed
        if m == 0 and e == 0:
            out.append((coef, ()))
        elif m == 0:
            chi_w1 = chi_to_w1_powers(e)
            for j, scalar in chi_w1.items():
                if scalar == 0:
                    continue
                word = (WILSON_FUND,) * j
                term = coef * _rl_int(scalar)
                if not term.is_zero():
                    out.append((term, word))
        elif m == 1:
            out.append((coef, (('H', e),)))
        elif m == 2:
            alpha, beta = _seed_to_h_pair(seed)
            phase = -(beta - alpha) if alpha != beta else 0
            word = (('H', alpha), ('H', beta))
            term = coef * _rl_q(phase)
            if not term.is_zero():
                out.append((term, word))
        else:
            # Invariant: h_mul_h outputs canonical seeds with m <= 2
            # (H*H has magnetic charge = #H-factors <= 2); m>2 cannot occur.
            raise AssertionError(
                f"_canonical_to_literal: invariant m<=2 violated (m={m}); "
                f"h_mul_h seeds carry m = #H-factors <= 2")
    return out


def _Hw_product(n: int) -> list:
    """`H_n · w_1`: q^{-1}·H_{n-1} + q·H_{n+1} + ε_n."""
    canonical: dict = {}
    for seed, c in w1_H_clebsch_right(n):
        canonical[seed] = canonical.get(seed, _rl_zero()) + c
    return _canonical_to_literal(canonical)


def _wH_product(n: int) -> list:
    """`w_1 · H_n`: q·H_{n-1} + q^{-1}·H_{n+1} + ε_n."""
    canonical: dict = {}
    for seed, c in w1_H_clebsch_left(n):
        canonical[seed] = canonical.get(seed, _rl_zero()) + c
    return _canonical_to_literal(canonical)


# -------------------------------------------------------------------
# Native ↔ literal conversion
# -------------------------------------------------------------------

def _native_to_psu2nf1(native: tuple) -> tuple:
    """Native `(h_factors, μ_power)` → `(m, e, μ_pow)`."""
    h_factors, mu_p = native
    m, e = 0, 0
    for entry in h_factors:
        gen, exp = entry
        if isinstance(gen, int):
            m += exp
            e += gen * exp
        elif isinstance(gen, tuple) and gen[0] == 'W':
            assert exp == 1
            e += gen[1]
        else:
            raise ValueError(f"_native_to_psu2nf1: unrecognised {entry}")
    return (m, e, int(mu_p))


def _psu2nf1_to_native(m: int, e: int, mu_p: int = 0) -> tuple:
    """`(m, e, μ_pow)` → max-diagonal native `(h_factors, μ_power)`.

    H-monomial max-diagonal: for m ≥ 1, factor as H-letters in
    consecutive indices summing to e (rank-2 cone structure).
    """
    if m == 0:
        return ((), int(mu_p)) if e == 0 else (((('W', e), 1),), int(mu_p))
    if m == 1:
        return (((e, 1),), int(mu_p))
    if m == 2:
        alpha, beta = _seed_to_h_pair((m, e))
        if alpha == beta:
            return (((alpha, 2),), int(mu_p))
        return (((alpha, 1), (beta, 1)), int(mu_p))
    # m ≥ 3: rank-2 cone monomial H_a^j · H_{a+1}^k with j+k=m, summing
    # to e.  e = j·a + k·(a+1) = m·a + k  ⇒  a = (e - k)/m for some
    # k ∈ {0, …, m}.  Pick the unique k mod m that makes a integer.
    k = e % m
    a = (e - k) // m
    j = m - k
    if j == m:                                           # k=0: H_a^m
        return (((a, m),), int(mu_p))
    if k == m:                                           # impossible (k<m)
        return (((a + 1, m),), int(mu_p))
    return (((a, j), (a + 1, k)), int(mu_p))


# -------------------------------------------------------------------
# Native → literal expansion
# -------------------------------------------------------------------

def _expand_native_to_literal(native: tuple) -> list:
    """Native → list of `(RLaurent, literal word)`.  μ_power lifts to
    an R-element factor."""
    h_factors, mu_p = native
    mu_coef = RLaurent(_R, {0: _R.basis_element((int(mu_p),))})
    if not h_factors:
        return [(mu_coef, ())]
    first = h_factors[0]
    if isinstance(first[0], tuple) and first[0][0] == 'W':
        e = first[0][1]
        if e == 0:
            return [(mu_coef, ())]
        chi_w1 = chi_to_w1_powers(e)
        return [
            (mu_coef * _rl_int(scalar), (WILSON_FUND,) * j)
            for j, scalar in chi_w1.items() if scalar != 0
        ]
    # H-monomial: build the literal word + cocycle phase.
    chain: list = []
    for entry in h_factors:
        n, exp = entry
        chain += [n] * exp
    chain.sort()
    # Canonical-vs-literal phase: literal H-chain = q^{phase} · canonical.
    # canonical = q^{-phase} · literal ⇒ when expanding canonical to literal,
    # multiply by q^{-phase}.
    phase = 0
    for i in range(len(chain)):
        for j in range(i + 1, len(chain)):
            phase += chain[j] - chain[i]
    word = tuple(('H', n) for n in chain)
    return [(mu_coef * _rl_q(-phase), word)]


# -------------------------------------------------------------------
# Literal word → canonical native polynomial
# -------------------------------------------------------------------

def _word_is_canonical(word: tuple) -> bool:
    if not word:
        return True
    if all(_is_W(l) for l in word):
        return True
    if all(_is_H(l) for l in word):
        chain = [l[1] for l in word]
        if any(chain[i] > chain[i + 1] for i in range(len(chain) - 1)):
            return False
        return _cone_idx_for(min(chain), max(chain)) is not None
    return False


def _letter_gt(g: tuple, h: tuple) -> bool:
    if _is_H(g) and _is_H(h):
        return g[1] > h[1]
    if _is_H(g) and _is_W(h):
        return False
    if _is_W(g) and _is_H(h):
        return True
    return False


def _find_action(word: tuple):
    if _word_is_canonical(word):
        return None
    for i in range(len(word) - 1):
        g, h = word[i], word[i + 1]
        if not _q_commute(g, h):
            return (i, 'cross')
    for i in range(len(word) - 1):
        g, h = word[i], word[i + 1]
        if g == h:
            continue
        if _q_commute(g, h) and _letter_gt(g, h):
            return (i, 'swap')
    for i in range(len(word)):
        for j in range(i + 1, len(word)):
            if not _q_commute(word[i], word[j]):
                return (i, 'bubble', j)
    return None


def _reduce(initial_terms: list) -> list:
    completed: list = []
    work = list(initial_terms)
    max_steps = 200_000
    steps = 0
    while work:
        steps += 1
        if steps > max_steps:
            raise RuntimeError(
                f"_reduce: too many steps ({max_steps})"
            )
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
            new_c = c * _rl_q(2 * _cocycle(g, h))
            work.append((new_c, w[:i] + (h, g) + w[i + 2:]))
        elif kind == 'cross':
            i = action[0]
            g, h = w[i], w[i + 1]
            for sub_c, sub_w in _ray_product(g, h):
                work.append((c * sub_c, w[:i] + sub_w + w[i + 2:]))
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
                    for sub_c, sub_w in _ray_product(partner, target):
                        new_w_split = cur_w[:k - 1] + sub_w + cur_w[k + 1:]
                        work.append((cur_c * sub_c, new_w_split))
                    fired = True
                    break
                phase_exp += 2 * _cocycle(partner, target)
                new_w[k - 1], new_w[k] = new_w[k], new_w[k - 1]
            if not fired:
                work.append((c * _rl_q(phase_exp), tuple(new_w)))
    return completed


def _word_to_native_poly(word: tuple) -> dict:
    """Sorted canonical-cone literal word → `{native: RLaurent}`."""
    out: dict = {}
    if not word:
        out[((), 0)] = _rl_one()
        return out
    if all(_is_W(l) for l in word):
        k = len(word)
        chi_decomp = w1_power_to_chi(k)
        for e, mult in chi_decomp.items():
            if mult == 0:
                continue
            native = ((), 0) if e == 0 else (((('W', e), 1),), 0)
            existing = out.get(native, _rl_zero())
            out[native] = existing + _rl_int(mult)
        return {k_: v for k_, v in out.items() if not v.is_zero()}
    if all(_is_H(l) for l in word):
        chain = [l[1] for l in word]
        if any(chain[i] > chain[i + 1] for i in range(len(chain) - 1)):
            raise ValueError(f"H-word not sorted: {word}")
        phase = 0
        for i in range(len(chain)):
            for j in range(i + 1, len(chain)):
                phase += chain[j] - chain[i]
        m = len(chain)
        e = sum(chain)
        try:
            native = _psu2nf1_to_native(m, e, 0)
        except NotImplementedError:
            # Fallback: tuple sorted (n, exp).
            counts: dict = {}
            for n in chain:
                counts[n] = counts.get(n, 0) + 1
            native = (tuple(sorted(counts.items())), 0)
        out[native] = _rl_q(phase)
        return out
    raise ValueError(
        f"_word_to_native_poly: mixed H+Wilson word: {word}"
    )


# -------------------------------------------------------------------
# Top-level: native × native → Element
# -------------------------------------------------------------------

def multiply_native(a: tuple, b: tuple) -> Element:
    """Native_a × native_b → Element with `RLaurent[R]` coefs.

    Pure axiom-derived: structure constants come from `h_mul_h`
    (W_1-walk cyclicity, verified against BPS for gaps 0..9) plus
    Nf=1 ε-corrected Clebsch.
    """
    # Pre-canonicalize input natives via the (m, e, μ) round-trip.
    a = _psu2nf1_to_native(*_native_to_psu2nf1(a))
    b = _psu2nf1_to_native(*_native_to_psu2nf1(b))
    terms_a = _expand_native_to_literal(a)
    terms_b = _expand_native_to_literal(b)
    initial: list = []
    for ca, wa in terms_a:
        for cb, wb in terms_b:
            initial.append((ca * cb, wa + wb))
    completed = _reduce(initial)
    out: dict = {}
    for c, w in completed:
        natives = _word_to_native_poly(w)
        for native, factor in natives.items():
            term = c * factor
            if term.is_zero():
                continue
            existing = out.get(native, _rl_zero())
            out[native] = existing + term
    out = {k: v for k, v in out.items() if not v.is_zero()}
    return Element(out)


__all__ = [
    'multiply_native', '_ray_product', 'WILSON_FUND',
    '_native_to_psu2nf1', '_psu2nf1_to_native',
]
