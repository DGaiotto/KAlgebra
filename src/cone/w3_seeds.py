"""Closed-form elementary traces of [A₁,E₆] and [A₁,E₈] from W₃ minimal models.

The finite-zoo classes `FiniteE6KAlgebra` and `FiniteE8KAlgebra` served their
seed traces from frozen tables until 2026-09-23 (E₆ to 𝖖¹²; E₈ capped at 𝖖⁶
by a hard memory wall in the seed bootstrap).  Here every seed is a finite
ℤ[𝖖^±]-combination of the distinct characters of W₃(3,7) (E₆) or W₃(3,8) (E₈),
exact to any order:

    Tr(seed) = Σ_{(h, j, v) ∈ recipe(seed)} v · 𝖖^j · χ_h(𝖖²),

χ_h the normalised character of conformal weight h (the series in
q_paper = 𝖖² divided by its leading power, so χ_h = 1 + O(q_paper)):
h ∈ {0, −3/7, −4/7, −5/7} for E₆, h ∈ {0, −1/2, −3/4, −7/8, −1} for E₈.
Tr(1) = χ₀.  Only integral powers of 𝖖 appear.

Characters: the Frenkel–Kac–Wakimoto form (a literature input, checked by the
controls in the tests: the W₂(2,5) = M(2,5) characters equal the Rogers–
Ramanujan sums and each vacuum equals the zoo's frozen Tr(1)),

    χ_{λ,μ} = η^{−(N−1)} Σ_{w∈W} ε(w) Σ_{a∈Q} q^{|p'(λ+ρ) − p w(μ+ρ) + pp' a|² / (2pp')},

λ ∈ P₊^{p−N}, μ ∈ P₊^{p'−N}, modules up to the diagonal ℤ_N.  (ρ there is the
Weyl vector of sl_N, the standard notation of the formula — not the algebra
automorphism.)

How the recipes were found and what certifies them.  A scouting run
(2026-09-23) solved the W₃ ansatz against every 𝖖^{≤0} equation of the
orthonormality bootstrap on each zoo class's own cone multiply; separately, the
ansatz-free solve of the same equations pins every seed uniquely — through 𝖖³⁶
(E₈) and 𝖖³⁷–𝖖⁴⁰ (E₆) — and the recipes reproduce it.  So in those windows the
recipes are forced by the axioms (not fitted); beyond, they are the
prediction.  Independent witnesses: the frozen BPS tables (E₈ 𝖖⁶, E₆ 𝖖¹²;
both off the serving path since 2026-09-23 — the E₈ one is kept as a fixture of
the suite in the source repository) and the `E8RGKAlgebra` flow (vacuum to 𝖖¹⁸).  Seeds
related by ρ have equal recipes, as ρ-equivariance of the trace requires.

Pure Python (fractions, itertools); no dependency.
"""
from __future__ import annotations

from fractions import Fraction as Fr
from itertools import product

# ---------------------------------------------------------------------------
# W_N(p, p') minimal-model characters, exact integer q_paper-series
# ---------------------------------------------------------------------------


def _inner(N):
    """Inverse Cartan matrix of sl_N: the Gram matrix (ω_i, ω_j)."""
    return [[Fr(min(i, j) * (N - max(i, j)), N) for j in range(1, N)]
            for i in range(1, N)]


def _norm2(v, G):
    n = len(v)
    return sum(G[i][j] * v[i] * v[j] for i in range(n) for j in range(n))


def _cartan(N):
    r = N - 1
    return [[2 if i == j else (-1 if abs(i - j) == 1 else 0) for j in range(r)]
            for i in range(r)]


def _weyl_group(N):
    """The Weyl group of sl_N as (word, sign) pairs, with its action on
    Dynkin-label vectors."""
    r = N - 1
    C = _cartan(N)

    def refl(i, v):
        v = list(v)
        vi = v[i]
        for j in range(r):
            v[j] -= vi * C[i][j]
        return tuple(v)

    start = tuple([1] * r)
    seen = {start: ()}
    frontier = [start]
    while frontier:
        nxt = []
        for v in frontier:
            for i in range(r):
                u = refl(i, v)
                if u not in seen:
                    seen[u] = seen[v] + (i,)
                    nxt.append(u)
        frontier = nxt

    def act(word, v):
        for i in reversed(word):
            v = refl(i, v)
        return v

    return [(w, (-1) ** len(w)) for w in seen.values()], act


def _eta_inv_pow(k, M):
    """∏_{n≥1}(1 − q^n)^{−k} to q^M."""
    s = [0] * (M + 1)
    s[0] = 1
    for _ in range(k):
        for n in range(1, M + 1):
            for m in range(n, M + 1):
                s[m] += s[m - n]
    return s


def w_character(N, p, pp, lam, mu, M):
    """Normalised character of the W_N(p, p') module (λ, μ) to q^M (a list of
    ints starting with 1) and the stripped exponent h − c/24."""
    r = N - 1
    G = _inner(N)
    W, act = _weyl_group(N)
    A = _cartan(N)
    lr = tuple(l + 1 for l in lam)
    mr = tuple(m + 1 for m in mu)

    def collect(B):
        out = {}
        for w, eps in W:
            wm = act(w, mr)
            for ns in product(range(-B, B + 1), repeat=r):
                a = [sum(ns[i] * A[i][j] for i in range(r)) for j in range(r)]
                v = [pp * lr[j] - p * wm[j] + p * pp * a[j] for j in range(r)]
                E = _norm2(v, G) / (2 * p * pp)
                out[E] = out.get(E, 0) + eps
        return {e: c for e, c in out.items() if c}

    prev, B = None, 2
    while True:                       # widen the lattice box until stable
        cur = collect(B)
        e0 = min(cur)
        trunc = {e: c for e, c in cur.items() if e - e0 <= M}
        if prev is not None and trunc == prev:
            break
        prev, B = trunc, B + 2
    e0 = min(prev)
    theta = [0] * (M + 1)
    for e, c in prev.items():
        d = e - e0
        if d.denominator != 1:
            raise ArithmeticError(f"non-integral relative exponent {d}")
        theta[int(d)] += c
    et = _eta_inv_pow(r, M)
    ch = [sum(theta[i] * et[m - i] for i in range(m + 1)) for m in range(M + 1)]
    k = 0
    while k <= M and ch[k] == 0:
        k += 1
    if k > M:
        return [], None
    return ch[k:], e0 - Fr(r, 24) + k


def central_charge(N, p, pp):
    return Fr(N - 1) * (1 - Fr(N * (N + 1) * (p - pp) ** 2, p * pp))


def _modules(N, p, pp):
    """Module labels (λ, μ) up to the diagonal ℤ_N action on affine labels."""
    def lev(k):
        return [labs for labs in product(range(k + 1), repeat=N - 1) if sum(labs) <= k]

    def aff(l, k):
        return (k - sum(l),) + tuple(l)

    def rot(a):
        return (a[-1],) + a[:-1]

    kl, km = p - N, pp - N
    seen, reps = set(), []
    for l in lev(kl):
        for m in lev(km):
            a, b = aff(l, kl), aff(m, km)
            if (a, b) in seen:
                continue
            for _ in range(N):
                seen.add((a, b))
                a, b = rot(a), rot(b)
            reps.append((l, m))
    return reps


def distinct_characters(N, p, pp, M):
    """`{h: normalised series to q^M}` over the distinct characters of
    W_N(p, p') (modules with equal series are identified; distinct series
    must have distinct h — checked)."""
    c = central_charge(N, p, pp)
    out = {}
    for lam, mu in _modules(N, p, pp):
        ch, lead = w_character(N, p, pp, lam, mu, M)
        h = lead + c / 24
        if h in out:
            if out[h] != ch:
                raise ArithmeticError(f"two characters at h = {h} differ")
            continue
        out[h] = ch
    return out


# ---------------------------------------------------------------------------
# the recipes
# ---------------------------------------------------------------------------

# seed (the zoo's ρ²-orbit representative index) -> [(h, 𝖖-shift j, coefficient)]
# [A₁,E₈]: W₃(3,8), weights 0, −1/2, −3/4, −7/8, −1
_E8_H = {"0": Fr(0), "-1/2": Fr(-1, 2), "-3/4": Fr(-3, 4), "-7/8": Fr(-7, 8), "-1": Fr(-1)}
_R_A = [("0", -2, 1), ("0", 0, 1), ("-1/2", -2, -2), ("-3/4", -2, 2),
        ("-7/8", -4, -1), ("-7/8", -2, -1), ("-1", -4, 1)]
_R_B = [("0", -1, 1), ("0", 1, 4), ("0", 3, 4), ("0", 5, 2), ("0", 7, 1),
        ("-1/2", -1, -2), ("-1/2", 1, -4), ("-1/2", 3, -2), ("-1/2", 5, -2),
        ("-3/4", -1, 4), ("-3/4", 1, 6), ("-3/4", 3, 2), ("-3/4", 5, 2),
        ("-7/8", -1, -4), ("-7/8", 1, -3), ("-7/8", 3, -2), ("-7/8", 5, -1),
        ("-1", -1, 1), ("-1", 3, 1)]
_R_C = [("0", 0, 3), ("0", 2, 2), ("0", 4, 1), ("-1/2", -2, -2), ("-1/2", 0, -2),
        ("-1/2", 2, -2), ("-3/4", -2, 4), ("-3/4", 0, 2), ("-3/4", 2, 2),
        ("-7/8", -2, -2), ("-7/8", 0, -2), ("-7/8", 2, -1), ("-1", 0, 1)]
_R_D = [("0", -1, 1), ("0", 1, 2), ("0", 3, 1), ("-1/2", -1, -2), ("-1/2", 1, -2),
        ("-3/4", -1, 2), ("-3/4", 1, 2), ("-7/8", -1, -2), ("-7/8", 1, -1),
        ("-1", -1, 1)]
_R_E = [("0", 0, 2), ("0", 2, 1), ("-1/2", -2, -1), ("-1/2", 0, -2), ("-3/4", -2, 2),
        ("-3/4", 0, 2), ("-7/8", -2, -2), ("-7/8", 0, -1), ("-1", -2, 1)]
_R_F = [("0", -1, 1), ("0", 1, 1), ("-1/2", -1, -2), ("-3/4", -1, 2), ("-7/8", -1, -1)]
_R_G = [("0", 0, 1), ("-1/2", -2, -1), ("-3/4", -2, 2), ("-7/8", -2, -1)]
_R_H = [("0", -1, 1), ("-3/4", -3, 1), ("-7/8", -3, -1)]

E8_RECIPES = {0: _R_A, 24: _R_A, 1: _R_B, 2: _R_B, 5: _R_C, 22: _R_C,
           8: _R_D, 15: _R_D, 10: _R_E, 13: _R_E, 17: _R_F, 27: _R_F,
           21: _R_G, 23: _R_G, 26: _R_H, 28: _R_H}

# [A₁,E₆]: W₃(3,7), weights 0, −3/7, −4/7, −5/7
_E6_H = {"0": Fr(0), "-3/7": Fr(-3, 7), "-4/7": Fr(-4, 7), "-5/7": Fr(-5, 7)}
_E6_A = [("0", -1, 1), ("-3/7", -1, -2), ("-4/7", -1, 1)]
_E6_B = [("-3/7", -2, -1), ("-4/7", -2, 1)]
E6_RECIPES = {
    0: [("0", -1, 1), ("-3/7", -1, -2), ("-4/7", -1, 2), ("-5/7", -1, -1), ("0", 1, 1)],
    2: _E6_A, 6: _E6_A,
    4: [("0", -1, 1), ("-4/7", -3, 1), ("-5/7", -3, -1)],
    5: _E6_B, 7: _E6_B,
}


class W3Seeds:
    """The seed traces of one theory: its W₃(3, p') characters and recipes."""

    def __init__(self, pp, weights, recipes):
        self.pp = pp
        self.weights = weights
        self.recipes = recipes
        self._cache = None

    def characters(self, Mp):
        if self._cache is None or self._cache[0] < Mp:
            ch = distinct_characters(3, 3, self.pp, Mp)
            if set(ch) != set(self.weights.values()):
                raise ArithmeticError(f"W3(3,{self.pp}): unexpected weights {sorted(ch)}")
            self._cache = (Mp, ch)
        return self._cache[1]

    def series(self, recipe, K):
        jmin = min(j for _h, j, _v in recipe)
        ch = self.characters(max(0, (K - jmin) // 2) + 1)
        out = {}
        for hs, j, v in recipe:
            for k, x in enumerate(ch[self.weights[hs]]):
                e = j + 2 * k
                if e > K:
                    break
                if x:
                    out[e] = out.get(e, 0) + v * x
        out = {e: c for e, c in out.items() if c}
        if out and min(out) < 0:
            raise ArithmeticError(f"negative power {min(out)} in a W3(3,{self.pp}) seed trace")
        return out

    def vacuum_trace(self, K: int) -> dict:
        """Tr(1) = χ₀ as `{𝖖-power: int}` through 𝖖^K."""
        return self.series([("0", 0, 1)], K)

    def seed_trace(self, orbit: int, K: int) -> dict:
        """Tr of the zoo seed `orbit` (a ρ²-orbit representative index) as
        `{𝖖-power: int}` through 𝖖^K."""
        return self.series(self.recipes[orbit], K)


E6 = W3Seeds(7, _E6_H, E6_RECIPES)
E8 = W3Seeds(8, _E8_H, E8_RECIPES)
