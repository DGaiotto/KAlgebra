"""Closed-form elementary traces of [A₁,E₇] (the zoo class `FiniteE7KAlgebra`).

Every one of the 90 elementary seed traces of `FiniteE7KAlgebra` is exact to any order in closed form:

    T_i = N_i / ((q;q)_∞ · θ(μ)),

in the repo's variable 𝖖 (`q_paper = 𝖖²`, `q_paper^{1/2} → −𝖖`) and the U(1)
fugacity μ, with

    θ(μ)     = Σ_{n∈ℤ} 𝖖^{n²} μⁿ,
    (q;q)_∞  = Π_{n≥1} (1 − 𝖖^{2n}),
    ψ_a      = Σ_{n∈ℤ} 𝖖^{5n² + (2a−4)n} μⁿ          (a = 0, …, 4),
    th2      = Π_{m≥0} (1 − 𝖖^{10m+2})(1 − 𝖖^{10m+8}),
    th4      = Π_{m≥0} (1 − 𝖖^{10m+4})(1 − 𝖖^{10m+6}),

and `N_i` a short integer combination of monomials `𝖖^s μ^t` times the ten
products `th · ψ_a ψ_b` (`th = th2` for `|a − b| ∈ {1, 4}`, `th4` for
`|a − b| ∈ {2, 3}` in the recipes below).  Nine ρ-orbit representatives carry
a recipe (`RECIPES`; seeds 0 and 1 share one); the other 81 seeds follow from
ρ-equivariance of the trace in its element
form on this cone tier, `T_{ρ(i)}(μ) = μ^{−δ_i} · T_i(μ⁻¹)`, with the class's own
ρ-permutation and δ-table (`E7_RHO_PERM`, `E7_RHO_DELTA`).

`Tr(1)` (`vacuum_trace`) is the product

    Tr(1) = Π_{n≥1} (1−𝖖^{10n})² (1−𝖖^{10n−2}) (1−𝖖^{10n+2})
                    (1+μ^{±1}𝖖^{10n−1}) (1+μ^{±1}𝖖^{10n+1})
          / Π_{n≥1} (1−𝖖^{2n}) (1−𝖖^{2n+2}) (1+μ^{±1}𝖖^{2n+1}),

(each `μ^{±1}` factor taken once for each sign), the Euler–Poincaré character of
the minimal reduction of the Kac–Wakimoto boundary-level `sl₃` character at
`k = −3 + 3/5` — the vacuum character of the Bershadsky–Polyakov algebra at
level `−12/5`, a literature prior.  It is measured equal to the Nahm sum on the
BPS spectrum (`elem_traces._vacuum_rps`, the exact spectrum route, kept as the
witness) through `𝖖²⁰`; the levels `u = 4, 7` in place of 5 disagree from `𝖖⁶`
and `𝖖⁸`.

How the recipes were found and what certifies them (MEASURED, not derived).
A research pass (2026-09-23) decomposed the seeds by U(1) charge: after
multiplying by `(q;q)_∞ θ(μ)` every charge sector of each product `th·ψ_aψ_b` is
a monomial times a Virasoro M(3,5) character (measured through `q¹²⁰`), and each
seed numerator is a sparse integer combination of the ten products, selected on
the orthonormality bootstrap's values through `𝖖¹¹`–`𝖖¹⁴` and validated on
deeper orders.  The recipes were then re-implemented from their text alone and
compared with the corrected, certified seed bootstrap
(`u1_bootstrap.generate_u1`, which rests on its recorded
μ-support hypothesis): all nine representatives, and through the ρ-rule all 90
seeds, are equal through `𝖖²⁰`.  The reading of the ten products as characters
of Bershadsky–Polyakov modules at level `−12/5` is a literature hypothesis,
confirmed only for the vacuum; nothing here depends on it.

Series are exact: dicts `{(𝖖-power, μ-power): int}`.  Pure Python.
"""
from __future__ import annotations

from collections import defaultdict

__all__ = ["RECIPES", "numerator", "representative_trace", "seed_trace",
           "rho_orbit_representative", "vacuum_trace"]

# (coefficient, 𝖖-shift s, μ-shift t, theta product, a, b): c · 𝖖^s μ^t · th·ψ_aψ_b
_N5 = [(1, -1, 0, "th2", 1, 2), (1, -1, 0, "th2", 2, 3), (1, -1, 0, "th2", 0, 4),
       (-1, -1, 0, "th4", 1, 3), (-1, -1, 0, "th4", 0, 3), (-1, -1, 0, "th4", 1, 4),
       (1, 1, 0, "th2", 0, 4)]
_N0 = [(1, -3, -1, "th2", 1, 2), (-1, -3, -1, "th4", 1, 3),
       (1, -2, 0, "th2", 3, 4), (-1, -2, 0, "th4", 2, 4),
       (1, -1, -1, "th2", 0, 1), (1, -1, -1, "th2", 1, 2), (1, -1, -1, "th2", 0, 4),
       (-1, -1, -1, "th4", 0, 2), (-1, -1, -1, "th4", 0, 3),
       (1, 0, 0, "th2", 3, 4)]
_R7 = [(2, -1, 0, "th2", 1, 2), (2, -1, 0, "th2", 2, 3), (1, -1, 0, "th2", 0, 4),
       (-1, -1, 0, "th4", 0, 2), (-1, -1, 0, "th4", 2, 4), (-1, -1, 0, "th4", 1, 3),
       (-1, -1, 0, "th4", 0, 3), (-1, -1, 0, "th4", 1, 4),
       (1, 0, -1, "th2", 0, 1), (1, 0, 1, "th2", 3, 4), (1, 1, 0, "th2", 0, 4)]
_R3 = [(2, -2, 0, "th2", 1, 2), (2, -2, 0, "th2", 2, 3), (-2, -2, 0, "th4", 1, 3),
       (-1, -2, 0, "th4", 0, 3), (-1, -2, 0, "th4", 1, 4),
       (1, -1, -1, "th2", 0, 1), (1, -1, 1, "th2", 3, 4), (2, 0, 0, "th2", 0, 4)]


def _merge(*parts):
    """Merge term lists `(shift_s, terms)`: add `shift_s` to each 𝖖-shift and sum
    the coefficients of equal monomials."""
    acc = defaultdict(int)
    for ds, terms in parts:
        for (c, s, t, th, a, b) in terms:
            acc[(s + ds, t, th, a, b)] += c
    return sorted(((c,) + k for k, c in acc.items() if c), key=lambda x: x[1:])


_N7 = _merge((2, _N5), (0, _R7))                   # N_7 = 𝖖² N_5 + R_7
RECIPES = {
    12: _merge((0, [(1, -1, -1, "th2", 0, 1), (1, -2, 0, "th2", 2, 3),
                    (-1, -2, 0, "th4", 2, 4)])),
    13: _merge((0, [(1, 0, -1, "th2", 0, 1), (1, -1, 0, "th2", 2, 3),
                    (1, -1, 0, "th2", 3, 4), (-1, -1, 0, "th4", 2, 4),
                    (-1, -1, 0, "th4", 1, 4)])),
    2: _merge((0, [(1, -2, -1, "th2", 1, 2), (1, -2, -1, "th2", 2, 3),
                   (-1, -2, -1, "th4", 1, 3), (-1, -2, -1, "th4", 0, 3),
                   (1, -1, 0, "th2", 3, 4), (1, 0, -1, "th2", 0, 4)])),
    5: _merge((0, _N5)),
    0: _merge((0, _N0)),
    1: _merge((0, _N0)),                            # seeds 0 and 1 have equal traces
    21: _merge((0, [(1, -1, -1, "th2", 0, 1), (2, -1, -1, "th2", 1, 2),
                    (-1, -1, -1, "th4", 0, 2), (-1, -1, -1, "th4", 1, 3),
                    (-1, -1, -1, "th4", 0, 3), (2, 0, 0, "th2", 3, 4),
                    (-1, 0, 0, "th4", 2, 4), (1, 1, -1, "th2", 0, 1),
                    (1, 1, -1, "th2", 1, 2), (1, 1, -1, "th2", 0, 4),
                    (-1, 1, -1, "th4", 0, 2), (-1, 1, -1, "th4", 0, 3),
                    (1, 2, 0, "th2", 3, 4)])),
    7: _N7,
    3: _merge((1, _N7), (0, _R3)),                  # N_3 = 𝖖 N_7 + R_3
}


# ---------------------------------------------------------------------------
# exact series arithmetic
# ---------------------------------------------------------------------------

def _mul(a, b, K):
    out = defaultdict(int)
    for (qa, ma), ca in a.items():
        for (qb, mb), cb in b.items():
            q = qa + qb
            if q <= K:
                out[(q, ma + mb)] += ca * cb
    return {k: v for k, v in out.items() if v}


def _shift(s, dq, dm):
    return {(q + dq, m + dm): v for (q, m), v in s.items()}


def _theta_mu(K):
    out, n = {}, 0
    while n * n <= K:
        out[(n * n, n)] = 1
        out[(n * n, -n)] = 1
        n += 1
    return out


def _psi(a, K):
    out = {}
    for n in range(-K - 2, K + 3):
        e = 5 * n * n + (2 * a - 4) * n
        if e <= K:
            out[(e, n)] = out.get((e, n), 0) + 1
    return out


def _prod_one_minus(exps, K):
    s = {(0, 0): 1}
    for e in exps:
        if e > K:
            continue
        t = dict(s)
        for (q, m), v in s.items():
            if q + e <= K:
                t[(q + e, m)] = t.get((q + e, m), 0) - v
        s = {k: v for k, v in t.items() if v}
    return s


def _th(r, K):
    ex, m = [], 0
    while 10 * m + min(r, 10 - r) <= K:
        ex += [10 * m + r, 10 * m + 10 - r]
        m += 1
    return _prod_one_minus(ex, K)


def _inverse(D, K):
    """`1/D` for `D = 1 + O(𝖖)` with Laurent-polynomial μ-coefficients."""
    if D.get((0, 0)) != 1 or any(q < 0 for q, _ in D):
        raise ValueError("_inverse: the series must be 1 + O(𝖖)")
    byq = defaultdict(dict)
    for (q, m), v in D.items():
        byq[q][m] = v
    inv = {0: {0: 1}}
    for k in range(1, K + 1):
        acc = defaultdict(int)
        for j in range(1, k + 1):
            for m1, c1 in byq.get(j, {}).items():
                for m2, c2 in inv.get(k - j, {}).items():
                    acc[m1 + m2] -= c1 * c2
        inv[k] = {m: c for m, c in acc.items() if c}
    return {(q, m): c for q, d in inv.items() for m, c in d.items() if c}


_MARGIN = 6          # the recipes' most negative 𝖖-shift is −3
_BLOCKS = {}         # depth -> (products, 1/((q;q)θ(μ)))


def _blocks(K):
    """The ten products and `1/((q;q)_∞ θ(μ))`, exact through `𝖖^{K+_MARGIN}`."""
    for KK, val in _BLOCKS.items():
        if KK >= K + _MARGIN:
            return KK, val
    KK = K + _MARGIN
    psi = [_psi(a, KK) for a in range(5)]
    ths = {"th2": _th(2, KK), "th4": _th(4, KK)}
    X = {(th, a, b): _mul(ths[th], _mul(psi[a], psi[b], KK), KK)
         for th in ths for a in range(5) for b in range(a + 1, 5)}
    qq = _prod_one_minus([2 * n for n in range(1, KK // 2 + 1)], KK)
    Dinv = _inverse(_mul(qq, _theta_mu(KK), KK), KK)
    _BLOCKS.clear()
    _BLOCKS[KK] = (X, Dinv)
    return KK, (X, Dinv)


def numerator(rep, K):
    """`N_rep` through `𝖖^{K+6}` (a representative of `RECIPES`)."""
    if rep not in RECIPES:
        raise KeyError(f"e7_seeds.numerator: {rep} is not a ρ-orbit representative "
                       f"with a recipe ({sorted(RECIPES)})")
    KK, (X, _) = _blocks(K)
    out = defaultdict(int)
    for (c, s, t, th, a, b) in RECIPES[rep]:
        for (q, m), v in X[(th, a, b)].items():
            if q + s <= KK:
                out[(q + s, m + t)] += c * v
    return {k: v for k, v in out.items() if v}


def representative_trace(rep, K):
    """`T_rep = N_rep / ((q;q)_∞ θ(μ))` through `𝖖^K`."""
    KK, (_, Dinv) = _blocks(K)
    T = _mul(numerator(rep, K), Dinv, K)
    if any(q <= 0 for q, _ in T):
        raise ArithmeticError(f"e7_seeds: seed {rep} came out with a 𝖖^{{≤0}} term; "
                              f"a seed trace must be O(𝖖)")
    return T


# ---------------------------------------------------------------------------
# the other 81 seeds: ρ-equivariance of the trace
# ---------------------------------------------------------------------------

_ORBITS = None       # seed -> (representative, [the ρ-steps from it, as seeds])


def _orbits():
    global _ORBITS
    if _ORBITS is None:
        from finite_e7_kalg import E7_RHO_PERM
        perm = {int(k): int(v) for k, v in E7_RHO_PERM.items()}
        orb = {}
        for rep in RECIPES:
            x, path = rep, []
            while True:
                if x in orb and orb[x][0] != rep:
                    raise AssertionError(f"e7_seeds: seed {x} lies on the ρ-orbits of "
                                         f"{orb[x][0]} and {rep}")
                orb.setdefault(x, (rep, list(path)))
                path.append(x)
                x = perm.get(x, x)
                if x == rep:
                    break
        if len(orb) != 90:
            raise AssertionError(f"e7_seeds: the recipes' ρ-orbits cover {len(orb)} of "
                                 f"the 90 seeds")
        _ORBITS = orb
    return _ORBITS


def rho_orbit_representative(i):
    """The representative (a key of `RECIPES`) on seed `i`'s ρ-orbit."""
    return _orbits()[i][0]


def _transport(T, steps):
    from finite_e7_kalg import E7_RHO_DELTA
    for x in steps:            # T_{ρ(x)}(μ) = μ^{−δ_x} T_x(μ⁻¹)
        d = E7_RHO_DELTA.get(x, E7_RHO_DELTA.get(str(x), (0,)))[0]
        T = {(q, -m - d): v for (q, m), v in T.items()}
    return T


def seed_trace(i, K):
    """The trace of seed `i` (`0 ≤ i < 90`) through `𝖖^K`, in the zoo's data
    format `{𝖖-power: {(μ-power,): int}}`."""
    rep, steps = _orbits()[i]
    T = _transport(representative_trace(rep, K), steps)
    out = defaultdict(dict)
    for (q, m), v in T.items():
        out[q][(m,)] = v
    return {q: dict(sorted(d.items())) for q, d in sorted(out.items())}


# ---------------------------------------------------------------------------
# the vacuum: the Bershadsky–Polyakov product at level −12/5
# ---------------------------------------------------------------------------

def _times_lin(s, c, e, m, K):
    """`s · (1 + c 𝖖^e μ^m)`."""
    out = dict(s)
    for (q, mm), v in s.items():
        if q + e <= K:
            out[(q + e, mm + m)] = out.get((q + e, mm + m), 0) + c * v
    return {k: v for k, v in out.items() if v}


def _over_lin(s, c, e, m, K):
    """`s / (1 + c 𝖖^e μ^m)`, `e ≥ 1`: `out = s − c 𝖖^e μ^m · out`, order by order."""
    byq = defaultdict(dict)
    for (q, mm), v in s.items():
        byq[q][mm] = v
    out = {}
    for q in range(K + 1):
        row = dict(byq.get(q, {}))
        for mm, v in out.get(q - e, {}).items():
            row[mm + m] = row.get(mm + m, 0) - c * v
        row = {k: v for k, v in row.items() if v}
        if row:
            out[q] = row
    return {(q, mm): v for q, r in out.items() for mm, v in r.items()}


_U = 5                   # the boundary level k = −3 + 3/u


def vacuum_trace(K):
    """`Tr(1)` through `𝖖^K`, in the zoo's data format `{𝖖-power: {(μ-power,): int}}`."""
    s = {(0, 0): 1}
    n = 1
    while 2 * n <= K:
        s = _over_lin(s, -1, 2 * n, 0, K)
        s = _over_lin(s, -1, 2 * n + 2, 0, K)
        s = _over_lin(s, +1, 2 * n + 1, +1, K)
        s = _over_lin(s, +1, 2 * n + 1, -1, K)
        n += 1
    n = 1
    while 2 * _U * n - 2 <= K:
        e = 2 * _U * n
        for (c, q, m) in ((-1, e, 0), (-1, e, 0), (-1, e - 2, 0), (-1, e + 2, 0),
                          (+1, e - 1, +1), (+1, e - 1, -1), (+1, e + 1, +1), (+1, e + 1, -1)):
            s = _times_lin(s, c, q, m, K)
        n += 1
    out = defaultdict(dict)
    for (q, m), v in s.items():
        out[q][(m,)] = v
    return {q: dict(sorted(d.items())) for q, d in sorted(out.items())}
