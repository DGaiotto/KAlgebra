"""Exact Schur contour functional for SU(2) with four fundamental
hypermultiplets — the measure side of the K_q conjecture.

The functional is the Schur-index contour integral

    <f>  =  (1/2) [z^0]  N(z) / D(z) * f(z),

        N(z) = (q;q)^2 (z^2;q) (q z^2;q) (z^-2;q) (q z^-2;q)
        D(z) = prod_{i=1..4} prod_{eps,del = +-1}
                   (q^{1/2} z^eps mu_i^del ; q)_infty

(SU(2) Haar measure folded into N; half-hypers in (2, 8_v) of
SU(2) x SO(8), with mu_i the SO(8) ORTHOGONAL-frame fugacities, so the
8_v weights are mu_i^{+-1}).  Everything is exact integer arithmetic
on the coefficient lattice Z[z^{+-1}, mu_i^{+-1}] graded by half-integer
q-powers, truncated at a chosen top power AT THE END of each
multiplication (formal-series extraction: each q-coefficient receives
finitely many exact contributions, so this is not the dangerous early
truncation of the exact-arithmetic rule).

Series are dicts  {(qh, z, m1, m2, m3, m4): int}  with q-exponent
qh/2 and z, m_i the fugacity exponents.

Insertions are z-Laurent polynomials  {z_exp: int}; `chi(k)` gives the
SU(2) character of integer spin k in adjoint-normalised units
(z^{2m}, m = -k..k) — the family matching the canonical curve tower's
GNO-dual fusion L_j L_k = sum L_l — and `chi_half(d)` the d-dimensional
character (z-steps of 2 from z^{d-1} down), for the odd/spinor-class
probes.
"""

from __future__ import annotations

NFLAV = 4
ZERO_MU = (0,) * NFLAV


def series_mul(A: dict, B: dict, qh_max: int) -> dict:
    out: dict = {}
    for ka, va in A.items():
        qa = ka[0]
        if qa > qh_max:
            continue
        for kb, vb in B.items():
            qh = qa + kb[0]
            if qh > qh_max:
                continue
            key = (qh, ka[1] + kb[1],
                   ka[2] + kb[2], ka[3] + kb[3],
                   ka[4] + kb[4], ka[5] + kb[5])
            c = out.get(key, 0) + va * vb
            if c:
                out[key] = c
            else:
                out.pop(key, None)
    return out


def _qpow(n: int, z: int = 0, mu: tuple = ZERO_MU, coeff: int = 1) -> dict:
    return {(n, z) + mu: coeff}


def one() -> dict:
    return {(0, 0) + ZERO_MU: 1}


def invqq_series(k: int, qh_max: int) -> dict:
    """1/(q;q)_k as an integer q-series: partitions into parts <= k."""
    top = qh_max // 2
    p = [1] + [0] * top
    for part in range(1, k + 1):
        for n in range(part, top + 1):
            p[n] += p[n - part]
    return {(2 * n, 0) + ZERO_MU: p[n] for n in range(top + 1) if p[n]}


def euler_series(qh_max: int) -> dict:
    """(q;q)_infty truncated."""
    s = one()
    for n in range(1, qh_max // 2 + 1):
        s = series_mul(s, {(0, 0) + ZERO_MU: 1, (2 * n, 0) + ZERO_MU: -1},
                       qh_max)
    return s


def pochhammer_z(z_exp: int, n_from: int, qh_max: int) -> dict:
    """prod_{n >= n_from} (1 - q^n z^{z_exp}) truncated."""
    s = one()
    for n in range(n_from, qh_max // 2 + 1):
        s = series_mul(s, {(0, 0) + ZERO_MU: 1,
                           (2 * n, z_exp) + ZERO_MU: -1}, qh_max)
    return s


def numerator(qh_max: int) -> dict:
    e = euler_series(qh_max)
    s = series_mul(e, e, qh_max)
    s = series_mul(s, pochhammer_z(+2, 0, qh_max), qh_max)
    s = series_mul(s, pochhammer_z(+2, 1, qh_max), qh_max)
    s = series_mul(s, pochhammer_z(-2, 0, qh_max), qh_max)
    s = series_mul(s, pochhammer_z(-2, 1, qh_max), qh_max)
    return s


def inv_hyper_factor(z_eps: int, i: int, delta: int, qh_max: int) -> dict:
    """1/(q^{1/2} z^eps mu_i^delta ; q)_infty = sum_k x^k/(q;q)_k."""
    s: dict = {}
    for k in range(qh_max + 1):
        mu = [0] * NFLAV
        mu[i] = delta * k
        xk = {(k, z_eps * k) + tuple(mu): 1}
        term = series_mul(xk, invqq_series(k, qh_max - k), qh_max)
        for key, v in term.items():
            s[key] = s.get(key, 0) + v
    return s


def full_measure(qh_max: int) -> dict:
    """N(z)/D(z) truncated at q^{qh_max/2}."""
    s = numerator(qh_max)
    for i in range(NFLAV):
        for delta in (1, -1):
            for eps in (1, -1):
                s = series_mul(s, inv_hyper_factor(eps, i, delta, qh_max),
                               qh_max)
    return s


def chi(k: int) -> dict:
    """SU(2) integer-spin-k character, adjoint-normalised: sum z^{2m}."""
    return {2 * m: 1 for m in range(-k, k + 1)}


def chi_half(dim: int) -> dict:
    """d-dimensional SU(2) character: z^{d-1} + z^{d-3} + ... + z^{1-d}."""
    return {dim - 1 - 2 * j: 1 for j in range(dim)}


def moment(measure: dict, insertion: dict) -> dict:
    """<f> = (1/2)[z^0] measure * f, as {(qh, m1..m4): int}."""
    acc: dict = {}
    for key, v in measure.items():
        ze = key[1]
        f = insertion.get(-ze)
        if not f:
            continue
        mk = (key[0],) + key[2:]
        acc[mk] = acc.get(mk, 0) + v * f
    out = {}
    for mk, v in acc.items():
        if v % 2 != 0:
            raise ArithmeticError(f"odd z^0 coefficient at {mk}: {v}")
        if v:
            out[mk] = v // 2
    return out


def insertion_product(f: dict, g: dict) -> dict:
    out: dict = {}
    for a, va in f.items():
        for b, vb in g.items():
            out[a + b] = out.get(a + b, 0) + va * vb
    return out


def by_q(mom: dict) -> dict:
    """Regroup a moment {(qh, mu): c} as {qh: {mu: c}}."""
    out: dict = {}
    for key, v in mom.items():
        out.setdefault(key[0], {})[key[1:]] = v
    return out


# --- general punctured-sphere (linear quiver) functional --------------
#
# Class-S linear quiver for S^2_{0,n}: gauge SU(2)^{n-3} with fugacities
# z_1..z_{n-3}; matter = trifundamental half-hypers, one per trinion:
#   end:    (mu_1, mu_2, z_1)   and   (z_{n-3}, mu_{n-1}, mu_n)
#   middle: (z_i, z_{i+1}, mu_{i+2})       i = 1..n-4
# Keys: (qh, z_1..z_m, mu_1..mu_n).

def sphere_keylen(n: int) -> tuple[int, int]:
    m = n - 3
    return m, n


def _smul(A, B, qh_max, klen):
    out = {}
    for ka, va in A.items():
        qa = ka[0]
        if qa > qh_max:
            continue
        for kb, vb in B.items():
            qh = qa + kb[0]
            if qh > qh_max:
                continue
            key = (qh,) + tuple(ka[i] + kb[i] for i in range(1, klen))
            c = out.get(key, 0) + va * vb
            if c:
                out[key] = c
            else:
                out.pop(key, None)
    return out


def sphere_measure(n: int, qh_max: int) -> dict:
    """N(z_1)...N(z_m)/D for S^2_{0,n}, truncated at q^{qh_max/2}."""
    m, nf = sphere_keylen(n)
    klen = 1 + m + nf
    zero = (0,) * (klen - 1)
    one_s = {(0,) + zero: 1}

    def shift_key(qh, zi=None, ze=0, mus=()):
        k = [qh] + [0] * (klen - 1)
        if zi is not None:
            k[1 + zi] = ze
        for (i, e) in mus:
            k[1 + m + i] = e
        return tuple(k)

    s = one_s
    e = euler_series(qh_max)
    e2 = series_mul(e, e, qh_max)
    e2 = {shift_key(k[0]): v for k, v in e2.items()}
    for zi in range(m):
        s = _smul(s, e2, qh_max, klen)
        for n0 in (0, 1):
            for sgn in (1, -1):
                # (q^{n0} z^{2 sgn}; q)_infty
                f = one_s
                for nn in range(n0, qh_max // 2 + 1):
                    f = _smul(f, {(0,) + zero: 1,
                                  shift_key(2 * nn, zi, 2 * sgn): -1},
                              qh_max, klen)
                s = _smul(s, f, qh_max, klen)
    trinions = []
    trinions.append([(("z", 0),), [0, 1]])
    for i in range(n - 4):
        trinions.append([(("z", i), ("z", i + 1)), [i + 2]])
    trinions.append([(("z", m - 1),), [nf - 2, nf - 1]])
    for gauges, flavs in trinions:
        legs = [g[1] for g in gauges]
        for eps in _signs(len(legs)):
            for deltas in _signs(len(flavs)):
                f = {}
                for k in range(qh_max + 1):
                    key = [k] + [0] * (klen - 1)
                    for zi, ez in zip(legs, eps):
                        key[1 + zi] = ez * k
                    for fi, df in zip(flavs, deltas):
                        key[1 + m + fi] = df * k
                    inv = invqq_series(k, qh_max - k)
                    for kq, v in inv.items():
                        kk = list(key)
                        kk[0] += kq[0]
                        if kk[0] <= qh_max:
                            kk = tuple(kk)
                            f[kk] = f.get(kk, 0) + v
                s = _smul(s, f, qh_max, klen)
    return s


def _signs(k):
    out = [[]]
    for _ in range(k):
        out = [s + [e] for s in out for e in (1, -1)]
    return out


def sphere_moment(n: int, measure: dict, insertion: dict) -> dict:
    """<f>; insertion maps (z_1exp..z_mexp) -> int; result
    {(qh, mu_1..mu_n): int}."""
    m, nf = sphere_keylen(n)
    acc = {}
    for key, v in measure.items():
        ze = key[1:1 + m]
        f = insertion.get(tuple(-x for x in ze))
        if not f:
            continue
        mk = (key[0],) + key[1 + m:]
        acc[mk] = acc.get(mk, 0) + v * f
    out = {}
    half = 2 ** m
    for mk, v in acc.items():
        if v % half != 0:
            raise ArithmeticError(f"z^0 coefficient not divisible at {mk}")
        if v // half:
            out[mk] = v // half
    return out
