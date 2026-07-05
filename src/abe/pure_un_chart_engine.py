"""pure_un_chart_engine — the stateless pure-U(N) chart engine.

The closed-form DOp images of the canonical generators (dressed E/F
monopole rays, det, Wilson characters) and the truncated Schur-measure /
Pochhammer / Kostka helpers.  **Stateless**: pure functions of
`(labels, N, K)` over `abelianized_torus` types — no class, no registry.

This engine (Plan 30 T5a, 2026-06-11) is load-bearing for the canonical
surface (`pure_un_kalgebra`, the U(N)+N_f and quiver flows).
"""
from __future__ import annotations

from typing import Sequence

from abelianized_torus import DOp, VRational, VLaurent
from laurent_poly import LaurentPoly


def _v_monomial(exponents: Sequence[int], N: int) -> VLaurent:
    """`v_0^{e_0} ⋯ v_{N−1}^{e_{N−1}}` as a `VLaurent` at chart rank `N`."""
    e = tuple(int(x) for x in exponents)
    if len(e) != N:
        raise ValueError(f"_v_monomial: expected {N} exponents, got {len(e)}")
    return VLaurent({e: LaurentPoly.one()}, n=N)


def _det_img(p: int, N: int) -> DOp:
    """`det^p = (u_0 u_1 ⋯ u_{N−1})^p`."""
    p = int(p)
    if p == 0:
        return DOp.one(N)
    base = DOp.one(N)
    for i in range(N):
        base = base * DOp.u(i, 1 if p > 0 else -1, n=N)
    out = DOp.one(N)
    for _ in range(abs(p)):
        out = out * base
    return out


def _ray_img_E(n: int, electric: Sequence[int], N: int) -> DOp:
    """`E_{n, f}` image per coulomb-2026 eq (786), with the Wilson
    decoration `f` taken to be the **S_n × S_{N-n}-symmetric** monomial
    polynomial determined by the partition `electric`.

        E_{n, f}  =  Σ_{|J|=n}  f(v_J, v_{notJ})  ·  Π_{r∈J, s∉J}
                                1/(1 − v_s/v_r)  ·  Π_{j∈J} u_j

    Wilson convention: `electric` is an `N`-tuple `(e_0, …, e_{N−1})`.
    The first `n` entries are the J-side partition; the remaining
    `N − n` are the non-J-side partition.  The function sums over

      * J subsets of size `n` (Weyl-orbit of the magnetic ϖ_n),
      * permutations of the J-side partition (`m_(e_1, …, e_n)`),
      * permutations of the non-J-side partition (`m_(e_{n+1}, …, e_N)`).

    This makes `f` a genuine `S_n × S_{N-n}`-invariant polynomial — the
    Weyl-symmetrisation required for canonical-basis elements.

    Linearised denominator form: `1/(1 − v_s/v_r) = v_r/(v_r − v_s)`,
    matching `abelianized_torus.VRational.root_inv(i, j, m, n)` for
    the `(v_i − q^m v_j)` factor (here `m = 0`)."""
    from itertools import combinations, permutations
    if not (0 <= n <= N):
        raise ValueError(f"_ray_img_E: need 0 <= n <= N, got n={n}, N={N}")
    if len(electric) != N:
        raise ValueError(
            f"_ray_img_E: electric length {len(electric)} != N {N}")
    e_J = tuple(int(x) for x in electric[:n])
    e_notJ = tuple(int(x) for x in electric[n:])
    # Levi gl_n x gl_(N-n) Schur-CHARACTER dressing: s_{e_J}(v_J) s_{e_notJ}(v_notJ),
    # as (exponent-perm, integer Kostka coeff) terms on each side.
    J_terms = _schur_char_terms(e_J) if n > 0 else {(): 1}
    notJ_terms = _schur_char_terms(e_notJ) if N - n > 0 else {(): 1}
    out = None
    for J in combinations(range(N), n):
        notJ = tuple(k for k in range(N) if k not in J)
        for J_perm, cJ in J_terms.items():
            for notJ_perm, cN in notJ_terms.items():
                v_exp = [0] * N
                for k, j in enumerate(J):
                    v_exp[j] = J_perm[k]
                for k, s in enumerate(notJ):
                    v_exp[s] = notJ_perm[k]
                for r in J:
                    v_exp[r] += (N - n)
                coef = VRational.from_vlaurent(
                    _v_monomial(v_exp, N)) * VRational.from_scalar(
                        LaurentPoly({0: cJ * cN}), n=N)
                for r in J:
                    for s in notJ:
                        if r < s:
                            coef = coef * VRational.root_inv(r, s, 0, n=N)
                        else:
                            coef = coef * VRational.root_inv(s, r, 0, n=N)
                            coef = coef * VRational.from_scalar(
                                LaurentPoly({0: -1}), n=N)
                u_exp = tuple(1 if i in J else 0 for i in range(N))
                term = DOp({u_exp: coef}, n=N, n_gauge=N)
                out = term if out is None else out + term
    return out if out is not None else DOp.one(N)


def _ray_img_F(n: int, electric: Sequence[int], N: int) -> DOp:
    """`F_{n, f}` image per coulomb-2026 eq (788):

        F_{n, f}  =  Σ_{|J|=n}  f(v_J, v_{notJ})  ·  Π_{r∈J, s∉J}
                                1/(1 − v_r/v_s)  ·  Π_{j∈J} u_j⁻¹

    Differs from E_{n, f} in TWO ways:
      * Vandermonde flipped: `1/(1 − v_r/v_s) = v_s/(v_s − v_r)`
        (vs E's `v_r/(v_r − v_s)`).
      * Shifts negated on J: `Π u_j⁻¹` (vs E's `Π u_j`).

    The magnetic charge is in the Weyl orbit of `-ϖ_n = ϖ_n^*` —
    the `{(-1)^n, 0^{N-n}}` orbit, NOT `E_n · det^{-1}` which sits
    in the `ϖ_{N-n}` orbit (Weyl-distinct for N ≥ 3).

    Wilson convention and symmetrisation identical to `_ray_img_E`."""
    from itertools import combinations, permutations
    if not (0 <= n <= N):
        raise ValueError(f"_ray_img_F: need 0 <= n <= N, got n={n}, N={N}")
    if len(electric) != N:
        raise ValueError(
            f"_ray_img_F: electric length {len(electric)} != N {N}")
    e_J = tuple(int(x) for x in electric[:n])
    e_notJ = tuple(int(x) for x in electric[n:])

    # Levi gl_n x gl_(N-n) Schur-CHARACTER dressing (identical convention to _ray_img_E).
    J_terms = _schur_char_terms(e_J) if n > 0 else {(): 1}
    notJ_terms = _schur_char_terms(e_notJ) if N - n > 0 else {(): 1}
    out = None
    for J in combinations(range(N), n):
        notJ = tuple(k for k in range(N) if k not in J)
        for J_perm, cJ in J_terms.items():
            for notJ_perm, cN in notJ_terms.items():
                v_exp = [0] * N
                for k, j in enumerate(J):
                    v_exp[j] = J_perm[k]
                for k, s in enumerate(notJ):
                    v_exp[s] = notJ_perm[k]
                # Flipped Vandermonde: v_s numerator for each s ∈ notJ,
                # collected per r ∈ J -> v_s^{|J|} = v_s^n.
                for s in notJ:
                    v_exp[s] += n
                coef = VRational.from_vlaurent(
                    _v_monomial(v_exp, N)) * VRational.from_scalar(
                        LaurentPoly({0: cJ * cN}), n=N)
                for r in J:
                    for s in notJ:
                        if r < s:
                            coef = coef * VRational.root_inv(r, s, 0, n=N)
                            # 1/(v_r - v_s) = -1/(v_s - v_r), but our
                            # target is 1/(v_s - v_r), so flip sign.
                            coef = coef * VRational.from_scalar(
                                LaurentPoly({0: -1}), n=N)
                        else:
                            coef = coef * VRational.root_inv(s, r, 0, n=N)
                u_exp = tuple(-1 if i in J else 0 for i in range(N))
                term = DOp({u_exp: coef}, n=N, n_gauge=N)
                out = term if out is None else out + term
    return out if out is not None else DOp.one(N)
    """Monomial-symmetric Wilson character `m_{electric}(v)` for `electric`
    a weakly-decreasing N-tuple (Weyl-orbit-sum over distinct permutations)."""
    e = sorted((int(x) for x in electric), reverse=True)
    if len(e) != N:
        raise ValueError(f"_wilson_sym: expected {N}-tuple, got {len(e)}")
    # Collect distinct permutations of `e`.
    seen: set[tuple[int, ...]] = set()
    def _perms(prefix: tuple[int, ...], rest: tuple[int, ...]):
        if not rest:
            seen.add(prefix)
            return
        used: set[int] = set()
        for i, x in enumerate(rest):
            if x in used:
                continue
            used.add(x)
            _perms(prefix + (x,), rest[:i] + rest[i+1:])
    _perms((), tuple(e))
    vl = VLaurent.zero(N)
    for perm in seen:
        vl = vl + VLaurent({perm: LaurentPoly.one()}, n=N)
    return VRational.from_vlaurent(vl)


def _kostka(lam, mu) -> int:
    """Kostka number `K_{lam, mu}` = #SSYT of shape `lam` (partition) and
    content `mu` (composition), via backtracking."""
    lam = [int(x) for x in lam if x > 0]
    mu = [int(x) for x in mu]
    if sum(lam) != sum(mu):
        return 0
    rows = len(lam)
    T = [[0] * lam[r] for r in range(rows)]
    counts = list(mu)
    cnt = [0]
    cells = [(r, c) for r in range(rows) for c in range(lam[r])]

    def bt(i):
        if i == len(cells):
            cnt[0] += 1
            return
        r, c = cells[i]
        lo = 1
        if c > 0:
            lo = max(lo, T[r][c - 1])          # weakly increasing along row
        for v in range(lo, len(mu) + 1):
            if counts[v - 1] == 0:
                continue
            if r > 0 and v <= T[r - 1][c]:      # strictly increasing down column
                continue
            T[r][c] = v
            counts[v - 1] -= 1
            bt(i + 1)
            counts[v - 1] += 1
            T[r][c] = 0

    bt(0)
    return cnt[0]


def _partitions_n_parts(total: int, parts: int):
    """Weakly-decreasing nonnegative `parts`-tuples summing to `total`."""
    res = []

    def rec(prefix, remaining, slots, cap):
        if slots == 0:
            if remaining == 0:
                res.append(tuple(prefix))
            return
        for v in range(min(cap, remaining), -1, -1):
            rec(prefix + [v], remaining - v, slots - 1, v)

    rec([], total, parts, total)
    return res


def _distinct_perms_tuple(t):
    seen = set()

    def _perms(prefix, rest):
        if not rest:
            seen.add(prefix)
            return
        used = set()
        for i, x in enumerate(rest):
            if x in used:
                continue
            used.add(x)
            _perms(prefix + (x,), rest[:i] + rest[i + 1:])

    _perms((), tuple(t))
    return seen


def _schur_char_terms(weight) -> dict:
    """Irreducible `gl_k` (Schur) character `s_weight` on `k = len(weight)`
    variables, as `{monomial-exponent-tuple: integer coeff}`.

    Expands the Schur function in monomials with Kostka coefficients,
    `s_lambda = sum_{mu <= lambda} K_{lambda mu} m_mu`, after shifting
    `weight` by `c*(1,...,1)` to a non-negative partition (the `det^{-c}`
    twist is undone on the exponents).  This is the canonical-basis
    dressing — for `gl_1` (k=1) it is the single monomial `v^weight`."""
    w = sorted((int(x) for x in weight), reverse=True)
    k = len(w)
    if k == 0:
        return {(): 1}
    c = -min(w) if min(w) < 0 else 0
    lam = tuple(x + c for x in w)                   # non-negative partition
    out: dict = {}
    for mu in _partitions_n_parts(sum(lam), k):
        K = _kostka(lam, mu)
        if not K:
            continue
        for perm in _distinct_perms_tuple(tuple(x - c for x in mu)):  # undo det-twist
            out[perm] = out.get(perm, 0) + K
    return out


def _wilson_sym(electric: Sequence[int], N: int) -> VRational:
    """Irreducible `gl_N` (Schur) character `chi_{electric}(v)` for
    `electric` a dominant (weakly-decreasing) `N`-tuple weight.

    This is the **canonical-basis Wilson character**, uniform in `N` (see
    `_schur_char_terms`).  For `N = 2` it reproduces
    `chi_(a,b) = sum_{j=0}^{a-b} v_1^{a-j} v_2^{b+j}`.

    Earlier this returned the bare monomial symmetric `m_lambda` for
    `N >= 3` (not canonical) — that made the Wilson sector non-
    orthonormal at `N >= 3` and inconsistent with the `N = 2` character.
    """
    e = [int(x) for x in electric]
    if len(e) != N:
        raise ValueError(f"_wilson_sym: expected {N}-tuple, got {len(e)}")
    vl = VLaurent.zero(N)
    for perm, K in _schur_char_terms(e).items():
        vl = vl + VLaurent({perm: LaurentPoly({0: K})}, n=N)
    return VRational.from_vlaurent(vl)


def _q_poch_q2_q2_truncated(K: int) -> LaurentPoly:
    """`(q²; q²)_∞` truncated to terms `q^e` with `e ≤ K`.

    `(q²; q²)_∞ = Π_{k≥1} (1 − q^{2k})`.  Only factors with `2k ≤ K`
    can contribute non-trivially to the truncated product."""
    result = LaurentPoly({0: 1})
    for k in range(1, K // 2 + 1):
        result = _laurent_truncate(result * LaurentPoly({0: 1, 2 * k: -1}), K)
    return result


def _laurent_truncate(p: LaurentPoly, K: int) -> LaurentPoly:
    """Truncate `p` to q^e with e ≤ K (drop higher-degree terms)."""
    out = {e: c for e, c in p._coeffs.items() if e <= K}
    return LaurentPoly._from_clean_dict(out)


def _vlaurent_truncate(p: "VLaurent", K: int) -> "VLaurent":
    """Truncate each LaurentPoly coefficient of `p` to q^e with e ≤ K."""
    out: dict = {}
    for ve, c in p._terms.items():
        ct = _laurent_truncate(c, K)
        if not ct.is_zero():
            out[ve] = ct
    return VLaurent._from_clean_terms(out, p._n)


def _schur_measure_truncated(N: int, K: int) -> "VLaurent":
    """Pure-U(N) Schur measure

        ∏_{i<j} (v_i/v_j; q²)_∞ (q² v_i/v_j; q²)_∞

    truncated to q^K, as a VLaurent over LaurentPoly(q).

    Each (a; q²)_∞ = Π_{k≥0} (1 − q^{2k} a) truncates to `k ≤ K/2`
    factors.  Combined product:

        (v_i/v_j; q²)_∞ · (q² v_i/v_j; q²)_∞
            = (1 − v_i/v_j) · Π_{k≥1} (1 − q^{2k} v_i/v_j)^2

    Polynomial in (v_i/v_j) factors → Laurent polynomial in v with
    LaurentPoly(q) coefficients.  Computationally expensive at high
    N, K (degree ~ K per pair (i, j))."""
    result = VLaurent({(0,) * N: LaurentPoly({0: 1})}, n=N)
    # Sum over ordered pairs (i, j) with i != j gives the full
    # U(N) gauge measure (i<j alone is the SU(N) convention; the
    # extra (v_j/v_i;q²)_∞ factors are what produces the N! Weyl
    # cancellation that matches `1/N!` in the prefactor).
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            e_ratio = tuple((1 if k == i else (-1 if k == j else 0))
                            for k in range(N))
            # k=0 factor: (1 − v_i/v_j)
            factor = VLaurent({(0,)*N: LaurentPoly({0: 1}),
                               e_ratio: LaurentPoly({0: -1})}, n=N)
            result = _vlaurent_truncate(result * factor, K)
            # k ≥ 1 factors: (1 − q^{2k} v_i/v_j)
            for k in range(1, K // 2 + 1):
                factor = VLaurent({(0,)*N: LaurentPoly({0: 1}),
                                   e_ratio: LaurentPoly({2 * k: -1})}, n=N)
                result = _vlaurent_truncate(result * factor, K)
    return result

