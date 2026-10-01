"""URQTorus — the U(N) Rational Quantum Torus (enriched cone-atom presentation).

The leading **U** is the gauge group **U(N)**: a different group, or a parity
(τ) extension, carries a *different* G-cocycle (the √measure ρ) and is therefore a
*different class* — `URQTorus` bakes in U(N)'s ρ as a class attribute, and siblings
subclass it / supply their own `_RHO` rather than passing a runtime twist.

A `URQTorus` is a pure-U(N) chart operator presented on the **dominant cone atoms**
``U_m = u^m · ψ_m(v)`` (a single u-power with a closed-form rational dressing):

    x  =  Σ_{m∈Λ}  f_m(q^m v) · U_m          ↔  the residual vector  (f_m)_{m∈Λ} .

The **residual vector `(f_m)` is the primary content** other algorithms consume
(structure-constant recognition, traces, the leading Levi character, …).  The
`U_m` form a rational-2-cocycle-twisted quantum torus,
``U_a U_b = R_{a,b}(v) U_{a+b}`` (single u-power, NO sum), so the algebra is carried
by the closed forms `ψ, R` (multiply) and a **G-cocycle** (the √measure ρ); the
Schur trace is the magnetic-0 measure residue; the design covers ρ²-cyclicity,
the inner-product picture, and the q²-Levi-Vandermonde weight.

Conventions
-----------
`q = 𝖖`; `v_1..v_N` abelianized gauge variables; magnetic `m` an integer N-tuple,
*dominant* = descending.  Two shifts: the **full** normal-ordering `S_m: v→q^{2m}v`
(used by the cocycles) and the **half**-shift `v→q^m v` (used to pass between the
algebra-internal *bare* residual `c_m = d_m/ψ_m` and the public `f_m`, `c_m(v) =
f_m(q^m v)`).

LABEL CONVENTION — read this.  A canonical has exactly two valid Kapustin
't Hooft-Wilson labels, related by the joint `w_0` involution (reverse `m` AND `e`
*together*):

  * **lower Kapustin** — `(anti-dominant m, Levi-anti-dominant e)`: `m` ascending,
    `e` ascending within each Levi block;
  * **upper Kapustin** — `(dominant m, Levi-dominant e)`: both descending.

The mixed states `(dom m, anti-dom e)` and `(anti-dom m, dom e)` are NOT labels.
**Everything here uses LOWER Kapustin, uniformly** — `recognize_leading`,
`recognize_q_extreme`, the builders `PureUNKAlgebra.minuscule` / `wilson` (retired
2026-09-19), and the
engine's `labels()`/`urqt` (the canonical engine now speaks lower Kapustin);
`leading()` returns an anti-dominant `m`.  `recognize_*` read the residual in the
upper (dominant) frame internally and apply the joint `w_0` once, so their output
is lower — never a mixed state.

Storage / interop
-----------------
Internally the element holds the **bare residual** `c_m` (what the cocycle machinery
operates on); `f_m` is produced at the boundary by the half-shift, so callers see the
half-shift convention while the algebra stays direct.  *How* `f_m` is stored/computed
is deliberately behind this surface — the engine rework is free to replace the
reference closed forms below without changing the public class.  Legacy access
(a probe in the source repository, `to_U`) remains available in the interim.
"""
from __future__ import annotations

import itertools

from abelianized_torus import DOp, VRational, VLaurent
from laurent_poly import LaurentPoly
from math import factorial
import functools

from abelianized_torus import simplify_dop as _simp


# ===========================================================================
# Torus-level helpers re-homed from the retired type-A layer (2026-09-19).
# These were imported from `pure_un_kalgebra` / `pure_un_chart_engine` /
# `pure_un_closed_form` / `un_nf_dressed_generators` while those modules were
# live; when the optimized type-A classes were retired
# (PureUNKAlgebra, PureSUNKAlgebra, UNNfKAlgebra, UNQuiverKAlgebra) the
# URQ substrate keeps its own copies here, so the spine has no import edge
# into the source repository's archive.  They are the Schur-measure / Pochhammer / Vandermonde
# arithmetic, the Levi decomposition and the Kostka / Schur-monomial data of
# the U(N) torus — torus facts, not class facts.
# ===========================================================================

_MEASURE_CACHE: dict = {}
_POCH_CACHE: dict = {}
_INVPOCH_CACHE: dict = {}


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


def _q_poch_q2_q2_truncated(K: int) -> LaurentPoly:
    """`(q²; q²)_∞` truncated to terms `q^e` with `e ≤ K`.

    `(q²; q²)_∞ = Π_{k≥1} (1 − q^{2k})`.  Only factors with `2k ≤ K`
    can contribute non-trivially to the truncated product."""
    result = LaurentPoly({0: 1})
    for k in range(1, K // 2 + 1):
        result = _laurent_truncate(result * LaurentPoly({0: 1, 2 * k: -1}), K)
    return result


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


def _vdiv(num: dict, i: int, j: int, N: int) -> dict:
    """Exact division of {v-exp: {q-exp: int}} by the m=0 factor (v_i - v_j)."""
    work = {ve: dict(lp) for ve, lp in num.items()}; Q: dict = {}; guard = 0
    while work:
        guard += 1
        if guard > 1_000_000:
            raise RuntimeError("pure_un trace: (v_i-v_j) division did not terminate")
        hi = max(ve[i] for ve in work)
        for ve in [v for v in work if v[i] == hi]:
            lp = work.pop(ve)
            qe = list(ve); qe[i] = hi - 1; qe = tuple(qe)
            cur = Q.get(qe, {}); Q[qe] = {e: cur.get(e, 0) + lp.get(e, 0) for e in set(cur) | set(lp)}
            t2 = list(ve); t2[i] = hi - 1; t2[j] += 1; t2 = tuple(t2)
            cur = work.get(t2, {}); nw = {e: cur.get(e, 0) + lp.get(e, 0) for e in set(cur) | set(lp)}
            nw = {e: c for e, c in nw.items() if c}
            if nw: work[t2] = nw
            elif t2 in work: del work[t2]
    return {ve: lp for ve, lp in Q.items() if any(lp.values())}


def _inv_root(i: int, j: int, m: int, N: int, K: int) -> VLaurent:
    """q-series of 1/(v_i - q^m v_j) up to q^K (m != 0)."""
    out: dict = {}
    if m > 0:
        n = 0
        while m * n <= K:
            out[tuple((-n - 1) if t == i else (n if t == j else 0) for t in range(N))] = LaurentPoly({m * n: 1}); n += 1
    elif m < 0:
        am = -m; n = 0
        while am * (n + 1) <= K:
            out[tuple(n if t == i else (-n - 1 if t == j else 0) for t in range(N))] = LaurentPoly({am * (n + 1): -1}); n += 1
    else:
        raise ValueError("_inv_root: m=0 must be handled by numerator division")
    return VLaurent(out, n=N)


def _v0_coeff(s: "VLaurent", M: "VLaurent", K: int) -> LaurentPoly:
    """The `v^0` coefficient of the product `s·M`, `[s·M]_{v^0} = Σ_w s[w]·M[-w]`,
    truncated to `q^K` -- computed directly, WITHOUT forming the full v-product
    (only v^0 is needed for the trace residue).  O(|s|) LaurentPoly products vs
    O(|s|·|M|) for the full product."""
    out = LaurentPoly.zero()
    Mterms = M._terms
    for w, sc in s._terms.items():
        mc = Mterms.get(tuple(-x for x in w))
        if mc is not None:
            out = _laurent_truncate(out + sc * mc, K)
    return out


def _inv_qpoch_n(n: int, K: int) -> LaurentPoly:
    """`1/(q^2;q^2)_n = 1/∏_{l=1}^n (1 - q^{2l})` as a q-series to q^K."""
    key = (n, K)
    r = _INVPOCH_CACHE.get(key)
    if r is None:
        r = LaurentPoly({0: 1})
        for l in range(1, n + 1):
            geom = LaurentPoly({2 * l * t: 1 for t in range(K // (2 * l) + 1)})
            r = _laurent_truncate(r * geom, K)        # 1/(1-q^{2l}) = Σ q^{2l t}
        _INVPOCH_CACHE[key] = r
    return r


def _schur_measure_euler(N: int, K: int) -> "VLaurent":
    """Pure-U(N) Schur measure ∏_{i≠j} (v_i/v_j;q²)_∞ (q² v_i/v_j;q²)_∞ to q^K,
    via the **Euler / Habiro** expansion of each factor (quadratic q-power):

        (q^k v_i/v_j; q²)_∞ = Σ_n (-q^{k-1} v_i/v_j)^n q^{n²} / (q²;q²)_n
                            = Σ_n (-1)^n q^{n²+(k-1)n} (v_i/v_j)^n / (q²;q²)_n.

    Because the q-power is `n²+(k-1)n`, reaching q^K needs only `n ≲ √K` terms
    per factor (v-degree √K) -- vs the product form's ~K/2.  Exact, and far
    cheaper at the large internal cutoffs the q^0 pairings require."""
    result = VLaurent({(0,) * N: LaurentPoly({0: 1})}, n=N)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            e_ratio = tuple(1 if t == i else (-1 if t == j else 0)
                            for t in range(N))
            for k in (0, 2):                          # the two Pochhammer factors
                terms: dict = {}
                n = 0
                while n * n + (k - 1) * n <= K:
                    qpow = n * n + (k - 1) * n
                    coeff = _inv_qpoch_n(n, K - qpow)
                    sgn = -1 if n % 2 else 1
                    lp = LaurentPoly({qpow + e: sgn * c
                                      for e, c in coeff._coeffs.items()})
                    terms[tuple(n * x for x in e_ratio)] = lp
                    n += 1
                factor = VLaurent(terms, n=N)
                result = _vlaurent_truncate(result * factor, K)
    return result


def _cached_measure(N: int, K: int) -> "VLaurent":
    """Memoised Schur measure (the dominant cost of `trace`; rebuilt every call
    before)."""
    m = _MEASURE_CACHE.get((N, K))
    if m is None:
        m = _schur_measure_euler(N, K)
        _MEASURE_CACHE[(N, K)] = m
    return m


def _cached_poch2N(N: int, K: int) -> LaurentPoly:
    """Memoised `(q^2;q^2)_inf^{2N}` truncated to q^K."""
    p = _POCH_CACHE.get((N, K))
    if p is None:
        p = LaurentPoly({0: 1}); poch = _q_poch_q2_q2_truncated(K)
        for _ in range(2 * N):
            p = _laurent_truncate(p * poch, K)
        _POCH_CACHE[(N, K)] = p
    return p


def trace_v0(u0, N: int, K: int = 8, adaptive: bool = False,
             watch: int | None = None) -> LaurentPoly:
    """v-only Schur-measure residue of the magnetic-0 coefficient `u0` (a VRational
    in v): `Tr = (q²;q²)_∞^{2N}/N! · ∮ dv/v · measure · u0`.  The trace needs only
    the residual, no chart/DOp.  `u0=None` → 0."""
    if u0 is None:
        return LaurentPoly.zero()
    if u0._sq:
        raise NotImplementedError("pure_un trace: sq denominators not yet handled")
    num = {ve: dict(lp._coeffs) for ve, lp in u0._num._terms.items()}
    nz = []
    for (i, j, m), mult in u0._den.items():
        for _ in range(mult):
            if m == 0:
                num = _vdiv(num, i, j, N)
            else:
                nz.append((i, j, m))
    nf = factorial(N)

    def core(K_int: int) -> LaurentPoly:
        s = VLaurent({ve: LaurentPoly(lp) for ve, lp in num.items()}, n=N)
        for (i, j, m) in nz:
            s = _vlaurent_truncate(s * _inv_root(i, j, m, N, K_int), K_int)
        v0 = _v0_coeff(s, _cached_measure(N, K_int), K_int)   # only the v^0 residue
        if v0.is_zero():
            return LaurentPoly.zero()
        qs = _laurent_truncate(v0 * _cached_poch2N(N, K_int), K)
        # Genuine Schur coefficients are divisible by N! (Weyl average); near the
        # cutoff the truncated measure leaves an incomplete (non-divisible)
        # coefficient -- the first such order is the reliable boundary.
        out = {}
        for e in sorted(qs._coeffs):
            qd, r = divmod(qs._coeffs[e], nf)
            if r != 0:
                break
            if qd:
                out[e] = qd
        return LaurentPoly(out)

    if not adaptive:
        return core(K)
    # ADAPTIVE: the q^0/low-order coefficients are exact once the internal cutoff
    # reaches the operator's negative-q extent that actually feeds v^0.  The raw
    # |min-q| (over all numerator terms) is a SAFE upper bound but usually wildly
    # over-pads (its deep-q terms sit at high v-degree, reached only by high-q
    # measure terms, so they never feed low q).  Escalate from a small cutoff in
    # steps of 8 until the q<=K result stabilises -- fast for the common
    # over-padded case -- capped at the proven-safe K + |min-q|.
    min_q = 0
    for lp in num.values():
        if lp:
            min_q = min(min_q, min(lp))
    safe = ((K - min_q + 4 + 7) // 8) * 8
    K_int = min(safe, ((K + 4 + 7) // 8) * 8)
    res = core(K_int)
    while K_int < safe:
        K_int = min(safe, K_int + 8)
        nxt = core(K_int)
        if watch is None:
            if nxt._coeffs == res._coeffs:
                return nxt
        elif nxt._coeffs.get(watch, 0) == res._coeffs.get(watch, 0):
            return nxt
        res = nxt
    return res


def _vinv_vlaurent(vl: VLaurent, N: int) -> VLaurent:
    """Substitute v_i -> 1/v_i in a VLaurent (flip every v-exponent)."""
    return VLaurent({tuple(-x for x in w): lp for w, lp in vl._terms.items()}, n=N)


def _vinv_vrational(vr: VRational, N: int) -> VRational:
    """Substitute v_i -> 1/v_i in a VRational.  The numerator flips v-exponents;
    each denominator factor inverts as  1/(v_i - q^m v_j) -> (v_i v_j)/(v_j - q^m v_i)."""
    if vr._sq:
        raise NotImplementedError("_vinv_vrational: single-variable sq factors are "
                                  "unsupported (they never arise for pure U(N))")
    out = VRational.from_vlaurent(_vinv_vlaurent(vr.num, N))
    for (i, j, m), mult in vr.den.items():
        vij = VRational.from_vlaurent(VLaurent(
            {tuple(1 if t in (i, j) else 0 for t in range(N)): LaurentPoly({0: 1})}, n=N))
        inv = VRational.root_inv(j, i, m, N)           # 1/(v_j - q^m v_i)
        for _ in range(mult):
            out = out * vij * inv
    return out


def _rho_block_data(k, N):
    """`(sign, qpow, wexp)` of the per-u-block ρ conjugation at u-power `k`,
    qpow already including J's `q^{2 Σ_t t k_t}` grading shift."""
    S = sum(k)
    sign = -1 if (sum((N - 1 - 2 * t) * k[t] for t in range(N)) % 2) else 1
    qf = -sum((k[i] - k[j]) * (k[i] - k[j] - 1)
              for i in range(N) for j in range(i + 1, N))
    qJ = 2 * sum(t * k[t] for t in range(N))
    wexp = tuple(S - N * k[t] for t in range(N))
    return sign, qf + qJ, wexp


def _perm_vr(vr: VRational, sig, N: int) -> VRational:
    """Permute the v-indices of a VRational by ``sig`` (sig[i] = new slot of i)."""
    num = {}
    for ve, lp in vr.num._terms.items():
        ne = [0] * N
        for i, e in enumerate(ve):
            ne[sig[i]] = e
        num[tuple(ne)] = lp
    den = {}
    for (i, j, m), mu in vr.den.items():
        den[(sig[i], sig[j], m)] = den.get((sig[i], sig[j], m), 0) + mu
    sq = {}
    for (i, m), mu in vr.sq.items():
        sq[(sig[i], m)] = sq.get((sig[i], m), 0) + mu
    return VRational(VLaurent(num, n=N), den, n=N, sq=sq)


def _levi_blocks(m):
    """Index runs of equal value in anti-dominant ``m`` (the Levi block structure)."""
    m = list(m)
    blocks = []
    start = 0
    for i in range(1, len(m) + 1):
        if i == len(m) or m[i] != m[start]:
            blocks.append(list(range(start, i)))
            start = i
    return blocks


def _dominance_key(p):
    """Sort key picking the most-dominant magnetic charge (descending sort)."""
    return tuple(sorted(p, reverse=True))


def _schur(lam, idx, N):
    """Schur polynomial s_lam in the variables ``idx`` (a list of v-indices),
    as a VLaurent over all N variables.  Recursive GL_n -> GL_{n-1} interlacing."""
    lam = list(lam)
    if len(idx) == 1:
        ve = [0] * N
        ve[idx[0]] = lam[0] if lam else 0
        return VLaurent({tuple(ve): LaurentPoly({0: 1})}, n=N)
    out = VLaurent.zero(N)
    last = idx[-1]
    # mu interlaces lam:  lam_0 >= mu_0 >= lam_1 >= mu_1 >= ... (mu length len-1)
    def gen(i, lo):
        if i == len(lam) - 1:
            yield []
            return
        hi = lam[i]
        floor = lam[i + 1]
        for mval in range(floor, hi + 1):
            for rest in gen(i + 1, mval):
                yield [mval] + rest
    for mu in gen(0, 0):
        sub = _schur(mu, idx[:-1], N)
        power = sum(lam) - sum(mu)
        # multiply sub by x_last^power: shift v-exponent
        terms = {}
        for ve, lp in sub._terms.items():
            nve = list(ve); nve[last] += power
            terms[tuple(nve)] = lp
        out = out + VLaurent(terms, n=N)
    return out


def _vl_mul(a: VLaurent, b: VLaurent, N):
    out = {}
    for ve1, lp1 in a._terms.items():
        for ve2, lp2 in b._terms.items():
            nve = tuple(ve1[i] + ve2[i] for i in range(N))
            prod = lp1 * lp2
            cur = out.get(nve)
            out[nve] = prod if cur is None else cur + prod
    return VLaurent({k: v for k, v in out.items() if not v.is_zero()}, n=N)


def _levi_decompose(ratio: VLaurent, blocks, N):
    """Decompose a W_m-symmetric VLaurent into Levi Schur characters.
    Returns ``[(e_tuple, coeff_LaurentPoly)]`` (e = dominant weight per block)."""
    work = {ve: dict(lp._coeffs) for ve, lp in ratio._terms.items()}
    out = []
    guard = 0
    while any(any(v for v in lp.values()) for lp in work.values()):
        guard += 1
        if guard > 2000:
            raise RuntimeError("_levi_decompose: no termination")
        def key(ve):
            return tuple(tuple(sorted((ve[i] for i in blk), reverse=True)) for blk in blocks)
        top = max((ve for ve, lp in work.items() if any(lp.values())), key=key)
        # e = dominant weight per block
        e = [0] * N
        for blk in blocks:
            vals = sorted((top[i] for i in blk), reverse=True)
            for i, v in zip(blk, vals):
                e[i] = v
        e = tuple(e)
        coeff = dict(work[top])
        out.append((e, LaurentPoly(coeff)))
        # subtract coeff * prod_blocks s_{e|block}
        char = None
        for blk in blocks:
            lam = sorted((e[i] for i in blk), reverse=True)
            sb = _schur(lam, blk, N)
            char = sb if char is None else _vl_mul(char, sb, N)
        for ve, lp in char._terms.items():
            for eq, cc in lp._coeffs.items():
                for ce, cv in coeff.items():
                    cur = work.get(ve, {})
                    cur[eq + ce] = cur.get(eq + ce, 0) - cc * cv
                    work[ve] = cur
        work = {ve: {e2: v for e2, v in lp.items() if v} for ve, lp in work.items()}
    return out


def _center_palindromic(lp: LaurentPoly):
    """Return (C_centered, shift) where C = q^{-shift}*lp is palindromic, or
    (None, None) if lp's support is not symmetric after integer centering."""
    es = list(lp._coeffs)
    if not es:
        return LaurentPoly.zero(), 0
    lo, hi = min(es), max(es)
    if (lo + hi) % 2 != 0:
        return None, None
    shift = (lo + hi) // 2
    cen = LaurentPoly({e - shift: v for e, v in lp._coeffs.items()})
    if all(cen._coeffs.get(e, 0) == cen._coeffs.get(-e, 0) for e in cen._coeffs):
        return cen, shift
    return None, None


def _lp_divide(a: LaurentPoly, b: LaurentPoly) -> LaurentPoly | None:
    """Exact division a/b when both are single-... general Laurent division by
    matching: returns C with a == C*b, else None.  Implemented via the leading
    terms + verification."""
    bc = b._coeffs
    if not bc:
        return None
    ac = a._coeffs
    if not ac:
        return LaurentPoly.zero()
    # shift so that the lowest exponent of b is 0, deduce C as a*b^{-1} only if
    # b is a monomial; otherwise attempt polynomial division.
    if len(bc) == 1:
        (eb, vb), = bc.items()
        if any(v % vb != 0 for v in ac.values()):
            return None
        return LaurentPoly({e - eb: v // vb for e, v in ac.items()})
    # general: try C = a // b via long division on exponents
    from fractions import Fraction
    rem = dict(ac)
    quot: dict = {}
    blo = min(bc)
    bhi = max(bc)
    guard = 0
    while rem:
        guard += 1
        if guard > 10000:
            return None
        rlo = min(rem)
        ce = rlo - blo
        # leading coeff division
        if rem[rlo] % bc[blo] != 0:
            return None
        cc = rem[rlo] // bc[blo]
        quot[ce] = quot.get(ce, 0) + cc
        for be, bv in bc.items():
            rem[ce + be] = rem.get(ce + be, 0) - cc * bv
        rem = {e: v for e, v in rem.items() if v != 0}
    return LaurentPoly(quot)


def _struct_const(P, Lc, m_dom, e, cAmono, N):
    """Structure constant C(q): P's coeff / Lc's coeff at the top (u,v) monomial."""
    top_v = tuple(e[i] + cAmono[i] for i in range(N))
    cP = P._terms[m_dom].simplify().num._terms.get(top_v)
    cL = Lc._terms[m_dom].simplify().num._terms.get(top_v)
    if cP is None or cL is None:
        raise RuntimeError("_struct_const: top monomial missing")
    C = _lp_divide(cP, cL)
    if C is None:
        raise RuntimeError("_struct_const: not a q-Laurent scalar")
    return C


def _partitions_into(k: int, parts: int):
    """All partitions of `k` into ≤ `parts` parts, as descending tuples
    padded to length `parts`."""
    def gen(rem, mx, acc):
        if len(acc) == parts:
            if rem == 0:
                yield tuple(acc)
            return
        for x in range(min(rem, mx), -1, -1):
            yield from gen(rem - x, x, acc + [x])
    yield from gen(k, k, [])


def _schur_monomials(lam):
    """`s_λ` as `{x-exponent tuple: int}` (all permutations of each `m_μ`,
    Kostka-weighted).  Negative entries via the det-shift trick."""
    lam = tuple(int(x) for x in lam)
    d = len(lam)
    shift = -min(lam + (0,))
    lam_pos = tuple(x + shift for x in lam)
    from itertools import permutations
    out: dict = {}
    for mu in _partitions_into(sum(lam_pos), d):
        k_num = _kostka(lam_pos, mu)
        if not k_num:
            continue
        for perm in set(permutations(mu)):
            xe = tuple(x - shift for x in perm)
            out[xe] = out.get(xe, 0) + k_num
    return out


_schur_trace_v0 = trace_v0   # the name the substrate used for its import alias



# ===========================================================================
# Reference closed forms (promoted from the prototype; the swappable internals).
# ===========================================================================
def _psi_dom(m, N):
    """Closed-form dressing ψ_m for DOMINANT (descending) m, E+F sectors."""
    den = {}
    for i in range(N):
        for j in range(i + 1, N):
            for l in range(m[i] - m[j]):
                den[(i, j, -2 * l)] = den.get((i, j, -2 * l), 0) + 1
    ve = tuple(sum(m[i] - m[j] for j in range(i + 1, N)) for i in range(N))
    S = sum(j * m[j] for j in range(N))
    return VRational(VLaurent({ve: LaurentPoly({-S: (-1) ** S})}, n=N), den, n=N)


_PSI: dict = {}


def _psi(a, N):
    """ψ_a at an arbitrary u-power a (Weyl transport of ψ_{sorted(a)})."""
    a = tuple(a)
    key = (N, a)
    if key in _PSI:
        return _PSI[key]
    mdom = tuple(sorted(a, reverse=True))
    pc = _psi_dom(mdom, N)
    for sig in itertools.permutations(range(N)):
        if tuple(mdom[sig.index(k)] for k in range(N)) == a:
            r = _perm_vr(pc, sig, N).simplify()
            _PSI[key] = r
            return r
    raise RuntimeError(f"no Weyl chamber transport to u={a}")


def _psi_inv(a, N):
    """1/ψ_a (denominator-free: poles → numerator, monomial → v-shift)."""
    p = _psi(a, N)
    (mve, mlp), = p.num._terms.items()
    (meq, mev), = mlp._coeffs.items()
    poles = VLaurent.one(N)
    for (i, j, M), mult in p.den.items():
        f = VLaurent({tuple(1 if t == i else 0 for t in range(N)): LaurentPoly({0: 1}),
                      tuple(1 if t == j else 0 for t in range(N)): LaurentPoly({M: -1})}, n=N)
        for _ in range(mult):
            poles = poles * f
    acc: dict = {}
    for ve, lp in poles._terms.items():
        nve = tuple(ve[t] - mve[t] for t in range(N))
        acc[nve] = acc.get(nve, LaurentPoly.zero()) + LaurentPoly(
            {e - meq: c // mev for e, c in lp._coeffs.items()})
    return VRational(VLaurent(acc, n=N), {}, n=N)


def _shift(vr, s, N):
    """Monomial torus shift v_i → q^{s_i} v_i on a VRational (s an int N-tuple).

    The **full** normal-ordering is `s = 2m` (used by the cocycles); the **half**
    shift is `s = m` (used to pass between the bare residual `c_m` and `f_m`)."""
    # NB callers pass already-simplified VRationals (residuals from `_f`, cached
    # `ψ`/`R`); the shift commutes with simplify, so we skip a redundant pass here.
    qextra = 0
    den: dict = {}
    for (i, j, M), mult in vr.den.items():
        nk = (i, j, M + s[j] - s[i])
        den[nk] = den.get(nk, 0) + mult
        qextra += -s[i] * mult
    num: dict = {}
    for ve, lp in vr.num._terms.items():
        sh = sum(s[i] * ve[i] for i in range(N)) + qextra
        num[ve] = LaurentPoly({e + sh: c for e, c in lp._coeffs.items()})
    return VRational(VLaurent(num, n=N), den, n=N)


def _qshift(vr, m, N):
    """Full normal-ordering shift v→q^{2m}v."""
    return _shift(vr, tuple(2 * x for x in m), N)


_R: dict = {}


def _Rcoc(a, b, N):
    """The multiply cocycle: U_a U_b = R_{a,b} U_{a+b}, R = ψ_a·T_{2a}(ψ_b)/ψ_{a+b}
    (`T_p : v^λ ↦ 𝖖^{⟨p,λ⟩}v^λ`; written `S_a` until 2026-08-25 — see
    `wrq_torus.cocycle_R` for why the `S_` spelling was retired)."""
    key = (N, tuple(a), tuple(b))
    if key in _R:
        return _R[key]
    tot = tuple(a[i] + b[i] for i in range(N))
    R = (_psi(a, N) * _qshift(_psi(b, N), a, N) * _psi_inv(tot, N)).simplify()
    _R[key] = R
    return R


_RT: dict = {}


def _Rtilde(a, b, N):
    """The multiply cocycle in the f-representation: R̃_{a,b} = T_{−(a+b)}(R_{a,b})."""
    key = (N, tuple(a), tuple(b))
    rt = _RT.get(key)
    if rt is None:
        tot = tuple(a[i] + b[i] for i in range(N))
        rt = _shift(_Rcoc(a, b, N), tuple(-x for x in tot), N).simplify()
        _RT[key] = rt
    return rt


def _atom(a, N):
    return DOp({tuple(a): _psi(a, N)}, n=N, n_gauge=N)


def _qbar(vr, N):
    """`q ↦ q⁻¹` on a VRational with `v` fixed: negate every q-power in the
    numerator and flip each pole `(v_i − q^M v_j) → (v_i − q^{−M} v_j)`."""
    vr = vr.simplify()
    if getattr(vr, "_sq", None):
        raise NotImplementedError("_qbar: sqrt denominators not handled")
    num = {ve: LaurentPoly({-e: c for e, c in lp._coeffs.items()})
           for ve, lp in vr.num._terms.items()}
    den: dict = {}
    for (i, j, M), mult in vr.den.items():
        den[(i, j, -M)] = den.get((i, j, -M), 0) + mult
    return VRational(VLaurent(num, n=N), den, n=N)


# ===========================================================================
# G-twists — the √measure ρ-cocycle, and the hook for OTHER G's later.
# ===========================================================================
class GTwist:
    """A magnetic-antipode twist `m → −m` with a closed-form block monomial.

    U(N)'s `ρ` (the √measure conjugation) is `RHO`; its inverse is `RHO_INV`.  A
    *different* G (another gauge group, a parity τ extension, an RG-side twist) is
    realised as a sibling `URQTorus` subclass that binds its own `GTwist` as
    `_RHO` — supplying a different `block(m, N) -> (sign, qpow, wexp)` / `q_sign`.
    The G-cocycle machinery (`_G`, `apply`) is itself twist-agnostic."""

    def __init__(self, block, q_sign=+1, use_image_block=False, name="rho"):
        self._block = block            # m -> (sign, qpow, wexp)
        self.q_sign = q_sign           # ρ⁻¹ negates the q-power
        self.use_image_block = use_image_block   # ρ⁻¹ reads block data at k=−m
        self.name = name
        self._gt: dict = {}            # G̃_m = T_m(G_m) cache, keyed (N, m)

    def _G(self, m, N):
        """The closed-form twist factor G_m at magnetic m (rational in v, c-coords)."""
        k = tuple(-x for x in m)
        sign, qpow, wexp = self._block(k if self.use_image_block else m, N)
        qpow = self.q_sign * qpow
        mono = VRational.from_vlaurent(
            VLaurent({tuple(-x for x in wexp): LaurentPoly({qpow: sign})}, n=N))
        return (_vinv_vrational(_psi(m, N), N) * mono * _psi_inv(k, N)).simplify()

    def _Gtilde(self, m, N):
        """G̃_m = T_m(G_m) — the twist factor in the f-coordinate representation."""
        key = (N, tuple(m))
        gt = self._gt.get(key)
        if gt is None:
            gt = _shift(self._G(m, N), tuple(m), N).simplify()
            self._gt[key] = gt
        return gt

    def apply(self, f_dict, N):
        """ρ-style action in the f-representation: {m: f} ↦ {−m: v̄(f_m)·G̃_m}."""
        out: dict = {}
        for m, f in f_dict.items():
            k = tuple(-x for x in m)
            cur = (_vinv_vrational(f, N) * self._Gtilde(m, N)).simplify()
            out[k] = (out[k] + cur).simplify() if k in out else cur
        return {k: v for k, v in out.items() if not v.is_zero()}


RHO = GTwist(_rho_block_data, q_sign=+1, use_image_block=False, name="rho")
RHO_INV = GTwist(_rho_block_data, q_sign=-1, use_image_block=True, name="rho_inv")


# ===========================================================================
# The class.
# ===========================================================================
class URQTorus:
    """Enriched rational quantum torus element for pure U(N): the residual vector
    `(f_m)` over the cocharacter lattice, `x = Σ_m f_m(q^m v) U_m`.

    Construct from a chart operator (`from_chart`) or from an explicit `f_m` dict
    (`from_f` / `__init__`).  The residual vector `f_m` is the **internal storage**,
    so the `f_m` accessors (`residual`, `residuals`, `leading`) are shift-free; the
    algebra (`*`, `rho`, `rho_inverse`, `trace`, `inner`) and recognition (`mtest`)
    work in f-coordinates on the closed forms above (`R̃`, `G̃`).  Elements are
    immutable value objects keyed by magnetic charge."""

    __slots__ = ("_f", "_N", "_rho_cache", "_rho_content_cache")

    # U(N)'s G-cocycle (the √measure ρ) and its inverse, bound at the class.
    # "Other G = other class": a different gauge group or a parity (τ) extension
    # is a sibling subclass that overrides `_RHO`/`_RHO_INV` — not a runtime arg.
    _RHO = RHO
    _RHO_INV = RHO_INV

    def __init__(self, f_residuals: dict, N: int):
        # store the residual vector f_m DIRECTLY (the public content): shift-free
        # access, and the algebra (multiply/ρ) works in f-coords.  Nonzero only.
        self._N = int(N)
        f = {}
        for m, vr in f_residuals.items():
            vr = vr.simplify()
            if not vr.is_zero():
                f[tuple(m)] = vr
        self._f = f
        self._rho_cache = None      # lazy ρ(self) memo (immutable value object)
        self._rho_content_cache = {}   # lazy per-n ρ-content memo for `inner` (§6b)

    # ----- construction / interop ------------------------------------------
    @classmethod
    def zero(cls, N: int) -> "URQTorus":
        return cls({}, N)

    @classmethod
    def from_chart(cls, D: DOp, N: int) -> "URQTorus":
        """Ingest a chart operator `D = Σ_a u^a d_a(v)`: bare residual `c_a = d_a/ψ_a`,
        then `f_a = T_{−a}(c_a)` (the promoted `to_U` + the half-shift to f-coords)."""
        D = _simp(D)
        f = {}
        for a, vr in D._terms.items():
            vr = vr.simplify()
            if not vr.is_zero():
                a = tuple(a)
                c = (vr * _psi_inv(a, N)).simplify()
                f[a] = _shift(c, tuple(-x for x in a), N).simplify()
        return cls(f, N)

    @classmethod
    def from_f(cls, f_dict: dict, N: int) -> "URQTorus":
        """Build from an explicit `f_m(v)` dict (half-shift convention) — stored as
        is (no shift); the element is `x = Σ_m f_m(q^m v) U_m`."""
        return cls(f_dict, N)

    def to_chart(self) -> DOp:
        """Re-expand to a chart operator: `c_a = T_a(f_a)`, then `Σ_a c_a·U_a`."""
        out = DOp.zero(self._N)
        for a, fa in self._f.items():
            c = _shift(fa, tuple(a), self._N)
            out = out + _simp(DOp.from_scalar(c, n=self._N) * _atom(a, self._N))
        return _simp(out)

    # ----- the f_m content (primary) -- shift-free, the hot read path ------
    def residual(self, m) -> VRational:
        """`f_m(v)` — the residual at magnetic `m` (`0` if absent).  Direct."""
        f = self._f.get(tuple(m))
        return f if f is not None else VRational.from_scalar(LaurentPoly.zero(), n=self._N)

    def residuals(self) -> dict:
        """The whole vector `{m: f_m(v)}` (half-shift convention).  Direct."""
        return dict(self._f)

    def support(self) -> list:
        return sorted(self._f)

    def leading(self):
        """The leading piece of a canonical: `(m_anti, f_{m_dom}(v))` at the most
        dominant magnetic — the (Weyl-rotated) Levi character the M-test reads."""
        if not self._f:
            return None
        m_dom = max(self._f, key=_dominance_key)
        return (tuple(sorted(m_dom)), self._f[m_dom])

    def bare_residuals(self) -> dict:
        """The algebra-internal bare residuals `{m: c_m}` (`c_m = T_m(f_m) = to_U`
        output).  Exposed for legacy U-basis interop; prefer `residuals()`."""
        return {m: _shift(f, tuple(m), self._N).simplify() for m, f in self._f.items()}

    # ----- algebra (f-coordinate representation) ---------------------------
    def __add__(self, other: "URQTorus") -> "URQTorus":
        assert self._N == other._N
        out = dict(self._f)
        for m, f in other._f.items():
            out[m] = (out[m] + f).simplify() if m in out else f
        return type(self)(out, self._N)

    def __mul__(self, other: "URQTorus") -> "URQTorus":
        # f^C_{m+m'} += T_{−m'}(f_m) · T_m(g_{m'}) · R̃_{m,m'}
        N = self._N
        out: dict = {}
        for m, fm in self._f.items():
            for mp, gmp in other._f.items():
                tot = tuple(m[i] + mp[i] for i in range(N))
                term = (_shift(fm, tuple(-x for x in mp), N)
                        * _shift(gmp, m, N) * _Rtilde(m, mp, N)).simplify()
                # simplify each term and each partial sum: keeps the running
                # denominators minimal (incremental pole cancellation), which is
                # markedly faster than one big simplify at the end (≈17% at N=4).
                out[tot] = term if tot not in out else (out[tot] + term).simplify()
        return type(self)(out, N)

    def held_multiply(self, other: "URQTorus") -> "HeldURQTorus":
        """Like `*`, but **held**: accumulate each magnetic component `f^C_{m+m'}`
        as an unsimplified **formal sum** of the per-pair terms, deferring the
        combine+simplify to whatever a consumer actually reads.

        Producing the terms (the double loop) is cheap; the per-component
        combine+simplify is essentially all the cost.  Terminal consumers read
        few components (`trace` → `f_0`; `recognize_leading` → the leading
        magnetic), so a held product lets each pay for one component instead of
        the ~N produced (≈40× at N=4 for `trace`).  `materialize()` max-simplifies
        the held element back to a plain `URQTorus` (`held_multiply(...).materialize()
        == self * other`)."""
        N = self._N
        buckets: dict = {}
        for m, fm in self._f.items():
            for mp, gmp in other._f.items():
                tot = tuple(m[i] + mp[i] for i in range(N))
                term = (_shift(fm, tuple(-x for x in mp), N)
                        * _shift(gmp, m, N) * _Rtilde(m, mp, N))
                buckets.setdefault(tot, []).append(term)
        return HeldURQTorus(buckets, N, cls=type(self))

    def rho(self) -> "URQTorus":
        """ρ(x) — the √measure conjugation (magnetic antipode + v-inversion + the
        U(N) G̃-cocycle `self._RHO`), preserving the class.  A different G is a
        different class (override `_RHO`/`_RHO_INV`), not a runtime argument.

        Memoised: `inner`/`orthonormality` call `rho()` on the left argument once
        per pairing, so a Gram-matrix sweep over a basis recomputes `ρ(a)` for
        every `b` without this cache."""
        r = self._rho_cache
        if r is None:
            r = type(self)(self._RHO.apply(self._f, self._N), self._N)
            self._rho_cache = r
        return r

    def rho_inverse(self) -> "URQTorus":
        return type(self)(self._RHO_INV.apply(self._f, self._N), self._N)

    def bar(self) -> "URQTorus":
        """The bar involution β: `q ↦ q⁻¹` on each residual `f_m` (with `v` and the
        magnetic `m` fixed).  In the URQTorus bar is just this q-inversion of the
        coefficients -- transparent.  It is an **anti-automorphism**
        (`β(xy)=β(y)β(x)`, the algebra is a q-twisted quantum torus) and an
        involution, and it **fixes the canonical basis** (`β(L_a)=L_a`): a
        canonical's residuals `f_m` are all q-palindromic, so bar-invariance ⟺
        every `f_m` is palindromic.  The bubbling (lower-magnetic) residuals are
        thus simultaneously bar-invariant and the `O(q)` correction to the leading
        Levi character -- the Kazhdan-Lusztig characterisation of `{L_a}`."""
        return type(self)({m: _qbar(f, self._N) for m, f in self._f.items()}, self._N)

    def trace(self, K: int = 8, adaptive: bool = True) -> LaurentPoly:
        """Pure-U(N) Schur trace — the **v-only** Schur-measure residue of the
        magnetic-0 residual `f_0` (`= c_0` since the `m=0` half-shift is trivial,
        `ψ_0=1`).  No chart/DOp: the trace acts on the residual `f_0(v)` directly."""
        return trace_v0(self._f.get((0,) * self._N), self._N, K, adaptive=adaptive)

    def _pairing_f0(self, other: "URQTorus"):
        """The magnetic-0 residual of `ρ(self)·other`, built **directly** from the
        §6b inner-product formula — a sum over magnetic charge of per-`m` weighted
        residual pairings, with **no `ρ` element constructed** and only the shared
        magnetic support iterated.

        `Tr(ρ(a)·b)` reads only the `m=0` output of `ρ(a)·b`, which (§3, §6b) is the
        diagonal contraction `Σ_n ρ(a)_{−n}·b_n` over the cocharacter lattice.  `ρ`
        maps charge `n → −n` with the √measure cocycle, so `ρ(a)_{−n} = v̄(f^a_n)·G̃_n`
        — computed **inline** per shared `n` (the `G̃_n` and `R̃_{−n,n}` are cached
        per-`n` weights), never as a full `ρ(a)` object:

            f_0  =  Σ_{n ∈ supp(a)∩supp(b)}  T_{−n}( v̄(f^a_n)·G̃_n ) · T_{−n}(f^b_n) · R̃_{−n,n}.

        This is the **Schur-quantization auxiliary-space inner product** `⊕_n (H_n,
        ⟨·,·⟩_n)` realised directly (§6b); validated `==` the old `ρ(a)*b` route on
        all canonical/product pairs (N=3,4).  Returns the simplified `f_0` (or
        `None` if the shared support is empty).  Requires admissible (cone-interior,
        in-domain `H_n`) poles — like every in-span tool, it is meaningless on
        off-domain residuals."""
        N = self._N
        bf = other._f
        f0 = None
        for n, fan in self._f.items():
            fbn = bf.get(n)                            # block-diagonal: same charge n
            if fbn is None:
                continue
            nn = tuple(-x for x in n)
            am = self._rho_content(n)                  # ρ(a)_{−n} content, lazy per-n (no ρ object)
            term = (_shift(am, nn, N) * _shift(fbn, nn, N) * _Rtilde(nn, n, N))
            f0 = term.simplify() if f0 is None else (f0 + term.simplify()).simplify()
        return f0

    def _rho_content(self, n) -> VRational:
        """`ρ(self)_{−n} = v̄(f^a_n)·G̃_n` — the per-`n` ρ-content the §6b inner product
        pairs against `f^b_n`.  Lazily memoised per `n` (the `G̃_n` √measure cocycle is
        itself cached on `_RHO`), so a Gram-matrix sweep `inner(a, ·)` reuses it across
        every `b`, and only the magnetic charges actually paired are ever built —
        strictly lazier than constructing the full `ρ(a)` element."""
        c = self._rho_content_cache.get(n)
        if c is None:
            c = (_vinv_vrational(self._f[n], self._N)
                 * self._RHO._Gtilde(n, self._N)).simplify()
            self._rho_content_cache[n] = c
        return c

    def inner(self, other: "URQTorus", K: int = 8) -> LaurentPoly:
        """`⟨a, b⟩ = Tr(ρ(a)·b)` — the block-diagonal inner product on the residual
        vectors (`= Σ_m ⟨·,·⟩_m`, §6b): the **Schur-quantization auxiliary-space
        pairing** `⊕_m (H_m, ⟨·,·⟩_m)`.  Orthonormal on the canonical basis to
        `O(q)`, and the same pairing other goals (orbi-modules, real
        Schur) reuse.

        Computed **directly from the §6b formula — not via a multiply and not via a
        full `ρ(a)`**: only the magnetic-0 output of `ρ(a)·b` feeds the trace, and
        that is the per-`m` weighted residual pairing assembled inline over the
        shared magnetic support (`_pairing_f0`), using the cached per-`m` weights.
        (On bar-invariant elements — all canonicals — the bilinear and bar-Hermitian
        forms of this pairing coincide, so the conjugation convention is immaterial
        here; it requires admissible/in-domain poles.)

        **Block-diagonal fast-zero (§6b).**  The pairing is block-diagonal in
        magnetic charge and `ρ` flips `m→−m`, so the contraction is nonempty *iff*
        `supp(a) ∩ supp(b) ≠ ∅`.  Disjoint magnetic supports ⇒ `⟨a,b⟩ = 0` to all
        orders — returned via an O(support) set test, with no products and no `ρ`."""
        if self._f.keys().isdisjoint(other._f.keys()):
            return LaurentPoly.zero()
        f0 = self._pairing_f0(other)
        return trace_v0(None if (f0 is None or f0.is_zero()) else f0,
                               self._N, K, adaptive=True)

    def orthonormality(self, other: "URQTorus", K: int = 2):
        """Orthonormality pairing `Tr(ρ(self)·other) = δ + O(q)`, via the direct
        inner product (`_pairing_f0`) with small `K`.

        Returns the trace as a `LaurentPoly` (`= integer + O(q)`, whose `q⁰` is the
        Kronecker δ) when it is a genuine **power** series in `q` (no negative
        powers — the M-test soundness condition, §6b); returns ``False`` when
        negative-`q` content appears (the pairing is then not `integer + O(q)`, so
        the elements are off-span / the leading orbit is wrong).

        **Not the right tool for certifying a canonical — use `well_formed`.**  This
        pairing reads only `q⁰`, so it is *redundant* on an in-span element (any
        element obtained as a polynomial in canonicals with leading orbit `L_{m,e}`
        and `O(q)` bubbling already IS `L_{m,e}` by Kazhdan-Lusztig uniqueness, §6c —
        the `δ` follows for free) and *useless* off-span (it auto-passes correct-
        leading + arbitrary `O(q)` bubbling, and off-span garbage — the
        constructive-build rule).  The trace-free `well_formed()` (bar-invariance + `O(q)`
        bubbling *shape*) is the better and faster post-build certificate.

        `K` is small: `δ + O(q)` needs only `q⁰` plus detection of `q^{<0}` (the
        adaptive trace still reaches the full negative-`q` extent internally)."""
        t = self.inner(other, K)
        if any(c != 0 for e, c in t._coeffs.items() if e < 0):
            return False
        return t

    def well_formed(self):
        """Trace-free well-formedness certificate for a single-canonical build output.

        A canonical `L_a` is, by Kazhdan-Lusztig (§6c), the **unique bar-invariant**
        element of its leading Weyl orbit whose bubbling is `O(q)`.  So for an element
        obtained as a **polynomial in canonicals** (in-span — the constructive build)
        with a single intended leading orbit, checking the *shape* certifies it IS
        that canonical, with **no Schur trace**:

          (W1) bar-invariant — every residual `f_m` is q-palindromic (`β(self)==self`);
          (W2) the bubbling is `O(q)` — the q-extreme slice (§6c) is a single leading
               Weyl orbit of multiplicity 1.

        Returns the canonical's label `(m, e)` when well-formed, else ``False``.  This
        is faster than — and the proper replacement for — the orthonormality self-norm
        on a build output (which needs the Schur trace, and in-span is implied by KL
        anyway).

        **Scope.**  It presupposes in-span, single-target.  It does *not* detect a
        superposition of canonicals with distinct leadings, nor an off-span element —
        **no post-hoc test does** (the constructive-build rule).  The only guard against bad
        bubbling / leaving the span is **constructive building** itself."""
        if self.bar() != self:                     # (W1) bar-invariance — palindrome
            return False
        r0 = self.recognize_q_extreme(False)        # (W2) q-extreme slice (no trace)
        if len(r0) != 1:
            return False
        (label, c), = r0.items()
        return label if c == 1 else False

    # ----- recognition tool 1: leading magnetic (for the multiply engine) ---
    def recognize_leading(self) -> dict:
        """**Tool 1** (multiply engine): recognise `self` as a linear combination
        `Σ_e c_e(q)·L_{m,e}` by reading its **leading Weyl orbit** — the extreme
        (most anti-dominant) magnetic, whose residual is bubbling-free and so is a
        clean Levi character — and Levi-decomposing it.

        Returns `{(m, e): c_e(q)}` with `c_e` a LaurentPoly in `q` (the coefficient
        of `L_{m,e}`'s leading piece), labels in the canonical **(anti-dominant m,
        Levi-anti-dominant e)** convention.  Residual-native: the U-basis residual
        *is* the character, so no division and no measure are involved."""
        N = self._N
        if not self._f:
            return {}
        # Dominance-key TIES (several atoms of the leading orbit in support)
        # must resolve to the in-frame dominant representative — the Levi
        # e-read below is frame-dependent.  Previously implicit via dict
        # insertion order (constructors insert the dominant rep first);
        # explicit.  When no dominant rep is present the
        # historical first-max pick is kept (Weyl-closed supports always
        # carry it in practice).
        best = max(_dominance_key(m) for m in self._f)
        cands = [m for m in self._f if _dominance_key(m) == best]
        dom = [m for m in cands if tuple(m) == tuple(sorted(m, reverse=True))]
        m_dom = dom[0] if dom else cands[0]              # leading orbit
        c = _shift(self._f[m_dom], tuple(m_dom), N).simplify()    # the residual there
        if c.den or c._sq:
            raise RuntimeError("recognize_leading: leading residual not polynomial")
        blocks = _levi_blocks(m_dom)
        # `_levi_decompose` reads e as the dominant weight in the m_dom frame, so
        # (m_dom, e) is the UPPER Kapustin label.  Return the LOWER Kapustin label
        # (anti-dom m, Levi-anti-dom e) via the joint w_0 involution -- reverse BOTH
        # m and e together.  Lower Kapustin is THE canonical convention shared with
        # the engine's labels and with minuscule/wilson; reversing only one of m,e
        # would give a mixed (dom m, anti-dom e) state, which is not a label.
        return {(tuple(reversed(m_dom)), tuple(reversed(e))): c_e
                for e, c_e in _levi_decompose(c.num, blocks, N)}

    # ----- recognition tool 2: q-extreme (for the build engine) -------------
    def recognize_q_extreme(self, at_infinity: bool = False) -> dict:
        """**Tool 2** (build engine): the leading term of the power-series expansion
        in `q` of the **residuals `f_m(v)`** around `q=0` (`at_infinity=False`) or
        `q=∞` (`True`).

        On `f_m` the bubbling contributions are subleading in `q` (their poles
        `(v_i−q^{−2l}v_j)` expand as `O(q^{2l})` around `q=0`), so the extreme-`q`
        slice is the bubbling-free leading Weyl orbit, which it decomposes into
        leading pieces of `L`'s.  Returns `{(m, e): int}`, labels in the canonical
        **(anti-dominant m, Levi-anti-dominant e)** convention.  `q=∞` is the bar
        (`q→1/q`) image of `q=0` (and `bar` fixes `{L_a}`)."""
        if at_infinity:
            return self.bar().recognize_q_extreme(at_infinity=False)
        N = self._N
        # power-series q-valuation of each residual f_m around q=0:
        # val = val(num) − val(den), val(den) = Σ_{M<0} M (only those factors'
        # lowest-q part is q^M < 0; poles with M>0 expand as O(q^M) ≥ 0).
        vals: dict = {}
        for m, f in self._f.items():
            f = f.simplify()
            if f.num.is_zero():
                continue
            vnum = min(min(lp._coeffs) for lp in f.num._terms.values())
            vden = sum(M * mult for (i, j, M), mult in f.den.items() if M < 0)
            vals[tuple(m)] = (vnum - vden, f, vnum)
        if not vals:
            return {}
        qext = min(v[0] for v in vals.values())
        # the leading (q^qext) coefficient of each f_m reaching qext, as a VRational
        sl_f: dict = {}
        for m, (val, f, vnum) in vals.items():
            if val != qext:
                continue
            sign = 1
            shift = [0] * N
            den_lead: dict = {}
            for (i, j, M), mult in f.den.items():
                for _ in range(mult):
                    if M > 0:
                        shift[i] -= 1                     # leading factor v_i → /v_i
                    elif M == 0:
                        den_lead[(i, j, 0)] = den_lead.get((i, j, 0), 0) + 1
                    else:
                        shift[j] -= 1; sign = -sign       # leading −q^M v_j → /(−v_j)
            acc: dict = {}
            for ve, lp in f.num._terms.items():
                if vnum in lp._coeffs:                    # the q^vnum (leading) part
                    nve = tuple(ve[t] + shift[t] for t in range(N))
                    acc[nve] = LaurentPoly({0: sign * lp._coeffs[vnum]})
            sl_f[m] = VRational(VLaurent(acc, n=N), den_lead, n=N)
        # the extreme slice is a residual vector at q^0 (bubbling-free); recognise
        # its leading pieces as L's and read off the integer multiplicities
        slc = type(self).from_f(sl_f, N)
        out: dict = {}
        for k, c in slc.recognize_leading().items():
            vv = list(c._coeffs.values())
            out[k] = int(vv[0]) if len(vv) == 1 else int(sum(vv))
        return out

    # ----- dunders ----------------------------------------------------------
    def __eq__(self, other) -> bool:
        # equality is on the residual vectors directly — no chart involved
        if not (isinstance(other, URQTorus) and self._N == other._N):
            return False
        z = VRational.from_scalar(LaurentPoly.zero(), n=self._N)
        for m in set(self._f) | set(other._f):
            if not (self._f.get(m, z) + other._f.get(m, z) * (-1)).simplify().is_zero():
                return False
        return True

    def __repr__(self) -> str:
        return f"URQTorus(N={self._N}, support={self.support()})"


# ===========================================================================
# Held (lazy) variant — formal-sum residuals, combined per component on demand.
# ===========================================================================
class HeldURQTorus:
    """A **held** URQTorus element: each magnetic component `f_m` is kept as an
    unsimplified **formal sum** of rational terms and is combined+simplified only
    when (and as) a consumer reads it.

    Produced by `URQTorus.held_multiply`.  The point is the asymmetry measured on
    the multiply: producing the per-pair terms is cheap, but the per-component
    combine+simplify is essentially all the cost, and most *terminal* consumers
    touch only one of the ~N magnetic components — `trace` reads only `f_0`,
    `recognize_leading` only the leading magnetic.  A held element lets each such
    consumer materialise just the component(s) it needs (≈40× cheaper than a full
    eager product at N=4 for `trace`).

    A held canonical element does **not** need to be max-simplified.  In the
    build engine its natural finish is *recognise → rebuild*: read its label with
    `recognize_leading` / `recognize_q_extreme` (which combine only the few
    components recognition touches) and reconstruct it as a clean standard
    element from that label — never paying to simplify the remaining components.
    `materialize()` (alias `simplify()`) is still available for when you genuinely
    need every component combined into a plain `URQTorus` (equality, or feeding
    `rho` / `inner`); by construction `a.held_multiply(b).materialize() == a * b`.

    Components materialise *incrementally* (simplify each term, then fold with a
    simplify per add), exactly as eager `__mul__` does, so even materialising
    **all** components costs the same as the eager product — there is no penalty
    for the read-everything path, only a saving for the read-few path."""

    __slots__ = ("_buckets", "_N", "_cache", "_cls")

    def __init__(self, buckets: dict, N: int, cls=URQTorus):
        self._N = int(N)
        self._buckets = buckets        # m -> list[VRational] (unsimplified terms)
        self._cache: dict = {}         # m -> simplified VRational (lazy, memoised)
        self._cls = cls                # eager class to materialise back into

    # ----- candidate support / lazy materialisation ------------------------
    def support_candidates(self) -> list:
        """Magnetic charges carrying terms.  A candidate may still combine to
        zero — that is only known after `materialize_component`."""
        return sorted(self._buckets)

    def materialize_component(self, m) -> VRational:
        """Combine + simplify the formal sum at magnetic `m` (incremental: keeps
        the running denominator minimal, matching eager `__mul__` per-component
        cost).  Memoised; returns the zero VRational if `m` is absent/cancels."""
        m = tuple(m)
        c = self._cache.get(m)
        if c is not None:
            return c
        terms = self._buckets.get(m)
        if not terms:
            c = VRational.from_scalar(LaurentPoly.zero(), n=self._N)
        else:
            c = terms[0].simplify()
            for t in terms[1:]:
                c = (c + t.simplify()).simplify()
        self._cache[m] = c
        return c

    def materialize(self) -> "URQTorus":
        """Max-simplify every component → a plain (canonical) `URQTorus`."""
        f = {}
        for m in self._buckets:
            c = self.materialize_component(m)
            if not c.is_zero():
                f[m] = c
        return self._cls.from_f(f, self._N)

    simplify = materialize         # "held → canonical" reads more naturally

    # ----- terminal consumers: pull only the components they need ----------
    def trace(self, K: int = 8, adaptive: bool = True) -> LaurentPoly:
        """Schur trace — materialise only `f_0` (the v-only magnetic-0 residue)."""
        f0 = self.materialize_component((0,) * self._N)
        return trace_v0(None if f0.is_zero() else f0,
                               self._N, K, adaptive=adaptive)

    def recognize_leading(self) -> dict:
        """**Tool 1** on a held element — materialise only the leading magnetic.

        Scans candidate magnetics in descending dominance and takes the first
        whose materialised residual is nonzero (the leading Weyl orbit); other
        components are never combined.  Same lower-Kapustin output as
        `URQTorus.recognize_leading`."""
        N = self._N
        # same tie-break as the eager class: among equal
        # dominance keys, materialise the in-frame dominant representative
        # first — the Levi e-read is frame-dependent.
        for m_dom in sorted(
                self._buckets,
                key=lambda m: (_dominance_key(m),
                               tuple(m) == tuple(sorted(m, reverse=True))),
                reverse=True):
            f = self.materialize_component(m_dom)
            if f.is_zero():
                continue
            c = _shift(f, tuple(m_dom), N).simplify()
            if c.den or c._sq:
                raise RuntimeError("recognize_leading: leading residual not polynomial")
            blocks = _levi_blocks(m_dom)
            return {(tuple(reversed(m_dom)), tuple(reversed(e))): c_e
                    for e, c_e in _levi_decompose(c.num, blocks, N)}
        return {}

    def recognize_q_extreme(self, at_infinity: bool = False) -> dict:
        """**Tool 2** reads every component's q-valuation, so it materialises the
        whole element first (no per-component shortcut available here)."""
        return self.materialize().recognize_q_extreme(at_infinity=at_infinity)

    # ----- the rest delegate through a full materialisation ----------------
    def inner(self, other, K: int = 8) -> LaurentPoly:
        return self.materialize().inner(other, K)

    def rho(self) -> "URQTorus":
        return self.materialize().rho()

    def rho_inverse(self) -> "URQTorus":
        return self.materialize().rho_inverse()

    def bar(self) -> "URQTorus":
        return self.materialize().bar()

    def residual(self, m) -> VRational:
        return self.materialize_component(m)

    def __eq__(self, other) -> bool:
        if isinstance(other, HeldURQTorus):
            other = other.materialize()
        return self.materialize() == other

    def __repr__(self) -> str:
        return f"HeldURQTorus(N={self._N}, candidates={self.support_candidates()})"

