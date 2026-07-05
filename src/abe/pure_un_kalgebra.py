"""`PureUNKAlgebra` — the pure U(N) K-theoretic Coulomb branch algebra as a
verifier-passing `KAlgebra`, **unified** into a single self-contained module.

This module folds the former `pure_un_canonical` (bar-centered loops + Schur
trace + ρ) and `pure_un_construct` (`CanonicalBasis` constructor) layers into
one place.  The only external pieces it leans on are:

  * the **stateless chart engine** in `pure_un_chart_engine` — the raw ray/det images
    `_ray_img_E/F`, `_det_img`, the Wilson character `_wilson_sym`, and the
    Schur-measure helpers (`_schur_measure_truncated`, `_q_poch_q2_q2_*`,
    the truncators);
  * two **framework statics** from `AbeKAlgebra` — `_simplify_dop` and
    `_u_order` (generic DOp normalisation / monomial ordering).

The three sections below are the old trilogy, in dependency order:

  1. canonical bar-centered loops `L_E/L_F/L_det/L_W`, the Schur `trace`,
     and the ρ-automorphism on generator labels (`rho_label`);
  2. the `CanonicalBasis` registry — canonical basis built on demand by the
     Principal-QTCone / `w_R·cone` engine (`_build_clean`), plus `multiply`
     (decompose `image(a)·image(b)`) and on-demand registry growth;
  3. `PureUNKAlgebra` — the `KAlgebra` over this canonical basis.

**Labels are 't Hooft-Wilson charges `(m, λ)`** (Kapustin): `m` the anti-dominant
magnetic cocharacter, `λ` the dominant weight (irrep) of the Levi `L_m`
(centralizer of `m`); the v-dressing is the Levi-irrep character.  Keying is by
`_lowest_charge` = the **leading label** (lowest-u block, dominant v-weight of
its lowest-q term).  Under this keying ρ is a SIGN-FREE permutation directly (no
eps-absorption).  The Schur `trace` is exact (Euler/Habiro measure + adaptive
q⁰ cutoff).  `pure_un_canonical.py` /
`pure_un_construct.py` remain thin re-export shims.
"""
from __future__ import annotations
from functools import lru_cache
from math import factorial

from kalgebra import KAlgebra, Element
from abelianized_torus import simplify_dop as _abe_simplify_dop, u_order as _abe_u_order
from abe_kalgebra import AbeKAlgebra
from pure_un_chart_engine import (              # stateless chart engine
    _ray_img_E, _ray_img_F, _det_img, _wilson_sym,
    _schur_measure_truncated, _q_poch_q2_q2_truncated,
    _laurent_truncate, _vlaurent_truncate,
)
from abelianized_torus import DOp, VRational, VLaurent
from laurent_poly import LaurentPoly
from zplus_ring import TrivialZPlusRing, RPowerSeries


# =====================================================================
# SECTION 1 — canonical bar-centered loops, Schur trace, ρ on labels
# (formerly pure_un_canonical.py)
#
# Canonical normalization, with
# <w_k,f> = f[0]+...+f[k-1]:
#
#     L_E(k,f) = q^{ lam_E(k,f) } * E_k(f),   lam_E(k,f) = -k(k-1)/2 + <w_k,f>
#     L_F(k,g) = q^{ lam_F(k,g) } * F_k(g),   lam_F(k,g) =  k(2N-k-1)/2 - <w_k,g>
#     L_det^p  = q^{ -p N(N-1)/2 } * det^p
#     L_W(lam) = W_lam                         (Wilson lines, lam-degree 0)
# =====================================================================

def _ipair(k: int, f) -> int:
    """<omega_k, f> = f[0] + ... + f[k-1]."""
    return sum(int(x) for x in f[:k])

def lam_E(k: int, f, N: int) -> int:
    return -k * (k - 1) // 2 + _ipair(k, f)

def lam_F(k: int, g, N: int) -> int:
    return k * (2 * N - k - 1) // 2 - _ipair(k, g)

def lam_det(p: int, N: int) -> int:
    return p * (-N * (N - 1) // 2)


# --------------------------- canonical loops ---------------------------
# The bar-fixed normalization is (-q)^{f=1 exponent} · q^{dressing} on each
# generator: the SIGN sits on the f-independent (f=1) part of the exponent,
# (-1)^{lam at f=1}, NOT on the dressing.  This is the unique choice that makes
# rho a SIGN-FREE permutation (Tr(L_a·L_{rho(a)})=delta+O(q), +1 on the diagonal)
# AND the M-test clean (lowest deg_W of L·M = 0 for every canonical).  The
# q-power magnitude is the existing lam_E/lam_F/lam_det.
def _qpow(c: int, N: int, sign: int = 1) -> DOp:
    return DOp.from_scalar(VRational.from_scalar(LaurentPoly({c: sign}), n=N), n=N)

def _sgn_E(k: int) -> int:        # sign of (-q)^{-k(k-1)/2}
    return -1 if (k * (k - 1) // 2) % 2 else 1

def _sgn_F(k: int, N: int) -> int:  # sign of (-q)^{+k(2N-k-1)/2}
    return -1 if (k * (2 * N - k - 1) // 2) % 2 else 1

def _sgn_det(p: int, N: int) -> int:  # det^p = E_N(0)^p / F_N(0)^{|p|}
    return -1 if (abs(p) * N * (N - 1) // 2) % 2 else 1

# The dressed-minuscule generators are memoized: pure functions returning
# immutable DOps (value objects), reused across every label in a basis
# build — the batch reuse that dominates AbeKAlgebra L-finding.
@lru_cache(maxsize=16384)
def _L_E_cached(k: int, f: tuple, N: int) -> DOp:
    return _qpow(lam_E(k, f, N), N, _sgn_E(k)) * _ray_img_E(k, f, N)

@lru_cache(maxsize=16384)
def _L_F_cached(k: int, g: tuple, N: int) -> DOp:
    return _qpow(lam_F(k, g, N), N, _sgn_F(k, N)) * _ray_img_F(k, g, N)

@lru_cache(maxsize=4096)
def _L_det_cached(p: int, N: int) -> DOp:
    return _qpow(lam_det(p, N), N, _sgn_det(p, N)) * _det_img(p, N)

def L_E(k: int, f, N: int) -> DOp:
    return _L_E_cached(k, tuple(int(x) for x in f), N)

def L_F(k: int, g, N: int) -> DOp:
    return _L_F_cached(k, tuple(int(x) for x in g), N)

def L_det(p: int, N: int) -> DOp:
    return _L_det_cached(p, N)

def L_W(lam, N: int) -> DOp:
    """Wilson canonical `χ_λ(v)` — the irreducible gl_N character of the
    dominant weight `λ`.  Degree-0 (no q-prefactor); the balanced weight
    `(a,…,a)` folds to the det-Wilson monomial `(v_1…v_N)^a` automatically
    (a single Schur term)."""
    e = tuple(sorted((int(x) for x in lam), reverse=True))
    return DOp.from_scalar(_wilson_sym(e, N), n=N)


# --------------------------- Schur trace ---------------------------
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

_MEASURE_CACHE: dict = {}
_POCH_CACHE: dict = {}
_INVPOCH_CACHE: dict = {}

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

def trace(D: DOp, N: int, K: int = 8, adaptive: bool = False,
          watch: int | None = None) -> LaurentPoly:
    """Pure-U(N) Schur-index trace of a chart operator D, to q^K.

        Tr(D) = (q^2;q^2)_inf^{2N}/N! * oint dv/(2pi i v) Schur-measure * D(u=0,v)

    Works on the *rational* u^0 sector (m=0 denominators cancel the
    measure -> exact numerator division; m!=0 ones are q-expanded).

    `adaptive`: when True, the internal q-cutoff is raised to reach the
    operator's NEGATIVE q-extent so that the requested low orders (esp. q^0) are
    EXACT -- needed for the q^0 orthonormality/Gram-Schmidt pairings of high-λ
    canonicals.  `watch`: when an int (used with `adaptive`), the escalation
    stops as soon as THAT single coefficient stabilises (rather than the whole
    `q<=K` poly) -- for callers that consume only one order (e.g. the `q^0`
    self-norm gate), this reaches the answer in far fewer `core` evaluations.
    When False (default) the fixed cutoff K is used: correct for
    the LEADING power (`min`, used by centering -- truncation only drops high q)
    and for low-charge operators, and far cheaper.  (The exact-at-all-orders
    Habiro form -- residual (q^k v_i/v_j;q^2)_inf with Habiro coefficients --
    would remove the adaptive blow-up entirely.)"""
    return trace_v0(D._terms.get((0,) * N), N, K, adaptive=adaptive, watch=watch)


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

def inner(a: DOp, b: DOp, N: int, K: int = 8) -> LaurentPoly:
    """Tr(a*b) for canonical loops a,b -> element of Z[[q]] (q^K-truncated)."""
    return trace(a * b, N, K)


# --------- rho: charge conjugation + Witten root-shift (on labels) ---------
def witten_shift(k: int, N: int) -> tuple:
    """Witten shift  delta_k = sum of the positive roots omega_k crosses
       = ((N-k)^k, (-k)^{N-k}).  delta_1 = (1,-1) at N=2; (2,-1,-1) at N=3."""
    return tuple([N - k] * k + [-k] * (N - k))

def _w0neg(lam) -> tuple:
    return tuple(-int(x) for x in reversed(lam))

# NOTE: rho is a SIGN-FREE permutation of canonical basis labels.  The sign that
# would otherwise appear in rho(E_k)~F_k is ALREADY absorbed into the generator
# normalization L_E/L_F (the (-q)^{...} prefactors), so NO separate rho-sign is
# applied anywhere.  rho_label / rho_inverse_label return a bare image label.

def rho_label(label, N: int):
    """rho (the rho-automorphism / charge-conjugation half-monodromy) on a
    canonical generator label -> image_label (a SIGN-FREE permutation; the sign
    is already carried by the L_E/L_F normalization, so none appears here).

    Labels:  ('E',k,f)  ('F',k,g)  ('W',lam)  ('det',p).

        rho(E_k(f)) = F_k(delta_k - f)
        rho(F_k(g)) = E_k(-delta_k - g)
        rho(W_lam)  = W_{-w0 lam},   rho(det^p) = det^{-p}

    with delta_k = witten_shift(k,N).  Reconstructed from the Schur-trace
    orthogonality  Tr(rho(L_a) L_b) = delta_{a,b} + O(q): rho(a) is the
    q^0 trace-dual of a.  Verified at N=2,3.  Consequence:
        rho^2(E_k(f)) = E_k(f - 2 delta_k),  rho^2(F_k(g)) = F_k(g + 2 delta_k),
    so rho^2 != id on every monopole (delta_k != 0) and rho^2 = id only on
    the central Wilson/det sector -- the Witten effect."""
    t = label[0]
    if t == 'E':
        k = int(label[1]); f = tuple(int(x) for x in label[2]); d = witten_shift(k, N)
        return ('F', k, tuple(d[i] - f[i] for i in range(N)))
    if t == 'F':
        k = int(label[1]); g = tuple(int(x) for x in label[2]); d = witten_shift(k, N)
        return ('E', k, tuple(-d[i] - g[i] for i in range(N)))
    if t == 'W':
        return ('W', _w0neg(label[1]))
    if t == 'det':
        return ('det', -int(label[1]))
    raise ValueError(f"rho_label: unknown generator {label!r}")

def rho2_label(label, N: int):
    """rho^2 applied to a label -> image_label (a bare permutation; no sign)."""
    return rho_label(rho_label(label, N), N)


def rho_inverse_label(label, N: int):
    """rho^{-1} (the inverse half-monodromy) on a canonical generator label
    -> image_label (a SIGN-FREE permutation).  The exact inverse of `rho_label`
    (just swap the E<->F roles and the shift sign):

        rho^{-1}(E_k(f)) = F_k(-delta_k - f)
        rho^{-1}(F_k(g)) = E_k( delta_k - g)
        rho^{-1}(W_lam)  = W_{-w0 lam},   rho^{-1}(det^p) = det^{-p}

    with delta_k = witten_shift(k, N).  Verified `rho(rho^{-1}(x)) = x` and
    `rho^{-1}(rho(x)) = x` on every generator: e.g.
    rho(rho^{-1}(E_k(f))) = rho(F_k(-d-f)) = E_k(-d-(-d-f)) = E_k(f).
    `W`/`det` are rho-involutive (rho^2 = id on the central sector), so
    rho^{-1} = rho there."""
    t = label[0]
    if t == 'E':
        k = int(label[1]); f = tuple(int(x) for x in label[2]); d = witten_shift(k, N)
        return ('F', k, tuple(-d[i] - f[i] for i in range(N)))
    if t == 'F':
        k = int(label[1]); g = tuple(int(x) for x in label[2]); d = witten_shift(k, N)
        return ('E', k, tuple(d[i] - g[i] for i in range(N)))
    if t == 'W':
        return ('W', _w0neg(label[1]))
    if t == 'det':
        return ('det', -int(label[1]))
    raise ValueError(f"rho_inverse_label: unknown generator {label!r}")


# --------------------------- generator chart image ---------------------------
def dop_of(label, N: int) -> DOp:
    """Canonical DOp (bar-centered loop) for a generator label.

    Labels:  ('E',k,f)  ('F',k,g)  ('W',lam)  ('det',p)."""
    t = label[0]
    if t == 'E':
        return L_E(int(label[1]), label[2], N)
    if t == 'F':
        return L_F(int(label[1]), label[2], N)
    if t == 'W':
        return L_W(label[1], N)
    if t == 'det':
        return L_det(int(label[1]), N)
    raise ValueError(f"dop_of: unknown generator {label!r}")

def rho_pairing_const(a_label, b_label, N: int, K: int = 4) -> int:
    """Constant (q^0) part of  Tr(rho(a) . b) = <a,b> + O(q).

    This is the *leading orthonormality functional* <a,b> := [Tr(rho(L_a) L_b)]_{q^0}.
    For a genuine canonical pair it is delta_{a,b}; off-diagonal q^0 entries
    signal that the dressing lattice over-/under-closes rho.  Use it to test
    which dressings f are canonical and whether anything beyond same-colour +
    uniform (w2) is needed.

    Only the q^0 coefficient is returned, so a small trace cutoff K suffices
    (the trace's boundary artifact eats only the top order)."""
    a2 = rho_label(a_label, N)
    p = inner(dop_of(a2, N), dop_of(b_label, N), N, K)
    return p._coeffs.get(0, 0)


# =====================================================================
# SECTION 2 — canonical-basis construction (formerly pure_un_construct.py)
#
# The canonical basis L_{(m,λ)} is built on demand by `CanonicalBasis._build_clean`
# (the only legitimate constructor), in order of cost:
#   - closed-form generators (Wilson, E_k/F_k, det);
#   - PRINCIPAL QTCone monomials `_cone_build`: q-commuting products of the cone
#     rays (low_k, mut_k = φ_{-1}(low_k), det, w2) slid by the free φ_n / w2^l --
#     self-norm 1 by construction;
#   - φ-tower reduction (free slides of the tower representative);
#   - `w_R·cone` peel `_wR_cone_build`: an asymmetric-Levi dressing as a single
#     Wilson·cone product peeled against the cone;
#   - a two-canonical-product M-test peel as the general fallback.
# `multiply` decomposes `image(a)·image(b)` onto this basis.
# =====================================================================

_trace = trace            # Schur trace


def _sc(c: int, N: int) -> DOp:
    return DOp.from_scalar(VRational.from_scalar(LaurentPoly({0: int(c)}), n=N), n=N)


# Serialized-registry cache version.  BUMP whenever the canonical-basis
# *generation* changes, so a stale on-disk cache is rejected and rebuilt rather
# than silently serving an out-of-date basis.
_CANONICAL_CACHE_VERSION = 17


def _gen(label, N: int) -> DOp:
    """Chart image of a single centered generator label."""
    return dop_of(label, N)


def _rgen(label, N: int) -> DOp:
    """rho(centered generator) = dop_of(rho_label) -- sign-free (the sign is
    already in the L_E/L_F normalization)."""
    return dop_of(rho_label(label, N), N)


def _product(specs, N: int, conj: bool = False) -> DOp:
    """Product of generator labels.  `conj=True` builds the rho-image
    (rho is an algebra automorphism: rho(prod) = prod rho, same order)."""
    P = DOp.one(N)
    g = (lambda l: _rgen(l, N)) if conj else (lambda l: _gen(l, N))
    for l in specs:
        P = P * g(l)
    return P


def _rigen(label, N: int) -> DOp:
    """rho^{-1}(centered generator) = dop_of(rho_inverse_label) -- sign-free."""
    return dop_of(rho_inverse_label(label, N), N)


def _riproduct(specs, N: int) -> DOp:
    """Product of generator labels under rho^{-1} (an algebra automorphism, so
    rho^{-1}(prod) = prod of rho^{-1}, same order) -- the rho^{-1}-image used to
    key `rho_inverse` on canonical (m, e) labels symmetrically with `_product(.,
    conj=True)` for `rho`."""
    P = DOp.one(N)
    for l in specs:
        P = P * _rigen(l, N)
    return P


def _u0_block(rA: DOp, B: DOp, N: int):
    """The `u^0` (constant-magnetic) block of the product `rA·B`, formed directly
    as `Σ_a (rA)_a · q_shift_a((B)_{-a})` -- the *only* block the trace pairing
    reads.  Skips the other `u`-blocks of the full product, the dominant cost of
    the Gram-Schmidt build (`DOp.__mul__`).  Returns a `VRational` or `None`."""
    Bt = B._terms
    out = None
    for a, fa in rA._terms.items():
        gb = Bt.get(tuple(-x for x in a))
        if gb is None:
            continue
        coef = fa * gb.q_shift(a)
        out = coef if out is None else out + coef
    return out


def _trace_prod(rA: DOp, B: DOp, N: int, K: int, adaptive: bool = False,
                watch: int | None = None):
    """`Tr(rA·B)` computing only the `u^0` block of the product (see `_u0_block`)."""
    u0 = _u0_block(rA, B, N)
    if u0 is None or u0.is_zero():
        return LaurentPoly.zero()
    return _trace(DOp({(0,) * N: u0}, n=N), N, K, adaptive=adaptive, watch=watch)


def _q0(rA: DOp, B: DOp, N: int, K: int, adaptive: bool = False) -> int:
    """q^0 coefficient of the trace pairing Tr(rA . B).  `adaptive=True` makes
    q^0 EXACT for high-charge operators (reaches the negative-q extent via the
    cheap Euler measure); default False keeps the fast fixed cutoff -- correct
    for low charges now that the measure is exact (Euler form).

    When adaptive, only the `q^0` coefficient is consumed, so the escalation
    watches `q^0` alone (`watch=0`) and stops as soon as it stabilises -- the
    full `q<=K` poly need not converge."""
    return _trace_prod(rA, B, N, K, adaptive=adaptive,
                       watch=0 if adaptive else None)._coeffs.get(0, 0)


def _lowest_charge(L: DOp, N: int):
    """Canonical label `(m, λ)` of a chart image: the **leading 't Hooft-Wilson
    label** -- the anti-dominant magnetic `m = u_bot` (lowest u) and the
    **dominant (lex-max) v-weight of the lowest-q term there** (the Levi-irrep
    highest weight).  (Was lex-min-v, which conflated distinct Levi-weights at
    regular m.)"""
    Ls = _abe_simplify_dop(L)
    ks = [u for u, c in Ls._terms.items() if not c.simplify().is_zero()]
    if not ks:
        return None
    u_bot = min(ks, key=_abe_u_order)
    num = Ls._terms[u_bot].simplify().num
    lowq = None
    best = None
    for v, lp in num._terms.items():
        if lp.is_zero():
            continue
        q = min(lp._coeffs)
        if lowq is None or q < lowq:
            lowq = q
            best = v
        elif q == lowq and v > best:
            best = v
    return (u_bot, best)


def _ansatz_label(L: DOp, N: int):
    """The Kapustin 't Hooft-Wilson label `(m, e)` of a chart image, read off
    its **leading u-orbit** by matching the ansatz

        L_{m,e} = q^λ · Σ_{w∈W/W_m} χ_e[Levi_m](w·v) · w(U_m^dom)  +  (lower-u bubbling)

    `m` is the **dominant** magnetic weight (the most-dominant u-power); `e` is
    the **Levi-dominant** electric weight (highest weight of the `L_m`-irrep
    `χ_e`), obtained by dividing the leading block's coefficient by the dominant
    cone atom `U_m^dom` and taking the per-Levi-block dominant weight of the
    residual character.  This is chamber-free and `δ(m)`-independent: a bare
    monopole (trivial character `χ_0`) reads `e = 0` at every `m`.

    Returns `(m_dom, e)`, or `None` if the leading block is not a single
    `U_m^dom`-monomial dressing (i.e. `L` is not a clean single canonical's
    leading orbit).  Validated over the full N=2,3 basis: Levi-dominant `e`,
    injective per canonical, `e=0` on bare monopoles.
    """
    # lazy import: pure_un_closed_form imports from this module (L_E/L_det/...),
    # so the dependency is resolved at call time, not import time.
    from pure_un_closed_form import (dom_atom, _dom_weight, _levi_blocks,
                                     _div_num_by_monomial, _dominance_key)
    Ls = _abe_simplify_dop(L)
    terms = [u for u, c in Ls._terms.items() if not c.simplify().is_zero()]
    if not terms:
        return None
    p_lead = max(terms, key=_dominance_key)
    m_dom = tuple(sorted(p_lead, reverse=True))
    cD = Ls._terms[m_dom].simplify()
    cA = dom_atom(m_dom, N)._terms[m_dom].simplify()
    if cD.den != cA.den or cD.sq != cA.sq:
        return None
    ratio = _div_num_by_monomial(cD.num, cA.num, N)   # = the Levi character χ_e
    if ratio is None:
        return None
    e = _dom_weight(ratio, _levi_blocks(m_dom), N)
    return (m_dom, tuple(e))


# =====================================================================
# rho / rho^{-1} on a chart image DIRECTLY (sqrt-measure conjugation).
#
# Exact per-u-block identity ("ρ as conjugation by
# √measure"):  with J = [u_i -> q^{2i}/u_i, v_i -> 1/v_i] (an involution),
#
#     rho(L) = J( Σ_k L_k · factor_k · u^k ),
#     factor_k = ∏_{i<j} (-1)^{m_ij} q^{-m_ij(m_ij-1)} (v_i/v_j)^{-m_ij},  m_ij = k_i-k_j.
#
# Worked out closed-form per block (k = u-power, S = Σ_t k_t):
#     factor_k = sign_k · q^{qf_k} · v^{wexp_k},
#       sign_k  = (-1)^{Σ_t (N-1-2t) k_t},
#       qf_k    = -Σ_{i<j} (k_i-k_j)(k_i-k_j-1),
#       wexp_k[t] = S - N·k_t,
# and J contributes q^{2 Σ_t t k_t} with u^k -> u^{-k}, v -> 1/v.  Hence
#
#     rho(L)_{-k} = sign_k · q^{qf_k + 2Σ_t t k_t} · v^{-wexp_k} · vinv(L_k),
#
# hard-coded from L: no build, no search.  rho^{-1} inverts the same monomial.
# =====================================================================

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


def _rho_chart(L: DOp, N: int) -> DOp:
    """`ρ(L)` computed directly from the chart image (√measure conjugation, exact,
    no build/search):  `ρ(L)_{-k} = sign_k q^{qpow_k} v^{-wexp_k} · vinv(L_k)`."""
    Ls = _abe_simplify_dop(L)
    out = {}
    for k, ck in Ls._terms.items():
        sign, qpow, wexp = _rho_block_data(k, N)
        mono = VRational.from_vlaurent(VLaurent(
            {tuple(-x for x in wexp): LaurentPoly({qpow: sign})}, n=N))
        out[tuple(-x for x in k)] = _vinv_vrational(ck, N) * mono
    return _abe_simplify_dop(DOp(out, n=N, n_gauge=N))


def _rho_inv_chart(R: DOp, N: int) -> DOp:
    """`ρ⁻¹(R)` from the chart image -- the exact inverse of `_rho_chart`.  For R's
    block at `u^{k''}` the preimage block sits at `u^{-k''}=u^k`:
    `L_k = sign_k q^{-qpow_k} v^{-wexp_k} · vinv(R_{-k})`."""
    Rs = _abe_simplify_dop(R)
    out = {}
    for kpp, ckpp in Rs._terms.items():
        k = tuple(-x for x in kpp)
        sign, qpow, wexp = _rho_block_data(k, N)
        mono = VRational.from_vlaurent(VLaurent(
            {tuple(-x for x in wexp): LaurentPoly({-qpow: sign})}, n=N))
        out[k] = _vinv_vrational(ckpp, N) * mono
    return _abe_simplify_dop(DOp(out, n=N, n_gauge=N))


def _levi_perms(m, N):
    """Permutations of 0..N-1 that fix the magnetic charge `m` -- i.e. the Weyl
    group `W_{L_m}` of the Levi `L_m = ∏_b U(n_b)` (permutes only equal-`m`
    positions)."""
    from itertools import permutations
    return [p for p in permutations(range(N))
            if all(m[p[i]] == m[i] for i in range(N))]


def _perm_sign(p):
    s = 1
    seen = [False] * len(p)
    for x in range(len(p)):
        if not seen[x]:
            length = 0
            y = x
            while not seen[y]:
                seen[y] = True
                y = p[y]
                length += 1
            if length % 2 == 0:
                s = -s
    return s


# =====================================================================
# Symmetric (Weyl) M-test: decompose products into canonicals via the
# u-shifted measure, read in the symmetric grading.  This is the exact,
# bar-aware M-test grading -- it never
# relies on the (bar-blind) q^0 pairing, and it RESOLVES multi-canonical
# seeds (a product is sum_c q^{k_c} L_c, each summand a q-power monomial;
# bar antimultiplicativity forces C^c_{ab}(q) = C^c_{ba}(1/q)).
#
#   deg_W(q^e v^w u^a) = e - <u,w> + <rho,u>,   rho = (0,1,...,N-1)
#
# is the symmetric ordering :f(qv)u: (half-shift -<u,w>) plus the
# Weyl-vector centering supplied by M.  Under it EVERY canonical's leading
# Weyl orbit sits at deg_W = 0 with all bubbling at deg_W >= 1 (verified
# N=2,3): L_c . M = (signed Weyl orbit) + O(q), in ONE universal grading
# (no per-canonical t_c^0).  A summand q^{k_c} L_c therefore has its
# leading orbit exactly at deg_W = k_c, read off directly; bar sends
# deg_W -> -deg_W, so the two orderings expose k_c and -k_c and only the
# deg_W <= 0 window is ever needed.  M = prod_{i<j} theta(v_i/v_j; q^2) is
# kept FACTORED: f_a's two-variable denominators (v_i - q^m v_j) cancel the
# u-shifted M factors symbolically (each leaves a monomial), and only the
# surviving factors are expanded -- no full-M expansion, no series division.
# (The pure-U(N) chart only ever has two-variable denominators -- multiply
# q-shifts and add over a common (v_i-q^m v_j) denominator -- so no
# single-variable v_i^2 factor can arise; the `_sq` guard never fires.)
# =====================================================================

def _pun_weyl_deg(e, u, v, N):
    """Symmetric (Weyl) grading deg_W = e - <u,v> + <rho,u>, rho=(0..N-1)."""
    return e - sum(u[t] * v[t] for t in range(N)) + sum(t * u[t] for t in range(N))


def _pun_LM_le(D: DOp, N: int, cap: int = 0, KF: int | None = None):
    """Fast symbolic `L . M` graded by `deg_W`, keeping `deg_W <= cap`.

    Returns `{(deg_W, u_tuple, v_tuple): int}`.  `M = prod_{i<j}
    theta(v_i/v_j; q^2)/(q^2;q^2)_inf` is kept as its Jacobi binomial factors;
    per `u^a` term the two-variable denominators are cancelled against the
    `u`-shifted factors symbolically and only the surviving factors (those whose
    per-term `deg_W`-cost `e_base + <a,d>` is within reach of `cap`) are expanded.
    Exact: validated == the expand-then-divide measure product on the `deg_W<=cap`
    window (N=2 canonicals + products) and against `multiply()`."""
    Ds = _abe_simplify_dop(D)
    rho = tuple(range(N))
    if KF is None:
        # generous factor budget: enough to cancel denominators and to cover the
        # cap window plus the negative-cost lowering budget (u-spread).  Pruning
        # below keeps the actual work small; this only bounds factor *generation*.
        spread = 0
        for a in Ds._terms:
            if a:
                spread = max(spread, max(a) - min(a))
        KF = max(12, cap + 2 * spread + 8)
    out: dict = {}
    for a, vr in Ds._terms.items():
        f = vr.simplify()
        if f.num.is_zero():
            continue
        if f._sq:
            raise RuntimeError("pure U(N) chart cannot have single-variable denominators")
        rhoa = sum(rho[t] * a[t] for t in range(N))
        # --- cancel f's denominators against u-shifted M factors (symbolic) ---
        monq = 0
        monv = [0] * N
        sign = 1
        used = set()
        for (i, j, m), mult in f._den.items():
            for _ in range(mult):
                kA = (-m) - 2 * (a[i] - a[j])      # family A: (1 - q^{2k} v_i/v_j)
                kB = (m - 2) - 2 * (a[j] - a[i])    # family B: (1 - q^{2k+2} v_j/v_i)
                if m % 2 == 0 and kA >= 0 and kA % 2 == 0 and ('A', i, j, kA // 2) not in used:
                    used.add(('A', i, j, kA // 2)); monq += -m; monv[j] -= 1; sign = -sign
                elif m % 2 == 0 and kB >= 0 and kB % 2 == 0 and ('B', i, j, kB // 2) not in used:
                    used.add(('B', i, j, kB // 2)); monv[i] -= 1
                else:
                    raise RuntimeError(
                        f"denominator (v_{i}-q^{m} v_{j})^... not cancelled by M (u^{a})")
        # --- surviving M factors: cost + the TWO v-positions each shifts (the
        # factor multiplies by (1 - q^cost v_pi / v_pj), so v_pi += 1, v_pj -= 1).
        # Storing (cost, pi, pj) lets the hot expansion below mutate two entries
        # instead of rebuilding the whole v-tuple via a range(N) genexpr. ---
        surv = []  # (cost, pi, pj)
        for i in range(N):
            for j in range(i + 1, N):
                for k in range(KF + 1):
                    if ('A', i, j, k) not in used:
                        surv.append((2 * k + a[i] - a[j], i, j))
                    if ('B', i, j, k) not in used:
                        surv.append((2 * k + 2 + a[j] - a[i], j, i))
        # --- base monomials: numerator * leftover cancellation monomial ---
        base: dict = {}
        for ve, lp in f.num._terms.items():
            for qe, c in lp._coeffs.items():
                w = tuple(ve[t] + monv[t] for t in range(N))
                dw = (qe + monq) - sum(a[t] * w[t] for t in range(N)) + rhoa
                base[(dw, w)] = base.get((dw, w), 0) + sign * c
        # --- expand surviving factors, pruning deg_W beyond reach of lowerers ---
        order = sorted(surv)
        suff_neg = [0] * (len(order) + 1)
        for idx in range(len(order) - 1, -1, -1):
            suff_neg[idx] = suff_neg[idx + 1] + (order[idx][0] if order[idx][0] < 0 else 0)
        cur = dict(base)
        for idx, (cost, pi, pj) in enumerate(order):
            bound = cap - suff_neg[idx + 1]
            # Multiply cur by (1 - q^cost v_pi/v_pj): the "1" keeps each term, the
            # second shifts it by (cost, +pi, -pj) and negates.  Build the next
            # dict in ONE pass with the deg_W <= bound prune applied inline (a term
            # beyond `bound` can never be pulled back under `cap` by the remaining
            # negative-cost factors), so we avoid both the full `dict(cur)` copy and
            # a separate prune-comprehension -- the hot line of the build.
            nxt: dict = {}
            ng = nxt.get
            for (dw, w), coef in cur.items():
                if dw <= bound:
                    nxt[(dw, w)] = ng((dw, w), 0) + coef
                ndw = dw + cost
                if ndw <= bound:
                    wl = list(w); wl[pi] += 1; wl[pj] -= 1
                    key = (ndw, tuple(wl))
                    nxt[key] = ng(key, 0) - coef
            cur = {kw: c for kw, c in nxt.items() if c}
        for (dw, w), coef in cur.items():
            if dw <= cap and coef:
                out[(dw, a, w)] = out.get((dw, a, w), 0) + coef
    return {k: v for k, v in out.items() if v}


def _pun_leading_dw(D: DOp, N: int):
    """EXACT minimum `deg_W` of the covariant M-test `_pun_LM_le(D)`, computed
    WITHOUT the full slice -- the fast leading-order read for `_cone_recenter`.

    The surviving `M`-factors multiply by `(1 - 𝖖^{cost} v_pi/v_pj)`; the shift
    branch moves `deg_W` by `cost`.  POSITIVE-cost factors therefore only RAISE
    `deg_W`, so they cannot affect the minimum and are dropped entirely; only the
    NEGATIVE-cost factors are expanded (with coefficients, so any cancellation at
    the extremal degree is handled exactly).  This is provably equal to
    `min(dw for (dw,u,v) in _pun_LM_le(D))` (validated == on every cone-recenter
    call across the N=2..5 scopes) while skipping the bulk positive-cost expansion
    that dominates the full M-test.  Returns the min `deg_W`, or `None` if a
    denominator is not cancelled by `M` (caller falls back to the full M-test)."""
    Ds = _abe_simplify_dop(D)
    rho = tuple(range(N))
    spread = 0
    for a in Ds._terms:
        if a:
            spread = max(spread, max(a) - min(a))
    KF = max(12, 2 * spread + 8)
    glob = None
    for a, vr in Ds._terms.items():
        f = vr.simplify()
        if f.num.is_zero():
            continue
        if f._sq:
            return None
        rhoa = sum(rho[t] * a[t] for t in range(N))
        monq = 0
        monv = [0] * N
        sign = 1
        used = set()
        for (i, j, m), mult in f._den.items():
            for _ in range(mult):
                kA = (-m) - 2 * (a[i] - a[j])
                kB = (m - 2) - 2 * (a[j] - a[i])
                if m % 2 == 0 and kA >= 0 and kA % 2 == 0 and ('A', i, j, kA // 2) not in used:
                    used.add(('A', i, j, kA // 2)); monq += -m; monv[j] -= 1; sign = -sign
                elif m % 2 == 0 and kB >= 0 and kB % 2 == 0 and ('B', i, j, kB // 2) not in used:
                    used.add(('B', i, j, kB // 2)); monv[i] -= 1
                else:
                    return None                       # uncancelled denom -> fall back
        negf = []                                     # negative-cost factors only
        for i in range(N):
            for j in range(i + 1, N):
                for k in range(KF + 1):
                    if ('A', i, j, k) not in used:
                        c = 2 * k + a[i] - a[j]
                        if c < 0:
                            negf.append((c, i, j))
                    if ('B', i, j, k) not in used:
                        c = 2 * k + 2 + a[j] - a[i]
                        if c < 0:
                            negf.append((c, j, i))
        cur: dict = {}
        for ve, lp in f.num._terms.items():
            for qe, cc in lp._coeffs.items():
                w = tuple(ve[t] + monv[t] for t in range(N))
                dw = (qe + monq) - sum(a[t] * w[t] for t in range(N)) + rhoa
                cur[(dw, w)] = cur.get((dw, w), 0) + sign * cc
        for (c, pi, pj) in sorted(negf):
            nxt = dict(cur)
            for (dw, w), coef in cur.items():
                wl = list(w); wl[pi] += 1; wl[pj] -= 1
                key = (dw + c, tuple(wl))
                nxt[key] = nxt.get(key, 0) - coef
            cur = {k2: v for k2, v in nxt.items() if v}
        if cur:
            mk = min(dw for (dw, w) in cur)
            glob = mk if glob is None else min(glob, mk)
    return glob


def _pun_phi(D: DOp, n: int, N: int) -> DOp:
    """Dressing-shift quantum-torus automorphism `φ_n: u_i ↦ 𝖖^{−n} u_i v_i^n`.

    On the canonical basis `φ_n` is an EXACT, `ρ`-equivariant permutation
    `φ_n(L_{(m,e)}) = L_{(m, e + n·m)}` -- no `𝖖`-power -- i.e. it slides along the
    dressing tower at fixed magnetic charge `m`.  (A product/image of trusted
    `L`'s stays trusted under `φ_n`; verified `L` and `ρL` exact, N=2.)

    QUANTUM-TORUS CONVENTION -- the thing that bites, document once:
    `DOp` normal form is **v-LEFT, u-RIGHT**, with `u_i v_j = 𝖖^{2δ_ij} v_j u_i`.
    So a `u`-left generator image like `𝖖^{−n} u_i v_i^n` MUST be normal-ordered
    before use: moving `u_i` right past `v_i^n` adds `𝖖^{2n}`, so the DOp form is

        φ_n(u_i)    = 𝖖^{+n} · v_i^{ n} · u_i        (NOT 𝖖^{−n})
        φ_n(u_i^-1) = 𝖖^{+n} · v_i^{-n} · u_i^-1      (forced by φ(u_i)φ(u_i^-1)=1)

    Using the raw `𝖖^{−n}` as the `v^n` coefficient is the classic mis-ordering --
    it leaves a spurious `𝖖`-power (`𝖖^{−2}` instead of `𝖖^0` per slide)."""
    # Direct per-term map (no operator multiplication).  φ_n fixes every `v_i`, so
    # the v-only coefficient (numerator AND denominators) is untouched on the v
    # functions -- only the `u^a` factor transforms.  Normal-ordering
    # `(𝖖^{+n} v_i^n u_i)^{a_i}` (and its inverse) collects, per direction `i`,
    # exactly `𝖖^{n·a_i^2} v_i^{n·a_i} u_i^{a_i}` (the `2n` from each `u_i v_i =
    # 𝖖^2 v_i u_i` pass, summed over the `a_i(a_i-1)/2` crossings, plus the `n·a_i`
    # leading 𝖖's, gives `n·a_i^2`).  Distinct directions commute, so a term
    # `vr(v)·u^a` maps to `𝖖^{n·Σa_i^2} · v^{n·a} · vr(v) · u^a`: a numerator
    # v-monomial shift + uniform 𝖖-shift, denominators carried unchanged.  Validated
    # == the former iterated-product form (𝖖^{+n} ordering) on every built canonical
    # across the N=2..5 scopes (1000+ slides, 0 mismatches).
    out: dict = {}
    for a, vr in D._terms.items():
        qp = n * sum(ai * ai for ai in a)            # 𝖖^{n·Σ a_i^2}
        vsh = tuple(n * ai for ai in a)              # v^{n·a}
        nt = {}
        for ve, lp in vr.num._terms.items():
            nve = tuple(ve[t] + vsh[t] for t in range(N))
            nt[nve] = LaurentPoly({e + qp: c for e, c in lp._coeffs.items()})
        out[a] = VRational(VLaurent(nt, n=N), dict(vr.den), n=N, sq=dict(vr.sq))
    return _abe_simplify_dop(DOp(out, n=N, n_gauge=D._n_gauge))


def _default_deg(specs):
    """Charge-order key for a generator-product seed: (n_factors, magnetic, electric).

    Wilson factors carry no magnetic charge but add electric weight `Σ|λ|`, so
    a singly-Wilson-dressed monopole sorts just after its bare monopole — the
    low-charge-first order that minimises defer-retry passes."""
    md = ed = 0
    for l in specs:
        if l[0] in ('E', 'F'):
            md += int(l[1])
            ed += sum(int(x) for x in l[2])
        elif l[0] == 'det':
            md += abs(int(l[1]))
        elif l[0] == 'W':
            ed += sum(abs(int(x)) for x in l[1])
    return (len(specs), md, ed)


def _dop_to_obj(D: DOp):
    """JSON-able representation of a chart `DOp` (all-integer data)."""
    terms = []
    for u, vr in D._terms.items():
        vr = vr.simplify()
        num = [[list(v), [[e, c] for e, c in lp._coeffs.items()]]
               for v, lp in vr._num._terms.items() if not lp.is_zero()]
        den = [[i, j, m, mult] for (i, j, m), mult in vr._den.items()]
        sq = [[i, m, mult] for (i, m), mult in vr._sq.items()]
        terms.append([list(u), num, den, sq])
    return [D._n, D._n_gauge, terms]


def _dop_from_obj(o, N: int):
    """Reconstruct a chart `DOp` from `_dop_to_obj`."""
    n, ng, terms = o
    out = {}
    for ulist, num, den, sq in terms:
        vt = {tuple(v): LaurentPoly({int(e): int(c) for e, c in coeffs})
              for v, coeffs in num}
        vr = VRational(VLaurent(vt, n=n),
                       {(i, j, m): mult for i, j, m, mult in den}, n=n,
                       sq={(i, m): mult for i, m, mult in sq})
        out[tuple(ulist)] = vr
    return DOp(out, n=n, n_gauge=ng)


def _affine_apply(M, x):
    """Apply integer matrix `M` (list of rows) to vector `x`."""
    N = len(x)
    return tuple(sum(M[i][j] * x[j] for j in range(N)) for i in range(N))


def _mat_inverse(M):
    """Rational inverse of an integer N×N matrix (Gauss-Jordan over Fraction)."""
    from fractions import Fraction
    N = len(M)
    A = [[Fraction(M[i][j]) for j in range(N)]
         + [Fraction(1 if k == i else 0) for k in range(N)] for i in range(N)]
    for col in range(N):
        piv = next((r for r in range(col, N) if A[r][col] != 0), None)
        if piv is None:
            return None
        A[col], A[piv] = A[piv], A[col]
        pv = A[col][col]
        A[col] = [x / pv for x in A[col]]
        for r in range(N):
            if r != col and A[r][col] != 0:
                f = A[r][col]
                A[r] = [A[r][k] - f * A[col][k] for k in range(2 * N)]
    return [[A[i][N + j] for j in range(N)] for i in range(N)]


def _fit_affine_map(pts, N):
    """Fit an integer affine map `head_v = M·λ + b` from `(λ, head_v)` samples
    (need ≥ N+1 with N affinely-independent `λ`); returns `(M, b, M⁻¹)` or None.
    Used to invert a fingerprint head back to its canonical label `λ` at fixed
    magnetic `m` (the map is affine there)."""
    import itertools
    if len(pts) < N + 1:
        return None
    lam0, h0 = pts[0]
    rows = [tuple(lam[t] - lam0[t] for t in range(N)) for (lam, _h) in pts[1:]]
    rhs = [tuple(h[t] - h0[t] for t in range(N)) for (_l, h) in pts[1:]]
    for combo in itertools.combinations(range(len(rows)), N):
        R = [rows[i] for i in combo]
        Rmat = [[R[k][j] for k in range(N)] for j in range(N)]   # columns = r_k
        Rinv = _mat_inverse(Rmat)
        if Rinv is None:
            continue
        S = [rhs[i] for i in combo]
        Smat = [[S[k][i] for k in range(N)] for i in range(N)]   # row i = (s_k)_i
        Mf = [[sum(Smat[i][k] * Rinv[k][j] for k in range(N)) for j in range(N)]
              for i in range(N)]
        if any(Mf[i][j].denominator != 1 for i in range(N) for j in range(N)):
            continue
        M = [[int(Mf[i][j]) for j in range(N)] for i in range(N)]
        b = tuple(h0[t] - _affine_apply(M, lam0)[t] for t in range(N))
        if all(_affine_apply(M, lam) == tuple(h[t] - b[t] for t in range(N))
               for (lam, h) in pts):
            Minv = _mat_inverse(M)
            if Minv is not None:
                return (M, b, Minv)
    return None


# ---- closed-form fingerprint-head -> canonical-label (m, lambda) inversion ----
#
# The M-test fingerprint head of a pure-U(N) canonical L_{(m,lambda)} (the
# rho-shifted lex-max v of its normalized deg_W=0 covariant signed-Weyl orbit)
# determines lambda by an EXACT affine map at fixed magnetic m:
#
#       lambda = M^{-1} . (head_v - w0m . rho) + delta(m)
#
# with rho=(0,1,..,N-1).  This replaces the learned per-m head index (the old
# `_cone_head_map`/`_calibrate_m` fit): the relation is closed-form, O(N log N),
# and round-trips every materialized canonical (N=2,3) exactly.  Regression:
# `tests/test_pure_un_head_inversion.py`.

def _pun_delta(m):
    """Bare-monopole dressing `delta(m)_i = sum_{j<i}(m_i - m_j)` for `m` sorted
    ascending (anti-dominant): the magnetic-rank shift folded into the gauge
    charge (`e_gauge = e - delta(m)`).  `delta(c.(1,..,1)) = 0`; additive in m."""
    ms = sorted(m)
    return tuple(sum(ms[i] - ms[j] for j in range(i)) for i in range(len(m)))


def _pun_src_desc(m):
    """Stable descending-m sort `s`: `m[s[0]] >= m[s[1]] >= ...`, ties by
    ascending index.  The linear part `M` of the head_v<-lambda map permutes by
    `s`; `_pun_minv_apply` scatters back (its inverse)."""
    return sorted(range(len(m)), key=lambda i: (-m[i], i))


def _pun_minv_apply(m, w):
    """Apply `M^{-1}` (inverse of the descending-m sort permutation) to `w`."""
    s = _pun_src_desc(m)
    out = [0] * len(m)
    for i in range(len(m)):
        out[s[i]] = w[i]
    return tuple(out)


def _pun_w0m_rho(m):
    """`w0m . rho`, rho=(0,1,..,N-1): the constant head offset.  `w0m` reverses
    each equal-m block in the DOMINANT (descending-m) frame -- the long Weyl
    element of the Levi `L_m` acting on rho."""
    N = len(m)
    rho = tuple(range(N))
    dm = sorted(m, reverse=True)
    p = list(range(N))
    i = 0
    while i < N:
        j = i
        while j < N and dm[j] == dm[i]:
            j += 1
        p[i:j] = p[i:j][::-1]
        i = j
    return tuple(rho[p[i]] for i in range(N))


def _pun_head_to_lambda(m, head_v):
    """Closed-form fingerprint-head -> canonical label `lambda` for pure U(N).

    `m` anti-dominant magnetic cocharacter; `head_v` the rho-shifted lex-max `v`
    of the canonical's normalized deg_W=0 covariant signed-Weyl orbit (its
    fingerprint head, as `_norm_mtest`/`_fp_orbit0` produce it).  Returns

        lambda = M^{-1} . (head_v - w0m . rho) + delta(m)

    -- the dominant Levi-irrep weight labelling `L_{(m,lambda)}`.  Exact and
    O(N log N).  Inverts the head produced by the engine's own M-test, so feeding
    a built canonical's head back recovers its label (verified on the full
    N=2,3 basis)."""
    N = len(m)
    c = _pun_w0m_rho(m)
    eg = _pun_minv_apply(m, tuple(head_v[t] - c[t] for t in range(N)))
    d = _pun_delta(m)
    return tuple(eg[t] + d[t] for t in range(N))


def _weyl_antidom(label, N):
    """Anti-dominant-normalize a `(m, e)` build-key: the Weyl group acts on `m` and
    `e` JOINTLY, so a non-anti-dominant `m` labels the SAME canonical viewed in a
    rotated frame.  Sort `m` ascending and apply the same permutation to `e` -- the
    canonical's proper (anti-dominant) build key.  (e.g. N=2 ((0,-1),(-1,0)) ->
    ((-1,0),(0,-1)); `_lowest_charge` of a non-single-canonical product can read a
    descending-m label, which without this fails to build.)"""
    m, e = tuple(label[0]), tuple(label[1])
    if list(m) == sorted(m):
        return (m, e)
    sig = sorted(range(N), key=lambda i: m[i])
    return (tuple(m[i] for i in sig), tuple(e[i] for i in sig))


class CanonicalBasis:
    """Memoized registry of pure-U(N) canonical-basis chart images.

    `image(label)` returns the **centered rational-quantum-torus image**
    (`DOp`) of `L_{(m,λ)}` — the linchpin that unlocks `multiply` (decompose
    `image(a)·image(b)`) and `trace` (`trace(image(a))`).

    Canonicals are built on demand by `_build_clean` (the only legitimate
    constructor): closed forms, then the Principal-QTCone engine `_cone_build`
    (q-commuting cone-ray products slid by the free φ/w2 moves), the φ-tower
    reduction, and the `w_R·cone` peel for asymmetric Levi dressings, with a
    two-canonical-product M-test peel as the general fallback.  Each image is
    **memoized**; `rho_image`/`rho_inverse_image` ride alongside (ρ is an
    algebra automorphism, so it tracks the same construction).
    """

    def __init__(self, N: int, K: int = 6) -> None:
        self._N = N
        self._K = K
        # NB: the GS / Gram pairings below MUST use the full cutoff self._K.
        # A reduced cutoff is NOT safe: the q^0 coefficient of a bar-centered
        # trace is *not* K-independent -- the Weyl-integration prefactor
        # (q^2;q^2)^{2N} couples the measure's positive powers to the
        # operator's negative powers, so truncating the measure early corrupts
        # q^0.  (Verified: a K=2 cutoff misreads the centered E_1^2 self-norm
        # as 0 instead of 1, silently dropping every higher-gap canonical.)
        self._reg = []      # ordered [(label, L, rL)], lower charge first
        self._idx = {}      # label -> index
        self._riL = {}      # label -> rho^{-1}(L) DOp (parallel to _reg's rL)
        self._scope_rays = None  # ray set defining the buildable scope (lazy ensure)
        self._scope_mlen = 2     # max product length for the scope
        self._materialized = False  # whether the full scope basis has been built

    # ----- persistence: save/load the built registry (skip the ~minutes-long
    #       exact-trace rebuild) -----

    def save(self, path) -> None:
        """Serialize the built canonical registry (`L_{m,e}`, `ρ(L)`, `ρ⁻¹(L)`,
        keyed by `(m,e)`) to a JSON file, so it can be reloaded without rerunning
        the cone / w_R rebuild."""
        import json
        data = {
            "version": _CANONICAL_CACHE_VERSION, "N": self._N, "K": self._K,
            "reg": [[[list(ch[0]), list(ch[1])], _dop_to_obj(L), _dop_to_obj(rL)]
                    for (ch, L, rL) in self._reg],
            "riL": [[[list(ch[0]), list(ch[1])], _dop_to_obj(riL)]
                    for ch, riL in self._riL.items()],
        }
        with open(path, "w") as f:
            json.dump(data, f, separators=(",", ":"))

    @classmethod
    def load(cls, path) -> "CanonicalBasis":
        """Reconstruct a `CanonicalBasis` from a `save()` file (no rebuild)."""
        import json
        with open(path) as f:
            data = json.load(f)
        B = cls(int(data["N"]), int(data["K"]))
        B._cache_version = int(data.get("version", 0))
        N = B._N
        for chl, Lo, rLo in data["reg"]:
            ch = (tuple(chl[0]), tuple(chl[1]))
            B._idx[ch] = len(B._reg)
            B._reg.append((ch, _dop_from_obj(Lo, N), _dop_from_obj(rLo, N)))
        for chl, riLo in data["riL"]:
            ch = (tuple(chl[0]), tuple(chl[1]))
            B._riL[ch] = _dop_from_obj(riLo, N)
        B._materialized = True
        return B

    def add(self, label, seed_specs):
        """Construct + memoize `L_{label}` from a generator-product seed."""
        if label in self._idx:
            return self._reg[self._idx[label]][1]
        L = _product(seed_specs, self._N)
        rL = _product(seed_specs, self._N, conj=True)
        riL = _riproduct(seed_specs, self._N)
        for (_lb, Lb, rLb) in self._reg:
            c = _q0(rLb, L, self._N, self._K)
            if c:
                L = L + _sc(-c, self._N) * Lb
                rL = rL + _sc(-c, self._N) * rLb
                riL = riL + _sc(-c, self._N) * self._riL[_lb]
        self._idx[label] = len(self._reg)
        self._reg.append((label, L, rL))
        self._riL[label] = riL
        return L

    def add_image(self, charge, L: DOp, rL: DOp, riL: DOp):
        """Register a canonical directly by its chart image + rho-image +
        rho^{-1}-image (e.g. the identity, or Wilson canonicals that are not
        E/F products)."""
        if charge not in self._idx:
            self._idx[charge] = len(self._reg)
            self._reg.append((charge, L, rL))
            self._riL[charge] = riL
        return self

    def ensure_identity(self):
        """Register `L_{(0,..),(0,..)} = 1` (rho(1)=1=rho^{-1}(1))."""
        idc = ((0,) * self._N, (0,) * self._N)
        if idc not in self._idx:
            one = DOp.one(self._N)
            self.add_image(idc, one, one, one)
        return self

    def image(self, label) -> DOp:
        """Centered chart image (DOp) of `L_{label}` (memoized; constructed
        on demand if not yet registered)."""
        label = _weyl_antidom(label, self._N)
        if label not in self._idx:
            self.ensure(label)
        return self._reg[self._idx[label]][1]

    def rho_image(self, label) -> DOp:
        """Chart image of `rho(L_{label})` (memoized; on-demand)."""
        label = _weyl_antidom(label, self._N)
        if label not in self._idx:
            self.ensure(label)
        return self._reg[self._idx[label]][2]

    def rho_inverse_image(self, label) -> DOp:
        """Chart image of `rho^{-1}(L_{label})` (memoized; on-demand) -- the
        rho^{-1} counterpart of `rho_image`, built from the same seed via the
        inverse generator map (`_riproduct` / `rho_inverse_label`), so
        `rho_inverse` is TOTAL and closed-form, not a registry reverse-lookup."""
        label = _weyl_antidom(label, self._N)
        if label not in self._idx:
            self.ensure(label)
        return self._riL[label]

    def inner_q0(self, a, b) -> int:
        """`[Tr(rho(L_a) L_b)]_{q^0}` from the cached images."""
        return _q0(self.rho_image(a), self.image(b), self._N, self._K)

    def _reg_labels(self):
        """The currently-BUILT (registered/memoized) charges -- the lazy set.
        Used internally (M-test anchor map, ρ tables) so those never trigger a
        full-scope build."""
        return [lab for (lab, _L, _rL) in self._reg]

    # ----- ansatz-label bijection: (build key) <-> Kapustin (m, e) -----------
    # The registry is memoized by the BUILD key (`_lowest_charge`, the u_bot/
    # δ(m)-folded reading the constructor emits).  The CANONICAL public label is
    # the Kapustin 't Hooft-Wilson `(m, e)` read off the leading orbit
    # (`_ansatz_label`: dominant m, Levi-dominant e, bare monopole -> e=0).  The
    # two are in bijection (one canonical, two names); these maps carry it.

    def _sync_ansatz_index(self):
        """(Re)build the build-key <-> ansatz-(m,e) bijection over every
        registered canonical.  Idempotent; cheap.  Skips any canonical whose
        leading block is not a clean single-character dressing (`_ansatz_label`
        returns None) -- those are surfaced by `ansatz_index_health`."""
        if not hasattr(self, '_build2ansatz'):
            self._build2ansatz = {}
            self._ansatz2build = {}
        for (bk, L, _rL) in self._reg:
            if bk in self._build2ansatz:
                continue
            a = _ansatz_label(L, self._N)
            if a is None:
                continue
            self._build2ansatz[bk] = a
            self._ansatz2build[a] = bk
        return self

    def ansatz_of(self, build_key):
        """Kapustin `(m, e)` of a registered canonical given its build key."""
        self._sync_ansatz_index()
        return self._build2ansatz.get(build_key)

    def build_key_of(self, ansatz_label):
        """Build key of a registered canonical given its Kapustin `(m, e)`."""
        self._sync_ansatz_index()
        ansatz_label = (tuple(ansatz_label[0]), tuple(ansatz_label[1]))
        return self._ansatz2build.get(ansatz_label)

    def ansatz_index_health(self):
        """Diagnostic: `(n_built, n_labelled, injective, [collisions])` for the
        ansatz bijection over the current registry."""
        self._sync_ansatz_index()
        seen = {}
        for bk, a in self._build2ansatz.items():
            seen.setdefault(a, []).append(bk)
        coll = {a: bks for a, bks in seen.items() if len(bks) > 1}
        return (len(self._reg), len(self._build2ansatz), not coll, coll)

    def materialize(self):
        """Build the ENTIRE scope basis (all cwr ray-products of `set_scope`),
        once.  For full-basis iteration (tests, dictionary).  Lazy uses
        (multiply/trace/image) don't need it -- they build on demand via `ensure`.
        Idempotent."""
        if getattr(self, '_materialized', False) or self._scope_rays is None:
            return self
        from itertools import combinations_with_replacement as cwr
        # The scope defines TARGET CHARGES (lowest charge of each ray-product);
        # each is then built by the ONLY legitimate construction, `ensure`
        # (`_build_clean`).  `_product` here is used purely to read off the target
        # charge -- never to register a canonical.
        charges = set()
        for ln in range(1, self._scope_mlen + 1):
            for combo in cwr(range(len(self._scope_rays)), ln):
                specs = [self._scope_rays[i] for i in combo]
                ch = _lowest_charge(_product(specs, self._N), self._N)
                if ch is not None:
                    charges.add(ch)
        for ch in sorted(charges, key=lambda c: (self._mrank(c),
                                                 sum(abs(x) for x in c[1]))):
            self.ensure(ch)
        self._materialized = True
        return self

    def labels(self):
        """The basis labels.  If a scope was declared but not yet materialized,
        build it first (full-scope) -- so external iteration sees the whole basis;
        internal hot paths use `_reg_labels()` to stay lazy."""
        if not getattr(self, '_materialized', False) and self._scope_rays is not None:
            self.materialize()
        return self._reg_labels()

    # ----- symmetric (Weyl) M-test: bar-aware product decomposition ----------

    def build(self, seeds):
        """Bulk-add an ordered `[(label, seed), ...]` list."""
        for label, seed in seeds:
            self.add(label, seed)
        return self

    # ----- autogenerator: build the basis from a ray set, keyed by (m,e) -----

    def register_wilson(self, max_deg: int):
        """Register the pure Wilson canonicals `L_W(λ) = χ_λ(v)` (the
        magnetic-neutral / `m=0` sector) for every `λ = (a, b≤a, …)` with
        `Σ|λ_i| ≤ max_deg`, keyed by their lowest tropical charge.

        Wilson lines are λ-degree-0 (no q-commutation), mutually orthonormal
        and orthogonal to the monopoles at `q^0`, so each is registered
        directly (`add_image`) — no Gram-Schmidt.  `ρ(W_λ) = W_{-w0 λ}` is a
        sign-free Wilson, supplied as the cached rho-image.  This closes the
        Wilson sector of the registry (Wilson×Wilson = Littlewood-Richardson),
        which a matter (Wilson-valued `S_RG`) RG flow exercises."""
        from itertools import product as _iproduct
        N = self._N
        recs = []
        rng = range(-max_deg, max_deg + 1)
        for lam in _iproduct(rng, repeat=N):
            if list(lam) != sorted(lam, reverse=True):
                continue                       # dominant weights only
            if sum(abs(x) for x in lam) > max_deg:
                continue
            L = L_W(lam, N)
            rl = rho_label(('W', lam), N)
            ril = rho_inverse_label(('W', lam), N)       # rho^{-1}(W_λ)=W_{-w0 λ}
            recs.append((_lowest_charge(L, N), L, dop_of(rl, N), dop_of(ril, N), lam))
        for ch, L, rL, riL, lam in sorted(
                recs, key=lambda t: (sum(abs(x) for x in t[4]), t[0])):
            if ch is not None and not self.has(ch):
                self.add_image(ch, L, rL, riL)
        return self

    def image_at(self, m, e) -> DOp:
        """Chart image of `L_{m,e}` by lowest tropical charge (lowest-tropical-charge key)."""
        return self.image((tuple(m), tuple(e)))

    # ----- on-demand registry extension (structured batch build) ---------

    def _structured_seeds(self, dmax: int, mlen: int):
        """ρ-symmetric structured **seeds** for the monopole sector: products
        of bare cone monopoles (`E_k(0)`/`F_k(0)`, `det^±1`) up to `mlen`
        factors, each optionally dressed by a **single** Wilson character `χ_λ`
        (`Σ|λ| ≤ dmax`).

        Capping the Wilson dressing at one factor is **complete**: by
        Littlewood-Richardson `χ_λ·χ_μ = Σ_ν c^ν_{λμ} χ_ν`, so a multi-Wilson
        dressing is a linear combination of single-Wilson ones — it adds no new
        canonical and would only collapse to norm 0 in Gram-Schmidt.  This is
        the key optimisation: it replaces the `combinations_with_replacement`
        over the *combined* (monopole + all-Wilson) ray set — which generated
        ~thousands of wasteful multi-Wilson products — with `O(monopole-products
        × #Wilson)` productive seeds.

        ρ-symmetric (E↔F, det↔det⁻¹, W_λ↔W_{-w0λ}) so the build is ρ-closed —
        ρ permutes the registry, and `rho`/`rho_inverse` resolve on it."""
        from itertools import combinations_with_replacement as cwr, product as ip
        N = self._N
        monopole = ([('E', k, (0,) * N) for k in range(1, N)]
                    + [('F', k, (0,) * N) for k in range(1, N)]
                    + [('det', 1), ('det', -1)])
        wilson = [('W', lam) for lam in ip(range(-dmax, dmax + 1), repeat=N)
                  if list(lam) == sorted(lam, reverse=True)
                  and 0 < sum(abs(x) for x in lam) <= dmax]
        seeds = []
        for ln in range(1, mlen + 1):
            for combo in cwr(range(len(monopole)), ln):
                base = tuple(monopole[i] for i in combo)
                seeds.append(base)                       # bare monopole product
                for w in wilson:
                    seeds.append(base + (w,))            # singly Wilson-dressed
        return seeds

    # ----- targeted on-demand build (structure-informed, no search) ----------

    def _gen_charges(self):
        """Memoised additive lowest charges of the base generators `E_k(0)`,
        `det^{±1}` -- the building blocks of the closed-form seed."""
        g = getattr(self, "_gchg", None)
        if g is None:
            N = self._N
            g = {}
            for k in range(1, N):
                g[("E", k)] = _lowest_charge(_product((("E", k, (0,) * N),), N), N)
            g["det+"] = _lowest_charge(_product((("det", 1),), N), N)
            g["det-"] = _lowest_charge(_product((("det", -1),), N), N)
            self._gchg = g
        return g

    def set_scope(self, rays, max_len):
        """Declare the ray set + max product length defining the buildable scope.
        `materialize` reads off each scope charge's lowest ('t Hooft-Wilson) label
        and builds it on demand via `ensure` (the cone / `w_R·cone` engine)."""
        self._scope_rays = list(rays)
        self._scope_mlen = max_len
        return self

    @staticmethod
    def _mrank(c):
        return sum(abs(x) for x in c[0])

    def _closed_form_label(self, charge):
        """Analytic (m, λ) -> generator label for a KNOWN closed-form canonical
        (no bubbling, no enumeration, no product): W (m=0), E_k(f)/F_k(g) for
        1<=k<=N -- the single-box magnetic patterns; k=N is the det-sector with
        f=λ, so det·W = E_N(λ) -- and bare det^c (central m=c·(1,..,1), λ=0).
        Returns None for a genuine product charge.  The caller re-verifies that
        the built image actually has lowest charge `charge` (a mis-mapped f then
        falls through to the product construction rather than registering wrong)."""
        N = self._N
        m, lam = charge
        if not any(m):
            return ('W', tuple(lam)) if list(lam) == sorted(lam, reverse=True) else None
        asc = list(m) == sorted(m)
        if asc and all(x in (0, 1) for x in m):
            k = sum(m)
            if 1 <= k <= N and tuple(m) == tuple([0] * (N - k) + [1] * k):
                if k < N:
                    f = tuple(x - (N - k) for x in lam[N - k:]) + tuple(lam[:N - k])
                else:
                    f = tuple(lam)
                return ('E', k, f)
        if asc and all(x in (-1, 0) for x in m):
            k = sum(1 for x in m if x == -1)
            if 1 <= k <= N and tuple(m) == tuple([-1] * k + [0] * (N - k)):
                if k < N:
                    g = tuple(lam[:k]) + tuple(x - (N - k) for x in lam[k:])
                else:
                    g = tuple(lam)
                return ('F', k, g)
        if len(set(m)) == 1 and m[0] != 0 and not any(lam):
            return ('det', m[0])
        return None

    def _clean_tower_rep(self, charge):
        """Representative of the φ-tower of `charge` (minimal-norm λ in
        λ+Z·m) -- the one actually product-built; others are φ-slides."""
        m, lam = charge
        if not any(m):
            return charge
        best = lam
        for n in range(-10, 11):
            cand = tuple(lam[i] - n * m[i] for i in range(self._N))
            if sum(x * x for x in cand) < sum(x * x for x in best):
                best = cand
        return (m, best)

    def _tower_shift(self, charge, rep):
        m = charge[0]
        for n in range(-16, 17):
            if all(charge[1][i] == rep[1][i] + n * m[i] for i in range(self._N)):
                return n
        return None

    def _clean_factors(self, charge):
        """Candidate factor charges `g` (KNOWN closed-form generators) with
        `g + b = charge`, both of strictly lower magnetic rank, ordered by
        COMBINED simplicity of both factors (so e.g. `(0,2;0,0)` is taken as
        `L_{0,1;0,0}²`, not `L_{0,1;0,1}·L_{0,1;0,-1}`).  Analytic: enumerates
        only charge tuples (no image building); each candidate must be a
        recognizable closed-form generator."""
        import itertools
        N = self._N
        m, lam = charge
        R = self._mrank(charge)
        pats = set()
        for k in range(1, N):
            pats.add(tuple([0] * (N - k) + [1] * k))      # E_k box
            pats.add(tuple([-1] * k + [0] * (N - k)))     # F_k box
        for c in range(-R, R + 1):
            if c != 0:
                pats.add(tuple([c] * N))                  # det^c central
        cands = []
        for mg in pats:
            if self._mrank((mg, ())) == 0 or self._mrank((mg, ())) >= R:
                continue
            mb = tuple(m[i] - mg[i] for i in range(N))
            if sum(abs(x) for x in mb) >= R:
                continue
            for lg in itertools.product(range(-2, 3), repeat=N):
                gch = (mg, tuple(lg))
                if self._closed_form_label(gch) is None:
                    continue
                cands.append(gch)
        def _is_central(g):
            return len(set(g[0])) == 1 and g[0][0] != 0 and not any(g[1])
        def _score(g):
            mb = tuple(m[i] - g[0][i] for i in range(N))
            lb = tuple(lam[i] - g[1][i] for i in range(N))
            # A CENTRAL (det^c) factor is preferred above all: det q-commutes, so
            # `det^c·L_base` is a SINGLE canonical -- the peel finds nothing to
            # subtract (the M-test recognizes det·L as a single entry) and the
            # build is one cheap product, no recursion into companions.  This is
            # the det-shift expressed as the cheapest a1·a2 choice, not a peel
            # special-case.  Among centrals, smaller |c| (closer to the base)
            # first.
            central = 0 if _is_central(g) else 1
            return (central, abs(g[0][0]) if central == 0 else 0,
                    self._mrank(g) + sum(abs(x) for x in mb),
                    sum(abs(x) for x in g[1]) + sum(abs(x) for x in lb))
        cands.sort(key=_score)
        return cands[:24]

    def _cone_recenter(self, X):
        """Scale a GENUINE PRODUCT `X` by the q-power that puts its leading
        covariant orbit at deg_W = 0 (the bar-centering).  Used for the base
        ray product `∏ E_k^low/mut · det^p`, whose cocycle is not known a priori
        so its leading deg_W must be read off the M-test.  The cap widens only
        when `deg_W ≤ 0` is empty (a product leading at deg_W > 0) -- this is
        computing a product's leading order, not hunting for a known constant
        (the *free* moves φ/w2 are recentered by closed form, never here).
        None if the M-test stays empty.

        FAST PATH: the only datum needed is `min(deg_W)`, which `_pun_leading_dw`
        computes exactly without the full M-test slice (positive-cost factors can't
        lower the minimum -- see its docstring).  Falls back to the full M-test
        only when the fast path declines (an uncancelled denominator -> None)."""
        N = self._N
        k = _pun_leading_dw(X, N)
        if k is None:
            cov = None
            for cap in (0, 4, 8, 16):
                cov = _pun_LM_le(X, N, cap=cap)
                if cov:
                    break
            if not cov:
                return None
            k = min(dw for (dw, u, v) in cov)
        fac = DOp.from_scalar(VRational.from_scalar(LaurentPoly({-k: 1}), n=N), n=N)
        return _abe_simplify_dop(fac * X)

    def _cone_build(self, charge):
        """Cached `_cone_build_impl` (cone monomials recur heavily in the cone-peel
        of `w_R·cone`, so memoize by charge)."""
        cache = getattr(self, '_cone_cache', None)
        if cache is None:
            cache = self._cone_cache = {}
        charge = (tuple(charge[0]), tuple(charge[1]))
        if charge not in cache:
            cache[charge] = self._cone_build_impl(charge)
        return cache[charge]

    def _cone_build_impl(self, charge):
        """Direct Principal-QTCone construction of `L_{charge}` (NO peel).

        Builds the canonical as a q-commuting product of QTCone RAYS per magnetic
        direction k -- the low ray `low_k = L_{(0^{N-k},1^k),(0^{N-k},(N-k)^k)}`
        and the mut ray `mut_k = φ_{-1}(low_k) = L_{(0^{N-k},1^k),(0^{N-k},(N-k-1)^k)}`
        (dressing lowered by m) -- times `det^p`, then slid to the target dressing
        by the free moves `φ_n` (λ → λ + n·m) and `w2^l = (∏v)^l` (λ → λ + l·1).
        The `(low↔mut)` box gives the `c_k+1` choices per direction (mixed low·mut
        products are the odd-c symmetric cone L's).  All these rays mutually
        q-commute, so EVERY product is a single canonical with self-norm 1 by
        construction (mut squares to a canonical, `mut² = φ_{-1}(low²)`) -- the
        cone is a genuine cone, no gate.  Construct-and-verify against the target
        `_lowest_charge`.  Returns `(L, ρL, ρ⁻¹L)` or None if `charge` is not a
        QTCone monomial (asymmetric / Wilson-dressing charges fall through to the
        `w_R·cone` peel).  ρ-images are built in parallel via the ρ-mapped rays;
        `ρ(w2^l) = w2^{-l}` and `φ_n` is ρ-equivariant."""
        N = self._N
        m, lam = charge
        if list(m) != sorted(m):                      # anti-dominant magnetic only
            return None
        p = m[0]
        c = [m[N - j] - m[N - j - 1] for j in range(1, N)]   # c[j-1] = c_j
        if any(x < 0 for x in c):
            return None
        detr = [('det', 1)] * max(p, 0) + [('det', -1)] * max(-p, 0)
        import itertools
        # Cone monomial = product of the QTCone RAY-L's per magnetic direction --
        # low_k = L_{(0^{N-k},1^k),(0^{N-k},(N-k)^k)} and mut_k = φ_{-1}(low_k) =
        # L_{(0^{N-k},1^k),(0^{N-k},(N-k-1)^k)} -- times det^p, then slid by the free
        # φ_n / w2^l.  These rays all mutually q-commute, so EVERY product is a single
        # canonical (self-norm 1) -- no gate.  The (low↔mut) box gives the c_k+1
        # choices per direction; mixed low·mut products are the odd-c symmetric cone
        # L's.  (mut squares to a canonical: mut² = φ_{-1}(low²) -- the cone is a cone.)
        for bs in itertools.product(*[range(ck + 1) for ck in c]):
            def _prod(mapf):
                D = DOp.one(N)
                for j in range(1, N):
                    ck = c[j - 1]; bk = bs[j - 1]
                    low = _abe_simplify_dop(dop_of(mapf(('E', j, (0,) * N)), N))
                    mut = _abe_simplify_dop(_pun_phi(low, -1, N))  # mut = φ_{-1}(low)
                    for _ in range(ck - bk):
                        D = _abe_simplify_dop(D * low)
                    for _ in range(bk):
                        D = _abe_simplify_dop(D * mut)
                for lb in detr:
                    D = _abe_simplify_dop(D * dop_of(mapf(lb), N))
                return D
            Lbox = self._cone_recenter(_prod(lambda x: x))
            if Lbox is None:
                continue
            lbox = _lowest_charge(Lbox, N)[1]
            diff = tuple(lam[i] - lbox[i] for i in range(N))
            # solve  diff = n·m + l·ones  (φ_n shifts λ by n·m, w2 by ones)
            if len(set(m)) == 1:                       # central m: φ degenerate
                if len(set(diff)) != 1:
                    continue
                n, l = 0, diff[0]
            else:
                i = next(t for t in range(N) if m[t] != m[0])
                dm = m[i] - m[0]
                if (diff[i] - diff[0]) % dm != 0:
                    continue
                n = (diff[i] - diff[0]) // dm
                l = diff[0] - n * m[0]
            if tuple(n * m[t] + l for t in range(N)) != diff:
                continue

            def _finish(D, lpow):
                D = _pun_phi(D, n, N)                   # free move φ_n: q^0
                w = ('W', (1,) * N) if lpow > 0 else ('W', (-1,) * N)
                for _ in range(abs(lpow)):
                    D = _abe_simplify_dop(D * dop_of(w, N))
                # recenter by the M-test leading order (w2 leads at deg_W = ∓Σm, the
                # sign flipping between L and its ρ-image, so read it per image).
                return self._cone_recenter(D)
            L = _finish(_prod(lambda x: x), l)
            if L is None or _lowest_charge(L, N) != charge:
                continue
            rL = _finish(_prod(lambda x: rho_label(x, N)), -l)
            riL = _finish(_prod(lambda x: rho_inverse_label(x, N)), -l)
            if rL is None or riL is None:
                continue
            return L, rL, riL
        return None

    @staticmethod
    def _dop_terms(D):
        """Flatten a DOp to {(u_shift, v_monomial): LaurentPoly(q)}."""
        o = {}
        for u, vr in D.items():
            for vmon, lp in vr.num.items():
                if not lp.is_zero():
                    o[(u, vmon)] = lp
        return o

    def _peel_cone(self, X, m, lo, hi):
        """Subtract EVERY cone monomial at magnetic `m` with dressing in the box
        `[lo, hi]` that appears in `X` -- matched at its own leading chart monomial
        and removed wherever it sits (above OR below the out-of-cone target), so
        the residual is the target orthogonal to all cone.  (Tropical-lowest-first
        would stop at a target that is itself the lowest charge, leaving cone-above
        un-peeled and the target non-orthogonal -- the bug this fixes.)  Cone
        monomials are built on demand via the cached `_cone_build`.  Returns
        `(residual, subtractions=[(cone_charge, C_q)])`, replayed on ρ/ρ⁻¹."""
        import itertools
        N = self._N
        conemons = []
        for lp in itertools.product(*[range(lo[i], hi[i] + 1) for i in range(N)]):
            cb = self._cone_build((m, lp))
            if cb is not None and _lowest_charge(cb[0], N) == (m, lp):
                conemons.append(((m, lp), cb[0], min(self._dop_terms(cb[0]))))
        subs = []
        R = X
        changed = True
        while changed:
            changed = False
            t = self._dop_terms(R)
            for cch, L, lk in conemons:
                if lk not in t:
                    continue
                ea = sorted(t[lk]._coeffs.items()); eb = sorted(self._dop_terms(L)[lk]._coeffs.items())
                Cq = LaurentPoly({ea[0][0] - eb[0][0]: ea[0][1] // eb[0][1]})
                R = _abe_simplify_dop(
                    R - DOp.from_scalar(VRational.from_scalar(Cq, n=N), n=N) * L)
                subs.append((cch, Cq)); changed = True
                t = self._dop_terms(R)
        return R, subs

    def _replay_peel(self, X, subs, side):
        """Subtract `Σ C_q · (side-image of L_cone)` from `X` -- the ρ (side=1) or
        ρ⁻¹ (side=2) replay of a forward `_peel_cone` (same C_q)."""
        N = self._N
        R = X
        for (cch, Cq) in subs:
            Lc = self._cone_build(cch)[side]
            R = _abe_simplify_dop(
                R - DOp.from_scalar(VRational.from_scalar(Cq, n=N), n=N) * Lc)
        return R

    def _qtcone_recipe(self, m, lam_c):
        """Closed-form WHICH-QTcone + HOW for dressing `lam_c` at anti-dominant `m`.

        Returns `(b, n, l, p)` -- the cone is built as the q-commuting ray product
        `Π_j (c_j−b_j)·low_j · b_j·mut_j · det^p` slid by the free moves `φ_n`
        (λ→λ+n·m) and `w2^l` (λ→λ+l·𝟙) -- or None if `lam_c` is not a principal-
        QTcone monomial.  This REPLACES the per-R `_cone_build` PROBE in the
        `w_R·cone` sweep: one O(N) linear solve per candidate instead of a full
        cone construction (the ~1500-probe → ~1 fix).  A genuine cone always passes
        (0 false-negatives vs `_cone_build`, verified N=2,3); the few false-positives
        fail harmlessly when actually built.

        Derivation: a cone monomial has dressing
        `λ_cone − Σ_j b_j ϖ_j + n·m + l·𝟙` (low↔mut box `0≤b_j≤c_j`; φ_n / w2^l free).
        With `x = lam_c − λ_cone`, its ϖ_j coordinate is the consecutive difference
        `a_j = x[N−j]−x[N−j−1]` and its 𝟙 coordinate is `x[0]`; matching forces
        `b_j = n·c_j − a_j` (so `0≤b_j≤c_j`) for a SINGLE shared `n`, and
        `l = x[0] − n·p`.  Inactive directions (`c_j=0`) require `a_j=0`."""
        N = self._N
        if list(m) != sorted(m):
            return None
        ref = self._cone_ref(tuple(m))
        if ref is None:
            return None
        lam_cone = ref[1]
        p = m[0]
        c = [m[N - j] - m[N - j - 1] for j in range(1, N)]    # c[j-1] = c_j
        x = [lam_c[i] - lam_cone[i] for i in range(N)]
        a = [x[N - j] - x[N - j - 1] for j in range(1, N)]    # a[j-1] = a_j (ϖ_j coord)
        for j in range(N - 1):
            if c[j] == 0 and a[j] != 0:                       # inactive: must be 0
                return None
        if all(c[j] == 0 for j in range(N - 1)):              # central m: φ degenerate
            return ([0] * (N - 1), 0, x[0], p) if len(set(x)) == 1 else None
        lo, hi = -10**9, 10**9                                # shared-n window
        for j in range(N - 1):
            if c[j] > 0:
                lo = max(lo, -(-a[j] // c[j]))                # ceil(a_j / c_j)
                hi = min(hi, (a[j] + c[j]) // c[j])           # floor(a_j / c_j + 1)
        wp = [tuple(1 if t >= N - j else 0 for t in range(N)) for j in range(1, N)]
        for n in range(lo, hi + 1):
            b = [(n * c[j] - a[j]) if c[j] > 0 else 0 for j in range(N - 1)]
            l = x[0] - n * p
            lbox = [lam_cone[i] - sum(b[j] * wp[j][i] for j in range(N - 1))
                    for i in range(N)]
            if tuple(lbox[i] + n * m[i] + l for i in range(N)) == tuple(lam_c):
                return (b, n, l, p)
        return None

    def _wR_cone_build(self, charge, _depth=0):
        """Build a non-cone canonical `L_{(m,λ)}` by the `W_R·(lower)` M-TEST peel
        (the simultaneous bar-symmetric peel, `_peel_to`).  Two factorization
        families, both `W_R · L_{(m,λ')}` with the target a summand and every other
        summand a closer-to-cone `L'` peeled in bar-pairs `q^{±n}`:

          (1) a dominant Wilson `R` with `λ−R` a CONE dressing (R adds boxes onto a
              cone), and
          (2) a single Wilson BOX — fundamental `e_i` or antifundamental `−e_i`
              (your `W_{-1}·cone`) — onto a strictly-closer-to-cone canonical
              `L_{(m, λ−box)}`, built recursively (the L1 distance to the bare-cone
              dressing strictly decreases, so the recursion bottoms out on cones).

        The bar-fixed canonical is then the symmetric-centered residual.  Returns
        `(L, ρL, ρ⁻¹L)` or None.  NO tropical peel, NO one-sided recenter."""
        import itertools
        N = self._N
        m, lam = charge
        ref = self._cone_ref(m)
        lam_cone = ref[1] if ref is not None else None

        def cplx(l):
            return (sum(abs(l[i] - lam_cone[i]) for i in range(N))
                    if lam_cone is not None else None)
        c_lam = cplx(lam)
        facs = []                       # (Rch, bch) factorizations to try
        # (1) dominant R . cone.  Pick R by the CLOSED-FORM cone-membership test
        # (_qtcone_recipe, O(N)) instead of building a cone for every R -- the
        # ~1500-probe → ~1 fix.  The recipe never misses a buildable cone (0 false-
        # negatives vs _cone_build, N=2,3); a false-positive R is harmless (its
        # _build_clean below just fails and the next R is tried).
        rng = range(0, max(2, max(lam) - min(lam) + 2))
        for R in sorted((r for r in itertools.product(rng, repeat=N)
                         if list(r) == sorted(r, reverse=True) and any(r)),
                        key=lambda r: (sum(r), r)):
            lam_c = tuple(lam[i] - R[i] for i in range(N))
            if self._qtcone_recipe(m, lam_c) is not None:
                facs.append((((0,) * N, R), (m, lam_c)))
        # (2) single Wilson box toward the cone (fundamental / antifundamental)
        fund_hw = tuple([1] + [0] * (N - 1))
        anti_hw = tuple([0] * (N - 1) + [-1])
        boxes = ([tuple(1 if t == i else 0 for t in range(N)) for i in range(N)]
                 + [tuple(-1 if t == i else 0 for t in range(N)) for i in range(N)])
        box_facs = []
        for w in boxes:
            lam_p = tuple(lam[i] - w[i] for i in range(N))
            cp = cplx(lam_p)
            if c_lam is None or cp is None or cp >= c_lam:
                continue
            Rhw = fund_hw if sum(w) > 0 else anti_hw
            box_facs.append((cp, ((0,) * N, Rhw), (m, lam_p)))
        box_facs.sort(key=lambda t: t[0])            # closest-to-cone factor first
        facs += [(rc, bc) for _c, rc, bc in box_facs]

        for Rch, bch in facs:
            if bch[1] == lam:
                continue
            if not self._build_clean(Rch, _depth + 1):
                continue
            if not self._build_clean(bch, _depth + 1):
                continue
            La, rLa, riLa = (self.image(Rch), self.rho_image(Rch),
                             self.rho_inverse_image(Rch))
            Lb, rLb, riLb = (self.image(bch), self.rho_image(bch),
                             self.rho_inverse_image(bch))
            P = _abe_simplify_dop(La * Lb)
            Q = _abe_simplify_dop(Lb * La)
            rP = _abe_simplify_dop(rLa * rLb)
            rQ = _abe_simplify_dop(rLb * rLa)
            riP = _abe_simplify_dop(riLa * riLb)
            riQ = _abe_simplify_dop(riLb * riLa)
            res = self._peel_to(charge, P, Q, rP, rQ, riP, riQ, _depth)
            if res is not None:
                return res
        return None

    def _build_clean(self, charge, _depth=0):
        """Recursive two-canonical-product construction of `L_{charge}`."""
        N = self._N
        charge = (tuple(charge[0]), tuple(charge[1]))
        if self.has(charge):
            return True
        if charge == ((0,) * N, (0,) * N):
            self.ensure_identity()
            return True
        building = getattr(self, "_building", None)
        if building is None:
            building = self._building = set()
        if charge in building:                     # cycle guard -> this path fails
            return False
        m, lam = charge
        # Fail-fast: a valid canonical's dressing is a DOMINANT Levi irrep, i.e.
        # weakly DESCENDING within each equal-m block.  A label that INCREASES
        # inside a block (e.g. L_{(0,0,2),(0,2,0)}: the m=0 block reads (0,2)) is
        # not constructible -- reject it in O(N) instead of grinding the whole
        # cone / w_R / product-peel ladder before raising.  (Verified: 0 built
        # canonicals violate this, N=2,3.)
        for i in range(1, N):
            if m[i] == m[i - 1] and lam[i] > lam[i - 1]:
                return False
        # base case: a KNOWN closed-form generator (no enumeration / no product).
        # SAFE closed forms used for building: W (m=0, Wilson character) and bare
        # det^c (central) -- both honest (no fingerprint↔label confusion).  The
        # dressed NON-CONE monopoles E_k(f)/F_k(g) are NOT used for building by
        # default: their `_closed_form_label` (m,λ)->generator map is a separate,
        # partly-broken code route (the F-sector λ↔g dictionary is wrong -- L_F is
        # built correctly, but mapped to the wrong charge), so dressed monopoles are
        # built through the verified cone / Wilson·cone bar-symmetric peel instead.
        # Set `self._use_closed_ef = True` to restore the E/F build shortcut for
        # speed once it is validated for the sector in use (it is still guarded by
        # the `_lowest_charge` re-check, which rejects every mis-mapped label).
        lbl = self._closed_form_label(charge)
        if lbl is not None and (lbl[0] in ('W', 'det')
                                or getattr(self, "_use_closed_ef", False)):
            L = _abe_simplify_dop(dop_of(lbl, N))
            if _lowest_charge(L, N) == charge:          # re-verify the analytic map
                self.add_image(charge, L,
                               _abe_simplify_dop(dop_of(rho_label(lbl, N), N)),
                               _abe_simplify_dop(dop_of(rho_inverse_label(lbl, N), N)))
                return True
        # Principal-QTCone fast path: build cone monomials (+ φ/w2/(low↔mut)
        # translates) directly as q-commuting ray products — no peel, no recursion.
        # Non-cone charges (asymmetric Levi / Wilson dressings) return None here
        # and fall through to the product peel below.
        cone = self._cone_build(charge)
        if cone is not None:
            self.add_image(charge, *cone)
            return True
        if _depth > 64:
            return False
        building.add(charge)
        try:
            # φ-tower reduction FIRST: a non-representative dressing is a FREE φ-slide
            # of its tower rep (min |λ mod m|).  Build the rep (which may itself use
            # the w_R·cone peel) and slide -- so φ-translates like (2,0,4)=φ_2·(2,0,0)
            # are never rebuilt the expensive way.
            rep = self._clean_tower_rep(charge)
            if rep != charge and any(m) and self._build_clean(rep, _depth + 1):
                n = self._tower_shift(charge, rep)
                if n is not None:
                    self.add_image(charge,
                        _abe_simplify_dop(_pun_phi(self.image(rep), n, N)),
                        _abe_simplify_dop(_pun_phi(self.rho_image(rep), n, N)),
                        _abe_simplify_dop(_pun_phi(self.rho_inverse_image(rep), n, N)))
                    return True
            # w_R·cone peel: build the tower REPRESENTATIVE of a non-cone (asymmetric-
            # Levi / Wilson) dressing as a single Wilson·cone product peeled against
            # the cone (one out-of-cone piece).
            wrc = self._wR_cone_build(charge, _depth)
            if wrc is not None:
                self.add_image(charge, *wrc)
                return True
            # two-canonical product peel (general fallback): factor charge = gch + bch
            # with both of strictly lower magnetic rank.
            for gch in self._clean_factors(charge):
                bch = (tuple(m[i] - gch[0][i] for i in range(N)),
                       tuple(lam[i] - gch[1][i] for i in range(N)))
                if self._mrank(bch) >= self._mrank(charge):
                    continue
                if self._extract_pair(charge, gch, bch, _depth):
                    return True
            return False
        finally:
            building.discard(charge)

    def _extract_pair(self, charge, gch, bch, _depth):
        """Build the two factor canonicals and extract `L_charge` from the
        products `L_gch·L_bch` and `L_bch·L_gch` by the most-negative-orbit peel
        (`_peel_to`).  Registers `charge` if it succeeds; returns whether it did."""
        if not self._build_clean(gch, _depth + 1):
            return False
        if not self._build_clean(bch, _depth + 1):
            return False
        N = self._N
        La, rLa, riLa = self.image(gch), self.rho_image(gch), self.rho_inverse_image(gch)
        Lb, rLb, riLb = self.image(bch), self.rho_image(bch), self.rho_inverse_image(bch)
        P = _abe_simplify_dop(La * Lb)
        Q = _abe_simplify_dop(Lb * La)
        rP = _abe_simplify_dop(rLa * rLb)
        rQ = _abe_simplify_dop(rLb * rLa)
        riP = _abe_simplify_dop(riLa * riLb)
        riQ = _abe_simplify_dop(riLb * riLa)
        res = self._peel_to(charge, P, Q, rP, rQ, riP, riQ, _depth)
        if res is not None:
            self.add_image(charge, *res)
            return True
        return False

    @staticmethod
    def _lam_bare(m):
        """Bare-monopole dressing `λ_bare(m)_i = #{j : m_j < m_i}` (the rank of
        each magnetic component) -- the `m`-dependent shift in the orbit→label
        rule.  (E_k → (N−k) on the magnetic block, det → 0; verified N=2,3,4.)"""
        return tuple(sum(1 for x in m if x < mi) for mi in m)

    def _cov_orbit(self, D):
        """Weyl-COVARIANT M-test of `D`: my M-test with `v → v+ρ_inc` and the
        global sign `(-1)^{N(N-1)/2}`, i.e. against `M_weyl = Vandermonde·∏_{i≠j}
        (q²v_i/v_j;q²)_∞` instead of the symmetric half-measure.  In this frame
        each canonical's q^0 slice is a clean signed Weyl orbit, so the label λ is
        readable.  Returns `{(dw, u, v): int}`."""
        N = self._N
        rho = tuple(range(N))
        sgn = (-1) ** (N * (N - 1) // 2)
        return {(dw, u, tuple(v[t] + rho[t] for t in range(N))): sgn * c
                for (dw, u, v), c in _pun_LM_le(D, N, cap=0).items()}

    def _cone_ref(self, m):
        """Reference `(v_ref, λ_ref)` for magnetic `m`: the leading-Weyl-orbit
        dominant `v` and the dressing of the BARE cone monomial at `m` (the
        q-commuting product of low rays · det^p).  Used to read a slice's dressing
        by the EXACT linear relation `λ = λ_ref + (v_dom − v_ref)` (the λ↔v_dom map
        is affine at fixed `m`), which is correct for non-minuscule `m` too --
        unlike the rank-`λ_bare` formula.  Cached per `m`; None if `m` not
        anti-dominant / not a valid cone magnetic."""
        cache = getattr(self, "_cone_ref_cache", None)
        if cache is None:
            cache = self._cone_ref_cache = {}
        if m in cache:
            return cache[m]
        N = self._N
        res = None
        if list(m) == sorted(m):
            p = m[0]
            c = [m[N - j] - m[N - j - 1] for j in range(1, N)]
            if all(x >= 0 for x in c):
                rays = []
                for j in range(1, N):
                    rays += [('E', j, (0,) * N)] * c[j - 1]
                rays += [('det', 1)] * max(p, 0) + [('det', -1)] * max(-p, 0)
                D = DOp.one(N)
                for lb in rays:
                    D = _abe_simplify_dop(D * dop_of(lb, N))
                L = self._cone_recenter(D)
                if L is not None:
                    orb = {(u, v): cc for (dw, u, v), cc in self._cov_orbit(L).items()
                           if dw == 0}
                    if orb:
                        u_bot = min((u for (u, v) in orb), key=_abe_u_order)
                        v_ref = max(v for (u, v) in orb if u == u_bot)
                        res = (v_ref, _lowest_charge(L, N)[1])
        cache[m] = res
        return res

    def _cov_label(self, sl):
        """`(label, lead_pt)` of a covariant signed-Weyl-orbit slice `{(u,v): int}`:
        `m` = anti-dominant `u`, `v_dom` the dominant `v` there, and the dressing by
        the EXACT affine map `λ = λ_ref + (v_dom − v_ref)` against the bare-cone
        reference at `m` (`_cone_ref`).  This is correct for non-minuscule `m`
        (where the old `v_dom − ρ_dec + λ_bare` mislabels, e.g. it called the
        `(0,0,6)` orbit `(0,0,4)`).  Falls back to the rank formula if no cone ref."""
        N = self._N
        u_bot = min((u for (u, v) in sl), key=_abe_u_order)
        v_dom = max(v for (u, v) in sl if u == u_bot)
        ref = self._cone_ref(u_bot)
        if ref is not None:
            v_ref, lam_ref = ref
            lam = tuple(lam_ref[t] + (v_dom[t] - v_ref[t]) for t in range(N))
        else:
            rho_dec = tuple(range(N - 1, -1, -1))
            lb = self._lam_bare(u_bot)
            lam = tuple(v_dom[t] - rho_dec[t] + lb[t] for t in range(N))
        return (u_bot, lam), (u_bot, v_dom)

    def _mtest_mostneg(self, D):
        """`(nd, slice, label, lead_pt)` of the most-negative covariant M-test
        slice of `D` -- power, covariant signed orbit, the canonical label it is,
        and the leading point.  None if the M-test has no q^{<=0} content."""
        X = self._cov_orbit(D)
        if not X:
            return None
        nd = min(dw for (dw, u, v) in X)
        sl = {(u, v): c for (dw, u, v), c in X.items() if dw == nd}
        label, lead = self._cov_label(sl)
        return nd, sl, label, lead

    @staticmethod
    def _cf_magnetic(m):
        """Is `m` a CLOSED-FORM (minuscule/central) magnetic sector -- a single-
        box E_k/F_k pattern (anti-dominant, entries in {0,1} or {0,-1}) or a
        central c·(1,..,1)?  Only for these is the orbit→label rule valid; other
        m are non-minuscule (products), whose label is not read from the orbit."""
        asc = list(m) == sorted(m)
        if asc and (all(x in (0, 1) for x in m) or all(x in (0, -1) for x in m)):
            return True
        return len(set(m)) == 1

    def _orbit_scalar(self, sl, charge, lead):
        """If covariant slice `sl` is a scalar multiple of built canonical
        `charge`'s covariant q^0 orbit, return that integer scalar, else None."""
        orb = {(u, v): c for (dw, u, v), c in self._cov_orbit(self.image(charge)).items()
               if dw == 0}
        if lead not in orb or orb[lead] == 0 or lead not in sl or set(orb) != set(sl):
            return None
        s, r = divmod(sl[lead], orb[lead])
        if r != 0 or s == 0 or any(sl[k] != s * orb[k] for k in orb):
            return None
        return s

    def _identify_lower(self, sl, lead, target, _depth):
        """Identify the canonical of a most-negative covariant slice as a
        strictly-lower, peelable `L'`, LABELLED BY ITS FINGERPRINT (the signed
        Weyl orbit the M-test returns).

        Two routes, cheap first:
          (1) match an ALREADY-BUILT canonical (any `m`, incl. non-minuscule);
          (2) read the slice's `(m, λ)` via the orbit→label rule and BUILD it
              recursively.  The fingerprint *is* the label, so a non-minuscule
              (det-content) companion sharing the target's `m` -- the `q·L'` the
              product carries alongside the target -- is built on demand here,
              not skipped (the old `_cf_magnetic` gate dropped exactly these and
              left the contamination in, the N=2 machinery bug).

        Termination: the recursive build (2) goes through `_build_clean`, whose
        `_building` cycle guard returns False if `(m,λ)` is already on the build
        stack -- so a circular descent (target→L'→…→target, or any L' "defined
        by a product with the same property") fails-safe rather than looping;
        the `_depth>64` cap is the backstop.  We never recurse on the target
        itself or on a charge mid-build.

        The `_orbit_scalar` re-check guards a possible orbit→label mis-read: if
        the built canonical's `q^0` orbit does not match the slice, it is not
        peeled (that factorization then simply fails and a cleaner one is tried).
        Returns `(charge', scalar)` or None (slice is the target / no lower L')."""
        m_lab = min((u for (u, v) in sl), key=_abe_u_order)
        for lab in self._reg_labels():           # (1) already-built (incl non-minuscule)
            if lab == target or lab[0] != m_lab:
                continue
            s = self._orbit_scalar(sl, lab, lead)
            if s is not None:
                return lab, s
        (m, lam), _lead = self._cov_label(sl)     # (2) build the fingerprint on demand
        building = getattr(self, "_building", None) or set()
        if (m, lam) != target and (m, lam) not in building and not self.has((m, lam)):
            if self._build_clean((m, lam), _depth + 1):
                s = self._orbit_scalar(sl, (m, lam), lead)
                if s is not None:
                    return (m, lam), s
        return None

    def _peel_to(self, charge, P, Q, rP, rQ, riP, riQ, _depth):
        """The canonical construction (user's algorithm).  M-test(L_a L_b) =
        Σ_d C^d_{ab}(q)·(WeylOrbit_d + O(q)); M-test(L_b L_a) the same with
        q→1/q.  The target appears at q^{⟨a,b⟩} (NOT q^0), so it is never the
        most-negative power in BOTH orderings -- the most-negative slice in at
        least one ordering is a strictly-lower, recursively-buildable L'.
        Identify L' (lowest-tropical point of that signed Weyl orbit), build it,
        and subtract the multiple that CANCELS that leading M-test term -- from
        the ordering it was found in at q^{nd}, and (bar) from the other at
        q^{-nd} (plus the ρ/ρ⁻¹ images).  Repeat until the only thing either
        M-test reveals is the target; the residual is then q^k·L_target --
        recenter (so M-test(L) = WeylOrbit + O(q)) and return."""
        N = self._N
        for _it in range(500):
            mp = self._mtest_mostneg(P)
            mq = self._mtest_mostneg(Q)
            # find a most-negative slice that is a peelable LOWER L' (a closed-form
            # minuscule via the rule, or an already-built canonical via match);
            # the target is non-minuscule-and-unbuilt, so it is NOT identifiable
            # here -- when neither ordering's most-negative is a lower L', only the
            # target remains.
            pick = None
            for (mm, fwd) in ((mp, True), (mq, False)):
                if mm is None:
                    continue
                lid = self._identify_lower(mm[1], mm[3], charge, _depth)
                if lid is not None:
                    pick = (mm[0], mm[1], lid[0], lid[1], fwd)
                    break
            if pick is None:
                break
            nd, sl, cpr, scalar, fwd = pick
            Lc = self.image(cpr); rLc = self.rho_image(cpr); riLc = self.rho_inverse_image(cpr)
            fp = DOp.from_scalar(VRational.from_scalar(LaurentPoly({nd: scalar}), n=N), n=N)
            fm = DOp.from_scalar(VRational.from_scalar(LaurentPoly({-nd: scalar}), n=N), n=N)
            here = (fp, fm) if fwd else (fm, fp)     # (P-side power, Q-side power)
            P = _abe_simplify_dop(P - here[0] * Lc)
            rP = _abe_simplify_dop(rP - here[0] * rLc)
            riP = _abe_simplify_dop(riP - here[0] * riLc)
            Q = _abe_simplify_dop(Q - here[1] * Lc)
            rQ = _abe_simplify_dop(rQ - here[1] * rLc)
            riQ = _abe_simplify_dop(riQ - here[1] * riLc)
        else:
            return None
        # Only the target remains.  Bar-symmetric extraction (the DEFINITION of the
        # canonical): the target sits at `q^{p}·L` in `P` and `q^{-p}·L` in `Q` for
        # the SAME bar-fixed object `L`.  So `P == q^{2p}·Q` exactly; the symmetric
        # centering `q^{-p}` produces the unique bar-fixed `L` (`bar(L)=L`).  This
        # replaces the old one-sided `min-dw → q⁰` recenter, which only fixed the
        # leading power and left the higher-q corrections un-bar-fixed (orthonormal
        # and centered, yet `bar(L)≠L`).  If `P` and `Q` are NOT q-proportional, a
        # peeled lower term was itself not bar-fixed -> fail (it must be built
        # correctly first); the recursion bottoms out on the bar-fixed cones.
        if not P._terms or not Q._terms:
            return None
        e = self._monomial_ratio(P, Q)      # integer with P == q^e · Q, else None
        if e is None or e % 2 != 0:
            return None
        p = e // 2
        fac = DOp.from_scalar(VRational.from_scalar(LaurentPoly({-p: 1}), n=N), n=N)
        L = _abe_simplify_dop(fac * P)
        rL = _abe_simplify_dop(fac * rP)
        riL = _abe_simplify_dop(fac * riP)
        if _lowest_charge(L, N) != charge:
            return None
        return (L, rL, riL)

    def _monomial_ratio(self, P, Q):
        """Return the integer `e` with `P == 𝖖^e · Q` exactly (as DOps), else None.
        Used by the bar-symmetric extraction: when only the target remains in both
        orderings it is `𝖖^{±p}·L` for one bar-fixed `L`, so `P` and `Q` are
        q-proportional and `e = 2p`."""
        N = self._N
        Ps = _abe_simplify_dop(P)
        Qs = _abe_simplify_dop(Q)
        for u in sorted(Qs._terms):
            qnum = Qs._terms[u].simplify().num
            for v in sorted(qnum._terms):
                qc = qnum._terms[v]
                if qc.is_zero():
                    continue
                if u not in Ps._terms:
                    return None
                pnum = Ps._terms[u].simplify().num
                if v not in pnum._terms or pnum._terms[v].is_zero():
                    return None
                e = min(pnum._terms[v]._coeffs) - min(qc._coeffs)
                qe = DOp.from_scalar(
                    VRational.from_scalar(LaurentPoly({e: 1}), n=N), n=N)
                if not _abe_simplify_dop(Ps - qe * Qs)._terms:
                    return e
                return None
        return None

    def ensure(self, charge):
        """Ensure `L_{charge}` is registered, built on demand.

        LIVE build path: the closed-form engine (`cf_build_engine` for the image,
        with ρ/ρ⁻¹ from the exact √measure conjugation `_rho_chart`/`_rho_inv_chart`),
        gated on the built image re-reading its own `_lowest_charge` so a wrong
        canonical is never registered.

        Rare-case fallback: the closed-form `W_R·cone` peel can leave a norm>1
        residue when a summand is hidden in the bubbling of another; there
        `_ensure_via_chart` declines (returns False) and we fall back to the M-test
        two-canonical-product peel `_build_clean` (the covariant peel resolves the
        bubbling-hidden summand).  The two are complementary, so the fallback does
        not hit the dressed-monopole hang.  If neither builds it, raise."""
        if charge is None or self.has(charge):
            return self
        charge = _weyl_antidom(charge, self._N)      # Weyl-normalize (m,e) to anti-dom m
        # O(N) fail-fast: a build-key dressing that INCREASES inside an equal-m
        # block is never a dominant Levi irrep, so it labels no canonical -- reject
        # immediately instead of grinding either engine.  (No built canonical
        # violates this; test_pure_un_cone_membership.test_built_canonicals_are_block_dominant.)
        m, lam = charge
        for i in range(1, self._N):
            if m[i] == m[i - 1] and lam[i] > lam[i - 1]:
                raise NotImplementedError(
                    f"pure_un: L_{charge} is not a canonical -- dressing increases "
                    f"inside the equal-m block at {i-1},{i} (not Levi-dominant).")
        if self._ensure_via_chart(charge):          # closed-form (primary)
            return self
        if self._build_clean(charge):               # M-test (rare bubbling-hidden)
            return self
        raise NotImplementedError(
            f"pure_un: L_{charge} is not constructible (N={self._N}) by the "
            f"closed-form engine nor the M-test fallback.")

    def _ensure_via_chart(self, charge):
        """Build `L_charge` via the closed-form `cf_build_engine` and register it
        with ρ/ρ⁻¹ from the exact chart conjugation.  Returns whether it did
        (False if out of the closed-form scope or the image mis-reads its charge)."""
        N = self._N
        try:
            from pure_un_closed_form import cf_build_engine
            L = cf_build_engine(self, charge)
        except (NotImplementedError, RuntimeError, ValueError):
            return False
        if L is None:
            return False
        L = _abe_simplify_dop(L)
        if _lowest_charge(L, N) != charge:             # re-verify the build target
            return False
        self.add_image(charge, L, _rho_chart(L, N), _rho_inv_chart(L, N))
        return True

    def has(self, label) -> bool:
        return _weyl_antidom(label, self._N) in self._idx

    # ----- multiply: decompose the chart product onto the basis -----

    def _coeff_at(self, L: DOp, charge):
        """q-Laurent coefficient of `L` at the `(u, v)` monomial `charge`."""
        Ls = _abe_simplify_dop(L)
        u, v = charge
        if u not in Ls._terms:
            return LaurentPoly.zero()
        num = Ls._terms[u].simplify().num
        return num._terms.get(v, LaurentPoly.zero())

    def decompose(self, P: DOp, max_terms: int = 4000):
        """Express a chart element `P` in the canonical basis.

        Returns `{label: C(q)}` with `C(q) ∈ Z[q^±]` exact, by peeling the
        **lowest tropical monomial** repeatedly: it pins the canonical
        `L_c` (basis is keyed by lowest charge), `C(q) = [P]_{charge} /
        [L_c]_{charge}`, then subtract `C(q)·L_c`.  Exact linear algebra in
        the chart (no `O(q)` truncation)."""
        out = {}
        P = _abe_simplify_dop(P)
        for _ in range(max_terms):
            charge = _lowest_charge(P, self._N)
            if charge is None:
                return out
            if charge not in self._idx:
                self.ensure(charge)            # grow the registry on demand
            Lc = self.image(charge)
            cP = self._coeff_at(P, charge)
            cL = self._coeff_at(Lc, charge)        # leading coeff of the centered L_c
            lc = list(cL._coeffs.items())
            if len(lc) != 1:
                raise RuntimeError(f"decompose: L_c leading not a q-monomial: {cL}")
            (eb, vb), = lc
            Cq = LaurentPoly({e - eb: (v if vb == 1 else v // vb)
                              for e, v in cP._coeffs.items()})
            out[charge] = Cq
            sub = DOp.from_scalar(VRational.from_scalar(Cq, n=self._N), n=self._N) * Lc
            P = _abe_simplify_dop(P + DOp.from_scalar(
                VRational.from_scalar(LaurentPoly({0: -1}), n=self._N), n=self._N) * sub)
        raise RuntimeError("decompose: did not terminate")

    def _norm_mtest(self, D, cap):
        """Normalized covariant M-test of `D` (signed Weyl orbits), `deg_W ≤ cap`:
        `{(deg_W, u, v): int}` with `v` shifted by ρ and the global sign, so each
        canonical's leading orbit sits at `deg_W = 0`."""
        N = self._N
        rho = tuple(range(N))
        sgn = (-1) ** (N * (N - 1) // 2)
        return {(dw, u, tuple(v[t] + rho[t] for t in range(N))): sgn * c
                for (dw, u, v), c in _pun_LM_le(D, N, cap=cap).items()}

    def _decompose_one(self, P, KDW=6):
        """One-ordering M-test decomposition: read the `deg_W ≤ 0` leading Weyl
        orbits of `P` and peel them against the canonical basis (built on demand
        via the fixed `cov_label` identification), returning `{c: C^c(q)}` for the
        summands with `k_c ≤ 0`.  This is the Weyl-covariant peel — NOT the tropical
        chart-monomial peel."""
        N = self._N
        X = self._norm_mtest(P, 0)
        out = {}
        guard = 0
        while X:
            guard += 1
            if guard > 20000:
                raise RuntimeError("decompose_one: peel did not terminate")
            nd = min(dw for (dw, u, v) in X)
            sl = {(u, v): c for (dw, u, v), c in X.items() if dw == nd}
            (mc, lamc), lead = self._cov_label(sl)
            if not self._build_clean((mc, lamc)):
                raise RuntimeError(f"decompose_one: cannot build L_{(mc, lamc)}")
            sig = self._norm_mtest(self.image((mc, lamc)), KDW)
            ac = sig[(0,) + lead]
            mult = sl[lead] // ac
            out[(mc, lamc)] = out.get((mc, lamc), LaurentPoly.zero()) + LaurentPoly({nd: mult})
            for (dw, u, v), c in sig.items():
                key = (dw + nd, u, v)
                X[key] = X.get(key, 0) - mult * c
                if X.get(key, 0) == 0 and key in X:
                    del X[key]
            X = {k: v for k, v in X.items() if v and k[0] <= 0}
        return out

    # ---- fingerprint-head identification (for the double M-test peel) ----

    def _fp_orbit0(self, charge):
        """The `deg_W = 0` covariant signed-Weyl orbit (`{(u,v): int}`) of a
        canonical -- its fingerprint head-orbit (leading order, normalized)."""
        return {k[1:]: c for k, c in self._norm_mtest(self.image(charge), 0).items()
                if k[0] == 0}

    def _head_record(self, charge):
        """Index a built canonical by its lex-max fingerprint head and collect a
        per-`m` calibration point `(λ, head_v)` for the affine head→λ map."""
        o0 = self._fp_orbit0(charge)
        if not o0:
            return
        head = max(o0)
        self._head_idx[head] = charge
        self._cal_pts.setdefault(charge[0], []).append((charge[1], head[1]))

    def _ensure_head_index(self):
        if getattr(self, "_head_idx", None) is None:
            self._head_idx = {}
            self._cal_pts = {}
            self._cal = {}
            for c in list(self._reg_labels()):
                self._head_record(c)

    def _calibrate_m(self, m):
        """Fit the integer affine map `head_v = M·λ + b` for magnetic `m` from
        registered calibration points (≥ N+1 affinely-independent), and return
        `(M, b, M⁻¹)` (or None).  Inverting it recovers `λ` from a fingerprint
        head, so out-of-scope summands are identified and built on demand."""
        if m in self._cal:
            return self._cal[m]
        res = _fit_affine_map(self._cal_pts.get(m, []), self._N)
        self._cal[m] = res
        return res

    def _cone_head_map(self, m, R):
        """Learned `{fingerprint-head: λ}` for the CONE monomials at magnetic `m`,
        built once and cached (rebuilt only if a larger reach `R` is needed).  The
        cone dressings are the free-move tower `λ = λ_cone + n·m + l·1` (`|n|,|l|≤R`);
        each cone monomial is built via the cheap `_cone_build` and indexed by its
        M-test fingerprint head.  Lets `_ensure_head` identify a cone summand by an
        O(1) head lookup instead of a per-call sweep."""
        N = self._N
        cache = getattr(self, "_cone_head_cache", None)
        if cache is None:
            cache = self._cone_head_cache = {}
        prev = cache.get(m)
        if prev is not None and prev[0] >= R:
            return prev[1]
        out = {}
        ref = self._cone_ref(m)
        if ref is not None:
            lam_cone = ref[1]
            seen = set()
            for n in range(-R, R + 1):
                for l in range(-R, R + 1):
                    lam = tuple(lam_cone[t] + n * m[t] + l for t in range(N))
                    if lam in seen:
                        continue
                    seen.add(lam)
                    cone = self._cone_build((m, lam))
                    if cone is None:
                        continue
                    o0 = {k[1:]: c for k, c in self._norm_mtest(cone[0], 0).items()
                          if k[0] == 0}
                    if o0:
                        out.setdefault(max(o0), lam)
        cache[m] = (R, out)
        return out

    def _ensure_head(self, sl):
        """The canonical whose most-negative covariant orbit is the slice `sl`,
        built on demand.  The lex-max point `head = max(sl)` is the PURE anchor
        (the canonical's highest weight, shared with no other present canonical).
        The PRIMARY candidate is the CLOSED-FORM head→λ inversion
        `λ = M⁻¹·(head_v − w0m·ρ) + δ(m)` (`_pun_head_to_lambda`); further
        candidates are gathered from (1) `_cov_label(sl)` (the full orbit pins
        the anti-dominant `m`), (2) the per-`m` affine head→λ fit, (3)
        det-translates of registered heads (`det^p·L_c`), and (4) central λ=0.
        Each is built and ACCEPTED only if its own fingerprint head matches
        `head` exactly (self-correcting), so the closed form can never return a
        wrong label -- a frame surprise just falls through to the fallbacks."""
        self._ensure_head_index()
        head = max(sl)
        if head in self._head_idx:
            return self._head_idx[head]
        N = self._N
        u_star, v_star = head
        m = tuple(sorted(u_star))
        # PRIMARY: closed-form head -> λ inversion (the #363 follow-up that
        # supersedes the learned index below).  The head→λ map is exactly affine
        # at fixed m: λ = M⁻¹·(v_star − w0m·ρ) + δ(m).  Built through the general
        # builder and accepted ONLY on an exact head match, so it is impossible to
        # return a wrong label -- the O(1) common case, with the learned cone index
        # / affine fit kept below as a self-correcting safety net.
        lam_cf = _pun_head_to_lambda(m, v_star)
        if self._build_clean((m, lam_cf)):
            o0 = self._fp_orbit0((m, lam_cf))
            if o0 and max(o0) == head:
                self._head_record((m, lam_cf))
                return (m, lam_cf)
        # Fallback A: cone-fingerprint lookup via a LEARNED per-m index.  The cone
        # monomials of m's free-move tower are built ONCE per magnetic m and their
        # M-test fingerprint heads cached (head -> λ); identification is then an O(1)
        # lookup (no per-call sweep).  Retained as a safety net for any head the
        # closed form's exact-match gate above declines.
        R = max((abs(x) for x in v_star), default=0) + 3
        chmap = self._cone_head_map(m, R)
        lam = chmap.get(head)
        if lam is not None:
            cone = self._cone_build((m, lam))
            if cone is not None:
                self.add_image((m, lam), *cone)
                self._head_record((m, lam))
                return (m, lam)
        # Fallback B: non-cone guesses -- cov_label, the affine head→λ fit,
        # det-translates of registered heads, central λ=0 -- each built through the
        # general builder and accepted only on an exact head match.
        cands = []
        try:
            (mc, lamc), _lead = self._cov_label(sl)
            if mc == m:
                cands.append(tuple(lamc))
        except Exception:
            pass
        cal = self._calibrate_m(m)
        if cal is not None:
            M, b, Minv = cal
            lamf = _affine_apply(Minv, tuple(v_star[t] - b[t] for t in range(N)))
            if all(x.denominator == 1 for x in lamf):
                cands.append(tuple(int(x) for x in lamf))
        shift = max((abs(x) for x in m), default=0) + 2
        for p in range(-shift, shift + 1):
            base = self._head_idx.get((tuple(u_star[t] - p for t in range(N)), v_star))
            if base is not None:
                cands.append(base[1])
        if len(set(m)) == 1:
            cands.append((0,) * N)
        seen2 = set()
        for lam in cands:
            if lam in seen2:
                continue
            seen2.add(lam)
            if not self._build_clean((m, lam)):
                continue
            o0 = self._fp_orbit0((m, lam))
            if o0 and max(o0) == head:
                self._head_record((m, lam))
                return (m, lam)
        return None

    def _mn_slice(self, P):
        """Most-negative `deg_W` slice of the covariant M-test of `P` (adaptive
        `deg_W` window): `(nd, {(u,v): int})`, or None if the M-test is empty."""
        N = self._N
        cap = 0
        X = self._norm_mtest(P, cap)
        while not X and cap < 40:
            cap += 4
            X = self._norm_mtest(P, cap)
        if not X:
            return None
        nd = min(dw for (dw, u, v) in X)
        return nd, {(u, v): c for (dw, u, v), c in X.items() if dw == nd}

    def multiply(self, a, b):
        """Structure constants `C^c_{ab}(q)` of `L_a · L_b = Σ_c C^c L_c`.

        LIVE engine: the closed-form `multiply_exact` (bar-invariant by
        construction, non-hanging, exact-reconstructing -- it returns the exact
        answer or RAISES, never a silent wrong answer).

        Rare-case fallback: `multiply_exact`'s lowest-tropical-charge peel can fail
        to resolve a summand whose leading orbit is hidden in the BUBBLING of a
        larger-magnetic summand -- the residue then won't clear and it raises.  In
        that case fall back to the M-test double peel `_multiply_mtest`, whose
        covariant deg_W grading separates every leading orbit (deg_W=0) from all
        bubbling (deg_W≥1) and so resolves these.  (The two engines are complementary:
        `multiply_exact` handles the dressed-monopole products the M-test HANGS on;
        the M-test handles the bubbling-hidden summands `multiply_exact` can't peel.)"""
        a = (tuple(a[0]), tuple(a[1]))
        b = (tuple(b[0]), tuple(b[1]))
        from pure_un_closed_form import multiply_exact
        try:
            res = multiply_exact(self, a, b)
        except (RuntimeError, NotImplementedError):
            res = self._multiply_mtest(a, b)       # rare bubbling-hidden case
        return {c: v for c, v in res.items() if not v.is_zero()}

    def _multiply_mtest(self, a, b):
        """The DOUBLE M-test peel multiply -- **RETIRED from the live path**,
        kept as a callable cross-validation oracle (see `test_closed_form_vs_engine`
        / `test_pure_un_engine_crosscheck`).

        Peels `P = L_a·L_b` and `Q = L_b·L_a` simultaneously: at each step identify
        one summand `L_c` at the most-negative `q` in either ordering (its
        fingerprint head -- lex-max of the most-negative orbit -- is pure), record
        the structure-constant term, subtract the bar pair `q^{±n}·L_c`.  Peel until
        both residues are empty.  NB it can HANG on dressed-monopole products and is
        bar-blind at the q⁰ gate -- which is exactly why the live path uses
        `multiply_exact`."""
        N = self._N
        a = (tuple(a[0]), tuple(a[1]))
        b = (tuple(b[0]), tuple(b[1]))
        P = _abe_simplify_dop(self.image(a) * self.image(b))
        Q = _abe_simplify_dop(self.image(b) * self.image(a))
        out = {}
        seen = {}                                     # (which, c, e) -> count, ping-pong guard
        for _ in range(20000):
            mp = self._mn_slice(P) if P._terms else None
            mq = self._mn_slice(Q) if Q._terms else None
            if mp is None and mq is None:
                break
            opts = ([('P', mp[0], mp[1])] if mp else []) + \
                   ([('Q', mq[0], mq[1])] if mq else [])
            picked = None
            for which, nd, sl in sorted(opts, key=lambda t: t[1]):
                head = max(sl)
                c = self._ensure_head(sl)
                if c is None:
                    continue
                fp = self._fp_orbit0(c)
                if head not in fp or fp[head] == 0:
                    continue
                s = sl[head] // fp[head]
                if s == 0:
                    continue
                picked = (which, nd, c, s)
                break
            if picked is None:
                raise RuntimeError(
                    f"pure_un._multiply_mtest: most-negative orbit unidentifiable "
                    f"(basis not bar-fixed?) for L_{a}·L_{b}")
            which, nd, c, s = picked
            e = nd if which == 'P' else -nd
            key = (which, c, e)
            seen[key] = seen.get(key, 0) + 1
            if seen[key] > 2:                          # ping-pong: no net progress
                raise RuntimeError(
                    f"pure_un._multiply_mtest: peel does not close -- L_{c} is not "
                    f"bar-fixed (build defect), reached while multiplying "
                    f"L_{a}·L_{b}")
            out[c] = out.get(c, LaurentPoly.zero()) + LaurentPoly({e: s})
            Lc = self.image(c)
            fpos = DOp.from_scalar(VRational.from_scalar(LaurentPoly({e: s}), n=N), n=N)
            fneg = DOp.from_scalar(VRational.from_scalar(LaurentPoly({-e: s}), n=N), n=N)
            P = _abe_simplify_dop(P - fpos * Lc)
            Q = _abe_simplify_dop(Q - fneg * Lc)
        else:
            raise RuntimeError("pure_un._multiply_mtest: peel did not terminate")
        if P._terms or Q._terms:
            raise RuntimeError(
                f"pure_un._multiply_mtest: non-zero residue after M-test exhausted "
                f"(basis not bar-fixed) for L_{a}·L_{b}")
        return {c: v for c, v in out.items() if not v.is_zero()}



# Default N=3 seed set spanning bare / dressed-minuscule / compound-dressed
# (incl. a reducible compound-dressed canonical), ordered low charge first.
def default_rays(N: int):
    """Canonical pure-U(N) generator ray set: the level-k minuscule monopoles
    `E_k`/`F_k` (k = 1..N-1), the determinant `det^{±1}`, and the two extreme
    Wilson lines `W_{(1,0,..,0)}` / `W_{(0,..,0,-1)}`.  Products of these (up to
    `max_len=2`) span the canonical basis.  Used by `PureUNKAlgebra.cached`."""
    z = (0,) * N
    return ([('E', k, z) for k in range(1, N)]
            + [('F', k, z) for k in range(1, N)]
            + [('det', 1), ('det', -1),
               ('W', tuple([1] + [0] * (N - 1))),
               ('W', tuple([0] * (N - 1) + [-1]))])


# =====================================================================
# SECTION 3 — the KAlgebra (formerly pure_un_kalgebra.py)
# =====================================================================

_schur_trace = trace      # Schur trace (alias used inside PureUNKAlgebra)


class PureUNKAlgebra(AbeKAlgebra):
    """Pure U(N) KAlgebra over the autobuilt canonical basis.

    `rays` + `max_len` seed the `CanonicalBasis` build (`materialize`); labels are the
    resulting `(m,e)` charges.  N=3 is sign-free (`ρ` a pure permutation)."""

    def __init__(self, N: int, rays, max_len: int = 2, K: int = 8,
                 wilson_deg: int = 0, cache=None) -> None:
        self._N = int(N)
        self._K = int(K)
        self._R = TrivialZPlusRing()
        # `cache`: a JSON path.  If it exists (and matches N,K) load the built
        # registry — skipping the minutes-long exact-trace rebuild;
        # otherwise build and write it.  (N=3 max_len=2: ~27 s build vs ~5 ms load.)
        import os
        B = None
        if cache is not None and os.path.exists(cache):
            B = CanonicalBasis.load(cache)
            if (B._N != self._N or B._K != self._K
                    or getattr(B, "_cache_version", 0) != _CANONICAL_CACHE_VERSION):
                B = None                      # stale cache (wrong N/K or older
                #                               basis-generation version) -> rebuild
        if B is None:
            # LAZY: declare the scope (rays + max_len) but build NOTHING eagerly.
            # Canonicals are built on demand by multiply/trace/image/rho via
            # ensure (box-free, memoized); a full-basis request (labels()) builds
            # the whole scope once via materialize().  ("memoize and build as
            # needed" -- the eager product-box autobuild is retired.)
            B = CanonicalBasis(self._N, K)
            B.set_scope(rays, max_len)
            if wilson_deg:
                # close the Wilson (magnetic-neutral) sector — needed when the
                # registry is the auxiliary of a matter RG flow (Wilson-valued S_RG)
                B.register_wilson(wilson_deg)
            B.ensure_identity()
            if cache is not None:
                B.materialize()           # cache persists the FULL built basis
                B.save(cache)
        self._B = B
        self._B.ensure_identity()
        # rho permutation tables (charge -> charge), (re)built over the CURRENT
        # registry; the registry grows on demand (matter-flow auxiliary), so
        # the tables are re-synced lazily on a lookup miss.  The SIGN of
        # rho(L_a) is the LEADING coefficient of rho_image at its lowest charge.
        self._rho = {}
        self._rho_inv = {}
        self._rho_residual = []
        self._sync_rho()

    @classmethod
    def cached(cls, N: int, K: int = 8, max_len: int = 2,
               cache_dir=None, rays=None, **kw) -> "PureUNKAlgebra":
        """Build-or-load the canonical pure-U(N) KAlgebra, persisting the built
        registry to JSON so repeat constructions reload in ms instead of
        rebuilding (N=4 ~6 s build / N=5 ~2 min build -> ~ms load).

        Uses `default_rays(N)` unless `rays` is given.  `cache_dir` defaults to
        `$PURE_UN_CACHE`, else a `pure_un_cache/` directory at the repo root; the
        filename encodes `(N, K, max_len, cache-version)` so a stale generation
        is never served.  N>=6 is not yet tractable to build; call with a
        smaller N or supply a prebuilt cache file."""
        import os
        if cache_dir is None:
            cache_dir = os.environ.get("PURE_UN_CACHE") or os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "pure_un_cache")
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(
            cache_dir, f"pure_u{N}_K{K}_len{max_len}_v{_CANONICAL_CACHE_VERSION}.json")
        if rays is None:
            rays = default_rays(N)
        return cls(N, rays, max_len=max_len, K=K, cache=path, **kw)

    def _sync_rho(self) -> None:
        """(Re)build the rho permutation tables over every registered canonical
        (idempotent; cheap).  Under the correct (m, λ) 't Hooft-Wilson keying ρ
        is a SIGN-FREE permutation: ρ(L_a) = L_{ρ(a)} EXACTLY, with no applied
        sign.  NB the canonical L_{ρ(a)} may itself have leading coefficient ±1
        -- the (-q)^{...} generator normalization that makes orthonormality
        (Tr(ρ(L_a)L_a)=+1) hold IS the leading coefficient, and is necessarily
        -1 on the canonicals whose exponent is odd.  So "sign-free" is checked
        as ρ(L_a) == the registered canonical at ρ(a) (same leading sign),
        NOT as leading coefficient +1 absolutely."""
        self._rho_residual = []
        for lab in self._B._reg_labels():
            if lab in self._rho:
                continue
            rimg = self._B.rho_image(lab)
            ch = _lowest_charge(rimg, self._N)
            self._rho[lab] = ch
            self._rho_inv[ch] = lab
            cL = self._B._coeff_at(rimg, ch)._coeffs
            s_rho = cL[min(cL)] if cL else 0          # leading coeff of ρ(L_a)
            if ch in self._B._idx:                    # leading coeff of canonical L_{ρ(a)}
                cC = self._B._coeff_at(self._B.image(ch), ch)._coeffs
                s_can = cC[min(cC)] if cC else 0
            else:
                s_can = s_rho                         # ρ(a) not registered: nothing to compare
            if s_rho != s_can:                        # an ACTUAL applied sign -> not sign-free
                self._rho_residual.append((lab, ch, s_rho))

    def verify_rho_signfree(self) -> bool:
        """KEY AXIOM (defines the KAlgebra): ρ permutes the canonical basis
        WITHOUT applied signs (`ρ(L_a) = L_{ρ(a)}` exactly).  The target
        canonical's own leading coefficient may be ±1 (see `_sync_rho`); what
        must vanish is any EXTRA sign ρ would apply on top of it."""
        return not self._rho_residual

    # ----- public label <-> internal build-key translation -------------------
    # The PUBLIC label is the Kapustin 't Hooft-Wilson `(m, e)` (dominant m,
    # Levi-dominant e; bare monopole -> e=0), read off the leading orbit by the
    # ansatz `_ansatz_label`.  The registry is still memoized by the BUILD key
    # (`_lowest_charge`); these translate at the boundary so the build engine is
    # untouched.  `_an` labels an internal build key; `_bk` resolves a public
    # `(m, e)` to its build key (the canonical must be registered -- on-demand
    # ansatz->build construction of a never-built label is a tracked follow-up).
    def _an(self, build_key):
        a = self._B.ansatz_of(build_key)
        if a is not None:
            return a
        # An out-of-scope multiply summand / ρ-image may not be registered yet.
        # `ensure` is now non-hanging (closed-form + chart-ρ), so register it and
        # read its Kapustin (m,e).  This also keeps rho/rho_inverse round-tripping:
        # the label is registered BOTH ways, so `_bk` can reverse it.
        try:
            self._B.ensure(build_key)
            a = self._B.ansatz_of(build_key)
            if a is not None:
                return a
        except (NotImplementedError, RuntimeError, ValueError):
            pass
        return build_key

    def _bk(self, a):
        a = (tuple(a[0]), tuple(a[1]))
        bk = self._B.build_key_of(a)
        if bk is not None:
            return bk                              # registered Kapustin (m,e)
        if a in self._B._idx:
            return a                               # already a registered build key
        # Never-built input.  A Kapustin ansatz label has DOMINANT (descending) m;
        # translate it to the build key via the leading-orbit map, register it
        # (non-hanging), and accept only if it round-trips (ansatz_of == a) -- so a
        # build-key input (anti-dominant m) that isn't a genuine ansatz falls
        # through to the build-key path below rather than being mis-translated.
        if list(a[0]) == sorted(a[0], reverse=True):
            try:
                from pure_un_closed_form import build_key_from_ansatz
                cand = build_key_from_ansatz(a[0], a[1], self._N)
                self._B.ensure(cand)
                if self._B.ansatz_of(cand) == a:
                    return cand
            except (NotImplementedError, RuntimeError, ValueError, KeyError):
                pass
        # Fall back: treat `a` as a build key (the identity, a ρ-image outside the
        # registered set, etc.); ensure/rho_inverse_image rebuild from seed.
        return a

    # ----- KAlgebra contract -----
    # ----- the AbeKAlgebra contract triple (Plan 30 T3 retrofit) ---------
    # Delegation-only: every public method below keeps its certified
    # engine path (the contract's derived methods are overridden); the
    # triple exposes the same engine through the contract surface, and the
    # retrofit suite cross-checks derived-vs-engine equality.

    def torus_shape(self):
        """Pure U(N): one node, no fundamentals."""
        from abe_kalgebra import TorusShape
        return TorusShape.from_ranks_nf((self._N,), (0,))

    def chart(self, label):
        """`L_label` as a `WRQTorus` — the group-general substrate selected by
        `torus_shape()` (D9/D10: WRQTorus universally).  The full-scope engine
        image (`urqt`, a URQTorus) transported residual-by-residual via
        `vr_to_tr`; the fast engine (`self._B`) and its method overrides are
        unchanged, so no canonical scope is lost."""
        from wrq_torus import WRQTorus                # lazy: avoids a cycle
        from matter_wrq_torus import vr_to_tr
        D = self.torus().datum
        x = self.urqt(label)
        return WRQTorus(D, {m: vr_to_tr(D, vr) for m, vr in x.residuals().items()})

    def decompose(self, x, max_terms: int = 5000):
        """Canonical-basis read of a pure-shape torus element: the
        leading-Weyl-orbit peel **against the registry's own images**
        (recognize the leading orbit on the residuals, subtract
        `C·urqt(label)`, recurse) — full registry scope, residual-native.
        Honest-fails if the peel does not terminate.

        The live substrate is a `WRQTorus` (`chart`'s output); the full-scope
        peel stays URQ-native, so the WRQ input is transported back via
        `tr_to_vr` (a `QuiverURQTorus`/`URQTorus` legacy input still routes
        through its `()` level)."""
        from urq_torus import URQTorus                # lazy: avoids a cycle
        from abelianized_torus import VRational
        from wrq_torus import WRQTorus
        N = self._N
        if isinstance(x, WRQTorus):
            from matter_wrq_torus import tr_to_vr
            D = self.torus().datum
            f = {m: tr_to_vr(D, tr) for m, tr in x.residuals().items()}
        else:
            f = {m: row[()] for m, row in x.residuals().items() if () in row}
        cur = URQTorus.from_f(f, N)
        out: dict = {}
        for _ in range(max_terms):
            if not cur.residuals():
                return Element({lab: C for lab, C in out.items()
                                if not C.is_zero()})
            rec = cur.recognize_leading()      # {lower-Kapustin: C(q)}
            sub = {}
            for (m, e), C in rec.items():
                eng = (tuple(reversed(m)), tuple(reversed(e)))   # joint w0
                img = self.urqt(eng)
                # normalize by the image's own leading monomial in the SAME
                # read frame (dressed labels carry a Witten-frame q-power)
                c0 = img.recognize_leading().get((m, e))
                if c0 is None or len(c0._coeffs) != 1:
                    raise NotImplementedError(
                        f"decompose: image lead of {eng} not a monomial")
                (p0, z0), = c0._coeffs.items()
                if z0 not in (1, -1):
                    raise NotImplementedError(
                        f"decompose: image lead of {eng} not unit: {c0}")
                Ct = LaurentPoly({e0 - p0: z0 * c
                                  for e0, c in C._coeffs.items()})
                out[eng] = out.get(eng, LaurentPoly.zero()) + Ct
                cv = VRational.from_scalar(Ct * LaurentPoly({0: -1}), n=N)
                for mm, ff in img.residuals().items():
                    t = (ff * cv).simplify()
                    sub[mm] = t if mm not in sub else (sub[mm] + t).simplify()
            cur = cur + URQTorus.from_f(sub, N)
        raise NotImplementedError(
            f"PureUNKAlgebra.decompose: peel did not terminate within "
            f"{max_terms} rounds")

    def coefficient_ring(self):
        return self._R

    def identity(self):
        return ((0,) * self._N, (0,) * self._N)   # bare monopole at m=0 -> e=0

    def cache_identity(self) -> str:
        """Distinguish pure-U(N) structure-constant caches by rank + trace
        depth (parametric refinement of the base `cache_identity`)."""
        return f"PureUNKAlgebra(N={self._N},K={self._K})"

    def labels(self):
        return [self._an(bk) for bk in self._B.labels()]

    def multiply(self, a, b) -> Element:
        # operate internally in build keys; the correct (m,e) keying makes ρ
        # sign-free, so no eps-twist of the structure constants is needed.
        raw = self._B.multiply(self._bk(a), self._bk(b))
        return Element({self._an(c): v for c, v in raw.items()})

    def rho(self, a):
        bk = self._bk(a)
        if bk not in self._rho:
            self._B.ensure(bk)
            self._sync_rho()
        return self._an(self._rho[bk])

    def rho_inverse(self, a):
        """ρ⁻¹(L_a) -- TOTAL and closed-form, the exact inverse of `rho`
        (`rho_inverse_label` swaps E<->F).  Reads the lowest tropical charge of
        the ρ⁻¹-image (`rho_inverse_image`), built from the same seed via the
        inverse generator map -- NOT a registry reverse-lookup, so it never
        misses a preimage -- then relabels to the public `(m, e)`."""
        bk = self._bk(a)
        x = _lowest_charge(self._B.rho_inverse_image(bk), self._N)
        return self._an(x)

    def trace(self, a, K: int = 20) -> RPowerSeries:
        lp = _schur_trace(self._B.image(self._bk(a)), self._N, K, adaptive=True)
        return RPowerSeries(self._R, {e: c for e, c in lp._coeffs.items()
                                      if 0 <= e <= K}, K)

    def _label_section_decompose(self, label):
        """**Aspirationally obsolete (Plan 32)** — pure U(N) is unflavoured
        (`TrivialZPlusRing`), so the section is the label and the coefficient
        is `1`.  Superseded by the inherited (trivial) `r_label_decompose`
        from `AbeKAlgebra`; kept while `to_R_form` routes through it."""
        return (label, self._R.one())

    # ----- override the derived inner_product with the verified direct pairing
    def inner_product(self, a, b, K: int = 20) -> RPowerSeries:
        """`I_{a,b} = [Tr(ρ(L_a)·L_b)]` straight from the registry images --
        exact (adaptive trace), and independent of multiply-/ρ-closure."""
        lp = _trace_prod(self._B.rho_image(self._bk(a)), self._B.image(self._bk(b)),
                         self._N, K, adaptive=True)   # only the u^0 block is needed
        return RPowerSeries(self._R, {e: c for e, c in lp._coeffs.items()
                                      if 0 <= e <= K}, K)

    def urqt(self, label):
        """The canonical `L_label` as a `URQTorus` (enriched rational quantum
        torus) element -- its residual vector `(f_m)`.  Read-only view onto the
        chart image; the `f_m` dict, traces, the leading Levi piece and the
        recognition tools are then available off the returned object.
        (`urq_torus` is imported lazily to avoid an import cycle.)

        CONVENTION WARNING: `label` here is the ENGINE public label
        `(dominant m, Levi-dominant e)` (what `labels()` returns) -- this is the
        `w_0` image of, and DIFFERS from, the URQTorus canonical convention
        `(anti-dominant m, Levi-anti-dominant e)` used by `URQTorus.recognize_*`
        and by `minuscule`/`wilson`.  To build from the canonical convention, use
        `self.minuscule(m, e)` / `self.wilson(e)` instead of `urqt`."""
        from urq_torus import URQTorus
        return URQTorus.from_chart(self._B.image(self._bk(label)), self._N)

    def minuscule(self, m, e):
        """`L_{m,e}` for a MINUSCULE magnetic `m = φ_k` (+ optional `det`), OR a
        central **dressed det** `m = p·det` (all entries equal), and an ARBITRARY
        dressing `e`, returned as a `URQTorus` (the residual vector).  Dressed det
        builds as `det^p · χ_e` (central monopole × Wilson line).

        Labels are the canonical **Weyl-orbit representative** convention
        `(m, e) = (anti-dominant m, Levi-anti-dominant e)` -- the Weyl group acts
        *jointly* on `m` and `e`.  Arbitrary input `(m, e)` is canonicalised to
        that representative (sort `m` ascending carrying `e` along, then sort `e`
        ascending within each Levi block) before building, so any Weyl image of a
        valid label resolves to the same canonical.  Internally it translates to
        the engine's `(dominant m, Levi-dominant e)` label (its `w_0` image).
        Raises if `m` is not minuscule."""
        from urq_torus import URQTorus
        from pure_un_closed_form import _is_minuscule, _levi_blocks
        N = self._N
        m = [int(x) for x in m]
        e = [int(x) for x in e]
        # joint canonicalisation under W = S_N: one permutation acts on both
        order = sorted(range(N), key=lambda i: m[i])      # m ascending (anti-dom)
        m_anti = tuple(m[i] for i in order)
        e1 = [e[i] for i in order]
        e_anti = list(e1)
        for blk in _levi_blocks(m_anti):                  # Levi-anti-dom: sort e up
            for i, v in zip(blk, sorted(e1[i] for i in blk)):
                e_anti[i] = v
        e_anti = tuple(e_anti)
        k, p = _is_minuscule(m_anti, N)
        if k is None:
            # DRESSED DET: m = p*det (central -- all entries equal), dressed by the
            # full-U(N) irrep e.  Like Wilson at the shifted central magnetic: the
            # residual f_{p*det} is the bare character chi_e(v) (q-free => palindromic
            # => bar-invariant and orthonormal).  (NB the chart product det^p.chi_e is
            # NOT this canonical -- its residual is a non-palindromic q-monomial.)
            if len(set(m_anti)) == 1:
                chi = L_W(e_anti[::-1], N)._terms[(0,) * N]      # chi_e(v), q-free
                return URQTorus.from_f({(m_anti[0],) * N: chi}, N)
            raise ValueError(f"m={tuple(m)} is not minuscule (phi_k + p*det) nor central det")
        # engine public label is (dominant m, Levi-dominant e) = w_0 of the rep
        return URQTorus.from_chart(
            self._B.image(self._bk((m_anti[::-1], e_anti[::-1]))), N)

    def wilson(self, e):
        """`L_{0,e}` -- the Wilson line for the U(N) irrep of weight `e` (the `m=0`
        canonical `χ_e(v)`), as a `URQTorus`.  The `m=0` special case of the same
        `(anti-dom m, Levi-anti-dom e)` convention: `m=0` is central, so the Levi
        is all of U(N) (one block) and 'valid e' is any U(N) weight; arbitrary `e`
        is canonicalised by the full Weyl group (just sorting) to its
        representative.  `L_{0,0}` is the identity `χ_0 = 1`."""
        from urq_torus import URQTorus
        N = self._N
        e_anti = tuple(sorted(int(x) for x in e))           # Levi-anti-dom (full U(N))
        return URQTorus.from_chart(L_W(e_anti[::-1], N), N)  # χ_e for the dominant weight


def pure_u3(rays=None, max_len: int = 3, K: int = 8) -> PureUNKAlgebra:
    """A ready pure-U(3) KAlgebra over the pure-E + det cone (sign-free)."""
    if rays is None:
        rays = [('E', 1, (0, 0, 0)), ('E', 1, (0, 1, 0)), ('E', 1, (0, 1, 1)),
                ('E', 2, (0, 0, 0)), ('E', 2, (0, 0, 1)), ('det', 1)]
    return PureUNKAlgebra(3, rays, max_len=max_len, K=K)
