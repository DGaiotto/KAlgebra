"""Spine-free multiplication for the SU(2)+N_f=3 standalone `ConeKAlgebra`.

The SU(4) sibling of `su2_nf2_h_multiply`, using the **same literal-word
reducer** — and total on every magnetic level.  Everything is kept as literal
H/Wilson words; only adjacent PAIRS are multiplied (`h_mul_h`, always
magnetic ≤ 2); the word is reduced to canonical (sorted, single-cone) form, and
the canonical native label is read off at the end.  The ONLY N_f=3 difference
from N_f=2 is the **within-cone matter** (the gap-1 onset
`H_a·H_{a+1}=q·M_{2a+1}+1`):

  * the q-commuting reorder is matter-aware,
        `H_x H_{x+1} = q²·H_{x+1} H_x + (1−q²)·1`
    (N_f=2 has only the phase; N_f=3 adds the flavour-trivial identity term);
  * an input native label (= the **canonical** `M^(m)_e`) is expanded to literal
    H-words via the inverse Gaussian q-binomial (`canon_in_mono`), and a reduced
    sorted within-cone word is read back to canonical labels via the forward
    Gaussian (`gaussian`).  (N_f=2 uses count + phase, since there the literal
    monomial *is* the canonical.)

The within-cone monomials are q-commuting (the matter is the flavour-trivial
SU(4)-singlet); the cross-cone matter (SU(4) characters 4/4̄/6/15/…) rides in
the `h_mul_h` ray products' `RLaurent`-over-SU(4) coefficients and is expanded
to SU(4) Cartan weights + shifted by the input flavour weights at assembly.
Base inputs: the validated 2-letter product `su2_nf3_h_gap_k.h_mul_h` (all gaps)
and the Wilson Clebsch `w1_H`.  NO BPS/RG/quantum-torus engine; no tropical
labels — the reducer is pure (m,e) cone arithmetic.

Native label: `(h_factors, (m1, m2, m3))` — `h_factors` a sorted tuple of
`(n, exp)` (H-monomial) or `((('W', e), 1),)` (Wilson χ_e), `(m1, m2, m3)` the
SU(4) Cartan flavour weight (Dynkin/ω-basis).
"""
from __future__ import annotations

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from laurent_poly import LaurentPoly
from kalgebra import Element
from zplus_ring import RLaurent, RElement
from su2_nf3_h_gap_k import R_SU4, rl, rl_zero, h_mul_h, w1_H
from pure_su2_h_wilson import (
    WILSON_FUND, is_wilson_label, chi_to_w1_powers, w1_power_to_chi,
)

_R = R_SU4
_TRIV = _R.basis_element((0, 0, 0))


# ---- RLaurent[SU4] helpers ------------------------------------------

def _rl_q(n: int) -> RLaurent:
    return RLaurent(_R, {n: _TRIV})


def _rl_int(c: int) -> RLaurent:
    return RLaurent(_R, {0: _TRIV * c}) if c else rl_zero()


def _z() -> RLaurent:
    return rl_zero()


def _barsm(sm: dict) -> dict:
    return {s: RLaurent(_R, {-q: r for q, r in c.coeffs.items()})
            for s, c in sm.items() if not c.is_zero()}


# ---- Gaussian q-binomial in x = q² ----------------------------------

def _gauss_q2(n: int, k: int) -> dict:
    """Gaussian binomial [n choose k]_{q²} as {q-power: int}."""
    if k < 0 or k > n:
        return {}

    def qfac(m):
        out = {0: 1}
        for i in range(1, m + 1):
            nf = {}
            for a, ca in out.items():
                for j in range(i):
                    nf[a + 2 * j] = nf.get(a + 2 * j, 0) + ca
            out = nf
        return out
    num = qfac(n)
    d1, d2 = qfac(k), qfac(n - k)
    den = {}
    for a, ca in d1.items():
        for b, cb in d2.items():
            den[a + b] = den.get(a + b, 0) + ca * cb
    work = dict(num)
    dmin = min(den)
    out: dict = {}
    while work:
        m0 = min(work)
        c = work[m0] // den[dmin]
        if c == 0:
            del work[m0]
            continue
        out[m0 - dmin] = out.get(m0 - dmin, 0) + c
        for de, dc in den.items():
            work[m0 - dmin + de] = work.get(m0 - dmin + de, 0) - c * dc
            if work[m0 - dmin + de] == 0:
                del work[m0 - dmin + de]
    return {q: c for q, c in out.items() if c}


# ---- forward Gaussian: H_b^p H_{b+1}^q (SORTED) -> canonical seeds ---
# Flavour-trivial: M^(p+q-2k)_{(q-k)+(p+q-2k)b}, coeff q^{(p-k)(q-k)}·[min;k]_{q²}.
_GA: dict = {}


def gaussian(b: int, p: int, q: int) -> dict:
    """Within-cone literal H_b^p H_{b+1}^q in canonical cone-monomial seeds."""
    key = (b, p, q)
    if key in _GA:
        return _GA[key]
    mn = min(p, q)
    out: dict = {}
    for k in range(mn + 1):
        m = p + q - 2 * k
        e = (q - k) + m * b
        gb = _gauss_q2(mn, k)
        base = (p - k) * (q - k)
        coeff = RLaurent(_R, {base + qq: _TRIV * cc for qq, cc in gb.items()})
        if not coeff.is_zero():
            seed = (m, e) if m >= 1 else (0, 0)        # m=0 -> identity
            out[seed] = out.get(seed, _z()) + coeff
    res = {s: c for s, c in out.items() if not c.is_zero()}
    _GA[key] = res
    return res


# ---- inverse Gaussian: canonical M^(m)_e -> sorted literals ----------
_CIM: dict = {}


def _recone(lit: tuple, b: int) -> tuple:
    """Re-express literal (bk,Pk,Qk)=H_bk^Pk H_{bk+1}^Qk in cone b (indices in
    {b, b+1})."""
    bk, Pk, Qk = lit
    if bk == b:
        return (b, Pk, Qk)
    if bk == b + 1:
        assert Qk == 0, f"recone {lit}->{b}: H_{{{bk + 1}}} not in cone {b}"
        return (b, 0, Pk)
    if bk == b - 1:
        assert Pk == 0, f"recone {lit}->{b}: H_{{{bk}}} not in cone {b}"
        return (b, Qk, 0)
    raise AssertionError(f"recone {lit} -> cone {b}: out of {{b, b+1}}")


def canon_in_mono(m: int, e: int) -> dict:
    """Canonical M^(m)_e as Σ coeff·(H_b^P H_{b+1}^Q) -> {(b,P,Q): RLaurent}, all
    literals re-coned to b = e // m.  Identity = the empty literal (b,0,0)."""
    if (m, e) in _CIM:
        return _CIM[(m, e)]
    if m == 0:
        assert e == 0, f"canon_in_mono(0,{e}): only identity expected"
        return {(0, 0, 0): rl(0)}
    if m == 1:
        return {(e, 1, 0): rl(0)}
    b, r = divmod(e, m)
    p, q = m - r, r
    mn = min(p, q)
    inv_c0 = rl(-p * q)
    res = {(b, p, q): inv_c0}
    for k in range(1, mn + 1):
        mk = m - 2 * k
        ek = e - k * (1 + 2 * b)
        gb = _gauss_q2(mn, k)
        ck = RLaurent(_R, {(p - k) * (q - k) + qq: _TRIV * cc for qq, cc in gb.items()})
        fac = ck * inv_c0
        for lit, sc in canon_in_mono(mk, ek).items():
            key = (b, 0, 0) if (lit[1] == 0 and lit[2] == 0) else _recone(lit, b)
            res[key] = res.get(key, _z()) - fac * sc
    res = {s: c for s, c in res.items() if not c.is_zero()}
    _CIM[(m, e)] = res
    return res


# ---- native gauge label <-> clean (m,e) seed ------------------------

def _native_gauge(seed: tuple) -> tuple:
    """clean seed (m,e) -> gauge h_factors (max-diagonal)."""
    m, e = seed
    if m == 0:
        return () if e == 0 else ((('W', e), 1),)
    if m == 1:
        return ((e, 1),)
    base, rem = divmod(e, m)
    out = []
    if m - rem > 0:
        out.append((base, m - rem))
    if rem > 0:
        out.append((base + 1, rem))
    return tuple(out)


def _seed_of(hf: tuple) -> tuple:
    """gauge h_factors -> clean seed (m,e); m=0 Wilson(e)/identity(e=0)."""
    if not hf:
        return (0, 0)
    first = hf[0]
    if isinstance(first[0], tuple) and first[0][0] == 'W':
        return (0, first[0][1])
    m = sum(x for _, x in hf)
    e = sum(n * x for n, x in hf)
    return (m, e)


# ---- literal words: letters are ('H', n) or WILSON_FUND -------------

def _is_H(l) -> bool:
    return not is_wilson_label(l)


def _q_commute(g, h) -> bool:
    if g == h:
        return True
    gw, hw = is_wilson_label(g), is_wilson_label(h)
    if gw and hw:
        return True
    if gw or hw:
        return False
    return abs(g[1] - h[1]) <= 1


def _cocycle(g, h) -> int:
    return h[1] - g[1]


def _lit_word(b: int, P: int, Q: int) -> tuple:
    return tuple([('H', b)] * P + [('H', b + 1)] * Q)


def _wilson_words(coef: RLaurent, e: int) -> list:
    """coef·χ_e (Wilson) -> [(RLaurent, w_1^j word)] via chi_to_w1_powers."""
    if e == 0:
        return [(coef, ())]
    out = []
    for j, scalar in chi_to_w1_powers(e).items():
        if scalar:
            t = coef * _rl_int(scalar)
            if not t.is_zero():
                out.append((t, (WILSON_FUND,) * j))
    return out


def _canon_to_lit_words(m: int, e: int) -> list:
    """canonical M^(m)_e -> [(RLaurent, literal word)] (inverse Gaussian)."""
    if m == 0:
        return _wilson_words(rl(0), e)
    return [(c, _lit_word(b, P, Q)) for (b, P, Q), c in canon_in_mono(m, e).items()]


# ---- ray products (cross) for non-q-commuting adjacency -------------

_RAY: dict = {}


def _ray_product(g, h) -> list:
    """L_g · L_h for a non-q-commuting adjacent letter pair -> [(RLaurent,
    literal word)] via the closed 2-letter products (h_mul_h / Wilson Clebsch)."""
    key = (g, h)
    if key in _RAY:
        return _RAY[key]
    if _is_H(g) and _is_H(h):
        a, b = g[1], h[1]
        canonical = h_mul_h(a, b) if a <= b else _barsm(h_mul_h(b, a))
        res = []
        for (m, e), coef in canonical.items():
            if coef.is_zero():
                continue
            if m == 0:
                res.extend(_wilson_words(coef, e))
            else:
                for c2, w in _canon_to_lit_words(m, e):
                    t = coef * c2
                    if not t.is_zero():
                        res.append((t, w))
    elif _is_H(g) and is_wilson_label(h):
        # H_n · w_1 = right Clebsch = (left Clebsch with q<->q^{-1} on H terms)
        res = []
        for (m, e), c in w1_H(g[1]).items():
            if m == 1:
                cc = RLaurent(_R, {-q: r for q, r in c.coeffs.items()})
                res.append((cc, (('H', e),)))
            else:
                res.extend(_wilson_words(c, e))
    elif is_wilson_label(g) and _is_H(h):
        res = []                                       # w_1 · H_n (left Clebsch)
        for (m, e), c in w1_H(h[1]).items():
            if m == 1:
                res.append((c, (('H', e),)))
            else:
                res.extend(_wilson_words(c, e))
    else:
        raise ValueError(f"_ray_product: unexpected ({g}, {h})")
    _RAY[key] = res
    return res


# ---- word canonicalisation (cross / matter-aware swap / bubble) -----

def _word_is_canonical(word: tuple) -> bool:
    if not word:
        return True
    if all(is_wilson_label(l) for l in word):
        return True
    if all(_is_H(l) for l in word):
        chain = [l[1] for l in word]
        if any(chain[i] > chain[i + 1] for i in range(len(chain) - 1)):
            return False
        return (max(chain) - min(chain)) <= 1
    return False


def _letter_gt(g, h) -> bool:
    if _is_H(g) and _is_H(h):
        return g[1] > h[1]
    if is_wilson_label(g) and _is_H(h):
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
    """Reduce literal-word terms [(RLaurent, word)] to canonical form."""
    completed: list = []
    work = list(initial)
    steps = 0
    while work:
        steps += 1
        if steps > 4_000_000:
            raise RuntimeError("su2_nf3 _reduce: step budget exceeded")
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
            cc = 2 * _cocycle(g, h)                     # g>h: q^{2·cocycle} (=q^{-2})
            work.append((c * _rl_q(cc), w[:i] + (h, g) + w[i + 2:]))
            mat = c * (_rl_q(0) - _rl_q(cc))            # within-cone matter (1-q^{cc})
            if not mat.is_zero():
                work.append((mat, w[:i] + w[i + 2:]))   # drop the reordered pair
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
                # move target left past q-commuting partner (matter-aware):
                # H_partner H_target = q^{2·cocycle} H_target H_partner + (1-q^{..})·1
                cc = 2 * _cocycle(partner, target)
                mat = c * _rl_q(phase_exp) * (_rl_q(0) - _rl_q(cc))
                if not mat.is_zero():
                    cur = tuple(new_w)
                    work.append((mat, cur[:k - 1] + cur[k + 1:]))
                phase_exp += cc
                new_w[k - 1], new_w[k] = new_w[k], new_w[k - 1]
            if not fired:
                work.append((c * _rl_q(phase_exp), tuple(new_w)))
    return completed


# ---- canonical word -> native gauge labels (forward Gaussian) -------

def _word_to_native(word: tuple) -> dict:
    """canonical literal word -> {gauge h_factors: RLaurent} (flavour-trivial)."""
    if not word:
        return {(): rl(0)}
    if all(is_wilson_label(l) for l in word):
        out: dict = {}
        for e, mult in w1_power_to_chi(len(word)).items():
            if mult:
                hf = () if e == 0 else ((('W', e), 1),)
                out[hf] = out.get(hf, _z()) + _rl_int(mult)
        return {k: v for k, v in out.items() if not v.is_zero()}
    chain = sorted(l[1] for l in word)
    b = chain[0]
    P = sum(1 for x in chain if x == b)
    Q = sum(1 for x in chain if x == b + 1)
    assert P + Q == len(chain) and max(chain) - min(chain) <= 1, \
        f"_word_to_native: non-within-cone {word}"
    out = {}
    for (m, e), c in gaussian(b, P, Q).items():
        hf = _native_gauge((m, e))
        out[hf] = out.get(hf, _z()) + c
    return {k: v for k, v in out.items() if not v.is_zero()}


# ---- gauge product (flavour-neutral, native gauge labels) -----------

def _expand_gauge_to_literal(h_factors: tuple) -> list:
    """gauge native label (= canonical M / Wilson) -> [(RLaurent, literal word)]
    via the inverse Gaussian."""
    if not h_factors:
        return [(rl(0), ())]
    first = h_factors[0]
    if isinstance(first[0], tuple) and first[0][0] == 'W':
        return _canon_to_lit_words(0, first[0][1])
    m = sum(x for _, x in h_factors)
    e = sum(n * x for n, x in h_factors)
    return _canon_to_lit_words(m, e)


def _gauge_product(hf_a: tuple, hf_b: tuple) -> dict:
    """h_factors_a · h_factors_b -> {gauge h_factors: RLaurent[SU4]}."""
    terms_a = _expand_gauge_to_literal(hf_a)
    terms_b = _expand_gauge_to_literal(hf_b)
    initial = [(ca * cb, wa + wb) for (ca, wa) in terms_a for (cb, wb) in terms_b]
    completed = _reduce(initial)
    out: dict = {}
    for c, w in completed:
        for hf, factor in _word_to_native(w).items():
            t = c * factor
            if not t.is_zero():
                out[hf] = out.get(hf, _z()) + t
    return {k: v for k, v in out.items() if not v.is_zero()}


def _gauge_mul(sA: tuple, sB: tuple) -> dict:
    """L_sA · L_sB -> {clean seed (m,e): RLaurent[SU4]} (clean-seed wrapper over
    the word reducer; total on every magnetic level).  Used by the trace's
    inner-product OPE and the cone-data cross-product."""
    out: dict = {}
    for hf, c in _gauge_product(_native_gauge(sA), _native_gauge(sB)).items():
        out[_seed_of(hf)] = out.get(_seed_of(hf), _z()) + c
    return {s: v for s, v in out.items() if not v.is_zero()}


# ---- flavour: SU(4) char coeff -> Cartan weights --------------------

def _weights_of(relt) -> dict:
    out: dict = {}
    for pqr, c in relt.terms.items():
        for wt, m in _R._irrep_weights(pqr).items():
            out[wt] = out.get(wt, 0) + c * m
    return out


# ---- top-level multiply ---------------------------------------------

def multiply_native(a: tuple, b: tuple) -> Element:
    """(h_factors_a, w_a) · (h_factors_b, w_b) -> Element over native labels
    (h_factors, (m1,m2,m3)) with integer LaurentPoly coeffs.  Total on every
    magnetic level."""
    hfa, wa = a[0], tuple(a[1])
    hfb, wb = b[0], tuple(b[1])
    wsum = tuple(wa[i] + wb[i] for i in range(3))
    gprod = _gauge_product(hfa, hfb)
    out: dict = {}
    for hf, rlc in gprod.items():
        for qexp, relt in rlc.coeffs.items():
            for wt, z in _weights_of(relt).items():
                if z == 0:
                    continue
                label = (hf, tuple(wsum[i] + wt[i] for i in range(3)))
                lp = out.setdefault(label, LaurentPoly({}))
                out[label] = lp + LaurentPoly({qexp: z})
    return Element({k: v for k, v in out.items() if not v.is_zero()})


__all__ = ['multiply_native', '_seed_of', '_native_gauge', '_gauge_mul',
           '_gauge_product', 'gaussian', 'canon_in_mono', 'WILSON_FUND']
