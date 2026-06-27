"""Analytic pure-SU(2) trace: exact Q(q) reduction to the Wilson seeds.

Every canonical trace `Tr(L_{(m,e)})` is reduced, by ρ²-cyclicity alone,
to an *exact* `Q(q)`-linear combination of the Wilson seeds `Tr W_e`,

    Tr(L_{(m,e)}) = Σ_e  c_{m,e}(q) · Tr W_e ,      c_{m,e}(q) ∈ Q(q),

with no q-series truncation anywhere upstream.  The numeric Schur values
of the Wilson seeds are substituted only at the very end, in
`trace_series`.  This removes the precision-buffer machinery of the
numeric solver entirely (there is exactly one power-series inversion, in
the final `SymTr.to_series`), and lets the prefix-independence of the
cyclicity solve be checked *exactly* by cross-multiplying rational
functions rather than comparing truncated series.

The cyclicity matrix and the m-letter product expansions are reused from
`pure_su2_layer2_identities` (the same axiom-derived `multiply_native`
ray table feeds both).  The only change here is the coefficient ring:
`RatFunc` (exact `Q(q)`) instead of truncated `LaurentPoly`.
"""
from __future__ import annotations

from fractions import Fraction

from laurent_poly import LaurentPoly
from ratfunc_q import RatFunc
from pure_su2_layer2_identities import (
    _build_M_for_prefix, _PREFIX_CANDIDATES_BASE, _det_n,
)


# ---------------------------------------------------------------------------
# Wilson two-seed reduction:  Tr W_{2n} = A_n(q)·Tr W_0 + B_n(q)·Tr W_2.
# ---------------------------------------------------------------------------
#
# From the cute recursion  G(n) := Tr W_{2n} − Tr W_{2n+2} = q^{2n} G(n−2)
# with the Weyl boundary  G(1) = −2 q² Tr W_0  (= W_{-2} = −W_0), every
# even Wilson trace reduces to the two genuine seeds {Tr 1, Tr W_2}:
#
#     B_n = Σ_{0 ≤ 2j < n} q^{2j(j+1)},
#     A_n = 1 − Σ_{0 ≤ 2j < n} q^{2j(j+1)} + 2 Σ_{0 ≤ 2j+1 < n} q^{2j(j+2)+2}.
#
# (A_n, B_n are exact Laurent polynomials; see `pure_su2_wilson_uniqueness`
# for the derivation and the n→∞ stabilisation `B_n→Tr 1`, `A_n→−Tr W_2`.)

_two_seed_cache: dict = {}


def _wilson_two_seed(n: int):
    """`(A_n, B_n)` as `LaurentPoly` with `Tr W_{2n} = A_n·Tr W_0 + B_n·Tr W_2`.

    `n` may be negative; the reflection `W_{-2m} = −W_{2m-2}` is folded in.
    """
    if n in _two_seed_cache:
        return _two_seed_cache[n]
    if n < 0:
        # Tr W_{2n} = Tr W_{-2|n|} = − Tr W_{2(|n|-1)}.
        A, B = _wilson_two_seed(-n - 1)
        res = (LaurentPoly({e: -c for e, c in A._coeffs.items()}),
               LaurentPoly({e: -c for e, c in B._coeffs.items()}))
        _two_seed_cache[n] = res
        return res
    if n == 0:
        res = (LaurentPoly({0: 1}), LaurentPoly.zero())          # Tr W_0
        _two_seed_cache[n] = res
        return res
    if n == 1:
        res = (LaurentPoly.zero(), LaurentPoly({0: 1}))          # Tr W_2
        _two_seed_cache[n] = res
        return res
    B = {}
    A = {0: 1}
    j = 0
    while 2 * j < n:
        e = 2 * j * (j + 1)
        B[e] = B.get(e, 0) + 1
        A[e] = A.get(e, 0) - 1
        j += 1
    j = 0
    while 2 * j + 1 < n:
        e = 2 * j * (j + 2) + 2
        A[e] = A.get(e, 0) + 2
        j += 1
    res = (LaurentPoly({e: c for e, c in A.items() if c}),
           LaurentPoly({e: c for e, c in B.items() if c}))
    _two_seed_cache[n] = res
    return res


# ---------------------------------------------------------------------------
# SymTr: a symbolic trace, Σ_e coef_e(q) · Tr W_e  with coef_e ∈ Q(q).
# ---------------------------------------------------------------------------

class SymTr:
    """Linear combination of Wilson seeds with `RatFunc` coefficients."""

    __slots__ = ("terms",)

    def __init__(self, terms: dict | None = None):
        # terms: dict[int e -> RatFunc], zero coefficients dropped.
        self.terms = {}
        if terms:
            for e, c in terms.items():
                if not c.is_zero():
                    self.terms[e] = c

    @staticmethod
    def seed(e: int) -> "SymTr":
        """Tr W_e itself."""
        return SymTr({e: RatFunc.one()})

    @staticmethod
    def zero() -> "SymTr":
        return SymTr()

    def canonical(self) -> "SymTr":
        """Reduce to the two genuine Wilson seeds `{W_0, W_2}`.

        First fold the reflection `W_{-e} = -W_{e-2}`, then apply the
        Wilson `cute` recursion `Tr W_{2n} = A_n·Tr W_0 + B_n·Tr W_2`
        (Laurent-polynomial `A_n, B_n` from `_wilson_two_seed`) so that
        every seed collapses onto `{0, 2}`.  Two SymTrs are equal *as
        traces* iff their canonical forms match — the m≥2 cyclicity solve
        is prefix-independent only after this reduction, since distinct
        prefixes differ precisely by multiples of the Wilson recursion."""
        out: dict = {}
        for e, c in self.terms.items():
            if e >= 0:
                ce, cc = e, c
            else:
                ce, cc = (-e - 2), (-c)
            n = ce // 2
            A_n, B_n = _wilson_two_seed(n)
            out[0] = (out.get(0, RatFunc.zero())
                      + cc * RatFunc.from_laurent(A_n))
            out[2] = (out.get(2, RatFunc.zero())
                      + cc * RatFunc.from_laurent(B_n))
        return SymTr({k: v for k, v in out.items() if not v.is_zero()})

    def equals(self, o: "SymTr") -> bool:
        """Exact functional equality over Q(q)."""
        a, b = self.canonical().terms, o.canonical().terms
        keys = set(a) | set(b)
        zero = RatFunc.zero()
        return all((a.get(k, zero) - b.get(k, zero)).is_zero() for k in keys)

    def __add__(self, o: "SymTr") -> "SymTr":
        out = dict(self.terms)
        for e, c in o.terms.items():
            out[e] = (out[e] + c) if e in out else c
        return SymTr(out)

    def scale(self, r: RatFunc) -> "SymTr":
        if r.is_zero():
            return SymTr.zero()
        return SymTr({e: c * r for e, c in self.terms.items()})

    def to_series(self, wilson_series, q_max: int) -> LaurentPoly:
        """Substitute numeric Wilson seeds and collapse to a q-series.

        `wilson_series(e, q_max)` returns `Tr W_e` as a `LaurentPoly`.
        This is the single point where numbers enter.  Each coefficient
        is an exact `RatFunc` whose lowest q-power may be strongly
        negative (deep H-shift cones), so we evaluate both the coefficient
        and the Wilson seed with a generous buffer below `q_max` and only
        truncate the *product* at the end.
        """
        # Largest negative shift among the coefficients sets how deep the
        # seed series must run for the product to be exact through q^q_max.
        max_neg = 0
        for c in self.terms.values():
            if c.shift < max_neg:
                max_neg = c.shift
        buf = -max_neg + 8
        out = LaurentPoly.zero()
        for e, c in self.terms.items():
            coef_series = c.to_series(LaurentPoly, q_max + buf)
            out = out + coef_series * wilson_series(abs(e), q_max + buf)
        return LaurentPoly({k: v for k, v in out._coeffs.items() if k <= q_max})


# ---------------------------------------------------------------------------
# Wilson reflection: Tr W_{-e} = - Tr W_{e-2}  (SU(2) Weyl, χ_{-e}=-χ_{e-2}).
# ---------------------------------------------------------------------------

def _seed_symtr(e: int) -> SymTr:
    """SymTr for a single Wilson trace `Tr W_e`, with even-e and the Weyl
    reflection folded in.  Odd e vanish."""
    if e % 2 == 1:
        return SymTr.zero()
    if e >= 0:
        return SymTr.seed(e)
    # e < 0:  Tr W_e = - Tr W_{-e-2}
    return SymTr.seed(-e - 2).scale(RatFunc(0, {0: Fraction(-1)}, {0: Fraction(1)}))


# ---------------------------------------------------------------------------
# Matrix helpers over RatFunc.
# ---------------------------------------------------------------------------

def _lp_to_rf(lp: LaurentPoly) -> RatFunc:
    return RatFunc.from_laurent(lp)


def _det_rf(M: list) -> RatFunc:
    """Determinant of a square matrix of `RatFunc` (cofactor expansion)."""
    n = len(M)
    if n == 1:
        return M[0][0]
    if n == 2:
        return M[0][0] * M[1][1] - M[0][1] * M[1][0]
    out = RatFunc.zero()
    for j in range(n):
        minor = [row[:j] + row[j + 1:] for row in M[1:]]
        cof = _det_rf(minor)
        term = M[0][j] * cof
        out = out + (term if j % 2 == 0 else -term)
    return out


def _det_span(det_lp: LaurentPoly) -> int:
    """q-span (max-min exponent) of a LaurentPoly det; +inf if zero."""
    if det_lp.is_zero():
        return 10 ** 9
    ks = det_lp._coeffs.keys()
    return max(ks) - min(ks)


# ---------------------------------------------------------------------------
# Analytic anchor solver.
# ---------------------------------------------------------------------------

_anchor_cache: dict = {}


def _anchor_symtr(m: int, e_anchor: int) -> SymTr:
    """Exact `Tr(L_{(m, e_anchor)})` as a SymTr over the Wilson seeds.

    m = 0: the Wilson seed itself.  m >= 1: solved from the m×m
    ρ²-cyclicity system with `RatFunc` (exact `Q(q)`) coefficients.
    """
    if m == 0:
        return _seed_symtr(e_anchor)
    if m == 1:
        # The level-1 cyclicity system is vacuous (Tr H_c = Tr H_{c-8} is
        # just ρ²-invariance); the genuine datum is the 2-letter bridge
        #   Tr H_0 = (q·Tr W_0 + q^{-1}·Tr W_2) / (1 - q²),
        # derived in `pure_su2_layer2_identities.tr_h0_bridge` and verified
        # against the Schur trace.  We carry it here in exact Q(q) form.
        inv_1mq2 = RatFunc(0, {0: Fraction(1)}, {0: Fraction(1), 2: Fraction(-1)})
        c0 = RatFunc(1, {0: Fraction(1)}, {0: Fraction(1)}) * inv_1mq2   # q/(1-q²)
        c2 = RatFunc(-1, {0: Fraction(1)}, {0: Fraction(1)}) * inv_1mq2  # q^{-1}/(1-q²)
        return SymTr.seed(0).scale(c0) + SymTr.seed(2).scale(c2)
    key = (m, e_anchor)
    if key in _anchor_cache:
        return _anchor_cache[key]
    sols = _solve_level(m)
    for j, e in enumerate(range(0, 2 * m, 2)):
        _anchor_cache[(m, e)] = sols[j]
    return _anchor_cache[key]


def _best_prefix(m: int):
    """Among the candidate prefixes, return `(prefix, M_rf, expansions)`
    for the non-singular matrix with the smallest determinant q-span
    (the cheapest exact solve).  Falls back to first non-singular."""
    best = None
    for gen in _PREFIX_CANDIDATES_BASE:
        prefix = gen(m)
        if len(prefix) != m - 1:
            continue
        M_lp, expansions = _build_M_for_prefix(m, prefix)
        det_lp = _det_n(M_lp)
        if det_lp.is_zero():
            continue
        span = _det_span(det_lp)
        if best is None or span < best[0]:
            M_rf = [[_lp_to_rf(x) for x in row] for row in M_lp]
            best = (span, prefix, M_rf, expansions)
    if best is None:
        raise RuntimeError(f"_best_prefix(m={m}): no non-singular prefix")
    return best[1], best[2], best[3]


def _solve_level(m: int, prefix_override=None) -> list:
    """Solve the m-anchor cyclicity system exactly over Q(q).

    Returns `[SymTr]` for anchors e = 0, 2, ..., 2m-2.
    """
    if prefix_override is None:
        prefix, M_rf, expansions = _best_prefix(m)
    else:
        M_lp, expansions = _build_M_for_prefix(m, prefix_override)
        if _det_n(M_lp).is_zero():
            raise RuntimeError(f"_solve_level: prefix {prefix_override} singular")
        M_rf = [[_lp_to_rf(x) for x in row] for row in M_lp]

    # RHS vector b (SymTr): lower-m contributions, moved to the RHS.
    #   cyclicity:  Σ_{m'=m} (...) X  =  -Σ_{m'<m} (...) Tr(seed_{m'}).
    b = []
    for (c, lhs, rhs) in expansions:
        const = SymTr.zero()
        for terms, sign in ((lhs, 1), (rhs, -1)):
            for (m_, e_), coef in terms.items():
                if e_ % 2 == 1 or m_ >= m:
                    continue
                # lower-m trace, recursively (exact).
                lo = _anchor_symtr(m_, e_ % (2 * m_)) if m_ >= 1 else _seed_symtr(e_)
                rf = _lp_to_rf(coef)
                contrib = lo.scale(rf if sign == 1 else (-rf))
                const = const + contrib
        b.append(const.scale(RatFunc(0, {0: Fraction(-1)}, {0: Fraction(1)})))

    # Cramer over Q(q): X_j = det(M with col j -> b) / det(M).
    det = _det_rf(M_rf)
    inv_det = RatFunc(-det.shift, det.den, det.num)   # exact 1/det
    sols = []
    n = m
    for j in range(n):
        # det of matrix with column j replaced by SymTr vector b:
        # expand along column j -> Σ_i (±) b[i] · cofactor_ij  (SymTr).
        col_det = SymTr.zero()
        for i in range(n):
            minor = [M_rf[r][:j] + M_rf[r][j + 1:]
                     for r in range(n) if r != i]
            cof = _det_rf(minor)
            sign = 1 if (i + j) % 2 == 0 else -1
            col_det = col_det + b[i].scale(cof if sign == 1 else (-cof))
        sols.append(col_det.scale(inv_det))
    return sols


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------

def trace_symbolic(m: int, e: int) -> SymTr:
    """Exact `Q(q)`-reduction of `Tr(L_{(m,e)})` to the Wilson seeds.

    Z₂ vanishing (odd e → 0) and H-shift folding (e → e mod 2m) applied.
    """
    if e % 2 == 1:
        return SymTr.zero()
    if m == 0:
        return _seed_symtr(e)
    return _anchor_symtr(m, e % (2 * m))


def trace_series(m: int, e: int, q_max: int = 30,
                  wilson_series=None) -> LaurentPoly:
    """Numeric trace: reduce analytically, then substitute Schur Wilson
    seeds at the very end.  `wilson_series(e, q_max)` defaults to the
    Schur `tr_W` from `pure_su2_h_trace`."""
    if wilson_series is None:
        from pure_su2_h_trace import tr_W
        wilson_series = tr_W
    return trace_symbolic(m, e).to_series(wilson_series, q_max)


def prefix_independence_ok(m: int) -> bool:
    """Exact check: solving level m with two distinct non-singular
    prefixes gives identical `Q(q)` anchors (no truncation)."""
    found = []
    for gen in _PREFIX_CANDIDATES_BASE:
        prefix = gen(m)
        if len(prefix) != m - 1:
            continue
        M_lp, _ = _build_M_for_prefix(m, prefix)
        if _det_n(M_lp).is_zero():
            continue
        found.append(prefix)
        if len(found) == 2:
            break
    if len(found) < 2:
        return True   # only one non-singular prefix; nothing to compare
    _anchor_cache.clear()
    a = _solve_level(m, prefix_override=found[0])
    _anchor_cache.clear()
    b = _solve_level(m, prefix_override=found[1])
    _anchor_cache.clear()
    return all(x.equals(y) for x, y in zip(a, b))
