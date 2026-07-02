"""Axiom-derived trace for `PureSU2KAlg`.

Computes `Tr(L_{(m, e)})` as a q-series WITHOUT consulting a BPS
realisation, using only the SU(2) Schur closed form for
Wilson characters and ρ²-twisted cyclicity for the rest.

Ingredients
-----------

1.  **Tr(W_n) for all n ≥ 0** — `tr_W`.  Read off as the χ_n Schur
    coefficient in the SU(2) Schur closed form

        F(v) = (q²v²; q²)_∞² · (q²v⁻²; q²)_∞² · (q²; q²)_∞²

    i.e. `Tr(W_n) = [v^n] F − [v^{n+2}] F`.

2.  **Tr(H_0) — `tr_H0`**.  Cyclicity bridge formula

        Tr(H_0) = −q^{−1}·Tr(W_0)
                + [(2q² − 1)·Tr(W_2) + Tr(W_4)] / [2q³(1 − q²)]

    derived from `Tr(H_{−a}·H_a − q^{2a}·H_0²)` cyclicity at a = 1, 2
    (`pure_su2_layer2_identities.tr_h0_bridge`).

3.  **m=2 anchors — `tr_H0sq`, `tr_H1sq`**.  Same shape as `tr_H0`
    but from 2-letter cyclicity at `(a, b) ∈ {(-1, 1), (0, 2)}`;
    bridge formulas in `tr_h0sq_bridge`, `tr_h1sq_bridge`.

4.  **m=3 anchors — `tr_H0cube`, `tr_L32`, `tr_L34`**.  3-letter
    cyclicity at triples `(-1, 0, 1)`, `(-1, 0, 3)`, `(-1, 0, 5)`;
    3×3 system solved in `solve_m3_anchors`.

5.  **m ≥ 4 anchors — `_tr_anchor`**.  Generic m-letter cyclicity
    system solved by `solve_anchors_via_cyclicity` in
    `pure_su2_layer2_identities`.  Lower-m anchors are recursively
    consumed at precision tuned to the actual determinant leading
    order (`_det_leading_order(m)`), so the buffer is exactly tight.

6.  **H-shift symmetry + Z₂**.  Tr(L_{(m, e)}) is invariant under
    `e → e + 2m`, and vanishes for odd e (PSU(2) = SU(2)/Z₂ projects
    out half-integer reps).  `trace_pSU2_label(m, e, q_max)` performs
    this canonicalisation before dispatching to the m-anchor solver.
"""
from __future__ import annotations
from fractions import Fraction

from laurent_poly import LaurentPoly
from pure_su2_layer2_identities import (
    tr_h0_bridge, tr_h0sq_bridge, tr_h1sq_bridge, solve_m3_anchors,
    solve_anchors_via_cyclicity, find_nonsingular_prefix,
)
from pure_su2_layer2_identities import _det_n as _layer2_det_n


# ---------- SU(2) Schur F(v) for Tr(W_n) ------------------------------

_Q_MAX_DEFAULT = 30
_V_MAX_DEFAULT = 32

_F_cache: dict = {}


def _f_mul(f1: dict, f2: dict, v_max: int, q_max: int) -> dict:
    out = {}
    for v1, p1 in f1.items():
        if abs(v1) > v_max: continue
        for v2, p2 in f2.items():
            v_total = v1 + v2
            if abs(v_total) > v_max: continue
            prod = p1 * p2
            trunc = {e: c for e, c in prod._coeffs.items() if e <= q_max}
            prod = LaurentPoly(trunc)
            if prod.is_zero(): continue
            out[v_total] = out.get(v_total, LaurentPoly.zero()) + prod
    return {k: v for k, v in out.items() if not v.is_zero()}


def _pochhammer(a_dict: dict, q_max: int, v_max: int) -> dict:
    """(a; q²)_∞ truncated to q^{q_max}."""
    result = {0: LaurentPoly({0: 1})}
    k = 0
    while 2 * k <= q_max:
        a_shifted = {v: p * LaurentPoly({2 * k: -1}) for v, p in a_dict.items()}
        factor = {0: LaurentPoly({0: 1})}
        for v, p in a_shifted.items():
            factor[v] = factor.get(v, LaurentPoly.zero()) + p
        factor = {k_: v for k_, v in factor.items() if not v.is_zero()}
        result = _f_mul(result, factor, v_max, q_max)
        k += 1
    return result


def _power_dict(f: dict, n: int, v_max: int, q_max: int) -> dict:
    out = {0: LaurentPoly({0: 1})}
    for _ in range(n):
        out = _f_mul(out, f, v_max, q_max)
    return out


def _build_F(q_max: int, v_max: int) -> dict:
    """SU(2) Schur F(v) = (q²v²; q²)_∞² · (q²v⁻²; q²)_∞² · (q²; q²)_∞²."""
    key = (q_max, v_max)
    if key in _F_cache:
        return _F_cache[key]
    P1 = _pochhammer({2: LaurentPoly({2: 1})}, q_max, v_max)
    P2 = _pochhammer({-2: LaurentPoly({2: 1})}, q_max, v_max)
    P3 = _pochhammer({0: LaurentPoly({2: 1})}, q_max, v_max)
    F = _f_mul(_power_dict(P1, 2, v_max, q_max),
               _power_dict(P2, 2, v_max, q_max), v_max, q_max)
    F = _f_mul(F, _power_dict(P3, 2, v_max, q_max), v_max, q_max)
    _F_cache[key] = F
    return F


def tr_W(n: int, q_max: int = _Q_MAX_DEFAULT) -> LaurentPoly:
    """Tr(W_n) = [v^n] F(v) − [v^{n+2}] F(v) as a q-Laurent polynomial
    truncated to q^{q_max}."""
    v_max = max(_V_MAX_DEFAULT, n + 6)
    F = _build_F(q_max, v_max)
    a = F.get(n, LaurentPoly.zero())
    b = F.get(n + 2, LaurentPoly.zero())
    out = dict(a._coeffs)
    for e, c in b._coeffs.items():
        out[e] = out.get(e, 0) - c
    return LaurentPoly({e: c for e, c in out.items() if c != 0})


def tr_H0(q_max: int = _Q_MAX_DEFAULT) -> LaurentPoly:
    """Tr(H_0) via the cyclicity bridge formula."""
    W0 = tr_W(0, q_max)
    W2 = tr_W(2, q_max)
    W4 = tr_W(4, q_max)
    return tr_h0_bridge(W0, W2, W4, q_max=q_max)


# Buffer for the m=2 bridge's geometric-series inversion: numerator
# accumulates up to q^{5} shifts of the elementary traces, so we
# compute inputs at q_max + 8 internally and truncate at the end.
_M2_BRIDGE_BUFFER = 8


def tr_H0sq(q_max: int = _Q_MAX_DEFAULT) -> LaurentPoly:
    """Tr(L_{(2, 0)}) = Tr(H_0²) via the m=2 anchor bridge."""
    qm = q_max + _M2_BRIDGE_BUFFER
    W0 = tr_W(0, qm)
    W2 = tr_W(2, qm)
    H0 = tr_H0(qm)
    return tr_h0sq_bridge(W0, W2, H0, q_max=q_max)


def tr_H1sq(q_max: int = _Q_MAX_DEFAULT) -> LaurentPoly:
    """Tr(L_{(2, 2)}) = Tr(H_1²) via the m=2 anchor bridge."""
    qm = q_max + _M2_BRIDGE_BUFFER
    W0 = tr_W(0, qm)
    W2 = tr_W(2, qm)
    H0 = tr_H0(qm)
    return tr_h1sq_bridge(W0, W2, H0, q_max=q_max)


_M3_BRIDGE_BUFFER = 24                                # det = q^24(1-q^6)
_m3_cache: dict = {}


def _tr_m3_anchors(q_max: int):
    """Memoised: returns (TrL30, TrL32, TrL34) at the given precision."""
    if q_max in _m3_cache:
        return _m3_cache[q_max]
    qm = q_max + _M3_BRIDGE_BUFFER
    traces = {
        'TrH0':   tr_H0(qm),
        'TrH0sq': tr_H0sq(qm),
        'TrH1sq': tr_H1sq(qm),
    }
    # Wilson traces up to q^16 (defensive — multiply expansion produces
    # χ_e terms up to roughly the gap between letters).
    for e in range(0, 18, 2):
        traces[f'TrW_{e}'] = tr_W(e, qm)
    result = solve_m3_anchors(traces, q_max=qm)
    # Truncate to the requested q_max for each anchor.
    out = tuple(
        LaurentPoly({e: c for e, c in s._coeffs.items() if e <= q_max})
        for s in result
    )
    _m3_cache[q_max] = out
    return out


def tr_H0cube(q_max: int = _Q_MAX_DEFAULT) -> LaurentPoly:
    """Tr(L_{(3, 0)}) = Tr(H_0³) via the m=3 anchor cyclicity bridge."""
    return _tr_m3_anchors(q_max)[0]


def tr_L32(q_max: int = _Q_MAX_DEFAULT) -> LaurentPoly:
    """Tr(L_{(3, 2)}) via the m=3 anchor cyclicity bridge."""
    return _tr_m3_anchors(q_max)[1]


def tr_L34(q_max: int = _Q_MAX_DEFAULT) -> LaurentPoly:
    """Tr(L_{(3, 4)}) via the m=3 anchor cyclicity bridge."""
    return _tr_m3_anchors(q_max)[2]


# -------------------------------------------------------------------
# Generic m-anchor recursion (all m).
# -------------------------------------------------------------------
#
# `_tr_anchor(m, e_anchor, q_max)` recursively computes all anchors
# `Tr(L_{(m, e_anchor)})` using:
#   * m=0, 1, 2: closed-form bridges (tr_W, tr_H0, tr_H0sq, tr_H1sq).
#   * m=3: 3-letter cyclicity (specific solver).
#   * m>=4: generic m-letter cyclicity (solve_anchors_via_cyclicity)
#     with tuples (0, 0, ..., 0, c).
#
# Precision: the generic solver inverts `det M` which has growing
# leading order with m; we buffer the recursive inputs accordingly.

# Precision buffer for the m-anchor cyclicity solve: we need the
# lower-m elementary traces accurate to `q_max + |k0(det M)| + slack`
# so that `num · inv_det` truncated at `q^q_max` is exact.  k0 is
# determined dynamically by computing the (exact, lower-m-independent)
# matrix M and reading off the lowest-order term of det M.
_DET_LEAD_CACHE: dict = {}


def _det_leading_order(m: int) -> int:
    """Lowest q-exponent of det(M) for the m-anchor cyclicity matrix."""
    if m in _DET_LEAD_CACHE:
        return _DET_LEAD_CACHE[m]
    _prefix, M, _exp = find_nonsingular_prefix(m)
    det = _layer2_det_n(M)
    k0 = min(det._coeffs.keys())
    _DET_LEAD_CACHE[m] = k0
    return k0


def _buffer_for(m: int) -> int:
    # Lower-m traces are computed at q_max + buffer; for m<=3 we use a
    # closed-form bridge already buffered locally (8 suffices).  For
    # m>=4 the generic solver needs the lower-m precision tuned to the
    # actual det leading order, plus a slack of 8.
    if m <= 3:
        return 8
    return _det_leading_order(m) + 8


_anchor_cache: dict = {}


def _tr_anchor(m: int, e_anchor: int, q_max: int) -> LaurentPoly:
    """`Tr(L_{(m, e_anchor)})` as a truncated Laurent series in q.

    `e_anchor` is the H-shift canonical representative in
    `{0, 2, ..., 2m-2}` for `m >= 1`, or `abs(e_anchor)` for `m = 0`.
    Memoised per `(m, e_anchor, q_max)`.
    """
    key = (m, e_anchor, q_max)
    if key in _anchor_cache:
        return _anchor_cache[key]
    if m == 0:
        result = tr_W(abs(e_anchor), q_max)
    elif m == 1:
        result = tr_H0(q_max)
    elif m == 2:
        result = tr_H0sq(q_max) if e_anchor == 0 else tr_H1sq(q_max)
    elif m == 3:
        idx = (e_anchor % 6) // 2
        result = _tr_m3_anchors(q_max)[idx]
    else:
        # m >= 4: generic cyclicity solver.
        result = _solve_anchors_at(m, q_max)[e_anchor]
    _anchor_cache[key] = result
    return result


def _solve_anchors_at(m: int, q_max: int) -> dict:
    """Compute all m-anchors at precision `q_max`, recursing into lower m."""
    qm = q_max + _buffer_for(m)
    traces_lp: dict = {}
    # Wilson traces (defensive ceiling: enough for the m-letter expansion).
    for e in range(0, 4 * m + 4, 2):
        traces_lp[f'TrW_{e}'] = tr_W(e, qm)
    # Lower-m anchors.
    for m_lo in range(1, m):
        for e_anc in range(0, 2 * m_lo, 2):
            traces_lp[f'TrL{m_lo}_{e_anc}'] = _tr_anchor(m_lo, e_anc, qm)
    sols = solve_anchors_via_cyclicity(m, traces_lp, qm)
    # Truncate to the requested q_max.
    return {
        e_anc: LaurentPoly({e: c for e, c in lp._coeffs.items() if e <= q_max})
        for e_anc, lp in sols.items()
    }


# ---------- Reduce (m, e) to {Tr(W_n), Tr(H_0)} -----------------------

def trace_pSU2_label(m: int, e: int, q_max: int = _Q_MAX_DEFAULT
                      ) -> LaurentPoly:
    """Axiom-derived trace of the pSU2 canonical-basis seed `(m, e)`.

    Z₂ vanishing: odd `e` → `0`.  H-shift canonicalisation: even `e`
    is folded to `e mod 2m` and dispatched to the appropriate m-anchor
    bridge (`tr_W` for m=0, `tr_H0` for m=1, `tr_H0sq` / `tr_H1sq` for
    m=2, the m=3 specific solver for m=3, the generic m-anchor solver
    `_tr_anchor` for m ≥ 4).
    """
    if e % 2 == 1:
        return LaurentPoly.zero()
    if m == 0:
        return tr_W(e, q_max)
    if m == 1:
        # By H-shift, Tr((1, e)) = Tr(H_0) for even e.
        return tr_H0(q_max)
    if m == 2:
        # H-shift anchor: Tr(L_{(2, e)}) only depends on e mod 4 (since
        # H-shift sends e → e + 4 for m=2).  Two even-e anchors:
        #   e ≡ 0 (mod 4): Tr(H_0²) via tr_h0sq_bridge.
        #   e ≡ 2 (mod 4): Tr(H_1²) via tr_h1sq_bridge.
        if e % 4 == 0:
            return tr_H0sq(q_max)
        return tr_H1sq(q_max)
    # General m: H-shift e → e + 2m.  Even-e anchors are e mod 2m
    # ∈ {0, 2, ..., 2m-2}.
    e_anchor = e % (2 * m)
    return _tr_anchor(m, e_anchor, q_max)
