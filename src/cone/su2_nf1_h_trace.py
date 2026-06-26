"""Axiom-derived trace for `SU2Nf1KAlgebra`.

Mirrors `pure_su2_h_trace.py` with U(1)_F flavour μ enriching every
ingredient.  The trace lives in `RPowerSeries[AbelianZPlusRing(rank=1)]`
— q-series with `μ`-polynomial coefficients.

Ingredients (TBD — bracketed items pending derivation):

1.  **Tr(W_n, μ)** — Schur F(v, μ) for the SU(2)+Nf=1 Schur index.
    Pure-SU(2) had

        F(v) = (q² v²; q²)_∞² · (q² v^{-2}; q²)_∞² · (q²; q²)_∞²

    For Nf=1 with one fundamental hyper, an extra hypermultiplet
    Schur factor is needed — TBD which exact form (likely
    `(μ v; q²)_∞ · (μ v^{-1}; q²)_∞` or its inverse, depending on
    chirality convention).

2.  **Tr(H_0, μ)** — cyclicity bridge formula from `ρ²(H_n) = μ^{-2}
    · H_{n-6}` evaluated at small a.

3.  **m=2 anchors** — `Tr(H_0²)`, `Tr(H_0·H_1)`, `Tr(H_1²)` from
    2-letter cyclicity bridges (now 3 anchors instead of 2 for pure
    SU(2), because the rank-2 cone structure has period 1 with 3
    inequivalent residues mod ρ-shift-3).

4.  **m ≥ 3** — generic cyclicity system (analogous to pure-SU(2)'s
    `solve_anchors_via_cyclicity`) but with `RLaurent[R]` coefs.

5.  **H-shift symmetry** — `Tr((m, e+6, μ_p)) = Tr((m, e, μ_p + 2m))`
    (cyclicity ρ on the H-tower acts as `e → e+6, μ_p → μ_p + 2m`).

For now this module is a STUB; the cyclicity setup is plumbed but
the Schur F(v, μ) needs user input on the matter contribution form.
"""
from __future__ import annotations
from fractions import Fraction

from laurent_poly import LaurentPoly
from zplus_ring import AbelianZPlusRing, RLaurent, RPowerSeries


_R = AbelianZPlusRing(rank=1)
_Q_MAX_DEFAULT = 30
_V_MAX_DEFAULT = 32


# ---------- RLaurent helpers ----------------------------------------

def _rl_one() -> RLaurent:
    return RLaurent(_R, {0: _R.one()})


def _rl_zero() -> RLaurent:
    return RLaurent(_R, {})


def _rl_q_mu(q_pow: int, mu_pow: int) -> RLaurent:
    """`q^{q_pow} · μ^{mu_pow}` as `RLaurent[R]`."""
    return RLaurent(_R, {q_pow: _R.basis_element((mu_pow,))})


def _rl_truncate_qmu(r: RLaurent, q_max: int) -> RLaurent:
    return RLaurent(_R, {e: c for e, c in r.coeffs.items() if e <= q_max})


_F_cache: dict = {}


def _f_mul(f1: dict, f2: dict, v_max: int, q_max: int) -> dict:
    """Multiply two `{v_exp: RLaurent}` dicts, truncating at |v| ≤ v_max
    and q-exp ≤ q_max."""
    out: dict = {}
    for v1, p1 in f1.items():
        if abs(v1) > v_max:
            continue
        for v2, p2 in f2.items():
            v_total = v1 + v2
            if abs(v_total) > v_max:
                continue
            prod = _rl_truncate_qmu(p1 * p2, q_max)
            if prod.is_zero():
                continue
            existing = out.get(v_total, _rl_zero())
            out[v_total] = existing + prod
    return {k: v for k, v in out.items() if not v.is_zero()}


def _pochhammer_qsq(a_dict: dict, q_max: int, v_max: int) -> dict:
    """`(a; q²)_∞ = Π_{k≥0} (1 - a · q^{2k})`, truncated to `q^{q_max}`.

    `a_dict = {v_exp: RLaurent}` represents `a` (a v-Laurent polynomial
    with q-Laurent coefficients in R).
    """
    result = {0: _rl_one()}
    k = 0
    while 2 * k <= q_max:
        # Factor: 1 - a · q^{2k}.
        q2k = _rl_q_mu(2 * k, 0)
        # a · q^{2k} contribution: scale a_dict by q^{2k}, then negate.
        factor: dict = {0: _rl_one()}
        for v_exp, p in a_dict.items():
            shifted = q2k * p
            factor[v_exp] = factor.get(v_exp, _rl_zero()) - shifted
        factor = {k_: v for k_, v in factor.items() if not v.is_zero()}
        result = _f_mul(result, factor, v_max, q_max)
        k += 1
    return result


def _pochhammer_inverse_qsq(a_dict: dict, q_max: int, v_max: int) -> dict:
    """`1 / (a; q²)_∞ = Π_{k≥0} 1/(1 - a · q^{2k})`, truncated to
    `q^{q_max}`.

    Each factor `1/(1 - a·q^{2k})` is expanded as a geometric series
    `Σ_{n≥0} (a · q^{2k})^n` truncated.
    """
    result = {0: _rl_one()}
    k = 0
    while 2 * k <= q_max:
        # Build geometric series sum_n (a·q^{2k})^n truncated.
        factor: dict = {0: _rl_one()}
        term: dict = {0: _rl_one()}
        q2k = _rl_q_mu(2 * k, 0)
        n = 0
        while True:
            # Multiply term by a · q^{2k}.
            new_term: dict = {}
            for v_exp, p in term.items():
                for v_a, p_a in a_dict.items():
                    nv = v_exp + v_a
                    if abs(nv) > v_max:
                        continue
                    contrib = _rl_truncate_qmu(p * p_a * q2k, q_max)
                    if contrib.is_zero():
                        continue
                    existing = new_term.get(nv, _rl_zero())
                    new_term[nv] = existing + contrib
            new_term = {k_: v for k_, v in new_term.items() if not v.is_zero()}
            if not new_term:
                break
            term = new_term
            for v_exp, p in term.items():
                existing = factor.get(v_exp, _rl_zero())
                factor[v_exp] = existing + p
            n += 1
            if n > q_max + 4:
                break                            # safety
        factor = {k_: v for k_, v in factor.items() if not v.is_zero()}
        result = _f_mul(result, factor, v_max, q_max)
        k += 1
    return result


def _power_dict(f: dict, n: int, v_max: int, q_max: int) -> dict:
    out = {0: _rl_one()}
    for _ in range(n):
        out = _f_mul(out, f, v_max, q_max)
    return out


def _build_F(q_max: int, v_max: int) -> dict:
    """SU(2)+Nf=1 Schur F(v, μ).

        F(v, μ) = (q²v²; q²)_∞² · (q²v⁻²; q²)_∞² · (q²; q²)_∞²
                  / [(q μ v; q²)_∞ · (q μ v⁻¹; q²)_∞
                     · (q μ⁻¹ v; q²)_∞ · (q μ⁻¹ v⁻¹; q²)_∞]

    Pure-SU(2) numerator (vector multiplet) over four `(q μ^± v^±; q²)_∞`
    factors from the one Nf=1 fundamental hyper (doublet × two-flavor
    weights ±1 in U(1)_F).
    """
    key = (q_max, v_max)
    if key in _F_cache:
        return _F_cache[key]
    # Pure SU(2) numerator factors.
    P_v2  = _pochhammer_qsq({2:  _rl_q_mu(2, 0)},  q_max, v_max)   # (q² v²; q²)
    P_v_2 = _pochhammer_qsq({-2: _rl_q_mu(2, 0)},  q_max, v_max)   # (q² v⁻²; q²)
    P_0   = _pochhammer_qsq({0:  _rl_q_mu(2, 0)},  q_max, v_max)   # (q²;    q²)
    F = _f_mul(_power_dict(P_v2,  2, v_max, q_max),
               _power_dict(P_v_2, 2, v_max, q_max), v_max, q_max)
    F = _f_mul(F, _power_dict(P_0, 2, v_max, q_max), v_max, q_max)
    # Nf=1 hyper: four `(−q μ^σ v^τ; q²)_∞` denominator factors.  The
    # `−q` (vs `+q`) gives the v ↔ −v parity that flips Wilson-line W_n
    # by `(−1)^n` between the SU(2)-character convention and the
    # BPS-tropical-charge convention.
    for mu_sign in (+1, -1):
        for v_sign in (+1, -1):
            arg = {v_sign: -_rl_q_mu(1, mu_sign)}  # −q · μ^{mu_sign} · v^{v_sign}
            inv = _pochhammer_inverse_qsq(arg, q_max, v_max)
            F = _f_mul(F, inv, v_max, q_max)
    _F_cache[key] = F
    return F


def tr_W(n: int, q_max: int = _Q_MAX_DEFAULT) -> RLaurent:
    """`Tr(W_n) = [v^n] F − [v^{n+2}] F` for SU(2)+Nf=1.

    Returns `RLaurent[R]` (q-series with μ-polynomial R-coefficients).
    """
    v_max = max(_V_MAX_DEFAULT, n + 6)
    F = _build_F(q_max, v_max)
    a = F.get(n, _rl_zero())
    b = F.get(n + 2, _rl_zero())
    return _rl_truncate_qmu(a - b, q_max)


def tr_H0(q_max: int = _Q_MAX_DEFAULT):
    """Tr(H_0) via cyclicity bridge from `ρ²(H_n) = μ^{-2}·H_{n-6}`.

    PLACEHOLDER pending bridge derivation.

    Note: Z₂ projection is BROKEN in Nf=1 (Tr(W_1) is non-zero: μ + μ^{-1}
    at q^1).  Fundamental matter brings half-integer SU(2) reps back, so
    odd-e traces don't vanish.  This means there are SIX m=1 anchors
    (one per `n mod 6` via the ρ-shift-6 / μ²-cycle ρ²) instead of just
    Tr(H_0) and zeroes as in pure SU(2).  Bridge formulas will be a
    larger linear system.

    Structural lemma (V_n self-cancellation): cyclicity at
    `(a, b) = (-1, 1)` produces `q·V_0 + q²·V_0^{m=2} = ...` with V_0
    appearing only on LHS — clean for V_0.  Cyclicity at
    `(a, b) = (0, 2)` produces V_1 on both sides with identical
    coefficient `μ q`, so V_1 self-cancels and the equation determines
    V_2^{m=2} cleanly.  Similar self-cancellations should isolate
    individual m=1 anchors when paired with the right cyclicity (a, b).
    """
    raise NotImplementedError(
        "tr_H0: SU(2)+Nf=1 cyclicity bridge pending "
        "(6 m=1 + 12 m=2 = 18 coupled anchors, structural cancellations "
        "identified)"
    )


_V1_V2_CACHE: dict = {}


def _solve_v1_v2_anchors(q_max: int):
    """Cyclicity-Schur reduction for V1 (period 2) and V2 (raw e) anchors.

    Combines the two user-prescribed tricks over a wide range of (a, b)
    to span all m=2 anchors `(2, e)` that appear (positive AND negative
    e — they're distinct since Tr(L_(2, e)) is sign-asymmetric).

    Cached per `q_max`.
    """
    if q_max in _V1_V2_CACHE:
        return _V1_V2_CACHE[q_max]
    from qmu_laurent import QmuLaurent

    pairs = [(-1, 1), (-2, 2), (-3, 3),                  # first trick
             (-3, 4), (-2, 3), (-1, 2), (0, 1)]          # second trick
    eqs = [_cyclicity_equation(a, b, q_max=q_max) for (a, b) in pairs]
    anchors = sorted({k for ea, _ in eqs for k in ea},
                     key=lambda x: (x[0], x[1]))
    K = len(anchors); N = len(eqs)
    M = [[QmuLaurent.zero()] * K for _ in range(N)]
    b = [QmuLaurent.zero() for _ in range(N)]
    for i, (ea, ec) in enumerate(eqs):
        for j, anc in enumerate(anchors):
            M[i][j] = QmuLaurent.from_rlaurent(ea.get(anc, _rl_zero()))
        b[i] = QmuLaurent.from_rlaurent(ec)

    inner_q = q_max + 12
    row = 0
    for col in range(K):
        pivot = next((r for r in range(row, N) if not M[r][col].is_zero()),
                     None)
        if pivot is None:
            continue
        if pivot != row:
            M[row], M[pivot] = M[pivot], M[row]
            b[row], b[pivot] = b[pivot], b[row]
        inv_p = M[row][col].series_inverse(inner_q)
        for j in range(K):
            M[row][j] = (M[row][j] * inv_p).truncate(inner_q)
        b[row] = (b[row] * inv_p).truncate(inner_q)
        for r in range(N):
            if r == row or M[r][col].is_zero():
                continue
            f = M[r][col]
            for j in range(K):
                M[r][j] = (M[r][j] - f * M[row][j]).truncate(inner_q)
            b[r] = (b[r] - f * b[row]).truncate(inner_q)
        row += 1
        if row >= N:
            break

    sols: dict = {}
    for i in range(row):
        pc = next((j for j in range(K) if not M[i][j].is_zero()), None)
        if pc is not None:
            sols[anchors[pc]] = b[i].truncate(q_max)
    _V1_V2_CACHE[q_max] = sols
    return sols


def _qmu_to_rl(qmu, q_max: int) -> RLaurent:
    """Convert a `QmuLaurent` (q-Laurent with `RatFuncMu` coefs) back to
    an `RLaurent` (q-Laurent with R-element coefs), truncated to q^q_max.

    Assumes each `RatFuncMu` coefficient is a μ-polynomial (denominator
    `{0: 1}`); for finite Schur traces this holds after the recursion.
    """
    out = {}
    for q_e, rf in qmu.coeffs.items():
        if q_e > q_max:
            continue
        if not rf.num:
            continue
        if rf.den != {0: __import__('fractions').Fraction(1)}:
            # Non-trivial denominator — bail out; recursion needs more
            # equations or higher precision.
            raise ValueError(
                f"_qmu_to_rl: anchor coefficient at q^{q_e} has non-trivial "
                f"μ-denominator {rf.den} — recursion is under-determined or "
                f"truncated."
            )
        # Build R-element with the rational μ-numerator.
        coef = _R.zero()
        for k, v in rf.num.items():
            if v == 0:
                continue
            # μ-power index = k + rf.shift.
            mu_pow = k + rf.shift
            from fractions import Fraction
            if not isinstance(v, Fraction):
                v = Fraction(v)
            if v.denominator != 1:
                raise ValueError(
                    f"_qmu_to_rl: non-integer μ-coefficient {v} at q^{q_e}"
                )
            coef = coef + _R.basis_element((mu_pow,)) * int(v)
        if not coef.is_zero():
            out[q_e] = out.get(q_e, _R.zero()) + coef
    return RLaurent(_R, out)


_Vm_CACHE: dict = {}


def _solve_vm_anchors(m: int, q_max: int):
    """Generic cyclicity-Schur solver for `Vm` anchors at any m ≥ 2.

    Builds m+1 diagonal m-letter cyclicity equations via tuples
    `(0, 0, …, 0, c)` for c = 0, 1, …, m — each picks out a unique
    V{m}_k anchor on the LHS with low-q-power coefficient.

    Recursively solves lower-m anchors first (cached) and substitutes
    them into the m-letter equations.  Inflated `q_max + buffer` for
    lower-m calls so precision survives multiplication by high q-power
    coefficients on Vm anchors.

    Cached per `(m, q_max)`.
    """
    key = (m, q_max)
    if key in _Vm_CACHE:
        return _Vm_CACHE[key]
    from qmu_laurent import QmuLaurent

    # Recursively solve lower-m anchors at INFLATED precision so that
    # substituted values survive the high q-power Vm equation coefs.
    buffer = 12 + 4 * m
    if m == 2:
        lower = _solve_v1_v2_anchors(q_max + buffer)
    else:
        lower = _solve_vm_anchors(m - 1, q_max + buffer)

    # m+1 cone-aligned tuples: tuple_k = [0]*(m-k) + [1]*k.  Each is the
    # rank-2 cone monomial H_0^{m-k} · H_1^{k}, giving LHS = single seed
    # (m, k) → V{m}_k with low-q-power coef.  Diagonal under reflective
    # fold k ↔ 2m-k since k ∈ {0..m}.
    tuples = [tuple([0] * (m - k) + [1] * k) for k in range(m + 1)]
    eqs = [_cyclicity_equation_multi(t, q_max=q_max + buffer) for t in tuples]

    # Substitute lower-m anchors into each equation.
    reduced_eqs = []
    for (eq_a, eq_c) in eqs:
        new_eq_a = {}
        new_const = QmuLaurent.from_rlaurent(eq_c)
        for anc, coef_rl in eq_a.items():
            if anc in lower:
                coef = QmuLaurent.from_rlaurent(coef_rl)
                new_const = new_const - (coef * lower[anc])
            else:
                new_eq_a[anc] = coef_rl
        reduced_eqs.append((new_eq_a, new_const))

    anchors = sorted({k for ea, _ in reduced_eqs for k in ea},
                     key=lambda x: (x[0], x[1]))
    K = len(anchors); N = len(reduced_eqs)
    M = [[QmuLaurent.zero()] * K for _ in range(N)]
    b = [QmuLaurent.zero() for _ in range(N)]
    for i, (ea, ec) in enumerate(reduced_eqs):
        for j, anc in enumerate(anchors):
            M[i][j] = QmuLaurent.from_rlaurent(ea.get(anc, _rl_zero()))
        b[i] = ec.truncate(q_max + buffer)

    inner_q = q_max + buffer
    row = 0
    for col in range(K):
        pivot = next((r for r in range(row, N) if not M[r][col].is_zero()),
                     None)
        if pivot is None:
            continue
        if pivot != row:
            M[row], M[pivot] = M[pivot], M[row]
            b[row], b[pivot] = b[pivot], b[row]
        inv_p = M[row][col].series_inverse(inner_q)
        for j in range(K):
            M[row][j] = (M[row][j] * inv_p).truncate(inner_q)
        b[row] = (b[row] * inv_p).truncate(inner_q)
        for r in range(N):
            if r == row or M[r][col].is_zero():
                continue
            f = M[r][col]
            for j in range(K):
                M[r][j] = (M[r][j] - f * M[row][j]).truncate(inner_q)
            b[r] = (b[r] - f * b[row]).truncate(inner_q)
        row += 1
        if row >= N:
            break

    sols: dict = dict(lower)
    for i in range(row):
        pc = next((j for j in range(K) if not M[i][j].is_zero()), None)
        if pc is not None:
            sols[anchors[pc]] = b[i].truncate(q_max)
    _Vm_CACHE[key] = sols
    return sols


_V3_CACHE: dict = {}


def _solve_v3_anchors(q_max: int):
    """Cyclicity-Schur reduction for m=3 anchors V3_0, V3_1, V3_2, V3_3.

    Uses 3-letter cyclicity at four triples that each pick out a unique
    V3 anchor diagonally:
      (-1, 0, 1): determines V3_0
      (-1, 0, 2): determines V3_1
      (-1, 0, 3): determines V3_2
      (-1, 0, 4): determines V3_3

    Each equation also contains V1, V2 anchors (already solved) and
    Wilson tr_W constants.  Solve simultaneously via QmuLaurent Gauss-
    Jordan after substituting V1, V2 known values.

    Cached per `q_max`.
    """
    if q_max in _V3_CACHE:
        return _V3_CACHE[q_max]
    from qmu_laurent import QmuLaurent

    # Pre-solve V1, V2 at INFLATED precision: V3 equation coefs involve
    # high q-powers (e.g. q² - q¹⁰ at (0, 0, 1)) so substituted V1, V2
    # must be accurate at q_max + buffer to avoid precision loss.
    buffer = 12
    v12 = _solve_v1_v2_anchors(q_max + buffer)

    # Triples chosen for clean low-q coefs on each V3 anchor:
    #   (0, 0, 0)  → V3_0 with coef 1     (lowest q-power)
    #   (0, 0, 1)  → V3_1 with coef q² - q¹⁰
    #   (0, 0, 2)  → V3_2 with coef q⁴ - q⁸
    #   (-1, 0, 4) → V3_3 with coef -q⁴ + q¹⁰
    triples = [(0, 0, 0), (0, 0, 1), (0, 0, 2), (-1, 0, 4)]
    eqs = [_cyclicity_equation_3letter(*t, q_max=q_max + 12) for t in triples]

    # Substitute V1, V2 into each equation: move their contributions to the
    # const side.  Remaining anchors should be V3_*.
    reduced_eqs = []
    for (eq_a, eq_c) in eqs:
        new_eq_a = {}
        new_const = QmuLaurent.from_rlaurent(eq_c)
        for anc, coef_rl in eq_a.items():
            if anc in v12:
                coef = QmuLaurent.from_rlaurent(coef_rl)
                contrib = coef * v12[anc]
                new_const = new_const - contrib
            else:
                new_eq_a[anc] = coef_rl
        reduced_eqs.append((new_eq_a, new_const))

    anchors = sorted({k for ea, _ in reduced_eqs for k in ea},
                     key=lambda x: (x[0], x[1]))
    K = len(anchors); N = len(reduced_eqs)
    M = [[QmuLaurent.zero()] * K for _ in range(N)]
    b = [QmuLaurent.zero() for _ in range(N)]
    for i, (ea, ec) in enumerate(reduced_eqs):
        for j, anc in enumerate(anchors):
            M[i][j] = QmuLaurent.from_rlaurent(ea.get(anc, _rl_zero()))
        b[i] = ec.truncate(q_max + 12)

    inner_q = q_max + 12
    row = 0
    for col in range(K):
        pivot = next((r for r in range(row, N) if not M[r][col].is_zero()),
                     None)
        if pivot is None:
            continue
        if pivot != row:
            M[row], M[pivot] = M[pivot], M[row]
            b[row], b[pivot] = b[pivot], b[row]
        inv_p = M[row][col].series_inverse(inner_q)
        for j in range(K):
            M[row][j] = (M[row][j] * inv_p).truncate(inner_q)
        b[row] = (b[row] * inv_p).truncate(inner_q)
        for r in range(N):
            if r == row or M[r][col].is_zero():
                continue
            f = M[r][col]
            for j in range(K):
                M[r][j] = (M[r][j] - f * M[row][j]).truncate(inner_q)
            b[r] = (b[r] - f * b[row]).truncate(inner_q)
        row += 1
        if row >= N:
            break

    sols: dict = dict(v12)
    for i in range(row):
        pc = next((j for j in range(K) if not M[i][j].is_zero()), None)
        if pc is not None:
            sols[anchors[pc]] = b[i].truncate(q_max)
    _V3_CACHE[q_max] = sols
    return sols


def trace_pSU2nf1_label(m: int, e: int, mu_p: int = 0,
                          q_max: int = _Q_MAX_DEFAULT):
    """Axiom-derived trace of pSU2nf1 canonical-basis seed `(m, e, μ_p)`.

    Dispatches by `m`:
      * m = 0: Tr(W_e) via Schur F(v, μ).
      * m = 1: V1_0 (e even) or V1_1 (e odd) via cyclicity-Schur solve.
      * m = 2: V2_0 (e mod 4=0), V2_2 (e mod 4=2), V2_1 (e odd).
      * m ≥ 3: TBD (further cyclicity bridges needed).

    All m ≥ 1 traces are Q(μ, q)-rational combinations of `Tr(W_n)`'s
    (Schur F closed form), per the user-prescribed two-trick recursion.
    """
    if m == 0:
        # Z₂ is BROKEN in Nf=1 (Tr(W_odd) ≠ 0), no parity-based vanishing.
        return tr_W(e, q_max) * _rl_q_mu(0, mu_p)
    # m ≥ 1: dispatch via generic cyclicity solver.  m=1 still keeps the
    # period-2 (V1_0, V1_1) fold per the [w_1, H_{n+1}] commutator.
    if m == 1:
        sols = _solve_v1_v2_anchors(q_max)
    else:
        sols = _solve_vm_anchors(m, q_max)
    # Anchor key via period-2m reflective fold (m+1 anchors total).
    period = 2 * m
    r = e % period
    e_anc = min(r, period - r)
    key = (f'V{m}', e_anc)
    if key not in sols:
        raise NotImplementedError(
            f"trace_pSU2nf1_label(m={m}, e={e}): anchor {key} not in "
            f"solver output (rank-deficient or out-of-range)"
        )
    rl = _qmu_to_rl(sols[key], q_max)
    return rl * _rl_q_mu(0, mu_p)


def trace_seed(seed_label: tuple, K: int):
    """Trace dispatch on the native `(h_factors, μ_power)` label.

    Section decomposition (Tr is R-linear so the μ-power factors out):

        Tr(L_(h_factors, μ_pow))  =  μ^{μ_pow} · Tr(L_(h_factors, 0))

    The section trace `Tr(L_(h_factors, 0))` then dispatches by m (the
    number of H-letters) using the ad-hoc reduction procedures:

      * m = 0 (Wilson):   Schur F(v, μ) closed form via `tr_W`.
      * m ≥ 1:            cyclicity-bridge reduction to Tr(W_n)
                          (under construction — see `_cyclicity_equation`,
                          `m2_eliminated_combination`, `solve_v1_anchors`).

    Returns `RPowerSeries[R]` over `R = AbelianZPlusRing(rank=1)`.
    """
    from su2_nf1_h_multiply import _native_to_psu2nf1
    from zplus_ring import RPowerSeries, AbelianZPlusRing
    R = AbelianZPlusRing(rank=1)
    m, e, mu_p = _native_to_psu2nf1(seed_label)
    rl = trace_pSU2nf1_label(m, e, mu_p, q_max=K)
    return RPowerSeries(R,
                        {k: v for k, v in rl.coeffs.items() if 0 <= k <= K},
                        K)


# -----------------------------------------------------------------
# Cyclicity solver for m=1 and m=2 anchors
# -----------------------------------------------------------------
#
# Anchors:
#   m=1: V_0, V_1, V_2, V_3, V_4, V_5 — Tr(L_{(1, e)}) for e mod 6.
#   m=2: W_0, ..., W_11               — Tr(L_{(2, e)}) for e mod 12.
#
# Cyclicity identity Tr(H_a·H_b) = μ^{-2}·Tr(H_{b-6}·H_a) at chosen
# (a, b) pairs gives one linear equation per pair in the 18 anchors,
# with `RLaurent[R]` coefficients.

from su2_nf1_h_gap_k import h_mul_h


def _multi_letter_native(indices) -> dict:
    """Native decomposition of `H_{a_0} · H_{a_1} · … · H_{a_{n-1}}` in
    `(m, e)` seeds.

    Folds the product left-to-right via `multiply_native`; converts each
    output native label into a seed via `_native_to_psu2nf1`.
    """
    from su2_nf1_h_multiply import multiply_native, _native_to_psu2nf1
    from kalgebra import Element
    indices = list(indices)
    if not indices:
        return {(0, 0): _rl_one()}
    H_first = (((indices[0], 1),), 0)
    cur = Element({H_first: _rl_one()})
    for n in indices[1:]:
        H_n = (((n, 1),), 0)
        next_terms = {}
        for lbl, coef in cur.terms.items():
            prod = multiply_native(lbl, H_n)
            for k, v in prod.terms.items():
                term = coef * v
                if k in next_terms:
                    next_terms[k] = next_terms[k] + term
                else:
                    next_terms[k] = term
        cur = Element({k: v for k, v in next_terms.items() if not v.is_zero()})
    out = {}
    for lbl, coef in cur.terms.items():
        m, e, mu_p = _native_to_psu2nf1(lbl)
        seed = (m, e)
        term = coef * _rl_q_mu(0, mu_p)
        if seed in out:
            out[seed] = out[seed] + term
        else:
            out[seed] = term
    return {k: v for k, v in out.items() if not v.is_zero()}


def _three_letter_native(a: int, b: int, c: int) -> dict:
    """Backward-compat wrapper for 3-letter products."""
    return _multi_letter_native([a, b, c])


def _cyclicity_equation_multi(indices, q_max: int):
    """General multi-letter cyclicity equation
        Tr(H_{a_0}·…·H_{a_{n-1}}) = Tr(ρ²(H_{a_{n-1}})·H_{a_0}·…·H_{a_{n-2}})
                                  = Tr(H_{a_{n-1}-6}·H_{a_0}·…·H_{a_{n-2}})
    Returns ({anchor_id: coef}, const) for LHS - RHS = 0.
    """
    indices = list(indices)
    lhs = _multi_letter_native(indices)
    rhs_indices = [indices[-1] - 6] + indices[:-1]
    rhs = _multi_letter_native(rhs_indices)

    def _accumulate(seeds, sign):
        out_a = {}
        out_c = _rl_zero()
        for (m, e), coef in seeds.items():
            anc_id, mu_sh, const_val = _tr_of_seed(m, e)
            if anc_id is None:
                out_c = out_c + _rl_truncate_qmu(sign * coef * const_val, q_max)
            else:
                contrib = _rl_truncate_qmu(sign * coef * mu_sh, q_max)
                if contrib.is_zero():
                    continue
                if anc_id in out_a:
                    out_a[anc_id] = out_a[anc_id] + contrib
                else:
                    out_a[anc_id] = contrib
        return out_a, out_c

    lhs_a, lhs_c = _accumulate(lhs, _rl_one())
    rhs_a, rhs_c = _accumulate(rhs, _rl_one())

    eq_a = {}
    for k in set(lhs_a) | set(rhs_a):
        diff = lhs_a.get(k, _rl_zero()) - rhs_a.get(k, _rl_zero())
        if not diff.is_zero():
            eq_a[k] = diff
    eq_c = rhs_c - lhs_c
    return eq_a, _rl_truncate_qmu(eq_c, q_max)


def _cyclicity_equation_3letter(a: int, b: int, c: int, q_max: int):
    """Backward-compat wrapper."""
    return _cyclicity_equation_multi([a, b, c], q_max=q_max)


def _tr_of_seed(m: int, e: int):
    """Return (anchor_id_or_None, RLaurent_factor, const_RLaurent_value).

    For m=0 (Wilson): Tr(W_e) is KNOWN via `tr_W` (Schur F).  Return
    (None, 1, Tr(W_e)).

    For m=1: period-2 collapse — Tr(H_n) depends only on `n mod 2`
    (verified empirically + structural: H_{n+2} − H_n = (q-q⁻¹)⁻¹·
    [w_1, H_{n+1}] is a commutator whose trace vanishes under
    ρ²-twisted cyclicity).  No μ-shift (corrected ρ²(H_n) = H_{n-6}).
    Two anchors: ('V1', 0) = Tr(H_even), ('V1', 1) = Tr(H_odd).

    For m=2: native (2, e) cone monomial.  ρ²(H_a·H_{a+1}) =
    H_{a-6}·H_{a-5}, so ρ²-cyclicity gives period 12 in e: Tr((2, e))
    = Tr((2, e-12)).  Period-2 (`[w_1, H_a H_{a+1}+...]` analogue) may
    further collapse to a smaller period; for now use period 12.
    """
    if m == 0:
        if e < 0:
            return (None, _rl_zero(), _rl_zero())
        return (None, _rl_one(), tr_W(e))         # Tr(W_e) — known.
    # m ≥ 1 general pattern: period 2m in e with REFLECTIVE fold k ↔ 2m−k
    # → m+1 distinct anchors {V{m}_k : k ∈ 0..m}.
    # Verified empirically (vs BPS) for m = 1, 2, 3, 4.
    period = 2 * m
    r = e % period
    e_anc = min(r, period - r)                           # reflective fold.
    return ((f'V{m}', e_anc), _rl_one(), _rl_zero())


def _cyclicity_equation(a: int, b: int, q_max: int):
    """Cyclicity at (a, b): Tr(H_a·H_b) − μ^{-2}·Tr(H_{b-6}·H_a) = 0.

    Returns ({anchor_id: RLaurent coef}, RLaurent const) where the
    equation is `sum coef · anchor = const`.  The const accumulates
    all the Tr(W_e) contributions.
    """
    # LHS = h_mul_h(min(a, b), max(a, b)) with cocycle reorder if a > b.
    if a <= b:
        lhs = h_mul_h(a, b)
    else:
        # H_a · H_b for a > b: within rank-2 cone if |a-b| ≤ 1, else bar.
        if abs(a - b) <= 1:
            fwd = h_mul_h(b, a)
            phase = _rl_q_mu(2 * (b - a), 0)
            lhs = {s: phase * c for s, c in fwd.items() if not c.is_zero()}
        else:
            fwd = h_mul_h(b, a)
            lhs = {}
            for s, c in fwd.items():
                bar = RLaurent(_R, {-e: r for e, r in c.coeffs.items()})
                if not bar.is_zero():
                    lhs[s] = bar
    # RHS = h_mul_h(b-6, a)
    rhs_lo, rhs_hi = (b - 6, a) if (b - 6) <= a else (a, b - 6)
    if (b - 6) <= a:
        rhs_pre = h_mul_h(b - 6, a)
    else:
        # a > b-6 unlikely with our choices but handle.
        fwd = h_mul_h(a, b - 6)
        if abs(a - (b - 6)) <= 1:
            phase = _rl_q_mu(2 * (a - (b - 6)), 0)
            rhs_pre = {s: phase * c for s, c in fwd.items() if not c.is_zero()}
        else:
            rhs_pre = {}
            for s, c in fwd.items():
                bar = RLaurent(_R, {-e: r for e, r in c.coeffs.items()})
                if not bar.is_zero():
                    rhs_pre[s] = bar
    mu2_inv = _rl_q_mu(0, -2)

    def _accumulate(canonical, sign):
        out_anchors = {}
        out_const = _rl_zero()
        for (m, e), coef in canonical.items():
            anchor_id, mu_shift, const_val = _tr_of_seed(m, e)
            if anchor_id is None:
                out_const = out_const + _rl_truncate_qmu(
                    sign * coef * const_val, q_max
                )
            else:
                contrib = _rl_truncate_qmu(sign * coef * mu_shift, q_max)
                if contrib.is_zero():
                    continue
                if anchor_id in out_anchors:
                    out_anchors[anchor_id] = out_anchors[anchor_id] + contrib
                else:
                    out_anchors[anchor_id] = contrib
        return out_anchors, out_const

    lhs_a, lhs_c = _accumulate(lhs, _rl_one())
    rhs_a_raw, rhs_c_raw = _accumulate(rhs_pre, _rl_one())   # corrected ρ²: no μ⁻² factor

    # Equation: LHS = RHS  ⇒  LHS_a · X + LHS_c = RHS_a · X + RHS_c
    #                       (LHS_a − RHS_a) · X = RHS_c − LHS_c
    eq_anchors = {}
    for k in set(lhs_a) | set(rhs_a_raw):
        diff = lhs_a.get(k, _rl_zero()) - rhs_a_raw.get(k, _rl_zero())
        if not diff.is_zero():
            eq_anchors[k] = diff
    eq_const = rhs_c_raw - lhs_c
    return eq_anchors, eq_const


# -----------------------------------------------------------------
# Linear-system solver over RLaurent[R]
# -----------------------------------------------------------------

def _rl_negate(r: RLaurent) -> RLaurent:
    return RLaurent(_R, {e: -c for e, c in r.coeffs.items()})


def _rl_series_inverse(D: RLaurent, q_max: int,
                        mu_window: int = 8) -> RLaurent:
    """`1/D` as a truncated q-Laurent power series with R-coefficients.

    Requires `D` to have a unique lowest q-exponent `k0` with coefficient
    `c0 = D.coeffs[k0]` an R-element with a *unit-like* leading μ-content
    (we require c0 to have a single dominant μ-monomial; the typical
    case is `c0 = ±μ^{p}`).  We factor `D = q^{k0} · c0 · (1 + p)` with
    `p = sum_{e > k0} (c_e / c0) · q^{e - k0}`, then expand
    `1/(1 + p) = sum_n (−p)^n` truncated.
    """
    if not D.coeffs:
        raise ZeroDivisionError("series_inverse of zero")
    k0 = min(D.coeffs.keys())
    c0 = D.coeffs[k0]
    # Need c0 to be invertible in R.  For AbelianZPlusRing(rank=1),
    # R-elements are Z-linear combos of basis_element((k,)) = μ^k.
    # Invertible iff a single ±μ^k term.
    c0_terms = list(c0.terms.items()) if hasattr(c0, 'terms') else None
    if c0_terms is None or len(c0_terms) != 1:
        raise ValueError(
            f"_rl_series_inverse: leading R-coef not a single μ-monomial: {c0}"
        )
    mu_label, scalar = c0_terms[0]
    if scalar not in (1, -1, Fraction(1), Fraction(-1)):
        # Allow scalar invertible in Q.
        pass
    # 1/c0 = (1/scalar) · μ^{-mu_label[0]}
    inv_c0_mu = (-mu_label[0],) if isinstance(mu_label, tuple) else (-mu_label,)
    inv_c0_base = _R.basis_element(inv_c0_mu)
    if scalar == 1:
        inv_c0 = inv_c0_base
    elif scalar == -1:
        from zplus_ring import RElement
        inv_c0 = RElement(_R, {k: -v for k, v in inv_c0_base.terms.items()})
    else:
        from zplus_ring import RElement
        inv_c0 = RElement(_R, {k: Fraction(v, scalar)
                                for k, v in inv_c0_base.terms.items()})
    # p = sum_{e > k0} (c_e / c0) · q^{e - k0}.
    p_coeffs = {}
    for e, c in D.coeffs.items():
        if e == k0:
            continue
        # c / c0 = c · μ^{-mu_label[0]} / scalar.
        c_div = c * _R.basis_element(inv_c0_mu)
        if scalar != 1:
            # Each R basis element has int coefficient.  Divide by scalar
            # if integer division works; otherwise convert to Fraction.
            new_terms = {}
            for lbl, sc in c_div.terms.items():
                new_terms[lbl] = sc / scalar if scalar != 1 else sc
            from zplus_ring import RElement
            c_div = RElement(_R, new_terms)
        p_coeffs[e - k0] = c_div
    p = RLaurent(_R, p_coeffs)
    # 1/(1+p) = sum (-p)^n.
    inv_1plus_p = _rl_one()
    term = _rl_one()
    inner_qmax = q_max + k0 + 4
    sign = 1
    while True:
        term = _rl_truncate_qmu(term * p, inner_qmax)
        if term.is_zero():
            break
        sign = -sign
        signed_term = RLaurent(_R, {e: (c if sign > 0 else -c)
                                     for e, c in term.coeffs.items()})
        inv_1plus_p = inv_1plus_p + signed_term
    # 1/D = q^{-k0} · 1/c0 · 1/(1+p).
    scaled = RLaurent(_R, {e: c * inv_c0 for e, c in inv_1plus_p.coeffs.items()})
    shifted = RLaurent(_R, {e - k0: c for e, c in scaled.coeffs.items()})
    return _rl_truncate_qmu(shifted, q_max)


def solve_anchors(pairs, q_max: int = _Q_MAX_DEFAULT):
    """Solve the cyclicity linear system for m=1 and m=2 anchors.

    `pairs` is a list of `(a, b)` tuples.  Each gives one equation
    `eq_anc · X = eq_const`; we collect them into an N×K linear system
    (N = len(pairs), K = unique anchor count) and solve via Gauss-
    Jordan with `_rl_series_inverse` for pivot inversion.

    Returns `{anchor_id: RLaurent}`.
    """
    eqs = [_cyclicity_equation(a, b, q_max) for (a, b) in pairs]
    # Collect anchors.
    anchors = sorted({k for (ea, _) in eqs for k in ea},
                      key=lambda x: (x[0], x[1]))
    K = len(anchors)
    N = len(eqs)
    # Build augmented matrix M | b.  Rows N, cols K + 1.
    M = [[_rl_zero()] * K for _ in range(N)]
    b = [_rl_zero()] * N
    for i, (ea, ec) in enumerate(eqs):
        for j, anc in enumerate(anchors):
            M[i][j] = ea.get(anc, _rl_zero())
        b[i] = ec
    # Gauss-Jordan elimination.
    row = 0
    for col in range(K):
        # Find pivot row with non-zero entry at column col.
        pivot = None
        for r in range(row, N):
            if not M[r][col].is_zero():
                pivot = r
                break
        if pivot is None:
            continue
        if pivot != row:
            M[row], M[pivot] = M[pivot], M[row]
            b[row], b[pivot] = b[pivot], b[row]
        try:
            inv_p = _rl_series_inverse(M[row][col], q_max + 8)
        except (ValueError, ZeroDivisionError) as e:
            raise RuntimeError(
                f"solve_anchors: cannot invert pivot at col {col}: {e}"
            )
        # Scale row.
        for j in range(K):
            M[row][j] = _rl_truncate_qmu(M[row][j] * inv_p, q_max + 4)
        b[row] = _rl_truncate_qmu(b[row] * inv_p, q_max + 4)
        # Eliminate column.
        for r in range(N):
            if r == row:
                continue
            factor = M[r][col]
            if factor.is_zero():
                continue
            for j in range(K):
                M[r][j] = _rl_truncate_qmu(M[r][j] - factor * M[row][j], q_max + 4)
            b[r] = _rl_truncate_qmu(b[r] - factor * b[row], q_max + 4)
        row += 1
        if row >= N:
            break
    # Read off solution from reduced rows.
    sols = {}
    for i in range(row):
        pivot_col = None
        for j in range(K):
            if not M[i][j].is_zero():
                pivot_col = j
                break
        if pivot_col is None:
            continue
        sols[anchors[pivot_col]] = _rl_truncate_qmu(b[i], q_max)
    return sols, anchors


# -----------------------------------------------------------------
# Updated solver using QmuLaurent (= q-Laurent with RatFuncMu coefs)
# -----------------------------------------------------------------

from qmu_laurent import QmuLaurent


def solve_anchors_qmu(pairs, q_max: int = _Q_MAX_DEFAULT):
    """Same as `solve_anchors` but converts every RLaurent[R] entry to
    `QmuLaurent` (q-Laurent over `Q(μ)`) before row-reducing.  Avoids
    the multi-μ-monomial-pivot blocker.
    """
    eqs = [_cyclicity_equation(a, b, q_max) for (a, b) in pairs]
    anchors = sorted({k for (ea, _) in eqs for k in ea},
                      key=lambda x: (x[0], x[1]))
    K = len(anchors)
    N = len(eqs)
    M = [[QmuLaurent.zero()] * K for _ in range(N)]
    b = [QmuLaurent.zero() for _ in range(N)]
    for i, (ea, ec) in enumerate(eqs):
        for j, anc in enumerate(anchors):
            M[i][j] = QmuLaurent.from_rlaurent(ea.get(anc, _rl_zero()))
        b[i] = QmuLaurent.from_rlaurent(ec)
    # Gauss-Jordan with truncation budget.
    inner_q = q_max + 12
    row = 0
    for col in range(K):
        pivot = None
        for r in range(row, N):
            if not M[r][col].is_zero():
                pivot = r
                break
        if pivot is None:
            continue
        if pivot != row:
            M[row], M[pivot] = M[pivot], M[row]
            b[row], b[pivot] = b[pivot], b[row]
        inv_p = M[row][col].series_inverse(inner_q)
        for j in range(K):
            M[row][j] = (M[row][j] * inv_p).truncate(inner_q)
        b[row] = (b[row] * inv_p).truncate(inner_q)
        for r in range(N):
            if r == row:
                continue
            factor = M[r][col]
            if factor.is_zero():
                continue
            for j in range(K):
                M[r][j] = (M[r][j] - factor * M[row][j]).truncate(inner_q)
            b[r] = (b[r] - factor * b[row]).truncate(inner_q)
        row += 1
        if row >= N:
            break
    sols = {}
    for i in range(row):
        pivot_col = None
        for j in range(K):
            if not M[i][j].is_zero():
                pivot_col = j
                break
        if pivot_col is None:
            continue
        sols[anchors[pivot_col]] = b[i].truncate(q_max)
    return sols, anchors


def solve_anchors_triangular(pairs, q_max: int = _Q_MAX_DEFAULT):
    """Triangularize on m: pivot on V2 columns first (q-monomial leading
    coefs are cheap to invert), eliminating m=2 anchors from all rows.
    Remaining 6 equations should be purely in V1 anchors -- solve those
    next.  Finally back-substitute to recover V2 values.
    """
    eqs = [_cyclicity_equation(a, b, q_max) for (a, b) in pairs]
    all_anchors = sorted({k for (ea, _) in eqs for k in ea},
                          key=lambda x: (x[0], x[1]))
    # Re-order: V2 first, then V1.
    v2_anchors = [a for a in all_anchors if a[0] == 'V2']
    v1_anchors = [a for a in all_anchors if a[0] == 'V1']
    anchors = v2_anchors + v1_anchors
    K = len(anchors)
    K2 = len(v2_anchors)
    N = len(eqs)
    M = [[QmuLaurent.zero()] * K for _ in range(N)]
    b = [QmuLaurent.zero() for _ in range(N)]
    for i, (ea, ec) in enumerate(eqs):
        for j, anc in enumerate(anchors):
            M[i][j] = QmuLaurent.from_rlaurent(ea.get(anc, _rl_zero()))
        b[i] = QmuLaurent.from_rlaurent(ec)
    inner_q = q_max + 12
    # Phase 1: eliminate V2 columns (col 0 .. K2-1).
    row = 0
    for col in range(K2):
        pivot = None
        for r in range(row, N):
            if not M[r][col].is_zero():
                pivot = r; break
        if pivot is None:
            continue
        if pivot != row:
            M[row], M[pivot] = M[pivot], M[row]
            b[row], b[pivot] = b[pivot], b[row]
        inv_p = M[row][col].series_inverse(inner_q)
        for j in range(K):
            M[row][j] = (M[row][j] * inv_p).truncate(inner_q)
        b[row] = (b[row] * inv_p).truncate(inner_q)
        for r in range(N):
            if r == row or M[r][col].is_zero(): continue
            factor = M[r][col]
            for j in range(K):
                M[r][j] = (M[r][j] - factor * M[row][j]).truncate(inner_q)
            b[r] = (b[r] - factor * b[row]).truncate(inner_q)
        row += 1
        if row >= N: break
    # After Phase 1, rows [row, N) should have all-zero V2 columns:
    # those are equations purely in V1.  Continue elimination on V1 cols.
    v1_start_row = row
    for col in range(K2, K):
        pivot = None
        for r in range(row, N):
            if not M[r][col].is_zero():
                pivot = r; break
        if pivot is None:
            continue
        if pivot != row:
            M[row], M[pivot] = M[pivot], M[row]
            b[row], b[pivot] = b[pivot], b[row]
        inv_p = M[row][col].series_inverse(inner_q)
        for j in range(K):
            M[row][j] = (M[row][j] * inv_p).truncate(inner_q)
        b[row] = (b[row] * inv_p).truncate(inner_q)
        for r in range(N):
            if r == row or M[r][col].is_zero(): continue
            factor = M[r][col]
            for j in range(K):
                M[r][j] = (M[r][j] - factor * M[row][j]).truncate(inner_q)
            b[r] = (b[r] - factor * b[row]).truncate(inner_q)
        row += 1
        if row >= N: break
    sols = {}
    for i in range(row):
        pivot_col = None
        for j in range(K):
            if not M[i][j].is_zero():
                pivot_col = j; break
        if pivot_col is None: continue
        sols[anchors[pivot_col]] = b[i].truncate(q_max)
    return sols, anchors


# -----------------------------------------------------------------
# Wilson constraints from m=2-eliminating combinations
# -----------------------------------------------------------------
#
# Canonical ρ²-twisted trace cyclicity (KAlgebra contract, kalgebra.py:534):
#   Tr(L_a · L_b)  =  Tr(ρ²(L_b) · L_a)
#
# Applied to L_a = H_{-a}, L_b = H_a:
#   Tr(H_{-a} · H_a)  =  Tr(ρ²(H_a) · H_{-a})  =  Tr(H_{a-6} · H_{-a})
# (ρ²(H_n) = H_{n-6}, no μ shift — bar-twist cancels the two μ⁻¹'s.)
#
# Applied to L_a = H_0, L_b = H_0:
#   Tr(H_0²)  =  Tr(H_{-6} · H_0)
#
# Both LHS pieces have leading m=2 anchor L_{(2, 0)}: h_mul_h(-a, a)
# has (2, 0) coef q^{2a}, h_mul_h(0, 0) = {(2, 0): 1}.  Subtract q^{2a}·
# Tr(H_0²) to cancel the (2, 0) anchor.  Similarly on the RHS:
# h_mul_h(a-6, -a) has leading m=2 anchor L_{(2, -6)} which is cancelled
# by q^{2a}·Tr(H_{-6}·H_0).  The two sides give equal m≤1 + Wilson
# expressions, hence a linear equation in the m=1 anchors.

def m2_eliminated_combination(a: int, q_max: int):
    """Raw cyclicity equation at parameter a (imitating pure SU(2)):
        Tr(H_{-a} · H_a)  =  Tr(H_{a−6} · H_{-a})
    (Canonical contract `Tr(L_a · L_b) = Tr(ρ²(L_b) · L_a)`, with
    ρ²(H_a) = H_{a−6}, no μ shift.)

    Return ({anchor_id: coef}, const) for LHS − RHS = 0 expanded in
    canonical (m, e) seeds via native `h_mul_h`, with m=1/m=2 anchors
    folded by period-2 and m=0 (Wilson) terms replaced by `tr_W`.

    No q^{2a}·Tr(H_0²) elimination: that turned the equation into a
    tautology (LHS_paren = RHS_paren both reduce by cyclicity at a=0).
    Pure SU(2) `tr_h0_bridge` uses the raw equation directly.
    """
    # LHS: H_{-a}·H_a — native h_mul_h with low ≤ high.
    lo, hi = (-a, a) if -a <= a else (a, -a)
    lhs_a = h_mul_h(lo, hi)

    # RHS: Tr(H_{a-6} · H_{-a}) — by canonical cyclicity ρ²(H_a) = H_{a-6}.
    rhs_lo1, rhs_hi1 = (a - 6, -a) if (a - 6) <= -a else (-a, a - 6)
    rhs1 = h_mul_h(rhs_lo1, rhs_hi1)

    def _accumulate(canonical, sign):
        out_a = {}
        out_c = _rl_zero()
        for (m, e), coef in canonical.items():
            anc_id, mu_sh, const_val = _tr_of_seed(m, e)
            if anc_id is None:
                out_c = out_c + sign * coef * const_val
            else:
                term = sign * coef * mu_sh
                if anc_id in out_a:
                    out_a[anc_id] = out_a[anc_id] + term
                else:
                    out_a[anc_id] = term
        return out_a, out_c

    # Raw cyclicity: LHS = Tr(H_{-a}·H_a),  RHS = Tr(H_{a-6}·H_{-a}).
    lhs_a_dict, lhs_c = _accumulate(lhs_a, _rl_one())
    rhs1_a, rhs1_c = _accumulate(rhs1, _rl_one())

    # Merge.  Equation: LHS - RHS = 0.
    #   (lhs_a - rhs_a)·X + (lhs_c - rhs_c) = 0
    #   ⟹  eq_a·X = -(lhs_c - rhs_c) = rhs_c - lhs_c.
    eq_a = {}
    for d, sign in [(lhs_a_dict, +1), (rhs1_a, -1)]:
        for k, v in d.items():
            eq_a[k] = eq_a.get(k, _rl_zero()) + sign * v
    eq_c = rhs1_c - lhs_c
    # Drop zero entries.
    eq_a = {k: _rl_truncate_qmu(v, q_max) for k, v in eq_a.items()
            if not v.is_zero()}
    eq_a = {k: v for k, v in eq_a.items() if not v.is_zero()}
    eq_c = _rl_truncate_qmu(eq_c, q_max)
    return eq_a, eq_c


def solve_v1_anchors(q_max: int = _Q_MAX_DEFAULT):
    """Solve the 6 m=1 anchors V_0..V_5 via m=2-eliminated combinations.

    Uses `m2_eliminated_combination(a, q_max)` for `a = 1, 2, 3, 4, 5, 6`
    — six equations in six V1 anchors plus a Tr(W_*)-only constant.
    The combination already cancels Tr(H_0^2); the remaining m=2
    contributions from the cyclicity RHS at higher gap may still
    appear, but for small a (a ≤ 3) they're absent.

    Returns `{('V1', e): QmuLaurent}` for e in 0..5.

    Implementation: use QmuLaurent (Q(μ)[[q]]) for row reduction since
    derived pivots may have multi-μ-monomial leading even when the
    starting equation coefs were single-monomial.
    """
    eqs = [m2_eliminated_combination(a, q_max) for a in (1, 2, 3, 4, 5, 6)]
    anchors = sorted({k for (ea, _) in eqs for k in ea},
                      key=lambda x: (x[0], x[1]))
    K = len(anchors)
    N = len(eqs)
    M = [[QmuLaurent.zero()] * K for _ in range(N)]
    b = [QmuLaurent.zero() for _ in range(N)]
    for i, (ea, ec) in enumerate(eqs):
        for j, anc in enumerate(anchors):
            M[i][j] = QmuLaurent.from_rlaurent(ea.get(anc, _rl_zero()))
        b[i] = QmuLaurent.from_rlaurent(ec)
    # Gauss-Jordan over Q(μ)[[q]].
    row = 0
    inner_q = q_max + 12
    for col in range(K):
        pivot = None
        for r in range(row, N):
            if not M[r][col].is_zero():
                pivot = r; break
        if pivot is None:
            continue
        if pivot != row:
            M[row], M[pivot] = M[pivot], M[row]
            b[row], b[pivot] = b[pivot], b[row]
        inv_p = M[row][col].series_inverse(inner_q)
        for j in range(K):
            M[row][j] = (M[row][j] * inv_p).truncate(inner_q)
        b[row] = (b[row] * inv_p).truncate(inner_q)
        for r in range(N):
            if r == row or M[r][col].is_zero(): continue
            factor = M[r][col]
            for j in range(K):
                M[r][j] = (M[r][j] - factor * M[row][j]).truncate(inner_q)
            b[r] = (b[r] - factor * b[row]).truncate(inner_q)
        row += 1
        if row >= N: break
    sols = {}
    for i in range(row):
        pivot_col = None
        for j in range(K):
            if not M[i][j].is_zero():
                pivot_col = j; break
        if pivot_col is None: continue
        sols[anchors[pivot_col]] = b[i].truncate(q_max)
    return sols, anchors


def derive_wilson_constraint(q_max: int = 8):
    """Derive a pure-Tr(W_n) constraint by:

      1. Solving V_0..V_5 from m=2-eliminated combinations at a=1..6.
      2. Substituting them into the equation at a=7.
      3. The residual is an identity in Tr(W_n) only; if it equals
         zero, the Schur F(v, mu) is consistent with cyclicity.

    Returns the residual `QmuLaurent` (zero iff Schur is consistent).
    """
    sols, _ = solve_v1_anchors(q_max)
    # Get equation at a=7.
    eq_a, eq_c = m2_eliminated_combination(7, q_max)
    eq_a_qmu = {k: QmuLaurent.from_rlaurent(v) for k, v in eq_a.items()}
    eq_c_qmu = QmuLaurent.from_rlaurent(eq_c)
    # Substitute: residual = eq_c - sum_k coef_k · V_k.
    residual = eq_c_qmu
    for k, coef in eq_a_qmu.items():
        if k not in sols:
            continue
        residual = residual - coef * sols[k]
    return residual.truncate(q_max)
