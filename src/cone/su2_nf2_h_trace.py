"""`su2_nf2_h_trace` — Schur F(v, μ_1, μ_2) and trace machinery for
SU(2) + N_f = 2 with Spin(4) = SU(2)_L × SU(2)_R global symmetry.

Schur F:

    F(v, μ_1, μ_2) = (q²v²; q²)_∞² · (q²/v²; q²)_∞² · (q²; q²)_∞²
                     / ∏_{σ, τ, ρ ∈ {±1}} (−q · v^σ · μ_1^τ · μ_2^ρ; q²)_∞

Numerator = same as pure-SU(2) / Nf=1 vector multiplet (squared).
Denominator: 8 factors corresponding to the 8 weights of (SU(2)_gauge fund)
× (SU(2)_L fund) × (SU(2)_R fund) = (fund) × (Spin(4) vector).

`Tr(W_n) = [v^n] F − [v^{n+2}] F`  (standard SU(2) Wilson projection).
"""
from __future__ import annotations

from zplus_ring import AbelianZPlusRing, RLaurent, SU2ZPlusRing, RElement
from tensor_zplus_ring import TensorZPlusRing

# Cartan-level ring (Laurent polynomials in μ_1, μ_2) — used internally
# for the q-Pochhammer expansion of Schur F.
_R = AbelianZPlusRing(rank=2)
# Spin(4) = SU(2)_L × SU(2)_R character ring — the EXPLICIT non-abelian
# flavour ring exposed to consumers of `tr_W`.
_R_SP4 = TensorZPlusRing(SU2ZPlusRing(), SU2ZPlusRing())

_Q_MAX_DEFAULT = 20
_V_MAX_DEFAULT = 24


def _rl_one() -> RLaurent:
    return RLaurent(_R, {0: _R.one()})


def _rl_zero() -> RLaurent:
    return RLaurent(_R, {})


def _rl_q_mu(q_pow: int, mu_pows=(0, 0)) -> RLaurent:
    """`q^{q_pow} · μ_1^{mu_pows[0]} · μ_2^{mu_pows[1]}` as `RLaurent[R]`."""
    return RLaurent(_R, {q_pow: _R.basis_element(tuple(mu_pows))})


def _f_mul(f1: dict, f2: dict, v_max: int, q_max: int) -> dict:
    out: dict = {}
    for v1, p1 in f1.items():
        if abs(v1) > v_max:
            continue
        for v2, p2 in f2.items():
            v_total = v1 + v2
            if abs(v_total) > v_max:
                continue
            prod = p1 * p2
            trunc = RLaurent(_R, {e: c for e, c in prod.coeffs.items()
                                    if e <= q_max})
            if trunc.is_zero():
                continue
            out[v_total] = out.get(v_total, _rl_zero()) + trunc
    return {k: v for k, v in out.items() if not v.is_zero()}


def _pochhammer_qsq(a_dict: dict, q_max: int, v_max: int) -> dict:
    """`(a; q²)_∞ = ∏_{k≥0} (1 − a · q^{2k})` truncated.

    `a_dict` is a dict {v-exp: RLaurent} representing `a` as a polynomial
    in v with RLaurent (q, μ_1, μ_2)-coefficients.
    """
    result = {0: _rl_one()}
    k = 0
    while 2 * k <= q_max:
        q2k = _rl_q_mu(2 * k)
        a_shifted = {v: p * q2k for v, p in a_dict.items()}
        factor = {0: _rl_one()}
        for v, p in a_shifted.items():
            factor[v] = factor.get(v, _rl_zero()) + (-p)
        factor = {kk: vv for kk, vv in factor.items() if not vv.is_zero()}
        result = _f_mul(result, factor, v_max, q_max)
        k += 1
    return result


def _pochhammer_inverse_qsq(a_dict: dict, q_max: int, v_max: int) -> dict:
    """`1/(a; q²)_∞ = ∏_{k≥0} (1 − a·q^{2k})^{-1}` truncated, expanded as
    a geometric series in `a` at each level."""
    result = {0: _rl_one()}
    k = 0
    while 2 * k <= q_max:
        q2k = _rl_q_mu(2 * k)
        a_shifted = {v: p * q2k for v, p in a_dict.items()}
        # 1/(1 − a·q²ᵏ) = Σ_{n≥0} (a·q²ᵏ)^n.
        inv_factor = {0: _rl_one()}
        cur = {0: _rl_one()}
        n = 0
        while True:
            n += 1
            cur = _f_mul(cur, a_shifted, v_max, q_max)
            if not cur:
                break
            for v, p in cur.items():
                inv_factor[v] = inv_factor.get(v, _rl_zero()) + p
            inv_factor = {kk: vv for kk, vv in inv_factor.items()
                          if not vv.is_zero()}
        result = _f_mul(result, inv_factor, v_max, q_max)
        k += 1
    return result


def _power_dict(f: dict, n: int, v_max: int, q_max: int) -> dict:
    out = {0: _rl_one()}
    for _ in range(n):
        out = _f_mul(out, f, v_max, q_max)
    return out


_F_cache: dict = {}


def _build_F(q_max: int, v_max: int) -> dict:
    """SU(2)+Nf=2 Schur F(v, μ_1, μ_2) — single hyper in
    (gauge fund) × (Spin(4) vector) ≡ 8 weights.

        F(v, μ_1, μ_2) = (q²v²;q²)_∞² · (q²/v²;q²)_∞² · (q²;q²)_∞²
                         / ∏_{σ,τ,ρ∈{±1}} (−q v^σ μ_1^τ μ_2^ρ; q²)_∞
    """
    key = (q_max, v_max)
    if key in _F_cache:
        return _F_cache[key]
    # Numerator (SU(2) vector multiplet, squared).
    P_v2  = _pochhammer_qsq({ 2: _rl_q_mu(2, (0, 0))}, q_max, v_max)
    P_v_2 = _pochhammer_qsq({-2: _rl_q_mu(2, (0, 0))}, q_max, v_max)
    P_0   = _pochhammer_qsq({ 0: _rl_q_mu(2, (0, 0))}, q_max, v_max)
    F = _f_mul(_power_dict(P_v2,  2, v_max, q_max),
               _power_dict(P_v_2, 2, v_max, q_max), v_max, q_max)
    F = _f_mul(F, _power_dict(P_0, 2, v_max, q_max), v_max, q_max)
    # Hyper denominator: 8 factors with `−q · v^σ · μ_1^τ · μ_2^ρ` arguments.
    for sigma in (+1, -1):
        for tau in (+1, -1):
            for rho in (+1, -1):
                arg = {sigma: -_rl_q_mu(1, (tau, rho))}
                inv = _pochhammer_inverse_qsq(arg, q_max, v_max)
                F = _f_mul(F, inv, v_max, q_max)
    _F_cache[key] = F
    return F


def _rl_truncate_qmu(r: RLaurent, q_max: int) -> RLaurent:
    return RLaurent(_R, {e: c for e, c in r.coeffs.items() if e <= q_max})


def tr_W(n: int, q_max: int = _Q_MAX_DEFAULT) -> RLaurent:
    """`Tr(W_n) = [v^n] F − [v^{n+2}] F` for SU(2)+Nf=2.

    Output is `RLaurent` over the **Cartan ring** `AbelianZPlusRing(rank=2)`
    (Laurent polynomial in μ_1, μ_2).  For the Spin(4) character form, use
    `tr_W_su2xsu2`.
    """
    v_max = max(_V_MAX_DEFAULT, n + 6)
    F = _build_F(q_max, v_max)
    a = F.get(n, _rl_zero())
    b = F.get(n + 2, _rl_zero())
    return _rl_truncate_qmu(a - b, q_max)


def _cartan_to_su2(coef_dict: dict) -> dict:
    """Convert a Cartan-level dict `{(τ_L, τ_R) μ-power: Z-coef}` to a
    SU(2)_L × SU(2)_R character dict `{(n_L, n_R): Z-coef}` (irrep
    multiplicities of Spin(4)).

    Algorithm: greedy peel the highest weight (n_L, n_R) and subtract its
    character `χ_(n_L, n_R) = (μ_L^{n_L} + … + μ_L^{-n_L}) · (μ_R^{n_R} + …
    + μ_R^{-n_R})` until no monomials remain.  Assumes input is
    Weyl-invariant under (μ_L → μ_L⁻¹, μ_R → μ_R⁻¹).
    """
    # Weyl-invariance pre-check.  The greedy peel below assumes the input is
    # a Spin(4) class function — invariant under μ_L→μ_L⁻¹ and μ_R→μ_R⁻¹.
    # A flavour-charged line (e.g. a single SU(2)_L/R doublet generator) has
    # a non-symmetric trace; without this check the peel finds a "corner"
    # via the sign-flipped fallback every iteration and oscillates forever
    # (never empties `work`).  Detect it up front and bail.
    for (kL, kR), c in coef_dict.items():
        if (coef_dict.get((-kL, kR), 0) != c
                or coef_dict.get((kL, -kR), 0) != c
                or coef_dict.get((-kL, -kR), 0) != c):
            raise ValueError(
                f"_cartan_to_su2: input is not Weyl-invariant "
                f"(μ_L→μ_L⁻¹, μ_R→μ_R⁻¹) — not a Spin(4) class function: "
                f"{coef_dict}")
    work = dict(coef_dict)
    out: dict = {}
    while work:
        # Find highest weight: max (|τ_L|, |τ_R|) lex.
        keys = list(work.keys())
        max_L = max(abs(k[0]) for k in keys)
        sub_keys = [k for k in keys if abs(k[0]) == max_L]
        max_R = max(abs(k[1]) for k in sub_keys)
        hw = (max_L, max_R)
        c = work.get(hw, 0)
        if c == 0:
            # Try (max_L, -max_R) instead.
            c = work.get((max_L, -max_R), 0)
            if c == 0:
                c = work.get((-max_L, max_R), 0)
            if c == 0:
                c = work.get((-max_L, -max_R), 0)
        if c == 0:
            # Couldn't find a corner; bail out (non-Weyl-invariant input).
            raise ValueError(
                f"_cartan_to_su2: residual {work} is not Weyl-invariant"
            )
        # Subtract c · χ_(max_L, max_R).
        out[hw] = out.get(hw, 0) + c
        for kL in range(-max_L, max_L + 1, 2):
            for kR in range(-max_R, max_R + 1, 2):
                key = (kL, kR)
                new = work.get(key, 0) - c
                if new == 0:
                    work.pop(key, None)
                else:
                    work[key] = new
    return out


def tr_W_su2xsu2(n: int, q_max: int = _Q_MAX_DEFAULT):
    """`Tr(W_n)` for SU(2)+Nf=2 with output in the SU(2)_L × SU(2)_R
    character ring `TensorZPlusRing(SU2ZPlusRing(), SU2ZPlusRing())`.

    Returns `RLaurent[_R_SP4]` — each q-coefficient is a Z-linear
    combination of SU(2)_L × SU(2)_R irrep characters `χ_(n_L, n_R)`.
    """
    cartan = tr_W(n, q_max=q_max)
    out: dict = {}
    for q_e, r_elt in cartan.coeffs.items():
        # r_elt is RElement over _R = AbelianZPlusRing(rank=2).  Its
        # `.terms` is {(τ_L, τ_R): Z-coef}.
        coef_dict = {k: v for k, v in r_elt.terms.items()}
        sp4_dict = _cartan_to_su2(coef_dict)
        sp4_elt = RElement(_R_SP4, sp4_dict)
        if not sp4_elt.is_zero():
            out[q_e] = sp4_elt
    return RLaurent(_R_SP4, out)


def spin4_to_cartan(sp4_elt) -> "RElement":
    """Convert an `RElement` over `TensorZPlusRing(SU2 × SU2)` (Spin(4)
    irrep characters) to an `RElement` over `AbelianZPlusRing(rank=2)`
    (Cartan-level Laurent polynomial in μ_L, μ_R), via the natural
    character embedding
        χ_{n_L} ⊗ χ_{n_R}  ↦  (μ_L^{n_L} + μ_L^{n_L-2} + … + μ_L^{-n_L})
                              · (μ_R^{n_R} + … + μ_R^{-n_R}).
    """
    cartan_dict: dict = {}
    for (n_L, n_R), c in sp4_elt.terms.items():
        for kL in range(-n_L, n_L + 1, 2):
            for kR in range(-n_R, n_R + 1, 2):
                key = (kL, kR)
                cartan_dict[key] = cartan_dict.get(key, 0) + c
    return RElement(_R, {k: v for k, v in cartan_dict.items() if v != 0})


def spin4_rpowerseries_to_cartan(rps):
    """Lift `spin4_to_cartan` over a full `RPowerSeries[_R_SP4]`."""
    from zplus_ring import RPowerSeries
    cartan_coeffs = {e: spin4_to_cartan(c) for e, c in rps.coeffs.items()
                     if not c.is_zero()}
    return RPowerSeries(_R, cartan_coeffs, rps.K)
