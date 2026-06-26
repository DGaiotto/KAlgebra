"""`su2_nf3_h_trace` — Schur F(v, μ) and trace machinery for
SU(2) + N_f = 3 with **manifest SU(4) flavour symmetry**.

Conventions
===========
Rank-3 SU(4) Cartan in Dynkin basis (Σ w_k = 0 implicit):
  w_1 = ( 1,  0,  0)
  w_2 = (-1,  1,  0)
  w_3 = ( 0, -1,  1)
  w_4 = ( 0,  0, -1)

Matter content: 1 hyper in (2_gauge, 6_v) of SU(2)×Spin(6), with
6_v = ∧²(SU(4) fund).  The 6 antisymmetric weights w_i+w_j (i<j)
× v^± gauge sign → 12 hyper denominator factors:

    F(v, μ) = (q²v²;q²)_∞² · (q²/v²;q²)_∞² · (q²;q²)_∞²
              / ∏_{σ=±, 1≤i<j≤4} (−q · v^σ · μ^{w_i + w_j}; q²)_∞

`Tr(W_n) = [v^n] F − [v^{n+2}] F` (standard SU(2) Wilson projection).

Wilson lattice identification (BPS side)
========================================
In the SU(4)-manifest BPS quiver (`bps_su2_nf3`), the fundamental SU(2)
Wilson lattice charge is **γ_W = (-1, -2, 0, 0, 0)** — verified
ρ-fixed and satisfying `F_γ · F_γ = F_{2γ} + F_0` (Wilson closure).
So `Tr(W_n)` via the Schur F here corresponds to `BPS.trace(n · γ_W)`,
NOT `BPS.trace((0, -n, 0, 0, 0))` (which is a pure tropical cone).

Verified for n=0,1,2 at q_max≤4 against the BPS oracle.

Output: `RLaurent` over `AbelianZPlusRing(rank=3)` (rank-3 SU(4)
Cartan); for SU(4) irrep characters, post-process via S_4-Weyl
symmetrisation (separate utility).
"""
from __future__ import annotations

from zplus_ring import AbelianZPlusRing, RLaurent

_R = AbelianZPlusRing(rank=3)                  # rank-3 SU(4) Cartan (μ_1, μ_2, μ_3).
_Q_MAX_DEFAULT = 12
_V_MAX_DEFAULT = 16

WILSON_LATTICE = (-1, -2, 0, 0, 0)
"""Fundamental SU(2) Wilson lattice charge γ_W on the BPS quiver.

ρ-fixed; F_{γ_W} · F_{γ_W} = F_{2 γ_W} + F_0 (Wilson closure).
The n-th Wilson `W_n` lives at lattice `n · γ_W = (-n, -2n, 0, 0, 0)`.
"""


def _rl_one() -> RLaurent:
    return RLaurent(_R, {0: _R.one()})


def _rl_zero() -> RLaurent:
    return RLaurent(_R, {})


def _rl_q_mu(q_pow: int, mu_pows=(0, 0, 0)) -> RLaurent:
    return RLaurent(_R, {q_pow: _R.basis_element(tuple(mu_pows))})


def _rl_truncate_qmu(r: RLaurent, q_max: int) -> RLaurent:
    return RLaurent(_R, {e: c for e, c in r.coeffs.items() if e <= q_max})


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
    """`(a; q²)_∞`."""
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
    """`1/(a; q²)_∞`."""
    result = {0: _rl_one()}
    k = 0
    while 2 * k <= q_max:
        q2k = _rl_q_mu(2 * k)
        a_shifted = {v: p * q2k for v, p in a_dict.items()}
        inv_factor = {0: _rl_one()}
        cur = {0: _rl_one()}
        while True:
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
    """SU(2)+Nf=3 Schur F(v, μ) — matter is the (2_gauge, 6_v)
    of SU(2) × Spin(6), with 6_v = ∧² SU(4)-fund.

    Matter weight set: μ^{w_i + w_j} for 1 ≤ i < j ≤ 4 (the 6 weights
    of ∧²4 = Spin(6) vector), each × v^± from the SU(2) gauge fund,
    giving 12 chiral pairs (= 6 hypers).

        F = (q²v²;q²)_∞² · (q²/v²;q²)_∞² · (q²;q²)_∞²
            / ∏_{σ=±, i<j} (−q v^σ μ^{w_i + w_j}; q²)_∞

    Verified vs `bps_su2_nf3.trace(n · γ_W)` at Wilson lattice
    γ_W = (-1, -2, 0, 0, 0) for n = 0, 1, 2 (q_max ≤ 4):
      Tr(identity)[q²] = 12 SU(4) roots + 3 = χ_{adjoint(SU(4))}
      Tr(W_1)[q¹]      = −χ_{6_v} = −Σ_{i<j} μ^{w_i+w_j}
      Tr(W_2)[q²]      = 10 weights of Sym²(6_v) (matter rep squared).
    """
    key = (q_max, v_max)
    if key in _F_cache:
        return _F_cache[key]
    P_v2  = _pochhammer_qsq({ 2: _rl_q_mu(2)}, q_max, v_max)
    P_v_2 = _pochhammer_qsq({-2: _rl_q_mu(2)}, q_max, v_max)
    P_0   = _pochhammer_qsq({ 0: _rl_q_mu(2)}, q_max, v_max)
    F = _f_mul(_power_dict(P_v2,  2, v_max, q_max),
               _power_dict(P_v_2, 2, v_max, q_max), v_max, q_max)
    F = _f_mul(F, _power_dict(P_0, 2, v_max, q_max), v_max, q_max)
    fund_weights = [
        ( 1,  0,  0),                          # w_1
        (-1,  1,  0),                          # w_2
        ( 0, -1,  1),                          # w_3
        ( 0,  0, -1),                          # w_4
    ]
    antisym_weights = [
        tuple(fund_weights[i][k] + fund_weights[j][k] for k in range(3))
        for i in range(4) for j in range(i + 1, 4)
    ]
    for sigma in (+1, -1):
        for w in antisym_weights:
            arg = {sigma: -_rl_q_mu(1, w)}
            inv = _pochhammer_inverse_qsq(arg, q_max, v_max)
            F = _f_mul(F, inv, v_max, q_max)
    _F_cache[key] = F
    return F


def tr_W(n: int, q_max: int = _Q_MAX_DEFAULT) -> RLaurent:
    """`Tr(W_n)` for SU(2)+Nf=3 via the standard Wilson projection
    `Tr(W_n) = [v^n] F − [v^{n+2}] F`.

    Matter is the SO(6)-vector (real rep), so F is invariant under
    v ↔ v^{-1} combined with μ↔μ^{-1} (Weyl flavour); the SU(2) Weyl
    projection of F·χ_n(v) onto v^0 reduces to the standard difference
    of v-coefficients.

    Output: `RLaurent` over rank-3 SU(4) Cartan.
    """
    v_max = max(_V_MAX_DEFAULT, n + 6)
    F = _build_F(q_max, v_max)
    a = F.get(n, _rl_zero())
    b = F.get(n + 2, _rl_zero())
    return _rl_truncate_qmu(a - b, q_max)
