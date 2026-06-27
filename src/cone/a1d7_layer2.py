"""Explicit Layer-2 characters for [A₁, D₇] = sl(2)₋₁₂/₇ (closed form, BPS-free).

The v=7 sibling of `a1d5_layer2`: the six elementary traces of
`FiniteA1D7KAlgebra` (the ρ²-orbit-representative single-mult-gen traces
`Tr(seed_i)`, i = 0..5) plus the vacuum, as explicit affine sl(2)₋₁₂/₇
admissible-character combinations — exact to arbitrary q-order, no bootstrap,
no BPS.  Same construction as D₅, now using the **three** discrete modules
s=1,2,3 (vs D₅'s two):

    Tr(1) = κ₀
    seed0 = κ₀[−1] + κ₁ˢʸᵐ[−3] − κ₂ˢʸᵐ[−3] − (χ₁·κ₂ᵃⁿᵗⁱ)[−3]
                   − κ₃ˢʸᵐ[−1] − (χ₁·κ₃ᵃⁿᵗⁱ)[−1]
    seed1 = κ₀     + κ₁ˢʸᵐ[−2] − κ₃ˢʸᵐ[−2] − (χ₁·κ₃ᵃⁿᵗⁱ)[−2]
    seed2 = κ₀[−1]             − κ₃ˢʸᵐ[−1] − (χ₁·κ₃ᵃⁿᵗⁱ)[−1]
    seed3 = −κ₁ᵃⁿᵗⁱ[−1] − κ₂ᵃⁿᵗⁱ[−1] − κ₃ᵃⁿᵗⁱ[−1]
    seed4 = −κ₁ᵃⁿᵗⁱ[−2] − κ₂ᵃⁿᵗⁱ[−2]
    seed5 = −κ₂ᵃⁿᵗⁱ[−3]

κ's built from the σⱼ(u=2,v=7,s) numerator formula (shared with a1d5_layer2);
`[n]` = 𝖖ⁿ grading shift; χ₁· = ⊗ SU(2) doublet.  Validated to q¹⁵ against the
reliable su2 bootstrap (`finite_kalgebras.su2_reliable`).
"""
from __future__ import annotations

from fractions import Fraction as Fr

from a1d3_kalg import (
    _bps_verma, _laurent_mul, _divide_each, _divide_by_1_minus_mu2,
    _divide_by_mu_minus_muinv, _laurent_clean, _sigma_j_add,
    _laurent_combine, _laurent_negate, _laurent_reflect_mu, _laurent_shift_mu,
)

_U, _V = 2, 7                      # k+2 = u/v = 2/7  (sl(2)₋₁₂/₇)


def _sigma(K: int, s: int, u: int = _U, v: int = _V) -> dict:
    out: dict = {}
    jb = max(8, int((K // (2 * u * v)) ** 0.5) + 6)
    for j in range(-jb, jb + 1):
        e1 = 2 * u * v * j * j + (-2 * v + 4 * u * s) * j
        if 0 <= e1 <= K:
            _sigma_j_add(out, e1, 2 * u * j, +1)
        e2 = 2 * u * v * j * j + (-6 * v + 4 * u * s) * j + (2 * v - 2 * u * s)
        if 0 <= e2 <= K:
            _sigma_j_add(out, e2, 2 * u * j - 2, -1)
    return _laurent_clean(out)


def kappa0(K: int) -> dict:
    return _laurent_mul(_bps_verma(K),
                        _divide_each(_sigma(K, 0), _divide_by_1_minus_mu2), K)


def kappa_sym(K: int, s: int) -> dict:
    sp = _sigma(K, s)
    sm = _laurent_reflect_mu(sp)
    num = _divide_each(_laurent_combine(sp, _laurent_shift_mu(_laurent_negate(sm), 2)),
                       _divide_by_1_minus_mu2)
    return _laurent_mul(_bps_verma(K), num, K)


def kappa_anti(K: int, s: int) -> dict:
    sp = _sigma(K, s)
    sm = _laurent_reflect_mu(sp)
    num = _divide_each(_laurent_combine(sp, _laurent_negate(sm)),
                       _divide_by_mu_minus_muinv)
    return _laurent_mul(_bps_verma(K), num, K)


def _shift(d, n):
    return {q + n: mud for q, mud in d.items()}


def _chi1(d):
    return _laurent_combine(_laurent_shift_mu(d, 1), _laurent_shift_mu(d, -1))


def _scale(d, c):
    return {q: {m: c * v for m, v in mud.items()} for q, mud in d.items()}


# recipe: (builder, s, q-shift, χ₁-dress?, coeff).  Validated to q¹⁵.
_SEED_RECIPE = {
    "vac": [("dir", 0, 0, False, 1)],
    0: [("dir", 0, -1, False, 1), ("sym", 1, -3, False, 1),
        ("sym", 2, -3, False, -1), ("anti", 2, -3, True, -1),
        ("sym", 3, -1, False, -1), ("anti", 3, -1, True, -1)],
    1: [("dir", 0, 0, False, 1), ("sym", 1, -2, False, 1),
        ("sym", 3, -2, False, -1), ("anti", 3, -2, True, -1)],
    2: [("dir", 0, -1, False, 1), ("sym", 3, -1, False, -1),
        ("anti", 3, -1, True, -1)],
    3: [("anti", 1, -1, False, -1), ("anti", 2, -1, False, -1),
        ("anti", 3, -1, False, -1)],
    4: [("anti", 1, -2, False, -1), ("anti", 2, -2, False, -1)],
    5: [("anti", 2, -3, False, -1)],
}


def _build(key, K: int) -> dict:
    out: dict = {}
    for kind, s, sh, dress, c in _SEED_RECIPE[key]:
        base = (kappa0(K - sh) if kind == "dir"
                else kappa_sym(K - sh, s) if kind == "sym"
                else kappa_anti(K - sh, s))
        if dress:
            base = _chi1(base)
        out = _laurent_combine(out, _scale(_shift(base, sh), c))
    return {q: mud for q, mud in _laurent_clean(out).items() if q <= K}


def _fug_to_su2(mud):
    md: dict = {}
    for p, c in mud.items():
        if Fr(p).denominator != 1:
            return None
        md[int(p)] = md.get(int(p), 0) + c
    coeffs: dict = {}
    guard = 0
    while any(c for c in md.values()):
        guard += 1
        if guard > 4000:
            return None
        mx = max(p for p, c in md.items() if c)
        if mx < 0:
            return None
        c = md[mx]
        coeffs[mx] = coeffs.get(mx, 0) + c
        for k in range(mx, -mx - 1, -2):
            md[k] = md.get(k, 0) - c
        md = {p: cc for p, cc in md.items() if cc}
    return {n: c for n, c in coeffs.items() if c}


def _to_irrep(qmu, K):
    out = {}
    for q, mud in qmu.items():
        if q > K:
            continue
        ch = _fug_to_su2(mud)
        if ch is None:
            raise RuntimeError(f"a1d7_layer2: non-symmetric μ-content at 𝖖^{q}")
        if ch:
            out[q] = ch
    return out


def vacuum_trace(K: int) -> dict:
    """Tr(1) = κ₀, the sl(2)₋₁₂/₇ vacuum character, to 𝖖-order K."""
    return _to_irrep(_build("vac", K), K)


def seed_trace(idx: int, K: int) -> dict:
    """Elementary trace `Tr(seed_idx)` (idx ∈ {0..5}) to 𝖖-order K."""
    return _to_irrep(_build(idx, K), K)
