"""Explicit Layer-2 characters for [A₁, D₅] = sl(2)₋₈/₅ (closed form, BPS-free).

The four elementary traces of `FiniteA1D5KAlgebra` (the ρ²-orbit-representative
single-mult-gen traces `Tr(seed_i)`, i = 0..3) plus the vacuum `Tr(1)` as
**explicit affine sl(2)₋₈/₅ admissible-character combinations** — exact to any
q-order, no bootstrap, no BPS:

    Tr(1)   = κ₀                                              (vacuum)
    seed0   = κ₀          + κ₁^sym[−2] − κ₂^sym[−2] − (χ₁·κ₂^anti)[−2]
    seed1   = κ₀[−1]                   − κ₂^sym[−1] − (χ₁·κ₂^anti)[−1]
    seed2   = −κ₁^anti[−1] − κ₂^anti[−1]
    seed3   = −κ₁^anti[−2]

where (everything in Cartan fugacity μ, q_paper grade = 𝖖²):

  * κ₀ = the sl(2)₋₈/₅ **vacuum** character (Kac–Wakimoto N/D, base ρ);
  * κ_s^{sym,anti} = the **discrete-module** characters built from the
    discrete-module numerator
        σ_j(u,v,s) = Σ_j[ +𝖖^{2uv j²+(−2v+4us)j}·μ^{2uj}
                          −𝖖^{2uv j²+(−6v+4us)j+(2v−2us)}·μ^{2uj−2} ]
    (u=2, v=5; the same formula reproduces a3's _sigma_j_L10 / _sigma_j_Dplus11
    at v=3 exactly), via the a3 recipe shapes
        κ_s^sym  = verma·(σ_s − μ²σ̄_s)/(1−μ²),
        κ_s^anti = verma·(σ_s − σ̄_s)/(μ−μ⁻¹),    σ̄ = reflect μ→μ⁻¹;
  * `[n]` = the q-grading shift 𝖖ⁿ; `χ₁·` = tensor with the SU(2) doublet.

The verma/division helpers are shared with `a1d3_kalg`.  This is the v=5
analogue of `a1d3_kalg`'s explicit su(2)₋₄/₃ Layer-2 (κ₀/κ₁^sym/κ₁^anti).
Derivation + validation history: `a1dodd_layer2_status.md`.
"""
from __future__ import annotations

from fractions import Fraction as Fr

from a1d3_kalg import (
    _bps_verma, _laurent_mul, _divide_each, _divide_by_1_minus_mu2,
    _divide_by_mu_minus_muinv, _laurent_clean, _sigma_j_add,
    _laurent_combine, _laurent_negate, _laurent_reflect_mu, _laurent_shift_mu,
)

_U, _V = 2, 5                      # k+2 = u/v = 2/5  (sl(2)₋₈/₅)

# ---------------------------------------------------------------------------
# discrete-module numerator σ_j and the κ_s building blocks
# ---------------------------------------------------------------------------

def _sigma(K: int, s: int, u: int = _U, v: int = _V) -> dict:
    """σ_j(u,v,s) numerator theta as {𝖖-power: {μ-power: coeff}}."""
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


def _verma(K: int) -> dict:
    return _bps_verma(K)


def kappa0(K: int) -> dict:
    """Vacuum character κ₀ = verma·σ₀/(1−μ²)  (= char_fug base ρ)."""
    return _laurent_mul(_verma(K), _divide_each(_sigma(K, 0),
                                                _divide_by_1_minus_mu2), K)


def kappa_sym(K: int, s: int) -> dict:
    sp = _sigma(K, s)
    sm = _laurent_reflect_mu(sp)
    num = _divide_each(_laurent_combine(sp, _laurent_shift_mu(_laurent_negate(sm), 2)),
                       _divide_by_1_minus_mu2)
    return _laurent_mul(_verma(K), num, K)


def kappa_anti(K: int, s: int) -> dict:
    sp = _sigma(K, s)
    sm = _laurent_reflect_mu(sp)
    num = _divide_each(_laurent_combine(sp, _laurent_negate(sm)),
                       _divide_by_mu_minus_muinv)
    return _laurent_mul(_verma(K), num, K)


def _shift(d: dict, n: int) -> dict:
    return {q + n: mud for q, mud in d.items()}


def _chi1(d: dict) -> dict:
    """⊗ the SU(2) doublet χ₁ = μ + μ⁻¹."""
    return _laurent_combine(_laurent_shift_mu(d, 1), _laurent_shift_mu(d, -1))


def _scale(d: dict, c: int) -> dict:
    return {q: {m: c * v for m, v in mud.items()} for q, mud in d.items()}


# ---------------------------------------------------------------------------
# the four elementary-trace seeds + vacuum, as (q, μ)-Laurent
# ---------------------------------------------------------------------------
# Each entry: list of (builder, s, q-shift, χ₁-dress?, coeff).  Validated to
# q²⁶ against the reliable bootstrap oracle (a1dodd_layer2_status.md).
_SEED_RECIPE = {
    "vac": [("dir", 0, 0, False, 1)],
    0:     [("dir", 0, 0, False, 1), ("sym", 1, -2, False, 1),
            ("sym", 2, -2, False, -1), ("anti", 2, -2, True, -1)],
    1:     [("dir", 0, -1, False, 1), ("sym", 2, -1, False, -1),
            ("anti", 2, -1, True, -1)],
    2:     [("anti", 1, -1, False, -1), ("anti", 2, -1, False, -1)],
    3:     [("anti", 1, -2, False, -1)],
}


def _build(key: str | int, K: int) -> dict:
    """Assemble a seed/vacuum recipe as a (q, μ)-Laurent to 𝖖-order K."""
    out: dict = {}
    for kind, s, sh, dress, c in _SEED_RECIPE[key]:
        base = (kappa0(K - sh) if kind == "dir"
                else kappa_sym(K - sh, s) if kind == "sym"
                else kappa_anti(K - sh, s))
        if dress:
            base = _chi1(base)
        out = _laurent_combine(out, _scale(_shift(base, sh), c))
    return {q: mud for q, mud in _laurent_clean(out).items() if q <= K}


def _fug_to_su2(mud: dict):
    """Weyl-symmetric integer-μ Laurent {μ:c} → {SU(2) irrep n: c} (or None)."""
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


# ---------------------------------------------------------------------------
# public API: elementary traces in SU(2)-irrep form {𝖖-power: {n: int}}
# ---------------------------------------------------------------------------

def vacuum_trace(K: int) -> dict:
    """Tr(1) = κ₀, the sl(2)₋₈/₅ vacuum character, to 𝖖-order K."""
    return _to_irrep(_build("vac", K), K)


def seed_trace(idx: int, K: int) -> dict:
    """Elementary trace `Tr(seed_idx)` (idx ∈ {0,1,2,3}) to 𝖖-order K."""
    return _to_irrep(_build(idx, K), K)


def _to_irrep(qmu: dict, K: int) -> dict:
    out = {}
    for q, mud in qmu.items():
        if q > K:
            continue
        ch = _fug_to_su2(mud)
        if ch is None:
            raise RuntimeError(f"a1d5_layer2: non-symmetric μ-content at 𝖖^{q}")
        if ch:
            out[q] = ch
    return out
