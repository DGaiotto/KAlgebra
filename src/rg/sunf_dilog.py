"""`sunf_dilog` — the SU(N_f)-flavoured spectrum generator of `N_f` collinear
hypermultiplets in the **fundamental** of `SU(N_f)`: the image of

    S  =  ∏_{i=1}^{N_f}  E_𝖖(μ_i · x)

(the `μ_i` = the `N_f` weights of the `SU(N_f)` fundamental) under the
**Weyl-symmetric `μ → χ` map**, i.e. re-expressed in `SU(N_f)` irrep characters.

The level-`N` coefficient `[x^N]` is `Σ_{a_1+…+a_{N_f}=N} (∏_i c_{a_i}) μ^{(a_1,…)}`
(`c_m = [x^m]E_𝖖(x)`); the `SU(N_f)`-irrep content is read off by the
**bialternant / Weyl peel** (`I·δ = Σ_λ c_λ N_λ`, `c_λ = [x^{λ+ρ}](I·δ)`),
carried with **exact `HabiroElement` coefficients** and normalised to `SU(N_f)`
partition labels.  General-purpose: this is the `S_RG` of any RG flow dropping an
`SU(N_f)` fundamental of collinear hypers over a gauged U(1) — `SQED_{N_f}`
(`SQEDNfRGKAlgebra`).

`N_f = 2` reduces **exactly** to the SU(2)-doublet dilogarithm (the two
fundamental weights `μ^{±1}`); `N_f = 1` is the bare `E_𝖖(x)` (trivial flavour).
Built on `sun_characters` (the type-A Weyl/Schur machinery) + exact arithmetic
in the localization `Z[𝖖^±][(1−𝖖^{2n})^{-1}, n ≥ 1]` (`HabiroElement`).
"""

from __future__ import annotations

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from laurent_poly import LaurentPoly
from habiro import HabiroElement
import sun_characters as _sun

__all__ = ["eq_coeff", "sunf_components"]


def eq_coeff(m: int) -> HabiroElement:
    """`c_m = [x^m] E_𝖖(x) = (−𝖖)^m / (𝖖²;𝖖²)_m` — the m-th quantum-dilogarithm
    coefficient for a **self-pairing-free** generator (a single unit-coefficient
    label per `x^m`).  As a `HabiroElement`: numerator `(−1)^m 𝖖^m`, denominator
    `∏_{j=1}^{m}(1 − 𝖖^{2j})`."""
    if m < 0:
        raise ValueError(f"eq_coeff requires m >= 0, got {m}")
    return HabiroElement(LaurentPoly({m: (-1) ** m}),
                         {j: 1 for j in range(1, m + 1)})


def _compositions(N: int, k: int):
    """All length-`k` tuples of non-negative ints summing to `N`."""
    if k == 1:
        yield (N,)
        return
    for first in range(N + 1):
        for rest in _compositions(N - first, k - 1):
            yield (first,) + rest


def _norm_su_weight(lam: tuple, Nf: int) -> tuple:
    """`U(N_f)` dominant weight → `SU(N_f)` partition label (subtract the last
    part, strip trailing zeros).  The det direction is trivial in SU."""
    if not lam:
        return ()
    base = lam[-1]
    part = tuple(x - base for x in lam[:-1])
    while part and part[-1] == 0:
        part = part[:-1]
    return part


def sunf_components(Nf: int, N: int) -> dict:
    """The `SU(N_f)`-irrep decomposition of `[x^N] ∏_{i=1}^{N_f} E_𝖖(μ_i x)`
    under the Weyl-symmetric `μ → χ` map: `{SU(N_f) partition label: coeff}` with
    **exact `HabiroElement` coefficients**.

    Empty for `N < 0`; `{(): 1}` (the singlet) for `N = 0`.  This is the level-`N`
    graded component `[S_RG]_N` of the `SU(N_f)`-fundamental matter spectrum
    generator, up to the carrier label `x^N` — exactly the dict a flow's
    `_s_rg_component` wraps.

    `N_f = 1`: SU(1) is trivial, so every level is the singlet `()` with
    coefficient `c_N`.
    """
    if N < 0:
        return {}
    if Nf == 1:
        c = eq_coeff(N)
        return {(): c} if not c.is_zero() else {}

    # I = Σ_{Σa_i=N} (∏_i c_{a_i})  x^{(a_1,…,a_{N_f})}   (weight = exponent vector)
    I: dict = {}
    for comp in _compositions(N, Nf):
        coeff = eq_coeff(comp[0])
        for a in comp[1:]:
            coeff = coeff * eq_coeff(a)
        if coeff.is_zero():
            continue
        I[comp] = I.get(comp, HabiroElement.zero()) + coeff

    # I·δ, then read c_λ at each strictly-decreasing β = λ+ρ (Weyl bialternant),
    # carrying the HabiroElement coefficients exactly.
    delta = _sun.weyl_denominator(Nf)               # {exp-tuple: int}
    prod: dict = {}
    for we, hc in I.items():
        if hc.is_zero():
            continue
        for de, dc in delta.items():
            e = tuple(we[i] + de[i] for i in range(Nf))
            term = HabiroElement(hc.numerator * LaurentPoly({0: dc}), dict(hc.denom))
            prod[e] = prod.get(e, HabiroElement.zero()) + term

    r = _sun.rho(Nf)
    out: dict = {}
    for beta, hc in prod.items():
        if hc.is_zero() or not _sun._is_strictly_decreasing(beta):
            continue
        lam = tuple(beta[i] - r[i] for i in range(Nf))
        label = _norm_su_weight(lam, Nf)
        out[label] = out.get(label, HabiroElement.zero()) + hc
    return {k: v for k, v in out.items() if not v.is_zero()}
