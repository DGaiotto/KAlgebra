"""`su2_doublet_dilog` — the SU(2)-flavoured spectrum generator of a collinear
hypermultiplet **doublet**, i.e. the image of

    S  =  E_𝖖(μ·x) · E_𝖖(μ⁻¹·x)

under the **Weyl-symmetric μ → χ map** (`μ + μ⁻¹ → χ₁`, etc.).  Each `x^N`
coefficient is the Weyl-symmetric μ-Laurent polynomial
`Σ_{a+b=N} c_a c_b μ^{a−b}` (`c_m = [x^m]E_𝖖(x)`), re-expressed in SU(2) irrep
characters `χ_κ` (highest weight `κ = a−b`, spin `κ/2`).

General-purpose tool.  This is the `S_RG` of *any* RG flow that drops an SU(2)
doublet of **collinear** hypers (weights `μ^{±1}` on the same gauge charge)
over a gauged U(1), where the doublet's μ-content combines into honest SU(2)
characters: SQED₂ (`U1A1D2RGKAlgebra`), `[A₁, D_{2k+3}]`
(`A1DoddRGKAlgebra`), … — the carrier label (which `x^N`) is supplied by the
flow; the **SU(2)-character content per level** is exactly what this module
computes.

(Distinct from the U(2)-Cartan presentation with two *independent* fugacities
`μ₁, μ₂` — the abelian `SimpleSRGFlow` `su2` mode, SU(2) recognised after.
Here a single `μ` with weights `±1` yields `R(SU(2))` natively via the
inverse-Kostka / Weyl peel below.)
"""

from __future__ import annotations

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from laurent_poly import LaurentPoly
from habiro import HabiroElement

__all__ = ["eq_coeff", "su2_char_coeff", "su2_doublet_components"]


def eq_coeff(m: int) -> HabiroElement:
    """`c_m = [x^m] E_𝖖(x) = (−𝖖)^m / (𝖖²;𝖖²)_m` — the m-th quantum-dilogarithm
    coefficient for a **self-pairing-free** generator (`⟨γ,γ⟩ = 0`, so `x^m` is
    a single unit-coefficient label).  Habiro form: numerator `(−1)^m 𝖖^m`,
    denominator `∏_{j=1}^{m}(1 − 𝖖^{2j})`."""
    if m < 0:
        raise ValueError(f"eq_coeff requires m >= 0, got {m}")
    return HabiroElement(LaurentPoly({m: (-1) ** m}),
                         {j: 1 for j in range(1, m + 1)})


def su2_char_coeff(a: int, b: int) -> HabiroElement:
    """The SU(2)-irrep peel coefficient `c_a c_b − c_{a+1} c_{b-1}` (`a ≥ b ≥ 0`):
    the (Habiro) multiplicity of the highest-weight-`(a−b)` SU(2) irrep in
    `Σ_{m+n=a+b} c_m c_n μ^{m−n}` — the inverse-Kostka / Weyl peel from the top
    weight (`c_{-1} = 0`, handled by the `b ≥ 1` guard)."""
    term = eq_coeff(a) * eq_coeff(b)
    if b >= 1:
        term = term - eq_coeff(a + 1) * eq_coeff(b - 1)
    return term


def su2_doublet_components(N: int) -> dict[int, HabiroElement]:
    """The SU(2)-irrep decomposition of `[x^N] (E_𝖖(μx)·E_𝖖(μ⁻¹x))` under the
    Weyl-symmetric `μ → χ` map: `{κ: coeff}` over highest weights
    `κ ∈ {N, N−2, …, 0 or 1}` (spin `κ/2`), with
    `coeff = c_a c_b − c_{a+1} c_{b-1}`, `a = (N+κ)/2`, `b = (N−κ)/2`.

    Empty for `N < 0`; `{0: 1}` (the SU(2) singlet / identity) for `N = 0`.
    This is the level-`N` graded component `[S_RG]_N` of the doublet spectrum
    generator, up to the flow's carrier label `x^N` and the SU(2) flavour label
    `κ` — exactly the dict a flow's `_s_rg_component` wraps."""
    if N < 0:
        return {}
    out: dict[int, HabiroElement] = {}
    for kappa in range(N, -1, -2):
        a, b = (N + kappa) // 2, (N - kappa) // 2
        coeff = su2_char_coeff(a, b)
        if not coeff.is_zero():
            out[kappa] = coeff
    return out
