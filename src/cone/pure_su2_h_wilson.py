"""SU(2) Wilson character cone: Chebyshev / Weyl-character utilities.

The Wilson sector of pure SU(2) BPS K-algebra is a **character cone**:
canonical basis elements are the irreducible characters `χ_e` for
`e ≥ 0` (spin-`e/2` reps), expressed as q-Chebyshev polynomials of the
fundamental ray `w_1`:

    χ_0 = 1
    χ_1 = w_1
    χ_n = w_1 · χ_{n-1} − χ_{n-2}      (SU(2) Chebyshev / Weyl recursion)

Inverse (PBW monomials in characters):

    w_1^k = Σ_{e ≡ k (mod 2), 0 ≤ e ≤ k} mult(k, e) · χ_e

where `mult(k, e) = C(k, (k-e)/2) − C(k, (k-e)/2 − 1)` (SU(2) Young
tableau count = Catalan-style binomial difference).

Wilsons q-commute trivially with each other (DSZ pairing 0) so the
cone is "abelian"; the non-trivial structure is the Clebsch-Gordan
rule, packaged here as the χ-basis decomposition.

This module provides the basic conversion routines; integration into
`PureSU2HConeData` as a `CharacterCone` is left to the caller.
"""
from __future__ import annotations
from math import comb

from laurent_poly import LaurentPoly


# ---------- SU(2) Clebsch / Chebyshev conversion -----------------------

def chi_to_w1_powers(e: int) -> dict[int, int]:
    """χ_e expanded as a Z-polynomial in w_1.

    Returns `{k: coef}` such that χ_e = Σ_k coef · w_1^k.  Uses the
    Chebyshev recursion χ_n = w_1·χ_{n−1} − χ_{n−2}.
    """
    if e < 0:
        raise ValueError(f"chi_to_w1_powers: e must be ≥ 0 (got {e})")
    if e == 0:
        return {0: 1}
    if e == 1:
        return {1: 1}
    prev2 = {0: 1}      # χ_0
    prev1 = {1: 1}      # χ_1
    for n in range(2, e + 1):
        # χ_n = w_1·prev1 − prev2
        # w_1·prev1: shift powers by +1
        shifted = {k + 1: v for k, v in prev1.items()}
        # subtract prev2
        new = dict(shifted)
        for k, v in prev2.items():
            new[k] = new.get(k, 0) - v
        new = {k: v for k, v in new.items() if v != 0}
        prev2, prev1 = prev1, new
    return prev1


def w1_power_to_chi(k: int) -> dict[int, int]:
    """w_1^k expanded as a Z-linear combination of χ_e's (SU(2) Clebsch).

    Returns `{e: mult}` such that w_1^k = Σ_e mult · χ_e.
    Multiplicity: mult(k, e) = C(k, (k-e)/2) − C(k, (k-e)/2 − 1) for
    k ≥ e ≥ 0 with k − e even; zero otherwise.
    """
    if k < 0:
        raise ValueError(f"w1_power_to_chi: k must be ≥ 0 (got {k})")
    out: dict[int, int] = {}
    for e in range(k % 2, k + 1, 2):
        j = (k - e) // 2
        mult = comb(k, j) - (comb(k, j - 1) if j >= 1 else 0)
        if mult != 0:
            out[e] = mult
    return out


def chi_clebsch(a: int, b: int) -> dict[int, int]:
    """SU(2) Clebsch: χ_a · χ_b = Σ_c χ_c for c = |a−b|, |a−b|+2, ..., a+b.

    All multiplicities are 1.
    """
    if a < 0 or b < 0:
        raise ValueError(f"chi_clebsch: indices must be ≥ 0 (got {a}, {b})")
    lo = abs(a - b)
    hi = a + b
    return {c: 1 for c in range(lo, hi + 1, 2)}


# ---------- Native-label convention for the Wilson cone -----------------
#
# Within `PureSU2HConeData` (extended), we represent Wilson canonical-
# basis elements with the native-label tag `('W', e)` for `χ_e`.  The
# fundamental ray (the cone's PBW generator) is `('W', 1)`.  PBW
# monomials `w_1^k` correspond to native label `(('W', 1), k)`-style;
# we keep the canonical (χ-basis) form as the persistent representation.


WILSON_FUND = ('W', 1)


def chi_label(e: int) -> tuple:
    """Native label for χ_e: `('W', e)`."""
    return ('W', e)


def is_wilson_label(g) -> bool:
    """True iff `g` is a Wilson generator label of form ('W', e)."""
    return isinstance(g, tuple) and len(g) == 2 and g[0] == 'W' and isinstance(g[1], int)


# ---------- Polynomial structure constant for χ_a · χ_b ----------------

def chi_mul_chi(a: int, b: int) -> dict[tuple, LaurentPoly]:
    """χ_a · χ_b decomposition in the canonical χ basis.

    Returns `{('W', c): LaurentPoly}` with q-coefficient = 1 (since the
    SU(2) Clebsch has trivial q-deformation in the pSU(2) chamber for
    even-charge Wilsons).
    """
    out: dict[tuple, LaurentPoly] = {}
    for c, mult in chi_clebsch(a, b).items():
        out[chi_label(c)] = LaurentPoly({0: mult})
    return out
