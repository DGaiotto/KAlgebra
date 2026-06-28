"""Tropical sigma — chart-internal spectrum-generator-induced map on Γ.

Canonical-surface migration of `lattice_canonical.{sigma_forward,
sigma_inverse}` (Plan 07 Stage A4).  Self-contained: only depends
on `lattice.Lattice` for the bracket pairing.

For a spec ``[g_1, …, g_N]`` and a charge γ ∈ Γ:

    σ(γ)     =  −(μ_N^t ∘ … ∘ μ_1^t)(γ),
    σ^{-1}(γ) =  apply (μ_k^t)^{-1} in reverse order to −γ,

where μ_k^t(α) = α + max(⟨α, g_k⟩, 0) · g_k.
"""

from __future__ import annotations

from typing import Sequence

from lattice import Lattice, Vec


def _trop_mut_lower(alpha: Vec, gk: Vec, lattice: Lattice) -> Vec:
    """Tropical mutation acting on alpha with pairing point gk (first slot).

        alpha -> alpha + max(<alpha, gk>, 0) * gk
    """
    m = lattice.bracket(alpha, gk)
    if m <= 0:
        return tuple(alpha)
    return tuple(a + m * g for a, g in zip(alpha, gk))


def sigma_forward(lattice: Lattice, spec: Sequence[Sequence[int]],
                  gamma: Sequence[int]) -> Vec:
    """sigma(gamma) = - (mu_N^t o ... o mu_1^t)(gamma).

    Uses the spectrum-generator ordering spec = [g_1, ..., g_N] with
    mu_k^t acting by alpha -> alpha + max(<alpha, g_k>, 0) * g_k.
    """
    c: Vec = lattice.check(gamma)
    for g in spec:
        c = _trop_mut_lower(c, lattice.check(g), lattice)
    return tuple(-x for x in c)


def sigma_inverse(lattice: Lattice, spec: Sequence[Sequence[int]],
                  gamma: Sequence[int]) -> Vec:
    """sigma^{-1}(gamma): negate, then (mu_k^t)^{-1} in reverse order.

        (mu_k^t)^{-1}(alpha) = alpha - max(<alpha, g_k>, 0) * g_k
    """
    c: Vec = tuple(-x for x in lattice.check(gamma))
    for g in reversed(list(spec)):
        gk = lattice.check(g)
        m = lattice.bracket(c, gk)
        if m > 0:
            c = tuple(a - m * gi for a, gi in zip(c, gk))
    return c
