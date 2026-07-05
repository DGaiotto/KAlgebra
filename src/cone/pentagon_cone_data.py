"""Cone-filtration data for the pentagon K-algebra `A_𝖖([A_1, A_2])`.

The pentagon's cone-filtration structure (recap; matches the empirical
validation in `scripts/test_pentagon_from_cone_data.py`):

  * Multiplicative generators: `L_0, …, L_4`, indexed by `i ∈ ℤ/5`.
  * Cones (maximal q-commuting subsets of mult-gens):
        C_i = {L_i, L_{i+1 mod 5}}        i ∈ ℤ/5.
    Five cones, each of size 2 — the five clusters of the A_2 cluster
    algebra.
  * q-commute cocycle: `L_i L_{i+1} = q⁻² L_{i+1} L_i`,
    so `cocycle(L_i, L_{i+1}) = -1`.
  * Cross-products (distance 2 in ℤ/5):
        L_i L_{i+2} = 1 + q⁻¹ L_{i+1}     (di = 2)
        L_i L_{i-2} = 1 + q   L_{i-1}     (di = 3)
  * Native canonical-basis labels: `(i, a, b) ↔ L_{i;a,b} = q^{ab}
    L_i^a L_{i+1}^b`.  The cone-anchored form `L_i^a L_{i+1}^b` (in
    canonical order `L_i` before `L_{i+1}`) is the literal-word
    interpretation of `(gens={i, i+1}, powers={i: a, i+1: b})`; the
    `q^{ab}` phase is absorbed into the *definition* of `L_{i;a,b}`, so
    the cone-label bijection itself has no implicit phase factor.

Wrap-around cone `{4, 0}`: pentagon's convention has `L_{4;a,b} = q^{ab}
L_4^a L_0^b`, so the canonical-order tuple is `(4, 0)` rather than
Python's default `sorted({4, 0}) == (0, 4)`.  `canonical_cone_order` is
overridden for this case.
"""

from __future__ import annotations

from typing import Sequence

from cone_data import ConeData, CrossProductTerm, FiniteConeData
from laurent_poly import LaurentPoly


__all__ = ["PentagonConeData", "PENTAGON_CONE_DATA"]


PentagonLabel = tuple[int, int, int]
"""Pentagon canonical-basis label `(i, a, b)` for `L_{i;a,b}`."""


def _idx5(i: int) -> int:
    return i % 5


class PentagonConeData(FiniteConeData):
    """`ConeData` for `PentagonKAlg`."""

    # -- finite enumeration ------------------------------------------------

    def mult_gens(self) -> Sequence[int]:
        return (0, 1, 2, 3, 4)

    def cones(self) -> Sequence[frozenset[int]]:
        return tuple(frozenset({i, (i + 1) % 5}) for i in range(5))

    # Pentagon's convention `L_{i;a,b} = q^{ab} L_i^a L_{i+1}^b` is
    # automatically recovered by the universal bar-invariance formula
    # `phase = -Σ_{i<j} c(a_i, a_j)` (= `a*b · (-c(L_i, L_{i+1})) = a*b`)
    # — no `cone_label_phase` override needed.

    def cycle_period_bound(self) -> int:
        """Pentagon's `ρ` shifts `i → i+2` in ℤ/5; `ρ²` shifts by 4 in
        ℤ/5, period 5 (since `gcd(2, 5) = 1`).  So at most 5 cycles
        before any tagged letter visits every ρ²-image — i.e. before a
        Plücker collision is guaranteed."""
        return 5

    # -- canonical cone order (wrap-aware) --------------------------------

    def canonical_cone_order(self, gens: frozenset[int]) -> tuple[int, ...]:
        """Pentagon's basis convention `L_{i;a,b} = q^{ab} L_i^a L_{i+1}^b`
        fixes the order: within cone `{i, (i+1) mod 5}`, the lower-i
        element comes first — *except* for the wrap cone `{0, 4}` whose
        natural order is `(4, 0)` (since the cone is `(L_4, L_0)`)."""
        gs = sorted(gens)
        if gs == [0, 4]:
            return (4, 0)
        return tuple(gs)

    # -- cocycle / cross-product ------------------------------------------

    def cocycle(self, g: int, h: int) -> int:
        if g == h:
            return 0
        di = (h - g) % 5
        if di == 1:
            return -1          # L_g L_{g+1} = q⁻² L_{g+1} L_g
        if di == 4:
            return 1           # L_g L_{g-1} = q⁺² L_{g-1} L_g
        raise ValueError(
            f"cocycle: L_{g}, L_{h} do not q-commute (di={di})"
        )

    def cross_product(self, g: int, h: int) -> Sequence[CrossProductTerm]:
        di = (h - g) % 5
        if di == 2:
            # L_g L_{g+2} = 1 + q⁻¹ L_{g+1}
            return (
                (LaurentPoly({0: 1}), ()),
                (LaurentPoly({-1: 1}), ((g + 1) % 5,)),
            )
        if di == 3:
            # L_g L_{g-2} = 1 + q L_{g-1}
            return (
                (LaurentPoly({0: 1}), ()),
                (LaurentPoly({1: 1}), ((g - 1) % 5,)),
            )
        raise ValueError(
            f"cross_product: L_{g}, L_{h} q-commute (di={di}) — "
            f"use cocycle instead"
        )

    # -- cone-label bijection ---------------------------------------------

    def to_cone_label(
        self, native_label: PentagonLabel
    ) -> tuple[frozenset[int], dict[int, int]]:
        i, a, b = native_label
        if a == 0 and b == 0:
            return frozenset(), {}
        if b == 0:
            return frozenset({_idx5(i)}), {_idx5(i): a}
        if a == 0:
            j = _idx5(i + 1)
            return frozenset({j}), {j: b}
        i_, j_ = _idx5(i), _idx5(i + 1)
        return frozenset({i_, j_}), {i_: a, j_: b}

    def from_cone_label(
        self, gens: frozenset[int], powers: dict[int, int]
    ) -> PentagonLabel:
        if not gens:
            return (0, 0, 0)
        if len(gens) == 1:
            g = next(iter(gens))
            return (g, powers[g], 0)
        # Two mult-gens.  The cone is {i, (i+1) mod 5}; identify i.
        gs = sorted(gens)
        if gs == [0, 4]:
            # Wrap cone: (i, j) = (4, 0).  L_{4;a,b} = q^{ab} L_4^a L_0^b.
            return (4, powers[4], powers[0])
        i, j = gs
        return (i, powers[i], powers[j])


# Module-level singleton — `PentagonConeData` is stateless.
PENTAGON_CONE_DATA = PentagonConeData()
