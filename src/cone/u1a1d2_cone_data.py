"""
u1a1d2_cone_data.py
===================

`U1A1D2ConeData(FiniteConeData)` — cone-data wiring for SQED₂ = [A₁, D₂]
= U(1) gauge + two charged hypers = `U_𝖖(𝔰𝔩₂)`, over `SU2ZPlusRing`
(= R(SU(2))).

This is the **gauge-part** cone structure of SQED₂: the
multiplicative generators are the `U_𝖖(𝔰𝔩₂)` letters `E`, `F`, `K`, `K⁻¹`,
with the Cartan `K` (and its inverse `K⁻¹`) a **torus** direction (per the
`U1SquareConeData` template), and the single non-q-commuting pair `(E, F)`
carrying the `U_𝖖(𝔰𝔩₂)` cross-products

    E·F = χ₁ + 𝖖·K + 𝖖⁻¹·K⁻¹      (`FE` reversed: χ₁ + 𝖖⁻¹·K + 𝖖·K⁻¹)

— the SU(2) character `χ₁` rides the cross-product coefficient as an
`RLaurent[SU(2)]` daughter (mirroring `A1D3ConeData`), so the χ-content
lives entirely in the R-coefficient (NOT in the cone-data label), per the
cone-data convention.  The public `U1A1D2ConeKAlgebra` re-attaches the spin
index `k` at the 3-tuple boundary (A1D3-style; see `u1a1d2_cone_kalg.py`).

Native label
------------
`(m, n) ∈ ℤ²` — the **gauge part** (χ-index stripped), where
  * `m` signs the `E`(+) / `F`(−) power (the magnetic / charged-hyper
    direction),
  * `n` is the `K` (Cartan) power.

The cone-label bijection lands each `(m, n)` in exactly one of the two
Laurent cones `{E, K, K⁻¹}` / `{F, K, K⁻¹}` (or a boundary / identity
cone).  In `U_𝖖(𝔰𝔩₂)` PBW terms the canonical-basis element of `(m, n)` is

    m > 0 :  E_{m,n}  = 𝖖^{−mn} Eᵐ Kⁿ
    m < 0 :  F_{−m,n} = 𝖖^{mn}  F^{−m} Kⁿ
    m = 0 :  Kⁿ

— the SQED₂ oracle's `(gauge, k)` gauge labels with `k = 0`.

Cocycles (convention `L_g L_h = 𝖖^{2c} L_h L_g`)
------------------------------------------------
From `K E = 𝖖⁻² E K`  ⇒  `E K = 𝖖² K E`  ⇒  `c(E, K) = +1`;
from `K F = 𝖖²  F K`  ⇒  `F K = 𝖖⁻² K F`  ⇒  `c(F, K) = −1`;
`K` and `K⁻¹` commute (`c = 0`; their cancellation `K·K⁻¹ = 1` is handled
by `_word_to_gens_powers`, not `cross_product`).

q-commute graph (TRUE everywhere except the `E · F` pair).
"""
from __future__ import annotations

from typing import Sequence

from cone_data import CrossProductTerm, FiniteConeData, Cone
from laurent_poly import LaurentPoly
from zplus_ring import SU2ZPlusRing, RLaurent


__all__ = ["U1A1D2ConeData", "U1A1D2_CONE_DATA"]


# Mult-gen identifiers — 2-tuples of ints so Python's `sorted` works and the
# default `canonical_cone_order` applies.
#   (0, 0): E,    the raising / m=+1 charged hyper
#   (0, 1): F,    the lowering / m=-1 charged hyper
#   (1, 0): K,    the Cartan / gauge-monopole (positive)
#   (1, 1): K⁻¹,  the Cartan inverse (negative)
U1A1D2MultGen = tuple

E_GEN: U1A1D2MultGen = (0, 0)
F_GEN: U1A1D2MultGen = (0, 1)
K_GEN: U1A1D2MultGen = (1, 0)
KINV_GEN: U1A1D2MultGen = (1, 1)

_ALL_MULT_GENS: tuple[U1A1D2MultGen, ...] = (E_GEN, F_GEN, K_GEN, KINV_GEN)
_TORUS_GENS: frozenset[U1A1D2MultGen] = frozenset({K_GEN, KINV_GEN})

# Two Laurent cones (K and K⁻¹ q-commute → both live in each cone).
_CONES: tuple[frozenset[U1A1D2MultGen], ...] = (
    frozenset({E_GEN, K_GEN, KINV_GEN}),    # {E, K Laurent}
    frozenset({F_GEN, K_GEN, KINV_GEN}),    # {F, K Laurent}
)


class U1A1D2ConeData(FiniteConeData):
    """Cone-data for SQED₂ = [A₁, D₂] = `U_𝖖(𝔰𝔩₂)` over R = R(SU(2)).

    Stateless singleton via `U1A1D2_CONE_DATA`.  See module docstring for
    the structure (two Laurent `{E/F, K^±}` cones + the `E·F` cross-product
    carrying `χ₁` as an `RLaurent[SU(2)]` daughter)."""

    def __init__(self):
        self._R = SU2ZPlusRing()

    # ---- R-widening hook ------------------------------------------------

    def coefficient_ring(self):
        return self._R

    # ---- ConeData primitives --------------------------------------------

    def mult_gens(self) -> Sequence[U1A1D2MultGen]:
        return _ALL_MULT_GENS

    def cones(self) -> Sequence[frozenset[U1A1D2MultGen]]:
        return _CONES

    def q_commute(self, g: U1A1D2MultGen, h: U1A1D2MultGen) -> bool:
        if g == h:
            return True
        pair = frozenset({g, h})
        # Only E and F DON'T q-commute (cross_product fires).
        if pair == frozenset({E_GEN, F_GEN}):
            return False
        # K and K⁻¹ DO q-commute (cocycle 0); their cancellation K·K⁻¹ = 1
        # is handled by `_word_to_gens_powers`.  All other pairs q-commute.
        return True

    def cocycle(self, g: U1A1D2MultGen, h: U1A1D2MultGen) -> int:
        if g == h:
            return 0
        # convention L_g L_h = 𝖖^{2c} L_h L_g
        sign_table = {
            (E_GEN, K_GEN): +1, (K_GEN, E_GEN): -1,
            (E_GEN, KINV_GEN): -1, (KINV_GEN, E_GEN): +1,
            (F_GEN, K_GEN): -1, (K_GEN, F_GEN): +1,
            (F_GEN, KINV_GEN): +1, (KINV_GEN, F_GEN): -1,
            # K and K⁻¹ commute (cocycle 0).
            (K_GEN, KINV_GEN): 0, (KINV_GEN, K_GEN): 0,
        }
        c = sign_table.get((g, h))
        if c is None:
            raise ValueError(
                f"cocycle({g}, {h}): undefined (non-q-commuting pair)"
            )
        return c

    # ---- cross-product --------------------------------------------------

    def cross_product(
        self, g: U1A1D2MultGen, h: U1A1D2MultGen
    ) -> Sequence[CrossProductTerm]:
        R = self._R
        chi1 = RLaurent(R, {0: R.basis_element(1)})    # χ₁ daughter (𝖖⁰)
        # Only one non-q-commuting pair: E · F.
        if g == E_GEN and h == F_GEN:
            # E·F = χ₁ + 𝖖·K + 𝖖⁻¹·K⁻¹.
            return (
                (chi1, ()),
                (LaurentPoly({1: 1}), (K_GEN,)),
                (LaurentPoly({-1: 1}), (KINV_GEN,)),
            )
        if g == F_GEN and h == E_GEN:
            # F·E = χ₁ + 𝖖⁻¹·K + 𝖖·K⁻¹.
            return (
                (chi1, ()),
                (LaurentPoly({-1: 1}), (K_GEN,)),
                (LaurentPoly({1: 1}), (KINV_GEN,)),
            )
        raise ValueError(
            f"cross_product({g}, {h}): undefined (q-commuting pair)"
        )

    # ---- torus inverse letter (for the Laurent collapse) ----------------

    def _torus_inverse_letter(self, g: U1A1D2MultGen):
        """K ↔ K⁻¹ pairing.  Triggers the cancellation collapse in
        `_word_to_gens_powers` (`K · K⁻¹ = 1`)."""
        if g == K_GEN:
            return KINV_GEN
        if g == KINV_GEN:
            return K_GEN
        return None

    # ---- tagged-cyclicity trigger ---------------------------------------

    def tagged_cyclicity_trigger(self, word) -> bool:
        """Like `U1SquareConeData`: ρ² preserves the cone of every
        cone-monomial factor (the U_𝖖(𝔰𝔩₂) braid ρ² shears the K-direction
        but keeps E/F in their Laurent cones), so no Plücker can fire via
        tagged-cyclicity.  Skip the wasted cycles and go straight to the
        defensive emit + ρ²-canonicalisation pipeline."""
        return False

    # ---- cone-label bijection -------------------------------------------

    def to_cone_label(
        self, native_label: tuple[int, int]
    ) -> tuple[frozenset[U1A1D2MultGen], dict[U1A1D2MultGen, int]]:
        m, n = native_label
        gens: set[U1A1D2MultGen] = set()
        powers: dict[U1A1D2MultGen, int] = {}
        if m > 0:
            gens.add(E_GEN)
            powers[E_GEN] = m
        elif m < 0:
            gens.add(F_GEN)
            powers[F_GEN] = -m
        if n > 0:
            gens.add(K_GEN)
            powers[K_GEN] = n
        elif n < 0:
            gens.add(KINV_GEN)
            powers[KINV_GEN] = -n
        return frozenset(gens), powers

    def from_cone_label(
        self,
        gens: frozenset[U1A1D2MultGen],
        powers: dict[U1A1D2MultGen, int],
    ) -> tuple[int, int]:
        m = 0
        n = 0
        for g, p in powers.items():
            if p <= 0:
                continue
            if g == E_GEN:
                m += p
            elif g == F_GEN:
                m -= p
            elif g == K_GEN:
                n += p
            elif g == KINV_GEN:
                n -= p
            else:
                raise ValueError(f"from_cone_label: unknown mult-gen {g}")
        return (m, n)

    # ---- QTCone iteration -----------------------------------------------

    def iter_cones(self):
        """Yield `Cone` instances — 2 total, one per Laurent cone.  Each
        cone has `mult_gens = {E/F, K, K⁻¹}` (3 letters, with K, K⁻¹ the two
        halves of the Laurent Cartan direction) and `torus_gens = {K, K⁻¹}`."""
        for cone_gens in _CONES:
            yield Cone(self, cone_gens, torus_gens=_TORUS_GENS & cone_gens)


U1A1D2_CONE_DATA = U1A1D2ConeData()


if __name__ == "__main__":
    D = U1A1D2ConeData()
    print(f"U1A1D2ConeData: {len(D.mult_gens())} mult-gens")
    print(f"  coefficient_ring = {D.coefficient_ring()}")
    print(f"  {len(D.cones())} cones:")
    for c in sorted(D.cones(), key=lambda x: tuple(sorted(x))):
        print(f"    {sorted(c)}")
    print()
    print("Cross-product E · F:")
    for coef, word in D.cross_product(E_GEN, F_GEN):
        print(f"  {coef} · {word}")
    print("Cross-product F · E:")
    for coef, word in D.cross_product(F_GEN, E_GEN):
        print(f"  {coef} · {word}")
