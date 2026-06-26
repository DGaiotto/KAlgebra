"""Cone-filtration data for `U1SquareKAlg` = SQED N_f=1 = U(1)-gauged
[A_1, A_1] AD theory = `BPSKAlgebra(O→F)`.

  Geometric naming
  ----------------
  k=0 corner of the `U1A1AoddKAlg(k)` family on the (2k+4)-gon = 4-gon
  (= square) at k=0.  The "Square" in the class name reflects the BPS-
  quiver-square geometry.  Algebraically identical to `Sqed1KAlg` and
  to `BPSKAlgebra(O→F)` per `sqed1_bps_iso.py`.

  Cone-monomial structure: 2 Laurent cones
  ----------------------------------------
  4 INTERNAL mult-gens used for word arithmetic:

      u_p     -- (m=+1) charged hyper, u_+ direction
      u_n     -- (m=-1) charged hyper, u_- direction
      v_p     -- (n=+1) gauge monopole, v direction
      v_n     -- (n=-1) gauge monopole, v^{-1} direction

  But only **2 EXTERNAL cones** (the maximal q-commuting subsets when
  v_p, v_n are declared q-commuting with cocycle 0):

      C_+  =  {u_p, v_p, v_n}    -- conceptually "{u_+, v Laurent}"
      C_-  =  {u_n, v_p, v_n}    -- conceptually "{u_-, v Laurent}"

  Each cone is ρ²-shear-fixed (for k=0: ρ² fixes each cone setwise,
  with v-direction drift = magnetic charge m).  Each cone exposed as
  a `QTCone` with `torus_gens = {v_p, v_n}` (= the v-Laurent direction
  manifested as two paired letters internally).

  Cancellation `v_p · v_n = 1` is handled at the cone-monomial-output
  boundary by `ConeData._word_to_gens_powers` (which collapses paired
  torus letters via `_torus_inverse_letter`) — NOT via cross_product.
  v_p and v_n q-commute with cocycle 0.

  q-commute graph (TRUE everywhere except the u_+ · u_- pair):

      (u_p, u_n): FALSE  -- cross_product: u_+ · u_- = 1 + q · v
      All other pairs:    TRUE
      (v_p, v_n): TRUE with cocycle 0  (commute; cancellation
                                        handled by _word_to_gens_powers)

  Cocycles (from Sqed1's `_mono_step`; convention `L_g L_h = q^{2c} L_h L_g`):

      cocycle(u_p, v_p) = +1    (u_+ · v       = q^{+2} v       · u_+)
      cocycle(u_p, v_n) = -1    (u_+ · v^{-1}  = q^{-2} v^{-1}  · u_+)
      cocycle(u_n, v_p) = -1
      cocycle(u_n, v_n) = +1
      cocycle(v_p, v_n) = 0     (commute)

  Cross-products (only one, the u_+ · u_- one):

      cross_product(u_p, u_n) = [(1, ()), (q, (v_p,))]
                              -- Sqed1's u_+ · u_- = 1 + q v
      cross_product(u_n, u_p) = [(1, ()), (q^{-1}, (v_p,))]

  Native labels (matching Sqed1KAlg's convention)
  -----------------------------------------------
  `(m, n) ∈ Z²`, where m signs the u-direction and n signs the
  v-direction.  `L_{m, n} = q^{-mn} · u_{sign(m)}^|m| · v_{sign(n)}^|n|`
  (Sqed1's `_basis_to_mono`).  The cone-label bijection lands each
  (m, n) in exactly one of the 4 cones (or a 1-gen boundary cone, or
  the empty/identity cone for (0, 0)):

      (m > 0, n > 0)  -->  C_pp
      (m > 0, n < 0)  -->  C_pn
      (m > 0, n = 0)  -->  {u_p}
      (m < 0, n > 0)  -->  C_np
      (m < 0, n < 0)  -->  C_nn
      (m < 0, n = 0)  -->  {u_n}
      (m = 0, n > 0)  -->  {v_p}
      (m = 0, n < 0)  -->  {v_n}
      (m = 0, n = 0)  -->  identity

  Bar-invariance phase is the universal formula
  `phase = -Σ_{i<j} c(a_i, a_j)`; matches Sqed1's `_basis_to_mono`
  exactly.  E.g. for (m > 0, n > 0): word = (u_p,)*m + (v_p,)*n,
  phase = -m·n·cocycle(u_p, v_p) = -m·n.  ✓
"""

from __future__ import annotations

from typing import Sequence

from cone_data import (
    CrossProductTerm,
    FiniteConeData,
    Cone,
)
from laurent_poly import LaurentPoly


__all__ = ["U1SquareConeData", "U1SQUARE_CONE_DATA"]


# Mult-gen identifiers — 2-tuples of ints so Python's `sorted` works
# and `canonical_cone_order` is the default.
#   (1, 0): u_+, the m=+1 charged hyper
#   (1, 1): u_-, the m=-1 charged hyper
#   (2, 0): v,    the gauge monopole (positive)
#   (2, 1): v^{-1}, the gauge monopole (negative)
U1SqMultGen = tuple[int, int]

U_P: U1SqMultGen = (1, 0)
U_N: U1SqMultGen = (1, 1)
V_P: U1SqMultGen = (2, 0)
V_N: U1SqMultGen = (2, 1)

_ALL_MULT_GENS: tuple[U1SqMultGen, ...] = (U_P, U_N, V_P, V_N)
_TORUS_GENS: frozenset[U1SqMultGen] = frozenset({V_P, V_N})

# Two Laurent cones (v_p and v_n q-commute → both in same cone).
_CONES: tuple[frozenset[U1SqMultGen], ...] = (
    frozenset({U_P, V_P, V_N}),    # C_+ ≡ {u_+, v Laurent}
    frozenset({U_N, V_P, V_N}),    # C_- ≡ {u_-, v Laurent}
)


class U1SquareConeData(FiniteConeData):
    """ConeData for U1SquareKAlg.  Stateless singleton via
    `U1SQUARE_CONE_DATA`."""

    def mult_gens(self) -> Sequence[U1SqMultGen]:
        return _ALL_MULT_GENS

    def cones(self) -> Sequence[frozenset[U1SqMultGen]]:
        return _CONES

    # -- q-commute / cocycle ----------------------------------------------

    def q_commute(self, g: U1SqMultGen, h: U1SqMultGen) -> bool:
        if g == h:
            return True
        pair = frozenset({g, h})
        # Only u_+ and u_- DON'T q-commute (cross_product fires).
        if pair == frozenset({U_P, U_N}):
            return False
        # v_p and v_n DO q-commute (cocycle 0); their cancellation
        # v · v^{-1} = 1 is handled by `_word_to_gens_powers` collapse,
        # not by cross_product.
        # All other pairs q-commute too.
        return True

    def cocycle(self, g: U1SqMultGen, h: U1SqMultGen) -> int:
        if g == h:
            return 0
        sign_table = {
            (U_P, V_P): +1, (V_P, U_P): -1,
            (U_P, V_N): -1, (V_N, U_P): +1,
            (U_N, V_P): -1, (V_P, U_N): +1,
            (U_N, V_N): +1, (V_N, U_N): -1,
            # v_p and v_n commute (cocycle 0).
            (V_P, V_N): 0, (V_N, V_P): 0,
        }
        c = sign_table.get((g, h))
        if c is None:
            raise ValueError(
                f"cocycle({g}, {h}): undefined (non-q-commuting pair)"
            )
        return c

    # -- cross-product ----------------------------------------------------

    def cross_product(
        self, g: U1SqMultGen, h: U1SqMultGen
    ) -> Sequence[CrossProductTerm]:
        # Only one non-q-commuting pair: u_+ · u_-.
        # u_+ · u_- = 1 + q · v_p (Sqed1's mid_one + mid_v[q^+1]).
        if g == U_P and h == U_N:
            return (
                (LaurentPoly({0: 1}), ()),
                (LaurentPoly({1: 1}), (V_P,)),
            )
        # u_- · u_+ = 1 + q^{-1} · v_p.
        if g == U_N and h == U_P:
            return (
                (LaurentPoly({0: 1}), ()),
                (LaurentPoly({-1: 1}), (V_P,)),
            )
        raise ValueError(
            f"cross_product({g}, {h}): undefined (q-commuting pair)"
        )

    # -- torus inverse letter (for the Laurent collapse) ------------------

    def _torus_inverse_letter(self, g: U1SqMultGen):
        """v_p ↔ v_n pairing.  Triggers the cancellation collapse in
        `_word_to_gens_powers`."""
        if g == V_P:
            return V_N
        if g == V_N:
            return V_P
        return None

    # -- tagged-cyclicity trigger -----------------------------------------

    def tagged_cyclicity_trigger(self, word) -> bool:
        """U1Square's ρ² preserves the cone of every cone-monomial
        factor: u_+, u_-, v_p, v_n all stay in their respective
        Laurent cones {u_+, v_p, v_n} / {u_-, v_p, v_n} under every
        power of ρ².  No Plücker can ever fire via tagged-cyclicity.
        Returning False here skips the wasted cycles in
        `simplify_trace_via_cone_data` and goes straight to the
        defensive emit + ρ²-canonicalisation pipeline.
        """
        return False

    # -- cone-label bijection ---------------------------------------------

    def to_cone_label(
        self, native_label: tuple[int, int]
    ) -> tuple[frozenset[U1SqMultGen], dict[U1SqMultGen, int]]:
        m, n = native_label
        if m == 0 and n == 0:
            return frozenset(), {}
        if m == 0:
            if n > 0:
                return frozenset({V_P}), {V_P: n}
            return frozenset({V_N}), {V_N: -n}
        if n == 0:
            if m > 0:
                return frozenset({U_P}), {U_P: m}
            return frozenset({U_N}), {U_N: -m}
        # m != 0 and n != 0: pick the appropriate cone.
        if m > 0 and n > 0:
            return frozenset({U_P, V_P}), {U_P: m, V_P: n}
        if m > 0 and n < 0:
            return frozenset({U_P, V_N}), {U_P: m, V_N: -n}
        if m < 0 and n > 0:
            return frozenset({U_N, V_P}), {U_N: -m, V_P: n}
        # m < 0 and n < 0
        return frozenset({U_N, V_N}), {U_N: -m, V_N: -n}

    def from_cone_label(
        self,
        gens: frozenset[U1SqMultGen],
        powers: dict[U1SqMultGen, int],
    ) -> tuple[int, int]:
        # Inverse of to_cone_label.
        m = 0
        n = 0
        for g, p in powers.items():
            if p <= 0:
                continue
            if g == U_P:
                m += p
            elif g == U_N:
                m -= p
            elif g == V_P:
                n += p
            elif g == V_N:
                n -= p
            else:
                raise ValueError(
                    f"from_cone_label: unknown mult-gen {g}"
                )
        return (m, n)

    # -- QTCone iter ------------------------------------------------------

    def iter_cones(self):
        """Yield `QTCone` instances — 2 total, one per Laurent cone.
        Each cone has `mult_gens = {u_±, v_p, v_n}` (3 letters, with
        v_p, v_n the two halves of the Laurent v direction) and
        `torus_gens = {v_p, v_n}`."""
        for cone_gens in _CONES:
            yield Cone(self, cone_gens, torus_gens=_TORUS_GENS & cone_gens)


U1SQUARE_CONE_DATA = U1SquareConeData()
