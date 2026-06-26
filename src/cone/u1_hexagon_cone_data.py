"""Cone-filtration data for `U1HexagonKAlg`, the K-algebra of the
u(1)-gauged hexagon = `A_𝖖([A_1, A_3])` with the u(1) flavour gauged.

Differs structurally from `pentagon/heptagon/A1A2k` in two ways:

  * **Two mult-gen orbits with different periods.** `L((1, i))` has
    period 6 in `i ∈ ℤ/6` (6 generators); `L((2, i))` has period 3 in
    `i ∈ ℤ/3` (3 generators) — total 9 L-generators.

  * **`E` is an additional dynamical mult-gen** (not a flavour
    fugacity).  Native labels carry a separate integer `e_E ∈ ℤ`
    (signed!) tracking the `E` power.  To fit cone_data's
    strictly-positive-powers contract, we treat **`E` and `E⁻¹` as
    two distinct mult-gens** (your suggestion); cross_product
    daughters from `MULT_TABLE_LL` that produce `e_E > 0` go through
    `E`, those with `e_E < 0` go through `E⁻¹`, and the
    `E · E⁻¹ = 1` collapse is encoded as a Plücker pair.

So 11 mult-gens total:

  * 9 L-letters `(a, i)`: `(1, 0)..(1, 5)`, `(2, 0)..(2, 2)`.
  * `E`  encoded as `(3, 0)`.
  * `E⁻¹` encoded as `(3, 1)`.

(Using 2-tuples of ints for every mult-gen so Python's default
`sorted()` works and `canonical_cone_order` is the default.)

**Convention phase.** U1Hex's authoritative bar-invariance phase
`cone_algebra.cone_T_half = T_LL/2 - e_E · mu_qp` is reproduced
exactly by cone_data's universal bar-invariance formula
`phase = -Σ_{i<j} c(a_i, a_j)` summed over the canonical-order
word that includes L-letters AND `E` (or `E⁻¹`) — the L-L pair
contribution gives `T_LL/2`, and the L-E pair contribution
(via `c(L_l, E) = MU_LETTER_QPOWER[l]`) gives `-e_E · mu_qp`.
No `cone_label_phase` override needed.
"""

from __future__ import annotations

from typing import Sequence

from cone_data import CrossProductTerm, FiniteConeData, Cone
from cone_algebra import (
    MULT_TABLE_LL,
    MU_LETTER_QPOWER,
)
from laurent_poly import LaurentPoly


__all__ = ["U1HexagonConeData", "U1HEXAGON_CONE_DATA"]


# Mult-gen identifiers as 2-tuples of ints.
# L-letter (a, i): (a, i) directly, with a ∈ {1, 2}.
# E:     (3, 0)
# E⁻¹:   (3, 1)
U1HexMultGen = tuple[int, int]
E_GEN: U1HexMultGen = (3, 0)
E_INV: U1HexMultGen = (3, 1)


def _is_L_letter(g: U1HexMultGen) -> bool:
    return g[0] in (1, 2)


def _l_letters() -> list[U1HexMultGen]:
    return [(1, i) for i in range(6)] + [(2, i) for i in range(3)]


def _all_mult_gens() -> list[U1HexMultGen]:
    return _l_letters() + [E_GEN, E_INV]


def _word_from_factors_and_e_E(
    factors: tuple[tuple[int, int, int], ...],
    e_E: int,
) -> tuple[U1HexMultGen, ...]:
    """Build a canonical-order literal word from `(factors, e_E)`."""
    word: list[U1HexMultGen] = []
    for (a, i, exp) in factors:
        word.extend([(a, i)] * exp)
    if e_E > 0:
        word.extend([E_GEN] * e_E)
    elif e_E < 0:
        word.extend([E_INV] * (-e_E))
    return tuple(word)


class U1HexagonConeData(FiniteConeData):
    """`ConeData` for `U1HexagonKAlg`.

    Stateless: the 11 mult-gens, q-commute graph, and cocycles are
    fully determined by the `cone_algebra` tables (`MULT_TABLE_LL`,
    `MU_LETTER_QPOWER`).  Cones are enumerated lazily.
    """

    def __init__(self) -> None:
        self._mult_gens: tuple[U1HexMultGen, ...] = tuple(_all_mult_gens())
        # (a, i) chord generators — read by the orthonormality-bootstrap
        # driver (`u1aodd_trace_bootstrap.solve_intermediate`).
        self._chords: list[U1HexMultGen] = _l_letters()
        self._cones: tuple[frozenset[U1HexMultGen], ...] | None = None

    # -- finite enumeration -----------------------------------------------

    def mult_gens(self) -> Sequence[U1HexMultGen]:
        return self._mult_gens

    def cones(self) -> Sequence[frozenset[U1HexMultGen]]:
        if self._cones is None:
            # Bron-Kerbosch maximal cliques on the q-commute graph.
            V = list(self._mult_gens)
            neighbours = {
                v: frozenset(u for u in V if u != v and self.q_commute(v, u))
                for v in V
            }
            cliques: list[frozenset[U1HexMultGen]] = []

            def bk(R, P, X):
                if not P and not X:
                    cliques.append(R)
                    return
                pivot = max(P | X, key=lambda u: len(P & neighbours[u]))
                for v in list(P - neighbours[pivot]):
                    bk(R | {v}, P & neighbours[v], X & neighbours[v])
                    P = P - {v}
                    X = X | {v}

            bk(frozenset(), frozenset(V), frozenset())
            self._cones = tuple(cliques)
        return self._cones

    # -- q_commute / cocycle / cross_product ------------------------------

    def q_commute(self, g: U1HexMultGen, h: U1HexMultGen) -> bool:
        if g == h:
            return True
        # E and E⁻¹ now q-commute with cocycle 0 (Laurent-cone /
        # QTCone convention): they live in the SAME cone as a torus
        # direction.  The cancellation E · E⁻¹ = 1 is handled by the
        # universal `_torus_inverse_letter` collapse in
        # `ConeData._word_to_gens_powers`, NOT via cross_product.
        if {g, h} == {E_GEN, E_INV}:
            return True
        if _is_L_letter(g) and _is_L_letter(h):
            # L-L: q-commute iff `MULT_TABLE_LL[(g, h)]` is single-term.
            fwd = MULT_TABLE_LL.get((g, h))
            if fwd is None or len(fwd) != 1:
                return False
            return True
        # L-E and L-E⁻¹: always q-commute via MU_LETTER_QPOWER.
        return True

    def cocycle(self, g: U1HexMultGen, h: U1HexMultGen) -> int:
        """`c` such that `L_g L_h = q^{2c} L_h L_g`."""
        if g == h:
            return 0
        # E and E⁻¹ commute (cocycle 0) — torus direction.
        if {g, h} == {E_GEN, E_INV}:
            return 0
        if _is_L_letter(g) and _is_L_letter(h):
            fwd = MULT_TABLE_LL.get((g, h))
            bwd = MULT_TABLE_LL.get((h, g))
            if fwd is None or len(fwd) != 1 or bwd is None or len(bwd) != 1:
                raise ValueError(f"cocycle: L{g}, L{h} are not q-commuting")
            c_full = fwd[0][0] - bwd[0][0]
            if c_full % 2 != 0:
                raise AssertionError(
                    f"cocycle: (fwd-bwd)/2 = {c_full}/2 not integer at "
                    f"({g}, {h})"
                )
            return c_full // 2
        # L-letter ↔ E:  L_l · E = q^{2·MU_LETTER_QPOWER[l]} · E · L_l
        # (derived from `E · L_l = q^{-2·MU_LETTER_QPOWER[l]} · L_l · E`).
        # So c(L_l, E) = MU_LETTER_QPOWER[l], c(E, L_l) = -c(L_l, E).
        # L-letter ↔ E⁻¹: opposite sign.
        if g == E_GEN and _is_L_letter(h):
            return -MU_LETTER_QPOWER[h]
        if h == E_GEN and _is_L_letter(g):
            return MU_LETTER_QPOWER[g]
        if g == E_INV and _is_L_letter(h):
            return MU_LETTER_QPOWER[h]
        if h == E_INV and _is_L_letter(g):
            return -MU_LETTER_QPOWER[g]
        raise ValueError(
            f"cocycle: L({g}), L({h}) are not q-commuting (E↔E⁻¹ pair)"
        )

    def cross_product(
        self, g: U1HexMultGen, h: U1HexMultGen,
    ) -> Sequence[CrossProductTerm]:
        """`L_g · L_h` as a sum of `(coeff, word)` pairs in the
        literal-mult-gen-product convention."""
        if self.q_commute(g, h):
            raise ValueError(
                f"cross_product: L({g}), L({h}) are q-commuting"
            )
        # Note: (E, E⁻¹) is now q-commuting (cocycle 0); the cancellation
        # E · E⁻¹ = 1 is handled by `_torus_inverse_letter` collapse in
        # `ConeData._word_to_gens_powers`, not via cross_product.
        # L-L Plücker from MULT_TABLE_LL.
        if _is_L_letter(g) and _is_L_letter(h):
            entries = MULT_TABLE_LL.get((g, h))
            if entries is None:
                raise ValueError(
                    f"cross_product: no MULT_TABLE_LL entry for ({g}, {h})"
                )
            terms: list[CrossProductTerm] = []
            for (fq_pow, sign, factors, e_E) in entries:
                assert sign == 1, (
                    f"cross_product: unexpected sign {sign} at "
                    f"({g}, {h}) entry {(fq_pow, sign, factors, e_E)}"
                )
                # Build the canonical-order literal word.
                word = _word_from_factors_and_e_E(factors, e_E)
                # Bar-invariance correction: c_lit = c_can + cone_label_phase.
                # NOTE: U1Hex's canonical-basis convention only twists by
                # L-L pairs (see `cone_label_phase` override below), so
                # `cone_label_phase` for the daughter (factors, e_E) gives
                # exactly the right twist for L-L; L-E pair phases are
                # NOT included.
                gens, powers = self._gens_powers_from_word(word)
                phase = self.cone_label_phase(gens, powers)
                terms.append((LaurentPoly({fq_pow + phase: 1}), word))
            return tuple(terms)
        raise AssertionError(
            f"cross_product: unhandled non-q-commuting pair ({g}, {h})"
        )

    # `cone_label_phase` uses the universal bar-invariance formula
    # from `cone_data.ConeData` — no override needed.  It reproduces
    # `cone_algebra.cone_T_half`: the L-L pair contribution is the
    # `T_LL/2` term, and the L-E (and L-E⁻¹) pair contributions give
    # the `-e_E · mu_qp` term in `cone_T_half`.  (U1Hex's own
    # `as_generator_monomial` computes a different quantity that omits
    # the L-E terms; that's a separate utility, not on the
    # multiplication path.)
    #
    # The E ↔ E⁻¹ pair has cocycle 0, so it contributes 0 to the
    # universal phase formula — no impact.  The cancellation E · E⁻¹
    # = 1 happens at the cone-monomial output boundary via
    # `_torus_inverse_letter` (see below).

    # -- torus inverse letter (Laurent E direction) -----------------------

    def _torus_inverse_letter(self, g: U1HexMultGen):
        """E ↔ E⁻¹ pairing.  Triggers the universal cancellation
        collapse in `ConeData._word_to_gens_powers`.  This replaces
        the previous cross_product(E, E_inv) = identity Plücker."""
        if g == E_GEN:
            return E_INV
        if g == E_INV:
            return E_GEN
        return None

    # -- QTCone iter ------------------------------------------------------

    def iter_cones(self):
        """Yield QTCone instances (one per maximal q-commuting clique).
        Each cone has `torus_gens = {E, E⁻¹} ∩ cone_gens` (= the
        Laurent direction in this cone, when present)."""
        for cone_gens in self.cones():
            torus_in_cone = frozenset({E_GEN, E_INV}) & cone_gens
            yield Cone(self, cone_gens, torus_gens=torus_in_cone)

    # -- cone-label bijection ---------------------------------------------

    def to_cone_label(
        self,
        native_label: tuple[tuple[tuple[int, int, int], ...], int],
    ) -> tuple[frozenset[U1HexMultGen], dict[U1HexMultGen, int]]:
        factors, e_E = native_label
        gens: set[U1HexMultGen] = set()
        powers: dict[U1HexMultGen, int] = {}
        for (a, i, exp) in factors:
            if exp <= 0:
                continue
            g = (a, i)
            gens.add(g)
            powers[g] = powers.get(g, 0) + exp
        if e_E > 0:
            gens.add(E_GEN)
            powers[E_GEN] = e_E
        elif e_E < 0:
            gens.add(E_INV)
            powers[E_INV] = -e_E
        return frozenset(gens), powers

    def from_cone_label(
        self,
        gens: frozenset[U1HexMultGen],
        powers: dict[U1HexMultGen, int],
    ) -> tuple[tuple[tuple[int, int, int], ...], int]:
        l_factors = sorted(
            ((g, powers[g]) for g in gens if _is_L_letter(g)),
            key=lambda x: x[0],
        )
        factors = tuple((g[0], g[1], e) for (g, e) in l_factors)
        e_E = powers.get(E_GEN, 0) - powers.get(E_INV, 0)
        return factors, e_E

    # -- cycle period for the tagged-cyclicity engine ---------------------

    def cycle_period_bound(self) -> int:
        """U1Hex's `ρ` shifts i → i+1 in ℤ/6 and inverts the E-power.
        `ρ²` shifts i → i+2 (no E inversion).  On L((1, *)) the
        ρ²-orbit has period 3 in ℤ/6 (since gcd(2, 6) = 2);
        on L((2, *)) the orbit has period 3 in ℤ/3 (also gcd(2, 3) = 1).
        Bound: 6 covers both."""
        return 6

    # -- helpers ----------------------------------------------------------

    def _gens_powers_from_word(
        self, word: tuple[U1HexMultGen, ...],
    ) -> tuple[frozenset[U1HexMultGen], dict[U1HexMultGen, int]]:
        """Convert a literal word to (gens, powers)."""
        powers: dict[U1HexMultGen, int] = {}
        for g in word:
            powers[g] = powers.get(g, 0) + 1
        return frozenset(powers.keys()), powers


# Module-level singleton.
U1HEXAGON_CONE_DATA = U1HexagonConeData()
