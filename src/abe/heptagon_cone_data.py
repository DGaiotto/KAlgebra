"""Cone-filtration data for the heptagon K-algebra `A_𝖖([A_1, A_4])`.

The heptagon's structure (mirroring pentagon, but larger and with two
orbit-types of mult-gens):

  * **Multiplicative generators**: `L((k, i))` for `k ∈ {1, 2}`,
    `i ∈ ℤ/7` — 14 total.  Orbit `k=1` is the short-diagonal chord
    `(i, i+2)` of the heptagon; orbit `k=2` is the long-diagonal
    chord `(i+3, i+6)`.
  * **Cones** (maximal q-commuting subsets of mult-gens): the non-
    crossing chord arrangements of the heptagon, computed at instance
    init via a Bron-Kerbosch maximal-clique enumeration on the
    q-commute graph.  Each `L((1, i))` q-commutes with 9 partners,
    each `L((2, i))` q-commutes with 7.
  * **q-commute cocycle**: `c(g, h) = _hept_qcommute_factor(g, h) / 2`.
    Heptagon's helper returns the *full* exponent `c'` such that
    `L_g L_h = q^{c'} L_h L_g`; cone_data's convention is the half
    exponent `c` with `L_g L_h = q^{2c} L_h L_g`.  `_hept_qcommute_factor`
    only takes even values `{-2, 0, 2}`, so the halving is exact.
  * **Cross-products** (Plücker pairs): `_hept_pair_product(g, h)`
    returns a list of `(kind, c)` summands where `kind` is one of
    `('I',)` (identity), `('letter', (k, i))` (single letter), or
    `('pair', ((k1, i1), (k2, i2)))` (two-letter q-commuting product).
    Lifted into cone_data's `(coeff, word)` format.
  * **Native canonical-basis labels**: sorted tuples
    `((k₁, i₁, e₁), …, (k_m, i_m, e_m))` with pairwise q-commuting
    `(k_j, i_j)` letters and strictly-positive exponents.  Identity
    is the empty tuple `()`.
  * **Bijection to cone-label form**: `(label) ↔ ({(k_j, i_j)}, {(k_j, i_j): e_j})`
    via direct unpacking.

The convention phase relating `L_canonical[label]` to the literal
mult-gen product is `−T_bps(label)`, and `T_bps = Σ_{i<j} _hept_forward_q_coeff(...) · e_i · e_j`.
This is recovered automatically by the universal bar-invariance
formula `phase = −Σ_{i<j} c(a_i, a_j)` (with `c` = the cone_data
cocycle = half the qcommute exponent), so no per-subclass override is
needed — `cone_label_phase` works out of the box.
"""

from __future__ import annotations

from typing import Sequence

from cone_data import CrossProductTerm, FiniteConeData
from kalgebra_samples import (
    _HEPT_H,
    _hept_pair_product,
    _hept_qcommute_factor,
)
from laurent_poly import LaurentPoly


__all__ = ["HeptagonConeData", "HEPTAGON_CONE_DATA"]


HeptMultGen = tuple[int, int]                          # (k, i)
HeptNativeLabel = tuple[tuple[int, int, int], ...]     # ((k_1, i_1, e_1), ...)


def _maximal_cliques(
    vertices: Sequence[HeptMultGen],
    qcom: callable,
) -> tuple[frozenset[HeptMultGen], ...]:
    """Enumerate maximal cliques of the q-commute graph via Bron-Kerbosch."""
    V = list(vertices)
    neighbours = {v: frozenset(u for u in V if u != v and qcom(v, u))
                  for v in V}
    cliques: list[frozenset[HeptMultGen]] = []

    def bk(R: frozenset, P: frozenset, X: frozenset):
        if not P and not X:
            cliques.append(R)
            return
        # Pivot selection (Tomita): pick u in P ∪ X maximising |P ∩ N(u)|.
        pivot = max(P | X, key=lambda u: len(P & neighbours[u]))
        for v in list(P - neighbours[pivot]):
            bk(R | {v}, P & neighbours[v], X & neighbours[v])
            P = P - {v}
            X = X | {v}

    bk(frozenset(), frozenset(V), frozenset())
    return tuple(cliques)


class HeptagonConeData(FiniteConeData):
    """`ConeData` for `HeptagonKAlg` — the K-algebra `A_𝖖([A_1, A_4])`."""

    def __init__(self) -> None:
        self._mult_gens: tuple[HeptMultGen, ...] = tuple(
            (k, i) for k in (1, 2) for i in range(_HEPT_H)
        )
        # Maximal q-commuting cliques.  Each is a frozenset of (k, i)
        # mult-gens with all pairs q-commuting (= non-crossing chord set).
        self._cones: tuple[frozenset[HeptMultGen], ...] = _maximal_cliques(
            self._mult_gens,
            lambda u, v: _hept_qcommute_factor(u, v) is not None,
        )

    # -- finite enumeration -----------------------------------------------

    def mult_gens(self) -> Sequence[HeptMultGen]:
        return self._mult_gens

    def cones(self) -> Sequence[frozenset[HeptMultGen]]:
        return self._cones

    # -- q_commute / cocycle / cross_product ------------------------------

    def q_commute(self, g: HeptMultGen, h: HeptMultGen) -> bool:
        if g == h:
            return True
        return _hept_qcommute_factor(g, h) is not None

    def cocycle(self, g: HeptMultGen, h: HeptMultGen) -> int:
        """`c` such that `L_g L_h = q^{2c} L_h L_g`.

        Heptagon's `_hept_qcommute_factor` returns the *full* exponent
        `c'` with `L_g L_h = q^{c'} L_h L_g`.  cone_data's convention is
        the half exponent.  Empirically `c' ∈ {-2, 0, 2}` so the
        halving is exact."""
        if g == h:
            return 0
        c_full = _hept_qcommute_factor(g, h)
        if c_full is None:
            raise ValueError(
                f"cocycle: L({g}), L({h}) are not q-commuting"
            )
        if c_full % 2 != 0:
            raise AssertionError(
                f"cocycle: _hept_qcommute_factor({g}, {h}) = {c_full} is odd; "
                f"expected even (heptagon convention)"
            )
        return c_full // 2

    def cross_product(
        self, g: HeptMultGen, h: HeptMultGen,
    ) -> Sequence[CrossProductTerm]:
        """`L_g · L_h` as a sum of `(coeff, word)` pairs, in the
        literal-mult-gen-product convention that cone_data expects.

        Lifted from `_hept_pair_product`, which returns `((kind, c_can), …)`
        summands where `kind` describes the canonical-basis label of
        the summand and `c_can` is its q-exponent in the
        *canonical-basis* form
        (`L(g) · L(h) = … + q^{c_can} · L_canonical[kind]`).  cone_data's
        cross_product wants the LITERAL coefficient
        (`L(g) · L(h) = … + q^{c_lit} · literal-word(kind)`); since
        `L_canonical[kind] = q^{cone_label_phase(kind)} · literal-word(kind)`,
        we have `c_lit = c_can + cone_label_phase(kind)`.
        """
        if _hept_qcommute_factor(g, h) is not None:
            raise ValueError(
                f"cross_product: L({g}), L({h}) are q-commuting; "
                f"use cocycle instead"
            )
        terms: list[CrossProductTerm] = []
        for (kind, c_can) in _hept_pair_product(g, h):
            if kind == ('I',):
                gens: frozenset[HeptMultGen] = frozenset()
                powers: dict[HeptMultGen, int] = {}
                word: tuple[HeptMultGen, ...] = ()
            elif kind[0] == 'letter':
                mg = kind[1]
                gens = frozenset({mg})
                powers = {mg: 1}
                word = (mg,)
            elif kind[0] == 'pair':
                mg1, mg2 = kind[1]
                if mg1 == mg2:
                    gens = frozenset({mg1})
                    powers = {mg1: 2}
                    word = (mg1, mg1)
                else:
                    gens = frozenset({mg1, mg2})
                    powers = {mg1: 1, mg2: 1}
                    # Canonical-order the pair so cone_data's downstream
                    # `_sort_within_cone` doesn't introduce a spurious
                    # swap phase.  The canonical-basis label associated
                    # with the pair is order-independent (identified by
                    # the multiset of letters), so the c_can coefficient
                    # applies equally to either ordering.
                    word = tuple(sorted([mg1, mg2]))
            else:
                raise AssertionError(
                    f"cross_product: unexpected kind {kind!r} in "
                    f"_hept_pair_product({g}, {h})"
                )
            phase = self.cone_label_phase(gens, powers)
            terms.append((LaurentPoly({c_can + phase: 1}), word))
        return tuple(terms)

    # -- canonical cone order ---------------------------------------------

    # Default `tuple(sorted(gens))` matches heptagon's native label
    # convention (sorted tuples of (k, i, e) triples), so no override
    # needed.  The convention phase `cone_label_phase` is derived
    # universally from bar-invariance: it equals `−T_bps(label)`, the
    # BPS X-vs-L-product offset used by heptagon's hand-rolled
    # multiply.

    # -- cone-label bijection ---------------------------------------------

    def to_cone_label(
        self, native_label: HeptNativeLabel,
    ) -> tuple[frozenset[HeptMultGen], dict[HeptMultGen, int]]:
        if not native_label:
            return frozenset(), {}
        gens: set[HeptMultGen] = set()
        powers: dict[HeptMultGen, int] = {}
        for (k, i, e) in native_label:
            if e <= 0:
                continue
            g = (k, i)
            gens.add(g)
            powers[g] = powers.get(g, 0) + e
        return frozenset(gens), powers

    def from_cone_label(
        self,
        gens: frozenset[HeptMultGen],
        powers: dict[HeptMultGen, int],
    ) -> HeptNativeLabel:
        if not gens:
            return ()
        ordered = sorted(gens)
        return tuple((k, i, powers[(k, i)]) for (k, i) in ordered)

    # -- cycle period for the tagged-cyclicity engine ---------------------

    def cycle_period_bound(self) -> int:
        """Heptagon's `ρ` shifts `i → i+1` in ℤ/7; `ρ²` shifts by 2,
        period `7 / gcd(2, 7) = 7`.  So at most 7 cycles before the
        tagged letter visits every ρ²-image."""
        return _HEPT_H


# Module-level singleton — `HeptagonConeData` is stateful (caches the
# maximal-clique enumeration) but the state is fully determined by the
# heptagon parameters, so a singleton is correct.
HEPTAGON_CONE_DATA = HeptagonConeData()
