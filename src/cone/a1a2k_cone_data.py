"""Cone-filtration data for `A1A2kKAlg`, the K-algebra
`A_𝖖([A_1, A_{2k}])` parameterised by `k ≥ 1`.

Structurally identical to `HeptagonConeData` (which is the `k = 2`
case) but parametric: each `A1A2kKAlg` instance carries its own
`A1A2kConeData` since the `_base_table`, q-commute, and forward-q
caches depend on `k`.

  * **Multiplicative generators**: `L((k_letter, i))` for
    `k_letter ∈ {1, …, k}`, `i ∈ ℤ/H` with `H = 2k + 3` — total
    `k · H = k(2k+3)` generators.
  * **Cones**: maximal q-commuting cliques in the q-commute graph,
    computed by Bron-Kerbosch at instance init.
  * **q-commute / cocycle**: delegated to the algebra's `_qc_cache`;
    cocycle = `_qc_cache` value / 2 (heptagon-style: all factors are
    even, so the halving is exact).
  * **Cross-product**: lifted from the algebra's `_pair_product` with
    the literal-vs-canonical-basis phase correction
    `c_lit = c_can + cone_label_phase(kind)`.
  * **Bijection**: `((k₁, i₁, e₁), …)` ↔ `({(k_j, i_j)}, {(k_j, i_j): e_j})`.

The convention phase `L_canonical[label] = q^{-T_bps(label)} · L-product`
where `T_bps(label) = Σ_{i<j} _forward_q_coeff(l_i, l_j) · e_i · e_j` is
recovered automatically by the universal bar-invariance formula in
`cone_data.ConeData.cone_label_phase`.
"""

from __future__ import annotations

from typing import Sequence, TYPE_CHECKING

from cone_data import CrossProductTerm, FiniteConeData
from laurent_poly import LaurentPoly

if TYPE_CHECKING:
    from a1a2k_kalg import A1A2kKAlg


__all__ = ["A1A2kConeData"]


A1A2kMultGen = tuple[int, int]                          # (k_letter, i)
A1A2kNativeLabel = tuple[tuple[int, int, int], ...]     # ((k, i, e), ...)


def _maximal_cliques(
    vertices: Sequence[A1A2kMultGen],
    qcom: callable,
) -> tuple[frozenset[A1A2kMultGen], ...]:
    """Enumerate maximal cliques of the q-commute graph (Bron-Kerbosch
    with Tomita pivot)."""
    V = list(vertices)
    neighbours = {
        v: frozenset(u for u in V if u != v and qcom(v, u))
        for v in V
    }
    cliques: list[frozenset[A1A2kMultGen]] = []

    def bk(R: frozenset, P: frozenset, X: frozenset):
        if not P and not X:
            cliques.append(R)
            return
        pivot = max(P | X, key=lambda u: len(P & neighbours[u]))
        for v in list(P - neighbours[pivot]):
            bk(R | {v}, P & neighbours[v], X & neighbours[v])
            P = P - {v}
            X = X | {v}

    bk(frozenset(), frozenset(V), frozenset())
    return tuple(cliques)


class A1A2kConeData(FiniteConeData):
    """`ConeData` for `A1A2kKAlg(k)`.  Per-instance (depends on `k`)."""

    def __init__(self, alg: "A1A2kKAlg") -> None:
        self._alg = alg
        self.k = alg.k
        self.H = alg.H
        self._mult_gens: tuple[A1A2kMultGen, ...] = tuple(
            (k_l, i) for k_l in range(1, self.k + 1) for i in range(self.H)
        )
        # Cones lazily computed on first `cones()` call.  Maximal-clique
        # enumeration on the q-commute graph is Catalan-exponential in
        # `k` and we don't need it on the live multiply / trace path
        # (`q_commute` is overridden to delegate to the algebra's
        # `_qcommute_factor` cache directly), so deferring is a big win
        # for high-k instances.
        self._cones: tuple[frozenset[A1A2kMultGen], ...] | None = None

    # -- finite enumeration -----------------------------------------------

    def mult_gens(self) -> Sequence[A1A2kMultGen]:
        return self._mult_gens

    def cones(self) -> Sequence[frozenset[A1A2kMultGen]]:
        if self._cones is None:
            self._cones = _maximal_cliques(
                self._mult_gens,
                lambda u, v: self._alg._qcommute_factor(u, v) is not None,
            )
        return self._cones

    # -- q_commute / cocycle / cross_product ------------------------------

    def q_commute(self, g: A1A2kMultGen, h: A1A2kMultGen) -> bool:
        if g == h:
            return True
        return self._alg._qcommute_factor(g, h) is not None

    def cocycle(self, g: A1A2kMultGen, h: A1A2kMultGen) -> int:
        """`c` such that `L_g L_h = q^{2c} L_h L_g`.

        A1A2k's `_qcommute_factor` returns the *full* exponent `c'`
        with `L_g L_h = q^{c'} L_h L_g`.  cone_data's convention is the
        half exponent; `c'` is always even by construction (same as
        heptagon)."""
        if g == h:
            return 0
        c_full = self._alg._qcommute_factor(g, h)
        if c_full is None:
            raise ValueError(
                f"cocycle: L({g}), L({h}) are not q-commuting"
            )
        if c_full % 2 != 0:
            raise AssertionError(
                f"cocycle: _qcommute_factor({g}, {h}) = {c_full} is odd; "
                f"expected even (A1A2k convention)"
            )
        return c_full // 2

    def cross_product(
        self, g: A1A2kMultGen, h: A1A2kMultGen,
    ) -> Sequence[CrossProductTerm]:
        """`L_g · L_h` as a sum of `(coeff, word)` pairs, in the
        literal-mult-gen-product convention that cone_data expects.

        Lifted from `_pair_product`, which returns `((kind, c_can), …)`
        summands where `kind` describes the canonical-basis label and
        `c_can` is its q-exponent in the canonical-basis form
        (`L(g) · L(h) = … + q^{c_can} · L_canonical[kind]`).
        cone_data's `cross_product` wants the LITERAL coefficient
        (`L(g) · L(h) = … + q^{c_lit} · literal-word(kind)`); since
        `L_canonical[kind] = q^{cone_label_phase(kind)} · literal-word(kind)`,
        we have `c_lit = c_can + cone_label_phase(kind)`.
        """
        if self._alg._qcommute_factor(g, h) is not None:
            raise ValueError(
                f"cross_product: L({g}), L({h}) are q-commuting; "
                f"use cocycle instead"
            )
        terms: list[CrossProductTerm] = []
        for (kind, c_can) in self._alg._pair_product(g, h):
            if kind == ('I',):
                gens: frozenset[A1A2kMultGen] = frozenset()
                powers: dict[A1A2kMultGen, int] = {}
                word: tuple[A1A2kMultGen, ...] = ()
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
                    # swap phase.  The canonical-basis label is order-
                    # independent (it's identified by the multiset of
                    # letters), so the c_can coefficient applies equally
                    # to either ordering.
                    word = tuple(sorted([mg1, mg2]))
            else:
                raise AssertionError(
                    f"cross_product: unexpected kind {kind!r} in "
                    f"_pair_product({g}, {h})"
                )
            phase = self.cone_label_phase(gens, powers)
            terms.append((LaurentPoly({c_can + phase: 1}), word))
        return tuple(terms)

    # -- cone-label bijection ---------------------------------------------

    def to_cone_label(
        self, native_label: A1A2kNativeLabel,
    ) -> tuple[frozenset[A1A2kMultGen], dict[A1A2kMultGen, int]]:
        if not native_label:
            return frozenset(), {}
        gens: set[A1A2kMultGen] = set()
        powers: dict[A1A2kMultGen, int] = {}
        for (k, i, e) in native_label:
            if e <= 0:
                continue
            g = (k, i)
            gens.add(g)
            powers[g] = powers.get(g, 0) + e
        return frozenset(gens), powers

    def from_cone_label(
        self,
        gens: frozenset[A1A2kMultGen],
        powers: dict[A1A2kMultGen, int],
    ) -> A1A2kNativeLabel:
        if not gens:
            return ()
        ordered = sorted(gens)
        return tuple((k, i, powers[(k, i)]) for (k, i) in ordered)

    # -- cycle period for the tagged-cyclicity engine ---------------------

    def cycle_period_bound(self) -> int:
        """A1A2k's `ρ` shifts `i → i+1` in ℤ/H with `H = 2k+3`; `ρ²`
        shifts by 2, period `H / gcd(2, H) = H` (since `H` is odd, so
        `gcd(2, H) = 1`).  At most `H` cycles guarantee a Plücker
        collision is exposed."""
        return self.H
